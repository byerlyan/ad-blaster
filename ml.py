"""Lightweight feature collection and optional commercial ML inference."""

from __future__ import annotations

import logging
import os
import platform
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from detection_pipeline import Classification, DetectionInput, DetectorResult
from ml_artifacts import FEATURE_NAMES, FeatureDatasetStore, PlattCalibratedModel

logger = logging.getLogger(__name__)

class RuntimeFeatureBuilder:
    """Build a bounded temporal feature vector from current audio/video data."""

    def __init__(self, history_size: int = 20):
        self._previous_gray: np.ndarray | None = None
        self._history: deque[dict[str, float]] = deque(maxlen=history_size)

    def build(
        self,
        frame: np.ndarray | None,
        visual: Mapping[str, Any] | None,
        audio: Any | None,
    ) -> dict[str, float]:
        visual = dict(visual or {})
        if audio is None:
            audio_values: dict[str, Any] = {}
        elif hasattr(audio, "to_dict"):
            audio_values = audio.to_dict()
        else:
            audio_values = dict(audio)

        scene_cut = 0.0
        if frame is not None and getattr(frame, "size", 0):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
            if self._previous_gray is not None:
                delta = float(np.mean(cv2.absdiff(gray, self._previous_gray))) / 255.0
                scene_cut = float(delta >= 0.18)
            self._previous_gray = gray

        mfcc = list(audio_values.get("mfcc", []))[:13]
        mfcc.extend([0.0] * (13 - len(mfcc)))
        visual_max = max(
            float(visual.get("edge_score", 0.0)),
            float(visual.get("template_score", 0.0)),
            float(visual.get("color_score", 0.0)),
        )
        current = {
            "scene_cut": scene_cut,
            "visual": visual_max,
            "audio": float(audio_values.get("rms_dbfs", -80.0)),
        }
        self._history.append(current)
        recent_scene = float(np.mean([item["scene_cut"] for item in self._history]))
        recent_visual = float(np.mean([item["visual"] for item in self._history]))
        recent_audio = float(np.mean([item["audio"] for item in self._history]))
        vector = {
            "visual_edge": float(visual.get("edge_score", 0.0)),
            "visual_template": float(visual.get("template_score", 0.0)),
            "visual_color": float(visual.get("color_score", 0.0)),
            "audio_rms_dbfs": float(audio_values.get("rms_dbfs", -80.0)),
            "audio_dynamic_range_db": float(audio_values.get("dynamic_range_db", 0.0)),
            "audio_silence_ratio": float(audio_values.get("silence_ratio", 0.0)),
            "audio_spectral_centroid_hz": float(audio_values.get("spectral_centroid_hz", 0.0)),
            "audio_spectral_flux": float(audio_values.get("spectral_flux", 0.0)),
            "audio_loudness_change_db": float(audio_values.get("loudness_change_db", 0.0)),
            "scene_cut": scene_cut,
            "recent_scene_cut_rate": recent_scene,
            "recent_visual_mean": recent_visual,
            "recent_audio_energy_mean": recent_audio,
        }
        vector.update({f"mfcc_{i}": float(mfcc[i]) for i in range(13)})
        return vector


class FeatureDatasetWriter:
    """Bounded background writer so labeling never blocks video capture."""

    def __init__(self, store: FeatureDatasetStore, max_queue: int = 1024):
        self.store = store
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="feature-writer")
        self.dropped = 0
        self._thread.start()

    def submit(self, **row: Any) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                row = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.store.append(**row)
            except Exception as exc:
                logger.warning("Feature dataset write failed: %s", exc)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        self._stop.set()
        self.flush()
        self._thread.join(timeout=2)


@dataclass(frozen=True)
class ModelStatus:
    loaded: bool
    path: str
    version: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] | None = None
    resolved_path: str = ""
    file_exists: bool = False
    file_size_bytes: int | None = None
    modified_at: float | None = None
    load_stage: str = "not_checked"
    schema_valid: bool | None = None
    artifact_feature_count: int | None = None
    expected_feature_count: int = len(FEATURE_NAMES)
    model_class: str | None = None
    loaded_at: float | None = None
    last_inference_at: float | None = None
    last_probability: float | None = None
    last_inference_ms: float | None = None
    dependencies: Mapping[str, str | None] | None = None


class MLCommercialDetector:
    """Optional serialized scikit-learn inference with graceful fallback."""

    name = "ml_classifier"

    def __init__(
        self,
        model_path: str,
        *,
        commercial_threshold: float = 0.60,
        program_threshold: float = 0.40,
        inference_interval: float = 0.5,
    ):
        self.model_path = model_path
        self.commercial_threshold = commercial_threshold
        self.program_threshold = program_threshold
        self.inference_interval = inference_interval
        self._builder = RuntimeFeatureBuilder()
        self._model = None
        self._feature_names = tuple(FEATURE_NAMES)
        self._metadata: dict[str, Any] = {}
        self._error: str | None = None
        self._load_stage = "not_checked"
        self._schema_valid: bool | None = None
        self._artifact_feature_count: int | None = None
        self._model_class: str | None = None
        self._loaded_at: float | None = None
        self._last_evaluation_timestamp = float("-inf")
        self._last_probability: float | None = None
        self._last_inference_at: float | None = None
        self._last_inference_ms: float | None = None
        self._last_vector: dict[str, float] | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            self._error = "model file not found"
            self._load_stage = "file_not_found"
            return
        try:
            import joblib
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._load_stage = "dependency_error"
            return

        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._load_stage = "artifact_read_error"
            return

        try:
            if not isinstance(artifact, Mapping):
                raise ValueError("model artifact must be a mapping")
            if "model" not in artifact or "feature_names" not in artifact:
                raise ValueError("model artifact is missing model or feature_names")
            candidate_model = artifact["model"]
            if not callable(getattr(candidate_model, "predict_proba", None)):
                raise ValueError("model artifact does not provide predict_proba")
            self._feature_names = tuple(str(name) for name in artifact["feature_names"])
            self._artifact_feature_count = len(self._feature_names)
            self._metadata = dict(artifact.get("metadata", {}))
            if tuple(self._feature_names) != tuple(FEATURE_NAMES):
                raise ValueError("model feature schema does not match this application version")
            self._schema_valid = True
            self._model = candidate_model
            model_type = type(candidate_model)
            self._model_class = f"{model_type.__module__}.{model_type.__qualname__}"
            self._loaded_at = time.time()
            self._load_stage = "loaded"
            self._error = None
        except Exception as exc:
            self._model = None
            self._schema_valid = False
            self._load_stage = "artifact_validation_error"
            self._error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _dependency_versions() -> dict[str, str | None]:
        versions: dict[str, str | None] = {"python": platform.python_version()}
        for distribution in ("numpy", "joblib", "scikit-learn"):
            try:
                versions[distribution] = package_version(distribution)
            except PackageNotFoundError:
                versions[distribution] = None
        return versions

    def status(self) -> ModelStatus:
        resolved = str(Path(self.model_path).expanduser().resolve(strict=False)) if self.model_path else ""
        file_size = None
        modified_at = None
        exists = False
        if resolved:
            try:
                stat = Path(resolved).stat()
                exists = True
                file_size = stat.st_size
                modified_at = stat.st_mtime
            except OSError:
                pass
        return ModelStatus(
            loaded=self._model is not None,
            path=self.model_path,
            version=self._metadata.get("model_version"),
            error=self._error,
            metadata=self._metadata,
            resolved_path=resolved,
            file_exists=exists,
            file_size_bytes=file_size,
            modified_at=modified_at,
            load_stage=self._load_stage,
            schema_valid=self._schema_valid,
            artifact_feature_count=self._artifact_feature_count,
            model_class=self._model_class,
            loaded_at=self._loaded_at,
            last_inference_at=self._last_inference_at,
            last_probability=self._last_probability,
            last_inference_ms=self._last_inference_ms,
            dependencies=self._dependency_versions(),
        )

    def evaluate(self, sample: DetectionInput) -> DetectorResult:
        should_collect = bool(sample.features.get("collect_ml_features", False))
        inference_enabled = bool(sample.features.get("ml_inference_enabled", False))
        if not inference_enabled and not should_collect:
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "ml_detector_disabled",
                model_error=self._error,
            )
        if self._model is None and not should_collect:
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "ml_model_unavailable",
                model_error=self._error,
            )
        if sample.timestamp - self._last_evaluation_timestamp < self.inference_interval:
            if not inference_enabled or self._last_probability is None:
                return DetectorResult.unavailable(
                    self.name,
                    sample.timestamp,
                    "ml_sampling_throttled",
                    cached=True,
                )
            probability = self._last_probability
            if probability >= self.commercial_threshold:
                classification = Classification.COMMERCIAL
            elif probability <= self.program_threshold:
                classification = Classification.PROGRAM
            else:
                classification = Classification.UNKNOWN
            confidence = probability if classification == Classification.COMMERCIAL else 1.0 - probability
            return DetectorResult(
                detector_name=self.name,
                classification=classification,
                confidence=confidence,
                timestamp=sample.timestamp,
                diagnostics={"cached": True, "model_version": self._metadata.get("model_version")},
                confidence_kind="calibrated_probability",
                raw_score=probability,
                calibrated_probability=probability,
            )
        vector = self._builder.build(
            sample.frame,
            sample.features.get("visual"),
            sample.features.get("audio"),
        )
        self._last_evaluation_timestamp = sample.timestamp
        self._last_vector = vector
        if self._model is None or not inference_enabled:
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "ml_model_unavailable",
                model_error=self._error,
                feature_vector=vector,
            )
        started = time.perf_counter()
        ordered = np.array([[vector[name] for name in self._feature_names]], dtype=np.float32)
        probability = float(self._model.predict_proba(ordered)[0][1])
        self._last_probability = probability
        inference_ms = (time.perf_counter() - started) * 1000.0
        self._last_inference_at = sample.timestamp
        self._last_inference_ms = inference_ms
        if probability >= self.commercial_threshold:
            classification = Classification.COMMERCIAL
        elif probability <= self.program_threshold:
            classification = Classification.PROGRAM
        else:
            classification = Classification.UNKNOWN
        confidence = probability if classification == Classification.COMMERCIAL else 1.0 - probability
        return DetectorResult(
            detector_name=self.name,
            classification=classification,
            confidence=confidence,
            timestamp=sample.timestamp,
            diagnostics={
                "feature_vector": vector,
                "inference_ms": inference_ms,
                "model_version": self._metadata.get("model_version"),
                "supporting_evidence_only": True,
            },
            confidence_kind="calibrated_probability",
            raw_score=probability,
            calibrated_probability=probability,
        )
