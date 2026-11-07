"""Common detector contracts and the hybrid fusion pipeline.

The existing :class:`detector.LogoDetector` remains responsible for its
well-tested debounce, hysteresis, and break-length behavior.  This module
adapts that detector to a generic result contract and combines independent
detector results without treating detector-local raw scores as probabilities.

Later phases can add audio, fingerprint, and ML implementations by exposing
the ``Detector`` protocol below.  An unavailable implementation is represented
as ``UNKNOWN`` and is ignored by fusion.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class Classification(str, Enum):
    """Content classification shared by every detector."""

    PROGRAM = "program"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"


def _unit_interval(value: float, *, field_name: str) -> float:
    """Validate and normalize a confidence-like value to the unit interval."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class DetectorResult:
    """One detector's result for a timestamped sample.

    ``confidence`` is detector-local and is intentionally accompanied by
    ``confidence_kind``.  ``raw_score`` retains the detector's native value;
    it is never consumed directly by :class:`FusionEngine`.
    ``calibrated_probability`` is specifically P(COMMERCIAL), and must only be
    populated by detectors whose output has actually been calibrated.
    """

    detector_name: str
    classification: Classification
    confidence: float
    timestamp: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    confidence_kind: str = "detector_score"
    raw_score: float | None = None
    calibrated_probability: float | None = None
    available: bool = True

    def __post_init__(self) -> None:
        if not self.detector_name.strip():
            raise ValueError("detector_name must not be empty")
        classification = self.classification
        if not isinstance(classification, Classification):
            classification = Classification(str(classification).lower())
            object.__setattr__(self, "classification", classification)
        object.__setattr__(
            self,
            "confidence",
            _unit_interval(self.confidence, field_name="confidence"),
        )
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        if self.calibrated_probability is not None:
            object.__setattr__(
                self,
                "calibrated_probability",
                _unit_interval(
                    self.calibrated_probability,
                    field_name="calibrated_probability",
                ),
            )
        if not self.available:
            object.__setattr__(self, "classification", Classification.UNKNOWN)
            object.__setattr__(self, "confidence", 0.0)
            object.__setattr__(self, "calibrated_probability", None)

    @classmethod
    def unavailable(
        cls,
        detector_name: str,
        timestamp: float,
        reason: str,
        **diagnostics: Any,
    ) -> "DetectorResult":
        return cls(
            detector_name=detector_name,
            classification=Classification.UNKNOWN,
            confidence=0.0,
            timestamp=timestamp,
            diagnostics={"reason": reason, **diagnostics},
            confidence_kind="unavailable",
            available=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for dashboard/API telemetry."""
        return {
            "detector_name": self.detector_name,
            "classification": self.classification.value,
            "confidence": self.confidence,
            "confidence_kind": self.confidence_kind,
            "timestamp": self.timestamp,
            "available": self.available,
            "raw_score": self.raw_score,
            "calibrated_probability": self.calibrated_probability,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class DetectionInput:
    """Timestamp-aligned input shared by modular detectors."""

    timestamp: float
    frame: Any | None = None
    features: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Detector(Protocol):
    """Substitutable detector interface used by the hybrid pipeline."""

    name: str

    def evaluate(self, sample: DetectionInput) -> DetectorResult:
        """Classify one sample without raising for unavailable input."""


class LegacyDetectorAdapter:
    """Adapt the existing stateful visual detector to ``DetectorResult``."""

    name = "legacy_visual"

    def __init__(self, detector: Any):
        self._detector = detector

    def evaluate(self, sample: DetectionInput) -> DetectorResult:
        if sample.frame is None:
            return DetectorResult.unavailable(
                self.name, sample.timestamp, "missing_video_frame"
            )

        if sample.features.get("validate_video_signal", False):
            frame = sample.frame
            if getattr(frame, "size", 0) == 0:
                return DetectorResult.unavailable(
                    self.name, sample.timestamp, "empty_video_frame"
                )
            try:
                # Sample sparsely to keep this guard negligible on an older
                # CPU and avoid extra full-frame memory copies.
                signal_sample = frame[::16, ::16]
                mean_b, mean_g, mean_r = signal_sample.reshape(-1, 3).mean(axis=0)
                # Treat only essentially uniform black as unavailable. A dark
                # movie scene with visible variation remains valid input.
                if (
                    float(mean_b + mean_g + mean_r) / 3.0 < 2.0
                    and float(signal_sample.std()) < 2.0
                ):
                    return DetectorResult.unavailable(
                        self.name, sample.timestamp, "blank_video_signal"
                    )
                if mean_g > 180 and mean_b < 60 and mean_r < 60:
                    return DetectorResult.unavailable(
                        self.name, sample.timestamp, "capture_sync_artifact"
                    )
            except (AttributeError, TypeError, ValueError):
                return DetectorResult.unavailable(
                    self.name, sample.timestamp, "invalid_video_frame"
                )

        # Imported lazily so the result/fusion primitives remain usable in
        # tooling that does not install OpenCV (training/evaluation scripts).
        from detector import DetectionState

        state = self._detector.check_frame(sample.frame, now=sample.timestamp)
        edge, template = self._detector.last_confidences
        color = float(getattr(self._detector, "last_color_confidence", 0.0))
        profile_mode = getattr(self._detector, "_mode", "logo")
        if profile_mode == "color":
            raw_score = color
        else:
            raw_score = max(float(edge), float(template), color)

        classification = (
            Classification.PROGRAM
            if state == DetectionState.WATCHING
            else Classification.COMMERCIAL
        )
        result = DetectorResult(
            detector_name=self.name,
            classification=classification,
            confidence=_unit_interval(raw_score, field_name="legacy_visual score"),
            timestamp=sample.timestamp,
            diagnostics={
                "profile_mode": profile_mode,
                "edge_score": float(edge),
                "template_score": float(template),
                "color_score": color,
                "legacy_state": state.value,
                "temporally_smoothed": True,
            },
            confidence_kind="raw_profile_match",
            raw_score=raw_score,
            # A profile-match score is not P(COMMERCIAL).  Fusion therefore
            # uses its configured categorical reliability, not this value.
            calibrated_probability=None,
        )
        if isinstance(sample.features, dict):
            sample.features["visual"] = dict(result.diagnostics)
        return result


@dataclass(frozen=True)
class FusionPolicy:
    """Thresholds and source-specific priors for explainable fusion."""

    commercial_enter_threshold: float = 0.70
    program_exit_threshold: float = 0.40
    detector_weights: Mapping[str, float] = field(
        default_factory=lambda: {"legacy_visual": 1.0}
    )
    # These are policy priors, not transformed raw detector scores.
    uncalibrated_commercial_evidence: Mapping[str, float] = field(
        default_factory=lambda: {"legacy_visual": 0.85}
    )
    uncalibrated_program_evidence: Mapping[str, float] = field(
        default_factory=lambda: {"legacy_visual": 0.10}
    )
    logo_absence_commercial_evidence: float = 0.65
    minimum_commercial_sources: int = 1

    def __post_init__(self) -> None:
        enter = _unit_interval(
            self.commercial_enter_threshold,
            field_name="commercial_enter_threshold",
        )
        exit_ = _unit_interval(
            self.program_exit_threshold,
            field_name="program_exit_threshold",
        )
        if exit_ >= enter:
            raise ValueError(
                "program_exit_threshold must be lower than "
                "commercial_enter_threshold to preserve hysteresis"
            )
        object.__setattr__(self, "commercial_enter_threshold", enter)
        object.__setattr__(self, "program_exit_threshold", exit_)
        object.__setattr__(
            self,
            "logo_absence_commercial_evidence",
            _unit_interval(
                self.logo_absence_commercial_evidence,
                field_name="logo_absence_commercial_evidence",
            ),
        )
        if self.minimum_commercial_sources < 1:
            raise ValueError("minimum_commercial_sources must be at least 1")


class FusionEngine:
    """Fuse independent results while keeping raw scores incomparable."""

    name = "hybrid_fusion"

    def __init__(self, policy: FusionPolicy | None = None):
        self.policy = policy or FusionPolicy()

    def fuse(
        self,
        results: Sequence[DetectorResult],
        timestamp: float | None = None,
    ) -> DetectorResult:
        timestamp = time.time() if timestamp is None else timestamp
        contributions: list[dict[str, Any]] = []
        weighted_sum = 0.0
        total_weight = 0.0
        commercial_sources = 0
        decisive_evidence = False

        for result in results:
            if not result.available or result.classification == Classification.UNKNOWN:
                continue
            weight = max(
                0.0,
                float(self.policy.detector_weights.get(result.detector_name, 1.0)),
            )
            if weight == 0.0:
                continue

            if result.calibrated_probability is not None:
                evidence = result.calibrated_probability
                source = "calibrated_probability"
            elif result.classification == Classification.COMMERCIAL:
                evidence = float(
                    self.policy.uncalibrated_commercial_evidence.get(
                        result.detector_name, 0.60
                    )
                )
                source = "categorical_policy_prior"
                # Logo absence by itself remains below the default entry
                # threshold. Color/combo profiles provide additional positive
                # visual evidence and keep the normal categorical prior.
                if (
                    result.detector_name == "legacy_visual"
                    and result.diagnostics.get("profile_mode") == "logo"
                ):
                    evidence = self.policy.logo_absence_commercial_evidence
                    source = "logo_absence_policy_prior"
            else:
                evidence = float(
                    self.policy.uncalibrated_program_evidence.get(
                        result.detector_name, 0.25
                    )
                )
                source = "categorical_policy_prior"

            evidence = _unit_interval(evidence, field_name="fusion evidence")
            if evidence > 0.5:
                commercial_sources += 1
            decisive = bool(result.diagnostics.get("decisive_evidence", False))
            if (
                result.detector_name == "legacy_visual"
                and result.classification == Classification.COMMERCIAL
                and result.diagnostics.get("profile_mode") in ("color", "combo")
            ):
                decisive = True
            decisive_evidence = decisive_evidence or decisive
            weighted_sum += evidence * weight
            total_weight += weight
            contributions.append(
                {
                    "detector_name": result.detector_name,
                    "classification": result.classification.value,
                    "evidence": evidence,
                    "source": source,
                    "weight": weight,
                    "decisive": decisive,
                }
            )

        if total_weight == 0.0:
            return DetectorResult.unavailable(
                self.name,
                timestamp,
                "no_available_detector_evidence",
                inputs=[result.to_dict() for result in results],
            )

        commercial_evidence = weighted_sum / total_weight
        enough_sources = (
            commercial_sources >= self.policy.minimum_commercial_sources
            or decisive_evidence
        )
        if commercial_evidence >= self.policy.commercial_enter_threshold and enough_sources:
            classification = Classification.COMMERCIAL
        elif commercial_evidence <= self.policy.program_exit_threshold:
            classification = Classification.PROGRAM
        else:
            classification = Classification.UNKNOWN

        # Margin is a useful certainty diagnostic, but deliberately is not
        # called a probability and is not reused as detector evidence.
        margin = min(1.0, abs(commercial_evidence - 0.5) * 2.0)
        return DetectorResult(
            detector_name=self.name,
            classification=classification,
            confidence=margin,
            timestamp=timestamp,
            diagnostics={
                "commercial_evidence": commercial_evidence,
                "commercial_enter_threshold": self.policy.commercial_enter_threshold,
                "program_exit_threshold": self.policy.program_exit_threshold,
                "contributions": contributions,
                "input_count": len(results),
                "commercial_sources": commercial_sources,
                "minimum_commercial_sources": self.policy.minimum_commercial_sources,
                "decisive_evidence": decisive_evidence,
            },
            confidence_kind="fusion_margin",
            raw_score=commercial_evidence,
            calibrated_probability=None,
        )


@dataclass(frozen=True)
class PipelineDecision:
    detector_results: tuple[DetectorResult, ...]
    fused_result: DetectorResult


class DetectionPipeline:
    """Run enabled detectors independently and safely fuse their outputs."""

    def __init__(
        self,
        detectors: Sequence[Detector],
        fusion: FusionEngine,
        enabled: Mapping[str, bool] | None = None,
        contain_errors: bool = True,
    ):
        self._detectors = tuple(detectors)
        self._fusion = fusion
        self._enabled = dict(enabled or {})
        self._contain_errors = contain_errors

    def evaluate(self, sample: DetectionInput) -> PipelineDecision:
        results: list[DetectorResult] = []
        for detector in self._detectors:
            if not self._enabled.get(detector.name, True):
                results.append(
                    DetectorResult.unavailable(
                        detector.name, sample.timestamp, "detector_disabled"
                    )
                )
                continue
            try:
                results.append(detector.evaluate(sample))
            except Exception as exc:  # optional detectors must fail closed
                if not self._contain_errors:
                    raise
                results.append(
                    DetectorResult.unavailable(
                        detector.name,
                        sample.timestamp,
                        "detector_error",
                        error=f"{type(exc).__name__}: {exc}"[:240],
                    )
                )
        fused = self._fusion.fuse(results, timestamp=sample.timestamp)
        return PipelineDecision(tuple(results), fused)
