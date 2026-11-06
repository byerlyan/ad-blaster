import numpy as np

from detection_pipeline import (
    Classification,
    DetectionPipeline,
    DetectorResult,
    FusionEngine,
    LegacyDetectorAdapter,
)
from detector import DetectionState
from engine import Engine
from ml import ModelStatus


class StaticResultDetector:
    name = "legacy_visual"

    def __init__(self, classification, profile_mode="logo"):
        self.classification = classification
        self.profile_mode = profile_mode

    def evaluate(self, sample):
        return DetectorResult(
            detector_name=self.name,
            classification=self.classification,
            confidence=0.99,
            confidence_kind="raw_profile_match",
            raw_score=0.99,
            calibrated_probability=None,
            timestamp=sample.timestamp,
            diagnostics={"profile_mode": self.profile_mode},
        )


def pipeline_for(classification, profile_mode="logo"):
    return DetectionPipeline(
        [StaticResultDetector(classification, profile_mode)],
        FusionEngine(),
    )


def test_legacy_mode_passes_existing_detector_classification_through():
    engine = Engine()
    engine._detection_mode = "legacy"
    engine._pipeline = pipeline_for(Classification.COMMERCIAL)

    state = engine._classify_frame(object(), now=123.0)

    assert state == DetectionState.COMMERCIAL
    assert engine._latest_fused_result is None


def test_hybrid_unknown_holds_previous_state_and_cannot_start_commercial():
    engine = Engine()
    engine._detection_mode = "hybrid"
    engine._detection_state = DetectionState.WATCHING
    # Logo absence is deliberately below the hybrid entry threshold.
    engine._pipeline = pipeline_for(Classification.COMMERCIAL, profile_mode="logo")

    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    state = engine._classify_frame(frame, now=123.0)

    assert state == DetectionState.WATCHING
    assert engine._latest_fused_result["classification"] == "unknown"


def test_hybrid_missing_capture_is_unknown_and_holds_previous_state():
    engine = Engine()
    engine._detection_mode = "hybrid"
    engine._detection_state = DetectionState.WATCHING
    engine._pipeline = DetectionPipeline(
        [LegacyDetectorAdapter(detector=object())],
        FusionEngine(),
    )

    state = engine._classify_frame(None, now=123.0)

    assert state == DetectionState.WATCHING
    assert engine._latest_detector_results[0]["available"] is False
    assert engine._latest_fused_result["classification"] == "unknown"


def test_detection_mode_validation_does_not_mutate_engine_state():
    engine = Engine()

    try:
        engine._normalize_detection_mode("experimental")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid detection mode should fail")

    assert engine.state.value == "stopped"


def test_shadow_status_exposes_legacy_hybrid_comparison_inputs():
    engine = Engine()
    engine._shadow = True
    engine._detection_mode = "hybrid"
    engine._pipeline = pipeline_for(Classification.COMMERCIAL, profile_mode="logo")
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)

    engine._classify_frame(frame, now=123.0)
    status = engine.get_status()

    assert status.shadow_mode is True
    assert status.detector_results[0]["classification"] == "commercial"
    assert status.fused_result["classification"] == "unknown"


def test_hybrid_status_serializes_ml_detector_status():
    class StubMLDetector:
        @staticmethod
        def status():
            return ModelStatus(
                loaded=False,
                path="commercial-model.joblib",
                error="model file not found",
            )

    engine = Engine()
    engine._detection_mode = "hybrid"
    engine._ml_detector = StubMLDetector()

    status = engine.get_status()

    assert status.ml == {
        "loaded": False,
        "path": "commercial-model.joblib",
        "version": None,
        "error": "model file not found",
        "metadata": None,
    }

