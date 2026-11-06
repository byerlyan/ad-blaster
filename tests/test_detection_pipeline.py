import pytest
import numpy as np

from detection_pipeline import (
    Classification,
    DetectionInput,
    DetectionPipeline,
    DetectorResult,
    FusionEngine,
    FusionPolicy,
    LegacyDetectorAdapter,
)


def result(
    classification=Classification.PROGRAM,
    *,
    name="test",
    confidence=0.5,
    raw_score=None,
    calibrated_probability=None,
    diagnostics=None,
    available=True,
):
    return DetectorResult(
        detector_name=name,
        classification=classification,
        confidence=confidence,
        confidence_kind="test_score",
        raw_score=raw_score,
        calibrated_probability=calibrated_probability,
        timestamp=123.0,
        diagnostics=diagnostics or {},
        available=available,
    )


def test_result_normalizes_classification_and_confidence():
    actual = result("PROGRAM", confidence=1.7, calibrated_probability=-0.5)

    assert actual.classification == Classification.PROGRAM
    assert actual.confidence == 1.0
    assert actual.calibrated_probability == 0.0


def test_unavailable_result_is_always_unknown_and_zero_confidence():
    actual = result(
        Classification.COMMERCIAL,
        confidence=0.99,
        calibrated_probability=0.99,
        available=False,
    )

    assert actual.classification == Classification.UNKNOWN
    assert actual.confidence == 0.0
    assert actual.calibrated_probability is None


def test_missing_detector_evidence_fuses_to_unknown():
    fused = FusionEngine().fuse(
        [DetectorResult.unavailable("audio", 123.0, "missing_audio")],
        timestamp=123.0,
    )

    assert fused.classification == Classification.UNKNOWN
    assert not fused.available
    assert fused.diagnostics["reason"] == "no_available_detector_evidence"


def test_legacy_adapter_reports_missing_video_as_unknown():
    adapter = LegacyDetectorAdapter(detector=object())

    actual = adapter.evaluate(DetectionInput(timestamp=123.0, frame=None))

    assert actual.classification == Classification.UNKNOWN
    assert not actual.available
    assert actual.diagnostics["reason"] == "missing_video_frame"


def test_hybrid_video_validation_rejects_blank_signal():
    adapter = LegacyDetectorAdapter(detector=object())

    actual = adapter.evaluate(
        DetectionInput(
            timestamp=123.0,
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            features={"validate_video_signal": True},
        )
    )

    assert actual.classification == Classification.UNKNOWN
    assert actual.diagnostics["reason"] == "blank_video_signal"


def test_logo_absence_alone_does_not_enter_commercial():
    visual = result(
        Classification.COMMERCIAL,
        name="legacy_visual",
        confidence=0.99,
        raw_score=0.01,
        diagnostics={"profile_mode": "logo"},
    )

    fused = FusionEngine().fuse([visual], timestamp=123.0)

    assert fused.classification == Classification.UNKNOWN
    assert fused.diagnostics["commercial_evidence"] == pytest.approx(0.65)
    assert fused.diagnostics["contributions"][0]["source"] == "logo_absence_policy_prior"


def test_raw_scores_are_not_compared_or_treated_as_probabilities():
    low_raw = result(
        Classification.COMMERCIAL,
        name="legacy_visual",
        confidence=0.01,
        raw_score=-9.0,
        diagnostics={"profile_mode": "combo"},
    )
    high_raw = result(
        Classification.COMMERCIAL,
        name="legacy_visual",
        confidence=0.99,
        raw_score=99.0,
        diagnostics={"profile_mode": "combo"},
    )

    low_fused = FusionEngine().fuse([low_raw], timestamp=123.0)
    high_fused = FusionEngine().fuse([high_raw], timestamp=123.0)

    assert low_fused.classification == Classification.COMMERCIAL
    assert high_fused.classification == Classification.COMMERCIAL
    assert low_fused.raw_score == high_fused.raw_score


def test_calibrated_probability_is_eligible_for_fusion():
    ml = result(
        Classification.COMMERCIAL,
        name="ml",
        confidence=0.97,
        raw_score=4.2,
        calibrated_probability=0.97,
    )

    fused = FusionEngine().fuse([ml], timestamp=123.0)

    assert fused.classification == Classification.COMMERCIAL
    assert fused.diagnostics["commercial_evidence"] == pytest.approx(0.97)
    assert fused.diagnostics["contributions"][0]["source"] == "calibrated_probability"


def test_sports_scene_change_ml_signal_cannot_override_visual_program_alone():
    policy = FusionPolicy(
        minimum_commercial_sources=2,
        detector_weights={"legacy_visual": 1.0, "ml_classifier": 1.5},
    )
    visual = result(
        Classification.PROGRAM,
        name="legacy_visual",
        diagnostics={"profile_mode": "logo"},
    )
    ml = result(
        Classification.COMMERCIAL,
        name="ml_classifier",
        calibrated_probability=0.8,
    )

    fused = FusionEngine(policy).fuse([visual, ml], timestamp=123.0)

    assert fused.classification != Classification.COMMERCIAL


def test_program_evidence_exits_commercial_side_of_hysteresis():
    visual = result(
        Classification.PROGRAM,
        name="legacy_visual",
        diagnostics={"profile_mode": "logo"},
    )

    fused = FusionEngine().fuse([visual], timestamp=123.0)

    assert fused.classification == Classification.PROGRAM


class BrokenDetector:
    name = "audio"

    def evaluate(self, sample):
        raise RuntimeError("device disappeared")


class StaticDetector:
    name = "visual"

    def evaluate(self, sample):
        return DetectorResult(
            detector_name=self.name,
            classification=Classification.PROGRAM,
            confidence=0.8,
            timestamp=sample.timestamp,
        )


def test_pipeline_contains_optional_detector_failures():
    pipeline = DetectionPipeline(
        [BrokenDetector(), StaticDetector()],
        FusionEngine(),
    )

    decision = pipeline.evaluate(DetectionInput(timestamp=123.0, frame=object()))

    assert decision.detector_results[0].classification == Classification.UNKNOWN
    assert not decision.detector_results[0].available
    assert decision.detector_results[0].diagnostics["reason"] == "detector_error"
    assert decision.detector_results[1].classification == Classification.PROGRAM
    assert decision.fused_result.classification == Classification.PROGRAM


def test_pipeline_can_preserve_legacy_error_behavior():
    pipeline = DetectionPipeline(
        [BrokenDetector()],
        FusionEngine(),
        contain_errors=False,
    )

    with pytest.raises(RuntimeError, match="device disappeared"):
        pipeline.evaluate(DetectionInput(timestamp=123.0, frame=object()))


def test_disabled_detector_is_unknown_not_commercial():
    pipeline = DetectionPipeline(
        [StaticDetector()],
        FusionEngine(),
        enabled={"visual": False},
    )

    decision = pipeline.evaluate(DetectionInput(timestamp=123.0, frame=object()))

    assert not decision.detector_results[0].available
    assert decision.detector_results[0].classification == Classification.UNKNOWN
    assert decision.fused_result.classification == Classification.UNKNOWN


def test_fusion_policy_requires_hysteresis_gap():
    with pytest.raises(ValueError, match="hysteresis"):
        FusionPolicy(
            commercial_enter_threshold=0.6,
            program_exit_threshold=0.6,
        )
