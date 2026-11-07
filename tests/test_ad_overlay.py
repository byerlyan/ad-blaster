import subprocess
from types import SimpleNamespace

import numpy as np

from ad_overlay import AdOverlayDetector, OCRObservation, TesseractOCR
from detection_pipeline import (
    Classification,
    DetectionInput,
    DetectorResult,
    FusionEngine,
    FusionPolicy,
)


FRAME = np.full((90, 160, 3), 32, dtype=np.uint8)


def sample(timestamp: float) -> DetectionInput:
    return DetectionInput(timestamp=timestamp, frame=FRAME)


class ScriptedOCR:
    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, _image):
        index = min(self.calls, len(self.texts) - 1)
        self.calls += 1
        return OCRObservation(self.texts[index], 0.91)


def detector(ocr, **kwargs):
    return AdOverlayDetector(
        ocr_runner=ocr,
        regions=((0.0, 0.0, 1.0, 1.0),),
        scan_interval=1.0,
        required_confirmations=2,
        hold_seconds=2.0,
        **kwargs,
    )


def test_missing_tesseract_is_unavailable_not_commercial():
    actual = AdOverlayDetector(tesseract_binary="definitely-not-tesseract").evaluate(
        sample(1.0)
    )

    assert not actual.available
    assert actual.classification == Classification.UNKNOWN
    assert actual.diagnostics["reason"] == "tesseract_unavailable"


def test_tesseract_uses_explicit_discovered_language_directory(tmp_path, monkeypatch):
    tessdata = tmp_path / "share" / "tessdata"
    tessdata.mkdir(parents=True)
    (tessdata / "eng.traineddata").write_bytes(b"test fixture")
    captured = {}

    monkeypatch.setattr("ad_overlay.shutil.which", lambda _binary: "/usr/bin/tesseract")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("ad_overlay.subprocess.run", fake_run)
    runner = TesseractOCR(
        "tesseract",
        tessdata_directory=str(tmp_path),
    )

    runner(np.full((20, 40, 3), 255, dtype=np.uint8))

    assert runner.tessdata_directory == str(tessdata)
    index = captured["command"].index("--tessdata-dir")
    assert captured["command"][index + 1] == str(tessdata)
    assert captured["environment"]["TESSDATA_PREFIX"] == str(tessdata)


def test_tesseract_timeout_is_reported_without_crashing_pipeline():
    def timed_out(_image):
        raise subprocess.TimeoutExpired(["tesseract"], 2.0)

    actual = detector(timed_out).evaluate(sample(1.0))

    assert not actual.available
    assert actual.classification == Classification.UNKNOWN
    assert actual.diagnostics["reason"] == "ocr_timeout"
    assert actual.diagnostics["error"] == "Tesseract exceeded its 2s timeout"
    assert actual.diagnostics["retryable"] is True


def test_ocr_failure_exposes_a_bounded_diagnostic():
    def failed(_image):
        raise RuntimeError("language data could not be loaded")

    actual = detector(failed).evaluate(sample(1.0))

    assert not actual.available
    assert actual.diagnostics["reason"] == "ocr_error"
    assert actual.diagnostics["error"] == "RuntimeError: language data could not be loaded"


def test_countdown_without_ad_label_is_only_a_diagnostic_flag():
    actual = detector(ScriptedOCR("2nd quarter 0:15")).evaluate(sample(1.0))

    assert actual.available
    assert actual.classification == Classification.UNKNOWN
    assert actual.diagnostics["countdown_seconds"] == 15
    assert actual.diagnostics["reason"] == "countdown_without_ad_label"


def test_sponsored_programming_is_not_treated_as_an_ad():
    actual = detector(ScriptedOCR("Tonight's game sponsored by Example")).evaluate(
        sample(1.0)
    )

    assert actual.classification == Classification.UNKNOWN
    assert actual.diagnostics["sponsored_candidate"] is True


def test_low_confidence_ad_text_is_not_supporting_evidence():
    actual = detector(lambda _image: ("Ad 1 of 2 0:15", 0.12)).evaluate(sample(1.0))

    assert actual.classification == Classification.UNKNOWN
    assert actual.diagnostics["reason"] == "low_ocr_confidence"
    assert actual.diagnostics["temporal_matches"] == 0


def test_explicit_ad_countdown_requires_repeated_ocr_observations():
    ocr = ScriptedOCR("Ad 1 of 2 0:15", "Ad 1 of 2 0:14")
    actual_detector = detector(ocr)

    first = actual_detector.evaluate(sample(1.0))
    cached = actual_detector.evaluate(sample(1.5))
    confirmed = actual_detector.evaluate(sample(2.1))

    assert first.classification == Classification.UNKNOWN
    assert first.diagnostics["reason"] == "awaiting_confirmation"
    assert cached.classification == Classification.UNKNOWN
    assert cached.diagnostics["cached"] is True
    assert ocr.calls == 2
    assert confirmed.classification == Classification.COMMERCIAL
    assert confirmed.diagnostics["countdown_seconds"] == 14
    assert confirmed.diagnostics["countdown_confirmed"] is True
    assert confirmed.diagnostics["decisive_evidence"] is False


def test_confirmed_overlay_survives_one_short_ocr_dropout():
    ocr = ScriptedOCR("Advertisement 0:10", "Advertisement 0:09", "")
    actual_detector = detector(ocr)

    actual_detector.evaluate(sample(1.0))
    confirmed = actual_detector.evaluate(sample(2.1))
    held = actual_detector.evaluate(sample(3.2))

    assert confirmed.classification == Classification.COMMERCIAL
    assert held.classification == Classification.COMMERCIAL
    assert held.diagnostics["held"] is True


def test_overlay_must_be_corroborated_before_fusion_enters_commercial():
    policy = FusionPolicy(
        minimum_commercial_sources=2,
        detector_weights={"legacy_visual": 1.0, "ad_overlay": 2.0},
        uncalibrated_commercial_evidence={
            "legacy_visual": 0.85,
            "ad_overlay": 0.95,
        },
        logo_absence_commercial_evidence=0.65,
    )
    overlay = DetectorResult(
        detector_name="ad_overlay",
        classification=Classification.COMMERCIAL,
        confidence=0.98,
        timestamp=10.0,
        diagnostics={"explicit_ad_label": True, "decisive_evidence": False},
    )
    logo_absent = DetectorResult(
        detector_name="legacy_visual",
        classification=Classification.COMMERCIAL,
        confidence=0.1,
        timestamp=10.0,
        diagnostics={"profile_mode": "logo"},
    )

    overlay_only = FusionEngine(policy).fuse([overlay], timestamp=10.0)
    corroborated = FusionEngine(policy).fuse(
        [logo_absent, overlay], timestamp=10.0
    )

    assert overlay_only.classification == Classification.UNKNOWN
    assert corroborated.classification == Classification.COMMERCIAL
    assert corroborated.diagnostics["commercial_sources"] == 2
