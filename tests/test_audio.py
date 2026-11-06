import math

import numpy as np

from audio import (
    AudioFeatureExtractor,
    AudioFeatures,
    AudioTransitionDetector,
    parse_avfoundation_audio_devices,
)
from detection_pipeline import (
    Classification,
    DetectionInput,
    DetectorResult,
    FusionEngine,
    FusionPolicy,
)


def test_parse_avfoundation_audio_devices_uses_audio_section_only():
    output = """
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x1] [1] USB Digital Audio
"""

    devices = parse_avfoundation_audio_devices(output)

    assert [(d.index, d.name) for d in devices] == [
        (0, "MacBook Pro Microphone"),
        (1, "USB Digital Audio"),
    ]


def test_audio_features_detect_silence_without_nan():
    extractor = AudioFeatureExtractor(sample_rate=16000)

    features = extractor.extract(np.zeros(32000, dtype=np.float32), timestamp=10.0)

    assert features.silence_ratio == 1.0
    assert features.rms == 0.0
    assert math.isfinite(features.spectral_centroid_hz)
    assert len(features.mfcc) == 13
    assert all(math.isfinite(v) for v in features.mfcc)


def test_audio_features_report_tone_centroid_and_loudness_change():
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    extractor = AudioFeatureExtractor(sample_rate=sample_rate)
    quiet = 0.05 * np.sin(2 * np.pi * 1000 * t)
    loud = 0.5 * np.sin(2 * np.pi * 1000 * t)
    extractor.extract(quiet, timestamp=1.0)

    features = extractor.extract(loud, timestamp=2.0)

    assert 900 < features.spectral_centroid_hz < 1100
    assert features.loudness_change_db > 15
    assert features.silence_ratio == 0.0


def make_audio(**overrides):
    values = {
        "timestamp": 10.0,
        "sample_rate": 16000,
        "window_seconds": 2.0,
        "rms": 0.2,
        "rms_dbfs": -14.0,
        "dynamic_range_db": 12.0,
        "silence_ratio": 0.8,
        "spectral_centroid_hz": 2500.0,
        "spectral_flux": 0.02,
        "loudness_change_db": 10.0,
        "mfcc": tuple([0.0] * 13),
        "capture_latency_ms": 5.0,
    }
    values.update(overrides)
    return AudioFeatures(**values)


def test_missing_and_stale_audio_are_unavailable_unknown():
    detector = AudioTransitionDetector(max_age_seconds=2.0)

    missing = detector.evaluate(DetectionInput(timestamp=10.0))
    stale = detector.evaluate(
        DetectionInput(timestamp=20.0, features={"audio": make_audio()})
    )

    assert missing.classification == Classification.UNKNOWN
    assert not missing.available
    assert stale.classification == Classification.UNKNOWN
    assert stale.diagnostics["reason"] == "stale_audio"


def test_strong_audio_transition_is_supporting_commercial_evidence():
    detector = AudioTransitionDetector()

    actual = detector.evaluate(
        DetectionInput(timestamp=10.5, features={"audio": make_audio()})
    )

    assert actual.classification == Classification.COMMERCIAL
    assert actual.diagnostics["supporting_evidence_only"] is True


def test_movie_silence_alone_is_not_commercial_evidence():
    detector = AudioTransitionDetector()
    silence = make_audio(
        silence_ratio=1.0,
        spectral_flux=0.0,
        loudness_change_db=0.0,
        rms=0.0,
        rms_dbfs=-80.0,
    )

    actual = detector.evaluate(
        DetectionInput(timestamp=10.5, features={"audio": silence})
    )

    assert actual.classification == Classification.UNKNOWN


def test_audio_transition_requires_corroboration_in_hybrid_policy():
    policy = FusionPolicy(
        minimum_commercial_sources=2,
        detector_weights={"audio_transition": 1.0, "legacy_visual": 1.0},
        uncalibrated_commercial_evidence={
            "audio_transition": 0.85,
            "legacy_visual": 0.85,
        },
    )
    audio = DetectorResult(
        detector_name="audio_transition",
        classification=Classification.COMMERCIAL,
        confidence=0.9,
        timestamp=10.0,
    )
    logo_absent = DetectorResult(
        detector_name="legacy_visual",
        classification=Classification.COMMERCIAL,
        confidence=0.1,
        timestamp=10.0,
        diagnostics={"profile_mode": "logo"},
    )

    audio_only = FusionEngine(policy).fuse([audio], timestamp=10.0)
    corroborated = FusionEngine(policy).fuse([audio, logo_absent], timestamp=10.0)

    assert audio_only.classification == Classification.UNKNOWN
    assert corroborated.classification == Classification.COMMERCIAL
