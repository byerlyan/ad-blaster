import numpy as np

from detection_pipeline import Classification, DetectionInput
from fingerprint import (
    AudioFingerprintDetector,
    FingerprintDatabase,
    FingerprintMatch,
)


def synthetic_ad(sample_rate=8000, seconds=16.0, base=220.0):
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    # Time-varying tones produce distinct local tokens without fixture files.
    phase = 2 * np.pi * (base * t + 18.0 * t * t)
    signal = 0.4 * np.sin(phase)
    signal += 0.2 * np.sin(2 * np.pi * (base * 2.3 * t + 7.0 * t * t))
    signal *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.7 * t)
    return signal.astype(np.float32), sample_rate


def make_db(tmp_path):
    return FingerprintDatabase(
        str(tmp_path / "fingerprints.sqlite3"),
        match_threshold=0.45,
        minimum_duration=2.0,
        minimum_unique_tokens=4,
    )


def test_enroll_match_position_duplicate_delete_and_persistence(tmp_path):
    db = make_db(tmp_path)
    samples, sample_rate = synthetic_ad()

    enrolled = db.enroll(
        samples,
        sample_rate,
        name="Local Auto Spot",
        advertisement_id="ad-1",
        metadata={"campaign": "test"},
    )
    query_start = 5.0
    query = samples[int(query_start * sample_rate) : int((query_start + 6.0) * sample_rate)]
    match = db.match(query, sample_rate)
    duplicate = db.enroll(samples, sample_rate, name="Duplicate")

    assert enrolled["created"] is True
    assert match.matched
    assert match.advertisement_id == "ad-1"
    assert abs(match.estimated_position - query_start) <= 0.5
    assert match.matched_duration >= 5.0
    assert duplicate["created"] is False

    reopened = make_db(tmp_path)
    ads = reopened.list_advertisements()
    assert ads[0]["metadata"] == {"campaign": "test"}
    assert reopened.delete("ad-1") is True
    assert reopened.list_advertisements() == []


def test_false_match_is_not_accepted(tmp_path):
    db = make_db(tmp_path)
    known, sample_rate = synthetic_ad(base=220.0)
    other, _ = synthetic_ad(base=930.0)
    db.enroll(known, sample_rate, name="Known")

    match = db.match(other[: sample_rate * 6], sample_rate)

    assert not match.matched


def test_too_short_or_repetitive_audio_is_not_a_match(tmp_path):
    db = make_db(tmp_path)
    known, sample_rate = synthetic_ad()
    db.enroll(known, sample_rate, name="Known")
    repetitive = np.sin(
        2 * np.pi * 440 * np.arange(sample_rate * 6) / sample_rate
    ).astype(np.float32)

    match = db.match(repetitive, sample_rate)

    assert not match.matched


def test_validated_fingerprint_is_decisive_commercial_evidence():
    detector = AudioFingerprintDetector(max_age_seconds=5.0)
    match = FingerprintMatch(
        matched=True,
        advertisement_id="ad-1",
        name="Spot",
        confidence=0.9,
        matched_duration=4.0,
        estimated_position=3.5,
        aligned_tokens=16,
        query_tokens=18,
        timestamp=10.0,
    )

    actual = detector.evaluate(
        DetectionInput(timestamp=11.0, features={"fingerprint_match": match})
    )

    assert actual.classification == Classification.COMMERCIAL
    assert actual.diagnostics["decisive_evidence"] is True
    assert actual.diagnostics["estimated_position"] == 3.5


def test_weak_or_missing_fingerprint_does_not_classify_commercial():
    detector = AudioFingerprintDetector()
    weak = FingerprintMatch(matched=False, confidence=0.4, timestamp=10.0)

    missing = detector.evaluate(DetectionInput(timestamp=10.0))
    actual = detector.evaluate(
        DetectionInput(timestamp=10.5, features={"fingerprint_match": weak})
    )

    assert missing.classification == Classification.UNKNOWN
    assert actual.classification == Classification.UNKNOWN
