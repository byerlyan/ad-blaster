"""Persistent, local, replaceable audio fingerprint matching.

The default algorithm stores compact spectral tokens and timing offsets in
SQLite. It intentionally stores no PCM. The interfaces allow Chromaprint,
audfprint, or another matcher to replace it later without changing the engine.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from detection_pipeline import Classification, DetectionInput, DetectorResult


@dataclass(frozen=True)
class FingerprintToken:
    offset_ms: int
    value: str


@dataclass(frozen=True)
class FingerprintMatch:
    matched: bool
    advertisement_id: str | None = None
    name: str | None = None
    confidence: float = 0.0
    matched_duration: float = 0.0
    estimated_position: float | None = None
    aligned_tokens: int = 0
    query_tokens: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FingerprintAlgorithm(Protocol):
    name: str
    version: str

    def fingerprint(self, samples: np.ndarray, sample_rate: int) -> list[FingerprintToken]:
        ...


class SpectralTokenFingerprint:
    """Small exact-signal spectral token fingerprint suitable for HDMI PCM."""

    name = "spectral_token"
    version = "1"

    def __init__(self, frame_seconds: float = 0.50, hop_seconds: float = 0.25):
        self.frame_seconds = frame_seconds
        self.hop_seconds = hop_seconds

    def fingerprint(self, samples: np.ndarray, sample_rate: int) -> list[FingerprintToken]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        frame_size = max(256, int(sample_rate * self.frame_seconds))
        hop = max(128, int(sample_rate * self.hop_seconds))
        if samples.size < frame_size:
            return []
        fft_size = 1 << int(frame_size - 1).bit_length()
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        edges = np.geomspace(80.0, min(7000.0, sample_rate / 2.0), 25)
        band_masks = [(freqs >= left) & (freqs < right) for left, right in zip(edges, edges[1:])]
        tokens: list[FingerprintToken] = []
        previous_signature = b""
        window = np.hanning(frame_size).astype(np.float32)
        for start in range(0, samples.size - frame_size + 1, hop):
            frame = samples[start : start + frame_size]
            spectrum = np.abs(np.fft.rfft(frame * window, n=fft_size)) ** 2
            energies = np.array(
                [float(spectrum[mask].sum()) if np.any(mask) else 0.0 for mask in band_masks],
                dtype=np.float32,
            )
            log_energy = np.log1p(energies)
            centered = log_energy - np.median(log_energy)
            sign_bits = np.packbits((centered > 0).astype(np.uint8)).tobytes()
            top = np.sort(np.argsort(log_energy)[-4:]).astype(np.uint8).tobytes()
            signature = sign_bits + top
            digest = hashlib.blake2b(previous_signature + signature, digest_size=8).hexdigest()
            tokens.append(FingerprintToken(int(round(start * 1000.0 / sample_rate)), digest))
            previous_signature = signature
        return tokens


class FingerprintDatabase:
    """SQLite metadata/token store with alignment-vote matching."""

    def __init__(
        self,
        path: str,
        algorithm: FingerprintAlgorithm | None = None,
        *,
        match_threshold: float = 0.55,
        minimum_duration: float = 2.0,
        minimum_unique_tokens: int = 4,
    ):
        self.path = str(path)
        self.algorithm = algorithm or SpectralTokenFingerprint()
        self.match_threshold = match_threshold
        self.minimum_duration = minimum_duration
        self.minimum_unique_tokens = minimum_unique_tokens
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS advertisements (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    duration REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fingerprint_tokens (
                    advertisement_id TEXT NOT NULL REFERENCES advertisements(id) ON DELETE CASCADE,
                    offset_ms INTEGER NOT NULL,
                    token TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fingerprint_token ON fingerprint_tokens(token);
                CREATE INDEX IF NOT EXISTS idx_fingerprint_ad ON fingerprint_tokens(advertisement_id);
                PRAGMA foreign_keys=ON;
                """
            )

    def list_advertisements(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM advertisements ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "duration": row["duration"],
                "metadata": json.loads(row["metadata_json"]),
                "algorithm": row["algorithm"],
                "algorithm_version": row["algorithm_version"],
                "token_count": row["token_count"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def enroll(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        name: str,
        advertisement_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        duplicate_threshold: float = 0.75,
    ) -> dict[str, Any]:
        tokens = self.algorithm.fingerprint(samples, sample_rate)
        if len(tokens) < self.minimum_unique_tokens:
            raise ValueError("not enough audio fingerprint tokens to enroll")
        duplicate = self.match_tokens(
            tokens,
            threshold=duplicate_threshold,
            minimum_duration=self.minimum_duration,
        )
        if duplicate.matched:
            return {"created": False, "duplicate": duplicate.to_dict()}
        ad_id = advertisement_id or str(uuid.uuid4())
        duration = len(samples) / float(sample_rate)
        with self._connect() as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute(
                """INSERT INTO advertisements
                (id, name, duration, metadata_json, algorithm, algorithm_version, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ad_id,
                    name,
                    duration,
                    json.dumps(metadata or {}, sort_keys=True),
                    self.algorithm.name,
                    self.algorithm.version,
                    len(tokens),
                    time.time(),
                ),
            )
            db.executemany(
                "INSERT INTO fingerprint_tokens(advertisement_id, offset_ms, token) VALUES (?, ?, ?)",
                [(ad_id, token.offset_ms, token.value) for token in tokens],
            )
        return {"created": True, "advertisement_id": ad_id, "token_count": len(tokens)}

    def delete(self, advertisement_id: str) -> bool:
        with self._connect() as db:
            db.execute("PRAGMA foreign_keys=ON")
            cursor = db.execute(
                "DELETE FROM advertisements WHERE id = ?", (advertisement_id,)
            )
        return cursor.rowcount > 0

    def match(self, samples: np.ndarray, sample_rate: int) -> FingerprintMatch:
        return self.match_tokens(self.algorithm.fingerprint(samples, sample_rate))

    def match_tokens(
        self,
        tokens: Sequence[FingerprintToken],
        *,
        threshold: float | None = None,
        minimum_duration: float | None = None,
    ) -> FingerprintMatch:
        timestamp = time.time()
        unique_values = {token.value for token in tokens}
        if len(tokens) < self.minimum_unique_tokens or len(unique_values) < self.minimum_unique_tokens:
            return FingerprintMatch(False, query_tokens=len(tokens), timestamp=timestamp)
        rows: list[sqlite3.Row] = []
        values = sorted(unique_values)
        with self._connect() as db:
            for start in range(0, len(values), 400):
                batch = values[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    db.execute(
                        f"""SELECT t.advertisement_id, t.offset_ms, t.token, a.name
                        FROM fingerprint_tokens t JOIN advertisements a ON a.id=t.advertisement_id
                        WHERE t.token IN ({placeholders})""",
                        batch,
                    ).fetchall()
                )
        query_by_value: dict[str, list[int]] = {}
        for token in tokens:
            query_by_value.setdefault(token.value, []).append(token.offset_ms)
        aligned: dict[tuple[str, int], set[tuple[int, int, str]]] = {}
        names: dict[str, str] = {}
        for row in rows:
            ad_id = row["advertisement_id"]
            names[ad_id] = row["name"]
            db_offset = int(row["offset_ms"])
            for query_offset in query_by_value[row["token"]]:
                delta = int(round((db_offset - query_offset) / 250.0) * 250)
                aligned.setdefault((ad_id, delta), set()).add(
                    (query_offset, db_offset, row["token"])
                )
        if not aligned:
            return FingerprintMatch(False, query_tokens=len(tokens), timestamp=timestamp)
        (ad_id, delta), hits = max(aligned.items(), key=lambda item: len(item[1]))
        aligned_query_offsets = sorted({hit[0] for hit in hits})
        hit_count = len(aligned_query_offsets)
        confidence = hit_count / max(1, len(tokens))
        matched_duration = (
            0.0
            if not aligned_query_offsets
            else (aligned_query_offsets[-1] - aligned_query_offsets[0]) / 1000.0
            + float(getattr(self.algorithm, "frame_seconds", 0.5))
        )
        required_confidence = self.match_threshold if threshold is None else threshold
        required_duration = self.minimum_duration if minimum_duration is None else minimum_duration
        matched = confidence >= required_confidence and matched_duration >= required_duration
        return FingerprintMatch(
            matched=matched,
            advertisement_id=ad_id if matched else None,
            name=names.get(ad_id) if matched else None,
            confidence=confidence,
            matched_duration=matched_duration,
            estimated_position=max(0.0, delta / 1000.0) if matched else None,
            aligned_tokens=hit_count,
            query_tokens=len(tokens),
            timestamp=timestamp,
        )


class FingerprintMatcherWorker:
    """Periodically match bounded recent audio off the video event loop."""

    def __init__(self, analyzer, database: FingerprintDatabase, query_seconds: float = 6.0, interval: float = 2.0):
        self.analyzer = analyzer
        self.database = database
        self.query_seconds = query_seconds
        self.interval = interval
        self._latest: FingerprintMatch | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="fingerprint-matcher")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            samples, sample_rate = self.analyzer.recent_audio(self.query_seconds)
            if samples.size < sample_rate * min(2.0, self.query_seconds):
                continue
            try:
                match = self.database.match(samples, sample_rate)
            except Exception as exc:
                self.error = f"fingerprint match failed: {exc}"
                continue
            with self._lock:
                self._latest = match

    def latest(self) -> FingerprintMatch | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)

    def status(self) -> dict[str, Any]:
        latest = self.latest()
        return {"available": True, "error": self.error, "latest": latest.to_dict() if latest else None}


class AudioFingerprintDetector:
    name = "audio_fingerprint"

    def __init__(self, max_age_seconds: float = 5.0):
        self.max_age_seconds = max_age_seconds

    def evaluate(self, sample: DetectionInput) -> DetectorResult:
        match = sample.features.get("fingerprint_match")
        if match is None:
            return DetectorResult.unavailable(self.name, sample.timestamp, "no_fingerprint_match")
        values = match.to_dict() if isinstance(match, FingerprintMatch) else dict(match)
        age = max(0.0, sample.timestamp - float(values.get("timestamp", 0.0)))
        if age > self.max_age_seconds:
            return DetectorResult.unavailable(self.name, sample.timestamp, "stale_fingerprint_match", age_seconds=age)
        if not values.get("matched", False):
            return DetectorResult(
                detector_name=self.name,
                classification=Classification.UNKNOWN,
                confidence=float(values.get("confidence", 0.0)),
                timestamp=sample.timestamp,
                diagnostics={"decisive_evidence": False, **values},
                confidence_kind="fingerprint_match_quality",
            )
        return DetectorResult(
            detector_name=self.name,
            classification=Classification.COMMERCIAL,
            confidence=float(values["confidence"]),
            timestamp=sample.timestamp,
            diagnostics={"decisive_evidence": True, **values},
            confidence_kind="fingerprint_match_quality",
            raw_score=float(values["confidence"]),
        )
