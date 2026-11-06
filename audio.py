"""Optional, bounded FFmpeg audio capture and lightweight feature analysis.

Audio is captured in background threads so a slow or missing device cannot
block video capture, the dashboard, or playback control. Only a bounded PCM
ring buffer is retained in memory; audio is never written to disk here.
"""

from __future__ import annotations

import logging
import math
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from detection_pipeline import Classification, DetectionInput, DetectorResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str


@dataclass(frozen=True)
class AudioFeatures:
    timestamp: float
    sample_rate: int
    window_seconds: float
    rms: float
    rms_dbfs: float
    dynamic_range_db: float
    silence_ratio: float
    spectral_centroid_hz: float
    spectral_flux: float
    loudness_change_db: float
    mfcc: tuple[float, ...]
    capture_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mfcc"] = list(self.mfcc)
        return value


def parse_avfoundation_audio_devices(stderr: str) -> list[AudioDevice]:
    """Parse the audio section of ``ffmpeg -list_devices true`` output."""
    devices: list[AudioDevice] = []
    in_audio = False
    for line in stderr.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if "AVFoundation video devices" in line:
            in_audio = False
            continue
        if not in_audio:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if match:
            devices.append(AudioDevice(int(match.group(1)), match.group(2)))
    return devices


def probe_avfoundation_audio_devices(ffmpeg_binary: str = "ffmpeg") -> list[AudioDevice]:
    """Return macOS audio inputs exposed to FFmpeg/AVFoundation."""
    if platform.system() != "Darwin":
        return []
    binary = shutil.which(ffmpeg_binary)
    if binary is None:
        return []
    try:
        completed = subprocess.run(
            [
                binary,
                "-hide_banner",
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_avfoundation_audio_devices(completed.stderr)


class AudioFeatureExtractor:
    """Compute inexpensive audio descriptors using NumPy only."""

    def __init__(self, sample_rate: int = 16000, silence_dbfs: float = -45.0):
        self.sample_rate = sample_rate
        self.silence_dbfs = silence_dbfs
        self._previous_spectrum: np.ndarray | None = None
        self._previous_rms_db: float | None = None
        self._mel_cache: dict[int, np.ndarray] = {}
        self._dct = np.cos(
            np.pi
            * np.arange(13, dtype=np.float32)[:, None]
            * (np.arange(20, dtype=np.float32)[None, :] + 0.5)
            / 20.0
        )

    @staticmethod
    def _db(value: float) -> float:
        return 20.0 * math.log10(max(value, 1e-8))

    def _mel_filters(self, fft_size: int) -> np.ndarray:
        cached = self._mel_cache.get(fft_size)
        if cached is not None:
            return cached
        n_bins = fft_size // 2 + 1
        hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
        mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        mel_points = np.linspace(
            hz_to_mel(50.0), hz_to_mel(self.sample_rate / 2.0), 22
        )
        bins = np.floor((fft_size + 1) * mel_to_hz(mel_points) / self.sample_rate).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        filters = np.zeros((20, n_bins), dtype=np.float32)
        for i in range(20):
            left, center, right = bins[i : i + 3]
            if center > left:
                filters[i, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
            if right > center:
                filters[i, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
        self._mel_cache[fft_size] = filters
        return filters

    def extract(
        self,
        samples: np.ndarray,
        timestamp: float,
        capture_latency_ms: float = 0.0,
    ) -> AudioFeatures:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ValueError("audio window is empty")
        samples = np.clip(samples, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(samples * samples)))
        rms_db = self._db(rms)

        block_size = max(1, int(self.sample_rate * 0.02))
        usable = samples[: (samples.size // block_size) * block_size]
        if usable.size:
            blocks = usable.reshape(-1, block_size)
            block_rms = np.sqrt(np.mean(blocks * blocks, axis=1))
            block_db = 20.0 * np.log10(np.maximum(block_rms, 1e-8))
            silence_ratio = float(np.mean(block_db <= self.silence_dbfs))
            dynamic_range = float(np.percentile(block_db, 95) - np.percentile(block_db, 10))
        else:
            silence_ratio = float(rms_db <= self.silence_dbfs)
            dynamic_range = 0.0

        fft_size = 1 << max(8, int(samples.size - 1).bit_length())
        windowed = samples * np.hanning(samples.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(windowed, n=fft_size)).astype(np.float32)
        spectrum_sum = float(spectrum.sum())
        normalized = spectrum / max(spectrum_sum, 1e-8)
        freqs = np.fft.rfftfreq(fft_size, 1.0 / self.sample_rate)
        centroid = float(np.sum(freqs * normalized))
        if self._previous_spectrum is None or self._previous_spectrum.shape != normalized.shape:
            flux = 0.0
        else:
            flux = float(np.sqrt(np.mean((normalized - self._previous_spectrum) ** 2)))
        self._previous_spectrum = normalized

        power = spectrum * spectrum
        mel_energy = self._mel_filters(fft_size) @ power
        mfcc = self._dct @ np.log(np.maximum(mel_energy, 1e-10))
        mfcc = tuple(float(v) for v in mfcc)

        loudness_change = (
            0.0 if self._previous_rms_db is None else rms_db - self._previous_rms_db
        )
        self._previous_rms_db = rms_db
        return AudioFeatures(
            timestamp=float(timestamp),
            sample_rate=self.sample_rate,
            window_seconds=samples.size / self.sample_rate,
            rms=rms,
            rms_dbfs=rms_db,
            dynamic_range_db=dynamic_range,
            silence_ratio=silence_ratio,
            spectral_centroid_hz=centroid,
            spectral_flux=flux,
            loudness_change_db=loudness_change,
            mfcc=mfcc,
            capture_latency_ms=max(0.0, float(capture_latency_ms)),
        )


class AudioAnalyzer:
    """Background FFmpeg PCM reader with bounded queue and ring buffer."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        device_name: str = "",
        sample_rate: int = 16000,
        chunk_seconds: float = 0.25,
        window_seconds: float = 2.0,
        analysis_interval: float = 0.5,
        retention_seconds: float = 180.0,
        silence_dbfs: float = -45.0,
    ):
        self.ffmpeg_binary = ffmpeg_binary
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.window_seconds = window_seconds
        self.analysis_interval = analysis_interval
        self.retention_seconds = retention_seconds
        self._extractor = AudioFeatureExtractor(sample_rate, silence_dbfs)
        self._queue: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=4)
        max_chunks = max(2, int(math.ceil(retention_seconds / chunk_seconds)))
        self._ring: deque[tuple[float, np.ndarray]] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._latest: AudioFeatures | None = None
        self.available = False
        self.error: str | None = None
        self.device: AudioDevice | None = None
        self.dropped_chunks = 0

    def _select_device(self, devices: list[AudioDevice]) -> AudioDevice | None:
        if not devices:
            return None
        if self.device_name:
            wanted = self.device_name.casefold()
            for device in devices:
                if wanted in device.name.casefold():
                    return device
            return None
        # Never assume that the first input belongs to the HDMI capture card.
        return None

    def start(self) -> bool:
        if platform.system() != "Darwin":
            self.error = "audio capture currently supports macOS AVFoundation only"
            return False
        binary = shutil.which(self.ffmpeg_binary)
        if binary is None:
            self.error = f"FFmpeg executable not found: {self.ffmpeg_binary}"
            return False
        devices = probe_avfoundation_audio_devices(binary)
        self.device = self._select_device(devices)
        if self.device is None:
            available = ", ".join(d.name for d in devices) or "none"
            self.error = (
                f"configured audio device {self.device_name!r} not found; "
                f"available AVFoundation audio inputs: {available}"
            )
            return False
        command = [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-i",
            f":{self.device.index}",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as exc:
            self.error = str(exc)
            return False
        self._stop.clear()
        self.available = True
        self.error = None
        self._reader = threading.Thread(target=self._reader_loop, daemon=True, name="audio-reader")
        self._worker = threading.Thread(target=self._analysis_loop, daemon=True, name="audio-analysis")
        self._reader.start()
        self._worker.start()
        logger.info("Audio capture started: %s", self.device.name)
        return True

    def _reader_loop(self) -> None:
        chunk_bytes = max(2, int(self.sample_rate * self.chunk_seconds) * 2)
        assert self._process is not None and self._process.stdout is not None
        while not self._stop.is_set():
            data = self._process.stdout.read(chunk_bytes)
            if not data:
                if self._process.poll() is not None:
                    self.error = f"FFmpeg audio process exited ({self._process.returncode})"
                    self.available = False
                    return
                continue
            item = (time.time(), data)
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self.dropped_chunks += 1
                self._queue.put_nowait(item)

    def _analysis_loop(self) -> None:
        last_analysis = 0.0
        required = max(1, int(self.window_seconds * self.sample_rate))
        while not self._stop.is_set():
            try:
                end_timestamp, data = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            with self._lock:
                self._ring.append((end_timestamp, samples))
                chunks = list(self._ring)
            if end_timestamp - last_analysis < self.analysis_interval:
                continue
            window = np.concatenate([chunk for _, chunk in chunks])
            if window.size < required:
                continue
            window = window[-required:]
            try:
                features = self._extractor.extract(
                    window,
                    end_timestamp,
                    capture_latency_ms=max(0.0, (time.time() - end_timestamp) * 1000.0),
                )
            except Exception as exc:
                self.error = f"audio feature extraction failed: {exc}"
                logger.warning(self.error)
                continue
            with self._lock:
                self._latest = features
            last_analysis = end_timestamp

    def latest(self) -> AudioFeatures | None:
        with self._lock:
            return self._latest

    def recent_audio(self, seconds: float) -> tuple[np.ndarray, int]:
        """Copy at most ``seconds`` of bounded in-memory PCM for explicit use."""
        count = max(1, int(float(seconds) * self.sample_rate))
        with self._lock:
            chunks = [chunk for _, chunk in self._ring]
        if not chunks:
            return np.empty(0, dtype=np.float32), self.sample_rate
        return np.concatenate(chunks)[-count:].copy(), self.sample_rate

    def status(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "available": self.available,
            "device": self.device.name if self.device else None,
            "error": self.error,
            "dropped_chunks": self.dropped_chunks,
            "queue_backlog": self._queue.qsize(),
            "latest": latest.to_dict() if latest else None,
        }

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        for thread in (self._reader, self._worker):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2)
        self.available = False
        self._process = None


class AudioTransitionDetector:
    """Treat transitions as supporting boundary evidence, never as decisive."""

    name = "audio_transition"

    def __init__(
        self,
        *,
        max_age_seconds: float = 3.0,
        loudness_change_db: float = 8.0,
        spectral_flux: float = 0.015,
        silence_ratio: float = 0.60,
        score_threshold: float = 0.65,
    ):
        self.max_age_seconds = max_age_seconds
        self.loudness_threshold = loudness_change_db
        self.flux_threshold = spectral_flux
        self.silence_threshold = silence_ratio
        self.score_threshold = score_threshold

    def evaluate(self, sample: DetectionInput) -> DetectorResult:
        features = sample.features.get("audio")
        if features is None:
            return DetectorResult.unavailable(self.name, sample.timestamp, "missing_audio")
        if isinstance(features, AudioFeatures):
            values = features.to_dict()
        else:
            values = dict(features)
        age = max(0.0, sample.timestamp - float(values["timestamp"]))
        if age > self.max_age_seconds:
            return DetectorResult.unavailable(
                self.name, sample.timestamp, "stale_audio", age_seconds=age
            )
        loudness = min(1.0, abs(float(values["loudness_change_db"])) / max(self.loudness_threshold, 1e-6))
        flux = min(1.0, float(values["spectral_flux"]) / max(self.flux_threshold, 1e-6))
        silence = min(1.0, float(values["silence_ratio"]) / max(self.silence_threshold, 1e-6))
        score = 0.45 * loudness + 0.35 * flux + 0.20 * silence
        classification = (
            Classification.COMMERCIAL
            if score >= self.score_threshold
            else Classification.UNKNOWN
        )
        return DetectorResult(
            detector_name=self.name,
            classification=classification,
            confidence=score,
            timestamp=sample.timestamp,
            diagnostics={
                "age_seconds": age,
                "loudness_component": loudness,
                "flux_component": flux,
                "silence_component": silence,
                "supporting_evidence_only": True,
                **values,
            },
            confidence_kind="transition_score",
            raw_score=score,
            calibrated_probability=None,
        )
