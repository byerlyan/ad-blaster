"""On-screen advertisement label and countdown detection.

Streaming services often render stable UI such as ``Ad 1 of 2``, ``Skip ad``
or ``Advertisement 0:15``.  This detector extracts text from the upper and
lower screen bands with Tesseract, then emits supporting commercial evidence.

A countdown by itself is deliberately never commercial evidence: sports,
news, and game shows all contain legitimate clocks.  The detector also never
emits PROGRAM from the absence of an overlay.  It requires repeated OCR hits
and Hybrid fusion still requires an independent corroborating source.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from detection_pipeline import Classification, DetectionInput, DetectorResult


_STRONG_AD_PATTERNS = (
    re.compile(r"\bskip\s+(?:this\s+)?ad\b", re.IGNORECASE),
    re.compile(r"\bad(?:s|vertisement)?\s*\d+\s*(?:of|/)\s*\d+\b", re.IGNORECASE),
    # Tesseract commonly reads the narrow numeral "1" in "Ad 1 of 2" as
    # l, I, t, or |. Keep this tolerance limited to the full counter phrase.
    re.compile(r"\bad\s*[1ilt|]\s*(?:of|0f)\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bcommercial\s+break\b", re.IGNORECASE),
    re.compile(r"\badvertisement\b", re.IGNORECASE),
)
_BARE_AD_PATTERN = re.compile(r"\bads?\b", re.IGNORECASE)
_SPONSORED_PATTERN = re.compile(r"\bsponsored\b", re.IGNORECASE)
_COUNTDOWN_PATTERN = re.compile(r"(?<!\d)(\d{1,2}):([0-5]\d)(?!\d)")


@dataclass(frozen=True)
class OCRObservation:
    text: str
    confidence: float


OCRRunner = Callable[[np.ndarray], OCRObservation | tuple[str, float]]


def _normalized_regions(
    frame: np.ndarray,
    regions: Sequence[tuple[float, float, float, float]],
) -> list[np.ndarray]:
    height, width = frame.shape[:2]
    crops: list[np.ndarray] = []
    for x, y, w, h in regions:
        left = max(0, min(width - 1, int(round(x * width))))
        top = max(0, min(height - 1, int(round(y * height))))
        right = max(left + 1, min(width, int(round((x + w) * width))))
        bottom = max(top + 1, min(height, int(round((y + h) * height))))
        crops.append(frame[top:bottom, left:right])
    return crops


def _stack_regions(crops: Sequence[np.ndarray]) -> np.ndarray:
    """Combine search bands so one bounded OCR process handles each frame."""
    if len(crops) == 1:
        return crops[0]
    width = max(crop.shape[1] for crop in crops)
    prepared: list[np.ndarray] = []
    for crop in crops:
        if crop.shape[1] != width:
            crop = cv2.resize(
                crop,
                (width, max(1, round(crop.shape[0] * width / crop.shape[1]))),
                interpolation=cv2.INTER_AREA,
            )
        prepared.append(crop)
        separator_shape = (8, width) + (() if crop.ndim == 2 else (crop.shape[2],))
        prepared.append(np.zeros(separator_shape, dtype=crop.dtype))
    return np.vstack(prepared[:-1])


def _countdown_seconds(text: str) -> int | None:
    match = _COUNTDOWN_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


class TesseractOCR:
    """Small subprocess adapter that keeps OCR an optional system feature."""

    def __init__(self, binary: str = "tesseract", timeout_seconds: float = 2.0):
        self.binary = shutil.which(binary) or ""
        self.timeout_seconds = max(0.1, float(timeout_seconds))

    @property
    def available(self) -> bool:
        return bool(self.binary)

    def __call__(self, image: np.ndarray) -> OCRObservation:
        if not self.available:
            raise FileNotFoundError("tesseract executable not found")
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        # Keeping the OCR input near 1000px wide bounds CPU use while still
        # enlarging small streaming-service labels from 720p sources.
        target_width = min(1100, max(700, gray.shape[1]))
        scale = target_width / max(1, gray.shape[1])
        if abs(scale - 1.0) > 0.05:
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
            )
        encoded, png = cv2.imencode(".png", gray)
        if not encoded:
            raise RuntimeError("could not encode OCR region")
        completed = subprocess.run(
            [
                self.binary,
                "stdin",
                "stdout",
                "--psm",
                "6",
                "-l",
                "eng",
                "tsv",
            ],
            input=png.tobytes(),
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or f"tesseract exited {completed.returncode}")

        words: list[str] = []
        confidences: list[float] = []
        output = completed.stdout.decode("utf-8", errors="replace")
        for row in csv.DictReader(io.StringIO(output), delimiter="\t"):
            word = (row.get("text") or "").strip()
            if not word:
                continue
            try:
                confidence = float(row.get("conf", "-1"))
            except ValueError:
                confidence = -1.0
            if confidence >= 0:
                words.append(word)
                confidences.append(confidence / 100.0)
        return OCRObservation(
            text=" ".join(words),
            confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        )


class AdOverlayDetector:
    """Detect explicit on-screen ad UI as corroborating Hybrid evidence."""

    name = "ad_overlay"

    def __init__(
        self,
        *,
        ocr_runner: OCRRunner | None = None,
        tesseract_binary: str = "tesseract",
        regions: Sequence[tuple[float, float, float, float]] = (
            (0.0, 0.0, 1.0, 0.32),
            (0.0, 0.62, 1.0, 0.38),
        ),
        scan_interval: float = 1.0,
        required_confirmations: int = 2,
        hold_seconds: float = 2.0,
        minimum_ocr_confidence: float = 0.35,
        timeout_seconds: float = 2.0,
    ):
        default_ocr = TesseractOCR(tesseract_binary, timeout_seconds)
        self._ocr = ocr_runner or default_ocr
        self._available = (
            default_ocr.available if ocr_runner is None else True
        )
        self._regions = tuple(regions)
        self._scan_interval = max(0.1, float(scan_interval))
        self._required_confirmations = max(1, int(required_confirmations))
        self._hold_seconds = max(0.0, float(hold_seconds))
        self._minimum_ocr_confidence = min(
            1.0, max(0.0, float(minimum_ocr_confidence))
        )
        self._last_scan_at: float | None = None
        self._last_confirmed_at: float | None = None
        self._consecutive_matches = 0
        self._previous_countdown: int | None = None
        self._last_result: DetectorResult | None = None

    @property
    def available(self) -> bool:
        return self._available

    def _unknown(
        self,
        timestamp: float,
        *,
        reason: str,
        text: str = "",
        ocr_confidence: float = 0.0,
        countdown: int | None = None,
        sponsored: bool = False,
        cached: bool = False,
    ) -> DetectorResult:
        return DetectorResult(
            detector_name=self.name,
            classification=Classification.UNKNOWN,
            confidence=0.0,
            timestamp=timestamp,
            confidence_kind="ad_overlay_ocr",
            diagnostics={
                "reason": reason,
                "ocr_text": text[:240],
                "ocr_confidence": ocr_confidence,
                "countdown_seconds": countdown,
                "sponsored_candidate": sponsored,
                "temporal_matches": self._consecutive_matches,
                "required_confirmations": self._required_confirmations,
                "cached": cached,
                "decisive_evidence": False,
            },
        )

    def _commercial(
        self,
        timestamp: float,
        *,
        text: str,
        ocr_confidence: float,
        countdown: int | None,
        countdown_confirmed: bool,
        cached: bool = False,
        held: bool = False,
    ) -> DetectorResult:
        confidence = 0.88
        if countdown is not None:
            confidence = 0.94
        if countdown_confirmed:
            confidence = 0.98
        return DetectorResult(
            detector_name=self.name,
            classification=Classification.COMMERCIAL,
            confidence=confidence,
            timestamp=timestamp,
            confidence_kind="ad_overlay_ocr",
            raw_score=confidence,
            calibrated_probability=None,
            diagnostics={
                "explicit_ad_label": True,
                "ocr_text": text[:240],
                "ocr_confidence": ocr_confidence,
                "countdown_seconds": countdown,
                "countdown_confirmed": countdown_confirmed,
                "temporal_matches": self._consecutive_matches,
                "required_confirmations": self._required_confirmations,
                "cached": cached,
                "held": held,
                # Strong supporting evidence, but never enough to bypass the
                # fusion policy's independent-source requirement.
                "decisive_evidence": False,
            },
        )

    @staticmethod
    def _coerce_observation(
        observation: OCRObservation | tuple[str, float],
    ) -> OCRObservation:
        if isinstance(observation, OCRObservation):
            return observation
        text, confidence = observation
        return OCRObservation(str(text), min(1.0, max(0.0, float(confidence))))

    def evaluate(self, sample: DetectionInput) -> DetectorResult:
        if not self.available:
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "tesseract_unavailable",
            )
        if sample.frame is None or getattr(sample.frame, "size", 0) == 0:
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "missing_video_frame",
            )

        if (
            self._last_scan_at is not None
            and sample.timestamp - self._last_scan_at < self._scan_interval
        ):
            if (
                self._last_confirmed_at is not None
                and sample.timestamp - self._last_confirmed_at <= self._hold_seconds
                and self._last_result is not None
                and self._last_result.classification == Classification.COMMERCIAL
            ):
                diagnostics = self._last_result.diagnostics
                return self._commercial(
                    sample.timestamp,
                    text=str(diagnostics.get("ocr_text", "")),
                    ocr_confidence=float(diagnostics.get("ocr_confidence", 0.0)),
                    countdown=diagnostics.get("countdown_seconds"),
                    countdown_confirmed=bool(diagnostics.get("countdown_confirmed")),
                    cached=True,
                )
            return self._unknown(
                sample.timestamp,
                reason="scan_interval",
                cached=True,
            )

        self._last_scan_at = sample.timestamp
        try:
            ocr_input = _stack_regions(_normalized_regions(sample.frame, self._regions))
            observations = [self._coerce_observation(self._ocr(ocr_input))]
        except subprocess.TimeoutExpired as exc:
            self._consecutive_matches = 0
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "ocr_timeout",
                error=f"Tesseract exceeded its {float(exc.timeout):g}s timeout",
                retryable=True,
            )
        except FileNotFoundError as exc:
            self._available = False
            self._consecutive_matches = 0
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "tesseract_unavailable",
                error=f"{type(exc).__name__}: {exc}"[:240],
                retryable=False,
            )
        except Exception as exc:
            self._consecutive_matches = 0
            return DetectorResult.unavailable(
                self.name,
                sample.timestamp,
                "ocr_error",
                error=f"{type(exc).__name__}: {exc}"[:240],
                retryable=True,
            )
        text = " | ".join(item.text for item in observations if item.text).strip()
        ocr_confidence = max((item.confidence for item in observations), default=0.0)
        countdown = _countdown_seconds(text)
        strong_label = any(pattern.search(text) for pattern in _STRONG_AD_PATTERNS)
        bare_ad = bool(_BARE_AD_PATTERN.search(text))
        sponsored = bool(_SPONSORED_PATTERN.search(text))
        explicit_label = (
            strong_label or bare_ad
        ) and ocr_confidence >= self._minimum_ocr_confidence

        countdown_confirmed = False
        if countdown is not None and self._previous_countdown is not None:
            delta = self._previous_countdown - countdown
            countdown_confirmed = 0 <= delta <= 3
        self._previous_countdown = countdown

        if explicit_label:
            self._consecutive_matches += 1
        else:
            self._consecutive_matches = 0

        if self._consecutive_matches >= self._required_confirmations:
            self._last_confirmed_at = sample.timestamp
            result = self._commercial(
                sample.timestamp,
                text=text,
                ocr_confidence=ocr_confidence,
                countdown=countdown,
                countdown_confirmed=countdown_confirmed,
            )
            self._last_result = result
            return result

        if (
            self._last_confirmed_at is not None
            and sample.timestamp - self._last_confirmed_at <= self._hold_seconds
            and self._last_result is not None
        ):
            diagnostics = self._last_result.diagnostics
            return self._commercial(
                sample.timestamp,
                text=str(diagnostics.get("ocr_text", text)),
                ocr_confidence=float(diagnostics.get("ocr_confidence", ocr_confidence)),
                countdown=diagnostics.get("countdown_seconds"),
                countdown_confirmed=bool(diagnostics.get("countdown_confirmed")),
                held=True,
            )

        reason = (
            "awaiting_confirmation"
            if explicit_label
            else "low_ocr_confidence"
            if (strong_label or bare_ad)
            else "countdown_without_ad_label"
            if countdown is not None
            else "sponsored_without_explicit_ad_label"
            if sponsored
            else "no_explicit_ad_overlay"
        )
        result = self._unknown(
            sample.timestamp,
            reason=reason,
            text=text,
            ocr_confidence=ocr_confidence,
            countdown=countdown,
            sponsored=sponsored,
        )
        self._last_result = result
        return result
