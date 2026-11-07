"""FastAPI web server, dashboard and API for controlling the ad blocker."""

import asyncio
import base64
import ipaddress
import json
import logging
import os
import pathlib
import platform
import socket
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import partial
from io import BytesIO
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from calibrate import delete_profile
from capture import FrameCapture
from corrections import CorrectionStore
from device_setup import (
    AppleTVPairingError,
    AppleTVSetup,
    capture_device_diagnostics,
)
from engine import Engine

logger = logging.getLogger(__name__)

engine = Engine()
corrections = CorrectionStore(config.CORRECTIONS_DB_PATH)
apple_tv_setup = AppleTVSetup(os.path.join(os.path.dirname(__file__), "credentials.json"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Resume prior work and always restore engine-owned volume on shutdown."""
    await engine.resume_last_session()
    try:
        yield
    finally:
        await engine.shutdown()


app = FastAPI(title="The Clicker", lifespan=lifespan)


def _is_local_network_client(host: str | None) -> bool:
    """Allow direct use only from loopback, private, or link-local clients."""
    normalized = (host or "").split("%", 1)[0]
    if "pytest" in sys.modules and normalized in {"testclient", "testserver"}:
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


@app.middleware("http")
async def protect_lan_dashboard(request: Request, call_next):
    """Permit direct access from the local machine and private LAN only."""
    client_host = request.client.host if request.client else None
    if not _is_local_network_client(client_host):
        return JSONResponse(
            {"detail": "The Clicker is available only from the local network"},
            status_code=403,
        )
    return await call_next(request)


@app.middleware("http")
async def dashboard_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    return response


# ------------------------------------------------------------------
# API models
# ------------------------------------------------------------------

class StartRequest(BaseModel):
    channel: str | None = None
    shadow: bool = False
    detection_mode: Literal["legacy", "hybrid"] | None = None


class CalibrateRequest(BaseModel):
    channel: str | None = None


class InteractiveCalibrateRequest(BaseModel):
    channel: str
    x: int
    y: int
    w: int
    h: int
    # Seconds of footage to sample. Omit to use the CALIBRATION_DURATION setting.
    duration: int | None = None


class ColorCalibrateRequest(BaseModel):
    channel: str
    # Optional ROI rectangle. If absent, the whole frame is used.
    x: int | None = None
    y: int | None = None
    w: int | None = None
    h: int | None = None
    duration: int | None = None


class ComboCalibrateRequest(BaseModel):
    """Combo profile: logo rect (required) + full frame color signal."""
    channel: str
    x: int
    y: int
    w: int
    h: int
    duration: int | None = None


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Current engine status, polled by the dashboard."""
    return asdict(engine.get_status())


@app.post("/api/start")
async def start_engine(req: StartRequest = StartRequest()):
    """Start the detection engine."""
    try:
        await engine.start(
            channel=req.channel,
            shadow=req.shadow,
            detection_mode=req.detection_mode,
        )
        return {
            "ok": True,
            "state": engine.state.value,
            "detection_mode": engine.get_status().detection_mode,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_engine():
    """Stop the detection engine."""
    await engine.stop()
    return {"ok": True, "state": engine.state.value}


class HoldRequest(BaseModel):
    seconds: float
    reason: str = ""


@app.post("/api/hold")
async def hold_capture(req: HoldRequest):
    """Keep the capture card free for another process for `seconds`.

    The overnight capture rig calls this before it records; the engine and
    the dashboard's ad hoc frame grabs refuse to open the card until the hold
    expires or is released.
    """
    until = engine.hold_capture(req.seconds, req.reason)
    return {"ok": True, "held_until": until}


@app.post("/api/hold/release")
async def release_capture():
    engine.release_capture()
    return {"ok": True}


def _refuse_if_held():
    if engine._cap is None and engine.capture_held():
        raise HTTPException(status_code=409, detail="capture card is held by another process")


@app.post("/api/disarm")
async def disarm_engine():
    """Emergency restore, unmute, reset detection, keep running."""
    await engine.disarm()
    return {"ok": True}


# ------------------------------------------------------------------
# Threshold tuning, labeled sample collection
# ------------------------------------------------------------------

class MarkRequest(BaseModel):
    kind: str  # "content" | "commercial"


class FingerprintEnrollRequest(BaseModel):
    name: str
    seconds: float = 10.0
    advertisement_id: str | None = None
    metadata: dict = Field(default_factory=dict)


@app.post("/api/mark/start")
async def mark_start(req: MarkRequest):
    """Begin collecting confidence samples labeled as content or commercial."""
    try:
        return engine.start_marking(req.kind)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/mark/stop")
async def mark_stop():
    """Stop the active marking window. Samples gathered so far are retained."""
    return engine.stop_marking()


@app.post("/api/mark/clear")
async def mark_clear():
    """Discard all collected samples and reset both buckets."""
    engine.clear_samples()
    return {"ok": True}


@app.get("/api/mark/status")
async def mark_status():
    """Current marking state and sample counts."""
    return engine._marking_status()


@app.get("/api/suggest-thresholds")
async def suggest_thresholds():
    """Compute suggested match + recovery thresholds from labeled samples."""
    return engine.suggest_thresholds()


@app.get("/api/stats")
async def get_stats():
    """Session statistics."""
    stats = dict(engine._stats)
    if stats.get("session_start"):
        stats["session_duration"] = round(time.time() - stats["session_start"], 1)
        stats["time_saved_min"] = round(stats["total_commercial_time"] / 60, 1)
    else:
        stats["session_duration"] = 0
        stats["time_saved_min"] = 0
    return stats


@app.post("/api/control/mute")
async def manual_mute():
    try:
        changed = await engine.manual_mute()
        return {"ok": True, "changed": changed}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/control/unmute")
async def manual_unmute():
    try:
        changed = await engine.manual_unmute()
        return {"ok": True, "changed": changed}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class CorrectionRequest(BaseModel):
    label: Literal["program", "commercial"]
    action: Literal["none", "mute", "unmute"] = "none"


class RelabelRequest(BaseModel):
    label: Literal["program", "commercial"]


@app.post("/api/corrections")
async def create_correction(req: CorrectionRequest):
    status = engine.get_status()
    if not status.session_id:
        raise HTTPException(status_code=409, detail="Start a capture session before labeling")
    if req.action == "mute":
        await manual_mute()
    elif req.action == "unmute":
        await manual_unmute()
    item, created = corrections.add(
        label=req.label,
        session_id=status.session_id,
        channel=status.channel,
        action=req.action,
        evidence=engine.correction_evidence(),
    )
    return {"ok": True, "created": created, "correction": item}


@app.get("/api/corrections")
async def list_corrections(limit: int = 100):
    return {"corrections": corrections.list(limit)}


@app.post("/api/corrections/{correction_id}/undo")
async def undo_correction(correction_id: str):
    item = corrections.undo(correction_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Active correction not found")
    return {"ok": True, "correction": item}


@app.patch("/api/corrections/{correction_id}")
async def relabel_correction(correction_id: str, req: RelabelRequest):
    item = corrections.relabel(correction_id, req.label)
    if item is None:
        raise HTTPException(status_code=404, detail="Correction not found")
    return {"ok": True, "correction": item}


@app.get("/api/activity")
async def activity(limit: int = 100):
    def event_evidence(event: dict) -> dict:
        timestamp = float(event.get("t") or 0)
        nearest = min(
            engine._history,
            key=lambda point: abs(float(point.get("t") or 0) - timestamp),
            default={},
        )
        return {
            "detection_mode": engine.get_status().detection_mode,
            "confidence_template": nearest.get("tmpl"),
            "confidence_edge": nearest.get("edge"),
            "confidence_color": nearest.get("color"),
        }

    events = [
        {
            "id": f"event-{index}-{event.get('t')}",
            "timestamp": event.get("t"),
            "kind": event.get("type", "event"),
            "reason": event.get("reason"),
            "source": "engine",
            "evidence": event_evidence(event),
        }
        for index, event in enumerate(engine._events[-max(1, min(limit, 500)):])
    ]
    labels = [
        {
            "id": item["id"],
            "timestamp": item["timestamp"],
            "kind": f"labeled_{item['label']}",
            "reason": item["action"],
            "source": "correction",
            "undone": item["undone"],
            "evidence": item["evidence"],
        }
        for item in corrections.list(limit)
    ]
    combined = sorted(events + labels, key=lambda item: item["timestamp"] or 0, reverse=True)
    return {"activity": combined[:max(1, min(limit, 500))]}


@app.get("/api/devices/capture")
async def capture_devices():
    diagnostics = await asyncio.to_thread(capture_device_diagnostics)
    return {
        **diagnostics,
        "selected": (
            config.CAPTURE_DEVICE_NAME
            if platform.system() == "Darwin"
            else str(config.CAPTURE_DEVICE)
        ),
    }


class CaptureDeviceSelectRequest(BaseModel):
    id: str
    name: str


@app.post("/api/devices/capture/select")
async def select_capture_device(req: CaptureDeviceSelectRequest):
    if platform.system() == "Darwin":
        snapshot = _apply_settings({"CAPTURE_DEVICE_NAME": req.name})
        selected = req.name
    else:
        snapshot = _apply_settings({"CAPTURE_DEVICE": req.id})
        selected = req.id
    return {"ok": True, "selected": selected, **snapshot}


@app.get("/api/devices/appletv")
async def apple_tv_devices():
    try:
        return {"devices": await apple_tv_setup.scan(), "selected": config.APPLE_TV_ID}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Apple TV discovery failed: {exc}")


class AppleTVPairStartRequest(BaseModel):
    identifier: str
    protocol: Literal["Companion", "AirPlay"] = "Companion"


class AppleTVPairFinishRequest(BaseModel):
    token: str
    pin: int | None = None


@app.post("/api/devices/appletv/pair/start")
async def apple_tv_pair_start(req: AppleTVPairStartRequest):
    try:
        result = await apple_tv_setup.begin(req.identifier, req.protocol)
        config.APPLE_TV_ID = req.identifier
        _write_runtime_settings()
        return result
    except (ValueError, AppleTVPairingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error starting %s pairing", req.protocol)
        message = str(exc).strip() or type(exc).__name__
        raise HTTPException(
            status_code=502,
            detail=f"Could not start {req.protocol} pairing: {message}",
        ) from exc


@app.post("/api/devices/appletv/pair/finish")
async def apple_tv_pair_finish(req: AppleTVPairFinishRequest):
    try:
        return await apple_tv_setup.finish(req.token, req.pin)
    except (ValueError, AppleTVPairingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error finishing Apple TV pairing")
        message = str(exc).strip() or type(exc).__name__
        raise HTTPException(
            status_code=502,
            detail=f"Could not finish Apple TV pairing: {message}",
        ) from exc


@app.get("/api/fingerprints")
async def list_fingerprints():
    return {"advertisements": await engine.list_fingerprints()}


@app.post("/api/fingerprints/enroll")
async def enroll_fingerprint(req: FingerprintEnrollRequest):
    """Explicitly trust recent buffered audio as a confirmed advertisement."""
    try:
        return await engine.enroll_fingerprint(
            name=req.name,
            seconds=req.seconds,
            advertisement_id=req.advertisement_id,
            metadata=req.metadata,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/fingerprints/{advertisement_id}")
async def delete_fingerprint(advertisement_id: str):
    deleted = await engine.delete_fingerprint(advertisement_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="fingerprint not found")
    return {"ok": True}


@app.get("/api/ml/dataset/export")
async def export_ml_dataset():
    await engine.export_feature_dataset(config.ML_EXPORT_PATH)
    return FileResponse(
        config.ML_EXPORT_PATH,
        media_type="text/csv",
        filename="the-clicker-features.csv",
    )


@app.post("/api/toggle")
async def toggle_engine():
    """Toggle between active (intervening) and disabled (monitoring only)."""
    enabled = engine.toggle()
    return {"ok": True, "enabled": enabled, "state": engine.state.value}


@app.post("/api/calibrate")
async def calibrate_channel(req: CalibrateRequest = CalibrateRequest()):
    """Run calibration for a channel. Auto detects if channel is omitted."""
    try:
        channel = await engine.calibrate(channel=req.channel)
        return {"ok": True, "channel": channel}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profiles")
async def list_profiles():
    """List all saved channel profiles."""
    profiles_dir = config.PROFILES_DIR
    if not os.path.exists(profiles_dir):
        return {"profiles": []}

    profiles = []
    for f in os.listdir(profiles_dir):
        if f.endswith(".json"):
            profiles.append(f.removesuffix(".json"))
    return {"profiles": sorted(profiles)}


@app.delete("/api/profile/{channel}")
async def delete_profile_endpoint(channel: str):
    """Delete a channel profile."""
    json_path = os.path.join(config.PROFILES_DIR, f"{channel}.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail=f"No profile for '{channel}'")
    delete_profile(channel)
    return {"ok": True, "deleted": channel}


def _render_palette(histogram: np.ndarray, width: int = 360, height: int = 60, n_colors: int = 12) -> np.ndarray:
    """Render an HSV hue/saturation histogram as a horizontal swatch strip of dominant colors.

    Picks the top `n_colors` bins by weight and lays them out left to right,
    each rectangle's width proportional to its weight. Bin centers are converted
    to BGR with V=255 for visibility.
    """
    hist = histogram.astype(np.float32)
    if hist.sum() <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    h_bins, s_bins = hist.shape
    flat = hist.flatten()
    top_idx = np.argsort(flat)[::-1][:n_colors]
    weights = flat[top_idx]
    weights = weights / weights.sum()

    img = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for idx, w in zip(top_idx, weights):
        hb, sb = divmod(int(idx), s_bins)
        # Bin centers, mapped back to OpenCV HSV ranges (H: 0-180, S: 0-255)
        hue = int((hb + 0.5) * (180.0 / h_bins))
        sat = int((sb + 0.5) * (256.0 / s_bins))
        swatch_hsv = np.array([[[hue, sat, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        ww = max(1, int(round(w * width)))
        img[:, x:x + ww] = bgr
        x += ww
        if x >= width:
            break
    if x < width:
        img[:, x:] = img[:, x - 1:x] if x > 0 else 0
    return img


def _palette_b64(histogram: np.ndarray) -> str:
    """JPEG encode a histogram palette swatch as base64."""
    img = _render_palette(histogram)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/api/profile/{channel}")
async def get_profile_details(channel: str):
    """Return calibration diagnostics for a channel profile.

    Includes: metadata, per corner scores, reference template image, and
    either the reference edge map (logo mode) or the color palette swatch
    (color mode).
    """
    json_path = os.path.join(config.PROFILES_DIR, f"{channel}.json")
    edges_path = os.path.join(config.PROFILES_DIR, f"{channel}_edges.npy")
    hist_path = os.path.join(config.PROFILES_DIR, f"{channel}_hist.npy")
    template_path = os.path.join(config.PROFILES_DIR, f"{channel}_template.png")
    preview_path = os.path.join(config.PROFILES_DIR, f"{channel}_preview.png")

    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail=f"No profile for '{channel}'")

    with open(json_path) as f:
        metadata = json.load(f)

    mode = metadata.get("mode", "logo")

    # Reference template (grayscale for logo + combo, BGR for color)
    template_b64 = None
    if os.path.exists(template_path):
        flag = cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE
        template = cv2.imread(template_path, flag)
        if template is not None:
            _, buf = cv2.imencode(".jpg", template, [cv2.IMWRITE_JPEG_QUALITY, 85])
            template_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    edges_b64 = None
    palette_b64 = None
    if os.path.exists(edges_path) and mode in ("logo", "combo"):
        edge_profile = np.load(edges_path)
        edge_vis = (edge_profile * 255).clip(0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", edge_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        edges_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    if os.path.exists(hist_path) and mode in ("color", "combo"):
        palette_b64 = _palette_b64(np.load(hist_path))

    return {
        **metadata,
        "template_b64": template_b64,
        "edges_b64": edges_b64,
        "palette_b64": palette_b64,
    }


@app.get("/api/live-palette")
async def get_live_palette():
    """Return a swatch of the current frame's color palette (in the active profile's ROI).

    Used by the dashboard to show the live color palette next to the saved one.
    Falls back to the full frame if no profile/ROI is active.
    """
    from calibrate import compute_hs_histogram

    # Borrow the engine's open capture if it's running; otherwise open briefly.
    _refuse_if_held()
    cap = None
    try:
        if engine._cap is not None:
            frame = engine._cap.grab_frame()
        else:
            cap = FrameCapture()
            cap.open()
            # Capture cards return green sync artifact frames briefly after open.
            for _ in range(30):
                cap.grab_frame()
            frame = cap.grab_frame()

        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to grab frame")

        # If the frame is still a sync artifact (rare midstream glitch), skip it.
        from capture import is_sync_artifact_frame
        if is_sync_artifact_frame(frame):
            raise HTTPException(status_code=503, detail="Capture card returned a sync artifact frame; try again")

        # Determine the region the active profile uses for color (combo always
        # uses the full frame; color may use a rect; logo profiles fall back to
        # the full frame for visualization).
        det = engine._detector
        mode = getattr(det, "_mode", None)
        if det is not None and hasattr(det, "color_region"):
            region = det.color_region(frame)
        else:
            region = frame
        roi_rect = list(det._roi_rect) if (det is not None and det._roi_rect is not None and mode == "color") else None

        hist = compute_hs_histogram(region)
        match = None
        if det is not None and mode in ("color", "combo") and hasattr(det, "_ref_hist"):
            match = float(cv2.compareHist(det._ref_hist, hist, cv2.HISTCMP_CORREL))

        return {
            "palette_b64": _palette_b64(hist),
            "roi_rect": roi_rect,
            "match": match,
            "threshold": config.COLOR_MATCH_THRESHOLD,
            "mode": mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        # Surface the real capture error instead of a generic 500.
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cap is not None:
            cap.close()


TUNABLE_SETTINGS = {
    "DETECTION_MODE": str,
    "FIRST_SKIP_COUNT": int,
    "SUBSEQUENT_SKIP_COUNT": int,
    "SKIP_SETTLE_TIME": float,
    "LOGO_MATCH_THRESHOLD": float,
    "LOGO_RECOVERY_THRESHOLD": float,
    "DEBOUNCE_FRAMES": int,
    "RECOVERY_DEBOUNCE_FRAMES": int,
    "COLOR_MATCH_THRESHOLD": float,
    "COLOR_RECOVERY_THRESHOLD": float,
    "COLOR_DEBOUNCE_FRAMES": int,
    "COLOR_RECOVERY_DEBOUNCE_FRAMES": int,
    "MUTE_STEPS": int,
    "UNMUTE_STEPS": int,
    "PLAYBACK_MODE_OVERRIDE": str,
    "MAX_COMMERCIAL_DURATION": int,
    "MANUAL_OVERRIDE_COOLDOWN_SECONDS": int,
    "REENTRY_COOLDOWN_SECONDS": int,
    "REENTRY_DEBOUNCE_FRAMES": int,
    "CALIBRATION_DURATION": int,
    "BREAK_DECAY_STRENGTH": float,
    "EXIT_DECAY_MAX_FRAMES": int,
    "HYBRID_COMMERCIAL_ENTER_THRESHOLD": float,
    "HYBRID_PROGRAM_EXIT_THRESHOLD": float,
    "HYBRID_LEGACY_VISUAL_ENABLED": bool,
    "AUDIO_TRANSITION_ENABLED": bool,
    "AUDIO_FINGERPRINT_ENABLED": bool,
    "AD_OVERLAY_ENABLED": bool,
    "ML_CLASSIFIER_ENABLED": bool,
    "AUTOMATIC_MUTING_ENABLED": bool,
    "APPLE_TV_ID": str,
    "CAPTURE_DEVICE_NAME": str,
    "CAPTURE_DEVICE": type(config.CAPTURE_DEVICE),
    "AUDIO_DEVICE_NAME": str,
    "HYBRID_COMMERCIAL_ENTER_THRESHOLD": float,
    "HYBRID_PROGRAM_EXIT_THRESHOLD": float,
    "ML_COMMERCIAL_THRESHOLD": float,
    "ML_PROGRAM_THRESHOLD": float,
    "FINGERPRINT_MATCH_THRESHOLD": float,
    "FINGERPRINT_MIN_MATCH_DURATION": float,
    "AUDIO_TRANSITION_SCORE_THRESHOLD": float,
    "HYBRID_AUDIO_TRANSITION_WEIGHT": float,
    "HYBRID_FINGERPRINT_WEIGHT": float,
    "HYBRID_AD_OVERLAY_WEIGHT": float,
    "HYBRID_ML_WEIGHT": float,
}

SETTING_GROUPS = {
    "video": [
        "CAPTURE_DEVICE_NAME", "CAPTURE_DEVICE", "DETECTION_MODE", "HYBRID_LEGACY_VISUAL_ENABLED",
        "AD_OVERLAY_ENABLED",
        "LOGO_MATCH_THRESHOLD", "LOGO_RECOVERY_THRESHOLD", "DEBOUNCE_FRAMES",
        "RECOVERY_DEBOUNCE_FRAMES", "COLOR_MATCH_THRESHOLD", "COLOR_RECOVERY_THRESHOLD",
        "COLOR_DEBOUNCE_FRAMES", "COLOR_RECOVERY_DEBOUNCE_FRAMES",
    ],
    "audio": ["AUDIO_DEVICE_NAME", "AUDIO_TRANSITION_ENABLED", "AUDIO_TRANSITION_SCORE_THRESHOLD"],
    "fingerprinting": ["AUDIO_FINGERPRINT_ENABLED", "FINGERPRINT_MATCH_THRESHOLD", "FINGERPRINT_MIN_MATCH_DURATION"],
    "ml": ["ML_CLASSIFIER_ENABLED", "ML_COMMERCIAL_THRESHOLD", "ML_PROGRAM_THRESHOLD"],
    "fusion": [
        "HYBRID_COMMERCIAL_ENTER_THRESHOLD", "HYBRID_PROGRAM_EXIT_THRESHOLD",
        "HYBRID_AUDIO_TRANSITION_WEIGHT", "HYBRID_FINGERPRINT_WEIGHT",
        "HYBRID_AD_OVERLAY_WEIGHT", "HYBRID_ML_WEIGHT",
    ],
    "intervention": [
        "AUTOMATIC_MUTING_ENABLED", "APPLE_TV_ID", "MUTE_STEPS", "UNMUTE_STEPS",
        "FIRST_SKIP_COUNT", "SUBSEQUENT_SKIP_COUNT", "SKIP_SETTLE_TIME",
        "PLAYBACK_MODE_OVERRIDE", "MAX_COMMERCIAL_DURATION",
    ],
    "safety": [
        "MANUAL_OVERRIDE_COOLDOWN_SECONDS", "REENTRY_COOLDOWN_SECONDS",
        "REENTRY_DEBOUNCE_FRAMES", "BREAK_DECAY_STRENGTH", "EXIT_DECAY_MAX_FRAMES",
    ],
    "calibration": ["CALIBRATION_DURATION"],
}

SETTING_BOUNDS = {
    "LOGO_MATCH_THRESHOLD": (0.0, 1.0), "LOGO_RECOVERY_THRESHOLD": (0.0, 1.0),
    "COLOR_MATCH_THRESHOLD": (0.0, 1.0), "COLOR_RECOVERY_THRESHOLD": (0.0, 1.0),
    "HYBRID_COMMERCIAL_ENTER_THRESHOLD": (0.0, 1.0),
    "HYBRID_PROGRAM_EXIT_THRESHOLD": (0.0, 1.0),
    "ML_COMMERCIAL_THRESHOLD": (0.0, 1.0), "ML_PROGRAM_THRESHOLD": (0.0, 1.0),
    "FINGERPRINT_MATCH_THRESHOLD": (0.0, 1.0),
    "AUDIO_TRANSITION_SCORE_THRESHOLD": (0.0, 1.0),
    "BREAK_DECAY_STRENGTH": (0.0, 1.0),
    "DEBOUNCE_FRAMES": (1, 120), "RECOVERY_DEBOUNCE_FRAMES": (1, 120),
    "COLOR_DEBOUNCE_FRAMES": (1, 120), "COLOR_RECOVERY_DEBOUNCE_FRAMES": (1, 120),
    "MUTE_STEPS": (1, 30), "UNMUTE_STEPS": (1, 30),
    "MAX_COMMERCIAL_DURATION": (30, 1800), "CALIBRATION_DURATION": (5, 600),
    "FINGERPRINT_MIN_MATCH_DURATION": (0.5, 120.0),
    "HYBRID_AUDIO_TRANSITION_WEIGHT": (0.0, 10.0),
    "HYBRID_FINGERPRINT_WEIGHT": (0.0, 10.0),
    "HYBRID_AD_OVERLAY_WEIGHT": (0.0, 10.0), "HYBRID_ML_WEIGHT": (0.0, 10.0),
}

DEFAULT_SETTINGS = {key: getattr(config, key) for key in TUNABLE_SETTINGS}

# Named detection presets, bundles applied through _apply_settings. Each is a
# point on one axis: how eager the system is to call something a commercial.
# Conservative risks a little ad bleed; Aggressive risks clipping a bit of game.
PRESETS = {
    "conservative": {
        # Lowest match thresholds = reluctant to call a commercial (won't mute
        # the game on a brief logo dip); lower recovery = quick to return.
        "DEBOUNCE_FRAMES": 5, "LOGO_MATCH_THRESHOLD": 0.30,
        "LOGO_RECOVERY_THRESHOLD": 0.45, "RECOVERY_DEBOUNCE_FRAMES": 1,
        "COLOR_MATCH_THRESHOLD": 0.50, "COLOR_RECOVERY_THRESHOLD": 0.65,
        "COLOR_DEBOUNCE_FRAMES": 10, "REENTRY_COOLDOWN_SECONDS": 45,
        "BREAK_DECAY_STRENGTH": 0.0,
    },
    "balanced": {
        "DEBOUNCE_FRAMES": 3, "LOGO_MATCH_THRESHOLD": 0.35,
        "LOGO_RECOVERY_THRESHOLD": 0.50, "RECOVERY_DEBOUNCE_FRAMES": 1,
        "COLOR_MATCH_THRESHOLD": 0.55, "COLOR_RECOVERY_THRESHOLD": 0.70,
        "COLOR_DEBOUNCE_FRAMES": 8, "REENTRY_COOLDOWN_SECONDS": 30,
        "BREAK_DECAY_STRENGTH": 0.5,
    },
    "aggressive": {
        # Highest match thresholds = eager to call a commercial (mutes fast on
        # the first dip); higher recovery + full decay = commits to the break.
        "DEBOUNCE_FRAMES": 2, "LOGO_MATCH_THRESHOLD": 0.45,
        "LOGO_RECOVERY_THRESHOLD": 0.55, "RECOVERY_DEBOUNCE_FRAMES": 1,
        "COLOR_MATCH_THRESHOLD": 0.60, "COLOR_RECOVERY_THRESHOLD": 0.75,
        "COLOR_DEBOUNCE_FRAMES": 6, "REENTRY_COOLDOWN_SECONDS": 20,
        "BREAK_DECAY_STRENGTH": 1.0,
    },
}


class PresetRequest(BaseModel):
    name: str


@app.get("/api/settings")
async def get_settings():
    """Return current tunable settings."""
    snapshot = {key: getattr(config, key) for key in TUNABLE_SETTINGS}
    snapshot["_meta"] = {
        "active_preset": _active_preset(snapshot),
        "custom_overrides": _active_preset(snapshot) is None,
        "groups": SETTING_GROUPS,
        "defaults": DEFAULT_SETTINGS,
    }
    return snapshot


def _active_preset(snapshot: dict | None = None) -> str | None:
    values = snapshot or {key: getattr(config, key) for key in TUNABLE_SETTINGS}
    for name, bundle in PRESETS.items():
        if all(
            (abs(float(values[key]) - float(value)) < 1e-9)
            if isinstance(value, float)
            else values[key] == value
            for key, value in bundle.items()
        ):
            return name
    return None


def _write_runtime_settings() -> None:
    if "pytest" in sys.modules:
        return
    target = pathlib.Path(config.RUNTIME_SETTINGS_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: getattr(config, key) for key in TUNABLE_SETTINGS}
    with tempfile.NamedTemporaryFile(
        "w", dir=target.parent, prefix="settings-", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temp_name = handle.name
    os.replace(temp_name, target)


def _apply_settings(settings: dict, *, persist: bool = True) -> dict:
    """Validate + apply a {key: value} bundle onto config. Raises 400 on an
    unknown key. Returns the full current tunable settings snapshot."""
    proposed = {key: getattr(config, key) for key in TUNABLE_SETTINGS}
    for key, value in settings.items():
        if key not in TUNABLE_SETTINGS:
            raise HTTPException(status_code=400, detail=f"Unknown setting: {key}")
        cast = TUNABLE_SETTINGS[key]
        try:
            if cast is bool:
                if isinstance(value, bool):
                    converted = value
                elif str(value).lower() in {"1", "true", "yes", "on"}:
                    converted = True
                elif str(value).lower() in {"0", "false", "no", "off"}:
                    converted = False
                else:
                    raise ValueError("expected boolean")
            else:
                converted = cast(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid {key}: {exc}")
        if key == "DETECTION_MODE" and converted not in {"legacy", "hybrid"}:
            raise HTTPException(status_code=400, detail="DETECTION_MODE must be legacy or hybrid")
        if key == "PLAYBACK_MODE_OVERRIDE" and converted not in {"", "live", "delayed"}:
            raise HTTPException(status_code=400, detail="Playback override must be blank, live, or delayed")
        if key in SETTING_BOUNDS:
            minimum, maximum = SETTING_BOUNDS[key]
            if converted < minimum or converted > maximum:
                raise HTTPException(
                    status_code=400,
                    detail=f"{key} must be between {minimum} and {maximum}",
                )
        proposed[key] = converted
    if proposed["HYBRID_PROGRAM_EXIT_THRESHOLD"] >= proposed["HYBRID_COMMERCIAL_ENTER_THRESHOLD"]:
        raise HTTPException(status_code=400, detail="Hybrid exit threshold must be below entry threshold")
    if proposed["ML_PROGRAM_THRESHOLD"] >= proposed["ML_COMMERCIAL_THRESHOLD"]:
        raise HTTPException(status_code=400, detail="ML program threshold must be below commercial threshold")
    if proposed["LOGO_RECOVERY_THRESHOLD"] <= proposed["LOGO_MATCH_THRESHOLD"]:
        raise HTTPException(status_code=400, detail="Logo recovery threshold must exceed match threshold")
    if proposed["COLOR_RECOVERY_THRESHOLD"] <= proposed["COLOR_MATCH_THRESHOLD"]:
        raise HTTPException(status_code=400, detail="Color recovery threshold must exceed match threshold")
    for key, value in settings.items():
        converted = proposed[key]
        setattr(config, key, converted)
        logger.info("Setting %s = %s", key, converted)
    if settings and persist:
        _write_runtime_settings()
    return {key: getattr(config, key) for key in TUNABLE_SETTINGS}


@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update tunable settings at runtime."""
    return {"ok": True, **_apply_settings(settings)}


@app.post("/api/preset")
async def apply_preset(req: PresetRequest):
    """Apply a named detection preset (conservative/balanced/aggressive)."""
    bundle = PRESETS.get(req.name.lower())
    if bundle is None:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.name}")
    snapshot = _apply_settings(bundle)
    logger.info("Applied preset: %s", req.name)
    return {"ok": True, "preset": req.name.lower(), **snapshot}


class ResetSettingsRequest(BaseModel):
    group: str | None = None


@app.post("/api/settings/reset")
async def reset_settings(req: ResetSettingsRequest):
    if req.group is None or req.group == "all":
        keys = list(TUNABLE_SETTINGS)
    elif req.group in SETTING_GROUPS:
        keys = SETTING_GROUPS[req.group]
    else:
        raise HTTPException(status_code=400, detail="Unknown settings group")
    snapshot = _apply_settings({key: DEFAULT_SETTINGS[key] for key in keys})
    return {"ok": True, "group": req.group or "all", **snapshot}


@app.get("/api/settings/export")
async def export_settings():
    # Strict allowlist: credentials, session tokens and paths never leave the host.
    payload = {
        "schema_version": 1,
        "settings": {key: getattr(config, key) for key in TUNABLE_SETTINGS},
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": "attachment; filename=clicker-settings.json"},
    )


@app.post("/api/settings/import")
async def import_settings(payload: dict):
    if payload.get("schema_version") != 1 or not isinstance(payload.get("settings"), dict):
        raise HTTPException(status_code=400, detail="Unsupported settings export")
    snapshot = _apply_settings(payload["settings"])
    return {"ok": True, **snapshot}


def _load_persisted_settings() -> None:
    if "pytest" in sys.modules:
        return
    try:
        with open(config.RUNTIME_SETTINGS_FILE) as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("settings root must be an object")
        _apply_settings(values, persist=False)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning("Ignoring invalid persisted settings: %s", exc)


_load_persisted_settings()


@app.get("/api/history")
async def get_history():
    """Return time series confidence data and events for charting."""
    threshold = config.LOGO_MATCH_THRESHOLD
    mode = "logo"
    if engine._detector is not None:
        mode = getattr(engine._detector, "_mode", "logo")
        if mode == "color":
            threshold = config.COLOR_MATCH_THRESHOLD
    return {
        "history": engine._history,
        "events": engine._events,
        "threshold": threshold,
        "color_threshold": config.COLOR_MATCH_THRESHOLD,
        "logo_threshold": config.LOGO_MATCH_THRESHOLD,
        "mode": mode,
    }


@app.get("/api/capture-check")
async def capture_check():
    """Check if the capture card is connected and working."""
    import platform
    from capture import _find_capture_card_index

    diagnostics = await asyncio.to_thread(capture_device_diagnostics)
    result = {
        "found": False,
        "device_name": config.CAPTURE_DEVICE_NAME,
        "index": None,
        "frame": False,
        "resolution": None,
        "permission": diagnostics.get("permission"),
        "backend_available": diagnostics.get("backend_available"),
        "available_devices": [item["name"] for item in diagnostics.get("devices", [])],
    }
    if diagnostics.get("error"):
        result["error"] = diagnostics["error"]

    if platform.system() == "Darwin":
        idx = _find_capture_card_index()
        if idx is not None:
            result["found"] = True
            result["index"] = idx
            # Try grabbing a frame
            cap = None
            try:
                cap = FrameCapture(idx)
                cap.open()
                for _ in range(3):
                    cap.grab_frame()
                frame = cap.grab_frame()
                if frame is not None:
                    result["frame"] = True
                    result["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
            except Exception as e:
                result["error"] = str(e)
            finally:
                if cap:
                    cap.close()
    else:
        # Linux, just check if device path exists
        import os
        path = config.CAPTURE_DEVICE
        result["found"] = os.path.exists(path) if isinstance(path, str) else True
        result["index"] = path

    return result


@app.get("/api/snapshot")
async def get_snapshot():
    """Grab a single frame from the capture card. Works without the engine running."""
    _refuse_if_held()
    cap = None
    try:
        if engine._cap is not None:
            frame = engine._cap.grab_frame()
        else:
            cap = FrameCapture()
            cap.open()
            # Capture cards return green sync artifact frames briefly after open.
            for _ in range(30):
                cap.grab_frame()
            frame = cap.grab_frame()

        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to grab frame")

        h, w = frame.shape[:2]
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return {"frame_b64": b64, "width": w, "height": h}

    except HTTPException:
        raise
    except Exception as e:
        # Surface the real capture error (e.g. permission denied) instead of a
        # generic "Internal Server Error".
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cap is not None:
            cap.close()


@app.post("/api/calibrate/interactive")
async def calibrate_interactive(req: InteractiveCalibrateRequest):
    """Calibrate using a user selected ROI region."""
    from calibrate import calibrate_roi

    try:
        path = await asyncio.get_running_loop().run_in_executor(
            None, partial(calibrate_roi, req.channel, req.x, req.y, req.w, req.h,
                          duration=req.duration)
        )
        return {"ok": True, "channel": req.channel, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibrate/color")
async def calibrate_color_endpoint(req: ColorCalibrateRequest):
    """Build a color signature profile by averaging HSV histograms over ~30s of footage.

    If x/y/w/h are provided, samples that region; otherwise uses the full frame.
    """
    from calibrate import calibrate_color

    roi = None
    if req.x is not None and req.y is not None and req.w is not None and req.h is not None:
        roi = (req.x, req.y, req.w, req.h)

    try:
        path = await asyncio.get_running_loop().run_in_executor(
            None, partial(calibrate_color, req.channel, roi, duration=req.duration)
        )
        return {"ok": True, "channel": req.channel, "path": path, "roi": roi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibrate/combo")
async def calibrate_combo_endpoint(req: ComboCalibrateRequest):
    """Build a combo (logo + full frame color) profile.

    The rectangle is the logo region for edge/template matching; the color
    histogram is always sampled from the full frame.
    """
    from calibrate import calibrate_combo

    try:
        path = await asyncio.get_running_loop().run_in_executor(
            None, partial(calibrate_combo, req.channel, (req.x, req.y, req.w, req.h),
                          duration=req.duration)
        )
        return {"ok": True, "channel": req.channel, "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _active_roi_annotation(detector, frame_width: int, frame_height: int):
    """Resolve a safe diagnostic rectangle for logo, color, and combo profiles."""
    mode = getattr(detector, "_mode", "logo")
    roi_rect = getattr(detector, "_roi_rect", None)
    if mode in {"color", "combo"} and roi_rect is not None:
        x, y, width, height = (int(value) for value in roi_rect)
        x1 = max(0, min(frame_width - 1, x))
        y1 = max(0, min(frame_height - 1, y))
        x2 = max(x1, min(frame_width - 1, x + max(1, width) - 1))
        y2 = max(y1, min(frame_height - 1, y + max(1, height) - 1))
        label = "color" if mode == "color" else "combo logo"
        return (x1, y1, x2, y2), label

    if mode == "color":
        return (0, 0, frame_width - 1, frame_height - 1), "color full frame"

    roi_w = min(config.CORNER_ROI_WIDTH, frame_width)
    roi_h = min(config.CORNER_ROI_HEIGHT, frame_height)
    corner = getattr(detector, "corner", "unknown")
    corners = {
        "top_left": (0, 0, roi_w - 1, roi_h - 1),
        "top_right": (frame_width - roi_w, 0, frame_width - 1, roi_h - 1),
        "bottom_left": (0, frame_height - roi_h, roi_w - 1, frame_height - 1),
        "bottom_right": (
            frame_width - roi_w,
            frame_height - roi_h,
            frame_width - 1,
            frame_height - 1,
        ),
    }
    return corners.get(
        corner, (0, 0, frame_width - 1, frame_height - 1)
    ), str(corner)


@app.get("/api/frame")
async def get_full_frame():
    """Return the latest full frame with the active profile region highlighted.

    Used for the diagnostics view so you can see where on screen the
    detector is looking.
    """
    if engine._cap is None or engine._detector is None:
        raise HTTPException(status_code=409, detail="Engine not running")

    frame = engine._cap.grab_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to grab frame")

    # Draw the actual signal region. Color profiles can use a custom rectangle
    # or the full frame; combo profiles display their logo rectangle.
    h, w = frame.shape[:2]
    (x1, y1, x2, y2), region_label = _active_roi_annotation(
        engine._detector, w, h
    )
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Position text label inside the ROI to avoid clipping off screen
    text_x = x1 + 5
    text_y = y1 + 20 if y1 + 20 < h else max(15, y2 - 10)
    cv2.putText(
        annotated, f"ROI: {region_label}", (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )

    # Encode as JPEG (scaled down to save bandwidth)
    scale = 0.5
    small = cv2.resize(annotated, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {"frame_b64": b64, "corner": region_label}


def _lan_address() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))
        return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        sock.close()


@app.get("/api/network/info")
async def network_info():
    address = _lan_address()
    return {
        "bind_host": config.DASHBOARD_HOST,
        "port": config.DASHBOARD_PORT,
        "lan_enabled": config.DASHBOARD_HOST not in {"127.0.0.1", "::1", "localhost"},
        "access": "lan_direct",
        "address": address,
        "url": f"http://{address}:{config.DASHBOARD_PORT}/",
        "enable_hint": f"CLICKER_HOST=0.0.0.0 CLICKER_PORT={config.DASHBOARD_PORT} python3 server.py",
    }


@app.get("/api/network/qr")
async def network_qr():
    try:
        import qrcode
    except ImportError:
        raise HTTPException(status_code=503, detail="Install the qrcode dependency to generate a QR code")
    image = qrcode.make(f"http://{_lan_address()}:{config.DASHBOARD_PORT}/")
    output = BytesIO()
    image.save(output, format="PNG")
    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/setup")
async def get_setup_state():
    try:
        with open(config.SETUP_STATE_FILE) as handle:
            state = json.load(handle)
    except (FileNotFoundError, ValueError, OSError):
        state = {"complete": False, "step": 0}
    return state


@app.post("/api/setup")
async def save_setup_state(state: dict):
    allowed = {
        "complete": bool(state.get("complete", False)),
        "step": max(0, min(int(state.get("step", 0)), 6)),
    }
    path = pathlib.Path(config.SETUP_STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(allowed, handle, indent=2)
    return {"ok": True, **allowed}


# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the single page dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path) as f:
        return f.read()


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    """Serve the local setup, Legacy, and model-training guide."""
    html_path = os.path.join(os.path.dirname(__file__), "help.html")
    with open(html_path) as f:
        return f.read()


@app.get("/manifest.webmanifest")
async def manifest():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "manifest.webmanifest"),
        media_type="application/manifest+json",
    )


@app.get("/clicker-icon.svg")
async def clicker_icon():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "clicker-icon.svg"),
        media_type="image/svg+xml",
    )


@app.get("/clicker-icon-{size}.png")
async def clicker_png_icon(size: int):
    if size not in {192, 512}:
        raise HTTPException(status_code=404, detail="Icon not found")
    return FileResponse(
        os.path.join(os.path.dirname(__file__), f"clicker-icon-{size}.png"),
        media_type="image/png",
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    import uvicorn

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    uvicorn.run(
        app,
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
