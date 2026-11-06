"""Configuration for the live TV ad blocker."""

import os
import pathlib
import platform
import logging


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Apple TV
# ---------------------------------------------------------------------------
# Which playback device the engine controls. Only "appletv" ships today; add
# others by implementing PlaybackController in intervention.py (see README).
CONTROLLER_BACKEND = os.environ.get("CONTROLLER_BACKEND", "appletv")

# CHANGE THIS to your own Apple TV's identifier (used by the "appletv" backend).
# Find it by running `atvremote scan` (from the pyatv package) on the same
# network, it's the "Identifier" field. You can also override it without
# editing this file by setting the APPLE_TV_ID environment variable.
APPLE_TV_ID = os.environ.get("APPLE_TV_ID", "YOUR_APPLE_TV_ID")

# ---------------------------------------------------------------------------
# Dashboard and LAN access
# ---------------------------------------------------------------------------
# Localhost is deliberately the safe default. Set CLICKER_HOST=0.0.0.0 to
# expose the dashboard to the LAN; doing so automatically enables pairing.
DASHBOARD_HOST = os.environ.get("CLICKER_HOST", "127.0.0.1").strip()
DASHBOARD_PORT = int(os.environ.get("CLICKER_PORT", "8080"))
DASHBOARD_AUTH_ENABLED = _env_bool(
    "CLICKER_AUTH_ENABLED", DASHBOARD_HOST not in {"127.0.0.1", "::1", "localhost"}
)
DASHBOARD_SESSION_IDLE_SECONDS = int(os.environ.get("CLICKER_SESSION_IDLE_SECONDS", "1800"))
DASHBOARD_SESSION_ABSOLUTE_SECONDS = int(os.environ.get("CLICKER_SESSION_ABSOLUTE_SECONDS", "28800"))
DASHBOARD_AUTH_DB_PATH = os.environ.get(
    "CLICKER_AUTH_DB_PATH",
    str(pathlib.Path(__file__).parent / "data" / "dashboard_auth.sqlite3"),
)
DASHBOARD_PAIRING_CODE_TTL_SECONDS = 600

# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------
MUTE_STEPS = 10                # volume_down presses to fully mute
UNMUTE_STEPS = 7               # volume_up presses to restore normal level
VOLUME_COMMAND_DELAY = 0.1     # seconds between consecutive volume commands
AUTOMATIC_MUTING_ENABLED = True

# ---------------------------------------------------------------------------
# Skip control
# ---------------------------------------------------------------------------
FIRST_SKIP_COUNT = 8           # skip_forward presses on first jump (120s)
SUBSEQUENT_SKIP_COUNT = 1      # skip_forward presses after first (15s)
SKIP_SETTLE_TIME = 0.33        # seconds to wait after skip for frame to settle
SEEK_SETTLE_TIME = 1.0         # seconds to wait after seek (right+select) for frame to settle
SKIP_COMMAND_DELAY = 0.1       # seconds between consecutive skip_forward presses

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# Legacy remains the default and follows the exact pre-hybrid path. Hybrid is
# opt in while its optional detectors are added and validated.
DETECTION_MODE = os.environ.get("DETECTION_MODE", "legacy").strip().lower()

DEBOUNCE_FRAMES = 3            # consecutive frames to confirm WATCHING→COMMERCIAL
RECOVERY_DEBOUNCE_FRAMES = 1  # consecutive frames to confirm COMMERCIAL→WATCHING (faster exit)

# Anti flapping: after exiting a commercial, reentering requires sustained
# evidence for a while. A wrong exit corrects itself a little late (cheap);
# a mute/unmute waffle on the TV is the expensive failure.
REENTRY_COOLDOWN_SECONDS = 30  # window after commercial_end with extra entry friction
REENTRY_DEBOUNCE_FRAMES = 30   # consecutive frames (~3s) to reenter during that window
FRAME_CHECK_INTERVAL = 0.1     # seconds between frame captures in main loop
HYBRID_FRAME_CHECK_INTERVAL = 0.5  # 2 FPS, lower CPU on older Intel hardware
LOGO_MATCH_THRESHOLD = 0.35    # confidence threshold for logo detection (0-1)
LOGO_RECOVERY_THRESHOLD = 0.50 # higher bar required to LEAVE commercial state (hysteresis)
MAX_COMMERCIAL_DURATION = 300  # seconds, safety timeout if detection seems wrong
MANUAL_OVERRIDE_COOLDOWN_SECONDS = 60

# Hybrid fusion. These thresholds operate on normalized fusion evidence, not
# on logo/color raw scores. The gap is intentional hysteresis: evidence in the
# middle produces UNKNOWN and the engine holds its prior safe state.
HYBRID_COMMERCIAL_ENTER_THRESHOLD = 0.70
HYBRID_PROGRAM_EXIT_THRESHOLD = 0.40
HYBRID_LEGACY_VISUAL_ENABLED = True
HYBRID_LEGACY_VISUAL_WEIGHT = 1.0
HYBRID_LEGACY_COMMERCIAL_EVIDENCE = 0.85
HYBRID_LEGACY_PROGRAM_EVIDENCE = 0.10
# Logo absence alone stays below the default entry threshold. Color/combo
# profiles can still provide sufficient positive visual evidence.
HYBRID_LOGO_ABSENCE_EVIDENCE = 0.65
HYBRID_MIN_COMMERCIAL_SOURCES = 2

# ---------------------------------------------------------------------------
# On-screen ad labels and countdowns (hybrid mode only)
# ---------------------------------------------------------------------------
# OCR only treats an explicit ad label as evidence. A timer alone is reported
# diagnostically but ignored because sports/game clocks are common.
AD_OVERLAY_ENABLED = _env_bool("AD_OVERLAY_ENABLED", True)
TESSERACT_BINARY = os.environ.get("TESSERACT_BINARY", "tesseract")
AD_OVERLAY_SCAN_INTERVAL = 1.0
AD_OVERLAY_REQUIRED_CONFIRMATIONS = 2
AD_OVERLAY_HOLD_SECONDS = 2.0
AD_OVERLAY_MIN_OCR_CONFIDENCE = 0.35
AD_OVERLAY_OCR_TIMEOUT = 2.0
AD_OVERLAY_REGIONS = (
    (0.0, 0.0, 1.0, 0.32),
    (0.0, 0.62, 1.0, 0.38),
)
HYBRID_AD_OVERLAY_WEIGHT = 2.0
HYBRID_AD_OVERLAY_EVIDENCE = 0.95

# ---------------------------------------------------------------------------
# Optional audio capture and transition evidence (hybrid mode only)
# ---------------------------------------------------------------------------
AUDIO_TRANSITION_ENABLED = _env_bool("AUDIO_TRANSITION_ENABLED", False)
FFMPEG_BINARY = os.environ.get("FFMPEG_BINARY", "ffmpeg")
# Must match an AVFoundation audio input name (substring, case insensitive).
# It is intentionally blank by default: the application probes and reports
# devices rather than assuming the HDMI capture card exposes usable audio.
AUDIO_DEVICE_NAME = os.environ.get("AUDIO_DEVICE_NAME", "")
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 0.25
AUDIO_WINDOW_SECONDS = 2.0
AUDIO_ANALYSIS_INTERVAL = 0.5
AUDIO_RETENTION_SECONDS = 180
AUDIO_SILENCE_DBFS = -45.0
AUDIO_MAX_FEATURE_AGE = 3.0
AUDIO_LOUDNESS_CHANGE_DB = 8.0
AUDIO_SPECTRAL_FLUX_THRESHOLD = 0.015
AUDIO_SILENCE_RATIO_THRESHOLD = 0.60
AUDIO_TRANSITION_SCORE_THRESHOLD = 0.65
HYBRID_AUDIO_TRANSITION_WEIGHT = 1.0
HYBRID_AUDIO_TRANSITION_EVIDENCE = 0.85

# ---------------------------------------------------------------------------
# Optional local audio fingerprints (hybrid mode only)
# ---------------------------------------------------------------------------
AUDIO_FINGERPRINT_ENABLED = _env_bool("AUDIO_FINGERPRINT_ENABLED", False)
FINGERPRINT_DB_PATH = os.environ.get(
    "FINGERPRINT_DB_PATH",
    str(pathlib.Path(__file__).parent / "data" / "fingerprints.sqlite3"),
)
FINGERPRINT_QUERY_SECONDS = 6.0
FINGERPRINT_MATCH_INTERVAL = 2.0
FINGERPRINT_MATCH_THRESHOLD = 0.55
FINGERPRINT_MIN_MATCH_DURATION = 2.0
FINGERPRINT_MIN_UNIQUE_TOKENS = 4
FINGERPRINT_MAX_RESULT_AGE = 5.0
HYBRID_FINGERPRINT_WEIGHT = 3.0
HYBRID_FINGERPRINT_EVIDENCE = 0.98

# ---------------------------------------------------------------------------
# Optional lightweight ML and durable labeled features
# ---------------------------------------------------------------------------
ML_CLASSIFIER_ENABLED = _env_bool("ML_CLASSIFIER_ENABLED", False)
ML_MODEL_PATH = os.environ.get(
    "ML_MODEL_PATH", str(pathlib.Path(__file__).parent / "data" / "commercial_model.joblib")
)
ML_DATASET_PATH = os.environ.get(
    "ML_DATASET_PATH", str(pathlib.Path(__file__).parent / "data" / "features.sqlite3")
)
ML_EXPORT_PATH = os.environ.get(
    "ML_EXPORT_PATH", str(pathlib.Path(__file__).parent / "data" / "features_export.csv")
)
CORRECTIONS_DB_PATH = os.environ.get(
    "CORRECTIONS_DB_PATH", str(pathlib.Path(__file__).parent / "data" / "corrections.sqlite3")
)
ML_COMMERCIAL_THRESHOLD = 0.60
ML_PROGRAM_THRESHOLD = 0.40
ML_INFERENCE_INTERVAL = 0.5
HYBRID_ML_WEIGHT = 1.5

# ---------------------------------------------------------------------------
# Color profile detection (alternative to logo template matching)
# ---------------------------------------------------------------------------
COLOR_MATCH_THRESHOLD = 0.55   # HSV histogram correlation: 1.0=identical, lower=different
COLOR_RECOVERY_THRESHOLD = 0.70 # higher bar required to LEAVE commercial state (hysteresis)
COLOR_DEBOUNCE_FRAMES = 8      # consecutive frames to confirm WATCHING→COMMERCIAL
COLOR_RECOVERY_DEBOUNCE_FRAMES = 2  # consecutive frames to confirm COMMERCIAL→WATCHING
COLOR_HIST_BINS_H = 50         # hue bins
COLOR_HIST_BINS_S = 60         # saturation bins
MIN_COLOR_REBUILD_FRAMES = 10  # min marked content frames before rebuilding the color profile

# ---------------------------------------------------------------------------
# Live vs delayed
# ---------------------------------------------------------------------------
LIVE_THRESHOLD = 0.90          # position/total_time above this = "live"
PLAYBACK_MODE_OVERRIDE = ""    # "", "live", or "delayed", empty = auto detect

# ---------------------------------------------------------------------------
# Capture device
# ---------------------------------------------------------------------------
if platform.system() == "Darwin":
    CAPTURE_DEVICE = 1         # macOS: use device index (probe if wrong)
else:
    CAPTURE_DEVICE = "/dev/video0"  # Linux: V4L2 device path

CAPTURE_DEVICE_NAME = "USB3.0 Video"  # AVFoundation name for device lookup on macOS
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CALIBRATION_DURATION = 60      # seconds to capture frames during calibration (runtime tunable)
CALIBRATION_FPS = 2            # frames per second during calibration
CORNER_ROI_WIDTH = 200         # pixels, width of each corner ROI
CORNER_ROI_HEIGHT = 150        # pixels, height of each corner ROI
EDGE_PERSISTENCE_THRESHOLD = 0.3  # fraction of frames an edge must appear in to be "persistent"
CANNY_LOW = 50                 # Canny edge detector low threshold
CANNY_HIGH = 150               # Canny edge detector high threshold

# ---------------------------------------------------------------------------
# Break length decay (mute/live mode), resist believing a break ended at an
# implausible time. Breaks cluster near these lengths; an exit far from any of
# them is more likely a false logo blip than a real return.
# ---------------------------------------------------------------------------
BREAK_BOUNDARIES = [30, 60, 90, 120]  # typical break lengths, seconds
BREAK_BOUNDARY_WINDOW = 10            # +/- seconds around a boundary = "plausible"
EXIT_DECAY_MAX_FRAMES = 25            # cap on added exit debounce (~2.5s @ 10fps)
BREAK_DECAY_STRENGTH = 0.5            # 0=off .. 1=full; set by preset

PROFILES_DIR = str(pathlib.Path(__file__).parent / "profiles")

# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------
# Records the last actively running channel so the engine can auto resume after
# a service restart / reboot. Cleared on an explicit Stop.
LAST_SESSION_FILE = str(pathlib.Path(__file__).parent / ".last_session.json")
# Crash-recovery marker: exists only while this process believes it muted the TV.
MUTED_STATE_FILE = str(pathlib.Path(__file__).parent / ".muted_by_clicker.json")
RUNTIME_SETTINGS_FILE = str(pathlib.Path(__file__).parent / "data" / "settings.json")
SETUP_STATE_FILE = str(pathlib.Path(__file__).parent / "data" / "setup.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
