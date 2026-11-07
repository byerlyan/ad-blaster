#!/bin/bash
# Install or update The Clicker as a per-user macOS LaunchAgent.
#
# The service starts when this user logs in. A LaunchAgent (not a system
# LaunchDaemon) is intentional: AVFoundation camera/capture and Local Network
# permissions belong to the interactive user on macOS.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash deploy/setup-macos.sh [options]

Options:
  --lan          Listen on the home LAN (0.0.0.0); pairing auth is automatic.
  --local-only   Listen only on 127.0.0.1 (safe default).
  --no-start     Install/update files and LaunchAgent without starting it.
  --help         Show this help.

Environment overrides:
  CLICKER_INSTALL_DIR       App directory (default: ~/Library/Application Support/The Clicker)
  CLICKER_PYTHON_FORMULA    Homebrew Python formula (default: python@3.12)
  CLICKER_PORT              Dashboard port (default: 8080)
EOF
}

BIND_MODE="ask"
START_AGENT="true"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --lan) BIND_MODE="lan" ;;
        --local-only) BIND_MODE="local" ;;
        --no-start) START_AGENT="false" ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This installer is for macOS. Use deploy/setup.sh on Debian/Linux." >&2
    exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
    echo "Do not run this installer with sudo; it creates a LaunchAgent for the logged-in user." >&2
    exit 1
fi

echo "=== The Clicker macOS Setup ==="

SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_HOME="${HOME:?HOME is required}"
INSTALL_DIR="${CLICKER_INSTALL_DIR:-$USER_HOME/Library/Application Support/The Clicker}"
LAUNCH_AGENTS_DIR="$USER_HOME/Library/LaunchAgents"
LOG_DIR="$USER_HOME/Library/Logs/The Clicker"
PLIST_PATH="$LAUNCH_AGENTS_DIR/com.theclicker.app.plist"
AGENT_LABEL="com.theclicker.app"
PYTHON_FORMULA="${CLICKER_PYTHON_FORMULA:-python@3.12}"
CLICKER_PORT_VALUE="${CLICKER_PORT:-8080}"
USER_ID="$(id -u)"
PACKAGE_MANAGER="${CLICKER_PACKAGE_MANAGER:-homebrew}"

case "$CLICKER_PORT_VALUE" in
    ''|*[!0-9]*) echo "CLICKER_PORT must be a number." >&2; exit 2 ;;
esac
if [ "$CLICKER_PORT_VALUE" -lt 1 ] || [ "$CLICKER_PORT_VALUE" -gt 65535 ]; then
    echo "CLICKER_PORT must be between 1 and 65535." >&2
    exit 2
fi

RUNTIME_NAME=""
RUNTIME_PREFIX=""
REQUIREMENTS_NAME="requirements.txt"
VENV_NAME="venv"
VENV_SYSTEM_SITE="false"

if [ "$PACKAGE_MANAGER" = "macports" ]; then
    if [ "$(uname -m)" != "x86_64" ]; then
        echo "The MacPorts installer is intended for Intel Macs (x86_64)." >&2
        exit 1
    fi
    RUNTIME_NAME="MacPorts"
    RUNTIME_PREFIX="${CLICKER_MACPORTS_PREFIX:-/opt/local}"
    PORT_BIN="$(command -v port 2>/dev/null || true)"
    if [ -z "$PORT_BIN" ] && [ -x "$RUNTIME_PREFIX/bin/port" ]; then
        PORT_BIN="$RUNTIME_PREFIX/bin/port"
    fi
    if [ -z "$PORT_BIN" ]; then
        echo "MacPorts is required for the Intel installer." >&2
        echo "Install the matching macOS package from https://www.macports.org/install.php and rerun." >&2
        exit 1
    fi
    echo "[1/6] Installing MacPorts Intel runtime packages..."
    /usr/bin/sudo "$PORT_BIN" install \
        python312 py312-pip py312-virtualenv py312-opencv4 py312-pyobjc \
        ffmpeg tesseract
    PYTHON_BIN="$RUNTIME_PREFIX/bin/python3.12"
    REQUIREMENTS_NAME="requirements-intel-macos.txt"
    VENV_NAME="venv-macports"
    VENV_SYSTEM_SITE="true"
elif [ "$PACKAGE_MANAGER" = "homebrew" ]; then
    RUNTIME_NAME="Homebrew"
    BREW_BIN="$(command -v brew 2>/dev/null || true)"
    if [ -z "$BREW_BIN" ]; then
        for brew_candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
            if [ -x "$brew_candidate" ]; then
                BREW_BIN="$brew_candidate"
                break
            fi
        done
    fi
    if [ -z "$BREW_BIN" ]; then
        echo "Homebrew is required. Install it from https://brew.sh, then rerun this script." >&2
        exit 1
    fi
    RUNTIME_PREFIX="$("$BREW_BIN" --prefix)"

    echo "[1/6] Installing Homebrew runtime packages..."
    for formula in "$PYTHON_FORMULA" ffmpeg tesseract; do
        if ! "$BREW_BIN" list --versions "$formula" >/dev/null 2>&1; then
            "$BREW_BIN" install "$formula"
        else
            echo "  $formula is already installed."
        fi
    done

    PYTHON_PREFIX="$("$BREW_BIN" --prefix "$PYTHON_FORMULA")"
    PYTHON_BIN="$PYTHON_PREFIX/bin/python3.12"
    if [ ! -x "$PYTHON_BIN" ]; then
        PYTHON_BIN="$PYTHON_PREFIX/bin/python3"
    fi
    if [ ! -x "$PYTHON_BIN" ]; then
        PYTHON_BIN=""
        for python_candidate in "$PYTHON_PREFIX"/bin/python3.*; do
            case "$python_candidate" in
                *-config) continue ;;
            esac
            if [ -x "$python_candidate" ]; then
                PYTHON_BIN="$python_candidate"
                break
            fi
        done
    fi
else
    echo "Unsupported package manager: $PACKAGE_MANAGER" >&2
    exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Could not find $RUNTIME_NAME Python at $PYTHON_BIN." >&2
    exit 1
fi
FFMPEG_BIN="$RUNTIME_PREFIX/bin/ffmpeg"
TESSERACT_BIN="$RUNTIME_PREFIX/bin/tesseract"
if [ ! -x "$FFMPEG_BIN" ]; then
    echo "Could not find $RUNTIME_NAME FFmpeg at $FFMPEG_BIN." >&2
    exit 1
fi
if [ ! -x "$TESSERACT_BIN" ]; then
    echo "Could not find $RUNTIME_NAME Tesseract at $TESSERACT_BIN." >&2
    exit 1
fi

echo "[2/6] Creating the application directory..."
mkdir -p "$INSTALL_DIR" "$INSTALL_DIR/profiles" "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/scripts" "$INSTALL_DIR/docs" "$LAUNCH_AGENTS_DIR" "$LOG_DIR"

echo "[3/6] Copying application files (runtime data is preserved)..."
APP_FILES=(
    server.py engine.py capture.py detector.py detection_pipeline.py audio.py ad_overlay.py
    fingerprint.py ml.py evaluation.py calibrate.py config.py intervention.py
    pair.py corrections.py dashboard_auth.py device_setup.py dashboard.html help.html
    manifest.webmanifest clicker-icon.svg clicker-icon-192.png clicker-icon-512.png
    requirements.txt requirements-intel-macos.txt requirements-dev.txt
    requirements-ml.txt requirements-browser.txt
)
if [ "$SOURCE_DIR" != "$INSTALL_DIR" ]; then
    for file_name in "${APP_FILES[@]}"; do
        if [ -f "$SOURCE_DIR/$file_name" ]; then
            cp "$SOURCE_DIR/$file_name" "$INSTALL_DIR/$file_name"
        fi
    done
    if [ -d "$SOURCE_DIR/scripts" ]; then
        for script_file in "$SOURCE_DIR"/scripts/*.py; do
            [ -e "$script_file" ] || continue
            cp "$script_file" "$INSTALL_DIR/scripts/"
        done
    fi
    if [ -f "$SOURCE_DIR/docs/hybrid-detection.md" ]; then
        cp "$SOURCE_DIR/docs/hybrid-detection.md" "$INSTALL_DIR/docs/"
    fi
    # Seed machine-specific state only when the installed copy has none.
    if [ ! -f "$INSTALL_DIR/credentials.json" ] && [ -f "$SOURCE_DIR/credentials.json" ]; then
        cp "$SOURCE_DIR/credentials.json" "$INSTALL_DIR/credentials.json"
        chmod 600 "$INSTALL_DIR/credentials.json"
        echo "  Copied existing Apple TV credentials."
    fi
    if [ -d "$SOURCE_DIR/profiles" ]; then
        for profile_file in "$SOURCE_DIR"/profiles/*; do
            [ -f "$profile_file" ] || continue
            profile_name="$(basename "$profile_file")"
            if [ ! -e "$INSTALL_DIR/profiles/$profile_name" ]; then
                cp "$profile_file" "$INSTALL_DIR/profiles/$profile_name"
            fi
        done
    fi
fi

echo "[4/6] Creating the Python environment..."
VENV_DIR="$INSTALL_DIR/$VENV_NAME"
if [ ! -x "$VENV_DIR/bin/python3" ]; then
    if [ "$VENV_SYSTEM_SITE" = "true" ]; then
        "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
fi
"$VENV_DIR/bin/python3" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python3" -m pip install --quiet -r "$INSTALL_DIR/$REQUIREMENTS_NAME"
if [ "$PACKAGE_MANAGER" = "macports" ]; then
    if ! "$VENV_DIR/bin/python3" -c 'import AVFoundation, cv2, numpy; assert cv2.videoio_registry.hasBackend(cv2.CAP_AVFOUNDATION)' >/dev/null 2>&1; then
        echo "MacPorts Python cannot use the OpenCV AVFoundation capture backend." >&2
        echo "Verify that py312-pyobjc and py312-opencv4 are active, then rerun." >&2
        exit 1
    fi
fi

if [ "$BIND_MODE" = "ask" ]; then
    BIND_MODE="local"
    if [ -t 0 ]; then
        read -r -p "Expose the dashboard to paired devices on your home LAN? [y/N] " ENABLE_LAN
        if [[ "$ENABLE_LAN" =~ ^[Yy]$ ]]; then
            BIND_MODE="lan"
        fi
    fi
fi
if [ "$BIND_MODE" = "lan" ]; then
    BIND_HOST="0.0.0.0"
else
    BIND_HOST="127.0.0.1"
fi

echo "[5/6] Writing the LaunchAgent..."
"$VENV_DIR/bin/python3" - "$PLIST_PATH" "$AGENT_LABEL" "$INSTALL_DIR" \
    "$VENV_DIR/bin/python3" "$LOG_DIR" "$BIND_HOST" "$CLICKER_PORT_VALUE" \
    "$FFMPEG_BIN" "$TESSERACT_BIN" "$RUNTIME_PREFIX" <<'PY'
import plistlib
import sys

(
    plist_path,
    label,
    install_dir,
    python_bin,
    log_dir,
    bind_host,
    port,
    ffmpeg_bin,
    tesseract_bin,
    runtime_prefix,
) = sys.argv[1:]

payload = {
    "Label": label,
    "ProgramArguments": [python_bin, "server.py"],
    "WorkingDirectory": install_dir,
    "RunAtLoad": True,
    "KeepAlive": {"SuccessfulExit": False},
    "ThrottleInterval": 5,
    "ProcessType": "Background",
    "EnvironmentVariables": {
        "PYTHONUNBUFFERED": "1",
        "CLICKER_HOST": bind_host,
        "CLICKER_PORT": port,
        "FFMPEG_BINARY": ffmpeg_bin,
        "TESSERACT_BINARY": tesseract_bin,
        "PATH": f"{runtime_prefix}/bin:{runtime_prefix}/sbin:/usr/bin:/bin:/usr/sbin:/sbin",
    },
    "StandardOutPath": f"{log_dir}/clicker.log",
    "StandardErrorPath": f"{log_dir}/clicker-error.log",
}
with open(plist_path, "wb") as handle:
    plistlib.dump(payload, handle, sort_keys=False)
PY
chmod 600 "$PLIST_PATH"
plutil -lint "$PLIST_PATH" >/dev/null

echo "[6/6] Installing the LaunchAgent..."
launchctl bootout "gui/$USER_ID" "$PLIST_PATH" >/dev/null 2>&1 || true
if [ "$START_AGENT" = "true" ]; then
    if ! launchctl bootstrap "gui/$USER_ID" "$PLIST_PATH"; then
        echo "launchctl bootstrap failed. Make sure this script runs in the logged-in user's Terminal." >&2
        exit 1
    fi
    launchctl enable "gui/$USER_ID/$AGENT_LABEL"
    launchctl kickstart -k "gui/$USER_ID/$AGENT_LABEL"

    DASHBOARD_READY="false"
    for _attempt in 1 2 3 4 5 6 7 8 9 10; do
        if /usr/bin/curl --silent --fail --max-time 1 \
            "http://127.0.0.1:$CLICKER_PORT_VALUE/api/status" >/dev/null 2>&1; then
            DASHBOARD_READY="true"
            break
        fi
        sleep 1
    done
    if [ "$DASHBOARD_READY" != "true" ]; then
        echo "WARNING: the LaunchAgent started but the dashboard health check failed." >&2
        echo "Review: $LOG_DIR/clicker-error.log" >&2
    fi
else
    echo "  LaunchAgent written but not started (--no-start)."
fi

LAN_ADDRESS=""
if [ "$BIND_MODE" = "lan" ]; then
    LAN_ADDRESS="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
fi

echo ""
echo "=== macOS Setup Complete ==="
echo ""
echo "Install directory: $INSTALL_DIR"
echo "Local dashboard:  http://127.0.0.1:$CLICKER_PORT_VALUE"
if [ -n "$LAN_ADDRESS" ]; then
    echo "LAN dashboard:    http://$LAN_ADDRESS:$CLICKER_PORT_VALUE"
    echo "Pairing code:     tail -f \"$LOG_DIR/clicker-error.log\""
elif [ "$BIND_MODE" = "lan" ]; then
    echo "LAN mode is enabled; find this Mac's address in System Settings → Network."
fi
echo ""
echo "macOS may ask for Camera and Local Network access on first use. Allow the"
echo "$RUNTIME_NAME Python process, then use Calibration → Check Capture Card."
echo ""
if [ ! -f "$INSTALL_DIR/credentials.json" ]; then
    echo "Pair the Apple TV in Settings → Devices, or run:"
    echo "  cd \"$INSTALL_DIR\" && \"$VENV_DIR/bin/python3\" pair.py"
    echo ""
fi
echo "Manage the service:"
echo "  Status:  launchctl print gui/$USER_ID/$AGENT_LABEL"
echo "  Restart: launchctl kickstart -k gui/$USER_ID/$AGENT_LABEL"
echo "  Stop:    launchctl bootout gui/$USER_ID \"$PLIST_PATH\""
echo "  Start:   launchctl bootstrap gui/$USER_ID \"$PLIST_PATH\""
echo "  Logs:    tail -f \"$LOG_DIR/clicker.log\" \"$LOG_DIR/clicker-error.log\""
