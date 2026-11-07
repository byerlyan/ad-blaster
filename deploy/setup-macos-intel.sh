#!/bin/bash
# Intel macOS installer for The Clicker using MacPorts instead of Homebrew.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash deploy/setup-macos-intel.sh [options]

Intel-only, Homebrew-free installer. Runtime packages come from MacPorts and
the app runs as the logged-in user through a LaunchAgent.

Prerequisite:
  Install the MacPorts package matching your macOS version from:
  https://www.macports.org/install.php

Options:
  --lan          Listen on the home LAN (0.0.0.0); this is the default.
  --local-only   Listen only on 127.0.0.1.
  --no-start     Install/update files and LaunchAgent without starting it.
  --help         Show this help.

Environment overrides:
  CLICKER_INSTALL_DIR       App directory (default: ~/Library/Application Support/The Clicker)
  CLICKER_MACPORTS_PREFIX   MacPorts prefix (default: /opt/local)
  CLICKER_PORT              Dashboard port (default: 8080)

Optional ML training after installation:
  cd "$HOME/Library/Application Support/The Clicker"
  ./venv-macports/bin/python3 -m pip install -r requirements-ml.txt
  ./venv-macports/bin/python3 scripts/export_feature_dataset.py --output data/features.csv
EOF
}

for argument in "$@"; do
    case "$argument" in
        --help|-h) usage; exit 0 ;;
    esac
done

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This installer is for Intel macOS." >&2
    exit 1
fi
if [ "$(uname -m)" != "x86_64" ]; then
    echo "This installer requires an Intel Mac (x86_64)." >&2
    echo "Use deploy/setup-macos.sh on Apple silicon." >&2
    exit 1
fi
if [ "$(id -u)" -eq 0 ]; then
    echo "Do not run this installer with sudo; it requests sudo only for MacPorts packages." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CLICKER_PACKAGE_MANAGER="macports"
exec /bin/bash "$SCRIPT_DIR/setup-macos.sh" "$@"
