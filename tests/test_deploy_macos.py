import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "deploy" / "setup-macos.sh"
REQUIREMENTS = Path(__file__).parents[1] / "requirements.txt"


def test_macos_installer_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_macos_installer_help_is_safe_and_documents_network_modes():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--lan" in result.stdout
    assert "--local-only" in result.stdout
    assert "--no-start" in result.stdout


def test_macos_installer_uses_user_launch_agent_and_preserves_runtime_state():
    script = SCRIPT.read_text()
    assert "Do not run this installer with sudo" in script
    assert "com.theclicker.app.plist" in script
    assert 'launchctl bootstrap "gui/$USER_ID"' in script
    assert 'if [ ! -f "$INSTALL_DIR/credentials.json" ]' in script
    assert 'if [ ! -e "$INSTALL_DIR/profiles/$profile_name" ]' in script
    assert '"CLICKER_HOST": bind_host' in script
    assert '"FFMPEG_BINARY": ffmpeg_bin' in script
    assert "dashboard.html help.html" in script
    assert 'ffmpeg tesseract' in script
    assert "audio.py ad_overlay.py" in script


def test_appletv_srp_dependency_is_pinned_below_breaking_release():
    requirements = REQUIREMENTS.read_text()
    assert "pyatv>=0.16,<0.19" in requirements
    assert "srptools>=1.0.1,<2" in requirements
