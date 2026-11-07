import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "deploy" / "setup-macos.sh"
INTEL_SCRIPT = Path(__file__).parents[1] / "deploy" / "setup-macos-intel.sh"
REQUIREMENTS = Path(__file__).parents[1] / "requirements.txt"
INTEL_REQUIREMENTS = Path(__file__).parents[1] / "requirements-intel-macos.txt"
ML_REQUIREMENTS = Path(__file__).parents[1] / "requirements-ml.txt"


def test_macos_installer_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    subprocess.run(["bash", "-n", str(INTEL_SCRIPT)], check=True)


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
    assert "dashboard_auth.py" not in script
    assert 'ffmpeg tesseract' in script
    assert "tesseract-eng" in script
    assert "audio.py ad_overlay.py" in script
    assert "ml.py ml_artifacts.py" in script
    assert "scripts/train_commercial_model.py --dataset data/features.csv" in script
    assert '"TESSERACT_BINARY": tesseract_bin' in script
    assert '"TESSDATA_DIR": tessdata_dir' in script
    assert "eng.traineddata" in script


def test_intel_installer_is_homebrew_free_and_uses_native_macports_packages():
    result = subprocess.run(
        ["bash", str(INTEL_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    wrapper = INTEL_SCRIPT.read_text()
    common = SCRIPT.read_text()
    requirements = INTEL_REQUIREMENTS.read_text()
    assert "Intel-only, Homebrew-free installer" in result.stdout
    assert "CLICKER_MACPORTS_PREFIX" in result.stdout
    assert '"$(uname -m)" != "x86_64"' in wrapper
    assert 'CLICKER_PACKAGE_MANAGER="macports"' in wrapper
    assert 'CLICKER_TESSERACT_LANGUAGE_PORT="tesseract-eng"' in wrapper
    for package in (
        "python312",
        "py312-pip",
        "py312-virtualenv",
        "py312-opencv4",
        "py312-pyobjc",
        "ffmpeg",
        "tesseract",
        "tesseract-eng",
    ):
        assert package in common
    assert "--system-site-packages" in common
    assert 'VENV_NAME="venv-macports"' in common
    assert "ml.py ml_artifacts.py" in common
    assert "venv-macports/bin/python3 -m pip install -r requirements-ml.txt" in result.stdout
    assert "venv-macports/bin/python3 scripts/export_feature_dataset.py" in result.stdout
    assert "opencv-python" not in requirements
    assert "pyobjc-framework" not in requirements.lower()
    assert "fastapi>=0.110" in requirements
    assert "no dashboard pairing code" in common


def test_appletv_srp_dependency_is_pinned_below_breaking_release():
    requirements = REQUIREMENTS.read_text()
    assert "pyatv>=0.16,<0.19" in requirements
    assert "srptools>=1.0.1,<2" in requirements
    assert "opencv-python-headless>=4.8,<5" in requirements


def test_ml_requirements_are_additive_and_do_not_pull_platform_runtime_or_tests():
    requirements = ML_REQUIREMENTS.read_text()
    entries = [
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "scikit-learn>=1.3" in requirements
    assert "joblib>=1.3" in requirements
    assert not any(line.startswith("-r ") for line in entries)
    assert not any("pytest" in line for line in entries)
    assert not any("opencv" in line for line in entries)
