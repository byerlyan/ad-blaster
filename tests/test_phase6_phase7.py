import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import config
import device_setup as device_setup_module
import server
from corrections import CorrectionStore
from device_setup import AppleTVPairingError, AppleTVSetup


@pytest.fixture
def restore_settings():
    saved = {key: getattr(config, key) for key in server.TUNABLE_SETTINGS}
    yield
    for key, value in saved.items():
        setattr(config, key, value)


def test_settings_reports_preset_and_custom_overrides(restore_settings):
    client = TestClient(server.app)
    assert client.post("/api/preset", json={"name": "balanced"}).status_code == 200
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.json()["_meta"]["active_preset"] == "balanced"

    assert client.post("/api/settings", json={"DEBOUNCE_FRAMES": 17}).status_code == 200
    assert client.get("/api/settings").json()["_meta"]["custom_overrides"] is True


def test_settings_validation_and_secret_free_export(restore_settings):
    client = TestClient(server.app)
    assert client.post("/api/settings", json={"LOGO_MATCH_THRESHOLD": 2}).status_code == 400
    assert client.post(
        "/api/settings",
        json={
            "HYBRID_PROGRAM_EXIT_THRESHOLD": 0.8,
            "HYBRID_COMMERCIAL_ENTER_THRESHOLD": 0.7,
        },
    ).status_code == 400
    exported = client.get("/api/settings/export").json()
    serialized = json.dumps(exported).lower()
    assert exported["schema_version"] == 1
    assert "credential" not in serialized
    assert "session" not in serialized


def test_corrections_are_durable_deduplicated_relabelable_and_undoable(tmp_path):
    store = CorrectionStore(tmp_path / "corrections.sqlite3", duplicate_window_seconds=3)
    first, created = store.add(
        label="program",
        session_id="capture-1",
        channel="espn",
        action="none",
        evidence={"confidence": 0.4, "features": [1, 2]},
        timestamp=100,
    )
    duplicate, duplicate_created = store.add(
        label="program",
        session_id="capture-1",
        channel="espn",
        action="none",
        evidence={},
        timestamp=101,
    )
    assert created is True
    assert duplicate_created is False
    assert duplicate["id"] == first["id"]
    assert store.relabel(first["id"], "commercial")["label"] == "commercial"
    assert store.undo(first["id"])["undone"] is True
    assert CorrectionStore(tmp_path / "corrections.sqlite3").list()[0]["evidence"]["features"] == [1, 2]


def test_dashboard_uses_direct_private_lan_access_without_pairing(restore_settings):
    client = TestClient(server.app)

    assert client.get("/api/settings").status_code == 200
    assert client.post(
        "/api/settings", json={"AUTOMATIC_MUTING_ENABLED": False}
    ).status_code == 200
    assert client.post(
        "/api/settings",
        json={"AUTOMATIC_MUTING_ENABLED": True, "AUTOMATIC_SKIPPING_ENABLED": False},
    ).json()["AUTOMATIC_SKIPPING_ENABLED"] is False
    assert client.get("/api/auth/status").status_code == 404
    assert server._is_local_network_client("127.0.0.1")
    assert server._is_local_network_client("192.168.1.25")
    assert server._is_local_network_client("10.20.30.40")
    assert not server._is_local_network_client("8.8.8.8")


def test_audio_device_api_returns_dropdown_choices(monkeypatch, restore_settings):
    monkeypatch.setattr(config, "AUDIO_DEVICE_NAME", "USB Digital Audio")
    monkeypatch.setattr(
        server,
        "audio_device_diagnostics",
        lambda _binary: {
            "devices": [
                {"id": "0", "name": "MacBook Pro Microphone"},
                {"id": "1", "name": "USB Digital Audio"},
            ],
            "supported": True,
            "error": None,
        },
    )

    response = TestClient(server.app).get("/api/devices/audio")

    assert response.status_code == 200
    assert response.json()["selected"] == "USB Digital Audio"
    assert response.json()["devices"][1]["name"] == "USB Digital Audio"


def test_dashboard_mobile_navigation_touch_calibration_and_pwa_files_exist():
    root = Path(__file__).parents[1]
    html = (root / "dashboard.html").read_text()
    manifest = json.loads((root / "manifest.webmanifest").read_text())
    for destination in ("watch", "activity", "setup", "tune", "diagnostics"):
        assert f'bottom-btn-{destination}' in html
        assert f'tab-{destination}' in html
    assert 'id="bottom-btn-help" href="/help"' in html
    assert "touchstart" in html and "touchmove" in html
    assert "resize-" in html and "pinch" in html
    assert "@media (max-width: 430px)" in html
    assert "min-height:44px" in html
    assert manifest["display"] == "standalone"
    assert {icon["sizes"] for icon in manifest["icons"]} == {"192x192", "512x512"}
    assert 'id="pair-modal"' not in html
    assert "csrfToken" not in html
    assert "no Clicker pairing code or dashboard login" in html
    assert 'id="basic-auto-muting"' in html
    assert 'id="basic-auto-skipping"' in html
    assert "Leave skipping off for mute-only operation" in html
    assert '<select class="input" id="set-audio-device"' in html
    assert 'input class="input" id="set-audio-device"' not in html
    assert "/api/devices/audio" in html
    assert "function startSampleCountdown" in html
    assert "Processing profile..." in html
    assert "seconds remaining" in html
    assert 'id="control-mode-bar"' in html
    assert "SHADOW MODE · Detection and training only" in html
    assert "shadow:false" in html
    assert "shadow:localStorage" not in html
    assert 'id="basic-shadow"' not in html


def test_help_page_documents_legacy_training_and_safety_workflows():
    root = Path(__file__).parents[1]
    html = (root / "help.html").read_text()
    server = (root / "server.py").read_text()
    assert '@app.get("/help", response_class=HTMLResponse)' in server
    assert "Using the Legacy system" in html
    assert "Mark Content" in html and "Mark Commercial" in html
    assert "scripts/export_feature_dataset.py" in html
    assert "scripts/train_commercial_model.py" in html
    assert "data/commercial_model.joblib" in html
    assert "at least four distinct" in html
    assert "Shadow mode" in html
    assert "Flagging ad labels and countdowns" in html
    assert "A timer alone is ignored" in html
    assert "Streaming shows without a logo" in html
    assert "What the Program and Commercial markers do" in html
    assert "One timestamped correction" in html
    assert "continuous window" in html
    assert 'href="/"' in html


def test_dashboard_clarifies_quick_corrections_and_no_logo_profiles():
    html = (Path(__file__).parents[1] / "dashboard.html").read_text()
    assert 'id="quick-label-help"' in html
    assert "save one timestamped evidence snapshot" in html
    assert "do not immediately change detector state" in html
    assert "click once to start and again to stop" in html
    assert "Streaming show with no persistent logo?" in html
    assert "visual scores remain available to diagnostics and ML features" in html
    assert 'id="preview-capture"' in html
    assert "Older profile — source not recorded" in html


def test_threshold_suggestions_can_be_cleared_without_clearing_samples():
    html = (Path(__file__).parents[1] / "dashboard.html").read_text()
    assert 'id="btn-suggest-thresholds"' in html
    assert 'id="btn-clear-suggestions"' in html
    assert 'id="btn-clear-samples"' in html
    assert "Suggestions cleared. Labeled samples were kept." in html
    clear_suggestions = html.split("function clearSuggestions", 1)[1].split(
        "async function clearSamples", 1
    )[0]
    assert "/api/mark/clear" not in clear_suggestions
    assert "Logo (combined edge + template)" in html


def test_full_frame_diagnostics_support_no_logo_profile_regions():
    full_frame_color = SimpleNamespace(_mode="color", _roi_rect=None, corner="color")
    custom_color = SimpleNamespace(
        _mode="color", _roi_rect=(100, 50, 300, 200), corner="color"
    )
    combo = SimpleNamespace(
        _mode="combo", _roi_rect=(1500, 20, 300, 120), corner="custom"
    )

    assert server._active_roi_annotation(full_frame_color, 1920, 1080) == (
        (0, 0, 1919, 1079),
        "color full frame",
    )
    assert server._active_roi_annotation(custom_color, 1920, 1080) == (
        (100, 50, 399, 249),
        "color",
    )
    assert server._active_roi_annotation(combo, 1920, 1080) == (
        (1500, 20, 1799, 139),
        "combo logo",
    )


def test_dashboard_exposes_safe_ad_overlay_settings_and_diagnostics():
    root = Path(__file__).parents[1]
    html = (root / "dashboard.html").read_text()
    server = (root / "server.py").read_text()
    assert 'id="set-ad-overlay"' in html
    assert "timer ignored without ad label" in html
    assert "'set-ad-overlay':'AD_OVERLAY_ENABLED'" in html
    assert '"AD_OVERLAY_ENABLED": bool' in server
    assert '"HYBRID_AD_OVERLAY_WEIGHT": float' in server


def test_setup_wizard_capture_check_has_visible_inline_feedback():
    html = (Path(__file__).parents[1] / "dashboard.html").read_text()
    assert 'id="wizard-capture-select"' in html
    assert 'id="wizard-capture-check" onclick="checkWizardCapture()"' in html
    assert 'id="wizard-capture-status"' in html
    assert 'role="status" aria-live="polite"' in html
    assert "async function scanWizardCaptureDevices()" in html
    assert "async function checkWizardCapture()" in html
    assert "await api('POST','/api/devices/capture/select'" in html


def test_capture_defaults_follow_runtime_device_selection(monkeypatch):
    from capture import FrameCapture, _find_capture_card_index

    monkeypatch.setattr(config, "CAPTURE_DEVICE", 7)
    monkeypatch.setattr(config, "CAPTURE_DEVICE_NAME", "Selected Capture Card")
    assert FrameCapture()._device == 7
    assert _find_capture_card_index.__defaults__ == (None,)


def test_setup_wizard_supports_auth_and_airplay_code_submission():
    html = (Path(__file__).parents[1] / "dashboard.html").read_text()
    assert 'id="wizard-appletv-select"' in html
    assert 'id="wizard-auth-pin"' in html
    assert 'id="wizard-airplay-pin"' in html
    assert "beginWizardAppleTVPair(\\'Companion\\')" in html
    assert "beginWizardAppleTVPair(\\'AirPlay\\')" in html
    assert "finishWizardAppleTVPair(\\'Companion\\')" in html
    assert "finishWizardAppleTVPair(\\'AirPlay\\')" in html


def test_all_wizard_steps_use_inline_api_routes():
    html = (Path(__file__).parents[1] / "dashboard.html").read_text()
    wizard = html.split("const WIZARD_STEPS=[", 1)[1].split("];", 1)[0]
    assert "showTab(" not in wizard
    assert 'id="wizard-profile-select"' in wizard
    assert 'id="wizard-calibrate-start"' in wizard
    assert 'onclick="startWizardShadow()"' in wizard
    assert 'id="wizard-finish-status"' in wizard
    for function in (
        "saveWizardDetectionMode",
        "loadWizardProfiles",
        "selectWizardProfile",
        "runWizardCalibration",
        "startWizardShadow",
        "finishWizardSetup",
    ):
        assert f"function {function}" in html
    for route in (
        "/api/settings",
        "/api/profiles",
        "/api/profile/",
        "/api/calibrate",
        "/api/status",
        "/api/stop",
        "/api/start",
        "/api/setup",
    ):
        assert route in html


def test_appletv_setup_stores_both_pairing_codes_and_credentials(tmp_path):
    class Service:
        def __init__(self, protocol, credentials):
            self.protocol = protocol
            self.credentials = credentials

    class Handler:
        device_provides_pin = True

        def __init__(self, protocol, credentials):
            self.service = Service(protocol, credentials)
            self.has_paired = False
            self.submitted_pin = None

        def pin(self, value):
            self.submitted_pin = value

        async def finish(self):
            self.has_paired = True

        async def close(self):
            pass

    credentials_path = tmp_path / "credentials.json"
    setup = AppleTVSetup(str(credentials_path))
    auth = Handler("Protocol.Companion", "auth-credentials")
    airplay = Handler("Protocol.AirPlay", "airplay-credentials")
    setup._pending = {
        "auth-token": (auth, time.time() + 60),
        "airplay-token": (airplay, time.time() + 60),
    }

    asyncio.run(setup.finish("auth-token", 1234))
    asyncio.run(setup.finish("airplay-token", 5678))

    assert auth.submitted_pin == 1234
    assert airplay.submitted_pin == 5678
    assert json.loads(credentials_path.read_text()) == {
        "Protocol.Companion": "auth-credentials",
        "Protocol.AirPlay": "airplay-credentials",
    }


def test_appletv_scan_excludes_macs_and_homepods(monkeypatch, tmp_path):
    def device(name, identifier, operating_system, model):
        return SimpleNamespace(
            name=name,
            identifier=identifier,
            address="192.0.2.1",
            services=[],
            device_info=SimpleNamespace(
                operating_system=SimpleNamespace(name=operating_system),
                model=SimpleNamespace(name=model),
            ),
        )

    async def fake_scan(*_args, **_kwargs):
        return [
            device("MacBook", "mac", "MacOS", "Unknown"),
            device("Living Room", "tv", "TvOS", "AppleTV4KGen2"),
            device("Kitchen HomePod", "speaker", "TvOS", "HomePodMini"),
        ]

    monkeypatch.setattr(device_setup_module.pyatv, "scan", fake_scan)
    devices = asyncio.run(AppleTVSetup(str(tmp_path / "credentials.json")).scan())
    assert [item["id"] for item in devices] == ["tv"]


def test_appletv_pairing_error_survives_cleanup_failure(tmp_path):
    class Handler:
        device_provides_pin = True
        has_paired = False
        service = SimpleNamespace(protocol="Protocol.Companion", credentials=None)

        def pin(self, _value):
            pass

        async def finish(self):
            raise RuntimeError("authentication code was rejected")

        async def close(self):
            raise RuntimeError("cleanup failed")

    setup = AppleTVSetup(str(tmp_path / "credentials.json"))
    setup._pending = {"token": (Handler(), time.time() + 60)}
    with pytest.raises(AppleTVPairingError, match="authentication code was rejected"):
        asyncio.run(setup.finish("token", 1234))


def test_appletv_pairing_endpoint_returns_actionable_error(monkeypatch):
    async def fail_pairing(_identifier, _protocol):
        raise RuntimeError("network handshake failed")

    monkeypatch.setattr(server.apple_tv_setup, "begin", fail_pairing)
    response = TestClient(server.app).post(
        "/api/devices/appletv/pair/start",
        json={"identifier": "tv", "protocol": "Companion"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Could not start Companion pairing: network handshake failed"
    )
