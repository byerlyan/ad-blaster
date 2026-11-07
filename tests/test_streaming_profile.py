import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import calibrate
import config
import server
from detector import LogoDetector


def test_streaming_default_is_reusable_color_profile_and_enables_hybrid(
    monkeypatch, tmp_path
):
    applied = {}

    def fake_calibrate_color(channel, roi, duration=None):
        assert channel == "streaming-default"
        assert roi is None
        assert duration == 15
        return calibrate.save_color_profile(
            channel,
            np.ones((4, 4), dtype=np.float32),
            np.full((20, 30, 3), 80, dtype=np.uint8),
            profiles_dir=str(tmp_path),
            capture_source="Test Capture",
        )

    def fake_apply_settings(settings, *, persist=True):
        applied.update(settings)
        return settings

    monkeypatch.setattr(config, "PROFILES_DIR", str(tmp_path))
    monkeypatch.setattr(config, "STREAMING_DEFAULT_PROFILE", "streaming-default")
    monkeypatch.setattr(calibrate, "calibrate_color", fake_calibrate_color)
    monkeypatch.setattr(server, "_apply_settings", fake_apply_settings)

    response = TestClient(server.app).post(
        "/api/profiles/streaming-default", json={"duration": 15}
    )

    assert response.status_code == 200
    assert response.json()["channel"] == "streaming-default"
    assert applied == {
        "DETECTION_MODE": "hybrid",
        "AD_OVERLAY_ENABLED": True,
        "HYBRID_LEGACY_VISUAL_ENABLED": True,
    }
    metadata = json.loads((tmp_path / "streaming-default.json").read_text())
    assert metadata["profile_role"] == "streaming_default"
    assert metadata["reusable_across_apps"] is True
    assert metadata["overlay_focused"] is True
    assert metadata["color_scope"] == "full_frame"

    profile = calibrate.load_profile("streaming-default", profiles_dir=str(tmp_path))
    detector = LogoDetector(profile)
    assert detector._mode == "color"
    assert detector._roi_rect is None

    profiles = TestClient(server.app).get("/api/profiles").json()
    assert profiles["default_profile"] == "streaming-default"
    assert "streaming-default" in profiles["profiles"]


def test_dashboard_exposes_streaming_default_in_calibration_and_wizard():
    html = (Path(__file__).parents[1] / "dashboard.html").read_text()
    assert 'id="btn-create-streaming-default"' in html
    assert 'id="wizard-streaming-default"' in html
    assert "async function createStreamingDefault" in html
    assert "/api/profiles/streaming-default" in html
    assert "STREAMING DEFAULT · reusable" in html
    assert "Hybrid + ad-overlay OCR enabled" in html
