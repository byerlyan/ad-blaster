import sys
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import calibrate
import capture


class FakeDevice:
    def __init__(self, name, unique_id, *, connected=True, in_use=False):
        self._name = name
        self._unique_id = unique_id
        self._connected = connected
        self._in_use = in_use

    def localizedName(self):
        return self._name

    def uniqueID(self):
        return self._unique_id

    def modelID(self):
        return "USB Video"

    def isConnected(self):
        return self._connected

    def isInUseByAnotherApplication(self):
        return self._in_use


def monterey_avfoundation(devices, authorization=3):
    class CaptureDevice:
        @staticmethod
        def devicesWithMediaType_(media_type):
            return devices if media_type == "video" else []

        @staticmethod
        def authorizationStatusForMediaType_(_media_type):
            return authorization

    # Intentionally omit AVCaptureDeviceTypeExternal. Monterey exposes the
    # older ExternalUnknown symbol; legacy enumeration should still work.
    return SimpleNamespace(
        AVCaptureDevice=CaptureDevice,
        AVCaptureDeviceTypeBuiltInWideAngleCamera="wide",
        AVCaptureDeviceTypeExternalUnknown="external-unknown",
        AVCaptureDevicePositionUnspecified=0,
        AVMediaTypeVideo="video",
        AVMediaTypeMuxed="muxed",
        AVAuthorizationStatusNotDetermined=0,
        AVAuthorizationStatusRestricted=1,
        AVAuthorizationStatusDenied=2,
        AVAuthorizationStatusAuthorized=3,
    )


def test_monterey_external_unknown_devices_match_opencv_unique_id_order(monkeypatch):
    facetime = FakeDevice("FaceTime HD Camera", "built-in")
    capture_card = FakeDevice("Guermok USB3 Video", "usb-card")
    monkeypatch.setattr(capture.platform, "system", lambda: "Darwin")
    monkeypatch.setitem(
        sys.modules,
        "AVFoundation",
        # Feed the devices in reverse order. OpenCV sorts its combined
        # AVFoundation list by uniqueID before applying integer indexes.
        monterey_avfoundation([capture_card, facetime]),
    )

    devices = capture.discover_avfoundation_devices()

    assert [(item["name"], item["index"]) for item in devices] == [
        ("FaceTime HD Camera", 0),
        ("Guermok USB3 Video", 1),
    ]
    assert capture._find_capture_card_index("Guermok USB3 Video") == 1


def test_monterey_camera_denial_is_reported(monkeypatch):
    monkeypatch.setattr(capture.platform, "system", lambda: "Darwin")
    monkeypatch.setitem(
        sys.modules,
        "AVFoundation",
        monterey_avfoundation([], authorization=2),
    )

    assert capture.camera_authorization_status() == "denied"


def test_explicit_macos_index_is_not_remapped_and_non_1080p_is_accepted(monkeypatch):
    class FakeVideoCapture:
        def __init__(self, index, backend):
            assert index == 2
            assert backend == capture.cv2.CAP_AVFOUNDATION

        def isOpened(self):
            return True

        def set(self, _prop, _value):
            return False

        def get(self, prop):
            if prop == capture.cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            if prop == capture.cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            return 0

        def release(self):
            pass

    monkeypatch.setattr(capture.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(capture.cv2, "VideoCapture", FakeVideoCapture)
    monkeypatch.setattr(
        capture,
        "_find_capture_card_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit index should not be remapped")
        ),
    )

    cap = capture.FrameCapture(2)
    cap.open()

    assert cap._device == 2
    cap.close()


def test_default_macos_capture_never_falls_back_to_facetime_index(monkeypatch):
    monkeypatch.setattr(capture.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(capture.config, "CAPTURE_DEVICE", 0)
    monkeypatch.setattr(capture.config, "CAPTURE_DEVICE_NAME", "Missing HDMI card")
    monkeypatch.setattr(capture, "_find_capture_card_index", lambda: None)
    monkeypatch.setattr(
        capture.cv2,
        "VideoCapture",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("an unselected numeric camera must never be opened")
        ),
    )

    with pytest.raises(RuntimeError, match="numeric fallback is disabled"):
        capture.FrameCapture().open()


def test_calibration_resolves_the_selected_capture_device_at_call_time(monkeypatch):
    for function in (
        calibrate.calibrate,
        calibrate.calibrate_roi,
        calibrate.calibrate_combo,
        calibrate.calibrate_color,
    ):
        assert inspect.signature(function).parameters["device"].default is None

    monkeypatch.setattr(calibrate.os, "uname", lambda: SimpleNamespace(sysname="Darwin"))
    monkeypatch.setattr(calibrate.config, "CAPTURE_DEVICE_NAME", "Guermok USB3 Video")
    assert calibrate._capture_source_label() == "Guermok USB3 Video"


def test_color_profile_records_its_capture_source(tmp_path):
    calibrate.save_color_profile(
        "streaming",
        np.ones((2, 2), dtype=np.float32),
        np.zeros((8, 8, 3), dtype=np.uint8),
        profiles_dir=str(tmp_path),
        capture_source="Guermok USB3 Video",
    )

    metadata = json.loads((tmp_path / "streaming.json").read_text())
    assert metadata["capture_source"] == "Guermok USB3 Video"
