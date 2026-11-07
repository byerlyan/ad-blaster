"""Frame capture from a USB HDMI capture card via OpenCV.

Supports both Linux V4L2 device paths and macOS device indices.
"""

import logging
import os
import pathlib
import platform

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def _objc_value(obj, name: str, default=None):
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def camera_authorization_status() -> str:
    """Return the macOS camera authorization state without prompting."""
    if platform.system() != "Darwin":
        return "not_applicable"
    try:
        import AVFoundation as av

        status = av.AVCaptureDevice.authorizationStatusForMediaType_(
            av.AVMediaTypeVideo
        )
        mapping = {
            int(getattr(av, "AVAuthorizationStatusNotDetermined", 0)): "not_determined",
            int(getattr(av, "AVAuthorizationStatusRestricted", 1)): "restricted",
            int(getattr(av, "AVAuthorizationStatusDenied", 2)): "denied",
            int(getattr(av, "AVAuthorizationStatusAuthorized", 3)): "authorized",
        }
        return mapping.get(int(status), f"unknown_{int(status)}")
    except Exception as exc:
        logger.warning("Could not inspect camera authorization: %s", exc)
        return "unknown"


def opencv_avfoundation_available() -> bool:
    """Whether this OpenCV build exposes its AVFoundation capture backend."""
    if platform.system() != "Darwin":
        return False
    try:
        registry = getattr(cv2, "videoio_registry", None)
        if registry is not None and hasattr(registry, "hasBackend"):
            return bool(registry.hasBackend(cv2.CAP_AVFOUNDATION))
        return "AVFoundation: YES" in cv2.getBuildInformation()
    except Exception:
        return False


def discover_avfoundation_devices() -> list[dict]:
    """Enumerate macOS video devices in the order OpenCV uses for indices.

    Monterey exposes ``AVCaptureDeviceTypeExternalUnknown`` while newer macOS
    releases expose ``AVCaptureDeviceTypeExternal``. Importing the latter by
    name on Monterey makes the entire scan fail, so constants are selected at
    runtime. The legacy ``devicesWithMediaType`` order is preferred because
    OpenCV's macOS backend uses that same ordering for integer device indices.
    """
    if platform.system() != "Darwin":
        return []

    import AVFoundation as av

    devices = []
    capture_device = av.AVCaptureDevice
    devices_for_media = getattr(capture_device, "devicesWithMediaType_", None)
    if callable(devices_for_media):
        for media_type in (
            av.AVMediaTypeVideo,
            getattr(av, "AVMediaTypeMuxed", None),
        ):
            if media_type is None:
                continue
            try:
                devices.extend(list(devices_for_media(media_type)))
            except Exception:
                logger.debug("Legacy AVFoundation enumeration failed", exc_info=True)

    # Newer PyObjC/SDK combinations can remove the deprecated class method.
    # DiscoverySession is the fallback, with device types chosen by OS support.
    if not devices:
        discovery = getattr(av, "AVCaptureDeviceDiscoverySession", None)
        if discovery is not None:
            device_types = []
            for constant_name in (
                "AVCaptureDeviceTypeBuiltInWideAngleCamera",
                "AVCaptureDeviceTypeExternal",
                "AVCaptureDeviceTypeExternalUnknown",
            ):
                value = getattr(av, constant_name, None)
                if value is not None and value not in device_types:
                    device_types.append(value)
            for media_type in (
                av.AVMediaTypeVideo,
                getattr(av, "AVMediaTypeMuxed", None),
            ):
                if media_type is None:
                    continue
                session = discovery.discoverySessionWithDeviceTypes_mediaType_position_(
                    device_types,
                    media_type,
                    av.AVCaptureDevicePositionUnspecified,
                )
                devices.extend(list(session.devices()))

    unique_devices = []
    seen = set()
    for device in devices:
        unique_id = str(_objc_value(device, "uniqueID", ""))
        name = str(_objc_value(device, "localizedName", "Unknown camera"))
        identity = unique_id or name
        if identity in seen:
            continue
        seen.add(identity)
        unique_devices.append((identity, unique_id, name, device))

    # OpenCV's cap_avfoundation_mac.mm sorts the combined video/muxed list by
    # uniqueID before resolving its integer camera index. Mirror it exactly.
    unique_devices.sort(key=lambda item: item[0])
    result = []
    for identity, unique_id, name, device in unique_devices:
        result.append(
            {
                "id": unique_id or name,
                "name": name,
                "index": len(result),
                "model_id": str(_objc_value(device, "modelID", "")),
                "connected": bool(_objc_value(device, "isConnected", True)),
                "in_use": bool(
                    _objc_value(device, "isInUseByAnotherApplication", False)
                ),
            }
        )
    return result


def _find_capture_card_index(expected_name: str | None = None) -> int | None:
    """Find the OpenCV index for a capture device by its AVFoundation name.

    Replicates OpenCV's cap_avfoundation_mac.mm device enumeration:
    discovers video + muxed devices, sorts by uniqueID, returns position.
    Uses AVCaptureDeviceDiscoverySession (required on modern macOS to
    find external USB devices).
    """
    if expected_name is None:
        # Resolve this at call time so a device selected in the dashboard takes
        # effect without requiring the Python process to restart.
        expected_name = config.CAPTURE_DEVICE_NAME

    try:
        devices = discover_avfoundation_devices()
        wanted = str(expected_name).strip().casefold()
        for device in devices:
            if device["name"].strip().casefold() == wanted:
                logger.info(
                    "Found '%s' at OpenCV index %d (uniqueID=%s)",
                    expected_name, device["index"], device["id"],
                )
                return int(device["index"])

        available = [(d["name"], d["id"]) for d in devices]
        logger.warning("Device '%s' not found. Available: %s", expected_name, available)

    except ImportError:
        logger.warning("pyobjc-framework-AVFoundation not installed, cannot enumerate devices")
    except Exception as e:
        logger.warning("Failed to enumerate AVFoundation devices: %s", e)

    return None


class FrameCapture:
    """Manages the OpenCV VideoCapture device."""

    def __init__(self, device=None):
        # CAPTURE_DEVICE can be updated by the setup wizard at runtime.
        self._device = config.CAPTURE_DEVICE if device is None else device
        self._resolve_by_name = device is None
        self._cap = None

    def open(self):
        """Open the capture device."""
        if (
            platform.system() == "Darwin"
            and isinstance(self._device, int)
            and self._resolve_by_name
        ):
            # Find capture card by name to avoid index instability
            idx = _find_capture_card_index()
            if idx is not None:
                self._device = idx
            else:
                logger.warning(
                    "Could not find '%s', falling back to index %d",
                    config.CAPTURE_DEVICE_NAME, self._device,
                )

        logger.info("Opening capture device: %s", self._device)

        if isinstance(self._device, int):
            if platform.system() == "Darwin":
                self._cap = cv2.VideoCapture(self._device, cv2.CAP_AVFOUNDATION)
            else:
                self._cap = cv2.VideoCapture(self._device)
        else:
            self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open capture device: {self._device}{self._open_failure_hint()}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w <= 0 or actual_h <= 0:
            raise RuntimeError("Capture device opened but did not report a video resolution")
        if actual_w != config.CAPTURE_WIDTH or actual_h != config.CAPTURE_HEIGHT:
            logger.warning(
                "Capture device uses %dx%d instead of requested %dx%d; continuing",
                actual_w,
                actual_h,
                config.CAPTURE_WIDTH,
                config.CAPTURE_HEIGHT,
            )
        logger.info("Capture device opened. Resolution: %dx%d", actual_w, actual_h)

    def _open_failure_hint(self) -> str:
        """Explain a likely cause when a V4L2 device path won't open.

        The bare "Failed to open" is opaque; the usual culprits on a Linux/LXC
        deploy are a missing passthrough or a permission denied node (common in
        unprivileged Proxmox containers where the host video device stays
        root:video 0660 and container root maps to an unprivileged uid).
        """
        if not isinstance(self._device, str):
            return ""
        if not os.path.exists(self._device):
            return ", device path does not exist (check USB passthrough to the container)."
        if not os.access(self._device, os.R_OK | os.W_OK):
            return (
                ", the node exists but isn't readable/writable by this user. On an "
                "unprivileged Proxmox LXC, fix the HOST permissions with a udev rule "
                '(KERNEL=="video[0-9]*", MODE="0666") or chmod the host node.'
            )
        return ", node is accessible but the capture backend couldn't open it (device busy or unsupported format?)."

    def grab_frame(self) -> np.ndarray | None:
        """Grab a single frame. Returns BGR numpy array or None on failure."""
        if self._cap is None or not self._cap.isOpened():
            logger.error("Capture device not open.")
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame.")
            return None

        return frame

    def close(self):
        """Release the capture device."""
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("Capture device closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def is_sync_artifact_frame(frame: np.ndarray) -> bool:
    """Detect the saturated green frame USB capture cards emit while syncing to HDMI.

    The artifact frame has an extreme color signature: mean green channel near
    255, blue and red near 0. Real broadcast content never looks like that, 
    even the greenest court advertisement keeps nontrivial blue/red components.
    """
    if frame is None or frame.size == 0:
        return False
    # BGR ordering in OpenCV
    mean_b, mean_g, mean_r = frame.reshape(-1, 3).mean(axis=0)
    return mean_g > 180 and mean_b < 60 and mean_r < 60


def extract_corner_rois(
    frame: np.ndarray,
    roi_w: int = config.CORNER_ROI_WIDTH,
    roi_h: int = config.CORNER_ROI_HEIGHT,
) -> dict[str, np.ndarray]:
    """Extract ROIs from all four corners of a frame.

    Returns a dict keyed by corner name: "top_left", "top_right",
    "bottom_left", "bottom_right".
    """
    h, w = frame.shape[:2]
    return {
        "top_left": frame[0:roi_h, 0:roi_w],
        "top_right": frame[0:roi_h, w - roi_w : w],
        "bottom_left": frame[h - roi_h : h, 0:roi_w],
        "bottom_right": frame[h - roi_h : h, w - roi_w : w],
    }


# ----------------------------------------------------------------------
# Quick test: grab and display a frame
# ----------------------------------------------------------------------

def _test():
    # Dev-only helper: dump a frame + corner ROIs into the gitignored scratch/
    # dir so they don't litter the repo root.
    scratch = pathlib.Path(__file__).parent / "scratch"
    scratch.mkdir(exist_ok=True)
    with FrameCapture() as cap:
        print("Grabbing a test frame...")
        frame = cap.grab_frame()
        if frame is not None:
            print(f"Got frame: {frame.shape}")
            frame_path = scratch / "test_frame.png"
            cv2.imwrite(str(frame_path), frame)
            print(f"Saved to {frame_path}")

            rois = extract_corner_rois(frame)
            for name, roi in rois.items():
                roi_path = scratch / f"test_roi_{name}.png"
                cv2.imwrite(str(roi_path), roi)
                print(f"Saved ROI {name}: {roi.shape} -> {roi_path}")
        else:
            print("No frame captured.")


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )
    _test()
