"""Dashboard-safe capture enumeration and Apple TV discovery/pairing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import secrets
import shutil
import time
from pathlib import Path

import pyatv

logger = logging.getLogger(__name__)


class AppleTVPairingError(RuntimeError):
    """Actionable discovery or pairing failure safe to show in the dashboard."""


def _is_apple_tv(config) -> bool:
    """Exclude Macs and HomePods that also advertise AirPlay/Companion."""
    try:
        info = config.device_info
        os_name = getattr(info.operating_system, "name", str(info.operating_system))
        model_name = getattr(info.model, "name", str(info.model))
        return os_name == "TvOS" and not model_name.startswith("HomePod")
    except Exception:
        return False


def list_capture_devices() -> list[dict]:
    if platform.system() == "Darwin":
        return capture_device_diagnostics()["devices"]
    return [
        {"id": str(path), "name": path.name}
        for path in sorted(Path("/dev").glob("video*"))
        if os.access(path, os.R_OK)
    ]


def capture_device_diagnostics() -> dict:
    """Return device enumeration plus actionable macOS backend diagnostics."""
    if platform.system() != "Darwin":
        return {"devices": list_capture_devices(), "permission": "not_applicable"}
    try:
        from capture import (
            camera_authorization_status,
            discover_avfoundation_devices,
            opencv_avfoundation_available,
        )

        return {
            "devices": discover_avfoundation_devices(),
            "permission": camera_authorization_status(),
            "backend_available": opencv_avfoundation_available(),
        }
    except Exception as exc:
        logger.exception("Could not enumerate macOS capture devices")
        return {
            "devices": [],
            "permission": "unknown",
            "backend_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def audio_device_diagnostics(ffmpeg_binary: str = "ffmpeg") -> dict:
    """Return the audio inputs exposed to the runtime's FFmpeg.

    AudioAnalyzer currently uses AVFoundation, so probing through FFmpeg keeps
    dashboard names identical to the names accepted when the engine starts.
    """
    if platform.system() != "Darwin":
        return {
            "devices": [],
            "supported": False,
            "error": "Audio capture currently supports macOS AVFoundation only",
        }
    try:
        from audio import probe_avfoundation_audio_devices

        binary = shutil.which(ffmpeg_binary)
        if binary is None:
            return {
                "devices": [],
                "supported": True,
                "error": f"FFmpeg executable not found: {ffmpeg_binary}",
            }
        devices = probe_avfoundation_audio_devices(binary)
        return {
            "devices": [
                {"id": str(device.index), "name": device.name}
                for device in devices
            ],
            "supported": True,
            "error": None if devices else "No AVFoundation audio inputs were found",
        }
    except Exception as exc:
        logger.exception("Could not enumerate macOS audio devices")
        return {
            "devices": [],
            "supported": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


class AppleTVSetup:
    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self._pending: dict[str, tuple[object, float]] = {}

    async def scan(self) -> list[dict]:
        configs = await pyatv.scan(asyncio.get_running_loop(), timeout=5)
        return [
            {
                "id": str(conf.identifier),
                "name": conf.name,
                "address": str(conf.address),
                "services": [str(service.protocol) for service in conf.services],
                "model": getattr(conf.device_info.model, "name", str(conf.device_info.model)),
            }
            for conf in configs
            if _is_apple_tv(conf)
        ]

    @staticmethod
    async def _close_handler(handler) -> None:
        if handler is None:
            return
        try:
            await handler.close()
        except Exception:
            # Cleanup must never replace the useful pairing error (or turn an
            # otherwise successful pairing into an HTTP 500).
            logger.warning("Failed to close Apple TV pairing handler", exc_info=True)

    async def begin(self, identifier: str, protocol_name: str = "Companion") -> dict:
        protocol = getattr(pyatv.const.Protocol, protocol_name, None)
        if protocol not in {pyatv.const.Protocol.Companion, pyatv.const.Protocol.AirPlay}:
            raise ValueError("Pairing protocol must be Companion or AirPlay")
        handler = None
        try:
            configs = await pyatv.scan(
                asyncio.get_running_loop(), identifier=identifier, timeout=5
            )
            if not configs:
                raise ValueError("Apple TV not found. Refresh the device list and try again.")
            selected = configs[0]
            if not _is_apple_tv(selected):
                raise ValueError(f'"{selected.name}" is not an Apple TV')
            if selected.get_service(protocol) is None:
                raise ValueError(
                    f'"{selected.name}" does not advertise the {protocol_name} service'
                )
            handler = await pyatv.pair(selected, protocol, asyncio.get_running_loop())
            await handler.begin()
            token = secrets.token_urlsafe(24)
            self._pending[token] = (handler, time.time() + 300)
            return {
                "token": token,
                "protocol": protocol_name,
                "device_provides_pin": bool(handler.device_provides_pin),
                "pin_code": getattr(handler, "pin_code", None),
            }
        except ValueError:
            await self._close_handler(handler)
            raise
        except Exception as exc:
            await self._close_handler(handler)
            message = str(exc).strip() or type(exc).__name__
            raise AppleTVPairingError(
                f"Could not start {protocol_name} pairing: {message}"
            ) from exc

    async def finish(self, token: str, pin: int | None = None) -> dict:
        pending = self._pending.pop(token, None)
        if pending is None:
            raise ValueError("Pairing session expired")
        handler = pending[0]
        if time.time() >= pending[1]:
            await self._close_handler(handler)
            raise ValueError("Pairing session expired")
        try:
            if handler.device_provides_pin:
                if pin is None:
                    raise ValueError("Enter the PIN shown on the Apple TV")
                handler.pin(int(pin))
            await handler.finish()
            if not handler.has_paired:
                raise ValueError("Apple TV pairing was not accepted")
            path = Path(self.credentials_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open() as handle:
                    credentials = json.load(handle)
            except FileNotFoundError:
                credentials = {}
            credentials[str(handler.service.protocol)] = handler.service.credentials
            with path.open("w") as handle:
                json.dump(credentials, handle, indent=2)
            os.chmod(path, 0o600)
            return {"paired": True, "protocol": str(handler.service.protocol)}
        except ValueError:
            raise
        except Exception as exc:
            protocol_name = getattr(
                getattr(handler, "service", None), "protocol", "Apple TV"
            )
            message = str(exc).strip() or type(exc).__name__
            raise AppleTVPairingError(
                f"{protocol_name} pairing failed: {message}"
            ) from exc
        finally:
            await self._close_handler(handler)
