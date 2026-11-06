import asyncio

import config
from engine import Engine
from intervention import AppleTVController


class FakeController:
    def __init__(self, fail_first_mute=False):
        self.mute_calls = 0
        self.unmute_calls = 0
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.fail_first_mute = fail_first_mute

    async def connect(self):
        self.connect_calls += 1

    async def disconnect(self):
        self.disconnect_calls += 1

    async def mute(self):
        self.mute_calls += 1
        if self.fail_first_mute and self.mute_calls == 1:
            raise ConnectionError("lost connection")

    async def unmute(self):
        self.unmute_calls += 1


def test_appletv_controller_uses_runtime_selected_device(monkeypatch):
    monkeypatch.setattr(config, "APPLE_TV_ID", "selected-at-runtime")

    controller = AppleTVController()

    assert controller._device_id == "selected-at-runtime"


def test_mute_and_unmute_are_idempotent_and_persist_recovery_marker(tmp_path, monkeypatch):
    marker = tmp_path / "muted.json"
    monkeypatch.setattr(config, "MUTED_STATE_FILE", str(marker))
    engine = Engine()
    controller = FakeController()
    engine._controller = controller

    async def exercise():
        assert await engine._mute_if_needed() is True
        assert await engine._mute_if_needed() is False
        assert marker.exists()
        assert await engine._unmute_if_needed() is True
        assert await engine._unmute_if_needed() is False

    asyncio.run(exercise())

    assert controller.mute_calls == 1
    assert controller.unmute_calls == 1
    assert not marker.exists()


def test_controller_action_reconnects_once(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MUTED_STATE_FILE", str(tmp_path / "muted.json"))
    engine = Engine()
    controller = FakeController(fail_first_mute=True)
    engine._controller = controller

    asyncio.run(engine._mute_if_needed())

    assert controller.mute_calls == 2
    assert controller.disconnect_calls == 1
    assert controller.connect_calls == 1


def test_cleanup_restores_volume_before_disconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MUTED_STATE_FILE", str(tmp_path / "muted.json"))
    engine = Engine()
    controller = FakeController()
    engine._controller = controller
    engine._muted_by_engine = True

    asyncio.run(engine._cleanup())

    assert controller.unmute_calls == 1
    assert controller.disconnect_calls == 1
    assert engine._controller is None


def test_restart_marker_restores_volume_and_clears_marker(tmp_path, monkeypatch):
    marker = tmp_path / "muted.json"
    marker.write_text('{"muted_at": 1}')
    monkeypatch.setattr(config, "MUTED_STATE_FILE", str(marker))
    engine = Engine()
    controller = FakeController()
    engine._controller = controller

    recovered = asyncio.run(engine._recover_interrupted_mute())

    assert recovered is True
    assert controller.unmute_calls == 1
    assert not marker.exists()
    assert engine._muted_by_engine is False


def test_disarm_sets_manual_override_cooldown(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MUTED_STATE_FILE", str(tmp_path / "muted.json"))
    monkeypatch.setattr(config, "MANUAL_OVERRIDE_COOLDOWN_SECONDS", 90)
    engine = Engine()
    engine._controller = FakeController()

    asyncio.run(engine.disarm())

    assert engine._manual_override_until > 0
    assert engine._events[-1]["reason"] == "manual_override"
