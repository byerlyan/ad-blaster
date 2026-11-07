import config
from engine import Engine
from intervention import PlaybackMode


class StubDetector:
    def __init__(self):
        self.decay = None

    def set_break_decay(self, enabled):
        self.decay = enabled


def test_live_mode_enables_decay():
    e = Engine()
    e._detector = StubDetector()
    e._set_intervention_mode(PlaybackMode.LIVE)
    assert e._mode == PlaybackMode.LIVE
    assert e._detector.decay is True


def test_delayed_mode_disables_decay():
    e = Engine()
    e._detector = StubDetector()
    e._set_intervention_mode(PlaybackMode.DELAYED)
    assert e._detector.decay is False


def test_none_mode_disables_decay():
    e = Engine()
    e._detector = StubDetector()
    e._set_intervention_mode(None)
    assert e._mode is None
    assert e._detector.decay is False


def test_delayed_mute_only_keeps_wall_clock_decay():
    e = Engine()
    e._detector = StubDetector()
    e._set_intervention_mode(PlaybackMode.DELAYED, skipping=False)
    assert e._mode == PlaybackMode.DELAYED
    assert e._detector.decay is True


def test_mute_and_skip_are_independent(monkeypatch):
    monkeypatch.setattr(config, "AUTOMATIC_MUTING_ENABLED", True)
    monkeypatch.setattr(config, "AUTOMATIC_SKIPPING_ENABLED", False)
    assert Engine._automatic_intervention_actions(PlaybackMode.DELAYED) == (True, False)

    monkeypatch.setattr(config, "AUTOMATIC_MUTING_ENABLED", False)
    monkeypatch.setattr(config, "AUTOMATIC_SKIPPING_ENABLED", True)
    assert Engine._automatic_intervention_actions(PlaybackMode.DELAYED) == (False, True)
    assert Engine._automatic_intervention_actions(PlaybackMode.LIVE) == (False, False)
    assert Engine._automatic_intervention_actions(None) == (False, False)
