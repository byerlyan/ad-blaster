from copy import deepcopy

import pytest

from engine import Engine


def _sample(edge: float, template: float, color: float = 0.0) -> dict:
    return {"edge": edge, "tmpl": template, "color": color}


def test_suggestions_use_the_same_combined_logo_signal_as_detector():
    engine = Engine()
    engine._content_sessions = [
        [_sample(0.80, 0.05) for _ in range(12)],
        [_sample(0.90, 0.10) for _ in range(12)],
    ]
    engine._commercial_sessions = [
        [_sample(0.10, 0.04) for _ in range(12)],
        [_sample(0.20, 0.08) for _ in range(12)],
    ]

    result = engine.suggest_thresholds()

    assert result["ok"] is True
    assert result["logo_combined"]["available"] is True
    assert result["logo_combined"]["match"] == pytest.approx(0.5)
    assert result["logo_template"]["match"] < result["logo_combined"]["match"]


def test_generating_suggestions_does_not_mutate_current_samples():
    engine = Engine()
    engine._content_sessions = [[_sample(0.8, 0.7) for _ in range(12)]]
    engine._commercial_sessions = [[_sample(0.2, 0.1) for _ in range(12)]]
    before_content = deepcopy(engine._content_sessions)
    before_commercial = deepcopy(engine._commercial_sessions)

    first = engine.suggest_thresholds()
    second = engine.suggest_thresholds()

    assert first == second
    assert engine._content_sessions == before_content
    assert engine._commercial_sessions == before_commercial
