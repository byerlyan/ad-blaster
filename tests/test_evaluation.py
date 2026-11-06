from evaluation import compare_modes, evaluate_mode


def rows():
    # 10 seconds program, 20 seconds commercial, 10 seconds program.
    values = []
    for timestamp in range(40):
        commercial = 10 <= timestamp < 30
        values.append(
            {
                "timestamp": timestamp,
                "label": "commercial" if commercial else "program",
                "legacy": "commercial" if 12 <= timestamp < 32 else "program",
                "hybrid": "commercial" if 11 <= timestamp < 30 else "program",
            }
        )
    return values


def test_evaluation_reports_latency_false_mute_and_coverage():
    legacy = evaluate_mode(rows(), "legacy")

    assert legacy["commercial_precision"] == 0.9
    assert legacy["commercial_recall"] == 0.9
    assert legacy["commercial_audio_muted_percent"] == 90.0
    assert legacy["mute_start_latency_seconds_mean"] == 2.0
    assert legacy["unmute_latency_seconds_mean"] == 2.0
    assert legacy["false_muted_programming_seconds_per_hour"] == 180.0
    assert legacy["state_transition_count"] == 2


def test_comparison_distinguishes_simulated_results():
    report = compare_modes(rows(), data_kind="simulated")

    assert report["results_are_simulated"] is True
    assert report["hybrid"]["commercial_recall"] > report["legacy"]["commercial_recall"]
