"""Time-weighted commercial detection evaluation metrics."""

from __future__ import annotations

import statistics
from typing import Any, Sequence


def _commercial(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "commercial", "ad"}


def evaluate_mode(rows: Sequence[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["timestamp"]))
    if len(ordered) < 2:
        raise ValueError("evaluation requires at least two timestamped rows")
    steps = [
        max(0.0, float(right["timestamp"]) - float(left["timestamp"]))
        for left, right in zip(ordered, ordered[1:])
    ]
    default_step = statistics.median(step for step in steps if step > 0) if any(steps) else 1.0
    durations = steps + [default_step]
    tp = fp = fn = tn = 0.0
    for row, duration in zip(ordered, durations):
        truth = _commercial(row["label"])
        predicted = _commercial(row[prediction_key])
        if truth and predicted:
            tp += duration
        elif truth:
            fn += duration
        elif predicted:
            fp += duration
        else:
            tn += duration
    total = tp + fp + fn + tn
    commercial_total = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / commercial_total if commercial_total else 0.0

    starts: list[float] = []
    ends: list[float] = []
    previous_truth = False
    for row in ordered:
        truth = _commercial(row["label"])
        timestamp = float(row["timestamp"])
        if truth and not previous_truth:
            starts.append(timestamp)
        elif previous_truth and not truth:
            ends.append(timestamp)
        previous_truth = truth

    mute_latencies: list[float] = []
    for start in starts:
        later_end = next((end for end in ends if end > start), float("inf"))
        detection = next(
            (
                float(row["timestamp"])
                for row in ordered
                if start <= float(row["timestamp"]) < later_end
                and _commercial(row[prediction_key])
            ),
            None,
        )
        if detection is not None:
            mute_latencies.append(detection - start)

    unmute_latencies: list[float] = []
    for end in ends:
        resumed = next(
            (
                float(row["timestamp"])
                for row in ordered
                if float(row["timestamp"]) >= end
                and not _commercial(row[prediction_key])
            ),
            None,
        )
        if resumed is not None:
            unmute_latencies.append(resumed - end)

    transitions = 0
    previous_prediction = _commercial(ordered[0][prediction_key])
    for row in ordered[1:]:
        prediction = _commercial(row[prediction_key])
        transitions += int(prediction != previous_prediction)
        previous_prediction = prediction

    return {
        "commercial_precision": precision,
        "commercial_recall": recall,
        "false_muted_programming_seconds_per_hour": fp / total * 3600.0 if total else 0.0,
        "commercial_audio_muted_percent": tp / commercial_total * 100.0 if commercial_total else 0.0,
        "mute_start_latency_seconds_mean": statistics.mean(mute_latencies) if mute_latencies else None,
        "unmute_latency_seconds_mean": statistics.mean(unmute_latencies) if unmute_latencies else None,
        "state_transition_count": transitions,
        "evaluated_seconds": total,
        "commercial_seconds": commercial_total,
        "false_muted_seconds": fp,
    }


def compare_modes(rows: Sequence[dict[str, Any]], *, data_kind: str) -> dict[str, Any]:
    if data_kind not in {"captured", "simulated"}:
        raise ValueError("data_kind must be 'captured' or 'simulated'")
    return {
        "data_kind": data_kind,
        "results_are_simulated": data_kind == "simulated",
        "legacy": evaluate_mode(rows, "legacy"),
        "hybrid": evaluate_mode(rows, "hybrid"),
    }
