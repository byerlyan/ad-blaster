#!/usr/bin/env python3
"""Train and evaluate the optional lightweight classifier offline.

Splits are group-aware: adjacent samples from the same source/marking session
cannot cross fit, calibration, and test partitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

try:
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing training dependency {exc.name!r}. Run this script with The Clicker's "
        "virtual-environment Python after installing requirements-ml.txt; for a source "
        "checkout, use: ./.venv/bin/python3 -m pip install -r requirements-ml.txt"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_artifacts import FEATURE_NAMES, PlattCalibratedModel  # noqa: E402


def load_csv(path: str):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError("dataset is empty")
    x = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.array([int(row["label"]) for row in rows], dtype=np.int32)
    groups = np.array(
        [f"{row.get('source_group', 'unknown')}::{row['session_id']}" for row in rows]
    )
    return x, y, groups


def grouped_split(x, y, groups, test_size, seed):
    for attempt in range(50):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        train_index, test_index = next(splitter.split(x, y, groups))
        if len(np.unique(y[train_index])) == 2 and len(np.unique(y[test_index])) == 2:
            return train_index, test_index
    raise ValueError("could not create group-separated partitions containing both labels")


def train(dataset: str, output: str, seed: int = 42) -> dict:
    x, y, groups = load_csv(dataset)
    if len(np.unique(groups)) < 4:
        raise ValueError("need at least four distinct source/session groups")
    train_cal, test = grouped_split(x, y, groups, 0.25, seed)
    fit_rel, calibration_rel = grouped_split(
        x[train_cal], y[train_cal], groups[train_cal], 0.25, seed + 100
    )
    fit = train_cal[fit_rel]
    calibration = train_cal[calibration_rel]

    scaler = StandardScaler().fit(x[fit])
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
    ).fit(scaler.transform(x[fit]), y[fit])
    calibration_scores = classifier.decision_function(
        scaler.transform(x[calibration])
    ).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=300, random_state=seed).fit(
        calibration_scores, y[calibration]
    )
    model = PlattCalibratedModel(scaler, classifier, calibrator)
    probabilities = model.predict_proba(x[test])[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int32)
    digest = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()[:12]
    metadata = {
        "model_version": f"linear-platt-{int(time.time())}-{digest}",
        "trained_at": time.time(),
        "algorithm": "standardized_logistic_regression_with_platt_calibration",
        "dataset_sha256_prefix": digest,
        "sample_count": int(len(y)),
        "group_count": int(len(np.unique(groups))),
        "fit_groups": sorted(set(groups[fit].tolist())),
        "calibration_groups": sorted(set(groups[calibration].tolist())),
        "test_groups": sorted(set(groups[test].tolist())),
        "commercial_precision": float(precision_score(y[test], predictions, zero_division=0)),
        "commercial_recall": float(recall_score(y[test], predictions, zero_division=0)),
        "results_are_simulated": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": list(FEATURE_NAMES), "metadata": metadata},
        output,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="CSV exported by The Clicker")
    parser.add_argument("--output", required=True, help="Destination .joblib artifact")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
