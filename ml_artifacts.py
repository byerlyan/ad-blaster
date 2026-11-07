"""Dependency-light ML feature schema, dataset storage, and model artifact types.

This module deliberately avoids importing OpenCV and scikit-learn. Dataset
export should work without the live video stack, while scikit-learn objects are
supplied by the offline trainer when a model is created.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping


FEATURE_NAMES = (
    "visual_edge",
    "visual_template",
    "visual_color",
    "audio_rms_dbfs",
    "audio_dynamic_range_db",
    "audio_silence_ratio",
    "audio_spectral_centroid_hz",
    "audio_spectral_flux",
    "audio_loudness_change_db",
    "scene_cut",
    "recent_scene_cut_rate",
    "recent_visual_mean",
    "recent_audio_energy_mean",
    *tuple(f"mfcc_{i}" for i in range(13)),
)


class FeatureDatasetStore:
    """Durable labeled features grouped by marking/recording session."""

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS feature_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    label INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    source_group TEXT NOT NULL,
                    features_json TEXT NOT NULL
                )"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_session ON feature_samples(session_id)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def append(
        self,
        *,
        timestamp: float,
        label: str | int,
        session_id: str,
        source_group: str,
        features: Mapping[str, float],
    ) -> None:
        numeric_label = int(label) if isinstance(label, int) else int(label == "commercial")
        normalized = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
        with self._connect() as db:
            db.execute(
                """INSERT INTO feature_samples
                (timestamp, label, session_id, source_group, features_json)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    float(timestamp),
                    numeric_label,
                    session_id,
                    source_group,
                    json.dumps(normalized, sort_keys=True),
                ),
            )

    def rows(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM feature_samples ORDER BY id").fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "label": row["label"],
                "session_id": row["session_id"],
                "source_group": row["source_group"],
                "features": json.loads(row["features_json"]),
            }
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM feature_samples").fetchone()[0])

    def export_csv(self, destination: str) -> int:
        rows = self.rows()
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["timestamp", "label", "session_id", "source_group", *FEATURE_NAMES],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "timestamp": row["timestamp"],
                        "label": row["label"],
                        "session_id": row["session_id"],
                        "source_group": row["source_group"],
                        **row["features"],
                    }
                )
        return len(rows)


class PlattCalibratedModel:
    """Serializable wrapper around a scaled linear model and Platt calibrator."""

    def __init__(self, scaler, classifier, calibrator):
        self.scaler = scaler
        self.classifier = classifier
        self.calibrator = calibrator

    def predict_proba(self, values):
        scaled = self.scaler.transform(values)
        scores = self.classifier.decision_function(scaled).reshape(-1, 1)
        return self.calibrator.predict_proba(scores)
