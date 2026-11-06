"""Durable one-tap program/commercial corrections and audit history."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class CorrectionStore:
    def __init__(self, path: str, duplicate_window_seconds: float = 2.0):
        self.path = str(path)
        self.duplicate_window_seconds = float(duplicate_window_seconds)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS corrections (
                    id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    label TEXT NOT NULL CHECK(label IN ('program', 'commercial')),
                    channel TEXT,
                    action TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    undone_at REAL
                )"""
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_correction_time ON corrections(timestamp DESC)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["evidence"] = json.loads(item.pop("evidence_json"))
        item["undone"] = item["undone_at"] is not None
        return item

    def add(
        self,
        *,
        label: str,
        session_id: str,
        channel: str | None,
        action: str,
        evidence: dict[str, Any],
        timestamp: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if label not in {"program", "commercial"}:
            raise ValueError("label must be program or commercial")
        if action not in {"none", "mute", "unmute"}:
            raise ValueError("action must be none, mute, or unmute")
        now = time.time() if timestamp is None else float(timestamp)
        with self._connect() as db:
            prior = db.execute(
                """SELECT * FROM corrections
                WHERE session_id = ? AND label = ? AND undone_at IS NULL
                ORDER BY timestamp DESC LIMIT 1""",
                (session_id, label),
            ).fetchone()
            if prior is not None and now - prior["timestamp"] < self.duplicate_window_seconds:
                return self._decode(prior), False
            correction_id = str(uuid.uuid4())
            db.execute(
                """INSERT INTO corrections
                (id, timestamp, session_id, label, channel, action, evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    correction_id,
                    now,
                    session_id,
                    label,
                    channel,
                    action,
                    json.dumps(evidence, sort_keys=True, separators=(",", ":")),
                ),
            )
            row = db.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,)).fetchone()
        return self._decode(row), True

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM corrections ORDER BY timestamp DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def relabel(self, correction_id: str, label: str) -> dict[str, Any] | None:
        if label not in {"program", "commercial"}:
            raise ValueError("label must be program or commercial")
        with self._connect() as db:
            result = db.execute(
                "UPDATE corrections SET label = ?, undone_at = NULL WHERE id = ?",
                (label, correction_id),
            )
            row = db.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,)).fetchone()
        return self._decode(row) if result.rowcount and row is not None else None

    def undo(self, correction_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            result = db.execute(
                "UPDATE corrections SET undone_at = ? WHERE id = ? AND undone_at IS NULL",
                (time.time(), correction_id),
            )
            row = db.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,)).fetchone()
        return self._decode(row) if result.rowcount and row is not None else None
