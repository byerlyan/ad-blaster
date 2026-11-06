"""Local-network dashboard pairing and server-side trusted sessions."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_COOKIE = "clicker_session"


class DashboardAuth:
    """Short pairing code -> opaque HttpOnly cookie with durable revocation.

    Raw session secrets are never stored in SQLite. CSRF secrets are returned
    through an authenticated same-origin JSON request and kept in page memory.
    """

    def __init__(
        self,
        path: str,
        *,
        enabled: bool,
        idle_seconds: int = 1800,
        absolute_seconds: int = 28800,
        pairing_ttl_seconds: int = 600,
    ):
        self.path = str(path)
        self.enabled = bool(enabled)
        self.idle_seconds = max(1, int(idle_seconds))
        self.absolute_seconds = max(self.idle_seconds, int(absolute_seconds))
        self.pairing_ttl_seconds = max(30, int(pairing_ttl_seconds))
        self._pairing_code = ""
        self._pairing_expires = 0.0
        self._failed_pairing_attempts: list[float] = []
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if self.enabled:
            self.rotate_pairing_code()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS dashboard_sessions (
                    id TEXT PRIMARY KEY,
                    verifier_hash TEXT NOT NULL,
                    csrf_hash TEXT NOT NULL,
                    csrf_token TEXT NOT NULL,
                    device_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_at REAL
                )"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(dashboard_sessions)")}
            if "csrf_token" not in columns:
                db.execute(
                    "ALTER TABLE dashboard_sessions ADD COLUMN csrf_token TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _split(token: str) -> tuple[str, str] | None:
        try:
            session_id, verifier = token.split(".", 1)
        except ValueError:
            return None
        if not session_id or not verifier or len(session_id) > 128 or len(verifier) > 256:
            return None
        return session_id, verifier

    def rotate_pairing_code(self) -> str:
        self._pairing_code = f"{secrets.randbelow(1_000_000):06d}"
        self._pairing_expires = time.time() + self.pairing_ttl_seconds
        logger.warning(
            "The Clicker LAN pairing code is %s (valid for %d minutes)",
            self._pairing_code,
            self.pairing_ttl_seconds // 60,
        )
        return self._pairing_code

    def pairing_code_for_host(self) -> str:
        if time.time() >= self._pairing_expires:
            return self.rotate_pairing_code()
        return self._pairing_code

    def pair(self, code: str, device_name: str) -> tuple[str, str, dict]:
        now = time.time()
        self._failed_pairing_attempts = [
            attempted for attempted in self._failed_pairing_attempts if now - attempted < 60
        ]
        if len(self._failed_pairing_attempts) >= 8:
            raise ValueError("Too many pairing attempts; wait one minute")
        if now >= self._pairing_expires:
            self.rotate_pairing_code()
            raise ValueError("Pairing code expired; use the new code shown on the host")
        if not hmac.compare_digest(str(code).strip(), self._pairing_code):
            self._failed_pairing_attempts.append(now)
            raise ValueError("Invalid pairing code")

        session_id = secrets.token_urlsafe(18)
        verifier = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        expires_at = now + self.absolute_seconds
        clean_name = (device_name or "Browser").strip()[:80] or "Browser"
        with self._connect() as db:
            db.execute(
                """INSERT INTO dashboard_sessions
                (id, verifier_hash, csrf_hash, csrf_token, device_name, created_at, last_seen, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    self._hash(verifier),
                    self._hash(csrf),
                    csrf,
                    clean_name,
                    now,
                    now,
                    expires_at,
                ),
            )
        self.rotate_pairing_code()  # one-time use
        self._failed_pairing_attempts.clear()
        return f"{session_id}.{verifier}", csrf, {
            "id": session_id,
            "device_name": clean_name,
            "created_at": now,
            "expires_at": expires_at,
        }

    def authenticate(self, token: str | None, *, touch: bool = True) -> dict | None:
        parts = self._split(token or "")
        if parts is None:
            return None
        session_id, verifier = parts
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM dashboard_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            now = time.time()
            expired = now >= row["expires_at"] or now - row["last_seen"] >= self.idle_seconds
            valid = hmac.compare_digest(row["verifier_hash"], self._hash(verifier))
            if expired or not valid:
                if expired:
                    db.execute(
                        "UPDATE dashboard_sessions SET revoked_at = ? WHERE id = ?",
                        (now, session_id),
                    )
                return None
            if touch:
                db.execute(
                    "UPDATE dashboard_sessions SET last_seen = ? WHERE id = ?",
                    (now, session_id),
                )
        return dict(row)

    def csrf_valid(self, session: dict, csrf: str | None) -> bool:
        return bool(csrf) and hmac.compare_digest(
            str(session["csrf_hash"]), self._hash(str(csrf))
        )

    def csrf_for_token(self, token: str | None) -> str | None:
        session = self.authenticate(token, touch=False)
        if session is None:
            return None
        return str(session.get("csrf_token") or "") or None

    def trusted_devices(self) -> list[dict]:
        now = time.time()
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, device_name, created_at, last_seen, expires_at, revoked_at
                FROM dashboard_sessions ORDER BY created_at DESC"""
            ).fetchall()
        return [
            {
                **dict(row),
                "active": (
                    row["revoked_at"] is None
                    and row["expires_at"] > now
                    and now - row["last_seen"] < self.idle_seconds
                ),
            }
            for row in rows
        ]

    def revoke(self, session_id: str) -> bool:
        with self._connect() as db:
            result = db.execute(
                "UPDATE dashboard_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (time.time(), session_id),
            )
        return bool(result.rowcount)

    def revoke_all(self) -> int:
        with self._connect() as db:
            result = db.execute(
                "UPDATE dashboard_sessions SET revoked_at = ? WHERE revoked_at IS NULL",
                (time.time(),),
            )
        return int(result.rowcount)

    def csrf_for_session_for_tests(self, session_id: str, csrf: str) -> bool:
        """Validate, never retrieve, a CSRF value (also useful to API tests)."""
        with self._connect() as db:
            row = db.execute(
                "SELECT csrf_hash FROM dashboard_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return bool(row) and hmac.compare_digest(row["csrf_hash"], self._hash(csrf))
