"""Fila local persistente entre o worker de visão e a API de auditoria."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class VisionEventOutbox:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_vision_events (
                    event_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT NOT NULL,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    sent_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pending_vision_delivery
                ON pending_vision_events(status, next_attempt_at)
                """
            )

    def enqueue(self, payload: dict[str, Any], *, now: datetime | None = None) -> bool:
        event_id = str(payload.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("payload precisa de event_id")
        created_at = now or datetime.now(UTC)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO pending_vision_events
                    (event_id, payload, next_attempt_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (event_id, encoded, _iso(created_at), _iso(created_at)),
            )
        return cursor.rowcount == 1

    def due(self, *, now: datetime | None = None, limit: int = 20) -> list[dict[str, Any]]:
        instant = now or datetime.now(UTC)
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT event_id, payload, attempts
                FROM pending_vision_events
                WHERE status IN ('PENDING', 'RETRYING') AND next_attempt_at <= ?
                ORDER BY created_at LIMIT ?
                """,
                (_iso(instant), limit),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "payload": json.loads(row["payload"]),
                "attempts": row["attempts"],
            }
            for row in rows
        ]

    def mark_sent(self, event_id: str, *, now: datetime | None = None) -> None:
        instant = now or datetime.now(UTC)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE pending_vision_events
                SET status='SENT', sent_at=?, last_error=NULL
                WHERE event_id=?
                """,
                (_iso(instant), event_id),
            )

    def mark_failure(
        self,
        event_id: str,
        *,
        error_code: str,
        attempts: int,
        now: datetime | None = None,
    ) -> None:
        instant = now or datetime.now(UTC)
        delay_seconds = min(300, 2 ** min(attempts + 1, 8))
        next_attempt = instant + timedelta(seconds=delay_seconds)
        safe_error = error_code[:100]
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE pending_vision_events
                SET status='RETRYING', attempts=?, next_attempt_at=?, last_error=?
                WHERE event_id=?
                """,
                (attempts + 1, _iso(next_attempt), safe_error, event_id),
            )

    def counts(self) -> dict[str, int]:
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM pending_vision_events GROUP BY status"
            ).fetchall()
        return {status: count for status, count in rows}
