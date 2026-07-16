from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
_MAX_PAYLOAD_BYTES = 256 * 1024


class VisionOutboxError(RuntimeError):
    pass


class VisionOutboxIntegrityError(VisionOutboxError):
    pass


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("O instante da outbox precisa incluir fuso horário.")
    return value.astimezone(UTC).isoformat()


def _parse_iso(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise VisionOutboxIntegrityError(
            "A outbox contém um instante inválido."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VisionOutboxIntegrityError(
            "A outbox contém um instante sem fuso."
        )
    return parsed.astimezone(UTC)


class VisionEventOutbox:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        parent = self.database_path.parent
        if os.path.lexists(parent) and self._is_link_or_reparse(parent):
            raise VisionOutboxError("O diretório da outbox não pode ser um link.")
        parent.mkdir(parents=True, exist_ok=True)
        self._set_private_permissions(parent, directory=True)
        if os.path.lexists(self.database_path) and self._is_link_or_reparse(
            self.database_path
        ):
            raise VisionOutboxError("O banco da outbox não pode ser um link.")
        with self._connect() as connection:
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
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vision_sent_retention
                ON pending_vision_events(status, sent_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_worker_leases (
                    camera_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
        self._set_private_permissions(self.database_path, directory=False)

    def enqueue(self, payload: dict[str, Any], *, now: datetime | None = None) -> bool:
        if not isinstance(payload, dict):
            raise TypeError("payload precisa ser um objeto.")
        event_id = self._event_id(payload.get("event_id"))
        created_at = now or datetime.now(UTC)
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("payload da outbox não é serializável.") from exc
        if len(encoded.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise ValueError("payload da outbox excede o limite.")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO pending_vision_events
                    (event_id, payload, next_attempt_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (event_id, encoded, _iso(created_at), _iso(created_at)),
            )
            if cursor.rowcount == 1:
                return True
            existing = connection.execute(
                "SELECT payload FROM pending_vision_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
        try:
            existing_payload = (
                json.loads(existing["payload"])
                if existing is not None
                else None
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise VisionOutboxIntegrityError(
                "A outbox contém um payload inválido."
            ) from exc
        if existing_payload != payload:
            raise VisionOutboxIntegrityError(
                "event_id já existe na outbox com conteúdo diferente."
            )
        return False

    def due(
        self,
        *,
        now: datetime | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not 1 <= limit <= 1_000:
            raise ValueError("limit da outbox deve ficar entre 1 e 1000.")
        instant = now or datetime.now(UTC)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, payload, attempts, created_at
                FROM pending_vision_events
                WHERE status IN ('PENDING', 'RETRYING') AND next_attempt_at <= ?
                ORDER BY created_at, event_id LIMIT ?
                """,
                (_iso(instant), limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise VisionOutboxIntegrityError(
                    "A outbox contém um payload inválido."
                ) from exc
            if (
                not isinstance(payload, dict)
                or payload.get("event_id") != row["event_id"]
            ):
                raise VisionOutboxIntegrityError(
                    "A identidade de um item da outbox não confere."
                )
            attempts = int(row["attempts"])
            if attempts < 0:
                raise VisionOutboxIntegrityError(
                    "A outbox contém um contador inválido."
                )
            items.append(
                {
                    "event_id": row["event_id"],
                    "payload": payload,
                    "attempts": attempts,
                    "created_at": _parse_iso(row["created_at"]),
                }
            )
        return items

    def mark_sent(self, event_id: str, *, now: datetime | None = None) -> None:
        instant = now or datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE pending_vision_events
                SET status='SENT', sent_at=?, last_error=NULL
                WHERE event_id=? AND status IN ('PENDING', 'RETRYING')
                """,
                (_iso(instant), self._event_id(event_id)),
            )

    def mark_failure(
        self,
        event_id: str,
        *,
        error_code: str,
        attempts: int,
        now: datetime | None = None,
    ) -> None:
        if isinstance(attempts, bool) or attempts < 0:
            raise ValueError("attempts deve ser um inteiro não negativo.")
        instant = now or datetime.now(UTC)
        delay_seconds = min(300, 2 ** min(attempts + 1, 8))
        next_attempt = instant + timedelta(seconds=delay_seconds)
        safe_error = str(error_code).strip()[:100] or "UNKNOWN_ERROR"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE pending_vision_events
                SET status='RETRYING', attempts=?, next_attempt_at=?, last_error=?
                WHERE event_id=? AND status IN ('PENDING', 'RETRYING')
                """,
                (
                    attempts + 1,
                    _iso(next_attempt),
                    safe_error,
                    self._event_id(event_id),
                ),
            )

    def mark_dead(
        self,
        event_id: str,
        *,
        error_code: str,
        attempts: int,
    ) -> None:
        if isinstance(attempts, bool) or attempts < 0:
            raise ValueError("attempts deve ser um inteiro não negativo.")
        safe_error = str(error_code).strip()[:100] or "PERMANENT_ERROR"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE pending_vision_events
                SET status='DEAD', attempts=?, last_error=?
                WHERE event_id=? AND status IN ('PENDING', 'RETRYING')
                """,
                (
                    attempts + 1,
                    safe_error,
                    self._event_id(event_id),
                ),
            )

    def acquire_lease(
        self,
        *,
        camera_id: str,
        owner_id: str,
        now: datetime,
        ttl: timedelta = timedelta(seconds=90),
    ) -> bool:
        camera = self._event_id(camera_id)
        owner = self._event_id(owner_id)
        if ttl <= timedelta(0):
            raise ValueError("ttl do lease deve ser positivo.")
        instant = _parse_iso(_iso(now))
        expires_at = instant + ttl
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_id, expires_at FROM vision_worker_leases WHERE camera_id=?",
                (camera,),
            ).fetchone()
            if row is not None:
                current_expiry = _parse_iso(row["expires_at"])
                if row["owner_id"] != owner and instant < current_expiry:
                    return False
            connection.execute(
                """
                INSERT INTO vision_worker_leases(camera_id, owner_id, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    expires_at=excluded.expires_at
                """,
                (camera, owner, _iso(expires_at)),
            )
        return True

    def release_lease(self, *, camera_id: str, owner_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM vision_worker_leases WHERE camera_id=? AND owner_id=?",
                (self._event_id(camera_id), self._event_id(owner_id)),
            )

    def purge_sent(self, *, before: datetime) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM pending_vision_events
                WHERE status='SENT' AND sent_at IS NOT NULL AND sent_at < ?
                """,
                (_iso(before),),
            )
        return max(0, cursor.rowcount)

    def purge_dry_run(
        self,
        *,
        before: datetime,
        max_events: int,
    ) -> int:
        if isinstance(max_events, bool) or not 100 <= max_events <= 1_000_000:
            raise ValueError("max_events deve ficar entre 100 e 1000000.")
        removed = 0
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM pending_vision_events
                WHERE status IN ('PENDING', 'RETRYING') AND created_at < ?
                """,
                (_iso(before),),
            )
            removed += max(0, cursor.rowcount)
            overflow = int(
                connection.execute(
                    """
                    SELECT CASE
                        WHEN COUNT(*) > ? THEN COUNT(*) - ?
                        ELSE 0
                    END
                    FROM pending_vision_events
                    WHERE status IN ('PENDING', 'RETRYING')
                    """,
                    (max_events, max_events),
                ).fetchone()[0]
            )
            if overflow:
                cursor = connection.execute(
                    """
                    DELETE FROM pending_vision_events
                    WHERE event_id IN (
                        SELECT event_id
                        FROM pending_vision_events
                        WHERE status IN ('PENDING', 'RETRYING')
                        ORDER BY created_at, event_id
                        LIMIT ?
                    )
                    """,
                    (overflow,),
                )
                removed += max(0, cursor.rowcount)
        return removed

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS total "
                "FROM pending_vision_events GROUP BY status"
            ).fetchall()
        result: dict[str, int] = {}
        for row in rows:
            count = int(row["total"])
            status_value = str(row["status"])
            if count < 0 or status_value not in {
                "PENDING",
                "RETRYING",
                "SENT",
                "DEAD",
            }:
                raise VisionOutboxIntegrityError(
                    "A outbox contém uma contagem inválida."
                )
            result[status_value] = count
        return result

    @staticmethod
    def _event_id(value: Any) -> str:
        event_id = str(value or "").strip()
        if not _EVENT_ID_RE.fullmatch(event_id):
            raise ValueError("event_id da outbox é inválido.")
        return event_id

    @staticmethod
    def _set_private_permissions(path: Path, *, directory: bool) -> None:
        try:
            path.chmod(0o700 if directory else 0o600)
        except OSError:
            pass

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        try:
            metadata = path.lstat()
        except OSError:
            return True
        if stat.S_ISLNK(metadata.st_mode):
            return True
        attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attributes & reparse_flag)


__all__ = [
    "VisionEventOutbox",
    "VisionOutboxError",
    "VisionOutboxIntegrityError",
]
