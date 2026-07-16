"""Inspeciona e administra uma outbox de visão sem exibir payloads."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sqlite3
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
OPERATOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,99}$")
STATUSES = ("PENDING", "RETRYING", "SENT", "DEAD")
ACTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS vision_outbox_admin_actions (
    action_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    previous_status TEXT NOT NULL,
    previous_error TEXT,
    attempts INTEGER NOT NULL,
    action TEXT NOT NULL,
    operator TEXT NOT NULL,
    reason TEXT NOT NULL,
    occurred_at TEXT NOT NULL
)
"""


class OutboxAdminError(RuntimeError):
    pass


def _identifier(value: str, *, label: str, pattern: re.Pattern[str]) -> str:
    normalized = value.strip()
    if not pattern.fullmatch(normalized):
        raise OutboxAdminError(f"{label} inválido")
    return normalized


def _reason(value: str) -> str:
    normalized = value.strip()
    if not 3 <= len(normalized) <= 300:
        raise OutboxAdminError("reason deve ter entre 3 e 300 caracteres")
    return normalized


@contextmanager
def connect_outbox(path: Path) -> Iterator[sqlite3.Connection]:
    database = path.expanduser()
    if database.is_symlink() or not database.is_file():
        raise OutboxAdminError(f"outbox não encontrada ou insegura: {database}")
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='pending_vision_events'
            """
        ).fetchone()
        if table is None:
            raise OutboxAdminError("arquivo não contém uma outbox de visão")
        yield connection
    finally:
        connection.close()


def summary(path: Path) -> dict[str, int]:
    with connect_outbox(path) as connection:
        rows = connection.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM pending_vision_events GROUP BY status
            """
        ).fetchall()
    result = {status: 0 for status in STATUSES}
    for row in rows:
        status = str(row["status"])
        if status not in result:
            raise OutboxAdminError(f"status desconhecido na outbox: {status}")
        result[status] = int(row["total"])
    return result


def list_items(
    path: Path,
    *,
    statuses: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 1_000:
        raise OutboxAdminError("limit deve ficar entre 1 e 1000")
    selected = statuses or STATUSES
    if any(status not in STATUSES for status in selected):
        raise OutboxAdminError("status inválido")
    placeholders = ", ".join("?" for _ in selected)
    with connect_outbox(path) as connection:
        rows = connection.execute(
            f"""
            SELECT event_id, status, attempts, next_attempt_at, last_error,
                   created_at, sent_at
            FROM pending_vision_events
            WHERE status IN ({placeholders})
            ORDER BY created_at, event_id
            LIMIT ?
            """,
            (*selected, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def _record_action(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    previous_status: str,
    previous_error: str | None,
    attempts: int,
    action: str,
    operator: str,
    reason: str,
    now: datetime,
) -> None:
    connection.execute(ACTION_SCHEMA)
    connection.execute(
        """
        INSERT INTO vision_outbox_admin_actions
            (action_id, event_id, previous_status, previous_error, attempts,
             action, operator, reason, occurred_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            secrets.token_hex(16),
            event_id,
            previous_status,
            previous_error,
            attempts,
            action,
            operator,
            reason,
            now.astimezone(UTC).isoformat(),
        ),
    )


def requeue(
    path: Path,
    event_id: str,
    *,
    confirmation: str,
    operator: str,
    reason: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    event = _identifier(event_id, label="event_id", pattern=EVENT_ID_RE)
    if confirmation != event:
        raise OutboxAdminError("--confirm deve repetir exatamente o event_id")
    actor = _identifier(operator, label="operator", pattern=OPERATOR_RE)
    justification = _reason(reason)
    instant = now or datetime.now(UTC)
    with connect_outbox(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT status, last_error, attempts
            FROM pending_vision_events WHERE event_id=?
            """,
            (event,),
        ).fetchone()
        if row is None:
            raise OutboxAdminError("evento não encontrado")
        if row["status"] != "DEAD":
            raise OutboxAdminError("somente eventos DEAD podem ser reenfileirados")
        connection.execute(
            """
            UPDATE pending_vision_events
            SET status='RETRYING', next_attempt_at=?, last_error='MANUAL_REQUEUE'
            WHERE event_id=?
            """,
            (instant.astimezone(UTC).isoformat(), event),
        )
        _record_action(
            connection,
            event_id=event,
            previous_status="DEAD",
            previous_error=row["last_error"],
            attempts=int(row["attempts"]),
            action="REQUEUE",
            operator=actor,
            reason=justification,
            now=instant,
        )
        connection.commit()
    return {"event_id": event, "previous_status": "DEAD", "status": "RETRYING"}


def delete_terminal(
    path: Path,
    event_id: str,
    *,
    confirmation: str,
    operator: str,
    reason: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    event = _identifier(event_id, label="event_id", pattern=EVENT_ID_RE)
    if confirmation != event:
        raise OutboxAdminError("--confirm deve repetir exatamente o event_id")
    actor = _identifier(operator, label="operator", pattern=OPERATOR_RE)
    justification = _reason(reason)
    instant = now or datetime.now(UTC)
    with connect_outbox(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT status, last_error, attempts
            FROM pending_vision_events WHERE event_id=?
            """,
            (event,),
        ).fetchone()
        if row is None:
            raise OutboxAdminError("evento não encontrado")
        previous_status = str(row["status"])
        if previous_status not in {"SENT", "DEAD"}:
            raise OutboxAdminError("somente eventos SENT ou DEAD podem ser excluídos")
        _record_action(
            connection,
            event_id=event,
            previous_status=previous_status,
            previous_error=row["last_error"],
            attempts=int(row["attempts"]),
            action="DELETE",
            operator=actor,
            reason=justification,
            now=instant,
        )
        connection.execute(
            "DELETE FROM pending_vision_events WHERE event_id=?",
            (event,),
        )
        connection.commit()
    return {"event_id": event, "previous_status": previous_status, "status": "DELETED"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="json_output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="conta itens por status")
    list_parser = subparsers.add_parser("list", help="lista somente metadados")
    list_parser.add_argument("--status", action="append", choices=STATUSES, default=[])
    list_parser.add_argument("--limit", type=int, default=100)

    for command, help_text in (
        ("requeue", "recoloca um item DEAD na fila"),
        ("delete", "exclui um item SENT ou DEAD"),
    ):
        action_parser = subparsers.add_parser(command, help=help_text)
        action_parser.add_argument("event_id")
        action_parser.add_argument("--confirm", required=True)
        action_parser.add_argument("--operator", required=True)
        action_parser.add_argument("--reason", required=True)
    return parser


def execute(args: argparse.Namespace) -> Any:
    if args.command == "summary":
        return summary(args.database)
    if args.command == "list":
        return list_items(
            args.database,
            statuses=tuple(args.status),
            limit=args.limit,
        )
    options = {
        "confirmation": args.confirm,
        "operator": args.operator,
        "reason": args.reason,
    }
    if args.command == "requeue":
        return requeue(args.database, args.event_id, **options)
    if args.command == "delete":
        return delete_terminal(args.database, args.event_id, **options)
    raise OutboxAdminError("comando inválido")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except (OutboxAdminError, OSError, sqlite3.Error) as exc:
        if args.json_output:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"Erro: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif isinstance(result, list):
        if not result:
            print("Nenhum item encontrado.")
        for item in result:
            print(
                f"{item['event_id']}  {item['status']}  tentativas={item['attempts']}  "
                f"criado={item['created_at']}  erro={item['last_error'] or '-'}"
            )
    elif args.command == "summary":
        for status in STATUSES:
            print(f"{status}: {result[status]}")
    else:
        print(f"{result['event_id']}: {result['previous_status']} -> {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
