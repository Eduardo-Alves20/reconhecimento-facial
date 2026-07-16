from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.vision.outbox import VisionEventOutbox
from scripts.manage_vision_outbox import (
    OutboxAdminError,
    delete_terminal,
    list_items,
    requeue,
    summary,
)


def create_outbox(path: Path) -> VisionEventOutbox:
    outbox = VisionEventOutbox(path)
    outbox.initialize()
    return outbox


def payload(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "camera_id": "cam-ti-01",
        "user_id": "EMP001",
    }


def test_list_and_summary_do_not_return_payload(tmp_path: Path) -> None:
    path = tmp_path / "cam-ti-01.db"
    outbox = create_outbox(path)
    outbox.enqueue(payload("evt-001"))

    items = list_items(path, statuses=(), limit=10)

    assert summary(path)["PENDING"] == 1
    assert items[0]["event_id"] == "evt-001"
    assert "payload" not in items[0]


def test_requeue_requires_dead_and_records_operator_action(tmp_path: Path) -> None:
    path = tmp_path / "cam-ti-01.db"
    outbox = create_outbox(path)
    outbox.enqueue(payload("evt-dead"))
    outbox.mark_dead("evt-dead", error_code="HTTP_422", attempts=0)

    result = requeue(
        path,
        "evt-dead",
        confirmation="evt-dead",
        operator="ops.user",
        reason="cadastro corrigido",
        now=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert result["status"] == "RETRYING"
    assert summary(path)["RETRYING"] == 1
    with sqlite3.connect(path) as connection:
        action = connection.execute(
            """
            SELECT action, operator, reason, previous_error, attempts
            FROM vision_outbox_admin_actions
            """
        ).fetchone()
    assert action == ("REQUEUE", "ops.user", "cadastro corrigido", "HTTP_422", 1)


def test_requeue_rejects_wrong_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "cam-ti-01.db"
    outbox = create_outbox(path)
    outbox.enqueue(payload("evt-dead"))
    outbox.mark_dead("evt-dead", error_code="HTTP_400", attempts=0)

    with pytest.raises(OutboxAdminError, match="--confirm"):
        requeue(
            path,
            "evt-dead",
            confirmation="outro-evento",
            operator="ops.user",
            reason="cadastro corrigido",
        )


def test_delete_accepts_terminal_item_and_keeps_admin_audit(tmp_path: Path) -> None:
    path = tmp_path / "cam-ti-01.db"
    outbox = create_outbox(path)
    outbox.enqueue(payload("evt-sent"))
    outbox.mark_sent("evt-sent")

    result = delete_terminal(
        path,
        "evt-sent",
        confirmation="evt-sent",
        operator="ops.user",
        reason="retenção encerrada",
    )

    assert result["status"] == "DELETED"
    assert list_items(path, statuses=(), limit=10) == []
    with sqlite3.connect(path) as connection:
        action = connection.execute(
            "SELECT event_id, previous_status, action FROM vision_outbox_admin_actions"
        ).fetchone()
    assert action == ("evt-sent", "SENT", "DELETE")


def test_delete_rejects_pending_item(tmp_path: Path) -> None:
    path = tmp_path / "cam-ti-01.db"
    outbox = create_outbox(path)
    outbox.enqueue(payload("evt-pending"))

    with pytest.raises(OutboxAdminError, match="SENT ou DEAD"):
        delete_terminal(
            path,
            "evt-pending",
            confirmation="evt-pending",
            operator="ops.user",
            reason="não deveria remover",
        )
