from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.vision.outbox import VisionEventOutbox


NOW = datetime(2026, 7, 14, 15, 0, tzinfo=UTC)


def test_outbox_is_idempotent_and_marks_delivery(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    payload = {"event_id": "entry-1", "user_id": "EMP001"}
    assert outbox.enqueue(payload, now=NOW) is True
    assert outbox.enqueue(payload, now=NOW) is False
    assert outbox.due(now=NOW) == [
        {"event_id": "entry-1", "payload": payload, "attempts": 0}
    ]
    outbox.mark_sent("entry-1", now=NOW)
    assert outbox.due(now=NOW + timedelta(days=1)) == []
    assert outbox.counts() == {"SENT": 1}


def test_failure_retries_without_storing_exception_details(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    outbox.enqueue({"event_id": "entry-2"}, now=NOW)
    outbox.mark_failure(
        "entry-2",
        error_code="API_UNAVAILABLE:" + "x" * 200,
        attempts=0,
        now=NOW,
    )
    assert outbox.due(now=NOW + timedelta(seconds=1)) == []
    assert outbox.due(now=NOW + timedelta(seconds=2))[0]["attempts"] == 1
    assert outbox.counts() == {"RETRYING": 1}
