from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.vision.camera import CameraConfig
from app.vision.outbox import (
    VisionEventOutbox,
    VisionOutboxIntegrityError,
)
from app.vision.worker import (
    VisionWorkerSettings,
    _WorkerServices,
    flush_outbox,
)


NOW = datetime(2026, 7, 14, 15, 0, tzinfo=UTC)


def test_outbox_is_idempotent_and_marks_delivery(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    payload = {"event_id": "entry-1", "user_id": "EMP001"}
    assert outbox.enqueue(payload, now=NOW) is True
    assert outbox.enqueue(payload, now=NOW) is False
    assert outbox.due(now=NOW) == [
        {
            "event_id": "entry-1",
            "payload": payload,
            "attempts": 0,
            "created_at": NOW,
        }
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


def test_outbox_rejects_naive_time_and_invalid_event_id(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()

    with pytest.raises(ValueError, match="event_id"):
        outbox.enqueue({"event_id": "../escape"}, now=NOW)
    with pytest.raises(ValueError, match="fuso"):
        outbox.enqueue(
            {"event_id": "entry-3"},
            now=datetime(2026, 7, 14, 15, 0),
        )


def test_outbox_detects_payload_tampering(tmp_path) -> None:
    path = tmp_path / "vision.db"
    outbox = VisionEventOutbox(path)
    outbox.initialize()
    outbox.enqueue({"event_id": "entry-4"}, now=NOW)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE pending_vision_events SET payload=? WHERE event_id=?",
            ('{"event_id":"other"}', "entry-4"),
        )

    with pytest.raises(VisionOutboxIntegrityError, match="não confere"):
        outbox.due(now=NOW)


def test_sent_items_can_be_purged_after_retention(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    outbox.enqueue({"event_id": "entry-5"}, now=NOW)
    outbox.mark_sent("entry-5", now=NOW)

    assert outbox.purge_sent(before=NOW) == 0
    assert outbox.purge_sent(before=NOW + timedelta(seconds=1)) == 1
    assert outbox.counts() == {}


def test_dry_run_retention_limits_age_and_pending_count(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    for index in range(105):
        created_at = NOW if index < 3 else NOW + timedelta(days=1)
        outbox.enqueue(
            {"event_id": f"entry-dry-{index:03d}"},
            now=created_at,
        )

    removed = outbox.purge_dry_run(
        before=NOW + timedelta(seconds=1),
        max_events=100,
    )

    assert removed == 5
    assert outbox.counts() == {"PENDING": 100}
    remaining = outbox.due(now=NOW + timedelta(days=2), limit=100)
    assert remaining[0]["event_id"] == "entry-dry-005"


def test_duplicate_event_id_with_different_payload_is_rejected(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    outbox.enqueue({"event_id": "entry-6", "user_id": "EMP001"}, now=NOW)

    with pytest.raises(VisionOutboxIntegrityError, match="conteúdo diferente"):
        outbox.enqueue(
            {"event_id": "entry-6", "user_id": "EMP002"},
            now=NOW,
        )


def test_only_one_worker_can_hold_the_camera_lease(tmp_path) -> None:
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()

    assert outbox.acquire_lease(
        camera_id="cam-ti-01",
        owner_id="worker:one",
        now=NOW,
    )
    assert not outbox.acquire_lease(
        camera_id="cam-ti-01",
        owner_id="worker:two",
        now=NOW + timedelta(seconds=10),
    )
    assert outbox.acquire_lease(
        camera_id="cam-ti-01",
        owner_id="worker:two",
        now=NOW + timedelta(seconds=91),
    )


def test_flush_separates_retryable_and_permanent_api_failures(tmp_path) -> None:
    class FakeApi:
        def __init__(self, code: str) -> None:
            self.code = code

        def send(self, payload, *, queued_at):
            assert payload["event_id"]
            assert queued_at == NOW
            return False, self.code

    retry_outbox = VisionEventOutbox(tmp_path / "retry.db")
    retry_outbox.initialize()
    retry_outbox.enqueue({"event_id": "entry-retry"}, now=NOW)
    assert flush_outbox(retry_outbox, FakeApi("API_UNAVAILABLE"), now=NOW) == (
        0,
        1,
        0,
    )
    assert retry_outbox.counts() == {"RETRYING": 1}

    dead_outbox = VisionEventOutbox(tmp_path / "dead.db")
    dead_outbox.initialize()
    dead_outbox.enqueue({"event_id": "entry-dead"}, now=NOW)
    assert flush_outbox(dead_outbox, FakeApi("API_HTTP_422"), now=NOW) == (
        0,
        0,
        1,
    )
    assert dead_outbox.counts() == {"DEAD": 1}


def test_delivery_service_drains_outbox_without_camera_loop(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeApi:
        def __init__(self, base_url, api_key):
            assert base_url
            assert api_key

        def send(self, payload, *, queued_at):
            assert payload["event_id"] == "entry-background"
            assert payload["camera_id"] == "cam-ti-01"
            assert queued_at == NOW
            return True, "DELIVERED"

        def close(self):
            pass

    monkeypatch.setattr("app.vision.worker.AuditApiClient", FakeApi)
    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    outbox.enqueue(
        {
            "event_id": "entry-background",
            "camera_id": "cam-ti-01",
        },
        now=NOW,
    )
    settings = VisionWorkerSettings(
        camera=CameraConfig("127.0.0.1", "operator", "secret"),
        camera_id="cam-ti-01",
        room_id="sala-ti-01",
        api_base_url="http://127.0.0.1:8000",
        api_key="camera-secret",
        gallery_manifest=tmp_path / "manifest.json",
        calibration_path=tmp_path / "calibration.json",
        models_dir=tmp_path / "models",
        model_fingerprint="a" * 64,
        outbox_path=tmp_path / "vision.db",
        evidence_dir=tmp_path / "evidence",
        learned_path=tmp_path / "learned.db",
        dry_run=False,
    )
    processor = SimpleNamespace(outbox=outbox, evidence_store=None)
    service = _WorkerServices(processor, settings)

    service.start()
    service.wake()
    deadline = time.monotonic() + 2
    while outbox.counts() != {"SENT": 1} and time.monotonic() < deadline:
        time.sleep(0.01)
    service.close(drain=True)

    assert outbox.counts() == {"SENT": 1}


def test_delivery_never_sends_an_event_from_another_camera(tmp_path) -> None:
    class MustNotSend:
        def send(self, payload, *, queued_at):
            raise AssertionError((payload, queued_at))

    outbox = VisionEventOutbox(tmp_path / "vision.db")
    outbox.initialize()
    outbox.enqueue(
        {
            "event_id": "entry-other-camera",
            "camera_id": "cam-ti-02",
        },
        now=NOW,
    )

    result = flush_outbox(
        outbox,
        MustNotSend(),
        now=NOW,
        expected_camera_id="cam-ti-01",
    )

    assert result == (0, 0, 1)
    assert outbox.counts() == {"DEAD": 1}
