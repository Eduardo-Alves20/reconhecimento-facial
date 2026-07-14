from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any

import httpx
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.config import Settings
from app.main import create_app


def _anomaly_payload(
    event_payload: Callable[..., dict[str, Any]], event_id: str
) -> dict[str, Any]:
    return event_payload(
        event_id=event_id,
        user_id="EMP003",
        timestamp="2026-07-19T23:00:00-03:00",
    )


def _outbox_rows(client: TestClient, event_id: str) -> list[dict[str, Any]]:
    repository = client.app.state.repository
    with repository.connect() as connection:
        rows = connection.execute(
            """
            SELECT alert_id, event_id, status, attempts, next_attempt_at,
                   last_error, created_at, sent_at
            FROM alert_outbox
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchall()
    return [dict(row) for row in rows]


class FakeAsyncClient:
    """Small async httpx stand-in used by the outbox delivery tests."""

    def __init__(
        self,
        calls: list[dict[str, Any]],
        *,
        failure: Exception | None = None,
        status_code: int = 202,
    ) -> None:
        self.calls = calls
        self.failure = failure
        self.status_code = status_code

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self.failure is not None:
            raise self.failure
        request = httpx.Request("POST", url)
        return httpx.Response(self.status_code, request=request)


def test_anomaly_creates_exactly_one_outbox_alert_even_after_replay(
    client: TestClient,
    camera_headers: dict[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    payload = _anomaly_payload(event_payload, "evt-outbox-single")

    created = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=payload
    )
    replayed = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=payload
    )

    assert created.status_code == 201, created.text
    assert replayed.status_code == 200, replayed.text
    rows = _outbox_rows(client, payload["event_id"])
    assert len(rows) == 1
    assert rows[0]["event_id"] == payload["event_id"]
    assert rows[0]["alert_id"] == created.json()["event"]["alert"]["alert_id"]


def test_worker_without_webhook_marks_not_configured_and_preserves_event(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    event_id = "evt-outbox-not-configured"
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=_anomaly_payload(event_payload, event_id),
    )

    assert response.status_code == 201, response.text
    saved = client.get(f"/v1/access-events/{event_id}", auth=admin_auth)
    assert saved.status_code == 200
    event = saved.json()
    assert event["event_id"] == event_id
    assert event["decision"] == "ANOMALY"
    assert event["alert_required"] is True
    assert event["alert"]["status"] == "NOT_CONFIGURED"
    assert event["alert"]["attempts"] == 0
    assert "não configurado" in event["alert"]["last_error"]
    assert len(_outbox_rows(client, event_id)) == 1


def test_temporary_delivery_failure_keeps_and_reschedules_alert_without_changing_decision(
    settings: Settings,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    request = httpx.Request("POST", "https://alerts.test/webhook")
    failure = httpx.ConnectError("temporary failure", request=request)

    def fake_client(*args: object, **kwargs: object) -> FakeAsyncClient:
        return FakeAsyncClient(calls, failure=failure)

    monkeypatch.setattr("app.alerts.httpx.AsyncClient", fake_client)
    custom_app = create_app(
        replace(settings, alert_webhook_url="https://alerts.test/webhook")
    )
    event_id = "evt-outbox-retry"

    with TestClient(custom_app) as client:
        response = client.post(
            "/v1/webhooks/access-events",
            headers=camera_headers,
            json=_anomaly_payload(event_payload, event_id),
        )
        assert response.status_code == 201, response.text

        event = client.get(f"/v1/access-events/{event_id}", auth=admin_auth).json()
        rows = _outbox_rows(client, event_id)

    assert len(calls) == 1
    assert event["decision"] == "ANOMALY"
    assert event["risk_level"] == "HIGH"
    assert event["alert_required"] is True
    assert event["alert"]["status"] == "RETRYING"
    assert event["alert"]["attempts"] == 1
    assert len(rows) == 1
    assert rows[0]["status"] == "RETRYING"
    assert rows[0]["attempts"] == 1
    assert rows[0]["last_error"]
    assert datetime.fromisoformat(rows[0]["next_attempt_at"]) > datetime.fromisoformat(
        rows[0]["created_at"]
    )


def test_successful_delivery_marks_sent_and_uses_event_id_as_idempotency_key(
    settings: Settings,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_client(*args: object, **kwargs: object) -> FakeAsyncClient:
        return FakeAsyncClient(calls, status_code=202)

    monkeypatch.setattr("app.alerts.httpx.AsyncClient", fake_client)
    webhook_url = "https://alerts.test/webhook"
    custom_app = create_app(replace(settings, alert_webhook_url=webhook_url))
    event_id = "evt-outbox-sent"

    with TestClient(custom_app) as client:
        response = client.post(
            "/v1/webhooks/access-events",
            headers=camera_headers,
            json=_anomaly_payload(event_payload, event_id),
        )
        assert response.status_code == 201, response.text

        event = client.get(f"/v1/access-events/{event_id}", auth=admin_auth).json()
        rows = _outbox_rows(client, event_id)

    assert len(calls) == 1
    assert calls[0]["url"] == webhook_url
    assert calls[0]["headers"] == {"Idempotency-Key": event_id}
    assert calls[0]["json"]["event_id"] == event_id
    assert calls[0]["json"]["dedupe_key"] == f"access:{event_id}"
    assert set(calls[0]["json"]["person"]) == {"external_id"}
    assert "narrative" not in calls[0]["json"]
    assert "recognition_confidence" not in calls[0]["json"]
    assert event["decision"] == "ANOMALY"
    assert event["alert"]["status"] == "SENT"
    assert event["alert"]["attempts"] == 1
    assert event["alert"]["sent_at"] is not None
    assert len(rows) == 1
    assert rows[0]["status"] == "SENT"
    assert rows[0]["attempts"] == 1
    assert rows[0]["sent_at"] is not None
