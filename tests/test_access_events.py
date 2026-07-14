from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    (
        "payload_overrides",
        "expected_decision",
        "expected_risk",
        "expected_reasons",
        "expected_sources",
        "narrative_facts",
        "alert_required",
    ),
    [
        pytest.param(
            {
                "event_id": "evt-c1-lucas",
                "user_id": "EMP001",
                "timestamp": "2026-07-14T14:00:00-03:00",
            },
            "AUTHORIZED",
            "LOW",
            {"ROOM_PERMISSION_CONFIRMED", "WITHIN_SCHEDULE"},
            {"EMP001", "cam-ti-01", "sala_ti_01"},
            ("Lucas", "14:00", "terça-feira", "dentro da escala"),
            False,
            id="c1-acesso-padrao-autorizado",
        ),
        pytest.param(
            {
                "event_id": "evt-c2-mariana",
                "user_id": "EMP002",
                "timestamp": "2026-07-15T02:00:00-03:00",
            },
            "JUSTIFIED",
            "MEDIUM",
            {
                "ROOM_PERMISSION_CONFIRMED",
                "OUTSIDE_SCHEDULE",
                "QUALIFYING_INCIDENT",
            },
            {"EMP002", "cam-ti-01", "sala_ti_01", "INC-402"},
            ("Mariana", "02:00", "#402", "Queda de servidor"),
            False,
            id="c2-fora-do-horario-justificado",
        ),
        pytest.param(
            {
                "event_id": "evt-c3-roberto",
                "user_id": "EMP003",
                "timestamp": "2026-07-19T23:00:00-03:00",
            },
            "ANOMALY",
            "HIGH",
            {
                "NO_ROOM_PERMISSION",
                "OUTSIDE_SCHEDULE",
                "NO_QUALIFYING_INCIDENT",
            },
            {"EMP003", "cam-ti-01", "sala_ti_01"},
            ("Roberto", "23:00", "domingo", "ALERTA DE SEGURANÇA"),
            True,
            id="c3-anomalia-com-alerta",
        ),
    ],
)
def test_canonical_rag_audit_scenarios(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
    payload_overrides: dict[str, Any],
    expected_decision: str,
    expected_risk: str,
    expected_reasons: set[str],
    expected_sources: set[str],
    narrative_facts: tuple[str, ...],
    alert_required: bool,
) -> None:
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(**payload_overrides),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["idempotent_replay"] is False

    event = body["event"]
    assert event["decision"] == expected_decision
    assert event["risk_level"] == expected_risk
    assert event["alert_required"] is alert_required
    assert expected_reasons <= set(event["reason_codes"])
    assert event["processing_ms"] < 3_000

    # Evidências e dados pessoais ficam somente na API administrativa.
    detail_response = client.get(
        f"/v1/access-events/{event['event_id']}", auth=admin_auth
    )
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert expected_sources <= set(detail["source_ids"])
    assert all(fact in detail["narrative"] for fact in narrative_facts)

    if alert_required:
        assert event["alert"] is not None
        assert event["alert"]["alert_id"]
    else:
        assert event["alert"] is None


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="sem-chave"),
        pytest.param({"X-Camera-Key": "invalid"}, id="chave-invalida"),
    ],
)
def test_webhook_rejects_missing_or_invalid_camera_key_without_persisting(
    client: TestClient,
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
    headers: dict[str, str],
) -> None:
    response = client.post(
        "/v1/webhooks/access-events", headers=headers, json=event_payload()
    )

    assert response.status_code == 401
    assert client.get("/v1/access-events", auth=admin_auth).json()["total"] == 0


@pytest.mark.parametrize(
    "payload_overrides",
    [
        pytest.param(
            {"event_id": "evt-unknown-camera", "camera_id": "cam-inexistente"},
            id="camera-desconhecida",
        ),
        pytest.param(
            {
                "event_id": "evt-room-mismatch",
                "camera_id": "cam-ti-01",
                "room_id": "sala_ti_02",
            },
            id="camera-de-outra-sala",
        ),
    ],
)
def test_webhook_rejects_unknown_camera_or_room_mismatch(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
    payload_overrides: dict[str, Any],
) -> None:
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(**payload_overrides),
    )

    assert response.status_code == 403
    assert client.get("/v1/access-events", auth=admin_auth).json()["total"] == 0


def test_webhook_rejects_timestamp_without_offset(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    payload = event_payload(
        event_id="evt-naive-time", timestamp="2026-07-14T14:00:00"
    )

    response = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=payload
    )

    assert response.status_code == 422
    assert "offset" in response.text
    assert client.get("/v1/access-events", auth=admin_auth).json()["total"] == 0


@pytest.mark.parametrize(
    "timestamp",
    ["0001-01-01T00:00:00+14:00", "9999-12-31T23:59:59-12:00"],
)
def test_webhook_rejects_extreme_timestamp_instead_of_returning_500(
    client: TestClient,
    camera_headers: dict[str, str],
    event_payload: Callable[..., dict[str, Any]],
    timestamp: str,
) -> None:
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(event_id="evt-extreme-time", timestamp=timestamp),
    )
    assert response.status_code == 422


def test_identical_retry_is_idempotent_and_does_not_duplicate_alert(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    payload = event_payload(
        event_id="evt-idempotent-anomaly",
        user_id="EMP003",
        timestamp="2026-07-19T23:00:00-03:00",
    )

    created = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=payload
    )
    replayed = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=payload
    )

    assert created.status_code == 201, created.text
    assert replayed.status_code == 200, replayed.text
    assert replayed.json()["idempotent_replay"] is True
    assert replayed.json()["event"]["event_id"] == payload["event_id"]
    assert (
        replayed.json()["event"]["alert"]["alert_id"]
        == created.json()["event"]["alert"]["alert_id"]
    )

    listing = client.get("/v1/access-events", auth=admin_auth).json()
    assert listing["total"] == 1
    assert len(listing["items"]) == 1


def test_same_event_id_with_different_payload_returns_conflict_and_keeps_original(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    original = event_payload(event_id="evt-conflicting-retry")
    changed = {**original, "door_result": "DENIED"}

    created = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=original
    )
    conflict = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=changed
    )

    assert created.status_code == 201
    assert conflict.status_code == 409

    saved = client.get(
        "/v1/access-events/evt-conflicting-retry", auth=admin_auth
    ).json()
    assert saved["door_result"] == "GRANTED"
    assert saved["raw_payload"]["door_result"] == "GRANTED"
    assert client.get("/v1/access-events", auth=admin_auth).json()["total"] == 1


def test_equivalent_utc_and_sao_paulo_instants_have_the_same_classification(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    local_payload = event_payload(
        event_id="evt-local-time",
        user_id="EMP003",
        timestamp="2026-07-19T23:00:00-03:00",
    )
    utc_payload = event_payload(
        event_id="evt-utc-time",
        user_id="EMP003",
        timestamp="2026-07-20T02:00:00Z",
    )

    local_response = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=local_payload
    )
    utc_response = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=utc_payload
    )

    assert local_response.status_code == utc_response.status_code == 201
    local_event = client.get(
        "/v1/access-events/evt-local-time", auth=admin_auth
    ).json()
    utc_event = client.get(
        "/v1/access-events/evt-utc-time", auth=admin_auth
    ).json()
    assert local_event["occurred_at"] == utc_event["occurred_at"]
    assert local_event["decision"] == utc_event["decision"] == "ANOMALY"
    assert local_event["risk_level"] == utc_event["risk_level"] == "HIGH"
    assert local_event["reason_codes"] == utc_event["reason_codes"]
    assert (
        local_event["context_snapshot"]["local_timestamp"]
        == utc_event["context_snapshot"]["local_timestamp"]
        == "2026-07-19T23:00:00-03:00"
    )


def test_equivalent_timestamp_spelling_is_an_idempotent_retry_not_a_conflict(
    client: TestClient,
    camera_headers: dict[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    local_payload = event_payload(
        event_id="evt-canonical-timestamp",
        user_id="EMP003",
        timestamp="2026-07-19T23:00:00-03:00",
    )
    utc_payload = {**local_payload, "timestamp": "2026-07-20T02:00:00Z"}

    first = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=local_payload
    )
    retry = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=utc_payload
    )

    assert first.status_code == 201
    assert retry.status_code == 200
    assert retry.json()["idempotent_replay"] is True
