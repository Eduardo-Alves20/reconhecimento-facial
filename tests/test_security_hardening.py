from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_webhook_receipt_does_not_expose_person_or_internal_context(
    client: TestClient,
    camera_headers: dict[str, str],
    event_payload,
) -> None:
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(event_id="evt-minimal-receipt"),
    )

    assert response.status_code == 201
    receipt = response.json()["event"]
    assert set(receipt) == {
        "event_id",
        "decision",
        "risk_level",
        "risk_score",
        "reason_codes",
        "alert_required",
        "alert",
        "policy_version",
        "processing_ms",
    }
    serialized = response.text
    for forbidden in ("Lucas", "Infraestrutura", "context_snapshot", "source_ids", "raw_payload"):
        assert forbidden not in serialized


def test_policy_retrieval_uses_the_decision_policy_not_any_intersecting_code(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload,
) -> None:
    cases = [
        ("evt-policy-standard", "EMP001", "2026-07-14T14:00:00-03:00", ["POL-001"]),
        ("evt-policy-incident", "EMP002", "2026-07-15T02:00:00-03:00", ["POL-002"]),
        ("evt-policy-anomaly", "EMP003", "2026-07-19T23:00:00-03:00", ["POL-003"]),
    ]
    for event_id, user_id, timestamp, expected_policy_ids in cases:
        created = client.post(
            "/v1/webhooks/access-events",
            headers=camera_headers,
            json=event_payload(event_id=event_id, user_id=user_id, timestamp=timestamp),
        )
        assert created.status_code == 201
        detail = client.get(f"/v1/access-events/{event_id}", auth=admin_auth).json()
        policy_ids = [
            item["policy_id"] for item in detail["context_snapshot"]["policies"]
        ]
        assert policy_ids == expected_policy_ids


def test_low_confidence_uses_identity_policy_without_false_permission_policy(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload,
) -> None:
    event_id = "evt-low-confidence-policy"
    created = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(
            event_id=event_id,
            user_id="EMP001",
            timestamp="2026-07-14T14:00:00-03:00",
            recognition_confidence=0.1,
        ),
    )
    assert created.status_code == 201
    detail = client.get(f"/v1/access-events/{event_id}", auth=admin_auth).json()
    assert detail["risk_level"] == "CRITICAL"
    assert [
        item["policy_id"] for item in detail["context_snapshot"]["policies"]
    ] == ["POL-004"]


def test_event_is_not_recorded_when_no_applicable_policy_exists(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload,
) -> None:
    repository = client.app.state.repository
    with repository.connect() as connection:
        connection.execute("DELETE FROM policy_documents")
        connection.commit()

    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(event_id="evt-without-policy"),
    )
    assert response.status_code == 503
    assert client.get("/health/ready").status_code == 503
    assert client.get("/v1/access-events", auth=admin_auth).json()["total"] == 0


def test_only_explicit_p1_p2_severity_can_justify_out_of_hours_access(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload,
) -> None:
    repository = client.app.state.repository
    with repository.connect() as connection:
        connection.execute(
            "UPDATE incidents SET severity = 'URGENT' WHERE incident_id = 'INC-402'"
        )
        connection.commit()

    event_id = "evt-urgent-is-not-p2"
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(
            event_id=event_id,
            user_id="EMP002",
            timestamp="2026-07-15T02:00:00-03:00",
        ),
    )
    assert response.status_code == 201
    assert response.json()["event"]["decision"] == "ANOMALY"
    detail = client.get(f"/v1/access-events/{event_id}", auth=admin_auth).json()
    assert "NO_QUALIFYING_INCIDENT" in detail["reason_codes"]
    assert detail["context_snapshot"]["qualifying_incidents"] == []


def test_docs_and_sensitive_responses_are_protected_and_not_cacheable(
    client: TestClient, admin_auth: tuple[str, str]
) -> None:
    assert client.get("/docs").status_code == 401
    assert client.get("/openapi.json").status_code == 401

    docs = client.get("/docs", auth=admin_auth)
    schema = client.get("/openapi.json", auth=admin_auth)
    listing = client.get("/v1/access-events", auth=admin_auth)
    assert docs.status_code == 200
    assert schema.status_code == 200
    assert schema.json()["info"]["title"] == "QTA"
    assert listing.headers["cache-control"] == "no-store, max-age=0"
    assert listing.headers["x-content-type-options"] == "nosniff"


def test_dashboard_identifies_seeded_data_as_demo(
    client: TestClient, admin_auth: tuple[str, str]
) -> None:
    dashboard = client.get("/dashboard", auth=admin_auth)
    assert dashboard.status_code == 200
    assert "Dados de demonstração" in dashboard.text
    assert "câmera Intelbras" in dashboard.text


def test_freshness_guard_rejects_old_and_future_events(
    settings: Settings,
    camera_headers: dict[str, str],
    event_payload,
) -> None:
    guarded_app = create_app(
        replace(
            settings,
            enforce_event_freshness=True,
            event_max_age_seconds=60,
            event_future_skew_seconds=30,
        )
    )
    now = datetime.now(UTC)
    old = (now - timedelta(minutes=10)).isoformat()
    future = (now + timedelta(minutes=10)).isoformat()

    with TestClient(guarded_app) as guarded_client:
        old_response = guarded_client.post(
            "/v1/webhooks/access-events",
            headers=camera_headers,
            json=event_payload(event_id="evt-too-old", timestamp=old),
        )
        future_response = guarded_client.post(
            "/v1/webhooks/access-events",
            headers=camera_headers,
            json=event_payload(event_id="evt-too-future", timestamp=future),
        )

    assert old_response.status_code == 422
    assert future_response.status_code == 422


def test_durable_outbox_has_a_bounded_extended_delivery_window(
    settings: Settings,
    camera_headers: dict[str, str],
    event_payload,
) -> None:
    guarded = replace(
        settings,
        enforce_event_freshness=True,
        event_max_age_seconds=300,
        queued_event_max_age_seconds=3_600,
    )
    guarded_app = create_app(guarded)
    now = datetime.now(UTC)
    queued = now - timedelta(minutes=10)
    too_old = now - timedelta(hours=2)

    with TestClient(guarded_app) as guarded_client:
        accepted = guarded_client.post(
            "/v1/webhooks/access-events",
            headers={
                **camera_headers,
                "X-Delivery-Mode": "durable-outbox",
                "X-Event-Queued-At": queued.isoformat(),
            },
            json=event_payload(
                event_id="evt-queued-accepted",
                timestamp=queued.isoformat(),
            ),
        )
        rejected = guarded_client.post(
            "/v1/webhooks/access-events",
            headers={
                **camera_headers,
                "X-Delivery-Mode": "durable-outbox",
                "X-Event-Queued-At": too_old.isoformat(),
            },
            json=event_payload(
                event_id="evt-queued-too-old",
                timestamp=too_old.isoformat(),
            ),
        )

    assert accepted.status_code == 201
    assert rejected.status_code == 422


def test_production_rejects_demo_seed_weak_or_ambiguous_configuration(
    settings: Settings,
) -> None:
    strong = replace(
        settings,
        environment="production ",
        camera_api_key="c" * 32,
        camera_api_keys=(("cam-ti-01", "c" * 32),),
        admin_password="a" * 20,
        seed_demo_data=False,
        enforce_event_freshness=True,
    )
    strong.validate()

    with pytest.raises(RuntimeError, match="development, test ou production"):
        replace(strong, environment="prod").validate()
    with pytest.raises(RuntimeError, match="demonstração"):
        replace(strong, seed_demo_data=True).validate()
    with pytest.raises(RuntimeError, match="frescor"):
        replace(strong, enforce_event_freshness=False).validate()
    with pytest.raises(RuntimeError, match="HTTPS"):
        replace(strong, alert_webhook_url="http://alerts.internal").validate()
    with pytest.raises(RuntimeError, match="exclusiva"):
        replace(
            strong,
            camera_api_keys=(
                ("cam-ti-01", "c" * 32),
                ("cam-ti-02", "c" * 32),
            ),
        ).validate()


def test_production_app_starts_without_demo_people(
    settings: Settings,
    admin_auth: tuple[str, str],
) -> None:
    production_settings = replace(
        settings,
        environment="production",
        camera_api_key="c" * 32,
        camera_api_keys=(("cam-ti-01", "c" * 32),),
        admin_password="a" * 20,
        seed_demo_data=False,
        enforce_event_freshness=True,
    )
    production_app = create_app(production_settings)
    production_auth = (production_settings.admin_username, production_settings.admin_password)
    with TestClient(production_app) as production_client:
        rooms = production_client.get("/v1/rooms", auth=production_auth)
        assert rooms.status_code == 200
        assert rooms.json() == {"items": []}


def test_camera_key_is_bound_to_camera_id(
    settings: Settings,
    event_payload,
) -> None:
    isolated = replace(
        settings,
        camera_api_keys=(
            ("cam-ti-01", "key-for-camera-one"),
            ("cam-ti-02", "key-for-camera-two"),
        ),
    )
    app = create_app(isolated)

    with TestClient(app) as client:
        response = client.post(
            "/v1/webhooks/access-events",
            headers={"X-Camera-Key": "key-for-camera-one"},
            json=event_payload(
                event_id="evt-cross-camera-key",
                camera_id="cam-ti-02",
                room_id="sala_ti_02",
            ),
        )

    assert response.status_code == 401


def test_settings_repr_does_not_expose_secrets(settings: Settings) -> None:
    protected = replace(
        settings,
        camera_api_key="global-camera-secret",
        camera_api_keys=(("cam-ti-01", "per-camera-secret"),),
        admin_password="admin-secret",
    )

    rendered = repr(protected)

    assert "global-camera-secret" not in rendered
    assert "per-camera-secret" not in rendered
    assert "admin-secret" not in rendered
