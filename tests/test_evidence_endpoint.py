from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient


JPEG = b"\xff\xd8\xff\x00\xff\xd9"
NOW = datetime(2026, 7, 14, 17, 0, tzinfo=UTC)


def _create_event_with_evidence(
    client: TestClient,
    camera_headers: dict[str, str],
    event_payload,
    *,
    event_id: str,
    reference: str,
    captured_at: datetime,
) -> None:
    payload = event_payload(
        event_id=event_id,
        timestamp=captured_at.isoformat(),
    )
    payload.update(
        {
            "evidence_ref": reference,
            "evidence_captured_at": captured_at.isoformat(),
        }
    )
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=payload,
    )
    assert response.status_code == 201


def test_photo_endpoint_reads_full_and_falls_back_when_thumb_is_absent(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth,
    event_payload,
) -> None:
    record = client.app.state.evidence_store.save(JPEG, created_at=NOW)
    _create_event_with_evidence(
        client,
        camera_headers,
        event_payload,
        event_id="evt-photo-full",
        reference=record.reference,
        captured_at=NOW,
    )

    full = client.get(
        "/v1/access-events/evt-photo-full/photo",
        auth=admin_auth,
    )
    thumb = client.get(
        "/v1/access-events/evt-photo-full/photo?variant=thumb",
        auth=admin_auth,
    )

    assert full.status_code == 200
    assert full.content == JPEG
    assert thumb.status_code == 200
    assert thumb.content == JPEG
    assert full.headers["cache-control"] == "no-store, max-age=0"


def test_photo_endpoint_blocks_tampered_evidence(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth,
    event_payload,
) -> None:
    store = client.app.state.evidence_store
    record = store.save(JPEG, created_at=NOW)
    _create_event_with_evidence(
        client,
        camera_headers,
        event_payload,
        event_id="evt-photo-tampered",
        reference=record.reference,
        captured_at=NOW,
    )
    store.path_for(record.reference).write_bytes(b"\xff\xd8\xffBAD\xff\xd9")

    response = client.get(
        "/v1/access-events/evt-photo-tampered/photo",
        auth=admin_auth,
    )

    assert response.status_code == 409


def test_photo_endpoint_removes_expired_evidence(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth,
    event_payload,
) -> None:
    expired_at = datetime(2000, 1, 1, tzinfo=UTC)
    record = client.app.state.evidence_store.save(
        JPEG,
        created_at=expired_at,
        ttl=timedelta(seconds=1),
    )
    _create_event_with_evidence(
        client,
        camera_headers,
        event_payload,
        event_id="evt-photo-expired",
        reference=record.reference,
        captured_at=expired_at,
    )

    response = client.get(
        "/v1/access-events/evt-photo-expired/photo",
        auth=admin_auth,
    )

    assert response.status_code == 404
    assert not client.app.state.evidence_store.path_for(record.reference).exists()
