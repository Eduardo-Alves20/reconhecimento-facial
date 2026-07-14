from __future__ import annotations

from fastapi.testclient import TestClient


def _visual_payload(event_payload, **overrides):
    payload = event_payload(
        event_id="vision-entry-001",
        user_id="EMP001",
        timestamp="2026-07-14T14:00:00-03:00",
        door_result="NOT_REPORTED",
    )
    payload.update(
        {
            "entry_evidence": "VISION_LINE_CROSSING",
            "identity_status": "MATCHED",
            "recognition_source": "LOCAL_SFACE",
            "track_id": "boot-01:track-42",
            "recognition_model": "opencv-sface-2021dec",
            "recognition_margin": 0.22,
            "face_quality": 0.91,
            "entry_confidence": 0.95,
        }
    )
    payload.update(overrides)
    return payload


def test_visual_entry_is_described_as_observed_not_door_granted(
    client: TestClient, camera_headers: dict[str, str], admin_auth, event_payload
) -> None:
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=_visual_payload(event_payload),
    )
    assert response.status_code == 201

    detail = client.get(
        "/v1/access-events/vision-entry-001", auth=admin_auth
    ).json()
    assert "entrada observada pela câmera" in detail["narrative"]
    assert "entrada liberada" not in detail["narrative"]
    assert detail["door_result"] == "NOT_REPORTED"
    assert detail["context_snapshot"]["entry_observation"] == {
        "identity_status": "MATCHED",
        "entry_evidence": "VISION_LINE_CROSSING",
        "recognition_source": "LOCAL_SFACE",
        "track_id": "boot-01:track-42",
        "recognition_model": "opencv-sface-2021dec",
        "recognition_margin": 0.22,
        "face_quality": 0.91,
        "entry_confidence": 0.95,
    }


def test_visual_entry_cannot_claim_that_door_was_granted(
    client: TestClient, camera_headers: dict[str, str], event_payload
) -> None:
    response = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=_visual_payload(event_payload, door_result="GRANTED"),
    )
    assert response.status_code == 422
    assert "a câmera não prova" in response.text


def test_ambiguous_identity_is_critical_and_never_assigned_to_known_person(
    client: TestClient, camera_headers: dict[str, str], admin_auth, event_payload
) -> None:
    payload = _visual_payload(
        event_payload,
        event_id="vision-ambiguous-001",
        user_id="AMBIGUOUS:boot-01:track-43",
        identity_status="AMBIGUOUS",
        recognition_confidence=None,
        recognition_margin=0.01,
    )
    response = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=payload
    )
    assert response.status_code == 201
    assert response.json()["event"]["risk_level"] == "CRITICAL"

    detail = client.get(
        "/v1/access-events/vision-ambiguous-001", auth=admin_auth
    ).json()
    assert "AMBIGUOUS_IDENTITY" in detail["reason_codes"]
    assert detail["person_name"] is None
