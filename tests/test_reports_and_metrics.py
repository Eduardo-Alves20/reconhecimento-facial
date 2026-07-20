from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient
from pypdf import PdfReader


def _seed_three_scenarios(
    client: TestClient,
    camera_headers: dict[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads = [
        event_payload(
            event_id="evt-report-lucas",
            user_id="EMP001",
            timestamp="2026-07-14T14:00:00-03:00",
        ),
        event_payload(
            event_id="evt-report-mariana",
            user_id="EMP002",
            timestamp="2026-07-15T02:00:00-03:00",
        ),
        event_payload(
            event_id="evt-report-roberto",
            user_id="EMP003",
            timestamp="2026-07-19T23:00:00-03:00",
        ),
    ]
    for payload in payloads:
        response = client.post(
            "/v1/webhooks/access-events", headers=camera_headers, json=payload
        )
        assert response.status_code == 201, response.text
    return payloads


def _assert_valid_pdf(response) -> str:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/pdf")
    assert "attachment" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")
    assert response.content.rstrip().endswith(b"%%EOF")

    reader = PdfReader(BytesIO(response.content))
    assert len(reader.pages) >= 1
    assert reader.metadata is not None
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_read_and_pdf_routes_require_basic_auth(client: TestClient) -> None:
    protected_paths = (
        "/v1/access-events",
        "/v1/metrics",
        "/v1/access-events/does-not-matter/report.pdf",
        "/v1/reports/access-events.pdf",
    )

    for path in protected_paths:
        missing = client.get(path)
        invalid = client.get(path, auth=("auditor", "wrong-password"))
        assert missing.status_code == 401
        assert invalid.status_code == 401
        assert missing.headers["www-authenticate"] == 'Basic realm="QTA"'


def test_individual_event_pdf_is_parseable_and_contains_audit_facts(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    payload = event_payload(event_id="evt-pdf-lucas")
    created = client.post(
        "/v1/webhooks/access-events", headers=camera_headers, json=payload
    )
    assert created.status_code == 201

    response = client.get(
        "/v1/access-events/evt-pdf-lucas/report.pdf", auth=admin_auth
    )
    text = _assert_valid_pdf(response)

    assert "evt-pdf-lucas" in response.headers["content-disposition"]
    assert "QTA" in text
    assert "evt-pdf-lucas" in text
    assert "Lucas" in text
    assert "Sala de TI 01" in text


def test_individual_pdf_handles_oversized_incident_title(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    repository = client.app.state.repository
    with repository.connect() as connection:
        connection.execute(
            "UPDATE incidents SET title = ? WHERE incident_id = 'INC-402'",
            ("servidor indisponível " * 2_000,),
        )
        connection.commit()
    event_id = "evt-pdf-long-incident"
    created = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(
            event_id=event_id,
            user_id="EMP002",
            timestamp="2026-07-15T02:00:00-03:00",
        ),
    )
    assert created.status_code == 201
    response = client.get(
        f"/v1/access-events/{event_id}/report.pdf", auth=admin_auth
    )
    _assert_valid_pdf(response)


def test_consolidated_pdf_is_parseable_and_contains_all_three_scenarios(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    _seed_three_scenarios(client, camera_headers, event_payload)

    response = client.get("/v1/reports/access-events.pdf", auth=admin_auth)
    text = _assert_valid_pdf(response)

    assert "Relatório consolidado de acessos" in text
    assert "Total: 3" in text
    assert "Lucas" in text
    assert "Mariana" in text
    assert "Roberto" in text
    assert "Anomalias para revisão" in text


def test_consolidated_pdf_truncates_oversized_table_text_without_layout_error(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    event_id = "evt-report-long-text"
    created = client.post(
        "/v1/webhooks/access-events",
        headers=camera_headers,
        json=event_payload(event_id=event_id),
    )
    assert created.status_code == 201
    repository = client.app.state.repository
    with repository.connect() as connection:
        connection.execute(
            "UPDATE access_events SET narrative = ? WHERE event_id = ?",
            ("contexto muito longo " * 5_000, event_id),
        )
        connection.commit()

    response = client.get("/v1/reports/access-events.pdf", auth=admin_auth)
    _assert_valid_pdf(response)


def test_metrics_report_scenario_counts_and_less_than_three_second_sla(
    client: TestClient,
    camera_headers: dict[str, str],
    admin_auth: tuple[str, str],
    event_payload: Callable[..., dict[str, Any]],
) -> None:
    payloads = [
        event_payload(
            event_id="evt-sla-lucas",
            user_id="EMP001",
            timestamp="2026-07-14T14:00:00-03:00",
        ),
        event_payload(
            event_id="evt-sla-mariana",
            user_id="EMP002",
            timestamp="2026-07-15T02:00:00-03:00",
        ),
        event_payload(
            event_id="evt-sla-roberto",
            user_id="EMP003",
            timestamp="2026-07-19T23:00:00-03:00",
        ),
    ]

    for payload in payloads:
        started = perf_counter()
        response = client.post(
            "/v1/webhooks/access-events", headers=camera_headers, json=payload
        )
        elapsed_ms = (perf_counter() - started) * 1_000
        assert response.status_code == 201, response.text
        assert elapsed_ms < 3_000
        assert response.json()["event"]["processing_ms"] < 3_000

    response = client.get("/v1/metrics", auth=admin_auth)
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["total"] == 3
    assert metrics["authorized"] == 1
    assert metrics["justified"] == 1
    assert metrics["anomalies"] == 1
    assert metrics["within_sla"] == 3
    assert metrics["sla_percentage"] == 100.0
    assert metrics["average_processing_ms"] < 3_000
    assert metrics["p95_processing_ms"] < 3_000
