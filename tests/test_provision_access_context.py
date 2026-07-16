from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.provision_access_context import (
    ProvisioningError,
    _chunks,
    apply_manifest,
    load_manifest,
)


def manifest_payload() -> dict:
    return {
        "schema_version": 1,
        "policy_version": "2026.1",
        "rooms": [
            {
                "room_id": "sala_ti_01",
                "display_name": "Sala de TI 01",
                "timezone": "America/Sao_Paulo",
                "criticality": "CRITICAL",
            }
        ],
        "cameras": [
            {
                "camera_id": "cam-ti-01",
                "room_id": "sala_ti_01",
                "active": True,
                "recognition_threshold": 0.85,
            }
        ],
        "people": [
            {
                "person_id": "EMP001",
                "display_name": "Pessoa Teste",
                "role_name": "Analista",
                "department": "Infraestrutura",
                "active": True,
            }
        ],
        "room_permissions": [{"person_id": "EMP001", "room_id": "sala_ti_01"}],
        "work_schedules": [
            {
                "schedule_id": "SCH-EMP001-0",
                "person_id": "EMP001",
                "weekday": 0,
                "start_time": "08:00",
                "end_time": "18:00",
            }
        ],
        "policies": [
            {
                "policy_id": "POL-001",
                "version": "2026.1",
                "applies_to_decision": "AUTHORIZED",
                "title": "Autorizado",
                "content": "Contexto regular.",
                "reason_codes": ["WITHIN_SCHEDULE", "ROOM_PERMISSION_CONFIRMED"],
            },
            {
                "policy_id": "POL-002",
                "version": "2026.1",
                "applies_to_decision": "JUSTIFIED",
                "title": "Justificado",
                "content": "Contexto emergencial.",
                "reason_codes": ["OUTSIDE_SCHEDULE", "QUALIFYING_INCIDENT"],
            },
            {
                "policy_id": "POL-003",
                "version": "2026.1",
                "applies_to_decision": "ANOMALY",
                "title": "Anomalia",
                "content": "Contexto atípico.",
                "reason_codes": ["NO_ROOM_PERMISSION", "UNKNOWN_PERSON"],
            },
        ],
    }


def write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_apply_is_transactional_and_idempotent(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path / "context.json", manifest_payload()))
    database = tmp_path / "api" / "rag_audit.db"

    first = apply_manifest(manifest, database, dry_run=False)
    second = apply_manifest(manifest, database, dry_run=False)

    assert first["summary"]["people"]["created"] == 1
    assert first["summary"]["policy_documents"]["created"] == 3
    assert second["summary"]["people"]["unchanged"] == 1
    assert second["summary"]["policy_documents"]["unchanged"] == 3
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM room_permissions").fetchone()[0] == 1


def test_dry_run_does_not_create_database(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path / "context.json", manifest_payload()))
    database = tmp_path / "missing" / "rag_audit.db"

    result = apply_manifest(manifest, database, dry_run=True)

    assert result["dry_run"] is True
    assert result["summary"]["rooms"]["created"] == 1
    assert not database.exists()
    assert not database.parent.exists()


def test_dry_run_rolls_back_updates_to_existing_database(tmp_path: Path) -> None:
    payload = manifest_payload()
    path = write_manifest(tmp_path / "context.json", payload)
    database = tmp_path / "rag_audit.db"
    apply_manifest(load_manifest(path), database, dry_run=False)

    payload["people"][0]["display_name"] = "Nome alterado"
    apply_manifest(
        load_manifest(write_manifest(path, payload)),
        database,
        dry_run=True,
    )

    with sqlite3.connect(database) as connection:
        name = connection.execute(
            "SELECT display_name FROM people WHERE person_id='EMP001'"
        ).fetchone()[0]
    assert name == "Pessoa Teste"


def test_policy_version_is_immutable_and_rolls_back_other_updates(tmp_path: Path) -> None:
    payload = manifest_payload()
    path = write_manifest(tmp_path / "context.json", payload)
    database = tmp_path / "rag_audit.db"
    apply_manifest(load_manifest(path), database, dry_run=False)

    payload["rooms"][0]["display_name"] = "Nome que deve voltar"
    payload["policies"][0]["content"] = "Conteúdo alterado na mesma versão."
    with pytest.raises(ProvisioningError, match="é imutável"):
        apply_manifest(
            load_manifest(write_manifest(path, payload)),
            database,
            dry_run=False,
        )

    with sqlite3.connect(database) as connection:
        room_name = connection.execute(
            "SELECT display_name FROM rooms WHERE room_id='sala_ti_01'"
        ).fetchone()[0]
    assert room_name == "Sala de TI 01"


def test_existing_camera_cannot_be_moved_to_another_room(tmp_path: Path) -> None:
    payload = manifest_payload()
    path = write_manifest(tmp_path / "context.json", payload)
    database = tmp_path / "rag_audit.db"
    apply_manifest(load_manifest(path), database, dry_run=False)
    payload["rooms"].append(
        {
            "room_id": "sala_ti_02",
            "display_name": "Sala de TI 02",
            "timezone": "America/Sao_Paulo",
            "criticality": "HIGH",
        }
    )
    payload["cameras"][0]["room_id"] = "sala_ti_02"

    with pytest.raises(ProvisioningError, match="não pode mudar de sala"):
        apply_manifest(
            load_manifest(write_manifest(path, payload)),
            database,
            dry_run=False,
        )


def test_replace_assignments_removes_only_stale_rows_for_manifest_people(
    tmp_path: Path,
) -> None:
    payload = manifest_payload()
    payload["rooms"].append(
        {
            "room_id": "sala_ti_02",
            "display_name": "Sala de TI 02",
            "timezone": "America/Sao_Paulo",
            "criticality": "HIGH",
        }
    )
    payload["room_permissions"].append({"person_id": "EMP001", "room_id": "sala_ti_02"})
    payload["work_schedules"].append(
        {
            "schedule_id": "SCH-EMP001-1",
            "person_id": "EMP001",
            "weekday": 1,
            "start_time": "08:00",
            "end_time": "18:00",
        }
    )
    path = write_manifest(tmp_path / "context.json", payload)
    database = tmp_path / "rag_audit.db"
    apply_manifest(load_manifest(path), database, dry_run=False)

    payload["room_permissions"] = payload["room_permissions"][:1]
    payload["work_schedules"] = payload["work_schedules"][:1]
    result = apply_manifest(
        load_manifest(write_manifest(path, payload)),
        database,
        dry_run=False,
        replace_assignments=True,
    )

    assert result["summary"]["room_permissions"]["deleted"] == 1
    assert result["summary"]["work_schedules"]["deleted"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM room_permissions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM work_schedules").fetchone()[0] == 1


def test_manifest_rejects_secret_fields(tmp_path: Path) -> None:
    payload = manifest_payload()
    payload["camera_api_key"] = "not-allowed"

    with pytest.raises(ProvisioningError, match="não pode conter segredos"):
        load_manifest(write_manifest(tmp_path / "context.json", payload))


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "context.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    with pytest.raises(ProvisioningError, match="chave JSON duplicada"):
        load_manifest(path)


def test_manifest_requires_complete_policy_decisions(tmp_path: Path) -> None:
    payload = manifest_payload()
    payload["policies"] = payload["policies"][:2]

    with pytest.raises(ProvisioningError, match="manifesto recusado"):
        load_manifest(write_manifest(tmp_path / "context.json", payload))


def test_large_assignment_queries_are_split_for_sqlite() -> None:
    batches = _chunks([f"EMP{index:04d}" for index in range(1_001)])

    assert [len(batch) for batch in batches] == [500, 500, 1]
