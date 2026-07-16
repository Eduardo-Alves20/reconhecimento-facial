"""Aplica o contexto de acesso da API a partir de um manifesto privado."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env.api")
load_dotenv(PROJECT_ROOT / ".env")

from app.database import Repository, SCHEMA  # noqa: E402


IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$"
VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,49}$"
REASON_PATTERN = r"^[A-Z][A-Z0-9_]{1,99}$"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
REQUIRED_DECISIONS = {"AUTHORIZED", "JUSTIFIED", "ANOMALY"}
REQUIRED_TABLES = {
    "people",
    "rooms",
    "cameras",
    "room_permissions",
    "work_schedules",
    "policy_documents",
}
SECRET_FIELD_NAMES = {
    "api_key",
    "camera_api_key",
    "credential",
    "credentials",
    "password",
    "rtsp_url",
    "secret",
    "token",
}


class ProvisioningError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class RoomSpec(StrictModel):
    room_id: str = Field(pattern=IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    timezone: str = Field(min_length=1, max_length=100)
    criticality: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone deve ser um fuso IANA válido") from exc
        return value


class CameraSpec(StrictModel):
    camera_id: str = Field(pattern=IDENTIFIER_PATTERN)
    room_id: str = Field(pattern=IDENTIFIER_PATTERN)
    active: bool
    recognition_threshold: float = Field(ge=0.0, le=1.0)


class PersonSpec(StrictModel):
    person_id: str = Field(pattern=IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    role_name: str = Field(min_length=1, max_length=200)
    department: str = Field(min_length=1, max_length=200)
    active: bool


class PermissionSpec(StrictModel):
    person_id: str = Field(pattern=IDENTIFIER_PATTERN)
    room_id: str = Field(pattern=IDENTIFIER_PATTERN)


class ScheduleSpec(StrictModel):
    schedule_id: str = Field(pattern=IDENTIFIER_PATTERN)
    person_id: str = Field(pattern=IDENTIFIER_PATTERN)
    weekday: int = Field(ge=0, le=6)
    start_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    end_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")

    @model_validator(mode="after")
    def non_empty_interval(self) -> "ScheduleSpec":
        if time.fromisoformat(self.start_time) == time.fromisoformat(self.end_time):
            raise ValueError("start_time e end_time não podem ser iguais")
        return self


class PolicySpec(StrictModel):
    policy_id: str = Field(pattern=IDENTIFIER_PATTERN)
    version: str = Field(pattern=VERSION_PATTERN)
    applies_to_decision: Literal["AUTHORIZED", "JUSTIFIED", "ANOMALY"]
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=10_000)
    reason_codes: list[str] = Field(min_length=1, max_length=100)

    @field_validator("reason_codes")
    @classmethod
    def valid_reason_codes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("reason_codes não pode conter duplicatas")
        if any(not re.fullmatch(REASON_PATTERN, value) for value in values):
            raise ValueError("reason_codes contém um código inválido")
        return values


class AccessContextManifest(StrictModel):
    schema_version: Literal[1]
    policy_version: str = Field(pattern=VERSION_PATTERN)
    rooms: list[RoomSpec] = Field(min_length=1, max_length=10_000)
    cameras: list[CameraSpec] = Field(min_length=1, max_length=10_000)
    people: list[PersonSpec] = Field(min_length=1, max_length=100_000)
    room_permissions: list[PermissionSpec] = Field(max_length=500_000)
    work_schedules: list[ScheduleSpec] = Field(max_length=700_000)
    policies: list[PolicySpec] = Field(min_length=3, max_length=1_000)

    @model_validator(mode="after")
    def coherent_references(self) -> "AccessContextManifest":
        room_ids = _unique(self.rooms, "room_id")
        camera_ids = _unique(self.cameras, "camera_id")
        person_ids = _unique(self.people, "person_id")
        _unique(self.work_schedules, "schedule_id")
        _unique_pairs(self.room_permissions, "person_id", "room_id")
        _unique_pairs(self.policies, "policy_id", "version")

        for camera in self.cameras:
            if camera.room_id not in room_ids:
                raise ValueError(
                    f"câmera {camera.camera_id} referencia sala ausente: {camera.room_id}"
                )
        for permission in self.room_permissions:
            if permission.person_id not in person_ids or permission.room_id not in room_ids:
                raise ValueError(
                    "room_permissions deve referenciar pessoas e salas do mesmo manifesto"
                )
        for schedule in self.work_schedules:
            if schedule.person_id not in person_ids:
                raise ValueError(f"escala {schedule.schedule_id} referencia pessoa ausente")
        if any(policy.version != self.policy_version for policy in self.policies):
            raise ValueError("todas as políticas devem usar policy_version")
        decisions = {policy.applies_to_decision for policy in self.policies}
        if decisions != REQUIRED_DECISIONS:
            missing = ", ".join(sorted(REQUIRED_DECISIONS - decisions))
            raise ValueError(f"conjunto de políticas incompleto; faltando: {missing}")
        if not camera_ids:
            raise ValueError("ao menos uma câmera é obrigatória")
        return self


def _unique(items: list[Any], field: str) -> set[str]:
    values = [str(getattr(item, field)) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contém duplicatas")
    return set(values)


def _unique_pairs(items: list[Any], first: str, second: str) -> None:
    values = [(getattr(item, first), getattr(item, second)) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"{first}/{second} contém duplicatas")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvisioningError(f"chave JSON duplicada: {key}")
        result[key] = value
    return result


def _reject_secret_fields(value: Any, *, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.strip().lower()
            if normalized in SECRET_FIELD_NAMES:
                raise ProvisioningError(f"o manifesto não pode conter segredos ({location}.{key})")
            _reject_secret_fields(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_fields(item, location=f"{location}[{index}]")


def load_manifest(path: Path) -> AccessContextManifest:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ProvisioningError(f"manifesto não encontrado ou inseguro: {candidate}")
    size = candidate.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise ProvisioningError("manifesto vazio ou acima do limite de 2 MiB")
    try:
        payload = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvisioningError(f"manifesto JSON inválido: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProvisioningError("a raiz do manifesto deve ser um objeto JSON")
    _reject_secret_fields(payload)
    try:
        return AccessContextManifest.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ProvisioningError(f"manifesto recusado: {details}") from exc


TABLE_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "rooms": (
        ("room_id",),
        ("room_id", "display_name", "timezone", "criticality"),
    ),
    "cameras": (
        ("camera_id",),
        ("camera_id", "room_id", "active", "recognition_threshold"),
    ),
    "people": (
        ("person_id",),
        ("person_id", "display_name", "role_name", "department", "active"),
    ),
    "room_permissions": (
        ("person_id", "room_id"),
        ("person_id", "room_id"),
    ),
    "work_schedules": (
        ("schedule_id",),
        ("schedule_id", "person_id", "weekday", "start_time", "end_time"),
    ),
}


def _row_values(model: BaseModel, columns: tuple[str, ...]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in columns:
        value = getattr(model, column)
        values.append(int(value) if isinstance(value, bool) else value)
    return tuple(values)


def _upsert(
    connection: sqlite3.Connection,
    table: str,
    item: BaseModel,
) -> str:
    key_columns, columns = TABLE_SPECS[table]
    values = _row_values(item, columns)
    value_by_column = dict(zip(columns, values, strict=True))
    where = " AND ".join(f"{column}=?" for column in key_columns)
    existing = connection.execute(
        f"SELECT {', '.join(columns)} FROM {table} WHERE {where}",
        tuple(value_by_column[column] for column in key_columns),
    ).fetchone()
    if existing is not None and tuple(existing[column] for column in columns) == values:
        return "unchanged"
    if table == "cameras" and existing is not None:
        if existing["room_id"] != value_by_column["room_id"]:
            raise ProvisioningError(
                f"a câmera {value_by_column['camera_id']} não pode mudar de sala; "
                "cadastre um novo camera_id"
            )

    placeholders = ", ".join("?" for _ in columns)
    update_columns = tuple(column for column in columns if column not in key_columns)
    if update_columns:
        update = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
        conflict = f"DO UPDATE SET {update}"
    else:
        conflict = "DO NOTHING"
    connection.execute(
        f"""
        INSERT INTO {table} ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT ({", ".join(key_columns)}) {conflict}
        """,
        values,
    )
    return "created" if existing is None else "updated"


def _apply_policy(connection: sqlite3.Connection, policy: PolicySpec) -> str:
    columns = (
        "policy_id",
        "version",
        "applies_to_decision",
        "title",
        "content",
        "reason_codes",
    )
    values = (
        policy.policy_id,
        policy.version,
        policy.applies_to_decision,
        policy.title,
        policy.content,
        json.dumps(policy.reason_codes, ensure_ascii=False),
    )
    existing = connection.execute(
        """
        SELECT policy_id, version, applies_to_decision, title, content, reason_codes
        FROM policy_documents WHERE policy_id=? AND version=?
        """,
        (policy.policy_id, policy.version),
    ).fetchone()
    if existing is not None:
        current = tuple(existing[column] for column in columns)
        try:
            same_reasons = json.loads(current[-1]) == policy.reason_codes
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProvisioningError("política existente contém reason_codes inválidos") from exc
        if current[:-1] == values[:-1] and same_reasons:
            return "unchanged"
        raise ProvisioningError(
            f"a política {policy.policy_id}/{policy.version} é imutável; publique uma nova versão"
        )
    connection.execute(
        """
        INSERT INTO policy_documents
            (policy_id, version, applies_to_decision, title, content, reason_codes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return "created"


def _empty_summary() -> dict[str, dict[str, int]]:
    return {
        table: {"created": 0, "updated": 0, "unchanged": 0, "deleted": 0}
        for table in (*TABLE_SPECS, "policy_documents")
    }


def _chunks(values: list[str], size: int = 500) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _check_schema(connection: sqlite3.Connection) -> None:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = {str(row["name"]) for row in rows}
    missing = REQUIRED_TABLES - tables
    if missing:
        raise ProvisioningError(
            "banco sem o schema esperado; tabelas ausentes: " + ", ".join(sorted(missing))
        )


def apply_manifest(
    manifest: AccessContextManifest,
    database_path: Path,
    *,
    dry_run: bool,
    replace_assignments: bool = False,
) -> dict[str, Any]:
    path = database_path.expanduser()
    if path.is_symlink():
        raise ProvisioningError("o banco não pode ser um link simbólico")
    if path.exists() and not path.is_file():
        raise ProvisioningError("o caminho do banco não é um arquivo")

    in_memory = dry_run and not path.exists()
    if not dry_run:
        Repository(path).initialize()
    connection = sqlite3.connect(":memory:" if in_memory else path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        if in_memory:
            connection.executescript(SCHEMA)
        else:
            _check_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        summary = _empty_summary()
        if replace_assignments:
            person_ids = [person.person_id for person in manifest.people]
            desired_permissions = {
                (permission.person_id, permission.room_id)
                for permission in manifest.room_permissions
            }
            existing_permissions: list[sqlite3.Row] = []
            existing_schedules: list[sqlite3.Row] = []
            for batch in _chunks(person_ids):
                placeholders = ", ".join("?" for _ in batch)
                existing_permissions.extend(
                    connection.execute(
                        f"""
                        SELECT person_id, room_id FROM room_permissions
                        WHERE person_id IN ({placeholders})
                        """,
                        batch,
                    ).fetchall()
                )
                existing_schedules.extend(
                    connection.execute(
                        f"""
                        SELECT schedule_id FROM work_schedules
                        WHERE person_id IN ({placeholders})
                        """,
                        batch,
                    ).fetchall()
                )
            stale_permissions = [
                (row["person_id"], row["room_id"])
                for row in existing_permissions
                if (row["person_id"], row["room_id"]) not in desired_permissions
            ]
            connection.executemany(
                "DELETE FROM room_permissions WHERE person_id=? AND room_id=?",
                stale_permissions,
            )
            summary["room_permissions"]["deleted"] = len(stale_permissions)

            desired_schedule_ids = {schedule.schedule_id for schedule in manifest.work_schedules}
            stale_schedule_ids = [
                row["schedule_id"]
                for row in existing_schedules
                if row["schedule_id"] not in desired_schedule_ids
            ]
            connection.executemany(
                "DELETE FROM work_schedules WHERE schedule_id=?",
                [(schedule_id,) for schedule_id in stale_schedule_ids],
            )
            summary["work_schedules"]["deleted"] = len(stale_schedule_ids)
        groups: tuple[tuple[str, list[BaseModel]], ...] = (
            ("rooms", list(manifest.rooms)),
            ("people", list(manifest.people)),
            ("cameras", list(manifest.cameras)),
            ("room_permissions", list(manifest.room_permissions)),
            ("work_schedules", list(manifest.work_schedules)),
        )
        for table, items in groups:
            for item in items:
                outcome = _upsert(connection, table, item)
                summary[table][outcome] += 1
        for policy in manifest.policies:
            outcome = _apply_policy(connection, policy)
            summary["policy_documents"][outcome] += 1
        if dry_run:
            connection.rollback()
        else:
            connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    if not dry_run:
        try:
            path.chmod(0o600)
        except OSError:
            pass

    return {
        "ok": True,
        "dry_run": dry_run,
        "replace_assignments": replace_assignments,
        "database": str(path),
        "policy_version": manifest.policy_version,
        "summary": summary,
    }


def default_database_path() -> Path:
    return Path(
        os.getenv(
            "RAG_AUDIT_DB_PATH",
            str(PROJECT_ROOT / "data" / "api" / "rag_audit.db"),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="manifesto JSON privado")
    parser.add_argument(
        "--database",
        type=Path,
        default=default_database_path(),
        help="banco SQLite da API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="valida e simula a transação sem persistir alterações",
    )
    parser.add_argument(
        "--replace-assignments",
        action="store_true",
        help="remove permissões e escalas omitidas para as pessoas do manifesto",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        result = apply_manifest(
            manifest,
            args.database,
            dry_run=args.dry_run,
            replace_assignments=args.replace_assignments,
        )
    except (ProvisioningError, OSError, sqlite3.Error) as exc:
        if args.json_output:
            print(
                json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
                file=sys.stderr,
            )
        else:
            print(f"Erro: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        mode = "simulação concluída" if args.dry_run else "contexto aplicado"
        print(f"{mode}: {args.database}")
        for table, counts in result["summary"].items():
            print(
                f"- {table}: {counts['created']} criados, "
                f"{counts['updated']} atualizados, "
                f"{counts['unchanged']} inalterados, "
                f"{counts['deleted']} removidos"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
