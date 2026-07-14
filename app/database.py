from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    person_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role_name TEXT NOT NULL,
    department TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS external_identities (
    source_id TEXT NOT NULL,
    external_id_hash TEXT NOT NULL,
    person_id TEXT NOT NULL REFERENCES people(person_id),
    image_sha256 TEXT,
    synced_at TEXT NOT NULL,
    PRIMARY KEY (source_id, external_id_hash)
);

CREATE TABLE IF NOT EXISTS rooms (
    room_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    timezone TEXT NOT NULL,
    criticality TEXT NOT NULL DEFAULT 'HIGH'
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL REFERENCES rooms(room_id),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    recognition_threshold REAL NOT NULL DEFAULT 0.85
);

CREATE TABLE IF NOT EXISTS room_permissions (
    person_id TEXT NOT NULL REFERENCES people(person_id),
    room_id TEXT NOT NULL REFERENCES rooms(room_id),
    PRIMARY KEY (person_id, room_id)
);

CREATE TABLE IF NOT EXISTS work_schedules (
    schedule_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES people(person_id),
    weekday INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    person_id TEXT NOT NULL REFERENCES people(person_id),
    room_id TEXT NOT NULL REFERENCES rooms(room_id),
    active_from TEXT NOT NULL,
    active_until TEXT
);

CREATE TABLE IF NOT EXISTS policy_documents (
    policy_id TEXT NOT NULL,
    version TEXT NOT NULL,
    applies_to_decision TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    PRIMARY KEY (policy_id, version)
);

CREATE TABLE IF NOT EXISTS access_events (
    event_id TEXT PRIMARY KEY,
    payload_hash TEXT NOT NULL,
    raw_payload TEXT NOT NULL,
    person_id TEXT NOT NULL,
    person_name TEXT,
    role_name TEXT,
    department TEXT,
    room_id TEXT NOT NULL,
    room_name TEXT,
    camera_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    processed_at TEXT,
    door_result TEXT NOT NULL,
    recognition_confidence REAL,
    decision TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    risk_score INTEGER NOT NULL,
    reason_codes TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    narrative TEXT NOT NULL,
    context_snapshot TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    processing_ms REAL NOT NULL DEFAULT 0,
    alert_required INTEGER NOT NULL DEFAULT 0 CHECK (alert_required IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alert_outbox (
    alert_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES access_events(event_id),
    channel TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    last_error TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_access_events_occurred_at
    ON access_events(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_events_room_time
    ON access_events(room_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_events_person_time
    ON access_events(person_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_events_decision
    ON access_events(decision, risk_level);
CREATE INDEX IF NOT EXISTS idx_alert_outbox_delivery
    ON alert_outbox(status, next_attempt_at);
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class Repository:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            self._migrate_policy_documents(connection)
            connection.executescript(SCHEMA)
            connection.execute("PRAGMA user_version = 3")
            connection.commit()

    @staticmethod
    def _migrate_policy_documents(connection: sqlite3.Connection) -> None:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'policy_documents'"
        ).fetchone()
        if not exists:
            return
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(policy_documents)").fetchall()
        }
        if "applies_to_decision" in columns:
            return

        # Migração v1 → v2: preserva documentos existentes e acrescenta o tipo de decisão.
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE policy_documents_v2 (
                    policy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    applies_to_decision TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reason_codes TEXT NOT NULL,
                    PRIMARY KEY (policy_id, version)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO policy_documents_v2
                    (policy_id, version, applies_to_decision, title, content, reason_codes)
                SELECT policy_id, version,
                       CASE policy_id
                           WHEN 'POL-001' THEN 'AUTHORIZED'
                           WHEN 'POL-002' THEN 'JUSTIFIED'
                           ELSE 'ANOMALY'
                       END,
                       title, content, reason_codes
                FROM policy_documents
                """
            )
            connection.execute("DROP TABLE policy_documents")
            connection.execute("ALTER TABLE policy_documents_v2 RENAME TO policy_documents")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def seed_demo_data(self, policy_version: str) -> None:
        people = [
            ("EMP001", "Lucas", "Analista de Infraestrutura", "Infraestrutura", 1),
            ("EMP002", "Mariana", "Analista de Segurança da Informação", "SecOps", 1),
            ("EMP003", "Roberto", "Desenvolvedor Frontend", "Desenvolvimento", 1),
        ]
        rooms = [
            ("sala_ti_01", "Sala de TI 01", "America/Sao_Paulo", "CRITICAL"),
            ("sala_ti_02", "Sala de TI 02", "America/Sao_Paulo", "HIGH"),
        ]
        cameras = [
            ("cam-ti-01", "sala_ti_01", 1, 0.85),
            ("cam-ti-02", "sala_ti_02", 1, 0.85),
        ]
        schedules: list[tuple[Any, ...]] = []
        for person_id, start, end in (
            ("EMP001", "08:00", "18:00"),
            ("EMP002", "09:00", "18:00"),
            ("EMP003", "09:00", "18:00"),
        ):
            for weekday in range(5):
                schedules.append(
                    (f"SCH-{person_id}-{weekday}", person_id, weekday, start, end)
                )

        policies = [
            (
                "POL-001",
                policy_version,
                "AUTHORIZED",
                "Acesso em horário regular",
                "Pessoa ativa, com permissão para a sala e dentro da escala possui acesso contextual padrão.",
                json.dumps(["WITHIN_SCHEDULE", "ROOM_PERMISSION_CONFIRMED"]),
            ),
            (
                "POL-002",
                policy_version,
                "JUSTIFIED",
                "Acesso emergencial fora do horário",
                "Fora da escala, somente incidente P1/P2 ativo, atribuído à pessoa e ligado à sala justifica o contexto.",
                json.dumps(["OUTSIDE_SCHEDULE", "QUALIFYING_INCIDENT"]),
            ),
            (
                "POL-003",
                policy_version,
                "ANOMALY",
                "Acesso atípico a sala crítica",
                "Falta de permissão ou ausência de incidente qualificável exige alerta e revisão humana.",
                json.dumps(["NO_ROOM_PERMISSION", "NO_QUALIFYING_INCIDENT"]),
            ),
            (
                "POL-004",
                policy_version,
                "ANOMALY",
                "Integridade da identificação",
                "Pessoa, câmera ou confiança inválida gera evento crítico sem acusação automática.",
                json.dumps(
                    [
                        "UNKNOWN_PERSON",
                        "INACTIVE_PERSON",
                        "UNKNOWN_CAMERA",
                        "LOW_RECOGNITION_CONFIDENCE",
                    ]
                ),
            ),
        ]

        with self.connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO people VALUES (?, ?, ?, ?, ?)", people
            )
            connection.executemany(
                "INSERT OR IGNORE INTO rooms VALUES (?, ?, ?, ?)", rooms
            )
            connection.executemany(
                "INSERT OR IGNORE INTO cameras VALUES (?, ?, ?, ?)", cameras
            )
            connection.executemany(
                "INSERT OR IGNORE INTO room_permissions VALUES (?, ?)",
                [
                    ("EMP001", "sala_ti_01"),
                    ("EMP002", "sala_ti_01"),
                    ("EMP002", "sala_ti_02"),
                ],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO work_schedules VALUES (?, ?, ?, ?, ?)",
                schedules,
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO incidents
                    (incident_id, title, severity, status, person_id, room_id,
                     active_from, active_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "INC-402",
                    "Queda de servidor",
                    "P1",
                    "OPEN",
                    "EMP002",
                    "sala_ti_01",
                    "2026-07-01T00:00:00+00:00",
                    "2026-07-31T23:59:59+00:00",
                ),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO policy_documents
                    (policy_id, version, applies_to_decision, title, content, reason_codes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                policies,
            )
            connection.commit()

    def healthcheck(self) -> bool:
        try:
            with self.connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False

    def sync_external_people(
        self,
        *,
        source_id: str,
        entries: list[dict[str, str]],
        synced_at: datetime,
    ) -> int:
        """Sincroniza identidade/nome sem guardar o identificador externo em claro."""

        if not source_id.strip():
            raise ValueError("source_id é obrigatório")
        instant = utc_iso(synced_at)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for entry in entries:
                connection.execute(
                    """
                    INSERT INTO people
                        (person_id, display_name, role_name, department, active)
                    VALUES (?, ?, 'Não informado', 'Importado da Intelbras', 1)
                    ON CONFLICT(person_id) DO UPDATE SET
                        display_name=excluded.display_name,
                        active=1
                    """,
                    (entry["person_id"], entry["display_name"]),
                )
                connection.execute(
                    """
                    INSERT INTO external_identities
                        (source_id, external_id_hash, person_id, image_sha256, synced_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, external_id_hash) DO UPDATE SET
                        person_id=excluded.person_id,
                        image_sha256=excluded.image_sha256,
                        synced_at=excluded.synced_at
                    """,
                    (
                        source_id,
                        entry["external_id_hash"],
                        entry["person_id"],
                        entry.get("image_sha256"),
                        instant,
                    ),
                )
            connection.commit()
        return len(entries)

    def policy_set_ready(self, policy_version: str) -> bool:
        try:
            with self.connect() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(DISTINCT applies_to_decision)
                    FROM policy_documents
                    WHERE version = ?
                    """,
                    (policy_version,),
                ).fetchone()
            return bool(row and row[0] >= 3)
        except sqlite3.Error:
            return False

    def get_access_context(
        self, camera_id: str, person_id: str, policy_version: str | None = None
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            camera_row = connection.execute(
                """
                SELECT c.*, r.display_name AS room_name, r.timezone, r.criticality
                FROM cameras c
                JOIN rooms r ON r.room_id = c.room_id
                WHERE c.camera_id = ?
                """,
                (camera_id,),
            ).fetchone()
            person_row = connection.execute(
                "SELECT * FROM people WHERE person_id = ?", (person_id,)
            ).fetchone()
            permissions = connection.execute(
                "SELECT room_id FROM room_permissions WHERE person_id = ?",
                (person_id,),
            ).fetchall()
            schedules = connection.execute(
                "SELECT * FROM work_schedules WHERE person_id = ? ORDER BY weekday, start_time",
                (person_id,),
            ).fetchall()
            incidents = connection.execute(
                "SELECT * FROM incidents WHERE person_id = ?",
                (person_id,),
            ).fetchall()
            if policy_version is None:
                policies = connection.execute(
                    """
                    SELECT p.* FROM policy_documents p
                    JOIN (
                        SELECT policy_id, MAX(version) AS version
                        FROM policy_documents GROUP BY policy_id
                    ) latest ON latest.policy_id = p.policy_id AND latest.version = p.version
                    ORDER BY p.policy_id
                    """
                ).fetchall()
            else:
                policies = connection.execute(
                    "SELECT * FROM policy_documents WHERE version = ? ORDER BY policy_id",
                    (policy_version,),
                ).fetchall()
            connection.commit()

        return {
            "camera": dict(camera_row) if camera_row else None,
            "person": dict(person_row) if person_row else None,
            "permission_room_ids": [row["room_id"] for row in permissions],
            "schedules": [dict(row) for row in schedules],
            "incidents": [dict(row) for row in incidents],
            "policies": [
                {**dict(row), "reason_codes": json.loads(row["reason_codes"])}
                for row in policies
            ],
        }

    def get_idempotency_record(self, event_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT event_id, payload_hash FROM access_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_evaluated_event(
        self,
        *,
        event: dict[str, Any],
        payload_hash: str,
        evaluation: dict[str, Any],
        received_at: datetime,
        policy_version: str,
        alert_channel: str,
        processing_ms: float,
        alert_include_personal_data: bool = False,
        public_base_url: str | None = None,
    ) -> dict[str, Any]:
        now = utc_iso(received_at)
        snapshot = evaluation["context_snapshot"]
        person = snapshot.get("person") or {}
        room = snapshot.get("room") or {}
        occurred_at = utc_iso(datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")))
        raw_payload = json.dumps(event, ensure_ascii=False, sort_keys=True)
        alert_id: str | None = None

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_hash FROM access_events WHERE event_id = ?",
                (event["event_id"],),
            ).fetchone()
            if existing:
                connection.rollback()
                state = "duplicate" if existing["payload_hash"] == payload_hash else "conflict"
                return {"state": state, "alert_id": None}

            connection.execute(
                """
                INSERT INTO access_events (
                    event_id, payload_hash, raw_payload, person_id, person_name,
                    role_name, department, room_id, room_name, camera_id,
                    occurred_at, received_at, processed_at, door_result,
                    recognition_confidence, decision, risk_level, risk_score,
                    reason_codes, source_ids, narrative, context_snapshot,
                    policy_version, processing_ms, alert_required, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    event["event_id"],
                    payload_hash,
                    raw_payload,
                    event["user_id"],
                    person.get("display_name"),
                    person.get("role_name"),
                    person.get("department"),
                    event["room_id"],
                    room.get("display_name"),
                    event["camera_id"],
                    occurred_at,
                    now,
                    now,
                    event["door_result"],
                    event.get("recognition_confidence"),
                    evaluation["decision"],
                    evaluation["risk_level"],
                    evaluation["risk_score"],
                    json.dumps(evaluation["reason_codes"], ensure_ascii=False),
                    json.dumps(evaluation["source_ids"], ensure_ascii=False),
                    evaluation["narrative"],
                    json.dumps(snapshot, ensure_ascii=False),
                    policy_version,
                    processing_ms,
                    int(evaluation["alert_required"]),
                    now,
                ),
            )

            if evaluation["alert_required"]:
                alert_id = str(uuid.uuid4())
                person_payload = {"external_id": event["user_id"]}
                if alert_include_personal_data:
                    person_payload.update(
                        {
                            "display_name": person.get("display_name") or "Não identificada",
                            "role": person.get("role_name") or "Não identificado",
                        }
                    )
                event_path = f"/v1/access-events/{event['event_id']}"
                alert_payload = {
                    "schema_version": "1.0",
                    "alert_id": alert_id,
                    "dedupe_key": f"access:{event['event_id']}",
                    "type": "ATYPICAL_ACCESS",
                    "severity": evaluation["risk_level"],
                    "occurred_at": event["timestamp"],
                    "event_id": event["event_id"],
                    "person": person_payload,
                    "room": {
                        "id": event["room_id"],
                        "name": room.get("display_name") or event["room_id"],
                    },
                    "decision": evaluation["decision"],
                    "reason_codes": evaluation["reason_codes"],
                    "door_result": event["door_result"],
                    "evidence": {
                        "schedule_match": snapshot.get("schedule_match"),
                        "permission_match": snapshot.get("permission_match"),
                        "entry_observation": snapshot.get("entry_observation"),
                        "incident_ids": [
                            item["incident_id"]
                            for item in snapshot.get("qualifying_incidents", [])
                        ],
                        "policy_ids": [
                            item["policy_id"] for item in snapshot.get("policies", [])
                        ],
                    },
                    "recommended_action": "Validar o evento com a equipe responsável.",
                    "event_url": f"{public_base_url}{event_path}" if public_base_url else event_path,
                }
                if alert_include_personal_data:
                    alert_payload["recognition_confidence"] = event.get(
                        "recognition_confidence"
                    )
                    alert_payload["narrative"] = evaluation["narrative"]
                connection.execute(
                    """
                    INSERT INTO alert_outbox (
                        alert_id, event_id, channel, severity, status, payload,
                        attempts, next_attempt_at, created_at
                    ) VALUES (?, ?, ?, ?, 'PENDING', ?, 0, ?, ?)
                    """,
                    (
                        alert_id,
                        event["event_id"],
                        alert_channel,
                        evaluation["risk_level"],
                        json.dumps(alert_payload, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            connection.commit()

        return {"state": "created", "alert_id": alert_id}

    def finalize_processing(self, event_id: str, processed_at: datetime, processing_ms: float) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE access_events SET processed_at = ?, processing_ms = ? WHERE event_id = ?",
                (utc_iso(processed_at), processing_ms, event_id),
            )
            connection.commit()

    @staticmethod
    def _decode_event(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for field in ("raw_payload", "reason_codes", "source_ids", "context_snapshot"):
            result[field] = json.loads(result[field])
        result["alert_required"] = bool(result["alert_required"])
        occurred_at = datetime.fromisoformat(result["occurred_at"].replace("Z", "+00:00"))
        received_at = datetime.fromisoformat(result["received_at"].replace("Z", "+00:00"))
        processed_at = (
            datetime.fromisoformat(result["processed_at"].replace("Z", "+00:00"))
            if result.get("processed_at")
            else None
        )
        result["ingestion_delay_ms"] = round(
            (received_at - occurred_at).total_seconds() * 1000, 2
        )
        result["decision_e2e_ms"] = (
            round((processed_at - occurred_at).total_seconds() * 1000, 2)
            if processed_at
            else None
        )
        if result.get("alert_id"):
            result["alert"] = {
                "alert_id": result.pop("alert_id"),
                "status": result.pop("alert_status"),
                "channel": result.pop("alert_channel"),
                "attempts": result.pop("alert_attempts"),
                "last_error": result.pop("alert_last_error"),
                "sent_at": result.pop("alert_sent_at"),
            }
        else:
            for key in (
                "alert_id",
                "alert_status",
                "alert_channel",
                "alert_attempts",
                "alert_last_error",
                "alert_sent_at",
            ):
                result.pop(key, None)
            result["alert"] = None
        result.pop("payload_hash", None)
        return result

    @staticmethod
    def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        mappings = {
            "from_timestamp": ("e.occurred_at >= ?", lambda value: utc_iso(value)),
            "to_timestamp": ("e.occurred_at <= ?", lambda value: utc_iso(value)),
            "room_id": ("e.room_id = ?", str),
            "user_id": ("e.person_id = ?", str),
            "decision": ("e.decision = ?", str),
            "risk_level": ("e.risk_level = ?", str),
            "alert_status": ("COALESCE(a.status, 'NONE') = ?", str),
        }
        for key, (clause, transform) in mappings.items():
            value = filters.get(key)
            if value is not None:
                clauses.append(clause)
                parameters.append(transform(value))
        query = filters.get("q")
        if query:
            clauses.append(
                "(e.event_id LIKE ? OR e.person_id LIKE ? OR e.person_name LIKE ? "
                "OR e.narrative LIKE ? OR e.role_name LIKE ?)"
            )
            wildcard = f"%{query}%"
            parameters.extend([wildcard] * 5)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", parameters

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT e.*,
                       a.alert_id,
                       a.status AS alert_status,
                       a.channel AS alert_channel,
                       a.attempts AS alert_attempts,
                       a.last_error AS alert_last_error,
                       a.sent_at AS alert_sent_at
                FROM access_events e
                LEFT JOIN alert_outbox a ON a.event_id = e.event_id
                WHERE e.event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return self._decode_event(row)

    def list_events(
        self,
        filters: dict[str, Any],
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where, parameters = self._where(filters)
        base = """
            FROM access_events e
            LEFT JOIN alert_outbox a ON a.event_id = e.event_id
        """
        with self.connect() as connection:
            connection.execute("BEGIN")
            total = connection.execute(
                f"SELECT COUNT(*) {base} {where}", parameters
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT e.*,
                       a.alert_id,
                       a.status AS alert_status,
                       a.channel AS alert_channel,
                       a.attempts AS alert_attempts,
                       a.last_error AS alert_last_error,
                       a.sent_at AS alert_sent_at
                {base} {where}
                ORDER BY e.occurred_at DESC, e.event_id DESC
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()
            connection.commit()
        return [self._decode_event(row) for row in rows], total  # type: ignore[misc]

    def metrics(self, filters: dict[str, Any]) -> dict[str, Any]:
        where, parameters = self._where(filters)
        with self.connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN e.decision = 'AUTHORIZED' THEN 1 ELSE 0 END) AS authorized,
                    SUM(CASE WHEN e.decision = 'JUSTIFIED' THEN 1 ELSE 0 END) AS justified,
                    SUM(CASE WHEN e.decision = 'ANOMALY' THEN 1 ELSE 0 END) AS anomalies,
                    SUM(CASE WHEN a.status = 'SENT' THEN 1 ELSE 0 END) AS alerts_sent,
                    SUM(CASE WHEN a.status IN ('FAILED', 'RETRYING', 'NOT_CONFIGURED') THEN 1 ELSE 0 END) AS alerts_failed,
                    SUM(CASE WHEN e.processing_ms < 3000 THEN 1 ELSE 0 END) AS within_sla,
                    AVG(e.processing_ms) AS average_processing_ms,
                    SUM(
                        CASE WHEN e.processed_at IS NOT NULL
                             AND (julianday(e.processed_at) - julianday(e.occurred_at)) * 86400000 >= 0
                             AND (julianday(e.processed_at) - julianday(e.occurred_at)) * 86400000 < 3000
                        THEN 1 ELSE 0 END
                    ) AS e2e_within_sla,
                    AVG((julianday(e.received_at) - julianday(e.occurred_at)) * 86400000)
                        AS average_ingestion_delay_ms,
                    AVG((julianday(e.processed_at) - julianday(e.occurred_at)) * 86400000)
                        AS average_decision_e2e_ms
                FROM access_events e
                LEFT JOIN alert_outbox a ON a.event_id = e.event_id
                {where}
                """,
                parameters,
            ).fetchone()
            timings = [
                timing[0]
                for timing in connection.execute(
                    f"""
                    SELECT e.processing_ms
                    FROM access_events e
                    LEFT JOIN alert_outbox a ON a.event_id = e.event_id
                    {where}
                    ORDER BY e.processing_ms
                    """,
                    parameters,
                ).fetchall()
            ]
            connection.commit()

        result = {key: (value or 0) for key, value in dict(row).items()}
        if timings and result["total"]:
            p95_index = max(0, min(len(timings) - 1, int(0.95 * len(timings) + 0.9999) - 1))
            result["p95_processing_ms"] = timings[p95_index]
            result["sla_percentage"] = round(result["within_sla"] * 100 / result["total"], 1)
            result["e2e_sla_percentage"] = round(
                result["e2e_within_sla"] * 100 / result["total"], 1
            )
        else:
            result["p95_processing_ms"] = 0
            result["sla_percentage"] = 100.0
            result["e2e_sla_percentage"] = 100.0
        result["average_processing_ms"] = round(result["average_processing_ms"], 2)
        result["average_ingestion_delay_ms"] = round(
            result["average_ingestion_delay_ms"], 2
        )
        result["average_decision_e2e_ms"] = round(
            result["average_decision_e2e_ms"], 2
        )
        result["api_within_sla"] = result["within_sla"]
        result["api_sla_percentage"] = result["sla_percentage"]
        return result

    def list_rooms(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT room_id, display_name, criticality FROM rooms ORDER BY display_name"
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_pending_alerts(
        self,
        now: datetime,
        *,
        include_not_configured: bool = False,
        limit: int = 1,
        lease_seconds: int = 30,
    ) -> list[dict[str, Any]]:
        statuses = ["PENDING", "RETRYING"]
        if include_not_configured:
            statuses.append("NOT_CONFIGURED")
        placeholders = ",".join("?" for _ in statuses)
        now_text = utc_iso(now)
        lease_until = utc_iso(now + timedelta(seconds=lease_seconds))
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                SELECT * FROM alert_outbox
                WHERE (
                    status IN ({placeholders})
                    OR (status = 'SENDING' AND next_attempt_at <= ?)
                )
                AND next_attempt_at <= ?
                ORDER BY created_at
                LIMIT ?
                """,
                (*statuses, now_text, now_text, limit),
            ).fetchall()
            if rows:
                connection.executemany(
                    """
                    UPDATE alert_outbox
                    SET status = 'SENDING', next_attempt_at = ?
                    WHERE alert_id = ?
                    """,
                    [(lease_until, row["alert_id"]) for row in rows],
                )
            connection.commit()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def update_alert_delivery(
        self,
        alert_id: str,
        *,
        status: str,
        attempts: int,
        error: str | None = None,
        retry_after_seconds: int = 0,
    ) -> None:
        now = utc_now()
        sent_at = utc_iso(now) if status == "SENT" else None
        next_attempt_at = utc_iso(now + timedelta(seconds=retry_after_seconds))
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE alert_outbox
                SET status = ?, attempts = ?, last_error = ?, next_attempt_at = ?, sent_at = ?
                WHERE alert_id = ?
                """,
                (status, attempts, error, next_attempt_at, sent_at, alert_id),
            )
            connection.commit()
