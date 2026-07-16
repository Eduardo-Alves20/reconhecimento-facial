from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from app.vision.evidence import EvidenceStoreError


class SupportsReference(Protocol):
    def add_reference(self, external_id: str, display_name: str, feature: Any) -> None: ...


class SupportsEvidenceStore(Protocol):
    def read(self, reference: str, *, now: datetime | None = None) -> bytes: ...


@dataclass(slots=True)
class _ReferenceCollector:
    model_version: str
    model_fingerprint: str
    entries: list[tuple[str, str, Any]]

    def add_reference(self, external_id: str, display_name: str, feature: Any) -> None:
        self.entries.append((external_id, display_name, feature))


class ReferenceStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVOKED = "REVOKED"


class LearnedGalleryError(RuntimeError):
    pass


class ReferenceNotFoundError(LearnedGalleryError):
    pass


class ReferenceStateError(LearnedGalleryError):
    pass


class ReferenceLimitError(LearnedGalleryError):
    pass


@dataclass(frozen=True, slots=True)
class LearnedReference:
    reference_id: str
    external_id: str
    display_name: str
    status: ReferenceStatus
    evidence_ref: str | None
    similarity: float | None
    quality: float | None
    model_version: str
    model_fingerprint: str
    embedding_fingerprint: str
    provenance: dict[str, Any]
    created_at: datetime
    reviewed_at: datetime | None
    reviewed_by: str | None
    rejection_reason: str | None
    revoked_at: datetime | None
    revoked_by: str | None
    revocation_reason: str | None


_SCHEMA = """
CREATE TABLE learned_refs (
    reference_id TEXT PRIMARY KEY,
    external_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    embedding_fingerprint TEXT NOT NULL,
    evidence_ref TEXT,
    similarity REAL,
    quality REAL,
    model_version TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    reviewed_by TEXT,
    rejection_reason TEXT,
    revoked_at TEXT,
    revoked_by TEXT,
    revocation_reason TEXT,
    CHECK (length(reference_id) = 32),
    CHECK (dim > 0),
    CHECK (length(embedding_fingerprint) = 64),
    CHECK (length(model_fingerprint) = 64),
    CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED', 'REVOKED'))
)
"""

_COLUMNS = {
    "reference_id",
    "external_id",
    "display_name",
    "embedding",
    "dim",
    "embedding_fingerprint",
    "evidence_ref",
    "similarity",
    "quality",
    "model_version",
    "model_fingerprint",
    "provenance",
    "status",
    "created_at",
    "reviewed_at",
    "reviewed_by",
    "rejection_reason",
    "revoked_at",
    "revoked_by",
    "revocation_reason",
}

_HEX_64 = frozenset("0123456789abcdef")


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("O instante precisa incluir fuso horário.")
    return value.astimezone(UTC).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _clean_text(value: Any, *, name: str, maximum: int) -> str:
    cleaned = unicodedata.normalize("NFC", str(value)).strip()
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{name} é inválido.")
    if any(unicodedata.category(character).startswith("C") for character in cleaned):
        raise ValueError(f"{name} é inválido.")
    return cleaned


def _clean_optional(value: Any, *, maximum: int) -> str | None:
    if value is None:
        return None
    cleaned = unicodedata.normalize("NFC", str(value)).strip()
    if not cleaned or len(cleaned) > maximum:
        return None
    if any(unicodedata.category(character).startswith("C") for character in cleaned):
        return None
    return cleaned


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX_64 for character in value)


def _model_fingerprint(version: str, supplied: Any = None) -> str:
    raw = str(supplied).strip().lower() if supplied is not None else ""
    if _is_sha256(raw):
        return raw
    material = raw or f"model-version:{version}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class LearnedGallery:
    def __init__(
        self,
        database_path: str | Path,
        *,
        max_per_person: int = 5,
        deduplication_similarity: float = 0.995,
        model_version: str | None = None,
        model_fingerprint: str | None = None,
    ) -> None:
        if max_per_person < 1:
            raise ValueError("max_per_person precisa ser positivo.")
        if not math.isfinite(deduplication_similarity) or not -1 <= deduplication_similarity <= 1:
            raise ValueError("deduplication_similarity precisa estar entre -1 e 1.")
        self.database_path = Path(database_path)
        self.max_per_person = max_per_person
        self.deduplication_similarity = deduplication_similarity
        self.model_version = (
            _clean_text(model_version, name="model_version", maximum=200)
            if model_version is not None
            else None
        )
        self.model_fingerprint = (
            _model_fingerprint(self.model_version or "unknown", model_fingerprint)
            if model_fingerprint is not None
            else None
        )
        self._counts: dict[str, int] = {}
        self.initialized = False
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy é necessário para o aprendizado facial.") from exc
        self.np = np

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._set_private_permissions(self.database_path.parent, directory=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='learned_refs'"
            ).fetchone()
            if exists is None:
                connection.execute(_SCHEMA)
            else:
                columns = {
                    row["name"] for row in connection.execute("PRAGMA table_info(learned_refs)")
                }
                if not _COLUMNS.issubset(columns):
                    self._migrate_legacy(connection, columns)
            self._create_indexes(connection)
            connection.execute("PRAGMA user_version=2")
        self._set_private_permissions(self.database_path, directory=False)
        self.initialized = True

    def _create_indexes(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learned_person_status
            ON learned_refs(external_id, status)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learned_model_status
            ON learned_refs(model_fingerprint, status)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learned_embedding
            ON learned_refs(model_fingerprint, embedding_fingerprint)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learned_model_person
            ON learned_refs(model_fingerprint, external_id, status)
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_learned_active_embedding
            ON learned_refs(model_fingerprint, embedding_fingerprint)
            WHERE status IN ('PENDING', 'APPROVED') AND revoked_at IS NULL
            """
        )

    def _migrate_legacy(self, connection: sqlite3.Connection, columns: set[str]) -> None:
        required = {"external_id", "display_name", "embedding", "dim"}
        if not required.issubset(columns):
            raise LearnedGalleryError("O banco de aprendizado possui esquema incompatível.")
        suffix = secrets.token_hex(4)
        legacy_table = f"learned_refs_legacy_{suffix}"
        connection.execute(f'ALTER TABLE learned_refs RENAME TO "{legacy_table}"')
        connection.execute(_SCHEMA)
        rows = connection.execute(f'SELECT * FROM "{legacy_table}" ORDER BY rowid').fetchall()
        seen: set[tuple[str, str]] = set()
        version = "legacy-unknown"
        model_fingerprint = _model_fingerprint(version)
        migrated_at = datetime.now(UTC)
        for position, row in enumerate(rows, start=1):
            row_keys = set(row.keys())
            raw = bytes(row["embedding"])
            dim = int(row["dim"])
            status = ReferenceStatus.PENDING
            rejection_reason = None
            try:
                embedding = self._normalise_embedding(
                    self.np.frombuffer(raw, dtype=self.np.float32)
                )
                if dim != 512:
                    raise ValueError("dimensão divergente")
                raw = embedding.tobytes()
                dim = 512
            except (TypeError, ValueError):
                status = ReferenceStatus.REJECTED
                rejection_reason = "MIGRATION_INVALID_EMBEDDING"
            embedding_fingerprint = hashlib.sha256(raw).hexdigest()
            duplicate_key = (model_fingerprint, embedding_fingerprint)
            if status == ReferenceStatus.PENDING and duplicate_key in seen:
                status = ReferenceStatus.REJECTED
                rejection_reason = "MIGRATION_DUPLICATE"
            seen.add(duplicate_key)
            created_at = (
                _parse_time(row["learned_at"])
                if "learned_at" in row_keys
                else _parse_time(row["created_at"])
                if "created_at" in row_keys
                else None
            ) or migrated_at
            legacy_id = row["id"] if "id" in row_keys else position
            provenance = json.dumps(
                {"source": "legacy_migration", "legacy_id": legacy_id},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            connection.execute(
                """
                INSERT INTO learned_refs (
                    reference_id, external_id, display_name, embedding, dim,
                    embedding_fingerprint, evidence_ref, similarity, quality,
                    model_version, model_fingerprint, provenance, status,
                    created_at, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._new_reference_id(connection),
                    _clean_text(row["external_id"], name="external_id", maximum=128),
                    _clean_text(row["display_name"], name="display_name", maximum=256),
                    raw,
                    dim,
                    embedding_fingerprint,
                    _clean_optional(
                        row["evidence_ref"] if "evidence_ref" in row_keys else None,
                        maximum=512,
                    ),
                    self._finite_or_none(row["similarity"] if "similarity" in row_keys else None),
                    self._finite_or_none(row["quality"] if "quality" in row_keys else None),
                    version,
                    model_fingerprint,
                    provenance,
                    status.value,
                    _iso(created_at),
                    rejection_reason,
                ),
            )
        connection.execute(f'DROP TABLE "{legacy_table}"')

    def load_into(
        self,
        engine: SupportsReference,
        *,
        allowed_external_ids: Iterable[str] | None = None,
    ) -> int:
        version, fingerprint = self._resolve_model(engine)
        loaded = 0
        self._counts.clear()
        if isinstance(allowed_external_ids, (str, bytes)):
            raise ValueError("allowed_external_ids precisa ser uma coleção de IDs.")
        allowed = (
            {_clean_text(value, name="external_id", maximum=128) for value in allowed_external_ids}
            if allowed_external_ids is not None
            else None
        )
        if allowed == set():
            return 0
        with self._connect() as connection:
            join = ""
            if allowed is not None:
                connection.execute(
                    """
                    CREATE TEMP TABLE allowed_learned_identities (
                        external_id TEXT PRIMARY KEY
                    ) WITHOUT ROWID
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO allowed_learned_identities(external_id)
                    VALUES (?)
                    """,
                    [(external_id,) for external_id in sorted(allowed)],
                )
                join = (
                    "JOIN allowed_learned_identities AS allowed "
                    "ON allowed.external_id = learned_refs.external_id"
                )
            rows = connection.execute(
                f"""
                SELECT learned_refs.external_id, learned_refs.display_name,
                       learned_refs.embedding, learned_refs.dim,
                       learned_refs.embedding_fingerprint
                FROM learned_refs {join}
                WHERE learned_refs.status='APPROVED'
                  AND learned_refs.revoked_at IS NULL
                  AND learned_refs.model_version=?
                  AND learned_refs.model_fingerprint=?
                ORDER BY learned_refs.created_at, learned_refs.reference_id
                """,
                (version, fingerprint),
            ).fetchall()
        for row in rows:
            embedding = self._validated_row_embedding(row)
            if embedding is None:
                continue
            external_id = str(row["external_id"])
            engine.add_reference(external_id, str(row["display_name"]), embedding)
            self._counts[external_id] = self._counts.get(external_id, 0) + 1
            loaded += 1
        return loaded

    def approved_references(
        self,
        engine: Any,
        *,
        allowed_external_ids: Iterable[str] | None = None,
    ) -> tuple[tuple[str, str, Any], ...]:
        version, fingerprint = self._resolve_model(engine)
        collector = _ReferenceCollector(version, fingerprint, [])
        self.load_into(
            collector,
            allowed_external_ids=allowed_external_ids,
        )
        return tuple(collector.entries)

    def consider(
        self,
        engine: SupportsReference,
        *,
        external_id: str | None,
        display_name: str | None,
        feature: Any,
        evidence_ref: str | None,
        when: datetime,
        similarity: float | None,
        quality: float | None,
        min_similarity: float,
        max_similarity: float,
        min_quality: float,
        model_version: str | None = None,
        model_fingerprint: str | None = None,
        provenance: Mapping[str, Any] | str | None = None,
    ) -> bool:
        if feature is None or not external_id or not display_name:
            return False
        if not self._passes_decision_gate(
            similarity=similarity,
            quality=quality,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            min_quality=min_quality,
        ):
            return False
        identity = _clean_text(external_id, name="external_id", maximum=128)
        name = _clean_text(display_name, name="display_name", maximum=256)
        if evidence_ref is not None and not _is_sha256(evidence_ref):
            raise ValueError("evidence_ref precisa ser uma referência opaca válida.")
        embedding = self._normalise_embedding(feature)
        embedding_fingerprint = hashlib.sha256(embedding.tobytes()).hexdigest()
        version, fingerprint = self._resolve_model(
            engine,
            version_override=model_version,
            fingerprint_override=model_fingerprint,
        )
        provenance_json = self._encode_provenance(provenance)
        created_at = _iso(when)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM learned_refs
                WHERE model_fingerprint=?
                  AND external_id=?
                  AND status IN ('PENDING', 'APPROVED')
                  AND revoked_at IS NULL
                """,
                (fingerprint, identity),
            ).fetchone()[0]
            if int(count) >= self.max_per_person:
                return False
            if self._is_duplicate(
                connection,
                external_id=identity,
                embedding=embedding,
                embedding_fingerprint=embedding_fingerprint,
                model_fingerprint=fingerprint,
            ):
                return False
            connection.execute(
                """
                INSERT INTO learned_refs (
                    reference_id, external_id, display_name, embedding, dim,
                    embedding_fingerprint, evidence_ref, similarity, quality,
                    model_version, model_fingerprint, provenance, status,
                    created_at
                ) VALUES (?, ?, ?, ?, 512, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (
                    self._new_reference_id(connection),
                    identity,
                    name,
                    embedding.tobytes(),
                    embedding_fingerprint,
                    evidence_ref,
                    float(similarity),
                    float(quality),
                    version,
                    fingerprint,
                    provenance_json,
                    created_at,
                ),
            )
        return True

    def list_references(
        self,
        *,
        status: ReferenceStatus | str | None = None,
        external_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LearnedReference]:
        if limit < 1 or limit > 1_000 or offset < 0:
            raise ValueError("Paginação inválida.")
        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            resolved_status = ReferenceStatus(status)
            clauses.append("status=?")
            values.append(resolved_status.value)
        if external_id is not None:
            clauses.append("external_id=?")
            values.append(_clean_text(external_id, name="external_id", maximum=128))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM learned_refs
                {where}
                ORDER BY created_at DESC, reference_id
                LIMIT ? OFFSET ?
                """,
                values,
            ).fetchall()
        return [self._to_reference(row) for row in rows]

    def get(self, reference_id: str) -> LearnedReference:
        reference_id = self._validate_reference_id(reference_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM learned_refs WHERE reference_id=?", (reference_id,)
            ).fetchone()
        if row is None:
            raise ReferenceNotFoundError("Referência não encontrada.")
        return self._to_reference(row)

    def approve(
        self,
        reference_id: str,
        *,
        reviewed_by: str,
        when: datetime | None = None,
        engine: Any = None,
        model_version: str | None = None,
        model_fingerprint: str | None = None,
        evidence_store: SupportsEvidenceStore | None = None,
        require_evidence: bool = False,
    ) -> LearnedReference:
        if require_evidence and evidence_store is None:
            raise ValueError("evidence_store é obrigatório para validar a evidência.")
        if model_fingerprint is not None and engine is None and model_version is None:
            raise ValueError("model_version é obrigatório ao informar model_fingerprint.")
        binding = (
            self._resolve_model(
                engine,
                version_override=model_version,
                fingerprint_override=model_fingerprint,
            )
            if engine is not None or model_version is not None or model_fingerprint is not None
            else None
        )
        return self._review(
            reference_id,
            target=ReferenceStatus.APPROVED,
            reviewed_by=reviewed_by,
            reason=None,
            when=when,
            model_binding=binding,
            evidence_store=evidence_store,
            require_evidence=require_evidence,
        )

    def reject(
        self,
        reference_id: str,
        *,
        reviewed_by: str,
        reason: str,
        when: datetime | None = None,
    ) -> LearnedReference:
        return self._review(
            reference_id,
            target=ReferenceStatus.REJECTED,
            reviewed_by=reviewed_by,
            reason=reason,
            when=when,
            model_binding=None,
            evidence_store=None,
            require_evidence=False,
        )

    def revoke(
        self,
        reference_id: str,
        *,
        revoked_by: str,
        reason: str,
        when: datetime | None = None,
    ) -> LearnedReference:
        reference_id = self._validate_reference_id(reference_id)
        reviewer = _clean_text(revoked_by, name="revoked_by", maximum=128)
        safe_reason = _clean_text(reason, name="reason", maximum=500)
        instant = _iso(when or datetime.now(UTC))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM learned_refs WHERE reference_id=?", (reference_id,)
            ).fetchone()
            if row is None:
                raise ReferenceNotFoundError("Referência não encontrada.")
            current = ReferenceStatus(row["status"])
            if current == ReferenceStatus.REVOKED:
                return self._to_reference(row)
            if current != ReferenceStatus.APPROVED:
                raise ReferenceStateError("Somente referências aprovadas podem ser revogadas.")
            connection.execute(
                """
                UPDATE learned_refs
                SET status='REVOKED', revoked_at=?, revoked_by=?,
                    revocation_reason=?
                WHERE reference_id=? AND status='APPROVED'
                """,
                (instant, reviewer, safe_reason, reference_id),
            )
            updated = connection.execute(
                "SELECT * FROM learned_refs WHERE reference_id=?", (reference_id,)
            ).fetchone()
        return self._to_reference(updated)

    def active_reference_ids(self, external_id: str | None = None) -> set[str]:
        values: tuple[Any, ...] = ()
        filter_sql = ""
        if external_id is not None:
            filter_sql = " AND external_id=?"
            values = (_clean_text(external_id, name="external_id", maximum=128),)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT reference_id FROM learned_refs
                WHERE status='APPROVED' AND revoked_at IS NULL{filter_sql}
                """,
                values,
            ).fetchall()
        return {str(row["reference_id"]) for row in rows}

    def active_external_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT external_id FROM learned_refs
                WHERE status='APPROVED' AND revoked_at IS NULL
                """
            ).fetchall()
        return {str(row["external_id"]) for row in rows}

    def _review(
        self,
        reference_id: str,
        *,
        target: ReferenceStatus,
        reviewed_by: str,
        reason: str | None,
        when: datetime | None,
        model_binding: tuple[str, str] | None,
        evidence_store: SupportsEvidenceStore | None,
        require_evidence: bool,
    ) -> LearnedReference:
        reference_id = self._validate_reference_id(reference_id)
        reviewer = _clean_text(reviewed_by, name="reviewed_by", maximum=128)
        safe_reason = (
            _clean_text(reason, name="reason", maximum=500) if reason is not None else None
        )
        review_instant = when or datetime.now(UTC)
        instant = _iso(review_instant)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM learned_refs WHERE reference_id=?", (reference_id,)
            ).fetchone()
            if row is None:
                raise ReferenceNotFoundError("Referência não encontrada.")
            current = ReferenceStatus(row["status"])
            if current == target:
                if target == ReferenceStatus.APPROVED:
                    self._validate_review_evidence(
                        row,
                        evidence_store=evidence_store,
                        require_evidence=require_evidence,
                        when=review_instant,
                    )
                if model_binding is not None and model_binding != (
                    row["model_version"],
                    row["model_fingerprint"],
                ):
                    raise ReferenceStateError("Uma referência aprovada não pode trocar de modelo.")
                return self._to_reference(row)
            if current != ReferenceStatus.PENDING:
                raise ReferenceStateError("A referência não está pendente.")
            version = str(row["model_version"])
            fingerprint = str(row["model_fingerprint"])
            if target == ReferenceStatus.APPROVED:
                embedding = self._validated_row_embedding(row)
                if embedding is None:
                    raise ReferenceStateError("A referência falhou na validação de integridade.")
                self._validate_review_evidence(
                    row,
                    evidence_store=evidence_store,
                    require_evidence=require_evidence,
                    when=review_instant,
                )
                if model_binding is not None:
                    if version != "legacy-unknown" and model_binding != (
                        version,
                        fingerprint,
                    ):
                        raise ReferenceStateError("O candidato pertence a outro modelo.")
                    version, fingerprint = model_binding
                if not _is_sha256(fingerprint):
                    raise ReferenceStateError("O fingerprint do modelo é inválido.")
                if self._is_duplicate(
                    connection,
                    external_id=str(row["external_id"]),
                    embedding=embedding,
                    embedding_fingerprint=str(row["embedding_fingerprint"]),
                    model_fingerprint=fingerprint,
                    exclude_reference_id=reference_id,
                ):
                    raise ReferenceStateError(
                        "Já existe uma referência equivalente para este modelo."
                    )
                approved_count = connection.execute(
                    """
                    SELECT COUNT(*) FROM learned_refs
                    WHERE model_fingerprint=? AND external_id=?
                      AND status='APPROVED'
                      AND revoked_at IS NULL AND reference_id<>?
                    """,
                    (fingerprint, row["external_id"], reference_id),
                ).fetchone()[0]
                if int(approved_count) >= self.max_per_person:
                    raise ReferenceLimitError("A pessoa atingiu o limite de referências.")
            connection.execute(
                """
                UPDATE learned_refs
                SET status=?, reviewed_at=?, reviewed_by=?,
                    rejection_reason=?, model_version=?,
                    model_fingerprint=?
                WHERE reference_id=? AND status='PENDING'
                """,
                (
                    target.value,
                    instant,
                    reviewer,
                    safe_reason if target == ReferenceStatus.REJECTED else None,
                    version,
                    fingerprint,
                    reference_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM learned_refs WHERE reference_id=?", (reference_id,)
            ).fetchone()
        return self._to_reference(updated)

    @staticmethod
    def _validate_review_evidence(
        row: sqlite3.Row,
        *,
        evidence_store: SupportsEvidenceStore | None,
        require_evidence: bool,
        when: datetime,
    ) -> None:
        if not require_evidence and evidence_store is None:
            return
        if evidence_store is None:
            raise ReferenceStateError("A aprovação exige o armazenamento de evidências.")
        evidence_ref = row["evidence_ref"]
        if not isinstance(evidence_ref, str) or not _is_sha256(evidence_ref):
            raise ReferenceStateError("O candidato não possui uma evidência válida.")
        try:
            evidence_store.read(evidence_ref, now=when)
        except (EvidenceStoreError, OSError, ValueError, sqlite3.Error) as exc:
            raise ReferenceStateError(
                "A evidência está ausente, expirada ou sem integridade."
            ) from exc

    def _passes_decision_gate(
        self,
        *,
        similarity: float | None,
        quality: float | None,
        min_similarity: float,
        max_similarity: float,
        min_quality: float,
    ) -> bool:
        values = (min_similarity, max_similarity, min_quality)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Os limites precisam ser finitos.")
        if not -1 <= min_similarity <= 1 or not -1 <= max_similarity <= 1:
            raise ValueError("Os limites de similaridade precisam estar entre -1 e 1.")
        if not 0 <= min_quality <= 1:
            raise ValueError("min_quality precisa estar entre 0 e 1.")
        if min_similarity > max_similarity:
            raise ValueError("min_similarity não pode superar max_similarity.")
        if similarity is None or quality is None:
            return False
        if not math.isfinite(similarity) or not math.isfinite(quality):
            return False
        if not -1 <= similarity <= 1 or not 0 <= quality <= 1:
            return False
        return quality >= min_quality and min_similarity <= similarity < max_similarity

    def _normalise_embedding(self, feature: Any) -> Any:
        try:
            embedding = self.np.asarray(feature, dtype=self.np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Embedding inválido.") from exc
        if embedding.shape != (512,) or not bool(self.np.isfinite(embedding).all()):
            raise ValueError("O embedding precisa ter 512 valores finitos.")
        norm = float(self.np.linalg.norm(embedding.astype(self.np.float64)))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("O embedding não pode ser nulo.")
        normalized = self.np.asarray(embedding / norm, dtype="<f4")
        if not bool(self.np.isfinite(normalized).all()):
            raise ValueError("O embedding normalizado é inválido.")
        return normalized

    def _validated_row_embedding(self, row: sqlite3.Row) -> Any | None:
        try:
            if int(row["dim"]) != 512:
                return None
            raw = bytes(row["embedding"])
            if len(raw) != 512 * 4:
                return None
            fingerprint = hashlib.sha256(raw).hexdigest()
            if not secrets.compare_digest(fingerprint, str(row["embedding_fingerprint"])):
                return None
            embedding = self.np.frombuffer(raw, dtype="<f4")
            if embedding.shape != (512,) or not bool(self.np.isfinite(embedding).all()):
                return None
            norm = float(self.np.linalg.norm(embedding.astype(self.np.float64)))
            if not math.isfinite(norm) or not 0.999 <= norm <= 1.001:
                return None
        except (TypeError, ValueError):
            return None
        return embedding.copy()

    def _is_duplicate(
        self,
        connection: sqlite3.Connection,
        *,
        external_id: str,
        embedding: Any,
        embedding_fingerprint: str,
        model_fingerprint: str,
        exclude_reference_id: str | None = None,
    ) -> bool:
        exclusion = " AND reference_id<>?" if exclude_reference_id is not None else ""
        exact_values: tuple[Any, ...] = (
            (model_fingerprint, embedding_fingerprint, exclude_reference_id)
            if exclude_reference_id is not None
            else (model_fingerprint, embedding_fingerprint)
        )
        exact = connection.execute(
            f"""
            SELECT 1 FROM learned_refs
            WHERE model_fingerprint=? AND embedding_fingerprint=?
              AND status IN ('PENDING', 'APPROVED')
              AND revoked_at IS NULL
              {exclusion}
            LIMIT 1
            """,
            exact_values,
        ).fetchone()
        if exact is not None:
            return True
        person_values: tuple[Any, ...] = (
            (model_fingerprint, external_id, exclude_reference_id)
            if exclude_reference_id is not None
            else (model_fingerprint, external_id)
        )
        rows = connection.execute(
            f"""
            SELECT embedding, dim, embedding_fingerprint
            FROM learned_refs
            WHERE model_fingerprint=? AND external_id=?
              AND status IN ('PENDING', 'APPROVED')
              AND revoked_at IS NULL
              {exclusion}
            """,
            person_values,
        ).fetchall()
        for row in rows:
            existing = self._validated_row_embedding(row)
            if existing is None:
                continue
            if float(self.np.dot(embedding, existing)) >= self.deduplication_similarity:
                return True
        return False

    def _resolve_model(
        self,
        engine: Any,
        *,
        version_override: str | None = None,
        fingerprint_override: str | None = None,
    ) -> tuple[str, str]:
        raw_version = (
            version_override
            or self.model_version
            or getattr(engine, "model_version", None)
            or "unknown"
        )
        version = _clean_text(raw_version, name="model_version", maximum=200)
        raw_fingerprint = (
            fingerprint_override
            or self.model_fingerprint
            or getattr(engine, "model_fingerprint", None)
        )
        return version, _model_fingerprint(version, raw_fingerprint)

    def _encode_provenance(self, provenance: Mapping[str, Any] | str | None) -> str:
        if provenance is None:
            value: dict[str, Any] = {"source": "vision_auto"}
        elif isinstance(provenance, str):
            value = {"source": _clean_text(provenance, name="provenance", maximum=128)}
        elif isinstance(provenance, Mapping):
            value = dict(provenance)
            value.setdefault("source", "vision_auto")
        else:
            raise ValueError("provenance precisa ser texto ou objeto JSON.")
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("provenance precisa ser serializável como JSON.") from exc
        if len(encoded.encode("utf-8")) > 4_096:
            raise ValueError("provenance excede o limite permitido.")
        return encoded

    def _new_reference_id(self, connection: sqlite3.Connection) -> str:
        for _ in range(10):
            candidate = secrets.token_hex(16)
            exists = connection.execute(
                "SELECT 1 FROM learned_refs WHERE reference_id=?", (candidate,)
            ).fetchone()
            if exists is None:
                return candidate
        raise LearnedGalleryError("Não foi possível gerar uma referência única.")

    def _validate_reference_id(self, value: str) -> str:
        reference_id = str(value).strip().lower()
        if len(reference_id) != 32 or any(character not in _HEX_64 for character in reference_id):
            raise ValueError("reference_id inválido.")
        return reference_id

    def _to_reference(self, row: sqlite3.Row) -> LearnedReference:
        try:
            provenance = json.loads(row["provenance"])
        except (TypeError, ValueError):
            provenance = {"source": "invalid"}
        if not isinstance(provenance, dict):
            provenance = {"source": "invalid"}
        created_at = _parse_time(row["created_at"])
        if created_at is None:
            raise LearnedGalleryError("A referência possui data inválida.")
        return LearnedReference(
            reference_id=str(row["reference_id"]),
            external_id=str(row["external_id"]),
            display_name=str(row["display_name"]),
            status=ReferenceStatus(row["status"]),
            evidence_ref=row["evidence_ref"],
            similarity=self._finite_or_none(row["similarity"]),
            quality=self._finite_or_none(row["quality"]),
            model_version=str(row["model_version"]),
            model_fingerprint=str(row["model_fingerprint"]),
            embedding_fingerprint=str(row["embedding_fingerprint"]),
            provenance=provenance,
            created_at=created_at,
            reviewed_at=_parse_time(row["reviewed_at"]),
            reviewed_by=row["reviewed_by"],
            rejection_reason=row["rejection_reason"],
            revoked_at=_parse_time(row["revoked_at"]),
            revoked_by=row["revoked_by"],
            revocation_reason=row["revocation_reason"],
        )

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _set_private_permissions(path: Path, *, directory: bool) -> None:
        if not path.exists():
            return
        try:
            os.chmod(path, 0o700 if directory else 0o600)
        except OSError:
            pass


__all__ = [
    "LearnedGallery",
    "LearnedGalleryError",
    "LearnedReference",
    "ReferenceLimitError",
    "ReferenceNotFoundError",
    "ReferenceStateError",
    "ReferenceStatus",
    "SupportsReference",
    "SupportsEvidenceStore",
]
