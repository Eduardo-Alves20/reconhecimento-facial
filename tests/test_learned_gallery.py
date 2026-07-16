from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.vision.evidence import EvidenceStore
from app.vision.learned import (
    LearnedGallery,
    LearnedGalleryError,
    ReferenceStateError,
    ReferenceStatus,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
EVIDENCE_REF = "a" * 64


class FakeEngine:
    def __init__(self, version: str = "arcface-test-v1") -> None:
        self.model_version = version
        self.model_fingerprint = hashlib.sha256(f"weights:{version}".encode()).hexdigest()
        self.references: list[tuple[str, str, np.ndarray]] = []

    def add_reference(self, external_id: str, display_name: str, feature: np.ndarray) -> None:
        self.references.append((external_id, display_name, feature))


def embedding(index: int, *, scale: float = 1.0) -> np.ndarray:
    value = np.zeros(512, dtype=np.float32)
    value[index] = scale
    return value


def consider(
    gallery: LearnedGallery,
    engine: FakeEngine,
    feature: np.ndarray,
    *,
    external_id: str = "EMP001",
    display_name: str = "Ana",
    evidence_ref: str = EVIDENCE_REF,
) -> bool:
    return gallery.consider(
        engine,
        external_id=external_id,
        display_name=display_name,
        feature=feature,
        evidence_ref=evidence_ref,
        when=NOW,
        similarity=0.72,
        quality=0.91,
        min_similarity=0.60,
        max_similarity=0.85,
        min_quality=0.70,
        provenance={"source": "camera", "camera_id": "cam-ti-01"},
    )


def test_candidate_stays_quarantined_until_review(tmp_path: Path) -> None:
    gallery = LearnedGallery(tmp_path / "learned.db")
    gallery.initialize()
    engine = FakeEngine()

    assert consider(gallery, engine, embedding(0, scale=3.0)) is True
    assert engine.references == []
    assert gallery.load_into(engine) == 0

    [candidate] = gallery.list_references()
    assert candidate.status == ReferenceStatus.PENDING
    assert candidate.model_version == engine.model_version
    assert candidate.model_fingerprint == engine.model_fingerprint
    assert candidate.provenance == {"source": "camera", "camera_id": "cam-ti-01"}
    assert candidate.evidence_ref == EVIDENCE_REF

    approved = gallery.approve(candidate.reference_id, reviewed_by="auditor")
    assert approved.status == ReferenceStatus.APPROVED
    assert gallery.active_reference_ids("EMP001") == {candidate.reference_id}
    assert gallery.active_external_ids() == {"EMP001"}
    assert gallery.load_into(engine) == 1
    assert len(engine.references) == 1
    assert float(np.linalg.norm(engine.references[0][2])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(512, dtype=np.float32),
        np.zeros(511, dtype=np.float32),
        np.full(512, np.nan, dtype=np.float32),
        np.full(512, np.inf, dtype=np.float32),
        np.zeros((1, 512), dtype=np.float32),
    ],
)
def test_rejects_invalid_embeddings(tmp_path: Path, invalid: np.ndarray) -> None:
    gallery = LearnedGallery(tmp_path / "learned.db")
    gallery.initialize()

    with pytest.raises(ValueError, match="embedding|Embedding"):
        consider(gallery, FakeEngine(), invalid)
    assert gallery.list_references() == []


def test_deduplication_and_limit_include_pending_candidates(tmp_path: Path) -> None:
    gallery = LearnedGallery(tmp_path / "learned.db", max_per_person=2)
    gallery.initialize()
    engine = FakeEngine()

    assert consider(gallery, engine, embedding(0)) is True
    near_duplicate = embedding(0)
    near_duplicate[1] = 0.01
    assert consider(gallery, engine, near_duplicate) is False
    assert consider(gallery, engine, embedding(1)) is True
    assert consider(gallery, engine, embedding(2)) is False

    pending = gallery.list_references(status=ReferenceStatus.PENDING)
    rejected = gallery.reject(
        pending[0].reference_id,
        reviewed_by="auditor",
        reason="imagem desfocada",
    )
    assert rejected.status == ReferenceStatus.REJECTED
    assert consider(gallery, engine, embedding(2)) is True


def test_limit_and_deduplication_are_scoped_to_model_fingerprint(
    tmp_path: Path,
) -> None:
    gallery = LearnedGallery(tmp_path / "learned.db", max_per_person=1)
    gallery.initialize()
    first_bundle = FakeEngine("arcface-test-v1")
    second_bundle = FakeEngine("arcface-test-v2")

    assert consider(gallery, first_bundle, embedding(0)) is True
    assert consider(gallery, first_bundle, embedding(1)) is False
    assert consider(gallery, second_bundle, embedding(0)) is True

    candidates = gallery.list_references(external_id="EMP001")
    assert {item.model_fingerprint for item in candidates} == {
        first_bundle.model_fingerprint,
        second_bundle.model_fingerprint,
    }
    for candidate in candidates:
        gallery.approve(candidate.reference_id, reviewed_by="auditor")
    assert {
        item.model_fingerprint for item in gallery.list_references(status=ReferenceStatus.APPROVED)
    } == {first_bundle.model_fingerprint, second_bundle.model_fingerprint}


def test_required_evidence_must_exist_be_current_and_match_its_digest(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(tmp_path / "evidence", ttl=timedelta(hours=1))
    evidence.initialize()
    jpeg = b"\xff\xd8\xff\xe0test\xff\xd9"
    record = evidence.save(jpeg, created_at=NOW)
    gallery = LearnedGallery(tmp_path / "learned.db")
    gallery.initialize()
    engine = FakeEngine()
    assert consider(
        gallery,
        engine,
        embedding(0),
        evidence_ref=record.reference,
    )
    candidate = gallery.list_references()[0]

    with pytest.raises(ValueError, match="evidence_store"):
        gallery.approve(
            candidate.reference_id,
            reviewed_by="auditor",
            require_evidence=True,
            when=NOW,
        )

    evidence.path_for(record.reference).write_bytes(b"\xff\xd8\xff\xe0altered\xff\xd9")
    with pytest.raises(ReferenceStateError, match="integridade"):
        gallery.approve(
            candidate.reference_id,
            reviewed_by="auditor",
            evidence_store=evidence,
            require_evidence=True,
            when=NOW,
        )

    evidence.path_for(record.reference).write_bytes(jpeg)
    approved = gallery.approve(
        candidate.reference_id,
        reviewed_by="auditor",
        evidence_store=evidence,
        require_evidence=True,
        when=NOW,
    )
    assert approved.status == ReferenceStatus.APPROVED

    expired_record = evidence.save(
        jpeg,
        created_at=NOW,
        ttl=timedelta(seconds=1),
    )
    assert consider(
        gallery,
        engine,
        embedding(1),
        evidence_ref=expired_record.reference,
    )
    expired_candidate = next(
        item
        for item in gallery.list_references(status=ReferenceStatus.PENDING)
        if item.evidence_ref == expired_record.reference
    )
    with pytest.raises(ReferenceStateError, match="expirada"):
        gallery.approve(
            expired_candidate.reference_id,
            reviewed_by="auditor",
            evidence_store=evidence,
            require_evidence=True,
            when=NOW + timedelta(seconds=2),
        )


def test_revoke_removes_reference_from_active_set_and_load(tmp_path: Path) -> None:
    gallery = LearnedGallery(tmp_path / "learned.db")
    gallery.initialize()
    engine = FakeEngine()
    assert consider(gallery, engine, embedding(3))
    candidate = gallery.list_references()[0]
    gallery.approve(candidate.reference_id, reviewed_by="auditor")

    revoked = gallery.revoke(
        candidate.reference_id,
        revoked_by="security-admin",
        reason="cadastro substituído",
    )
    assert revoked.status == ReferenceStatus.REVOKED
    assert revoked.revoked_by == "security-admin"
    assert gallery.active_reference_ids() == set()
    assert gallery.load_into(FakeEngine()) == 0

    with pytest.raises(ReferenceStateError):
        gallery.approve(candidate.reference_id, reviewed_by="auditor")


def test_load_only_accepts_people_present_in_official_gallery(tmp_path: Path) -> None:
    gallery = LearnedGallery(tmp_path / "learned.db")
    gallery.initialize()
    source_engine = FakeEngine()
    assert consider(gallery, source_engine, embedding(0))
    assert consider(
        gallery,
        source_engine,
        embedding(1),
        external_id="EMP002",
        display_name="Bruno",
    )
    for candidate in gallery.list_references():
        gallery.approve(candidate.reference_id, reviewed_by="auditor")

    target = FakeEngine()
    assert gallery.load_into(target, allowed_external_ids={"EMP001", "EMP999"}) == 1
    assert [item[0] for item in target.references] == ["EMP001"]
    assert gallery.load_into(FakeEngine(), allowed_external_ids=set()) == 0


def test_model_mismatch_and_embedding_tamper_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "learned.db"
    gallery = LearnedGallery(path)
    gallery.initialize()
    engine = FakeEngine()
    assert consider(gallery, engine, embedding(4))
    candidate = gallery.list_references()[0]
    gallery.approve(candidate.reference_id, reviewed_by="auditor")

    assert gallery.load_into(FakeEngine("other-model")) == 0
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE learned_refs SET embedding=? WHERE reference_id=?",
            (embedding(5).tobytes(), candidate.reference_id),
        )
    assert gallery.load_into(engine) == 0


def test_migrates_legacy_rows_to_quarantine_without_data_loss(tmp_path: Path) -> None:
    path = tmp_path / "learned.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE learned_refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                evidence_ref TEXT,
                similarity REAL,
                quality REAL,
                learned_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO learned_refs (
                external_id, display_name, embedding, dim, evidence_ref,
                similarity, quality, learned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "EMP001",
                "Ana",
                embedding(0, scale=2.0).tobytes(),
                512,
                EVIDENCE_REF,
                0.70,
                0.90,
                NOW.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO learned_refs (
                external_id, display_name, embedding, dim, evidence_ref,
                similarity, quality, learned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "EMP002",
                "Bruno",
                np.zeros(511, dtype=np.float32).tobytes(),
                511,
                None,
                0.75,
                0.92,
                NOW.isoformat(),
            ),
        )

    gallery = LearnedGallery(path)
    gallery.initialize()
    gallery.initialize()
    records = gallery.list_references(limit=10)

    assert len(records) == 2
    assert {record.status for record in records} == {
        ReferenceStatus.PENDING,
        ReferenceStatus.REJECTED,
    }
    migrated = next(record for record in records if record.status == ReferenceStatus.PENDING)
    invalid = next(record for record in records if record.status == ReferenceStatus.REJECTED)
    assert migrated.provenance["source"] == "legacy_migration"
    assert invalid.rejection_reason == "MIGRATION_INVALID_EMBEDDING"
    target_engine = FakeEngine()
    assert gallery.load_into(target_engine) == 0

    approved = gallery.approve(
        migrated.reference_id,
        reviewed_by="migration-review",
        engine=target_engine,
    )
    assert approved.model_version == target_engine.model_version
    assert approved.model_fingerprint == target_engine.model_fingerprint
    assert gallery.load_into(target_engine) == 1


def test_rejects_invalid_evidence_reference_and_non_json_provenance(
    tmp_path: Path,
) -> None:
    gallery = LearnedGallery(tmp_path / "learned.db")
    gallery.initialize()
    engine = FakeEngine()

    with pytest.raises(ValueError, match="evidence_ref"):
        gallery.consider(
            engine,
            external_id="EMP001",
            display_name="Ana",
            feature=embedding(0),
            evidence_ref="../../foto",
            when=NOW,
            similarity=0.72,
            quality=0.91,
            min_similarity=0.60,
            max_similarity=0.85,
            min_quality=0.70,
        )
    with pytest.raises(ValueError, match="JSON"):
        gallery.consider(
            engine,
            external_id="EMP001",
            display_name="Ana",
            feature=embedding(0),
            evidence_ref=EVIDENCE_REF,
            when=NOW,
            similarity=0.72,
            quality=0.91,
            min_similarity=0.60,
            max_similarity=0.85,
            min_quality=0.70,
            provenance={"bad": object()},
        )


def test_incompatible_legacy_schema_is_left_untouched(tmp_path: Path) -> None:
    path = tmp_path / "learned.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE learned_refs (legacy_value TEXT)")
        connection.execute("INSERT INTO learned_refs VALUES ('preserve-me')")

    with pytest.raises(LearnedGalleryError, match="incompatível"):
        LearnedGallery(path).initialize()

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT legacy_value FROM learned_refs").fetchall() == [
            ("preserve-me",)
        ]
