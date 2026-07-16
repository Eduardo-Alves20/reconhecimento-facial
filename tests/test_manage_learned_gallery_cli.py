from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.vision.evidence import EvidenceStore
from app.vision.learned import LearnedGallery, ReferenceStatus
from scripts.manage_learned_gallery import build_parser, main


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


class FakeEngine:
    model_version = "arcface-test-v1"
    model_fingerprint = hashlib.sha256(b"arcface-test-v1").hexdigest()

    def add_reference(self, external_id: str, display_name: str, feature: np.ndarray) -> None:
        pass


def seed_candidate(
    database: Path,
    *,
    external_id: str = "EMP001",
    display_name: str = "Ana",
    index: int = 0,
    evidence_ref: str = "a" * 64,
) -> str:
    gallery = LearnedGallery(database)
    gallery.initialize()
    feature = np.zeros(512, dtype=np.float32)
    feature[index] = 1
    assert gallery.consider(
        FakeEngine(),
        external_id=external_id,
        display_name=display_name,
        feature=feature,
        evidence_ref=evidence_ref,
        when=NOW,
        similarity=0.75,
        quality=0.90,
        min_similarity=0.60,
        max_similarity=0.85,
        min_quality=0.70,
    )
    return gallery.list_references(external_id=external_id)[0].reference_id


def create_evidence(root: Path) -> tuple[Path, str]:
    evidence_dir = root / "evidence"
    store = EvidenceStore(evidence_dir, evict_oldest=True)
    store.initialize()
    record = store.save(b"\xff\xd8\xff\xe0test\xff\xd9")
    return evidence_dir, record.reference


def create_legacy_candidate(database: Path) -> None:
    feature = np.zeros(512, dtype=np.float32)
    feature[0] = 1
    with sqlite3.connect(database) as connection:
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
                feature.tobytes(),
                512,
                "b" * 64,
                0.75,
                0.90,
                NOW.isoformat(),
            ),
        )


@pytest.mark.parametrize(
    "arguments",
    (
        ["--json", "list"],
        ["list", "--json"],
    ),
)
def test_json_option_works_before_or_after_subcommand(arguments: list[str]) -> None:
    assert build_parser().parse_args(arguments).json_output is True


def test_list_show_approve_and_revoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    database = tmp_path / "learned.db"
    evidence_dir, evidence_ref = create_evidence(tmp_path)
    reference_id = seed_candidate(database, evidence_ref=evidence_ref)

    assert main(["--database", str(database), "--json", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["reference_id"] == reference_id
    assert listed[0]["status"] == "PENDING"

    assert (
        main(
            [
                "--database",
                str(database),
                "approve",
                reference_id,
                "--operator",
                "auditor",
                "--evidence-dir",
                str(evidence_dir),
                "--json",
            ]
        )
        == 0
    )
    approved = json.loads(capsys.readouterr().out)
    assert approved["status"] == "APPROVED"
    assert approved["reviewed_by"] == "auditor"

    assert (
        main(
            [
                "--database",
                str(database),
                "revoke",
                reference_id,
                "--operator",
                "security-admin",
                "--reason",
                "cadastro substituído",
                "--json",
            ]
        )
        == 0
    )
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["status"] == "REVOKED"
    assert revoked["revocation_reason"] == "cadastro substituído"


def test_non_legacy_approval_requires_intact_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "learned.db"
    evidence_dir, evidence_ref = create_evidence(tmp_path)
    reference_id = seed_candidate(database, evidence_ref=evidence_ref)

    assert (
        main(
            [
                "--database",
                str(database),
                "approve",
                reference_id,
                "--operator",
                "auditor",
                "--json",
            ]
        )
        == 1
    )
    assert "--evidence-dir" in json.loads(capsys.readouterr().err)["error"]

    EvidenceStore(evidence_dir).path_for(evidence_ref).write_bytes(
        b"\xff\xd8\xff\xe0altered\xff\xd9"
    )
    assert (
        main(
            [
                "--database",
                str(database),
                "approve",
                reference_id,
                "--operator",
                "auditor",
                "--evidence-dir",
                str(evidence_dir),
                "--json",
            ]
        )
        == 1
    )
    assert "integridade" in json.loads(capsys.readouterr().err)["error"]


def test_non_legacy_approval_rejects_missing_evidence_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "learned.db"
    evidence_dir, _ = create_evidence(tmp_path)
    reference_id = seed_candidate(database, evidence_ref="f" * 64)

    assert (
        main(
            [
                "--database",
                str(database),
                "approve",
                reference_id,
                "--operator",
                "auditor",
                "--evidence-dir",
                str(evidence_dir),
                "--json",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert "ausente" in error["error"]


def test_reject_requires_reason_and_records_operator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "learned.db"
    reference_id = seed_candidate(database)

    with pytest.raises(SystemExit):
        main(
            [
                "--database",
                str(database),
                "reject",
                reference_id,
                "--operator",
                "auditor",
            ]
        )

    assert (
        main(
            [
                "--database",
                str(database),
                "reject",
                reference_id,
                "--operator",
                "auditor",
                "--reason",
                "rosto parcialmente coberto",
                "--json",
            ]
        )
        == 0
    )
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["status"] == "REJECTED"
    assert rejected["reviewed_by"] == "auditor"
    assert rejected["rejection_reason"] == "rosto parcialmente coberto"


def test_legacy_approval_requires_verified_model_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "legacy.db"
    create_legacy_candidate(database)
    gallery = LearnedGallery(database)
    gallery.initialize()
    reference_id = gallery.list_references(status=ReferenceStatus.PENDING)[0].reference_id

    assert (
        main(
            [
                "--database",
                str(database),
                "approve",
                reference_id,
                "--operator",
                "migration-review",
                "--json",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert "legadas" in error["error"]

    assert (
        main(
            [
                "--database",
                str(database),
                "approve",
                reference_id,
                "--operator",
                "migration-review",
                "--model-version",
                "insightface-test-v2",
                "--model-fingerprint",
                "invalid",
                "--json",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert "SHA-256" in error["error"]

    fingerprint = "c" * 64
    assert (
        main(
            [
                "--database",
                str(database),
                "approve",
                reference_id,
                "--operator",
                "migration-review",
                "--model-version",
                "insightface-test-v2",
                "--model-fingerprint",
                fingerprint,
                "--json",
            ]
        )
        == 0
    )
    approved = json.loads(capsys.readouterr().out)
    assert approved["model_version"] == "insightface-test-v2"
    assert approved["model_fingerprint"] == fingerprint
