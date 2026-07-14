from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.database import Repository


def test_initialize_migrates_v1_policy_table_without_losing_documents(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE policy_documents (
                policy_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                reason_codes TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO policy_documents VALUES (?, ?, ?, ?, ?)",
            (
                "POL-002",
                "2026.1",
                "Política legada",
                "Conteúdo preservado",
                json.dumps(["QUALIFYING_INCIDENT"]),
            ),
        )

    repository = Repository(database_path)
    repository.initialize()

    with repository.connect() as connection:
        columns = {
            row["name"] for row in connection.execute(
                "PRAGMA table_info(policy_documents)"
            ).fetchall()
        }
        row = connection.execute(
            "SELECT * FROM policy_documents WHERE policy_id = 'POL-002'"
        ).fetchone()
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert "applies_to_decision" in columns
    assert row["applies_to_decision"] == "JUSTIFIED"
    assert row["title"] == "Política legada"
    assert row["content"] == "Conteúdo preservado"
    assert user_version == 3


def test_policy_versions_are_kept_as_immutable_rows(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "versions.sqlite3")
    repository.initialize()
    repository.seed_demo_data("2026.1")
    repository.seed_demo_data("2026.2")

    with repository.connect() as connection:
        versions = [
            row["version"]
            for row in connection.execute(
                """
                SELECT version FROM policy_documents
                WHERE policy_id = 'POL-001' ORDER BY version
                """
            ).fetchall()
        ]

    assert versions == ["2026.1", "2026.2"]
