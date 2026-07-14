from __future__ import annotations

import json
from datetime import UTC, datetime

from app.database import Repository
from app.vision.directory_sync import sync_gallery_directory
from app.vision.identity import external_id_hash


def test_sync_imports_names_but_hashes_external_identifier(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    raw_external_id = "123.456.789-00"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "external_id": raw_external_id,
                        "display_name": "Pessoa Teste",
                        "image_path": "images/a.jpg",
                        "image_sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    repository = Repository(tmp_path / "audit.db")
    repository.initialize()
    count = sync_gallery_directory(
        manifest, repository, synced_at=datetime(2026, 7, 14, tzinfo=UTC)
    )
    assert count == 1
    with repository.connect() as connection:
        person = connection.execute("SELECT * FROM people").fetchone()
        identity = connection.execute("SELECT * FROM external_identities").fetchone()
    assert person["display_name"] == "Pessoa Teste"
    assert raw_external_id not in person["person_id"]
    assert identity["external_id_hash"] == external_id_hash(raw_external_id)
    assert raw_external_id not in identity["external_id_hash"]
