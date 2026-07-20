"""Sincroniza o manifesto sanitizado da galeria com o cadastro do QTA."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.database import Repository

from .identity import external_id_hash, safe_person_id


class DirectorySyncError(ValueError):
    pass


def sync_gallery_directory(
    manifest_path: str | Path,
    repository: Repository,
    *,
    source_id: str = "intelbras-incontrol",
    synced_at: datetime | None = None,
) -> int:
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DirectorySyncError("Manifesto da galeria inválido.") from exc
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("entries"), list):
        raise DirectorySyncError("Estrutura do manifesto não suportada.")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in manifest["entries"]:
        if not isinstance(item, dict):
            continue
        external_id = str(item.get("external_id", "")).strip()
        display_name = str(item.get("display_name", "")).strip()
        image_sha256 = str(item.get("image_sha256", "")).strip().lower()
        if not external_id or not display_name or len(display_name) > 200:
            continue
        person_id = safe_person_id(external_id)
        if person_id in seen:
            continue
        seen.add(person_id)
        entries.append(
            {
                "person_id": person_id,
                "display_name": display_name,
                "external_id_hash": external_id_hash(external_id),
                "image_sha256": image_sha256,
            }
        )
    if not entries:
        raise DirectorySyncError("Nenhuma identidade válida no manifesto.")
    return repository.sync_external_people(
        source_id=source_id,
        entries=entries,
        synced_at=synced_at or datetime.now(UTC),
    )
