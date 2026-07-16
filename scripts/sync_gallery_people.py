"""Sincroniza IDs e nomes da galeria privada com o banco do RAG-Audit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env.api")
load_dotenv(PROJECT_ROOT / ".env.vision")
load_dotenv(PROJECT_ROOT / ".env")

from app.database import Repository  # noqa: E402
from app.vision.directory_sync import DirectorySyncError, sync_gallery_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path(
            os.getenv(
                "RAG_AUDIT_GALLERY_MANIFEST",
                str(PROJECT_ROOT / "data" / "private" / "gallery" / "manifest.json"),
            )
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            os.getenv(
                "RAG_AUDIT_DB_PATH",
                str(PROJECT_ROOT / "data" / "api" / "rag_audit.db"),
            )
        ),
        help="banco SQLite da API",
    )
    args = parser.parse_args()
    repository = Repository(args.database)
    repository.initialize()
    try:
        count = sync_gallery_directory(args.manifest, repository)
    except DirectorySyncError as exc:
        print(f"Sincronização recusada: {exc}", file=sys.stderr)
        return 2
    print(f"{count} pessoa(s) sincronizada(s); credenciais e fotos não foram copiadas ao banco.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
