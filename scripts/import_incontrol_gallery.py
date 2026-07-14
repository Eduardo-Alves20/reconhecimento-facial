from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Keep direct execution (`python scripts/import_incontrol_gallery.py`) working.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.gallery import (  # noqa: E402
    ColumnMapping,
    GalleryImportError,
    ImportLimits,
    import_gallery,
)


DEFAULT_LIMITS = ImportLimits()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Importa com segurança um ZIP do Intelbras InControl ou uma pasta "
            "de fotos para uma galeria privada."
        )
    )
    parser.add_argument("source", type=Path, help="ZIP exportado ou diretório de origem")
    parser.add_argument("output", type=Path, help="novo diretório privado de saída")
    parser.add_argument("--id-column", help="nome exato da coluna de ID externo")
    parser.add_argument("--name-column", help="nome exato da coluna com o nome da pessoa")
    parser.add_argument("--photo-column", help="nome exato da coluna com o caminho da foto")
    parser.add_argument("--max-files", type=int, default=DEFAULT_LIMITS.max_files)
    parser.add_argument("--max-total-mib", type=int, default=1024)
    parser.add_argument("--max-file-mib", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_LIMITS.max_rows)
    parser.add_argument("--max-entries", type=int, default=DEFAULT_LIMITS.max_entries)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.id_column) != bool(args.name_column):
        parser.error("--id-column e --name-column devem ser informadas juntas")
    if args.photo_column and not args.id_column:
        parser.error("--photo-column exige --id-column e --name-column")

    mapping = None
    if args.id_column:
        mapping = ColumnMapping(args.id_column, args.name_column, args.photo_column)
    try:
        limits = ImportLimits(
            max_files=args.max_files,
            max_total_bytes=args.max_total_mib * 1024 * 1024,
            max_file_bytes=args.max_file_mib * 1024 * 1024,
            max_csv_bytes=min(16, args.max_file_mib) * 1024 * 1024,
            max_rows=args.max_rows,
            max_entries=args.max_entries,
        )
        result = import_gallery(
            args.source,
            args.output,
            mapping=mapping,
            limits=limits,
        )
    except (GalleryImportError, OSError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest_path": str(result.manifest_path),
                "imported_count": result.imported_count,
                "skipped_rows": result.skipped_rows,
                "warnings": list(result.warnings),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
