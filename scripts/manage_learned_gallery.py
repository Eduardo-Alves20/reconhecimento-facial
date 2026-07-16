from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env.vision")
load_dotenv(PROJECT_ROOT / ".env")

from app.vision.evidence import EvidenceStore, EvidenceStoreError  # noqa: E402
from app.vision.learned import (  # noqa: E402
    LearnedGallery,
    LearnedGalleryError,
    LearnedReference,
    ReferenceStatus,
)


def default_database_path() -> Path:
    configured = os.getenv("RAG_AUDIT_LEARNED_DB_PATH")
    if configured:
        return Path(configured)
    gallery_manifest = Path(
        os.getenv(
            "RAG_AUDIT_GALLERY_MANIFEST",
            str(PROJECT_ROOT / "data" / "private" / "gallery" / "manifest.json"),
        )
    )
    return gallery_manifest.parent / "learned.db"


def evidence_store_from_env(path: Path) -> EvidenceStore:
    evict_value = os.getenv("RAG_AUDIT_EVIDENCE_EVICT_OLDEST", "true").strip().lower()
    if evict_value in {"1", "true", "yes", "on", "sim"}:
        evict_oldest = True
    elif evict_value in {"0", "false", "no", "off", "nao", "não"}:
        evict_oldest = False
    else:
        raise ValueError("RAG_AUDIT_EVIDENCE_EVICT_OLDEST deve ser true ou false")
    return EvidenceStore(
        path,
        ttl=timedelta(days=int(os.getenv("RAG_AUDIT_EVIDENCE_TTL_DAYS", "30"))),
        max_storage_bytes=round(
            float(os.getenv("RAG_AUDIT_EVIDENCE_MAX_GB", "10")) * 1024 * 1024 * 1024
        ),
        max_item_bytes=round(
            float(os.getenv("RAG_AUDIT_EVIDENCE_MAX_ITEM_MB", "25")) * 1024 * 1024
        ),
        evict_oldest=evict_oldest,
    )


def reference_payload(reference: LearnedReference) -> dict[str, Any]:
    payload = asdict(reference)
    payload["status"] = reference.status.value
    for key, value in tuple(payload.items()):
        if isinstance(value, datetime):
            payload[key] = value.isoformat()
    return payload


def render_reference(reference: LearnedReference) -> str:
    payload = reference_payload(reference)
    labels = (
        ("Referência", "reference_id"),
        ("Pessoa", "external_id"),
        ("Nome", "display_name"),
        ("Status", "status"),
        ("Criada em", "created_at"),
        ("Similaridade", "similarity"),
        ("Qualidade", "quality"),
        ("Evidência", "evidence_ref"),
        ("Modelo", "model_version"),
        ("Fingerprint do modelo", "model_fingerprint"),
        ("Revisada em", "reviewed_at"),
        ("Revisada por", "reviewed_by"),
        ("Motivo da rejeição", "rejection_reason"),
        ("Revogada em", "revoked_at"),
        ("Revogada por", "revoked_by"),
        ("Motivo da revogação", "revocation_reason"),
    )
    lines = [
        f"{label}: {payload[key] if payload[key] is not None else '-'}" for label, key in labels
    ]
    lines.append(
        "Proveniência: " + json.dumps(payload["provenance"], ensure_ascii=False, sort_keys=True)
    )
    return "\n".join(lines)


def render_table(references: list[LearnedReference]) -> str:
    if not references:
        return "Nenhuma referência encontrada."
    columns = (
        ("REFERÊNCIA", 32),
        ("STATUS", 9),
        ("PESSOA", 18),
        ("NOME", 24),
        ("CRIADA EM", 25),
        ("QUAL.", 6),
        ("SIM.", 6),
    )
    header = "  ".join(label.ljust(width) for label, width in columns)
    rows = [header, "-" * len(header)]
    for item in references:
        quality = "-" if item.quality is None else f"{item.quality:.3f}"
        similarity = "-" if item.similarity is None else f"{item.similarity:.3f}"
        values = (
            item.reference_id,
            item.status.value,
            item.external_id,
            item.display_name,
            item.created_at.isoformat(),
            quality,
            similarity,
        )
        rows.append(
            "  ".join(
                _truncate(value, width).ljust(width)
                for value, (_, width) in zip(values, columns, strict=True)
            )
        )
    return "\n".join(rows)


def _truncate(value: str, width: int) -> str:
    text = str(value)
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _add_output_option(parser: argparse.ArgumentParser, *, preserve_existing: bool = False) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        default=argparse.SUPPRESS if preserve_existing else False,
        help="Emite JSON em vez da saída humana.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revisa referências faciais mantidas em quarentena."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=default_database_path(),
        help="Banco SQLite da galeria aprendida.",
    )
    _add_output_option(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Lista referências.")
    list_parser.add_argument(
        "--status",
        choices=[status.value for status in ReferenceStatus],
    )
    list_parser.add_argument("--external-id")
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.add_argument("--offset", type=int, default=0)
    _add_output_option(list_parser, preserve_existing=True)

    show_parser = subparsers.add_parser("show", help="Exibe uma referência.")
    show_parser.add_argument("reference_id")
    _add_output_option(show_parser, preserve_existing=True)

    approve_parser = subparsers.add_parser("approve", help="Aprova uma referência pendente.")
    approve_parser.add_argument("reference_id")
    approve_parser.add_argument("--operator", required=True)
    approve_parser.add_argument("--model-version")
    approve_parser.add_argument("--model-fingerprint")
    approve_parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Diretório privado gerenciado pelo EvidenceStore.",
    )
    _add_output_option(approve_parser, preserve_existing=True)

    reject_parser = subparsers.add_parser("reject", help="Rejeita uma referência pendente.")
    reject_parser.add_argument("reference_id")
    reject_parser.add_argument("--operator", required=True)
    reject_parser.add_argument("--reason", required=True)
    _add_output_option(reject_parser, preserve_existing=True)

    revoke_parser = subparsers.add_parser("revoke", help="Revoga uma referência aprovada.")
    revoke_parser.add_argument("reference_id")
    revoke_parser.add_argument("--operator", required=True)
    revoke_parser.add_argument("--reason", required=True)
    _add_output_option(revoke_parser, preserve_existing=True)
    return parser


def execute(args: argparse.Namespace) -> LearnedReference | list[LearnedReference]:
    database = args.database.expanduser()
    if database.is_symlink():
        raise LearnedGalleryError("O banco da galeria não pode ser um link simbólico.")
    if not database.is_file():
        raise LearnedGalleryError(f"Banco da galeria não encontrado: {database}")
    gallery = LearnedGallery(database)
    gallery.initialize()

    if args.command == "list":
        return gallery.list_references(
            status=args.status,
            external_id=args.external_id,
            limit=args.limit,
            offset=args.offset,
        )
    if args.command == "show":
        return gallery.get(args.reference_id)
    if args.command == "approve":
        current = gallery.get(args.reference_id)
        has_version = bool(args.model_version)
        has_fingerprint = bool(args.model_fingerprint)
        if has_version != has_fingerprint:
            raise ValueError("--model-version e --model-fingerprint devem ser informados juntos.")
        if has_fingerprint and (
            len(args.model_fingerprint) != 64
            or any(
                character not in "0123456789abcdef" for character in args.model_fingerprint.lower()
            )
        ):
            raise ValueError("--model-fingerprint deve ser um SHA-256 hexadecimal.")
        if current.model_version == "legacy-unknown" and not has_version:
            raise ValueError("Referências legadas exigem o modelo e o fingerprint verificados.")
        requires_evidence = current.model_version != "legacy-unknown"
        evidence_store = None
        if requires_evidence and args.evidence_dir is None:
            raise ValueError("--evidence-dir é obrigatório para candidatos não legados.")
        if args.evidence_dir is not None:
            evidence_dir = args.evidence_dir.expanduser()
            evidence_index = evidence_dir / ".evidence-index.sqlite3"
            if (
                evidence_dir.is_symlink()
                or not evidence_dir.is_dir()
                or evidence_index.is_symlink()
                or not evidence_index.is_file()
            ):
                raise ValueError("Diretório de evidências inválido ou não inicializado.")
            evidence_store = evidence_store_from_env(evidence_dir)
            evidence_store.initialize()
        return gallery.approve(
            args.reference_id,
            reviewed_by=args.operator,
            model_version=args.model_version,
            model_fingerprint=args.model_fingerprint,
            evidence_store=evidence_store,
            require_evidence=requires_evidence,
        )
    if args.command == "reject":
        return gallery.reject(
            args.reference_id,
            reviewed_by=args.operator,
            reason=args.reason,
        )
    if args.command == "revoke":
        return gallery.revoke(
            args.reference_id,
            revoked_by=args.operator,
            reason=args.reason,
        )
    raise ValueError("Comando inválido.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except (EvidenceStoreError, LearnedGalleryError, ValueError, OSError) as exc:
        if args.json_output:
            print(
                json.dumps(
                    {"ok": False, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        else:
            print(f"Erro: {exc}", file=sys.stderr)
        return 1

    if isinstance(result, list):
        if args.json_output:
            output: Any = [reference_payload(item) for item in result]
            print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        else:
            print(render_table(result))
    elif args.json_output:
        print(json.dumps(reference_payload(result), ensure_ascii=False, sort_keys=True))
    else:
        print(render_reference(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
