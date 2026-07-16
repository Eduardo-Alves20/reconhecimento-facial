from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_vision_models import (  # noqa: E402
    ALLOWED_MODELS,
    DEFAULT_ROOT,
    MANIFEST_NAME,
    MAX_MODEL_FILES,
    MODEL_DIR,
    REQUIRED_MODELS,
    ModelBundleError,
    bundle_fingerprint,
    sha256_file,
)


def prepare_bundle(
    root: str | Path,
    *,
    license_reference: str,
    replace: bool = False,
) -> str:
    bundle_root = Path(root)
    reference = license_reference.strip()
    if not reference or len(reference) > 500 or any(ord(char) < 32 for char in reference):
        raise ModelBundleError("A referência da licença é inválida.")
    if bundle_root.is_symlink():
        raise ModelBundleError("A raiz do bundle não pode ser um link simbólico.")

    model_directory = bundle_root / MODEL_DIR
    try:
        resolved_root = bundle_root.resolve(strict=True)
        resolved_models = model_directory.resolve(strict=True)
        resolved_models.relative_to(resolved_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ModelBundleError(f"Coloque os modelos licenciados em {model_directory}.") from exc
    if model_directory.is_symlink() or not resolved_models.is_dir():
        raise ModelBundleError("O diretório do modelo é inválido.")

    model_files = sorted(
        path
        for path in resolved_models.iterdir()
        if path.is_file() and path.suffix.lower() == ".onnx"
    )
    if not 1 <= len(model_files) <= MAX_MODEL_FILES:
        raise ModelBundleError("Quantidade de arquivos ONNX inválida.")
    if any(path.is_symlink() for path in model_files):
        raise ModelBundleError("Links simbólicos não são aceitos no bundle.")

    names = {path.name for path in model_files}
    unexpected = names - ALLOWED_MODELS
    if unexpected:
        raise ModelBundleError("Modelos não suportados: " + ", ".join(sorted(unexpected)) + ".")
    missing = REQUIRED_MODELS - names
    if missing:
        raise ModelBundleError("Modelos obrigatórios ausentes: " + ", ".join(sorted(missing)) + ".")

    files = [
        {
            "path": path.relative_to(resolved_root).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in model_files
    ]
    payload = {
        "schema_version": 1,
        "provider": "insightface",
        "model_pack": "buffalo_l",
        "license": {
            "acknowledged": True,
            "reference": reference,
        },
        "files": files,
    }
    payload["fingerprint"] = bundle_fingerprint(payload)

    destination = resolved_root / MANIFEST_NAME
    if destination.exists() and not replace:
        raise ModelBundleError(f"{MANIFEST_NAME} já existe; use --replace para substituí-lo.")
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{MANIFEST_NAME}.",
            suffix=".tmp",
            dir=resolved_root,
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return payload["fingerprint"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra um bundle InsightFace obtido e licenciado pelo operador."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("RAG_AUDIT_VISION_MODELS_DIR", str(DEFAULT_ROOT))),
    )
    parser.add_argument("--license-reference", required=True)
    parser.add_argument("--accept-model-license", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if not args.accept_model_license:
        print(
            "Confirme a autorização de uso com --accept-model-license.",
            file=sys.stderr,
        )
        return 2
    try:
        fingerprint = prepare_bundle(
            args.root,
            license_reference=args.license_reference,
            replace=args.replace,
        )
    except ModelBundleError as exc:
        print(f"Não foi possível preparar o bundle: {exc}", file=sys.stderr)
        return 1
    print(f"Bundle preparado. Defina RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256={fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
