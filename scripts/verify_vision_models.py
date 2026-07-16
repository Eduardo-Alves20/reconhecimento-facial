from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "vision-models"
MANIFEST_NAME = "bundle-manifest.json"
MODEL_DIR = Path("models") / "buffalo_l"
REQUIRED_MODELS = frozenset({"det_10g.onnx", "w600k_r50.onnx"})
ALLOWED_MODELS = REQUIRED_MODELS | {
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MODEL_FILES = 32


class ModelBundleError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_fingerprint(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "fingerprint"}
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_model_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parent != MODEL_DIR
        or relative.suffix.lower() != ".onnx"
    ):
        raise ModelBundleError(f"Caminho de modelo inválido: {relative_path!r}.")
    target = root / relative
    if target.is_symlink():
        raise ModelBundleError(f"Links simbólicos não são aceitos: {relative_path}.")
    try:
        target.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ModelBundleError(f"Modelo ausente ou fora do bundle: {relative_path}.") from exc
    if not target.is_file():
        raise ModelBundleError(f"Modelo ausente: {relative_path}.")
    return target


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ModelBundleError("O manifesto não pode ser um link simbólico.")
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ModelBundleError("O manifesto excede o limite de tamanho.")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelBundleError(f"Manifesto ausente: {manifest_path}.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelBundleError("Não foi possível ler o manifesto do bundle.") from exc
    if not isinstance(payload, dict):
        raise ModelBundleError("O manifesto deve ser um objeto JSON.")
    return payload


def verify_bundle(root: str | Path, expected_fingerprint: str) -> str:
    bundle_root = Path(root)
    if bundle_root.is_symlink():
        raise ModelBundleError("A raiz do bundle não pode ser um link simbólico.")
    try:
        resolved_root = bundle_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelBundleError(f"Diretório de modelos ausente: {bundle_root}.") from exc
    if not resolved_root.is_dir():
        raise ModelBundleError("A raiz do bundle não é um diretório.")

    expected = expected_fingerprint.strip().lower()
    if not SHA256_RE.fullmatch(expected):
        raise ModelBundleError(
            "RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256 deve conter 64 caracteres hexadecimais."
        )

    payload = _load_manifest(resolved_root)
    if set(payload) != {
        "schema_version",
        "provider",
        "model_pack",
        "license",
        "files",
        "fingerprint",
    }:
        raise ModelBundleError("O manifesto contém campos ausentes ou não suportados.")
    if (
        payload["schema_version"] != 1
        or payload["provider"] != "insightface"
        or payload["model_pack"] != "buffalo_l"
    ):
        raise ModelBundleError("Bundle InsightFace não suportado.")

    license_data = payload["license"]
    if (
        not isinstance(license_data, dict)
        or set(license_data) != {"acknowledged", "reference"}
        or license_data.get("acknowledged") is not True
        or not isinstance(license_data.get("reference"), str)
        or not license_data["reference"].strip()
    ):
        raise ModelBundleError("A autorização de uso dos pesos não foi registrada.")

    files = payload["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_MODEL_FILES:
        raise ModelBundleError("A lista de modelos é inválida.")

    declared: set[str] = set()
    names: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise ModelBundleError("Entrada de modelo inválida no manifesto.")
        relative_path = item["path"]
        expected_hash = item["sha256"]
        expected_size = item["size"]
        if (
            not isinstance(relative_path, str)
            or relative_path in declared
            or not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise ModelBundleError("Metadados de modelo inválidos no manifesto.")
        model_path = _safe_model_path(resolved_root, relative_path)
        if model_path.stat().st_size != expected_size:
            raise ModelBundleError(f"Tamanho divergente para {relative_path}.")
        if not hmac.compare_digest(sha256_file(model_path), expected_hash):
            raise ModelBundleError(f"SHA-256 divergente para {relative_path}.")
        declared.add(relative_path)
        names.add(model_path.name)

    unexpected = names - ALLOWED_MODELS
    if unexpected:
        raise ModelBundleError("Modelos não suportados: " + ", ".join(sorted(unexpected)) + ".")
    missing = REQUIRED_MODELS - names
    if missing:
        raise ModelBundleError("Modelos obrigatórios ausentes: " + ", ".join(sorted(missing)) + ".")

    model_directory = resolved_root / MODEL_DIR
    actual = {
        path.relative_to(resolved_root).as_posix()
        for path in model_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".onnx"
    }
    if actual != declared:
        raise ModelBundleError("Há modelos ONNX não declarados ou ausentes no bundle.")

    manifest_fingerprint = payload["fingerprint"]
    if not isinstance(manifest_fingerprint, str) or not SHA256_RE.fullmatch(manifest_fingerprint):
        raise ModelBundleError("Fingerprint inválido no manifesto.")
    calculated = bundle_fingerprint(payload)
    if not hmac.compare_digest(calculated, manifest_fingerprint):
        raise ModelBundleError("O manifesto foi alterado após sua geração.")
    if not hmac.compare_digest(calculated, expected):
        raise ModelBundleError("O bundle não corresponde ao fingerprint autorizado.")
    return calculated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifica a integridade e a autorização do bundle InsightFace."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("RAG_AUDIT_VISION_MODELS_DIR", str(DEFAULT_ROOT))),
    )
    parser.add_argument(
        "--expected-fingerprint",
        default=os.getenv("RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256", ""),
    )
    args = parser.parse_args(argv)
    try:
        fingerprint = verify_bundle(args.root, args.expected_fingerprint)
    except ModelBundleError as exc:
        print(f"Bundle de modelos inválido: {exc}", file=sys.stderr)
        return 1
    print(f"Bundle InsightFace verificado: {fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
