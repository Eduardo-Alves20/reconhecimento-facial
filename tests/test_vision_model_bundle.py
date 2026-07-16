from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_vision_models import prepare_bundle
from scripts.verify_vision_models import (
    MANIFEST_NAME,
    ModelBundleError,
    verify_bundle,
)


def model_bundle(tmp_path: Path) -> Path:
    model_dir = tmp_path / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"detector")
    (model_dir / "w600k_r50.onnx").write_bytes(b"recognizer")
    return tmp_path


def test_prepared_bundle_is_verified(tmp_path: Path) -> None:
    root = model_bundle(tmp_path)

    fingerprint = prepare_bundle(
        root,
        license_reference="Contrato interno FACIAL-2026-01",
    )

    assert verify_bundle(root, fingerprint) == fingerprint


def test_modified_model_is_rejected(tmp_path: Path) -> None:
    root = model_bundle(tmp_path)
    fingerprint = prepare_bundle(root, license_reference="Licença de teste")
    (root / "models" / "buffalo_l" / "w600k_r50.onnx").write_bytes(b"changed")

    with pytest.raises(ModelBundleError, match="Tamanho divergente|SHA-256 divergente"):
        verify_bundle(root, fingerprint)


def test_external_fingerprint_is_required(tmp_path: Path) -> None:
    root = model_bundle(tmp_path)
    prepare_bundle(root, license_reference="Licença de teste")

    with pytest.raises(ModelBundleError, match="fingerprint autorizado"):
        verify_bundle(root, "0" * 64)


def test_unlisted_onnx_is_rejected(tmp_path: Path) -> None:
    root = model_bundle(tmp_path)
    fingerprint = prepare_bundle(root, license_reference="Licença de teste")
    (root / "models" / "buffalo_l" / "other.onnx").write_bytes(b"other")

    with pytest.raises(ModelBundleError, match="não declarados"):
        verify_bundle(root, fingerprint)


def test_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    root = model_bundle(tmp_path)
    fingerprint = prepare_bundle(root, license_reference="Licença de teste")
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["license"]["reference"] = "Outra licença"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelBundleError, match="alterado"):
        verify_bundle(root, fingerprint)


def test_missing_required_model_is_rejected(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"detector")

    with pytest.raises(ModelBundleError, match="w600k_r50.onnx"):
        prepare_bundle(tmp_path, license_reference="Licença de teste")
