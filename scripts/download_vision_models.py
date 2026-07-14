"""Baixa modelos oficiais do OpenCV Zoo com SHA-256 fixado."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "vision-models"


@dataclass(frozen=True, slots=True)
class ModelFile:
    filename: str
    url: str
    sha256: str


MODELS = (
    ModelFile(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    ModelFile(
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(output_dir: Path = DEFAULT_OUTPUT) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for model in MODELS:
        destination = output_dir / model.filename
        if destination.exists() and _sha256(destination) == model.sha256:
            print(f"OK: {model.filename} já existe e foi verificado.")
            downloaded.append(destination)
            continue

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{model.filename}.", suffix=".tmp", dir=output_dir, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                request = urllib.request.Request(
                    model.url,
                    headers={"User-Agent": "RAG-Audit-model-installer/1.0"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        temporary.write(chunk)

            actual_hash = _sha256(temporary_path)
            if actual_hash != model.sha256:
                raise RuntimeError(
                    f"SHA-256 inválido para {model.filename}; download descartado."
                )
            os.replace(temporary_path, destination)
            temporary_path = None
            print(f"Baixado e verificado: {model.filename}")
            downloaded.append(destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return downloaded


def main() -> int:
    output = Path(os.getenv("RAG_AUDIT_VISION_MODELS_DIR", str(DEFAULT_OUTPUT)))
    try:
        paths = download_models(output)
    except Exception as exc:
        print(f"Falha ao instalar modelos: {exc}", file=sys.stderr)
        return 1
    print(f"{len(paths)} modelos prontos em {output.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
