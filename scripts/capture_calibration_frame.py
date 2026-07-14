"""Captura um único quadro privado para calibrar as zonas da porta."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.vision.camera import CameraConfig, build_rtsp_url  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "private" / "camera-calibration-frame.jpg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--empty-room-confirmed",
        action="store_true",
        help="confirma que a captura será feita sem pessoas no enquadramento",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.empty_room_confirmed:
        print(
            "Captura recusada: esvazie o enquadramento e use --empty-room-confirmed.",
            file=sys.stderr,
        )
        return 2
    required = {
        "host": os.getenv("INTELBRAS_CAMERA_HOST", ""),
        "username": os.getenv("INTELBRAS_CAMERA_USER", ""),
        "password": os.getenv("INTELBRAS_CAMERA_PASSWORD", ""),
    }
    if not all(required.values()):
        print("Preencha IP, usuário e senha da câmera no .env.", file=sys.stderr)
        return 2
    try:
        import cv2

        config = CameraConfig(
            **required,
            port=int(os.getenv("INTELBRAS_CAMERA_RTSP_PORT", "554")),
            channel=int(os.getenv("INTELBRAS_CAMERA_CHANNEL", "1")),
            subtype=0,
            timeout_seconds=float(os.getenv("INTELBRAS_CAMERA_TIMEOUT_SECONDS", "5")),
        )
    except Exception as exc:
        print(f"Configuração/camada de vídeo inválida: {exc}", file=sys.stderr)
        return 2

    output = DEFAULT_OUTPUT
    if output.exists() and not args.overwrite:
        print(f"O quadro já existe: {output}. Use --overwrite para substituir.", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture()
    timeout_ms = round(config.timeout_seconds * 1000)
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    try:
        if not capture.open(build_rtsp_url(config), getattr(cv2, "CAP_FFMPEG", 0)):
            print("Não foi possível abrir o RTSP; confira a configuração local.", file=sys.stderr)
            return 1
        frame = None
        for _ in range(10):
            received, candidate = capture.read()
            if received and candidate is not None:
                frame = candidate
        if frame is None:
            print("O stream abriu, mas não entregou um quadro.", file=sys.stderr)
            return 1
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            print("Não foi possível codificar o quadro.", file=sys.stderr)
            return 1
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".calibration-", suffix=".jpg", dir=output.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded.tobytes())
            os.replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    finally:
        capture.release()
    print(f"Quadro privado salvo em: {output}")
    print("Exclua-o depois de criar camera-calibration.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
