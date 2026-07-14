"""Executa ou valida o worker de visão da VIPC 1230 G2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.vision.camera import probe_capture, probe_tcp  # noqa: E402
from app.vision.worker import (  # noqa: E402
    VisionConfigurationError,
    VisionWorkerSettings,
    create_processor,
    run_forever,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida câmera, modelos, galeria e calibração sem iniciar o monitoramento",
    )
    args = parser.parse_args()
    try:
        settings = VisionWorkerSettings.from_env(PROJECT_ROOT)
    except Exception as exc:
        print(f"Configuração incompleta: {exc}", file=sys.stderr)
        return 2

    if args.check:
        tcp = probe_tcp(settings.camera)
        print(tcp.message)
        if not tcp.ok:
            return 1
        capture = probe_capture(settings.camera)
        print(capture.message)
        if not capture.ok:
            return 1
        try:
            processor = create_processor(settings)
        except Exception as exc:
            print(f"Visão ainda não está pronta: {exc}", file=sys.stderr)
            return 1
        print(
            f"Configuração validada; {len(processor.engine.gallery)} pessoa(s) "
            "carregada(s) na galeria."
        )
        return 0

    try:
        run_forever(settings)
    except KeyboardInterrupt:
        print("Worker encerrado pelo operador.")
        return 0
    except VisionConfigurationError as exc:
        print(f"Configuração inválida: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Worker interrompido com segurança: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
