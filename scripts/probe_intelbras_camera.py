"""Testa a conectividade da Intelbras VIPC 1230 G2 sem exibir credenciais."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# Permite executar diretamente: ``python scripts/probe_intelbras_camera.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env.vision")
load_dotenv(PROJECT_ROOT / ".env")

from app.vision.camera import (  # noqa: E402
    CameraConfig,
    CameraConfigurationError,
    probe_capture,
    probe_tcp,
)


def _integer_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise CameraConfigurationError(f"A variável {name} deve ser um número inteiro.") from exc


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise CameraConfigurationError(f"A variável {name} deve ser um número.") from exc


def _flag_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "sim", "yes", "on"}:
        return True
    if normalized in {"0", "false", "não", "nao", "no", "off"}:
        return False
    raise CameraConfigurationError(f"A variável {name} deve ser 1/0, true/false ou sim/não.")


def config_from_environment() -> CameraConfig:
    """Lê todas as credenciais exclusivamente do ambiente do processo."""

    required_names = (
        "INTELBRAS_CAMERA_HOST",
        "INTELBRAS_CAMERA_USER",
        "INTELBRAS_CAMERA_PASSWORD",
    )
    missing = [name for name in required_names if not os.getenv(name)]
    if missing:
        raise CameraConfigurationError(
            "Defina as variáveis obrigatórias: " + ", ".join(missing) + "."
        )

    return CameraConfig(
        host=os.environ["INTELBRAS_CAMERA_HOST"],
        username=os.environ["INTELBRAS_CAMERA_USER"],
        password=os.environ["INTELBRAS_CAMERA_PASSWORD"],
        port=_integer_env("INTELBRAS_CAMERA_RTSP_PORT", 554),
        channel=_integer_env("INTELBRAS_CAMERA_CHANNEL", 1),
        subtype=_integer_env("INTELBRAS_CAMERA_SUBTYPE", 0),
        timeout_seconds=_float_env("INTELBRAS_CAMERA_TIMEOUT_SECONDS", 3.0),
    )


def main() -> int:
    try:
        config = config_from_environment()
        capture_enabled = _flag_env("INTELBRAS_CAMERA_CAPTURE_FRAME", False)
    except CameraConfigurationError as exc:
        print(f"Configuração inválida: {exc}", file=sys.stderr)
        return 2

    print(f"Câmera configurada: {config.display_url}")
    print("Verificando a porta RTSP (este passo ainda não valida a senha)...")
    tcp_result = probe_tcp(config)
    print(f"{tcp_result.message} Tempo: {tcp_result.elapsed_ms} ms.")
    if not tcp_result.ok:
        return 1

    if not capture_enabled:
        print(
            "Captura desativada por padrão. Para validar credenciais e vídeo, defina "
            "INTELBRAS_CAMERA_CAPTURE_FRAME=1."
        )
        return 0

    print("Lendo somente um quadro em memória; nenhuma imagem será salva...")
    capture_result = probe_capture(config)
    print(capture_result.message)
    if not capture_result.ok:
        return 1

    metadata = capture_result.metadata
    if metadata is not None:
        print(f"Vídeo confirmado: {metadata.width}x{metadata.height}, {metadata.channels} canal(is).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
