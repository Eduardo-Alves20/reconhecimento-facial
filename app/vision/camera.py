"""Conexão segura com o stream RTSP da Intelbras VIPC 1230 G2.

Este módulo cuida somente da configuração, do teste TCP e da leitura opcional
de um quadro. Ele não persiste imagens e nunca inclui credenciais nos resultados.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol
from urllib.parse import quote, urlsplit, urlunsplit


class CameraConfigurationError(ValueError):
    """Indica configuração de câmera ausente ou inválida."""


class CameraCaptureError(RuntimeError):
    """Indica que o backend não conseguiu obter um quadro."""


def _validate_host(host: str) -> str:
    normalized = host.strip()
    if not normalized:
        raise CameraConfigurationError("O endereço IP ou nome da câmera é obrigatório.")
    if any(character.isspace() or ord(character) < 32 for character in normalized):
        raise CameraConfigurationError("O endereço da câmera contém caracteres inválidos.")
    if any(fragment in normalized for fragment in ("://", "/", "?", "#", "@")):
        raise CameraConfigurationError(
            "Informe apenas o IP ou nome da câmera, sem protocolo, porta ou caminho."
        )

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        hostname = normalized.rstrip(".")
        if len(hostname) > 253:
            raise CameraConfigurationError("O nome da câmera é longo demais.") from None
        labels = hostname.split(".")
        valid_label = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
        if not labels or any(not valid_label.fullmatch(label) for label in labels):
            raise CameraConfigurationError("O IP ou nome da câmera é inválido.") from None
        return hostname
    return address.compressed


def _validate_secret(value: str, field_name: str) -> str:
    if not value:
        raise CameraConfigurationError(f"{field_name} da câmera é obrigatório(a).")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise CameraConfigurationError(f"{field_name} da câmera contém caracteres inválidos.")
    return value


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Configuração mínima do RTSP, mantida apenas em memória."""

    host: str
    username: str
    password: str = field(repr=False)
    port: int = 554
    channel: int = 1
    subtype: int = 0
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _validate_host(self.host))
        object.__setattr__(self, "username", _validate_secret(self.username, "O usuário"))
        object.__setattr__(self, "password", _validate_secret(self.password, "A senha"))
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65_535
        ):
            raise CameraConfigurationError("A porta RTSP deve estar entre 1 e 65535.")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int) or self.channel < 1:
            raise CameraConfigurationError("O canal RTSP deve ser um inteiro maior ou igual a 1.")
        if isinstance(self.subtype, bool) or not isinstance(self.subtype, int) or self.subtype not in (0, 1):
            raise CameraConfigurationError("O subtipo RTSP deve ser 0 (principal) ou 1 (extra).")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not 0 < self.timeout_seconds <= 60
        ):
            raise CameraConfigurationError("O tempo limite deve ser maior que 0 e de até 60 segundos.")

    @property
    def display_url(self) -> str:
        """URL apropriada para tela/log, sem usuário ou senha reais."""

        return redact_rtsp_url(build_rtsp_url(self))


def _url_host(host: str) -> str:
    """Adiciona colchetes quando o host é um IPv6."""

    try:
        return f"[{host}]" if ipaddress.ip_address(host).version == 6 else host
    except ValueError:
        return host


def build_rtsp_url(config: CameraConfig) -> str:
    """Monta a URL documentada pela Intelbras, escapando as credenciais."""

    username = quote(config.username, safe="")
    password = quote(config.password, safe="")
    return (
        f"rtsp://{username}:{password}@{_url_host(config.host)}:{config.port}"
        f"/cam/realmonitor?channel={config.channel}&subtype={config.subtype}"
    )


def redact_rtsp_url(value: str) -> str:
    """Substitui o *userinfo* de uma URL RTSP antes de exibi-la.

    Em entrada malformada, aplica também uma substituição conservadora para não
    devolver acidentalmente o trecho que antecede ``@``.
    """

    # Também funciona quando a URL aparece no meio de uma mensagem de erro.
    conservatively_redacted = re.sub(
        r"(?i)(rtsp://)[^/\s]+@",
        r"\1***:***@",
        value,
    )
    if conservatively_redacted != value:
        return conservatively_redacted

    try:
        parts = urlsplit(value)
        if parts.scheme.lower() != "rtsp" or "@" not in parts.netloc:
            return value
        _, separator, destination = parts.netloc.rpartition("@")
        if not separator:
            return value
        return urlunsplit((parts.scheme, f"***:***@{destination}", parts.path, parts.query, parts.fragment))
    except ValueError:
        return re.sub(r"(?i)(rtsp://)[^/\s]+@", r"\1***:***@", value)


@dataclass(frozen=True, slots=True)
class TcpProbeResult:
    ok: bool
    host: str
    port: int
    elapsed_ms: int
    message: str


SocketConnector = Callable[[tuple[str, int], float], object]


def probe_tcp(
    config: CameraConfig,
    *,
    connector: SocketConnector = socket.create_connection,
) -> TcpProbeResult:
    """Verifica se a porta RTSP aceita uma conexão TCP.

    O ``connector`` é injetável para testes. Este teste não autentica no RTSP.
    """

    started_at = perf_counter()
    connection: object | None = None
    try:
        connection = connector((config.host, config.port), config.timeout_seconds)
    except TimeoutError:
        message = "Tempo esgotado ao tentar alcançar a porta RTSP da câmera."
        ok = False
    except ConnectionRefusedError:
        message = "A câmera respondeu, mas recusou a conexão na porta RTSP."
        ok = False
    except OSError:
        message = "Não foi possível alcançar a porta RTSP da câmera pela rede."
        ok = False
    except Exception:  # protege a saída contra mensagens de backend contendo segredos
        message = "O teste TCP falhou de forma inesperada e segura."
        ok = False
    else:
        message = "A porta RTSP está acessível pela rede."
        ok = True
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            close()

    elapsed_ms = max(0, round((perf_counter() - started_at) * 1_000))
    return TcpProbeResult(ok, config.host, config.port, elapsed_ms, message)


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Somente metadados do quadro lido; os pixels não saem do backend."""

    width: int
    height: int
    channels: int


class CaptureBackend(Protocol):
    def capture(self, rtsp_url: str, timeout_seconds: float) -> FrameMetadata:
        """Lê um quadro em memória e devolve apenas seus metadados."""


class OpenCVCaptureBackend:
    """Backend opcional. O OpenCV só é importado quando a captura é pedida."""

    def capture(self, rtsp_url: str, timeout_seconds: float) -> FrameMetadata:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CameraCaptureError(
                "OpenCV não está instalado; o teste TCP ainda pode ser usado normalmente."
            ) from exc

        capture = cv2.VideoCapture()
        timeout_ms = round(timeout_seconds * 1_000)
        try:
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)

            api = getattr(cv2, "CAP_FFMPEG", getattr(cv2, "CAP_ANY", 0))
            if not capture.open(rtsp_url, api):
                raise CameraCaptureError(
                    "Não foi possível abrir o stream; confira usuário, senha e RTSP."
                )
            received, frame = capture.read()
            if not received or frame is None:
                raise CameraCaptureError("O stream abriu, mas não entregou um quadro válido.")

            shape = getattr(frame, "shape", ())
            if len(shape) < 2:
                raise CameraCaptureError("O quadro recebido possui formato inválido.")
            height, width = int(shape[0]), int(shape[1])
            channels = int(shape[2]) if len(shape) >= 3 else 1
            return FrameMetadata(width=width, height=height, channels=channels)
        finally:
            capture.release()


@dataclass(frozen=True, slots=True)
class CaptureProbeResult:
    ok: bool
    message: str
    metadata: FrameMetadata | None = None


def probe_capture(
    config: CameraConfig,
    *,
    backend: CaptureBackend | None = None,
) -> CaptureProbeResult:
    """Tenta ler um quadro sem salvá-lo e sem expor a URL autenticada."""

    selected_backend = backend or OpenCVCaptureBackend()
    try:
        metadata = selected_backend.capture(build_rtsp_url(config), config.timeout_seconds)
        if metadata.width < 1 or metadata.height < 1 or metadata.channels < 1:
            raise CameraCaptureError("O backend retornou dimensões inválidas.")
    except CameraCaptureError as exc:
        return CaptureProbeResult(ok=False, message=redact_rtsp_url(str(exc)))
    except Exception:
        # Não propaga str(exc): bibliotecas podem incluir a URL com senha na exceção.
        return CaptureProbeResult(
            ok=False,
            message="A captura falhou de forma inesperada; nenhuma credencial foi exibida.",
        )
    return CaptureProbeResult(
        ok=True,
        message="Um quadro foi lido em memória e descartado sem ser salvo.",
        metadata=metadata,
    )
