from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.vision.camera import (
    CameraConfig,
    CameraCaptureError,
    CameraConfigurationError,
    FrameMetadata,
    build_rtsp_url,
    probe_capture,
    probe_tcp,
    redact_rtsp_url,
)


def camera_config(**overrides: object) -> CameraConfig:
    values: dict[str, object] = {
        "host": "192.168.10.30",
        "username": "rag leitor",
        "password": "segredo@forte:/?#",
    }
    values.update(overrides)
    return CameraConfig(**values)  # type: ignore[arg-type]


def test_builds_intelbras_rtsp_url_with_escaped_credentials() -> None:
    url = build_rtsp_url(camera_config())

    assert url == (
        "rtsp://rag%20leitor:segredo%40forte%3A%2F%3F%23@192.168.10.30:554/"
        "cam/realmonitor?channel=1&subtype=0"
    )


def test_redaction_never_exposes_username_or_password() -> None:
    config = camera_config()
    redacted = redact_rtsp_url(build_rtsp_url(config))

    assert redacted == (
        "rtsp://***:***@192.168.10.30:554/cam/realmonitor?channel=1&subtype=0"
    )
    assert config.username not in redacted
    assert config.password not in redacted
    assert "segredo%40forte" not in redacted
    assert config.password not in repr(config)


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"host": "rtsp://192.168.1.2"}, "apenas o IP"),
        ({"port": 70_000}, "porta RTSP"),
        ({"subtype": 4}, "subtipo RTSP"),
        ({"password": ""}, "senha"),
    ],
)
def test_rejects_invalid_camera_configuration(
    overrides: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(CameraConfigurationError, match=expected_message):
        camera_config(**overrides)


@dataclass
class FakeConnection:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_tcp_probe_uses_injected_connector_and_closes_connection() -> None:
    connection = FakeConnection()
    calls: list[tuple[tuple[str, int], float]] = []

    def connector(address: tuple[str, int], timeout: float) -> FakeConnection:
        calls.append((address, timeout))
        return connection

    result = probe_tcp(camera_config(timeout_seconds=1.5), connector=connector)

    assert result.ok is True
    assert calls == [(('192.168.10.30', 554), 1.5)]
    assert connection.closed is True
    assert "senha" not in result.message.lower()


def test_tcp_probe_returns_safe_message_when_connector_leaks_url() -> None:
    secret = "uma-senha-que-nao-pode-vazar"

    def connector(address: tuple[str, int], timeout: float) -> object:
        del address, timeout
        raise RuntimeError(f"rtsp://admin:{secret}@camera.local")

    result = probe_tcp(
        camera_config(username="admin", password=secret),
        connector=connector,
    )

    assert result.ok is False
    assert secret not in result.message
    assert "inesperada" in result.message


class RecordingCaptureBackend:
    def __init__(self) -> None:
        self.received_url: str | None = None

    def capture(self, rtsp_url: str, timeout_seconds: float) -> FrameMetadata:
        self.received_url = rtsp_url
        assert timeout_seconds == 3.0
        return FrameMetadata(width=1920, height=1080, channels=3)


def test_capture_backend_is_injectable_and_frame_is_not_returned() -> None:
    backend = RecordingCaptureBackend()
    result = probe_capture(camera_config(), backend=backend)

    assert result.ok is True
    assert result.metadata == FrameMetadata(width=1920, height=1080, channels=3)
    assert backend.received_url == build_rtsp_url(camera_config())
    assert not hasattr(result, "frame")
    assert "descartado" in result.message


def test_capture_error_does_not_echo_authenticated_url() -> None:
    secret = "segredo-absoluto"

    class LeakyBackend:
        def capture(self, rtsp_url: str, timeout_seconds: float) -> FrameMetadata:
            del timeout_seconds
            raise RuntimeError(f"Falha ao abrir {rtsp_url}")

    result = probe_capture(
        camera_config(username="operador", password=secret),
        backend=LeakyBackend(),
    )

    assert result.ok is False
    assert secret not in result.message
    assert "operador" not in result.message
    assert result.metadata is None


def test_expected_capture_error_is_also_redacted() -> None:
    secret = "nao-vazar-esta-senha"

    class LeakyExpectedBackend:
        def capture(self, rtsp_url: str, timeout_seconds: float) -> FrameMetadata:
            del timeout_seconds
            raise CameraCaptureError(f"Falha prevista em {rtsp_url}")

    result = probe_capture(
        camera_config(username="leitor", password=secret),
        backend=LeakyExpectedBackend(),
    )

    assert result.ok is False
    assert secret not in result.message
    assert "leitor" not in result.message
    assert "rtsp://***:***@" in result.message
