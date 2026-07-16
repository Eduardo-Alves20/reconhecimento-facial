from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


TEST_CAMERA_KEY = "camera-key-for-tests"
TEST_ADMIN_USERNAME = "auditor"
TEST_ADMIN_PASSWORD = "strong-test-password"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Use one real, file-backed SQLite database per test."""
    return Settings(
        database_path=tmp_path / "rag-audit-test.sqlite3",
        evidence_dir=tmp_path / "evidence",
        camera_api_key=TEST_CAMERA_KEY,
        admin_username=TEST_ADMIN_USERNAME,
        admin_password=TEST_ADMIN_PASSWORD,
        local_timezone="America/Sao_Paulo",
        alert_webhook_url=None,
        alert_poll_seconds=3_600,
        report_event_limit=100,
        environment="test",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # Entering the context runs the app lifespan, migrations and demo seed.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def camera_headers() -> dict[str, str]:
    return {"X-Camera-Key": TEST_CAMERA_KEY}


@pytest.fixture
def admin_auth() -> tuple[str, str]:
    return TEST_ADMIN_USERNAME, TEST_ADMIN_PASSWORD


@pytest.fixture
def event_payload() -> Callable[..., dict[str, Any]]:
    def build(
        *,
        event_id: str = "evt-lucas-001",
        user_id: str = "EMP001",
        timestamp: str = "2026-07-14T14:00:00-03:00",
        camera_id: str = "cam-ti-01",
        room_id: str = "sala_ti_01",
        door_result: str = "GRANTED",
        recognition_confidence: float = 0.99,
    ) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "camera_id": camera_id,
            "user_id": user_id,
            "room_id": room_id,
            "timestamp": timestamp,
            "door_result": door_result,
            "recognition_confidence": recognition_confidence,
        }

    return build
