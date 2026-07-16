from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values

from app.config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_environment_examples_are_separated() -> None:
    api = dotenv_values(PROJECT_ROOT / ".env.api.example")
    vision = dotenv_values(PROJECT_ROOT / ".env.vision.example")

    assert api["RAG_AUDIT_DB_PATH"] == "data/api/rag_audit.db"
    assert api["RAG_AUDIT_CAMERA_API_KEYS_JSON"]
    assert "INTELBRAS_CAMERA_PASSWORD" not in api
    assert "RAG_AUDIT_ADMIN_PASSWORD" not in vision
    assert "RAG_AUDIT_ALERT_WEBHOOK_URL" not in vision
    assert vision["RAG_AUDIT_VISION_OUTBOX_PATH"] != vision["RAG_AUDIT_VISION_DRY_RUN_OUTBOX_PATH"]
    assert "{camera_id}" in vision["RAG_AUDIT_VISION_OUTBOX_PATH"]
    assert "{camera_id}" in vision["RAG_AUDIT_VISION_DRY_RUN_OUTBOX_PATH"]
    assert "{camera_id}" in vision["RAG_AUDIT_GALLERY_CACHE_PATH"]
    for name in (
        "RAG_AUDIT_EVIDENCE_TTL_DAYS",
        "RAG_AUDIT_EVIDENCE_MAX_GB",
        "RAG_AUDIT_EVIDENCE_MAX_ITEM_MB",
        "RAG_AUDIT_EVIDENCE_EVICT_OLDEST",
    ):
        assert api[name] == vision[name]


def test_api_example_passes_production_configuration_validation() -> None:
    values = {
        key: value
        for key, value in dotenv_values(PROJECT_ROOT / ".env.api.example").items()
        if value is not None
    }

    with patch.dict(os.environ, values, clear=True):
        Settings.from_env().validate()


def test_compose_keeps_service_env_and_mounts_separate() -> None:
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "- .env.api" in compose
    assert "- .env.vision" in compose
    assert "RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256:" not in compose
    assert "./data/private/gallery:/app/data/private/gallery:ro" in compose
    assert "./data/private/cache:/app/data/private/cache" in compose
    assert "./data/private/config:/app/data/private/config:ro" in compose
    assert "./data/private/outbox:/app/data/private/outbox" in compose
    assert "./data/private/learned:/app/data/private/learned" in compose


def test_local_secrets_are_excluded_from_git_and_docker_context() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert ".env.*" in gitignore
    assert "!.env.api.example" in gitignore
    assert "!.env.vision.example" in gitignore
    assert ".env*" in dockerignore
