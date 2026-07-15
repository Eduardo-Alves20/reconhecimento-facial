from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} deve ser true ou false")


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = PROJECT_DIR / "data" / "rag_audit.db"
    evidence_dir: Path = PROJECT_DIR / "data" / "private" / "evidence"
    camera_api_key: str = "dev-camera-key"
    admin_username: str = "admin"
    admin_password: str = "change-me"
    local_timezone: str = "America/Sao_Paulo"
    alert_webhook_url: str | None = None
    alert_channel: str = "generic-webhook"
    alert_include_personal_data: bool = False
    public_base_url: str | None = None
    alert_timeout_seconds: float = 2.0
    alert_poll_seconds: float = 5.0
    alert_max_attempts: int = 3
    report_event_limit: int = 5_000
    policy_version: str = "2026.1"
    environment: str = "development"
    seed_demo_data: bool = True
    enforce_event_freshness: bool = False
    event_max_age_seconds: int = 300
    event_future_skew_seconds: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        webhook = os.getenv("RAG_AUDIT_ALERT_WEBHOOK_URL", "").strip()
        public_base_url = os.getenv("RAG_AUDIT_PUBLIC_BASE_URL", "").strip()
        environment = os.getenv("RAG_AUDIT_ENV", "development").strip().lower()
        production = environment == "production"
        return cls(
            database_path=Path(
                os.getenv("RAG_AUDIT_DB_PATH", str(PROJECT_DIR / "data" / "rag_audit.db"))
            ),
            evidence_dir=Path(
                os.getenv(
                    "RAG_AUDIT_VISION_EVIDENCE_DIR",
                    str(PROJECT_DIR / "data" / "private" / "evidence"),
                )
            ),
            camera_api_key=os.getenv("RAG_AUDIT_CAMERA_API_KEY", "dev-camera-key"),
            admin_username=os.getenv("RAG_AUDIT_ADMIN_USERNAME", "admin"),
            admin_password=os.getenv("RAG_AUDIT_ADMIN_PASSWORD", "change-me"),
            local_timezone=os.getenv("RAG_AUDIT_TIMEZONE", "America/Sao_Paulo"),
            alert_webhook_url=webhook or None,
            alert_channel=os.getenv("RAG_AUDIT_ALERT_CHANNEL", "generic-webhook"),
            alert_include_personal_data=_environment_bool(
                "RAG_AUDIT_ALERT_INCLUDE_PERSONAL_DATA", False
            ),
            public_base_url=public_base_url.rstrip("/") or None,
            alert_timeout_seconds=float(os.getenv("RAG_AUDIT_ALERT_TIMEOUT_SECONDS", "2")),
            alert_poll_seconds=float(os.getenv("RAG_AUDIT_ALERT_POLL_SECONDS", "5")),
            alert_max_attempts=int(os.getenv("RAG_AUDIT_ALERT_MAX_ATTEMPTS", "3")),
            report_event_limit=int(os.getenv("RAG_AUDIT_REPORT_LIMIT", "5000")),
            policy_version=os.getenv("RAG_AUDIT_POLICY_VERSION", "2026.1"),
            environment=environment,
            seed_demo_data=_environment_bool("RAG_AUDIT_SEED_DEMO_DATA", not production),
            enforce_event_freshness=_environment_bool(
                "RAG_AUDIT_ENFORCE_EVENT_FRESHNESS", production
            ),
            event_max_age_seconds=int(
                os.getenv("RAG_AUDIT_EVENT_MAX_AGE_SECONDS", "300")
            ),
            event_future_skew_seconds=int(
                os.getenv("RAG_AUDIT_EVENT_FUTURE_SKEW_SECONDS", "60")
            ),
        )

    def validate(self) -> None:
        normalized_environment = self.environment.strip().lower()
        if normalized_environment not in {"development", "test", "production"}:
            raise RuntimeError("RAG_AUDIT_ENV deve ser development, test ou production.")
        try:
            ZoneInfo(self.local_timezone)
        except ZoneInfoNotFoundError as exc:
            raise RuntimeError(f"Fuso IANA inválido: {self.local_timezone}") from exc
        numeric_values = {
            "alert_timeout_seconds": self.alert_timeout_seconds,
            "alert_poll_seconds": self.alert_poll_seconds,
            "alert_max_attempts": self.alert_max_attempts,
            "report_event_limit": self.report_event_limit,
            "event_max_age_seconds": self.event_max_age_seconds,
            "event_future_skew_seconds": self.event_future_skew_seconds,
        }
        invalid = [
            name
            for name, value in numeric_values.items()
            if not math.isfinite(float(value)) or value <= 0
        ]
        if invalid:
            raise RuntimeError(f"Configurações devem ser positivas: {', '.join(invalid)}")
        if not self.policy_version.strip():
            raise RuntimeError("RAG_AUDIT_POLICY_VERSION não pode ficar vazio.")
        if normalized_environment == "production":
            weak_values = {"dev-camera-key", "change-me", ""}
            if self.camera_api_key in weak_values or self.admin_password in weak_values:
                raise RuntimeError(
                    "Credenciais de desenvolvimento não podem ser usadas em produção."
                )
            if len(self.camera_api_key) < 32 or len(self.admin_password) < 12:
                raise RuntimeError("Use segredos fortes para câmera e administração em produção.")
            if self.seed_demo_data:
                raise RuntimeError("Dados de demonstração não podem ser inseridos em produção.")
            if not self.enforce_event_freshness:
                raise RuntimeError("Validação de frescor deve estar ativa em produção.")
            if self.alert_webhook_url and urlparse(self.alert_webhook_url).scheme != "https":
                raise RuntimeError("O webhook de alertas deve usar HTTPS em produção.")
            if self.public_base_url and urlparse(self.public_base_url).scheme != "https":
                raise RuntimeError("A URL pública deve usar HTTPS em produção.")
