from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
# Configuração unica em .env (API + worker + login). .env.api ainda é lido se
# existir, para compatibilidade com deploys que separam os segredos por processo.
load_dotenv(PROJECT_DIR / ".env.api")
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


def _real_token(value: str) -> str:
    token = value.strip()
    if not token or token.upper().startswith("COLE_AQUI"):
        return ""
    return token


def _camera_keys_from_env() -> tuple[tuple[str, str], ...]:
    raw = os.getenv("RAG_AUDIT_CAMERA_API_KEYS_JSON", "").strip()
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("RAG_AUDIT_CAMERA_API_KEYS_JSON deve ser JSON válido") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("RAG_AUDIT_CAMERA_API_KEYS_JSON deve ser um objeto não vazio")
    result: list[tuple[str, str]] = []
    for camera_id, key in payload.items():
        if (
            not isinstance(camera_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}", camera_id)
            or not isinstance(key, str)
            or not key
            or any(ord(character) < 32 or ord(character) == 127 for character in key)
        ):
            raise ValueError("RAG_AUDIT_CAMERA_API_KEYS_JSON contém uma entrada inválida")
        result.append((camera_id, key))
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = PROJECT_DIR / "data" / "rag_audit.db"
    evidence_dir: Path = PROJECT_DIR / "data" / "private" / "evidence"
    camera_api_key: str = field(default="dev-camera-key", repr=False)
    camera_api_keys: tuple[tuple[str, str], ...] = field(
        default=(),
        repr=False,
    )
    admin_username: str = "admin"
    admin_password: str = field(default="change-me", repr=False)
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
    queued_event_max_age_seconds: int = 604_800
    evidence_ttl_days: int = 30
    evidence_max_bytes: int = 10 * 1024 * 1024 * 1024
    evidence_max_item_bytes: int = 25 * 1024 * 1024
    evidence_evict_oldest: bool = True
    ad_api_url: str | None = None
    ad_api_token: str = field(default="", repr=False)
    ad_allowed_groups: tuple[str, ...] = ()
    session_cookie_name: str = "rag_audit_sso"
    allow_basic_fallback: bool = True
    ad_verify_ssl: bool = True
    ad_ca_cert: str | None = None

    @property
    def ad_login_enabled(self) -> bool:
        return bool(self.ad_api_url and self.ad_api_token)

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
            camera_api_keys=_camera_keys_from_env(),
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
            queued_event_max_age_seconds=int(
                os.getenv("RAG_AUDIT_QUEUED_EVENT_MAX_AGE_SECONDS", "604800")
            ),
            evidence_ttl_days=int(
                os.getenv("RAG_AUDIT_EVIDENCE_TTL_DAYS", "30")
            ),
            evidence_max_bytes=round(
                float(os.getenv("RAG_AUDIT_EVIDENCE_MAX_GB", "10"))
                * 1024
                * 1024
                * 1024
            ),
            evidence_max_item_bytes=round(
                float(os.getenv("RAG_AUDIT_EVIDENCE_MAX_ITEM_MB", "25"))
                * 1024
                * 1024
            ),
            evidence_evict_oldest=_environment_bool(
                "RAG_AUDIT_EVIDENCE_EVICT_OLDEST",
                True,
            ),
            ad_api_url=(os.getenv("CGE_ENV_API_URL", "").strip().rstrip("/") or None),
            ad_api_token=_real_token(os.getenv("CGE_ENV_API_TOKEN", "")),
            ad_allowed_groups=tuple(
                g.strip()
                for g in os.getenv("RAG_AUDIT_ALLOWED_AD_GROUPS", "").split(",")
                if g.strip()
            ),
            session_cookie_name=os.getenv(
                "RAG_AUDIT_SESSION_COOKIE", "rag_audit_sso"
            ).strip()
            or "rag_audit_sso",
            allow_basic_fallback=_environment_bool(
                "RAG_AUDIT_ALLOW_BASIC_FALLBACK", not production
            ),
            ad_verify_ssl=_environment_bool("CGE_ENV_API_VERIFY_SSL", True),
            ad_ca_cert=(os.getenv("CGE_ENV_API_CA_CERT", "").strip() or None),
        )

    def camera_key_for(self, camera_id: str) -> str | None:
        if self.camera_api_keys:
            return dict(self.camera_api_keys).get(camera_id)
        return self.camera_api_key

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
            "queued_event_max_age_seconds": self.queued_event_max_age_seconds,
            "evidence_ttl_days": self.evidence_ttl_days,
            "evidence_max_bytes": self.evidence_max_bytes,
            "evidence_max_item_bytes": self.evidence_max_item_bytes,
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
        if not self.camera_api_keys and not self.camera_api_key:
            raise RuntimeError("A credencial da câmera não pode ficar vazia.")
        if self.queued_event_max_age_seconds < self.event_max_age_seconds:
            raise RuntimeError(
                "A janela de eventos enfileirados não pode ser menor que a janela normal."
            )
        if self.evidence_max_item_bytes > self.evidence_max_bytes:
            raise RuntimeError(
                "O limite por evidência não pode superar a cota total."
            )
        if normalized_environment == "production":
            weak_values = {"dev-camera-key", "change-me", ""}
            if self.admin_password in weak_values:
                raise RuntimeError(
                    "Credenciais de desenvolvimento não podem ser usadas em produção."
                )
            if not self.camera_api_keys:
                raise RuntimeError(
                    "Produção exige RAG_AUDIT_CAMERA_API_KEYS_JSON por câmera."
                )
            if any(len(key) < 32 for _, key in self.camera_api_keys):
                raise RuntimeError("Use chaves de câmera com pelo menos 32 caracteres.")
            camera_keys = [key for _, key in self.camera_api_keys]
            if len(camera_keys) != len(set(camera_keys)):
                raise RuntimeError("Cada câmera deve usar uma chave exclusiva.")
            if len(self.admin_password) < 12:
                raise RuntimeError("Use uma senha administrativa forte em produção.")
            if self.seed_demo_data:
                raise RuntimeError("Dados de demonstração não podem ser inseridos em produção.")
            if not self.enforce_event_freshness:
                raise RuntimeError("Validação de frescor deve estar ativa em produção.")
            if self.alert_webhook_url and urlparse(self.alert_webhook_url).scheme != "https":
                raise RuntimeError("O webhook de alertas deve usar HTTPS em produção.")
            if self.public_base_url and urlparse(self.public_base_url).scheme != "https":
                raise RuntimeError("A URL pública deve usar HTTPS em produção.")
