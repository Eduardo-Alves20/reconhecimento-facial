from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import cge_environment_api
from cge_environment_api.api.authentication_api import AuthenticationApi
from cge_environment_api.exceptions import ApiException
from cge_environment_api.models.authenticate_user_request import AuthenticateUserRequest

from .config import Settings

logger = logging.getLogger("rag_audit.auth_ad")


@dataclass(slots=True)
class LoginResult:
    ok: bool
    token: str | None = None
    groups: list[str] = field(default_factory=list)
    username: str | None = None
    display_name: str | None = None
    message: str = ""
    error_type: str | None = None
    http_status: int | None = None


@dataclass(slots=True)
class SessionResult:
    ok: bool
    username: str | None = None
    groups: list[str] = field(default_factory=list)


def _client(settings: Settings) -> cge_environment_api.ApiClient:
    configuration = cge_environment_api.Configuration(host=settings.ad_api_url)
    configuration.api_key["AuthorizationApi"] = settings.ad_api_token
    configuration.verify_ssl = settings.ad_verify_ssl
    if settings.ad_ca_cert:
        configuration.ssl_ca_cert = settings.ad_ca_cert
    if not settings.ad_verify_ssl:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return cge_environment_api.ApiClient(configuration)


def _error_body(exc: ApiException) -> dict:
    try:
        return json.loads(exc.body) if exc.body else {}
    except (ValueError, TypeError):
        return {}


def authenticate(
    settings: Settings,
    username: str,
    password: str,
    *,
    origin: str | None = None,
) -> LoginResult:
    request = AuthenticateUserRequest(
        username=username, password=password, origin=origin
    )
    try:
        with _client(settings) as api_client:
            resp = AuthenticationApi(api_client).authenticate_user(
                authenticate_user_request=request, x_real_ip=origin
            )
    except ApiException as exc:
        body = _error_body(exc)
        return LoginResult(
            ok=False,
            message=body.get("message") or "Usuário ou senha inválidos.",
            error_type=body.get("errorType"),
            http_status=exc.status,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Erro inesperado no login AD")
        return LoginResult(
            ok=False,
            message="Serviço de autenticação temporariamente indisponível.",
            http_status=503,
        )
    return LoginResult(
        ok=bool(resp.authentication),
        token=resp.token,
        groups=list(resp.groups or []),
        username=username,
        display_name=getattr(resp, "name", None),
        message=resp.message or "",
    )


def verify_session(
    settings: Settings, token: str, *, origin: str | None = None
) -> SessionResult:
    try:
        with _client(settings) as api_client:
            resp = AuthenticationApi(api_client).verify_session(
                authorizationldap=token, x_real_ip=origin
            )
    except ApiException:
        return SessionResult(ok=False)
    except Exception:  # noqa: BLE001
        logger.exception("Erro inesperado ao verificar sessão AD")
        return SessionResult(ok=False)
    user = resp.user
    return SessionResult(
        ok=bool(resp.authenticated),
        username=getattr(user, "username", None) if user else None,
        groups=list(getattr(user, "groups", []) or []) if user else [],
    )


def logoff(settings: Settings, token: str) -> None:
    try:
        with _client(settings) as api_client:
            AuthenticationApi(api_client).logoff(authorizationldap=token)
    except Exception:  # noqa: BLE001
        logger.debug("logoff AD falhou")


def is_authorized(groups: list[str], allowed_groups: tuple[str, ...]) -> bool:
    if not allowed_groups:
        return True
    present = {g.strip().upper() for g in groups}
    return any(allowed.strip().upper() in present for allowed in allowed_groups)
