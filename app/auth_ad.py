"""Cliente do login institucional (AD) via CGE Environment API SDK.

Encapsula o pacote ``cge_environment_api``. As chamadas são síncronas (o SDK usa
urllib3), então as rotas FastAPI devem invocá-las com ``run_in_threadpool`` para
não bloquear o event loop.

Nenhuma senha é registrada em log. O token da aplicação vem de Settings e nunca
aparece em mensagens de erro devolvidas ao cliente.
"""
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
    """Login AD por usuário/senha. Devolve token JWT + grupos em caso de sucesso."""
    request = AuthenticateUserRequest(
        username=username, password=password, origin=origin
    )
    try:
        with _client(settings) as api_client:
            api = AuthenticationApi(api_client)
            resp = api.authenticate_user(
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
    except Exception:  # noqa: BLE001 - falha de rede/SDK: não vazar detalhe
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
    """Valida o JWT de sessão. Devolve username + grupos atuais do AD."""
    try:
        with _client(settings) as api_client:
            api = AuthenticationApi(api_client)
            resp = api.verify_session(authorizationldap=token, x_real_ip=origin)
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
    """Invalida a sessão SSO no servidor. Falha é silenciosa (best-effort)."""
    try:
        with _client(settings) as api_client:
            AuthenticationApi(api_client).logoff(authorizationldap=token)
    except Exception:  # noqa: BLE001
        logger.debug("logoff AD falhou; seguindo com a limpeza local do cookie")


def is_authorized(groups: list[str], allowed_groups: tuple[str, ...]) -> bool:
    """Autorizado se pertencer a pelo menos um grupo permitido.

    Sem grupos permitidos configurados, qualquer usuário autenticado passa.
    Comparação é case-insensitive pelo nome do grupo.
    """
    if not allowed_groups:
        return True
    present = {g.strip().upper() for g in groups}
    return any(allowed.strip().upper() in present for allowed in allowed_groups)
