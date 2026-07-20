from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import AsyncIterator
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import auth_ad
from .alerts import AlertWorker
from .config import PROJECT_DIR, Settings
from .database import Repository, utc_now
from .models import AccessEventIn, Decision, IdentityStatus, RiskLevel
from .reports import build_consolidated_pdf, build_event_pdf
from .risk_engine import evaluate_access
from .vision.evidence import (
    EvidenceIntegrityError,
    EvidenceNotFoundError,
    EvidenceStore,
)


templates = Jinja2Templates(directory=str(PROJECT_DIR / "app" / "templates"))
http_basic = HTTPBasic(auto_error=False)


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict]] = set()

    async def publish(self, message: dict) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(message)

    async def stream(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=50)
        self._subscribers.add(queue)
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"event: access-event\ndata: {json.dumps(message, ensure_ascii=False)}\n\n"
                except TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            self._subscribers.discard(queue)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _repository(request: Request) -> Repository:
    return request.app.state.repository


def require_camera_key(
    request: Request,
    camera_id: str,
    supplied_key: str | None,
) -> None:
    configured = _settings(request).camera_key_for(camera_id)
    if configured is None or supplied_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credencial da câmera inválida.",
        )
    expected = (configured or "").encode("utf-8")
    supplied = supplied_key.encode("utf-8")
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credencial da câmera inválida.",
        )


class LoginRequired(Exception):
    """Sinaliza que a rota exige login institucional (redireciona para /login)."""

    def __init__(self, next_url: str = "/dashboard") -> None:
        self.next_url = next_url


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def _current_session(request: Request) -> tuple[str, list[str]] | None:
    """(username, grupos) se o cookie de sessão for válido E autorizado; senão None."""
    settings = _settings(request)
    if not settings.ad_login_enabled:
        return None
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    result = auth_ad.verify_session(settings, token, origin=_client_ip(request))
    if not result.ok:
        return None
    if not auth_ad.is_authorized(result.groups, settings.ad_allowed_groups):
        return None
    return result.username or "?", result.groups


def _basic_admin(request: Request, credentials: HTTPBasicCredentials | None) -> str:
    settings = _settings(request)
    username = credentials.username if credentials else ""
    password = credentials.password if credentials else ""
    valid_username = hmac.compare_digest(
        username.encode("utf-8"), settings.admin_username.encode("utf-8")
    )
    valid_password = hmac.compare_digest(
        password.encode("utf-8"), settings.admin_password.encode("utf-8")
    )
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticação administrativa necessária.",
            headers={"WWW-Authenticate": 'Basic realm="RAG-Audit"'},
        )
    return username


def require_admin(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(http_basic),
) -> str:
    settings = _settings(request)
    # Com login AD configurado, a sessão institucional é a única porta.
    if settings.ad_login_enabled:
        session = _current_session(request)
        if session is not None:
            return session[0]
        raise LoginRequired(next_url=request.url.path)
    # Sem AD configurado (ex.: token ainda não cadastrado): HTTP Basic em dev.
    if settings.allow_basic_fallback:
        return _basic_admin(request, credentials)
    raise LoginRequired(next_url=request.url.path)


def _safe_next(target: str | None) -> str:
    """Só aceita caminhos locais, para evitar open redirect."""
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/dashboard"
    return target


def _set_session_cookie(
    response: Response, request: Request, settings: Settings, token: str
) -> None:
    secure = request.url.scheme == "https" or settings.environment == "production"
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _govbr_login_url(settings: Settings, next_url: str) -> str | None:
    """URL do login gov.br. Stub: exige rota de callback própria, incremento futuro.

    Retornar None mantém o botão "Entrar com gov.br" oculto no template até a
    integração do callback estar pronta, evitando um botão que não funciona.
    """
    return None


def _friendly_login_error(result: auth_ad.LoginResult) -> str:
    if result.error_type == "PASSWORD_CHANGE_REQUIRED":
        return "Sua senha expirou. Troque a senha de rede no AD e tente de novo."
    if result.http_status == 503:
        return "Serviço de autenticação indisponível. Tente novamente em instantes."
    return result.message or "Usuário ou senha inválidos."


def _canonical_payload(event: AccessEventIn) -> tuple[dict, str]:
    event_data = event.model_dump(mode="json")
    canonical = dict(event_data)
    canonical["timestamp"] = event.timestamp.astimezone(UTC).isoformat()
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return event_data, hashlib.sha256(encoded).hexdigest()


def _filters(
    *,
    from_timestamp: datetime | None,
    to_timestamp: datetime | None,
    room_id: str | None,
    user_id: str | None,
    decision: Decision | None,
    risk_level: RiskLevel | None,
    alert_status: str | None,
    q: str | None,
) -> dict:
    for value in (from_timestamp, to_timestamp):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise HTTPException(
                status_code=422,
                detail="Datas de filtro devem incluir fuso/offset.",
            )
        if value is not None:
            try:
                if not 2000 <= value.year <= 2100:
                    raise ValueError
                value.astimezone(UTC)
            except (OverflowError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Data de filtro fora da faixa operacional (2000–2100).",
                ) from exc
    if from_timestamp and to_timestamp and from_timestamp > to_timestamp:
        raise HTTPException(status_code=422, detail="O início do período deve preceder o fim.")
    return {
        "from_timestamp": from_timestamp,
        "to_timestamp": to_timestamp,
        "room_id": room_id,
        "user_id": user_id,
        "decision": decision.value if decision else None,
        "risk_level": risk_level.value if risk_level else None,
        "alert_status": alert_status,
        "q": q,
    }


def _summary(event: dict) -> dict:
    excluded = {"raw_payload", "context_snapshot", "source_ids"}
    summary = {key: value for key, value in event.items() if key not in excluded}
    payload = event.get("raw_payload")
    summary["has_photo"] = bool(
        isinstance(payload, dict) and payload.get("evidence_ref")
    )
    return summary


def _webhook_receipt(event: dict, *, idempotent_replay: bool) -> dict:
    alert = event.get("alert")
    safe_alert = (
        {"alert_id": alert["alert_id"], "status": alert["status"]} if alert else None
    )
    return {
        "idempotent_replay": idempotent_replay,
        "event": {
            "event_id": event["event_id"],
            "decision": event["decision"],
            "risk_level": event["risk_level"],
            "risk_score": event["risk_score"],
            "reason_codes": event["reason_codes"],
            "alert_required": event["alert_required"],
            "alert": safe_alert,
            "policy_version": event["policy_version"],
            "processing_ms": event["processing_ms"],
        },
    }


async def _maintain_evidence(
    evidence_store: EvidenceStore,
    stop_event: asyncio.Event,
) -> None:
    delay_seconds = 3_600
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay_seconds)
        except TimeoutError:
            try:
                await asyncio.to_thread(evidence_store.purge)
                delay_seconds = 3_600
            except Exception:
                print("A retenção de evidências da API será tentada novamente.")
                delay_seconds = 300


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()
    app_settings.validate()
    repository = Repository(app_settings.database_path)
    evidence_store = EvidenceStore(
        app_settings.evidence_dir,
        ttl=timedelta(days=app_settings.evidence_ttl_days),
        max_storage_bytes=app_settings.evidence_max_bytes,
        max_item_bytes=app_settings.evidence_max_item_bytes,
        evict_oldest=app_settings.evidence_evict_oldest,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        repository.initialize()
        evidence_store.initialize()
        evidence_store.purge()
        if app_settings.seed_demo_data:
            repository.seed_demo_data(app_settings.policy_version)
        stop_event = asyncio.Event()
        worker = AlertWorker(repository, app_settings)
        application.state.alert_worker = worker
        application.state.alert_stop_event = stop_event
        worker_task = asyncio.create_task(worker.run(stop_event), name="rag-audit-alert-worker")
        evidence_task = asyncio.create_task(
            _maintain_evidence(evidence_store, stop_event),
            name="rag-audit-evidence-maintenance",
        )
        application.state.alert_worker_task = worker_task
        application.state.evidence_maintenance_task = evidence_task
        try:
            yield
        finally:
            stop_event.set()
            await worker_task
            await evidence_task

    application = FastAPI(
        title="RAG-Audit",
        description=(
            "Auditoria contextual de acessos a salas críticas. "
            "A classificação não controla a fechadura nem substitui revisão humana."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.settings = app_settings
    application.state.repository = repository
    application.state.evidence_store = evidence_store
    application.state.broker = EventBroker()
    application.mount(
        "/static", StaticFiles(directory=str(PROJECT_DIR / "app" / "static")), name="static"
    )

    @application.middleware("http")
    async def privacy_and_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith(
            ("/v1/", "/dashboard", "/docs", "/openapi", "/login", "/logout")
        ):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        if request.url.path in ("/dashboard", "/login"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
            )
        return response

    @application.exception_handler(LoginRequired)
    async def _login_required_handler(request: Request, exc: LoginRequired):
        if "text/html" in request.headers.get("accept", ""):
            target = _safe_next(exc.next_url)
            suffix = f"?next={quote(target, safe='')}" if target != "/dashboard" else ""
            return RedirectResponse(url=f"/login{suffix}", status_code=303)
        return JSONResponse(
            {"detail": "Autenticação institucional necessária."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    def _render_login(
        request: Request, *, next_url: str, error: str | None, http_status: int = 200
    ) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "next": next_url,
                "error": error,
                "environment": app_settings.environment,
                "ad_enabled": app_settings.ad_login_enabled,
                "govbr_url": _govbr_login_url(app_settings, next_url),
            },
            status_code=http_status,
        )

    @application.get("/login", include_in_schema=False)
    async def login_page(request: Request, next: str = "/dashboard") -> Response:
        target = _safe_next(next)
        if app_settings.ad_login_enabled and _current_session(request) is not None:
            return RedirectResponse(url=target, status_code=303)
        return _render_login(request, next_url=target, error=None)

    @application.post("/login", include_in_schema=False)
    def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        next: str = Form("/dashboard"),
    ) -> Response:
        target = _safe_next(next)
        if not app_settings.ad_login_enabled:
            return _render_login(
                request,
                next_url=target,
                error="Login institucional ainda não configurado nesta instância.",
                http_status=503,
            )
        result = auth_ad.authenticate(
            app_settings, username.strip(), password, origin=_client_ip(request)
        )
        if not result.ok:
            return _render_login(
                request,
                next_url=target,
                error=_friendly_login_error(result),
                http_status=result.http_status or 401,
            )
        if not auth_ad.is_authorized(result.groups, app_settings.ad_allowed_groups):
            return _render_login(
                request,
                next_url=target,
                error="Seu usuário não tem permissão de acesso a este painel.",
                http_status=403,
            )
        response = RedirectResponse(url=target, status_code=303)
        _set_session_cookie(response, request, app_settings, result.token or "")
        return response

    @application.get("/logout", include_in_schema=False)
    def logout(request: Request) -> Response:
        token = request.cookies.get(app_settings.session_cookie_name)
        if token and app_settings.ad_login_enabled:
            auth_ad.logoff(app_settings, token)
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(app_settings.session_cookie_name, path="/")
        return response

    @application.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @application.get("/health/live", tags=["health"])
    async def health_live() -> dict:
        return {"status": "ok"}

    @application.get("/health/ready", tags=["health"])
    async def health_ready(request: Request) -> JSONResponse:
        database_ready = await asyncio.to_thread(_repository(request).healthcheck)
        policies_ready = await asyncio.to_thread(
            _repository(request).policy_set_ready, app_settings.policy_version
        )
        worker_task = getattr(request.app.state, "alert_worker_task", None)
        worker_ready = worker_task is not None and not worker_task.done()
        evidence_task = getattr(
            request.app.state,
            "evidence_maintenance_task",
            None,
        )
        evidence_ready = evidence_task is not None and not evidence_task.done()
        ready = database_ready and worker_ready and evidence_ready and policies_ready
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "unavailable",
                "database": database_ready,
                "alert_worker": worker_ready,
                "evidence_maintenance": evidence_ready,
                "policies": policies_ready,
            },
        )

    @application.get("/openapi.json", include_in_schema=False)
    async def protected_openapi(
        _admin: str = Depends(require_admin),
    ) -> JSONResponse:
        return JSONResponse(application.openapi())

    @application.get("/docs", include_in_schema=False)
    async def protected_docs(
        _admin: str = Depends(require_admin),
    ) -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="RAG-Audit · Documentação da API",
        )

    @application.post(
        "/v1/webhooks/access-events",
        tags=["camera"],
        summary="Receber um evento idempotente da câmera",
    )
    async def receive_access_event(
        event: AccessEventIn,
        request: Request,
        background_tasks: BackgroundTasks,
        x_camera_key: str | None = Header(default=None, alias="X-Camera-Key"),
        x_delivery_mode: str | None = Header(default=None, alias="X-Delivery-Mode"),
        x_event_queued_at: str | None = Header(
            default=None,
            alias="X-Event-Queued-At",
        ),
    ) -> JSONResponse:
        started = perf_counter()
        received_at = utc_now()
        repo = _repository(request)
        require_camera_key(request, event.camera_id, x_camera_key)
        if (x_delivery_mode is None) != (x_event_queued_at is None):
            raise HTTPException(
                status_code=422,
                detail="Metadados de entrega enfileirada incompletos.",
            )
        if x_delivery_mode not in {None, "durable-outbox"}:
            raise HTTPException(
                status_code=422,
                detail="X-Delivery-Mode não suportado.",
            )
        event_data, payload_hash = _canonical_payload(event)

        existing = await asyncio.to_thread(repo.get_idempotency_record, event.event_id)
        if existing:
            if existing["payload_hash"] != payload_hash:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="event_id já existe com conteúdo diferente.",
                )
            saved = await asyncio.to_thread(repo.get_event, event.event_id)
            return JSONResponse(
                status_code=200,
                content=_webhook_receipt(saved, idempotent_replay=True),
            )

        if app_settings.enforce_event_freshness:
            event_instant = event.timestamp.astimezone(UTC)
            max_age_seconds = app_settings.event_max_age_seconds
            if x_delivery_mode is not None:
                if x_delivery_mode != "durable-outbox" or not x_event_queued_at:
                    raise HTTPException(
                        status_code=422,
                        detail="Metadados de entrega enfileirada inválidos.",
                    )
                try:
                    parsed_queued_at = datetime.fromisoformat(x_event_queued_at)
                except (OverflowError, ValueError):
                    raise HTTPException(
                        status_code=422,
                        detail="X-Event-Queued-At inválido.",
                    ) from None
                if (
                    parsed_queued_at.tzinfo is None
                    or parsed_queued_at.utcoffset() is None
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="X-Event-Queued-At deve incluir fuso.",
                    )
                queued_at = parsed_queued_at.astimezone(UTC)
                if abs((queued_at - event_instant).total_seconds()) > 5:
                    raise HTTPException(
                        status_code=422,
                        detail="O instante da outbox não corresponde ao evento.",
                    )
                max_age_seconds = app_settings.queued_event_max_age_seconds
            age_seconds = (received_at - event_instant).total_seconds()
            if age_seconds > max_age_seconds:
                raise HTTPException(
                    status_code=422,
                    detail="Evento antigo demais para a janela de entrega.",
                )
            if age_seconds < -app_settings.event_future_skew_seconds:
                raise HTTPException(
                    status_code=422,
                    detail="Evento futuro além da tolerância de relógio.",
                )

        context = await asyncio.to_thread(
            repo.get_access_context,
            event.camera_id,
            (
                event.user_id
                if event.identity_status == IdentityStatus.MATCHED
                else f"UNRESOLVED:{event.event_id}"
            ),
            app_settings.policy_version,
        )
        camera = context.get("camera")
        if camera is None or not camera["active"]:
            raise HTTPException(status_code=403, detail="Câmera desconhecida ou inativa.")
        if camera["room_id"] != event.room_id:
            raise HTTPException(
                status_code=403,
                detail="A sala informada não corresponde à câmera cadastrada.",
            )

        evaluation = evaluate_access(
            event,
            context,
            fallback_timezone=app_settings.local_timezone,
        )
        if not evaluation.context_snapshot.get("policies"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Nenhuma política aplicável foi encontrada para a decisão; "
                    "o evento não foi gravado."
                ),
            )
        evaluation_data = evaluation.model_dump(mode="json")
        initial_ms = (perf_counter() - started) * 1000
        save_result = await asyncio.to_thread(
            repo.save_evaluated_event,
            event=event_data,
            payload_hash=payload_hash,
            evaluation=evaluation_data,
            received_at=received_at,
            policy_version=app_settings.policy_version,
            alert_channel=app_settings.alert_channel,
            processing_ms=initial_ms,
            alert_include_personal_data=app_settings.alert_include_personal_data,
            public_base_url=app_settings.public_base_url,
        )
        if save_result["state"] == "conflict":
            raise HTTPException(status_code=409, detail="event_id concorrente com conteúdo diferente.")
        if save_result["state"] == "duplicate":
            saved = await asyncio.to_thread(repo.get_event, event.event_id)
            return JSONResponse(
                status_code=200,
                content=_webhook_receipt(saved, idempotent_replay=True),
            )

        processing_ms = (perf_counter() - started) * 1000
        await asyncio.to_thread(
            repo.finalize_processing, event.event_id, utc_now(), processing_ms
        )
        saved = await asyncio.to_thread(repo.get_event, event.event_id)
        await request.app.state.broker.publish(
            {
                "event_id": event.event_id,
                "decision": evaluation.decision.value,
                "risk_level": evaluation.risk_level.value,
            }
        )
        if evaluation.alert_required:
            background_tasks.add_task(request.app.state.alert_worker.process_once)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=_webhook_receipt(saved, idempotent_replay=False),
        )

    @application.get("/v1/access-events", tags=["audit"])
    async def list_access_events(
        request: Request,
        _admin: str = Depends(require_admin),
        from_timestamp: datetime | None = Query(default=None, alias="from"),
        to_timestamp: datetime | None = Query(default=None, alias="to"),
        room_id: str | None = None,
        user_id: str | None = None,
        decision: Decision | None = None,
        risk_level: RiskLevel | None = None,
        alert_status: str | None = None,
        q: str | None = Query(default=None, max_length=100),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        event_filters = _filters(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            room_id=room_id,
            user_id=user_id,
            decision=decision,
            risk_level=risk_level,
            alert_status=alert_status,
            q=q,
        )
        events, total = await asyncio.to_thread(
            _repository(request).list_events,
            event_filters,
            limit=limit,
            offset=offset,
        )
        return {
            "items": [_summary(event) for event in events],
            "total": total,
            "limit": limit,
            "offset": offset,
            "generated_at": utc_now().isoformat(),
        }

    @application.get("/v1/access-events/stream", tags=["audit"])
    async def access_event_stream(
        request: Request, _admin: str = Depends(require_admin)
    ) -> StreamingResponse:
        return StreamingResponse(
            request.app.state.broker.stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/v1/access-events/{event_id}", tags=["audit"])
    async def get_access_event(
        event_id: str,
        request: Request,
        _admin: str = Depends(require_admin),
    ) -> dict:
        event = await asyncio.to_thread(_repository(request).get_event, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Evento não encontrado.")
        return event

    @application.get(
        "/v1/access-events/{event_id}/photo",
        include_in_schema=False,
    )
    async def access_event_photo(
        event_id: str,
        request: Request,
        _admin: str = Depends(require_admin),
        variant: str = Query(default="full"),
    ) -> Response:
        if variant not in {"full", "thumb"}:
            raise HTTPException(status_code=422, detail="variant deve ser full ou thumb.")
        event = await asyncio.to_thread(_repository(request).get_event, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Evento não encontrado.")
        payload = event.get("raw_payload") or {}
        reference = payload.get("evidence_ref") if isinstance(payload, dict) else None
        if (
            not isinstance(reference, str)
            or len(reference) != 64
            or any(character not in "0123456789abcdef" for character in reference)
        ):
            raise HTTPException(status_code=404, detail="Evento sem foto associada.")
        store: EvidenceStore = request.app.state.evidence_store
        try:
            data = await asyncio.to_thread(store.read, reference, variant=variant)
        except EvidenceNotFoundError:
            if variant != "thumb":
                raise HTTPException(status_code=404, detail="Foto indisponível.") from None
            try:
                data = await asyncio.to_thread(store.read, reference, variant="full")
            except EvidenceNotFoundError:
                raise HTTPException(status_code=404, detail="Foto indisponível.") from None
            except EvidenceIntegrityError:
                raise HTTPException(
                    status_code=409,
                    detail="A evidência falhou na verificação de integridade.",
                ) from None
        except EvidenceIntegrityError:
            raise HTTPException(
                status_code=409,
                detail="A evidência falhou na verificação de integridade.",
            ) from None
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=300"},
        )

    @application.get("/v1/metrics", tags=["audit"])
    async def get_metrics(
        request: Request,
        _admin: str = Depends(require_admin),
        from_timestamp: datetime | None = Query(default=None, alias="from"),
        to_timestamp: datetime | None = Query(default=None, alias="to"),
        room_id: str | None = None,
        user_id: str | None = None,
        decision: Decision | None = None,
        risk_level: RiskLevel | None = None,
        alert_status: str | None = None,
        q: str | None = Query(default=None, max_length=100),
    ) -> dict:
        event_filters = _filters(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            room_id=room_id,
            user_id=user_id,
            decision=decision,
            risk_level=risk_level,
            alert_status=alert_status,
            q=q,
        )
        return await asyncio.to_thread(_repository(request).metrics, event_filters)

    @application.get("/v1/rooms", tags=["audit"])
    async def get_rooms(
        request: Request, _admin: str = Depends(require_admin)
    ) -> dict:
        return {"items": await asyncio.to_thread(_repository(request).list_rooms)}

    @application.get("/v1/access-events/{event_id}/report.pdf", tags=["reports"])
    async def event_report(
        event_id: str,
        request: Request,
        _admin: str = Depends(require_admin),
    ) -> Response:
        event = await asyncio.to_thread(_repository(request).get_event, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Evento não encontrado.")
        event_timezone = (
            event.get("context_snapshot", {}).get("room", {}).get("timezone")
            or app_settings.local_timezone
        )
        content = await asyncio.to_thread(
            build_event_pdf, event, event_timezone
        )
        return Response(
            content=content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="rag-audit_evento_{event_id}.pdf"'
            },
        )

    @application.get("/v1/reports/access-events.pdf", tags=["reports"])
    async def consolidated_report(
        request: Request,
        admin: str = Depends(require_admin),
        from_timestamp: datetime | None = Query(default=None, alias="from"),
        to_timestamp: datetime | None = Query(default=None, alias="to"),
        room_id: str | None = None,
        user_id: str | None = None,
        decision: Decision | None = None,
        risk_level: RiskLevel | None = None,
        alert_status: str | None = None,
        q: str | None = Query(default=None, max_length=100),
    ) -> Response:
        event_filters = _filters(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            room_id=room_id,
            user_id=user_id,
            decision=decision,
            risk_level=risk_level,
            alert_status=alert_status,
            q=q,
        )
        events, total = await asyncio.to_thread(
            _repository(request).list_events,
            event_filters,
            limit=app_settings.report_event_limit + 1,
            offset=0,
        )
        if total > app_settings.report_event_limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"O relatório excede {app_settings.report_event_limit} eventos; "
                    "reduza o período ou aplique mais filtros."
                ),
            )
        content = await asyncio.to_thread(
            build_consolidated_pdf,
            events,
            {key: value for key, value in event_filters.items() if value is not None},
            app_settings.local_timezone,
            admin,
        )
        filename = datetime.now(ZoneInfo(app_settings.local_timezone)).strftime(
            "rag-audit_acessos_%Y-%m-%d_%H%M.pdf"
        )
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @application.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(
        request: Request, admin: str = Depends(require_admin)
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "admin": admin,
                "timezone": app_settings.local_timezone,
                "environment": app_settings.environment,
                "demo_mode": app_settings.seed_demo_data,
            },
        )

    return application


# Importado pelo Uvicorn: `uvicorn app.main:app`.
app = create_app()
