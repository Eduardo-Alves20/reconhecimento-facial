from __future__ import annotations

import asyncio
import logging

import httpx

from .config import Settings
from .database import Repository, utc_now


logger = logging.getLogger(__name__)


class AlertWorker:
    """Entrega alertas persistidos sem bloquear o webhook da câmera."""

    def __init__(self, repository: Repository, settings: Settings):
        self.repository = repository
        self.settings = settings
        self._lock = asyncio.Lock()

    async def process_once(self) -> int:
        async with self._lock:
            processed = 0
            while processed < 20:
                alerts = await asyncio.to_thread(
                    self.repository.claim_pending_alerts,
                    utc_now(),
                    include_not_configured=bool(self.settings.alert_webhook_url),
                    limit=1,
                    lease_seconds=max(30, int(self.settings.alert_timeout_seconds * 3)),
                )
                if not alerts:
                    break
                await self._deliver(alerts[0])
                processed += 1
            return processed

    async def _deliver(self, alert: dict) -> None:
        if not self.settings.alert_webhook_url:
            await asyncio.to_thread(
                self.repository.update_alert_delivery,
                alert["alert_id"],
                status="NOT_CONFIGURED",
                attempts=alert["attempts"],
                error="Canal externo não configurado; alerta preservado na outbox.",
            )
            return

        attempts = alert["attempts"] + 1
        try:
            async with httpx.AsyncClient(timeout=self.settings.alert_timeout_seconds) as client:
                response = await client.post(
                    self.settings.alert_webhook_url,
                    json=alert["payload"],
                    headers={"Idempotency-Key": alert["event_id"]},
                )
                response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            failed = attempts >= self.settings.alert_max_attempts
            await asyncio.to_thread(
                self.repository.update_alert_delivery,
                alert["alert_id"],
                status="FAILED" if failed else "RETRYING",
                attempts=attempts,
                error=f"{type(exc).__name__}: falha no canal externo"[:300],
                retry_after_seconds=0 if failed else min(60, 2**attempts),
            )
            return

        await asyncio.to_thread(
            self.repository.update_alert_delivery,
            alert["alert_id"],
            status="SENT",
            attempts=attempts,
        )

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.process_once()
            except Exception:
                logger.exception("Falha inesperada no worker de alertas; nova tentativa será feita.")
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=self.settings.alert_poll_seconds
                )
            except TimeoutError:
                pass
