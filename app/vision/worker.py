"""Worker RTSP: detecta rosto, confirma entrada e envia evento ao RAG-Audit."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from .camera import CameraConfig, build_rtsp_url
from .entry_tracker import (
    DirectedLine,
    EntryConfirmation,
    EntryTracker,
    EntryTrackerConfig,
    Point,
    Polygon,
    TrackObservation,
)
from .face_tracking import ConsensusResult, FaceCentroidTracker, IdentityConsensus
from .identity import IDENTIFIER_RE, safe_person_id
from .outbox import VisionEventOutbox
from .recognizer import MatchDecision, MatchStatus, MatchThresholds, OpenCVSFaceEngine


class VisionConfigurationError(ValueError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise VisionConfigurationError(f"Defina {name} no arquivo .env.")
    return value


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise VisionConfigurationError(f"{name} deve ser numérico.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise VisionConfigurationError(
            f"{name} deve estar entre {minimum} e {maximum}."
        )
    return value


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "sim", "yes", "on"}:
        return True
    if normalized in {"0", "false", "não", "nao", "no", "off"}:
        return False
    raise VisionConfigurationError(f"{name} deve ser true ou false.")


@dataclass(frozen=True, slots=True)
class VisionWorkerSettings:
    camera: CameraConfig
    camera_id: str
    room_id: str
    api_base_url: str
    api_key: str = field(repr=False)
    gallery_manifest: Path
    calibration_path: Path
    detector_model: Path
    recognizer_model: Path
    outbox_path: Path
    sample_fps: float = 4.0
    dry_run: bool = False
    thresholds: MatchThresholds = MatchThresholds()

    @classmethod
    def from_env(cls, project_root: Path) -> "VisionWorkerSettings":
        model_dir = Path(
            os.getenv(
                "RAG_AUDIT_VISION_MODELS_DIR",
                str(project_root / "data" / "vision-models"),
            )
        )
        api_base_url = os.getenv("RAG_AUDIT_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        parsed = urlparse(api_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise VisionConfigurationError("RAG_AUDIT_API_BASE_URL é inválida.")
        camera_id = os.getenv("RAG_AUDIT_VISION_CAMERA_ID", "cam-ti-01").strip()
        room_id = os.getenv("RAG_AUDIT_VISION_ROOM_ID", "sala_ti_01").strip()
        if not IDENTIFIER_RE.fullmatch(camera_id) or not IDENTIFIER_RE.fullmatch(room_id):
            raise VisionConfigurationError("IDs de câmera/sala possuem formato inválido.")
        try:
            camera_port = int(os.getenv("INTELBRAS_CAMERA_RTSP_PORT", "554"))
            camera_channel = int(os.getenv("INTELBRAS_CAMERA_CHANNEL", "1"))
            camera_subtype = int(os.getenv("INTELBRAS_CAMERA_SUBTYPE", "0"))
        except ValueError as exc:
            raise VisionConfigurationError("Porta, canal e subtipo da câmera devem ser inteiros.") from exc
        camera = CameraConfig(
            host=_required_env("INTELBRAS_CAMERA_HOST"),
            username=_required_env("INTELBRAS_CAMERA_USER"),
            password=_required_env("INTELBRAS_CAMERA_PASSWORD"),
            port=camera_port,
            channel=camera_channel,
            subtype=camera_subtype,
            timeout_seconds=_float_env(
                "INTELBRAS_CAMERA_TIMEOUT_SECONDS", 5.0, minimum=0.1, maximum=60
            ),
        )
        return cls(
            camera=camera,
            camera_id=camera_id,
            room_id=room_id,
            api_base_url=api_base_url,
            api_key=_required_env("RAG_AUDIT_CAMERA_API_KEY"),
            gallery_manifest=Path(
                os.getenv(
                    "RAG_AUDIT_GALLERY_MANIFEST",
                    str(project_root / "data" / "private" / "gallery" / "manifest.json"),
                )
            ),
            calibration_path=Path(
                os.getenv(
                    "RAG_AUDIT_CAMERA_CALIBRATION",
                    str(project_root / "data" / "private" / "camera-calibration.json"),
                )
            ),
            detector_model=model_dir / "face_detection_yunet_2023mar.onnx",
            recognizer_model=model_dir / "face_recognition_sface_2021dec.onnx",
            outbox_path=Path(
                os.getenv(
                    "RAG_AUDIT_VISION_OUTBOX_PATH",
                    str(project_root / "data" / "private" / "vision-outbox.db"),
                )
            ),
            sample_fps=_float_env(
                "RAG_AUDIT_VISION_SAMPLE_FPS", 4.0, minimum=0.5, maximum=15.0
            ),
            dry_run=_bool_env("RAG_AUDIT_VISION_DRY_RUN", False),
            thresholds=MatchThresholds(
                similarity=_float_env(
                    "RAG_AUDIT_FACE_SIMILARITY_THRESHOLD", 0.55, minimum=0, maximum=1
                ),
                margin=_float_env(
                    "RAG_AUDIT_FACE_MARGIN_THRESHOLD", 0.08, minimum=0, maximum=1
                ),
                face_quality=_float_env(
                    "RAG_AUDIT_FACE_QUALITY_THRESHOLD", 0.40, minimum=0, maximum=1
                ),
            ),
        )


def _point(value: Any) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise VisionConfigurationError("Cada ponto da calibração deve ser [x, y].")
    return Point(value[0], value[1])


def load_entry_tracker_config(
    path: str | Path, *, camera_id: str, room_id: str
) -> EntryTrackerConfig:
    calibration_path = Path(path)
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VisionConfigurationError(
            f"Calibração ausente: {calibration_path}."
        ) from exc
    except (OSError, ValueError) as exc:
        raise VisionConfigurationError("Arquivo de calibração inválido.") from exc
    if payload.get("schema_version") != 1:
        raise VisionConfigurationError("Versão de calibração não suportada.")
    if payload.get("camera_id") != camera_id or payload.get("room_id") != room_id:
        raise VisionConfigurationError("A calibração pertence a outra câmera ou sala.")
    try:
        door_zone = Polygon(tuple(_point(item) for item in payload["door_zone"]))
        inside_zone = Polygon(tuple(_point(item) for item in payload["inside_zone"]))
        line_payload = payload.get("entry_line")
        entry_line = (
            DirectedLine(
                _point(line_payload["start"]),
                _point(line_payload["end"]),
                line_payload.get("inside_side", "left"),
            )
            if line_payload
            else None
        )
        return EntryTrackerConfig(
            camera_id=camera_id,
            room_id=room_id,
            door_zone=door_zone,
            inside_zone=inside_zone,
            entry_line=entry_line,
            min_door_observations=int(payload.get("min_door_observations", 2)),
            min_inside_observations=int(payload.get("min_inside_observations", 2)),
            track_timeout=timedelta(seconds=float(payload.get("track_timeout_seconds", 5))),
            cooldown=timedelta(seconds=float(payload.get("cooldown_seconds", 10))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VisionConfigurationError(f"Geometria de calibração inválida: {exc}") from exc


def build_access_event(
    confirmation: EntryConfirmation,
    consensus: ConsensusResult,
    *,
    model_version: str,
) -> dict[str, Any]:
    if consensus.status == MatchStatus.MATCHED and consensus.external_id:
        user_id = safe_person_id(consensus.external_id)
        identity_status = "MATCHED"
        confidence = max(0.0, min(1.0, consensus.similarity or 0.0))
    elif consensus.status == MatchStatus.AMBIGUOUS:
        user_id = f"AMBIGUOUS:{confirmation.session_id}"
        identity_status = "AMBIGUOUS"
        confidence = None
    else:
        user_id = f"UNKNOWN:{confirmation.session_id}"
        identity_status = "UNKNOWN"
        confidence = None
    return {
        "event_id": f"entry:{confirmation.session_id}",
        "camera_id": confirmation.camera_id,
        "user_id": user_id,
        "room_id": confirmation.room_id,
        "timestamp": confirmation.confirmed_at.isoformat(),
        "door_result": "NOT_REPORTED",
        "recognition_confidence": confidence,
        "identity_status": identity_status,
        "entry_evidence": "VISION_LINE_CROSSING",
        "recognition_source": "LOCAL_SFACE",
        "track_id": confirmation.track_id,
        "recognition_model": model_version,
        "recognition_margin": consensus.margin,
        "face_quality": consensus.face_quality,
        "entry_confidence": min(1.0, confirmation.observation_count / 6.0),
    }


class RecognitionEngine(Protocol):
    model_version: str

    def detect(self, frame: Any) -> list[Any]: ...
    def match(self, face: Any) -> MatchDecision: ...


class VisionProcessor:
    def __init__(
        self,
        engine: RecognitionEngine,
        entry_tracker: EntryTracker,
        outbox: VisionEventOutbox,
    ) -> None:
        self.engine = engine
        self.entry_tracker = entry_tracker
        self.outbox = outbox
        self.face_tracker = FaceCentroidTracker()
        self.consensus = IdentityConsensus()
        self._observation_sequence = 0

    def process_frame(self, frame: Any, observed_at: datetime) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        faces = self.engine.detect(frame)
        tracked_faces = self.face_tracker.update(faces, observed_at)
        for tracked in tracked_faces:
            self._observation_sequence += 1
            observation_id = f"obs:{self.face_tracker.boot_id}:{self._observation_sequence}"
            self.consensus.add(tracked.track_id, self.engine.match(tracked.face))
            confirmation = self.entry_tracker.observe(
                TrackObservation(
                    track_id=tracked.track_id,
                    observed_at=observed_at,
                    centroid=Point(*tracked.face.centroid),
                    face_observation_id=observation_id,
                )
            )
            if confirmation is None:
                continue
            identity = self.consensus.resolve(tracked.track_id, consume=True)
            payload = build_access_event(
                confirmation, identity, model_version=self.engine.model_version
            )
            self.outbox.enqueue(payload, now=observed_at)
            emitted.append(payload)
        return emitted


class AuditApiClient:
    def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 3.0) -> None:
        self.client = httpx.Client(
            base_url=base_url,
            headers={"X-Camera-Key": api_key},
            timeout=timeout_seconds,
        )

    def close(self) -> None:
        self.client.close()

    def send(self, payload: dict[str, Any]) -> tuple[bool, str]:
        try:
            response = self.client.post("/v1/webhooks/access-events", json=payload)
        except httpx.HTTPError:
            return False, "API_UNAVAILABLE"
        if response.status_code in {200, 201}:
            return True, "DELIVERED"
        return False, f"API_HTTP_{response.status_code}"


def flush_outbox(
    outbox: VisionEventOutbox, api: AuditApiClient, *, now: datetime | None = None
) -> tuple[int, int]:
    instant = now or datetime.now(UTC)
    sent = failed = 0
    for item in outbox.due(now=instant):
        ok, code = api.send(item["payload"])
        if ok:
            outbox.mark_sent(item["event_id"], now=instant)
            sent += 1
        else:
            outbox.mark_failure(
                item["event_id"],
                error_code=code,
                attempts=item["attempts"],
                now=instant,
            )
            failed += 1
    return sent, failed


def create_processor(settings: VisionWorkerSettings) -> VisionProcessor:
    tracker_config = load_entry_tracker_config(
        settings.calibration_path,
        camera_id=settings.camera_id,
        room_id=settings.room_id,
    )
    engine = OpenCVSFaceEngine(
        settings.detector_model,
        settings.recognizer_model,
        thresholds=settings.thresholds,
    )
    enrolled = engine.load_gallery(settings.gallery_manifest)
    if enrolled < 1:
        raise VisionConfigurationError("A galeria facial está vazia.")
    outbox = VisionEventOutbox(settings.outbox_path)
    outbox.initialize()
    return VisionProcessor(engine, EntryTracker(tracker_config), outbox)


def run_forever(settings: VisionWorkerSettings) -> None:
    processor = create_processor(settings)
    api = AuditApiClient(settings.api_base_url, settings.api_key)
    cv2 = processor.engine.cv2  # type: ignore[attr-defined]
    rtsp_url = build_rtsp_url(settings.camera)
    sample_interval = 1.0 / settings.sample_fps
    last_processed = 0.0
    print(
        f"Worker iniciado para {settings.camera_id}/{settings.room_id}; "
        f"câmera {settings.camera.display_url}."
    )
    if settings.dry_run:
        print("Modo dry-run: entradas serão detectadas e enfileiradas, sem envio à API.")
    try:
        while True:
            capture = cv2.VideoCapture()
            timeout_ms = round(settings.camera.timeout_seconds * 1000)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            opened = capture.open(rtsp_url, getattr(cv2, "CAP_FFMPEG", 0))
            if not opened:
                print("Stream indisponível; nova tentativa em 3 segundos.")
                capture.release()
                time.sleep(3)
                continue
            print("Stream RTSP conectado.")
            try:
                while True:
                    received, frame = capture.read()
                    if not received or frame is None:
                        print("Stream interrompido; iniciando reconexão segura.")
                        break
                    monotonic_now = time.monotonic()
                    if monotonic_now - last_processed < sample_interval:
                        continue
                    last_processed = monotonic_now
                    emitted = processor.process_frame(frame, datetime.now(UTC))
                    for payload in emitted:
                        print(
                            f"Entrada visual enfileirada: {payload['event_id']} "
                            f"({payload['identity_status']})."
                        )
                    if not settings.dry_run:
                        sent, failed = flush_outbox(processor.outbox, api)
                        if sent:
                            print(f"{sent} evento(s) entregue(s) à API.")
                        if failed:
                            print(f"{failed} evento(s) mantido(s) para reenvio.")
            finally:
                capture.release()
            time.sleep(1)
    finally:
        api.close()
