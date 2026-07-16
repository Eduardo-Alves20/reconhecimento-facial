from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .camera import CameraConfig, build_rtsp_url
from .entry_tracker import DirectedLine, EntryTracker, EntryTrackerConfig, Point, Polygon
from .evidence import EvidenceStore
from .identity import safe_person_id
from .learned import LearnedGallery
from .outbox import VisionEventOutbox
from .pipeline import (
    ConsensusPolicy,
    DetectionRegion,
    DoorEventProcessor,
    LearningPolicy,
    VisionProcessor,
    build_access_event,
    build_door_event,
)
from .recognizer import ArcFaceEngine, FaceQualityPolicy, GalleryEntry, MatchThresholds


_WORKER_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class VisionConfigurationError(ValueError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise VisionConfigurationError(f"Defina {name} no arquivo .env.")
    return value


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise VisionConfigurationError(f"{name} deve ser numérico.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise VisionConfigurationError(
            f"{name} deve estar entre {minimum} e {maximum}."
        )
    return value


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    if not re.fullmatch(r"[+-]?\d+", raw):
        raise VisionConfigurationError(f"{name} deve ser inteiro.")
    value = int(raw)
    if not minimum <= value <= maximum:
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


def _path_env(name: str, default: Path, project_root: Path) -> Path:
    raw = os.getenv(name)
    value = str(default) if raw is None else raw.strip()
    if not value:
        raise VisionConfigurationError(f"{name} não pode ficar vazio.")
    configured = Path(value)
    return configured if configured.is_absolute() else (project_root / configured).resolve()


def _camera_path_env(
    name: str,
    default: Path,
    project_root: Path,
    camera_id: str,
) -> Path:
    raw = os.getenv(name)
    value = str(default) if raw is None else raw.strip()
    try:
        rendered = value.format(camera_id=camera_id)
    except (KeyError, ValueError) as exc:
        raise VisionConfigurationError(
            f"{name} contém um placeholder inválido."
        ) from exc
    if "{" in rendered or "}" in rendered:
        raise VisionConfigurationError(f"{name} contém um placeholder inválido.")
    configured = Path(rendered)
    return configured if configured.is_absolute() else (project_root / configured).resolve()


def _providers_env() -> tuple[str, ...]:
    raw = os.getenv("RAG_AUDIT_VISION_ONNX_PROVIDERS", "CPUExecutionProvider")
    providers = tuple(item.strip() for item in raw.split(",") if item.strip())
    if (
        not providers
        or len(providers) > 4
        or len(set(providers)) != len(providers)
        or any(
            len(provider) > 128
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*ExecutionProvider", provider)
            for provider in providers
        )
    ):
        raise VisionConfigurationError(
            "RAG_AUDIT_VISION_ONNX_PROVIDERS contém um provider inválido."
        )
    return providers


def _det_size_env() -> tuple[int, int]:
    raw = os.getenv("RAG_AUDIT_VISION_DETECTION_SIZE", "640x640").strip().lower()
    match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", raw)
    if match is None:
        raise VisionConfigurationError(
            "RAG_AUDIT_VISION_DETECTION_SIZE deve usar o formato 640x640."
        )
    width, height = (int(value) for value in match.groups())
    if not 320 <= width <= 1920 or not 320 <= height <= 1920:
        raise VisionConfigurationError(
            "RAG_AUDIT_VISION_DETECTION_SIZE deve ficar entre 320 e 1920 pixels."
        )
    return width, height


@dataclass(frozen=True, slots=True)
class VisionWorkerSettings:
    camera: CameraConfig
    camera_id: str
    room_id: str
    api_base_url: str
    api_key: str = field(repr=False)
    gallery_manifest: Path
    calibration_path: Path
    models_dir: Path
    model_fingerprint: str
    outbox_path: Path
    evidence_dir: Path
    learned_path: Path
    lease_path: Path | None = None
    sample_fps: float = 4.0
    dry_run: bool = True
    mode: str = "entry"
    allow_door_events: bool = False
    min_door_frames: int = 4
    roi_padding: float = 0.05
    decision_wait_seconds: float = 1.5
    evidence_ttl_days: int = 30
    evidence_max_bytes: int = 10 * 1024 * 1024 * 1024
    evidence_max_item_bytes: int = 25 * 1024 * 1024
    evidence_evict_oldest: bool = True
    outbox_sent_retention_days: int = 30
    thresholds: MatchThresholds = MatchThresholds()
    quality_policy: FaceQualityPolicy = FaceQualityPolicy()
    consensus_policy: ConsensusPolicy = ConsensusPolicy()
    learning_policy: LearningPolicy = LearningPolicy()
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    detection_size: tuple[int, int] = (640, 640)
    learn_max_per_person: int = 5
    learned_refresh_seconds: int = 15
    gallery_cache_path: Path | None = None
    dry_run_save_evidence: bool = False
    dry_run_outbox_retention_days: int = 7
    dry_run_outbox_max_events: int = 10_000

    def __post_init__(self) -> None:
        if not _WORKER_IDENTIFIER_RE.fullmatch(
            self.camera_id
        ) or not _WORKER_IDENTIFIER_RE.fullmatch(self.room_id):
            raise VisionConfigurationError("IDs de câmera/sala possuem formato inválido.")
        if self.mode not in {"entry", "door"}:
            raise VisionConfigurationError("RAG_AUDIT_VISION_MODE deve ser entry ou door.")
        if self.mode == "door" and not self.dry_run and not self.allow_door_events:
            raise VisionConfigurationError(
                "O modo door é apenas observacional. Para publicar esses eventos, "
                "defina RAG_AUDIT_VISION_ALLOW_DOOR_EVENTS=true de forma consciente."
            )
        fingerprint = self.model_fingerprint.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise VisionConfigurationError(
                "RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256 deve ser um SHA-256 válido."
            )
        if self.evidence_max_item_bytes > self.evidence_max_bytes:
            raise VisionConfigurationError(
                "O limite por evidência não pode superar a cota total."
            )
        if not 5 <= self.learned_refresh_seconds <= 3600:
            raise VisionConfigurationError(
                "O intervalo de recarga aprendida deve ficar entre 5 e 3600 segundos."
            )
        if not 1 <= self.dry_run_outbox_retention_days <= 365:
            raise VisionConfigurationError(
                "A retenção da outbox de dry-run deve ficar entre 1 e 365 dias."
            )
        if not 100 <= self.dry_run_outbox_max_events <= 1_000_000:
            raise VisionConfigurationError(
                "O limite da outbox de dry-run deve ficar entre 100 e 1000000 eventos."
            )
        object.__setattr__(self, "model_fingerprint", fingerprint)

    @classmethod
    def from_env(cls, project_root: Path) -> "VisionWorkerSettings":
        api_base_url = os.getenv(
            "RAG_AUDIT_API_BASE_URL",
            "http://127.0.0.1:8000",
        ).strip().rstrip("/")
        parsed = urlparse(api_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise VisionConfigurationError("RAG_AUDIT_API_BASE_URL é inválida.")

        camera_id = os.getenv("RAG_AUDIT_VISION_CAMERA_ID", "cam-ti-01").strip()
        room_id = os.getenv("RAG_AUDIT_VISION_ROOM_ID", "sala_ti_01").strip()
        if not _WORKER_IDENTIFIER_RE.fullmatch(
            camera_id
        ) or not _WORKER_IDENTIFIER_RE.fullmatch(room_id):
            raise VisionConfigurationError("IDs de câmera/sala possuem formato inválido.")

        camera = CameraConfig(
            host=_required_env("INTELBRAS_CAMERA_HOST"),
            username=_required_env("INTELBRAS_CAMERA_USER"),
            password=_required_env("INTELBRAS_CAMERA_PASSWORD"),
            port=_int_env("INTELBRAS_CAMERA_RTSP_PORT", 554, minimum=1, maximum=65_535),
            channel=_int_env("INTELBRAS_CAMERA_CHANNEL", 1, minimum=1, maximum=64),
            subtype=_int_env("INTELBRAS_CAMERA_SUBTYPE", 0, minimum=0, maximum=1),
            timeout_seconds=_float_env(
                "INTELBRAS_CAMERA_TIMEOUT_SECONDS",
                5.0,
                minimum=0.1,
                maximum=60,
            ),
        )
        private_root = project_root / "data" / "private"
        gallery_manifest = _path_env(
            "RAG_AUDIT_GALLERY_MANIFEST",
            private_root / "gallery" / "manifest.json",
            project_root,
        )
        mode = os.getenv("RAG_AUDIT_VISION_MODE", "entry").strip().lower() or "entry"
        dry_run = _bool_env("RAG_AUDIT_VISION_DRY_RUN", True)
        production_outbox = _camera_path_env(
            "RAG_AUDIT_VISION_OUTBOX_PATH",
            private_root / "outbox" / f"{camera_id}.db",
            project_root,
            camera_id,
        )
        dry_run_outbox = _camera_path_env(
            "RAG_AUDIT_VISION_DRY_RUN_OUTBOX_PATH",
            private_root / "outbox" / f"{camera_id}.dry-run.db",
            project_root,
            camera_id,
        )
        if production_outbox == dry_run_outbox:
            raise VisionConfigurationError(
                "As outboxes de dry-run e produção precisam ser diferentes."
            )
        if camera_id not in str(production_outbox) or camera_id not in str(
            dry_run_outbox
        ):
            raise VisionConfigurationError(
                "Os caminhos de outbox precisam incluir o camera_id."
            )
        gallery_cache_path = _camera_path_env(
            "RAG_AUDIT_GALLERY_CACHE_PATH",
            private_root / "cache" / camera_id / "embeddings.arcface.npz",
            project_root,
            camera_id,
        )
        if camera_id not in str(gallery_cache_path):
            raise VisionConfigurationError(
                "O caminho do cache da galeria precisa incluir o camera_id."
            )
        return cls(
            camera=camera,
            camera_id=camera_id,
            room_id=room_id,
            api_base_url=api_base_url,
            api_key=_required_env("RAG_AUDIT_CAMERA_API_KEY"),
            gallery_manifest=gallery_manifest,
            gallery_cache_path=gallery_cache_path,
            calibration_path=_path_env(
                "RAG_AUDIT_CAMERA_CALIBRATION",
                private_root / "camera-calibration.json",
                project_root,
            ),
            models_dir=_path_env(
                "RAG_AUDIT_VISION_MODELS_DIR",
                project_root / "data" / "vision-models",
                project_root,
            ),
            model_fingerprint=_required_env(
                "RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256"
            ),
            outbox_path=dry_run_outbox if dry_run else production_outbox,
            evidence_dir=_path_env(
                "RAG_AUDIT_VISION_EVIDENCE_DIR",
                private_root / "evidence",
                project_root,
            ),
            learned_path=_path_env(
                "RAG_AUDIT_LEARNED_DB_PATH",
                gallery_manifest.parent / "learned.db",
                project_root,
            ),
            lease_path=production_outbox,
            sample_fps=_float_env(
                "RAG_AUDIT_VISION_SAMPLE_FPS",
                4.0,
                minimum=0.5,
                maximum=15.0,
            ),
            dry_run=dry_run,
            mode=mode,
            allow_door_events=_bool_env(
                "RAG_AUDIT_VISION_ALLOW_DOOR_EVENTS",
                False,
            ),
            min_door_frames=_int_env(
                "RAG_AUDIT_VISION_MIN_DOOR_FRAMES",
                4,
                minimum=1,
                maximum=30,
            ),
            roi_padding=_float_env(
                "RAG_AUDIT_VISION_ROI_PADDING",
                0.05,
                minimum=0,
                maximum=0.5,
            ),
            decision_wait_seconds=_float_env(
                "RAG_AUDIT_VISION_DECISION_WAIT_SECONDS",
                1.5,
                minimum=0,
                maximum=5,
            ),
            evidence_ttl_days=_int_env(
                "RAG_AUDIT_EVIDENCE_TTL_DAYS",
                30,
                minimum=1,
                maximum=3650,
            ),
            evidence_max_bytes=round(
                _float_env(
                    "RAG_AUDIT_EVIDENCE_MAX_GB",
                    10,
                    minimum=0.1,
                    maximum=10_000,
                )
                * 1024
                * 1024
                * 1024
            ),
            evidence_max_item_bytes=round(
                _float_env(
                    "RAG_AUDIT_EVIDENCE_MAX_ITEM_MB",
                    25,
                    minimum=1,
                    maximum=500,
                )
                * 1024
                * 1024
            ),
            evidence_evict_oldest=_bool_env(
                "RAG_AUDIT_EVIDENCE_EVICT_OLDEST",
                True,
            ),
            outbox_sent_retention_days=_int_env(
                "RAG_AUDIT_VISION_OUTBOX_SENT_RETENTION_DAYS",
                30,
                minimum=1,
                maximum=3650,
            ),
            thresholds=MatchThresholds(
                similarity=_float_env(
                    "RAG_AUDIT_FACE_SIMILARITY_THRESHOLD",
                    0.55,
                    minimum=0,
                    maximum=1,
                ),
                margin=_float_env(
                    "RAG_AUDIT_FACE_MARGIN_THRESHOLD",
                    0.10,
                    minimum=0,
                    maximum=1,
                ),
                face_quality=_float_env(
                    "RAG_AUDIT_FACE_QUALITY_THRESHOLD",
                    0.50,
                    minimum=0,
                    maximum=1,
                ),
            ),
            quality_policy=FaceQualityPolicy(
                min_inter_eye_pixels=_float_env(
                    "RAG_AUDIT_FACE_MIN_INTER_EYE_PIXELS",
                    40,
                    minimum=10,
                    maximum=300,
                ),
                full_inter_eye_pixels=_float_env(
                    "RAG_AUDIT_FACE_FULL_INTER_EYE_PIXELS",
                    80,
                    minimum=20,
                    maximum=500,
                ),
                min_focus_variance=_float_env(
                    "RAG_AUDIT_FACE_MIN_FOCUS_VARIANCE",
                    20,
                    minimum=1,
                    maximum=1000,
                ),
                full_focus_variance=_float_env(
                    "RAG_AUDIT_FACE_FULL_FOCUS_VARIANCE",
                    260,
                    minimum=2,
                    maximum=5000,
                ),
                max_pitch_degrees=_float_env(
                    "RAG_AUDIT_FACE_MAX_PITCH_DEGREES",
                    25,
                    minimum=1,
                    maximum=90,
                ),
                max_yaw_degrees=_float_env(
                    "RAG_AUDIT_FACE_MAX_YAW_DEGREES",
                    35,
                    minimum=1,
                    maximum=90,
                ),
                max_roll_degrees=_float_env(
                    "RAG_AUDIT_FACE_MAX_ROLL_DEGREES",
                    25,
                    minimum=1,
                    maximum=90,
                ),
            ),
            consensus_policy=ConsensusPolicy(
                history_size=_int_env(
                    "RAG_AUDIT_CONSENSUS_HISTORY_SIZE",
                    12,
                    minimum=4,
                    maximum=120,
                ),
                minimum_matches=_int_env(
                    "RAG_AUDIT_CONSENSUS_MIN_MATCHES",
                    4,
                    minimum=2,
                    maximum=60,
                ),
                minimum_ratio=_float_env(
                    "RAG_AUDIT_CONSENSUS_MIN_RATIO",
                    0.70,
                    minimum=0.5,
                    maximum=1,
                ),
                minimum_consecutive=_int_env(
                    "RAG_AUDIT_CONSENSUS_MIN_CONSECUTIVE",
                    3,
                    minimum=1,
                    maximum=30,
                ),
            ),
            learning_policy=LearningPolicy(
                enabled=_bool_env("RAG_AUDIT_LEARN_ENABLED", False),
                allow_dry_run=_bool_env(
                    "RAG_AUDIT_LEARN_ALLOW_DRY_RUN",
                    False,
                ),
                min_similarity=_float_env(
                    "RAG_AUDIT_LEARN_MIN_SIMILARITY",
                    0.65,
                    minimum=0,
                    maximum=1,
                ),
                max_similarity=_float_env(
                    "RAG_AUDIT_LEARN_MAX_SIMILARITY",
                    0.95,
                    minimum=0,
                    maximum=1,
                ),
                min_margin=_float_env(
                    "RAG_AUDIT_LEARN_MIN_MARGIN",
                    0.12,
                    minimum=0,
                    maximum=1,
                ),
                min_quality=_float_env(
                    "RAG_AUDIT_LEARN_MIN_QUALITY",
                    0.60,
                    minimum=0,
                    maximum=1,
                ),
                min_supporting_frames=_int_env(
                    "RAG_AUDIT_LEARN_MIN_SUPPORTING_FRAMES",
                    4,
                    minimum=2,
                    maximum=60,
                ),
            ),
            providers=_providers_env(),
            detection_size=_det_size_env(),
            learn_max_per_person=_int_env(
                "RAG_AUDIT_LEARN_MAX_PER_PERSON",
                5,
                minimum=1,
                maximum=50,
            ),
            learned_refresh_seconds=_int_env(
                "RAG_AUDIT_LEARN_REFRESH_SECONDS",
                15,
                minimum=5,
                maximum=3600,
            ),
            dry_run_save_evidence=_bool_env(
                "RAG_AUDIT_VISION_DRY_RUN_SAVE_EVIDENCE",
                False,
            ),
            dry_run_outbox_retention_days=_int_env(
                "RAG_AUDIT_VISION_DRY_RUN_OUTBOX_RETENTION_DAYS",
                7,
                minimum=1,
                maximum=365,
            ),
            dry_run_outbox_max_events=_int_env(
                "RAG_AUDIT_VISION_DRY_RUN_OUTBOX_MAX_EVENTS",
                10_000,
                minimum=100,
                maximum=1_000_000,
            ),
        )


def _point(value: Any) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise VisionConfigurationError("Cada ponto da calibração deve ser [x, y].")
    return Point(value[0], value[1])


def load_entry_tracker_config(
    path: str | Path,
    *,
    camera_id: str,
    room_id: str,
) -> EntryTrackerConfig:
    calibration_path = Path(path)
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VisionConfigurationError(
            f"Calibração ausente: {calibration_path}."
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisionConfigurationError("Arquivo de calibração inválido.") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
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
            track_timeout=timedelta(
                seconds=float(payload.get("track_timeout_seconds", 5))
            ),
            cooldown=timedelta(
                seconds=float(payload.get("cooldown_seconds", 10))
            ),
            line_deadband=float(payload.get("line_deadband", 0.015)),
            line_segment_margin=float(payload.get("line_segment_margin", 0.05)),
            min_crossing_displacement=float(
                payload.get("min_crossing_displacement", 0.04)
            ),
            max_transition=timedelta(
                seconds=float(payload.get("max_transition_seconds", 3))
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VisionConfigurationError(
            f"Geometria de calibração inválida: {exc}"
        ) from exc


class AuditApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.client = httpx.Client(
            base_url=base_url,
            headers={"X-Camera-Key": api_key},
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        self.client.close()

    def send(
        self,
        payload: dict[str, Any],
        *,
        queued_at: datetime,
    ) -> tuple[bool, str]:
        try:
            response = self.client.post(
                "/v1/webhooks/access-events",
                json=payload,
                headers={
                    "X-Delivery-Mode": "durable-outbox",
                    "X-Event-Queued-At": queued_at.astimezone(UTC).isoformat(),
                },
            )
        except httpx.HTTPError:
            return False, "API_UNAVAILABLE"
        if response.status_code in {200, 201}:
            return True, "DELIVERED"
        return False, f"API_HTTP_{response.status_code}"


def flush_outbox(
    outbox: VisionEventOutbox,
    api: AuditApiClient,
    *,
    now: datetime | None = None,
    limit: int = 5,
    expected_camera_id: str | None = None,
) -> tuple[int, int, int]:
    instant = now or datetime.now(UTC)
    sent = retrying = dead = 0
    for item in outbox.due(now=instant, limit=limit):
        if (
            expected_camera_id is not None
            and item["payload"].get("camera_id") != expected_camera_id
        ):
            outbox.mark_dead(
                item["event_id"],
                error_code="OUTBOX_CAMERA_MISMATCH",
                attempts=item["attempts"],
            )
            dead += 1
            continue
        ok, code = api.send(
            item["payload"],
            queued_at=item["created_at"],
        )
        if ok:
            outbox.mark_sent(item["event_id"], now=instant)
            sent += 1
        elif code in {
            "API_HTTP_400",
            "API_HTTP_409",
            "API_HTTP_413",
            "API_HTTP_422",
        }:
            outbox.mark_dead(
                item["event_id"],
                error_code=code,
                attempts=item["attempts"],
            )
            dead += 1
        else:
            outbox.mark_failure(
                item["event_id"],
                error_code=code,
                attempts=item["attempts"],
                now=instant,
            )
            retrying += 1
    return sent, retrying, dead


class _WorkerServices:
    def __init__(
        self,
        processor: VisionProcessor | DoorEventProcessor,
        settings: VisionWorkerSettings,
    ) -> None:
        self.processor = processor
        self.settings = settings
        self.lease_outbox = (
            processor.outbox
            if settings.lease_path is None
            or settings.lease_path == settings.outbox_path
            else VisionEventOutbox(settings.lease_path)
        )
        if self.lease_outbox is not processor.outbox:
            self.lease_outbox.initialize()
        self.owner_id = f"worker:{uuid.uuid4().hex}"
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._drain = False
        self._failure: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"vision-services-{settings.camera_id}",
            daemon=True,
        )
        self._lease_thread = threading.Thread(
            target=self._renew_lease,
            name=f"vision-lease-{settings.camera_id}",
            daemon=True,
        )

    def start(self) -> None:
        if not self.lease_outbox.acquire_lease(
            camera_id=self.settings.camera_id,
            owner_id=self.owner_id,
            now=datetime.now(UTC),
        ):
            raise VisionConfigurationError(
                "Já existe um worker ativo para esta câmera e outbox."
            )
        self._lease_thread.start()
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def ensure_healthy(self) -> None:
        if self._failure is not None:
            raise VisionConfigurationError(
                "O serviço de entrega e manutenção foi interrompido."
            ) from self._failure

    def close(self, *, drain: bool) -> None:
        self._drain = drain
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=20)
        self._lease_thread.join(timeout=5)
        if self._thread.is_alive() or self._lease_thread.is_alive():
            print("Serviço de entrega ainda finalizando; o lease será preservado.")
            return
        self.lease_outbox.release_lease(
            camera_id=self.settings.camera_id,
            owner_id=self.owner_id,
        )

    def _renew_lease(self) -> None:
        try:
            while not self._stop.wait(timeout=30):
                if not self.lease_outbox.acquire_lease(
                    camera_id=self.settings.camera_id,
                    owner_id=self.owner_id,
                    now=datetime.now(UTC),
                ):
                    raise VisionConfigurationError(
                        "O lease exclusivo da câmera foi perdido."
                    )
        except Exception as exc:
            self._failure = exc
            self._stop.set()
            self._wake.set()

    def _run(self) -> None:
        api = (
            None
            if self.settings.dry_run
            else AuditApiClient(
                self.settings.api_base_url,
                self.settings.api_key,
            )
        )
        started_at = datetime.now(UTC)
        maintenance_offset = 10 + sum(ord(char) for char in self.settings.camera_id) % 40
        next_maintenance = started_at + timedelta(minutes=maintenance_offset)
        next_learning_refresh = started_at + timedelta(
            seconds=self.settings.learned_refresh_seconds
        )
        try:
            while not self._stop.is_set():
                now = datetime.now(UTC)
                if api is not None:
                    _report_delivery(
                        *flush_outbox(
                            self.processor.outbox,
                            api,
                            now=now,
                            expected_camera_id=self.settings.camera_id,
                        )
                    )
                if now >= next_learning_refresh and self.processor.learned is not None:
                    try:
                        if not self.processor.learned.initialized:
                            self.processor.learned.initialize()
                        _refresh_learned_references(
                            self.processor.engine,
                            self.processor.learned,
                        )
                    except Exception:
                        self.processor.learned.initialized = False
                        self.processor.engine.replace_learned_references(())
                        print(
                            "Referências aprendidas desativadas até a próxima "
                            "recarga válida."
                        )
                    next_learning_refresh = now + timedelta(
                        seconds=self.settings.learned_refresh_seconds
                    )
                if now >= next_maintenance:
                    try:
                        if self.settings.dry_run:
                            self.processor.outbox.purge_dry_run(
                                before=now
                                - timedelta(
                                    days=self.settings.dry_run_outbox_retention_days
                                ),
                                max_events=self.settings.dry_run_outbox_max_events,
                            )
                        else:
                            self.processor.outbox.purge_sent(
                                before=now
                                - timedelta(
                                    days=self.settings.outbox_sent_retention_days
                                )
                            )
                    except Exception:
                        print("A retenção da outbox será tentada novamente.")
                    if self.processor.evidence_store is not None:
                        try:
                            self.processor.evidence_store.purge(now=now)
                        except Exception:
                            print("A retenção de evidências será tentada novamente.")
                    next_maintenance = now + timedelta(hours=1)
                self._wake.wait(timeout=1)
                self._wake.clear()
            if self._drain and api is not None:
                _report_delivery(
                    *flush_outbox(
                        self.processor.outbox,
                        api,
                        now=datetime.now(UTC),
                        expected_camera_id=self.settings.camera_id,
                    )
                )
        except Exception as exc:
            self._failure = exc
        finally:
            if api is not None:
                api.close()


def _verified_model_fingerprint(settings: VisionWorkerSettings) -> str:
    try:
        from scripts.verify_vision_models import ModelBundleError, verify_bundle

        return verify_bundle(settings.models_dir, settings.model_fingerprint)
    except ImportError as exc:
        raise VisionConfigurationError(
            "O verificador do bundle de modelos não está disponível."
        ) from exc
    except ModelBundleError as exc:
        raise VisionConfigurationError(str(exc)) from exc


def _refresh_learned_references(
    engine: ArcFaceEngine,
    learned: LearnedGallery,
) -> int:
    approved = learned.approved_references(
        engine,
        allowed_external_ids=engine.official_external_ids,
    )
    engine.replace_learned_references(
        [
            GalleryEntry(external_id, display_name, feature)
            for external_id, display_name, feature in approved
        ]
    )
    return len(approved)


def create_processor(
    settings: VisionWorkerSettings,
) -> VisionProcessor | DoorEventProcessor:
    tracker_config = load_entry_tracker_config(
        settings.calibration_path,
        camera_id=settings.camera_id,
        room_id=settings.room_id,
    )
    if settings.mode == "entry" and tracker_config.entry_line is None:
        raise VisionConfigurationError(
            "O modo entry exige entry_line na calibração."
        )
    fingerprint = _verified_model_fingerprint(settings)
    engine = ArcFaceEngine(
        thresholds=settings.thresholds,
        model_root=settings.models_dir,
        model_fingerprint=fingerprint,
        det_size=settings.detection_size,
        providers=settings.providers,
        quality_policy=settings.quality_policy,
    )
    enrolled = engine.load_gallery(
        settings.gallery_manifest,
        cache_path=settings.gallery_cache_path,
    )
    if enrolled < 1:
        raise VisionConfigurationError("A galeria facial está vazia.")

    learned = LearnedGallery(
        settings.learned_path,
        max_per_person=settings.learn_max_per_person,
    )
    try:
        learned.initialize()
        learned_count = _refresh_learned_references(engine, learned)
    except Exception:
        engine.replace_learned_references(())
        learned_count = 0
        print("Referências aprendidas indisponíveis; a galeria oficial permanece ativa.")
    if learned_count:
        print(f"{learned_count} referência(s) aprovada(s) carregada(s).")

    evidence_store: EvidenceStore | None = None
    if not settings.dry_run or settings.dry_run_save_evidence:
        evidence_store = EvidenceStore(
            settings.evidence_dir,
            ttl=timedelta(days=settings.evidence_ttl_days),
            max_storage_bytes=settings.evidence_max_bytes,
            max_item_bytes=settings.evidence_max_item_bytes,
            evict_oldest=settings.evidence_evict_oldest,
        )
        evidence_store.initialize()
        evidence_store.purge()

    outbox = VisionEventOutbox(settings.outbox_path)
    outbox.initialize()
    if settings.dry_run:
        outbox.purge_dry_run(
            before=datetime.now(UTC)
            - timedelta(days=settings.dry_run_outbox_retention_days),
            max_events=settings.dry_run_outbox_max_events,
        )
    else:
        outbox.purge_sent(
            before=datetime.now(UTC)
            - timedelta(days=settings.outbox_sent_retention_days)
        )
    detection_region = DetectionRegion.from_polygons(
        tracker_config.door_zone,
        tracker_config.inside_zone,
        padding=settings.roi_padding,
    )
    common = {
        "evidence_store": evidence_store,
        "learned": learned,
        "learning_policy": settings.learning_policy,
        "consensus_policy": settings.consensus_policy,
        "detection_region": detection_region,
        "dry_run": settings.dry_run,
    }
    if settings.mode == "door":
        return DoorEventProcessor(
            engine,
            tracker_config.door_zone,
            outbox,
            camera_id=settings.camera_id,
            room_id=settings.room_id,
            min_door_frames=settings.min_door_frames,
            **common,
        )
    return VisionProcessor(
        engine,
        EntryTracker(tracker_config),
        outbox,
        decision_wait=timedelta(seconds=settings.decision_wait_seconds),
        **common,
    )


def _report_delivery(sent: int, retrying: int, dead: int) -> None:
    if sent:
        print(f"{sent} evento(s) entregue(s) à API.")
    if retrying:
        print(f"{retrying} evento(s) mantido(s) para reenvio.")
    if dead:
        print(f"{dead} evento(s) movido(s) para revisão manual.")


def _report_events(events: list[dict[str, Any]]) -> None:
    for payload in events:
        print(
            f"Evento visual enfileirado: {payload['event_id']} "
            f"({payload['identity_status']})."
        )


def run_forever(settings: VisionWorkerSettings) -> None:
    processor = create_processor(settings)
    services = _WorkerServices(processor, settings)
    services.start()
    cv2 = processor.engine.cv2
    rtsp_url = build_rtsp_url(settings.camera)
    sample_interval = 1.0 / settings.sample_fps
    last_processed = 0.0
    print(
        f"Worker iniciado para {settings.camera_id}/{settings.room_id}; "
        f"câmera {settings.camera.display_url}; modo {settings.mode}."
    )
    if settings.dry_run:
        print("Dry-run ativo: eventos ficam na outbox isolada de calibração.")
    try:
        while True:
            services.ensure_healthy()
            capture = cv2.VideoCapture()
            timeout_ms = round(settings.camera.timeout_seconds * 1000)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
            try:
                opened = capture.open(rtsp_url, getattr(cv2, "CAP_FFMPEG", 0))
            except Exception:
                opened = False
            if not opened:
                print("Stream indisponível; nova tentativa em 3 segundos.")
                capture.release()
                time.sleep(3)
                continue
            print("Stream RTSP conectado.")
            try:
                while True:
                    services.ensure_healthy()
                    try:
                        received, frame = capture.read()
                    except Exception:
                        received, frame = False, None
                    if not received or frame is None:
                        print("Stream interrompido; reiniciando o estado de rastreamento.")
                        _report_events(processor.flush(datetime.now(UTC)))
                        services.wake()
                        break
                    monotonic_now = time.monotonic()
                    if monotonic_now - last_processed < sample_interval:
                        continue
                    last_processed = monotonic_now
                    events = processor.process_frame(frame, datetime.now(UTC))
                    _report_events(events)
                    if events:
                        services.wake()
            finally:
                capture.release()
            time.sleep(1)
    finally:
        try:
            _report_events(processor.flush(datetime.now(UTC)))
            services.wake()
        finally:
            services.close(drain=not settings.dry_run)


__all__ = [
    "AuditApiClient",
    "DoorEventProcessor",
    "VisionConfigurationError",
    "VisionProcessor",
    "VisionWorkerSettings",
    "build_access_event",
    "build_door_event",
    "create_processor",
    "flush_outbox",
    "load_entry_tracker_config",
    "run_forever",
    "safe_person_id",
]
