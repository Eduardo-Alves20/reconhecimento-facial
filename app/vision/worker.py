"""Worker RTSP: detecta rosto, confirma entrada e envia evento ao RAG-Audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
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
from .learned import LearnedGallery
from .outbox import VisionEventOutbox
from .recognizer import ArcFaceEngine, MatchDecision, MatchStatus, MatchThresholds


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
    evidence_dir: Path
    sample_fps: float = 4.0
    dry_run: bool = False
    mode: str = "door"
    min_door_frames: int = 3
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
            evidence_dir=Path(
                os.getenv(
                    "RAG_AUDIT_VISION_EVIDENCE_DIR",
                    str(project_root / "data" / "private" / "evidence"),
                )
            ),
            sample_fps=_float_env(
                "RAG_AUDIT_VISION_SAMPLE_FPS", 4.0, minimum=0.5, maximum=15.0
            ),
            dry_run=_bool_env("RAG_AUDIT_VISION_DRY_RUN", False),
            mode=(os.getenv("RAG_AUDIT_VISION_MODE", "door").strip().lower() or "door"),
            min_door_frames=int(
                _float_env("RAG_AUDIT_VISION_MIN_DOOR_FRAMES", 3, minimum=1, maximum=30)
            ),
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


def _crop_face(frame: Any, face: Any, *, margin: float) -> Any | None:
    """Recorta o rosto com margem proporcional; ``None`` se inviável."""

    try:
        height, width = frame.shape[:2]
        x, y, box_width, box_height = (int(value) for value in face.bbox)
        margin_x = int(box_width * margin)
        margin_y = int(box_height * margin)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + box_width + margin_x)
        y2 = min(height, y + box_height + margin_y)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None
    except Exception:
        return None


def _write_jpeg(cv2: Any, image: Any, destination: Path, quality: int) -> bool:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(buffer.tobytes())
        os.replace(temporary, destination)
    return True


def save_evidence(cv2: Any, frame: Any, face: Any, evidence_dir: Path) -> str | None:
    """Guarda a cena completa e um recorte do rosto em armazenamento privado.

    Devolve o SHA-256 da cena completa como referência opaca; a imagem nunca
    trafega pelo webhook. O recorte fica em ``<sha>.thumb.jpg`` para a lista.
    """

    try:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return None
        data = buffer.tobytes()
        digest = hashlib.sha256(data).hexdigest()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        full_path = evidence_dir / f"{digest}.jpg"
        if not full_path.exists():
            temporary = full_path.with_suffix(".jpg.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, full_path)
        crop = _crop_face(frame, face, margin=0.6)
        if crop is not None:
            _write_jpeg(cv2, crop, evidence_dir / f"{digest}.thumb.jpg", 88)
        return digest
    except Exception:
        return None


def build_door_event(
    *,
    session_id: str,
    track_id: str,
    camera_id: str,
    room_id: str,
    consensus: ConsensusResult,
    model_version: str,
    evidence_ref: str | None,
    confirmed_at: datetime,
    observation_count: int,
) -> dict[str, Any]:
    """Evento de 'rosto reconhecido na porta' (sem prova de direção)."""

    if consensus.status == MatchStatus.MATCHED and consensus.external_id:
        user_id = safe_person_id(consensus.external_id)
        identity_status = "MATCHED"
        confidence = max(0.0, min(1.0, consensus.similarity or 0.0))
    elif consensus.status == MatchStatus.AMBIGUOUS:
        user_id = f"AMBIGUOUS:{session_id}"
        identity_status = "AMBIGUOUS"
        confidence = None
    else:
        user_id = f"UNKNOWN:{session_id}"
        identity_status = "UNKNOWN"
        confidence = None
    payload: dict[str, Any] = {
        "event_id": f"entry:{session_id}",
        "camera_id": camera_id,
        "user_id": user_id,
        "room_id": room_id,
        "timestamp": confirmed_at.isoformat(),
        "door_result": "NOT_REPORTED",
        "recognition_confidence": confidence,
        "identity_status": identity_status,
        "entry_evidence": "VISION_FACE_AT_DOOR",
        "recognition_source": "LOCAL_SFACE",
        "track_id": track_id,
        "recognition_model": model_version,
        "recognition_margin": consensus.margin,
        "face_quality": consensus.face_quality,
        "entry_confidence": min(1.0, observation_count / 6.0),
    }
    if evidence_ref:
        payload["evidence_ref"] = evidence_ref
    return payload


class DoorEventProcessor:
    """Registra o acesso quando um rosto persiste na zona da porta.

    Não tenta provar direção (entrar/sair). Como a porta tem fechadura, quem é
    reconhecido ali é quem está acessando. Guarda a melhor foto do rosto e
    enfileira o evento uma única vez por presença.
    """

    def __init__(
        self,
        engine: "RecognitionEngine",
        door_zone: Polygon,
        outbox: VisionEventOutbox,
        *,
        camera_id: str,
        room_id: str,
        evidence_dir: Path,
        min_door_frames: int = 3,
        cooldown: timedelta = timedelta(seconds=15),
        learned: LearnedGallery | None = None,
        learn_min_similarity: float = 0.50,
        learn_max_similarity: float = 0.85,
        learn_min_quality: float = 0.50,
    ) -> None:
        self.engine = engine
        self.door_zone = door_zone
        self.outbox = outbox
        self.camera_id = camera_id
        self.room_id = room_id
        self.evidence_dir = evidence_dir
        self.min_door_frames = min_door_frames
        self.cooldown = cooldown
        # Auto-aprendizado: aprende novas referências das pessoas reconhecidas.
        self.learned = learned
        self.learn_min_similarity = learn_min_similarity
        self.learn_max_similarity = learn_max_similarity
        self.learn_min_quality = learn_min_quality
        self.cv2 = engine.cv2  # type: ignore[attr-defined]
        # Mantém a mesma trilha viva por alguns segundos mesmo se o rosto some
        # por um instante (a pessoa vira a cabeça), para acumular vários ângulos
        # numa passagem só. Histórico maior guarda mais ângulos para a decisão.
        self.face_tracker = FaceCentroidTracker(max_age=timedelta(seconds=4))
        # ArcFace separa bem (errado ~0,00), então 2 quadros já bastam — ajuda a
        # reconhecer quem aparece por poucos quadros (a pessoa de trás).
        self.consensus = IdentityConsensus(history_size=30, minimum_matches=2)
        self._known: set[str] = set()
        self._door_hits: dict[str, int] = {}
        self._best: dict[str, tuple[float, Any, Any]] = {}
        self._last_seen: dict[str, datetime] = {}
        # Trilhas que já registraram alguém reconhecido (para não emitir também
        # um 'desconhecido' no fim). Uma trilha pode registrar mais de uma pessoa.
        self._track_matched: set[str] = set()
        # Presença: enquanto a pessoa continua aparecendo não registra de novo;
        # só reconta após ausência (saiu e voltou = nova entrada).
        self._present_until: dict[str, datetime] = {}
        self._absence_gap = timedelta(seconds=5)
        # Movimento por trilha: só conta quem CHEGOU (se moveu na porta), não quem
        # está parado no enquadramento (ex.: pessoa sentada na direção da porta).
        self._track_origin: dict[str, tuple[float, float]] = {}
        self._track_moved: set[str] = set()
        # Desconhecidos recentes (instante, embedding) para deduplicar por rosto.
        self._recent_unknowns: list[tuple[datetime, Any]] = []
        self._last_log: datetime | None = None

    def process_frame(self, frame: Any, observed_at: datetime) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        faces = self.engine.detect(frame)
        frame_log: list[str] = []
        for tracked in self.face_tracker.update(faces, observed_at):
            track_id = tracked.track_id
            self._known.add(track_id)
            decision = self.engine.match(tracked.face)
            # Se outra pessoa confiável assumiu a mesma trilha (carona: o de trás
            # ocupa o lugar do da frente), zera o consenso para ela não ser
            # absorvida pela identidade anterior.
            if decision.status == MatchStatus.MATCHED and decision.external_id:
                current = self.consensus.resolve(track_id)
                if (
                    current.status == MatchStatus.MATCHED
                    and current.external_id
                    and current.external_id != decision.external_id
                ):
                    self.consensus.discard(track_id)
                    self._track_matched.discard(track_id)
            self.consensus.add(track_id, decision)
            cx, cy = tracked.face.centroid
            in_door = self.door_zone.contains(Point(cx, cy))
            who = decision.display_name or decision.status.value
            frame_log.append(
                f"({cx:.2f},{cy:.2f}){'PORTA' if in_door else 'fora'}:"
                f"{who[:16]} q{tracked.face.quality:.2f}"
            )
            if not in_door:
                continue
            self._door_hits[track_id] = self._door_hits.get(track_id, 0) + 1
            self._last_seen[track_id] = observed_at
            # Marca a trilha como "chegou" quando o rosto se desloca na porta;
            # quem fica parado (sentado) nunca é marcado e não vira entrada.
            origin = self._track_origin.setdefault(track_id, (cx, cy))
            if track_id not in self._track_moved and (
                abs(cx - origin[0]) + abs(cy - origin[1]) > 0.05
            ):
                self._track_moved.add(track_id)
            quality = tracked.face.quality
            if track_id not in self._best or quality > self._best[track_id][0]:
                # Guarda o melhor quadro na memória; grava só ao decidir.
                self._best[track_id] = (quality, frame.copy(), tracked.face)
            if self._door_hits[track_id] < self.min_door_frames:
                continue
            result = self.consensus.resolve(track_id)
            if result.status == MatchStatus.MATCHED and result.external_id:
                previous = self._present_until.get(result.external_id)
                still_present = previous is not None and observed_at < previous
                if still_present:
                    # Já registrada nesta visita: só mantém a presença viva
                    # (para não recontar enquanto a pessoa continua na sala).
                    self._present_until[result.external_id] = (
                        observed_at + self._absence_gap
                    )
                elif track_id in self._track_moved:
                    # Chegou (se moveu na porta) e não está presente: registra
                    # e só ENTÃO marca presença. Marcar antes de registrar
                    # bloqueava a própria entrada.
                    payload = self._emit_decision(track_id, observed_at, result)
                    if payload is not None:
                        self._present_until[result.external_id] = (
                            observed_at + self._absence_gap
                        )
                        self._track_matched.add(track_id)
                        emitted.append(payload)
        # Log throttled do que a câmera vê, para diagnóstico ao vivo.
        if faces and (
            self._last_log is None
            or (observed_at - self._last_log).total_seconds() >= 1.5
        ):
            self._last_log = observed_at
            print(f"[visao] {len(faces)} rosto(s): {frame_log}", flush=True)
        emitted.extend(self._flush_expired(observed_at))
        return emitted

    def _same_recent_unknown(self, feature: Any, observed_at: datetime) -> bool:
        """Verdadeiro se este rosto já apareceu como desconhecido há pouco.

        Deduplica 'desconhecido' pela aparência (não pelo tempo), então dois
        desconhecidos DIFERENTES entrando juntos geram dois registros, mas a
        mesma pessoa perdida e readquirida gera um só.
        """

        self._recent_unknowns = [
            (moment, embedding)
            for moment, embedding in self._recent_unknowns
            if observed_at < moment + self.cooldown
        ]
        for _, embedding in self._recent_unknowns:
            if self.engine.similarity(feature, embedding) >= 0.45:
                return True
        return False

    def _emit_decision(
        self, track_id: str, when: datetime, result: ConsensusResult
    ) -> dict[str, Any] | None:
        best = self._best.get(track_id)
        feature = best[2].feature if best is not None else None
        matched = result.status == MatchStatus.MATCHED and bool(result.external_id)
        if not matched and feature is not None and self._same_recent_unknown(feature, when):
            return None
        session_id = f"vis-{uuid.uuid4().hex}"
        evidence_ref = (
            save_evidence(self.cv2, best[1], best[2], self.evidence_dir)
            if best is not None
            else None
        )
        payload = build_door_event(
            session_id=session_id,
            track_id=track_id,
            camera_id=self.camera_id,
            room_id=self.room_id,
            consensus=result,
            model_version=self.engine.model_version,
            evidence_ref=evidence_ref,
            confirmed_at=when,
            observation_count=self._door_hits.get(track_id, 0),
        )
        self.outbox.enqueue(payload, now=when)
        if matched and self.learned is not None and feature is not None:
            if self.learned.consider(
                self.engine,
                external_id=result.external_id,
                display_name=result.display_name,
                feature=feature,
                evidence_ref=evidence_ref,
                when=when,
                similarity=result.similarity,
                quality=result.face_quality,
                min_similarity=self.learn_min_similarity,
                max_similarity=self.learn_max_similarity,
                min_quality=self.learn_min_quality,
            ):
                print(
                    f"[aprendizado] nova referência de {result.display_name} "
                    f"(sim={result.similarity:.2f}).",
                    flush=True,
                )
        elif not matched and feature is not None:
            self._recent_unknowns.append((when, feature))
        return payload

    def _flush_expired(self, observed_at: datetime) -> list[dict[str, Any]]:
        """Finaliza trilhas que sumiram (a pessoa saiu da porta).

        Quem nunca foi reconhecido com certeza durante a passagem — e ficou
        tempo suficiente na porta — é registrado como desconhecido só agora.
        """

        active = set(self.face_tracker.active_track_ids)
        emitted: list[dict[str, Any]] = []
        for track_id in list(self._known - active):
            if (
                track_id not in self._track_matched
                and track_id in self._track_moved
                and self._door_hits.get(track_id, 0) >= self.min_door_frames
            ):
                result = self.consensus.resolve(track_id)
                when = self._last_seen.get(track_id, observed_at)
                payload = self._emit_decision(track_id, when, result)
                if payload is not None:
                    emitted.append(payload)
            self._door_hits.pop(track_id, None)
            self._best.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            self._track_matched.discard(track_id)
            self._track_origin.pop(track_id, None)
            self._track_moved.discard(track_id)
            self.consensus.discard(track_id)
            self._known.discard(track_id)
        for person_key in [
            key for key, until in self._present_until.items() if observed_at >= until
        ]:
            self._present_until.pop(person_key, None)
        return emitted


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


def create_processor(
    settings: VisionWorkerSettings,
) -> VisionProcessor | DoorEventProcessor:
    tracker_config = load_entry_tracker_config(
        settings.calibration_path,
        camera_id=settings.camera_id,
        room_id=settings.room_id,
    )
    engine = ArcFaceEngine(thresholds=settings.thresholds)
    enrolled = engine.load_gallery(settings.gallery_manifest)
    if enrolled < 1:
        raise VisionConfigurationError("A galeria facial está vazia.")
    learned: LearnedGallery | None = None
    if settings.mode == "door" and _bool_env("RAG_AUDIT_LEARN_ENABLED", True):
        learned = LearnedGallery(
            Path(settings.gallery_manifest).parent / "learned.db",
            max_per_person=int(
                _float_env("RAG_AUDIT_LEARN_MAX_PER_PERSON", 5, minimum=1, maximum=50)
            ),
        )
        learned.initialize()
        learned_count = learned.load_into(engine)
        if learned_count:
            print(f"{learned_count} referência(s) aprendida(s) carregada(s).")
    outbox = VisionEventOutbox(settings.outbox_path)
    outbox.initialize()
    if settings.mode == "door":
        return DoorEventProcessor(
            engine,
            tracker_config.door_zone,
            outbox,
            camera_id=settings.camera_id,
            room_id=settings.room_id,
            evidence_dir=settings.evidence_dir,
            min_door_frames=settings.min_door_frames,
            cooldown=tracker_config.cooldown,
            learned=learned,
            learn_min_similarity=_float_env(
                "RAG_AUDIT_LEARN_MIN_SIMILARITY", 0.50, minimum=0, maximum=1
            ),
            learn_max_similarity=_float_env(
                "RAG_AUDIT_LEARN_MAX_SIMILARITY", 0.85, minimum=0, maximum=1
            ),
            learn_min_quality=_float_env(
                "RAG_AUDIT_LEARN_MIN_QUALITY", 0.50, minimum=0, maximum=1
            ),
        )
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
