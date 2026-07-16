from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Protocol

from .entry_tracker import EntryConfirmation, EntryTracker, Point, Polygon, TrackObservation
from .evidence import EvidenceStore, EvidenceStoreError
from .face_tracking import ConsensusResult, FaceCentroidTracker, IdentityConsensus, TrackedFace
from .identity import safe_person_id
from .learned import LearnedGallery
from .outbox import VisionEventOutbox
from .recognizer import DetectedFace, MatchDecision, MatchStatus


class RecognitionEngine(Protocol):
    model_version: str
    model_fingerprint: str

    def detect(self, frame: Any) -> list[DetectedFace]: ...

    def match(self, face: DetectedFace) -> MatchDecision: ...


@dataclass(frozen=True, slots=True)
class ConsensusPolicy:
    history_size: int = 12
    minimum_matches: int = 4
    minimum_ratio: float = 0.70
    minimum_consecutive: int = 3

    def __post_init__(self) -> None:
        if (
            self.history_size < 1
            or self.minimum_matches < 1
            or self.minimum_matches > self.history_size
            or self.minimum_consecutive < 1
            or self.minimum_consecutive > self.minimum_matches
            or not math.isfinite(self.minimum_ratio)
            or not 0 < self.minimum_ratio <= 1
        ):
            raise ValueError("A política de consenso é inválida.")


@dataclass(frozen=True, slots=True)
class LearningPolicy:
    enabled: bool = False
    allow_dry_run: bool = False
    min_similarity: float = 0.65
    max_similarity: float = 0.95
    min_margin: float = 0.12
    min_quality: float = 0.60
    min_supporting_frames: int = 4

    def __post_init__(self) -> None:
        values = (
            self.min_similarity,
            self.max_similarity,
            self.min_margin,
            self.min_quality,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Os limites de aprendizado devem estar entre 0 e 1.")
        if self.min_similarity > self.max_similarity:
            raise ValueError("min_similarity não pode superar max_similarity.")
        if self.min_supporting_frames < 1:
            raise ValueError("min_supporting_frames deve ser positivo.")


@dataclass(frozen=True, slots=True)
class DetectionRegion:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_polygons(
        cls,
        *polygons: Polygon,
        padding: float = 0.05,
    ) -> "DetectionRegion":
        if not math.isfinite(padding) or not 0 <= padding <= 0.5:
            raise ValueError("padding da ROI deve estar entre 0 e 0,5.")
        points = [point for polygon in polygons for point in polygon.points]
        if not points:
            raise ValueError("A ROI precisa de ao menos um polígono.")
        return cls(
            max(0.0, min(point.x for point in points) - padding),
            max(0.0, min(point.y for point in points) - padding),
            min(1.0, max(point.x for point in points) + padding),
            min(1.0, max(point.y for point in points) + padding),
        )

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("A ROI deve usar coordenadas normalizadas.")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("A ROI possui área inválida.")


@dataclass(slots=True)
class _BestObservation:
    quality: float
    captured_at: datetime
    frame: Any
    face: DetectedFace
    decision: MatchDecision


@dataclass(frozen=True, slots=True)
class _PendingEntry:
    confirmation: EntryConfirmation
    deadline: datetime
    candidate_external_id: str | None
    best_at_confirmation: _BestObservation | None


def _identity_fields(
    session_id: str,
    consensus: ConsensusResult,
) -> tuple[str, str, float | None]:
    if consensus.status == MatchStatus.MATCHED and consensus.external_id:
        return (
            safe_person_id(consensus.external_id),
            "MATCHED",
            max(0.0, min(1.0, consensus.similarity or 0.0)),
        )
    if consensus.status == MatchStatus.AMBIGUOUS:
        return f"AMBIGUOUS:{session_id}", "AMBIGUOUS", None
    return f"UNKNOWN:{session_id}", "UNKNOWN", None


def build_access_event(
    confirmation: EntryConfirmation,
    consensus: ConsensusResult,
    *,
    model_version: str,
    model_fingerprint: str | None = None,
    evidence_ref: str | None = None,
    evidence_captured_at: datetime | None = None,
) -> dict[str, Any]:
    user_id, identity_status, confidence = _identity_fields(
        confirmation.session_id,
        consensus,
    )
    payload: dict[str, Any] = {
        "event_id": f"entry:{confirmation.session_id}",
        "camera_id": confirmation.camera_id,
        "user_id": user_id,
        "room_id": confirmation.room_id,
        "timestamp": confirmation.confirmed_at.isoformat(),
        "door_result": "NOT_REPORTED",
        "recognition_confidence": confidence,
        "identity_status": identity_status,
        "entry_evidence": "VISION_LINE_CROSSING",
        "recognition_source": "LOCAL_ARCFACE",
        "track_id": confirmation.track_id,
        "recognition_model": model_version,
        "recognition_margin": consensus.margin,
        "face_quality": consensus.face_quality,
        "entry_confidence": min(1.0, confirmation.observation_count / 8.0),
    }
    if model_fingerprint:
        payload["recognition_model_fingerprint"] = model_fingerprint
    if evidence_ref:
        payload["evidence_ref"] = evidence_ref
        payload["evidence_captured_at"] = (
            evidence_captured_at or confirmation.confirmed_at
        ).isoformat()
    return payload


def build_door_event(
    *,
    session_id: str,
    track_id: str,
    camera_id: str,
    room_id: str,
    consensus: ConsensusResult,
    model_version: str,
    evidence_ref: str | None,
    evidence_captured_at: datetime | None,
    confirmed_at: datetime,
    observation_count: int,
    model_fingerprint: str | None = None,
) -> dict[str, Any]:
    user_id, identity_status, confidence = _identity_fields(session_id, consensus)
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
        "recognition_source": "LOCAL_ARCFACE",
        "track_id": track_id,
        "recognition_model": model_version,
        "recognition_margin": consensus.margin,
        "face_quality": consensus.face_quality,
        "entry_confidence": min(1.0, observation_count / 8.0),
    }
    if model_fingerprint:
        payload["recognition_model_fingerprint"] = model_fingerprint
    if evidence_ref:
        payload["evidence_ref"] = evidence_ref
        payload["evidence_captured_at"] = (
            evidence_captured_at or confirmed_at
        ).isoformat()
    return payload


class _VisualProcessor:
    def __init__(
        self,
        engine: RecognitionEngine,
        outbox: VisionEventOutbox,
        *,
        camera_id: str,
        room_id: str,
        evidence_store: EvidenceStore | None = None,
        learned: LearnedGallery | None = None,
        learning_policy: LearningPolicy | None = None,
        consensus_policy: ConsensusPolicy | None = None,
        detection_region: DetectionRegion | None = None,
        dry_run: bool = False,
        tracker_max_age: timedelta = timedelta(seconds=3),
        presence_gap: timedelta = timedelta(seconds=5),
    ) -> None:
        if presence_gap <= timedelta(0):
            raise ValueError("As janelas temporais do processador são inválidas.")
        policy = consensus_policy or ConsensusPolicy()
        self.engine = engine
        self.outbox = outbox
        self.camera_id = camera_id
        self.room_id = room_id
        self.evidence_store = evidence_store
        self.learned = learned
        self.learning_policy = learning_policy or LearningPolicy()
        self.detection_region = detection_region
        self.dry_run = dry_run
        self.presence_gap = presence_gap
        self.face_tracker = FaceCentroidTracker(max_age=tracker_max_age)
        self.consensus = IdentityConsensus(
            history_size=policy.history_size,
            minimum_matches=policy.minimum_matches,
            minimum_ratio=policy.minimum_ratio,
            minimum_consecutive=policy.minimum_consecutive,
        )
        self._known_tracks: set[str] = set()
        self._best_any: dict[str, _BestObservation] = {}
        self._best_identity: dict[tuple[str, str], _BestObservation] = {}
        self._present_until: dict[str, datetime] = {}

    def _detect(self, frame: Any) -> list[DetectedFace]:
        region = self.detection_region
        if region is None:
            return self.engine.detect(frame)
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame inválido")
        height, width = (int(value) for value in frame.shape[:2])
        left = max(0, min(width - 1, math.floor(region.left * width)))
        top = max(0, min(height - 1, math.floor(region.top * height)))
        right = max(left + 1, min(width, math.ceil(region.right * width)))
        bottom = max(top + 1, min(height, math.ceil(region.bottom * height)))
        crop = frame[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        result: list[DetectedFace] = []
        for face in self.engine.detect(crop):
            x, y, box_width, box_height = face.bbox
            center_x = left + face.centroid[0] * crop_width
            center_y = top + face.centroid[1] * crop_height
            result.append(
                replace(
                    face,
                    bbox=(left + x, top + y, box_width, box_height),
                    centroid=(
                        max(0.0, min(1.0, center_x / width)),
                        max(0.0, min(1.0, center_y / height)),
                    ),
                )
            )
        return result

    def _track(
        self,
        frame: Any,
        observed_at: datetime,
    ) -> tuple[list[tuple[TrackedFace, MatchDecision]], set[str]]:
        tracked_faces = self.face_tracker.update(self._detect(frame), observed_at)
        active = set(self.face_tracker.active_track_ids)
        expired = self._known_tracks - active
        observations: list[tuple[TrackedFace, MatchDecision]] = []
        for tracked in tracked_faces:
            track_id = tracked.track_id
            decision = self.engine.match(tracked.face)
            current = self.consensus.resolve(track_id)
            if (
                current.status == MatchStatus.MATCHED
                and current.external_id
                and decision.status == MatchStatus.MATCHED
                and decision.external_id
                and current.external_id != decision.external_id
            ):
                self._discard_track(track_id, discard_tracker=True)
                continue
            self._known_tracks.add(track_id)
            self.consensus.add(track_id, decision)
            self._remember(
                frame,
                tracked.face,
                track_id,
                decision,
                observed_at,
            )
            observations.append((tracked, decision))
        return observations, expired

    def _remember(
        self,
        frame: Any,
        face: DetectedFace,
        track_id: str,
        decision: MatchDecision,
        observed_at: datetime,
    ) -> None:
        current = self._best_any.get(track_id)
        identity_key = (
            (track_id, decision.external_id)
            if decision.status == MatchStatus.MATCHED and decision.external_id
            else None
        )
        identity_current = (
            self._best_identity.get(identity_key)
            if identity_key is not None
            else None
        )
        needs_any = (
            current is None
            or observed_at - current.captured_at > timedelta(seconds=5)
            or face.quality > current.quality
        )
        needs_identity = (
            identity_key is not None
            and (
                identity_current is None
                or observed_at - identity_current.captured_at
                > timedelta(seconds=5)
                or face.quality > identity_current.quality
            )
        )
        if not needs_any and not needs_identity:
            return
        saved_frame, saved_face = self._evidence_view(frame, face)
        observation = _BestObservation(
            face.quality,
            observed_at,
            saved_frame,
            saved_face,
            decision,
        )
        if needs_any:
            self._best_any[track_id] = observation
        if needs_identity and identity_key is not None:
            self._best_identity[identity_key] = observation

    def _evidence_view(
        self,
        frame: Any,
        face: DetectedFace,
    ) -> tuple[Any, DetectedFace]:
        region = self.detection_region
        if (
            region is None
            or frame is None
            or not hasattr(frame, "shape")
            or len(frame.shape) < 2
        ):
            return (frame.copy() if hasattr(frame, "copy") else frame), face
        height, width = (int(value) for value in frame.shape[:2])
        left = max(0, min(width - 1, math.floor(region.left * width)))
        top = max(0, min(height - 1, math.floor(region.top * height)))
        right = max(left + 1, min(width, math.ceil(region.right * width)))
        bottom = max(top + 1, min(height, math.ceil(region.bottom * height)))
        cropped = frame[top:bottom, left:right].copy()
        x, y, box_width, box_height = face.bbox
        return cropped, replace(
            face,
            bbox=(x - left, y - top, box_width, box_height),
        )

    def _best_for(
        self,
        track_id: str,
        result: ConsensusResult,
    ) -> _BestObservation | None:
        if result.status == MatchStatus.MATCHED and result.external_id:
            return self._best_identity.get((track_id, result.external_id))
        return self._best_any.get(track_id)

    def _save_evidence(self, best: _BestObservation | None) -> str | None:
        if best is None or self.evidence_store is None:
            return None
        cv2 = getattr(self.engine, "cv2", None)
        if cv2 is None:
            return None
        try:
            ok, scene = cv2.imencode(
                ".jpg",
                best.frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            if not ok:
                return None
            thumbnail = self._face_thumbnail(cv2, best)
            record = self.evidence_store.save(
                scene.tobytes(),
                thumbnail=thumbnail,
                created_at=best.captured_at,
            )
            return record.reference
        except (EvidenceStoreError, OSError, TypeError, ValueError):
            return None
        except Exception:
            return None

    @staticmethod
    def _face_thumbnail(cv2: Any, best: _BestObservation) -> bytes | None:
        height, width = best.frame.shape[:2]
        x, y, box_width, box_height = best.face.bbox
        margin_x = round(box_width * 0.6)
        margin_y = round(box_height * 0.6)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(width, x + box_width + margin_x)
        bottom = min(height, y + box_height + margin_y)
        crop = best.frame[top:bottom, left:right]
        if getattr(crop, "size", 0) == 0:
            return None
        ok, encoded = cv2.imencode(
            ".jpg",
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        return encoded.tobytes() if ok else None

    def _stage_learning(
        self,
        *,
        track_id: str,
        result: ConsensusResult,
        best: _BestObservation | None,
        evidence_ref: str | None,
        event_id: str,
        when: datetime,
    ) -> bool:
        policy = self.learning_policy
        decision = best.decision if best is not None else None
        if (
            not policy.enabled
            or self.learned is None
            or best is None
            or evidence_ref is None
            or (self.dry_run and not policy.allow_dry_run)
            or result.status != MatchStatus.MATCHED
            or not result.external_id
            or not result.display_name
            or result.supporting_frames < policy.min_supporting_frames
            or decision is None
            or decision.status != MatchStatus.MATCHED
            or decision.external_id != result.external_id
            or decision.similarity is None
            or decision.margin is None
            or decision.margin < policy.min_margin
        ):
            return False
        try:
            return self.learned.consider(
                self.engine,
                external_id=result.external_id,
                display_name=result.display_name,
                feature=best.face.feature,
                evidence_ref=evidence_ref,
                when=best.captured_at,
                similarity=decision.similarity,
                quality=best.face.quality,
                min_similarity=policy.min_similarity,
                max_similarity=policy.max_similarity,
                min_quality=policy.min_quality,
                provenance={
                    "camera_id": self.camera_id,
                    "room_id": self.room_id,
                    "track_id": track_id,
                    "event_id": event_id,
                    "event_timestamp": when.isoformat(),
                    "evidence_captured_at": best.captured_at.isoformat(),
                    "supporting_frames": result.supporting_frames,
                },
            )
        except Exception:
            return False

    def _purge_presence(self, observed_at: datetime) -> None:
        for external_id in [
            key for key, until in self._present_until.items() if observed_at >= until
        ]:
            self._present_until.pop(external_id, None)

    def _clear_best(self, track_id: str) -> None:
        self._best_any.pop(track_id, None)
        for key in [key for key in self._best_identity if key[0] == track_id]:
            self._best_identity.pop(key, None)

    def _discard_track(self, track_id: str, *, discard_tracker: bool = False) -> None:
        self.consensus.discard(track_id)
        self._clear_best(track_id)
        self._known_tracks.discard(track_id)
        if discard_tracker:
            self.face_tracker.discard(track_id)
        self._discard_mode_state(track_id)

    def _discard_mode_state(self, track_id: str) -> None:
        del track_id

    def _reset(self) -> None:
        self.face_tracker.clear()
        self.consensus.clear()
        self._known_tracks.clear()
        self._best_any.clear()
        self._best_identity.clear()


class VisionProcessor(_VisualProcessor):
    def __init__(
        self,
        engine: RecognitionEngine,
        entry_tracker: EntryTracker,
        outbox: VisionEventOutbox,
        *,
        evidence_store: EvidenceStore | None = None,
        learned: LearnedGallery | None = None,
        learning_policy: LearningPolicy | None = None,
        consensus_policy: ConsensusPolicy | None = None,
        detection_region: DetectionRegion | None = None,
        dry_run: bool = False,
        decision_wait: timedelta = timedelta(seconds=1.5),
    ) -> None:
        if decision_wait < timedelta(0):
            raise ValueError("decision_wait não pode ser negativo.")
        super().__init__(
            engine,
            outbox,
            camera_id=entry_tracker.config.camera_id,
            room_id=entry_tracker.config.room_id,
            evidence_store=evidence_store,
            learned=learned,
            learning_policy=learning_policy,
            consensus_policy=consensus_policy,
            detection_region=detection_region,
            dry_run=dry_run,
        )
        self.entry_tracker = entry_tracker
        self.decision_wait = decision_wait
        self._pending: dict[str, _PendingEntry] = {}
        self._handled_tracks: set[str] = set()
        self._handled_identity: dict[str, str] = {}
        self._last_raw_identity: dict[str, str] = {}
        self._observation_sequence = 0

    def process_frame(self, frame: Any, observed_at: datetime) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        observations, expired = self._track(frame, observed_at)
        self.entry_tracker.advance_time(observed_at)
        self._purge_presence(observed_at)
        for tracked, decision in observations:
            track_id = tracked.track_id
            if decision.status == MatchStatus.MATCHED and decision.external_id:
                previous_identity = self._last_raw_identity.get(track_id)
                if (
                    previous_identity is not None
                    and previous_identity != decision.external_id
                    and track_id not in self._pending
                ):
                    self._discard_track(track_id, discard_tracker=True)
                    continue
                self._last_raw_identity[track_id] = decision.external_id
            handled_identity = self._handled_identity.get(track_id)
            if handled_identity:
                self._present_until[handled_identity] = observed_at + self.presence_gap
            self._observation_sequence += 1
            confirmation = self.entry_tracker.observe(
                TrackObservation(
                    track_id=track_id,
                    observed_at=observed_at,
                    centroid=Point(*tracked.face.centroid),
                    face_observation_id=(
                        f"obs:{self.face_tracker.boot_id}:{self._observation_sequence}"
                    ),
                )
            )
            if confirmation is not None and track_id not in self._handled_tracks:
                candidates = {
                    external_id
                    for known_track, external_id in self._best_identity
                    if known_track == track_id
                }
                candidate = next(iter(candidates)) if len(candidates) == 1 else None
                self._pending[track_id] = _PendingEntry(
                    confirmation,
                    observed_at + self.decision_wait,
                    candidate,
                    (
                        self._best_identity.get((track_id, candidate))
                        if candidate is not None
                        else self._best_any.get(track_id)
                    ),
                )

        for track_id in tuple(self._pending):
            result = self.consensus.resolve(track_id)
            force = track_id in expired or observed_at >= self._pending[track_id].deadline
            if result.status == MatchStatus.MATCHED or force:
                payload = self._finalize_pending(track_id, result, observed_at)
                if payload is not None:
                    emitted.append(payload)

        for track_id in expired:
            self._discard_track(track_id)
        return emitted

    def _finalize_pending(
        self,
        track_id: str,
        result: ConsensusResult,
        observed_at: datetime,
    ) -> dict[str, Any] | None:
        pending = self._pending.pop(track_id, None)
        if pending is None:
            return None
        confirmation = pending.confirmation
        if (
            result.status == MatchStatus.MATCHED
            and result.external_id != pending.candidate_external_id
        ):
            result = ConsensusResult(
                MatchStatus.UNKNOWN,
                None,
                None,
                None,
                None,
                result.face_quality,
                0,
            )
        best = (
            self._best_for(track_id, result)
            if result.status == MatchStatus.MATCHED
            else pending.best_at_confirmation
        )
        matched = result.status == MatchStatus.MATCHED and bool(result.external_id)
        if matched and result.external_id:
            present_until = self._present_until.get(result.external_id)
            if present_until is not None and observed_at < present_until:
                self._handled_tracks.add(track_id)
                self._handled_identity[track_id] = result.external_id
                self._present_until[result.external_id] = observed_at + self.presence_gap
                return None
        evidence_ref = self._save_evidence(best)
        payload = build_access_event(
            confirmation,
            result,
            model_version=self.engine.model_version,
            model_fingerprint=self.engine.model_fingerprint,
            evidence_ref=evidence_ref,
            evidence_captured_at=best.captured_at if best is not None else None,
        )
        self.outbox.enqueue(payload, now=confirmation.confirmed_at)
        self._handled_tracks.add(track_id)
        if matched and result.external_id:
            self._handled_identity[track_id] = result.external_id
            self._present_until[result.external_id] = observed_at + self.presence_gap
            self._stage_learning(
                track_id=track_id,
                result=result,
                best=best,
                evidence_ref=evidence_ref,
                event_id=payload["event_id"],
                when=confirmation.confirmed_at,
            )
        return payload

    def _discard_mode_state(self, track_id: str) -> None:
        self.entry_tracker.discard(track_id)
        self._pending.pop(track_id, None)
        self._handled_tracks.discard(track_id)
        self._handled_identity.pop(track_id, None)
        self._last_raw_identity.pop(track_id, None)

    def flush(self, observed_at: datetime) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        for track_id in tuple(self._pending):
            payload = self._finalize_pending(
                track_id,
                self.consensus.resolve(track_id),
                observed_at,
            )
            if payload is not None:
                emitted.append(payload)
        self.entry_tracker.reset()
        self._pending.clear()
        self._handled_tracks.clear()
        self._handled_identity.clear()
        self._last_raw_identity.clear()
        self._reset()
        return emitted


class DoorEventProcessor(_VisualProcessor):
    def __init__(
        self,
        engine: RecognitionEngine,
        door_zone: Polygon,
        outbox: VisionEventOutbox,
        *,
        camera_id: str,
        room_id: str,
        evidence_store: EvidenceStore | None = None,
        min_door_frames: int = 4,
        learned: LearnedGallery | None = None,
        learning_policy: LearningPolicy | None = None,
        consensus_policy: ConsensusPolicy | None = None,
        detection_region: DetectionRegion | None = None,
        dry_run: bool = False,
        minimum_movement: float = 0.04,
    ) -> None:
        if min_door_frames < 1:
            raise ValueError("min_door_frames deve ser positivo.")
        if not math.isfinite(minimum_movement) or not 0 <= minimum_movement <= 1:
            raise ValueError("minimum_movement deve estar entre 0 e 1.")
        super().__init__(
            engine,
            outbox,
            camera_id=camera_id,
            room_id=room_id,
            evidence_store=evidence_store,
            learned=learned,
            learning_policy=learning_policy,
            consensus_policy=consensus_policy,
            detection_region=detection_region,
            dry_run=dry_run,
            tracker_max_age=timedelta(seconds=4),
        )
        self.door_zone = door_zone
        self.min_door_frames = min_door_frames
        self.minimum_movement = minimum_movement
        self._door_hits: dict[str, int] = {}
        self._origin: dict[str, Point] = {}
        self._moved: set[str] = set()
        self._last_seen: dict[str, datetime] = {}
        self._handled_tracks: set[str] = set()
        self._last_raw_identity: dict[str, str] = {}

    def process_frame(self, frame: Any, observed_at: datetime) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        observations, expired = self._track(frame, observed_at)
        self._purge_presence(observed_at)
        seen_tracks: set[str] = set()
        for tracked, decision in observations:
            track_id = tracked.track_id
            if decision.status == MatchStatus.MATCHED and decision.external_id:
                previous_identity = self._last_raw_identity.get(track_id)
                if (
                    previous_identity is not None
                    and previous_identity != decision.external_id
                ):
                    self._discard_track(track_id, discard_tracker=True)
                    continue
                self._last_raw_identity[track_id] = decision.external_id
            seen_tracks.add(track_id)
            point = Point(*tracked.face.centroid)
            if not self.door_zone.contains(point):
                self._door_hits[track_id] = 0
                self._origin.pop(track_id, None)
                self._moved.discard(track_id)
                self._clear_best(track_id)
                continue
            self._last_seen[track_id] = observed_at
            self._door_hits[track_id] = self._door_hits.get(track_id, 0) + 1
            origin = self._origin.setdefault(track_id, point)
            if math.hypot(point.x - origin.x, point.y - origin.y) >= self.minimum_movement:
                self._moved.add(track_id)

            result = self.consensus.resolve(track_id)
            if result.status != MatchStatus.MATCHED or not result.external_id:
                continue
            present_until = self._present_until.get(result.external_id)
            if present_until is not None and observed_at < present_until:
                self._present_until[result.external_id] = observed_at + self.presence_gap
                self._handled_tracks.add(track_id)
                continue
            if (
                track_id not in self._handled_tracks
                and track_id in self._moved
                and self._door_hits[track_id] >= self.min_door_frames
            ):
                payload = self._emit(track_id, observed_at, result)
                if payload is not None:
                    emitted.append(payload)

        for track_id in self._known_tracks - seen_tracks - expired:
            self._door_hits[track_id] = 0
            self._origin.pop(track_id, None)
            self._moved.discard(track_id)
            self._clear_best(track_id)

        for track_id in expired:
            if (
                track_id not in self._handled_tracks
                and track_id in self._moved
                and self._door_hits.get(track_id, 0) >= self.min_door_frames
            ):
                payload = self._emit(
                    track_id,
                    self._last_seen.get(track_id, observed_at),
                    self.consensus.resolve(track_id),
                )
                if payload is not None:
                    emitted.append(payload)
            self._discard_track(track_id)
        return emitted

    def _emit(
        self,
        track_id: str,
        when: datetime,
        result: ConsensusResult,
    ) -> dict[str, Any] | None:
        best = self._best_for(track_id, result)
        matched = result.status == MatchStatus.MATCHED and bool(result.external_id)
        session_id = f"vis-{uuid.uuid4().hex}"
        evidence_ref = self._save_evidence(best)
        payload = build_door_event(
            session_id=session_id,
            track_id=track_id,
            camera_id=self.camera_id,
            room_id=self.room_id,
            consensus=result,
            model_version=self.engine.model_version,
            model_fingerprint=self.engine.model_fingerprint,
            evidence_ref=evidence_ref,
            evidence_captured_at=best.captured_at if best is not None else None,
            confirmed_at=when,
            observation_count=self._door_hits.get(track_id, 0),
        )
        self.outbox.enqueue(payload, now=when)
        self._handled_tracks.add(track_id)
        if matched and result.external_id:
            self._present_until[result.external_id] = when + self.presence_gap
            self._stage_learning(
                track_id=track_id,
                result=result,
                best=best,
                evidence_ref=evidence_ref,
                event_id=payload["event_id"],
                when=when,
            )
        return payload

    def _discard_mode_state(self, track_id: str) -> None:
        self._door_hits.pop(track_id, None)
        self._origin.pop(track_id, None)
        self._moved.discard(track_id)
        self._last_seen.pop(track_id, None)
        self._handled_tracks.discard(track_id)
        self._last_raw_identity.pop(track_id, None)

    def flush(self, observed_at: datetime) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        for track_id in tuple(self._known_tracks):
            if (
                track_id not in self._handled_tracks
                and track_id in self._moved
                and self._door_hits.get(track_id, 0) >= self.min_door_frames
            ):
                payload = self._emit(
                    track_id,
                    self._last_seen.get(track_id, observed_at),
                    self.consensus.resolve(track_id),
                )
                if payload is not None:
                    emitted.append(payload)
            self._discard_mode_state(track_id)
        self._reset()
        return emitted


__all__ = [
    "ConsensusPolicy",
    "DetectionRegion",
    "DoorEventProcessor",
    "LearningPolicy",
    "RecognitionEngine",
    "VisionProcessor",
    "build_access_event",
    "build_door_event",
]
