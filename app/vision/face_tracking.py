"""Associação leve de rostos entre quadros e consenso de identidade."""

from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .recognizer import DetectedFace, MatchDecision, MatchStatus


@dataclass(frozen=True, slots=True)
class TrackedFace:
    track_id: str
    face: DetectedFace


@dataclass(slots=True)
class _Track:
    centroid: tuple[float, float]
    last_seen: datetime


class FaceCentroidTracker:
    """Associa centróides próximos sem depender de um tracker nativo do OpenCV."""

    def __init__(
        self,
        *,
        max_distance: float = 0.18,
        max_age: timedelta = timedelta(seconds=2),
        boot_id: str | None = None,
    ) -> None:
        if not 0 < max_distance <= math.sqrt(2):
            raise ValueError("max_distance deve ser positivo e normalizado")
        if max_age <= timedelta(0):
            raise ValueError("max_age deve ser positivo")
        self.max_distance = max_distance
        self.max_age = max_age
        self.boot_id = boot_id or uuid.uuid4().hex[:12]
        self._sequence = 0
        self._tracks: dict[str, _Track] = {}

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        return tuple(self._tracks)

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def _expire(self, observed_at: datetime) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if observed_at - track.last_seen > self.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def update(self, faces: list[DetectedFace], observed_at: datetime) -> list[TrackedFace]:
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at deve incluir fuso")
        self._expire(observed_at)

        pairs: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            for face_index, face in enumerate(faces):
                distance = self._distance(track.centroid, face.centroid)
                if distance <= self.max_distance:
                    pairs.append((distance, track_id, face_index))
        pairs.sort()

        assigned_tracks: set[str] = set()
        assigned_faces: set[int] = set()
        assignments: dict[int, str] = {}
        for _, track_id, face_index in pairs:
            if track_id in assigned_tracks or face_index in assigned_faces:
                continue
            assigned_tracks.add(track_id)
            assigned_faces.add(face_index)
            assignments[face_index] = track_id

        results: list[TrackedFace] = []
        for index, face in enumerate(faces):
            track_id = assignments.get(index)
            if track_id is None:
                self._sequence += 1
                track_id = f"{self.boot_id}:face-{self._sequence}"
            self._tracks[track_id] = _Track(face.centroid, observed_at)
            results.append(TrackedFace(track_id, face))
        return results


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    status: MatchStatus
    external_id: str | None
    display_name: str | None
    similarity: float | None
    margin: float | None
    face_quality: float
    supporting_frames: int


class IdentityConsensus:
    """Exige repetição em vários quadros antes de atribuir uma pessoa."""

    def __init__(self, *, history_size: int = 12, minimum_matches: int = 3) -> None:
        if history_size < 1 or minimum_matches < 1 or minimum_matches > history_size:
            raise ValueError("histórico e mínimo de confirmações são inválidos")
        self.history_size = history_size
        self.minimum_matches = minimum_matches
        self._history: dict[str, deque[MatchDecision]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def add(self, track_id: str, decision: MatchDecision) -> None:
        self._history[track_id].append(decision)

    def discard(self, track_id: str) -> None:
        self._history.pop(track_id, None)

    def resolve(self, track_id: str, *, consume: bool = False) -> ConsensusResult:
        decisions = list(self._history.get(track_id, ()))
        matched = [
            item
            for item in decisions
            if item.status == MatchStatus.MATCHED and item.external_id is not None
        ]
        counts = Counter(item.external_id for item in matched)
        if counts:
            winner_id, winner_count = counts.most_common(1)[0]
            winner_items = [item for item in matched if item.external_id == winner_id]
            competing_ids = len(counts)
            if winner_count >= self.minimum_matches and winner_count / len(matched) >= 0.60:
                result = ConsensusResult(
                    status=MatchStatus.MATCHED,
                    external_id=winner_id,
                    display_name=winner_items[-1].display_name,
                    similarity=sum(item.similarity or 0.0 for item in winner_items)
                    / winner_count,
                    margin=min(item.margin or 0.0 for item in winner_items),
                    face_quality=sum(item.face_quality for item in winner_items) / winner_count,
                    supporting_frames=winner_count,
                )
            elif competing_ids > 1:
                result = self._unresolved(MatchStatus.AMBIGUOUS, decisions)
            else:
                result = self._unresolved(MatchStatus.UNKNOWN, decisions)
        elif any(item.status == MatchStatus.AMBIGUOUS for item in decisions):
            result = self._unresolved(MatchStatus.AMBIGUOUS, decisions)
        else:
            result = self._unresolved(MatchStatus.UNKNOWN, decisions)

        if consume:
            self.discard(track_id)
        return result

    @staticmethod
    def _unresolved(
        status: MatchStatus, decisions: list[MatchDecision]
    ) -> ConsensusResult:
        quality = (
            sum(item.face_quality for item in decisions) / len(decisions) if decisions else 0.0
        )
        return ConsensusResult(status, None, None, None, None, quality, 0)
