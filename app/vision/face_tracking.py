"""Associação de rostos entre quadros e consenso temporal de identidade."""

from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from .recognizer import DetectedFace, MatchDecision, MatchStatus


@dataclass(frozen=True, slots=True)
class TrackedFace:
    track_id: str
    face: DetectedFace


@dataclass(slots=True)
class _Track:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    last_seen: datetime
    velocity: tuple[float, float] = (0.0, 0.0)
    appearance: tuple[float, ...] | None = None
    hits: int = 1


def _unit_feature(feature: Any) -> tuple[float, ...] | None:
    if feature is None:
        return None
    try:
        flattened = feature.reshape(-1) if hasattr(feature, "reshape") else feature
        values = tuple(float(value) for value in flattened)
    except (TypeError, ValueError):
        return None
    if not values or not all(math.isfinite(value) for value in values):
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in values)


def _cosine(
    first: tuple[float, ...] | None, second: tuple[float, ...] | None
) -> float | None:
    if first is None or second is None or len(first) != len(second):
        return None
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second))))


def _bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def _size_similarity(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    first_area = first[2] * first[3]
    second_area = second[2] * second[3]
    return min(first_area, second_area) / max(first_area, second_area)


def _hungarian(costs: Sequence[Sequence[float]]) -> list[int]:
    row_count = len(costs)
    if row_count == 0:
        return []
    column_count = len(costs[0])
    if column_count < row_count or any(len(row) != column_count for row in costs):
        raise ValueError("A matriz de custo deve ser retangular e ter colunas suficientes.")

    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = math.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    previous_column[column] = current_column
                if (
                    minimum[column] < delta
                    or (
                        math.isclose(minimum[column], delta, abs_tol=1e-12)
                        and column < next_column
                    )
                ):
                    delta = minimum[column]
                    next_column = column
            if not math.isfinite(delta):
                raise ValueError("Não existe associação finita para a matriz.")
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            prior = previous_column[current_column]
            matched_row[current_column] = matched_row[prior]
            current_column = prior
            if current_column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return assignment


class FaceCentroidTracker:
    """Associa detecções por movimento, caixa facial e aparência."""

    def __init__(
        self,
        *,
        max_distance: float = 0.18,
        max_age: timedelta = timedelta(seconds=2),
        boot_id: str | None = None,
        appearance_weight: float = 0.45,
        min_appearance_similarity: float = 0.05,
        appearance_update_min_quality: float = 0.50,
    ) -> None:
        if not 0 < max_distance <= math.sqrt(2):
            raise ValueError("max_distance deve ser positivo e normalizado")
        if max_age <= timedelta(0):
            raise ValueError("max_age deve ser positivo")
        if not 0 <= appearance_weight < 1:
            raise ValueError("appearance_weight deve estar entre 0 e 1")
        if not -1 <= min_appearance_similarity <= 1:
            raise ValueError("min_appearance_similarity deve estar entre -1 e 1")
        if (
            not math.isfinite(appearance_update_min_quality)
            or not 0 <= appearance_update_min_quality <= 1
        ):
            raise ValueError("appearance_update_min_quality deve estar entre 0 e 1")
        self.max_distance = max_distance
        self.max_age = max_age
        self.boot_id = boot_id or uuid.uuid4().hex[:12]
        self.appearance_weight = appearance_weight
        self.min_appearance_similarity = min_appearance_similarity
        self.appearance_update_min_quality = appearance_update_min_quality
        self._sequence = 0
        self._tracks: dict[str, _Track] = {}
        self._last_observed_at: datetime | None = None

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        return tuple(self._tracks)

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _validate_face(face: DetectedFace) -> None:
        if not isinstance(face, DetectedFace):
            raise TypeError("O tracker aceita apenas DetectedFace.")
        if face.bbox[2] <= 0 or face.bbox[3] <= 0:
            raise ValueError("A caixa do rosto deve ter área positiva.")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in face.centroid):
            raise ValueError("O centro do rosto deve estar normalizado.")

    def _expire(self, observed_at: datetime) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if observed_at - track.last_seen > self.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def discard(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)

    def clear(self) -> None:
        self._tracks.clear()
        self._last_observed_at = None

    reset = clear

    def _prediction(
        self, track: _Track, observed_at: datetime
    ) -> tuple[tuple[float, float], float]:
        elapsed = max(0.0, (observed_at - track.last_seen).total_seconds())
        horizon = min(elapsed, self.max_age.total_seconds())
        predicted = (
            max(0.0, min(1.0, track.centroid[0] + track.velocity[0] * horizon)),
            max(0.0, min(1.0, track.centroid[1] + track.velocity[1] * horizon)),
        )
        age_ratio = horizon / self.max_age.total_seconds()
        gate = min(math.sqrt(2), self.max_distance * (1.0 + 0.60 * age_ratio))
        return predicted, gate

    def _association_cost(
        self,
        track: _Track,
        face: DetectedFace,
        appearance_feature: tuple[float, ...] | None,
        observed_at: datetime,
    ) -> float | None:
        predicted, gate = self._prediction(track, observed_at)
        distance = self._distance(predicted, face.centroid)
        if distance > gate:
            return None

        appearance = _cosine(track.appearance, appearance_feature)
        if appearance is not None and appearance < self.min_appearance_similarity:
            return None
        distance_cost = distance / gate
        overlap_cost = 1.0 - _bbox_iou(track.bbox, face.bbox)
        scale_cost = 1.0 - _size_similarity(track.bbox, face.bbox)

        geometric_weight = 1.0 - (
            self.appearance_weight if appearance is not None else 0.0
        )
        cost = geometric_weight * (
            distance_cost * 0.62 + overlap_cost * 0.25 + scale_cost * 0.13
        )
        if appearance is not None:
            cost += self.appearance_weight * (1.0 - max(0.0, appearance))
        return max(0.0, min(1.0, cost))

    def _assign(
        self,
        tracks: Sequence[tuple[str, _Track]],
        faces: Sequence[DetectedFace],
        appearances: Sequence[tuple[float, ...] | None],
        observed_at: datetime,
    ) -> dict[int, str]:
        if not tracks or not faces:
            return {}
        track_count = len(tracks)
        face_count = len(faces)
        size = track_count + face_count
        invalid = 1_000_000.0
        unmatched = 0.48
        costs = [[0.0] * size for _ in range(size)]

        for track_index, (_, track) in enumerate(tracks):
            for face_index, face in enumerate(faces):
                value = self._association_cost(
                    track,
                    face,
                    appearances[face_index],
                    observed_at,
                )
                costs[track_index][face_index] = (
                    invalid
                    if value is None
                    else value + track_index * 1e-10 + face_index * 1e-12
                )
            for dummy_column in range(face_count, size):
                costs[track_index][dummy_column] = unmatched + dummy_column * 1e-12

        for dummy_row in range(track_count, size):
            for face_index in range(face_count):
                costs[dummy_row][face_index] = unmatched + face_index * 1e-12
            for dummy_column in range(face_count, size):
                costs[dummy_row][dummy_column] = 0.0

        assignment = _hungarian(costs)
        result: dict[int, str] = {}
        for track_index in range(track_count):
            face_index = assignment[track_index]
            if (
                0 <= face_index < face_count
                and costs[track_index][face_index] < invalid
                and costs[track_index][face_index] < unmatched * 2
            ):
                result[face_index] = tracks[track_index][0]
        return result

    @staticmethod
    def _updated_appearance(
        current: tuple[float, ...] | None,
        observed: tuple[float, ...] | None,
        *,
        quality: float,
        minimum_quality: float,
    ) -> tuple[float, ...] | None:
        if observed is None or quality < minimum_quality:
            return current
        if current is None or len(current) != len(observed):
            return observed
        blended = tuple(0.75 * old + 0.25 * new for old, new in zip(current, observed))
        return _unit_feature(blended)

    def update(self, faces: list[DetectedFace], observed_at: datetime) -> list[TrackedFace]:
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at deve incluir fuso")
        if self._last_observed_at is not None and observed_at < self._last_observed_at:
            raise ValueError("observed_at não pode retroceder")
        for face in faces:
            self._validate_face(face)
        self._last_observed_at = observed_at
        self._expire(observed_at)

        tracks = list(self._tracks.items())
        appearances = [
            (
                _unit_feature(face.feature)
                if face.quality >= self.appearance_update_min_quality
                else None
            )
            for face in faces
        ]
        assignments = self._assign(tracks, faces, appearances, observed_at)
        results: list[TrackedFace] = []
        for index, face in enumerate(faces):
            track_id = assignments.get(index)
            if track_id is None:
                self._sequence += 1
                track_id = f"{self.boot_id}:face-{self._sequence}"
                self._tracks[track_id] = _Track(
                    face.centroid,
                    face.bbox,
                    observed_at,
                    appearance=(
                        appearances[index]
                        if face.quality >= self.appearance_update_min_quality
                        else None
                    ),
                )
            else:
                track = self._tracks[track_id]
                elapsed = (observed_at - track.last_seen).total_seconds()
                if elapsed > 0:
                    measured_velocity = (
                        (face.centroid[0] - track.centroid[0]) / elapsed,
                        (face.centroid[1] - track.centroid[1]) / elapsed,
                    )
                    if track.hits == 1:
                        velocity = measured_velocity
                    else:
                        velocity = (
                            track.velocity[0] * 0.65 + measured_velocity[0] * 0.35,
                            track.velocity[1] * 0.65 + measured_velocity[1] * 0.35,
                        )
                else:
                    velocity = track.velocity
                track.centroid = face.centroid
                track.bbox = face.bbox
                track.last_seen = observed_at
                track.velocity = velocity
                track.appearance = self._updated_appearance(
                    track.appearance,
                    appearances[index],
                    quality=face.quality,
                    minimum_quality=self.appearance_update_min_quality,
                )
                track.hits += 1
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
    """Confirma uma identidade apenas com suporte dominante e recente."""

    def __init__(
        self,
        *,
        history_size: int = 12,
        minimum_matches: int = 3,
        minimum_ratio: float = 0.60,
        minimum_consecutive: int | None = None,
    ) -> None:
        consecutive = (
            min(2, minimum_matches)
            if minimum_consecutive is None
            else minimum_consecutive
        )
        if history_size < 1 or minimum_matches < 1 or minimum_matches > history_size:
            raise ValueError("histórico e mínimo de confirmações são inválidos")
        if (
            not math.isfinite(minimum_ratio)
            or not 0 < minimum_ratio <= 1
            or consecutive < 1
            or consecutive > minimum_matches
        ):
            raise ValueError("regras de consenso são inválidas")
        self.history_size = history_size
        self.minimum_matches = minimum_matches
        self.minimum_ratio = minimum_ratio
        self.minimum_consecutive = consecutive
        self._history: dict[str, deque[MatchDecision]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def add(self, track_id: str, decision: MatchDecision) -> None:
        if not track_id:
            raise ValueError("track_id não pode ser vazio")
        self._history[track_id].append(decision)

    def discard(self, track_id: str) -> None:
        self._history.pop(track_id, None)

    def clear(self) -> None:
        self._history.clear()

    reset = clear

    def resolve(self, track_id: str, *, consume: bool = False) -> ConsensusResult:
        decisions = list(self._history.get(track_id, ()))
        matched = [
            item
            for item in decisions
            if item.status == MatchStatus.MATCHED and item.external_id is not None
        ]
        counts = Counter(item.external_id for item in matched)
        if counts:
            winner_id, winner_count = sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            winner_items = [
                item for item in matched if item.external_id == winner_id
            ]
            consecutive = 0
            for item in reversed(decisions):
                if (
                    item.status == MatchStatus.MATCHED
                    and item.external_id == winner_id
                ):
                    consecutive += 1
                else:
                    break
            support_ratio = winner_count / len(decisions)
            if (
                winner_count >= self.minimum_matches
                and support_ratio >= self.minimum_ratio
                and consecutive >= self.minimum_consecutive
            ):
                result = ConsensusResult(
                    status=MatchStatus.MATCHED,
                    external_id=winner_id,
                    display_name=winner_items[-1].display_name,
                    similarity=sum(
                        item.similarity if item.similarity is not None else 0.0
                        for item in winner_items
                    )
                    / winner_count,
                    margin=min(
                        item.margin if item.margin is not None else 0.0
                        for item in winner_items
                    ),
                    face_quality=sum(
                        item.face_quality for item in winner_items
                    )
                    / winner_count,
                    supporting_frames=winner_count,
                )
            elif len(counts) > 1 or any(
                item.status == MatchStatus.AMBIGUOUS for item in decisions
            ):
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
            sum(item.face_quality for item in decisions) / len(decisions)
            if decisions
            else 0.0
        )
        return ConsensusResult(status, None, None, None, None, quality, 0)
