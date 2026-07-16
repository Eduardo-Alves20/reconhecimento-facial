"""Confirma entradas a partir de trajetórias normalizadas."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal


_GEOMETRY_EPSILON = 1e-9


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} deve incluir fuso horário")


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if isinstance(self.x, bool) or isinstance(self.y, bool):
            raise ValueError("as coordenadas devem ser números entre 0 e 1")
        x = float(self.x)
        y = float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("as coordenadas devem ser finitas")
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("as coordenadas devem estar entre 0 e 1")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)


def _cross(a: Point, b: Point, point: Point) -> float:
    return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)


def _point_on_segment(point: Point, a: Point, b: Point) -> bool:
    if abs(_cross(a, b, point)) > _GEOMETRY_EPSILON:
        return False
    return (
        min(a.x, b.x) - _GEOMETRY_EPSILON
        <= point.x
        <= max(a.x, b.x) + _GEOMETRY_EPSILON
        and min(a.y, b.y) - _GEOMETRY_EPSILON
        <= point.y
        <= max(a.y, b.y) + _GEOMETRY_EPSILON
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    orientations = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    if (
        (orientations[0] > _GEOMETRY_EPSILON and orientations[1] < -_GEOMETRY_EPSILON)
        or (orientations[0] < -_GEOMETRY_EPSILON and orientations[1] > _GEOMETRY_EPSILON)
    ) and (
        (orientations[2] > _GEOMETRY_EPSILON and orientations[3] < -_GEOMETRY_EPSILON)
        or (orientations[2] < -_GEOMETRY_EPSILON and orientations[3] > _GEOMETRY_EPSILON)
    ):
        return True
    return (
        _point_on_segment(c, a, b)
        or _point_on_segment(d, a, b)
        or _point_on_segment(a, c, d)
        or _point_on_segment(b, c, d)
    )


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) < 3:
            raise ValueError("um polígono precisa de pelo menos três pontos")
        if any(not isinstance(point, Point) for point in points):
            raise TypeError("os vértices do polígono devem ser Point")
        if len(set(points)) != len(points):
            raise ValueError("um polígono não pode repetir vértices")

        edges = list(zip(points, points[1:] + points[:1], strict=True))
        for first_index, (first_start, first_end) in enumerate(edges):
            for second_index in range(first_index + 1, len(edges)):
                if (
                    second_index == first_index + 1
                    or (first_index == 0 and second_index == len(edges) - 1)
                ):
                    continue
                second_start, second_end = edges[second_index]
                if _segments_intersect(
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                ):
                    raise ValueError("o polígono não pode ser auto-intersectante")

        twice_area = sum(
            current.x * following.y - following.x * current.y
            for current, following in zip(points, points[1:] + points[:1], strict=True)
        )
        if abs(twice_area) <= _GEOMETRY_EPSILON:
            raise ValueError("o polígono não pode ter área zero")
        object.__setattr__(self, "points", points)

    @property
    def center(self) -> Point:
        return Point(
            sum(point.x for point in self.points) / len(self.points),
            sum(point.y for point in self.points) / len(self.points),
        )

    def contains(self, point: Point) -> bool:
        inside = False
        previous = self.points[-1]
        for current in self.points:
            if _point_on_segment(point, previous, current):
                return True
            if (current.y > point.y) != (previous.y > point.y):
                intersection_x = (
                    (previous.x - current.x)
                    * (point.y - current.y)
                    / (previous.y - current.y)
                    + current.x
                )
                if point.x < intersection_x:
                    inside = not inside
            previous = current
        return inside


@dataclass(frozen=True, slots=True)
class DirectedLine:
    start: Point
    end: Point
    inside_side: Literal["left", "right"] = "left"

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("a linha direcional precisa de dois pontos distintos")
        if self.inside_side not in ("left", "right"):
            raise ValueError("inside_side deve ser 'left' ou 'right'")

    def signed_distance(self, point: Point) -> float:
        length = math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)
        signed = _cross(self.start, self.end, point) / length
        if self.inside_side == "right":
            signed = -signed
        return signed

    def side_of(self, point: Point, *, deadband: float = 0.0) -> int:
        signed = self.signed_distance(point)
        if abs(signed) <= max(deadband, _GEOMETRY_EPSILON):
            return 0
        return 1 if signed > 0 else -1

    def crosses_segment(
        self,
        previous: Point,
        current: Point,
        *,
        segment_margin: float = 0.0,
    ) -> bool:
        rx = current.x - previous.x
        ry = current.y - previous.y
        sx = self.end.x - self.start.x
        sy = self.end.y - self.start.y
        denominator = rx * sy - ry * sx
        if abs(denominator) <= _GEOMETRY_EPSILON:
            return False
        qpx = self.start.x - previous.x
        qpy = self.start.y - previous.y
        movement_fraction = (qpx * sy - qpy * sx) / denominator
        line_fraction = (qpx * ry - qpy * rx) / denominator
        return (
            -_GEOMETRY_EPSILON <= movement_fraction <= 1 + _GEOMETRY_EPSILON
            and -segment_margin <= line_fraction <= 1 + segment_margin
        )


class EntryMethod(StrEnum):
    DIRECTED_LINE = "DIRECTED_LINE"
    ZONE_SEQUENCE = "ZONE_SEQUENCE"


@dataclass(frozen=True, slots=True)
class EntryTrackerConfig:
    camera_id: str
    room_id: str
    door_zone: Polygon
    inside_zone: Polygon
    entry_line: DirectedLine | None = None
    min_door_observations: int = 2
    min_inside_observations: int = 2
    track_timeout: timedelta = timedelta(seconds=5)
    cooldown: timedelta = timedelta(seconds=10)
    line_deadband: float = 0.015
    line_segment_margin: float = 0.05
    min_crossing_displacement: float = 0.04
    max_transition: timedelta = timedelta(seconds=3)

    def __post_init__(self) -> None:
        camera_id = self.camera_id.strip()
        room_id = self.room_id.strip()
        if not camera_id or not room_id:
            raise ValueError("camera_id e room_id são obrigatórios")
        if isinstance(self.min_door_observations, bool) or self.min_door_observations < 1:
            raise ValueError("min_door_observations deve ser maior ou igual a 1")
        if isinstance(self.min_inside_observations, bool) or self.min_inside_observations < 1:
            raise ValueError("min_inside_observations deve ser maior ou igual a 1")
        if self.track_timeout <= timedelta(0):
            raise ValueError("track_timeout deve ser positivo")
        if self.cooldown < timedelta(0):
            raise ValueError("cooldown não pode ser negativo")
        for name, value in (
            ("line_deadband", self.line_deadband),
            ("line_segment_margin", self.line_segment_margin),
            ("min_crossing_displacement", self.min_crossing_displacement),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} deve estar entre 0 e 1")
        if self.max_transition <= timedelta(0):
            raise ValueError("max_transition deve ser positivo")
        if self.entry_line is not None:
            door_distance = self.entry_line.signed_distance(self.door_zone.center)
            inside_distance = self.entry_line.signed_distance(self.inside_zone.center)
            if door_distance >= -self.line_deadband:
                raise ValueError("door_zone precisa ficar no lado externo da entry_line")
            if inside_distance <= self.line_deadband:
                raise ValueError("inside_zone precisa ficar no lado interno da entry_line")
            if not self.entry_line.crosses_segment(
                self.door_zone.center,
                self.inside_zone.center,
                segment_margin=self.line_segment_margin,
            ):
                raise ValueError("entry_line não cobre a passagem entre as zonas")
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "room_id", room_id)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: str
    observed_at: datetime
    centroid: Point
    face_observation_id: str | None = None

    def __post_init__(self) -> None:
        track_id = self.track_id.strip()
        if not track_id:
            raise ValueError("track_id é obrigatório")
        _require_aware(self.observed_at, "observed_at")
        face_id = self.face_observation_id
        if face_id is not None:
            face_id = face_id.strip()
            if not face_id:
                raise ValueError("face_observation_id não pode ser vazio")
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "face_observation_id", face_id)


@dataclass(frozen=True, slots=True)
class EntryConfirmation:
    session_id: str
    camera_id: str
    room_id: str
    track_id: str
    started_at: datetime
    confirmed_at: datetime
    method: EntryMethod
    face_observation_ids: tuple[str, ...]
    observation_count: int


@dataclass(slots=True)
class _TrackState:
    session_id: str
    first_seen: datetime
    last_seen: datetime
    observation_count: int = 0
    face_ids: list[str] = field(default_factory=list)
    face_id_set: set[str] = field(default_factory=set)
    origin_inside: bool = False
    door_count: int = 0
    door_ready: bool = False
    inside_count: int = 0
    last_line_side: int = 0
    line_crossed: bool = False
    door_ready_at: datetime | None = None
    door_origin: Point | None = None
    last_point: Point | None = None

    def record(self, observation: TrackObservation) -> None:
        self.last_seen = observation.observed_at
        self.observation_count += 1
        face_id = observation.face_observation_id
        if face_id is not None and face_id not in self.face_id_set:
            self.face_id_set.add(face_id)
            self.face_ids.append(face_id)
        self.last_point = observation.centroid


class EntryTracker:
    def __init__(self, config: EntryTrackerConfig) -> None:
        self.config = config
        self._tracks: dict[str, _TrackState] = {}
        self._cooldowns: dict[str, datetime] = {}
        self._watermark: datetime | None = None
        self._session_sequence = 0

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        return tuple(self._tracks)

    @property
    def active_session_ids(self) -> tuple[str, ...]:
        return tuple(state.session_id for state in self._tracks.values())

    def observe(self, observation: TrackObservation) -> EntryConfirmation | None:
        self._move_clock(observation.observed_at)

        cooldown_until = self._cooldowns.get(observation.track_id)
        if cooldown_until is not None:
            if observation.observed_at < cooldown_until:
                return None
            del self._cooldowns[observation.track_id]

        state = self._tracks.get(observation.track_id)
        if state is None:
            state = self._new_state(observation)
            self._tracks[observation.track_id] = state
        elif observation.observed_at < state.last_seen:
            raise ValueError("observações da trilha devem estar em ordem cronológica")

        previous_point = state.last_point
        state.record(observation)
        if self.config.entry_line is None:
            confirmed = self._observe_zone_sequence(state, observation)
        else:
            confirmed = self._observe_directed_line(state, observation, previous_point)

        if not confirmed:
            return None
        return self._confirm(observation.track_id, state, observation.observed_at)

    def advance_time(self, now: datetime) -> tuple[str, ...]:
        return self._move_clock(now)

    def discard(self, track_id: str) -> None:
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        self._tracks.clear()
        self._cooldowns.clear()
        self._watermark = None
        self._session_sequence = 0

    def _move_clock(self, now: datetime) -> tuple[str, ...]:
        _require_aware(now, "now")
        if self._watermark is not None and now < self._watermark:
            raise ValueError("observações devem chegar em ordem cronológica")
        self._watermark = now

        expired_track_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if now - state.last_seen > self.config.track_timeout
        ]
        expired_sessions = tuple(
            self._tracks[track_id].session_id for track_id in expired_track_ids
        )
        for track_id in expired_track_ids:
            del self._tracks[track_id]

        expired_cooldowns = [
            track_id
            for track_id, cooldown_until in self._cooldowns.items()
            if now >= cooldown_until
        ]
        for track_id in expired_cooldowns:
            del self._cooldowns[track_id]
        return expired_sessions

    def _new_state(self, observation: TrackObservation) -> _TrackState:
        self._session_sequence += 1
        material = "|".join(
            (
                self.config.camera_id,
                self.config.room_id,
                observation.track_id,
                observation.observed_at.astimezone(UTC).isoformat(),
                str(self._session_sequence),
            )
        )
        session_id = f"vis-{uuid.uuid5(uuid.NAMESPACE_URL, material).hex}"
        in_door = self.config.door_zone.contains(observation.centroid)
        in_inside = self.config.inside_zone.contains(observation.centroid)
        return _TrackState(
            session_id=session_id,
            first_seen=observation.observed_at,
            last_seen=observation.observed_at,
            origin_inside=in_inside and not in_door,
        )

    def _observe_zone_sequence(
        self, state: _TrackState, observation: TrackObservation
    ) -> bool:
        in_door = self.config.door_zone.contains(observation.centroid)
        in_inside = self.config.inside_zone.contains(observation.centroid)

        if state.origin_inside:
            if not in_inside and not in_door:
                state.origin_inside = False
            return False

        if not state.door_ready:
            if in_door:
                state.door_count += 1
                if state.door_count >= self.config.min_door_observations:
                    self._arm(state, observation)
            else:
                state.door_count = 0
                if in_inside:
                    state.origin_inside = True
            return False

        if not self._transition_valid(state, observation):
            self._reset_candidate(state)
            return False
        if in_inside:
            if not self._has_minimum_displacement(state, observation.centroid):
                state.inside_count = 0
                return False
            state.inside_count += 1
            return state.inside_count >= self.config.min_inside_observations
        if in_door:
            state.inside_count = 0
            return False

        self._reset_candidate(state)
        return False

    def _observe_directed_line(
        self,
        state: _TrackState,
        observation: TrackObservation,
        previous_point: Point | None,
    ) -> bool:
        line = self.config.entry_line
        assert line is not None
        side = line.side_of(
            observation.centroid,
            deadband=self.config.line_deadband,
        )
        in_door = self.config.door_zone.contains(observation.centroid)
        in_inside = self.config.inside_zone.contains(observation.centroid)

        if side < 0:
            if in_door:
                state.door_count += 1
                if state.door_count >= self.config.min_door_observations:
                    self._arm(state, observation)
            elif state.door_ready:
                self._reset_candidate(state)
            else:
                state.door_count = 0
            state.line_crossed = False
            state.inside_count = 0
            state.last_line_side = side
            return False

        if side == 0:
            return False

        if not self._transition_valid(state, observation):
            self._reset_candidate(state)
            state.last_line_side = side
            return False
        if (
            state.door_ready
            and state.last_line_side < 0
            and previous_point is not None
            and line.crosses_segment(
                previous_point,
                observation.centroid,
                segment_margin=self.config.line_segment_margin,
            )
            and self._has_minimum_displacement(state, observation.centroid)
        ):
            state.line_crossed = True
        state.last_line_side = side

        if not state.line_crossed:
            state.inside_count = 0
            return False
        if not in_inside:
            state.inside_count = 0
            return False

        state.inside_count += 1
        return state.inside_count >= self.config.min_inside_observations

    @staticmethod
    def _displacement(first: Point, second: Point) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)

    def _arm(self, state: _TrackState, observation: TrackObservation) -> None:
        if not state.door_ready:
            state.door_ready = True
            state.door_ready_at = observation.observed_at
            state.door_origin = observation.centroid

    def _transition_valid(
        self,
        state: _TrackState,
        observation: TrackObservation,
    ) -> bool:
        return (
            state.door_ready_at is not None
            and observation.observed_at - state.door_ready_at
            <= self.config.max_transition
        )

    def _has_minimum_displacement(
        self,
        state: _TrackState,
        current: Point,
    ) -> bool:
        return (
            state.door_origin is not None
            and self._displacement(state.door_origin, current)
            >= self.config.min_crossing_displacement
        )

    @staticmethod
    def _reset_candidate(state: _TrackState) -> None:
        state.door_count = 0
        state.door_ready = False
        state.inside_count = 0
        state.line_crossed = False
        state.door_ready_at = None
        state.door_origin = None

    def _confirm(
        self, track_id: str, state: _TrackState, confirmed_at: datetime
    ) -> EntryConfirmation:
        method = (
            EntryMethod.DIRECTED_LINE
            if self.config.entry_line is not None
            else EntryMethod.ZONE_SEQUENCE
        )
        confirmation = EntryConfirmation(
            session_id=state.session_id,
            camera_id=self.config.camera_id,
            room_id=self.config.room_id,
            track_id=track_id,
            started_at=state.first_seen,
            confirmed_at=confirmed_at,
            method=method,
            face_observation_ids=tuple(state.face_ids),
            observation_count=state.observation_count,
        )
        del self._tracks[track_id]
        if self.config.cooldown > timedelta(0):
            self._cooldowns[track_id] = confirmed_at + self.config.cooldown
        return confirmation
