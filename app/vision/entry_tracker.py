"""Confirmação determinística de entradas a partir de trilhas de pessoas.

O módulo não detecta pessoas nem rostos. Ele recebe centróides normalizados
produzidos por um detector/rastreador e confirma apenas o movimento compatível
com entrada. A decisão pode usar uma linha direcional (mais segura) ou, na
ausência dela, a sequência de zonas ``porta -> interior``.
"""

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
    """Ponto em coordenadas normalizadas da imagem (0 a 1)."""

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


@dataclass(frozen=True, slots=True)
class Polygon:
    """Zona poligonal fechada; pontos na borda pertencem à zona."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) < 3:
            raise ValueError("um polígono precisa de pelo menos três pontos")
        if any(not isinstance(point, Point) for point in points):
            raise TypeError("os vértices do polígono devem ser Point")

        twice_area = sum(
            current.x * following.y - following.x * current.y
            for current, following in zip(points, points[1:] + points[:1], strict=True)
        )
        if abs(twice_area) <= _GEOMETRY_EPSILON:
            raise ValueError("o polígono não pode ter área zero")
        object.__setattr__(self, "points", points)

    def contains(self, point: Point) -> bool:
        """Retorna ``True`` se o ponto estiver dentro ou sobre a borda."""

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
    """Linha cuja esquerda ou direita é declarada como lado interior."""

    start: Point
    end: Point
    inside_side: Literal["left", "right"] = "left"

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("a linha direcional precisa de dois pontos distintos")
        if self.inside_side not in ("left", "right"):
            raise ValueError("inside_side deve ser 'left' ou 'right'")

    def side_of(self, point: Point) -> int:
        """Retorna ``1`` para interior, ``-1`` para exterior e ``0`` na linha."""

        signed = _cross(self.start, self.end, point)
        if self.inside_side == "right":
            signed = -signed
        if abs(signed) <= _GEOMETRY_EPSILON:
            return 0
        return 1 if signed > 0 else -1


class EntryMethod(StrEnum):
    DIRECTED_LINE = "DIRECTED_LINE"
    ZONE_SEQUENCE = "ZONE_SEQUENCE"


@dataclass(frozen=True, slots=True)
class EntryTrackerConfig:
    """Geometria e tolerâncias de uma câmera/sala."""

    camera_id: str
    room_id: str
    door_zone: Polygon
    inside_zone: Polygon
    entry_line: DirectedLine | None = None
    min_door_observations: int = 2
    min_inside_observations: int = 2
    track_timeout: timedelta = timedelta(seconds=5)
    cooldown: timedelta = timedelta(seconds=10)

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
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "room_id", room_id)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Uma amostra de uma trilha, normalmente o centro inferior da pessoa."""

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
    """Entrada visual confirmada, ainda que a identidade seja desconhecida."""

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

    def record(self, observation: TrackObservation) -> None:
        self.last_seen = observation.observed_at
        self.observation_count += 1
        face_id = observation.face_observation_id
        if face_id is not None and face_id not in self.face_id_set:
            self.face_id_set.add(face_id)
            self.face_ids.append(face_id)


class EntryTracker:
    """Máquina de estados para uma única câmera.

    As observações devem chegar em ordem cronológica. ``observe`` retorna uma
    confirmação somente no instante em que a persistência interior é atingida.
    """

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
        """Processa uma amostra e, quando aplicável, confirma uma entrada."""

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
            # Normalmente alcançado pela verificação global, mantido também
            # aqui para preservar a invariância caso a implementação evolua.
            raise ValueError("observações da trilha devem estar em ordem cronológica")

        state.record(observation)
        if self.config.entry_line is None:
            confirmed = self._observe_zone_sequence(state, observation)
        else:
            confirmed = self._observe_directed_line(state, observation)

        if not confirmed:
            return None
        return self._confirm(observation.track_id, state, observation.observed_at)

    def advance_time(self, now: datetime) -> tuple[str, ...]:
        """Expira trilhas inativas e retorna os IDs das sessões removidas."""

        return self._move_clock(now)

    def reset(self) -> None:
        """Remove estados voláteis; a configuração permanece inalterada."""

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

        # Uma trilha que nasceu dentro da sala é uma possível saída. Ela só
        # pode ser armada após uma observação inequivocamente fora das zonas.
        if state.origin_inside:
            if not in_inside and not in_door:
                state.origin_inside = False
            return False

        if not state.door_ready:
            if in_door:
                state.door_count += 1
                if state.door_count >= self.config.min_door_observations:
                    state.door_ready = True
            else:
                state.door_count = 0
                # Pular diretamente para o interior não comprova travessia.
                if in_inside:
                    state.origin_inside = True
            return False

        # Depois de armada na porta, a zona interior tem prioridade em eventual
        # pequena sobreposição entre os dois polígonos.
        if in_inside:
            state.inside_count += 1
            return state.inside_count >= self.config.min_inside_observations
        if in_door:
            state.inside_count = 0
            return False

        # A pessoa recuou para fora antes de entrar; uma nova persistência na
        # porta será exigida.
        state.door_count = 0
        state.door_ready = False
        state.inside_count = 0
        return False

    def _observe_directed_line(
        self, state: _TrackState, observation: TrackObservation
    ) -> bool:
        line = self.config.entry_line
        assert line is not None
        side = line.side_of(observation.centroid)
        in_door = self.config.door_zone.contains(observation.centroid)
        in_inside = self.config.inside_zone.contains(observation.centroid)

        if side < 0:
            if in_door:
                state.door_count += 1
                if state.door_count >= self.config.min_door_observations:
                    state.door_ready = True
            elif not state.door_ready:
                state.door_count = 0

            # Voltar ao exterior cancela uma travessia interior ainda não
            # persistida, mas preserva a possibilidade de uma nova entrada.
            state.line_crossed = False
            state.inside_count = 0
            state.last_line_side = side
            return False

        if side == 0:
            # Pontos exatamente na linha não definem direção nem quebram a
            # persistência exterior já observada.
            return False

        if state.door_ready and state.last_line_side < 0:
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
