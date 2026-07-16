from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.vision.entry_tracker import (
    DirectedLine,
    EntryMethod,
    EntryTracker,
    EntryTrackerConfig,
    Point,
    Polygon,
    TrackObservation,
)


BASE_TIME = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def rectangle(left: float, top: float, right: float, bottom: float) -> Polygon:
    return Polygon(
        (
            Point(left, top),
            Point(right, top),
            Point(right, bottom),
            Point(left, bottom),
        )
    )


def config(*, with_line: bool = False, **overrides: object) -> EntryTrackerConfig:
    values: dict[str, object] = {
        "camera_id": "cam-ti-01",
        "room_id": "sala-ti-01",
        "door_zone": rectangle(0.30, 0.30, 0.70, 0.52),
        "inside_zone": rectangle(0.20, 0.50, 0.80, 0.95),
        "entry_line": (
            DirectedLine(Point(0.25, 0.50), Point(0.75, 0.50), "left")
            if with_line
            else None
        ),
        "min_door_observations": 2,
        "min_inside_observations": 2,
        "track_timeout": timedelta(seconds=5),
        "cooldown": timedelta(seconds=10),
    }
    values.update(overrides)
    return EntryTrackerConfig(**values)  # type: ignore[arg-type]


def observation(
    track_id: str,
    offset_seconds: float,
    x: float,
    y: float,
    *,
    face_id: str | None = None,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        observed_at=BASE_TIME + timedelta(seconds=offset_seconds),
        centroid=Point(x, y),
        face_observation_id=face_id,
    )


def test_zone_sequence_confirms_entry_only_after_minimum_persistence() -> None:
    tracker = EntryTracker(config())

    assert tracker.observe(observation("track-1", 0.0, 0.5, 0.40, face_id="face-a")) is None
    assert tracker.observe(observation("track-1", 0.1, 0.5, 0.42)) is None
    assert tracker.observe(observation("track-1", 0.2, 0.5, 0.70)) is None
    result = tracker.observe(observation("track-1", 0.3, 0.5, 0.72, face_id="face-b"))

    assert result is not None
    assert result.camera_id == "cam-ti-01"
    assert result.room_id == "sala-ti-01"
    assert result.track_id == "track-1"
    assert result.method is EntryMethod.ZONE_SEQUENCE
    assert result.started_at == BASE_TIME
    assert result.confirmed_at == BASE_TIME + timedelta(seconds=0.3)
    assert result.face_observation_ids == ("face-a", "face-b")
    assert result.observation_count == 4
    assert result.session_id.startswith("vis-")


def test_face_seen_inside_without_crossing_never_becomes_entry() -> None:
    tracker = EntryTracker(config())

    for index in range(5):
        result = tracker.observe(
            observation(
                "face-only",
                index / 10,
                0.5,
                0.75,
                face_id=f"face-{index}",
            )
        )
        assert result is None


def test_exit_inside_to_door_does_not_become_entry() -> None:
    tracker = EntryTracker(config())
    path = [
        (0.5, 0.75),
        (0.5, 0.70),
        (0.5, 0.46),
        (0.5, 0.42),
        (0.5, 0.20),
    ]

    results = [
        tracker.observe(observation("leaving", index / 10, x, y))
        for index, (x, y) in enumerate(path)
    ]

    assert results == [None] * len(path)


def test_two_distinct_tracks_generate_two_independent_entries() -> None:
    tracker = EntryTracker(config())
    emitted = []
    samples = [
        ("a", 0.00, 0.40),
        ("b", 0.01, 0.40),
        ("a", 0.02, 0.42),
        ("b", 0.03, 0.42),
        ("a", 0.04, 0.70),
        ("b", 0.05, 0.70),
        ("a", 0.06, 0.72),
        ("b", 0.07, 0.72),
    ]
    for track_id, offset, y in samples:
        result = tracker.observe(observation(track_id, offset, 0.5, y))
        if result is not None:
            emitted.append(result)

    assert [event.track_id for event in emitted] == ["a", "b"]
    assert len({event.session_id for event in emitted}) == 2


def test_timeout_discards_partial_candidate_and_returns_expired_session() -> None:
    tracker = EntryTracker(config(track_timeout=timedelta(seconds=1)))
    tracker.observe(observation("slow", 0.0, 0.5, 0.40))
    old_session = tracker.active_session_ids[0]

    expired = tracker.advance_time(BASE_TIME + timedelta(seconds=1.1))
    assert expired == (old_session,)
    assert tracker.active_track_ids == ()

    # Uma amostra na porta e duas internas não reaproveitam a persistência
    # da sessão expirada.
    assert tracker.observe(observation("slow", 1.2, 0.5, 0.42)) is None
    assert tracker.observe(observation("slow", 1.3, 0.5, 0.70)) is None
    assert tracker.observe(observation("slow", 1.4, 0.5, 0.72)) is None


def test_discard_restarts_partial_track_state() -> None:
    tracker = EntryTracker(config())
    tracker.observe(observation("handoff", 0.0, 0.5, 0.40))
    tracker.observe(observation("handoff", 0.1, 0.5, 0.42))
    tracker.discard("handoff")

    assert tracker.observe(observation("handoff", 0.2, 0.5, 0.70)) is None
    assert tracker.observe(observation("handoff", 0.3, 0.5, 0.72)) is None


def test_cooldown_deduplicates_same_track_then_allows_a_new_session() -> None:
    tracker = EntryTracker(config(cooldown=timedelta(seconds=2)))
    first = None
    for offset, y in ((0.0, 0.40), (0.1, 0.42), (0.2, 0.70), (0.3, 0.72)):
        first = tracker.observe(observation("reused", offset, 0.5, y)) or first
    assert first is not None

    for offset, y in ((0.4, 0.40), (0.5, 0.42), (0.6, 0.70), (0.7, 0.72)):
        assert tracker.observe(observation("reused", offset, 0.5, y)) is None

    second = None
    for offset, y in ((2.4, 0.40), (2.5, 0.42), (2.6, 0.70), (2.7, 0.72)):
        second = tracker.observe(observation("reused", offset, 0.5, y)) or second

    assert second is not None
    assert second.session_id != first.session_id


def test_directed_line_accepts_outside_to_inside_crossing() -> None:
    tracker = EntryTracker(config(with_line=True))

    results = [
        tracker.observe(observation("entering", offset, 0.5, y))
        for offset, y in ((0.0, 0.40), (0.1, 0.42), (0.2, 0.60), (0.3, 0.70))
    ]

    assert results[:3] == [None, None, None]
    assert results[3] is not None
    assert results[3].method is EntryMethod.DIRECTED_LINE


def test_directed_line_rejects_inside_to_outside_crossing() -> None:
    tracker = EntryTracker(config(with_line=True))
    path = [(0.5, 0.70), (0.5, 0.65), (0.5, 0.46), (0.5, 0.40), (0.5, 0.20)]

    results = [
        tracker.observe(observation("exit", index / 10, x, y))
        for index, (x, y) in enumerate(path)
    ]

    assert results == [None] * len(path)


def test_directed_line_requires_door_zone_persistence() -> None:
    tracker = EntryTracker(config(with_line=True))
    # Os dois primeiros pontos estão no lado externo da linha, mas fora da
    # zona da porta; portanto não armam uma entrada.
    path = [(0.1, 0.40), (0.1, 0.42), (0.5, 0.60), (0.5, 0.70)]

    assert all(
        tracker.observe(observation("wrong-door", index / 10, x, y)) is None
        for index, (x, y) in enumerate(path)
    )


def test_directed_line_rejects_crossing_outside_segment() -> None:
    with pytest.raises(ValueError, match="não cobre"):
        config(
            with_line=True,
            door_zone=rectangle(0.80, 0.30, 0.98, 0.52),
            inside_zone=rectangle(0.80, 0.50, 0.98, 0.95),
        )


def test_calibration_rejects_inverted_line_direction() -> None:
    with pytest.raises(ValueError, match="door_zone"):
        config(
            with_line=True,
            entry_line=DirectedLine(
                Point(0.25, 0.50),
                Point(0.75, 0.50),
                "right",
            ),
        )


def test_polygon_rejects_self_intersection() -> None:
    with pytest.raises(ValueError, match="auto-intersectante"):
        Polygon(
            (
                Point(0.1, 0.1),
                Point(0.9, 0.1),
                Point(0.2, 0.8),
                Point(0.8, 0.8),
                Point(0.5, 0.3),
            )
        )


def test_directed_line_ignores_jitter_inside_deadband() -> None:
    tracker = EntryTracker(
        config(
            with_line=True,
            line_deadband=0.02,
            min_inside_observations=1,
        )
    )
    path = [(0.50, 0.40), (0.50, 0.42), (0.50, 0.495), (0.50, 0.505)]

    assert all(
        tracker.observe(observation("jitter", index / 10, x, y)) is None
        for index, (x, y) in enumerate(path)
    )


def test_directed_line_rejects_slow_transition() -> None:
    tracker = EntryTracker(
        config(
            with_line=True,
            max_transition=timedelta(seconds=1),
        )
    )

    assert tracker.observe(observation("slow-crossing", 0.0, 0.5, 0.40)) is None
    assert tracker.observe(observation("slow-crossing", 0.1, 0.5, 0.42)) is None
    assert tracker.observe(observation("slow-crossing", 1.2, 0.5, 0.60)) is None
    assert tracker.observe(observation("slow-crossing", 1.3, 0.5, 0.70)) is None


def test_geometry_uses_normalized_coordinates_and_includes_boundaries() -> None:
    zone = rectangle(0.2, 0.3, 0.8, 0.9)

    assert zone.contains(Point(0.5, 0.5))
    assert zone.contains(Point(0.2, 0.3))
    assert not zone.contains(Point(0.1, 0.5))
    with pytest.raises(ValueError, match="entre 0 e 1"):
        Point(1.01, 0.5)
    with pytest.raises(ValueError, match="área zero"):
        Polygon((Point(0.1, 0.1), Point(0.2, 0.2), Point(0.3, 0.3)))


def test_observations_must_be_timezone_aware_and_chronological() -> None:
    with pytest.raises(ValueError, match="fuso horário"):
        TrackObservation("track", datetime(2026, 7, 14, 12), Point(0.5, 0.5))

    tracker = EntryTracker(config())
    tracker.observe(observation("track", 1.0, 0.5, 0.40))
    with pytest.raises(ValueError, match="ordem cronológica"):
        tracker.observe(observation("track", 0.5, 0.5, 0.42))
