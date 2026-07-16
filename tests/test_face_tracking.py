from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.vision.face_tracking import FaceCentroidTracker, IdentityConsensus
from app.vision.recognizer import DetectedFace, MatchDecision, MatchStatus


NOW = datetime(2026, 7, 14, 15, 0, tzinfo=UTC)


def face(x: float, y: float) -> DetectedFace:
    return DetectedFace((0, 0, 100, 100), (x, y), 0.99, 0.90, feature=None)


def decision(person_id: str, score: float = 0.8) -> MatchDecision:
    return MatchDecision(
        MatchStatus.MATCHED, person_id, f"Pessoa {person_id}", score, 0.2, 0.9
    )


def unresolved(status: MatchStatus = MatchStatus.UNKNOWN) -> MatchDecision:
    return MatchDecision(status, None, None, 0.2, 0.05, 0.9)


def embedding(index: int) -> np.ndarray:
    value = np.zeros(512, dtype=np.float32)
    value[index] = 1.0
    return value


def positioned_face(x: float, feature: np.ndarray) -> DetectedFace:
    return DetectedFace(
        (int(x * 1_000) - 50, 100, 100, 100),
        (x, 0.3),
        0.99,
        0.90,
        feature,
    )


def low_quality_positioned_face(x: float, feature: np.ndarray) -> DetectedFace:
    return DetectedFace(
        (int(x * 1_000) - 50, 100, 100, 100),
        (x, 0.3),
        0.99,
        0.20,
        feature,
    )


def test_nearby_face_keeps_track_and_two_people_get_distinct_tracks() -> None:
    tracker = FaceCentroidTracker(boot_id="boot")
    first = tracker.update([face(0.2, 0.2), face(0.8, 0.2)], NOW)
    second = tracker.update(
        [face(0.22, 0.24), face(0.78, 0.24)], NOW + timedelta(milliseconds=250)
    )
    assert [item.track_id for item in first] == ["boot:face-1", "boot:face-2"]
    assert [item.track_id for item in second] == ["boot:face-1", "boot:face-2"]


def test_distant_or_expired_face_starts_new_track() -> None:
    tracker = FaceCentroidTracker(boot_id="boot", max_age=timedelta(seconds=1))
    assert tracker.update([face(0.1, 0.1)], NOW)[0].track_id == "boot:face-1"
    assert tracker.update([face(0.9, 0.9)], NOW + timedelta(seconds=2))[0].track_id == "boot:face-2"


def test_consensus_requires_multiple_matching_frames() -> None:
    consensus = IdentityConsensus(minimum_matches=3)
    consensus.add("track", decision("EMP001"))
    consensus.add("track", decision("EMP001"))
    assert consensus.resolve("track").status == MatchStatus.UNKNOWN
    consensus.add("track", decision("EMP001", 0.9))
    result = consensus.resolve("track", consume=True)
    assert result.status == MatchStatus.MATCHED
    assert result.external_id == "EMP001"
    assert result.supporting_frames == 3
    assert consensus.resolve("track").status == MatchStatus.UNKNOWN


def test_competing_identities_are_not_forced_to_one_person() -> None:
    consensus = IdentityConsensus(minimum_matches=3)
    for person_id in ("EMP001", "EMP002", "EMP001", "EMP002"):
        consensus.add("track", decision(person_id))
    result = consensus.resolve("track")
    assert result.status == MatchStatus.AMBIGUOUS
    assert result.external_id is None


def test_unknown_frames_count_against_consensus_support() -> None:
    consensus = IdentityConsensus(history_size=30, minimum_matches=2)
    for _ in range(28):
        consensus.add("track", unresolved())
    consensus.add("track", decision("EMP001"))
    consensus.add("track", decision("EMP001"))

    result = consensus.resolve("track")

    assert result.status == MatchStatus.UNKNOWN
    assert result.external_id is None


def test_consensus_requires_recent_consecutive_support() -> None:
    consensus = IdentityConsensus(
        history_size=8,
        minimum_matches=3,
        minimum_ratio=0.60,
        minimum_consecutive=2,
    )
    consensus.add("track", decision("EMP001"))
    consensus.add("track", decision("EMP001"))
    consensus.add("track", unresolved())
    consensus.add("track", decision("EMP001"))

    assert consensus.resolve("track").status == MatchStatus.UNKNOWN

    consensus.add("track", decision("EMP001"))
    assert consensus.resolve("track").status == MatchStatus.MATCHED


def test_low_quality_frames_break_recent_identity_support() -> None:
    consensus = IdentityConsensus(
        history_size=8,
        minimum_matches=3,
        minimum_ratio=0.60,
        minimum_consecutive=2,
    )
    for item in (
        decision("EMP001"),
        unresolved(MatchStatus.LOW_QUALITY),
        decision("EMP001"),
        unresolved(MatchStatus.LOW_QUALITY),
        decision("EMP001"),
    ):
        consensus.add("track", item)

    assert consensus.resolve("track").status == MatchStatus.UNKNOWN

    consensus.add("track", decision("EMP001"))
    consensus.add("track", decision("EMP001"))

    assert consensus.resolve("track").status == MatchStatus.MATCHED


def test_appearance_keeps_ids_when_two_faces_cross() -> None:
    tracker = FaceCentroidTracker(boot_id="boot", max_distance=0.5)
    first = tracker.update(
        [
            positioned_face(0.25, embedding(0)),
            positioned_face(0.75, embedding(1)),
        ],
        NOW,
    )
    crossing = tracker.update(
        [
            positioned_face(0.48, embedding(1)),
            positioned_face(0.52, embedding(0)),
        ],
        NOW + timedelta(seconds=1),
    )
    separated = tracker.update(
        [
            positioned_face(0.25, embedding(1)),
            positioned_face(0.75, embedding(0)),
        ],
        NOW + timedelta(seconds=2),
    )

    assert [item.track_id for item in first] == ["boot:face-1", "boot:face-2"]
    assert [item.track_id for item in crossing] == ["boot:face-2", "boot:face-1"]
    assert [item.track_id for item in separated] == ["boot:face-2", "boot:face-1"]


def test_low_quality_embedding_does_not_replace_track_appearance() -> None:
    tracker = FaceCentroidTracker(boot_id="boot")
    track_id = tracker.update([positioned_face(0.4, embedding(0))], NOW)[0].track_id

    [updated] = tracker.update(
        [low_quality_positioned_face(0.41, embedding(1))],
        NOW + timedelta(milliseconds=250),
    )

    assert updated.track_id == track_id
    assert tracker._tracks[track_id].appearance == tuple(embedding(0))


def test_tracker_can_discard_or_reset_state() -> None:
    tracker = FaceCentroidTracker(boot_id="boot")
    track_id = tracker.update([face(0.2, 0.2)], NOW)[0].track_id

    tracker.discard(track_id)
    assert tracker.active_track_ids == ()

    tracker.update([face(0.2, 0.2)], NOW + timedelta(seconds=1))
    tracker.reset()
    assert tracker.active_track_ids == ()
    tracker.update([face(0.2, 0.2)], NOW)


def test_observation_time_cannot_move_backwards() -> None:
    tracker = FaceCentroidTracker()
    tracker.update([face(0.1, 0.1)], NOW)
    with pytest.raises(ValueError, match="retroceder"):
        tracker.update([face(0.1, 0.1)], NOW - timedelta(milliseconds=1))


def test_naive_datetime_is_rejected() -> None:
    tracker = FaceCentroidTracker()
    with pytest.raises(ValueError, match="fuso"):
        tracker.update([face(0.1, 0.1)], datetime(2026, 7, 14, 12, 0))
