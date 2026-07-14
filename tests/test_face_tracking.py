from __future__ import annotations

from datetime import UTC, datetime, timedelta

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


def test_naive_datetime_is_rejected() -> None:
    tracker = FaceCentroidTracker()
    with pytest.raises(ValueError, match="fuso"):
        tracker.update([face(0.1, 0.1)], datetime(2026, 7, 14, 12, 0))
