from __future__ import annotations

import pytest

from app.vision.recognizer import (
    CandidateScore,
    MatchStatus,
    MatchThresholds,
    decide_match,
)


THRESHOLDS = MatchThresholds(similarity=0.60, margin=0.10, face_quality=0.50)


def candidate(person_id: str, score: float) -> CandidateScore:
    return CandidateScore(person_id, f"Pessoa {person_id}", score)


def test_accepts_only_clear_candidate_above_calibrated_threshold() -> None:
    result = decide_match(
        [candidate("EMP001", 0.82), candidate("EMP002", 0.51)],
        face_quality=0.90,
        thresholds=THRESHOLDS,
    )
    assert result.status == MatchStatus.MATCHED
    assert result.external_id == "EMP001"
    assert result.margin == pytest.approx(0.31)


def test_does_not_force_best_candidate_below_threshold() -> None:
    result = decide_match(
        [candidate("EMP001", 0.58), candidate("EMP002", 0.20)],
        face_quality=0.90,
        thresholds=THRESHOLDS,
    )
    assert result.status == MatchStatus.UNKNOWN
    assert result.external_id is None


def test_close_candidates_are_ambiguous_without_identity_assignment() -> None:
    result = decide_match(
        [candidate("EMP001", 0.82), candidate("EMP002", 0.77)],
        face_quality=0.90,
        thresholds=THRESHOLDS,
    )
    assert result.status == MatchStatus.AMBIGUOUS
    assert result.external_id is None
    assert result.margin == pytest.approx(0.05)


def test_low_quality_face_is_rejected_before_matching() -> None:
    result = decide_match(
        [candidate("EMP001", 0.99)],
        face_quality=0.20,
        thresholds=THRESHOLDS,
    )
    assert result.status == MatchStatus.LOW_QUALITY
    assert result.external_id is None


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        MatchThresholds(similarity=1.5)
