from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.vision.entry_tracker import (
    DirectedLine,
    EntryConfirmation,
    EntryMethod,
    EntryTracker,
    EntryTrackerConfig,
    Point,
    Polygon,
)
from app.vision.face_tracking import ConsensusResult
from app.vision.outbox import VisionEventOutbox
from app.vision.recognizer import DetectedFace, MatchDecision, MatchStatus
from app.vision.worker import (
    VisionProcessor,
    VisionConfigurationError,
    build_access_event,
    load_entry_tracker_config,
    safe_person_id,
)


NOW = datetime(2026, 7, 14, 18, 0, tzinfo=UTC)


def confirmation() -> EntryConfirmation:
    return EntryConfirmation(
        session_id="vis-abc123",
        camera_id="cam-ti-01",
        room_id="sala_ti_01",
        track_id="boot:face-1",
        started_at=NOW,
        confirmed_at=NOW,
        method=EntryMethod.DIRECTED_LINE,
        face_observation_ids=("obs-1", "obs-2"),
        observation_count=6,
    )


def test_matched_event_uses_visual_evidence_and_never_claims_door_grant() -> None:
    consensus = ConsensusResult(
        MatchStatus.MATCHED, "EMP001", "Lucas", 0.88, 0.20, 0.90, 4
    )
    event = build_access_event(
        confirmation(), consensus, model_version="opencv-test"
    )
    assert event["user_id"] == "EMP001"
    assert event["door_result"] == "NOT_REPORTED"
    assert event["entry_evidence"] == "VISION_LINE_CROSSING"
    assert event["recognition_confidence"] == 0.88


@pytest.mark.parametrize(
    ("status", "prefix"),
    [(MatchStatus.UNKNOWN, "UNKNOWN:"), (MatchStatus.AMBIGUOUS, "AMBIGUOUS:")],
)
def test_unresolved_identity_uses_unique_session_id(status, prefix) -> None:
    event = build_access_event(
        confirmation(),
        ConsensusResult(status, None, None, None, None, 0.4, 0),
        model_version="opencv-test",
    )
    assert event["user_id"].startswith(prefix)
    assert event["recognition_confidence"] is None


def test_external_id_with_unsafe_characters_is_pseudonymized_deterministically() -> None:
    first = safe_person_id("123.456.789-00 / externo")
    second = safe_person_id("123.456.789-00 / externo")
    assert first == second
    assert first.startswith("INTELBRAS:")
    assert len(first) < 100


def test_calibration_loader_rejects_camera_mismatch(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_id": "other-camera",
                "room_id": "sala_ti_01",
                "door_zone": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]],
                "inside_zone": [[0.6, 0], [1, 0], [1, 1], [0.6, 1]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VisionConfigurationError, match="outra câmera"):
        load_entry_tracker_config(path, camera_id="cam-ti-01", room_id="sala_ti_01")


def test_processor_emits_once_after_same_face_moves_from_door_to_inside(tmp_path) -> None:
    class FakeEngine:
        model_version = "fake-arcface"
        model_fingerprint = "a" * 64

        def __init__(self) -> None:
            self.points = iter(((0.44, 0.5), (0.45, 0.5), (0.54, 0.5), (0.55, 0.5)))

        def detect(self, frame):
            del frame
            point = next(self.points)
            return [DetectedFace((0, 0, 120, 120), point, 0.99, 0.9, None)]

        def match(self, face):
            del face
            return MatchDecision(
                MatchStatus.MATCHED, "EMP001", "Lucas", 0.85, 0.2, 0.9
            )

    door = Polygon((Point(0.0, 0.0), Point(0.49, 0.0), Point(0.49, 1.0), Point(0.0, 1.0)))
    inside = Polygon((Point(0.51, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.51, 1.0)))
    tracker = EntryTracker(
        EntryTrackerConfig(
            camera_id="cam-ti-01",
            room_id="sala_ti_01",
            door_zone=door,
            inside_zone=inside,
            entry_line=DirectedLine(Point(0.5, 0.0), Point(0.5, 1.0), "right"),
            min_door_observations=2,
            min_inside_observations=2,
        )
    )
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    processor = VisionProcessor(FakeEngine(), tracker, outbox)

    emitted = []
    for index in range(4):
        emitted.extend(processor.process_frame(object(), NOW + timedelta(milliseconds=250 * index)))

    assert len(emitted) == 1
    assert emitted[0]["user_id"] == "EMP001"
    assert outbox.counts() == {"PENDING": 1}
