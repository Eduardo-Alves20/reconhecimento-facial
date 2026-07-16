from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.camera import CameraConfig
from app.vision.entry_tracker import (
    DirectedLine,
    EntryTracker,
    EntryTrackerConfig,
    Point,
    Polygon,
)
from app.vision.learned import LearnedGallery, ReferenceStatus
from app.vision.outbox import VisionEventOutbox
from app.vision.pipeline import (
    ConsensusPolicy,
    DetectionRegion,
    DoorEventProcessor,
    LearningPolicy,
    VisionProcessor,
)
from app.vision.recognizer import DetectedFace, MatchDecision, MatchStatus
from app.vision.worker import (
    VisionConfigurationError,
    VisionWorkerSettings,
    create_processor,
)


NOW = datetime(2026, 7, 14, 15, 0, tzinfo=UTC)


def _zones() -> tuple[Polygon, Polygon]:
    door = Polygon(
        (
            Point(0.0, 0.0),
            Point(0.49, 0.0),
            Point(0.49, 1.0),
            Point(0.0, 1.0),
        )
    )
    inside = Polygon(
        (
            Point(0.51, 0.0),
            Point(1.0, 0.0),
            Point(1.0, 1.0),
            Point(0.51, 1.0),
        )
    )
    return door, inside


def _entry_tracker() -> EntryTracker:
    door, inside = _zones()
    return EntryTracker(
        EntryTrackerConfig(
            camera_id="cam-ti-01",
            room_id="sala_ti_01",
            door_zone=door,
            inside_zone=inside,
            entry_line=DirectedLine(
                Point(0.5, 0.0),
                Point(0.5, 1.0),
                "right",
            ),
            min_door_observations=2,
            min_inside_observations=2,
            cooldown=timedelta(0),
        )
    )


class SequenceEngine:
    model_version = "insightface-test"
    model_fingerprint = "a" * 64

    class cv2:
        IMWRITE_JPEG_QUALITY = 1

        @staticmethod
        def imencode(extension, image, options):
            del extension, image, options
            return True, np.frombuffer(b"\xff\xd8\xff\x00\xff\xd9", dtype=np.uint8)

    def __init__(
        self,
        points: list[tuple[float, float]],
        identities: list[str],
    ) -> None:
        self.points = iter(points)
        self.identities = iter(identities)
        self.current_identity = ""

    def detect(self, frame):
        point = next(self.points)
        feature = np.zeros(512, dtype=np.float32)
        feature[0] = 1
        return [
            DetectedFace(
                (100, 80, 120, 120),
                point,
                0.99,
                0.90,
                feature,
            )
        ]

    def match(self, face):
        del face
        self.current_identity = next(self.identities)
        return MatchDecision(
            MatchStatus.MATCHED,
            self.current_identity,
            f"Pessoa {self.current_identity}",
            0.82,
            0.20,
            0.90,
        )

    @staticmethod
    def similarity(first, second):
        return float(np.dot(first, second))


class DecisionSequenceEngine(SequenceEngine):
    def __init__(
        self,
        points: list[tuple[float, float]],
        decisions: list[MatchDecision],
    ) -> None:
        super().__init__(points, [])
        self.decisions = iter(decisions)

    def match(self, face):
        del face
        return next(self.decisions)


class UnknownSequenceEngine(SequenceEngine):
    def __init__(
        self,
        points: list[tuple[float, float]],
        features: list[np.ndarray],
    ) -> None:
        super().__init__(points, [])
        self.features = iter(features)

    def detect(self, frame):
        del frame
        point = next(self.points)
        return [
            DetectedFace(
                (100, 80, 120, 120),
                point,
                0.99,
                0.90,
                next(self.features),
            )
        ]

    def match(self, face):
        del face
        return MatchDecision(MatchStatus.UNKNOWN, None, None, 0.42, 0.05, 0.90)


class FakeEvidenceStore:
    def save(self, scene, *, thumbnail, created_at):
        assert scene
        assert thumbnail
        assert created_at.tzinfo is not None
        return SimpleNamespace(reference="f" * 64)


class LearningSpy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def consider(self, engine, **values):
        del engine
        self.calls.append(values)
        return True


def _frame() -> np.ndarray:
    return np.full((480, 640, 3), 127, dtype=np.uint8)


def test_identity_handoff_resets_trajectory_and_never_learns_previous_person(
    tmp_path,
) -> None:
    points = [
        (0.40, 0.5),
        (0.41, 0.5),
        (0.42, 0.5),
        (0.43, 0.5),
        (0.44, 0.5),
        (0.42, 0.5),
        (0.44, 0.5),
        (0.54, 0.5),
        (0.56, 0.5),
    ]
    identities = ["EMP-A"] * 4 + ["EMP-B"] * 5
    engine = SequenceEngine(points, identities)
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    learned = LearningSpy()
    processor = VisionProcessor(
        engine,
        _entry_tracker(),
        outbox,
        evidence_store=FakeEvidenceStore(),
        learned=learned,
        learning_policy=LearningPolicy(enabled=True),
        decision_wait=timedelta(0),
    )

    emitted = []
    for index in range(len(points)):
        emitted.extend(
            processor.process_frame(
                _frame(),
                NOW + timedelta(milliseconds=250 * index),
            )
        )

    assert [event["user_id"] for event in emitted] == ["EMP-B"]
    assert [call["external_id"] for call in learned.calls] == ["EMP-B"]


def test_identity_established_after_crossing_cannot_rename_previous_person(
    tmp_path,
) -> None:
    points = [
        (0.42, 0.5),
        (0.44, 0.5),
        (0.54, 0.5),
        (0.56, 0.5),
        (0.58, 0.5),
        (0.59, 0.5),
        (0.60, 0.5),
        (0.61, 0.5),
    ]
    decisions = [
        MatchDecision(MatchStatus.MATCHED, "EMP-A", "Pessoa A", 0.82, 0.20, 0.90),
        *[
            MatchDecision(MatchStatus.LOW_QUALITY, None, None, None, None, 0.20)
            for _ in range(3)
        ],
        *[
            MatchDecision(MatchStatus.MATCHED, "EMP-B", "Pessoa B", 0.84, 0.22, 0.91)
            for _ in range(4)
        ],
    ]
    engine = DecisionSequenceEngine(points, decisions)
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    learned = LearningSpy()
    processor = VisionProcessor(
        engine,
        _entry_tracker(),
        outbox,
        evidence_store=FakeEvidenceStore(),
        learned=learned,
        learning_policy=LearningPolicy(enabled=True),
        decision_wait=timedelta(seconds=3),
    )

    emitted = []
    for index in range(len(points)):
        emitted.extend(
            processor.process_frame(
                _frame(),
                NOW + timedelta(milliseconds=250 * index),
            )
        )
    emitted.extend(processor.flush(NOW + timedelta(seconds=4)))

    assert len(emitted) == 1
    assert emitted[0]["identity_status"] == "AMBIGUOUS"
    assert emitted[0]["user_id"].startswith("AMBIGUOUS:")
    assert emitted[0]["evidence_captured_at"] == NOW.isoformat()
    assert learned.calls == []


def test_low_quality_handoff_cannot_inherit_previous_identity(tmp_path) -> None:
    points = [
        (0.40, 0.5),
        (0.41, 0.5),
        (0.42, 0.5),
        (0.44, 0.5),
        (0.54, 0.5),
        (0.56, 0.5),
    ]
    decisions = [
        *[
            MatchDecision(MatchStatus.MATCHED, "EMP-A", "Pessoa A", 0.82, 0.20, 0.90)
            for _ in range(4)
        ],
        *[
            MatchDecision(MatchStatus.LOW_QUALITY, None, None, None, None, 0.20)
            for _ in range(2)
        ],
    ]
    engine = DecisionSequenceEngine(points, decisions)
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    processor = VisionProcessor(
        engine,
        _entry_tracker(),
        outbox,
        decision_wait=timedelta(0),
    )

    emitted = []
    for index in range(len(points)):
        emitted.extend(
            processor.process_frame(
                _frame(),
                NOW + timedelta(milliseconds=250 * index),
            )
        )

    assert len(emitted) == 1
    assert emitted[0]["identity_status"] == "UNKNOWN"
    assert emitted[0]["user_id"].startswith("UNKNOWN:")


def test_fragmented_track_does_not_duplicate_same_presence(tmp_path) -> None:
    crossing = [(0.42, 0.5), (0.44, 0.5), (0.54, 0.5), (0.56, 0.5)]
    engine = SequenceEngine(crossing * 2, ["EMP001"] * 8)
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    processor = VisionProcessor(
        engine,
        _entry_tracker(),
        outbox,
        decision_wait=timedelta(0),
    )
    moments = [
        NOW,
        NOW + timedelta(milliseconds=250),
        NOW + timedelta(milliseconds=500),
        NOW + timedelta(milliseconds=750),
        NOW + timedelta(seconds=4),
        NOW + timedelta(seconds=4, milliseconds=250),
        NOW + timedelta(seconds=4, milliseconds=500),
        NOW + timedelta(seconds=4, milliseconds=750),
    ]

    emitted = []
    for moment in moments:
        emitted.extend(processor.process_frame(_frame(), moment))

    assert len(emitted) == 1
    assert outbox.counts() == {"PENDING": 1}


def test_similar_unknown_people_are_not_deduplicated(tmp_path) -> None:
    crossing = [(0.42, 0.5), (0.44, 0.5), (0.54, 0.5), (0.56, 0.5)]
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    second = np.zeros(512, dtype=np.float32)
    second[0] = 0.6
    second[1] = 0.8
    engine = UnknownSequenceEngine(
        crossing * 2,
        [first] * 4 + [second] * 4,
    )
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    processor = VisionProcessor(
        engine,
        _entry_tracker(),
        outbox,
        decision_wait=timedelta(0),
    )
    moments = [
        NOW,
        NOW + timedelta(milliseconds=250),
        NOW + timedelta(milliseconds=500),
        NOW + timedelta(milliseconds=750),
        NOW + timedelta(seconds=4),
        NOW + timedelta(seconds=4, milliseconds=250),
        NOW + timedelta(seconds=4, milliseconds=500),
        NOW + timedelta(seconds=4, milliseconds=750),
    ]

    emitted = []
    for moment in moments:
        emitted.extend(processor.process_frame(_frame(), moment))

    assert len(emitted) == 2
    assert {event["identity_status"] for event in emitted} == {"UNKNOWN"}
    assert outbox.counts() == {"PENDING": 2}


def test_learning_creates_only_a_quarantined_candidate(tmp_path) -> None:
    crossing = [(0.42, 0.5), (0.44, 0.5), (0.54, 0.5), (0.56, 0.5)]
    engine = SequenceEngine(crossing, ["EMP001"] * 4)
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    learned = LearnedGallery(tmp_path / "learned.db")
    learned.initialize()
    processor = VisionProcessor(
        engine,
        _entry_tracker(),
        outbox,
        evidence_store=FakeEvidenceStore(),
        learned=learned,
        learning_policy=LearningPolicy(enabled=True),
        decision_wait=timedelta(0),
    )

    for index in range(4):
        processor.process_frame(
            _frame(),
            NOW + timedelta(milliseconds=250 * index),
        )

    candidates = learned.list_references()
    assert len(candidates) == 1
    assert candidates[0].status == ReferenceStatus.PENDING


def test_door_observations_require_consecutive_frames(tmp_path) -> None:
    door, _ = _zones()
    points = [
        (0.20, 0.5),
        (0.23, 0.5),
        (0.80, 0.5),
        (0.20, 0.5),
        (0.23, 0.5),
        (0.26, 0.5),
    ]
    engine = SequenceEngine(points, ["EMP001"] * len(points))
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    processor = DoorEventProcessor(
        engine,
        door,
        outbox,
        camera_id="cam-ti-01",
        room_id="sala_ti_01",
        min_door_frames=3,
        minimum_movement=0.02,
        consensus_policy=ConsensusPolicy(
            history_size=8,
            minimum_matches=3,
            minimum_ratio=0.60,
            minimum_consecutive=2,
        ),
    )

    per_frame = [
        processor.process_frame(
            _frame(),
            NOW + timedelta(milliseconds=250 * index),
        )
        for index in range(len(points))
    ]

    assert all(not events for events in per_frame[:-1])
    assert len(per_frame[-1]) == 1


def test_door_mode_cannot_publish_without_explicit_override(tmp_path) -> None:
    with pytest.raises(VisionConfigurationError, match="apenas observacional"):
        VisionWorkerSettings(
            camera=CameraConfig("127.0.0.1", "operator", "secret"),
            camera_id="cam-ti-01",
            room_id="sala_ti_01",
            api_base_url="http://127.0.0.1:8000",
            api_key="camera-secret",
            gallery_manifest=tmp_path / "manifest.json",
            calibration_path=tmp_path / "calibration.json",
            models_dir=tmp_path / "models",
            model_fingerprint="a" * 64,
            outbox_path=tmp_path / "outbox.db",
            evidence_dir=tmp_path / "evidence",
            learned_path=tmp_path / "learned.db",
            dry_run=False,
            mode="door",
        )


def test_from_env_separates_dry_run_and_production_outboxes(
    tmp_path,
    monkeypatch,
) -> None:
    values = {
        "INTELBRAS_CAMERA_HOST": "127.0.0.1",
        "INTELBRAS_CAMERA_USER": "operator",
        "INTELBRAS_CAMERA_PASSWORD": "secret",
        "RAG_AUDIT_CAMERA_API_KEY": "camera-secret",
        "RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256": "a" * 64,
        "RAG_AUDIT_VISION_CAMERA_ID": "cam-ti-01",
        "RAG_AUDIT_VISION_MODE": "entry",
        "RAG_AUDIT_VISION_OUTBOX_PATH": "data/prod/{camera_id}.db",
        "RAG_AUDIT_VISION_DRY_RUN_OUTBOX_PATH": "data/dry/{camera_id}.db",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setenv("RAG_AUDIT_VISION_DRY_RUN", "true")
    dry = VisionWorkerSettings.from_env(tmp_path)
    monkeypatch.setenv("RAG_AUDIT_VISION_DRY_RUN", "false")
    production = VisionWorkerSettings.from_env(tmp_path)

    assert dry.outbox_path == (tmp_path / "data/dry/cam-ti-01.db").resolve()
    assert dry.lease_path == (tmp_path / "data/prod/cam-ti-01.db").resolve()
    assert production.outbox_path == (
        tmp_path / "data/prod/cam-ti-01.db"
    ).resolve()

    monkeypatch.setenv(
        "RAG_AUDIT_VISION_DRY_RUN_OUTBOX_PATH",
        "data/prod/{camera_id}.db",
    )
    with pytest.raises(VisionConfigurationError, match="precisam ser diferentes"):
        VisionWorkerSettings.from_env(tmp_path)

    monkeypatch.setenv(
        "RAG_AUDIT_VISION_DRY_RUN_OUTBOX_PATH",
        "data/dry/{camera_id}.db",
    )
    monkeypatch.setenv(
        "RAG_AUDIT_GALLERY_CACHE_PATH",
        "data/cache/shared.npz",
    )
    with pytest.raises(VisionConfigurationError, match="cache.*camera_id"):
        VisionWorkerSettings.from_env(tmp_path)


def test_roi_remaps_detection_and_limits_saved_scene(tmp_path) -> None:
    class RoiEngine(SequenceEngine):
        def __init__(self) -> None:
            self.detected_shape = None

        def detect(self, frame):
            self.detected_shape = frame.shape
            feature = np.zeros(512, dtype=np.float32)
            feature[0] = 1
            return [
                DetectedFace(
                    (10, 20, 100, 80),
                    (0.5, 0.5),
                    0.99,
                    0.90,
                    feature,
                )
            ]

    engine = RoiEngine()
    outbox = VisionEventOutbox(tmp_path / "outbox.db")
    outbox.initialize()
    processor = VisionProcessor(
        engine,
        _entry_tracker(),
        outbox,
        detection_region=DetectionRegion(0.25, 0.25, 0.75, 0.75),
    )
    frame = np.zeros((400, 800, 3), dtype=np.uint8)

    detected = processor._detect(frame)[0]
    saved_frame, saved_face = processor._evidence_view(frame, detected)

    assert engine.detected_shape == (200, 400, 3)
    assert detected.bbox == (210, 120, 100, 80)
    assert detected.centroid == (0.5, 0.5)
    assert saved_frame.shape == (200, 400, 3)
    assert saved_face.bbox == (10, 20, 100, 80)


def test_entry_mode_requires_a_directional_line(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_id": "cam-ti-01",
                "room_id": "sala-ti-01",
                "door_zone": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]],
                "inside_zone": [[0.6, 0], [1, 0], [1, 1], [0.6, 1]],
            }
        ),
        encoding="utf-8",
    )
    settings = VisionWorkerSettings(
        camera=CameraConfig("127.0.0.1", "operator", "secret"),
        camera_id="cam-ti-01",
        room_id="sala-ti-01",
        api_base_url="http://127.0.0.1:8000",
        api_key="camera-secret",
        gallery_manifest=tmp_path / "manifest.json",
        calibration_path=calibration,
        models_dir=tmp_path / "models",
        model_fingerprint="a" * 64,
        outbox_path=tmp_path / "outbox.db",
        evidence_dir=tmp_path / "evidence",
        learned_path=tmp_path / "learned.db",
    )

    with pytest.raises(VisionConfigurationError, match="entry_line"):
        create_processor(settings)
