from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.recognizer import (
    ArcFaceEngine,
    CandidateScore,
    DetectedFace,
    EMBEDDING_DIMENSION,
    FaceQualityPolicy,
    GalleryEntry,
    GalleryEnrollmentError,
    MatchStatus,
    MatchThresholds,
    VisionModelError,
    decide_match,
    evaluate_face_quality,
)
from app.vision.learned import LearnedGallery
from app.vision.worker import _refresh_learned_references


def engine(
    *,
    face_quality: float = 0.40,
    quality_policy: FaceQualityPolicy | None = None,
    det_size: tuple[int, int] = (640, 640),
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> ArcFaceEngine:
    instance = ArcFaceEngine.__new__(ArcFaceEngine)
    instance.np = np
    instance.thresholds = MatchThresholds(
        similarity=0.60,
        margin=0.10,
        face_quality=face_quality,
    )
    instance.quality_policy = quality_policy or FaceQualityPolicy()
    instance.det_size = det_size
    instance.providers = providers
    instance.model_fingerprint = "a" * 64
    instance.gallery = []
    instance._gallery_matrix = None
    instance._gallery_matrix_size = 0
    instance._gallery_lock = threading.RLock()
    instance._official_gallery = ()
    instance._learned_gallery_signature = None
    return instance


def embedding(index: int, scale: float = 1.0) -> np.ndarray:
    value = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    value[index] = scale
    return value


def detected(feature: np.ndarray, quality: float = 0.90) -> DetectedFace:
    return DetectedFace((10, 10, 120, 140), (0.4, 0.4), 0.99, quality, feature)


def test_engine_requires_a_verified_local_model_bundle() -> None:
    with pytest.raises(VisionModelError, match="bundle local"):
        ArcFaceEngine()
    with pytest.raises(ValueError, match="providers"):
        ArcFaceEngine(providers=[])
    with pytest.raises(ValueError, match="providers"):
        ArcFaceEngine(providers="CPUExecutionProvider")


def test_references_are_validated_and_normalized() -> None:
    recognizer = engine()
    recognizer.add_reference("EMP001", "Ana", embedding(0, scale=4.0))

    assert np.linalg.norm(recognizer.gallery[0].feature) == pytest.approx(1.0)
    assert not recognizer.gallery[0].feature.flags.writeable
    with pytest.raises(ValueError, match="512"):
        recognizer.add_reference("EMP002", "Bia", np.ones(128))
    with pytest.raises(ValueError, match="não finito"):
        invalid = embedding(1)
        invalid[2] = np.nan
        recognizer.add_reference("EMP002", "Bia", invalid)
    with pytest.raises(ValueError, match="nulo"):
        recognizer.add_reference("EMP002", "Bia", np.zeros(EMBEDDING_DIMENSION))


def test_rank_is_vectorized_sorted_and_collapsed_per_person() -> None:
    recognizer = engine()
    recognizer.add_reference("EMP001", "Ana", embedding(0))
    recognizer.add_reference(
        "EMP001",
        "Ana",
        embedding(0) * 0.9 + embedding(2) * 0.1,
    )
    recognizer.add_reference("EMP002", "Bia", embedding(1))
    query = detected(embedding(0) * 0.8 + embedding(1) * 0.2)

    ranked = recognizer.rank(query)

    assert [candidate.external_id for candidate in ranked] == ["EMP001", "EMP002"]
    assert ranked[0].similarity > ranked[1].similarity
    assert recognizer.rank(query, limit=1) == ranked[:1]
    assert recognizer.match(query).status == MatchStatus.MATCHED


def test_exact_tie_is_ambiguous_even_when_margin_threshold_is_zero() -> None:
    result = decide_match(
        [
            CandidateScore("EMP001", "Ana", 0.80),
            CandidateScore("EMP002", "Bia", 0.80),
        ],
        face_quality=0.90,
        thresholds=MatchThresholds(
            similarity=0.60,
            margin=0.0,
            face_quality=0.50,
        ),
    )

    assert result.status == MatchStatus.AMBIGUOUS
    assert result.external_id is None


def test_revoked_learned_reference_is_removed_without_restarting(tmp_path: Path) -> None:
    recognizer = engine()
    recognizer.add_reference("EMP001", "Ana", embedding(0))
    recognizer._official_gallery = tuple(recognizer.gallery)
    learned = LearnedGallery(tmp_path / "learned.db")
    learned.initialize()
    assert learned.consider(
        recognizer,
        external_id="EMP001",
        display_name="Ana",
        feature=embedding(1),
        evidence_ref="a" * 64,
        when=datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
        similarity=0.72,
        quality=0.91,
        min_similarity=0.60,
        max_similarity=0.85,
        min_quality=0.70,
    )
    candidate = learned.list_references()[0]
    learned.approve(candidate.reference_id, reviewed_by="auditor")

    assert _refresh_learned_references(recognizer, learned) == 1
    assert len(recognizer.gallery) == 2

    learned.revoke(
        candidate.reference_id,
        revoked_by="security-admin",
        reason="referência substituída",
    )
    assert _refresh_learned_references(recognizer, learned) == 0
    assert len(recognizer.gallery) == 1
    assert recognizer.official_external_ids == {"EMP001"}


def test_gallery_snapshot_remains_consistent_during_refresh() -> None:
    recognizer = engine()
    recognizer.add_reference("EMP001", "Ana", embedding(0))
    recognizer._official_gallery = tuple(recognizer.gallery)
    learned_entry = GalleryEntry("EMP001", "Ana", embedding(1))
    failures: list[Exception] = []

    def rank_repeatedly() -> None:
        try:
            for _ in range(300):
                recognizer.rank(detected(embedding(0)))
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=rank_repeatedly)
    thread.start()
    for index in range(100):
        recognizer.replace_learned_references(
            [learned_entry] if index % 2 else []
        )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []


def test_quality_combines_resolution_focus_exposure_and_pose() -> None:
    sharp = np.full((220, 220, 3), 128, dtype=np.uint8)
    yy, xx = np.indices((160, 140))
    pattern = np.where((xx // 4 + yy // 4) % 2, 40, 210).astype(np.uint8)
    sharp[30:190, 40:180] = pattern[..., None]
    landmarks = np.asarray(
        [[70, 80], [145, 80], [108, 115], [82, 155], [138, 155]],
        dtype=np.float64,
    )
    good = evaluate_face_quality(
        sharp,
        (40, 30, 140, 160),
        0.99,
        landmarks=landmarks,
        pose=(0, 0, 0),
        np_module=np,
    )
    dark_and_small = evaluate_face_quality(
        np.full((100, 100, 3), 3, dtype=np.uint8),
        (20, 20, 40, 40),
        0.99,
        np_module=np,
    )

    assert good.score > 0.85
    assert good.inter_eye_distance == pytest.approx(75.0)
    assert good.sharpness > 0.90
    assert dark_and_small.score < 0.25
    assert dark_and_small.size == 0.0
    assert dark_and_small.exposure == 0.0


def test_inter_eye_requirement_is_configurable_per_camera() -> None:
    frame = np.indices((160, 160)).sum(axis=0) % 2 * 160 + 40
    frame = np.repeat(frame[..., None], 3, axis=2).astype(np.uint8)
    landmarks = np.asarray(
        [[55, 60], [105, 60], [80, 85], [62, 115], [98, 115]],
        dtype=np.float64,
    )
    permissive = evaluate_face_quality(
        frame,
        (25, 20, 110, 125),
        0.99,
        landmarks=landmarks,
        pose=(0, 0, 0),
        policy=FaceQualityPolicy(
            min_inter_eye_pixels=30,
            full_inter_eye_pixels=50,
        ),
        np_module=np,
    )
    strict = evaluate_face_quality(
        frame,
        (25, 20, 110, 125),
        0.99,
        landmarks=landmarks,
        pose=(0, 0, 0),
        policy=FaceQualityPolicy(
            min_inter_eye_pixels=55,
            full_inter_eye_pixels=90,
        ),
        np_module=np,
    )

    assert permissive.size == 1.0
    assert strict.size == 0.0
    assert permissive.score > strict.score


def test_landmark_pose_fallback_uses_configured_limits() -> None:
    frame = np.indices((180, 180)).sum(axis=0) % 2 * 160 + 40
    frame = np.repeat(frame[..., None], 3, axis=2).astype(np.uint8)
    landmarks = np.asarray(
        [[50, 60], [110, 60], [102, 88], [62, 125], [108, 125]],
        dtype=np.float64,
    )
    strict = evaluate_face_quality(
        frame,
        (20, 20, 120, 130),
        0.99,
        landmarks=landmarks,
        policy=FaceQualityPolicy(max_yaw_degrees=10),
        np_module=np,
    )
    permissive = evaluate_face_quality(
        frame,
        (20, 20, 120, 130),
        0.99,
        landmarks=landmarks,
        policy=FaceQualityPolicy(max_yaw_degrees=80),
        np_module=np,
    )

    assert strict.pose < permissive.pose
    assert strict.score < permissive.score


def test_detect_penalizes_a_face_cut_by_the_frame_edge() -> None:
    recognizer = engine()
    frame = np.indices((200, 200)).sum(axis=0) % 2 * 160 + 40
    frame = np.repeat(frame[..., None], 3, axis=2).astype(np.uint8)
    raw_face = SimpleNamespace(
        bbox=np.asarray([-50.0, 30.0, 50.0, 170.0]),
        det_score=0.99,
        normed_embedding=embedding(0),
        kps=None,
        pose=None,
    )
    recognizer.app = SimpleNamespace(get=lambda _: [raw_face])

    clipped = recognizer.detect(frame)[0]

    raw_face.bbox = np.asarray([0.0, 30.0, 100.0, 170.0])
    internal = recognizer.detect(frame)[0]
    assert clipped.quality_metrics is not None
    assert internal.quality_metrics is not None
    assert clipped.quality_metrics.occlusion < internal.quality_metrics.occlusion
    assert clipped.quality < internal.quality


def test_gallery_rejects_the_same_image_for_different_people(tmp_path: Path) -> None:
    image = tmp_path / "face.jpg"
    image.write_bytes(b"same-image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    recognizer = engine()

    with pytest.raises(GalleryEnrollmentError, match="identidades diferentes"):
        recognizer._manifest_entries(
            tmp_path / "manifest.json",
            [
                {
                    "external_id": "EMP001",
                    "display_name": "Ana",
                    "image_path": image.name,
                    "image_sha256": digest,
                },
                {
                    "external_id": "EMP002",
                    "display_name": "Bia",
                    "image_path": image.name,
                    "image_sha256": digest,
                },
            ],
        )


def test_corrupt_embedding_cache_is_ignored(tmp_path: Path) -> None:
    recognizer = engine()
    cache = tmp_path / "embeddings.arcface.npz"
    np.savez(
        cache,
        fingerprint="expected",
        external_ids=np.asarray(["EMP001"]),
        display_names=np.asarray(["Ana"]),
        features=np.ones((1, 128), dtype=np.float32),
    )

    assert recognizer._load_cache(cache, "expected") is None


def test_gallery_checks_image_digest_and_reuses_valid_cache(tmp_path: Path) -> None:
    image = tmp_path / "face.jpg"
    image.write_bytes(b"valid-image-container-for-test")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "external_id": "EMP001",
                        "display_name": "Ana",
                        "image_path": "face.jpg",
                        "image_sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    recognizer = engine()
    recognizer.detect_image = lambda _: [detected(embedding(0))]

    assert recognizer.load_gallery(manifest) == 1
    assert (tmp_path / "embeddings.arcface.npz").is_file()

    cached = engine()

    def must_not_detect(_: object) -> list[DetectedFace]:
        raise AssertionError("O cache íntegro deveria evitar nova detecção.")

    cached.detect_image = must_not_detect
    assert cached.load_gallery(manifest) == 1
    assert cached.gallery[0].external_id == "EMP001"

    image.write_bytes(b"tampered")
    with pytest.raises(GalleryEnrollmentError, match="integridade"):
        cached.load_gallery(manifest)


def test_gallery_rejects_people_without_any_valid_reference(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "external_id": "EMP001",
                        "display_name": "Ana",
                        "image_path": first.name,
                        "image_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                    },
                    {
                        "external_id": "EMP002",
                        "display_name": "Bia",
                        "image_path": second.name,
                        "image_sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    recognizer = engine()
    recognizer.detect_image = lambda path: (
        [detected(embedding(0))] if Path(path).name == first.name else []
    )

    with pytest.raises(GalleryEnrollmentError, match="sem referência"):
        recognizer.load_gallery(manifest)


@pytest.mark.parametrize(
    "changed",
    (
        engine(face_quality=0.50),
        engine(
            quality_policy=FaceQualityPolicy(
                min_inter_eye_pixels=45,
                full_inter_eye_pixels=85,
            )
        ),
        engine(det_size=(800, 800)),
        engine(providers=("CUDAExecutionProvider", "CPUExecutionProvider")),
    ),
)
def test_gallery_cache_is_scoped_to_enrollment_policy(
    tmp_path: Path,
    changed: ArcFaceEngine,
) -> None:
    image = tmp_path / "face.jpg"
    image.write_bytes(b"valid-image-container-for-test")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "external_id": "EMP001",
                        "display_name": "Ana",
                        "image_path": "face.jpg",
                        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    original = engine()
    original.detect_image = lambda _: [detected(embedding(0))]
    assert original.load_gallery(manifest) == 1

    calls = 0

    def detect_again(_: object) -> list[DetectedFace]:
        nonlocal calls
        calls += 1
        return [detected(embedding(0))]

    changed.detect_image = detect_again
    assert changed.load_gallery(manifest) == 1
    assert calls == 1
