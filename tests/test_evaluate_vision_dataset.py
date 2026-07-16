from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.recognizer import FaceQualityPolicy
from scripts.evaluate_vision_dataset import (
    EvaluationError,
    EvaluationManifest,
    EvaluationSample,
    NormalizedRoi,
    Probe,
    RankedCandidate,
    ThresholdPoint,
    build_parser,
    build_report,
    collect_samples,
    evaluate_grid,
    evaluate_threshold,
    load_evaluation_manifest,
    parse_detection_size,
    parse_grid,
    parse_providers,
    parse_roi,
    quality_policy_from_args,
    recommend_thresholds,
    wilson_upper_bound,
    write_csv_report,
    write_json_report,
)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_private_manifest(root: Path) -> Path:
    gallery = b'{"schema_version":1,"entries":[]}'
    known = b"known-private-image"
    unknown = b"unknown-private-image"
    (root / "gallery.json").write_bytes(gallery)
    (root / "known.jpg").write_bytes(known)
    (root / "unknown.jpg").write_bytes(unknown)
    payload = {
        "schema_version": 1,
        "gallery_manifest": "gallery.json",
        "gallery_manifest_sha256": digest(gallery),
        "probes": [
            {
                "kind": "known",
                "external_id": "EMP-PRIVATE-001",
                "image_path": "known.jpg",
                "image_sha256": digest(known),
            },
            {
                "kind": "unknown",
                "image_path": "unknown.jpg",
                "image_sha256": digest(unknown),
            },
        ],
    }
    path = root / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def synthetic_samples() -> list[EvaluationSample]:
    return [
        EvaluationSample(
            "known",
            "A",
            "ok",
            0.90,
            (RankedCandidate("A", 0.80), RankedCandidate("B", 0.50)),
        ),
        EvaluationSample(
            "known",
            "B",
            "ok",
            0.90,
            (RankedCandidate("A", 0.75), RankedCandidate("B", 0.60)),
        ),
        EvaluationSample("known", "C", "no_face", None, ()),
        EvaluationSample(
            "unknown",
            None,
            "ok",
            0.90,
            (RankedCandidate("A", 0.82), RankedCandidate("B", 0.65)),
        ),
        EvaluationSample(
            "unknown",
            None,
            "ok",
            0.90,
            (RankedCandidate("A", 0.72), RankedCandidate("B", 0.70)),
        ),
    ]


def test_manifest_validates_paths_hashes_and_known_unknown_split(tmp_path: Path) -> None:
    manifest = load_evaluation_manifest(write_private_manifest(tmp_path))

    assert len(manifest.probes) == 2
    assert {probe.kind for probe in manifest.probes} == {"known", "unknown"}
    assert manifest.gallery_manifest_sha256 == digest((tmp_path / "gallery.json").read_bytes())
    assert manifest.fingerprint == digest((tmp_path / "evaluation.json").read_bytes())


def test_manifest_rejects_tampered_image_and_path_traversal(tmp_path: Path) -> None:
    path = write_private_manifest(tmp_path)
    (tmp_path / "known.jpg").write_bytes(b"tampered")
    with pytest.raises(EvaluationError, match="integridade"):
        load_evaluation_manifest(path)

    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"outside")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["probes"][0]["image_path"] = "../outside.jpg"
    payload["probes"][0]["image_sha256"] = digest(b"outside")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationError, match="inseguro"):
        load_evaluation_manifest(path)


def test_manifest_rejects_duplicate_probe_content_under_another_path(
    tmp_path: Path,
) -> None:
    path = write_private_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    copied = (tmp_path / "known.jpg").read_bytes()
    (tmp_path / "unknown-copy.jpg").write_bytes(copied)
    payload["probes"][1]["image_path"] = "unknown-copy.jpg"
    payload["probes"][1]["image_sha256"] = digest(copied)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="conteúdo.*duplicado"):
        load_evaluation_manifest(path)


def test_metrics_cover_rank1_fpir_fnir_ambiguity_and_failures() -> None:
    row = evaluate_threshold(
        synthetic_samples(),
        ThresholdPoint(similarity=0.70, margin=0.08, quality=0.50),
    )

    assert row["rank1_rate"] == pytest.approx(1 / 3)
    assert row["true_identification_rate"] == pytest.approx(1 / 3)
    assert row["fnir"] == pytest.approx(2 / 3)
    assert row["unknown_comparison_probes"] == 2
    assert row["fpir_operational"] == pytest.approx(1 / 2)
    assert row["fpir_conditional"] == pytest.approx(1 / 2)
    assert row["fpir_conditional_upper_95"] > row["fpir_conditional"]
    assert row["ambiguous_rate"] == pytest.approx(1 / 5)
    assert row["misidentification_rate"] == pytest.approx(1 / 3)
    assert row["detection_failure_rate"] == pytest.approx(1 / 5)


def test_operational_and_comparison_conditional_fpir_are_separate() -> None:
    samples = [
        EvaluationSample("unknown", None, "no_face", None, ()),
        EvaluationSample("unknown", None, "multiple_faces", None, ()),
        EvaluationSample(
            "unknown",
            None,
            "ok",
            0.90,
            (RankedCandidate("A", 0.82), RankedCandidate("B", 0.60)),
        ),
        EvaluationSample(
            "unknown",
            None,
            "ok",
            0.90,
            (RankedCandidate("A", 0.62), RankedCandidate("B", 0.50)),
        ),
    ]

    row = evaluate_threshold(samples, ThresholdPoint(0.70, 0.08, 0.50))

    assert row["unknown_probes"] == 4
    assert row["unknown_comparison_probes"] == 2
    assert row["false_identifications"] == 1
    assert row["fpir_operational"] == pytest.approx(1 / 4)
    assert row["fpir_conditional"] == pytest.approx(1 / 2)
    assert row["fpir_conditional_upper_95"] > row["fpir_operational_upper_95"]


def test_equal_candidates_are_ambiguous_with_zero_margin_threshold() -> None:
    sample = EvaluationSample(
        "unknown",
        None,
        "ok",
        0.90,
        (
            RankedCandidate("EMP001", 0.80),
            RankedCandidate("EMP002", 0.80),
        ),
    )

    row = evaluate_threshold([sample], ThresholdPoint(0.70, 0.0, 0.50))

    assert row["false_identifications"] == 0
    assert row["ambiguous_probes"] == 1


def test_recommendations_never_exceed_fpir_ceiling() -> None:
    rows = evaluate_grid(
        synthetic_samples(),
        [
            ThresholdPoint(0.70, 0.08, 0.50),
            ThresholdPoint(0.85, 0.08, 0.50),
        ],
    )
    unsafe = {**rows[0], "fpir_conditional_upper_95": 0.02}
    safe = {**rows[1], "fpir_conditional_upper_95": 0.009}
    recommended = recommend_thresholds([unsafe, safe], max_fpir=0.01, limit=10)

    assert len(recommended) == 1
    assert recommended[0]["similarity_threshold"] == 0.85
    assert all(row["fpir_conditional_upper_95"] <= 0.01 for row in recommended)


def test_zero_observed_false_identifications_need_enough_unknown_probes() -> None:
    assert wilson_upper_bound(0, 10) > 0.01
    assert wilson_upper_bound(0, 500) < 0.01


def test_reports_are_aggregate_and_do_not_include_pii_or_image_paths(
    tmp_path: Path,
) -> None:
    private_id = "EMP-PRIVATE-001"
    private_path = tmp_path / "secret-person.jpg"
    manifest = EvaluationManifest(
        path=tmp_path / "evaluation.json",
        fingerprint="a" * 64,
        gallery_manifest=tmp_path / "gallery.json",
        gallery_manifest_sha256="b" * 64,
        probes=(
            Probe("known", private_path, "c" * 64, private_id),
            Probe("unknown", tmp_path / "visitor.jpg", "d" * 64, None),
        ),
    )
    rows = evaluate_grid(
        [
            EvaluationSample(
                "known",
                private_id,
                "ok",
                0.9,
                (RankedCandidate(private_id, 0.9),),
            ),
            EvaluationSample("unknown", None, "no_face", None, ()),
        ],
        [ThresholdPoint(0.7, 0.08, 0.5)],
    )
    recommendations = recommend_thresholds(rows, max_fpir=0.01, limit=10)
    report = build_report(
        manifest=manifest,
        model_version="test-model",
        model_fingerprint="e" * 64,
        sample_states={"ok": 1, "no_face": 1, "multiple_faces": 0},
        rows=rows,
        recommendations=recommendations,
        max_fpir=0.01,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        detection_size=(800, 640),
        quality_policy=FaceQualityPolicy(
            min_inter_eye_pixels=45,
            full_inter_eye_pixels=90,
        ),
        gallery_min_quality=0.55,
        roi=NormalizedRoi(0.1, 0.2, 0.8, 0.9),
    )
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    write_json_report(json_path, report)
    write_csv_report(csv_path, rows, recommendations)
    output = json_path.read_text(encoding="utf-8") + csv_path.read_text(encoding="utf-8")

    assert private_id not in output
    assert private_path.name not in output
    assert "recommended" in output
    assert report["evaluation_scope"] == {
        "type": "frame_level_1_to_n",
        "tracking": False,
        "temporal_consensus": False,
        "entry_geometry": False,
        "multiple_faces": "failure_to_acquire",
        "roi": {"left": 0.1, "top": 0.2, "right": 0.8, "bottom": 0.9},
    }
    assert report["engine_configuration"]["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert report["engine_configuration"]["detection_size"] == {
        "width": 800,
        "height": 640,
    }
    assert report["engine_configuration"]["gallery_min_quality"] == 0.55
    assert report["engine_configuration"]["quality_policy"]["min_inter_eye_pixels"] == 45
    assert report["recommendation_policy"]["criterion"] == ("fpir_conditional_upper_95")


def test_roi_is_applied_and_multiple_faces_remain_failure_to_acquire(
    tmp_path: Path,
) -> None:
    frame = np.zeros((400, 800, 3), dtype=np.uint8)

    class FakeEngine:
        gallery = [SimpleNamespace(external_id="EMP001")]

        def __init__(self) -> None:
            self.detected_shapes: list[tuple[int, ...]] = []

        def _read_image(self, path: Path) -> np.ndarray:
            assert path.is_file()
            return frame

        def detect(self, image: np.ndarray) -> list[object]:
            self.detected_shapes.append(image.shape)
            return [object(), object()]

    known_path = tmp_path / "known.jpg"
    unknown_path = tmp_path / "unknown.jpg"
    known_path.write_bytes(b"known")
    unknown_path.write_bytes(b"unknown")
    manifest = EvaluationManifest(
        path=tmp_path / "evaluation.json",
        fingerprint="a" * 64,
        gallery_manifest=tmp_path / "gallery.json",
        gallery_manifest_sha256="b" * 64,
        probes=(
            Probe("known", known_path, digest(b"known"), "EMP001"),
            Probe("unknown", unknown_path, digest(b"unknown"), None),
        ),
    )
    engine = FakeEngine()

    samples, states = collect_samples(
        engine,  # type: ignore[arg-type]
        manifest,
        roi=NormalizedRoi(0.25, 0.25, 0.75, 0.75),
    )

    assert engine.detected_shapes == [(200, 400, 3), (200, 400, 3)]
    assert [sample.detection_state for sample in samples] == [
        "multiple_faces",
        "multiple_faces",
    ]
    assert states == {"ok": 0, "no_face": 0, "multiple_faces": 2}


def test_parser_defaults_follow_worker_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_AUDIT_VISION_DETECTION_SIZE", "800x640")
    monkeypatch.setenv(
        "RAG_AUDIT_VISION_ONNX_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider",
    )
    monkeypatch.setenv("RAG_AUDIT_FACE_QUALITY_THRESHOLD", "0.55")
    monkeypatch.setenv("RAG_AUDIT_FACE_MIN_INTER_EYE_PIXELS", "45")
    monkeypatch.setenv("RAG_AUDIT_FACE_FULL_INTER_EYE_PIXELS", "90")
    monkeypatch.setenv("RAG_AUDIT_FACE_MIN_FOCUS_VARIANCE", "25")
    monkeypatch.setenv("RAG_AUDIT_FACE_FULL_FOCUS_VARIANCE", "300")
    monkeypatch.setenv("RAG_AUDIT_FACE_MAX_PITCH_DEGREES", "20")
    monkeypatch.setenv("RAG_AUDIT_FACE_MAX_YAW_DEGREES", "30")
    monkeypatch.setenv("RAG_AUDIT_FACE_MAX_ROLL_DEGREES", "18")

    args = build_parser().parse_args(["evaluation.json"])
    policy = quality_policy_from_args(args)

    assert parse_detection_size(args.det_size) == (800, 640)
    assert parse_providers(args.providers) == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert args.gallery_min_quality == 0.55
    assert policy.min_inter_eye_pixels == 45
    assert policy.full_inter_eye_pixels == 90
    assert policy.min_focus_variance == 25
    assert policy.full_focus_variance == 300
    assert policy.max_pitch_degrees == 20
    assert policy.max_yaw_degrees == 30
    assert policy.max_roll_degrees == 18


@pytest.mark.parametrize(
    "raw",
    ("640", "159x640", "640x2000", "640*640", "nanx640"),
)
def test_detection_size_rejects_values_outside_worker_contract(raw: str) -> None:
    with pytest.raises(EvaluationError):
        parse_detection_size(raw)


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "CPU",
        "CPUExecutionProvider,CPUExecutionProvider",
        "invalid provider",
    ),
)
def test_provider_parser_rejects_invalid_or_duplicate_names(raw: str) -> None:
    with pytest.raises(EvaluationError):
        parse_providers(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.1,0.2,0.8,0.9", NormalizedRoi(0.1, 0.2, 0.8, 0.9)),
        (None, None),
    ],
)
def test_parse_roi(raw: str | None, expected: NormalizedRoi | None) -> None:
    assert parse_roi(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ("", "0,0,1", "0.5,0.5,0.4,0.8", "-0.1,0,1,1", "texto"),
)
def test_parse_roi_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(EvaluationError):
        parse_roi(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.60,0.50,0.60", (0.5, 0.6)),
        ("0,1", (0.0, 1.0)),
    ],
)
def test_parse_grid_normalizes_values(raw: str, expected: tuple[float, ...]) -> None:
    assert parse_grid(raw, name="grid") == expected


@pytest.mark.parametrize("raw", ("", "-0.1", "1.1", "nan", "texto"))
def test_parse_grid_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(EvaluationError):
        parse_grid(raw, name="grid")
