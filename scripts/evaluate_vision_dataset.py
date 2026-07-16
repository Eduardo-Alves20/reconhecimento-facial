from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import itertools
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env.vision")
load_dotenv(PROJECT_ROOT / ".env")

from app.vision.recognizer import (  # noqa: E402
    ArcFaceEngine,
    FaceQualityPolicy,
    GalleryEnrollmentError,
    MatchThresholds,
    VisionDependencyError,
    VisionModelError,
)
from scripts.verify_vision_models import ModelBundleError, verify_bundle  # noqa: E402


MAX_MANIFEST_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_PROBES = 50_000
SHA256_CHARS = frozenset("0123456789abcdef")
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
PROVIDER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*ExecutionProvider")


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Probe:
    kind: str
    image_path: Path
    image_sha256: str
    external_id: str | None


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    path: Path
    fingerprint: str
    gallery_manifest: Path
    gallery_manifest_sha256: str
    probes: tuple[Probe, ...]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    external_id: str
    similarity: float


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    kind: str
    expected_external_id: str | None
    detection_state: str
    quality: float | None
    candidates: tuple[RankedCandidate, ...]


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    similarity: float
    margin: float
    quality: float


@dataclass(frozen=True, slots=True)
class NormalizedRoi:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise EvaluationError("A ROI deve usar valores normalizados entre 0 e 1.")
        if self.left >= self.right or self.top >= self.bottom:
            raise EvaluationError("A ROI possui área inválida.")

    def bounds(self, frame: Any) -> tuple[int, int, int, int]:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise EvaluationError("A imagem do probe possui formato inválido.")
        height, width = (int(value) for value in frame.shape[:2])
        if width < 1 or height < 1:
            raise EvaluationError("A imagem do probe está vazia.")
        left = max(0, min(width - 1, math.floor(self.left * width)))
        top = max(0, min(height - 1, math.floor(self.top * height)))
        right = max(left + 1, min(width, math.ceil(self.right * width)))
        bottom = max(top + 1, min(height, math.ceil(self.bottom * height)))
        return left, top, right, bottom


def parse_detection_size(value: Any) -> tuple[int, int]:
    raw = str(value).strip().lower()
    match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", raw)
    if match is None:
        raise EvaluationError("det_size deve usar o formato 640x640.")
    width, height = (int(item) for item in match.groups())
    if not 320 <= width <= 1920 or not 320 <= height <= 1920:
        raise EvaluationError("det_size deve ficar entre 320 e 1920 pixels.")
    return width, height


def parse_providers(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        raw_values = [os.getenv("RAG_AUDIT_VISION_ONNX_PROVIDERS", "CPUExecutionProvider")]
    elif isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)
    providers = tuple(
        provider.strip()
        for raw in raw_values
        for provider in str(raw).split(",")
        if provider.strip()
    )
    if (
        not providers
        or len(providers) > 4
        or len(set(providers)) != len(providers)
        or any(
            len(provider) > 128 or PROVIDER_RE.fullmatch(provider) is None for provider in providers
        )
    ):
        raise EvaluationError("A lista de providers ONNX Runtime é inválida.")
    return providers


def parse_roi(value: str | None) -> NormalizedRoi | None:
    if value is None:
        return None
    try:
        parts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise EvaluationError("A ROI deve usar left,top,right,bottom.") from exc
    if len(parts) != 4:
        raise EvaluationError("A ROI deve usar left,top,right,bottom.")
    return NormalizedRoi(*parts)


def _bounded(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{name} deve ser numérico.") from exc
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise EvaluationError(f"{name} deve ficar entre {minimum} e {maximum}.")
    return numeric


def quality_policy_from_args(args: argparse.Namespace) -> FaceQualityPolicy:
    return FaceQualityPolicy(
        min_inter_eye_pixels=_bounded(
            args.min_inter_eye_pixels,
            name="min_inter_eye_pixels",
            minimum=10,
            maximum=300,
        ),
        full_inter_eye_pixels=_bounded(
            args.full_inter_eye_pixels,
            name="full_inter_eye_pixels",
            minimum=20,
            maximum=500,
        ),
        min_focus_variance=_bounded(
            args.min_focus_variance,
            name="min_focus_variance",
            minimum=1,
            maximum=1000,
        ),
        full_focus_variance=_bounded(
            args.full_focus_variance,
            name="full_focus_variance",
            minimum=2,
            maximum=5000,
        ),
        max_pitch_degrees=_bounded(
            args.max_pitch_degrees,
            name="max_pitch_degrees",
            minimum=1,
            maximum=90,
        ),
        max_yaw_degrees=_bounded(
            args.max_yaw_degrees,
            name="max_yaw_degrees",
            minimum=1,
            maximum=90,
        ),
        max_roll_degrees=_bounded(
            args.max_roll_degrees,
            name="max_roll_degrees",
            minimum=1,
            maximum=90,
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in SHA256_CHARS for character in value)
    )


def _safe_file(
    root: Path,
    relative_text: Any,
    *,
    expected_hash: Any,
    image: bool,
) -> tuple[Path, str]:
    if not isinstance(relative_text, str) or not relative_text.strip():
        raise EvaluationError("O manifesto contém um caminho vazio.")
    relative = Path(relative_text.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationError("O manifesto contém um caminho inseguro.")
    target = root / relative
    if target.is_symlink():
        raise EvaluationError("Links simbólicos não são aceitos no dataset.")
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvaluationError("Um arquivo do dataset está ausente ou fora da raiz.") from exc
    if not resolved.is_file():
        raise EvaluationError("Uma entrada do dataset não é um arquivo.")
    if image:
        if resolved.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise EvaluationError("O dataset contém um formato de imagem não suportado.")
        if resolved.stat().st_size > MAX_IMAGE_BYTES:
            raise EvaluationError("Uma imagem do dataset excede o limite de tamanho.")
    if not _is_sha256(expected_hash):
        raise EvaluationError("O manifesto contém um SHA-256 inválido.")
    actual_hash = sha256_file(resolved)
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise EvaluationError("A integridade de um arquivo do dataset não confere.")
    return resolved, actual_hash


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise EvaluationError("O manifesto de avaliação não pode ser um link.")
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
        if resolved_manifest.stat().st_size > MAX_MANIFEST_BYTES:
            raise EvaluationError("O manifesto de avaliação excede o limite de tamanho.")
        raw_bytes = resolved_manifest.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError("Manifesto de avaliação não encontrado.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError("Não foi possível ler o manifesto de avaliação.") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "gallery_manifest",
        "gallery_manifest_sha256",
        "probes",
    }:
        raise EvaluationError("Estrutura do manifesto de avaliação inválida.")
    if payload["schema_version"] != 1 or not isinstance(payload["probes"], list):
        raise EvaluationError("Versão do manifesto de avaliação não suportada.")
    if not 2 <= len(payload["probes"]) <= MAX_PROBES:
        raise EvaluationError("O dataset deve conter entre 2 e 50.000 probes.")

    root = resolved_manifest.parent.resolve(strict=True)
    gallery_manifest, gallery_hash = _safe_file(
        root,
        payload["gallery_manifest"],
        expected_hash=payload["gallery_manifest_sha256"],
        image=False,
    )
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    probes: list[Probe] = []
    kinds: set[str] = set()
    for item in payload["probes"]:
        if not isinstance(item, dict):
            raise EvaluationError("O manifesto contém um probe inválido.")
        kind = item.get("kind")
        expected_fields = (
            {"kind", "image_path", "image_sha256", "external_id"}
            if kind == "known"
            else {"kind", "image_path", "image_sha256"}
        )
        if kind not in {"known", "unknown"} or set(item) != expected_fields:
            raise EvaluationError("A estrutura de um probe é inválida.")
        external_id = item.get("external_id")
        if kind == "known" and (
            not isinstance(external_id, str) or not external_id.strip() or len(external_id) > 128
        ):
            raise EvaluationError("Um probe conhecido não possui ID válido.")
        image_path, image_hash = _safe_file(
            root,
            item["image_path"],
            expected_hash=item["image_sha256"],
            image=True,
        )
        if image_path in seen_paths:
            raise EvaluationError("O manifesto contém uma imagem duplicada.")
        if image_hash in seen_hashes:
            raise EvaluationError("O manifesto contém conteúdo de probe duplicado.")
        seen_paths.add(image_path)
        seen_hashes.add(image_hash)
        kinds.add(kind)
        probes.append(
            Probe(
                kind=kind,
                image_path=image_path,
                image_sha256=image_hash,
                external_id=external_id.strip() if isinstance(external_id, str) else None,
            )
        )
    if kinds != {"known", "unknown"}:
        raise EvaluationError("O dataset precisa de probes conhecidos e desconhecidos.")
    return EvaluationManifest(
        path=resolved_manifest,
        fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
        gallery_manifest=gallery_manifest,
        gallery_manifest_sha256=gallery_hash,
        probes=tuple(probes),
    )


def parse_grid(value: str, *, name: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise EvaluationError(f"{name} contém um número inválido.") from exc
    if not values or any(not math.isfinite(item) or not 0 <= item <= 1 for item in values):
        raise EvaluationError(f"{name} deve conter valores entre 0 e 1.")
    return values


def threshold_grid(
    similarities: Iterable[float],
    margins: Iterable[float],
    qualities: Iterable[float],
) -> tuple[ThresholdPoint, ...]:
    return tuple(
        ThresholdPoint(*values) for values in itertools.product(similarities, margins, qualities)
    )


def _decision(sample: EvaluationSample, threshold: ThresholdPoint) -> tuple[str, str | None]:
    if sample.detection_state != "ok" or sample.quality is None:
        return "NO_FACE", None
    if sample.quality < threshold.quality:
        return "LOW_QUALITY", None
    if not sample.candidates:
        return "UNKNOWN", None
    best = sample.candidates[0]
    if best.similarity < threshold.similarity:
        return "UNKNOWN", None
    if len(sample.candidates) > 1:
        margin = best.similarity - sample.candidates[1].similarity
        if margin <= threshold.margin:
            return "AMBIGUOUS", None
    return "MATCHED", best.external_id


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_upper_bound(successes: int, trials: int, *, z_score: float = 1.96) -> float | None:
    if (
        trials < 0
        or successes < 0
        or successes > trials
        or not math.isfinite(z_score)
        or z_score <= 0
    ):
        raise EvaluationError("Contagens inválidas para o intervalo de Wilson.")
    if trials == 0:
        return None
    proportion = successes / trials
    z_squared = z_score * z_score
    denominator = 1 + z_squared / trials
    center = proportion + z_squared / (2 * trials)
    distance = z_score * math.sqrt(
        proportion * (1 - proportion) / trials + z_squared / (4 * trials * trials)
    )
    return min(1.0, (center + distance) / denominator)


def evaluate_threshold(
    samples: Sequence[EvaluationSample], threshold: ThresholdPoint
) -> dict[str, Any]:
    known = sum(sample.kind == "known" for sample in samples)
    unknown = len(samples) - known
    unknown_comparisons = sum(
        sample.kind == "unknown"
        and sample.detection_state == "ok"
        and sample.quality is not None
        and sample.quality >= threshold.quality
        for sample in samples
    )
    raw_rank1 = sum(
        sample.kind == "known"
        and bool(sample.candidates)
        and sample.candidates[0].external_id == sample.expected_external_id
        for sample in samples
    )
    correct = false_identifications = misidentifications = ambiguous = 0
    low_quality = detection_failures = 0
    for sample in samples:
        status, matched_id = _decision(sample, threshold)
        if status == "AMBIGUOUS":
            ambiguous += 1
        elif status == "LOW_QUALITY":
            low_quality += 1
        elif status == "NO_FACE":
            detection_failures += 1
        if sample.kind == "known":
            if status == "MATCHED" and matched_id == sample.expected_external_id:
                correct += 1
            elif status == "MATCHED":
                misidentifications += 1
        elif status == "MATCHED":
            false_identifications += 1

    return {
        "similarity_threshold": threshold.similarity,
        "margin_threshold": threshold.margin,
        "quality_threshold": threshold.quality,
        "known_probes": known,
        "unknown_probes": unknown,
        "rank1_rate": _rate(raw_rank1, known),
        "true_identification_rate": _rate(correct, known),
        "fnir": _rate(known - correct, known),
        "unknown_comparison_probes": unknown_comparisons,
        "fpir_operational": _rate(false_identifications, unknown),
        "fpir_operational_upper_95": wilson_upper_bound(
            false_identifications,
            unknown,
        ),
        "fpir_conditional": _rate(false_identifications, unknown_comparisons),
        "fpir_conditional_upper_95": wilson_upper_bound(
            false_identifications,
            unknown_comparisons,
        ),
        "ambiguous_rate": _rate(ambiguous, len(samples)),
        "misidentification_rate": _rate(misidentifications, known),
        "low_quality_rate": _rate(low_quality, len(samples)),
        "detection_failure_rate": _rate(detection_failures, len(samples)),
        "correct_identifications": correct,
        "false_identifications": false_identifications,
        "misidentifications": misidentifications,
        "ambiguous_probes": ambiguous,
    }


def evaluate_grid(
    samples: Sequence[EvaluationSample], points: Sequence[ThresholdPoint]
) -> list[dict[str, Any]]:
    if not samples:
        raise EvaluationError("Não há amostras para avaliar.")
    if not points:
        raise EvaluationError("A grade de limiares está vazia.")
    return [evaluate_threshold(samples, point) for point in points]


def recommend_thresholds(
    rows: Sequence[dict[str, Any]],
    *,
    max_fpir: float,
    limit: int,
) -> list[dict[str, Any]]:
    if not math.isfinite(max_fpir) or not 0 <= max_fpir <= 1:
        raise EvaluationError("max_fpir deve estar entre 0 e 1.")
    if limit < 1 or limit > 100:
        raise EvaluationError("O limite de recomendações deve estar entre 1 e 100.")
    eligible = [
        row
        for row in rows
        if row.get("fpir_conditional_upper_95") is not None
        and float(row["fpir_conditional_upper_95"]) <= max_fpir
    ]
    eligible.sort(
        key=lambda row: (
            float(row["fnir"]),
            float(row["misidentification_rate"]),
            float(row["ambiguous_rate"]),
            -float(row["similarity_threshold"]),
            -float(row["margin_threshold"]),
            -float(row["quality_threshold"]),
        )
    )
    return eligible[:limit]


def _detect_probe(
    engine: ArcFaceEngine,
    image_path: Path,
    roi: NormalizedRoi | None,
) -> list[Any]:
    if roi is None:
        return engine.detect_image(image_path)
    frame = engine._read_image(image_path)
    left, top, right, bottom = roi.bounds(frame)
    return engine.detect(frame[top:bottom, left:right])


def collect_samples(
    engine: ArcFaceEngine,
    manifest: EvaluationManifest,
    *,
    roi: NormalizedRoi | None = None,
) -> tuple[list[EvaluationSample], dict[str, int]]:
    gallery_ids = {entry.external_id for entry in engine.gallery}
    missing_ids = {
        probe.external_id
        for probe in manifest.probes
        if probe.kind == "known" and probe.external_id not in gallery_ids
    }
    if missing_ids:
        raise EvaluationError(
            f"{len(missing_ids)} identidade(s) conhecida(s) não estão na galeria carregada."
        )

    samples: list[EvaluationSample] = []
    states = {"ok": 0, "no_face": 0, "multiple_faces": 0}
    for probe in manifest.probes:
        faces = _detect_probe(engine, probe.image_path, roi)
        if len(faces) != 1:
            state = "no_face" if not faces else "multiple_faces"
            states[state] += 1
            samples.append(EvaluationSample(probe.kind, probe.external_id, state, None, ()))
            continue
        face = faces[0]
        ranked = tuple(
            RankedCandidate(item.external_id, float(item.similarity))
            for item in engine.rank(face, limit=2)
        )
        states["ok"] += 1
        samples.append(
            EvaluationSample(
                probe.kind,
                probe.external_id,
                "ok",
                float(face.quality),
                ranked,
            )
        )
    return samples, states


def build_report(
    *,
    manifest: EvaluationManifest,
    model_version: str,
    model_fingerprint: str,
    sample_states: dict[str, int],
    rows: Sequence[dict[str, Any]],
    recommendations: Sequence[dict[str, Any]],
    max_fpir: float,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    detection_size: tuple[int, int] = (640, 640),
    quality_policy: FaceQualityPolicy | None = None,
    gallery_min_quality: float = 0.50,
    roi: NormalizedRoi | None = None,
) -> dict[str, Any]:
    known = sum(probe.kind == "known" for probe in manifest.probes)
    policy = quality_policy or FaceQualityPolicy()
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_fingerprint": manifest.fingerprint,
        "gallery_manifest_sha256": manifest.gallery_manifest_sha256,
        "model": {
            "version": model_version,
            "fingerprint": model_fingerprint,
        },
        "engine_configuration": {
            "providers": list(providers),
            "detection_size": {
                "width": detection_size[0],
                "height": detection_size[1],
            },
            "gallery_min_quality": gallery_min_quality,
            "quality_policy": asdict(policy),
        },
        "evaluation_scope": {
            "type": "frame_level_1_to_n",
            "tracking": False,
            "temporal_consensus": False,
            "entry_geometry": False,
            "multiple_faces": "failure_to_acquire",
            "roi": asdict(roi) if roi is not None else None,
        },
        "counts": {
            "probes": len(manifest.probes),
            "known": known,
            "unknown": len(manifest.probes) - known,
            "detection_states": sample_states,
        },
        "recommendation_policy": {
            "max_fpir": max_fpir,
            "criterion": "fpir_conditional_upper_95",
            "eligible_points": sum(
                row["fpir_conditional_upper_95"] is not None
                and row["fpir_conditional_upper_95"] <= max_fpir
                for row in rows
            ),
        },
        "recommendations": list(recommendations),
        "grid": list(rows),
    }


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    if path.is_symlink():
        raise EvaluationError("O relatório JSON não pode apontar para um link.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv_report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    recommendations: Sequence[dict[str, Any]],
) -> None:
    if path.is_symlink():
        raise EvaluationError("O relatório CSV não pode apontar para um link.")
    path.parent.mkdir(parents=True, exist_ok=True)
    recommended = {
        (
            row["similarity_threshold"],
            row["margin_threshold"],
            row["quality_threshold"],
        )
        for row in recommendations
    }
    fieldnames = [*rows[0].keys(), "recommended"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            key = (
                row["similarity_threshold"],
                row["margin_threshold"],
                row["quality_threshold"],
            )
            writer.writerow({**row, "recommended": key in recommended})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia reconhecimento facial em um dataset privado e íntegro."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(
            os.getenv(
                "RAG_AUDIT_VISION_MODELS_DIR",
                str(PROJECT_ROOT / "data" / "vision-models"),
            )
        ),
    )
    parser.add_argument(
        "--model-fingerprint",
        default=os.getenv("RAG_AUDIT_VISION_MODEL_BUNDLE_SHA256", ""),
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="Provider ONNX Runtime. Pode ser repetido.",
    )
    parser.add_argument(
        "--det-size",
        default=os.getenv("RAG_AUDIT_VISION_DETECTION_SIZE", "640x640"),
        help="Tamanho do detector no formato largura x altura.",
    )
    parser.add_argument(
        "--gallery-min-quality",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_QUALITY_THRESHOLD", "0.50"),
    )
    parser.add_argument(
        "--min-inter-eye-pixels",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_MIN_INTER_EYE_PIXELS", "40"),
    )
    parser.add_argument(
        "--full-inter-eye-pixels",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_FULL_INTER_EYE_PIXELS", "80"),
    )
    parser.add_argument(
        "--min-focus-variance",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_MIN_FOCUS_VARIANCE", "20"),
    )
    parser.add_argument(
        "--full-focus-variance",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_FULL_FOCUS_VARIANCE", "260"),
    )
    parser.add_argument(
        "--max-pitch-degrees",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_MAX_PITCH_DEGREES", "25"),
    )
    parser.add_argument(
        "--max-yaw-degrees",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_MAX_YAW_DEGREES", "35"),
    )
    parser.add_argument(
        "--max-roll-degrees",
        type=float,
        default=os.getenv("RAG_AUDIT_FACE_MAX_ROLL_DEGREES", "25"),
    )
    parser.add_argument(
        "--roi",
        help="ROI normalizada opcional: left,top,right,bottom.",
    )
    parser.add_argument(
        "--similarity-grid",
        default="0.45,0.50,0.55,0.60,0.65",
    )
    parser.add_argument("--margin-grid", default="0.05,0.08,0.10,0.12")
    parser.add_argument("--quality-grid", default="0.35,0.40,0.50,0.60")
    parser.add_argument("--max-fpir", type=float, default=0.01)
    parser.add_argument("--max-recommendations", type=int, default=10)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("vision-evaluation.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("vision-evaluation.csv"),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    gallery_min_quality = _bounded(
        args.gallery_min_quality,
        name="gallery_min_quality",
        minimum=0,
        maximum=1,
    )
    detection_size = parse_detection_size(args.det_size)
    providers = parse_providers(args.providers)
    quality_policy = quality_policy_from_args(args)
    roi = parse_roi(args.roi)
    if args.json_output.resolve() == args.csv_output.resolve():
        raise EvaluationError("Os relatórios JSON e CSV precisam de caminhos distintos.")
    manifest = load_evaluation_manifest(args.manifest)
    fingerprint = verify_bundle(args.models_root, args.model_fingerprint)
    engine = ArcFaceEngine(
        thresholds=MatchThresholds(face_quality=gallery_min_quality),
        quality_policy=quality_policy,
        model_root=args.models_root,
        model_fingerprint=fingerprint,
        det_size=detection_size,
        providers=providers,
    )
    engine.load_gallery(manifest.gallery_manifest)
    samples, states = collect_samples(engine, manifest, roi=roi)
    points = threshold_grid(
        parse_grid(args.similarity_grid, name="similarity_grid"),
        parse_grid(args.margin_grid, name="margin_grid"),
        parse_grid(args.quality_grid, name="quality_grid"),
    )
    rows = evaluate_grid(samples, points)
    recommendations = recommend_thresholds(
        rows,
        max_fpir=args.max_fpir,
        limit=args.max_recommendations,
    )
    report = build_report(
        manifest=manifest,
        model_version=engine.model_version,
        model_fingerprint=fingerprint,
        sample_states=states,
        rows=rows,
        recommendations=recommendations,
        max_fpir=args.max_fpir,
        providers=providers,
        detection_size=detection_size,
        quality_policy=quality_policy,
        gallery_min_quality=gallery_min_quality,
        roi=roi,
    )
    write_json_report(args.json_output, report)
    write_csv_report(args.csv_output, rows, recommendations)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (
        EvaluationError,
        GalleryEnrollmentError,
        ModelBundleError,
        OSError,
        ValueError,
        VisionDependencyError,
        VisionModelError,
    ) as exc:
        print(f"Avaliação não executada: {exc}", file=sys.stderr)
        return 1
    recommendations = report["recommendations"]
    counts = report["counts"]
    print(
        f"Avaliação concluída: {counts['known']} conhecidos, "
        f"{counts['unknown']} desconhecidos, {len(report['grid'])} pontos."
    )
    print(f"JSON: {args.json_output}")
    print(f"CSV: {args.csv_output}")
    if recommendations:
        best = recommendations[0]
        print(
            "Melhor ponto dentro do teto de FPIR: "
            f"similaridade={best['similarity_threshold']:.3f}, "
            f"margem={best['margin_threshold']:.3f}, "
            f"qualidade={best['quality_threshold']:.3f}, "
            f"FPIR operacional={best['fpir_operational']:.4f}, "
            f"FPIR condicional={best['fpir_conditional']:.4f}, "
            "limite superior condicional 95%="
            f"{best['fpir_conditional_upper_95']:.4f}, "
            f"FNIR={best['fnir']:.4f}."
        )
    else:
        print("Nenhum limiar respeitou o teto de FPIR informado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
