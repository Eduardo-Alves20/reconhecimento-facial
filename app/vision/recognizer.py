"""Detecção e reconhecimento facial local com InsightFace."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import zipfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Sequence


EMBEDDING_DIMENSION = 512
_MAX_GALLERY_ENTRIES = 50_000
_MAX_CACHE_BYTES = 512 * 1024 * 1024
_MAX_CACHE_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MODEL_FILES = ("det_10g.onnx", "w600k_r50.onnx")
_GALLERY_CACHE_SCHEMA = 2
_FACE_QUALITY_ALGORITHM_VERSION = 1


class VisionDependencyError(RuntimeError):
    pass


class VisionModelError(RuntimeError):
    pass


class GalleryEnrollmentError(RuntimeError):
    pass


class MatchStatus(StrEnum):
    MATCHED = "MATCHED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    LOW_QUALITY = "LOW_QUALITY"


@dataclass(frozen=True, slots=True)
class MatchThresholds:
    similarity: float = 0.55
    margin: float = 0.08
    face_quality: float = 0.40

    def __post_init__(self) -> None:
        for name, value in (
            ("similarity", self.similarity),
            ("margin", self.margin),
            ("face_quality", self.face_quality),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} deve estar entre 0 e 1")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    external_id: str
    display_name: str
    similarity: float

    def __post_init__(self) -> None:
        if not self.external_id.strip() or not self.display_name.strip():
            raise ValueError("Candidato sem identificação.")
        if not math.isfinite(self.similarity):
            raise ValueError("A similaridade do candidato deve ser finita.")


@dataclass(frozen=True, slots=True)
class MatchDecision:
    status: MatchStatus
    external_id: str | None
    display_name: str | None
    similarity: float | None
    margin: float | None
    face_quality: float


def _rank_candidates(candidates: Iterable[CandidateScore]) -> list[CandidateScore]:
    best_per_person: dict[str, CandidateScore] = {}
    names: dict[str, str] = {}
    for candidate in candidates:
        external_id = candidate.external_id.strip()
        display_name = candidate.display_name.strip()
        known_name = names.setdefault(external_id, display_name)
        if known_name != display_name:
            raise ValueError("O mesmo ID está associado a nomes diferentes.")
        score = max(-1.0, min(1.0, float(candidate.similarity)))
        normalized = CandidateScore(external_id, display_name, score)
        current = best_per_person.get(external_id)
        if current is None or normalized.similarity > current.similarity:
            best_per_person[external_id] = normalized
    return sorted(
        best_per_person.values(),
        key=lambda item: (-item.similarity, item.external_id),
    )


def decide_match(
    candidates: Iterable[CandidateScore],
    *,
    face_quality: float,
    thresholds: MatchThresholds,
) -> MatchDecision:
    if not math.isfinite(face_quality):
        raise ValueError("face_quality deve ser finita")
    quality = max(0.0, min(1.0, face_quality))
    ordered = _rank_candidates(candidates)
    if quality < thresholds.face_quality:
        return MatchDecision(MatchStatus.LOW_QUALITY, None, None, None, None, quality)
    if not ordered:
        return MatchDecision(MatchStatus.UNKNOWN, None, None, None, None, quality)

    best = ordered[0]
    second_score = ordered[1].similarity if len(ordered) > 1 else -1.0
    margin = best.similarity - second_score
    if best.similarity < thresholds.similarity:
        return MatchDecision(
            MatchStatus.UNKNOWN,
            None,
            None,
            best.similarity,
            margin,
            quality,
        )
    if len(ordered) > 1 and margin <= thresholds.margin:
        return MatchDecision(
            MatchStatus.AMBIGUOUS,
            None,
            None,
            best.similarity,
            margin,
            quality,
        )
    return MatchDecision(
        MatchStatus.MATCHED,
        best.external_id,
        best.display_name,
        best.similarity,
        margin,
        quality,
    )


@dataclass(frozen=True, slots=True)
class FaceQualityMetrics:
    score: float
    inter_eye_distance: float | None
    size: float
    sharpness: float
    exposure: float
    pose: float
    occlusion: float
    detector: float


@dataclass(frozen=True, slots=True)
class FaceQualityPolicy:
    min_inter_eye_pixels: float = 40.0
    full_inter_eye_pixels: float = 80.0
    min_focus_variance: float = 20.0
    full_focus_variance: float = 260.0
    min_luminance: float = 35.0
    ideal_luminance: float = 128.0
    max_luminance: float = 220.0
    min_contrast: float = 8.0
    full_contrast: float = 42.0
    max_clipped_fraction: float = 0.25
    max_pitch_degrees: float = 25.0
    max_yaw_degrees: float = 35.0
    max_roll_degrees: float = 25.0

    def __post_init__(self) -> None:
        values = (
            self.min_inter_eye_pixels,
            self.full_inter_eye_pixels,
            self.min_focus_variance,
            self.full_focus_variance,
            self.min_luminance,
            self.ideal_luminance,
            self.max_luminance,
            self.min_contrast,
            self.full_contrast,
            self.max_clipped_fraction,
            self.max_pitch_degrees,
            self.max_yaw_degrees,
            self.max_roll_degrees,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Os parâmetros de qualidade devem ser positivos.")
        if (
            self.full_inter_eye_pixels <= self.min_inter_eye_pixels
            or self.full_focus_variance <= self.min_focus_variance
            or self.full_contrast <= self.min_contrast
            or not self.min_luminance < self.ideal_luminance < self.max_luminance <= 255
            or self.max_clipped_fraction > 1
        ):
            raise ValueError("Os intervalos da política de qualidade são inválidos.")


@dataclass(frozen=True, slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    detector_score: float
    quality: float
    feature: Any
    quality_metrics: FaceQualityMetrics | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4 or self.bbox[2] <= 0 or self.bbox[3] <= 0:
            raise ValueError("A caixa do rosto é inválida.")
        if len(self.centroid) != 2 or not all(
            math.isfinite(value) and 0 <= value <= 1 for value in self.centroid
        ):
            raise ValueError("O centro do rosto deve estar normalizado.")
        if not math.isfinite(self.detector_score) or not 0 <= self.detector_score <= 1:
            raise ValueError("A confiança do detector deve estar entre 0 e 1.")
        if not math.isfinite(self.quality) or not 0 <= self.quality <= 1:
            raise ValueError("A qualidade do rosto deve estar entre 0 e 1.")


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    external_id: str
    display_name: str
    feature: Any


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    external_id: str
    display_name: str
    image_path: Path
    relative_path: str
    image_sha256: str


def _clamp01(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _ramp(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("Intervalo de normalização inválido.")
    return _clamp01((value - lower) / (upper - lower))


def _landmark_points(landmarks: Any, np_module: Any) -> Any | None:
    if landmarks is None:
        return None
    try:
        points = np_module.asarray(landmarks, dtype=np_module.float64).reshape(-1, 2)
    except Exception:
        return None
    if len(points) < 5 or not bool(np_module.isfinite(points).all()):
        return None
    return points


def _pose_quality(
    points: Any | None,
    pose: Any,
    np_module: Any,
    policy: FaceQualityPolicy,
) -> float:
    if pose is not None:
        try:
            angles = np_module.asarray(pose, dtype=np_module.float64).reshape(-1)
            if len(angles) >= 3 and bool(np_module.isfinite(angles[:3]).all()):
                pitch, yaw, roll = (abs(float(value)) for value in angles[:3])
                return min(
                    _clamp01(1.0 - pitch / policy.max_pitch_degrees),
                    _clamp01(1.0 - yaw / policy.max_yaw_degrees),
                    _clamp01(1.0 - roll / policy.max_roll_degrees),
                )
        except Exception:
            pass
    if points is None:
        return 0.65

    left_eye, right_eye, nose, left_mouth, right_mouth = points[:5]
    eye_vector = right_eye - left_eye
    inter_eye = float(np_module.linalg.norm(eye_vector))
    if inter_eye <= 1e-6:
        return 0.0
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0
    roll = abs(math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0]))))
    yaw_ratio = abs(float(nose[0] - eye_mid[0])) / inter_eye
    eye_mouth = max(float(np_module.linalg.norm(mouth_mid - eye_mid)), 1e-6)
    pitch_ratio = float(np_module.linalg.norm(nose - eye_mid)) / eye_mouth
    yaw_limit = 0.42 * policy.max_yaw_degrees / 35.0
    pitch_limit = 0.38 * policy.max_pitch_degrees / 25.0
    return min(
        _clamp01(1.0 - roll / policy.max_roll_degrees),
        _clamp01(1.0 - yaw_ratio / yaw_limit),
        _clamp01(1.0 - abs(pitch_ratio - 0.48) / pitch_limit),
    )


def evaluate_face_quality(
    frame: Any,
    bbox: tuple[int, int, int, int],
    detector_score: float,
    *,
    landmarks: Any = None,
    pose: Any = None,
    policy: FaceQualityPolicy | None = None,
    np_module: Any,
) -> FaceQualityMetrics:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame inválido")
    frame_height, frame_width = (int(value) for value in frame.shape[:2])
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("frame vazio")

    x, y, box_width, box_height = (int(value) for value in bbox)
    if box_width <= 0 or box_height <= 0:
        raise ValueError("bbox inválida")
    x1 = max(0, min(frame_width, x))
    y1 = max(0, min(frame_height, y))
    x2 = max(0, min(frame_width, x + box_width))
    y2 = max(0, min(frame_height, y + box_height))
    visible_area = max(0, x2 - x1) * max(0, y2 - y1)
    visible_ratio = visible_area / float(box_width * box_height)
    detector = _clamp01(detector_score) if math.isfinite(detector_score) else 0.0
    points = _landmark_points(landmarks, np_module)
    quality_policy = policy or FaceQualityPolicy()

    if points is not None:
        inter_eye = float(np_module.linalg.norm(points[1] - points[0]))
        inside = (
            (points[:, 0] >= x1) & (points[:, 0] < x2) & (points[:, 1] >= y1) & (points[:, 1] < y2)
        )
        landmark_visibility = float(np_module.mean(inside))
    else:
        inter_eye = min(box_width, box_height) * 0.38
        landmark_visibility = 0.75

    size_score = _ramp(
        inter_eye,
        quality_policy.min_inter_eye_pixels,
        quality_policy.full_inter_eye_pixels,
    )
    occlusion_score = _clamp01(
        math.sqrt(max(0.0, visible_ratio)) * (0.45 + 0.55 * landmark_visibility)
    )
    pose_score = _pose_quality(points, pose, np_module, quality_policy)

    if visible_area == 0:
        sharpness_score = 0.0
        exposure_score = 0.0
    else:
        crop = np_module.asarray(frame[y1:y2, x1:x2], dtype=np_module.float32)
        if crop.ndim == 3 and crop.shape[2] >= 3:
            gray = crop[..., 0] * 0.114 + crop[..., 1] * 0.587 + crop[..., 2] * 0.299
        elif crop.ndim == 2:
            gray = crop
        else:
            gray = crop.reshape(crop.shape[0], crop.shape[1])

        if gray.shape[0] >= 3 and gray.shape[1] >= 3:
            center = gray[1:-1, 1:-1]
            laplacian = (
                4.0 * center - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
            )
            focus_variance = float(np_module.var(laplacian))
            sharpness_score = _ramp(
                math.log1p(max(0.0, focus_variance)),
                math.log1p(quality_policy.min_focus_variance),
                math.log1p(quality_policy.full_focus_variance),
            )
        else:
            sharpness_score = 0.0

        mean = float(np_module.mean(gray))
        contrast = float(np_module.std(gray))
        clipped = float(np_module.mean((gray <= 8.0) | (gray >= 247.0)))
        if mean <= quality_policy.ideal_luminance:
            luminance = _ramp(
                mean,
                quality_policy.min_luminance,
                quality_policy.ideal_luminance,
            )
        else:
            luminance = _ramp(
                quality_policy.max_luminance - mean,
                0.0,
                quality_policy.max_luminance - quality_policy.ideal_luminance,
            )
        contrast_score = _ramp(
            contrast,
            quality_policy.min_contrast,
            quality_policy.full_contrast,
        )
        clipping_score = _clamp01(1.0 - clipped / quality_policy.max_clipped_fraction)
        exposure_score = _clamp01(luminance * clipping_score * (0.65 + 0.35 * contrast_score))

    weighted = (
        detector * 0.15
        + size_score * 0.25
        + sharpness_score * 0.20
        + exposure_score * 0.15
        + pose_score * 0.15
        + occlusion_score * 0.10
    )
    weakest = min(
        size_score,
        sharpness_score,
        exposure_score,
        pose_score,
        occlusion_score,
    )
    score = _clamp01(weighted * (0.5 + 0.5 * weakest))
    return FaceQualityMetrics(
        score=score,
        inter_eye_distance=inter_eye,
        size=size_score,
        sharpness=sharpness_score,
        exposure=exposure_score,
        pose=pose_score,
        occlusion=occlusion_score,
        detector=detector,
    )


class ArcFaceEngine:
    """Executa SCRFD e ArcFace localmente pelo pacote InsightFace."""

    model_version = "insightface-buffalo_l-scrfd10g-w600k-r50"

    def __init__(
        self,
        *,
        thresholds: MatchThresholds | None = None,
        model_root: str | Path | None = None,
        model_fingerprint: str | None = None,
        det_size: tuple[int, int] = (640, 640),
        providers: Sequence[str] = ("CPUExecutionProvider",),
        quality_policy: FaceQualityPolicy | None = None,
    ) -> None:
        if len(det_size) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in det_size
        ):
            raise ValueError("det_size deve conter duas dimensões positivas.")
        if isinstance(providers, (str, bytes)):
            raise ValueError("providers deve ser uma sequência de nomes.")
        try:
            provider_list = [str(provider).strip() for provider in providers]
        except TypeError as exc:
            raise ValueError("providers deve ser uma sequência de nomes.") from exc
        if (
            not provider_list
            or any(not provider or len(provider) > 128 for provider in provider_list)
            or len(set(provider_list)) != len(provider_list)
        ):
            raise ValueError("providers deve conter nomes únicos e não vazios.")
        if model_root is None:
            raise VisionModelError("Informe a raiz do bundle local de modelos.")
        supplied_root = Path(model_root)
        if supplied_root.is_symlink():
            raise VisionModelError("A raiz do bundle não pode ser um link.")
        try:
            resolved_root = supplied_root.resolve(strict=True)
        except OSError as exc:
            raise VisionModelError("A raiz do bundle de modelos não existe.") from exc
        model_directory = resolved_root / "models" / "buffalo_l"
        if (
            not resolved_root.is_dir()
            or model_directory.is_symlink()
            or not model_directory.is_dir()
            or any(
                not (model_directory / filename).is_file()
                or (model_directory / filename).is_symlink()
                for filename in _MODEL_FILES
            )
        ):
            raise VisionModelError("O bundle local buffalo_l está incompleto.")
        fingerprint = str(model_fingerprint or "").strip().lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise VisionModelError("O fingerprint verificado do modelo é obrigatório.")

        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VisionDependencyError(
                'Instale o extra de visão: python -m pip install -e ".[vision]"'
            ) from exc

        self.cv2 = cv2
        self.np = np
        self.thresholds = thresholds or MatchThresholds()
        self.quality_policy = quality_policy or FaceQualityPolicy()
        self.model_fingerprint = fingerprint
        self.model_root = resolved_root
        self.det_size = tuple(det_size)
        self.providers = tuple(provider_list)
        options: dict[str, Any] = {
            "name": "buffalo_l",
            "providers": provider_list,
            "allowed_modules": ["detection", "recognition"],
            "root": str(resolved_root),
        }
        try:
            self.app = FaceAnalysis(**options)
            self.app.prepare(ctx_id=-1, det_size=self.det_size)
        except Exception as exc:
            raise VisionModelError(
                "Não foi possível carregar o modelo ArcFace (buffalo_l)."
            ) from exc
        self.gallery: list[GalleryEntry] = []
        self._gallery_matrix: Any | None = None
        self._gallery_matrix_size = 0
        self._gallery_lock = threading.RLock()
        self._official_gallery: tuple[GalleryEntry, ...] = ()
        self._learned_gallery_signature: str | None = None

    def _normalize_embedding(self, feature: Any) -> Any:
        try:
            embedding = self.np.asarray(feature, dtype=self.np.float32).reshape(-1)
        except Exception as exc:
            raise ValueError("Embedding facial inválido.") from exc
        if embedding.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(f"O embedding facial deve ter {EMBEDDING_DIMENSION} dimensões.")
        if not bool(self.np.isfinite(embedding).all()):
            raise ValueError("O embedding facial contém valor não finito.")
        norm = float(self.np.linalg.norm(embedding))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("O embedding facial não pode ser nulo.")
        normalized = self.np.ascontiguousarray(embedding / norm, dtype=self.np.float32)
        normalized.setflags(write=False)
        return normalized

    @staticmethod
    def _identity(external_id: str, display_name: str) -> tuple[str, str]:
        identifier = str(external_id).strip()
        name = str(display_name).strip()
        if not identifier or len(identifier) > 128:
            raise ValueError("ID externo inválido.")
        if not name or len(name) > 256:
            raise ValueError("Nome de exibição inválido.")
        if any(ord(character) < 32 for character in identifier + name):
            raise ValueError("A identificação contém caractere de controle.")
        return identifier, name

    def _replace_gallery(self, entries: Sequence[GalleryEntry]) -> None:
        normalized: list[GalleryEntry] = []
        names: dict[str, str] = {}
        for entry in entries:
            external_id, display_name = self._identity(entry.external_id, entry.display_name)
            prior_name = names.setdefault(external_id, display_name)
            if prior_name != display_name:
                raise ValueError("O mesmo ID está associado a nomes diferentes.")
            normalized.append(
                GalleryEntry(
                    external_id,
                    display_name,
                    self._normalize_embedding(entry.feature),
                )
            )
        with self._gallery_lock:
            self.gallery = normalized
            self._gallery_matrix = None
            self._gallery_matrix_size = 0

    def _matrix(self) -> Any:
        with self._gallery_lock:
            if self._gallery_matrix is None or self._gallery_matrix_size != len(self.gallery):
                if not self.gallery:
                    matrix = self.np.empty((0, EMBEDDING_DIMENSION), dtype=self.np.float32)
                else:
                    refreshed = [
                        GalleryEntry(
                            *self._identity(entry.external_id, entry.display_name),
                            self._normalize_embedding(entry.feature),
                        )
                        for entry in self.gallery
                    ]
                    self.gallery = refreshed
                    matrix = self.np.stack([entry.feature for entry in refreshed])
                    matrix = self.np.ascontiguousarray(matrix, dtype=self.np.float32)
                matrix.setflags(write=False)
                self._gallery_matrix = matrix
                self._gallery_matrix_size = len(self.gallery)
            return self._gallery_matrix

    def similarity(self, first: Any, second: Any) -> float:
        first_embedding = self._normalize_embedding(first)
        second_embedding = self._normalize_embedding(second)
        score = float(self.np.dot(first_embedding, second_embedding))
        return max(-1.0, min(1.0, score))

    def add_reference(self, external_id: str, display_name: str, feature: Any) -> None:
        identifier, name = self._identity(external_id, display_name)
        normalized = self._normalize_embedding(feature)
        with self._gallery_lock:
            if len(self.gallery) >= _MAX_GALLERY_ENTRIES:
                raise ValueError("A galeria excede o limite de referências.")
            for entry in self.gallery:
                if entry.external_id == identifier and entry.display_name != name:
                    raise ValueError("O mesmo ID está associado a nomes diferentes.")
            self.gallery.append(GalleryEntry(identifier, name, normalized))
            self._gallery_matrix = None
            self._gallery_matrix_size = 0

    @property
    def official_external_ids(self) -> frozenset[str]:
        with self._gallery_lock:
            return frozenset(entry.external_id for entry in self._official_gallery)

    def replace_learned_references(
        self,
        entries: Sequence[GalleryEntry],
    ) -> bool:
        with self._gallery_lock:
            official = self._official_gallery
        if not official:
            raise GalleryEnrollmentError("A galeria oficial ainda não foi carregada.")
        if len(official) + len(entries) > _MAX_GALLERY_ENTRIES:
            raise GalleryEnrollmentError("A galeria excede o limite de referências.")
        learned: list[GalleryEntry] = []
        names = {entry.external_id: entry.display_name for entry in official}
        digest = hashlib.sha256()
        for entry in entries:
            external_id, display_name = self._identity(
                entry.external_id,
                entry.display_name,
            )
            prior_name = names.setdefault(external_id, display_name)
            if prior_name != display_name:
                raise ValueError("O mesmo ID está associado a nomes diferentes.")
            feature = self._normalize_embedding(entry.feature)
            learned.append(GalleryEntry(external_id, display_name, feature))
            digest.update(external_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(display_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(feature.tobytes())
        signature = digest.hexdigest()
        with self._gallery_lock:
            if self._learned_gallery_signature == signature:
                return False
        combined = [*official, *learned]
        matrix = self.np.stack([entry.feature for entry in combined])
        matrix = self.np.ascontiguousarray(matrix, dtype=self.np.float32)
        matrix.setflags(write=False)
        with self._gallery_lock:
            if self._learned_gallery_signature == signature:
                return False
            self.gallery = combined
            self._gallery_matrix = matrix
            self._gallery_matrix_size = len(combined)
            self._learned_gallery_signature = signature
        return True

    def detect(self, frame: Any) -> list[DetectedFace]:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame inválido")
        height, width = (int(value) for value in frame.shape[:2])
        if height <= 0 or width <= 0:
            raise ValueError("frame vazio")
        try:
            faces = self.app.get(frame)
        except Exception as exc:
            raise VisionModelError("Falha ao executar detecção/reconhecimento.") from exc

        detections: list[DetectedFace] = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            try:
                normalized = self._normalize_embedding(embedding)
                raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in face.bbox)
                detector_score = _clamp01(float(getattr(face, "det_score", 0.0)))
            except (TypeError, ValueError):
                continue
            if (
                not all(math.isfinite(value) for value in (raw_x1, raw_y1, raw_x2, raw_y2))
                or raw_x2 <= raw_x1
                or raw_y2 <= raw_y1
            ):
                continue
            x1 = max(0, math.floor(raw_x1))
            y1 = max(0, math.floor(raw_y1))
            x2 = min(width, math.ceil(raw_x2))
            y2 = min(height, math.ceil(raw_y2))
            if x2 <= x1 or y2 <= y1:
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            quality_x1 = math.floor(raw_x1)
            quality_y1 = math.floor(raw_y1)
            quality_x2 = math.ceil(raw_x2)
            quality_y2 = math.ceil(raw_y2)
            metrics = evaluate_face_quality(
                frame,
                (
                    quality_x1,
                    quality_y1,
                    quality_x2 - quality_x1,
                    quality_y2 - quality_y1,
                ),
                detector_score,
                landmarks=getattr(face, "kps", None),
                pose=getattr(face, "pose", None),
                policy=self.quality_policy,
                np_module=self.np,
            )
            detections.append(
                DetectedFace(
                    bbox=(x1, y1, box_width, box_height),
                    centroid=(
                        _clamp01((x1 + box_width / 2) / width),
                        _clamp01((y1 + box_height / 2) / height),
                    ),
                    detector_score=detector_score,
                    quality=metrics.score,
                    feature=normalized,
                    quality_metrics=metrics,
                )
            )
        return detections

    def _read_image(self, image_path: Path) -> Any:
        try:
            content = image_path.read_bytes()
            encoded = self.np.frombuffer(content, dtype=self.np.uint8)
            image = self.cv2.imdecode(encoded, self.cv2.IMREAD_COLOR)
        except Exception as exc:
            raise GalleryEnrollmentError(f"Não foi possível ler {image_path.name}.") from exc
        if image is None:
            raise GalleryEnrollmentError(f"Imagem inválida: {image_path.name}.")
        return image

    def detect_image(self, image_path: str | Path) -> list[DetectedFace]:
        return self.detect(self._read_image(Path(image_path)))

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise GalleryEnrollmentError(f"Não foi possível validar {path.name}.") from exc
        return digest.hexdigest()

    def _manifest_entries(
        self, manifest_file: Path, raw_entries: list[Any]
    ) -> list[_ManifestEntry]:
        root = manifest_file.parent.resolve()
        if len(raw_entries) > _MAX_GALLERY_ENTRIES:
            raise GalleryEnrollmentError("A galeria excede o limite de referências.")
        result: list[_ManifestEntry] = []
        names: dict[str, str] = {}
        seen: set[tuple[str, str]] = set()
        hash_owners: dict[str, str] = {}
        for item in raw_entries:
            if not isinstance(item, dict):
                raise GalleryEnrollmentError("O manifesto contém uma entrada inválida.")
            try:
                external_id, display_name = self._identity(
                    str(item.get("external_id", "")),
                    str(item.get("display_name", "")),
                )
            except ValueError as exc:
                raise GalleryEnrollmentError("Identidade inválida no manifesto.") from exc
            relative_text = str(item.get("image_path", "")).strip().replace("\\", "/")
            relative_path = Path(relative_text)
            expected_hash = str(item.get("image_sha256", "")).strip().lower()
            if (
                not relative_text
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise GalleryEnrollmentError("Referência de imagem inválida no manifesto.")
            image_path = (root / relative_path).resolve()
            try:
                image_path.relative_to(root)
            except ValueError as exc:
                raise GalleryEnrollmentError(
                    "Uma imagem da galeria aponta para fora do diretório."
                ) from exc
            if not image_path.is_file():
                raise GalleryEnrollmentError("Uma imagem da galeria não existe.")
            actual_hash = self._file_sha256(image_path)
            if actual_hash != expected_hash:
                raise GalleryEnrollmentError(f"A integridade de {image_path.name} não confere.")
            prior_name = names.setdefault(external_id, display_name)
            if prior_name != display_name:
                raise GalleryEnrollmentError("O mesmo ID está associado a nomes diferentes.")
            prior_owner = hash_owners.setdefault(expected_hash, external_id)
            if prior_owner != external_id:
                raise GalleryEnrollmentError(
                    "A mesma foto não pode ser associada a identidades diferentes."
                )
            pair = (external_id, expected_hash)
            if pair in seen:
                raise GalleryEnrollmentError("O manifesto contém referência duplicada.")
            seen.add(pair)
            result.append(
                _ManifestEntry(
                    external_id,
                    display_name,
                    image_path,
                    relative_path.as_posix(),
                    expected_hash,
                )
            )
        if not result:
            raise GalleryEnrollmentError("O manifesto não contém referências.")
        return result

    def load_gallery(
        self,
        manifest_path: str | Path,
        *,
        cache_path: str | Path | None = None,
    ) -> int:
        manifest_file = Path(manifest_path)
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GalleryEnrollmentError("Manifesto da galeria inválido.") from exc
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("entries"), list):
            raise GalleryEnrollmentError("Versão ou estrutura do manifesto não suportada.")

        entries = self._manifest_entries(manifest_file, manifest["entries"])
        fingerprint_payload = {
            "cache_schema": _GALLERY_CACHE_SCHEMA,
            "model": {
                "version": self.model_version,
                "fingerprint": self.model_fingerprint,
                "det_size": list(self.det_size),
                "providers": list(self.providers),
            },
            "enrollment": {
                "minimum_face_quality": self.thresholds.face_quality,
                "quality_algorithm_version": _FACE_QUALITY_ALGORITHM_VERSION,
                "quality_policy": asdict(self.quality_policy),
            },
            "entries": [
                {
                    "external_id": entry.external_id,
                    "display_name": entry.display_name,
                    "image_path": entry.relative_path,
                    "image_sha256": entry.image_sha256,
                }
                for entry in entries
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_file = (
            Path(cache_path)
            if cache_path is not None
            else manifest_file.parent / "embeddings.arcface.npz"
        )
        if cache_file.is_symlink() or cache_file.parent.is_symlink():
            raise GalleryEnrollmentError("O cache da galeria não pode usar links.")
        cached = self._load_cache(cache_file, fingerprint)
        if cached is not None:
            expected_ids = {entry.external_id for entry in entries}
            cached_ids = {entry.external_id for entry in cached}
            if cached_ids != expected_ids:
                cached = None
        if cached is not None:
            self._replace_gallery(cached)
            with self._gallery_lock:
                self._official_gallery = tuple(self.gallery)
                self._learned_gallery_signature = None
            return len(self.gallery)

        enrolled: list[GalleryEntry] = []
        for entry in entries:
            faces = self.detect_image(entry.image_path)
            if len(faces) != 1:
                continue
            face = faces[0]
            if face.quality < self.thresholds.face_quality:
                continue
            enrolled.append(GalleryEntry(entry.external_id, entry.display_name, face.feature))
        if not enrolled:
            raise GalleryEnrollmentError(
                "Nenhuma foto com um rosto único e qualidade suficiente foi importada."
            )
        missing_people = {
            entry.external_id for entry in entries
        } - {entry.external_id for entry in enrolled}
        if missing_people:
            raise GalleryEnrollmentError(
                f"{len(missing_people)} pessoa(s) ficaram sem referência facial válida."
            )
        self._replace_gallery(enrolled)
        with self._gallery_lock:
            self._official_gallery = tuple(self.gallery)
            self._learned_gallery_signature = None
        self._save_cache(cache_file, fingerprint, self.gallery)
        return len(self.gallery)

    @staticmethod
    def _cache_container_is_safe(cache_file: Path) -> bool:
        expected = {
            "fingerprint.npy",
            "external_ids.npy",
            "display_names.npy",
            "features.npy",
        }
        try:
            with zipfile.ZipFile(cache_file) as archive:
                members = archive.infolist()
                if {member.filename for member in members} != expected:
                    return False
                if sum(member.file_size for member in members) > _MAX_CACHE_UNCOMPRESSED_BYTES:
                    return False
                return all(
                    member.file_size <= max(member.compress_size, 1) * 250 for member in members
                )
        except (OSError, zipfile.BadZipFile):
            return False

    def _load_cache(self, cache_file: Path, fingerprint: str) -> list[GalleryEntry] | None:
        if not cache_file.is_file():
            return None
        try:
            if cache_file.stat().st_size > _MAX_CACHE_BYTES or not self._cache_container_is_safe(
                cache_file
            ):
                return None
            with self.np.load(cache_file, allow_pickle=False) as data:
                cached_fingerprint = str(data["fingerprint"].item())
                if cached_fingerprint != fingerprint:
                    return None
                ids = [str(value) for value in data["external_ids"].tolist()]
                names = [str(value) for value in data["display_names"].tolist()]
                features = self.np.asarray(data["features"], dtype=self.np.float32)
            if (
                not ids
                or len(ids) != len(names)
                or features.shape != (len(ids), EMBEDDING_DIMENSION)
            ):
                return None
            entries = [
                GalleryEntry(ids[index], names[index], features[index]) for index in range(len(ids))
            ]
            validated: list[GalleryEntry] = []
            known_names: dict[str, str] = {}
            for entry in entries:
                external_id, display_name = self._identity(entry.external_id, entry.display_name)
                prior_name = known_names.setdefault(external_id, display_name)
                if prior_name != display_name:
                    return None
                validated.append(
                    GalleryEntry(
                        external_id,
                        display_name,
                        self._normalize_embedding(entry.feature),
                    )
                )
            return validated
        except Exception:
            return None

    def _save_cache(
        self, cache_file: Path, fingerprint: str, entries: Sequence[GalleryEntry]
    ) -> None:
        temporary_path: Path | None = None
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{cache_file.name}.",
                suffix=".tmp",
                dir=cache_file.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                self.np.savez_compressed(
                    temporary,
                    fingerprint=fingerprint,
                    external_ids=self.np.asarray([entry.external_id for entry in entries]),
                    display_names=self.np.asarray([entry.display_name for entry in entries]),
                    features=self.np.stack([entry.feature for entry in entries]),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, cache_file)
            try:
                cache_file.chmod(0o600)
            except OSError:
                pass
        except Exception:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def rank(self, face: DetectedFace, *, limit: int | None = None) -> list[CandidateScore]:
        if limit is not None and limit < 1:
            raise ValueError("limit deve ser positivo.")
        feature = self._normalize_embedding(face.feature)
        with self._gallery_lock:
            matrix = self._matrix()
            entries = tuple(self.gallery)
        if matrix.shape[0] == 0:
            return []
        similarities = matrix @ feature
        candidates = [
            CandidateScore(
                external_id=entry.external_id,
                display_name=entry.display_name,
                similarity=max(-1.0, min(1.0, float(similarities[index]))),
            )
            for index, entry in enumerate(entries)
        ]
        ranked = _rank_candidates(candidates)
        return ranked if limit is None else ranked[:limit]

    def match(self, face: DetectedFace) -> MatchDecision:
        return decide_match(
            self.rank(face),
            face_quality=face.quality,
            thresholds=self.thresholds,
        )
