"""Reconhecimento facial local com OpenCV YuNet e SFace.

O módulo não importa OpenCV/NumPy durante o startup da API principal. Eles são
carregados somente ao construir o worker de visão.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable


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


@dataclass(frozen=True, slots=True)
class MatchDecision:
    status: MatchStatus
    external_id: str | None
    display_name: str | None
    similarity: float | None
    margin: float | None
    face_quality: float


def decide_match(
    candidates: Iterable[CandidateScore],
    *,
    face_quality: float,
    thresholds: MatchThresholds,
) -> MatchDecision:
    """Decide sem forçar o mais parecido quando faltam evidências."""

    if not math.isfinite(face_quality):
        raise ValueError("face_quality deve ser finita")
    quality = max(0.0, min(1.0, face_quality))
    ordered = sorted(candidates, key=lambda item: item.similarity, reverse=True)
    if quality < thresholds.face_quality:
        return MatchDecision(MatchStatus.LOW_QUALITY, None, None, None, None, quality)
    if not ordered:
        return MatchDecision(MatchStatus.UNKNOWN, None, None, None, None, quality)

    best = ordered[0]
    second_score = ordered[1].similarity if len(ordered) > 1 else -1.0
    margin = best.similarity - second_score
    similarity = max(-1.0, min(1.0, best.similarity))
    if similarity < thresholds.similarity:
        return MatchDecision(MatchStatus.UNKNOWN, None, None, similarity, margin, quality)
    if len(ordered) > 1 and margin < thresholds.margin:
        return MatchDecision(MatchStatus.AMBIGUOUS, None, None, similarity, margin, quality)
    return MatchDecision(
        MatchStatus.MATCHED,
        best.external_id,
        best.display_name,
        similarity,
        margin,
        quality,
    )


@dataclass(frozen=True, slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    detector_score: float
    quality: float
    feature: Any


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    external_id: str
    display_name: str
    feature: Any


class OpenCVSFaceEngine:
    """Adaptador CPU do YuNet/SFace; não oferece prova de vida."""

    model_version = "opencv-yunet-2023mar+sface-2021dec"

    def __init__(
        self,
        detector_model: str | Path,
        recognizer_model: str | Path,
        *,
        thresholds: MatchThresholds | None = None,
        detector_score_threshold: float = 0.90,
        detector_nms_threshold: float = 0.30,
        detector_top_k: int = 50,
    ) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VisionDependencyError(
                "Instale o extra de visão: python -m pip install -e \".[vision]\""
            ) from exc

        self.cv2 = cv2
        self.np = np
        self.thresholds = thresholds or MatchThresholds()
        detector_path = Path(detector_model)
        recognizer_path = Path(recognizer_model)
        for path in (detector_path, recognizer_path):
            if not path.is_file():
                raise VisionModelError(f"Modelo ausente: {path}")
        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(detector_path),
                "",
                (320, 320),
                detector_score_threshold,
                detector_nms_threshold,
                detector_top_k,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        except Exception as exc:
            raise VisionModelError("Não foi possível carregar os modelos YuNet/SFace.") from exc
        self.gallery: list[GalleryEntry] = []

    def _quality(self, frame: Any, face: Any) -> float:
        height, width = frame.shape[:2]
        x, y, box_width, box_height = (int(value) for value in face[:4])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + box_width), min(height, y + box_height)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        detector_score = float(face[-1])
        size_score = min(1.0, min(box_width, box_height) / 120.0)
        gray = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2GRAY)
        blur_variance = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        blur_score = min(1.0, blur_variance / 100.0)
        return max(0.0, min(1.0, detector_score * size_score * blur_score))

    def detect(self, frame: Any) -> list[DetectedFace]:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame inválido")
        height, width = frame.shape[:2]
        self.detector.setInputSize((int(width), int(height)))
        try:
            _, faces = self.detector.detect(frame)
        except Exception as exc:
            raise VisionModelError("Falha ao executar o detector facial.") from exc
        if faces is None:
            return []
        detections: list[DetectedFace] = []
        for face in faces:
            x, y, box_width, box_height = (int(value) for value in face[:4])
            try:
                aligned = self.recognizer.alignCrop(frame, face)
                feature = self.recognizer.feature(aligned).copy()
            except Exception:
                continue
            detections.append(
                DetectedFace(
                    bbox=(x, y, box_width, box_height),
                    centroid=(
                        max(0.0, min(1.0, (x + box_width / 2) / width)),
                        max(0.0, min(1.0, (y + box_height / 2) / height)),
                    ),
                    detector_score=float(face[-1]),
                    quality=self._quality(frame, face),
                    feature=feature,
                )
            )
        return detections

    def _read_image(self, image_path: Path) -> Any:
        try:
            content = self.np.fromfile(str(image_path), dtype=self.np.uint8)
            image = self.cv2.imdecode(content, self.cv2.IMREAD_COLOR)
        except Exception as exc:
            raise GalleryEnrollmentError(f"Não foi possível ler {image_path.name}.") from exc
        if image is None:
            raise GalleryEnrollmentError(f"Imagem inválida: {image_path.name}.")
        return image

    def load_gallery(self, manifest_path: str | Path) -> int:
        manifest_file = Path(manifest_path)
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GalleryEnrollmentError("Manifesto da galeria inválido.") from exc
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("entries"), list):
            raise GalleryEnrollmentError("Versão ou estrutura do manifesto não suportada.")

        enrolled: list[GalleryEntry] = []
        for item in manifest["entries"]:
            if not isinstance(item, dict):
                continue
            external_id = str(item.get("external_id", "")).strip()
            display_name = str(item.get("display_name", "")).strip()
            relative_path = Path(str(item.get("image_path", "")))
            if not external_id or not display_name or relative_path.is_absolute() or ".." in relative_path.parts:
                continue
            image_path = (manifest_file.parent / relative_path).resolve()
            try:
                image_path.relative_to(manifest_file.parent.resolve())
            except ValueError:
                continue
            faces = self.detect(self._read_image(image_path))
            if len(faces) != 1:
                continue
            enrolled.append(GalleryEntry(external_id, display_name, faces[0].feature))
        if not enrolled:
            raise GalleryEnrollmentError("Nenhuma foto com exatamente um rosto válido foi importada.")
        self.gallery = enrolled
        return len(enrolled)

    def match(self, face: DetectedFace) -> MatchDecision:
        candidates = [
            CandidateScore(
                external_id=item.external_id,
                display_name=item.display_name,
                similarity=float(
                    self.recognizer.match(
                        face.feature,
                        item.feature,
                        self.cv2.FaceRecognizerSF_FR_COSINE,
                    )
                ),
            )
            for item in self.gallery
        ]
        return decide_match(
            candidates,
            face_quality=face.quality,
            thresholds=self.thresholds,
        )
