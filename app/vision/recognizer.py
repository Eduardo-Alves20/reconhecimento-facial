"""Reconhecimento facial local com OpenCV YuNet e SFace.

O módulo não importa OpenCV/NumPy durante o startup da API principal. Eles são
carregados somente ao construir o worker de visão.
"""

from __future__ import annotations

import hashlib
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
    # Uma pessoa pode ter várias fotos de referência (ângulos diferentes).
    # Colapsa para a melhor nota por pessoa antes de ranquear, para que a
    # margem seja medida entre pessoas distintas — não entre fotos da mesma.
    best_per_person: dict[str, CandidateScore] = {}
    for candidate in candidates:
        current = best_per_person.get(candidate.external_id)
        if current is None or candidate.similarity > current.similarity:
            best_per_person[candidate.external_id] = candidate
    ordered = sorted(
        best_per_person.values(), key=lambda item: item.similarity, reverse=True
    )
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


class ArcFaceEngine:
    """Reconhecimento facial com InsightFace (RetinaFace + ArcFace r50).

    Embeddings ArcFace de 512 dimensões e normalizados; similaridade por cosseno
    (produto escalar). A separação entre pessoas é muito melhor que a do SFace,
    o que permite um limiar mais baixo sem confundir identidades. Não oferece
    prova de vida.
    """

    model_version = "insightface-buffalo_l-retinaface-arcface-r50"

    def __init__(
        self,
        *,
        thresholds: MatchThresholds | None = None,
        model_root: str | Path | None = None,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VisionDependencyError(
                "Instale o extra de visão: python -m pip install -e \".[vision]\""
            ) from exc

        self.cv2 = cv2
        self.np = np
        self.thresholds = thresholds or MatchThresholds()
        options: dict[str, Any] = {
            "name": "buffalo_l",
            "providers": ["CPUExecutionProvider"],
            "allowed_modules": ["detection", "recognition"],
        }
        if model_root is not None:
            options["root"] = str(model_root)
        try:
            self.app = FaceAnalysis(**options)
            self.app.prepare(ctx_id=-1, det_size=det_size)
        except Exception as exc:
            raise VisionModelError(
                "Não foi possível carregar o modelo ArcFace (buffalo_l)."
            ) from exc
        self.gallery: list[GalleryEntry] = []

    def _quality(self, det_score: float, box_width: int, box_height: int) -> float:
        # O ArcFace já alinha o rosto, então a qualidade foca em confiança da
        # detecção e tamanho do rosto (rosto minúsculo gera embedding instável).
        size_score = min(1.0, min(box_width, box_height) / 80.0)
        return max(0.0, min(1.0, float(det_score) * size_score))

    def similarity(self, first: Any, second: Any) -> float:
        """Cosseno entre dois embeddings normalizados (produto escalar)."""

        return float(self.np.dot(first, second))

    def add_reference(self, external_id: str, display_name: str, feature: Any) -> None:
        """Adiciona uma referência à galeria em memória (efeito imediato)."""

        self.gallery.append(
            GalleryEntry(
                str(external_id),
                str(display_name),
                self.np.asarray(feature, dtype=self.np.float32),
            )
        )

    def detect(self, frame: Any) -> list[DetectedFace]:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame inválido")
        height, width = frame.shape[:2]
        try:
            faces = self.app.get(frame)
        except Exception as exc:
            raise VisionModelError("Falha ao executar detecção/reconhecimento.") from exc
        detections: list[DetectedFace] = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            x1, y1, x2, y2 = (int(value) for value in face.bbox)
            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)
            detections.append(
                DetectedFace(
                    bbox=(x1, y1, box_width, box_height),
                    centroid=(
                        max(0.0, min(1.0, (x1 + box_width / 2) / width)),
                        max(0.0, min(1.0, (y1 + box_height / 2) / height)),
                    ),
                    detector_score=float(face.det_score),
                    quality=self._quality(face.det_score, box_width, box_height),
                    feature=self.np.asarray(embedding, dtype=self.np.float32),
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

        # Impressão digital do conjunto: muda quando alguém é adicionado/removido.
        fingerprint = hashlib.sha256(
            "|".join(
                sorted(
                    f"{item.get('external_id')}:{item.get('image_sha256', '')}"
                    for item in manifest["entries"]
                    if isinstance(item, dict)
                )
            ).encode("utf-8")
            + self.model_version.encode("utf-8")
        ).hexdigest()
        cache_file = manifest_file.parent / "embeddings.arcface.npz"
        cached = self._load_cache(cache_file, fingerprint)
        if cached is not None:
            self.gallery = cached
            return len(cached)

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
        self._save_cache(cache_file, fingerprint, enrolled)
        self.gallery = enrolled
        return len(enrolled)

    def _load_cache(self, cache_file: Path, fingerprint: str) -> list[GalleryEntry] | None:
        if not cache_file.is_file():
            return None
        try:
            data = self.np.load(cache_file, allow_pickle=False)
            if str(data["fingerprint"]) != fingerprint:
                return None
            ids = [str(v) for v in data["external_ids"]]
            names = [str(v) for v in data["display_names"]]
            features = data["features"]
            return [
                GalleryEntry(ids[i], names[i], features[i]) for i in range(len(ids))
            ]
        except Exception:
            return None

    def _save_cache(
        self, cache_file: Path, fingerprint: str, entries: list[GalleryEntry]
    ) -> None:
        try:
            self.np.savez(
                cache_file,
                fingerprint=fingerprint,
                external_ids=self.np.array([e.external_id for e in entries]),
                display_names=self.np.array([e.display_name for e in entries]),
                features=self.np.stack([e.feature for e in entries]),
            )
        except Exception:
            pass

    def match(self, face: DetectedFace) -> MatchDecision:
        candidates = [
            CandidateScore(
                external_id=item.external_id,
                display_name=item.display_name,
                similarity=self.similarity(face.feature, item.feature),
            )
            for item in self.gallery
        ]
        return decide_match(
            candidates,
            face_quality=face.quality,
            thresholds=self.thresholds,
        )
