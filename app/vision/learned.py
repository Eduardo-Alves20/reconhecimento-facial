"""Referências faciais aprendidas em operação (auto-enriquecimento da galeria).

Quando o sistema reconhece alguém com confiança, guarda o embedding daquele
rosto como uma nova referência da pessoa — nos ângulos reais da porta. Guarda
apenas o vetor (≈2 KB), não a imagem, então é leve, rápido e escalável.

Travas de segurança para não "envenenar" a base:
- só aprende com match confiante (similaridade acima de um piso);
- só aprende ângulo que ainda NÃO está bem coberto (similaridade abaixo de um
  teto — senão é redundante);
- qualidade mínima do rosto;
- limite de referências por pessoa.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class SupportsReference(Protocol):
    def add_reference(self, external_id: str, display_name: str, feature: Any) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS learned_refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    evidence_ref TEXT,
    similarity REAL,
    quality REAL,
    learned_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learned_person ON learned_refs(external_id);
"""


class LearnedGallery:
    def __init__(self, database_path: str | Path, *, max_per_person: int = 5) -> None:
        self.database_path = Path(database_path)
        self.max_per_person = max_per_person
        self._counts: dict[str, int] = {}
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - só no worker de visão
            raise RuntimeError("numpy é necessário para o auto-aprendizado.") from exc
        self.np = np

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(_SCHEMA)

    def load_into(self, engine: SupportsReference) -> int:
        """Carrega os embeddings aprendidos na galeria em memória do motor."""

        loaded = 0
        self._counts.clear()
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT external_id, display_name, embedding, dim FROM learned_refs ORDER BY id"
            ).fetchall()
        for external_id, display_name, blob, dim in rows:
            embedding = self.np.frombuffer(blob, dtype=self.np.float32)
            if embedding.shape[0] != int(dim):
                continue
            engine.add_reference(str(external_id), str(display_name), embedding.copy())
            self._counts[external_id] = self._counts.get(external_id, 0) + 1
            loaded += 1
        return loaded

    def consider(
        self,
        engine: SupportsReference,
        *,
        external_id: str | None,
        display_name: str | None,
        feature: Any,
        evidence_ref: str | None,
        when: datetime,
        similarity: float | None,
        quality: float | None,
        min_similarity: float,
        max_similarity: float,
        min_quality: float,
    ) -> bool:
        """Avalia e, se passar nas travas, aprende esta referência."""

        if feature is None or not external_id or not display_name:
            return False
        if quality is None or quality < min_quality:
            return False
        # Confiante o bastante para ser seguramente a pessoa, mas ainda não tão
        # perfeito a ponto de ser redundante (ângulo já coberto).
        if similarity is None or not (min_similarity <= similarity < max_similarity):
            return False
        if self._counts.get(external_id, 0) >= self.max_per_person:
            return False
        embedding = self.np.asarray(feature, dtype=self.np.float32)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO learned_refs
                    (external_id, display_name, embedding, dim, evidence_ref,
                     similarity, quality, learned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(external_id),
                    str(display_name),
                    embedding.tobytes(),
                    int(embedding.shape[0]),
                    evidence_ref,
                    float(similarity),
                    float(quality),
                    when.astimezone(UTC).isoformat(),
                ),
            )
        engine.add_reference(str(external_id), str(display_name), embedding)
        self._counts[external_id] = self._counts.get(external_id, 0) + 1
        return True
