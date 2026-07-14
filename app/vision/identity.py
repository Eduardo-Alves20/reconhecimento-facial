"""Identificadores estáveis e seguros para integrações biométricas."""

from __future__ import annotations

import hashlib
import re


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")


def safe_person_id(external_id: str) -> str:
    normalized = external_id.strip()
    # IDs corporativos legíveis, como EMP001, podem ser preservados. Valores
    # numéricos/documentais (por exemplo CPF) são sempre pseudonimizados.
    if normalized[:1].isalpha() and IDENTIFIER_RE.fullmatch(normalized):
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"INTELBRAS:{digest}"


def external_id_hash(external_id: str) -> str:
    return hashlib.sha256(external_id.strip().encode("utf-8")).hexdigest()
