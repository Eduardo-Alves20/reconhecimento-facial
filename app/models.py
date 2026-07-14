from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$"


class DoorResult(StrEnum):
    GRANTED = "GRANTED"
    DENIED = "DENIED"
    NOT_REPORTED = "NOT_REPORTED"


class Decision(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    JUSTIFIED = "JUSTIFIED"
    ANOMALY = "ANOMALY"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IdentityStatus(StrEnum):
    MATCHED = "MATCHED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"


class EntryEvidence(StrEnum):
    UNSPECIFIED = "UNSPECIFIED"
    VISION_LINE_CROSSING = "VISION_LINE_CROSSING"
    DOOR_SENSOR = "DOOR_SENSOR"
    ACCESS_CONTROLLER = "ACCESS_CONTROLLER"


class RecognitionSource(StrEnum):
    UNSPECIFIED = "UNSPECIFIED"
    LOCAL_SFACE = "LOCAL_SFACE"
    INTELBRAS = "INTELBRAS"


class AccessEventIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_id: str = Field(min_length=1, max_length=100, pattern=IDENTIFIER_PATTERN)
    camera_id: str = Field(min_length=1, max_length=100, pattern=IDENTIFIER_PATTERN)
    user_id: str = Field(min_length=1, max_length=100, pattern=IDENTIFIER_PATTERN)
    room_id: str = Field(min_length=1, max_length=100, pattern=IDENTIFIER_PATTERN)
    timestamp: datetime
    door_result: DoorResult = DoorResult.NOT_REPORTED
    recognition_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    identity_status: IdentityStatus = IdentityStatus.MATCHED
    entry_evidence: EntryEvidence = EntryEvidence.UNSPECIFIED
    recognition_source: RecognitionSource = RecognitionSource.UNSPECIFIED
    track_id: str | None = Field(
        default=None, min_length=1, max_length=100, pattern=IDENTIFIER_PATTERN
    )
    recognition_model: str | None = Field(default=None, min_length=1, max_length=100)
    recognition_margin: float | None = Field(default=None, ge=0.0, le=2.0)
    face_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    entry_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_offset(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp deve incluir o fuso/offset, por exemplo -03:00 ou Z")
        if not 2000 <= value.year <= 2100:
            raise ValueError("timestamp deve estar entre os anos 2000 e 2100")
        try:
            value.astimezone(UTC)
        except (OverflowError, ValueError) as exc:
            raise ValueError("timestamp está fora da faixa operacional") from exc
        return value

    @model_validator(mode="after")
    def visual_entry_has_honest_door_semantics(self) -> "AccessEventIn":
        if self.entry_evidence == EntryEvidence.VISION_LINE_CROSSING:
            if self.door_result != DoorResult.NOT_REPORTED:
                raise ValueError(
                    "entrada visual deve usar door_result=NOT_REPORTED; a câmera não prova a liberação da porta"
                )
            if self.track_id is None:
                raise ValueError("entrada visual deve informar track_id")
        return self


class EventFilters(BaseModel):
    from_timestamp: datetime | None = None
    to_timestamp: datetime | None = None
    room_id: str | None = None
    user_id: str | None = None
    decision: Decision | None = None
    risk_level: RiskLevel | None = None
    alert_status: str | None = None
    q: str | None = Field(default=None, max_length=100)

    @field_validator("from_timestamp", "to_timestamp")
    @classmethod
    def filter_timestamp_must_include_offset(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("datas de filtro devem incluir o fuso/offset")
        if value is not None and not 2000 <= value.year <= 2100:
            raise ValueError("datas de filtro devem estar entre os anos 2000 e 2100")
        return value


class Evaluation(BaseModel):
    decision: Decision
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    reason_codes: list[str]
    source_ids: list[str]
    narrative: str
    alert_required: bool
    context_snapshot: dict[str, Any]
