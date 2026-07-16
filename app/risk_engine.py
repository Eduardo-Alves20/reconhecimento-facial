from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from .models import (
    AccessEventIn,
    Decision,
    DoorResult,
    EntryEvidence,
    Evaluation,
    IdentityStatus,
    RecognitionSource,
    RiskLevel,
)


WEEKDAYS_PT = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)

REASON_TEXT = {
    "WITHIN_SCHEDULE": "dentro da escala cadastrada",
    "OUTSIDE_SCHEDULE": "fora da escala cadastrada",
    "ROOM_PERMISSION_CONFIRMED": "permissão para a sala confirmada",
    "NO_ROOM_PERMISSION": "sem permissão cadastrada para a sala",
    "QUALIFYING_INCIDENT": "incidente urgente ativo e atribuído à pessoa",
    "NO_QUALIFYING_INCIDENT": "nenhum incidente urgente qualificável encontrado",
    "UNKNOWN_PERSON": "pessoa não encontrada no cadastro",
    "INACTIVE_PERSON": "cadastro da pessoa está inativo",
    "LOW_RECOGNITION_CONFIDENCE": "confiança de reconhecimento abaixo do limiar da câmera",
    "AMBIGUOUS_IDENTITY": "comparação facial ambígua entre mais de uma pessoa",
}


def _parse_time(value: str) -> time:
    return time.fromisoformat(value)


def _schedule_matches(local_timestamp: datetime, schedule: dict) -> bool:
    start = _parse_time(schedule["start_time"])
    end = _parse_time(schedule["end_time"])
    current = local_timestamp.timetz().replace(tzinfo=None)

    if start <= end:
        return schedule["weekday"] == local_timestamp.weekday() and start <= current < end

    # Turno que cruza meia-noite: a parte após 00:00 pertence ao dia seguinte.
    if schedule["weekday"] == local_timestamp.weekday() and current >= start:
        return True
    previous_weekday = (local_timestamp.weekday() - 1) % 7
    return schedule["weekday"] == previous_weekday and current < end


def _is_qualifying_incident(
    incident: dict, *, event_timestamp: datetime, room_id: str
) -> bool:
    if incident["room_id"] != room_id:
        return False
    if incident["severity"].upper() not in {"P1", "P2"}:
        return False
    if incident["status"].upper() not in {"OPEN", "IN_PROGRESS"}:
        return False
    active_from = datetime.fromisoformat(incident["active_from"].replace("Z", "+00:00"))
    active_until = (
        datetime.fromisoformat(incident["active_until"].replace("Z", "+00:00"))
        if incident.get("active_until")
        else None
    )
    instant = event_timestamp.astimezone(UTC)
    return active_from.astimezone(UTC) <= instant and (
        active_until is None or instant <= active_until.astimezone(UTC)
    )


def _event_action(event: AccessEventIn) -> str:
    if event.entry_evidence == EntryEvidence.VISION_LINE_CROSSING:
        return "teve a entrada observada pela câmera"
    if event.entry_evidence == EntryEvidence.VISION_FACE_AT_DOOR:
        return "foi observado próximo à porta"
    if event.door_result == DoorResult.GRANTED:
        return "teve a entrada liberada"
    if event.door_result == DoorResult.DENIED:
        return "teve a tentativa de entrada negada"
    return "teve um evento de acesso registrado"


def evaluate_access(
    event: AccessEventIn,
    context: dict,
    *,
    fallback_timezone: str,
) -> Evaluation:
    camera = context.get("camera")
    person = context.get("person")
    room_timezone = camera.get("timezone") if camera else fallback_timezone
    local_timestamp = event.timestamp.astimezone(ZoneInfo(room_timezone))
    schedules = context.get("schedules", [])
    relevant_schedules = [
        schedule
        for schedule in schedules
        if schedule["weekday"] in {local_timestamp.weekday(), (local_timestamp.weekday() - 1) % 7}
    ]
    matching_schedules = [
        schedule for schedule in relevant_schedules if _schedule_matches(local_timestamp, schedule)
    ]
    permission_match = event.room_id in context.get("permission_room_ids", [])
    qualifying_incidents = [
        incident
        for incident in context.get("incidents", [])
        if _is_qualifying_incident(
            incident, event_timestamp=event.timestamp, room_id=event.room_id
        )
    ]

    reason_codes: list[str] = []
    critical_identity_problem = False
    if event.identity_status == IdentityStatus.AMBIGUOUS:
        reason_codes.append("AMBIGUOUS_IDENTITY")
        critical_identity_problem = True
    elif event.identity_status == IdentityStatus.UNKNOWN:
        reason_codes.append("UNKNOWN_PERSON")
        critical_identity_problem = True
    elif person is None:
        reason_codes.append("UNKNOWN_PERSON")
        critical_identity_problem = True
    elif not person["active"]:
        reason_codes.append("INACTIVE_PERSON")
        critical_identity_problem = True

    if (
        camera
        and event.recognition_confidence is not None
        and event.recognition_source != RecognitionSource.LOCAL_ARCFACE
    ):
        if event.recognition_confidence < camera["recognition_threshold"]:
            reason_codes.append("LOW_RECOGNITION_CONFIDENCE")
            critical_identity_problem = True

    if permission_match:
        reason_codes.append("ROOM_PERMISSION_CONFIRMED")
    else:
        reason_codes.append("NO_ROOM_PERMISSION")

    if matching_schedules:
        reason_codes.append("WITHIN_SCHEDULE")
    else:
        reason_codes.append("OUTSIDE_SCHEDULE")

    if qualifying_incidents:
        reason_codes.append("QUALIFYING_INCIDENT")
    elif not matching_schedules:
        reason_codes.append("NO_QUALIFYING_INCIDENT")

    if critical_identity_problem:
        decision = Decision.ANOMALY
        risk_level = RiskLevel.CRITICAL
        risk_score = 100
    elif permission_match and matching_schedules:
        decision = Decision.AUTHORIZED
        risk_level = RiskLevel.LOW
        risk_score = 10
    elif permission_match and qualifying_incidents:
        decision = Decision.JUSTIFIED
        risk_level = RiskLevel.MEDIUM
        risk_score = 45
    else:
        decision = Decision.ANOMALY
        risk_level = RiskLevel.HIGH
        risk_score = 85

    applicable_policies = [
        policy
        for policy in context.get("policies", [])
        if policy["applies_to_decision"] == decision.value
        and set(policy["reason_codes"]).intersection(reason_codes)
    ]
    room = (
        {
            "room_id": camera["room_id"],
            "display_name": camera["room_name"],
            "timezone": camera["timezone"],
            "criticality": camera["criticality"],
        }
        if camera
        else {"room_id": event.room_id, "display_name": event.room_id}
    )

    person_name = person["display_name"] if person else f"ID {event.user_id}"
    role = person["role_name"] if person else "função não identificada"
    room_name = room["display_name"]
    hour_text = local_timestamp.strftime("%H:%M")
    day_text = WEEKDAYS_PT[local_timestamp.weekday()]
    action = _event_action(event)

    if decision == Decision.AUTHORIZED:
        narrative = (
            f"Acesso contextual padrão. {person_name} ({role}) {action} em {room_name} "
            f"às {hour_text}, {day_text}, dentro da escala e com permissão cadastrada."
        )
    elif decision == Decision.JUSTIFIED:
        incident = qualifying_incidents[0]
        narrative = (
            f"Acesso fora do horário com contexto justificado. {person_name} ({role}) {action} "
            f"em {room_name} às {hour_text}, {day_text}. O incidente urgente "
            f"#{incident['incident_id'].removeprefix('INC-')} ({incident['title']}) estava ativo, "
            "atribuído à pessoa e relacionado à sala."
        )
    else:
        issue_codes = [
            code
            for code in reason_codes
            if code
            not in {
                "ROOM_PERMISSION_CONFIRMED",
                "WITHIN_SCHEDULE",
                "QUALIFYING_INCIDENT",
            }
        ]
        issues = "; ".join(REASON_TEXT.get(code, code) for code in issue_codes)
        narrative = (
            f"[ALERTA DE SEGURANÇA] Acesso requer verificação humana. {person_name} ({role}) "
            f"{action} em {room_name} às {hour_text}, {day_text}. Evidências: {issues}. "
            "O alerta não comprova irregularidade e não deve gerar sanção automática."
        )

    source_ids = [event.camera_id, event.room_id]
    if person:
        source_ids.append(person["person_id"])
    source_ids.extend(schedule["schedule_id"] for schedule in relevant_schedules)
    source_ids.extend(incident["incident_id"] for incident in qualifying_incidents)
    source_ids.extend(policy["policy_id"] for policy in applicable_policies)

    context_snapshot = {
        "person": person,
        "room": room,
        "camera": {
            "camera_id": camera["camera_id"],
            "recognition_threshold": camera["recognition_threshold"],
        }
        if camera
        else None,
        "local_timestamp": local_timestamp.isoformat(),
        "schedule_match": bool(matching_schedules),
        "schedules_considered": relevant_schedules,
        "permission_match": permission_match,
        "entry_observation": {
            "identity_status": event.identity_status.value,
            "entry_evidence": event.entry_evidence.value,
            "recognition_source": event.recognition_source.value,
            "track_id": event.track_id,
            "recognition_model": event.recognition_model,
            **(
                {
                    "recognition_model_fingerprint": event.recognition_model_fingerprint
                }
                if event.recognition_model_fingerprint
                else {}
            ),
            "recognition_margin": event.recognition_margin,
            "face_quality": event.face_quality,
            "entry_confidence": event.entry_confidence,
            "evidence_captured_at": (
                event.evidence_captured_at.isoformat()
                if event.evidence_captured_at
                else None
            ),
        },
        "qualifying_incidents": qualifying_incidents,
        "policies": [
            {
                "policy_id": policy["policy_id"],
                "version": policy["version"],
                "title": policy["title"],
                "content": policy["content"],
            }
            for policy in applicable_policies
        ],
    }

    return Evaluation(
        decision=decision,
        risk_level=risk_level,
        risk_score=risk_score,
        reason_codes=reason_codes,
        source_ids=list(dict.fromkeys(source_ids)),
        narrative=narrative,
        alert_required=decision == Decision.ANOMALY,
        context_snapshot=context_snapshot,
    )
