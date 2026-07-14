from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DECISION_PT = {
    "AUTHORIZED": "Padrão autorizado",
    "JUSTIFIED": "Justificado",
    "ANOMALY": "Anomalia",
}
RISK_PT = {"LOW": "Baixo", "MEDIUM": "Médio", "HIGH": "Alto", "CRITICAL": "Crítico"}
DOOR_PT = {"GRANTED": "Liberada", "DENIED": "Negada", "NOT_REPORTED": "Não informado"}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#17324d"),
            fontSize=20,
            leading=24,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#17324d"),
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(
        ParagraphStyle(
            name="FinePrint",
            parent=styles["BodyText"],
            textColor=colors.HexColor("#52606d"),
            fontSize=7.5,
            leading=10,
        )
    )
    return styles


def _safe(value: Any) -> str:
    if value is None:
        return "—"
    return escape(str(value))


def _truncated(value: Any, limit: int = 700) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _local_time(value: str, timezone_name: str) -> str:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return timestamp.astimezone(ZoneInfo(timezone_name)).strftime("%d/%m/%Y %H:%M:%S %z")


def _event_timezone(event: dict[str, Any], fallback: str) -> str:
    return (
        event.get("context_snapshot", {}).get("room", {}).get("timezone")
        or fallback
    )


def _event_local_time(event: dict[str, Any], fallback: str) -> str:
    timezone_name = _event_timezone(event, fallback)
    return f"{_local_time(event['occurred_at'], timezone_name)} [{timezone_name}]"


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#52606d"))
    canvas.drawString(document.leftMargin, 9 * mm, "RAG-Audit · Relatório de auditoria")
    canvas.drawRightString(
        document.pagesize[0] - document.rightMargin,
        9 * mm,
        f"Página {document.page}",
    )
    canvas.restoreState()


def _key_value_table(rows: list[tuple[str, Any]], styles) -> Table:
    data = [
        [
            Paragraph(f"<b>{_safe(label)}</b>", styles["SmallBody"]),
            Paragraph(_safe(_truncated(value, 500)), styles["SmallBody"]),
        ]
        for label, value in rows
    ]
    table = Table(data, colWidths=[48 * mm, 125 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf2f7")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d9e2ec")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_event_pdf(event: dict[str, Any], timezone_name: str) -> bytes:
    buffer = BytesIO()
    styles = _styles()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"RAG-Audit - Evento {event['event_id']}",
        author="RAG-Audit",
    )
    alert = event.get("alert") or {}
    context = event.get("context_snapshot") or {}
    observation = context.get("entry_observation") or {}
    story = [
        Paragraph("RAG-Audit", styles["ReportTitle"]),
        Paragraph("Relatório individual de evento de acesso", styles["Heading2"]),
        Paragraph(
            f"Gerado em {datetime.now(ZoneInfo(timezone_name)).strftime('%d/%m/%Y %H:%M:%S %z')}",
            styles["FinePrint"],
        ),
        Spacer(1, 5 * mm),
        _key_value_table(
            [
                ("ID do evento", event["event_id"]),
                ("Data/hora do evento", _local_time(event["occurred_at"], timezone_name)),
                ("Pessoa", f"{event.get('person_name') or 'Não identificada'} ({event['person_id']})"),
                ("Cargo / área", f"{event.get('role_name') or '—'} / {event.get('department') or '—'}"),
                ("Sala", f"{event.get('room_name') or event['room_id']} ({event['room_id']})"),
                ("Câmera", event["camera_id"]),
                ("Evidência de entrada", observation.get("entry_evidence", "UNSPECIFIED")),
                ("Estado da identidade", observation.get("identity_status", "MATCHED")),
                ("Resultado físico da porta", DOOR_PT.get(event["door_result"], event["door_result"])),
                ("Classificação contextual", DECISION_PT.get(event["decision"], event["decision"])),
                ("Risco", f"{RISK_PT.get(event['risk_level'], event['risk_level'])} ({event['risk_score']}/100)"),
                ("Processamento interno da API", f"{event['processing_ms']:.2f} ms"),
                ("Atraso até a ingestão", f"{event['ingestion_delay_ms']:.2f} ms"),
                (
                    "Decisão desde o evento",
                    f"{event['decision_e2e_ms']:.2f} ms"
                    if event.get("decision_e2e_ms") is not None
                    else "—",
                ),
                ("Versão da política", event["policy_version"]),
                ("Status do alerta", alert.get("status", "Não aplicável")),
            ],
            styles,
        ),
        Paragraph("Relato descritivo", styles["Section"]),
        Paragraph(_safe(event["narrative"]), styles["BodyText"]),
        Paragraph("Motivos estruturados", styles["Section"]),
        Paragraph(_safe(", ".join(event["reason_codes"])), styles["BodyText"]),
        Paragraph("Evidências recuperadas", styles["Section"]),
    ]

    incidents = context.get("qualifying_incidents", [])
    policies = context.get("policies", [])
    evidence_rows: list[tuple[str, Any]] = [
        ("Dentro da escala", "Sim" if context.get("schedule_match") else "Não"),
        ("Permissão para a sala", "Sim" if context.get("permission_match") else "Não"),
        ("Track visual", observation.get("track_id") or "Não informado"),
        ("Motor de reconhecimento", observation.get("recognition_model") or "Não informado"),
        (
            "Chamados qualificáveis",
            ", ".join(f"{item['incident_id']} - {item['title']}" for item in incidents)
            or "Nenhum",
        ),
        (
            "Políticas aplicadas",
            ", ".join(f"{item['policy_id']} v{item['version']}" for item in policies)
            or "Nenhuma",
        ),
        ("IDs das fontes", ", ".join(event["source_ids"])),
    ]
    story.extend(
        [
            _key_value_table(evidence_rows, styles),
            Spacer(1, 7 * mm),
            Paragraph(
                "Nota: esta classificação apoia auditoria e revisão humana. Ela não comprova "
                "conduta irregular e não deve ser usada isoladamente para sanção ou decisão trabalhista.",
                styles["FinePrint"],
            ),
        ]
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def build_consolidated_pdf(
    events: list[dict[str, Any]],
    filters: dict[str, Any],
    timezone_name: str,
    generated_by: str,
) -> bytes:
    buffer = BytesIO()
    styles = _styles()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="RAG-Audit - Relatório consolidado",
        author="RAG-Audit",
    )
    counts = {
        key: sum(event["decision"] == value for event in events)
        for key, value in (
            ("padrão", "AUTHORIZED"),
            ("justificado", "JUSTIFIED"),
            ("anomalia", "ANOMALY"),
        )
    }
    active_filters = ", ".join(
        f"{key}={value}" for key, value in filters.items() if value not in (None, "")
    ) or "sem filtros adicionais"
    story = [
        Paragraph("RAG-Audit", styles["ReportTitle"]),
        Paragraph("Relatório consolidado de acessos", styles["Heading2"]),
        Paragraph(
            f"Gerado por {_safe(generated_by)} em "
            f"{datetime.now(ZoneInfo(timezone_name)).strftime('%d/%m/%Y %H:%M:%S %z')} · "
            f"Fuso de apresentação: {_safe(timezone_name)}",
            styles["FinePrint"],
        ),
        Paragraph(f"Filtros: {_safe(active_filters)}", styles["FinePrint"]),
        Spacer(1, 4 * mm),
        Paragraph(
            f"Total: <b>{len(events)}</b> · Padrão: <b>{counts['padrão']}</b> · "
            f"Justificados: <b>{counts['justificado']}</b> · Anomalias: <b>{counts['anomalia']}</b>",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
    ]

    if not events:
        story.append(Paragraph("Nenhum acesso encontrado para os filtros informados.", styles["BodyText"]))
    else:
        headers = ["Data/hora", "Pessoa", "Sala", "Classificação", "Risco", "Contexto", "Alerta"]
        data: list[list[Any]] = [
            [Paragraph(f"<b>{header}</b>", styles["SmallBody"]) for header in headers]
        ]
        for event in events:
            data.append(
                [
                    Paragraph(_safe(_event_local_time(event, timezone_name)), styles["SmallBody"]),
                    Paragraph(_safe(f"{event.get('person_name') or 'Não identificada'}\n{event['person_id']}"), styles["SmallBody"]),
                    Paragraph(_safe(event.get("room_name") or event["room_id"]), styles["SmallBody"]),
                    Paragraph(_safe(DECISION_PT.get(event["decision"], event["decision"])), styles["SmallBody"]),
                    Paragraph(_safe(RISK_PT.get(event["risk_level"], event["risk_level"])), styles["SmallBody"]),
                    Paragraph(_safe(_truncated(event["narrative"])), styles["SmallBody"]),
                    Paragraph(_safe((event.get("alert") or {}).get("status", "—")), styles["SmallBody"]),
                ]
            )
        table = Table(
            data,
            repeatRows=1,
            colWidths=[30 * mm, 34 * mm, 30 * mm, 29 * mm, 19 * mm, 91 * mm, 26 * mm],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)

        anomalies = [event for event in events if event["decision"] == "ANOMALY"]
        if anomalies:
            story.extend([PageBreak(), Paragraph("Anomalias para revisão", styles["Section"])])
            for event in anomalies:
                story.extend(
                    [
                        Paragraph(
                            f"<b>{_safe(event['event_id'])}</b> · {_safe(event.get('person_name') or event['person_id'])} · "
                            f"{_safe(_event_local_time(event, timezone_name))}",
                            styles["BodyText"],
                        ),
                        Paragraph(_safe(event["narrative"]), styles["SmallBody"]),
                        Spacer(1, 3 * mm),
                    ]
                )

    story.extend(
        [
            Spacer(1, 7 * mm),
            Paragraph(
                "Documento de apoio à auditoria. Revisão humana é obrigatória para qualquer medida decorrente de alerta.",
                styles["FinePrint"],
            ),
        ]
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
