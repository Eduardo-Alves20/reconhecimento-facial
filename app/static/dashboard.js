(() => {
  "use strict";

  const body = document.body;
  const timezone = body.dataset.timezone || "America/Sao_Paulo";

  const DECISIONS = {
    AUTHORIZED: { label: "Padrão", className: "authorized" },
    JUSTIFIED: { label: "Justificado", className: "justified" },
    ANOMALY: { label: "Anomalia", className: "anomaly" },
  };

  const RISKS = {
    LOW: { label: "Baixo", className: "low" },
    MEDIUM: { label: "Médio", className: "medium" },
    HIGH: { label: "Alto", className: "high" },
    CRITICAL: { label: "Crítico", className: "critical" },
  };

  const ALERTS = {
    NONE: { label: "Sem alerta", className: "none" },
    PENDING: { label: "Pendente", className: "pending" },
    RETRYING: { label: "Reenviando", className: "retrying" },
    SENT: { label: "Enviado", className: "sent" },
    FAILED: { label: "Falhou", className: "failed" },
    NOT_CONFIGURED: { label: "Sem canal", className: "failed" },
  };

  const DOOR_RESULTS = {
    GRANTED: "Liberada",
    DENIED: "Negada",
    NOT_REPORTED: "Não informado",
  };

  const REASON_TEXT = {
    UNKNOWN_PERSON: "Pessoa não cadastrada",
    INACTIVE_PERSON: "Cadastro da pessoa está inativo",
    LOW_RECOGNITION_CONFIDENCE: "Confiança do reconhecimento abaixo do limite",
    ROOM_PERMISSION_CONFIRMED: "Permissão para a sala confirmada",
    NO_ROOM_PERMISSION: "Sem permissão cadastrada para a sala",
    WITHIN_SCHEDULE: "Evento dentro da escala",
    OUTSIDE_SCHEDULE: "Evento fora da escala",
    QUALIFYING_INCIDENT: "Incidente qualificável encontrado",
    NO_QUALIFYING_INCIDENT: "Nenhum incidente qualificável encontrado",
    UNKNOWN_CAMERA: "Câmera não cadastrada",
  };

  const RANGE_LABELS = {
    all: "Todo o período",
    today: "Hoje",
    "24h": "Últimas 24 horas",
    "7d": "Últimos 7 dias",
  };

  const elements = {
    liveStatus: document.querySelector("#liveStatus"),
    liveStatusText: document.querySelector("#liveStatusText"),
    liveToggle: document.querySelector("#liveToggle"),
    lastUpdated: document.querySelector("#lastUpdated"),
    globalNotice: document.querySelector("#globalNotice"),
    globalNoticeText: document.querySelector("#globalNoticeText"),
    globalNoticeClose: document.querySelector("#globalNoticeClose"),
    filtersForm: document.querySelector("#filtersForm"),
    filterFrom: document.querySelector("#filterFrom"),
    filterTo: document.querySelector("#filterTo"),
    filterRoom: document.querySelector("#filterRoom"),
    filterPerson: document.querySelector("#filterPerson"),
    filterDecision: document.querySelector("#filterDecision"),
    filterRisk: document.querySelector("#filterRisk"),
    filterAlert: document.querySelector("#filterAlert"),
    filterQuery: document.querySelector("#filterQuery"),
    filterError: document.querySelector("#filterError"),
    clearFilters: document.querySelector("#clearFilters"),
    exportReport: document.querySelector("#exportReport"),
    quickRanges: Array.from(document.querySelectorAll(".quick-range")),
    metricsGrid: document.querySelector("#metricsGrid"),
    metricsScope: document.querySelector("#metricsScope"),
    metricTotal: document.querySelector("#metricTotal"),
    metricAuthorized: document.querySelector("#metricAuthorized"),
    metricAuthorizedShare: document.querySelector("#metricAuthorizedShare"),
    metricJustified: document.querySelector("#metricJustified"),
    metricAnomalies: document.querySelector("#metricAnomalies"),
    metricAlerts: document.querySelector("#metricAlerts"),
    metricAlertFailures: document.querySelector("#metricAlertFailures"),
    metricSla: document.querySelector("#metricSla"),
    metricP95: document.querySelector("#metricP95"),
    pageSize: document.querySelector("#pageSize"),
    refreshEvents: document.querySelector("#refreshEvents"),
    newEventsBanner: document.querySelector("#newEventsBanner"),
    newEventsText: document.querySelector("#newEventsText"),
    eventsTableRegion: document.querySelector("#eventsTableRegion"),
    tableScroll: document.querySelector(".table-scroll"),
    eventsBody: document.querySelector("#eventsBody"),
    eventsLoading: document.querySelector("#eventsLoading"),
    eventsEmpty: document.querySelector("#eventsEmpty"),
    eventsError: document.querySelector("#eventsError"),
    eventsErrorText: document.querySelector("#eventsErrorText"),
    retryEvents: document.querySelector("#retryEvents"),
    paginationSummary: document.querySelector("#paginationSummary"),
    pageIndicator: document.querySelector("#pageIndicator"),
    previousPage: document.querySelector("#previousPage"),
    nextPage: document.querySelector("#nextPage"),
    eventDialog: document.querySelector("#eventDialog"),
    closeDialog: document.querySelector("#closeDialog"),
    detailEventId: document.querySelector("#detailEventId"),
    detailLoading: document.querySelector("#detailLoading"),
    detailError: document.querySelector("#detailError"),
    detailErrorText: document.querySelector("#detailErrorText"),
    detailContent: document.querySelector("#detailContent"),
    detailPdfButton: document.querySelector("#detailPdfButton"),
    dialogCloseButtons: Array.from(document.querySelectorAll("[data-dialog-close]")),
    toastRegion: document.querySelector("#toastRegion"),
  };

  const state = {
    offset: 0,
    limit: Number(elements.pageSize.value) || 25,
    total: 0,
    activeRange: "all",
    paused: false,
    eventSource: null,
    listController: null,
    detailController: null,
    requestVersion: 0,
    currentDetailId: null,
    pendingEvents: 0,
    refreshTimer: null,
    initialised: false,
  };

  function createElement(tagName, className, text) {
    const node = document.createElement(tagName);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function asText(value, fallback = "—") {
    if (value === null || value === undefined || value === "") {
      return fallback;
    }
    return String(value);
  }

  function asNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : 0;
  }

  function formatNumber(value, maximumFractionDigits = 0) {
    return new Intl.NumberFormat("pt-BR", { maximumFractionDigits }).format(asNumber(value));
  }

  function dateFromValue(value) {
    if (value === null || value === undefined || value === "") {
      return null;
    }
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatDateTime(value) {
    const date = dateFromValue(value);
    if (!date) {
      return "—";
    }
    return new Intl.DateTimeFormat("pt-BR", {
      timeZone: timezone,
      dateStyle: "short",
      timeStyle: "medium",
    }).format(date);
  }

  function formatTableDate(value) {
    const date = dateFromValue(value);
    if (!date) {
      return { date: "—", time: "" };
    }
    return {
      date: new Intl.DateTimeFormat("pt-BR", {
        timeZone: timezone,
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
      }).format(date),
      time: new Intl.DateTimeFormat("pt-BR", {
        timeZone: timezone,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }).format(date),
    };
  }

  function zonedParts(date) {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: timezone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hourCycle: "h23",
    }).formatToParts(date);
    return Object.fromEntries(parts.filter((part) => part.type !== "literal").map((part) => [part.type, part.value]));
  }

  function zonedInputValue(date) {
    const parts = zonedParts(date);
    return `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}`;
  }

  function timezoneOffsetMilliseconds(date) {
    const parts = zonedParts(date);
    const representedAsUtc = Date.UTC(
      Number(parts.year),
      Number(parts.month) - 1,
      Number(parts.day),
      Number(parts.hour),
      Number(parts.minute),
      Number(parts.second),
    );
    return Math.round((representedAsUtc - date.getTime()) / 60000) * 60000;
  }

  function zonedInputToIso(value) {
    const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value || "");
    if (!match) {
      throw new Error("Informe uma data e hora válidas.");
    }
    const wallTime = Date.UTC(
      Number(match[1]),
      Number(match[2]) - 1,
      Number(match[3]),
      Number(match[4]),
      Number(match[5]),
      0,
    );
    let instant = wallTime;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const next = wallTime - timezoneOffsetMilliseconds(new Date(instant));
      if (Math.abs(next - instant) < 1000) {
        instant = next;
        break;
      }
      instant = next;
    }
    const date = new Date(instant);
    if (Number.isNaN(date.getTime())) {
      throw new Error("Informe uma data e hora válidas.");
    }
    return date.toISOString();
  }

  function isoToZonedInput(value) {
    const date = dateFromValue(value);
    return date ? zonedInputValue(date) : "";
  }

  function refreshQuickRangeValues() {
    if (!state.activeRange) {
      return;
    }
    const now = new Date();
    let fromDate = now;
    if (state.activeRange === "all") {
      // Todo o período: sem início e sem fim, para nada ficar escondido por filtro.
      elements.filterFrom.value = "";
      elements.filterTo.value = "";
      return;
    }
    if (state.activeRange === "today") {
      const today = zonedInputValue(now).slice(0, 10);
      elements.filterFrom.value = `${today}T00:00`;
    } else if (state.activeRange === "24h") {
      fromDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      elements.filterFrom.value = zonedInputValue(fromDate);
    } else if (state.activeRange === "7d") {
      fromDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      elements.filterFrom.value = zonedInputValue(fromDate);
    }
    elements.filterTo.value = zonedInputValue(now);
  }

  function updateRangeButtons() {
    for (const button of elements.quickRanges) {
      const active = button.dataset.range === state.activeRange;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    }
    elements.metricsScope.textContent = state.activeRange
      ? RANGE_LABELS[state.activeRange]
      : "Período personalizado";
  }

  function setQuickRange(range, shouldLoad = true) {
    state.activeRange = range;
    refreshQuickRangeValues();
    updateRangeButtons();
    if (shouldLoad) {
      state.offset = 0;
      loadDashboard({ showLoading: true });
    }
  }

  function markCustomRange() {
    state.activeRange = "";
    updateRangeButtons();
  }

  function validateAndBuildFilterParams() {
    refreshQuickRangeValues();
    const params = new URLSearchParams();
    let fromIso = "";
    let toIso = "";

    if (elements.filterFrom.value) {
      fromIso = zonedInputToIso(elements.filterFrom.value);
      params.set("from", fromIso);
    }
    // Períodos rápidos são consultas abertas até o instante atual. O campo continua
    // visível para orientação, mas omitir `to` evita perder eventos do minuto corrente.
    if (elements.filterTo.value && !state.activeRange) {
      toIso = zonedInputToIso(elements.filterTo.value);
      params.set("to", toIso);
    }
    if (fromIso && toIso && new Date(fromIso) > new Date(toIso)) {
      throw new Error("O início do período deve ser anterior ao fim.");
    }

    const values = [
      ["room_id", elements.filterRoom.value],
      ["user_id", elements.filterPerson.value.trim()],
      ["decision", elements.filterDecision.value],
      ["risk_level", elements.filterRisk.value],
      ["alert_status", elements.filterAlert.value],
      ["q", elements.filterQuery.value.trim()],
    ];
    for (const [name, value] of values) {
      if (value) {
        params.set(name, value);
      }
    }
    return params;
  }

  function showFilterError(message) {
    elements.filterError.textContent = message;
    elements.filterError.hidden = false;
  }

  function clearFilterError() {
    elements.filterError.textContent = "";
    elements.filterError.hidden = true;
  }

  function showNotice(message) {
    elements.globalNoticeText.textContent = message;
    elements.globalNotice.hidden = false;
  }

  function hideNotice() {
    elements.globalNotice.hidden = true;
    elements.globalNoticeText.textContent = "";
  }

  async function responseError(response) {
    let message = `A solicitação falhou (${response.status}).`;
    try {
      const payload = await response.json();
      if (payload && payload.detail) {
        if (Array.isArray(payload.detail)) {
          message = payload.detail.map((item) => item.msg || String(item)).join("; ");
        } else {
          message = String(payload.detail);
        }
      }
    } catch (_error) {
      // A resposta pode não ser JSON.
    }
    if (response.status === 401) {
      message = "Sua sessão administrativa não está autenticada. Recarregue a página e entre novamente.";
    }
    return new Error(message);
  }

  function sameOriginUrl(path) {
    return new URL(path, window.location.origin).toString();
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(sameOriginUrl(url), {
      ...options,
      cache: "no-store",
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    if (!response.ok) {
      throw await responseError(response);
    }
    return response.json();
  }

  function setTableMode(mode, message = "") {
    const hasData = mode === "data";
    elements.eventsLoading.hidden = mode !== "loading";
    elements.eventsEmpty.hidden = mode !== "empty";
    elements.eventsError.hidden = mode !== "error";
    elements.tableScroll.hidden = !hasData;
    elements.eventsTableRegion.setAttribute("aria-busy", String(mode === "loading"));
    if (mode === "error" && message) {
      elements.eventsErrorText.textContent = message;
    }
  }

  function createBadge(configuration, fallback) {
    const data = configuration || { label: fallback || "Não informado", className: "neutral" };
    return createElement("span", `badge badge--${data.className}`, data.label);
  }

  function tableCell(label) {
    const cell = document.createElement("td");
    cell.dataset.label = label;
    return cell;
  }

  function appendPrimarySecondary(cell, primary, secondary) {
    cell.append(createElement("span", "cell-primary", asText(primary)));
    if (secondary) {
      cell.append(createElement("span", "cell-secondary", secondary));
    }
  }

  function formatLatency(milliseconds) {
    const value = asNumber(milliseconds);
    if (value < 1000) {
      return `${formatNumber(value, value < 10 ? 1 : 0)} ms`;
    }
    return `${formatNumber(value / 1000, 2)} s`;
  }

  function renderEvents(items) {
    const fragment = document.createDocumentFragment();
    for (const event of items) {
      const row = document.createElement("tr");
      if (event.decision === "ANOMALY") {
        row.classList.add("event-row--anomaly");
      }

      const occurred = formatTableDate(event.occurred_at);
      const dateCell = tableCell("Data e hora");
      appendPrimarySecondary(dateCell, occurred.date, occurred.time);

      const personCell = tableCell("Pessoa");
      const personMeta = [event.role_name, event.person_id].filter(Boolean).join(" · ");
      const personInner = createElement("div", "person-inner");
      if (event.has_photo) {
        const thumb = document.createElement("img");
        thumb.className = "person-thumb";
        thumb.src = `/v1/access-events/${encodeURIComponent(event.event_id)}/photo?variant=thumb`;
        thumb.alt = "Rosto capturado pela câmera";
        thumb.loading = "lazy";
        thumb.title = "Foto capturada — clique para ampliar";
        thumb.addEventListener("click", () => openEventDetail(event.event_id));
        personInner.append(thumb);
      }
      const personText = createElement("div", "person-text");
      appendPrimarySecondary(personText, event.person_name || "Pessoa não identificada", personMeta);
      personInner.append(personText);
      personCell.append(personInner);

      const roomCell = tableCell("Sala");
      appendPrimarySecondary(roomCell, event.room_name || event.room_id, event.camera_id || "");

      const decisionCell = tableCell("Classificação");
      decisionCell.append(createBadge(DECISIONS[event.decision], asText(event.decision)));

      const riskCell = tableCell("Risco");
      riskCell.append(createBadge(RISKS[event.risk_level], asText(event.risk_level)));
      riskCell.append(createElement("span", "cell-secondary", `Score ${formatNumber(event.risk_score)}/100`));

      const narrativeCell = tableCell("Contexto");
      narrativeCell.append(createElement("span", "cell-narrative", asText(event.narrative, "Contexto não informado")));

      const alertCell = tableCell("Alerta");
      const alertStatus = event.alert ? event.alert.status : "NONE";
      alertCell.append(createBadge(ALERTS[alertStatus], asText(alertStatus)));

      const latencyCell = tableCell("Tempo");
      const latency = createElement("span", "latency", formatLatency(event.processing_ms));
      if (asNumber(event.processing_ms) >= 3000) {
        latency.classList.add("latency--slow");
        latency.title = "Fora do SLA de 3 segundos";
      }
      latencyCell.append(latency);

      const actionCell = tableCell("Ações");
      const action = createElement("button", "row-action", "›");
      action.type = "button";
      action.title = "Ver detalhes";
      action.setAttribute("aria-label", `Ver detalhes do evento ${asText(event.event_id)}`);
      action.addEventListener("click", () => openEventDetail(event.event_id));
      actionCell.append(action);

      row.append(
        dateCell,
        personCell,
        roomCell,
        decisionCell,
        riskCell,
        narrativeCell,
        alertCell,
        latencyCell,
        actionCell,
      );
      fragment.append(row);
    }
    elements.eventsBody.replaceChildren(fragment);
  }

  function renderPagination() {
    const pageCount = Math.max(1, Math.ceil(state.total / state.limit));
    const currentPage = Math.min(pageCount, Math.floor(state.offset / state.limit) + 1);
    if (state.total === 0) {
      elements.paginationSummary.textContent = "0 eventos";
    } else {
      const first = state.offset + 1;
      const last = Math.min(state.offset + state.limit, state.total);
      elements.paginationSummary.textContent = `${formatNumber(first)}–${formatNumber(last)} de ${formatNumber(state.total)} eventos`;
    }
    elements.pageIndicator.textContent = `Página ${formatNumber(currentPage)} de ${formatNumber(pageCount)}`;
    elements.previousPage.disabled = state.offset <= 0;
    elements.nextPage.disabled = state.offset + state.limit >= state.total;
  }

  function renderMetrics(metrics) {
    const total = asNumber(metrics.total);
    const authorized = asNumber(metrics.authorized);
    const failures = asNumber(metrics.alerts_failed);
    elements.metricTotal.textContent = formatNumber(total);
    elements.metricAuthorized.textContent = formatNumber(authorized);
    elements.metricAuthorizedShare.textContent = total
      ? `${formatNumber((authorized * 100) / total, 1)}% do total`
      : "baixo risco";
    elements.metricJustified.textContent = formatNumber(metrics.justified);
    elements.metricAnomalies.textContent = formatNumber(metrics.anomalies);
    elements.metricAlerts.textContent = formatNumber(metrics.alerts_sent);
    elements.metricAlertFailures.textContent = `${formatNumber(failures)} ${failures === 1 ? "falha ou reenvio" : "falhas ou reenvios"}`;
    elements.metricSla.textContent = `${formatNumber(metrics.api_sla_percentage ?? metrics.sla_percentage, 1)}%`;
    elements.metricP95.textContent = `API p95: ${formatLatency(metrics.p95_processing_ms)}`;
    elements.metricsGrid.setAttribute("aria-busy", "false");
  }

  function syncDashboardUrl(filterParams) {
    const pageParams = new URLSearchParams(filterParams);
    // "Tudo" é o padrão e não vai para a URL: assim um F5 volta ao estado livre
    // em vez de congelar o recorte que estava aberto.
    if (state.activeRange && state.activeRange !== "all") {
      pageParams.set("range", state.activeRange);
    }
    if (state.offset) {
      pageParams.set("offset", String(state.offset));
    }
    if (state.limit !== 25) {
      pageParams.set("limit", String(state.limit));
    }
    const query = pageParams.toString();
    const nextUrl = query ? `${window.location.pathname}?${query}` : window.location.pathname;
    window.history.replaceState(null, "", nextUrl);
  }

  async function loadDashboard({ showLoading = false } = {}) {
    clearFilterError();
    let filterParams;
    try {
      filterParams = validateAndBuildFilterParams();
    } catch (error) {
      showFilterError(error.message);
      return;
    }

    state.requestVersion += 1;
    const requestVersion = state.requestVersion;
    if (state.listController) {
      state.listController.abort();
    }
    state.listController = new AbortController();
    const { signal } = state.listController;

    const eventParams = new URLSearchParams(filterParams);
    eventParams.set("limit", String(state.limit));
    eventParams.set("offset", String(state.offset));

    if (showLoading) {
      elements.eventsBody.replaceChildren();
      setTableMode("loading");
      elements.metricsGrid.setAttribute("aria-busy", "true");
    }

    try {
      const [eventPayload, metrics] = await Promise.all([
        fetchJson(`/v1/access-events?${eventParams.toString()}`, { signal }),
        fetchJson(`/v1/metrics?${filterParams.toString()}`, { signal }),
      ]);
      if (requestVersion !== state.requestVersion) {
        return;
      }

      const items = Array.isArray(eventPayload.items) ? eventPayload.items : [];
      state.total = asNumber(eventPayload.total);
      renderEvents(items);
      renderMetrics(metrics || {});
      renderPagination();
      setTableMode(items.length ? "data" : "empty");
      elements.lastUpdated.textContent = formatDateTime(eventPayload.generated_at || new Date().toISOString());
      state.pendingEvents = 0;
      elements.newEventsBanner.hidden = true;
      syncDashboardUrl(filterParams);
      hideNotice();
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }
      if (showLoading || !elements.eventsBody.children.length) {
        setTableMode("error", error.message);
      } else {
        showNotice(`Os dados não puderam ser atualizados: ${error.message}`);
      }
      elements.metricsGrid.setAttribute("aria-busy", "false");
    }
  }

  async function loadRooms() {
    const selectedRoom = elements.filterRoom.dataset.pendingValue || elements.filterRoom.value;
    try {
      const payload = await fetchJson("/v1/rooms");
      const fragment = document.createDocumentFragment();
      for (const room of payload.items || []) {
        const option = document.createElement("option");
        option.value = asText(room.room_id, "");
        const criticality = room.criticality ? ` · ${room.criticality}` : "";
        option.textContent = `${asText(room.display_name, room.room_id)}${criticality}`;
        fragment.append(option);
      }
      elements.filterRoom.append(fragment);
      if (selectedRoom) {
        elements.filterRoom.value = selectedRoom;
      }
      delete elements.filterRoom.dataset.pendingValue;
    } catch (error) {
      showNotice(`A lista de salas não pôde ser carregada: ${error.message}`);
    }
  }

  function setConnectionStatus(status, text) {
    elements.liveStatus.classList.remove(
      "connection-status--connecting",
      "connection-status--online",
      "connection-status--offline",
      "connection-status--paused",
    );
    elements.liveStatus.classList.add(`connection-status--${status}`);
    elements.liveStatusText.textContent = text;
  }

  function updatePendingBanner() {
    if (!state.pendingEvents) {
      elements.newEventsBanner.hidden = true;
      return;
    }
    elements.newEventsText.textContent = state.pendingEvents === 1
      ? "1 novo evento recebido"
      : `${formatNumber(state.pendingEvents)} novos eventos recebidos`;
    elements.newEventsBanner.hidden = false;
  }

  function showToast(title, message, danger = false) {
    const toast = createElement("div", danger ? "toast toast--danger" : "toast");
    toast.append(createElement("strong", "", title), createElement("span", "", message));
    elements.toastRegion.append(toast);
    window.setTimeout(() => toast.remove(), 6500);
  }

  function scheduleLiveRefresh(message) {
    state.pendingEvents += 1;
    const canRefreshImmediately =
      state.offset === 0
      && document.visibilityState === "visible"
      && !elements.eventDialog.open
      && Boolean(state.activeRange);

    if (canRefreshImmediately) {
      window.clearTimeout(state.refreshTimer);
      state.refreshTimer = window.setTimeout(() => loadDashboard({ showLoading: false }), 300);
    } else {
      updatePendingBanner();
    }

    if (message && message.decision === "ANOMALY") {
      const critical = message.risk_level === "CRITICAL";
      showToast(
        critical ? "Anomalia crítica recebida" : "Nova anomalia recebida",
        "O evento foi classificado para revisão humana.",
        true,
      );
    }
  }

  function disconnectStream() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  function connectStream() {
    disconnectStream();
    if (state.paused) {
      setConnectionStatus("paused", "Atualização pausada");
      return;
    }
    setConnectionStatus("connecting", "Conectando…");
    const source = new EventSource(sameOriginUrl("/v1/access-events/stream"));
    state.eventSource = source;

    source.addEventListener("open", () => {
      if (state.eventSource === source) {
        setConnectionStatus("online", "Ao vivo");
      }
    });

    source.addEventListener("access-event", (event) => {
      let message = null;
      try {
        message = JSON.parse(event.data);
      } catch (_error) {
        showNotice("Um evento ao vivo chegou em formato inesperado; atualize a lista para conferir.");
      }
      scheduleLiveRefresh(message);
    });

    source.addEventListener("error", () => {
      if (state.eventSource !== source || state.paused) {
        return;
      }
      const status = source.readyState === EventSource.CLOSED ? "offline" : "connecting";
      const text = status === "offline" ? "Desconectado" : "Reconectando…";
      setConnectionStatus(status, text);
    });
  }

  function toggleLiveUpdates() {
    state.paused = !state.paused;
    elements.liveToggle.setAttribute("aria-pressed", String(state.paused));
    elements.liveToggle.textContent = state.paused ? "Retomar atualização" : "Pausar atualização";
    if (state.paused) {
      disconnectStream();
      setConnectionStatus("paused", "Atualização pausada");
    } else {
      connectStream();
      loadDashboard({ showLoading: false });
    }
  }

  function definitionGrid(entries) {
    const list = createElement("dl", "definition-grid");
    for (const [label, value] of entries) {
      const pair = createElement("div", "definition-pair");
      pair.append(createElement("dt", "", label), createElement("dd", "", asText(value)));
      list.append(pair);
    }
    return list;
  }

  function detailSection(title, wide = false) {
    const section = createElement("section", wide ? "detail-section detail-section--wide" : "detail-section");
    section.append(createElement("h3", "", title));
    return section;
  }

  function evidenceList(items, emptyMessage) {
    if (!items.length) {
      return createElement("p", "empty-evidence", emptyMessage);
    }
    const list = createElement("ul", "evidence-list");
    for (const item of items) {
      const row = createElement("li", "evidence-item");
      row.append(createElement("strong", "", item.title));
      if (item.detail) {
        row.append(createElement("span", "", item.detail));
      }
      list.append(row);
    }
    return list;
  }

  function codeList(values) {
    const list = createElement("ul", "code-list");
    for (const value of values || []) {
      const item = createElement("li", "code-chip", asText(value));
      if (REASON_TEXT[value]) {
        item.title = REASON_TEXT[value];
      }
      list.append(item);
    }
    return list;
  }

  function jsonDetails(summary, value) {
    const details = createElement("details", "json-details");
    details.append(createElement("summary", "", summary));
    const pre = createElement("pre", "json-block");
    try {
      pre.textContent = JSON.stringify(value, null, 2);
    } catch (_error) {
      pre.textContent = "Conteúdo indisponível.";
    }
    details.append(pre);
    return details;
  }

  function renderEventDetail(event) {
    const fragment = document.createDocumentFragment();
    const context = event.context_snapshot || {};

    const overview = createElement("div", "detail-overview");
    const hasPhoto = Boolean(event.raw_payload && event.raw_payload.evidence_ref);
    if (hasPhoto) {
      const photoUrl = `/v1/access-events/${encodeURIComponent(event.event_id)}/photo`;
      const photoLink = document.createElement("a");
      photoLink.className = "detail-photo-link";
      photoLink.href = photoUrl;
      photoLink.target = "_blank";
      photoLink.rel = "noopener";
      photoLink.title = "Abrir a foto completa em tamanho real para avaliar";
      const photo = document.createElement("img");
      photo.className = "detail-photo";
      photo.src = photoUrl;
      photo.alt = "Cena capturada pela câmera no momento do acesso";
      photoLink.append(photo);
      const hint = createElement("span", "detail-photo-hint", "Clique para ampliar");
      photoLink.append(hint);
      overview.append(photoLink);
    }
    const identity = createElement("div", "detail-overview__identity");
    identity.append(
      createElement("strong", "", event.person_name || "Pessoa não identificada"),
      createElement("span", "", [event.role_name, event.department, event.person_id].filter(Boolean).join(" · ")),
    );
    const badges = createElement("div", "detail-badges");
    badges.append(
      createBadge(DECISIONS[event.decision], asText(event.decision)),
      createBadge(RISKS[event.risk_level], asText(event.risk_level)),
    );
    overview.append(identity, badges);
    fragment.append(overview);

    const narrative = createElement("section", "narrative-block");
    narrative.append(
      createElement("h3", "", "Relatório descritivo"),
      createElement("p", "", asText(event.narrative, "Nenhum relatório descritivo foi registrado.")),
    );
    fragment.append(narrative);

    const grid = createElement("div", "detail-grid");

    const accessSection = detailSection("Identificação e acesso");
    const confidence = event.recognition_confidence === null || event.recognition_confidence === undefined
      ? "Não informada"
      : `${formatNumber(asNumber(event.recognition_confidence) * 100, 1)}%`;
    accessSection.append(definitionGrid([
      ["Ocorrido em", formatDateTime(event.occurred_at)],
      ["Sala", event.room_name || event.room_id],
      ["Câmera", event.camera_id],
      ["Resultado da porta", DOOR_RESULTS[event.door_result] || event.door_result],
      ["Confiança facial", confidence],
      ["ID da pessoa", event.person_id],
    ]));

    const evaluationSection = detailSection("Avaliação contextual");
    evaluationSection.append(definitionGrid([
      ["Classificação", (DECISIONS[event.decision] || {}).label || event.decision],
      ["Nível de risco", (RISKS[event.risk_level] || {}).label || event.risk_level],
      ["Score", `${formatNumber(event.risk_score)}/100`],
      ["Versão da política", event.policy_version],
      ["Permissão para sala", context.permission_match === true ? "Confirmada" : "Não confirmada"],
      ["Compatível com escala", context.schedule_match === true ? "Sim" : "Não"],
    ]));
    if (Array.isArray(event.reason_codes) && event.reason_codes.length) {
      evaluationSection.append(codeList(event.reason_codes));
    }

    grid.append(accessSection, evaluationSection);

    const evidenceSection = detailSection("Evidências consideradas", true);
    const incidents = Array.isArray(context.qualifying_incidents) ? context.qualifying_incidents : [];
    const incidentItems = incidents.map((incident) => ({
      title: `${asText(incident.incident_id, "Incidente")} · ${asText(incident.title, "Sem título")}`,
      detail: [incident.severity, incident.status, incident.room_id].filter(Boolean).join(" · "),
    }));
    const policies = Array.isArray(context.policies) ? context.policies : [];
    const policyItems = policies.map((policy) => ({
      title: `${asText(policy.policy_id, "Política")} · ${asText(policy.title, "Sem título")}`,
      detail: [policy.version ? `versão ${policy.version}` : "", policy.content].filter(Boolean).join(" — "),
    }));
    const schedules = Array.isArray(context.schedules_considered) ? context.schedules_considered : [];
    const scheduleItems = schedules.map((schedule) => ({
      title: asText(schedule.schedule_id, "Escala cadastrada"),
      detail: `Dia ${asText(schedule.weekday)} · ${asText(schedule.start_time)}–${asText(schedule.end_time)}`,
    }));

    evidenceSection.append(createElement("h4", "", "Incidentes qualificáveis"));
    evidenceSection.append(evidenceList(incidentItems, "Nenhum incidente qualificável foi encontrado."));
    evidenceSection.append(createElement("h4", "", "Políticas aplicadas"));
    evidenceSection.append(evidenceList(policyItems, "Nenhuma política adicional foi associada."));
    evidenceSection.append(createElement("h4", "", "Escalas consultadas"));
    evidenceSection.append(evidenceList(scheduleItems, "Nenhuma escala aplicável foi registrada."));
    grid.append(evidenceSection);

    const operationSection = detailSection("Processamento");
    operationSection.append(definitionGrid([
      ["Recebido em", formatDateTime(event.received_at)],
      ["Concluído em", formatDateTime(event.processed_at)],
      ["Processamento interno da API", formatLatency(event.processing_ms)],
      ["Atraso câmera → API", formatLatency(event.ingestion_delay_ms)],
      ["Tempo câmera → decisão", formatLatency(event.decision_e2e_ms)],
      ["SLA interno da API < 3 s", asNumber(event.processing_ms) < 3000 ? "Cumprido" : "Fora do SLA"],
      ["SLA total desde a câmera < 3 s", asNumber(event.decision_e2e_ms) >= 0 && asNumber(event.decision_e2e_ms) < 3000 ? "Cumprido" : "Fora do SLA"],
      ["Criado em", formatDateTime(event.created_at)],
      ["Fuso exibido", timezone],
    ]));

    const alertSection = detailSection("Entrega do alerta");
    const alert = event.alert;
    if (alert) {
      alertSection.append(definitionGrid([
        ["Status", (ALERTS[alert.status] || {}).label || alert.status],
        ["Canal", alert.channel],
        ["Tentativas", formatNumber(alert.attempts)],
        ["Enviado em", formatDateTime(alert.sent_at)],
        ["ID do alerta", alert.alert_id],
        ["Último erro", alert.last_error || "Nenhum"],
      ]));
    } else {
      alertSection.append(createElement("p", "empty-evidence", event.alert_required
        ? "O evento exige alerta, mas a entrega ainda não foi registrada."
        : "Este evento não exige alerta de anomalia."));
    }
    grid.append(operationSection, alertSection);

    const auditSection = detailSection("Integridade e dados de origem", true);
    auditSection.append(definitionGrid([
      ["ID do evento", event.event_id],
      ["Fontes consultadas", Array.isArray(event.source_ids) ? event.source_ids.join(", ") : "—"],
      ["Horário local avaliado", context.local_timestamp ? formatDateTime(context.local_timestamp) : "—"],
      ["Alerta requerido", event.alert_required ? "Sim" : "Não"],
    ]));
    auditSection.append(jsonDetails("Ver JSON original recebido", event.raw_payload || {}));
    auditSection.append(jsonDetails("Ver snapshot do contexto", context));
    grid.append(auditSection);

    fragment.append(grid);
    elements.detailContent.replaceChildren(fragment);
  }

  function closeEventDialog() {
    if (elements.eventDialog.open) {
      elements.eventDialog.close();
    }
    if (state.detailController) {
      state.detailController.abort();
      state.detailController = null;
    }
    state.currentDetailId = null;
  }

  async function openEventDetail(eventId) {
    if (!eventId) {
      return;
    }
    state.currentDetailId = String(eventId);
    elements.detailEventId.textContent = state.currentDetailId;
    elements.detailContent.replaceChildren();
    elements.detailContent.hidden = true;
    elements.detailError.hidden = true;
    elements.detailLoading.hidden = false;
    elements.detailPdfButton.disabled = true;
    if (!elements.eventDialog.open) {
      elements.eventDialog.showModal();
    }

    if (state.detailController) {
      state.detailController.abort();
    }
    state.detailController = new AbortController();
    try {
      const event = await fetchJson(`/v1/access-events/${encodeURIComponent(state.currentDetailId)}`, {
        signal: state.detailController.signal,
      });
      if (String(event.event_id) !== state.currentDetailId) {
        return;
      }
      renderEventDetail(event);
      elements.detailLoading.hidden = true;
      elements.detailContent.hidden = false;
      elements.detailPdfButton.disabled = false;
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }
      elements.detailLoading.hidden = true;
      elements.detailErrorText.textContent = error.message;
      elements.detailError.hidden = false;
    }
  }

  function safeFilename(value, fallback) {
    const cleaned = String(value || "")
      .replace(/[^\w.\-À-ɏ]/g, "_")
      .slice(0, 160);
    return cleaned || fallback;
  }

  function filenameFromResponse(response, fallback) {
    const disposition = response.headers.get("Content-Disposition") || "";
    const encodedMatch = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
    const plainMatch = /filename="?([^";]+)"?/i.exec(disposition);
    let filename = fallback;
    if (encodedMatch) {
      try {
        filename = decodeURIComponent(encodedMatch[1]);
      } catch (_error) {
        filename = encodedMatch[1];
      }
    } else if (plainMatch) {
      filename = plainMatch[1];
    }
    return safeFilename(filename, fallback);
  }

  function setPdfButton(button, busy, normalLabel) {
    button.disabled = busy;
    button.replaceChildren();
    if (!busy) {
      button.append(createElement("span", "", "↓"));
    }
    button.append(document.createTextNode(busy ? " Gerando PDF…" : ` ${normalLabel}`));
  }

  async function downloadPdf(url, button, normalLabel, fallbackFilename) {
    setPdfButton(button, true, normalLabel);
    try {
      const response = await fetch(sameOriginUrl(url), { headers: { Accept: "application/pdf" } });
      if (!response.ok) {
        throw await responseError(response);
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = filenameFromResponse(response, fallbackFilename);
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      showToast("Relatório pronto", "O download do PDF foi iniciado.");
    } catch (error) {
      showNotice(`Não foi possível gerar o PDF: ${error.message}`);
    } finally {
      setPdfButton(button, false, normalLabel);
      if (button === elements.detailPdfButton && !state.currentDetailId) {
        button.disabled = true;
      }
    }
  }

  async function exportConsolidatedReport() {
    clearFilterError();
    let params;
    try {
      params = validateAndBuildFilterParams();
    } catch (error) {
      showFilterError(error.message);
      return;
    }
    const query = params.toString();
    const url = query ? `/v1/reports/access-events.pdf?${query}` : "/v1/reports/access-events.pdf";
    await downloadPdf(url, elements.exportReport, "Gerar PDF", "rag-audit_acessos.pdf");
  }

  function restoreFiltersFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const range = params.get("range");
    if (range && RANGE_LABELS[range]) {
      state.activeRange = range;
      refreshQuickRangeValues();
    } else if (params.has("from") || params.has("to")) {
      state.activeRange = "";
      elements.filterFrom.value = isoToZonedInput(params.get("from"));
      elements.filterTo.value = isoToZonedInput(params.get("to"));
    } else {
      state.activeRange = "all";
      refreshQuickRangeValues();
    }

    elements.filterRoom.dataset.pendingValue = params.get("room_id") || "";
    elements.filterPerson.value = params.get("user_id") || "";
    elements.filterDecision.value = params.get("decision") || "";
    elements.filterRisk.value = params.get("risk_level") || "";
    elements.filterAlert.value = params.get("alert_status") || "";
    elements.filterQuery.value = params.get("q") || "";

    const limit = Number(params.get("limit"));
    if ([25, 50, 100].includes(limit)) {
      state.limit = limit;
      elements.pageSize.value = String(limit);
    }
    const offset = Number(params.get("offset"));
    state.offset = Number.isInteger(offset) && offset >= 0 ? offset : 0;
    updateRangeButtons();
  }

  function clearAllFilters() {
    elements.filterRoom.value = "";
    elements.filterPerson.value = "";
    elements.filterDecision.value = "";
    elements.filterRisk.value = "";
    elements.filterAlert.value = "";
    elements.filterQuery.value = "";
    clearFilterError();
    setQuickRange("all", true);
  }

  function bindEvents() {
    elements.filtersForm.addEventListener("submit", (event) => {
      event.preventDefault();
      state.offset = 0;
      loadDashboard({ showLoading: true });
    });

    elements.filterFrom.addEventListener("input", markCustomRange);
    elements.filterTo.addEventListener("input", markCustomRange);
    for (const button of elements.quickRanges) {
      button.addEventListener("click", () => setQuickRange(button.dataset.range));
    }

    elements.clearFilters.addEventListener("click", clearAllFilters);
    elements.exportReport.addEventListener("click", exportConsolidatedReport);
    elements.refreshEvents.addEventListener("click", () => loadDashboard({ showLoading: false }));
    elements.retryEvents.addEventListener("click", () => loadDashboard({ showLoading: true }));
    elements.newEventsBanner.addEventListener("click", () => {
      state.offset = 0;
      loadDashboard({ showLoading: false });
    });

    elements.pageSize.addEventListener("change", () => {
      state.limit = Number(elements.pageSize.value) || 25;
      state.offset = 0;
      loadDashboard({ showLoading: true });
    });
    elements.previousPage.addEventListener("click", () => {
      state.offset = Math.max(0, state.offset - state.limit);
      loadDashboard({ showLoading: true });
    });
    elements.nextPage.addEventListener("click", () => {
      if (state.offset + state.limit < state.total) {
        state.offset += state.limit;
        loadDashboard({ showLoading: true });
      }
    });

    elements.liveToggle.addEventListener("click", toggleLiveUpdates);
    elements.globalNoticeClose.addEventListener("click", hideNotice);
    elements.closeDialog.addEventListener("click", closeEventDialog);
    for (const button of elements.dialogCloseButtons) {
      button.addEventListener("click", closeEventDialog);
    }
    elements.eventDialog.addEventListener("click", (event) => {
      if (event.target === elements.eventDialog) {
        closeEventDialog();
      }
    });
    elements.eventDialog.addEventListener("close", () => {
      if (state.detailController) {
        state.detailController.abort();
        state.detailController = null;
      }
      state.currentDetailId = null;
    });
    elements.detailPdfButton.addEventListener("click", () => {
      if (!state.currentDetailId) {
        return;
      }
      const eventId = encodeURIComponent(state.currentDetailId);
      downloadPdf(
        `/v1/access-events/${eventId}/report.pdf`,
        elements.detailPdfButton,
        "Baixar PDF do evento",
        `rag-audit_evento_${safeFilename(state.currentDetailId, "evento")}.pdf`,
      );
    });

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && state.pendingEvents && !state.paused) {
        loadDashboard({ showLoading: false });
      }
    });
    window.addEventListener("beforeunload", disconnectStream);
  }

  async function initialise() {
    restoreFiltersFromUrl();
    bindEvents();
    await loadRooms();
    await loadDashboard({ showLoading: true });
    connectStream();
    state.initialised = true;
  }

  initialise().catch((error) => {
    setTableMode("error", error.message || "Falha inesperada ao iniciar o painel.");
    setConnectionStatus("offline", "Falha ao iniciar");
  });
})();
