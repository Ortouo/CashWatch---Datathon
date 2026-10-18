(() => {
  "use strict";

  const MAX_UPLOAD_MIB = 100;
  const POLL_DEADLINE_MS = 300_000;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const byId = (id) => document.getElementById(id);
  const ui = {
    server: byId("server-status"), serverText: byId("server-status-text"),
    accessForm: byId("access-form"), key: byId("api-key"), toggleKey: byId("toggle-key"),
    connect: byId("connect-button"), connection: byId("connection-message"),
    input: byId("video-input"), drop: byId("drop-zone"), uploadHelp: byId("upload-help"),
    formatChip: byId("format-chip"), filePanel: byId("file-panel"), video: byId("source-video"),
    mediaWarning: byId("media-warning"), fileName: byId("file-name"), fileInfo: byId("file-info"),
    changeFile: byId("change-file"), sourceMessage: byId("source-message"),
    analyze: byId("analyze-button"), cancel: byId("cancel-button"), prerequisite: byId("analysis-prerequisite"),
    progressPanel: byId("progress-panel"), progressLabel: byId("progress-label"),
    progressPercent: byId("progress-percent"), progress: byId("upload-progress"),
    progressDescription: byId("progress-description"), elapsed: byId("elapsed-time"),
    resultRegion: byId("result-region"), resultState: byId("result-state"),
    empty: byId("empty-result"), busy: byId("busy-result"), busyTitle: byId("busy-title"),
    busyDescription: byId("busy-description"), busyJob: byId("busy-job-id"),
    error: byId("error-result"), errorTitle: byId("error-title"), errorDescription: byId("error-description"),
    errorJob: byId("error-job-id"), retry: byId("retry-button"),
    completed: byId("completed-result"), verdict: byId("verdict-card"), verdictIcon: byId("verdict-icon"),
    verdictTitle: byId("verdict-title"), verdictDescription: byId("verdict-description"),
    resultJob: byId("result-job-id"), download: byId("download-button"), warnings: byId("review-warnings"),
    events: byId("events-list"), noEvents: byId("no-events-note"),
    details: byId("analysis-details"), detailsList: byId("details-list"),
    sourceDetails: byId("source-details"), sources: byId("sources-list"),
    dialog: byId("frame-dialog"), dialogImage: byId("dialog-image"), dialogTime: byId("dialog-timecode"),
    closeDialog: byId("close-dialog"), seekFromDialog: byId("seek-from-dialog"), announcer: byId("announcer"),
    manualFields: byId("manual-fields"), manualStatus: byId("manual-status"),
    manualGroup: byId("manual-group"), manualCamera: byId("manual-camera"), manualNote: byId("manual-note"),
    manualTime: byId("manual-time"), manualMarkNote: byId("manual-mark-note"),
    manualAddMark: byId("manual-add-mark"), manualMarks: byId("manual-marks"),
    manualRuleId: byId("manual-rule-id"), manualApplicability: byId("manual-applicability"),
    manualRuleNote: byId("manual-rule-note"), manualSaveRule: byId("manual-save-rule"),
    manualRules: byId("manual-rules"), manualRuleCount: byId("manual-rule-count"), manualExport: byId("manual-export"),
  };

  // API keys intentionally have no persistent storage or URL representation.
  const state = {
    apiKey: "", authorized: false, checkingAccess: false, capabilities: null, blocker: null,
    acceptedExtensions: [".mov"], maxUploadMiB: MAX_UPLOAD_MIB,
    file: null, videoUrl: null, busy: false, phase: "idle", operation: 0, accessCheck: 0,
    xhr: null, controllers: new Set(), pollTimer: null, resolvePollWait: null, clock: null,
    startedAt: 0, jobId: null, result: null, details: null, eventCards: [],
    warnings: new Set(), modalTimestamp: 0, healthCheck: 0,
  };

  // Manual records never enter the analysis API or browser persistent storage.
  const manual = { byHash: new Map(), byFile: new WeakMap(), active: null, generation: 0 };
  const applicabilityLabels = {
    unknown: "Применимость не установлена", confirmed: "Применимость подтверждена", not_applicable: "Правило не применимо",
  };

  function activeManualDraft() {
    return manual.active?.file === state.file ? manual.active.draft : null;
  }

  function manualChanged() {
    const draft = activeManualDraft();
    if (!draft) return;
    draft.revision += 1;
    showManualStatus();
  }

  function showManualStatus(message = "") {
    const draft = activeManualDraft();
    const ready = draft?.hash_status === "ready";
    ui.manualFields.disabled = !ready;
    ui.manualExport.disabled = !ready;
    ui.manualStatus.textContent = message || (draft?.hash_status === "failed"
      ? "Не удалось вычислить SHA-256. Журнал и экспорт заблокированы. Повторно выберите исходник; нужен браузер с Web Crypto на localhost или HTTPS."
      : !ready ? "Вычисляем SHA-256 на устройстве. Просмотр видео доступен."
      : draft.revision > draft.exported_revision ? "Изменения хранятся в памяти страницы. Скачайте JSON, чтобы сохранить их."
      : "SHA-256 готов. Журнал доступен без модели и подключения к API.");
    updateManualTime();
  }

  function populateManualFields() {
    const draft = activeManualDraft();
    if (!draft) return;
    for (const [input, field] of manualInputs) input.value = draft[field];
    renderManualLists();
    showManualStatus();
  }

  async function attachManualFile(file) {
    const generation = ++manual.generation;
    let draft = manual.byFile.get(file);
    if (!draft) {
      draft = {
        sha256: null, hash_status: "pending", duration_sec: null,
        group_id: "", camera_id: "", note: "", marks: [], rule_reviews: [],
        mark_note: "", rule_id_input: "", applicability_input: "unknown", rule_note_input: "",
        revision: 0, exported_revision: 0,
      };
      manual.byFile.set(file, draft);
    }
    manual.active = { file, draft };
    populateManualFields();
    if (draft.hash_status === "ready") return;
    try {
      if (!globalThis.crypto?.subtle) throw new Error("Web Crypto unavailable");
      const bytes = await file.arrayBuffer();
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      const sha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
      if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error("Invalid digest");
      // Fields stay disabled while hashing, so an existing same-content draft can be restored without overwriting edits.
      draft = manual.byHash.get(sha256) || draft;
      draft.sha256 = sha256;
      draft.hash_status = "ready";
      manual.byHash.set(sha256, draft);
      manual.byFile.set(file, draft);
      if (generation !== manual.generation || state.file !== file) return;
      manual.active = { file, draft };
      populateManualFields();
    } catch {
      draft.hash_status = "failed";
      if (generation === manual.generation && state.file === file) showManualStatus();
    }
  }

  function updateManualTime() {
    const draft = activeManualDraft();
    const duration = ui.video.duration;
    const ready = Boolean(draft && !ui.video.error && ui.video.readyState > 0 && finiteNumber(duration) && duration > 0);
    if (draft) draft.duration_sec = ready ? duration : null;
    const time = ui.video.currentTime;
    const validTime = ready && finiteNumber(time) && time <= duration;
    ui.manualTime.textContent = validTime ? `${formatTimecode(time)} · ${formatSeconds(time)}` : "—";
    ui.manualAddMark.disabled = !validTime || draft?.hash_status !== "ready";
  }

  function renderManualLists() {
    const draft = activeManualDraft();
    ui.manualMarks.replaceChildren();
    ui.manualRules.replaceChildren();
    ui.manualRuleCount.textContent = String(draft?.rule_reviews.length || 0);
    if (!draft) return;
    const addItem = (list, heading, note, remove, seek = null) => {
      const item = element("li", "manual-item");
      const copy = element("div", "manual-item-copy");
      const title = element(seek ? "button" : "strong", seek ? "text-button" : "", heading);
      if (seek) { title.type = "button"; title.addEventListener("click", seek); }
      copy.append(title);
      if (note) copy.append(element("p", "", note));
      const button = element("button", "text-button", "Удалить");
      button.type = "button";
      button.setAttribute("aria-label", `Удалить: ${heading}`);
      button.addEventListener("click", () => { remove(); manualChanged(); renderManualLists(); });
      item.append(copy, button);
      list.append(item);
    };
    draft.marks.forEach((mark, index) => addItem(ui.manualMarks, formatTimecode(mark.timestamp_sec), mark.note,
      () => draft.marks.splice(index, 1), () => seekVideo(mark.timestamp_sec)));
    draft.rule_reviews.forEach((review, index) => addItem(ui.manualRules,
      `${review.rule_id} · ${applicabilityLabels[review.applicability]}`, review.note,
      () => draft.rule_reviews.splice(index, 1)));
  }

  function addManualMark() {
    const draft = activeManualDraft();
    updateManualTime();
    if (!draft || ui.manualAddMark.disabled) return;
    draft.marks.push({ timestamp_sec: Math.min(ui.video.duration, Math.round(ui.video.currentTime * 1000) / 1000), note: draft.mark_note.trim() });
    draft.marks.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
    draft.mark_note = "";
    ui.manualMarkNote.value = "";
    manualChanged();
    renderManualLists();
    announce("Ручная временная отметка добавлена. Скачайте журнал, чтобы сохранить её.");
  }

  function saveManualRule() {
    const draft = activeManualDraft();
    if (!draft || draft.hash_status !== "ready") return;
    const ruleId = draft.rule_id_input.trim();
    if (!ruleId) { showManualStatus("Укажите rule_id из проверяемого реестра. Правило не выбирается автоматически."); ui.manualRuleId.focus(); return; }
    if (!Object.hasOwn(applicabilityLabels, draft.applicability_input)) return;
    const review = { rule_id: ruleId, applicability: draft.applicability_input, note: draft.rule_note_input.trim() };
    const index = draft.rule_reviews.findIndex((item) => item.rule_id === ruleId);
    if (index < 0) draft.rule_reviews.push(review);
    else draft.rule_reviews[index] = review;
    draft.rule_id_input = "";
    draft.applicability_input = "unknown";
    draft.rule_note_input = "";
    ui.manualRuleId.value = "";
    ui.manualApplicability.value = "unknown";
    ui.manualRuleNote.value = "";
    manualChanged();
    renderManualLists();
    showManualStatus("Проверка применимости сохранена в журнале. Правила сервера не изменены.");
  }

  function exportManualReview() {
    const draft = activeManualDraft();
    if (!draft || draft.hash_status !== "ready" || !/^[a-f0-9]{64}$/.test(draft.sha256)) {
      showManualStatus("Экспорт невозможен без проверенного SHA-256 исходного файла."); return;
    }
    if (draft.mark_note.trim()) {
      showManualStatus("Добавьте временную отметку для её заметки или перенесите текст в общую заметку перед экспортом."); return;
    }
    if (draft.rule_id_input.trim() || draft.rule_note_input.trim() || draft.applicability_input !== "unknown") {
      showManualStatus("Сначала сохраните введённую проверку правила, чтобы она попала в JSON."); return;
    }
    updateManualTime();
    const payload = {
      schema_version: 1, kind: "cashwatch_manual_review", is_ground_truth: false,
      source: { filename: state.file.name, sha256: draft.sha256, size_bytes: state.file.size, duration_sec: draft.duration_sec },
      group_id: draft.group_id.trim(), camera_id: draft.camera_id.trim(),
      rule_reviews: draft.rule_reviews.map((review) => ({ ...review })),
      marks: draft.marks.map((mark) => ({ ...mark })), note: draft.note.trim(),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2) + "\n"], { type: "application/json;charset=utf-8" }));
    const link = element("a");
    link.href = url;
    link.download = `cashwatch_manual_review_v1-${draft.sha256.slice(0, 12)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    draft.exported_revision = draft.revision;
    showManualStatus(draft.duration_sec === null
      ? "Скачивание начато. Длительность не прочитана браузером: duration_sec = null. Сохранённые ручные отметки требуют проверки. Проверьте файл."
      : "Скачивание ручного журнала начато. Проверьте сохранённый файл; это не предсказания модели.");
  }

  const manualInputs = [
    [ui.manualGroup, "group_id"], [ui.manualCamera, "camera_id"], [ui.manualNote, "note"],
    [ui.manualMarkNote, "mark_note"], [ui.manualRuleId, "rule_id_input"],
    [ui.manualApplicability, "applicability_input"], [ui.manualRuleNote, "rule_note_input"],
  ];

  class ApiFault extends Error {
    constructor(message, { status = 0, code = "", retryAfterMs = 0 } = {}) {
      super(message);
      this.name = "ApiFault";
      this.status = status;
      this.code = code;
      this.retryAfterMs = retryAfterMs;
    }
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function announce(message) { ui.announcer.textContent = message; }
  function finiteNumber(value) { return typeof value === "number" && Number.isFinite(value) && value >= 0; }
  function safeText(value, fallback = "") { return typeof value === "string" ? value : fallback; }

  function formatClock(seconds) {
    if (!finiteNumber(seconds)) return "—";
    const total = Math.floor(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = String(total % 60).padStart(2, "0");
    return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
  }

  function formatTimecode(seconds) {
    const centiseconds = Math.round(seconds * 100);
    return `${formatClock(Math.floor(centiseconds / 100))},${String(centiseconds % 100).padStart(2, "0")}`;
  }

  function formatSeconds(seconds) {
    return `${new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(seconds)} с`;
  }

  function setConnection(message, style = "") {
    ui.connection.className = `connection-message${style ? ` ${style}` : ""}`;
    ui.connection.textContent = message;
  }

  function setResultState(message, style = "") {
    ui.resultState.className = `result-state${style ? ` ${style}` : ""}`;
    ui.resultState.textContent = message;
  }

  function readinessIssue(capabilities) {
    // Specific negative checks explain an aggregate model_configured=false.
    // Missing optional fields stay unknown for older servers; only false blocks here.
    if (capabilities.weights_present === false) return {
      title: "Веса модели не найдены",
      message: "На сервере указана модель, но её локальные веса пока недоступны. Завершите настройку и проверьте подключение снова.",
    };
    if (capabilities.ml_packages_present === false || capabilities.decoders_present === false) return {
      title: "Среда анализа не готова",
      message: "На сервере не хватает компонентов для модели или чтения видео. Завершите настройку сервера и проверьте подключение снова.",
    };
    if (!capabilities.model_configured) return {
      title: "Локальная модель не настроена",
      message: "Доступ к серверу подтверждён. Настройте локальную модель и повторите подключение.",
    };
    if (typeof capabilities.active_rules === "number" && capabilities.active_rules < 1) return {
      title: "Правила требуют настройки",
      message: "На сервере нет активных правил. Проверьте их по исходным материалам и включите перед анализом видео.",
    };
    return null;
  }

  function refreshEmptyState() {
    const heading = ui.empty.querySelector("h3");
    const description = ui.empty.querySelector("p:not(.eyebrow)");
    if (state.authorized && state.blocker) {
      heading.textContent = state.blocker.title;
      description.textContent = state.blocker.message;
      ui.empty.querySelector(".empty-features").hidden = true;
      if (!state.busy && !state.result && ui.error.hidden) setResultState("Требуется настройка", "warning");
    } else {
      heading.replaceChildren(document.createTextNode("Каждое событие —"), document.createElement("br"), document.createTextNode("с опорой на видео"));
      description.textContent = "После анализа здесь появятся наблюдения модели, временные отметки и кадры для проверки.";
      ui.empty.querySelector(".empty-features").hidden = false;
      if (!state.busy && !state.result && ui.error.hidden) {
        setResultState(state.file ? (state.authorized ? "Готово к запуску" : "Ожидание доступа") : "Ожидание видео");
      }
    }
  }

  function updateAvailability() {
    const canAnalyze = Boolean(state.file && state.authorized && !state.blocker && !state.checkingAccess);
    ui.analyze.disabled = !canAnalyze || state.busy;
    ui.analyze.querySelector("span").textContent = state.busy ? "Анализ выполняется…" : "Запустить анализ";
    ui.key.disabled = state.busy || state.checkingAccess;
    ui.toggleKey.disabled = state.busy || state.checkingAccess;
    ui.connect.disabled = state.busy || state.checkingAccess;
    ui.connect.replaceChildren(document.createTextNode(state.checkingAccess ? "Проверяем…" : state.authorized ? "Проверить снова" : "Подключить"), icon("arrow"));
    ui.cancel.hidden = !state.busy;
    ui.prerequisite.hidden = state.busy;
    if (!state.authorized) ui.prerequisite.textContent = state.file ? "Подключите модель, чтобы запустить анализ." : "Подключите модель и добавьте видео.";
    else if (state.blocker) ui.prerequisite.textContent = state.blocker.title + ".";
    else if (!state.file) ui.prerequisite.textContent = "Добавьте видео для проверки.";
    else ui.prerequisite.textContent = "После анализа нажмите на кадр или временную отметку, чтобы проверить момент.";
    refreshEmptyState();
  }

  function announceSource(message) {
    ui.sourceMessage.textContent = message;
    ui.sourceMessage.hidden = !message;
  }

  function retryAfterMilliseconds(value) {
    if (!value) return 0;
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return Math.max(0, numeric * 1000);
    const date = Date.parse(value);
    return Number.isFinite(date) ? Math.max(0, date - Date.now()) : 0;
  }

  function faultFromBody(body, status, retryAfterMs = 0) {
    let detail = body && typeof body === "object" ? (body.error ?? body.detail ?? body) : null;
    if (detail && typeof detail === "object" && detail.error) detail = detail.error;
    const message = typeof detail === "string" ? detail : safeText(detail?.message);
    const code = safeText(detail?.code);
    return new ApiFault(message.slice(0, 600), { status, code, retryAfterMs });
  }

  async function requestJSON(path, { key = "", timeout = 15_000, tracked = true } = {}) {
    const controller = new AbortController();
    if (tracked) state.controllers.add(controller);
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    try {
      const headers = { Accept: "application/json" };
      if (key) headers["X-API-Key"] = key;
      const response = await fetch(path, {
        method: "GET", headers, signal: controller.signal, credentials: "same-origin",
        cache: "no-store", redirect: "error", referrerPolicy: "no-referrer",
      });
      let payload;
      try { payload = await response.json(); }
      catch { throw new ApiFault("Сервер вернул ответ, который не удалось прочитать.", { status: response.status, code: "invalid_response" }); }
      if (!response.ok) throw faultFromBody(payload, response.status, retryAfterMilliseconds(response.headers.get("Retry-After")));
      return payload;
    } catch (error) {
      if (timedOut) throw new ApiFault("Сервер не ответил вовремя.", { code: "request_timeout" });
      if (error.name === "AbortError" || error instanceof ApiFault) throw error;
      throw new ApiFault("Не удалось связаться с сервером.", { code: "network_error" });
    } finally {
      clearTimeout(timer);
      state.controllers.delete(controller);
    }
  }

  async function checkHealth() {
    const check = ++state.healthCheck;
    ui.server.className = "server-status checking";
    ui.serverText.textContent = "Проверяем сервер";
    try {
      await requestJSON("/healthz", { timeout: 6000, tracked: false });
      if (check !== state.healthCheck) return;
      ui.server.className = "server-status available";
      ui.serverText.textContent = "Сервер доступен";
    } catch {
      if (check !== state.healthCheck) return;
      ui.server.className = "server-status unavailable";
      ui.serverText.textContent = "Нет связи с сервером";
    }
  }

  async function checkAccess(event) {
    if (event) event.preventDefault();
    if (state.busy || state.checkingAccess) return;
    const key = ui.key.value.trim();
    if (!key) {
      setConnection("Введите ключ доступа к локальному серверу.", "error");
      ui.key.focus();
      return;
    }
    const check = ++state.accessCheck;
    state.apiKey = key;
    state.checkingAccess = true;
    state.authorized = false;
    state.blocker = null;
    setConnection("Проверяем доступ и готовность модели…");
    updateAvailability();
    try {
      const capabilities = await requestJSON("/api/v1/capabilities", { key, timeout: 15_000, tracked: false });
      if (check !== state.accessCheck) return;
      if (!capabilities || typeof capabilities !== "object" || typeof capabilities.model_configured !== "boolean") {
        throw new ApiFault("Сервер не сообщил готовность модели.", { code: "invalid_response" });
      }
      state.capabilities = capabilities;
      state.authorized = true;
      state.blocker = readinessIssue(capabilities);
      const extensions = Array.isArray(capabilities.accepted_extensions)
        ? capabilities.accepted_extensions.map((value) => safeText(value).toLowerCase()).map((value) => value.startsWith(".") ? value : `.${value}`).filter((value) => [".mov", ".mp4"].includes(value))
        : [];
      state.acceptedExtensions = extensions.length ? [...new Set(extensions)] : [".mov"];
      state.maxUploadMiB = finiteNumber(capabilities.max_upload_mb) && capabilities.max_upload_mb > 0
        ? Math.min(MAX_UPLOAD_MIB, capabilities.max_upload_mb) : MAX_UPLOAD_MIB;
      refreshFileRequirements();
      if (state.file) {
        const issue = fileIssue(state.file);
        if (issue) {
          clearSourceFile();
          resetResult();
          announceSource(issue);
        }
      }
      if (state.blocker) setConnection(state.blocker.title + ". " + state.blocker.message, "warning");
      else if (capabilities.quality_validated_on_dataset === false) setConnection("Модель доступна для запуска. Качество на наборе данных пока не подтверждено.", "warning");
      else if (capabilities.ready_for_submission === false) setConnection("Модель доступна для запуска. Готовность к сдаче пока не подтверждена.", "warning");
      else setConnection("Доступ подтверждён. Модель доступна для запуска.", "success");
      ui.server.className = "server-status available";
      ui.serverText.textContent = "Сервер доступен";
      if (!ui.error.hidden && ui.retry.dataset.action === "connect") resetResult();
      announce(state.blocker ? state.blocker.title : "Доступ подтверждён. Можно добавить видео.");
    } catch (error) {
      if (check !== state.accessCheck) return;
      state.authorized = false;
      state.capabilities = null;
      const description = describeFault(error);
      setConnection(`${description.title}. ${description.message}`, "error");
    } finally {
      if (check === state.accessCheck) {
        state.checkingAccess = false;
        updateAvailability();
      }
    }
  }

  function refreshFileRequirements() {
    ui.input.accept = state.acceptedExtensions.join(",");
    const formats = state.acceptedExtensions.join(", ");
    ui.uploadHelp.textContent = `Формат ${formats} · один файл до ${state.maxUploadMiB} МиБ`;
    ui.formatChip.textContent = `${state.acceptedExtensions.map((item) => item.slice(1).toUpperCase()).join(" / ")} · до ${state.maxUploadMiB} МиБ`;
  }

  function fileIssue(file) {
    if (!state.acceptedExtensions.some((extension) => file.name.toLowerCase().endsWith(extension))) {
      return `Выберите видео в формате ${state.acceptedExtensions.join(" или ")}. Остальные форматы сервер не объявил поддерживаемыми.`;
    }
    if (file.size === 0) return "Выбранный файл пуст. Добавьте исходное видео.";
    if (file.size > state.maxUploadMiB * 1024 * 1024) return `Файл превышает ${state.maxUploadMiB} МиБ. Выберите файл меньшего размера.`;
    return "";
  }

  function clearSourceFile() {
    manual.generation += 1;
    manual.active = null;
    ui.manualFields.disabled = true;
    ui.manualExport.disabled = true;
    ui.video.pause();
    ui.video.removeAttribute("src");
    ui.video.load();
    if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
    state.videoUrl = null;
    state.file = null;
    ui.input.value = "";
    ui.filePanel.hidden = true;
    ui.drop.hidden = false;
    ui.mediaWarning.hidden = true;
    if (ui.dialog.open) ui.dialog.close();
  }

  function updateFileInfo() {
    if (!state.file) return;
    const formatter = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
    const size = state.file.size < 1024 ? `${state.file.size} Б`
      : state.file.size < 1024 * 1024 ? `${formatter.format(state.file.size / 1024)} КиБ`
      : `${formatter.format(state.file.size / 1024 / 1024)} МиБ`;
    const duration = !ui.video.error && ui.video.readyState > 0 && finiteNumber(ui.video.duration) && ui.video.duration > 0
      ? ` · ${formatClock(ui.video.duration)}` : "";
    ui.fileInfo.textContent = `${size}${duration}`;
  }

  function chooseFiles(files) {
    if (!files || files.length === 0) return;
    const wasBusy = state.busy;
    const oldPhase = state.phase;
    stopOperation();
    clearSourceFile();
    resetResult();
    if (files.length !== 1) {
      announceSource("Добавьте одно видео за раз. Выбрано файлов: " + files.length + ".");
      updateAvailability();
      return;
    }
    const file = files[0];
    const issue = fileIssue(file);
    if (issue) {
      announceSource(issue);
      updateAvailability();
      return;
    }
    state.file = file;
    state.videoUrl = URL.createObjectURL(file);
    ui.video.src = state.videoUrl;
    void attachManualFile(file);
    ui.fileName.textContent = file.name;
    ui.fileName.title = file.name;
    ui.filePanel.hidden = false;
    ui.drop.hidden = true;
    updateFileInfo();
    const interrupted = oldPhase === "uploading"
      ? "Предыдущая отправка прервана. Выбран новый файл."
      : "Ожидание предыдущего результата остановлено. Задание может продолжать выполняться на сервере.";
    announceSource(wasBusy ? interrupted : "");
    announce(`Выбрано видео: ${file.name}.`);
    updateAvailability();
  }

  function resetResult() {
    state.result = null;
    state.details = null;
    state.jobId = null;
    state.eventCards = [];
    state.warnings.clear();
    ui.completed.hidden = true;
    ui.busy.hidden = true;
    ui.error.hidden = true;
    ui.empty.hidden = false;
    ui.events.replaceChildren();
    ui.warnings.replaceChildren();
    ui.warnings.hidden = true;
    ui.details.hidden = true;
    ui.details.open = false;
    ui.sourceDetails.hidden = true;
    ui.sourceDetails.open = false;
    ui.detailsList.replaceChildren();
    ui.sources.replaceChildren();
    ui.resultRegion.setAttribute("aria-busy", "false");
    refreshEmptyState();
  }

  function clearTimers() {
    if (state.clock !== null) clearInterval(state.clock);
    state.clock = null;
    if (state.pollTimer !== null) clearTimeout(state.pollTimer);
    state.pollTimer = null;
    if (state.resolvePollWait) state.resolvePollWait(false);
    state.resolvePollWait = null;
  }

  function stopOperation() {
    ++state.operation;
    clearTimers();
    if (state.xhr) state.xhr.abort();
    state.xhr = null;
    for (const controller of state.controllers) controller.abort();
    state.controllers.clear();
    state.busy = false;
    state.phase = "idle";
    ui.progressPanel.hidden = true;
    ui.cancel.hidden = true;
    ui.resultRegion.setAttribute("aria-busy", "false");
  }

  function finishOperation() {
    clearTimers();
    state.busy = false;
    state.phase = "idle";
    state.xhr = null;
    ui.progressPanel.hidden = true;
    ui.resultRegion.setAttribute("aria-busy", "false");
    updateAvailability();
  }

  function startClock() {
    state.startedAt = performance.now();
    ui.elapsed.textContent = "0:00";
    state.clock = setInterval(() => {
      ui.elapsed.textContent = formatClock((performance.now() - state.startedAt) / 1000);
    }, 1000);
  }

  function showBusy(phase) {
    state.phase = phase;
    ui.empty.hidden = true;
    ui.error.hidden = true;
    ui.completed.hidden = true;
    ui.busy.hidden = false;
    ui.progressPanel.hidden = false;
    ui.resultRegion.setAttribute("aria-busy", "true");
    if (phase === "uploading") {
      setResultState("Отправка файла", "processing");
      ui.busyTitle.textContent = "Отправляем видео";
      ui.busyDescription.textContent = "Прогресс отправки показан рядом. Исходное видео доступно для просмотра.";
      ui.progressLabel.textContent = "Отправляем видео";
      ui.progressDescription.textContent = "Готовим файл к отправке.";
      ui.progress.setAttribute("aria-label", "Отправка видео");
      ui.progress.value = 0;
      ui.progressPercent.textContent = "0 %";
      ui.cancel.textContent = "Прервать отправку";
      ui.busyJob.hidden = true;
    } else {
      setResultState("Анализ видео", "processing");
      ui.busyTitle.textContent = "Модель анализирует запись";
      ui.busyDescription.textContent = "Статус обновляется автоматически. Пока идёт анализ, можно просмотреть исходное видео.";
      ui.progressLabel.textContent = "Ожидаем результат модели";
      ui.progressDescription.textContent = "Проверяем статус задания. Ожидание — до 5 минут.";
      ui.progress.setAttribute("aria-label", "Ожидание результата анализа");
      ui.progress.removeAttribute("value");
      ui.progressPercent.textContent = "";
      ui.cancel.textContent = "Остановить ожидание";
      ui.busyJob.textContent = `Задание ${state.jobId}`;
      ui.busyJob.hidden = false;
    }
  }

  function uploadVideo(file, key, operation) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      state.xhr = xhr;
      xhr.open("POST", "/api/v1/analyze", true);
      xhr.setRequestHeader("X-API-Key", key);
      xhr.setRequestHeader("Accept", "application/json");
      xhr.responseType = "json";
      xhr.timeout = 180_000;
      xhr.upload.onprogress = (event) => {
        if (operation !== state.operation) return;
        if (event.lengthComputable) {
          const percent = Math.min(100, Math.floor(event.loaded / event.total * 100));
          ui.progress.value = percent;
          ui.progressPercent.textContent = `${percent} %`;
          ui.progressDescription.textContent = percent === 100 ? "Видео отправлено. Сервер создаёт задание…" : "Передаём исходный файл на сервер.";
        } else {
          ui.progress.removeAttribute("value");
          ui.progressPercent.textContent = "";
          ui.progressDescription.textContent = "Передаём исходный файл на сервер.";
        }
      };
      xhr.onload = () => {
        if (operation !== state.operation) return reject(new DOMException("Запрос отменён", "AbortError"));
        if (xhr.status >= 200 && xhr.status < 300) {
          if (!xhr.response || typeof xhr.response !== "object") reject(new ApiFault("Сервер не вернул сведения о задании.", { code: "invalid_response" }));
          else resolve(xhr.response);
        } else reject(faultFromBody(xhr.response, xhr.status, retryAfterMilliseconds(xhr.getResponseHeader("Retry-After"))));
      };
      xhr.onerror = () => reject(new ApiFault("Не удалось отправить файл на сервер.", { code: "network_error" }));
      xhr.ontimeout = () => reject(new ApiFault("Отправка видео не завершилась за 3 минуты. Проверьте соединение и повторите попытку.", { code: "upload_timeout" }));
      xhr.onabort = () => reject(new DOMException("Запрос отменён", "AbortError"));
      const data = new FormData();
      data.append("file", file, file.name);
      xhr.send(data);
    });
  }

  function waitBeforePoll(milliseconds, operation) {
    return new Promise((resolve) => {
      if (operation !== state.operation) return resolve(false);
      state.resolvePollWait = resolve;
      state.pollTimer = setTimeout(() => {
        state.pollTimer = null;
        state.resolvePollWait = null;
        resolve(operation === state.operation);
      }, milliseconds);
    });
  }

  function validJobId(value) { return typeof value === "string" && /^[A-Za-z0-9_-]{1,200}$/.test(value); }

  async function startAnalysis() {
    if (state.busy || state.checkingAccess || !state.file || !state.authorized || state.blocker) return;
    const issue = fileIssue(state.file);
    if (issue) { announceSource(issue); return; }
    const file = state.file;
    const key = state.apiKey;
    stopOperation();
    resetResult();
    const operation = state.operation;
    state.busy = true;
    announceSource("");
    showBusy("uploading");
    startClock();
    updateAvailability();
    announce("Началась отправка видео.");
    try {
      const job = await uploadVideo(file, key, operation);
      if (operation !== state.operation) return;
      state.xhr = null;
      if (!validJobId(job.job_id) || job.status !== "processing") {
        throw new ApiFault("Сервер вернул некорректные сведения о созданном задании.", { code: "invalid_response" });
      }
      state.jobId = job.job_id;
      showBusy("processing");
      announce("Видео отправлено. Модель анализирует запись.");
      const result = await pollJob(job.job_id, key, operation);
      if (!result || operation !== state.operation) return;
      renderResult(result);
      finishOperation();
      void loadDetails(job.job_id, key, operation);
    } catch (error) {
      if (operation !== state.operation || error.name === "AbortError") return;
      showError(error);
      finishOperation();
    }
  }

  async function pollJob(jobId, key, operation) {
    const deadline = performance.now() + POLL_DEADLINE_MS;
    let delay = 1000;
    let connectionFailures = 0;
    while (operation === state.operation) {
      const remaining = deadline - performance.now();
      if (remaining <= 0) throw new ApiFault("", { code: "poll_timeout" });
      if (!await waitBeforePoll(Math.min(delay, remaining), operation)) return null;
      if (performance.now() >= deadline) throw new ApiFault("", { code: "poll_timeout" });
      try {
        const payload = await requestJSON(`/api/v1/jobs/${encodeURIComponent(jobId)}`, {
          key, timeout: Math.min(15_000, Math.max(1, deadline - performance.now())),
        });
        if (operation !== state.operation) return null;
        if (performance.now() >= deadline) throw new ApiFault("", { code: "poll_timeout" });
        connectionFailures = 0;
        if (!payload || payload.job_id !== jobId) throw new ApiFault("Ответ сервера не соответствует текущему заданию.", { code: "invalid_response" });
        if (payload.status === "done") return validateResult(payload);
        if (payload.status === "failed") throw faultFromBody(payload, 500);
        if (payload.status !== "processing") throw new ApiFault("Сервер вернул неизвестное состояние задания.", { code: "invalid_response" });
        delay = 2000;
        ui.progressDescription.textContent = "Модель продолжает анализ. Проверяем статус задания.";
      } catch (error) {
        if (operation !== state.operation || error.name === "AbortError") return null;
        const notConfigured = /model.*not.*configured|rules.*not.*configured|no.*active.*rules/i.test(error.code)
          || /локальная модель не настроена|ни одно проверенное правило|нет активных правил/i.test(safeText(error.message));
        if ((error.status === 429 || error.status === 503) && !notConfigured) {
          delay = Math.max(2000, error.retryAfterMs || 5000);
          ui.progressDescription.textContent = "Сервер временно занят. Статус будет запрошен повторно.";
          continue;
        }
        if (["network_error", "request_timeout"].includes(error.code) && ++connectionFailures <= 3) {
          delay = Math.min(8000, 2000 * connectionFailures);
          ui.progressDescription.textContent = "Связь с сервером прервалась. Повторяем запрос статуса.";
          continue;
        }
        throw error;
      }
    }
    return null;
  }

  function validateResult(payload) {
    if (!["violation", "no_violation"].includes(payload.verdict) || !Array.isArray(payload.violations)) {
      throw new ApiFault("Результат не содержит корректного заключения и списка событий.", { code: "invalid_response" });
    }
    if ((payload.verdict === "violation" && payload.violations.length === 0) || (payload.verdict === "no_violation" && payload.violations.length > 0)) {
      throw new ApiFault("Заключение и список событий противоречат друг другу. Проверьте ответ сервера.", { code: "invalid_response" });
    }
    for (const item of payload.violations) {
      if (!item || typeof item.type !== "string" || !item.type.trim() || typeof item.description !== "string" || !finiteNumber(item.timecode_sec) || !Array.isArray(item.evidence_frames)) {
        throw new ApiFault("В результате есть событие без обязательных сведений. Не удалось показать достоверный разбор.", { code: "invalid_response" });
      }
    }
    return payload;
  }

  function jpegSource(value) {
    if (typeof value !== "string") return null;
    const raw = value.startsWith("data:image/jpeg;base64,") ? value.slice(23) : value;
    if (raw.length < 8 || raw.length > 16 * 1024 * 1024 || raw.length % 4 !== 0) return null;
    if (!raw.startsWith("/9j/") || !/^[A-Za-z0-9+/]+={0,2}$/.test(raw)) return null;
    return `data:image/jpeg;base64,${raw}`;
  }

  function addWarning(message) {
    if (!message) return;
    state.warnings.add(message);
    ui.warnings.replaceChildren(...[...state.warnings].map((text) => element("p", "", text)));
    ui.warnings.hidden = state.warnings.size === 0;
  }

  function seekVideo(timestamp) {
    if (!state.file) return;
    if (ui.video.error || ui.video.readyState === 0 || !finiteNumber(ui.video.duration) || ui.video.duration <= 0) {
      announceSource("Браузер пока не может перейти к моменту в этом видео. Дождитесь загрузки записи или откройте исходный файл в проигрывателе.");
      return;
    }
    if (timestamp > ui.video.duration + 0.25) {
      announceSource(`Отметка ${formatTimecode(timestamp)} выходит за длительность видео, указанную браузером. Проверьте исходный файл и ответ сервера.`);
      return;
    }
    try {
      ui.video.currentTime = Math.min(timestamp, Math.max(0, ui.video.duration - 0.01));
      ui.video.focus({ preventScroll: true });
      ui.video.scrollIntoView({ block: "nearest", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      announce(`Видео установлено на отметку ${formatTimecode(timestamp)}, ${formatSeconds(timestamp)} от начала.`);
    } catch {
      announceSource("Браузер не смог перейти к этому моменту. Проверьте исходную запись в проигрывателе.");
    }
  }

  function showFrame(source, timestamp) {
    state.modalTimestamp = timestamp;
    ui.dialogImage.src = source;
    ui.dialogImage.alt = `Кадр исходного видео на отметке ${formatTimecode(timestamp)} (${formatSeconds(timestamp)} от начала)`;
    ui.dialogTime.textContent = `${formatTimecode(timestamp)} · ${formatSeconds(timestamp)} от начала`;
    if (!ui.dialog.open) ui.dialog.showModal();
  }

  function renderEvidence(frame, eventIndex) {
    if (!frame || !finiteNumber(frame.timestamp_sec)) return null;
    const source = jpegSource(frame.image_base64);
    if (!source) return null;
    const operation = state.operation;
    const jobId = state.result?.job_id;
    const timestamp = frame.timestamp_sec;
    const card = element("figure", "evidence-card");
    const thumbnail = element("button", "evidence-thumb");
    thumbnail.type = "button";
    thumbnail.setAttribute("aria-label", `Событие ${eventIndex + 1}: перейти к ${formatTimecode(timestamp)} в исходном видео`);
    thumbnail.title = `Перейти к ${formatTimecode(timestamp)} в видео`;
    thumbnail.addEventListener("click", () => seekVideo(timestamp));
    const picture = element("img");
    picture.src = source;
    picture.alt = `Исходный кадр · ${formatTimecode(timestamp)}`;
    picture.loading = "lazy";
    picture.decoding = "async";
    picture.draggable = false;
    picture.addEventListener("error", () => {
      if (operation !== state.operation || state.result?.job_id !== jobId) return;
      const message = element("p", "evidence-missing", "Кадр не удалось открыть.");
      card.replaceChildren(message);
      addWarning(`Не удалось прочитать один из кадров события ${eventIndex + 1}. Проверьте исходное видео.`);
    }, { once: true });
    thumbnail.append(picture);
    const footer = element("figcaption", "evidence-footer");
    const timeButton = element("button", "frame-time-button", formatTimecode(timestamp));
    timeButton.type = "button";
    timeButton.title = `${formatSeconds(timestamp)} от начала исходного видео`;
    timeButton.setAttribute("aria-label", `Перейти к ${formatTimecode(timestamp)} в видео`);
    timeButton.addEventListener("click", () => seekVideo(timestamp));
    const expand = element("button", "icon-button");
    expand.type = "button";
    expand.title = "Увеличить кадр";
    expand.setAttribute("aria-label", `Увеличить кадр на отметке ${formatTimecode(timestamp)}`);
    expand.append(icon("expand"));
    expand.addEventListener("click", () => showFrame(source, timestamp));
    footer.append(timeButton, expand);
    card.append(thumbnail, footer);
    return card;
  }

  function renderEvent(item, index) {
    const card = element("article", "event-card");
    const header = element("div", "event-header");
    const heading = element("div", "event-heading");
    heading.append(element("span", "event-index", `СОБЫТИЕ ${String(index + 1).padStart(2, "0")}`));
    heading.append(element("h4", "", item.type));
    const timestampButton = element("button", "timecode-button");
    timestampButton.type = "button";
    timestampButton.append(icon("play"), document.createTextNode(formatTimecode(item.timecode_sec)));
    timestampButton.setAttribute("aria-label", `Перейти к событию ${index + 1}, ${formatSeconds(item.timecode_sec)} от начала видео`);
    timestampButton.title = `${formatSeconds(item.timecode_sec)} от начала исходного видео`;
    timestampButton.addEventListener("click", () => seekVideo(item.timecode_sec));
    header.append(heading, timestampButton);
    card.append(header, element("p", "event-description", item.description));
    const frames = item.evidence_frames.slice(0, 3).map((frame) => renderEvidence(frame, index)).filter(Boolean);
    if (frames.length) {
      card.append(element("p", "evidence-heading", "КАДРЫ ИЗ ИСХОДНОЙ ЗАПИСИ · НАЖМИТЕ ДЛЯ ПЕРЕХОДА"));
      const grid = element("div", `evidence-grid count-${frames.length}`);
      grid.append(...frames);
      card.append(grid);
    }
    if (!frames.length || frames.length < Math.min(item.evidence_frames.length, 3)) {
      card.append(element("p", "evidence-missing", "Часть подтверждающих кадров отсутствует или не читается. Проверьте событие в исходном видео."));
      addWarning(`Для события ${index + 1} доступны не все кадры. Это требует дополнительной проверки записи.`);
    }
    const sourceSlot = element("div", "event-source");
    sourceSlot.hidden = true;
    card.append(sourceSlot);
    state.eventCards.push({ card, sourceSlot });
    return card;
  }

  function renderResult(result) {
    state.result = result;
    state.details = null;
    state.warnings.clear();
    state.eventCards = [];
    ui.empty.hidden = true;
    ui.busy.hidden = true;
    ui.error.hidden = true;
    ui.completed.hidden = false;
    ui.resultJob.textContent = `Задание ${result.job_id.slice(0, 12)}`;
    ui.resultJob.title = result.job_id;
    ui.verdict.className = result.verdict === "no_violation" ? "verdict-card clear" : "verdict-card";
    ui.verdictIcon.replaceChildren(icon(result.verdict === "no_violation" ? "check" : "alert"));
    ui.verdictTitle.textContent = result.verdict === "no_violation" ? "Нарушение не обнаружено" : "Обнаружены признаки возможных нарушений";
    ui.verdictDescription.textContent = result.verdict === "no_violation"
      ? "Модель не указала нарушений. Сопоставьте результат с исходной записью."
      : `Событий для проверки: ${result.violations.length}. Откройте кадры и проверьте контекст в видео.`;
    ui.noEvents.hidden = result.violations.length > 0;
    ui.events.replaceChildren(...result.violations.map((item, index) => renderEvent(item, index)));
    if (state.capabilities?.quality_validated_on_dataset === false) addWarning("Качество модели на наборе данных пока не подтверждено.");
    setResultState("Анализ завершён", result.verdict === "violation" ? "warning" : "done");
    announce(ui.verdictTitle.textContent + (result.violations.length ? `. Событий для проверки: ${result.violations.length}.` : "."));
  }

  function allowedSourceURL(value) {
    if (typeof value !== "string") return null;
    try {
      const url = new URL(value);
      if (url.origin !== "https://adilet.zan.kz" || url.username || url.password) return null;
      return url.href;
    } catch { return null; }
  }

  function sourceElement(event) {
    const source = event?.source;
    if (!source || typeof source !== "object") return null;
    const title = safeText(source.title).trim();
    const clause = safeText(source.clause).trim();
    if (!title && !clause) return null;
    const node = element("div", "source-item");
    const eventTitle = safeText(event.title).trim();
    if (eventTitle) node.append(element("p", "source-item-title", eventTitle));
    const url = allowedSourceURL(source.url);
    if (url && title) {
      const link = element("a", "source-link", title);
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.append(icon("external"));
      node.append(link);
    } else if (title) node.append(element("p", "", title));
    if (clause) node.append(element("p", "", clause));
    return node;
  }

  function addDetail(label, value) {
    ui.detailsList.append(element("dt", "", label), element("dd", "", value));
  }

  async function loadDetails(jobId, key, operation) {
    try {
      const details = await requestJSON(`/api/v1/jobs/${encodeURIComponent(jobId)}/details`, { key, timeout: 10_000 });
      if (operation !== state.operation || state.result?.job_id !== jobId || !details || typeof details !== "object") return;
      state.details = details;
      ui.detailsList.replaceChildren();
      if (safeText(details.model_id)) addDetail("Модель", details.model_id);
      if (finiteNumber(details.duration_sec)) addDetail("Длительность видео", `${formatClock(details.duration_sec)} (${formatSeconds(details.duration_sec)})`);
      if (finiteNumber(details.elapsed_sec)) addDetail("Время анализа", formatSeconds(details.elapsed_sec));
      if (typeof details.complete === "boolean") addDetail("Обработка записи", details.complete ? "Завершена по данным сервера" : "Выполнена не полностью");
      ui.details.hidden = ui.detailsList.childElementCount === 0;
      if (details.complete === false) {
        addWarning("Проверка выполнена не полностью. Отсутствие обнаруженных событий не означает, что нарушений не было.");
        setResultState("Неполная проверка", "warning");
        if (state.result.verdict === "no_violation") {
          ui.verdict.className = "verdict-card";
          ui.verdictIcon.replaceChildren(icon("alert"));
          ui.verdictTitle.textContent = "Проверка выполнена не полностью";
          ui.verdictDescription.textContent = "В обработанной части модель не указала нарушений. Требуется просмотр исходной записи.";
          announce("Проверка выполнена не полностью. Требуется просмотр исходной записи.");
        }
      }
      if (Array.isArray(details.warnings)) {
        for (const warning of details.warnings) {
          const message = typeof warning === "string" ? warning : safeText(warning?.message);
          if (message) addWarning(message);
        }
      }
      if (Array.isArray(details.events)) {
        const sourceNodes = [];
        const seen = new Set();
        details.events.forEach((event, index) => {
          const inline = sourceElement(event);
          if (inline && state.eventCards[index]) {
            state.eventCards[index].sourceSlot.replaceChildren(element("p", "evidence-heading", "ОСНОВАНИЕ ПРОВЕРКИ"), inline);
            state.eventCards[index].sourceSlot.hidden = false;
          }
          const fingerprint = JSON.stringify([event?.title, event?.source?.title, event?.source?.url, event?.source?.clause]);
          if (!seen.has(fingerprint)) {
            seen.add(fingerprint);
            const node = sourceElement(event);
            if (node) sourceNodes.push(node);
          }
        });
        ui.sources.replaceChildren(...sourceNodes);
        ui.sourceDetails.hidden = sourceNodes.length === 0;
      }
    } catch {
      // Supplemental metadata never hides or invalidates a successfully received core response.
      // Completeness is never assumed when this optional endpoint is unavailable.
    }
  }

  function describeFault(error) {
    const code = safeText(error.code).toLowerCase();
    const message = safeText(error.message);
    if (/model.*not.*configured/.test(code) || /локальная модель не настроена/i.test(message)) return { title: "Локальная модель не настроена", message: "Настройте модель на сервере и проверьте подключение снова.", action: "connect" };
    if (/rules.*not.*configured|no.*active.*rules/.test(code) || /ни одно проверенное правило|нет активных правил/i.test(message)) return { title: "Правила требуют настройки", message: "На сервере нет активных правил для анализа. Проверьте настройки и подключитесь снова.", action: "connect" };
    if (error.status === 401) return { title: "Не удалось подтвердить доступ", message: "Ключ доступа отсутствует или недействителен. Введите корректный ключ и подключитесь снова.", action: "connect" };
    if (error.status === 413) return { title: "Файл слишком большой", message: `Выберите видео размером до ${state.maxUploadMiB} МиБ.`, action: "file" };
    if (error.status === 415) return { title: "Формат видео не поддерживается", message: "Сервер не смог принять формат или кодек записи. " + (message || `Используйте ${state.acceptedExtensions.join(" или ")}.`), action: "file" };
    if (error.status === 429) return { title: "Сервер занят другими запросами", message: "Достигнут предел одновременных заданий. Подождите и повторите анализ." };
    if (error.status === 503) return { title: "Анализ временно недоступен", message: message || "Сервер сейчас не готов к анализу. Проверьте модель и повторите подключение.", action: "connect" };
    if (code === "poll_timeout") return { title: "Время ожидания истекло", message: "За 5 минут результат не получен. Задание может продолжать выполняться на сервере. Проверьте его состояние перед повторным запуском." };
    if (code === "upload_timeout") return { title: "Отправка видео не завершена", message };
    if (code === "network_error") return { title: "Нет соединения с сервером", message: "Проверьте подключение и доступность сервера. " + (state.jobId ? "Созданное задание может продолжать выполняться на сервере." : "Повторите попытку после восстановления связи.") };
    if (code === "request_timeout") return { title: "Сервер не ответил вовремя", message: "Проверьте доступность сервера и повторите попытку." };
    if (code === "invalid_response") return { title: "Ответ анализа неполон", message: message || "Сервер вернул некорректный ответ. Достоверный результат не получен." };
    if (error.status === 404) return { title: "Задание не найдено", message: "Сервер не нашёл это задание. Возможно, оно уже удалено. Запустите анализ заново." };
    if (error.status === 422) return { title: "Не удалось принять видео", message: message || "Проверьте, что файл содержит поддерживаемую видеозапись.", action: "file" };
    return { title: "Анализ не завершён", message: message || "Сервер не смог завершить анализ. Проверьте его состояние и повторите попытку." };
  }

  function showError(error) {
    const failure = describeFault(error);
    ui.empty.hidden = true;
    ui.busy.hidden = true;
    ui.completed.hidden = true;
    ui.error.hidden = false;
    ui.errorTitle.textContent = failure.title;
    ui.errorDescription.textContent = failure.message;
    ui.errorJob.hidden = !state.jobId;
    ui.errorJob.textContent = state.jobId ? `Задание ${state.jobId}` : "";
    ui.retry.dataset.action = failure.action || "analyze";
    ui.retry.textContent = failure.action === "connect" ? "Проверить подключение" : failure.action === "file" ? "Выбрать другое видео" : "Повторить анализ";
    if (failure.action === "connect") {
      state.authorized = false;
      setConnection(failure.title + ". " + failure.message, "warning");
    }
    setResultState("Нет результата", "failed");
    announce(failure.title + ". " + failure.message);
  }

  function downloadResult() {
    if (!state.result) return;
    const blob = new Blob([JSON.stringify(state.result, null, 2) + "\n"], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = element("a");
    anchor.href = url;
    anchor.download = `cashwatch-${state.result.job_id}.json`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  ui.accessForm.addEventListener("submit", checkAccess);
  ui.key.addEventListener("input", () => {
    state.apiKey = "";
    state.authorized = false;
    state.capabilities = null;
    state.blocker = null;
    setConnection("Подключитесь с новым ключом, чтобы проверить доступ.");
    updateAvailability();
  });
  ui.toggleKey.addEventListener("click", () => {
    const reveal = ui.key.type === "password";
    ui.key.type = reveal ? "text" : "password";
    ui.toggleKey.setAttribute("aria-pressed", String(reveal));
    ui.toggleKey.setAttribute("aria-label", reveal ? "Скрыть ключ доступа" : "Показать ключ доступа");
  });
  ui.server.addEventListener("click", checkHealth);
  ui.drop.addEventListener("click", () => ui.input.click());
  ui.changeFile.addEventListener("click", () => ui.input.click());
  ui.input.addEventListener("change", () => chooseFiles(Array.from(ui.input.files || [])));
  let dragDepth = 0;
  ui.drop.addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth += 1; ui.drop.classList.add("drag-active"); });
  ui.drop.addEventListener("dragover", (event) => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"; });
  ui.drop.addEventListener("dragleave", (event) => { event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) ui.drop.classList.remove("drag-active"); });
  ui.drop.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    ui.drop.classList.remove("drag-active");
    chooseFiles(Array.from(event.dataTransfer?.files || []));
  });
  // Prevent an accidental drop elsewhere from navigating away with a sensitive recording.
  window.addEventListener("dragover", (event) => { if (Array.from(event.dataTransfer?.types || []).includes("Files")) event.preventDefault(); });
  window.addEventListener("drop", (event) => { if (Array.from(event.dataTransfer?.types || []).includes("Files")) event.preventDefault(); });
  ui.video.addEventListener("loadedmetadata", () => { ui.mediaWarning.hidden = true; updateFileInfo(); updateManualTime(); });
  ui.video.addEventListener("durationchange", () => { updateFileInfo(); updateManualTime(); });
  ui.video.addEventListener("timeupdate", updateManualTime);
  ui.video.addEventListener("emptied", () => { updateFileInfo(); updateManualTime(); });
  ui.video.addEventListener("error", () => {
    if (!state.file) return;
    updateFileInfo();
    updateManualTime();
    ui.mediaWarning.textContent = "Браузер не воспроизводит кодек этого видео. Файл всё равно можно отправить: возможность чтения проверит сервер.";
    ui.mediaWarning.hidden = false;
  });
  ui.analyze.addEventListener("click", startAnalysis);
  ui.cancel.addEventListener("click", () => {
    const wasUploading = state.phase === "uploading";
    stopOperation();
    resetResult();
    announceSource(wasUploading
      ? "Отправка прервана. Если сервер уже принял файл, задание может продолжать выполняться."
      : "Ожидание результата остановлено. Задание может продолжать выполняться на сервере.");
    updateAvailability();
    announce(wasUploading ? "Отправка прервана." : "Ожидание результата остановлено.");
  });
  ui.retry.addEventListener("click", () => {
    if (ui.retry.dataset.action === "file") ui.input.click();
    else if (ui.retry.dataset.action === "connect" || !state.authorized || state.blocker) {
      if (ui.key.value.trim()) void checkAccess();
      else ui.key.focus();
    } else void startAnalysis();
  });
  ui.download.addEventListener("click", downloadResult);
  for (const [input, field] of manualInputs) input.addEventListener("input", () => {
    const draft = activeManualDraft();
    if (!draft || draft.hash_status !== "ready") return;
    draft[field] = input.value;
    manualChanged();
  });
  ui.manualAddMark.addEventListener("click", addManualMark);
  ui.manualSaveRule.addEventListener("click", saveManualRule);
  ui.manualExport.addEventListener("click", exportManualReview);
  window.addEventListener("beforeunload", (event) => {
    if (!Array.from(manual.byHash.values()).some((draft) => draft.revision > draft.exported_revision)) return;
    event.preventDefault();
    event.returnValue = "";
  });
  ui.closeDialog.addEventListener("click", () => ui.dialog.close());
  ui.seekFromDialog.addEventListener("click", () => { ui.dialog.close(); seekVideo(state.modalTimestamp); });
  ui.dialog.addEventListener("click", (event) => {
    if (event.target !== ui.dialog) return;
    const rect = ui.dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) ui.dialog.close();
  });
  ui.dialog.addEventListener("close", () => { ui.dialogImage.removeAttribute("src"); ui.dialogImage.alt = ""; });
  window.addEventListener("pagehide", (event) => {
    stopOperation();
    state.apiKey = "";
    ui.key.value = "";
    state.authorized = false;
    if (!event.persisted && state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  });
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    resetResult();
    setConnection("Страница восстановлена. Введите ключ и подключитесь снова.");
    updateAvailability();
    void checkHealth();
  });

  refreshFileRequirements();
  updateAvailability();
  void checkHealth();
})();
