// Single-file vanilla JS dashboard. No framework, no build step.

const MOOD_LABEL = {
  loved_it: "loved it",
  liked_it: "liked it",
  meh: "meh",
  disliked_it: "disliked it",
  hated_it: "hated it",
};

const DIMENSIONS = [
  "visual",
  "theme",
  "win_frequency",
  "win_size",
  "volatility_feel",
  "bonus_frequency",
  "bonus_quality",
  "base_pacing",
  "anticipation",
  "session_flow",
  "mechanic_depth",
  "feature_variety",
  "anticipation_honesty",
  "celebration_calibration",
  "clarity",
];

const DIM_LABEL = {
  visual: "visual",
  theme: "theme",
  win_frequency: "win frequency",
  win_size: "win size",
  volatility_feel: "volatility feel",
  bonus_frequency: "bonus frequency",
  bonus_quality: "bonus quality",
  base_pacing: "base pacing",
  anticipation: "anticipation",
  session_flow: "session flow",
  mechanic_depth: "mechanic depth",
  feature_variety: "feature variety",
  anticipation_honesty: "anticipation honesty",
  celebration_calibration: "celebration calibration",
  clarity: "clarity",
};

const OPENROUTER_KEY_STORAGE = "personaEval.openrouterApiKey";
const OPENROUTER_MODEL_STORAGE = "personaEval.openrouterModel";
const OPENROUTER_LIST_STORAGE = "personaEval.openrouterModelList";
const FRAME_STEP_STORAGE = "personaEval.frameStep";
const MAX_FRAMES_STORAGE = "personaEval.maxFrames";

// Server-side defaults populated from /api/models on load.
let OPENROUTER_MODELS_LIST = [
  "google/gemini-2.5-flash",
  "google/gemini-2.5-flash-lite",
  "anthropic/claude-haiku-4.5",
  "openai/gpt-4o-mini",
  "meta-llama/llama-3.2-11b-vision-instruct",
  "qwen/qwen2.5-vl-72b-instruct",
  "mistralai/pixtral-large-2411",
];

document.addEventListener("DOMContentLoaded", async () => {
  await loadModels();
  syncFormUI();
  document.getElementById("model-select").addEventListener("change", (e) => {
    localStorage.setItem(OPENROUTER_MODEL_STORAGE, e.target.value);
    document.getElementById("model-note").style.display = "none";
  });
  document.getElementById("api-key-input").addEventListener("input", (e) => {
    localStorage.setItem(OPENROUTER_KEY_STORAGE, e.target.value);
  });
  document.getElementById("frame-step-input").addEventListener("input", (e) => {
    localStorage.setItem(FRAME_STEP_STORAGE, e.target.value);
  });
  document.getElementById("max-frames-input").addEventListener("input", (e) => {
    localStorage.setItem(MAX_FRAMES_STORAGE, e.target.value);
  });
  document.getElementById("refresh-models").addEventListener("click", refreshModelsFromApi);
  document.getElementById("requests-show-all").addEventListener("click", () => {
    FEED.showAll = !FEED.showAll;
    applyShowAllPolicy();
  });

  loadHistory();
  initAdmin();
  document.getElementById("upload-form").addEventListener("submit", onSubmit);
  document.getElementById("detail-close").addEventListener("click", closeDetail);
  document.getElementById("detail-backdrop").addEventListener("click", closeDetail);

  const params = new URLSearchParams(location.search);
  if (params.get("id")) {
    loadEvaluation(params.get("id"));
  }
});

async function loadModels() {
  // Server defaults.
  try {
    const res = await fetch("/api/models");
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data.openrouter_models) && data.openrouter_models.length) {
        OPENROUTER_MODELS_LIST = data.openrouter_models;
      }
    }
  } catch (_) { /* ignore — fallbacks already set */ }

  // Live OpenRouter catalog — replaces the static OPENROUTER_MODELS_LIST with
  // what OpenRouter actually has right now, filtered to vision-capable. Falls
  // back silently to the static list on network / parse failure.
  try {
    const res = await fetch("/api/openrouter/models");
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data.models) && data.models.length) {
        OPENROUTER_MODELS_LIST = data.models;
      }
    }
  } catch (_) { /* ignore — static fallback already set */ }

  // Cached list (from a previous explicit refresh) overrides the live catalog.
  try {
    const cached = localStorage.getItem(OPENROUTER_LIST_STORAGE);
    if (cached) {
      const arr = JSON.parse(cached);
      if (Array.isArray(arr) && arr.length) OPENROUTER_MODELS_LIST = arr;
    }
  } catch (_) { /* ignore */ }
}

function populateModelSelect(selectEl, models, savedValue) {
  const saved = savedValue || models[0];
  selectEl.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === saved) opt.selected = true;
    selectEl.appendChild(opt);
  });
  if (!models.includes(saved)) {
    const opt = document.createElement("option");
    opt.value = saved;
    opt.textContent = saved + " (custom)";
    opt.selected = true;
    selectEl.appendChild(opt);
  }
}

async function refreshModelsFromApi() {
  const note = document.getElementById("model-note");
  const btn = document.getElementById("refresh-models");

  // Fetch the live OpenRouter model catalog (no key needed — public API).
  btn.disabled = true;
  note.textContent = "Fetching live OpenRouter vision models…";
  note.style.display = "";
  try {
    const res = await fetch("/api/openrouter/models");
    const data = await res.json();
    if (!Array.isArray(data.models) || !data.models.length) {
      note.textContent = "Couldn't fetch OpenRouter catalog — using static list. " + (data.error || "");
      return;
    }
    OPENROUTER_MODELS_LIST = data.models;
    localStorage.setItem(OPENROUTER_LIST_STORAGE, JSON.stringify(data.models));
    syncFormUI();
    note.textContent = `Loaded ${data.models.length} vision-capable models from OpenRouter (${data.source || "live"}).`;
  } catch (e) {
    note.textContent = "Failed to fetch OpenRouter catalog: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// --------------------------------------------------------------------------- //
// Admin (prod, read-only): connect with a fresh 2FA, then fetch player data.
// --------------------------------------------------------------------------- //

const ADMIN_USER_STORAGE = "vox.adminUsername";
const ADMIN_PASS_STORAGE = "vox.adminPassword";

function initAdmin() {
  // Restore + persist username/password in-browser (same as the API key).
  const userInput = document.getElementById("admin-username-input");
  const passInput = document.getElementById("admin-password-input");
  userInput.value = localStorage.getItem(ADMIN_USER_STORAGE) || "";
  passInput.value = localStorage.getItem(ADMIN_PASS_STORAGE) || "";
  userInput.addEventListener("input", (e) => localStorage.setItem(ADMIN_USER_STORAGE, e.target.value));
  passInput.addEventListener("input", (e) => localStorage.setItem(ADMIN_PASS_STORAGE, e.target.value));

  document.getElementById("admin-connect-btn").addEventListener("click", adminConnect);
  document.getElementById("admin-fetch-btn").addEventListener("click", adminFetchPlayer);
  document.getElementById("admin-2fa-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") adminConnect();
  });
  refreshAdminStatus();
}

function renderAdminStatus(st) {
  const statusEl = document.getElementById("admin-status");
  const playerRow = document.getElementById("admin-player");
  if (st && st.authenticated) {
    const mins = st.expires_in_seconds ? Math.round(st.expires_in_seconds / 60) : null;
    statusEl.textContent = `connected as ${st.username || "?"}${st.casino ? ` · ${st.casino}` : ""}` +
      (mins !== null ? ` · session ~${mins} min left` : "");
    statusEl.style.color = "var(--accent)";
    playerRow.style.display = "";
  } else {
    statusEl.textContent = "not connected — enter a fresh 2FA code";
    statusEl.style.color = "var(--muted)";
    playerRow.style.display = "none";
  }
}

async function refreshAdminStatus() {
  try {
    const res = await fetch("/api/admin/status");
    renderAdminStatus(await res.json());
  } catch (_) {
    renderAdminStatus(null);
  }
}

async function adminConnect() {
  const input = document.getElementById("admin-2fa-input");
  const btn = document.getElementById("admin-connect-btn");
  const note = document.getElementById("admin-note");
  const username = document.getElementById("admin-username-input").value.trim();
  const password = document.getElementById("admin-password-input").value;
  const code = input.value.trim();
  if (!username) { document.getElementById("admin-username-input").focus(); return; }
  if (!password) { document.getElementById("admin-password-input").focus(); return; }
  if (!code) { input.focus(); return; }
  btn.disabled = true;
  note.style.display = "none";
  document.getElementById("admin-status").textContent = "connecting…";
  try {
    const res = await fetch("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, code }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    input.value = "";
    renderAdminStatus(data);
  } catch (e) {
    note.textContent = "Login failed: " + e.message;
    note.style.display = "";
    renderAdminStatus(null);
  } finally {
    btn.disabled = false;
  }
}

async function adminFetchPlayer() {
  const nickname = document.getElementById("admin-nickname-input").value.trim();
  const note = document.getElementById("admin-note");
  const result = document.getElementById("admin-result");
  const btn = document.getElementById("admin-fetch-btn");
  if (!nickname) { document.getElementById("admin-nickname-input").focus(); return; }
  const params = new URLSearchParams({ nickname });
  const from = document.getElementById("admin-from-input").value.trim();
  const to = document.getElementById("admin-to-input").value.trim();
  if (from) params.set("date_from", from);
  if (to) params.set("date_to", to);

  btn.disabled = true;
  note.textContent = "Fetching…";
  note.style.display = "";
  result.style.display = "none";
  try {
    const res = await fetch(`/api/admin/player?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) {
      if (data.need_login) refreshAdminStatus();
      throw new Error(data.error || res.statusText);
    }
    const s = data.summary || {};
    note.textContent = `${data.matched_users?.length || 0} match · ${data.game_count || 0} rounds · ` +
      `${s.distinct_games || 0} games · bet €${s.total_bet_eur ?? "?"} · profit €${s.total_profit_eur ?? "?"}`;
    result.textContent = JSON.stringify(data, null, 2);
    result.style.display = "";
  } catch (e) {
    note.textContent = "Fetch failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function syncFormUI() {
  // Restore the saved frame-step (default 30 ≈ 1fps for 30fps source).
  document.getElementById("frame-step-input").value =
    localStorage.getItem(FRAME_STEP_STORAGE) || "30";
  // And the frame cap (default 48 — the per-call payload sweet spot).
  document.getElementById("max-frames-input").value =
    localStorage.getItem(MAX_FRAMES_STORAGE) || "48";

  document.getElementById("api-key-input").value =
    localStorage.getItem(OPENROUTER_KEY_STORAGE) || "";
  populateModelSelect(
    document.getElementById("model-select"),
    OPENROUTER_MODELS_LIST,
    localStorage.getItem(OPENROUTER_MODEL_STORAGE),
  );
}

// Live request feed state. Reset on each new run.
const FEED = {
  rows: new Map(),       // call_id (number) -> {el, attempts, model, status}
  order: [],             // call_ids in arrival order
  showAll: false,
  succeeded: 0,
  failed: 0,
  retried: 0,
  source: null,          // current EventSource
};

function resetFeed() {
  if (FEED.source) { FEED.source.close(); FEED.source = null; }
  FEED.rows.clear();
  FEED.order.length = 0;
  FEED.showAll = false;
  FEED.succeeded = 0;
  FEED.failed = 0;
  FEED.retried = 0;
  document.getElementById("requests-list").innerHTML = "";
  document.getElementById("requests-summary").textContent = "";
  document.getElementById("requests-stage").style.display = "none";
  document.getElementById("requests-stage").textContent = "";
  document.getElementById("requests-show-all").style.display = "none";
}

function showFeed() {
  document.getElementById("requests-section").style.display = "";
}

function updateSummary() {
  const total = FEED.order.length;
  const pending = total - FEED.succeeded - FEED.failed;
  const parts = [`${total} request${total === 1 ? "" : "s"}`];
  if (pending > 0) parts.push(`${pending} pending`);
  if (FEED.succeeded > 0) parts.push(`${FEED.succeeded} ok`);
  if (FEED.failed > 0) parts.push(`${FEED.failed} failed`);
  if (FEED.retried > 0) parts.push(`${FEED.retried} retried`);
  document.getElementById("requests-summary").textContent = parts.join(" · ");
}

function renderStageEvent(ev) {
  const el = document.getElementById("requests-stage");
  let text = "";
  switch (ev.stage) {
    case "extracting_frames":
      text = `Extracting frames (step=${ev.frame_step}, cap=${ev.frame_cap || "none"})…`; break;
    case "frames_extracted":
      text = `${ev.frame_count} frames extracted (${Math.round((ev.bytes || 0) / 1024)} KB).`; break;
    case "extracting_stats":
      text = "Extracting stats from the video…"; break;
    case "stats_extracted":
      text = "Stats extracted. Starting persona evaluations…"; break;
    case "personas_started":
      text = `Running ${ev.persona_count} personas × ${ev.runs} runs (${ev.workers} parallel workers)…`; break;
    case "personas_complete":
      text = "Persona evaluations complete. Finalizing…"; break;
    default:
      text = ev.stage || "";
  }
  el.textContent = text;
  el.style.display = "";
}

function appendCallRow(ev) {
  if (FEED.rows.has(ev.id)) return; // duplicate started, ignore
  const row = document.createElement("div");
  row.className = "req-row pending";
  row.dataset.id = ev.id;
  row.innerHTML = `
    <span class="req-status">…</span>
    <span class="req-label"></span>
    <span class="req-model"></span>
    <span class="req-retry"></span>
    <span class="req-duration">…</span>
    <div class="req-detail"></div>
  `;
  row.querySelector(".req-label").textContent = ev.label;
  row.querySelector(".req-model").textContent = ev.model;
  row.addEventListener("click", () => row.classList.toggle("expanded"));

  // Sticky-bottom: only auto-scroll if the user was already near the bottom
  // before this row arrived. That way, if they scrolled up to inspect an
  // earlier row, we don't yank them back down on every new event.
  const list = document.getElementById("requests-list");
  const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
  list.appendChild(row);
  FEED.rows.set(ev.id, { el: row, attempts: 1, model: ev.model, status: "pending" });
  FEED.order.push(ev.id);
  applyShowAllPolicy();
  updateSummary();
  if (nearBottom) list.scrollTop = list.scrollHeight;
}

function updateCallRow(ev) {
  const entry = FEED.rows.get(ev.id);
  if (!entry) return;
  const row = entry.el;

  if (ev.type === "retrying") {
    if (!row.classList.contains("retrying")) {
      row.classList.remove("pending", "succeeded", "failed");
      row.classList.add("retrying");
    }
    entry.attempts = ev.attempt || entry.attempts + 1;
    entry.model = ev.model;
    row.querySelector(".req-status").textContent = "↻";
    row.querySelector(".req-model").textContent = ev.model;
    row.querySelector(".req-retry").textContent = `retry ${entry.attempts}`;
    appendDetail(row, `[retry ${entry.attempts}] ${ev.error || ""}${ev.wait_ms ? ` (wait ${ev.wait_ms}ms)` : ""}`);
    FEED.retried++;
  } else if (ev.type === "succeeded") {
    row.classList.remove("pending", "retrying", "failed");
    row.classList.add("succeeded");
    entry.attempts = ev.attempts || entry.attempts;
    entry.status = "succeeded";
    row.querySelector(".req-status").textContent = "✓";
    row.querySelector(".req-model").textContent = ev.model;
    row.querySelector(".req-duration").textContent = `${ev.duration_ms} ms`;
    if (entry.attempts > 1) {
      row.querySelector(".req-retry").textContent = `(${entry.attempts - 1}× retried)`;
    } else {
      row.querySelector(".req-retry").textContent = "";
    }
    FEED.succeeded++;
  } else if (ev.type === "failed") {
    row.classList.remove("pending", "retrying", "succeeded");
    row.classList.add("failed");
    entry.attempts = ev.attempts || entry.attempts;
    entry.status = "failed";
    row.querySelector(".req-status").textContent = "✗";
    row.querySelector(".req-model").textContent = ev.model;
    row.querySelector(".req-duration").textContent = `${ev.duration_ms} ms`;
    if (entry.attempts > 1) {
      row.querySelector(".req-retry").textContent = `(${entry.attempts - 1}× retried)`;
    }
    appendDetail(row, `[failed] ${ev.error || ""}`);
  }
  updateSummary();
}

function appendDetail(row, text) {
  const d = row.querySelector(".req-detail");
  d.textContent = d.textContent ? `${d.textContent}\n${text}` : text;
}

function applyShowAllPolicy() {
  const list = document.getElementById("requests-list");
  const rows = list.querySelectorAll(".req-row");
  const btn = document.getElementById("requests-show-all");
  if (FEED.showAll || rows.length <= 10) {
    rows.forEach((r) => (r.style.display = ""));
    btn.style.display = rows.length > 10 ? "" : "none";
    btn.textContent = rows.length > 10 ? "Hide older" : "";
    return;
  }
  const hideCount = rows.length - 10;
  rows.forEach((r, i) => (r.style.display = i < hideCount ? "none" : ""));
  document.getElementById("requests-hidden-count").textContent = hideCount;
  btn.style.display = "";
  btn.innerHTML = `Show all (<span id="requests-hidden-count">${hideCount}</span>)`;
}

function subscribeProgress(streamUrl, onDone, onError) {
  const es = new EventSource(streamUrl);
  FEED.source = es;

  es.addEventListener("stage", (e) => renderStageEvent(JSON.parse(e.data)));
  es.addEventListener("started", (e) => appendCallRow(JSON.parse(e.data)));
  es.addEventListener("retrying", (e) => updateCallRow(JSON.parse(e.data)));
  es.addEventListener("succeeded", (e) => updateCallRow(JSON.parse(e.data)));
  es.addEventListener("failed", (e) => updateCallRow(JSON.parse(e.data)));
  es.addEventListener("done", (e) => {
    es.close();
    FEED.source = null;
    onDone(JSON.parse(e.data));
  });
  es.addEventListener("error", (e) => {
    // SSE delivers two flavors here: server-emitted error events (with JSON
    // payload) and transport-level errors (no data). Distinguish them.
    if (e.data) {
      es.close();
      FEED.source = null;
      onError(JSON.parse(e.data).message || "stream error");
    }
    // else: transport blip — EventSource will auto-reconnect.
  });
}

async function onSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById("run-btn");
  const status = document.getElementById("status");

  // Persist whatever the user typed.
  const key = document.getElementById("api-key-input").value.trim();
  if (key) {
    localStorage.setItem(OPENROUTER_KEY_STORAGE, key);
  }
  const selectedModel = document.getElementById("model-select").value;
  if (selectedModel) {
    localStorage.setItem(OPENROUTER_MODEL_STORAGE, selectedModel);
  }
  const frameStep = document.getElementById("frame-step-input").value;
  if (frameStep) localStorage.setItem(FRAME_STEP_STORAGE, frameStep);
  const maxFrames = document.getElementById("max-frames-input").value;
  if (maxFrames !== "") localStorage.setItem(MAX_FRAMES_STORAGE, maxFrames);

  btn.disabled = true;
  status.textContent = "Extracting frames, then running 16 personas × 5 evaluations on OpenRouter. This can take a couple of minutes…";
  document.getElementById("model-note").style.display = "none";

  const fd = new FormData(form);

  // Reset and show the live feed before kicking off the run.
  resetFeed();
  showFeed();

  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || res.statusText);
    }
    const data = await res.json();
    if (!data.stream_url) {
      // Fallback (older server) — just navigate to the result.
      status.textContent = "Done.";
      history.pushState({}, "", `/?id=${data.id}`);
      await loadHistory();
      await loadEvaluation(data.id);
      return;
    }

    // Subscribe to live progress. onDone / onError unlock the button.
    await new Promise((resolve) => {
      subscribeProgress(
        data.stream_url,
        async (doneEv) => {
          status.textContent = `Done in ${doneEv.elapsed_seconds ?? "?"}s.`;
          history.pushState({}, "", `/?id=${data.id}`);
          await loadHistory();
          await loadEvaluation(data.id);
          resolve();
        },
        (errMsg) => {
          status.textContent = "Failed: " + errMsg;
          resolve();
        },
      );
    });
  } catch (err) {
    status.textContent = "Failed: " + err.message;
  } finally {
    btn.disabled = false;
  }
}

function applyFallback(evaluation) {
  // If the server fell back to a different model than the user chose, update
  // the dropdown + localStorage to reflect what actually worked, and show a
  // small note describing the fallback path.
  if (!evaluation) return;
  if (evaluation.backend !== "openrouter") return;
  const used = evaluation.model_used;
  const requested = evaluation.requested_model;
  const log = evaluation.fallback_log || [];
  const select = document.getElementById("model-select");
  const modelStorage = OPENROUTER_MODEL_STORAGE;
  if (used && select && select.value !== used) {
    let opt = Array.from(select.options).find((o) => o.value === used);
    if (!opt) {
      opt = document.createElement("option");
      opt.value = used;
      opt.textContent = used;
      select.appendChild(opt);
    }
    select.value = used;
    localStorage.setItem(modelStorage, used);
  }
  const note = document.getElementById("model-note");
  if (log.length) {
    const path = [requested, ...log.map((s) => s.to)].filter(Boolean).join(" → ");
    const reason = log[log.length - 1]?.reason || "";
    note.textContent = `Fell back: ${path}${reason ? ` (${reason})` : ""}`;
    note.style.display = "";
  } else {
    note.style.display = "none";
  }
}

async function loadHistory() {
  const el = document.getElementById("history-list");
  try {
    const res = await fetch("/api/evaluations");
    const data = await res.json();
    if (!data.evaluations.length) {
      el.textContent = "(none yet)";
      return;
    }
    el.innerHTML = "";
    data.evaluations.forEach((e) => {
      const row = document.createElement("div");
      row.className = "history-row";
      const when = e.created_at ? new Date(e.created_at * 1000).toLocaleString() : "";
      row.innerHTML = `
        <span class="filename">${escapeHtml(e.video_filename || "(no video)")}</span>
        <span class="when">${e.persona_count} personas · ${e.backend || ""} · ${when}</span>
      `;
      row.addEventListener("click", () => {
        history.pushState({}, "", `/?id=${e.id}`);
        loadEvaluation(e.id);
      });
      el.appendChild(row);
    });
  } catch (err) {
    el.textContent = "Failed to load history: " + err.message;
  }
}

async function loadEvaluation(id) {
  const res = await fetch(`/api/evaluations/${id}`);
  if (!res.ok) return;
  const data = await res.json();
  renderResults(data);
}

function renderResults(data) {
  const section = document.getElementById("results-section");
  section.classList.remove("hidden");
  document.getElementById("results-title").textContent = data.video_filename
    ? `Results — ${data.video_filename}`
    : `Results`;
  const modelPart = data.model_used ? ` · model: ${data.model_used}` : "";
  document.getElementById("results-meta").textContent =
    `${data.personas.length} personas · ${data.runs_per_persona} runs each · backend: ${data.backend}${modelPart} · ${data.elapsed_seconds}s`;

  applyFallback(data);
  renderStatsPanel(data.stats || {});

  const grid = document.getElementById("persona-grid");
  grid.innerHTML = "";
  const sorted = [...data.personas].sort((a, b) => b.rating - a.rating);
  sorted.forEach((p) => grid.appendChild(buildPersonaCard(p)));

  section.scrollIntoView({ behavior: "smooth" });
}

function renderStatsPanel(stats) {
  const el = document.getElementById("stats-panel");
  if (!stats || !Object.keys(stats).length) {
    el.innerHTML = "";
    return;
  }
  const themes = (stats.themes || []).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("");
  const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—");
  const num = (v, suffix = "") => (typeof v === "number" ? `${v}${suffix}` : "—");

  // base_pacing is the new name; fall back to old `pacing` for archived runs.
  const basePacing = stats.base_pacing || stats.pacing || "—";
  el.innerHTML = `
    <div class="stats-title">Extracted from the video</div>
    <div class="stats-grid">
      <div><span class="k">base pacing</span><span class="v">${escapeHtml(basePacing)}</span></div>
      <div><span class="k">anticipation</span><span class="v">${escapeHtml(stats.anticipation_level || "—")}</span></div>
      <div><span class="k">session arc</span><span class="v">${escapeHtml(stats.session_arc || "—")}</span></div>
      <div><span class="k">near-miss</span><span class="v">${escapeHtml(stats.near_miss_frequency || "—")}</span></div>
      <div><span class="k">dry streaks</span><span class="v">${escapeHtml(stats.dry_streak_feel || "—")}</span></div>
      <div><span class="k">brightness</span><span class="v">${escapeHtml(stats.visual_brightness || "—")}</span></div>
      <div><span class="k">theme</span><span class="v">${escapeHtml(stats.theme || "—")}</span></div>
      <div><span class="k">rounds</span><span class="v">${num(stats.rounds_played)}</span></div>
      <div><span class="k">win rate</span><span class="v">${pct(stats.win_rate)}</span></div>
      <div><span class="k">bonus rate</span><span class="v">${pct(stats.bonus_rate)}</span></div>
      <div><span class="k">avg win</span><span class="v">${num(stats.avg_win_size_multiplier, "×")}</span></div>
      <div><span class="k">biggest win</span><span class="v">${num(stats.biggest_win_multiplier, "×")}</span></div>
      <div><span class="k">volatility</span><span class="v">${escapeHtml(stats.volatility || "—")}</span></div>
      <div><span class="k">complexity</span><span class="v">${escapeHtml(stats.mechanic_complexity || "—")}</span></div>
      <div><span class="k">feature mix</span><span class="v">${escapeHtml(stats.feature_variety || "—")}</span></div>
      <div><span class="k">bonus quality</span><span class="v">${escapeHtml(stats.bonus_quality || "—")}</span></div>
      <div><span class="k">small-win anim</span><span class="v">${escapeHtml(stats.small_win_celebration_intensity || "—")}</span></div>
      <div><span class="k">tier distinction</span><span class="v">${escapeHtml(stats.win_tier_distinction || "—")}</span></div>
      <div><span class="k">anticipation honesty</span><span class="v">${escapeHtml(stats.anticipation_honesty || "—")}</span></div>
      <div><span class="k">reel stop order</span><span class="v">${escapeHtml(stats.reel_stop_order || "—")}</span></div>
      <div><span class="k">visual readability</span><span class="v">${escapeHtml(stats.visual_readability || "—")}</span></div>
      <div><span class="k">bonus rule clarity</span><span class="v">${escapeHtml(stats.bonus_rule_clarity || "—")}</span></div>
      <div><span class="k">win display</span><span class="v">${escapeHtml(stats.win_display_format || "—")}</span></div>
    </div>
    ${themes ? `<div class="stats-tags">${themes}</div>` : ""}
  `;
}

function buildPersonaCard(p) {
  const div = document.createElement("div");
  div.className = `persona-card mood-${p.mood}`;
  const pct = Math.round((p.rating / 10) * 100);
  div.innerHTML = `
    <div class="top">
      <div class="avatar">${p.persona_avatar}</div>
      <div class="who">
        <span class="name">${escapeHtml(p.persona_name)}</span>
        <span class="age">${p.persona_age} · ${escapeHtml(p.persona_archetype)}</span>
      </div>
    </div>
    <div class="rating">
      <span class="num">${p.rating.toFixed(1)}</span>
      <span class="out">/10 · ${MOOD_LABEL[p.mood] || p.mood}</span>
    </div>
    <div class="mood-bar"><div style="width:${pct}%"></div></div>
    <div class="quote">"${escapeHtml(p.quote)}"</div>
  `;
  div.addEventListener("click", () => openDetail(p));
  return div;
}

function openDetail(p) {
  const body = document.getElementById("detail-body");
  const dimBars = DIMENSIONS
    .map((d) => {
      const score = p.dimension_scores[d] || 0;
      const variance = p.dimension_variance[d] || 0;
      const pct = Math.round((score / 10) * 100);
      return `
        <div class="bar-row">
          <div class="label">${DIM_LABEL[d]}</div>
          <div class="bar"><div style="width:${pct}%"></div></div>
          <div class="val">${score}${variance ? ` <span style="color:var(--muted)">±${variance}</span>` : ""}</div>
        </div>
      `;
    })
    .join("");

  const likes = (p.likes || []).map((l) => `<li>${escapeHtml(l)}</li>`).join("");
  const dislikes = (p.dislikes || []).map((l) => `<li>${escapeHtml(l)}</li>`).join("");

  body.innerHTML = `
    <div class="detail-header">
      <div class="avatar">${p.persona_avatar}</div>
      <div>
        <h2>${escapeHtml(p.persona_name)}, ${p.persona_age}</h2>
        <div class="meta">${escapeHtml(p.persona_archetype)}</div>
        <div class="meta" style="margin-top:4px;">${escapeHtml(p.persona_bio)}</div>
      </div>
    </div>

    <div class="detail-rating mood-${p.mood}">
      <span class="num">${p.rating.toFixed(1)}</span>
      <span class="out">/10 · ${MOOD_LABEL[p.mood] || p.mood}</span>
      <span class="variance">run variance ±${(p.rating_variance || 0).toFixed(2)}</span>
    </div>

    <div class="quote-box">"${escapeHtml(p.quote)}"</div>

    <div class="section-title">Dimension scores (median of ${p.runs.length} runs)</div>
    <div class="bars">${dimBars}</div>

    <div class="section-title">What I liked</div>
    <ul class="list">${likes || "<li>(nothing)</li>"}</ul>

    <div class="section-title">What I didn't</div>
    <ul class="list">${dislikes || "<li>(nothing)</li>"}</ul>

    <div class="section-title">Would I play again?</div>
    <div>
      <span class="replay replay-${p.would_replay}">${p.would_replay}</span>
      <span>${escapeHtml(p.would_replay_reason || "")}</span>
    </div>

    <div class="section-title">Full reaction</div>
    <div class="review">${escapeHtml(p.review)}</div>
  `;

  document.getElementById("detail-panel").classList.remove("hidden");
  document.getElementById("detail-backdrop").classList.remove("hidden");
}

function closeDetail() {
  document.getElementById("detail-panel").classList.add("hidden");
  document.getElementById("detail-backdrop").classList.add("hidden");
}

function escapeHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
