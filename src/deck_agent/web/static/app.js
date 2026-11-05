"use strict";
/* Deck Agent UI: plain JavaScript, no dependencies. */

const $ = (sel) => document.querySelector(sel);
const els = {
  messages: $("#messages"), welcome: $("#welcome"), status: $("#status"), statusText: $("#statusText"),
  statusHint: $("#statusHint"),
  form: $("#composer"), input: $("#input"), send: $("#send"), stop: $("#stop"), file: $("#file"),
  chips: $("#attachments"), model: $("#model"), web: $("#web"), factcheck: $("#factcheck"),
  newDeck: $("#newDeck"), title: $("#deckTitle"), count: $("#deckCount"), theme: $("#theme"),
  frame: $("#frame"), overlay: $("#overlay"), overlayText: $("#overlayText"), outline: $("#outline"),
  pdfTab: $("#pdfTab"), toast: $("#toast"), pages: $("#pages"), pageImg: $("#pageImg"), pageNum: $("#pageNum"),
};

const ui = {
  running: false,
  deck: { version: 0, count: 0, slides: [], title: "", theme: "light" },
  current: 1,
  mode: "html",            // "html" (live preview) | "pdf" (PowerPoint render via LibreOffice)
  pdfAvailable: false,
  pagesVersion: -1,        // deck version of the rendered page images
  pageCount: 0,
  attachments: [],
  turn: null,              // { box, list, summary, steps, start }
  reloadTimer: null,
  knownSlides: 0,
  step: 0,                 // current model step, for the status timer
  stepStart: 0,            // ms since epoch when the step started (0 = not waiting for the model)
  stepTokens: 0,
  stepPhase: "",
  loading: false,          // Ollama is still reading the model into memory
  runMode: "cpu",          // where the model runs: "cpu" (local Ollama) | "server" (remote GPU server)
  remote: false,
  modes: [],
  tool: "",                // label of the tool currently running
};

const FIRST_STEP_HINT = "The model is reading the agent's instructions (about 3,500 tokens) before it can answer. " +
  "Without a suitable GPU this first step can take a few minutes; later steps are quicker. " +
  "The terminal running deck-agent logs every step, and running 'deck-agent doctor' estimates your speed.";
const LOADING_HINT = "Ollama is reading the model file into memory. The first load after starting Ollama or your " +
  "computer can take minutes with a slow disk, little free RAM or antivirus scanning; after that the model stays " +
  "loaded for 30 minutes. If it fails after about 5 minutes, start Ollama with OLLAMA_LOAD_TIMEOUT=15m.";
const SERVER_HINT = "Waiting for the GPU server. Each step finishes with one tool call (for example one slide), " +
  "shown when it is complete. If this is much slower than usual the server may be busy or the network slow; " +
  "the terminal running deck-agent shows details, and 'deck-agent doctor --mode server' measures the server.";
const LATER_STEP_HINT = "Each step finishes with one tool call (for example one slide), which only appears when " +
  "it is complete. On a CPU a step can take a minute or more; the terminal running deck-agent shows details.";

const TOOL_LABELS = {
  web_search: "Searching the web", read_webpage: "Reading", search_images: "Finding images",
  set_deck_info: "Setting up the deck", add_bullet_slide: "Adding slide", add_two_column_slide: "Adding comparison",
  add_chart_slide: "Adding chart", add_table_slide: "Adding table", add_image_slide: "Adding image slide",
  add_section_slide: "Adding section", add_quote_slide: "Adding quote", list_slides: "Reviewing outline",
  update_slide: "Editing slide", delete_slide: "Deleting slide", move_slide: "Reordering", save_deck: "Saving files",
};
const KIND_LABELS = {
  title: "Title", section: "Section", bullets: "Bullets", two_column: "Compare", chart: "Chart",
  table: "Table", image: "Image", quote: "Quote",
};

/* ------------------------------------------------------------------ helpers */
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");

function markdown(text) {
  let html = "", inList = false;
  for (const raw of escapeHtml(text).split("\n")) {
    const line = raw.trimEnd();
    const item = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(item[1])}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

function formatSeconds(s) {
  if (s == null) return "";
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

let toastTimer;
function toast(text) {
  els.toast.textContent = text;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 4500);
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function scrollChat() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

/* ------------------------------------------------------------------ chat rendering */
function addBubble(cls, html) {
  els.welcome.hidden = true;
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.innerHTML = html;
  els.messages.appendChild(div);
  scrollChat();
  return div;
}

function addNote(text) {
  const div = document.createElement("div");
  div.className = "note";
  div.textContent = text;
  els.messages.appendChild(div);
  scrollChat();
}

function startTurn() {
  const box = document.createElement("details");
  box.className = "activity";
  box.open = true;
  box.innerHTML = "<summary>Working…</summary><ol></ol>";
  els.messages.appendChild(box);
  ui.turn = { box, list: box.querySelector("ol"), summary: box.querySelector("summary"), steps: 0, errors: 0 };
  scrollChat();
}

function addStep(name, detail) {
  if (!ui.turn) startTurn();
  const li = document.createElement("li");
  li.className = "step pending";
  li.dataset.tool = name;
  li.innerHTML = `<span class="icon"></span><span class="label">${escapeHtml(TOOL_LABELS[name] || name)}</span>` +
    `<span class="detail">${escapeHtml(detail || "")}</span>`;
  ui.turn.list.appendChild(li);
  ui.turn.steps += 1;
  ui.turn.summary.textContent = `Working… ${ui.turn.steps} step${ui.turn.steps === 1 ? "" : "s"}`;
  scrollChat();
}

function finishStep(name, ok, summary) {
  if (!ui.turn) return;
  const pending = [...ui.turn.list.querySelectorAll(".step.pending")].filter((li) => li.dataset.tool === name);
  const li = pending[0] || ui.turn.list.lastElementChild;
  if (!li) return;
  li.classList.remove("pending");
  li.classList.add(ok ? "ok" : "err");
  li.querySelector(".icon").textContent = ok ? "✓" : "!";
  const detail = li.querySelector(".detail");
  if (!ok) { detail.textContent = summary; ui.turn.errors += 1; }
  else if (summary && !detail.textContent) detail.textContent = summary;
  li.title = summary || "";
}

function endTurn(seconds) {
  if (!ui.turn) return;
  const { steps, errors, box, summary } = ui.turn;
  box.querySelectorAll(".step.pending").forEach((li) => { li.classList.remove("pending"); li.querySelector(".icon").textContent = "–"; });
  summary.textContent = `${steps} step${steps === 1 ? "" : "s"}` +
    (errors ? ` · ${errors} corrected` : "") + (seconds != null ? ` · ${formatSeconds(seconds)}` : "");
  box.open = false;
  if (!steps) box.remove();
  ui.turn = null;
}

function setRunning(running) {
  ui.running = running;
  els.send.hidden = running;
  els.stop.hidden = !running;
  els.status.hidden = !running;
  els.model.disabled = els.web.disabled = els.factcheck.disabled = els.newDeck.disabled = running;
  renderModes();
  ui.stepStart = 0;
  ui.tool = "";
  els.statusHint.hidden = true;
  if (running) els.statusText.textContent = "Starting…";
}

/* Status line: a live timer per model step, plus an explanation when a step is slow. */
function updateStatus() {
  if (!ui.running) return;
  if (ui.tool) {
    els.statusText.textContent = `${ui.tool}…`;
    els.statusHint.hidden = true;
    return;
  }
  if (!ui.stepStart) return;
  const secs = Math.max(0, Math.round((Date.now() - ui.stepStart) / 1000));
  if (ui.stepTokens) {
    els.statusText.textContent = `Step ${ui.step} · ${ui.stepPhase === "thinking" ? "reasoning" : "writing"}… ${ui.stepTokens} tokens · ${formatSeconds(secs)}`;
  } else if (ui.loading) {
    els.statusText.textContent = `Step ${ui.step} · Ollama is loading the model into memory… ${formatSeconds(secs)}`;
  } else {
    els.statusText.textContent = `Step ${ui.step} · waiting for the model… ${formatSeconds(secs)}`;
  }
  const slow = !ui.stepTokens && (ui.loading ? secs >= 20 : secs >= 45);
  els.statusHint.hidden = !slow;
  if (slow) {
    els.statusHint.textContent = ui.remote ? SERVER_HINT
      : ui.loading ? LOADING_HINT : ui.step === 1 ? FIRST_STEP_HINT : LATER_STEP_HINT;
  }
}
setInterval(updateStatus, 1000);

/* ------------------------------------------------------------------ deck preview */
function renderOutline(focus) {
  const slides = ui.deck.slides || [];
  els.outline.innerHTML = "";
  slides.forEach((s) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.n = s.n;
    b.innerHTML = `<span class="n">${s.n}</span><span class="kind">${escapeHtml(KIND_LABELS[s.kind] || s.kind)}</span>` +
      `<span class="t">${escapeHtml(s.title || "")}</span>`;
    if (s.n === ui.current) b.classList.add("active");
    if (focus && s.n === focus) b.classList.add("fresh");
    b.addEventListener("click", () => goToSlide(s.n));
    li.appendChild(b);
    els.outline.appendChild(li);
  });
  const active = els.outline.querySelector("button.active");
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function setCurrent(n) {
  ui.current = n;
  els.outline.querySelectorAll("button").forEach((b) => b.classList.toggle("active", Number(b.dataset.n) === n));
  const active = els.outline.querySelector("button.active");
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function goToSlide(n) {
  const max = ui.mode === "pdf" ? ui.pageCount : ui.deck.count;
  n = Math.min(Math.max(1, n), Math.max(1, max));
  setCurrent(n);
  if (ui.mode === "html") {
    try { els.frame.contentWindow.location.hash = String(n); } catch { reloadPreview(); }
  } else {
    showPage();
  }
}

function updateDeck(info, focus) {
  ui.deck = info;
  const has = info.count > 0;
  els.title.textContent = has ? (info.title || "Untitled deck") : "No deck yet";
  els.count.textContent = has ? `${info.count} slide${info.count === 1 ? "" : "s"}` : "";
  if (info.theme) els.theme.value = info.theme;
  document.querySelectorAll(".downloads .btn").forEach((a) => {
    const off = !has || (a.dataset.fmt === "pdf" && !ui.pdfAvailable);
    a.classList.toggle("disabled", off);
    a.setAttribute("aria-disabled", String(off));
  });
  if (focus) ui.current = focus;
  if (ui.current > info.count) ui.current = Math.max(1, info.count);
  renderOutline(focus);
  schedulePreview();
}

function schedulePreview() {
  clearTimeout(ui.reloadTimer);
  ui.reloadTimer = setTimeout(() => {
    if (ui.mode === "html") reloadPreview();
    else if (!ui.running) showPdf();
    else { els.overlay.hidden = false; els.overlayText.textContent = "The PowerPoint render refreshes when the agent finishes."; }
  }, 200);
}

function reloadPreview() {
  els.overlay.hidden = true;
  els.frame.src = `/api/preview.html?v=${ui.deck.version}#${ui.current}`;
}

function showPage() {
  els.pageImg.src = `/api/preview/page/${ui.current}.png?v=${ui.pagesVersion}`;
  els.pageNum.textContent = `${ui.current} / ${ui.pageCount}`;
}

async function showPdf() {
  els.pages.hidden = !ui.deck.count;  // hidden for an empty deck, shown again once slides exist
  if (!ui.deck.count) return;
  if (ui.pagesVersion === ui.deck.version) { showPage(); return; }
  els.overlay.hidden = false;
  els.overlayText.textContent = "Rendering the PowerPoint file with LibreOffice…";
  try {
    const res = await fetch(`/api/preview/pages?v=${ui.deck.version}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    ui.pagesVersion = data.version;
    ui.pageCount = data.count;
    showPage();
    els.overlay.hidden = true;
  } catch (err) {
    els.overlayText.textContent = `Could not render: ${err.message}`;
  }
}

function setMode(mode) {
  ui.mode = mode;
  document.querySelectorAll(".preview-bar .tab").forEach((t) => {
    const active = t.dataset.mode === mode;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", String(active));
  });
  els.frame.hidden = mode !== "html";
  els.pages.hidden = mode !== "pdf";
  if (mode === "html") reloadPreview();
  else if (ui.running) { els.overlay.hidden = false; els.overlayText.textContent = "The PowerPoint render refreshes when the agent finishes."; }
  else showPdf();
}

els.frame.addEventListener("load", () => {
  if (ui.mode !== "html") return;
  try {
    const w = els.frame.contentWindow;
    const sync = () => { const n = parseInt(w.location.hash.slice(1), 10); if (n) setCurrent(n); };
    w.addEventListener("hashchange", sync);
    w.addEventListener("slidechange", sync);  // clicks and arrow keys inside the preview
    sync();
  } catch { /* cross-origin (should not happen) */ }
});

$("#pagePrev").addEventListener("click", () => goToSlide(ui.current - 1));
$("#pageNext").addEventListener("click", () => goToSlide(ui.current + 1));
els.pageImg.addEventListener("click", () => goToSlide(ui.current + 1));
document.addEventListener("keydown", (e) => {
  if (ui.mode !== "pdf" || /^(TEXTAREA|INPUT|SELECT)$/.test(e.target.tagName)) return;
  if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); goToSlide(ui.current + 1); }
  if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); goToSlide(ui.current - 1); }
});

/* ------------------------------------------------------------------ server events */
function handle(ev) {
  switch (ev.type) {
    case "user": {
      const files = (ev.attachments || []).map((n) => `<span>📎 ${escapeHtml(n)}</span>`).join("");
      addBubble("user", escapeHtml(ev.text || "(see attached notes)") + (files ? `<div class="files">${files}</div>` : ""));
      if (ev.omitted && ev.omitted.length) {
        addNote(`Left out to fit the model's context window: ${ev.omitted.join(", ")}. Send them in a later message.`);
      } else if (ev.truncated) {
        addNote("Attachments were long and have been shortened to fit the model's context window.");
      }
      startTurn();
      break;
    }
    case "status":
      setRunning(ev.running);
      if (!ev.running) {
        endTurn(ev.seconds);
        if (ui.mode === "pdf") showPdf();
      }
      break;
    case "llm_start":
      ui.step = ev.step;
      ui.stepStart = ev.t ? ev.t * 1000 : Date.now();
      ui.stepTokens = 0;
      ui.stepPhase = "";
      ui.loading = false;
      ui.tool = "";
      updateStatus();
      break;
    case "waiting":
      if (ev.step === ui.step) { ui.loading = !!ev.loading; updateStatus(); }
      break;
    case "progress":
      ui.stepTokens = ev.tokens;
      ui.stepPhase = ev.phase;
      updateStatus();
      break;
    case "tool_call":
      addStep(ev.name, ev.summary);
      ui.tool = TOOL_LABELS[ev.name] || ev.name;
      updateStatus();
      break;
    case "tool_result":
      finishStep(ev.name, ev.ok, ev.summary);
      break;
    case "deck":
      updateDeck(ev, ev.focus);
      break;
    case "assistant":
      endTurn(null);
      addBubble("assistant", markdown(ev.text));
      break;
    case "error":
      addBubble("error", escapeHtml(ev.text));
      break;
    case "note":
      addNote(ev.text);
      break;
    case "saved":
      break; // auto-saves go to the output folder; downloads are explicit in the UI
    case "settings":
      applySettings(ev);
      break;
    case "reset":
      els.messages.querySelectorAll(".msg, .note, .activity").forEach((n) => n.remove());
      els.welcome.hidden = false;
      ui.turn = null;
      ui.current = 1;
      updateDeck(ev);
      break;
  }
}

function connect() {
  const source = new EventSource("/api/events?since=0");
  source.onmessage = (e) => handle(JSON.parse(e.data));
  source.onerror = () => { if (source.readyState === EventSource.CLOSED) setTimeout(connect, 2000); };
}

/* ------------------------------------------------------------------ settings */
/* ------------------------------------------------------------------ CPU / GPU server mode */
function renderModes() {
  document.querySelectorAll("#modeSwitch .tab").forEach((b) => {
    const info = ui.modes.find((m) => m.id === b.dataset.mode);
    const active = b.dataset.mode === ui.runMode;
    b.classList.toggle("active", active);
    b.setAttribute("aria-checked", String(active));
    b.disabled = ui.running || !info || !info.available;
    b.title = info ? info.detail : "";
  });
}

async function loadModels() {
  const s = await (await fetch("/api/state")).json();
  ui.modes = s.modes || [];
  els.model.innerHTML = "";
  (s.models || [s.model]).forEach((m) => els.model.add(new Option(m, m)));
  applySettings(s);
}

document.querySelectorAll("#modeSwitch .tab").forEach((b) => b.addEventListener("click", async () => {
  if (b.disabled || b.dataset.mode === ui.runMode) return;
  const label = b.textContent;
  b.disabled = true;
  try {
    await api("/api/settings", { mode: b.dataset.mode });
    await loadModels();
    toast(`Switched to ${label}: ${els.model.value}`);
  } catch (err) {
    toast(err.message);
  } finally {
    renderModes();
  }
}));

function applySettings(s) {
  if (s.mode) {
    ui.runMode = s.mode;
    ui.remote = !!s.remote;
    renderModes();
  }
  if (s.model) {
    if (![...els.model.options].some((o) => o.value === s.model)) els.model.add(new Option(s.model, s.model));
    els.model.value = s.model;
    els.model.dataset.prev = s.model;
  }
  els.web.checked = !!s.web;
  els.factcheck.checked = !!s.verify_numbers;
}

async function changeSetting(body, revert) {
  try { await api("/api/settings", body); }
  catch (err) { revert(); toast(err.message); }
}

els.model.addEventListener("change", () => {
  const prev = els.model.dataset.prev;
  els.model.dataset.prev = els.model.value;
  changeSetting({ model: els.model.value }, () => { els.model.value = prev; els.model.dataset.prev = prev; });
});
els.web.addEventListener("change", () => changeSetting({ web: els.web.checked }, () => { els.web.checked = !els.web.checked; }));
els.factcheck.addEventListener("change", () =>
  changeSetting({ verify_numbers: els.factcheck.checked }, () => { els.factcheck.checked = !els.factcheck.checked; }));
els.theme.addEventListener("change", async () => {
  try { await api("/api/theme", { theme: els.theme.value }); } catch (err) { toast(err.message); }
});
els.newDeck.addEventListener("click", async () => {
  if (ui.deck.count && !confirm("Start a new deck? The current conversation and deck will be cleared (saved files stay in the output folder).")) return;
  try { await api("/api/reset"); } catch (err) { toast(err.message); }
});
document.querySelectorAll(".preview-bar .tab").forEach((t) => t.addEventListener("click", () => { if (!t.disabled) setMode(t.dataset.mode); }));

/* ------------------------------------------------------------------ composer */
function renderChips() {
  els.chips.innerHTML = "";
  ui.attachments.forEach((a, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const size = a.text.length >= 1000 ? `${(a.text.length / 1000).toFixed(1)}k` : a.text.length;
    chip.innerHTML = `📎 ${escapeHtml(a.name)} <span class="muted">${size} chars</span><button type="button" aria-label="Remove">×</button>`;
    chip.querySelector("button").addEventListener("click", () => { ui.attachments.splice(i, 1); renderChips(); });
    els.chips.appendChild(chip);
  });
}

els.file.addEventListener("change", async () => {
  for (const file of [...els.file.files]) {
    try {
      let text;
      if (/\.pdf$/i.test(file.name)) {
        const res = await fetch(`/api/extract?name=${encodeURIComponent(file.name)}`, {
          method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        text = data.text;
      } else {
        text = await file.text();
      }
      ui.attachments.push({ name: file.name, text });
    } catch (err) {
      toast(`${file.name}: ${err.message}`);
    }
  }
  els.file.value = "";
  renderChips();
});

async function send() {
  const message = els.input.value.trim();
  if (ui.running) { toast("The agent is still working. Press Stop to interrupt it."); return; }
  if (!message && !ui.attachments.length) return;
  try {
    await api("/api/chat", {
      message: message || "Create a presentation from the attached material.",
      attachments: ui.attachments.map(({ name, text }) => ({ name, text })),
    });
    els.input.value = "";
    ui.attachments = [];
    renderChips();
  } catch (err) {
    toast(err.message);
  }
}

els.form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
els.stop.addEventListener("click", async () => {
  els.statusText.textContent = "Stopping…";
  try { await api("/api/stop"); } catch (err) { toast(err.message); }
});
document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => {
  els.input.value = b.textContent;
  els.input.focus();
}));
document.querySelectorAll(".downloads .btn").forEach((a) => a.addEventListener("click", (e) => {
  if (a.classList.contains("disabled")) { e.preventDefault(); return; }
  if (a.dataset.fmt === "pdf") toast("Preparing the PDF with LibreOffice…");
}));

/* ------------------------------------------------------------------ start */
async function init() {
  const res = await fetch("/api/state");
  const s = await res.json();
  ui.modes = s.modes || [];
  els.model.innerHTML = "";
  (s.models || [s.model]).forEach((m) => els.model.add(new Option(m, m)));
  els.model.dataset.prev = s.model;
  applySettings(s);
  s.themes.forEach((t) => els.theme.add(new Option(t.name.charAt(0).toUpperCase() + t.name.slice(1), t.name)));
  ui.pdfAvailable = s.pdf_available;
  if (!ui.pdfAvailable) {
    els.pdfTab.disabled = true;
    els.pdfTab.title = "Install LibreOffice to render the exact PowerPoint output (and to export PDF).";
  }
  updateDeck(s.deck);
  setRunning(s.running);
  connect();
}

init().catch((err) => toast(`Could not reach the deck-agent server: ${err.message}`));
