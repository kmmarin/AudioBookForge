"use strict";

const S = {
  cfg: null,
  books: new Map(),
  rows: new Map(),
  remote: new Map(),   // book id -> its progress on the PC voicing it again (pollRemote)
  audio: new Audio(),
  audioUrl: null,
  playingBtn: null,
  updating: new Set(),  // books with an Update info lookup running
  sampleToken: 0,
  compareToken: 0,     // bumps when the compare dialog closes, stopping its requests
  compareId: null,
  compareEntries: null,
  detailsId: null,
  editor: null, // {bookId, idx}
  listeners: [],
  listener: null,  // id of the listener picked on this device; null = the default one
  es: null,
};
const BUSY = new Set(["voicing", "assembling", "checking", "transcribing"]);
const CANCELLABLE = new Set(["queued", "voicing", "transcribing"]);
const STARTABLE = new Set(["ready", "done", "failed", "cancelled"]);
const $ = (sel, root = document) => root.querySelector(sel);

// ---------------------------------------------------------------- utilities
async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers = { "Content-Type": "application/json" };
    delete init.json;
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    const err = new Error(typeof detail === "string" ? detail : detail?.message || "Request failed");
    err.status = res.status;
    if (detail && typeof detail === "object") Object.assign(err, detail);  // e.g. .conflict
    throw err;
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

let toastTimer;
function toast(text, isError = false) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), isError ? 6000 : 3000);
}
const fail = (err) => toast(err.message || String(err), true);

function duration(sec) {
  if (!sec || sec < 60) return "under a minute";
  const total = Math.round(sec / 60);  // round once, so 2 h 59.6 m reads 3 h, not 2 h 60 m
  const h = Math.floor(total / 60);
  const m = total % 60;
  return h ? (m ? `${h} h ${m} m` : `${h} h`) : `${m} m`;
}
const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

function fillVoiceSelect(select, withPrompt = false) {
  select.replaceChildren();
  if (withPrompt) select.append(el("option", { value: "", textContent: "Choose a voice" }));
  const groups = new Map();
  for (const v of S.cfg.voices) {
    if (!groups.has(v.group)) groups.set(v.group, el("optgroup", { label: v.group }));
    groups.get(v.group).append(el("option", { value: v.id, textContent: v.label, dataset: { label: v.label } }));
  }
  select.append(...groups.values());
}

// Shows the mood dropdown only for voices that have moods, filled with that voice's moods.
function syncMoodSelect(row, voiceId, mood) {
  const field = $(".mood-field", row);
  const select = $(".mood", row);
  const moods = S.cfg.voices.find((v) => v.id === voiceId)?.moods || [];
  field.hidden = moods.length === 0;
  if (select.dataset.voice !== voiceId) {
    select.dataset.voice = voiceId;
    select.replaceChildren(...moods.map((m) => el("option", { value: m.id, textContent: m.label, dataset: { label: m.label } })));
  }
  if (moods.length && document.activeElement !== select) select.value = mood || moods[0].id;
}

// Copies of a book that can be switched to: made at its speed, from today's text.
const madeCopies = (b) => (b.versions || []).filter((v) => !v.outdated && v.speed === (b.speed || 1));

// "✓" marks the voices (and moods) a book has already been made in: picking
// one switches to that copy at once, without voicing anything.
function markMade(row, b) {
  const made = madeCopies(b);
  for (const o of $(".voice", row).options) {
    if (!o.dataset.label) continue;
    o.textContent = made.some((v) => v.voice === o.value) ? `${o.dataset.label} ✓` : o.dataset.label;
  }
  const voiceId = $(".voice", row).value;
  for (const o of $(".mood", row).options) {
    o.textContent = made.some((v) => v.voice === voiceId && v.mood === o.value) ? `${o.dataset.label} ✓` : o.dataset.label;
  }
}

// After picking a voice: say so when the book switched to a copy it already had.
function noteSwitch(before, after) {
  const was = before?.versions?.find((v) => v.current)?.id;
  const now = after.versions?.find((v) => v.current);
  if (now && now.id !== was) toast(`Switched to ${now.label}. It was made before, so it’s ready right away.`);
}

function sampleUrl(b, voice, mood, speed) {
  const q = new URLSearchParams({ voice, speed });
  if (mood) q.set("mood", mood);
  return b?.chapters.some((c) => c.include) ? `/api/books/${b.id}/sample?${q}` : `/api/preview?${q}`;
}

function fillFormatSelect(select, withPrompt = false) {
  select.replaceChildren();
  if (withPrompt) select.append(el("option", { value: "", textContent: "Choose a format" }));
  for (const f of S.cfg.formats) select.append(el("option", { value: f.id, textContent: f.label }));
}

// -------------------------------------------------------------------- rows
function createRow(book) {
  const row = $("#row-tpl").content.firstElementChild.cloneNode(true);
  row.dataset.id = book.id;
  const voice = $(".voice", row);
  const mood = $(".mood", row);
  const fmt = $(".fmt", row);
  voice.id = `voice-${book.id}`;
  mood.id = `mood-${book.id}`;
  fmt.id = `fmt-${book.id}`;
  $(".lbl-voice", row).htmlFor = voice.id;
  $(".lbl-mood", row).htmlFor = mood.id;
  $(".lbl-fmt", row).htmlFor = fmt.id;
  fillVoiceSelect(voice);
  fillFormatSelect(fmt);

  voice.title = "✓ means this book is already made in that voice: picking it switches over at once.";
  const pick = async (json) => {
    const before = S.books.get(book.id);
    try { noteSwitch(before, await api(`/api/books/${book.id}`, { method: "PATCH", json })); } catch (err) { fail(err); }
  };
  voice.addEventListener("change", () => {
    syncMoodSelect(row, voice.value, mood.value);
    pick({ voice: voice.value });
  });
  mood.addEventListener("change", () => pick({ mood: mood.value }));
  fmt.addEventListener("change", () =>
    api(`/api/books/${book.id}`, { method: "PATCH", json: { fmt: fmt.value } }).catch(fail));
  $(".play", row).addEventListener("click", (e) => {
    const b = S.books.get(book.id);
    const moodValue = $(".mood-field", row).hidden ? null : mood.value;
    playSample(e.currentTarget, sampleUrl(b, voice.value, moodValue, b?.speed || 1));
  });
  $(".act", row).addEventListener("click", () => primaryAction(book.id));
  $(".redo", row).addEventListener("click", () => redoMisread(book.id));
  $(".send", row).addEventListener("click", (e) => sendToServer(book.id, e.currentTarget));
  $(".compare", row).addEventListener("click", () => openCompare(book.id));
  $(".details", row).addEventListener("click", () => openDetails(book.id));
  $(".update", row).addEventListener("click", (e) => updateInfo(book.id, e.currentTarget));
  $(".remove", row).addEventListener("click", () => removeBook(book.id));
  $(".meter", row).addEventListener("click", () => openDetails(book.id));
  moreMenu($(".more", row), () => bookExports(S.books.get(book.id)));
  return row;
}

// A finished book whose voice, text or details changed since its file was made
// (and isn't already being voiced again on the PC).
const needsRebuild = (b) => b.status === "done" && b.changed && !b.voicing_elsewhere;

// "28× real time": seconds of audio made per second of work.
const pace = (b) => (b.rtf ? `${b.rtf >= 10 ? Math.round(b.rtf) : b.rtf}× real time` : "");

// A book being voiced again on the PC that made it (voicing_elsewhere): nothing
// changes here meanwhile, so its progress there is asked for every few seconds.
async function pollRemote() {
  const ids = [...S.books.values()].filter((b) => b.voicing_elsewhere).map((b) => b.id);
  for (const id of S.remote.keys()) if (!ids.includes(id)) S.remote.delete(id);
  await Promise.all(ids.map(async (id) => {
    try { S.remote.set(id, await api(`/api/books/${id}/revoice`)); } catch { return; }
    const row = S.rows.get(id), b = S.books.get(id);
    if (row && b) $(".status", row).textContent = statusText(b);
  }));
}

function remoteText(b) {
  const s = S.remote.get(b.id);
  if (!s) return `Voicing again on ${b.voicing_elsewhere}. This copy is replaced when it’s done.`;
  const v = S.cfg.voices.find((x) => x.id === s.voice);
  const what = `Voicing in ${v ? v.narrator || v.label : s.voice} on ${s.where}`;
  switch (s.status) {
    case "unreachable": return `${what}: waiting for ${s.where} to answer.`;
    case "queued": return `${what}: waiting for a worker there.`;
    case "voicing": return `${what}, ${Math.round(Math.min(1, (s.progress || 0) / 0.9) * 100)}%`
      + (s.message ? ` (${s.message.replace(/^Voiced /, "").toLowerCase()})` : "") + ". This copy is replaced when it’s done.";
    case "assembling": case "checking": return `${what}: finishing up.`;
    case "done": return `Voiced in ${v ? v.narrator || v.label : s.voice} on ${s.where}. Bringing it here…`;
    case "failed": return `Voicing on ${s.where} failed: ${s.message || "see the book there"}`;
    default: return `${what}.`;
  }
}

function statusText(b) {
  const doneSeconds = b.chapters.reduce((t, c) => t + (c.include && c.duration ? c.duration : 0), 0);
  if (b.status === "done" && b.voicing_elsewhere) return remoteText(b);
  switch (b.status) {
    case "parsing": return "Reading the file and finding chapters";
    case "ready": return b.message || "Ready";
    case "queued": return b.message?.startsWith("Waiting for the GPU") ? b.message : "Waiting for a free worker";
    case "voicing": return `Voicing, ${Math.round((b.progress / 0.9) * 100)}%` + (b.rtf ? `, ${pace(b)}` : "");
    case "assembling": return "Building the audiobook file";
    case "checking": return "Checking the narration with Whisper";
    case "transcribing": return `${b.message || "Transcribing"}, ${Math.round(b.progress * 100)}%` + (b.rtf ? `, ${pace(b)}` : "");
    case "done": {
      // Older books keep "Changes saved…" in their message; it's added below when it still applies.
      const note = (b.message || "").replace(/^Done\.?\s*/, "").replace(/\s*Changes saved\. Rebuild to apply them\./, "").trim();
      const extra = note ? ` ${note}` : "";
      const changes = !needsRebuild(b) ? ""
        : b.pending === "voice" ? " Not made in this voice yet."
        : " Changes saved. Rebuild to apply them.";
      const listened = b.listened ? ` ${listenedBy()} listened to ${duration(b.listened)}.` : "";
      const made = b.kind === "audio" ? "transcribed" : "voiced";
      return `Done, ${duration(doneSeconds)}${b.rtf ? `, ${made} at ${pace(b)}` : ""}.${extra}${changes}${listened}`;
    }
    case "cancelled": return b.message || "Cancelled";
    default: return b.message || "Something went wrong";
  }
}

function renderMeter(meter, b) {
  const chapters = b.chapters;
  // An added audiobook's chapters are measured in time: its text comes later.
  const size = (c) => (b.kind === "audio" ? c.duration || 0 : c.words);
  const maxWords = Math.max(1, ...chapters.map(size));
  while (meter.children.length > chapters.length) meter.lastChild.remove();
  while (meter.children.length < chapters.length) meter.append(el("span", { class: "bar" }, el("i")));
  chapters.forEach((c, i) => {
    const bar = meter.children[i];
    const h = 22 + 78 * Math.sqrt(size(c) / maxWords);
    const fill = c.units_total ? (100 * c.units_done) / c.units_total : (b.status === "done" && c.include ? 100 : 0);
    const flagged = c.wer !== null && c.wer >= S.cfg.qa_flag;
    bar.style.setProperty("--h", `${h}%`);
    bar.style.setProperty("--fill", `${c.include ? fill : 0}%`);
    bar.className = "bar" + (c.include ? "" : " off") + (fill >= 100 ? " done" : "") + (flagged ? " flag" : "");
    const length = b.kind === "audio" ? duration(c.duration) + (c.words ? `, ${plural(c.words, "word")}` : "") : plural(c.words, "word");
    bar.title = `${c.title}, ${length}` + (c.include ? "" : " (skipped)") + (flagged ? ", may be misread" : "");
  });
  const on = chapters.filter((c) => c.include).length;
  meter.setAttribute("aria-label", `${plural(on, "chapter")} to read out of ${chapters.length}`);
}

function updateRow(row, b) {
  const busy = BUSY.has(b.status) || b.status === "queued" || b.status === "parsing";
  const audio = b.kind === "audio";  // an audiobook added for Read along: no voice or format to pick
  row.dataset.status = b.status;
  row.classList.toggle("audio", audio);
  const flagged = b.chapters.some((c) => c.include && c.wer !== null && c.wer >= S.cfg.qa_flag);
  row.toggleAttribute("data-flagged", flagged && b.status === "done");

  const title = b.meta.title || b.filename;
  $(".title", row).textContent = title;
  const on = b.chapters.filter((c) => c.include).length;
  const parts = [];
  if (b.meta.author) parts.push(b.meta.author);
  if (b.chapters.length) parts.push(`${plural(on, "chapter")}`, `${audio ? "" : "about "}${duration(b.est_seconds)}`);
  if (title !== b.filename) parts.push(b.filename);
  $(".byline", row).textContent = parts.join(", ");
  $(".status", row).textContent = statusText(b);

  const cover = $(".cover", row);
  const coverSrc = `/api/books/${b.id}/cover?v=${b.has_cover ? "c" : encodeURIComponent(title)}`;
  if (b.status !== "parsing" && cover.dataset.src !== coverSrc) {
    cover.dataset.src = coverSrc;
    cover.src = coverSrc;
  }

  renderMeter($(".meter", row), b);

  const voice = $(".voice", row);
  const fmt = $(".fmt", row);
  if (document.activeElement !== voice && voice.value !== b.voice) voice.value = b.voice;
  if (document.activeElement !== fmt && fmt.value !== b.fmt) fmt.value = b.fmt;
  if (document.activeElement !== voice) syncMoodSelect(row, b.voice, b.mood);
  markMade(row, b);
  voice.disabled = fmt.disabled = $(".mood", row).disabled = busy;

  const act = $(".act", row);
  const rebuild = needsRebuild(b);
  act.disabled = false;
  if (rebuild) act.textContent = b.pending === "voice" ? "Make this voice" : "Rebuild";
  else if (b.status === "done") act.textContent = "Download";
  else if (CANCELLABLE.has(b.status)) act.textContent = "Cancel";
  else if (b.status === "parsing") { act.textContent = "Reading"; act.disabled = true; }
  else if (BUSY.has(b.status)) { act.textContent = "Finishing"; act.disabled = true; }
  else {
    // A failed book starts again from where it stopped: finished parts are kept.
    act.textContent = b.status === "failed" ? "Retry" : audio ? "Make Read along" : "Start";
    act.disabled = !b.chapters.some((c) => c.include);
  }
  act.title = b.status === "failed"
    ? (audio ? "Try again. Chapters already transcribed are kept." : "Try again. Parts already voiced are kept.")
    : !rebuild ? ""
    : b.origin ? `Voice it on ${b.origin} with your changes. This copy is replaced when it’s done.`
    : b.pending === "voice" ? "Voice the book in this voice. The voices it’s already made in are kept."
    : "Make the audiobook file again with your changes. Parts that didn’t change are reused.";
  // While a rebuild is waiting, the file as it was last made can still be downloaded.
  const download = $(".download", row);
  const current = b.versions?.find((v) => v.current);
  download.hidden = !rebuild;
  download.href = `/api/books/${b.id}/file`;
  download.title = current ? `The copy in ${current.label}` : "The file as it was last made";
  // Once a book is on the server, Read along opens there: one place per book
  // for every device.
  // A book not voiced yet can be read already; Read along comes once it's voiced.
  const read = $(".read", row);
  const textOnly = b.status !== "done" && !audio && b.status !== "parsing" && b.chapters.some((c) => c.include && c.words);
  read.hidden = b.status !== "done" && !textOnly;
  read.textContent = textOnly ? "Read" : "Read along";
  read.href = b.read_url || `/read/${b.id}`;
  read.title = textOnly ? "Read the text now. Voice it to listen along."
    : b.read_url ? `Opens on ${S.cfg.publish_host}, where your place is kept for every device`
    : b.readalong ? "Listen while the text follows along"
    : "Made before Read along existed: open it to rebuild with word timings";
  const send = $(".send", row);
  send.hidden = !(S.cfg.publish_url && b.status === "done");
  if (!send.disabled) send.textContent = b.read_url ? "Send again" : "Send to server";
  send.title = `Store this book on ${S.cfg.publish_host} so Read along works there from any device`;
  const misread = b.chapters.filter((c) => c.include && c.wer !== null && c.wer >= S.cfg.qa_flag).length;
  const redo = $(".redo", row);
  redo.hidden = !(misread && STARTABLE.has(b.status));
  redo.title = `Voice the ${plural(misread, "chapter")} flagged as misread again, then rebuild the file`;
  $(".remove", row).disabled = b.status === "assembling" || b.status === "checking";
  $(".update", row).disabled = busy || b.status === "parsing" || S.updating.has(b.id);
}

function render(list) {
  S.books = new Map(list.map((b) => [b.id, b]));
  const ol = $("#books");
  for (const [id, row] of S.rows) {
    if (!S.books.has(id)) { row.remove(); S.rows.delete(id); }
  }
  list.forEach((b, i) => {
    let row = S.rows.get(b.id);
    if (!row) { row = createRow(b); S.rows.set(b.id, row); }
    updateRow(row, b);
    if (ol.children[i] !== row) ol.insertBefore(row, ol.children[i] || null);
  });

  $("#empty").hidden = list.length > 0;
  $("#batch").hidden = list.length === 0;
  const ready = list.filter((b) => STARTABLE.has(b.status) && b.status !== "done" && b.chapters.some((c) => c.include)).length;
  const working = list.filter((b) => BUSY.has(b.status) || b.status === "queued").length;
  $("#count").textContent = `${plural(list.length, "book")}` + (working ? `, ${working} in progress` : "");
  $("#start-all").disabled = ready === 0;
  $("#start-all").textContent = ready ? `Start ${ready} ready` : "Start all ready";
  // Added audiobooks live only in Forge, so clearing the list leaves them.
  $("#clear-done").disabled = !list.some((b) => b.status === "done" && b.kind !== "audio");

  if (S.detailsId && S.books.has(S.detailsId)) syncDetailsBusy(S.books.get(S.detailsId));
}

// ----------------------------------------------------------------- actions
async function primaryAction(id) {
  const b = S.books.get(id);
  if (!b) return;
  try {
    if (needsRebuild(b)) await rebuildBook(id);
    else if (b.status === "done") window.location.href = `/api/books/${id}/file`;
    // A server copy is voiced on the PC; an added audiobook's text is made where it is.
    else if (b.origin && b.kind !== "audio" && STARTABLE.has(b.status)) await rebuildBook(id);
    else if (CANCELLABLE.has(b.status)) await api(`/api/books/${id}/cancel`, { method: "POST" });
    else if (STARTABLE.has(b.status)) await api("/api/start", { method: "POST", json: { ids: [id] } });
  } catch (err) { fail(err); }
}

// A copy voiced on the PC goes back there; if the PC is off, it can be voiced here instead.
async function rebuildBook(id, here = false) {
  try {
    const voiceOnly = S.books.get(id)?.pending === "voice";
    const res = await api(`/api/books/${id}/rebuild${here ? "?here=true" : ""}`, { method: "POST" });
    toast(res.sent ? `${res.where} had already made this voice and sent it over.`
      : res.where !== "here" ? `Voicing on ${res.where}. This copy is replaced when it’s done.`
      : voiceOnly ? "Voicing it in the new voice. The voices it’s already made in are kept."
      : "Rebuilding with your changes.");
    return true;
  } catch (err) {
    if (err.offline && !here) {
      if (confirm(`${err.message}\n\nVoice it here on the server now?`)) return rebuildBook(id, true);
      return false;
    }
    fail(err);
    return false;
  }
}

// Icon buttons show state with classes; text buttons (data-idle) also change their label.
function sampleState(btn, state) {
  btn.classList.toggle("loading", state === "loading");
  btn.classList.toggle("on", state === "on");
  if (btn.dataset.idle) {
    btn.textContent = state === "loading" ? "Generating sample" : state === "on" ? "Stop sample" : btn.dataset.idle;
  }
}

async function sendToServer(id, btn, replace = false) {
  btn.disabled = true;
  btn.textContent = "Sending";
  try {
    await api(`/api/books/${id}/publish${replace ? "?replace=true" : ""}`, { method: "POST" });
    toast(`Sent to ${S.cfg.publish_host}. Read along there works from any device.`);
  } catch (err) {
    if (err.conflict && !replace) {
      const ok = confirm(`${err.message} Replace it with this one? Its listening progress and checkpoints on the server go with it.`);
      btn.disabled = false;
      if (ok) return sendToServer(id, btn, true);
    } else fail(err);
  }
  btn.disabled = false;
  const b = S.books.get(id);
  btn.textContent = b?.read_url ? "Send again" : "Send to server";
}

async function redoMisread(id) {
  try {
    const res = await api(`/api/books/${id}/redo`, { method: "POST", json: {} });
    toast(`Voicing ${plural(res.chapters.length, "chapter")} again. The rest is reused.`);
  } catch (err) { fail(err); }
}

async function removeBook(id) {
  const b = S.books.get(id);
  if (!b) return false;
  const note = b.kind === "audio"
    ? (CANCELLABLE.has(b.status) ? " This stops transcribing it." : "")
      + (b.in_library ? " Its text goes; the audiobook stays in your library." : " Forge’s copy of the audiobook and its text go with it.")
    : b.status === "queued" || b.status === "voicing" ? " This stops voicing it."
    : b.status === "done" ? " The audiobook file stays in your output folder." : "";
  if (!confirm(`Remove “${b.meta.title || b.filename}”?${note}`)) return false;
  try {
    await api(`/api/books/${id}`, { method: "DELETE" });
    return true;
  } catch (err) {
    fail(err);
    return false;
  }
}

// ------------------------------------------------------------------ exports
// A small ⋯ menu for what's rarely wanted, out of the way until it's looked
// for. items() lists what there is to export right now: [{label, href, title}].
// The menu sits beside its button so it works inside a modal dialog too.
function moreMenu(btn, items) {
  const menu = el("div", { class: "more-menu", popover: "auto", role: "menu" });
  btn.after(menu);
  btn.setAttribute("aria-haspopup", "menu");
  btn.setAttribute("aria-expanded", "false");
  menu.addEventListener("toggle", (e) => btn.setAttribute("aria-expanded", String(e.newState === "open")));
  let wasOpen = false;
  btn.addEventListener("pointerdown", () => { wasOpen = menu.matches(":popover-open"); });
  btn.addEventListener("click", () => {
    if (wasOpen) { wasOpen = false; return; }  // pressing the button already closed it
    const list = items();
    menu.replaceChildren(el("p", { class: "more-title", textContent: "Export" }),
      ...(list.length ? list.map((it) => el("a", { href: it.href, role: "menuitem", textContent: it.label, title: it.title || "" }))
        : [el("p", { class: "more-none", textContent: "Nothing to export yet." })]));
    menu.showPopover();
    const r = btn.getBoundingClientRect();
    const w = menu.offsetWidth, h = menu.offsetHeight;
    menu.style.left = `${Math.max(8, Math.min(r.right - w, innerWidth - w - 8))}px`;
    menu.style.top = `${r.bottom + 4 + h > innerHeight - 8 ? Math.max(8, r.top - h - 4) : r.bottom + 4}px`;
    $("a", menu)?.focus();
  });
  menu.addEventListener("click", (e) => { if (e.target.closest("a")) menu.hidePopover(); });
}

const voiceExport = (v) => ({
  label: `Voice: ${v.narrator} (ZIP)`, href: `/api/voices/${encodeURIComponent(v.id)}/export`,
  title: "Its files, for another Audiobook Forge or the engine it was made for",
});

function bookExports(b) {
  if (!b) return [];
  const items = [];
  if (b.output_name) {
    items.push({ label: `Audio (${b.output_name.split(".").pop().toUpperCase()})`, href: `/api/books/${b.id}/file`,
                 title: needsRebuild(b) ? "The file as it was last made" : b.output_name });
  }
  if (b.chapters.some((c) => c.include && c.words)) {
    items.push({ label: "Text (TXT)", href: `/api/books/${b.id}/text`, title: "The chapters read aloud, as plain text" });
  }
  const v = b.kind !== "audio" && S.cfg.voices.find((x) => x.id === b.voice);
  if (v && v.group === "Your voices") items.push(voiceExport(v));
  return items;
}

function applyTheme(theme) {
  if (theme === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  $("#theme").textContent = theme === "light" ? "Dark theme" : "Light theme";
}

// A sample is made on the voice worker, which can take a while (the voice
// creator pausing, the voices starting). Asked with progress=1 the server
// doesn't wait: until the clip is made it answers 202 with how far along it
// is, which fills a bar under the button. Each stage eases toward its share
// of the bar: [from, to, seconds it usually takes].
const SAMPLE_BAR = { queued: [0, 0.15, 4], gpu: [0, 0.4, 25], starting: [0.4, 0.8, 12], voicing: [0.8, 0.97, 4] };
const SAMPLE_POLL_MS = 600;
const SAMPLE_GIVE_UP_MS = 4 * 60 * 1000;

// The clip at `url` as a blob, or null if current() turns false (stopped or replaced).
async function fetchSample(url, btn, current) {
  const ask = `${url}${url.includes("?") ? "&" : "?"}progress=1`;
  const started = Date.now();
  const was = { title: btn.getAttribute("title"), label: btn.getAttribute("aria-label") };
  const build = String(Number(btn.dataset.build || 0) + 1);
  btn.dataset.build = build;
  let stage = null, since = started, fill = 0, timer = null;
  const draw = () => {
    const [from, to, secs] = SAMPLE_BAR[stage] || SAMPLE_BAR.queued;
    fill = Math.max(fill, from + (to - from) * (1 - Math.exp(-(Date.now() - since) / 1000 / secs)));
    btn.style.setProperty("--fill", `${(fill * 100).toFixed(1)}%`);
  };
  try {
    for (;;) {
      const got = await api(ask);
      if (!current()) return null;
      if (got instanceof Response) {
        fill = 1;
        if (timer) draw();
        return await got.blob();
      }
      if (!timer) { btn.classList.add("building"); timer = setInterval(draw, 100); }  // not for a clip made before
      if (got.stage !== stage) { stage = got.stage; since = Date.now(); }
      btn.title = got.message;
      if (was.label) btn.setAttribute("aria-label", got.message);
      if (Date.now() - started > SAMPLE_GIVE_UP_MS) {
        throw new Error("The sample is taking too long. The voice worker may be busy or stopped: try again in a minute.");
      }
      await new Promise((resolve) => setTimeout(resolve, SAMPLE_POLL_MS));
      if (!current()) return null;
    }
  } finally {
    clearInterval(timer);
    for (const [attr, value] of [["title", was.title], ["aria-label", was.label]]) {
      if (value === null) btn.removeAttribute(attr); else btn.setAttribute(attr, value);
    }
    // A moment at full, then the bar goes (unless the button is making another one by then).
    setTimeout(() => { if (btn.dataset.build === build) btn.classList.remove("building"); }, timer ? 300 : 0);
  }
}

function stopSample() {
  S.sampleToken++;
  S.audio.pause();
  if (S.playingBtn) sampleState(S.playingBtn, "idle");
  S.playingBtn = null;
  if (S.audioUrl) { URL.revokeObjectURL(S.audioUrl); S.audioUrl = null; }
}

async function playSample(btn, url) {
  if (S.playingBtn === btn) { stopSample(); return; }
  stopSample();
  const token = S.sampleToken;
  S.playingBtn = btn;
  sampleState(btn, "loading");
  try {
    const blob = await fetchSample(url, btn, () => token === S.sampleToken);
    if (!blob || token !== S.sampleToken) return; // stopped or replaced while generating
    S.audioUrl = URL.createObjectURL(blob);
    S.audio.src = S.audioUrl;
    await S.audio.play();
    sampleState(btn, "on");
  } catch (err) {
    if (token !== S.sampleToken) return;
    stopSample();
    toast(err.message && err.message !== "Request failed" ? err.message
      : "That sample couldn’t play. The voice may still be downloading, so try again in a minute.", true);
  }
}
S.audio.addEventListener("ended", () => { stopSample(); autoPlayNext(); });

async function uploadFiles(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  const pick = $("#pick");
  pick.disabled = true;
  pick.textContent = `Uploading ${plural(files.length, "file")}`;
  try {
    await api("/api/books", { method: "POST", body: form });
    toast(`Added ${plural(files.length, "book")}. Chapters appear as each file is read.`);
  } catch (err) { fail(err); }
  pick.disabled = false;
  pick.textContent = "Choose files";
}

// ------------------------------------------------------------------ details
function syncDetailsBusy(b) {
  const busy = BUSY.has(b.status) || b.status === "queued";
  const audio = b.kind === "audio";
  $("#d-busy").hidden = !busy;
  $("#d-busy").textContent = audio ? "Whisper is writing down this audiobook’s text. Cancel it to make changes."
    : "This book is being voiced. Cancel it to make changes.";
  for (const node of $("#details").querySelectorAll("input, textarea, #d-save, #d-rebuild, #d-cover-btn, #d-lookup, .chapters button, .versions button")) {
    node.disabled = busy;
  }
  $("#d-rebuild").textContent = audio ? "Save and make Read along" : b.status === "done" ? "Save and rebuild" : "Save and start";
}

// Every voice the book was made in: name it, switch to it, download it, remove it.
function renderVersions(b) {
  const list = $("#d-versions");
  list.replaceChildren();
  $("#d-versions-box").hidden = !(b.versions || []).length;
  for (const v of b.versions || []) {
    const name = el("input", { class: "v-label", value: v.name || "", placeholder: v.voice_label,
                               maxLength: 80, "aria-label": `Name for the copy in ${v.voice_label}` });
    const facts = [
      v.current ? "Current" : null,
      v.outdated ? "Made before the text changed" : null,
      v.name ? v.voice_label : null,
      duration(v.duration),
      v.size ? `${Math.max(1, Math.round(v.size / 1e6))} MB` : null,
      `made ${new Date(v.created * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" })}`,
    ].filter(Boolean).join(" · ");
    const use = el("button", { type: "button", class: "quiet", textContent: "Use", hidden: v.current || v.outdated,
                               title: "Switch the book to this copy. Nothing is voiced again." });
    const download = el("a", { class: "quiet", textContent: "Download", href: `/api/books/${b.id}/file?v=${v.id}` });
    const remove = el("button", { type: "button", class: "quiet danger", textContent: "Remove" });
    list.append(el("li", { class: (v.current ? "current" : "") + (v.outdated ? " outdated" : "") },
      el("div", {}, name, el("p", { class: "v-facts", textContent: facts })),
      el("div", { class: "v-actions" }, use, download, remove)));

    name.addEventListener("change", async () => {
      try { renderVersions(await api(`/api/books/${b.id}/versions/${v.id}`, { method: "PATCH", json: { name: name.value } })); }
      catch (err) { fail(err); }
    });
    use.addEventListener("click", async () => {
      use.disabled = true;
      try {
        renderVersions(await api(`/api/books/${b.id}/versions/${v.id}/use`, { method: "POST" }));
        toast(`Now in ${v.label}.`);
      } catch (err) { use.disabled = false; fail(err); }
    });
    remove.addEventListener("click", async () => {
      const others = b.versions.filter((x) => x.id !== v.id && !x.outdated).length;
      const note = !v.current ? ""
        : others ? " It’s the copy in use, so another voice it was made in takes over."
        : " It’s the only copy, so the book goes back to ready to voice.";
      if (!confirm(`Remove the copy in ${v.label}?${note}`)) return;
      remove.disabled = true;
      try {
        const res = await api(`/api/books/${b.id}/versions/${v.id}`, { method: "DELETE" });
        renderVersions(res.book);
        const now = res.book.versions.find((x) => x.current);
        toast(res.result === "switched" ? `Removed. Now in ${now?.label}.`
          : res.result === "emptied" ? "Removed. Pick a voice and start to make it again." : "Removed.");
      } catch (err) { remove.disabled = false; fail(err); }
    });
  }
}

async function openDetails(id) {
  let b;
  try { b = await api(`/api/books/${id}`); } catch (err) { fail(err); return; }
  S.detailsId = id;
  const dlg = $("#details");
  const audio = b.kind === "audio";
  dlg.classList.toggle("audio", audio);  // no speed, sample or narration check: it has its own narration
  $("#d-heading").textContent = b.meta.title || b.filename;
  for (const input of dlg.querySelectorAll("[data-meta]")) input.value = b.meta[input.dataset.meta] || "";
  const voice = S.cfg.voices.find((v) => v.id === b.voice);
  $("#m-narrator").placeholder = voice && !audio ? voice.narrator : "";
  $("#d-chapters-hint").textContent = audio
    ? "Turn off chapters you don’t need text for, like opening credits. Fix a word Whisper misheard with Edit text; Read along follows the change."
    : "Turn off chapters that shouldn’t be read aloud, like copyright pages. Titles become the chapter markers.";
  $("#d-cover").src = `/api/books/${id}/cover?t=${Date.now()}`;
  $("#d-speed").value = b.speed;
  $("#d-speed-out").textContent = `${Number(b.speed).toFixed(2)}×`;
  $("#d-qa").checked = b.qa;
  renderVersions(b);
  showPlaces(b);

  const list = $("#d-chapters");
  list.replaceChildren();
  for (const c of b.chapters) {
    const check = el("input", { type: "checkbox", checked: !!c.include, "aria-label": `Read ${c.title}` });
    const title = el("input", { class: "ch-title", value: c.title, "aria-label": "Chapter title" });
    const flagged = c.wer !== null && c.wer >= S.cfg.qa_flag;
    const qa = el("span", {
      class: "qa" + (flagged ? " bad" : ""),
      textContent: audio ? (c.words ? plural(c.words, "word") : "")
        : c.wer === null ? "" : flagged ? `${Math.round(c.wer * 100)}% misheard` : "Reads cleanly",
    });
    const edit = el("button", { type: "button", class: "quiet", textContent: "Edit text" });
    edit.hidden = audio && !c.words;  // no text until Whisper has been through it
    const size = audio ? duration(c.duration) : plural(c.words, "word");
    const li = el("li", { class: c.include ? "" : "off", dataset: { idx: c.idx } },
      check, title, el("span", { class: "words", textContent: size }), qa, edit);
    check.addEventListener("change", () => li.classList.toggle("off", !check.checked));
    edit.addEventListener("click", () => openEditor(id, c.idx, title.value));
    list.append(li);
  }
  syncDetailsBusy(b);
  if (!dlg.open) dlg.showModal();
}

function detailsPayload() {
  const meta = {};
  for (const input of $("#details").querySelectorAll("[data-meta]")) meta[input.dataset.meta] = input.value;
  const chapters = [...$("#d-chapters").children].map((li) => ({
    idx: Number(li.dataset.idx),
    include: $("input[type=checkbox]", li).checked,
    title: $(".ch-title", li).value,
  }));
  return { meta, chapters, speed: Number($("#d-speed").value), qa: $("#d-qa").checked };
}

async function saveDetails(andStart) {
  const id = S.detailsId;
  try {
    await api(`/api/books/${id}`, { method: "PATCH", json: detailsPayload() });
    if (andStart && S.books.get(id)?.status === "done" && S.books.get(id)?.kind !== "audio") {
      $("#details").close();
      await rebuildBook(id);  // says where it's being voiced
      return;
    }
    if (andStart) {
      const res = await api("/api/start", { method: "POST", json: { ids: [id] } });
      if (!res.started) throw new Error("Couldn’t start this book. Make sure at least one chapter is turned on.");
    }
    $("#details").close();
    toast(andStart ? "Saved and started" : "Saved");
  } catch (err) { fail(err); }
}

function refreshCover(id) {
  const src = `/api/books/${id}/cover?t=${Date.now()}`;
  $("#d-cover").src = src;
  const row = S.rows.get(id);
  if (row) { const img = $(".cover", row); img.dataset.src = ""; img.src = src; }
}

// ------------------------------------------------------------------- lookup
const SOURCE_LABELS = { openlibrary: "Open Library", audible: "Audible", google: "Google Books" };
const LOOKUP_FIELDS = ["title", "author", "series", "series_index", "year", "genre", "publisher", "description"];

function openLookup(preloaded = null, message = "") {
  $("#l-title").value = $("#m-title").value;
  $("#l-author").value = $("#m-author").value;
  showResults([], "");
  $("#lookup").showModal();
  if (preloaded) showResults(preloaded, message);
  else runLookup();
}

async function runLookup() {
  const id = S.detailsId;
  const title = $("#l-title").value.trim();
  if (!id || !title) return;
  showResults([], "Searching");
  $("#l-search").disabled = true;
  try {
    const q = new URLSearchParams({ title, author: $("#l-author").value.trim() });
    const { results, sources } = await api(`/api/books/${id}/lookup?${q}`);
    showResults(results, results.length
      ? "Pick the book that fits. Its details fill the form; save to keep them."
      : `Nothing close on ${sources.join(" or ")}. Try a shorter title, or check the author’s name.`);
  } catch (err) {
    showResults([], "");
    fail(err);
  }
  $("#l-search").disabled = false;
}

function showResults(results, message) {
  const list = $("#l-results");
  const grid = $("#l-covers");
  list.replaceChildren(...results.map(matchRow));
  const covers = [...new Set(results.map((r) => r.cover_url).filter(Boolean))].slice(0, 12);
  grid.replaceChildren(...covers.map((url) => coverPick(url, results.find((r) => r.cover_url === url))));
  $("#l-cover-box").hidden = covers.length === 0;
  $("#l-match-head").hidden = results.length === 0;
  $("#l-status").textContent = message;
}

function matchRow(r) {
  const img = el("img", { class: "m-cover", alt: "", loading: "lazy" });
  if (r.cover_url) img.src = r.cover_url;
  const series = r.series && (r.series_index ? `${r.series} #${r.series_index}` : r.series);
  const info = el("div", { class: "m-info" },
    el("p", { class: "m-title", textContent: r.title + (r.subtitle ? `: ${r.subtitle}` : "") }),
    el("p", { class: "m-by", textContent: r.author || "Unknown author" }),
    el("p", { class: "m-facts", textContent: [series, r.year, r.genre].filter(Boolean).join(", ") }),
    el("p", { class: "m-src", textContent: `${SOURCE_LABELS[r.source] || r.source}, ${Math.round(r.score * 100)}% match` }));
  const use = el("button", { type: "button", class: "quiet", textContent: "Use details" });
  use.addEventListener("click", () => useDetails(r));
  return el("li", {}, img, info, el("div", { class: "m-actions" }, use));
}

// A cover from the results; clicking it makes it the book's cover right away.
function coverPick(url, r) {
  const from = `${SOURCE_LABELS[r.source] || r.source}: ${r.title}`;
  const img = el("img", { alt: "", loading: "lazy" });
  const btn = el("button", { type: "button", class: "cover-pick", title: from, "aria-label": `Use the cover from ${from}` }, img);
  img.addEventListener("error", () => {  // the source had no image after all
    btn.remove();
    if (!$("#l-covers").children.length) $("#l-cover-box").hidden = true;
  });
  img.src = url;
  btn.addEventListener("click", () => useCover(url, btn));
  return btn;
}

function useDetails(r) {
  for (const key of LOOKUP_FIELDS) {
    const input = $(`#details [data-meta="${key}"]`);
    if (input && r[key]) input.value = r[key];
  }
  if (r.series && !r.series_index) $('#details [data-meta="series_index"]').value = "";
  $("#lookup").close();
  toast("Details filled in. Save changes to keep them.");
}

async function useCover(url, btn) {
  btn.disabled = true;
  try {
    await api(`/api/books/${S.detailsId}/cover/fetch`, { method: "POST", json: { url } });
    refreshCover(S.detailsId);
    for (const pick of $("#l-covers").children) pick.classList.remove("on");
    btn.classList.add("on");
    toast("Cover changed");
  } catch (err) { fail(err); }
  btn.disabled = false;
}

// Update info: fill in a clear match straight away, otherwise let the user pick.
async function updateInfo(id, btn) {
  S.updating.add(id);
  btn.disabled = true;
  btn.textContent = "Looking up";
  try {
    const r = await api(`/api/books/${id}/match`, { method: "POST" });
    if (r.applied) {
      toast(r.note);
      refreshCover(id);
    } else {
      await openDetails(id);
      openLookup(r.results, !r.results.length
        ? "Nothing close was found. Try a shorter title, or check the author’s name."
        : r.clear
          ? "The best match has nothing new to add. Pick another book or a cover, or close this."
          : "No clear match. Pick the right book; its details fill the form, and you save to keep them.");
    }
  } catch (err) { fail(err); }
  S.updating.delete(id);
  btn.textContent = "Update info";
  const b = S.books.get(id);
  if (b) updateRow(btn.closest(".book"), b);
}

async function openEditor(bookId, idx, title) {
  try {
    const { text } = await api(`/api/books/${bookId}/chapters/${idx}/text`);
    S.editor = { bookId, idx };
    $("#e-hint").textContent = S.books.get(bookId)?.kind === "audio"
      ? "Fix words Whisper misheard and names it spelled its own way. Blank lines separate paragraphs; a line of * * * marks a scene break. Read along keeps following the audio."
      : "Fix misread words, remove footnotes, or spell names the way they should sound. Blank lines separate paragraphs; a line of * * * adds a pause.";
    $("#e-heading").textContent = title;
    $("#e-text").value = text;
    $("#editor").showModal();
  } catch (err) { fail(err); }
}

// ------------------------------------------------------------ compare voices
// Every voice (and every mood) saying the same 10-second passage: its saved
// sample, made ahead of time by the worker (previews.py), so they play at once.
const SAVED_SECONDS = 10;
const COMPARE_PARALLEL = 3;

function savedSampleUrl(voice, mood) {
  const q = new URLSearchParams({ voice, speed: 1, seconds: SAVED_SECONDS });
  if (mood) q.set("mood", mood);
  return `/api/preview?${q}`;
}

function compareEntries() {
  // Your own voices first, then Kokoro's, then Piper's.
  const order = (v) => (v.group === "Your voices" ? -1 : v.group.startsWith("Kokoro") ? 0 : 1);
  const entries = [];
  for (const v of [...S.cfg.voices].sort((a, b) => order(a) - order(b))) {
    if (v.moods.length) for (const m of v.moods) entries.push({ voice: v, mood: m });
    else entries.push({ voice: v, mood: null });
  }
  return entries;
}

async function openCompare(id) {
  const b = S.books.get(id);
  if (!b) return;
  const token = ++S.compareToken;
  S.compareId = id;
  $("#c-heading").textContent = `Compare voices: ${b.meta.title || b.filename}`;
  const list = $("#c-voices");
  const entries = compareEntries();
  list.replaceChildren(...entries.map((e) => {
    const url = savedSampleUrl(e.voice.id, e.mood?.id);
    const play = el("button", { type: "button", class: "icon play loading", disabled: true,
      "aria-label": `Play ${e.voice.label}${e.mood ? `, ${e.mood.label}` : ""}` },
      ...[".i-play", ".i-stop"].map((icon) => $(`#row-tpl`).content.querySelector(`.play ${icon}`).cloneNode(true)));
    const meta = el("p", { class: "v-meta", textContent: "Getting its sample" });
    const made = madeCopies(b).some((v) => v.voice === e.voice.id && (v.mood || null) === (e.mood?.id || null));
    const use = el("button", { type: "button", class: "quiet", textContent: made ? "Switch to it" : "Use this voice",
                               title: made ? "Already made in this voice: switches at once" : "" });
    let acts = use;
    if (e.voice.group === "Your voices") {  // a voice made here can be taken elsewhere
      const more = $("#row-tpl").content.querySelector(".more").cloneNode(true);
      more.setAttribute("aria-label", `Export ${e.voice.label}`);
      acts = el("div", { class: "c-acts" }, use, more);
      moreMenu(more, () => [voiceExport(e.voice)]);
    }
    const li = el("li", {}, play,
      el("div", {}, el("p", { class: "v-name", textContent: e.voice.label + (e.mood ? `: ${e.mood.label}` : "") + (made ? " ✓" : "") }), meta),
      acts);
    li.classList.toggle("current", e.voice.id === b.voice && (!e.mood || e.mood.id === b.mood));
    play.addEventListener("click", () => playSample(play, url));
    use.addEventListener("click", () => useVoice(id, e));
    Object.assign(e, { url, play, meta, li, ready: false, played: false });
    return li;
  }));
  $("#c-status").textContent = `The same ${SAVED_SECONDS}-second passage in every voice. Play any as soon as it’s ready.`;
  if (!$("#compare").open) $("#compare").showModal();

  let next = 0, done = 0;
  const worker = async () => {
    while (next < entries.length && token === S.compareToken) {
      const e = entries[next++];
      try {
        // Made and cached on the server; its bar fills under its play button meanwhile.
        if (!await fetchSample(e.url, e.play, () => token === S.compareToken)) return;  // closed
        e.ready = true;
        e.meta.textContent = e.voice.group + (e.mood ? `, ${e.mood.label.toLowerCase()} mood` : "");
        e.play.disabled = false;
      } catch (err) {
        e.li.classList.add("failed");
        e.meta.textContent = err.message;
      }
      e.play.classList.remove("loading");
      done++;
      if (token !== S.compareToken) return;
      $("#c-status").textContent = done < entries.length
        ? `${done} of ${entries.length} samples ready.`
        : `The same ${SAVED_SECONDS}-second passage in every voice. Pick the one you like.`;
      autoPlayNext(entries);
    }
  };
  S.compareEntries = entries;
  await Promise.all(Array.from({ length: COMPARE_PARALLEL }, worker));
}

// With "Play each clip as it's ready" on, clips play in list order, one at a time.
function autoPlayNext(entries = S.compareEntries) {
  if (!$("#compare").open || !$("#c-auto").checked || S.playingBtn || !entries) return;
  const e = entries.find((x) => !x.played);
  if (!e || !e.ready) return;
  e.played = true;
  playSample(e.play, e.url);
}

async function useVoice(id, e) {
  const before = S.books.get(id);
  try {
    const after = await api(`/api/books/${id}`, { method: "PATCH",
      json: e.mood ? { voice: e.voice.id, mood: e.mood.id } : { voice: e.voice.id } });
    stopSample();
    $("#compare").close();
    toast(`Voice set to ${e.voice.label}${e.mood ? `, ${e.mood.label.toLowerCase()}` : ""}`);
    noteSwitch(before, after);
  } catch (err) { fail(err); }
}

// --------------------------------------------------------------------- init
function wireStatic() {
  const drop = $("#drop");
  const file = $("#file");
  file.accept = S.cfg.accept.join(",");
  const has = (...exts) => exts.some((e) => S.cfg.accept.includes(e));
  const named = [["EPUB", ".epub"], ["PDF", ".pdf"], ["Word", ".docx", ".doc"], ["OpenDocument", ".odt"],
    ["RTF", ".rtf"], ["HTML", ".html"], ["Markdown", ".md"], ["FictionBook", ".fb2"],
    ["Kindle", ".mobi", ".azw3"], ["plain text", ".txt"]].filter(([, ...exts]) => has(...exts)).map(([n]) => n);
  const list = $("#accept-list");
  list.textContent = `Add as many as you like at once. Reads ${named.slice(0, -1).join(", ")} and ${named.at(-1)}.`
    + (has(".m4b", ".mp3") ? " Add an audiobook (M4B, MP3 and more) and Whisper writes down its text for Read along." : "");
  list.title = S.cfg.accept.join(" ");
  $("#pick").addEventListener("click", () => file.click());
  file.addEventListener("change", () => { uploadFiles([...file.files]); file.value = ""; });

  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { depth++; drop.classList.add("over"); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) drop.classList.remove("over"); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    drop.classList.remove("over");
    if (e.dataTransfer?.files.length) uploadFiles([...e.dataTransfer.files]);
  });

  const allVoice = $("#all-voice");
  const allFmt = $("#all-fmt");
  fillVoiceSelect(allVoice, true);
  fillFormatSelect(allFmt, true);
  const bulk = async (field, select) => {
    if (!select.value) return;
    const ids = [...S.books.values()].filter((b) => !BUSY.has(b.status) && b.status !== "queued").map((b) => b.id);
    try {
      const res = await api("/api/bulk", { method: "POST", json: { ids, [field]: select.value } });
      toast(`Updated ${plural(res.changed, "book")}`);
    } catch (err) { fail(err); }
    select.value = "";
  };
  allVoice.addEventListener("change", () => bulk("voice", allVoice));
  allFmt.addEventListener("change", () => bulk("fmt", allFmt));

  $("#start-all").addEventListener("click", async () => {
    const ids = [...S.books.values()].filter((b) => STARTABLE.has(b.status) && b.status !== "done").map((b) => b.id);
    try {
      const res = await api("/api/start", { method: "POST", json: { ids } });
      toast(`Started ${plural(res.started, "book")}`);
    } catch (err) { fail(err); }
  });
  $("#clear-done").addEventListener("click", async () => {
    const done = [...S.books.values()].filter((b) => b.status === "done" && b.kind !== "audio");
    if (!confirm(`Remove ${plural(done.length, "finished book")} from this list? The audiobook files stay in your output folder.`)) return;
    for (const b of done) await api(`/api/books/${b.id}`, { method: "DELETE" }).catch(fail);
  });

  for (const btn of document.querySelectorAll("[data-close]")) {
    btn.addEventListener("click", () => btn.closest("dialog").close());
  }
  $("#details").addEventListener("close", () => {
    if (S.playingBtn === $("#d-sample")) stopSample();
    S.detailsId = null;
  });
  $("#compare").addEventListener("close", () => {
    S.compareToken++;  // stop making clips nobody is waiting for
    S.compareEntries = null;
    stopSample();
  });
  $("#c-auto").addEventListener("change", () => autoPlayNext());
  $("#d-sample").addEventListener("click", (e) => {
    const b = S.books.get(S.detailsId);
    if (!b) return;
    playSample(e.currentTarget, sampleUrl(b, b.voice, b.mood, $("#d-speed").value));
  });
  $("#d-speed").addEventListener("input", (e) => { $("#d-speed-out").textContent = `${Number(e.target.value).toFixed(2)}×`; });
  $("#d-save").addEventListener("click", () => saveDetails(false));
  $("#d-rebuild").addEventListener("click", () => saveDetails(true));
  $("#d-remove").addEventListener("click", async () => {
    if (await removeBook(S.detailsId)) $("#details").close();
  });
  applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  $("#theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    try { localStorage.setItem("theme", next); } catch {}
    applyTheme(next);
  });
  $("#d-cover-btn").addEventListener("click", () => $("#d-cover-file").click());
  $("#d-cover-file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const form = new FormData();
    form.append("file", f, f.name);
    try {
      await api(`/api/books/${S.detailsId}/cover`, { method: "POST", body: form });
      refreshCover(S.detailsId);
    } catch (err) { fail(err); }
    e.target.value = "";
  });
  $("#d-lookup").addEventListener("click", () => openLookup());
  $("#l-form").addEventListener("submit", (e) => { e.preventDefault(); runLookup(); });
  $("#e-save").addEventListener("click", async () => {
    const { bookId, idx } = S.editor;
    try {
      const res = await api(`/api/books/${bookId}/chapters/${idx}/text`, { method: "PUT", json: { text: $("#e-text").value } });
      const li = $(`#d-chapters li[data-idx="${idx}"]`);
      if (li) {
        $(".words", li).textContent = plural(res.words, "word");
        $(".qa", li).textContent = "";
      }
      $("#editor").close();
      const book = S.books.get(bookId);
      toast(book?.kind !== "audio" ? "Text saved. Only the changed parts are voiced again."
        : book.status === "done" ? "Text saved. Read along shows it now." : "Text saved");
    } catch (err) { fail(err); }
  });
}

function connect() {
  S.es?.close();
  const es = S.es = new EventSource(`/api/events${listenerQuery()}`);
  es.addEventListener("books", (e) => render(JSON.parse(e.data)));
  es.onerror = () => { es.close(); if (S.es === es) setTimeout(connect, 3000); };
}

// ---------------------------------------------------------------- listeners
// Everyone who reads along keeps their own place in each book. This device
// remembers who uses it (Read along asks, or pick here); "listened" in the
// book list is theirs. Without a pick it's the default (first) listener's.

const LISTENER_KEY = "listener";  // shared with Read along

function storedListener() {
  try { return JSON.parse(localStorage.getItem(LISTENER_KEY)); } catch { return null; }
}

function rememberListener(id) {
  try {
    if (id === null) localStorage.removeItem(LISTENER_KEY);
    else localStorage.setItem(LISTENER_KEY, JSON.stringify(id));
  } catch {}
}

const listenerQuery = () => (S.listener ? `?listener=${S.listener}` : "");
const deviceListener = () => S.listeners.find((l) => l.id === S.listener) || S.listeners[0];
const initial = (name) => ([...name.trim()][0] || "?").toLocaleUpperCase();

// "You’ve" with one listener; with several, who it is.
function listenedBy() {
  return S.listeners.length > 1 && deviceListener() ? `${deviceListener().name} has` : "You’ve";
}

async function loadListeners() {
  S.listeners = await api("/api/listeners").catch(() => S.listeners);
  const stored = storedListener();
  S.listener = S.listeners.some((l) => l.id === stored) ? stored : null;
}

function dayLabel(sec) {
  const day = new Date(sec * 1000); day.setHours(0, 0, 0, 0);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const diff = Math.round((today - day) / 86400000);
  return diff === 0 ? "today" : diff === 1 ? "yesterday"
    : day.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}

// Details: how far each listener has got in this book.
function showPlaces(b) {
  const places = b.places || [];
  $("#d-places").hidden = places.length === 0;
  $("#d-place-list").replaceChildren(...places.map((p) => {
    const ch = b.chapters.find((c) => c.idx === p.chapter);
    return el("li", {},
      el("span", { class: "avatar", textContent: initial(p.name), "aria-hidden": "true" }),
      el("span", { class: "p-name", textContent: p.name }),
      el("span", { class: "p-where", textContent: `${ch ? `${ch.title}, ` : ""}${duration(p.position)} into the book. Last listened ${dayLabel(p.updated)}.` }));
  }));
}

function renderListeners() {
  const mine = deviceListener();
  const only = S.listeners.length === 1;
  $("#w-list").replaceChildren(...S.listeners.map((l) => {
    const here = l === mine;
    const name = el("p", { class: "w-name", textContent: l.name });
    const note = el("p", { class: "w-note", textContent: here ? "Listening on this device" : "" });
    const who = el("div", { class: "w-who" }, name, note);
    const use = el("button", { type: "button", class: "quiet", textContent: "Use on this device", hidden: here });
    const rename = el("button", { type: "button", class: "quiet", textContent: "Rename" });
    const remove = el("button", { type: "button", class: "quiet danger", textContent: "Remove", disabled: only,
      title: only ? "Someone has to be listening, so the last listener stays. Rename it instead." : "" });
    const actions = el("div", { class: "w-actions" }, use, rename, remove);
    const li = el("li", { class: here ? "here" : "" },
      el("span", { class: "avatar", textContent: initial(l.name), "aria-hidden": "true" }), who, actions);
    use.addEventListener("click", () => useOnDevice(l.id));
    rename.addEventListener("click", () => startRename(l, who, actions));
    remove.addEventListener("click", () => removeListener(l));
    return li;
  }));
}

function startRename(l, who, actions) {
  const input = el("input", { value: l.name, maxLength: 40, "aria-label": `New name for ${l.name}` });
  const save = el("button", { type: "submit", class: "primary", textContent: "Save" });
  const cancel = el("button", { type: "button", class: "quiet", textContent: "Cancel" });
  const form = el("form", { class: "w-rename" }, input, save, cancel);
  who.replaceWith(form);
  actions.hidden = true;
  input.select();
  cancel.addEventListener("click", renderListeners);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api(`/api/listeners/${l.id}`, { method: "PATCH", json: { name: input.value } });
      await listenersChanged();
    } catch (err) { fail(err); }
  });
}

async function removeListener(l) {
  if (!confirm(`Remove ${l.name}? Their place and checkpoints in every book are deleted, on every device.`)) return;
  try {
    await api(`/api/listeners/${l.id}`, { method: "DELETE" });
    toast(`${l.name} removed`);
    await listenersChanged();
  } catch (err) { fail(err); }
}

function useOnDevice(id) {
  S.listener = id;
  rememberListener(id);
  renderListeners();
  connect();  // the book list's "listened" is theirs now
}

async function listenersChanged() {
  const before = S.listener;
  await loadListeners();
  if (before !== null && S.listener === null) rememberListener(null);  // removed: forget them here
  renderListeners();
  render([...S.books.values()]);
  connect();
}

function wireListeners() {
  $("#who").addEventListener("click", async () => {
    await loadListeners();
    renderListeners();
    $("#listeners").showModal();
  });
  $("#w-add").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#w-name");
    if (!input.value.trim()) { input.focus(); return; }
    try {
      const l = await api("/api/listeners", { method: "POST", json: { name: input.value } });
      input.value = "";
      toast(`${l.name} added`);
      await listenersChanged();
    } catch (err) { fail(err); }
  });
}

// ------------------------------------------------------- settings: GPU servers
// Machines running the clone server, and which jobs each does (app/gpus.py).
// Each server can be tested, and changed in its row: Save appears once something changed.
const G = { list: [] };

function testText(t) {
  if (!t.ok) return { text: t.error || "Not answering", bad: true };
  if (t.token_ok === false) return { text: "It answers, but doesn’t take that token.", bad: true };
  const gpu = t.gpu ? `${t.gpu}${t.gpu_gb ? `, ${t.gpu_gb} GB` : ""}` : "CPU only";
  return { text: `It answers: ${gpu}${t.draws ? "" : ", and can’t draw yet (update its clone server)"}${t.token_ok ? ", token accepted" : ""}.`,
           bad: !t.draws };
}

async function testGpu(fields, line) {
  line.classList.remove("bad");
  line.textContent = "Testing…";
  try {
    const r = testText(await api("/api/gpu-servers/test", { method: "POST", json: fields }));
    line.classList.toggle("bad", r.bad);
    line.textContent = r.text;
  } catch (err) { line.classList.add("bad"); line.textContent = err.message; }
}

function gpuStatus(s) {
  const h = s.health;
  if (!h.ok) return { text: h.error || "Not answering", bad: true };
  const parts = [h.gpu ? `${h.gpu}${h.gpu_gb ? `, ${h.gpu_gb} GB` : ""}` : "CPU only"];
  if (s.voices && !h.voices.length) parts.push("no voice model fits");
  if (s.pictures && !h.draws) parts.push("can’t draw yet: update its clone server");
  if (h.loaded) parts.push(`${h.loaded} loaded`);
  return { text: parts.join(" · "), bad: (s.voices && !h.voices.length) || (s.pictures && !h.draws) };
}

function jobBoxes(s, disabled, onChange) {
  const boxes = {};
  const row = el("div", { class: "g-row-jobs" }, ...["voices", "pictures"].map((job) => {
    boxes[job] = el("input", { type: "checkbox", checked: s[job], disabled });
    boxes[job].addEventListener("change", onChange);
    return el("label", {}, boxes[job], job === "voices" ? "Voices" : "Pictures");
  }));
  return { row, boxes };
}

function gpuRow(s, n, count) {
  const env = s.source === "env";
  const li = el("li", {});
  const st = gpuStatus(s);
  const head = el("p", { class: "w-name", textContent: `${s.name || s.url}${n === 0 && count > 1 ? " (tried first)" : ""}` });
  const where = el("p", { class: "g-status", textContent: `${s.url}${env ? " · set in .env (CLONE_URL): change it there" : s.has_token ? "" : " · no token"}` });
  const status = el("p", { class: `g-status${st.bad ? " bad" : ""}`, textContent: st.text });
  const save = el("button", { type: "button", class: "primary", textContent: "Save", hidden: true });
  const cancel = el("button", { type: "button", class: "quiet", textContent: "Cancel", hidden: true });
  // The jobs change in place; Save and Cancel show once they differ from what's saved.
  const jobs = jobBoxes(s, env, () => {
    const changed = ["voices", "pictures"].some((job) => jobs.boxes[job].checked !== s[job]);
    save.hidden = cancel.hidden = !changed;
  });
  save.addEventListener("click", () => gpuCall(`/api/gpu-servers/${s.id}`, { method: "PATCH",
    json: { voices: jobs.boxes.voices.checked, pictures: jobs.boxes.pictures.checked } }));
  cancel.addEventListener("click", () => li.replaceWith(gpuRow(s, n, count)));
  const test = el("button", { type: "button", class: "quiet", textContent: "Test" });
  test.addEventListener("click", () => testGpu({ url: s.url }, status));
  const buttons = [save, cancel, test];
  if (!env) {
    if (n > 0) {
      const first = el("button", { type: "button", class: "quiet", textContent: "Try first" });
      first.addEventListener("click", () => gpuCall(`/api/gpu-servers/${s.id}/first`, { method: "POST" }));
      buttons.push(first);
    }
    const edit = el("button", { type: "button", class: "quiet", textContent: "Edit" });
    edit.addEventListener("click", () => li.replaceWith(gpuEditRow(s, n, count)));
    const remove = el("button", { type: "button", class: "quiet danger", textContent: "Remove" });
    remove.addEventListener("click", () => {
      if (confirm(`Remove ${s.name || s.url}? Its jobs go to the other servers.`)) gpuCall(`/api/gpu-servers/${s.id}`, { method: "DELETE" });
    });
    buttons.push(edit, remove);
  }
  li.append(el("div", { class: "w-who" }, head, where, status, jobs.row), el("div", { class: "w-actions" }, ...buttons));
  return li;
}

// A server changed in its own row: Save is there only once something differs.
function gpuEditRow(s, n, count) {
  const li = el("li", { class: "g-editing" });
  const name = el("input", { value: s.name, maxLength: 60, "aria-label": "Name" });
  const url = el("input", { value: s.url, "aria-label": "Address" });
  const token = el("input", { type: "password", autocomplete: "new-password", "aria-label": "Token",
    placeholder: s.has_token ? "Blank keeps the saved token" : "Its token" });
  const result = el("p", { class: "g-status" });
  const save = el("button", { type: "button", class: "primary", textContent: "Save", disabled: true });
  const changed = () => {
    save.disabled = !(name.value.trim() !== s.name || url.value.trim() !== s.url || token.value.trim()
      || jobs.boxes.voices.checked !== s.voices || jobs.boxes.pictures.checked !== s.pictures);
  };
  const jobs = jobBoxes(s, false, changed);
  for (const input of [name, url, token]) input.addEventListener("input", changed);
  const fields = () => ({ name: name.value, url: url.value, voices: jobs.boxes.voices.checked,
    pictures: jobs.boxes.pictures.checked, ...(token.value.trim() ? { token: token.value } : {}) });
  save.addEventListener("click", async () => {
    try {
      renderGpus(await api(`/api/gpu-servers/${s.id}`, { method: "PATCH", json: fields() }));
      toast("Saved");
    } catch (err) { fail(err); }
  });
  const test = el("button", { type: "button", class: "quiet", textContent: "Test" });
  test.addEventListener("click", () => testGpu(fields(), result));
  const cancel = el("button", { type: "button", class: "quiet", textContent: "Cancel" });
  cancel.addEventListener("click", () => li.replaceWith(gpuRow(s, n, count)));
  li.append(el("div", { class: "w-who g-edit" },
    el("div", { class: "g-fields" },
      el("div", { class: "field" }, el("label", { textContent: "Name" }), name),
      el("div", { class: "field" }, el("label", { textContent: "Address" }), url),
      el("div", { class: "field" }, el("label", { textContent: "Token" }), token)),
    jobs.row, result), el("div", { class: "w-actions" }, save, test, cancel));
  name.focus();
  return li;
}

function renderGpus(data) {
  G.list = data.servers;
  $("#g-list").replaceChildren(...data.servers.map((s, n) => gpuRow(s, n, data.servers.length)));
  if (!data.servers.length) $("#g-list").append(el("li", {}, el("p", { class: "g-status", textContent: "None yet. Cloned voices and pictures need one." })));
  if (!data.servers.length) $("#g-howto").open = true;  // nothing set up yet: show how
  $("#g-where").textContent = `This list is kept in ${data.file} on the server, shared with the voice creator. A server set in .env (CLONE_URL) is listed too.`;
}

async function gpuCall(path, opts) {
  try { renderGpus(await api(path, opts)); } catch (err) { fail(err); }
}

function gpuForm() {
  return { name: $("#g-name").value, url: $("#g-url").value, token: $("#g-token").value,
           voices: $("#g-voices").checked, pictures: $("#g-pictures").checked };
}

function resetGpuForm() {
  $("#g-form").reset();
  $("#g-result").textContent = "";
}

function wireSettings() {
  for (const btn of document.querySelectorAll("#g-howto [data-copy]")) {
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(btn.parentElement.querySelector("[data-cmd]").textContent);
        toast("Copied");
      } catch { toast("Couldn’t copy here: select the command and copy it instead.", true); }
    });
  }
  $("#settings-btn").addEventListener("click", async () => {
    resetGpuForm();
    $("#g-list").replaceChildren(el("li", {}, el("p", { class: "g-status", textContent: "Checking the servers…" })));
    $("#settings").showModal();
    await gpuCall("/api/gpu-servers");
  });
  $("#g-test").addEventListener("click", () => {
    if (!$("#g-url").value.trim()) { $("#g-url").focus(); return; }
    testGpu(gpuForm(), $("#g-result"));
  });
  $("#g-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!$("#g-url").value.trim()) { $("#g-url").focus(); return; }
    try {
      renderGpus(await api("/api/gpu-servers", { method: "POST", json: gpuForm() }));
      toast("Server added");
      resetGpuForm();
    } catch (err) { fail(err); }
  });
}

// ---------------------------------------------------------- free books
// Project Gutenberg: the server searches its own copy of their catalog, and
// fetches one book per press of Add.
function wireFree() {
  const dialog = $("#free"), status = $("#f-status"), list = $("#f-results");
  $("#free-btn").hidden = !S.cfg.gutenberg;
  let token = 0;
  const search = async () => {
    const q = $("#f-q").value.trim();
    const mine = ++token;
    status.textContent = q ? "Searching…" : "";
    try {
      const res = await api(`/api/gutenberg?${new URLSearchParams({ q, lang: $("#f-lang").value })}`);
      if (mine !== token) return;
      if (!res.ready) {
        status.textContent = res.error || "Downloading Project Gutenberg’s catalog (about 20 MB, once a week). Searching starts in a moment…";
        if (!res.error) setTimeout(() => { if (dialog.open && mine === token) search(); }, 3000);
        return;
      }
      list.replaceChildren(...res.results.map(freeRow));
      status.textContent = !q ? `${res.books.toLocaleString()} books. Type a title or an author.`
        : res.results.length ? `${res.results.length === 40 ? "The first 40" : res.results.length} found.`
        : "Nothing found. Try fewer words, or another language.";
    } catch (err) { if (mine === token) status.textContent = err.message; }
  };
  $("#free-btn").addEventListener("click", () => { dialog.showModal(); $("#f-q").focus(); search(); });
  $("#f-form").addEventListener("submit", (e) => { e.preventDefault(); search(); });
  $("#f-lang").addEventListener("change", search);
}

function freeRow(b) {
  const add = el("button", { type: "button", class: "quiet", textContent: "Add" });
  const link = el("a", { href: b.url, target: "_blank", rel: "noopener", textContent: "Project Gutenberg page" });
  const meta = el("p", { class: "v-meta" }, [b.author, b.language.toUpperCase()].filter(Boolean).join(" · ") + " · ", link);
  add.addEventListener("click", async () => {
    add.disabled = true;
    add.textContent = "Adding";
    try {
      await api(`/api/gutenberg/${b.id}`, { method: "POST" });
      add.textContent = "Added";
      toast(`Added “${b.title}”. Its chapters appear in your list as it’s read.`);
    } catch (err) {
      add.disabled = false;
      add.textContent = "Add";
      fail(err);
    }
  });
  return el("li", {}, el("div", {}, el("p", { class: "v-name", textContent: b.title }), meta), add);
}

(async function init() {
  try {
    S.cfg = await api("/api/config");
  } catch (err) {
    $("#empty").textContent = "Forge isn’t responding. Check that the web container is running, then reload.";
    return;
  }
  $("#workers").textContent = `${plural(S.cfg.workers, "voice worker")}`;
  $("#create-voice").hidden = !S.cfg.voicelab;
  for (const p of document.querySelectorAll(".make-voice")) p.hidden = !S.cfg.voicelab;
  wireStatic();
  wireListeners();
  wireSettings();
  wireFree();
  await loadListeners();
  render(await api(`/api/books${listenerQuery()}`));
  connect();
  setInterval(pollRemote, 5000);
})();
