"use strict";

const BASE = "/createvoice";
const CHUNK = 16 * 1024 * 1024;  // upload parts: small enough for any proxy, big enough to be quick
const S = {
  cfg: null,
  projects: [],
  voices: [],
  current: null,     // open project id
  detail: null,      // its full details
  uploads: new Map(),  // source id -> {loaded, total}
  audio: new Audio(),
  playingBtn: null,
  nameTimer: null,
  refreshing: false,
  again: false,
};
const $ = (sel, root = document) => root.querySelector(sel);
const ICONS = {
  play: '<svg class="i-play" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>',
  stop: '<svg class="i-stop" viewBox="0 0 16 16" aria-hidden="true"><rect x="3.5" y="3.5" width="9" height="9" rx="1"/></svg>',
  remove: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 3.5l9 9m0-9l-9 9" fill="none"/></svg>',
};

// ---------------------------------------------------------------- utilities
async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers = { "Content-Type": "application/json" };
    delete init.json;
  }
  const res = await fetch(BASE + path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    const err = new Error(typeof detail === "string" ? detail : detail?.message || "Request failed");
    err.status = res.status;
    if (detail && typeof detail === "object") Object.assign(err, detail);
    throw err;
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

let toastTimer;
function toast(text, isError = false) {
  const t = $("#toast");
  t.textContent = text;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), isError ? 7000 : 3000);
}
const fail = (err) => toast(err.message || String(err), true);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "html") node.innerHTML = v;
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

function mins(m) {
  if (m === null || m === undefined) return "";
  if (m < 1) return m > 0 ? "under a minute" : "0 minutes";
  if (m < 90) return `${Math.round(m)} minute${Math.round(m) === 1 ? "" : "s"}`;
  return `${(m / 60).toFixed(1).replace(/\.0$/, "")} hours`;
}
const clock = (sec) => {
  sec = Math.round(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
};
const pct = (x) => `${Math.round((x || 0) * 100)}%`;
// 1 min 40 s: for the short stretches of the report, where "2 minutes" would hide too much.
function span(sec) {
  sec = Math.round(sec || 0);
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m >= 90) return mins(sec / 60);
  return m ? `${m} min${s ? ` ${s} s` : ""}` : `${s} s`;
}
// Book order: Part2 before Part10; a sync tool's "conflicted copy" after its original.
function bookOrder(a, b) {
  const conflicted = (f) => /conflicted copy/i.test(f.name);
  if (conflicted(a) !== conflicted(b)) return conflicted(a) ? 1 : -1;
  return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" });
}
const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

// ------------------------------------------------------------------ playing
function stopAudio() {
  S.audio.pause();
  if (S.playingBtn) S.playingBtn.classList.remove("on");
  S.playingBtn = null;
}

function playButton(url, label) {
  const btn = el("button", { type: "button", class: "icon play", "aria-label": label, title: label, html: ICONS.play + ICONS.stop });
  btn.addEventListener("click", async () => {
    if (S.playingBtn === btn) { stopAudio(); return; }
    stopAudio();
    S.playingBtn = btn;
    btn.classList.add("on");
    S.audio.src = BASE + url;
    try { await S.audio.play(); } catch (err) { stopAudio(); toast("That clip couldn’t play.", true); }
  });
  return btn;
}
S.audio.addEventListener("ended", stopAudio);

// ------------------------------------------------------------------ dialogs
function confirmDialog(title, text, yes = "Remove") {
  return new Promise((resolve) => {
    const d = $("#confirm");
    $("#c-title").textContent = title;
    $("#c-text").textContent = text;
    $("#c-yes").textContent = yes;
    let answer = false;
    const onYes = () => { answer = true; d.close(); };
    $("#c-yes").addEventListener("click", onYes, { once: true });
    d.addEventListener("close", () => { $("#c-yes").removeEventListener("click", onYes); resolve(answer); }, { once: true });
    d.showModal();
  });
}

function renameDialog(current) {
  return new Promise((resolve) => {
    const d = $("#rename");
    $("#n-name").value = current;
    let answer = null;
    const onSubmit = (e) => { e.preventDefault(); answer = $("#n-name").value.trim(); d.close(); };
    $("#n-form").addEventListener("submit", onSubmit);
    d.addEventListener("close", () => { $("#n-form").removeEventListener("submit", onSubmit); resolve(answer || null); }, { once: true });
    d.showModal();
    $("#n-name").select();
  });
}

// ------------------------------------------------------------------- models
// What a voice is made with: a clone model (from one clip, no training) or Piper (trained).
const modelChoice = (id) => (S.cfg.models || []).find((m) => m.id === id);
const modelLabel = (id) => modelChoice(id)?.label || id;
const isClone = (id) => modelChoice(id)?.kind === "clone";

function modelOptions(select, current, kinds = ["clone", "piper"]) {
  select.replaceChildren(...(S.cfg.models || []).filter((m) => kinds.includes(m.kind)).map((m) =>
    el("option", { value: m.id, disabled: !m.fits && m.id !== current,
      textContent: m.fits ? m.label : `${m.label} (${m.note.replace(/\.$/, "").toLowerCase()})` })));
  select.value = current;
}

// --------------------------------------------------------------------- home
// A cloned voice: samples to pick from, then the one picked is in Forge.
function cloneLine(p) {
  if (p.voice_kind === "clone") return `Cloned with ${modelLabel(p.model)}. The sample you picked is in Forge’s voice list.`;
  if (p.clone_samples) return `Samples made with ${modelLabel(p.model)}: open it, listen and pick one.`;
  if (p.status === "ready") return `${p.rating?.label || "Ready"}: ${mins(p.usable_minutes)} usable. Ready to make samples with ${modelLabel(p.model)}.`;
  return `Make samples with ${modelLabel(p.model)} to clone this voice.`;
}

function projectLine(p) {
  if (isClone(p.model) && ["ready", "done"].includes(p.status)) return cloneLine(p);
  switch (p.status) {
    case "new": return "Add recordings of the narrator to begin";
    case "ready":
      if (!p.name) return `${p.rating?.label || "Ready"}: ${mins(p.usable_minutes)} usable. Name it, then train it.`;
      return `${p.rating?.label || "Ready"}: ${mins(p.usable_minutes)} usable. Ready to train.`;
    case "done": {
      // Nothing goes into Forge until you've listened and picked a version.
      const where = p.voice_slug ? "A version you picked is in Forge’s voice list."
        : "Not in Forge: open it, listen to the checks and pick one.";
      if (p.likeness !== null && p.likeness !== undefined) {
        return `Sounds like the narrator ${pct(p.likeness)} (their own recordings ${pct(p.real_likeness)})` +
          (p.wer !== null && p.wer !== undefined ? `, ${pct(p.wer)} of words misheard` : "") +
          `${p.round === 1 ? " after its first round" : ""}. ${where}`;
      }
      return `Trained${p.quality_overall ? `, turned out ${p.quality_overall.toLowerCase()}` : ""}. ${where}`;
    }
    default: return p.message || p.status;
  }
}

// A finished voice reading the same 10-second passage as every other, to compare them.
function sampleButton(p) {
  if (p.status !== "done" || !p.voice_slug) return el("span");
  const btn = playButton(`/api/voices/${p.voice_slug}/sample`, `Play a 10-second sample of ${p.name}`);
  btn.addEventListener("click", (e) => e.stopPropagation());  // play here, don't open the voice's page
  return btn;
}

// A finished voice you'd like closer to the narrator: train it further, right
// from the list. It trains until it stops sounding any closer (Stop any time).
function trainFurther(p) {
  if (p.status !== "done" || isClone(p.model)) return el("span");
  const btn = el("button", { type: "button", class: "quiet small", textContent: "Train further",
    title: "Keeps training until it stops getting closer to the narrator and less distorted, often several hours. "
      + "Stop any time; then listen to the checks and pick one for Forge." });
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    btn.disabled = true;
    try {
      await api(`/api/projects/${p.id}/train`, { method: "POST" });
      toast(`${p.name} will train further. Forge keeps the version you picked until you pick another.`);
    } catch (err) { btn.disabled = false; fail(err); }
  });
  return btn;
}

function renderHome() {
  const list = $("#projects");
  list.replaceChildren(...S.projects.map((p) => {
    const li = el("li", { class: "project", dataset: { status: p.status }, tabIndex: 0 },
      el("div", {},
        el("p", { class: "project-name" + (p.name ? "" : " unnamed"), textContent: p.name || "Voice without a name yet" }),
        el("p", { class: "project-line", textContent: projectLine(p) +
          (p.minutes && !["analyzing", "training"].includes(p.status) ? ` ${mins(p.minutes)} of recordings.` : "") })),
      el("div", { class: "project-side" },
        // A trained voice's verdict, else how much clear audio there is.
        p.rating ? (() => {
          const overall = isClone(p.model) ? null : p.quality_overall;
          return el("span", { class: "badge", dataset: { tier: overall || p.rating.key }, textContent: overall || p.rating.label });
        })() : el("span"),
        el("div", { class: "project-actions" }, sampleButton(p), trainFurther(p))));
    li.addEventListener("click", () => openProject(p.id));
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") openProject(p.id); });
    return li;
  }));
  $("#projects-head").hidden = S.projects.length === 0;

  const voices = $("#voices");
  voices.replaceChildren(...S.voices.map((v) => {
    const sample = playButton(`/api/voices/${v.slug}/sample`, `Play a 10-second sample of ${v.name}`);
    const rename = el("button", { type: "button", class: "quiet small", textContent: "Rename" });
    const remove = el("button", { type: "button", class: "quiet small danger", textContent: "Remove" });
    rename.addEventListener("click", async () => {
      const name = await renameDialog(v.name);
      if (!name || name === v.name) return;
      try { S.voices = await api(`/api/voices/${v.slug}`, { method: "PATCH", json: { name } }); toast("Renamed. Forge shows the new name from now on."); refresh(); }
      catch (err) { fail(err); }
    });
    remove.addEventListener("click", () => removeVoice(v));
    const gone = v.project ? "" : ", its project was removed";
    let model = null;
    if (v.kind === "clone") {
      // A cloned voice can read with another model any time: same clip of the narrator.
      model = el("select", { class: "v-model", "aria-label": `Model for ${v.name}`, title: "The model Forge reads with" });
      modelOptions(model, v.model, ["clone"]);
      model.addEventListener("change", async () => {
        try {
          S.voices = await api(`/api/voices/${v.slug}`, { method: "PATCH", json: { model: model.value } });
          toast(`${v.name} reads with ${modelLabel(model.value)} from now on.`);
          refresh();
        } catch (err) { model.value = v.model; fail(err); }
      });
    }
    return el("li", {}, sample,
      el("div", {}, el("p", { class: "v-name", textContent: v.match !== null && v.match !== undefined ? `${v.name} · ${pct(v.match)} match` : v.name }),
        el("p", { class: "v-meta", textContent: v.kind === "clone" ? `Cloned${gone}` : `Trained with Piper, ${v.size_mb} MB${gone}` }),
        model),
      rename, remove);
  }));
  $("#voices-head").hidden = S.voices.length === 0;
}

async function removeVoice(v) {
  const ok = await confirmDialog(`Remove “${v.name}” from Forge?`,
    "Forge stops offering it. Books already voiced with it keep their audio; books not yet voiced switch to the default voice. " +
    (v.kind === "clone" ? "Its project stays here, so it can be put back by picking a sample again."
      : "The recordings and training stay here, so it can be exported again by training it once more."), "Remove voice");
  if (!ok) return;
  try { S.voices = await api(`/api/voices/${v.slug}`, { method: "DELETE" }); toast("Removed from Forge."); refresh(); }
  catch (err) { fail(err); }
}

// ------------------------------------------------------------------ project
function sourceRow(s) {
  const up = S.uploads.get(s.id);
  let fill = 0, meta = "", done = false;
  if (up) {
    fill = up.total ? up.loaded / up.total : 0;
    meta = `Uploading, ${pct(fill)}`;
  } else if (s.status === "uploading") {
    meta = `Upload stopped at ${pct(s.size ? s.received / s.size : 0)}. Add the same file again to carry on from there.`;
    fill = s.size ? s.received / s.size : 0;
  } else if (s.status === "failed") {
    meta = s.message || "Couldn’t read it";
  } else if (s.status === "duplicate") {
    meta = s.message || "Skipped: the same audio as another file";
  } else {
    fill = s.duration ? Math.min(1, s.done_until / s.duration) : 0;
    done = s.status === "done";
    const part = done && fill < 0.99 ? `heard the first ${clock(s.done_until)}` : "heard";
    meta = `${clock(s.duration)} long, ` + (done ? part : s.status === "analyzing" ? `listening: ${s.message || "starting"}` : "waiting to be heard");
  }
  const bar = el("div", { class: "bar-mini" + (done ? " done" : "") }, el("i"));
  bar.style.setProperty("--fill", `${Math.round(fill * 100)}%`);
  const remove = el("button", { type: "button", class: "icon", "aria-label": `Remove ${s.filename}`, title: "Remove this recording", html: ICONS.remove });
  remove.addEventListener("click", async () => {
    const ok = await confirmDialog(`Remove ${s.filename}?`, "Its clips won’t be used, and the report is made again without it.", "Remove recording");
    if (!ok) return;
    try { await api(`/api/projects/${S.current}/sources/${s.id}`, { method: "DELETE" }); refresh(); } catch (err) { fail(err); }
  });
  return el("li", { dataset: { status: s.status } },
    el("div", {}, el("p", { class: "s-name", textContent: s.filename }), el("p", { class: "s-meta", textContent: meta })),
    bar, remove);
}

// The four tiers share the bar equally; each fills by how far into it the usable minutes reach.
function renderTiers(box, usable, t, key) {
  const edges = [0, t.usable, t.good, t.great, t.plenty];
  box.replaceChildren(...[0, 1, 2, 3].map((i) => {
    const span = el("span", {}, el("i"));
    const f = Math.max(0, Math.min(1, (usable - edges[i]) / (edges[i + 1] - edges[i])));
    span.firstChild.style.setProperty("--fill", `${Math.round(f * 100)}%`);
    return span;
  }));
  box.dataset.tier = key;
  box.setAttribute("aria-label", `${mins(usable)} usable: ${key.replace("_", " ")}`);
}

function renderReadiness(d) {
  const r = d.readiness;
  $("#ready-card").hidden = !r;
  if (!r) return;
  const t = r.thresholds || S.cfg.thresholds;
  $("#r-badge").textContent = r.rating.label;
  $("#r-badge").dataset.tier = r.rating.key;
  $("#r-lead").textContent = `${mins(r.usable_minutes)} of the narrator speaking clearly, out of ${mins(r.total_minutes)} of recordings. ${r.rating.text}`;
  renderTiers($("#r-tiers"), r.usable_minutes, t, r.rating.key);
  $("#r-advice").replaceChildren(...(r.advice || []).map((a) => el("li", { textContent: a })));

  const cov = r.coverage;
  $("#r-coverage").textContent = pct(cov.share);
  $("#r-coverage-note").textContent = `${cov.covered} of ${cov.of} English sounds heard at least ${cov.min_count} times` +
    (cov.rare.length ? `. Rarest: ${cov.rare.slice(0, 3).map((x) => x.example).join(", ")}.` : ".");
  $("#r-speaker").textContent = r.speaker.label;
  $("#r-speaker-note").textContent = `${pct(r.speaker.consistency)} of the clear speech is the main voice` +
    (r.speaker.similarity !== null ? `; typical likeness to it ${pct(r.speaker.similarity)}.` : ".");
  $("#r-train").textContent = mins(r.train_minutes);
  $("#r-train-note").textContent = `${plural(r.train_clips, "clip")}, plus ${plural(r.holdout_clips, "clip")} kept aside to measure the result.`;

  // Where every minute went: what's used first, then what wasn't and why.
  const parts = [["train", "Training on it", r.train_minutes], ["holdout", "Kept aside for the quality check", r.holdout_minutes]];
  for (const x of r.dropped) parts.push([x.reason, x.label, (x.seconds ?? x.minutes * 60) / 60]);
  const total = parts.reduce((a, p) => a + p[2], 0) || 1;
  const spread = $("#r-spread");
  const legend = $("#r-legend");
  spread.replaceChildren();
  legend.replaceChildren();
  for (const [key, label, m] of parts) {
    if (m <= 0) continue;
    const kind = ["train", "holdout", "not_needed", "pause"].includes(key) ? key
      : ["skipped", "unheard"].includes(key) ? "skipped" : "drop";
    const seg = el("span", { class: `k-${kind}`, title: `${label}: ${mins(m)}` });
    seg.style.flex = `${m / total} 1 0`;
    spread.append(seg);
    const ex = (r.examples || {})[key === "train" || key === "not_needed" ? "keep" : key] || [];
    const examples = el("span", { class: "examples" },
      ...(["pause", "other", "skipped", "unheard", "holdout"].includes(key) ? [] :
        ex.slice(0, 3).map((id, n) => playButton(`/api/projects/${d.id}/clips/${id}.wav`, `Play example ${n + 1}: ${label}`))));
    const li = el("li", {}, el("span", { class: `swatch k-${kind}` }), el("span", { textContent: label }),
      el("span", { class: "mins", textContent: span(m * 60) }), examples);
    legend.append(li);
  }
  spread.setAttribute("aria-label", parts.filter((p) => p[2] > 0).map((p) => `${p[1]}: ${mins(p[2])}`).join("; "));
}

// ------------------------------------------------------- pick the clips
// The owner listens and approves the clips that are the narrator's own voice
// (lab/readiness.py). Choices save as they go, a few at a time.
const P = { project: null, filter: "review", sort: "auto", page: 1, data: null, stamp: null,
            pending: new Map(), prev: new Map(), timer: null };
const MATCH_LABELS = { close: "Close match", maybe: "Maybe", unlike: "Unlike" };
const PICK_TABS = [["review", "To review"], ["approved", "Approved"], ["left_out", "Left out"], ["dropped", "Dropped by the check"]];

function renderPicks(d) {
  const card = $("#pick-card");
  card.hidden = !(d.readiness || d.picking);
  if (card.hidden) return;
  if (document.activeElement !== $("#pk-on")) $("#pk-on").checked = d.picking;
  const stamp = JSON.stringify((d.readiness || {}).picks || null);
  if (P.project !== d.id) {
    Object.assign(P, { project: d.id, filter: "review", sort: "auto", page: 1, data: null, stamp });
    P.pending.clear();
    P.prev.clear();
    loadPicks();
  } else if (stamp !== P.stamp) {
    P.stamp = stamp;  // more heard, or picked elsewhere: fresh counts, and fresh rows unless a row is in use
    if (!P.pending.size) loadPicks(card.contains(document.activeElement));
  }
  pickTotal();
}

async function loadPicks(countsOnly = false) {
  const id = P.project;
  const q = new URLSearchParams({ filter: P.filter, sort: P.sort, page: P.page, size: 50 });
  try {
    const data = await api(`/api/projects/${id}/picks?${q}`);
    if (id !== P.project) return;
    if (countsOnly && P.data) Object.assign(P.data, { counts: data.counts, approved_minutes: data.approved_minutes });
    else { P.data = data; drawPicks(); }
    pickTotal();
  } catch (err) { fail(err); }
}

function pickTotal() {
  const d = P.data;
  if (!d) return;
  const t = d.thresholds, m = d.approved_minutes;
  const tiers = [["Usable", t.usable], ["Good", t.good], ["Great", t.great]];
  const reached = tiers.filter(([, need]) => m >= need).pop();
  const next = tiers.find(([, need]) => m < need);
  let text = `${plural(d.counts.approved, "clip")} approved · ${m.toFixed(1)} min`;
  if (reached) text += `: ${reached[0]} (${reached[1]} min) reached`;
  if (next) text += `${reached ? ";" : ":"} ${next[0]} needs ${next[1]}`;
  if (!S.detail?.picking) text += ". Picks aren’t used until “Only use the clips I approve” is on.";
  $("#pk-total").textContent = text;
  for (const btn of $("#pk-tabs").children) {
    btn.textContent = `${btn.dataset.label} ${(d.counts[btn.dataset.filter] || 0).toLocaleString()}`;
  }
  // Matching: against the clips marked ★, else the approved ones.
  const mt = d.matching || {};
  $("#pk-match").textContent = !mt.source ? "Nothing to match against yet: mark a clip or two that sound right with ★."
    : `Matching against ${plural(mt.refs, mt.source === "reference" ? "clip you marked" : "clip you approved")}: `
      + `${plural(mt.close, "close match")} not reviewed yet.`;
  const close = $("#pk-approve-close");
  close.hidden = !mt.source || !mt.close;
  close.textContent = `Approve the ${plural(mt.close || 0, "close match")}`;
}

function drawPicks() {
  const d = P.data;
  $("#pk-tabs").replaceChildren(...PICK_TABS.map(([key, label]) => {
    const btn = el("button", { type: "button", class: "quiet", dataset: { filter: key, label } });
    btn.setAttribute("aria-pressed", String(key === P.filter));
    btn.addEventListener("click", async () => {
      await flushPicks();
      Object.assign(P, { filter: key, page: 1 });
      loadPicks();
    });
    return btn;
  }));
  $("#pk-sort").value = d.sort || P.sort;
  $("#pk-sort").querySelector('[value="match"]').disabled = !d.matching?.source;
  const rows = $("#pk-rows");
  rows.replaceChildren(...d.items.map(pickRow));
  if (!d.items.length) {
    rows.append(el("li", { class: "pick-empty", textContent: P.filter === "review"
      ? "Nothing left to review here. The clips are still being heard, or every one has been picked."
      : "No clips here." }));
  }
  const pages = Math.max(1, Math.ceil(d.total / d.size));
  $("#pk-page").textContent = `Page ${d.page} of ${pages}`;
  $("#pk-prev").disabled = d.page <= 1;
  $("#pk-next").disabled = d.page >= pages;
  $("#pk-approve-page").disabled = $("#pk-leave-page").disabled = !d.items.length;
}

function pickRow(item) {
  const d = P.data;
  const li = el("li", { tabIndex: 0, dataset: { id: item.id } });
  li.setAttribute("aria-label", item.text);
  const play = playButton(`/api/projects/${P.project}/clips/${item.id}.wav`, "Play this clip");
  play.tabIndex = -1;
  const use = el("input", { type: "checkbox", checked: item.pick === 1 });
  use.setAttribute("aria-label", `Use “${item.text.slice(0, 60)}”`);
  use.addEventListener("change", () => setPick(item, use.checked ? 1 : 0, li));
  // How much it sounds like the voice marked ★ (else like the narrator overall),
  // against the bar (the mark): a close match from there up.
  const matching = item.match !== null && item.match !== undefined;
  const value = matching ? item.match : item.sim ?? null;
  const bar = matching ? (d.matching.typical ?? null) : d.threshold;
  const like = el("div", { class: "pk-like" }, el("i"), el("b"));
  if (value !== null) {
    like.firstChild.style.width = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
    like.classList.toggle("low", matching ? item.match_label === "unlike" : d.threshold !== null && value < d.threshold);
    like.classList.toggle("close", matching && item.match_label === "close");
    like.title = matching ? `Like the voice you marked: ${pct(value)} (${MATCH_LABELS[item.match_label]})`
      : `Like the narrator: ${pct(value)}${d.threshold !== null ? ` (the check drops below ${pct(d.threshold)})` : ""}`;
  } else like.title = "Not measured yet";
  if (bar !== null) like.lastChild.style.left = `${Math.round(bar * 100)}%`;
  else like.lastChild.hidden = true;
  const meta = [`${item.seconds.toFixed(1)} s`, item.source, clock(item.at)];
  if (matching) meta.unshift(MATCH_LABELS[item.match_label]);
  if (["other_voice", "announcer"].includes(item.verdict) && item.pick === null) meta.push("dropped as another voice");
  // ★: this clip is how the narrator should sound; every clip is matched against these.
  const star = el("button", { type: "button", class: "icon pk-ref", textContent: item.reference ? "★" : "☆", tabIndex: -1 });
  star.setAttribute("aria-pressed", String(item.reference));
  star.title = item.reference ? "One of the voice to match: press to take it out" : "Mark as the voice to match";
  star.addEventListener("click", () => toggleReference(item));
  const metaLine = el("p", { class: "pk-meta" }, meta.join(" · "));
  if (matching) metaLine.dataset.match = item.match_label;
  li.append(play, el("div", {}, el("p", { class: "pk-text", textContent: item.text }), metaLine),
    like, el("div", { class: "pk-acts" }, star, el("label", { class: "pk-use" }, use, "Use")));
  stylePick(li, item.pick);
  return li;
}

async function toggleReference(item) {
  await flushPicks();
  try {
    await api(`/api/projects/${P.project}/picks/reference/${item.id}`, { method: "POST", json: { on: !item.reference } });
    toast(item.reference ? "Taken out of the voice to match." : "Marked as the voice to match. Every clip is matched against the marked ones now.");
    loadPicks();
  } catch (err) { fail(err); }
}

function stylePick(li, pick) {
  li.classList.toggle("approved", pick === 1);
  li.classList.toggle("left-out", pick === 0);
  const box = $("input[type=checkbox]", li);
  if (box) box.checked = pick === 1;
}

function setPick(item, value, li) {
  if (!P.prev.has(item.id)) P.prev.set(item.id, item.pick);
  item.pick = value;
  stylePick(li, value);
  P.pending.set(item.id, value);
  clearTimeout(P.timer);
  P.timer = setTimeout(flushPicks, 500);
}

async function flushPicks() {
  clearTimeout(P.timer);
  P.timer = null;
  if (!P.pending.size) return;
  const batch = new Map(P.pending), prev = new Map(P.prev);
  P.pending.clear();
  P.prev.clear();
  try {
    const res = await api(`/api/projects/${P.project}/picks`, { method: "PUT", json: { picks: Object.fromEntries(batch) } });
    Object.assign(P.data, { counts: res.counts, approved_minutes: res.approved_minutes });
    if (S.detail && res.readiness) {
      S.detail.readiness = res.readiness;
      P.stamp = JSON.stringify(res.readiness.picks || null);
      renderReadiness(S.detail);
    }
    pickTotal();
  } catch (err) {
    fail(err);
    for (const [id, old] of prev) {  // back as they were
      const item = P.data?.items.find((x) => x.id === id);
      if (item) item.pick = old;
      const li = $(`#pk-rows li[data-id="${id}"]`);
      if (li) stylePick(li, old);
    }
  }
}

function wirePicks() {
  $("#pk-on").addEventListener("change", async (e) => {
    const on = e.target.checked;
    try {
      S.detail = await api(`/api/projects/${S.current}`, { method: "PATCH", json: { picking: on } });
      renderProject();
      loadPicks();
      toast(on ? "Only the clips you approve are used now. Every recording is listened to, so there’s all of it to pick from."
        : "The automatic choice is back. Your picks are kept for later.");
    } catch (err) { e.target.checked = !on; fail(err); }
  });
  $("#pk-sort").addEventListener("change", async () => {
    await flushPicks();
    Object.assign(P, { sort: $("#pk-sort").value, page: 1 });
    loadPicks();
  });
  for (const [id, step] of [["#pk-prev", -1], ["#pk-next", 1]]) {
    $(id).addEventListener("click", async () => {
      await flushPicks();
      P.page += step;
      loadPicks();
    });
  }
  for (const [id, value] of [["#pk-approve-page", 1], ["#pk-leave-page", 0]]) {
    $(id).addEventListener("click", () => {
      for (const li of $("#pk-rows").querySelectorAll("li[data-id]")) {
        const item = P.data.items.find((x) => x.id === Number(li.dataset.id));
        if (item && item.pick !== value) setPick(item, value, li);
      }
    });
  }
  $("#pk-approve-close").addEventListener("click", async () => {
    await flushPicks();
    try {
      const res = await api(`/api/projects/${P.project}/picks/approve-close`, { method: "POST" });
      toast(`Approved ${plural(res.approved_now, "close match")}. Leave out any that don’t sound right.`);
      if (S.detail && res.readiness) { S.detail.readiness = res.readiness; renderReadiness(S.detail); }
      loadPicks();
    } catch (err) { fail(err); }
  });
  $("#pk-relisten").addEventListener("click", async () => {
    const ok = await confirmDialog("Cut the clips again?",
      "Every recording is listened to again, and clips are cut wherever the speaker may change, so the narrator " +
      "and his character voices come apart. Your picks and ★ marks so far go; the recordings, voices made and " +
      "training stay. It takes a while on a long book.", "Cut them again");
    if (!ok) return;
    try {
      S.detail = await api(`/api/projects/${S.current}/relisten`, { method: "POST" });
      P.project = null;  // fresh counts and rows once it's heard
      renderProject();
    } catch (err) { fail(err); }
  });
  // On a row: Space plays, A approves, X leaves out (both move on), R marks the voice to match, arrows move.
  $("#pk-rows").addEventListener("keydown", (e) => {
    const li = e.target.closest("li[data-id]");
    if (!li || e.target.matches("input") && !["a", "x", "r"].includes(e.key) || e.metaKey || e.ctrlKey || e.altKey) return;
    const item = P.data.items.find((x) => x.id === Number(li.dataset.id));
    const move = (to) => { if (to?.dataset.id) to.focus(); };
    const key = e.key.toLowerCase();
    if (key === " ") { e.preventDefault(); $("button.play", li).click(); }
    else if (key === "r") { e.preventDefault(); toggleReference(item); }
    else if (key === "a" || key === "x") {
      e.preventDefault();
      setPick(item, key === "a" ? 1 : 0, li);
      move(li.nextElementSibling);
    } else if (e.key === "ArrowDown") { e.preventDefault(); move(li.nextElementSibling); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(li.previousElementSibling); }
  });
  addEventListener("pagehide", flushPicks);
}

function trainingText(d) {
  const t = d.training || {};
  const busy = d.status === "training";
  if (busy) {
    switch (t.stage) {
      case "queued": return "Waiting to start";
      case "waiting": return d.message;
      case "dataset": return d.message || "Cutting the clips for training";
      case "training": {
        if (!t.batches) return d.message || "Training is starting";
        // It trains until it stops getting closer to the narrator, measured every so often.
        let text = `Step ${t.batches.toLocaleString()}`;
        if (t.best_likeness !== null && t.best_likeness !== undefined) {
          text += `. Sounds like the narrator ${Math.round(t.best_likeness * 100)}%`;
          if (t.real_likeness) text += ` (their own recordings score ${Math.round(t.real_likeness * 100)}%)`;
          text += ". It keeps going while that, or the distortion, still improves.";
        } else {
          text += ". The first check of how much it sounds like the narrator comes soon.";
        }
        return text;
      }
      default: return d.message;
    }
  }
  if (t.stage === "failed" || d.status === "failed") return d.message;
  if (t.stage === "done") {
    const why = { converged: "It stopped by itself once it stopped sounding any closer to the narrator.",
      time: "That was its first round: Train further if it doesn’t sound enough like the narrator yet.",
      limit: "It reached the step limit.",
      stopped: "You stopped it." }[t.ended] || "";
    const taught = t.teacher ? " It learned from the cloned voice." : "";
    return `Trained for ${plural(t.batches || 0, "step")}${t.round > 1 ? ` over ${t.round} rounds` : ""}.${taught} ${why}`;
  }
  if (t.stage === "stopped") return "Training was stopped before it started.";
  return "";
}

function renderTraining(d) {
  const r = d.readiness;
  const t = d.training || {};
  const busy = d.status === "training";
  // A clone project keeps showing Piper training it has had (the same narrator, both ways),
  // and once its cloned voice is in Forge, offers to teach it to Piper.
  const cloneOnly = isClone(d.model) && !t.round;
  $("#train-card").hidden = (!r && !t.round) || (cloneOnly && !d.cloned_voice);
  const status = $("#t-status");
  status.textContent = trainingText(d);
  status.classList.toggle("bad", d.status === "failed" && t.stage === "failed");
  const prog = $("#t-progress");
  prog.hidden = !(busy && t.stage === "training" && t.batches);
  if (!prog.hidden) prog.firstElementChild.style.setProperty("--fill", `${Math.round(100 * t.batches / t.max_batches)}%`);
  renderSpark(t.history || []);
  const start = $("#t-start");
  start.hidden = busy || cloneOnly;
  start.textContent = t.stage === "done" ? "Train further" : t.round ? "Carry on training" : "Train this voice";
  let why = "";
  if (!d.name) why = "Name the voice first.";
  else if (d.listening) why = "Wait until the recordings have been heard.";
  else if (!r || r.train_clips < 10) why = "There aren’t enough usable clips yet.";
  start.disabled = Boolean(why);
  start.title = why;
  // A new round only: one already going carries on the way it started.
  const teach = $("#t-teach");
  teach.hidden = busy || !d.cloned_voice || Boolean(t.round && t.stage !== "done");
  teach.disabled = Boolean(why);
  if (cloneOnly) teach.className = "primary";
  $("#t-stop").hidden = !busy;
  let hint = why ? why + " " : "";
  if (!busy && t.stage === "done") hint += "Happy with it? Leave it. Otherwise Train further keeps training until it stops "
    + "getting closer to the narrator and less distorted, checking every 1,500 steps (often several hours). Stop any time. "
    + "Nothing goes into Forge until you pick a check below. Adding recordings first helps too. ";
  if (S.cfg.device === "cpu") hint += "This voice creator trains on the CPU, which is very slow: a real voice needs the GPU.";
  $("#t-hint").textContent = hint.trim();
  renderAlong(d);
}

// Each training check's sample, to hear how it's going and pick a version by ear.
function renderAlong(d) {
  const rows = (d.along || []).filter((a) => a.sample).reverse();
  $("#t-along-head").hidden = $("#t-along-hint").hidden = !rows.length;
  const own = d.training?.real_artifacts;
  $("#t-along").replaceChildren(...rows.map((a) => {
    const facts = [`Sounds like the narrator ${pct(a.likeness)}`];
    if (a.artifacts !== null && a.artifacts !== undefined) {
      facts.push(`distortion ${a.artifacts.toFixed(1)} dB${own !== null && own !== undefined ? ` (their recordings ${own.toFixed(1)})` : ""}`);
    }
    let use = null;
    if (a.in_forge) {
      use = el("span", { class: "badge", dataset: { tier: "Good" }, textContent: "In Forge" });
    } else if (a.voice) {
      use = el("button", { type: "button", class: "quiet small", textContent: "Use this one",
        title: "Put this version into Forge as the voice. If it's still training, what it ends with replaces it again." });
      use.addEventListener("click", async () => {
        use.disabled = true;
        try {
          S.detail = await api(`/api/projects/${d.id}/along/${a.step}/use`, { method: "POST" });
          renderProject();
          toast(`The voice from step ${a.step.toLocaleString()} is in Forge now.`);
        } catch (err) { use.disabled = false; fail(err); }
      });
    }
    return el("li", {}, playButton(`/api/projects/${d.id}/along/${a.step}.wav`, `Play step ${a.step.toLocaleString()}`),
      el("div", {}, el("p", { class: "v-name", textContent: `Step ${a.step.toLocaleString()}${a.best ? " · best by the numbers" : ""}` }),
        el("p", { class: "v-meta", textContent: facts.join(" · ") })), use);
  }));
}

function renderSpark(history) {
  const svg = $("#t-spark");
  svg.hidden = history.length < 2;
  if (svg.hidden) return;
  const xs = history.map((h) => h[0]), ys = history.map((h) => h[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pts = history.map(([x, y]) => `${(300 * (x - x0) / Math.max(1, x1 - x0)).toFixed(1)},${(56 - 52 * (y - y0) / Math.max(1e-6, y1 - y0)).toFixed(1)}`);
  svg.innerHTML = `<line x1="0" y1="59" x2="300" y2="59"/><polyline points="${pts.join(" ")}"/>`;
}

// A cloned voice: the clips of the narrator tried as its reference, each with
// the same sample passage, to pick one by ear.
function renderClone(d) {
  const clone = isClone(d.model);
  $("#clone-card").hidden = !clone;
  if (!clone) return;
  const c = d.clone || {};
  const rows = c.candidates || [];
  const choice = modelChoice(d.model);
  const real = c.real_likeness;
  $("#cl-rows").replaceChildren(...rows.filter((x) => x.sample).map((x, n) => {
    const facts = [`${x.seconds} s clip`];
    if (x.likeness !== null && x.likeness !== undefined) {
      facts.push(`sounds like the narrator ${pct(x.likeness)}${real ? ` (their own clips ${pct(real)})` : ""}`);
    }
    let use;
    if (x.in_forge) {
      use = el("span", { class: "badge", dataset: { tier: "Good" }, textContent: "In Forge" });
    } else {
      use = el("button", { type: "button", class: "quiet small", textContent: "Use this one",
        title: "Put the voice cloned from this clip into Forge. Picking another one later replaces it." });
      use.addEventListener("click", async () => {
        if (!$("#p-name").value.trim()) { $("#p-name").focus(); toast("Name the voice first.", true); return; }
        if (S.nameTimer) saveName(true);
        use.disabled = true;
        try {
          S.detail = await api(`/api/projects/${d.id}/clone/${x.clip}/use`, { method: "POST" });
          renderProject();
          toast(`The voice cloned from sample ${n + 1} is in Forge now.`);
        } catch (err) { use.disabled = false; fail(err); }
      });
    }
    const play = el("div", { class: "q-play" },
      el("figure", {}, playButton(`/api/projects/${d.id}/clone/${x.clip}/ref.wav`, `Play the narrator’s clip ${n + 1}`), el("figcaption", { textContent: "Narrator" })),
      el("figure", {}, playButton(`/api/projects/${d.id}/clone/${x.clip}/sample.wav`, `Play sample ${n + 1}`), el("figcaption", { textContent: "Clone" })));
    return el("li", {}, play,
      el("div", {}, el("p", { class: "v-name", textContent: `Sample ${n + 1}` }),
        el("p", { class: "v-meta", textContent: facts.join(" · ") }),
        el("p", { class: "q-text", textContent: `From: “${x.text}”` })),
      use);
  }));
  $("#cl-hint").hidden = !rows.length;
  const status = $("#cl-status");
  let text;
  if (c.busy) text = d.message || "Making samples";
  else if (c.stage === "failed") text = d.message || "Couldn’t make samples.";
  else if (rows.length) text = `Samples made with ${modelLabel(d.model)}.` + (d.voice_kind === "clone" ? "" : " Nothing is in Forge until you pick one.");
  else text = `Make samples to hear the narrator cloned with ${modelLabel(d.model)}: a few clips of them are tried, no training.`;
  status.textContent = text;
  status.classList.toggle("bad", !c.busy && c.stage === "failed");
  const make = $("#cl-make");
  make.hidden = c.busy;
  make.textContent = rows.length ? "Make new samples" : "Make samples";
  let why = "";
  if (choice && !choice.fits) why = choice.note;
  else if (d.listening) why = "Wait until the recordings have been heard.";
  else if (!d.readiness) why = "Add recordings of the narrator first.";
  make.disabled = Boolean(why);
  make.title = why;
}

function renderQuality(d) {
  const q = isClone(d.model) ? null : d.quality;  // a trained voice's report
  $("#quality-card").hidden = !q;
  if (!q) return;
  $("#q-badge").hidden = !q.overall;
  $("#q-badge").textContent = q.overall || "";
  $("#q-badge").dataset.tier = q.overall || "";
  $("#q-text").textContent = q.text;
  const facts = $("#q-facts");
  facts.replaceChildren();
  $("#q-rows-head").hidden = $("#q-rows-hint").hidden = !q.rows;
  if (!q.rows) { $("#q-rows").replaceChildren(); return; }
  const fact = (label, value, note) => el("div", { class: "fact" }, el("p", { class: "fact-label", textContent: label }),
    el("p", { class: "fact-value", textContent: value }), el("p", { class: "fact-note", textContent: note }));
  facts.append(
    fact("Clarity", `${Math.round(q.wer * 100)}% misheard`,
      `Whisper’s transcript of the new voice against the text.${q.base_wer !== null ? ` ${q.base_name}: ${Math.round(q.base_wer * 100)}%.` : ""}`),
    fact("Likeness", pct(q.likeness),
      `To the narrator’s voice. The narrator’s own clips score ${pct(q.real_likeness)}` +
      (q.base_likeness !== null ? `; ${q.base_name}, where training started, ${pct(q.base_likeness)}.` : ".")),
    fact("Measured on", plural(q.sentences, "sentence"), "Never used for training."));
  $("#q-rows").replaceChildren(...q.rows.map((row) => {
    const play = el("div", { class: "q-play" },
      el("figure", {}, playButton(`/api/projects/${d.id}/quality/${q.round}/real/${row.file}`, "Play the narrator"), el("figcaption", { textContent: "Narrator" })),
      el("figure", {}, playButton(`/api/projects/${d.id}/quality/${q.round}/new/${row.file}`, "Play the new voice"), el("figcaption", { textContent: "New voice" })),
      ...(row.base_wer !== undefined ? [el("figure", {}, playButton(`/api/projects/${d.id}/quality/${q.round}/base/${row.file}`, "Play the starting voice"), el("figcaption", { textContent: "Starting voice" }))] : []));
    const heard = row.wer > 0 ? el("p", { class: "q-heard" + (row.wer > 0.15 ? " off" : ""), textContent: `Heard: “${row.heard}”` }) : null;
    return el("li", {}, el("div", {}, el("p", { class: "q-text", textContent: row.text }), heard), play);
  }));
}

function renderVoice(d) {
  const v = d.voice;
  $("#voice-card").hidden = !v;
  if (!v) return;
  $("#v-text").textContent = v.kind === "clone"
    ? `Forge lists it as “${v.name}”, under Your voices, read with ${modelLabel(v.model)}. Pick it for any book like the other voices.`
    : `Forge lists it as “${v.name} (custom)”, under Your voices. Pick it for any book like the other voices.`;
}

function renderProject() {
  const d = S.detail;
  if (!d) return;
  const name = $("#p-name");
  if (document.activeElement !== name && !S.nameTimer) name.value = d.name || "";
  const hasAudio = d.sources.length > 0;
  name.classList.toggle("missing", !d.name && hasAudio);
  $("#p-name-note").textContent = d.name && d.name_note
    ? `${d.name_note}. Change it if you like: it’s the name Forge lists the voice under.`
    : d.name ? "This is the name Forge lists the voice under."
    : d.name_note ? `${d.name_note}. A name is needed before training; it’s what Forge lists the voice under.`
    : "Required before training. This is the name Forge lists the voice under.";
  // By itself: samples for a cloned voice; training only for a name typed here, before the first round.
  const clone = isClone(d.model);
  $("#p-auto-box").hidden = clone ? Boolean((d.clone || {}).made) : Boolean(d.training?.round);
  $("#p-auto").checked = d.auto_train;
  $("#p-auto-text").textContent = clone ? "Make samples as soon as the audio is ready"
    : d.name_source === "user" || !d.name
    ? "Start training as soon as the audio is ready"
    : "Start training as soon as the audio is ready (a name found in the recordings waits for you to check it, then press Train this voice)";
  const model = $("#p-model");
  if (document.activeElement !== model) modelOptions(model, d.model);
  model.disabled = d.status === "training";
  const choice = modelChoice(d.model);
  $("#p-model-note").textContent = d.status === "training" ? "Fixed while it trains."
    : choice?.kind === "clone" ? "Cloned from one clip of the narrator: no training." : "Trained on this machine for hours.";
  const accent = $("#p-accent");
  if (document.activeElement !== accent) accent.value = d.accent;
  accent.disabled = Boolean(d.training?.round);
  const starts = S.cfg.accents.find((a) => a.id === d.accent)?.starts || [];
  accent.title = accent.disabled ? "Fixed once training has started"
    : `Training starts from whichever of ${starts.join(", ")} sounds most like the narrator`;

  $("#p-sources").replaceChildren(...d.sources.map(sourceRow));
  $("#stop-listening").hidden = !d.listening;
  $("#drop-more-text").textContent = hasAudio ? "Drop more recordings of the same narrator here, or" : "Drop recordings of the narrator here, or";
  const st = $("#p-status");
  const showStatus = ["new", "analyzing"].includes(d.status) || (d.status === "failed" && d.training?.stage !== "failed");
  st.textContent = showStatus ? (d.status === "analyzing" ? `${d.message || "Listening"}. The report appears when it’s done.` : d.message || "") : "";
  st.classList.toggle("bad", d.status === "failed");
  renderReadiness(d);
  renderPicks(d);
  renderClone(d);
  renderTraining(d);
  renderQuality(d);
  renderVoice(d);
}

// -------------------------------------------------------------- navigation
async function openProject(id) {
  if (location.hash !== `#${id}`) history.pushState(null, "", `#${id}`);
  S.current = id;
  S.detail = null;
  $("#home").hidden = true;
  $("#project").hidden = false;
  try {
    try { S.cfg = await api("/api/config"); } catch {}  // which models the clone server can run now
    S.detail = await api(`/api/projects/${id}`);
    renderProject();
    window.scrollTo(0, 0);
  } catch (err) {
    fail(err);
    goHome();
  }
}

function goHome() {
  if (location.hash) history.pushState(null, "", location.pathname);
  S.current = null;
  S.detail = null;
  stopAudio();
  $("#project").hidden = true;
  $("#home").hidden = false;
  refresh();
}

async function refresh() {
  if (S.refreshing) { S.again = true; return; }
  S.refreshing = true;
  try {
    if (S.current) {
      S.detail = await api(`/api/projects/${S.current}`);
      renderProject();
    } else {
      [S.projects, S.voices] = await Promise.all([api("/api/projects"), api("/api/voices")]);
      renderHome();
    }
    refreshGpu();
  } catch (err) {
    if (err.status === 404 && S.current) goHome();
  } finally {
    S.refreshing = false;
    if (S.again) { S.again = false; refresh(); }
  }
}

async function refreshGpu() {
  try {
    const g = await api("/api/gpu");
    let text = g.device === "cuda" ? "Trains on the GPU" : "Trains on the CPU";
    if (g.lab.state === "holding") text = "Using the GPU now";
    else if (g.lab.state === "waiting") text = g.forge?.busy ? "Waiting for Forge to finish voicing" : "Waiting for the GPU";
    $("#gpu-line").textContent = text;
  } catch {}
}

// ------------------------------------------------------------------ uploads
function putPart(url, blob, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", BASE + url);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => onProgress(e.loaded);
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else {
        const detail = body.detail;
        const err = new Error(typeof detail === "string" ? detail : detail?.message || `Upload failed (${xhr.status})`);
        if (detail && typeof detail === "object") Object.assign(err, detail);
        err.status = xhr.status;
        reject(err);
      }
    };
    xhr.onerror = () => reject(new Error("The connection dropped"));
    xhr.send(blob);
  });
}

async function uploadFiles(projectId, files) {
  const accept = new Set(S.cfg.audio_exts);
  // Originals before their copies, so a duplicate is always the one skipped.
  for (const file of [...files].sort(bookOrder)) {
    const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    if (!accept.has(ext)) { toast(`${file.name} isn’t an audio file the voice creator can read.`, true); continue; }
    let src;
    try {
      src = await api(`/api/projects/${projectId}/uploads`, { method: "POST", json: { filename: file.name, size: file.size } });
    } catch (err) { fail(err); continue; }
    const up = { loaded: src.received || 0, total: file.size };
    S.uploads.set(src.id, up);
    refresh();
    let offset = src.received || 0, tries = 0, last = 0;  // an unfinished earlier upload carries on
    try {
      while (offset < file.size) {
        const end = Math.min(file.size, offset + CHUNK);
        try {
          const r = await putPart(`/api/projects/${projectId}/uploads/${src.id}?offset=${offset}`, file.slice(offset, end), (n) => {
            up.loaded = offset + n;
            if (Date.now() - last > 500 && S.current === projectId) { last = Date.now(); renderUploadProgress(src.id); }
          });
          offset = r.received;
          tries = 0;
        } catch (err) {
          if (err.received !== undefined) { offset = err.received; continue; }
          if (++tries > 6) throw err;
          await new Promise((res) => setTimeout(res, 2000 * tries));  // a hiccup: try this part again
          try { offset = (await api(`/api/projects/${projectId}/uploads/${src.id}`)).received; } catch {}
        }
      }
      await api(`/api/projects/${projectId}/uploads/${src.id}/done`, { method: "POST" });
    } catch (err) {
      toast(`${file.name} didn’t upload: ${err.message}`, true);
    } finally {
      S.uploads.delete(src.id);
      refresh();
    }
  }
}

function renderUploadProgress(id) {
  if (!S.detail) return;
  const s = S.detail.sources.find((x) => x.id === id);
  if (!s) return;
  const items = [...$("#p-sources").children];
  const i = S.detail.sources.indexOf(s);
  if (items[i]) items[i].replaceWith(sourceRow(s));
}

async function newProject(files) {
  try {
    const name = $("#new-name").value.trim();
    const p = await api("/api/projects", { method: "POST",
      json: { name, accent: S.cfg.default_accent, auto_train: $("#new-auto").checked } });
    $("#new-name").value = "";
    await openProject(p.id);
    if (!name) $("#p-name").focus();
    if (files?.length) await uploadFiles(p.id, files);
  } catch (err) { fail(err); }
}

function wireDrop(zone, onFiles) {
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]);
  });
}

// --------------------------------------------------------------------- wiring
function saveName(immediate = false) {
  clearTimeout(S.nameTimer);
  const run = async () => {
    S.nameTimer = null;
    if (!S.current) return;
    const name = $("#p-name").value.trim();
    if (S.detail && name === (S.detail.name || "")) return;
    try {
      S.detail = await api(`/api/projects/${S.current}`, { method: "PATCH", json: { name } });
      renderProject();
      if (S.detail.voice) toast("Saved. Forge shows the new name from now on.");
    } catch (err) { fail(err); }
  };
  if (immediate) run(); else S.nameTimer = setTimeout(run, 700);
}

function wire() {
  wireDrop($("#drop-new"), (files) => newProject(files));
  $("#pick-new").addEventListener("click", () => $("#file-new").click());
  $("#file-new").addEventListener("change", (e) => { const f = [...e.target.files]; e.target.value = ""; if (f.length) newProject(f); });
  $("#new-empty").addEventListener("click", () => newProject([]));
  wireDrop($("#drop-more"), (files) => uploadFiles(S.current, files));
  $("#pick-more").addEventListener("click", () => $("#file-more").click());
  $("#file-more").addEventListener("change", (e) => { const f = [...e.target.files]; e.target.value = ""; if (f.length) uploadFiles(S.current, f); });
  $("#to-home").addEventListener("click", (e) => { e.preventDefault(); goHome(); });
  window.addEventListener("popstate", () => { const id = location.hash.slice(1); if (id) openProject(id); else goHome(); });

  $("#p-name").addEventListener("input", () => saveName());
  $("#p-name").addEventListener("blur", () => { if (S.nameTimer) saveName(true); });
  $("#p-name").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); saveName(true); } });
  $("#p-accent").replaceChildren(...S.cfg.accents.map((a) => el("option", { value: a.id, textContent: a.label })));
  $("#p-accent").addEventListener("change", async (e) => {
    try { S.detail = await api(`/api/projects/${S.current}`, { method: "PATCH", json: { accent: e.target.value } }); renderProject(); }
    catch (err) { fail(err); refresh(); }
  });
  $("#p-auto").addEventListener("change", async (e) => {
    try { S.detail = await api(`/api/projects/${S.current}`, { method: "PATCH", json: { auto_train: e.target.checked } }); renderProject(); }
    catch (err) { fail(err); }
  });
  $("#p-model").addEventListener("change", async (e) => {
    try {
      S.detail = await api(`/api/projects/${S.current}`, { method: "PATCH", json: { model: e.target.value } });
      e.target.blur();
      renderProject();
    } catch (err) { fail(err); refresh(); }
  });
  $("#cl-make").addEventListener("click", async () => {
    try { S.detail = await api(`/api/projects/${S.current}/clone`, { method: "POST" }); renderProject(); }
    catch (err) { fail(err); }
  });
  $("#stop-listening").addEventListener("click", async () => {
    try { S.detail = await api(`/api/projects/${S.current}/stop-listening`, { method: "POST" }); renderProject(); }
    catch (err) { fail(err); }
  });
  $("#t-start").addEventListener("click", async () => {
    if (!$("#p-name").value.trim()) { $("#p-name").focus(); toast("Name the voice first.", true); return; }
    if (S.nameTimer) saveName(true);
    try { S.detail = await api(`/api/projects/${S.current}/train`, { method: "POST" }); renderProject(); }
    catch (err) { fail(err); }
  });
  $("#t-teach").addEventListener("click", async () => {
    if (!$("#p-name").value.trim()) { $("#p-name").focus(); toast("Name the voice first.", true); return; }
    if (S.nameTimer) saveName(true);
    try { S.detail = await api(`/api/projects/${S.current}/train?from_clone=true`, { method: "POST" }); renderProject(); }
    catch (err) { fail(err); }
  });
  $("#t-stop").addEventListener("click", async () => {
    try { S.detail = await api(`/api/projects/${S.current}/stop`, { method: "POST" }); renderProject(); }
    catch (err) { fail(err); }
  });
  wirePicks();
  $("#v-remove").addEventListener("click", () => S.detail?.voice && removeVoice(S.detail.voice));
  $("#p-remove").addEventListener("click", async () => {
    const d = S.detail;
    const ok = await confirmDialog(`Remove ${d.name ? `“${d.name}”` : "this voice project"}?`,
      "Its recordings, clips and training are deleted from the voice creator." +
      (d.voice ? " The finished voice stays in Forge; remove it there separately if you like." : ""), "Remove project");
    if (!ok) return;
    try { await api(`/api/projects/${d.id}`, { method: "DELETE" }); goHome(); toast("Removed."); } catch (err) { fail(err); }
  });
  for (const btn of document.querySelectorAll("[data-close]")) btn.addEventListener("click", () => btn.closest("dialog").close());

  const themeBtn = $("#theme");
  const syncTheme = () => { themeBtn.textContent = document.documentElement.dataset.theme === "light" ? "Dark theme" : "Light theme"; };
  themeBtn.addEventListener("click", () => {
    const light = document.documentElement.dataset.theme !== "light";
    if (light) document.documentElement.dataset.theme = "light"; else delete document.documentElement.dataset.theme;
    try { localStorage.setItem("theme", light ? "light" : "dark"); } catch {}
    syncTheme();
  });
  syncTheme();
}

function connect() {
  const es = new EventSource(BASE + "/api/events");
  es.addEventListener("changed", () => refresh());
  es.onerror = () => { es.close(); setTimeout(connect, 4000); };
}

(async function init() {
  try {
    S.cfg = await api("/api/config");
  } catch (err) {
    $("#home").replaceChildren(el("p", { class: "empty", textContent: "The voice creator isn’t answering. Reload in a minute." }));
    return;
  }
  wire();
  const id = location.hash.slice(1);
  if (/^[a-f0-9]{12}$/.test(id)) await openProject(id);
  else await refresh();
  connect();
})();
