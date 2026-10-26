"use strict";
// Read along: the book's audio plays and its text follows, word by word.
//
// The server's map (/api/books/<id>/readalong) says when every word is spoken.
// Each animation frame finds the current block and word by binary search, a
// highlight glides to the word, and the page eases so the line being read sits
// about a third of the way down the screen. Progress and checkpoints are saved
// on the server, so a listener can stop on one device and carry on on another;
// each listener (see "listeners" below) has their own.

const $ = (sel, root = document) => root.querySelector(sel);
const bookId = decodeURIComponent(location.pathname.split("/").filter(Boolean).pop());
const audio = $("#r-audio");
const textEl = $("#r-text");
const pill = $("#r-pill");

const SPEEDS = [0.8, 0.9, 1, 1.1, 1.2, 1.35, 1.5, 1.75, 2];  // the speed button steps through these
const SPEED_MIN = 0.5, SPEED_MAX = 3;  // − and + move a tenth at a time between these
const AUTO_CHECKPOINT_SECONDS = 300;
const SIZES = [16, 17, 18, 19, 20, 22, 24, 26, 28];

const R = {
  book: null,
  map: null,
  items: [],          // headings and blocks in reading order
  starts: null,       // Int32Array of item start times (centiseconds)
  chapters: [],       // {idx, title, start, end} in seconds
  cur: -1,            // current item
  word: -1,           // current word in the current item
  pastUpTo: 0,        // items[0 .. pastUpTo-1] carry r-past
  chapterIdx: null,
  jumped: false,      // the last move was a seek, not playback
  follow: true,
  center: false,      // recenter on the next frame, whatever the dead zone says
  pillY: null,
  pillShown: false,
  goal: null,         // scroll position being eased towards
  lastUserScroll: 0,
  raf: 0,
  lastFrame: 0,
  lastUi: 0,
  lastSave: 0,
  listened: 0,        // seconds played since the last automatic checkpoint
  sleep: null,        // {left: ms} or {chapter: idx}
  fading: false,
  marks: [],
  dragging: false,
  listener: null,     // {id, name}: whose place and checkpoints these are
  listeners: [],
  lastSaved: null,    // the position this page last saved for them
};

// ------------------------------------------------------------------ helpers
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
    if (detail && typeof detail === "object") Object.assign(err, detail);  // e.g. .offline
    throw err;
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

let toastTimer;
function toast(text, isError = false) {
  const t = $("#r-toast");
  t.textContent = text;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), isError ? 6000 : 3500);
}

function clock(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

function minutes(sec) {
  const m = Math.round(sec / 60);
  if (m < 1) return "under a minute";
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} },
};

// -------------------------------------------------------------- the text
function addItem(node, ch, t0, t1, text, w, kind) {
  node.dataset.i = R.items.length;
  R.items.push({ el: node, ch, t0, t1, text, w, kind, spans: null });
}

function buildText() {
  const frag = document.createDocumentFragment();
  for (const c of R.map.chapters) {
    const ch = { idx: c.idx, title: c.title, start: c.start / 100, end: c.end / 100 };
    R.chapters.push(ch);
    const section = el("section", { class: "r-ch" });
    const heading = el("h2", { class: "r-h", textContent: c.title });
    section.append(heading);
    if (c.heading) addItem(heading, ch, c.heading[0], c.heading[1], c.title, [0, c.title.length, c.heading[0], c.heading[1]], "h");
    let lastP = null;
    for (const b of c.blocks) {
      const p = el("p", { class: "r-b", textContent: b.text });
      if (b.brk) p.classList.add("r-break");
      else if (lastP !== null && b.p !== lastP) p.classList.add("r-new");
      section.append(p);
      addItem(p, ch, b.t[0], b.t[1], b.text, b.w, "b");
      lastP = b.p;
    }
    frag.append(section);
  }
  textEl.append(frag);
  R.starts = Int32Array.from(R.items, (it) => it.t0);
}

// Last item that has started by time t (centiseconds); -1 before the first.
function itemAt(t) {
  const s = R.starts;
  let lo = 0, hi = s.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (s[mid] <= t) { found = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return found;
}

// Last word of item `it` that has started by time t.
function wordAt(it, t) {
  const w = it.w, n = w.length / 4;
  let lo = 0, hi = n - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (w[mid * 4 + 2] <= t) { found = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return found;
}

function chapterAt(sec) {
  let found = R.chapters[0];
  for (const ch of R.chapters) if (ch.start <= sec + 0.01) found = ch;
  return found;
}

// Wrap the current item's words in spans; other items stay plain text, which
// keeps a 13-hour book's page light.
function wrap(it) {
  const { text, w } = it;
  const frag = document.createDocumentFragment();
  const spans = [];
  let pos = 0;
  for (let k = 0; k < w.length; k += 4) {
    const s = w[k], e = w[k + 1];
    if (s > pos) frag.append(text.slice(pos, s));
    const span = document.createElement("span");
    span.className = "r-w";
    span.dataset.k = spans.length;
    span.textContent = text.slice(s, e);
    frag.append(span);
    spans.push(span);
    pos = e;
  }
  if (pos < text.length) frag.append(text.slice(pos));
  it.el.replaceChildren(frag);
  it.spans = spans;
}

// Which word of a plain (unwrapped) block is under the pointer: the character
// under it, then the word that holds that character.
function wordAtPoint(it, x, y) {
  let node = null, offset = 0;
  if (document.caretPositionFromPoint) {
    const pos = document.caretPositionFromPoint(x, y);
    if (pos) { node = pos.offsetNode; offset = pos.offset; }
  } else if (document.caretRangeFromPoint) {
    const range = document.caretRangeFromPoint(x, y);
    if (range) { node = range.startContainer; offset = range.startOffset; }
  }
  if (!node || !it.el.contains(node)) return 0;
  const w = it.w;
  let found = 0;
  for (let k = 0; k < w.length / 4; k++) if (w[k * 4] <= offset) found = k;
  return found;
}

function unwrap(it) {
  it.el.textContent = it.text;
  it.spans = null;
}

function activate(i) {
  const prev = R.items[R.cur];
  if (prev) {
    prev.el.classList.remove("r-on");
    unwrap(prev);
  }
  // Everything before the current item fades back.
  const past = Math.max(0, i);
  if (past > R.pastUpTo) for (let j = R.pastUpTo; j < past; j++) R.items[j].el.classList.add("r-past");
  else for (let j = past; j < R.pastUpTo; j++) R.items[j].el.classList.remove("r-past");
  R.pastUpTo = past;
  R.cur = i;
  R.word = -1;
  const it = R.items[i];
  if (!it) { hidePill(); return; }
  it.el.classList.add("r-on");
  wrap(it);
  textEl.classList.add("r-live");
  if (R.chapterIdx !== it.ch.idx) {
    const first = R.chapterIdx === null;
    R.chapterIdx = it.ch.idx;
    $("#r-chapter").textContent = it.ch.title;
    updateMediaMeta();
    if (!first && !R.jumped && !audio.paused) addCheckpoint("chapter");
    if (R.sleep?.chapter !== undefined && R.sleep.chapter !== it.ch.idx && !R.jumped) sleepNow();
  }
}

function setWord(k) {
  const it = R.items[R.cur];
  if (!it || !it.spans || k === R.word) return;
  const spans = it.spans;
  if (k > R.word) for (let j = Math.max(0, R.word); j < k; j++) spans[j].classList.add("r-read");
  else for (let j = Math.max(0, k); j <= R.word && j < spans.length; j++) spans[j].classList.remove("r-read");
  if (R.word >= 0 && spans[R.word]) spans[R.word].classList.remove("r-cur");
  R.word = k;
  if (k < 0 || !spans[k]) { hidePill(); return; }
  spans[k].classList.add("r-cur");
  spans[k].classList.remove("r-read");
  movePill(spans[k]);
}

// A word that breaks across lines ("pink-" / "frosted") has a box on each
// line; the second piece of highlight covers the part on the next line.
function placeTail(rects, box, padX, padY) {
  const tail = $("#r-pill-tail");
  if (rects.length < 2) { tail.classList.remove("r-show"); return; }
  const last = rects[rects.length - 1];
  tail.style.transform = `translate(${(last.left - box.left - padX).toFixed(1)}px, ${(last.top - box.top - padY).toFixed(1)}px)`;
  tail.style.width = `${(last.width + padX * 2).toFixed(1)}px`;
  tail.style.height = `${(last.height + padY * 2).toFixed(1)}px`;
  tail.classList.add("r-show");
}

// The highlight: slides along a line, jumps (with a quick fade) to a new one.
function movePill(span) {
  const rects = [...span.getClientRects()].filter((r) => r.width > 0);
  const rect = rects[0];
  if (!rect) return;
  const box = textEl.getBoundingClientRect();
  const em = parseFloat(getComputedStyle(textEl).fontSize);
  const padX = em * 0.2, padY = em * 0.1;
  placeTail(rects, box, padX, padY);
  const x = rect.left - box.left - padX;
  const y = rect.top - box.top - padY;
  const snap = !R.pillShown || R.pillY === null || Math.abs(y - R.pillY) > em * 0.5;
  if (snap) pill.classList.add("r-snap");
  pill.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
  pill.style.width = `${(rect.width + padX * 2).toFixed(1)}px`;
  pill.style.height = `${(rect.height + padY * 2).toFixed(1)}px`;
  if (snap) {
    void pill.offsetWidth;  // commit the jump before turning the glide back on
    pill.classList.remove("r-snap");
  }
  pill.classList.add("r-show");
  R.pillShown = true;
  R.pillY = y;
}

function hidePill() {
  pill.classList.remove("r-show");
  $("#r-pill-tail").classList.remove("r-show");
  R.pillShown = false;
  R.pillY = null;
}

function refreshPill() {
  const it = R.items[R.cur];
  if (it?.spans?.[R.word]) { R.pillShown = false; movePill(it.spans[R.word]); }
}

// ---------------------------------------------------------- following
function viewBand() {
  const top = $(".r-top").offsetHeight;
  const bottom = innerHeight - $("#r-player").offsetHeight;
  return { top, height: Math.max(120, bottom - top) };
}

// Keep the line being read at a fixed height, a little above the middle, like
// a teleprompter: when the reading moves to the next line the page glides up
// by one line. Words along a line don't move the page at all.
function aim() {
  if (!R.follow) return;
  const it = R.items[R.cur];
  if (!it) return;
  const target = (it.spans?.[R.word]) || it.el;
  const rect = target.getClientRects()[0] || target.getBoundingClientRect();
  const band = viewBand();
  const y = rect.top + rect.height / 2;
  const want = band.top + band.height * 0.38;
  const slack = rect.height * 0.6;  // less than a line: any line change re-aims
  if (R.center || Math.abs(y - want) > slack) {
    const max = document.documentElement.scrollHeight - innerHeight;
    R.goal = Math.max(0, Math.min(max, scrollY + (y - want)));
    R.center = false;
  }
}

function scrollStep(dt) {
  if (R.goal === null || !R.follow) return false;
  const diff = R.goal - scrollY;
  if (Math.abs(diff) < 0.6) { R.goal = null; return false; }
  // A far jump (resume, a checkpoint, a seek) goes straight there: pages of
  // text flying past are hard to watch and tell the reader nothing.
  if (Math.abs(diff) > innerHeight * 1.5) { window.scrollTo(0, R.goal); R.goal = null; return false; }
  // A long jump (a seek) moves fast; a line change glides in about a third of a second.
  const speed = Math.abs(diff) > innerHeight ? 14 : 7;
  const step = diff * (1 - Math.exp(-dt * speed));
  window.scrollTo(0, scrollY + (Math.abs(step) < 0.6 ? Math.sign(diff) * 0.6 : step));
  return true;
}

function stopFollowing() {
  R.follow = false;
  R.goal = null;
  const it = R.items[R.cur];
  if (!it) return;
  // Point the way back: up if the listener scrolled past the reading spot.
  const above = it.el.getBoundingClientRect().bottom < viewBand().top;
  $("#r-follow").classList.toggle("r-up", above);
  $("#r-follow").hidden = false;
}

function resumeFollowing() {
  R.follow = true;
  R.center = true;
  $("#r-follow").hidden = true;
  kick();
}

// ---------------------------------------------------------------- the loop
function tick() {
  if (!R.starts || R.textOnly) return;  // no map yet, or no audio to follow
  const t = Math.round(audio.currentTime * 100);
  const i = itemAt(t);
  if (i !== R.cur) activate(i);
  const it = R.items[i];
  if (it) setWord(wordAt(it, t));
  R.jumped = false;
}

function frame(now) {
  R.raf = 0;
  const dt = Math.min(0.1, R.lastFrame ? (now - R.lastFrame) / 1000 : 0.016);
  R.lastFrame = now;
  tick();
  aim();
  const scrolling = scrollStep(dt);
  if (!audio.paused) {
    R.listened += dt * audio.playbackRate;
    if (R.listened >= AUTO_CHECKPOINT_SECONDS) { R.listened = 0; addCheckpoint("auto"); }
    if (R.sleep?.left !== undefined) {
      R.sleep.left -= dt * 1000;
      if (R.sleep.left <= 0) sleepNow();
    }
    // Scrolled away and forgot? Pick the reading spot back up after a while.
    if (!R.follow && now - R.lastUserScroll > 20000) resumeFollowing();
  }
  if (now - R.lastUi > 250) { R.lastUi = now; updatePlayerUi(); }
  if (!audio.paused && now - R.lastSave > 5000) { R.lastSave = now; saveProgress(); }
  if (!audio.paused || scrolling || R.fading) R.raf = requestAnimationFrame(frame);
  else R.lastFrame = 0;
}

function kick() {
  if (!R.raf) R.raf = requestAnimationFrame(frame);
}

// --------------------------------------------------------------- player
function duration() {
  return isFinite(audio.duration) && audio.duration > 0 ? audio.duration : (R.map?.duration || 0);
}

// quiet: the page moving itself (resuming), not the listener choosing a place.
function seekTo(sec, { center = true, quiet = false } = {}) {
  if (!quiet) R.touched = true;
  audio.currentTime = Math.max(0, Math.min(duration() - 0.2, sec));
  R.jumped = true;
  if (center) { R.follow = true; R.center = true; $("#r-follow").hidden = true; }
  tick();
  aim();
  kick();
  updatePlayerUi();
  clearTimeout(R.saveSoon);
  if (!quiet) R.saveSoon = setTimeout(saveProgress, 800);
}

function togglePlay() {
  if (audio.paused) {
    $("#r-resume").hidden = true;
    audio.play().catch((err) => toast(`The audio didn’t start: ${err.message}`, true));
  } else audio.pause();
}

function updatePlayerUi() {
  const t = audio.currentTime, total = duration();
  if (!R.dragging) {
    $("#r-seek").max = total || 1;
    $("#r-seek").value = t;
  }
  $("#r-fill").style.width = `${total ? (100 * t) / total : 0}%`;
  $("#r-now").textContent = clock(t);
  // Time left is listening time: at 1.5× an hour of book takes 40 minutes.
  const rate = R.speed || 1;
  $("#r-total").textContent = `-${clock((total - t) / rate)}`;
  const ch = chapterAt(t);
  $("#r-ch-left").textContent = ch ? `${minutes(Math.max(0, ch.end - t) / rate)} left in this chapter` : "";
  if (R.sleep?.left !== undefined) $("#r-sleep-label").textContent = clock(R.sleep.left / 1000);
  if ("mediaSession" in navigator && total && !audio.paused) {
    try { navigator.mediaSession.setPositionState({ duration: total, playbackRate: audio.playbackRate, position: Math.min(t, total) }); } catch {}
  }
}

function drawTicks() {
  const total = duration();
  const ticks = $("#r-ticks");
  ticks.replaceChildren();
  if (!total || R.chapters.length > 120) return;
  for (const ch of R.chapters.slice(1)) ticks.append(el("i", { style: `left:${(100 * ch.start) / total}%` }));
}

function setSpeed(rate) {
  rate = Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, Number(rate) || 1)) * 100) / 100;
  R.speed = rate;
  // Loading audio (the book, or another voice of it) resets playbackRate to
  // defaultPlaybackRate, so both are set; "play" and "loadedmetadata" put it back too.
  audio.defaultPlaybackRate = rate;
  audio.playbackRate = rate;
  $("#r-speed").textContent = `${rate}×`;
  $("#r-slower").disabled = rate <= SPEED_MIN;
  $("#r-faster").disabled = rate >= SPEED_MAX;
  store.set("readSpeed", rate);
  if (R.map) updatePlayerUi();  // the time left changes with it
}

function keepSpeed() {
  if (R.speed && audio.playbackRate !== R.speed) audio.playbackRate = R.speed;
}

// A tenth faster or slower, from the tenth the speed is at or just past (1.35 goes to 1.4 or 1.3).
function stepSpeed(dir) {
  const tenths = R.speed * 10;
  setSpeed((dir > 0 ? Math.floor(tenths + 1e-6) + 1 : Math.ceil(tenths - 1e-6) - 1) / 10);
}

function goChapter(step) {
  const t = audio.currentTime;
  const n = R.chapters.indexOf(chapterAt(t));
  if (step < 0 && t - R.chapters[n].start > 3) return seekTo(R.chapters[n].start);
  const next = R.chapters[Math.max(0, Math.min(R.chapters.length - 1, n + step))];
  seekTo(next.start);
}

// ----------------------------------------------------------- sleep timer
function setSleep(choice) {
  $("#r-sleep-menu").hidden = true;
  if (choice === "off") {
    R.sleep = null;
    $("#r-sleep").classList.remove("r-armed");
    $("#r-sleep-label").textContent = "Sleep";
    toast("Sleep timer off");
    return;
  }
  if (choice === "chapter") {
    R.sleep = { chapter: chapterAt(audio.currentTime).idx };
    $("#r-sleep-label").textContent = "Chapter";
  } else {
    R.sleep = { left: Number(choice) * 60000 };
    $("#r-sleep-label").textContent = clock(R.sleep.left / 1000);
  }
  $("#r-sleep").classList.add("r-armed");
  addCheckpoint("sleep");
  toast(choice === "chapter" ? "Playback stops at the end of this chapter." : `Playback stops in ${choice} minutes.`);
}

// Fade out over a few seconds rather than cutting off mid-sentence.
function sleepNow() {
  if (R.fading) return;
  R.sleep = null;
  R.fading = true;
  $("#r-sleep").classList.remove("r-armed");
  $("#r-sleep-label").textContent = "Sleep";
  const start = performance.now(), from = audio.volume;
  const step = (now) => {
    const k = Math.min(1, (now - start) / 6000);
    audio.volume = from * (1 - k);
    if (k < 1 && !audio.paused) return requestAnimationFrame(step);
    audio.pause();
    audio.volume = from;
    R.fading = false;
    saveProgress();
  };
  requestAnimationFrame(step);
}

// ---------------------------------------------------- progress, checkpoints
// A place is kept as chapter + word number in it: a new voice or a rebuild
// changes every time in the book, but not the words, so the place survives.
function wordInChapter(sec) {
  const t = Math.round(sec * 100);
  const i = itemAt(t);
  if (i < 0) return 0;
  const it = R.items[i];
  let n = 0;
  for (let j = i - 1; j >= 0 && R.items[j].ch === it.ch; j--) n += R.items[j].w.length / 4;
  return n + Math.max(0, wordAt(it, t));
}

function timeOfWord(chapterIdx, n) {
  for (const it of R.items) {
    if (it.ch.idx !== chapterIdx) continue;
    const count = it.w.length / 4;
    if (n < count) return it.w[n * 4 + 2] / 100;
    n -= count;
  }
  return null;
}

function spot(sec = audio.currentTime) {
  const ch = chapterAt(sec);
  return { chapter: ch.idx, offset: Math.max(0, sec - ch.start), position: sec, word: wordInChapter(sec) };
}

function positionOf(saved) {
  const ch = R.chapters.find((c) => c.idx === saved.chapter);
  if (!ch) return Math.min(saved.position, duration());
  if (saved.word !== null && saved.word !== undefined) {
    const at = timeOfWord(ch.idx, saved.word);
    if (at !== null) return at;
  }
  return Math.min(ch.end - 0.5, ch.start + saved.offset);
}

// Only a page that has been listened on saves: one opened and closed without
// playing must not overwrite the place reached on another device.
function saveProgress(leaving = false) {
  if (!R.map || !R.touched || !(audio.currentTime > 0) || nobody()) return;
  const place = spot(), listener = R.listener;
  // Already saved from here: saving it again (as a paused tab closes, say)
  // could only undo listening done on another device since.
  if (R.lastSaved !== null && Math.abs(place.position - R.lastSaved) < 0.5) return;
  fetch(`/api/books/${bookId}/progress${who()}`, {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(place), keepalive: leaving,
  }).then((res) => {
    R.savedAt = Date.now() / 1000;
    if (res.ok && listener === R.listener) R.lastSaved = place.position;
  }).catch(() => {});
}

function showResume(at, updated, elsewhere) {
  const when = new Date(updated * 1000);
  const ch = chapterAt(at);
  $(".r-resume-title").textContent = elsewhere ? "Picked up from your other device" : "Pick up where you left off";
  $("#r-resume-where").textContent =
    `${ch.title}, ${clock(at - ch.start)} in. Last listened ${dayLabel(when).toLowerCase()} at ${when.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}.`;
  $("#r-resume").hidden = false;
}

// Back on a paused page (switching from the phone to the tablet, say): if the
// place moved on elsewhere meanwhile, go there instead of offering an old one.
async function catchUp() {
  // Not before this device's listener is known and their place has loaded.
  if (!R.ready || !audio.paused || document.hidden || $("#r-who").open) return;
  const listener = R.listener;
  const saved = await api(`/api/books/${bookId}/progress${who()}`).catch((err) => {
    if (err.listener_gone) listenerGone();
    return null;
  });
  if (listener !== R.listener) return;  // switched meanwhile: that's someone else's place
  if (!saved || saved.updated <= (R.savedAt || 0) + 1) return;
  R.savedAt = saved.updated;
  const at = positionOf(saved);
  if (Math.abs(at - audio.currentTime) < 3) return;
  seekTo(at, { quiet: true });
  showResume(at, saved.updated, true);
}

async function addCheckpoint(kind) {
  if (!R.map || nobody()) return;
  const body = { ...spot(), kind };
  try {
    await api(`/api/books/${bookId}/checkpoints${who()}`, { method: "POST", json: body });
    if (kind === "manual") toast("Checkpoint added");
    if ($("#r-panel").open) loadMarks();
  } catch (err) {
    if (err.listener_gone) listenerGone();
    else if (kind === "manual") toast(err.message, true);
  }
}

// When the sentence playing at `sec` began: going back to a checkpoint starts
// the sentence over, not mid-phrase.
function sentenceStart(sec) {
  const t = Math.round(sec * 100);
  const it = R.items[itemAt(t)];
  if (!it) return sec;
  const k = Math.max(0, wordAt(it, t));
  const m = it.text.slice(0, it.w[k * 4] || 0).match(/^[\s\S]*[.!?…]["”’)]*\s+/);
  const from = m ? m[0].length : 0;
  for (let j = 0; j < it.w.length / 4; j++) if (it.w[j * 4] >= from) return it.w[j * 4 + 2] / 100;
  return sec;
}

// The text being read at `sec`, from the start of its sentence: what a
// listener would recognise as "the last bit I remember". A short ending runs
// on into the next block so there's always enough to go on.
function snippet(sec, length = 150) {
  const t = Math.round(sec * 100);
  const i = itemAt(t);
  const it = R.items[i];
  if (!it) return "";
  const k = Math.max(0, wordAt(it, t));
  const at = it.w[k * 4] || 0;
  const before = it.text.slice(0, at);
  const m = before.match(/^[\s\S]*[.!?…]["”’)]*\s+/);
  let text = it.text.slice(m ? m[0].length : 0);
  if (text.length < 90 && R.items[i + 1]) text += ` ${R.items[i + 1].text}`;
  return text.length > length ? `${text.slice(0, length).replace(/\s+\S*$/, "")}…` : text;
}

const KIND_LABELS = { auto: "While listening", chapter: "New chapter", sleep: "Sleep timer set", manual: "Your checkpoint" };

function dayLabel(date) {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const day = new Date(date); day.setHours(0, 0, 0, 0);
  const diff = Math.round((today - day) / 86400000);
  if (diff === 0) return "Today";
  if (diff === 1) return "Yesterday";
  return day.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
}

async function loadMarks() {
  const listener = R.listener;
  let marks = [];
  try { marks = await api(`/api/books/${bookId}/checkpoints${who()}`); } catch {}
  if (listener !== R.listener) return;  // switched meanwhile; the new listener's are loading
  R.marks = marks;
  const list = $("#r-mark-list");
  list.replaceChildren();
  if (!R.marks.length) {
    list.append(el("li", { class: "r-mark-empty", textContent: "No checkpoints yet. They appear as you listen." }));
    return;
  }
  let lastDay = null;
  for (const m of R.marks) {
    const when = new Date(m.created * 1000);
    const day = dayLabel(when);
    if (day !== lastDay) { list.append(el("li", { class: "r-day", textContent: day })); lastDay = day; }
    const sec = positionOf(m);
    const ch = chapterAt(sec);
    const remove = el("button", { type: "button", class: "quiet", textContent: "Remove" });
    const row = el("li", { class: "r-mark", tabIndex: 0 },
      el("div", {},
        el("div", { class: "r-mark-time", textContent: when.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }) }),
        el("div", { class: "r-mark-kind", textContent: KIND_LABELS[m.kind] || "" })),
      el("div", {},
        el("p", { class: "r-mark-where", textContent: `${ch?.title || ""}, ${clock(sec - (ch?.start || 0))} in` }),
        el("p", { class: "r-mark-snip", textContent: snippet(sec) })),
      remove);
    const go = () => {
      $("#r-panel").close();
      const at = sentenceStart(sec);
      seekTo(at);
      toast(`Back to ${ch?.title || ""}, ${clock(at - (ch?.start || 0))} in`);
    };
    row.addEventListener("click", (e) => { if (e.target !== remove) go(); });
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    remove.addEventListener("click", async () => {
      await api(`/api/books/${bookId}/checkpoints/${m.id}${who()}`, { method: "DELETE" }).catch(() => {});
      loadMarks();
    });
    list.append(row);
  }
}

// --------------------------------------------------------------- listeners
// Each person who reads along has their own place and checkpoints, kept on the
// server so they follow that person from device to device. The device
// remembers who was picked; with only one listener nobody is asked anything.

// ?listener=<id> on every progress and checkpoint call. Without one (the
// list couldn't load) the server uses its default listener, as it did before.
function who() {
  return R.listener ? `?listener=${R.listener.id}` : "";
}

// Several listeners and nobody picked yet: there's no one to save a place for.
const nobody = () => R.listeners.length > 1 && !R.listener;

const initial = (name) => ([...name.trim()][0] || "?").toLocaleUpperCase();

function useListener(l, remember) {
  R.listener = l;
  if (remember) store.set("listener", l.id);
  const btn = $("#r-who-btn");
  btn.hidden = false;
  $("#r-who-initial").textContent = initial(l.name);
  $("#r-who-name").textContent = l.name;
  btn.setAttribute("aria-label", `Listening: ${l.name}. Tap to change who’s listening.`);
}

// Before the first place loads: this device's listener, or the only one, or ask.
async function startListener() {
  const list = await api("/api/listeners").catch(() => null);
  if (!list?.length) return;
  R.listeners = list;
  const remembered = list.find((l) => l.id === store.get("listener", null));
  if (remembered || list.length === 1) return useListener(remembered || list[0], false);
  let l = null;
  while (!l) l = await openWho(true);  // even if the sheet was closed: someone has to be picked
  useListener(l, true);
}

function renderWho() {
  $("#r-who-list").replaceChildren(...R.listeners.map((l) => {
    const current = l.id === R.listener?.id;
    const btn = el("button", { type: "button", class: `r-who-pick${current ? " current" : ""}` },
      el("span", { class: "r-avatar", textContent: initial(l.name), "aria-hidden": "true" }),
      el("span", { class: "r-who-label", textContent: l.name }),
      current ? el("span", { class: "r-who-now", textContent: "Listening now" }) : "");
    btn.addEventListener("click", () => whoChosen(l));
    return el("li", {}, btn);
  }));
}

// "Who's listening?" Resolves with the listener picked or added, or null if
// closed. required: the first open on a device with several listeners.
function openWho(required = false) {
  const dlg = $("#r-who");
  dlg.dataset.required = required ? "1" : "";
  $("#r-who-close").hidden = required;
  $("#r-who-add").hidden = true;
  $("#r-who-new").hidden = false;
  $("#r-who-error").hidden = true;
  renderWho();
  if (!dlg.open) dlg.showModal();
  // Someone may have been added on another device since.
  api("/api/listeners").then((list) => { if (list.length) { R.listeners = list; renderWho(); } }).catch(() => {});
  return new Promise((resolve) => { R.whoDone = resolve; });
}

function whoChosen(l) {
  const done = R.whoDone;
  R.whoDone = null;
  $("#r-who").close();
  done?.(l);
}

// Someone else picks up the device: the last person's place is saved for them,
// then the new person's place and checkpoints load.
async function switchListener(l) {
  if (!l) return;
  store.set("listener", l.id);
  if (l.id === R.listener?.id) return;
  clearTimeout(R.saveSoon);
  saveProgress();
  R.touched = false;  // the new listener hasn't listened here yet (this also quiets the pause below)
  audio.pause();
  R.sleep = null;     // a timer set for one person mustn't fade out the next
  $("#r-sleep").classList.remove("r-armed");
  $("#r-sleep-label").textContent = "Sleep";
  R.listened = 0;     // their next automatic checkpoint is 5 minutes into their listening
  R.savedAt = 0;
  R.lastSaved = null;
  $("#r-resume").hidden = true;
  useListener(l, true);
  const saved = await api(`/api/books/${bookId}/progress${who()}`).catch(() => null);
  if (R.listener !== l) return;
  R.savedAt = saved?.updated || 0;
  const at = saved ? positionOf(saved) : 0;
  seekTo(at > 3 ? at : 0, { quiet: true });
  if (at > 3) showResume(at, saved.updated, false);
  toast(at > 3 ? `Listening as ${l.name}` : `Listening as ${l.name}, from the start`);
  if ($("#r-panel").open) loadMarks();
}

// This device's listener was removed, on another device perhaps: pick again.
async function listenerGone() {
  if (R.regone) return;
  R.regone = true;
  R.touched = false;  // nothing of theirs left to save
  const list = await api("/api/listeners").catch(() => null);
  if (list?.length) {
    R.listeners = list;
    let l = list.length === 1 ? list[0] : null;
    while (!l) l = await openWho(true);
    await switchListener(l);
  }
  R.regone = false;
}

function wireListeners() {
  const dlg = $("#r-who");
  $("#r-who-btn").addEventListener("click", async () => switchListener(await openWho()));
  dlg.addEventListener("cancel", (e) => { if (dlg.dataset.required) e.preventDefault(); });
  dlg.addEventListener("close", () => {
    const done = R.whoDone;
    R.whoDone = null;
    done?.(null);
  });
  $("#r-who-new").addEventListener("click", () => {
    $("#r-who-new").hidden = true;
    $("#r-who-add").hidden = false;
    $("#r-who-input").value = "";
    $("#r-who-input").focus();
  });
  $("#r-who-input").addEventListener("input", () => { $("#r-who-error").hidden = true; });
  $("#r-who-add").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#r-who-input").value.trim();
    const add = $("#r-who-add button");
    if (!name || add.disabled) return;
    add.disabled = true;
    try {
      const l = await api("/api/listeners", { method: "POST", json: { name } });
      R.listeners.push(l);
      whoChosen(l);
    } catch (err) {
      $("#r-who-error").textContent = err.message;
      $("#r-who-error").hidden = false;
    }
    add.disabled = false;
  });
}

// ----------------------------------------------------------- media session
function updateMediaMeta() {
  if (!("mediaSession" in navigator) || !R.book) return;
  const ch = R.chapters.find((c) => c.idx === R.chapterIdx);
  navigator.mediaSession.metadata = new MediaMetadata({
    title: R.book.meta.title || R.book.filename,
    artist: R.book.meta.author || "",
    album: ch?.title || "",
    artwork: [{ src: `/api/books/${bookId}/cover`, sizes: "480x480", type: "image/jpeg" }],
  });
}

function setupMediaSession() {
  if (!("mediaSession" in navigator)) return;
  const ms = navigator.mediaSession;
  const handlers = {
    play: () => audio.play(),
    pause: () => audio.pause(),
    seekbackward: () => seekTo(audio.currentTime - 15),
    seekforward: () => seekTo(audio.currentTime + 30),
    seekto: (d) => seekTo(d.seekTime),
    previoustrack: () => goChapter(-1),
    nexttrack: () => goChapter(1),
  };
  for (const [action, fn] of Object.entries(handlers)) {
    try { ms.setActionHandler(action, fn); } catch {}
  }
}

// ----------------------------------------------------------- voice change
// Clips play on their own audio element; the book pauses for them and carries
// on after.
const clip = new Audio();
let clipBtn = null, resumeAfterClip = false;
let clipToken = 0;  // bumps on every stop, so a clip still being made for an earlier tap is dropped
const ICONS = {
  play: '<svg class="i-play" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>',
  stop: '<svg class="i-stop" viewBox="0 0 16 16" aria-hidden="true"><rect x="3.5" y="3.5" width="9" height="9" rx="1"/></svg>',
};

// As in app.js: asked with progress=1, the server answers 202 with how far
// along a clip is until it's made, which fills a bar under its button.
// Each stage: [from, to, seconds it usually takes].
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
        throw new Error("The clip is taking too long. The voice worker may be busy or stopped: try again in a minute.");
      }
      await new Promise((resolve) => setTimeout(resolve, SAMPLE_POLL_MS));
      if (!current()) return null;
    }
  } finally {
    clearInterval(timer);
    for (const [attr, value] of [["title", was.title], ["aria-label", was.label]]) {
      if (value === null) btn.removeAttribute(attr); else btn.setAttribute(attr, value);
    }
    setTimeout(() => { if (btn.dataset.build === build) btn.classList.remove("building"); }, timer ? 300 : 0);
  }
}

function stopClip() {
  clipToken++;
  clip.pause();
  clipBtn?.classList.remove("on", "loading");
  clipBtn = null;
  if (resumeAfterClip) { resumeAfterClip = false; audio.play().catch(() => {}); }
}
clip.addEventListener("ended", stopClip);

async function playClip(btn, url) {
  if (clipBtn === btn) { stopClip(); return; }
  stopClip();
  const token = clipToken;
  clipBtn = btn;
  btn.classList.add("loading");
  if (!audio.paused) { resumeAfterClip = true; audio.pause(); }
  try {
    const blob = await fetchSample(url, btn, () => token === clipToken);
    if (!blob || token !== clipToken) return;
    clip.src = URL.createObjectURL(blob);
    await clip.play();
    btn.classList.remove("loading");
    btn.classList.add("on");
  } catch (err) {
    if (token === clipToken) { stopClip(); toast(err.message || "That clip couldn’t play.", true); }
  }
}

function voiceLabel(voiceId, moodId) {
  const v = R.cfg?.voices.find((x) => x.id === voiceId);
  const m = v?.moods.find((x) => x.id === moodId);
  return v ? v.label + (m ? `, ${m.label.toLowerCase()}` : "") : voiceId;
}

async function openVoices() {
  if (!R.cfg) R.cfg = await api("/api/config");
  $("#r-voice .make-voice").hidden = !R.cfg.voicelab;
  const list = $("#r-voice-list");
  list.replaceChildren();
  $("#r-voice-offline").hidden = true;
  const order = (v) => (v.group === "Your voices" ? -1 : v.group.startsWith("Kokoro") ? 0 : 1);  // your own voices first
  // Voices the book is already made in (at this speed, from today's text) switch at once.
  const made = (R.book.versions || []).filter((x) => !x.outdated && x.speed === (R.book.speed || 1));
  const playing = made.find((x) => x.current);
  const ready = [];
  const others = [];
  for (const v of [...R.cfg.voices].sort((a, b) => order(a) - order(b))) {
    for (const m of v.moods.length ? v.moods : [null]) {
      const copy = made.find((x) => x.voice === v.id && (x.mood || null) === (m?.id || null));
      const current = playing ? copy === playing : v.id === R.book.voice && (!m || m.id === R.book.mood);
      // The voice's saved sample: the same 10-second passage in every voice, made ahead of time.
      const query = new URLSearchParams({ voice: v.id, seconds: 10, speed: 1, ...(m ? { mood: m.id } : {}) });
      const play = el("button", { type: "button", class: "icon play", "aria-label": `Play ${v.label}${m ? `, ${m.label}` : ""}` });
      play.innerHTML = ICONS.play + ICONS.stop;
      const idle = current ? "Current voice" : copy ? "Switch now" : "Make this voice";
      const use = el("button", { type: "button", class: copy && !current ? "primary" : "quiet", textContent: idle, disabled: current });
      use.dataset.idle = idle;
      const name = copy?.name || v.label + (m ? `: ${m.label}` : "");
      const meta = [copy?.name ? v.label + (m ? `, ${m.label.toLowerCase()}` : "") : v.group + (m ? `, ${m.label.toLowerCase()} mood` : ""),
                    copy && !current ? "made, switches instantly" : null].filter(Boolean).join(" · ");
      const li = el("li", { class: (current ? "current" : "") + (copy ? " made" : "") }, play,
        el("div", {}, el("p", { class: "v-name", textContent: name }), el("p", { class: "v-meta", textContent: meta })),
        use);
      (copy ? ready : others).push(li);
      play.addEventListener("click", () => playClip(play, `/api/preview?${query}`));
      use.addEventListener("click", () => chooseVoice(v, m, use, false, !!copy));
    }
  }
  list.append(...ready, ...others);  // the ones it can switch to right away come first
  $("#r-voice").showModal();
}

// A voice the book is made in switches with one tap. Making a new one takes two:
// the first says what will happen, the second does it.
async function chooseVoice(v, m, btn, here = false, made = false) {
  if (!here && !made && btn.dataset.armed !== "1") {
    for (const b of document.querySelectorAll("#r-voice-list [data-armed='1']")) { b.dataset.armed = ""; b.textContent = b.dataset.idle; }
    btn.dataset.armed = "1";
    btn.textContent = "Tap again to voice the book";
    return;
  }
  const label = v.label + (m ? `, ${m.label.toLowerCase()}` : "");
  btn.disabled = true;
  btn.textContent = made ? "Switching" : "Starting";
  try {
    const r = await api(`/api/books/${bookId}/revoice`, { method: "POST", json: { voice: v.id, mood: m?.id || null, here } });
    if (r.where === "ready") {  // already made: same words, its own times, carry on from here
      stopClip();
      $("#r-voice").close();
      R.book = await api(`/api/books/${bookId}`);
      await swapToNewVersion(R.book.versions.find((x) => x.current)?.label || label);
      return;
    }
    R.revoice = { voice: v.id, mood: m?.id || null, label };
    stopClip();
    $("#r-voice").close();
    showRevoice(`Voicing the book in ${label}${r.where !== "here" ? ` on ${r.where}` : ""}. Keep listening; it switches over when it’s ready.`, 0);
    pollRevoice();
  } catch (err) {
    btn.disabled = false;
    btn.dataset.armed = "";
    btn.textContent = btn.dataset.idle;
    if (err.offline) {
      $("#r-voice-offline-text").textContent = err.message;
      $("#r-voice-offline").hidden = false;
      R.offline = [v, m, btn];
    } else toast(err.message, true);
  }
}

function showRevoice(text, fraction) {
  $("#r-revoice").hidden = false;
  $("#r-revoice-text").textContent = text;
  if (fraction !== null) $("#r-revoice-fill").style.width = `${Math.round(fraction * 100)}%`;
}

async function loadMap() {
  const res = await fetch(`/api/books/${bookId}/readalong`, { cache: "no-store" });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return { map: await res.json(), etag: res.headers.get("etag") };
}

async function mapChanged() {
  const res = await fetch(`/api/books/${bookId}/readalong`, { cache: "no-store", headers: { "If-None-Match": R.etag || "" } });
  return res.status === 200;
}

const ACTIVE = ["queued", "voicing", "assembling", "checking"];

async function pollRevoice() {
  clearTimeout(R.revoiceTimer);
  const target = R.revoice;
  if (!target) return;
  try {
    const [s, book] = await Promise.all([api(`/api/books/${bookId}/revoice`), api(`/api/books/${bookId}`)]);
    const arrived = book.voice === target.voice && (book.mood || null) === (target.mood || null)
      && book.status === "done" && book.readalong;
    if (arrived && await mapChanged()) { R.book = book; await swapToNewVersion(target.label); return; }
    if (s.status === "failed") {
      showRevoice(`Voicing in ${target.label} failed: ${s.message || "see the book list"}`, null);
      R.revoice = null;
      return;
    }
    if (s.status === "unreachable") showRevoice(`Waiting for ${s.where} to answer…`, null);
    else if (s.status === "queued") showRevoice(`Voicing in ${target.label}: waiting for a worker`, 0);
    else if (s.status === "voicing") {
      const p = Math.min(1, (s.progress || 0) / 0.9);
      showRevoice(`Voicing in ${target.label}${s.where !== "here" ? ` on ${s.where}` : ""}: ${Math.round(p * 100)}%`, p);
    } else if (s.status === "assembling" || s.status === "checking") showRevoice(`Voicing in ${target.label}: finishing up`, 0.97);
    else if (s.status === "done") showRevoice(`Voiced in ${target.label}. Bringing it here…`, 1);
  } catch {}
  R.revoiceTimer = setTimeout(pollRevoice, 3000);
}

// The new voice is ready: same words, new times. Carry on from the same word.
async function swapToNewVersion(label) {
  const anchor = spot();
  const wasPlaying = !audio.paused;
  const { map, etag } = await loadMap();
  audio.pause();
  for (const node of textEl.querySelectorAll(".r-ch")) node.remove();
  hidePill();
  Object.assign(R, { map, etag, items: [], chapters: [], cur: -1, word: -1, pastUpTo: 0, chapterIdx: null, revoice: null });
  buildText();
  const at = positionOf(anchor);
  audio.src = `/api/books/${bookId}/audio?v=${Date.now()}`;
  audio.addEventListener("loadedmetadata", () => {
    seekTo(at);
    drawTicks();
    if (wasPlaying) audio.play().catch(() => {});
  }, { once: true });
  updateMediaMeta();
  $("#r-revoice").hidden = true;
  toast(`Now reading in ${label}`);
}

async function checkRevoice() {
  if (R.book.kind === "audio") return;  // an audiobook added for Read along keeps its own voice
  if (!R.book.origin && !ACTIVE.includes(R.book.status)) return;
  const s = await api(`/api/books/${bookId}/revoice`).catch(() => null);
  if (!s || !ACTIVE.includes(s.status)) return;
  // The PC rebuilding what's already playing here isn't news; a voice not made here yet (pending) is.
  if (s.where !== "here" && !R.book.pending && s.voice === R.book.voice && (s.mood || null) === (R.book.mood || null)) return;
  if (!R.cfg) R.cfg = await api("/api/config").catch(() => null);
  R.revoice = { voice: s.voice, mood: s.mood || null, label: voiceLabel(s.voice, s.mood) };
  pollRevoice();
}

// ---------------------------------------------------------------- wiring
function applyTheme(theme) {
  if (theme === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
}

function setSize(px) {
  document.body.style.setProperty("--r-size", `${px}px`);
  store.set("readSize", px);
  requestAnimationFrame(() => { refreshPill(); R.center = true; kick(); });
}

function wire() {
  $("#r-play").addEventListener("click", togglePlay);
  $("#r-back").addEventListener("click", () => seekTo(audio.currentTime - 15));
  $("#r-fwd").addEventListener("click", () => seekTo(audio.currentTime + 30));
  $("#r-prev-ch").addEventListener("click", () => goChapter(-1));
  $("#r-next-ch").addEventListener("click", () => goChapter(1));
  $("#r-speed").addEventListener("click", () => setSpeed(SPEEDS.find((s) => s > R.speed + 1e-6) ?? SPEEDS[0]));
  $("#r-slower").addEventListener("click", () => stepSpeed(-1));
  $("#r-faster").addEventListener("click", () => stepSpeed(1));
  $("#r-follow").addEventListener("click", resumeFollowing);

  $("#r-resume-go").addEventListener("click", () => { $("#r-resume").hidden = true; R.touched = true; audio.play(); });
  $("#r-resume-start").addEventListener("click", () => { $("#r-resume").hidden = true; seekTo(0); audio.play(); });
  audio.addEventListener("play", () => { R.touched = true; keepSpeed(); $("#r-play").classList.add("r-playing"); $("#r-play").setAttribute("aria-label", "Pause"); R.center = true; kick(); });
  audio.addEventListener("pause", () => { $("#r-play").classList.remove("r-playing"); $("#r-play").setAttribute("aria-label", "Play"); saveProgress(); updatePlayerUi(); });
  audio.addEventListener("ended", () => { saveProgress(); toast("That’s the end of the book."); });
  audio.addEventListener("seeked", () => { tick(); kick(); });
  audio.addEventListener("loadedmetadata", () => { keepSpeed(); drawTicks(); updatePlayerUi(); });
  audio.addEventListener("error", () => toast("The audio file couldn’t load. Rebuild the book, then try again.", true));

  // Scrubber, with the chapter under the pointer shown above it.
  const seek = $("#r-seek"), peek = $("#r-peek");
  seek.addEventListener("input", () => {
    R.dragging = true;
    const v = Number(seek.value), total = duration();
    $("#r-fill").style.width = `${(100 * v) / total}%`;
    $("#r-now").textContent = clock(v);
    peek.hidden = false;
    peek.style.left = `${(100 * v) / total}%`;
    peek.textContent = `${chapterAt(v).title}, ${clock(v - chapterAt(v).start)}`;
  });
  seek.addEventListener("change", () => { R.dragging = false; peek.hidden = true; seekTo(Number(seek.value)); });

  // Tap a word (or a block) to listen from there.
  textEl.addEventListener("click", (e) => {
    if (R.textOnly || !window.getSelection()?.isCollapsed) return;
    const node = e.target.closest(".r-b, .r-h");
    if (!node) return;
    const it = R.items[Number(node.dataset.i)];
    if (!it) return;
    const span = e.target.closest(".r-w");
    const k = span && it.spans ? Number(span.dataset.k) : wordAtPoint(it, e.clientX, e.clientY);
    seekTo(it.w[k * 4 + 2] / 100);
  });

  // The listener scrolling themselves pauses following until they come back.
  const user = () => { R.lastUserScroll = performance.now(); stopFollowing(); };
  addEventListener("wheel", user, { passive: true });
  addEventListener("touchmove", user, { passive: true });
  addEventListener("keydown", (e) => {
    if (["PageUp", "PageDown", "Home", "End"].includes(e.key)) user();
  });
  addEventListener("resize", () => { refreshPill(); R.center = true; kick(); });

  // Sleep timer menu.
  $("#r-sleep").addEventListener("click", (e) => { e.stopPropagation(); $("#r-sleep-menu").hidden = !$("#r-sleep-menu").hidden; });
  for (const b of document.querySelectorAll("#r-sleep-menu [data-min]")) b.addEventListener("click", () => setSleep(b.dataset.min));
  document.addEventListener("click", (e) => { if (!e.target.closest("#r-sleep-menu, #r-sleep")) $("#r-sleep-menu").hidden = true; });

  // Voice change.
  $("#r-voice-btn").addEventListener("click", () => openVoices().catch((err) => toast(err.message, true)));
  $("#r-voice").addEventListener("close", stopClip);
  $("#r-voice-here").addEventListener("click", () => { if (R.offline) chooseVoice(...R.offline, true); });

  // Checkpoints.
  $("#r-marks").addEventListener("click", () => { loadMarks(); $("#r-panel").showModal(); });
  $("#r-add-mark").addEventListener("click", () => addCheckpoint("manual"));
  for (const b of document.querySelectorAll("[data-close]")) b.addEventListener("click", () => b.closest("dialog").close());
  wireListeners();

  // Text size and theme.
  $("#r-smaller").addEventListener("click", () => setSize(SIZES[Math.max(0, SIZES.indexOf(currentSize()) - 1)]));
  $("#r-bigger").addEventListener("click", () => setSize(SIZES[Math.min(SIZES.length - 1, SIZES.indexOf(currentSize()) + 1)]));
  $("#r-theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    try { localStorage.setItem("theme", next); } catch {}
    applyTheme(next);
  });

  // Keyboard: space plays, arrows skip, shift+arrows change chapter, B marks.
  addEventListener("keydown", (e) => {
    if (R.textOnly || e.target.closest?.("input, textarea, select, dialog") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); e.shiftKey ? goChapter(-1) : seekTo(audio.currentTime - 15); }
    else if (e.key === "ArrowRight") { e.preventDefault(); e.shiftKey ? goChapter(1) : seekTo(audio.currentTime + 30); }
    else if (e.key === "b" || e.key === "B") addCheckpoint("manual");
    else if (e.key === "f" || e.key === "F") resumeFollowing();
  });

  // Save the spot when the page goes away (closing the tab, locking the phone).
  addEventListener("pagehide", () => saveProgress(true));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) saveProgress(true);
    else { R.center = true; kick(); catchUp(); }
  });
  addEventListener("focus", catchUp);
}

function currentSize() {
  const px = store.get("readSize", innerWidth <= 640 ? 18 : 20);
  return SIZES.includes(px) ? px : 20;
}

// ------------------------------------------------- books without a map yet
// A book not voiced yet: its text to read now, with a note on voicing it above.
async function openTextOnly() {
  try {
    R.map = await api(`/api/books/${bookId}/textmap`);
  } catch {
    return false;
  }
  R.textOnly = true;
  document.body.classList.add("text-only");
  buildText();
  textEl.hidden = false;
  needsRebuild("");
  return true;
}

async function needsRebuild(message) {
  const status = $("#r-status");
  if (!R.textOnly) $(".r-tools").hidden = true;  // nothing to read or mark yet
  status.classList.toggle("r-status-bar", Boolean(R.textOnly));
  const b = R.book;
  const audio = b.kind === "audio";
  const busy = ["queued", "voicing", "assembling", "checking", "parsing", "transcribing"].includes(b.status);
  const rebuild = el("button", { type: "button", class: "primary",
    textContent: audio ? "Make Read along" : R.textOnly ? "Voice it now" : "Rebuild it now" });
  status.replaceChildren(
    el("h2", { textContent: busy ? (audio ? "Read along is being made" : "This book is being made")
      : R.textOnly ? "Not voiced yet" : "Read along isn’t ready for this book" }),
    el("p", { textContent: busy
      ? (R.textOnly ? "Read it here meanwhile. " : "") + "Read along opens by itself when it’s done."
      : audio ? "Whisper listens to the audiobook and writes down its text, with the time of every word. "
        + "Without a graphics card that takes a while; this page opens Read along by itself when it’s done."
      : R.textOnly ? "Read it here now. Voice it to listen along, with every word lit up as it’s read."
      : `${message} A rebuild voices it again with the same voice and chapters, and records when each word is spoken.` }));
  if (!busy) {
    status.append(rebuild);
    rebuild.addEventListener("click", async () => {
      rebuild.disabled = true;
      try {
        await api("/api/start", { method: "POST", json: { ids: [bookId] } });
        R.book.status = "queued";  // show it waiting, and watch it
        needsRebuild("");
      }
      catch (err) { toast(err.message, true); rebuild.disabled = false; }
    });
    return;
  }
  // Wait for it.
  const poll = async () => {
    try {
      R.book = await api(`/api/books/${bookId}`);
      if (R.book.readalong) return location.reload();
      if (R.book.status === "transcribing") {
        status.querySelector("p").textContent =
          `${R.book.message}, ${Math.round(R.book.progress * 100)}%. Read along opens by itself when it’s done.`;
      } else if (["voicing", "assembling", "checking", "queued"].includes(R.book.status)) {
        const p = Math.round(Math.min(1, R.book.progress / 0.9) * 100);
        status.querySelector("p").textContent = R.book.status === "voicing"
          ? `Voicing, ${p}%. Read along opens by itself when it’s done.`
          : "Finishing up. Read along opens by itself when it’s done.";
      } else if (R.book.status === "failed") {
        status.querySelector("p").textContent = R.book.message || "Building the book failed.";
        return;
      }
    } catch {}
    setTimeout(poll, 3000);
  };
  poll();
}

// ----------------------------------------------------------------- start
async function init() {
  wire();
  setSize(currentSize());
  setSpeed(store.get("readSpeed", 1));
  try {
    R.book = await api(`/api/books/${bookId}`);
  } catch {
    $("#r-status").textContent = "That book isn’t here any more.";
    return;
  }
  const title = R.book.meta.title || R.book.filename;
  $("#r-book").textContent = title;
  document.title = `${title}: Read along`;
  $("#r-voice-btn").hidden = R.book.kind === "audio";  // it has its own narration
  try {
    ({ map: R.map, etag: R.etag } = await loadMap());
  } catch (err) {
    // A book not voiced yet can still be read.
    if (R.book.kind === "audio" || R.book.status === "done" || !await openTextOnly()) needsRebuild(err.message);
    return;
  }
  buildText();
  $("#r-status").hidden = true;
  textEl.hidden = false;
  $("#r-player").hidden = false;
  audio.src = `/api/books/${bookId}/audio`;
  setupMediaSession();
  updateMediaMeta();
  drawTicks();

  await startListener();
  const saved = await api(`/api/books/${bookId}/progress${who()}`).catch(() => null);
  R.savedAt = saved?.updated || 0;
  R.ready = true;
  const at = saved ? positionOf(saved) : 0;
  if (at > 3) {
    audio.currentTime = at;
    showResume(at, saved.updated, false);
    // Setting currentTime before the file's metadata arrives can be ignored.
    audio.addEventListener("loadedmetadata", () => {
      if (audio.currentTime < 1) audio.currentTime = at;
      seekTo(at, { quiet: true });
    }, { once: true });
  }
  R.jumped = true;
  R.center = true;
  tick();
  aim();
  kick();
  updatePlayerUi();
  checkRevoice();  // a voice change started earlier, maybe on another device
}

init();
