"""Watch and read: simple hand-drawn pictures of what's happening, shown above
the text as the book is read.

A book's scenes come from its Read along map: about every SCENE_WORDS words,
at a paragraph's start, one scene. Grouping by words, not seconds, keeps the
scenes (and so the pictures) the same in every voice the book is made in; their
times always come from the current map, worked out again when it changes.

Each scene's picture is drawn from its passage's most paintable sentence: a
concrete, non-dialogue one, with the words that make image models write text
taken out, and the style added. Pictures are drawn by the GPU server
(clone/server.py, /api/draw), named after what they were drawn from (so a scene
planned again with the same words keeps its picture), and given the same ink on
paper look, so every one seems drawn by the same hand.

Scene choice and prompts are adapted from SeeStory's director
(https://github.com/pgotta/SeeStory, MIT License, Copyright (c) 2026 pgotta;
see THIRD_PARTY_NOTICES.md).

    /data/books/<id>/scenes/scenes.json   the plan, for the map it was made from
    /data/books/<id>/scenes/<hash>.jpg    the pictures
    /data/books/<id>/scenes/queued        drawing asked for (the worker draws them)
    /data/books/<id>/scenes/state.json    how drawing went (written by the worker)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from . import config, gpus, readalong

MIN_WORDS = 25  # a scene shorter than this joins the one before it
WIDTH, HEIGHT = 1024, 576
# Slow movement on each still, played by the reader (SeeStory's Ken Burns presets).
MOTIONS = ["gentle_drift", "pan_right", "slow_reveal", "pan_left", "drift_up"]

_IMAGERY = {
    "storm", "rain", "lightning", "thunder", "wind", "wave", "ocean", "sea",
    "lighthouse", "mountain", "forest", "tree", "river", "fire", "flame",
    "snow", "ice", "fog", "mist", "cloud", "sky", "sun", "moon", "star",
    "castle", "tower", "bridge", "city", "street", "ship", "boat", "train",
    "horse", "wolf", "bird", "dragon", "sword", "candle", "lantern", "window",
    "door", "garden", "field", "desert", "cliff", "cave", "ruins", "church",
    "ballroom", "throne", "blood", "shadow", "dawn", "dusk", "sunset",
    "sunrise", "rose", "flower", "mansion", "cottage", "harbor", "valley",
    "car", "truck", "road", "house", "kitchen", "table", "room", "bed", "stairs",
    "beach", "farm", "dog", "cat", "soldiers", "uniform", "coat", "boots",
}
_TENSION = {
    "suddenly", "scream", "screamed", "blood", "death", "died", "killed",
    "fire", "burning", "explosion", "gun", "knife", "fell", "crash", "storm",
    "betrayed", "revealed", "secret", "truth", "discovered", "vanished",
    "chase", "ran", "fled", "battle", "fight", "war", "kiss", "kissed",
    "wept", "tears", "darkness", "terror", "horror", "monster", "ghost",
}
_HUMAN = {
    "man", "woman", "boy", "girl", "child", "person", "people", "mother",
    "father", "mom", "dad", "brother", "sister", "husband", "wife", "soldier",
    "doctor", "teacher", "officer", "he", "she", "him", "her", "his", "hers",
}

_STOP_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_QUOTE = re.compile(r"[\"“”‘’']")
_DIALOGUE = re.compile(r"^\s*[\"“].*?[\"”]\s*$")
_SPEECH_SPAN = re.compile(r"[\"“][^\"”]*[\"”]")

STYLES = {
    "sketch": "loose hand-drawn pencil sketch, black ink lines on plain white paper, light cross-hatching, "
              "minimal shading, monochrome, simple uncluttered composition",
    "ink": "detailed pen-and-ink illustration with selective watercolor washes, fine linework, coherent composition",
    "storybook": "warm storybook watercolor illustration, soft edges, gentle light, hand-painted texture, "
                 "coherent composition",
    "cinematic": "cinematic painterly illustration, dramatic natural lighting, rich depth, coherent composition",
    "noir": "moody film-noir illustration, high contrast, deep shadows, monochrome with one warm accent, "
            "coherent composition",
    "oil": "classical oil painting, visible brushwork, golden-hour light, romantic realism, coherent composition",
}
NEGATIVE = (
    "text, words, letters, title, book cover, captions, subtitle, watermark, signature, logo, frame, border, "
    "low quality, blurry, bad anatomy, bad proportions, deformed body, disfigured, malformed hands, bad hands, "
    "fused fingers, extra fingers, missing fingers, extra arms, extra legs, extra limbs, duplicate person, "
    "extra face, extra head, deformed face"
)
SKETCH_NEGATIVE = ", color, colorful, painting, photo, photograph, 3d render"  # keeps the sketch a sketch

_BAD_PROMPT_TERMS = re.compile(
    r"\b(audio ?book|book|novel|ebook|e-book|paperback|hardcover|cover|"
    r"title|titled|chapter|page|text|words?|lettering|caption|subtitles?|"
    r"logo|watermark|signature|"
    r"thriller|mystery|suspense|horror|romance|fantasy|sci-?fi|"
    r"science fiction|drama|comedy|crime|noir fiction|"
    r"adult[- ]?oriented|adults?|nsfw|explicit|erotic|porn(ographic)?|"
    r"gore|gory|graphic)\b",
    re.I,
)


# --------------------------------------------------------------------------
# The most paintable sentence (SeeStory's director)
# --------------------------------------------------------------------------

def _strip_dialogue(s: str) -> str:
    s = _SPEECH_SPAN.sub("", s or "")
    s = _QUOTE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ,;:—-")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def _has_human(text: str) -> bool:
    return any(w in _HUMAN for w in _words(text))


def focus_candidates(text: str, max_chars: int = 280) -> list[str]:
    """Paintable sentences, best first, without duplicates: imagery and tension
    count for, dialogue and questions against, and a sentence that opens with a
    pronoun takes the sentence before it along, so the picture knows who."""
    sents = [s.strip() for s in _STOP_SENTENCE.split(text or "") if s.strip()]
    scored = []
    for idx, sent in enumerate(sents):
        words = _words(sent)
        score = sum(2 for w in words if w in _IMAGERY) + sum(1 for w in words if w in _TENSION)
        if _DIALOGUE.match(sent):
            score -= 5
        if "?" in sent:
            score -= 3
        if sent.lstrip()[:1] in "\"“‘'":
            score -= 2
        score -= abs(len(sent) - 150) / 180.0
        scored.append((score, idx, sent))
    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _score, idx, raw in scored:
        best = _strip_dialogue(raw) or _QUOTE.sub("", raw)
        low = _words(best)
        if low and low[0] in {"he", "she", "they", "his", "her", "their"} and idx > 0:
            context = _strip_dialogue(sents[idx - 1])
            if context and len(context) <= 140:
                best = f"{context}. {best}"
        best = best[:max_chars].strip()
        key = re.sub(r"[^a-z0-9]+", " ", best.lower()).strip()
        if best and key and key not in seen:
            seen.add(key)
            out.append(best)
    return out


def _too_similar(a: str, b: str) -> bool:
    """Near-identical scenes one after another make the same picture twice."""
    a, b = (re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip() for s in (a, b))
    return bool(a and b) and (a == b or SequenceMatcher(None, a, b).ratio() >= 0.88)


def _scrub(s: str) -> str:
    s = _BAD_PROMPT_TERMS.sub("", s or "")
    s = re.sub(r"\s*,(?:\s*,)+", ", ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" ,;")


def _approx_tokens(s: str) -> int:
    return int(len(s.split()) * 1.35) + s.count(",")


def _trim_to_tokens(s: str, max_tokens: int) -> str:
    out: list[str] = []
    for word in s.split():
        out.append(word)
        if _approx_tokens(" ".join(out)) >= max_tokens:
            break
    return " ".join(out).rstrip(" ,;:—-")


def build_prompt(focus: str, text: str, style: str) -> str:
    """The sentence, anatomy help when people are in it, and the style, within
    the image model's 77-token prompt (the style goes last and must fit)."""
    focus = re.sub(r"\s+", " ", _scrub(_QUOTE.sub("", focus))).strip().rstrip(".") or "a quiet atmospheric scene"
    look = _scrub(STYLES.get(style, STYLES["sketch"]))
    anatomy = "natural anatomy, two arms, five fingers per hand, no duplicated body parts" if _has_human(text) else ""
    coherence = "single coherent scene, one moment"
    tail = ", ".join(part for part in (anatomy, coherence, look) if part)
    focus = _trim_to_tokens(focus, max(10, 72 - _approx_tokens(tail) - 1))
    return ", ".join(part for part in (focus, anatomy, coherence, look) if part)


# --------------------------------------------------------------------------
# Scenes from the map
# --------------------------------------------------------------------------

def folder(book_id: str) -> Path:
    return config.BOOKS_DIR / book_id / "scenes"


def _map_key(ra: dict) -> str:
    """Changes when the map does (a new voice, edited text): the scenes are timed again."""
    return hashlib.sha1(json.dumps([ra.get("duration")] + [[c["start"], c["end"], len(c["blocks"])]
                                                             for c in ra["chapters"]]).encode()).hexdigest()[:16]


def picture_name(prompt: str, seed: int, model: str) -> str:
    return hashlib.sha1(f"{model}\0{seed}\0{prompt}".encode()).hexdigest()[:16] + ".jpg"


def plan(book_id: str, style: str | None = None) -> dict:
    """The book's scenes, from its current map: worked out again only when the
    map or the style changed. Times are in seconds."""
    style = style or config.DRAW_STYLE
    path = readalong.path_for(book_id)
    if not path.exists():
        raise FileNotFoundError("This book has no Read along yet.")
    ra = json.loads(path.read_text(encoding="utf-8"))
    key = _map_key(ra)
    # The book's own title in a sentence ("…the age of Annemarie in Number the Stars")
    # gets drawn as lettering: it comes out of the prompts.
    from . import db
    title = ((db.get_book(book_id) or {}).get("meta") or {}).get("title") or ""
    title_re = re.compile(rf"\s*\b(?:in |of )?{re.escape(title)}\b", re.I) if len(title) >= 4 else None
    saved = folder(book_id) / "scenes.json"
    if saved.exists():
        try:
            old = json.loads(saved.read_text(encoding="utf-8"))
            if old.get("map") == key and old.get("style") == style and old.get("model") == config.DRAW_MODEL:
                return old
        except ValueError:
            pass
    scenes: list[dict] = []
    for ch in ra["chapters"]:
        groups: list[list[dict]] = []
        words = 0
        for b in ch["blocks"]:
            n = len(b["w"]) // 4
            # A scene closes at a paragraph's start once it has its words (or at
            # any block, once it has half as many again).
            if groups and (words >= config.SCENE_WORDS and b["p"] != groups[-1][-1]["p"]
                           or words >= config.SCENE_WORDS * 1.5):
                groups.append([])
                words = 0
            if not groups:
                groups.append([])
            groups[-1].append(b)
            words += n
        if len(groups) > 1 and sum(len(b["w"]) // 4 for b in groups[-1]) < MIN_WORDS:
            groups[-2] += groups.pop()
        history: list[str] = []
        for k, group in enumerate(groups):
            text = " ".join(b["text"] for b in group)
            candidates = focus_candidates(text)
            focus = next((c for c in candidates if not any(_too_similar(c, old) for old in history[-4:])),
                         candidates[0] if candidates else "")
            if focus:
                history.append(focus)
            n = len(scenes)
            seed = int(hashlib.sha1(f"{book_id}:{ch['idx']}:{k}".encode()).hexdigest()[:8], 16)
            prompt = build_prompt(title_re.sub("", focus) if title_re else focus, text, style)
            scenes.append({
                "id": n, "chapter": ch["idx"],
                "start": round((ch["start"] if k == 0 else group[0]["t"][0]) / 100, 2),
                "focus": focus, "prompt": prompt, "seed": seed,
                "motion": "slow_reveal" if k == 0 else MOTIONS[n % len(MOTIONS)],
                "image": picture_name(prompt, seed, config.DRAW_MODEL),
            })
        if scenes:
            scenes[-1]["chapter_end"] = round(ch["end"] / 100, 2)
    for a, b in zip(scenes, scenes[1:]):
        a["end"] = a.pop("chapter_end", b["start"])
    if scenes:
        scenes[-1]["end"] = scenes[-1].pop("chapter_end", round(ra["duration"], 2))
    out = {"v": 1, "map": key, "style": style, "model": config.DRAW_MODEL,
           "negative": NEGATIVE + (SKETCH_NEGATIVE if style == "sketch" else ""), "scenes": scenes}
    saved.parent.mkdir(parents=True, exist_ok=True)
    tmp = saved.with_name(".scenes.json.part")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, saved)
    return out


def ink(data: bytes) -> bytes:
    """Every picture the same ink on paper: grey tones mapped from dark ink to
    warm paper, so pictures from any model look drawn by one hand."""
    import io

    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(data)).convert("RGB")
    grey = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)
    out = ImageOps.colorize(grey, black="#2a2520", white="#f6f1e5")
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=86, optimize=True)
    return buf.getvalue()


def state(book_id: str) -> dict:
    """How drawing is going: {"state": idle|queued|drawing|done|failed, "message"}."""
    f = folder(book_id)
    try:
        st = json.loads((f / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {"state": "idle"}
    if (f / "queued").exists():
        if st.get("state") != "drawing":
            st["state"] = "queued"
    elif st.get("state") == "queued":  # stopped before it started
        st = {"state": "idle", "message": "Stopped"}
    return st


def set_state(book_id: str, **fields) -> None:
    f = folder(book_id)
    f.mkdir(parents=True, exist_ok=True)
    tmp = f / ".state.json.part"
    tmp.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, f / "state.json")


def queue(book_id: str) -> None:
    f = folder(book_id)
    f.mkdir(parents=True, exist_ok=True)
    (f / "queued").touch()


def draw_book(book_id: str, stop) -> None:
    """Draw the pictures a book is missing, one at a time, on the clone server.
    While it speaks a cloned voice, drawing waits: books come first. Stops early
    when the queued flag goes (Stop) or the worker does."""
    import logging

    from . import gpus
    from .clone import CloneBusy, CloneError
    log = logging.getLogger("forge.scenes")
    f = folder(book_id)
    flag = f / "queued"
    if not gpus.has("pictures"):
        set_state(book_id, state="failed", message="Pictures are drawn by a GPU server: add one in Settings.")
        flag.unlink(missing_ok=True)
        return
    try:
        p = plan(book_id)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        set_state(book_id, state="failed", message=f"Couldn’t plan the pictures: {exc}")
        flag.unlink(missing_ok=True)
        return
    total = len(p["scenes"])
    done = sum(1 for s in p["scenes"] if (f / s["image"]).exists())
    for s in p["scenes"]:
        if (f / s["image"]).exists():
            continue
        if stop.is_set() or not flag.exists():
            set_state(book_id, state="idle", message=f"Stopped at {done} of {total} pictures", done=done, total=total)
            return
        set_state(book_id, state="drawing", message=f"Drawing pictures: {done} of {total}", done=done, total=total)
        while True:
            try:
                # Picked for every picture: the next server that draws takes over if one stops.
                client = gpus.client("pictures", timeout=600)
                if client is None:
                    raise CloneError("No GPU server draws pictures any more: add one in Settings.")
                data = client.draw(s["prompt"], p["negative"], p["model"], WIDTH, HEIGHT, s["seed"])
                break
            except CloneBusy:
                set_state(book_id, state="drawing", done=done, total=total,
                          message=f"Drawing pictures: {done} of {total}. Waiting while the GPU server speaks a voice")
                if stop.wait(60):
                    return
            except CloneError as exc:
                log.warning("Drawing %s: %s", book_id, exc)
                set_state(book_id, state="failed", message=f"Couldn’t draw: {exc}", done=done, total=total)
                flag.unlink(missing_ok=True)
                return
        tmp = f / f".{s['image']}.part"
        tmp.write_bytes(ink(data))
        os.replace(tmp, f / s["image"])
        done += 1
    set_state(book_id, state="done", message=f"{total} pictures", done=done, total=total)
    flag.unlink(missing_ok=True)


def draw_loop(stop) -> None:
    """The worker's thread for pictures: the books queued for them, oldest first."""
    import logging
    while not stop.wait(10):
        for flag in sorted(config.BOOKS_DIR.glob("*/scenes/queued"), key=lambda p: p.stat().st_mtime):
            if stop.is_set():
                return
            try:
                draw_book(flag.parent.parent.name, stop)
            except Exception:
                logging.getLogger("forge.scenes").exception("Drawing pictures for %s failed", flag.parent.parent.name)
                set_state(flag.parent.parent.name, state="failed", message="Drawing failed: see the worker's log.")
                flag.unlink(missing_ok=True)


def public(book_id: str) -> dict:
    """What the reader needs: every scene's time, motion and picture (once drawn)."""
    plan_ = plan(book_id)
    f = folder(book_id)
    have = {p.name for p in f.glob("*.jpg")}
    scenes = [{"id": s["id"], "start": s["start"], "end": s["end"], "motion": s["motion"],
               "alt": s["focus"], "image": s["image"] if s["image"] in have else None}
              for s in plan_["scenes"]]
    st = state(book_id)
    return {**st, "total": len(scenes), "drawn": sum(1 for s in scenes if s["image"]), "scenes": scenes,
            "can_draw": gpus.has("pictures")}
