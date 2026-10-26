"""Prepare chapter text for speech and cut it into parallel work units.

A unit is a list of items: ["say", text] or ["pause", seconds]. Units are
independent, so they can be voiced on any worker in any order and stitched
back together afterwards.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from . import config

SCENE_BREAK_RE = re.compile(r"^[\s*#~•·°=_—–-]{1,20}$")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
FOOTNOTE_RE = re.compile(r"\[(?:\d{1,3}|[a-z]|[ivx]{1,4})\]")
DASH_RE = re.compile(r"(?<=\w)\s*(?:—|--|―)\s*(?=\w)")
EMPHASIS_RE = re.compile(r"(?<!\w)[*_]{1,3}(?=\w)|(?<=\w)[*_]{1,3}(?!\w)")
ABBREVIATIONS = [(re.compile(r"\bMr\."), "Mister"), (re.compile(r"\bMrs\."), "Missus"),
                 (re.compile(r"\bMs\."), "Miz"), (re.compile(r"\bDr\."), "Doctor"),
                 (re.compile(r"\bProf\."), "Professor")]
# Sound words both engines would spell out letter by letter ("Shhh" -> "ess
# aitch aitch aitch"), respelled to what they sound like. Each respelling was
# checked against Kokoro's and Piper's phonemes. Hmm, psst, tsk, ugh and the
# like are already read as sounds.
SOUND_WORDS = [
    (re.compile(r"\bs*sh+\b(?![-‐]?\d)", re.I), "shush"),  # not "SH-2" or "SH2"
    (re.compile(r"\bbr{2,}\b", re.I), "burr"),
    (re.compile(r"\bgr{2,}\b", re.I), "grur"),
    (re.compile(r"\bh+u?m+p+h+\b", re.I), "humph"),
    (re.compile(r"\b(?:m+h+m+|mm-?hm+)\b", re.I), "uh-huh"),
    (re.compile(r"(?<![\d.,]\s)(?<!\d)\bm{2,}\b", re.I), "hmm"),  # not "20 mm"
    (re.compile(r"\bpf+t+\b|\bpf{2,}\b", re.I), "pfuh"),
    (re.compile(r"\bs{3,}\b", re.I), "hiss"),
    (re.compile(r"\bz{3,}\b", re.I), "zuzz"),
]


def _respell(pattern: re.Pattern, word: str, text: str) -> str:
    def keep_case(m: re.Match) -> str:
        found = m.group(0)
        if found.isupper() and len(found) > 1:
            return word.upper()
        return word.capitalize() if found[0].isupper() else word
    return pattern.sub(keep_case, text)


ROMAN_TITLE_RE = re.compile(
    r"^((?:chapter|part|book|volume|section|letter|canto|stave)\s+)([IVXLCDM]+)\b", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+")
SENTENCE_END_RE = re.compile(r"[.!?…]+[\"”’')\]]*(?=\s+\S)")
# A period after these isn't the end of a sentence ("St. John", "a.m. on").
NO_BREAK_AFTER = {"st", "jr", "sr", "vs", "etc", "eg", "ie", "mt", "lt", "col", "gen", "capt",
                  "sgt", "no", "vol", "fig", "approx", "dept", "inc", "ltd", "co", "ave", "rd",
                  "am", "pm", "mister", "missus", "doctor", "professor"}


def sentences(text: str) -> list[str]:
    """Split text into sentences, keeping every character (closing quotes too).
    Doesn't split after an abbreviation or an initial ("J. K. Rowling"), or
    before a lowercase word ('"Run!" she shouted' is one sentence)."""
    out, start = [], 0
    for m in SENTENCE_END_RE.finditer(text):
        end = m.end()
        before = re.sub(r"[^\w.]", "", text[start:m.start()].split()[-1] if text[start:m.start()].split() else "")
        word = before.lower().replace(".", "")
        rest = text[end:].lstrip()
        if (m.group().startswith(".") and (word in NO_BREAK_AFTER or re.fullmatch(r"[A-Za-z]", before)))\
                or (rest[:1].islower()):
            continue
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        value = ROMAN[ch]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub("", text)
    text = FOOTNOTE_RE.sub("", text)
    text = DASH_RE.sub(", ", text)
    text = EMPHASIS_RE.sub("", text)
    for pattern, replacement in ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    for pattern, word in SOUND_WORDS:
        text = _respell(pattern, word, text)
    return re.sub(r"\s+", " ", text).strip()


def spoken_title(title: str) -> str:
    """'CHAPTER XIV: The Road' -> 'Chapter 14: The Road.'"""
    title = clean(title)
    title = ROMAN_TITLE_RE.sub(lambda m: f"{m.group(1)}{roman_to_int(m.group(2))}", title)
    if re.fullmatch(r"[IVXLCDM]{1,7}\.?", title):
        title = f"Chapter {roman_to_int(title.rstrip('.'))}"
    if title and title[-1] not in ".!?":
        title += "."
    return title


# Paragraph ends mid-sentence: on a dash, or on a comma inside closing quotes
# ('"…to your bed,"' + 'Mama said one morning…').
CONTINUES_RE = re.compile(r"[—–-]$|--$|,[\"”’']+$")
OPEN_END_RE = re.compile(r"[\w,]$")  # ...or on a word or comma
# A question or exclamation followed by a lowercase paragraph is one sentence
# whose dialogue tag got its own paragraph: "Would I have done that?" + "they ask…".
ASKED_RE = re.compile(r"[?!][\"”’']*$")


def _continues(prev: str, nxt: str) -> bool:
    """True when an ebook broke one sentence across two paragraphs:
    'in the corner of this bedroom—' + 'Annemarie could see…', or 'The Free Danes—' + 'was…'."""
    prev, nxt = prev.rstrip(), nxt.lstrip()
    if not prev or not nxt:
        return False
    if CONTINUES_RE.search(prev):
        return True
    if ASKED_RE.search(prev) and nxt[0].islower():
        return True
    return bool(OPEN_END_RE.search(prev)) and (nxt[0].islower() or nxt[0] in "—–")


def paragraphs(text: str) -> list[str | None]:
    """Clean paragraphs; None marks a scene break. A sentence an ebook split
    across two paragraphs is joined back, so it isn't voiced with a paragraph
    pause in the middle."""
    return [None if p is None else p[1] for p in paragraph_pairs(text)]


def display_text(raw: str) -> str:
    """A paragraph as Read along shows it: the book's own wording ("Mrs.", "Shhh"),
    on one line, without footnote markers or emphasis marks."""
    text = unicodedata.normalize("NFC", raw)
    text = FOOTNOTE_RE.sub("", text)
    text = EMPHASIS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def paragraph_pairs(text: str) -> list[tuple[str, str] | None]:
    """(as printed, as spoken) for every paragraph; None marks a scene break.
    Paragraphs that clean down to nothing are dropped from both."""
    raws: list[str | None] = []
    for raw in re.split(r"\n\s*\n", text):
        raw = raw.strip()
        if not raw:
            continue
        if SCENE_BREAK_RE.match(raw):
            raws.append(None)
        elif raws and raws[-1] is not None and _continues(raws[-1], raw):
            raws[-1] = f"{raws[-1]} {raw}"
        else:
            raws.append(raw)
    out: list[tuple[str, str] | None] = []
    for raw in raws:
        if raw is None:
            out.append(None)
            continue
        cleaned = clean(raw)
        if cleaned:
            out.append((display_text(raw), cleaned))
    return out


def split_long(paragraph: str, limit: int) -> list[str]:
    pieces, current = [], ""
    for sentence in sentences(paragraph):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def build_units(title: str | None, text: str, limit: int | None = None) -> list[list]:
    return plan_units(title, text, limit)[0]


def plan_units(title: str | None, text: str,
               limit: int | None = None) -> tuple[list[list], list[list], list[str]]:
    """Cut a chapter into units. Returns (units, sources, display):

    units    [[["say", text], ["pause", s], ...], ...]: what gets voiced
    sources  same shape: the paragraph each "say" came from (-1 = the title,
             None for pauses), so Read along can put words back in place
    display  each paragraph as printed, for Read along to show
    """
    limit = limit or config.UNIT_CHARS
    units: list[list] = []
    sources: list[list] = []
    current: list = []
    current_src: list = []
    size = 0
    if title:
        current = [["say", spoken_title(title)], ["pause", config.TITLE_PAUSE]]
        current_src = [-1, None]
        size = len(title)
    display: list[str] = []
    for pair in paragraph_pairs(text):
        if pair is None:
            if current:
                current.append(["pause", config.SCENE_PAUSE])
                current_src.append(None)
            continue
        shown, para = pair
        n = len(display)
        display.append(shown)
        for piece in split_long(para, limit) if len(para) > limit else [para]:
            if size and size + len(piece) > limit:
                units.append(current)
                sources.append(current_src)
                current, current_src, size = [], [], 0
            current += [["say", piece], ["pause", config.PARA_PAUSE]]
            current_src += [n, None]
            size += len(piece)
    if current:
        units.append(current)
        sources.append(current_src)
    return units, sources, display


def sample_items(title: str | None, text: str, seconds: float, speed: float = 1.0) -> list[list]:
    """The opening of a chapter, about `seconds` long, as a single unit."""
    per_second = config.WORDS_PER_MINUTE / 60 * speed
    budget = seconds * per_second
    items: list = []
    if title:
        spoken = spoken_title(title)
        items += [["say", spoken], ["pause", config.TITLE_PAUSE]]
        budget -= len(spoken.split()) + config.TITLE_PAUSE * per_second
    # Takes whole sentences until the budget runs out, including the one that
    # crosses it; the caller cuts the audio to length.
    for para in paragraphs(text):
        if para is None:
            continue
        taken = []
        for sentence in sentences(para):
            words = sentence.split()
            if len(words) > budget * 2:  # don't voice a whole page-long sentence
                words = words[:int(budget) + 8]
            taken.append(" ".join(words))
            budget -= len(words)
            if budget <= 0:
                break
        if taken:
            items += [["say", " ".join(taken)], ["pause", config.PARA_PAUSE]]
        if budget <= 0:
            break
    return items


def unit_text(items: list) -> str:
    return " ".join(value for kind, value in items if kind == "say")


def unit_key(voice_id: str, speed: float, items: list) -> str:
    """Content hash: a unit is re-voiced only when its text, voice or speed changes."""
    payload = json.dumps([voice_id, round(speed, 3), items], ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def word_count(text: str) -> int:
    return len(text.split())
