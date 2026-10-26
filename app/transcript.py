"""An added audiobook's text, from its Whisper transcript, and its Read along map.

Whisper's punctuation gives the sentences. Paragraphs break at long pauses, or
after a few sentences, so the reader never faces a wall of text. A chapter
that opens with its own title read out gets that as its heading rather than as
body text. The map is written in exactly readalong.py's format, so the reader
works as it does for books Forge voiced.

The text is kept in text/NNNN.txt like any book's and can be edited (a name
Whisper misheard, say). The map is then rebuilt from the cached transcript,
matching the edited words to the spoken ones, without listening again.
"""
from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from . import config, db, parsers, readalong, textprep

log = logging.getLogger("forge.transcript")

# Paragraphs: narrators pause longer between paragraphs than between sentences.
# The pauses after a chapter's sentences, measured in the audio, are split into
# short and long (_split, kept within these bounds), and a long one starts a
# new paragraph. A paragraph is cut after a few sentences anyway.
PARA_GAP_MIN, PARA_GAP_MAX = 0.55, 1.5  # seconds
PARA_SENTENCES = 5
PARA_CHARS = 600
SENTENCE_END_RE = re.compile(r"[.!?…][\"”’')\]]*$")
NO_BREAK_AFTER = textprep.NO_BREAK_AFTER | {"mr", "mrs", "ms", "dr", "prof"}
HEADING_WORDS = {"prologue", "epilogue", "introduction", "preface", "foreword", "afterword",
                 "interlude", "acknowledgments", "acknowledgements"}
GENERIC_HEADING_RE = re.compile(r"^(?:chapter|part|book|section|volume) \d+$")
HEADING_PAUSE = 0.6  # a title read out ends with a full stop, or at least a pause this long


def transcript_path(book_id: str, idx: int) -> Path:
    return config.BOOKS_DIR / book_id / "transcript" / f"{idx:04d}.json"


def load(book_id: str, idx: int) -> dict | None:
    path = transcript_path(book_id, idx)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _said(words: list) -> str:
    return re.sub(r"\s+", " ", "".join(w[0] for w in words)).strip()


def _is_title(said: str, title: str) -> bool:
    """Whether words read out are the chapter's title, in any form: "Chapter
    one." or "1. Why are you running?" under "1. Why Are You Running?"."""
    s, t = parsers._canon_heading(said), parsers._canon_heading(title)
    if not s:
        return False
    if s == t or parsers._same_heading(s, t) or GENERIC_HEADING_RE.match(s) or s in HEADING_WORDS:
        return True
    parts = [p for p in parsers.TITLE_PARTS_RE.split(title)[::2] if p.strip()]
    if any(parsers._same_heading(s, parsers._canon_heading(p)) for p in parts):
        return True
    # Whisper misheard a word of it.
    return len(t) >= 10 and 0.75 <= len(s) / len(t) <= 1.3 and SequenceMatcher(None, s, t).ratio() >= 0.85


def heading_count(words: list, title: str, quiet: list | None = None) -> int:
    """How many of a chapter's first words are its title read out."""
    best = 0
    limit = min(len(words), len((title or "").split()) + 4, 30)
    for n in range(1, limit + 1):
        said = _said(words[:n])
        ends = (n == len(words) or SENTENCE_END_RE.search(said) or said.endswith(":")
                or words[n][1] - words[n - 1][2] >= HEADING_PAUSE or _silence_after(words[n - 1], quiet))
        if ends and _is_title(said, title):
            best = n
    return best


def _silence_after(word: list, quiet: list | None) -> bool:
    """A pause of HEADING_PAUSE or more in the audio right after this word.
    Whisper often runs the next word's start back over the pause, so its own
    times can't tell ("Introduction" ends at 0.78 s, "It's" from 0.78 to 2.18)."""
    return any(word[2] - 0.2 <= q0 <= word[2] + 0.5 and q1 - q0 >= HEADING_PAUSE for q0, q1 in quiet or [])


def _sentences(words: list) -> list[tuple[int, int]]:
    """Word ranges of sentences, from Whisper's punctuation. No break after
    "Mr." or an initial."""
    out, begin = [], 0
    for i, w in enumerate(words):
        token = w[0].strip()
        if not SENTENCE_END_RE.search(token):
            continue
        core = re.sub(r"[^\w.]", "", token).lower().rstrip(".")
        if token.endswith(".") and (core in NO_BREAK_AFTER or re.fullmatch(r"[a-z]", core)):
            continue
        out.append((begin, i + 1))
        begin = i + 1
    if begin < len(words):
        out.append((begin, len(words)))
    return out


def _pause(words: list, a_end: int, b: int, quiet: list | None) -> float:
    """How long the reader pauses between two sentences. Silence measured in the
    audio when the transcript has it; Whisper's word times stretch into pauses."""
    prev_end, next_start = words[a_end - 1][2], words[b][1]
    if quiet is None:
        return next_start - prev_end
    lo, hi = prev_end - 0.3, next_start + 0.3
    return max([min(e, hi) - max(s, lo) for s, e in quiet if e > lo and s < hi] or [0.0])


def _split(values: list[float]) -> float:
    """The pause length that tells a chapter's long pauses (between paragraphs)
    from its short ones (between sentences): two bell curves are fitted to the
    pauses, and this is where the long one becomes the likelier. A plain
    midpoint (Otsu) sat too low: pauses within a paragraph vary much more than
    those between paragraphs."""
    v = np.clip(np.asarray(values, dtype=float), 0.05, 3.0)
    if len(v) < 8:
        return PARA_GAP_MAX
    mu, sd, w = np.percentile(v, [25, 90]), np.full(2, v.std() / 2 + 0.01), np.full(2, 0.5)
    for _ in range(60):
        p = w[:, None] * np.exp(-0.5 * ((v - mu[:, None]) / sd[:, None]) ** 2) / sd[:, None] + 1e-12
        r = p / p.sum(axis=0)
        w = r.mean(axis=1)
        mu = (r * v).sum(axis=1) / r.sum(axis=1)
        sd = np.sqrt((r * (v - mu[:, None]) ** 2).sum(axis=1) / r.sum(axis=1)) + 0.01
    short, long = np.argsort(mu)
    if mu[long] - mu[short] < 0.12 or w[long] < 0.05:
        return PARA_GAP_MAX  # no second kind of pause to go by
    xs = np.linspace(mu[short], mu[long], 200)
    density = lambda k: w[k] * np.exp(-0.5 * ((xs - mu[k]) / sd[k]) ** 2) / sd[k]
    longer = np.flatnonzero(density(long) > density(short))
    return float(xs[longer[0]]) if len(longer) else float(mu[long])


def _paragraphs(words: list, sentences: list[tuple[int, int]], quiet: list | None = None) -> list[tuple[int, int]]:
    """Group sentences into paragraphs: a new one after a long pause, or once
    this one has enough sentences."""
    gaps = [_pause(words, a_end, b, quiet) for (_, a_end), (b, _) in zip(sentences, sentences[1:])]
    limit = min(PARA_GAP_MAX, max(PARA_GAP_MIN, _split(gaps)))
    paras, begin, count, chars = [], 0, 0, 0
    for n, (a, b) in enumerate(sentences):
        if count and (gaps[n - 1] >= limit or count >= PARA_SENTENCES or chars >= PARA_CHARS):
            paras.append((begin, a))
            begin, count, chars = a, 0, 0
        count += 1
        chars += len(_said(words[a:b]))
    if sentences:
        paras.append((begin, sentences[-1][1]))
    return paras


def make_text(words: list, title: str, quiet: list | None = None) -> str:
    """The chapter's text as paragraphs, without the title read out at the start."""
    body = words[heading_count(words, title, quiet):]
    return "\n\n".join(_said(body[a:b]) for a, b in _paragraphs(body, _sentences(body), quiet))


def write_text(book_id: str, idx: int, title: str, data: dict) -> int:
    """Write text/NNNN.txt from a chapter's transcript. Returns its word count."""
    text = make_text(data["words"], title, data.get("quiet"))
    path = config.BOOKS_DIR / book_id / "text" / f"{idx:04d}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return textprep.word_count(text)


def _align(text: str, marks: list, start: float, end: float) -> list[tuple[int, int, float, float]]:
    """Word times for the text. Text Whisper wrote and nobody edited matches the
    spoken words one for one; edited text goes through readalong's diff."""
    tokens = [(m.start(), m.end()) for m in readalong.WORD_RE.finditer(text)
              if readalong._key(m.group())]
    if [readalong._key(text[a:b]) for a, b in tokens] == [readalong._key(m[0]) for m in marks]:
        return [(a, b, m[1], m[2]) for (a, b), m in zip(tokens, marks)]
    return readalong._align(text, marks, start, end)


def chapter_blocks(text: str, marks: list, start: float, end: float) -> list[dict]:
    """Blocks for one chapter, laid out as readalong.build lays out a voiced one."""
    cs = lambda seconds: int(round(seconds * 100))
    paras, breaks, brk = [], [], False
    for pair in textprep.paragraph_pairs(text):
        if pair is None:
            brk = True
            continue
        paras.append(pair[0])
        breaks.append(brk)
        brk = False
    # One alignment for the whole chapter, so text moved between paragraphs still finds its words.
    joined, offsets = "", []
    for p in paras:
        offsets.append(len(joined))
        joined += p + "\n\n"
    aligned = _align(joined, marks, start, end)
    blocks, k = [], 0
    for n, (p, off) in enumerate(zip(paras, offsets)):
        mine = []
        while k < len(aligned) and aligned[k][0] < off + len(p):
            s, e, t0, t1 = aligned[k]
            mine.append((s - off, e - off, t0, t1))
            k += 1
        first = True
        for a, b in readalong._blocks(p):
            inside = [(s - a, e - a, t0, t1) for s, e, t0, t1 in mine if s >= a and e <= b]
            if not inside:
                continue
            part = p[a:b]
            lead = len(part) - len(part.lstrip())
            flat = []
            for s, e, t0, t1 in inside:
                flat += [s - lead, e - lead, cs(t0), cs(t1)]
            block = {"p": n, "text": part.strip(), "t": [flat[2], flat[-1]], "w": flat}
            if first and breaks[n]:
                block["brk"] = 1  # a scene break comes before this paragraph
            first = False
            blocks.append(block)
    return blocks


def build_map(book_id: str) -> Path:
    """Write readalong.json from the cached transcripts and the current text and
    chapter titles. A chapter without text (turned off, or only music) is just
    its heading, so the reader still shows where it is."""
    cs = lambda seconds: int(round(seconds * 100))
    text_dir = config.BOOKS_DIR / book_id / "text"
    out, total, model = [], 0.0, None
    for ch in db.chapters(book_id):
        start = ch["start"] or 0.0
        end = start + (ch["duration"] or 0.0)
        total = max(total, end)
        entry = {"idx": ch["idx"], "title": ch["title"], "start": cs(start), "end": cs(end),
                 "heading": [cs(start), cs(end)], "blocks": []}
        data = load(book_id, ch["idx"]) if ch["include"] else None
        text_path = text_dir / f"{ch['idx']:04d}.txt"
        if data and data.get("words") and text_path.exists():
            model = model or data.get("model")
            words = data["words"]
            h = heading_count(words, ch["title"], data.get("quiet"))
            entry["heading"] = [cs(words[0][1]), cs(words[h - 1][2])] if h else None
            marks = [[w[0].strip(), w[1], w[2]] for w in words[h:]]
            entry["blocks"] = chapter_blocks(text_path.read_text(encoding="utf-8"), marks, start, end)
            if not entry["blocks"] and not h:
                entry["heading"] = [cs(start), cs(end)]
        out.append(entry)
    doc = {"v": readalong.VERSION, "duration": round(total, 2), "estimated_units": 0,
           "transcribed": model or config.TRANSCRIBE_MODEL, "chapters": out}
    target = readalong.path_for(book_id)
    tmp = target.with_suffix(".part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(target)
    log.info("Read along for %s: %d chapters, %d blocks, from Whisper", book_id, len(out),
             sum(len(c["blocks"]) for c in out))
    return target


def has_all(book_id: str) -> bool:
    """Whether every turned-on chapter has been transcribed."""
    return all(transcript_path(book_id, c["idx"]).exists() for c in db.chapters(book_id) if c["include"])
