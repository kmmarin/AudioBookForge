"""Read along: where every word of a finished book is in its audio file.

Built when a book is assembled, from the word timings each unit recorded while
it was voiced (tts.synth_unit). Saved as /data/books/<id>/readalong.json:

    {"v": 1, "duration": 10046.2,
     "chapters": [{"idx": 4, "title": "1. Why Are You Running?", "start": 19840, "end": 90030,
                   "heading": [19840, 21490],
                   "blocks": [{"p": 0, "text": "…", "t": [21500, 29870],
                               "w": [0, 4, 21500, 21730, 5, 11, 21730, 22010, …]}]}]}

Times are centiseconds into the file. "w" holds four numbers per word: where it
starts and ends in "text", and when it starts and ends in the audio. Blocks are
a few sentences at most, so the reader never faces a wall of text; "p" numbers
the paragraph a block came from, so blocks of one paragraph stay together.
"""
from __future__ import annotations

import json
import logging
import re
import wave
from difflib import SequenceMatcher
from pathlib import Path

from . import config, tts

log = logging.getLogger("forge.readalong")
VERSION = 1
BLOCK_CHARS = 330  # about four or five lines in the reader
BLOCK_MAX = 430  # a paragraph up to this long stays whole
SENTENCE_END_RE = re.compile(r"[.!?…][\"”’')\]]*\s+(?=\S)")
WORD_RE = re.compile(r"[^\s—–―]+")  # dashes part words ("book—some"); hyphens don't


def path_for(book_id: str) -> Path:
    return config.BOOKS_DIR / book_id / "readalong.json"


def _key(word: str) -> str:
    return re.sub(r"[^0-9a-z]", "", word.lower())


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _estimate_says(items: list, seconds: float) -> list[dict]:
    """Timings for a unit voiced before units recorded them: pauses are known
    exactly, speech shares the rest by length."""
    pauses = sum(float(v) for k, v in items if k != "say")
    says = [v for k, v in items if k == "say"]
    speech = max(0.0, seconds - pauses)
    chars = sum(len(s) for s in says) or 1
    out, t = [], 0.0
    for kind, value in items:
        if kind == "say":
            span = speech * len(value) / chars
            out.append({"start": t, "end": t + span, "words": tts.spread_marks(value, t, t + span)})
            t += span
        else:
            t += float(value)
    return out


def _blocks(text: str) -> list[tuple[int, int]]:
    """Cut a paragraph into blocks of whole sentences, about BLOCK_CHARS each."""
    if len(text) <= BLOCK_MAX:
        return [(0, len(text))]
    cuts = [m.end() for m in SENTENCE_END_RE.finditer(text)]
    blocks, start = [], 0
    for cut in cuts:
        if cut - start >= BLOCK_CHARS and len(text) - cut > 80:
            blocks.append((start, cut))
            start = cut
    blocks.append((start, len(text)))
    return [(a, b) for a, b in blocks if text[a:b].strip()]


def _align(text: str, marks: list, start: float, end: float) -> list[tuple[int, int, float, float]]:
    """Give every printed word a time, from the spoken words' times.

    The spoken text differs a little from the printed one ("Missus" for "Mrs.",
    "shush" for "Shhh"), so they're matched with a diff: matching words take
    their own times, a changed run shares the times of what replaced it, and a
    printed word that wasn't spoken sits at the moment between its neighbours."""
    tokens = [(m.start(), m.end(), _key(m.group())) for m in WORD_RE.finditer(text)]
    tokens = [t for t in tokens if t[2]]  # a lone dash or quote isn't a word to follow
    if not tokens:
        return []
    marks = [m for m in marks if _key(m[0])]
    if not marks:
        marks = tts.spread_marks(text, start, end)
    spoken = [_key(m[0]) for m in marks]
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    matcher = SequenceMatcher(None, spoken, [t[2] for t in tokens], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(j2 - j1):
                times[j1 + k] = (marks[i1 + k][1], marks[i1 + k][2])
        elif op == "replace":
            t0, t1 = marks[i1][1], marks[i2 - 1][2]
            weights = [len(tokens[j][2]) + 1 for j in range(j1, j2)]
            total, t = sum(weights), t0
            for j, w in zip(range(j1, j2), weights):
                span = (t1 - t0) * w / total
                times[j] = (t, t + span)
                t += span
        elif op == "insert":
            at = marks[i1 - 1][2] if i1 > 0 else start
            for j in range(j1, j2):
                times[j] = (at, at)
    return [(a, b, *times[n]) for n, (a, b, _) in enumerate(tokens)]


def build(book_id: str, chapters: list[dict], unit_paths: dict[int, list[Path]],
          unit_items: dict[int, list], unit_sources: dict[int, list],
          display: dict[int, list[str]], durations: list[float]) -> Path:
    """Write the book's readalong.json. `durations` are the chapters' lengths in
    the finished file, so chapter starts match what the player plays."""
    cs = lambda seconds: int(round(seconds * 100))
    out_chapters = []
    chapter_start = 0.0
    estimated = 0
    for ch, length in zip(chapters, durations):
        idx = ch["idx"]
        paras: dict[int, dict] = {}
        heading = None
        scene_break = False
        t = chapter_start
        for path, items, sources in zip(unit_paths[idx], unit_items[idx], unit_sources[idx]):
            seconds = _wav_seconds(path)
            timing = Path(tts.timing_path(path))
            says = None
            if timing.exists():
                says = json.loads(timing.read_text(encoding="utf-8")).get("says")
            if says is None or len(says) != sum(1 for k, _ in items if k == "say"):
                says = _estimate_says(items, seconds)
                estimated += 1
            say_n = 0
            for (kind, value), src in zip(items, sources):
                if kind != "say":
                    if float(value) == config.SCENE_PAUSE:
                        scene_break = True
                    continue
                s = says[say_n]
                say_n += 1
                words = [[w, t + a, t + b] for w, a, b in s["words"]]
                if src == -1:
                    heading = [t + s["start"], t + s["end"]]
                    continue
                p = paras.setdefault(src, {"start": t + s["start"], "end": t + s["end"], "words": [],
                                           "brk": scene_break})
                scene_break = False
                p["end"] = t + s["end"]
                p["words"] += words
            t += seconds
        blocks = []
        for n in sorted(paras):
            p, text = paras[n], display[idx][n]
            aligned = _align(text, p["words"], p["start"], p["end"])
            first = True
            for a, b in _blocks(text):
                inside = [(s - a, e - a, t0, t1) for s, e, t0, t1 in aligned if s >= a and e <= b]
                if not inside:
                    continue
                part = text[a:b]
                lead = len(part) - len(part.lstrip())
                flat = []
                for s, e, t0, t1 in inside:
                    flat += [s - lead, e - lead, cs(t0), cs(t1)]
                block = {"p": n, "text": part.strip(), "t": [flat[2], flat[-1]], "w": flat}
                if first and p["brk"]:
                    block["brk"] = 1  # a scene break comes before this paragraph
                first = False
                blocks.append(block)
        out_chapters.append({
            "idx": idx, "title": ch["title"], "start": cs(chapter_start),
            "end": cs(chapter_start + length),
            "heading": [cs(heading[0]), cs(heading[1])] if heading else None,
            "blocks": blocks,
        })
        chapter_start += length
    doc = {"v": VERSION, "duration": round(chapter_start, 2), "estimated_units": estimated,
           "chapters": out_chapters}
    target = path_for(book_id)
    tmp = target.with_suffix(".part")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(target)
    log.info("Read along for %s: %d chapters, %d blocks, %d units estimated",
             book_id, len(out_chapters), sum(len(c["blocks"]) for c in out_chapters), estimated)
    return target
