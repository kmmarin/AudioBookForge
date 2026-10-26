"""Reading recordings: what ffprobe says about them, who narrates them, and
their audio, decoded a window at a time so a whole audiobook never has to
fit in memory or be converted in one go."""
from __future__ import annotations

import json
import re
import subprocess

import numpy as np

from . import config

# Tag names that hold the narrator, best first. Forge and Audible-style files
# put the narrator in composer; MP3s may carry a TXXX:NARRATOR frame.
NARRATOR_TAGS = ("narrator", "narrated_by", "narratedby", "©nrt", "nrt", "reader", "read_by", "composer", "performer")
TEXT_TAGS = ("artist", "album_artist", "comment", "description", "synopsis", "subtitle", "album", "title")
# A name: capitalised words or initials ("J. R. Smith"), and the small words some names have.
_WORD = r"[A-Z](?:\.|[\w’'-]*)"
NAME = rf"({_WORD}(?:\s+(?:{_WORD}|(?:de|der|den|van|von|da|del|la|le)\b)){{0,4}})"
NARRATED_RE = re.compile(r"\b(?i:narrated|read|performed|voiced)\s+(?i:by)[:\s]+" + NAME)
NOT_NAMES = {"The", "This", "A", "An", "Audible", "Chapter", "Me", "Him", "Her", "Them"}


CREDITS_RE = re.compile(r"\b(?:opening|closing|end|ending|final|beginning)\s+credits?\b|^\s*credits?\s*$", re.I)
INTRO_RE = re.compile(r"credits?|^\s*intro(?:duction)?\b|^\s*(?:about|meet) the (?:narrator|reader)", re.I)


def credit_spans(chapters: list[dict] | None) -> list[tuple[float, float]]:
    """Chapters that are credits ("Opening Credits", "End Credits"): not training material."""
    return [(c["start"], c["end"]) for c in chapters or [] if CREDITS_RE.search(c.get("title") or "")]


def intro_end(chapters: list[dict] | None, at_least: float) -> float:
    """How far into a recording to look for "read by": the opening minutes, or
    to the end of an "Opening Credits" or "Intro" chapter near the start."""
    end = at_least
    for c in (chapters or [])[:3]:
        if INTRO_RE.search(c.get("title") or "") and c["start"] < 600:
            end = max(end, min(c["end"], 900.0))
    return end


def probe(path) -> dict:
    """Duration, format tags (lowercased keys), chapters and the first audio stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_chapters",
                          "-of", "json", str(path)], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise ValueError(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "ffprobe failed")
    data = json.loads(out.stdout or "{}")
    fmt = data.get("format") or {}
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError("there’s no audio in it")
    tags = {k.lower(): str(v) for k, v in (fmt.get("tags") or {}).items()}
    for k, v in (audio.get("tags") or {}).items():
        tags.setdefault(k.lower(), str(v))
    duration = float(fmt.get("duration") or audio.get("duration") or 0)
    chapters = [{"start": round(float(c.get("start_time") or 0), 3), "end": round(float(c.get("end_time") or 0), 3),
                 "title": str((c.get("tags") or {}).get("title") or "")} for c in data.get("chapters", [])]
    return {"duration": duration, "tags": tags, "chapters": chapters, "codec": audio.get("codec_name"),
            "rate": int(audio.get("sample_rate") or 0), "channels": int(audio.get("channels") or 0)}


# Credits run on without a full stop in a transcript ("read by Jane Smith
# Performance Copyright…"): a name ends at the first of these words.
NAME_ENDS = {"performance", "copyright", "produced", "production", "productions", "published", "publishing",
             "publisher", "recorded", "recording", "unabridged", "abridged", "audio", "audiobook", "audiobooks",
             "books", "records", "presented", "presents", "directed", "edited", "chapter", "part", "and", "with",
             "for", "the", "this", "at", "in", "on", "from", "an", "a", "inc", "ltd", "llc"}


def _clean_name(name: str) -> str | None:
    name = re.sub(r"\s+", " ", name or "").strip(" .,;:-–—\"“”'’")
    name = re.sub(r"\s*\((?:narrator|reader)\)$", "", name, flags=re.I)
    words = []
    for word in name.split():
        if word.lower().strip(".,") in NAME_ENDS:
            break
        words.append(word)
    name = " ".join(words).strip(" .,;:")
    if not name or len(name) > 80 or name.split()[0] in NOT_NAMES:
        return None
    return name


def narrator_phrase(text: str) -> tuple[str, str] | None:
    """(name, the words it was found in) for "narrated by Jane Smith" or
    "read by J. R. Smith" in a sentence."""
    for match in NARRATED_RE.finditer(text or ""):
        name = _clean_name(match.group(1))
        if name:
            verb = re.sub(r"\s+", " ", match.group(0)[:match.start(1) - match.start(0)]).strip()
            return name, f"{verb} {name}"
    return None


def narrator_from_text(text: str) -> str | None:
    found = narrator_phrase(text)
    return found[0] if found else None


def narrator_from_tags(tags: dict) -> tuple[str, str] | None:
    """(name, which tag) from the file's tags, if they say who narrates it."""
    for key in NARRATOR_TAGS:
        for tag, value in tags.items():
            if tag == key or tag.endswith(":" + key) or tag.replace(" ", "_") == key:
                # Several narrators: "A, B and C" or "A; B" -> keep the first.
                first = re.split(r"\s*(?:;|/|,|\band\b|&)\s*", value.strip())[0]
                name = _clean_name(first)
                if name:
                    return name, tag
    for key in TEXT_TAGS:
        name = narrator_from_text(tags.get(key, ""))
        if name:
            return name, key
    return None


def accent_from_tags(tags: dict) -> str | None:
    """en-gb when the tags say British English; None when they don't say."""
    text = " ".join(tags.get(k, "") for k in ("language", "lang", "locale", "accent")).lower()
    if re.search(r"\ben[-_](gb|uk)\b|brit|united kingdom", text):
        return "en-gb"
    if re.search(r"\ben[-_]us\b|american", text):
        return "en-us"
    return None


def language_from_tags(tags: dict) -> str | None:
    """ISO 639-1 language if the tags say; "und" and similar count as unknown."""
    raw = (tags.get("language") or tags.get("lang") or "").strip().lower()
    if not raw or raw in {"und", "unknown", "mul", "zxx"}:
        return None
    raw = re.split(r"[-_,; ]", raw)[0]
    three = {"eng": "en", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es", "ita": "it",
             "nld": "nl", "dut": "nl", "por": "pt", "pol": "pl", "rus": "ru", "jpn": "ja", "zho": "zh", "chi": "zh"}
    return three.get(raw, raw if len(raw) == 2 else None)


def decode(path, start: float, seconds: float, rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Mono float32 audio from start for seconds (less at the end of the file)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, start):.3f}", "-t", f"{seconds:.3f}",
           "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=max(120, seconds))
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace").strip()[-300:] or "ffmpeg failed")
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


FP_RATE, FP_FRAME, FP_SECONDS = 8000, 400, 600  # 50 ms loudness steps over the first 10 minutes


def fingerprint(path) -> bytes:
    """The loudness contour of a recording's first ten minutes, one byte per
    50 ms: enough to recognise the same audio again, re-encoded or not."""
    audio = decode(path, 0, FP_SECONDS, rate=FP_RATE)
    usable = len(audio) // FP_FRAME * FP_FRAME
    if not usable:
        return b""
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, FP_FRAME) ** 2, axis=1))
    db = np.clip(20 * np.log10(np.maximum(rms, 1e-5)), -100, 0)
    return np.round(db + 100).astype(np.uint8).tobytes()


def same_start(a: bytes, b: bytes, windows: int = 4, size: int = 400, lag: int = 40) -> bool:
    """Whether two recordings begin with the same audio: 20-second stretches
    of one line up with the other's (within two seconds) almost exactly."""
    x = np.frombuffer(a, np.uint8).astype(np.float32)
    y = np.frombuffer(b, np.uint8).astype(np.float32)
    n = min(len(x), len(y)) - size - 2 * lag
    if n <= 0:
        return False
    scores = []
    for start in np.linspace(lag, lag + n, windows).astype(int):
        w = x[start:start + size]
        if w.std() < 1.0:
            continue  # silence matches anything
        w = (w - w.mean()) / w.std()
        best = -1.0
        for shift in range(-lag, lag + 1):
            v = y[start + shift:start + shift + size]
            if len(v) == size and v.std() >= 1.0:
                best = max(best, float(np.mean(w * (v - v.mean()) / v.std())))
        scores.append(best)
    return len(scores) >= 2 and min(scores) >= 0.9


def write_wav(path, audio: np.ndarray, rate: int = config.SAMPLE_RATE) -> None:
    import os

    import soundfile as sf
    tmp = f"{path}.part"
    sf.write(tmp, np.clip(audio, -1.0, 1.0), rate, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)


def resample(audio: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    import soxr
    return soxr.resample(audio, rate_in, rate_out).astype(np.float32)
