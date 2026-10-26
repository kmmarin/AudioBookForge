"""Turning recordings into clean training clips.

Each recording is read a window at a time (five minutes by default): decoded
to 22.05 kHz mono, transcribed by faster-whisper with word timings, and cut
into 1 to 15 second clips at pauses, ending at sentence ends where possible.
Every clip is measured (background level, clipping, Whisper's confidence, a
speaker fingerprint) and the time between clips is sorted into pauses and
other sounds. Each window is saved in one transaction with the new position,
so a restart carries on with the next window.

Whether a clip is the narrator's own voice is decided later, in readiness.py,
once every recording has been heard: it needs the main voice of all of them.
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np

from . import config, db, media, speaker

log = logging.getLogger("voicelab.analyze")

PRE = 5.0  # seconds decoded before a window, so its first clip isn't cut mid-word
POST = 20.0  # and after it, so a clip starting near its end is complete (> CLIP_MAX)
HARD_GAP = 0.8  # a pause this long always ends a clip
SOFT_GAP = 0.25  # a pause this long is a clean place to cut
SENTENCE_GAP = 0.12  # after a full stop, even a short pause is
NARRATOR_SEARCH = 300  # seconds at the start of the first recording searched for "narrated by"
MUSIC_RE = re.compile(r"[♪♫🎵🎶]|[\[(](?:music|singing|song|applause|laughter|sound|instrumental)[^\])]*[\])]",
                      re.I)
SENTENCE_END = re.compile(r"[.!?…][\"”’')\]]*$")

_whisper = None


def whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        compute = "int8" if config.WHISPER_DEVICE == "cpu" else "float16"
        _whisper = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type=compute,
                                cpu_threads=config.THREADS, download_root=str(config.MODELS_DIR / "whisper"))
    return _whisper


def transcribe(audio16: np.ndarray, language: str | None, on_progress=None) -> tuple[list[dict], object]:
    """Words with times (seconds from the start of audio16) and Whisper's
    confidence, plus the segment details each word came from."""
    segments, info = whisper().transcribe(
        audio16, language=language, word_timestamps=True, vad_filter=True, beam_size=5,
        condition_on_previous_text=False, vad_parameters={"min_silence_duration_ms": 400})
    words = []
    for seg in segments:
        music = bool(MUSIC_RE.search(seg.text))
        for w in seg.words or []:
            words.append({"start": float(w.start), "end": float(w.end), "word": w.word,
                          "prob": float(w.probability), "logprob": float(seg.avg_logprob),
                          "ratio": float(seg.compression_ratio), "music": music})
        if on_progress:
            on_progress(seg.end)
    return words, info


def ends_sentence(word: dict) -> bool:
    return bool(SENTENCE_END.search(word["word"].strip()))


def group_words(words: list[dict]) -> list[tuple[int, int, bool]]:
    """Cut the word list into clips: (first word, last word, forced). Clips end
    at the first sentence end past CLIP_TARGET, always at long pauses, and
    never run past CLIP_MAX; forced means no clean pause was found in time."""
    out, i, n = [], 0, len(words)
    while i < n:
        j, chosen, forced = i, None, False
        last_soft = last_sentence = None
        while j + 1 < n:
            gap = words[j + 1]["start"] - words[j]["end"]
            dur = words[j]["end"] - words[i]["start"]
            if gap >= HARD_GAP or words[j + 1]["music"] != words[j]["music"]:
                chosen = j
                break
            sentence = ends_sentence(words[j])
            if gap >= SOFT_GAP or (sentence and gap >= SENTENCE_GAP):
                if sentence and dur >= config.CLIP_TARGET:
                    chosen = j
                    break
                last_soft = j
                if sentence:
                    last_sentence = j
            if words[j + 1]["end"] - words[i]["start"] > config.CLIP_MAX:
                for k in (last_sentence, last_soft):
                    if k is not None and words[k]["end"] - words[i]["start"] >= max(config.CLIP_MIN, 2.5):
                        chosen = k
                        break
                if chosen is None:
                    chosen, forced = (last_soft, False) if last_soft is not None else (j, True)
                break
            j += 1
        if chosen is None:
            chosen = j
        out.append((i, chosen, forced))
        i = chosen + 1
    return out


def frame_db(audio: np.ndarray, rate: int, frame_s: float) -> np.ndarray:
    frame = max(1, int(rate * frame_s))
    usable = len(audio) // frame * frame
    if usable == 0:
        return np.full(1, -100.0)
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, frame) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-5))


def measure(audio: np.ndarray, rate: int) -> dict:
    """Background level and clipping of one clip (with a little context)."""
    levels = frame_db(audio, rate, 0.02)
    snr = float(np.percentile(levels, 95) - np.percentile(levels, 10)) if len(levels) > 5 else 0.0
    loud = np.abs(audio) >= 0.98
    # Clipping flattens the tops of waves: runs of samples stuck at full scale.
    runs = loud[2:] & loud[1:-1] & loud[:-2] if len(loud) > 2 else np.zeros(0, bool)
    return {"snr": round(snr, 1), "clipped": float(runs.mean()) if len(runs) else 0.0}


def judge(clip: dict, forced: bool, music: bool) -> str | None:
    """Why a clip can't be used, or None. The speaker check comes later."""
    if music:
        return "music"
    if clip["end"] - clip["start"] < config.CLIP_MIN or clip["words"] < config.MIN_WORDS:
        return "too_short"
    if forced:
        return "too_long"
    if clip["clipped"] > config.MAX_CLIPPED:
        return "clipped"
    if clip["snr"] < config.MIN_SNR_DB:
        return "noisy"
    if (clip["conf"] < config.MIN_WORD_PROB or clip["unsure"] > config.MAX_UNSURE_WORDS
            or clip["_logprob"] < -1.0 or clip["_ratio"] > 2.4):
        return "unclear"
    return None


def window_clips(words: list[dict], audio: np.ndarray, offset: float, t0: float, t1: float,
                 last_end: float) -> list[dict]:
    """Clips that start in [t0, t1) of the recording, measured and judged."""
    rate = config.SAMPLE_RATE
    clips = []
    for i, k, forced in group_words(words):
        first, last = words[i], words[k]
        start_abs = offset + first["start"]
        if not (max(t0, last_end) <= start_abs < t1):
            continue
        prev_end = words[i - 1]["end"] if i > 0 else first["start"] - 0.5
        next_start = words[k + 1]["start"] if k + 1 < len(words) else last["end"] + 0.6
        # Cut halfway into the pauses around the clip, at most 0.25 s before and 0.3 s after.
        start = max(first["start"] - 0.25, (prev_end + first["start"]) / 2, 0.0)
        end = min(last["end"] + 0.3, (last["end"] + next_start) / 2, len(audio) / rate)
        chunk = audio[int(start * rate):int(end * rate)]
        context = audio[int(max(0.0, start - 0.3) * rate):int(min(len(audio) / rate, end + 0.3) * rate)]
        span = words[i:k + 1]
        text = re.sub(r"\s+", " ", "".join(w["word"] for w in span)).strip()
        probs = np.array([w["prob"] for w in span])
        clip = {"start": round(offset + start, 3), "end": round(offset + end, 3), "text": text,
                "words": len(span), "conf": round(float(probs.mean()), 3),
                "unsure": round(float((probs < 0.5).mean()), 3),
                "_logprob": min(w["logprob"] for w in span), "_ratio": max(w["ratio"] for w in span),
                "_audio": chunk, **measure(context, rate)}
        clip["reason"] = judge(clip, forced, any(w["music"] for w in span))
        clips.append(clip)
        last_end = clip["end"]
    return clips


def gap_seconds(audio: np.ndarray, offset: float, t0: float, t1: float, covered: list[tuple[float, float]]) -> dict:
    """Time in [t0, t1) outside every clip: quiet pauses, or other sounds
    (music, effects, speech Whisper skipped)."""
    rate, frame_s = config.SAMPLE_RATE, 0.05
    levels = frame_db(audio, rate, frame_s)
    times = offset + (np.arange(len(levels)) + 0.5) * frame_s
    outside = (times >= t0) & (times < t1)
    for a, b in covered:
        outside &= ~((times >= a) & (times < b))
    quiet = outside & (levels < config.SILENCE_DB)
    return {"pause": float(quiet.sum() * frame_s), "other": float((outside & ~quiet).sum() * frame_s)}


def usable_minutes(project_id: str) -> float:
    row = db.conn().execute("SELECT COALESCE(SUM(end - start), 0) FROM clips WHERE project_id = ? AND reason IS NULL",
                            (project_id,)).fetchone()
    return row[0] / 60


_openings: dict[str, list[str]] = {}  # source id -> words of its opening heard so far ("read by")


def analyze_source(project: dict, source: dict, should_stop, report, windows: int = 10**9) -> bool:
    """Work through one recording from where it stopped, at most `windows`
    windows of it. True once it's finished."""
    pid, sid = project["id"], source["id"]
    duration = float(source["duration"] or 0)
    done_until, last_end = float(source["done_until"]), float(source["last_end"])
    gaps = dict(source.get("gaps") or {})
    language = "en"
    # The book's first part (by track number, then name) is where an intro says
    # who reads it, even when that intro is someone else's voice: its opening
    # minutes, or its "Opening Credits" or "Intro" chapter.
    first_part = next((s["id"] for s in db.sources(pid) if s["status"] not in ("failed", "duplicate")), None) == sid
    chapters = source.get("chapters") or []
    credits = media.credit_spans(chapters)  # "Opening Credits", "End Credits": never training material
    search_end = media.intro_end(chapters, NARRATOR_SEARCH)
    opening = _openings.setdefault(sid, [])
    while done_until < duration - 0.5:
        if should_stop():
            return False
        if usable_minutes(pid) >= config.ANALYZE_ENOUGH_MINUTES:
            gaps["skipped"] = gaps.get("skipped", 0) + duration - done_until
            db.add_clips(pid, sid, [], duration, last_end, gaps)
            log.info("%s: enough usable audio, skipping the rest of %s", pid, source["filename"])
            break
        t0 = done_until
        t1 = duration if duration - t0 < config.WINDOW_SECONDS * 1.2 else t0 + config.WINDOW_SECONDS
        a0, a1 = max(0.0, t0 - PRE), min(duration, t1 + POST)
        started = time.monotonic()
        audio = media.decode(source["path"], a0, a1 - a0)
        audio16 = media.resample(audio, config.SAMPLE_RATE, speaker.RATE)
        report(sid, t0, 0.0)
        detect = t0 == 0
        words, info = transcribe(audio16, None if detect else language,
                                 on_progress=lambda s: report(sid, t0, max(0.0, s - (t0 - a0))))
        if detect and info.language != "en" and info.language_probability > 0.7:
            raise ValueError(f"This recording sounds like {info.language.upper()} rather than English. "
                             "Only English voices can be made for now.")
        clips = window_clips(words, audio, a0, t0, t1, last_end)
        for clip in clips:
            middle = (clip["start"] + clip["end"]) / 2
            if any(a <= middle < b for a, b in credits):
                clip["reason"] = "credits"
        good = [c for c in clips if c["reason"] is None]
        for clip, emb in zip(good, speaker.embed([media.resample(speaker.middle(c["_audio"], config.SAMPLE_RATE),
                                                                 config.SAMPLE_RATE, speaker.RATE) for c in good])):
            clip["embedding"] = speaker.to_blob(emb)
        covered = [(c["start"], c["end"]) for c in clips] + ([(t0, last_end)] if last_end > t0 else [])
        for key, value in gap_seconds(audio, a0, t0, t1, covered).items():
            gaps[key] = round(gaps.get(key, 0) + value, 2)
        if clips:
            last_end = max(last_end, clips[-1]["end"])
        fresh = db.get_project(pid)
        found_here = fresh["name_source"] == "whisper" and source["filename"] in (fresh["name_note"] or "")
        if first_part and t0 < search_end and fresh["name_source"] not in ("user", "tags") and not found_here:
            # A name typed by the person (or found in the tags) is never second-guessed;
            # one heard in a later part gives way to what the first part says.
            opening.append("".join(w["word"] for w in words if t0 <= a0 + w["start"] < min(t1, search_end)))
            found = media.narrator_phrase(" ".join(opening))
            if found:
                name, phrase = found
                db.update_project(pid, name=name, name_source="whisper",
                                  name_note=f"Heard at the start of {source['filename']}: “{phrase}”")
            elif t1 >= min(search_end, duration) and not fresh["name"]:
                db.update_project(pid, name_note="No narrator was named at the start. Add the book’s first "
                                                 "part to find the narrator’s name, or type it")
        db.add_clips(pid, sid, [{k: v for k, v in c.items() if not k.startswith("_")} for c in clips],
                     t1, last_end, gaps)
        done_until = t1
        log.info("%s %s: %.0f-%.0f s, %d clips (%d usable) in %.0f s", pid, source["filename"], t0, t1,
                 len(clips), len(good), time.monotonic() - started)
        windows -= 1
        if windows <= 0 and done_until < duration - 0.5:
            return False  # the caller picks what to hear next: an earlier part may have arrived
    db.update_source(sid, status="done", done_until=duration)
    _openings.pop(sid, None)
    return True
