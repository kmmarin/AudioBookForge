"""Narration check with Whisper.

Each work unit has known source text, so Whisper only has to transcribe a
sample of units and compare word-for-word. No alignment step is needed, and
checking a quarter of the units keeps the cost low.
"""
from __future__ import annotations

import re
import threading

from . import config

_model = None
_lock = threading.Lock()
_users = 0  # books checking right now; on the GPU the model is dropped when it reaches 0


class session:
    """Hold Whisper for one book's check. On the GPU it's unloaded when no book
    is being checked, so it doesn't sit in video memory the voices need."""

    def __enter__(self):
        global _users
        with _lock:
            _users += 1
        return self

    def __exit__(self, *exc):
        global _users, _model
        with _lock:
            _users -= 1
            if _users == 0 and config.WHISPER_DEVICE == "cuda" and _model is not None:
                _model = None
                import gc
                gc.collect()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            compute = "int8" if config.WHISPER_DEVICE == "cpu" else "float16"
            _model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                                  compute_type=compute, cpu_threads=config.WHISPER_THREADS,
                                  download_root=str(config.MODELS_DIR / "whisper"))
    return _model


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# Narration expands these (textprep.ABBREVIATIONS); Whisper writes them short.
SAME_WORD = {"mister": "mr", "missus": "mrs", "miz": "ms", "doctor": "dr", "professor": "prof"}


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return ONES[n // 100] + " hundred" + (" " + number_words(rest) if rest else "")
    if n < 1_000_000:
        rest = n % 1000
        return number_words(n // 1000) + " thousand" + (" " + number_words(rest) if rest else "")
    return str(n)


def _normalize(text: str) -> str:
    """Lowercase words only, so formatting differences don't count as misreads."""
    from .textprep import SOUND_WORDS, _respell
    for pattern, word in SOUND_WORDS:  # Whisper writes "Shh" for the narrated "shush"
        text = _respell(pattern, word, text)
    text = re.sub(r"[’']", "", text.lower())
    text = re.sub(r"\b(\w{3,})our\b", r"\1or", text)  # British voices: harbour = harbor
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,000 -> 1000
    text = re.sub(r"\d+", lambda m: f" {number_words(int(m.group()))} ", text)
    words = re.sub(r"[^\w\s]", " ", text).replace("_", " ").split()
    return " ".join(SAME_WORD.get(w, w) for w in words if w != "and")


def sample_indexes(count: int, fraction: float) -> list[int]:
    if count <= 0:
        return []
    k = min(count, max(1, round(count * fraction)))
    if k == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (k - 1)) for i in range(k)})


def word_error_rate(expected: str, heard: str) -> float:
    import jiwer
    ref, hyp = _normalize(expected), _normalize(heard)
    if not ref:
        return 0.0
    return min(1.0, jiwer.wer(ref, hyp))


def transcribe(path, language: str | None) -> str:
    model = _get_model()
    with _lock:  # one transcription at a time; Whisper already uses several threads
        # Keep timestamps on and use beam search: without timestamps, or with
        # greedy decoding after a pause (like the one after a chapter title),
        # Whisper ends the 30-second window early and skips the rest, which
        # reads as dropped narration.
        segments, _ = model.transcribe(str(path), language=language, beam_size=5,
                                       condition_on_previous_text=False)
        return " ".join(segment.text for segment in segments)


def score_unit(path, expected: str, language: str | None) -> float:
    if config.DEV_FAKE_TTS:
        return 0.0
    return word_error_rate(expected, transcribe(path, language))
