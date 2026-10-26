"""Measuring a finished voice against the narrator.

The held-out clips were never trained on. The new voice says the same
sentences; Whisper transcribes what it said (clarity: the share of words
misheard), and speaker fingerprints compare it with the real clips
(likeness). The starting voice is measured the same way, and the narrator's
own clips against each other, so the numbers have something to stand next to.
Runs on the CPU: it's a few dozen sentences."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import config, dataset, media, speaker

log = logging.getLogger("voicelab.quality")

TIERS = ["Poor", "Fair", "Good", "Great"]


def synthesize(voice, text: str) -> np.ndarray:
    from piper import SynthesisConfig
    chunks = [np.asarray(c.audio_float_array, np.float32)
              for c in voice.synthesize(text, syn_config=SynthesisConfig(volume=0.9))]
    return np.concatenate(chunks) if chunks else np.zeros(1, np.float32)


def artifacts(clips: list[np.ndarray], rate: int) -> float:
    """Distortion, in dB: tones at fixed pitches in the long-term spectrum
    (1.5-10 kHz), the buzz of an undertrained voice. Real speech has none to
    speak of: its harmonics follow the pitch and average out, so the averaged
    spectrum is smooth. Measured: narrators' own clips 2-3, stock voices 2-3,
    a narrator's voice after 6,000 steps 4.4, another's after 450 steps 11."""
    from scipy.ndimage import median_filter
    n_fft, hop = 2048, 512
    total, frames_used = None, 0
    for audio in clips:
        n = 1 + (len(audio) - n_fft) // hop
        if n < 1:
            continue
        frames = np.stack([audio[i * hop:i * hop + n_fft] for i in range(n)]) * np.hanning(n_fft)
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        energy = power.sum(axis=1)
        loud = energy > energy.max() * 1e-3  # speech, not the pauses
        total = power[loud].sum(axis=0) if total is None else total + power[loud].sum(axis=0)
        frames_used += int(loud.sum())
    if not frames_used:
        return 0.0
    db = 10 * np.log10(total / frames_used + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    peaks = (db - median_filter(db, size=31))[(freqs > 1500) & (freqs < 10000)]  # above the spectrum's shape
    return round(float(np.sort(peaks)[-10:].mean()), 2)


def transcribe(path: Path) -> str:
    from .analyze import whisper
    segments, _ = whisper().transcribe(str(path), language="en", beam_size=5, condition_on_previous_text=False)
    return " ".join(s.text for s in segments)


def clarity_tier(wer: float) -> int:
    return 3 if wer <= 0.05 else 2 if wer <= 0.12 else 1 if wer <= 0.25 else 0


def likeness_tier(sim: float) -> int:
    return 3 if sim >= 0.75 else 2 if sim >= 0.6 else 1 if sim >= 0.45 else 0


def clean_tier(excess: float) -> int:
    """Distortion beyond the narrator's own recordings, in dB."""
    return 3 if excess <= 0.5 else 2 if excess <= 1.5 else 1 if excess <= 3 else 0


def verdict(wer: float, like: float, base_like: float | None, usable_minutes: float,
            excess: float | None = None) -> dict:
    c, k = clarity_tier(wer), likeness_tier(like)
    d = clean_tier(excess) if excess is not None else 3
    clear = ["Often hard to understand", "Understandable, with slips", "Mostly clear speech", "Clear speech"][c]
    overall = TIERS[min(c, k, d)]
    buzz = [", with heavy distortion", ", with some distortion", ", slightly distorted", ""][d]
    text = (f"{clear} ({round(wer * 100)}% misheard){buzz}, sounds like the narrator "
            f"{round(like * 100)}% — {overall}.")
    advice = []
    if d < 2:
        advice.append("The distortion fades with more training: Train further.")
    if k < 3 and usable_minutes < config.READY_GREAT:
        advice.append("More audio would most likely improve likeness.")
    elif k < 3:
        advice.append("Training longer would most likely improve likeness.")
    if c < 2:
        advice.append("Clearer recordings, or training longer, would most likely help clarity.")
    if base_like is not None and like < base_like + 0.05:
        advice.append("It has barely moved from the starting voice: train longer or add more audio.")
    return {"overall": overall, "text": " ".join([text, *advice]), "advice": advice}


def check(project: dict, onnx: Path, round_no: int, report=lambda text: None) -> dict:
    from app.qa import word_error_rate
    from app.tts import piper_model_path
    from piper.voice import PiperVoice

    base_dir = dataset.project_dir(project["id"])
    hold_dir = base_dir / "holdout"
    items = json.loads((hold_dir / "holdout.json").read_text(encoding="utf-8")) if (
        hold_dir / "holdout.json").exists() else []
    items = [i for i in items if (hold_dir / i["file"]).exists()]
    if not items:
        return {"skipped": True, "text": "There were too few clips to keep some aside, so there’s no measured report. "
                                         "Listen to a sample in Forge instead."}
    out = base_dir / "quality" / f"round-{round_no}"
    (out / "new").mkdir(parents=True, exist_ok=True)
    (out / "base").mkdir(parents=True, exist_ok=True)
    start = config.start_of(project)
    new_voice = PiperVoice.load(str(onnx), use_cuda=False)
    try:
        base_voice = PiperVoice.load(piper_model_path(start["voice"]), use_cuda=False)
    except Exception:  # offline: the report still works without the comparison
        log.warning("Couldn't load the starting voice for comparison", exc_info=True)
        base_voice = None

    rows = []
    for n, item in enumerate(items, 1):
        report(f"Checking the new voice: sentence {n} of {len(items)}")
        new_path = out / "new" / item["file"]
        media.write_wav(new_path, synthesize(new_voice, item["text"]), new_voice.config.sample_rate)
        heard = transcribe(new_path)
        row = {"id": item["id"], "text": item["text"], "file": item["file"], "heard": heard.strip(),
               "wer": round(word_error_rate(item["text"], heard), 3)}
        if base_voice is not None:
            base_path = out / "base" / item["file"]
            media.write_wav(base_path, synthesize(base_voice, item["text"]), base_voice.config.sample_rate)
            row["base_wer"] = round(word_error_rate(item["text"], transcribe(base_path)), 3)
        rows.append(row)

    report("Comparing the new voice with the narrator")

    def load16(path: Path) -> np.ndarray:
        import soundfile as sf
        audio, rate = sf.read(str(path), dtype="float32")
        return media.resample(audio, rate, speaker.RATE)

    real = speaker.embed([load16(hold_dir / r["file"]) for r in rows])
    new = speaker.embed([load16(out / "new" / r["file"]) for r in rows])
    center = real.mean(axis=0)
    center /= max(np.linalg.norm(center), 1e-9)
    like = new @ center
    # The narrator against their own other clips: the most a voice could score.
    ceiling = [float(real[i] @ _unit(np.delete(real, i, axis=0).mean(axis=0))) for i in range(len(real))] \
        if len(real) > 1 else [1.0]
    base_like = None
    if base_voice is not None:
        base = speaker.embed([load16(out / "base" / r["file"]) for r in rows])
        base_like = base @ center
    for i, row in enumerate(rows):
        row["likeness"] = round(float(like[i]), 3)
        if base_like is not None:
            row["base_likeness"] = round(float(base_like[i]), 3)

    wer = float(np.mean([r["wer"] for r in rows]))
    likeness = float(np.mean(like))
    usable = float((project.get("readiness") or {}).get("usable_minutes") or 0)

    def clips(folder: Path) -> list[np.ndarray]:
        return [_load(folder / r["file"]) for r in rows]

    real_art = artifacts(clips(hold_dir), config.SAMPLE_RATE)
    new_art = artifacts(clips(out / "new"), config.SAMPLE_RATE)
    result = {
        "round": round_no, "sentences": len(rows), "wer": round(wer, 3), "likeness": round(likeness, 3),
        "real_likeness": round(float(np.mean(ceiling)), 3),
        "artifacts": new_art, "real_artifacts": real_art,
        "base_artifacts": artifacts(clips(out / "base"), config.SAMPLE_RATE) if base_voice is not None else None,
        "base_wer": round(float(np.mean([r["base_wer"] for r in rows])), 3) if base_voice is not None else None,
        "base_likeness": round(float(np.mean(base_like)), 3) if base_like is not None else None,
        "base_name": start["name"], "rows": rows,
    }
    result.update(verdict(wer, likeness, result["base_likeness"], usable, excess=new_art - real_art))
    return result


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-9)


def _load(path: Path, rate: int = config.SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf
    audio, have = sf.read(str(path), dtype="float32")
    return media.resample(audio, have, rate) if have != rate else audio


def _load16(path: Path) -> np.ndarray:
    return _load(path, speaker.RATE)


class Probe:
    """Likeness and distortion while training, the same measures as the final
    report: the voice so far says held-out sentences, its fingerprint is
    compared with the narrator's real clips, and its spectrum is checked for
    buzz. Training stops once neither improves any more, rather than on
    validation loss, which says nothing direct about either.
    CPU only: it runs inside the training process, beside the GPU's capped memory."""

    def __init__(self, project_id: str, sentences: int):
        hold_dir = dataset.project_dir(project_id) / "holdout"
        items = json.loads((hold_dir / "holdout.json").read_text(encoding="utf-8")) if (
            hold_dir / "holdout.json").exists() else []
        items = [i for i in items if (hold_dir / i["file"]).exists()]
        self.items = items[:sentences]
        self.center = None
        self.ceiling = None
        self.real_artifacts = None
        if len(items) >= 3:
            real = speaker.embed([_load16(hold_dir / i["file"]) for i in items])
            self.center = _unit(real.mean(axis=0))
            # The narrator against their own other clips: what a perfect copy would score.
            self.ceiling = float(np.mean([real[i] @ _unit(np.delete(real, i, axis=0).mean(axis=0))
                                          for i in range(len(real))]))
            self.real_artifacts = artifacts([_load(hold_dir / i["file"]) for i in self.items], config.SAMPLE_RATE)

    def measure(self, onnx: Path, sample_to: Path | None = None) -> dict:
        """{"likeness", "artifacts"} of a voice; sample_to: also write it reading
        the sample passage there, to listen to how training is going."""
        from piper.voice import PiperVoice

        from . import voices
        voice = PiperVoice.load(str(onnx), use_cuda=False)
        rate = voice.config.sample_rate
        said = [synthesize(voice, i["text"]) for i in self.items]
        if sample_to is not None:
            sample_to.parent.mkdir(parents=True, exist_ok=True)
            media.write_wav(sample_to, synthesize(voice, voices.SAMPLE_TEXT), rate)
        like = float(np.mean(speaker.embed([media.resample(a, rate, speaker.RATE) for a in said]) @ self.center))
        return {"likeness": like, "artifacts": artifacts(said, rate)}
