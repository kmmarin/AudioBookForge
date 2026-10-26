"""Speech synthesis. Runs inside worker processes; heavy imports stay lazy
so each process only loads what its voices need."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import soundfile as sf

from . import config

KOKORO_REPO = "hexgrad/Kokoro-82M"
PIPER_REPO = "rhasspy/piper-voices"
PIPER_ID_RE = re.compile(r"^([a-z]{2,3})_([A-Za-z]{2})-(.+)-(x_low|low|medium|high)$")

_engines: dict[tuple, object] = {}
_device = "cpu"
_gpu_limit_bytes = 0  # this process's share of video memory; 0 = no cap
CONTEXT_BYTES = int(0.15 * 1024**3)  # CUDA context + workspaces: ~0.08 GB measured under WSL


class GpuMemoryError(RuntimeError):
    """A unit ran out of video memory. The worker puts it back in the queue."""


def init_worker(threads: int, device: str, gpu_limit_gb: float = 0.0) -> None:
    """Process-pool initializer: pin math libraries to a fixed thread budget and,
    on the GPU, cap this process's video memory so it fails cleanly instead of
    pushing the driver over the edge."""
    global _device, _gpu_limit_bytes
    _device = device
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if device == "cuda" and gpu_limit_gb > 0 and torch.cuda.is_available():
            _gpu_limit_bytes = int(gpu_limit_gb * 1024**3)
            # The CUDA context and library workspaces sit outside PyTorch's allocator.
            usable = max(0.25 * 1024**3, _gpu_limit_bytes - CONTEXT_BYTES)
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, usable / total))
    except ImportError:
        pass


def _free_gpu() -> None:
    import gc
    gc.collect()
    if _device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def _is_gpu_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return ("out of memory" in text or "failed to allocate memory" in text
            or "is smaller than requested bytes" in text  # onnxruntime's arena hit its cap
            or "bfc_arena" in text or ("cuda" in text and "alloc" in text))


# Every engine's synth() returns (audio, marks): marks are [word, start, end] in
# seconds from the start of that audio. Read along uses them to follow the text.
Marks = list[list]


class FakeEngine:
    """Tones instead of speech, for testing the pipeline without models."""
    sample_rate = 24000

    def synth(self, text: str, voice: dict, speed: float) -> tuple[np.ndarray, Marks]:
        seconds = max(0.2, len(text.split()) / (2.6 * speed))
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), spread_marks(text, 0.0, seconds)


class KokoroEngine:
    sample_rate = 24000
    _model = None  # shared by every language pipeline in this process
    _packs: dict[tuple[str, int], object] = {}  # voicepacks made for a narrator, by file and version

    def __init__(self, lang: str):
        from kokoro import KModel, KPipeline
        if KokoroEngine._model is None:
            KokoroEngine._model = KModel(repo_id=KOKORO_REPO).to(_device).eval()
        self.pipeline = KPipeline(lang_code=lang, repo_id=KOKORO_REPO, model=KokoroEngine._model)

    def _voice(self, voice: dict):
        """A stock voice's name, or a narrator's voicepack loaded here: Kokoro would
        keep the first version of a file for good, and this one can be improved."""
        if not voice["voice"].endswith(".pt"):
            return voice["voice"]
        key = (voice["voice"], voice.get("rev") or 0)
        if key not in KokoroEngine._packs:
            import torch
            KokoroEngine._packs = {k: v for k, v in KokoroEngine._packs.items() if k[0] != key[0]}
            KokoroEngine._packs[key] = torch.load(voice["voice"], map_location="cpu", weights_only=True).float()
        return KokoroEngine._packs[key]

    def synth(self, text: str, voice: dict, speed: float) -> tuple[np.ndarray, Marks]:
        # Kokoro cuts long paragraphs into chunks (often at a comma) and pads each
        # with ~0.3 s of silence in front and ~0.5 s behind, so joins came out as
        # 0.9 s gaps wherever the cut fell. Trim the padding and pause by punctuation.
        from .textprep import sentences
        rate = self.sample_rate
        pieces = []
        # One sentence per chunk: inside a chunk Kokoro's pause after a period is
        # sometimes no longer than after a comma. Separate chunks get a proper
        # sentence pause from join_pieces every time.
        text = "\n".join(sentences(text))
        for result in self.pipeline(text, voice=self._voice(voice), speed=speed, split_pattern=r"\n+"):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, tuple):
                audio = result[2]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32)
            graphemes = getattr(result, "graphemes", None) or (result[0] if isinstance(result, tuple) else "")
            start, end = speech_bounds(audio, rate)
            # Kokoro times every word; shift them to the trimmed audio.
            marks = [[t.text, max(0.0, t.start_ts - start / rate), max(0.0, t.end_ts - start / rate)]
                     for t in (getattr(result, "tokens", None) or [])
                     if t.start_ts is not None and t.end_ts is not None and re.search(r"\w", t.text)]
            if not marks:
                marks = spread_marks(graphemes, 0.0, (end - start) / rate)
            pieces.append((audio[start:end], graphemes, marks))
        return join_pieces(pieces, rate, speed)


def piper_model_path(model_id: str) -> str:
    if model_id.endswith(".onnx") and os.path.isabs(model_id):
        return model_id  # a voice made with the voice creator, already on disk
    match = PIPER_ID_RE.match(model_id)
    if not match:
        raise ValueError(f"Not a Piper voice id: {model_id}")
    lang, region, name, quality = match.groups()
    rel = f"{lang}/{lang}_{region}/{name}/{quality}/{model_id}.onnx"
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(PIPER_REPO, rel)
    hf_hub_download(PIPER_REPO, rel + ".json")
    return path


class PiperEngine:
    def __init__(self, model_id: str):
        from piper.voice import ESPEAK_DATA_DIR, PiperConfig, PiperVoice
        path = piper_model_path(model_id)
        self.model_path = path
        self.cpu_voice = None  # loaded only if a sentence is too long for the GPU share
        if _device == "cuda":
            # Built directly: PiperVoice.load would first load the model on the CPU too.
            import json
            from pathlib import Path
            with open(f"{path}.json", encoding="utf-8") as f:
                cfg = PiperConfig.from_dict(json.load(f))
            self.voice = PiperVoice(config=cfg, session=self._cuda_session(path),
                                    espeak_data_dir=Path(ESPEAK_DATA_DIR))
        else:
            self.voice = PiperVoice.load(path, use_cuda=False)
        self.sample_rate = self.voice.config.sample_rate
        self.speakers = dict(getattr(self.voice.config, "speaker_id_map", None) or {})

    @staticmethod
    def _cuda_session(path: str):
        """Piper's own loader can't cap memory; onnxruntime's default arena grows
        in powers of two and never shrinks."""
        import onnxruntime
        options = {"cudnn_conv_algo_search": "HEURISTIC",
                   "arena_extend_strategy": "kSameAsRequested"}
        if _gpu_limit_bytes:
            options["gpu_mem_limit"] = str(max(256 * 1024**2, _gpu_limit_bytes - CONTEXT_BYTES))
        return onnxruntime.InferenceSession(path, sess_options=onnxruntime.SessionOptions(),
                                            providers=[("CUDAExecutionProvider", options),
                                                       "CPUExecutionProvider"])

    def synth(self, text: str, voice: dict, speed: float) -> tuple[np.ndarray, Marks]:
        from piper import SynthesisConfig
        speaker_id = None
        if voice.get("speaker"):
            if voice["speaker"] not in self.speakers:
                raise ValueError(f"{voice['model']} has no speaker named {voice['speaker']}. "
                                 f"It has: {', '.join(self.speakers) or 'one speaker only'}.")
            speaker_id = self.speakers[voice["speaker"]]
        elif self.speakers:
            speaker_id = 0
        cfg = SynthesisConfig(speaker_id=speaker_id, length_scale=1.0 / max(speed, 0.1), volume=0.85)
        try:
            return self._synth_with(self.voice, text, cfg, speed)
        except Exception as exc:
            if _device != "cuda" or not _is_gpu_oom(exc):
                raise
            # Piper voices a whole sentence at once, so a very long one can need
            # more than this worker's share of video memory. Voice this text on
            # the CPU instead of failing the book.
            if self.cpu_voice is None:
                from piper.voice import PiperVoice
                self.cpu_voice = PiperVoice.load(self.model_path, use_cuda=False)
            return self._synth_with(self.cpu_voice, text, cfg, speed)

    def _synth_with(self, piper_voice, text: str, cfg, speed: float) -> tuple[np.ndarray, Marks]:
        """One sentence at a time, so each sentence's start and end are exact for
        Read along; Piper has no word timings, so words are spread within it."""
        from .textprep import sentences
        rate = self.sample_rate
        pieces = []
        for sentence in sentences(text):
            chunks = [np.asarray(c.audio_float_array, dtype=np.float32)
                      for c in piper_voice.synthesize(sentence, syn_config=cfg)]
            if not chunks:
                continue
            gap = np.zeros(int(rate * config.SENTENCE_PAUSE / max(speed, 0.5)), np.float32)
            joined = []
            for n, c in enumerate(chunks):  # espeak sometimes splits further
                start, end = speech_bounds(c, rate)
                joined += [c[start:end]] + ([gap] if n < len(chunks) - 1 else [])
            audio = np.concatenate(joined)
            margin_in, margin_out = 0.12, 0.15  # speech_bounds' margins: words sit inside them
            marks = spread_marks(sentence, min(margin_in, len(audio) / rate),
                                 max(0.0, len(audio) / rate - margin_out))
            pieces.append((audio, sentence, marks))
        return join_pieces(pieces, rate, speed)


class CloneEngine:
    """A voice cloned from a reference clip of a narrator (Qwen3-TTS), voiced by
    the clone server (app/clone.py) on whichever machine has the GPU for it.
    Nothing is loaded in this process, so it holds no video memory here.

    Qwen3-TTS has no speed setting, so a book's speed doesn't change these
    voices; nor word timings, so words are spread within each sentence."""

    sample_rate = 24000  # Qwen3-TTS's 12 Hz tokenizer decodes to 24 kHz

    def __init__(self, model: str):
        from . import gpus
        if not gpus.has("voices"):
            raise ValueError("This voice was cloned with the voice creator and needs a GPU server for voices: "
                             "add one in Settings, or set CLONE_URL in .env (clone/README.md).")
        self.model = model

    @property
    def client(self):
        """The GPU server doing voices, picked again for each request: if one stops
        answering, the next one that does voices carries on (app/gpus.py)."""
        from . import gpus
        found = gpus.client("voices", model=self.model)
        if found is None:
            raise ValueError("No GPU server does voices any more: add one in Settings.")
        return found

    PER_REQUEST = 32  # the clone server's limit on sentences per request

    def synth(self, text: str, voice: dict, speed: float) -> tuple[np.ndarray, Marks]:
        return self.synth_many([text], voice, speed)[0]

    def synth_many(self, texts: list[str], voice: dict, speed: float) -> list[tuple[np.ndarray, Marks]]:
        """Several paragraphs in as few requests as possible: the clone server makes
        many sentences at once for little more than the time of one (an RX 6800: one
        at 0.47x real time, sixteen at 3.8x)."""
        from .textprep import sentences
        said = [[s for s in sentences(t) if re.search(r"\w", s)] for t in texts]
        flat = [s for group in said for s in group]
        clips = []
        for start in range(0, len(flat), self.PER_REQUEST):
            clips += self._speak(voice, flat[start:start + self.PER_REQUEST])
        out, n = [], 0
        for group in said:
            pieces = []
            for sentence in group:
                audio = clips[n]
                n += 1
                # A sentence far longer than it should take ran on past its text: say it again.
                if len(audio) / self.sample_rate > 3 + len(sentence) / 5:
                    audio = min([audio, self._speak(voice, [sentence])[0]], key=len)
                start, end = speech_bounds(audio, self.sample_rate)
                audio = audio[start:end]
                margin_in, margin_out = 0.12, 0.15  # speech_bounds' margins: words sit inside them
                marks = spread_marks(sentence, min(margin_in, len(audio) / self.sample_rate),
                                     max(0.0, len(audio) / self.sample_rate - margin_out))
                pieces.append((audio, sentence, marks))
            out.append(join_pieces(pieces, self.sample_rate, 1.0) if pieces else (np.zeros(0, np.float32), []))
        return out

    def _speak(self, voice: dict, said: list[str]) -> list[np.ndarray]:
        if not said:
            return []
        rate, clips = self.client.speak(self.model, voice["ref_audio"], voice["ref_text"], said,
                                        language=voice.get("language") or "English")
        if rate != self.sample_rate:
            clips = [np.interp(np.arange(int(len(c) * self.sample_rate / rate)) * rate / self.sample_rate,
                               np.arange(len(c)), c).astype(np.float32) for c in clips]
        return clips


def speech_bounds(audio: np.ndarray, rate: int, lead: float = 0.12, tail: float = 0.15) -> tuple[int, int]:
    """Where speech starts and ends in an engine's output, with a margin.

    Speech is found by loudness over 10 ms windows, at -40 dB below the loudest
    window, so quiet openings and endings ("ch", "sh", "s", breath) count as
    speech; a margin is kept on both sides. A sample-peak threshold clipped
    "Chapter" to "hapter" and "She" to "e"."""
    if not len(audio):
        return 0, 0
    frame = max(1, rate // 100)
    usable = len(audio) // frame * frame
    if not usable:
        return 0, len(audio)
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, frame) ** 2, axis=1))
    voiced = np.flatnonzero(rms > max(1e-4, float(rms.max()) * 0.01))
    if not len(voiced):
        return 0, 0
    start = max(0, voiced[0] * frame - int(lead * rate))
    end = min(len(audio), (voiced[-1] + 1) * frame + int(tail * rate))
    return start, end


def trim_silence(audio: np.ndarray, rate: int, lead: float = 0.12, tail: float = 0.15) -> np.ndarray:
    start, end = speech_bounds(audio, rate, lead, tail)
    return audio[start:end]


def spread_marks(text: str, start: float, end: float) -> Marks:
    """Word times for an engine that only says when a stretch of speech starts and
    ends: each word gets a share in proportion to its length."""
    words = re.findall(r"\S+", text or "")
    weights = [len(re.sub(r"\W", "", w)) + 1 for w in words]
    total = sum(weights) or 1
    marks, t = [], start
    for word, weight in zip(words, weights):
        span = (end - start) * weight / total
        if re.search(r"\w", word):
            marks.append([word, t, t + span])
        t += span
    return marks


def pause_after(text: str) -> float:
    """Seconds of silence after a stretch of speech, from how it ends."""
    end = (text or "").rstrip(" \t\"'”’)]»")
    if end.endswith((".", "!", "?", "…")):
        return config.SENTENCE_PAUSE
    if end.endswith((",", ";", ":", "—", "–", "-")):
        return config.CLAUSE_PAUSE
    return 0.0  # cut mid-phrase: just the trim margin


def join_pieces(pieces: list[tuple[np.ndarray, str, Marks]], rate: int,
                speed: float) -> tuple[np.ndarray, Marks]:
    """Join trimmed pieces of one paragraph with pauses that fit their punctuation,
    carrying each piece's word marks along. Nothing is added after the last piece:
    the paragraph pause comes from the unit."""
    out, marks, t = [], [], 0.0
    for n, (audio, text, piece_marks) in enumerate(pieces):
        out.append(audio)
        marks += [[w, round(t + a, 3), round(t + b, 3)] for w, a, b in piece_marks]
        t += len(audio) / rate
        if n < len(pieces) - 1:
            gap = np.zeros(int(rate * pause_after(text) / max(speed, 0.5)), np.float32)
            out.append(gap)
            t += len(gap) / rate
    return (np.concatenate(out) if out else np.zeros(0, np.float32)), marks


def get_engine(voice: dict):
    if config.DEV_FAKE_TTS:
        key: tuple = ("fake",)
    elif voice["engine"] == "kokoro":
        key = ("kokoro", voice.get("lang", "a"))
    elif voice["engine"] == "piper":
        key = ("piper", voice["model"], voice.get("rev"))  # rev: a retrained custom voice loads anew
    elif voice["engine"] == "clone":
        key = ("clone", voice["model"])
    else:
        raise ValueError(f"Unknown engine: {voice['engine']}")
    if key not in _engines:
        if _device == "cuda" and key[0] != "clone" and any(k[0] != "clone" for k in _engines):
            # One model per process on the GPU: a worker that switches voices
            # drops the old one instead of holding both in video memory.
            # (A cloned voice holds none here: the clone server has it.)
            _engines.clear()
            if key[0] != "kokoro":
                KokoroEngine._model = None
            _free_gpu()
        if key[0] == "fake":
            _engines[key] = FakeEngine()
        elif key[0] == "kokoro":
            _engines[key] = KokoroEngine(key[1])
        elif key[0] == "clone":
            _engines[key] = CloneEngine(key[1])
        else:
            _engines[key] = PiperEngine(key[1])
    return _engines[key]


def warm(voice: dict) -> None:
    """Load a voice into this worker process ahead of time, so the first
    sample or book after a restart doesn't wait for it."""
    get_engine(voice).synth("Ready.", voice, 1.0)


def synth_unit(job: dict) -> dict:
    """Voice one work unit and write it as a 16-bit mono WAV.

    job = {"voice": {...}, "speed": 1.0, "items": [["say", "..."], ["pause", 0.3]], "out": path}
    An optional "limit" (seconds) cuts the audio to that length with a short fade.
    """
    try:
        engine = get_engine(job["voice"])
        rate = engine.sample_rate
        parts, timeline, t = [], [], 0.0
        # An engine that makes many at once (a clone server) gets the whole unit in one go.
        voiced = iter(engine.synth_many([v for k, v in job["items"] if k == "say"], job["voice"],
                                        float(job["speed"]))) if hasattr(engine, "synth_many") else None
        for kind, value in job["items"]:
            if kind == "say":
                audio, marks = next(voiced) if voiced else engine.synth(value, job["voice"], float(job["speed"]))
                timeline.append({"start": round(t, 3), "end": round(t + len(audio) / rate, 3),
                                 "words": [[w, round(t + a, 3), round(t + b, 3)] for w, a, b in marks]})
            else:
                audio = np.zeros(int(rate * float(value)), np.float32)
            parts.append(audio)
            t += len(audio) / rate
    except Exception as exc:
        if _device == "cuda" and _is_gpu_oom(exc):
            _engines.clear()
            KokoroEngine._model = None
            _free_gpu()
            raise GpuMemoryError(f"Ran out of video memory: {str(exc)[:300]}") from None
        raise
    audio = np.clip(np.concatenate(parts) if parts else np.zeros(0, np.float32), -1.0, 1.0)
    if job.get("limit") and len(audio) > int(rate * job["limit"]):
        audio = audio[:int(rate * job["limit"])].copy()
        fade = min(len(audio), int(rate * 0.6))
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    out = job["out"]
    if not job.get("limit"):
        # Word timings for Read along, one entry per "say" item, in unit time.
        timing = timing_path(out)
        with open(timing + ".part", "w", encoding="utf-8") as f:
            json.dump({"says": timeline}, f, separators=(",", ":"))
        os.replace(timing + ".part", timing)
    tmp = out + ".part"
    sf.write(tmp, audio, rate, subtype="PCM_16", format="WAV")
    os.replace(tmp, out)
    return {"out": out, "frames": len(audio), "rate": rate}


def timing_path(wav_path) -> str:
    """Where a unit's word timings live: next to its WAV."""
    return str(wav_path)[:-4] + ".json"
