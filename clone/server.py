"""Clone server: voices cloned from a reference clip of a narrator, with Qwen3-TTS.

Forge and the voice creator (app/clone.py) send it a reference clip once and
then sentences to say in that voice. It runs on whichever machine has a GPU
big enough (docker-compose.clone.yml), which needn't be Forge's; Forge points
CLONE_URL at it. One model is in memory at a time: loaded on first use, let go
after CLONE_IDLE_SECONDS without work, so the GPU is free for other things.

It lives in an image of its own: qwen-tts needs transformers 4.57, and
Forge's image has 5.x (Kokoro), which qwen-tts's model code doesn't run on.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import hmac
import io
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clone")

DATA = Path(os.environ.get("CLONE_DATA", "/data"))
REFS = DATA / "refs"
TOKEN = os.environ.get("CLONE_TOKEN", "")
IDLE_SECONDS = float(os.environ.get("CLONE_IDLE_SECONDS") or 300)
GPU_SHARE = float(os.environ.get("CLONE_GPU_SHARE") or 0.9)
ROCM = bool(getattr(torch.version, "hip", None))  # an AMD card (PyTorch's "cuda" is HIP there)
if ROCM and os.environ.get("CLONE_MIOPEN", "").lower() not in ("1", "on", "true"):
    # MIOpen, AMD's cuDNN, searches for convolution kernels anew for every input
    # length: on an RX 6800 that was 95% of the time (0.07x real time); PyTorch's
    # own kernels made it 0.47x, and 3.8x in batches of 16.
    torch.backends.cudnn.enabled = False
MAX_SENTENCES, MAX_CHARS = 32, 1000
TOKENS_PER_SECOND = 12  # the 12 Hz tokenizer: audio codes per second of speech

# Video memory each needs (measured on a 6 GB GTX 1660 in fp32: 0.6B peaked
# about 4.9 GB; 1.7B in fp16 ran out), in bf16 and in fp32.
MODELS = {
    "qwen3-tts-1.7b": {"label": "Qwen3-TTS 1.7B", "hf": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                       "bf16_gb": 7.0, "fp32_gb": 10.0},
    "qwen3-tts-0.6b": {"label": "Qwen3-TTS 0.6B", "hf": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                       "bf16_gb": 3.0, "fp32_gb": 5.5},
}


class BadAudio(RuntimeError):
    """The model said nothing usable (NaN or silence) twice."""


class TooLong(RuntimeError):
    """A single sentence ran out of video memory."""


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


_gpu: tuple[str | None, float | None] | None = None


def gpu_info() -> tuple[str | None, float | None]:
    """The GPU's name and total memory in GB, or (None, None) on the CPU."""
    global _gpu
    if _gpu is None:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            _gpu = (props.name, props.total_memory / 1024**3)
        else:
            _gpu = (None, None)
    return _gpu


def dtype_name() -> str:
    """bfloat16 where the GPU does it natively (NVIDIA Ampere and later), float16
    on AMD cards (an RX 6800 does bfloat16 at a quarter of float16's speed, and it
    measured a little less like the narrator), else float32: GTX 16xx cards give
    NaNs in float16. CLONE_DTYPE overrides."""
    override = (os.environ.get("CLONE_DTYPE") or "").strip()
    if override in ("bfloat16", "float16", "float32"):
        return override
    if not torch.cuda.is_available():
        return "float32"
    if ROCM:
        return "float16"
    if torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def batch_size() -> int:
    """Sentences made at once: CLONE_BATCH, else by video memory. Batches cost
    little more time than one sentence (RX 6800, 1.7B: 1 at 0.47x real time,
    16 at 3.8x, peaking at 10.2 GB)."""
    if os.environ.get("CLONE_BATCH"):
        return max(1, int(os.environ["CLONE_BATCH"]))
    _, total = gpu_info()
    return 16 if (total or 0) >= 14 else 8 if (total or 0) >= 10 else 4


def model_fits(model: dict) -> tuple[bool, str]:
    name, total = gpu_info()
    if name is None:
        return True, "very slow on the CPU"
    dtype = dtype_name()
    need = model["fp32_gb"] if dtype == "float32" else model["bf16_gb"]
    if total >= need:
        return True, f"needs about {need:g} GB of video memory in {dtype}"
    return False, f"needs about {need:g} GB of video memory in {dtype}; this GPU has {total:.1f} GB"


def good(wav: np.ndarray) -> bool:
    return len(wav) > 0 and bool(np.isfinite(wav).all()) and float(np.abs(wav).max()) >= 1e-3


class Engine:
    """The model in memory (one at a time) and the reference clips' prompts."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model_id: str | None = None
        self.model = None
        self.prompts: dict[str, list] = {}  # ref key -> prompt items, for the loaded model
        self.last_used = time.monotonic()

    def _load(self, model_id: str) -> None:
        if self.model_id == model_id:
            return
        self._unload()
        from qwen_tts import Qwen3TTSModel
        dtype = dtype_name()
        cuda = torch.cuda.is_available()
        attn = "sdpa"
        if cuda and dtype == "bfloat16":
            try:
                import flash_attn  # noqa: F401
                attn = "flash_attention_2"
            except ImportError:
                pass
        if cuda:
            # A hard cap below the card's size: running out raises an error this
            # server handles, instead of pushing the driver (and the machine) over.
            torch.cuda.set_per_process_memory_fraction(GPU_SHARE, 0)
        t0 = time.monotonic()
        # From the local copy (downloaded on first use): given a hub id, qwen-tts
        # asks the hub's API every time, which fails without internet.
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODELS[model_id]["hf"])
        self.model = Qwen3TTSModel.from_pretrained(path, device_map="cuda:0" if cuda else "cpu",
                                                   dtype=getattr(torch, dtype), attn_implementation=attn)
        self.model_id = model_id
        log.info("Loaded %s (%s, %s) in %.1f s", model_id, dtype, attn, time.monotonic() - t0)

    def _unload(self) -> None:
        if self.model is None:
            return
        log.info("Letting go of %s", self.model_id)
        self.model, self.model_id = None, None
        self.prompts.clear()
        self._free()

    @staticmethod
    def _free() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        with self.lock:
            self._unload()

    def unload_if_idle(self) -> None:
        if not self.lock.acquire(blocking=False):
            return  # speaking: not idle
        try:
            if self.model is not None and time.monotonic() - self.last_used > IDLE_SECONDS:
                self._unload()
        finally:
            self.lock.release()

    def _prompt(self, key: str) -> list:
        if key not in self.prompts:
            text = (REFS / f"{key}.txt").read_text(encoding="utf-8").strip()
            self.prompts[key] = self.model.create_voice_clone_prompt(
                ref_audio=str(REFS / f"{key}.wav"), ref_text=text, x_vector_only_mode=False)
        return self.prompts[key]

    def _generate(self, prompt: list, texts: list[str], language: str) -> tuple[list[np.ndarray], int]:
        # A bound on how long it may talk: generous for the text (speech runs about
        # 0.07 s a character), so a sentence it never stops saying can't run on.
        seconds = 6 + 0.2 * max(len(t) for t in texts)
        wavs, rate = self.model.generate_voice_clone(
            text=texts, language=[language] * len(texts), voice_clone_prompt=prompt,
            max_new_tokens=min(2048, int(TOKENS_PER_SECOND * seconds)))
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs], int(rate)

    def _one(self, prompt: list, text: str, language: str) -> tuple[np.ndarray, int]:
        try:
            wavs, rate = self._generate(prompt, [text], language)
        except Exception as exc:
            if not is_oom(exc):
                raise
            self._free()
            raise TooLong("A single sentence ran out of video memory. Split it, or use the smaller model.") from None
        return wavs[0], rate

    def speak(self, model_id: str, key: str, sentences: list[str], language: str) -> tuple[int, list[np.ndarray]]:
        with self.lock:
            self.last_used = time.monotonic()
            try:
                self._load(model_id)
                prompt = self._prompt(key)
                out: list[np.ndarray | None] = [None] * len(sentences)
                rate = 0
                size = batch_size()
                # Sentences of about the same length together: a batch runs as long as its longest.
                order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
                for start in range(0, len(order), size):
                    idx = order[start:start + size]
                    try:
                        wavs, rate = self._generate(prompt, [sentences[i] for i in idx], language)
                    except Exception as exc:
                        if not is_oom(exc) or len(idx) == 1:
                            if is_oom(exc):
                                self._free()
                                raise TooLong("A single sentence ran out of video memory. "
                                              "Split it, or use the smaller model.") from None
                            raise
                        self._free()  # a batch too big for the card: one at a time
                        log.warning("Out of video memory with %d sentences at once; one at a time", len(idx))
                        wavs = []
                        for i in idx:
                            wav, rate = self._one(prompt, sentences[i], language)
                            wavs.append(wav)
                    for i, wav in zip(idx, wavs):
                        out[i] = wav
                for i, wav in enumerate(out):
                    if not good(wav):
                        log.warning("Sentence %d came out empty or broken; trying it again", i + 1)
                        wav, rate = self._one(prompt, sentences[i], language)
                        if not good(wav):
                            raise BadAudio(f"Sentence {i + 1} came out silent or broken twice.")
                        out[i] = wav
                return rate, out  # type: ignore[return-value]
            finally:
                self.last_used = time.monotonic()


engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    def watch():  # nothing loads at startup; an idle model is let go
        while True:
            time.sleep(10)
            try:
                engine.unload_if_idle()
            except Exception:
                log.exception("Unloading the idle model failed")
    threading.Thread(target=watch, daemon=True, name="idle").start()
    yield


app = FastAPI(title="Clone server", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def check_token(request: Request, call_next):
    path = request.url.path
    if TOKEN and path.startswith("/api/") and path != "/api/health":
        given = request.headers.get("x-forge-token", "")
        if not hmac.compare_digest(given.encode(), TOKEN.encode()):
            return JSONResponse({"detail": "This clone server needs its token (X-Forge-Token)."}, status_code=401)
    return await call_next(request)


@app.get("/api/health")
def health():
    name, total = gpu_info()
    models = []
    for model_id, m in MODELS.items():
        fits, note = model_fits(m)
        models.append({"id": model_id, "label": m["label"], "hf": m["hf"], "fits": fits, "note": note})
    return {"ok": True, "device": "cuda" if name else "cpu", "gpu": name,
            "gpu_gb": round(total, 1) if total else None, "dtype": dtype_name(),
            "loaded": engine.model_id, "models": models}


def _key_ok(key: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", key))


class RefBody(BaseModel):
    text: str
    audio: str  # base64 WAV


@app.put("/api/refs/{key}")
def put_ref(key: str, body: RefBody):
    if not _key_ok(key):
        raise HTTPException(400, "A reference clip's key is 40 hex characters.")
    try:
        audio = base64.b64decode(body.audio, validate=True)
    except ValueError:
        raise HTTPException(400, "The reference clip isn’t base64.")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "The reference clip needs its transcript.")
    if not hmac.compare_digest(hashlib.sha1(audio + b"\0" + text.encode("utf-8")).hexdigest(), key):
        raise HTTPException(400, "The key doesn’t match the clip and its text.")
    try:
        wav, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=False)
    except Exception:
        raise HTTPException(400, "The reference clip isn’t a WAV file.")
    if len(wav) < rate * 1:
        raise HTTPException(400, "The reference clip is shorter than a second.")
    REFS.mkdir(parents=True, exist_ok=True)
    # The transcript first, then the clip: a clip on disk always has its text.
    for path, data in ((REFS / f"{key}.txt", text.encode("utf-8")), (REFS / f"{key}.wav", audio)):
        tmp = path.with_name(f".{path.name}.part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    log.info("Reference clip %s: %.1f s, “%s”", key[:8], len(wav) / rate, text[:60])
    return {"ok": True}


@app.get("/api/refs/{key}")
def get_ref(key: str):
    if not _key_ok(key) or not (REFS / f"{key}.wav").exists() or not (REFS / f"{key}.txt").exists():
        raise HTTPException(404, "No such reference clip")
    return {"ok": True}


class SpeakBody(BaseModel):
    model: str
    ref: str
    sentences: list[str]
    language: str = "English"


@app.post("/api/speak")
def speak(body: SpeakBody):
    model = MODELS.get(body.model)
    if model is None:
        raise HTTPException(400, f"Unknown model {body.model!r}. This server has: {', '.join(MODELS)}.")
    fits, note = model_fits(model)
    if not fits:
        raise HTTPException(400, f"{model['label']} doesn’t fit this GPU: it {note}.")
    sentences = [s.strip() for s in body.sentences]
    if not sentences or len(sentences) > MAX_SENTENCES:
        raise HTTPException(400, f"Send 1 to {MAX_SENTENCES} sentences at a time.")
    if any(not s for s in sentences):
        raise HTTPException(400, "A sentence is empty.")
    if any(len(s) > MAX_CHARS for s in sentences):
        raise HTTPException(400, f"A sentence is longer than {MAX_CHARS} characters: split it.")
    if not _key_ok(body.ref) or not (REFS / f"{body.ref}.wav").exists() or not (REFS / f"{body.ref}.txt").exists():
        raise HTTPException(404, "Unknown reference clip: send it first (PUT /api/refs/<key>).")
    t0 = time.monotonic()
    try:
        rate, wavs = engine.speak(body.model, body.ref, sentences, body.language)
    except TooLong as exc:
        raise HTTPException(507, str(exc))
    except BadAudio as exc:
        raise HTTPException(500, str(exc))
    out, seconds = [], 0.0
    for wav in wavs:
        buf = io.BytesIO()
        sf.write(buf, np.clip(wav, -1.0, 1.0), rate, subtype="PCM_16", format="WAV")
        out.append(base64.b64encode(buf.getvalue()).decode())
        seconds += len(wav) / rate
    took = time.monotonic() - t0
    log.info("%s: %d sentence%s, %.1f s of audio in %.1f s (%.2fx real time)", body.model, len(sentences),
             "" if len(sentences) == 1 else "s", seconds, took, seconds / took if took else 0)
    return {"rate": rate, "audio": out}


@app.post("/api/unload")
def unload():
    engine.unload()
    return {"ok": True, "loaded": None}
