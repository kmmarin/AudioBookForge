"""Clone server: voices cloned from a reference clip of a narrator, with Qwen3-TTS.

Forge and the voice creator (app/clone.py) send it a reference clip once and
then sentences to say in that voice. It runs on whichever machine has a GPU
big enough (docker-compose.clone.yml), which needn't be Forge's; Forge points
CLONE_URL at it.

Every card in the machine can work, each on its own requests at once (Forge's
Settings chooses which are used). On each card one model is in memory at a
time: loaded on first use, let go after CLONE_IDLE_SECONDS without work, so the
card is free for other things. Before a card's first job with a model it's
benchmarked: how many sentences at once it says fastest, and how much of the
card that takes, measured in what's free on it then. The results are kept per
card (cards.json) and used from then on; Settings can run them again.

It lives in an image of its own: qwen-tts needs transformers 4.57, and
Forge's image has 5.x (Kokoro), which qwen-tts's model code doesn't run on.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import hmac
import io
import json
import logging
import math
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


GIB = 1024**3
STATE = DATA / "cards.json"  # which cards are used, and what each one's benchmark found
HEADROOM_GB = 0.5  # left free above a benchmark's peak: a longer sentence than the test's
BENCH_BATCHES = (2, 4, 8, 12, 16, 24, 32)
# Narration-like sentences of everyday lengths, for the benchmark (written for it).
BENCH_SENTENCES = [
    "The rain had stopped by the time the bus pulled into the square, and the streets shone under the lamps.",
    "She counted the coins twice before she handed them over.",
    "Nobody in the village could remember a winter as long as that one.",
    "He folded the letter, put it back in its envelope, and set it on the table without a word.",
    "The dog lifted its head, listened for a moment, and went back to sleep.",
    "By noon the heat was so heavy that even the birds had gone quiet.",
    "There were three doors at the end of the corridor, and only one of them was locked.",
    "I asked him where he had been all week, but he only shrugged.",
    "The river was higher than anyone had seen it, brown and fast and full of branches.",
    "They walked in silence until the lights of the farmhouse came into view.",
    "Somewhere below, a radio was playing an old song about the sea.",
    "The map was useless now; half the roads on it no longer existed.",
    "She laughed, and for a second she looked twenty years younger.",
    "The train was late, as it always was on Fridays, and the platform was crowded.",
    "He had promised himself he would not look back, and he did not.",
    "Every window in the house was dark except the one at the top of the stairs.",
]


def _state() -> dict:
    try:
        data = json.loads(STATE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(data: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_name(f".{STATE.name}.part")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


_state_lock = threading.Lock()


def dtype_name(device: int | None) -> str:
    """bfloat16 where the card does it natively (NVIDIA Ampere and later), float16
    on AMD cards (an RX 6800 does bfloat16 at a quarter of float16's speed, and it
    measured a little less like the narrator), else float32: GTX 16xx cards give
    NaNs in float16. CLONE_DTYPE overrides."""
    override = (os.environ.get("CLONE_DTYPE") or "").strip()
    if override in ("bfloat16", "float16", "float32"):
        return override
    if device is None:
        return "float32"
    if ROCM:
        return "float16"
    if torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def good(wav: np.ndarray) -> bool:
    return len(wav) > 0 and bool(np.isfinite(wav).all()) and float(np.abs(wav).max()) >= 1e-3


class Engine:
    """One card (or the CPU): the model in memory there, one at a time, the
    reference clips' prompts, and what its benchmark found."""

    def __init__(self, device: int | None):
        self.device = device
        if device is None:
            self.name, self.total = None, None
        else:
            props = torch.cuda.get_device_properties(device)
            self.name, self.total = props.name, props.total_memory / GIB
        self.key = f"{device}:{self.name}"  # a card in another slot, or another card here, is benchmarked again
        self.dev = f"cuda:{device}" if device is not None else "cpu"
        self.lock = threading.Lock()
        self.model_id: str | None = None
        self.model = None
        self.prompts: dict[str, list] = {}  # ref key -> prompt items, for the loaded model
        self.last_used = time.monotonic()
        self.busy = 0  # requests on this card or waiting for it
        self.fell_back = False  # the last request ran out of video memory and went one sentence at a time
        self.benchmarking: str | None = None  # the model being benchmarked now
        self.cap: float | None = None  # while benchmarking: the share of the card free to measure in

    # --- What it can do ----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.key not in (_state().get("disabled") or [])

    def need_gb(self, model_id: str) -> float:
        m = MODELS[model_id]
        return m["fp32_gb"] if dtype_name(self.device) == "float32" else m["bf16_gb"]

    def fits(self, model_id: str) -> tuple[bool, str]:
        if self.device is None:
            return True, "very slow on the CPU"
        need, dtype = self.need_gb(model_id), dtype_name(self.device)
        if self.total >= need:
            return True, f"needs about {need:g} GB of video memory in {dtype}"
        return False, f"needs about {need:g} GB of video memory in {dtype}; {self.name} has {self.total:.1f} GB"

    def free_gb(self) -> float | None:
        return torch.cuda.mem_get_info(self.device)[0] / GIB if self.device is not None else None

    def bench(self, model_id: str) -> dict | None:
        """This card's benchmark for the model: {"batch", "rtf", "peak_gb", "share", ...}."""
        return ((_state().get("bench") or {}).get(self.key) or {}).get(model_id)

    def share(self, model_id: str) -> float:
        """How much of the card it may use: what the benchmark measured (peak plus
        headroom), else CLONE_GPU_SHARE."""
        if self.cap is not None:
            return self.cap
        b = self.bench(model_id)
        return b["share"] if b else GPU_SHARE

    def batch(self, model_id: str) -> int:
        """Sentences at once: the benchmark's fastest, else CLONE_BATCH, else by video memory."""
        b = self.bench(model_id)
        if b:
            return b["batch"]
        if os.environ.get("CLONE_BATCH"):
            return max(1, int(os.environ["CLONE_BATCH"]))
        return 16 if (self.total or 0) >= 14 else 8 if (self.total or 0) >= 10 else 4

    # --- The model -------------------------------------------------------------------
    def _load(self, model_id: str) -> None:
        if self.model_id == model_id:
            return
        self._unload()
        from qwen_tts import Qwen3TTSModel
        dtype = dtype_name(self.device)
        cuda = self.device is not None
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
            torch.cuda.set_per_process_memory_fraction(self.share(model_id), self.device)
        t0 = time.monotonic()
        # From the local copy (downloaded on first use): given a hub id, qwen-tts
        # asks the hub's API every time, which fails without internet.
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODELS[model_id]["hf"])
        self.model = Qwen3TTSModel.from_pretrained(path, device_map=self.dev, dtype=getattr(torch, dtype),
                                                   attn_implementation=attn)
        self.model_id = model_id
        log.info("%s: loaded %s (%s, %s) in %.1f s", self.label, model_id, dtype, attn, time.monotonic() - t0)

    @property
    def label(self) -> str:
        return f"card {self.device} ({self.name})" if self.device is not None else "CPU"

    def _unload(self) -> None:
        if self.model is None:
            return
        log.info("%s: letting go of %s", self.label, self.model_id)
        self.model, self.model_id = None, None
        self.prompts.clear()
        self._free()

    def _free(self) -> None:
        gc.collect()
        if self.device is not None:
            with torch.cuda.device(self.device):
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

    # --- Speaking ----------------------------------------------------------------------
    def speak(self, model_id: str, key: str, sentences: list[str], language: str) -> tuple[int, list[np.ndarray]]:
        with self.lock:
            self.last_used = time.monotonic()
            self.fell_back = False
            try:
                # A card's first job with a model measures how hard it can be pushed first.
                if self.device is not None and self.bench(model_id) is None:
                    self._benchmark(model_id, key)
                self._load(model_id)
                prompt = self._prompt(key)
                out: list[np.ndarray | None] = [None] * len(sentences)
                rate = 0
                size = self.batch(model_id)
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
                        self._free()  # a batch too big for the card now: one at a time
                        self.fell_back = True
                        log.warning("%s: out of video memory with %d sentences at once; one at a time",
                                    self.label, len(idx))
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

    # --- The benchmark -------------------------------------------------------------------
    def benchmark(self, model_id: str, key: str) -> dict:
        with self.lock:
            try:
                return self._benchmark(model_id, key)
            finally:
                self.last_used = time.monotonic()

    def _benchmark(self, model_id: str, key: str) -> dict:
        """How hard this card can be pushed with this model, in what's free on it
        now: one batch at each size, until a batch runs out of memory or speed stops
        growing. Keeps the fastest (the smallest within 5% of it) and the share of
        the card it needs, plus headroom."""
        self.benchmarking = model_id
        log.info("%s: benchmarking %s", self.label, model_id)
        try:
            self._unload()
            free = self.free_gb()
            self.cap = max(0.3, min(0.95, (free - HEADROOM_GB) / self.total))
            self._load(model_id)
            prompt = self._prompt(key)
            self._generate(prompt, BENCH_SENTENCES[:2], "English")  # warms the card up: not timed
            trials: list[dict] = []
            for batch in BENCH_BATCHES:
                texts = (BENCH_SENTENCES * 3)[:batch]
                torch.cuda.reset_peak_memory_stats(self.device)
                t0 = time.monotonic()
                try:
                    wavs, rate = self._generate(prompt, texts, "English")
                except Exception as exc:
                    if not is_oom(exc):
                        raise
                    self._free()
                    trials.append({"batch": batch, "out_of_memory": True})
                    break
                took = time.monotonic() - t0
                audio = sum(len(w) for w in wavs) / rate
                trials.append({"batch": batch, "rtf": round(audio / took, 2),
                               "peak_gb": round(torch.cuda.max_memory_reserved(self.device) / GIB, 2)})
                log.info("%s: %d at once, %.2fx real time, %.1f GB", self.label, batch, trials[-1]["rtf"],
                         trials[-1]["peak_gb"])
                done = [t for t in trials if "rtf" in t]
                if len(done) >= 3 and done[-1]["rtf"] < done[-2]["rtf"] * 1.03 and done[-2]["rtf"] < done[-3]["rtf"] * 1.03:
                    break  # bigger batches stopped paying
            result = pick(trials, self.total, self.cap)
            result.update(model=model_id, card=self.name, measured=int(time.time()), free_gb=round(free, 1))
            with _state_lock:
                data = _state()
                data.setdefault("bench", {}).setdefault(self.key, {})[model_id] = result
                _save_state(data)
            log.info("%s: %s at its best with %d at once, %.2fx real time, up to %.0f%% of the card",
                     self.label, model_id, result["batch"], result["rtf"], result["share"] * 100)
            return result
        finally:
            self.cap = None
            self.benchmarking = None
            if self.model is not None:  # from now on, within what the benchmark found
                torch.cuda.set_per_process_memory_fraction(self.share(model_id), self.device)


def pick(trials: list[dict], total_gb: float, cap: float) -> dict:
    """From a benchmark's trials: the fastest batch that fit (the smallest within
    5% of it, to keep memory free), and the share of the card it needs."""
    ok = [t for t in trials if "rtf" in t]
    if not ok:
        return {"batch": 1, "rtf": 0.0, "peak_gb": None, "share": round(cap, 2), "trials": trials}
    top = max(t["rtf"] for t in ok)
    best = min((t for t in ok if t["rtf"] >= 0.95 * top), key=lambda t: t["batch"])
    share = min(cap, math.ceil((best["peak_gb"] + HEADROOM_GB) / total_gb * 20) / 20)
    return {"batch": best["batch"], "rtf": best["rtf"], "peak_gb": best["peak_gb"], "share": round(share, 2),
            "trials": trials}


ENGINES: dict[int | None, Engine] = {}
_dispatch = threading.Lock()


def engines() -> list[Engine]:
    if not ENGINES:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                ENGINES[i] = Engine(i)
        else:
            ENGINES[None] = Engine(None)
    return list(ENGINES.values())


def able(model_id: str) -> list[Engine]:
    """The cards in use that the model fits."""
    return [e for e in engines() if e.enabled and e.fits(model_id)[0]]


def choose(model_id: str) -> Engine:
    """The card for a request: the least busy that can run the model, one with it
    loaded first; counted busy until done()."""
    with _dispatch:
        cards = able(model_id)
        if not cards:
            notes = "; ".join(e.fits(model_id)[1] for e in engines() if e.enabled) or "no card is in use"
            raise HTTPException(400, f"{MODELS[model_id]['label']} doesn’t fit any card here: {notes}.")
        # A card being benchmarked from Settings comes last: it takes nothing else meanwhile.
        e = min(cards, key=lambda e: (e.benchmarking is not None, e.busy, e.model_id != model_id, -(e.total or 0)))
        e.busy += 1
        return e


def done(e: Engine) -> None:
    with _dispatch:
        e.busy -= 1


@asynccontextmanager
async def lifespan(_: FastAPI):
    engines()

    def watch():  # nothing loads at startup; an idle model is let go
        while True:
            time.sleep(10)
            for e in engines():
                try:
                    e.unload_if_idle()
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


def card_json(e: Engine) -> dict:
    free = e.free_gb()
    return {"index": e.device, "name": e.name, "gb": round(e.total, 1) if e.total else None,
            "free_gb": round(free, 1) if free is not None else None, "dtype": dtype_name(e.device),
            "enabled": e.enabled, "loaded": e.model_id, "busy": e.busy, "benchmarking": e.benchmarking,
            "models": {m: {"fits": e.fits(m)[0], "note": e.fits(m)[1], "need_gb": e.need_gb(m) if e.device is not None else None,
                           "bench": e.bench(m), "batch": e.batch(m)} for m in MODELS}}


@app.get("/api/health")
def health():
    """Every card and what it can do; models fit when a card in use fits them,
    and are ready now when one has them loaded or enough free memory. The flat
    fields describe the cards in use together, for older Forges."""
    cards = [card_json(e) for e in engines()]
    used = [c for c in cards if c["enabled"]]
    models = []
    for model_id, m in MODELS.items():
        fitting = [c for c in used if c["models"][model_id]["fits"]]
        ready = any(c["loaded"] == model_id or c["free_gb"] is None
                    or c["free_gb"] >= c["models"][model_id]["need_gb"] for c in fitting)
        note = (fitting[0]["models"][model_id]["note"] if fitting
                else (used[0]["models"][model_id]["note"] if used else "No card is in use."))
        needs = [c["models"][model_id]["need_gb"] for c in fitting if c["models"][model_id]["need_gb"]]
        models.append({"id": model_id, "label": m["label"], "hf": m["hf"], "fits": bool(fitting), "ready": ready,
                       "note": note, "need_gb": min(needs) if needs else None, "cards": len(fitting)})
    gpus_used = [c for c in used if c["index"] is not None]
    return {"ok": True, "device": "cuda" if gpus_used else "cpu",
            "gpu": ", ".join(c["name"] for c in gpus_used) or None,
            "gpu_gb": max((c["gb"] for c in gpus_used), default=None),
            "free_gb": max((c["free_gb"] for c in gpus_used), default=None),
            "dtype": used[0]["dtype"] if used else None,
            "loaded": next((c["loaded"] for c in used if c["loaded"]), None),
            "parallel": len(used), "models": models, "cards": cards}


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


def _ref_ok(key: str) -> bool:
    return _key_ok(key) and (REFS / f"{key}.wav").exists() and (REFS / f"{key}.txt").exists()


@app.post("/api/speak")
def speak(body: SpeakBody):
    """On the least busy card that can run the model: several cards work at once."""
    if body.model not in MODELS:
        raise HTTPException(400, f"Unknown model {body.model!r}. This server has: {', '.join(MODELS)}.")
    sentences = [s.strip() for s in body.sentences]
    if not sentences or len(sentences) > MAX_SENTENCES:
        raise HTTPException(400, f"Send 1 to {MAX_SENTENCES} sentences at a time.")
    if any(not s for s in sentences):
        raise HTTPException(400, "A sentence is empty.")
    if any(len(s) > MAX_CHARS for s in sentences):
        raise HTTPException(400, f"A sentence is longer than {MAX_CHARS} characters: split it.")
    if not _ref_ok(body.ref):
        raise HTTPException(404, "Unknown reference clip: send it first (PUT /api/refs/<key>).")
    e = choose(body.model)
    t0 = time.monotonic()
    try:
        rate, wavs = e.speak(body.model, body.ref, sentences, body.language)
    except TooLong as exc:
        raise HTTPException(507, str(exc))
    except BadAudio as exc:
        raise HTTPException(500, str(exc))
    finally:
        done(e)
    out, seconds = [], 0.0
    for wav in wavs:
        buf = io.BytesIO()
        sf.write(buf, np.clip(wav, -1.0, 1.0), rate, subtype="PCM_16", format="WAV")
        out.append(base64.b64encode(buf.getvalue()).decode())
        seconds += len(wav) / rate
    took = time.monotonic() - t0
    log.info("%s, %s: %d sentence%s, %.1f s of audio in %.1f s (%.2fx real time)", e.label, body.model,
             len(sentences), "" if len(sentences) == 1 else "s", seconds, took, seconds / took if took else 0)
    return {"rate": rate, "audio": out, "card": e.device, "seconds": round(took, 2), "fell_back": e.fell_back}


@app.post("/api/unload")
def unload():
    for e in engines():
        e.unload()
    return {"ok": True, "loaded": None}


class CardsBody(BaseModel):
    enabled: list[int]  # the cards to use, by index; the others are left alone


@app.post("/api/cards")
def set_cards(body: CardsBody):
    """Which cards do the work (Forge's Settings). At least one stays in use."""
    cards = [e for e in engines() if e.device is not None]
    if cards and not any(e.device in body.enabled for e in cards):
        raise HTTPException(400, "Keep at least one card in use.")
    with _state_lock:
        data = _state()
        data["disabled"] = [e.key for e in cards if e.device not in body.enabled]
        _save_state(data)
    for e in cards:
        if not e.enabled:
            e.unload()
    return health()


class BenchBody(BaseModel):
    ref: str  # a reference clip already sent (PUT /api/refs/<key>)
    model: str | None = None  # default: the best model that fits each card
    cards: list[int] | None = None  # default: every card in use


@app.post("/api/bench")
def start_bench(body: BenchBody):
    """Benchmark cards again (Forge's Settings): each in the background, while it
    takes no other work; health shows "benchmarking" until it's done."""
    if not _ref_ok(body.ref):
        raise HTTPException(404, "Unknown reference clip: send it first (PUT /api/refs/<key>).")
    if body.model and body.model not in MODELS:
        raise HTTPException(400, f"Unknown model {body.model!r}.")
    started = []
    for e in engines():
        if e.device is None or not e.enabled or (body.cards is not None and e.device not in body.cards):
            continue
        model = body.model or next((m for m in MODELS if e.fits(m)[0]), None)  # MODELS is best first
        if model is None or not e.fits(model)[0] or e.benchmarking:
            continue

        def run(e=e, model=model):
            try:
                e.benchmark(model, body.ref)
            except Exception:
                log.exception("%s: benchmarking %s failed", e.label, model)
        e.benchmarking = model  # shows at once; the run sets it again
        threading.Thread(target=run, daemon=True, name=f"bench-{e.device}").start()
        started.append({"card": e.device, "model": model})
    if not started:
        raise HTTPException(409, "No card to benchmark: none in use fits a model, or they're benchmarking already.")
    return {"started": started}
