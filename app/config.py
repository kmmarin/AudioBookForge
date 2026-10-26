"""Settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import math
import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def available_cpus() -> int:
    """CPUs this container may really use (respects cpusets and cgroup quotas)."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def available_memory_gb() -> float:
    """Memory limit of this container (cgroup limit, else MemAvailable)."""
    limit = None
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw != "max":
            limit = int(raw) / 1024**3
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 1024**2
                limit = avail if limit is None else min(limit, avail)
    except (OSError, ValueError):
        pass
    return limit or 4.0


def gpu_memory_gb() -> tuple[float, float] | None:
    """(free, total) video memory of the first GPU, or None without one."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        free, total = (float(x) / 1024 for x in out.splitlines()[0].split(","))
        return free, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(_env("DATA_DIR", "/data"))
OUTPUT_DIR = Path(_env("OUTPUT_DIR", "/output"))
MODELS_DIR = Path(_env("MODELS_DIR", "/models"))
BOOKS_DIR = DATA_DIR / "books"
PREVIEW_DIR = DATA_DIR / "previews"
DB_PATH = DATA_DIR / "forge.db"
VOICES_FILE = Path(_env("VOICES_FILE", str(APP_DIR / "voices.yaml")))

# --- Speed -----------------------------------------------------------------
# Each TTS worker is a separate process with its own copy of the model, so
# workers scale across cores without fighting over Python's GIL.
DEVICE = _env("DEVICE", "cpu")  # cpu | cuda
CPUS = available_cpus()
# Measured on a 12-core Ryzen: 6x4, 8x3, 12x2 and 24x1 all gave 5.3-5.5x real
# time, so fewer, wider workers win: same speed, less memory. Auto gives one
# worker per 4 cores (capped by memory), then splits every core among them.
_MEM_PER_WORKER_GB = 1.4  # measured: Kokoro on CPU, torch 2.14
_auto_workers = (
    max(1, CPUS // _int("TORCH_THREADS", 4)) if DEVICE == "cpu" else 2
)
_auto_workers = max(1, min(_auto_workers, int((available_memory_gb() - 1.5) / _MEM_PER_WORKER_GB)))

# On the GPU, video memory sets the worker count. Each voice worker gets a hard
# cap (its CUDA context, one model and working space), Whisper gets its share,
# and a margin stays free for the desktop and other programs. Going over on a
# shared Windows GPU can take down the whole machine, so this errs low.
# Measured under WSL on a GTX 1660, one 1800-character unit: Kokoro peaks at
# 1.33 GB per process, Piper at 0.7-1.0 GB, Whisper base at 0.27 GB.
GPU_MEM_PER_WORKER_GB = _float("GPU_MEM_PER_WORKER_GB", 1.5)
GPU_RESERVE_GB = _float("GPU_RESERVE_GB", 0.75)
WHISPER_GPU_GB = _float("WHISPER_GPU_GB", 0.5)
GPU_FREE_GB = GPU_TOTAL_GB = None
import sys  # noqa: E402  (only the scheduler, `python -m app.worker`, sizes the pool)
if DEVICE == "cuda" and sys.argv and sys.argv[0].endswith(os.path.join("app", "worker.py")):
    _gpu = gpu_memory_gb()
    if _gpu:
        GPU_FREE_GB, GPU_TOTAL_GB = _gpu
        _whisper = WHISPER_GPU_GB if _env("WHISPER_DEVICE", "cpu") == "cuda" and _bool("QA_DEFAULT", True) else 0
        _fit = int((GPU_FREE_GB - GPU_RESERVE_GB - _whisper) / GPU_MEM_PER_WORKER_GB)
        _auto_workers = max(1, min(4, _fit))
TTS_WORKERS = _int("TTS_WORKERS", _auto_workers)
TORCH_THREADS = _int("TORCH_THREADS", max(1, CPUS // TTS_WORKERS) if DEVICE == "cpu" else 1)
ENCODE_WORKERS = _int("ENCODE_WORKERS", max(1, CPUS // 2))
BOOK_CONCURRENCY = _int("BOOK_CONCURRENCY", 2)  # books voiced at the same time
UNIT_CHARS = _int("UNIT_CHARS", 1800)  # text per parallel work unit
DEV_FAKE_TTS = _bool("DEV_FAKE_TTS", False)  # tones instead of speech, for testing

# --- Output ----------------------------------------------------------------
DEFAULT_FORMAT = _env("DEFAULT_FORMAT", "m4b")
BITRATE = _env("BITRATE", "64k")
OPUS_BITRATE = _env("OPUS_BITRATE", "32k")
NORMALIZE = _bool("NORMALIZE", True)  # EBU R128 loudness: every book at one volume; a little slower
OUTPUT_LAYOUT = _env("OUTPUT_LAYOUT", "abs")  # abs (Author/Series/Title/) | flat

# --- Whisper narration check --------------------------------------------------
QA_DEFAULT = _bool("QA_DEFAULT", True)
QA_SAMPLE = _float("QA_SAMPLE", 0.25)  # share of each chapter that gets checked
QA_FLAG_WER = _float("QA_FLAG_WER", 0.12)  # word error rate that flags a chapter
WHISPER_MODEL = _env("WHISPER_MODEL", "base")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")
WHISPER_THREADS = _int("WHISPER_THREADS", max(1, CPUS // 4))

# --- Read along for audiobooks you add (Whisper writes down their text) -------
# Runs on WHISPER_DEVICE. Measured on CPU (int8, 6 threads) against Forge's own
# text and timings: see transcribe.py for the numbers behind this default.
TRANSCRIBE_MODEL = _env("TRANSCRIBE_MODEL", "small")
TRANSCRIBE_THREADS = _int("TRANSCRIBE_THREADS", max(2, CPUS // 2))
# Pieces decoded at once on the CPU: faster, about 1 GB more memory at 8 (small).
TRANSCRIBE_BATCH = _int("TRANSCRIBE_BATCH", 8)
# Video memory the model needs on the GPU, weights plus decoding (beam search,
# word timings). Blank: estimated from the model size in transcribe.GPU_NEEDS_GB.
TRANSCRIBE_GPU_GB = _float("TRANSCRIBE_GPU_GB", 0)

# --- Book details lookup ------------------------------------------------------
METADATA_LOOKUP = _bool("METADATA_LOOKUP", True)  # match new books online
METADATA_SOURCES = [s.strip() for s in _env("METADATA_SOURCES", "openlibrary,audible,google").split(",")
                    if s.strip()]
GOOGLE_BOOKS_API_KEY = _env("GOOGLE_BOOKS_API_KEY")  # Google is skipped without one
AUDIBLE_REGION = _env("AUDIBLE_REGION", "com")  # com, co.uk, ca, com.au, de, fr …
LOOKUP_TIMEOUT = _float("LOOKUP_TIMEOUT", 8)  # seconds per request
MATCH_MIN = _float("MATCH_MIN", 0.8)  # how close a result must be to fill in details
MATCH_CLEAR = _float("MATCH_CLEAR", 0.9)  # Update info applies a match this close without asking

# --- Publishing to a server -------------------------------------------------------
# A Forge with a GPU can voice books and send them to an always-on Forge (a
# server), so Read along works there from any device with one place per book.
PUBLISH_URL = _env("PUBLISH_URL").rstrip("/")  # the server, e.g. http://192.168.1.10:8080
PUBLISH_AUTO = _bool("PUBLISH_AUTO", True)  # send every book when it's finished
PUBLIC_URL = _env("PUBLIC_URL").rstrip("/")  # how the server reaches this Forge, for voice changes
SHARE_TOKEN = _env("SHARE_TOKEN")  # the same secret on both: only they can send books or voice requests
SITE_URL = _env("SITE_URL").rstrip("/")  # the address people open, e.g. https://books.example.com (https-setup.sh)

# --- Voice creator ---------------------------------------------------------------
# Where /createvoice is proxied to: the voice creator service, or on a server
# without a GPU, the PC's Forge (which proxies it on). Blank = feature off.
VOICELAB_URL = _env("VOICELAB_URL").rstrip("/")
# Taking turns on the GPU with the voice creator (app/gpushare.py). auto = when
# this worker uses the GPU; on = always (also for testing on the CPU); off = never.
GPU_SHARE = _env("GPU_SHARE", "auto").lower()
GPU_SHARE_DIR = Path(_env("GPU_SHARE_DIR", str(MODELS_DIR / "gpu-share")))
GPU_SHARE_STALE = _float("GPU_SHARE_STALE", 120)  # seconds without a heartbeat = that side is gone
CUSTOM_VOICES_DIR = Path(_env("CUSTOM_VOICES_DIR", str(MODELS_DIR / "custom-voices")))
# Voices cloned from a reference clip (one folder each: voice.json, ref.wav,
# ref.txt), voiced by a clone server running Qwen3-TTS on a machine with the
# GPU for it (clone/server.py). Blank CLONE_URL = they're listed but can't speak.
CLONED_VOICES_DIR = Path(_env("CLONED_VOICES_DIR", str(MODELS_DIR / "cloned-voices")))
CLONE_URL = _env("CLONE_URL").rstrip("/")
CLONE_TOKEN = _env("CLONE_TOKEN") or SHARE_TOKEN
# Parts in a cloned voice sent to the clone server at once: it makes one batch at
# a time, so two keep it busy (one being made, the next one waiting).
CLONE_WORKERS = _int("CLONE_WORKERS", 2)

# --- Watch and read (app/scenes.py) ------------------------------------------------
# Pictures of what's happening, above the text as it's read: one about every
# SCENE_WORDS words of narration (75 is about 30 seconds), drawn by the clone
# server (CLONE_URL, /api/draw) in DRAW_STYLE, and kept with the book.
SCENE_WORDS = _int("SCENE_WORDS", 75)
DRAW_STYLE = _env("DRAW_STYLE", "sketch")  # sketch, ink, storybook, cinematic, noir, oil
DRAW_MODEL = _env("DRAW_MODEL", "dreamshaper-xl-lightning")

# --- Narration pacing (seconds) -----------------------------------------------
# The engines' own silence around speech is trimmed to a 0.27 s margin (0.12 s
# before speech, 0.15 s after), and these are added on top of it. Totals: a
# clause ~0.35 s, a sentence ~0.5 s, a paragraph ~0.7 s, a title ~1.1 s.
CLAUSE_PAUSE = _float("CLAUSE_PAUSE", 0.08)  # where an engine split at a comma or dash
# Every sentence is voiced on its own, so this is the pause after every period:
# clearly longer than a comma's, shorter than a paragraph's.
SENTENCE_PAUSE = _float("SENTENCE_PAUSE", 0.4)
PARA_PAUSE = _float("PARA_PAUSE", 0.6)
SCENE_PAUSE = _float("SCENE_PAUSE", 1.35)
TITLE_PAUSE = _float("TITLE_PAUSE", 0.85)
CHAPTER_TAIL = _float("CHAPTER_TAIL", 1.5)
WORDS_PER_MINUTE = 155

FORMATS: dict[str, dict] = {
    "m4b": {"label": "M4B", "ext": "m4b", "muxer": "ipod", "codec": "aac",
            "mode": "segments", "cover": True, "chapters": True},
    "m4a": {"label": "M4A", "ext": "m4a", "muxer": "ipod", "codec": "aac",
            "mode": "segments", "cover": True, "chapters": True},
    "mp3": {"label": "MP3", "ext": "mp3", "muxer": "mp3", "codec": "libmp3lame",
            "mode": "segments", "cover": True, "chapters": True},
    "opus": {"label": "Opus", "ext": "opus", "muxer": "ogg", "codec": "libopus",
             "mode": "single", "cover": False, "chapters": True},
    "flac": {"label": "FLAC", "ext": "flac", "muxer": "flac", "codec": "flac",
             "mode": "single", "cover": True, "chapters": False},
}
if DEFAULT_FORMAT not in FORMATS:
    DEFAULT_FORMAT = "m4b"
GPU_SHARED = GPU_SHARE == "on" or (GPU_SHARE == "auto" and "cuda" in (DEVICE, WHISPER_DEVICE))
