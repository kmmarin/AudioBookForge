"""Settings. Every value can be overridden with an environment variable.

The readiness thresholds, clip filters and training limits are starting
values: they all live here so they can be tuned in one place."""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _cpus() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        count = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, -(-int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(_env("LAB_DATA", "/labdata"))
PROJECTS_DIR = DATA_DIR / "projects"
DB_PATH = DATA_DIR / "voicelab.db"
MODELS_DIR = Path(_env("MODELS_DIR", "/models"))
LAB_MODELS = MODELS_DIR / "voicelab"  # base checkpoints, speaker model
CUSTOM_VOICES_DIR = Path(_env("CUSTOM_VOICES_DIR", str(MODELS_DIR / "custom-voices")))
CLONED_VOICES_DIR = Path(_env("CLONED_VOICES_DIR", str(MODELS_DIR / "cloned-voices")))
FORGE_CSS = Path(_env("FORGE_CSS", str(APP_DIR.parent / "app" / "static" / "app.css")))

# --- Cloned voices (app/clone.py) ---------------------------------------------------
# A clone server makes them with Qwen3-TTS from one clip of the narrator, no
# training: on this machine or on one with a bigger GPU. Empty: cloning is off
# and voices are trained with Piper as before.
CLONE_URL = _env("CLONE_URL")
CLONE_TOKEN = _env("CLONE_TOKEN", _env("SHARE_TOKEN"))
CLONE_RATE = 24000  # reference clips are cut at the rate the clone server works at
CLONE_REF_MIN, CLONE_REF_MAX = 8.0, 14.0  # seconds: a reference clip's length
CLONE_CANDIDATES = _int("CLONE_CANDIDATES", 4)  # clips tried as the reference, each making a sample
# A Piper voice can learn from the project's cloned voice instead of the
# recordings: the clone reads the narrator's own sentences, as steady audio that
# says exactly its text, as much of it as wanted (the recordings give what
# Whisper cut from an MP3, character voices and all). Minutes of it to make.
TEACH_MINUTES = _float("TEACH_MINUTES", 180)
TEACH_PER_REQUEST = 32  # sentences per request: the clone server's limit

# --- Machine ---------------------------------------------------------------------
CPUS = _cpus()
# Forge may be voicing on the same machine: by default use half the cores.
THREADS = _int("LAB_THREADS", max(1, CPUS // 2))
TRAIN_DEVICE = _env("TRAIN_DEVICE", "cpu")  # cpu | cuda
WHISPER_MODEL = _env("LAB_WHISPER_MODEL", "small")
WHISPER_DEVICE = _env("LAB_WHISPER_DEVICE", "cpu")  # cuda takes its turn on the GPU too

# --- Sharing the GPU with Forge (app/gpushare.py) ---------------------------------
GPU_SHARE = _env("GPU_SHARE", "auto").lower()  # auto (when using the GPU) | on | off
GPU_SHARE_DIR = Path(_env("GPU_SHARE_DIR", str(MODELS_DIR / "gpu-share")))
GPU_SHARE_STALE = _float("GPU_SHARE_STALE", 120)
GPU_SHARED = GPU_SHARE == "on" or (GPU_SHARE == "auto" and "cuda" in (TRAIN_DEVICE, WHISPER_DEVICE))
# Video memory: what must be free before training starts, and what stays free
# for the desktop and other programs. A GTX 1660 has 6 GB.
TRAIN_MIN_FREE_GB = _float("TRAIN_MIN_FREE_GB", 3.5)
GPU_RESERVE_GB = _float("GPU_RESERVE_GB", 0.75)

# --- Preparing clips ------------------------------------------------------------------
SAMPLE_RATE = 22050  # Piper medium voices
WINDOW_SECONDS = _float("ANALYZE_WINDOW", 300)  # audio transcribed per step; progress is saved after each
CLIP_MIN = _float("CLIP_MIN", 1.0)  # seconds
CLIP_MAX = _float("CLIP_MAX", 15.0)
CLIP_TARGET = _float("CLIP_TARGET", 6.0)  # end a clip at the first sentence end after this long
MIN_WORDS = _int("CLIP_MIN_WORDS", 2)
MIN_SNR_DB = _float("MIN_SNR_DB", 22)  # speech over the background between words
MAX_CLIPPED = _float("MAX_CLIPPED", 0.0005)  # share of samples flattened at full scale
MIN_WORD_PROB = _float("MIN_WORD_PROB", 0.7)  # Whisper's mean confidence per word
MAX_UNSURE_WORDS = _float("MAX_UNSURE_WORDS", 0.2)  # share of words Whisper was unsure of
SPEAKER_MIN_SIM = _float("SPEAKER_MIN_SIM", 0.45)  # likeness to the main voice, 0 to 1
SPEAKER_MAX_SIM = _float("SPEAKER_MAX_SIM", 0.6)  # the cut-off never goes above this
SPEAKER_SPREAD = _float("SPEAKER_SPREAD", 3.5)  # and no further below typical than this many MADs
SILENCE_DB = _float("SILENCE_DB", -45)  # quieter than this between clips counts as a pause
# A whole audiobook is more than training needs: stop analysing once this much is usable.
ANALYZE_ENOUGH_MINUTES = _float("ANALYZE_ENOUGH_MINUTES", 300)

# --- Readiness (minutes of usable speech) ----------------------------------------------
READY_USABLE = _float("READY_USABLE", 5)
READY_GOOD = _float("READY_GOOD", 15)
READY_GREAT = _float("READY_GREAT", 60)
READY_PLENTY = _float("READY_PLENTY", 180)  # beyond this, more audio adds little
TRAIN_MAX_MINUTES = _float("TRAIN_MAX_MINUTES", 210)  # the best 3½ hours are used
# Listening stops once this much clean speech is heard: a 16-hour audiobook
# would otherwise take two hours of GPU time to hear, for clips never used.
LISTEN_ENOUGH_MINUTES = _float("LISTEN_ENOUGH_MINUTES", 90)
# A cloned voice needs a few clean 8-14 s clips to try as its reference, not
# hours: listening to a narrator being cloned stops after this much clean speech.
LISTEN_CLONE_MINUTES = _float("LISTEN_CLONE_MINUTES", 10)
# A voice's first round is this many minutes of training, then it goes into
# Forge with its measured match (10 minutes was too little to sound like anyone).
# "Train further" then trains until it stops sounding any closer to the
# narrator. 0 = the first round does that too.
TRAIN_FIRST_MINUTES = _float("TRAIN_FIRST_MINUTES", 60)
HOLDOUT_CLIPS = _int("HOLDOUT_CLIPS", 20)  # never trained on: the quality check uses them
PHONEME_MIN_COUNT = _int("PHONEME_MIN_COUNT", 5)  # a sound counts as covered with this many examples

# --- Training ------------------------------------------------------------------------
# Blank batch size = picked from video memory (6 GB: 12), halved after running out.
TRAIN_BATCH = _int("TRAIN_BATCH", 0)
# 32-true by default: GTX 16xx cards are known for NaNs in half precision.
TRAIN_PRECISION = _env("TRAIN_PRECISION", "32-true")
# Training runs until the voice stops getting closer to the narrator (measured,
# see LIKENESS_* below); these bound it. At ~1 step a second on a GTX 1660,
# 40,000 steps is about 11 hours: a safety limit, not a target.
TRAIN_MAX_BATCHES = _int("TRAIN_MAX_BATCHES", 40000)  # per round, the most it ever trains
TRAIN_MIN_BATCHES = _int("TRAIN_MIN_BATCHES", 4500)  # before it may stop by itself
TRAIN_VAL_EVERY = _int("TRAIN_VAL_EVERY", 500)  # batches between validations (and saved checkpoints)
# Every LIKENESS_EVERY steps the voice so far says LIKENESS_SENTENCES held-out
# sentences, is compared with the narrator's real clips, and its distortion is
# measured (quality.artifacts: tones at fixed pitches, the buzz an undertrained
# voice has). It stops after LIKENESS_PATIENCE checks in a row that got neither
# LIKENESS_MIN_GAIN closer nor CLEAN_MIN_GAIN dB less distorted. Likeness
# levels off long before the distortion does, so it can't stop on that alone.
LIKENESS_EVERY = _int("LIKENESS_EVERY", 1500)
LIKENESS_SENTENCES = _int("LIKENESS_SENTENCES", 10)
LIKENESS_PATIENCE = _int("LIKENESS_PATIENCE", 3)
LIKENESS_MIN_GAIN = _float("LIKENESS_MIN_GAIN", 0.01)
CLEAN_MIN_GAIN = _float("CLEAN_MIN_GAIN", 0.2)
LIKENESS_SLACK = 0.02  # a less distorted check counts as better unless it lost more likeness than this
ALONG_KEEP = _int("ALONG_KEEP", 12)  # voices kept from the latest checks, to pick one by ear
TRAIN_VAL_SHARE = _float("TRAIN_VAL_SHARE", 0.05)
TRAIN_LOADERS = _int("TRAIN_LOADERS", 2)  # data loader processes

# The checkpoints training can start from (rhasspy/piper-checkpoints), by
# accent. Before its first round a voice starts from whichever one's released
# voice already sounds most like the narrator (a deep male voice from a male
# voice): less to change, so it comes out closer and cleaner for the same
# training. The quality check compares against that released voice.
BASES = {
    "en-us": {"label": "US English", "espeak": "en-us", "locale": "en_US", "starts": [
        {"key": "lessac", "name": "Lessac (US)", "voice": "en_US-lessac-medium",
         "checkpoint": "en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt"},
        {"key": "ryan", "name": "Ryan (US)", "voice": "en_US-ryan-medium",
         "checkpoint": "en/en_US/ryan/medium/epoch=4641-step=3104302.ckpt"},
        {"key": "hfc_male", "name": "HFC male (US)", "voice": "en_US-hfc_male-medium",
         "checkpoint": "en/en_US/hfc_male/medium/epoch=2785-step=2128064.ckpt"},
        {"key": "joe", "name": "Joe (US)", "voice": "en_US-joe-medium",
         "checkpoint": "en/en_US/joe/medium/epoch=7889-step=1221224.ckpt"},
    ]},
    "en-gb": {"label": "British English", "espeak": "en-gb-x-rp", "locale": "en_GB", "starts": [
        {"key": "alba", "name": "Alba (UK)", "voice": "en_GB-alba-medium",
         "checkpoint": "en/en_GB/alba/medium/epoch=4179-step=2101090.ckpt"},
        {"key": "alan", "name": "Alan (UK)", "voice": "en_GB-alan-medium",
         "checkpoint": "en/en_GB/alan/medium/epoch=6339-step=1647790.ckpt"},
        {"key": "northern_english_male", "name": "Northern English male (UK)",
         "voice": "en_GB-northern_english_male-medium",
         "checkpoint": "en/en_GB/northern_english_male/medium/epoch=9029-step=2261720.ckpt"},
    ]},
}
DEFAULT_ACCENT = _env("DEFAULT_ACCENT", "en-us")


def start_of(project: dict) -> dict:
    """The starting voice a project trains from: the one picked before its first
    round, else the accent's first (what voices made before the pick used)."""
    accent = BASES.get(project["accent"]) or BASES[DEFAULT_ACCENT]
    key = (project.get("training") or {}).get("start")
    return next((s for s in accent["starts"] if s["key"] == key), accent["starts"][0])
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
