"""Voice samples: the same short passage in any voice, made by the worker and kept
in PREVIEW_DIR. Every voice and mood gets a SAVED_SECONDS one made ahead of time
(worker.premake_samples), so picking a voice (Compare voices, Read along's voice
list) plays it at once instead of waiting for it to be made."""
from __future__ import annotations

import re

from . import config, textprep, voices

PREVIEW_TEXT = ("Chapter one. The rain had stopped by the time the ferry reached the harbor, "
                "and the whole town looked freshly washed in the late afternoon light.")
SAVED_SECONDS = 10.0


def job(voice: dict, speed: float = 1.0, seconds: float = SAVED_SECONDS) -> dict:
    """The sample job for a voice (with its mood filled in: voices.with_mood); its
    "out" is where the clip is kept, so the same voice, speed and length are made once."""
    items = textprep.sample_items(None, PREVIEW_TEXT, seconds * 1.4, speed)
    path = config.PREVIEW_DIR / f"{re.sub(r'[^A-Za-z0-9_-]', '_', voice['key'])}-{speed}-{seconds}.wav"
    return {"voice": voice, "speed": speed, "items": items, "out": str(path), "limit": seconds}


def saved_jobs() -> list[dict]:
    """A saved sample's job for every voice and mood this Forge lists. Cloned
    voices only with a clone server to make them."""
    from . import gpus
    out = []
    can_clone = gpus.has("voices")
    for v in voices.list_voices():
        if v["engine"] == "clone" and not can_clone:
            continue
        for mood in voices.mood_ids(v) or [None]:
            out.append(job(voices.with_mood(v, mood)))
    return out
