"""Voice registry loaded from voices.yaml, plus the voices made with the
voice creator: trained Piper voices (.onnx + .onnx.json files in
CUSTOM_VOICES_DIR), Kokoro voices found for a narrator (a voicepack, .pt +
.pt.json, in the same folder) and voices cloned from a reference clip (a folder
each in CLONED_VOICES_DIR, voiced by the clone server: app/clone.py)."""
from __future__ import annotations

import json
from functools import lru_cache

import yaml

from . import config
from .clone import MODELS as CLONE_MODELS

ENGINE_GROUPS = {"kokoro": "Kokoro (most natural)", "piper": "Piper (fastest)"}
CUSTOM_GROUP = "Your voices"
KOKORO_LANG_TO_ISO = {"a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi",
                      "i": "it", "j": "ja", "p": "pt", "z": "zh"}


def _custom_stamp() -> tuple[int, ...]:
    """Changes whenever a voice is added, renamed, removed or switched to another
    model: the voice creator only ever replaces files, which updates the times
    of their folders."""
    stamps = []
    for folder in (config.CUSTOM_VOICES_DIR, config.CLONED_VOICES_DIR):
        try:
            stamps.append(folder.stat().st_mtime_ns)
        except OSError:
            stamps.append(0)
    try:
        stamps += sorted(p.stat().st_mtime_ns for p in config.CLONED_VOICES_DIR.iterdir() if p.is_dir())
    except OSError:
        pass
    return tuple(stamps)


CLONED_FILES = ("ref.wav", "ref.txt", "voice.json")  # a cloned voice's folder; voice.json is written last


def cloned_fingerprint(folder) -> str:
    """Changes whenever a cloned voice's clip, text, name, model or match does."""
    import hashlib
    h = hashlib.sha1()
    for name in CLONED_FILES:
        h.update((folder / name).read_bytes())
    return h.hexdigest()


def _cloned_voices() -> dict[str, dict]:
    found: dict[str, dict] = {}
    try:
        folders = sorted(p for p in config.CLONED_VOICES_DIR.iterdir() if p.is_dir())
    except OSError:
        return found
    for folder in folders:
        meta, ref = folder / "voice.json", folder / "ref.wav"
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            ref_text = (folder / "ref.txt").read_text(encoding="utf-8").strip()
            rev = int(meta.stat().st_mtime)
        except (OSError, ValueError):
            continue  # being written or removed
        model = info.get("model")
        if model not in CLONE_MODELS or not ref.exists() or not ref_text:
            continue
        name = str(info.get("name") or folder.name).strip()
        voice_id = f"clone-{folder.name}"
        match = info.get("match")
        label = f"{name} · cloned" + (f" · {round(match * 100)}% match" if isinstance(match, (int, float)) else "")
        found[voice_id] = {
            "id": voice_id, "label": label, "engine": "clone", "model": model,
            "ref_audio": str(ref), "ref_text": ref_text, "narrator": name, "group": CUSTOM_GROUP,
            "moods": [], "language": str(info.get("language") or "English"),
            "rev": rev,  # another model or reference clip: new audio, not the old from the cache
        }
    return found


def _custom_voices() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for meta in sorted(config.CUSTOM_VOICES_DIR.glob("*.onnx.json")):
        model = meta.with_suffix("")  # name.onnx
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            rev = int(model.stat().st_mtime)
        except (OSError, ValueError):
            continue  # half-copied or removed meanwhile
        forge = info.get("forge") or {}
        name = str(forge.get("name") or model.stem).strip()
        voice_id = f"custom-{model.stem}"
        # How much it sounds like the narrator, from the voice creator's measured check.
        match = forge.get("match")
        label = f"{name} · {round(match * 100)}% match" if isinstance(match, (int, float)) else name
        found[voice_id] = {
            "id": voice_id, "label": label, "engine": "piper", "model": str(model),
            "narrator": name, "group": CUSTOM_GROUP, "moods": [],
            "language": str((info.get("espeak") or {}).get("voice") or "en")[:2],
            "rev": rev,  # a retrained voice gets new audio, not the old one from the cache
        }
    return found


def _kokoro_voices() -> dict[str, dict]:
    """Kokoro voicepacks made for a narrator: <slug>.pt, with <slug>.pt.json
    ({"name", "lang", "match", ...}) written last."""
    found: dict[str, dict] = {}
    for meta in sorted(config.CUSTOM_VOICES_DIR.glob("*.pt.json")):
        pack = meta.with_suffix("")  # name.pt
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            rev = int(pack.stat().st_mtime)
        except (OSError, ValueError):
            continue  # half-copied or removed meanwhile
        name = str(info.get("name") or pack.stem).strip()
        voice_id = f"custom-{pack.stem}"
        match = info.get("match")
        label = f"{name} · Kokoro" + (f" · {round(match * 100)}% match" if isinstance(match, (int, float)) else "")
        found[voice_id] = {
            "id": voice_id, "label": label, "engine": "kokoro", "voice": str(pack),
            "lang": str(info.get("lang") or "a"), "narrator": name, "group": CUSTOM_GROUP, "moods": [],
            "rev": rev,  # a better voicepack later: new audio, not the old from the cache
        }
    return found


@lru_cache(maxsize=1)
def _load(custom_stamp: tuple = ()) -> tuple[dict[str, dict], str | None]:
    data = yaml.safe_load(config.VOICES_FILE.read_text()) or {}
    found: dict[str, dict] = {}
    for voice in data.get("voices", []):
        if not voice.get("id") or voice.get("engine") not in ENGINE_GROUPS:
            continue
        voice["moods"] = [m for m in voice.get("moods") or [] if m.get("id") and m.get("speaker")]
        found[voice["id"]] = voice
    default = data.get("default")
    if default not in found:
        default = next(iter(found), None)
    # Your own voices come first: their own section at the top of every voice list.
    yours = {**_cloned_voices(), **_kokoro_voices(), **_custom_voices()}
    found = {**{k: v for k, v in yours.items() if k not in found}, **found}
    return found, default


def _voices() -> tuple[dict[str, dict], str | None]:
    return _load(_custom_stamp())


def list_voices() -> list[dict]:
    return list(_voices()[0].values())


def get(voice_id: str | None) -> dict | None:
    return _voices()[0].get(voice_id or "")


def default_id() -> str | None:
    return _voices()[1]


def language(voice: dict) -> str:
    """ISO 639-1 language for Whisper."""
    if voice["engine"] == "kokoro":
        return KOKORO_LANG_TO_ISO.get(voice.get("lang", "a"), "en")
    if voice["engine"] == "clone":
        return "en"  # Qwen3-TTS takes a language name; the voice creator makes English voices
    return voice.get("language") or voice.get("model", "en")[:2]


def gone_name(voice_id: str | None) -> str:
    """A readable name for a voice that isn't listed any more, such as a custom
    voice removed in the voice creator: books keep copies made in it."""
    for prefix in ("custom-", "clone-"):
        if voice_id and voice_id.startswith(prefix):
            return voice_id[len(prefix):].replace("-", " ").capitalize() + " (removed)"
    return voice_id or ""


def narrator(voice: dict) -> str:
    return voice.get("narrator") or voice.get("label") or voice["id"]


def mood_ids(voice: dict) -> list[str]:
    return [m["id"] for m in voice["moods"]]


def pick_mood(voice: dict, mood: str | None) -> str | None:
    """The mood to use: this one if the voice has it, else the voice's first, else none."""
    ids = mood_ids(voice)
    if not ids:
        return None
    return mood if mood in ids else ids[0]


def with_mood(voice: dict, mood: str | None) -> dict:
    """The voice as the engine needs it, with the mood's speaker filled in.
    "key" names the exact sound, so cached audio is never shared across moods."""
    mood = pick_mood(voice, mood)
    found = next((m for m in voice["moods"] if m["id"] == mood), None)
    key = voice["id"] + (f"@{voice['rev']}" if voice.get("rev") else "")
    if found is None:
        return {**voice, "key": key}
    return {**voice, "speaker": found["speaker"], "mood": mood, "key": f"{key}:{mood}"}


def public(voice: dict) -> dict:
    return {"id": voice["id"], "label": voice.get("label", voice["id"]),
            "group": voice.get("group") or ENGINE_GROUPS[voice["engine"]], "narrator": narrator(voice),
            "moods": [{"id": m["id"], "label": m.get("label", m["id"])} for m in voice["moods"]]}
