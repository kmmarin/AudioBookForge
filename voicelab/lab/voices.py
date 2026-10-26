"""The finished voices, in Forge's models volume.

A trained (Piper) voice is <slug>.onnx plus <slug>.onnx.json in
CUSTOM_VOICES_DIR, the Piper config, which also carries a "forge" section with
its name. A cloned voice is a folder <slug>/ in CLONED_VOICES_DIR: ref.wav (the
narrator's clip it's cloned from), ref.txt (what they say in it) and
voice.json ({"name", "model", "language", "project", "clip", "installed",
"match"}), written last. Forge lists every voice it finds there. Files are only
ever replaced in one step (written beside, then renamed), so Forge never sees
half a voice, and the folder's time changes, which is how Forge notices
without a restart. Slugs are unique across both kinds."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

from . import config


def _replace(src: Path, dest: Path, copy: bool = True) -> None:
    tmp = dest.with_name(f".{dest.name}.part")
    if copy:
        shutil.copyfile(src, tmp)
    else:
        os.replace(src, tmp)
    os.replace(tmp, dest)


def _write_json(dest: Path, data: dict) -> None:
    _write_text(dest, json.dumps(data, ensure_ascii=False, indent=2))


def _write_text(dest: Path, text: str) -> None:
    tmp = dest.with_name(f".{dest.name}.part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dest)


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "voice"


def taken(slug: str) -> bool:
    """A voice of either kind already has this name."""
    return (config.CUSTOM_VOICES_DIR / f"{slug}.onnx").exists() or (config.CLONED_VOICES_DIR / slug).exists()


def free_slug(name: str, keep: str | None = None) -> str:
    """A file name for a new voice that no other voice uses."""
    base = slugify(name)
    slug, n = base, 2
    while slug != keep and taken(slug):
        slug, n = f"{base}-{n}", n + 1
    return slug


def clone_folder(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{1,48}", slug or ""):
        raise ValueError("Not a voice name")
    return config.CLONED_VOICES_DIR / slug


def is_clone(slug: str) -> bool:
    try:
        return (clone_folder(slug) / "voice.json").exists()
    except ValueError:
        return False


def cloned_for(project_id: str) -> str | None:
    """The slug of the cloned voice in Forge made from this project, if any
    (the project's own voice may since be a Piper one, taught by it)."""
    for meta in sorted(config.CLONED_VOICES_DIR.glob("*/voice.json")):
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("project") == project_id:
                return meta.parent.name
        except (OSError, ValueError):
            continue
    return None


def install_clone(slug: str, ref_wav: Path, ref_text: str, info: dict, sample: Path | None = None) -> None:
    """Put a cloned voice into Forge: the clip and its words first, voice.json
    last, so Forge only lists it once it's complete. Picking another clip
    later replaces them in place. sample: its sample passage, if already made."""
    folder = clone_folder(slug)
    folder.mkdir(parents=True, exist_ok=True)
    _replace(ref_wav, folder / "ref.wav")
    _write_text(folder / "ref.txt", ref_text.strip() + "\n")
    for old in folder.glob("sample-*.wav"):  # made from the clip it had before
        old.unlink(missing_ok=True)
    if sample is not None and sample.exists():
        _replace(sample, folder / f"sample-{info['model']}.wav")
    _write_json(folder / "voice.json", {**info, "language": info.get("language") or "English",
                                        "installed": time.time()})


def paths(slug: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-z0-9-]{1,48}", slug or ""):
        raise ValueError("Not a voice name")
    model = config.CUSTOM_VOICES_DIR / f"{slug}.onnx"
    return model, model.with_name(model.name + ".json")


def install(slug: str, onnx: Path, piper_config: dict, info: dict) -> None:
    """Copy a voice into Forge's folder: the model first, then its config, so
    Forge only lists it once it's complete."""
    config.CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model, meta = paths(slug)
    _replace(onnx, model)
    data = dict(piper_config)
    data["dataset"] = slug
    data["forge"] = {**info, "installed": time.time()}
    _write_json(meta, data)


def listing() -> list[dict]:
    out = []
    for meta in sorted(config.CUSTOM_VOICES_DIR.glob("*.onnx.json")):
        model = meta.with_suffix("")
        if not model.exists():
            continue
        try:
            forge = json.loads(meta.read_text(encoding="utf-8")).get("forge") or {}
        except (OSError, ValueError):
            continue
        out.append({"slug": model.stem, "id": f"custom-{model.stem}", "kind": "piper", "match": forge.get("match"),
                    "name": forge.get("name") or model.stem, "project": forge.get("project"),
                    "accent": forge.get("accent"), "installed": forge.get("installed"),
                    "round": forge.get("round"), "step": forge.get("step"),  # the check it was picked from
                    "size_mb": round(model.stat().st_size / 1e6, 1)})
    for meta in sorted(config.CLONED_VOICES_DIR.glob("*/voice.json")):
        ref = meta.with_name("ref.wav")
        if not ref.exists():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        slug = meta.parent.name
        out.append({"slug": slug, "id": f"clone-{slug}", "kind": "clone", "model": data.get("model"),
                    "match": data.get("match"), "name": data.get("name") or slug, "project": data.get("project"),
                    "clip": data.get("clip"), "installed": data.get("installed"),
                    "size_mb": round(ref.stat().st_size / 1e6, 1)})
    return out


def rename(slug: str, name: str) -> None:
    if is_clone(slug):
        meta = clone_folder(slug) / "voice.json"
        _write_json(meta, {**json.loads(meta.read_text(encoding="utf-8")), "name": name})
        return
    model, meta = paths(slug)
    data = json.loads(meta.read_text(encoding="utf-8"))
    data.setdefault("forge", {})["name"] = name
    _write_json(meta, data)


def set_model(slug: str, model: str) -> None:
    """A cloned voice reads with another model from now on (same clip)."""
    if not is_clone(slug):
        raise ValueError("Only a cloned voice can change its model")
    meta = clone_folder(slug) / "voice.json"
    _write_json(meta, {**json.loads(meta.read_text(encoding="utf-8")), "model": model})


# The same passage for every voice, about 10 seconds read aloud, so they're easy to compare.
SAMPLE_TEXT = ("It was the kind of evening that made the whole town slow down. The lights came on one by one, "
               "and somewhere down the street a screen door opened, and someone called a name.")


def sample_path(slug: str) -> Path:
    model, _ = paths(slug)
    return model.with_name(f"{slug}.sample.wav")


def make_sample(slug: str) -> Path:
    """The voice reading SAMPLE_TEXT, made once per version of the voice: on the
    CPU for a trained voice, by the clone server for a cloned one (CloneError
    if it can't)."""
    if is_clone(slug):
        return _clone_sample(slug)
    from piper.voice import PiperVoice

    from . import media, quality
    model, _ = paths(slug)
    out = sample_path(slug)
    if out.exists() and out.stat().st_mtime >= model.stat().st_mtime:
        return out
    voice = PiperVoice.load(str(model), use_cuda=False)
    media.write_wav(out, quality.synthesize(voice, SAMPLE_TEXT), voice.config.sample_rate)
    return out


def set_match(slug: str, likeness: float) -> None:
    """How much the voice sounds like the narrator (measured), shown in Forge's list."""
    model, meta = paths(slug)
    data = json.loads(meta.read_text(encoding="utf-8"))
    data.setdefault("forge", {})["match"] = round(float(likeness), 3)
    _write_json(meta, data)


def _clone_sample(slug: str) -> Path:
    from . import media
    from .clone import client
    folder = clone_folder(slug)
    data = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
    ref = folder / "ref.wav"
    out = folder / f"sample-{data['model']}.wav"
    if out.exists() and out.stat().st_mtime >= ref.stat().st_mtime:
        return out
    rate, (audio,) = client(data["model"]).speak(
        data["model"], ref, (folder / "ref.txt").read_text(encoding="utf-8"), [SAMPLE_TEXT],
        language=data.get("language") or "English")
    media.write_wav(out, audio, rate)
    return out


def remove(slug: str) -> None:
    """The config goes first: Forge stops listing the voice before its model goes."""
    if is_clone(slug):
        folder = clone_folder(slug)
        (folder / "voice.json").unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
        return
    model, meta = paths(slug)
    meta.unlink(missing_ok=True)
    model.unlink(missing_ok=True)
    sample_path(slug).unlink(missing_ok=True)
