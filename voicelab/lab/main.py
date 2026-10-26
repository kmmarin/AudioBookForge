"""Voice creator web app, served under /createvoice (Forge passes that path through)."""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import clone, config, dataset, db, jobs, media, voices

BASE = "/createvoice"
AUDIO_EXTS = {".m4b", ".m4a", ".mp3", ".flac", ".wav", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mka", ".webm", ".aiff", ".aif"}
TRAINING = {"training"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    config.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs.start()
    yield
    jobs.shutdown()


app = FastAPI(title="Voice creator", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """After an update, browsers ask before reusing the old page and scripts."""
    response = await call_next(request)
    if request.url.path in (BASE + "/", BASE) or request.url.path.startswith(BASE + "/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def require_project(project_id: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{12}", project_id):
        raise HTTPException(404, "Voice project not found")
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Voice project not found")
    return project


def require_source(project_id: str, source_id: str) -> dict:
    source = db.get_source(source_id) if re.fullmatch(r"[a-f0-9]{12}", source_id) else None
    if source is None or source["project_id"] != project_id:
        raise HTTPException(404, "Recording not found")
    return source


def source_json(s: dict) -> dict:
    return {"id": s["id"], "filename": s["filename"], "size": s["size"], "received": s["received"],
            "duration": s["duration"], "status": s["status"], "done_until": s["done_until"],
            "message": s["message"]}


def project_json(p: dict, full: bool = False, choices: list[dict] | None = None) -> dict:
    sources = db.sources(p["id"])
    model = clone.effective_model(p, choices)
    state = p["clone"] or {}
    out = {
        "model": model, "model_set": p["model"],
        "voice_kind": ("clone" if voices.is_clone(p["voice_slug"]) else "piper") if p["voice_slug"] else None,
        "clone_samples": len(state.get("candidates") or []) if state.get("model") == model else 0,
        "id": p["id"], "name": p["name"], "name_source": p["name_source"], "name_note": p["name_note"],
        "accent": p["accent"], "status": p["status"], "progress": p["progress"], "message": p["message"],
        "voice_slug": p["voice_slug"], "updated": p["updated"], "created": p["created"],
        "auto_train": bool(p["auto_train"]),
        "minutes": round(sum(float(s["duration"] or 0) for s in sources) / 60, 1),
        "usable_minutes": (p["readiness"] or {}).get("usable_minutes"),
        "rating": (p["readiness"] or {}).get("rating"),
        "quality_overall": (p["quality"] or {}).get("overall"),
        # The measured match, for the list: how much it sounds like the narrator, and their own clips' score.
        "likeness": (p["quality"] or {}).get("likeness"), "real_likeness": (p["quality"] or {}).get("real_likeness"),
        "wer": (p["quality"] or {}).get("wer"), "round": (p["training"] or {}).get("round"),
        "listening": db.job_waiting(p["id"], "analyze"),
    }
    if full:
        training = dict(p["training"] or {})
        training.pop("data", None)
        voice = next((v for v in voices.listing() if v["slug"] == p["voice_slug"]), None)
        out.update(sources=[source_json(s) for s in sources], readiness=p["readiness"], training=training,
                   quality=p["quality"], voice=voice,
                   start=config.start_of(p)["name"] if training.get("start") else None, along=along_json(p),
                   clone=clone_json(p, model, voice), cloned_voice=voices.cloned_for(p["id"]))
    return out


def clone_json(p: dict, model: str, voice: dict | None) -> dict:
    """The clips tried as a cloned voice's reference, for the project's model:
    which samples exist and which one is the voice in Forge."""
    state = dict(p["clone"] or {})
    current = state.get("model") == model
    state["busy"] = db.job_waiting(p["id"], "clone")
    state["candidates"] = [
        {**c, "sample": clone.sample_path(p["id"], model, c["clip"]).exists(),
         "in_forge": bool(voice and voice.get("kind") == "clone" and voice.get("clip") == c["clip"]
                          and voice.get("model") == model)}
        for c in (state.get("candidates") or [])] if current else []
    return state


def along_path(p: dict, step: int) -> Path:
    """A check's files this round (without the extension): its sample and, for
    the latest few, the voice as it was then."""
    round_no = int((p["training"] or {}).get("round") or 1)
    return dataset.project_dir(p["id"]) / "train" / f"round-{round_no}" / "along" / f"{step:06d}"


def along_json(p: dict) -> list[dict]:
    """Every check of this round: [step, likeness, distortion] in the training state."""
    training = p["training"] or {}
    installed = next((v for v in voices.listing() if v["slug"] == p["voice_slug"]), None) if p["voice_slug"] else None
    out = []
    for check in training.get("likeness") or []:
        base = along_path(p, int(check[0]))
        out.append({"step": check[0], "likeness": check[1], "artifacts": check[2] if len(check) > 2 else None,
                    "sample": base.with_suffix(".wav").exists(), "voice": base.with_suffix(".onnx").exists(),
                    "best": check[0] == training.get("best_step"),
                    "in_forge": bool(installed and installed.get("step") == check[0]
                                     and installed.get("round") == training.get("round"))})
    return out


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(BASE + "/", status_code=307)


@app.get(BASE)
def no_slash():
    return RedirectResponse(BASE + "/", status_code=307)  # relative, so it works behind Forge


@app.get(BASE + "/")
def index():
    return FileResponse(config.APP_DIR / "static" / "index.html")


@app.get(BASE + "/static/{name}")
def static(name: str):
    if name == "app.css":
        return FileResponse(config.FORGE_CSS, media_type="text/css")  # Forge's look, shared
    path = config.APP_DIR / "static" / name
    if not re.fullmatch(r"[\w.-]+", name) or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.get(BASE + "/api/health")
def health():
    return {"ok": True}


@app.get(BASE + "/api/config")
def get_config():
    return {
        "accents": [{"id": k, "label": v["label"], "starts": [s["name"] for s in v["starts"]]}
                    for k, v in config.BASES.items()],
        "default_accent": config.DEFAULT_ACCENT,
        "thresholds": {"usable": config.READY_USABLE, "good": config.READY_GOOD,
                       "great": config.READY_GREAT, "plenty": config.READY_PLENTY},
        "train_max_minutes": config.TRAIN_MAX_MINUTES,
        "device": config.TRAIN_DEVICE,
        "audio_exts": sorted(AUDIO_EXTS),
        # What a voice can be made with, and the clone server that makes cloned ones.
        "clone": clone.health(), "models": clone.model_choices(clone.health()),
    }


@app.get(BASE + "/api/gpu")
def gpu_state():
    """Who has the GPU, for the status line."""
    from app import gpushare
    share = gpushare.Share(config.GPU_SHARE_DIR, config.GPU_SHARE_STALE)
    forge = share.forge()
    return {"shared": config.GPU_SHARED, "device": config.TRAIN_DEVICE, "lab": share.lab(),
            "forge": {k: forge.get(k) for k in ("gpu", "busy", "pool", "yielded")} if forge else None}


@app.get(BASE + "/api/projects")
def list_projects():
    choices = clone.model_choices(clone.health())
    return [project_json(p, choices=choices) for p in db.list_projects()]


class NewProject(BaseModel):
    name: str | None = None
    accent: str | None = None
    auto_train: bool = True  # "Start training as soon as the audio is ready"


@app.post(BASE + "/api/projects")
def create_project(body: NewProject):
    project_id = _new_id()
    accent = body.accent if body.accent in config.BASES else config.DEFAULT_ACCENT
    db.create_project(project_id, accent, body.auto_train)
    name = re.sub(r"\s+", " ", body.name or "").strip()[:80]
    if name:
        # A name typed by the person: no looking for the narrator's name at all.
        db.update_project(project_id, name=name, name_source="user")
    (config.PROJECTS_DIR / project_id / "sources").mkdir(parents=True, exist_ok=True)
    return project_json(db.get_project(project_id), full=True)


@app.get(BASE + "/api/projects/{project_id}")
def get_project(project_id: str):
    return project_json(require_project(project_id), full=True)


class ProjectPatch(BaseModel):
    name: str | None = None
    accent: str | None = None
    auto_train: bool | None = None
    model: str | None = None  # a clone model, or "piper" (lab/clone.py)


@app.patch(BASE + "/api/projects/{project_id}")
def patch_project(project_id: str, body: ProjectPatch):
    project = require_project(project_id)
    fields: dict = {}
    if body.name is not None:
        name = re.sub(r"\s+", " ", body.name).strip()[:80]
        fields.update(name=name, name_source="user" if name else "", name_note=None)
        if name and project["voice_slug"]:
            try:
                voices.rename(project["voice_slug"], name)  # Forge shows the new name straight away
            except (OSError, ValueError):
                pass  # removed from Forge meanwhile
    if body.accent is not None:
        if body.accent not in config.BASES:
            raise HTTPException(400, "Unknown accent")
        if (project["training"] or {}).get("round") and body.accent != project["accent"]:
            raise HTTPException(409, "The accent can’t change once training has started: "
                                     "it decides which voice training starts from. Start a new voice instead.")
        fields["accent"] = body.accent
    if body.auto_train is not None:
        fields["auto_train"] = int(body.auto_train)
    if body.model is not None:
        try:
            clone.check_model(body.model)
        except clone.CantClone as exc:
            raise HTTPException(exc.status, str(exc))
        fields["model"] = body.model
    if fields:
        db.update_project(project_id, **fields)
    return project_json(db.get_project(project_id), full=True)


@app.delete(BASE + "/api/projects/{project_id}")
def delete_project(project_id: str):
    """Remove the project and its recordings. A voice already in Forge stays."""
    project = require_project(project_id)
    if project["status"] in TRAINING:
        raise HTTPException(409, "This voice is training. Stop it first.")
    jobs.stop_analysis(project_id)
    db.cancel_jobs(project_id)
    db.delete_project(project_id)
    shutil.rmtree(config.PROJECTS_DIR / project_id, ignore_errors=True)
    return {"deleted": project_id}


# --- Uploads: in parts, so huge files survive a hiccup and pass any proxy limit ---

class NewUpload(BaseModel):
    filename: str
    size: int


@app.post(BASE + "/api/projects/{project_id}/uploads")
def start_upload(project_id: str, body: NewUpload):
    require_project(project_id)
    name = Path(body.filename or "recording").name[:200]
    ext = Path(name).suffix.lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, f"{name} isn’t an audio file the voice creator can read. "
                                 f"Use M4B, MP3, FLAC, WAV or similar.")
    # The same file again while an earlier upload of it never finished (a closed
    # tab, a restart): carry on from where that one got to.
    for s in db.sources(project_id):
        if s["status"] == "uploading" and s["filename"] == name and s["size"] == body.size:
            part = Path(s["path"] + ".part")
            db.update_source(s["id"], received=part.stat().st_size if part.exists() else 0)
            return source_json(db.get_source(s["id"]))
    free = shutil.disk_usage(config.DATA_DIR).free
    if body.size > free - 2 * 1024**3:
        raise HTTPException(507, f"There isn’t room for {name}: {body.size / 1e9:.1f} GB, with "
                                 f"{free / 1e9:.1f} GB free. Free up some space and try again.")
    source_id = _new_id()
    folder = config.PROJECTS_DIR / project_id / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source_id}{ext}"
    path.with_name(path.name + ".part").touch()
    db.create_source(source_id, project_id, name, str(path), max(0, body.size))
    return source_json(db.get_source(source_id))


@app.get(BASE + "/api/projects/{project_id}/uploads/{source_id}")
def upload_state(project_id: str, source_id: str):
    return source_json(require_source(project_id, source_id))


@app.put(BASE + "/api/projects/{project_id}/uploads/{source_id}")
async def upload_part(project_id: str, source_id: str, request: Request, offset: int = 0):
    """Append one part, streamed straight to disk. offset must be what has
    arrived so far; otherwise the answer says where to carry on."""
    source = require_source(project_id, source_id)
    if source["status"] != "uploading":
        raise HTTPException(409, "This recording has already arrived.")
    part = Path(source["path"] + ".part")
    have = part.stat().st_size if part.exists() else 0
    if offset != have:
        raise HTTPException(409, {"message": "Carry on from where the upload got to.", "received": have})
    written = 0
    with part.open("ab") as out:
        async for chunk in request.stream():
            out.write(chunk)
            written += len(chunk)
    received = have + written
    db.update_source(source_id, received=received)
    return {"received": received}


@app.post(BASE + "/api/projects/{project_id}/uploads/{source_id}/done")
def finish_upload(project_id: str, source_id: str):
    project = require_project(project_id)
    source = require_source(project_id, source_id)
    if source["status"] != "uploading":
        return source_json(source)
    part = Path(source["path"] + ".part")
    have = part.stat().st_size if part.exists() else 0
    if source["size"] and have != source["size"]:
        raise HTTPException(409, {"message": "Part of the file is missing. Carry on uploading.", "received": have})
    os.replace(part, source["path"])
    try:
        info = media.probe(source["path"])
    except Exception as exc:
        db.update_source(source_id, status="failed", received=have,
                         message=f"This isn’t audio the voice creator can read ({exc}).")
        return source_json(db.get_source(source_id))
    if info["duration"] < 5:
        db.update_source(source_id, status="failed", received=have, duration=info["duration"],
                         message="This recording is only a few seconds long.")
        return source_json(db.get_source(source_id))
    # The same audio twice (a sync tool's "conflicted copy", say) would count its
    # speech twice and could put a held-out sentence into training.
    try:
        fp = media.fingerprint(source["path"])
    except Exception:
        fp = b""
    twin = next((s for s in db.sources(project_id) if s["id"] != source_id and s["fingerprint"]
                 and s["status"] not in ("failed", "duplicate", "uploading")
                 and abs(float(s["duration"] or 0) - info["duration"]) <= max(2.0, 0.02 * info["duration"])
                 and media.same_start(fp, s["fingerprint"])), None) if fp else None
    def is_copy(name: str) -> bool:
        return bool(re.search(db.CONFLICTED_RE, name, re.I))

    if twin and is_copy(twin["filename"]) and not is_copy(source["filename"]):
        # The original came second: it's kept, and the conflicted copy steps aside.
        db.conn().execute("DELETE FROM clips WHERE source_id = ?", (twin["id"],))
        Path(twin["path"]).unlink(missing_ok=True)
        db.update_source(twin["id"], status="duplicate",
                         message=f"Skipped: a conflicted copy of “{source['filename']}”, so nothing is counted twice.")
        twin = None
    if twin:
        Path(source["path"]).unlink(missing_ok=True)
        kind = "a conflicted copy of" if is_copy(source["filename"]) else "the same audio as"
        db.update_source(source_id, status="duplicate", received=have, duration=info["duration"],
                         tags=info["tags"], fingerprint=fp,
                         message=f"Skipped: {kind} “{twin['filename']}”, so nothing is counted twice.")
        return source_json(db.get_source(source_id))
    db.update_source(source_id, status="queued", received=have, duration=info["duration"],
                     tags=info["tags"], chapters=info["chapters"], fingerprint=fp, message="Waiting to be heard")
    # Who narrates it: the tags say first; Whisper listens for "read by" at the
    # start of the book's first part later.
    found = media.narrator_from_tags(info["tags"])
    if found and project["name_source"] not in ("user", "tags"):
        name, tag = found
        db.update_project(project_id, name=name, name_source="tags",
                          name_note=f"From the file’s {tag} tag")
    accent = media.accent_from_tags(info["tags"])
    if accent and not (project["training"] or {}).get("round") and len(db.sources(project_id)) == 1:
        db.update_project(project_id, accent=accent)
    if project["status"] in ("new", "failed"):
        db.update_project(project_id, status="analyzing", message="Waiting to listen to the recordings",
                          progress=0.0)
    db.add_job(project_id, "analyze")
    return source_json(db.get_source(source_id))


@app.delete(BASE + "/api/projects/{project_id}/sources/{source_id}")
def delete_source(project_id: str, source_id: str):
    project = require_project(project_id)
    source = require_source(project_id, source_id)
    if jobs.running(project_id, "analyze"):
        jobs.stop_analysis(project_id)
    db.delete_source(source_id)
    for path in (Path(source["path"]), Path(source["path"] + ".part")):
        path.unlink(missing_ok=True)
    if db.sources(project_id):
        db.add_job(project_id, "analyze")  # finish the others, then redo the report without this one
    else:
        db.update_project(project_id, readiness=None, message="Add recordings of the narrator",
                          **({"status": "new"} if project["status"] not in TRAINING else {}))
    return {"deleted": source_id}


@app.post(BASE + "/api/projects/{project_id}/stop-listening")
def stop_listening(project_id: str):
    """Use what's been heard so far: the rest of each recording is skipped."""
    from . import readiness
    require_project(project_id)
    jobs.stop_analysis(project_id)
    for _ in range(50):
        if not jobs.running(project_id, "analyze"):
            break
        time.sleep(0.1)
    for s in db.sources(project_id):
        if s["status"] in ("queued", "analyzing"):
            gaps = dict(s["gaps"] or {})
            gaps["unheard"] = gaps.get("unheard", 0) + float(s["duration"] or 0) - float(s["done_until"] or 0)
            db.update_source(s["id"], status="done", gaps=gaps, message=None)
    report = readiness.finalize(project_id)
    project = db.get_project(project_id)
    fields = {"readiness": report}
    if project["status"] not in TRAINING:
        fields.update(status="ready", progress=1.0,
                      message=f"{report['rating']['label']}: {report['usable_minutes']:g} usable minutes")
    db.update_project(project_id, **fields)
    if project["status"] not in TRAINING:
        jobs.auto_start(project_id)
    return project_json(db.get_project(project_id), full=True)


# --- Training ---------------------------------------------------------------------

@app.post(BASE + "/api/projects/{project_id}/train")
def start_training(project_id: str, from_clone: bool = False):
    """from_clone: a new round learns from the project's cloned voice (lab/train.py)."""
    require_project(project_id)
    try:
        jobs.queue_training(project_id, from_clone=from_clone)
    except jobs.CantTrain as exc:
        raise HTTPException(exc.status, str(exc))
    return project_json(db.get_project(project_id), full=True)


@app.post(BASE + "/api/projects/{project_id}/stop")
def stop_training(project_id: str):
    """Stop training. Its checks stay, to listen to and pick one for Forge."""
    project = require_project(project_id)
    state = project["training"] or {}
    if project["status"] not in TRAINING:
        return project_json(project, full=True)
    if not jobs.running(project_id, "train"):
        db.cancel_jobs(project_id, "train")
        db.update_project(project_id, status="ready", message="Training was cancelled before it started.")
        return project_json(db.get_project(project_id), full=True)
    stop = dataset.project_dir(project_id) / "train" / f"round-{int(state.get('round') or 1)}" / "stop"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.touch()
    db.update_project(project_id, message="Stopping")
    return project_json(db.get_project(project_id), full=True)


# --- Cloned voices (lab/clone.py) ------------------------------------------------------

@app.post(BASE + "/api/projects/{project_id}/clone")
def make_samples(project_id: str):
    """Try a few clips of the narrator as the reference: one sample each, to pick from."""
    require_project(project_id)
    try:
        clone.queue(project_id)
    except clone.CantClone as exc:
        raise HTTPException(exc.status, str(exc))
    return project_json(db.get_project(project_id), full=True)


@app.get(BASE + "/api/projects/{project_id}/clone/{clip_id}/ref.wav")
def clone_ref(project_id: str, clip_id: int):
    """The narrator's own clip that a sample was cloned from."""
    require_project(project_id)
    path = clone.ref_path(project_id, clip_id)
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


@app.get(BASE + "/api/projects/{project_id}/clone/{clip_id}/sample.wav")
def clone_sample(project_id: str, clip_id: int):
    """The sample passage cloned from that clip, with the project's model."""
    path = clone.sample_path(project_id, clone.effective_model(require_project(project_id)), clip_id)
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


@app.post(BASE + "/api/projects/{project_id}/clone/{clip_id}/use")
def use_clone(project_id: str, clip_id: int):
    """Put the voice cloned from this clip into Forge, picked by ear. Picking
    another clip later replaces it."""
    project = require_project(project_id)
    if not (project["name"] or "").strip():
        raise HTTPException(400, "Name the voice first: that’s the name Forge lists it under.")
    model = clone.effective_model(project)
    state = project["clone"] or {}
    candidate = next((c for c in state.get("candidates") or [] if c["clip"] == clip_id), None)
    ref = clone.ref_path(project_id, clip_id)
    if state.get("model") != model or candidate is None or not ref.exists():
        raise HTTPException(404, "There’s no sample from that clip with this model: make samples first.")
    slug = project["voice_slug"] if project["voice_slug"] and voices.is_clone(project["voice_slug"]) \
        else voices.free_slug(project["name"])
    voices.install_clone(slug, ref, candidate["text"], {
        "name": project["name"], "model": model, "project": project_id, "clip": clip_id,
        "match": candidate["likeness"]}, sample=clone.sample_path(project_id, model, clip_id))
    db.update_project(project_id, voice_slug=slug, status="done", progress=1.0,
                      message=f"Cloned with {clone.label(model)} and in Forge’s voice list (picked by ear).")
    return project_json(db.get_project(project_id), full=True)


# --- Listening --------------------------------------------------------------------------

def _wav(audio, rate: int) -> Response:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, rate, subtype="PCM_16", format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})


@app.get(BASE + "/api/projects/{project_id}/clips/{clip_id}.wav")
def clip_audio(project_id: str, clip_id: int):
    """One clip, cut from its recording on the spot."""
    require_project(project_id)
    clip = db.get_clip(clip_id)
    if clip is None or clip["project_id"] != project_id:
        raise HTTPException(404, "Clip not found")
    source = db.get_source(clip["source_id"])
    audio = media.decode(source["path"], clip["start"], clip["end"] - clip["start"])
    return _wav(audio, config.SAMPLE_RATE)


@app.get(BASE + "/api/projects/{project_id}/clips/{clip_id}")
def clip_info(project_id: str, clip_id: int):
    require_project(project_id)
    clip = db.get_clip(clip_id)
    if clip is None or clip["project_id"] != project_id:
        raise HTTPException(404, "Clip not found")
    return {k: clip[k] for k in ("id", "text", "verdict", "role", "start", "end")}


@app.get(BASE + "/api/projects/{project_id}/along/{step}.wav")
def along_sample(project_id: str, step: int):
    """The sample passage, as the voice read it at this training check."""
    path = along_path(require_project(project_id), step).with_suffix(".wav")
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


@app.post(BASE + "/api/projects/{project_id}/along/{step}/use")
def use_along(project_id: str, step: int):
    """Put the voice as it was at this check into Forge, picked by ear. Training,
    if it's still going, carries on; what it ends with replaces this again."""
    project = require_project(project_id)
    onnx = along_path(project, step).with_suffix(".onnx")
    if not onnx.exists():
        raise HTTPException(404, "The voice from that check isn’t kept any more: only the latest few are.")
    piper_config = json.loads(onnx.with_name(onnx.name + ".json").read_text(encoding="utf-8"))
    # A cloned voice in Forge stays beside it, under its own name.
    slug = project["voice_slug"] if project["voice_slug"] and not voices.is_clone(project["voice_slug"]) \
        else voices.free_slug(project["name"] or "voice")
    voices.install(slug, onnx, piper_config, {
        "name": project["name"], "project": project_id, "accent": project["accent"],
        "base": config.start_of(project)["voice"], "round": (project["training"] or {}).get("round"), "step": step,
        "minutes": (project["readiness"] or {}).get("train_minutes")})
    check = next((c for c in (project["training"] or {}).get("likeness") or [] if c[0] == step), None)
    if check:
        voices.set_match(slug, check[1])
    fields = {"voice_slug": slug}
    if project["status"] not in TRAINING:
        fields["message"] = f"The voice from step {step:,} is in Forge’s voice list (picked by ear)."
    db.update_project(project_id, **fields)
    return project_json(db.get_project(project_id), full=True)


@app.get(BASE + "/api/projects/{project_id}/quality/{round_no}/{kind}/{name}")
def quality_audio(project_id: str, round_no: int, kind: str, name: str):
    require_project(project_id)
    if kind not in ("new", "base", "real") or not re.fullmatch(r"c\d+\.wav", name):
        raise HTTPException(404, "Not found")
    base = dataset.project_dir(project_id)
    path = base / "holdout" / name if kind == "real" else base / "quality" / f"round-{round_no}" / kind / name
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav")


# --- Voices in Forge ---------------------------------------------------------------------

@app.get(BASE + "/api/voices")
def list_voices():
    return voices.listing()


class VoicePatch(BaseModel):
    name: str | None = None
    model: str | None = None  # a cloned voice only: read with another model from now on


@app.patch(BASE + "/api/voices/{slug}")
def patch_voice(slug: str, body: VoicePatch):
    if body.model is not None:
        try:
            choice = clone.check_model(body.model)
            if choice["kind"] != "clone":
                raise clone.CantClone("A cloned voice can only change to another clone model.")
            voices.set_model(slug, body.model)
        except clone.CantClone as exc:
            raise HTTPException(exc.status, str(exc))
        except (OSError, ValueError):
            raise HTTPException(404, "That cloned voice isn’t in Forge any more.")
    if body.name is not None:
        name = re.sub(r"\s+", " ", body.name).strip()[:80]
        if not name:
            raise HTTPException(400, "Give the voice a name.")
        try:
            voices.rename(slug, name)
        except (OSError, ValueError):
            raise HTTPException(404, "That voice isn’t in Forge any more.")
        for p in db.list_projects():
            if p["voice_slug"] == slug:
                db.update_project(p["id"], name=name, name_source="user", name_note=None)
    return voices.listing()


@app.get(BASE + "/api/voices/{slug}/sample")
def voice_sample(slug: str):
    """About 10 seconds of the voice reading the same passage as every other voice
    (made on first request if the quality check didn't make it already)."""
    from app.clone import CloneError
    try:
        path = voices.make_sample(slug)
    except (OSError, ValueError):
        raise HTTPException(404, "That voice isn’t in Forge any more.")
    except CloneError as exc:
        raise HTTPException(503, str(exc))
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-cache"})


@app.delete(BASE + "/api/voices/{slug}")
def delete_voice(slug: str):
    try:
        voices.remove(slug)
    except ValueError:
        raise HTTPException(404, "That voice isn’t in Forge any more.")
    for p in db.list_projects():
        if p["voice_slug"] == slug:
            put_back = ("pick a sample with Use this one to put it back"
                        if clone.effective_model(p) != clone.PIPER
                        else "pick a check (train further first if you like) to put a version back")
            db.update_project(p["id"], voice_slug=None, message=(
                f"Removed from Forge. Open it and {put_back}." if p["status"] == "done" else p["message"]))
    return voices.listing()


# --- Live updates ---------------------------------------------------------------------------

@app.get(BASE + "/api/events")
async def events(request: Request):
    """A server-sent event whenever any project changes; the page then fetches what it shows."""
    async def stream():
        last, idle = None, 0
        while not await request.is_disconnected():
            stamp = await asyncio.to_thread(db.stamp)
            if stamp != last:
                last, idle = stamp, 0
                yield f"event: changed\ndata: {json.dumps(list(stamp))}\n\n"
            else:
                idle += 1
                if idle % 15 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
