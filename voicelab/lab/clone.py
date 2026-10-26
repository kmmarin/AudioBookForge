"""Cloned voices: a narrator's voice made by a clone server (app/clone.py,
Qwen3-TTS) from one clip of them, with no training.

Each voice project picks its model: a clone model, or "piper", the old way of
training a voice for hours. For a clone model, a few clean clips of the
narrator are tried as the reference: each makes the same sample passage, its
likeness to the narrator is measured, and the person listens and picks one
(Use this one in lab/main.py), which goes into Forge as the voice.

Runs as its own job (python -m lab.clone <project id>) in a lane of its own.
The clone server does the GPU work; this only cuts clips, asks and measures,
on the CPU.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

from app.clone import MODELS, Client, CloneError

from . import config, dataset, db, media

log = logging.getLogger("voicelab.clone")
PIPER = "piper"
PIPER_LABEL = "Piper, trained (the old way: hours of training)"
HEALTH_SECONDS = 30
_health: tuple[float, dict | None] | None = None


class CantClone(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def configured() -> bool:
    """A GPU server does voices: in Forge's Settings (app/gpus.py, which lists CLONE_URL too)."""
    from app import gpus
    return gpus.has("voices") or bool(config.CLONE_URL)


def client(model: str | None = None) -> Client:
    """The GPU server doing voices: the first in Settings that answers and, for
    a model, can run it now; this machine's own clone server when it's the one
    (app/gpus.py)."""
    from app import gpus
    server = gpus.pick("voices", model)
    if server is None:
        return Client(config.CLONE_URL, config.CLONE_TOKEN)
    return Client(server["url"], server.get("token") or "")


def health(max_age: float = HEALTH_SECONDS) -> dict | None:
    """What the voice servers can run, together (cached briefly): a model fits
    when any server that answers fits it. None without a clone server;
    {"ok": False, "error"} when none is answering."""
    global _health
    from app import gpus
    if not configured():
        return None
    if _health is None or time.monotonic() - _health[0] > max_age:
        answering = [h for h in (gpus.health(s, max_age) for s in gpus.servers() if s.get("voices")) if h.get("ok")]
        if not answering and config.CLONE_URL and not gpus.has("voices"):
            try:  # a voice creator set up with CLONE_URL alone
                answering = [client().health(timeout=5)]
            except CloneError:
                pass
        if answering:
            models: dict[str, dict] = {}
            for h in answering:  # the first server that fits a model speaks for it
                for m in h.get("models") or []:
                    if m["id"] not in models or (m.get("fits") and not models[m["id"]].get("fits")):
                        models[m["id"]] = m
            state = {**answering[0], "ok": True, "models": list(models.values())}
        else:
            state = {"ok": False, "error": "The clone server isn’t answering."}
        _health = (time.monotonic(), state)
    return _health[1]


def model_choices(state: dict | None) -> list[dict]:
    """Every model a voice can be made with, best first, and whether this setup runs it."""
    served = {m["id"]: m for m in state.get("models") or []} if state and state.get("ok") else {}
    out = []
    for model_id, label in MODELS.items():
        if not configured():
            fits, note = False, "Needs a GPU server for voices (Forge’s Settings)."
        elif not state or not state.get("ok"):
            fits, note = False, "The clone server isn’t answering."
        elif model_id not in served:
            fits, note = False, "The clone server doesn’t offer it."
        else:
            fits, note = bool(served[model_id].get("fits")), served[model_id].get("note") or ""
        out.append({"id": model_id, "label": label, "kind": "clone", "fits": fits, "note": note})
    out.append({"id": PIPER, "label": PIPER_LABEL, "kind": "piper", "fits": True,
                "note": "Trains on this machine’s GPU for hours; voices made this way so far sounded buzzy."})
    return out


def default_model(choices: list[dict]) -> str:
    """The best clone model that fits; with a clone server that isn't answering,
    the smallest (it fits most GPUs); with no clone server, Piper."""
    clones = [c for c in choices if c["kind"] == "clone"]
    fitting = [c for c in clones if c["fits"]]
    if fitting:
        return fitting[0]["id"]
    return clones[-1]["id"] if configured() and clones else PIPER


def effective_model(project: dict, choices: list[dict] | None = None) -> str:
    return project.get("model") or default_model(choices if choices is not None else model_choices(health()))


def check_model(model: str, choices: list[dict] | None = None) -> dict:
    """The choice for this model, or CantClone saying why it can't be used."""
    choices = choices if choices is not None else model_choices(health())
    choice = next((c for c in choices if c["id"] == model), None)
    if choice is None:
        raise CantClone("Unknown model")
    if choice["kind"] == "clone":
        if not configured():
            raise CantClone("Cloned voices need a GPU server for voices: add one in Forge’s Settings.")
        state = health()
        if state and state.get("ok") and not choice["fits"]:
            raise CantClone(f"{choice['label']} can’t run on the clone server: {choice['note'] or 'it doesn’t fit'}")
    return choice


def label(model: str) -> str:
    return MODELS.get(model) or (PIPER_LABEL if model == PIPER else model)


# --------------------------------------------------------------------------
# Picking reference clips
# --------------------------------------------------------------------------

def _score(c: dict) -> float:
    """Whisper sure of every word, a quiet background, and the narrator's usual
    voice: a clip of them doing a character's voice clones that character."""
    return ((c.get("conf") or 0.0) - (c.get("unsure") or 0.0) + 0.004 * min(float(c.get("snr") or 0.0), 50.0)
            + (c.get("sim") or 0.0))


def pick_candidates(clips: list[dict], n: int | None = None) -> list[dict]:
    """Up to n clean clips of the narrator to try as the reference: 8-14 s,
    kept, the clearest first, preferring different recordings and stretches
    of a recording far apart (a narrator sounds different chapter to chapter)."""
    n = n or config.CLONE_CANDIDATES
    pool = [c for c in clips if c.get("verdict") == "keep" and c.get("role") in ("train", "holdout")
            and config.CLONE_REF_MIN <= c["end"] - c["start"] <= config.CLONE_REF_MAX]
    ranked = sorted(pool, key=_score, reverse=True)[:max(40, n * 10)]
    picked: list[dict] = []

    def value(c: dict) -> float:
        v = _score(c)
        for p in picked:
            if p["source_id"] == c["source_id"]:
                v -= 0.05  # another recording, if there's a good one
                if abs(p["start"] - c["start"]) < 600:
                    v -= 0.1  # or at least another stretch of this one
        return v

    while ranked and len(picked) < n:
        best = max(ranked, key=value)
        picked.append(best)
        ranked.remove(best)
    return picked


def ref_path(project_id: str, clip_id: int) -> Path:
    return dataset.project_dir(project_id) / "clone" / f"ref-{clip_id}.wav"


def sample_path(project_id: str, model: str, clip_id: int) -> Path:
    return dataset.project_dir(project_id) / "clone" / model / f"sample-{clip_id}.wav"


# --------------------------------------------------------------------------
# Queueing and the job
# --------------------------------------------------------------------------

def queue(project_id: str, model: str | None = None, waiting_ok: bool = False) -> str:
    """Queue making samples with the project's model. Returns the model.
    waiting_ok: called as the listening finishes, while its job still counts as running."""
    project = db.get_project(project_id)
    model = model or effective_model(project)
    choice = check_model(model)
    if choice["kind"] != "clone":
        raise CantClone("This voice is set to be trained with Piper: pick a clone model first.")
    if not waiting_ok and db.job_waiting(project_id, "analyze"):
        raise CantClone("The recordings are still being heard. Make samples once the report is ready.", 409)
    if not pick_candidates(db.clips(project_id)):
        raise CantClone("There’s no clean 8 to 14 second clip of the narrator yet. Add more recordings.")
    state = dict(project["clone"] or {})
    if state.get("model") != model:
        state = {"model": model}
    state["stage"] = "queued"
    db.update_project(project_id, clone=state, message=f"Waiting to make samples with {label(model)}")
    db.add_job(project_id, "clone")
    return model


def _set(project_id: str, **changes) -> dict:
    state = dict(db.get_project(project_id).get("clone") or {})
    state.update(changes)
    db.update_project(project_id, clone=state)
    return state


def main(project_id: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    from . import quality, speaker, voices
    project = db.get_project(project_id)
    if project is None:
        return 0
    state = project.get("clone") or {}
    model = state.get("model") or effective_model(project)
    tell = lambda text: db.update_project(project_id, message=text)  # noqa: E731
    clips = pick_candidates(db.clips(project_id))
    if not clips:
        _set(project_id, stage="failed")
        tell("There’s no clean 8 to 14 second clip of the narrator yet. Add more recordings.")
        return 0
    paths = {s["id"]: s["path"] for s in db.sources(project_id)}
    try:
        tell("Making samples: getting the narrator’s clips ready")
        _set(project_id, model=model, stage="working")
        # Measured against the held-out clips, like a trained voice, so the numbers compare.
        if not (dataset.project_dir(project_id) / "holdout" / "holdout.json").exists():
            dataset.write_holdout(project_id)
        config.WHISPER_DEVICE = "cpu"  # the speaker model stays off the GPU the clone server may be using
        probe = quality.Probe(project_id, config.LIKENESS_SENTENCES)
        rows = []
        for n, c in enumerate(clips, 1):
            tell(f"Making samples with {label(model)}: {n} of {len(clips)}")
            ref = ref_path(project_id, c["id"])
            if not ref.exists():
                ref.parent.mkdir(parents=True, exist_ok=True)
                media.write_wav(ref, media.decode(paths[c["source_id"]], c["start"], c["end"] - c["start"],
                                                  rate=config.CLONE_RATE), config.CLONE_RATE)
            rate, (audio,) = client(model).speak(model, ref, c["text"].strip(), [voices.SAMPLE_TEXT])
            out = sample_path(project_id, model, c["id"])
            out.parent.mkdir(parents=True, exist_ok=True)
            media.write_wav(out, audio, rate)
            likeness = None
            if probe.center is not None:
                likeness = round(float(speaker.embed([media.resample(audio, rate, speaker.RATE)])[0] @ probe.center), 3)
            rows.append({"clip": c["id"], "text": c["text"].strip(), "seconds": round(c["end"] - c["start"], 1),
                         "likeness": likeness, "artifacts": quality.artifacts([audio], rate)})
        _set(project_id, model=model, stage="done", made=time.time(), candidates=rows,
             real_likeness=round(probe.ceiling, 3) if probe.ceiling else None, real_artifacts=probe.real_artifacts)
        project = db.get_project(project_id)
        fields = {"message": f"Samples made with {label(model)}: listen and pick one to put it into Forge."}
        if project["status"] in ("new", "analyzing", "failed"):
            fields.update(status="ready", progress=1.0)
        db.update_project(project_id, **fields)
        return 0
    except CloneError as exc:
        log.warning("Making samples for %s: %s", project_id, exc)
        _set(project_id, stage="failed")
        tell(f"Couldn’t make samples: {exc}")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
