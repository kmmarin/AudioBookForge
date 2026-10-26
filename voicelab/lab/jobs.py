"""Background work: hearing recordings and training voices.

The web app runs two lanes, analysis and training, each one job at a time.
Every job is its own process, so heavy libraries and their memory never sit
in the web app, a crash only ends that job, and Stop can always end it. Jobs
save their progress as they go; after a restart, interrupted jobs run again
and carry on where they stopped.

    python -m lab.jobs analyze <project id>     (what the analysis lane runs)
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time

from . import config, db

log = logging.getLogger("voicelab.jobs")
# quality: measuring a finished voice, on the CPU; clone: samples of a cloned
# voice, made by the clone server (lab/clone.py)
LANES = ("analyze", "train", "quality", "clone")
COMMANDS = {"train": ["lab.train"], "quality": ["lab.train", "--quality"], "clone": ["lab.clone"]}


class Lane(threading.Thread):
    def __init__(self, kind: str):
        super().__init__(daemon=True, name=f"lane-{kind}")
        self.kind = kind
        self.proc: subprocess.Popen | None = None
        self.job: dict | None = None
        self.stopping = threading.Event()

    def run(self) -> None:
        while not self.stopping.is_set():
            job = db.next_job(self.kind)
            if job is None:
                self.stopping.wait(1.0)
                continue
            self.job = job
            args = [sys.executable, "-m", *COMMANDS.get(self.kind, ["lab.jobs", "analyze"]), job["project_id"]]
            log.info("Job %d: %s %s", job["id"], self.kind, job["project_id"])
            try:
                self.proc = subprocess.Popen(args, cwd=str(config.APP_DIR.parent))
                code = self.proc.wait()
            except OSError as exc:
                log.error("Couldn’t start job %d: %s", job["id"], exc)
                code = 1
            self.proc = None
            if self.stopping.is_set():
                break  # shutting down: the job stays "running" and is picked up again at start
            if code == 75:  # wants another go (paused for Forge, or a smaller batch after running out of memory)
                db.finish_job(job["id"], "retry")
                db.add_job(job["project_id"], self.kind, created=job["created"])  # keeps its place in the queue
            elif code == 0:
                db.finish_job(job["id"], "done")
            elif code < 0 or code == 143:  # ended by Stop (SIGTERM)
                db.finish_job(job["id"], "cancelled")
            else:
                db.finish_job(job["id"], "failed")
                project = db.get_project(job["project_id"])
                # A failed measurement or sample leaves the voice as it is; only listening or training fail the project.
                if project and project["status"] not in ("failed",) and self.kind not in ("quality", "clone"):
                    db.update_project(job["project_id"], status="failed",
                                      message="Something went wrong. The voice creator’s log says what; "
                                              "try again, it carries on where it stopped.")
            self.job = None

    def stop_project(self, project_id: str) -> bool:
        """End this lane's job if it belongs to the project."""
        if self.job and self.job["project_id"] == project_id and self.proc:
            self.proc.send_signal(signal.SIGTERM)
            return True
        return False


_lanes: dict[str, Lane] = {}


def start() -> None:
    db.requeue_running()
    for kind in LANES:
        if kind not in _lanes:
            _lanes[kind] = Lane(kind)
            _lanes[kind].start()


def shutdown() -> None:
    for lane in _lanes.values():
        lane.stopping.set()
        if lane.proc:
            lane.proc.send_signal(signal.SIGTERM)


def running(project_id: str, kind: str) -> bool:
    lane = _lanes.get(kind)
    return bool(lane and lane.job and lane.job["project_id"] == project_id)


def stop_analysis(project_id: str) -> None:
    db.cancel_jobs(project_id, "analyze")
    if "analyze" in _lanes:
        _lanes["analyze"].stop_project(project_id)


class CantTrain(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def queue_training(project_id: str, waiting_ok: bool = False, from_clone: bool = False) -> None:
    """Queue training: the first round, the same round again after a stop or
    failure (it carries on from its last checkpoint), or, once a round is
    done, the next round from its best checkpoint with whatever audio there is.
    from_clone: a new round learns from the project's cloned voice, not the recordings."""
    project = db.get_project(project_id)
    if not (project["name"] or "").strip():
        raise CantTrain("Name the voice first: that’s the name Forge lists it under.")
    if project["status"] == "training":
        raise CantTrain("This voice is already training.", 409)
    if not waiting_ok and db.job_waiting(project_id, "analyze"):
        raise CantTrain("The recordings are still being heard. Train once the report is ready.", 409)
    report = project["readiness"] or {}
    if (report.get("train_clips") or 0) < 10:
        raise CantTrain("There aren’t enough usable clips to train on yet. Add more recordings of the narrator.")
    state = dict(project["training"] or {})
    round_no = int(state.get("round") or 0)
    if round_no == 0:
        round_no = 1
    elif state.get("stage") == "done":
        round_no += 1
    if state.get("round") != round_no:
        state["history"] = []
        state["teacher"] = None
        state["batch"] = None  # picked again from free video memory, not the last round's fallback
        if from_clone:
            from . import clone, voices
            if not clone.configured():
                raise CantTrain("Learning from the cloned voice needs a GPU server for voices: "
                                "add one in Forge’s Settings.")
            state["teacher"] = voices.cloned_for(project_id)
            if not state["teacher"]:
                raise CantTrain("Put a cloned voice of the narrator into Forge first: that’s what Piper learns from.")
    state.update(round=round_no, stage="queued")
    db.update_project(project_id, status="training", progress=0.0, training=state,
                      message="Waiting to start training")
    db.add_job(project_id, "train")


# --------------------------------------------------------------------------
# The analysis job
# --------------------------------------------------------------------------

def analyze_main(project_id: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    project = db.get_project(project_id)
    if project is None:
        return 0
    training = project["status"] == "training"

    def set_status(**fields):
        if not training:
            db.update_project(project_id, **fields)

    last = [0.0]
    totals: dict = {}

    def report(source_id: str, t0: float, seconds: float) -> None:
        if time.monotonic() - last[0] < 3:
            return
        last[0] = time.monotonic()
        heard = totals["heard"] - totals["base"].get(source_id, 0.0) + t0 + seconds
        set_status(progress=min(0.99, heard / totals["total"]),
                   message=f"Listening to the recordings: {round(heard / 60)} of {round(totals['total'] / 60)} minutes")
        db.update_source(source_id, message=f"{round((t0 + seconds) / 60)} of "
                                            f"{round(float(db.get_source(source_id)['duration'] or 0) / 60)} minutes")

    set_status(status="analyzing", progress=0.0, message="Listening to the recordings")
    # Whisper on the GPU (LAB_WHISPER_DEVICE=cuda) takes its turn like training does.
    from . import gpu
    try:
        with gpu.Lease("Hearing recordings for a new voice", on_wait=lambda text: set_status(message=text),
                       shared=config.GPU_SHARED and config.WHISPER_DEVICE == "cuda",
                       gpu=config.WHISPER_DEVICE == "cuda"):
            if config.WHISPER_DEVICE == "cuda":
                gpu.wait_for_memory(1.5, on_wait=lambda text: set_status(message=text))  # Whisper small + ECAPA
            paused = hear_all(project_id, report, totals)
    except gpu.NotEnoughMemory as exc:
        db.update_project(project_id, status="failed", message=str(exc))
        return 1
    if paused:
        set_status(message=f"{gpu.gpushare.WAITING_PREFIX}: paused while Forge voices a book. "
                           "Listening carries on from here when it’s done.")
        return 75  # runs again, in its place in the queue
    finish_analysis(project_id)
    return 0


def hear_all(project_id: str, report, totals: dict) -> bool:
    """One window at a time, always from the earliest unfinished part in book
    order: a first part added while a later one is being heard goes next, and
    recordings that arrive meanwhile are picked up too. Returns True if it
    paused for Forge (each window's results are saved, so it carries on later)."""
    from . import analyze, gpu
    while True:
        sources = [s for s in db.sources(project_id) if s["status"] not in ("duplicate", "failed", "uploading")]
        todo = [s for s in sources if s["status"] in ("queued", "analyzing")]
        if not todo:
            return False
        if config.WHISPER_DEVICE == "cuda" and gpu.forge_wants():
            log.info("%s: pausing so Forge can voice", project_id)
            return True
        totals.update(total=sum(float(s["duration"] or 0) for s in sources) or 1.0,
                      base={s["id"]: float(s["done_until"] or 0) for s in sources})
        totals["heard"] = sum(totals["base"].values())
        source = todo[0]
        if source["status"] != "analyzing":
            db.update_source(source["id"], status="analyzing", message=None)
        try:
            if analyze.analyze_source(db.get_project(project_id), db.get_source(source["id"]),
                                      should_stop=lambda: False, report=report, windows=1):
                db.update_source(source["id"], message=None)
        except ValueError as exc:
            db.update_source(source["id"], status="failed", message=str(exc))
        except Exception as exc:
            log.exception("Analysing %s failed", source["filename"])
            db.update_source(source["id"], status="failed",
                             message=f"Couldn’t read this recording: {str(exc)[:200]}")
        if heard_enough(project_id):
            return False


def heard_enough(project_id: str) -> bool:
    """A whole audiobook is far more than training uses (the best TRAIN_MAX_MINUTES):
    once the clean clips heard so far comfortably cover that, the rest is
    skipped instead of spending hours of GPU time listening to it. A cloned
    voice needs only a few clean clips to pick from: LISTEN_CLONE_MINUTES."""
    from . import clone
    clean = db.conn().execute("SELECT COALESCE(SUM(end - start), 0) FROM clips WHERE project_id = ? AND reason IS NULL",
                              (project_id,)).fetchone()[0] / 60
    enough = config.LISTEN_ENOUGH_MINUTES
    if clean >= config.LISTEN_CLONE_MINUTES and clone.effective_model(db.get_project(project_id)) != clone.PIPER:
        enough = config.LISTEN_CLONE_MINUTES
    if clean < enough:
        return False
    skipped = 0.0
    for s in db.sources(project_id):
        if s["status"] in ("queued", "analyzing"):
            gaps = dict(s["gaps"] or {})
            rest = float(s["duration"] or 0) - float(s["done_until"] or 0)
            gaps["unheard"] = gaps.get("unheard", 0) + rest
            skipped += rest
            db.update_source(s["id"], status="done", gaps=gaps,
                             message="Not needed: enough clean speech was heard already")
    log.info("%s: %.0f clean minutes heard, plenty; skipping the other %.0f minutes", project_id, clean, skipped / 60)
    return True


def finish_analysis(project_id: str) -> None:
    from . import readiness
    report_data = readiness.finalize(project_id)
    project = db.get_project(project_id)
    fields = {"readiness": report_data}
    if project["status"] != "training":
        rating = report_data["rating"]
        fields.update(status="ready", progress=1.0,
                      message=f"{rating['label']}: {report_data['usable_minutes']:g} usable minutes")
    db.update_project(project_id, **fields)
    if project["status"] != "training" and not db.job_waiting(project_id, "analyze", besides_running=True):
        auto_start(project_id)


def auto_start(project_id: str) -> None:
    """"Make the voice as soon as the audio is ready". A cloned voice: its
    samples, once per model, which need no name yet. A trained (Piper) voice:
    only for a name the person typed (a detected one waits to be confirmed),
    only for the first round. Never on "Not enough", which asks for more
    audio instead."""
    from . import clone
    project = db.get_project(project_id)
    report = project["readiness"] or {}
    model = clone.effective_model(project)
    if model != clone.PIPER:
        if (not project["auto_train"] or report.get("rating", {}).get("key") == "not_enough"
                or (project["clone"] or {}).get("model") == model):
            return
        try:
            clone.queue(project_id, model, waiting_ok=True)
            log.info("%s: the audio is ready; making samples with %s", project_id, model)
        except clone.CantClone as exc:
            db.update_project(project_id, message=str(exc))
        return
    if not project["auto_train"] or project["name_source"] != "user" or (project["training"] or {}).get("round"):
        return
    if report.get("rating", {}).get("key") == "not_enough":
        db.update_project(project_id, message="Not enough usable speech to train on yet. Add more recordings "
                                              "of the narrator: training starts by itself once there’s enough.")
        return
    try:
        queue_training(project_id, waiting_ok=True)
        log.info("%s: the audio is ready; training starts by itself", project_id)
    except CantTrain as exc:
        db.update_project(project_id, message=str(exc))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "analyze":
        os.environ.setdefault("OMP_NUM_THREADS", str(config.THREADS))
        sys.exit(analyze_main(sys.argv[2]))
    print(__doc__)
    sys.exit(2)
