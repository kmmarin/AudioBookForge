"""Making the voice: fine-tune Piper, export it, put it in Forge, measure it.

Runs as its own process (python -m lab.train <project id>), started by the
job runner, so a crash or an out-of-memory error never takes the web app with
it. Every stage is saved as it finishes, and training saves a checkpoint at
every validation, so a restart carries on where it stopped.

Training uses piper1-gpl's own VITS code (piper.train.vits, release 1.3.0,
the one Forge voices with) through Lightning, the same as its documented

    python -m piper.train fit --data.csv_path … --ckpt_path <base checkpoint>

but built here so the run can report progress, stop by itself once the voice
stops getting closer to the narrator and less distorted, keep a sample of
each check to listen to, and take turns on the GPU with Forge.

Rounds: the first fine-tunes the stock voice (for the accent) that already
sounds most like the narrator. Train further starts another round from the
last round's best checkpoint. A round can learn from the project's cloned
voice instead of the recordings (dataset.speak): it reads the narrator's
sentences for Piper, so a voice that needs no GPU can sound like the clone.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config

# Before numpy and torch start their thread pools: share the CPU with Forge.
os.environ.setdefault("OMP_NUM_THREADS", str(config.THREADS))
from app.clone import CloneError  # noqa: E402

from . import dataset, db, gpu, quality  # noqa: E402

log = logging.getLogger("voicelab.train")
RETRY = 75  # exit code: run again (after running out of video memory, with a smaller batch)
CUDA_CONTEXT_GB = 0.45


class Failed(RuntimeError):
    """A problem the person can do something about; the message says what."""


def round_dir(project_id: str, round_no: int) -> Path:
    return dataset.project_dir(project_id) / "train" / f"round-{round_no}"


def save(project_id: str, **changes) -> dict:
    """Merge changes into the project's training state."""
    state = dict(db.get_project(project_id).get("training") or {})
    state.update(changes)
    db.update_project(project_id, training=state)
    return state


def status(project_id: str, message: str, progress: float | None = None, **training) -> None:
    fields = {"message": message}
    if progress is not None:
        fields["progress"] = round(progress, 4)
    if training:
        state = dict(db.get_project(project_id).get("training") or {})
        state.update(training)
        fields["training"] = state
    db.update_project(project_id, **fields)


def base_checkpoint(start: dict) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download("rhasspy/piper-checkpoints", start["checkpoint"], repo_type="dataset")


def pick_start(project: dict, accent: dict) -> dict:
    """Before the first round: the stock voice whose released voice already
    sounds most like the narrator, measured on the held-out clips (CPU). Kept
    in the training state, so every later round and check uses the same one.
    Measured for a deep-voiced male narrator: Lessac 5%, Ryan 14%, HFC male 17%, Joe 15%."""
    if (project.get("training") or {}).get("start") or len(accent["starts"]) == 1:
        return config.start_of(project)
    from app.tts import piper_model_path
    config.WHISPER_DEVICE = "cpu"
    probe = quality.Probe(project["id"], config.LIKENESS_SENTENCES)
    if probe.center is None:
        return config.start_of(project)
    scores = {}
    for s in accent["starts"]:
        status(project["id"], f"Picking the voice to start from: trying {s['name']}", stage="dataset")
        try:
            scores[s["key"]] = round(probe.measure(Path(piper_model_path(s["voice"])))["likeness"], 3)
        except Exception:  # offline, say: the others still count
            log.warning("Couldn’t try the starting voice %s", s["key"], exc_info=True)
    if scores:
        best = max(scores, key=scores.get)
        log.info("Starting from %s: %s", best, scores)
        save(project["id"], start=best, start_scores=scores)
    return config.start_of(db.get_project(project["id"]))


def teacher_of(project_id: str, round_no: int, slug: str) -> dict:
    """The cloned voice a round learns from, copied beside its dataset once, so
    picking another clip for it meanwhile doesn't change voices halfway."""
    from . import voices
    here = dataset.project_dir(project_id) / "dataset" / f"round-{round_no}" / "teacher"
    if not (here / "voice.json").exists():
        folder = voices.clone_folder(slug)
        if not (folder / "voice.json").exists():
            raise Failed("The cloned voice this round was to learn from isn’t in Forge any more.")
        here.mkdir(parents=True, exist_ok=True)
        for name in ("ref.wav", "ref.txt", "voice.json"):  # voice.json last: then the copy is whole
            shutil.copyfile(folder / name, here / name)
    info = json.loads((here / "voice.json").read_text(encoding="utf-8"))
    return {"model": info["model"], "ref": str(here / "ref.wav"),
            "text": (here / "ref.txt").read_text(encoding="utf-8").strip()}


def start_checkpoint(project_id: str, round_no: int, start: dict) -> str:
    """Where this round starts from: its own last checkpoint after a restart,
    else the starting voice (round 1) or the previous round's best (later rounds).
    A previous round's checkpoint loses its callback state (its early-stopping
    count and best score), so the new round judges itself afresh."""
    here = round_dir(project_id, round_no)
    if (here / "last.ckpt").exists():
        return str(here / "last.ckpt")
    if round_no == 1:
        return base_checkpoint(start)
    init = here / "init.ckpt"
    if not init.exists():
        import torch
        prev = round_dir(project_id, round_no - 1)
        source = prev / "best.ckpt" if (prev / "best.ckpt").exists() else prev / "last.ckpt"
        if not source.exists():
            raise Failed("The previous training round left no checkpoint to continue from. Start over instead.")
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
        ckpt.pop("callbacks", None)
        here.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, init.with_suffix(".part"))
        os.replace(init.with_suffix(".part"), init)
    return str(init)


def make_guide(project_id: str, here: Path, max_batches: int, min_batches: int, val_every: int,
               voice_config: Path, time_limit: float | None = None):
    """A Lightning callback that reports progress, honours Stop, and ends
    training once the voice stops getting better: every LIKENESS_EVERY steps
    the voice so far is exported (on the CPU), says held-out sentences, is
    compared with the narrator's real clips and checked for distortion, and
    reads the sample passage into along/<step>.wav to listen to. A check that
    sounds closer, or as close and less distorted, becomes best.ckpt, which is
    what gets exported. Validation loss is still recorded, but it's too noisy to
    stop on and says nothing direct about either."""
    import lightning as L

    from . import quality

    class Guide(L.Callback):
        def __init__(self):
            self.batches = 0
            self.best = math.inf  # validation loss: shown, not stopped on
            self.history: list[list[float]] = []
            self.like_history: list[list[float]] = []  # [step, likeness, distortion]
            self.best_like = -1.0
            self.best_art = math.inf
            self.best_step = None  # the check best.ckpt is from: what the round ends with
            self.like_wait = 0
            self.last_probe = 0
            self.seconds = 0.0  # time spent training this round (the first round has a limit)
            self._last_batch_end = 0.0
            self.reason = "limit"
            self._last_report = 0.0
            self._last_forge_check = 0.0
            self._started = time.monotonic()
            self._start_batches = 0
            self.probe = quality.Probe(project_id, config.LIKENESS_SENTENCES)
            if self.probe.center is None:
                log.warning("Too few held-out clips to measure likeness while training: it runs to the step limit")

        def state_dict(self):
            return {"batches": self.batches, "best": self.best, "history": self.history,
                    "like_history": self.like_history, "best_like": self.best_like, "best_art": self.best_art,
                    "like_wait": self.like_wait, "last_probe": self.last_probe, "seconds": self.seconds,
                    "best_step": self.best_step}

        def load_state_dict(self, state):
            self.batches = state.get("batches", 0)
            self.best = state.get("best", math.inf)
            self.history = state.get("history", [])
            self.like_history = state.get("like_history", [])
            self.best_like = state.get("best_like", -1.0)
            self.best_art = state.get("best_art", math.inf)
            self.best_step = state.get("best_step")
            self.like_wait = state.get("like_wait", 0)
            self.last_probe = state.get("last_probe", 0)
            self.seconds = state.get("seconds", 0.0)

        def on_train_start(self, trainer, pl_module):
            self._started, self._start_batches = time.monotonic(), self.batches

        def _likeness_text(self) -> str:
            text = ""
            if time_limit:
                left = max(0, math.ceil((time_limit - self.seconds) / 60))
                text = f" · first round, about {left} minute{'' if left == 1 else 's'} left"
            if not self.like_history:
                return text or f" (the first check of how much it sounds like the narrator comes at {config.LIKENESS_EVERY:,})"
            text += f" · sounds like the narrator {round(self.best_like * 100)}%"
            if self.probe.ceiling:
                text += f" (their own recordings score {round(self.probe.ceiling * 100)}%)"
            if len(self.like_history[-1]) < 3:
                return text  # checked before distortion was measured
            text += f" · distortion {self.like_history[-1][2]:.1f} dB"
            if self.probe.real_artifacts is not None:
                text += f" (their recordings {self.probe.real_artifacts:.1f})"
            return text

        def _report(self, force=False):
            now = time.monotonic()
            if not force and now - self._last_report < 5:
                return
            self._last_report = now
            done = self.batches - self._start_batches
            rate = done / max(1e-6, now - self._started)
            ceiling = self.probe.ceiling
            progress = (min(0.99, max(0.0, self.best_like) / ceiling) if ceiling and self.like_history
                        else min(0.99, self.batches / max_batches))
            status(project_id, f"Training: {self.batches:,} steps{self._likeness_text()}",
                   progress=progress, stage="training", batches=self.batches,
                   max_batches=max_batches, history=self.history[-200:],
                   best=None if math.isinf(self.best) else round(self.best, 3),
                   likeness=self.like_history[-50:], best_likeness=round(self.best_like, 3) if self.like_history else None,
                   real_likeness=round(ceiling, 3) if ceiling else None, real_artifacts=self.probe.real_artifacts,
                   per_batch=round(1 / rate, 2) if rate > 0 else None,
                   patience_left=config.LIKENESS_PATIENCE - self.like_wait, best_step=self.best_step)

        def _check_likeness(self, trainer):
            """Export the voice so far and measure it. CPU only (CUDA hidden from
            the exporter), so the GPU's capped memory is left alone."""
            self.last_probe = self.batches
            status(project_id, f"Training: {self.batches:,} steps · checking how much it sounds like the narrator")
            ckpt, onnx = here / "probe.ckpt", here / "probe.onnx"
            trainer.save_checkpoint(str(ckpt))
            run = subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ckpt),
                                  "--output-file", str(onnx)], capture_output=True, text=True,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            if run.returncode != 0 or not onnx.exists():
                log.warning("Couldn’t export the voice to check it: %s", run.stderr.strip()[-300:])
                ckpt.unlink(missing_ok=True)
                return
            shutil.copyfile(voice_config, onnx.with_name(onnx.name + ".json"))  # Piper reads it beside the model
            along = here / "along"
            m = self.probe.measure(onnx, sample_to=along / f"{self.batches:06d}.wav")
            like, art = m["likeness"], m["artifacts"]
            # The voice as it was at this check, to pick by ear; the latest few are kept.
            os.replace(onnx, along / f"{self.batches:06d}.onnx")
            os.replace(onnx.with_name(onnx.name + ".json"), along / f"{self.batches:06d}.onnx.json")
            for old in sorted(along.glob("*.onnx"))[:-config.ALONG_KEEP]:
                old.unlink(missing_ok=True)
                old.with_name(old.name + ".json").unlink(missing_ok=True)
            self.like_history.append([self.batches, round(like, 3), art])
            closer = like > self.best_like + config.LIKENESS_MIN_GAIN
            cleaner = art < self.best_art - config.CLEAN_MIN_GAIN and like >= self.best_like - config.LIKENESS_SLACK
            if closer or cleaner:
                self.best_like, self.best_art, self.like_wait = max(like, self.best_like), min(art, self.best_art), 0
                self.best_step = self.batches
                os.replace(ckpt, here / "best.ckpt")  # the best so far: this is what becomes the voice
            else:
                self.like_wait += 1
                ckpt.unlink(missing_ok=True)
                if self.like_wait >= config.LIKENESS_PATIENCE and self.batches >= min_batches:
                    self.reason = "converged"
                    trainer.should_stop = True
            log.info("Step %d: sounds like the narrator %.3f (best %.3f, their own clips %.3f), "
                     "distortion %.2f dB (best %.2f, their own clips %.2f)", self.batches, like, self.best_like,
                     self.probe.ceiling or 0, art, self.best_art, self.probe.real_artifacts or 0)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.batches += 1
            now = time.monotonic()
            if self._last_batch_end:
                self.seconds += min(now - self._last_batch_end, 30)  # a validation or check counts little
            self._last_batch_end = now
            if (here / "stop").exists():
                self.reason = "stopped"
                trainer.should_stop = True
            elif self.batches >= max_batches:
                self.reason = "limit"
                trainer.should_stop = True
            elif time_limit and self.seconds >= time_limit:
                # The first round is done. One last check, so the voice it ends
                # with is the closer of now and the last check.
                if self.probe.center is not None and self.batches > self.last_probe:
                    self._check_likeness(trainer)
                self.reason = "time"
                trainer.should_stop = True
            elif time.monotonic() - self._last_forge_check > 5:
                self._last_forge_check = time.monotonic()
                if gpu.forge_wants():
                    # Forge has a book or sample to voice: it goes first. Save this
                    # exact step; training carries on from it once Forge is done.
                    trainer.save_checkpoint(str(here / "last.ckpt"))
                    log.info("Step %d: pausing so Forge can voice", self.batches)
                    self.reason = "paused"
                    trainer.should_stop = True
            loss = trainer.callback_metrics.get("loss_g")
            if loss is not None and not math.isfinite(float(loss)):
                self.reason = "unstable"
                trainer.should_stop = True
            self._report()

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking or "val_loss" not in trainer.callback_metrics:
                return
            val = float(trainer.callback_metrics["val_loss"])
            self.history.append([self.batches, round(val, 3)])
            if not math.isfinite(val):
                self.reason = "unstable"
                trainer.should_stop = True
                return
            self.best = min(self.best, val)
            if self.probe.center is not None and self.batches - self.last_probe >= config.LIKENESS_EVERY:
                self._check_likeness(trainer)
            self._report(force=True)

    return Guide()


def pick_batch(state: dict, cap_gb: float | None) -> int:
    if state.get("batch"):
        return int(state["batch"])  # set after running out of memory
    if config.TRAIN_BATCH:
        return config.TRAIN_BATCH
    if cap_gb is None:
        return 4  # the CPU: memory is plenty, speed is the limit
    # VITS medium (70 M parameters with the discriminator): ~1.2 GB for weights,
    # gradients and AdamW state in fp32, and about 0.5 GB per clip of up to 15 s
    # at the peak. Measured on a GTX 1660 (6 GB, a 4.0 GB cap): batch 8 ran out
    # of memory, so this gives it batch 4.
    return max(2, min(32, int((cap_gb - 1.3) / 0.5) // 2 * 2))


def loaders() -> int:
    """Data loader processes hand batches over through /dev/shm. Docker's default
    of 64 MB is far too small for audio batches (they die with a bus error), so
    with little shared memory the data is loaded in the training process itself."""
    try:
        shm = shutil.disk_usage("/dev/shm").total
    except OSError:
        shm = 0
    if shm < 1024**3:
        log.info("Only %.0f MB of shared memory: loading data without helper processes", shm / 1024**2)
        return 0
    return config.TRAIN_LOADERS


def fit(project: dict, round_no: int, data: dict, accent: dict, start: dict) -> str:
    """Train this round. Returns why it ended: converged, limit or stopped."""
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from piper.train.vits.dataset import VitsDataModule
    from piper.train.vits.lightning import VitsModel

    pid = project["id"]
    here = round_dir(pid, round_no)
    here.mkdir(parents=True, exist_ok=True)
    state = project.get("training") or {}
    cuda = config.TRAIN_DEVICE == "cuda"
    config.WHISPER_DEVICE = "cpu"  # the likeness checks' speaker model stays off the GPU training is using
    torch.set_num_threads(config.THREADS)
    if cuda and not torch.cuda.is_available():
        raise Failed("Training is set to use the GPU (TRAIN_DEVICE=cuda) but the voice creator can’t see one. "
                     "Start it with docker-compose.voice.yml, which gives it the GPU.")

    cap_gb = None
    if cuda:
        free, total = gpu.wait_for_memory(config.TRAIN_MIN_FREE_GB,
                                          on_wait=lambda text: status(pid, text, stage="waiting"))
        # A hard cap below what's free: PyTorch then fails with an out-of-memory
        # error it can recover from, instead of pushing the driver over the edge.
        # The CUDA context (~0.4 GB) sits outside PyTorch's allocator, so it comes off too.
        cap_gb = free - config.GPU_RESERVE_GB - CUDA_CONTEXT_GB
        torch.cuda.set_per_process_memory_fraction(min(0.95, cap_gb / total), 0)
        log.info("Video memory: %.1f of %.1f GB free; training capped at %.1f GB", free, total, cap_gb)
    batch = pick_batch(state, cap_gb)

    clips = data["clips"]
    val_share = max(config.TRAIN_VAL_SHARE, 2 / clips)
    batches_per_epoch = max(1, (clips - int(clips * val_share)) // batch)
    # Small sets: validate at least once per pass over the data.
    val_every = max(1, min(config.TRAIN_VAL_EVERY, max(batches_per_epoch, 1)))
    max_batches = config.TRAIN_MAX_BATCHES
    min_batches = min(config.TRAIN_MIN_BATCHES, max_batches)
    status(pid, "Training: getting the clips ready", stage="training", batch=batch, device=config.TRAIN_DEVICE,
           max_batches=max_batches, round=round_no)

    model = VitsModel(batch_size=batch, sample_rate=config.SAMPLE_RATE, num_symbols=256, num_speakers=1)
    data_module = VitsDataModule(
        csv_path=data["csv"], audio_dir=data["wav"], cache_dir=str(Path(data["dir"]) / "cache"),
        config_path=str(Path(data["dir"]) / "config.json"), espeak_voice=accent["espeak"],
        voice_name=project["name"] or "voice", sample_rate=config.SAMPLE_RATE, batch_size=batch,
        validation_split=val_share, num_test_examples=0, num_workers=loaders())
    # The first round is TRAIN_FIRST_MINUTES long; later ones ("Train further")
    # run until the voice stops getting closer to the narrator.
    time_limit = config.TRAIN_FIRST_MINUTES * 60 if round_no == 1 and config.TRAIN_FIRST_MINUTES > 0 else None
    guide = make_guide(pid, here, max_batches, min_batches, val_every, Path(data["dir"]) / "config.json", time_limit)
    # last.ckpt for resuming; best.ckpt is the closest-sounding one, kept by the guide.
    checkpoints = ModelCheckpoint(dirpath=str(here), save_top_k=0, save_last=True, enable_version_counter=False)
    trainer = L.Trainer(
        accelerator="gpu" if cuda else "cpu", devices=1, precision=config.TRAIN_PRECISION,
        max_epochs=-1, max_steps=-1, val_check_interval=val_every, check_val_every_n_epoch=None,
        num_sanity_val_steps=0, logger=False, enable_progress_bar=False, enable_model_summary=False,
        callbacks=[guide, checkpoints], default_root_dir=str(here))
    # The same validation clips on every restart (piper splits with torch's random generator).
    L.seed_everything(1234, workers=True)
    resume = start_checkpoint(pid, round_no, start)
    log.info("Round %d: %d clips, batch %d, validating every %d batches, from %s",
             round_no, clips, batch, val_every, resume)
    try:
        trainer.fit(model, datamodule=data_module, ckpt_path=resume)
    except torch.cuda.OutOfMemoryError:
        if batch <= 2:
            raise Failed("The GPU ran out of video memory even with the smallest batch. "
                         "Close other programs that use the graphics card, then try again.")
        save(pid, batch=batch // 2)
        log.warning("Out of video memory with batch %d; trying again with %d", batch, batch // 2)
        gpu.keep_place()
        sys.exit(RETRY)
    if guide.reason == "unstable":
        raise Failed("Training became unstable (the loss stopped being a number). Start it again; if it "
                     "happens twice, set TRAIN_PRECISION=32-true and a smaller TRAIN_BATCH.")
    if guide.reason != "paused":
        # Where it ended, which validation-time saves may be behind: a round
        # without likeness checks (a short one) exports this, and the next round starts here.
        trainer.save_checkpoint(str(here / "last.ckpt"))
    return guide.reason


def export(project_id: str, round_no: int, data: dict) -> Path:
    """The best checkpoint as an ONNX model, with piper's own exporter."""
    here = round_dir(project_id, round_no)
    ckpt = here / "best.ckpt" if (here / "best.ckpt").exists() else here / "last.ckpt"
    if not ckpt.exists():
        raise Failed("Training stopped before it saved anything. Start it again and let it run a little longer.")
    out = here / "voice.onnx"
    run = subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ckpt),
                          "--output-file", str(out)], capture_output=True, text=True)
    if run.returncode != 0 or not out.exists():
        raise Failed(f"Exporting the voice failed: {run.stderr.strip().splitlines()[-1] if run.stderr.strip() else 'no output'}")
    return out


def main(project_id: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Stopping the container: unwind normally, so the GPU is let go at once
    # rather than when the claim goes stale. Training resumes from its last checkpoint.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    project = db.get_project(project_id)
    if project is None:
        return 0
    state = project.get("training") or {}
    round_no = int(state.get("round") or 1)
    accent = config.BASES.get(project["accent"]) or config.BASES[config.DEFAULT_ACCENT]
    here = round_dir(project_id, round_no)
    stop_file = here / "stop"
    try:
        if not (project["name"] or "").strip():
            raise Failed("Name the voice before training it.")
        # Each round's clips are cut once; a restart reuses them.
        data = state.get("data") if state.get("data_round") == round_no else None
        if not data or not Path(data["csv"]).exists():
            teacher = teacher_of(project_id, round_no, state["teacher"]) if state.get("teacher") else None
            status(project_id, "The cloned voice is getting ready to read for Piper" if teacher
                   else "Cutting the clips for training", progress=0.0, stage="dataset", round=round_no)
            data = dataset.build(project_id, round_no, teacher, report=lambda done, total: status(
                project_id, f"The cloned voice is reading sentences for Piper to learn from: {done:,} of {total:,}",
                progress=done / total, stage="dataset"))
            save(project_id, data=data, data_round=round_no,
                 trained_upto=max(int(state.get("trained_upto") or 0), data["max_clip"]))

        # Picked once, before the first step of the first round (CPU, a minute or two).
        start = (pick_start(db.get_project(project_id), accent)
                 if round_no == 1 and not (here / "last.ckpt").exists() else config.start_of(project))
        if not (here / "trained.json").exists():
            purpose = f"Training the voice “{project['name']}”"
            with gpu.Lease(purpose, on_wait=lambda text: status(project_id, text, stage="waiting"),
                           should_stop=stop_file.exists, gpu=config.TRAIN_DEVICE == "cuda", first=True):
                reason = fit(db.get_project(project_id), round_no, data, accent, start)
            if reason == "paused":
                # Forge goes first. This job runs again (keeping its place in the queue)
                # and carries on from the step it saved once Forge is done.
                status(project_id, f"{gpu.gpushare.WAITING_PREFIX}: paused while Forge voices a book. "
                                   "Training carries on from here when it’s done.", stage="waiting")
                gpu.keep_place()
                sys.exit(RETRY)
            here.mkdir(parents=True, exist_ok=True)
            (here / "trained.json").write_text(json.dumps({"reason": reason, "at": time.time()}))
            save(project_id, ended=reason)

        status(project_id, "Exporting the voice", stage="exporting")
        onnx = export(project_id, round_no, data)
        piper_config = json.loads((Path(data["dir"]) / "config.json").read_text(encoding="utf-8"))
        # Piper finds a voice's config beside it, as <model>.onnx.json.
        onnx.with_name(onnx.name + ".json").write_text(json.dumps(piper_config, indent=2), encoding="utf-8")
        # Nothing goes into Forge by itself: you listen to the checks and pick one
        # (Use this one, lab/main.py). The voice already there, if any, stays.
        db.update_project(project_id, status="done", progress=1.0,
                          message="Trained. Listen to the checks and pick one to put it into Forge.")
        save(project_id, stage="done")
        # Measuring it takes ~10 minutes on the CPU: a lane of its own, so the next
        # voice can train on the GPU meanwhile.
        db.add_job(project_id, "quality")
        return 0
    except gpu.Stopped:
        status(project_id, "Training was stopped before it started.", stage="stopped")
        db.update_project(project_id, status="ready")
        return 0
    except (Failed, gpu.NotEnoughMemory, ValueError, CloneError) as exc:
        log.warning("Training %s: %s", project_id, exc)
        db.update_project(project_id, status="failed", message=str(exc))
        save(project_id, stage="failed")
        return 1
    finally:
        stop_file.unlink(missing_ok=True)


def quality_main(project_id: str) -> int:
    """Measure the round's finished voice against the narrator (CPU only): the
    report on the voice creator's page. Forge's list shows the match of the
    version you picked, measured at its check."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    config.WHISPER_DEVICE = "cpu"  # a few dozen sentences; the GPU stays free for training
    project = db.get_project(project_id)
    if project is None:
        return 0
    round_no = int((project.get("training") or {}).get("round") or 1)
    onnx = round_dir(project_id, round_no) / "voice.onnx"
    if not onnx.exists():
        return 0

    def tell(text: str) -> None:
        if db.get_project(project_id)["status"] == "done":  # not if it's training further meanwhile
            db.update_project(project_id, message=text)

    try:
        report = quality.check(project, onnx, round_no, report=tell)
    except Exception as exc:  # the voice is fine without its report
        log.exception("Quality check failed")
        report = {"skipped": True, "text": f"The quality check couldn’t run: {exc}"}
    message = "Trained. Listen to the checks and pick one to put it into Forge."
    if report.get("likeness") is not None:
        message = (f"Trained: sounds like the narrator {round(report['likeness'] * 100)}%"
                   f" (their own recordings score {round(report['real_likeness'] * 100)}%)."
                   " Listen to the checks and pick one to put it into Forge.")
    fields = {"quality": report}
    if db.get_project(project_id)["status"] == "done":
        fields["message"] = message
    db.update_project(project_id, **fields)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--quality":
        sys.exit(quality_main(sys.argv[2]))
    sys.exit(main(sys.argv[1]))
