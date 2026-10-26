"""Taking turns on the GPU with Forge, and checking video memory before using it.

The rules live in Forge's app/gpushare.py, shared by both sides."""
from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
import threading
import time

from app import gpushare

from . import config

log = logging.getLogger("voicelab.gpu")


class Stopped(Exception):
    """Someone pressed Stop while this was waiting for the GPU."""


class NotEnoughMemory(RuntimeError):
    pass


def memory_gb() -> tuple[float, float] | None:
    """(free, total) video memory of the first GPU from nvidia-smi, or None."""
    # Under WSL the Windows driver's nvidia-smi lives here, outside PATH.
    smi = shutil.which("nvidia-smi") or next(
        (p for p in ("/usr/lib/wsl/lib/nvidia-smi",) if os.path.exists(p)), None)
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        free, total = (float(x) / 1024 for x in out.splitlines()[0].split(","))
        return free, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


TRAINING_WAITS_FRESH = 60  # seconds: a waiting training refreshes its marker every 2 s


def forge_wants() -> bool:
    """Forge has a book or voice sample to voice: it goes first, so training and
    listening save their place and let go of the GPU."""
    if not config.GPU_SHARED:
        return False
    return gpushare.LabSide(config.GPU_SHARE_DIR, stale=config.GPU_SHARE_STALE).forge_wants()


def keep_place() -> None:
    """A training restarting itself (a smaller batch after running out of video
    memory) keeps its place: recordings waiting to be heard hold back while the
    new process starts up, instead of taking the GPU in that gap."""
    config.GPU_SHARE_DIR.mkdir(parents=True, exist_ok=True)
    (config.GPU_SHARE_DIR / "voicelab.training-waits").write_text(str(os.getpid()))


class Lease:
    """Hold the GPU for a with-block: ask, wait for Forge to step aside, hold,
    and let go at the end, even after an error. A heartbeat thread keeps the
    claim fresh; if this process dies, the claim goes stale and Forge resumes.

    The voice creator's own jobs take turns as well (one voice trains while
    another's recordings are heard): both at once wouldn't fit on a 6 GB card,
    and they share the one claim file Forge reads, so one letting go would tell
    Forge the GPU is free while the other still uses it.

    on_wait(text) is called with a status line while waiting; should_stop()
    ends the wait with Stopped."""

    def __init__(self, purpose: str, on_wait=lambda text: None, should_stop=lambda: False,
                 shared: bool | None = None, gpu: bool = True, first: bool = False):
        self.shared = config.GPU_SHARED if shared is None else shared
        self.gpu = gpu or self.shared
        self.first = first  # training: goes before any recordings waiting to be heard
        self.side = gpushare.LabSide(config.GPU_SHARE_DIR, purpose, stale=config.GPU_SHARE_STALE)
        self.on_wait = on_wait
        self.should_stop = should_stop
        self._done = threading.Event()
        self._beat: threading.Thread | None = None
        self._turn = None  # the open lock file while this job has the voice creator's turn

    def _take_turn(self) -> None:
        """Wait until no other job of the voice creator is using the GPU. A
        training waiting for its turn goes before recordings waiting to be
        heard, so finished voices come sooner."""
        config.GPU_SHARE_DIR.mkdir(parents=True, exist_ok=True)
        turn = open(config.GPU_SHARE_DIR / "voicelab.lock", "w")
        waiting = config.GPU_SHARE_DIR / "voicelab.training-waits"
        said = None
        try:
            while True:
                if self.first:
                    waiting.write_text(str(os.getpid()))  # fresh while it waits; stale if it dies
                elif waiting.exists() and time.time() - waiting.stat().st_mtime < TRAINING_WAITS_FRESH:
                    if said != "training":
                        self.on_wait(f"{gpushare.WAITING_PREFIX}: a voice is training first. This one goes next.")
                        said = "training"
                    if self.should_stop():
                        raise Stopped()
                    time.sleep(2)
                    continue
                try:
                    fcntl.flock(turn, fcntl.LOCK_EX | fcntl.LOCK_NB)  # let go by the system if this process dies
                    break
                except BlockingIOError:
                    if said != "busy":
                        self.on_wait(f"{gpushare.WAITING_PREFIX}: another voice is using it. This one goes next.")
                        said = "busy"
                    if self.should_stop():
                        raise Stopped()
                    time.sleep(2)
        except BaseException:
            turn.close()
            raise
        finally:
            if self.first:
                waiting.unlink(missing_ok=True)
        self._turn = turn

    def _heartbeat(self) -> None:
        while not self._done.wait(min(10.0, config.GPU_SHARE_STALE / 6)):
            try:
                self.side.heartbeat()
            except OSError:
                log.warning("Couldn't refresh the GPU claim", exc_info=True)

    def __enter__(self) -> "Lease":
        if self.gpu:
            self._take_turn()
        if not self.shared:
            return self
        self.side.request()
        self._beat = threading.Thread(target=self._heartbeat, daemon=True)
        self._beat.start()
        try:
            last = ""
            while True:
                ok, why = self.side.can_take()
                if ok:
                    break
                if why != last:
                    log.info(why)
                    self.on_wait(why)
                    last = why
                if self.should_stop():
                    raise Stopped()
                time.sleep(2)
            self.side.hold()
            log.info("Holding the GPU: %s", self.side.purpose)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._done.set()
        if self.shared:
            self.side.release()
            log.info("Let go of the GPU")
        if self._turn is not None:
            self._turn.close()  # closing the file ends the lock: the next job goes
            self._turn = None


def wait_for_memory(need_gb: float, on_wait=lambda text: None, timeout: float = 90) -> tuple[float, float]:
    """Free and total video memory once at least need_gb is free. Forge's
    processes take a few seconds to give theirs back after stepping aside;
    if it isn't free by the timeout, say so instead of trying anyway."""
    deadline = time.monotonic() + timeout
    while True:
        mem = memory_gb()
        if mem is None:
            raise NotEnoughMemory("nvidia-smi can’t see a GPU in the voice creator’s container. Start it "
                                  "with docker-compose.voice.yml, which gives it the GPU, or set TRAIN_DEVICE=cpu.")
        free, total = mem
        if free >= need_gb:
            return free, total
        if time.monotonic() > deadline:
            raise NotEnoughMemory(
                f"Only {free:.1f} GB of video memory is free and training needs {need_gb:.1f} GB. "
                "Close programs that use the graphics card (games, video editors, other AI tools), "
                "then start training again.")
        on_wait(f"{gpushare.WAITING_PREFIX}: waiting for video memory to be freed "
                f"({free:.1f} of {need_gb:.1f} GB free)")
        time.sleep(3)
