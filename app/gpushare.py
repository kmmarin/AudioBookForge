"""Sharing one GPU between Forge and the voice creator.

Only one heavy GPU user at a time: running out of video memory on a shared
Windows GPU can take the whole machine down. Each side writes its own small
state file in the shared models volume and rewrites it as a heartbeat. Files
are replaced atomically, so a reader never sees half of one, and a file that
hasn't been refreshed for a couple of minutes counts as gone, so a side that
crashed never blocks the other for long.

  forge.json     Forge's worker: gpu, busy, pool (up or down), yielded, wants
  voicelab.json  the voice creator: state (idle, waiting or holding), purpose

Forge goes first: voicing a book (or a voice sample) interrupts training,
which carries on from where it was once Forge is done.

The handshake:
  1. The voice creator asks for the GPU (waiting).
  2. While Forge has books or samples to voice it carries on (starting new
     ones too); the voice creator waits. Once Forge has nothing left, it shuts
     its voice process pool down, so its video memory is really freed, and
     says so (yielded).
  3. Only then does the voice creator take the GPU (holding), after checking
     with nvidia-smi that enough video memory is free.
  4. If a book or sample arrives while it holds the GPU, Forge says it wants
     the GPU (wants). The voice creator saves its place, lets go and asks
     again (back to 1): Forge's pool starts only once it has let go.
  5. When it lets go (idle, or its heartbeat goes stale), Forge starts its
     pool again and carries on with the books that waited.

Standard library only: the voice creator imports this file too.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

STALE_SECONDS = 120  # a heartbeat older than this means that side is gone
BEAT_SECONDS = 5  # how often each side rewrites its file while nothing changes

IDLE, WAITING, HOLDING = "idle", "waiting", "holding"
WAITING_PREFIX = "Waiting for the GPU"
QUEUED_HOLDING = f"{WAITING_PREFIX}: voice training is pausing so this goes first"
SAMPLE_BUSY = "The GPU is busy training a voice. Try the sample again when it’s done."


def read(path: Path) -> dict | None:
    """A side's state, or None if there isn't one (or it's unreadable)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write(path: Path, data: dict) -> None:
    """Replace the file in one step: written next to it, then renamed over it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


class Share:
    """The two state files, read with staleness in mind."""

    def __init__(self, folder: Path | str, stale: float = STALE_SECONDS, clock=time.time):
        self.folder = Path(folder)
        self.forge_path = self.folder / "forge.json"
        self.lab_path = self.folder / "voicelab.json"
        self.stale = stale
        self.clock = clock

    def _fresh(self, state: dict | None) -> dict | None:
        if not state:
            return None
        try:
            age = self.clock() - float(state.get("heartbeat", 0))
        except (TypeError, ValueError):
            return None
        return state if -self.stale < age < self.stale else None

    def lab(self) -> dict:
        """The voice creator's state; idle when it has none or went quiet."""
        state = self._fresh(read(self.lab_path))
        if not state or state.get("state") not in (WAITING, HOLDING):
            return {"state": IDLE}
        return state

    def forge(self) -> dict | None:
        """Forge's state, or None when Forge isn't running (or went quiet)."""
        return self._fresh(read(self.forge_path))


# --------------------------------------------------------------------------
# Forge's side
# --------------------------------------------------------------------------

def forge_plan(lab_state: str, busy: bool, in_flight: int, pool_up: bool, wants: bool = False) -> dict:
    """What Forge's worker should do now.

    busy: books are being voiced or finished; in_flight: parts and samples in
    the pool right now; wants: books or samples waiting to be voiced. Returns
    whether it may start new books, send more parts to the pool, and what to do
    with the pool: keep, start or stop.
    """
    if lab_state == IDLE:
        return {"start_books": True, "submit": True, "pool": "keep" if pool_up else "start"}
    if lab_state == WAITING:
        # Forge goes first: training (re)starts only once Forge has nothing to voice.
        if busy or in_flight or wants:
            return {"start_books": True, "submit": True, "pool": "keep" if pool_up else "start"}
        return {"start_books": False, "submit": False, "pool": "stop" if pool_up else "keep"}
    # The voice creator has the GPU. Nothing starts until it lets go, which it
    # does when Forge reports work waiting (wants); finish what's in flight.
    if in_flight:
        return {"start_books": False, "submit": False, "pool": "keep"}
    return {"start_books": False, "submit": False, "pool": "stop" if pool_up else "keep"}


def queued_message(lab_state: str) -> str | None:
    return QUEUED_HOLDING if lab_state == HOLDING else None


class ForgeSide:
    """Forge's worker: reads what the voice creator wants, reports what Forge is doing.

    With uses_gpu False (Forge voices on the CPU) it never steps aside, but
    still reports, so the voice creator knows it needn't wait."""

    def __init__(self, folder: Path | str, uses_gpu: bool, stale: float = STALE_SECONDS,
                 clock=time.time, beat: float = BEAT_SECONDS):
        self.share = Share(folder, stale, clock)
        self.uses_gpu = uses_gpu
        self.clock = clock
        self.beat = beat
        self._last: tuple | None = None
        self._last_write = 0.0
        self._lab_cache: tuple[float, str, str] | None = None  # (read at, state, purpose)

    def lab_state(self) -> str:
        if not self.uses_gpu:
            return IDLE
        now = self.clock()
        if self._lab_cache is None or now - self._lab_cache[0] >= 1.0:  # the worker loops 4x a second
            lab = self.share.lab()
            self._lab_cache = (now, lab["state"], str(lab.get("purpose") or ""))
        return self._lab_cache[1]

    def report(self, busy: bool, pool_up: bool, lab: str | None = None, wants: bool = False) -> None:
        """Write forge.json when something changed, else every few seconds as a
        heartbeat. lab: the voice creator's state the worker just acted on;
        wants: books or samples are waiting for the GPU (training pauses for them)."""
        lab = lab or self.lab_state()
        state = (self.uses_gpu, bool(busy), bool(pool_up), lab, bool(wants))
        now = self.clock()
        if state == self._last and now - self._last_write < self.beat:
            return
        try:
            write(self.share.forge_path, {
                "heartbeat": now, "pid": os.getpid(), "gpu": self.uses_gpu, "busy": bool(busy),
                "pool": "up" if pool_up else "down",
                # Only true while the voice creator asks: it may take the GPU now.
                "yielded": self.uses_gpu and not pool_up and lab != IDLE and not wants,
                "wants": self.uses_gpu and bool(wants)})
        except OSError:
            return  # a full or read-only volume mustn't stop the worker; try again next time
        self._last, self._last_write = state, now


# --------------------------------------------------------------------------
# The voice creator's side
# --------------------------------------------------------------------------

class LabSide:
    """The voice creator: asks for the GPU, holds it, lets it go."""

    def __init__(self, folder: Path | str, purpose: str = "", stale: float = STALE_SECONDS,
                 clock=time.time):
        self.share = Share(folder, stale, clock)
        self.purpose = purpose
        self.clock = clock
        self.state = IDLE
        self.since = 0.0

    def _write(self) -> None:
        write(self.share.lab_path, {"heartbeat": self.clock(), "pid": os.getpid(),
                                    "state": self.state, "purpose": self.purpose,
                                    "since": self.since})

    def request(self) -> None:
        if self.state == IDLE:
            self.state, self.since = WAITING, self.clock()
        self._write()

    def can_take(self) -> tuple[bool, str]:
        """Whether Forge has stepped aside; if not, a status line saying why not."""
        forge = self.share.forge()
        if forge is None or not forge.get("gpu"):
            return True, ""  # Forge isn't running, or voices on the CPU
        if forge.get("yielded"):
            return True, ""
        if forge.get("busy") or forge.get("wants"):
            return False, f"{WAITING_PREFIX}: Forge is voicing a book. Training carries on when it’s done."
        return False, f"{WAITING_PREFIX}: waiting for Forge to free it."

    def forge_wants(self) -> bool:
        """Forge has books or samples waiting: it goes first, so whoever holds
        the GPU saves its place and lets go."""
        forge = self.share.forge()
        return bool(forge and forge.get("gpu") and forge.get("wants"))

    def hold(self) -> None:
        self.state, self.since = HOLDING, self.clock()
        self._write()

    def heartbeat(self) -> None:
        if self.state != IDLE:
            self._write()

    def release(self) -> None:
        self.state = IDLE
        try:
            self._write()
        except OSError:
            pass  # it goes stale on its own
