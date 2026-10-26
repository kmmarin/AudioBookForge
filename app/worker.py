"""Background worker.

Every queued book is cut into small work units (a few paragraphs each) that
go to one shared process pool, longest first. Chapters are voiced in
parallel, and long chapters are split across several workers too, so a
single 40-page chapter can’t hold up the end of the book. Up to
BOOK_CONCURRENCY books share the pool, taking turns, so workers stay busy
while one book is being assembled.

Only a few units are in the pool at a time. That keeps books taking turns
and lets voice samples from the web app go next, on the models the workers
already have loaded, instead of waiting behind a whole book.

Finished units are cached on disk under a content hash. A restart, a
cancelled job, or a text edit only re-voices what actually changed.

Books and samples in a cloned voice go to the clone server instead (app/clone.py,
on whichever machine has the GPU for it), through a few threads of their own:
they never wait for this machine's GPU, nor take it from the voice creator.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from . import (assemble, config, covers, db, gpus, gpushare, previews, publish, qa, readalong, scenes, textprep,
               transcribe, tts, versions, voices)

log = logging.getLogger("forge.worker")
_stop = threading.Event()
GPU_RETRIES = 3  # times a part that ran out of video memory is tried again


def is_clone(voice: dict | None) -> bool:
    """A cloned voice: the clone server voices it, not this machine's GPU."""
    return bool(voice) and voice.get("engine") == "clone"


def _book_is_clone(book: dict) -> bool:
    return book["kind"] != "audio" and is_clone(voices.get(book["voice"]))


class ActiveBook:
    def __init__(self, book_id: str, voice: dict, speed: float):
        self.id = book_id
        self.voice = voice
        self.speed = speed
        self.clone = is_clone(voice)  # voiced by the clone server, not the pool
        self.queue: list[tuple[int, dict]] = []  # (chapter idx, job) not yet sent to the pool
        self.futures: dict[Future, tuple[int, dict]] = {}  # unit in the pool -> (chapter idx, job)
        self.retries: dict[str, int] = {}  # out path -> out-of-memory retries so far
        self.unit_paths: dict[int, list[Path]] = {}
        self.unit_items: dict[int, list[list]] = {}
        self.unit_sources: dict[int, list[list]] = {}  # paragraph behind each "say" (Read along)
        self.display: dict[int, list[str]] = {}  # paragraphs as printed (Read along)
        self.chapters: list[dict] = []
        self.built = ""  # the settings this build uses (db.built_with)
        self.done = 0
        self.total = 0
        self.ch_done: dict[int, int] = {}
        self.error: BaseException | None = None
        self.finishing: Future | None = None
        self.dirty = True
        # Real-time speed: seconds of audio voiced per second of wall time, this run only.
        self.started = time.monotonic()
        self.audio_seconds = 0.0


def new_pool() -> ProcessPoolExecutor:
    # Spawned children inherit this before numpy and torch start their thread
    # pools; setting it in the initializer is too late for numpy's BLAS.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(config.TORCH_THREADS)
    return ProcessPoolExecutor(
        max_workers=config.TTS_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=tts.init_worker,
        initargs=(config.TORCH_THREADS, config.DEVICE,
                  config.GPU_MEM_PER_WORKER_GB if config.DEVICE == "cuda" else 0.0),
    )


def start_pool(default_voice: dict | None) -> tuple[ProcessPoolExecutor, list[Future]]:
    """The pool, and its warm-up jobs (samples sent now wait behind them)."""
    pool = new_pool()
    warming = []
    if default_voice:
        # Start every worker process now and load the default voice, so the first
        # sample after a restart takes a second or two instead of half a minute.
        for _ in range(config.TTS_WORKERS):
            warming.append(pool.submit(tts.warm, voices.with_mood(default_voice, None)))
    return pool, warming


def note_queued(message: str | None) -> None:
    """Say on queued books why they wait while the voice creator has the GPU,
    or (message None) take that note off again."""
    for book in db.books_by_status("queued"):
        if _book_is_clone(book):
            continue  # it doesn't wait for the GPU here
        current = book["message"] or ""
        if message and current != message:
            db.update_book(book["id"], message=message)
        elif not message and current.startswith(gpushare.WAITING_PREFIX):
            db.update_book(book["id"], message="Waiting for a worker")


def gpu_work_waiting(active: dict, gpu_whisper: bool) -> bool:
    """Voice samples or books waiting to be voiced (and audiobooks to transcribe,
    when Whisper runs on the GPU). Cloned voices don't count: the clone server
    voices them, so they needn't pause the voice creator here."""
    if db.samples_waiting(clone=False):
        return True
    return any(b["id"] not in active and (gpu_whisper if b["kind"] == "audio" else not _book_is_clone(b))
               for b in db.books_by_status("queued"))


PREMAKE_AT_ONCE = 4  # saved samples queued at a time, so a real request never waits long behind them
PREMAKE_RETRY = 1800  # seconds before a saved sample that failed is tried again


def premake_samples() -> int:
    """Queue the saved sample (previews.py) of voices that haven't one yet, a few
    at a time: then picking a voice plays it at once. Returns how many it queued."""
    queued = 0
    for job in previews.saved_jobs():
        if Path(job["out"]).exists():
            continue
        row = db.get_sample(db.sample_key(job["out"]))
        if row and (row["status"] in ("pending", "working")
                    or row["status"] == "failed" and time.time() - row["created"] < PREMAKE_RETRY):
            continue
        Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
        db.add_sample(job)
        queued += 1
        if queued >= PREMAKE_AT_ONCE:
            break
    return queued


def chapter_text(book_id: str, idx: int) -> str:
    return (config.BOOKS_DIR / book_id / "text" / f"{idx:04d}.txt").read_text(encoding="utf-8")


def start_book(book: dict) -> ActiveBook:
    voice = voices.get(book["voice"]) or voices.get(voices.default_id())
    if voice is None:
        raise ValueError("No voices are configured. Check voices.yaml.")
    voice = voices.with_mood(voice, book["mood"])
    speed = float(book["speed"] or 1.0)
    chapters = [c for c in db.chapters(book["id"]) if c["include"]]
    if not chapters:
        raise ValueError("No chapters are turned on. Turn on at least one chapter.")

    active = ActiveBook(book["id"], voice, speed)
    active.chapters = chapters
    active.built = db.built_with(book)
    unit_dir = config.BOOKS_DIR / book["id"] / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ch in chapters:
        idx = ch["idx"]
        units, sources, display = textprep.plan_units(ch["title"], chapter_text(book["id"], idx))
        active.unit_sources[idx] = sources
        active.display[idx] = display
        paths = []
        active.ch_done[idx] = 0
        for n, items in enumerate(units):
            key = textprep.unit_key(voice["key"], speed, items)
            path = unit_dir / f"{idx:04d}-{n:04d}-{key}.wav"
            paths.append(path)
            # A unit voiced before word timings were recorded is voiced again,
            # so Read along gets exact timings rather than estimates.
            if path.exists() and Path(tts.timing_path(path)).exists():
                active.ch_done[idx] += 1
                active.done += 1
            else:
                size = len(textprep.unit_text(items))
                jobs.append((size, idx, {"voice": voice, "speed": speed,
                                         "items": items, "out": str(path)}))
        active.unit_paths[idx] = paths
        active.unit_items[idx] = units
        active.total += len(units)

    db.update_chapters(book["id"], [
        {"idx": c["idx"], "units_total": len(active.unit_paths[c["idx"]]),
         "units_done": active.ch_done[c["idx"]], "state": "pending", "wer": None}
        for c in chapters])

    # Longest units first: big jobs start early and small ones fill the gaps at the end.
    jobs.sort(key=lambda job: -job[0])
    active.queue = [(idx, job) for _, idx, job in jobs]

    where = f"{config.TTS_WORKERS} workers"
    if active.clone:
        server = gpus.pick("voices")
        where = f"the GPU server at {server['url'].split('://', 1)[-1]}" if server else "a GPU server"
    db.update_book(book["id"], status="voicing", progress=_progress(active), output=None, rtf=None,
                   message=f"Voicing {len(chapters)} chapters on {where}")
    log.info("Started %s: %d units (%d cached)", book["id"], active.total, active.done)
    return active


def _progress(active: ActiveBook) -> float:
    return round(0.9 * active.done / max(1, active.total), 4)


def flush_progress(active: ActiveBook) -> None:
    if not active.dirty:
        return
    active.dirty = False
    rows = []
    for ch in active.chapters:
        idx = ch["idx"]
        total = len(active.unit_paths[idx])
        done = active.ch_done[idx]
        state = "done" if done >= total else ("voicing" if done else "pending")
        rows.append({"idx": idx, "units_done": done, "state": state})
    db.update_chapters(active.id, rows)
    fields = {}
    elapsed = time.monotonic() - active.started
    if active.audio_seconds and elapsed > 2:
        fields["rtf"] = round(active.audio_seconds / elapsed, 1)
    db.update_book(active.id, progress=_progress(active),
                   message=f"Voiced {active.done} of {active.total} parts", **fields)


def run_qa(active: ActiveBook, book: dict) -> dict[int, float]:
    language = voices.language(active.voice)
    scores: dict[int, float] = {}
    with qa.session():
        for ch in active.chapters:
            idx = ch["idx"]
            units = active.unit_items[idx]
            worst = 0.0
            for n in qa.sample_indexes(len(units), config.QA_SAMPLE):
                worst = max(worst, qa.score_unit(active.unit_paths[idx][n],
                                                 textprep.unit_text(units[n]), language))
            scores[idx] = round(worst, 3)
    return scores


def finish_book(active: ActiveBook) -> None:
    book_id = active.id
    try:
        book = db.get_book(book_id)
        if book is None:
            return
        workdir = config.BOOKS_DIR / book_id
        meta = book["meta"]
        cover = workdir / "cover.jpg"
        if not cover.exists():
            cover = workdir / "cover-auto.jpg"
            cover.write_bytes(covers.generate(meta.get("title", ""), meta.get("author", "")))

        qa_scores: dict[int, float] = {}
        qa_error: list[str] = []
        qa_thread = None
        if book["qa"]:
            def qa_job():
                try:
                    qa_scores.update(run_qa(active, book))
                except Exception as exc:  # the audiobook is still usable without the check
                    log.exception("Narration check failed for %s", book_id)
                    qa_error.append(str(exc))
            qa_thread = threading.Thread(target=qa_job, daemon=True)
            qa_thread.start()

        db.update_book(book_id, status="assembling", progress=0.92,
                       message="Building the audiobook file")
        out, durations = assemble.build(book, active.chapters, active.unit_paths, workdir,
                                        voices.narrator(active.voice), cover)
        try:
            readalong.build(book_id, active.chapters, active.unit_paths, active.unit_items,
                            active.unit_sources, active.display, durations)
        except Exception:  # the audiobook is fine without it
            log.exception("Read along map failed for %s", book_id)
            readalong.path_for(book_id).unlink(missing_ok=True)

        if qa_thread:
            db.update_book(book_id, status="checking", progress=0.97,
                           message="Checking narration with Whisper")
            qa_thread.join()

        db.update_chapters(book_id, [
            {"idx": c["idx"], "duration": round(d, 2), "state": "done",
             "wer": qa_scores.get(c["idx"])}
            for c, d in zip(active.chapters, durations)])

        flagged = [s for s in qa_scores.values() if s >= config.QA_FLAG_WER]
        if qa_error:
            message = "Done. The narration check couldn’t run."
        elif flagged:
            message = (f"Done. {len(flagged)} chapter{'s' if len(flagged) > 1 else ''} "
                       "may have misread text.")
        else:
            message = "Done"
        db.update_book(book_id, status="done", progress=1.0, output=str(out), message=message,
                       built=active.built, edited=0)
        try:
            versions.keep(book_id)  # so switching back to this voice later is instant
        except Exception:
            log.exception("Couldn’t keep a copy of %s in this voice", book_id)
        _remove_stale_units(workdir / "units", active)
        log.info("Finished %s -> %s", book_id, out)
        if config.PUBLISH_URL and config.PUBLISH_AUTO:
            send_to_server(book_id, message)
    except Exception as exc:
        log.exception("Assembly failed for %s", book_id)
        db.update_book(book_id, status="failed", message=f"Building the file failed: {exc}")


def send_to_server(book_id: str, message: str) -> None:
    """Publish a finished book so Read along works on the server from any device.
    The book is done either way; the status line says how sending went."""
    host = config.PUBLISH_URL.split("://", 1)[-1]
    try:
        publish.send(book_id)
        note = f"Sent to {host}."
    except publish.PublishError as exc:
        log.warning("Publishing %s failed: %s", book_id, exc)
        if exc.conflict:
            note = (f"Not sent: {host} has its own copy of this book "
                    f"({exc.conflict.get('voice')}). Use Send to server to replace it.")
        else:
            note = f"Sending to {host} failed: {exc} Use Send to server to try again."
    base = message if message.endswith(".") else f"{message}."
    db.update_book(book_id, message=f"{base} {note}")


def _remove_stale_units(unit_dir: Path, active: ActiveBook) -> None:
    keep = {p.stem for paths in active.unit_paths.values() for p in paths}
    for path in [*unit_dir.glob("*.wav"), *unit_dir.glob("*.json")]:
        if path.stem not in keep:
            path.unlink(missing_ok=True)


def top_up(executor, active: dict[str, "ActiveBook"], samples: dict, clone: bool, limit: int,
           stage: str = "voicing") -> None:
    """Keep an executor just full, books taking turns: the pool on this machine's
    GPU (one unit per worker plus one spare), or, with clone, the threads that
    send cloned voices to the clone server. Samples are sent as soon as they
    arrive, so they go next."""
    for sample_id, job in db.take_samples(stage, clone=clone):
        samples[executor.submit(tts.synth_unit, job)] = sample_id
    voicing = [a for a in active.values() if a.finishing is None and a.queue and a.clone == clone]
    in_flight = len(samples) + sum(len(a.futures) for a in active.values() if a.clone == clone)
    while voicing and in_flight < limit:
        for item in list(voicing):
            if in_flight >= limit:
                break
            idx, job = item.queue.pop(0)
            item.futures[executor.submit(tts.synth_unit, job)] = (idx, job)
            in_flight += 1
            if not item.queue:
                voicing.remove(item)


def fail(book_id: str, exc: BaseException) -> None:
    log.error("Book %s failed: %s", book_id, exc)
    db.update_book(book_id, status="failed", message=str(exc)[:500] or exc.__class__.__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init()
    for book in db.books_by_status("voicing", "assembling", "checking", "transcribing"):
        db.update_book(book["id"], status="queued", message="Resuming after restart")
    log.info("Worker ready: %d CPUs, %d TTS workers x %d threads, device %s, %d encoders",
             config.CPUS, config.TTS_WORKERS, config.TORCH_THREADS, config.DEVICE,
             config.ENCODE_WORKERS)
    if config.GPU_TOTAL_GB:
        log.info("Video memory: %.1f of %.1f GB free at start; %d workers x %.2f GB cap, "
                 "%.2f GB kept free for other programs",
                 config.GPU_FREE_GB, config.GPU_TOTAL_GB, config.TTS_WORKERS,
                 config.GPU_MEM_PER_WORKER_GB, config.GPU_RESERVE_GB)

    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    db.reset_samples()
    default_voice = voices.get(voices.default_id())
    # The voice creator may have the GPU (app/gpushare.py): then the pool, and
    # the video memory it takes, waits until it's done.
    share = gpushare.ForgeSide(config.GPU_SHARE_DIR, config.GPU_SHARED, stale=config.GPU_SHARE_STALE)
    pool: ProcessPoolExecutor | None = None
    warming: list[Future] = []  # the pool loading the default voice
    if share.lab_state() == gpushare.IDLE:
        pool, warming = start_pool(default_voice)
    else:
        log.info("The voice creator is using the GPU; voices start when it's done")
    finisher = ThreadPoolExecutor(max_workers=max(1, config.BOOK_CONCURRENCY))
    transcriber = transcribe.Runner(_stop, send_to_server)  # audiobooks added for Read along, beside the pool
    if config.PUBLISH_URL:  # your voices go to the server too, so they can be picked there
        threading.Thread(target=publish.voice_sync_loop, args=(_stop,), daemon=True, name="voice-sync").start()
    # Cloned voices: threads that wait on the clone server, beside the pool.
    clone_exec = ThreadPoolExecutor(max_workers=max(1, config.CLONE_WORKERS), thread_name_prefix="clone")
    # Watch and read's pictures: drawn by the clone server too, one at a time.
    threading.Thread(target=scenes.draw_loop, args=(_stop,), daemon=True, name="pictures").start()
    active: dict[str, ActiveBook] = {}
    samples: dict[Future, str] = {}  # sample in the pool -> sample id
    clone_samples: dict[Future, str] = {}  # sample at the clone server -> sample id
    last_flush = 0.0
    last_note = 0.0
    last_premake = 0.0

    # Transcribing added audiobooks is GPU work too when Whisper runs there.
    gpu_whisper = config.WHISPER_DEVICE == "cuda" or config.GPU_SHARE == "on"

    while not _stop.is_set():
        lab = share.lab_state()
        # Only what uses this machine's GPU counts for taking turns with the voice creator.
        in_flight = len(samples) + sum(len(a.futures) for a in active.values() if not a.clone)
        transcribing = gpu_whisper and transcriber.current is not None and not transcriber.current.done()
        busy = any(not a.clone for a in active.values()) or transcribing
        # Books or samples waiting while the voice creator has (or asks for) the
        # GPU: Forge goes first, and a voice in training pauses for them.
        wants = lab != gpushare.IDLE and gpu_work_waiting(active, gpu_whisper)
        plan = gpushare.forge_plan(lab, busy, in_flight, pool is not None, wants)
        if plan["pool"] == "start":
            log.info("The GPU is free again; starting the voices")
            pool, warming = start_pool(default_voice)
            note_queued(None)
        elif plan["pool"] == "stop" and pool is not None:
            log.info("Stepping aside: the voice creator needs the GPU")
            pool.shutdown(wait=True, cancel_futures=True)  # the processes exit: their video memory is freed
            pool, warming = None, []
        share.report(busy=busy, pool_up=pool is not None, lab=lab, wants=wants)
        # How far waiting samples are, for the bar under their play buttons.
        was_warming = bool(warming)
        warming = [f for f in warming if not f.done()]
        if was_warming and not warming and samples:
            db.note_samples("voicing", status="working")
        if db.samples_waiting(clone=False):
            if warming:
                db.note_samples("starting")
            elif pool is None and lab != gpushare.IDLE:
                db.note_samples("gpu")  # the voice creator is pausing so they go first
            else:
                db.note_samples("queued")
        if lab != gpushare.IDLE and time.monotonic() - last_note > 2.0:
            last_note = time.monotonic()
            note_queued(gpushare.queued_message(lab))

        # Drop finished books; stop books that were cancelled or removed.
        for book_id, item in list(active.items()):
            if item.finishing is not None:
                if item.finishing.done():
                    active.pop(book_id)
                continue
            if db.get_status(book_id) in (None, "cancelled"):
                item.queue.clear()
                for fut in item.futures:
                    fut.cancel()
                active.pop(book_id)
                log.info("Stopped %s", book_id)

        # Pull more books from the queue: those in a cloned voice whenever there's
        # room (the clone server voices them), the rest when the GPU is Forge's.
        voicing = sum(1 for a in active.values() if a.finishing is None)
        if voicing < config.BOOK_CONCURRENCY:
            gpu_ok = plan["start_books"] and pool is not None
            queued = [b for b in db.books_by_status("queued") if b["id"] not in active and b["kind"] != "audio"
                      and (gpu_ok or _book_is_clone(b))]
            for book in queued[: config.BOOK_CONCURRENCY - voicing]:
                try:
                    active[book["id"]] = start_book(book)
                except Exception as exc:
                    fail(book["id"], exc)

        top_up(clone_exec, active, clone_samples, clone=True, limit=max(1, config.CLONE_WORKERS))
        if plan["submit"] and pool is not None:
            top_up(pool, active, samples, clone=False, limit=config.TTS_WORKERS + 1,
                   stage="starting" if warming else "voicing")
        else:
            # Samples wait their turn like books: they count as work waiting, so
            # training pauses for them and the pool starts once it has let go.
            if lab == gpushare.HOLDING:
                # The voice creator has the GPU: books wait (finished parts are kept).
                for book_id, item in list(active.items()):
                    if item.finishing is None and not item.clone and not item.futures and item.queue:
                        flush_progress(item)
                        db.update_book(book_id, status="queued", message=gpushare.QUEUED_HOLDING)
                        active.pop(book_id)
        if plan["start_books"] or not gpu_whisper:
            transcriber.poll()  # Whisper on the GPU waits for its turn like voicing does

        # Wait for units and samples to finish. Short waits so new samples start quickly.
        pending = {fut: a.id for a in active.values() if a.finishing is None for fut in a.futures}
        if pending or samples or clone_samples:
            done, _ = wait([*pending, *samples, *clone_samples], timeout=0.25, return_when=FIRST_COMPLETED)
        else:
            _stop.wait(0.25)
            done = set()

        broken = False
        for waiting in (samples, clone_samples):
            for fut in [f for f in done if f in waiting]:
                sample_id = waiting.pop(fut)
                exc = None if fut.cancelled() else fut.exception()
                if isinstance(exc, BrokenProcessPool):
                    broken = True
                db.finish_sample(sample_id, f"{exc}"[:300] if exc else None)
        for fut in done:
            if fut not in pending:
                continue
            item = active.get(pending[fut])
            if item is None or fut not in item.futures:
                continue
            idx, job = item.futures.pop(fut)
            if fut.cancelled():
                continue
            exc = fut.exception()
            if isinstance(exc, BrokenProcessPool):
                broken = True
            elif isinstance(exc, tts.GpuMemoryError) and item.retries.get(job["out"], 0) < GPU_RETRIES:
                # The worker already dropped its model and cache; try the part again.
                item.retries[job["out"]] = item.retries.get(job["out"], 0) + 1
                log.warning("%s: part ran out of video memory, retry %d of %d",
                            item.id, item.retries[job["out"]], GPU_RETRIES)
                item.queue.insert(0, (idx, job))
            elif exc is not None:
                item.error = item.error or exc
            else:
                result = fut.result()
                item.audio_seconds += result["frames"] / result["rate"]
                item.done += 1
                item.ch_done[idx] += 1
                item.dirty = True

        if broken and pool is not None:
            log.error("A voice worker crashed; restarting the pool")
            pool.shutdown(wait=False, cancel_futures=True)
            pool, warming = new_pool(), []
            for sample_id in samples.values():
                db.finish_sample(sample_id, "A voice worker crashed. Try the sample again.")
            samples.clear()
            for book_id, item in list(active.items()):
                if item.finishing is None and not item.clone:  # cloned voices weren't in the pool
                    fail(book_id, RuntimeError(
                        "A voice worker crashed, usually from running out of memory. "
                        "Retry to carry on; finished parts are kept. If it happens again, "
                        "lower TTS_WORKERS."))
                    active.pop(book_id)

        for book_id, item in list(active.items()):
            if item.finishing is not None:
                continue
            if item.error is not None:
                item.queue.clear()
                for fut in item.futures:
                    fut.cancel()
                fail(book_id, item.error)
                active.pop(book_id)
            elif not item.futures and not item.queue:
                flush_progress(item)
                db.update_book(book_id, status="assembling", progress=0.9,
                               message="Building the audiobook file")
                item.finishing = finisher.submit(finish_book, item)

        if time.monotonic() - last_flush > 1.0:
            last_flush = time.monotonic()
            for item in active.values():
                if item.finishing is None:
                    flush_progress(item)

        # Idle: make the voices' saved samples that are missing (new voices, a changed one).
        if not active and not samples and not clone_samples and time.monotonic() - last_premake > 30:
            last_premake = time.monotonic()
            if not db.samples_waiting():
                try:
                    if premake_samples():
                        log.info("Making the saved samples of voices that haven't one")
                except Exception:
                    log.exception("Couldn’t queue the voices' saved samples")

    log.info("Shutting down; unfinished books will resume on next start")
    for book_id, item in active.items():
        if item.finishing is None:
            db.update_book(book_id, status="queued", message="Resuming after restart")
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    clone_exec.shutdown(wait=False, cancel_futures=True)
    finisher.shutdown(wait=True)
    transcriber.shutdown()


if __name__ == "__main__":
    main()
