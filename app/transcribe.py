"""Read along for audiobooks you add: Whisper writes down what's said, with the
time of every word.

Runs on a thread of the worker, outside the voice workers' process pool, so
voice samples and voicing carry on meanwhile. One book and one chapter at a
time: ffmpeg decodes the book to 16 kHz mono as a stream, and Whisper gets it
in windows of a few minutes, each cut at a quiet moment, so even a long book is
never held in memory. Each chapter's transcript is cached
(/data/books/<id>/transcript/NNNN.json), so a restart carries on where it was.

TRANSCRIBE_MODEL, measured on three chapters (21 min) of a Forge-voiced
Number the Stars against its own text and word timings, on a 6-core i7-5820K
(int8, 6 threads, other containers busy alongside):
  small, batched    2.9% word errors, word starts 0.09 s off (median), 0.17 s (95%), 24 min per audio hour
  small, one by one 3.0%                          0.10 s          0.17 s       54 min
  base, batched     4.5% (two chapters)           0.08 s          0.15 s        7 min
  base, one by one  4.3%                          0.09 s          0.17 s       16 min
Timing is as good with either; small mishears about a third fewer words, names
especially ("Hans Christian Andersen", not "Hands Christian Anderson"). The
text is what people read, so small is the default; base is the quick choice.

On the GPU (WHISPER_DEVICE=cuda) the model is loaded only if it fits beside
everything else that may use video memory: the voice workers at their caps
(TTS_WORKERS x GPU_MEM_PER_WORKER_GB), the narration check's share
(WHISPER_GPU_GB) and the GPU_RESERVE_GB margin, and only if that much is free
right now. Otherwise it runs on the CPU. It's unloaded when the book is done,
and a book that runs out of video memory carries on on the CPU.
"""
from __future__ import annotations

import gc
import json
import logging
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config, db, qa, transcript

log = logging.getLogger("forge.transcribe")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # not a line per window

RATE = 16000  # what Whisper listens at
WINDOW = 8 * 60  # seconds of audio given to Whisper at once...
WINDOW_SLACK = 30  # ...cut at the quietest moment this close to the window's end
BEAM = 5
# Video memory each model needs on the GPU (float16 weights, beam search with
# word timings, the CUDA context), in GB. Whisper base measured 0.27 GB for the
# narration check; the rest are estimated from model size. TRANSCRIBE_GPU_GB overrides.
GPU_NEEDS_GB = {"turbo": 2.2, "large": 4.2, "medium": 2.4, "small": 1.1, "base": 0.5, "tiny": 0.4}
# ISO 639-2 language tags in audio files, as Whisper's codes.
LANGUAGES = {"eng": "en", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es", "ita": "it",
             "nld": "nl", "dut": "nl", "por": "pt", "swe": "sv", "dan": "da", "nor": "no", "nob": "no",
             "fin": "fi", "pol": "pl", "rus": "ru", "ukr": "uk", "ces": "cs", "cze": "cs", "hun": "hu",
             "ell": "el", "gre": "el", "tur": "tr", "jpn": "ja", "zho": "zh", "chi": "zh", "kor": "ko",
             "ara": "ar", "heb": "he", "hin": "hi"}


class Stopped(Exception):
    """The worker is shutting down; the book carries on after a restart."""


class Cancelled(Exception):
    """The book was cancelled or removed."""


def gpu_need_gb(model: str) -> float:
    if config.TRANSCRIBE_GPU_GB:
        return config.TRANSCRIBE_GPU_GB
    return next((gb for key, gb in GPU_NEEDS_GB.items() if key in model.lower()), 2.0)


def pick_device() -> tuple[str, str]:
    """("cuda" or "cpu", why it's the CPU when the GPU was asked for)."""
    if config.WHISPER_DEVICE != "cuda":
        return "cpu", ""
    need = gpu_need_gb(config.TRANSCRIBE_MODEL)
    gpu = config.gpu_memory_gb()
    if not gpu:
        return "cpu", "no GPU was found"
    free, _ = gpu
    # The narration check may load its own Whisper at any time; keep its share.
    check = config.WHISPER_GPU_GB if config.QA_DEFAULT and qa._model is None else 0.0
    room = free - config.GPU_RESERVE_GB - check
    if config.DEVICE == "cuda" and config.GPU_FREE_GB is not None:
        # Voice workers may not be using their whole share at this moment, but may soon.
        room = min(room, config.GPU_FREE_GB - config.GPU_RESERVE_GB - check
                   - config.TTS_WORKERS * config.GPU_MEM_PER_WORKER_GB)
    if need > room:
        return "cpu", f"the GPU doesn’t have {need:.1f} GB of video memory to spare"
    return "cuda", ""


class Whisper:
    """The model for one book: loaded when it's first needed, dropped when the
    book is done so it doesn't sit in (video) memory."""

    def __init__(self):
        self.model = None
        self.device = None
        self.why_cpu = ""

    def get(self):
        if self.model is None:
            device, self.why_cpu = pick_device()
            if self.why_cpu:
                log.info("Transcribing on the CPU: %s", self.why_cpu)
            self.model = self._load(device)
        return self.model

    def transcribe(self, audio: np.ndarray, language: str | None):
        """(segments, info). On the CPU, VAD-cut pieces of the window are decoded
        TRANSCRIBE_BATCH at a time, which measured 2.7x faster than one by one
        at the same accuracy. On the GPU one at a time: a batch multiplies the
        beam search's memory, and one at a time is already fast there."""
        model = self.get()
        options = dict(language=language, beam_size=BEAM, word_timestamps=True, vad_filter=True,
                       language_detection_segments=4)
        if self.device == "cpu" and config.TRANSCRIBE_BATCH > 1:
            from faster_whisper import BatchedInferencePipeline
            return BatchedInferencePipeline(model).transcribe(audio, batch_size=config.TRANSCRIBE_BATCH, **options)
        return model.transcribe(audio, condition_on_previous_text=False, **options)

    def _load(self, device: str):
        from faster_whisper import WhisperModel
        options = dict(device=device, compute_type="float16" if device == "cuda" else "int8",
                       cpu_threads=config.TRANSCRIBE_THREADS, download_root=str(config.MODELS_DIR / "whisper"))
        try:
            try:  # the copy on disk, without asking Hugging Face about a newer one
                model = WhisperModel(config.TRANSCRIBE_MODEL, local_files_only=True, **options)
            except Exception as exc:
                if device == "cuda" and _out_of_memory(exc):
                    raise
                model = WhisperModel(config.TRANSCRIBE_MODEL, **options)  # first use: download it
        except Exception as exc:
            if device != "cuda":
                raise
            log.warning("Whisper couldn’t load on the GPU (%s); using the CPU", exc)
            self.why_cpu = "Whisper couldn’t load on the GPU"
            return self._load("cpu")
        self.device = device
        log.info("Whisper %s loaded on the %s", config.TRANSCRIBE_MODEL, device.upper())
        return model

    def to_cpu(self) -> None:
        """After running out of video memory: carry on on the CPU."""
        self.release()
        self.why_cpu = "the GPU ran out of video memory"
        self.model = self._load("cpu")

    def release(self) -> None:
        self.model = None
        gc.collect()


def _out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda" in text and "alloc" in text


class PcmStream:
    """The book as 16 kHz mono samples, decoded by ffmpeg as they're read."""

    def __init__(self, source: Path, start: float):
        self.pos = int(round(start * RATE))  # where the next sample is in the book
        seek = ["-ss", f"{start:.3f}"] if start > 0 else []
        # MP3 seeks by estimate (a VBR file can land seconds off), so it's
        # decoded from the start; other formats seek to the exact sample.
        before, after = ([], seek) if source.suffix.lower() == ".mp3" else (seek, [])
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", *before, "-i", str(source), *after,
             "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self, count: int) -> np.ndarray:
        data = bytearray()
        while len(data) < count * 2:
            chunk = self.proc.stdout.read(min(1 << 20, count * 2 - len(data)))
            if not chunk:
                break
            data += chunk
        samples = np.frombuffer(bytes(data[:len(data) // 2 * 2]), np.int16)
        self.pos += len(samples)
        return samples

    def skip(self, count: int) -> None:
        while count > 0:
            got = len(self.read(min(count, RATE * 60)))
            if not got:
                return
            count -= got

    def close(self) -> None:
        self.proc.kill()
        self.proc.wait()


def _quiet_cut(samples: np.ndarray, lo: int, hi: int) -> int:
    """The quietest moment (0.3 s average) between samples lo and hi."""
    frame = RATE // 50
    part = samples[lo:hi].astype(np.float32)
    n = len(part) // frame
    if n < 16:
        return hi
    energy = (part[:n * frame].reshape(n, frame) ** 2).mean(axis=1)
    smooth = np.convolve(energy, np.ones(15) / 15, mode="same")
    return lo + int(np.argmin(smooth[7:-7]) + 7) * frame + frame // 2


def quiet_stretches(audio: np.ndarray, at: float) -> list[list[float]]:
    """Where the reader is silent for 0.15 s or more, in seconds into the book.
    Long pauses mark paragraphs (transcript._paragraphs). Silence is judged
    against this window's own levels, so room tone and quiet readers both work."""
    frame = RATE // 100
    n = len(audio) // frame
    if n < 20:
        return []
    rms = np.sqrt((audio[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
    threshold = max(float(np.percentile(rms, 95)) * 0.03, float(np.percentile(rms, 10)) * 2, 1e-4)
    quiet = np.concatenate([[False], rms < threshold, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    return [[round(at + a / 100, 2), round(at + b / 100, 2)]
            for a, b in zip(edges[::2], edges[1::2]) if b - a >= 15]


def chapter_windows(stream: PcmStream, start: float, end: float):
    """Yield (time in the book, samples) for one chapter, in windows of about WINDOW seconds."""
    first, stop = int(round(start * RATE)), int(round(end * RATE))
    if stream.pos < first:
        stream.skip(first - stream.pos)  # a chapter that's turned off, or already done
    full = int((WINDOW + WINDOW_SLACK) * RATE)
    carry = np.zeros(0, np.int16)
    while True:
        at = stream.pos - len(carry)
        want = min(stop - stream.pos, full - len(carry))
        fresh = stream.read(want) if want > 0 else np.zeros(0, np.int16)
        buf = np.concatenate([carry, fresh]) if len(carry) else fresh
        ended = stream.pos >= stop or len(fresh) < want
        # A file's stated length can be a little off; much shorter means it couldn't be read.
        if ended and stop - stream.pos > max(5 * RATE, (stop - first) // 50):
            raise RuntimeError(f"the audio stops at {stream.pos / RATE / 60:.0f} minutes, before this chapter "
                               "ends. The file may be damaged; check that it plays to the end.")
        if not len(buf):
            return
        cut = len(buf) if ended else _quiet_cut(buf, int((WINDOW - WINDOW_SLACK) * RATE), len(buf))
        yield at / RATE, buf[:cut].astype(np.float32) / 32768.0
        carry = buf[cut:]
        if ended and not len(carry):
            return


def _language(book: dict, chapters: list[dict]) -> str | None:
    """The file's language tag, or what Whisper heard in a chapter done before."""
    for ch in chapters:
        data = transcript.load(book["id"], ch["idx"])
        if data and data.get("language"):
            return data["language"]
    tag = (book["meta"].get("language") or "").lower().split("-")[0]
    tag = LANGUAGES.get(tag, tag)
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        return tag if tag in _LANGUAGE_CODES else None
    except ImportError:
        return tag if len(tag) == 2 else None


def _cached(book_id: str, ch: dict) -> bool:
    data = transcript.load(book_id, ch["idx"])
    end = (ch["start"] or 0) + (ch["duration"] or 0)
    return bool(data) and abs(data.get("start", -1) - (ch["start"] or 0)) < 0.05 \
        and abs(data.get("end", -1) - end) < 0.05


class _Progress:
    """Status line, chapter meter and speed, written at most once a second."""

    def __init__(self, book_id: str, total: float, done: float, count: int):
        self.book_id, self.total, self.done, self.count = book_id, max(1.0, total), done, count
        self.started = time.monotonic()
        self.heard = 0.0  # seconds of audio transcribed in this run
        self.last = 0.0
        self.last_seconds = 0.0
        self.last_check = 0.0

    def update(self, ch: dict, n: int, seconds: float, force: bool = False) -> None:
        now = time.monotonic()
        # A batch's segments arrive all at once: still show how far it got.
        if not force and now - self.last < 1.0 and seconds - self.last_seconds < 20:
            return
        self.last, self.last_seconds = now, seconds
        if db.get_status(self.book_id) != "transcribing":
            raise Cancelled()  # don't write over "Cancelled" with progress
        elapsed = now - self.started
        fields = {"rtf": round((self.heard + seconds) / elapsed, 1)} if elapsed > 5 and self.heard + seconds else {}
        db.update_chapters(self.book_id, [{"idx": ch["idx"], "units_done": int(seconds), "state": "transcribing"}])
        db.update_book(self.book_id, progress=round(min(0.99, (self.done + seconds) / self.total), 4),
                       message=f"Transcribing chapter {n} of {self.count}", **fields)

    def check(self, stop: threading.Event) -> None:
        if stop.is_set():
            raise Stopped()
        now = time.monotonic()
        if now - self.last_check >= 2.0:
            self.last_check = now
            if db.get_status(self.book_id) != "transcribing":
                raise Cancelled()


def transcribe_chapter(whisper: Whisper, stream: PcmStream, ch: dict, language: str | None,
                       report, check) -> tuple[list, list, str | None]:
    """Words of one chapter, [text, start, end, probability], and its quiet
    stretches; times in seconds into the book."""
    start = ch["start"] or 0.0
    words: list = []
    quiet: list = []
    for at, audio in chapter_windows(stream, start, start + (ch["duration"] or 0.0)):
        quiet += quiet_stretches(audio, at)
        for attempt in (1, 2):
            got: list = []
            try:
                segments, info = whisper.transcribe(audio, language)
                language = language or info.language
                for seg in segments:  # Whisper works as the segments are read
                    got += [[w.word, round(at + w.start, 3), round(at + w.end, 3), round(w.probability, 3)]
                            for w in seg.words or []]
                    report(at + seg.end - start)
                    check()
                break
            except RuntimeError as exc:
                if attempt == 2 or whisper.device != "cuda" or not _out_of_memory(exc):
                    raise
                log.warning("Whisper ran out of video memory; carrying on on the CPU")
                whisper.to_cpu()
        words += got
    joined: list = []  # a pause split by a window's edge is one pause
    for s, e in quiet:
        if joined and s - joined[-1][1] <= 0.02:
            joined[-1][1] = e
        else:
            joined.append([s, e])
    return words, joined, language


def run_book(book_id: str, stop: threading.Event, publish=None) -> None:
    """Transcribe an added audiobook's chapters, write its text and its Read along
    map. `publish(book_id, message)` sends it to the server, as finished voiced books are."""
    whisper, stream = Whisper(), None
    try:
        book = db.get_book(book_id)
        if book is None:
            return
        source = Path(book["output"] or "")
        if not source.is_file():
            raise ValueError("The audiobook file isn’t there any more. Remove the book and add it again.")
        chapters = sorted((c for c in db.chapters(book_id) if c["include"]), key=lambda c: c["start"] or 0)
        if not chapters:
            raise ValueError("No chapters are turned on. Turn on at least one chapter.")
        todo = {c["idx"] for c in chapters if not _cached(book_id, c)}
        seconds = {c["idx"]: c["duration"] or 0.0 for c in chapters}
        progress = _Progress(book_id, sum(seconds.values()),
                             sum(s for i, s in seconds.items() if i not in todo), len(chapters))
        db.update_chapters(book_id, [
            {"idx": c["idx"], "units_total": max(1, int(seconds[c["idx"]])),
             "units_done": 0 if c["idx"] in todo else max(1, int(seconds[c["idx"]])),
             "state": "pending" if c["idx"] in todo else "done"} for c in chapters])
        language = _language(book, chapters)
        folder = config.BOOKS_DIR / book_id
        for n, ch in enumerate(chapters, 1):
            if ch["idx"] not in todo:
                continue
            progress.check(stop)
            progress.update(ch, n, 0.0, force=True)
            if stream is None:
                stream = PcmStream(source, ch["start"] or 0.0)
            words, quiet, language = transcribe_chapter(
                whisper, stream, ch, language,
                lambda s, ch=ch, n=n: progress.update(ch, n, s), lambda: progress.check(stop))
            data = {"v": 1, "model": config.TRANSCRIBE_MODEL, "device": whisper.device, "language": language,
                    "start": ch["start"] or 0.0, "end": (ch["start"] or 0.0) + seconds[ch["idx"]],
                    "words": words, "quiet": quiet}
            path = transcript.transcript_path(book_id, ch["idx"])
            path.parent.mkdir(exist_ok=True)  # not parents: a removed book's folder stays gone
            path.with_suffix(".part").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                                                 encoding="utf-8")
            path.with_suffix(".part").replace(path)
            count = transcript.write_text(book_id, ch["idx"], ch["title"], data)
            progress.heard += seconds[ch["idx"]]
            progress.done += seconds[ch["idx"]]
            db.update_chapters(book_id, [{"idx": ch["idx"], "words": count, "state": "done",
                                          "units_done": max(1, int(seconds[ch["idx"]]))}])
        for ch in chapters:  # text for chapters transcribed before (or sent here) without it
            data = transcript.load(book_id, ch["idx"])
            if data and not (folder / "text" / f"{ch['idx']:04d}.txt").exists():
                count = transcript.write_text(book_id, ch["idx"], ch["title"], data)
                db.update_chapters(book_id, [{"idx": ch["idx"], "words": count}])
        transcript.build_map(book_id)
        note = f" Transcribed on the CPU: {whisper.why_cpu}." if whisper.why_cpu else ""
        fields = {}
        elapsed = time.monotonic() - progress.started
        if progress.heard and elapsed > 1:
            fields["rtf"] = round(progress.heard / elapsed, 1)
        message = f"Done.{note}" if note else "Done"
        db.update_book(book_id, status="done", progress=1.0, message=message, **fields)
        log.info("Transcribed %s: %.0f s of audio in %.0f s", book_id, progress.heard, elapsed)
        if publish and config.PUBLISH_URL and config.PUBLISH_AUTO:
            publish(book_id, message)
    except Stopped:
        db.update_book(book_id, status="queued", message="Resuming after restart")
    except Cancelled:
        log.info("Stopped transcribing %s", book_id)
    except Exception as exc:
        log.exception("Transcribing %s failed", book_id)
        if db.get_status(book_id) == "transcribing":
            db.update_book(book_id, status="failed",
                           message=f"Transcribing failed: {exc}"[:500] + " Retry to carry on; finished chapters are kept.")
    finally:
        if stream is not None:
            stream.close()
        whisper.release()


class Runner:
    """Takes queued audiobooks one at a time and transcribes them on a thread
    of the worker, beside the voice workers' pool."""

    def __init__(self, stop: threading.Event, publish=None):
        self.stop = stop
        self.publish = publish
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe")
        self.current: Future | None = None
        self.last_poll = 0.0

    def poll(self) -> None:
        if self.current is not None and not self.current.done():
            return
        if time.monotonic() - self.last_poll < 1.0:
            return
        self.last_poll = time.monotonic()
        queued = [b for b in db.books_by_status("queued") if b["kind"] == "audio"]
        if queued:
            book = queued[0]
            db.update_book(book["id"], status="transcribing", progress=0, rtf=None, message="Starting Whisper")
            self.current = self.pool.submit(run_book, book["id"], self.stop, self.publish)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
