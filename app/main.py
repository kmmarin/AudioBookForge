"""Web app: upload, review, voice selection, live progress and downloads."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.error
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import (audiofile, config, covers, db, gpus, listeners, lookup, parsers, previews, publish, readalong,
               scenes, textprep, transcript, versions, voices)

BUSY = {"voicing", "assembling", "checking", "transcribing"}
STARTABLE = {"ready", "done", "failed", "cancelled"}
SAMPLE_SECONDS = 15
log = logging.getLogger("forge.web")

parse_pool = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    config.BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for book in db.books_by_status("parsing"):
        parse_pool.submit(parse_book, book["id"])
    parse_pool.submit(versions.keep_all)  # books finished before every voice was kept
    yield


app = FastAPI(title="Audiobook Forge", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.APP_DIR / "static"), name="static")
listeners.install(app)


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """After an update, browsers must not keep running the old app.js and app.css
    against the new server. no-cache still lets them reuse a file that hasn't
    changed (a quick 304), it just makes them ask first."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def book_dir(book_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", book_id):
        raise HTTPException(404, "Book not found")
    return config.BOOKS_DIR / book_id


def require_book(book_id: str) -> dict:
    book_dir(book_id)
    book = db.get_book(book_id)
    if book is None:
        raise HTTPException(404, "Book not found")
    return book


def _host(url: str | None) -> str:
    return urlparse(url).netloc if url else ""


FORWARD_HOURS = 12  # a voice-again request sent to the PC counts as running for this long


def pending_change(book: dict) -> str | None:
    """Why a finished book's file no longer matches it: "voice" when it's set to
    a voice (mood, speed) it hasn't been made in, "edits" when its text,
    details, cover or format changed. None when the file is up to date."""
    if book["status"] != "done" or book["kind"] == "audio":  # an added audiobook is never rebuilt
        return None
    if not book["built"]:  # finished before builds recorded their settings
        return "edits" if "Rebuild to apply" in (book["message"] or "") else None
    built = json.loads(book["built"])
    if any(built.get(k) != book[k] for k in ("voice", "mood", "speed")):
        return "voice"
    if book["edited"] or built.get("fmt") != book["fmt"]:
        return "edits"
    return None


def voicing_elsewhere(book: dict) -> str | None:
    """The PC voicing this server copy again, while the new copy is on its way."""
    if book["origin"] and book["forwarded"] and time.time() - book["forwarded"] < FORWARD_HOURS * 3600:
        return _host(book["origin"])
    return None


def book_json(book: dict, chapters: list[dict] | None = None, progress: dict | None = None,
              kept: list[dict] | None = None) -> dict:
    chapters = db.chapters(book["id"]) if chapters is None else chapters
    if progress is None:
        progress = db.get_progress(book["id"])
    kept = db.versions(book["id"]) if kept is None else kept
    pending = pending_change(book)
    words = sum(c["words"] for c in chapters if c["include"])
    seconds = words / config.WORDS_PER_MINUTE * 60 / (book["speed"] or 1.0)
    if book["kind"] == "audio":  # an audiobook added for Read along: its real length
        seconds = sum(c["duration"] or 0 for c in chapters)
    return {
        "id": book["id"], "filename": book["filename"], "status": book["status"], "kind": book["kind"],
        "meta": book["meta"], "voice": book["voice"], "mood": book["mood"], "speed": book["speed"],
        "fmt": book["fmt"], "qa": bool(book["qa"]), "progress": book["progress"], "rtf": book["rtf"],
        "message": book["message"], "has_cover": bool(book["has_cover"]),
        "output_name": Path(book["output"]).name if book["output"] else None,
        "in_library": bool(book["output"]) and not Path(book["output"]).is_relative_to(config.BOOKS_DIR),
        # The map stays usable while a new voice is being made: the old audio plays until it's replaced.
        "readalong": bool(book["output"]) and readalong.path_for(book["id"]).exists(),
        "read_url": publish.read_url(book),  # on the server once published there
        "origin": _host(book["origin"]) if book["origin"] else None,  # voiced on another Forge
        "changed": pending is not None, "pending": pending, "voicing_elsewhere": voicing_elsewhere(book),
        "versions": versions.public(book, kept),  # every voice it's been made in
        "listened": (progress or {}).get("position") if progress is not None else None,
        "updated": book["updated"], "est_seconds": round(seconds),
        "chapters": [
            {k: c[k] for k in ("idx", "title", "include", "words", "units_total",
                               "units_done", "state", "duration", "wer")}
            for c in chapters
        ],
    }


def all_books_json(listener: int | None = None) -> list[dict]:
    """listened is that listener's (the default one's without a listener)."""
    grouped = db.all_chapters()
    places = db.all_progress(listener)
    kept = db.all_versions()
    return [book_json(b, grouped.get(b["id"], []), places.get(b["id"], {}), kept.get(b["id"], []))
            for b in db.list_books()]


SOURCE_NAMES = {"openlibrary": "Open Library", "audible": "Audible", "google": "Google Books"}
FIELD_NAMES = {"title": "title", "author": "author", "series": "series",
               "series_index": "number in series", "year": "year", "genre": "genre",
               "publisher": "publisher", "description": "description", "cover": "cover"}


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def match_online(folder: Path, meta: dict, has_cover: bool,
                 results: list[dict] | None = None) -> tuple[dict, str, bool]:
    """Fill in what the file didn't say from the lookup sources.

    Returns the new meta, a note for the status line, and whether there's a cover.
    """
    try:
        found = lookup.best_match(meta, results)
    except Exception:
        return meta, "", has_cover
    if not found:
        return meta, "", has_cover
    meta, filled = lookup.fill_missing(meta, found)
    if not has_cover and found.get("cover_url"):
        try:
            (folder / "cover.jpg").write_bytes(covers.normalize(lookup.fetch_cover(found["cover_url"])))
            has_cover = True
            filled.append("cover")
        except Exception:
            pass
    if not filled:
        return meta, "", has_cover
    sources = _and([SOURCE_NAMES[s] for s in found["sources"]])
    return meta, f" Added {_and([FIELD_NAMES[f] for f in filled])} from {sources}.", has_cover


def parse_book(book_id: str) -> None:
    book = db.get_book(book_id)
    if book is None:
        return
    folder = config.BOOKS_DIR / book_id
    try:
        audio = book["kind"] == "audio"
        parts = audiofile.parts_of(folder) if audio else []
        source = None
        if parts:
            names = json.loads((folder / audiofile.PARTS_JSON).read_text(encoding="utf-8"))
            parsed = audiofile.read_parts(parts, names)
        else:
            # An added audiobook already imported has its file in the library.
            source = next(folder.glob("source*"), None) or (Path(book["output"]) if audio and book["output"] else None)
            if source is None or not source.exists():
                raise parsers.UnsupportedFormat("The file isn’t there any more. Remove this book and add it again.")
            parsed = audiofile.read(source, book["filename"]) if audio else parsers.parse(source, book["filename"])
        text_dir = folder / "text"
        shutil.rmtree(text_dir, ignore_errors=True)
        text_dir.mkdir(parents=True)
        rows = []
        for idx, chapter in enumerate(parsed.chapters):
            if audio:  # its text comes from Whisper when Read along is made
                rows.append({"idx": idx, "title": chapter["title"], "include": True, "words": 0,
                             "start": chapter["start"], "duration": round(chapter["end"] - chapter["start"], 3)})
                continue
            (text_dir / f"{idx:04d}.txt").write_text(chapter.text, encoding="utf-8")
            rows.append({"idx": idx, "title": chapter.title, "include": chapter.include,
                         "words": textprep.word_count(chapter.text)})
        has_cover = False
        if parsed.cover:
            try:
                (folder / "cover.jpg").write_bytes(covers.normalize(parsed.cover))
                has_cover = True
            except Exception:
                pass
        meta, note = parsed.meta, ""
        if config.METADATA_LOOKUP:
            meta, note, has_cover = match_online(folder, meta, has_cover)
        db.set_chapters(book_id, rows)
        on = sum(1 for r in rows if r["include"])
        found = f"Found {len(rows)} chapter{'' if len(rows) == 1 else 's'}, {on} turned on."
        output = {}
        if audio:  # the file itself is what plays and downloads
            db.update_book(book_id, meta=meta, has_cover=int(has_cover))
            output["output"], placed = import_audiobook(book_id, meta, parsed.chapters, parts, source)
            found = (f"Found {len(rows)} chapter{'' if len(rows) == 1 else 's'}, "
                     f"{audiofile.length_label(parsed.duration)}.{placed} Make Read along to get its text.")
        db.update_book(book_id, status="ready", meta=meta, has_cover=int(has_cover), message=found + note, **output)
    except parsers.UnsupportedFormat as exc:
        db.update_book(book_id, status="failed", message=str(exc))
    except Exception as exc:
        db.update_book(book_id, status="failed", message=f"Couldn’t read this file: {exc}")


def import_audiobook(book_id: str, meta: dict, chapters: list[dict], parts: list[Path],
                     source: Path | None) -> tuple[str, str]:
    """Put an added audiobook into the library, where a voiced copy of it would
    go: its parts combined into one M4B, or its single file moved there as it is.
    Never over another file: then it stays with the book in Forge. Returns (the
    file it has now, a note for the status line)."""
    from . import assemble
    folder = config.BOOKS_DIR / book_id
    cover = folder / "cover.jpg"
    cover = cover if cover.exists() else None
    if parts:
        out = assemble.output_path(meta, "m4b")
        target = audiofile.free_path(out, None) or folder / "combined.m4b"
        db.update_book(book_id, message=f"Combining {len(parts)} parts into one M4B")
        audiofile.combine(parts, [(c["title"], c["end"] - c["start"]) for c in chapters], meta, cover, target)
        shutil.rmtree(folder / audiofile.PARTS_DIR, ignore_errors=True)
        if target != out:
            return str(target), (f" Combined its {len(parts)} parts into one M4B. There’s already a file at "
                                 f"“{out.name}” in your library, so it stays in Forge.")
        return str(out), f" Combined its {len(parts)} parts into one M4B in your library."
    out = assemble.output_path(meta, source.suffix.lstrip(".").lower())
    if source == out:
        return str(out), ""
    if audiofile.free_path(out, None) is None:
        return str(source), f" There’s already a file at “{out.name}” in your library, so this one stays in Forge."
    audiofile.place(source, out, cover)
    return str(out), " Added to your library as it is."


_refile_lock = threading.Lock()


def refile_audiobook(book_id: str) -> None:
    """An imported audiobook follows changes to its details: it moves to where
    they now put it, and the file Forge made from parts is retagged with them,
    its chapter titles and cover (in seconds: nothing is re-encoded). A file
    that came as one keeps its own tags; only its place and cover change."""
    from . import assemble
    with _refile_lock:
        book = db.get_book(book_id)
        if not book or book["kind"] != "audio" or not book["output"] or book["status"] in BUSY | {"queued", "parsing"}:
            return
        src = Path(book["output"])
        if not src.exists():
            return
        folder = config.BOOKS_DIR / book_id
        out = assemble.output_path(book["meta"], src.suffix.lstrip(".").lower())
        if out != src and audiofile.free_path(out, None) is None:
            return  # another file is there: this one stays where it is
        cover = folder / "cover.jpg"
        cover = cover if cover.exists() else None
        try:
            if audiofile.combined(folder):
                chapters = [(c["title"], c["duration"] or 0) for c in db.chapters(book_id)]
                assemble.retag(src, out, book["meta"], "", chapters, "m4b", cover)
                if out != src:
                    src.unlink(missing_ok=True)
            elif out != src:
                audiofile.place(src, out, cover)
            elif cover is not None and config.OUTPUT_LAYOUT == "abs":
                shutil.copyfile(cover, out.parent / "cover.jpg")
        except Exception:
            log.exception("Couldn’t move %s in the library", book_id)
            return
        if out != src:
            if not src.is_relative_to(config.BOOKS_DIR):
                assemble.remove_output(src)  # the folders it leaves empty
            db.update_book(book_id, output=str(out))


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    return {
        "voices": [voices.public(v) for v in voices.list_voices()],
        "default_voice": voices.default_id(),
        "formats": [{"id": k, "label": v["label"]} for k, v in config.FORMATS.items()],
        "default_format": config.DEFAULT_FORMAT,
        "accept": parsers.supported_extensions() + audiofile.EXTENSIONS,
        "qa_default": config.QA_DEFAULT,
        "qa_flag": config.QA_FLAG_WER,
        "workers": config.TTS_WORKERS,
        "publish_url": config.PUBLISH_URL or None,
        "publish_host": _host(publish.site_url() or config.PUBLISH_URL) or None,
        "site_url": config.SITE_URL or None,
        "voicelab": bool(config.VOICELAB_URL),  # shows "Create a voice", even while it's offline
    }


@app.get("/api/books")
def list_books(listener: int | None = None):
    return all_books_json(listener)


@app.get("/api/books/{book_id}")
def get_book(book_id: str, listener: int | None = None):
    book = require_book(book_id)
    # places: how far everyone has got, for the book's Details.
    return {**book_json(book, progress=db.get_progress(book_id, listener) or {}),
            "places": [{"listener": p["listener_id"], "name": p["name"], "chapter": p["chapter"],
                        "position": p["position"], "updated": p["updated"]} for p in db.book_places(book_id)]}


@app.post("/api/books")
async def upload_books(files: list[UploadFile] = File(...)):
    """Each document is a book. Audio files are audiobooks to import and get the
    text of; the parts of one audiobook uploaded together become one book."""
    created = []
    staged: list[tuple[str, Path]] = []
    staging = config.BOOKS_DIR / f".upload-{uuid.uuid4().hex[:8]}"
    try:
        for upload in files:
            filename = Path(upload.filename or "book.txt").name
            if audiofile.is_audio(filename):
                staging.mkdir(parents=True, exist_ok=True)
                path = staging / f"{len(staged):04d}{Path(filename).suffix.lower()}"
                with path.open("wb") as out:
                    while chunk := await upload.read(1 << 20):
                        out.write(chunk)
                staged.append((filename, path))
                continue
            book_id = uuid.uuid4().hex[:12]
            folder = config.BOOKS_DIR / book_id
            folder.mkdir(parents=True)
            with (folder / f"source{Path(filename).suffix.lower()}").open("wb") as out:
                while chunk := await upload.read(1 << 20):
                    out.write(chunk)
            db.create_book(book_id, filename, voices.default_id(), config.DEFAULT_FORMAT, config.QA_DEFAULT)
            parse_pool.submit(parse_book, book_id)
            created.append(book_id)
        for parts in await asyncio.to_thread(audiofile.group, staged) if staged else []:
            book_id = uuid.uuid4().hex[:12]
            folder = config.BOOKS_DIR / book_id
            folder.mkdir(parents=True)
            if len(parts) == 1:
                filename, path = parts[0]
                shutil.move(str(path), folder / f"source{Path(filename).suffix.lower()}")
            else:
                filename = audiofile.book_name([f for f, _ in parts])
                audiofile.save_parts(folder, parts)
            db.create_book(book_id, filename, None, config.DEFAULT_FORMAT, config.QA_DEFAULT, kind="audio")
            parse_pool.submit(parse_book, book_id)
            created.append(book_id)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"ids": created}


class ChapterPatch(BaseModel):
    idx: int
    title: str | None = None
    include: bool | None = None


class BookPatch(BaseModel):
    meta: dict | None = None
    voice: str | None = None
    mood: str | None = None
    speed: float | None = None
    fmt: str | None = None
    qa: bool | None = None
    chapters: list[ChapterPatch] | None = None


def apply_patch(book: dict, patch: BookPatch) -> None:
    if book["status"] in BUSY:
        raise HTTPException(409, "This book is being voiced. Cancel it to make changes.")
    fields: dict = {}
    if patch.voice is not None:
        if voices.get(patch.voice) is None:
            raise HTTPException(400, "Unknown voice")
        fields["voice"] = patch.voice
    if patch.voice is not None or patch.mood is not None:
        voice = voices.get(fields.get("voice", book["voice"])) or voices.get(voices.default_id())
        if patch.mood is not None and patch.mood not in voices.mood_ids(voice):
            raise HTTPException(400, f"{voice['label']} has no {patch.mood} mood.")
        # A new voice keeps the mood if it has one by the same name, else takes its first.
        fields["mood"] = voices.pick_mood(voice, patch.mood or book["mood"])
    if patch.fmt is not None:
        if patch.fmt not in config.FORMATS:
            raise HTTPException(400, "Unknown format")
        fields["fmt"] = patch.fmt
    if patch.speed is not None:
        fields["speed"] = round(min(max(patch.speed, 0.5), 2.0), 2)
    if patch.qa is not None:
        fields["qa"] = int(patch.qa)
    if patch.meta is not None:
        allowed = {"title", "author", "narrator", "series", "series_index", "year", "genre",
                   "description", "language", "publisher"}
        meta = dict(book["meta"])
        for key, value in patch.meta.items():
            if key in allowed:
                value = str(value or "").strip()
                if value:
                    meta[key] = value[:4000]
                else:
                    meta.pop(key, None)
        if meta != book["meta"]:
            fields["meta"] = meta
    chapters_changed = False
    if patch.chapters:
        have = {c["idx"]: c for c in db.chapters(book["id"])}
        rows = []
        for ch in patch.chapters:
            old = have.get(ch.idx)
            row = {"idx": ch.idx}
            if ch.title is not None and ch.title.strip() and (not old or old["title"] != ch.title.strip()[:200]):
                row["title"] = ch.title.strip()[:200]
            if ch.include is not None and (not old or bool(old["include"]) != ch.include):
                row["include"] = int(ch.include)
            if len(row) > 1:
                rows.append(row)
        db.update_chapters(book["id"], rows)
        chapters_changed = bool(rows)
    # An added audiobook is never rebuilt: its Read along takes changes at once.
    audio = book["kind"] == "audio"
    # Voice, mood, speed and format are compared with what the file was built
    # with (pending_change); details and chapters can only be marked as edited.
    if book["status"] == "done" and not audio and ("meta" in fields or chapters_changed):
        fields["edited"] = 1
    if chapters_changed and not audio:
        db.outdate_versions(book["id"])  # chapter titles are read out, and skipped chapters change the book
    if fields:
        db.update_book(book["id"], **fields)
    if book["status"] == "done" and not audio and fields.keys() & {"voice", "mood", "speed"}:
        _switch_if_made(book["id"])
    if audio and chapters_changed and book["status"] == "done":
        refresh_audio_map(book["id"])
    if audio and ("meta" in fields or chapters_changed):
        parse_pool.submit(refile_audiobook, book["id"])  # its place in the library, and its tags if Forge made it


def _switch_if_made(book_id: str) -> bool:
    """A finished book set to a voice it was already made in switches to that copy."""
    book = db.get_book(book_id)
    vid = versions.version_id(book["voice"], book["mood"], book["speed"])
    if vid == versions.current_id(book):
        return False
    return versions.use(book, vid)


def refresh_audio_map(book_id: str) -> None:
    """An added audiobook's Read along takes new chapter titles and text at once:
    the map is rebuilt from what Whisper heard, without listening again."""
    transcript.build_map(book_id)
    _map_cache.pop(book_id, None)
    if not transcript.has_all(book_id):
        db.update_book(book_id, message="Changes saved. Make Read along to add the chapters you turned on.")


@app.patch("/api/books/{book_id}")
def patch_book(book_id: str, patch: BookPatch):
    apply_patch(require_book(book_id), patch)
    return book_json(require_book(book_id))


class BulkPatch(BaseModel):
    ids: list[str]
    voice: str | None = None
    fmt: str | None = None
    speed: float | None = None
    qa: bool | None = None


@app.post("/api/bulk")
def bulk_patch(body: BulkPatch):
    changed = 0
    for book_id in body.ids:
        book = db.get_book(book_id)
        if book is None or book["status"] in BUSY or book["kind"] == "audio":  # no voice or format to set
            continue
        apply_patch(book, BookPatch(voice=body.voice, fmt=body.fmt, speed=body.speed, qa=body.qa))
        changed += 1
    return {"changed": changed}


class Ids(BaseModel):
    ids: list[str]


@app.post("/api/start")
def start_books(body: Ids):
    started = 0
    for book_id in body.ids:
        book = db.get_book(book_id)
        if book and book["status"] in STARTABLE and any(c["include"] for c in db.chapters(book_id)):
            db.update_book(book_id, status="queued", progress=0, message="Waiting for a worker")
            started += 1
    return {"started": started}


class RedoBody(BaseModel):
    chapters: list[int] | None = None  # None = every chapter the narration check flagged


@app.post("/api/books/{book_id}/redo")
def redo_chapters(book_id: str, body: RedoBody):
    """Voice some chapters again (by default the ones flagged as misread) and
    rebuild the book. Every other chapter's audio is reused."""
    book = require_book(book_id)
    if book["status"] not in STARTABLE:
        raise HTTPException(409, "This book is busy. Try again once it’s done.")
    chapters = {c["idx"]: c for c in db.chapters(book_id) if c["include"]}
    if body.chapters is None:
        picked = [i for i, c in chapters.items()
                  if c["wer"] is not None and c["wer"] >= config.QA_FLAG_WER]
    else:
        picked = [i for i in body.chapters if i in chapters]
    if not picked:
        raise HTTPException(400, "No chapters to redo. Only turned-on chapters can be voiced again.")
    units = book_dir(book_id) / "units"
    for idx in picked:
        for path in units.glob(f"{idx:04d}-*"):  # audio and its word timings
            path.unlink(missing_ok=True)
    db.update_chapters(book_id, [{"idx": i, "wer": None, "units_done": 0, "state": "pending"}
                                 for i in picked])
    db.update_book(book_id, status="queued", progress=0,
                   message=f"Voicing {len(picked)} chapter{'' if len(picked) == 1 else 's'} again")
    return {"chapters": picked}


@app.post("/api/books/{book_id}/cancel")
def cancel_book(book_id: str):
    book = require_book(book_id)
    if book["status"] in {"queued", "voicing", "transcribing"}:
        db.update_book(book_id, status="cancelled", message="Cancelled. Finished parts are kept.")
    return book_json(require_book(book_id))


@app.delete("/api/books/{book_id}")
def delete_book(book_id: str):
    book = require_book(book_id)
    if book["status"] in {"assembling", "checking"}:
        raise HTTPException(409, "This book is being finished. Remove it once it’s done.")
    if book["status"] in {"queued", "voicing", "transcribing"}:
        db.update_book(book_id, status="cancelled")  # the worker drops it before the files go
    db.delete_book(book_id)
    shutil.rmtree(book_dir(book_id), ignore_errors=True)
    return {"deleted": book_id}


class TextBody(BaseModel):
    text: str


@app.get("/api/books/{book_id}/chapters/{idx}/text")
def get_chapter_text(book_id: str, idx: int):
    require_book(book_id)
    path = book_dir(book_id) / "text" / f"{idx:04d}.txt"
    if not path.exists():
        raise HTTPException(404, "Chapter not found")
    return {"text": path.read_text(encoding="utf-8")}


@app.put("/api/books/{book_id}/chapters/{idx}/text")
def put_chapter_text(book_id: str, idx: int, body: TextBody):
    book = require_book(book_id)
    if book["status"] in BUSY:
        raise HTTPException(409, "This book is being voiced. Cancel it to edit text.")
    path = book_dir(book_id) / "text" / f"{idx:04d}.txt"
    if not path.exists():
        raise HTTPException(404, "Chapter not found")
    if path.read_text(encoding="utf-8") == body.text:
        return {"words": textprep.word_count(body.text)}
    path.write_text(body.text, encoding="utf-8")
    db.update_chapters(book_id, [{"idx": idx, "words": textprep.word_count(body.text), "wer": None}])
    if book["kind"] == "audio":
        if book["status"] == "done":
            refresh_audio_map(book_id)
        return {"words": textprep.word_count(body.text)}
    db.outdate_versions(book_id)  # copies made from the old words no longer match
    if book["status"] == "done":
        db.update_book(book_id, edited=1)
    return {"words": textprep.word_count(body.text)}


@app.get("/api/books/{book_id}/cover")
def get_cover(book_id: str):
    book = require_book(book_id)
    path = book_dir(book_id) / "cover.jpg"
    if path.exists():
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})
    data = covers.generate(book["meta"].get("title") or Path(book["filename"]).stem,
                           book["meta"].get("author", ""), size=480)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.post("/api/books/{book_id}/cover")
async def put_cover(book_id: str, file: UploadFile = File(...)):
    require_book(book_id)
    try:
        data = covers.normalize(await file.read())
    except Exception:
        raise HTTPException(400, "That file isn’t an image Forge can read. Use JPEG or PNG.")
    (book_dir(book_id) / "cover.jpg").write_bytes(data)
    _cover_changed(book_id)
    return {"ok": True}


@app.get("/api/books/{book_id}/lookup")
def lookup_book(book_id: str, title: str | None = None, author: str | None = None):
    """Online matches for this book, best first. title and author override the saved ones."""
    book = require_book(book_id)
    meta = dict(book["meta"])
    for key, value in (("title", title), ("author", author)):
        if value is not None and value.strip() != meta.get(key, ""):
            meta[key] = value.strip()
            meta.pop("isbn", None)  # searching for something else now
    if not meta.get("title") and not meta.get("isbn"):
        raise HTTPException(400, "Type a title to search for.")
    results = [c for c in lookup.search(meta) if c["score"] >= 0.45]
    return {"results": [lookup.public(c) for c in results[:12]],
            "sources": [SOURCE_NAMES[s] for s in config.METADATA_SOURCES
                        if s in SOURCE_NAMES and (s != "google" or config.GOOGLE_BOOKS_API_KEY)]}


@app.post("/api/books/{book_id}/match")
def match_book(book_id: str):
    """Update info: if one match is clearly right, add the details the book is
    missing; otherwise return the matches so the user can pick one."""
    book = require_book(book_id)
    if book["status"] in BUSY | {"queued", "parsing"}:
        raise HTTPException(409, "This book is busy. Update its info once it’s ready or done.")
    results = lookup.search(book["meta"])
    listed = [lookup.public(c) for c in results if c["score"] >= 0.45][:12]
    clear = bool(results) and max(c["score"] for c in results) >= config.MATCH_CLEAR
    note, has_cover = "", bool(book["has_cover"])
    if clear:
        meta, note, has_cover = match_online(book_dir(book_id), book["meta"], has_cover, results)
        note = note.strip()
        if note:
            done = book["status"] == "done"
            db.update_book(book_id, meta=meta, has_cover=int(has_cover), message=note, **({"edited": 1} if done else {}))
            if book["kind"] == "audio":
                parse_pool.submit(refile_audiobook, book_id)
    return {"clear": clear, "applied": bool(note), "note": note, "results": listed}


class CoverUrl(BaseModel):
    url: str


@app.post("/api/books/{book_id}/cover/fetch")
def fetch_cover(book_id: str, body: CoverUrl):
    """Use a cover found by the lookup."""
    require_book(book_id)
    try:
        data = covers.normalize(lookup.fetch_cover(body.url))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(502, "Couldn’t download that cover. Try another one, or upload your own.")
    (book_dir(book_id) / "cover.jpg").write_bytes(data)
    _cover_changed(book_id)
    return {"ok": True}


def _cover_changed(book_id: str) -> None:
    """The finished file carries its cover, so a new one needs a rebuild (an
    imported audiobook just takes it, beside it and in Forge's own file)."""
    book = db.get_book(book_id)
    if book["kind"] == "audio":
        db.update_book(book_id, has_cover=1)
        parse_pool.submit(refile_audiobook, book_id)
        return
    db.update_book(book_id, has_cover=1, **({"edited": 1} if book["status"] == "done" else {}))


@app.get("/api/books/{book_id}/file")
def download(book_id: str, v: str | None = None):
    """The finished file; v= picks the copy in another voice it was made in."""
    book = require_book(book_id)
    if v:
        kept = db.get_version(book_id, v)
        if kept is None or not Path(kept["file"]).exists():
            raise HTTPException(404, "That voice’s copy isn’t there any more.")
        title = Path(book["output"]).stem if book["output"] else (book["meta"].get("title") or "Audiobook")
        name = f"{title} ({versions.label(kept)}){Path(kept['file']).suffix}".replace("/", "-")
        return FileResponse(kept["file"], filename=name, media_type="application/octet-stream")
    if not book["output"] or not Path(book["output"]).exists():
        raise HTTPException(404, "The finished file isn’t there. Rebuild the book.")
    path = Path(book["output"])
    name = book["filename"] if path.is_relative_to(config.BOOKS_DIR) else path.name  # not "source.m4b"
    return FileResponse(path, filename=name, media_type="application/octet-stream")


def _attachment(name: str) -> dict:
    """A download's Content-Disposition, as FileResponse writes it (names beyond ASCII too)."""
    from urllib.parse import quote
    name = name.replace("/", "-").replace("\\", "-")
    quoted = quote(name)
    return {"Content-Disposition": f"attachment; filename*=utf-8''{quoted}" if quoted != name
            else f'attachment; filename="{name}"'}


@app.get("/api/books/{book_id}/text")
def export_text(book_id: str):
    """The book's text as one plain-text file: the chapters read aloud, each under its title."""
    book = require_book(book_id)
    title = book["meta"].get("title") or Path(book["filename"]).stem
    head = title + (f"\nby {book['meta']['author']}" if book["meta"].get("author") else "")
    parts = []
    for c in db.chapters(book_id):
        path = book_dir(book_id) / "text" / f"{c['idx']:04d}.txt"
        text = path.read_text(encoding="utf-8").strip() if c["include"] and path.exists() else ""
        if text:
            first = text.split("\n", 1)[0].strip()
            # Most chapters start with their own heading; the rest get their title on top.
            parts.append(text if not c["title"] or first.lower() == c["title"].strip().lower()
                         else f"{c['title']}\n\n{text}")
    if not parts:
        raise HTTPException(404, "This book has no text yet." if book["kind"] == "audio"
                            else "Every chapter is turned off, so there’s no text to export.")
    return Response("\n\n\n".join([head, *parts]) + "\n", media_type="text/plain; charset=utf-8",
                    headers=_attachment(f"{title}.txt"))


def _kept(book_id: str, version_id: str) -> tuple[dict, dict]:
    book = require_book(book_id)
    kept = db.get_version(book_id, version_id)
    if kept is None:
        raise HTTPException(404, "This book has no copy in that voice.")
    return book, kept


class VersionPatch(BaseModel):
    name: str | None = None


@app.patch("/api/books/{book_id}/versions/{version_id}")
def rename_version(book_id: str, version_id: str, body: VersionPatch):
    """Name a voiced copy ("Bedtime", "Grandpa"); blank goes back to the voice's name."""
    _kept(book_id, version_id)
    db.rename_version(book_id, version_id, (body.name or "").strip()[:80] or None)
    db.touch(book_id)
    return book_json(require_book(book_id))


@app.post("/api/books/{book_id}/versions/{version_id}/use")
def use_version(book_id: str, version_id: str):
    """Switch the book to a voice it was already made in: no voicing, seconds."""
    book, kept = _kept(book_id, version_id)
    if book["status"] in BUSY | {"queued"}:
        raise HTTPException(409, "This book is being voiced. Switch voices once it’s done.")
    if kept["outdated"]:
        raise HTTPException(409, "That copy was made before the text changed, so it no longer matches. "
                                 "Pick the voice and rebuild instead.")
    if not versions.use(book, version_id):
        raise HTTPException(404, "That voice’s copy isn’t there any more.")
    return book_json(require_book(book_id))


@app.delete("/api/books/{book_id}/versions/{version_id}")
def delete_version(book_id: str, version_id: str):
    """Remove a voiced copy, even the current one (another made voice takes over,
    or the book goes back to ready to voice)."""
    book, _ = _kept(book_id, version_id)
    if book["status"] in BUSY | {"queued"}:
        raise HTTPException(409, "This book is being voiced. Remove copies once it’s done.")
    result = versions.remove(book, version_id)
    return {"result": result, "book": book_json(require_book(book_id))}


# --------------------------------------------------------------------------
# Read along
# --------------------------------------------------------------------------

AUDIO_TYPES = {"m4b": "audio/mp4", "m4a": "audio/mp4", "mp3": "audio/mpeg",
               "opus": "audio/ogg", "flac": "audio/flac",
               # audiobooks added for Read along play as they came
               "aac": "audio/aac", "ogg": "audio/ogg", "oga": "audio/ogg", "wav": "audio/wav"}
_map_cache: dict[str, tuple[float, bytes]] = {}  # book id -> (mtime, gzipped map)


@app.get("/read/{book_id}")
def read_page(book_id: str):
    require_book(book_id)
    return FileResponse(config.APP_DIR / "static" / "read.html")


@app.get("/api/books/{book_id}/audio")
def stream_audio(book_id: str):
    """The finished file for the player: the right type, and seekable (Range)."""
    book = require_book(book_id)
    if not book["output"] or not Path(book["output"]).exists():
        raise HTTPException(404, "The finished file isn’t there. Rebuild the book.")
    path = Path(book["output"])
    return FileResponse(path, media_type=AUDIO_TYPES.get(path.suffix.lstrip("."), "application/octet-stream"))


@app.get("/api/books/{book_id}/scenes")
def get_scenes(book_id: str):
    """Watch and read: the book's scenes (times, movement, picture once drawn) and how drawing is going."""
    require_book(book_id)
    try:
        return scenes.public(book_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/books/{book_id}/scenes/{name}")
def scene_picture(book_id: str, name: str):
    require_book(book_id)
    path = scenes.folder(book_id) / name
    if not re.fullmatch(r"[0-9a-f]{16}\.jpg", name) or not path.exists():
        raise HTTPException(404, "No such picture")
    # Named after what it was drawn from: the same name is always the same picture.
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=31536000, immutable"})


@app.post("/api/books/{book_id}/scenes")
def make_scenes(book_id: str):
    """Draw the book's pictures (those it hasn't got yet), in the worker, on the clone server."""
    require_book(book_id)
    if not readalong.path_for(book_id).exists():
        raise HTTPException(409, "Make Read along first: the pictures follow its timing.")
    if not gpus.has("pictures"):
        raise HTTPException(400, "Pictures are drawn by a GPU server: add one in Settings.")
    scenes.plan(book_id)
    scenes.queue(book_id)
    scenes.set_state(book_id, state="queued", message="Waiting to draw the pictures")
    return scenes.public(book_id)


@app.delete("/api/books/{book_id}/scenes/queued")
def stop_scenes(book_id: str):
    """Stop drawing; the pictures already drawn stay."""
    require_book(book_id)
    (scenes.folder(book_id) / "queued").unlink(missing_ok=True)
    return scenes.public(book_id)


@app.get("/api/books/{book_id}/readalong")
def get_readalong(book_id: str, request: Request):
    """Where every word is in the audio. A long book's map is a few megabytes of
    JSON, so it goes out compressed (and is kept compressed in memory)."""
    import gzip
    book = require_book(book_id)
    path = readalong.path_for(book_id)
    if not path.exists() or not book["output"]:
        if book["status"] == "done":
            raise HTTPException(404, "This book was made before Read along existed.")
        raise HTTPException(404, "This book is still being made.")
    mtime = path.stat().st_mtime
    cached = _map_cache.get(book_id)
    if not cached or cached[0] != mtime:
        cached = (mtime, gzip.compress(path.read_bytes(), compresslevel=6))
        _map_cache[book_id] = cached
    headers = {"Cache-Control": "no-cache", "ETag": f'"{int(mtime)}"'}
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(cached[1], media_type="application/json",
                        headers={**headers, "Content-Encoding": "gzip"})
    return Response(gzip.decompress(cached[1]), media_type="application/json", headers=headers)


@app.get("/api/books/{book_id}/textmap")
def get_text_map(book_id: str):
    """A book's text in the reader's shape, before it has audio: read it now, voice it later."""
    book = require_book(book_id)
    chapters = []
    for c in db.chapters(book_id):
        path = book_dir(book_id) / "text" / f"{c['idx']:04d}.txt"
        if c["include"] and path.exists():
            chapters.append((c["title"], path.read_text(encoding="utf-8")))
    doc = readalong.text_only(chapters)
    if not any(c["blocks"] for c in doc["chapters"]):
        raise HTTPException(404, "This book has no text to show yet.")
    return doc


class ProgressBody(BaseModel):
    chapter: int
    offset: float
    position: float
    word: int | None = None  # word number in the chapter: survives a new voice or a rebuild


# Progress and checkpoints belong to a listener: ?listener=<id>, or the default
# listener without one (see listeners.resolve).

@app.get("/api/books/{book_id}/progress")
def get_progress(book_id: str, listener: int | None = None):
    require_book(book_id)
    return db.get_progress(book_id, listeners.resolve(listener))


@app.put("/api/books/{book_id}/progress")
def put_progress(book_id: str, body: ProgressBody, listener: int | None = None):
    require_book(book_id)
    db.set_progress(book_id, body.chapter, max(0.0, body.offset), max(0.0, body.position), body.word,
                    listener_id=listeners.resolve(listener))
    return {"ok": True}


class CheckpointBody(ProgressBody):
    kind: str = "manual"  # manual | auto | chapter | sleep
    label: str | None = None


@app.get("/api/books/{book_id}/checkpoints")
def get_checkpoints(book_id: str, listener: int | None = None):
    require_book(book_id)
    return db.list_checkpoints(book_id, listeners.resolve(listener))


@app.post("/api/books/{book_id}/checkpoints")
def post_checkpoint(book_id: str, body: CheckpointBody, listener: int | None = None):
    require_book(book_id)
    if body.kind not in {"manual", "auto", "chapter", "sleep"}:
        raise HTTPException(400, "Unknown checkpoint kind")
    label = (body.label or "").strip()[:300] or None
    new_id = db.add_checkpoint(book_id, body.chapter, max(0.0, body.offset),
                               max(0.0, body.position), body.kind, label, body.word,
                               listener_id=listeners.resolve(listener))
    return {"id": new_id}


@app.delete("/api/books/{book_id}/checkpoints/{checkpoint_id}")
def delete_checkpoint(book_id: str, checkpoint_id: int, listener: int | None = None):
    require_book(book_id)
    db.delete_checkpoint(book_id, checkpoint_id, listeners.resolve(listener))
    return {"ok": True}


# --------------------------------------------------------------------------
# Publishing: a Forge with a GPU sends finished books to a server
# --------------------------------------------------------------------------

IMPORT_NAMES_RE = re.compile(r"^(?:source\.[\w.]+|cover(?:-auto)?\.jpg|readalong\.json|text/\d{4}\.txt|text"
                             r"|transcript/\d{4}\.json|transcript|parts\.json)$")


def _check_token(request: Request) -> None:
    if config.SHARE_TOKEN and request.headers.get(publish.TOKEN_HEADER) != config.SHARE_TOKEN:
        raise HTTPException(403, "This Forge doesn’t accept books from that one. Set the same SHARE_TOKEN on both.")


# --- Settings: GPU servers (app/gpus.py) -------------------------------------------
# Machines running the clone server, and which jobs each does: voices, pictures.
# Tokens go in but never come back out.

class GpuServerBody(BaseModel):
    name: str | None = None
    url: str | None = None
    token: str | None = None  # omitted: keep the saved one; "": none
    voices: bool | None = None
    pictures: bool | None = None


def _gpu_json(s: dict) -> dict:
    h = gpus.health(s)
    return {"id": s["id"], "name": s.get("name") or "", "url": s["url"], "source": s["source"],
            "voices": bool(s.get("voices")), "pictures": bool(s.get("pictures")), "has_token": bool(s.get("token")),
            "health": {"ok": bool(h.get("ok")), "error": h.get("error"), "gpu": h.get("gpu"), "gpu_gb": h.get("gpu_gb"),
                       "loaded": h.get("loaded"), "draws": bool(h.get("draw_models")),
                       "voices": [m["label"] for m in h.get("models") or [] if m.get("fits")]}}


def _gpu_url(url: str | None) -> str:
    url = (url or "").strip().rstrip("/")
    if not re.fullmatch(r"https?://[^\s/]+(:\d+)?(/\S*)?", url):
        raise HTTPException(400, "Give the address with http://, like http://192.168.1.30:8100.")
    return url


def _gpu_list() -> dict:
    return {"servers": [_gpu_json(s) for s in gpus.servers()], "file": str(gpus.FILE)}


@app.get("/api/gpu-servers")
def list_gpu_servers():
    return _gpu_list()


@app.post("/api/gpu-servers")
def add_gpu_server(body: GpuServerBody):
    url = _gpu_url(body.url)
    servers = gpus.load()
    if any(s["url"].rstrip("/") == url for s in servers):
        raise HTTPException(409, "That server is in the list already.")
    servers.append({"id": gpus.new_id(), "name": (body.name or "").strip()[:60] or urlparse(url).hostname,
                    "url": url, "token": (body.token or "").strip(),
                    "voices": body.voices is not False, "pictures": body.pictures is not False})
    gpus.save(servers)
    return _gpu_list()


def _settings_server(servers: list[dict], server_id: str) -> dict:
    found = next((s for s in servers if s["id"] == server_id), None)
    if found is None:
        raise HTTPException(404 if server_id != "env" else 400,
                            "That server is set in .env (CLONE_URL): change it there, or add it here instead."
                            if server_id == "env" else "No such server.")
    return found


@app.patch("/api/gpu-servers/{server_id}")
def change_gpu_server(server_id: str, body: GpuServerBody):
    servers = gpus.load()
    s = _settings_server(servers, server_id)
    if body.url is not None:
        s["url"] = _gpu_url(body.url)
    if body.name is not None:
        s["name"] = body.name.strip()[:60] or urlparse(s["url"]).hostname
    if body.token is not None:
        s["token"] = body.token.strip()
    for role in gpus.ROLES:
        if getattr(body, role) is not None:
            s[role] = getattr(body, role)
    gpus.save(servers)
    return _gpu_list()


@app.delete("/api/gpu-servers/{server_id}")
def remove_gpu_server(server_id: str):
    servers = gpus.load()
    s = _settings_server(servers, server_id)
    gpus.save([x for x in servers if x is not s])
    return _gpu_list()


@app.post("/api/gpu-servers/{server_id}/first")
def gpu_server_first(server_id: str):
    """Try this server first for its jobs."""
    servers = gpus.load()
    s = _settings_server(servers, server_id)
    gpus.save([s] + [x for x in servers if x is not s])
    return _gpu_list()


@app.post("/api/gpu-servers/test")
def test_gpu_server(body: GpuServerBody):
    """Before saving: does it answer, and does it take the token?"""
    from .clone import Client
    url = _gpu_url(body.url)
    h = gpus.health({"url": url}, max_age=0)
    out = {"ok": bool(h.get("ok")), "error": h.get("error"), "gpu": h.get("gpu"), "gpu_gb": h.get("gpu_gb"),
           "draws": bool(h.get("draw_models")), "token_ok": None}
    if out["ok"]:
        token = body.token
        if token is None:  # editing: the saved one
            token = next((s.get("token") for s in gpus.servers() if s["url"].rstrip("/") == url), "")
        # Any call but health needs the token: an unknown clip is 404 with it, 401 without.
        status, _ = Client(url, token or "", timeout=10)._call("GET", "/api/refs/" + "0" * 40)
        out["token_ok"] = status != 401
    return out


# Your voices on the server: the PC that makes them keeps this copy in step
# (publish.sync_voices), so they can be picked from any device.

CUSTOM_KINDS = ("onnx", "pt")  # a trained Piper voice, or a Kokoro voicepack


def _custom_voice_files(slug: str, kind: str = "onnx") -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-z0-9-]{1,48}", slug) or kind not in CUSTOM_KINDS:
        raise HTTPException(404, "Voice not found")
    model = config.CUSTOM_VOICES_DIR / f"{slug}.{kind}"
    return model, model.with_name(model.name + ".json")


@app.get("/api/custom-voices")
def list_custom_voices(request: Request):
    """Each voice's config fingerprint: it changes whenever the voice is retrained, renamed or measured."""
    import hashlib
    _check_token(request)
    out = {}
    for kind in CUSTOM_KINDS:
        for meta in config.CUSTOM_VOICES_DIR.glob(f"*.{kind}.json"):
            if meta.with_suffix("").exists():
                out[meta.with_suffix("").stem] = {"config": hashlib.sha1(meta.read_bytes()).hexdigest(), "kind": kind}
    return out


@app.post("/api/custom-voices/{slug}")
def put_custom_voice(slug: str, request: Request, voice_config: UploadFile = File(...), model: UploadFile = File(...)):
    _check_token(request)
    kind = "pt" if (model.filename or "").endswith(".pt") else "onnx"
    model_path, meta_path = _custom_voice_files(slug, kind)
    for other in CUSTOM_KINDS:  # a voice made the other way before: this one replaces it
        if other != kind:
            for path in _custom_voice_files(slug, other)[::-1]:
                path.unlink(missing_ok=True)
    data = voice_config.file.read()
    try:
        json.loads(data)
    except ValueError:
        raise HTTPException(400, "That isn’t a voice’s config.")
    config.CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    # The model first, then its config: this Forge lists a voice only once both are complete.
    tmp = model_path.with_name(f".{model_path.name}.part")
    with tmp.open("wb") as f:
        shutil.copyfileobj(model.file, f, 1 << 20)
    os.replace(tmp, model_path)
    tmp = meta_path.with_name(f".{meta_path.name}.part")
    tmp.write_bytes(data)
    os.replace(tmp, meta_path)
    return {"ok": True}


@app.delete("/api/custom-voices/{slug}")
def delete_custom_voice(slug: str, request: Request):
    _check_token(request)
    for kind in CUSTOM_KINDS:
        model_path, meta_path = _custom_voice_files(slug, kind)
        meta_path.unlink(missing_ok=True)  # the config first: it stops being listed before its model goes
        model_path.unlink(missing_ok=True)
    return {"ok": True}


# Cloned voices the same way: a folder each (voices.CLONED_FILES). Here they're
# voiced by the clone server too (CLONE_URL), or, for a book from the PC, there.

def _cloned_voice_dir(slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{1,48}", slug):
        raise HTTPException(404, "Voice not found")
    return config.CLONED_VOICES_DIR / slug


@app.get("/api/cloned-voices")
def list_cloned_voices(request: Request):
    _check_token(request)
    out = {}
    if config.CLONED_VOICES_DIR.is_dir():
        for folder in config.CLONED_VOICES_DIR.iterdir():
            if all((folder / name).exists() for name in voices.CLONED_FILES):
                out[folder.name] = {"hash": voices.cloned_fingerprint(folder)}
    return out


@app.post("/api/cloned-voices/{slug}")
def put_cloned_voice(slug: str, request: Request, voice: UploadFile = File(...),
                     ref_text: UploadFile = File(...), ref_audio: UploadFile = File(...)):
    _check_token(request)
    folder = _cloned_voice_dir(slug)
    meta = voice.file.read()
    try:
        json.loads(meta)
    except ValueError:
        raise HTTPException(400, "That isn’t a cloned voice’s settings.")
    folder.mkdir(parents=True, exist_ok=True)
    # The clip and its text first, voice.json last: this Forge lists a voice once that's there.
    for name, source in (("ref.wav", ref_audio.file), ("ref.txt", ref_text.file)):
        tmp = folder / f".{name}.part"
        with tmp.open("wb") as f:
            shutil.copyfileobj(source, f, 1 << 20)
        os.replace(tmp, folder / name)
    (folder / ".voice.json.part").write_bytes(meta)
    os.replace(folder / ".voice.json.part", folder / "voice.json")
    return {"ok": True}


@app.delete("/api/cloned-voices/{slug}")
def delete_cloned_voice(slug: str, request: Request):
    _check_token(request)
    folder = _cloned_voice_dir(slug)
    (folder / "voice.json").unlink(missing_ok=True)  # first: it stops being listed
    shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True}


VOICE_EXPORTS = {
    "piper": ("Piper voice",
              "{model} is the voice and {model}.json its settings: keep the two together.\n"
              "- In another Audiobook Forge: put both in its custom-voices folder (/models/custom-voices).\n"
              "  It lists the voice without a restart.\n"
              "- Piper (https://github.com/OHF-Voice/piper1-gpl) reads with it like any other voice:\n"
              "  give it the .onnx file as the model. It runs on any CPU."),
    "kokoro": ("Kokoro voice",
               "{model} is a voicepack for Kokoro-82M and {model}.json its settings: keep the two together.\n"
               "- In another Audiobook Forge: put both in its custom-voices folder (/models/custom-voices).\n"
               "  It lists the voice without a restart.\n"
               "- Kokoro (https://github.com/hexgrad/kokoro) loads the .pt file as a voice."),
    "clone": ("cloned voice",
              "A voice for Qwen3-TTS: ref.wav is a clip of the narrator, ref.txt the words said in it,\n"
              "and voice.json the model it was picked with.\n"
              "- In another Audiobook Forge: put the {model} folder in its cloned-voices folder\n"
              "  (/models/cloned-voices). It needs a GPU server, in Settings.\n"
              "- Qwen3-TTS (https://github.com/QwenLM/Qwen3-TTS) clones it from ref.wav and ref.txt."),
}


@app.get("/api/voices/{voice_id}/export")
def export_voice(voice_id: str):
    """A voice made with the voice creator as a zip: its files as Forge keeps them, and how to use them."""
    import tempfile
    import zipfile
    from starlette.background import BackgroundTask
    voice = voices.get(voice_id)
    if not voice or voice.get("group") != voices.CUSTOM_GROUP:
        raise HTTPException(404, "Only voices made with the voice creator can be exported.")
    if voice["engine"] == "clone":
        folder = Path(voice["ref_audio"]).parent
        model, files = folder.name, [(folder / n, f"{folder.name}/{n}") for n in voices.CLONED_FILES]
    else:
        path = Path(voice["model"] if voice["engine"] == "piper" else voice["voice"])
        model, files = path.name, [(path, path.name), (path.with_name(path.name + ".json"), path.name + ".json")]
    kind, how = VOICE_EXPORTS[voice["engine"]]
    readme = (f"{voice['narrator']}: a {kind} made with Audiobook Forge's voice creator.\n\n"
              + how.format(model=model)
              + f"\n\nIt was made from recordings of {voice['narrator']}. Share it only with their permission.\n")
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    try:
        with os.fdopen(fd, "wb") as f, zipfile.ZipFile(f, "w") as z:
            z.writestr("README.txt", readme, zipfile.ZIP_DEFLATED)
            for path, name in files:  # models hardly compress: stored as they are
                z.write(path, name, zipfile.ZIP_STORED if path.suffix in (".onnx", ".pt") else zipfile.ZIP_DEFLATED)
    except FileNotFoundError:
        os.unlink(tmp)
        raise HTTPException(404, "That voice was just removed or replaced. Try again.")
    return FileResponse(tmp, filename=f"{voice['narrator']} ({kind}).zip".replace("/", "-"),
                        media_type="application/zip", background=BackgroundTask(os.unlink, tmp))


@app.post("/api/import")
def import_book(request: Request, manifest: str = Form(...), pack: UploadFile = File(...),
                audio: UploadFile = File(...), replace: bool = False):
    """Store a book another Forge voiced: its audio where a local build would
    put it, plus text, cover and read-along map. Progress here is kept."""
    import tarfile
    from . import assemble
    _check_token(request)
    m = json.loads(manifest)
    book_id = str(m.get("id", ""))
    ext = str(m.get("ext", ""))
    if not re.fullmatch(r"[a-f0-9]{12}", book_id) or ext not in AUDIO_TYPES:
        raise HTTPException(400, "That isn’t a book this Forge can store.")
    existing = db.get_book(book_id)
    if existing and existing["status"] in BUSY | {"queued", "parsing"}:
        raise HTTPException(409, "This book is being voiced on the server right now. Send it again once that’s done.")
    audio_book = m.get("kind") == "audio"
    # An audiobook added for Read along goes into the library like a voiced one.
    out = assemble.output_path(m.get("meta") or {}, ext)
    clash = [b for b in db.list_books() if b["id"] != book_id and b["output"] == str(out)]
    if clash and not replace:
        other = clash[0]
        voice = voices.get(other["voice"])
        raise HTTPException(409, {
            "message": f"The server already has “{other['meta'].get('title') or other['filename']}” "
                       f"({voice['label'] if voice else other['voice']}) in the same place.",
            "conflict": {"id": other["id"], "title": other["meta"].get("title"),
                         "voice": voice["label"] if voice else other["voice"]}})
    for other in clash:  # replacing: the server's own copy of this book goes
        db.delete_book(other["id"])
        shutil.rmtree(config.BOOKS_DIR / other["id"], ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with tmp.open("wb") as f:
        shutil.copyfileobj(audio.file, f, 1 << 20)
    os.replace(tmp, out)
    if existing and existing["output"] and existing["output"] != str(out):
        Path(existing["output"]).unlink(missing_ok=True)  # renamed since last time
    folder = config.BOOKS_DIR / book_id
    folder.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(folder / "text", ignore_errors=True)
    with tarfile.open(fileobj=pack.file, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not IMPORT_NAMES_RE.match(member.name) or not (member.isfile() or member.isdir()):
                continue  # only the files a book has, never a path outside its folder
            tar.extract(member, folder, set_attrs=False)
    cover = folder / "cover.jpg"
    if cover.exists() and config.OUTPUT_LAYOUT == "abs":
        shutil.copyfile(cover, out.parent / "cover.jpg")
    db.import_book(m, str(out))
    try:
        versions.keep(book_id)  # every voice sent here stays, so switching back is instant
    except Exception:
        pass  # the book itself arrived fine
    _map_cache.pop(book_id, None)
    # SITE_URL: the address people use (https once set up), for the sender's links.
    return {"id": book_id, "read": f"/read/{book_id}", "site_url": config.SITE_URL or None}


@app.post("/api/books/{book_id}/publish")
def publish_book(book_id: str, replace: bool = False):
    """Send a finished book to the server (PUBLISH_URL)."""
    require_book(book_id)
    try:
        publish.send(book_id, replace=replace)
    except publish.PublishError as exc:
        if exc.conflict:
            raise HTTPException(409, {"message": str(exc), "conflict": exc.conflict})
        raise HTTPException(502, str(exc))
    return {"ok": True, "read_url": publish.read_url(db.get_book(book_id))}


def _origin_call(origin: str, method: str, path: str, body: dict | None = None, timeout: float = 10):
    """Ask the Forge that voiced a book to do something with it."""
    import urllib.request
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if config.SHARE_TOKEN:
        headers[publish.TOKEN_HEADER] = config.SHARE_TOKEN
    req = urllib.request.Request(origin + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get("content-type", "").startswith("application/json") else raw


class RevoiceBody(BaseModel):
    voice: str
    mood: str | None = None
    here: bool = False  # voice it on this Forge even when another one voiced it


@app.post("/api/books/{book_id}/revoice")
def revoice(book_id: str, body: RevoiceBody):
    """Voice a finished book again in another voice. A book that came from a
    Forge with a GPU is voiced there and sent back; the old audio keeps
    playing until the new one arrives."""
    book = require_book(book_id)
    if book["kind"] == "audio":
        raise HTTPException(400, "This is an audiobook you added, so Forge can’t change its voice.")
    voice = voices.get(body.voice)
    if voice is None:
        raise HTTPException(400, "Unknown voice")
    mood = voices.pick_mood(voice, body.mood)
    patch = {"voice": body.voice, **({"mood": mood} if mood else {})}
    label = voice["label"] + (f", {mood}" if mood else "")
    if book["status"] == "done" and versions.usable(book_id, versions.version_id(body.voice, mood, book["speed"])):
        apply_patch(book, BookPatch(**patch))  # switches to the copy already made in that voice
        return {"where": "ready"}
    if book["origin"] and not body.here:
        host, sent = _voice_at_origin(book, patch)
        if not sent:
            db.update_book(book_id, message=f"Voicing again in {label} on {host}")
        return {"where": host}
    if book["status"] not in STARTABLE:
        raise HTTPException(409, "This book is busy. Try again when it’s done.")
    apply_patch(book, BookPatch(**patch))
    db.update_book(book_id, status="queued", progress=0, message=f"Voicing again in {label}")
    return {"where": "here"}


def _voice_at_origin(book: dict, patch: dict, texts: dict[int, str] | None = None) -> tuple[str, bool]:
    """Have the Forge that voiced a book voice it again with these settings (and
    chapter texts). It sends the new copy here when it's done, replacing this one.
    Returns its host, and whether the copy was sent already (it had made that voice)."""
    host = _host(book["origin"])
    try:
        _origin_call(book["origin"], "PATCH", f"/api/books/{book['id']}", patch)
        for idx, text in (texts or {}).items():
            _origin_call(book["origin"], "PUT", f"/api/books/{book['id']}/chapters/{idx}/text", {"text": text})
        theirs = _origin_call(book["origin"], "GET", f"/api/books/{book['id']}")
        if theirs.get("status") == "done" and not theirs.get("changed"):
            # It had made this voice before and switched to it at once: it just sends it.
            _origin_call(book["origin"], "POST", f"/api/books/{book['id']}/publish", timeout=900)
            return host, True
        started = _origin_call(book["origin"], "POST", "/api/start", {"ids": [book["id"]]})
    except urllib.error.HTTPError as exc:  # it answered, and said no
        try:
            reason = json.loads(exc.read()).get("detail")
        except Exception:
            reason = None
        if exc.code == 404:
            reason = f"{host} no longer has this book, so it can’t voice it again. Voice it here instead."
        raise HTTPException(409, str(reason or f"{host} couldn’t start ({exc.code})."))
    except Exception:
        raise HTTPException(503, {
            "message": f"The PC that voices books ({host}) didn’t answer. Turn it on, "
                       "or voice it here on the server instead (much slower).",
            "offline": True})
    if not started.get("started"):
        raise HTTPException(409, f"{host} is already working on this book. Try again when it’s done.")
    db.update_book(book["id"], forwarded=time.time())
    return host, False


def _texts_changed_here(book: dict) -> dict[int, str]:
    """Chapter texts edited on this copy, which the PC needs before voicing again."""
    changed = {}
    for c in db.chapters(book["id"]):
        path = book_dir(book["id"]) / "text" / f"{c['idx']:04d}.txt"
        if not path.exists():
            continue
        mine = path.read_text(encoding="utf-8")
        theirs = _origin_call(book["origin"], "GET", f"/api/books/{book['id']}/chapters/{c['idx']}/text")
        if mine != theirs.get("text"):
            changed[c["idx"]] = mine
    return changed


@app.post("/api/books/{book_id}/rebuild")
def rebuild(book_id: str, here: bool = False):
    """Make a finished book's file again with the changes made to it. A copy
    voiced on another Forge goes back there, with the changes made here."""
    book = require_book(book_id)
    if book["origin"] and not here and book["kind"] != "audio":  # an added audiobook's text is made here
        chapters = db.chapters(book_id)
        patch = {k: book[k] for k in ("voice", "mood", "speed", "fmt")}
        patch["meta"] = book["meta"]
        patch["chapters"] = [{"idx": c["idx"], "title": c["title"], "include": bool(c["include"])} for c in chapters]
        try:
            texts = _texts_changed_here(book) if book["edited"] else {}
        except Exception:
            texts = {}  # _voice_at_origin reports the PC being off
        host, sent = _voice_at_origin(book, patch, texts)
        if not sent:
            db.update_book(book_id, message=f"Voicing again on {host}. This copy is replaced when it’s done.")
        return {"where": host, "sent": sent}
    if book["status"] not in STARTABLE:
        raise HTTPException(409, "This book is busy. Try again when it’s done.")
    if not any(c["include"] for c in db.chapters(book_id)):
        raise HTTPException(400, "Turn on at least one chapter to rebuild this book.")
    db.update_book(book_id, status="queued", progress=0, message="Waiting for a worker")
    return {"where": "here"}


@app.get("/api/books/{book_id}/revoice")
def revoice_status(book_id: str):
    """How a voice change is going, wherever it's being voiced."""
    book = require_book(book_id)
    if book["origin"]:
        try:
            r = _origin_call(book["origin"], "GET", f"/api/books/{book_id}", timeout=5)
            return {"where": _host(book["origin"]), "status": r["status"], "progress": r["progress"],
                    "voice": r["voice"], "mood": r["mood"], "message": r["message"]}
        except Exception:
            return {"where": _host(book["origin"]), "status": "unreachable"}
    return {"where": "here", "status": book["status"], "progress": book["progress"],
            "voice": book["voice"], "mood": book["mood"], "message": book["message"]}


# Seconds; covers a first-time voice download on a busy worker, or a voice in
# training pausing so the sample goes first (it saves its place, then the voices start).
SAMPLE_WAIT = 180


# Where a sample being made is (db.note_samples), as said under its play button.
SAMPLE_STAGES = {
    "queued": "Waiting for a voice worker",
    "gpu": "Waiting for the voice creator to pause",
    "starting": "Starting the voices",
    "voicing": "Making the sample",
}


def queue_preview(voice: dict, speed: float, items: list, path: Path, limit: float | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return db.add_sample({"voice": voice, "speed": speed, "items": items, "out": str(path), "limit": limit})


def synth_preview(voice: dict, speed: float, items: list, path: Path,
                  limit: float | None = None) -> None:
    """Have the worker voice a short clip, cached at `path`.

    The worker's processes already have voices loaded (on the GPU when there is
    one), so the web app never loads a model of its own. The clip goes ahead of
    any book parts waiting for a worker."""
    if path.exists():
        return
    sample_id = queue_preview(voice, speed, items, path, limit)
    deadline = time.monotonic() + SAMPLE_WAIT
    while time.monotonic() < deadline:
        row = db.get_sample(sample_id)
        if row and row["status"] == "failed":
            raise RuntimeError(row["message"] or "the voice worker couldn’t make it")
        if path.exists():
            return
        time.sleep(0.2)
    raise RuntimeError("the voice worker didn’t answer. Check that the worker container is running")


def preview_progress(voice: dict, speed: float, items: list, path: Path, limit: float | None = None):
    """For a progress bar (progress=1): the clip once it's made; until then it's
    started (or joined) and the answer is 202 with how far along it is. The page
    asks again until the clip comes back."""
    sample_id = db.sample_key(str(path))
    row = db.get_sample(sample_id)
    if path.exists():
        return FileResponse(path, media_type="audio/wav")
    if row and row["status"] == "failed":
        db.forget_sample(sample_id)  # said once: asking again tries again
        raise HTTPException(503, f"Couldn’t make a sample with this voice: "
                                 f"{row['message'] or 'the voice worker couldn’t make it'}")
    queue_preview(voice, speed, items, path, limit)
    stage = (db.get_sample(sample_id) or {}).get("stage") or "queued"
    return JSONResponse({"stage": stage, "message": SAMPLE_STAGES.get(stage, SAMPLE_STAGES["queued"])},
                        status_code=202)


def _seconds(value: float | None) -> float:
    """Sample length: 3 seconds for comparing voices, up to the normal 15."""
    return round(min(max(value or SAMPLE_SECONDS, 2.0), SAMPLE_SECONDS), 1)


@app.get("/api/preview")
def preview(voice: str, speed: float = 1.0, mood: str | None = None, seconds: float | None = None,
            progress: bool = False):
    v = voices.get(voice)
    if v is None:
        raise HTTPException(404, "Unknown voice")
    v = voices.with_mood(v, mood)
    speed = round(min(max(speed, 0.5), 2.0), 2)
    seconds = _seconds(seconds)
    # At speed 1 and previews.SAVED_SECONDS, this is the voice's saved sample, made ahead of time.
    sample = previews.job(v, speed, seconds)
    items, path = sample["items"], Path(sample["out"])
    if progress:
        return preview_progress(v, speed, items, path, limit=seconds)
    if not path.exists():
        try:
            synth_preview(v, speed, items, path, limit=seconds)
        except Exception as exc:
            raise HTTPException(503, f"Couldn’t make a sample with this voice: {exc}")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/books/{book_id}/sample")
def book_sample(book_id: str, voice: str | None = None, speed: float | None = None,
                mood: str | None = None, seconds: float | None = None, progress: bool = False):
    """This book's opening as it would be narrated: 15 seconds, or as short as 2
    for comparing voices. progress: don't wait, say how far along it is (202)."""
    seconds = _seconds(seconds)
    book = require_book(book_id)
    v = voices.get(voice or book["voice"])
    if v is None:
        raise HTTPException(404, "Unknown voice")
    v = voices.with_mood(v, mood or book["mood"])
    speed = round(min(max(speed or book["speed"] or 1.0, 0.5), 2.0), 2)
    chapters = [c for c in db.chapters(book_id) if c["include"]]
    if not chapters:
        if book["status"] == "parsing":
            raise HTTPException(409, "This book is still being read. Try the sample again in a moment.")
        raise HTTPException(409, "Turn on at least one chapter to hear a sample.")
    chapter = next((c for c in chapters if c["words"] >= 60), chapters[0])
    text_path = book_dir(book_id) / "text" / f"{chapter['idx']:04d}.txt"
    if not text_path.exists():
        raise HTTPException(404, "Chapter not found")
    # A little extra text, then cut to length, so the clip is always the full length.
    # A short clip skips the chapter title, so all 3 seconds are the narration itself.
    items = textprep.sample_items(chapter["title"] if seconds >= 8 else None,
                                  text_path.read_text(encoding="utf-8"), seconds * 1.4, speed)
    path = book_dir(book_id) / "samples" / f"{textprep.unit_key(v['key'], speed, items)}-{seconds}.wav"
    here = db.get_sample(db.sample_key(str(path)))
    if not path.exists() and book["origin"] and not (here and here["status"] in ("pending", "working")):
        # Voiced on a Forge with a GPU: ask it for the clip, it's much faster.
        # (Unless it's already being made here, after that Forge couldn't.)
        query = urlencode({k: x for k, x in (("voice", v["id"]), ("mood", v.get("mood")), ("speed", speed),
                                              ("seconds", seconds), ("progress", 1 if progress else None)) if x})
        try:
            data = _origin_call(book["origin"], "GET", f"/api/books/{book_id}/sample?{query}",
                                timeout=10 if progress else 90)
            if isinstance(data, bytes) and data[:4] == b"RIFF":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_suffix(".part").write_bytes(data)
                path.with_suffix(".part").replace(path)
            elif progress and isinstance(data, dict) and data.get("stage"):
                return JSONResponse(data, status_code=202)  # being made there: how far along
        except Exception:
            pass  # not there or not on: make it here
    if progress:
        return preview_progress(v, speed, items, path, limit=seconds)
    if not path.exists():
        try:
            synth_preview(v, speed, items, path, limit=seconds)
        except Exception as exc:
            raise HTTPException(503, f"Couldn’t make a sample with this voice: {exc}")
    return FileResponse(path, media_type="audio/wav")


# --------------------------------------------------------------------------
# Voice creator: /createvoice is passed through to VOICELAB_URL
# --------------------------------------------------------------------------

# Headers that belong to one connection, not to the request being passed on.
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
               "trailer", "trailers", "transfer-encoding", "upgrade", "host"}
_lab_client = None


def _lab_http():
    global _lab_client
    if _lab_client is None:
        import httpx
        # No read or write limit: uploads of whole audiobooks and live progress take
        # long. A short connect limit: a PC that's off should get its page quickly.
        _lab_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=None, write=None, pool=30.0))
    return _lab_client


def _lab_is_local() -> bool:
    """The voice creator runs next to this Forge (compose service or localhost),
    rather than on another machine reached through its Forge."""
    host = urlparse(config.VOICELAB_URL).hostname or ""
    return host == "localhost" or ("." not in host and ":" not in host) or host.startswith("127.")


def _lab_offline(request: Request) -> Response:
    if _lab_is_local():
        title = "The voice creator isn’t running right now"
        text = ("Start it on this machine (docker compose with docker-compose.voice.yml), "
                "then try again. Your books aren’t affected.")
    else:
        title = "The PC that trains voices is off right now"
        text = (f"Voices are made on the PC with the graphics card ({_host(config.VOICELAB_URL)}), "
                "and it isn’t answering. Turn it on or wake it up, then try again. "
                "Your books and your place in them aren’t affected.")
    if "text/html" not in request.headers.get("accept", ""):
        return Response(json.dumps({"detail": f"{title}. {text}", "offline": True}), status_code=503,
                        media_type="application/json")
    page = (config.APP_DIR / "static" / "createvoice-offline.html").read_text(encoding="utf-8")
    page = page.replace("{{title}}", title).replace("{{text}}", text)
    return Response(page, status_code=503, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.api_route("/createvoice", methods=["GET", "HEAD"])
@app.api_route("/createvoice/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"])
async def createvoice(request: Request, rest: str = ""):
    """Pass the voice creator through, streaming both ways: uploads can be
    whole audiobooks, and its progress arrives as a live event stream."""
    import httpx
    from starlette.background import BackgroundTask
    if not config.VOICELAB_URL:
        raise HTTPException(404, "The voice creator isn’t set up on this Forge. Set VOICELAB_URL to turn it on.")
    url = config.VOICELAB_URL + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
    client_ip = request.client.host if request.client else ""
    headers["x-forwarded-for"] = ", ".join(x for x in (request.headers.get("x-forwarded-for"), client_ip) if x)
    headers["x-forwarded-host"] = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    headers["x-forwarded-proto"] = request.headers.get("x-forwarded-proto") or request.url.scheme
    has_body = request.method in {"POST", "PUT", "PATCH", "DELETE"}
    client = _lab_http()
    try:
        upstream = await client.send(
            client.build_request(request.method, url, headers=headers,
                                 content=request.stream() if has_body else None),
            stream=True)
    except httpx.TransportError:
        return _lab_offline(request)
    response = StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                                 background=BackgroundTask(upstream.aclose))
    # Passed on as they came (raw bytes, so Content-Length and Content-Encoding still fit).
    response.raw_headers = [
        (k.lower().encode("latin-1"),
         (v.replace(config.VOICELAB_URL, "", 1) if k.lower() == "location" else v).encode("latin-1"))
        for k, v in upstream.headers.multi_items() if k.lower() not in HOP_HEADERS]
    return response


@app.get("/api/events")
async def events(request: Request, listener: int | None = None):
    async def stream():
        last, idle = None, 0
        while not await request.is_disconnected():
            stamp = await asyncio.to_thread(db.stamp)
            if stamp != last:
                last, idle = stamp, 0
                data = await asyncio.to_thread(all_books_json, listener)
                yield f"event: books\ndata: {json.dumps(data)}\n\n"
            else:
                idle += 1
                if idle % 15 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(config.APP_DIR / "static" / "index.html")
