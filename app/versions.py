"""Finished copies of a book in every voice it has been made in.

The book's own files are always its current voice: its file in the library,
readalong.json and the chapter lengths. Every finished build is also kept
under /data/books/<id>/voices/<version id>/ (audio and read-along map), so
picking a voice the book was already made in switches to it in seconds, with
no voicing: the kept file is copied into the library with today's details,
and Read along changes voice on the same word.

A version is one voice, mood and speed. Editing the text makes the kept
copies outdated (their words no longer match); they can still be downloaded
or removed, but not switched to.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

from . import assemble, config, covers, db, readalong, voices

log = logging.getLogger("forge.versions")


def version_id(voice: str | None, mood: str | None, speed: float | None) -> str:
    key = f"{voice}|{mood or ''}|{float(speed or 1.0):g}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def folder(book_id: str, vid: str) -> Path:
    return config.BOOKS_DIR / book_id / "voices" / vid


def current_id(book: dict) -> str | None:
    """The version the book's current file is: the one it was built with."""
    if book["status"] != "done" or not book.get("built"):
        return None
    built = json.loads(book["built"])
    return version_id(built.get("voice"), built.get("mood"), built.get("speed")) if built.get("voice") else None


def label(v: dict) -> str:
    """The name to show: the user's, else the voice's own (and its mood and speed)."""
    if v.get("name"):
        return v["name"]
    voice = voices.get(v["voice"])
    text = voice.get("label", v["voice"]) if voice else voices.gone_name(v["voice"])
    mood = next((m for m in (voice or {}).get("moods", []) if m["id"] == v.get("mood")), None)
    if mood:
        text += f", {mood.get('label', mood['id']).lower()}"
    if float(v.get("speed") or 1.0) != 1.0:
        text += f", {float(v['speed']):g}×"
    return text


def _copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def keep(book_id: str) -> str | None:
    """File a finished book's current file and map under the voice it was made in."""
    book = db.get_book(book_id)
    vid = current_id(book) if book else None
    if vid is None or not book["output"] or not Path(book["output"]).exists():
        return None
    built = json.loads(book["built"])
    src = Path(book["output"])
    dest = folder(book_id, vid)
    dest.mkdir(parents=True, exist_ok=True)
    audio = dest / f"audio{src.suffix}"
    _copy(src, audio)
    for old in dest.glob("audio.*"):
        if old != audio:
            old.unlink(missing_ok=True)
    ra = readalong.path_for(book_id)
    if ra.exists():
        _copy(ra, dest / "readalong.json")
    else:
        (dest / "readalong.json").unlink(missing_ok=True)
    chapters = [{"idx": c["idx"], "duration": c["duration"], "wer": c["wer"]}
                for c in db.chapters(book_id) if c["include"]]
    db.save_version({
        "book_id": book_id, "id": vid, "voice": built["voice"], "mood": built.get("mood"),
        "speed": built.get("speed") or 1.0, "fmt": built.get("fmt") or book["fmt"], "file": str(audio),
        "duration": round(sum(c["duration"] or 0 for c in chapters), 2), "rtf": book["rtf"],
        "chapters": json.dumps(chapters),
        "outdated": int(bool(book["edited"])),  # text edited after this file was made
        "created": time.time()})
    return vid


def keep_all() -> None:
    """Start-up: file away the current copy of books finished before versions were kept."""
    for book in db.books_by_status("done"):
        vid = current_id(book)
        if vid and db.get_version(book["id"], vid) is None:
            try:
                keep(book["id"])
            except Exception:
                log.exception("Couldn’t keep the current copy of %s", book["id"])


def usable(book_id: str, vid: str) -> dict | None:
    v = db.get_version(book_id, vid)
    return v if v and not v["outdated"] and Path(v["file"]).exists() else None


def use(book: dict, vid: str) -> bool:
    """Make a kept copy the book's current one: into the library with today's
    details and cover; its read-along map and chapter lengths take over."""
    v = usable(book["id"], vid)
    if v is None:
        return False
    fmt = v["fmt"] if v["fmt"] in config.FORMATS else book["fmt"]
    chapters = json.loads(v["chapters"])
    titles = {c["idx"]: c["title"] for c in db.chapters(book["id"])}
    workdir = config.BOOKS_DIR / book["id"]
    cover = workdir / "cover.jpg"
    if not cover.exists():
        cover = workdir / "cover-auto.jpg"
        if not cover.exists():
            cover.write_bytes(covers.generate(book["meta"].get("title", ""), book["meta"].get("author", "")))
    voice = voices.get(v["voice"])
    narrator = voices.narrator(voices.with_mood(voice, v["mood"])) if voice else v["voice"]
    out = assemble.output_path(book["meta"], config.FORMATS[fmt]["ext"])
    assemble.retag(Path(v["file"]), out, book["meta"], narrator,
                   [(titles.get(c["idx"], ""), c["duration"] or 0) for c in chapters], fmt, cover)
    if book["output"] and book["output"] != str(out):
        Path(book["output"]).unlink(missing_ok=True)
    ra = folder(book["id"], vid) / "readalong.json"
    if ra.exists():
        _copy(ra, readalong.path_for(book["id"]))
    else:
        readalong.path_for(book["id"]).unlink(missing_ok=True)
    db.update_chapters(book["id"], [{"idx": c["idx"], "duration": c["duration"], "wer": c["wer"], "state": "done"}
                                    for c in chapters])
    fields = {"voice": v["voice"], "mood": v["mood"], "speed": v["speed"], "fmt": fmt}
    # Kept copies are never outdated, so anything still marked as edited was
    # only details or the cover, and retag just applied those.
    db.update_book(book["id"], **fields, status="done", progress=1.0, output=str(out), rtf=v["rtf"],
                   built=db.built_with(fields), edited=0, message=f"Now in {label(v)}")
    log.info("%s now in %s (kept copy)", book["id"], label(v))
    return True


def remove(book: dict, vid: str) -> str:
    """Delete a kept copy. Removing the current one switches to the newest other
    copy; with none left, the book goes back to ready to voice.
    Returns "removed", "switched" or "emptied"."""
    was_current = current_id(book) == vid
    shutil.rmtree(folder(book["id"], vid), ignore_errors=True)
    db.delete_version(book["id"], vid)
    if not was_current:
        return "removed"
    for other in db.versions(book["id"]):
        if use(book, other["id"]):
            return "switched"
    if book["output"]:
        assemble.remove_output(Path(book["output"]))
    readalong.path_for(book["id"]).unlink(missing_ok=True)
    db.update_chapters(book["id"], [{"idx": c["idx"], "duration": None, "wer": None, "units_done": 0,
                                     "state": "pending"} for c in db.chapters(book["id"])])
    db.update_book(book["id"], status="ready", progress=0, output=None, built=None, edited=0,
                   message="Its voiced copy was removed. Pick a voice and start to make it again.")
    return "emptied"


def public(book: dict, rows: list[dict]) -> list[dict]:
    current = current_id(book)
    out = []
    for v in rows:
        path = Path(v["file"])
        out.append({"id": v["id"], "voice": v["voice"], "mood": v["mood"], "speed": v["speed"], "fmt": v["fmt"],
                    "name": v["name"], "label": label(v), "voice_label": label({**v, "name": None}),
                    "duration": v["duration"],
                    "outdated": bool(v["outdated"]), "current": v["id"] == current, "created": v["created"],
                    "size": path.stat().st_size if path.exists() else None})
    return out
