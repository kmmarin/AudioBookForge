"""Free books from Project Gutenberg, kept within its terms of use
(https://www.gutenberg.org/policy/terms_of_use.html):

- Searching never touches their site. Their catalog, which they put in the
  public domain and ask tools to use "instead of crawling or roboting the
  website", is downloaded once a week and searched here.
- A book is fetched only when someone presses Add on it, one at a time, the way
  a browser would, with a user-agent that says who we are and how to reach us.
  At most DAILY_LIMIT a day: they ask for more than about 100 a day to come
  from a mirror.
- Before fetching, the book's own catalog record must say "Public domain in the
  USA." (a few of their books are still under copyright).
- The book arrives as they publish it, its Project Gutenberg license included.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
import urllib.request
from pathlib import Path

from . import config

log = logging.getLogger("forge.gutenberg")
SITE = "https://www.gutenberg.org"
CATALOG_URL = f"{SITE}/cache/epub/feeds/pg_catalog.csv"
USER_AGENT = "AudiobookForge/1.0 (+https://github.com/kmmarin/AudioBookForge)"
FOLDER = config.DATA_DIR / "gutenberg"
CATALOG = FOLDER / "pg_catalog.csv"
REFRESH_SECONDS = 7 * 86400  # they regenerate it weekly
DAILY_LIMIT = 50
MAX_BYTES = 200 << 20
PUBLIC_DOMAIN = "Public domain in the USA."

_lock = threading.Lock()  # one fetch at a time
_state = {"downloading": False, "error": None}
_rows: list[tuple] = []  # (id, title, authors, language, subjects, search text)
_loaded_mtime = 0.0


class GutenbergError(Exception):
    pass


def _get(url: str, timeout: float = 60, limit: int = MAX_BYTES) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise GutenbergError("That file is too big.")
    return data


# ---------------------------------------------------------------- the catalog

def _download_catalog() -> None:
    try:
        FOLDER.mkdir(parents=True, exist_ok=True)
        tmp = CATALOG.with_suffix(".part")
        tmp.write_bytes(_get(CATALOG_URL, timeout=300, limit=200 << 20))
        tmp.replace(CATALOG)
        _state["error"] = None
        log.info("Project Gutenberg catalog updated")
    except Exception as exc:
        _state["error"] = f"Couldn’t download Project Gutenberg’s catalog: {exc}"
        log.warning(_state["error"])
    finally:
        _state["downloading"] = False


def refresh() -> None:
    """Fetch the catalog when it's missing or a week old (in the background)."""
    fresh = CATALOG.exists() and time.time() - CATALOG.stat().st_mtime < REFRESH_SECONDS
    if fresh or _state["downloading"]:
        return
    _state["downloading"] = True
    threading.Thread(target=_download_catalog, daemon=True).start()


def _load() -> list[tuple]:
    global _rows, _loaded_mtime
    mtime = CATALOG.stat().st_mtime
    if mtime != _loaded_mtime:
        rows = []
        with CATALOG.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("Type") != "Text" or not r.get("Text#", "").isdigit():
                    continue
                lines = [" ".join(line.split()) for line in (r.get("Title") or "").splitlines()]
                title = ": ".join(line for line in lines if line)  # "Title: subtitle"
                authors = r.get("Authors") or ""
                subjects = f"{r.get('Subjects') or ''}; {r.get('Bookshelves') or ''}"
                rows.append((int(r["Text#"]), title, authors, r.get("Language") or "", subjects,
                             f"{title} {authors}".lower()))
        _rows, _loaded_mtime = rows, mtime
    return _rows


def state() -> dict:
    ready = CATALOG.exists()
    return {"ready": ready, "downloading": _state["downloading"], "error": _state["error"],
            "updated": int(CATALOG.stat().st_mtime) if ready else None,
            "books": len(_load()) if ready else 0}


def _author_name(authors: str) -> str:
    """"Twain, Mark, 1835-1910; Kemble, E. W." -> "Mark Twain"."""
    first = authors.split(";")[0].strip()
    first = re.sub(r",?\s*(\d{3,4}\??-\d{0,4}\??|\d{3,4}\??-?|BCE?.*|active .*)$", "", first).strip(" ,")
    if "," in first:
        last, rest = first.split(",", 1)
        rest = re.sub(r"\s*\(.*?\)", "", rest).strip()
        first = f"{rest} {last}".strip()
    return first


def search(query: str, language: str = "en", limit: int = 40) -> list[dict]:
    words = [w for w in re.split(r"\W+", query.lower()) if w]
    if not words or not CATALOG.exists():
        return []
    found = []
    for pg_id, title, authors, lang, subjects, text in _load():
        if language and language not in lang.split("; "):
            continue
        if not all(w in text for w in words):
            continue
        t = title.lower()
        score = sum(3 if w in t else 1 for w in words) + (5 if t.startswith(query.lower().strip()) else 0)
        found.append((score, -pg_id, pg_id, title, authors, lang, subjects))
    found.sort(reverse=True)
    return [{"id": pg_id, "title": title, "author": _author_name(authors), "authors": authors,
             "language": lang, "subjects": subjects.strip("; ")[:200],
             "url": f"{SITE}/ebooks/{pg_id}"}
            for _, _, pg_id, title, authors, lang, subjects in found[:limit]]


def book(pg_id: int) -> dict | None:
    for row in _load() if CATALOG.exists() else []:
        if row[0] == pg_id:
            return {"id": pg_id, "title": row[1], "author": _author_name(row[2])}
    return None


# ------------------------------------------------------------------- one book

def _count_today() -> tuple[Path, dict]:
    path = FOLDER / "today.json"
    day = time.strftime("%Y-%m-%d")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    return path, data if data.get("day") == day else {"day": day, "count": 0}


def fetch(pg_id: int) -> tuple[str, bytes]:
    """The book's EPUB as Project Gutenberg publishes it, and a file name for it."""
    info = book(pg_id)
    if info is None:
        raise GutenbergError("That book isn’t in Project Gutenberg’s catalog.")
    if not _lock.acquire(blocking=False):
        raise GutenbergError("Another book is being fetched. Try again in a moment.")
    try:
        path, today = _count_today()
        if today["count"] >= DAILY_LIMIT:
            raise GutenbergError(f"That’s {DAILY_LIMIT} books from Project Gutenberg today, "
                                 "the most Forge fetches in a day. Try again tomorrow.")
        rdf = _get(f"{SITE}/cache/epub/{pg_id}/pg{pg_id}.rdf", timeout=30, limit=5 << 20).decode("utf-8", "replace")
        rights = re.search(r"<dcterms:rights>(.*?)</dcterms:rights>", rdf, re.S)
        if not rights or rights.group(1).strip() != PUBLIC_DOMAIN:
            raise GutenbergError("Project Gutenberg doesn’t list this book as public domain in the USA, "
                                 "so Forge won’t fetch it.")
        data = _get(f"{SITE}/ebooks/{pg_id}.epub3.images", timeout=120)
        if not data.startswith(b"PK"):
            raise GutenbergError("Project Gutenberg didn’t send an EPUB for this book.")
        today["count"] += 1
        FOLDER.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(today))
        short = info["title"].split(": ")[0]
        name = f"{short} - {info['author']}" if info["author"] else short
        name = re.sub(r'[\\/:*?"<>|]+', "", name).strip()[:120] or f"Gutenberg {pg_id}"
        return f"{name}.epub", data
    except GutenbergError:
        raise
    except Exception as exc:
        raise GutenbergError(f"Couldn’t fetch the book from Project Gutenberg: {exc}")
    finally:
        _lock.release()
