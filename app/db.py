"""SQLite state shared by the web app and the worker container."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    voice TEXT,
    mood TEXT,
    speed REAL NOT NULL DEFAULT 1.0,
    fmt TEXT NOT NULL DEFAULT 'm4b',
    qa INTEGER NOT NULL DEFAULT 1,
    progress REAL NOT NULL DEFAULT 0,
    rtf REAL,
    message TEXT,
    output TEXT,
    has_cover INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chapters (
    book_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    title TEXT NOT NULL,
    include INTEGER NOT NULL DEFAULT 1,
    words INTEGER NOT NULL DEFAULT 0,
    units_total INTEGER NOT NULL DEFAULT 0,
    units_done INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'pending',
    duration REAL,
    wer REAL,
    PRIMARY KEY (book_id, idx)
);
CREATE INDEX IF NOT EXISTS books_status ON books(status);
CREATE TABLE IF NOT EXISTS listeners (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    book_id TEXT NOT NULL,
    listener_id INTEGER NOT NULL,
    chapter INTEGER NOT NULL,
    offset REAL NOT NULL,
    position REAL NOT NULL,
    word INTEGER,
    updated REAL NOT NULL,
    PRIMARY KEY (book_id, listener_id)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    offset REAL NOT NULL,
    position REAL NOT NULL,
    kind TEXT NOT NULL,
    label TEXT,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS checkpoints_book ON checkpoints(book_id, created);
CREATE TABLE IF NOT EXISTS versions (
    book_id TEXT NOT NULL,
    id TEXT NOT NULL,
    voice TEXT NOT NULL,
    mood TEXT,
    speed REAL NOT NULL DEFAULT 1.0,
    fmt TEXT NOT NULL,
    name TEXT,
    file TEXT NOT NULL,
    duration REAL,
    rtf REAL,
    chapters TEXT NOT NULL DEFAULT '[]',
    outdated INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    PRIMARY KEY (book_id, id)
);
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    job TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created REAL NOT NULL
);
"""

BOOK_COLUMNS = {"filename", "status", "meta", "voice", "mood", "speed", "fmt", "qa", "progress", "rtf", "published", "origin",
                "message", "output", "has_cover", "built", "edited", "forwarded", "kind"}
CHAPTER_COLUMNS = {"title", "include", "words", "units_total", "units_done", "state",
                   "duration", "wer", "start"}
# What a finished file was made with: changing any of these calls for a rebuild.
BUILT_KEYS = ("voice", "mood", "speed", "fmt")


def built_with(book: dict) -> str:
    return json.dumps({k: book.get(k) for k in BUILT_KEYS})

_local = threading.local()


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=30000")
        _local.conn = c
    return c


@contextmanager
def transaction():
    c = conn()
    c.execute("BEGIN IMMEDIATE")
    try:
        yield c
        c.execute("COMMIT")
    except BaseException:
        c.execute("ROLLBACK")
        raise


def init() -> None:
    c = conn()
    c.executescript(SCHEMA)
    # Columns added after the first release; ALTER is safe to race between containers.
    # kind: 'tts' for a book Forge voices, 'audio' for an audiobook added to get its text.
    # start: where an audiobook's chapter begins in its file (seconds).
    added = {"books": (("mood", "TEXT"), ("rtf", "REAL"), ("published", "REAL"), ("origin", "TEXT"),
                       ("built", "TEXT"), ("edited", "INTEGER"), ("forwarded", "REAL"),
                       ("kind", "TEXT NOT NULL DEFAULT 'tts'")),
             "chapters": (("start", "REAL"),),
             "progress": (("word", "INTEGER"),),
             "samples": (("stage", "TEXT"),),
             "checkpoints": (("word", "INTEGER"), ("listener_id", "INTEGER"))}
    for table, columns in added.items():
        have = {row["name"] for row in c.execute(f"PRAGMA table_info({table})")}
        for column, kind in columns:
            if column in have:
                continue
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc):
                    raise
    # Books finished before builds recorded their settings were built with what
    # they have now, unless they were changed since (said so in their message).
    # (An added audiobook wasn't built with a voice: it has none.)
    for row in c.execute("SELECT id, voice, mood, speed, fmt FROM books WHERE built IS NULL AND status = 'done'"
                         " AND kind != 'audio' AND COALESCE(message, '') NOT LIKE '%Rebuild to apply%'").fetchall():
        c.execute("UPDATE books SET built = ? WHERE id = ? AND built IS NULL", (built_with(dict(row)), row["id"]))
    _migrate_listeners()


def _migrate_listeners() -> None:
    """From one place per book to one per (book, listener).

    Everything saved before listeners existed goes to the first listener, "Me",
    created here. The progress table's key changes, which SQLite can only do by
    building a new table, so it all happens in one write transaction: if the web
    app and the worker start together, the second waits for the first, then
    finds nothing left to do."""
    with transaction() as c:
        default = _default_listener(c)
        have = {row["name"] for row in c.execute("PRAGMA table_info(progress)")}
        if "listener_id" not in have:
            c.execute("DROP TABLE IF EXISTS progress_new")
            c.execute("CREATE TABLE progress_new (book_id TEXT NOT NULL, listener_id INTEGER NOT NULL,"
                      " chapter INTEGER NOT NULL, offset REAL NOT NULL, position REAL NOT NULL,"
                      " word INTEGER, updated REAL NOT NULL, PRIMARY KEY (book_id, listener_id))")
            c.execute("INSERT INTO progress_new (book_id, listener_id, chapter, offset, position, word, updated)"
                      " SELECT book_id, ?, chapter, offset, position, word, updated FROM progress", (default,))
            old = c.execute("SELECT COUNT(*) FROM progress").fetchone()[0]
            new = c.execute("SELECT COUNT(*) FROM progress_new").fetchone()[0]
            if old != new:  # never lose a place: roll back and leave the old table as it was
                raise RuntimeError(f"Moving listening progress kept {new} of {old} rows")
            c.execute("DROP TABLE progress")
            c.execute("ALTER TABLE progress_new RENAME TO progress")
        # Also picks up a checkpoint an old copy of the app saved during an update.
        c.execute("UPDATE checkpoints SET listener_id = ? WHERE listener_id IS NULL", (default,))
        c.execute("CREATE INDEX IF NOT EXISTS checkpoints_listener ON checkpoints(book_id, listener_id, created)")


def _book(row) -> dict | None:
    if row is None:
        return None
    book = dict(row)
    book["meta"] = json.loads(book["meta"] or "{}")
    return book


def create_book(book_id: str, filename: str, voice: str | None, fmt: str, qa: bool,
                kind: str = "tts") -> None:
    now = time.time()
    conn().execute(
        "INSERT INTO books (id, filename, status, meta, voice, fmt, qa, message, created, updated, kind)"
        " VALUES (?, ?, 'parsing', '{}', ?, ?, ?, 'Reading file', ?, ?, ?)",
        (book_id, filename, voice, fmt, int(qa), now, now, kind),
    )


def update_book(book_id: str, **fields) -> None:
    unknown = set(fields) - BOOK_COLUMNS
    if unknown:
        raise ValueError(f"Unknown book fields: {unknown}")
    if isinstance(fields.get("meta"), dict):
        fields["meta"] = json.dumps(fields["meta"])
    fields["updated"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn().execute(f"UPDATE books SET {cols} WHERE id = ?", (*fields.values(), book_id))


def touch(book_id: str) -> None:
    conn().execute("UPDATE books SET updated = ? WHERE id = ?", (time.time(), book_id))


def get_book(book_id: str) -> dict | None:
    return _book(conn().execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone())


def get_status(book_id: str) -> str | None:
    row = conn().execute("SELECT status FROM books WHERE id = ?", (book_id,)).fetchone()
    return row["status"] if row else None


def list_books() -> list[dict]:
    rows = conn().execute("SELECT * FROM books ORDER BY created, filename").fetchall()
    return [_book(r) for r in rows]


def books_by_status(*statuses: str) -> list[dict]:
    marks = ",".join("?" * len(statuses))
    rows = conn().execute(
        f"SELECT * FROM books WHERE status IN ({marks}) ORDER BY updated", statuses
    ).fetchall()
    return [_book(r) for r in rows]


def delete_book(book_id: str) -> None:
    with transaction() as c:
        c.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM progress WHERE book_id = ?", (book_id,))  # every listener's
        c.execute("DELETE FROM checkpoints WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM versions WHERE book_id = ?", (book_id,))
        c.execute("DELETE FROM books WHERE id = ?", (book_id,))


# Versions: a finished copy of a book in each voice it was made in (versions.py).

def versions(book_id: str) -> list[dict]:
    rows = conn().execute("SELECT * FROM versions WHERE book_id = ? ORDER BY created DESC", (book_id,))
    return [dict(r) for r in rows]


def all_versions() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in conn().execute("SELECT * FROM versions ORDER BY created DESC"):
        grouped.setdefault(r["book_id"], []).append(dict(r))
    return grouped


def get_version(book_id: str, version_id: str) -> dict | None:
    row = conn().execute("SELECT * FROM versions WHERE book_id = ? AND id = ?", (book_id, version_id)).fetchone()
    return dict(row) if row else None


def save_version(v: dict) -> None:
    """Add or refresh a version; a name the user gave it stays."""
    cols = [k for k in v if k != "name"]
    conn().execute(
        f"INSERT INTO versions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        f" ON CONFLICT(book_id, id) DO UPDATE SET {', '.join(f'{k} = excluded.{k}' for k in cols)}",
        [v[k] for k in cols])


def rename_version(book_id: str, version_id: str, name: str | None) -> None:
    conn().execute("UPDATE versions SET name = ? WHERE book_id = ? AND id = ?", (name, book_id, version_id))


def delete_version(book_id: str, version_id: str) -> None:
    conn().execute("DELETE FROM versions WHERE book_id = ? AND id = ?", (book_id, version_id))


def outdate_versions(book_id: str) -> None:
    """The text changed: copies made from the old words no longer match it."""
    conn().execute("UPDATE versions SET outdated = 1 WHERE book_id = ?", (book_id,))


# Listeners: the people who read along. Each keeps their own place and
# checkpoints in every book. There's always at least one; the oldest is the
# default, used by anything that doesn't say who's listening (a reader page
# opened before listeners existed, say).

class ListenerGone(LookupError):
    """The listener was removed (perhaps on another device)."""


class NameTaken(ValueError):
    """Another listener already has this name."""


def _default_listener(c: sqlite3.Connection) -> int:
    """Call inside a transaction, so two processes can't both add "Me"."""
    row = c.execute("SELECT id FROM listeners ORDER BY id LIMIT 1").fetchone()
    if row:
        return row["id"]
    return c.execute("INSERT INTO listeners (name, created) VALUES ('Me', ?)", (time.time(),)).lastrowid


def default_listener() -> int:
    row = conn().execute("SELECT id FROM listeners ORDER BY id LIMIT 1").fetchone()
    if row:
        return row["id"]
    with transaction() as c:
        return _default_listener(c)


def list_listeners() -> list[dict]:
    return [dict(r) for r in conn().execute("SELECT * FROM listeners ORDER BY id")]


def get_listener(listener_id: int) -> dict | None:
    row = conn().execute("SELECT * FROM listeners WHERE id = ?", (listener_id,)).fetchone()
    return dict(row) if row else None


def _check_name(c: sqlite3.Connection, name: str, listener_id: int = 0) -> None:
    # Two people called the same would look identical on the "Who's listening?" buttons.
    row = c.execute("SELECT name FROM listeners WHERE name = ? COLLATE NOCASE AND id != ?",
                    (name, listener_id)).fetchone()
    if row:
        raise NameTaken(row["name"])


def add_listener(name: str) -> int:
    with transaction() as c:
        _check_name(c, name)
        return c.execute("INSERT INTO listeners (name, created) VALUES (?, ?)",
                         (name, time.time())).lastrowid


def rename_listener(listener_id: int, name: str) -> None:
    with transaction() as c:
        _require_listener(c, listener_id)
        _check_name(c, name, listener_id)
        c.execute("UPDATE listeners SET name = ? WHERE id = ?", (name, listener_id))


def delete_listener(listener_id: int) -> None:
    """Remove a listener with their places and checkpoints. Raises ValueError
    for the last one: someone has to be listening."""
    with transaction() as c:
        if not c.execute("SELECT 1 FROM listeners WHERE id = ?", (listener_id,)).fetchone():
            raise ListenerGone(listener_id)
        if c.execute("SELECT COUNT(*) FROM listeners").fetchone()[0] <= 1:
            raise ValueError("last listener")
        c.execute("DELETE FROM progress WHERE listener_id = ?", (listener_id,))
        c.execute("DELETE FROM checkpoints WHERE listener_id = ?", (listener_id,))
        c.execute("DELETE FROM listeners WHERE id = ?", (listener_id,))


def _require_listener(c: sqlite3.Connection, listener_id: int) -> None:
    # Checked in the same transaction as the write, so a place can't be saved
    # for someone being removed at that moment.
    if not c.execute("SELECT 1 FROM listeners WHERE id = ?", (listener_id,)).fetchone():
        raise ListenerGone(listener_id)


# Read along: where each listener is, and the places they can go back to.
# Stored as chapter + offset into it, so a rebuild that shifts times keeps them.
AUTO_CHECKPOINTS_KEPT = 60


def get_progress(book_id: str, listener_id: int | None = None) -> dict | None:
    if listener_id is None:
        listener_id = default_listener()
    row = conn().execute("SELECT * FROM progress WHERE book_id = ? AND listener_id = ?",
                         (book_id, listener_id)).fetchone()
    return dict(row) if row else None


def set_progress(book_id: str, chapter: int, offset: float, position: float,
                 word: int | None = None, listener_id: int | None = None) -> None:
    with transaction() as c:
        if listener_id is None:
            listener_id = _default_listener(c)
        _require_listener(c, listener_id)
        c.execute(
            "INSERT INTO progress (book_id, listener_id, chapter, offset, position, word, updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(book_id, listener_id) DO UPDATE SET chapter = excluded.chapter,"
            " offset = excluded.offset, position = excluded.position, word = excluded.word,"
            " updated = excluded.updated",
            (book_id, listener_id, chapter, offset, position, word, time.time()))


def all_progress(listener_id: int | None = None) -> dict[str, dict]:
    """One listener's place in every book they've started."""
    if listener_id is None:
        listener_id = default_listener()
    return {r["book_id"]: dict(r) for r in
            conn().execute("SELECT * FROM progress WHERE listener_id = ?", (listener_id,))}


def book_places(book_id: str) -> list[dict]:
    """Every listener's place in one book, with their names."""
    rows = conn().execute(
        "SELECT p.*, l.name FROM progress p JOIN listeners l ON l.id = p.listener_id"
        " WHERE p.book_id = ? ORDER BY l.id", (book_id,)).fetchall()
    return [dict(r) for r in rows]


def list_checkpoints(book_id: str, listener_id: int | None = None) -> list[dict]:
    if listener_id is None:
        listener_id = default_listener()
    rows = conn().execute("SELECT * FROM checkpoints WHERE book_id = ? AND listener_id = ?"
                          " ORDER BY created DESC", (book_id, listener_id)).fetchall()
    return [dict(r) for r in rows]


def add_checkpoint(book_id: str, chapter: int, offset: float, position: float,
                   kind: str, label: str | None, word: int | None = None,
                   listener_id: int | None = None) -> int:
    with transaction() as c:
        if listener_id is None:
            listener_id = _default_listener(c)
        _require_listener(c, listener_id)
        cur = c.execute(
            "INSERT INTO checkpoints (book_id, listener_id, chapter, offset, position, word, kind, label, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (book_id, listener_id, chapter, offset, position, word, kind, label, time.time()))
        # Automatic ones pile up over a long book; keep each listener's most recent.
        c.execute(
            "DELETE FROM checkpoints WHERE book_id = ? AND listener_id = ? AND kind != 'manual' AND id NOT IN ("
            " SELECT id FROM checkpoints WHERE book_id = ? AND listener_id = ? AND kind != 'manual'"
            " ORDER BY created DESC LIMIT ?)",
            (book_id, listener_id, book_id, listener_id, AUTO_CHECKPOINTS_KEPT))
        return cur.lastrowid


def import_book(m: dict, output: str) -> None:
    """Create or update a book published by another Forge: finished, with its
    chapters as they were built. Every listener's progress and checkpoints are kept."""
    now = time.time()
    fields = {"filename": m["filename"], "status": "done", "meta": json.dumps(m.get("meta") or {}),
              "voice": m.get("voice"), "mood": m.get("mood"), "speed": m.get("speed") or 1.0,
              "fmt": m.get("fmt") or "m4b", "qa": int(bool(m.get("qa"))), "progress": 1.0,
              "message": m.get("message") or "Done", "output": output,
              "has_cover": int(bool(m.get("has_cover"))), "rtf": m.get("rtf"),
              "origin": m.get("origin"), "published": now, "edited": 0, "forwarded": None,
              "kind": m.get("kind") or "tts"}
    # An added audiobook wasn't built with a voice, so it has no voice versions.
    fields["built"] = built_with(fields) if fields["kind"] != "audio" else None
    with transaction() as c:
        if c.execute("SELECT 1 FROM books WHERE id = ?", (m["id"],)).fetchone():
            cols = ", ".join(f"{k} = ?" for k in fields)
            c.execute(f"UPDATE books SET {cols}, updated = ? WHERE id = ?", (*fields.values(), now, m["id"]))
        else:
            cols = ", ".join(fields)
            c.execute(f"INSERT INTO books (id, {cols}, created, updated) VALUES (?, {', '.join('?' * len(fields))}, ?, ?)",
                      (m["id"], *fields.values(), now, now))
        c.execute("DELETE FROM chapters WHERE book_id = ?", (m["id"],))
        c.executemany(
            "INSERT INTO chapters (book_id, idx, title, include, words, units_total, units_done, state, duration, wer, start)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(m["id"], ch["idx"], ch["title"], int(ch["include"]), ch.get("words") or 0,
              ch.get("units_total") or 0, ch.get("units_done") or 0, ch.get("state") or "done",
              ch.get("duration"), ch.get("wer"), ch.get("start")) for ch in m.get("chapters", [])])


def delete_checkpoint(book_id: str, checkpoint_id: int, listener_id: int | None = None) -> None:
    if listener_id is None:
        listener_id = default_listener()
    conn().execute("DELETE FROM checkpoints WHERE book_id = ? AND listener_id = ? AND id = ?",
                   (book_id, listener_id, checkpoint_id))


def set_chapters(book_id: str, rows: list[dict]) -> None:
    with transaction() as c:
        c.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
        c.executemany(
            "INSERT INTO chapters (book_id, idx, title, include, words, start, duration)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(book_id, r["idx"], r["title"], int(r["include"]), r["words"], r.get("start"),
              r.get("duration")) for r in rows],
        )


def chapters(book_id: str) -> list[dict]:
    rows = conn().execute(
        "SELECT * FROM chapters WHERE book_id = ? ORDER BY idx", (book_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def all_chapters() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in conn().execute("SELECT * FROM chapters ORDER BY book_id, idx"):
        grouped.setdefault(r["book_id"], []).append(dict(r))
    return grouped


def update_chapters(book_id: str, rows: list[dict]) -> None:
    """rows: [{"idx": 3, "units_done": 4, ...}, ...]"""
    if not rows:
        return
    with transaction() as c:
        for row in rows:
            fields = {k: v for k, v in row.items() if k != "idx"}
            unknown = set(fields) - CHAPTER_COLUMNS
            if unknown:
                raise ValueError(f"Unknown chapter fields: {unknown}")
            cols = ", ".join(f"{k} = ?" for k in fields)
            c.execute(
                f"UPDATE chapters SET {cols} WHERE book_id = ? AND idx = ?",
                (*fields.values(), book_id, row["idx"]),
            )
        c.execute("UPDATE books SET updated = ? WHERE id = ?", (time.time(), book_id))


# Voice samples: the web app files a request, the worker voices it on the
# models it already has loaded, and the web app waits for the file (or, for a
# progress bar, asks how far along it is: its stage).

def sample_key(out: str) -> str:
    """A sample's id: one per output file."""
    return hashlib.sha1(out.encode()).hexdigest()[:16]


def add_sample(job: dict) -> str:
    """Queue a sample (or join the one already queued for the same file)."""
    sample_id = sample_key(job["out"])
    conn().execute(
        "INSERT INTO samples (id, job, status, created) VALUES (?, ?, 'pending', ?)"
        " ON CONFLICT(id) DO UPDATE SET job = excluded.job, status = 'pending', stage = NULL,"
        " message = NULL, created = excluded.created WHERE samples.status IN ('done', 'failed')",
        (sample_id, json.dumps(job), time.time()))
    return sample_id


def get_sample(sample_id: str) -> dict | None:
    row = conn().execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
    return dict(row) if row else None


def _cloned(job: str) -> bool:
    """A sample in a cloned voice: the clone server makes it, not this machine's GPU."""
    return (json.loads(job).get("voice") or {}).get("engine") == "clone"


def samples_waiting(clone: bool | None = None) -> bool:
    """Samples waiting to be voiced; clone: only those in cloned voices (True) or the rest (False)."""
    if clone is None:
        return conn().execute("SELECT 1 FROM samples WHERE status = 'pending' LIMIT 1").fetchone() is not None
    return any(_cloned(r["job"]) == clone
               for r in conn().execute("SELECT job FROM samples WHERE status = 'pending'"))


def note_samples(stage: str, status: str = "pending") -> None:
    """Where samples with this status are: gpu (the voice creator is pausing for
    them), starting (the voices are loading), voicing. Shown as a progress bar."""
    conn().execute("UPDATE samples SET stage = ? WHERE status = ? AND stage IS NOT ?", (stage, status, stage))


def take_samples(stage: str = "voicing", clone: bool | None = None) -> list[tuple[str, dict]]:
    """Waiting samples, marked as being voiced; clone as for samples_waiting."""
    # Polled several times a second: only take the write lock when there's work.
    if not conn().execute("SELECT 1 FROM samples WHERE status = 'pending' LIMIT 1").fetchone():
        return []
    with transaction() as c:
        rows = [r for r in c.execute("SELECT id, job FROM samples WHERE status = 'pending' ORDER BY created")
                if clone is None or _cloned(r["job"]) == clone]
        c.executemany("UPDATE samples SET status = 'working', stage = ? WHERE id = ?",
                      [(stage, r["id"]) for r in rows])
    return [(r["id"], json.loads(r["job"])) for r in rows]


def finish_sample(sample_id: str, error: str | None = None) -> None:
    conn().execute("UPDATE samples SET status = ?, message = ? WHERE id = ?",
                   ("failed" if error else "done", error, sample_id))


def forget_sample(sample_id: str) -> None:
    conn().execute("DELETE FROM samples WHERE id = ?", (sample_id,))


def reset_samples() -> None:
    """Worker start: requeue samples a previous run didn't finish; forget old ones."""
    conn().execute("UPDATE samples SET status = 'pending', stage = NULL WHERE status = 'working'")
    conn().execute("DELETE FROM samples WHERE created < ?", (time.time() - 3600,))


def stamp() -> tuple:
    row = conn().execute("SELECT COUNT(*), COALESCE(MAX(updated), 0) FROM books").fetchone()
    return tuple(row)
