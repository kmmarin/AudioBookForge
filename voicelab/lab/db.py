"""SQLite state: voice projects, their recordings, clips and jobs.

The web app and the job processes share it (WAL mode). Jobs save their
progress here as they go, so a restart carries on where it stopped."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    name_source TEXT NOT NULL DEFAULT '',
    name_note TEXT,
    accent TEXT NOT NULL DEFAULT 'en-us',
    status TEXT NOT NULL DEFAULT 'new',
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    readiness TEXT,
    training TEXT,
    quality TEXT,
    voice_slug TEXT,
    auto_train INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    received INTEGER NOT NULL DEFAULT 0,
    duration REAL,
    tags TEXT,
    status TEXT NOT NULL,
    done_until REAL NOT NULL DEFAULT 0,
    last_end REAL NOT NULL DEFAULT 0,
    gaps TEXT,
    message TEXT,
    fingerprint BLOB,
    chapters TEXT,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sources_project ON sources(project_id);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    words INTEGER NOT NULL,
    conf REAL,
    unsure REAL,
    snr REAL,
    clipped REAL,
    reason TEXT,
    sim REAL,
    verdict TEXT,
    role TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS clips_project ON clips(project_id, source_id, start);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created REAL NOT NULL,
    started REAL,
    finished REAL
);
"""

# Columns added after the first release (ALTER is safe to race between processes).
# model: what the voice is made with (lab/clone.py), null = the best this setup runs;
# clone: the reference clips tried for a cloned voice and how each sample turned out;
# picking: 1 = train and check only on the clips the owner approved (clips.pick:
# null not reviewed, 1 approved, 0 left out).
# reference: the clips the owner marked as how the narrator should sound; every
# clip is matched against them when picking.
ADDED = {"projects": (("model", "TEXT"), ("clone", "TEXT"), ("picking", "INTEGER NOT NULL DEFAULT 0"),
                      ("reference", "TEXT")),
         "clips": (("pick", "INTEGER"),)}

JSON_FIELDS = {"readiness", "training", "quality", "tags", "gaps", "chapters", "clone", "reference"}
PROJECT_COLUMNS = {"name", "name_source", "name_note", "accent", "status", "progress", "message",
                   "readiness", "training", "quality", "voice_slug", "auto_train", "model", "clone", "picking",
                   "reference"}
SOURCE_COLUMNS = {"filename", "path", "size", "received", "duration", "tags", "status", "done_until",
                  "last_end", "gaps", "message", "fingerprint", "chapters"}

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
    for table, columns in ADDED.items():
        have = {row["name"] for row in c.execute(f"PRAGMA table_info({table})")}
        for column, kind in columns:
            if column in have:
                continue
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc):
                    raise


def _row(row) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key in JSON_FIELDS & out.keys():
        out[key] = json.loads(out[key]) if out[key] else None
    return out


def _encode(fields: dict) -> dict:
    return {k: (json.dumps(v) if k in JSON_FIELDS and v is not None else v) for k, v in fields.items()}


# --- Projects -----------------------------------------------------------------------

def create_project(project_id: str, accent: str, auto_train: bool = False) -> None:
    now = time.time()
    conn().execute("INSERT INTO projects (id, accent, auto_train, status, message, created, updated)"
                   " VALUES (?, ?, ?, 'new', 'Add recordings of the narrator', ?, ?)",
                   (project_id, accent, int(auto_train), now, now))


def update_project(project_id: str, **fields) -> None:
    unknown = set(fields) - PROJECT_COLUMNS
    if unknown:
        raise ValueError(f"Unknown project fields: {unknown}")
    fields = _encode(fields)
    fields["updated"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn().execute(f"UPDATE projects SET {cols} WHERE id = ?", (*fields.values(), project_id))


def get_project(project_id: str) -> dict | None:
    return _row(conn().execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone())


def list_projects() -> list[dict]:
    return [_row(r) for r in conn().execute("SELECT * FROM projects ORDER BY created DESC")]


def delete_project(project_id: str) -> None:
    with transaction() as c:
        for table in ("clips", "sources", "jobs"):
            c.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
        c.execute("DELETE FROM projects WHERE id = ?", (project_id,))


def stamp() -> tuple:
    row = conn().execute("SELECT COUNT(*), COALESCE(MAX(updated), 0) FROM projects").fetchone()
    return tuple(row)


def touch(project_id: str) -> None:
    conn().execute("UPDATE projects SET updated = ? WHERE id = ?", (time.time(), project_id))


# --- Sources (uploaded recordings) ----------------------------------------------------

def create_source(source_id: str, project_id: str, filename: str, path: str, size: int) -> None:
    conn().execute("INSERT INTO sources (id, project_id, filename, path, size, status, created)"
                   " VALUES (?, ?, ?, ?, ?, 'uploading', ?)",
                   (source_id, project_id, filename, path, size, time.time()))
    touch(project_id)


def update_source(source_id: str, **fields) -> None:
    unknown = set(fields) - SOURCE_COLUMNS
    if unknown:
        raise ValueError(f"Unknown source fields: {unknown}")
    fields = _encode(fields)
    cols = ", ".join(f"{k} = ?" for k in fields)
    c = conn()
    c.execute(f"UPDATE sources SET {cols} WHERE id = ?", (*fields.values(), source_id))
    c.execute("UPDATE projects SET updated = ? WHERE id = (SELECT project_id FROM sources WHERE id = ?)",
              (time.time(), source_id))


def get_source(source_id: str) -> dict | None:
    return _row(conn().execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone())


def sources(project_id: str) -> list[dict]:
    """A project's recordings in book order: by track number, then by name
    with numbers compared as numbers (Part2 before Part10), then by arrival."""
    rows = [_row(r) for r in conn().execute(
        "SELECT * FROM sources WHERE project_id = ? ORDER BY created", (project_id,))]
    return sorted(rows, key=book_order)


CONFLICTED_RE = r"\s*\([^)]*conflicted copy[^)]*\)"


def book_order(source: dict) -> tuple:
    """Sync tools' "(… conflicted copy …)" files sort right after their original."""
    import re
    track = str((source.get("tags") or {}).get("track") or "").split("/")[0].strip()
    base = re.sub(CONFLICTED_RE, "", source["filename"].lower())
    name = [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", base)]
    conflicted = base != source["filename"].lower()
    return (int(track) if track.isdigit() else 10**6, name, conflicted, source["created"])


def delete_source(source_id: str) -> None:
    with transaction() as c:
        c.execute("DELETE FROM clips WHERE source_id = ?", (source_id,))
        c.execute("DELETE FROM sources WHERE id = ?", (source_id,))


# --- Clips ---------------------------------------------------------------------------------

CLIP_FIELDS = ("start", "end", "text", "words", "conf", "unsure", "snr", "clipped", "reason", "embedding")


def add_clips(project_id: str, source_id: str, clips: list[dict], done_until: float,
              last_end: float, gaps: dict) -> None:
    """One analysed window: its clips and the source's new position, in one go,
    so a restart never counts a window twice."""
    with transaction() as c:
        c.executemany(
            f"INSERT INTO clips (project_id, source_id, {', '.join(CLIP_FIELDS)})"
            f" VALUES (?, ?, {', '.join('?' * len(CLIP_FIELDS))})",
            [(project_id, source_id, *(clip.get(k) for k in CLIP_FIELDS)) for clip in clips])
        c.execute("UPDATE sources SET done_until = ?, last_end = ?, gaps = ? WHERE id = ?",
                  (done_until, last_end, json.dumps(gaps), source_id))
        c.execute("UPDATE projects SET updated = ? WHERE id = ?", (time.time(), project_id))


def clips(project_id: str, with_embedding: bool = False) -> list[dict]:
    cols = "*" if with_embedding else ", ".join(
        ["id", "source_id", "start", "end", "text", "words", "conf", "unsure", "snr", "clipped",
         "reason", "sim", "verdict", "role", "pick"])
    return [dict(r) for r in conn().execute(
        f"SELECT {cols} FROM clips WHERE project_id = ? ORDER BY source_id, start", (project_id,))]


def get_clip(clip_id: int) -> dict | None:
    row = conn().execute("SELECT id, project_id, source_id, start, end, text, verdict, role, pick"
                         " FROM clips WHERE id = ?", (clip_id,)).fetchone()
    return dict(row) if row else None


def relisten(project_id: str) -> None:
    """Forget every clip, so the recordings are heard and cut again from the start
    (the recordings themselves stay)."""
    with transaction() as c:
        c.execute("DELETE FROM clips WHERE project_id = ?", (project_id,))
        c.execute("UPDATE sources SET status = 'queued', done_until = 0, last_end = 0, gaps = NULL,"
                  " message = 'Waiting to be heard again' WHERE project_id = ? AND status IN ('done', 'analyzing', 'queued')",
                  (project_id,))
        c.execute("UPDATE projects SET updated = ? WHERE id = ?", (time.time(), project_id))


def set_picks(project_id: str, picks: dict[int, int | None]) -> None:
    """The owner's choices (1 approved, 0 left out, None not reviewed), in one
    go; only this project's clips are touched."""
    with transaction() as c:
        c.executemany("UPDATE clips SET pick = ? WHERE id = ? AND project_id = ?",
                      [(pick, int(clip_id), project_id) for clip_id, pick in picks.items()])


def set_verdicts(rows: list[tuple]) -> None:
    """rows: (sim, verdict, role, clip id)"""
    with transaction() as c:
        c.executemany("UPDATE clips SET sim = ?, verdict = ?, role = ? WHERE id = ?", rows)


# --- Jobs ----------------------------------------------------------------------------------

def add_job(project_id: str, kind: str, created: float | None = None) -> int:
    """Queue a job unless the same one is already waiting. One that's running
    may be just finishing, so another is queued behind it and finds whatever
    it missed (or nothing, quickly). created: a job running again keeps its
    original place in the queue."""
    with transaction() as c:
        row = c.execute("SELECT id FROM jobs WHERE project_id = ? AND kind = ? AND status = 'queued'",
                        (project_id, kind)).fetchone()
        if row:
            return row["id"]
        return c.execute("INSERT INTO jobs (project_id, kind, status, created) VALUES (?, ?, 'queued', ?)",
                         (project_id, kind, created or time.time())).lastrowid


def next_job(kind: str) -> dict | None:
    """The oldest queued job of this kind, marked running. Analysis and
    training each run one at a time, in lanes of their own, so new
    recordings are heard while a voice trains."""
    with transaction() as c:
        row = c.execute("SELECT * FROM jobs WHERE status = 'queued' AND kind = ? ORDER BY created LIMIT 1",
                        (kind,)).fetchone()
        if row is None:
            return None
        c.execute("UPDATE jobs SET status = 'running', started = ? WHERE id = ?", (time.time(), row["id"]))
        return dict(row)


def finish_job(job_id: int, status: str, message: str | None = None) -> None:
    conn().execute("UPDATE jobs SET status = ?, message = ?, finished = ? WHERE id = ?",
                   (status, message, time.time(), job_id))


def requeue_running() -> None:
    conn().execute("UPDATE jobs SET status = 'queued' WHERE status = 'running'")


def job_waiting(project_id: str, kind: str, besides_running: bool = False) -> bool:
    """A job of this kind queued or running (besides_running: queued only)."""
    states = "('queued')" if besides_running else "('queued', 'running')"
    return conn().execute(f"SELECT 1 FROM jobs WHERE project_id = ? AND kind = ? AND status IN {states}",
                          (project_id, kind)).fetchone() is not None


def cancel_jobs(project_id: str, kind: str | None = None) -> None:
    sql = "UPDATE jobs SET status = 'cancelled', finished = ? WHERE project_id = ? AND status = 'queued'"
    args: tuple = (time.time(), project_id)
    if kind:
        sql += " AND kind = ?"
        args += (kind,)
    conn().execute(sql, args)
