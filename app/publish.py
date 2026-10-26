"""Send finished books to another Forge, an always-on server, so Read along works
there from any device (phone, tablet, browser) and keeps one place per book.

The Forge with the GPU voices books and publishes them (PUBLISH_URL). The
server stores the audio, cover, chapter text, source file and read-along map,
and keeps listening progress. It remembers where each book came from (its
"origin"), so a voice change asked for on the server is voiced back here.
"""
from __future__ import annotations

import http.client
import io
import json
import logging
import tarfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from . import config, db, readalong

log = logging.getLogger("forge.publish")
TOKEN_HEADER = "X-Forge-Token"


class PublishError(Exception):
    def __init__(self, message: str, status: int = 0, conflict: dict | None = None):
        super().__init__(message)
        self.status = status
        self.conflict = conflict


def manifest(book: dict) -> dict:
    chapters = [{k: c[k] for k in ("idx", "title", "include", "words", "units_total", "units_done",
                                   "state", "duration", "wer", "start")} for c in db.chapters(book["id"])]
    return {
        "v": 1, "id": book["id"], "filename": book["filename"], "meta": book["meta"], "kind": book["kind"],
        "voice": book["voice"], "mood": book["mood"], "speed": book["speed"], "fmt": book["fmt"],
        "qa": book["qa"], "has_cover": book["has_cover"], "message": book["message"], "rtf": book["rtf"],
        "ext": Path(book["output"]).suffix.lstrip("."), "chapters": chapters,
        "origin": config.PUBLIC_URL or None,
    }


def pack(book_id: str) -> bytes:
    """Everything but the audio, which goes up separately and streamed: the
    chapter text (so the server can show and even rebuild the book), the
    source file, the cover and the read-along map."""
    folder = config.BOOKS_DIR / book_id
    book = db.get_book(book_id)
    # An added audiobook's source is its audio, which goes up on its own; its
    # transcripts let the server rebuild Read along after a text edit.
    audio_book = bool(book) and book["kind"] == "audio"
    sources = [] if audio_book else sorted(folder.glob("source*"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sources + sorted(folder.glob("cover*.jpg")):
            tar.add(path, arcname=path.name)
        for name in ("text", "transcript") if audio_book else ("text",):
            if (folder / name).is_dir():
                tar.add(folder / name, arcname=name)
        if audio_book and (folder / "parts.json").exists():  # Forge made its file: the server may retag it too
            tar.add(folder / "parts.json", arcname="parts.json")
        ra = readalong.path_for(book_id)
        if ra.exists():
            tar.add(ra, arcname=ra.name)
    return buf.getvalue()


def _post_multipart(url: str, fields: dict[str, bytes], files: dict[str, tuple[str, Path | bytes]],
                    timeout: float = 600) -> dict:
    """POST multipart/form-data, streaming file parts from disk (a long book's
    audio is hundreds of megabytes)."""
    boundary = uuid.uuid4().hex
    parts: list[tuple[bytes, Path | bytes | None, bytes]] = []
    for name, value in fields.items():
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        parts.append((head, value, b"\r\n"))
    for name, (filename, src) in files.items():
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n").encode()
        parts.append((head, src, b"\r\n"))
    tail = f"--{boundary}--\r\n".encode()
    size = sum(len(h) + (src.stat().st_size if isinstance(src, Path) else len(src)) + len(t)
               for h, src, t in parts) + len(tail)
    target = urlparse(url)
    conn_cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(target.hostname, target.port, timeout=timeout)
    try:
        conn.putrequest("POST", target.path + (f"?{target.query}" if target.query else ""))
        conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        conn.putheader("Content-Length", str(size))
        if config.SHARE_TOKEN:
            conn.putheader(TOKEN_HEADER, config.SHARE_TOKEN)
        conn.endheaders()
        for head, src, end in parts:
            conn.send(head)
            if isinstance(src, Path):
                with src.open("rb") as f:
                    while chunk := f.read(1 << 20):
                        conn.send(chunk)
            else:
                conn.send(src)
            conn.send(end)
        conn.send(tail)
        resp = conn.getresponse()
        body = resp.read()
    finally:
        conn.close()
    try:
        data = json.loads(body or b"null")
    except ValueError:
        data = {"detail": body[:300].decode(errors="replace")}
    if resp.status >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        if isinstance(detail, dict):
            raise PublishError(detail.get("message", "The server refused the book."), resp.status, detail.get("conflict"))
        raise PublishError(str(detail or f"The server answered {resp.status}."), resp.status)
    return data


def send(book_id: str, replace: bool = False) -> dict:
    """Publish one finished book. Raises PublishError (with .conflict when the
    server already has a different copy at the same place and replace is off)."""
    if not config.PUBLISH_URL:
        raise PublishError("No server is set up. Set PUBLISH_URL in .env.")
    book = db.get_book(book_id)
    if not book or book["status"] != "done" or not book["output"] or not Path(book["output"]).exists():
        raise PublishError("Only finished books can be sent. Build this one first.")
    audio = Path(book["output"])
    started = time.monotonic()
    url = f"{config.PUBLISH_URL}/api/import" + ("?replace=true" if replace else "")
    try:
        result = _post_multipart(url, {"manifest": json.dumps(manifest(book)).encode()},
                                 {"pack": ("book.tgz", pack(book_id)), "audio": (audio.name, audio)})
    except PublishError:
        raise
    except OSError as exc:
        raise PublishError(f"The server at {config.PUBLISH_URL} didn’t answer ({exc}).") from None
    db.update_book(book_id, published=time.time())
    _remember_site(result.get("site_url") if isinstance(result, dict) else None)
    log.info("Published %s to %s in %.1fs (%.0f MB)", book_id, config.PUBLISH_URL,
             time.monotonic() - started, audio.stat().st_size / 1e6)
    return result


def _call(method: str, path: str, timeout: float = 30):
    import urllib.request
    headers = {TOKEN_HEADER: config.SHARE_TOKEN} if config.SHARE_TOKEN else {}
    req = urllib.request.Request(config.PUBLISH_URL + path, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def sync_voices() -> tuple[int, int]:
    """Keep the server's copy of the voices made here in step: send new,
    retrained, renamed or newly measured ones, and remove deleted ones. Then
    they can be picked on the server too (a book is still voiced here).
    Returns (sent, removed)."""
    import hashlib
    if not config.PUBLISH_URL:
        return 0, 0
    local = {meta.with_suffix("").stem: meta for kind in ("onnx", "pt")  # Piper voices and Kokoro voicepacks
             for meta in config.CUSTOM_VOICES_DIR.glob(f"*.{kind}.json") if meta.with_suffix("").exists()}
    remote = _call("GET", "/api/custom-voices") or {}
    sent = removed = 0
    for slug, meta in sorted(local.items()):
        data = meta.read_bytes()
        if (remote.get(slug) or {}).get("config") == hashlib.sha1(data).hexdigest():
            continue
        _post_multipart(f"{config.PUBLISH_URL}/api/custom-voices/{slug}", {},
                        {"voice_config": (meta.name, data), "model": (meta.with_suffix("").name, meta.with_suffix(""))})
        sent += 1
    for slug in sorted(remote.keys() - local.keys()):
        _call("DELETE", f"/api/custom-voices/{slug}")
        removed += 1
    # Cloned voices: their clip, its text and settings, a few hundred KB each.
    from .voices import CLONED_FILES, cloned_fingerprint
    folders = [p for p in config.CLONED_VOICES_DIR.iterdir() if p.is_dir()] if config.CLONED_VOICES_DIR.is_dir() else []
    cloned = {p.name: p for p in folders if all((p / name).exists() for name in CLONED_FILES)}
    remote = _call("GET", "/api/cloned-voices") or {}
    for slug, folder in sorted(cloned.items()):
        if (remote.get(slug) or {}).get("hash") == cloned_fingerprint(folder):
            continue
        _post_multipart(f"{config.PUBLISH_URL}/api/cloned-voices/{slug}", {},
                        {"voice": ("voice.json", (folder / "voice.json").read_bytes()),
                         "ref_text": ("ref.txt", (folder / "ref.txt").read_bytes()),
                         "ref_audio": ("ref.wav", folder / "ref.wav")})
        sent += 1
    for slug in sorted(remote.keys() - cloned.keys()):
        _call("DELETE", f"/api/cloned-voices/{slug}")
        removed += 1
    if sent or removed:
        log.info("Voices on %s: %d sent, %d removed", config.PUBLISH_URL, sent, removed)
    return sent, removed


def voice_sync_loop(stop) -> None:
    """The worker's thread: syncs whenever the voices folder changes (checked
    every minute), and every quarter hour anyway, in case the server missed one."""
    from . import voices
    last, last_full = None, 0.0
    while not stop.wait(60):
        stamp = voices._custom_stamp()
        if stamp == last and time.monotonic() - last_full < 900:
            continue
        try:
            sync_voices()
            last, last_full = stamp, time.monotonic()
        except Exception as exc:  # the server is off or busy: next time
            log.warning("Couldn’t sync voices to %s: %s", config.PUBLISH_URL, exc)


SITE_FILE = config.DATA_DIR / "server.json"


def _remember_site(site_url: str | None) -> None:
    """The server says which address people use (https once it's set up)."""
    try:
        SITE_FILE.write_text(json.dumps({"site_url": site_url}), encoding="utf-8")
    except OSError:
        pass


def site_url() -> str | None:
    """Where people open the server: its https address if it has one."""
    if not config.PUBLISH_URL:
        return None
    try:
        return json.loads(SITE_FILE.read_text(encoding="utf-8")).get("site_url") or config.PUBLISH_URL
    except (OSError, ValueError):
        return config.PUBLISH_URL


def read_url(book: dict) -> str | None:
    """Where this book's Read along lives once it's on the server."""
    if config.PUBLISH_URL and book.get("published"):
        return f"{site_url()}/read/{book['id']}"
    return None
