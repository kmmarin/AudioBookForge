"""GPU servers: the machines running the clone server (clone/server.py) that
speak cloned voices.

The list is kept in GPU_SERVERS_FILE, in the shared models volume so Forge and
the voice creator read the same one, and changed on Forge's Settings page:

    {"servers": [{"id": "a1b2c3d4", "name": "gaming-pc", "url": "http://192.168.1.30:8100",
                  "token": "…", "voices": true}]}

A server set in .env (CLONE_URL, with CLONE_TOKEN or SHARE_TOKEN) is listed
too, after them. Last comes this machine's own clone server when the installer
runs one beside Forge as a fallback on a smaller card (CLONE_LOCAL_URL).

A job goes to the first server that is answering (checked at most every
HEALTH_SECONDS) and, for a given model, can run it: it fits the card and the
video memory free now, or it's loaded already. Else to the first answering
one, else the first, so its error says what's wrong. So one server stands in
for another, and this machine's card is used when it can take the model.

Standard library only (and app/clone.py for a client): the voice creator imports this too.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROLES = ("voices",)
FILE = Path(os.environ.get("GPU_SERVERS_FILE")
            or Path(os.environ.get("MODELS_DIR") or "/models") / "gpu-servers.json")
HEALTH_SECONDS = 30
_lock = threading.Lock()
_health: dict[str, tuple[float, dict]] = {}


def _env_server() -> dict | None:
    url = (os.environ.get("CLONE_URL") or "").strip().rstrip("/")
    if not url:
        return None
    token = (os.environ.get("CLONE_TOKEN") or "").strip() or (os.environ.get("SHARE_TOKEN") or "").strip()
    return {"id": "env", "name": "Set in .env", "url": url, "token": token,
            "voices": True, "source": "env"}


def _local_server() -> dict | None:
    """The clone server beside Forge on this machine's own card, as a fallback."""
    url = (os.environ.get("CLONE_LOCAL_URL") or "").strip().rstrip("/")
    if not url:
        return None
    token = (os.environ.get("CLONE_TOKEN") or "").strip() or (os.environ.get("SHARE_TOKEN") or "").strip()
    return {"id": "local", "name": "This machine", "url": url, "token": token, "voices": True, "source": "local"}


def _data() -> dict:
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load() -> list[dict]:
    """The servers added on the Settings page, in order."""
    # Every server speaks voices (one saved for pictures only, before they were dropped, too).
    return [{**s, "voices": True, "source": "settings"}
            for s in _data().get("servers") or [] if isinstance(s, dict) and s.get("url")]


def hidden() -> list[str]:
    """Addresses from .env or this machine's fallback that were removed or taken
    over on the Settings page: not listed again unless .env names another one."""
    return [u for u in _data().get("hidden") or [] if isinstance(u, str)]


def save(servers: list[dict], hide: list[str] | None = None) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    keep = ("id", "name", "url", "token") + ROLES
    data = {"servers": [{k: s.get(k) for k in keep} for s in servers],
            "hidden": sorted(set(hidden() if hide is None else hide))}
    tmp = FILE.with_name(f".{FILE.name}.part")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)  # it holds the servers' tokens
    os.replace(tmp, FILE)
    _health.clear()


def extras(with_hidden: bool = False) -> list[dict]:
    """The servers from .env (CLONE_URL) and this machine's fallback (CLONE_LOCAL_URL)."""
    gone = set(hidden())
    return [s for s in (_env_server(), _local_server()) if s and (with_hidden or s["url"] not in gone)]


def servers() -> list[dict]:
    """Every server: Settings' first, then .env's, then this machine's fallback
    (each unless one before it has the same address, or it was removed in Settings)."""
    found = load()
    for extra in extras():
        if not any(s["url"].rstrip("/") == extra["url"] for s in found):
            found.append(extra)
    return found


def new_id() -> str:
    return secrets.token_hex(4)


def health(server: dict, max_age: float = HEALTH_SECONDS) -> dict:
    """The server's /api/health ({"ok", "gpu", "gpu_gb", "loaded", "models"}),
    or {"ok": False, "error"}; cached briefly."""
    url = server["url"].rstrip("/")
    with _lock:
        hit = _health.get(url)
        if hit and time.monotonic() - hit[0] < max_age:
            return hit[1]
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=4) as r:
            out = json.loads(r.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        out = {"ok": False, "error": f"Not answering ({getattr(exc, 'reason', exc)})"}
    with _lock:
        _health[url] = (time.monotonic(), out)
    return out


def runs(server: dict, model: str) -> bool:
    """Whether this server can take the model now: answering, the model fits its
    card, and it's loaded or fits the video memory free (when the server says)."""
    h = health(server)
    if not h.get("ok"):
        return False
    m = next((m for m in h.get("models") or [] if m.get("id") == model), None)
    if m is None or not m.get("fits"):
        return False
    if h.get("loaded") == model or h.get("free_gb") is None or m.get("need_gb") is None:
        return True
    return h["free_gb"] >= m["need_gb"]


def pick(role: str, model: str | None = None) -> dict | None:
    """The server for a job: the first that does it, answers and (for a model)
    can run it now; else the first that answers; else the first that does it."""
    able = [s for s in servers() if s.get(role)]
    if model:
        found = next((s for s in able if runs(s, model)), None)
        if found:
            return found
    return next((s for s in able if health(s).get("ok")), able[0] if able else None)


def has(role: str) -> bool:
    return any(s.get(role) for s in servers())


def client(role: str, timeout: float = 900, model: str | None = None):
    """A client (app/clone.py) for the server doing this job, or None without one."""
    from .clone import Client
    server = pick(role, model)
    return Client(server["url"], server.get("token") or "", timeout=timeout) if server else None
