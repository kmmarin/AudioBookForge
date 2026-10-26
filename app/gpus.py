"""GPU servers: the machines running the clone server (clone/server.py) that
speak cloned voices.

The list is kept in GPU_SERVERS_FILE, in the shared models volume so Forge and
the voice creator read the same one, and changed on Forge's Settings page:

    {"servers": [{"id": "a1b2c3d4", "name": "gaming-pc", "url": "http://192.168.1.30:8100",
                  "token": "…", "voices": true}]}

A server set in .env (CLONE_URL, with CLONE_TOKEN or SHARE_TOKEN) is listed
too, after them; it's changed in .env. A job goes to the first server that is
answering (checked at most every HEALTH_SECONDS), else to the first, so its
error says what's wrong. So one server can stand in for another.

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


def load() -> list[dict]:
    """The servers added on the Settings page, in order."""
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    # Every server speaks voices (one saved for pictures only, before they were dropped, too).
    return [{**s, "voices": True, "source": "settings"}
            for s in data.get("servers") or [] if isinstance(s, dict) and s.get("url")]


def save(servers: list[dict]) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    keep = ("id", "name", "url", "token") + ROLES
    tmp = FILE.with_name(f".{FILE.name}.part")
    tmp.write_text(json.dumps({"servers": [{k: s.get(k) for k in keep} for s in servers]}, indent=1),
                   encoding="utf-8")
    os.chmod(tmp, 0o600)  # it holds the servers' tokens
    os.replace(tmp, FILE)
    _health.clear()


def servers() -> list[dict]:
    """Every server, Settings' first, then .env's (unless Settings has the same address)."""
    found = load()
    env = _env_server()
    if env and not any(s["url"].rstrip("/") == env["url"] for s in found):
        found.append(env)
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


def pick(role: str) -> dict | None:
    """The server for a job: the first that does it and answers, else the first that does it."""
    able = [s for s in servers() if s.get(role)]
    return next((s for s in able if health(s).get("ok")), able[0] if able else None)


def has(role: str) -> bool:
    return any(s.get(role) for s in servers())


def client(role: str, timeout: float = 900):
    """A client (app/clone.py) for the server doing this job, or None without one."""
    from .clone import Client
    server = pick(role)
    return Client(server["url"], server.get("token") or "", timeout=timeout) if server else None
