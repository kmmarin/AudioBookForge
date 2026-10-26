"""Talking to a clone server (clone/server.py): voices cloned from a reference
clip of a narrator with Qwen3-TTS, on whichever machine has the GPU for them
(CLONE_URL). Forge voices books with it; the voice creator makes samples.

The server's API (JSON; X-Forge-Token when it has a token set):
    GET  /api/health            {"ok", "device", "gpu", "gpu_gb", "dtype", "loaded", "models": [...]}
                                each model: {"id", "label", "hf", "fits", "note"}
    PUT  /api/refs/<key>        {"text", "audio": base64 WAV}: a reference clip, key = ref_key()
    GET  /api/refs/<key>        200 if the server has it, else 404
    POST /api/speak             {"model", "ref", "sentences": [...], "language"}
                                -> {"rate", "audio": [base64 16-bit WAV per sentence]}; 404 = unknown ref
    POST /api/unload            frees the model's video memory
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

TOKEN_HEADER = "X-Forge-Token"
MODELS = {  # what voice.json "model" can say, best first
    "qwen3-tts-1.7b": "Qwen3-TTS 1.7B",
    "qwen3-tts-0.6b": "Qwen3-TTS 0.6B",
}


class CloneError(RuntimeError):
    """The clone server couldn't be reached or couldn't make the audio."""


def ref_key(audio: bytes, text: str) -> str:
    """A reference clip's name on the server: the same clip and text, the same key."""
    return hashlib.sha1(audio + b"\0" + text.strip().encode("utf-8")).hexdigest()


class Client:
    def __init__(self, url: str, token: str = "", timeout: float = 900):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None,
              timeout: float | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        if self.token:
            headers[TOKEN_HEADER] = self.token
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read() or b"{}")
            except ValueError:
                detail = {}
            return exc.code, detail if isinstance(detail, dict) else {"detail": str(detail)}
        except (urllib.error.URLError, OSError) as exc:
            raise CloneError(f"The clone server at {self.url} isn’t answering ({exc}). "
                             "Check that it’s running and CLONE_URL is right.") from None

    def health(self, timeout: float = 5) -> dict:
        status, out = self._call("GET", "/api/health", timeout=timeout)
        if status != 200:
            raise CloneError(out.get("detail") or f"The clone server answered {status}")
        return out

    def speak(self, model: str, ref_audio: Path | str, ref_text: str, sentences: list[str],
              language: str = "English") -> tuple[int, list[np.ndarray]]:
        """Each sentence in the cloned voice: (sample rate, one float32 array per sentence)."""
        import soundfile as sf
        audio = Path(ref_audio).read_bytes()
        key = ref_key(audio, ref_text)
        body = {"model": model, "ref": key, "sentences": sentences, "language": language}
        status, out = self._call("POST", "/api/speak", body)
        if status == 404:  # the server hasn't got this reference clip yet (or lost it): send it
            status, up = self._call("PUT", f"/api/refs/{key}",
                                    {"text": ref_text.strip(), "audio": base64.b64encode(audio).decode()})
            if status != 200:
                raise CloneError(up.get("detail") or f"Couldn’t send the reference clip ({status})")
            status, out = self._call("POST", "/api/speak", body)
        if status != 200:
            raise CloneError(out.get("detail") or f"The clone server answered {status}")
        clips = []
        for item in out["audio"]:
            wav, _ = sf.read(io.BytesIO(base64.b64decode(item)), dtype="float32", always_2d=False)
            clips.append(wav if wav.ndim == 1 else wav.mean(axis=1))
        return int(out["rate"]), clips

    def unload(self) -> None:
        self._call("POST", "/api/unload", timeout=30)
