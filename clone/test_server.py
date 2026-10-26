"""End-to-end check of a running clone server, through Forge's own client
(app/clone.py). The CPU is fine for it: the sentences are short.

    PYTHONPATH=<repo> python clone/test_server.py URL REF_WAV REF_TXT [TOKEN] [OUT_DIR]

With a TOKEN, the server must have been started with that CLONE_TOKEN.
OUT_DIR, if given, gets the spoken sentences as WAVs to listen to.
"""
import base64
import json
import sys
import time
from pathlib import Path

import soundfile as sf

from app.clone import Client, ref_key

MODEL = "qwen3-tts-0.6b"
url, ref_wav, ref_txt = sys.argv[1:4]
token = sys.argv[4] if len(sys.argv) > 4 else ""
out_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else None
text = Path(ref_txt).read_text(encoding="utf-8").strip()
audio = Path(ref_wav).read_bytes()
key = ref_key(audio, text)
client = Client(url, token, timeout=1800)
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
    results.append(ok)


health = client.health()
check("health", health["ok"] and any(m["id"] == MODEL and m["fits"] for m in health["models"]),
      json.dumps({k: health[k] for k in ("device", "gpu", "gpu_gb", "dtype", "loaded")}))

if token:
    anonymous = Client(url, "", timeout=30)
    status, out = anonymous._call("GET", f"/api/refs/{key}")
    check("token required", status == 401, out.get("detail", ""))
    status, _ = anonymous._call("GET", "/api/health")
    check("health needs no token", status == 200)
    status, _ = Client(url, "wrong", timeout=30)._call("POST", "/api/unload")
    check("wrong token refused", status == 401)

status, out = client._call("POST", "/api/speak", {"model": MODEL, "ref": "0" * 40, "sentences": ["Hello."]})
check("unknown reference clip is 404", status == 404, out.get("detail", ""))
status, out = client._call("PUT", f"/api/refs/{'1' * 40}", {"text": text, "audio": base64.b64encode(audio).decode()})
check("a key that doesn't match is refused", status == 400, out.get("detail", ""))

# Speaking: the client sends the reference clip on the server's 404, then speaks.
t0 = time.time()
rate, clips = client.speak(MODEL, ref_wav, text, ["He drove north.", "It was cold."])
took = time.time() - t0
seconds = sum(len(c) for c in clips) / rate
check("speak two sentences", len(clips) == 2 and all(len(c) > 0.3 * rate for c in clips),
      f"{seconds:.1f} s of audio at {rate} Hz in {took:.1f} s, model load included")
status, _ = client._call("GET", f"/api/refs/{key}")
check("reference clip kept", status == 200)
t0 = time.time()
rate, again = client.speak(MODEL, ref_wav, text, ["Then he stopped."])
took = time.time() - t0
check("speak again, model loaded", len(again) == 1 and len(again[0]) > 0.3 * rate,
      f"{len(again[0]) / rate:.1f} s of audio in {took:.1f} s ({len(again[0]) / rate / took:.2f}x real time)")
if out_dir:
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, clip in enumerate(clips + again, 1):
        sf.write(out_dir / f"clone-test-{n}.wav", clip, rate)

status, out = client._call("POST", "/api/speak", {"model": "nope", "ref": key, "sentences": ["Hi."]})
check("unknown model is 400", status == 400, out.get("detail", ""))
status, out = client._call("POST", "/api/speak", {"model": MODEL, "ref": key, "sentences": ["x" * 1001]})
check("too long a sentence is 400", status == 400, out.get("detail", ""))
status, out = client._call("POST", "/api/speak", {"model": MODEL, "ref": key, "sentences": ["Hi."] * 33})
check("too many sentences is 400", status == 400, out.get("detail", ""))

client.unload()
check("unload", client.health()["loaded"] is None)
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
