"""Export a voice creator project's clips for Qwen3-TTS: 24 kHz mono WAVs cut
straight from the original recordings (not the 22.05 kHz training copies),
train.jsonl in Qwen's fine-tuning format, and one reference clip.

Runs in the voice creator container (CPU only):
    docker exec -i -e PYTHONPATH=/srv -w /tmp audiobook-forge-voicelab-1 \
        python3 - <project id> <out dir> < export_clips.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from lab import db, media

RATE = 24000
pid, out = sys.argv[1], Path(sys.argv[2])
(out / "wav").mkdir(parents=True, exist_ok=True)
project = db.get_project(pid)
paths = {s["id"]: s["path"] for s in db.sources(pid)}
clips = [c for c in db.clips(pid) if c["role"] in ("train", "holdout")]

# The reference: a held-out clip of 8-14 s with Whisper's most confident transcript.
refs = [c for c in clips if c["role"] == "holdout" and 8 <= c["end"] - c["start"] <= 14] or \
       [c for c in clips if 8 <= c["end"] - c["start"] <= 14]
ref = max(refs, key=lambda c: (c["conf"] or 0) - (c["unsure"] or 0))

rows, seconds = {"train": [], "holdout": []}, 0.0
for n, c in enumerate(sorted(clips, key=lambda c: (c["source_id"], c["start"])), 1):
    name = f"c{c['id']}.wav"
    path = out / "wav" / name
    if not path.exists():
        audio = media.decode(paths[c["source_id"]], c["start"], c["end"] - c["start"], rate=RATE)
        media.write_wav(path, audio, RATE)
    seconds += c["end"] - c["start"]
    rows[c["role"]].append({"audio": f"wav/{name}", "text": c["text"].strip(), "ref_audio": "ref.wav"})
    if n % 100 == 0:
        print(f"{n} of {len(clips)} clips", flush=True)

ref_audio = media.decode(paths[ref["source_id"]], ref["start"], ref["end"] - ref["start"], rate=RATE)
media.write_wav(out / "ref.wav", ref_audio, RATE)
(out / "ref.txt").write_text(ref["text"].strip() + "\n", encoding="utf-8")
for role, items in rows.items():
    with open(out / f"{role}.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
(out / "info.json").write_text(json.dumps({
    "name": project["name"], "project": pid, "accent": project["accent"], "rate": RATE,
    "train_clips": len(rows["train"]), "holdout_clips": len(rows["holdout"]),
    "minutes": round(seconds / 60, 1), "ref_seconds": round(len(ref_audio) / RATE, 1),
    "ref_peak": float(np.abs(ref_audio).max())}, indent=1), encoding="utf-8")
print(f"{project['name']}: {len(rows['train'])} train + {len(rows['holdout'])} held-out clips, "
      f"{seconds / 60:.1f} min; reference {len(ref_audio) / RATE:.1f} s: {ref['text'].strip()}")
