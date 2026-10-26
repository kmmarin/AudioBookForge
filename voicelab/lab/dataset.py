"""Writing the chosen clips out as Piper's training set: one WAV per clip
(22.05 kHz mono, 16-bit) and a metadata.csv of "file|text" lines, plus the
held-out clips the quality check compares against."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config, db, media

WINDOW = 300.0  # seconds decoded at once when cutting clips


def project_dir(project_id: str) -> Path:
    return config.PROJECTS_DIR / project_id


def clean_text(text: str) -> str:
    """One line of plain text for espeak: no column separators or stray spaces,
    and no straight double quotes. Piper reads metadata.csv with Python's csv
    module, where a line starting with " swallows the lines after it (dialogue
    in a novel often does); espeak doesn't voice quotes anyway."""
    return re.sub(r"\s+", " ", text.replace("|", " ").replace('"', "")).strip()


def cut(clips: list[dict], out_dir: Path) -> None:
    """Write each clip as out_dir/c<id>.wav, decoding each recording a
    window at a time; clips already written are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips if not (out_dir / f"c{c['id']}.wav").exists()]
    paths = {s["id"]: s["path"] for s in db.sources(clips[0]["project_id"])} if clips else {}
    todo.sort(key=lambda c: (c["source_id"], c["start"]))
    i = 0
    while i < len(todo):
        first = todo[i]
        group = [first]
        while (i + len(group) < len(todo) and todo[i + len(group)]["source_id"] == first["source_id"]
               and todo[i + len(group)]["end"] - first["start"] <= WINDOW):
            group.append(todo[i + len(group)])
        start = first["start"]
        audio = media.decode(paths[first["source_id"]], start, group[-1]["end"] - start + 0.05)
        for c in group:
            a = int((c["start"] - start) * config.SAMPLE_RATE)
            b = int((c["end"] - start) * config.SAMPLE_RATE)
            media.write_wav(out_dir / f"c{c['id']}.wav", audio[a:b])
        i += len(group)


def teach_rows(project_id: str) -> list[dict]:
    """The sentences a cloned voice reads for Piper to learn from: the
    narrator's own, from every clip heard but the held-out ones (only the
    words count here, not how the clip sounded), the surest transcripts
    first, up to TEACH_MINUTES of them."""
    rows = [dict(r, project_id=project_id) for r in db.clips(project_id)
            if r["role"] != "holdout" and r["reason"] != "duplicate"
            and len(clean_text(r["text"] or "").split()) >= config.MIN_WORDS]
    rows.sort(key=lambda r: (r["role"] != "train", -(r["conf"] or 0.0)))
    out, seconds = [], 0.0
    for r in rows:
        if seconds >= config.TEACH_MINUTES * 60:
            break
        out.append(r)
        seconds += r["end"] - r["start"]
    return out


def speak(rows: list[dict], out_dir: Path, teacher: dict, report=None) -> list[dict]:
    """Write each row's sentence as out_dir/c<id>.wav, read by the cloned voice
    (teacher: its "model", "ref" clip and "text"); rows already done are
    skipped, so a restart carries on. A take that ran on past its text or
    stopped short (Qwen does, now and then) is left out. Returns the rows
    written. report(done, total) after each request."""
    from .clone import client as clone_client
    out_dir.mkdir(parents=True, exist_ok=True)
    client = clone_client()

    def done(r: dict) -> bool:
        return (out_dir / f"c{r['id']}.wav").exists() or (out_dir / f"c{r['id']}.skip").exists()

    todo = sorted((r for r in rows if not done(r)), key=lambda r: len(r["text"]))  # like lengths batch well
    made = len(rows) - len(todo)
    for i in range(0, len(todo), config.TEACH_PER_REQUEST):
        part = todo[i:i + config.TEACH_PER_REQUEST]
        texts = [clean_text(r["text"]) for r in part]
        rate, clips = client.speak(teacher["model"], teacher["ref"], teacher["text"], texts)
        for r, text, audio in zip(part, texts, clips):
            if len(text) / 40 <= len(audio) / rate <= 3 + len(text) / 5:
                media.write_wav(out_dir / f"c{r['id']}.wav", media.resample(audio, rate, config.SAMPLE_RATE))
            else:
                (out_dir / f"c{r['id']}.skip").touch()
        made += len(part)
        if report:
            report(made, len(rows))
    return [r for r in rows if (out_dir / f"c{r['id']}.wav").exists()]


def build(project_id: str, round_no: int, teacher: dict | None = None, report=None) -> dict:
    """The training set for this round, and the held-out set. Returns where they are.
    teacher: the cloned voice reads the sentences (speak) instead of clips being
    cut from the recordings; the held-out clips are the real narrator either way."""
    rows = [dict(r, project_id=project_id) for r in db.clips(project_id) if r["role"] in ("train", "holdout")]
    train = [r for r in rows if r["role"] == "train"]
    holdout = [r for r in rows if r["role"] == "holdout"]
    if len(train) < 10:
        raise ValueError("There aren’t enough usable clips to train on yet. Add more recordings of the narrator.")
    base = project_dir(project_id)
    data_dir = base / "dataset" / f"round-{round_no}"
    wav_dir = data_dir / "wav"
    if teacher:
        train = speak(teach_rows(project_id), wav_dir, teacher, report)
        if len(train) < 10:
            raise ValueError("The cloned voice couldn’t read enough sentences to learn from.")
    else:
        cut(train, wav_dir)
    lines = [f"c{r['id']}.wav|{clean_text(r['text'])}" for r in sorted(train, key=lambda r: r["id"])]
    (data_dir / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_holdout(project_id, holdout)
    return {"dir": str(data_dir), "csv": str(data_dir / "metadata.csv"), "wav": str(wav_dir),
            "clips": len(train), "holdout": len(holdout), "taught": bool(teacher),
            "max_clip": max(r["id"] for r in rows)}


def write_holdout(project_id: str, holdout: list[dict] | None = None) -> int:
    """Cut the held-out clips (never trained on: voices are measured against
    them) into holdout/, listed in holdout.json. Returns how many there are."""
    if holdout is None:
        holdout = [dict(r, project_id=project_id) for r in db.clips(project_id) if r["role"] == "holdout"]
    hold_dir = project_dir(project_id) / "holdout"
    cut(holdout, hold_dir)
    (hold_dir / "holdout.json").write_text(json.dumps(
        [{"id": r["id"], "text": clean_text(r["text"]), "file": f"c{r['id']}.wav"}
         for r in sorted(holdout, key=lambda r: r["id"])], indent=1), encoding="utf-8")
    return len(holdout)
