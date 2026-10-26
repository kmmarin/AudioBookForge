"""Stitch voiced units into chapters, encode chapters in parallel, and mux the
final tagged file with chapter markers and cover art."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config


def ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-1200:]}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return float(proc.stdout.strip() or 0)


def concat_wavs(paths: list[Path], out: Path, tail_seconds: float) -> tuple[float, int]:
    """Join same-format WAVs without re-encoding. Returns (seconds, sample rate)."""
    with wave.open(str(paths[0]), "rb") as first:
        channels, width, rate = first.getnchannels(), first.getsampwidth(), first.getframerate()
    frames = 0
    with wave.open(str(out), "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(width)
        dst.setframerate(rate)
        for path in paths:
            with wave.open(str(path), "rb") as src:
                data = src.readframes(src.getnframes())
                dst.writeframes(data)
                frames += src.getnframes()
        tail = int(rate * tail_seconds)
        dst.writeframes(b"\x00" * tail * width * channels)
        frames += tail
    return frames / rate, rate


def safe_name(value: str, limit: int = 120) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", value or "")
    value = re.sub(r"\s+", " ", value).strip().strip(".")
    return value[:limit].strip() or "Untitled"


def output_path(meta: dict, ext: str) -> Path:
    title = safe_name(meta.get("title") or "Untitled")
    author = safe_name(meta.get("author") or "Unknown Author")
    if config.OUTPUT_LAYOUT == "flat":
        return config.OUTPUT_DIR / f"{author} - {title}.{ext}"
    folder = config.OUTPUT_DIR / author
    leaf = title
    if meta.get("series"):
        folder = folder / safe_name(meta["series"])
        if meta.get("series_index"):
            leaf = f"{meta['series_index']} - {title}"
    return folder / safe_name(leaf) / f"{title}.{ext}"


def _esc(value) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", str(value))


def ffmetadata(meta: dict, narrator: str, chapters: list[tuple[str, float]], muxer: str) -> str:
    title = meta.get("title") or "Untitled"
    author = meta.get("author") or ""
    tags = {
        "title": title, "album": title, "artist": author, "album_artist": author,
        "composer": meta.get("narrator") or narrator, "genre": meta.get("genre") or "Audiobook",
        "date": meta.get("year"), "comment": (meta.get("description") or "")[:1000],
        "description": meta.get("description"), "publisher": meta.get("publisher"),
        "language": meta.get("language"),
    }
    if meta.get("series"):
        tags["grouping"] = f"{meta['series']} #{meta['series_index']}" if meta.get(
            "series_index") else meta["series"]
    if muxer == "ipod":
        tags["media_type"] = "2"  # "Audiobook" in Apple's players
    lines = [";FFMETADATA1"]
    lines += [f"{k}={_esc(v)}" for k, v in tags.items() if v]
    start = 0.0
    for name, seconds in chapters:
        end = start + seconds
        lines += ["", "[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}",
                  f"END={int(end * 1000)}", f"title={_esc(name)}"]
        start = end
    return "\n".join(lines) + "\n"


def codec_args(fmt: dict, rate: int) -> list[str]:
    args: list[str] = []
    if config.NORMALIZE:
        args += ["-af", "loudnorm=I=-18:TP=-2:LRA=11", "-ar", str(rate)]
    if fmt["codec"] == "aac":
        return args + ["-c:a", "aac", "-b:a", config.BITRATE, "-ac", "1"]
    if fmt["codec"] == "libmp3lame":
        return args + ["-c:a", "libmp3lame", "-b:a", config.BITRATE, "-ac", "1"]
    if fmt["codec"] == "libopus":
        return args + ["-c:a", "libopus", "-b:a", config.OPUS_BITRATE, "-ac", "1"]
    return args + ["-c:a", "flac"]


def build(book: dict, chapters: list[dict], unit_paths: dict[int, list[Path]],
          workdir: Path, narrator: str, cover: Path | None) -> tuple[Path, list[float]]:
    """Returns (output file, chapter durations in seconds)."""
    fmt = config.FORMATS[book["fmt"]]
    chapter_dir = workdir / "chapters"
    shutil.rmtree(chapter_dir, ignore_errors=True)
    chapter_dir.mkdir(parents=True)

    def make_wav(ch):
        out = chapter_dir / f"{ch['idx']:04d}.wav"
        seconds, rate = concat_wavs(unit_paths[ch["idx"]], out, config.CHAPTER_TAIL)
        return out, seconds, rate

    with ThreadPoolExecutor(config.ENCODE_WORKERS) as pool:
        wavs = list(pool.map(make_wav, chapters))
        rate = wavs[0][2]
        if fmt["mode"] == "segments":
            # Encode every chapter at once, then join the encoded audio without re-encoding.
            seg_ext = "m4a" if fmt["codec"] == "aac" else "mp3"

            def encode(item):
                wav, _, _ = item
                seg = wav.with_suffix(f".{seg_ext}")
                ffmpeg(["-i", str(wav), *codec_args(fmt, rate), str(seg)])
                wav.unlink()
                return seg, probe_duration(seg)

            encoded = list(pool.map(encode, wavs))
            inputs = [seg for seg, _ in encoded]
            durations = [seconds for _, seconds in encoded]
        else:
            inputs = [wav for wav, _, _ in wavs]
            durations = [seconds for _, seconds, _ in wavs]

    concat_list = chapter_dir / "concat.txt"
    concat_list.write_text("".join(
        "file '{}'\n".format(str(p).replace("'", "'\\''")) for p in inputs))
    meta_file = chapter_dir / "ffmetadata.txt"
    meta_file.write_text(ffmetadata(
        book["meta"], narrator, [(c["title"], d) for c, d in zip(chapters, durations)],
        fmt["muxer"]), encoding="utf-8")

    out = output_path(book["meta"], fmt["ext"])
    audio_args = ["-c:a", "copy"] if fmt["mode"] == "segments" else codec_args(fmt, rate)
    _mux(["-f", "concat", "-safe", "0", "-i", str(concat_list)], meta_file, cover, fmt, audio_args, out)
    shutil.rmtree(chapter_dir, ignore_errors=True)
    return out, durations


def _mux(audio_input: list[str], meta_file: Path, cover: Path | None, fmt: dict,
         audio_args: list[str], out: Path) -> None:
    """Write the finished file: audio, tags and chapter markers, cover art."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    use_cover = cover is not None and cover.exists() and fmt["cover"]
    args = [*audio_input, "-i", str(meta_file)]
    if use_cover:
        args += ["-i", str(cover)]
    args += ["-map", "0:a", "-map_metadata", "1",
             "-map_chapters", "1" if fmt["chapters"] else "-1"]
    if use_cover:
        args += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    args += audio_args
    if fmt["muxer"] == "ipod":
        args += ["-movflags", "+faststart"]
    if fmt["muxer"] == "mp3":
        args += ["-id3v2_version", "3"]
    args += ["-f", fmt["muxer"], str(tmp)]
    ffmpeg(args)
    os.replace(tmp, out)
    if use_cover and config.OUTPUT_LAYOUT == "abs" and out.parent != cover.parent:
        shutil.copyfile(cover, out.parent / "cover.jpg")


def retag(src: Path, out: Path, meta: dict, narrator: str, chapters: list[tuple[str, float]],
          fmt_id: str, cover: Path | None) -> None:
    """Copy a finished file with today's details, chapter titles and cover,
    without re-encoding: a few seconds even for a long book."""
    fmt = config.FORMATS[fmt_id]
    with tempfile.TemporaryDirectory() as tmp:
        meta_file = Path(tmp) / "ffmetadata.txt"
        meta_file.write_text(ffmetadata(meta, narrator, chapters, fmt["muxer"]), encoding="utf-8")
        _mux(["-i", str(src)], meta_file, cover, fmt, ["-c:a", "copy"], out)


def remove_output(path: Path) -> None:
    """Delete a finished file from the library, and its folders once they're empty
    (a cover.jpg left alone in the book's folder goes too)."""
    path.unlink(missing_ok=True)
    folder = path.parent
    if config.OUTPUT_LAYOUT == "abs" and folder.is_dir() and [p.name for p in folder.iterdir()] == ["cover.jpg"]:
        (folder / "cover.jpg").unlink()
    while folder != config.OUTPUT_DIR and folder.is_relative_to(config.OUTPUT_DIR):
        try:
            folder.rmdir()  # only when empty
        except OSError:
            break
        folder = folder.parent
