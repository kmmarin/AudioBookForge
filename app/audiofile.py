"""Audiobooks added for Read along: tags, cover and chapters, read with ffprobe,
and the book imported into the library (OUTPUT_DIR, laid out like voiced books).

Files uploaded together that are parts of one audiobook (the same album and
artist tags, or without tags the same name but for their numbers) become one
book. A book that came as one file goes into the library as it is, never
re-encoded. One that came in parts is combined into a single M4B, with a
chapter marker for every part (or every chapter a part has), the book's details
and cover: AAC parts are joined as they are, others encoded once.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import config, parsers

EXTENSIONS = [".m4b", ".m4a", ".mp3", ".aac", ".ogg", ".oga", ".opus", ".flac", ".wav"]
SINGLE_MAX = 75 * 60  # a file without chapters stays one chapter up to this long...
PART_SECONDS = 45 * 60  # ...and is cut into parts about this long beyond it
QUIET_SEARCH = 180  # seconds around each cut to look for a pause in

# Tags people use for the same thing, best first (lowercased ffprobe names).
TITLE_TAGS = ("album", "title")
AUTHOR_TAGS = ("artist", "album_artist", "author", "writer")
NARRATOR_TAGS = ("narrator", "narratedby", "©nrt", "nrt", "performer", "composer")
SERIES_TAGS = ("series", "mvnm", "movementname", "movement_name")
SERIES_INDEX_TAGS = ("series-part", "series_part", "seriespart", "series_index", "mvin", "movementnumber")
YEAR_TAGS = ("date", "year", "originaldate", "original_date", "releasedate", "release_date", "©day")
DESCRIPTION_TAGS = ("description", "synopsis", "ldes", "desc", "summary")
GROUPING_RE = re.compile(r"^(.+?)\s*(?:#|,?\s*book\s+|,?\s*vol\.?\s*)(\d+(?:\.\d+)?)\s*$", re.I)


def is_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in EXTENSIONS


def length_label(seconds: float) -> str:
    """'2 h 38 min', '45 min'"""
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{max(1, minutes)} min"
    return f"{minutes // 60} h {minutes % 60} min"


@dataclass
class AudioBook:
    meta: dict = field(default_factory=dict)
    chapters: list[dict] = field(default_factory=list)  # {"title", "start", "end"} in seconds
    duration: float = 0.0
    cover: bytes | None = None


def probe(path: Path) -> dict:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_chapters",
                           "-of", "json", str(path)], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise parsers.UnsupportedFormat("Forge couldn’t read the sound in this file. Check that it plays.")
    return json.loads(proc.stdout or "{}")


def _tags(info: dict) -> dict:
    """Format and audio stream tags, lowercased; the format's win."""
    tags: dict[str, str] = {}
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            for k, v in (stream.get("tags") or {}).items():
                tags.setdefault(k.lower(), str(v).strip())
    for k, v in (info.get("format", {}).get("tags") or {}).items():
        tags[k.lower()] = str(v).strip()
    return {k: v for k, v in tags.items() if v}


def _first(tags: dict, keys: tuple) -> str:
    return next((tags[k] for k in keys if tags.get(k)), "")


def read_meta(tags: dict, filename: str) -> dict:
    meta: dict = {}
    meta["title"] = parsers.tidy_title(_first(tags, TITLE_TAGS))
    meta["author"] = _first(tags, AUTHOR_TAGS)
    meta["narrator"] = _first(tags, NARRATOR_TAGS)
    if meta["narrator"] and meta["narrator"] == meta["author"]:
        meta.pop("narrator")  # composer often just repeats the artist
    meta["series"] = _first(tags, SERIES_TAGS)
    meta["series_index"] = _first(tags, SERIES_INDEX_TAGS)
    grouping = GROUPING_RE.match(tags.get("grouping", ""))  # Forge writes "Series #2"
    if grouping and not meta["series"]:
        meta["series"], meta["series_index"] = grouping.group(1), grouping.group(2)
    year = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", _first(tags, YEAR_TAGS))
    meta["year"] = year.group(1) if year else ""
    meta["genre"] = "" if tags.get("genre", "").lower() in {"audiobook", "audiobooks", "speech", "other"} \
        else tags.get("genre", "")
    description = _first(tags, DESCRIPTION_TAGS)
    comment = tags.get("comment", "")
    if not description and len(comment) > 60:  # a short comment is usually "Ripped by …"
        description = comment
    meta["description"] = description[:4000]
    meta["publisher"] = _first(tags, ("publisher", "label"))
    meta["isbn"] = parsers.clean_isbn(tags.get("isbn", "")) if tags.get("isbn") else ""
    language = tags.get("language", "")
    if re.fullmatch(r"[a-z]{2,3}", language.lower()) and language.lower() not in {"und", "mul", "zxx"}:
        meta["language"] = language.lower()
    meta = {k: str(v).strip() for k, v in meta.items() if v and str(v).strip()}
    for key, value in parsers.meta_from_filename(Path(filename).stem).items():
        meta.setdefault(key, value)
    return meta


def _cover(path: Path, info: dict) -> bytes | None:
    pics = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
            and (s.get("disposition") or {}).get("attached_pic")]
    if not pics:
        return None
    proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:{pics[0]['index']}",
                           "-frames:v", "1", "-c", "copy", "-f", "image2pipe", "-"],
                          capture_output=True, timeout=60)
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


def quiet_point(path: Path, around: float, span: float = QUIET_SEARCH) -> float:
    """The middle of the longest pause near `around` (seconds), so a part
    never starts mid-word. Only that stretch of the file is decoded."""
    start = max(0.0, around - span / 2)
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-ss", f"{start:.2f}", "-t", f"{span:.2f}",
                           "-i", str(path), "-vn", "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"],
                          capture_output=True, text=True, timeout=120)
    best, best_len = around, 0.0
    for m in re.finditer(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", proc.stderr):
        end, length = float(m.group(1)), float(m.group(2))
        if length > best_len:
            best, best_len = start + end - length / 2, length
    return best


def plan_chapters(path: Path, info: dict, duration: float, title: str) -> list[dict]:
    """The file's own chapters; without them, one chapter, or parts of about
    PART_SECONDS cut at pauses for a long file."""
    chapters = []
    for n, c in enumerate(sorted(info.get("chapters", []), key=lambda c: float(c.get("start_time", 0)))):
        start = max(0.0, float(c.get("start_time", 0)))
        end = min(duration, float(c.get("end_time", duration))) if duration else float(c.get("end_time", 0))
        if end - start < 0.5:
            continue  # a zero-length marker
        tags = {k.lower(): str(v) for k, v in (c.get("tags") or {}).items()}
        name = parsers.tidy_title(tags.get("title", "")) or f"Chapter {len(chapters) + 1}"
        chapters.append({"title": name, "start": round(start, 3), "end": round(end, 3)})
    if len(chapters) > 1:
        # Some files leave a gap or start late; the first chapter takes the opening.
        chapters[0]["start"] = 0.0
        return chapters
    single = chapters[0]["title"] if chapters else (title or "Full book")
    if duration <= SINGLE_MAX:
        return [{"title": single, "start": 0.0, "end": round(duration, 3)}]
    parts = max(2, round(duration / PART_SECONDS))
    cuts = [0.0] + [round(quiet_point(path, duration * k / parts), 3) for k in range(1, parts)] + [round(duration, 3)]
    return [{"title": f"Part {k + 1}", "start": a, "end": b} for k, (a, b) in enumerate(zip(cuts, cuts[1:]))]


def _duration(info: dict) -> float:
    return float(info.get("format", {}).get("duration") or 0) or max(
        (float(s.get("duration") or 0) for s in info.get("streams", []) if s.get("codec_type") == "audio"), default=0)


def _check(info: dict, name: str = "") -> float:
    """The file's length, or UnsupportedFormat when it isn't usable audio."""
    of = f" ({name})" if name else ""
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise parsers.UnsupportedFormat(f"This file has no sound in it{of}. Add the audiobook’s audio file.")
    duration = _duration(info)
    if duration < 1:
        raise parsers.UnsupportedFormat(f"This audio file is empty or too short to be an audiobook{of}.")
    return duration


def read(path: Path, filename: str) -> AudioBook:
    info = probe(path)
    duration = _check(info)
    meta = read_meta(_tags(info), filename)
    chapters = plan_chapters(path, info, duration, meta.get("title", ""))
    return AudioBook(meta=meta, chapters=chapters, duration=duration, cover=_cover(path, info))


# --------------------------------------------------------------------------
# An audiobook in parts
# --------------------------------------------------------------------------

PARTS_DIR = "parts"  # /data/books/<id>/parts/NNN.<ext>, in order, until they're combined
PARTS_JSON = "parts.json"  # the parts' own file names, in order: what the book came as
NUMBERS_RE = re.compile(r"(?i)\b(?:part|pt|cd|disc|disk|track|tr|chapter|ch|file|section)\b\.?|\d+|\bof\b")


def _natural(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _plain(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text or "").strip().lower()


def name_stem(filename: str) -> str:
    """A part's file name without its numbers and part words: "Dune - Part 03" -> "dune"."""
    return _plain(NUMBERS_RE.sub(" ", Path(filename).stem))


def _number(value: str | None) -> int | None:
    m = re.match(r"\s*(\d+)", value or "")
    return int(m.group(1)) if m else None


def group(files: list[tuple[str, Path]]) -> list[list[tuple[str, Path]]]:
    """Audio files uploaded together, as books: the parts of one audiobook (the
    same album and artist tags, or with no album tag the same name but for its
    numbers) together, in order (disc and track tags when every part has them,
    else the file names' natural order). Files that aren't readable audio stay
    on their own, so reading them reports the problem."""
    groups: dict[str, list[tuple[str, Path, dict]]] = {}
    for n, (filename, path) in enumerate(files):
        try:
            tags = _tags(probe(path))
        except parsers.UnsupportedFormat:
            groups[f"unreadable:{n}"] = [(filename, path, {})]
            continue
        album = _plain(tags.get("album", ""))
        key = (f"album:{album}|{_plain(tags.get('artist') or tags.get('album_artist', ''))}" if album
               else f"name:{name_stem(filename)}")
        groups.setdefault(key, []).append((filename, path, tags))
    books = []
    for members in groups.values():
        if all(_number(t.get("track")) is not None for _, _, t in members):
            members.sort(key=lambda m: (_number(m[2].get("disc")) or 0, _number(m[2].get("track")), _natural(m[0])))
        else:
            members.sort(key=lambda m: _natural(m[0]))
        books.append([(filename, path) for filename, path, _ in members])
    return books


def book_name(filenames: list[str], first_tags: dict | None = None) -> str:
    """What an audiobook in parts is called in the list: its album, else the
    name its files share, with how many parts it came in."""
    name = (first_tags or {}).get("album") or ""
    if not name:
        common = os.path.commonprefix([Path(f).stem for f in filenames])
        name = re.sub(r"[\s\-_.,(\[#]*(?:part|pt|cd|disc|track|chapter)?[\s\-_.]*\d*$", "", common, flags=re.I).strip()
    return f"{name or Path(filenames[0]).stem} ({len(filenames)} parts)"


def save_parts(folder: Path, files: list[tuple[str, Path]]) -> None:
    """Move an audiobook's parts into its folder, numbered in order, and note their names."""
    parts = folder / PARTS_DIR
    parts.mkdir(parents=True, exist_ok=True)
    for n, (filename, path) in enumerate(files, 1):
        shutil.move(str(path), parts / f"{n:03d}{Path(filename).suffix.lower()}")
    (folder / PARTS_JSON).write_text(json.dumps([f for f, _ in files], ensure_ascii=False), encoding="utf-8")


def parts_of(folder: Path) -> list[Path]:
    """The parts waiting to be combined, in order; empty for a single file or once combined."""
    return sorted((folder / PARTS_DIR).glob("[0-9][0-9][0-9].*")) if (folder / PARTS_DIR).is_dir() else []


def combined(folder: Path) -> bool:
    """Whether Forge made this book's file (from parts), so it may retag it."""
    return (folder / PARTS_JSON).exists()


def _part_title(n: int, tags: dict, filename: str, album: str, titles_differ: bool) -> str:
    title = parsers.tidy_title(tags.get("title", ""))
    if title and titles_differ and _plain(title) != _plain(album):
        return title
    stem = parsers.tidy_title(re.sub(r"^[\d\s\-_.]+", "", Path(filename).stem))
    if stem and name_stem(stem) and name_stem(stem) != _plain(album) and _plain(stem) != _plain(album):
        return stem
    return f"Part {n}"


def read_parts(paths: list[Path], filenames: list[str]) -> AudioBook:
    """An audiobook in parts read as one: the first part's details, and chapters
    running on from part to part (each part's own chapters, or one per part)."""
    infos = [probe(p) for p in paths]
    lengths = [_check(info, name) for info, name in zip(infos, filenames)]
    tags = [_tags(info) for info in infos]
    # The name the parts share stands in for a file name: "Author - Title" is read from it too.
    meta = read_meta(tags[0], book_name(filenames, tags[0]).rsplit(" (", 1)[0] + ".m4b")
    album = tags[0].get("album", "") or meta.get("title", "")
    titles_differ = len({_plain(t.get("title", "")) for t in tags}) > 1
    chapters, offset = [], 0.0
    for n, (path, info, length, t, filename) in enumerate(zip(paths, infos, lengths, tags, filenames), 1):
        title = _part_title(n, t, filename, album, titles_differ)
        planned = plan_chapters(path, info, length, title)
        if len(planned) > 1 and not info.get("chapters"):  # a long part cut at pauses
            for k, c in enumerate(planned, 1):
                c["title"] = f"{title} ({k})"
        for k, c in enumerate(planned):
            if k == 0 and chapters and chapters[-1]["title"] == c["title"]:
                chapters[-1]["end"] = round(offset + c["end"], 3)  # a chapter running on into the next part
                continue
            chapters.append({"title": c["title"], "start": round(offset + c["start"], 3),
                             "end": round(offset + c["end"], 3)})
        offset += length
    chapters[-1]["end"] = round(offset, 3)
    cover = next((c for c in (_cover(p, i) for p, i in zip(paths, infos)) if c), None)
    return AudioBook(meta=meta, chapters=chapters, duration=offset, cover=cover)


def combine(paths: list[Path], chapters: list[tuple[str, float]], meta: dict, cover: Path | None, out: Path) -> None:
    """One M4B from an audiobook's parts, in order, with a chapter marker for each
    of chapters ((title, seconds), back to back), the book's details and cover.
    AAC parts at one sample rate are joined as they are; anything else is
    encoded once, to AAC at BITRATE."""
    from . import assemble
    streams = []
    for p in paths:
        audio = next(s for s in probe(p).get("streams", []) if s.get("codec_type") == "audio")
        streams.append((audio.get("codec_name"), audio.get("sample_rate"), audio.get("channels")))
    as_is = all(s[0] == "aac" for s in streams) and len(set(streams)) == 1
    fmt = config.FORMATS["m4b"]
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "parts.txt"
        listing.write_text("".join("file '{}'\n".format(str(p).replace("'", "'\\''")) for p in paths),
                           encoding="utf-8")
        meta_file = Path(tmp) / "ffmetadata.txt"
        meta_file.write_text(assemble.ffmetadata(meta, "", chapters, fmt["muxer"]), encoding="utf-8")
        rate = streams[0][1] or "44100"
        audio_args = ["-c:a", "copy"] if as_is else ["-c:a", "aac", "-b:a", config.BITRATE, "-ac", "1", "-ar", str(rate)]
        assemble._mux(["-f", "concat", "-safe", "0", "-i", str(listing)], meta_file, cover, fmt, audio_args, out)


def free_path(out: Path, mine: str | None) -> Path | None:
    """Where an imported book goes: out, unless another file is already there
    (mine: the file this book has now, which may be replaced)."""
    return out if not out.exists() or str(out) == mine else None


def place(src: Path, out: Path, cover: Path | None) -> None:
    """Put a single-file audiobook into the library as it is (moved, not re-encoded),
    with its cover beside it in the Audiobookshelf layout."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    shutil.move(str(src), tmp)
    os.replace(tmp, out)
    if cover is not None and cover.exists() and config.OUTPUT_LAYOUT == "abs":
        shutil.copyfile(cover, out.parent / "cover.jpg")
