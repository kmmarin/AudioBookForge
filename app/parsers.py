"""Turn any supported document into metadata plus ordered chapters.

Routing:
  .epub                        -> ebooklib (spine + table of contents)
  .pdf                         -> pypdf (bookmarks, else heading detection)
  .txt .text .asc              -> heading detection (Gutenberg aware)
  .docx .odt .rtf .html .md …  -> pandoc, split on document headings
  .mobi .azw3 .lit …           -> Calibre ebook-convert to EPUB (if installed)
  .doc                         -> antiword (if installed)
"""
from __future__ import annotations

import os
import posixpath
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from . import config


@dataclass
class Chapter:
    title: str
    text: str
    include: bool = True


@dataclass
class ParsedBook:
    meta: dict = field(default_factory=dict)
    chapters: list[Chapter] = field(default_factory=list)
    cover: bytes | None = None


class UnsupportedFormat(Exception):
    pass


PANDOC_FORMATS = {
    ".docx": "docx", ".odt": "odt", ".rtf": "rtf", ".html": "html", ".htm": "html",
    ".xhtml": "html", ".md": "markdown", ".markdown": "markdown", ".fb2": "fb2",
    ".rst": "rst", ".org": "org", ".tex": "latex", ".latex": "latex",
    ".textile": "textile", ".wiki": "mediawiki", ".mediawiki": "mediawiki",
    ".dbk": "docbook", ".docbook": "docbook", ".opml": "opml", ".ipynb": "ipynb",
    ".muse": "muse", ".t2t": "t2t",
}
TEXT_EXT = {".txt", ".text", ".asc"}
CALIBRE_EXT = {".mobi", ".azw", ".azw3", ".prc", ".pdb", ".lit", ".lrf", ".chm",
               ".pml", ".snb", ".tcr", ".rb"}


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def supported_extensions() -> list[str]:
    exts = {".epub", ".pdf"} | TEXT_EXT
    if _has("pandoc"):
        exts |= set(PANDOC_FORMATS)
    if _has("ebook-convert"):
        exts |= CALIBRE_EXT
    if _has("antiword"):
        exts.add(".doc")
    return sorted(exts)


def parse(path: Path, filename: str) -> ParsedBook:
    ext = Path(filename).suffix.lower() or path.suffix.lower()
    if ext == ".epub":
        book = parse_epub(path)
    elif ext == ".pdf":
        book = parse_pdf(path)
    elif ext in TEXT_EXT:
        book = parse_text(read_text(path))
    elif ext == ".doc" and _has("antiword"):
        book = parse_text(_run(["antiword", "-w", "0", str(path)]))
    elif ext in PANDOC_FORMATS and _has("pandoc"):
        book = parse_pandoc(path, PANDOC_FORMATS[ext])
    elif ext in CALIBRE_EXT and _has("ebook-convert"):
        book = parse_with_calibre(path)
    else:
        hint = " Kindle and other ebook formats need INSTALL_CALIBRE=true." if ext in CALIBRE_EXT else ""
        raise UnsupportedFormat(f"{ext or 'This file type'} isn’t supported.{hint}")
    return _finalize(book, filename)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 900) -> str:
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[-600:]}")
    return proc.stdout


def read_text(path: Path) -> str:
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        text = str(best) if best else data.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
JUNK_RE = re.compile(
    r"\s*[\[(](?:z-?lib(?:\.org)?|ebook|e-book|epub|mobi|azw3|pdf|retail|v\d+(?:\.\d+)?|calibre|libgen)[\])]",
    re.I,
)
SERIES_RE = re.compile(
    r"\s*[\[(]([^\[\]()]+?)[\s,]*(?:#|book|bk\.?|vol\.?|volume|no\.?)\s*(\d+(?:\.\d+)?)[\])]", re.I
)


def meta_from_filename(stem: str) -> dict:
    """Guess metadata from names like 'Author - Title (Series #2)' or 'Title by Author'."""
    s = JUNK_RE.sub("", stem.replace("_", " "))
    s = re.sub(r"\s{2,}", " ", s).strip()
    meta: dict = {}
    m = SERIES_RE.search(s)
    if m:
        meta["series"], meta["series_index"] = m.group(1).strip(), m.group(2)
        s = (s[: m.start()] + s[m.end():]).strip()
    m = re.match(r"^(.+?)\s+by\s+(.+)$", s, re.I)
    if m:
        meta["title"], meta["author"] = m.group(1).strip(), m.group(2).strip()
    elif " - " in s:
        author, title = s.split(" - ", 1)
        meta["author"], meta["title"] = author.strip(), title.strip()
    else:
        meta["title"] = s
    return meta


def tidy_title(title: str) -> str:
    title = " ".join((title or "").split()).strip(" .:-—")
    if title.isupper() and len(title) > 3:
        title = " ".join(w if ROMAN_RE.match(w.strip(".:,;")) else w.capitalize()
                         for w in title.split())
    return title[:200]


SKIP_TITLES = {"cover", "contents", "table of contents", "index", "copyright",
               "copyright page", "title page", "half title", "dedication", "imprint",
               "landmarks", "colophon", "toc", "document outline", "outline", "navigation"}
SKIP_PHRASES = re.compile(
    r"\b(also by|other (?:books|titles) by|acknowledg(?:e)?ments?|about the (?:author|publisher)|"
    r"newsletter|praise for|sign up|books by)\b", re.I)


def _norm_words(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS_WORDS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
ORDINAL_WORDS = {w: i for i, w in enumerate(
    "_ first second third fourth fifth sixth seventh eighth ninth tenth".split()) if w != "_"}
HEADING_WORDS = {"chapter": "chapter", "chap": "chapter", "part": "part", "book": "book",
                 "section": "section", "volume": "volume", "vol": "volume", "no": "", "number": ""}
TITLE_PARTS_RE = re.compile(r"(\s*(?::|\.|—|–| - )\s+)")


def _canon_heading(text: str) -> str:
    """A heading reduced to what it says, so different spellings compare equal:
    'CHAPTER ONE' and 'Chapter 1' -> 'chapter 1'; 'I' -> '1'."""
    from .textprep import roman_to_int
    words = _norm_words(text).split()
    headingish = any(w in HEADING_WORDS for w in words) or len(words) == 1
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if w in HEADING_WORDS:
            if HEADING_WORDS[w]:
                out.append(HEADING_WORDS[w])
        elif w.isdigit():
            out.append(str(int(w)))
        elif w in TENS_WORDS:
            n = TENS_WORDS[w]
            if i + 1 < len(words) and NUMBER_WORDS.get(words[i + 1], 10) < 10:
                n += NUMBER_WORDS[words[i + 1]]
                i += 1
            out.append(str(n))
        elif w in NUMBER_WORDS or w in ORDINAL_WORDS:
            out.append(str(NUMBER_WORDS.get(w, ORDINAL_WORDS.get(w))))
        elif headingish and ROMAN_RE.match(w.upper()):
            out.append(str(roman_to_int(w)))
        else:
            out.append(w)
        i += 1
    return " ".join(out)


def _same_heading(a: str, b: str) -> bool:
    """Canonical headings that say the same thing. 'chapter 1' matches '1', but
    'part 1' doesn't match 'chapter 1'."""
    if not a or not b:
        return False
    if a == b:
        return True
    kinds = set(HEADING_WORDS.values()) - {""}
    strip = lambda s: " ".join(w for w in s.split() if w not in kinds)
    kind = lambda s: next((w for w in s.split() if w in kinds), None)
    return strip(a) == strip(b) and strip(a) != "" and (kind(a) is None or kind(b) is None)


def dedupe_title(title: str) -> str:
    """Drop parts of a chapter title that repeat an earlier part in another form:
    'Chapter 1: Chapter One' -> 'Chapter 1', 'ONE: One' -> 'ONE',
    'Chapter 1 Chapter 1' -> 'Chapter 1'."""
    pieces = TITLE_PARTS_RE.split(title)  # text, separator, text, ...
    kept, seen = [pieces[0]], [_canon_heading(pieces[0])]
    for sep, part in zip(pieces[1::2], pieces[2::2]):
        canon = _canon_heading(part)
        if any(_same_heading(canon, s) for s in seen):
            continue
        seen.append(canon)
        kept += [sep, part]
    title = "".join(kept).strip()
    words = title.split()
    half = len(words) // 2
    if half and len(words) % 2 == 0:
        first, second = _canon_heading(" ".join(words[:half])), _canon_heading(" ".join(words[half:]))
        # Numbered headings only: "New York, New York" is a real title.
        if _same_heading(first, second) and re.search(r"\d", first):
            title = " ".join(words[:half])
    return title


def _repeats_title(title: str, line: str) -> bool:
    """A short line that is the chapter title in any form ('ONE' or
    'Chapter One: The Storm' under a title of 'Chapter 1')."""
    n, t = _norm_words(line), _norm_words(title)
    if not n or not t or len(line) >= 120:
        return False
    if n == t or t.startswith(n) or t.endswith(n) or n.startswith(t):
        return True
    c = _canon_heading(line)
    parts = [p for p in TITLE_PARTS_RE.split(title)[::2] if p.strip()]
    return any(_same_heading(c, p) for p in (_canon_heading(title), *map(_canon_heading, parts)) if p)


def _drop_repeated_title(title: str, text: str) -> str:
    """Remove up to two leading heading paragraphs that repeat the chapter title."""
    for _ in range(2):
        first, sep, rest = text.partition("\n\n")
        if not (sep and _repeats_title(title, first)):
            break
        text = rest
    return text


def _finalize(book: ParsedBook, filename: str) -> ParsedBook:
    guessed = meta_from_filename(Path(filename).stem)
    meta = {k: str(v).strip() for k, v in book.meta.items() if v and str(v).strip()}
    for key, value in guessed.items():
        meta.setdefault(key, value)
    if meta.get("date") and not meta.get("year"):
        m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", meta["date"])
        if m:
            meta["year"] = m.group(1)
    meta.pop("date", None)
    if meta.get("series_index"):
        try:
            num = float(meta["series_index"])
            meta["series_index"] = str(int(num)) if num.is_integer() else str(num)
        except ValueError:
            pass

    chapters: list[Chapter] = []
    single = len(book.chapters) == 1 and book.chapters[0].title == "Full text"
    for ch in book.chapters:
        if single and meta.get("title"):
            ch.title = meta["title"]  # a short story or essay: its chapter is the work itself
        title = dedupe_title(tidy_title(ch.title)) or f"Chapter {len(chapters) + 1}"
        text = _drop_repeated_title(title, ch.text.strip()).strip()
        if not text:
            continue
        words = len(text.split())
        skip = title.lower() in SKIP_TITLES or bool(SKIP_PHRASES.search(title)) or words < 40
        chapters.append(Chapter(title, text, ch.include and not skip))
    if not chapters:
        raise ValueError("No readable text was found in this file.")
    if not any(c.include for c in chapters):
        for c in chapters:
            c.include = True
    if not meta.get("isbn"):
        isbn = find_isbn(c.text for c in book.chapters)
        if isbn:
            meta["isbn"] = isbn
    return ParsedBook(meta=meta, chapters=chapters, cover=book.cover)


ISBN_RE = re.compile(r"\bISBN(?:-1[03])?[\s:]*((?:97[89][\s‐–-]?)?(?:\d[\s‐–-]?){9}[\dXx])\b")


def clean_isbn(value: str) -> str | None:
    """A valid ISBN-10 or ISBN-13 from text like 'urn:isbn:978-0-7653-2635-5'."""
    digits = re.sub(r"[^0-9Xx]", "", value or "").upper()
    if len(digits) == 13 and digits.isdigit() and digits[:3] in ("978", "979"):
        check = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:12]))
        return digits if (10 - check % 10) % 10 == int(digits[12]) else None
    if len(digits) == 10 and digits[:9].isdigit():
        total = sum((10 - i) * (10 if d == "X" else int(d)) for i, d in enumerate(digits))
        return digits if total % 11 == 0 else None
    return None


def find_isbn(texts) -> str | None:
    """First valid ISBN printed in the book, normally on the copyright page.
    Only the front of the book is searched: the back often lists other titles."""
    seen = 0
    for text in texts:
        for match in ISBN_RE.finditer(text):
            isbn = clean_isbn(match.group(1))
            if isbn:
                return isbn
        seen += len(text)
        if seen > 60_000:
            break
    return None


# --------------------------------------------------------------------------
# Plain text and heading detection
# --------------------------------------------------------------------------

NUM_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
             "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
             "fifty|sixty|seventy|eighty|ninety|hundred|first|second|third|fourth|fifth|"
             "sixth|seventh|eighth|ninth|tenth|last|final")
NUMBERED_RE = re.compile(
    rf"^(?:chapter|chap\.?|part|book|volume|vol\.?|section|letter|canto|stave)\s+"
    rf"(?:\d+|[ivxlcdm]+\b|(?:{NUM_WORDS})(?:[\s-](?:{NUM_WORDS}))*\b)", re.I)
STANDALONE_RE = re.compile(
    r"^(?:prologue|epilogue|interlude|introduction|preface|foreword|afterword|"
    r"appendix|postscript|coda)\b", re.I)
NUMERAL_RE = re.compile(r"^(?:[IVXLCDM]{1,7}|\d{1,3})\.?$")
MD_HEAD_RE = re.compile(r"^#{1,3}\s+(.+)$")
GUT_START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)
GUT_END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.I | re.M)


def gutenberg_meta(text: str) -> dict:
    m = GUT_START.search(text)
    if not m:
        return {}
    head = text[: m.start()]
    meta = {}
    for key, label in (("title", "Title"), ("author", "Author"), ("language", "Language")):
        found = re.search(rf"^{label}:\s*(.+)$", head, re.M)
        if found:
            meta[key] = found.group(1).strip()
    return meta


def strip_gutenberg(text: str) -> str:
    m = GUT_START.search(text)
    if m:
        text = text[m.end():]
    m = GUT_END.search(text)
    if m:
        text = text[: m.start()]
    return text


def text_paragraphs(text: str) -> list[str]:
    lines = text.split("\n")
    blank = sum(1 for line in lines if not line.strip())
    if len(lines) > 20 and blank < len(lines) * 0.05:
        return [line.strip() for line in lines if line.strip()]  # one paragraph per line
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def unwrap(paragraph: str) -> str:
    return " ".join(line.strip() for line in paragraph.splitlines() if line.strip())


def heading_kind(raw: str) -> str | None:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines or len(lines) > 3 or any(len(line) > 80 for line in lines):
        return None
    first = lines[0].strip("*_ ")
    if MD_HEAD_RE.match(first):
        return "md"
    words = first.split()
    sentence_like = first.endswith((".", "!", "?", ",", ";")) and len(words) > 3
    if NUMBERED_RE.match(first) and len(words) <= 10 and not sentence_like:
        return "kw"
    if STANDALONE_RE.match(first) and len(words) <= 6 and not sentence_like:
        return "kw"
    if NUMERAL_RE.match(first) and len(lines) <= 2:
        return "num"
    return None


def split_by_headings(paras: list[str]) -> list[Chapter]:
    found = [(i, kind) for i, raw in enumerate(paras) if (kind := heading_kind(raw))]
    counts = Counter(kind for _, kind in found)
    use: set[str] = set()
    if counts["kw"] + counts["md"] >= 2:
        use = {"kw", "md"}
    elif counts["num"] >= 3:
        use = {"num"}
    heads = [(i, k) for i, k in found if k in use]
    if len(heads) < 2:
        return []

    chapters: list[Chapter] = []
    front = paras[: heads[0][0]]
    if front:
        chapters.append(Chapter("Front matter", "\n\n".join(unwrap(p) for p in front), include=False))
    for n, (i, kind) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(paras)
        lines = [line.strip().strip("*_#").strip() for line in paras[i].splitlines() if line.strip()]
        title = ": ".join(lines)
        body_start = i + 1
        # "CHAPTER ONE" followed by a short subtitle paragraph
        if len(lines) == 1 and body_start < end:
            nxt = paras[body_start].strip()
            if ("\n" not in nxt and len(nxt) < 70 and not heading_kind(nxt)
                    and not nxt.endswith((".", "!", "?", ",", ";", ":", "\"", "”"))
                    and not nxt.startswith(("\"", "“", "'"))):
                title = f"{title}: {nxt}"
                body_start += 1
        body = "\n\n".join(unwrap(p) for p in paras[body_start:end])
        chapters.append(Chapter(title, body))
    return chapters


def split_by_length(paras: list[str], words_per_part: int = 4500) -> list[Chapter]:
    """Fallback when a book has no detectable chapters: ~30-minute parts."""
    parts, current, count = [], [], 0
    for p in paras:
        current.append(unwrap(p))
        count += len(p.split())
        if count >= words_per_part:
            parts.append(current)
            current, count = [], 0
    if current:
        parts.append(current)
    if len(parts) == 1:
        return [Chapter("Full text", "\n\n".join(parts[0]))]
    return [Chapter(f"Part {n}", "\n\n".join(p)) for n, p in enumerate(parts, 1)]


def parse_text(text: str) -> ParsedBook:
    meta = gutenberg_meta(text)
    paras = text_paragraphs(strip_gutenberg(text))
    return ParsedBook(meta=meta, chapters=split_by_headings(paras) or split_by_length(paras))


# --------------------------------------------------------------------------
# Pandoc: Word, ODT, RTF, HTML, Markdown, FB2, LaTeX, …
# --------------------------------------------------------------------------

HEAD_MARK_RE = re.compile(r"^@@H(\d)@@\s*(.*)$", re.S)


def parse_pandoc(path: Path, fmt: str) -> ParsedBook:
    out = _run([
        "pandoc", "-f", fmt, "-t", "plain", "--wrap=none",
        "--lua-filter", str(config.APP_DIR / "pandoc" / "flatten.lua"),
        "--template", str(config.APP_DIR / "pandoc" / "meta.plain"),
        str(path),
    ])
    head, _, body = out.partition("@@BODY@@")
    meta = {}
    for line in head.splitlines():
        if line.startswith("@@META "):
            key, _, value = line[7:].partition("=")
            if value.strip():
                meta[key] = value.strip()
    if fmt in ("docx", "odt"):
        for key, value in _office_properties(path).items():
            meta.setdefault(key, value)
    body = strip_gutenberg(body)
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    paras = ["* * *" if p == "@@BREAK@@" else p for p in paras]

    heads = [(i, int(m.group(1)), " ".join(m.group(2).split()))
             for i, p in enumerate(paras) if (m := HEAD_MARK_RE.match(p))]
    levels = Counter(level for _, level, _ in heads)
    eligible = {lvl: n for lvl, n in levels.items() if n >= 2}
    if not eligible:
        plain = [HEAD_MARK_RE.sub(r"\2", p) for p in paras]
        chapters = split_by_headings(plain) or split_by_length(plain)
        return ParsedBook(meta=meta, chapters=chapters)

    split_level = max(eligible, key=lambda lvl: (eligible[lvl], -lvl))
    book_title = _norm_words(meta.get("title", ""))

    def role(level: int, text: str) -> str:
        if level < split_level and book_title and _norm_words(text) == book_title:
            return "skip"  # the book's own title, not a part
        if level == split_level or (level < split_level and levels[level] >= 2):
            return "split"  # chapters, and parts that group them
        if level < split_level and (not book_title or _norm_words(text) == book_title):
            return "skip"  # a one-off top heading is the book title
        return "speak"

    chapters: list[Chapter] = []
    title, buf, prefix = None, [], ""

    def flush():
        nonlocal title, buf, prefix
        if buf:
            chapters.append(Chapter(f"{prefix}{title or 'Opening'}", "\n\n".join(buf),
                                    include=title is not None))
            prefix = ""
        elif title:  # a heading with no text of its own, e.g. "Part One"
            prefix = f"{prefix}{title}: "
        buf = []

    for p in paras:
        m = HEAD_MARK_RE.match(p)
        kind = role(int(m.group(1)), m.group(2)) if m else None
        if kind == "split":
            flush()
            title = " ".join(m.group(2).split())
        elif kind == "speak":
            buf.append(" ".join(m.group(2).split()))  # sub-heading: read as text
        elif kind == "skip":
            continue
        else:
            buf.append(" ".join(p.split()) if p != "* * *" else p)
    flush()
    return ParsedBook(meta=meta, chapters=chapters)


def _office_properties(path: Path) -> dict:
    """Title and author from a .docx or .odt file's document properties."""
    import zipfile
    from xml.etree import ElementTree
    names = {"title": "title", "creator": "author", "initial-creator": "author",
             "subject": "genre", "description": "description"}
    meta: dict = {}
    try:
        with zipfile.ZipFile(path) as zf:
            for member in ("docProps/core.xml", "meta.xml"):
                if member not in zf.namelist():
                    continue
                for node in ElementTree.fromstring(zf.read(member)).iter():
                    key = names.get(node.tag.rsplit("}", 1)[-1])
                    if key and node.text and node.text.strip():
                        meta.setdefault(key, node.text.strip())
    except (zipfile.BadZipFile, ElementTree.ParseError, OSError):
        pass
    return meta


# --------------------------------------------------------------------------
# EPUB
# --------------------------------------------------------------------------

BLOCK_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
              "pre", "dt", "dd", "figcaption", "section", "article", "td"]
BLOCK_SET = set(BLOCK_TAGS)
REMOVE_SELECTORS = ["script", "style", "nav", "aside", "img", "svg", "table", "sup", "head",
                    ".pg-boilerplate",  # Project Gutenberg header and license
                    '[epub\\:type~="footnote"]', '[epub\\:type~="endnote"]',
                    '[epub\\:type~="noteref"]', '[role="doc-noteref"]',
                    '[role="doc-footnote"]', '[role="doc-endnote"]']


def _soup(html: bytes):
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for selector in REMOVE_SELECTORS:
        try:
            for el in soup.select(selector):
                el.decompose()
        except Exception:
            continue
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return soup


def _block_text(el) -> str:
    return " ".join(el.get_text("").split())


def _lead_bold(el) -> str:
    """Bold text opening a paragraph: '4. Title' in <p><a/><b>4. Title</b> First words…</p>."""
    for child in el.children:
        name = getattr(child, "name", None)
        if name is None:
            if str(child).strip():
                return ""
            continue
        text = _block_text(child)
        if text:
            return text if name in ("b", "strong") else ""
    return ""


def _walk(soup, fragments: dict[str, str]):
    """Yield ('start', id) and ('text', text, tag) in document order.

    An anchor inside a paragraph (<p><a id="p29"/><b>4. Title</b> First words…)
    starts its chapter before that paragraph's text, not after it, and a title
    set in bold at its start becomes a heading of its own."""
    body = soup.body or soup
    started: set[str] = set()

    def starts(nodes):
        for node in nodes:
            for attr in ("id", "name"):
                value = node.get(attr)
                if value and value in fragments and value not in started:
                    started.add(value)
                    yield value

    for el in body.descendants:
        name = getattr(el, "name", None)
        if not name:
            continue
        leaf = name in BLOCK_SET and not el.find(BLOCK_TAGS)
        began = list(starts([el, *el.find_all(True)] if leaf else [el])) if fragments else []
        for value in began:
            yield ("start", value)
        if leaf:
            text = _block_text(el)
            head = _lead_bold(el) if began else ""
            if head and text.startswith(head) and len(text) > len(head) and _repeats_title(fragments[began[-1]], head):
                yield ("text", head, "h2")
                text = text[len(head):].strip()
            if text:
                yield ("text", text, name)


DIVISION_RE = re.compile(
    r"^(?:part|book|volume|vol\.?|stave|act|section|canto|chapter)\b|^[IVXLCDM]{1,7}\.?$|^\d{1,3}\.?$",
    re.I)


def _caption_ids(html: bytes, ids: set[str]) -> set[str]:
    """TOC targets that are illustration captions, not chapter starts.

    Gutenberg (and others) list every illustration in the TOC; the target is a
    heading right after an image whose alt text matches, or a figure caption.
    """
    if not ids:
        return set()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    found = set()
    for target in ids:
        el = soup.find(id=target) or soup.find(attrs={"name": target})
        if el is None:
            continue
        if el.name == "figcaption" or el.find_parent("figure") or el.find_parent("figcaption"):
            found.add(target)
            continue
        prev = el.find_previous_sibling()
        img = prev if prev is not None and prev.name == "img" else (
            prev.find("img") if prev is not None else None)
        if img is not None and _norm_words(img.get("alt") or "") == _norm_words(el.get_text(" ")):
            found.add(target)
    return found


def _flatten_toc(toc, parent=None):
    """Yield (href, title, parent href) for every entry, depth first."""
    for entry in toc:
        if isinstance(entry, (list, tuple)):
            section = entry[0]
            children = entry[1] if len(entry) > 1 else []
            href = getattr(section, "href", None)
            yield href, getattr(section, "title", None), parent
            yield from _flatten_toc(children, href or parent)
        else:
            yield getattr(entry, "href", None), getattr(entry, "title", None), parent


def _norm_href(href: str) -> str:
    return posixpath.normpath(unquote(href).replace("\\", "/")).lstrip("./")


def _epub_meta(book) -> dict:
    from bs4 import BeautifulSoup

    def dc(name):
        return [v[0] for v in book.get_metadata("DC", name) if v and v[0]]

    meta = {}
    if dc("title"):
        meta["title"] = dc("title")[0]
    if dc("creator"):
        meta["author"] = " & ".join(dict.fromkeys(dc("creator")))
    if dc("date"):
        meta["date"] = dc("date")[0]
    if dc("description"):
        meta["description"] = " ".join(
            BeautifulSoup(dc("description")[0], "lxml").get_text(" ").split())
    if dc("subject"):
        meta["genre"] = dc("subject")[0]
    if dc("language"):
        meta["language"] = dc("language")[0]
    if dc("publisher"):
        meta["publisher"] = dc("publisher")[0]
    for identifier in dc("identifier"):
        isbn = clean_isbn(identifier)
        if isbn:
            meta["isbn"] = isbn
            break
    for namespace in book.metadata.values():
        for entries in namespace.values():
            for value, attrs in entries:
                attrs = attrs or {}
                key = attrs.get("name") or attrs.get("property")
                if key == "calibre:series" and attrs.get("content"):
                    meta["series"] = attrs["content"]
                elif key == "calibre:series_index" and attrs.get("content"):
                    meta["series_index"] = attrs["content"]
                elif key == "belongs-to-collection" and value:
                    meta.setdefault("series", value)
                elif key == "group-position" and value:
                    meta.setdefault("series_index", value)
    return meta


def _epub_cover(book) -> bytes | None:
    import ebooklib
    candidates = []
    for _, attrs in book.get_metadata("OPF", "cover"):
        item = book.get_item_with_id((attrs or {}).get("content", ""))
        if item is not None:
            candidates.append(item)
    candidates += list(book.get_items_of_type(ebooklib.ITEM_COVER))
    candidates += [i for i in book.get_items_of_type(ebooklib.ITEM_IMAGE)
                   if "cover" in f"{i.get_name()} {i.get_id()}".lower()]
    for item in candidates:
        data = item.get_content()
        if data and len(data) > 2000 and not data.lstrip().startswith(b"<"):
            return data
    return None


def parse_epub(path: Path) -> ParsedBook:
    import warnings

    import ebooklib
    from ebooklib import epub

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(str(path), options={"ignore_ncx": False})

    def toc_key(href: str) -> tuple[str, str]:
        file, _, frag = href.partition("#")
        return os.path.basename(_norm_href(file)), unquote(frag)

    toc_full: dict[str, list[tuple[str, str]]] = {}
    toc_base: dict[str, list[tuple[str, str]]] = {}
    parent_of: dict[tuple[str, str], tuple[str, str]] = {}
    for href, title, parent in _flatten_toc(book.toc):
        if not href:
            continue
        file, _, frag = href.partition("#")
        entry = (unquote(frag), (title or "").strip())
        toc_full.setdefault(_norm_href(file), []).append(entry)
        toc_base.setdefault(os.path.basename(_norm_href(file)), []).append(entry)
        if parent:
            parent_of.setdefault(toc_key(href), toc_key(parent))

    chapters: list[Chapter] = []
    keys: list[tuple[str, str] | None] = []  # TOC entry behind each chapter
    for spine_entry in book.spine:
        idref, linear = (spine_entry if isinstance(spine_entry, (list, tuple))
                         else (spine_entry, "yes"))
        if str(linear).lower() == "no":
            continue
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        name = _norm_href(item.get_name())
        base = os.path.basename(name)
        entries = toc_full.get(name) or toc_base.get(base) or []
        html = item.get_content()
        captions = _caption_ids(html, {f for f, _ in entries if f})
        frag_titles = {}
        for frag, title in entries:
            if frag and frag not in frag_titles and frag not in captions:
                frag_titles[frag] = title
        file_title = next((t for f, t in entries if not f and t), None)
        file_key = (base, "")
        if not file_title and len(frag_titles) == 1:
            file_key = (base, next(iter(frag_titles)))
            file_title = frag_titles[file_key[1]]
        split_at = frag_titles if len(frag_titles) >= 2 else {}

        soup = _soup(html)
        for target in captions:
            el = soup.find(id=target) or soup.find(attrs={"name": target})
            if el is not None:
                el.decompose()  # an illustration's caption isn't read aloud

        segments: list[list] = []
        seg = [file_title, [], None, file_key]  # title, texts, first heading, TOC key
        for event in _walk(soup, split_at):
            if event[0] == "start":
                if seg[1] or seg[0]:
                    segments.append(seg)
                seg = [split_at[event[1]], [], None, (base, event[1])]
            else:
                _, text, tag = event
                if seg[2] is None and tag in ("h1", "h2", "h3"):
                    seg[2] = text
                seg[1].append(text)
        segments.append(seg)

        for title, texts, heading, key in segments:
            if not texts:
                continue
            body = "\n\n".join(texts)
            label = title or heading
            if label is None and chapters:
                chapters[-1].text += "\n\n" + body  # chapter split across files
            else:
                chapters.append(Chapter(label or f"Section {len(chapters) + 1}", body))
                keys.append(key if title else None)

    # A TOC entry whose own text is just its heading ("Stave One", "Part Two")
    # names the chapters under it: "Stave One: Marley’s Ghost".
    merged: list[Chapter] = []
    for i, ch in enumerate(chapters):
        kids = [j for j in range(i + 1, len(chapters))
                if keys[j] is not None and parent_of.get(keys[j]) == keys[i]]
        if (keys[i] is not None and kids and len(ch.text.split()) <= 12
                and DIVISION_RE.match(ch.title.strip())):
            prefix = " ".join(ch.title.split()).strip(" .:-—")
            for j in kids:
                chapters[j].title = f"{prefix}: {chapters[j].title}"
            continue
        merged.append(ch)

    return ParsedBook(meta=_epub_meta(book), chapters=merged, cover=_epub_cover(book))


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)


def _strip_running_lines(pages: list[str]) -> list[str]:
    """Remove headers and footers that repeat on many pages."""
    if len(pages) < 6:
        return pages

    def key(line):
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter = Counter()
    for page in pages:
        lines = [line for line in page.split("\n") if line.strip()]
        for line in set(lines[:2] + lines[-2:]):
            counts[key(line)] += 1
    repeated = {k for k, n in counts.items() if n > len(pages) * 0.4 and k}
    cleaned = []
    for page in pages:
        lines = page.split("\n")
        edge = {i for i, line in enumerate(lines) if line.strip()}
        edge_idx = sorted(edge)[:2] + sorted(edge)[-2:]
        cleaned.append("\n".join(
            line for i, line in enumerate(lines) if not (i in edge_idx and key(line) in repeated)))
    return cleaned


def _pdf_paragraphs(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n")]
    widths = sorted(len(line) for line in lines if line)
    median = widths[len(widths) // 2] if widths else 60
    paras, cur = [], []

    def flush():
        if cur:
            paras.append(" ".join(cur))
            cur.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if PAGE_NUM_RE.match(line):
            continue
        if heading_kind(line) and len(line) < 60:
            flush()
            paras.append(line)
            continue
        if cur and cur[-1].endswith("-") and line[:1].islower():
            cur[-1] = cur[-1][:-1] + line
        else:
            cur.append(line)
        if line.endswith((".", "!", "?", "\"", "”", "’", ":")) and len(line) < 0.85 * median:
            flush()
    flush()
    return paras


def _pdf_outline(reader) -> list[tuple[int, str]]:
    try:
        outline = reader.outline
    except Exception:
        return []
    levels: dict[int, list[tuple[int, str]]] = {}

    def walk(items, depth):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                page = reader.get_destination_page_number(item)
            except Exception:
                continue
            if page is not None and depth <= 1:
                levels.setdefault(depth, []).append((page, str(item.title or "").strip()))

    walk(outline, 0)
    usable = {d: entries for d, entries in levels.items() if len(entries) >= 3}
    if not usable:
        return []
    entries = sorted(usable[max(usable, key=lambda d: len(usable[d]))])
    deduped, seen = [], set()
    for page, title in entries:
        if page not in seen:
            seen.add(page)
            deduped.append((page, title))
    return deduped


def _pdf_cover(path: Path) -> bytes | None:
    if not _has("pdftoppm"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "cover"
        try:
            _run(["pdftoppm", "-jpeg", "-f", "1", "-l", "1", "-scale-to", "1400",
                  "-singlefile", str(path), str(target)], timeout=60)
            return target.with_suffix(".jpg").read_bytes()
        except Exception:
            return None


def parse_pdf(path: Path) -> ParsedBook:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    if sum(len(p.strip()) for p in pages) < 200:
        raise ValueError("This PDF has no text layer (it’s scanned images). "
                         "Run it through OCR first, for example with ocrmypdf.")
    pages = _strip_running_lines(pages)

    meta = {}
    info = reader.metadata
    if info:
        if info.title:
            meta["title"] = str(info.title)
        if info.author:
            meta["author"] = str(info.author)

    outline = _pdf_outline(reader)
    if outline:
        chapters = []
        if outline[0][0] > 0:
            front = _pdf_paragraphs("\n".join(pages[: outline[0][0]]))
            chapters.append(Chapter("Front matter", "\n\n".join(front), include=False))
        for n, (start, title) in enumerate(outline):
            end = outline[n + 1][0] if n + 1 < len(outline) else len(pages)
            paras = _pdf_paragraphs("\n".join(pages[start:end]))
            chapters.append(Chapter(title, "\n\n".join(paras)))
    else:
        paras = _pdf_paragraphs("\n\n".join(pages))
        chapters = split_by_headings(paras) or split_by_length(paras)
    return ParsedBook(meta=meta, chapters=chapters, cover=_pdf_cover(path))


# --------------------------------------------------------------------------
# Calibre (Kindle and legacy ebook formats)
# --------------------------------------------------------------------------

def parse_with_calibre(path: Path) -> ParsedBook:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "book.epub"
        _run(["ebook-convert", str(path), str(target)], timeout=900)
        return parse_epub(target)
