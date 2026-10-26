"""Look up a book's details online and match them to the file.

Sources, all queried in parallel:
  openlibrary  original publication year, subjects; exact match by ISBN
  audible      series and number in series, description, genre, large cover
  google       description, categories (only with GOOGLE_BOOKS_API_KEY: the
               keyless quota is shared worldwide and usually used up)

Every result is scored against the file's title and author. A book's fields
then come from the best-scoring result of the source that's best at each.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from . import config

log = logging.getLogger("forge.lookup")
USER_AGENT = "AudiobookForge/1.0 (self-hosted audiobook maker)"
FIELDS = ("title", "author", "series", "series_index", "year", "genre", "publisher", "description")
# Which source to trust first for each field.
PREFER = {
    "series": ("audible", "openlibrary", "google"),
    "series_index": ("audible", "openlibrary", "google"),
    "description": ("audible", "google", "openlibrary"),
    "genre": ("audible", "google", "openlibrary"),
    "year": ("openlibrary", "google", "audible"),  # first publication, not this edition
    # Open Library's publisher is whichever edition comes first, often a translation,
    # and Audible's is the audio publisher, so only Google's edition data is used.
    "publisher": ("google",),
    "cover_url": ("audible", "openlibrary", "google"),
}
COVER_HOSTS = {"covers.openlibrary.org", "m.media-amazon.com", "books.google.com",
               "books.googleusercontent.com"}
AUDIBLE_HOSTS = {"com": "api.audible.com", "co.uk": "api.audible.co.uk", "ca": "api.audible.ca",
                 "com.au": "api.audible.com.au", "de": "api.audible.de", "fr": "api.audible.fr",
                 "it": "api.audible.it", "es": "api.audible.es", "in": "api.audible.in",
                 "co.jp": "api.audible.co.jp"}
# Open Library subjects are noisy; only these become a genre.
GENRES = {
    "fantasy", "epic fantasy", "science fiction", "mystery", "detective and mystery stories",
    "thriller", "thrillers", "suspense", "horror", "romance", "historical fiction", "fiction",
    "classics", "classic literature", "literary fiction", "adventure", "adventure stories",
    "young adult", "young adult fiction", "children's fiction", "juvenile fiction", "humor",
    "short stories", "poetry", "drama", "biography", "autobiography", "memoir", "history",
    "philosophy", "psychology", "self-help", "business", "science", "true crime",
    "christmas stories", "ghost stories", "fairy tales", "dystopian", "satire", "war",
}


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[’'`]", "", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"^(the|a|an) ", "", text)


EDITION_RE = re.compile(
    r"\s*[(\[][^)\]]*\b(?:edition|unabridged|abridged|dramati[sz]ed|version|audiobook|annotated)\b[^)\]]*[)\]]",
    re.I)


def clean_title(title: str) -> str:
    """'A Christmas Carol (Blackstone Edition)' -> 'A Christmas Carol'."""
    return EDITION_RE.sub("", title or "").strip()


def main_title(title: str) -> str:
    """'A Christmas Carol in Prose; Being a Ghost Story' -> 'A Christmas Carol in Prose'."""
    return re.split(r"\s*[:;(\[]\s*|\s+[-–—]\s+", title or "", maxsplit=1)[0].strip()


def title_score(want: str, got: str) -> float:
    a, b = _norm(main_title(want)), _norm(main_title(got))
    if not a or not b:
        return 0.0
    main = SequenceMatcher(None, a, b).ratio()
    shorter, longer = sorted((a, b), key=len)
    if f" {shorter} " in f" {longer} " and len(shorter) >= 4:
        main = max(main, 0.9)  # "christmas carol" inside "christmas carol in prose"
    # Mostly the main title, a little of the full one: "Project Hail Mary: How
    # Rocky Eats" is a different book from "Project Hail Mary".
    full = SequenceMatcher(None, _norm(want), _norm(got)).ratio()
    return max(0.85 * main + 0.15 * full, full)


def author_score(want: str, got: str) -> float:
    if not want:
        return 0.5  # nothing to compare: neither helps nor hurts
    if not got:
        return 0.0
    last = lambda name: {_norm(p).split()[-1] for p in re.split(r"\s*(?:&|,| and )\s*", name) if _norm(p)}
    if last(want) & last(got):
        return 1.0
    return SequenceMatcher(None, _norm(want), _norm(got)).ratio()


LANGUAGES = {"english": "en", "spanish": "es", "german": "de", "french": "fr", "italian": "it",
             "portuguese": "pt", "dutch": "nl", "japanese": "ja", "polish": "pl", "swedish": "sv"}


def score(meta: dict, cand: dict) -> float:
    want, got, sub = meta.get("title", ""), cand.get("title", ""), cand.get("subtitle") or ""
    # "Mistborn" is listed as "The Final Empire: Mistborn Book 1".
    titles = [got] + ([sub, f"{got}: {sub}"] if sub else [])
    t_score = max(title_score(want, t) for t in titles)
    a_score = author_score(meta.get("author", ""), cand.get("author", ""))
    # An ISBN is decisive only when the title or author agrees too: copyright
    # pages also print the ISBNs of other editions, other books and placeholders.
    if meta.get("isbn") and meta["isbn"] in cand.get("isbns", ()) and (
            t_score >= 0.6 or a_score >= 0.9 or not want):
        return 1.0
    value = 0.65 * t_score + 0.35 * a_score
    if meta.get("author") and cand.get("author") and a_score < 0.75:
        value -= 0.25  # same title, someone else's book
    if meta.get("isbn") and meta["isbn"] in cand.get("isbns", ()) and value < 0.6:
        value = 0.0  # the ISBN pointed at an unrelated book
    lang = cand.get("language")
    if lang and LANGUAGES.get(lang.lower(), lang[:2].lower()) != (meta.get("language") or "en")[:2].lower():
        value -= 0.15  # another language's edition
    if cand.get("ai_narrated"):
        value -= 0.3  # low-effort listings with boilerplate descriptions
    return round(max(0.0, value), 3)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def _get_json(url: str):
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=config.LOOKUP_TIMEOUT) as resp:
        return json.load(resp)


def _text(html: str) -> str:
    """Plain text from a source's HTML description."""
    if not html:
        return ""
    from bs4 import BeautifulSoup
    for tag in ("<br>", "<br/>", "<br />", "</p>"):
        html = html.replace(tag, "\n")
    text = BeautifulSoup(html, "lxml").get_text()
    text = re.sub(r"\n\s*(?:©|\(P\)|℗).*$", "", text.strip(), flags=re.S)  # trailing copyright
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()[:4000]


def _year(value) -> str | None:
    m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", str(value or ""))
    return m.group(1) if m else None


def _openlibrary_genre(subjects: list[str]) -> str | None:
    for subject in subjects or []:
        if subject.strip().lower() in GENRES:
            return subject.strip().capitalize() if subject.islower() else subject.strip()
    return None


def search_openlibrary(title: str, author: str, isbn: str | None) -> list[dict]:
    fields = "key,title,subtitle,author_name,first_publish_year,publisher,subject,cover_i,isbn,edition_count"
    if isbn:
        query = {"isbn": isbn, "fields": fields, "limit": 1}
        docs = _get_json("https://openlibrary.org/search.json?" + urlencode(query)).get("docs", [])
    else:
        docs = []
    if not docs:
        query = {"title": main_title(title), "fields": fields, "limit": 10}
        if author:
            query["author"] = author
        docs = _get_json("https://openlibrary.org/search.json?" + urlencode(query)).get("docs", [])
    out = []
    for d in docs:
        out.append({
            "source": "openlibrary", "title": clean_title(d.get("title", "")),
            "author": " & ".join(d.get("author_name", [])[:2]),
            "year": _year(d.get("first_publish_year")),
            "publisher": (d.get("publisher") or [None])[0],
            "genre": _openlibrary_genre(d.get("subject")),
            "cover_url": f"https://covers.openlibrary.org/b/id/{d['cover_i']}-L.jpg?default=false" if d.get("cover_i") else None,
            "isbns": set(d.get("isbn", [])), "work": d.get("key"),
            "popularity": d.get("edition_count") or 0,
            "link": f"https://openlibrary.org{d['key']}" if d.get("key") else None,
        })
    return out


def _openlibrary_description(work_key: str) -> str | None:
    work = _get_json(f"https://openlibrary.org{work_key}.json")
    desc = work.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")
    return _text(desc) or None if desc else None


def search_audible(title: str, author: str, isbn: str | None) -> list[dict]:
    host = AUDIBLE_HOSTS.get(config.AUDIBLE_REGION, "api.audible.com")
    common = {"num_results": 8, "products_sort_by": "Relevance", "image_sizes": "1000,500",
              "response_groups": "product_desc,product_extended_attrs,series,contributors,"
                                 "product_attrs,media,category_ladders,rating"}
    # A title search alone misses books whose file title is longer than the
    # listing ("A Christmas Carol in Prose" vs "A Christmas Carol"); a keyword
    # search finds those. Both run, and duplicates are dropped.
    by_title = {"title": main_title(title), **({"author": author} if author else {})}
    by_keywords = {"keywords": f"{main_title(title)} {author}".strip()}
    products, seen = [], set()
    for query in (by_title, by_keywords):
        data = _get_json(f"https://{host}/1.0/catalog/products?" + urlencode({**query, **common}))
        for p in data.get("products", []):
            if p.get("asin") not in seen:
                seen.add(p.get("asin"))
                products.append(p)
    out = []
    for p in products:
        # A book can sit in several series ("The Stormlight Archive" 1 and the
        # unnumbered "The Cosmere"); the numbered one is the series that matters.
        all_series = p.get("series") or [{}]
        series = next((s for s in all_series if s.get("sequence")), all_series[0])
        narrators = " ".join(n.get("name", "") for n in p.get("narrators", []))
        summary = p.get("publisher_summary") or p.get("merchandising_summary") or ""
        ladders = [[c["name"] for c in l.get("ladder", [])] for l in p.get("category_ladders", [])]
        genre = None
        if ladders and ladders[0]:
            genre = ladders[0][1] if len(ladders[0]) > 1 else ladders[0][0]
        images = p.get("product_images") or {}
        out.append({
            "source": "audible", "title": clean_title(p.get("title", "")), "subtitle": p.get("subtitle"),
            "author": " & ".join(a["name"] for a in p.get("authors", [])[:2]),
            "series": series.get("title"), "series_index": series.get("sequence") or None,
            "year": _year(p.get("release_date")), "publisher": p.get("publisher_name"),
            "genre": genre,
            "description": _text(summary) or None,
            "cover_url": images.get("1000") or images.get("500"),
            "language": p.get("language"),
            "popularity": ((p.get("rating") or {}).get("overall_distribution") or {}).get("num_ratings") or 0,
            "ai_narrated": bool(re.search(r"virtual voice|\bAI voice\b", narrators, re.I)
                                or re.search(r"narrated by an? (?:AI|virtual)|AI[- ]narrated", summary, re.I)),
            "link": f"https://www.audible.{config.AUDIBLE_REGION}/pd/{p['asin']}" if p.get("asin") else None,
        })
    return out


def search_google(title: str, author: str, isbn: str | None) -> list[dict]:
    if not config.GOOGLE_BOOKS_API_KEY:
        return []
    q = f"isbn:{isbn}" if isbn else f'intitle:"{main_title(title)}"' + (f' inauthor:"{author}"' if author else "")
    url = ("https://www.googleapis.com/books/v1/volumes?q=" + quote(q)
           + f"&maxResults=5&printType=books&key={config.GOOGLE_BOOKS_API_KEY}")
    out = []
    for item in _get_json(url).get("items", []):
        v = item.get("volumeInfo", {})
        cover = (v.get("imageLinks") or {}).get("thumbnail")
        out.append({
            "source": "google", "title": clean_title(v.get("title", "")), "subtitle": v.get("subtitle"),
            "author": " & ".join(v.get("authors", [])[:2]),
            "year": _year(v.get("publishedDate")), "publisher": v.get("publisher"),
            "genre": (v.get("categories") or [None])[0],
            "description": _text(v.get("description")) or None,
            "cover_url": cover.replace("http://", "https://").replace("&edge=curl", "") if cover else None,
            "isbns": {i["identifier"] for i in v.get("industryIdentifiers", []) if "identifier" in i},
            "link": v.get("infoLink"),
        })
    return out


def rank(cand: dict) -> float:
    """Match score plus a little for popularity (Audible ratings, Open Library
    editions): a well-known listing beats a slightly closer one-off reissue or
    junk upload, but never a clearly better match."""
    return cand["score"] + 0.05 * math.log10(1 + cand.get("popularity", 0))


SEARCHERS = {"openlibrary": search_openlibrary, "audible": search_audible, "google": search_google}


def search(meta: dict) -> list[dict]:
    """Every source's results for this book, best match first."""
    title, author, isbn = meta.get("title", ""), meta.get("author", ""), meta.get("isbn")
    if not title and not isbn:
        return []
    names = [s for s in config.METADATA_SOURCES if s in SEARCHERS]

    def run(name):
        try:
            return SEARCHERS[name](title, author, isbn)
        except Exception as exc:  # one source being down shouldn't stop the others
            log.warning("Lookup on %s failed: %s", name, exc)
            return []

    with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
        results = [c for found in pool.map(run, names) for c in found]
    for cand in results:
        cand["score"] = score(meta, cand)
    results.sort(key=rank, reverse=True)

    # Search results carry no description; fetch it for the likeliest works.
    def describe(cand):
        try:
            cand["description"] = _openlibrary_description(cand["work"])
        except Exception as exc:
            log.warning("Open Library description failed: %s", exc)

    works = [c for c in results if c["source"] == "openlibrary" and c.get("work")
             and c["score"] >= config.MATCH_MIN][:3]
    if works:
        with ThreadPoolExecutor(max_workers=len(works)) as pool:
            list(pool.map(describe, works))
    return results


def public(cand: dict) -> dict:
    """A result as the API returns it."""
    out = {k: cand.get(k) for k in FIELDS + ("subtitle", "source", "score", "cover_url", "link")}
    return {k: v for k, v in out.items() if v not in (None, "", set())}


def best_match(meta: dict, results: list[dict] | None = None) -> dict | None:
    """Combine the best result of each source that clears MATCH_MIN.

    Returns {field: value, ..., "cover_url": ..., "sources": [...]} or None.
    """
    results = search(meta) if results is None else results
    # Per source, the best-ranked result that clears MATCH_MIN.
    best: dict[str, dict] = {}
    for cand in results:
        if cand["score"] >= config.MATCH_MIN and cand["source"] not in best:
            best[cand["source"]] = cand
    if not best:
        return None
    merged: dict = {"sources": []}
    for field in FIELDS + ("cover_url",):
        order = PREFER.get(field, ("openlibrary", "audible", "google"))
        for source in order:
            value = best.get(source, {}).get(field)
            if field == "series" and not best.get(source, {}).get("series_index"):
                continue  # unnumbered "series" are mostly publishers' collections
            if value:
                merged[field] = value
                if source not in merged["sources"]:
                    merged["sources"].append(source)
                break
    if not merged.get("series"):
        merged.pop("series_index", None)
    # Open Library keeps separate works for many printings of old books; the
    # earliest first-publication year among the matching ones is the real one.
    years = [c["year"] for c in results if c["source"] == "openlibrary" and c.get("year")
             and c["score"] >= config.MATCH_MIN]
    if years:
        merged["year"] = min(years)
    return merged


def fill_missing(meta: dict, found: dict) -> tuple[dict, list[str]]:
    """Add what the file didn't have; never overwrite what it did.

    The one exception is the year: a file's year is often its ebook release
    (Project Gutenberg's 2004 for an 1843 novel), so an earlier first-publication
    year wins.
    """
    meta = dict(meta)
    filled = []
    for field in FIELDS:
        value = found.get(field)
        if not value:
            continue
        if field == "year" and meta.get("year") and value < meta["year"]:
            meta["year"] = value
            filled.append(field)
        elif not meta.get(field):
            meta[field] = str(value)
            filled.append(field)
    if "series_index" in filled and meta.get("series") != found.get("series"):
        meta.pop("series_index")  # a number only means something with its own series
        filled.remove("series_index")
    return meta, filled


def fetch_cover(url: str) -> bytes:
    """Download a cover image from one of the lookup sources."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in COVER_HOSTS:
        raise ValueError("Covers can only come from the lookup sources.")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=config.LOOKUP_TIMEOUT * 2) as resp:
        data = resp.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("That cover is over 10 MB.")
    return data
