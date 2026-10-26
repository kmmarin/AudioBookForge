"""After the recordings are heard: which clips train the voice, and how ready it is.

Skips speech heard twice (the same audio in two files), keeps only the
narrator's own voice (each clip's fingerprint against the main voice of all
recordings), sets aside clips for the quality check that training never sees,
picks the best clips up to TRAIN_MAX_MINUTES, and writes the readiness
report: usable minutes, what was dropped and why, sound coverage, speaker
consistency, a rating and advice."""
from __future__ import annotations

import random
import re
import unicodedata

import numpy as np

from . import config, db, speaker

# Why audio wasn't used, in the order the report lists them.
REASONS = {
    "pause": "Pauses between sentences",
    "other": "Music, sound effects and other sounds",
    "music": "Music or singing",
    "credits": "Opening and closing credits",
    "announcer": "Someone else speaking: an intro or announcer",
    "other_voice": "Other voices: other speakers, or character voices far from the narrator’s own",
    "duplicate": "Heard twice: the same audio is in two files",
    "noisy": "Background noise or music under the voice",
    "clipped": "Distorted: recorded too loud",
    "unclear": "Words Whisper wasn’t sure of",
    "too_short": "Too short to use",
    "too_long": "No clean place to cut",
    "skipped": "Not listened to: there was already plenty",
    "unheard": "Not listened to: you went ahead with what had been heard",
    "not_needed": "Not needed: the best clips are enough",
}
INTRO_SECONDS = 180  # another person's voice this close to a file's start or end is an intro or announcer
ANNOUNCER_SIM = 0.35  # "another person": a narrator's own character voices stay above this

# The 39 sounds of English (ARPAbet), with a word to hear each in.
SOUNDS = {
    "AA": "“o” in hot", "AE": "“a” in cat", "AH": "“u” in cup", "AO": "“aw” in saw", "AW": "“ow” in now",
    "AY": "“i” in time", "B": "“b” in bed", "CH": "“ch” in chair", "D": "“d” in day", "DH": "“th” in this",
    "EH": "“e” in bed", "ER": "“ur” in hurt", "EY": "“a” in day", "F": "“f” in fish", "G": "“g” in go",
    "HH": "“h” in hat", "IH": "“i” in sit", "IY": "“ee” in see", "JH": "“j” in jam", "K": "“k” in kite",
    "L": "“l” in leg", "M": "“m” in man", "N": "“n” in no", "NG": "“ng” in sing", "OW": "“o” in go",
    "OY": "“oy” in boy", "P": "“p” in pen", "R": "“r” in red", "S": "“s” in sun", "SH": "“sh” in she",
    "T": "“t” in top", "TH": "“th” in thin", "UH": "“oo” in book", "UW": "“oo” in food", "V": "“v” in van",
    "W": "“w” in wet", "Y": "“y” in yes", "Z": "“z” in zoo", "ZH": "“s” in measure",
}
# espeak-ng's IPA (US and British) to those sounds; longest symbols first.
IPA = {
    "tʃ": "CH", "dʒ": "JH", "eɪ": "EY", "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY", "oʊ": "OW", "əʊ": "OW",
    "ɪə": "IH", "eə": "EH", "ʊə": "UH", "ɚ": "ER", "ɜ": "ER", "ᵻ": "IH", "ɐ": "AH", "ə": "AH", "ʌ": "AH",
    "æ": "AE", "ɑ": "AA", "ɒ": "AA", "a": "AA", "ɔ": "AO", "ɛ": "EH", "e": "EH", "ɪ": "IH", "i": "IY",
    "ʊ": "UH", "u": "UW", "o": "OW", "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z", "ʃ": "SH", "ʒ": "ZH", "h": "HH",
    "m": "M", "n": "N", "ŋ": "NG", "l": "L", "ɫ": "L", "ɹ": "R", "r": "R", "w": "W", "ʍ": "W", "j": "Y",
    "ɾ": "T",
}
_IPA_KEYS = sorted(IPA, key=len, reverse=True)

RATINGS = [  # (minutes at least, key, label, what it means)
    (0, "not_enough", "Not enough", "The new voice would be mostly the starting voice, with a hint of the narrator."),
    (config.READY_USABLE, "usable", "Usable", "Recognisable as the narrator, but rough in places."),
    (config.READY_GOOD, "good", "Good", "Clearly the narrator, with the odd rough word."),
    (config.READY_GREAT, "great", "Great", "As close as this kind of voice gets; more audio adds little."),
]


def sounds_in(phonemes: str) -> list[str]:
    """English sounds in an espeak IPA string (stress and length marks ignored)."""
    text = unicodedata.normalize("NFD", phonemes)
    text = "".join(ch for ch in text if not unicodedata.combining(ch) and ch not in "ˈˌːˑ")
    out, i = [], 0
    while i < len(text):
        for key in _IPA_KEYS:
            if text.startswith(key, i):
                out.append(IPA[key])
                i += len(key)
                break
        else:
            i += 1  # spaces, punctuation, glottal stops and the like
    return out


def coverage(texts: list[str], espeak_voice: str) -> dict:
    counts = dict.fromkeys(SOUNDS, 0)
    if texts:
        from piper.phonemize_espeak import EspeakPhonemizer
        phonemizer = EspeakPhonemizer()
        for text in texts:
            for sentence in phonemizer.phonemize(espeak_voice, text):
                for sound in sounds_in("".join(sentence)):
                    counts[sound] += 1
    covered = [s for s, n in counts.items() if n >= config.PHONEME_MIN_COUNT]
    rare = sorted((s for s, n in counts.items() if n < config.PHONEME_MIN_COUNT), key=lambda s: counts[s])
    return {"share": round(len(covered) / len(SOUNDS), 3), "covered": len(covered), "of": len(SOUNDS),
            "rare": [{"sound": s, "example": SOUNDS[s], "count": counts[s]} for s in rare],
            "min_count": config.PHONEME_MIN_COUNT}


def _rating(minutes: float) -> dict:
    found = RATINGS[0]
    for row in RATINGS:
        if minutes >= row[0]:
            found = row
    return {"key": found[1], "label": found[2], "text": found[3]}


def _minutes(seconds: float) -> str:
    m = round(seconds / 60)
    if m <= 1:
        return "a minute"
    if m < 90:
        return f"{m} minute{'s' if m != 1 else ''}"
    return f"{seconds / 3600:.1f} hours".replace(".0 ", " ")


def span(seconds: float) -> str:
    """1 min 40 s, 45 s, 2 h 5 min."""
    s = round(seconds)
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} h {m} min" if m else f"{h} h"
    if m:
        return f"{m} min {s} s" if s else f"{m} min"
    return f"{s} s"


def _advice(usable: float, heard: float, cov: dict, consistency: float, train: float,
            holdout: float = 0.0) -> list[str]:
    tips = []
    keep_ratio = usable / heard if heard > 0 else 0.6
    for limit, _key, label, _text in RATINGS[1:]:
        if usable < limit * 60:
            need = limit * 60 - usable
            raw = need / max(keep_ratio, 0.2)
            tip = f"About {_minutes(need)} more of usable speech would reach {label}"
            if raw - need >= 60:
                tip += f" (roughly {_minutes(raw)} of recordings like these)"
            tips.append(tip + ".")
            break
    if usable >= config.READY_PLENTY * 60:
        tips.append("More audio won’t add much now.")
    if train < usable - holdout - 60:
        tips.append(f"Training uses the best {_minutes(train)}, to keep training time down.")
    if cov["share"] < 0.9 and cov["rare"]:
        examples = ", ".join(r["example"] for r in cov["rare"][:3])
        tips.append(f"Some sounds are rare here ({examples}). Recordings with more varied text would help.")
    if consistency < 0.85:
        tips.append("Several voices were found. Only the main one is used: play the dropped examples to check it’s the narrator.")
    return tips


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", re.sub(r"[’']", "", text.lower())).split()


def duplicates(rows: list[dict], order: dict[str, int], size: int = 5, share: float = 0.6) -> set[int]:
    """Clips whose words were already heard in an earlier file (book order):
    the same audio added twice, or two files that overlap. Their runs of five
    words are compared, so slightly different cuts still match."""
    seen: set[tuple] = set()
    found: set[int] = set()
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], []).append(r)
    for source_id in sorted(by_source, key=lambda s: order.get(s, 0)):
        mine: set[tuple] = set()
        for r in by_source[source_id]:
            words = _words(r["text"])
            grams = {tuple(words[i:i + size]) for i in range(len(words) - size + 1)} or (
                {tuple(words)} if len(words) >= 3 else set())
            if grams and len(grams & seen) >= share * len(grams):
                found.add(r["id"])
            mine |= grams
        seen |= mine  # repeats inside one file are the book's own, not a duplicate
    return found


def finalize(project_id: str) -> dict:
    """Decide every clip's fate and write the readiness report."""
    project = db.get_project(project_id)
    base = config.BASES.get(project["accent"]) or config.BASES[config.DEFAULT_ACCENT]
    sources = db.sources(project_id)
    live = {s["id"] for s in sources if s["status"] not in ("duplicate", "failed", "uploading")}
    rows = [r for r in db.clips(project_id, with_embedding=True) if r["source_id"] in live]
    order = {s["id"]: n for n, s in enumerate(sources)}
    lengths = {s["id"]: float(s["duration"] or 0) for s in sources}
    training = project.get("training") or {}
    trained_upto = int(training.get("trained_upto") or 0)  # clips up to this id may have been trained on

    dups = duplicates(rows, order)
    good = [r for r in rows if r["reason"] is None and r["embedding"] and r["id"] not in dups]
    sims: dict[int, float] = {}
    threshold = config.SPEAKER_MIN_SIM
    if good:
        embs = np.stack([speaker.from_blob(r["embedding"]) for r in good])
        center = speaker.centroid(embs)
        values = embs @ center
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) or 0.02
        # Below typical by several spreads, but never stricter than SPEAKER_MAX_SIM:
        # a short or very even sample has a tiny spread, and the narrator's own
        # character voices sit well above another person's (~0.15 to 0.3).
        threshold = min(config.SPEAKER_MAX_SIM,
                        max(config.SPEAKER_MIN_SIM, median - config.SPEAKER_SPREAD * mad))
        sims = {r["id"]: float(v) for r, v in zip(good, values)}

    verdicts: dict[int, str] = {}
    for r in rows:
        if r["reason"]:
            verdicts[r["id"]] = r["reason"]
        elif r["id"] in dups:
            verdicts[r["id"]] = "duplicate"
        elif r["id"] not in sims:
            verdicts[r["id"]] = "unclear"
        elif sims[r["id"]] < threshold:
            verdicts[r["id"]] = "other_voice"
        else:
            verdicts[r["id"]] = "keep"
    kept = sorted((r for r in rows if verdicts[r["id"]] == "keep"),
                  key=lambda r: (order.get(r["source_id"], 0), r["start"]))

    # Clips for the quality check: whole sentences of 2.5-10 s, spread through
    # the recordings. Once chosen they stay chosen, and new ones only come from
    # clips added since the last training, so none was ever trained on.
    target = max(3, min(config.HOLDOUT_CLIPS, round(len(kept) * 0.1))) if len(kept) >= 20 else 0
    holdout = [r for r in kept if r["role"] == "holdout"]
    if len(holdout) < target:
        candidates = [r for r in kept if r["role"] != "holdout" and r["id"] > trained_upto
                      and 2.5 <= r["end"] - r["start"] <= 10 and r["words"] >= 5
                      and re.search(r"[.!?][\"”’)]*$", r["text"])]
        need = target - len(holdout)
        if candidates:
            step = len(candidates) / need
            picked = {candidates[int(i * step)]["id"] for i in range(min(need, len(candidates)))}
            holdout += [r for r in candidates if r["id"] in picked]
    held = {r["id"] for r in holdout}

    # Training: the best clips first, up to the cap.
    def score(r):
        return (min(r["snr"] or 0, 50) / 50 + (r["conf"] or 0) + sims.get(r["id"], 0)
                + (0.3 if 2 <= r["end"] - r["start"] <= 12 else 0))
    budget = config.TRAIN_MAX_MINUTES * 60
    train_ids, train_seconds = set(), 0.0
    for r in sorted((r for r in kept if r["id"] not in held), key=score, reverse=True):
        if train_seconds + (r["end"] - r["start"]) > budget:
            break
        train_ids.add(r["id"])
        train_seconds += r["end"] - r["start"]

    # Where every second went. Another voice right at a file's start or end is
    # an intro or announcer; elsewhere it's another speaker or a character voice.
    updates, seconds = [], dict.fromkeys(REASONS, 0.0)
    shown: dict[str, list[int]] = {}
    for r in rows:
        verdict = verdicts[r["id"]]
        role = "holdout" if r["id"] in held else "train" if r["id"] in train_ids else None
        if verdict == "keep" and role is None:
            verdict = "not_needed"
        updates.append((sims.get(r["id"]), verdict, role, r["id"]))
        key = verdict
        if (verdict == "other_voice" and sims.get(r["id"], 1.0) < ANNOUNCER_SIM
                and (r["start"] < INTRO_SECONDS or lengths.get(r["source_id"], 0) - r["end"] < INTRO_SECONDS)):
            key = "announcer"
        if key in seconds:
            seconds[key] += r["end"] - r["start"]
        shown.setdefault("keep" if verdict in ("keep", "not_needed") else key, []).append(r["id"])
    db.set_verdicts(updates)

    for s in sources:
        if s["status"] == "duplicate":
            seconds["duplicate"] += float(s["duration"] or 0)
        for key, value in (s.get("gaps") or {}).items():
            if key in seconds:
                seconds[key] += value
    total = sum(float(s["duration"] or 0) for s in sources)
    # Skipped ("plenty already") counts as passed; unheard (stopped by you) never did.
    heard = sum(float(s["done_until"] or 0) for s in sources if s["id"] in live) - seconds["skipped"]
    usable = sum(r["end"] - r["start"] for r in kept)
    holdout_seconds = sum(r["end"] - r["start"] for r in holdout)
    train_texts = [r["text"] for r in rows if r["id"] in train_ids]
    cov = coverage(train_texts, base["espeak"])
    others = seconds["other_voice"] + seconds["announcer"]
    consistency = usable / (usable + others) if usable + others else 0.0

    notes = []  # plain words about what was left out, before the advice
    if seconds["credits"] >= 1:
        from .media import CREDITS_RE
        titles = sorted({c["title"] for s in sources for c in (s.get("chapters") or [])
                         if CREDITS_RE.search(c.get("title") or "")})
        notes.append(f"Left out {span(seconds['credits'])} of credits (the chapters "
                     + " and ".join(f"“{t}”" for t in titles[:3]) + ").")
    if seconds["announcer"] >= 3:
        notes.append(f"Dropped {span(seconds['announcer'])} spoken by someone else (an intro or announcer).")
    if seconds["other_voice"] >= 3:
        notes.append(f"Dropped {span(seconds['other_voice'])} in other voices: other speakers, or character "
                     "voices far from the narrator’s usual one.")
    if seconds["duplicate"] >= 3:
        notes.append(f"Skipped {span(seconds['duplicate'])} heard twice: the same audio is in two files.")
    rng = random.Random(project_id)  # the same examples each time the report is made
    report = {
        "total_minutes": round(total / 60, 1),
        "heard_minutes": round(heard / 60, 1),
        "usable_minutes": round(usable / 60, 1),
        "train_minutes": round(train_seconds / 60, 1),
        "holdout_minutes": round(holdout_seconds / 60, 1),
        "train_clips": len(train_ids),
        "holdout_clips": len(holdout),
        "dropped": [{"reason": k, "label": REASONS[k], "minutes": round(v / 60, 1), "seconds": round(v)}
                    for k, v in seconds.items() if v >= 1],
        "coverage": cov,
        "speaker": {"consistency": round(consistency, 3),
                    "similarity": round(float(np.median(list(sims.values()))), 3) if sims else None,
                    "threshold": round(threshold, 3),
                    "other_seconds": round(others),
                    "label": ("One voice throughout" if consistency >= 0.95 else
                              "Mostly one voice" if consistency >= 0.8 else
                              "Several voices: only the main one is used")},
        "rating": _rating(usable / 60),
        "advice": notes + _advice(usable, heard, cov, consistency, train_seconds, holdout_seconds),
        "thresholds": {"usable": config.READY_USABLE, "good": config.READY_GOOD,
                       "great": config.READY_GREAT, "plenty": config.READY_PLENTY},
        "examples": {k: rng.sample(v, min(3, len(v))) for k, v in shown.items()},
    }
    return report
