"""Unit tests for the voice creator's pure parts (no models, no GPU).

    python -m unittest discover -s voicelab/tests      (from the project root, in the voicelab image)
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "voicelab")]
_tmp = tempfile.mkdtemp()
os.environ.update(LAB_DATA=_tmp, MODELS_DIR=_tmp, GPU_SHARE="on", GPU_SHARE_STALE="120")

from app import gpushare  # noqa: E402
from lab import analyze, config, gpu, media, readiness  # noqa: E402


class TestNarrator(unittest.TestCase):
    def test_tags(self):
        self.assertEqual(media.narrator_from_tags({"composer": "Heart (Kokoro)"}), ("Heart (Kokoro)", "composer"))
        self.assertEqual(media.narrator_from_tags({"narrator": "Jim Dale; Stephen Fry", "composer": "X"}),
                         ("Jim Dale", "narrator"))
        self.assertEqual(media.narrator_from_tags({"comment": "Unabridged. Narrated by Blair Brown."}),
                         ("Blair Brown", "comment"))
        self.assertEqual(media.narrator_from_tags({"artist": "Lois Lowry, read by J. R. Smith"}),
                         ("J. R. Smith", "artist"))
        self.assertIsNone(media.narrator_from_tags({"title": "Number the Stars", "artist": "Lois Lowry"}))

    def test_text(self):
        self.assertEqual(media.narrator_from_text("Number the Stars, by Lois Lowry. Narrated by Blair Brown. Chapter one."),
                         "Blair Brown")
        self.assertEqual(media.narrator_from_text("This audiobook is read by Jane van der Berg."), "Jane van der Berg")
        self.assertEqual(media.narrator_phrase(" The Long Road, by Ann Author. Read by Jack Reade."),
                         ("Jack Reade", "Read by Jack Reade"))
        # The way Whisper writes a book's opening credits: no full stop after the name.
        self.assertEqual(media.narrator_phrase("Read by Jack Reade Performance Copyright 2001"),
                         ("Jack Reade", "Read by Jack Reade"))
        self.assertIsNone(media.narrator_from_text("Chapter one. Why are you running?"))

    def test_accent_and_language(self):
        self.assertEqual(media.accent_from_tags({"language": "en-GB"}), "en-gb")
        self.assertIsNone(media.accent_from_tags({"language": "eng"}))
        self.assertEqual(media.language_from_tags({"language": "eng"}), "en")
        self.assertIsNone(media.language_from_tags({"language": "und"}))


def words(spec):
    """[(word, start, end), ...] -> Whisper-like word dicts."""
    return [{"word": " " + w, "start": a, "end": b, "prob": 0.9, "logprob": -0.2, "ratio": 1.5, "music": False}
            for w, a, b in spec]


class TestSegmenting(unittest.TestCase):
    def test_cuts_at_sentence_end_after_target(self):
        spec, t = [], 0.0
        for n in range(40):  # 0.35 s words, 0.05 s apart; a sentence end every 8 words with a 0.3 s pause
            end = n % 8 == 7
            spec.append((f"w{n}" + ("." if end else ""), t, t + 0.35))
            t += 0.35 + (0.3 if end else 0.05)
        groups = analyze.group_words(words(spec))
        for i, k, forced in groups[:-1]:
            self.assertTrue(spec[k][0].endswith("."))
            self.assertFalse(forced)
            self.assertLessEqual(spec[k][2] - spec[i][1], config.CLIP_MAX)

    def test_long_pause_always_cuts(self):
        g = analyze.group_words(words([("a", 0, 0.3), ("b", 0.35, 0.6), ("c", 2.0, 2.3)]))
        self.assertEqual([(i, k) for i, k, _ in g], [(0, 1), (2, 2)])

    def test_no_pause_for_too_long_is_forced(self):
        spec = [(f"w{n}", n * 0.4, n * 0.4 + 0.38) for n in range(60)]  # 24 s, never a pause
        g = analyze.group_words(words(spec))
        self.assertTrue(g[0][2])
        self.assertLessEqual(spec[g[0][1]][2] - spec[g[0][0]][1], config.CLIP_MAX)

    def test_judge(self):
        clip = {"start": 0, "end": 4, "words": 8, "conf": 0.95, "unsure": 0.0, "snr": 40, "clipped": 0,
                "_logprob": -0.2, "_ratio": 1.4}
        self.assertIsNone(analyze.judge(clip, False, False))
        self.assertEqual(analyze.judge(clip, False, True), "music")
        self.assertEqual(analyze.judge({**clip, "snr": 12}, False, False), "noisy")
        self.assertEqual(analyze.judge({**clip, "conf": 0.4}, False, False), "unclear")
        self.assertEqual(analyze.judge({**clip, "clipped": 0.01}, False, False), "clipped")
        self.assertEqual(analyze.judge({**clip, "end": 0.6}, False, False), "too_short")


class TestDuplicates(unittest.TestCase):
    def test_overlapping_files_count_once(self):
        text = ["Reacher walked out of the town at noon.", "The heat came up off the road like a wall.",
                "A car slowed and the driver looked him over.", "He said nothing and kept on walking west."]
        rows = [{"id": i + 1, "source_id": "a", "text": t} for i, t in enumerate(text)]
        # The same speech again in a second file, cut slightly differently, plus one new sentence.
        rows += [{"id": 10, "source_id": "b", "text": "out of the town at noon. The heat came up off the road like a wall."},
                 {"id": 11, "source_id": "b", "text": "A car slowed and the driver looked him over."},
                 {"id": 12, "source_id": "b", "text": "Nobody in Echo had seen a stranger in years."}]
        self.assertEqual(readiness.duplicates(rows, {"a": 0, "b": 1}), {10, 11})

    def test_repeats_inside_one_file_are_kept(self):
        rows = [{"id": 1, "source_id": "a", "text": "Chapter seven. The long road west."},
                {"id": 2, "source_id": "a", "text": "Chapter seven. The long road west."}]
        self.assertEqual(readiness.duplicates(rows, {"a": 0}), set())

    def test_book_order(self):
        from lab import db
        names = ["Echo Burning-Part10.mp3", "Echo Burning-Part06 (James Hall's conflicted copy 2013-05-23).mp3",
                 "Echo Burning-Part06.mp3", "Echo Burning-Part2.mp3"]
        rows = [{"filename": n, "tags": {}, "created": i} for i, n in enumerate(names)]
        self.assertEqual([r["filename"] for r in sorted(rows, key=db.book_order)],
                         ["Echo Burning-Part2.mp3", "Echo Burning-Part06.mp3", names[1], "Echo Burning-Part10.mp3"])
        rows[0]["tags"] = {"track": "1/16"}
        self.assertEqual(sorted(rows, key=db.book_order)[0]["filename"], "Echo Burning-Part10.mp3")

    def test_credit_chapters(self):
        chapters = [{"start": 0, "end": 75, "title": "Opening Credits"}, {"start": 75, "end": 600, "title": "Chapter 1"},
                    {"start": 600, "end": 640, "title": "End Credits"}]
        self.assertEqual(media.credit_spans(chapters), [(0, 75), (600, 640)])
        self.assertEqual(media.intro_end(chapters, 300), 300)
        self.assertEqual(media.intro_end([{"start": 0, "end": 420, "title": "Opening Credits"}], 300), 420)
        self.assertEqual(readiness.span(100), "1 min 40 s")


class TestDataset(unittest.TestCase):
    def test_quotes_dont_break_piper_csv(self):
        import csv
        import io
        from lab import dataset
        texts = ['"\'You must think I\'m crazy,\' she said.', 'Why do you think?', 'He said "no" | twice.']
        lines = "".join(f"c{i}.wav|{dataset.clean_text(t)}\n" for i, t in enumerate(texts))
        rows = list(csv.reader(io.StringIO(lines), delimiter="|"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1], ["c1.wav", "Why do you think?"])


class TestReadiness(unittest.TestCase):
    def test_sounds(self):
        # "the rain in Spain" as espeak writes it (US)
        self.assertEqual(readiness.sounds_in("ðə ɹˈeɪn ɪn spˈeɪn"), ["DH", "AH", "R", "EY", "N", "IH", "N", "S", "P", "EY", "N"])
        self.assertEqual(readiness.sounds_in("tʃˈɜːtʃ dʒˈʌdʒ mˈɛʒɚ"), ["CH", "ER", "CH", "JH", "AH", "JH", "M", "EH", "ZH", "ER"])

    def test_ratings(self):
        self.assertEqual(readiness._rating(3)["key"], "not_enough")
        self.assertEqual(readiness._rating(5)["key"], "usable")
        self.assertEqual(readiness._rating(30)["key"], "good")
        self.assertEqual(readiness._rating(200)["key"], "great")

    def test_advice_says_how_much_more(self):
        cov = {"share": 1.0, "rare": []}
        tips = readiness._advice(usable=40 * 60, heard=60 * 60, cov=cov, consistency=1.0, train=40 * 60)
        self.assertTrue(tips[0].startswith("About 20 minutes more"), tips)
        self.assertIn("reach Great", tips[0])


class TestPicking(unittest.TestCase):
    """The owner approves or leaves out clips (clips.pick, projects.picking)."""

    def setUp(self):
        import uuid

        import numpy as np
        from lab import db, speaker
        db.init()
        self.db = db
        self.pid, self.sid = uuid.uuid4().hex[:12], uuid.uuid4().hex[:12]
        db.create_project(self.pid, "en-us")
        db.create_source(self.sid, self.pid, "Book-Part01.mp3", "/nowhere.mp3", 0)
        db.update_source(self.sid, status="done", duration=3600.0, done_until=3600.0)
        rng = np.random.default_rng(7)
        unit = lambda v: v / np.linalg.norm(v)
        narrator = unit(rng.normal(size=192))
        clips = []
        for n in range(66):  # 60 in the narrator's voice, then 6 others; all mid-file, whole sentences
            other = n >= 60
            emb = unit(rng.normal(size=192)) if other else unit(narrator + 0.15 * unit(rng.normal(size=192)))
            clips.append({"start": 300.0 + n * 20, "end": 304.0 + n * 20, "words": 7, "conf": 0.95, "unsure": 0.0,
                          "snr": 40.0, "clipped": 0.0, "reason": None, "embedding": speaker.to_blob(emb),
                          "text": f"The road ran on west for mile {n} today."})
        clips += [{**clips[0], "start": 1800.0 + n * 20, "end": 1804.0 + n * 20, "reason": "noisy",
                   "text": f"A noisy line number {n} here."} for n in range(2)]
        db.add_clips(self.pid, self.sid, clips, 3600.0, 3600.0, {})
        rows = sorted(db.clips(self.pid), key=lambda r: r["start"])
        self.narrator = [r["id"] for r in rows if r["reason"] is None][:60]
        self.others = [r["id"] for r in rows if r["reason"] is None][60:]
        self.noisy = [r["id"] for r in rows if r["reason"]]

    def fates(self):
        readiness.finalize(self.pid)
        return {r["id"]: (r["verdict"], r["role"]) for r in self.db.clips(self.pid)}

    def picking(self, on=True):
        self.db.update_project(self.pid, picking=int(on))

    def test_off_is_as_before(self):
        before = self.fates()
        self.db.set_picks(self.pid, {self.narrator[0]: 0, self.others[0]: 1})
        self.assertEqual(self.fates(), before)
        self.assertTrue(all(before[i][0] == "other_voice" for i in self.others))

    def test_left_out_never_trains_or_checks(self):
        self.picking()
        self.db.set_picks(self.pid, {**{i: 1 for i in self.narrator[3:]}, **{i: 0 for i in self.narrator[:3]}})
        fates = self.fates()
        for i in self.narrator[:3]:
            self.assertEqual(fates[i], ("left_out", None))

    def test_approved_other_voice_is_kept_but_hard_reason_stays(self):
        self.picking()
        self.db.set_picks(self.pid, {**{i: 1 for i in self.narrator}, self.others[0]: 1, self.noisy[0]: 1})
        fates = self.fates()
        self.assertEqual(fates[self.others[0]][0], "keep")
        self.assertIn(fates[self.others[0]][1], ("train", "holdout"))
        self.assertEqual(fates[self.noisy[0]], ("noisy", None))

    def test_only_approved_clips_train_and_check(self):
        self.picking()
        approved = set(self.narrator[:40])
        self.db.set_picks(self.pid, {i: 1 for i in approved})
        fates = self.fates()
        used = {i for i, (_v, role) in fates.items() if role}
        self.assertTrue(used and used <= approved)
        self.assertTrue(any(role == "holdout" for _v, role in fates.values()))
        self.assertTrue(all(fates[i][0] == "not_needed" for i in self.narrator[40:]))

    def test_leaving_out_a_held_out_clip_replaces_it(self):
        self.picking()
        self.db.set_picks(self.pid, {i: 1 for i in self.narrator})
        held = [i for i, (_v, role) in self.fates().items() if role == "holdout"]
        self.db.set_picks(self.pid, {held[0]: 0})
        fates = self.fates()
        now = [i for i, (_v, role) in fates.items() if role == "holdout"]
        self.assertEqual(len(now), len(held))
        self.assertNotIn(held[0], now)
        self.assertEqual(fates[held[0]], ("left_out", None))

    def test_report_counts(self):
        self.picking()
        self.db.set_picks(self.pid, {self.narrator[0]: 1, self.narrator[1]: 0})
        picks = readiness.finalize(self.pid)["picks"]
        self.assertEqual((picks["on"], picks["approved"], picks["left_out"], picks["dropped_rescuable"]), (True, 1, 1, 6))
        self.assertEqual(picks["to_review"], 58)

    def test_picks_api(self):
        from fastapi import HTTPException
        from lab import main
        self.picking()
        readiness.finalize(self.pid)
        page = main.list_picks(self.pid, filter="all", sort="likeness", page=1, size=10)
        self.assertEqual(page["total"], 66)  # never the noisy ones
        self.assertEqual({x["id"] for x in page["items"][:6]}, set(self.others))  # least like the narrator first
        self.assertEqual(page["counts"], {"review": 60, "approved": 0, "left_out": 0, "dropped": 6, "all": 66})
        self.assertEqual(len(main.list_picks(self.pid, filter="review", page=7, size=10)["items"]), 0)
        res = main.save_picks(self.pid, main.PicksBody(picks={str(self.others[0]): 1, str(self.narrator[0]): 0}))
        self.assertEqual((res["counts"]["approved"], res["counts"]["left_out"], res["counts"]["dropped"]), (1, 1, 5))
        with self.assertRaises(HTTPException) as hard:
            main.save_picks(self.pid, main.PicksBody(picks={str(self.noisy[0]): 1}))
        self.assertEqual(hard.exception.status_code, 409)
        other = TestPicking("test_picks_api")
        other.setUp()
        with self.assertRaises(HTTPException) as foreign:
            main.save_picks(self.pid, main.PicksBody(picks={str(other.narrator[0]): 1}))
        self.assertEqual(foreign.exception.status_code, 404)

    def test_listening_hears_everything(self):
        from lab import jobs
        self.db.update_source(self.sid, gaps={"skipped": 1000.0})
        self.picking()
        self.assertFalse(jobs.heard_enough(self.pid))
        self.assertEqual(jobs.hear_everything(self.pid), 1)
        s = self.db.get_source(self.sid)
        self.assertEqual((s["status"], s["done_until"], s["gaps"]), ("queued", 2600.0, {}))
        self.assertTrue(self.db.job_waiting(self.pid, "analyze"))


class TestLease(unittest.TestCase):
    """The voice creator's side, against a simulated Forge that steps aside."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp()) / "gpu-share"
        config.GPU_SHARE_DIR = self.dir

    def test_waits_for_forge_then_holds_and_releases(self):
        forge = gpushare.ForgeSide(self.dir, uses_gpu=True)
        forge.report(busy=True, pool_up=True)
        seen = []

        def forge_loop():  # a Forge that finishes its book after a moment, then steps aside
            time.sleep(0.5)
            forge.report(busy=False, pool_up=True)
            while (lab := forge.share.lab()["state"]) == gpushare.IDLE:
                time.sleep(0.05)
            forge.report(busy=False, pool_up=False, lab=lab)

        threading.Thread(target=forge_loop, daemon=True).start()
        with gpu.Lease("test", on_wait=seen.append, shared=True) as lease:
            self.assertEqual(forge.share.lab()["state"], gpushare.HOLDING)
            self.assertEqual(lease.side.state, gpushare.HOLDING)
        self.assertEqual(forge.share.lab()["state"], gpushare.IDLE)
        self.assertTrue(any("Forge is voicing" in s for s in seen), seen)

    def test_stop_while_waiting(self):
        forge = gpushare.ForgeSide(self.dir, uses_gpu=True)
        forge.report(busy=True, pool_up=True)
        with self.assertRaises(gpu.Stopped):
            with gpu.Lease("test", should_stop=lambda: True, shared=True):
                pass
        self.assertEqual(forge.share.lab()["state"], gpushare.IDLE, "a stopped wait lets go of its claim")


if __name__ == "__main__":
    unittest.main()
