"""Cloned voices in the voice creator: picking reference clips, putting a voice
into Forge, the model choice, and the job, against a fake clone server.

    python -m unittest discover -s voicelab/tests      (in the voicelab image; no GPU, no network)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "voicelab")]
_tmp = tempfile.mkdtemp()
os.environ.update(LAB_DATA=_tmp, MODELS_DIR=_tmp, GPU_SHARE="on", GPU_SHARE_STALE="120")

import soundfile as sf  # noqa: E402

from app.clone import CloneError  # noqa: E402
from lab import clone, config, db, jobs, voices  # noqa: E402

BIG, SMALL = "qwen3-tts-1.7b", "qwen3-tts-0.6b"


def clip(i, source="s1", start=0.0, seconds=10.0, conf=0.9, unsure=0.0, snr=40.0, verdict="keep", role="train"):
    return {"id": i, "source_id": source, "start": start, "end": start + seconds, "text": f"clip {i}",
            "conf": conf, "unsure": unsure, "snr": snr, "verdict": verdict, "role": role}


def served(big_fits: bool) -> dict:
    return {"ok": True, "models": [{"id": BIG, "fits": big_fits, "note": "Needs 8 GB of video memory."},
                                   {"id": SMALL, "fits": True, "note": ""}]}


def tone(path: Path, seconds: float, rate: int) -> None:
    t = np.arange(int(seconds * rate)) / rate
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)


class TestPicking(unittest.TestCase):
    def test_only_clean_clips_of_the_right_length(self):
        clips = [clip(1, seconds=5), clip(2, seconds=16), clip(3, verdict="other_voice"), clip(4, role=None),
                 clip(5, seconds=9), clip(6, role="holdout", seconds=13.5)]
        self.assertEqual(sorted(c["id"] for c in clone.pick_candidates(clips)), [5, 6])

    def test_clearest_first_then_other_recordings_and_stretches(self):
        clips = [clip(1, "a", 0, conf=0.95), clip(2, "a", 100, conf=0.94), clip(3, "b", 0, conf=0.90),
                 clip(4, "a", 3000, conf=0.93)]
        picked = [c["id"] for c in clone.pick_candidates(clips, n=3)]
        self.assertEqual(picked[0], 1)
        self.assertEqual(picked[1], 3)  # another recording beats a clip right next to the first
        self.assertEqual(picked[2], 4)  # then a far-off stretch of the first recording

    def test_at_most_n(self):
        clips = [clip(i, f"s{i}") for i in range(10)]
        self.assertEqual(len(clone.pick_candidates(clips, n=4)), 4)


class TestModels(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(config, "CLONE_URL", "http://clone.test")
        patch.start()
        self.addCleanup(patch.stop)

    def test_best_that_fits(self):
        self.assertEqual(clone.default_model(clone.model_choices(served(True))), BIG)
        self.assertEqual(clone.default_model(clone.model_choices(served(False))), SMALL)

    def test_server_not_answering_still_clones(self):
        choices = clone.model_choices({"ok": False, "error": "down"})
        self.assertFalse(any(c["fits"] for c in choices if c["kind"] == "clone"))
        self.assertEqual(clone.default_model(choices), SMALL)

    def test_no_clone_server_means_piper(self):
        with mock.patch.object(config, "CLONE_URL", ""):
            choices = clone.model_choices(None)
            self.assertEqual(clone.default_model(choices), clone.PIPER)
            with self.assertRaises(clone.CantClone):
                clone.check_model(SMALL, choices)
            self.assertEqual(clone.check_model(clone.PIPER, choices)["kind"], "piper")

    def test_check_model(self):
        with mock.patch.object(clone, "health", return_value=served(False)):
            choices = clone.model_choices(served(False))
            with self.assertRaises(clone.CantClone):
                clone.check_model(BIG, choices)  # doesn't fit the clone server's GPU
            self.assertEqual(clone.check_model(SMALL, choices)["kind"], "clone")
            with self.assertRaises(clone.CantClone):
                clone.check_model("nonsense", choices)

    def test_a_projects_own_choice_wins(self):
        self.assertEqual(clone.effective_model({"model": clone.PIPER}, clone.model_choices(served(True))), clone.PIPER)
        self.assertEqual(clone.effective_model({"model": None}, clone.model_choices(served(True))), BIG)


class TestInstall(unittest.TestCase):
    def setUp(self):
        base = Path(tempfile.mkdtemp())
        for name, path in (("CUSTOM_VOICES_DIR", base / "custom"), ("CLONED_VOICES_DIR", base / "cloned")):
            patch = mock.patch.object(config, name, path)
            patch.start()
            self.addCleanup(patch.stop)
        config.CUSTOM_VOICES_DIR.mkdir(parents=True)
        self.ref = base / "ref.wav"
        tone(self.ref, 9.0, 24000)
        self.info = {"name": "Ann Reader", "model": BIG, "project": "p1", "clip": 7, "match": 0.73}

    def test_layout_and_voice_json_last(self):
        order, real = [], os.replace

        def spy(src, dst):
            order.append(Path(dst).name)
            return real(src, dst)

        with mock.patch.object(voices.os, "replace", spy):
            voices.install_clone("ann-reader", self.ref, "He drove north.", self.info)
        self.assertEqual(order[-1], "voice.json")  # Forge lists it only once the rest is there
        self.assertLess(order.index("ref.wav"), order.index("voice.json"))
        self.assertLess(order.index("ref.txt"), order.index("voice.json"))
        folder = config.CLONED_VOICES_DIR / "ann-reader"
        data = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
        self.assertEqual((data["name"], data["model"], data["language"], data["clip"]), ("Ann Reader", BIG, "English", 7))
        self.assertIn("installed", data)
        self.assertEqual((folder / "ref.txt").read_text(encoding="utf-8").strip(), "He drove north.")
        self.assertEqual(sf.info(str(folder / "ref.wav")).samplerate, 24000)
        self.assertTrue(self.ref.exists())  # copied, not moved
        (listed,) = voices.listing()
        self.assertEqual((listed["id"], listed["kind"], listed["model"], listed["match"]),
                         ("clone-ann-reader", "clone", BIG, 0.73))

    def test_slugs_unique_across_both_kinds(self):
        (config.CUSTOM_VOICES_DIR / "ann-reader.onnx").write_bytes(b"x")
        (config.CLONED_VOICES_DIR / "ann-reader-2").mkdir(parents=True)
        self.assertEqual(voices.free_slug("Ann Reader"), "ann-reader-3")

    def test_model_rename_and_remove(self):
        voices.install_clone("ann-reader", self.ref, "He drove north.", self.info)
        voices.set_model("ann-reader", SMALL)
        voices.rename("ann-reader", "Ann")
        (listed,) = voices.listing()
        self.assertEqual((listed["model"], listed["name"]), (SMALL, "Ann"))
        with self.assertRaises(ValueError):
            voices.set_model("not-cloned", SMALL)
        voices.remove("ann-reader")
        self.assertEqual(voices.listing(), [])
        self.assertFalse((config.CLONED_VOICES_DIR / "ann-reader").exists())


class FakeProbe:
    def __init__(self, project_id, sentences):
        self.center = np.ones(192, np.float32) / np.sqrt(192)
        self.ceiling = 0.76
        self.real_artifacts = 3.3


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def speak(self, model, ref_audio, ref_text, sentences, language="English"):
        if self.fail:
            raise CloneError("The clone server isn’t answering")
        self.calls.append((model, Path(ref_audio), ref_text, sentences))
        assert Path(ref_audio).exists() and sf.info(str(ref_audio)).samplerate == 24000
        t = np.arange(24000 * 3) / 24000
        return 24000, [(0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)]


class TestJob(unittest.TestCase):
    """The job end to end, with a fake clone server and speaker model (no network, no GPU)."""
    def setUp(self):
        db.init()
        self.pid = os.urandom(6).hex()
        db.create_project(self.pid, "en-us", auto_train=True)
        source_dir = config.PROJECTS_DIR / self.pid / "sources"
        source_dir.mkdir(parents=True)
        path = source_dir / "a.wav"
        tone(path, 80.0, 44100)
        self.sid = os.urandom(6).hex()
        db.create_source(self.sid, self.pid, "a.wav", str(path), path.stat().st_size)
        db.update_source(self.sid, status="done", duration=80.0)
        spans = [(0, 9), (12, 22), (25, 36), (40, 52), (55, 57.5), (58, 60.5), (61, 63.5)]
        db.add_clips(self.pid, self.sid, [
            {"start": a, "end": b, "text": f"Sentence {n}.", "words": 3, "conf": 0.9 - n / 100, "unsure": 0.0,
             "snr": 40.0, "clipped": 0.0, "reason": None, "embedding": None} for n, (a, b) in enumerate(spans)],
            done_until=80.0, last_end=80.0, gaps={})
        ids = [c["id"] for c in db.clips(self.pid)]
        db.set_verdicts([(0.9, "keep", "train" if n < 4 else "holdout", i) for n, i in enumerate(ids)])
        db.update_project(self.pid, readiness={"rating": {"key": "good", "label": "Good"}}, status="ready",
                          clone={"model": SMALL})
        for target, value in (("lab.quality.Probe", FakeProbe),
                              ("lab.speaker.embed", lambda clips: np.tile(FakeProbe(0, 0).center, (len(clips), 1)))):
            patch = mock.patch(target, value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_samples_made_and_measured(self):
        fake = FakeClient()
        with mock.patch.object(clone, "client", return_value=fake):
            self.assertEqual(clone.main(self.pid), 0)
        p = db.get_project(self.pid)
        state = p["clone"]
        self.assertEqual((state["model"], state["stage"]), (SMALL, "done"))
        self.assertEqual(len(state["candidates"]), 4)  # the four 9-12 s clips; the short ones aren't tried
        self.assertEqual(len(fake.calls), 4)
        for c in state["candidates"]:
            self.assertAlmostEqual(c["likeness"], 1.0, places=3)
            self.assertIsNotNone(c["artifacts"])
            self.assertTrue(clone.sample_path(self.pid, SMALL, c["clip"]).exists())
            self.assertTrue(clone.ref_path(self.pid, c["clip"]).exists())
        self.assertEqual(state["real_likeness"], 0.76)
        self.assertEqual(fake.calls[0][3], [voices.SAMPLE_TEXT])  # the same passage as every voice
        self.assertTrue(p["message"].startswith("Samples made"))
        self.assertTrue((config.PROJECTS_DIR / self.pid / "holdout" / "holdout.json").exists())

    def test_clone_server_down_leaves_the_project_alone(self):
        with mock.patch.object(clone, "client", return_value=FakeClient(fail=True)):
            self.assertEqual(clone.main(self.pid), 0)
        p = db.get_project(self.pid)
        self.assertEqual(p["status"], "ready")
        self.assertEqual(p["clone"]["stage"], "failed")
        self.assertIn("isn’t answering", p["message"])

    def test_queue_and_auto_start(self):
        with mock.patch.object(config, "CLONE_URL", "http://clone.test"), \
                mock.patch.object(clone, "health", return_value=served(True)):
            db.update_project(self.pid, clone=None)
            jobs.auto_start(self.pid)
            self.assertTrue(db.job_waiting(self.pid, "clone"))
            self.assertFalse(db.job_waiting(self.pid, "train"))  # a cloned voice isn't trained
            self.assertEqual(db.get_project(self.pid)["clone"]["model"], BIG)
            with self.assertRaises(clone.CantClone):
                db.update_project(self.pid, model=clone.PIPER)
                clone.queue(self.pid)  # set to Piper: nothing to clone


if __name__ == "__main__":
    unittest.main()
