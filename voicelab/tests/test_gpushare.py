"""The GPU turn-taking rules (app/gpushare.py), with simulated state files and
a fake clock: no GPU, no Forge, no voice creator running.

    python -m unittest discover -s voicelab/tests      (from the project root)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app import gpushare as g


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "gpu-share"
        self.clock = Clock()
        self.forge = g.ForgeSide(self.dir, uses_gpu=True, clock=self.clock)
        self.lab = g.LabSide(self.dir, "Training the voice “Test”", clock=self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    def tick(self, seconds: float = 1.1):
        """Time passes (past Forge's one-second read cache)."""
        self.clock.advance(seconds)


class TestPlan(unittest.TestCase):
    """What Forge's worker does in each situation."""

    def test_idle_lab_forge_works_normally(self):
        self.assertEqual(g.forge_plan(g.IDLE, busy=False, in_flight=0, pool_up=True),
                         {"start_books": True, "submit": True, "pool": "keep"})

    def test_idle_lab_restarts_a_stopped_pool(self):
        self.assertEqual(g.forge_plan(g.IDLE, False, 0, pool_up=False)["pool"], "start")

    def test_waiting_lab_lets_forge_go_first(self):
        # Forge goes first: it carries on, and starts the books that are waiting too.
        plan = g.forge_plan(g.WAITING, busy=True, in_flight=3, pool_up=True)
        self.assertEqual(plan, {"start_books": True, "submit": True, "pool": "keep"})

    def test_waiting_lab_and_work_waiting_starts_the_pool(self):
        # Training paused for a book: once it has let go, Forge starts its voices.
        plan = g.forge_plan(g.WAITING, busy=False, in_flight=0, pool_up=False, wants=True)
        self.assertEqual(plan, {"start_books": True, "submit": True, "pool": "start"})

    def test_waiting_lab_and_idle_forge_stops_the_pool(self):
        self.assertEqual(g.forge_plan(g.WAITING, busy=False, in_flight=0, pool_up=True)["pool"], "stop")

    def test_holding_lab_finishes_in_flight_then_stops(self):
        self.assertEqual(g.forge_plan(g.HOLDING, busy=True, in_flight=2, pool_up=True),
                         {"start_books": False, "submit": False, "pool": "keep"})
        self.assertEqual(g.forge_plan(g.HOLDING, busy=True, in_flight=0, pool_up=True)["pool"], "stop")

    def test_holding_lab_never_starts_the_pool(self):
        # Even with work waiting: the voices start only after the voice creator lets go.
        for busy in (False, True):
            for wants in (False, True):
                plan = g.forge_plan(g.HOLDING, busy, 0, pool_up=False, wants=wants)
                self.assertEqual(plan["pool"], "keep")
                self.assertFalse(plan["start_books"])

    def test_queued_messages(self):
        self.assertIsNone(g.queued_message(g.IDLE))
        self.assertIsNone(g.queued_message(g.WAITING), "Forge goes first: its books just start")
        self.assertIn("pausing", g.queued_message(g.HOLDING))


class TestHandshake(Base):
    def test_lab_waits_while_forge_voices(self):
        self.forge.report(busy=True, pool_up=True)
        self.lab.request()
        self.tick()
        self.assertEqual(self.forge.lab_state(), g.WAITING)
        ok, why = self.lab.can_take()
        self.assertFalse(ok)
        self.assertIn("Forge is voicing a book", why)

    def test_full_turn(self):
        # Forge is idle with its pool up; the voice creator asks.
        self.forge.report(busy=False, pool_up=True)
        self.lab.request()
        self.tick()
        self.assertFalse(self.lab.can_take()[0], "Forge hasn't stepped aside yet")
        # Forge sees the request, stops its pool and says so.
        lab = self.forge.lab_state()
        self.assertEqual(g.forge_plan(lab, False, 0, True)["pool"], "stop")
        self.forge.report(busy=False, pool_up=False)
        self.assertTrue(json.loads(self.forge.share.forge_path.read_text())["yielded"])
        self.assertEqual(self.lab.can_take(), (True, ""))
        self.lab.hold()
        self.tick()
        self.assertEqual(self.forge.lab_state(), g.HOLDING)
        # A book queued meanwhile waits, with a message saying why.
        plan = g.forge_plan(self.forge.lab_state(), False, 0, False)
        self.assertFalse(plan["start_books"])
        self.assertEqual(plan["pool"], "keep")
        # Done: Forge starts its pool again.
        self.lab.release()
        self.tick()
        self.assertEqual(self.forge.lab_state(), g.IDLE)
        self.assertEqual(g.forge_plan(g.IDLE, False, 0, False)["pool"], "start")
        self.forge.report(busy=False, pool_up=True)
        self.assertFalse(json.loads(self.forge.share.forge_path.read_text())["yielded"])

    def test_a_book_interrupts_training(self):
        # The voice creator is training; a book is queued in Forge.
        self.forge.report(busy=False, pool_up=False)
        self.lab.request()
        self.lab.hold()
        self.tick()
        self.assertFalse(self.lab.forge_wants())
        self.forge.report(busy=False, pool_up=False, lab=self.forge.lab_state(), wants=True)
        self.assertTrue(self.lab.forge_wants(), "training sees Forge wants the GPU and pauses")
        forge = json.loads(self.forge.share.forge_path.read_text())
        self.assertFalse(forge["yielded"], "a Forge with work waiting hasn't stepped aside")
        # Training saves its place and lets go; the job runs again and asks again.
        self.lab.release()
        self.lab.request()
        self.tick()
        self.assertEqual(self.forge.lab_state(), g.WAITING)
        self.assertEqual(g.forge_plan(g.WAITING, False, 0, False, wants=True)["pool"], "start")
        self.forge.report(busy=True, pool_up=True, lab=g.WAITING, wants=False)
        self.assertFalse(self.lab.can_take()[0], "training waits while Forge voices")
        # Forge is done: it steps aside, and training carries on.
        self.assertEqual(g.forge_plan(g.WAITING, False, 0, True)["pool"], "stop")
        self.forge.report(busy=False, pool_up=False, lab=g.WAITING)
        self.assertEqual(self.lab.can_take(), (True, ""))

    def test_stale_lab_claim_is_ignored(self):
        self.lab.request()
        self.lab.hold()
        self.tick()
        self.assertEqual(self.forge.lab_state(), g.HOLDING)
        self.clock.advance(g.STALE_SECONDS + 1)  # the voice creator crashed: no more heartbeats
        self.assertEqual(self.forge.lab_state(), g.IDLE)

    def test_heartbeat_keeps_the_claim(self):
        self.lab.request()
        self.lab.hold()
        for _ in range(10):
            self.clock.advance(60)
            self.lab.heartbeat()
        self.tick()
        self.assertEqual(self.forge.lab_state(), g.HOLDING)

    def test_stale_forge_lets_the_lab_go_ahead(self):
        self.forge.report(busy=True, pool_up=True)
        self.lab.request()
        self.clock.advance(g.STALE_SECONDS + 1)  # Forge's worker is gone
        self.assertEqual(self.lab.can_take(), (True, ""))

    def test_no_forge_at_all(self):
        self.lab.request()
        self.assertEqual(self.lab.can_take(), (True, ""))

    def test_forge_on_the_cpu_never_steps_aside(self):
        forge = g.ForgeSide(self.dir, uses_gpu=False, clock=self.clock)
        self.lab.request()
        self.lab.hold()
        self.tick()
        self.assertEqual(forge.lab_state(), g.IDLE)
        forge.report(busy=True, pool_up=True)
        self.assertEqual(self.lab.can_take(), (True, ""), "a CPU Forge doesn't need the GPU")

    def test_heartbeat_rewrites_only_when_due(self):
        self.forge.report(busy=False, pool_up=True)
        first = json.loads(self.forge.share.forge_path.read_text())["heartbeat"]
        self.clock.advance(1)
        self.forge.report(busy=False, pool_up=True)
        self.assertEqual(json.loads(self.forge.share.forge_path.read_text())["heartbeat"], first)
        self.clock.advance(g.BEAT_SECONDS)
        self.forge.report(busy=False, pool_up=True)
        self.assertGreater(json.loads(self.forge.share.forge_path.read_text())["heartbeat"], first)

    def test_a_change_is_written_at_once(self):
        self.forge.report(busy=False, pool_up=True)
        self.clock.advance(0.5)
        self.forge.report(busy=True, pool_up=True)
        self.assertTrue(json.loads(self.forge.share.forge_path.read_text())["busy"])


class TestFiles(Base):
    def test_corrupt_or_half_written_file_counts_as_gone(self):
        self.dir.mkdir(parents=True)
        self.forge.share.lab_path.write_text('{"state": "hold')
        self.assertEqual(self.forge.lab_state(), g.IDLE)
        self.forge.share.lab_path.write_text("[1, 2]")
        self.tick()
        self.assertEqual(self.forge.lab_state(), g.IDLE)

    def test_unknown_state_counts_as_idle(self):
        g.write(self.forge.share.lab_path, {"heartbeat": self.clock(), "state": "dancing"})
        self.assertEqual(self.forge.lab_state(), g.IDLE)

    def test_heartbeat_from_the_future_is_stale(self):
        g.write(self.forge.share.lab_path, {"heartbeat": self.clock() + 10 * g.STALE_SECONDS, "state": g.HOLDING})
        self.assertEqual(self.forge.lab_state(), g.IDLE)

    def test_write_is_atomic_and_leaves_no_temp_files(self):
        g.write(self.forge.share.lab_path, {"heartbeat": 1, "state": g.WAITING})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["voicelab.json"])

    def test_release_survives_a_read_only_volume(self):
        self.lab.request()
        self.dir.chmod(0o500)
        try:
            self.lab.release()  # mustn't raise: the claim goes stale on its own
        finally:
            self.dir.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
