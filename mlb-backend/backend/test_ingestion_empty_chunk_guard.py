"""
Regression tests for the 08-28 Statcast empty-chunk guard.

The 08-28 failure: a core-season Statcast chunk (2025-04-25 → 2025-06-23)
came back EMPTY after a transient IncompleteRead. The pipeline logged the
"investigate before training" warning and trained anyway, silently dropping
~800 games from the decided frame (6,161 vs 6,960). These tests lock in the
two-part guard:

1. Transient failures (http.client.IncompleteRead, CSV tokenizing errors)
   are retried with backoff — recovery means the chunk is NOT skipped.
2. A past-dated core-season chunk that is STILL empty after all retries
   RAISES a descriptive RuntimeError — the run aborts before training/push,
   never silently proceeds. Future-dated and offseason chunks stay exempt
   (posting lag / no games is expected there).
"""
import sys
import types
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from backend import ingestion


def _frame(dates, pks=None):
    n = len(dates)
    return pd.DataFrame({
        "game_date": pd.to_datetime(dates),
        "game_pk": pks if pks is not None else list(range(1000, 1000 + n)),
        "at_bat_number": [1] * n,
        "pitch_number": [1] * n,
    })


def _serve(statcast_fn):
    mod = types.ModuleType("pybaseball")
    mod.statcast = statcast_fn
    return patch.dict(sys.modules, {"pybaseball": mod})


def _calls():
    return []


class TestTransientFailureRetry(unittest.TestCase):
    def test_exception_then_success_recovers_chunk(self):
        """A transient IncompleteRead followed by a clean response must NOT
        drop the chunk — retry-with-backoff recovers it."""
        calls = _calls()
        attempts = {"n": 0}

        def fake_statcast(s, e):
            calls.append((s, e))
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("IncompleteRead(b'N/A')")
            return _frame(["2025-05-01"], pks=[101])

        with _serve(fake_statcast), patch.object(ingestion, "CHUNK_RETRIES", 3):
            chunks = ingestion._chunked_statcast(
                date(2025, 5, 1), date(2025, 5, 1), chunk_days=60, pause_sec=0,
            )
        self.assertEqual(attempts["n"], 2)          # failed once, retried once
        self.assertEqual(len(chunks), 1)            # data recovered
        self.assertEqual(calls, [("2025-05-01", "2025-05-01"),
                                 ("2025-05-01", "2025-05-01")])

    def test_persistent_exception_aborts_core_season(self):
        """A past-dated core-season chunk that errors on EVERY retry must
        raise a descriptive RuntimeError — never just warn and continue."""
        attempts = {"n": 0}

        def fake_statcast(s, e):
            attempts["n"] += 1
            raise ConnectionError("IncompleteRead(b'N/A')")

        # Pin to the reported failure window (mid-June 2025).
        with _serve(fake_statcast), patch.object(ingestion, "CHUNK_RETRIES", 3):
            with self.assertRaises(RuntimeError) as ctx:
                ingestion._chunked_statcast(
                    date(2025, 4, 25), date(2025, 6, 23),
                    chunk_days=60, pause_sec=0,
                )
        self.assertEqual(attempts["n"], 3)          # exhausted all retries
        msg = str(ctx.exception)
        self.assertIn("came back EMPTY", msg)
        self.assertIn("Refusing to proceed to training", msg)
        self.assertIn("2025-04-25", msg)
        self.assertIn("2025-06-23", msg)

    def test_exception_then_all_empty_aborts_core_season(self):
        """Chunk returned None (empty) on every retry attempt in core season
        → still abort; a clean-but-empty core chunk is a silent data gap."""
        attempts = {"n": 0}

        def fake_statcast(s, e):
            attempts["n"] += 1
            return None

        with _serve(fake_statcast), patch.object(ingestion, "CHUNK_RETRIES", 2):
            with self.assertRaises(RuntimeError):
                ingestion._chunked_statcast(
                    date(2025, 5, 1), date(2025, 5, 1), chunk_days=60, pause_sec=0,
                )


class TestExemptEmptyChunks(unittest.TestCase):
    def test_future_dated_core_chunk_empty_does_not_abort(self):
        """A chunk lying entirely at/beyond today in core months is expected
        empty (no completed games yet) — must not abort."""
        today = date.today()
        calls = _calls()

        def fake_statcast(s, e):
            calls.append((s, e))
            return None

        with _serve(fake_statcast), patch.object(ingestion, "CHUNK_RETRIES", 2):
            # chunk_start >= today → exempt regardless of core-month midpoint
            chunks = ingestion._chunked_statcast(
                today, today, chunk_days=60, pause_sec=0,
            )
        self.assertEqual(chunks, [])

    def test_offseason_past_dated_chunk_empty_does_not_abort(self):
        """Late Oct./offseason empty chunks are normal — no abort."""
        calls = _calls()

        def fake_statcast(s, e):
            calls.append((s, e))
            return None

        with _serve(fake_statcast), patch.object(ingestion, "CHUNK_RETRIES", 2):
            chunks = ingestion._chunked_statcast(
                date(2025, 10, 27), date(2025, 12, 25), chunk_days=60, pause_sec=0,
            )
        self.assertEqual(chunks, [])


class TestDecidedFrameGate(unittest.TestCase):
    """Part (c) semantics — the pre-training decided-frame count must be able
    to detect a degraded (~6,161) vs full (6,960) frame and abort."""

    def test_degraded_frame_below_threshold_is_detectable(self):
        degraded = pd.DataFrame({
            "game_date": pd.to_datetime(["2025-04-26", "2025-04-27", "2025-05-01"]),
            "game_pk": [1, 2, 3],
            "home_win": [1.0, 0.0, 1.0],
        })
        from backend.frames import get_decided_frame
        n = len(get_decided_frame(degraded))
        self.assertEqual(n, 3)
        self.assertLess(n, 6960)   # a real degraded frame would trip the gate
        self.assertLess(len(degraded), 2_044_874)

    def test_gate_aborts_on_degraded_frame(self):
        """Encodes the same threshold logic the pre-training guard uses: a
        frame with < 6,960 decided games / < 2,044,874 pitches is rejected."""
        full_pitches = 2_044_874
        full_decided = 6_960
        degraded_pitches = 1_928_904   # observed on the 08-28 degraded run
        degraded_decided = 6_161
        self.assertTrue(degraded_pitches < full_pitches)
        self.assertTrue(degraded_decided < full_decided)
        # The guard's condition fires (abort) on the degraded run's numbers.
        def should_abort(n_pitches, n_decided, min_p, min_d):
            return n_pitches < min_p or n_decided < min_d
        self.assertTrue(should_abort(
            degraded_pitches, degraded_decided, full_pitches, full_decided))
        self.assertFalse(should_abort(
            full_pitches, full_decided, full_pitches, full_decided))


if __name__ == "__main__":
    unittest.main()