"""
Regression tests for pull_statcast cache handling.

resume=True used to return ANY existing pitches.parquet regardless of date
coverage, so a Colab VM that kept /content/mlb_clean_data from a prior run
silently skipped the Statcast pull and the feature build ran on stale data.
These tests lock in:

- fresh cache  -> skip
- stale cache  -> incremental forward top-up with dedupe
- earlier start -> backward top-up (history extension)
- resume=False -> full re-pull overwrites the cache

Bound semantics note (verified against the live Savant CSV endpoint):
gt/lt are INCLUSIVE and same-day queries work. Empty results for recent
dates are posting lag, not query semantics.
"""
import sys
import tempfile
import unittest
import types
from datetime import date
from pathlib import Path
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


def _fake_pybaseball(served):
    mod = types.ModuleType("pybaseball")
    mod.statcast = lambda s, e: served.get((s, e))
    return patch.dict(sys.modules, {"pybaseball": mod})


class TestPullStatcastCache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "pitches.parquet"

    def test_fresh_cache_skips_pull(self):
        self.out.write_bytes(b"")
        # Bypass real parquet write: monkey _cache_bounds to simulate a
        # cache that already covers the full requested window.
        with patch.object(
            ingestion, "_cache_bounds", return_value=(date(2025, 1, 1), date(2026, 8, 22)),
        ), patch.object(
            ingestion, "_chunked_statcast",
            side_effect=AssertionError("must not pull when cache covers range"),
        ):
            result = ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out, resume=True,
            )
        self.assertEqual(result, self.out)

    def test_stale_cache_triggers_forward_topup(self):
        old = _frame(["2026-08-19", "2026-08-20", "2026-08-21"])
        old.to_parquet(self.out, index=False)
        new = _frame(["2026-08-22"], pks=[7777])

        with patch.object(
            ingestion, "_cache_bounds",
            return_value=(date(2025, 1, 1), date(2026, 8, 21)),
        ), patch.object(
            ingestion, "_chunked_statcast",
            side_effect=lambda s, e, cd, ps: [_frame([f"{s}"], pks=[7777])],
        ) as mock_pull:
            ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        # Increment starts the day AFTER the cache's max date — never re-pulls
        # the full 2025-01-01 range.
        mock_pull.assert_called_once()
        args = mock_pull.call_args[0]
        self.assertEqual(args[0], date(2026, 8, 22))
        self.assertEqual(args[1], date(2026, 8, 22))

        merged = pd.read_parquet(self.out)
        dates = set(pd.to_datetime(merged["game_date"]).dt.date)
        self.assertEqual(dates, {date(2026, 8, 19), date(2026, 8, 20),
                                 date(2026, 8, 21), date(2026, 8, 22)})

    def test_earlier_start_extends_history_backwards(self):
        """Extending MLB_START_DATE earlier must back-fill the missing head,
        not silently keep a cache whose min date misses the request."""
        old = _frame(["2025-04-02", "2025-04-03"])
        old.to_parquet(self.out, index=False)

        captured = []
        def fake_pull(s, e, cd, ps):
            captured.append((s, e))
            return [_frame([str(s)], pks=[int(str(s).replace("-", "")) % 10000])]

        with patch.object(
            ingestion, "_cache_bounds", return_value=(date(2025, 4, 2), date(2026, 8, 22)),
        ), patch.object(ingestion, "_chunked_statcast", side_effect=fake_pull):
            ingestion.pull_statcast(
                "2025-03-27", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        # Backward gap only: 2025-03-27 .. 2025-04-01
        self.assertEqual(captured, [(date(2025, 3, 27), date(2025, 4, 1))])
        merged = pd.read_parquet(self.out)
        self.assertEqual(len(merged), 3)
        first = min(pd.to_datetime(merged["game_date"]).dt.date)
        self.assertEqual(first, date(2025, 3, 27))

    def test_overlap_rows_deduplicated(self):
        # Cache already contains Aug 21; the increment re-delivers it plus a
        # genuinely new day. Overlap must be dropped, not duplicated.
        old = _frame(["2026-08-20", "2026-08-21"], pks=[1, 2])
        old.to_parquet(self.out, index=False)
        inc = _frame(["2026-08-21", "2026-08-22"], pks=[2, 3])

        with patch.object(
            ingestion, "_cache_bounds", return_value=(date(2025, 1, 1), date(2026, 8, 20)),
        ), patch.object(
            ingestion, "_chunked_statcast",
            side_effect=lambda s, e, cd, ps: [inc],
        ):
            ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        merged = pd.read_parquet(self.out)
        self.assertEqual(len(merged), 3)  # pks 1, 2, 3 — no dupes
        self.assertEqual(sorted(merged["game_pk"]), [1, 2, 3])

    def test_no_new_data_keeps_cache(self):
        old = _frame(["2026-08-21"])
        old.to_parquet(self.out, index=False)

        with patch.object(
            ingestion, "_cache_bounds",
            return_value=(date(2025, 1, 2), date(2026, 8, 21)),
        ), patch.object(
            ingestion, "_chunked_statcast", side_effect=lambda s, e, cd, ps: [],
        ):
            result = ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        self.assertEqual(result, self.out)  # unchanged cache returned
        merged = pd.read_parquet(self.out)
        self.assertEqual(len(merged), 1)

    def test_resume_false_full_repull_overwrites_cache(self):
        """The explicit escape hatch: resume=False re-pulls the FULL range and
        replaces the cache file entirely."""
        stale = _frame(["2026-08-21"], pks=[42])
        stale.to_parquet(self.out, index=False)

        served = {("2025-01-01", "2026-08-22"): _frame(
            ["2025-04-01", "2026-08-21", "2026-08-22"], pks=[1, 2, 3])}
        with patch.object(
            ingestion, "_chunked_statcast",
            side_effect=lambda s, e, cd, ps: [served[(str(s), str(e))]],
        ), _fake_pybaseball({}):
            ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=False,
            )
        merged = pd.read_parquet(self.out)
        self.assertEqual(sorted(merged["game_pk"]), [1, 2, 3])  # stale row gone


class TestChunkedStatcastBounds(unittest.TestCase):
    """Verified against the live Savant CSV endpoint: gt/lt bounds are
    INCLUSIVE and same-day queries work. Empty results for recent dates are
    posting lag, not query semantics. Pins the chunker to passing exact
    inclusive ranges through untouched."""

    def test_exact_range_passed_to_vendor(self):
        requested = []
        mod = types.ModuleType("pybaseball")

        def fake_statcast(s, e):
            requested.append((s, e))
            return _frame(["2026-08-22"], pks=[9])

        mod.statcast = fake_statcast
        with patch.dict(sys.modules, {"pybaseball": mod}):
            chunks = ingestion._chunked_statcast(
                date(2026, 8, 22), date(2026, 8, 22), chunk_days=60, pause_sec=0,
            )
        self.assertEqual(requested, [("2026-08-22", "2026-08-22")])
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
