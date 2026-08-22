"""
Regression tests for pull_statcast cache handling.

resume=True used to return ANY existing pitches.parquet regardless of date
coverage, so a Colab VM that kept /content/mlb_clean_data from a prior run
silently skipped the Statcast pull and the feature build ran on stale data
(4690 games "through" Aug 22 that actually ended Aug 21). These tests lock
in: fresh cache → skip; stale cache → incremental top-up with dedupe.
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


def _fake_pybaseball(calls: list, frames: dict):
    """Install a fake pybaseball module whose statcast() records calls and
    returns canned frames keyed by 'YYYY-MM-DD_YYYY-MM-DD'."""
    mod = types.ModuleType("pybaseball")

    def statcast(start, end):
        calls.append((start, end))
        return frames.get(f"{start}_{end}")

    mod.statcast = statcast
    return patch.dict(sys.modules, {"pybaseball": mod})


def _frame(dates, pks=None):
    n = len(dates)
    return pd.DataFrame({
        "game_date": pd.to_datetime(dates),
        "game_pk": pks if pks is not None else list(range(1000, 1000 + n)),
        "at_bat_number": [1] * n,
        "pitch_number": [1] * n,
    })


class TestPullStatcastCache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "pitches.parquet"

    def test_fresh_cache_skips_pull(self):
        self.out.write_bytes(b"")
        # Bypass real parquet write: monkey _cache_max_date to simulate a
        # cache that already covers end_date.
        with patch.object(ingestion, "_cache_max_date", return_value=date(2026, 8, 22)):
            with patch.object(
                ingestion, "_chunked_statcast",
                side_effect=AssertionError("must not pull when cache is fresh"),
            ):
                result = ingestion.pull_statcast(
                    "2025-01-01", "2026-08-22", out_path=self.out, resume=True,
                )
        self.assertEqual(result, self.out)

    def test_stale_cache_triggers_incremental_topup(self):
        old = _frame(["2026-08-19", "2026-08-20", "2026-08-21"])
        old.to_parquet(self.out, index=False)
        new = _frame(["2026-08-22"], pks=[7777])

        calls = []
        with patch.object(
            ingestion, "_cache_max_date", return_value=date(2026, 8, 21),
        ), patch.object(
            ingestion, "_chunked_statcast",
            side_effect=lambda s, e, cd, ps: (calls.append((s, e)), [new])[1],
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

    def test_overlap_rows_deduplicated(self):
        # Cache already contains Aug 21; the increment re-delivers it plus a
        # genuinely new day. Overlap must be dropped, not duplicated.
        old = _frame(["2026-08-20", "2026-08-21"], pks=[1, 2])
        old.to_parquet(self.out, index=False)
        inc = _frame(["2026-08-21", "2026-08-22"], pks=[2, 3])

        with patch.object(
            ingestion, "_cache_max_date", return_value=date(2026, 8, 20),
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
            ingestion, "_cache_max_date", return_value=date(2026, 8, 21),
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


if __name__ == "__main__":
    unittest.main()
