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

    def test_fresh_cache_skips_full_pull_but_refreshes_tail(self):
        # A valid cache whose bounds cover the whole request
        _frame(["2025-04-01", "2026-08-22"], pks=[1, 2]).to_parquet(self.out, index=False)
        # Cache covers the full requested range — the 2025 history is NOT
        # re-pulled, but the trailing REFRESH_TAIL_DAYS window still is, so
        # games cached mid-progress get their real finals.
        calls = []
        def fake_pull(s, e, cd, ps):
            calls.append((s, e))
            return [_frame([str(s)], pks=[7777])]
        with patch.object(
            ingestion, "_cache_bounds", return_value=(date(2025, 1, 1), date(2026, 8, 22)),
        ), patch.object(
            ingestion, "_chunked_statcast", side_effect=fake_pull,
        ):
            ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        self.assertEqual(calls, [(date(2026, 8, 20), date(2026, 8, 22))])

    def test_stale_cache_triggers_forward_topup(self):
        old = _frame(["2026-08-19", "2026-08-20", "2026-08-21"], pks=[1, 2, 3])
        old.to_parquet(self.out, index=False)

        # The top-up re-pulls the last REFRESH_TAIL_DAYS days PLUS any new day:
        # games cached mid-game must get their real finals.
        def fake_pull(s, e, cd, ps):
            assert (s, e) == (date(2026, 8, 20), date(2026, 8, 22))
            return [_frame(["2026-08-20", "2026-08-21", "2026-08-22"],
                           pks=[2, 4, 5])]

        with patch.object(
            ingestion, "_cache_bounds",
            return_value=(date(2025, 1, 1), date(2026, 8, 21)),
        ), patch.object(
            ingestion, "_chunked_statcast", side_effect=fake_pull,
        ) as mock_pull:
            ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        mock_pull.assert_called_once()
        args = mock_pull.call_args[0]
        self.assertEqual(args[0], date(2026, 8, 20))   # tail refresh window
        self.assertEqual(args[1], date(2026, 8, 22))

        merged = pd.read_parquet(self.out)
        dates = set(pd.to_datetime(merged["game_date"]).dt.date)
        self.assertEqual(dates, {date(2026, 8, 19), date(2026, 8, 20),
                                 date(2026, 8, 21), date(2026, 8, 22)})
        # pks: cached 1,2,3 + re-delivered 2 (dupe) + new 4,5
        self.assertEqual(sorted(merged["game_pk"]), [1, 2, 3, 4, 5])

    def test_earlier_start_extends_history_backwards(self):
        """Extending MLB_START_DATE earlier must back-fill the missing head,
        not silently keep a cache whose min date misses the request."""
        old = _frame(["2025-04-02", "2025-04-03"])
        old.to_parquet(self.out, index=False)

        captured = []
        def fake_pull(s, e, cd, ps):
            captured.append((s, e))
            if s == date(2025, 3, 27):
                return [_frame([str(s)], pks=[int(str(s).replace("-", "")) % 10000])]
            return []  # tail refresh found nothing new

        with patch.object(
            ingestion, "_cache_bounds", return_value=(date(2025, 4, 2), date(2026, 8, 22)),
        ), patch.object(ingestion, "_chunked_statcast", side_effect=fake_pull):
            ingestion.pull_statcast(
                "2025-03-27", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        # Backward gap first (2025-03-27 .. 04-01), then the tail refresh
        self.assertEqual(captured, [
            (date(2025, 3, 27), date(2025, 4, 1)),
            (date(2026, 8, 20), date(2026, 8, 22)),
        ])
        merged = pd.read_parquet(self.out)
        self.assertEqual(len(merged), 3)
        first = min(pd.to_datetime(merged["game_date"]).dt.date)
        self.assertEqual(first, date(2025, 3, 27))

    def test_tail_refresh_fixes_partially_cached_games(self):
        """A game cached mid-progress (one pitch) gains its missing innings
        when the tail refresh re-delivers the completed game."""
        partial = _frame(["2026-08-21"], pks=[9])
        partial.to_parquet(self.out, index=False)
        complete = pd.concat([
            _frame(["2026-08-21"], pks=[9]),
            _frame(["2026-08-21"], pks=[9]).assign(at_bat_number=[2], pitch_number=[3]),
            _frame(["2026-08-22"], pks=[10]),
        ], ignore_index=True)

        with patch.object(
            ingestion, "_cache_bounds",
            return_value=(date(2026, 8, 19), date(2026, 8, 21)),
        ), patch.object(
            ingestion, "_chunked_statcast",
            side_effect=lambda s, e, cd, ps: [complete],
        ):
            ingestion.pull_statcast(
                "2026-08-19", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        merged = pd.read_parquet(self.out)
        self.assertEqual(len(merged[merged["game_pk"] == 9]), 2)   # full game
        self.assertEqual(len(merged[merged["game_pk"] == 10]), 1)

    def test_fresh_looking_cache_with_partial_tail_still_refreshes(self):
        """A prior run stored partial Aug-22 pitches (cache 'covers' end),
        which used to skip ALL repairs — frozen finals persisted forever.
        The tail refresh must run even when cached_hi >= end."""
        # Cache: complete Aug 20 + ONE pitch of an Aug 22 game in progress
        old = pd.concat([
            _frame(["2026-08-20"], pks=[1]),
            _frame(["2026-08-22"], pks=[9]),
        ], ignore_index=True)
        old.to_parquet(self.out, index=False)

        complete_game9 = pd.DataFrame({
            "game_date": pd.to_datetime(["2026-08-22"] * 2),
            "game_pk": [9, 9],
            "at_bat_number": [1, 2],
            "pitch_number": [1, 2],
        })
        captured = []
        def fake_pull(s, e, cd, ps):
            captured.append((s, e))
            return [complete_game9, _frame(["2026-08-22"], pks=[10])]

        with patch.object(
            ingestion, "_cache_bounds",
            return_value=(date(2025, 1, 1), date(2026, 8, 22)),
        ), patch.object(ingestion, "_chunked_statcast", side_effect=fake_pull):
            ingestion.pull_statcast(
                "2025-01-01", "2026-08-22", out_path=self.out,
                chunk_days=60, pause_sec=0, resume=True,
            )
        self.assertEqual(captured, [(date(2026, 8, 20), date(2026, 8, 22))])
        merged = pd.read_parquet(self.out)
        # Game 9 now has both pitches; game 10 arrived
        self.assertEqual(len(merged[merged["game_pk"] == 9]), 2)
        self.assertEqual(len(merged[merged["game_pk"] == 10]), 1)

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


class TestGameTypeFilter(unittest.TestCase):
    """Savant posts pitch data for Spring Training ('S') and Exhibition
    ('E') games; a pull spanning Feb–March must not ingest them."""

    def test_spring_and_exhibition_dropped_regular_and_postseason_kept(self):
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2026-03-15"] * 5),
            "game_pk": [1, 2, 3, 4, 5],
            "at_bat_number": [1] * 5,
            "pitch_number": [1] * 5,
            "game_type": ["S", "E", "R", "W", "D"],
        })
        out = ingestion._normalize_columns(df)
        self.assertEqual(sorted(out["game_pk"]), [3, 4, 5])


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
