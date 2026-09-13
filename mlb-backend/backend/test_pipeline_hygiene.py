"""Regression tests for the v2026.08.24 pipeline-hygiene fixes.

Covers:
- Issue 2: future-dated empty Statcast chunks are EXPECTED (DEBUG, not WARNING)
- Issue 3: Open-Meteo batches skip windows containing no scheduled games
- Issue 4: StatsAPI game-feed weather parses + converts under honest-fill rules
- Issue 5: win_pct_diff warns loudly ONLY on the final diff computation;
  pre-overlay absence is a DEBUG note, and records-present frames ship no NaNs

Imports use the TOP-LEVEL module names exactly as production code does
(``from results import ...``, ``import ingestion``) so patches hit the same
module instances the code under test reads.
"""
import re
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import git
import github_sync
import numpy as np
import pandas as pd

import ingestion
import results as results_mod
from features import add_diff_features
from weather import (
    STADIUMS,
    compute_wind_multiplier,
    statsapi_weather_to_record,
    _fetch_batched_weather,
)


class TestFutureChunkGuard(unittest.TestCase):
    """Empty Statcast chunks entirely at/after today are expected."""

    def test_future_chunk_is_debug_not_warning(self):
        today = date(2026, 8, 24)
        with patch.object(ingestion, "date", _FrozenDate(today)):
            with self.assertNoLogs(ingestion.logger, level="WARNING"):
                with self.assertLogs(ingestion.logger, level="DEBUG") as captured:
                    ingestion._warn_if_core_season_chunk_empty(
                        today, today + timedelta(days=5), "empty response")
        self.assertIn("future-dated", " ".join(captured.output))

    def test_past_core_season_chunk_still_warns(self):
        past = date(2026, 7, 1)
        with patch.object(ingestion, "date", _FrozenDate(date(2026, 8, 24))):
            with self.assertLogs(ingestion.logger, level="WARNING") as captured:
                ingestion._warn_if_core_season_chunk_empty(
                    past, past + timedelta(days=3), "empty response")
        self.assertIn("EMPTY", " ".join(captured.output))


class _FrozenDate:
    """Stand-in for the module-level ``date`` name pinning today()."""

    def __init__(self, today: date):
        self._today = today

    def today(self):  # noqa: N802 - matches datetime.date API
        return self._today


class TestWeatherWindowSkipping(unittest.TestCase):
    """Open-Meteo batches must skip windows containing no scheduled games."""

    def test_gameless_offseason_windows_are_skipped(self):
        requested = []

        def fake_batch(locations, chunk_start, chunk_end, source="", **kw):
            requested.append((chunk_start, chunk_end))
            return {}

        needed = {date(2026, 3, 28), date(2026, 4, 15)}
        with patch("time.sleep"), \
             patch("weather._fetch_batch_range", side_effect=fake_batch):
            out = _fetch_batched_weather(
                [("NYY", STADIUMS["NYY"])],
                date(2025, 11, 1), date(2026, 4, 30),
                needed_days=needed,
            )
        self.assertEqual(out, {})
        self.assertTrue(requested, "in-season windows must still be fetched")
        for start, end in requested:
            overlaps_needed = any(start <= d <= end for d in needed)
            self.assertTrue(
                overlaps_needed,
                f"window {start}→{end} contains no game dates but was fetched")

    def test_none_needed_days_preserves_fetch_everything(self):
        requested = []

        def fake_batch(locations, chunk_start, chunk_end, source="", **kw):
            requested.append((chunk_start, chunk_end))
            return {}

        with patch("time.sleep"), \
             patch("weather._fetch_batch_range", side_effect=fake_batch):
            _fetch_batched_weather(
                [("NYY", STADIUMS["NYY"])],
                date(2025, 12, 1), date(2025, 12, 20),
                needed_days=None,
            )
        # Old behavior: every window in range gets a request.
        spans = [(s, e) for s, e in requested]
        self.assertTrue(any(s <= date(2025, 12, 10) <= e for s, e in spans),
                        "December window must be fetched when no game knowledge")


class TestStatsapiWeatherFiller(unittest.TestCase):
    """gameData.weather parses and converts under the honest-fill rules."""

    def test_feed_parsing(self):
        class Resp:
            status_code = 200

            def json(self):
                return {"gameData": {"weather": {
                    "condition": "Clear", "temp": 70, "wind": "8 mph, In from LF"}}}

        with patch.object(results_mod.requests, "get", return_value=Resp()), \
             patch("time.sleep"):
            out = results_mod.fetch_statsapi_weather([12345])
        self.assertEqual(out[12345]["wind_mph"], 8.0)
        self.assertEqual(out[12345]["temp_f"], 70.0)

    def test_record_conversion_out_wind_positive(self):
        rec = statsapi_weather_to_record(
            {"temp_f": 72.0, "wind_mph": 9.0, "wind_text": "9 mph, Out to CF"},
            home_team="NYY", venue="Yankee Stadium")
        bearing = STADIUMS["NYY"]["bearing"]
        expected = compute_wind_multiplier(bearing, 9.0 * 1.60934, bearing)
        self.assertTrue(rec["available"])
        self.assertAlmostEqual(rec["wind_multiplier"], expected, places=6)
        self.assertGreater(rec["wind_multiplier"], 0.0)   # tailwind
        self.assertAlmostEqual(rec["temp_c"], 22.222, places=2)
        # Honest nulls: feed carries neither RH nor pressure → density NULL.
        self.assertIsNone(rec["air_density"])
        self.assertIsNone(rec["rh_pct"])

    def test_record_conversion_in_wind_negative(self):
        rec = statsapi_weather_to_record(
            {"temp_f": 70.0, "wind_mph": 8.0, "wind_text": "8 mph, In from LF"},
            home_team="NYY", venue="Yankee Stadium")
        self.assertTrue(rec["available"])
        self.assertLess(rec["wind_multiplier"], 0.0)      # headwind

    def test_unusable_observation_marks_unavailable(self):
        rec = statsapi_weather_to_record(
            {"temp_f": None, "wind_mph": None, "wind_text": "calm"},
            home_team="NYY", venue="Yankee Stadium")
        self.assertFalse(rec["available"])
        self.assertIn("unusable", rec["source"])


def _base_frame(**extra) -> pd.DataFrame:
    n = 4
    base = {
        "game_pk": list(range(1, n + 1)),
        "game_date": pd.date_range("2026-08-20", periods=n).strftime("%Y-%m-%d"),
        "home_team": ["NYY"] * n,
        "away_team": ["BOS"] * n,
        "home_elo": [1500.0] * n,
        "away_elo": [1480.0] * n,
    }
    base.update(extra)
    return pd.DataFrame(base)


class TestWinPctDiffStageAwareness(unittest.TestCase):
    """Loud warning ONLY on the final computation; pre-overlay is DEBUG."""

    RECORDS = dict(home_wins=[50] * 4, home_losses=[40] * 4,
                   away_wins=[45] * 4, away_losses=[45] * 4)

    def test_records_present_no_warning_no_nans(self):
        df = _base_frame(**self.RECORDS)
        with self.assertNoLogs(level="WARNING"):
            out = add_diff_features(df, require_records=True)
        self.assertTrue(np.isfinite(out["win_pct_diff"]).all(),
                        "win_pct_diff must never be NaN when records exist")
        # Smoothed rates: equal-ish records → small diff, not garbage.

    def test_pre_overlay_absence_is_debug_only(self):
        df = _base_frame()
        with self.assertLogs("features", level="DEBUG") as captured:
            out = add_diff_features(df, require_records=False)
        joined = " ".join(captured.output)
        self.assertNotIn(" WARNING ", joined)
        self.assertIn("not present yet", joined)
        self.assertTrue(pd.isna(out["win_pct_diff"]).all())

    def test_final_computation_missing_records_warns(self):
        df = _base_frame()
        with self.assertLogs("features", level="WARNING") as captured:
            out = add_diff_features(df, require_records=True)
        self.assertIn("FINAL computation", " ".join(captured.output))
        self.assertTrue(pd.isna(out["win_pct_diff"]).all())


class TestSyncRemoteTipAndPushRetry(unittest.TestCase):
    """The 2026-09-13 rejected-push race, pinned with REAL git mechanics.

    A warm sync clone (/content/mlb_sync_tmp) never fetched: after any
    other push landed, the daily run committed on a stale snapshot and its
    artifact push was hard-rejected (non-fast-forward), losing the run's
    delivery. sync_remote_tip must heal a reused clone to the current
    remote tip, and push_with_retry must recover by re-syncing + replaying
    the run's restage() onto the new tip.

    Every test here runs against local throwaway git repos (a bare origin
    plus clones), so the rejection is genuine, not mocked.
    """

    def _init_repo(self, path: Path, bare: bool = False):
        repo = git.Repo.init(str(path), bare=bare, initial_branch="main")
        if not bare:
            with repo.config_writer() as cw:
                cw.set_value("user", "email", "ci@example.com")
                cw.set_value("user", "name", "CI Test")
        return repo

    def _commit_file(self, repo, fname: str, content: str, msg: str):
        p = Path(repo.working_tree_dir) / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        repo.index.add([fname])
        return repo.index.commit(msg)

    def _seed_origin(self, root: Path):
        origin_path = root / "origin.git"
        origin = self._init_repo(origin_path, bare=True)
        # file:/// URI: a plain Windows path (C:\...) is parsed as scp syntax
        # (drive letter = hostname) and the push dies in SSH.
        origin_url = origin_path.as_uri()
        seed = self._init_repo(root / "seed")
        self._commit_file(seed, "mlb/data_delivery/seed.csv", "pk\n", "seed")
        seed.create_remote("origin", origin_url)
        seed.remote("origin").push("main")
        return origin, seed

    def test_sync_remote_tip_heals_stale_clone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            _origin, seed = self._seed_origin(root)
            branch = seed.active_branch.name
            clone = git.Repo.clone_from(
                (root / "origin.git").as_uri(),
                str(root / "clone"), branch=branch)
            with clone.config_writer() as cw:
                cw.set_value("user", "email", "ci@example.com")
                cw.set_value("user", "name", "CI Test")
            # Remote moves ahead; the clone never fetches (the stale-clone
            # condition that caused the rejected pushes).
            self._commit_file(seed, "mlb/data_delivery/newer_20260913.csv",
                              "x\n", "second push")
            seed.remote("origin").push(branch)
            stray = Path(clone.working_tree_dir) / "stray.txt"
            stray.write_text("junk", encoding="utf-8")

            self.assertNotEqual(clone.head.commit.hexsha,
                                seed.head.commit.hexsha,
                                "precondition: clone must be stale")
            github_sync.sync_remote_tip(clone, branch)
            self.assertEqual(clone.head.commit.hexsha, seed.head.commit.hexsha,
                             "clone must sit on the CURRENT remote tip")
            self.assertFalse(stray.exists(),
                             "stray untracked files must be cleaned")

    def test_push_with_retry_replays_after_rejection(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            _origin, seed = self._seed_origin(root)
            clone = git.Repo.clone_from(str(seed.remote("origin").url),
                                        str(root / "clone"), branch="main")
            with clone.config_writer() as cw:
                cw.set_value("user", "email", "ci@example.com")
                cw.set_value("user", "name", "CI Test")
            # A concurrent actor advances the remote after our clone.
            self._commit_file(seed, "other/concurrent.csv", "c\n", "concurrent")
            seed.remote("origin").push("main")
            # Simulate the daily run: stage + commit on the STALE clone.
            artifact = Path(clone.working_tree_dir) / "mlb/data_delivery/run.csv"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("run1", encoding="utf-8")
            clone.index.add(["mlb/data_delivery/run.csv"])
            clone.index.commit("Update MLB features + predictions")

            def restage():
                # Mirror production _restage_artifacts: re-copy the run's
                # files (reset --hard dropped them) and commit again.
                artifact.write_text("run1", encoding="utf-8")
                clone.index.add(["mlb/data_delivery/run.csv"])
                clone.index.commit("Update MLB features + predictions")

            github_sync.push_with_retry(clone, "main", restage=restage,
                                        attempts=3, log=lambda msg: None)
            origin_repo = git.Repo(str(root / "origin.git"))
            blobs = [e.path for e in
                     origin_repo.commit("main").tree.traverse()
                     if e.type == "blob"]
            self.assertIn("mlb/data_delivery/run.csv", blobs,
                          "the replayed run artifact must be on the remote")
            self.assertIn("other/concurrent.csv", blobs,
                          "the concurrent commit must not be clobbered")
            self.assertEqual(origin_repo.commit("main").hexsha,
                             clone.head.commit.hexsha,
                             "remote tip must equal the retry's final commit")

    def test_push_with_retry_success_without_retry(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            _origin, seed = self._seed_origin(root)
            clone = git.Repo.clone_from(str(seed.remote("origin").url),
                                        str(root / "clone"), branch="main")
            with clone.config_writer() as cw:
                cw.set_value("user", "email", "ci@example.com")
                cw.set_value("user", "name", "CI Test")
            self._commit_file(clone, "mlb/data_delivery/fresh.csv", "f\n",
                              "fresh run")
            calls = []
            github_sync.push_with_retry(clone, "main",
                                        restage=lambda: calls.append(1),
                                        attempts=3, log=lambda msg: None)
            self.assertEqual(calls, [],
                             "a healthy tip must never trigger a restage")

    def test_push_with_retry_raises_when_attempts_exhausted(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            root = Path(td)
            _origin, seed = self._seed_origin(root)
            clone = git.Repo.clone_from(str(seed.remote("origin").url),
                                        str(root / "clone"), branch="main")
            with clone.config_writer() as cw:
                cw.set_value("user", "email", "ci@example.com")
                cw.set_value("user", "name", "CI Test")
            self._commit_file(seed, "other/concurrent.csv", "c\n", "concurrent")
            seed.remote("origin").push("main")
            self._commit_file(clone, "mlb/data_delivery/run.csv", "r\n",
                              "stale-base run commit")
            restage_calls = []
            with self.assertRaises(RuntimeError):
                github_sync.push_with_retry(
                    clone, "main",
                    restage=lambda: restage_calls.append(1), attempts=1)
            self.assertEqual(restage_calls, [],
                             "no restage may run after the final attempt")


if __name__ == "__main__":
    unittest.main()
