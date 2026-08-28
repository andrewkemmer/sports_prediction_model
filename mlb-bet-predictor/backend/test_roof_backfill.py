"""Tests for the roof-state cache backfill and pipeline auto top-up.

Verifies:
- Backfill idempotency (re-running _fetch_roofs.py adds nothing).
- 2024 coverage: all retractable-home games have cached roof state.
- Dome refinement: refine_dome_game_level produces 0 UNKNOWN retractable games.
- Pipeline helper functions (_load_roof_cache, _topup_roof_cache,
  _roof_from_statsapi_condition).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_delivery"
BACKEND = ROOT / "backend"


class TestRoofCacheBackfillIdempotent(unittest.TestCase):
    """Re-running the backfill script adds no new entries."""

    def test_idempotent_rerun(self):
        """The current cache covers all retractable-home games in the CSV."""
        from features import RETRACTABLE_ROOF_TEAMS, load_roof_cache

        df = pd.read_csv(DATA / "game_level_features.csv")
        cache = load_roof_cache(DATA / "statsapi_roof_cache.json")
        self.assertGreater(len(cache), 1000,
                           "Cache should have 1000+ entries after backfill")

        home = df["home_team"].astype(str).str.upper().str.strip()
        retract_mask = home.isin(RETRACTABLE_ROOF_TEAMS)
        retract_pks = set(int(pk) for pk in
                          df.loc[retract_mask, "game_pk"].tolist())
        cached_pks = set(cache.keys())
        missing = retract_pks - cached_pks
        self.assertEqual(len(missing), 0,
                         f"Backfill incomplete: {len(missing)} retractable "
                         f"games still missing from cache")


class TestRoofCache2024Coverage(unittest.TestCase):
    """All 2024 retractable-home games have cached roof state."""

    def test_2024_fully_covered(self):
        from features import RETRACTABLE_ROOF_TEAMS, load_roof_cache

        df = pd.read_csv(DATA / "game_level_features.csv")
        cache = load_roof_cache(DATA / "statsapi_roof_cache.json")

        home = df["home_team"].astype(str).str.upper().str.strip()
        yr_2024 = df["game_date"].astype(str).str[:4] == "2024"
        retract_mask = home.isin(RETRACTABLE_ROOF_TEAMS) & yr_2024
        pks_2024 = set(int(pk) for pk in
                       df.loc[retract_mask, "game_pk"].tolist())
        missing = pks_2024 - set(cache.keys())
        self.assertEqual(len(missing), 0,
                         f"2024 retractable games missing: {missing}")

    def test_per_team_counts(self):
        """Each retractable team has all its home games cached."""
        from features import RETRACTABLE_ROOF_TEAMS, load_roof_cache

        df = pd.read_csv(DATA / "game_level_features.csv")
        cache = load_roof_cache(DATA / "statsapi_roof_cache.json")

        home = df["home_team"].astype(str).str.upper().str.strip()
        retract_mask = home.isin(RETRACTABLE_ROOF_TEAMS)
        retract_df = df[retract_mask].copy()
        retract_df["_pk"] = retract_df["game_pk"].astype(int)
        retract_df["_cached"] = retract_df["_pk"].apply(
            lambda pk: pk in cache)

        by_team = retract_df.groupby("home_team")["_cached"].agg(
            ["sum", "count"])
        for team, row in by_team.iterrows():
            self.assertEqual(
                row["sum"], row["count"],
                f"{team}: {int(row['count'] - row['sum'])} games uncached "
                f"({int(row['sum'])}/{int(row['count'])})")


class TestDomeRefinementZeroUnknown(unittest.TestCase):
    """refine_dome_game_level resolves ALL retractable-home games."""

    def test_zero_unknown_retractable(self):
        from features import (RETRACTABLE_ROOF_TEAMS,
                              load_roof_cache, refine_dome_game_level)

        df = pd.read_csv(DATA / "game_level_features.csv")
        cache = load_roof_cache(DATA / "statsapi_roof_cache.json")
        games = refine_dome_game_level(df.copy(), roof_states=cache)

        home = games["home_team"].astype(str).str.upper().str.strip()
        retract_mask = home.isin(RETRACTABLE_ROOF_TEAMS)
        retract_games = games[retract_mask]

        # dome_is_neutral_game must be 0 (open) or 1 (closed), never NaN
        na_count = int(retract_games["dome_is_neutral_game"].isna().sum())
        self.assertEqual(na_count, 0,
                         f"{na_count} retractable games have NaN "
                         "dome_is_neutral_game (UNKNOWN roof state)")

    def test_open_games_use_real_weather(self):
        """Open retractable games should NOT be treated as domed."""
        from features import (RETRACTABLE_ROOF_TEAMS,
                              load_roof_cache, refine_dome_game_level)

        df = pd.read_csv(DATA / "game_level_features.csv")
        cache = load_roof_cache(DATA / "statsapi_roof_cache.json")
        games = refine_dome_game_level(df.copy(), roof_states=cache)

        home = games["home_team"].astype(str).str.upper().str.strip()
        retract_mask = home.isin(RETRACTABLE_ROOF_TEAMS)
        retract = games[retract_mask].copy()
        retract["_pk"] = retract["game_pk"].astype(int)
        open_mask = retract["_pk"].apply(lambda pk: cache.get(pk) == "open")
        open_games = retract[open_mask]

        if len(open_games) > 0:
            all_zero = (open_games["dome_is_neutral_game"] == 0.0).all()
            self.assertTrue(
                all_zero,
                f"Some open retractable games are not flagged as open: "
                f"{int((open_games['dome_is_neutral_game'] != 0.0).sum())}")


class TestPipelineRoofHelpers(unittest.TestCase):
    """Test the pipeline helper functions for roof cache and top-up."""

    def test_load_roof_cache_empty_path(self):
        from pipeline import _load_roof_cache
        with tempfile.TemporaryDirectory() as td:
            result = _load_roof_cache(Path(td) / "nonexistent.json")
            self.assertEqual(result, {})

    def test_load_roof_cache_valid(self):
        from pipeline import _load_roof_cache
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False) as f:
            json.dump({"12345": "open", "67890": "closed", "99999": "bogus"},
                      f)
            f.flush()
            result = _load_roof_cache(Path(f.name))
        self.assertEqual(result, {12345: "open", 67890: "closed"})
        self.assertNotIn(99999, result, "bogus values should be filtered")

    def test_roof_from_statsapi_condition(self):
        from pipeline import _roof_from_statsapi_condition
        self.assertEqual(_roof_from_statsapi_condition("Roof Closed"),
                         "closed")
        self.assertEqual(_roof_from_statsapi_condition("Dome"), "closed")
        self.assertEqual(_roof_from_statsapi_condition("Indoor"), "closed")
        self.assertEqual(_roof_from_statsapi_condition("Clear"), "open")
        self.assertEqual(_roof_from_statsapi_condition("Sunny"), "open")
        self.assertEqual(_roof_from_statsapi_condition("Roof Open"), "open")
        self.assertIsNone(_roof_from_statsapi_condition(None))
        self.assertIsNone(_roof_from_statsapi_condition(""))
        self.assertIsNone(_roof_from_statsapi_condition("   "))

    def test_topup_roof_cache_idempotent_when_complete(self):
        """When no games are missing, top-up returns unchanged dict."""
        from pipeline import _topup_roof_cache

        df = pd.DataFrame({
            "game_pk": [100, 200, 300],
            "home_team": ["HOU", "TEX", "NYY"],
        })
        cache = {100: "open", 200: "closed", 300: "open"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False) as f:
            json.dump({str(k): v for k, v in cache.items()}, f)
            cache_path = Path(f.name)

        result = _topup_roof_cache(df, dict(cache), cache_path,
                                   budget_sec=5.0)
        # No network calls should be needed; cache unchanged
        self.assertEqual(len(result), 3)
        self.assertEqual(result, cache)

    def test_topup_roof_cache_fetches_missing(self):
        """Missing retractable games get fetched from the StatsAPI feed."""
        from pipeline import _topup_roof_cache

        df = pd.DataFrame({
            "game_pk": [100, 200, 300],
            "home_team": ["HOU", "TEX", "NYY"],  # NYY is not retractable
        })
        # Only HOU cached, TEX missing
        cache = {100: "open"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False) as f:
            json.dump({str(k): v for k, v in cache.items()}, f)
            cache_path = Path(f.name)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "gameData": {"weather": {"condition": "Roof Closed"}}}

        # Mock requests.get and time.monotonic inside the function scope
        with patch("requests.get", return_value=mock_resp):
            with patch("time.monotonic", side_effect=[0.0] * 200):
                result = _topup_roof_cache(df, dict(cache), cache_path,
                                           budget_sec=128.0)

        # TEX (200) should now be cached as "closed"
        self.assertIn(200, result)
        self.assertEqual(result[200], "closed")
        # HOU (100) still there
        self.assertEqual(result[100], "open")
        # NYY (300) not retractable, not fetched
        self.assertNotIn(300, result)


class TestRoofCacheArtifactExists(unittest.TestCase):
    """The committed cache artifact survives cleanup."""

    def test_cache_file_exists(self):
        self.assertTrue(DATA.exists(), "data_delivery directory missing")
        self.assertTrue(
            (DATA / "statsapi_roof_cache.json").exists(),
            "statsapi_roof_cache.json missing from data_delivery/")

    def test_cache_is_valid_json(self):
        path = DATA / "statsapi_roof_cache.json"
        data = json.loads(path.read_text())
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 1000,
                           "Cache should have 1000+ entries")

    def test_cache_values_are_valid(self):
        path = DATA / "statsapi_roof_cache.json"
        data = json.loads(path.read_text())
        valid = {"open", "closed"}
        invalid = {k: v for k, v in data.items() if v not in valid}
        self.assertEqual(len(invalid), 0,
                         f"Invalid cache values: {invalid}")


if __name__ == "__main__":
    unittest.main()
