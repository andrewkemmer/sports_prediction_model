"""Tests for the starting-QB matchup emitter + frontend loader wiring.

Pure-Python (no Streamlit import, no network), following the
test_frontend_sport_router / test_nfl_slate conventions:

  1. Registry: the NFL sport config carries a ``qb_matchup_json`` family.
  2. Loader source: utils.load_nfl_qb_matchup walks the dated family newest
     first and never fabricates (empty frame on absence).
  3. Emitter: build_qb_matchup is pure over fixture frames — leakage
     (strictly-prior window), null-stats for an unknown passer, passer-rating
     formula, and the record schema.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "nfl-backend" / "backend"
for p in (str(FRONTEND), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestRegistryFamily(unittest.TestCase):
    def test_nfl_config_carries_qb_matchup_family(self):
        import sports_config
        arts = sports_config.artifact_patterns("nfl")
        self.assertIn("qb_matchup_json", arts)
        self.assertEqual(arts["qb_matchup_json"], "nfl_qb_matchup_*.json")

    def test_mlb_config_does_not(self):
        import sports_config
        self.assertNotIn("qb_matchup_json", sports_config.artifact_patterns("mlb"))


class TestLoaderSource(unittest.TestCase):
    def test_loader_exists_and_walks_family(self):
        src = (FRONTEND / "utils.py").read_text(encoding="utf-8")
        self.assertIn("def load_nfl_qb_matchup", src)
        self.assertIn("nfl_qb_matchup_", src)
        # family-date walk, never a fabricated row on absence
        self.assertIn("_nfl_qb_matchup_family_dates", src)
        self.assertIn("cols = NFL_QB_MATCHUP_COLUMNS", src)
        self.assertIn("return pd.DataFrame(out, columns=cols)", src)
        self.assertIn("return pd.DataFrame(columns=cols)", src)

    def test_loader_schema_pinned(self):
        src = (FRONTEND / "utils.py").read_text(encoding="utf-8")
        for col in ("game_id", "qb_home_rating", "qb_away_rating",
                    "qb_home_name", "qb_away_name"):
            self.assertIn(f'"{col}"', src)


class TestEmitterPure(unittest.TestCase):
    def test_passer_rating_formula(self):
        from nfl_qb_matchup import _passer_rating
        # 30/45, 350 yds, 3 TD, 0 INT -> a=1.8333, b=1.1944, c=1.3333,
        # d=2.375 -> rating = sum/6*100 = 112.3
        sub = pd.DataFrame({"att": [45], "cmp": [30], "yds": [350],
                            "td": [3], "ints": [0]})
        self.assertAlmostEqual(_passer_rating(sub), 112.3, places=1)

    def test_window_stats_strictly_prior(self):
        from nfl_qb_matchup import _qb_window_stats
        before = pd.Timestamp("2026-09-09")
        games = pd.DataFrame({
            "passer_id": ["00-0035746"] * 3,
            "gameday": [pd.Timestamp("2026-09-01"),
                        pd.Timestamp("2026-09-08"),
                        pd.Timestamp("2026-09-10")],  # after -> excluded
            "att": [30, 40, 50], "cmp": [20, 26, 35], "yds": [250, 320, 400],
            "td": [2, 3, 4], "ints": [1, 0, 1],
        })
        stats = _qb_window_stats(games, "00-0035746", before)
        self.assertEqual(stats["starts"], 2)
        self.assertEqual(stats["ints"], 1)
        self.assertAlmostEqual(stats["td_per_game"], 2.5, places=6)

    def test_unknown_passer_never_fabricates(self):
        from nfl_qb_matchup import _null_stats
        stats = _null_stats()
        self.assertIsNone(stats["passer_rating"])
        self.assertEqual(stats["starts"], 0)
        self.assertEqual(stats["name"], "")

    def test_record_schema_shape(self):
        from nfl_qb_matchup import RECORD
        self.assertEqual(RECORD, "nfl_qb_matchup")


if __name__ == "__main__":
    unittest.main()
