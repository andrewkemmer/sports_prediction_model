"""NFL game-frame ingestion (nfl_game_frame.py) — validation mirroring the
ingestion spike's go/no-go criteria (nfl-spike-scratch/, GO given).

Pure-function tests (no network, no artifact) pin the decided-frame rules:
post-game rows only, deterministic dedup by game_id (latest gameday wins),
stable chronological order, and the pbp play-count join.

Artifact tests read the real ``data_delivery/nfl_game_level_features.csv``
(like mlb-backend's test_frames_canonical reads game_level_features.csv):
per-season decided counts must match the spike (267/269/285/284/285/285 for
2019-2024) PLUS 2025 = 285 (the model's sealed hold-out; 1,960 decided games
2019-2025), 0 duplicate game_ids, 0 missing scores, and a spot-check of a
known game's score + spread-line sign (2019 W1 KC@JAX: 40-26 per ESPN;
spread -3.5 = away favorite per the nflverse schedules dictionary — positive
= home favored, negative = away favored).

The CSV is a generated artifact (not committed per project guardrails), so
the artifact tests skip gracefully when it is absent (fresh clone); they run
and pass after ``python3 nfl_game_frame.py`` produces the frame.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nfl_game_frame import (  # noqa: E402
    DEFAULT_SEASONS,
    GAME_LEVEL_COLUMNS,
    aggregate_game_frame,
    canonical_decided_frame,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data_delivery" / "nfl_game_level_features.csv"

# Spike-verified per-season decided-game counts (2019-2025). 2025 (285) is the
# sealed hold-out for the moneyline model.
SPIKE_PER_SEASON = {2019: 267, 2020: 269, 2021: 285, 2022: 284, 2023: 285,
                    2024: 285, 2025: 285}


def _minimal_game(**overrides) -> pd.DataFrame:
    """A tiny decided-shaped frame with sensible defaults for rule tests."""
    rows = {
        "game_id": ["2019_01_KC_JAX"],
        "season": [2019],
        "week": [1],
        "game_type": ["REG"],
        "gameday": ["2019-09-08"],
        "away_team": ["KC"],
        "home_team": ["JAX"],
        "away_score": [40],
        "home_score": [26],
        "result": [-14.0],
        "total": [66.0],
        "spread_line": [-3.5],
        "total_line": [49.0],
        "n_plays": [151],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


class TestAggregateAndDecidedRules(unittest.TestCase):
    def test_decided_filter_excludes_null_scores(self):
        """Pregame/undecided rows (any of away_score/home_score/result null)
        never enter the decided frame."""
        game = pd.concat([
            _minimal_game(),
            _minimal_game(game_id="2019_01_X_Y", away_score=None, home_score=None,
                          result=None, gameday="2019-09-08"),
            _minimal_game(game_id="2019_01_Y_Z", result=None, gameday="2019-09-08"),
            _minimal_game(game_id="2019_01_Z_W", home_score=None, gameday="2019-09-08"),
        ], ignore_index=True)
        decided = canonical_decided_frame(game)
        self.assertEqual(len(decided), 1)
        self.assertEqual(decided["game_id"].iloc[0], "2019_01_KC_JAX")

    def test_dedup_latest_gameday_wins_deterministic(self):
        """One row per game_id — the LATEST gameday wins; identical inputs
        produce identical outputs (deterministic)."""
        game = pd.concat([
            _minimal_game(game_id="G1", gameday="2019-09-08", home_score=20, away_score=10),
            _minimal_game(game_id="G1", gameday="2019-09-15", home_score=30, away_score=14),
            _minimal_game(game_id="G1", gameday="2019-09-01", home_score=7, away_score=3),
            _minimal_game(game_id="G2", gameday="2019-09-08"),
        ], ignore_index=True)
        out = canonical_decided_frame(game)
        self.assertEqual(len(out), 2)
        kept = out[out["game_id"] == "G1"]
        self.assertEqual(len(kept), 1)
        self.assertEqual(str(kept["gameday"].iloc[0])[:10], "2019-09-15")
        pd.testing.assert_frame_equal(out, canonical_decided_frame(game))

    def test_order_stable_chronological_within_day_preserved(self):
        """Mergesort by gameday: chronological order, and within-day rows keep
        their input order (the same contract mlb-backend's canonical frame
        enforces — row order feeds order-sensitive training)."""
        game = pd.concat([
            _minimal_game(game_id="D", gameday="2019-09-15"),
            _minimal_game(game_id="B", gameday="2019-09-08"),
            _minimal_game(game_id="A", gameday="2019-09-01"),
            _minimal_game(game_id="C", gameday="2019-09-08"),
        ], ignore_index=True)
        out = canonical_decided_frame(game)
        self.assertEqual(out["game_id"].tolist(), ["A", "B", "C", "D"])

    def test_aggregate_joins_play_counts(self):
        """n_plays comes from pbp play_id counts per game_id; a schedule game
        with no pbp rows keeps n_plays NaN (never drops the game)."""
        schedule = pd.concat([
            _minimal_game().drop(columns="n_plays"),
            _minimal_game(game_id="2019_01_GB_CHI", away_team="GB", home_team="CHI",
                          away_score=10, home_score=3, result=-7.0, total=13.0,
                          spread_line=3.5, total_line=47.0).drop(columns="n_plays"),
        ], ignore_index=True)
        pbp = pd.DataFrame({
            "game_id": ["2019_01_KC_JAX"] * 3 + ["2019_01_KC_JAX", "2019_01_KC_JAX"],
            "play_id": [1, 2, 3, 4, 5],
        })
        game = aggregate_game_frame(schedule, pbp)
        counts = game.set_index("game_id")["n_plays"]
        self.assertEqual(counts["2019_01_KC_JAX"], 5)
        self.assertTrue(pd.isna(counts["2019_01_GB_CHI"]))

    def test_schema_columns_match_contract(self):
        """The aggregated frame exposes exactly the documented game-level
        columns, in order (the spike schema — consumers can rely on it)."""
        schedule = _minimal_game().drop(columns="n_plays")
        pbp = pd.DataFrame({"game_id": ["2019_01_KC_JAX"], "play_id": [1]})
        game = aggregate_game_frame(schedule, pbp)
        self.assertEqual(list(game.columns), GAME_LEVEL_COLUMNS)


@unittest.skipUnless(FEATURES.exists(),
                     "data_delivery/nfl_game_level_features.csv not present — "
                     "run `python3 nfl_game_frame.py` first")
class TestRealArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(FEATURES)

    def test_per_season_counts_match_spike(self):
        """The spike's go/no-go table, re-verified on the real artifact."""
        counts = self.df.groupby("season")["game_id"].count()
        for season, expected in SPIKE_PER_SEASON.items():
            self.assertEqual(int(counts.get(season, 0)), expected,
                             f"season {season} decided-game count")

    def test_no_duplicate_game_ids_no_missing_scores(self):
        self.assertEqual(int(self.df["game_id"].duplicated().sum()), 0)
        self.assertEqual(int(self.df[["away_score", "home_score"]].isna().any(axis=1).sum()), 0)

    def test_spot_check_known_game_score_and_spread_sign(self):
        """2019 W1 KC@JAX: 40-26 (ESPN-verified) with spread -3.5 — per the
        nflverse schedules dictionary a NEGATIVE spread means the AWAY team
        was favored (KC). GB@CHI +3.5 is the mirror case: POSITIVE = HOME
        team favored (CHI)."""
        g = self.df[self.df["game_id"] == "2019_01_KC_JAX"]
        self.assertEqual(len(g), 1)
        self.assertEqual(int(g["away_score"].iloc[0]), 40)
        self.assertEqual(int(g["home_score"].iloc[0]), 26)
        self.assertEqual(float(g["spread_line"].iloc[0]), -3.5)

        c = self.df[self.df["game_id"] == "2019_01_GB_CHI"]
        self.assertEqual(len(c), 1)
        self.assertEqual(int(c["away_score"].iloc[0]), 10)
        self.assertEqual(int(c["home_score"].iloc[0]), 3)
        self.assertEqual(float(c["spread_line"].iloc[0]), 3.5)

    def test_season_range_covers_defaults(self):
        self.assertEqual(sorted(self.df["season"].unique()), DEFAULT_SEASONS)


if __name__ == "__main__":
    unittest.main()
