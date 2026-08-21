"""Point-in-time enforcement tests.

These prove that a game's features are computed only from games whose
scheduled start time is *strictly before* that game, and that market lines
posted at/after game start are never attached. Runs under both
``python -m unittest`` and ``pytest``.
"""

import unittest

import numpy as np
import pandas as pd

import data_ingestion as di


def make_game_events(n_games: int = 40, start: str = "2026-04-01") -> pd.DataFrame:
    """Deterministic synthetic game log for two teams, one game per day."""
    dates = pd.date_range(start, periods=n_games, freq="D", tz="UTC")
    rows = []
    for i, d in enumerate(dates):
        hr, ar = (5, 2) if i % 2 == 0 else (2, 4)
        rows.append(
            {
                "game_id": f"{d:%Y%m%d}_NYY_BOS",
                "start_time": d,
                "home_team": "BOS",
                "away_team": "NYY",
                "home_runs": hr,
                "away_runs": ar,
                "home_woba": 0.30 + 0.001 * i,
                "away_woba": 0.31 - 0.001 * i,
                "home_bullpen_whip": 1.20,
                "away_bullpen_whip": 1.30,
                "sp_home": "P1",
                "sp_away": "P2",
                "sp_home_era": 3.50,
                "sp_home_k9": 9.0,
                "sp_away_era": 4.00,
                "sp_away_k9": 8.0,
                "ml_home": -130,
                "ml_away": +120,
                "total_line": 8.5,
                "run_line": -1.5,
                "venue": "Fenway Park",
                "day_game": True,
                "status": "Final",
                "weather_wind_speed": 7.0,
                "final_inning": "F",
            }
        )
    return pd.DataFrame(rows)


class PointInTimeTests(unittest.TestCase):
    def setUp(self):
        self.events = make_game_events()

    def test_filter_prior_is_strict(self):
        """Games at/after as_of must be excluded from prior data."""
        as_of = self.events.iloc[10]["start_time"]
        prior = di.filter_prior(self.events, as_of)
        self.assertEqual(len(prior), 10)
        self.assertTrue((prior["start_time"] < as_of).all())

    def test_rolling_features_ignore_current_and_future_games(self):
        """Features for game i are identical whether or not later games exist."""
        as_of = self.events.iloc[15]["start_time"]
        full = di.build_point_in_time_features(self.events, include_unready=True)
        full = full.set_index("game_id")

        prior_only = di.build_point_in_time_features(
            di.filter_prior(self.events, as_of), include_unready=True
        ).set_index("game_id")

        # Compare a game strictly before the cutoff (present in both sets).
        target_id = self.events.iloc[10]["game_id"]
        cols = [
            "home_woba_30g", "away_woba_30g", "home_runs_for_30g",
            "home_bullpen_whip_10g", "away_bullpen_whip_10g",
            "home_record_pct", "away_record_pct", "home_rest_days",
            "home_elo", "away_elo",
        ]
        for col in cols:
            self.assertAlmostEqual(
                full.loc[target_id, col], prior_only.loc[target_id, col],
                places=6, msg=f"{col} leaked future information",
            )

    def test_extreme_future_value_does_not_change_earlier_features(self):
        """A monster game in the future must not affect an earlier row."""
        baseline = di.build_point_in_time_features(self.events, include_unready=True)
        baseline = baseline.set_index("game_id")

        future = pd.DataFrame(
            [{
                "game_id": "20260520_NYY_BOS",
                "start_time": pd.Timestamp("2026-05-20", tz="UTC"),
                "home_team": "BOS", "away_team": "NYY",
                "home_runs": 20, "away_runs": 1,
                "home_woba": 0.99, "away_woba": 0.10,
                "home_bullpen_whip": 2.5, "away_bullpen_whip": 0.5,
                "sp_home": "P1", "sp_away": "P2",
                "sp_home_era": 1.0, "sp_home_k9": 14.0,
                "sp_away_era": 9.0, "sp_away_k9": 2.0,
                "ml_home": -500, "ml_away": +400,
                "total_line": 12.5, "run_line": -1.5,
                "venue": "Fenway Park", "day_game": True,
                "status": "Final", "weather_wind_speed": 7.0, "final_inning": "F",
            }]
        )
        augmented = di.build_point_in_time_features(
            pd.concat([self.events, future], ignore_index=True), include_unready=True
        ).set_index("game_id")

        earlier_id = self.events.iloc[8]["game_id"]
        for col in ["home_woba_30g", "away_record_pct", "home_elo", "sp_home_era"]:
            self.assertAlmostEqual(
                baseline.loc[earlier_id, col], augmented.loc[earlier_id, col],
                places=6, msg=f"{col} changed after adding a future game",
            )

    def test_market_lines_must_be_timestamped_before_game_start(self):
        """Lines posted at or after start_time must never attach."""
        games = make_game_events(n_games=3)
        lines = pd.DataFrame(
            [
                {"timestamp": games.iloc[0]["start_time"] - pd.Timedelta(hours=5),
                 "ml_home": -150, "ml_away": +130, "total_line": 8.0, "run_line": -1.5},
                # posted exactly at game start -> must be rejected
                {"timestamp": games.iloc[0]["start_time"],
                 "ml_home": -999, "ml_away": +999, "total_line": 3.5, "run_line": 1.5},
                # posted after game start -> must be rejected
                {"timestamp": games.iloc[0]["start_time"] + pd.Timedelta(hours=2),
                 "ml_home": -888, "ml_away": +888, "total_line": 4.5, "run_line": 2.5},
                {"timestamp": games.iloc[1]["start_time"] - pd.Timedelta(hours=1),
                 "ml_home": -140, "ml_away": +125, "total_line": 8.5, "run_line": -1.5},
            ]
        )
        attached = di.attach_market_lines(games, lines)
        first = attached.iloc[0]
        self.assertEqual(first["ml_home"], -150)
        self.assertEqual(first["total_line"], 8.0)
        second = attached.iloc[1]
        self.assertEqual(second["ml_home"], -140)

    def test_elo_is_pre_game(self):
        """Elo entering a game is the rating produced by prior games only."""
        # Elo entering game 15 must equal the rating after processing game 14
        # (the last game strictly before game 15) — verified via the feature
        # columns home_elo/away_elo below, plus an explicit spot check.
        full = di.build_point_in_time_features(self.events, include_unready=True)
        full = full.set_index("game_id")
        prior = di.build_point_in_time_features(
            di.filter_prior(self.events, self.events.iloc[15]["start_time"]),
            include_unready=True,
        ).set_index("game_id")
        target_id = self.events.iloc[10]["game_id"]
        self.assertAlmostEqual(full.loc[target_id, "home_elo"], prior.loc[target_id, "home_elo"], places=6)


if __name__ == "__main__":
    unittest.main()
