"""
Regression tests for season-boundary handling of cumulative state.

Before these fixes, records/run-diff/Elo carried straight across the
2025→2026 offseason, producing dashboard artifacts like "PHI 167-127"
and structurally inflated PSI drift on every cumulative feature.
Locks in:
  - W/L and run-diff reset to zero at each season boundary
  - Elo regresses toward 1500 by ELO_REVERT_FACTOR at the boundary
  - _final_team_records (pre-game slate carry-forward) counts only the
    current season's decided games
  - load_game_features emits clean season-opening rows end-to-end
"""
import unittest

import numpy as np
import pandas as pd

from backend import data_ingestion as di
from backend.config import ELO_K, ELO_REVERT_FACTOR


def _game(date_str, home, away, home_win, hs=5, asc=3):
    return {
        "game_date": date_str,
        "home_team": home,
        "away_team": away,
        "home_win": home_win,
        "home_score": hs,
        "away_score": asc,
    }


class TestEloSeasonRevert(unittest.TestCase):
    def test_elo_regresses_toward_mean_at_offseason(self):
        """A dominant 2025 team must enter 2026 closer to 1500 than its final Elo."""
        games = pd.DataFrame([
            # 2025: BOS wins 4 straight as home favorite
            _game("2025-08-01", "BOS", "NYY", 1),
            _game("2025-08-02", "BOS", "NYY", 1),
            _game("2025-08-03", "BOS", "NYY", 1),
            _game("2025-08-04", "BOS", "NYY", 1),
            # 2026 opener
            _game("2026-04-01", "BOS", "NYY", 0),
        ])
        entry = di.compute_elos(games)

        # Elo AFTER the last decided 2025 game (compute_elos_up_to applies updates)
        from datetime import datetime
        games_for_upto = games.iloc[:4].copy()
        games_for_upto["start_time_utc"] = pd.to_datetime(games_for_upto["game_date"])
        final_2025 = di.compute_elos_up_to(
            games_for_upto, as_of=datetime(2025, 12, 31)
        )["BOS"]
        self.assertGreater(final_2025, 1500.0)

        post = entry.iloc[4]  # entering first 2026 game, before its update
        expected = final_2025 + ELO_REVERT_FACTOR * (1500.0 - final_2025)
        self.assertAlmostEqual(post, expected, places=6)
        self.assertLess(post, final_2025)

    def test_elo_within_season_does_not_revert(self):
        """No spurious revert between consecutive same-season games."""
        games = pd.DataFrame([
            _game("2026-05-01", "BOS", "NYY", 1),
            _game("2026-05-02", "NYY", "BOS", 0),  # BOS wins again (away)
        ])
        entry = di.compute_elos(games)
        # Entering game 2, the HOME team is NYY — its Elo equals its
        # post-game-1 value (it lost game 1 as the away side).
        exp_g1 = 1.0 / (1.0 + 10 ** ((1500.0 - 1500.0 - di.ELO_HOME_ADV) / 400))
        nyy_after_g1 = 1500.0 + ELO_K * ((1 - 1) - (1 - exp_g1))
        self.assertAlmostEqual(entry.iloc[1], nyy_after_g1, places=6)


class TestRecordsSeasonReset(unittest.TestCase):
    def test_final_team_records_counts_only_current_season(self):
        """Slate carry-forward must show current-season records, not career sums."""
        hist = pd.DataFrame([
            # 2025: BOS goes 2-0
            _game("2025-09-28", "BOS", "NYY", 1, hs=7, asc=2),
            _game("2025-09-29", "BOS", "SEA", 1, hs=6, asc=1),
            # 2026: BOS goes 1-1
            _game("2026-08-20", "BOS", "SEA", 1, hs=4, asc=3),
            _game("2026-08-21", "NYY", "BOS", 0, hs=2, asc=9),
        ])
        rec = di._final_team_records(hist)
        # 2026 only: BOS won 4-3 vs SEA and 9-2 @NYY
        self.assertEqual(rec["BOS"], {"w": 2, "l": 0, "rs": 13, "ra": 5})
        self.assertEqual(rec["SEA"], {"w": 0, "l": 1, "rs": 3, "ra": 4})
        self.assertEqual(rec["NYY"], {"w": 0, "l": 1, "rs": 2, "ra": 9})

    def test_load_game_features_resets_records_and_run_diff(self):
        """First 2026 game must see 0-0 records and zero run diff despite 2025 wins."""
        df_in = pd.DataFrame([
            {**_game("2025-09-28", "BOS", "NYY", 1, hs=8, asc=2), "game_pk": 1},
            {**_game("2025-09-29", "BOS", "NYY", 1, hs=7, asc=1), "game_pk": 2},
            {**_game("2026-04-01", "BOS", "NYY", float("nan"), hs=0, asc=0), "game_pk": 3},
        ])
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "features.csv"
        df_in.to_csv(tmp, index=False)

        out = di.load_game_features(tmp)
        row26 = out[out["game_pk"] == 3].iloc[0]
        self.assertEqual(int(row26["home_wins"]), 0)
        self.assertEqual(int(row26["home_losses"]), 0)
        self.assertEqual(row26["home_record"], "0-0")
        self.assertEqual(row26["home_run_diff"], 0)
        self.assertEqual(row26["away_run_diff"], 0)

        # And the last 2025 game still saw the cumulative 2025 state
        row25 = out[out["game_pk"] == 2].iloc[0]
        self.assertEqual(row25["home_record"], "1-0")
        self.assertEqual(row25["home_run_diff"], 6)

        # Season-opener win% is undefined → NULL, never a fake 0.0
        self.assertTrue(np.isnan(row26["home_win_pct"]))
        self.assertTrue(np.isnan(row26["away_win_pct"]))


if __name__ == "__main__":
    unittest.main()
