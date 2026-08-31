"""
Regression tests for the NFL power-rankings artifact builder
(``nfl_moneyline._power_rankings_rows`` / ``_power_rankings_csv``).

Mirrors mlb-backend/backend/test_power_rankings.py: the shared page must
tolerate NULL outcomes — undecided games are skipped, never crashing int()
or the pct math, and a team with zero decided games still yields a valid
``0-0`` row. Also pins the MLB-identical column set, the 1-based rank index,
and the signed NFL point-differential (RUN DIFF) semantics.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_moneyline import (  # noqa: E402
    _power_rankings_csv,
    _power_rankings_rows,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Wrap game rows with the columns team_events/compute_elo require."""
    out = []
    for r in rows:
        g = r.copy()
        g.setdefault("season", 2021)
        g.setdefault("game_id", f"g{len(out) + 1}")
        out.append(g)
    return pd.DataFrame(out)


class TestNflPowerRankingsNullOutcome(unittest.TestCase):
    def test_undecided_games_are_skipped_not_crashed(self):
        """home_win = NULL rows are excluded from records + L10, never crash."""
        feats = _frame([
            {"week": 1, "gameday": "2021-09-12", "home_team": "BOS",
             "away_team": "NYY", "home_win": 1.0, "home_score": 28, "away_score": 7},
            {"week": 1, "gameday": "2021-09-19", "home_team": "BOS",
             "away_team": "SEA", "home_win": float("nan"),
             "home_score": float("nan"), "away_score": float("nan")},  # postponed
            {"week": 2, "gameday": "2021-09-26", "home_team": "SEA",
             "away_team": "BOS", "home_win": 0.0, "home_score": 10, "away_score": 21},
        ])
        df = _power_rankings_rows(feats)
        bos = df[df["team"] == "BOS"].iloc[0]
        # Decided only: W at home vs NYY + W away @ SEA -> 2-0; the NaN game
        # (postponed) is excluded entirely.
        self.assertEqual(int(bos["wins"]), 2)
        self.assertEqual(int(bos["losses"]), 0)
        self.assertEqual(bos["record"], "2-0")
        wins10, losses10 = bos["l10"].split("-")
        self.assertEqual(int(wins10) + int(losses10), 2)  # NaN game not counted
        # signed point differential: +21 (home) + (+11 away) = +32
        self.assertEqual(int(bos["run_diff"]), 32)

    def test_all_undecided_team_gets_neutral_record(self):
        """A team with zero decided games produces a valid 0-0 row (elo 1500)."""
        feats = _frame([
            {"week": 1, "gameday": "2021-09-12", "home_team": "LAD",
             "away_team": "SF", "home_win": float("nan"),
             "home_score": float("nan"), "away_score": float("nan")},
        ])
        df = _power_rankings_rows(feats)
        lad = df[df["team"] == "LAD"].iloc[0]
        self.assertEqual(lad["record"], "0-0")
        self.assertEqual(float(lad["elo"]), 1500.0)
        self.assertEqual(int(lad["wins"]), 0)
        self.assertEqual(int(lad["losses"]), 0)
        self.assertEqual(int(lad["run_diff"]), 0)

    def test_csv_shape_matches_mlb_columns_and_rank(self):
        """Written CSV carries the MLB-identical column set + 1-based rank."""
        with tempfile.TemporaryDirectory() as td:
            feats = _frame([
                {"week": 1, "gameday": "2021-09-12", "home_team": "BUF",
                 "away_team": "KC", "home_win": 1.0,
                 "home_score": 31, "away_score": 20},
                {"week": 1, "gameday": "2021-09-12", "home_team": "MIA",
                 "away_team": "NE", "home_win": 0.0,
                 "home_score": 17, "away_score": 24},
            ])
            path = _power_rankings_csv(feats, "20990101", Path(td))
            self.assertTrue(path.name.startswith("nfl_power_rankings_"))
            dfs = pd.read_csv(path, index_col=0)
            for col in ("team", "team_name", "elo", "wins", "losses", "record",
                        "pct", "run_diff", "l10", "home_pct", "away_pct"):
                self.assertIn(col, dfs.columns, f"missing column {col}")
            # rank index is 1-based descending Elo
            self.assertEqual(list(dfs.index), [1, 2])
            self.assertEqual(dfs.index.name, "rank")
            # BUF (home W, 31-20) sorts above MIA (home L)
            self.assertEqual(dfs.iloc[0]["team"], "BUF")
            self.assertEqual(int(dfs.iloc[1]["run_diff"]), -7)


if __name__ == "__main__":
    unittest.main()