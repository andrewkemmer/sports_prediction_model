"""Unit + artifact tests for the past-date final-score reconciliation in
frontend/utils.py (_reconcile_board_finals).

todays_games_<date>.csv is a point-in-time snapshot: a day's later games can
still be recorded as 'pre'/'in' at 0-0 after they are actually final in
game_level_features.csv. These tests pin that reconcile overlays authoritative
finals onto stale boards (fixing missing scores for e.g. 2026-08-29) without
fabricating a result for a genuinely still-scheduled future game.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

# Make frontend/ importable (utils imports streamlit, which is available in
# this test env; we only exercise pure helpers, never session_state flows).
_FE = Path(__file__).resolve().parents[2] / "frontend"
if str(_FE) not in sys.path:
    sys.path.insert(0, str(_FE))

import utils  # noqa: E402

_DD = Path(__file__).resolve().parents[1] / "data_delivery"


def _board(*rows):
    """Minimal board frame: game_id, game_date, teams, scores, state."""
    return pd.DataFrame([{"game_id": r.get("game_id", ""),
                          "game_date": r["game_date"],
                          "home_team": r["home_team"],
                          "away_team": r["away_team"],
                          "home_score": r.get("home_score", 0),
                          "away_score": r.get("away_score", 0),
                          "home_win": r.get("home_win"),
                          "game_state": r.get("game_state", "pre")}
                         for r in rows])


def _finals(*rows):
    return pd.DataFrame(
        [{"game_pk": r["game_pk"], "game_date": r["game_date"],
          "home_team": r["home_team"], "away_team": r["away_team"],
          "home_score": r["home_score"], "away_score": r["away_score"],
          "home_win": r["home_win"]} for r in rows])


class TestTeamNormalization(unittest.TestCase):
    def test_aliases_map_to_canonical(self):
        self.assertEqual(utils._normalize_team("CHW"), "CWS")
        self.assertEqual(utils._normalize_team("CWS"), "CWS")
        self.assertEqual(utils._normalize_team("AZ"), "ARI")
        self.assertEqual(utils._normalize_team("ARI"), "ARI")
        self.assertEqual(utils._normalize_team("ATH"), "OAK")
        self.assertEqual(utils._normalize_team("OAK"), "OAK")

    def test_unknown_and_empty(self):
        self.assertEqual(utils._normalize_team("NYY"), "NYY")
        self.assertEqual(utils._normalize_team(" nyy "), "NYY")
        self.assertEqual(utils._normalize_team(None), "")

    def test_date_compact(self):
        self.assertEqual(utils._game_date_compact("2026-08-29"), "20260829")
        self.assertEqual(utils._game_date_compact("2026-08-29 19:00:00"), "20260829")
        self.assertEqual(utils._game_date_compact(None), "")


class TestReconcileBoardFinals(unittest.TestCase):
    def test_single_stale_row_gets_final(self):
        board = _board({"game_id": "20260829_TEX@MIL", "game_date": "2026-08-29",
                        "home_team": "MIL", "away_team": "TEX",
                        "home_score": 0, "away_score": 0, "game_state": "pre"})
        finals = _finals({"game_pk": 823741, "game_date": "2026-08-29",
                          "home_team": "MIL", "away_team": "TEX",
                          "home_score": 5, "away_score": 3, "home_win": 1.0})
        out = utils._reconcile_board_finals(board.copy(), finals)
        self.assertEqual(out.at[0, "home_score"], 5)
        self.assertEqual(out.at[0, "away_score"], 3)
        self.assertEqual(out.at[0, "home_win"], 1.0)
        self.assertEqual(out.at[0, "game_state"], "post")

    def test_team_alias_still_matches(self):
        board = _board({"game_id": "20260829_CHW@MIN", "game_date": "2026-08-29",
                        "home_team": "MIN", "away_team": "CHW", "game_state": "pre"})
        finals = _finals({"game_pk": 123, "game_date": "2026-08-29",
                          "home_team": "MIN", "away_team": "CWS",
                          "home_score": 2, "away_score": 3, "home_win": 0.0})
        out = utils._reconcile_board_finals(board.copy(), finals)
        self.assertEqual(out.at[0, "game_state"], "post")
        self.assertEqual(out.at[0, "away_score"], 3)

    def test_future_still_scheduled_row_untouched(self):
        # No authority result for this game -> must stay pre/0-0 (not fabricated).
        board = _board({"game_id": "20261101_A@B", "game_date": "2026-11-01",
                        "home_team": "B", "away_team": "A",
                        "home_score": 0, "away_score": 0, "game_state": "pre"})
        finals = _finals({"game_pk": 500, "game_date": "2026-08-29",
                          "home_team": "X", "away_team": "Y",
                          "home_score": 1, "away_score": 2, "home_win": 0.0})
        out = utils._reconcile_board_finals(board.copy(), finals)
        self.assertEqual(out.at[0, "game_state"], "pre")
        self.assertEqual(out.at[0, "home_score"], 0)

    def test_doubleheader_both_legs_get_distinct_finals(self):
        # Unsuffixed board row already captured a real final (anchors the leg);
        # the _2_2 sibling gets the remaining leg.
        board = _board(
            {"game_id": "20260829_BOS@NYY", "game_date": "2026-08-29",
             "home_team": "NYY", "away_team": "BOS", "home_score": 0,
             "away_score": 6, "game_state": "post"},
            {"game_id": "20260829_BOS@NYY_2_2", "game_date": "2026-08-29",
             "home_team": "NYY", "away_team": "BOS", "home_score": 0,
             "away_score": 0, "game_state": "pre"},
        )
        finals = _finals(
            {"game_pk": 823501, "game_date": "2026-08-29", "home_team": "NYY",
             "away_team": "BOS", "home_score": 9, "away_score": 2, "home_win": 1.0},
            {"game_pk": 823539, "game_date": "2026-08-29", "home_team": "NYY",
             "away_team": "BOS", "home_score": 0, "away_score": 6, "home_win": 0.0},
        )
        out = utils._reconcile_board_finals(board.copy(), finals)
        by_id = dict(zip(out["game_id"], zip(out["home_score"], out["away_score"],
                                             out["game_state"])))
        self.assertEqual(by_id["20260829_BOS@NYY"], (0, 6, "post"))
        self.assertEqual(by_id["20260829_BOS@NYY_2_2"], (9, 2, "post"))

    def test_empty_finals_is_noop(self):
        board = _board({"game_id": "20260829_TEX@MIL", "game_date": "2026-08-29",
                        "home_team": "MIL", "away_team": "TEX", "game_state": "pre"})
        out = utils._reconcile_board_finals(board.copy(),
                                            pd.DataFrame(columns=["game_pk",
                                                                 "game_date",
                                                                 "home_team",
                                                                 "away_team",
                                                                 "home_score",
                                                                 "away_score",
                                                                 "home_win"]))
        self.assertEqual(out.at[0, "game_state"], "pre")

    def test_already_final_row_is_idempotent(self):
        board = _board({"game_id": "20260829_CIN@CHC", "game_date": "2026-08-29",
                        "home_team": "CHC", "away_team": "CIN", "home_score": 17,
                        "away_score": 5, "game_state": "post"})
        finals = _finals({"game_pk": 55, "game_date": "2026-08-29",
                          "home_team": "CHC", "away_team": "CIN",
                          "home_score": 17, "away_score": 5, "home_win": 1.0})
        out = utils._reconcile_board_finals(board.copy(), finals)
        self.assertEqual(out.at[0, "game_state"], "post")
        self.assertEqual(out.at[0, "home_score"], 17)


class TestWiring(unittest.TestCase):
    def test_normalize_games_calls_reconcile_first(self):
        """normalize_games must run reconcile before deriving win/status so the
        card grades refreshed finals correctly."""
        src = (_FE / "utils.py").read_text(encoding="utf-8")
        self.assertIn("_reconcile_board_finals(df)", src)
        # The reconcile call must sit between df.copy() and the win read.
        self.assertLess(src.index("df = _reconcile_board_finals(df)"),
                        src.index('# Outcome flag: 1.0 home won'))


@unittest.skipUnless((_DD / "todays_games_20260829.csv").exists()
                     and (_DD / "game_level_features.csv").exists(),
                     "Aug 29 + decided-frame artifacts not committed")
class TestAug29Artifact(unittest.TestCase):
    """Real committed artifacts: the stale Aug 29 board must reconcile so every
    game has a Final score (the user-visible missing-scores bug)."""

    def test_aug29_all_games_become_final(self):
        board = pd.read_csv(_DD / "todays_games_20260829.csv")
        finals = pd.read_csv(_DD / "game_level_features.csv",
                             usecols=lambda c: c in ("game_pk", "game_date",
                                                     "home_team", "away_team",
                                                     "home_score", "away_score",
                                                     "home_win"))
        out = utils._reconcile_board_finals(board.copy(), finals)
        self.assertEqual(len(out), 17)
        self.assertEqual(set(out["game_state"]), {"post"})
        self.assertEqual(int(out[["home_score", "away_score"]]
                             .isna().any(axis=1).sum()), 0)
        by_id = dict(zip(out["game_id"], zip(out["home_score"], out["away_score"])))
        # Previously-pre-game games now carry finals.
        self.assertEqual(by_id["20260829_TEX@MIL"], (5, 3))
        self.assertEqual(by_id["20260829_BAL@ATH"], (3, 5))
        self.assertEqual(by_id["20260829_PHI@LAA"], (2, 4))
        # Doubleheader legs resolved distinctly.
        self.assertEqual(by_id["20260829_BOS@NYY"], (0, 6))
        self.assertEqual(by_id["20260829_BOS@NYY_2_2"], (9, 2))


if __name__ == "__main__":
    unittest.main()