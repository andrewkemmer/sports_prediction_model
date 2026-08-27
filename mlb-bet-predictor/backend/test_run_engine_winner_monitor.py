"""Tests for the v2 run-engine WINNER cards.

Covers: per-game line assignment, whole-number-line push exclusion, the
>50% pick rule, fixed-reference AUC on the real artifact, v2 rolling
migration (renamed field), and the CROSS-CHECK that the winner win rates
match the Totals & Run Lines history tables (~54% totals / ~64% run line).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_frontend = _ROOT / "frontend"
if str(_frontend) not in sys.path:
    sys.path.insert(0, str(_frontend))


def _mk_frame() -> pd.DataFrame:
    """Synthetic OOF markets frame with known lines/picks.

    Games 1-3 price at line 8.5 (λ sum 8.3); game 4 is a whole-number-line
    PUSH (line 8.0, total 8); game 5 prices at 9.5 (λ sum 9.3).
    """
    rows = [
        # game_pk, date, λh, λa, total, hs, as_, p_over_8_5, p_over_8_0,
        # p_over_9_5, p_home_cover_1_5, p_home_win_derived
        (1, "2026-08-20", 4.2, 4.1, 10, 6, 4, 0.60, 0.70, 0.60, 0.6, 0.6),
        (2, "2026-08-20", 4.2, 4.1, 8, 4, 4, 0.55, 0.65, 0.55, 0.6, 0.6),
        (3, "2026-08-20", 4.2, 4.1, 7, 3, 4, 0.45, 0.55, 0.45, 0.4, 0.4),
        (4, "2026-08-20", 4.1, 3.9, 8, 5, 3, 0.60, 0.70, 0.60, 0.7, 0.7),
        (5, "2026-08-20", 4.6, 4.7, 9, 4, 5, 0.40, 0.50, 0.40, 0.4, 0.4),
    ]
    df = pd.DataFrame(rows, columns=[
        "game_pk", "game_date", "home_expected_runs", "away_expected_runs",
        "total_runs", "home_score", "away_score", "p_over_8_5", "p_over_8_0",
        "p_over_9_5", "p_home_cover_1_5", "p_home_win_derived"])
    df["kind"] = "oof"
    return df


class TestWinnerCardAggregation(unittest.TestCase):
    def test_over_under_line_assignment_and_pick_rule(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        ou = cards["over_under"]
        # Game 4 pushes (line 8.0, total 8) -> excluded from n and win_rate.
        # Non-push games: 1 (over, 10>8.5, win), 2 (over, 8<8.5, loss),
        # 3 (under, 7<8.5, win), 5 (under, 9<9.5, win) -> 3/4.
        self.assertEqual(ou["n"], 4)
        self.assertAlmostEqual(ou["win_rate"], 0.75)
        self.assertAlmostEqual(ou["actual_win_rate"], 0.75)
        # Holdout window covers all synthetic dates -> same rate.
        self.assertEqual(ou["holdout"]["n"], 4)
        self.assertAlmostEqual(ou["holdout"]["win_rate"], 0.75)

    def test_over_under_pushes_only_on_whole_number_lines(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        ou = cards["over_under"]
        # Only game 4 (whole line 8.0 == total 8) is a push; the 9.5 line
        # game with total 9 is NOT a push (integer total != X.5 line).
        self.assertEqual(ou["n"], 4)          # 5 games - 1 push
        self.assertNotIn("n_pushes", ou)      # excluded, not counted as wins

    def test_run_line_half_run_never_pushes(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        rl = cards["run_line"]
        # pick home -1.5 when p>=0.5: games 1 (margin 2 -> win), 2 (margin 0
        # -> away +1.5 wins -> loss), 4 (margin 2 -> win); pick away +1.5:
        # game 3 (margin -1 -> win), 5 (margin -1 -> win) -> 4/5.
        self.assertEqual(rl["n"], 5)
        self.assertAlmostEqual(rl["win_rate"], 0.8)

    def test_derived_ml_pick_home_rule(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        ml = cards["derived_ml"]
        # pick home when p>=0.5: games 1 (win), 2 (loss), 4 (win); pick away:
        # games 3 (hs<as -> win), 5 (hs<as -> win) -> 4/5.
        self.assertEqual(ml["n"], 5)
        self.assertAlmostEqual(ml["win_rate"], 0.8)

    def test_auc_attached_from_reference_line(self):
        from run_engine import compute_winner_cards
        metrics = {
            "over_8_5": {"auc": 0.55051},
            "home_cover_1_5": {"auc": 0.54320},
            "derived_moneyline": {"auc": 0.55447},
        }
        cards = compute_winner_cards(_mk_frame(), market_metrics=metrics)
        self.assertAlmostEqual(cards["over_under"]["auc"], 0.55051)
        self.assertAlmostEqual(cards["run_line"]["auc"], 0.54320)
        self.assertAlmostEqual(cards["derived_ml"]["auc"], 0.55447)


class TestScoreAtAuc(unittest.TestCase):
    def test_score_at_auc_finite_on_real_oof_artifact(self):
        """AUC present + finite on the REAL run-engine OOF (fixed reference
        lines over_8_5 / home_cover_1_5 / derived moneyline)."""
        import run_engine
        oof = pd.read_csv(_ROOT / "data_delivery"
                          / "run_engine_oof_20260827.csv")
        # The shipped artifact strips fold_idx on export; a dummy fold keeps
        # the prequential path honest-free (identity) while score_at's AUC
        # uses the same real y/p vectors the pipeline scores.
        oof["fold_idx"] = 0
        res = run_engine.derive_markets_v3(oof, n_draws=20)
        s = res["summary"]
        for key in ("over_8_5", "home_cover_1_5", "derived_moneyline"):
            m = s[f"market_{key}"]
            self.assertIsNotNone(m.get("auc"))
            self.assertTrue(np.isfinite(m["auc"]), f"{key} auc not finite")
            self.assertGreater(m["auc"], 0.5)
            self.assertLess(m["auc"], 1.0)

    def test_score_at_auc_single_class_is_none_not_crash(self):
        import run_engine
        rng = np.random.default_rng(3)
        n = 60
        dates = pd.date_range("2026-07-20", periods=n, freq="D")
        hs = rng.integers(1, 5, n).astype(float)
        as_ = hs + rng.integers(1, 6, n)     # away ALWAYS wins -> single class
        oof = pd.DataFrame({
            "game_pk": list(range(1000, 1000 + n)),
            "game_date": dates,
            "home_expected_runs": rng.uniform(3.8, 5.2, n),
            "away_expected_runs": rng.uniform(3.8, 5.2, n),
            "home_score": hs,
            "away_score": as_,
            "fold_idx": list(range(n)),
        })
        res = run_engine.derive_markets_v3(oof, n_draws=10)
        m = res["summary"]["market_derived_moneyline"]
        self.assertIsNone(m.get("auc"))


class TestCrossCheckHistoryTables(unittest.TestCase):
    def test_winner_win_rates_match_history_tables_on_real_csv(self):
        """The winner cards must reproduce the Totals & Run Lines tables
        exactly (same frame, same pick/push logic) — ~54% totals, ~64% run
        line on the real 08-27 artifact."""
        from market_diagnostics import (decided_rows, history_win_rate,
                                        runline_history_frame,
                                        totals_history_frame)
        from run_engine import compute_winner_cards

        markets = pd.read_csv(_ROOT / "data_delivery"
                              / "run_engine_markets_20260827.csv")
        cards = compute_winner_cards(markets)
        decided = decided_rows(markets)

        tl = history_win_rate(totals_history_frame(decided))
        rl = history_win_rate(runline_history_frame(decided))

        # Exact agreement with the history tables (same n, same win rate).
        self.assertEqual(cards["over_under"]["n"], tl["n_games"])
        self.assertAlmostEqual(cards["over_under"]["win_rate"],
                               tl["win_rate"], places=4)
        self.assertEqual(cards["run_line"]["n"], rl["n_games"])
        self.assertAlmostEqual(cards["run_line"]["win_rate"],
                               rl["win_rate"], places=4)

        # Acceptance ranges: totals ~54%, run-line ~64%.
        self.assertGreater(cards["over_under"]["win_rate"], 0.52)
        self.assertLess(cards["over_under"]["win_rate"], 0.56)
        self.assertGreater(cards["run_line"]["win_rate"], 0.62)
        self.assertLess(cards["run_line"]["win_rate"], 0.66)


class TestRollingV2Migration(unittest.TestCase):
    def test_builder_folds_v2_prior_and_renames_field(self):
        """The v2 rolling fold reader uses winner_cards with the renamed
        actual_win_rate field; prior v1 files map onto the cards (over_8_5
        -> over_under) so the series stays continuous across the cutover."""
        from pipeline import _run_engine_monitor_json
        block = {
            "winner_cards": {
                "over_under": {"n": 4126, "actual_win_rate": 0.5414,
                               "win_rate": 0.5414, "predicted_mean": 0.5310,
                               "auc": 0.5505, "ece_raw": 0.019,
                               "ece_calibrated": 0.0105, "brier": 0.247,
                               "logloss": 0.680, "holdout": {}},
            },
            "market_metrics": {}, "alpha_home": {}, "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {}, "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        prior = {
            "schema": "run-engine-monitor/v2",
            "date": "20260825",
            "winner_cards": {
                "over_under": {"n": 3900, "actual_win_rate": 0.5390,
                               "win_rate": 0.5390, "predicted_mean": 0.5300,
                               "auc": 0.55, "ece_calibrated": 0.015,
                               "brier": 0.250, "logloss": 0.685},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260825.json").write_text(
                json.dumps(prior))
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826", True, None)
            data = json.loads(path.read_text())
        rolling = data["rolling"]["over_under"]
        self.assertEqual([r["date"] for r in rolling],
                         ["2026-08-25", "2026-08-26"])
        self.assertIn("actual_win_rate",
                      data["winner_cards"]["over_under"])
        self.assertNotIn("base_rate",
                         data["winner_cards"]["over_under"])

    def test_v1_prior_mapped_to_v2_card(self):
        """A v1 per_line file folds in through the line->card map:
        over_8_5 becomes the over_under rolling point (v1->v2 continuity)."""
        from pipeline import _run_engine_monitor_json
        block = {
            "winner_cards": {
                "over_under": {"n": 4126, "actual_win_rate": 0.5414,
                               "win_rate": 0.5414, "predicted_mean": 0.5310,
                               "auc": 0.5505, "ece_raw": 0.019,
                               "ece_calibrated": 0.0105, "brier": 0.247,
                               "logloss": 0.680, "holdout": {}},
            },
            "market_metrics": {}, "alpha_home": {}, "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {}, "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        prior = {
            "schema": "run-engine-monitor/v1",
            "date": "20260825",
            "per_line": {"over_8_5": {"n": 3900, "ece_calibrated": 0.015,
                                      "brier": 0.250, "logloss": 0.685,
                                      "predicted_mean": 0.448,
                                      "base_rate": 0.449}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260825.json").write_text(
                json.dumps(prior))
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826", True, None)
            data = json.loads(path.read_text())
        rolling = data["rolling"]["over_under"]
        self.assertEqual([[r["date"], r["n"]] for r in rolling],
                         [["2026-08-25", 3900], ["2026-08-26", 4126]])
        self.assertEqual(rolling[0]["ece_calibrated"], 0.015)
        self.assertEqual(rolling[0]["predicted_mean"], 0.448)


if __name__ == "__main__":
    unittest.main()
