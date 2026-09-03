"""Tests for the NFL W2016 frame-expansion measurement (pure, offline).

Pins the Step-1.5 diagnostics: elo delta stats, the outcome regression,
the ewm/rolling/static sanity classification, the decision rule, and the
by-season away-bias table — all on synthetic frames. No network, no engine
imports beyond the module under test.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from nfl_frame_expansion import (
    MATERIAL_P95_DELTA, away_bias_by_season, decision_rule, elo_delta_stats,
    feature_sanity, ols, outcome_regressions,
)


def _frame(season_offset: int = 0) -> pd.DataFrame:
    """Synthetic game frame (2019-frame shaped)."""
    n = 60
    rng = np.random.default_rng(7)
    teams = [f"T{i % 6}" for i in range(n)]
    return pd.DataFrame({
        "game_id": [f"{2021 + season_offset}_{i:03d}" for i in range(n)],
        "season": 2021 + season_offset,
        "home_team": teams,
        "away_team": [f"A{i % 6}" for i in range(n)],
        "home_score": rng.integers(10, 40, n),
        "away_score": rng.integers(3, 35, n),
        "home_win": rng.integers(0, 2, n).astype(float),
        "elo_diff": rng.normal(0, 60, n),
        "home_elo": rng.normal(1500, 60, n),
        "away_elo": rng.normal(1500, 60, n),
        "ewm_net_pts_diff": rng.normal(0, 4, n),
        "ewm_ypp_diff": rng.normal(0, 0.8, n),
        "win_pct_diff": rng.normal(0, 0.25, n),
        "pace_plays_min_diff": rng.normal(0, 0.4, n),
        "rest_days_diff": rng.normal(0, 2, n),
        "is_dome_home": rng.integers(0, 2, n).astype(float),
        "div_game": rng.integers(0, 2, n).astype(float),
        "travel_miles_diff": rng.normal(0, 500, n),
        "altitude_home": rng.normal(0, 300, n),
        "prime_time": rng.integers(0, 2, n).astype(float),
    })


class TestOls(unittest.TestCase):
    def test_recovers_slope_with_ci(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 50, 2000)
        y = 0.01 * x + rng.normal(0, 1, 2000)      # beta = 0.01
        r = ols(x, y)
        self.assertAlmostEqual(r["beta"], 0.01, delta=0.004)
        self.assertTrue(r["ci_lo"] < 0.01 < r["ci_hi"])
        self.assertGreater(r["n"], 0)

    def test_zero_effect(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 50, 1500)
        y = rng.normal(0, 1, 1500)
        r = ols(x, y)
        self.assertLess(abs(r["beta"]), 0.005)
        self.assertAlmostEqual(r["r2"], 0.0, delta=0.01)

    def test_nan_pairs_dropped(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        r = ols(x, y)
        self.assertEqual(r["n"], 3)

    def test_constant_x_returns_none(self):
        r = ols(np.ones(10), np.arange(10.0))
        self.assertIsNone(r["beta"])


class TestEloDeltaStats(unittest.TestCase):
    def test_identical_frames_zero_delta(self):
        f16 = _frame()
        f19 = _frame()
        stats = elo_delta_stats(f16, f19, f16["game_id"].to_numpy())
        self.assertTrue(stats)
        for rec in stats:
            for col in ("elo_diff", "home_elo", "away_elo"):
                self.assertEqual(rec[col]["max_abs_delta"], 0.0)

    def test_shifted_frames_material_delta(self):
        f16 = _frame()
        f19 = _frame()
        f19["elo_diff"] = f19["elo_diff"] - 40.0     # uniform -40 shift
        stats = elo_delta_stats(f16, f19, f16["game_id"].to_numpy())
        rec = stats[0]
        self.assertGreater(rec["elo_diff"]["mean_abs_delta"], 35.0)
        self.assertGreater(rec["elo_diff"]["p95_abs_delta"], 35.0)

    def test_shared_ids_respected(self):
        f16 = _frame()
        f19 = _frame()
        stats = elo_delta_stats(f16, f19, f16["game_id"].to_numpy()[:10])
        self.assertEqual(stats[0]["n"], 10)


class TestOutcomeRegressions(unittest.TestCase):
    def test_signal_detected(self):
        rng = np.random.default_rng(11)
        n = 1500
        delta = rng.normal(0, 30, n)
        p = 0.5 + 0.004 * delta                    # positive slope
        y = (rng.uniform(0, 1, n) < p).astype(float)
        margin = 2.0 * delta + rng.normal(0, 12, n)
        shared = pd.DataFrame({
            "elo_diff_2016": delta + 10, "elo_diff_2019": 10.0,
            "home_win": y, "home_score": margin + 21, "away_score": 21.0,
        })
        r = outcome_regressions(shared)
        self.assertGreater(r["home_win"]["t"], 2.0)
        self.assertGreater(r["home_margin"]["beta"], 1.0)

    def test_noise_only(self):
        rng = np.random.default_rng(12)
        n = 1500
        delta = rng.normal(0, 30, n)
        y = rng.integers(0, 2, n).astype(float)
        margin = rng.normal(0, 12, n)
        shared = pd.DataFrame({
            "elo_diff_2016": delta + 10, "elo_diff_2019": 10.0,
            "home_win": y, "home_score": margin + 21, "away_score": 21.0,
        })
        r = outcome_regressions(shared)
        self.assertLess(abs(r["home_win"]["t"]), 2.0)
        self.assertLess(abs(r["home_margin"]["t"]), 2.0)


class TestFeatureSanity(unittest.TestCase):
    def _shared_pair(self):
        a = _frame()
        b = _frame()
        b["ewm_net_pts_diff"] = a["ewm_net_pts_diff"] + 1e-8   # tiny
        b["ewm_ypp_diff"] = a["ewm_ypp_diff"] + 5e-4
        b["win_pct_diff"] = a["win_pct_diff"] + 0.01            # rolling shifts
        shared = pd.DataFrame()
        for c in a.columns:
            shared[c + "_2016"] = a[c].to_numpy()
            shared[c + "_2019"] = b[c].to_numpy()
        return shared

    def test_ewm_ok_rolling_not_zero(self):
        shared = self._shared_pair()
        sanity = feature_sanity(
            shared, ["ewm_net_pts_diff", "ewm_ypp_diff"],
            ["win_pct_diff", "pace_plays_min_diff", "rest_days_diff"],
            ["is_dome_home", "div_game", "travel_miles_diff",
             "altitude_home", "prime_time"])
        self.assertTrue(sanity["ewm_ok"])
        self.assertLessEqual(sanity["ewm"]["max_abs_delta"], 1e-3)
        self.assertGreater(sanity["rolling"]["max_abs_delta"], 0.0)
        self.assertEqual(sanity["static"]["max_abs_delta"], 0.0)
        self.assertFalse(sanity["rolling_static_exact_zero"])

    def test_exact_zero_classes(self):
        a = _frame()
        shared = pd.DataFrame()
        for c in a.columns:
            shared[c + "_2016"] = a[c].to_numpy()
            shared[c + "_2019"] = a[c].to_numpy()
        sanity = feature_sanity(
            shared, ["ewm_net_pts_diff", "ewm_ypp_diff"],
            ["win_pct_diff", "pace_plays_min_diff", "rest_days_diff"],
            ["is_dome_home", "div_game", "travel_miles_diff",
             "altitude_home", "prime_time"])
        self.assertTrue(sanity["rolling_static_exact_zero"])
        self.assertTrue(sanity["ewm_ok"])


class TestDecisionRule(unittest.TestCase):
    def test_branches(self):
        self.assertEqual(decision_rule(False, False)["verdict"], "negligible")
        self.assertEqual(decision_rule(True, True)["verdict"],
                         "material_plus_signal")
        self.assertEqual(decision_rule(True, False)["verdict"],
                         "material_plus_noise")
        self.assertIn("reason", decision_rule(True, False))


class TestAwayBiasBySeason(unittest.TestCase):
    def test_season_rows_and_magnitude(self):
        rng = np.random.default_rng(5)
        n = 400
        art = pd.DataFrame({
            "game_id": [f"{s}_{i}" for s in (2021, 2022, 2023, 2024)
                        for i in range(n // 4)],
            "season": [s for s in (2021, 2022, 2023, 2024)
                       for _ in range(n // 4)],
            "resid_away": rng.normal(-2.0, 1.0, n),
            "resid_home": rng.normal(0, 1.0, n),
        })
        sealed = pd.DataFrame({
            "game_id": [f"2025_{i}" for i in range(50)],
            "season": [2025] * 50,
            "pred_away": rng.normal(21, 3, 50),
            "pred_home": rng.normal(24, 3, 50),
            "away_score": rng.normal(21, 3, 50),
            "home_score": rng.normal(24, 3, 50),
        })
        rows = away_bias_by_season(art, sealed)
        self.assertEqual([r["season"] for r in rows],
                         [2021, 2022, 2023, 2024, 2025])
        self.assertLess(rows[0]["away_bias"], -1.0)    # negative bias visible
        self.assertEqual(rows[-1]["n"], 50)


if __name__ == "__main__":
    unittest.main()