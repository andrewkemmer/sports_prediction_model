"""Tests for the SHIPPED Experiment #2 implementation (2026-09-07 decision).

Covers:
  * features.add_exp2_features produces BYTE-IDENTICAL values to the frozen
    experiment runner's construction (run_exp2_feature_test.add_candidates);
  * NaN propagation: missing source columns → all-NaN candidates, never 0;
  * the run-line target helper (true −1.5: home_runs − away_runs ≥ 2) keeps
    NULL alignment so fold geometry stays byte-identical;
  * FEATURE_COLS width invariant (61 = 59 baseline − 6 E/F removals + 8 candidates).
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import features as features_mod
from features import EXP2_CANDIDATE_COLS, add_exp2_features
from training import FEATURE_COLS, with_run_line_target


def _synthetic_frame(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "game_pk": np.arange(n),
        "sp_k9_home": rng.uniform(6, 12, n),
        "sp_k9_away": rng.uniform(6, 12, n),
        "team_k_rate_30g_home": rng.uniform(0.15, 0.28, n),
        "team_k_rate_30g_away": rng.uniform(0.15, 0.28, n),
        "league_k_pct": rng.uniform(0.21, 0.23, n),
        "opp_lefty_share_home": rng.uniform(0.2, 0.6, n),
        "opp_lefty_share_away": rng.uniform(0.2, 0.6, n),
    })
    for cat in ("fastball", "breaking", "offspeed"):
        for side in ("home", "away"):
            df[f"sp_usage_cat_{cat}_{side}"] = rng.uniform(0.2, 0.6, n)
            df[f"sp_k_pct_cat_{cat}_{side}"] = rng.uniform(0.1, 0.4, n)
            df[f"team_k_pct_cat_{cat}_{side}"] = rng.uniform(0.1, 0.4, n)
            df[f"sp_xwoba_cat_{cat}_{side}"] = rng.uniform(0.25, 0.4, n)
            df[f"team_xwoba_cat_{cat}_{side}"] = rng.uniform(0.25, 0.4, n)
        df[f"league_k_pct_cat_{cat}"] = rng.uniform(0.15, 0.3, n)
        df[f"league_xwoba_cat_{cat}"] = rng.uniform(0.27, 0.35, n)
    for hand in ("l", "r"):
        for side in ("home", "away"):
            df[f"sp_k_pct_fb_vs_{hand}_{side}"] = rng.uniform(0.1, 0.35, n)
            df[f"team_k_pct_fb_vs_{hand}_{side}"] = rng.uniform(0.1, 0.35, n)
    df["league_k_pct_fb_vs_l"] = rng.uniform(0.15, 0.25, n)
    df["league_k_pct_fb_vs_r"] = rng.uniform(0.15, 0.25, n)
    return df


class TestCandidateParity(unittest.TestCase):
    def test_production_construction_matches_frozen_experiment(self):
        """The shipped pipeline construction must be byte-identical to the
        frozen experiment formulas (same inputs → same columns)."""
        from run_exp2_feature_test import add_candidates  # frozen reference

        df = _synthetic_frame()
        got = add_exp2_features(df)
        want, _cov, missing = add_candidates(df)
        self.assertEqual(missing, [])
        for c in EXP2_CANDIDATE_COLS:
            a, b = got[c].to_numpy(float), want[c].to_numpy(float)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12,
                                       err_msg=f"{c} drifted from frozen formula")

    def test_no_helper_columns_leak(self):
        df = add_exp2_features(_synthetic_frame())
        leaked = [c for c in df.columns if c.startswith("_exp2_side_")]
        self.assertEqual(leaked, [], "intermediate side columns must be dropped")

    def test_missing_source_columns_ship_all_nan(self):
        df = _synthetic_frame(10).drop(columns=["league_k_pct"])
        out = add_exp2_features(df)
        for c in EXP2_CANDIDATE_COLS:
            self.assertTrue(out[c].isna().all(),
                            f"{c} must stay NaN without its sources, never 0")

    def test_nan_inputs_propagate_not_zero(self):
        df = _synthetic_frame(10)
        df.loc[3, "sp_k9_home"] = np.nan
        out = add_exp2_features(df)
        self.assertTrue(np.isnan(out["exp2_centered_k_diff"].iloc[3]))


class TestRunLineTarget(unittest.TestCase):
    def test_true_minus_1_5_target(self):
        df = pd.DataFrame({
            "home_score": [5, 3, 2, 1, 4],
            "away_score": [3, 3, 0, 5, 2],
        })
        out = with_run_line_target(df)
        # margins: +2, 0, +2, −4, +2 → covers: 1, 0, 1, 0, 1
        np.testing.assert_array_equal(out["home_win"].to_numpy(), [1, 0, 1, 0, 1])

    def test_null_alignment_preserved(self):
        df = pd.DataFrame({
            "home_score": [5.0, np.nan, 2.0],
            "away_score": [3.0, 1.0, np.nan],
        })
        out = with_run_line_target(df)
        self.assertTrue(np.isnan(out["home_win"].iloc[0 + 1]))
        self.assertTrue(np.isnan(out["home_win"].iloc[2]))
        self.assertEqual(out["home_win"].iloc[0], 1.0)

    def test_requires_scores(self):
        with self.assertRaises(ValueError):
            with_run_line_target(pd.DataFrame({"home_win": [1.0]}))


class TestFeatureColsInvariant(unittest.TestCase):
    def test_width_and_uniqueness(self):
        # CORRECTED frozen decision: replacement, not expansion —
        # 59 baseline − 6 unique E/F removals + 8 exp2 candidates = 61.
        removals = {"sp_k9_diff", "sp_k9_5g_diff", "sp_fbpct_diff",
                    "sp_whiff_diff", "sp_xwoba_diff", "sp_xwoba_vs_l_diff"}
        self.assertEqual(len(FEATURE_COLS), 61)
        self.assertEqual(len(set(FEATURE_COLS)), 61)
        self.assertEqual(len(FEATURE_COLS), 59 - len(removals) + len(EXP2_CANDIDATE_COLS))
        self.assertEqual(removals & set(FEATURE_COLS), set(),
                         "E/F removal families must be absent")
        self.assertNotIn("sp_xwoba_vs_r_diff", FEATURE_COLS)


if __name__ == "__main__":
    unittest.main()
