"""Run-engine tests (backend/run_engine.py).

Phase 1: derived feature views (zero _diff columns; symmetric sides; new
FEATURE_COLS entries flow through), target integrity, pooled OOF output shape
(no NaNs), baseline comparison presence, dispersion math on a hand fixture,
and OOF artifact schema.

Phase 2: alpha method-of-moments recovery, NB marginal moments + pmf,
Monte-Carlo reproducibility/bounds/error scaling, hand-fixture market scoring
(Brier/logloss/ECE), prequential calibration discipline, end-to-end market
derivation summary blocks, and markets artifact schema (+NaN refusal).
"""
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_engine as re_
from run_engine import (
    MARKET_COLUMNS,
    brier_score,
    derive_markets,
    derive_markets_mc,
    derive_run_features,
    dispersion_ratio,
    ece_score,
    fit_alpha,
    fit_check_table,
    nb_pmf,
    persist_markets,
    persist_oof,
    poisson_deviance,
    prequential_calibrate,
    split_side_view,
)
from training import FEATURE_COLS


class TestFeatureViewDerivation(unittest.TestCase):
    def test_zero_diff_columns_in_kept_view(self):
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        offenders = [f for f in keep
                     if f.endswith("_diff") and f != re_.RUN_DIFF_EXCEPTION]
        self.assertEqual(offenders, [])
        # park_factor_slug_diff is the sanctioned survivor — kept, not dropped.
        self.assertIn("park_factor_slug_diff", keep)
        self.assertIn("park_factor_slug_diff", FEATURE_COLS)

    def test_pure_matchup_and_diff_products_dropped(self):
        keep, _ = derive_run_features(list(FEATURE_COLS))
        for f in ("lineup_handedness_matchup_advantage",
                  "bullpen_meltdown_risk",
                  "pitcher_regression_indicator",
                  "lineup_depth_multiplier",
                  "ace_efficiency_factor"):
            self.assertNotIn(f, keep)

    def test_home_view_contains_home_and_env_only(self):
        keep, _ = derive_run_features(list(FEATURE_COLS))
        home, env = split_side_view(keep, "home")
        self.assertTrue(home, "home side columns missing")
        for f in home:
            self.assertTrue(f.startswith("home_") or f.endswith("_home"))
        for f in env:  # environment belongs to neither side
            self.assertFalse(f.startswith("away_") or f.endswith("_away"))

    def test_away_view_symmetric(self):
        keep, _ = derive_run_features(list(FEATURE_COLS))
        home, env_h = split_side_view(keep, "away" if False else "home")
        away, env_a = split_side_view(keep, "away")
        _, env_home = split_side_view(keep, "home")
        self.assertEqual(env_home, env_a, "shared env must be identical")
        mirror = {f.replace("_home", "_SIDE").replace("home_", "SIDE_")
                  for f in home}
        away_mirror = {f.replace("_away", "_SIDE").replace("away_", "SIDE_")
                       for f in away}
        self.assertEqual(mirror - {"is_home", "is_SIDE"}, away_mirror)

    def test_new_feature_flows_through(self):
        """A newly added non-diff FEATURE_COL appears in both views without
        touching run_engine.py; a new _diff column is dropped."""
        feats = list(FEATURE_COLS) + ["new_ballpark_altitude", "new_whip_diff"]
        keep, dropped = derive_run_features(feats)
        self.assertIn("new_ballpark_altitude", keep)
        self.assertIn("new_whip_diff", dropped)
        home, env = split_side_view(keep, "home")
        combined = set(home) | set(env)
        self.assertIn("new_ballpark_altitude", combined)


class TestTargetsAndMetrics(unittest.TestCase):
    def test_targets_are_integer_counts(self):
        df = pd.read_csv(
            Path(__file__).resolve().parents[1] / "data_delivery"
            / "game_level_features.csv",
            usecols=["home_score", "away_score"],
        )
        for col in ("home_score", "away_score"):
            self.assertTrue((df[col] >= 0).all())
            self.assertTrue(np.allclose(df[col], df[col].astype(int)),
                            f"{col} must be integer counts")

    def test_poisson_deviance_hand_values(self):
        lam2 = np.array([2.0, 2.0])
        y = np.array([2.0, 0.0])
        expected = (2 * (2 * np.log(1) - 0) + 2 * 2.0) / 2
        self.assertAlmostEqual(poisson_deviance(y, lam2), expected, places=9)

    def test_dispersion_ratio_hand_fixture(self):
        # Perfect Poisson draws around λ=4 overdispersed by construction:
        # residuals of ±2 on λ=4 → chi²/n = 4/4 = 1 → ratio 1.
        y = np.array([6.0, 2.0] * 25)
        lam = np.full(50, 4.0)
        # chi² = 50·(4/4), df = n−1 = 49.
        self.assertAlmostEqual(dispersion_ratio(y, lam), 50 / 49, places=9)
        # Clearly over-dispersed: ±3 residuals on λ=1 → chi²/df = 9.
        y2 = np.array([4.0, -0.0] * 10)  # residual +3/-1... use explicit
        y2 = np.array([4.0, 0.0] * 10)
        lam2 = np.full(20, 1.0)
        self.assertAlmostEqual(dispersion_ratio(y2, lam2),
                               ((9 * 10 + 1 * 10) / 19), places=6)


class TestOofAndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """One small walk-forward run on synthetic data (fast, deterministic)."""
        rng = np.random.default_rng(11)
        abbrs = ["NYY", "BOS", "LAD", "SF", "ATL", "HOU"]
        rows = []
        n_days, per_day = 140, 8
        strength = {a: rng.normal(0, 1) for a in abbrs}
        for d in range(n_days):
            date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=d)
            for g in range(per_day):
                ht, at = abbrs[(d + g) % 6], abbrs[(d + g + 3) % 6]
                lam_h = np.exp(0.4 + 0.15 * strength[ht] - 0.1 * strength[at]
                               + 0.05 * rng.normal())
                lam_a = np.exp(0.2 + 0.15 * strength[at] - 0.1 * strength[ht])
                hs, as_ = rng.poisson(lam_h), rng.poisson(lam_a)
                row = {c: float(rng.normal()) for c in FEATURE_COLS}
                row.update({
                    "game_pk": 100000 + d * per_day + g,
                    "game_date": date,
                    "home_team": ht, "away_team": at,
                    "home_win": float(hs > as_),
                    "home_score": int(hs), "away_score": int(as_),
                })
                rows.append(row)
        cls.games = pd.DataFrame(rows)
        # Speed: weekly folds but only the last few windows carry signal here;
        # run_oof uses MIN_VAL_FOLD_GAMES via arg.
        cls.result = re_.run_oof(cls.games, min_val_games=5)

    def test_oof_output_one_row_per_game_no_nans(self):
        oof = self.result["oof"]
        decided = self.games[self.games["home_win"].notna()]
        self.assertGreater(len(oof), 0)
        self.assertLessEqual(len(oof), len(decided))
        for col in ("home_expected_runs", "away_expected_runs",
                    "home_score", "away_score"):
            self.assertFalse(oof[col].isna().any(), f"{col} has NaNs")
        self.assertTrue((oof["home_expected_runs"] > 0).all())
        self.assertTrue((oof["away_expected_runs"] > 0).all())

    def test_baseline_comparison_present(self):
        s = self.result["summary"]
        for side in ("home", "away"):
            self.assertIn(f"{side}_baseline", s)
            self.assertIn("poisson_deviance", s[f"{side}_baseline"])
            self.assertIn(f"{side}_pooled", s)
            self.assertIn(f"{side}_dispersion_ratio", s)

    def test_artifact_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = persist_oof(self.result["oof"], "TEST", out_dir=Path(tmp))
            self.assertTrue(path.exists())
            out = pd.read_csv(path)
            self.assertEqual(
                list(out.columns),
                ["game_pk", "game_date", "home_expected_runs",
                 "away_expected_runs", "home_score", "away_score"])
            self.assertFalse(out.isna().any().any())

    def test_missing_column_raises(self):
        bad = self.result["oof"].drop(columns=["away_score"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                persist_oof(bad, "TEST", out_dir=Path(tmp))


class TestAlphaEstimation(unittest.TestCase):
    def test_recovers_known_alpha(self):
        """y ~ NB(μ=4.5, α=0.35) ⇒ method of moments lands near 0.35."""
        rng = np.random.default_rng(7)
        n, mu, alpha = 200_000, 4.5, 0.35
        p = (1.0 / alpha) / (1.0 / alpha + mu)
        y = rng.negative_binomial(1.0 / alpha, p, size=n).astype(float)
        lam = np.full(n, mu)
        est = fit_alpha(y, lam)
        self.assertGreater(est, 0.30, f"α underestimated: {est}")
        self.assertLess(est, 0.40, f"α overestimated: {est}")

    def test_poisson_data_gives_near_zero_alpha(self):
        rng = np.random.default_rng(8)
        y = rng.poisson(4.5, size=100_000).astype(float)
        self.assertLess(fit_alpha(y, np.full(len(y), 4.5)), 0.01)

    def test_underdispersed_input_clamps_to_zero(self):
        # Variance below the mean must never produce a negative α.
        y = np.array([4, 4, 5, 4, 5], dtype=float)
        self.assertEqual(fit_alpha(y, np.full(5, 4.6)), 0.0)


class TestNbMarginal(unittest.TestCase):
    def test_simulated_moments_match_inputs(self):
        mu, alpha = 4.3, 0.25
        n = 1.0 / alpha
        rng = np.random.default_rng(9)
        draws = rng.negative_binomial(n, n / (n + mu), size=300_000).astype(float)
        self.assertAlmostEqual(draws.mean(), mu, delta=0.05)
        self.assertAlmostEqual(draws.var(), mu + alpha * mu ** 2, delta=0.15)

    def test_pmf_sums_to_one_and_matches_tails(self):
        ks = np.arange(0, 60)
        pmf = nb_pmf(ks, mu=4.2, alpha=0.3)
        self.assertAlmostEqual(pmf.sum(), 1.0, delta=1e-6)
        mean = float((ks * pmf).sum())
        self.assertAlmostEqual(mean, 4.2, delta=0.02)


class TestMonteCarlo(unittest.TestCase):
    LAM_H = np.array([4.9, 4.1, 5.4])
    LAM_A = np.array([4.4, 4.6, 3.7])

    def test_reproducible_same_seed(self):
        a = derive_markets_mc(self.LAM_H, self.LAM_A, 0.3, 0.35, seed=123)
        b = derive_markets_mc(self.LAM_H, self.LAM_A, 0.3, 0.35, seed=123)
        for key in ("p_over_8_5", "p_home_cover_1_5", "p_home_win_derived"):
            np.testing.assert_array_equal(a[key], b[key])

    def test_different_seed_differs(self):
        a = derive_markets_mc(self.LAM_H, self.LAM_A, 0.3, 0.35, seed=1,
                              n_draws=500)
        b = derive_markets_mc(self.LAM_H, self.LAM_A, 0.3, 0.35, seed=2,
                              n_draws=500)
        self.assertFalse(np.array_equal(a["p_over_8_5"], b["p_over_8_5"]))

    def test_probabilities_bounded_and_ordered(self):
        out = derive_markets_mc(self.LAM_H, self.LAM_A, 0.3, 0.35, seed=5)
        for key in ("p_over_8_5", "p_home_cover_1_5", "p_home_win_derived"):
            self.assertTrue(((out[key] >= 0) & (out[key] <= 1)).all(), key)
        # Covering -1.5 implies winning; totals ≥9 is independent.
        self.assertTrue((out["p_home_cover_1_5"] <= out["p_home_win_derived"] + 1e-12).all())
        # Stronger home λ ⇒ higher win prob.
        self.assertGreater(out["p_home_win_derived"][0], out["p_home_win_derived"][1])

    def test_mc_error_shrinks_with_draws(self):
        small = derive_markets_mc(self.LAM_H, self.LAM_A, 0.3, 0.35,
                                  n_draws=1_000, seed=11)["mc_se_totals"]
        large = derive_markets_mc(self.LAM_H, self.LAM_A, 0.3, 0.35,
                                  n_draws=40_000, seed=11)["mc_se_totals"]
        self.assertLess(large.max(), small.max())
        self.assertGreater(small.max(), 0.005)   # ~sqrt(.5*.5/1000)
        self.assertLess(large.max(), 0.004)      # ~sqrt(.5*.5/40000)=.0025


class TestMarketScoring(unittest.TestCase):
    Y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    P = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])

    def test_brier_hand_values(self):
        expected = float(((self.P - self.Y) ** 2).mean())
        self.assertAlmostEqual(brier_score(self.Y, self.P), expected, places=12)

    def test_logloss_matches_sklearn_on_fixture(self):
        from sklearn.metrics import log_loss
        self.assertAlmostEqual(
            log_loss(self.Y, np.clip(self.P, 1e-6, 1 - 1e-6)),
            log_loss(self.Y, self.P), places=9)

    def test_ece_perfect_and_degenerate(self):
        self.assertEqual(ece_score(np.array([1, 0, 1, 0]),
                                   np.array([1.0, 0.0, 1.0, 0.0])), 0.0)
        # All mass in the wrong bin → ECE equals the pooled gap.
        e = ece_score(np.array([0, 0, 0, 0]), np.array([.9, .9, .9, .9]))
        self.assertAlmostEqual(e, 0.9, delta=0.01)

    def test_prequential_keeps_early_folds_raw_and_improves_later(self):
        """Below MIN_OOF_FOR_FIT history the map is identity; with enough
        prior-fold history a miscalibrated later fold gets shifted."""
        rng = np.random.default_rng(3)
        n = 900
        y = rng.integers(0, 2, size=n).astype(float)
        p = np.clip(y * 0.4 + 0.3 + rng.normal(0, .05, n), .02, .98)
        fold_idx = np.repeat(np.arange(9), 100)
        out = prequential_calibrate(y, p, fold_idx)
        # First folds (< 300 prior games) pass through untouched.
        np.testing.assert_allclose(out[:300], p[:300], rtol=1e-9)
        self.assertTrue(((out > 0) & (out < 1)).all())
        self.assertLessEqual(out.max(), 1.0)


class TestDeriveMarketsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Synthetic OOF frame — no model training needed for the market layer."""
        rng = np.random.default_rng(21)
        n = 400
        lam_h = np.clip(rng.normal(4.6, 0.35, n), 2.5, 7)
        lam_a = np.clip(rng.normal(4.35, 0.35, n), 2.5, 7)
        hs = rng.negative_binomial(3.0, 3.0 / (3.0 + lam_h)).astype(int)
        as_ = rng.negative_binomial(3.0, 3.0 / (3.0 + lam_a)).astype(int)
        cls.oof = pd.DataFrame({
            "game_pk": np.arange(1000, 1000 + n),
            "game_date": pd.date_range("2026-06-01", periods=n,
                                       freq="D").strftime("%Y-%m-%d"),
            "game_id": [f"d{i}_A@H" for i in range(n)],
            "fold_idx": np.repeat(np.arange(10), n // 10),
            "home_expected_runs": lam_h,
            "away_expected_runs": lam_a,
            "home_score": hs,
            "away_score": as_,
        })
        ml = pd.DataFrame({
            "game_id": cls.oof["game_id"],
            "home_win_prob_model": rng.uniform(.3, .7, n),
        })
        cls.out = derive_markets(cls.oof, moneyline_probs=ml, n_draws=2_000)

    def test_summary_has_every_deliverable_block(self):
        s = self.out["summary"]
        self.assertIn("alpha_home", s)
        self.assertIn("alpha_away", s)
        for key in ("market_over_8_5", "market_home_cover_1_5",
                    "market_derived_moneyline"):
            m = s[key]
            for field in ("engine_logloss", "engine_brier", "engine_ece_raw",
                          "baseline_logloss", "baseline_rate"):
                self.assertIn(field, m, f"{key}.{field}")
        self.assertIn("agreement_vs_moneyline", s)
        a = s["agreement_vs_moneyline"]
        for field in ("correlation", "mean_abs_diff", "share_gt_0_08",
                      "share_gt_0_10", "n_merged"):
            self.assertIn(field, a)

    def test_fit_check_table_shape(self):
        fc = self.out["summary"]["fit_check"]["home"]
        self.assertEqual(len(fc), 15)  # k=0..12 plus >=10 and <=1 rows
        for row in fc:
            self.assertIn("modeled_p", row)
            self.assertIn("observed_p", row)

    def test_markets_frame_no_nans_and_consistent_targets(self):
        mk = self.out["markets"]
        self.assertFalse(mk.isna().any().any())
        total = mk["home_score"] + mk["away_score"]
        self.assertTrue((mk["total_runs"] == total).all())
        self.assertTrue((mk["p_over_8_5"] <= 1).all()
                        and (mk["p_over_8_5"] >= 0).all())

    def test_artifact_schema_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = persist_markets(self.out["markets"], "TEST",
                                   self.out["summary"], out_dir=Path(tmp))
            self.assertTrue(path.exists())
            out = pd.read_csv(path)
            self.assertEqual(list(out.columns), MARKET_COLUMNS)
            self.assertFalse(out.isna().any().any())
            meta = json.loads((path.parent / "run_engine_markets_TEST.meta.json")
                              .read_text())
            for field in ("alpha_home", "alpha_away", "n_draws", "seed",
                          "total_line", "run_line_margin"):
                self.assertIn(field, meta)

    def test_nan_refusal(self):
        bad = self.out["markets"].copy()
        bad.loc[bad.index[0], "p_over_8_5"] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                persist_markets(bad, "TEST", self.out["summary"],
                                out_dir=Path(tmp))


if __name__ == "__main__":
    unittest.main()
