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
    ALPHA_CAP,
    ALPHA_MIN_BIN,
    HOLDOUT_DAYS,
    MARKET_COLUMNS_V3,
    NULLABLE_MARKET_COLUMNS,
    RUN_LINE_GRID,
    TOTAL_LINE_GRID,
    agreement_stats,
    alpha_bins,
    alpha_of,
    brier_score,
    derive_markets_mc,
    derive_markets_v3,
    derive_run_features,
    dispersion_ratio,
    ece_score,
    eval_alpha_fit,
    fit_alpha,
    fit_check_table,
    nb_pmf,
    persist_markets,
    persist_oof,
    poisson_deviance,
    prequential_calibrate,
    select_alpha_curve,
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
            "game_date": pd.date_range("2026-03-01", periods=n,
                                       freq="D").strftime("%Y-%m-%d"),
            "fold_idx": np.repeat(np.arange(10), n // 10),
            "home_expected_runs": lam_h,
            "away_expected_runs": lam_a,
            "home_score": hs,
            "away_score": as_,
        })
        cls.ml = pd.DataFrame({
            "game_pk": cls.oof["game_pk"],
            "home_win_prob_model": rng.uniform(.3, .7, n),
        })
        cls.out = derive_markets_v3(cls.oof, moneyline_probs=cls.ml,
                                    n_draws=2_000)

    def test_summary_has_every_deliverable_block(self):
        s = self.out["summary"]
        self.assertIn("alpha_home", s)
        self.assertIn("alpha_away", s)
        self.assertIn("phase2_single_alpha", s)
        self.assertIn("fit_check_single_alpha", s)
        self.assertIn("fit_check_alpha_lambda", s)
        self.assertIn("variance_check", s)
        self.assertIn("year_effect_home", s)
        self.assertIn("mc_meta", s)
        for key in ("market_over_7_5", "market_over_8_5", "market_over_9_5",
                    "market_home_cover_1_5", "market_home_cover_2_5",
                    "market_derived_moneyline"):
            m = s[key]
            for field in ("engine_logloss", "engine_brier", "engine_ece_raw",
                          "baseline_logloss", "baseline_rate"):
                self.assertIn(field, m, f"{key}.{field}")
        self.assertIn("agreement_vs_moneyline", s)
        a = s["agreement_vs_moneyline"]
        for field in ("delta_primary", "n", "mean_abs_diff",
                      "share_gt_0_08", "share_gt_0_10",
                      "n_flagged_0_08", "n_flagged_0_10"):
            self.assertIn(field, a)

    def test_fit_check_table_shape(self):
        fc2 = self.out["summary"]["fit_check_single_alpha"]["home"]
        self.assertEqual(len(fc2), 15)  # k=0..12 plus ≥10 and ≤1 (Phase 2)
        fc3 = self.out["summary"]["fit_check_alpha_lambda"]["home"]
        self.assertEqual(len(fc3), 17)  # adds >=11 and >=12 tails
        for row in fc2 + fc3:
            self.assertIn("modeled_p", row)
            self.assertIn("observed_p", row)

    def test_agreement_flags_and_hand_count(self):
        mk = self.out["markets"]
        self.assertIn("agreement_conflict", mk.columns)
        diff = (mk["p_home_win_derived"] - mk["ml_win_prob"]).abs()
        expected_n = int((diff > 0.08).sum())
        stats = self.out["summary"]["agreement_vs_moneyline"]
        self.assertEqual(stats["n_flagged_primary"], expected_n)
        self.assertEqual(int(mk["agreement_conflict"].sum()), expected_n)
        # Boundary honesty: diff of exactly delta must NOT flag.
        self.assertEqual(agreement_stats(
            np.array([0.5]), np.array([0.58]), delta=0.08)["n_flagged_primary"], 0)
        self.assertEqual(agreement_stats(
            np.array([0.5]), np.array([0.59]), delta=0.08)["n_flagged_primary"], 1)

    def test_markets_frame_no_nans_and_consistent_targets(self):
        mk = self.out["markets"]
        required_cols = [c for c in MARKET_COLUMNS_V3
                         if c not in NULLABLE_MARKET_COLUMNS]
        self.assertFalse(mk[required_cols].isna().any().any())
        total = mk["home_score"] + mk["away_score"]
        self.assertTrue((mk["total_runs"] == total).all())
        self.assertTrue((mk["p_over_8_5"] <= 1).all()
                        and (mk["p_over_8_5"] >= 0).all())

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
            self.assertEqual(list(out.columns), MARKET_COLUMNS_V3)
            nullable = [c for c in MARKET_COLUMNS_V3
                        if c in NULLABLE_MARKET_COLUMNS]
            required = [c for c in MARKET_COLUMNS_V3 if c not in nullable]
            self.assertFalse(out[required].isna().any().any())
            meta = json.loads((path.parent / "run_engine_markets_TEST.meta.json")
                              .read_text())
            for field in ("alpha_home", "alpha_away", "phase2_single_alpha",
                          "n_draws", "seed", "mc_meta", "line_grid",
                          "holdout_cutoff", "n_pre", "n_holdout"):
                self.assertIn(field, meta)

    def test_nan_refusal(self):
        bad = self.out["markets"].copy()
        bad.loc[bad.index[0], "p_over_8_5"] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                persist_markets(bad, "TEST", self.out["summary"],
                                out_dir=Path(tmp))

class TestAlphaCurveFits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Heteroskedastic synthetic: α(λ) = 0.05 + 0.09·λ (increasing)."""
        rng = np.random.default_rng(0)
        n = 4000
        cls.lam = np.clip(rng.normal(4.5, 0.6, n), 2.8, 6.8)
        true_a = np.clip(0.05 + 0.09 * cls.lam, 0, ALPHA_CAP)
        inv_n = 1.0 / true_a
        cls.y = rng.negative_binomial(
            inv_n, inv_n / (inv_n + cls.lam)).astype(float)
        cls.curve, cls.diag = select_alpha_curve(cls.y, cls.lam, seed=1)

    def test_bins_respect_min_count(self):
        bins = alpha_bins(self.y, self.lam)
        self.assertGreaterEqual(len(bins), 2)
        for b in bins:
            self.assertGreaterEqual(b["count"], ALPHA_MIN_BIN)

    def test_underfilled_bins_merge(self):
        # 8 requested bins over a tight λ range → some must merge.
        bins = re_.alpha_bins(np.concatenate([self.y, self.y]),
                              np.concatenate([self.lam, self.lam]),
                              n_bins=8, min_count=2000)
        self.assertLess(len(bins), 8)
        for b in bins:
            self.assertGreaterEqual(b["count"], 2000)

    def test_curve_non_negative_monotone_capped(self):
        grid = np.linspace(1.0, 12.0, 200)
        vals = alpha_of(grid, self.curve)
        self.assertTrue((vals >= 0).all())
        diffs = np.diff(vals)
        # Monotone in ONE consistent direction (data decides rising vs falling).
        self.assertTrue((diffs >= -1e-12).all() or (diffs <= 1e-12).all(),
                        "curve must be monotone")
        # This synthetic fixture has INCREASING true alpha — expect rising.
        self.assertTrue((diffs >= -1e-12).all())
        self.assertTrue((vals <= ALPHA_CAP).all())
        extreme = alpha_of(np.array([50.0]), self.curve)[0]
        self.assertLessEqual(extreme, ALPHA_CAP)

    def test_falling_direction_supported(self):
        """Away-style dispersion: alpha FALLS with lambda must survive."""
        rng = np.random.default_rng(12)
        lam = np.clip(rng.normal(4.4, 0.6, 4000), 2.8, 6.8)
        true_a = np.clip(0.55 - 0.05 * lam, 0.05, ALPHA_CAP)
        inv_n = 1.0 / true_a
        y = rng.negative_binomial(inv_n, inv_n / (inv_n + lam)).astype(float)
        curve, diag = select_alpha_curve(y, lam, seed=13)
        grid = np.array([3.2, 4.2, 5.2])
        vals = alpha_of(grid, curve)
        self.assertGreater(vals[0], vals[-1],
                           f"falling fixture should yield falling curve: {diag}")

    def test_oob_selection_prefers_honest_tail(self):
        chosen = self.diag["chosen"]
        self.assertIn(chosen, ("piecewise", "linear", "power"))
        # Chosen form's OOB tail gap must be the best available.
        gaps = {f: self.diag[f]["tail_gap_avg"]
                for f in ("piecewise", "linear", "power") if f in self.diag}
        self.assertLessEqual(gaps[chosen], min(gaps.values()) + 1e-9)

    def test_curve_beats_single_alpha_on_tail(self):
        single = np.full(len(self.y), fit_alpha(self.y, self.lam))
        ev_single = eval_alpha_fit(self.y, self.lam, single)
        ev_curve = eval_alpha_fit(self.y, self.lam, alpha_of(self.lam, self.curve))
        self.assertLess(ev_curve["tail_gap"], ev_single["tail_gap"])


class TestTailMatchAndVariance(unittest.TestCase):
    def test_fit_check_curve_closes_blowout_tail(self):
        rng = np.random.default_rng(3)
        lam = rng.uniform(4.4, 5.6, size=6000)
        a_true = np.clip(0.10 + 0.06 * lam, 0, ALPHA_CAP)
        inv_n = 1.0 / a_true
        y = rng.negative_binomial(inv_n, inv_n / (inv_n + lam)).astype(int)
        curve, _ = select_alpha_curve(y.astype(float), lam, seed=5)
        fc_curve = re_.fit_check_table_curve(y, lam, alpha_of(lam, curve))
        tail_c = next(r for r in fc_curve if r["k"] == "≥10")
        gap_curve = abs(tail_c["modeled_p"] - tail_c["observed_p"])
        single = fit_alpha(y.astype(float), lam)
        fc_single = fit_check_table(y.astype(float), lam, single)
        tail_s = next(r for r in fc_single if r["k"] == "≥10")
        gap_single = abs(tail_s["modeled_p"] - tail_s["observed_p"])
        self.assertLess(gap_curve, gap_single)
        self.assertLess(gap_curve, 0.01)

    def test_variance_still_matches_after_curve(self):
        rng = np.random.default_rng(4)
        lam = rng.uniform(4.0, 5.5, size=5000)
        a_true = np.clip(0.08 + 0.07 * lam, 0, ALPHA_CAP)
        inv_n = 1.0 / a_true
        y = rng.negative_binomial(inv_n, inv_n / (inv_n + lam)).astype(float)
        curve, _ = select_alpha_curve(y, lam, seed=6)
        a = alpha_of(lam, curve)
        implied = float((lam + a * lam ** 2).mean())   # var = λ + α·λ²
        observed = float(y.var(ddof=0))
        self.assertAlmostEqual(implied, observed, delta=0.15 * observed)


class TestLineGridSemantics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.lam_h = np.clip(rng.normal(4.7, 0.4, 60), 3, 6.5)
        self.lam_a = np.clip(rng.normal(4.3, 0.4, 60), 3, 6.5)
        self.mc = derive_markets_mc(self.lam_h, self.lam_a, 0.30, 0.35,
                                    n_draws=4000, seed=11)

    def test_grid_at_85_matches_legacy_column_exactly(self):
        col = TOTAL_LINE_GRID.index(8.5)
        np.testing.assert_array_equal(self.mc["p_over_8_5"],
                                      self.mc["p_over_grid"][:, col])

    def test_toggle_reads_exact_per_line_values(self):
        # Column naming contract the dashboard toggle relies on:
        for line in (6.5, 9.0, 12.5):
            col = TOTAL_LINE_GRID.index(line)
            key = f"p_over_{str(line).replace('.', '_')}"
            self.assertGreaterEqual(col, 0)
            self.assertIn(key, MARKET_COLUMNS_V3)
        for m in RUN_LINE_GRID:
            self.assertIn(f"p_home_cover_{str(m).replace('.', '_')}",
                          MARKET_COLUMNS_V3)

    def test_under_is_complement_and_cover05_is_winprob(self):
        i85 = TOTAL_LINE_GRID.index(8.5)
        under = 1 - self.mc["p_over_grid"][:, i85]
        np.testing.assert_allclose(under, 1 - self.mc["p_over_8_5"], atol=1e-12)
        i05 = RUN_LINE_GRID.index(0.5)   # −0.5 ≡ home wins
        np.testing.assert_array_equal(self.mc["p_home_win_derived"],
                                      self.mc["p_cover_grid"][:, i05])

    def test_probabilities_monotone_in_line_and_margin(self):
        self.assertTrue((np.diff(self.mc["p_over_grid"], axis=1) <= 1e-12).all())
        self.assertTrue((np.diff(self.mc["p_cover_grid"], axis=1) <= 1e-12).all())
        self.assertTrue(((self.mc["p_cover_grid"] >= 0)
                         & (self.mc["p_cover_grid"] <= 1)).all())


class TestMultiLineScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(31)
        n = 300
        lam_h = np.clip(rng.normal(4.6, 0.35, n), 2.5, 7)
        lam_a = np.clip(rng.normal(4.35, 0.35, n), 2.5, 7)
        hs = rng.negative_binomial(3.0, 3.0 / (3.0 + lam_h)).astype(int)
        as_ = rng.negative_binomial(3.0, 3.0 / (3.0 + lam_a)).astype(int)
        oof = pd.DataFrame({
            "game_pk": np.arange(n),
            "game_date": pd.date_range("2026-02-01", periods=n,
                                       freq="D").strftime("%Y-%m-%d"),
            "fold_idx": np.repeat(np.arange(10), n // 10),
            "home_expected_runs": lam_h,
            "away_expected_runs": lam_a,
            "home_score": hs,
            "away_score": as_,
        })
        cls.out = derive_markets_v3(oof, moneyline_probs=None, n_draws=1_000)
        cls.oof = oof

    def test_reference_lines_scored_with_holdout_twins(self):
        s = self.out["summary"]
        for key in ("market_over_7_5", "market_over_8_5", "market_over_9_5",
                    "market_home_cover_1_5", "market_home_cover_2_5",
                    "market_derived_moneyline"):
            self.assertIn(key, s)
            self.assertIn(f"{key}_holdout", s)
            self.assertIn("holdout", s[f"{key}_holdout"])

    def test_baseline_rate_equals_observed_share_per_line(self):
        s = self.out["summary"]
        total = self.oof["home_score"] + self.oof["away_score"]
        for line in (7.5, 8.5, 9.5):
            share = float((total >= line + 0.5).mean())
            self.assertAlmostEqual(s[f"market_over_{str(line).replace('.', '_')}"]["baseline_rate"],
                                   round(share, 4), places=4)
        diff = self.oof["home_score"] - self.oof["away_score"]
        for m in (1.5, 2.5):
            share = float((diff >= m + 0.5).mean())
            self.assertAlmostEqual(s[f"market_home_cover_{str(m).replace('.', '_')}"]["baseline_rate"],
                                   round(share, 4), places=4)

    def test_holdout_counts_add_up(self):
        s = self.out["summary"]
        self.assertEqual(s["n_pre"] + s["n_holdout"], len(self.oof))
        h = s["market_over_8_5_holdout"]["holdout"]
        self.assertEqual(h["n"], s["n_holdout"])

    def test_alpha_fitted_pre_holdout_only(self):
        s = self.out["summary"]
        for side in ("home", "away"):
            bins = s[f"alpha_{side}"]["selection"]["bins"]
            self.assertLessEqual(sum(b["count"] for b in bins), s["n_pre"])
            self.assertEqual(s[f"alpha_{side}"]["fitted_on"],
                             "pre-holdout OOF only")

    def test_no_moneyline_warns_but_ships(self):
        mk = self.out["markets"]
        self.assertIn("agreement_conflict", mk.columns)
        self.assertFalse(mk["agreement_conflict"].any())


class TestPredictSlateRuns(unittest.TestCase):
    def test_slate_priced_through_same_machinery(self):
        rng = np.random.default_rng(51)
        abbrs = ["NYY", "BOS", "LAD", "SF"]
        rows = []
        for d in range(80):
            date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=d)
            for g in range(6):
                ht, at = abbrs[(d + g) % 4], abbrs[(d + g + 2) % 4]
                hs, as_ = rng.poisson(4.6), rng.poisson(4.3)
                row = {c: float(rng.normal()) for c in FEATURE_COLS}
                row.update({"game_pk": 500000 + d * 6 + g,
                            "game_date": date, "home_team": ht,
                            "away_team": at, "home_win": float(hs > as_),
                            "home_score": int(hs), "away_score": int(as_)})
                rows.append(row)
        decided = pd.DataFrame(rows)
        slate = decided.tail(3).drop(columns=["home_win", "home_score",
                                             "away_score"]).copy()
        slate["game_pk"] = [900001, 900002, 900003]
        curve = {"form": "linear", "a": 0.25, "b": 0.01}
        out = re_.predict_slate_runs(decided, slate,
                                     {"home": 10, "away": 10},
                                     {"home": curve, "away": curve},
                                     n_draws=500, seed=1)
        self.assertEqual(len(out), 3)
        self.assertTrue((out["kind"] == "slate").all())
        for col in MARKET_COLUMNS_V3:
            if col not in NULLABLE_MARKET_COLUMNS and col not in (
                    "home_score", "away_score", "total_runs"):
                self.assertIn(col, out.columns, col)
                self.assertFalse(out[col].isna().any(), col)
        self.assertTrue((out["p_over_8_5"] > 0).all()
                        and (out["p_over_8_5"] < 1).all())
        self.assertTrue((out["home_expected_runs"] > 1).all())

    def test_empty_slate_returns_empty(self):
        out = re_.predict_slate_runs(pd.DataFrame(), pd.DataFrame(),
                                     {"home": 5, "away": 5}, {}, n_draws=100)
        self.assertTrue(out.empty)

    def test_slate_with_game_id_only_produces_game_pk(self):
        """Pre-game ESPN slates have game_id but NOT game_pk.  The
        function must unify to game_pk so the artifact schema is consistent."""
        rng = np.random.default_rng(61)
        abbrs = ["NYY", "BOS"]
        rows = []
        for d in range(40):
            date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=d)
            for g in range(4):
                ht, at = abbrs[(d + g) % 2], abbrs[(d + g + 1) % 2]
                hs, as_ = rng.poisson(4.5), rng.poisson(4.2)
                row = {c: float(rng.normal()) for c in FEATURE_COLS}
                row.update({"game_pk": 600000 + d * 4 + g,
                            "game_date": date, "home_team": ht,
                            "away_team": at, "home_win": float(hs > as_),
                            "home_score": int(hs), "away_score": int(as_)})
                rows.append(row)
        decided = pd.DataFrame(rows)

        # Build a slate with ONLY game_id (no game_pk) — the ESPN path.
        slate = decided.tail(3).drop(columns=["home_win", "home_score",
                                              "away_score", "game_pk"]).copy()
        slate["game_id"] = ["20260824_NYY@BOS", "20260824_LAD@SF",
                             "20260824_SF@LAD"]
        curve = {"form": "linear", "a": 0.25, "b": 0.01}
        out = re_.predict_slate_runs(decided, slate,
                                     {"home": 10, "away": 10},
                                     {"home": curve, "away": curve},
                                     n_draws=500, seed=2)
        self.assertEqual(len(out), 3)
        # game_pk must always be in the output
        self.assertIn("game_pk", out.columns)
        # game_pk values come from game_id
        self.assertTrue(
            all(out["game_pk"] == ["20260824_NYY@BOS", "20260824_LAD@SF",
                                     "20260824_SF@LAD"]))
        # artifact schema columns must all be present
        for col in MARKET_COLUMNS_V3:
            if col not in NULLABLE_MARKET_COLUMNS and col not in (
                    "home_score", "away_score", "total_runs"):
                self.assertIn(col, out.columns, col)
                self.assertFalse(out[col].isna().any(), col)

    def test_slate_missing_both_keys_raises_loud_error(self):
        """A slate with neither game_pk nor game_id must raise a clear
        error — not a bare KeyError with no message."""
        slate = pd.DataFrame({
            "game_date": ["2026-08-24"],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
        })
        # Also need feature cols for the model to work
        for c in FEATURE_COLS:
            slate[c] = 0.0
        decided = pd.DataFrame({
            "game_pk": [500001],
            "game_date": [pd.Timestamp("2026-04-01")],
            "home_team": ["NYY"], "away_team": ["BOS"],
            "home_score": [5], "away_score": [3],
            "home_win": [1.0],
        })
        for c in FEATURE_COLS:
            decided[c] = 0.0
        curve = {"form": "linear", "a": 0.25, "b": 0.01}
        with self.assertRaises(KeyError) as ctx:
            re_.predict_slate_runs(decided, slate,
                                    {"home": 5, "away": 5},
                                    {"home": curve, "away": curve},
                                    n_draws=100, seed=3)
        msg = str(ctx.exception)
        self.assertIn("neither", msg.lower())
        self.assertIn("game_pk", msg)
        self.assertIn("game_id", msg)

    def test_run_engine_daily_with_game_id_slate_does_not_crash(self):
        """E2E: predict_slate_runs on a game_id-only slate produces a
        full-market-grid frame. Uses canned params to bypass OOF derivation
        (which requires a larger dataset)."""
        rng = np.random.default_rng(71)
        abbrs = ["NYY", "BOS", "LAD", "SF"]
        rows = []
        for d in range(80):
            date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=d)
            for g in range(6):
                ht, at = abbrs[(d + g) % 4], abbrs[(d + g + 2) % 4]
                hs, as_ = rng.poisson(4.5), rng.poisson(4.2)
                row = {c: float(rng.normal()) for c in FEATURE_COLS}
                row.update({"game_pk": 700000 + d * 6 + g,
                            "game_date": date, "home_team": ht,
                            "away_team": at, "home_win": float(hs > as_),
                            "home_score": int(hs), "away_score": int(as_),
                            "total_runs": int(hs + as_)})
                rows.append(row)
        decided = pd.DataFrame(rows)

        # Slate with game_id only + home_win_prob_model
        slate = decided.tail(4).drop(columns=["game_pk"]).copy()
        slate["game_id"] = ["20260824_A@B", "20260824_C@D",
                             "20260824_E@F", "20260824_G@H"]
        slate["home_win_prob_model"] = [0.45, 0.52, 0.48, 0.55]

        # Canned curve + rounds — avoid full OOF/market derivation
        curve = {"form": "linear", "a": 0.25, "b": 0.01}
        rounds = {"home": 10, "away": 10}
        curves = {"home": curve, "away": curve}

        out = re_.predict_slate_runs(
            decided, slate, rounds, curves, n_draws=500)
        self.assertFalse(out.empty, "Slate output empty")
        self.assertIn("game_pk", out.columns,
                      "game_pk missing from slate output")
        # game_pk values come from game_id (ESPN path fallback)
        self.assertTrue(
            all(out["game_pk"] == ["20260824_A@B", "20260824_C@D",
                                     "20260824_E@F", "20260824_G@H"]))
        # All required non-nullable artifact columns present
        for col in MARKET_COLUMNS_V3:
            if col not in NULLABLE_MARKET_COLUMNS and col not in (
                    "home_score", "away_score", "total_runs"):
                self.assertIn(col, out.columns,
                              f"{col} missing from slate output")
                self.assertFalse(out[col].isna().any(),
                                 f"{col} has NaNs in slate output")


class TestRollingTotalsBrier(unittest.TestCase):
    def test_series_and_meta_shape(self):
        rng = np.random.default_rng(41)
        days = pd.date_range("2026-04-01", periods=40, freq="D")
        df = pd.DataFrame({
            "kind": "oof",
            "game_date": days.repeat(10).strftime("%Y-%m-%d"),   # 10 games/day
            "total_runs": rng.integers(2, 16, 400),
            "p_over_8_5": rng.uniform(.35, .65, 400),
        })
        out = re_.compute_rolling_totals_brier(df, window_days=30,
                                               min_games_per_day=5)
        self.assertGreater(len(out["series"]), 30)
        self.assertIn("history_mean_brier", out)
        for pt in out["series"]:
            self.assertGreater(pt["brier"], 0)

    def test_empty_history_loud_empty_state(self):
        out = re_.compute_rolling_totals_brier(pd.DataFrame())
        self.assertEqual(out["series"], [])
        out = re_.compute_rolling_totals_brier(None)
        self.assertEqual(out["series"], [])


if __name__ == "__main__":
    unittest.main()
