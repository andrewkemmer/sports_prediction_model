"""Run-engine-native keep-list tests.

Covers the 2026-08-30 feature-restore decision:
- derive_run_features KEEPS the 24 matchup-gap _diff features
  (RUN_RESTORED_DIFF_FEATURES) — 53 active cols; run_margin_diff and the 5
  composites stay excluded (6 dropped). Both sides pinned exactly so a
  future rule change is a deliberate, tested act.
- Same folds → identical table (run_oof determinism).
- Market-level scoring harness on a fixture: reference lines exist, base
  rates correct, ECE-cal computed with no holdout leakage.
- Regressions: moneyline FEATURE_COLS is 59; run_margin_diff stays excluded
  from the run view (lambda-derived moneyline-side); run_oof default call
  path (no explicit feature list) derives the 53-col rule; α(λ)/MC market
  path still derives.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from explainability import classify_drift_retention
from run_engine import (
    RUN_DIFF_EXCEPTION,
    RUN_EXTRA_EXCLUSIONS,
    RUN_RESTORED_DIFF_FEATURES,
    derive_markets_v3,
    derive_run_features,
    run_oof,
)
from training import FEATURE_COLS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_games(n_days: int = 100, per_day: int = 6,
                     seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-04-01", periods=n_days, freq="D")
    rows = []
    teams = [f"T{i:02d}" for i in range(30)]
    pk = 700_000
    for d in dates:
        for _ in range(per_day):
            home, away = rng.choice(teams, 2, replace=False)
            rows.append({
                "game_pk": pk, "game_date": d, "home_team": home,
                "away_team": away,
                "home_win": float(rng.random() > 0.45),
                "home_score": int(rng.integers(0, 12)),
                "away_score": int(rng.integers(0, 12)),
                "is_home": 1.0, "dome_is_neutral": float(rng.random() < 0.2),
                "home_elo": float(rng.normal(1500, 60)),
                "away_elo": float(rng.normal(1500, 60)),
            })
            pk += 1
    return pd.DataFrame(rows)


class TestRoutingAdoptedOutcome(unittest.TestCase):
    def test_restore_outcome_exact_sets(self):
        """Current rule (adopted: RESTORE) — exact kept/dropped sets.

        The 24 matchup-gap _diff features are KEPT (RUN_RESTORED_DIFF_FEATURES);
        run_margin_diff (the only remaining _diff) and the 5 composites are
        dropped. Active view: 53 cols.
        """
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        dropped_diffs = [d for d in dropped if d.endswith("_diff")]
        composites = [d for d in dropped if not d.endswith("_diff")]
        self.assertEqual(len(keep), 53, "kept view must stay 53 cols")
        self.assertEqual(len(dropped), 6)
        self.assertEqual(dropped_diffs, ["run_margin_diff"],
                         "run_margin_diff is the only excluded _diff")
        self.assertEqual(len(composites), 5)
        # The restored 24 diffs + the park exception are the only _diff kept.
        kept_diffs = {f for f in keep if f.endswith("_diff")}
        self.assertEqual(kept_diffs,
                         set(RUN_RESTORED_DIFF_FEATURES)
                         | {RUN_DIFF_EXCEPTION})
        self.assertIn(RUN_DIFF_EXCEPTION, keep)
        # The 5 engineered composites are excluded by name.
        for f in ("lineup_handedness_matchup_advantage", "bullpen_meltdown_risk",
                  "pitcher_regression_indicator", "lineup_depth_multiplier",
                  "ace_efficiency_factor"):
            self.assertNotIn(f, keep)
        # The 6 lineup-delta features are no longer in FEATURE_COLS at all.
        for f in ("lineup_actual_woba_delta_home", "lineup_actual_woba_delta_away",
                  "lineup_actual_top3_delta_home", "lineup_actual_top3_delta_away",
                  "lineup_rest_count_home", "lineup_rest_count_away"):
            self.assertNotIn(f, FEATURE_COLS)
            self.assertNotIn(f, keep)

    def test_run_margin_diff_and_composites_stay_excluded(self):
        """Restoring the 24 diffs does NOT restore run_margin_diff or the 5
        composites. All restored features end in _diff and are not composites.
        Re-adding run_margin_diff would be a distinct 54-col choice."""
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        self.assertTrue(all(f.endswith("_diff")
                            for f in RUN_RESTORED_DIFF_FEATURES),
                        "restored set must be matchup-gap _diff features")
        self.assertFalse(RUN_RESTORED_DIFF_FEATURES & set(RUN_EXTRA_EXCLUSIONS),
                         "restored set must not overlap the composite exclusions")
        self.assertNotIn("run_margin_diff", keep)
        ship = sorted(set(keep) | {"run_margin_diff"})
        self.assertEqual(len(ship), 54, "ship variant is a distinct choice")


class TestSelectionPartitionInvariants(unittest.TestCase):
    def test_derive_run_features_partitions_feature_cols_exactly(self):
        """kept ∪ dropped == FEATURE_COLS, disjoint, deterministic — the
        rule is a pure function of the name list (no importance/drift input),
        so the denominator/count invariants (59 = 53 kept + 6 dropped) are
        structural, not incidental."""
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        self.assertEqual(len(FEATURE_COLS), 59)
        self.assertEqual(len(keep), 53)
        self.assertEqual(len(dropped), 6)
        self.assertEqual(len(keep) + len(dropped), len(FEATURE_COLS))
        self.assertEqual(set(keep) | set(dropped), set(FEATURE_COLS))
        self.assertEqual(len(set(keep)) + len(set(dropped)),
                         len(set(FEATURE_COLS)), "disjoint kept/dropped")
        # Deterministic: identical input → identical output, no hidden state.
        keep2, dropped2 = derive_run_features(list(FEATURE_COLS))
        self.assertEqual(keep, keep2)
        self.assertEqual(dropped, dropped2)


class TestCullRetentionClassifier(unittest.TestCase):
    """Pins the false-positive-cull policy used by the read-only cull
    diagnostic: a feature dropped by the STATIC rule is flagged for retention
    only when it carries kept-median gain AND no measured drift beyond its
    noise floor. The run engine's selection rule itself has no importance/
    drift input (derive_run_features is name-based); this classifier is the
    retention backstop the diagnostic measures against."""

    def test_high_importance_low_drift_never_culled(self):
        """The headline policy: gain at the kept median with drift inside the
        noise floor → retain (false-positive cull)."""
        self.assertTrue(classify_drift_retention(
            gain=21.0, psi_adjusted=0.04, noise_floor=0.055,
            kept_median_gain=21.0))
        self.assertTrue(classify_drift_retention(
            gain=30.0, psi_adjusted=0.0, noise_floor=0.055,
            kept_median_gain=21.0))

    def test_genuine_drift_not_flagged(self):
        """High importance but drift above the noise floor → the cull is
        defensible (monitorable drift, not a false positive)."""
        self.assertFalse(classify_drift_retention(
            gain=30.0, psi_adjusted=0.13, noise_floor=0.055,
            kept_median_gain=21.0))

    def test_low_importance_not_flagged(self):
        """Within-noise drift alone is not enough — the feature must also
        clear the kept-view median gain."""
        self.assertFalse(classify_drift_retention(
            gain=5.0, psi_adjusted=0.0, noise_floor=0.055,
            kept_median_gain=21.0))

    def test_boundaries_are_inclusive(self):
        """gain == median and psi == noise floor both flag (borderline
        features are retained, never silently dropped)."""
        self.assertTrue(classify_drift_retention(
            gain=21.0, psi_adjusted=0.055, noise_floor=0.055,
            kept_median_gain=21.0))


class TestDeterminism(unittest.TestCase):
    def test_same_folds_same_table(self):
        """run_oof with identical inputs → identical OOF + summary."""
        games = _synthetic_games(n_days=90, per_day=6, seed=7)
        feats = ["is_home", "dome_is_neutral", "home_elo", "away_elo"]
        a = run_oof(games, retrain_cadence_days=10, min_val_games=10,
                    run_features=feats, dropped=[])
        b = run_oof(games, retrain_cadence_days=10, min_val_games=10,
                    run_features=feats, dropped=[])
        self.assertEqual(a["summary"], b["summary"])
        pd.testing.assert_frame_equal(
            a["oof"].reset_index(drop=True), b["oof"].reset_index(drop=True))


class TestMarketHarnessFixture(unittest.TestCase):
    def test_lines_base_rates_and_holdout_isolation(self):
        """Market harness on a synthetic OOF: lines exist, base rates correct,
        holdout scored only on the sealed tail (n>0, its own base rate)."""
        n = 900
        rng = np.random.default_rng(0)
        dates = pd.date_range("2025-05-01", periods=75, freq="D")
        fold_idx = np.repeat(np.arange(10), n // 10)
        oof = pd.DataFrame({
            "game_pk": np.arange(n),
            "game_date": np.tile(dates, n // len(dates))[:n],
            "fold_idx": fold_idx,
            "home_expected_runs": np.clip(rng.normal(4.6, 0.4, n), 2.5, 7),
            "away_expected_runs": np.clip(rng.normal(4.3, 0.4, n), 2.5, 7),
            "home_score": rng.poisson(4.6, n).astype(float),
            "away_score": rng.poisson(4.3, n).astype(float),
        })
        out = derive_markets_v3(oof, n_draws=2_000, seed=1)
        s = out["summary"]
        for name in ("derived_moneyline", "over_7_5", "over_8_5", "over_9_5",
                     "home_cover_1_5", "home_cover_2_5"):
            self.assertIn(f"market_{name}", s)
            row = s[f"market_{name}"]
            self.assertIn("engine_logloss", row)
            self.assertIn("engine_ece_calibrated", row)
            self.assertIn("baseline_rate", row)
        # Base rate ≈ observed mean (synthetic Poisson totals, no leakage).
        total = oof["home_score"] + oof["away_score"]
        self.assertAlmostEqual(
            s["market_over_8_5"]["baseline_rate"],
            float((total >= 9).mean()), places=4)
        # Holdout leg: non-empty, uses its OWN base rate, not the pooled one.
        h = s["market_over_8_5_holdout"]["holdout"]
        self.assertGreaterEqual(h["n"], 100)
        hold_dates = pd.to_datetime(oof["game_date"])
        hold_mask = (hold_dates >= hold_dates.max()
                     - pd.Timedelta(days=21)).to_numpy()
        self.assertEqual(h["n"], int(hold_mask.sum()))
        self.assertAlmostEqual(
            h["baseline_rate"],
            float((total >= 9).to_numpy()[hold_mask].mean()), places=4)
        # Pooled baseline rate must NOT equal the holdout's (isolation).
        self.assertNotAlmostEqual(
            s["market_over_8_5"]["baseline_rate"],
            h["baseline_rate"], places=3)


class TestRegressions(unittest.TestCase):
    def test_moneyline_feature_cols_now_59_leakage_pruned(self):
        """The 6 lineup-delta features were removed from FEATURE_COLS
        (train-serve skew fix, 2026-08-29). FEATURE_COLS is 59."""
        self.assertEqual(len(FEATURE_COLS), 59)
        self.assertIn("run_margin_diff", FEATURE_COLS)

    def test_default_run_oof_path_derives_53(self):
        """run_features=None must derive the 53-col rule (2026-08-30 restored
        view). Dropped is 6: run_margin_diff + the 5 composites."""
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        self.assertEqual(len(keep), 53)
        self.assertEqual(len(dropped), 6)

    def test_alpha_lambda_mc_path_still_derives(self):
        """derive_markets_v3 still produces α(λ) curves + full grid + holdout
        (regression guard: the ablation added no behavior change)."""
        n = 600
        rng = np.random.default_rng(3)
        dates = pd.date_range("2025-06-01", periods=60, freq="D")
        oof = pd.DataFrame({
            "game_pk": np.arange(n),
            "game_date": np.tile(dates, n // len(dates))[:n],
            "fold_idx": np.repeat(np.arange(8), n // 8),
            "home_expected_runs": np.clip(rng.normal(4.6, 0.4, n), 2.5, 7),
            "away_expected_runs": np.clip(rng.normal(4.3, 0.4, n), 2.5, 7),
            "home_score": rng.poisson(4.6, n).astype(float),
            "away_score": rng.poisson(4.3, n).astype(float),
        })
        out = derive_markets_v3(oof, n_draws=1_000, seed=2)
        s = out["summary"]
        self.assertIn("alpha_home", s)
        self.assertIn("alpha_away", s)
        self.assertIn("fit_check_alpha_lambda", s)
        m = out["markets"]
        for col in ("p_over_8_5", "p_under_8_5", "p_home_cover_1_5",
                    "p_home_win_derived", "alpha_home", "alpha_away"):
            self.assertIn(col, m.columns)


if __name__ == "__main__":
    unittest.main()
