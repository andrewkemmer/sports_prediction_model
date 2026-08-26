"""Run-engine-native keep-list ablation tests.

Covers the 2026-08 keep-list decision:
- derive_run_features routing matches the adopted (DON'T SHIP) outcome, and the
  ship-outcome variant is exactly kept + 25 diffs (54 cols, incl. the shipped
  run_margin_diff) — both sides of the decision are pinned so a future rule
  change is a deliberate, tested act.
- Same folds → identical table (run_oof determinism).
- Market-level scoring harness on a fixture: reference lines exist, base rates
  correct, ECE-cal computed with no holdout leakage.
- Regressions: moneyline FEATURE_COLS is 65 (run_margin_diff shipped
  2026-08-26, excluded from the run view by the *_diff rule — the 29-col
  keep-list is unchanged); run_oof default call path (no explicit feature
  list) unchanged; α(λ)/MC market path still derives.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import (
    RUN_EXTRA_EXCLUSIONS,
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
    def test_dont_ship_outcome_exact_sets(self):
        """Current rule (adopted: DO NOT SHIP) — exact kept/dropped sets."""
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        diffs = [d for d in dropped if d.endswith("_diff")]
        composites = [d for d in dropped
                      if not d.endswith("_diff")
                      and "lineup_actual" not in d
                      and "lineup_rest_count" not in d]
        lineup = [d for d in dropped if "lineup_actual" in d
                  or "lineup_rest_count" in d]
        self.assertEqual(len(keep), 29, "kept view must stay 29 cols")
        # 25 diffs = the original 24 matchup-gap diffs + run_margin_diff
        # (shipped 2026-08-26, moneyline-only, excluded by the same *_diff
        # rule) — the kept view is byte-identical to pre-margin.
        self.assertEqual(len(dropped), 36)
        self.assertEqual(len(diffs), 25)
        self.assertEqual(len(composites), 5)
        self.assertEqual(len(lineup), 6)
        # No _diff in the kept view except the sanctioned survivor.
        for f in keep:
            if f.endswith("_diff"):
                self.assertEqual(f, "park_factor_slug_diff")
        self.assertIn("park_factor_slug_diff", keep)
        # The 5 engineered composites are excluded by name.
        for f in ("lineup_handedness_matchup_advantage", "bullpen_meltdown_risk",
                  "pitcher_regression_indicator", "lineup_depth_multiplier",
                  "ace_efficiency_factor"):
            self.assertNotIn(f, keep)
        # The 6 moneyline-scoped lineup columns stay out of the run view.
        for f in ("lineup_actual_woba_delta_home", "lineup_actual_woba_delta_away",
                  "lineup_actual_top3_delta_home", "lineup_actual_top3_delta_away",
                  "lineup_rest_count_home", "lineup_rest_count_away"):
            self.assertNotIn(f, keep)
            self.assertIn(f, RUN_EXTRA_EXCLUSIONS)

    def test_ship_outcome_exact_sets(self):
        """Hypothetical ship rule (kept + 25 diffs incl. run_margin_diff) —
        54 cols, no lineup."""
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        diffs = [d for d in dropped if d.endswith("_diff")]
        ship = list(keep) + diffs
        self.assertEqual(len(ship), 54)
        self.assertEqual(len(set(ship)), 54, "ship variant must not duplicate")
        for f in diffs:
            self.assertTrue(f.endswith("_diff"))
            self.assertNotEqual(f, "park_factor_slug_diff")
        for f in ship:
            self.assertNotIn("lineup_actual", f)
            self.assertNotIn("lineup_rest_count", f)


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
    def test_moneyline_feature_cols_now_65_with_margin(self):
        """The run-margin feature SHIPPED (2026-08-26, sealed-holdout gate
        passed) — FEATURE_COLS grew to 65 and carries run_margin_diff."""
        self.assertEqual(len(FEATURE_COLS), 65)
        self.assertIn("run_margin_diff", FEATURE_COLS)

    def test_default_run_oof_path_unchanged(self):
        """run_features=None must still derive the 29-col rule (backward
        compatible hook — the ablation arm A path). Dropped is 36 now: the
        original 35 + run_margin_diff, which the *_diff rule excludes — the
        KEPT view is byte-identical to pre-margin."""
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        self.assertEqual(len(keep), 29)
        self.assertEqual(len(dropped), 36)

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
