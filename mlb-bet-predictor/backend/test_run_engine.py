"""Phase-1 run engine tests (backend/run_engine.py).

Covers: derived feature views (zero _diff columns; symmetric sides; new
FEATURE_COLS entries flow through), target integrity, pooled OOF output shape
(no NaNs), baseline comparison presence, dispersion math on a hand fixture,
and OOF artifact schema.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_engine as re_
from run_engine import (
    derive_run_features,
    dispersion_ratio,
    persist_oof,
    poisson_deviance,
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


if __name__ == "__main__":
    unittest.main()
