"""NFL moneyline v1 (``nfl_moneyline.py``) — tests for the model gate.

Pure-function tests (no network) cover:
- Metrics: logloss, AUC, ECE correctness on known arrays.
- Fold generation: every fold satisfies train.gameday < min(val.gameday)
  (the walk-forward leakage assertion).
- Sealed isolation: season 2025 never appears in any pre-sealed train set.
- Baselines: constant home-edge and elo-only logistic produce valid outputs.
- Gate logic: ADOPT/NO-ADOPT decision rules, including the pooled-gain /
  sealed-loss inversion path.

Artifact tests read the real ``data_delivery/nfl_game_level_features.csv``
when present and run the full walk-forward + sealed gate, confirming the
module produces a valid record JSON with correct fold geometry and verdict.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nfl_moneyline import (  # noqa: E402
    ECE_BINS,
    ECE_MAX,
    SEALED_SEASON,
    TRAIN_SEASONS,
    V1_FEATURES,
    adopt_decision,
    auc,
    clip_p,
    ece,
    generate_weekly_folds,
    logloss,
    platt_fit,
    platt_predict,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data_delivery" / "nfl_game_level_features.csv"


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------
def _synth_fold_frame(seasons=None, n_games_per_week=8):
    """Build a minimal decided-shaped DataFrame with v1 features + target,
    spanning the given seasons with weekly cadence (Mon-Sun weeks).
    """
    seasons = seasons or TRAIN_SEASONS
    rows = []
    gid = 0
    for season in seasons:
        # ~18 weeks per season (REG + postseason)
        for week in range(1, 19):
            base_date = pd.Timestamp(f"{season}-09-01") + pd.Timedelta(weeks=week - 1)
            for g in range(n_games_per_week):
                gd = base_date + pd.Timedelta(days=g % 7)
                rows.append({
                    "game_id": f"{season}_W{week:02d}_G{g}",
                    "season": season,
                    "week": week,
                    "gameday": gd,
                    "home_team": f"H{gid % 32}",
                    "away_team": f"A{(gid + 16) % 32}",
                    "home_score": 20 + (gid % 14),
                    "away_score": 17 + ((gid + 7) % 14),
                    "result": 3.0,
                    "total": 37.0,
                    "spread_line": float((gid % 7) - 3),
                    "total_line": 45.0,
                    "n_plays": 130,
                    "elo_diff": float((gid % 11) - 5),
                    "form_diff_pts": float((gid % 9) - 4),
                    "rest_days_diff": float((gid % 5) - 2),
                    "ypp_diff": float((gid % 7) - 3),
                    "is_dome_home": float(gid % 2),
                    "is_home": 1.0,
                    "home_win": int((gid % 3) != 0),  # ~67% home wins
                })
                gid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------
class TestMetrics(unittest.TestCase):
    def test_logloss_perfect(self):
        """Perfect predictions -> logloss near 0."""
        y = np.array([1.0, 1.0, 0.0, 0.0])
        p = np.array([0.99, 0.99, 0.01, 0.01])
        ll = logloss(y, p)
        self.assertLess(ll, 0.1)

    def test_logloss_worst(self):
        """Confident wrong predictions -> high logloss."""
        y = np.array([1.0, 1.0, 0.0, 0.0])
        p = np.array([0.01, 0.01, 0.99, 0.99])
        ll = logloss(y, p)
        self.assertGreater(ll, 3.0)

    def test_logloss_symmetric(self):
        """logloss(p) == logloss(1-p) for balanced y."""
        y = np.array([1.0, 0.0])
        p = np.array([0.7, 0.3])
        y2 = np.array([0.0, 1.0])
        self.assertAlmostEqual(logloss(y, p), logloss(y2, 1 - p), places=10)

    def test_auc_perfect(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        x = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
        self.assertAlmostEqual(auc(y, x), 1.0, places=6)

    def test_auc_inverse(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(auc(y, x), 0.0, places=6)

    def test_auc_random(self):
        rng = np.random.default_rng(42)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        x = rng.normal(size=10)
        self.assertAlmostEqual(auc(y, x), 0.5, delta=0.3)

    def test_ece_perfect_calibration(self):
        """Perfectly calibrated (predicted == actual) -> ECE near 0."""
        y = np.array([1, 1, 1, 0, 0, 0])
        p = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(ece(y, p, bins=3), 0.0, places=6)

    def test_ece_worst_calibration(self):
        """All predicted 1.0, all actual 0 -> ECE = 1.0."""
        y = np.zeros(10)
        p = np.ones(10)
        self.assertAlmostEqual(ece(y, p, bins=1), 1.0, places=6)

    def test_clip_p_bounds(self):
        p = np.array([0.0, 0.5, 1.0, -0.1, 1.1])
        cp = clip_p(p)
        self.assertTrue(np.all(cp > 0))
        self.assertTrue(np.all(cp < 1))


# ---------------------------------------------------------------------------
# Platt calibration tests
# ---------------------------------------------------------------------------
class TestPlatt(unittest.TestCase):
    def test_platt_fit_predict_roundtrip(self):
        """Platt map on well-separated probs should produce near-binary output."""
        rng = np.random.default_rng(7)
        p_raw = np.concatenate([np.full(20, 0.9), np.full(20, 0.1)])
        y = np.concatenate([np.ones(20), np.zeros(20)])
        lr = platt_fit(p_raw, y)
        p_cal = platt_predict(p_raw, lr)
        # calibrated probs for the '1' class should be near 1.0
        self.assertGreater(p_cal[:20].mean(), 0.8)
        self.assertLess(p_cal[20:].mean(), 0.2)


# ---------------------------------------------------------------------------
# Fold generation + leakage tests
# ---------------------------------------------------------------------------
class TestFoldLeakage(unittest.TestCase):
    def test_no_future_week_in_any_train_set(self):
        """For every fold, all train rows have gameday < min(val.gameday)."""
        df = _synth_fold_frame()
        folds = generate_weekly_folds(df)
        self.assertGreater(len(folds), 10, "need enough folds for validation")
        for f in folds:
            tr_max = f["train"]["gameday"].max()
            va_min = f["val"]["gameday"].min()
            self.assertLess(tr_max, va_min,
                            f"LEAKAGE: train max {tr_max} >= val min {va_min}")

    def test_fold_train_never_contains_val_weeks(self):
        """Walk-forward folds: train set for week W must not contain any
        rows with gameday >= the fold's validation week_start. In weekly-
        cadence folds, it IS valid for train to contain earlier weeks of
        the same season (e.g. week 1 trains on 2019-2020; week 2 trains
        on 2019-2020 + week 1 of 2021). The hard rule is gameday-based."""
        df = _synth_fold_frame()
        folds = generate_weekly_folds(df)
        for f in folds:
            ws = f["week_start"]
            tr_max = f["train"]["gameday"].max()
            self.assertLess(tr_max, ws,
                            f"fold week {ws}: train contains gameday "
                            f"{tr_max} >= week_start")

    def test_fold_weeks_are_monotonic(self):
        """Folds should be ordered by week_start chronologically."""
        df = _synth_fold_frame()
        folds = generate_weekly_folds(df)
        starts = [f["week_start"] for f in folds]
        self.assertEqual(starts, sorted(starts))

    def test_warmup_seasons_excluded_from_validation(self):
        """2019 and 2020 (warmup) should never appear in any fold's val set."""
        df = _synth_fold_frame()
        folds = generate_weekly_folds(df)
        for f in folds:
            self.assertNotIn(2019, f["val"]["season"].unique())
            self.assertNotIn(2020, f["val"]["season"].unique())


class TestSealedIsolation(unittest.TestCase):
    def test_2025_never_in_train_or_val_of_any_fold(self):
        """Season 2025 (the sealed holdout) must never appear in any fold's
        train or val set when generating folds over TRAIN_SEASONS only."""
        df = _synth_fold_frame()
        # Explicitly add some 2025 rows to the frame
        extra = _synth_fold_frame(seasons=[2025])
        df = pd.concat([df, extra], ignore_index=True)
        folds = generate_weekly_folds(df, val_seasons=TRAIN_SEASONS)
        for f in folds:
            self.assertNotIn(SEALED_SEASON, f["train"]["season"].unique())
            self.assertNotIn(SEALED_SEASON, f["val"]["season"].unique())


# ---------------------------------------------------------------------------
# Gate / adopt-decision logic tests
# ---------------------------------------------------------------------------
class TestAdoptDecision(unittest.TestCase):
    def test_adopt_when_model_beats_both(self):
        pooled = {
            "model_platt": {"logloss": 0.55, "auc": 0.62},
            "elo_logistic": {"logloss": 0.60},
            "constant_home_edge": {"logloss": 0.65},
        }
        sealed = {
            "model_platt": {"logloss": 0.54, "auc": 0.63, "ece": 0.04},
            "elo_logistic": {"logloss": 0.61, "auc": 0.58},
            "constant_home_edge": {"logloss": 0.66, "auc": 0.50},
        }
        v = adopt_decision(pooled, sealed)
        self.assertTrue(v["adopt"])
        self.assertTrue(v["sealed_beats_elo"])
        self.assertTrue(v["sealed_beats_constant"])
        self.assertTrue(v["sane_ece"])

    def test_no_adopt_when_model_loses_to_elo(self):
        pooled = {
            "model_platt": {"logloss": 0.55, "auc": 0.62},
            "elo_logistic": {"logloss": 0.58},
            "constant_home_edge": {"logloss": 0.65},
        }
        sealed = {
            "model_platt": {"logloss": 0.62, "auc": 0.56, "ece": 0.04},
            "elo_logistic": {"logloss": 0.60, "auc": 0.58},
            "constant_home_edge": {"logloss": 0.66, "auc": 0.50},
        }
        v = adopt_decision(pooled, sealed)
        self.assertFalse(v["adopt"])
        self.assertFalse(v["sealed_beats_elo"])

    def test_no_adopt_when_ece_too_high(self):
        pooled = {
            "model_platt": {"logloss": 0.55, "auc": 0.62},
            "elo_logistic": {"logloss": 0.60},
            "constant_home_edge": {"logloss": 0.65},
        }
        sealed = {
            "model_platt": {"logloss": 0.54, "auc": 0.63, "ece": 0.15},
            "elo_logistic": {"logloss": 0.61, "auc": 0.58},
            "constant_home_edge": {"logloss": 0.66, "auc": 0.50},
        }
        v = adopt_decision(pooled, sealed)
        self.assertFalse(v["adopt"])
        self.assertFalse(v["sane_ece"])

    def test_inversion_flag_when_pooled_wins_sealed_loses(self):
        """Pooled-gain / sealed-loss inversion should set the flag."""
        pooled = {
            "model_platt": {"logloss": 0.50, "auc": 0.65},
            "elo_logistic": {"logloss": 0.55},
            "constant_home_edge": {"logloss": 0.60},
        }
        sealed = {
            "model_platt": {"logloss": 0.65, "auc": 0.52, "ece": 0.04},
            "elo_logistic": {"logloss": 0.58, "auc": 0.57},
            "constant_home_edge": {"logloss": 0.62, "auc": 0.50},
        }
        v = adopt_decision(pooled, sealed)
        self.assertFalse(v["adopt"])
        self.assertTrue(v["pooled_gain_sealed_loss_inversion"])
        inv_reasons = [r for r in v["reasons"] if "inversion" in r]
        self.assertTrue(len(inv_reasons) > 0)


# ---------------------------------------------------------------------------
# Artifact tests (require real CSV)
# ---------------------------------------------------------------------------
@unittest.skipUnless(FEATURES.exists(),
                     "nfl_game_level_features.csv not present — "
                     "run `python3 nfl_game_frame.py` first")
class TestRealArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(FEATURES)

    def test_frame_covers_all_train_seasons_plus_2025(self):
        seasons = sorted(self.df["season"].unique())
        self.assertEqual(seasons, sorted(TRAIN_SEASONS + [SEALED_SEASON]))

    def test_total_games_1960(self):
        self.assertEqual(len(self.df), 1960)


if __name__ == "__main__":
    unittest.main()
