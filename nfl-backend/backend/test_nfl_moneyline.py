"""NFL moneyline (``nfl_moneyline.py``) — tests for the model gate.

Pure-function tests (no network) cover:
- Metrics: logloss, AUC, ECE correctness on known arrays.
- Fold generation: every fold satisfies train.gameday < min(val.gameday)
  (the walk-forward leakage assertion).
- Sealed isolation: season 2025 never appears in any pre-sealed train set.
- Ensemble construction: the 5-member trainer (XGB/LGB/Logistic/RF/MLP),
  per-member degradation, blend weights, train-only median imputation, and
  predict-time UNK clamping for unseen teams.
- Adaptive blend weights: sum to exactly 1.0 with floor/cap respected.
- Baselines: constant home-edge and elo-only logistic produce valid outputs.
- Gate logic: ADOPT/NO-ADOPT decision rules, including the pooled-gain /
  sealed-loss inversion path.
- games[] -> frontend adapter mapping: a record's per-game list adapts onto
  the shared card frame; a blocked (not adopted) record yields an empty
  schema'd frame, never a crash.

Artifact tests read the real ``data_delivery/nfl_game_level_features.csv``
when present and confirm the frame covers the train + sealed seasons.
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
    UNK_TEAM_ID,
    _member_weights,
    adopt_decision,
    auc,
    build_games_list,
    clip_p,
    compute_adaptive_weights,
    ece,
    ensemble_predict,
    format_table,
    generate_weekly_folds,
    logloss,
    platt_fit,
    platt_predict,
    train_ensemble,
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

    def test_gate_tables_contain_no_market_arm(self):
        """Market-independence policy: the gate's comparison table carries
        exactly the four model/baseline arms and NO market-derived arm."""
        arms = {
            "constant_home_edge": {"logloss": 0.6909, "auc": 0.5000},
            "elo_logistic": {"logloss": 0.6538, "auc": 0.6687},
            "model_raw": {"logloss": 0.6142, "auc": 0.7214},
            "model_platt": {"logloss": 0.6246, "auc": 0.7214, "ece": 0.0766},
        }
        self.assertNotIn("market_line", arms)
        out = format_table("sealed_2025", arms)
        self.assertIn("constant_home_edge", out)
        self.assertIn("elo_logistic", out)
        self.assertIn("model_raw", out)
        self.assertIn("model_platt", out)
        self.assertNotIn("market", out.lower())


# ---------------------------------------------------------------------------
# Ensemble construction (Part 1)
# ---------------------------------------------------------------------------
class TestEnsembleConstruction(unittest.TestCase):
    """5-member trainer + predictor on a synthetic frame (no network)."""

    def _split(self):
        df = _synth_fold_frame()  # 8 games/week x 18 weeks x 6 seasons
        df = df[_valid_cols(df)].copy()
        tr = df[df["season"] <= 2021]
        va = df[df["season"] == 2022]
        return tr, va

    def test_five_members_train_with_bundle(self):
        tr, va = self._split()
        models, mets = train_ensemble(tr, va, features=V1_FEATURES)
        for m in ("xgboost", "lightgbm", "logistic", "randomforest", "mlp"):
            self.assertIn(m, models, f"{m} member should train")
        for k in ("scaler", "impute_median", "categorical_vocab"):
            self.assertIn(k, models, f"bundle must carry {k}")
        # val metrics present (fit-only refits return {})
        self.assertIn("auc", mets)
        self.assertIn("logloss", mets)

    def test_fit_only_refit_returns_empty_metrics(self):
        tr, _ = self._split()
        models, mets = train_ensemble(tr, None, features=V1_FEATURES)
        self.assertEqual(mets, {})
        self.assertIn("xgboost", models)

    def test_score_sealed_member_table(self):
        """Per-member sealed scoring returns full metrics per member and is
        empty-safe (the ablation harness's member-level read)."""
        import numpy as np
        from nfl_moneyline import _score_member_table
        y = np.array([1, 0, 1, 0, 1, 0])
        members = {
            "xgboost": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
            "mlp": np.array([0.5] * 6),
        }
        t = _score_member_table(y, members)
        self.assertEqual(set(t), {"xgboost", "mlp"})
        for key in ("logloss", "auc", "ece", "brier"):
            self.assertIn(key, t["xgboost"])
        self.assertAlmostEqual(t["mlp"]["auc"], 0.5, places=6)
        # wrong-length member skipped, empty members -> empty table
        t2 = _score_member_table(y, {"xgboost": np.array([0.9, 0.1])})
        self.assertEqual(t2, {})
        self.assertEqual(_score_member_table(y, {}), {})

    def test_blend_probs_in_unit_interval_weights_sum_to_one(self):
        tr, va = self._split()
        models, _ = train_ensemble(tr, va, features=V1_FEATURES)
        blend, members, weights = ensemble_predict(models, va, features=V1_FEATURES)
        self.assertTrue(np.all(blend >= 0.0) and np.all(blend <= 1.0))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertEqual(set(members.keys()), set(weights.keys()))
        for name, p in members.items():
            self.assertEqual(len(p), len(va))

    def test_unseen_team_clamped_to_unk(self):
        """A predict-time team never seen in training must route to the
        reserved UNK category (no 'category not in training set' crash)."""
        tr, va = self._split()
        models, _ = train_ensemble(tr, va, features=V1_FEATURES)
        probe = va.head(3).copy()
        probe["home_team"] = "ZZ"
        probe["away_team"] = "Q7"
        blend, members, _ = ensemble_predict(models, probe, features=V1_FEATURES)
        self.assertEqual(len(blend), 3)
        self.assertTrue(np.all(np.isfinite(blend)))

    def test_member_weights_renormalize_over_present_members(self):
        w = _member_weights(["xgboost", "lightgbm", "logistic"])
        self.assertEqual(set(w.keys()), {"xgboost", "lightgbm", "logistic"})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertGreater(w["logistic"], w["xgboost"])

    def test_adaptive_weights_sum_to_one_with_floor_and_cap(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 200).astype(float)
        oof = {
            "xgboost": rng.uniform(0.2, 0.9, 200).tolist(),
            "lightgbm": rng.uniform(0.2, 0.9, 200).tolist(),
            "logistic": rng.uniform(0.2, 0.9, 200).tolist(),
            "randomforest": rng.uniform(0.2, 0.9, 200).tolist(),
            "mlp": rng.uniform(0.2, 0.9, 200).tolist(),
        }
        w = compute_adaptive_weights(oof, y)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=4)
        for v in w.values():
            self.assertGreaterEqual(v, 0.04)   # floor, rounding tolerance
            self.assertLessEqual(v, 0.46)      # cap, rounding tolerance

    def test_imputation_medians_come_from_train_only(self):
        """The bundle's impute_median must equal the TRAIN column medians
        (a val-median leak would shift the stored medians)."""
        tr, va = self._split()
        # Give the val set a DIFFERENT ypp_diff distribution (shift +10) and
        # NaN rows in both; the stored medians must track TRAIN, never val.
        tr2 = tr.copy(); va2 = va.copy()
        tr2.loc[tr2.index[:10], "ypp_diff"] = np.nan
        va2["ypp_diff"] = va2["ypp_diff"] + 10.0
        va2.loc[va2.index[:10], "ypp_diff"] = np.nan
        models, _ = train_ensemble(tr2, va2, features=V1_FEATURES)
        med = models["impute_median"]
        i = V1_FEATURES.index("ypp_diff")
        train_med = np.nanmedian(tr2["ypp_diff"].to_numpy(dtype=float))
        val_med = np.nanmedian(va2["ypp_diff"].to_numpy(dtype=float))
        self.assertAlmostEqual(med[i], train_med, places=4)
        self.assertNotAlmostEqual(med[i], val_med, places=1)


def _valid_cols(df: pd.DataFrame) -> pd.Series:
    return df[V1_FEATURES + ["home_win"]].notna().all(axis=1)


# ---------------------------------------------------------------------------
# games[] -> adapter mapping (Part 3)
# ---------------------------------------------------------------------------
class TestGamesAdapterMapping(unittest.TestCase):
    """A record's per-game list must adapt onto the shared card frame exactly
    like the existing 20260830 reference (contract: never edit the adapter)."""

    @staticmethod
    def _record(with_games: bool = True) -> dict:
        rec = {"created_utc": "2026-08-30T00:00:00Z",
               "verdict": {"adopt": with_games}}
        if with_games:
            rec["games"] = [{
                "game_id": "2026_01_NE_SEA", "game_date": "2026-09-09",
                "home_team": "SEA", "away_team": "NE",
                "home_win_prob": 0.7132, "away_win_prob": 0.2868,
                "game_status": "pre", "start_time_utc": "2026-09-10T00:20:00Z",
                "venue": "Lumen Field", "model_pick": "SEA",
                "home_record": "86-55", "away_record": "76-66",
            }]
        else:
            rec["predictions"] = {"status": "blocked (not adopted)"}
        return rec

    @staticmethod
    def _adapter():
        _frontend = Path(__file__).resolve().parents[2] / "frontend"
        if str(_frontend) not in sys.path:
            sys.path.insert(0, str(_frontend))
        from utils import nfl_moneyline_to_frame
        return nfl_moneyline_to_frame

    def test_games_adapt_to_shared_card_frame(self):
        adapter = self._adapter()
        frame = adapter(self._record(with_games=True))
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["game_id"], "2026_01_NE_SEA")
        self.assertEqual(row["home_team"], "SEA")
        self.assertEqual(row["away_team"], "NE")
        self.assertAlmostEqual(row["home_win_prob_model"], 0.7132, places=4)
        self.assertAlmostEqual(row["away_win_prob_model"], 0.2868, places=4)
        self.assertEqual(row["model_pick"], "SEA")
        self.assertEqual(row["game_status"], "Scheduled")  # 'pre' -> Scheduled
        self.assertEqual(row["start_time_utc"], "2026-09-10T00:20:00Z")
        self.assertEqual(row["venue"], "Lumen Field")
        self.assertEqual(row["home_record"], "86-55")

    def test_blocked_record_adapts_to_empty_schema_frame(self):
        adapter = self._adapter()
        frame = adapter(self._record(with_games=False))
        self.assertEqual(len(frame), 0)  # never a crash, never fabricated rows
        self.assertIn("home_win_prob_model", frame.columns)
        self.assertIn("game_status", frame.columns)

    def test_build_games_list_emits_reference_shape(self):
        tr = _synth_fold_frame(seasons=[2019, 2020, 2021])
        va = _synth_fold_frame(seasons=[2022]).head(4)
        models, _ = train_ensemble(tr, va, features=V1_FEATURES)
        sf = va.copy()
        sf["stadium"] = "Lumen Field"
        sf["gametime"] = "20:20"
        sf["home_record"] = "86-55"
        sf["away_record"] = "76-66"
        lr = platt_fit(np.linspace(0.3, 0.7, 40), np.tile([1, 0], 20))
        games = build_games_list(sf, models, lr, V1_FEATURES)
        self.assertEqual(len(games), 4)
        keys = {"game_id", "game_date", "home_team", "away_team",
                "home_win_prob", "away_win_prob", "home_score", "away_score",
                "game_status", "start_time_utc", "venue", "model_pick",
                "home_record", "away_record"}
        self.assertEqual(set(games[0].keys()), keys)
        self.assertEqual(games[0]["game_status"], "pre")
        self.assertAlmostEqual(games[0]["home_win_prob"] +
                               games[0]["away_win_prob"], 1.0, places=4)
        # market-independence: games[] carries NO line/market fields.
        for key in ("spread_line", "total_line", "home_moneyline",
                    "away_moneyline", "market_home_implied"):
            self.assertNotIn(key, games[0])


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


# ---------------------------------------------------------------------------
# Default-path regression: pull_and_run with seasons=None must not raise
# UnboundLocalError on DEFAULT_SEASONS (the from-import makes the name
# function-local, so it must be bound before first use).
# ---------------------------------------------------------------------------
class TestDefaultSeasonsPath(unittest.TestCase):
    def test_default_seasons_resolvable_without_calling_full_run(self):
        # Regression for UnboundLocalError at pull_and_run: the name must be
        # importable and bound BEFORE first use on the seasons=None path.
        import nfl_features
        import nfl_moneyline
        self.assertTrue(hasattr(nfl_features, "DEFAULT_SEASONS"))
        self.assertTrue(hasattr(nfl_moneyline, "DEFAULT_SEASONS"))
        self.assertTrue(len(nfl_features.DEFAULT_SEASONS) > 0)

    def test_default_window_matches_nfl_features(self):
        # The module-level export must match the real feed window (warmup
        # 2018 + core 2019-2025 incl. the sealed 2025), or the default path
        # would silently break the sealed gate.
        import nfl_features
        import nfl_moneyline
        self.assertEqual(nfl_moneyline.DEFAULT_SEASONS,
                         nfl_features.DEFAULT_SEASONS)

    def test_explicit_seasons_win_over_default(self):
        # The seasons-or-default expression must prefer explicit seasons.
        seasons = [2021, 2022]
        feed_seasons = seasons or [2019, 2020]
        self.assertEqual(feed_seasons, seasons)


class TestPullAndRunSeasons(unittest.TestCase):
    """Regression: pull_and_run's ``feed_seasons`` binding. The default path
    (seasons=None) previously raised UnboundLocalError because the
    ``from nfl_features import DEFAULT_SEASONS`` below the use made the name
    function-local for the whole body (2026-09-01 fix: import moved above
    the use)."""

    def _csv(self) -> str:
        import os
        import tempfile
        feats = _synth_fold_frame([2024, 2025], n_games_per_week=2)
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        feats.to_csv(path, index=False)
        return path

    def _call(self, seasons=None):
        from unittest import mock

        import nfl_moneyline as nm
        path = self._csv()
        try:
            stub = {"_deployed": {"features": ["elo_diff"]}}
            with mock.patch.object(nm, "run_walk_forward",
                                   return_value=stub) as rwf, \
                    mock.patch.object(nm, "_print_report", lambda r: None):
                out = nm.pull_and_run(features_csv=Path(path),
                                      write_record=False,
                                      seasons=seasons)
            self.assertEqual(out, stub)
            rwf.assert_called_once()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_default_seasons_does_not_raise(self):
        self._call(seasons=None)          # the UnboundLocalError path

    def test_explicit_seasons_wins(self):
        self._call(seasons=[2022])        # explicit must stay honored, no crash

    def test_default_seasons_bound_before_use(self):
        """Structural guard for the UnboundLocalError shape: pull_and_run must
        NOT from-import DEFAULT_SEASONS inside its body (module-level binding
        only, matching nfl_features), so the default seasons=None path can
        never shadow-unbind the name."""
        import inspect

        import nfl_features
        import nfl_moneyline as nm
        self.assertEqual(nm.DEFAULT_SEASONS, nfl_features.DEFAULT_SEASONS)
        src = inspect.getsource(nm.pull_and_run)
        shadowing = [ln for ln in src.splitlines()
                     if "from nfl_features import" in ln
                     and "DEFAULT_SEASONS" in ln]
        self.assertEqual(shadowing, [])
        self.assertIn("seasons or DEFAULT_SEASONS", src)


if __name__ == "__main__":
    unittest.main()
