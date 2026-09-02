"""Unit tests for the NFL Model & Data Drift Monitor emitter (nfl_monitor.py).

All the emitters are pure (no I/O / Streamlit), so these tests pin the
MLB-shaped ``nfl_model_monitor_*.json`` contract the shared Model Monitor page
renders: TRUE PSI drift rows with OK/WARN/ALERT/INSUFFICIENT statuses, feature
coverage, the rolling 30-day Brier timeline vs a constant-home-edge baseline,
the per-member ensemble rows, version history, and the composed record keys.

Run from nfl-backend/backend:
    PYTHONUTF8=1 python -m unittest test_nfl_monitor -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import nfl_monitor as mon


class PsiTest(unittest.TestCase):
    def test_constant_feature_psi_zero(self):
        base = np.ones(50)
        cur = np.ones(40)
        self.assertEqual(mon.psi_score(base, cur), 0.0)

    def test_same_distribution_psi_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        self.assertAlmostEqual(mon.psi_score(x, x.copy()), 0.0, places=6)

    def test_shifted_distribution_positive_psi(self):
        base = np.linspace(0.0, 1.0, 300)
        cur = np.linspace(0.9, 1.0, 120)   # concentrated high -> shift
        self.assertGreater(mon.psi_score(base, cur), 0.05)

    def test_all_nan_returns_zero(self):
        self.assertEqual(mon.psi_score(np.array([np.nan]), np.array([np.nan])), 0.0)


class DriftStatusTest(unittest.TestCase):
    def test_insufficient_below_sample_floor(self):
        self.assertEqual(mon.drift_status(0.9, n_current=10), "INSUFFICIENT")

    def test_threshold_bands(self):
        self.assertEqual(mon.drift_status(0.01, n_current=60), "OK")
        self.assertEqual(mon.drift_status(0.15, n_current=60), "WARN")
        self.assertEqual(mon.drift_status(0.40, n_current=60), "ALERT")


class FeatureDriftRowsTest(unittest.TestCase):
    def test_rows_shape_and_weight(self):
        feat = pd.DataFrame({
            "gameday": ["2025-09-07"] * 6 + ["2026-09-10"] * 4,
            "elo_diff": [10.0, 20.0, 15.0, 30.0, 25.0, 12.0,
                         18.0, 22.0, 28.0, 24.0],
            "rest_days_diff": [0.0, 1.0, 0.0, 2.0, 0.0, 1.0,
                               1.0, 0.0, 2.0, 0.0],
        })
        base = feat["gameday"].astype(str) < "2026-01-01"
        cur = ~base
        rows = mon.feature_drift_rows(
            feat, ["elo_diff", "rest_days_diff"], base.to_numpy(), cur.to_numpy(),
            weight={"elo_diff": 62.0, "rest_days_diff": 38.0})
        self.assertEqual([r["feature"] for r in rows],
                         ["elo_diff", "rest_days_diff"])
        for r in rows:
            self.assertIn("current_mean", r)
            self.assertIn("baseline_mean", r)
            self.assertIn("psi", r)
            self.assertIn("status", r)
            self.assertEqual(r["n_current"], int(cur.sum()))
        # weight_pct carries MLB's PERCENT semantics (dict values sum to 100)
        # rounded to 3 decimals — the raw value, like MLB's own row field.
        self.assertEqual(rows[0]["weight_pct"], 62.0)
        self.assertEqual(rows[1]["weight_pct"], 38.0)

    def test_no_weight_omits_weight_pct(self):
        feat = pd.DataFrame({"gameday": ["2025-09-07"] * 3 + ["2026-09-10"] * 2,
                             "elo_diff": [1.0, 2.0, 3.0, 4.0, 5.0]})
        base = feat["gameday"].astype(str) < "2026-01-01"
        cur = ~base
        rows = mon.feature_drift_rows(feat, ["elo_diff"], base.to_numpy(),
                                      cur.to_numpy(), weight=None)
        self.assertIsNone(rows[0]["weight_pct"])

    def test_missing_column_skipped(self):
        feat = pd.DataFrame({"gameday": ["2025-01-01", "2026-01-01"],
                             "elo_diff": [1.0, 2.0]})
        base = feat["gameday"].astype(str) < "2026-01-01"
        cur = ~base
        rows = mon.feature_drift_rows(feat, ["nope"], base.to_numpy(),
                                      cur.to_numpy())
        self.assertEqual(rows, [])


class WeightImportanceTest(unittest.TestCase):
    """MODEL WEIGHT column — the backend must replicate MLB's algorithm, not
    approximate it: same inputs -> same weights (see the embedded MLB
    reference in test_mlb_reference_equivalence)."""

    class _Tree:
        def __init__(self, imp):
            self.feature_importances_ = np.asarray(imp, dtype=float)

    class _Lin:
        def __init__(self, coef):
            self.coef_ = np.asarray(coef, dtype=float).reshape(1, -1)

    class _Mlp:
        """No feature_importances_/coef_ — the MLP contributes nothing."""

    @staticmethod
    def _mlb_reference(models, feature_cols, adaptive_weights=None):
        """VERBATIM mlb-backend/backend/training.py::feature_importance_weights
        (+ its _member_weights helper), parameterized by feature_cols so the
        shared synthetic case can drive both implementations."""
        import numpy as _np

        ens_weights = {"xgboost": 0.25, "lightgbm": 0.25, "logistic": 0.30,
                       "randomforest": 0.10, "mlp": 0.10}
        floor = 0.05

        def _member_weights(member_names):
            source = adaptive_weights or ens_weights
            raw = {n: float(source.get(n, 0.0)) for n in member_names}
            for n in [n for n, v in raw.items() if v <= 0]:
                prior = float(ens_weights.get(n, 0.0))
                if prior > 0:
                    raw[n] = min(prior, floor * 2)
            total = sum(raw.values())
            if total <= 0:
                return {n: 1.0 / max(len(member_names), 1)
                        for n in member_names}
            return {n: v / total for n, v in raw.items()}

        members = {n: m for n, m in models.items()
                   if n not in ("scaler", "impute_median", "categorical_vocab")}
        eff = _member_weights(list(members.keys()))
        raw = {n: float(eff.get(n, 0.0)) for n in members}
        total = sum(raw.values())
        if total <= 0:
            raw = {n: 1.0 / len(members) for n in members}
            total = 1.0
        nfc = len(feature_cols)
        agg = _np.zeros(nfc)
        contributed = False
        for name, model in members.items():
            if hasattr(model, "feature_importances_"):
                imp = _np.asarray(model.feature_importances_,
                                  dtype=float).ravel()
            elif hasattr(model, "coef_"):
                imp = _np.abs(_np.asarray(model.coef_,
                                          dtype=float)).ravel()
            else:
                continue
            if len(imp) >= nfc:
                imp = imp[:nfc]
            if len(imp) != nfc or imp.sum() <= 0:
                continue
            agg += (raw[name] / total) * (imp / imp.sum())
            contributed = True
        if not contributed or agg.sum() <= 0:
            return None
        return {f: round(float(w), 4)
                for f, w in zip(feature_cols, agg / agg.sum() * 100.0)}

    def _shared_case(self):
        """The shared synthetic case: 4 features; trees carry the fit-time
        team-ID tail (2 extra values) that must be trimmed, logistic carries
        |coef|, the MLP exposes nothing."""
        cols = ["feature_a", "feature_b", "feature_c", "feature_d"]
        models = {
            "xgboost": self._Tree([3.0, 1.0, 0.5, 0.2, 9.9, 9.9]),
            "lightgbm": self._Tree([2.5, 2.0, 0.4, 0.1, 9.9, 9.9]),
            "randomforest": self._Tree([1.0, 0.8, 0.6, 0.4, 9.9, 9.9]),
            "logistic": self._Lin([0.9, -0.3, 0.2, -0.05]),
            "mlp": self._Mlp(),
            "scaler": object(), "impute_median": object(),
            "categorical_vocab": {},
        }
        adaptive = {"xgboost": 0.45, "lightgbm": 0.30, "logistic": 0.15,
                    "randomforest": 0.10, "mlp": 0.0}
        return cols, models, adaptive

    def test_mlb_reference_equivalence(self):
        """Same inputs -> same weights: NFL's implementation must be
        numerically IDENTICAL to MLB's reference on a shared synthetic case."""
        cols, models, adaptive = self._shared_case()
        nfl = mon.feature_importance_weights(models, cols,
                                             adaptive_weights=adaptive)
        mlb = self._mlb_reference(models, cols, adaptive)
        self.assertIsNotNone(nfl)
        self.assertEqual(nfl, mlb)
        self.assertAlmostEqual(sum(nfl.values()), 100.0, places=3)

    def test_mlp_skipped_members_renormalize(self):
        """MLP (no importances surface) contributes nothing; the remaining
        members renormalize to 100 — MLB's exact fallback."""
        cols, models, adaptive = self._shared_case()
        w = mon.feature_importance_weights(models, cols,
                                           adaptive_weights=adaptive)
        self.assertEqual(set(w), set(cols))          # only model columns
        # Per-value rounding to 4dp (MLB's convention) can leave the sum at
        # 100.00±0.01 — assert the ~100 contract, not exact closure.
        self.assertAlmostEqual(sum(w.values()), 100.0, places=2)
        # Deterministic on a repeat call (stable, no row-order dependence).
        self.assertEqual(w, mon.feature_importance_weights(
            models, cols, adaptive_weights=adaptive))

    def test_zeroed_member_gets_static_prior(self):
        """A member with zero adaptive weight still contributes its capped
        static prior (MLB's _member_weights zeroed-member rule)."""
        cols, models, adaptive = self._shared_case()
        adaptive_no_mlp = dict(adaptive)
        adaptive_no_mlp.pop("mlp", None)
        w = mon.feature_importance_weights(models, cols,
                                           adaptive_weights=adaptive_no_mlp)
        self.assertEqual(w, self._mlb_reference(models, cols,
                                                adaptive_no_mlp))

    def test_none_when_no_member_exposes_importances(self):
        cols = ["a", "b"]
        models = {"xgboost": self._Mlp(), "mlp": self._Mlp()}
        self.assertIsNone(mon.feature_importance_weights(models, cols))

    def test_static_fallback_without_adaptive_weights(self):
        """No adaptive weights -> static ENSEMBLE_WEIGHTS priors, still
        matching the MLB reference on the same inputs."""
        cols, models, _ = self._shared_case()
        self.assertEqual(
            mon.feature_importance_weights(models, cols),
            self._mlb_reference(models, cols))


class MetadataTest(unittest.TestCase):
    def test_metadata_map_shapes(self):
        meta = mon.feature_metadata_map(
            ["elo_diff", "div_game"],
            descriptions={"elo_diff": "rating gap", "div_game": "flag"})
        self.assertEqual(set(meta), {"elo_diff", "div_game"})
        for c in meta:
            self.assertIn("tooltip", meta[c])
            self.assertIn("definition", meta[c])
            self.assertIn("source", meta[c])
        self.assertIn("rating gap", meta["elo_diff"]["tooltip"])

    def test_unknown_feature_falls_back_to_name(self):
        meta = mon.feature_metadata_map(["mystery_col"], descriptions={})
        self.assertIn("mystery_col", meta["mystery_col"]["tooltip"])

    def test_no_stale_gate_text_default(self):
        meta = mon.feature_metadata_map(["elo_diff"], descriptions=None)
        self.assertNotIn("admission gate", meta["elo_diff"]["tooltip"].lower())
        self.assertNotIn("admission gate", meta["elo_diff"]["definition"].lower())


class FeatureCoverageTest(unittest.TestCase):
    def test_status_bands(self):
        feat = pd.DataFrame({
            "gameday": ["2025-09-07"] * 10,
            "ok": [1.0] * 10,
            "low": list(range(8)) + [np.nan, np.nan],   # 80% non-null -> just OK floor? not
            "starved": [1.0, np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan, np.nan],  # 10%
        })
        rows = {r["feature"]: r for r in mon.feature_coverage_rows(
            feat, ["ok", "low", "starved"])}
        self.assertEqual(rows["ok"]["status"], "OK")
        self.assertEqual(rows["low"]["status"], "OK")       # 80% hits the floor
        self.assertEqual(rows["starved"]["status"], "STARVED")
        self.assertEqual(rows["ok"]["n_default_zero"], 0)
        self.assertEqual(rows["ok"]["pct_measured"], 100.0)


class RollingBrierTest(unittest.TestCase):
    def _hist(self, probs, winners_home):
        n = len(probs)
        hist = pd.DataFrame({
            "game_date": [f"2026-09-{i % 20 + 1:02d}" for i in range(n)],
            "home_team": [f"H{i % 8:02d}" for i in range(n)],
            "away_team": [f"A{i % 8:02d}" for i in range(n)],
            "home_win_prob_model": probs,
            "actual_winner": [("H%02d" % (i % 8)) if w else ("A%02d" % (i % 8))
                              for i, w in enumerate(winners_home)],
        })
        return hist

    def test_perfect_forecast_brier_zero(self):
        hist = self._hist([1.0, 1.0, 1.0], [True, True, True])
        rows, _meta, base, _label = mon.rolling_brier_rows(hist, window_days=30)
        self.assertEqual(rows, [])  # < min_games_per_day (2) for any single day

    def test_rows_min_games_gate(self):
        # 40 decided games spread 2/day over 20 dates (01..20), within the
        # 30-day window; each day meets min_games_per_day=2 -> all included.
        hist = self._hist([0.6] * 40, [True] * 30 + [False] * 10)
        rows, meta, base, label = mon.rolling_brier_rows(hist, window_days=30,
                                                         min_games_per_day=2)
        self.assertEqual(len(rows), 20)          # dates 01..20
        self.assertEqual(meta["excluded_sparse_days"], 0)
        self.assertIsNotNone(base)
        self.assertEqual(label, "Constant home-edge")
        # Each emitted row carries a real Brier on the standard scale.
        self.assertTrue(all(0.0 <= r["brier"] <= 1.0 for r in rows))

    def test_constant_home_edge_baseline_expected(self):
        # All 40 games home wins with p=0.9 predicted: model Brier tiny;
        # baseline = constant home rate forecast = predict p=1 (home always):
        # mean(2*(1 - y)^2) with y all-1 -> 0.
        hist = self._hist([0.9] * 40, [True] * 40)
        rows, _meta, base, _label = mon.rolling_brier_rows(hist, window_days=30)
        self.assertTrue(rows)
        self.assertAlmostEqual(base, 0.0, places=6)
        self.assertEqual(_label, "Constant home-edge")


class EnsembleAndVersionTest(unittest.TestCase):
    def test_ensemble_rows_weights_and_members(self):
        result = {
            "adaptive_weights": {"xgboost": 0.45, "lightgbm": 0.0,
                                 "logistic": 0.0, "randomforest": 0.0,
                                 "mlp": 0.0},
            "members": {"xgboost": {"auc": 0.69, "brier": 0.22, "logloss": 0.63}},
            "verdict": {"adopt": True},
            "trained_at": "2026-08-31T01:00:00.000000Z",
        }
        rows = mon.ensemble_rows(result, history_len=1107)
        by_name = {r["name"]: r for r in rows}
        self.assertAlmostEqual(by_name["xgboost"]["weight"], 0.45)
        self.assertEqual(by_name["xgboost"]["auc"], 0.69)
        self.assertAlmostEqual(by_name["lightgbm"]["weight"], 0.0)
        self.assertEqual(by_name["xgboost"]["n_eval"], 1107)

    def test_version_history_fields(self):
        recs = [{
            "_date": "20260831",
            "date": "2026-08-31",
            "adaptive_weights": {"xgboost": 0.45, "lightgbm": 0.0,
                                 "logistic": 0.0, "randomforest": 0.0,
                                 "mlp": 0.0},
            "pooled_preq_2021_2024": {"model_platt": {"auc": 0.69,
                                                      "logloss": 0.63,
                                                      "ece": 0.041}},
            "calibration": {"params": {"a": 2.3, "b": 0.1}},
            "verdict": {"adopt": True},
        }]
        rows = mon.version_history_rows(recs, "20260831")
        self.assertEqual(rows[0]["version"], "20260831")
        self.assertEqual(rows[0]["auc"], 0.69)
        self.assertEqual(rows[0]["ece_calibrated"], 0.041)
        self.assertEqual(rows[0]["calibration"], {"a": 2.3, "b": 0.1})
        self.assertTrue(rows[0]["adopt"])


class ExpandDriftCutTest(unittest.TestCase):
    def test_expands_backward_to_reach_floor(self):
        # A seasonal snap: 120 prior-season games (daily, ending before the
        # tail) + a 13-game post-season tail in the last 30 days. The 30-day
        # cut leaves only ~12 current games (< MIN_DRIFT_SAMPLES), so the
        # window must pull in prior games until the floor is met.
        earlier = pd.date_range("2025-09-07", periods=120, freq="D")
        tail = pd.date_range("2026-01-20", periods=13, freq="D")
        gd = pd.Series(list(earlier) + list(tail))
        cut = pd.to_datetime("2026-01-21")   # 'last 30 days' boundary
        new_cut = mon.expand_drift_cut(gd, cut)
        n_current = int((gd >= new_cut).sum())
        self.assertGreaterEqual(n_current, mon.MIN_DRIFT_SAMPLES)
        # Just-enough: it should not sweep in far more than the floor.
        self.assertLessEqual(n_current, mon.MIN_DRIFT_SAMPLES + 1)

    def test_window_already_over_floor_unchanged(self):
        gd = pd.Series(pd.date_range("2026-01-01", periods=60, freq="D"))
        cut = pd.to_datetime("2026-01-31")
        self.assertEqual(mon.expand_drift_cut(gd, cut), cut)

    def test_empty_or_nat_cut_passthrough(self):
        gd = pd.Series(pd.to_datetime(["2025-09-07", "2026-09-10"]))
        self.assertIsNone(mon.expand_drift_cut(gd, None))


class BuildMonitorTest(unittest.TestCase):
    def _inputs(self):
        # 300 baseline games + 120 current games (>= MIN_DRIFT_SAMPLES) so
        # the 30-day split needs no expansion and current/baseline are both
        # judgeable.
        feats = pd.DataFrame({
            "gameday": ["2025-09-07"] * 300 + ["2026-09-10"] * 120,
            "elo_diff": [10.0] * 150 + [20.0] * 150 + [18.0] * 60 + [22.0] * 60,
            "div_game": [0] * 150 + [1] * 150 + [1] * 60 + [0] * 60,
        })
        result = {
            "trained_at": "2026-08-31T01:00:00.000000Z",
            "_deployed": {"features": ["elo_diff", "div_game"]},
            "adaptive_weights": {"xgboost": 0.45, "lightgbm": 0.0,
                                 "logistic": 0.0, "randomforest": 0.0,
                                 "mlp": 0.0},
            "members": {"xgboost": {"auc": 0.69, "brier": 0.22, "logloss": 0.63}},
            "verdict": {"adopt": True},
        }
        history = pd.DataFrame({
            "game_date": ["2026-09-01"] * 40,
            "home_team": ["A"] * 40, "away_team": ["B"] * 40,
            "home_win_prob_model": [0.6] * 40,
            "actual_winner": ["A"] * 30 + ["B"] * 10,
        })
        cal = {"params": {"a": 2.3, "b": 0.1}}
        recs = [{"_date": "20260830", "verdict": {"adopt": False},
                 "pooled_preq_2021_2024": {},
                 "calibration": {"params": {"a": 1.0, "b": 0.0}}}]
        return feats, result, history, cal, recs

    def test_compose_record_has_all_page_keys(self):
        feats, result, history, cal, recs = self._inputs()
        rec = mon.build_model_monitor(
            feats=feats, result=result, history_df=history, calibration=cal,
            moneyline_records=recs, current_date="20260831",
            baseline_cut_date="2026-01-01")
        for key in ("last_retrained", "next_retrain", "upset_note",
                    "feature_drift", "features_metadata", "feature_coverage",
                    "ensemble", "rolling_brier", "rolling_brier_meta",
                    "brier_baseline", "brier_baseline_label",
                    "version_history"):
            self.assertIn(key, rec)

        self.assertEqual(rec["last_retrained"], "2026-08-31")
        self.assertGreaterEqual(rec["next_retrain"], "2026-09-01")
        # Drift current window = the 120 games >= baseline_cut; both sides
        # clear MIN_DRIFT_SAMPLES so statuses are judgeable (not INSUFFICIENT).
        self.assertEqual(rec["feature_drift"][0]["n_current"], 120)
        self.assertEqual(rec["feature_drift"][0]["n_baseline"], 300)
        self.assertNotEqual(rec["feature_drift"][0]["status"], "INSUFFICIENT")
        # Coverage rows cover every deployed feature.
        self.assertEqual([r["feature"] for r in rec["feature_coverage"]],
                         ["elo_diff", "div_game"])
        # Ensemble row for the deployed + zero members.
        self.assertEqual(len(rec["ensemble"]), 5)
        # Version history carries the prior dated record.
        self.assertEqual(len(rec["version_history"]), 1)
        # Rolling brier from the 40-game dense day.
        self.assertTrue(rec["rolling_brier"])

    def test_deployed_models_emit_weight_column_and_descriptions(self):
        """With the deployed re-fit's model objects in result (the production
        path), every drift row carries MLB-semantics weight_pct (percent,
        summing to ~100) and features_metadata is the real description map —
        never the retired-gate stub."""
        feats, result, history, cal, recs = self._inputs()
        cols = ["elo_diff", "div_game"]
        result["_models"] = {
            "xgboost": WeightImportanceTest._Tree(
                [4.0, 1.0, 0.9, 0.9]),   # numeric cols + 2 team-ID slots
            "lightgbm": WeightImportanceTest._Tree(
                [3.0, 2.0, 0.9, 0.9]),
            "logistic": WeightImportanceTest._Lin([1.2, -0.4]),
            "randomforest": WeightImportanceTest._Tree(
                [0.8, 0.6, 0.9, 0.9]),
            "mlp": WeightImportanceTest._Mlp(),
            "scaler": object(),
            "impute_median": object(),
            "categorical_vocab": {},
        }
        result["_deployed"] = {"features": cols}
        descs = {"elo_diff": "rating gap", "div_game": "division flag"}
        rec = mon.build_model_monitor(
            feats=feats, result=result, history_df=history,
            calibration=cal, moneyline_records=recs,
            current_date="20260831", baseline_cut_date="2026-01-01",
            feature_descriptions=descs)

        rows = {r["feature"]: r for r in rec["feature_drift"]}
        self.assertEqual(set(rows), set(cols))
        wp = [rows[c]["weight_pct"] for c in cols]
        self.assertTrue(all(w is not None for w in wp))
        self.assertAlmostEqual(sum(wp), 100.0, places=2)
        # No stale admission-gate text anywhere in the drift metadata.
        meta = rec["features_metadata"]
        self.assertEqual(set(meta), set(cols))
        self.assertIn("rating gap", meta["elo_diff"]["tooltip"])
        self.assertIn("division flag", meta["div_game"]["tooltip"])
        for c in cols:
            self.assertNotIn("admission gate",
                             meta[c]["tooltip"].lower())
            self.assertNotIn("admission gate",
                             meta[c]["definition"].lower())

    def test_no_models_omits_weight_column(self):
        """Back-compat: without model objects the page contract must stay
        valid with the weight column hidden (weight_pct None), exactly like
        MLB's None path."""
        feats, result, history, cal, recs = self._inputs()
        self.assertNotIn("_models", result)
        rec = mon.build_model_monitor(
            feats=feats, result=result, history_df=history,
            calibration=cal, moneyline_records=recs,
            current_date="20260831", baseline_cut_date="2026-01-01")
        self.assertTrue(all(r.get("weight_pct") is None
                            for r in rec["feature_drift"]))


if __name__ == "__main__":
    unittest.main()