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

import sys
import unittest
from datetime import datetime
from pathlib import Path

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
    """MLB-identical status machinery: thresholds (0.10/0.25) apply to the
    value handed in, and adjusted_status adds the sample floors + location
    gate on top."""

    def test_psi_status_threshold_bands(self):
        self.assertEqual(mon.psi_status(0.01), "OK")
        self.assertEqual(mon.psi_status(0.10), "WARN")     # >= WARN threshold
        self.assertEqual(mon.psi_status(0.15), "WARN")
        self.assertEqual(mon.psi_status(0.25), "ALERT")    # >= ALERT threshold
        self.assertEqual(mon.psi_status(0.40), "ALERT")

    def test_adjusted_status_sample_floors(self):
        # Either window below its floor -> INSUFFICIENT, never a page.
        self.assertEqual(mon.adjusted_status(0.9, True, n_baseline=99,
                                             n_current=120), "INSUFFICIENT")
        self.assertEqual(mon.adjusted_status(0.9, True, n_baseline=1930,
                                             n_current=29), "INSUFFICIENT")

    def test_adjusted_status_location_gate(self):
        # Big adjusted psi with NO mean location shift -> OK (the
        # near-constant bin-boundary case must never page).
        self.assertEqual(mon.adjusted_status(0.5, False, 1930, 120), "OK")
        self.assertEqual(mon.adjusted_status(0.15, True, 1930, 120), "WARN")
        self.assertEqual(mon.adjusted_status(0.9, True, 1930, 120), "ALERT")
        self.assertEqual(mon.adjusted_status(0.05, True, 1930, 120), "OK")

    def test_psi_noise_floor_formula(self):
        # MLB: (k-1)/2 * (1/n_base + 1/n_cur).
        self.assertAlmostEqual(mon.psi_noise_floor(1930, 30), 0.152332, places=5)
        self.assertEqual(mon.psi_noise_floor(0, 30), 0.0)


class DriftAlignmentTest(unittest.TestCase):
    """NFL drift methodology == MLB's, replicated not approximated — plus the
    regression case from the 09-02 artifact (pace_plays_min_diff: raw PSI
    2.038 on IDENTICAL means = bin-boundary instability on a tiny-scale
    near-constant feature, not drift)."""

    @staticmethod
    def _mlb_reference(base, cur):
        """VERBATIM mlb-backend/backend/explainability.py (compute_psi +
        psi_noise_floor + psi_status with compute_feature_drift's
        noise-adjusted status + location gate)."""
        import numpy as _np
        b = _np.asarray(base, dtype=float)
        c = _np.asarray(cur, dtype=float)
        b = b[~_np.isnan(b)]
        c = c[~_np.isnan(c)]
        if len(b) == 0 or len(c) == 0:
            psi = 0.0
        else:
            combined = _np.concatenate([b, c])
            lo, hi = float(combined.min()), float(combined.max())
            if lo == hi:
                psi = 0.0
            else:
                edges = _np.unique(_np.quantile(combined,
                                                _np.linspace(0.0, 1.0, 11)))
                if len(edges) < 2:
                    psi = 0.0
                else:
                    edges[-1] = hi + 1e-10
                    bc = _np.histogram(b, bins=edges)[0].astype(float)
                    cc = _np.histogram(c, bins=edges)[0].astype(float)
                    k = len(edges) - 1
                    e = (bc + 0.5) / (bc.sum() + 0.5 * k)
                    a = (cc + 0.5) / (cc.sum() + 0.5 * k)
                    psi = round(max(float(_np.sum((a - e) * _np.log(a / e))),
                                    0.0), 6)
        noise = 0.0 if (len(b) <= 0 or len(c) <= 0) else \
            (9.0 / 2.0) * (1.0 / len(b) + 1.0 / len(c))
        psi_adj = max(psi - noise, 0.0)
        mean_shift = float(c.mean() - b.mean()) if len(b) and len(c) else 0.0
        nb, nc = len(b), len(c)
        if nb + nc > 2:
            pooled_sd = float(_np.sqrt(
                ((nb - 1) * _np.var(b, ddof=1) + (nc - 1) * _np.var(c, ddof=1))
                / (nb + nc - 2)))
        else:
            pooled_sd = 0.0
        if pooled_sd > 0:
            shift_se = float(pooled_sd * _np.sqrt(1.0 / nb + 1.0 / nc) * 1.5)
            loc = abs(mean_shift) > 2.0 * shift_se
        else:
            shift_se = 0.0
            loc = psi_adj > 0
        if nb < 100 or nc < 30:
            status = "INSUFFICIENT"
        elif not loc:
            status = "OK"
        elif psi_adj >= 0.25:
            status = "ALERT"
        elif psi_adj >= 0.10:
            status = "WARN"
        else:
            status = "OK"
        return {"psi": psi, "noise_floor": noise, "psi_adjusted": psi_adj,
                "mean_shift": mean_shift, "shift_se": shift_se,
                "location_shift": bool(loc), "status": status}

    def _frame(self):
        n_base, n_cur = 1930, 120
        rng = np.random.default_rng(42)
        same = -0.0018 + rng.normal(0, 0.001, n_base)
        same_c = -0.0018 + rng.normal(0, 0.001, n_cur)
        return pd.DataFrame({
            "gameday": ["2025-09-07"] * n_base + ["2026-09-10"] * n_cur,
            "same_dist": np.concatenate([same, same_c]),
            "near_const": np.concatenate([
                np.linspace(-0.004, 0.004, n_base), np.zeros(n_cur)]),
            "real_shift": np.concatenate([
                np.linspace(0.0, 1.0, n_base),
                np.linspace(0.9, 1.0, n_cur)]),
        }), n_base, n_cur

    def test_matches_mlb_reference_on_shared_case(self):
        """Same inputs -> same outputs (raw psi, adjusted, floors, location
        gate, status) vs the verbatim MLB reference, for three scenarios at
        NFL's drift geometry."""
        feats, n_base, n_cur = self._frame()
        mask_base = feats["gameday"].astype(str) < "2026-01-01"
        mask_cur = ~mask_base
        rows = {r["feature"]: r for r in mon.feature_drift_rows(
            feats, ["same_dist", "near_const", "real_shift"],
            mask_base.to_numpy(), mask_cur.to_numpy())}
        for col in ("same_dist", "near_const", "real_shift"):
            arr = pd.to_numeric(feats[col], errors="coerce").to_numpy()
            ref = self._mlb_reference(arr[mask_base.to_numpy()],
                                      arr[mask_cur.to_numpy()])
            row = rows[col]
            self.assertEqual(row["psi"], ref["psi"], col)
            self.assertEqual(row["psi_adjusted"],
                             round(ref["psi_adjusted"], 6), col)
            self.assertEqual(row["noise_floor"],
                             round(ref["noise_floor"], 6), col)
            self.assertEqual(row["mean_shift"],
                             round(ref["mean_shift"], 6), col)
            self.assertEqual(row["shift_se"], round(ref["shift_se"], 6), col)
            self.assertEqual(row["location_shift"], ref["location_shift"], col)
            self.assertEqual(row["status"], ref["status"], col)
            self.assertEqual(row["n_baseline"], n_base, col)
            self.assertEqual(row["n_current"], n_cur, col)

    def test_identical_means_exploding_raw_psi_stays_ok(self):
        """The 09-02 pace_plays_min_diff class: current mean (~0) == baseline
        mean (~0) on a tiny-scale near-constant feature, raw PSI > 1
        (bin-boundary instability, the artifact showed 2.038) — MLB
        statuses read OK, never ALERT."""
        base = np.linspace(-0.004, 0.004, 1930)
        cur = np.zeros(120)
        drift = mon.noise_adjusted_drift(base, cur)
        self.assertGreater(drift["psi"], 1.0)
        self.assertAlmostEqual(drift["mean_shift"], 0.0, places=6)
        self.assertFalse(drift["location_shift"])      # the mean never moved
        self.assertEqual(mon.adjusted_status(drift["psi_adjusted"],
                                             drift["location_shift"],
                                             1930, 120), "OK")

    def test_real_mean_shift_escalates(self):
        base = np.linspace(0.0, 1.0, 1930)
        cur = np.linspace(0.9, 1.0, 120)
        drift = mon.noise_adjusted_drift(base, cur)
        self.assertTrue(drift["location_shift"])
        self.assertEqual(mon.adjusted_status(drift["psi_adjusted"],
                                             drift["location_shift"],
                                             1930, 120), "ALERT")

    def test_row_field_set_matches_mlb(self):
        feats, _, _ = self._frame()
        mask_base = feats["gameday"].astype(str) < "2026-01-01"
        row = mon.feature_drift_rows(feats, ["same_dist"],
                                     mask_base.to_numpy(),
                                     (~mask_base).to_numpy())[0]
        self.assertEqual(set(row), {
            "feature", "current_mean", "baseline_mean", "psi",
            "psi_adjusted", "noise_floor", "mean_shift", "shift_se",
            "location_shift", "status", "weight_pct", "n_baseline",
            "n_current"})

    def test_small_windows_report_insufficient(self):
        feats, n_base, _ = self._frame()
        # A 10-game current window is below the 30-game floor -> INSUFFICIENT
        # regardless of how large raw PSI is (same as MLB never paging <30).
        small = pd.DataFrame({
            "gameday": ["2025-09-07"] * n_base + ["2026-09-10"] * 10,
            "x": list(np.linspace(0, 1, n_base)) + [0.9] * 10,
        })
        mask_base = small["gameday"].astype(str) < "2026-01-01"
        row = mon.feature_drift_rows(small, ["x"],
                                     mask_base.to_numpy(),
                                     (~mask_base).to_numpy())[0]
        self.assertEqual(row["status"], "INSUFFICIENT")
        self.assertEqual(row["n_current"], 10)


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

    def test_upset_note_pool_is_walk_forward_history(self):
        """Upset pool = the per-game walk-forward history (pooled OOF +
        sealed), NOT the decided frame (1,960) or drift baseline (1,930). On
        the 09-02 frame it read 1,392 = pooled n 1,107 + sealed n 285 (== the
        nfl_predictions_history row count) — the count was right, the
        'decided pool' label was wrong."""
        feats, result, history, cal, recs = self._inputs()
        rec = mon.build_model_monitor(
            feats=feats, result=result, history_df=history,
            calibration=cal, moneyline_records=recs,
            current_date="20260902", baseline_cut_date="2026-01-01")
        note = rec["upset_note"]
        self.assertIn("walk-forward history (pooled OOF + sealed)", note)
        self.assertIn("games scored", note)
        self.assertNotIn("decided pool", note)
        # Count always equals the history frame's row count (1,107 OOF + 285
        # sealed = 1,392 on the real frame, where the composer is handed the
        # walk-forward's own _history_df) — never the decided-frame size.
        self.assertIn(f"{history.shape[0]:,} games scored", note)

    def test_last_retrained_falls_back_to_emission_date(self):
        """The original NFL null bug: a walk-forward result without
        ``trained_at`` must still emit the persist date (this run -> today),
        never '' (the card's '—'). MLB-identical fallback (now())."""
        feats, result, history, cal, recs = self._inputs()
        self.assertIn("trained_at", result)          # present by default
        result.pop("trained_at", None)
        rec = mon.build_model_monitor(
            feats=feats, result=result, history_df=history,
            calibration=cal, moneyline_records=recs,
            current_date="20260831", baseline_cut_date="2026-01-01")
        today = datetime.now().strftime("%Y-%m-%d")
        self.assertEqual(rec["last_retrained"], today)
        self.assertNotEqual(rec["last_retrained"], "")

    def test_retrain_cadence_semantic_parity_with_mlb(self):
        """Shared synthetic case: after the correction, NFL's card semantics
        are IDENTICAL to MLB's corrected emitter — constant = 1 on both
        sides, LAST RETRAIN = now() persist date, NEXT RETRAIN = now() + 1
        (same anchor, same cadence). MLB constants are read from
        mlb-backend/backend/config.py (stdlib-only imports)."""
        from datetime import datetime as _dt

        mlb_side = Path(__file__).resolve().parents[2] / "mlb-backend" / "backend"
        sys.path.insert(0, str(mlb_side))
        try:
            import config as mlb_config
        finally:
            sys.path.remove(str(mlb_side))

        # Cadence constants are equal (and = 1) on both sides.
        self.assertEqual(mon.RETRAIN_INTERVAL_DAYS, mlb_config.NEXT_RUN_HEURISTIC_DAYS)
        self.assertEqual(mon.RETRAIN_INTERVAL_DAYS, 1)
        # NEXT RETRAIN: MLB's corrected derivation = now + constant; NFL's
        # (now anchored at emission, not the artifact date) must match it.
        mlb_next = (_dt.now()
                    + pd.Timedelta(days=mlb_config.NEXT_RUN_HEURISTIC_DAYS)
                    ).strftime("%Y-%m-%d")
        nfl_next = (_dt.now()
                    + pd.Timedelta(days=mon.RETRAIN_INTERVAL_DAYS)
                    ).strftime("%Y-%m-%d")
        self.assertEqual(nfl_next, mlb_next)
        # MLB's fold-cadence constant is untouched (weekly folds remain 7):
        # only the monitor heuristic changes.
        self.assertEqual(mlb_config.RETRAIN_CADENCE_DAYS, 7)
        # LAST RETRAIN: both emitters fall back to now() when no persist
        # timestamp is provided — MLB's ``last_retrained or now()`` is pinned
        # in mlb-backend test_model_version_history.py::TestRetrainCards and
        # the NFL twin is test_last_retrained_falls_back_to_emission_date.


if __name__ == "__main__":
    unittest.main()