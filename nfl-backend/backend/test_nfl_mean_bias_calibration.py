"""NFL per-side mean-bias calibration — tests (pure, no-network).

Covers ``nfl_bias_calibration`` (the prediction-layer linear recalibration
transform) and its joint-chain contract:
- Leakage pin: fit_calibration raises on ANY row with season >= sealed;
  apply_calibration refuses cal dicts whose fit_on != pooled_oof — sealed
  rows are structurally excluded from the fit.
- Bias-after-recalibration ~= 0 (< 0.1 pts pooled per side); raw pred/resid
  columns left untouched (transform rides alongside).
- Sigma re-estimate matches the pinned value (constant RMSE of CALIBRATED
  residuals — the old sigma was fit on biased residuals).
- Joint rebuild calls the EXISTING engine entrypoints unmodified
  (fit_joint_params / build_joint_pmfs on the engine-table output; the
  calibration module itself imports no engine and no moneyline).
- Determinism: two identical fit+apply chains byte-identical.
- Diagnostics: offset / slope-tilt labels, season + decile tables, OLS a/b
  CIs, construction-change flag at |a-1| > 0.15.
- FEATURE_COLUMNS untouched.

Run: python -m unittest test_nfl_mean_bias_calibration -v  (no network)
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_bias_calibration as bc
import nfl_features as nf
import nfl_joint_engine as je

FEATURES_BEFORE = list(nf.FEATURE_COLUMNS)


def _synth_pooled(n: int = 300, seed: int = 11) -> pd.DataFrame:
    """Deterministic synthetic pooled-OOF table with away-style bias.

    home: y = pred + noise (near-unbiased). away: y = -1.5 + 0.96*pred +
    noise — offset + mild shrinkage, the diagnosed away shape.
    """
    rng = np.random.default_rng(seed)
    pred_h = rng.uniform(17, 27, n)
    pred_a = rng.uniform(15, 25, n)
    y_h = np.clip(np.round(pred_h + rng.normal(0, 9.0, n)), 0, 75)
    y_a = np.clip(np.round(-1.5 + 0.96 * pred_a + rng.normal(0, 9.0, n)),
                  0, 75)
    return pd.DataFrame({
        "game_id": [f"G{i}" for i in range(n)],
        "fold_idx": np.arange(n) % 88,
        "season": np.repeat([2021, 2022, 2023, 2024], n // 4),
        "pred_home": np.round(pred_h, 4),
        "pred_away": np.round(pred_a, 4),
        "resid_home": np.round(y_h - pred_h, 4),
        "resid_away": np.round(y_a - pred_a, 4),
        "best_iter_home": 30,
        "best_iter_away": 30,
        "home_score": y_h,
        "away_score": y_a,
    })


def _away_frame(pred: np.ndarray, score: np.ndarray, n: int = 300
              ) -> pd.DataFrame:
    """Minimal diagnose-shaped frame around a custom away pred/score pair
    (home kept near-identity; season col present for the trend row)."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "game_id": [f"F{i}" for i in range(n)],
        "season": np.repeat([2021, 2022, 2023, 2024], n // 4),
        "pred_home": np.round(rng.uniform(17, 27, n), 4),
        "pred_away": np.round(np.asarray(pred, dtype=float), 4),
        "home_score": np.clip(np.round(rng.uniform(17, 27, n)), 0, 75),
        "away_score": np.clip(np.round(np.asarray(score, dtype=float)), 0, 75),
    })


def _synth_recovery() -> pd.DataFrame:
    """High-n, low-noise frame → tight a/b estimates (shape recovery test)."""
    rng = np.random.default_rng(13)
    n = 3000
    p = rng.uniform(5, 35, n)
    y = np.clip(np.round(-1.5 + 0.96 * p + rng.normal(0, 3.0, n)), 0, 75)
    return pd.DataFrame({
        "game_id": [f"R{i}" for i in range(n)],
        "fold_idx": np.arange(n) % 88,
        "season": np.repeat([2021, 2022, 2023, 2024], n // 4),
        "pred_home": np.round(p, 4),
        "pred_away": np.round(p, 4),
        "resid_home": np.round(y - p, 4),
        "resid_away": np.round(y - p, 4),
        "best_iter_home": 30, "best_iter_away": 30,
        "home_score": y, "away_score": y,
    })


class TestCalibrationFitApply(unittest.TestCase):
    """The prediction-layer transform: fit pooled-only, apply leak-free."""

    def setUp(self) -> None:
        self.pooled = _synth_pooled()
        self.cal = bc.fit_calibration(self.pooled)

    def test_fit_apply_removes_bias_lt_0_1(self) -> None:
        """Pooled bias-after-recalibration ~= 0 (< 0.1 pts per side)."""
        applied = bc.apply_calibration(self.pooled, self.cal)
        for side, rcol in bc.CAL_RESID.items():
            self.assertLess(abs(float(applied[rcol].mean())), 0.1)

    def test_fit_recovers_away_shape(self) -> None:
        """Away shape (offset −1.5, slope 0.96) recovered on high n: b ~ −1.5,
        a ~ 0.96; home stays near identity on the same data."""
        rec = _synth_recovery()
        cal = bc.fit_calibration(rec)
        m = cal["away"]
        self.assertLess(abs(m["a"] - 0.96), 0.05)
        self.assertLess(abs(m["b"] + 1.5), 0.5)
        # a/b CIs bracket the point estimates
        self.assertLessEqual(m["a_ci_low"], m["a"])
        self.assertGreaterEqual(m["a_ci_high"], m["a"])
        self.assertLessEqual(m["b_ci_low"], m["b"])
        self.assertGreaterEqual(m["b_ci_high"], m["b"])

    def test_season_leak_guard(self) -> None:
        """Any season >= sealed in the FIT input raises ValueError."""
        bad = pd.concat([self.pooled, self.pooled.assign(season=2025)],
                        ignore_index=True)
        with self.assertRaises(ValueError):
            bc.fit_calibration(bad)

    def test_apply_refuses_non_pooled_marker(self) -> None:
        """apply_calibration refuses cal params not fit on pooled OOF."""
        leaked = dict(self.cal, fit_on="sealed_2025")
        with self.assertRaises(ValueError):
            bc.apply_calibration(self.pooled, leaked)

    def test_apply_leaves_raw_columns_untouched(self) -> None:
        applied = bc.apply_calibration(self.pooled, self.cal)
        for side, pcol in bc.PRED_COLS.items():
            np.testing.assert_array_equal(applied[pcol].to_numpy(),
                                          self.pooled[pcol].to_numpy())
            self.assertIn(bc.CAL_PRED[side], applied.columns)
            self.assertIn(bc.CAL_RESID[side], applied.columns)
        # pred_cal + resid_cal == actual exactly (artifact consistency rule)
        for side, acol in bc.ACTUAL_COLS.items():
            rebuilt = applied[bc.CAL_PRED[side]].to_numpy(float) \
                + applied[bc.CAL_RESID[side]].to_numpy(float)
            np.testing.assert_allclose(rebuilt, applied[acol].astype(float),
                                       rtol=0, atol=1e-9)

    def test_ols_map_identity(self) -> None:
        rng = np.random.default_rng(3)
        p = rng.uniform(10, 40, 200)
        m = bc.ols_map(p, p)
        self.assertLess(abs(m["a"] - 1.0), 1e-9)
        self.assertLess(abs(m["b"]), 1e-9)
        self.assertGreater(m["r2"], 0.9999)
        self.assertLessEqual(m["a_ci_low"], m["a"])
        self.assertGreaterEqual(m["a_ci_high"], m["a"])

    def test_determinism_byte_identical(self) -> None:
        applied1 = bc.apply_calibration(self.pooled, self.cal)
        cal2 = bc.fit_calibration(self.pooled)
        applied2 = bc.apply_calibration(self.pooled, cal2)
        a = applied1.sort_values("game_id").reset_index(drop=True)
        b = applied2.sort_values("game_id").reset_index(drop=True)
        self.assertEqual(a.to_csv(index=False), b.to_csv(index=False))


class TestEngineTableAndChain(unittest.TestCase):
    """The chain contract: engine entrypoints, unmodified, fed calibrated rows."""

    def setUp(self) -> None:
        self.pooled = _synth_pooled()
        self.cal = bc.fit_calibration(self.pooled)
        self.applied = bc.apply_calibration(self.pooled, self.cal)
        self.eng = bc.engine_table(self.applied)

    def test_engine_table_holds_calibrated_standard_names(self) -> None:
        for side, pcol in bc.PRED_COLS.items():
            np.testing.assert_array_equal(
                self.eng[pcol].to_numpy(),
                self.applied[bc.CAL_PRED[side]].to_numpy())
            np.testing.assert_array_equal(
                self.eng[bc.RESID_COLS[side]].to_numpy(),
                self.applied[bc.CAL_RESID[side]].to_numpy())
        for side, acol in bc.ACTUAL_COLS.items():
            self.assertIn(acol, self.eng.columns)

    def test_engine_table_missing_cal_column_raises(self) -> None:
        bad = self.applied.drop(columns=[bc.CAL_RESID["away"]])
        with self.assertRaises(ValueError):
            bc.engine_table(bad)

    def test_chain_uses_engine_entrypoints_unmodified(self) -> None:
        """fit_joint_params + build_joint_pmfs on the engine table — the
        joint machinery is imported, never reimplemented."""
        params = je.fit_joint_params(self.eng)
        self.assertEqual(params["fit_on"], "pooled_oof")
        pmfs, summary = je.build_joint_pmfs(self.eng, params, p_tie=0.004)
        # GRID-INDEX CONVENTION FIX (joint-engine commit): marginal_breakpoints
        # now places P(score k) at index k on the 0..75 grid -> joints are
        # 76x76 (pre-fix they were 77x77, with cell k holding score k-1).
        # The engine's own grid, never reimplemented here.
        self.assertEqual(pmfs.shape, (len(self.eng), 76, 76))
        self.assertEqual(len(summary["derived"]), len(self.eng))
        # the calibration module is a pure transform — imports neither engine
        src = Path(bc.__file__).read_text()
        for mod in ("nfl_joint_engine", "nfl_per_side_engine",
                    "nfl_moneyline", "nfl_features"):
            self.assertNotIn(f"import {mod}", src)
            self.assertNotIn(f"from {mod}", src)

    def test_sigma_reestimate_matches_pinned_rmse(self) -> None:
        """Sigma re-estimation is in scope: constant sigma on CALIBRATED
        residuals == sqrt(mean(resid_cal^2)) — the pin the runner relies on."""
        for side in bc.PRED_COLS:
            rc = self.applied[bc.CAL_RESID[side]].to_numpy(float)
            pinned = round(float(np.sqrt(np.mean(rc ** 2))), 4)
            spec = je.constant_sigma(rc)
            self.assertEqual(spec["sigma0"], pinned)
        # When the engine chain picks constant sigma (homoskedastic noise
        # regime), it must carry exactly the pinned RMSE.
        params = je.fit_joint_params(self.eng)
        for side, key in (("home", "sigma_h"), ("away", "sigma_a")):
            rc = self.applied[bc.CAL_RESID[side]].to_numpy(float)
            pinned = round(float(np.sqrt(np.mean(rc ** 2))), 4)
            if params[key]["spec"] == "const":
                self.assertEqual(params[key]["sigma0"], pinned)

    def test_leak_guard_on_engine_params(self) -> None:
        params = je.fit_joint_params(self.eng)
        bad = dict(params, fit_on="sealed_2025")
        with self.assertRaises(ValueError):
            je.build_joint_pmfs(self.eng, bad, p_tie=0.004)


class TestDiagnostics(unittest.TestCase):
    """Step-1 diagnostics: report tables + advisory labels."""

    def setUp(self) -> None:
        self.pooled = _synth_pooled()
        self.diag = bc.diagnose(self.pooled)

    def test_away_offset_label(self) -> None:
        """Away bias −1.5 → offset label (|mean resid| >= 0.30, deterministic)."""
        labels = self.diag["away"]["classification"]["labels"]
        self.assertIn("offset", labels)
        self.assertGreater(abs(self.diag["away"]["stats"]["mean_resid"]), 0.5)

    def test_slope_tilt_label_on_shrunk_synthetic(self) -> None:
        """Steep slope (a = 0.5, tight noise) → slope-tilt label fires."""
        rng = np.random.default_rng(5)
        n = 400
        p = rng.uniform(10, 40, n)
        y = 20.0 + 0.5 * p + rng.normal(0, 2.0, n)   # a = 0.5, |a-1| = 0.5
        fake = _away_frame(p, y, n)
        d = bc.diagnose(fake)["away"]
        self.assertIn("slope tilt", d["classification"]["labels"])
        self.assertLess(abs(d["ols_actual_on_pred"]["a"] - 0.5), 0.05)

    def test_by_season_and_decile_tables(self) -> None:
        for side in bc.PRED_COLS:
            d = self.diag[side]
            self.assertEqual({r["season"] for r in d["by_season"]},
                             {2021, 2022, 2023, 2024})
            self.assertGreater(len(d["by_pred_decile"]), 3)
            ols = d["ols_actual_on_pred"]
            self.assertLessEqual(ols["a_ci_low"], ols["a"])
            self.assertGreaterEqual(ols["a_ci_high"], ols["a"])
            means = np.array([r["mean_resid"] for r in d["by_season"]])
            self.assertLess(abs(float(means.mean())
                                - d["stats"]["mean_resid"]), 1.0)
            self.assertGreaterEqual(len(d["by_pred_decile"]),
                                    len(d["by_pred_decile"][0]))

    def test_construction_change_flag(self) -> None:
        """|a-1| > 0.15 → construction-change flag (advisory, not a blocker)."""
        rng = np.random.default_rng(7)
        n = 400
        p = rng.uniform(10, 40, n)
        y = 20.0 + 0.5 * p + rng.normal(0, 2.0, n)   # |a-1| = 0.5 > 0.15
        fake = _away_frame(p, y, n)
        d = bc.diagnose(fake)["away"]
        self.assertGreater(abs(d["ols_actual_on_pred"]["a"] - 1.0), 0.15)
        self.assertTrue(d["construction_change_flag"])
        # near-identity slope (|a-1| < 0.15, tight noise) does NOT flag
        mild = self.pooled.copy()
        mild["pred_away"] = mild["away_score"].astype(float)
        mild["resid_away"] = 0.0
        m = bc.diagnose(mild)["away"]
        self.assertLess(abs(m["ols_actual_on_pred"]["a"] - 1.0), 0.05)
        self.assertFalse(m["construction_change_flag"])


class TestPins(unittest.TestCase):
    """FEATURE_COLUMNS untouched; transform never touches the moneyline."""

    def test_feature_columns_untouched(self) -> None:
        self.assertEqual(list(nf.FEATURE_COLUMNS), FEATURES_BEFORE)
        src = Path(bc.__file__).read_text()
        for bad in ("FEATURE_COLUMNS", "home_win", "is_home"):
            self.assertNotIn(bad, src)

    def test_transform_imports_nothing_production(self) -> None:
        src = Path(bc.__file__).read_text()
        for mod in ("nfl_moneyline", "nfl_features", "nfl_margin_engine"):
            self.assertNotIn(f"import {mod}", src)
            self.assertNotIn(f"from {mod}", src)


if __name__ == "__main__":
    unittest.main()
