"""Tests for the run-engine challenger ablation (Phase A base fix + Phase B
underlying-model sweep).

The harness (run_engine_challenger_ablation.py) is a standalone,
read-only challenger program — it writes ONLY the date-stamped record JSON
and per-arm parquet caches under /tmp. These tests assert the pure-math
core and the discipline rules:

  - C2/C3 edge corrections are fit on the PRE-sealed window only (a sealed
    observation never touches the fitted k / isotonic map / p).
  - C2/C3 preserve the LEVEL: λ'_H + λ'_A == λ_H + λ_A per game.
  - The linear edge correction recovers a known slope with an offset
    (the probe's actual_margin ≈ 0.014 + 1.66·λ_edge form).
  - The power form recovers a known exponent p.
  - CRPS is computed correctly (point-mass forecast ⇒ 0 on its own game).
  - `_nb_grad` matches the numerical NB-NLL gradient (benign regime).
  - RELAXED_LGBM_PARAMS is the production params with regularization
    strictly relaxed (num_leaves up, min_child_samples down,
    min_gain_to_split down) — C1's defining change.
  - The Phase-A leader rule requires sealed CRPS improvement + pooled
    corroboration + totals stability.
  - The NB distributional-NN forward pass produces finite (λ, α) outputs
    with α floored at ALPHA_FLOOR.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from run_engine import ALPHA_FLOOR, RUN_LGBM_PARAMS
from run_engine_challenger_ablation import (
    RELAXED_LGBM_PARAMS, _crps, _fit_dist_nn, _nb_grad, _nb_logpmf,
    _phase_a_verdict, apply_isotonic_edge, apply_linear_edge, calibrate,
    fit_isotonic, fit_linear_k, fit_power_p, price_arm, score_arm,
    sealed_masks)


def _results_like(c0, c1, c2, c3, pooled0, pooled1, pooled2, pooled3,
                  totals):
    """Minimal results dict shaped like the harness's score_arm output."""
    def r(mc, pc, tc):
        return {"margin_crps_sealed": mc, "margin_crps_pooled": pc,
                "totals_crps_sealed": tc, "totals_crps_pooled": 0.0,
                "cover_cal_sealed": {"extreme": [{}, {}]},
                "pwin_sd_sealed": 0.03}
    return {"C0": r(c0, pooled0, 0.0), "C1": r(c1, pooled1, 0.0),
            "C2": r(c2, pooled2, 0.0), "C3": r(c3, pooled3, totals)}


class TestEdgeCorrections(unittest.TestCase):

    def test_linear_k_recovers_slope_with_offset(self):
        rng = np.random.default_rng(7)
        d = rng.normal(0.6, 0.25, 4000)
        # λ_edge is the FULL differential: lam_h - lam_a = 2d.
        # probe form: actual_margin = 0.014 + 1.66 * λ_edge + noise
        edge = 2.0 * d
        margin = 0.014 + 1.66 * edge + rng.normal(0, 0.35, len(d))
        lam_h = 4.5 + d
        lam_a = 4.5 - d
        k = fit_linear_k(lam_h, lam_a, margin)
        self.assertAlmostEqual(k, 1.66, delta=0.08)

    def test_apply_linear_edge_preserves_sum(self):
        rng = np.random.default_rng(1)
        lh = rng.uniform(2.0, 6.0, 200)
        la = rng.uniform(2.0, 6.0, 200)
        k = 1.8
        lh2, la2 = apply_linear_edge(lh, la, k)
        np.testing.assert_allclose(lh2 + la2, lh + la, atol=1e-12)
        # edge is scaled by k: (h'-a') = k (h-a)
        np.testing.assert_allclose(lh2 - la2, k * (lh - la), atol=1e-12)

    def test_fit_on_oof_only_sealed_untouched(self):
        """Changing a SEALED game's margin must not move k / p / isotonic map."""
        rng = np.random.default_rng(3)
        n = 500
        d = rng.normal(0.5, 0.3, n)
        margin = 0.0 + 1.5 * d + rng.normal(0, 0.3, n)
        lh = 4.0 + d
        la = 4.0 - d
        pre = np.zeros(n, bool)
        pre[:420] = True
        k1 = fit_linear_k(lh[pre], la[pre], margin[pre])
        p1 = fit_power_p(lh[pre], la[pre], margin[pre])
        iso1 = fit_isotonic(lh[pre], la[pre], margin[pre])
        m1 = iso1.predict(lh - la)
        # corrupt every SEALED margin — fits must be identical
        margin2 = margin.copy()
        margin2[~pre] += 50.0
        k2 = fit_linear_k(lh[pre], la[pre], margin2[pre])
        p2 = fit_power_p(lh[pre], la[pre], margin2[pre])
        iso2 = fit_isotonic(lh[pre], la[pre], margin2[pre])
        m2 = iso2.predict(lh - la)
        self.assertAlmostEqual(k1, k2, places=10)
        self.assertAlmostEqual(p1, p2, places=10)
        np.testing.assert_array_equal(m1, m2)

    def test_power_recovers_exponent(self):
        rng = np.random.default_rng(5)
        d = rng.normal(0.4, 0.2, 3000)
        d = np.clip(d, 0.05, None)
        # margin is a superlinear function of the FULL edge (2d)
        edge = 2.0 * d
        margin = np.sign(edge) * np.abs(edge) ** 1.7
        lam_h = 4.0 + d
        lam_a = 4.0 - d
        p = fit_power_p(lam_h, lam_a, margin)
        self.assertAlmostEqual(p, 1.7, delta=0.15)

    def test_isotonic_applies_and_preserves_sum(self):
        rng = np.random.default_rng(9)
        d = rng.normal(0.5, 0.3, 300)
        lam_h = 4.0 + d
        lam_a = 4.0 - d
        margin = 1.6 * d + rng.normal(0, 0.2, len(d))
        iso = fit_isotonic(lam_h, lam_a, margin)
        lh2, la2 = apply_isotonic_edge(lam_h, lam_a, iso)
        np.testing.assert_allclose(lh2 + la2, lam_h + lam_a, atol=1e-9)
        # isotonic map is monotone on the edge
        edges = np.linspace(-1.5, 1.5, 50)
        preds = iso.predict(edges)
        self.assertTrue(np.all(np.diff(preds) >= -1e-12))


class TestCrpsAndCalibration(unittest.TestCase):

    def test_crps_point_mass_is_zero(self):
        pmf = np.zeros((3, 121))
        margins = list(range(-60, 61))
        y = np.array([-4, 0, 13])
        for i, yi in enumerate(y):
            pmf[i, margins.index(yi)] = 1.0
        self.assertAlmostEqual(_crps(pmf, margins, y), 0.0, places=10)

    def test_crps_misplaced_mass_positive(self):
        margins = list(range(-60, 61))
        pmf = np.zeros((1, 121))
        pmf[0, margins.index(5)] = 1.0
        self.assertGreater(_crps(pmf, margins, np.array([-5])), 0.0)

    def test_calibrate_tables(self):
        rng = np.random.default_rng(11)
        p = np.clip(rng.normal(0.5, 0.15, 2000), 0.01, 0.99)
        y = (rng.uniform(size=len(p)) < p).astype(float)
        cal = calibrate(p, y)
        self.assertEqual(len(cal["deciles"]), 10)
        self.assertEqual(len(cal["extreme"]), 2)
        self.assertIn("overall_delta", cal)
        # decile bins partition [0,1)
        lo = 0.0
        for row in cal["deciles"]:
            self.assertTrue(row["bin"].startswith(f"[{lo:.2f}"))
            lo = round(lo + 0.1, 2)


class TestNbGrad(unittest.TestCase):

    def test_grad_matches_numerical(self):
        from scipy.special import gammaln

        def ll(k, mu, alpha):
            a = np.maximum(np.asarray(alpha, float), ALPHA_FLOOR)
            n = 1.0 / a
            p = n / (n + mu)
            return (gammaln(k + n) - gammaln(n) - gammaln(k + 1.0)
                    + n * np.log(p) + k * np.log1p(-p))

        k = np.array([0.0, 1.0, 2.0, 3.0])
        mu = np.array([3.0, 3.5, 4.0, 4.5])
        alpha = np.array([0.5, 0.4, 0.3, 0.2])
        eps = 1e-7
        nm = (ll(k, mu + eps, alpha) - ll(k, mu - eps, alpha)) / (2 * eps)
        na = (ll(k, mu, alpha + eps) - ll(k, mu, alpha - eps)) / (2 * eps)
        gm, ga = _nb_grad(k, mu, alpha)
        self.assertLess(np.abs(-nm - gm).max(), 1e-6)
        self.assertLess(np.abs(-na - ga).max(), 1e-6)


class TestRelaxedParams(unittest.TestCase):

    def test_relaxed_is_strictly_relaxed(self):
        self.assertGreater(RELAXED_LGBM_PARAMS["num_leaves"],
                           RUN_LGBM_PARAMS["num_leaves"])
        self.assertLess(RELAXED_LGBM_PARAMS["min_child_samples"],
                        RUN_LGBM_PARAMS["min_child_samples"])
        self.assertLess(RELAXED_LGBM_PARAMS["min_gain_to_split"],
                        RUN_LGBM_PARAMS["min_gain_to_split"])
        # everything else carried through unchanged
        for k, v in RUN_LGBM_PARAMS.items():
            if k not in ("num_leaves", "min_child_samples", "min_gain_to_split"):
                self.assertEqual(RELAXED_LGBM_PARAMS[k], v)


class TestPhaseAVerdict(unittest.TestCase):

    def test_no_challenger_wins_when_none_beat_sealed(self):
        # C2 beats sealed but NOT pooled → no challenger wins.
        res = _results_like(0.620, 0.622, 0.619, 0.621,
                            0.630, 0.631, 0.631, 0.628, totals=0.0)
        v = _phase_a_verdict(res)
        self.assertEqual(v["winner"], "C0")

    def test_first_challenger_that_beats_sealed_and_pooled_wins(self):
        # C1 wins sealed but NOT pooled → no; C2 wins both → C2.
        res = _results_like(0.620, 0.618, 0.615, 0.616,
                            0.630, 0.631, 0.629, 0.632, totals=0.0)
        v = _phase_a_verdict(res)
        self.assertEqual(v["winner"], "C2")

    def test_totals_degradation_blocks_win(self):
        # C3 wins sealed + pooled but blows up totals → blocked.
        res = _results_like(0.620, 0.625, 0.624, 0.610,
                            0.630, 0.628, 0.629, 0.625, totals=0.30)
        v = _phase_a_verdict(res)
        self.assertNotEqual(v["winner"], "C3")


class TestDistNn(unittest.TestCase):

    def test_nn_forward_finite_and_alpha_floored(self):
        rng = np.random.default_rng(0)
        Xtr = rng.normal(size=(300, 8))
        ytr = np.column_stack([rng.poisson(4.0, 300), rng.poisson(3.5, 300)])
        Xva = rng.normal(size=(40, 8))
        yva = np.column_stack([rng.poisson(4.0, 40), rng.poisson(3.5, 40)])
        pred = _fit_dist_nn(Xtr, ytr, Xva, yva)
        self.assertEqual(pred.shape, (40, 4))
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertTrue(np.all(pred[:, 0] > 0) and np.all(pred[:, 2] > 0))
        self.assertTrue(np.all(pred[:, 1] >= ALPHA_FLOOR - 1e-9))
        self.assertTrue(np.all(pred[:, 3] >= ALPHA_FLOOR - 1e-9))


class TestScoring(unittest.TestCase):

    def test_score_arm_runs_and_shapes(self):
        rng = np.random.default_rng(2)
        n = 400
        lam_h = rng.uniform(3.0, 5.5, n)
        lam_a = rng.uniform(3.0, 5.5, n)
        alpha_h = np.full(n, 0.12)
        alpha_a = np.full(n, 0.12)
        hs = rng.poisson(lam_h).astype(float)
        as_ = rng.poisson(lam_a).astype(float)
        pre = np.zeros(n, bool)
        pre[:350] = True
        hold = ~pre
        out = score_arm(lam_h, lam_a, alpha_h, alpha_a, hs, as_, pre, hold)
        for key in ("margin_crps_pooled", "margin_crps_sealed",
                    "totals_crps_pooled", "totals_crps_sealed",
                    "cover_cal_pooled", "cover_cal_sealed",
                    "pwin_sd_pooled", "pwin_sd_sealed",
                    "pwin_ece_pooled", "pwin_ece_sealed",
                    "over_cal_pooled", "over_cal_sealed"):
            self.assertIn(key, out)
        self.assertLessEqual(out["pwin_sd_sealed"], 0.5)

    def test_price_arm_probabilities_are_valid(self):
        rng = np.random.default_rng(4)
        n = 50
        lam_h = rng.uniform(3.0, 5.5, n)
        lam_a = rng.uniform(3.0, 5.5, n)
        al = np.full(n, 0.10)
        pmf_m, pmf_t = price_arm(None, lam_h, lam_a, al, al)
        self.assertEqual(pmf_m.shape, (n, 121))
        self.assertEqual(pmf_t.shape, (n, 26))
        np.testing.assert_allclose(pmf_m.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(pmf_t.sum(axis=1), 1.0, atol=1e-6)


class TestSealedMasks(unittest.TestCase):

    def test_sealed_is_chronological_last(self):
        rng = np.random.default_rng(6)
        dates = pd.date_range("2025-03-01", periods=400, freq="D")
        oof = pd.DataFrame({"game_date": rng.choice(
            dates.strftime("%Y-%m-%d"), 400)})
        pre, hold = sealed_masks(oof)
        self.assertEqual(int(hold.sum()), 284)
        self.assertEqual(int(pre.sum()), 116)
        # every held game is chronologically after every pre game
        pre_dates = pd.to_datetime(oof["game_date"][pre])
        hold_dates = pd.to_datetime(oof["game_date"][hold])
        self.assertLessEqual(pre_dates.max(), hold_dates.min())


if __name__ == "__main__":
    unittest.main()
