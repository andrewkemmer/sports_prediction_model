"""NFL joint-engine grid-index fix — regression pins (pure, no-network).

The fix (this commit): ``nfl_joint_engine.marginal_breakpoints`` now places
P(score k) at index k, matching ``dn_pmf`` and the engine's documented
convention. Pre-fix, the breakpoints sat at F(arange(76) - 0.5) plus
endpoints, so cell k held the mass of score k-1: argmax of
``marginal_pmf(25, 9, "dn")`` sat at index 26, joints were 77x77 (off the
0..75 grid), and every derived total carried a systematic +2 shift.

Pins (spec):
 1. Convention pin: dn_pmf == marginal_pmf for a grid of DN means —
    P(score k) lives at index k in BOTH paths (len 76 on the 0..75 grid).
 2. Total-mean pin: the convolution mean equals the sum of the marginal
    means to 1e-9, and sits within 0.5 of mu_H + mu_A (the +2 regression
    sat ~2.05 away). The residual off mu_H + mu_A is the DN 0-floor clamp
    bias (~0.07 at these mu/sigma), not the index bug.
 3. Argmax regression: DN mean 25 -> index 25 in both paths (not 26).
 4. Determinism: two identical builds byte-identical (the G5 pin).
 5. Marginal sums to 1 (no mass lost/gained by the fix); joint 76x76.
 6. FEATURE_COLUMNS untouched + no moneyline/margin import (house pins).

Run: python -m unittest test_nfl_joint_grid_fix -v   (no network needed)
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_features as nf
import nfl_joint_engine as je


def _params() -> dict:
    """The era-record joint params (dn / const sigma 9.663/9.0789 / rho)."""
    return {
        "family": "dn",
        "sigma_h": {"spec": "const", "sigma0": 9.663, "q": 0.0},
        "sigma_a": {"spec": "const", "sigma0": 9.0789, "q": 0.0},
        "rho": 0.0076,
        "fit_on": "pooled_oof",
        "grid_max": je.GRID_MAX,
    }


def _synth_rows(n: int = 12, seed: int = 5) -> pd.DataFrame:
    """Small deterministic prediction frame (mu/sigma pair for each game)."""
    rng = np.random.default_rng(seed)
    mu_h = rng.uniform(18, 27, n)
    mu_a = rng.uniform(15, 25, n)
    y_h = np.clip(np.round(rng.normal(mu_h, 9.0)), 0, 75)
    y_a = np.clip(np.round(rng.normal(mu_a, 9.0)), 0, 75)
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "pred_home": np.round(mu_h, 4),
        "pred_away": np.round(mu_a, 4),
        "home_score": y_h,
        "away_score": y_a,
    })


class GridIndexFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = _params()

    # ── 1. convention pin: P(score k) at index k in both paths ─────────────

    def test_marginals_equal_on_grid_of_means(self) -> None:
        for mu in (5.0, 15.0, 25.0, 40.0, 60.0):
            a = je.dn_pmf(mu, 9.0)
            b = je.marginal_pmf(mu, 9.0, "dn")
            self.assertEqual(a.shape, (76,))
            self.assertEqual(b.shape, (76,))
            np.testing.assert_allclose(a, b, atol=1e-12)
        # the NB path got the same alignment
        for mu in (10.0, 22.0, 35.0):
            a = je.nb_pmf(mu, 7.0)
            b = je.marginal_pmf(mu, 7.0, "nb")
            self.assertEqual(a.shape, (76,))
            self.assertEqual(b.shape, (76,))
            np.testing.assert_allclose(a, b, atol=1e-12)

    # ── 2. total-mean pin: convolution identity + no +2 regression ─────────

    def test_total_mean_convolution_identity(self) -> None:
        mu_h, mu_a = 23.0, 20.0
        J = je.joint_pmf_copula(mu_h, mu_a, self.params)
        tot = je.total_pmf_from_joint(J)
        s = np.arange(len(tot), dtype=float)
        e_tot = float((s * tot).sum())
        sig_h = je.sigma_callable(self.params["sigma_h"])(mu_h)
        sig_a = je.sigma_callable(self.params["sigma_a"])(mu_a)
        mh = je.marginal_pmf(mu_h, sig_h, "dn")
        ma = je.marginal_pmf(mu_a, sig_a, "dn")
        e_h = float((np.arange(76) * mh).sum())
        e_a = float((np.arange(76) * ma).sum())
        # exact identity: convolution mean == sum of marginal means
        self.assertLess(abs(e_tot - (e_h + e_a)), 1e-9)
        # regression catcher: pre-fix this sat ~2.05 above mu_H + mu_A; the
        # remaining ~0.07 is the DN 0-floor clamp bias, not the index bug.
        self.assertLess(abs(e_tot - (mu_h + mu_a)), 0.5)
        self.assertLess(e_tot - (mu_h + mu_a), 0.5)
        self.assertGreater(e_tot - (mu_h + mu_a), 0.0)

    # ── 3. argmax regression: score-25 mode -> index 25, not 26 ────────────

    def test_argmax_regression_index_25(self) -> None:
        self.assertEqual(int(np.argmax(je.dn_pmf(25.0, 9.0))), 25)
        self.assertEqual(int(np.argmax(je.marginal_pmf(25.0, 9.0, "dn"))), 25)
        self.assertNotEqual(int(np.argmax(je.marginal_pmf(25.0, 9.0, "dn"))),
                            26)  # pre-fix value

    # ── 4. determinism (G5) ────────────────────────────────────────────────

    def test_determinism_byte_identical(self) -> None:
        rows = _synth_rows()
        pmfs1, s1 = je.build_joint_pmfs(rows, self.params, p_tie=0.004)
        pmfs2, s2 = je.build_joint_pmfs(rows, self.params, p_tie=0.004)
        np.testing.assert_array_equal(pmfs1, pmfs2)
        self.assertEqual(s1["derived"].to_csv(index=False),
                         s2["derived"].to_csv(index=False))

    # ── 5. mass conservation: sums to 1, 76-cell grid, 76x76 joint ────────

    def test_marginal_sums_to_one_no_mass_shift(self) -> None:
        for mu in (5.0, 25.0, 60.0):
            for pmf in (je.dn_pmf(mu, 9.0),
                        je.marginal_pmf(mu, 9.0, "dn"),
                        je.marginal_pmf(mu, 9.0, "nb")):
                self.assertLess(abs(float(pmf.sum()) - 1.0), 1e-12)
                self.assertEqual(len(pmf), 76)
        J = je.joint_pmf_copula(23.0, 20.0, self.params)
        self.assertEqual(J.shape, (76, 76))  # pre-fix this was 77x77
        self.assertLess(abs(float(J.sum()) - 1.0), 1e-4)

    # ── 6. house pins ──────────────────────────────────────────────────────

    def test_feature_columns_untouched(self) -> None:
        before = list(nf.FEATURE_COLUMNS)
        self.assertEqual(list(nf.FEATURE_COLUMNS), before)
        # engine never mutates the served pool (source-text pin, house style)
        src = Path(je.__file__).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) \
                    else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == "FEATURE_COLUMNS":
                        self.fail("nfl_joint_engine mutates FEATURE_COLUMNS")

    def test_no_moneyline_or_margin_import(self) -> None:
        src = Path(je.__file__).read_text()
        self.assertNotIn("nfl_moneyline", src)
        self.assertNotIn("nfl_margin_engine", src)


if __name__ == "__main__":
    unittest.main()