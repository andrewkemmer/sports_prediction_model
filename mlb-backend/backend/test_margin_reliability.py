"""Light fixture-driven tests for margin_reliability_diagnostic.py.

These pin the MATH of the harness on tiny hand-computed fixtures only —
no artifact-dependent assertions (the read-only diagnostic consumes
run_engine_oof_<date>.csv / run_engine_markets_<date>.csv, which are not
required to exist for these tests):

  1. The randomized PIT transform on a known 2-point distribution
     (U_i = F̂(y_i − 1) + V_i·p̂(y_i) lands in the right disjoint ranges).
  2. Tie renormalization BEFORE F̂ — P(margin=0) resolves to ±1
     home-weighted (share MARGIN_PLUS1_HOME_SHARE), every other margin
     unchanged, row sums to 1.
  3. The discrete CRPS formula on a toy case (point mass → 0; a fair
     2-point forecast on a 2-margin grid → the closed form).
"""
from __future__ import annotations

import unittest

import numpy as np

import margin_reliability_diagnostic as mrd
from run_engine import MARGIN_PLUS1_HOME_SHARE, nb_pmf_matrix


class TestRandomizedPit(unittest.TestCase):
    """Randomized PIT formula on a known 2-point distribution."""

    def test_pit_lands_in_disjoint_f_bands(self):
        # Margins {-1, 0, +1} with p = {0.25, 0.50, 0.25}. The randomized
        # PIT of y must fall in the band [F(y-1), F(y)):
        #   y = -1: [0.00, 0.25)   y = 0: [0.25, 0.75)   y = +1: [0.75, 1.00)
        # The three bands are disjoint, so the check is seed-independent.
        pmf = np.array([[0.25, 0.50, 0.25]])
        margins = [-1, 0, 1]
        for y, (lo, hi) in {-1: (0.0, 0.25),
                             0: (0.25, 0.75),
                             1: (0.75, 1.0)}.items():
            U = mrd._randomized_pit(pmf, margins, np.array([y]), seed=7)
            self.assertGreaterEqual(U[0], lo, f"y={y}")
            self.assertLess(U[0], hi, f"y={y}")

    def test_point_mass_pit_is_uniform(self):
        # A point-mass forecast at y is trivially correctly specified:
        # F(y-1) = 0 and p(y) = 1, so U = V ~ U(0,1). 4000 independent
        # games with a fixed seed → mean ~ 0.5.
        margins = [0, 1]
        pmf = np.tile([[0.0, 1.0]], (4000, 1))
        y = np.full(4000, 1)
        U = mrd._randomized_pit(pmf, margins, y, seed=11)
        self.assertGreater(U.mean(), 0.48)
        self.assertLess(U.mean(), 0.52)
        self.assertGreaterEqual(U.min(), 0.0)
        self.assertLess(U.max(), 1.0)

    def test_lowest_margin_uses_zero_cdf_before(self):
        # y at the grid's lowest margin: F(y-1) is outside the grid → 0.
        margins = [-2, -1, 0]
        pmf = np.array([[0.6, 0.3, 0.1]])
        U = mrd._randomized_pit(pmf, margins, np.array([-2]), seed=3)
        self.assertGreaterEqual(U[0], 0.0)
        self.assertLess(U[0], 0.6)  # = V * p(-2) with p(-2)=0.6


class TestTieRenormalization(unittest.TestCase):
    """The SHIPPED tie fix is applied BEFORE F̂: P(0) → ±1 home-weighted."""

    def _raw_conv(self, lam_h, lam_a, al_h, al_a, grid):
        # Independent convolution WITHOUT the tie fix (the harness applies
        # the fix on top of this exact product). G = max|margin| + 1 is the
        # harness's grid bound — the i=G/j=G cells contribute to margins
        # 0 and ±1 and must be included for an exact match.
        G = max(abs(int(m)) for m in grid) + 1
        ks = np.arange(G + 1)
        ph = nb_pmf_matrix(ks, np.array([[lam_h]]), np.array([[al_h]]))
        pa = nb_pmf_matrix(ks, np.array([[lam_a]]), np.array([[al_a]]))
        raw = np.zeros(len(grid))
        for i in range(G + 1):
            for j in range(G + 1):
                d = i - j
                if d in grid:
                    raw[grid.index(d)] += ph[0, i] * pa[0, j]
        return raw

    def test_tie_mass_resolves_home_weighted(self):
        # Small lambdas so P(margin=0) has real mass; alpha = the shipped
        # default dispersion value.
        lam_h, lam_a, al = 1.6, 1.2, mrd.ALPHA_FLOOR + 0.2
        grid = list(range(-4, 5))
        raw = self._raw_conv(lam_h, lam_a, al, al, grid)
        p0 = raw[grid.index(0)]
        p1 = raw[grid.index(1)]
        pn1 = raw[grid.index(-1)]
        # The shipped PMF from the harness builder.
        pmf = mrd._nb_score_pmf(np.array([lam_h]), np.array([lam_a]),
                                np.array([al]), np.array([al]), grid)
        row = pmf[0]
        self.assertAlmostEqual(row[grid.index(0)], 0.0, places=12)
        self.assertAlmostEqual(row[grid.index(1)],
                               p1 + MARGIN_PLUS1_HOME_SHARE * p0, places=12)
        self.assertAlmostEqual(row[grid.index(-1)],
                               pn1 + (1.0 - MARGIN_PLUS1_HOME_SHARE) * p0,
                               places=12)
        # Every other margin is untouched.
        for d in (-4, -3, -2, 2, 3, 4):
            self.assertAlmostEqual(row[grid.index(d)], raw[grid.index(d)],
                                   places=12, msg=f"margin {d}")
        # The small ±4 grid truncates the NB tails (≈3.4% here) — the
        # harness's normalization check runs on its wide ±60 grid. What
        # matters is that the tie fix CONSERVES mass: the renormalized row
        # carries exactly the raw row's mass.
        self.assertAlmostEqual(float(row.sum()), float(raw.sum()),
                               places=12)
        self.assertGreater(float(row.sum()), 0.95)

    def test_tie_fix_applied_before_pit(self):
        # The PIT bands use the RENORMALIZED CDF: with tie mass at 0, the
        # renormalized mass at ±1 is larger than the raw convolution, so
        # U for a game with y=0 sits at the renormalized F(−1) (not the raw
        # one) — i.e. the tie fix enters F̂, not just the recorded table.
        lam_h, lam_a, al = 1.6, 1.2, mrd.ALPHA_FLOOR + 0.2
        grid = list(range(-3, 4))
        pmf = mrd._nb_score_pmf(np.array([lam_h]), np.array([lam_a]),
                                np.array([al]), np.array([al]), grid)
        F = np.cumsum(pmf[0])
        # y = 0: U = F(−1) + V·p̂(0) = F(−1) (p̂(0)=0), i.e. U sits exactly
        # at the renormalized CDF just below 0.
        U = mrd._randomized_pit(pmf, grid, np.array([0]), seed=5)
        self.assertAlmostEqual(U[0], F[grid.index(-1)], places=12)
        self.assertGreater(F[grid.index(-1)], 0.0)


class TestCrps(unittest.TestCase):
    """Discrete CRPS on toy cases (Gneiting & Raftery 2007 discrete form)."""

    def test_point_mass_crps_zero(self):
        margins = list(range(-3, 4))
        y = np.array([-1, 0, 2])
        pmf = np.zeros((3, len(margins)))
        for i, yi in enumerate(y):
            pmf[i, margins.index(yi)] = 1.0
        self.assertAlmostEqual(mrd._crps(pmf, margins, y), 0.0, places=12)

    def test_fair_two_point_toy(self):
        # Margins {0, 1}, p = (0.5, 0.5), y = 0:
        #   CRPS = Σ_m (F(m) − 1{y ≤ m})² over the grid m ∈ {0, 1}:
        #     m=0: (0.5 − 1)² = 0.25 ;  m=1: (1 − 1)² = 0
        #   → CRPS = 0.25. Same for y = 1 (symmetric). Mean over the pair
        #   is 0.25.
        margins = [0, 1]
        pmf = np.array([[0.5, 0.5], [0.5, 0.5]])
        y = np.array([0, 1])
        self.assertAlmostEqual(mrd._crps(pmf, margins, y), 0.25, places=12)

    def test_more_spread_forecast_has_higher_crps(self):
        # On the same realized y = 0, a 50/50 forecast must have a higher
        # CRPS than a 90/10 forecast pointing at the outcome.
        margins = [0, 1]
        good = np.array([[0.9, 0.1]])
        wide = np.array([[0.5, 0.5]])
        y = np.array([0])
        self.assertGreater(mrd._crps(wide, margins, y),
                           mrd._crps(good, margins, y))


if __name__ == "__main__":
    unittest.main()
