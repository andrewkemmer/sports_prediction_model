"""Tests for run_game_structure_diagnostic.py — game-structure-aware
simulation diagnostic (last-bat hypothesis).

Covers:
  1. The game-flow resolution table (pure function): home leads after
     the top of the 9th -> home wins with margin h8 - a9; home trails/
     ties and walks off -> home wins by EXACTLY one run (MLB rule);
     tied after 9 -> extras resolved with p_home_extras and the
     empirical extras finals; home can't catch up -> away wins.
  2. The extras-credit identity: structured P(home win) ==
     P(full_h > a9) + P(full_h == a9) * p_home_extras, and the
     game-flow truncation changes only the totals/run-line surfaces.
  3. Marginal consistency: the exchangeability split (H8 ~ Binomial(
     full_h, 8/9), h9 = full_h - H8) sums exactly to the fitted
     marginal, and the current arm matches the shipped independent-NB
     sampler (derive_markets_mc) within MC tolerance on a synthetic
     frame.
"""
from __future__ import annotations

import unittest

import numpy as np

from run_engine import TOTAL_LINE_GRID, _nb_size_prob
from run_game_structure_diagnostic import (BOTTOM9_SHARE, MC_N, GS_SEED,
                                           draw_game_flow_chunk,
                                           resolve_game_flow,
                                           run_simulation)
from run_engine import derive_markets_mc


def _rng() -> np.random.Generator:
    return np.random.default_rng(GS_SEED)


class TestResolveGameFlow(unittest.TestCase):
    """Canonical per-draw cases from the MLB rules (exact, p_extras=1 so
    the extras outcome is forced home for the win check)."""

    def test_home_leads_after_top9_wins_with_margin(self):
        # h8 = 5 > a9 = 4: home is leading -> game over, margin 1.
        full_h = np.array([[5]])
        a9 = np.array([[4]])
        h8 = np.array([[5]])
        won, margin, total = resolve_game_flow(
            full_h, a9, h8, 1.0, _rng(), np.array([1.0]), np.array([1.0]))
        self.assertTrue(won[0, 0])
        self.assertEqual(margin[0, 0], 1)
        self.assertEqual(total[0, 0], 9)

    def test_walkoff_wins_are_exactly_one_run(self):
        # h8 = 3, a9 = 4 (trailing by 1), h9 = 3 -> scores the tying + go-
        # ahead runs, game ends at the go-ahead run: wins by exactly 1.
        full_h = np.array([[6]])   # h9 = 6 - 3 = 3 > d = 1
        a9 = np.array([[4]])
        h8 = np.array([[3]])
        won, margin, total = resolve_game_flow(
            full_h, a9, h8, 1.0, _rng(), np.array([1.0]), np.array([1.0]))
        self.assertTrue(won[0, 0])
        self.assertEqual(margin[0, 0], 1)          # never more than 1
        self.assertEqual(total[0, 0], 2 * 4 + 1)   # home runs = a9 + 1

    def test_tied_after_9_goes_to_extras(self):
        # h8 = 3, a9 = 5 (trailing by 2), h9 = 2 -> ties it: extras.
        full_h = np.array([[5]])
        a9 = np.array([[5]])
        h8 = np.array([[3]])
        extras_home = np.array([7.0])
        extras_away = np.array([6.0])
        # p_extras = 1 -> home wins the extras game (7-6).
        won, margin, total = resolve_game_flow(
            full_h, a9, h8, 1.0, _rng(), extras_home, extras_away)
        self.assertTrue(won[0, 0])
        self.assertEqual(margin[0, 0], 1)
        self.assertEqual(total[0, 0], 13)          # empirical extras final
        # p_extras = 0 -> home loses the extras game.
        won0, _, _ = resolve_game_flow(full_h, a9, h8, 0.0, _rng(),
                                       extras_home, extras_away)
        self.assertFalse(won0[0, 0])

    def test_home_fails_to_catch_up_away_wins(self):
        # h8 = 3, a9 = 5, h9 = 1 (< d = 2): away wins 5-4.
        full_h = np.array([[4]])
        a9 = np.array([[5]])
        h8 = np.array([[3]])
        won, margin, total = resolve_game_flow(
            full_h, a9, h8, 1.0, _rng(), np.array([1.0]), np.array([1.0]))
        self.assertFalse(won[0, 0])
        self.assertEqual(margin[0, 0], -1)
        self.assertEqual(total[0, 0], 9)

    def test_tie9_is_exactly_full_h_equals_a9_with_h8_leq_a9(self):
        """The extras event coincides with the full-total tie event."""
        rng = _rng()
        n, d = 2000, 4000
        full_h = rng.poisson(4.4, size=(n, d)).astype(int)
        a9 = rng.poisson(4.3, size=(n, d)).astype(int)
        h8 = rng.binomial(full_h, BOTTOM9_SHARE).astype(int)
        tie = full_h == a9
        h9 = full_h - h8
        tie9 = (h8 <= a9) & (h9 == a9 - h8)
        self.assertTrue(np.array_equal(tie, tie9))


class TestStructuredIdentity(unittest.TestCase):
    """Structured P(home win) == P(h > a) + P(h == a) * p_extras (the
    analytic identity); the truncation only moves totals/run-line."""

    def test_identity_holds_within_mc_tolerance(self):
        rng = _rng()
        n, d = 500, 8000
        full_h = rng.poisson(4.4, size=(n, d)).astype(int)
        a9 = rng.poisson(4.3, size=(n, d)).astype(int)
        h8 = rng.binomial(full_h, BOTTOM9_SHARE).astype(int)
        p_extras = 0.53
        won, m_gf, t_gf = resolve_game_flow(
            full_h, a9, h8, p_extras, rng,
            np.array([8.0, 7.0, 9.0]), np.array([6.0, 7.0, 8.0]))
        p_str = won.mean(axis=1)
        p_cur = (full_h > a9).mean(axis=1)
        p_tie = (full_h == a9).mean(axis=1)
        analytic = p_cur + p_tie * p_extras
        self.assertLessEqual(float(np.abs(p_str - analytic).max()), 0.02)

    def test_current_totals_unchanged_structured_differs(self):
        """Same draws: current totals = full_h + a9; structured totals are
        truncated (home skips a phantom 9th / stops at the walk-off). The
        ONLY draws where the structured total exceeds the naive sum are
        extras (tie after 9), where real extras runs are added back."""
        rng = _rng()
        n, d = 500, 8000
        full_h = rng.poisson(4.4, size=(n, d)).astype(int)
        a9 = rng.poisson(4.3, size=(n, d)).astype(int)
        h8 = rng.binomial(full_h, BOTTOM9_SHARE).astype(int)
        h9 = full_h - h8
        tie9 = (h8 <= a9) & (h9 == a9 - h8)
        _, _, t_gf = resolve_game_flow(
            full_h, a9, h8, 0.53, rng,
            np.array([8.0, 7.0, 9.0]), np.array([6.0, 7.0, 8.0]))
        t_cur = full_h + a9
        # Truncation (un-batted bottom 9th / walk-off stop) never adds runs;
        # extras resampling legitimately adds real extras runs.
        self.assertTrue(np.all(t_gf[~tie9] <= t_cur[~tie9]))
        # The truncation is material: a meaningful share of non-extras
        # draws lost runs (home led after the top of the 9th / walked off).
        self.assertGreater(float((t_gf[~tie9] < t_cur[~tie9]).mean()), 0.05)


class TestSamplerConsistency(unittest.TestCase):
    """The split-based sampler reproduces the shipped independent-NB MC."""

    def _frame(self, n: int = 300):
        rng = np.random.default_rng(11)
        lam_h = rng.uniform(3.6, 5.2, n)
        lam_a = rng.uniform(3.6, 5.2, n)
        a_h = np.full(n, 0.08)
        a_a = np.full(n, 0.10)
        return lam_h, lam_a, a_h, a_a

    def test_current_arm_matches_shipped_sampler(self):
        lam_h, lam_a, a_h, a_a = self._frame()
        sim = run_simulation(lam_h, lam_a, a_h, a_a, 0.53,
                             np.array([8.0, 7.0]), np.array([6.0, 7.0]))
        shipped = derive_markets_mc(lam_h, lam_a, a_h, a_a,
                                    n_draws=MC_N)
        # Both arms estimate the SAME quantity: home-win probability
        # conditioned on the game resolving (no tie). sim["p_win_current"]
        # is the structured sampler's RAW regulation win rate P(full_h > a9)
        # and sim["p_tie"] its tie rate; renormalizing on no tie the same way
        # derive_markets_mc does (P(margin>0) / (1 - P0)) recovers the
        # comparable home-win probability. The shipped renormalized
        # p_home_win_derived uses an INDEPENDENT seed, so per-game se at 10k
        # draws ~ 0.005 for raw and tie, and the 1/(1 - P0) denominator
        # amplifies the max over 300 games to ~0.03–0.04; 0.05 tolerates that
        # while still catching a real divergence (the tie-fix moved
        # p_home_win_derived ~11%, and a broken renormalization surfaces as
        # a gap > 0.05).
        p0 = np.maximum(sim["p_tie"].to_numpy(), 1e-9)
        renorm = sim["p_win_current"].to_numpy() / (1.0 - p0)
        gap = float(np.abs(renorm
                           - shipped["p_home_win_derived"]).max())
        self.assertLessEqual(gap, 0.05)

    def test_split_sums_exactly_to_the_marginal(self):
        """H8 + h9 == full_h by construction; the marginal mean is
        preserved through the split."""
        rng = _rng()
        n, d = 200, 20000
        lam = np.full(n, 4.4)
        alpha = np.full(n, 0.08)
        nh, ph = _nb_size_prob(lam[:, None], alpha[:, None])
        full_h = rng.negative_binomial(nh, ph, size=(n, d)).astype(int)
        h8 = rng.binomial(full_h, BOTTOM9_SHARE).astype(int)
        h9 = full_h - h8
        self.assertTrue(np.array_equal(h8 + h9, full_h))
        self.assertLessEqual(float(abs(h8.mean() - lam.mean() * BOTTOM9_SHARE)),
                             0.05)
        self.assertLessEqual(float(abs(h9.mean() - lam.mean() * (1 - BOTTOM9_SHARE))),
                             0.05)


class TestTotalsSurfaceLines(unittest.TestCase):
    def test_line_grid_assignment_uses_rounded_lambda_total(self):
        from run_engine import _rounded_total_line
        self.assertIn(_rounded_total_line(4.2, 4.3), TOTAL_LINE_GRID)
        self.assertAlmostEqual(_rounded_total_line(4.2, 4.3), 8.5)


if __name__ == "__main__":
    unittest.main()
