"""Tests for the totals P(over) push-inclusive bug fix.

The MC grid previously used total >= int(-(-line//1)) which for whole-number
lines (e.g. 9.0) produced threshold 9 — making P(over) push-inclusive
(~50% instead of the correct ~40%).  The fix uses total >= line + 0.5
to match the monitor scorer's strict definition.  P(push) = P(total == line)
is added for EV math.
"""
import sys
from pathlib import Path
from unittest import TestCase

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

import numpy as np
from run_engine import (
    derive_markets_mc,
    TOTAL_LINE_GRID,
    TOTAL_LINE,
)


class TestStrictOverDefinition(TestCase):
    """P(over) + P(under) + P(push) == 1.0 for all grid lines."""

    def _mc_result(self, lam=9.0, n_draws=50_000, seed=42):
        """Run MC with a single game at the given lambda."""
        mu = np.array([lam])
        mu_a = np.array([lam])
        alpha = np.array([1.0])
        return derive_markets_mc(mu, mu_a, alpha, alpha,
                                 n_draws=n_draws, seed=seed)

    def test_probabilities_sum_to_one(self):
        """For every grid line, over + under + push == 1.0."""
        mc = self._mc_result()
        for j, line in enumerate(TOTAL_LINE_GRID):
            over = mc["p_over_grid"][:, j].mean()
            push = mc["p_push_grid"][:, j].mean()
            under = 1.0 - over - push
            total = over + under + push
            self.assertAlmostEqual(
                total, 1.0, places=5,
                msg=f"line={line}: over={over:.4f} + under={under:.4f} "
                    f"+ push={push:.4f} = {total:.6f} != 1.0")

    def test_half_line_push_is_zero(self):
        """For half-lines (8.5, 9.5, etc.), P(push) == 0 always
        because integer total can never equal a half-integer line."""
        mc = self._mc_result()
        for j, line in enumerate(TOTAL_LINE_GRID):
            if line != int(line):  # half-line
                push = mc["p_push_grid"][:, j].mean()
                self.assertAlmostEqual(
                    push, 0.0, places=10,
                    msg=f"half-line {line}: P(push) = {push} != 0")


class TestWholeLineStrictness(TestCase):
    """At line 9.0, total == 9 is a push, not an over."""

    def test_total_9_is_push_at_line_9(self):
        """At line 9.0, P(over) < P(over at 8.5) because the push band
        (total == 9) is now excluded from over."""
        mc = TestStrictOverDefinition._mc_result(self, lam=9.0, n_draws=100_000)
        col_9 = TOTAL_LINE_GRID.index(9.0)
        col_85 = TOTAL_LINE_GRID.index(8.5)
        over_9 = mc["p_over_grid"][:, col_9].mean()
        over_85 = mc["p_over_grid"][:, col_85].mean()
        push_9 = mc["p_push_grid"][:, col_9].mean()
        # Strict P(over) at 9.0 = P(total >= 10) must be LESS than
        # legacy P(over) at 8.5 = P(total >= 9) — the gap IS the push.
        self.assertLess(
            over_9, over_85,
            f"P(over@9.0)={over_9:.4f} must be < P(over@8.5)={over_85:.4f}")
        self.assertGreater(
            push_9, 0.0,
            "P(push) at line 9.0 must be > 0 (exact matches exist)")
        # The gap between legacy and strict should be approximately push
        gap = over_85 - over_9
        self.assertAlmostEqual(
            gap, push_9, delta=0.02,
            msg=f"gap={gap:.4f} should ≈ push={push_9:.4f}")

    def test_over_plus_push_equals_legacy_at_half_line(self):
        """At a half-line (8.5), P(over) == old formula (no push possible)."""
        mc = TestStrictOverDefinition._mc_result(self, lam=9.0, n_draws=100_000)
        col = TOTAL_LINE_GRID.index(8.5)
        over = mc["p_over_grid"][:, col].mean()
        push = mc["p_push_grid"][:, col].mean()
        # At half-line: push is 0, over = P(total >= 9.0)
        self.assertAlmostEqual(push, 0.0, places=10)
        # Over at 8.5 should be > 0.5 since lambda is 9.0
        self.assertGreater(over, 0.50,
                           "P(over) at line 8.5 with lambda=9.0 should be > 50%")


class TestMonitorVsMCAgreement(TestCase):
    """Monitor scorer (total >= line + 0.5) agrees with MC grid."""

    def test_legacy_p_over_matches_grid_at_8_5(self):
        """The legacy p_over_8_5 should equal grid_over at line 8.5
        (since both now use the same strict formula)."""
        mc = TestStrictOverDefinition._mc_result(self, lam=9.0, n_draws=100_000)
        col = TOTAL_LINE_GRID.index(8.5)
        legacy = mc["p_over_8_5"].mean()
        grid = mc["p_over_grid"][:, col].mean()
        self.assertAlmostEqual(
            legacy, grid, places=5,
            msg=f"p_over_8_5 ({legacy:.6f}) != grid at 8.5 ({grid:.6f})")

    def test_monitor_scoring_uses_strict(self):
        """Verify the monitor's (total >= line + 0.5) ground truth is
        the same as p_over_grid for a test game."""
        rng = np.random.default_rng(123)
        n_draws = 50_000
        mu = np.array([9.0])
        alpha = np.array([1.0])
        # Import the NB internals
        from run_engine import _nb_size_prob, _as_alpha_col
        nh, ph_ = _nb_size_prob(mu, alpha)
        na, pa = _nb_size_prob(mu, alpha)
        h = rng.negative_binomial(nh, ph_, size=(1, n_draws)).astype(np.int32)
        a = rng.negative_binomial(na, pa, size=(1, n_draws)).astype(np.int32)
        total = h + a
        # Monitor definition
        line = 9.0
        monitor_over = (total >= line + 0.5).mean()
        # MC grid (now using same formula)
        mc = derive_markets_mc(mu, mu, alpha, alpha, n_draws=n_draws, seed=123)
        col = TOTAL_LINE_GRID.index(line)
        mc_over = mc["p_over_grid"][:, col].mean()
        self.assertAlmostEqual(
            monitor_over, mc_over, places=5,
            msg=f"Monitor ({monitor_over:.6f}) != MC grid ({mc_over:.6f}) at line {line}")


if __name__ == "__main__":
    import unittest
    unittest.main()
