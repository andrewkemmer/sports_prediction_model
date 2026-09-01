"""Tests for the win_pct_diff removal arm (run_feature_winpct_ablation.py).

- Arm composition: WITH_13 is the deployed 13; WITHOUT_12 is exactly that
  minus win_pct_diff; columns absent from the frame are dropped.
- The deployed list mirrors FEATURE_COLUMNS (13 + is_home anchor) so the arm
  always tests the TRUE served pool (market_home_implied policy-reverted out
  on 2026-09-01, so it is NOT part of the deployed list here).
"""

from __future__ import annotations

import unittest

import pandas as pd

import nfl_features as nf
from run_feature_winpct_ablation import (DEPLOYED_13, REMOVED, build_arms)


class TestWinpctArms(unittest.TestCase):
    def test_deployed_pool_mirrors_feature_columns(self):
        served = [f for f in nf.FEATURE_COLUMNS if f != "is_home"]
        self.assertEqual(DEPLOYED_13, served)
        self.assertEqual(len(DEPLOYED_13), 13)
        self.assertNotIn("market_home_implied", DEPLOYED_13)

    def test_without_is_deployed_minus_win_pct_diff(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_13}))
        self.assertEqual(arms["WITH_13"], DEPLOYED_13)
        self.assertEqual(len(arms["WITHOUT_12"]), 12)
        self.assertEqual(set(DEPLOYED_13) - set(arms["WITHOUT_12"]),
                         {REMOVED})

    def test_absent_columns_dropped(self):
        feats = pd.DataFrame({c: [0.0] for c in DEPLOYED_13
                              if c != REMOVED})
        arms = build_arms(feats)
        self.assertEqual(len(arms["WITH_13"]), 12)
        self.assertEqual(len(arms["WITHOUT_12"]), 12)


if __name__ == "__main__":
    unittest.main()