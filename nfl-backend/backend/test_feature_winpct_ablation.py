"""Tests for the win_pct_diff removal arm (run_feature_winpct_ablation.py).

- Arm composition: WITH_14 is the deployed 14; WITHOUT_13 is exactly that
  minus win_pct_diff; columns absent from the frame are dropped.
- The deployed list mirrors FEATURE_COLUMNS (14 + is_home anchor) so the arm
  always tests the TRUE served pool.

Run: python -m unittest test_feature_winpct_ablation -v
"""

from __future__ import annotations

import unittest

import pandas as pd

import nfl_features as nf
from run_feature_winpct_ablation import (DEPLOYED_14, REMOVED, build_arms)


class TestWinpctArms(unittest.TestCase):
    def test_deployed_pool_mirrors_feature_columns(self):
        served = [f for f in nf.FEATURE_COLUMNS if f != "is_home"]
        self.assertEqual(DEPLOYED_14, served)
        self.assertEqual(len(DEPLOYED_14), 14)

    def test_without_is_deployed_minus_win_pct_diff(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_14}))
        self.assertEqual(arms["WITH_14"], DEPLOYED_14)
        self.assertEqual(len(arms["WITHOUT_13"]), 13)
        self.assertEqual(set(DEPLOYED_14) - set(arms["WITHOUT_13"]),
                         {REMOVED})

    def test_absent_columns_dropped(self):
        feats = pd.DataFrame({c: [0.0] for c in DEPLOYED_14
                              if c != REMOVED})
        arms = build_arms(feats)
        self.assertEqual(len(arms["WITH_14"]), 13)
        self.assertEqual(len(arms["WITHOUT_13"]), 13)


if __name__ == "__main__":
    unittest.main()