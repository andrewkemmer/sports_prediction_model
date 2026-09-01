"""Tests for the win_pct_diff removal arm (run_feature_winpct_ablation.py).

- Arm composition: WITH_12 is the deployed 12; WITHOUT_11 is exactly that
  minus win_pct_diff; columns absent from the frame are dropped.
- The deployed list mirrors FEATURE_COLUMNS (12 + is_home anchor) so the arm
  always tests the TRUE served pool (market_home_implied policy-reverted out
  and ewm_qb_epa_play_diff removed by the corr-pair verdict, both 2026-09-01,
  so neither is part of the deployed list here).
"""

from __future__ import annotations

import unittest

import pandas as pd

import nfl_features as nf
from run_feature_winpct_ablation import (DEPLOYED_12, REMOVED, build_arms)


class TestWinpctArms(unittest.TestCase):
    def test_deployed_pool_mirrors_feature_columns(self):
        served = [f for f in nf.FEATURE_COLUMNS if f != "is_home"]
        self.assertEqual(DEPLOYED_12, served)
        self.assertEqual(len(DEPLOYED_12), 12)
        self.assertNotIn("market_home_implied", DEPLOYED_12)

    def test_without_is_deployed_minus_win_pct_diff(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_12}))
        self.assertEqual(arms["WITH_12"], DEPLOYED_12)
        self.assertEqual(len(arms["WITHOUT_11"]), 11)
        self.assertEqual(set(DEPLOYED_12) - set(arms["WITHOUT_11"]),
                         {REMOVED})

    def test_absent_columns_dropped(self):
        feats = pd.DataFrame({c: [0.0] for c in DEPLOYED_12
                              if c != REMOVED})
        arms = build_arms(feats)
        self.assertEqual(len(arms["WITH_12"]), 11)
        self.assertEqual(len(arms["WITHOUT_11"]), 11)


if __name__ == "__main__":
    unittest.main()
