"""Tests for the corr-pair removal ablation (run_feature_corr_ablation.py).

- WITH_12 is the deployed 12; WITHOUT_YPP is exactly that minus the
  remaining twin; the WITHOUT_QBEPA arm was executed (ADOPT-REMOVE,
  cd3c26b) and ewm_qb_epa_play_diff is no longer in the deployed list.
- The deployed list mirrors FEATURE_COLUMNS (12 + is_home anchor) —
  market_home_implied is policy-reverted out (2026-09-01), so it is NOT part
  of the deployed list here.
- Columns absent from the frame are dropped (never silently all-NaN).

Run: python -m unittest test_feature_corr_ablation -v
"""

from __future__ import annotations

import unittest

import pandas as pd

import nfl_features as nf
from run_feature_corr_ablation import (QBEPA, YPP, build_arms)
from run_feature_winpct_ablation import DEPLOYED_12


class TestCorrArms(unittest.TestCase):
    def test_deployed_pool_mirrors_feature_columns(self):
        served = [f for f in nf.FEATURE_COLUMNS if f != "is_home"]
        self.assertEqual(DEPLOYED_12, served)
        self.assertEqual(len(DEPLOYED_12), 12)
        self.assertNotIn("market_home_implied", DEPLOYED_12)
        self.assertNotIn(QBEPA, DEPLOYED_12)   # verdict executed (cd3c26b)

    def test_with_12_is_unmodified_baseline(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_12}))
        self.assertEqual(arms["WITH_12"], DEPLOYED_12)

    def test_ypp_removal_drops_exactly_ypp(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_12}))
        self.assertEqual(len(arms["WITHOUT_YPP"]), 11)
        self.assertEqual(set(DEPLOYED_12) - set(arms["WITHOUT_YPP"]), {YPP})

    def test_absent_columns_dropped_never_all_nan(self):
        feats = pd.DataFrame({c: [0.0] for c in DEPLOYED_12
                              if c != YPP})
        arms = build_arms(feats)
        self.assertEqual(len(arms["WITH_12"]), 11)
        self.assertEqual(len(arms["WITHOUT_YPP"]), 11)


if __name__ == "__main__":
    unittest.main()
