"""Tests for the corr-pair removal ablation (run_feature_corr_ablation.py).

- WITH_13 is the deployed 13; each WITHOUT_* arm is exactly that minus one
  twin; WITHOUT_BOTH drops both.
- The deployed list mirrors FEATURE_COLUMNS (13 + is_home anchor) —
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
from run_feature_winpct_ablation import DEPLOYED_13


class TestCorrArms(unittest.TestCase):
    def test_deployed_pool_mirrors_feature_columns(self):
        served = [f for f in nf.FEATURE_COLUMNS if f != "is_home"]
        self.assertEqual(DEPLOYED_13, served)
        self.assertEqual(len(DEPLOYED_13), 13)
        self.assertNotIn("market_home_implied", DEPLOYED_13)

    def test_with_13_is_unmodified_baseline(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_13}))
        self.assertEqual(arms["WITH_13"], DEPLOYED_13)

    def test_single_removals_drop_exactly_one_twin(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_13}))
        self.assertEqual(len(arms["WITHOUT_QBEPA"]), 12)
        self.assertEqual(set(DEPLOYED_13) - set(arms["WITHOUT_QBEPA"]),
                         {QBEPA})
        self.assertEqual(len(arms["WITHOUT_YPP"]), 12)
        self.assertEqual(set(DEPLOYED_13) - set(arms["WITHOUT_YPP"]), {YPP})

    def test_without_both_drops_both_twins(self):
        arms = build_arms(pd.DataFrame({c: [0.0] for c in DEPLOYED_13}))
        self.assertEqual(len(arms["WITHOUT_BOTH"]), 11)
        self.assertEqual(set(DEPLOYED_13) - set(arms["WITHOUT_BOTH"]),
                         {QBEPA, YPP})

    def test_absent_columns_dropped_never_all_nan(self):
        feats = pd.DataFrame({c: [0.0] for c in DEPLOYED_13
                              if c not in (QBEPA, YPP)})
        arms = build_arms(feats)
        self.assertEqual(len(arms["WITH_13"]), 11)
        self.assertEqual(len(arms["WITHOUT_QBEPA"]), 11)
        self.assertEqual(len(arms["WITHOUT_BOTH"]), 11)


if __name__ == "__main__":
    unittest.main()