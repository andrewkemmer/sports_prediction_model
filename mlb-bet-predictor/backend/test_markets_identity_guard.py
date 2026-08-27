import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_engine


class TestMarketsIdentityGuard(unittest.TestCase):
    def test_saved_oof_identity_less_rows_are_excluded_before_persist(self):
        source = pd.read_csv(Path(__file__).parents[1] / "data_delivery" / "run_engine_oof_20260826.csv")
        malformed = source[source["game_pk"].isna()]
        self.assertEqual(len(malformed), 15)
        clean = source[source["game_pk"].notna()].copy()
        # The real artifact has the exact offending shape. Reconstruct the
        # persistence contract from it and prove no null key can be written.
        markets = pd.DataFrame({
            "game_pk": clean["game_pk"].astype(int),
            "game_date": clean["game_date"],
            "kind": "oof",
            "home_expected_runs": clean["home_expected_runs"],
            "away_expected_runs": clean["away_expected_runs"],
            "alpha_home": 0.1, "alpha_away": 0.1,
            **{c: 0.5 for c in run_engine.MARKET_COLUMNS_V3
               if c.startswith(("p_over_", "p_under_", "p_home_cover_"))},
            "p_home_win_derived": 0.5, "p_away_win_derived": 0.5,
            "home_score": clean["home_score"], "away_score": clean["away_score"],
            "total_runs": clean["home_score"] + clean["away_score"],
            "ml_win_prob": np.nan, "agreement_conflict": False,
        })
        with tempfile.TemporaryDirectory() as td:
            path = run_engine.persist_markets(markets, "20260826", {}, Path(td))
            self.assertTrue(path.exists())
            self.assertTrue(pd.read_csv(path)["game_pk"].notna().all())


if __name__ == "__main__":
    unittest.main()
