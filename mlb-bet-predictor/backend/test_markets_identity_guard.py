import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_engine


class TestMarketsIdentityGuard(unittest.TestCase):
    def test_saved_oof_has_zero_null_game_pk_on_real_artifact(self):
        """The f33b569 source-boundary filter held on the real 08-27 run.

        The previously-offending 15 identity-less rows (date + expected runs +
        scores only, no game_id) no longer exist in the shipped OOF, so a
        frame built from the real artifact persists with zero null keys.
        """
        source = pd.read_csv(
            Path(__file__).parents[1] / "data_delivery" / "run_engine_oof_20260827.csv")
        self.assertEqual(int(source["game_pk"].isna().sum()), 0)
        clean = source[source["game_pk"].notna()].copy()
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
            path = run_engine.persist_markets(markets, "20260827", {}, Path(td))
            self.assertTrue(path.exists())
            self.assertTrue(pd.read_csv(path)["game_pk"].notna().all())

    def test_identity_less_oof_rows_are_dropped_before_market_construction(self):
        """Regression for the 08-26 artifact's offending shape.

        Re-inject the 15 identity-less rows (game_pk NaN, no game_id — the
        exact rows that crashed persist_markets) and prove derive_markets_v3
        drops them with a loud warning before any market construction, so no
        NaN game_pk can ever reach persistence.
        """
        rng = np.random.default_rng(7)
        n = 60
        clean = pd.DataFrame({
            "game_date": pd.date_range("2026-07-20", periods=n, freq="D"),
            "game_pk": list(range(1000, 1000 + n)),
            "home_expected_runs": rng.uniform(3.8, 5.2, n),
            "away_expected_runs": rng.uniform(3.8, 5.2, n),
            "home_score": rng.poisson(4.5, n).astype(float),
            "away_score": rng.poisson(4.2, n).astype(float),
            "fold_idx": list(range(n)),
        })
        bad = pd.DataFrame({
            "game_date": pd.date_range("2026-07-01", periods=15, freq="D"),
            "home_expected_runs": [4.5] * 15, "away_expected_runs": [4.2] * 15,
            "home_score": [3] * 15, "away_score": [2] * 15, "fold_idx": 0,
        })
        oof = pd.concat([clean, bad], ignore_index=True)
        with self.assertLogs("run_engine", level="WARNING") as cm:
            res = run_engine.derive_markets_v3(oof, n_draws=20)
        joined = " ".join(cm.output)
        self.assertIn("dropping 15 identity-less OOF row(s)", joined)
        markets = res["markets"]
        self.assertEqual(int(markets["game_pk"].isna().sum()), 0)
        self.assertEqual(len(markets), n)


if __name__ == "__main__":
    unittest.main()
