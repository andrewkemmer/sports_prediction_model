"""Tests for the minimum validation-fold-size guard (MIN_VAL_FOLD_GAMES).

Tiny folds — postseason tails, offseason gaps — produce wild metric swings
(e.g. AUC 0.18 on 11 games) that pollute pooled metrics and the adaptive
blend weights. walk_forward_evaluate must skip them by default.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.training import walk_forward_evaluate


def _synthetic_games(n_days: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 3, 1)
    rows = []
    pk = 0
    for d in range(n_days):
        day = start + timedelta(days=d)
        for _ in range(8):  # 8 games/day → ~56 per 7-day fold (above the 40 floor)
            pk += 1
            p_home = 0.55 + rng.normal(0, 0.05)
            rows.append({
                "game_pk": pk,
                "game_date": day,
                "home_team": "NYY", "away_team": "BOS",
                "home_win": float(rng.random() < p_home),
                "elo_diff": rng.normal(0, 20),
                "sp_era_diff": rng.normal(0, 0.5),
                "lineup_woba_mean_diff": rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


class TestMinFoldGuard(unittest.TestCase):
    def test_small_folds_skipped_by_default(self):
        """Default guard drops folds under MIN_VAL_FOLD_GAMES."""
        df = _synthetic_games()
        _, _, preds = walk_forward_evaluate(
            df, retrain_cadence_days=7, min_train_days=0
        )
        if preds.empty:
            self.fail("all folds skipped — guard too aggressive")
        sizes = preds.groupby("fold_idx").size()
        self.assertTrue((sizes >= 40).all(),
                        f"fold below minimum survived: {sizes.to_dict()}")

    def test_min_val_games_zero_keeps_all_folds(self):
        """opt-in override restores the legacy keep-everything behavior."""
        df = _synthetic_games()
        _, _, preds = walk_forward_evaluate(
            df, retrain_cadence_days=7, min_train_days=0, min_val_games=0
        )
        sizes_default = None
        _, _, preds_guarded = walk_forward_evaluate(
            df, retrain_cadence_days=7, min_train_days=0
        )
        sizes_default = preds.groupby("fold_idx").size()
        sizes_guarded = preds_guarded.groupby("fold_idx").size()
        self.assertGreaterEqual(len(sizes_default), len(sizes_guarded))


if __name__ == "__main__":
    unittest.main()
