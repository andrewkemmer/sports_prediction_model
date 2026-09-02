"""Regression pins: MLB walk-forward blend state is reset per walk.

The NFL codebase had an A/B-state bug (fixed 5fd0549): the fold-loop blend
reads the module-global adaptive weights, which are only WRITTEN at the END
of a walk — so any walk after the first in one process blended with the
PREVIOUS walk's adaptive weights instead of the static ENSEMBLE_WEIGHTS
priors (two identical consecutive walks measured 0.6312 then 0.6201 pooled
ll pre-fix). MLB has the same architecture (training._LAST_ADAPTIVE_WEIGHTS,
written at the end of walk_forward_evaluate from pooled OOF, read by
_member_weights during fold blending with a static-prior fallback when
empty) BUT walk_forward_evaluate clears the global at entry (a fix that
predates the NFL discovery), so later walks cannot see an earlier walk's
earned weights. These tests pin that invariant:

1. The entry clear happens even when the walk fails early (poisoned global
   is wiped before walk_forward_splits can raise).
2. Two identical consecutive walks in ONE process produce byte-identical
   pooled ll/auc/ece/brier surfaces (+ calibrated twins), and deliberately
   poisoning the global between walks does NOT change the next walk (the
   entry reset wipes it) — the decisive experiment behind the NFL finding.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd

from backend import training
from backend.training import walk_forward_evaluate


def _synthetic_games(n_days: int = 150, seed: int = 7) -> pd.DataFrame:
    """Small synthetic frame (mirrors test_min_fold_guard's) — the walk
    trains every member each fold, so ~40s for three walks."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 3, 1)
    rows = []
    pk = 0
    for d in range(n_days):
        day = start + timedelta(days=d)
        for _ in range(8):
            pk += 1
            p_home = 0.55 + rng.normal(0, 0.05)
            rows.append({
                "game_pk": pk, "game_date": day,
                "home_team": "NYY", "away_team": "BOS",
                "home_win": float(rng.random() < p_home),
                "elo_diff": rng.normal(0, 20),
                "sp_era_diff": rng.normal(0, 0.5),
                "lineup_woba_mean_diff": rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


_POOLED_KEYS = ("auc", "logloss", "brier", "ece",
                "logloss_calibrated", "brier_calibrated", "ece_calibrated")


class TestBlendStateReset(unittest.TestCase):
    """_LAST_ADAPTIVE_WEIGHTS must be cleared at walk entry so no walk can
    blend with a previous walk's (or a harness's) earned weights."""

    def tearDown(self) -> None:
        training._LAST_ADAPTIVE_WEIGHTS.clear()

    def test_entry_reset_even_on_early_failure(self):
        """Poisoned global is wiped before walk_forward_splits can raise —
        the reset is the walk's FIRST action, so no failure ordering can
        leave a later walk blended with stale weights."""
        training.set_adaptive_weights(
            {n: (1.0 if i == 0 else 0.0)
             for i, n in enumerate(training.ENSEMBLE_WEIGHTS)})
        df = _synthetic_games()
        with mock.patch.object(training, "walk_forward_splits",
                               side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                walk_forward_evaluate(df, retrain_cadence_days=7,
                                      min_train_days=0)
        self.assertEqual(training._LAST_ADAPTIVE_WEIGHTS, {},
                         "entry reset must clear the global before splits")

    def test_consecutive_walks_identical_and_poison_wiped(self):
        """Two identical walks in one process are byte-identical on the
        pooled surfaces, and a deliberately poisoned global between walks
        does not change the next walk — MLB has no cross-walk blend
        contamination (the NFL 5fd0549 A/B-state bug is absent)."""
        df = _synthetic_games()
        _, po1, p1 = walk_forward_evaluate(df, retrain_cadence_days=7,
                                           min_train_days=0)
        _, po2, p2 = walk_forward_evaluate(df, retrain_cadence_days=7,
                                           min_train_days=0)
        for k in _POOLED_KEYS:
            self.assertEqual(po1.get(k), po2.get(k),
                             f"pooled {k} differs across identical walks")
        # Per-game probabilities agree to float noise (< 1e-9); a blend
        # contamination would move them by ~1e-3.
        a = p1.sort_values("game_pk").reset_index(drop=True)
        b = p2.sort_values("game_pk").reset_index(drop=True)
        self.assertAlmostEqual(
            float((a["home_win_prob_model"] - b["home_win_prob_model"])
                  .abs().max()),
            0.0, places=9)

        # Poison the global as the NFL bug would have left it (walk 1's
        # earned weights ARE a poison for walk 2 without the reset).
        training.set_adaptive_weights(
            {n: (1.0 if i == 0 else 0.0)
             for i, n in enumerate(training.ENSEMBLE_WEIGHTS)})
        _, po3, _ = walk_forward_evaluate(df, retrain_cadence_days=7,
                                          min_train_days=0)
        for k in _POOLED_KEYS:
            self.assertEqual(po1.get(k), po3.get(k),
                             f"pooled {k} changed after a poisoned global "
                             "(entry reset must have wiped it)")


if __name__ == "__main__":
    unittest.main()
