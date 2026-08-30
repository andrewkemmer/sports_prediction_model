"""OOF run-margin feature tests (moneyline margin feature, SHIPPED).

Covers the 2026-08 margin-feature task's crux requirements:
- Leakage guard: every predicted game sits STRICTLY AFTER its fold's training
  window (positive case holds; a violating split raises AssertionError).
- The margin table covers exactly the union of executed folds' val windows;
  games outside any executed fold get NO row (counted as uncovered → NaN →
  the moneyline's train-median imputation path downstream).
- Same folds + seed → byte-identical table (determinism).
- refit_run_margins: fit-only refit at FIXED round counts, finite λs,
  margin ≡ lam_home − lam_away, one prediction row per requested game.
- Production wiring (post-gate SHIP): FEATURE_COLS is 65 with
  run_margin_diff IN it; walk_forward_evaluate attaches leakage-free OOF
  margins on its own folds when the frame carries run-engine inputs, keeps
  the fold geometry identical, and degrades to an all-NaN column with a loud
  warning (never a crash) on frames without scores; derive_run_features
  still drops run_margin_diff from the run view (the run engine stays
  READ-ONLY, 29-feature keep-list unchanged); α(λ)/MC path untouched.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import build_oof_margin as bom
from run_engine import derive_run_features
from training import FEATURE_COLS


# Test-visible min-val gate so fixtures produce deterministic fold sets
# regardless of production config tuning.
MIN_FOLD_VAL_GAMES_TEST = 6


def _synthetic_games(n_days: int = 80, per_day: int = 6,
                     seed: int = 42) -> pd.DataFrame:
    """Decided games whose scores depend on a real signal column so the
    Poisson fits learn something non-degenerate."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-04-01", periods=n_days, freq="D")
    rows = []
    teams = [f"T{i:02d}" for i in range(30)]
    pk = 800_000
    for d in dates:
        for _ in range(per_day):
            home, away = rng.choice(teams, 2, replace=False)
            home_edge = float(rng.normal(0, 1))
            hs = max(0, int(rng.poisson(4.4 + 0.5 * home_edge)))
            as_ = max(0, int(rng.poisson(4.4 - 0.5 * home_edge)))
            rows.append({
                "game_pk": pk, "game_date": d, "home_team": home,
                "away_team": away,
                "home_win": float(hs > as_),
                "home_score": hs, "away_score": as_,
                # Real signal for both side models (levels, not diffs).
                "sp_era_home": float(rng.normal(4.0, 0.5) - 0.3 * home_edge),
                "sp_era_away": float(rng.normal(4.0, 0.5) + 0.3 * home_edge),
                "woba_30g_home": float(rng.normal(0.320, 0.02)),
                "woba_30g_away": float(rng.normal(0.315, 0.02)),
                "is_home": 1.0,
            })
            pk += 1
    return pd.DataFrame(rows)


def _manual_folds(games: pd.DataFrame, cadence_days: int = 10,
                  min_train_days: int = 20) -> list[dict]:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import training

    splits = training.walk_forward_splits(
        games, retrain_cadence_days=cadence_days,
        min_train_days=min_train_days)
    return [s for s in splits
            if len(s["val_games"]) >= MIN_FOLD_VAL_GAMES_TEST]


class TestLeakageGuard(unittest.TestCase):
    def test_every_predicted_game_strictly_after_training_window(self):
        games = _synthetic_games()
        folds = _manual_folds(games)
        self.assertGreaterEqual(len(folds), 3, "fixture must yield folds")
        margins, _rounds, _uncov = bom.oof_run_margins(games, folds)

        covered = set(margins["game_pk"])
        expected = set().union(*(set(f["val_games"]["game_pk"]) for f in folds))
        self.assertEqual(covered, expected,
                         "margin table must cover exactly the val windows")
        # Margin identity holds row-wise.
        np.testing.assert_allclose(
            margins[bom.MARGIN_COL].to_numpy(),
            (margins["lam_home"] - margins["lam_away"]).to_numpy(), atol=1e-9)
        # All λs positive (Poisson clip floor).
        self.assertTrue((margins["lam_home"] > 0).all())
        self.assertTrue((margins["lam_away"] > 0).all())

    def test_violating_split_raises(self):
        games = _synthetic_games(n_days=40)
        good = _manual_folds(games)[0]
        bad = dict(good)
        bad["fold_idx"] = 999
        # Overlap: validation window moved INTO the training window.
        bad["val_games"] = pd.concat(
            [good["train_games"].tail(5), good["val_games"]],
            ignore_index=True)
        with self.assertRaises(AssertionError):
            bom.oof_run_margins(games, [bad])


class TestUncoveredGames(unittest.TestCase):
    def test_games_outside_executed_folds_are_uncovered_not_fabricated(self):
        games = _synthetic_games(n_days=60)
        folds = _manual_folds(games)
        # Drop the earliest executed fold → its val games lose coverage.
        trimmed = folds[1:]
        margins, _rounds, uncov = bom.oof_run_margins(games, trimmed)
        lost = set(folds[0]["val_games"]["game_pk"])
        self.assertGreater(uncov, 0, "dropped fold's games must be uncovered")
        # Uncovered = every decided game NOT in any executed fold's val
        # window: the dropped fold's games PLUS the pre-first-fold warm-up
        # rows (never covered by construction).
        covered = set(margins["game_pk"])
        expected_uncov = len(games) - len(covered)
        self.assertEqual(uncov, expected_uncov)
        self.assertEqual(len(covered & lost), 0,
                         "a dropped fold's games must NOT gain coverage")
        self.assertTrue(margins["game_pk"].is_unique)


class TestDeterminism(unittest.TestCase):
    def test_same_folds_identical_table(self):
        games = _synthetic_games(seed=7)
        folds_a = _manual_folds(games)
        folds_b = [
            {"train_games": f["train_games"].copy(),
             "val_games": f["val_games"].copy(),
             "fold_idx": f["fold_idx"],
             "val_start": f["val_start"], "val_end": f["val_end"]}
            for f in folds_a]
        a = bom.oof_run_margins(games, folds_a)
        b = bom.oof_run_margins(games, folds_b)
        self.assertEqual(a[2], b[2], "uncovered count must match")
        self.assertEqual(a[1], b[1], "median best-rounds must match")
        pd.testing.assert_frame_equal(
            a[0].reset_index(drop=True), b[0].reset_index(drop=True))


class TestRefitMargins(unittest.TestCase):
    def test_fixed_rounds_refit_contract(self):
        games = _synthetic_games(seed=11)
        folds = _manual_folds(games)
        margins, rounds, _uncov = bom.oof_run_margins(games, folds)
        cut = games["game_date"].max() - pd.Timedelta(days=5)
        tune_df = games[games["game_date"] <= cut].reset_index(drop=True)
        pred_df = games[games["game_date"] > cut].reset_index(drop=True)
        self.assertFalse(pred_df.empty)
        out = bom.refit_run_margins(tune_df, pred_df, rounds)
        self.assertEqual(len(out), len(pred_df))
        self.assertEqual(set(out["game_pk"]), set(pred_df["game_pk"]))
        self.assertTrue(np.isfinite(out["lam_home"]).all())
        self.assertTrue(np.isfinite(out["lam_away"]).all())
        self.assertTrue((out["lam_home"] > 0).all())
        np.testing.assert_allclose(
            out[bom.MARGIN_COL].to_numpy(),
            (out["lam_home"] - out["lam_away"]).to_numpy(), atol=1e-9)


class TestConfigRegressions(unittest.TestCase):
    def test_feature_cols_now_59_leakage_pruned(self):
        """Post-prune pin: the 6 lineup-delta features were removed from
        FEATURE_COLS (train-serve skew — actuals at train time, zeros at
        prediction time). FEATURE_COLS is 59; margin is still shipped."""
        self.assertEqual(len(FEATURE_COLS), 59)
        self.assertIn(bom.MARGIN_COL, FEATURE_COLS)

    def test_run_engine_stays_read_only_wrt_margin(self):
        """The run view drops run_margin_diff by the *_diff rule (the only
        survivor is park_factor_slug_diff) — the run engine cannot consume
        the margin, so the margin path cannot leak into itself. The 53-col
        keep-list (2026-08-30 restore) keeps run_margin_diff excluded."""
        feats, dropped = derive_run_features(list(FEATURE_COLS))
        self.assertNotIn(bom.MARGIN_COL, feats)
        self.assertIn(bom.MARGIN_COL, dropped)
        self.assertEqual(len(feats), 53)


class TestProductionWiring(unittest.TestCase):
    """walk_forward_evaluate / _attach_oof_run_margins enrichment (SHIPPED)."""

    def test_walk_forward_enriches_oof_margins_on_own_folds(self):
        """A frame WITH scores gets leakage-free OOF margins attached inside
        walk_forward_evaluate: the combined OOF output carries
        run_margin_diff with finite values on covered games, and the final
        fit-only refit's frame is the enriched one (no missing-column
        warning path)."""
        games = _synthetic_games(seed=3)
        _, _pooled, combined = _run_wfe(games)
        self.assertIn(bom.MARGIN_COL, combined.columns)
        self.assertGreater(combined[bom.MARGIN_COL].notna().mean(), 0.5,
                           "most OOF games must carry a real margin")
        self.assertTrue(np.isfinite(
            combined.loc[combined[bom.MARGIN_COL].notna(),
                         bom.MARGIN_COL]).all())

    def test_walk_forward_without_scores_nans_without_crashing(self):
        """Frames that cannot produce run-engine inputs (no scores — e.g.
        synthetic test frames) get an all-NaN margin column and a loud
        warning; training proceeds unchanged."""
        games = _synthetic_games(seed=5).drop(columns=["home_score", "away_score"])
        with self.assertLogs("training", level="WARNING") as logs:
            _, _pooled, combined = _run_wfe(games)
        self.assertTrue(any("run_margin_diff" in line for line in logs.output),
                        "missing inputs must warn naming the feature")
        self.assertIn(bom.MARGIN_COL, combined.columns)
        self.assertEqual(combined[bom.MARGIN_COL].isna().all(), True)

    def test_attach_oof_margins_keeps_fold_geometry_identical(self):
        """Regenerating the splits over the enriched frame must not change
        fold_idx/val_start/val game_pks (walk_forward_splits is a pure
        function of game_date/home_win); a desync raises AssertionError."""
        import training
        games = _synthetic_games(seed=9)
        splits = [s for s in training.walk_forward_splits(
            games, retrain_cadence_days=10) if len(s["val_games"]) >= 6]
        self.assertGreaterEqual(len(splits), 2)
        enriched, regen = training._attach_oof_run_margins(
            games, splits, min_val_games=6, max_eval_folds=0,
            retrain_cadence_days=10, min_train_days=0)
        self.assertEqual(len(regen), len(splits))
        for a, b in zip(regen, splits):
            self.assertEqual(a["fold_idx"], b["fold_idx"])
            self.assertEqual(pd.Timestamp(a["val_start"]),
                             pd.Timestamp(b["val_start"]))
            self.assertEqual(a["val_games"]["game_pk"].tolist(),
                             b["val_games"]["game_pk"].tolist())
        self.assertIn(bom.MARGIN_COL, enriched.columns)
        self.assertGreater(enriched[bom.MARGIN_COL].notna().mean(), 0.5)

    def test_margin_rounds_exposed_for_slate_refit(self):
        """The walk-forward margin build records the median fold round counts
        so the slate path can fit-only refit without a second run-engine OOF."""
        import training
        training.set_last_margin_rounds(None)
        self.assertIsNone(training.get_last_margin_rounds())
        games = _synthetic_games(seed=13)
        splits = [s for s in training.walk_forward_splits(
            games, retrain_cadence_days=10) if len(s["val_games"]) >= 6]
        training._attach_oof_run_margins(
            games, splits, min_val_games=6, max_eval_folds=0,
            retrain_cadence_days=10, min_train_days=0)
        rounds = training.get_last_margin_rounds()
        self.assertIsNotNone(rounds)
        self.assertIn("home", rounds)
        self.assertIn("away", rounds)
        self.assertGreater(rounds["home"], 0)

    def test_slate_attach_with_game_id_only_board(self):
        """Pre-game ESPN boards carry game_id (no game_pk) — the slate margin
        attach must synthesize game_pk from game_id (the 145d841 slate-key
        convention) so refit_run_margins AND the merge both work (v26
        regression: KeyError ('game_pk') left today's board without the
        shipped margin feature)."""
        import pipeline
        import training
        games = _synthetic_games(seed=21)
        # Board = the last 3 games, ESPN-style: game_id only, no game_pk.
        board = games.tail(3).drop(columns=["game_pk"]).copy()
        board["game_id"] = ["20260824_NYY@BOS", "20260824_LAD@SF",
                            "20260824_SF@LAD"]
        training.set_last_margin_rounds({"home": 8, "away": 8})
        out = pipeline._attach_slate_run_margins(board, games)
        self.assertIn(bom.MARGIN_COL, out.columns)
        # Every board row gets a REAL (non-NaN) margin — no silent loss.
        self.assertEqual(out[bom.MARGIN_COL].notna().sum(), len(board))
        # The board now carries game_pk derived from game_id.
        self.assertIn("game_pk", out.columns)
        self.assertEqual(out["game_pk"].tolist(), board["game_id"].tolist())


def _run_wfe(games):
    """Run walk_forward_evaluate and return (models, pooled, combined)."""
    import training
    # Save/restore module state so runs don't leak into other tests.
    prev_w = dict(training._LAST_ADAPTIVE_WEIGHTS)
    try:
        return training.walk_forward_evaluate(
            games, retrain_cadence_days=10, min_train_days=0, min_val_games=6)
    finally:
        training.set_adaptive_weights(prev_w)
        training.set_calibration(None)


if __name__ == "__main__":
    unittest.main()
