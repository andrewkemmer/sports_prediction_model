"""NFL per-side final-score engine — step-1 tests (pure, no-network).

Covers ``nfl_per_side_engine`` (per-side mean regressors, OOF + fit-only
refill, residual persistence) and the step-1 runner conventions:
- Fold-aligned OOF: per-game preds exist ONLY for fold val rows, one row per
  game; warmup games never validated → counted uncovered.
- Leakage-assertion trigger: a fold whose val overlaps train raises
  AssertionError (mirror of the margin engine's boundary guard).
- Residual-artifact guard: persist_residuals FAILS LOUDLY (RuntimeError) on
  missing columns / empty frame / unwritable path; the written artifact is
  internally consistent (pred + resid == actual exactly).
- Imputation discipline: uncovered games are ABSENT (NaN on a keyed left
  merge — never zero-filled); the fit-only refill covers 100% of valid
  predicted rows.
- Determinism pin: identical folds + seed ⇒ byte-identical tables.
- Family-as-parameter: unknown family raises ValueError; rf/xgb walks honor
  the parameter and produce the same output shape.
- FEATURE_COLUMNS untouched: pred/resid columns are not in the served pool,
  SIDE_FEATURES == FEATURE_COLUMNS minus is_home, and the engine never
  imports the moneyline (feature-producer only).

Run: python -m unittest test_nfl_per_side -v   (no network needed)
"""
from __future__ import annotations

import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_features as nf
import nfl_moneyline as ml
import nfl_per_side_engine as pe

# Small-model test overrides (module globals read at call time).
pe.MAX_ROUNDS = 100
pe.EARLY_STOPPING_ROUNDS = 10
pe.RF_PARAMS = {**pe.RF_PARAMS, "n_estimators": 10}
pe.XGB_PARAMS = {**pe.XGB_PARAMS, "n_estimators": 100}

FEATS = ["f1", "f2", "elo_diff", "win_pct_diff"]


def _synth_games(n_seasons: int = 4, weeks: int = 8,
                 games_per_week: int = 6) -> pd.DataFrame:
    """Deterministic synthetic decided frame: seasons 2019.., weekly games,
    features correlated with per-side scores so the regressors can learn."""
    rng = np.random.default_rng(7)
    teams = [f"T{i}" for i in range(10)]
    rows = []
    gid = 1000
    for s in range(n_seasons):
        season = 2019 + s
        base = pd.Timestamp(f"{season}-09-01")
        mon = base + pd.Timedelta(days=(0 - base.weekday()) % 7)
        for w in range(weeks):
            gd = mon + pd.Timedelta(days=7 * w)
            for _g in range(games_per_week):
                home = teams[int(rng.integers(len(teams)))]
                away = teams[int(rng.integers(len(teams)))]
                while away == home:
                    away = teams[int(rng.integers(len(teams)))]
                f1 = float(rng.normal(0, 10))
                f2 = float(rng.normal(0, 1))
                margin = 2.0 * f1 - 3.0 * f2 + float(rng.normal(0, 3))
                hsc = int(round(21 + margin / 2 + rng.normal(0, 2)))
                asc = int(round(21 - margin / 2 + rng.normal(0, 2)))
                rows.append({
                    "game_id": f"G{gid}", "season": season, "week": w + 1,
                    "gameday": gd, "home_team": home, "away_team": away,
                    "home_score": max(hsc, 0), "away_score": max(asc, 0),
                    "f1": f1, "f2": f2, "elo_diff": 10 * f1,
                    "win_pct_diff": f2 * 0.1,
                })
                gid += 1
    df = pd.DataFrame(rows)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df


def _val_folds(games: pd.DataFrame,
               val_seasons: list[int] | None = None) -> list[dict]:
    """The moneyline's own weekly folds (the folds per-side must use)."""
    val_seasons = val_seasons or [2021, 2022]
    return ml.generate_weekly_folds(games, val_seasons=val_seasons)


class TestPerSideOof(unittest.TestCase):
    """Fold-aligned OOF per-side predictions + residuals."""

    def setUp(self):
        self.games = _synth_games()
        self.folds = _val_folds(self.games)
        self.out, self.rounds, self.n_uncov = pe.oof_per_side(
            self.folds, FEATS, self.games)

    def test_every_prediction_belongs_to_a_fold_val_row(self):
        """One row per covered game, only fold val rows; warmup uncovered."""
        val_ids = set()
        for f in self.folds:
            val_ids |= set(f["val"]["game_id"])
        self.assertGreater(len(self.out), 0)
        self.assertEqual(int(self.out["game_id"].duplicated().sum()), 0)
        self.assertTrue(set(self.out["game_id"]) <= val_ids)
        warmup = self.games[self.games["season"] <= 2020]
        self.assertEqual(self.n_uncov, int(len(warmup)))

    def test_both_sides_present_and_resid_consistent(self):
        """pred + resid == actual exactly (the artifact's contract)."""
        merged = self.games.merge(self.out, on="game_id", how="inner")
        for side, col in pe.SIDE_TARGETS.items():
            pred_col = pe.PRED_HOME if side == "home" else pe.PRED_AWAY
            resid_col = pe.RESID_HOME if side == "home" else pe.RESID_AWAY
            rebuilt = merged[pred_col] + merged[resid_col]
            np.testing.assert_allclose(rebuilt, merged[col].astype(float),
                                       rtol=0, atol=1e-9)

    def test_leakage_assertion_triggers(self):
        """A fold whose val overlaps train raises AssertionError."""
        g = self.games.sort_values("gameday").reset_index(drop=True)
        mid = len(g) // 2
        bad_folds = [{"week_start": pd.Timestamp("2021-09-01"),
                      "train": g.iloc[: mid + 3].copy(),
                      "val": g.iloc[mid:].copy()}]
        with self.assertRaises(AssertionError):
            pe.oof_per_side(bad_folds, FEATS, g)

    def test_read_only_no_fold_mutation(self):
        """The engine never mutates the caller's folds (READ-ONLY producer)."""
        import copy
        folds = copy.deepcopy(self.folds)
        _ = pe.oof_per_side(folds, FEATS, self.games)
        for f in folds:
            self.assertNotIn(pe.PRED_HOME, f["train"].columns)
            self.assertNotIn(pe.PRED_AWAY, f["val"].columns)
            self.assertNotIn(pe.RESID_HOME, f["train"].columns)

    def test_small_fold_val_left_uncovered(self):
        """A fold with <5 val rows is skipped: its games get no preds and are
        counted in the uncovered total (NaN downstream, never zero)."""
        g = self.games.sort_values("gameday").reset_index(drop=True)
        small = g[g["season"] == 2022].head(3).copy()
        train = g[g["gameday"] < small["gameday"].min()].copy()
        folds = [{"week_start": small["gameday"].min(),
                  "train": train, "val": small}]
        out, _r, n_uncov = pe.oof_per_side(folds, FEATS, g)
        self.assertEqual(len(out), 0)
        self.assertEqual(n_uncov, int(len(g)))

    def test_determinism_byte_identical(self):
        """Identical folds + seed ⇒ byte-identical per-side table."""
        out2, _r2, _u2 = pe.oof_per_side(self.folds, FEATS, self.games)
        a = self.out.sort_values("game_id").reset_index(drop=True)
        b = out2.sort_values("game_id").reset_index(drop=True)
        self.assertEqual(a.to_csv(index=False), b.to_csv(index=False))

    def test_uncovered_rows_nan_not_zero(self):
        """Imputation discipline: a keyed left merge leaves uncovered games
        NaN (never 0 / never median) — fill happens downstream in the model."""
        merged = self.games.merge(self.out, on="game_id", how="left")
        warmup = merged[merged["season"] <= 2020]
        self.assertEqual(int(warmup[pe.PRED_HOME].notna().sum()), 0)
        self.assertGreater(int(warmup[pe.PRED_HOME].isna().sum()), 0)
        covered = merged[merged["season"] >= 2021]
        self.assertFalse(bool(covered[pe.PRED_HOME].isna().any()))

    def test_refit_covers_all_predicted_rows(self):
        """Fit-only refill (sealed/slate path): 100% of feature-valid rows
        get per-side preds."""
        pre = self.games[self.games["season"] <= 2020].copy()
        sld = self.games[self.games["season"] == 2022].copy()
        refit = pe.refit_per_side(pre, sld, n_rounds=50, features=FEATS)
        self.assertEqual(len(refit), len(sld))
        self.assertEqual(int(refit[pe.PRED_HOME].isna().sum()), 0)
        self.assertEqual(int(refit[pe.PRED_AWAY].isna().sum()), 0)


class TestResidualArtifactGuard(unittest.TestCase):
    """persist_residuals FAILS LOUDLY — the artifact is raw material."""

    def setUp(self):
        self.games = _synth_games()
        self.folds = _val_folds(self.games)
        self.out, _r, _u = pe.oof_per_side(self.folds, FEATS, self.games)

    def test_artifact_written_with_all_required_columns(self):
        path = Path(self.id().replace(".", "_") + ".csv")
        try:
            written = pe.persist_residuals(self.out, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            back = pd.read_csv(written)
            for c in ["game_id", "fold_idx"] + pe.PRED_COLS + pe.RESID_COLS:
                self.assertIn(c, back.columns)
            self.assertEqual(len(back), len(self.out))
        finally:
            if path.exists():
                path.unlink()

    def test_missing_columns_refused(self):
        bad = self.out.drop(columns=[pe.RESID_AWAY])
        with self.assertRaises(RuntimeError):
            pe.persist_residuals(bad, "/tmp/never_written.csv")

    def test_empty_frame_refused(self):
        empty = self.out.iloc[0:0].copy()
        with self.assertRaises(RuntimeError):
            pe.persist_residuals(empty, "/tmp/never_written.csv")

    def test_unwritable_path_raises(self):
        with self.assertRaises(RuntimeError):
            pe.persist_residuals(self.out, "/proc/definitely/not/writable.csv")


class TestFamilyParameter(unittest.TestCase):
    """Family is a parameter, not hardcoded."""

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            pe._fit_side("nope", np.zeros((10, 2)), np.zeros(10),
                         np.zeros((4, 2)), np.zeros(4))

    def test_rf_walk_honors_family_and_shape(self):
        games = _synth_games(n_seasons=3, weeks=6)
        folds = _val_folds(games)
        out, rounds, _u = pe.oof_per_side(folds, FEATS, games, family="rf")
        self.assertEqual(list(out.columns),
                         list(pe.oof_per_side(folds, FEATS, games,
                                              family="lgb")[0].columns))
        self.assertIn("home", rounds)
        self.assertGreater(len(out), 0)
        self.assertEqual(int(out["game_id"].duplicated().sum()), 0)


class TestFeatureColumnsUntouched(unittest.TestCase):
    """FEATURE_COLUMNS / 12-pool pins — no wiring, no self-reference."""

    def test_pred_cols_not_in_served_pool(self):
        for c in pe.PRED_COLS + pe.RESID_COLS:
            self.assertNotIn(c, nf.FEATURE_COLUMNS)
            self.assertNotIn(c, ml.admitted_model_features())

    def test_side_view_is_exactly_the_12_pool(self):
        self.assertEqual(sorted(pe.SIDE_FEATURES),
                         sorted(f for f in nf.FEATURE_COLUMNS if f != "is_home"))
        self.assertNotIn("is_home", pe.SIDE_FEATURES)

    def test_engine_never_imports_moneyline(self):
        """READ-ONLY producer: no moneyline outputs consumed."""
        src = io.open(pe.__file__, encoding="utf-8").read()
        self.assertNotIn("import nfl_moneyline", src)
        self.assertNotIn("from nfl_moneyline", src)
        self.assertNotIn("home_win", src)


if __name__ == "__main__":
    unittest.main()