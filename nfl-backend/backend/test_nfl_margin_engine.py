"""NFL margin engine Phase-1 tests — pure, no-network.

Covers ``nfl_margin_engine`` (the fold-aligned OOF attach + fit-only refill)
and ``run_nfl_margin_ablation`` (the attach helper):
- Fold-aligned OOF attach (MLB parity): every margin prediction belongs to
  its fold's val rows only; the attach is a pure keyed left-merge on
  game_id; warmup games never appear in any fold's val window.
- Leakage-assertion trigger: a fold whose val overlaps train raises
  AssertionError (mirror of MLB's oof_run_margins fold-boundary guard).
- READ-ONLY producer: the caller's fold DataFrames are never mutated.
- Imputation discipline: uncovered rows are NaN (never zero/median-filled
  at attach time); the fit-only refill covers 100% of sealed rows.
- Determinism pin: identical folds + seed ⇒ byte-identical margins.
- FEATURE_COLUMNS untouched: pt_margin_diff is not in the served pool, and
  the margin engine never imports the moneyline (feature-producer only).

Run: python -m unittest test_nfl_margin_engine -v   (no network needed)
"""
from __future__ import annotations

import io
import unittest

import numpy as np
import pandas as pd

import nfl_features as nf
import nfl_margin_engine as me
import nfl_moneyline as ml
from run_nfl_margin_ablation import attach_margins

# Small-model test overrides (module globals read at call time).
me.MAX_ROUNDS = 100
me.EARLY_STOPPING_ROUNDS = 10

FEATS = ["f1", "f2", "elo_diff", "win_pct_diff"]


def _synth_games(n_seasons: int = 4, weeks: int = 8,
                 games_per_week: int = 6) -> pd.DataFrame:
    """Deterministic synthetic decided frame: seasons 2019.., weekly games,
    features correlated with the margin so the regressor can learn."""
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
    """The moneyline's own weekly folds (the folds the margin engine must use)."""
    val_seasons = val_seasons or [2021, 2022]
    return ml.generate_weekly_folds(games, val_seasons=val_seasons)


class TestMarginEngineAttach(unittest.TestCase):
    """Fold-aligned OOF attach — the four MLB-mirrored invariants."""

    def setUp(self):
        self.games = _synth_games()
        self.folds = _val_folds(self.games)
        self.margins, self.rounds, self.n_uncov = me.oof_margins(
            self.folds, FEATS, self.games)

    def test_every_margin_belongs_to_a_fold_val_row(self):
        """Parity: predictions exist ONLY for fold val rows (2021-22), one
        row per game — the same shape MLB's oof_run_margins produces."""
        val_ids = set()
        for f in self.folds:
            val_ids |= set(f["val"]["game_id"])
        self.assertGreater(len(self.margins), 0)
        self.assertEqual(
            int(self.margins["game_id"].duplicated().sum()), 0,
            "one margin row per game")
        self.assertTrue(
            set(self.margins["game_id"]) <= val_ids,
            "no margin outside a fold's val window")
        # Warmup seasons (2019-20) never validated → counted uncovered.
        warmup = self.games[self.games["season"] <= 2020]
        self.assertEqual(self.n_uncov, int(len(warmup)))

    def test_folds_are_strictly_prequential(self):
        """Every fold: max(train.gameday) < min(val.gameday) — the moneyline
        guard the margin engine re-asserts (MLB parity)."""
        for i, f in enumerate(self.folds):
            tr_max = pd.to_datetime(f["train"]["gameday"]).max()
            va_min = pd.to_datetime(f["val"]["gameday"]).min()
            self.assertLess(tr_max, va_min, f"fold {i} leaks")

    def test_leakage_assertion_triggers(self):
        """A fold whose val overlaps train raises AssertionError."""
        g = self.games.sort_values("gameday").reset_index(drop=True)
        mid = len(g) // 2
        bad_folds = [{"week_start": pd.Timestamp("2021-09-01"),
                      "train": g.iloc[: mid + 3].copy(),
                      "val": g.iloc[mid:].copy()}]
        with self.assertRaises(AssertionError):
            me.oof_margins(bad_folds, FEATS, g)

    def test_read_only_no_fold_mutation(self):
        """The margin engine never mutates the caller's folds (READ-ONLY
        producer) — no target column may be injected into them."""
        import copy
        folds = copy.deepcopy(self.folds)
        _ = me.oof_margins(folds, FEATS, self.games)
        for f in folds:
            self.assertNotIn(me.TARGET_MARGIN, f["train"].columns)
            self.assertNotIn(me.TARGET_MARGIN, f["val"].columns)
            self.assertNotIn(me.MARGIN_COL, f["train"].columns)
            self.assertNotIn(me.MARGIN_COL, f["val"].columns)

    def test_attach_is_keyed_left_join_mlb_parity(self):
        """MLB _attach_oof_run_margins parity: the attach is a pure keyed
        left-merge on game_id — shuffling the margins frame changes nothing,
        and every val-season game lands on exactly its own margin."""
        shuffled = self.margins.sample(frac=1, random_state=3).reset_index(drop=True)
        out = attach_margins(self.games, shuffled)
        m = self.margins.set_index("game_id")[me.MARGIN_COL]
        # Shuffled attach == unshuffled attach, keyed by game_id.
        expect = self.games.merge(
            self.margins[["game_id", me.MARGIN_COL]], on="game_id", how="left")
        pd.testing.assert_series_equal(
            out[me.MARGIN_COL].reset_index(drop=True),
            expect[me.MARGIN_COL].reset_index(drop=True))
        # Covered rows carry the fold's own prediction.
        spot = self.games["game_id"].iloc[0]
        if spot in m.index:
            got = out.loc[out["game_id"] == spot, me.MARGIN_COL].iloc[0]
            self.assertEqual(got, m.loc[spot])

    def test_uncovered_rows_are_nan_not_zero(self):
        """Imputation discipline: warmup rows with no OOF margin stay NaN at
        attach time (never 0 / never median) — the moneyline's imputation
        (trees route NaN natively; logistic/MLP get train medians) is the
        only fill, and it happens inside the model, not in the frame."""
        out = attach_margins(self.games, self.margins)
        warmup = out[out["season"] <= 2020]
        self.assertEqual(int(warmup[me.MARGIN_COL].notna().sum()), 0)
        self.assertGreater(int(warmup[me.MARGIN_COL].isna().sum()), 0)
        # Covered rows are finite.
        covered = out[out["season"] >= 2021][me.MARGIN_COL]
        self.assertFalse(bool(covered.isna().any()))

    def test_small_fold_val_left_uncovered(self):
        """A fold with <5 val rows is skipped (MLB MIN_VAL_FOLD_GAMES-style):
        its games get NO OOF margin → NaN, reported in the uncovered count."""
        g = self.games.sort_values("gameday").reset_index(drop=True)
        small = g[g["season"] == 2022].head(3).copy()
        train = g[g["gameday"] < small["gameday"].min()].copy()
        folds = [{"week_start": small["gameday"].min(),
                  "train": train, "val": small}]
        margins, _rounds, n_uncov = me.oof_margins(folds, FEATS, g)
        self.assertEqual(len(margins), 0)
        self.assertEqual(n_uncov, int(len(g)))

    def test_determinism_byte_identical(self):
        """Identical folds + seed ⇒ byte-identical margin table."""
        margins2, _r2, _u2 = me.oof_margins(self.folds, FEATS, self.games)
        a = self.margins.sort_values("game_id").reset_index(drop=True)
        b = margins2.sort_values("game_id").reset_index(drop=True)
        self.assertEqual(a.to_csv(index=False), b.to_csv(index=False))

    def test_refit_margins_covers_all_predicted_rows(self):
        """Fit-only refill (sealed/slate path): fit on pre-2021 decided,
        predict 2022 → 100% of feature-valid rows get a margin."""
        pre = self.games[self.games["season"] <= 2020].copy()
        sld = self.games[self.games["season"] == 2022].copy()
        refit = me.refit_margins(pre, sld, n_rounds=50, features=FEATS)
        self.assertEqual(int(refit[me.MARGIN_COL].isna().sum()), 0)
        self.assertEqual(len(refit), len(sld))


class TestFeatureColumnsUntouched(unittest.TestCase):
    """FEATURE_COLUMNS / 12-pool pins — no wiring, no self-reference."""

    def test_margin_col_not_in_served_pool(self):
        self.assertNotIn(me.MARGIN_COL, nf.FEATURE_COLUMNS)
        self.assertNotIn(me.MARGIN_COL, ml.admitted_model_features())

    def test_margin_view_is_exactly_the_12_pool(self):
        self.assertEqual(sorted(me.MARGIN_FEATURES),
                         sorted(f for f in nf.FEATURE_COLUMNS if f != "is_home"))
        self.assertNotIn("is_home", me.MARGIN_FEATURES)

    def test_margin_engine_never_imports_moneyline(self):
        """READ-ONLY producer: the margin engine must not consume moneyline
        outputs (no shared targets/weights/validation windows)."""
        src = io.open(me.__file__, encoding="utf-8").read()
        self.assertNotIn("import nfl_moneyline", src)
        self.assertNotIn("from nfl_moneyline", src)
        self.assertNotIn("home_win", src)


if __name__ == "__main__":
    unittest.main()