"""Moneyline margin-k ablation tests (C0 raw vs C1 fold-local k-expanded).

Covers the 2026-09 margin-k task's crucial requirements WITHOUT the full
run-engine derivation (pure unit tests on synthetic fixtures, mirroring
test_build_oof_margin.py's conventions):

- k is fitted on the STRICTLY-PRIOR training portion only: corrupting later
  folds' rows leaves earlier folds' k byte-identical; corrupting the sealed
  (post-holdout) rows leaves EVERY fitted k unchanged (sealed never
  participates in any k fit).
- PIT: changing a future game's outcome leaves earlier rows' k-expanded
  margins byte-identical.
- The k-expanded margin is EXACTLY k × raw margin under the C2 two-sided
  expansion, which preserves the sum (λ'_H + λ'_A == λ_H + λ_A per game).
- C0 vs C1 differ ONLY in the margin column (all other columns
  byte-identical through build_c1).
- Warm-up guard: <100 strictly-prior rows or degenerate edge variance → k=1.
- Holdout (sealed) margin scales by k_last (the final executed fold's k).
- FEATURE_COLS / served pool untouched (the harness never assigns
  training.FEATURE_COLS — both arms share the identical column layout).

Run from mlb-backend/:
    python -m unittest backend.test_mlb_margin_k_ablation
or directly (adds the backend dir to sys.path):
    python backend/test_mlb_margin_k_ablation.py
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_mlb_margin_k_ablation as mk
from build_oof_margin import MARGIN_COL


def _synthetic_margins(n_folds: int = 6, rows_per_fold: int = 30,
                       seed: int = 42) -> pd.DataFrame:
    """margins-table fixture: per-fold OOF rows with λs, raw margin and
    actuals — the shape prepare_data writes to the margin_oof_cache."""
    rng = np.random.default_rng(seed)
    rows = []
    pk = 800_000
    base_h = rng.uniform(3.6, 4.4, rows_per_fold * n_folds)
    edges = rng.normal(0, 0.9, rows_per_fold * n_folds)
    for f in range(n_folds):
        start = f * rows_per_fold
        for i in range(rows_per_fold):
            lam_h = base_h[start + i]
            lam_a = lam_h - edges[start + i]
            actual = 0.05 + 1.6 * edges[start + i] + rng.normal(0, 1.0)
            rows.append({
                "game_pk": pk, "fold_idx": f,
                "lam_home": round(lam_h, 5), "lam_away": round(lam_a, 5),
                MARGIN_COL: round(lam_h - lam_a, 5),
                "home_score": max(1, int(lam_h + 1.4)),
                "away_score": max(1, int(lam_a + 1.4)),
                "home_win": float(rng.integers(0, 2)),
            })
            pk += 1
    m = pd.DataFrame(rows)
    m["game_date"] = pd.to_datetime(
        "2025-04-01") + pd.to_timedelta(m.index, unit="D")
    return m


def _sealed_rows() -> pd.DataFrame:
    """Post-holdout games that must NEVER enter any k fit (they have no
    fold_idx / no OOF λ by construction)."""
    return pd.DataFrame({
        "game_pk": [999_001, 999_002, 999_003],
        "game_date": pd.to_datetime(["2026-08-20", "2026-08-21",
                                     "2026-08-22"]),
        "home_score": [5, 2, 8], "away_score": [3, 1, 4],
    })


class TestKFitStrictlyPriorOnly(unittest.TestCase):
    def test_corrupting_later_folds_leaves_earlier_k_unchanged(self):
        m = _synthetic_margins(seed=1)
        base = mk.per_fold_k(m, m)
        # Flip every later fold's λs + actuals (as if a data fix changed
        # those games): previously-fitted k must be byte-identical.
        corrupted = m.copy()
        mask = corrupted["fold_idx"] >= 3
        corrupted.loc[mask, "lam_home"] *= 1.5
        corrupted.loc[mask, "lam_away"] *= 0.7
        corrupted.loc[mask, MARGIN_COL] = (
            corrupted.loc[mask, "lam_home"]
            - corrupted.loc[mask, "lam_away"]).round(5)
        corrupted.loc[mask, "home_score"] += 9
        again = mk.per_fold_k(corrupted, corrupted)
        for f in base:
            if f < 3:
                self.assertEqual(base[f], again[f],
                                 f"fold {f} k must ignore later folds")
        # k for fold >= 3 DOES reflect its own strictly-prior set changing
        # (fits on folds < 3 which corrupted) — assert it moved somewhere.
        self.assertTrue(any(base[f] != again[f] for f in base if f >= 3))

    def test_sealed_rows_never_participate_in_any_k_fit(self):
        m = _synthetic_margins(seed=7)
        sealed = _sealed_rows()
        base = mk.per_fold_k(m, m)
        # Seal corruption: flip the sealed games' outcomes/scores — every k
        # must be unchanged (margins carries no sealed rows, so the fit set
        # is literally the same object).
        sealed_bad = sealed.copy()
        sealed_bad["home_score"] = sealed_bad["home_score"] + 500
        frames = [m, pd.concat([m, sealed_bad], ignore_index=True)]
        ks = [mk.per_fold_k(fr, fr) for fr in frames]
        self.assertEqual(ks[0], ks[1],
                         "sealed rows must never enter any k fit")

    def test_warmup_guard_identity(self):
        # < K_MIN_PRIOR_ROWS prior rows → k=1.0 (identity, degrade to raw).
        self.assertEqual(mk.fit_fold_k(np.array([0.1, 0.2, 0.3]),
                                       np.array([1.0, 2.0, 3.0])), 1.0)
        # Degenerate edge variance → k=1.0.
        self.assertEqual(
            mk.fit_fold_k(np.zeros(150), np.arange(150, dtype=float)), 1.0)
        # A real slope survives both guards.
        rng = np.random.default_rng(0)
        d = rng.normal(0, 1, 200)
        k = mk.fit_fold_k(d, 0.05 + 1.6 * d)
        self.assertAlmostEqual(k, 1.6, delta=0.15)


class TestPIT(unittest.TestCase):
    def test_future_outcome_change_leaves_past_c1_margins_identical(self):
        m = _synthetic_margins(seed=3)
        k0 = mk.per_fold_k(m, m)
        c0 = mk.k_expanded_margins(m, k0)
        # PIT: a FUTURE game's outcome flips (its fold's λs/scores change).
        flipped = m.copy()
        flip = flipped["fold_idx"] == flipped["fold_idx"].max()
        flipped.loc[flip, "home_score"] += 7
        flipped.loc[flip, MARGIN_COL] *= 1.3
        flipped.loc[flip, "lam_home"] += 0.4
        k1 = mk.per_fold_k(flipped, flipped)
        c1 = mk.k_expanded_margins(flipped, k1)
        for f in sorted(m["fold_idx"].unique())[:-1]:
            a = c0[c0["fold_idx"] == f].sort_values("game_pk").reset_index(drop=True)
            b = c1[c1["fold_idx"] == f].sort_values("game_pk").reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(a, b, check_like=True)
            except AssertionError as exc:
                self.fail(f"fold {f} margins must be byte-identical under "
                          f"PIT of a future game: {exc}")


class TestKMath(unittest.TestCase):
    def test_two_sided_expansion_preserves_sum_and_diff_equals_k_times_raw(self):
        lam_h = np.array([3.2, 4.4, 5.1, 3.9, 4.7])
        lam_a = np.array([3.0, 4.1, 5.3, 3.7, 4.0])
        marg = lam_h - lam_a
        for k in (0.8, 1.0, 1.49, 1.9, 2.5):
            mu = (lam_h + lam_a) / 2.0
            lh2 = mu + k * (lam_h - mu)
            la2 = mu + k * (lam_a - mu)
            np.testing.assert_allclose(lh2 + la2, lam_h + lam_a,
                                       rtol=0, atol=1e-12,
                                       err_msg=f"k={k}: sum must be preserved")
            np.testing.assert_allclose(lh2 - la2, k * marg,
                                       rtol=0, atol=1e-12,
                                       err_msg=f"k={k}: diff == k*raw")
        # And the harness's k_expanded_margins is exactly k × raw.
        m = _synthetic_margins(n_folds=2, rows_per_fold=5, seed=5)
        k_by = {f: 1.3 + 0.1 * f for f in sorted(m["fold_idx"].unique())}
        out = mk.k_expanded_margins(m, k_by)
        np.testing.assert_allclose(
            out[MARGIN_COL].to_numpy(),
            m["fold_idx"].map(k_by).to_numpy()
            * m[MARGIN_COL].to_numpy(), atol=1e-5)


class TestC0C1Isolation(unittest.TestCase):
    def test_build_c1_differs_only_in_margin_column(self):
        m = _synthetic_margins(seed=11)
        # A minimal base frame with the non-margin moneyline columns.
        base_cols = ["game_pk", "game_date", "home_win", "home_team",
                     "away_team", "elo_diff", "win_pct_diff"]
        base = m[["game_pk"]].copy()
        base["game_date"] = m["game_date"].values
        rng = np.random.default_rng(11)
        base["home_win"] = rng.integers(0, 2, len(base)).astype(float)
        base["home_team"] = "T01"
        base["away_team"] = "T02"
        base["elo_diff"] = rng.normal(0, 120, len(base))
        base["win_pct_diff"] = rng.normal(0, 0.2, len(base))
        tune_df = base.drop(columns=["home_win"]).merge(
            base[["game_pk", "home_win"]], on="game_pk", how="left")
        tune_df = tune_df[["game_pk", "game_date", "home_win", "home_team",
                           "away_team", "elo_diff", "win_pct_diff"]]

        # C0 frame: raw margin attached.
        c0 = mk.attach(tune_df, m[["game_pk", MARGIN_COL]])
        k_by = {f: 1.2 + 0.1 * f for f in sorted(m["fold_idx"].unique())}
        m1 = mk.k_expanded_margins(m, k_by)
        c1 = mk.attach(tune_df, m1[["game_pk", MARGIN_COL]])

        self.assertNotIn(MARGIN_COL, tune_df.columns)
        non_margin = [c for c in c0.columns if c != MARGIN_COL]
        for c in non_margin:
            try:
                pd.testing.assert_series_equal(c0[c], c1[c],
                                               check_names=False)
            except AssertionError as exc:
                self.fail(f"column {c} must be identical between C0 and C1: "
                          f"{exc}")
        self.assertNotEqual(
            c0[MARGIN_COL].fillna(0).sum(), c1[MARGIN_COL].fillna(0).sum(),
            "the margin VALUES must actually differ")

    def test_holdout_margin_scaled_by_last_fold_k(self):
        m = _synthetic_margins(seed=13)
        k_by = mk.per_fold_k(m, m)
        k_last = k_by[max(k_by)]
        # A sealed-holdout frame with RAW refit margins (production
        # refit_run_margins convention) + extra columns that must survive.
        hold = pd.DataFrame({
            "game_pk": [999_001, 999_002],
            "game_date": pd.to_datetime(["2026-08-20", "2026-08-21"]),
            "home_win": [1.0, 0.0],
            "elo_diff": [12.0, -45.0],
            MARGIN_COL: [0.42, -0.11],
        })
        scaled = mk.scale_hold_margin(hold, k_last)
        np.testing.assert_allclose(scaled.to_numpy(),
                                   np.round(k_last * np.array([0.42, -0.11]), 5))
        # Non-margin columns untouched in the returned full frame via build_c1
        # (folds=[] keeps walk_forward geometry trivially consistent).
        hold2 = hold.copy()
        tune_c1, hold_c1, c1_splits = mk.build_c1(m.copy(), hold2, m, k_by, [])
        self.assertEqual(c1_splits, [])
        pd.testing.assert_series_equal(hold_c1["elo_diff"], hold2["elo_diff"])
        np.testing.assert_allclose(
            hold_c1[MARGIN_COL].to_numpy(),
            np.round(k_last * np.array([0.42, -0.11]), 5))

    def test_feature_cols_untouched(self):
        """The harness never assigns training.FEATURE_COLS (both arms share
        the identical 65-col layout; only the margin VALUES differ). Assert
        module state is untouched by the k machinery + a source-level guard
        against a future FEATURE_COLS assignment."""
        import training
        before = list(training.FEATURE_COLS)
        m = _synthetic_margins(seed=17)
        _ = mk.per_fold_k(m, m)
        _ = mk.k_expanded_margins(m, {f: 1.1 for f in m["fold_idx"].unique()})
        _ = mk.fit_fold_k(np.array([0.1, 0.2]), np.array([1.0, 2.0]))
        self.assertEqual(list(training.FEATURE_COLS), before,
                         "FEATURE_COLS must be untouched by harness helpers")
        src = Path(mk.__file__).read_text()
        self.assertNotIn("training.FEATURE_COLS =", src,
                         "harness must never assign FEATURE_COLS")


if __name__ == "__main__":
    unittest.main()