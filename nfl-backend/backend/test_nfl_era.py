"""NFL era/conditional-mean layer — tests (pure, no-network).

Covers ``nfl_era_features`` (league-level era centers + the centered-target
E2 walk) and the runner conventions:
- Leakage guard: no center value references any same-date game (synthetic
  frames); centers for pre-sealed rows are invariant to later seasons being
  present (the property that makes spec selection sealed-safe — 2025 never
  touches selection).
- Exact values: EWM weights 0.5^(days_lag/halflife); ps = prior-season mean;
  first-history rows filled with the documented neutral constant.
- Synthetic era proof: a season-level mean shift on the away leg produces a
  systematic C0 bias that the E2 centered-target arm removes.
- E2 walk: leakage-assertion trigger, coverage geometry, pred+resid ==
  actual (1e-9), byte-identical determinism, missing-center-col guard.
- Pins: FEATURE_COLUMNS / SIDE_FEATURES untouched; nfl_era_features never
  imports the moneyline (feature-producer only).

Run: python -m unittest test_nfl_era -v   (no network needed)
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import nfl_features as nf
import nfl_run_engine_legacy_windows as ml
import nfl_per_side_engine as pe
from nfl_era_features import (
    CENTER_AWAY, CENTER_COLS, CENTER_HOME, NEUTRAL_CENTER, SPECS,
    attach_centers, compute_centers, oof_centered_per_side, refit_centered_per_side,
)
from test_nfl_per_side import FEATS, _synth_games

# Small-model test overrides (module globals read at call time).
pe.MAX_ROUNDS = 100
pe.EARLY_STOPPING_ROUNDS = 10

VAL_SEASONS = [2021, 2022]


def _synth_era_games(seed: int = 11, weeks: int = 8, games_per_week: int = 6,
                     away_base: dict[int, float] | None = None,
                     home_base: float = 21.0) -> pd.DataFrame:
    """Deterministic synthetic decided frame with per-season AWAY levels.

    home_base is constant across seasons; away_base maps season → level
    (default: 21 everywhere). Features carry no level information (pure
    noise), so any systematic per-side bias is the era effect.
    """
    rng = np.random.default_rng(seed)
    away_base = away_base or {}
    teams = [f"T{i}" for i in range(10)]
    rows = []
    gid = 2000
    for s in range(4):  # seasons 2019..2022
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
                f1 = float(rng.normal(0, 1))
                f2 = float(rng.normal(0, 1))
                hsc = int(round(home_base + f1 * 2.0 + rng.normal(0, 1.2)))
                asc = int(round(away_base.get(season, 21.0)
                                + f2 * 1.5 + rng.normal(0, 1.2)))
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


def _fold_games(df: pd.DataFrame) -> pd.DataFrame:
    """The view compute_centers + fold generation need (schedule facts)."""
    return df[["game_id", "season", "week", "gameday", "home_score",
               "away_score"]].copy()


def _val_folds(games: pd.DataFrame) -> list[dict]:
    return ml.generate_weekly_folds(games, val_seasons=VAL_SEASONS)


class TestCentersLeakageAndValues(unittest.TestCase):
    """Center construction: exact weights, strict prior-days, no NaN."""

    def test_unknown_spec_raises(self):
        df = _synth_games()
        with self.assertRaises(ValueError):
            compute_centers(_fold_games(df), "ewm_99w")

    def test_ewm_exact_weights_and_same_day_exclusion(self):
        """Hand-computed EWM: weight 0.5^(lag_days/14); same-day games never
        contribute to each other."""
        rows = [
            {"game_id": "A", "season": 2021, "gameday": pd.Timestamp("2021-09-12"),
             "home_score": 10, "away_score": 8},
            {"game_id": "C", "season": 2021, "gameday": pd.Timestamp("2021-09-12"),
             "home_score": 30, "away_score": 4},
            {"game_id": "B", "season": 2021, "gameday": pd.Timestamp("2021-09-26"),
             "home_score": 22, "away_score": 20},
            {"game_id": "E", "season": 2021, "gameday": pd.Timestamp("2021-09-26"),
             "home_score": 50, "away_score": 40},
            {"game_id": "D", "season": 2021, "gameday": pd.Timestamp("2021-10-10"),
             "home_score": 40, "away_score": 2},
        ]
        df = pd.DataFrame(rows)
        c = compute_centers(df, "ewm_2w").set_index("game_id")
        # B and E (same day): only A/C (lag 14d, w=0.5 each) → mean 20 / 6.
        self.assertAlmostEqual(float(c.loc["B", CENTER_HOME]), 20.0, places=4)
        self.assertAlmostEqual(float(c.loc["B", CENTER_AWAY]), 6.0, places=4)
        self.assertAlmostEqual(float(c.loc["E", CENTER_HOME]), 20.0, places=4)
        # D: A/C at lag 28d (w=0.25), B/E at lag 14d (w=0.5).
        home_d = (0.25 * 10 + 0.25 * 30 + 0.5 * 22 + 0.5 * 50) / 1.5
        away_d = (0.25 * 8 + 0.25 * 4 + 0.5 * 20 + 0.5 * 40) / 1.5
        self.assertAlmostEqual(float(c.loc["D", CENTER_HOME]), home_d, places=4)
        self.assertAlmostEqual(float(c.loc["D", CENTER_AWAY]), away_d, places=4)

    def test_same_day_game_never_changes_other_centers(self):
        """Same-day games never contribute to EACH OTHER's centers (strictly
        prior-day exclusion), but they DO legitimately feed later days."""
        base_rows = [
            {"game_id": "A", "season": 2021,
             "gameday": pd.Timestamp("2021-09-12"), "home_score": 10,
             "away_score": 8},
            {"game_id": "B", "season": 2021,
             "gameday": pd.Timestamp("2021-09-26"), "home_score": 22,
             "away_score": 20},
            {"game_id": "D", "season": 2021,
             "gameday": pd.Timestamp("2021-10-10"), "home_score": 40,
             "away_score": 2},
        ]
        extra_same_day = {"game_id": "E", "season": 2021,
                          "gameday": pd.Timestamp("2021-09-26"),
                          "home_score": 99, "away_score": 0}
        c1 = compute_centers(pd.DataFrame(base_rows), "ewm_2w")
        c2 = compute_centers(pd.DataFrame(base_rows + [extra_same_day]),
                             "ewm_2w")
        get = lambda c, gid, col: float(c.loc[c["game_id"] == gid, col]  # noqa: E731
                                         .iloc[0])
        # B and E share 09-26: E must NOT change B's center (nor A's), but
        # D (10-10) legitimately sees E as prior history.
        for gid in ("A", "B"):
            for col in CENTER_COLS:
                self.assertEqual(get(c1, gid, col), get(c2, gid, col),
                                 f"{gid} {col} changed by same-day E")
        self.assertEqual(get(c2, "E", CENTER_HOME), get(c2, "B", CENTER_HOME),
                         "E and B (same day) must share the same center")
        self.assertNotEqual(get(c1, "D", CENTER_HOME),
                            get(c2, "D", CENTER_HOME),
                            "D's center should legitimately include E")

    def test_ps_prior_season_mean_and_first_season_fallback(self):
        """ps center = full prior-season mean; 2019 (no prior season) gets
        the documented neutral constant."""
        rng = np.random.default_rng(3)
        rows = []
        for s, off in ((2019, 0.0), (2020, 5.0), (2021, 2.0)):
            for _ in range(8):
                rows.append({
                    "game_id": f"G{s}_{_}", "season": s,
                    "gameday": pd.Timestamp(f"{s}-09-1{_}"),
                    "home_score": int(20 + off), "away_score": int(18 + off),
                })
        df = pd.DataFrame(rows)
        c = compute_centers(df, "ps").set_index("game_id")
        for gid in c.index:
            s = int(gid[1:5])
            if s == 2019:
                self.assertEqual(float(c.loc[gid, CENTER_HOME]),
                                 NEUTRAL_CENTER)
                self.assertEqual(float(c.loc[gid, CENTER_AWAY]),
                                 NEUTRAL_CENTER)
            elif s == 2020:
                self.assertAlmostEqual(float(c.loc[gid, CENTER_HOME]), 20.0,
                                       places=4)
                self.assertAlmostEqual(float(c.loc[gid, CENTER_AWAY]), 18.0,
                                       places=4)
            else:  # 2021 → 2020 mean (20+5)
                self.assertAlmostEqual(float(c.loc[gid, CENTER_HOME]), 25.0,
                                       places=4)
                self.assertAlmostEqual(float(c.loc[gid, CENTER_AWAY]), 23.0,
                                       places=4)

    def test_no_nan_centers_all_specs(self):
        df = _synth_games()
        for spec in SPECS:
            c = compute_centers(_fold_games(df), spec)
            self.assertEqual(int(c[CENTER_COLS].isna().sum().sum()), 0,
                             f"{spec}: NaN centers survived the fill")

    def test_centers_of_pre_sealed_rows_invariant_to_later_seasons(self):
        """The structural sealed-safety property: centers for seasons < 2025
        are identical whether 2025 rows exist in the frame or not (centers
        look strictly backward). This is why spec selection on the pooled
        2021-24 rows can never be contaminated by sealed data."""
        df = _synth_games(n_seasons=5)  # 2019..2023
        later = pd.DataFrame({
            "game_id": [f"X{i}" for i in range(20)], "season": 2025,
            "week": 1, "gameday": pd.date_range("2025-09-07", periods=20,
                                                freq="3h"),
            "home_score": np.full(20, 50), "away_score": np.full(20, 3),
        })
        full = pd.concat([df, later], ignore_index=True)
        for spec in SPECS:
            c_full = compute_centers(_fold_games(full), spec).set_index(
                "game_id")
            c_sub = compute_centers(_fold_games(df), spec).set_index("game_id")
            for gid in df["game_id"]:
                for col in CENTER_COLS:
                    self.assertEqual(float(c_full.loc[gid, col]),
                                     float(c_sub.loc[gid, col]),
                                     f"{spec} {gid} {col}: later seasons "
                                     "changed the center")


class TestE2Walk(unittest.TestCase):
    """Centered-target fold walk mirrors the per-side engine's discipline."""

    def setUp(self):
        self.games = _synth_games()
        decided = _fold_games(self.games)
        self.centers = compute_centers(decided, "ewm_2w")
        self.frame = attach_centers(self.games, self.centers)
        # Folds are generated from the CENTER-ATTACHED frame so fold rows
        # carry the center columns (mirror of the runner's discipline).
        self.folds = _val_folds(self.frame)
        self.out, self.rounds, self.n_uncov = oof_centered_per_side(
            self.folds, FEATS, self.frame)

    def test_leakage_assertion_triggers(self):
        g = self.games.sort_values("gameday").reset_index(drop=True)
        mid = len(g) // 2
        bad_folds = [{"week_start": pd.Timestamp("2021-09-01"),
                      "train": g.iloc[: mid + 3].copy(),
                      "val": g.iloc[mid - 3:].copy()}]
        with self.assertRaises(AssertionError):
            oof_centered_per_side(bad_folds, FEATS, self.frame)

    def test_coverage_geometry(self):
        """One row per covered game, only fold val rows; preds on both
        sides; residual round-trip to 1e-9 (the artifact contract)."""
        val_ids = set()
        for f in self.folds:
            val_ids |= set(f["val"]["game_id"])
        self.assertGreater(len(self.out), 0)
        self.assertEqual(int(self.out["game_id"].duplicated().sum()), 0)
        self.assertTrue(set(self.out["game_id"]) <= val_ids)
        for side, pcol, rcol in (("home", "pred_home", "resid_home"),
                                 ("away", "pred_away", "resid_away")):
            merged = self.games.merge(self.out, on="game_id", how="inner")
            rebuilt = merged[pcol] + merged[rcol]
            np.testing.assert_allclose(
                rebuilt, merged[f"{side}_score"].astype(float),
                rtol=0, atol=1e-9)
        self.assertEqual(self.n_uncov, int(self.games["season"].isin(
            [2019, 2020]).sum()))

    def test_determinism_byte_identical(self):
        out2, _r, _u = oof_centered_per_side(self.folds, FEATS, self.frame)
        a = self.out.sort_values("game_id").reset_index(drop=True)
        b = out2.sort_values("game_id").reset_index(drop=True)
        self.assertEqual(a.to_csv(index=False), b.to_csv(index=False))

    def test_missing_center_columns_raise(self):
        with self.assertRaises(ValueError):
            oof_centered_per_side(self.folds, FEATS, self.games.copy())

    def test_attach_centers_mismatch_raises(self):
        bad = self.centers.copy()
        bad["game_id"] = "ZZ_" + bad["game_id"]
        with self.assertRaises(RuntimeError):
            attach_centers(self.games, bad)


class TestSyntheticEraProof(unittest.TestCase):
    """A season-level away scoring shift must produce systematic C0 bias that
    the centered-target arm removes (the spec's core falsifiable claim)."""

    def _walk(self, games, spec):
        decided = _fold_games(games)
        centers = compute_centers(decided, spec)
        frame = attach_centers(games, centers)
        folds = _val_folds(frame)
        c0, _r, _u = pe.oof_per_side(folds, FEATS, games)
        c0 = c0.merge(games[["game_id", "season", "home_score",
                             "away_score"]], on="game_id", how="left")
        e2, rounds, _u2 = oof_centered_per_side(folds, FEATS, frame)
        e2 = e2.merge(games[["game_id", "season", "home_score",
                             "away_score"]], on="game_id", how="left")
        return c0, e2

    def _assert_sides_finite(self, *outs):
        for label, out in outs:
            for side in ("home", "away"):
                resid = (out[f"{side}_score"].to_numpy(float)
                         - out[f"pred_{side}"].to_numpy(float))
                self.assertTrue(np.all(np.isfinite(resid)), label)

    def _away_bias(self, out, season: int | None = None) -> float:
        sub = out if season is None else out[out["season"] == season]
        return float((sub["away_score"] - sub["pred_away"]).mean())

    def test_ewm_arm_removes_established_level_shift(self):
        """A season-level away shift established BEFORE the val window
        (warm 2019 = 21, 2020-22 = 27): the EWM center tracks the level, so
        C0's raw-target regressor shows systematic away bias (anchored to the
        21/27 training mix) while the E2 centered-target arm removes it."""
        away_base = {2019: 21.0, 2020: 27.0, 2021: 27.0, 2022: 27.0}
        games = _synth_era_games(away_base=away_base)
        c0, e2 = self._walk(games, "ewm_2w")
        self._assert_sides_finite(("C0", c0), ("E2", e2))
        c0_away = self._away_bias(c0)
        e2_away = self._away_bias(e2)
        c0_home = float((c0["home_score"] - c0["pred_home"]).mean())
        e2_home = float((e2["home_score"] - e2["pred_home"]).mean())
        # C0: away underpredicted systematically (level shift unseen); the
        # home leg (no shift) stays near zero.
        self.assertGreater(abs(c0_away), 1.0,
                           f"C0 away bias {c0_away:.2f} not systematic")
        self.assertLess(abs(c0_home), 0.75,
                        f"C0 home bias {c0_home:.2f} should be ~0")
        # E2 (centered targets): away level bias removed (measured ~0).
        self.assertLess(abs(e2_away), 0.6,
                        f"E2 away bias {e2_away:.2f} not removed")
        self.assertLess(abs(e2_away), 0.35 * abs(c0_away),
                        f"E2 ({e2_away:.2f}) not a large cut of C0 "
                        f"({c0_away:.2f})")
        # E2 must not worsen the (already clean) home leg materially.
        self.assertLess(abs(e2_home - c0_home), 0.75)

    def test_ewm_arm_cuts_shift_into_val_season(self):
        """A shift landing at the START of a val season (2022: 21 → 27) is
        unknowable before the season, but the trailing EWM center tracks it
        within a few weeks → E2 cuts the shifted leg's bias vs C0 by most of
        it (measured ~86%). NOTE: the ps (prior-season anchor) spec cannot
        pass this test — a lagged anchor also contaminates the centered
        targets of in-training-window rows at the new level; the trailing
        EWM is the mechanism the CV selection exists to find."""
        away_base = {2019: 21.0, 2020: 21.0, 2021: 21.0, 2022: 27.0}
        games = _synth_era_games(away_base=away_base)
        c0, e2 = self._walk(games, "ewm_2w")
        self._assert_sides_finite(("C0", c0), ("E2", e2))
        c0_22 = self._away_bias(c0, 2022)
        e2_22 = self._away_bias(e2, 2022)
        self.assertGreater(abs(c0_22), 1.5,
                           f"C0 2022 away bias {c0_22:.2f} not systematic")
        # Measured 0.71 vs 4.95 — assert a >= 60% cut (2.5x margin).
        self.assertLessEqual(abs(e2_22), 0.4 * abs(c0_22),
                             f"E2 ewm cut insufficient: C0 {c0_22:.2f} → "
                             f"E2 {e2_22:.2f}")

    def test_refit_centered_covers_all_valid_predicted_rows(self):
        """Fit-only refill: every feature-valid predicted row gets a pred
        (never NaN) — the sealed/slate 100%-coverage discipline."""
        games = _synth_era_games(away_base={2020: 27.0, 2021: 27.0,
                                            2022: 27.0})
        decided = _fold_games(games)
        centers = compute_centers(decided, "ps")
        frame = attach_centers(games, centers)
        folds = _val_folds(frame)
        _, rounds, _u = oof_centered_per_side(folds, FEATS, frame)
        tr = frame[frame["season"].isin([2019, 2020, 2021])].copy()
        pr = frame[frame["season"] == 2022].copy()
        refit = refit_centered_per_side(tr, pr, rounds, FEATS)
        merged = pr.merge(refit, on="game_id", how="left")
        self.assertEqual(len(merged), len(pr))
        self.assertEqual(int(merged["pred_home"].isna().sum()), 0)
        self.assertEqual(int(merged["pred_away"].isna().sum()), 0)


class TestPins(unittest.TestCase):
    """FEATURE_COLUMNS untouched; feature-producer module hygiene."""

    def test_feature_columns_untouched(self):
        pinned = ["elo_diff", "win_pct_diff", "rest_days_diff",
                  "is_dome_home", "ewm_net_pts_diff", "ewm_ypp_diff",
                  "pace_plays_min_diff", "rest_short_diff", "div_game",
                  "travel_miles_diff", "altitude_home", "prime_time",
                  "is_home"]
        self.assertEqual(nf.FEATURE_COLUMNS, pinned)
        self.assertEqual(pe.SIDE_FEATURES,
                         [f for f in pinned if f != "is_home"])
        for col in CENTER_COLS + ["pred_home", "pred_away", "resid_home",
                                  "resid_away"]:
            self.assertNotIn(col, nf.FEATURE_COLUMNS)

    def test_no_moneyline_import(self):
        src = io.open("nfl_era_features.py", encoding="utf-8").read()
        self.assertNotIn("import nfl_moneyline", src)
        self.assertNotIn("from nfl_moneyline", src)
        # no silent feature mutation: the module only READS decided scores.
        self.assertNotIn("FEATURE_COLUMNS =", src)

    def test_runner_never_selects_on_sealed(self):
        """The runner's selection metric is pooled-only by construction:
        simulate selection over frames that differ ONLY in sealed rows and
        assert the pooled tables are identical (leak-free selection)."""
        games = _synth_games(n_seasons=5)  # 2019..2023
        later = pd.DataFrame({
            "game_id": [f"Y{i}" for i in range(24)], "season": 2025,
            "week": 1, "gameday": pd.date_range("2025-09-07", periods=24,
                                                freq="2h"),
            "home_score": np.full(24, 60), "away_score": np.full(24, 0),
            "f1": 0.0, "f2": 0.0, "home_win": 1,
        })
        pooled = games[games["season"] < 2025].copy()
        for spec in ("ps", "ewm_2w"):
            c_full = compute_centers(_fold_games(pd.concat([pooled, later],
                                                           ignore_index=True)),
                                     spec)
            c_pool = compute_centers(_fold_games(pooled), spec)
            f_full = attach_centers(pooled, c_full)
            f_pool = attach_centers(pooled, c_pool)
            folds_full = ml.generate_weekly_folds(f_full,
                                                  val_seasons=[2021, 2022])
            folds_pool = ml.generate_weekly_folds(f_pool,
                                                  val_seasons=[2021, 2022])
            o_full, _r, _u = oof_centered_per_side(folds_full, FEATS, f_full)
            o_pool, _r, _u = oof_centered_per_side(folds_pool, FEATS, f_pool)
            a = o_full.sort_values("game_id").reset_index(drop=True)
            b = o_pool.sort_values("game_id").reset_index(drop=True)
            self.assertEqual(a.to_csv(index=False), b.to_csv(index=False),
                             f"{spec}: pooled walk changed by sealed rows")


if __name__ == "__main__":
    with redirect_stdout(io.StringIO()):
        unittest.main(verbosity=0)
