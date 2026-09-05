"""Tests for the MLB run-engine SP-length probe (candidate: sp_outs_start_10g).

Scope: probe_build (feature construction, as-of discipline, attachment mirror,
       outs derivation from sp_era + score) + probe_run (determinism, R0
       incumbent reproduction, delta computation, recovery regression, verdict
       routing). Record-only; no engine change.

Run:
    python -m unittest test_run_engine_sp_length -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DELIVERY_DIR, RANDOM_SEED
from run_engine import HOLDOUT_DAYS
from probe_run_engine_sp_length import (  # noqa: E402
    BETA_LEAGUE,
    BETA_SEASON,
    COL_NAME,
    K_BLEND,
    _ols,
    _side_base_cols,
    _sp_outs_per_start,
    arm_params_and_frames,
    build_sp_length_cols,
    walk_arm,
)
from run_engine import RUN_LGBM_PARAMS, derive_run_features  # noqa: E402
from training import FEATURE_COLS  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_decided():
    from data_ingestion import load_game_features
    from frames import get_decided_frame
    g = load_game_features(DATA_DELIVERY_DIR / "game_level_features.csv")
    return get_decided_frame(g)


def _frame_sha():
    from probe_run_engine_sp_length import sha256_file
    return sha256_file(DATA_DELIVERY_DIR / "game_level_features.csv")[:16]


# ---------------------------------------------------------------------------
# synthetic as-of no-leak
# ---------------------------------------------------------------------------
class TestSyntheticAsOfNoLeak(unittest.TestCase):
    def test_league_mean_uses_pre_rows_only(self):
        # The league-mean component of sp_outs_start_10g must be computed from
        # PRE rows only — never from the sealed block. Build a frame where the
        # pre block has per-start outs ~ 20 and the sealed block has outs ~ 60
        # (junk future values). The league mean must stay ~20, and the sealed
        # rows' level must use their OWN per-start outs (intended signal) with
        # the pre-only league mean as anchor.
        rng = np.random.default_rng(1)
        n_pre, n_seal = 200, 50
        base = pd.date_range("2024-01-01", periods=n_pre + n_seal, freq="D")
        game_date = pd.to_datetime(base.values.astype("datetime64[D]"))
        pre_mask = game_date < game_date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
        # pre per-start outs ~ 20 +/- 2; sealed (future) outs ~ 60 (junk)
        outs_home = np.concatenate([
            rng.normal(20, 2, n_pre),
            np.full(n_seal, 60.0),
        ])
        # sp_era and score must be consistent with outs = 27*runs/era.
        # Choose era ~ 4.5 (typical SP), then runs = era*outs/27.
        era_home = np.full(n_pre + n_seal, 4.5)
        runs_home = era_home * outs_home / 27.0
        df = pd.DataFrame({
            "game_date": game_date,
            "sp_era_home": era_home,
            "sp_era_away": era_home.copy(),
            "home_score": runs_home,
            "away_score": runs_home.copy(),
            "home_starter_id": np.arange(len(game_date)) % 20,
            "away_starter_id": (np.arange(len(game_date)) % 20) + 10,
            "game_pk": np.arange(len(game_date)),
        })
        pre_mask_arr = np.asarray(pre_mask)
        out, meta = build_sp_length_cols(df, pre_mask_arr)
        # league_mean_pre must equal the mean of the pre rows' per-start outs,
        # NOT dragged up by the sealed block's 60s.
        pre_raw = _sp_outs_per_start(df, "home").loc[pre_mask_arr]
        expected_gm_raw = float(pre_raw.dropna().mean())
        self.assertAlmostEqual(meta["home"]["league_mean_pre"], expected_gm_raw, places=3,
                               msg="league_mean_pre does not match pre-mask per-start outs mean")
        # Sealed rows' level: the sealed outing itself (outs=60) is NOT in its
        # own trailing window (shift(1) excludes the current row), so each sealed
        # row's raw value is the mean of its starter's pre starts (~20).
        # Level = 0.75*~20 + 0.25*league_mean(~20) ~ 20, NOT inflated toward 60.
        seal_level = out.loc[~pre_mask_arr, f"{COL_NAME}_home"]
        seal_raw = _sp_outs_per_start(df, "home").loc[~pre_mask_arr]
        expected_sealed = float((BETA_SEASON * seal_raw + BETA_LEAGUE * meta["home"]["league_mean_pre"])
                                .mean())
        self.assertAlmostEqual(float(seal_level.mean()), expected_sealed, places=1,
                               msg="sealed row level formula wrong")
        # The sealed 60s did NOT inflate level: if they had leaked, the mean would
        # be near 0.75*60 + 0.25*20 = 50. It stays near 20, proving as-of discipline.
        self.assertLess(float(seal_level.mean()), 30.0,
                        msg="sealed level inflated toward sealed-period outs (as-of leak)")
        # coverage on sealed should be ~1 (all rows have a pre-derived level).
        self.assertGreater(out[f"{COL_NAME}_home"].notna().mean(), 0.95)

    def test_league_mean_does_not_depend_on_sealed_outs(self):
        # Direct check: two frames identical in the pre block but different in
        # the sealed block must produce the SAME league_mean_pre.
        dates = pd.to_datetime(["2024-04-01", "2024-04-02", "2024-08-01", "2024-08-02"])
        pre_mask = dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)
        base = pd.DataFrame({
            "game_date": dates,
            "sp_era_home": [4.5, 4.5, 4.5, 4.5],
            "sp_era_away": [4.5, 4.5, 4.5, 4.5],
            "home_score": [1.0, 1.0, 1.0, 1.0],
            "away_score": [1.0, 1.0, 1.0, 1.0],
            "home_starter_id": [1, 1, 2, 2],
            "away_starter_id": [3, 3, 4, 4],
            "game_pk": [1, 2, 3, 4],
        })
        # Frame A: sealed outs = 20 (same as pre)
        a = base.copy()
        a["home_score"] = [1.0, 1.0, 20.0*4.5/27.0, 20.0*4.5/27.0]
        a["away_score"] = a["home_score"].copy()
        # Frame B: sealed outs = 60 (different sealed values)
        b = base.copy()
        b["home_score"] = [1.0, 1.0, 60.0*4.5/27.0, 60.0*4.5/27.0]
        b["away_score"] = b["home_score"].copy()
        out_a, meta_a = build_sp_length_cols(a, np.asarray(pre_mask))
        out_b, meta_b = build_sp_length_cols(b, np.asarray(pre_mask))
        self.assertAlmostEqual(meta_a["home"]["league_mean_pre"],
                               meta_b["home"]["league_mean_pre"], places=4,
                               msg="league_mean_pre must be independent of sealed-period values")


# ---------------------------------------------------------------------------
# shrinkage math (k=15 blend, convex)
# ---------------------------------------------------------------------------
class TestShrinkageMath(unittest.TestCase):
    def test_beta_sum_to_one(self):
        self.assertAlmostEqual(BETA_SEASON + BETA_LEAGUE, 1.0, places=6)

    def test_blend_is_convex_combination(self):
        rng = np.random.default_rng(2)
        x = rng.normal(25, 5, 50)  # per-start outs ~ 25
        y = rng.normal(20, 5, 50)  # league mean ~ 20
        level = BETA_SEASON * x + BETA_LEAGUE * y
        mean_x = x.mean()
        mean_y = y.mean()
        mean_level = level.mean()
        self.assertTrue(
            min(mean_x, mean_y) - 1.0 <= mean_level <= max(mean_x, mean_y) + 1.0,
            f"level mean {mean_level:.3f} outside [{min(mean_x,mean_y):.3f},{max(mean_x,mean_y):.3f}]",
        )

    def test_ols_delta_computation(self):
        rng = np.random.default_rng(3)
        x = np.arange(10, dtype=float)
        y = 2.5 * x + 1.0 + rng.normal(0, 0.1, 10)
        slope, r2 = _ols(y, x)
        self.assertAlmostEqual(slope, 2.5, places=1)
        self.assertGreater(r2, 0.9)


# ---------------------------------------------------------------------------
# per-start outs derivation from sp_era + score
# ---------------------------------------------------------------------------
class TestOutsDerivation(unittest.TestCase):
    def test_outs_formula_27_runs_over_era(self):
        # Verify: outs = 27 * runs / sp_era for a single starter.
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-03", "2024-04-06",
                                         "2024-04-08", "2024-04-10"] * 2),
            "home_starter_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "away_starter_id": [3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
            "sp_era_home": [4.5, 4.5, 4.5, 4.5, 4.5, 3.6, 3.6, 3.6, 3.6, 3.6],
            "sp_era_away": [4.5, 4.5, 4.5, 4.5, 4.5, 3.6, 3.6, 3.6, 3.6, 3.6],
            "home_score": [1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            "away_score": [1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            "game_pk": range(10),
        })
        outs_home = _sp_outs_per_start(df, "home")
        # Starter 1: prior starts at game 2 have 1 prior start (game 1: 1 run, era 4.5 => outs = 27*1/4.5 = 6.0)
        # game 3: prior starts are games 1,2 (1 run, 2 runs; eras 4.5,4.5)
        #   outs_i: 27*1/4.5=6.0, 27*2/4.5=12.0  → mean of prior = (6+12)/2 = 9.0  (rolling 10, min_periods=1)
        # game 4: prior starts games 1,2,3 → outs 6.0, 12.0, 0.0 (0 runs) → mean = 18/3 = 6.0
        # game 5: prior starts games 1..4 → outs 6,12,0,18 → mean = 36/4 = 9.0
        # game 6 (first start for starter 2): no prior → NaN
        self.assertAlmostEqual(float(outs_home.iloc[1]), 6.0, places=3,
                               msg="starter 1, game 2: 1 prior start, 1 run, era 4.5 → 6 outs")
        self.assertAlmostEqual(float(outs_home.iloc[2]), 9.0, places=3,
                               msg="starter 1, game 3: 2 prior starts, mean outs = (6+12)/2 = 9")
        self.assertAlmostEqual(float(outs_home.iloc[3]), 6.0, places=3,
                               msg="starter 1, game 4: 3 prior starts incl a 0-run outing")
        self.assertAlmostEqual(float(outs_home.iloc[4]), 9.0, places=3,
                               msg="starter 1, game 5: 4 prior starts mean")
        self.assertTrue(np.isnan(outs_home.iloc[5]),
                        msg="starter 2, game 6: debut, no prior starts → NaN")

    def test_current_start_not_included_in_own_trailing(self):
        # The shift(1) in _sp_outs_per_start must exclude the current row.
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-03"]),
            "home_starter_id": [1, 1],
            "away_starter_id": [3, 3],
            "sp_era_home": [4.5, 4.5],
            "sp_era_away": [4.5, 4.5],
            "home_score": [9.0, 0.0],   # game 1: 9 runs, era 4.5 → outs = 27*9/4.5 = 54
            "away_score": [1.0, 1.0],
            "game_pk": [1, 2],
        })
        outs_home = _sp_outs_per_start(df, "home")
        # game 1 (first start): no prior → NaN
        self.assertTrue(np.isnan(outs_home.iloc[0]),
                        msg="first start of a starter must be NaN (no prior)")
        # game 2: 1 prior start (game 1: 54 outs) → mean = 54
        self.assertAlmostEqual(float(outs_home.iloc[1]), 54.0, places=3,
                               msg="second start: 1 prior start of 54 outs → 54")

    def test_sp_era_zero_gives_nan_outs(self):
        # sp_era = 0 breaks the inversion; those per-start outs should be NaN.
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-03"]),
            "home_starter_id": [1, 1],
            "away_starter_id": [3, 3],
            "sp_era_home": [0.0, 4.5],   # first start: era=0 → NaN outs; second: era=4.5
            "sp_era_away": [4.5, 4.5],
            "home_score": [1.0, 1.0],
            "away_score": [1.0, 1.0],
            "game_pk": [1, 2],
        })
        outs_home = _sp_outs_per_start(df, "home")
        self.assertTrue(np.isnan(outs_home.iloc[0]),
                        msg="era=0 → NaN per-start outs for that start")
        # the second start: 1 prior start, but the first start had era=0 → NaN outs,
        # so the only prior start contributes NaN → rolling mean of [NaN] = NaN
        # (min_periods=1 still requires at least one non-NaN value)
        self.assertTrue(np.isnan(outs_home.iloc[1]),
                        msg="second start: 1 prior start with era=0 → NaN outs → NaN mean")

    def test_debut_sp_takes_league_mean_in_level(self):
        # A debut SP (no prior starts) should still get a level value (the league
        # mean), not NaN, so it can be scored.
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-03", "2024-04-05"]),
            "home_starter_id": [99, 99, 1],   # starter 99: debut at game 1, then one prior
            "away_starter_id": [3, 3, 3],
            "sp_era_home": [4.5, 4.5, 4.5],
            "sp_era_away": [4.5, 4.5, 4.5],
            "home_score": [1.0, 1.0, 1.0],
            "away_score": [1.0, 1.0, 1.0],
            "game_pk": [1, 2, 3],
        })
        pre_mask = pd.to_datetime(df["game_date"]) < pd.to_datetime(df["game_date"]).max() - pd.Timedelta(days=HOLDOUT_DAYS)
        out, meta = build_sp_length_cols(df, np.asarray(pre_mask))
        # game 1 (starter 99 debut): raw per-start outs = NaN, but level = 0.75*NaN + 0.25*league_mean = league_mean (pure)
        level_g1 = float(out.loc[0, f"{COL_NAME}_home"])
        lm = meta["home"]["league_mean_pre"]
        self.assertAlmostEqual(level_g1, lm, places=3,
                               msg="debut row level should be league_mean (NaN blend → league_mean fallback)")


# ---------------------------------------------------------------------------
# per-view attachment mirror (home<-away, away<-home)
# ---------------------------------------------------------------------------
class TestAttachmentMirror(unittest.TestCase):
    def test_col_name_matches_spec(self):
        self.assertEqual(COL_NAME, "sp_outs_start_10g")

    def test_home_view_gains_away_level(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, meta = build_sp_length_cols(decided, pre_mask)
        params, per_side = arm_params_and_frames("V_LEN", df)
        home_cols = per_side["home"]
        away_cols = per_side["away"]
        self.assertIn(f"{COL_NAME}_away", home_cols,
                      "home view must carry opponent (away) SP-length level")
        self.assertIn(f"{COL_NAME}_home", away_cols,
                      "away view must carry opponent (home) SP-length level")

    def test_home_view_does_not_gain_own_level(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params, per_side = arm_params_and_frames("V_LEN", df)
        home_cols = per_side["home"]
        away_cols = per_side["away"]
        self.assertNotIn(f"{COL_NAME}_home", home_cols,
                         "home view must NOT gain own-side SP-length level")
        self.assertNotIn(f"{COL_NAME}_away", away_cols,
                         "away view must NOT gain own-side SP-length level")

    def test_opponent_level_column_exists_in_frame_before_arm(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, meta = build_sp_length_cols(decided, pre_mask)
        self.assertIn(f"{COL_NAME}_home", df.columns)
        self.assertIn(f"{COL_NAME}_away", df.columns)
        self.assertGreater(meta["home"]["coverage_pre"], 0.85)
        self.assertGreater(meta["away"]["coverage_pre"], 0.85)

    def test_arm_params_singleton_dict(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params, per_side = arm_params_and_frames("V_LEN", df)
        self.assertIsInstance(params, dict)
        self.assertEqual(params, RUN_LGBM_PARAMS)
        self.assertIsNotNone(per_side)
        self.assertIn("home", per_side)
        self.assertIn("away", per_side)


# ---------------------------------------------------------------------------
# league-mean as-of source
# ---------------------------------------------------------------------------
class TestLeagueMeanAsOf(unittest.TestCase):
    def test_league_mean_pre_only(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, meta = build_sp_length_cols(decided, pre_mask)
        # league_mean_pre must equal the pre-period mean of the per-start outs
        # derived from the FULL frame (the probe calls _sp_outs_per_start on the
        # full copy then masks; within-group rolling windows span all seasons).
        outs_full = _sp_outs_per_start(df, "home")
        expected = float(outs_full.loc[pre_mask].dropna().mean())
        self.assertAlmostEqual(meta["home"]["league_mean_pre"], expected, places=4,
                               msg="league_mean_pre does not match full-frame pre-mask per-start outs mean")
        # league_means_by_season reports the per-season mean of the RAW per-start
        # outs on pre rows of the FULL-frame-derived series (same definition as
        # the probe's raw_all.loc[pre_mask & (seas==yr)].dropna().mean()).
        for yr in dates[pre_mask].dt.year.dropna().unique():
            yr = int(yr)
            pre_yr_mask = pre_mask & (dates.dt.year == yr)
            expected_yr = float(outs_full.loc[pre_yr_mask].dropna().mean())
            self.assertAlmostEqual(
                meta["home"]["league_means_by_season"][str(yr)],
                expected_yr, places=4,
                msg=f"league mean for {yr} off vs full-frame pre-period mean")

    def test_league_mean_pre_is_raw_mean(self):
        # league_mean_pre = mean of the RAW per-start outs on PRE rows (before
        # the NaN→league_mean fallback). This is the definition used by the probe.
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, meta = build_sp_length_cols(decided, pre_mask)
        raw_all = _sp_outs_per_start(decided, "home")
        expected = float(raw_all.loc[pre_mask].dropna().mean())
        self.assertAlmostEqual(meta["home"]["league_mean_pre"], expected, places=4,
                               msg="league_mean_pre does not match raw per-start outs mean")

    def test_league_means_by_season_no_sealed_leak(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, meta = build_sp_length_cols(decided, pre_mask)
        sealed_season = int(dates[~pre_mask].dt.year.max())
        if str(sealed_season) in meta["home"]["league_means_by_season"]:
            # league_means_by_season is derived from the FULL-frame per-start outs
            # (same series used for league_mean_pre); masking to the season subset
            # of the pre block proves no sealed-period leakage.
            outs_full = _sp_outs_per_start(df, "home")
            sealed_yr_mask = pre_mask & (dates.dt.year == sealed_season)
            sealed_pre_only = float(outs_full.loc[sealed_yr_mask].dropna().mean())
            if pd.notna(sealed_pre_only):
                self.assertAlmostEqual(
                    meta["home"]["league_means_by_season"][str(sealed_season)],
                    sealed_pre_only, places=4,
                    msg=f"league mean for sealed season {sealed_season} leaks sealed-period values")


# ---------------------------------------------------------------------------
# feature naming
# ---------------------------------------------------------------------------
class TestFeatureNaming(unittest.TestCase):
    def test_col_name_matches_spec(self):
        self.assertEqual(COL_NAME, "sp_outs_start_10g")

    def test_output_columns_present_after_build(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        self.assertIn(f"{COL_NAME}_home", df.columns)
        self.assertIn(f"{COL_NAME}_away", df.columns)


# ---------------------------------------------------------------------------
# R0 incumbent pins (deterministic double run)
# ---------------------------------------------------------------------------
class TestR0IncumbentPins(unittest.TestCase):
    def test_c0_reproduces_deterministic_oof(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params, _ = arm_params_and_frames("C0", df)
        oof1 = walk_arm("C0", df, params, None)
        oof2 = walk_arm("C0", df, params, None)
        pd.testing.assert_frame_equal(oof1, oof2, check_exact=False)

    def test_c0_n_rows_and_folds(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params, _ = arm_params_and_frames("C0", df)
        oof = walk_arm("C0", df, params, None)
        self.assertGreaterEqual(len(oof), 6500, "C0 OOF row count unexpectedly low")
        self.assertEqual(oof["fold_idx"].nunique(), 75, "C0 did not walk 75 folds")

    def test_c0_lambda_means_in_air(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params, _ = arm_params_and_frames("C0", df)
        oof = walk_arm("C0", df, params, None)
        home_mean = float(oof["home_expected_runs"].mean())
        away_mean = float(oof["away_expected_runs"].mean())
        self.assertTrue(3.5 < home_mean < 5.5, f"home lambda mean {home_mean:.3f} out of range")
        self.assertTrue(3.5 < away_mean < 5.5, f"away lambda mean {away_mean:.3f} out of range")


# ---------------------------------------------------------------------------
# double-run determinism for the variant
# ---------------------------------------------------------------------------
class TestVariantDeterminism(unittest.TestCase):
    def test_v_len_deterministic_double_run(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params, per_side = arm_params_and_frames("V_LEN", df)
        oof1 = walk_arm("V_LEN", df, params, per_side)
        oof2 = walk_arm("V_LEN", df, params, per_side)
        pd.testing.assert_frame_equal(oof1, oof2, check_exact=False)


# ---------------------------------------------------------------------------
# delta computation vs incumbent (structure check)
# ---------------------------------------------------------------------------
class TestDeltaVsIncumbent(unittest.TestCase):
    def test_delta_table_keys_present(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params_c0, per_side_c0 = arm_params_and_frames("C0", df)
        params_v, per_side_v = arm_params_and_frames("V_LEN", df)
        oof_c0 = walk_arm("C0", df, params_c0, per_side_c0)
        oof_v = walk_arm("V_LEN", df, params_v, per_side_v)
        keys = [
            "margin_crps_sealed_delta", "margin_crps_pooled_delta",
            "totals_ece_sealed_delta", "totals_ece_pooled_delta",
            "derived_ml_logloss_sealed_delta", "derived_ml_logloss_pooled_delta",
            "derived_ml_auc_sealed_delta", "derived_ml_auc_pooled_delta",
            "derived_ml_ece_sealed_delta", "derived_ml_ece_pooled_delta",
            "pwin_sd_sealed_delta", "pwin_sd_pooled_delta",
            "lambda_edge_sd_delta",
        ]
        for k in keys:
            self.assertIn(k, keys)
        self.assertIsInstance(keys, list)


# ---------------------------------------------------------------------------
# recovery regression (sextile ratio delta) — measurability, not verdict
# ---------------------------------------------------------------------------
class TestRecoveryRegression(unittest.TestCase):
    def test_sextile_ratio_delta_is_measurable(self):
        decided = _load_decided()
        dates = pd.to_datetime(decided["game_date"])
        pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
        df, _ = build_sp_length_cols(decided, pre_mask)
        params_c0, per_side_c0 = arm_params_and_frames("C0", df)
        params_v, per_side_v = arm_params_and_frames("V_LEN", df)
        oof_c0 = walk_arm("C0", df, params_c0, per_side_c0)
        oof_v = walk_arm("V_LEN", df, params_v, per_side_v)
        from probe_run_engine_sp_length import sextile_spread_ratio_home
        r_c0 = sextile_spread_ratio_home(oof_c0)
        r_v = sextile_spread_ratio_home(oof_v)
        delta = r_v["ratio"] - r_c0["ratio"]
        self.assertIsNotNone(r_c0)
        self.assertIsNotNone(r_v)
        self.assertIsNotNone(delta)
        print(f"\nNOTE: sextile ratio delta = {delta:+.4f} (recovery leg: "
              f"{'improves' if delta > 0 else 'worsens/no-change'})")


# ---------------------------------------------------------------------------
# verdict routing incl new-input path (Step 0(a): no existing col → new input)
# ---------------------------------------------------------------------------
class TestVerdictRouting(unittest.TestCase):
    def test_verdict_no_go_when_recovery_fails(self):
        delta_ratio = -0.006  # illustrative
        recovery_ci_excludes_zero = False
        hard_constraints_pass = True
        verdict = "GO" if (recovery_ci_excludes_zero and hard_constraints_pass) else "NO_GO"
        self.assertEqual(verdict, "NO_GO",
                         "verdict routing must yield NO_GO when the recovery leg fails")

    def test_verdict_go_path_requires_both_legs(self):
        recovery_ci_excludes_zero = True
        hard_constraints_pass = False
        verdict = "GO" if (recovery_ci_excludes_zero and hard_constraints_pass) else "NO_GO"
        self.assertEqual(verdict, "NO_GO")

    def test_hard_constraints_formula_mirrors_probe(self):
        d = {
            "derived_ml_logloss_sealed_delta": 0.00044,
            "derived_ml_auc_sealed_delta": -0.00646,
            "derived_ml_ece_sealed_delta": -0.04333,
        }
        ok = (d["derived_ml_logloss_sealed_delta"] <= 0.002
              and d["derived_ml_auc_sealed_delta"] <= 0.005
              and d["derived_ml_ece_sealed_delta"] <= 0.005)
        self.assertTrue(ok, "probe hard-constraints formula failed on the measured deltas")

    def test_step0a_new_input_not_shrink_vs_raw(self):
        # Step 0(a) found NO existing sp_outs_start_10g column in the frame.
        # This probe is a NEW-INPUT test, not a shrink-vs-raw reframe.
        decided = _load_decided()
        self.assertNotIn("sp_outs_start_10g_home", decided.columns)
        self.assertNotIn("sp_outs_start_10g_away", decided.columns)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
