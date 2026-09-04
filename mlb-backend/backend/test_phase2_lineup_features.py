"""
Tests for Phase 2 lineup-delta moneyline features (actual starting-9 wOBA vs
team season-to-date, per side).

Covers:
- build_batter_woba._point_in_time: prefix sums over strictly-earlier dates,
  season-partitioned (the 2026 opener's row carries nothing from 2025) — the
  no-lookahead core shared by the batter and team tables.
- add_lineup_delta_features (with injected caches): delta identity
  (mean of the actual 9's sd-wOBA − team sd-wOBA == feature), top-3 delta,
  min-PA rule (a sub-floor batter uses the team season mean, never their own
  tiny-sample wOBA), rest count (team's top-5 regulars not in the 9),
  NaN when a game has no lineup row, idempotence (existing columns win).
- Metadata: the 6 columns get synthesized authored entries (no placeholders).
- Run-engine guardrail: derive_run_features drops the 6 lineup columns (the
  run engine's raw-only view stays unchanged).
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as features_mod
from build_batter_woba import _point_in_time
from feature_metadata import _rich_entry
from features import (
    LINEUP_DELTA_COLS,
    LINEUP_MIN_PA,
    add_lineup_delta_features,
)
from run_engine import derive_run_features


def _seed_caches():
    """Inject tiny hand-built lineup/batter/team tables into the module cache.

    Batter sd-wOBA is keyed by (season, game_date, batter) and is ALREADY the
    point-in-time value (through games strictly before game_date) — exactly
    what the builder's tables carry.
    """
    lineups = pd.DataFrame({
        "game_pk": [101, 102, 103],
        "home_order": [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9]],
        "away_order": [[21, 22, 23, 24, 25, 26, 27, 28, 29],
                       [21, 22, 23, 24, 25, 26, 27, 28, 29],
                       [21, 22, 23, 24, 25, 26, 27, 28, 29]],
    })
    # batters 1..9 (home): 1-4 hot (0.40), 5-9 league (0.32), all >= min-PA
    # batter 7 below min-PA with a wild 3-PA 0.90 (must be ignored)
    rows = []
    for b in range(1, 10):
        woba = 0.40 if b <= 4 else 0.32
        pa = 200
        if b == 7:
            woba, pa = 0.90, 3  # below LINEUP_MIN_PA → team-mean fallback
        rows.append({"season": 2026, "game_date": pd.Timestamp("2026-07-01"),
                     "batter": b, "sd_woba": woba, "prior_pa": pa,
                     "last_team": "HOM"})
    for b in range(21, 30):
        woba = 0.33 if b <= 25 else 0.31
        rows.append({"season": 2026, "game_date": pd.Timestamp("2026-07-01"),
                     "batter": b, "sd_woba": woba, "prior_pa": 200,
                     "last_team": "AWY"})
    batter = pd.DataFrame(rows)
    teams = pd.DataFrame({
        "season": [2026, 2026],
        "game_date": [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-01")],
        "team": ["HOM", "AWY"],
        "sd_woba": [0.315, 0.310],
        "prior_pa": [3000, 2900],
        "top3_woba": [0.385, 0.370],
        "top5_ids": ['[1, 2, 3, 4, 5]', '[21, 22, 23, 24, 25]'],
    })
    features_mod._lineup_cache = {"lineups": lineups, "batter": batter,
                                  "team": teams}


class TestPointInTime(unittest.TestCase):
    def test_prior_sums_exclude_current_date(self):
        agg = pd.DataFrame({
            "season": [2026, 2026, 2026, 2026],
            "key": ["A", "A", "A", "A"],
            "game_date": pd.to_datetime(["2026-04-01", "2026-04-02",
                                         "2026-04-02", "2026-04-05"]),
            "num": [10.0, 5.0, 5.0, 20.0],
            "den": [40, 20, 20, 80],
            "pa": [40, 20, 20, 80],
        })
        out = _point_in_time(agg, ["num", "den"], "pa")
        out = out.sort_values("game_date")
        # April 1 has no prior games (prior sums are 0; the caller's sd_woba
        # = prior_num/prior_den is then NaN via the den>0 guard)
        self.assertEqual(out.iloc[0]["prior_pa"], 0)
        self.assertEqual(out.iloc[0]["prior_num"], 0.0)
        # April 2 (both rows collapsed into one date) sees only April 1
        self.assertEqual(out.iloc[1]["prior_pa"], 40)
        self.assertEqual(out.iloc[1]["prior_num"], 10.0)
        # April 5 sees April 1 + both April 2 rows (same-date fully excluded)
        self.assertEqual(out.iloc[2]["prior_pa"], 80)
        self.assertEqual(out.iloc[2]["prior_num"], 20.0)

    def test_season_partition_no_cross_season_leak(self):
        agg = pd.DataFrame({
            "season": [2025, 2025, 2026],
            "key": ["A", "A", "A"],
            "game_date": pd.to_datetime(["2025-10-01", "2025-10-02",
                                         "2026-03-20"]),
            "num": [50.0, 60.0, 10.0],
            "den": [200, 240, 40],
            "pa": [200, 240, 40],
        })
        out = _point_in_time(agg, ["num", "den"], "pa")
        opener = out[(out["season"] == 2026)].iloc[0]
        self.assertEqual(opener["prior_pa"], 0)  # 2025 tail must not leak in
        self.assertEqual(opener["prior_num"], 0.0)
        last25 = out[(out["season"] == 2025)].sort_values("game_date").iloc[-1]
        self.assertEqual(last25["prior_pa"], 200)


class TestLineupDeltaFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _seed_caches()
        cls.games = pd.DataFrame({
            "game_pk": [101, 102, 103],
            "game_date": ["2026-07-01", "2026-07-01", "2026-07-02"],
            "home_team": ["HOM", "HOM", "HOM"],
            "away_team": ["AWY", "AWY", "AWY"],
        })
        cls.out = add_lineup_delta_features(cls.games)

    def test_columns_present(self):
        for c in LINEUP_DELTA_COLS:
            self.assertIn(c, self.out.columns)

    def test_delta_identity(self):
        # Home actual-9 effective wOBA: batters 1-4 @ 0.40, 5,6,8,9 @ 0.32,
        # batter 7 (3 PA) falls back to the team mean 0.315.
        eff = [0.40] * 4 + [0.32, 0.32, 0.315, 0.32, 0.32]
        mean9 = np.mean(eff)
        self.assertAlmostEqual(self.out.loc[0, "lineup_actual_woba_delta_home"],
                               mean9 - 0.315, places=6)
        # Away side
        eff_a = [0.33] * 5 + [0.31] * 4
        self.assertAlmostEqual(self.out.loc[0, "lineup_actual_woba_delta_away"],
                               np.mean(eff_a) - 0.310, places=6)

    def test_min_pa_rule_ignores_tiny_sample(self):
        # batter 7 has a 3-PA 0.90 wOBA; the feature must use the team mean.
        # Recompute without the rule (pure mean) would be higher than with it.
        pure = np.mean([0.40] * 4 + [0.32, 0.32, 0.90, 0.32, 0.32])
        with_rule = self.out.loc[0, "lineup_actual_woba_delta_home"] + 0.315
        self.assertLess(with_rule, pure)  # 0.315 < 0.90 tamed the mean
        self.assertAlmostEqual(with_rule, np.mean(
            [0.40] * 4 + [0.32, 0.32, 0.315, 0.32, 0.32]), places=6)

    def test_top3_delta(self):
        # home top-3 of the actual 9 = 0.40, 0.40, 0.40 → mean 0.40
        # team top-3 baseline = 0.385
        self.assertAlmostEqual(self.out.loc[0, "lineup_actual_top3_delta_home"],
                               0.40 - 0.385, places=6)
        self.assertAlmostEqual(self.out.loc[0, "lineup_actual_top3_delta_away"],
                               0.33 - 0.370, places=6)

    def test_rest_count(self):
        # home top-5 = [1,2,3,4,5], all in the lineup → 0
        self.assertEqual(self.out.loc[0, "lineup_rest_count_home"], 0)
        # away top-5 = [21..25], all in the lineup → 0
        self.assertEqual(self.out.loc[0, "lineup_rest_count_away"], 0)

    def test_rest_count_counts_missing_stars(self):
        # game 102: home lineup missing 1, 2, 9 → rest_count = 2
        _seed_caches()
        lineups = features_mod._lineup_cache["lineups"].copy()
        idx = lineups.index[lineups["game_pk"] == 102][0]
        lineups.at[idx, "home_order"] = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        features_mod._lineup_cache["lineups"] = lineups
        out = add_lineup_delta_features(self.games)
        self.assertEqual(out.loc[1, "lineup_rest_count_home"], 2)
        _seed_caches()

    def test_missing_lineup_row_is_nan(self):
        # game 103 has a date (2026-07-02) with no team/batter rows → the
        # team baseline join yields NaN; feature must be NaN, not fabricated.
        out = self.out
        self.assertTrue(np.isnan(out.loc[2, "lineup_actual_woba_delta_home"]))

    def test_idempotent_existing_columns_win(self):
        df = self.games.copy()
        df["lineup_actual_woba_delta_home"] = 123.0
        out = add_lineup_delta_features(df)
        self.assertEqual(out.loc[0, "lineup_actual_woba_delta_home"], 123.0)

    def test_missing_key_columns_degrade_gracefully(self):
        df = self.games.copy().drop(columns=["home_team"])
        out = add_lineup_delta_features(df)
        self.assertTrue(out["lineup_actual_woba_delta_home"].isna().all())


class TestMetadataAndRunEngine(unittest.TestCase):
    def test_metadata_synthesized(self):
        for c in LINEUP_DELTA_COLS:
            self.assertIsNotNone(_rich_entry(c), f"{c} missing metadata")

    def test_run_engine_excludes_lineup_columns(self):
        _, dropped = derive_run_features(LINEUP_DELTA_COLS)
        self.assertEqual(set(dropped), set(LINEUP_DELTA_COLS))


class TestMissingArtifactsFailLoud(unittest.TestCase):
    """The 42ef3f7 cleanup incident: artifacts absent → the TRAINING path must
    fail LOUD (FileNotFoundError naming the missing file), never silently
    project/train with dead columns."""

    def setUp(self):
        import tempfile
        self._saved_cache = features_mod._lineup_cache
        self._saved_base = features_mod._lineup_base_dir
        self._tmp = tempfile.TemporaryDirectory()
        features_mod._lineup_cache = {}
        features_mod._lineup_base_dir = lambda: Path(self._tmp.name)
        self.df = pd.DataFrame({
            "game_pk": [700001, 700002],
            "game_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "home_team": ["HOM", "HOM"],
            "away_team": ["AWY", "AWY"],
        })

    def tearDown(self):
        features_mod._lineup_cache = self._saved_cache
        features_mod._lineup_base_dir = self._saved_base
        self._tmp.cleanup()

    def test_require_caches_raises_naming_missing_files(self):
        with self.assertRaises(FileNotFoundError) as cm:
            features_mod.add_lineup_delta_features(self.df, require_caches=True)
        msg = str(cm.exception)
        for name in features_mod._LINEUP_REQUIRED_FILES:
            self.assertIn(name, msg, f"error must name {name}")
        self.assertIn("REQUIRED", msg)
        self.assertIn("DEAD", msg)

    def test_default_still_degrades_to_nan(self):
        """Non-training callers (slate, tests) keep the graceful NaN path."""
        out = features_mod.add_lineup_delta_features(self.df)
        self.assertTrue(out["lineup_actual_woba_delta_home"].isna().all())


class TestTrainingPathGuard(unittest.TestCase):
    """v26 "computed 0 column(s)" incident regression: the TRAINING path
    (require_caches=True) must never silently train with dead lineup columns.
    Covers (1) all-NaN placeholder columns shipped by a prior broken run are
    RECOMPUTED + overwritten, (2) real values stay authoritative, (3) a
    loaded-but-stale cache (empty team baseline → all-NaN) raises a
    descriptive RuntimeError naming the caches, (4) the slate path
    (require_caches=False) keeps the graceful NaN fallback."""

    @classmethod
    def setUpClass(cls):
        _seed_caches()
        cls.games = pd.DataFrame({
            "game_pk": [101, 102, 103],
            "game_date": ["2026-07-01", "2026-07-01", "2026-07-02"],
            "home_team": ["HOM", "HOM", "HOM"],
            "away_team": ["AWY", "AWY", "AWY"],
        })

    def tearDown(self):
        _seed_caches()   # restore the canonical caches after each test

    def test_all_nan_placeholder_recomputed_on_training_path(self):
        # The v26 game_level_features.csv ships the 6 columns as all-NaN
        # placeholders (the rebind bug re-saved them). require_caches=True
        # must RECOMPUTE + overwrite them — not early-return on "present".
        df = self.games.copy()
        for c in LINEUP_DELTA_COLS:
            df[c] = np.nan
        out = add_lineup_delta_features(df, require_caches=True)
        self.assertGreater(
            out["lineup_actual_woba_delta_home"].notna().mean(), 0.5)
        self.assertAlmostEqual(
            out.loc[0, "lineup_actual_woba_delta_home"],
            np.mean([0.40] * 4 + [0.32, 0.32, 0.315, 0.32, 0.32]) - 0.315,
            places=6)

    def test_real_values_untouched_on_training_path(self):
        # A genuinely computed value is authoritative even with
        # require_caches=True — only all-NaN placeholders are overwritten.
        df = self.games.copy()
        df["lineup_actual_woba_delta_home"] = 123.0
        out = add_lineup_delta_features(df, require_caches=True)
        self.assertEqual(out.loc[0, "lineup_actual_woba_delta_home"], 123.0)

    def test_stale_caches_raise_on_training_path(self):
        # Caches are PRESENT (file guard passes) but the team baseline is
        # empty → every feature stays NaN → the zero-column sentinel must
        # raise a descriptive RuntimeError naming the caches, never a quiet
        # "computed 0" log.
        stale = features_mod._lineup_cache.copy()
        stale["team"] = stale["team"].iloc[0:0]   # empty team baseline
        features_mod._lineup_cache = stale
        with self.assertRaises(RuntimeError) as cm:
            add_lineup_delta_features(self.games, require_caches=True)
        msg = str(cm.exception)
        self.assertIn("TRAINING-path enrichment produced NO live values", msg)
        self.assertIn("lineup_actual_woba_delta_home", msg)
        for name in ("lineups.parquet", "batter_woba.parquet",
                     "team_woba.parquet"):
            self.assertIn(name, msg)

    def test_slate_path_graceful_with_stale_caches(self):
        # Same stale caches WITHOUT require_caches (the slate path): the
        # projected-only fallback stays NaN and never raises.
        stale = features_mod._lineup_cache.copy()
        stale["team"] = stale["team"].iloc[0:0]
        features_mod._lineup_cache = stale
        out = add_lineup_delta_features(self.games)
        self.assertTrue(out["lineup_actual_woba_delta_home"].isna().all())


class TestFreshCloneCoverage(unittest.TestCase):
    """With the committed runtime inputs present, the enrichment computes the
    6 columns at full coverage on the committed game CSV (fresh-cache load)."""

    def test_six_columns_computed_at_high_coverage(self):
        csv = Path(__file__).resolve().parents[1] / "data_delivery" / "game_level_features.csv"
        missing = features_mod._missing_lineup_artifacts()
        if not csv.exists() or missing:
            self.skipTest(
                f"committed artifacts absent (csv={csv.exists()}, missing={missing})")
        features_mod._lineup_cache = {}  # force a fresh disk load
        df = pd.read_csv(csv)
        out = features_mod.add_lineup_delta_features(df, require_caches=True)
        yr = df["game_date"].astype(str).str[:4]
        recent = yr.isin(["2025", "2026"]).to_numpy()
        for c in LINEUP_DELTA_COLS:
            self.assertIn(c, out.columns)
            cov = out[c].notna().mean()
            # The lineup-actual feed has no 2024 coverage (data-era gap), so
            # overall coverage on the expanded frame is ~61-63%. The REAL
            # behavior contract is that the recent era stays trainable.
            # Floor by data requirement: the top3 deltas need the batting
            # ORDER, which strict-PIT NULLing (lineups not posted pre-game)
            # leaves absent ~6% of recent games by design (~93.7%); the
            # roster-level woba/rest columns only need the lineup itself
            # (~96-97%). A per-type floor keeps each tripwire real.
            recent_cov = out.loc[recent, c].notna().mean()
            needs_order = "top3" in c
            floor = 0.92 if needs_order else 0.95
            self.assertGreaterEqual(
                recent_cov, floor,
                f"{c} recent-era coverage {recent_cov:.1%} — feature would "
                "train dead on the 2025-26 artifact rows")
            # Overall (all-era) floor tracks the same per-type split: the
            # order-requiring top3 columns sit ~60.8% on the expanded frame
            # (2024 has no lineup feed AND strict-PIT NULLs compound), the
            # roster-level columns ~63%. Pins were 0.61 for both on the
            # 6,953-frame; re-based per type on the 7,048-frame.
            overall_floor = 0.60 if needs_order else 0.61
            self.assertGreaterEqual(
                cov, overall_floor,
                f"{c} overall coverage {cov:.1%} — below the expanded-frame "
                "era-weighted floor (2024 has no lineup feed)")
        # rest count is NaN for games without a lineup row; deltas must be real
        # (never the projected 0.0-only / all-NaN state)
        self.assertGreater(out["lineup_actual_woba_delta_home"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()
