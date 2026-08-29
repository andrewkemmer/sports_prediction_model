"""
Tests for momentum form-delta features (recent window − season-to-date
baseline, per side).

Covers:
- DuckDB layer: the 38 *_delta_* columns ship in game_level; the SP era/K9
  deltas satisfy delta == recent − season (identity against shipped cols);
  season baselines are season-partitioned (2026 opener excludes 2025) and
  point-in-time (current game excluded).
- pandas helper add_form_delta_features: recent − season math, NaN-safety
  (one missing term → NaN, never fabricated), idempotence (SQL-shipped
  columns win, never overwritten).
- Coverage on the committed CSV: only the SP era/K9 deltas are computable
  pre-refresh; the other 34 are all-NaN by design.
- Plumbing: the 38 columns are computed/shipped in the artifact and have
  authored metadata, but the 2026-08 ablation measured them negative (WITH
  loses both pooled OOF and the sealed holdout) so they are NOT in the
  active moneyline FEATURE_COLS — re-enabling is a one-line append.
- Run-engine guardrail: derive_run_features excludes every *_delta_* column
  (the run engine's 29-feature raw-only view is unchanged).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (
    FORM_DELTA_COLS,
    FORM_DELTA_SPECS,
    add_form_delta_features,
    build_features,
)
from training import FEATURE_COLS, _feature_matrix
from run_engine import derive_run_features

# Per-game team wOBA levels of the synthetic fixture by season. Each team
# bats 6 PAs per game: the 4-event _PATTERNS pattern in inning 1 plus 2
# reliever-era PAs in inning 2 (single, field_out / strikeout, single):
#   2025 -> (0.878 + 0.000 + 2.007 + 0.000 + 0.000 + 0.878) / 6 = 0.627167
#   2026 -> (0.878 + 0.000 + 0.000 + 0.000 + 0.000 + 0.878) / 6 = 0.292667
WOBA_2025 = (0.878 + 2.007 + 0.878) / 6
WOBA_2026 = (0.878 + 0.878) / 6

_PATTERNS = {
    2025: ["single", "field_out", "home_run", "strikeout"],
    2026: ["single", "field_out", "field_out", "strikeout"],
}

TEAMS = ["NYY", "BOS", "SEA", "LAD"]


def _pitch_row(g: int, pk: int, day: pd.Timestamp, home: str, away: str,
               half: str, batter_team: str, event: str, hs: int, as_: int,
               pitcher: float, batter: float, at_bat_number: int,
               inning: int = 1) -> dict:
    return dict(
        game_pk=pk, game_date=day.strftime("%Y-%m-%d"), game_type="R",
        home_team=home, away_team=away,
        inning=inning, inning_topbot=half, outs_when_up=0, balls=0, strikes=0,
        on_1b=np.nan, on_2b=np.nan, on_3b=np.nan,
        at_bat_number=at_bat_number, pitch_number=1,
        pitcher=pitcher, batter=batter,
        p_throws="R", stand="L",
        pitch_type="FF", release_speed=94.0,
        description="hit_into_play" if event != "strikeout" else "swinging_strike",
        events=event,
        barrel=np.nan, hard_contact=np.nan,
        launch_speed=99.0, launch_angle=28.0,
        estimated_woba_using_speedangle=0.350,
        estimated_ba_using_speedangle=0.250,
        zone=5, home_score=hs, away_score=as_, spin_rate=2300.0,
        woba_value=0.0, babip_value=0.0, iso_value=0.0,
        delta_home_win_exp=0.0, delta_run_exp=-0.1,
        player_name=f"P{g}", hit_distance_sc=300.0,
        release_pos_x=0.0, release_pos_z=5.0,
        release_spin_rate=2300.0, release_extension=6.0,
        pfx_x=0.0, pfx_z=0.0,
    )


def build_fixture(n_2025: int = 40, n_2026: int = 32) -> pd.DataFrame:
    """Deterministic 2-season fixture: ~17 games/team/season, team wOBA level
    per season from _PATTERNS. Returns the game-level feature frame."""
    rng = np.random.RandomState(7)
    rows = []
    g = 0
    for year, n_games in ((2025, n_2025), (2026, n_2026)):
        for k in range(n_games):
            home = TEAMS[g % 4]
            away = TEAMS[(g + 1) % 4]
            if home == away:
                continue
            base = "2025-09-01" if year == 2025 else "2026-04-01"
            # 2026 games restart at the opener (day offset within the season)
            # so the season-partition tests have an actual 2026-04-01 opener.
            day = pd.Timestamp(base) + pd.Timedelta(
                days=g if year == 2025 else g - n_2025)
            pk = 700000 + g
            hs, as_ = int(rng.randint(0, 8)), int(rng.randint(0, 8))
            # Home starter pitches the Top 1st (away bats); away starter the
            # Bottom 1st. Each PA gets a unique globally-monotone at_bat_number
            # (pa_boundary's score-progression LAG depends on it). Two reliever
            # PAs per half in inning 2 (non-starter IDs, 600+) give the bullpen
            # tables data: Top half → home bullpen, Bottom half → away bullpen.
            pitcher_h = float(500 + (g % 4) * 2)
            pitcher_a = float(500 + (g % 4) * 2 + 1)
            # Small rotating rosters (4 batters/team) so batters repeat across
            # games and batter_rolling has prior PAs to build shrunk_woba from.
            home_bat = float(100 + (g % 4) * 10 + (g // 4) % 4)
            away_bat = float(100 + ((g + 1) % 4) * 10 + (g // 4) % 4)
            ab = 0
            for half, batter_team, starter, batter in (
                    ("Top", away, pitcher_h, away_bat),
                    ("Bot", home, pitcher_a, home_bat)):
                for e in _PATTERNS[year]:
                    ab += 1
                    rows.append(_pitch_row(
                        g, pk, day, home, away, half, batter_team, e,
                        hs, as_, starter, batter, at_bat_number=ab))
            reliever_home = float(600 + g)
            reliever_away = float(601 + g)
            for half, batter_team, reliever, batter, evs in (
                    ("Top", away, reliever_home, away_bat, ["single", "field_out"]),
                    ("Bot", home, reliever_away, home_bat, ["strikeout", "single"])):
                for e in evs:
                    ab += 1
                    rows.append(_pitch_row(
                        g, pk, day, home, away, half, batter_team, e,
                        hs, as_, reliever, batter, at_bat_number=ab, inning=2))
            g += 1
    df = pd.DataFrame(rows)
    out_dir = Path(tempfile.mkdtemp())
    pitches = out_dir / "pitches.parquet"
    df.to_parquet(pitches, index=False)
    game_df, _pbp = build_features(pitches, out_dir)
    # Mirror load_game_features' canonical mapping so the frame carries the
    # same woba_30g_* names training consumes (the DuckDB layer ships the
    # team_-prefixed originals).
    for src, dst in (("team_woba_30g_home", "woba_30g_home"),
                     ("team_woba_30g_away", "woba_30g_away")):
        if src in game_df.columns and dst not in game_df.columns:
            game_df[dst] = game_df[src]
    return game_df


class TestSqlFormDeltas(unittest.TestCase):
    """DuckDB layer: columns ship, math is right, no season leakage."""

    @classmethod
    def setUpClass(cls):
        cls.games = build_fixture()

    def test_all_delta_columns_ship(self):
        missing = [c for c in FORM_DELTA_COLS if c not in self.games.columns]
        self.assertEqual(missing, [])
        # spec count sanity: 19 families x 2 sides = 38
        self.assertEqual(len(FORM_DELTA_COLS), 38)

    def test_sp_deltas_are_recent_minus_season(self):
        # identity against SHIPPED columns: era/k9 season baselines exist in
        # the CSV, so delta must equal 5g − season (float32 tolerance).
        for side in ("home", "away"):
            sub = self.games.dropna(subset=[
                f"sp_era_delta_{side}", f"sp_era_5g_{side}", f"sp_era_{side}"])
            self.assertGreater(len(sub), 0)
            err = (sub[f"sp_era_5g_{side}"] - sub[f"sp_era_{side}"]
                   - sub[f"sp_era_delta_{side}"]).abs().max()
            self.assertLess(float(err), 1e-4)
            subk = self.games.dropna(subset=[
                f"sp_k9_delta_{side}", f"sp_k9_5g_{side}", f"sp_k9_{side}"])
            self.assertGreater(len(subk), 0)
            errk = (subk[f"sp_k9_5g_{side}"] - subk[f"sp_k9_{side}"]
                    - subk[f"sp_k9_delta_{side}"]).abs().max()
            self.assertLess(float(errk), 1e-4)

    def test_every_family_has_late_season_coverage(self):
        # deep into 2026 every family must have observed deltas (the very
        # first in-season games are NULL by point-in-time design).
        g = self.games[pd.to_datetime(self.games["game_date"]) > "2026-04-20"]
        for col in FORM_DELTA_COLS:
            self.assertGreater(
                int(g[col].notna().sum()), 0, f"{col} all-NaN late in season")

    def test_season_partition_no_2025_leakage(self):
        # A team's 2026 season baseline must be built ONLY from 2026 priors.
        # Pick the first home row where the home team has exactly one prior
        # 2026 game: team_woba_std must equal WOBA_2026 (its single prior
        # game's wOBA), NOT a blend with 2025 (which would be ≈0.67).
        g = self.games[pd.to_datetime(self.games["game_date"]) >= "2026-04-01"].copy()
        g["game_date"] = pd.to_datetime(g["game_date"])
        g = g.sort_values("game_date")
        prior: dict[tuple[str, int], int] = {}
        for _, row in g.iterrows():
            key = (row["home_team"], row["game_date"].year)
            n = prior.get(key, 0)
            if n == 1 and pd.notna(row.get("woba_delta_home")) \
                    and pd.notna(row.get("woba_30g_home")):
                expected = float(row["woba_30g_home"]) - WOBA_2026
                self.assertAlmostEqual(
                    float(row["woba_delta_home"]), expected, places=3,
                    msg="2026 season baseline leaked 2025 games")
                break
            prior[key] = n + 1
        else:
            self.fail("no fixture row found with exactly one prior 2026 game")

    def test_season_opener_delta_is_null(self):
        # The 2026 opener has no in-season priors -> season baseline NULL ->
        # delta NULL (never a fabricated 0).
        g = self.games[pd.to_datetime(self.games["game_date"]) == "2026-04-01"]
        self.assertGreater(len(g), 0)
        for col in ("woba_delta_home", "team_barrel_delta_home",
                    "bullpen_whip_delta_home"):
            self.assertTrue(pd.isna(g[col]).all(),
                            f"{col} should be NULL on the season opener")


class TestPandasHelper(unittest.TestCase):
    def test_recent_minus_season(self):
        df = pd.DataFrame({
            "sp_era_5g_home": [3.0, 4.0],
            "sp_era_home": [3.5, 4.5],
            "sp_era_5g_away": [5.0, 5.0],
            "sp_era_away": [4.0, 5.0],
        })
        out = add_form_delta_features(df)
        self.assertEqual(list(out["sp_era_delta_home"]), [-0.5, -0.5])
        self.assertEqual(list(out["sp_era_delta_away"]), [1.0, 0.0])

    def test_nan_safe_when_one_term_missing(self):
        df = pd.DataFrame({
            "sp_era_5g_home": [3.0, np.nan, np.nan],
            "sp_era_home": [np.nan, 3.0, 4.0],
        })
        out = add_form_delta_features(df)
        self.assertTrue(pd.isna(out["sp_era_delta_home"]).all())

    def test_missing_season_baseline_creates_nan_not_fabricated(self):
        # On the committed CSV only sp_era/sp_k9 season baselines ship; a
        # family whose season column is absent gets all-NaN (never faked).
        df = pd.DataFrame({"woba_30g_home": [0.32, 0.34]})
        out = add_form_delta_features(df)
        self.assertTrue("woba_delta_home" in out.columns)
        self.assertTrue(pd.isna(out["woba_delta_home"]).all())

    def test_idempotent_existing_columns_win(self):
        # A SQL-shipped delta column is authoritative and never overwritten.
        df = pd.DataFrame({
            "sp_era_5g_home": [3.0],
            "sp_era_home": [3.5],
            "sp_era_delta_home": [99.0],  # SQL-computed value
        })
        out = add_form_delta_features(df)
        self.assertEqual(list(out["sp_era_delta_home"]), [99.0])

    def test_all_38_columns_created_when_missing(self):
        df = pd.DataFrame()
        out = add_form_delta_features(df)
        missing = [c for c in FORM_DELTA_COLS if c not in out.columns]
        self.assertEqual(missing, [])
        self.assertTrue(pd.isna(out[FORM_DELTA_COLS]).all().all())

    def test_spec_recent_and_season_bases_are_coherent(self):
        # Every spec's recent/season base must end in neither _home nor _away
        # (the helper appends the side suffix). The recent-window base must
        # be a real column on the committed artifact; the season baseline is
        # computed inside the DuckDB layer (not exported as a level), which
        # the SQL identity/coverage tests above verify. The delta is
        # computable on a frame whenever BOTH twins are present.
        from config import DATA_DELIVERY_DIR
        from data_ingestion import load_game_features
        cols = set(load_game_features(
            DATA_DELIVERY_DIR / "game_level_features.csv").columns)
        for base, recent, season, _win in FORM_DELTA_SPECS:
            self.assertFalse(recent.endswith(("_home", "_away")))
            self.assertFalse(season.endswith(("_home", "_away")))
            self.assertIn(f"{recent}_home", cols)


class TestRealCsvCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from config import DATA_DELIVERY_DIR
        from data_ingestion import load_game_features
        csv = DATA_DELIVERY_DIR / "game_level_features.csv"
        cls.games = load_game_features(csv)
        cls.games = add_form_delta_features(cls.games)

    def test_era_k9_deltas_computable_on_committed_csv(self):
        for side in ("home", "away"):
            cov = self.games[f"sp_era_delta_{side}"].notna().mean()
            self.assertGreater(cov, 0.5,
                               f"sp_era_delta_{side} coverage {cov:.3f}")
            covk = self.games[f"sp_k9_delta_{side}"].notna().mean()
            self.assertGreater(covk, 0.5,
                               f"sp_k9_delta_{side} coverage {covk:.3f}")

    def test_all_deltas_shipped_after_refresh(self):
        # The 2026-08-25 pipeline refresh (605013a) regenerated the CSV with
        # the SQL-computed deltas, fulfilling the original "NaN until refresh"
        # contract — every family now ships (identity vs recent−season is
        # proven by the SQL fixture tests; these columns stay OUT of
        # FEATURE_COLS per the don't-ship verdict).
        for base, _r, _s, _w in FORM_DELTA_SPECS:
            for side in ("home", "away"):
                col = f"{base}_{side}"
                self.assertIn(col, self.games.columns,
                              f"{col} missing from refreshed CSV")
                cov = self.games[col].notna().mean()
                self.assertGreater(
                    cov, 0.5,
                    f"{col} coverage {cov:.3f} — expected the refresh to ship it")


class TestPlumbing(unittest.TestCase):
    def test_deltas_excluded_from_active_feature_cols(self):
        # 2026-08 ablation verdict: WITH lost BOTH pooled OOF and the sealed
        # 21-day holdout vs the 58-column baseline (run_form_delta_ablation.py).
        # Per the honesty contract the deltas are NOT in the active moneyline
        # set — the wiring surface is the one-line append documented in
        # training.FEATURE_COLS, and the run engine drops them regardless.
        live = [c for c in FORM_DELTA_COLS if c in FEATURE_COLS]
        self.assertEqual(live, [])

    def test_feature_matrix_width_invariant(self):
        df = pd.DataFrame({FEATURE_COLS[0]: [1.0, 2.0]})
        X = _feature_matrix(df)
        self.assertEqual(X.shape, (2, len(FEATURE_COLS)))

    def test_logistic_routing_excludes_deltas(self):
        from training import _logistic_feature_cols
        lr = set(_logistic_feature_cols())
        live = [c for c in FORM_DELTA_COLS if c in lr]
        self.assertEqual(live, [])

    def test_metadata_provenance_for_deltas(self):
        # The deltas are NOT model features (2026-08 ablation: don't ship), so
        # the generated metadata excludes them — but authored entries exist for
        # all 38 so re-enabling FEATURE_COLS picks them up with zero work.
        import feature_metadata as fm
        meta, _warnings = fm.build_features_metadata()
        for c in FORM_DELTA_COLS:
            self.assertNotIn(c, meta)
            entry = fm._rich_entry(c)
            self.assertIsNotNone(entry, f"no authored metadata for {c}")
            self.assertIn("momentum", entry["definition"].lower())


class TestRunEngineGuardrail(unittest.TestCase):
    def test_derive_run_features_excludes_all_deltas(self):
        # Guardrail stays meaningful even though the deltas are currently out
        # of FEATURE_COLS: feed the full 58 + 38 list so that IF a future
        # re-test re-enables them, the run engine still drops every one and
        # keeps its raw-only 29-feature view byte-identical.
        full = list(FEATURE_COLS) + list(FORM_DELTA_COLS)
        kept, dropped = derive_run_features(full)
        leaked = [c for c in kept if c.endswith(("_delta_home", "_delta_away"))]
        self.assertEqual(leaked, [])
        self.assertEqual(len(kept), 29)
        self.assertEqual(len(dropped), len(full) - 29)
        dropped_deltas = [c for c in dropped if c.endswith(("_delta_home", "_delta_away"))]
        # 38 momentum deltas (never re-enabled). The 4 SHIPPED Phase-2 lineup
        # deltas (lineup_actual_*_delta_*) were removed from FEATURE_COLS on
        # 2026-08-29 (train-serve skew fix) — no longer in the input.
        self.assertEqual(len(dropped_deltas), 38)


if __name__ == "__main__":
    unittest.main()
