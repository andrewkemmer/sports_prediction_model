"""Tier-4 (v6) candidate + harness tests — pure, no-network.

Covers the game-script (GS), opponent-adjusted, drive-level, and
QB-conditional PBP families composed by ``nfl_tier4``:
- GS mask: nflfastR-style garbage-time classification via ``wp`` (Q1-3 bar
  0.01/0.99, Q4 bar 0.05/0.95, OT never garbage, missing wp kept), and the
  deliberate use of ``wp`` — never ``vegas_wp`` (market-independence policy).
- qb_map_from_schedule: (game_id, team) -> announced/recorded starter id,
  skipping pending rows (no id -> the QB axis stays NaN on the slate).
- tier4_team_agg: non-garbage sums/counts, non-garbage net points (own minus
  allowed), drive rates, and the starter restriction.
- compose_tier4_features: all 10 candidates attach to a built frame; slate
  (pending) rows get every candidate from strictly-prior decided games,
  including QB-conditional (the trailing shift uses only PAST games' recorded
  starters — the current game's starter is never assumed or faked).
- run_tier4_ablation.build_arms: WITHOUT = deployed 12; GS/OPPADJ/GSOPP/DRIVE/
  QB compose as 15/15/18/15/13; conditional arms are skipped below the 95%
  coverage floor.
- Invariant: none of the Tier-4 candidates is in FEATURE_COLUMNS
  (composed-but-unregistered until the sealed-2025 ablation admits them).

Run: python -m unittest test_nfl_tier4 -v   (no network needed)
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import nfl_features as nf
from nfl_tier4 import (TIER4_CANDIDATES, TIER4_DRIVE_FEATURES,
                       TIER4_GS_FEATURES, TIER4_OPPADJ_FEATURES,
                       TIER4_QB_FEATURES, compose_tier4_features,
                       non_garbage_mask, qb_map_from_schedule,
                       tier4_team_agg)
from run_feature_winpct_ablation import DEPLOYED_12
from run_tier4_ablation import build_arms


def _pbp_row(**kw) -> dict:
    base = dict(game_id="g1", posteam="A", defteam="B", qb_epa=0.0, epa=0.0,
                yards_gained=0, drive=1, wp=0.6, qtr=2, touchdown=0,
                field_goal_result=np.nan, passer_id=None)
    base.update(kw)
    return base


def _offense(post: str, deff: str, **kw) -> dict:
    """An offensive play for ``post`` against ``deff`` (real PBP semantics:
    defteam = the defending team)."""
    return _pbp_row(posteam=post, defteam=deff, **kw)


class TestNonGarbageMask(unittest.TestCase):
    def test_nflfastr_bars(self):
        pbp = pd.DataFrame([
            _pbp_row(game_id="g1", posteam="A", wp=0.03, qtr=4),   # Q4 garbage
            _pbp_row(game_id="g1", posteam="B", wp=0.03, qtr=1),   # Q1: keep
            _pbp_row(game_id="g1", posteam="A", wp=0.995, qtr=2),  # >=0.99 garbage
            _pbp_row(game_id="g1", posteam="B", wp=0.5, qtr=3),    # keep
            _pbp_row(game_id="g1", posteam="A", wp=0.03, qtr=5),   # OT never garbage
            _pbp_row(game_id="g1", posteam="B", wp=np.nan, qtr=4),  # missing kept
        ])
        m = non_garbage_mask(pbp)
        self.assertEqual(list(m), [False, True, False, True, True, True])

    def test_absent_columns_return_none(self):
        pbp = pd.DataFrame([dict(game_id="g1", posteam="A")])
        self.assertIsNone(non_garbage_mask(pbp))
        self.assertIsNone(non_garbage_mask(pd.DataFrame()))


class TestQbMap(unittest.TestCase):
    def test_map_and_pending_skip(self):
        sched = pd.DataFrame([
            dict(game_id="g1", home_team="A", away_team="B",
                 home_qb_id="qA", away_qb_id="qB"),
            dict(game_id="g2", home_team="A", away_team="B",
                 home_qb_id=None, away_qb_id=None),   # pending: no starters
        ])
        m = qb_map_from_schedule(sched)
        self.assertEqual(m, {("g1", "A"): "qA", ("g1", "B"): "qB"})
        self.assertEqual(qb_map_from_schedule(None), {})


class TestTier4TeamAgg(unittest.TestCase):
    def test_gs_drive_starter_aggregates(self):
        pbp = pd.DataFrame([
            # --- g1: team A scores a TD in non-garbage, a TD in garbage ---
            _offense("A", "B", game_id="g1", qb_epa=0.4, epa=0.5,
                     yards_gained=25, drive=1, wp=0.6, qtr=2,
                     touchdown=1, passer_id="qA"),
            _offense("A", "B", game_id="g1", qb_epa=0.2, epa=0.3,
                     yards_gained=8, drive=2, wp=0.02, qtr=4,
                     touchdown=1, passer_id="qA"),          # GARBAGE Q4
            _offense("B", "A", game_id="g1", qb_epa=0.1, epa=0.1,
                     yards_gained=4, drive=3, wp=0.6, qtr=2,
                     touchdown=0, passer_id="qB"),
            _offense("B", "A", game_id="g1", qb_epa=-0.3, epa=-0.2,
                     yards_gained=-2, drive=4, wp=0.8, qtr=3,
                     touchdown=0, passer_id="qZ"),          # backup QB
            # --- g2: field goal in non-garbage ---
            _offense("A", "B", game_id="g2", qb_epa=0.0, epa=0.0,
                     yards_gained=10, drive=1, wp=0.5, qtr=1,
                     touchdown=0, field_goal_result="made", passer_id="qA"),
        ])
        qb_map = {("g1", "A"): "qA", ("g1", "B"): "qB", ("g2", "A"): "qA"}
        out = tier4_team_agg(pbp, qb_map)
        a1 = out[(out["game_id"] == "g1") & (out["team"] == "A")].iloc[0]
        b1 = out[(out["game_id"] == "g1") & (out["team"] == "B")].iloc[0]
        # GS: A's garbage TD excluded -> qb_epa_sum_gs = 0.4, n=1, yds=25
        self.assertAlmostEqual(a1["qb_epa_sum_gs"], 0.4)
        self.assertEqual(a1["qb_epa_n_gs"], 1)
        self.assertAlmostEqual(a1["total_yards_gs"], 25.0)
        self.assertEqual(a1["n_plays_gs"], 1)
        # B has 2 non-garbage plays (garbage flag only fired for A's Q4 play)
        self.assertEqual(b1["qb_epa_n_gs"], 2)
        # Net points in non-garbage time: A scored 7 (one non-garbage TD);
        # B allowed 7 (defteam side) -> A net +7, B net -7.
        self.assertAlmostEqual(a1["pts_scored_gs"], 7.0)
        self.assertAlmostEqual(a1["pts_allowed_gs"], 0.0)
        self.assertAlmostEqual(b1["pts_allowed_gs"], 7.0)
        # Drives: A ran drives 1,2 -> n_drives 2; yds_per_drive = 33/2.
        self.assertEqual(a1["n_drives"], 2)
        self.assertAlmostEqual(a1["yds_per_drive"], 16.5)
        # Starter: A = only qA plays (both) -> qb_epa_sum_start 0.6, n 2;
        # B = qB only (qZ excluded) -> sum 0.1, n 1.
        self.assertAlmostEqual(a1["qb_epa_sum_start"], 0.6)
        self.assertEqual(a1["qb_epa_n_start"], 2)
        self.assertAlmostEqual(b1["qb_epa_sum_start"], 0.1)
        self.assertEqual(b1["qb_epa_n_start"], 1)
        # g2 A: FG made -> pts_scored_gs 3.
        a2 = out[(out["game_id"] == "g2") & (out["team"] == "A")].iloc[0]
        self.assertAlmostEqual(a2["pts_scored_gs"], 3.0)

    def test_absent_source_degrades_to_nan(self):
        out = tier4_team_agg(pd.DataFrame({"game_id": ["g1"],
                                           "posteam": ["A"],
                                           "yards_gained": [5.0]}))
        self.assertTrue(pd.isna(out["qb_epa_sum_gs"].iloc[0]))
        self.assertTrue(pd.isna(out["qb_epa_n_start"].iloc[0]))
        self.assertTrue(pd.isna(out["n_drives"].iloc[0]))   # drive absent -> NaN
        self.assertTrue(pd.isna(out["yds_per_drive"].iloc[0]))

    def test_empty_input(self):
        out = tier4_team_agg(None)
        self.assertEqual(list(out.columns),
                         list(__import__("nfl_tier4", fromlist=["TIER4_AGG_COLUMNS"])
                              .TIER4_AGG_COLUMNS))


def _synth_schedule(rows: list[dict]) -> pd.DataFrame:
    default = dict(season=2021, week=1, gameday="2021-09-12",
                   home_score=None, away_score=None)
    out = pd.DataFrame([{**default, **r} for r in rows])
    # enforce strictly-increasing gamedays per row index (the ladder asserts
    # strict monotonicity within each team)
    for i in range(len(out)):
        if "gameday" not in rows[i]:
            out.loc[out.index[i], "gameday"] = (
                pd.Timestamp("2021-09-12") + pd.Timedelta(days=7 * i))
    return out


def _small_pbp() -> pd.DataFrame:
    rows = []
    for gid, home, away, wday, hqb, aqb in [
            ("g1", "X", "Y", "2021-09-12", "qX", "qY"),
            ("g2", "Y", "X", "2021-09-19", "qY", "qX"),
            ("g3", "X", "Y", "2021-09-26", "qX", "qY")]:
        for team, opp, qb in ((home, away, hqb), (away, home, aqb)):
            for k in range(3):
                rows.append(_pbp_row(
                    game_id=gid, posteam=team, defteam=opp, qb_epa=0.3 + k * 0.1,
                    epa=0.4 + k * 0.1, yards_gained=6 + k * 4, drive=k + 1,
                    wp=0.5 + k * 0.1, qtr=2, touchdown=0,
                    field_goal_result=np.nan, passer_id=qb))
    return pd.DataFrame(rows)


class TestComposeTier4Features(unittest.TestCase):
    def test_decided_frame_all_candidates(self):
        sched = _synth_schedule([
            dict(game_id="g1", home_team="X", away_team="Y",
                 home_score=24, away_score=17, home_qb_id="qX", away_qb_id="qY"),
            dict(game_id="g2", home_team="Y", away_team="X",
                 home_score=10, away_score=20, home_qb_id="qY", away_qb_id="qX"),
            dict(game_id="g3", home_team="X", away_team="Y",
                 home_score=27, away_score=20, home_qb_id="qX", away_qb_id="qY"),
        ])
        decided = sched.copy()
        pbp = _small_pbp()
        feats = nf.build_features(decided, sched, pbp)
        out = compose_tier4_features(feats, sched, pbp)
        for c in TIER4_CANDIDATES:
            self.assertIn(c, out.columns)
        # g1 has no strictly-prior games -> all trailing tier4 values NaN.
        g1 = out[out["game_id"] == "g1"].iloc[0]
        for c in TIER4_CANDIDATES:
            self.assertTrue(pd.isna(g1[c]), c)
        # g3 has two strictly-prior games -> every candidate populated
        # (opp-adj included: the prior opponents each had >= 1 prior game by
        # then). g2's opp-adj is NaN by construction (its only prior game's
        # opponent had no history) — the 2018 warmup resolves this in real
        # runs; GS/drive/QB need only OWN-team history and are populated.
        g3 = out[out["game_id"] == "g3"].iloc[0]
        for c in TIER4_CANDIDATES:
            self.assertTrue(pd.notna(g3[c]), c)
        g2 = out[out["game_id"] == "g2"].iloc[0]
        for c in (TIER4_GS_FEATURES + TIER4_DRIVE_FEATURES
                  + TIER4_QB_FEATURES):
            self.assertTrue(pd.notna(g2[c]), c)

    def test_slate_rows_gs_oppadj_drive_but_qb_nan(self):
        sched = _synth_schedule([
            dict(game_id="g1", home_team="X", away_team="Y",
                 home_score=24, away_score=17, home_qb_id="qX", away_qb_id="qY"),
            dict(game_id="g2", home_team="Y", away_team="X",
                 home_score=10, away_score=20, home_qb_id="qY", away_qb_id="qX"),
            dict(game_id="g3", home_team="X", away_team="Y",
                 home_score=27, away_score=20, home_qb_id="qX", away_qb_id="qY"),
            # pending slate game: no scores, no announced starter
            dict(game_id="g4", home_team="Y", away_team="X",
                 home_score=None, away_score=None, home_qb_id=None, away_qb_id=None),
        ])
        decided = sched[sched["home_score"].notna()]
        pbp = _small_pbp()
        # build_slate_features returns the PENDING rows (the slate path); the
        # Tier-4 seam then attaches the candidates.
        feats = nf.build_slate_features(sched, pbp, decided, slate_season=2021)
        out = compose_tier4_features(feats, sched, pbp)
        self.assertEqual(list(out["game_id"]), ["g4"])
        g4 = out.iloc[0]
        for c in TIER4_CANDIDATES:
            self.assertTrue(pd.notna(g4[c]), c)  # incl. QB-conditional: the
        # trailing shift uses only PAST games' recorded starters (g1-g3), so
        # the slate row is populated honestly — the current game's own
        # starter is never assumed or faked.

    def test_missing_pbp_degrades_to_nan(self):
        sched = _synth_schedule([
            dict(game_id="g1", home_team="X", away_team="Y",
                 home_score=24, away_score=17)])
        feats = nf.build_features(sched.copy(), sched, None)
        out = compose_tier4_features(feats, sched, None)
        for c in TIER4_CANDIDATES:
            self.assertTrue(out[c].isna().all(), c)


class TestTier4Arms(unittest.TestCase):
    def _feats(self, with_t4: bool) -> pd.DataFrame:
        cols = dict.fromkeys(DEPLOYED_12, 1.0)
        if with_t4:
            for c in TIER4_CANDIDATES:
                cols[c] = 1.0
        return pd.DataFrame([cols])

    def test_arm_composition(self):
        arms = build_arms(self._feats(True))
        self.assertEqual(len(arms["WITHOUT"]), 12)
        self.assertEqual(len(arms["GS"]), 15)
        self.assertEqual(len(arms["OPPADJ"]), 15)
        self.assertEqual(len(arms["GSOPP"]), 18)
        self.assertEqual(len(arms["DRIVE"]), 15)
        self.assertEqual(len(arms["QB"]), 13)
        self.assertEqual(set(arms["GS"]) - set(arms["WITHOUT"]),
                         set(TIER4_GS_FEATURES))

    def test_conditional_arms_skip_when_coverage_below_floor(self):
        feats = self._feats(True)
        feats["ewm_qb_epa_starter_diff"] = np.nan   # QB coverage 0 -> skip
        arms = build_arms(feats)
        self.assertNotIn("QB", arms)
        self.assertIn("DRIVE", arms)

    def test_absent_columns_dropped(self):
        arms = build_arms(self._feats(False))
        self.assertEqual(arms["WITHOUT"], DEPLOYED_12)
        self.assertEqual(arms["GS"], DEPLOYED_12)      # no tier4 cols -> base only
        self.assertNotIn("DRIVE", arms)


class TestUnregisteredInvariant(unittest.TestCase):
    def test_tier4_not_in_served_pool(self):
        for c in TIER4_CANDIDATES:
            self.assertNotIn(c, nf.FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
