"""Tier-5 (v7, player-level) QB-starter identity tests — pure, no-network.

Covers the expected-QB-starter family composed by
``nfl_features.compose_tier5_qb_features`` (qb1_skill_diff /
qb1_continuity_diff / qb1_change_diff / qb1_primary_out_diff):
- PIT truncation invariance: deleting games after a date leaves earlier rows
  byte-identical; flipping a FUTURE row's starter/score/EPA changes nothing
  earlier (strictly-prior shift discipline).
- Expected-starter snapshot never reads the pbp/schedule ACTUAL starter of
  the target game for the feature value (identity comes from the depth
  chart; actuals enter only as strictly-prior facts).
- Rolling-snapshot mode (2025+): the snapshot STRICTLY BEFORE kickoff is
  used; a later snapshot is ignored.
- Coverage >= 95% (100% on a fully-charted synthetic grid); fallback chain
  (>= 4 current-season starts -> prior-season EPA -> league-average prior);
  continuity counts strictly-prior starts across seasons, capped.
- run_tier5_qb_ablation.build_arms: C0 = served 12-pool; A1/A2/A3 differ
  ONLY by the declared Tier-5 columns; tolerance constants are the shared
  nfl_moneyline rule; the 12-pool market-free invariant is untouched.

Run: python -m unittest test_nfl_tier5_qb -v   (no network needed)
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import nfl_features as nf
from nfl_features import (QB1_CONTINUITY_CAP, TIER5_QB_FEATURES,
                          compose_tier5_qb_features)
from nfl_moneyline import ECE_TOL, TOL_AUC, TOL_LL, tolerance_verdict
from run_feature_winpct_ablation import DEPLOYED_12
from run_tier5_qb_ablation import build_arms

# ---------------------------------------------------------------------------
# Synthetic world: teams A/B/C/D, seasons 2018 (warmup priors) + 2019 + 2020.
# Team A changes starters (QA1 -> QA2 -> QA2 -> QA1...); B/C/D keep one QB.
# ---------------------------------------------------------------------------
TEAMS = ["A", "B", "C", "D"]
BASE_EPA = {"QA1": 0.30, "QA2": 0.10, "QB1": 0.20, "QC1": 0.25, "QD1": 0.15,
            "QA9": 0.99}   # QA9: chart-only phantom (never actually starts)

SEASON_START = {2018: "2018-09-09", 2019: "2019-09-08", 2020: "2020-09-13",
                2024: "2024-09-08", 2025: "2025-09-07"}

# (season, week) -> games [(home, away), ...]; each team plays exactly once.
MATCHUPS = {
    (2018, 1): [("A", "B"), ("D", "C")],
    (2018, 2): [("C", "A"), ("B", "D")],
    (2019, 1): [("B", "A"), ("C", "D")],
    (2019, 2): [("A", "C"), ("D", "B")],
    (2019, 3): [("C", "B"), ("A", "D")],
    (2019, 4): [("B", "C"), ("D", "A")],
    (2019, 5): [("A", "B"), ("C", "D")],
    (2019, 6): [("D", "C"), ("B", "A")],
    (2020, 1): [("A", "D"), ("C", "B")],
    (2020, 2): [("B", "A"), ("D", "C")],
    (2024, 1): [("C", "A"), ("B", "D")],
    (2024, 2): [("A", "B"), ("D", "C")],
}

# team -> {season: {week: actual starter}} (defaults from a per-team QB)
def _actual_qb(team: str, season: int, week: int) -> str:
    if team == "A":
        if season == 2019:
            return "QA1" if week in (1, 4, 5, 6) else "QA2"
        return "QA1"   # 2018 / 2020 / 2024: QA1 every week
    return {"B": "QB1", "C": "QC1", "D": "QD1"}[team]


def _gameday(season: int, week: int) -> str:
    start = pd.Timestamp(SEASON_START[season])
    return str((start + pd.Timedelta(days=7 * (week - 1))).date())


def make_world(seasons=(2018, 2019, 2020), chart_override=None):
    """Decided schedule + pbp + weekly depth charts over the matchup grid.

    ``chart_override``: {(season, week, team): gsis_id} swaps the chart QB1
    (used to prove the identity comes from the CHART, not the actual)."""
    sched_rows = []
    for (season, week), games in MATCHUPS.items():
        if season not in seasons:
            continue
        for home, away in games:
            sched_rows.append(dict(
                game_id=f"{season}_{week}_{home}{away}", season=season,
                week=week, gameday=_gameday(season, week), game_type="REG",
                home_team=home, away_team=away,
                home_score=24, away_score=17,
                home_qb_id=_actual_qb(home, season, week),
                away_qb_id=_actual_qb(away, season, week)))
    sched = pd.DataFrame(sched_rows)
    for c in ("week", "season"):
        sched[c] = pd.to_numeric(sched[c])

    # pbp: one pass play per (game, team) by the actual starter (mean == epa)
    pbp_rows = []
    for r in sched_rows:
        for side, qb in (("home", r["home_qb_id"]), ("away", r["away_qb_id"])):
            team = r[f"{side}_team"]
            opp = r[f"{'away' if side == 'home' else 'home'}_team"]
            pbp_rows.append(dict(game_id=r["game_id"], posteam=team,
                                 defteam=opp, qb_epa=BASE_EPA[qb], epa=0.0,
                                 yards_gained=5, passer_id=qb))
    pbp = pd.DataFrame(pbp_rows)

    # weekly depth charts: QB1 = actual (or chart_override), QB2 filler
    dc_rows = []
    for (season, week), games in MATCHUPS.items():
        if season not in seasons:
            continue
        for home, away in games:
            for team in (home, away):
                qb1 = _actual_qb(team, season, week)
                qb1 = (chart_override or {}).get((season, week, team), qb1)
                dc_rows.append(dict(season=season, week=float(week),
                                    team=team, game_type="REG",
                                    position="QB", depth_team=1,
                                    gsis_id=qb1))
                dc_rows.append(dict(season=season, week=float(week),
                                    team=team, game_type="REG",
                                    position="QB", depth_team=2,
                                    gsis_id=qb1 + "b"))
    depth_weekly = pd.DataFrame(dc_rows)
    return sched, pbp, depth_weekly


def _feats(sched: pd.DataFrame, pbp: pd.DataFrame,
           depth_weekly: pd.DataFrame | None = None,
           depth_snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 12-pool feature frame + the 4 Tier-5 candidates (build + compose).

    2018 (and any other warmup seasons in the schedule) stays OUT of the
    scored frame - it only supplies strictly-prior history, like the real
    pipeline's warmup pull."""
    decided = sched[(sched["home_score"].notna())
                    & (pd.to_numeric(sched["season"]) >= 2019)].copy()
    feats = nf.build_features(decided, sched, pbp)
    return compose_tier5_qb_features(feats, sched, pbp,
                                     depth_weekly, depth_snapshots)


def _row(feats: pd.DataFrame, gid: str) -> pd.Series:
    return feats[feats["game_id"] == gid].iloc[0]


class TestBuilderBasics(unittest.TestCase):
    def setUp(self):
        sched, pbp, dcw = make_world()
        self.sched, self.pbp, self.dcw = sched, pbp, dcw
        self.feats = _feats(sched, pbp, dcw)

    def test_all_target_rows_populated(self):
        for c in TIER5_QB_FEATURES:
            self.assertEqual(self.feats[c].isna().sum(), 0, c)

    def test_columns_attach_and_registered_nowhere(self):
        for c in TIER5_QB_FEATURES:
            self.assertIn(c, self.feats.columns)
            self.assertNotIn(c, nf.FEATURE_COLUMNS)

    def test_change_flag_semantics(self):
        # 2019 w2 (A,C): home A switches QA1 -> QA2 (a change vs the w1
        # actual QA1); away C is stable -> diff = 1 - 0 = 1.
        r = _row(self.feats, "2019_2_AC")
        self.assertEqual(r["qb1_change_diff"], 1.0)
        # 2019 w3 (A,D): home A still expects QA2 == prior actual (w2 QA2)
        # -> not a change; D stable -> diff 0.
        r3 = _row(self.feats, "2019_3_AD")
        self.assertEqual(r3["qb1_change_diff"], 0.0)
        # 2019 w4 (D,A): away A expects QA1 again (QA2 started w2-3) -> a
        # change for A; home D stable -> home 0 - away 1 = -1.
        r4 = _row(self.feats, "2019_4_DA")
        self.assertEqual(r4["qb1_change_diff"], -1.0)

    def test_continuity_and_skill_fallback_chain(self):
        # 2019 w2 (A,C): home A expects brand-new QA2 -> 0 prior starts with
        # the team, no prior-season EPA -> continuity 0 / skill 0.0. Away C
        # QC1: prior starts = 2018x2 + 2019 w1 = 3; skill -> 2018 mean 0.25.
        r = _row(self.feats, "2019_2_AC")
        self.assertAlmostEqual(r["qb1_continuity_diff"], 0.0 - 3.0)
        self.assertAlmostEqual(r["qb1_skill_diff"], 0.0 - BASE_EPA["QC1"])
        # 2019 w1 (B,A): away A expects QA1 (returning). Continuity = QA1
        # starts strictly prior = 2018 x2 = 2; home B QB1 prior = 2 too -> 0.
        # Skill: current-season starts 0 < 4 on both sides -> prior-season
        # (2018) EPA: B 0.20 - A 0.30 = -0.10.
        r1 = _row(self.feats, "2019_1_BA")
        self.assertAlmostEqual(r1["qb1_continuity_diff"], 2.0 - 2.0)
        self.assertAlmostEqual(r1["qb1_skill_diff"], 0.20 - 0.30)
        # 2019 w4 (D,A): away A expects QA1 again (after QA2's w2-3 starts).
        # Continuity = QA1 prior starts = 2018x2 + 2019 w1 = 3. Home D QD1
        # prior = 2018x2 + 2019 w1-3 = 5 -> diff 5 - 3 = 2.
        r4 = _row(self.feats, "2019_4_DA")
        self.assertAlmostEqual(r4["qb1_continuity_diff"], 5.0 - 3.0)
        # 2019 w5 (A,B): home A expects QA1; current-season prior starts = 2
        # (w1, w4) < 4 -> 2018 EPA 0.30. Away B QB1: current-season prior
        # starts = 4 (w1-w4) >= 4 -> 2019 in-season mean (all 0.20) = 0.20.
        r5 = _row(self.feats, "2019_5_AB")
        self.assertAlmostEqual(r5["qb1_skill_diff"], 0.30 - 0.20)

    def test_primary_out_directional(self):
        # 2019 w2 A expects QA2; A's 2018 primary = QA1 -> home flag 1;
        # away C expects QC1 == primary QC1 -> 0. diff = 1.
        r = _row(self.feats, "2019_2_AC")
        self.assertEqual(r["qb1_primary_out_diff"], 1.0)
        # 2019 w5 home A expects QA1 == 2018 primary QA1 -> 0; B stable -> 0.
        r5 = _row(self.feats, "2019_5_AB")
        self.assertEqual(r5["qb1_primary_out_diff"], 0.0)

    def test_strictly_prior_continuity(self):
        # A 2019 w4 (QA1 returns after QA2's w2-3 starts): continuity counts
        # QA1's starts strictly before w4 = 2 (2018) + 1 (w1) = 3 - never the
        # future w5/w6 starts and never QA2's w2-3 starts. Home D prior = 5.
        r4 = _row(self.feats, "2019_4_DA")
        self.assertAlmostEqual(r4["qb1_continuity_diff"], 5.0 - 3.0)


class TestPitInvariance(unittest.TestCase):
    def _truncated(self):
        sched, pbp, dcw = make_world()
        cut = pd.Timestamp(_gameday(2019, 4))   # drop everything >= 2019 w4
        sched_cut = sched[pd.to_datetime(sched["gameday"]) < cut].reset_index(drop=True)
        sched_cut["week"] = pd.to_numeric(sched_cut["week"])
        pbp_cut = pbp[pbp["game_id"].isin(set(sched_cut["game_id"]))].copy()
        dcw_cut = dcw[dcw["week"] < 4.0].copy()  # charts through w3 (2019/20
        # charts for later weeks never resolve for earlier rows anyway)
        feats_cut = _feats(sched_cut, pbp_cut, dcw_cut)
        return feats_cut

    def test_deleting_future_games_leaves_earlier_rows_identical(self):
        sched, pbp, dcw = make_world()
        feats_full = _feats(sched, pbp, dcw)
        feats_cut = self._truncated()
        before = pd.to_datetime(sched["gameday"]) < pd.Timestamp(_gameday(2019, 4))
        gids = set(sched.loc[before, "game_id"])
        a = feats_full[feats_full["game_id"].isin(gids)].set_index("game_id")
        b = feats_cut.set_index("game_id")
        for c in TIER5_QB_FEATURES:
            x, y = a[c].astype(float), b[c].astype(float)
            both = x.notna() & y.notna()
            self.assertEqual(int((x.isna() != y.isna()).sum()), 0, c)
            self.assertAlmostEqual(float((x[both] - y[both]).abs().max()), 0.0,
                                   places=12, msg=c)

    def test_future_flips_do_not_touch_earlier_rows(self):
        sched, pbp, dcw = make_world()
        feats_orig = _feats(sched, pbp, dcw)
        # Flip a FUTURE game entirely: different actual starter (QA2 keeps
        # starting), different EPA, different score, different chart QB1.
        sched2 = sched.copy()
        mask = sched2["game_id"] == "2019_5_AB"
        sched2.loc[mask, "home_qb_id"] = "QA2"
        sched2.loc[mask, "home_score"] = 7
        sched2.loc[mask, "away_score"] = 41
        pbp2 = pbp.copy()
        pbp2.loc[pbp2["game_id"] == "2019_5_AB", "qb_epa"] = -1.0
        dcw2 = dcw.copy()
        dcw2.loc[(dcw2["season"] == 2019) & (dcw2["week"] == 5.0) &
                 (dcw2["team"] == "A") & (dcw2["depth_team"] == 1),
                 "gsis_id"] = "QA9"
        feats2 = _feats(sched2, pbp2, dcw2)
        before = pd.to_datetime(sched["gameday"]) < pd.Timestamp(_gameday(2019, 5))
        gids = set(sched.loc[before, "game_id"])
        a = feats_orig[feats_orig["game_id"].isin(gids)].set_index("game_id")
        b = feats2[feats2["game_id"].isin(gids)].set_index("game_id")
        for c in TIER5_QB_FEATURES:
            x, y = a[c].astype(float), b[c].astype(float)
            both = x.notna() & y.notna()
            self.assertEqual(int((x.isna() != y.isna()).sum()), 0, c)
            self.assertAlmostEqual(float((x[both] - y[both]).abs().max()), 0.0,
                                   places=12, msg=c)


class TestExpectedStarterDiscipline(unittest.TestCase):
    def test_identity_comes_from_chart_not_actual(self):
        # 2019 w3: chart QB1 = QA9 (a phantom who NEVER starts), while the
        # ACTUAL starter (schedule + pbp) is QA2. Every value must reflect
        # QA9 (the published expectation), never QA2.
        sched, pbp, dcw = make_world(chart_override={(2019, 3, "A"): "QA9"})
        feats = _feats(sched, pbp, dcw)
        # 2019 w3 A@D: away A expected QA9. A's prior actual (w2) = QA2.
        r = _row(feats, "2019_3_AD")
        # change = 1 (QA9 != prior actual QA2) - if the builder had used the
        # ACTUAL (QA2), change would be 0 (QA2 == prior actual QA2).
        self.assertEqual(r["qb1_change_diff"], 1.0)
        # continuity = QA9 prior starts with A = 0 (never started) - if the
        # builder had used actual QA2 it would read 1. Away D QD1 prior = 4
        # (2018x2 + 2019 w1 + w2).
        self.assertAlmostEqual(r["qb1_continuity_diff"], 0.0 - 4.0)
        # skill = QA9 has no prior-season starts -> replacement 0.0 (actual
        # QA2 would give 0.10); away D QD1 -> 2018 mean 0.15.
        self.assertAlmostEqual(r["qb1_skill_diff"], 0.0 - BASE_EPA["QD1"])

    def test_rolling_snapshot_uses_last_state_strictly_before_kickoff(self):
        # 2025 mode: no weekly charts; rolling snapshots for team A only.
        # Latest pre-kickoff state = QA1; a LATER snapshot (QA7) must be
        # ignored because it post-dates kickoff.
        sched, pbp, dcw = make_world(seasons=(2018, 2019, 2020, 2024, 2025))
        # 2025 chart QB1 rows exist? make_world emits charts for listed
        # seasons; drop any 2025 weekly rows (the 2025 feed is snapshot-only).
        dcw = dcw[dcw["season"] != 2025]
        snap_rows = [
            dict(dt="2025-08-01T10:00:00Z", team="A", gsis_id="QA9",
                 pos_abb="QB", pos_rank=1),
            dict(dt="2025-09-10T10:00:00Z", team="A", gsis_id="QA1",
                 pos_abb="QB", pos_rank=1),     # pre-kickoff -> should win
            dict(dt="2025-09-20T10:00:00Z", team="A", gsis_id="QA7",
                 pos_abb="QB", pos_rank=1),     # post-kickoff -> ignored
            dict(dt="2025-09-10T10:00:00Z", team="B", gsis_id="QB1",
                 pos_abb="QB", pos_rank=1),
        ]
        snaps = pd.DataFrame(snap_rows)
        # The 2025 grid needs matchups: add one 2025 w1 game A@B by hand.
        sched_extra = pd.DataFrame([dict(
            game_id="2025_1_BA", season=2025, week=1,
            gameday="2025-09-14", game_type="REG",
            home_team="B", away_team="A", home_score=20, away_score=17,
            home_qb_id="QB1", away_qb_id="QA1",
            # gametime ET drives the UTC kickoff used for the snapshot cut
            gametime="13:00")])
        sched = pd.concat([sched[sched["season"] != 2025], sched_extra],
                          ignore_index=True)
        feats = _feats(sched, pbp, dcw, depth_snapshots=snaps)
        r = _row(feats, "2025_1_BA")
        # away A expected QA1 == A's prior actual (2020 w2 = QA1) -> no change
        self.assertEqual(r["qb1_change_diff"], 0.0)
        # QA7 (the post-kickoff snapshot) must NOT have been used: if it had,
        # the change flag would be 1 (QA7 != QA1 prior). Continuity: home B
        # QB1 prior starts = 2018x2 + 2019x6 + 2020x2 + 2024x2 = 12; away A
        # QA1 prior = 2018x2 + 2019x4 + 2020x2 + 2024x2 = 10 -> diff 2.0.
        self.assertEqual(r["qb1_continuity_diff"], 2.0)
        self.assertEqual(r["qb1_skill_diff"], 0.20 - 0.30)
        # (skill: home B QB1 current-season(2025) starts = 0 < 4 -> prior-
        # season 2024 mean 0.20; away A QA1 -> 2024 mean 0.30.)


class TestCoverageAndArms(unittest.TestCase):
    def test_full_coverage_grid(self):
        sched, pbp, dcw = make_world()
        feats = _feats(sched, pbp, dcw)
        for c in TIER5_QB_FEATURES:
            self.assertGreaterEqual(feats[c].notna().mean(), 0.95, c)

    def test_arms_composition(self):
        carrier = pd.DataFrame({c: [1.0]
                                for c in set(DEPLOYED_12) | set(TIER5_QB_FEATURES)})
        arms = build_arms(carrier)
        self.assertEqual(arms["C0"], DEPLOYED_12)
        self.assertEqual(set(arms["A1"]) - set(arms["C0"]),
                         {"qb1_skill_diff"})
        self.assertEqual(set(arms["A2"]) - set(arms["C0"]),
                         {"qb1_continuity_diff", "qb1_change_diff",
                          "qb1_primary_out_diff"})
        self.assertEqual(set(arms["A3"]) - set(arms["C0"]),
                         set(TIER5_QB_FEATURES))
        self.assertEqual(len(arms["C0"]), 12)

    def test_served_pool_invariant_untouched(self):
        self.assertEqual(DEPLOYED_12, [f for f in nf.FEATURE_COLUMNS
                                       if f != "is_home"])
        for c in nf.FEATURE_COLUMNS:
            self.assertNotIn("market", c.lower())
            self.assertNotIn("moneyline", c.lower())
            self.assertNotIn("spread", c.lower())

    def test_shared_tolerance_rule(self):
        # The harness records the same tolerance constants as the ONE shared
        # nfl_moneyline gate rule (MLB-aligned): nothing local re-derived.
        self.assertEqual(TOL_LL, 0.012)
        self.assertEqual(TOL_AUC, 0.016)
        self.assertEqual(ECE_TOL, 0.01)
        self.assertTrue(callable(tolerance_verdict))


if __name__ == "__main__":
    unittest.main()
