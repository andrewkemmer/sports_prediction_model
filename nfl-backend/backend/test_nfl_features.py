"""NFL feature engineering v1 (``nfl_features.py``) — pure, no-network tests.

Pins the admission gate's invariants so feature ADMISSION never leaks the
future or admits a redundant candidate silently:
- ELO update rule + prior, and strict no-future use (a later game can never
  change an earlier game's entering rating / trailing stats).
- Trailing windows use ONLY strictly-prior games (per-team shift), and the
  code-level strict-monotonicity assertion actually fires on bad input.
- ``univariate_auc`` correctness (perfect separation = 1, inverse = 0, random ~0.5).
- coverage floor drops <95%-coverage features.
- ``build_features`` composes the dome flag, ELO/form/rest/ypp diffs end-to-end.
"""
import unittest
from datetime import date

import numpy as np
import pandas as pd

from nfl_features import (
    COVERAGE_FLOOR, FEATURE_COLUMNS, build_features, build_slate_features,
    compute_elo, run_feature_gate, team_events, team_stats_ladder,
    univariate_auc,
)


def synth_games(rows: list[dict]) -> pd.DataFrame:
    """Decided-shaped frame with the columns build_features/team_events need."""
    default = {
        "game_id": None, "season": 2019, "week": 1, "game_type": "REG",
        "gameday": None, "home_team": None, "away_team": None,
        "home_score": None, "away_score": None, "roof": "outdoors",
    }
    games = []
    for r in rows:
        d = dict(default)
        d.update(r)
        games.append(d)
    return pd.DataFrame(games)


class TestElo(unittest.TestCase):
    def test_update_rule_and_prior(self):
        """Two teams, one game, A wins: exp=0.5 for both (equal ratings) ->
        A 1516, B 1484. Entering ratings equal the prior (1500)."""
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=24, away_score=10),
        ])
        ev = compute_elo(team_events(g))
        a = ev[ev["team"] == "A"].iloc[0]
        b = ev[ev["team"] == "B"].iloc[0]
        self.assertEqual(a["elo_entering"], 1500.0)
        self.assertEqual(b["elo_entering"], 1500.0)
        # recompute post-game ratings (not stored) by replaying the update
        self.assertAlmostEqual(1500 + 32 * (a["team_win"] - 0.5), 1516.0, places=3)
        self.assertAlmostEqual(1500 + 32 * (b["team_win"] - 0.5), 1484.0, places=3)

    def test_entering_rating_uses_only_prior_games(self):
        """Ratings entering game k equal the ratings AFTER game k-1 for that
        team, so no future game can influence an earlier game."""
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=24, away_score=10),
            dict(game_id="G2", gameday="2019-09-15", home_team="B", away_team="A",
                 home_score=7, away_score=21),   # A wins again
            dict(game_id="G3", gameday="2019-09-22", home_team="A", away_team="C",
                 home_score=3, away_score=30),   # A gets blown out (future for G2)
        ])
        ev = compute_elo(team_events(g))
        # G2's entering ratings must reflect ONLY G1 (G2 is the 2nd A/B game).
        g2 = ev[ev["game_id"] == "G2"]
        self.assertEqual(g2[g2["team"] == "A"]["elo_entering"].iloc[0], 1516.0)
        self.assertEqual(g2[g2["team"] == "B"]["elo_entering"].iloc[0], 1484.0)


class TestTrailingLeakage(unittest.TestCase):
    def test_trailing_uses_only_strictly_prior_games(self):
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),   # A += 11
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=14, away_score=20),   # A += -6
            dict(game_id="G3", gameday="2019-09-15", home_team="A", away_team="D",
                 home_score=28, away_score=24),   # A += 4
        ])
        ladder = team_stats_ladder(team_events(g)).set_index(["game_id", "team"])
        # G3's trailing form (window 4) = mean over G1,G2 only = (11 + -6)/2 = 2.5
        self.assertAlmostEqual(ladder.loc[("G3", "A"), "form_pts"], 2.5, places=6)
        # G2's trailing form = just G1 = 11.0 (not affected by the later G3)
        self.assertAlmostEqual(ladder.loc[("G2", "A"), "form_pts"], 11.0, places=6)
        self.assertTrue(pd.isna(ladder.loc[("G1", "A"), "form_pts"]))

    def test_appending_future_game_does_not_change_earlier_stats(self):
        g1 = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=14, away_score=20),
        ])
        base = team_stats_ladder(team_events(g1)).set_index(["game_id", "team"])
        with_future = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=14, away_score=20),
            dict(game_id="G3", gameday="2019-12-01", home_team="A", away_team="D",
                 home_score=70, away_score=0),   # extreme future blowout
        ])
        alt = team_stats_ladder(team_events(with_future)).set_index(["game_id", "team"])
        for stat in ("form_pts", "win_pct", "rest_days", "ypp"):
            for gid in ("G1", "G2"):
                v_base = base.loc[(gid, "A"), stat]
                v_alt = alt.loc[(gid, "A"), stat]
                if pd.isna(v_base) and pd.isna(v_alt):
                    continue
                self.assertEqual(v_base, v_alt,
                                 f"{stat} of {gid} changed by a future game")

    def test_rest_days_diff(self):
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=7, away_score=14),
        ])
        ladder = team_stats_ladder(team_events(g)).set_index(["game_id", "team"])
        self.assertEqual(ladder.loc[("G2", "A"), "rest_days"], 7)
        self.assertEqual(ladder.loc[("G2", "B"), "rest_days"], 7)
        self.assertTrue(pd.isna(ladder.loc[("G1", "A"), "rest_days"]))

    def test_strict_monotonicity_assertion_fires(self):
        """Two games with the SAME gameday for one team must raise — proves the
        leakage assertion is live, not decorative."""
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-08", home_team="A", away_team="B",
                 home_score=21, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=7, away_score=3),   # A plays twice the same day
        ])
        with self.assertRaises(AssertionError):
            team_stats_ladder(team_events(g))


class TestAuc(unittest.TestCase):
    def test_perfect_separation(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        x = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
        self.assertAlmostEqual(univariate_auc(y, x), 1.0, places=6)

    def test_inverse_separation(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        x = -np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
        self.assertAlmostEqual(univariate_auc(y, x), 0.0, places=6)

    def test_random_and_nan_handling(self):
        rng = np.random.default_rng(7)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        x = rng.normal(size=8)
        self.assertAlmostEqual(univariate_auc(y, x), 0.5, delta=0.2)
        self.assertTrue(np.isnan(univariate_auc(np.array([1, 1]), np.array([2.0, np.nan]))))


class TestGate(unittest.TestCase):
    def _frame(self):
        frame = synth_games([dict(game_id=f"G{i}", gameday=f"2019-09-{1+2*i:02d}",
                                  home_team="H", away_team="A",
                                  home_score=24, away_score=10) for i in range(20)])
        frame["season"] = 2019
        for f in FEATURE_COLUMNS:
            frame[f] = 0.0                            # fully covered, but constant
        frame["elo_diff"] = np.linspace(-2, 2, len(frame))
        # make three REGISTERED features 50%-covered (feature names must be
        # in the served FEATURE_COLUMNS pool — the legacy twins are no longer
        # registered)
        half = np.array([i % 2 == 0 for i in range(len(frame))])
        for f in ("win_pct_diff", "rest_days_diff", "is_dome_home"):
            frame[f] = frame[f].mask(half)
        return frame

    def test_no_prune_keeps_pool_and_warns_below_floor(self):
        """Default policy (GATE_AUTO_PRUNE=False): the served pool is exactly
        FEATURE_COLUMNS minus the anchor — features below the coverage floor
        are REPORTED (below_coverage_floor) but never removed."""
        from nfl_features import GATE_AUTO_PRUNE, run_feature_gate
        self.assertFalse(GATE_AUTO_PRUNE)             # the user policy default
        res = run_feature_gate(self._frame())
        self.assertEqual(res["dropped"], [])
        for f in ("win_pct_diff", "rest_days_diff", "is_dome_home"):
            self.assertIn(f, res["below_coverage_floor"])
            self.assertIn(f, res["v1_features"])     # still served
        self.assertNotIn("is_home", res["v1_features"])
        self.assertEqual(len(res["v1_features"]), 14)
        self.assertFalse(res["auto_prune"])

    def test_legacy_prune_is_explicit_opt_in(self):
        """auto_prune=True preserves the LEGACY pruning behavior — bare
        coverage drops + redundant-pair pruning with recorded reasons."""
        from nfl_features import run_feature_gate
        res = run_feature_gate(self._frame(), auto_prune=True)
        for f in ("win_pct_diff", "rest_days_diff", "is_dome_home"):
            self.assertIn(f, res["dropped"], f"{f} should be dropped for coverage")
            self.assertIn("coverage", res["reasons"][f])
        self.assertIn("elo_diff", res["v1_features"])
        self.assertTrue(res["auto_prune"])


class TestV2TrailingLeakage(unittest.TestCase):
    """The v2 decaying-window / opponent-adjusted / pace candidates must obey
    the same strictly-prior discipline as the v1 windowed features."""

    def _pbp(self, game_id, plays):
        rows = []
        for team, yds, epa, qb_epa, clock in plays:
            rows.append(dict(game_id=game_id, posteam=team, yards_gained=yds,
                             epa=epa, qb_epa=qb_epa, game_seconds_remaining=clock))
        return pd.DataFrame(rows)

    def test_ewm_uses_only_strictly_prior_games(self):
        """ewm values at game k must equal the ewm of PRIOR games' values,
        and appending a future game must not change earlier rows."""
        g1 = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=30, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="B", away_team="A",
                 home_score=7, away_score=21),
            dict(game_id="G3", gameday="2019-09-15", home_team="A", away_team="C",
                 home_score=14, away_score=10),
        ])
        base = team_stats_ladder(team_events(g1)).set_index(["game_id", "team"])
        # append a FUTURE game for A (would leak if it changed G2/G3 rows)
        g2 = synth_games([
            dict(game_id="G4", gameday="2019-09-22", home_team="A", away_team="D",
                 home_score=3, away_score=40),
        ])
        alt = team_stats_ladder(team_events(pd.concat([g1, g2], ignore_index=True)))\
            .set_index(["game_id", "team"])
        for gid in ("G1", "G2", "G3"):
            for team in ("A", "B", "C"):
                if (gid, team) not in base.index:
                    continue
                v1 = base.loc[(gid, team), "ewm_net_pts"]
                v2 = alt.loc[(gid, team), "ewm_net_pts"]
                if pd.isna(v1) and pd.isna(v2):
                    continue  # both NaN = equal
                self.assertEqual(v1, v2)

    def test_ewm_first_game_nan_and_window_small(self):
        """A team's FIRST game has no prior ewm (NaN); a big early blowout
        decays within a few games (halflife=2), so ewm stays recent-form."""
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=40, away_score=3),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=10, away_score=9),
            dict(game_id="G3", gameday="2019-09-15", home_team="A", away_team="D",
                 home_score=12, away_score=7),
        ])
        ladder = team_stats_ladder(team_events(g))
        a = ladder[ladder["team"] == "A"].set_index("game_id")
        self.assertTrue(pd.isna(a.loc["G1", "ewm_net_pts"]))       # no prior
        self.assertEqual(a.loc["G2", "ewm_net_pts"], 37.0)          # only G1
        self.assertGreater(a.loc["G2", "ewm_net_pts"],
                           a.loc["G3", "ewm_net_pts"])              # decays
        self.assertGreater(a.loc["G3", "ewm_net_pts"], 1.0)         # still positive

    def test_opponent_adjusted_margin_uses_prior_opponents(self):
        """opp_adj_form subtracts the trailing form of the OPPONENTS faced
        (strictly-prior), so a team that faced weak opponents scores HIGHER
        (its raw margin is discounted)."""
        g = synth_games([
            # priors so every opponent has real trailing form
            dict(game_id="G0", gameday="2019-08-25", home_team="B", away_team="Z",
                 home_score=24, away_score=14),     # B +10
            dict(game_id="G0c", gameday="2019-08-25", home_team="C", away_team="Y",
                 home_score=45, away_score=15),    # C +30
            dict(game_id="G0d", gameday="2019-08-25", home_team="D", away_team="W",
                 home_score=42, away_score=3),     # D +39
            # A beats weak B, then loses to strong C, then beats strong D
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=20, away_score=10),    # A +10
            dict(game_id="G2", gameday="2019-09-08", home_team="C", away_team="A",
                 home_score=24, away_score=10),    # A -14
            dict(game_id="G3", gameday="2019-09-15", home_team="A", away_team="D",
                 home_score=21, away_score=14),    # A +7
        ])
        ladder = team_stats_ladder(team_events(g))
        a = ladder[ladder["team"] == "A"].set_index("game_id")
        self.assertIn("opp_adj_form", a.columns)
        # G2: A.form = mean([+10]) = 10; prior opponents' form = [B@G1 = 10]
        self.assertAlmostEqual(a.loc["G2", "opp_adj_form"], 0.0, places=6)
        # G3: A.form = mean([+10, -14]) = -2; prior opponents = [10 (B), 30 (C)]
        self.assertAlmostEqual(a.loc["G3", "opp_adj_form"], -22.0, places=6)
        self.assertLess(a.loc["G3", "opp_adj_form"], a.loc["G2", "opp_adj_form"])

    def test_pace_and_qb_epa_use_pbp_columns(self):
        pbp = pd.concat([
            self._pbp("G1", [("A", 5, 0.1, 0.2, 3000), ("A", 3, 0.2, 0.3, 2800),
                              ("B", -2, -0.3, None, 2600)]),
            self._pbp("G2", [("A", 7, 0.4, 0.5, 1500), ("B", 2, 0.1, 0.2, 1200)]),
        ])
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=20, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="B", away_team="A",
                 home_score=7, away_score=14),
        ])
        from nfl_features import _pbp_team_agg
        ladder = team_stats_ladder(team_events(g), team_game_agg=_pbp_team_agg(pbp))
        # A at G2 has prior pace (G1) and prior QB EPA (G1): both non-null
        a2 = ladder[(ladder["team"] == "A") & (ladder["game_id"] == "G2")].iloc[0]
        self.assertTrue(pd.notna(a2["pace_plays_min"]))
        self.assertTrue(pd.notna(a2["ewm_qb_epa"]))
        self.assertTrue(pd.notna(a2["ewm_epa"]))
        # G1 rows have no prior -> NaN
        a1 = ladder[(ladder["team"] == "A") & (ladder["game_id"] == "G1")].iloc[0]
        self.assertTrue(pd.isna(a1["ewm_qb_epa"]))


class TestSlateFeatures(unittest.TestCase):
    def test_scheduled_game_uses_only_prior_decided(self):
        """A 2026 scheduled game's trailing features must come from strictly
        prior decided games, and the games[] fields (records, venue, lines)
        must be present."""
        rows = [
            dict(game_id="2019_01_A_B", season=2019, week=1, gameday="2019-09-08",
                 home_team="A", away_team="B", home_score=24, away_score=10,
                 roof="outdoors", temp=70.0, wind=8.0, div_game=0,
                 stadium="S1", gametime="13:00", spread_line=1.5, total_line=44.0),
            dict(game_id="2019_01_B_A", season=2019, week=2, gameday="2019-09-15",
                 home_team="B", away_team="A", home_score=7, away_score=17,
                 roof="outdoors", temp=72.0, wind=6.0, div_game=0,
                 stadium="S2", gametime="13:00", spread_line=2.5, total_line=43.5),
            dict(game_id="2026_01_A_C", season=2026, week=1, gameday="2026-09-09",
                 home_team="A", away_team="C", home_score=None, away_score=None,
                 roof="dome", temp=74.0, wind=4.0, div_game=1,
                 stadium="Dome", gametime="20:20", spread_line=3.5, total_line=45.5),
        ]
        sched = pd.DataFrame(rows)
        decided = sched[sched["home_score"].notna()].copy()
        slate = build_slate_features(sched, None, decided, 2026)
        self.assertEqual(len(slate), 1)
        row = slate.iloc[0]
        self.assertEqual(row["game_id"], "2026_01_A_C")
        # A won both prior games -> 2-0; C never played -> record blank
        self.assertEqual(row["home_record"], "2-0")
        self.assertEqual(row["away_record"], "")
        self.assertEqual(row["stadium"], "Dome")
        self.assertEqual(row["spread_line"], 3.5)
        self.assertEqual(row["total_line"], 45.5)
        # elo_diff strictly from the two prior decided games (A's rating - 1500)
        self.assertTrue(pd.notna(row["elo_diff"]))
        self.assertTrue(pd.isna(row["form_diff_pts"]))  # C never played
        self.assertEqual(row["is_dome_home"], 1.0)


class TestV2Gate(unittest.TestCase):
    def test_v2_candidate_below_floor_reported_not_dropped(self):
        """Default policy: a below-floor candidate is reported
        (below_coverage_floor) and stays in the served pool; the legacy
        opt-in prune still drops it with a reason."""
        frame = synth_games([dict(game_id=f"G{i}", gameday=f"2019-09-{1+2*i:02d}",
                                  home_team="H", away_team="A",
                                  home_score=24, away_score=10) for i in range(20)])
        frame["season"] = 2019
        for f in FEATURE_COLUMNS:
            frame[f] = 0.0
        frame["elo_diff"] = np.linspace(-2, 2, len(frame))
        frame["ewm_qb_epa_play_diff"] = frame["ewm_qb_epa_play_diff"].mask(
            np.array([i % 2 == 0 for i in range(len(frame))]))
        res = run_feature_gate(frame)
        self.assertEqual(res["dropped"], [])
        self.assertIn("ewm_qb_epa_play_diff", res["below_coverage_floor"])
        self.assertIn("ewm_qb_epa_play_diff", res["v1_features"])
        res_legacy = run_feature_gate(frame, auto_prune=True)
        self.assertIn("ewm_qb_epa_play_diff", res_legacy["dropped"])
        self.assertIn("coverage", res_legacy["reasons"]["ewm_qb_epa_play_diff"])
        self.assertIn("elo_diff", res["v1_features"])

    def test_v2_admitted_features_and_sync(self):
        """FEATURE_COLUMNS is the SERVED pool: the gated 14 + the is_home
        anchor. The legacy v2 twins that were pruned at admission time
        (never by an ablation) are composed-but-unregistered — they are no
        longer FEATURE_COLUMNS members."""
        kept = ("ewm_net_pts_diff", "ewm_qb_epa_play_diff", "ewm_ypp_diff",
                "pace_plays_min_diff", "rest_short_diff", "div_game")
        for f in kept:
            self.assertIn(f, FEATURE_COLUMNS)
        unregistered = ("form_diff_pts", "ypp_diff", "ewm_epa_play_diff",
                        "ewm_scoring_diff", "opp_adj_net_pts_diff",
                        "temp_f", "wind_mph")
        for f in unregistered:
            self.assertNotIn(f, FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 15)   # 14 served + is_home anchor
        self.assertEqual(FEATURE_COLUMNS[-1], "is_home")


class TestBuildFeatures(unittest.TestCase):
    def test_compose_end_to_end(self):
        decided = synth_games([
            dict(game_id="2019_01_KC_JAX", gameday="2019-09-08", home_team="JAX",
                 away_team="KC", home_score=26, away_score=40, roof="outdoors"),
            dict(game_id="2019_01_GB_CHI", gameday="2019-09-05", home_team="CHI",
                 away_team="GB", home_score=3, away_score=10, roof="closed"),
        ])
        decided = decided.drop(columns=["roof"])
        schedule = pd.DataFrame({
            "game_id": ["2019_01_KC_JAX", "2019_01_GB_CHI"],
            "season": [2019, 2019], "week": [1, 1], "gameday": ["2019-09-08", "2019-09-05"],
            "home_team": ["JAX", "CHI"], "away_team": ["KC", "GB"],
            "home_score": [26, 3], "away_score": [40, 10], "roof": ["outdoors", "closed"],
        })
        pbp = pd.DataFrame({
            "game_id": ["2019_01_KC_JAX", "2019_01_KC_JAX", "2019_01_KC_JAX"],
            "posteam": ["KC", "KC", "JAX"], "yards_gained": [10, 6, -2],
        })
        feats = build_features(decided, schedule, pbp)
        self.assertEqual(list(feats["is_dome_home"]), [0.0, 1.0])
        jax = feats[feats["game_id"] == "2019_01_KC_JAX"].iloc[0]
        chi = feats[feats["game_id"] == "2019_01_GB_CHI"].iloc[0]
        self.assertEqual(jax["is_home"], 1.0)         # anchor
        # both are each team's only game -> trailing priors are NaN -> diff NaN
        self.assertTrue(pd.isna(jax["form_diff_pts"]))
        # ELO entering = the 1500 prior for this tiny 1-game-per-team timeline
        self.assertAlmostEqual(jax["elo_diff"], 0.0, places=6)

        for col in ("elo_diff", "form_diff_pts", "win_pct_diff",
                    "rest_days_diff", "ypp_diff", "is_dome_home", "is_home"):
            self.assertIn(col, feats.columns)


class TestTier1Trailing(unittest.TestCase):
    """Tier-1 (v3) PBP aggregates: strictly-prior decaying windows, turnover
    net, and the 9 diff candidates composed end-to-end."""

    def _pbp(self):
        return pd.DataFrame([
            dict(game_id="G1", posteam="A", defteam="B", yards_gained=5,
                 epa=0.2, interception=0, fumble_lost=0),
            dict(game_id="G1", posteam="A", defteam="B", yards_gained=0,
                 epa=-0.5, interception=1, fumble_lost=0),
            dict(game_id="G1", posteam="B", defteam="A", yards_gained=-1,
                 epa=-0.3, interception=0, fumble_lost=1),
            dict(game_id="G2", posteam="A", defteam="C", yards_gained=3,
                 epa=0.1, interception=0, fumble_lost=0),
        ])

    def test_tier1_prior_only_and_net_turnovers(self):
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=20, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="A", away_team="C",
                 home_score=14, away_score=7),
        ])
        from nfl_features import _pbp_team_agg
        base = team_stats_ladder(team_events(g), team_game_agg=_pbp_team_agg(self._pbp()))
        # A future G3 with a turnover explosion must not change G1/G2 rows.
        g3 = synth_games([
            dict(game_id="G3", gameday="2019-09-15", home_team="A", away_team="D",
                 home_score=35, away_score=0),
        ])
        pbp3 = pd.DataFrame([
            dict(game_id="G3", posteam="A", defteam="D", yards_gained=0,
                 epa=-0.9, interception=2, fumble_lost=1),
            dict(game_id="G3", posteam="D", defteam="A", yards_gained=0,
                 epa=0.0, interception=0, fumble_lost=0),
        ])
        alt = team_stats_ladder(
            team_events(pd.concat([g, g3], ignore_index=True)),
            team_game_agg=_pbp_team_agg(
                pd.concat([self._pbp(), pbp3], ignore_index=True)))
        a_base = base[base["team"] == "A"].set_index("game_id")
        a_alt = alt[alt["team"] == "A"].set_index("game_id")
        for stat in ("ewm_giveaways", "ewm_takeaways", "ewm_net_turnovers"):
            for gid in ("G1", "G2"):
                v_base, v_alt = a_base.loc[gid, stat], a_alt.loc[gid, stat]
                if pd.isna(v_base) and pd.isna(v_alt):
                    continue
                self.assertEqual(v_base, v_alt,
                                 f"{stat} of {gid} changed by a future game")
        # G1: A committed an INT (giveaways=1); B lost a fumble (A takeaways=1),
        # so G2's trailing net turnovers = 0. G1 itself has no prior -> NaN.
        self.assertTrue(pd.isna(a_base.loc["G1", "ewm_giveaways"]))
        self.assertEqual(a_base.loc["G2", "ewm_giveaways"], 1.0)
        self.assertEqual(a_base.loc["G2", "ewm_takeaways"], 1.0)
        self.assertEqual(a_base.loc["G2", "ewm_net_turnovers"], 0.0)

    def test_tier1_diff_columns_end_to_end(self):
        g = synth_games([
            dict(game_id="G1", gameday="2019-09-01", home_team="A", away_team="B",
                 home_score=20, away_score=10),
            dict(game_id="G2", gameday="2019-09-08", home_team="C", away_team="D",
                 home_score=7, away_score=14),
            dict(game_id="G3", gameday="2019-09-15", home_team="A", away_team="C",
                 home_score=24, away_score=21),
        ])
        pbp = pd.DataFrame([
            dict(game_id="G1", posteam="A", defteam="B", yards_gained=5, epa=0.2,
                 interception=0, fumble_lost=0),
            dict(game_id="G1", posteam="A", defteam="B", yards_gained=0, epa=-0.5,
                 interception=1, fumble_lost=0),
            dict(game_id="G1", posteam="B", defteam="A", yards_gained=-1, epa=-0.3,
                 interception=0, fumble_lost=0),
            dict(game_id="G2", posteam="C", defteam="D", yards_gained=2, epa=0.1,
                 interception=0, fumble_lost=0),
            dict(game_id="G2", posteam="D", defteam="C", yards_gained=0, epa=-0.2,
                 interception=0, fumble_lost=1),
            dict(game_id="G3", posteam="A", defteam="C", yards_gained=1, epa=0.0,
                 interception=0, fumble_lost=0),
            dict(game_id="G3", posteam="C", defteam="A", yards_gained=1, epa=0.0,
                 interception=0, fumble_lost=0),
        ])
        feats = build_features(g, g, pbp)
        for col in ("turnover_diff", "any_a_diff", "sack_rate_diff",
                    "success_rate_diff", "explosive_rate_diff", "penalty_diff",
                    "third_down_rate_diff", "redzone_td_rate_diff",
                    "pts_per_drive_diff"):
            self.assertIn(col, feats.columns)
        g3 = feats[feats["game_id"] == "G3"].iloc[0]
        # A (home) prior: net turnovers -1 (threw an INT), success 0.5;
        # C (away) prior: net +1 (forced a fumble), success 1.0.
        self.assertAlmostEqual(g3["turnover_diff"], -2.0, places=6)
        self.assertAlmostEqual(g3["success_rate_diff"], -0.5, places=6)


if __name__ == "__main__":
    unittest.main()