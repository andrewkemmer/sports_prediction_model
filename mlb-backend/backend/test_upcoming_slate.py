"""
Regression tests for build_upcoming_slate.

Statcast-derived history ends at the last PLAYED game, so a normal morning
run has zero rows for today. The old fallback recycled yesterday's completed
games into todays_games_<today>.csv. These tests lock in the replacement:
today's real schedule with each team/pitcher's latest point-in-time state.
"""
import unittest
from datetime import date, datetime

import numpy as np
import pandas as pd

from backend.data_ingestion import (
    _latest_pitcher_state,
    _norm_player_name,
    build_upcoming_slate,
)
from backend.features import add_diff_features
from backend.training import FEATURE_COLS


TARGET = date(2026, 8, 22)


def _history() -> pd.DataFrame:
    """Two teams with decided games; SEA's last woba is NaN to prove the
    carry-forward resolves to the most recent OBSERVED value."""
    return pd.DataFrame([
        {   # NYY wins at home vs SEA (Aug 19): NYY 1-0, SEA 0-1
            "game_id": "20260819_SEA@NYY", "game_date": "2026-08-19",
            "start_time_utc": datetime(2026, 8, 19, 23, 0),
            "home_team": "NYY", "away_team": "SEA",
            "home_win": 1.0, "home_score": 5, "away_score": 3,
            "home_starter_id": 101.0, "away_starter_id": 202.0,
            "woba_30g_home": 0.330, "woba_30g_away": 0.310,
            # Per-hand lineup OPS splits (team state): NYY batting at home
            "lineup_ops_vs_l_home": 0.700, "lineup_ops_vs_r_home": 0.780,
            # SEA batting on the road
            "lineup_ops_vs_l_away": 0.660, "lineup_ops_vs_r_away": 0.740,
            "closer_available_home": 1.0, "closer_available_away": 1.0,
            "sp_era_5g_home": 3.10, "sp_era_home": 3.10, "sp_k9_5g_home": 9.9,
        },
        {   # NYY wins again at SEA (Aug 20): NYY 2-0, SEA 0-2
            "game_id": "20260820_NYY@SEA", "game_date": "2026-08-20",
            "start_time_utc": datetime(2026, 8, 20, 23, 0),
            "home_team": "SEA", "away_team": "NYY",
            "home_win": 0.0, "home_score": 2, "away_score": 7,
            "home_starter_id": 202.0, "away_starter_id": 101.0,
            "woba_30g_home": np.nan, "woba_30g_away": 0.340,
            # SEA batting at home: newest L-split observation (R is missing —
            # proves the carry-forward resolves to the last OBSERVED value)
            "lineup_ops_vs_l_home": 0.680, "lineup_ops_vs_r_home": np.nan,
            "closer_available_away": 0.0,  # NYY closer unavailable Aug 20
            # Cole started as the AWAY pitcher here
            "sp_era_5g_away": 4.50,
        },
        {   # Postponed/tied (Aug 21): must not affect records or Elo updates
            "game_id": "20260821_BOS@NYY", "game_date": "2026-08-21",
            "start_time_utc": datetime(2026, 8, 21, 23, 0),
            "home_team": "NYY", "away_team": "BOS",
            "home_win": np.nan, "home_score": np.nan, "away_score": np.nan,
        },
    ])


def _schedule() -> pd.DataFrame:
    """Today's slate: one undecided game + one ESPN-final + one unknown pitcher."""
    return pd.DataFrame([
        {
            "game_id": "20260822_BOS@NYY", "game_date": TARGET,
            "start_time_utc": pd.Timestamp("2026-08-22 23:05:00"),
            "home_team": "NYY", "away_team": "BOS",
            "home_win": None, "home_score": None, "away_score": None,
            "sp_name_home": "Gerrit Cole", "sp_name_away": "Some Rookie",
            "venue": "Yankee Stadium",
        },
        {
            "game_id": "20260822_SEA@BOS", "game_date": TARGET,
            "start_time_utc": pd.Timestamp("2026-08-22 17:05:00"),
            "home_team": "BOS", "away_team": "SEA",
            "home_win": 1.0, "home_score": 4, "away_score": 1,  # already final
            "sp_name_home": "Zack Wheeler", "sp_name_away": "Luis Castillo",
            "venue": "Fenway Park",
        },
    ])


_PBP = pd.DataFrame([
    {"player_name": "Cole, Gerrit", "pitcher": 101.0},
    {"player_name": "Wheeler, Zack", "pitcher": 303.0},
])


class TestNormPlayerName(unittest.TestCase):
    def test_statcast_and_espn_names_collide(self):
        self.assertEqual(_norm_player_name("Wheeler, Zack"), "zack wheeler")
        self.assertEqual(_norm_player_name("Zack Wheeler"), "zack wheeler")


class TestBuildUpcomingSlate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slate = build_upcoming_slate(
            _history(), TARGET, pbp_df=_PBP, schedule_df=_schedule(),
        )
        cls.by_id = {r["game_id"]: r for _, r in cls.slate.iterrows()}

    def test_rows_are_todays_schedule_not_yesterday(self):
        self.assertEqual(len(self.slate), 2)
        self.assertTrue((pd.to_datetime(self.slate["game_date"]).dt.date == TARGET).all())
        ids = set(self.slate["game_id"])
        self.assertIn("20260822_BOS@NYY", ids)
        self.assertNotIn("20260821_BOS@NYY", ids)  # yesterday must NOT be recycled

    def test_all_feature_columns_present(self):
        """predict_games slices [c for c in FEATURE_COLS if c in columns] — a
        missing column would silently change the model's input shape.
        Diff features are computed by add_diff_features() in the pipeline
        after build_upcoming_slate(), so we apply it here too."""
        slate_with_diffs = add_diff_features(self.slate)
        missing = [c for c in FEATURE_COLS if c not in slate_with_diffs.columns]
        self.assertEqual(missing, [])

    def test_undecided_games_ship_null_labels(self):
        g = self.by_id["20260822_BOS@NYY"]
        self.assertTrue(pd.isna(g["home_win"]))
        self.assertTrue(pd.isna(g["home_score"]))
        # ESPN-final row keeps its real result so the card can grade it
        f = self.by_id["20260822_SEA@BOS"]
        self.assertEqual(float(f["home_win"]), 1.0)
        self.assertEqual(int(f["home_score"]), 4)

    def test_records_and_run_diff_carry_forward(self):
        g = self.by_id["20260822_BOS@NYY"]
        self.assertEqual(g["home_record"], "2-0")      # NYY after 2 wins
        self.assertEqual(g["away_record"], "0-0")      # BOS: only a postponement
        self.assertEqual(int(g["home_run_diff"]), 7)   # (5+7)-(3+2)
        self.assertAlmostEqual(float(g["home_win_pct"]), 1.0)

    def test_rolling_features_use_last_observed_value(self):
        g = self.by_id["20260822_BOS@NYY"]
        self.assertAlmostEqual(float(g["woba_30g_home"]), 0.340)  # NYY latest
        s = self.by_id["20260822_SEA@BOS"]
        # SEA's Aug-20 woba is NaN -> carries Aug-19 value, not 0 and not NaN
        self.assertAlmostEqual(float(s["woba_30g_away"]), 0.310)

    def test_lineup_hand_splits_carry_forward(self):
        """Per-hand lineup OPS splits are TEAM state (not opponent-indexed):
        each team's latest observed L/R value re-suffixes onto tonight's slot."""
        g = self.by_id["20260822_BOS@NYY"]          # NYY bats at home
        self.assertAlmostEqual(float(g["lineup_ops_vs_l_home"]), 0.700)
        self.assertAlmostEqual(float(g["lineup_ops_vs_r_home"]), 0.780)
        s = self.by_id["20260822_SEA@BOS"]          # SEA on the road
        # SEA's latest observed L-split (Aug 20) overrides Aug 19
        self.assertAlmostEqual(float(s["lineup_ops_vs_l_away"]), 0.680)
        self.assertAlmostEqual(float(s["lineup_ops_vs_r_away"]), 0.740)

    def test_travel_and_closer_state_carry_forward(self):
        """Travel fatigue recomputes from strictly-prior schedule; closer
        availability carries forward as team state."""
        g = self.by_id["20260822_BOS@NYY"]
        # NYY: Bronx (ET, Aug 19) -> Seattle (PT, Aug 20) -> Bronx (Aug 21):
        # both the trip out and the trip back count → 2 crossings
        self.assertEqual(int(g["time_zones_crossed_last_3d_home"]), 2)
        # NYY's latest closer observation (Aug 20 road game): unavailable
        self.assertEqual(int(g["closer_available_home"]), 0)
        s = self.by_id["20260822_SEA@BOS"]
        # SEA returned Seattle (PT, Aug 20) from Bronx (ET, Aug 19): 1
        self.assertEqual(int(s["time_zones_crossed_last_3d_away"]), 1)
        # SEA's latest closer observation (Aug 19 road game): available
        self.assertEqual(int(s["closer_available_away"]), 1)

    def test_pitcher_stats_via_statsapi_id_without_pbp(self):
        """Slate pitcher ERA/K9 must resolve from the StatsAPI person id
        alone (no pbp_df/name matching). load_espn_schedule now stamps
        sp_id_* on every row, so an evening/next-day run still gets real
        starter stats even though ESPN dropped probablePitcher."""
        sched = _schedule()
        sched["sp_id_home"] = [101.0, 303.0]  # Cole(id 101) / Wheeler(303, no stats)
        sched["sp_id_away"] = [202.0, 404.0]  # no observed stats either
        slate = build_upcoming_slate(_history(), TARGET, pbp_df=None, schedule_df=sched)
        by_id = {r["game_id"]: r for _, r in slate.iterrows()}

        g = by_id["20260822_BOS@NYY"]
        # Cole (101) via id -> his latest observed stats (same as name path)
        self.assertAlmostEqual(float(g["sp_era_5g_home"]), 4.50)
        self.assertAlmostEqual(float(g["sp_era_home"]), 3.10)
        self.assertAlmostEqual(float(g["sp_k9_5g_home"]), 9.9)
        # id 202 has NO observed sp stats -> NULL, never a fabricated 0
        self.assertTrue(pd.isna(g["sp_era_5g_away"]))
        self.assertTrue(pd.isna(g["sp_k9_5g_away"]))

        f = by_id["20260822_SEA@BOS"]
        # Wheeler (303) has no observed stats either -> NULLs
        self.assertTrue(pd.isna(f["sp_era_home"]))
        self.assertTrue(pd.isna(f["sp_k9_5g_home"]))

    def test_pitcher_features_via_name_mapping(self):
        g = self.by_id["20260822_BOS@NYY"]
        # Cole (id 101) is tonight's HOME starter. His latest start was Aug 20
        # (as a road starter) -> the newest observed value re-suffixes onto
        # this game's home slot regardless of which side it came from.
        self.assertAlmostEqual(float(g["sp_era_5g_home"]), 4.50)
        # sp_era (non-30g twin) was last observed on Aug 19
        self.assertAlmostEqual(float(g["sp_era_home"]), 3.10)
        self.assertAlmostEqual(float(g["sp_k9_5g_home"]), 9.9)
        # Unknown away probable -> NULLs, never fabricated zeros and never
        # Cole's values leaking into his slot.
        self.assertTrue(pd.isna(g["sp_era_5g_away"]))
        self.assertTrue(pd.isna(g["sp_k9_5g_away"]))

    def test_rest_days_since_last_appearance(self):
        g = self.by_id["20260822_BOS@NYY"]
        # Both teams' most recent calendar appearance is Aug 21 (postponement)
        self.assertEqual(int(g["rest_days_home"]), 1)
        self.assertEqual(int(g["rest_days_away"]), 1)

    def test_elos_present_and_positive_update(self):
        g = self.by_id["20260822_BOS@NYY"]
        self.assertFalse(pd.isna(g["home_elo"]))
        self.assertGreater(float(g["home_elo"]), 1500.0)  # two wins lift Elo

    def test_empty_schedule_returns_empty_frame(self):
        out = build_upcoming_slate(_history(), TARGET, schedule_df=pd.DataFrame())
        self.assertTrue(out.empty)


class TestLatestPitcherStateChronological(unittest.TestCase):
    """Regression for the _latest_pitcher_state overwrite bug.

    The OLD implementation walked home starts then away starts with
    last-write-wins ACROSS passes, so an OLDER away start always overwrote a
    NEWER home start. On the 2026-08-31 slate Elmer Rodríguez's May 17 AWAY
    line (3.24 ERA / 5.4 K/9) clobbered his Aug 26 HOME line
    (3.9375 / 5.625) on the NYY@LAA card. The fix resolves the
    chronologically LATEST non-null value per pitcher per feature base across
    BOTH columns (lag semantics unchanged — each value is still the stat
    through the pitcher's previous start).
    """

    @staticmethod
    def _hist(rows) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_fresh_home_start_beats_older_away_start(self):
        # Elmer Rodríguez (id 695684): May 17 AWAY start (stale) vs Aug 26
        # HOME start (fresh) — the fresh HOME line must win.
        hist = self._hist([
            {"game_id": "20260517_NYM@NYY", "game_date": "2026-05-17",
             "start_time_utc": datetime(2026, 5, 17, 23, 5),
             "home_team": "NYM", "away_team": "NYY", "home_win": 0.0,
             "away_starter_id": 695684.0,
             "sp_era_away": 3.24, "sp_k9_away": 5.4},
            {"game_id": "20260826_NYY@HOU", "game_date": "2026-08-26",
             "start_time_utc": datetime(2026, 8, 26, 23, 5),
             "home_team": "NYY", "away_team": "HOU", "home_win": 1.0,
             "home_starter_id": 695684.0,
             "sp_era_home": 3.9375, "sp_k9_home": 5.625},
        ])
        st = _latest_pitcher_state(hist)
        self.assertAlmostEqual(st[695684]["sp_era"], 3.9375)
        self.assertAlmostEqual(st[695684]["sp_k9"], 5.625)

    def test_fresh_away_start_beats_older_home_start(self):
        # Ureña-style mirror: the fresher AWAY start must beat an older HOME
        # start — chronology rules, not side preference.
        hist = self._hist([
            {"game_id": "20260701_LAA@SEA", "game_date": "2026-07-01",
             "start_time_utc": datetime(2026, 7, 1, 19, 5),
             "home_team": "SEA", "away_team": "LAA", "home_win": 1.0,
             "home_starter_id": 777.0,
             "sp_era_home": 2.89, "sp_k9_home": 8.94},
            {"game_id": "20260815_TEX@LAA", "game_date": "2026-08-15",
             "start_time_utc": datetime(2026, 8, 15, 23, 5),
             "home_team": "LAA", "away_team": "TEX", "home_win": 0.0,
             "away_starter_id": 777.0,
             "sp_era_away": 3.55, "sp_k9_away": 9.2},
        ])
        st = _latest_pitcher_state(hist)
        self.assertAlmostEqual(st[777]["sp_era"], 3.55)
        self.assertAlmostEqual(st[777]["sp_k9"], 9.2)

    def test_doubleheader_later_start_wins_regardless_of_row_order(self):
        """Same-day doubleheader legs: the LATER start must win, and the
        result must not depend on the input row order."""
        leg_early = {"game_id": "20260812_KC@CLE", "game_date": "2026-08-12",
                     "start_time_utc": datetime(2026, 8, 12, 13, 5),
                     "home_team": "CLE", "away_team": "KC", "home_win": 1.0,
                     "home_starter_id": 888.0,
                     "sp_era_home": 3.10, "sp_k9_home": 9.0}
        leg_late = {"game_id": "20260812_KC@CLE_2", "game_date": "2026-08-12",
                    "start_time_utc": datetime(2026, 8, 12, 19, 5),
                    "home_team": "CLE", "away_team": "KC", "home_win": 0.0,
                    "home_starter_id": 888.0,
                    "sp_era_home": 3.40, "sp_k9_home": 9.5}
        for rows in ([leg_early, leg_late], [leg_late, leg_early]):
            st = _latest_pitcher_state(self._hist(rows))
            self.assertAlmostEqual(
                st[888]["sp_era"], 3.40,
                msg="later-start leg must win regardless of row order")
            self.assertAlmostEqual(st[888]["sp_k9"], 9.5)

    def test_slate_uses_fresh_home_start_for_away_slot(self):
        """End-to-end regression for the shipped bug: Elmer Rodríguez on the
        2026-08-31 NYY@LAA slate (away slot) must carry his fresh Aug 26 HOME
        line (3.9375 / 5.625), not his older May 17 AWAY line (3.24 / 5.4)."""
        hist = self._hist([
            {"game_id": "20260517_NYM@NYY", "game_date": "2026-05-17",
             "start_time_utc": datetime(2026, 5, 17, 23, 5),
             "home_team": "NYM", "away_team": "NYY", "home_win": 0.0,
             "away_starter_id": 695684.0,
             "sp_era_away": 3.24, "sp_k9_away": 5.4},
            {"game_id": "20260826_NYY@HOU", "game_date": "2026-08-26",
             "start_time_utc": datetime(2026, 8, 26, 23, 5),
             "home_team": "NYY", "away_team": "HOU", "home_win": 1.0,
             "home_starter_id": 695684.0,
             "sp_era_home": 3.9375, "sp_k9_home": 5.625},
        ])
        sched = pd.DataFrame([{
            "game_id": "20260831_NYY@LAA", "game_date": date(2026, 8, 31),
            "start_time_utc": pd.Timestamp("2026-08-31 23:38:00"),
            "home_team": "LAA", "away_team": "NYY",
            "home_win": None, "home_score": None, "away_score": None,
            "sp_name_home": "Walbert Ureña", "sp_name_away": "Elmer Rodríguez",
            "sp_id_home": 999.0, "sp_id_away": 695684.0,
            "venue": "Angel Stadium",
        }])
        slate = build_upcoming_slate(hist, date(2026, 8, 31), pbp_df=None,
                                     schedule_df=sched)
        self.assertEqual(len(slate), 1)
        g = slate.iloc[0]
        self.assertAlmostEqual(float(g["sp_era_away"]), 3.9375)
        self.assertAlmostEqual(float(g["sp_k9_away"]), 5.625)


if __name__ == "__main__":
    unittest.main()
