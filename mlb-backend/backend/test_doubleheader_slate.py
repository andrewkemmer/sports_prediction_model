"""
Regression tests: doubleheader legs must be DISTINCT slate games.

Aug 29 had real doubleheaders (BOS@NYY 1:05 + 7:15 PM ET; ARI@SF 4:05 +
10:05 PM ET). The app rendered two IDENTICAL BOS@NYY cards (same time, same
pitcher, same SHAP key) because every slate key was matchup-based:

  - game_id = YYYYMMDD_AWAY@HOME -> both legs collide;
  - _fetch_statsapi_pitchers keyed (home, away) -> last leg's starter won,
    so BOTH legs showed Max Fried;
  - _fetch_slate_lineups keyed (home, away) -> both legs fetched the first
    game's lineup feed;
  - _carry_forward_slate_details drop_duplicates("game_id") -> collapsed the
    legs and restored both from the first leg's row;
  - _attach_slate_run_margins merged on game_pk := game_id -> a many-to-many
    merge exploded 17 slate rows into 21 board rows.

These tests lock in the per-game fix: distinct per-leg game_ids (deterministic
start-time ordinal suffix), per-leg pitchers by first-pitch time, per-leg
lineup game_pk resolution, per-leg carry-forward, and a merge that can never
explode rows (exact-duplicate game_pk rows are deduped as a true bug).
"""
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from backend.data_ingestion import (
    _disambiguate_slate_keys,
    _fetch_statsapi_pitchers,
    _match_statsapi_pitcher_leg,
    load_espn_schedule,
)
from backend import pipeline

TARGET = date(2026, 8, 29)

# ---------------------------------------------------------------------------
# load_espn_schedule: distinct keys + per-leg pitchers
# ---------------------------------------------------------------------------


def _make_event(date_str: str, home_abbr: str, away_abbr: str,
                home_sp: str = "", away_sp: str = ""):
    """One ESPN scoreboard event dict shaped like _parse_espn_event expects."""
    return {
        "date": date_str,
        "competitions": [{
            "competitors": [
                {"homeAway": "away", "team": {"abbreviation": away_abbr},
                 "score": "0",
                 "probablePitcher": {"fullName": away_sp} if away_sp else {}},
                {"homeAway": "home", "team": {"abbreviation": home_abbr},
                 "score": "0",
                 "probablePitcher": {"fullName": home_sp} if home_sp else {}},
            ],
            "status": {"type": {"state": "pre", "detail": "Scheduled"}},
            "venue": {"fullName": "Yankee Stadium"},
        }],
    }


class TestLoadEspnScheduleDoubleheader(unittest.TestCase):
    @mock.patch("backend.data_ingestion._fetch_statsapi_pitchers")
    @mock.patch("backend.data_ingestion._fetch_espn_scoreboard")
    def test_doubleheader_legs_get_distinct_keys_and_own_pitchers(
            self, mock_scoreboard, mock_pitchers):
        """BOS@NYY twice (13:05 + 19:15 ET) -> two rows with DISTINCT
        game_ids, each matched to ITS OWN StatsAPI leg's starter by first
        pitch — never the last leg's pitcher on both cards."""
        mock_scoreboard.return_value = [
            _make_event("2026-08-29T17:05:00Z", "NYY", "BOS",
                        home_sp="Nestor Cortes", away_sp="Tanner Houck"),
            _make_event("2026-08-29T23:15:00Z", "NYY", "BOS",
                        home_sp="Max Fried", away_sp="Brayan Bello"),
            _make_event("2026-08-29T17:10:00Z", "DET", "LAD",
                        home_sp="Keider Montero", away_sp="Blake Snell"),
        ]
        mock_pitchers.return_value = {
            ("NYY", "BOS"): [
                {"home_name": "Nestor Cortes", "away_name": "Tanner Houck",
                 "home_id": 1, "away_id": 2, "game_pk": 900111,
                 "game_date_utc": "2026-08-29T17:05:00Z"},
                {"home_name": "Max Fried", "away_name": "Brayan Bello",
                 "home_id": 3, "away_id": 4, "game_pk": 900112,
                 "game_date_utc": "2026-08-29T23:15:00Z"},
            ],
            ("DET", "LAD"): [
                {"home_name": "Keider Montero", "away_name": "Blake Snell",
                 "home_id": 5, "away_id": 6, "game_pk": 900113,
                 "game_date_utc": "2026-08-29T17:10:00Z"},
            ],
        }

        sched = load_espn_schedule(TARGET)
        self.assertEqual(len(sched), 3)
        by_id = {r["game_id"]: r for _, r in sched.iterrows()}
        leg1 = by_id["20260829_BOS@NYY"]
        leg2 = by_id["20260829_BOS@NYY_2"]
        # Per-leg pitchers: the 13:05 leg gets Cortes, the 19:15 leg gets
        # Fried — not the same starter on both cards.
        self.assertEqual(leg1["sp_name_home"], "Nestor Cortes")
        self.assertEqual(leg1["sp_id_home"], 1)
        self.assertEqual(leg2["sp_name_home"], "Max Fried")
        self.assertEqual(leg2["sp_id_home"], 3)
        self.assertEqual(leg2["sp_name_away"], "Brayan Bello")
        self.assertEqual(leg2["sp_id_away"], 4)
        # A single-game matchup keeps its legacy bare id.
        self.assertIn("20260829_LAD@DET", by_id)

    @mock.patch("backend.data_ingestion._fetch_statsapi_pitchers")
    @mock.patch("backend.data_ingestion._fetch_espn_scoreboard")
    def test_exact_duplicate_event_dropped(self, mock_scoreboard, mock_pitchers):
        """The same game listed twice (identical start time) is a true
        upstream bug — one row, not two identical cards."""
        mock_scoreboard.return_value = [
            _make_event("2026-08-29T17:05:00Z", "NYY", "BOS",
                        home_sp="Nestor Cortes", away_sp="Tanner Houck"),
            _make_event("2026-08-29T17:05:00Z", "NYY", "BOS",
                        home_sp="Nestor Cortes", away_sp="Tanner Houck"),
        ]
        mock_pitchers.return_value = {
            ("NYY", "BOS"): [
                {"home_name": "Nestor Cortes", "away_name": "Tanner Houck",
                 "home_id": 1, "away_id": 2, "game_pk": 900111,
                 "game_date_utc": "2026-08-29T17:05:00Z"},
            ],
        }
        sched = load_espn_schedule(TARGET)
        self.assertEqual(len(sched), 1)
        self.assertEqual(sched.iloc[0]["game_id"], "20260829_BOS@NYY")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestSlateKeyDisambiguation(unittest.TestCase):
    def test_single_games_untouched(self):
        df = pd.DataFrame([
            {"game_id": "20260829_LAD@DET", "home_team": "DET",
             "away_team": "LAD", "start_time_utc": pd.Timestamp("2026-08-29 13:10:00")},
        ])
        out = _disambiguate_slate_keys(df)
        self.assertEqual(out["game_id"].tolist(), ["20260829_LAD@DET"])

    def test_doubleheader_legs_distinct(self):
        df = pd.DataFrame([
            {"game_id": "20260829_BOS@NYY", "home_team": "NYY",
             "away_team": "BOS", "start_time_utc": pd.Timestamp("2026-08-29 19:15:00")},
            {"game_id": "20260829_BOS@NYY", "home_team": "NYY",
             "away_team": "BOS", "start_time_utc": pd.Timestamp("2026-08-29 13:05:00")},
        ])
        out = _disambiguate_slate_keys(df)
        ids = sorted(out["game_id"].tolist())
        self.assertEqual(ids, ["20260829_BOS@NYY", "20260829_BOS@NYY_2"])
        # The earlier leg keeps the base id; the later leg is the suffixed one.
        base_idx = out["game_id"].tolist().index("20260829_BOS@NYY")
        leg2_idx = out["game_id"].tolist().index("20260829_BOS@NYY_2")
        self.assertLess(base_idx, leg2_idx)

    def test_exact_duplicate_dropped_not_suffixed(self):
        df = pd.DataFrame([
            {"game_id": "20260829_BOS@NYY", "home_team": "NYY",
             "away_team": "BOS", "start_time_utc": pd.Timestamp("2026-08-29 13:05:00")},
            {"game_id": "20260829_BOS@NYY", "home_team": "NYY",
             "away_team": "BOS", "start_time_utc": pd.Timestamp("2026-08-29 13:05:00")},
        ])
        out = _disambiguate_slate_keys(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["game_id"], "20260829_BOS@NYY")


class TestMatchStatsapiPitcherLeg(unittest.TestCase):
    def setUp(self):
        self.legs = [
            {"home_name": "Nestor Cortes", "game_date_utc": "2026-08-29T17:05:00Z"},
            {"home_name": "Max Fried", "game_date_utc": "2026-08-29T23:15:00Z"},
        ]

    def test_each_leg_matches_its_own_first_pitch(self):
        m1 = _match_statsapi_pitcher_leg(
            self.legs, pd.Timestamp("2026-08-29 13:05:00").tz_localize("America/New_York"))
        m2 = _match_statsapi_pitcher_leg(
            self.legs, pd.Timestamp("2026-08-29 19:15:00").tz_localize("America/New_York"))
        self.assertEqual(m1["home_name"], "Nestor Cortes")
        self.assertEqual(m2["home_name"], "Max Fried")

    def test_far_away_time_returns_none(self):
        self.assertIsNone(_match_statsapi_pitcher_leg(
            self.legs, pd.Timestamp("2026-08-29 03:00:00").tz_localize("America/New_York")))

    def test_empty_legs_none(self):
        self.assertIsNone(_match_statsapi_pitcher_leg([], pd.Timestamp("2026-08-29 13:05:00")))


class TestNearestSlatePk(unittest.TestCase):
    def test_doubleheader_legs_resolve_own_pk(self):
        legs = [
            (pd.Timestamp("2026-08-29 13:05:00"), 900111),
            (pd.Timestamp("2026-08-29 19:15:00"), 900112),
        ]
        self.assertEqual(
            pipeline._nearest_slate_pk(legs, pd.Timestamp("2026-08-29 13:05:00")), 900111)
        self.assertEqual(
            pipeline._nearest_slate_pk(legs, pd.Timestamp("2026-08-29 19:15:00")), 900112)

    def test_unknown_or_far_matchup_none(self):
        self.assertIsNone(pipeline._nearest_slate_pk(None, pd.Timestamp("2026-08-29 13:05:00")))
        legs = [(pd.Timestamp("2026-08-29 13:05:00"), 900111)]
        self.assertIsNone(pipeline._nearest_slate_pk(
            legs, pd.Timestamp("2026-08-29 03:00:00")))


# ---------------------------------------------------------------------------
# build_upcoming_slate: distinct keys + per-leg pitcher stats
# ---------------------------------------------------------------------------


def _history() -> pd.DataFrame:
    """NYY starters 101 (3.10 ERA) and 303 (4.50 ERA) each with one start."""
    return pd.DataFrame([
        {"game_id": "20260827_TB@NYY", "game_date": "2026-08-27",
         "start_time_utc": datetime(2026, 8, 27, 23, 0),
         "home_team": "NYY", "away_team": "TB",
         "home_win": 1.0, "home_score": 3, "away_score": 1,
         "home_starter_id": 101.0, "away_starter_id": 501.0,
         "sp_era_home": 3.10, "sp_k9_home": 9.9,
         "sp_era_away": 4.00, "sp_k9_away": 7.0},
        {"game_id": "20260828_BOS@NYY", "game_date": "2026-08-28",
         "start_time_utc": datetime(2026, 8, 28, 23, 0),
         "home_team": "NYY", "away_team": "BOS",
         "home_win": 1.0, "home_score": 5, "away_score": 2,
         "home_starter_id": 303.0, "away_starter_id": 404.0,
         "sp_era_home": 4.50, "sp_k9_home": 8.1,
         "sp_era_away": 3.90, "sp_k9_away": 7.7},
    ])


def _dh_schedule() -> pd.DataFrame:
    """Two REAL BOS@NYY legs (13:05 + 19:15) sharing the matchup-based
    game_id, each carrying its own StatsAPI starter id."""
    return pd.DataFrame([
        {"game_id": "20260829_BOS@NYY", "game_date": TARGET,
         "start_time_utc": pd.Timestamp("2026-08-29 13:05:00"),
         "home_team": "NYY", "away_team": "BOS",
         "home_win": None, "home_score": None, "away_score": None,
         "sp_name_home": "TBD", "sp_name_away": "TBD",
         "sp_id_home": 101.0, "sp_id_away": 202.0,
         "venue": "Yankee Stadium", "game_state": "pre",
         "game_status_detail": "Scheduled"},
        {"game_id": "20260829_BOS@NYY", "game_date": TARGET,
         "start_time_utc": pd.Timestamp("2026-08-29 19:15:00"),
         "home_team": "NYY", "away_team": "BOS",
         "home_win": None, "home_score": None, "away_score": None,
         "sp_name_home": "TBD", "sp_name_away": "TBD",
         "sp_id_home": 303.0, "sp_id_away": 404.0,
         "venue": "Yankee Stadium", "game_state": "pre",
         "game_status_detail": "Scheduled"},
    ])


class TestBuildUpcomingSlateDoubleheader(unittest.TestCase):
    def test_two_distinct_legs_with_own_pitchers(self):
        from backend.data_ingestion import build_upcoming_slate
        slate = build_upcoming_slate(_history(), TARGET, pbp_df=None,
                                     schedule_df=_dh_schedule())
        self.assertEqual(len(slate), 2)  # BOTH legs survive — never one
        by_id = {r["game_id"]: r for _, r in slate.iterrows()}
        self.assertIn("20260829_BOS@NYY", by_id)
        self.assertIn("20260829_BOS@NYY_2", by_id)
        leg1, leg2 = by_id["20260829_BOS@NYY"], by_id["20260829_BOS@NYY_2"]
        # Each leg carries ITS OWN pitcher's rolling stats, by StatsAPI id.
        self.assertAlmostEqual(float(leg1["sp_era_home"]), 3.10)  # starter 101
        self.assertAlmostEqual(float(leg2["sp_era_home"]), 4.50)  # starter 303
        self.assertAlmostEqual(float(leg2["sp_k9_home"]), 8.1)
        # Distinct start times per leg.
        self.assertLess(pd.Timestamp(leg1["start_time_utc"]),
                        pd.Timestamp(leg2["start_time_utc"]))


# ---------------------------------------------------------------------------
# _attach_slate_run_margins: never explodes rows
# ---------------------------------------------------------------------------


def _decided_games() -> pd.DataFrame:
    return pd.DataFrame([
        {"game_pk": 1, "game_date": "2026-08-01", "home_team": "NYY",
         "away_team": "BOS", "home_win": 1.0, "home_score": 5,
         "away_score": 3, "total_runs": 8},
        {"game_pk": 2, "game_date": "2026-08-02", "home_team": "BOS",
         "away_team": "NYY", "home_win": 0.0, "home_score": 2,
         "away_score": 6, "total_runs": 8},
    ])


def _board_row(gid: str, start_et: str) -> dict:
    return {"game_id": gid, "game_date": "2026-08-29",
            "home_team": "NYY", "away_team": "BOS",
            "start_time_utc": pd.Timestamp(start_et),
            "home_win": np.nan, "home_score": np.nan, "away_score": np.nan}


class TestAttachSlateRunMarginsDoubleheader(unittest.TestCase):
    def setUp(self):
        import training
        self._rounds_patch = mock.patch.object(
            training, "get_last_margin_rounds", return_value={"home": 8, "away": 8})
        self._rounds_patch.start()
        self.addCleanup(self._rounds_patch.stop)

    def _attach(self, board, margins):
        with mock.patch("build_oof_margin.refit_run_margins",
                        return_value=pd.DataFrame(margins)):
            return pipeline._attach_slate_run_margins(board, _decided_games())

    def test_distinct_per_leg_keys_merge_one_to_one(self):
        """Distinct leg keys (the post-fix slate) merge 1:1 — two rows in,
        two rows out, each with ITS OWN margin. This is the 17→17 case (the
        old code exploded it to 21)."""
        board = pd.DataFrame([
            _board_row("20260829_BOS@NYY", "2026-08-29 13:05:00"),
            _board_row("20260829_BOS@NYY_2", "2026-08-29 19:15:00"),
        ])
        out = self._attach(board, {
            "game_pk": ["20260829_BOS@NYY", "20260829_BOS@NYY_2"],
            "run_margin_diff": [0.4, -0.3],
        })
        self.assertEqual(len(out), 2)
        got = dict(zip(out["game_pk"], out["run_margin_diff"]))
        self.assertAlmostEqual(got["20260829_BOS@NYY"], 0.4)
        self.assertAlmostEqual(got["20260829_BOS@NYY_2"], -0.3)

    def test_exact_duplicate_game_pk_deduped_not_exploded(self):
        """Two rows with the SAME game_pk (the pre-fix matchup keyed board)
        are a true bug: the merge must keep ONE row, never multiply it."""
        board = pd.DataFrame([
            _board_row("20260829_BOS@NYY", "2026-08-29 13:05:00"),
            _board_row("20260829_BOS@NYY", "2026-08-29 19:15:00"),
        ])
        out = self._attach(board, {
            "game_pk": ["20260829_BOS@NYY", "20260829_BOS@NYY"],
            "run_margin_diff": [0.4, -0.3],
        })
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.iloc[0]["run_margin_diff"], 0.4)


# ---------------------------------------------------------------------------
# _carry_forward_slate_details: each leg restored from ITS OWN morning row
# ---------------------------------------------------------------------------


def _carry_row(gid: str, sp_home: str, era: float) -> dict:
    return {"game_id": gid, "home_team": "NYY", "away_team": "BOS",
            "sp_name_home": sp_home, "sp_name_away": "TBD",
            "sp_era_home": era, "sp_k9_home": 9.0,
            "sp_era_away": np.nan, "sp_k9_away": np.nan,
            "moneyline_home": np.nan, "moneyline_away": np.nan,
            "total_line": np.nan, "run_line_home": np.nan, "juice": np.nan}


class TestCarryForwardDoubleheader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig = pipeline.DATA_DELIVERY_DIR
        pipeline.DATA_DELIVERY_DIR = self.tmp
        self.addCleanup(setattr, pipeline, "DATA_DELIVERY_DIR", self._orig)

    def test_each_leg_restored_from_its_own_morning_row(self):
        """Morning artifact holds BOTH legs with DIFFERENT starters; the
        evening rebuild (all TBD) must restore leg 1 from leg 1's row and
        leg 2 from leg 2's row — never the same pitcher on both."""
        morning = pd.DataFrame([
            _carry_row("20260829_BOS@NYY", "Cortes, Nestor", 3.10),
            _carry_row("20260829_BOS@NYY_2", "Fried, Max", 2.89),
        ])
        morning.to_csv(self.tmp / "todays_games_20990101.csv", index=False)

        rebuilt = pd.DataFrame([
            _carry_row("20260829_BOS@NYY", "TBD", np.nan),
            _carry_row("20260829_BOS@NYY_2", "TBD", np.nan),
        ])
        out = pipeline._carry_forward_slate_details(rebuilt, "20990101")
        by_id = {r["game_id"]: r for _, r in out.iterrows()}
        self.assertEqual(by_id["20260829_BOS@NYY"]["sp_name_home"], "Cortes, Nestor")
        self.assertAlmostEqual(float(by_id["20260829_BOS@NYY"]["sp_era_home"]), 3.10)
        self.assertEqual(by_id["20260829_BOS@NYY_2"]["sp_name_home"], "Fried, Max")
        self.assertAlmostEqual(float(by_id["20260829_BOS@NYY_2"]["sp_era_home"]), 2.89)


if __name__ == "__main__":
    unittest.main()
