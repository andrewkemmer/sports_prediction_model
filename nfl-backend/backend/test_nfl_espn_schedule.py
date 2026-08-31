"""Unit tests for the ESPN 2026 NFL schedule loader (network-free).

Focuses on the pure seams: alias mapping, ET-date derivation, roof/div/score
synthesis, the week threading, the ``pre``-only filter, and dedup/sort. The
network pull is exercised only through a patched ``_fetch_week``.
"""

import unittest
from datetime import date
from unittest import mock

import pandas as pd

import nfl_espn_schedule
from nfl_espn_schedule import (
    load_espn_schedule_rows,
    parse_event,
    season_weeks,
    team_alias,
)


def _event(home="KC", away="JAX", day="2026-09-13T20:15:00Z", state="pre",
           eid="401872658", stadium="Arrowhead Stadium"):
    return {
        "id": eid,
        "date": day,
        "competitions": [{
            "status": {"type": {"state": state}},
            "venue": {"fullName": stadium},
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": home}},
                {"homeAway": "away", "team": {"abbreviation": away}},
            ],
        }],
    }


class TestTeamAlias(unittest.TestCase):
    def test_espn_quirks_map_to_nflverse(self):
        self.assertEqual(team_alias("LA"), "LAR")
        self.assertEqual(team_alias("JAC"), "JAX")
        self.assertEqual(team_alias("WSH"), "WAS")

    def test_identity(self):
        self.assertEqual(team_alias("KC"), "KC")
        self.assertEqual(team_alias("SF"), "SF")

    def test_unknown_returns_none(self):
        self.assertIsNone(team_alias("ZZ"))
        self.assertIsNone(team_alias(""))


class TestEtDate(unittest.TestCase):
    def test_evening_utc_maps_to_et_day(self):
        # 2026-09-14T00:35Z = 2026-09-13 evening ET
        ev = _event(day="2026-09-14T00:35:00Z")
        row = parse_event(ev, 2026, 1)
        self.assertEqual(row["gameday"], date(2026, 9, 13))


class TestParseEvent(unittest.TestCase):
    def test_shapes_nflreadpy_row(self):
        ev = _event(home="LAR", away="ARI")
        row = parse_event(ev, 2026, 1)
        self.assertEqual(row["game_id"], "2026_espn_401872658")
        self.assertEqual(row["season"], 2026)
        self.assertEqual(row["week"], 1)
        self.assertEqual(row["home_team"], "LAR")
        self.assertEqual(row["away_team"], "ARI")
        self.assertIsNone(row["home_score"])
        self.assertIsNone(row["away_score"])

    def test_week_from_request(self):
        self.assertEqual(parse_event(_event(eid="a"), 2026, 10)["week"], 10)
        self.assertEqual(parse_event(_event(eid="b"), 2026, 18)["week"], 18)

    def test_dome_home(self):
        self.assertEqual(parse_event(_event(home="DET", away="CHI"), 2026, 1)["roof"], "dome")

    def test_outdoors_home(self):
        self.assertEqual(parse_event(_event(home="SF", away="SEA"), 2026, 1)["roof"], "outdoors")

    def test_started_game_dropped(self):
        self.assertIsNone(parse_event(_event(state="post", eid="x"), 2026, 1))

    def test_unknown_team_dropped(self):
        self.assertIsNone(parse_event(_event(home="ZZ", away="KC"), 2026, 1))

    def test_div_game(self):
        self.assertEqual(parse_event(_event(home="KC", away="DEN"), 2026, 1)["div_game"], 1)
        self.assertEqual(parse_event(_event(home="KC", away="GB"), 2026, 1)["div_game"], 0)


class TestSeasonWeeks(unittest.TestCase):
    def test_weeks_1_to_18(self):
        self.assertEqual(season_weeks(2026), list(range(1, 19)))


class TestReadScheduleCsv(unittest.TestCase):
    def test_round_trips_csv(self):
        import tempfile
        rows = [
            {"game_id": "2026_espn_a", "season": 2026, "week": 1,
             "gameday": "2026-09-10", "gametime": "2026-09-11T00:00Z",
             "stadium": "S", "home_team": "KC", "away_team": "NE",
             "home_score": None, "away_score": None, "roof": "outdoors",
             "temp": None, "wind": None, "div_game": 0},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            pd.DataFrame(rows).to_csv(fh.name, index=False)
            path = fh.name
        df = nfl_espn_schedule.read_schedule_csv(path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["game_id"], "2026_espn_a")
        self.assertTrue(pd.isna(df.iloc[0]["home_score"]))   # scheduled -> NaN
        self.assertFalse(pd.isna(df.iloc[0]["week"]))


class TestLoadPatchedWeek(unittest.TestCase):
    def test_empty_when_all_weeks_return_none(self):
        with mock.patch.object(nfl_espn_schedule, "_fetch_week", return_value=[]):
            df = load_espn_schedule_rows(2026)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_parses_and_dedupes_patched_week(self):
        ev1 = _event(home="KC", away="BUF", eid="g1")
        ev2 = _event(home="DET", away="CHI", eid="g2")
        with mock.patch.object(nfl_espn_schedule, "_fetch_week",
                               return_value=[ev1, ev2, ev1]):
            df = load_espn_schedule_rows(2026)
        self.assertEqual(len(df), 2)                       # 3 evs deduped to 2
        self.assertEqual(set(df["game_id"]), {"2026_espn_g1", "2026_espn_g2"})
        self.assertEqual(df["week"].nunique(), 1)         # same week (int) throughout


if __name__ == "__main__":
    unittest.main()