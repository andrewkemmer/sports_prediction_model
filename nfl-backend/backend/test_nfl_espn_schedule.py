"""Unit tests for the ESPN 2026 NFL schedule loader (network-free).

Focuses on the pure seams: alias mapping, ET-date derivation, roof/div/score
synthesis, week estimation against the season anchor, the ``pre``-only filter,
dedup/sort, and the nflreadpy-shaped columns.
"""

import unittest
from datetime import date

import pandas as pd

from nfl_espn_schedule import (
    ESPN_ABBREV_TO_KEY,
    load_espn_schedule_rows,
    parse_event,
    season_date_range,
    team_alias,
    _estimate_week,
)


def _event(home="KC", away="JAX", day="2026-09-13T20:15:00Z", state="pre",
           eid="401872658", stadium="Arrowhead Stadium", date_utc=None):
    return {
        "id": eid,
        "date": date_utc or day,
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
    def test_espo_quirks_map_to_nflverse(self):
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
    def test_evening_utc_maps_to_next_or_same_et_day(self):
        # 2026-09-14T00:35Z = 2026-09-13 evening ET
        ev = _event(day="2026-09-14T00:35:00Z")
        row = parse_event(ev, 2026, date(2026, 9, 1))
        self.assertEqual(row["gameday"], date(2026, 9, 13))


class TestParseEvent(unittest.TestCase):
    def test_shapes_nflreadpy_row(self):
        ev = _event(home="LAR", away="ARI")
        row = parse_event(ev, 2026, date(2026, 9, 1))
        self.assertEqual(row["game_id"], "2026_espn_401872658")
        self.assertEqual(row["season"], 2026)
        self.assertEqual(row["home_team"], "LAR")
        self.assertEqual(row["away_team"], "ARI")
        self.assertIsNone(row["home_score"])
        self.assertIsNone(row["away_score"])

    def test_dome_home(self):
        ev = _event(home="DET", away="CHI")
        row = parse_event(ev, 2026, date(2026, 9, 1))
        self.assertEqual(row["roof"], "dome")

    def test_outdoors_home(self):
        ev = _event(home="SF", away="SEA")
        row = parse_event(ev, 2026, date(2026, 9, 1))
        self.assertEqual(row["roof"], "outdoors")

    def test_filled_week_from_anchor(self):
        ev = _event(day="2026-09-13T20:15:00Z")
        row = parse_event(ev, 2026, date(2026, 9, 10))
        self.assertEqual(row["week"], 1)      # day 3 past the anchor -> week 1
        ev2 = _event(day="2026-10-11T20:15:00Z")
        row2 = parse_event(ev2, 2026, date(2026, 9, 10))
        self.assertGreaterEqual(row2["week"], 4)

    def test_started_game_dropped(self):
        ev = _event(state="post", eid="401872000")
        self.assertIsNone(parse_event(ev, 2026, date(2026, 9, 1)))

    def test_unknown_team_dropped(self):
        ev = _event(home="ZZ", away="KC")
        self.assertIsNone(parse_event(ev, 2026, date(2026, 9, 1)))

    def test_div_game(self):
        div = _event(home="KC", away="DEN")
        out = _event(home="KC", away="GB")
        self.assertEqual(parse_event(div, 2026, date(2026, 9, 1))["div_game"], 1)
        self.assertEqual(parse_event(out, 2026, date(2026, 9, 1))["div_game"], 0)


class TestSeasonDateRange(unittest.TestCase):
    def test_default_window_bounds(self):
        rng = season_date_range(2026)
        self.assertEqual(rng[0], date(2026, 9, 1))
        self.assertEqual(rng[-1], date(2027, 1, 31))

    def test_empty_when_end_before_start(self):
        self.assertEqual(season_date_range(2026, date(2026, 9, 1), date(2026, 8, 1)), [])


class TestEstimateWeek(unittest.TestCase):
    def test_week_edge(self):
        self.assertEqual(_estimate_week(date(2026, 9, 10), date(2026, 9, 10)), 1)
        self.assertEqual(_estimate_week(date(2026, 9, 17), date(2026, 9, 10)), 2)


class TestLoadNoNetwork(unittest.TestCase):
    """load_espn_schedule_rows must not raise for an empty window (no network)."""

    def test_empty_window_returns_empty_frame(self):
        df = load_espn_schedule_rows(2026, start=date(2026, 1, 1), end=date(2026, 1, 2))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()