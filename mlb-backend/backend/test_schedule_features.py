"""
Regression tests for travel_fatigue_diff and closer_availability_diff.

Both features previously collapsed to structural zeros:
* travel used the travelling team's OWN city offset instead of the game
  VENUE's offset (a team's home timezone never changes -> never crosses),
  and the TZ map was missing/mismatching codes (AZ/CWS/LAA/NYM).
* closer availability attached usage state only to games the closer
  appeared in, instead of evaluating availability point-in-time for every
  game (~0.8% nonzero).
"""
import unittest

import pandas as pd

import features


def _register(con, name, df):
    con.register(name, df)


def _pitch_row(game_pk, gd, home, away, inning=9, topbot="Top",
               pitcher=999, ab=1):
    return {
        "game_pk": game_pk, "game_date": pd.Timestamp(gd),
        "home_team": home, "away_team": away,
        "inning": inning, "inning_topbot": topbot,
        "pitcher": pitcher, "at_bat_number": ab,
    }


class TestTravelFatigue(unittest.TestCase):
    def _con(self):
        import duckdb
        return duckdb.connect()

    def test_crossing_detected_via_venue_offset(self):
        """NYY: East (d1, home) -> KC Central (d2, away) = 1 crossing in the
        trailing 3-day window entering its d4 game at LAD (Pacific)."""
        rows = [
            _pitch_row(1, "2026-08-01", "NYY", "BOS"),
            _pitch_row(2, "2026-08-02", "KC", "NYY"),
            _pitch_row(3, "2026-08-04", "LAD", "NYY"),
            # control: KC stays home all week -> no crossings
            _pitch_row(4, "2026-08-02", "KC", "LAA"),
            _pitch_row(5, "2026-08-04", "KC", "SEA"),
        ]
        con = self._con()
        _register(con, "pitches", pd.DataFrame(rows))
        features._build_travel_features(con)
        out = con.execute(
            "SELECT * FROM travel_fatigue ORDER BY game_pk"
        ).fetchdf().set_index("game_pk")
        self.assertEqual(out.loc[3, "time_zones_crossed_last_3d_away"], 1.0)
        self.assertEqual(out.loc[3, "time_zones_crossed_last_3d_home"], 0.0)
        # KC played two home games: no crossing either game
        self.assertEqual(out.loc[5, "time_zones_crossed_last_3d_home"], 0.0)

    def test_same_venue_never_crosses_even_for_travelling_team_code(self):
        """The old bug: offsets came from the TEAM code, so a team always
        matched itself. With venue offsets this still holds for stay-put
        teams, while genuine venue changes register."""
        rows = [
            _pitch_row(1, "2026-07-28", "BOS", "NYY"),
            _pitch_row(2, "2026-07-29", "BOS", "NYY"),  # series, same venue
        ]
        con = self._con()
        _register(con, "pitches", pd.DataFrame(rows))
        features._build_travel_features(con)
        out = con.execute("SELECT * FROM travel_fatigue").fetchdf()
        self.assertEqual(out["time_zones_crossed_last_3d_home"].sum(), 0.0)
        self.assertEqual(out["time_zones_crossed_last_3d_away"].sum(), 0.0)

    def test_all_statcast_codes_have_offsets(self):
        """Every real MLB code must resolve — an unmatched code silently
        becomes offset 0 (the original all-zero failure mode)."""
        missing = [t for t in features.REAL_TEAM_CODES if t not in features.TEAM_TZ_OFFSETS]
        self.assertEqual(missing, [])


class TestCloserAvailability(unittest.TestCase):
    def _setup(self, pitch_rows, starter_rows):
        import duckdb
        con = duckdb.connect()
        _register(con, "pitches", pd.DataFrame(pitch_rows))
        _register(con, "starters", pd.DataFrame(starter_rows))
        features._build_closer_features(con)
        out = con.execute(
            "SELECT * FROM closer_avail ORDER BY game_pk"
        ).fetchdf().set_index("game_pk")
        return out

    def test_back_to_back_to_back_is_unavailable(self):
        """CLE reliever 51 pitched Aug 1 AND Aug 2 -> unavailable Aug 3."""
        rows = [
            # Aug 1: CLE reliever 51 works the 9th (home game, Top half)
            _pitch_row(1, "2026-08-01", "CLE", "DET", 9, "Top", 51, 10),
            # Aug 2: same reliever again
            _pitch_row(2, "2026-08-02", "MIN", "CLE", 9, "Bot", 51, 20),
            # Aug 3: CLE hosts BOS — closer unavailable entering the game
            _pitch_row(3, "2026-08-03", "CLE", "BOS", 9, "Top", 7, 30),
        ]
        starter_rows = [
            {"game_pk": 1, "home_starter_id": 45, "away_starter_id": 46},
            {"game_pk": 2, "home_starter_id": 47, "away_starter_id": 48},
            {"game_pk": 3, "home_starter_id": 49, "away_starter_id": 50},
        ]
        out = self._setup(rows, starter_rows)
        self.assertEqual(out.loc[3, "closer_available_home"], 0.0)

    def test_one_rest_day_keeps_closer_available(self):
        """Reliever pitched Aug 1 only -> available on Aug 3."""
        rows = [
            _pitch_row(1, "2026-08-01", "CLE", "DET", 9, "Top", 51, 10),
            _pitch_row(3, "2026-08-03", "CLE", "BOS", 9, "Top", 7, 30),
        ]
        starter_rows = [
            {"game_pk": 1, "home_starter_id": 45, "away_starter_id": 46},
            {"game_pk": 3, "home_starter_id": 49, "away_starter_id": 50},
        ]
        out = self._setup(rows, starter_rows)
        self.assertEqual(out.loc[3, "closer_available_home"], 1.0)

    def test_no_history_defaults_available_and_pit_excludes_tonight(self):
        """Teams without an established closer default to available, and
        tonight's own appearance can never make tonight unavailable."""
        rows = [
            # COL reliever has NO prior appearances; game Aug 3 has his row
            _pitch_row(3, "2026-08-03", "COL", "SF", 9, "Top", 7, 30),
        ]
        starter_rows = [
            {"game_pk": 3, "home_starter_id": 49, "away_starter_id": 50},
        ]
        out = self._setup(rows, starter_rows)
        self.assertEqual(out.loc[3, "closer_available_home"], 1.0)
        self.assertEqual(out.loc[3, "closer_available_away"], 1.0)

    def test_usage_on_game_day_does_not_flag_that_game(self):
        """PIT check: appearances ON game day count only for FUTURE games."""
        rows = [
            _pitch_row(1, "2026-08-02", "CLE", "DET", 9, "Top", 51, 10),
            _pitch_row(2, "2026-08-03", "CLE", "DET", 9, "Top", 51, 11),
            # Aug 4: he pitched Aug 2 AND Aug 3 -> unavailable
            _pitch_row(3, "2026-08-04", "CLE", "BOS", 9, "Top", 7, 30),
        ]
        starter_rows = [
            {"game_pk": g, "home_starter_id": 100 + g, "away_starter_id": 200 + g}
            for g in (1, 2, 3)
        ]
        out = self._setup(rows, starter_rows)
        # Entering Aug 3 he had pitched only Aug 2 -> available that day
        self.assertEqual(out.loc[2, "closer_available_home"], 1.0)
        # Entering Aug 4 he had pitched both prior days -> unavailable
        self.assertEqual(out.loc[3, "closer_available_home"], 0.0)
class TestRestDaysSeasonScoped(unittest.TestCase):
    """rest_days must be season-partitioned and capped at REST_DAYS_CAP.

    Previously LAG() ran across the offseason: March openers inherited the
    October gap as rest_days ≈ 180 — an extreme outlier against a 0–4 day
    in-season distribution.
    """

    def _build(self, rows, team="NYY"):
        import duckdb
        con = duckdb.connect()
        con.register("pitches", pd.DataFrame(rows))
        features._build_rest_days(con)
        df = con.execute(
            "SELECT * FROM rest_days ORDER BY game_pk"
        ).fetchdf()
        df = df[df["team"] == team]
        return df.set_index("game_pk")["rest_days"]

    @staticmethod
    def _row(game_pk, gd, home, away):
        return {"game_pk": game_pk, "game_date": pd.Timestamp(gd),
                "home_team": home, "away_team": away}

    def test_in_season_gap_counted(self):
        out = self._build([
            self._row(1, "2026-04-01", "NYY", "BOS"),
            self._row(2, "2026-04-04", "NYY", "BOS"),  # 3 days later
        ])
        self.assertTrue(pd.isna(out.loc[1]))   # season opener → NULL
        self.assertEqual(int(out.loc[2]), 3)

    def test_gap_capped_at_six(self):
        out = self._build([
            self._row(1, "2026-06-10", "NYY", "BOS"),
            self._row(2, "2026-07-01", "NYY", "BOS"),  # 21 days (All-Star)
        ])
        self.assertEqual(int(out.loc[2]), features.REST_DAYS_CAP)

    def test_season_opener_is_null_not_offseason_gap(self):
        out = self._build([
            self._row(1, "2025-09-28", "NYY", "BOS"),
            self._row(2, "2026-03-28", "NYY", "BOS"),  # ~180 days
        ])
        self.assertTrue(pd.isna(out.loc[2]))

    def test_same_day_doubleheader_is_zero(self):
        out = self._build([
            self._row(1, "2026-05-02", "NYY", "BOS"),
            self._row(2, "2026-05-02", "NYY", "BOS"),
        ])
        self.assertEqual(int(out.loc[2]), 0)


class TestPitcherStuffSeasonPartition(unittest.TestCase):
    """sp_xwoba_vs_l/r cumulative windows must be partitioned by SEASON.

    The old unpartitioned UNBOUNDED PRECEDING window made April values
    average in the prior October (career-to-date, not season-to-date).
    """

    def _build(self, rows):
        import duckdb
        con = duckdb.connect()
        con.register("pitches", pd.DataFrame(rows))
        features._build_pitcher_stuff(con)
        return con.execute(
            "SELECT * FROM pitcher_stuff ORDER BY game_date"
        ).fetchdf().set_index("game_pk")

    @staticmethod
    def _start(game_pk, gd, xwoba_l, pitcher=51):
        return {
            "game_pk": game_pk, "game_date": pd.Timestamp(gd),
            "pitcher": pitcher, "pitch_type": "FF", "release_speed": 95.0,
            "description": "swinging_strike",
            "events": "strikeout", "stand": "L",
            "estimated_woba_using_speedangle": xwoba_l,
        }

    def test_first_start_of_season_has_no_prior_season_leakage(self):
        rows = [
            self._start(1, "2025-08-10", 0.400),
            self._start(2, "2025-08-20", 0.400),
            self._start(3, "2026-04-01", 0.200),   # first 2026 start
            self._start(4, "2026-04-10", 0.300),   # second 2026 start
        ]
        out = self._build(rows)
        # First 2026 start: no prior IN-SEASON start → NULL (was 0.400 avg)
        self.assertTrue(pd.isna(out.loc[3, "sp_xwoba_vs_l"]))
        # Second start: only the first 2026 start counts → 0.200 (not 0.30+)
        self.assertAlmostEqual(out.loc[4, "sp_xwoba_vs_l"], 0.200, places=6)

    def test_cumulative_within_season(self):
        rows = [
            self._start(1, "2026-04-01", 0.300),
            self._start(2, "2026-04-08", 0.500),
        ]
        out = self._build(rows)
        self.assertTrue(pd.isna(out.loc[1, "sp_xwoba_vs_l"]))
        # PIT-safe: entering game 2, only PRIOR starts count (current excluded)
        self.assertAlmostEqual(out.loc[2, "sp_xwoba_vs_l"], 0.300, places=6)


if __name__ == "__main__":
    unittest.main()
