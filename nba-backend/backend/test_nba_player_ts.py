"""Tests for the player-level True Shooting rating.

The rate itself is arithmetic. What is worth testing is every place the rating
can be quietly wrong while still producing numbers: a prior that forgets a
factor of two, a running total that gets summed instead of read, a player id
that joins in its float spelling, and a season boundary that lets last season
leak into this one's rating. Each of those produced a plausible frame during
development, which is why each gets its own test.

Everything here runs offline. The live probes that established what stats.nba.com
will and will not answer for a position are pinned in
``TestPositionSource``; reproducing their conclusions is what stops a future
adapter change from silently reintroducing a request the feed rejects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import nba_sources as src
import player_ts as ts


def _row(player, day, points, fga, fta, position="G", season="2024-25"):
    return {
        "player_id": player, "gameday": day, "season": season,
        "points": points, "fga": fga, "fta": fta, "position": position,
    }


def _frame(rows):
    return pd.DataFrame(rows)


def _with_plays(rows):
    frame = _frame(rows)
    frame["plays"] = ts.scoring_plays(frame.fga, frame.fta)
    return frame


class TestTrueShootingRate:
    def test_rate_is_points_over_two_scoring_plays(self):
        got = float(ts.true_shooting(pd.Series([10]), pd.Series([4]),
                                     pd.Series([2])).iloc[0])
        assert got == pytest.approx(10 / (2 * (4 + 0.44 * 2)))

    def test_a_zero_attempt_game_is_undefined_not_zero(self):
        """Zero attempts is genuinely unknown, not a performance of zero.

        Coding it 0.0 would drag every rate the player appears in toward zero,
        and it would do so silently - the column would look populated.
        """
        got = ts.true_shooting(pd.Series([0]), pd.Series([0]), pd.Series([0]))
        assert got.isna().all()

    def test_missing_counting_columns_yield_nan_not_a_crash(self):
        got = ts.true_shooting(pd.Series([10]), pd.Series([None]),
                               pd.Series([None]))
        assert got.isna().all()


class TestPlayerIdSpelling:
    def test_a_float_id_loses_its_decimal_tail(self):
        """The season log's player_id is a float column.

        A bare ``astype(str)`` yields ``"1628983.0"`` while the position table
        is keyed on the integer id the feed publishes, so the two never join -
        and the frame still produces ratings, just with no position and
        therefore no prior. The failure is invisible in the output.
        """
        got = ts._player_id_str(pd.Series([1628983.0, "2544", 203076.0]))
        assert list(got) == ["1628983", "2544", "203076"]

    def test_prepared_rows_carry_the_clean_id(self):
        out = ts.prepare_player_games(
            _frame([_row(1628983.0, "2024-11-01", 20, 10, 4)]))
        assert list(out.player_id) == ["1628983"]


class TestShrinkage:
    def test_a_zero_evidence_player_lands_exactly_on_the_league_prior(self):
        """The whole point of the prior, stated as its limiting case."""
        got = ts.shrunk_ts(pd.Series([0.0]), pd.Series([0.0]),
                           pd.Series([0.55]), pd.Series([100.0]))
        assert float(got.iloc[0]) == pytest.approx(0.55)

    def test_evidence_moves_the_rating_off_the_prior(self):
        got = ts.shrunk_ts(pd.Series([200.0]), pd.Series([100.0]),
                           pd.Series([0.55]), pd.Series([100.0]))
        # (200 + 2 * 0.55 * 100) / (2 * (100 + 100)) = 310 / 400
        assert float(got.iloc[0]) == pytest.approx(0.775)

    def test_the_rating_carries_the_factor_of_two(self):
        """Regression: dropping TS's factor of two returns ~0.93, not ~0.56.

        That is not an error, it is a wrong number in the right range, so
        nothing downstream objects to it.
        """
        got = float(ts.shrunk_ts(pd.Series([100.0]), pd.Series([100.0]),
                                 pd.Series([0.55]), pd.Series([0.0])).iloc[0])
        assert got == pytest.approx(0.5)
        assert got < 0.75

    def test_more_evidence_means_less_prior_weight(self):
        """Both players shoot .600; the one with more evidence sits nearer it.

        The prior's job is to pull a THIN record toward the league, so as
        evidence accumulates the rating should approach the player's own rate -
        and since .600 is above the .550 prior, "closer to own rate" means
        "further from the league".
        """
        # points chosen so the raw rate is .600 at each volume:
        # 30 / (2 * 25) and 300 / (2 * 250).
        thin = float(ts.shrunk_ts(pd.Series([30.0]), pd.Series([25.0]),
                                  pd.Series([0.55]), pd.Series([100.0])).iloc[0])
        thick = float(ts.shrunk_ts(pd.Series([300.0]), pd.Series([250.0]),
                                   pd.Series([0.55]), pd.Series([100.0])).iloc[0])
        assert thin < thick
        assert abs(thin - 0.55) < abs(thick - 0.55)
        assert abs(thick - 0.60) < abs(thin - 0.60)


class TestSeasonPlaysPriorTable:
    def test_k_is_twenty_percent_of_the_mean_player_season(self):
        """The MLB/NHL convention, carried across by its fraction.

        MLB's fixed 120-PA prior is 20% of a 600-PA season, so the fraction is
        the portable part and the season length is sport-specific.
        """
        games = _with_plays([
            _row("a", "2024-11-01", 20, 10, 4),
            _row("a", "2024-11-02", 20, 10, 4),
            _row("b", "2024-11-01", 20, 10, 4),
        ])
        table = ts.season_plays_table(games, shrink_fraction=0.20)
        per_season = games.groupby(["player_id", "season"]).plays.sum()
        # Two player-seasons, not three rows: "a"'s two November games are one
        # season of evidence, which is the whole point of the unit.
        assert len(per_season) == 2
        assert table["G"] == pytest.approx(0.20 * per_season.mean())

    def test_the_unit_is_a_player_season_not_a_player(self):
        """Averaging per player would weight a two-game cameo like a full season.

        This is the correction NHL documents at length: dividing by seasons
        first collapses the reference season, which would make k several times
        too small and under-shrink every rating. The two units are computed
        side by side here so the test says which one this implements rather
        than only what it does not.
        """
        rows = [_row("a", "2024-11-01", 20, 10, 4),
                _row("a", "2024-11-02", 20, 10, 4),
                _row("a", "2024-11-03", 20, 10, 4),
                _row("a", "2024-11-01", 20, 10, 4, season="2023-24"),
                _row("b", "2024-11-01", 20, 10, 4)]
        games = _with_plays(rows)
        one = 10 + 0.44 * 4
        table = ts.season_plays_table(games, shrink_fraction=0.20)

        # Per PLAYER-SEASON: (3u + u + u) / 3.
        per_player_season = (3 * one + one + one) / 3
        assert table["G"] == pytest.approx(0.20 * per_player_season)
        # Per PLAYER, dividing each career by its season count first: a is
        # (3u + u)/2 = 2u and b is u, so the mean is 1.5u - a different number.
        per_player = ((3 * one + one) / 2 + one) / 2
        assert per_player != pytest.approx(per_player_season)
        assert table["G"] != pytest.approx(0.20 * per_player)

    def test_an_empty_position_cell_keeps_a_usable_default(self):
        """A cell with no data must not raise at lookup time.

        If it raised, one unpopulated position would take the whole rating
        build down rather than degrading just that cell.
        """
        table = ts.season_plays_table(
            _with_plays([_row("a", "2024-11-01", 20, 10, 4)]))
        for position in config.PLAYER_TS_POSITIONS:
            assert position in table
            assert table[position] > 0
        assert table["C"] == config.PLAYER_TS_FALLBACK_K_PLAYS

    def test_no_data_yields_a_complete_table_rather_than_nothing(self):
        for games in (None, pd.DataFrame()):
            table = ts.season_plays_table(games)
            assert set(table) == set(config.PLAYER_TS_POSITIONS)


class TestPositionAssignment:
    def test_a_two_position_player_lands_in_exactly_one_cell(self):
        """Cells are the denominators of the league prior.

        A player counted in two cells is counted twice in the league mean and
        the cells stop being comparable. 52 of 569 players in 2024-25 are
        listed at both G and F, so this runs on a fifth of the league.
        """
        assigned = src.assign_positions(
            {"G": {"1", "2"}, "F": {"2", "3"}, "C": {"4"}})
        assert assigned == {"1": "G", "2": "G", "3": "F", "4": "C"}

    def test_every_listed_player_is_assigned_exactly_once(self):
        assigned = src.assign_positions({"G": {"1"}, "F": {"1", "2"},
                                         "C": {"2"}})
        assert sorted(assigned) == ["1", "2"]

    def test_a_position_the_feed_never_lists_is_simply_absent(self):
        frame = src.positions_frame({"G": {"1"}})
        assert list(frame.columns) == ["player_id", "position"]
        assert list(frame.player_id) == ["1"]

    def test_no_positions_yields_a_well_formed_empty_frame(self):
        frame = src.positions_frame({})
        assert frame.empty
        assert list(frame.columns) == ["player_id", "position"]


class TestPositionSource:
    def test_only_three_positions_are_asked_for(self):
        """Pinned because the five-way split is NOT available from this source.

        Measured against the live endpoint, stats.nba.com's ``PlayerPosition``
        filter answers G, F and C and returns HTTP 400 for PG/SG/SF/PF and for
        compound codes like ``G-F``. Adding a fourth code would change what the
        prior can mean, so it has to be a deliberate edit rather than a
        discovery at runtime.
        """
        assert config.PLAYER_TS_POSITIONS == ("G", "F", "C")

    def test_the_query_carries_the_position_and_a_full_parameter_set(self):
        query = src.position_query("2024-25", "C")
        assert "PlayerPosition=C" in query
        assert "Season=2024-25" in query
        # An incomplete set answers 500, which would look like an upstream
        # outage rather than a missing argument.
        for required in ("LeagueID=00", "PerMode=PerGame", "MeasureType=Base",
                         "SeasonType=Regular+Season"):
            assert required in query

    def test_players_at_position_reads_the_id_column(self):
        payload = {"resultSets": [{"headers": ["PLAYER_ID", "PLAYER_NAME"],
                                   "rowSet": [[1, "A"], [2, "B"]]}]}
        assert src.players_at_position(payload) == {1, 2}

    def test_a_payload_without_the_column_yields_nothing_rather_than_raising(self):
        payload = {"resultSets": [{"headers": ["TEAM_ID"], "rowSet": [[1]]}]}
        assert src.players_at_position(payload) == set()
        assert src.players_at_position(None) == set()


class TestPositionJoin:
    """The position join is keyed on the season, and that is load-bearing.

    Joining on ``player_id`` alone matched one game against that player's row
    in every season file, duplicating 92% of the rating frame. The rating's
    RATE is a ratio and survived, which is exactly why it went unnoticed -
    while ``prior_plays``, ``prior_games``, the shrinkage weight and both pool
    gates were quietly inflated up to 3x.
    """

    @staticmethod
    def _stats(n_games=3):
        return pd.DataFrame({
            "player_id": [1628983.0] * n_games,
            "gameday": pd.to_datetime(["2023-11-01", "2023-11-03", "2023-11-05"]),
            "points": [20.0, 30.0, 10.0],
            "fga": [15.0, 20.0, 8.0],
            "fta": [5.0, 6.0, 2.0],
        })

    def test_a_player_in_three_seasons_yields_one_row_per_game(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983"] * 3,
            "position": ["G", "G", "F"],
            "season": ["2023-24", "2024-25", "2025-26"],
        })
        frame = ts.prepare_player_games(stats, positions)
        assert len(frame) == 3
        assert frame.gameday.nunique() == 3

    def test_a_season_specific_position_is_matched_to_its_own_season(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983", "1628983"],
            "position": ["F", "G"],
            "season": ["2023-24", "2025-26"],
        })
        frame = ts.prepare_player_games(stats, positions)
        # every game here is 2023-24, so the 2025-26 row must not apply
        assert set(frame.position) == {"F"}

    def test_one_row_per_player_per_game_holds(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983"] * 2,
            "position": ["G", "C"],
            "season": ["2023-24", "2023-24"],
        })
        frame = ts.prepare_player_games(stats, positions)
        assert len(frame.drop_duplicates(["player_id", "gameday"])) == len(frame)

    def test_a_player_season_listed_twice_keeps_one_cell(self):
        """Two positions in ONE season must not fan the player out either."""
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983", "1628983"],
            "position": ["F", "G"],
            "season": ["2023-24", "2023-24"],
        })
        frame = ts.prepare_player_games(stats, positions)
        assert len(frame) == 3
        assert frame.position.nunique() == 1

    def test_without_a_season_column_the_join_still_dedupes(self):
        stats = self._stats()
        positions = pd.DataFrame({
            "player_id": ["1628983", "1628983"],
            "position": ["G", "F"],
        })
        frame = ts.prepare_player_games(stats, positions)
        assert len(frame) == 3


class TestAvailability:
    def test_status_words_map_to_availability(self):
        assert src.availability_status("Out") == "out"
        assert src.availability_status("Day-To-Day") == "day_to_day"
        assert src.availability_status("Active") == "healthy"
        assert src.availability_status(None) == "healthy"

    def test_the_official_designations_survive_normalisation(self):
        """``Questionable`` used to normalise to ``out`` on a substring match.

        That was right for ESPN and wrong for the official filing: over
        2025-26 a player designated Questionable before tipoff played 34
        times in 52. Folding it into ``out`` deletes a third of a real
        player's appearances and files the result as an absence.
        """
        for designation in ("Out", "Doubtful", "Recovery", "Questionable",
                            "Available", "Probable"):
            assert src.availability_status(designation) == designation.lower()

    def test_normalisation_is_idempotent_over_the_official_vocabulary(self):
        """The injury table stores a normalized status and the stint builder
        normalizes again, so a second pass must not move a player."""
        for designation in ("Out", "Questionable", "Probable", "Available"):
            once = src.availability_status(designation)
            assert src.availability_status(once) == once

    def test_an_unknown_status_resolves_healthy_rather_than_penalising(self):
        """An unrecognised word must not silently dock a player.

        The raw string is carried alongside so the unknown value stays visible
        in the artifact instead of resolving invisibly to healthy.
        """
        assert src.availability_status("Felt a little under the weather") \
            == "healthy"

    def test_the_weighted_table_has_no_caller_left(self):
        """The injury path decided for binary REMOVAL (a false flag drops the
        player from the pool and a replacement inherits the slot), so the
        weighted form of the same vocabulary was a second answer to a
        question the pipeline no longer asks - kept alive only by its own
        tests. The measured rates remain in
        ``config.PLAYER_TS_DESIGNATION_PLAY_RATE``; the multiplier is gone
        rather than kept as an audited corpse."""
        assert not hasattr(src, "availability_multiplier")
        import config
        assert config.PLAYER_TS_DESIGNATION_PLAY_RATE["out"] == 0.0
        assert config.PLAYER_TS_DESIGNATION_PLAY_RATE["available"] == 0.817

    def test_an_empty_injuries_list_is_healthy_not_missing(self):
        """The common case, and the one that must not read as no data.

        ESPN populates the array sparsely by design - 7 of 76 rostered players
        in a four-team sample carried an entry - so empty is normal.
        """
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                  "injuries": []}]}
        assert src.roster_availability(payload, team="BOS") == [
            ("1", "BOS", "healthy", "", None)]

    def test_an_injury_entry_sets_the_status_and_keeps_the_raw_text(self):
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                  "injuries": [{"status": {"name": "Out"}}]}]}
        assert src.roster_availability(payload, team="BOS") == [
            ("1", "BOS", "out", "Out", None)]

    def test_the_real_espn_shape_is_a_bare_status_string(self):
        """Pinned against the live feed, which sends ``status`` as a string.

        Measured 2024-25: every injury entry on five rosters was exactly
        ``{"status": "Out", "date": "..."}`` with no nested ``name``. A parser
        written for the nested shape reads ``status.name`` off a string, finds
        nothing, and reports every player healthy - the gate stays open and
        every check downstream still passes.
        """
        payload = {"athletes": [{"id": "6430", "displayName": "Jimmy Butler",
                                  "injuries": [{"status": "Out",
                                                "date": "2026-08-24T15:23Z"}]}]}
        assert src.roster_availability(payload, team="GSW") == [
            ("6430", "GSW", "out", "Out", "2026-08-24T15:23Z")]

    def test_a_bare_day_to_day_string_is_halved(self):
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                  "injuries": [{"status": "Day-To-Day",
                                                "date": "2026-09-02T16:32Z"}]}]}
        assert src.roster_availability(payload)[0][2] == "day_to_day"

    def test_the_published_date_is_carried_because_it_is_the_pit_floor(self):
        """The endpoint publishes current state only, so there is no history.

        What makes a snapshot usable is that each record carries the moment it
        was published. Dropping it would force the PIT floor to fall back to
        the snapshot's own date, which is the leak the column exists to
        prevent.
        """
        payload = {"athletes": [{"id": "1", "displayName": "A",
                                  "injuries": [{"status": "Out",
                                                "date": "2026-08-24T15:23Z"}]}]}
        assert src.roster_availability(payload)[0][4] == "2026-08-24T15:23Z"

    def test_the_status_mapping_is_idempotent(self):
        """Re-normalizing an already-normalized status must not change it.

        The injury table stores a normalized status and the stint builder
        normalizes it again. Without idempotence "day_to_day" falls through
        every branch and resolves to "healthy", so a partial absence is
        silently dropped and the player stays in the projected lineup.
        """
        for status in ("out", "day_to_day", "healthy"):
            once = src.availability_status(status)
            assert src.availability_status(once) == once

    def test_an_athlete_with_no_id_is_skipped(self):
        payload = {"athletes": [{"displayName": "A", "injuries": []}]}
        assert src.roster_availability(payload) == []


class TestPointInTimeDiscipline:
    def _season_games(self):
        return _frame([
            _row("a", "2024-11-01", 10, 5, 0),
            _row("a", "2024-11-02", 30, 10, 0),
            _row("a", "2024-11-03", 20, 20, 0),
        ])

    def test_a_games_own_line_never_enters_its_own_prior(self):
        """A rating dated on a player's own game must not see that game.

        The player's opener is the clean case: with the target day excluded
        there is nothing before it, so the prior is zero however much they
        scored in it - and the rating collapses to the league prior, which is
        the only defensible answer for a player with no evidence.
        """
        games = ts.prepare_player_games(self._season_games())
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-01"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_points == 0
        assert row.prior_plays == 0
        assert row.prior_games == 0

    def test_a_player_with_no_evidence_is_still_emitted(self):
        """"No rating yet" and "never in the data" are different facts.

        A projection that cannot tell them apart will happily project a player
        who does not exist, so the row is present with a zero prior rather than
        dropped.
        """
        games = ts.prepare_player_games(self._season_games())
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-01"]))
        assert list(ratings.player_id) == ["a"]
        assert ratings.iloc[0].ts_raw is np.nan or pd.isna(ratings.iloc[0].ts_raw)

    def test_a_rating_does_not_see_the_target_game(self):
        games = ts.prepare_player_games(self._season_games())
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-02"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        # Only the 10-point opener is prior; the 30-point target day is not.
        assert row.prior_points == 10
        assert row.prior_plays == pytest.approx(5)

    def test_a_rating_does_not_carry_the_previous_season(self):
        """The boundary the season partition exists to hold.

        A carry-over is invisible in the output: a player with a full season of
        evidence looks equally well-rated either way.
        """
        games = ts.prepare_player_games(_frame([
            _row("a", "2024-06-01", 40, 20, 0, season="2023-24"),
            _row("a", "2024-11-01", 10, 5, 0, season="2024-25"),
        ]))
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-02"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_points == 10

    def test_the_prior_is_a_running_total_read_not_summed(self):
        """Regression: summing the running total grows with the square of the
        season and returns a number several times the player's real points."""
        games = ts.prepare_player_games(self._season_games())
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-04"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_points == 60  # 10 + 30 + 20, not their partial sums

    def test_duplicate_player_games_do_not_double_count(self):
        """Overlapping fetch windows repeat a player-game, and a duplicate
        would double both sides of the rate - which looks like a much better
        performance than it was."""
        games = ts.prepare_player_games(_frame([
            _row("a", "2024-11-01", 10, 5, 0),
            _row("a", "2024-11-01", 10, 5, 0),
            _row("a", "2024-11-05", 4, 2, 0),
        ]))
        assert len(games) == 2
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-06"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_points == 14

    def test_a_season_spanning_new_year_is_one_season_not_two(self):
        """An NBA season runs October to June, so it crosses New Year.

        Deriving the season from the calendar year would split every season in
        half at 1 January and hand each half a thin prior.
        """
        games = ts.prepare_player_games(_frame([
            _row("a", "2024-12-30", 10, 5, 0, season="2024-25"),
            _row("a", "2025-01-02", 10, 5, 0, season="2024-25"),
        ]))
        assert set(games.season) == {"2024-25"}


class TestPositionSegmentedLeaguePrior:
    def _two_position_league(self):
        """A guard and a centre with deliberately different true shooting."""
        rows = []
        for day in ("2024-11-01", "2024-11-02", "2024-11-03"):
            rows.append(_row("g", day, 10, 10, 0, position="G"))
            rows.append(_row("c", day, 30, 10, 0, position="C"))
        return ts.prepare_player_games(_frame(rows))

    def test_the_league_mean_differs_by_position(self):
        """The reason the prior is segmented at all.

        One league mean would rate this centre as an above-average guard and
        that guard as a below-average centre - the NHL defenceman problem in a
        different sport.
        """
        games = self._two_position_league()
        league = ts.league_prior_table(games, pd.Series(["2024-11-04"]))
        table = dict(zip(league.position, league.lg_ts))
        assert table["C"] > table["G"]
        assert table["G"] == pytest.approx(0.5)   # 10 / (2 * 10)
        assert table["C"] == pytest.approx(1.5)   # 30 / (2 * 10)

    def test_a_rating_is_pulled_toward_its_own_position_not_the_league(self):
        games = self._two_position_league()
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-04"]))
        by_id = dict(zip(ratings.player_id, ratings.ts_shrunk))
        # Each player's evidence equals their own position's mean, so each
        # stays put. A single unsegmented prior would drag both toward one
        # number between them.
        assert by_id["g"] == pytest.approx(0.5)
        assert by_id["c"] == pytest.approx(1.5)
        assert by_id["g"] != by_id["c"]

    def test_prior_strength_is_defined_for_every_position_in_play(self):
        games = self._two_position_league()
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-04"]))
        k = dict(zip(ratings.position, ratings.k_plays))
        assert set(k) == {"G", "C"}
        assert all(value > 0 for value in k.values())


class TestShrinkageBehaviour:
    def test_a_one_game_rookie_is_pulled_back_toward_the_league(self):
        """The case the prior exists for, as an end-to-end assertion.

        A single game's raw rate is almost pure noise - two made shots on two
        attempts is a 1.0 TS - and the rating must not carry that onward. The
        league here is given realistic season volume so ``k`` is a meaningful
        weight; a two-game league would make every prior negligible and the
        assertion would pass or fail on the fixture rather than the rating.
        """
        rows = [_row("rookie", "2024-11-01", 4, 2, 0)]
        for day in range(1, 21):
            rows.append(_row(f"vet{day}", f"2024-11-{day:02d}", 22, 20, 4))
        games = ts.prepare_player_games(_frame(rows))
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-12-01"]))
        rookie = ratings[ratings.player_id == "rookie"].iloc[0]
        assert rookie.ts_raw == pytest.approx(1.0)
        # Moved off the raw rate, toward the league, but not all the way -
        # some of the 1.0 is real.
        assert rookie.lg_ts < rookie.ts_shrunk < rookie.ts_raw
        assert rookie.ts_shrunk < 0.75

    def test_shrinkage_preserves_rank_but_reduces_spread(self):
        """Both halves matter: a prior that reorders players is broken, and one
        that does not reduce spread is not doing anything."""
        rows = []
        for index, (player, points) in enumerate(
                [("a", 40), ("b", 30), ("c", 20), ("d", 10), ("e", 5)]):
            for day in range(index + 1):
                rows.append(_row(player, f"2024-11-0{day + 1}", points, 10, 0))
        games = ts.prepare_player_games(_frame(rows))
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-12-01"]))
        rated = ratings.dropna(subset=["ts_raw"])
        assert len(rated) == 5
        assert rated.ts_raw.corr(rated.ts_shrunk, method="spearman") > 0.9
        assert rated.ts_shrunk.std() < rated.ts_raw.std()

    def test_the_rating_carries_no_availability_multiplier(self):
        """The multiplier is gone, and its absence is the point.

        Zero-weighting an injured rating leaves the player IN the pool dragging
        the mean toward zero. Removal happens at pool construction instead, so
        a rating is a property of the player and nothing else - and a column
        that is constant "healthy" on every row is worse than no column.
        """
        games = ts.prepare_player_games(_frame([_row("a", "2024-11-01", 10, 5, 0)]))
        availability = pd.DataFrame({"player_id": ["a"], "status": ["out"]})
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-02"]),
            availability=availability)
        assert "availability_multiplier" not in ratings.columns
        assert "availability" not in ratings.columns
        # The rating is byte-identical with and without the injury input: a
        # player still HAS a true shooting percentage whether or not he is
        # dressing tonight. Only the pool changes.
        without = ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-02"]))
        assert ratings.iloc[0].ts_shrunk == pytest.approx(
            without.iloc[0].ts_shrunk)


class TestPlayerTsWithoutInputs:
    @pytest.mark.parametrize("stats", [None, pd.DataFrame()])
    def test_no_log_yields_an_empty_frame_not_none(self, stats):
        """A caller must be able to tell no data from a failed pull."""
        out = ts.prepare_player_games(stats)
        assert out is not None and out.empty

    def test_a_log_without_the_shooting_columns_yields_nothing(self):
        out = ts.prepare_player_games(pd.DataFrame(
            {"player_id": ["a"], "gameday": ["2024-11-01"], "points": [10]}))
        assert out.empty

    def test_a_frame_with_no_resolved_positions_refuses_to_rate(self):
        """Rating without a position would mean an UNSEGMENTED prior.

        That is the single-league-mean case the segmentation exists to avoid,
        so it is refused rather than quietly produced.
        """
        games = ts.prepare_player_games(_frame([_row("a", "2024-11-01", 10, 5, 0)]))
        games["position"] = np.nan
        assert ts.build_player_ts(
            games, target_dates=pd.Series(["2024-11-02"])).empty

    def test_no_prior_date_yields_an_empty_frame(self):
        games = ts.prepare_player_games(_frame([_row("a", "2024-11-01", 10, 5, 0)]))
        assert ts.build_player_ts(games, target_dates=pd.Series([])).empty

    def test_a_target_before_any_game_yields_nothing(self):
        games = ts.prepare_player_games(_frame([_row("a", "2024-11-01", 10, 5, 0)]))
        assert ts.build_player_ts(
            games, target_dates=pd.Series(["2024-10-01"])).empty


class TestEvidenceSeasonFallback:
    """A target whose OWN season has no games yet rates from the last one.

    This is the defect the production run reported as "player TS skipped: no
    player had strictly-prior evidence as of 2026-10-20": the games with no
    result are always the first games of a season, and the season partition
    refused to look across the boundary, so the artifact and the slate's
    lineup features were empty on exactly the days they exist to serve.
    """

    @staticmethod
    def _two_seasons():
        return ts.prepare_player_games(_frame([
            _row("a", "2024-11-01", 20, 10, 0, season="2024-25"),
            _row("a", "2024-11-05", 10, 5, 0, season="2024-25"),
        ]))

    def test_the_first_game_of_a_season_is_rated_from_the_completed_one(self):
        ratings = ts.build_player_ts(
            self._two_seasons(), target_dates=pd.Series(["2025-10-22"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_points == 30          # both completed-season games
        assert row.prior_games == 2
        # The shrink target has to exist too, or the fallback returns a row
        # whose rating is NaN - a rating, in form only.
        assert pd.notna(row.ts_shrunk)

    def test_the_fallback_stops_once_the_own_season_has_evidence(self):
        games = ts.prepare_player_games(_frame([
            _row("a", "2024-11-01", 20, 10, 0, season="2024-25"),
            _row("a", "2025-10-01", 4, 2, 0, season="2025-26"),
        ]))
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2025-10-22"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        # The partition still holds: one in-season game, and only that one.
        assert row.prior_points == 4
        assert row.prior_games == 1

    def test_season_boundary_opening_night_has_no_prior_and_no_league_mean(
            self):
        """A decided opening-night game rates from NOTHING.

        The season-boundary audit (2026-09-29, against MLB's structural
        guidance): the evidence fallback serves PRE-SEASON slate targets;
        a target on or after the season's first DECIDED game rates
        in-season from day one, so opening night's own prior is zero by
        construction and the league mean has no evidence either -
        ts_shrunk is NaN, not a rating wearing a prior's clothes.
        """
        games = ts.prepare_player_games(_frame([
            _row("a", "2025-10-22", 30, 15, 0, season="2025-26"),
        ]))
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2025-10-22"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_points == 0 and row.prior_plays == 0
        # With NO prior season to borrow from, the league cell stays empty
        # and the rating is honestly undefined. With one, opening night now
        # shrinks all the way to it - see
        # test_opening_night_shrinks_all_the_way_to_the_prior_seasons_cell.
        assert pd.isna(row.ts_shrunk)

    def test_opening_night_shrinks_all_the_way_to_the_prior_seasons_cell(
            self):
        """Game 1 gets the treatment game 2 already had.

        Game 2 leans ~94% on the league prior because a one-game prior is
        thin. Game 1's player prior is ZERO - thinner still - but until the
        2026-09-29 fix the league CELL was also empty on opening night, so
        the shrink target was NaN and the whole slate's ts_shrunk collapsed.
        The cell now borrows the most recent season with plays strictly
        before the target, so a zero-prior player lands EXACTLY on the prior
        season's position league mean: all-the-way shrinkage, strictly
        point-in-time (prior-season rows are before the target by
        construction).
        """
        games = ts.prepare_player_games(_frame([
            _row("a", "2024-11-01", 10, 5, 0, season="2024-25"),
            _row("b", "2024-11-01", 30, 15, 0, season="2024-25"),
            _row("c", "2025-10-22", 20, 10, 0, season="2025-26"),
        ]))
        target = pd.Series(["2025-10-22"])
        league = ts.league_prior_table(games, target)
        lrow = league[league.position == "G"].iloc[0]
        # The borrowed cell names its source and carries the 2024-25 mean.
        assert lrow.lg_season_source == "2024-25"
        assert lrow.lg_ts == pytest.approx(
            (10.0 + 30.0) / (2.0 * (5.0 + 15.0)))
        ratings = ts.build_player_ts(games, target_dates=target)
        row = ratings[ratings.player_id == "c"].iloc[0]
        assert row.prior_points == 0 and row.prior_plays == 0
        # Zero prior + real shrink target = exactly the league mean.
        assert row.ts_shrunk == pytest.approx(lrow.lg_ts)

    def test_the_borrow_never_touches_a_cell_with_in_season_evidence(self):
        """Day 2+ reads its OWN season; the borrow is opener-only.

        The season partition is the carryover guard - the borrow widens
        WHICH season an EMPTY cell reads, never replaces a populated one.
        """
        games = ts.prepare_player_games(_frame([
            _row("a", "2025-10-22", 10, 5, 0, season="2025-26"),
            _row("b", "2025-10-24", 30, 15, 0, season="2025-26"),
        ]))
        league = ts.league_prior_table(games, pd.Series(["2025-10-24"]))
        lrow = league[league.position == "G"].iloc[0]
        assert lrow.lg_season_source == "2025-26"
        assert lrow.lg_plays == pytest.approx(5.0)

    def test_season_boundary_day_two_is_in_season_not_carryover(self):
        """Day two+ rates strictly in-season: prior-season form is dropped.

        Pins the carryover answer the drift report's STARVED/LOW pl_ts
        coverage depends on: early-season thinness is the shrinkage doing
        its job, not missing evidence. The partition exists so a new
        season's rating never blends the previous one's tail - the same
        convention MLB states for its season-to-date ERA/K/9 LAGs.
        """
        games = ts.prepare_player_games(_frame([
            _row("a", "2025-04-01", 40, 20, 0, season="2024-25"),
            _row("a", "2025-10-21", 10, 5, 0, season="2025-26"),
        ]))
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2025-10-23"]))
        row = ratings[ratings.player_id == "a"].iloc[0]
        assert row.prior_points == 10
        assert row.prior_games == 1

    def test_the_fallback_never_reads_a_game_on_or_after_the_target(self):
        """Point-in-time is about WHEN, not about which season."""
        games = ts.prepare_player_games(_frame([
            _row("a", "2024-11-01", 20, 10, 0, season="2024-25"),
            _row("a", "2025-10-01", 999, 10, 0, season="2025-26"),
        ]))
        ratings = ts.build_player_ts(
            games, target_dates=pd.Series(["2025-10-22"]))
        # The 10-01 game is in the target's OWN season, so it is the evidence
        # season and it is strictly before the target: the fallback must not
        # prefer last season just because it is further away.
        assert ratings[ratings.player_id == "a"].iloc[0].prior_points == 999
        # ...and a target BEFORE that game falls back to the completed season,
        # still strictly earlier.
        early = ts.build_player_ts(
            games, target_dates=pd.Series(["2025-09-30"]))
        assert early[early.player_id == "a"].iloc[0].prior_points == 20

    def test_a_target_before_every_game_still_rates_nothing(self):
        """No season has evidence before the league's first game, so the
        fallback has nothing to offer and must not invent a prior."""
        assert ts.build_player_ts(
            self._two_seasons(), target_dates=pd.Series(["2024-10-01"])).empty

    def test_the_league_prior_follows_the_same_season(self):
        """The shrink target and the prior must agree on the season, or a
        fallback row is rated against a league mean from the wrong year."""
        games = self._two_seasons()
        league = ts.league_prior_table(
            games, pd.Series(["2025-10-22"]))
        assert len(league)
        assert league.lg_plays.iloc[0] > 0
