"""Tests for the three-source NBA data layer.

The suite is organised around the failures this rewrite actually hit, because
each of those was silent: a mis-joined row, a swapped venue, or a feature that
was populated by column count and meant nothing. A test that only checked the
happy path would have passed with every one of them in place.

Three groups:

* **Schedule** - the ESPN adapter, and the decisions that keep a scheduled game
  from becoming a result.
* **Features** - the season log joined to the schedule, and the team-name and
  orientation problems that join runs into.
* **Play-by-play** - the action list, the counted rollup, and the cross-check
  against the box score.

Everything runs offline. The live probes that established the field inventories
these tests pin were run once against the real endpoints; reproducing their
conclusions here is what stops a future adapter change from silently
reintroducing a field nobody returns.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import config
import features as feat
import ingestion as ing
import nba_sources as src
import source_contract as contract


# ---------------------------------------------------------------------------
# Fixtures: real payload shapes, captured from the live endpoints
# ---------------------------------------------------------------------------


def _espn_event(event_id="401585814", home="BOS", away="WSH", home_score="132",
                away_score="122", season_year=2024, season_type=2,
                state="post", completed=True, detail="Final", when=None):
    when = when or "2024-04-14T17:00Z"
    return {
        "id": event_id, "date": when, "name": f"{away} at {home}",
        "season": {"year": season_year, "type": season_type},
        "status": {"type": {"state": state, "completed": completed,
                            "detail": detail}},
        "competitions": [{
            "status": {"type": {"state": state, "completed": completed,
                                "detail": detail}},
            "date": when,
            "competitors": [
                {"homeAway": "home", "score": home_score, "winner": True,
                 "team": {"abbreviation": home, "name": home.title(),
                          "id": "2"}},
                {"homeAway": "away", "score": away_score, "winner": False,
                 "team": {"abbreviation": away, "name": away.title(),
                          "id": "27"}},
            ],
        }],
    }


def _log_row(game_id, date_str, matchup, team, player=1, name="Player",
             fga=10, fgm=5, fg3a=4, fg3m=2, fta=3, ftm=2, oreb=1, dreb=4,
             reb=5, ast=3, tov=2, stl=1, blk=1, pf=2, pts=12, minutes=30,
             win="W", plus_minus=5):
    return {
        "SEASON_ID": "22024", "PLAYER_ID": player, "PLAYER_NAME": name,
        "TEAM_ID": 1, "TEAM_ABBREVIATION": team, "TEAM_NAME": team.title(),
        "MATCHUP": matchup, "GAME_ID": game_id, "GAME_DATE": date_str,
        "WL": win, "MIN": minutes, "FGM": fgm, "FGA": fga, "FG_PCT": fgm / fga,
        "FG3M": fg3m, "FG3A": fg3a, "FG3_PCT": fg3m / fg3a, "FTM": ftm,
        "FTA": fta, "FT_PCT": ftm / fta, "OREB": oreb, "DREB": dreb,
        "REB": reb, "AST": ast, "TOV": tov, "STL": stl, "BLK": blk, "PF": pf,
        "PTS": pts, "PLUS_MINUS": plus_minus,
    }


def _pbp_action(number, kind, team="BOS", period=1, desc="", sub="",
                result="", value=0, distance=0, person_id=1, name="Player",
                home_score=None, away_score=None, action_id=None):
    return {
        "actionId": action_id if action_id is not None else number,
        "actionNumber": number, "period": period, "clock": "PT10M00.00S",
        "teamId": 1610612738, "teamTricode": team, "personId": person_id,
        "playerName": name,
        "playerNameI": (f"{name[0]}. {name}" if name else ""),
        "xLegacy": 0, "yLegacy": 0, "shotDistance": distance,
        "shotResult": result, "isFieldGoal": 1 if kind in ("Made Shot",
                                                            "Missed Shot") else 0,
        "shotValue": value, "pointsTotal": 0, "description": desc,
        "subType": sub, "actionType": kind, "location": "h",
        "videoAvailable": 0, "scoreHome": home_score, "scoreAway": away_score,
    }


def _schedule_row(game_id="401585814", nba_id="0022301200",
                  gameday="2024-01-10", home="BOS", away="NYK",
                  home_score=110.0, away_score=105.0, game_type=1):
    return {"game_id": game_id, "nba_game_id": nba_id, "gameday": gameday,
            "season": 2024, "home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
            "game_type": game_type, "is_final": True}


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


class TestEspnSchedule:
    def test_event_becomes_a_contract_games_row(self):
        frame = src.espn_schedule_frame([_espn_event()])
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row.game_id == "401585814"
        assert (row.home_team, row.away_team) == ("BOS", "WAS")
        assert row.home_score == 132 and row.away_score == 122
        assert row.game_type == config.GAME_TYPE_REG
        assert bool(row.is_final) is True

    def test_a_scheduled_game_has_no_scores(self):
        """A pending game carries a score of '0'; reading it would train on 0-0."""
        frame = src.espn_schedule_frame([_espn_event(
            home_score="0", away_score="0", state="pre", completed=False,
            detail="Tue, October 20th at 3:00 PM EDT",
            when="2026-10-20T19:00Z")])
        row = frame.iloc[0]
        assert pd.isna(row.home_score) and pd.isna(row.away_score)
        assert bool(row.is_final) is False

    def test_a_postponed_game_reported_as_final_is_not_settled(self):
        """ESPN files a postponed game as a completed 0-0. There is no such
        thing as an NBA game that ends 0-0, and accepting one would put a
        scoreless tie into the training set."""
        frame = src.espn_schedule_frame([_espn_event(
            home_score="0", away_score="0", completed=True,
            detail="Postponed")])
        row = frame.iloc[0]
        assert bool(row.is_final) is False
        assert pd.isna(row.home_score) and pd.isna(row.away_score)

    def test_a_zero_zero_final_without_the_word_postponed_is_also_rejected(self):
        frame = src.espn_schedule_frame([_espn_event(
            home_score="0", away_score="0", completed=True, detail="Final")])
        row = frame.iloc[0]
        assert bool(row.is_final) is False
        assert pd.isna(row.home_score)

    def test_a_late_game_keeps_its_eastern_date(self):
        """ESPN's `dates` is a UTC key, so a 10pm ET tipoff arrives under the
        following UTC day. Deciding the day in UTC moves it to the wrong date."""
        frame = src.espn_schedule_frame([_espn_event(
            when="2024-02-16T01:30Z")])
        assert frame.iloc[0].gameday == pd.Timestamp("2024-02-15")

    def test_season_type_maps_to_our_game_types(self):
        assert src.ESPN_GAME_TYPE[src.ESPN_SEASON_REGULAR] == config.GAME_TYPE_REG
        assert src.ESPN_GAME_TYPE[src.ESPN_SEASON_POST] == config.GAME_TYPE_POST

    def test_an_unknown_season_type_yields_no_row(self):
        """The all-star game arrives as `regular-season`. Admitting it would
        put one team's all-star roster into a season aggregate."""
        frame = src.espn_schedule_frame([_espn_event(season_type=1)])
        assert frame.empty

    def test_espn_team_spellings_are_folded_to_ours(self):
        """ESPN says GS/NO/NY/SA/UTAH/WSH where the season log says
        GSW/NOP/NYK/SAS/UTA/WAS. Un-normalized, the join drops every game
        involving five of thirty teams with no error anywhere."""
        for espn, ours in (("GS", "GSW"), ("NO", "NOP"), ("NY", "NYK"),
                           ("SA", "SAS"), ("UTAH", "UTA"), ("WSH", "WAS")):
            assert config.normalize_team_abbr(espn) == ours
        frame = src.espn_schedule_frame([_espn_event(home="NY", away="SA")])
        assert (frame.iloc[0].home_team, frame.iloc[0].away_team) == ("NYK", "SAS")

    def test_duplicate_events_from_overlapping_utc_probes_collapse(self):
        events = [_espn_event(), _espn_event()]
        assert len(src.espn_schedule_frame(events)) == 1

    def test_a_malformed_event_is_skipped_rather_than_guessed(self):
        assert src.espn_schedule_frame([{"id": "x", "competitions": []}]).empty
        assert src.espn_schedule_frame([{"id": "", "competitions": []}]).empty
        assert src.espn_schedule_frame([]).empty


# ---------------------------------------------------------------------------
# Features: the season log joined to the schedule
# ---------------------------------------------------------------------------


class TestSeasonLogJoin:
    def _log(self):
        """One game, both teams, written the way the log writes it: each row's
        matchup is spelled from that row's own team's perspective."""
        rows = [
            _log_row("0022301200", "2024-01-10", "BOS vs. NYK", "BOS",
                     player=1, fga=90, fgm=45, fg3a=40, fg3m=20, fta=25,
                     ftm=20, oreb=10, dreb=30, reb=40, ast=25, tov=12,
                     stl=8, blk=5, pf=18, pts=110),
            _log_row("0022301200", "2024-01-10", "NYK @ BOS", "NYK",
                     player=2, fga=88, fgm=42, fg3a=38, fg3m=18, fta=22,
                     ftm=15, oreb=8, dreb=33, reb=41, ast=22, tov=14,
                     stl=6, blk=4, pf=19, pts=105),
        ]
        return src.rename_log(pd.DataFrame(rows))

    def test_matchup_sides_reads_the_connector(self):
        assert src.matchup_teams("BOS vs. NYK") == ("BOS", "NYK")
        assert src.matchup_teams("NYK @ BOS") == ("NYK", "BOS")
        assert src.matchup_sides("NYK @ BOS") == ("NYK", "BOS")
        assert src.matchup_sides("BOS vs. NYK") == ("NYK", "BOS")
        assert src.matchup_teams("garbage") is None

    def test_both_spellings_of_one_game_reduce_to_one_row(self):
        """The log writes `X @ Y` on the visitors' rows and `Y vs. X` on the
        home team's, so a naive positional read picks one spelling at random and
        inverts the sides of about half the schedule."""
        games = src.games_from_log(self._log())
        assert len(games) == 1
        assert {games.iloc[0].team_a, games.iloc[0].team_b} == {"BOS", "NYK"}

    def test_pair_key_ignores_which_side_is_which(self):
        assert src.pair_key("2024-01-10", "BOS", "NYK") == \
            src.pair_key("2024-01-10", "NYK", "BOS")
        assert src.pair_key("2024-01-10", "BOS", "NYK") != \
            src.pair_key("2024-01-11", "BOS", "NYK")

    def test_the_join_finds_the_nba_id_through_the_orientation_free_key(self):
        log = self._log()
        games = pd.DataFrame([_schedule_row()])
        joined = ing._attach_game_ids(games, log)
        assert joined.iloc[0].nba_game_id == "0022301200"

    def test_the_schedule_supplies_home_away_not_the_log(self):
        """A schedule with swapped venues still looks like a schedule, and it
        would hand Elo's 65-point home advantage to the wrong team."""
        log = self._log()
        games = pd.DataFrame([_schedule_row(home="BOS", away="NYK")])
        joined = ing._attach_game_ids(games, log)
        assert joined.iloc[0].home_team == "BOS"
        assert joined.iloc[0].away_team == "NYK"

    def test_an_unjoined_game_is_reported_not_silently_dropped(self, caplog):
        log = self._log()
        games = pd.DataFrame([_schedule_row(), _schedule_row(
            game_id="401585999", nba_id="", home="LAL", away="MIA")])
        with caplog.at_level("WARNING"):
            joined = ing._attach_game_ids(games, log)
        assert len(joined) == 2
        assert "no season-log match" in caplog.text

    def test_team_facts_sum_the_player_lines(self):
        log = self._log()
        games = pd.DataFrame([_schedule_row()])
        joined = ing._attach_game_ids(games, log)
        team = src.team_stats_from_log(log, joined)
        bos = team[team.team == "BOS"].iloc[0]
        assert bos.fga == 90 and bos.fgm == 45
        assert bos.points_for == 110
        assert bos.points_against == 105
        assert bool(bos.is_home) is True

    def test_team_facts_recompute_derived_percentages(self):
        log = self._log()
        games = pd.DataFrame([_schedule_row()])
        team = src.team_stats_from_log(log, ing._attach_game_ids(games, log))
        bos = team[team.team == "BOS"].iloc[0]
        assert bos.efg_pct == pytest.approx((45 + 0.5 * 20) / 90)
        assert bos.net_points == 5
        assert bos.fg_pct == pytest.approx(0.5)

    def test_every_counting_column_is_renamed(self):
        """The log says PTS/FGA/FGM... and the contract says points/fga/fgm. A
        rollup that sums the upstream name produces a column the contract does
        not declare and the ladder never reads."""
        log = self._log()
        for canonical in ("points", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
                          "oreb", "dreb", "reb", "ast", "tov", "stl", "blk",
                          "pf", "minutes"):
            assert canonical in log.columns, canonical
        for upstream in ("PTS", "FGA", "FGM", "FG3A", "AST", "MIN"):
            assert upstream not in log.columns, upstream

    def test_log_rows_with_no_schedule_game_are_dropped(self):
        """A season log returns whole seasons, so a short window leaves most
        rows unmatched. Kept with an empty id they all share one game_id, which
        no duplicate check can tell from thousands of real duplicates."""
        log = self._log()
        games = pd.DataFrame([_schedule_row()])
        with_ids = ing._with_contract_ids(log, ing._attach_game_ids(games, log))
        assert len(with_ids) == 2
        assert (with_ids.game_id == "401585814").all()

    def test_contract_ids_are_attached_from_the_schedule(self):
        log = self._log()
        games = pd.DataFrame([_schedule_row()])
        with_ids = ing._with_contract_ids(log, ing._attach_game_ids(games, log))
        assert set(with_ids.game_id) == {"401585814"}


# ---------------------------------------------------------------------------
# Play-by-play
# ---------------------------------------------------------------------------


class TestPlayByPlayParsing:
    def test_the_response_is_not_a_resultset_envelope(self):
        """playbyplayv3 returns {game: {actions: [...]}}. A client that looks
        for `resultSets` finds none and reports zero actions for a game that
        returned 569 of them."""
        payload = {"game": {"gameId": "0022301200",
                            "actions": [_pbp_action(1, "Made Shot")]}}
        actions = src.play_by_play_actions(payload, "0022301200", "2024-01-10")
        assert len(actions) == 1
        assert actions.iloc[0].action_id == 1

    def test_an_empty_action_list_is_an_empty_frame(self):
        assert src.play_by_play_actions({"game": {}}, "x", "2024-01-10").empty
        assert src.play_by_play_actions(None, "x", "2024-01-10").empty

    def test_the_period_range_is_mandatory(self):
        """playbyplayv3 with no StartPeriod/EndPeriod answers HTTP 500, which a
        wrapper that forgets them reports as an upstream outage."""
        query = src.play_by_play_query("0022301200")
        assert "StartPeriod=0" in query and "EndPeriod=14" in query
        assert "GameID=0022301200" in query

    def test_actions_carry_the_fetch_key_not_the_contract_id(self):
        payload = {"game": {"actions": [_pbp_action(1, "Made Shot")]}}
        actions = src.play_by_play_actions(payload, "0022301200", "2024-01-10")
        assert "nba_game_id" in actions.columns
        assert actions.iloc[0].nba_game_id == "0022301200"


def _full_game_pbp():
    """A small but complete game, with the field encodings the live feed uses."""
    acts = [
        _pbp_action(2, "period", team="", sub="start"),
        _pbp_action(3, "Made Shot", team="BOS", result="Made", value=2,
                    distance=1, sub="Cutting Layup Shot",
                    desc="Player 1' Cutting Layup Shot (2 PTS) (Helper 1 AST)"),
        _pbp_action(4, "Rebound", team="NYK", sub="Unknown",
                    desc="Player REBOUND (Off:0 Def:1)"),
        _pbp_action(5, "Missed Shot", team="BOS", result="Missed", value=3,
                    distance=25, sub="Jump Shot", desc="MISS Player 25' 3PT"),
        _pbp_action(6, "Rebound", team="BOS", sub="Unknown",
                    desc="Player REBOUND (Off:1 Def:1)"),
        _pbp_action(7, "Foul", team="BOS", sub="Shooting",
                    desc="Helper S.FOUL (P1.T1)"),
        _pbp_action(8, "Free Throw", team="NYK", sub="Free Throw 1 of 1",
                    desc="Player Free Throw 1 of 1 (1 PTS)"),
        _pbp_action(9, "Turnover", team="NYK", sub="Bad Pass",
                    desc="Player Bad Pass Turnover (P1.T2)"),
        _pbp_action(10, "Free Throw", team="BOS", sub="Free Throw 1 of 2",
                    desc="MISS Player Free Throw 1 of 2"),
        _pbp_action(11, "Free Throw", team="BOS", sub="Free Throw 2 of 2",
                    desc="Player Free Throw 2 of 2 (2 PTS)"),
        _pbp_action(12, "Made Shot", team="NYK", result="Made", value=3,
                    distance=24, sub="Jump Shot",
                    desc="Player 24' 3PT Jump Shot (3 PTS)"),
        _pbp_action(13, "Foul", team="NYK", sub="Personal",
                    desc="Helper P.FOUL (P2.T1)"),
        _pbp_action(14, "period", team="", sub="start"),
        # A team rebound: the feed sends an empty player AND an empty team, so
        # there is nothing on the row to attribute it to.
        _pbp_action(15, "Rebound", team="", sub="Unknown",
                    desc="Hawks Rebound", name="", person_id=0),
    ]
    return {"game": {"gameId": "0022301200", "actions": acts}}


class TestPlayByPlayRollup:
    def _rollup(self, actions=None):
        payload = actions or _full_game_pbp()
        acts = src.play_by_play_actions(payload, "0022301200", "2024-01-10")
        games = pd.DataFrame([_schedule_row(home="BOS", away="NYK")])
        return src.team_events_from_actions(acts, games)

    def test_the_shot_bands_partition_the_attempts(self):
        events = self._rollup().set_index("team")
        for team, row in events.iterrows():
            bands = (row.rim_attempts + row.mid_attempts
                     + row.corner_three_attempts)
            assert bands == row.fga, team

    def test_a_made_free_throw_is_identified_without_shot_result(self):
        """shotResult is empty on every free throw this feed returns, and the
        trailing (N PTS) is cumulative across a trip, not the points on the
        shot. Only the absence of a leading MISS identifies a make."""
        events = self._rollup().set_index("team")
        assert events.loc["BOS", "fta"] == 2
        assert events.loc["BOS", "ftm"] == 1
        assert events.loc["NYK", "fta"] == 1
        assert events.loc["NYK", "ftm"] == 1

    def test_rebound_sides_come_from_the_running_counters(self):
        """`Off:N Def:N` is the rebounding PLAYER's running total, so a team
        total is the sum of each player's maximum - not the team maximum, and
        not the sum of the readings."""
        events = self._rollup().set_index("team")
        assert events.loc["BOS", "oreb"] == 1
        assert events.loc["BOS", "dreb"] == 1
        assert events.loc["NYK", "dreb"] == 1

    def test_a_team_rebound_is_counted_but_not_attributed(self):
        """The feed writes `Hawks Rebound` with no player and no team, so there
        is nothing to key it on. It must not silently become a defensive
        rebound for whichever side happened to be listed."""
        payload = _full_game_pbp()
        acts = src.play_by_play_actions(payload, "0022301200", "2024-01-10")
        assert src.unattributed_rebounds(acts) == 1
        events = self._rollup(payload).set_index("team")
        assert events.dreb.sum() == 2

    def test_an_and_one_is_found_across_the_two_teams(self):
        """A shooting foul is charged to the team defending and the free throw
        to the team attacking, so within one team's actions the pair is never
        adjacent. Grouping by team finds no and-ones at all."""
        events = self._rollup().set_index("team")
        assert events.loc["NYK", "and_in"] == 1
        assert events.loc["BOS", "and_in"] == 0

    def test_technical_fouls_are_not_player_fouls(self):
        events = self._rollup().set_index("team")
        assert events.loc["BOS", "pf"] == 1
        assert events.loc["NYK", "pf"] == 1

    def test_live_turnovers_are_a_subset_of_turnovers(self):
        events = self._rollup().set_index("team")
        for _, row in events.iterrows():
            assert row.live_turnovers <= row.tov

    def test_assists_are_counted_but_not_attributed_to_an_id(self):
        """The assister is named only in the description; the action's own
        personId is the shooter."""
        events = self._rollup().set_index("team")
        assert events.loc["BOS", "assists"] == 1
        assert events.loc["NYK", "assists"] == 0

    def test_quarter_points_sum_to_the_game_total(self):
        events = self._rollup()
        quarters = events[[f"q{q}_points" for q in (1, 2, 3, 4)]].sum(axis=1)
        quarters = quarters + events.ot_points
        assert np.allclose(quarters, events.points)

    def test_possessions_are_plausible(self):
        events = self._rollup().set_index("team")
        for _, row in events.iterrows():
            assert row.possessions > 0
            assert row.possessions < 200

    def test_an_empty_action_list_rolls_up_to_nothing(self):
        assert self._rollup({"game": {"actions": []}}).empty
        assert src.team_events_from_actions(pd.DataFrame(),
                                            pd.DataFrame()).empty


class TestCrossCheck:
    def test_the_rollup_and_the_box_score_agree(self):
        """Two independent counts of the same game that agree are evidence the
        pull is right. This is the only cross-check available."""
        log_rows = [
            _log_row("0022301200", "2024-01-10", "BOS vs. NYK", "BOS",
                     fga=2, fgm=1, fg3a=1, fg3m=0, fta=2, ftm=1, oreb=1,
                     dreb=1, reb=2, ast=1, tov=0, pf=1, pts=3),
            _log_row("0022301200", "2024-01-10", "NYK @ BOS", "NYK",
                     player=2, fga=1, fgm=1, fg3a=1, fg3m=1, fta=1, ftm=1,
                     oreb=0, dreb=1, reb=1, ast=0, tov=1, pf=1, pts=4),
        ]
        log = src.rename_log(pd.DataFrame(log_rows))
        games = pd.DataFrame([_schedule_row(home="BOS", away="NYK")])
        joined = ing._attach_game_ids(games, log)
        team_stats = contract.normalize(src.team_stats_from_log(log, joined),
                                        "team_stats")
        acts = src.play_by_play_actions(_full_game_pbp(), "0022301200",
                                        "2024-01-10")
        events = contract.normalize(
            src.team_events_from_actions(acts, games), "team_events")
        report = src.cross_check(events, team_stats)
        assert report, "the cross-check compared nothing"
        for column in ("fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "tov", "pf",
                       "oreb", "dreb", "points"):
            assert report[column]["max_abs_diff"] == 0, column

    def test_a_disagreement_is_reported_rather_than_ignored(self):
        report = src.cross_check(
            pd.DataFrame([{"game_id": "g", "team": "BOS", "fga": 90.0}]),
            pd.DataFrame([{"game_id": "g", "team": "BOS", "fga": 88.0}]))
        assert report["fga"]["exact"] == 0
        assert report["fga"]["max_abs_diff"] == 2.0


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class TestContract:
    def test_a_missing_required_column_is_named(self):
        with pytest.raises(contract.ContractError) as excinfo:
            contract.normalize(pd.DataFrame({"game_id": ["g"], "team": ["BOS"]}),
                               "team_stats")
        assert "points_for" in str(excinfo.value)

    def test_an_empty_frame_is_a_gap_not_a_contract_error(self):
        """A source that answered with no rows is a gap in the data; one that
        answered with rows and dropped a column is a bug in the adapter. They
        must not share an exit path."""
        out = contract.normalize(pd.DataFrame(), "team_stats",
                                 allow_missing=True)
        assert out.empty
        assert "points_for" in out.columns

    def test_a_missing_string_becomes_empty_not_the_word_nan(self):
        """nba_game_id and every abbreviation flow into a join key, and a join
        key of 'nan' matches nothing while looking perfectly populated."""
        out = contract.normalize(
            pd.DataFrame({"game_id": ["g"], "gameday": ["2024-01-10"],
                          "home_team": ["BOS"], "away_team": ["NYK"],
                          "home_score": [1.0], "away_score": [1.0],
                          "nba_game_id": [np.nan]}), "games")
        assert out.iloc[0].nba_game_id == ""

    def test_a_missing_flag_is_false_not_true(self):
        out = contract.normalize(
            pd.DataFrame({"game_id": ["g"], "gameday": ["2024-01-10"],
                          "home_team": ["BOS"], "away_team": ["NYK"],
                          "home_score": [1.0], "away_score": [1.0],
                          "is_final": [np.nan]}), "games")
        assert bool(out.iloc[0].is_final) is False

    def test_derived_columns_are_recomputed_not_taken_from_a_source(self):
        out = contract.normalize(
            pd.DataFrame({"game_id": ["g"], "gameday": ["2024-01-10"],
                          "home_team": ["BOS"], "away_team": ["NYK"],
                          "home_score": [110.0], "away_score": [105.0],
                          "margin": [999.0], "total": [999.0],
                          "home_win": [0.0]}), "games")
        assert out.iloc[0].margin == 5
        assert out.iloc[0].total == 215
        assert out.iloc[0].home_win == 1.0

    def test_season_is_derived_from_the_tipoff(self):
        out = contract.normalize(
            pd.DataFrame({"game_id": ["a", "b"], "gameday": ["2024-10-22",
                                                            "2024-01-10"],
                          "home_team": ["BOS", "BOS"], "away_team": ["NYK",
                                                                   "NYK"],
                          "home_score": [1.0, 1.0], "away_score": [1.0, 1.0]}),
            "games")
        assert list(out.season) == [2024, 2023]

    def test_the_event_rollup_is_a_declared_frame(self):
        assert "team_events" in contract.SCHEMAS
        for column in ("possessions", "rim_attempts", "live_turnovers",
                       "and_in", "shot_distance", "q4_points", "ot_points"):
            assert column in contract.TEAM_EVENTS_SCHEMA, column

    def test_there_is_no_team_rebound_column(self):
        """A per-team column for rebounds the feed attributes to nobody would be
        structurally always zero - populated by column count, carrying nothing."""
        assert "team_rebounds" not in contract.TEAM_EVENTS_SCHEMA

    def test_every_frame_has_a_declared_source(self):
        assert set(contract.SCHEMAS) == set(src.FRAME_SOURCE)
        for name, spec in src.FRAME_SOURCE.items():
            assert spec.can(name), spec.name

    def test_coverage_report_shows_what_a_frame_filled(self):
        frame = contract.normalize(
            pd.DataFrame({"game_id": ["g"], "gameday": ["2024-01-10"],
                          "home_team": ["BOS"], "away_team": ["NYK"],
                          "home_score": [1.0], "away_score": [1.0]}), "games")
        report = contract.coverage_report(frame, "games").set_index("column")
        assert report.loc["game_id", "coverage_pct"] == 100.0
        assert report.loc["nba_game_id", "coverage_pct"] == 0.0

    def test_a_column_no_source_can_supply_is_not_declared(self):
        """2026-10-08 coverage audit, low finding: ``team_stats``.

        ``ast_per_game`` was declared while the season log has no per-game
        assist column, so ``normalize`` manufactured an all-NaN SOURCE
        column every run - a naming artifact a coverage report had to keep
        explaining away. The schema must not declare it, the normalized
        frame must not carry it, and the served feature must still arrive
        (derived from ``ast`` in exactly one place, ``_attach_stats``).
        """
        assert "ast_per_game" not in contract.TEAM_STATS_SCHEMA
        frame = contract.normalize(
            pd.DataFrame({"game_id": ["g1"], "team": ["BOS"],
                          "points_for": [110.0], "points_against": [105.0],
                          "ast": [24.0]}), "team_stats", source="test")
        assert "ast_per_game" not in frame.columns
        events = pd.DataFrame({"game_id": ["g1"], "team": ["BOS"],
                               "for": [110.0], "against": [105.0]})
        out = feat._attach_stats(events, frame)
        # The served name still exists and still reads the player-summed
        # assists - dropping a dead SOURCE column must not touch the model.
        assert out["ast_per_game"].iloc[0] == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# Feature wiring
# ---------------------------------------------------------------------------


class TestEventFeatures:
    def _ladder_rows(self):
        games = []
        for day in range(1, 9):
            games.append({"game_id": f"g{day}", "season": 2024,
                          "gameday": f"2024-01-{day:02d}", "game_type": 1,
                          "home_team": "BOS", "away_team": "NYK",
                          "home_score": 110.0, "away_score": 100.0})
        return pd.DataFrame(games)

    def _event_stats(self, **overrides):
        rows = []
        for day in range(1, 9):
            for team, rim in (("BOS", 40), ("NYK", 25)):
                row = {"game_id": f"g{day}", "team": team, "fga": 90,
                       "possessions": 100.0, "rim_attempts": rim,
                       "mid_attempts": 25, "corner_three_attempts": 25,
                       "live_turnovers": 10, "and_in": 1, "shooting_fouls": 18,
                       "shot_distance": 13.5, "q4_points": 25, "ot_points": 0.0}
                row.update(overrides)
                rows.append(row)
        return pd.DataFrame(rows)

    def test_the_event_features_reach_the_model_frame(self):
        built = feat.build_game_features(self._ladder_rows(), None,
                                         self._event_stats())
        for column in config.EVENT_TRAILING_SPECS:
            assert f"event_{column}_diff" in built.columns, column
            assert f"event_{column}_diff" in config.MONEYLINE_FEATURE_COLS

    def test_the_event_features_have_values(self):
        built = feat.build_game_features(self._ladder_rows(), None,
                                         self._event_stats())
        assert built["event_rim_rate_diff"].notna().any()
        assert built["event_possessions_diff"].notna().any()

    def test_a_rim_heavier_team_reads_higher(self):
        built = feat.build_game_features(self._ladder_rows(), None,
                                         self._event_stats())
        # Boston takes 40 rim attempts to New York's 25 on equal volume.
        assert built["event_rim_rate_diff"].dropna().gt(0).all()

    def test_an_event_feature_is_a_difference_not_a_level(self):
        """A level feature would let the model learn a team's standing rather
        than the matchup, which is not what the contract is for."""
        built = feat.build_game_features(self._ladder_rows(), None,
                                         self._event_stats())
        column = "event_possessions_diff"
        assert built[column].abs().max() < 40

    def test_no_event_data_still_builds_every_other_feature(self):
        built = feat.build_game_features(self._ladder_rows(), None, None)
        for column in config.EVENT_TRAILING_SPECS:
            assert f"event_{column}_diff" in built.columns, column
        assert built["elo_diff"].notna().any()

    def test_a_missing_mean_is_nan_rather_than_zero(self):
        """Defaulting a mean to zero would make a team with no play-by-play look
        like the league's best close-range team."""
        built = feat.build_game_features(self._ladder_rows(), None, None)
        assert built["event_shot_distance_diff"].isna().all()

    def test_an_event_profile_carries_forward_across_missing_games(self):
        """The sweep is incremental, so most of a run's history has no
        play-by-play. An EWM over a column with interior NaNs propagates them,
        leaving every event feature empty."""
        events = self._event_stats()
        keep = events[events.game_id.isin(["g1", "g8"])]
        built = feat.build_game_features(self._ladder_rows(), None, keep)
        assert built["event_rim_rate_diff"].notna().any()

    def test_rates_are_averaged_not_averaged_after(self):
        """A rolling mean of a rate must be a mean of rates, so the rate is
        computed on the per-game value before any window is applied."""
        events = self._event_stats()
        events["possessions"] = 100.0
        ladder = feat.team_stats_ladder(feat.compute_elo(
            feat.team_events(self._ladder_rows())), None, events)
        bos = ladder[(ladder.team == "BOS") & ladder.fga.notna()]
        assert bos["rim_rate"].between(0, 1).all()

    def test_the_slate_builds_event_features_too(self):
        games = self._ladder_rows()
        slate_game = {"game_id": "g9", "season": 2024,
                      "gameday": "2024-01-09", "game_type": 1,
                      "home_team": "BOS", "away_team": "LAL",
                      "home_score": np.nan, "away_score": np.nan}
        combined = pd.concat([games, pd.DataFrame([slate_game])],
                             ignore_index=True)
        slate = feat.build_slate_features(combined, None, self._event_stats())
        assert len(slate) == 1
        for column in config.EVENT_TRAILING_SPECS:
            assert f"event_{column}_diff" in slate.columns, column

    def test_every_per_side_feature_is_published_with_its_diff(self):
        """The raw home and away values behind each diff are part of the
        contract, the way elo_home/elo_away have always been - and each pair
        is built from the SAME ladder column its diff reads, so the identity
        diff == home - away holds by construction rather than by luck."""
        built = feat.build_game_features(self._ladder_rows(), None,
                                         self._event_stats())
        for ladder_col, stem in config.PER_SIDE_SOURCES.items():
            for side in ("home", "away"):
                column = f"{stem}_{side}"
                assert column in built.columns, column
                assert column in config.MONEYLINE_FEATURE_COLS, column
            ident = (built[f"{stem}_home"] - built[f"{stem}_away"]
                     - built[f"{stem}_diff"]).abs()
            assert not (ident.dropna() > 1e-9).any(), stem

    def test_a_raw_side_is_excluded_from_the_linear_view(self):
        """A raw side value is exactly the level feature the diff form exists
        to avoid: the linear member trains on differences only, and the tree
        members are the ones that may see the sides."""
        linear = feat.linear_feature_columns()
        assert not set(config.PER_SIDE_FEATURE_COLS) & set(linear)
        # The pre-existing per-side columns stay excluded too - the new
        # families changed nothing about that rule.
        assert "elo_home" not in linear and "elo_away" not in linear


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class TestInputSemanticParity:
    def test_existing_smoke_fails_coverage_and_builds_rapm(self, monkeypatch):
        import smoke_nba as smoke
        import master_pipeline as mp
        from types import SimpleNamespace
        frame = TestEventFeatures()._ladder_rows()
        facts = SimpleNamespace(games=frame, team_stats=pd.DataFrame(),
            team_events=pd.DataFrame(), player_stats=pd.DataFrame(),
            play_by_play=pd.DataFrame(), manifest={})
        monkeypatch.setattr(smoke.ing, "load_ingested", lambda: facts)
        monkeypatch.setattr(smoke.ing, "eligible_games", lambda f: f)
        monkeypatch.setattr(smoke.ing, "trainable_games", lambda f: f)
        monkeypatch.setattr(smoke, "declared_features", lambda: ["elo_diff", "ewm_pace_diff"])
        calls = []
        monkeypatch.setattr(mp, "_build_position_rapm_features",
                            lambda *a: (calls.append(True) or None, None))
        assert smoke.run(95, 0, None) == 1
        assert calls == [True]
        assert smoke.coverage(pd.DataFrame({"x": [np.inf]}), "x") == 0

    def test_positive_home_advantage_and_conserved_elo(self):
        games = TestEventFeatures()._ladder_rows().iloc[:1]
        _, ratings = feat._elo_apply(feat.team_events(games))
        expected = 1 / (1 + 10 ** (-config.ELO_HOME_ADV / config.ELO_SCALE))
        assert expected > .5
        assert ratings["BOS"] == pytest.approx(config.ELO_PRIOR + config.ELO_K * (1 - expected))
        assert sum(ratings.values()) == pytest.approx(2 * config.ELO_PRIOR)

    def test_efficiencies_use_paired_possessions_and_ot_duration(self):
        games = pd.DataFrame([dict(nba_game_id="002", game_id="g", gameday="2024-01-01",
                    home_team="BOS", away_team="NYK", home_score=110., away_score=100.)])
        log = pd.DataFrame([dict(nba_game_id="002", team=t, points=p, fga=90.,
                    fta=20., oreb=10., tov=12., minutes=265., fgm=40., fg3m=10.)
                    for t, p in [("BOS", 110.), ("NYK", 100.)]])
        stats = contract.normalize(src.team_stats_from_log(log, games),
                                   "team_stats").set_index("team")
        poss = 90 + .44*20 - 10 + 12
        assert stats.loc["BOS", "off_rating"] == pytest.approx(11000/poss)
        assert stats.loc["BOS", "def_rating"] == pytest.approx(10000/poss)
        assert stats.loc["BOS", "pace"] == pytest.approx(48*poss/53)
        assert stats.loc["NYK", "off_rating"] == stats.loc["BOS", "def_rating"]
        log.loc[0, "fta"] = np.nan
        missing = src.team_stats_from_log(log, games)
        assert missing.off_rating.isna().all() and missing.pace.isna().all()

    def test_missing_facts_never_become_measurements(self):
        games = TestEventFeatures()._ladder_rows()
        frame = feat.build_game_features(games)
        for stem in ("ewm_off_rating", "ewm_def_rating", "ewm_pace", "ewm_efg_pct",
                     "ewm_ast_per_game", "event_shooting_fouls", "event_q4_points"):
            assert frame[f"{stem}_diff"].isna().all(), stem

    def test_pending_games_do_not_advance_form_or_rest(self):
        fixture = TestEventFeatures()
        games = fixture._ladder_rows()
        games.loc[games.index[-3:], ["home_score", "away_score"]] = np.nan
        target = games.iloc[-1].game_id
        full = feat.build_slate_features(games, None, fixture._event_stats()).set_index("game_id")
        reduced = feat.build_slate_features(games.drop(index=games.index[-3:-1]),
                    None, fixture._event_stats()).set_index("game_id")
        cols = config.MONEYLINE_FEATURE_COLS
        pd.testing.assert_series_equal(full.loc[target, cols], reduced.loc[target, cols])
        assert full.loc[target, "rest_days_home"] == 3

    def test_slate_elo_reverts_once_and_cannot_read_a_later_final(self):
        games = TestEventFeatures()._ladder_rows().iloc[:3].copy()
        games.loc[1, ["home_score", "away_score"]] = np.nan
        games.loc[1:, "season"] = 2025
        games.loc[1, "gameday"] = "2025-01-02"
        games.loc[2, "gameday"] = "2025-01-03"
        full = feat.build_slate_features(games).iloc[0]
        truncated = feat.build_slate_features(games.iloc[:2]).iloc[0]
        assert full.elo_home == truncated.elo_home
        _, ratings = feat._elo_apply(feat.team_events(games.iloc[:1]))
        assert full.elo_home == pytest.approx(ratings["BOS"] + config.ELO_REVERT_FACTOR *
                                              (config.ELO_PRIOR - ratings["BOS"]))

    def test_coverage_gate_detects_absent_nan_and_infinite_columns(self, monkeypatch):
        import master_pipeline as mp
        monkeypatch.setattr(mp.config, "_FEATURE_SUBSET", ["elo_diff", "win_pct_diff"])
        assert mp._empty_contract_columns(pd.DataFrame({"elo_diff": [1.]})) == ["win_pct_diff"]
        assert mp._empty_contract_columns(pd.DataFrame({"elo_diff": [np.inf],
                                       "win_pct_diff": [np.nan]})) == ["elo_diff", "win_pct_diff"]
        assert "NBA training feature coverage failed" in __import__("inspect").getsource(mp.run)


class TestHttp:
    def test_a_4xx_is_not_retried(self, monkeypatch):
        """A 400 means the request is wrong. Repeating it unchanged wastes a
        budget and delays the failure that would have explained it."""
        calls = []

        def boom(request, timeout=None):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url, 400, "Bad Request", {}, None)

        monkeypatch.setattr(ing.urllib.request, "urlopen", boom)
        with pytest.raises(ing.HostUnavailable) as excinfo:
            ing.http_json("https://example.test/x", {}, attempts=3)
        assert len(calls) == 1
        assert "400" in str(excinfo.value)

    def test_a_5xx_is_retried_then_reported(self, monkeypatch):
        calls = []

        def boom(request, timeout=None):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url, 500, "Internal Server Error", {}, None)

        monkeypatch.setattr(ing.urllib.request, "urlopen", boom)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        with pytest.raises(ing.HostUnavailable):
            ing.http_json("https://example.test/x", {}, attempts=3)
        assert len(calls) == 3

    def test_the_season_log_pull_uses_the_deep_retry_budget(self):
        """The LeagueGameLog pull has no second source - a missing slice is a
        stopped run - so it must retry deeper than the default 3. The
        2026-09-29 Kaggle run died because one slice's 3x90s (~4.7 min)
        expired during a transient brownout; the deeper budget rides those
        out at a bounded ~15 minute ceiling per slice."""
        import ast
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent
               / "ingestion.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "http_json"]
        season_log = [n for n in calls
                      if any(kw.arg == "attempts"
                             and getattr(kw.value, "id", "")
                             == "SEASON_LOG_ATTEMPTS"
                             for kw in n.keywords)]
        assert len(season_log) == 1, \
            "the season-log call site must use SEASON_LOG_ATTEMPTS"
        default_attempts = [n for n in calls
                            if any(kw.arg == "attempts"
                                   and getattr(kw.value, "id", "")
                                   == "DEFAULT_ATTEMPTS"
                                   for kw in n.keywords)]
        assert not default_attempts, \
            "no call site should pass the default explicitly"

    def test_the_deep_budget_is_actually_nine_attempts(self, monkeypatch):
        calls = []

        def boom(request, timeout=None):
            calls.append(request.full_url)
            raise TimeoutError("The read operation timed out")

        monkeypatch.setattr(ing.urllib.request, "urlopen", boom)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        with pytest.raises(ing.HostUnavailable) as excinfo:
            ing.http_json("https://example.test/x", {},
                          attempts=ing.SEASON_LOG_ATTEMPTS)
        assert len(calls) == 9
        assert "9 attempt(s)" in str(excinfo.value)

    def test_the_play_by_play_pull_retries_past_a_flaky_gateway(self):
        """The play-by-play sweep pins its own retry budget at three.

        It has no second source either, and two attempts demonstrably lost:
        the 2026-10-02 run hit a stats.nba.com 502 burst and three games
        (0022501121/1131/1133) failed the sweep while retries on neighbouring
        games succeeded seconds later. The literal is pinned here so a future
        DEFAULT_ATTEMPTS change cannot quietly thin the sweep back down.
        """
        import ast
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent
               / "ingestion.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_fetch_play_by_play")
        calls = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "http_json"]
        assert len(calls) == 1, \
            "the sweep has exactly one http_json call site"
        kw = {k.arg: k.value for k in calls[0].keywords}
        attempts = kw.get("attempts")
        assert (isinstance(attempts, ast.Constant)
                and isinstance(attempts.value, int)
                and attempts.value >= 3), (
            "the play-by-play sweep must retry at least three times; two "
            "attempts lost three games to a transient 502 burst on "
            "2026-10-02")

    def test_a_timeout_is_retried(self, monkeypatch):
        calls = []

        def boom(request, timeout=None):
            calls.append(request.full_url)
            raise TimeoutError("The read operation timed out")

        monkeypatch.setattr(ing.urllib.request, "urlopen", boom)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        with pytest.raises(ing.HostUnavailable) as excinfo:
            ing.http_json("https://example.test/x", {}, attempts=2)
        assert len(calls) == 2
        assert "timed out" in str(excinfo.value)

    def test_a_gzipped_body_is_decoded(self):
        raw = gzip_bytes = __import__("gzip").compress(b'{"a": 1}')
        assert json.loads(ing._decompress(gzip_bytes, "gzip")) == {"a": 1}
        assert ing._decompress(raw, None) == raw

    def test_the_user_agent_is_not_a_browser_impersonation(self):
        """A full Chrome agent is reproducibly refused by ESPN with an
        immediate 403 (5 runs out of 5), so the agent must not look like a
        browser.

        This test previously also asserted that stats.nba.com tarpits a
        browser agent. That was measured once during the rewrite and does not
        reproduce - the endpoint now answers 4.5 MB in about 3s with the full
        browser header set - so the claim was dropped rather than carried
        forward as fact. See the comment above ``USER_AGENT``.
        """
        assert "Chrome" not in ing.USER_AGENT
        assert "Safari" not in ing.USER_AGENT
        assert "Mozilla" not in ing.USER_AGENT
        for headers in (ing.STATS_HEADERS, ing.ESPN_HEADERS):
            assert headers["User-Agent"] == ing.USER_AGENT

    def test_espn_and_stats_nba_get_their_own_headers(self):
        assert "x-nba-stats-token" in ing.STATS_HEADERS
        assert "x-nba-stats-token" not in ing.ESPN_HEADERS


# ---------------------------------------------------------------------------
# Games that are not league games
# ---------------------------------------------------------------------------


class TestNonFranchiseGames:
    """The all-star event, which is not a game and used to stop the run.

    ESPN files it as ``regular-season``, so the season-type check cannot see it.
    In 2026 the all-star became a four-game tournament on a single date and
    played the same two squads twice, which made the date/team-pair join key
    ambiguous and raised ``MergeError`` for the whole pipeline. The participants
    are the tell: a league game is played by two franchises.
    """

    @staticmethod
    def _tournament() -> list[dict]:
        return [_espn_event(event_id="401838140", home="STARS",
                            away="WORLD", when="2026-02-15T19:00Z"),
                _espn_event(event_id="401838141", home="STRIPES",
                            away="STARS", when="2026-02-15T19:00Z"),
                _espn_event(event_id="401838142", home="STRIPES",
                            away="WORLD", when="2026-02-15T19:00Z"),
                _espn_event(event_id="401838143", home="STRIPES",
                            away="STARS", when="2026-02-15T19:00Z")]

    def test_the_tournament_is_dropped_entirely(self):
        assert src.espn_schedule_frame(self._tournament()).empty

    def test_a_real_game_on_the_same_date_survives(self):
        events = self._tournament() + [
            _espn_event(event_id="401585814", home="BOS", away="LAL",
                        when="2026-02-15T19:00Z")]
        frame = src.espn_schedule_frame(events)
        assert list(frame.game_id) == ["401585814"]

    def test_espn_aliases_are_still_recognised_as_franchises(self):
        """The franchise test goes through the alias table, so a team ESPN
        spells differently must not be mistaken for a non-franchise."""
        for espn_name in ("GS", "NO", "NY", "SA", "UTAH", "WSH"):
            event = _espn_event(event_id="401585815", home=espn_name,
                                away="BOS", when="2024-02-15T19:00Z")
            assert not src.espn_schedule_frame([event]).empty, espn_name

    def test_a_duplicate_join_key_drops_the_extra_game_instead_of_raising(
            self, monkeypatch):
        """The key is unique only because a team plays at most once a day. That
        is an assumption about the world, so a collision is resolved rather than
        allowed to fail the run."""
        games = pd.DataFrame({
            "game_id": ["401838141", "401838143", "401585814"],
            "gameday": [pd.Timestamp("2026-02-15")] * 3,
            "home_team": ["STRIPES", "STRIPES", "BOS"],
            "away_team": ["STARS", "STARS", "LAL"],
        })
        key = src.pair_key(pd.Timestamp("2026-02-15"), "STRIPES", "STARS")
        monkeypatch.setattr(
            ing.sources, "games_from_log",
            lambda _log: pd.DataFrame({"pair_key": [key],
                                       "nba_game_id": ["0022500001"]}))
        out = ing._attach_game_ids(games, pd.DataFrame({"nba_game_id": ["1"]}))
        assert len(out) == 2
        assert set(out.game_id) == {"401838141", "401585814"}


class TestPlayerFoulRule:
    """Which foul sub-types the box score charges to a player.

    The list was measured, not reasoned: 19 sub-types appear in the feed, 5
    were counted, and the rollup then agreed with the box score on 62-66% of
    team-games. These pin the sub-types that measurement put in, and - just as
    importantly - the one that measurement put out.
    """

    def test_the_measured_subtypes_are_counted(self):
        for sub_type in ("Personal", "Shooting", "Offensive",
                         "Offensive Charge", "Loose Ball",
                         "Personal Take", "Away From Play", "Flagrant Type 1"):
            assert sub_type in src.PLAYER_FOUL_SUBTYPES, sub_type

    def test_personal_take_is_the_bulk_of_the_shortfall(self):
        """Excluding this one alone held agreement to 80% rather than 92%."""
        assert "Personal Take" in src.PLAYER_FOUL_SUBTYPES

    def test_a_team_foul_is_not_a_player_foul(self):
        """``Defense 3 Second`` is charged to the team and appears in no
        player's PF. Counting it made agreement worse (55.5%), which is how it
        was identified as a team foul rather than guessed at."""
        assert "Defense 3 Second" not in src.PLAYER_FOUL_SUBTYPES

    @pytest.mark.parametrize("sub_type", [
        "Technical", "Double Technical", "Delay Technical",
        "Excess Timeout Technical", "Too Many Players Technical", "Flopping"])
    def test_technicals_appear_in_no_player_pf(self, sub_type):
        assert sub_type not in src.PLAYER_FOUL_SUBTYPES

    def test_a_take_foul_counts_once_in_the_rollup(self):
        payload = {"game": {"actions": [
            {"actionId": 1, "actionNumber": 1, "period": 1, "clock": "12:00",
             "teamId": 1610612737, "teamTricode": "BOS",
             "personId": 1628369, "playerName": "P. Player", "actionType": "Foul",
             "subType": "Personal Take", "description": "Take foul",
             "scoreHome": 0, "scoreAway": 0},
        ]}}
        frame = src.play_by_play_actions(payload, "0022400001", "2024-10-22")
        rollup = src.team_events_from_actions(frame, pd.DataFrame(
            {"game_id": ["g1"], "nba_game_id": ["0022400001"],
             "gameday": ["2024-10-22"], "home_team": ["BOS"],
             "away_team": ["LAL"], "home_score": [110], "away_score": [104]}))
        assert float(rollup.pf.iloc[0]) == 1.0


class TestDeliverySync:
    """The delivery phase, with git mocked out.

    No test here touches the network. The push path is exercised against fakes
    so that a regression in the retry or verification logic is caught without a
    test run being able to write to the repository.
    """

    def test_no_token_skips_and_says_why(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("NBA_PUSH", raising=False)
        import master_pipeline as mp
        monkeypatch.setattr(mp.config, "DATA_DELIVERY_DIR", tmp_path)
        (tmp_path / "artifact.json").write_text("{}")
        out = mp._sync_data_delivery(tmp_path)
        assert out["pushed"] is False
        assert out["staged_files"] == []
        assert "GITHUB_TOKEN" in out["skipped"]

    def test_push_off_skips_even_with_a_token(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GITHUB_TOKEN", "fake")
        monkeypatch.setenv("NBA_PUSH", "0")
        import master_pipeline as mp
        monkeypatch.setattr(mp.config, "DATA_DELIVERY_DIR", tmp_path)
        (tmp_path / "artifact.json").write_text("{}")
        out = mp._sync_data_delivery(tmp_path)
        assert out["pushed"] is False
        assert "NBA_PUSH" in out["skipped"]

    def test_summary_write_precedes_the_delivery_sync(self):
        """Pin the write-before-sync order in run()'s publish tail.

        The sync result is attached to the summary under either order, so no
        mocked-sync behavioral test can tell them apart - but only the
        write-first order ships this run's summary: written after the sync it
        dies with the ephemeral session and main keeps serving the stale copy
        _prune left behind (2026-10-01: main's summary was frozen at
        2026-09-27 while five later runs pushed fresh artifacts around it).
        """
        import inspect
        import master_pipeline as mp

        source = inspect.getsource(mp.run)
        write_at = source.index('nba_pipeline_summary.json").write_text')
        sync_at = source.index("sync = _sync_data_delivery(")
        assert write_at < sync_at, (
            "run() writes nba_pipeline_summary.json AFTER _sync_data_delivery,"
            " so the pushed delivery never contains this run's summary; "
            "write it before the sync, the order NFL and NHL already use")

    def test_the_publish_step_lands_in_the_pushed_log(self):
        """run() must announce publish BEFORE the sync stages the log.

        The sync commits the delivery directory - the run log included - so
        anything printed after it exists only on this machine: both
        2026-10-02 delivered logs end mid-bar at \"monitor 9/10\" with no
        completion line, and a reviewer cannot tell from the log alone that
        the run finished or what status it finished with.
        """
        import inspect
        import master_pipeline as mp

        source = inspect.getsource(mp.run)
        step_at = source.index('_step("publish"')
        sync_at = source.index("sync = _sync_data_delivery(")
        assert step_at < sync_at, (
            "run() announces publish AFTER the sync stages the log file, so "
            "the pushed log can never contain the run's completion line; "
            "print the publish step before the sync, the way the summary "
            "write already precedes it")

    def test_a_push_that_delivers_nothing_is_a_failure(self):
        """The run reports success off the same signal as the push, so an
        unverified push turns a delivery failure into a silent one."""
        from github_sync import verify_pushed_paths

        class FakeGit:
            def fetch(self, *_a): return ""
            def rev_parse(self, *_a): return "deadbeef"
            def ls_tree(self, *_a):
                return "nba-backend/data_delivery/present.json"

        class FakeRepo:
            git = FakeGit()

        with pytest.raises(RuntimeError) as excinfo:
            verify_pushed_paths(FakeRepo(), "main",
                                ["nba-backend/data_delivery/absent.json"])
        assert "absent.json" in str(excinfo.value)

    def test_a_rejected_push_is_retried_onto_the_new_tip(self, monkeypatch):
        """A non-fast-forward rejection is a race, not a verdict. Re-syncing and
        replaying is what stops a race from costing the run its artifacts."""
        import git
        from github_sync import push_with_retry

        class Info:
            def __init__(self, flags, summary="rejected"):
                self.flags = flags
                self.summary = summary

        class Remote:
            def __init__(self, results):
                self.results = list(results)
                self.calls = 0

            def push(self, _branch):
                self.calls += 1
                return [self.results.pop(0)]

        class FakeRepo:
            def __init__(self, remote):
                self.remote_obj = remote
                self.resynced = 0

            def remote(self, _name):
                return self.remote_obj

        remote = Remote([Info(git.PushInfo.REJECTED), Info(0)])
        repo = FakeRepo(remote)
        restaged = []
        monkeypatch.setattr("github_sync.sync_remote_tip",
                            lambda *_a, **_k: setattr(repo, "resynced", repo.resynced + 1))
        push_with_retry(repo, "main", restage=lambda: restaged.append(1))
        assert remote.calls == 2
        assert repo.resynced == 1, "must re-sync to the new tip before replaying"
        assert restaged == [1], "must replay this run's artifacts"

    def test_exhausted_retries_raise(self, monkeypatch):
        import git
        from github_sync import push_with_retry

        class Info:
            flags = git.PushInfo.REJECTED
            summary = "non-fast-forward"

        class Remote:
            def push(self, _branch):
                return [Info()]

        class FakeRepo:
            def remote(self, _name):
                return Remote()

        monkeypatch.setattr("github_sync.sync_remote_tip", lambda *_a, **_k: None)
        with pytest.raises(RuntimeError) as excinfo:
            push_with_retry(FakeRepo(), "main", attempts=2)
        assert "non-fast-forward" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A host that refuses this client
# ---------------------------------------------------------------------------


class _Response:
    """The minimum surface ``http_json`` touches."""

    def __init__(self, body: bytes):
        self._body = body
        self.headers: dict = {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _refuse(request, timeout=None):
    raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)


def _probe_then_refuse(calls: list):
    """Answer the preflight probe, refuse everything after it.

    Separating the two is what lets a test assert about the sweep rather than
    about the probe that precedes it.
    """
    def handler(request, timeout=None):
        calls.append(request.full_url)
        if len(calls) == 1:
            return _Response(b'{"events": [], "resultSets": []}')
        return _refuse(request, timeout)
    return handler


def _refuse_all(calls: list):
    """Refuse every request, the probe included."""
    def handler(request, timeout=None):
        calls.append(request.full_url)
        return _refuse(request, timeout)
    return handler


class _CapturedStream:
    """A stderr that is not a terminal, which is what Kaggle hands the run.

    ``tqdm`` resolves ``sys.stderr`` when a bar is constructed, so replacing it
    with this is enough to see exactly what a captured cell would have been
    sent - and ``isatty()`` answers False, which is the condition the removed
    gate used to hide the bar behind.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def write(self, text: str) -> int:
        self._chunks.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def text(self) -> str:
        return "".join(self._chunks)


class TestRefusedHost:
    """What happens when a host says no.

    The remote run that prompted this group answered 403 on every one of the
    1,024 days in its window, and the sweep walked all of them before reporting
    that the schedule was missing - which names the wrong culprit. These pin
    the opposite behaviour: one request to find out, and then a stop.
    """

    @pytest.fixture(autouse=True)
    def _clear_probes(self):
        ing._PROBED.clear()
        yield
        ing._PROBED.clear()

    @staticmethod
    def _games(count: int) -> pd.DataFrame:
        today = date.today()
        return pd.DataFrame({
            "nba_game_id": [f"0022400{i:03d}" for i in range(count)],
            "gameday": [pd.Timestamp(today - timedelta(days=i))
                        for i in range(count)],
        })

    def test_a_refused_host_costs_exactly_one_request(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(ing.urllib.request, "urlopen", _refuse_all(calls))
        with pytest.raises(ing.HostUnavailable) as excinfo:
            ing.preflight(["espn"])
        assert len(calls) == 1
        assert "ESPN" in str(excinfo.value)

    def test_the_refusal_says_what_continuing_would_cost(self, monkeypatch):
        """The number is the argument for stopping, so it belongs in the error."""
        calls: list = []
        monkeypatch.setattr(ing.urllib.request, "urlopen", _refuse_all(calls))
        with pytest.raises(ing.HostUnavailable) as excinfo:
            ing.preflight(["espn"])
        assert "1024" in str(excinfo.value)

    def test_a_slow_host_is_never_called_a_refusal(self, monkeypatch):
        """A read timeout is slowness, not a blacklist. The 2026-10-04 23:51
        crash delivery printed "a host refusing this client" for exactly this
        failure while the endpoint had been serving the run seconds earlier."""
        calls: list = []

        def timeout(request, timeout=None):
            calls.append(request.full_url)
            raise TimeoutError("The read operation timed out")

        monkeypatch.setattr(ing.urllib.request, "urlopen", timeout)
        with pytest.raises(ing.HostUnavailable) as excinfo:
            ing.preflight(["espn"])
        text = str(excinfo.value)
        assert "did not answer" in text
        assert "refusing this client" not in text
        # Slow is retried at the pull's patience, not believed once.
        assert len(calls) == ing.DEFAULT_PROBE_ATTEMPTS

    def test_the_probe_retries_a_slow_answer_into_a_pass(self, monkeypatch):
        """One slow read must not condemn a host that answers on the retry —
        the failure that killed the 2026-10-04 run."""
        calls: list = []

        def flaky(request, timeout=None):
            calls.append(request.full_url)
            if len(calls) == 1:
                raise TimeoutError("The read operation timed out")
            return _Response(b'{"events": [], "resultSets": []}')

        monkeypatch.setattr(ing.urllib.request, "urlopen", flaky)
        report = ing.preflight(["espn"])
        assert "espn" in report
        assert len(calls) == 2

    def test_a_host_is_probed_once_per_process(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(ing.urllib.request, "urlopen", _refuse_all(calls))
        with pytest.raises(ing.HostUnavailable):
            ing.preflight(["espn"])
        # Recorded even though it raised, so a caller that retries in-process
        # does not pay for the same answer twice.
        ing.preflight(["espn"])
        assert len(calls) == 1

    def test_the_espn_headers_carry_nothing_unjustified(self):
        """ESPN's edge is sensitive to header combinations in a way that is not
        reproducible: the same dictionary answers 200 through urllib and 403
        through requests. So the rule is not "which header is bad" but "ship
        only what a measurement justifies" - which here is the same minimal
        set MLB's working ESPN path sends."""
        assert set(ing.ESPN_HEADERS) == {"User-Agent", "Accept"}
        assert ing.ESPN_HEADERS["Accept"] == "*/*"
        assert "Accept-Language" not in ing.ESPN_HEADERS

    def test_the_stats_probe_is_one_day_of_the_real_endpoint(self):
        """A probe aimed elsewhere could answer while the call that matters is
        the one being refused, so it has to be the same endpoint and headers."""
        url, headers = ing._probe_url("stats")
        assert url.startswith(src.SEASON_LOG_URL)
        assert headers["User-Agent"] == ing.USER_AGENT
        query = dict(urllib.parse.parse_qsl(url.split("?", 1)[1]))
        assert query["DateFrom"] == query["DateTo"] != ""

    def test_the_schedule_sweep_stops_instead_of_asking_every_day(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        calls: list = []
        monkeypatch.setattr(ing.urllib.request, "urlopen",
                            _probe_then_refuse(calls))
        with pytest.raises(ing.ScheduleUnavailable) as excinfo:
            ing._fetch_schedule(date(2024, 1, 1), date(2024, 12, 31))
        # One probe plus the three consecutive failures, not 366 requests.
        assert len(calls) == 1 + ing.DEFAULT_SCHEDULE_MAX_FAILURES
        assert "in a row" in str(excinfo.value)

    def test_a_fully_cached_schedule_never_touches_the_network(
            self, monkeypatch, tmp_path):
        """A warm cache must not fail on a host it has no reason to ask."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        start = date(2024, 1, 1)
        for offset, day in enumerate([start, start + timedelta(days=1),
                                      start + timedelta(days=2)]):
            event = _espn_event(event_id=f"4015858{offset}",
                                when=f"2024-01-0{offset + 1}T19:00Z")
            ing._write_parquet(src.espn_schedule_frame([event]),
                               ing._schedule_path(day))

        def explode(*_a, **_k):
            raise AssertionError("a warm cache must not reach the network")

        monkeypatch.setattr(ing.urllib.request, "urlopen", explode)
        assert len(ing._fetch_schedule(start, start + timedelta(days=2))) == 3

    def test_the_season_log_pull_stops_after_consecutive_failures(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        calls: list = []
        monkeypatch.setattr(ing.urllib.request, "urlopen",
                            _probe_then_refuse(calls))
        with pytest.raises(ing.SeasonLogUnavailable) as excinfo:
            ing._fetch_season_logs(date(2024, 1, 1), date(2024, 6, 30))
        # That window is four season-types; the pull gave up after two.
        assert len(calls) == 1 + ing.DEFAULT_SEASON_LOG_MAX_FAILURES
        assert "stopped early" in str(excinfo.value)

    def test_a_refused_play_by_play_stops_the_sweep_but_not_the_run(
            self, monkeypatch, tmp_path):
        """Play-by-play feeds a feature that is forward-filled without it, so
        losing it is a thinner model rather than a wrong one. What is not
        acceptable is paying per game to rediscover a refusal."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        calls: list = []
        monkeypatch.setattr(ing.urllib.request, "urlopen",
                            _probe_then_refuse(calls))
        frame, info = ing._fetch_play_by_play(self._games(8))
        assert info["tripped"] is True
        assert len(calls) == 1 + ing.DEFAULT_PBP_MAX_FAILURES
        assert frame.empty

    def test_one_unreadable_game_does_not_trip_the_breaker(
            self, monkeypatch, tmp_path):
        """A single game with no play-by-play is ordinary; a host refusing every
        request is not. The threshold exists to tell those apart."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        calls: list = []

        def fail_once(request, timeout=None):
            calls.append(request.full_url)
            if len(calls) == 2:
                return _refuse(request, timeout)
            return _Response(b'{"game": {"actions": []}}')

        monkeypatch.setattr(ing.urllib.request, "urlopen", fail_once)
        _frame, info = ing._fetch_play_by_play(self._games(3))
        assert info["tripped"] is False
        assert info["requested"] == 3

    def test_a_missing_game_id_is_not_swept(self, monkeypatch, tmp_path):
        """A NaN id stringifies as ``nan`` and passes a length filter, so the
        2026-09-30 cold sweep spent a request on playbyplayv3?GameID=nan,
        collected a guaranteed HTTP 400, and reported "1 unavailable" every
        run. A game with no id is not a target: it is dropped by name, never
        requested, and the games that do carry ids are still swept."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        games = self._games(3)
        nameless = pd.DataFrame({
            "nba_game_id": [float("nan")],
            "gameday": [pd.Timestamp(date.today())],
        })
        calls: list = []

        def answer(request, timeout=None):
            calls.append(request.full_url)
            return _Response(b'{"game": {"actions": []}}')

        monkeypatch.setattr(ing.urllib.request, "urlopen", answer)
        _frame, info = ing._fetch_play_by_play(
            pd.concat([games, nameless], ignore_index=True))
        assert info["requested"] == 3, len(calls)
        assert all("GameID=nan" not in url for url in calls)
        assert info["tripped"] is False

    def test_the_sweep_backfills_the_oldest_cache_holes(
            self, monkeypatch, tmp_path):
        """A stretch of games that fell outside the lookback before any run
        fetched it stays uncached forever, and every first game after the hole
        has no event history to read. The budget is a deadline, not a count,
        so after the recent slice it is spent backward on the holes - but
        walking BACK from the recent slice, not jumping to the far end."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        games = self._games(6)
        old = pd.DataFrame({
            "nba_game_id": [f"0012400{i:03d}" for i in range(4)],
            "gameday": [pd.Timestamp(date.today() - timedelta(days=400 - i))
                        for i in range(4)],
        })
        for gid in games.nba_game_id:
            ing._write_parquet(pd.DataFrame({"x": [1.0]}),
                               ing._pbp_path(str(gid)))
        eligible = pd.concat([games, old], ignore_index=True)
        calls: list = []

        def answer(request, timeout=None):
            calls.append(request.full_url)
            return _Response(b'{"game": {"actions": []}}')

        monkeypatch.setattr(ing.urllib.request, "urlopen", answer)
        _frame, info = ing._fetch_play_by_play(eligible)
        # The recent games were all cached, so every request is one of the
        # four old uncached ones.
        assert info["requested"] == 4, len(calls)
        assert info["tripped"] is False

    def test_the_backfill_walks_back_from_the_recent_slice(
            self, monkeypatch, tmp_path):
        """The holes are filled newest-first, so the swept set is one
        contiguous block ending at the newest game.

        This is the pin for the 2026-09-29 run, which swept the OLDEST
        ``cap`` holes: 2,086 games, 0 from cache, the oldest 1,500 plus the
        most recent 586, and the ~1,376 games between them (2025-01-15 ..
        2026-01-31) never fetched at all. A hole is not local - the ladder
        forward-fills each team's event profile across it - so that band was
        a year of frozen EWMs while the coverage report still read 100%
        measured. Oldest-first buys the cheapest holes in the window and
        leaves the expensive band frozen.
        """
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        # 12 recent games (today back 11 days) and 12 far older ones, all
        # uncached. The old block sits OUTSIDE a 30-day lookback, so the
        # recent slice is exactly the 12 recent games and the backfill has to
        # choose 6 of the 12 old ones - which is what makes newest-first vs
        # oldest-first observable. `old` is written newest-first in time, so
        # ids 0..5 are the 6 nearest the recent slice and 6..11 the 6
        # furthest away.
        # (The retired NBA_PBP_LOOKBACK_DAYS / NBA_PBP_MAX_GAMES envs no
        # longer exist; setting them would be a no-op on nothing.)
        recent = self._games(12)
        old = pd.DataFrame({
            "nba_game_id": [f"0012400{i:03d}" for i in range(12)],
            "gameday": [pd.Timestamp(date.today() - timedelta(days=200 + i))
                        for i in range(12)],
        })
        eligible = pd.concat([recent, old], ignore_index=True)
        requested: list = []

        def answer(request, timeout=None):
            requested.append(request.full_url)
            return _Response(b'{"game": {"actions": []}}')

        monkeypatch.setattr(ing.urllib.request, "urlopen", answer)
        _frame, info = ing._fetch_play_by_play(eligible)
        assert info["tripped"] is False

        asked = {url.split("GameID=")[-1].split("&")[0] for url in requested[1:]}

        # The sweep is UNBOUNDED: all 24 eligible games are requested - the
        # 12 inside the retired lookback and the 12 old holes the retired
        # cap used to strand. (Directive 2026-09-29: every game limitation
        # is removed from ingestion and feature engineering.)
        assert info["requested"] == 24, info["requested"]
        assert asked == {str(g) for g in eligible.nba_game_id}, sorted(asked)
        # asked for 6..11 instead, which is what stranded a year of games
        # between the two swept blocks on the 2026-09-29 run.
        assert asked == ({str(g) for g in recent.nba_game_id}
                         | {f"0012400{i:03d}" for i in range(12)}), sorted(asked)

        # The decisive invariant, stated independently of which ids those
        # are: no unswept game may sit chronologically BETWEEN two swept
        # ones. That is precisely the condition that freezes a team's
        # forward-filled event profile across a band of the window.
        swept = (eligible[eligible.nba_game_id.astype(str).isin(asked)]
                 .sort_values("gameday"))
        lo, hi = swept.gameday.min(), swept.gameday.max()
        stranded = eligible[(eligible.gameday > lo) & (eligible.gameday < hi)
                            & ~eligible.nba_game_id.astype(str).isin(asked)]
        assert stranded.empty, stranded.gameday.tolist()

    def test_a_complete_cache_makes_the_backfill_free(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        games = self._games(6)
        for gid in games.nba_game_id:
            ing._write_parquet(pd.DataFrame({"x": [1.0]}),
                               ing._pbp_path(str(gid)))
        calls: list = []

        def explode(*_a, **_k):
            raise AssertionError("a complete cache must not reach the network")

        monkeypatch.setattr(ing.urllib.request, "urlopen", explode)
        _frame, info = ing._fetch_play_by_play(games)
        assert info["requested"] == 0
        assert info["cached"] == 6


# ---------------------------------------------------------------------------
# Window and env
# ---------------------------------------------------------------------------


class TestWindow:
    def test_the_window_defaults_to_the_eligible_seasons(self, monkeypatch):
        monkeypatch.delenv(ing.WINDOW_START_ENV, raising=False)
        monkeypatch.delenv(ing.WINDOW_END_ENV, raising=False)
        start, end = ing.window()
        assert start >= date(config.OOF_FIRST_SEASON, 1, 1)
        assert end >= start

    def test_an_explicit_window_is_honoured(self, monkeypatch):
        """An explicit window that still reaches today is honoured exactly
        as written — a forward-looking rebuild pin (the notebook's
        season-opening pin) passes through untouched (2026-10-06
        stale-pin guard, MLB/NHL/NFL parity: only pins BEFORE today are
        extended, and those are pinned in the stale-pin tests)."""
        monkeypatch.setenv(ing.WINDOW_START_ENV, "2024-01-01")
        monkeypatch.setenv(ing.WINDOW_END_ENV, "2099-06-01")
        assert ing.window() == (date(2024, 1, 1), date(2099, 6, 1))

    def test_a_reversed_window_is_swapped(self, monkeypatch):
        """Reversed ends still swap — exercised on a forward pair so the
        2026-10-06 stale-pin guard (which extends a past END before the
        swap check) does not mask the swap itself."""
        monkeypatch.setenv(ing.WINDOW_START_ENV, "2099-06-01")
        monkeypatch.setenv(ing.WINDOW_END_ENV, "2099-01-01")
        assert ing.window() == (date(2099, 1, 1), date(2099, 6, 1))

    def test_a_stale_explicit_end_extends_to_today(self, monkeypatch):
        """Stale-pin guard (MLB/NHL/NFL parity): a literal NBA_END_DATE
        pin before today extends to today so the daily slate cannot
        freeze when the pin's day passes — the notebook is Kaggle-owned
        (never edited from the repo), so the PIPELINE defends itself."""
        monkeypatch.setenv(ing.WINDOW_START_ENV, "2024-01-01")
        monkeypatch.setenv(ing.WINDOW_END_ENV, "2024-06-01")
        _start, end = ing.window()
        assert end == date.today()  # stale → today

    def test_a_season_label_turns_over_in_july(self):
        assert ing.season_label(date(2024, 10, 22)) == "2024-25"
        assert ing.season_label(date(2024, 1, 10)) == "2023-24"

    def test_seasons_are_bounded_by_overlap(self):
        assert ing.season_starts(date(2024, 1, 1), date(2024, 12, 31)) == [2023, 2024]
        assert ing.season_starts(date(2025, 3, 1), date(2025, 3, 2)) == [2024]

    def test_int_env_returns_the_value_not_a_boolean(self):
        """`value > 0 or default` yields True, which silently turns every
        integer budget into 1."""
        assert ing._int_env("NBA_TEST_INT", 99) == 99
        import os
        os.environ["NBA_TEST_INT"] = "6"
        try:
            assert ing._int_env("NBA_TEST_INT", 99) == 6
        finally:
            del os.environ["NBA_TEST_INT"]

    def test_float_env_falls_back_on_junk(self):
        import os
        os.environ["NBA_TEST_FLOAT"] = "not a number"
        try:
            assert ing._float_env("NBA_TEST_FLOAT", 2.5) == 2.5
        finally:
            del os.environ["NBA_TEST_FLOAT"]


# ---------------------------------------------------------------------------
# Eligibility and validation
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_undecided_games_are_kept_for_the_slate(self):
        games = pd.DataFrame([
            _schedule_row(game_id="a", home_score=1.0, away_score=2.0),
            _schedule_row(game_id="b", home_score=np.nan, away_score=np.nan),
        ])
        games["is_final"] = [True, False]
        out = ing.eligible_games(games)
        assert set(out.game_id) == {"a", "b"}

    def test_an_unplayable_game_type_is_dropped(self):
        games = pd.DataFrame([_schedule_row(game_type=99)])
        assert ing.eligible_games(games).empty

    def test_a_game_with_no_identity_is_dropped(self):
        games = pd.DataFrame([_schedule_row(game_id="")])
        assert ing.eligible_games(games).empty

    def test_a_too_small_window_is_refused_by_name(self):
        games = pd.DataFrame([_schedule_row(game_id=f"g{i}") for i in range(3)])
        games["is_final"] = True
        facts = ing.NBAFacts(games=games, team_stats=pd.DataFrame(),
                             player_stats=pd.DataFrame(), team_events=pd.DataFrame(),
                             play_by_play=pd.DataFrame(), team_names={})
        with pytest.raises(RuntimeError) as excinfo:
            ing._validate(facts)
        assert "settled" in str(excinfo.value)

    def test_a_duplicated_team_game_is_refused(self):
        """Two rows for one team-game would double every count in the ladder."""
        games = pd.DataFrame([_schedule_row(game_id=f"g{i}") for i in range(60)])
        games["is_final"] = True
        team = pd.DataFrame([{"game_id": "g1", "team": "BOS", "points_for": 1.0,
                              "points_against": 1.0},
                             {"game_id": "g1", "team": "BOS", "points_for": 1.0,
                              "points_against": 1.0}])
        player = pd.DataFrame([{"game_id": "g1", "player_id": "1",
                                "team": "BOS"}])
        facts = ing.NBAFacts(games=games, team_stats=team, player_stats=player,
                             team_events=pd.DataFrame(), play_by_play=pd.DataFrame(),
                             team_names={})
        with pytest.raises(RuntimeError) as excinfo:
            ing._validate(facts)
        assert "team_stats" in str(excinfo.value)

    def test_two_teams_per_game_is_not_a_duplicate(self):
        games = pd.DataFrame([_schedule_row(game_id=f"g{i}") for i in range(60)])
        games["is_final"] = True
        team = pd.DataFrame([{"game_id": f"g{i}", "team": t,
                              "points_for": 110.0 if t == "BOS" else 105.0,
                              "points_against": 105.0 if t == "BOS" else 110.0}
                             for i in range(60) for t in ("BOS", "NYK")])
        player = pd.DataFrame([{"game_id": f"g{i}", "player_id": str(p),
                                "team": "BOS"}
                               for i in range(60) for p in range(5)])
        facts = ing.NBAFacts(games=games, team_stats=team, player_stats=player,
                             team_events=pd.DataFrame(), play_by_play=pd.DataFrame(),
                             team_names={})
        ing._validate(facts)
        facts.team_stats.loc[0, "points_for"] += 1
        with pytest.raises(RuntimeError, match="disagree with final scores"):
            ing._validate(facts)
        facts.team_stats = team.iloc[1:]
        with pytest.raises(RuntimeError, match="missing or disagree"):
            ing._validate(facts)


class TestTrainableGames:
    """A finished game with no player lines cannot become a training label."""

    def _games(self):
        return pd.DataFrame([
            # A normal game: finished, with season-log lines.
            _schedule_row(game_id="a", nba_id="0022301200"),
            # The NBA Cup final: finished on ESPN, absent from LeagueGameLog.
            _schedule_row(game_id="b", nba_id="", home="OKC", away="MIL",
                          home_score=97.0, away_score=81.0),
            # An all-star game: finished, and its squads are not league teams.
            _schedule_row(game_id="c", nba_id="", home="KEN", away="CHK",
                          home_score=50.0, away_score=47.0),
            # A pending game: no score yet, and no lines yet either.
            _schedule_row(game_id="d", nba_id="", home="BOS", away="LAL",
                          home_score=np.nan, away_score=np.nan),
        ])

    def test_a_finished_game_with_no_lines_is_not_trainable(self):
        trainable = ing.trainable_games(self._games())
        assert set(trainable.game_id) == {"a"}

    def test_the_exclusion_is_reported_with_the_games_named(self, caplog):
        with caplog.at_level("INFO"):
            ing.trainable_games(self._games())
        assert "no season-log player lines" in caplog.text
        assert "MIL@OKC" in caplog.text

    def test_a_pending_game_is_left_for_the_slate_not_dropped(self):
        """trainable_games only narrows the settled side; the slate is built
        from the full eligible set."""
        eligible = ing.eligible_games(self._games())
        pending = eligible[eligible.home_score.isna() | eligible.away_score.isna()]
        assert set(pending.game_id) == {"d"}

    def test_an_empty_frame_is_handled(self):
        assert ing.trainable_games(pd.DataFrame()).empty


class TestEspnRosterSlug:
    """ESPN's roster path is strict, and two of our tokens are not its own.

    Measured 2026-09-27 against the live endpoint: of the thirty abbreviations
    stats.nba.com publishes, ``nop`` and ``uta`` answer HTTP 400 while ``no``
    (19 athletes) and ``utah`` (18) answer 200. A 400 is a malformed request, so
    the run's warning that "a request this pipeline sends will not change it" was
    correct - and the consequence was that two clubs' injury state was read as
    healthy, the one state this source must never be mistaken for.
    """

    def test_the_two_divergent_tokens_resolve_to_espn_segments(self):
        assert src.espn_roster_url("NOP").endswith("/teams/no/roster")
        assert src.espn_roster_url("UTA").endswith("/teams/utah/roster")

    def test_resolution_is_case_and_whitespace_insensitive(self):
        assert src.espn_roster_url(" nop ") == src.espn_roster_url("NOP")
        assert src.espn_roster_url("uta") == src.espn_roster_url("UTA")

    def test_every_other_token_is_passed_through_lowercased(self):
        for token in ("MIL", "GSW", "NYK", "SAS", "WAS", "LAL"):
            assert src.espn_roster_url(token).endswith(
                f"/teams/{token.lower()}/roster"), token

    def test_the_alias_table_covers_exactly_the_measured_divergences(self):
        """A table that grows without evidence re-creates the problem it fixed:
        a wrong entry turns a working request into a 400."""
        assert src.ESPN_ROSTER_SLUG_ALIASES == {"NOP": "no", "UTA": "utah"}


class TestCardPresentationFields:
    """Tipoff, arena, and the postponed state a card has to be able to show.

    The delivered moneyline JSON carried ``start_time_utc`` and ``venue`` as
    columns that were always null, and every ``p_*`` player field was null too,
    so the board showed no tipoff and no top player for either side. Three
    separate causes, all upstream of the frontend: the parser dropped fields
    ESPN had already sent, the contract discarded what survived, and the player
    lookup asked for a column name the contract never produced.
    """

    @staticmethod
    def _row(event) -> dict:
        return src._parse_espn_event(event, config.GAME_TYPE_REG)

    def test_the_tipoff_survives_as_a_utc_instant(self):
        row = self._row(_espn_event(when="2026-10-20T23:00Z"))
        assert row["start_time_utc"] == "2026-10-20T23:00:00Z"

    def test_the_tipoff_is_normalized_rather_than_copied(self):
        """A board date is an Eastern date; the instant is what converts back."""
        row = self._row(_espn_event(when="2025-01-10T03:30Z"))
        assert row["start_time_utc"].endswith("Z")
        # 03:30Z on Jan 10 is 22:30 ET on Jan 9 - the rollover the guard in
        # ``_answered_a_different_day`` already tolerates on the schedule side.
        assert str(row["gameday"])[:10] == "2025-01-09"

    def test_an_unparseable_tipoff_yields_nothing_rather_than_midnight(self):
        """A fabricated midnight UTC lands on the prior Eastern evening and
        renders as a plausible, wrong tipoff."""
        assert src._utc_iso(None) == ""
        assert src._utc_iso(pd.NaT) == ""
        assert src._utc_iso("not a date") == ""

    def test_the_contract_keeps_the_fields_rather_than_dropping_them(self):
        """``normalize`` projects to the declared schema, so an undeclared
        column is silently dropped - which is how a delivered artifact ends up
        with an empty ``start_time_utc`` column."""
        for column in ("start_time_utc", "venue", "game_state",
                       "game_status_detail"):
            assert column in contract.SCHEMAS["games"], column

    def test_the_venue_is_carried_when_the_schedule_publishes_one(self):
        event = _espn_event()
        event["competitions"][0]["venue"] = {"fullName": "TD Garden"}
        assert self._row(event)["venue"] == "TD Garden"

    def test_a_game_with_no_venue_reports_none_rather_than_placeholder(self):
        assert self._row(_espn_event())["venue"] == ""

    def test_the_state_and_detail_are_carried_for_the_card_to_label(self):
        row = self._row(_espn_event(state="pre", completed=False,
                                    detail="Postponed"))
        assert row["game_state"] == "pre"
        assert row["game_status_detail"] == "Postponed"
        assert not row["is_final"]

    def test_a_postponed_game_is_recognised_by_detail_not_by_a_word_list(self):
        assert src.is_postponed_detail("Postponed")
        assert src.is_postponed_detail("Canceled")
        assert src.is_postponed_detail("Suspended")
        assert src.is_postponed_detail("Rescheduled to a later date")
        assert not src.is_postponed_detail("Final")
        assert not src.is_postponed_detail("7:30 PM ET")
        assert not src.is_postponed_detail("")

    def test_serving_refuses_to_invent_a_tipoff(self):
        """Mirrors the NHL rule: a date with no time is not a tipoff."""
        import serving
        assert serving._start_time_utc({"start_time_utc": ""}) is None
        assert serving._start_time_utc({"start_time_utc": "2026-10-20"}) is None
        assert serving._start_time_utc(
            {"start_time_utc": "2026-10-20T23:00:00Z"}) == "2026-10-20T23:00:00Z"

    def test_a_postponed_game_is_labelled_postponed_on_the_card(self, tmp_path):
        import serving
        slate = pd.DataFrame([{
            "game_id": "1", "gameday": pd.Timestamp("2025-01-09"),
            "home_team": "LAL", "away_team": "CHA",
            "home_score": np.nan, "away_score": np.nan,
            "home_win_prob_model": 0.6, "away_win_prob_model": 0.4,
            "game_status_detail": "Postponed"}])
        record = serving.write_moneyline_json(tmp_path / "m.json", slate,
                                              [0.6], [0.6])
        assert record["games"][0]["game_status"] == "Postponed"

    def test_an_unplayed_game_with_no_detail_is_still_scheduled(self, tmp_path):
        import serving
        slate = pd.DataFrame([{
            "game_id": "1", "gameday": pd.Timestamp("2026-10-20"),
            "home_team": "BOS", "away_team": "DET",
            "home_score": np.nan, "away_score": np.nan,
            "home_win_prob_model": 0.6, "away_win_prob_model": 0.4}])
        record = serving.write_moneyline_json(tmp_path / "m.json", slate,
                                              [0.6], [0.6])
        assert record["games"][0]["game_status"] == "Scheduled"

    def test_a_qualifying_player_is_actually_found(self):
        """End to end over the real column names, because a name-only test
        cannot tell a fixed guard from a still-empty one."""
        import player_enrichment
        games = pd.DataFrame([
            {"game_id": f"g{n}", "gameday": pd.Timestamp("2026-01-0%d" % (n + 1)),
             "home_team": "BOS", "away_team": "DET",
             "home_score": 110.0, "away_score": 100.0}
            for n in range(config.PLAYER_WINDOW_GAMES)])
        rows = []
        for n in range(config.PLAYER_WINDOW_GAMES):
            for minutes, points, ast in ((36, 30, 11), (12, 4, 1), (5, 2, 0)):
                rows.append({
                    "game_id": f"g{n}", "team": "BOS", "player_id": "p1",
                    "player_name": "J. Player", "minutes": minutes,
                    "points": points, "ast": ast})
        record = player_enrichment._player_for_team(
            pd.DataFrame(rows), "BOS", pd.Timestamp("2026-02-01"), games)
        assert record.get("name") == "J. Player"
        assert record["ppg"] > record.get("apg", 0)
        assert record["games"] == config.PLAYER_WINDOW_GAMES

    def test_a_cached_day_from_an_older_parser_is_refetched(self, tmp_path):
        """A cache entry written before the frame gained a tipoff reads back as
        a hit and silently yields empty values, so a whole window reports no
        tipoff while a freshly fetched day reports one. A stale shape is a
        miss, and the refetch overwrites it in place."""
        path = tmp_path / "schedule" / "20250109.parquet"
        ing._write_parquet(pd.DataFrame({
            "game_id": ["1"], "gameday": [pd.Timestamp("2025-01-09")],
            "home_team": ["LAL"], "away_team": ["CHA"],
            "home_score": [np.nan], "away_score": [np.nan]}),
            path)
        stale = pd.DataFrame({"game_id": ["1"]})
        assert not ing._schedule_hit(stale, path)
        fresh = pd.DataFrame({"game_id": ["1"], "start_time_utc": ["x"],
                              "venue": ["TD Garden"], "game_state": ["pre"],
                              "game_status_detail": ["7:30 PM ET"]})
        assert ing._schedule_hit(fresh, path)

    def test_a_genuinely_empty_day_stays_a_hit(self, tmp_path):
        """Most days in a window have no games, and that emptiness is the fact
        worth keeping - refetching them would cost a request to learn nothing."""
        path = tmp_path / "schedule" / "20250110.parquet"
        ing._write_parquet(pd.DataFrame(), path)
        assert ing._schedule_hit(pd.DataFrame(), path)

    def test_a_postponed_game_is_not_offered_as_upcoming(self):
        """MLB's decided-frame rule in one line: postponements are excluded.
        Left in the slate it is indistinguishable from a real game, which is
        how a January 2025 postponement reached a board dated October 2026."""
        games = pd.DataFrame([{
            "game_id": "1", "gameday": pd.Timestamp("2025-01-09"),
            "home_team": "LAL", "away_team": "CHA", "season": 2025,
            "home_score": np.nan, "away_score": np.nan,
            "game_type": config.GAME_TYPE_REG,
            "game_status_detail": "Postponed"}])
        slate = feat.build_slate_features(games)
        assert slate.empty

    def test_a_real_upcoming_game_is_still_offered(self):
        games = pd.DataFrame([{
            "game_id": "1", "gameday": pd.Timestamp("2026-10-20"),
            "home_team": "BOS", "away_team": "DET", "season": 2027,
            "home_score": np.nan, "away_score": np.nan,
            "game_type": config.GAME_TYPE_REG,
            "game_status_detail": "7:30 PM ET"}])
        assert len(feat.build_slate_features(games)) == 1


class TestSixtyDaySlices:
    """The 60-day pull, and the reason it is only applied where it works.

    MLB chunks its schedule because StatsAPI's ``schedule`` takes
    ``startDate``/``endDate``; ``results.SCHEDULE_CHUNK_DAYS`` is 60 for the
    same reason ``statcast_chunk_days`` is. NBA's season log can be chunked the
    same way because ``LeagueGameLog`` already accepts ``DateFrom``/``DateTo``.
    ESPN's scoreboard cannot, which is measured rather than assumed and is what
    ``TestSilentFallback`` exists to defend.
    """

    def test_a_window_is_cut_into_sixty_day_slices(self):
        out = ing._slices(date(2024, 1, 1), date(2024, 6, 28), 60)
        # 60 days inclusive at both ends: Jan 1 + 59 days is Feb 29 in a leap
        # year, which is the sort of thing a "60" that is really 61 gets wrong.
        assert out[0] == (date(2024, 1, 1), date(2024, 2, 29))
        assert out[1] == (date(2024, 3, 1), date(2024, 4, 29))
        assert all((hi - lo).days + 1 == 60 for lo, hi in out)
        assert out[-1] == (date(2024, 4, 30), date(2024, 6, 28))

    def test_slices_cover_every_day_exactly_once(self):
        start, end = date(2023, 11, 3), date(2026, 10, 20)
        seen = [lo + timedelta(days=n)
                for lo, hi in ing._slices(start, end, 60)
                for n in range((hi - lo).days + 1)]
        assert seen == [start + timedelta(days=n)
                        for n in range((end - start).days + 1)]

    def test_the_last_slice_is_clipped_rather_than_padded(self):
        out = ing._slices(date(2024, 1, 1), date(2024, 1, 10), 60)
        assert out == [(date(2024, 1, 1), date(2024, 1, 10))]

    @pytest.mark.parametrize("start,end", [
        (date(2024, 5, 1), date(2024, 4, 30)),   # empty range
        (date(2024, 5, 1), date(2024, 5, 1)),     # single day
    ])
    def test_degenerate_windows_are_handled(self, start, end):
        out = ing._slices(start, end, 60)
        assert all(lo <= hi for lo, hi in out)

    def test_a_nonsense_width_yields_nothing_rather_than_looping_forever(self):
        assert ing._slices(date(2024, 1, 1), date(2024, 12, 31), 0) == []
        assert ing._slices(date(2024, 1, 1), date(2024, 12, 31), -5) == []

    def test_the_default_is_sixty_and_it_is_the_mlb_number(self):
        assert ing.DEFAULT_SLICE_DAYS == 60
        assert ing.SLICE_DAYS_ENV == "NBA_SLICE_DAYS"

    def test_the_width_is_overridable_for_a_host_that_needs_less(self, monkeypatch):
        monkeypatch.setenv(ing.SLICE_DAYS_ENV, "7")
        assert ing._int_env(ing.SLICE_DAYS_ENV, ing.DEFAULT_SLICE_DAYS) == 7

    def test_every_request_is_one_window_no_wider_than_the_slice(self):
        units = ing._season_log_units(date(2024, 1, 1), date(2024, 12, 31), 60)
        assert units
        for _label, _season_type, _game_type, lo, hi in units:
            assert (hi - lo).days + 1 <= 60
            assert lo <= hi

    def test_both_season_types_are_pulled_for_every_window(self):
        units = ing._season_log_units(date(2024, 1, 1), date(2024, 3, 31), 60)
        assert {u[1] for u in units} == {src.SEASON_TYPE_REGULAR,
                                        src.SEASON_TYPE_PLAYOFFS}
        regular = [u for u in units if u[1] == src.SEASON_TYPE_REGULAR]
        assert len(regular) == len({(u[3], u[4]) for u in regular})

    def test_the_cache_key_carries_the_window(self, monkeypatch, tmp_path):
        """A whole-season file and a slice are not interchangeable."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        one = ing._season_log_path("2024-25", src.SEASON_TYPE_REGULAR,
                                   date(2024, 1, 1), date(2024, 3, 1))
        two = ing._season_log_path("2024-25", src.SEASON_TYPE_REGULAR,
                                   date(2024, 3, 2), date(2024, 5, 1))
        assert one != two
        assert "20240101_20240301" in one.name

    def test_the_query_carries_the_window_in_the_format_the_endpoint_wants(self):
        query = dict(urllib.parse.parse_qsl(
            src.season_log_query("2024-25", src.SEASON_TYPE_REGULAR,
                                 date(2024, 1, 15), date(2024, 3, 14))))
        assert query["DateFrom"] == "01/15/2024"
        assert query["DateTo"] == "03/14/2024"
        assert query["Season"] == "2024-25"

    def test_an_unsliced_query_is_still_the_whole_season(self):
        """The preflight probe relies on the empty default."""
        query = dict(urllib.parse.parse_qsl(
            src.season_log_query("2024-25", src.SEASON_TYPE_REGULAR),
            keep_blank_values=True))
        assert query["DateFrom"] == "" and query["DateTo"] == ""

    def test_the_cache_only_reader_reads_the_keys_the_pull_writes(
            self, monkeypatch, tmp_path):
        """A reader on different keys reports a warm cache as empty."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        start, end = date(2024, 1, 1), date(2024, 6, 30)
        units = ing._season_log_units(start, end, 60)
        for number, (label, season_type, game_type, lo, hi) in enumerate(units):
            ing._write_parquet(
                pd.DataFrame({"nba_game_id": [f"00{number:04d}"],
                              "gameday": [pd.Timestamp(lo)],
                              "team": ["BOS"],
                              "game_type": [game_type]}),
                ing._season_log_path(label, season_type, lo, hi))
        assert len(ing._read_logs_only(start, end)) == len(units)

    def test_a_cached_empty_window_is_not_asked_for_again(
            self, monkeypatch, tmp_path):
        """``not cached.empty`` would re-ask forever for a window nobody played."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        start, end = date(2024, 1, 1), date(2024, 6, 30)
        for label, season_type, _gt, lo, hi in ing._season_log_units(start, end, 60):
            ing._write_parquet(pd.DataFrame({"nba_game_id": []}),
                               ing._season_log_path(label, season_type, lo, hi))

        def explode(*_a, **_k):
            raise AssertionError("a warm cache must not reach the network")

        monkeypatch.setattr(ing, "http_json", explode)
        # Nothing to return and nothing to fail: the point is that it returns.
        assert ing._fetch_season_logs(start, end).empty


class TestSilentFallback:
    """ESPN answering a question nobody asked.

    Asked for a date range, ESPN's scoreboard either returns HTTP 400 or
    ignores the date and answers with the current slate - measured 2026-09-26,
    where a request spanning January 2024 came back 200 with one game dated
    October 2026. Swept across a window that is a schedule that looks complete
    and is wrong, with no error raised anywhere. That is what a 60-day schedule
    sweep would have produced, which is why it is caught and not merely avoided.
    """

    @staticmethod
    def _frame(*days: str) -> pd.DataFrame:
        return pd.DataFrame({"gameday": [pd.Timestamp(d) for d in days],
                             "game_id": [f"g{i}" for i in range(len(days))]})

    def test_a_response_from_another_year_is_caught(self):
        assert ing._answered_a_different_day(
            self._frame("2026-10-03"), date(2024, 1, 1)) == date(2026, 10, 3)

    def test_the_eastern_rollover_is_not_caught(self):
        """ESPN's date is UTC and the frame is Eastern, so a late game
        legitimately lands on the next day. Catching that would condemn a
        correct answer, which is the failure mode a stricter check creates."""
        assert ing._answered_a_different_day(
            self._frame("2024-01-15", "2024-01-16"), date(2024, 1, 15)) is None
        assert ing._answered_a_different_day(
            self._frame("2024-01-14"), date(2024, 1, 15)) is None

    def test_one_stray_game_does_not_condemn_an_otherwise_right_answer(self):
        assert ing._answered_a_different_day(
            self._frame("2024-01-15", "2024-03-20"), date(2024, 1, 15)) is None

    def test_a_genuinely_empty_day_is_still_empty(self):
        assert ing._answered_a_different_day(
            pd.DataFrame(), date(2024, 1, 15)) is None

    def test_the_sweep_stops_instead_of_caching_someone_elses_answer(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(ing.SCHEDULE_MAX_FAILURES_ENV, "2")
        ing._PROBED.add("espn")
        try:
            frame = self._frame("2026-10-03")
            monkeypatch.setattr(ing.sources, "espn_schedule_frame",
                                lambda _events: frame)
            monkeypatch.setattr(ing, "http_json",
                                lambda *_a, **_k: {"events": [{}]})
            with pytest.raises(ing.ScheduleUnavailable) as excinfo:
                ing._fetch_schedule(date(2024, 1, 1), date(2024, 1, 5))
        finally:
            ing._PROBED.discard("espn")
        assert "2026-10-03" in str(excinfo.value)
        # And nothing was written to the cache, so the next run does not read
        # the wrong answer back as if it were this window's schedule.
        assert not list((tmp_path / "schedule").glob("*.parquet"))


class TestProgressIsVisible:
    """The run that produced no log at all.

    A captured stream - a Kaggle cell, a pipe - drew no bar, and the code
    suppressed the *count* along with it, so a run spent ten minutes walking
    1,024 schedule days saying nothing, which is indistinguishable from a hang.
    The bar now draws into a capture exactly as MLB's does, so what is left to
    test is the state where ``tqdm`` genuinely is not installed: there the
    heartbeat line is the whole report, and it has to carry the count, the
    rate and the ETA on its own.
    """

    @pytest.fixture
    def _captured(self, monkeypatch):
        """Force the no-tqdm state, and speed the heartbeat up."""
        monkeypatch.setattr(ing.progress, "_tqdm", lambda: None)
        monkeypatch.setattr(ing.progress._Counter, "HEARTBEAT_SEC", 0.0)
        yield

    def test_work_in_progress_is_reported_without_a_terminal(
            self, _captured, caplog):
        with caplog.at_level("INFO", logger="nba_progress"):
            with ing.progress.track(10, desc="things", unit="thing") as bar:
                for _ in range(5):
                    bar.update(1)
        assert "things: 5/10 (50%)" in caplog.text

    def test_the_heartbeat_carries_a_rate_and_an_eta(self, _captured, caplog):
        with caplog.at_level("INFO", logger="nba_progress"):
            with ing.progress.track(4, desc="games", unit="game") as bar:
                bar.update(1)
        assert "1/4 (25%)" in caplog.text
        assert "game/s" in caplog.text
        assert "eta" in caplog.text

    def test_an_unknown_total_still_reports_its_count(self, _captured, caplog):
        with caplog.at_level("INFO", logger="nba_progress"):
            with ing.progress.track(None, desc="open") as bar:
                bar.update(1)
        assert "open: 1" in caplog.text

    def test_the_bar_can_be_switched_off_outright(self, monkeypatch):
        monkeypatch.setenv(ing.progress.ENV, "0")
        assert not ing.progress.enabled()

    def test_it_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv(ing.progress.ENV, raising=False)
        assert ing.progress.enabled()

    def test_bars_stay_on_where_the_output_is_not_a_terminal(self, monkeypatch):
        """The Kaggle log that started this.

        MLB's bars animate inside a captured cell because ``tqdm`` writes to a
        non-terminal exactly as it writes to a terminal, and this backend had
        added a rule of its own that suppressed them there. If that rule comes
        back, the test that fails is this one and not a 70-minute run nobody
        was watching.
        """
        monkeypatch.delenv(ing.progress.ENV, raising=False)
        stream = _CapturedStream()
        monkeypatch.setattr(ing.progress.sys, "stderr", stream)
        with ing.progress.track(3, desc="days", unit="day") as bar:
            for _ in range(3):
                bar.update(1)
        drawn = stream.text()
        assert "3/3" in drawn
        assert "100%" in drawn
        assert "days" in drawn

    def test_every_phase_announces_itself(self, caplog):
        with caplog.at_level("INFO", logger="nba_progress"):
            bar = ing.progress.phases(("one", "two"))
            bar.advance("one")
            bar.advance("two")
            bar.close()
        assert "phase 1/2  one" in caplog.text
        assert "phase 2/2  two" in caplog.text

    def test_a_phase_bar_claims_no_rate_or_eta(self, _captured, caplog):
        """Ten uneven phases produce "0.0 phase/s, eta 6m03s" off one sample,
        which reads as a measurement and is not one."""
        with caplog.at_level("INFO", logger="nba_progress"):
            bar = ing.progress.phases(("quick", "slow", "later"))
            bar.advance("quick")
            bar.close()
        assert "quick: 1/3 (33%)" in caplog.text
        assert "phase/s" not in caplog.text
        assert "eta" not in caplog.text

    def test_a_bar_never_changes_what_the_caller_gets(self):
        """The guardrail: display only. Same items, same order, bar or no bar."""
        items = [3, 1, 2]
        assert list(ing.progress.wrap(iter(items), len(items), "x")) == items

    def test_the_count_follows_the_work_and_never_leads_it(self, _captured, caplog):
        """The 2026-09-26 Kaggle run closed a 1,024-day sweep on "1023 fetched"
        and then summarised "1024 fetched", because the bar was ticked at the
        top of the loop body. The count has to follow the work, and the closing
        line is the one that has to agree with the summary after it."""
        fetched = 0
        with caplog.at_level("INFO", logger="nba_progress"):
            with ing.progress.track(4, desc="days", unit="day") as bar:
                for _ in range(4):
                    with bar.item(lambda: f"{fetched} fetched"):
                        fetched += 1
        assert "days: 4/4 (100%)" in caplog.text
        assert "days: 4 of 4 days done (4 fetched)" in caplog.text

    def test_a_unit_that_continues_still_ticks_exactly_once(self, _captured, caplog):
        with caplog.at_level("INFO", logger="nba_progress"):
            with ing.progress.track(3, desc="days", unit="day") as bar:
                for _ in range(3):
                    with bar.item():
                        continue
        assert "days: 3/3 (100%)" in caplog.text

    def test_a_unit_that_fails_is_still_counted(self, _captured, caplog):
        """A failure is a completed attempt; hiding it would make a broken
        sweep look like a merely shorter one."""
        with caplog.at_level("INFO", logger="nba_progress"):
            with pytest.raises(ing.HostUnavailable):
                with ing.progress.track(2, desc="days", unit="day") as bar:
                    with bar.item():
                        raise ing.HostUnavailable("refused")
        assert "days: 1/2 (50%)" in caplog.text

    def test_a_unit_never_attempted_does_not_tick(self, _captured, caplog):
        """A budget break taken before the block leaves the count short of the
        total, which is the honest reading: those days were not asked about."""
        with caplog.at_level("INFO", logger="nba_progress"):
            with ing.progress.track(5, desc="days", unit="day") as bar:
                for index in range(5):
                    if index == 3:
                        break
                    with bar.item():
                        pass
        assert "days: 3/5 (60%)" in caplog.text

    def test_the_eta_reads_like_a_duration(self):
        assert ing.progress._eta(9) == "9s"
        assert ing.progress._eta(252) == "4m12s"
        assert ing.progress._eta(7200 + 300) == "2h05m"


class TestChunkedSweepBars:
    """MLB's per-chunk progress, measured against MLB's log.

    The reference is ``0%|  | 0/46 [00:00<?, ?it/s]`` becoming
    ``100%|...| 46/46 [00:52<00:00, 1.15s/it]`` between a ``Chunk: a -> b``
    line and an ``-> 164216 pitches`` line.  Three things have to be true for
    that to be a report of this pipeline's work rather than a decoration: the
    window has to be announced, the bar has to count the days in that window,
    and every tick has to be a day the sweep really walked.
    """

    def test_each_window_is_announced_and_counted(self, monkeypatch):
        stream = _CapturedStream()
        monkeypatch.setattr(ing.progress.sys, "stderr", stream)
        start = date(2024, 1, 1)
        with ing._DayWindows(start, start + timedelta(days=64), 60,
                             desc="schedule", unit="day") as windows:
            for offset in range(65):
                with windows.day(start + timedelta(days=offset)).item():
                    pass
        drawn = stream.text()
        # Sixty days, then the five that are left - one bar each, each
        # finished, which is the shape MLB's log has for every one of its
        # windows.
        assert "60/60" in drawn
        assert "5/5" in drawn
        assert "100%" in drawn

    def test_a_window_finishes_before_the_next_one_opens(self, monkeypatch):
        """MLB's log reads 46/46, then ``Chunk:``, then 0/60. Bars left open
        until the sweep ends put every window's 100% line in the wrong place
        and stack them on top of each other, so the order is the feature as
        much as the bar is."""
        stream = _CapturedStream()
        monkeypatch.setattr(ing.progress.sys, "stderr", stream)
        start = date(2024, 1, 1)
        with ing._DayWindows(start, start + timedelta(days=9), 5,
                             desc="schedule", unit="day") as windows:
            for offset in range(10):
                with windows.day(start + timedelta(days=offset)).item():
                    pass
        drawn = stream.text()
        first_closed = drawn.find("5/5")
        second_opened = drawn.find("0/5", first_closed)
        assert first_closed != -1 and second_opened != -1
        assert first_closed < second_opened
        # And nothing was left holding the cursor: one window, one bar, no
        # cursor-move escapes in a capture that a person has to read.
        assert "\x1b" not in drawn

    def test_a_closed_window_reports_what_it_walked(self, caplog):
        start = date(2024, 1, 1)
        with caplog.at_level("INFO", logger="nba_ingestion"):
            with ing._DayWindows(start, start + timedelta(days=1), 60,
                                 desc="schedule", unit="day") as windows:
                for offset in range(2):
                    with windows.day(start + timedelta(days=offset)).item():
                        windows.window_cached += 1
                        windows.window_games += 5
        assert "Chunk: 2024-01-01 -> 2024-01-02" in caplog.text
        assert "-> 10 game(s) over 2 day(s) (0 fetched, 2 from cache)" \
            in caplog.text

    def test_the_window_is_a_report_and_not_a_request(self, monkeypatch,
                                                      tmp_path):
        """The guardrail. Chunking is display; the days asked for are identical
        with it and without it, or the bar is measuring a different sweep than
        the one that runs."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(ing.FULL_REPULL_ENV, "1")
        ing._PROBED.add("espn")
        asked: list[date] = []
        monkeypatch.setattr(ing.sources, "espn_schedule_frame", lambda _e: pd.DataFrame())
        monkeypatch.setattr(ing, "http_json", lambda *_a, **_k: {"events": []})
        start = date(2024, 1, 1)
        try:
            for slice_days in ("7", "60", "10000"):
                asked.clear()
                for path in (tmp_path / "schedule").glob("*.parquet"):
                    path.unlink()
                monkeypatch.setenv(ing.SLICE_DAYS_ENV, slice_days)
                ing._fetch_schedule(start, start + timedelta(days=9))
                asked.extend(sorted(path.stem for path in
                                    (tmp_path / "schedule").glob("*.parquet")))
        finally:
            ing._PROBED.discard("espn")
        assert asked == [f"{start + timedelta(days=n):%Y%m%d}" for n in range(10)]

    def test_a_budget_break_closes_the_open_window(self, caplog):
        """A sweep that stops mid-window still reports the window it stopped
        in, rather than leaving the last bar undrawn and the count unsaid."""
        start = date(2024, 1, 1)
        with caplog.at_level("INFO", logger="nba_ingestion"):
            with ing._DayWindows(start, start + timedelta(days=59), 60,
                                 desc="schedule", unit="day") as windows:
                for offset in range(3):
                    with windows.day(start + timedelta(days=offset)).item():
                        windows.window_cached += 1
                    if offset == 2:
                        break  # the sweep gave up here
        assert "-> 0 game(s) over 60 day(s) (0 fetched, 3 from cache)" \
            in caplog.text

    def test_the_budget_still_stops_the_sweep_before_any_request(
            self, monkeypatch, tmp_path, caplog):
        """The windows are reporting, so they cannot be the reason a sweep runs
        long: a budget already spent still means zero requests.  The clock is
        moved rather than the budget set, because ``_float_env`` refuses a
        non-positive value and answers the default instead - a sweep cannot be
        made to overrun by typing a number, and this test leans on that rather
        than fighting it."""
        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        stamps = [1_000.0, 5_000.0]  # started, then the first budget check

        def clock() -> float:
            return stamps.pop(0) if len(stamps) > 1 else stamps[0]

        monkeypatch.setattr(ing.time, "time", clock)

        def explode(*_a, **_k):
            raise AssertionError("a spent budget must not buy a request")

        monkeypatch.setattr(ing, "http_json", explode)
        with caplog.at_level("INFO", logger="nba_ingestion"):
            ing._fetch_schedule(date(2024, 1, 1), date(2024, 3, 1))
        assert "hit its budget after 0 of 61 days" in caplog.text
        assert not list((tmp_path / "schedule").glob("*.parquet"))

    def test_a_zero_budget_is_the_default_and_not_an_off_switch(self, monkeypatch):
        """Worth pinning because it reads the other way: someone limiting a run
        to no time at all would type 0, and get the full 30 minutes instead of
        a sweep that stops immediately. The hardening is right - a budget is
        never accidentally unbounded - but it is silent, so it is a test."""
        monkeypatch.delenv(ing.SCHEDULE_BUDGET_ENV, raising=False)
        for value in ("0", "-5", "", "nonsense"):
            monkeypatch.setenv(ing.SCHEDULE_BUDGET_ENV, value)
            assert ing._float_env(ing.SCHEDULE_BUDGET_ENV,
                                  ing.DEFAULT_SCHEDULE_BUDGET_SEC) \
                == ing.DEFAULT_SCHEDULE_BUDGET_SEC


class TestEventRollupArchive:
    """The per-game play-by-play cache is machine-local and the production
    host is ephemeral: the 2026-09-29 20:13 run re-fetched the same 1,500
    newest games as every run before it ("1500 fetched, 0 from cache") while
    1,962 games from 2024-01-01 to 2025-04-01 stayed forward-filled forever.
    The rollup archive ships the accumulated team-game counts in the delivery
    (the way nba_designations.parquet already ships the PIT archive), so one
    sweep's work is permanent for every run after it, on any machine."""

    @staticmethod
    def _rows():
        games = []
        for day in range(1, 9):
            games.append({"game_id": f"g{day}", "season": 2024,
                          "gameday": f"2024-01-{day:02d}", "game_type": 1,
                          "home_team": "BOS", "away_team": "NYK",
                          "home_score": 110.0, "away_score": 100.0})
        return pd.DataFrame(games)

    @staticmethod
    def _rollup_rows(game_ids, team="BOS"):
        return pd.DataFrame([{"game_id": gid, "team": team, "rim_attempts": 40,
                              "possessions": 100.0}
                             for gid in game_ids])

    def test_an_absent_archive_degrades_to_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DELIVERY_DIR", tmp_path)
        frame, added = ing._absorb_event_rollup_archive(
            self._rollup_rows(["g1"]), self._rows())
        assert added == [] and len(frame) == 1

    def test_the_archive_fills_games_the_sweep_did_not_cover(self, tmp_path,
                                                            monkeypatch):
        monkeypatch.setattr(config, "DATA_DELIVERY_DIR", tmp_path)
        # g1..g4 were swept this run; the archive carries g5..g8 from a
        # previous run's sweep of the same window.
        fresh = self._rollup_rows(["g1", "g2", "g3", "g4"])
        archive = self._rollup_rows(["g5", "g6", "g7", "g8"])
        archive.to_parquet(tmp_path / ing.EVENT_ROLLUP_ARCHIVE, index=False)
        union, added = ing._absorb_event_rollup_archive(fresh, self._rows())
        # The provenance half of the 2026-10-08 audit finding: WHICH
        # team-games came from the archive, at team-game grain, sorted.
        assert added == [f"g{i}|BOS" for i in range(5, 9)]
        assert set(union.game_id) == {f"g{i}" for i in range(1, 9)}
        assert not union.duplicated(["game_id", "team"]).any()

    def test_fresh_rollups_win_over_archived_ones(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DELIVERY_DIR", tmp_path)
        fresh = self._rollup_rows(["g1"])
        stale = self._rollup_rows(["g1"]).assign(rim_attempts=99)
        stale.to_parquet(tmp_path / ing.EVENT_ROLLUP_ARCHIVE, index=False)
        union, added = ing._absorb_event_rollup_archive(fresh, self._rows())
        assert added == []
        assert (union.loc[union.game_id == "g1", "rim_attempts"] == 40).all()

    def test_the_absorbed_keys_are_team_game_grain_not_game_grain(self, tmp_path,
                                                                  monkeypatch):
        """A swept game's OTHER team can still come from the archive, so the
        provenance must be keyed (game_id, team) - an archive row for the
        opponent of a swept game is absorbed, not skipped."""
        monkeypatch.setattr(config, "DATA_DELIVERY_DIR", tmp_path)
        fresh = pd.DataFrame([{"game_id": "g1", "team": "BOS",
                               "rim_attempts": 40, "possessions": 100.0},
                              {"game_id": "g2", "team": "BOS",
                               "rim_attempts": 41, "possessions": 101.0}])
        archive = pd.DataFrame([{"game_id": "g1", "team": "NYK",
                                 "rim_attempts": 38, "possessions": 99.0},
                                {"game_id": "g2", "team": "NYK",
                                 "rim_attempts": 39, "possessions": 98.0}])
        archive.to_parquet(tmp_path / ing.EVENT_ROLLUP_ARCHIVE, index=False)
        union, added = ing._absorb_event_rollup_archive(fresh, self._rows())
        assert added == ["g1|NYK", "g2|NYK"]
        assert len(union) == 4
        assert not union.duplicated(["game_id", "team"]).any()

    def test_an_archive_game_outside_the_window_is_dropped(self, tmp_path,
                                                          monkeypatch):
        monkeypatch.setattr(config, "DATA_DELIVERY_DIR", tmp_path)
        archive = self._rollup_rows(["g99"]).assign(team="X")
        archive.to_parquet(tmp_path / ing.EVENT_ROLLUP_ARCHIVE, index=False)
        union, added = ing._absorb_event_rollup_archive(
            self._rollup_rows(["g1"]), self._rows())
        assert added == [] and "g99" not in set(union.game_id)

    def test_a_rollup_dedupes_on_reread(self, tmp_path):
        frame = pd.concat([self._rollup_rows(["g1"]),
                           self._rollup_rows(["g1"]).assign(rim_attempts=7)])
        frame.to_parquet(tmp_path / ing.EVENT_ROLLUP_ARCHIVE, index=False)
        reread = ing.read_event_rollup_archive(tmp_path)
        assert len(reread) == 1 and reread.rim_attempts.iloc[0] == 7


class TestPositionsCacheVersioning:
    """The positions table gained ``positions`` - the feed's full listing,
    which is what the team segments read. The file is versioned in its NAME
    rather than keyed on the shape (the roster precedent): a v1 table silently
    read as v2 would starve every F-C segment back to NaN while every other
    check passed, because the single-label frame is perfectly well-formed.

    Everything here runs offline - the point is the versioning and the
    outage fallback, not the feed.
    """

    def test_the_cache_key_carries_the_version(self):
        assert ing._positions_path("2025-26").name == "positions_v3_2025-26.parquet"
        assert (ing._positions_path_v2("2025-26").name
                == "positions_v2_2025-26.parquet")
        # The single-label table stays readable under its old name, which is
        # what makes it usable as the last outage fallback.
        assert ing._positions_path_v1("2025-26").name == "positions_2025-26.parquet"

    def test_a_live_pull_writes_the_versioned_file_with_the_listing(
            self, tmp_path, monkeypatch):
        import urllib.parse

        def payload(url, *_args, **_kwargs):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if "PlayerPosition" not in query:
                # The index answering for a player the filter never lists him
                # under is covered separately; here it stays silent so the
                # convention is what this test measures.
                return {"resultSets": []}
            which = query["PlayerPosition"][0]
            ids = {"G": [10], "F": [10, 20], "C": [20]}[which]
            return {"resultSets": [{"headers": ["PLAYER_ID"],
                                    "rowSet": [[i] for i in ids]}]}

        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(ing, "http_json", payload)
        frame = ing._fetch_positions("2025-26")
        # Player 20 is listed F-C: one collapsed cell for the prior...
        assert dict(zip(frame.player_id, frame.position)) == {10: "G", 20: "F"}
        # ...and the full listing for the segments.
        assert dict(zip(frame.player_id, frame.positions)) == {
            10: "G|F", 20: "F|C"}
        assert ing._positions_path("2025-26").exists()
        # A warm cache reads the same shape back without touching the feed.
        again = ing._fetch_positions("2025-26")
        assert dict(zip(again.player_id, again.positions)) == {
            10: "G|F", 20: "F|C"}

    def test_a_feed_outage_falls_back_to_the_single_label_table(
            self, tmp_path, monkeypatch):
        """A dead stats.nba.com degrades the segments to today's coverage
        instead of dropping the whole ``pl_rapm_*`` family to NaN."""
        cached = tmp_path / "positions"
        cached.mkdir(parents=True)
        pd.DataFrame({"player_id": [1], "position": ["C"]}).to_parquet(
            cached / ing._positions_path_v1("2025-26").name, index=False)

        def dead(*_args, **_kwargs):
            raise TimeoutError("the read operation timed out")

        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(ing, "http_json", dead)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        frame = ing._fetch_positions("2025-26")
        # Single-label shape: the caller cannot tell which cache served it,
        # and does not need to - the segment code reads the listing when
        # it exists and the collapsed cell when it does not.
        assert list(frame.columns) == ["player_id", "position"]
        assert list(frame.position) == ["C"]
        assert not ing._positions_path("2025-26").exists()


#: A ``playerindex`` payload naming player 20 a ``C-F`` - the league's own
#: reading, and the disagreement the override exists to honor. Ten's ``G``
#: agrees with the convention, so it is inert; 20's is the whole test.
_INDEX_PAYLOAD = {"resultSets": [{"headers": ["PERSON_ID", "POSITION"],
                                  "rowSet": [[10, "G"], [20, "C-F"]]}]}


class TestPositionPrimaryIndex:
    """The collapsed cell follows the league's own primary letter.

    ``playerindex`` is the only endpoint that publishes a position at all, and
    it publishes the code ORDERED - ``C-F`` and ``F-C`` are the same two
    letters in the league's order of preference. The local G->F->C convention
    cannot see that difference (an unordered set has no order to read) and
    disagrees with the league for 28 of 582 players in 2025-26, 17 of them
    ``C-F`` players who are actually starting centres.

    Everything here runs offline; the live coverage numbers in the coverage
    gate below were measured once against the real endpoint and are what the
    gate is set from.
    """

    def test_a_full_index_moves_the_cell_the_convention_got_wrong(
            self, tmp_path, monkeypatch):
        import urllib.parse

        def payload(url, *_args, **_kwargs):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if "PlayerPosition" not in query:
                return _INDEX_PAYLOAD
            ids = {"G": [10], "F": [10, 20], "C": [20]}[query["PlayerPosition"][0]]
            return {"resultSets": [{"headers": ["PLAYER_ID"],
                                    "rowSet": [[i] for i in ids]}]}

        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(ing, "http_json", payload)
        frame = ing._fetch_positions("2025-26")
        # 20 was F under the convention and is C by the league's reading.
        assert dict(zip(frame.player_id, frame.position)) == {10: "G", 20: "C"}
        # The listing is NOT rewritten - the segments ask who the club fields
        # at centre, which is indifferent to which of them is primary.
        assert dict(zip(frame.player_id, frame.positions)) == {
            10: "G|F", 20: "F|C"}

    def test_a_truncated_index_is_refused_whole(self, tmp_path, monkeypatch):
        """The endpoint degrades SILENTLY on a back season, returning 200.

        Measured live 2026-10-02: 100% of 2025-26's filter players are in the
        index, but only 23% of 2024-25's and 24% of 2023-24's. Applying that
        quarter would key one season's prior cells on two conventions at once,
        so below the floor the index is dropped and the season keeps the
        convention wholesale - partial is not an option here.
        """
        import urllib.parse

        def payload(url, *_args, **_kwargs):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if "PlayerPosition" not in query:
                # Names only the first of ten players: the shape a back season
                # returns, at 10% rather than the real ~24%.
                return {"resultSets": [{"headers": ["PERSON_ID", "POSITION"],
                                        "rowSet": [[10, "G"]]}]}
            ids = {"G": [10], "F": [10, 20], "C": [20]}[query["PlayerPosition"][0]]
            return {"resultSets": [{"headers": ["PLAYER_ID"],
                                    "rowSet": [[i] for i in ids]}]}

        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(ing, "http_json", payload)
        frame = ing._fetch_positions("2024-25")
        # Convention stands for everyone, including the one the index covered.
        assert dict(zip(frame.player_id, frame.position)) == {10: "G", 20: "F"}

    def test_an_unreachable_index_leaves_the_convention_in_charge(
            self, tmp_path, monkeypatch):
        """The listing and the collapsed cell are INDEPENDENT.

        The index is what decides the cell; the filter pull is what produces
        the listing the nine features actually read. So an index outage must
        cost the override and nothing else - not the listing, and not the
        features.
        """
        import urllib.parse

        def payload(url, *_args, **_kwargs):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if "PlayerPosition" not in query:
                raise TimeoutError("the read operation timed out")
            ids = {"G": [10], "F": [10, 20], "C": [20]}[query["PlayerPosition"][0]]
            return {"resultSets": [{"headers": ["PLAYER_ID"],
                                    "rowSet": [[i] for i in ids]}]}

        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(ing, "http_json", payload)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        frame = ing._fetch_positions("2025-26")
        assert dict(zip(frame.player_id, frame.position)) == {10: "G", 20: "F"}
        assert dict(zip(frame.player_id, frame.positions)) == {
            10: "G|F", 20: "F|C"}

    def test_the_gate_refuses_a_densely_covered_but_truncated_index(self):
        """A coverage fraction, not a count.

        The interesting case is an index that covers MOST of the league and
        still misses some: 90% looks trustworthy and is not. The floor is the
        only thing standing between that and a season keyed on two
        conventions, so the comparison is asserted at the boundary rather than
        only through the live-shaped case above.
        """
        floor = config.PLAYER_EPM_PRIMARY_MIN_COVERAGE
        assert 0.24 < floor, "a back season's index covers ~24%; the floor must clear it"
        assert floor <= 1.0

    def test_the_index_cache_is_a_separate_file_from_the_positions_table(self):
        """Different endpoint, different failure mode.

        The collapsed cell is the only thing that depends on the index, so a
        lost index must not invalidate the listing the segments read. Separate
        files are what makes that independence mechanical rather than a
        promise.
        """
        assert ing._position_index_path("2025-26").parent == (
            ing._positions_path("2025-26").parent)
        assert ing._position_index_path("2025-26").name != (
            ing._positions_path("2025-26").name)

    def test_an_outage_prefers_the_listing_table_over_the_single_label_one(
            self, tmp_path, monkeypatch):
        """v2 before v1: the fallback should cost the cell, not the features.

        Both are outage artifacts and both are on disk after a real run, so the
        order is a decision. v2 still carries ``positions``, which is what the
        nine features are computed from; v1 has no listing at all.
        """
        cached = tmp_path / "positions"
        cached.mkdir(parents=True)
        pd.DataFrame({"player_id": [1], "position": ["C"],
                      "positions": ["C"]}).to_parquet(
            cached / ing._positions_path_v2("2025-26").name, index=False)

        def dead(*_args, **_kwargs):
            raise TimeoutError("the read operation timed out")

        monkeypatch.setenv(ing.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(ing, "http_json", dead)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        frame = ing._fetch_positions("2025-26")
        assert "positions" in frame.columns
        assert list(frame.position) == ["C"]

