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


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


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
        monkeypatch.setenv(ing.WINDOW_START_ENV, "2024-01-01")
        monkeypatch.setenv(ing.WINDOW_END_ENV, "2024-06-01")
        assert ing.window() == (date(2024, 1, 1), date(2024, 6, 1))

    def test_a_reversed_window_is_swapped(self, monkeypatch):
        monkeypatch.setenv(ing.WINDOW_START_ENV, "2024-06-01")
        monkeypatch.setenv(ing.WINDOW_END_ENV, "2024-01-01")
        assert ing.window() == (date(2024, 1, 1), date(2024, 6, 1))

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
        team = pd.DataFrame([{"game_id": f"g{i}", "team": t, "points_for": 1.0,
                              "points_against": 1.0}
                             for i in range(60) for t in ("BOS", "NYK")])
        player = pd.DataFrame([{"game_id": f"g{i}", "player_id": str(p),
                                "team": "BOS"}
                               for i in range(60) for p in range(5)])
        facts = ing.NBAFacts(games=games, team_stats=team, player_stats=player,
                             team_events=pd.DataFrame(), play_by_play=pd.DataFrame(),
                             team_names={})
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
