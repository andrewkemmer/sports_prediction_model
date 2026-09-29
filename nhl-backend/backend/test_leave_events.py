"""Leave-of-absence channel + availability-aware goalie features.

Covers:
  * the ledger loader (absent/unreadable/malformed/refused rows — never
    fabricated, never silently dropped),
  * build_leave_stints (announcement-instant intervals, id/name resolution,
    open/closed leaves, attrs contract),
  * combine_stints (injury + leave union preserving the attrs contract),
  * names_match_player (boxscore abbreviation vs ledger full name),
  * the AVAILABILITY-AWARE expected starter: an Out/IR or on-leave goalie
    cannot win the workload vote once a snapshot/announcement strictly
    precedes the game; the fallback serves the backup's OWN strictly-prior
    form; pre-archive history stays NaN, never inferred healthy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import config  # noqa: E402
import features as feat  # noqa: E402
import ingestion as ing  # noqa: E402
import injury_stints as ist  # noqa: E402


def _patch_dd(monkeypatch, tmp_path):
    """Patch the config module the PRODUCTION modules actually hold (under
    pytest the backend package imports as backend.*; the test file's own
    `import config` is a different module object)."""
    monkeypatch.setattr(ing.config, "DATA_DELIVERY_DIR", tmp_path)


def _boxscores(rows: list[dict]) -> pd.DataFrame:
    cols = ["game_id", "gameday", "home_team", "away_team",
            "home_goalie_id", "away_goalie_id",
            "home_goalie_name", "away_goalie_name",
            "home_goalie_toi", "away_goalie_toi",
            "home_goals_against", "away_goals_against",
            "home_shots_against", "away_shots_against"]
    return pd.DataFrame([{**{c: None for c in cols}, **r} for r in rows])


def _games(rows: list[dict]) -> pd.DataFrame:
    cols = ["game_id", "gameday", "home_team", "away_team"]
    return pd.DataFrame([{**{c: None for c in cols}, **r} for r in rows])


def _two_goalie_boxscores() -> pd.DataFrame:
    """WPG: starter C. Hellebuyck through 11-18, then E. Comrie while the
    starter is away; Skinner-style PIT backup for CGY. Goalie ids are
    goalie-specific (production ids come from the NHL API and never alias
    two men), and every boxscored game appears in the team map."""
    def row(gid, day, home, away, hg, ag, hid, aid, htoi=60.0, atoi=60.0,
            hga=2.0, aga=3.0, hs=30.0, as_=30.0):
        return {"game_id": gid, "gameday": day, "home_team": home,
                "away_team": away,
                "home_goalie_id": hid, "away_goalie_id": aid,
                "home_goalie_name": hg, "away_goalie_name": ag,
                "home_goalie_toi": htoi, "away_goalie_toi": atoi,
                "home_goals_against": hga, "away_goals_against": aga,
                "home_shots_against": hs, "away_shots_against": as_}
    return _boxscores([
        row("G1", "2025-11-10", "WPG", "TOR", "C. Hellebuyck", "J. Woll",
            "g-helle", "g-woll"),
        row("G2", "2025-11-13", "WPG", "CGY", "C. Hellebuyck",
            "S. Skinner", "g-helle", "g-skinner"),
        row("G3", "2025-11-15", "WPG", "VAN", "E. Comrie", "K. Lankinen",
            "g-comrie", "g-lankinen"),
        row("G4", "2025-11-18", "WPG", "SEA", "C. Hellebuyck",
            "P. Grubauer", "g-helle", "g-grubauer"),
        row("G5", "2025-11-21", "WPG", "EDM", "E. Comrie", "S. Skinner",
            "g-comrie", "g-skinner"),
        row("G6", "2025-11-26", "WPG", "CGY", "E. Comrie", "S. Skinner",
            "g-comrie", "g-skinner"),
    ])


def _team_map() -> pd.DataFrame:
    """The full schedule map (production passes ``sched``), covering every
    boxscored game so historical starts resolve their team labels."""
    return pd.DataFrame([
        {"game_id": "G1", "gameday": "2025-11-10", "home_team": "WPG",
         "away_team": "TOR"},
        {"game_id": "G2", "gameday": "2025-11-13", "home_team": "WPG",
         "away_team": "CGY"},
        {"game_id": "G3", "gameday": "2025-11-15", "home_team": "WPG",
         "away_team": "VAN"},
        {"game_id": "G4", "gameday": "2025-11-18", "home_team": "WPG",
         "away_team": "SEA"},
        {"game_id": "G5", "gameday": "2025-11-21", "home_team": "WPG",
         "away_team": "EDM"},
        {"game_id": "G6", "gameday": "2025-11-26", "home_team": "WPG",
         "away_team": "CGY"},
    ])


def test_ledger_absent_is_honest_empty(tmp_path, monkeypatch):
    _patch_dd(monkeypatch, tmp_path)
    out = ing.load_leave_events()
    assert len(out) == 0
    assert "announced_at_utc" in out.columns


def test_ledger_unparseable_is_treated_as_absent(tmp_path, monkeypatch):
    _patch_dd(monkeypatch, tmp_path)
    (tmp_path / ing.config.LEAVE_EVENTS_LEDGER).write_text(
        "{not json", encoding="utf-8")
    assert len(ing.load_leave_events()) == 0


def test_ledger_refuses_rows_without_parseable_announcement(tmp_path,
                                                            monkeypatch,
                                                            caplog):
    _patch_dd(monkeypatch, tmp_path)
    (tmp_path / ing.config.LEAVE_EVENTS_LEDGER).write_text(
        json.dumps({"events": [
            {"player_name": "Ok Player", "team": "BUF",
             "announced_at_utc": "2025-11-07T17:00:00Z"},
            {"player_name": "Bad Player", "team": "BUF",
             "announced_at_utc": "not-a-date"},
        ]}), encoding="utf-8")
    out = ing.load_leave_events()
    assert len(out) == 1
    assert out.iloc[0]["player_name"] == "Ok Player"
    assert any("refused" in r.message for r in caplog.records)


def test_leave_intervals_replay_from_announcements(tmp_path, monkeypatch):
    _patch_dd(monkeypatch, tmp_path)
    (tmp_path / ing.config.LEAVE_EVENTS_LEDGER).write_text(
        json.dumps({"events": [
            {"player_name": "Rasmus Dahlin", "team": "BUF",
             "announced_at_utc": "2025-11-07T17:00:00Z"},
        ]}), encoding="utf-8")
    (tmp_path / config.LEAVE_EVENTS_LEDGER).write_text(
        json.dumps({"events": [
            {"player_name": "Rasmus Dahlin", "team": "BUF",
             "announced_at_utc": "2025-11-07T17:00:00Z"},
        ]}), encoding="utf-8")
    events = ing.load_leave_events()
    ratings = pd.DataFrame([
        {"player_id": "mp-dahlin", "player_name": "Rasmus Dahlin"},
    ])
    stints, audit = ist.build_leave_stints(events, ratings)
    assert audit["events"] == 1 and audit["intervals"] == 1
    assert audit["resolved_by_name"] == 1
    assert stints.attrs["snapshot_based"] is True
    assert len(stints.attrs["snapshot_times"]) == 1
    # Open interval: excluded from the announcement instant on.
    assert ist.is_unavailable(stints, "mp-dahlin", "2025-11-08T23:00:00Z",
                              strict_start=True)
    assert ist.is_unavailable(stints, "mp-dahlin", "2026-01-15T00:00:00Z",
                              strict_start=True)
    # And NOT excluded the evening before the announcement.
    assert not ist.is_unavailable(stints, "mp-dahlin", "2025-11-06T23:00:00Z",
                                  strict_start=True)


def test_names_match_player_abbreviations():
    assert ist.names_match_player("C. Hellebuyck", "Connor Hellebuyck")
    assert ist.names_match_player("Connor Hellebuyck", "C. Hellebuyck")
    assert ist.names_match_player("connor hellebuyck", "Connor Hellebuyck")
    # An initial is inherently ambiguous (C. = Connor OR Cameron) — the
    # string-level test matches; the CALLER must require a unique full-name
    # match in the id space before acting on it.
    assert ist.names_match_player("C. Hellebuyck", "Cameron Hellebuyck")
    assert not ist.names_match_player("J. Woll", "Connor Hellebuyck")
    assert not ist.names_match_player("", "Connor Hellebuyck")


def test_ambiguous_initial_match_never_excludes_anyone():
    # Two full names share the boxscore's surname + initial: the caller must
    # refuse (no exclusion) rather than bench a possibly-wrong identity.
    bs = _two_goalie_boxscores()
    games = _games([
        {"game_id": "G5", "gameday": "2025-11-21",
         "home_team": "WPG", "away_team": "EDM"},
    ])
    stints = pd.DataFrame([
        {"player_id": "pid-connor", "player_name": "Connor Hellebuyck",
         "stint_start": "2025-11-19T20:00:00Z", "stint_end": pd.NaT},
        {"player_id": "pid-cameron", "player_name": "Cameron Hellebuyck",
         "stint_start": "2025-11-19T20:00:00Z", "stint_end": pd.NaT},
    ])
    stints.attrs["snapshot_based"] = True
    stints.attrs["snapshot_times"] = [pd.Timestamp("2025-11-19 20:00:00")]
    stints.attrs["window_end"] = pd.Timestamp("2025-11-19 20:00:00")
    per_game, _ = feat.goalie_state(bs, games, _team_map(), stints=stints)
    # Ambiguous: the workload winner stands, nobody is benched on a guess.
    assert per_game.loc[0, "g_home_name"] == "C. Hellebuyck"


def test_combine_stints_unions_and_keeps_snapshot_contract():
    inj = pd.DataFrame([
        {"player_id": "p1", "stint_start": "2025-11-14T15:00:00Z",
         "stint_end": pd.NaT}])
    inj.attrs["snapshot_based"] = True
    inj.attrs["snapshot_times"] = [pd.Timestamp("2025-11-14 15:00:00")]
    inj.attrs["window_end"] = pd.Timestamp("2025-11-14 15:00:00")
    leave = pd.DataFrame([
        {"player_id": "p2", "stint_start": "2025-11-07T17:00:00Z",
         "stint_end": "2025-11-15T00:00:00Z"}])
    leave.attrs["snapshot_based"] = True
    leave.attrs["snapshot_times"] = [pd.Timestamp("2025-11-07 17:00:00")]
    leave.attrs["window_end"] = pd.Timestamp("2025-11-07 17:00:00")
    out = ist.combine_stints(inj, leave)
    assert len(out) == 2
    assert out.attrs["snapshot_based"] is True
    assert [t for t in out.attrs["snapshot_times"]] == [
        pd.Timestamp("2025-11-07 17:00:00"), pd.Timestamp("2025-11-14 15:00:00")]
    assert ist.is_unavailable(out, "p1", "2025-11-15T00:00:00Z",
                              strict_start=True)
    assert ist.is_unavailable(out, "p2", "2025-11-10T00:00:00Z",
                              strict_start=True)
    assert not ist.is_unavailable(out, "p2", "2025-11-20T00:00:00Z",
                                  strict_start=True)


def test_injured_starter_cannot_win_the_workload_vote():
    bs = _two_goalie_boxscores()
    games = _games([
        {"game_id": "G5", "gameday": "2025-11-21",
         "home_team": "WPG", "away_team": "EDM"},
    ])
    # The starter went Out on 11-19, captured strictly before G5, keyed by
    # his own goalie id (production ids are goalie-specific).
    stints = pd.DataFrame([
        {"player_id": "g-helle", "stint_start": "2025-11-19T20:00:00Z",
         "stint_end": pd.NaT}])
    stints.attrs["snapshot_based"] = True
    stints.attrs["snapshot_times"] = [pd.Timestamp("2025-11-19 20:00:00")]
    stints.attrs["window_end"] = pd.Timestamp("2025-11-19 20:00:00")
    per_game, _ = feat.goalie_state(bs, games, _team_map(), stints=stints)
    # The BACKUP's own strictly-prior form is served, never the hurt man's.
    assert per_game.loc[0, "g_home_name"] == "E. Comrie"
    assert pd.notna(per_game.loc[0, "goalie_sv_pct_home"])
    assert pd.notna(per_game.loc[0, "goalie_gaa_home"])


def test_on_leave_starter_excluded_via_abbreviated_name_match():
    bs = _two_goalie_boxscores()
    games = _games([
        {"game_id": "G5", "gameday": "2025-11-21",
         "home_team": "WPG", "away_team": "EDM"},
    ])
    # Leave keyed by the LEDGER's full name; the boxscore says C. Hellebuyck.
    stints = pd.DataFrame([
        {"player_id": "connor-hellebuyck-ledger",
         "player_name": "Connor Hellebuyck",
         "stint_start": "2025-11-19T20:00:00Z", "stint_end": pd.NaT}])
    stints.attrs["snapshot_based"] = True
    stints.attrs["snapshot_times"] = [pd.Timestamp("2025-11-19 20:00:00")]
    stints.attrs["window_end"] = pd.Timestamp("2025-11-19 20:00:00")
    per_game, _ = feat.goalie_state(bs, games, _team_map(), stints=stints)
    # Unique full-name match in the id space: the exclusion applies.
    assert per_game.loc[0, "g_home_name"] == "E. Comrie"


def test_pre_announcement_game_still_serves_the_starter():
    bs = _two_goalie_boxscores()
    games = _games([
        {"game_id": "G4", "gameday": "2025-11-18",
         "home_team": "WPG", "away_team": "SEA"},
    ])
    stints = pd.DataFrame([
        {"player_id": "connor-hellebuyck-ledger",
         "player_name": "Connor Hellebuyck",
         "stint_start": "2025-11-19T20:00:00Z", "stint_end": pd.NaT}])
    stints.attrs["snapshot_based"] = True
    stints.attrs["snapshot_times"] = [pd.Timestamp("2025-11-19 20:00:00")]
    stints.attrs["window_end"] = pd.Timestamp("2025-11-19 20:00:00")
    per_game, _ = feat.goalie_state(bs, games, _team_map(), stints=stints)
    # 11-18 precedes the 11-19 exclusion: the starter keeps the vote.
    assert per_game.loc[0, "g_home_name"] == "C. Hellebuyck"


def test_post_capture_snapshot_cannot_retroactively_exclude():
    bs = _two_goalie_boxscores()
    games = _games([
        {"game_id": "G4", "gameday": "2025-11-18",
         "home_team": "WPG", "away_team": "SEA"},
    ])
    stints = pd.DataFrame([
        {"player_id": "g-helle", "stint_start": "2025-11-18T23:00:00Z",
         "stint_end": pd.NaT}])
    stints.attrs["snapshot_based"] = True
    stints.attrs["snapshot_times"] = [pd.Timestamp("2025-11-18 23:00:00")]
    stints.attrs["window_end"] = pd.Timestamp("2025-11-18 23:00:00")
    per_game, _ = feat.goalie_state(bs, games, _team_map(), stints=stints)
    # G4's decision (gameday-only resolution at midnight) precedes the
    # 23:00 capture: the starter cannot be retroactively benched.
    assert per_game.loc[0, "g_home_name"] == "C. Hellebuyck"


def test_no_stints_means_no_gate_and_prior_behaviour_holds():
    bs = _two_goalie_boxscores()
    games = _games([
        {"game_id": "G5", "gameday": "2025-11-21",
         "home_team": "WPG", "away_team": "EDM"},
    ])
    per_game, _ = feat.goalie_state(bs, games, _team_map(), stints=None)
    # Without any exclusion table the workload winner stands (Hellebuyck
    # has 3 prior starts to Comrie's 1 by 11-21).
    assert per_game.loc[0, "g_home_name"] == "C. Hellebuyck"


def test_all_candidates_excluded_serves_honest_nan():
    bs = _boxscores([
        {"game_id": "G1", "gameday": "2025-11-10", "home_team": "WPG",
         "away_team": "TOR", "home_goalie_id": "h-wpg",
         "away_goalie_id": "a-tor", "home_goalie_name": "C. Hellebuyck",
         "away_goalie_name": "J. Woll", "home_goalie_toi": 60.0,
         "away_goalie_toi": 60.0},
    ])
    games = _games([
        {"game_id": "G2", "gameday": "2025-11-21",
         "home_team": "WPG", "away_team": "TOR"},
    ])
    stints = pd.DataFrame([
        {"player_id": "g-helle", "stint_start": "2025-11-12T00:00:00Z",
         "stint_end": pd.NaT}])
    stints.attrs["snapshot_based"] = True
    stints.attrs["snapshot_times"] = [pd.Timestamp("2025-11-12 00:00:00")]
    stints.attrs["window_end"] = pd.Timestamp("2025-11-12 00:00:00")
    per_game, _ = feat.goalie_state(bs, games, _team_map(), stints=stints)
    # The only candidate is excluded and there is no fallback: NaN, never
    # the excluded man's stats.
    assert pd.isna(per_game.loc[0, "goalie_sv_pct_home"])
    assert pd.isna(per_game.loc[0, "goalie_gaa_home"])
