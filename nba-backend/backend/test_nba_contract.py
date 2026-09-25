"""Offline contract tests for the NBA API ingestion path.

Every network call is stubbed.  The suite pins the parts that decide whether a
run trains on the right data: how a season log becomes games, team facts, and
player lines, how play-by-play is cached incrementally, and which conditions
must fail the run loudly.
"""
from __future__ import annotations

import json
import logging
import sys
import urllib.error
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[0]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import ingestion as ing  # noqa: E402
import folds as folds_mod  # noqa: E402
import master_pipeline as mp  # noqa: E402
import progress as prog  # noqa: E402

# Ingestion binds its config as ``backend.config`` when the backend directory is
# a package, and as ``config`` otherwise.  Patching anything but the object it
# actually holds would let a test write to the real on-disk cache.
config = ing.config

TEAMS = list(config.NBA_TEAM_ID)
GAME_IDS = [f"00224000{i:02d}" for i in range(len(TEAMS))]


# --------------------------------------------------------------------------
# Fixtures shaped like the real payloads
# --------------------------------------------------------------------------


def season_log_rows(season_type: str = "Regular Season") -> list[dict]:
    """A season log covering every current team exactly once per game."""
    rows: list[dict] = []
    for i, home in enumerate(TEAMS):
        away = TEAMS[(i + 1) % len(TEAMS)]
        game_id = GAME_IDS[i]
        matchup = f"{away} vs. {home}" if season_type == "Regular Season" else f"{away} @ {home}"
        for side, team, base in (("home", home, 110 + i), ("away", away, 100 + i)):
            rows.append({
                "SEASON_ID": 22025, "PLAYER_ID": 2000 + i * 2 + (side == "away"),
                "PLAYER_NAME": f"Player {i}{side[0]}", "TEAM_ID": config.NBA_TEAM_ID[team],
                "TEAM_ABBREVIATION": team, "TEAM_NAME": f"{team} Club",
                "GAME_ID": game_id, "GAME_DATE": "2024-10-22T00:00:00",
                "MATCHUP": matchup, "WL": "W" if side == "home" else "L",
                "MIN": "36:12", "PTS": base, "FGM": 40, "FGA": 88, "FG3M": 12,
                "FG3A": 35, "FTM": 16, "FTA": 20, "OREB": 10, "DREB": 30,
                "REB": 40, "AST": 25, "TOV": 12, "STL": 7, "BLK": 4, "PF": 18,
                "PLUS_MINUS": 5.0,
            })
    return rows


def stub_season(monkeypatch, *, playoffs: bool = True) -> list[str]:
    """Stub the season log so a run needs no network.  Returns the calls made."""
    calls: list[str] = []

    def fake(season: str, season_type: str, pause: float) -> pd.DataFrame:
        calls.append(f"{season}|{season_type}")
        if season_type == ing.SEASON_TYPE_PLAYOFFS and not playoffs:
            return pd.DataFrame()
        return pd.DataFrame(season_log_rows(
            "Regular Season" if season_type == ing.SEASON_TYPE_REGULAR else "Playoffs"))

    monkeypatch.setattr(ing, "_fetch_season_log", fake)
    monkeypatch.setattr(ing, "_pull_play_by_play",
                        lambda games, start, end, path, *, enabled: pd.DataFrame())
    return calls


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own cache, a fixed window, and no network.

    The CDN fallback is off by default here and the socket is fenced, so a test
    that intends to exercise a failure path cannot quietly walk real game ids
    over the internet.  Tests opt back in by stubbing ``ing._get_json`` or the
    opener underneath it.
    """
    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setenv(ing.START_DATE_ENV, "2024-01-01")
    monkeypatch.setenv(ing.END_DATE_ENV, "2025-07-01")
    monkeypatch.setenv(ing.FULL_REPULL_ENV, "0")
    monkeypatch.setenv(ing.PLAY_BY_PLAY_ENV, "0")
    monkeypatch.setenv(ing.CDN_FALLBACK_ENV, "0")

    def fenced(*args, **kwargs):
        raise AssertionError(
            "unexpected network call: stub ing._get_json or "
            "ing.urllib.request.urlopen in this test")

    monkeypatch.setattr(ing.urllib.request, "urlopen", fenced)
    return cache


# --------------------------------------------------------------------------
# Windows and seasons
# --------------------------------------------------------------------------


def test_window_defaults_to_the_first_eligible_season(monkeypatch) -> None:
    monkeypatch.delenv(ing.START_DATE_ENV, raising=False)
    monkeypatch.delenv(ing.END_DATE_ENV, raising=False)
    start, end = ing._window()
    assert start == date(config.OOF_FIRST_SEASON, 1, 1)
    assert end == date.today()


def test_window_swaps_reversed_bounds(monkeypatch) -> None:
    monkeypatch.setenv(ing.START_DATE_ENV, "2025-07-01")
    monkeypatch.setenv(ing.END_DATE_ENV, "2024-01-01")
    start, end = ing._window()
    assert (start, end) == (date(2024, 1, 1), date(2025, 7, 1))


def test_seasons_overlap_the_window_without_asking_for_the_future() -> None:
    seasons = ing._seasons_in(date(2024, 1, 1), date(2026, 9, 25))
    assert seasons == ["2023-24", "2024-25", "2025-26", "2026-27"]
    assert "2027-28" not in seasons


def test_season_label_uses_the_july_boundary() -> None:
    assert ing._season_label(date(2024, 10, 22)) == "2024-25"
    assert ing._season_label(date(2025, 6, 22)) == "2024-25"
    assert ing._season_label(date(2024, 1, 15)) == "2023-24"


def test_matchup_sides_handle_both_official_spellings() -> None:
    assert ing._matchup_sides("DEN vs. MIN") == ("DEN", "MIN")
    assert ing._matchup_sides("BOS @ DEN") == ("BOS", "DEN")
    assert ing._matchup_sides("") is None
    assert ing._matchup_sides(None) is None


def test_minutes_parse_every_official_spelling() -> None:
    assert ing._minutes("36:12") == pytest.approx(36.2)
    assert ing._minutes("PT35M56.00S") == pytest.approx(35.9333, abs=1e-3)
    assert ing._minutes(34.5) == pytest.approx(34.5)
    assert np.isnan(ing._minutes("")) and np.isnan(ing._minutes(None))


# --------------------------------------------------------------------------
# Season log to normalized frames
# --------------------------------------------------------------------------


def test_season_log_becomes_games_with_sides_and_score() -> None:
    log = ing._prepare_log(pd.DataFrame(season_log_rows()), config.GAME_TYPE_REG)
    games = ing._games_frame(log)
    assert len(games) == len(TEAMS)
    assert set(games.season) == {2024.0}
    assert set(games.home_team) | set(games.away_team) == set(TEAMS)
    assert games.home_score.notna().all() and games.away_score.notna().all()
    assert (games.home_score > games.away_score).all()
    assert (games.margin == games.home_score - games.away_score).all()
    assert set(games.game_type) == {config.GAME_TYPE_REG}


def test_playoff_games_carry_the_postseason_type() -> None:
    raw = pd.DataFrame(season_log_rows("Playoffs"))
    log = ing._prepare_log(raw, config.GAME_TYPE_POST)
    games = ing._games_frame(log)
    assert set(games.game_type) == {config.GAME_TYPE_POST}
    assert games.game_id.nunique() == len(TEAMS)


def test_team_facts_cover_every_team_with_opponent_context() -> None:
    log = ing._prepare_log(pd.DataFrame(season_log_rows()), config.GAME_TYPE_REG)
    games = ing._games_frame(log)
    stats = ing._team_stats_frame(log, games)
    assert len(stats) == 2 * len(TEAMS)
    assert set(stats.team) == set(TEAMS)
    assert stats.is_home.sum() == len(TEAMS)
    assert (stats.points_against > 0).all()
    assert (stats.net_points == stats.points_for - stats.points_against).all()
    home = stats[stats.is_home].iloc[0]
    assert home.points_against == home.points_for - home.net_points
    assert "points_for" in stats.columns and "points" not in stats.columns
    assert stats.fg_pct.between(0, 1).all()


def test_player_lines_are_one_row_per_player_game() -> None:
    log = ing._prepare_log(pd.DataFrame(season_log_rows()), config.GAME_TYPE_REG)
    players = ing._player_stats_frame(log)
    assert len(players) == 2 * len(TEAMS)
    assert not players.duplicated(["game_id", "player_id"]).any()
    assert players.minutes.notna().all()
    assert set(players.team) == set(TEAMS)
    assert (players.points > 0).all()


def test_unplayed_or_unmatched_games_are_not_invented() -> None:
    raw = pd.DataFrame(season_log_rows())
    broken = raw[raw.MATCHUP != "LAL vs. LAL"].copy()
    broken.loc[broken.index[:2], "MATCHUP"] = "garbage"
    log = ing._prepare_log(broken, config.GAME_TYPE_REG)
    games = ing._games_frame(log)
    assert len(games) == len(TEAMS) - 1
    assert games.home_score.notna().all()


def test_team_names_reach_the_games_frame() -> None:
    log = ing._prepare_log(pd.DataFrame(season_log_rows()), config.GAME_TYPE_REG)
    games = ing._games_frame(log)
    names = (log[["team", "team_name"]].dropna().drop_duplicates("team")
             .set_index("team")["team_name"].astype(str).to_dict())
    games = ing._attach_team_names(games, names)
    assert games.home_team_name.ne(games.home_team).all()
    assert "LAL Club" in set(games.home_team_name)


# --------------------------------------------------------------------------
# Play-by-play
# --------------------------------------------------------------------------


def pbp_payload() -> dict:
    return {"game": {"actions": [
        {"actionId": "1", "sequenceNumber": "1", "period": 1,
         "gameClock": "12:00", "teamTricode": "BOS", "personId": 100,
         "actionType": "Normal", "subType": "Jump Shot", "descriptor": "Made",
         "scoreHome": 2, "scoreAway": 0, "pointsTotal": 2, "shotDistance": 24,
         "shotResult": "Made", "isFieldGoalAttempted": True, "isMade": True,
         "loc": "260.00,0.00,250.00"},
        {"actionId": "2", "sequenceNumber": "2", "period": 1,
         "gameClock": "11:30", "teamTricode": "LAL", "personId": 200,
         "actionType": "Normal", "subType": "Jump Shot", "descriptor": "Missed",
         "scoreHome": 2, "scoreAway": 0, "pointsTotal": 0, "shotDistance": 28,
         "shotResult": "Missed", "isFieldGoalAttempted": True, "isMade": False,
         "loc": ""},
    ]}}


def test_play_by_play_flattens_the_action_list() -> None:
    frame = ing._play_by_play_frame(pbp_payload(), "0022400001",
                                    pd.Timestamp("2024-10-22"))
    assert len(frame) == 2
    assert set(frame.game_id) == {"0022400001"}
    assert set(frame.team) == {"BOS", "LAL"}
    assert frame.period.tolist() == [1, 1]
    assert frame.points.tolist() == [2.0, 0.0]
    assert set(frame.action_type) == {"Normal"}


def test_play_by_play_tolerates_an_empty_or_missing_payload() -> None:
    assert ing._play_by_play_frame(None, "1", pd.Timestamp("2024-10-22")).empty
    assert ing._play_by_play_frame({"game": {}}, "1", pd.Timestamp("2024-10-22")).empty


def test_play_by_play_only_fetches_games_the_cache_has_not_seen(monkeypatch) -> None:
    games = pd.DataFrame({"game_id": ["a", "b", "c"],
                          "gameday": pd.to_datetime(["2024-11-01"] * 3)})
    cached = ing._play_by_play_frame(pbp_payload(), "a", games.gameday.iloc[0])
    ing._write_cache({"play_by_play": cached},
                     {"play_by_play": ing._cache_paths()[3]},
                     {"play_by_play": ["game_id", "action_id"]})
    asked: list[str] = []

    def fake(url, **kwargs):
        if "playbyplay_" in url:
            asked.append(url.rsplit("_", 1)[-1].replace(".json", ""))
        return pbp_payload()

    monkeypatch.setattr(ing, "_get_json", fake)
    monkeypatch.setattr(ing, "date", type("D", (), {
        "today": staticmethod(lambda: date(2024, 11, 5))}))
    out = ing._pull_play_by_play(games, date(2024, 11, 1), date(2024, 11, 5),
                                 ing._cache_paths()[3], enabled=True)
    assert set(asked) == {"b", "c"}
    assert set(out.game_id) == {"a", "b", "c"}


def test_play_by_play_re_pulls_the_trailing_window(monkeypatch) -> None:
    """A game cached while it was live must be re-fetched, not kept partial."""
    recent = pd.Timestamp("2024-10-30")
    games = pd.DataFrame({"game_id": ["a"], "gameday": [recent]})
    ing._write_cache({"play_by_play": ing._play_by_play_frame(
        pbp_payload(), "a", recent)},
        {"play_by_play": ing._cache_paths()[3]},
        {"play_by_play": ["game_id", "action_id"]})
    asked: list[str] = []
    monkeypatch.setattr(ing, "_get_json", lambda url, **kw: (
        asked.append(url), pbp_payload())[1])
    monkeypatch.setattr(ing, "date", type("D", (), {
        "today": staticmethod(lambda: date(2024, 10, 31))}))
    ing._pull_play_by_play(games, recent.date(), date(2024, 10, 31),
                           ing._cache_paths()[3], enabled=True)
    assert len(asked) == 1


# --------------------------------------------------------------------------
# HTTP behaviour
# --------------------------------------------------------------------------


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_json_retries_a_throttled_response(monkeypatch) -> None:
    attempts = {"n": 0}

    class FakeOpener:
        def open(self, request, timeout=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise urllib.error.HTTPError(request.full_url, 429, "slow", {}, None)
            return _Response(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(ing.urllib.request, "urlopen", FakeOpener().open)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    assert ing._get_json("https://example.test/x", retries=4, pause=0) == {"ok": True}
    assert attempts["n"] == 3


def test_get_json_gives_up_and_names_the_url(monkeypatch) -> None:
    def always_403(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", always_403)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError) as exc:
        ing._get_json("https://example.test/missing", retries=2, pause=0)
    # An unknown host has one profile, so a refusal ends it there: no sleep, no
    # second request, and the endpoint is named for whoever reads the log.
    assert "failed after 1 attempt" in str(exc.value)
    assert "/missing" in str(exc.value)


def test_stats_forbidden_is_not_retried_into_a_stall(monkeypatch) -> None:
    """stats.nba.com answers 403 to a rejected client; retrying only wastes time.

    A caller that names its own header set gets exactly that set, once: the
    ladder is for hosts the pipeline has not already given instructions for.
    """
    attempts = {"n": 0}
    slept: list[float] = []

    def rejected(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", rejected)
    monkeypatch.setattr(ing.time, "sleep", lambda secs: slept.append(secs))
    with pytest.raises(RuntimeError, match="HTTP 403"):
        ing._get_json("https://stats.nba.com/stats/LeagueGameLog?Season=2024-25",
                      headers=ing._STATS_HEADERS)
    assert attempts["n"] == 1
    assert slept == []


def test_a_fast_refusal_advances_one_rung_of_the_host_ladder(monkeypatch) -> None:
    """A 403 in milliseconds may be about our headers, so ask in another voice."""
    sent: list[dict] = []

    def rejected(request, timeout=None):
        sent.append({k.lower(): v for k, v in request.header_items()})
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", rejected)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    ing._HEADER_RUNG.clear()
    with pytest.raises(RuntimeError, match="HTTP 403"):
        ing._get_json("https://site.api.espn.com/apis/site/v2/sports/"
                      "basketball/nba/teams")
    assert len(sent) == 2, "one rung walk, not a retry storm"
    assert sent[0] != sent[1], "the retry has to actually change what we send"
    assert sent[1] == {k.lower(): v for k, v in ing._HTTP_HEADERS.items()}


def test_espn_is_asked_as_a_plain_client_not_a_browser(monkeypatch) -> None:
    """Measured: ESPN 403s in 70ms to any User-Agent we supply, and serves none.

    This is the refusal the Kaggle log recorded as "answered and refused us",
    and the fix is to stop impersonating Chrome rather than to invent a subtler
    browser string.
    """
    assert ing._header_ladder("site.api.espn.com")[0] == {}, \
        "ESPN's first rung must send no User-Agent at all"
    # The nba.com hosts are the opposite: they need the full costume.
    assert ing._header_ladder("cdn.nba.com")[0] == ing._HTTP_HEADERS
    assert ing._header_ladder("stats.nba.com")[0] == ing._STATS_HEADERS

    sent: list[dict] = []

    class FakeOpener:
        def open(self, request, timeout=None):
            sent.append({k.lower(): v for k, v in request.header_items()})
            return _Response(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(ing.urllib.request, "urlopen", FakeOpener().open)
    ing._get_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams")
    assert "user-agent" not in sent[0], "urllib's default is the profile that works"


def test_the_rung_that_answers_is_reused_for_the_rest_of_the_run(monkeypatch) -> None:
    """One lesson per run, not one probe per request."""
    sent: list[dict] = []

    def first_refused_then_served(request, timeout=None):
        sent.append({k.lower(): v for k, v in request.header_items()})
        if len(sent) == 1:
            raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)
        return _Response(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(ing.urllib.request, "urlopen", first_refused_then_served)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    ing._HEADER_RUNG.clear()
    url = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022400001.json"
    assert ing._get_json(url) == {"ok": True}
    assert len(sent) == 2
    assert ing._HEADER_RUNG["cdn.nba.com"] == 1
    # The next request opens in the voice that worked, with no probe at all.
    ing._get_json(url)
    assert len(sent) == 3
    assert sent[2] == sent[1]


def test_a_silent_host_costs_one_probe_not_a_season_of_them(monkeypatch) -> None:
    """The all-refused verdict must be settled by re-asking, not assumed."""
    tried: list[int] = []

    def only_the_bare_profile_answers(request, timeout=None):
        headers = {k.lower(): v for k, v in request.header_items()}
        tried.append(len(headers))
        if headers == {}:
            return _Response(json.dumps({"gameId": "1"}).encode())
        return None  # allow_missing: this game is not served to that client

    monkeypatch.setattr(ing.urllib.request, "urlopen", only_the_bare_profile_answers)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    ing._HEADER_RUNG.clear()
    assert ing._reprobe_refusal("https://cdn.nba.com/x.json", "cdn.nba.com")
    assert ing._HEADER_RUNG["cdn.nba.com"] == 1
    assert tried == [0], "exactly one re-probe, in the next voice"


def test_a_reprobe_that_is_also_refused_leaves_the_verdict_alone(monkeypatch) -> None:
    """A season really can be unserved; say so instead of looping on it."""
    ing._HEADER_RUNG.clear()

    def always_refused(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", always_refused)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    assert not ing._reprobe_refusal("https://cdn.nba.com/x.json", "cdn.nba.com")
    assert ing._HEADER_RUNG.get("cdn.nba.com") is None


def test_a_host_configured_for_one_attempt_still_gets_the_header_retry(monkeypatch) -> None:
    """The header retry sits outside the retry budget, by design."""
    attempts = {"n": 0}

    def rejected(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", rejected)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    ing._HEADER_RUNG.clear()
    with pytest.raises(RuntimeError, match="HTTP 403"):
        ing._get_json("https://stats.nba.com/stats/LeagueGameLog?Season=2024-25",
                      retries=1, pause=0)
    assert attempts["n"] == 2


def test_espn_is_asked_as_espn_and_never_as_nba(monkeypatch) -> None:
    """ESPN is not a CORS peer of nba.com, and the run must not pretend it is."""
    first = ing._header_ladder("site.api.espn.com")[0]
    assert "Origin" not in first and "Referer" not in first
    assert ing._HTTP_HEADERS["Origin"] == "https://www.nba.com"


def test_only_silence_is_remembered_against_a_host(monkeypatch, tmp_path) -> None:
    """A refusal costs milliseconds to re-ask; a timeout costs the whole budget."""
    monkeypatch.setattr(ing.config, "CACHE_DIR", tmp_path)

    def rejected(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", rejected)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError):
        ing._get_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams")
    assert ing._known_blocked_host("site.api.espn.com") is None

    def silent(request, timeout=None):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(ing.urllib.request, "urlopen", silent)
    with pytest.raises(RuntimeError):
        ing._get_json("https://stats.nba.com/stats/LeagueGameLog", retries=1, pause=0)
    verdict = ing._known_blocked_host("stats.nba.com")
    assert verdict is not None and "no response" in verdict


def test_the_final_error_says_silence_where_there_was_silence() -> None:
    """One refusal and one dead host are different problems; say which is which."""
    assert ing._failure_verdict(
        RuntimeError("NBA request to stats.nba.com failed after 2 attempt(s) "
                     "(TimeoutError: The read operation timed out)")) == \
        "never answered (no response before the timeout)"
    assert ing._failure_verdict(
        ing.CdnUnavailable("site.api.espn.com failed (HTTP 403): Forbidden")) == \
        "answered and refused us (HTTP 403)"
    assert ing._failure_verdict(
        ing.CdnUnavailable("cdn.nba.com refused all 1723 box scores")) == \
        "answered and served nothing"
    assert "never answered" not in ing._failure_verdict(
        ing.CdnUnavailable("site.api.espn.com failed (HTTP 403)"))


def test_cdn_forbidden_is_treated_as_a_missing_game(monkeypatch) -> None:
    """The CDN's 403 means the game does not exist, so it must not be retried."""
    attempts = {"n": 0}

    def absent(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", absent)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    assert ing._get_json("https://cdn.nba.com/static/json/liveData/boxscore/"
                         "boxscore_0022409999.json", allow_missing=True) is None
    assert attempts["n"] == 1


def test_stats_requests_get_a_long_timeout(monkeypatch) -> None:
    """The season-log query is slow, so it gets a much longer ceiling than the CDN."""
    seen = {}

    class Slow:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode()

    def opener(request, timeout=None):
        seen["timeout"] = timeout
        return Slow()

    monkeypatch.setattr(ing.urllib.request, "urlopen", opener)
    ing._get_json("https://stats.nba.com/stats/x", headers=ing._STATS_HEADERS)
    assert seen["timeout"] == ing._HOST_POLICY["stats.nba.com"]["timeout"]
    assert seen["timeout"] >= 60


def test_every_attempt_is_logged_so_a_slow_host_is_visible(monkeypatch, caplog) -> None:
    """Silence is what made the last run look hung; each attempt must announce itself."""
    import logging

    calls = {"n": 0}

    def slow(request, timeout=None):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(ing.urllib.request, "urlopen", slow)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    with caplog.at_level(logging.INFO, logger="ingestion"):
        with pytest.raises(RuntimeError):
            ing._get_json("https://stats.nba.com/stats/LeagueGameLog",
                          headers=ing._STATS_HEADERS)
    messages = [r.getMessage() for r in caplog.records]
    attempts = [m for m in messages if "to stats.nba.com" in m and "request" in m]
    assert len(attempts) >= calls["n"], "every attempt must log before it starts"
    assert any("failed:" in m for m in messages)


def test_a_blackholed_host_cannot_block_past_the_deadline(monkeypatch) -> None:
    """DNS ignores a socket timeout, so an attempt is abandoned on a hard clock."""
    import time as real_time

    monkeypatch.setattr(ing, "DNS_GRACE_SEC", 0.05)

    def never_returns(*args, **kwargs):
        real_time.sleep(30)
        return None

    monkeypatch.setattr(ing.urllib.request, "urlopen", never_returns)
    started = real_time.monotonic()
    with pytest.raises(TimeoutError, match="no response within"):
        ing._request_json("https://stats.nba.com/stats/x", ing._STATS_HEADERS, 0)
    assert real_time.monotonic() - started < 10


def test_a_dead_host_is_not_re_hammered_for_every_season(monkeypatch, caplog) -> None:
    """A failing endpoint must stop the pull, not repeat for each remaining season."""
    import logging

    attempted: list[str] = []

    def fake(season, season_type, pause):
        attempted.append(f"{season}|{season_type}")
        raise ing.SeasonUnavailable(f"{season} {season_type}: timeout")

    monkeypatch.setattr(ing, "_fetch_season_log", fake)
    with caplog.at_level(logging.ERROR, logger="ingestion"):
        with pytest.raises(RuntimeError, match="every season log failed"):
            ing.load_ingested()
    assert len(attempted) == ing.MAX_CONSECUTIVE_FAILURES
    assert any("stopping after" in r.getMessage() for r in caplog.records)


def test_a_single_bad_season_does_not_stop_the_pull(monkeypatch) -> None:
    """One bad season is tolerated; only a repeated failure ends the pull."""
    attempted: list[str] = []

    def fake(season, season_type, pause):
        attempted.append(f"{season}|{season_type}")
        if season == "2024-25" and season_type == ing.SEASON_TYPE_PLAYOFFS:
            raise ing.SeasonUnavailable("2024-25 Playoffs: timeout")
        return pd.DataFrame(season_log_rows())

    monkeypatch.setattr(ing, "_fetch_season_log", fake)
    monkeypatch.setattr(ing, "_pull_play_by_play",
                        lambda *a, **k: pd.DataFrame())
    facts = ing.load_ingested()
    assert len(facts.games) == len(TEAMS)
    assert len(attempted) > ing.MAX_CONSECUTIVE_FAILURES


def test_the_pull_budget_stops_further_seasons(monkeypatch, caplog) -> None:
    """A struggling upstream must degrade into skipped seasons, not an open-ended run."""
    import logging

    def fake(season, season_type, pause):
        raise ing.SeasonUnavailable(f"{season} {season_type}: timeout")

    monkeypatch.setattr(ing, "_fetch_season_log", fake)
    monkeypatch.setenv(ing.PULL_DEADLINE_ENV, "0")
    with caplog.at_level(logging.WARNING, logger="ingestion"):
        with pytest.raises(RuntimeError, match="every season log failed"):
            ing.load_ingested()
    assert any("budget exhausted" in r.getMessage() for r in caplog.records)


def test_a_timeout_is_retried_then_reported(monkeypatch) -> None:
    attempts = {"n": 0}

    def slow(request, timeout=None):
        attempts["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(ing.urllib.request, "urlopen", slow)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError) as exc:
        ing._get_json("https://stats.nba.com/stats/LeagueGameLog",
                      headers=ing._STATS_HEADERS)
    assert attempts["n"] == ing._HOST_POLICY["stats.nba.com"]["attempts"]
    assert "TimeoutError" in str(exc.value)


def test_get_json_tolerates_a_missing_game(monkeypatch) -> None:
    def not_found(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "gone", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", not_found)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    assert ing._get_json("https://example.test/gone", retries=2, pause=0,
                         allow_missing=True) is None


def test_get_json_does_not_retry_a_client_error(monkeypatch) -> None:
    attempts = {"n": 0}

    def bad_request(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", bad_request)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError):
        ing._get_json("https://example.test/bad", retries=4, pause=0)
    assert attempts["n"] == 1


# --------------------------------------------------------------------------
# Season log caching and partial failure
# --------------------------------------------------------------------------


def test_season_log_is_cached_as_it_lands(monkeypatch) -> None:
    """A season that arrives must be durable and must not be re-requested."""
    payload = {"resultSets": [{"headers": ["GAME_ID", "PTS"],
                               "rowSet": [["0022400001", 110]]}]}
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return payload

    monkeypatch.setattr(ing, "_get_json", fake_get)
    first = ing._fetch_season_log("2024-25", ing.SEASON_TYPE_REGULAR, 0.0)
    assert len(first) == 1
    assert len(calls) == 1
    assert ing._season_log_path("2024-25", ing.SEASON_TYPE_REGULAR).exists()
    second = ing._fetch_season_log("2024-25", ing.SEASON_TYPE_REGULAR, 0.0)
    assert len(second) == 1
    assert len(calls) == 1, "a cached season must not hit the network again"


def test_one_unreadable_season_does_not_end_the_run(monkeypatch) -> None:
    def fake(season, season_type, pause):
        if season == "2024-25":
            raise ing.SeasonUnavailable(f"{season} {season_type}: timeout")
        return pd.DataFrame(season_log_rows())

    monkeypatch.setattr(ing, "_fetch_season_log", fake)
    monkeypatch.setattr(ing, "_pull_play_by_play",
                        lambda *a, **k: pd.DataFrame())
    facts = ing.load_ingested()
    assert len(facts.games) == len(TEAMS)
    assert set(facts.games.season) == {2024.0}


def test_a_total_season_failure_names_the_endpoint(monkeypatch) -> None:
    def all_failed(season, season_type, pause):
        raise ing.SeasonUnavailable(f"{season} {season_type}: timeout")

    monkeypatch.setattr(ing, "_fetch_season_log", all_failed)
    with pytest.raises(RuntimeError) as exc:
        ing.load_ingested()
    message = str(exc.value)
    assert "every season log failed" in message
    assert "stats.nba.com/stats/LeagueGameLog" in message


# --------------------------------------------------------------------------
# CDN fallback
# --------------------------------------------------------------------------


def boxscore_payload(game_id: str, code: str, home: str, away: str,
                     home_score: int, away_score: int) -> dict:
    return {"game": {
        "gameId": game_id, "gameCode": f"{code}/{away}{home}",
        "gameStatus": 3, "gameStatusText": "Final",
        "homeTeam": {"teamTricode": home, "teamName": home, "teamCity": home,
                     "score": home_score,
                     "statistics": {"points": home_score, "assists": 25,
                                    "fieldGoalsMade": 40, "fieldGoalsAttempted": 88,
                                    "threePointersMade": 12,
                                    "threePointersAttempted": 35,
                                    "freeThrowsMade": 16, "freeThrowsAttempted": 20,
                                    "reboundsOffensive": 10, "reboundsDefensive": 30,
                                    "reboundsTotal": 40, "turnoversTotal": 12,
                                    "steals": 7, "blocks": 4, "foulsPersonal": 18},
                     "players": [{"personId": 2001, "name": f"Home {home}",
                                  "starter": 1, "played": 1,
                                  "statistics": {"minutes": "PT36M00.00S",
                                                 "points": 20, "assists": 5,
                                                 "reboundsTotal": 6, "plus": 5.0,
                                                 "minus": 3.0}}]},
        "awayTeam": {"teamTricode": away, "teamName": away, "teamCity": away,
                     "score": away_score,
                     "statistics": {"points": away_score, "assists": 22,
                                    "fieldGoalsMade": 38, "fieldGoalsAttempted": 85,
                                    "threePointersMade": 10,
                                    "threePointersAttempted": 32,
                                    "freeThrowsMade": 14, "freeThrowsAttempted": 18,
                                    "reboundsOffensive": 9, "reboundsDefensive": 29,
                                    "reboundsTotal": 38, "turnoversTotal": 13,
                                    "steals": 6, "blocks": 3, "foulsPersonal": 17},
                     "players": [{"personId": 2002, "name": f"Away {away}",
                                  "starter": 1, "played": 1,
                                  "statistics": {"minutes": "PT35M00.00S",
                                                 "points": 18, "assists": 4,
                                                 "reboundsTotal": 5, "plus": 1.0,
                                                 "minus": 4.0}}]}}}


def test_game_id_prefix_and_gameday_from_code() -> None:
    assert ing._season_game_prefix(2024) == "00224"
    assert ing._season_game_prefix(1999) == "00299"
    assert ing._gameday_from_code("20241022/BOSNYK") == pd.Timestamp("2024-10-22")
    assert ing._gameday_from_code("20241022/ATLBOS") == pd.Timestamp("2024-10-22")
    assert ing._gameday_from_code("garbage") is None
    assert ing._gameday_from_code(None) is None


def test_the_game_id_prefix_decides_the_game_type() -> None:
    # The id is exact where a date is a guess: 004 is the postseason bracket,
    # 002 and 006 are decided regular-season games, 001 is preseason.
    assert ing._game_type_from_id("0022400001") == config.GAME_TYPE_REG
    assert ing._game_type_from_id("0042400407") == config.GAME_TYPE_POST
    assert ing._game_type_from_id("0062400001") == config.GAME_TYPE_REG
    assert ing._game_type_from_id("0012400001") is None
    assert ing._game_type_from_id("") is None


def test_the_postseason_bracket_is_enumerated_not_walked() -> None:
    ids = ing._playoff_game_ids(2024)
    # Sparse by construction: a bracket with no holes is not a bracket, and
    # only enumeration reaches it.  Round 0 is the play-in.
    assert "0042400407" in ids and "0042400401" in ids
    assert "0042400301" in ids
    assert "0042400001" in ids
    assert len(ids) == 5 * 9 * 7
    assert all(len(game_id) == 10 for game_id in ids)
    assert len(set(ids)) == len(ids)


def test_the_candidate_ids_cover_every_decided_game_of_a_season() -> None:
    ids = ing._candidate_game_ids(2024)
    assert "0022400001" in ids and "0022412300" not in ids
    assert "0062400001" in ids
    assert "0042400407" in ids
    assert not any(game_id.startswith("001") for game_id in ids), \
        "preseason is not a decided game and must not be trained on"


def test_games_are_stored_in_the_slice_they_belong_to() -> None:
    start = date(2024, 10, 1)
    days = 30
    assert ing._chunk_start(pd.Timestamp("2024-10-01"), start, days) == start
    assert ing._chunk_start(pd.Timestamp("2024-10-30"), start, days) == start
    assert ing._chunk_start(pd.Timestamp("2024-10-31"), start, days) == start + \
        ing.timedelta(days=30)
    assert ing._chunk_bounds(start, date(2025, 7, 1), 30) == (start,
                                                              start + ing.timedelta(days=29))
    assert ing._chunk_bounds(start, start + ing.timedelta(days=3), 30)[1] == \
        start + ing.timedelta(days=3)


def test_a_box_score_normalizes_into_the_three_frames() -> None:
    payload = boxscore_payload("0022400001", "20241022", "BOS", "NYK", 112, 104)
    built = ing._frames_from_boxscore(payload)
    assert built is not None
    games, team_stats, player_stats = built
    assert len(games) == 1 and games.iloc[0].season == 2024.0
    assert games.iloc[0].home_team == "BOS"
    assert games.iloc[0].away_team == "NYK"
    assert set(team_stats.team) == {"BOS", "NYK"}
    assert team_stats.points_for.sum() == 216
    assert set(team_stats.is_home) == {True, False}
    assert len(player_stats) == 2
    assert player_stats.minutes.notna().all()
    # The CDN carries the club name in two fields; serving wants them joined.
    assert set(team_stats.team_name) == {"BOS BOS", "NYK NYK"}


def test_an_unplayed_box_score_is_not_ingested() -> None:
    payload = boxscore_payload("0022400001", "20241022", "BOS", "NYK", 0, 0)
    assert ing._frames_from_boxscore(payload) is None
    assert ing._frames_from_boxscore(None) is None


def test_the_pull_falls_back_to_cdn_box_scores(monkeypatch) -> None:
    """With no season log available the run must still complete from the CDN."""
    monkeypatch.setenv(ing.CDN_FALLBACK_ENV, "1")
    # One slice, fully covered: the gap check is exercised separately.
    monkeypatch.setenv(ing.START_DATE_ENV, "2024-10-01")
    monkeypatch.setenv(ing.END_DATE_ENV, "2024-10-30")
    monkeypatch.setattr(ing, "_fetch_season_log",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ing.SeasonUnavailable("stats.nba.com is not answering")))

    def cdn(url, **kwargs):
        game_id = url.rsplit("_", 1)[-1].replace(".json", "")
        if not game_id.startswith("00224"):
            return None  # 403: only 2024-25 is published
        sequence = int(game_id[5:]) - 1
        if sequence >= len(TEAMS) // 2:
            return None  # 403: the season ends after the last stubbed game
        home, away = TEAMS[sequence], TEAMS[sequence + len(TEAMS) // 2]
        return boxscore_payload(game_id, "20241022", home, away, 112, 104)

    monkeypatch.setattr(ing, "_get_json", cdn)
    monkeypatch.setattr(ing, "_pull_play_by_play",
                        lambda *a, **k: pd.DataFrame())
    facts = ing.load_ingested()
    # Every current team has to appear or the run is not allowed to train.
    assert len(facts.games) == len(TEAMS) // 2
    assert len(facts.team_stats) == len(TEAMS)
    assert len(facts.player_stats) == len(TEAMS)
    assert set(facts.team_stats.team) == set(TEAMS)
    assert set(facts.games.game_type) == {config.GAME_TYPE_REG}
    assert facts.manifest["tables"]["games"] == len(TEAMS) // 2


def test_the_cdn_walk_reads_the_regular_season_and_the_bracket(monkeypatch) -> None:
    served = {
        "0022400001": boxscore_payload("0022400001", "20241022", TEAMS[0],
                                       TEAMS[1], 112, 104),
        "0042400407": boxscore_payload("0042400407", "20250622", TEAMS[2],
                                       TEAMS[3], 108, 99),
    }
    monkeypatch.setattr(ing, "_get_json",
                        lambda url, **kw: served.get(url.rsplit("_", 1)[-1][:-5]))
    frame = ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1),
                                      date(2025, 7, 1))
    assert frame is not None
    games = frame[frame.home_score.notna()]
    assert set(games.game_id) == {"0022400001", "0042400407"}
    finals = games[games.game_id == "0042400407"].iloc[0]
    assert finals.game_type == config.GAME_TYPE_POST
    assert finals.gameday == pd.Timestamp("2025-06-22")


def test_the_cdn_walks_only_seasons_that_can_overlap_the_window() -> None:
    # A season ending in June 2025 cannot hold a game inside a window that
    # stops on 2024-12-31, and walking it would cost a thousand requests.
    assert ing._cdn_seasons_in(date(2024, 10, 1), date(2025, 7, 1)) == [2024]
    assert ing._cdn_seasons_in(date(2024, 1, 1), date(2024, 12, 31)) == [2023, 2024]
    assert ing._cdn_seasons_in(date(2024, 1, 1), date(2026, 9, 25)) == [2023, 2024, 2025]


def test_a_blackholed_cdn_stops_the_walk_instead_of_hammering_it(monkeypatch) -> None:
    asked: list[str] = []

    def dead(url, **kwargs):
        asked.append(url)
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(ing, "_get_json", dead)
    with pytest.raises(ing.CdnUnavailable, match="cdn.nba.com"):
        ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    assert len(asked) == ing.MAX_CONSECUTIVE_FAILURES


def test_a_blackholed_cdn_ends_the_whole_fallback(monkeypatch) -> None:
    def dead(url, **kwargs):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(ing, "_get_json", dead)
    monkeypatch.setenv(ing.START_DATE_ENV, "2024-10-01")
    monkeypatch.setenv(ing.END_DATE_ENV, "2025-07-01")
    with pytest.raises(ing.CdnUnavailable):
        ing._pull_seasons_from_cdn(date(2024, 10, 1), date(2025, 7, 1))


def test_a_cached_cdn_season_keeps_its_team_and_player_rows(monkeypatch) -> None:
    """A game, its two team lines and its player lines must all survive a cache.

    They share a game id, so deduplicating the slice on the id alone quietly
    deletes the team and player rows the models train on.
    """
    def cdn(url, **kwargs):
        if url.endswith("0022400001.json"):
            return boxscore_payload("0022400001", "20241022", TEAMS[0],
                                    TEAMS[1], 112, 104)
        return None

    monkeypatch.setattr(ing, "_get_json", cdn)
    ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    stored = ing._read_chunk(ing._cdn_chunk_path(2024, date(2024, 10, 1)))
    assert len(stored[stored.row_kind == "game"]) == 1
    assert len(stored[stored.row_kind == "team"]) == 2
    assert len(stored[stored.row_kind == "player"]) == 2


# --------------------------------------------------------------------------
# ESPN schedules
# --------------------------------------------------------------------------


def espn_events(count: int = 2, postseason: bool = False) -> list[dict]:
    """Schedule events shaped like ESPN's per-team ``/schedule`` payload."""
    events = []
    for index in range(count):
        home, away = TEAMS[index % len(TEAMS)], TEAMS[(index + 1) % len(TEAMS)]
        events.append({
            "id": f"40170{index:04d}",
            # 7pm Eastern on the 22nd is 23:00/00:00Z on the 23rd, so a UTC
            # date would file this game a day late.
            "date": "2024-10-22T23:00Z",
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"abbreviation": home},
                 "score": {"value": 112.0}},
                {"homeAway": "away", "team": {"abbreviation": away},
                 "score": {"value": 104.0}},
            ]}],
        })
    return events


def test_espn_uses_our_abbreviations_not_theirs() -> None:
    assert ing._espn_abbr("LAL") == "LAL"
    assert ing._espn_abbr("GS") == "GSW"
    assert ing._espn_abbr("no") == "NOP"
    assert ing._espn_abbr("NY") == "NYK"
    assert ing._espn_abbr("UTAH") == "UTA"
    assert ing._espn_abbr("WSH") == "WAS"


def test_a_utc_tipoff_is_filed_on_its_eastern_date() -> None:
    # 2024-10-22T23:00Z is 7pm Eastern on the 22nd. Taking the UTC date
    # would put roughly half the league's games on the wrong day.
    assert ing._gameday_et("2024-10-22T23:00Z") == pd.Timestamp("2024-10-22")
    assert ing._gameday_et("2024-10-23T00:30Z") == pd.Timestamp("2024-10-22")
    assert ing._gameday_et("2024-11-03T01:00Z") == pd.Timestamp("2024-11-02")
    assert ing._gameday_et("") is None
    assert ing._gameday_et(None) is None
    assert ing._gameday_et("not a date") is None


def test_espn_events_become_games_and_team_rows() -> None:
    built = ing._frames_from_espn_events(espn_events(2), 2024,
                                         config.GAME_TYPE_REG)
    assert built is not None
    games = built[built.home_score.notna()]
    teams = built[built.team.notna()]
    assert len(games) == 2
    assert set(games.gameday) == {pd.Timestamp("2024-10-22")}
    assert set(games.season) == {2024.0}
    assert len(teams) == 4
    assert teams.points_for.sum() == 2 * (112.0 + 104.0)
    assert set(teams.net_points) == {8.0, 8.0, -8.0, -8.0}


def test_an_unscored_espn_event_is_not_invented() -> None:
    events = espn_events(1)
    events[0]["competitions"][0]["competitors"][0]["score"] = {}
    assert ing._frames_from_espn_events(events, 2024,
                                        config.GAME_TYPE_REG) is None


def test_the_espn_season_requests_the_year_the_season_ends(monkeypatch) -> None:
    asked: list[str] = []

    def fake(url, **kwargs):
        asked.append(url)
        return {"events": espn_events(1)} if "seasontype=2" in url else {"events": []}

    monkeypatch.setattr(ing, "_get_json", fake)
    monkeypatch.setattr(ing, "_espn_teams", lambda: {t: t for t in TEAMS})
    ing._pull_season_from_espn(2024, 0.0)
    # ESPN season=2025 opens on 2024-10-23, so a 2024 start means season 2025.
    assert all("season=2025" in url for url in asked)
    assert ing._read_chunk(ing.config.CACHE_DIR / "espn_season_2024.parquet") \
        .game_id.nunique() == 1


def test_a_blocked_host_is_not_probed_again() -> None:
    ing._record_host_verdict("stats.nba.com", "sinkholed")
    assert ing._known_blocked_host("stats.nba.com") == "sinkholed"
    with pytest.raises(ing.HostBlocked, match="verdict against it still stands"):
        ing._get_json("https://stats.nba.com/stats/LeagueGameLog")
    assert ing._known_blocked_host("cdn.nba.com") is None




# --------------------------------------------------------------------------
# Fold row order
# --------------------------------------------------------------------------


def fold_frame(games: int = 1023, dates: int = 140, seed: int = 0):
    """A season-shaped frame with many same-date games, as NBA has every day."""
    rng = np.random.default_rng(seed)
    rows, day, number = [], pd.Timestamp("2024-10-22"), 0
    for offset in range(dates):
        for _ in range(rng.integers(5, 10)):
            number += 1
            rows.append({"game_id": f"00224{number:05d}",
                         "gameday": day + pd.Timedelta(days=offset),
                         "season": 2024.0,
                         "home_score": float(rng.integers(85, 130)),
                         "away_score": float(rng.integers(85, 130))})
    return pd.DataFrame(rows)


def test_a_date_only_sort_is_not_enough_for_fold_labels() -> None:
    """The hazard this guard exists for: 86% of rows move between two sorts."""
    df = fold_frame()
    first = df.sort_values("gameday").reset_index(drop=True)
    second = df.sample(frac=1.0, random_state=7).sort_values("gameday") \
        .reset_index(drop=True)
    assert not (first.game_id.to_numpy() == second.game_id.to_numpy()).all(), \
        "if date-only sorting were stable this guard would be untestable"
    canonical = folds_mod.canonical_sort(df)
    shuffled = df.sample(frac=1.0, random_state=7)
    assert (canonical.game_id.tolist()
            == folds_mod.canonical_sort(shuffled).game_id.tolist()), \
        "canonical order must not depend on the order the frame arrived in"


def test_fold_labels_select_the_right_games_whatever_the_arrival_order() -> None:
    """Fold labels are positions, so a differently ordered consumer gets the
    WRONG validation games rather than the same ones in another order."""
    df = fold_frame()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    folds = folds_mod.make_folds(shuffled, "gameday")
    # What moneyline/distributions now do with those labels.
    consumer = folds_mod.canonical_sort(shuffled, "gameday")
    assert folds
    for fold in folds:
        intended = set(shuffled.loc[
            shuffled.gameday.between(fold.val_start, fold.val_end), "game_id"])
        applied = set(consumer.loc[fold.val_idx, "game_id"])
        assert intended == applied, f"fold {fold.fold_id} validated the wrong games"


def test_folds_do_not_depend_on_the_order_the_frame_arrived_in() -> None:
    """The structural guarantee: the same games in the same positions."""
    df = fold_frame()
    baseline = [(tuple(f.train_idx), tuple(f.val_idx))
                for f in folds_mod.make_folds(df, "gameday")]
    for seed in (7, 13, 42):
        other = folds_mod.make_folds(df.sample(frac=1.0, random_state=seed)
                                     .reset_index(drop=True), "gameday")
        assert [(tuple(f.train_idx), tuple(f.val_idx)) for f in other] \
            == baseline, f"arrival order changed the folds (seed {seed})"
    assert baseline, "the fixture must actually produce folds"


def test_canonical_sort_works_without_a_game_id_column() -> None:
    df = pd.DataFrame({"gameday": pd.to_datetime(
        ["2024-01-02", "2024-01-01", "2024-01-02"])})
    out = folds_mod.canonical_sort(df)
    assert out.gameday.tolist() == [pd.Timestamp("2024-01-01"),
                                    pd.Timestamp("2024-01-02"),
                                    pd.Timestamp("2024-01-02")]


def test_a_pull_with_no_game_rows_is_named_not_crashed(monkeypatch) -> None:
    """The failure the Kaggle run hit must read as a blocked host."""
    monkeypatch.setattr(ing, "_pull_season_from_cdn",
                        lambda *a, **k: pd.DataFrame({"game_id": ["0022400001"]}))
    with pytest.raises(RuntimeError, match="no game data"):
        ing._pull_seasons_from_cdn(date(2024, 10, 1), date(2025, 7, 1))


def test_the_probe_ledger_is_never_mistaken_for_a_data_slice(monkeypatch) -> None:
    """The ledger shares the season's file namespace; it must not be read as one.

    Reading it as a slice gave a season that had stored nothing a frame with a
    single ``game_id`` column, which surfaced much later as an AttributeError
    instead of an honest "no data".
    """
    ing._save_probed_ids(2024, {"0022400001", "0042400407"})
    assert ing._probed_path(2024).exists()
    assert ing._probed_path(2024) not in ing._season_chunks(2024)


def test_a_host_that_refuses_everything_says_so(monkeypatch) -> None:
    """A blocked host must be named, not left as a silent empty pull."""
    monkeypatch.setattr(ing, "_get_json", lambda url, **kw: None)
    monkeypatch.setattr(ing, "_reprobe_refusal", lambda url, host: False)
    monkeypatch.setenv(ing.MAX_SEQUENCE_ENV, "40")
    with pytest.raises(ing.CdnUnavailable, match="refused"):
        ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))


def test_a_season_refused_by_one_client_is_re_asked_by_another(monkeypatch) -> None:
    """1,723 identical 403s must not end the season without one re-ask.

    The Kaggle run refused every CDN box score and concluded the host was
    gone. If the refusal was about the client, the whole season was still there
    for the asking - and one request is what tells the two apart. The real
    ``_get_json`` runs here, because the header ladder lives inside it.
    """
    monkeypatch.setenv(ing.MAX_SEQUENCE_ENV, "40")
    ing._HEADER_RUNG.clear()
    asked: list[dict] = []

    def refused_by_the_bare_client_only(request, timeout=None):
        headers = {k.lower(): v for k, v in request.header_items()}
        asked.append(headers)
        if "user-agent" in headers:
            raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)
        if request.full_url.endswith("0022400001.json"):
            return _Response(json.dumps(boxscore_payload(
                "0022400001", "20241022", TEAMS[0], TEAMS[1], 112, 104)).encode())
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", refused_by_the_bare_client_only)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    frame = ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1),
                                      date(2025, 7, 1))
    bare = [h for h in asked if "user-agent" not in h]
    assert frame is not None and not frame.empty, "the season was written off"
    assert frame.game_id.nunique() == 1
    assert len(bare) > 1, "the re-ask is one request; the season is re-walked"
    assert all("user-agent" in h for h in asked[:len(asked) - len(bare)]), \
        "the first walk asked as the browser it always was"


def test_a_season_refused_by_every_client_is_not_retried_forever(monkeypatch) -> None:
    """The re-ask is bounded: a second refusal ends the season for good."""
    monkeypatch.setenv(ing.MAX_SEQUENCE_ENV, "40")
    ing._HEADER_RUNG.clear()
    calls: list[int] = []

    def re_asked_but_refused(url, host):
        calls.append(1)
        return False

    monkeypatch.setattr(ing, "_get_json", lambda url, **kw: None)
    monkeypatch.setattr(ing, "_reprobe_refusal", re_asked_but_refused)
    with pytest.raises(ing.CdnUnavailable, match="different headers"):
        ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    assert len(calls) == 1


def test_a_warm_run_never_asks_about_an_id_twice(monkeypatch) -> None:
    asked: list[str] = []

    def cdn(url, **kwargs):
        asked.append(url.rsplit("_", 1)[-1].replace(".json", ""))
        if url.endswith("0022400001.json"):
            return boxscore_payload("0022400001", "20241022", TEAMS[0],
                                    TEAMS[1], 112, 104)
        return None

    monkeypatch.setattr(ing, "_get_json", cdn)
    ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    first = len(asked)
    ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    # A 403 is as durable a fact as a 200, so nothing is re-asked.
    assert len(asked) == first


def test_a_cdn_season_is_cached_after_the_walk(monkeypatch) -> None:
    asked: list[str] = []

    def cdn(url, **kwargs):
        if url.endswith("0022400001.json"):
            asked.append(url)
            return boxscore_payload("0022400001", "20241022", TEAMS[0],
                                    TEAMS[1], 112, 104)
        return None

    monkeypatch.setattr(ing, "_get_json", cdn)
    ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    assert ing._cdn_chunk_path(2024, date(2024, 10, 1)).exists()
    ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    assert len(asked) == 1, "a stored game must never be requested twice"


def test_an_empty_core_season_slice_stops_the_run(monkeypatch) -> None:
    """A hole in the middle of a season must not be trained through."""
    games = pd.DataFrame({"gameday": [pd.Timestamp("2024-10-22"),
                                      pd.Timestamp("2024-10-23")]})
    empty = ing._report_chunk_gaps(games, date(2024, 10, 1), date(2025, 7, 1))
    assert empty, "a core-season slice with no games must be reported"
    with pytest.raises(RuntimeError, match="empty core-season slice"):
        ing._abort_on_empty_core_chunks(empty, date(2024, 10, 1), date(2025, 7, 1))


def test_a_game_is_counted_in_exactly_one_slice() -> None:
    games = pd.DataFrame({"gameday": [pd.Timestamp("2024-10-30"),
                                      pd.Timestamp("2024-10-31"),
                                      pd.Timestamp("2024-11-01")]})
    counted = []
    start, end = date(2024, 10, 1), date(2024, 11, 29)
    cursor = start
    while cursor <= end:
        _, chunk_end = ing._chunk_bounds(cursor, end, 30)
        counted.append(int(((games.gameday >= pd.Timestamp(cursor))
                            & (games.gameday < pd.Timestamp(chunk_end)
                               + pd.Timedelta(days=1))).sum()))
        cursor = chunk_end + ing.timedelta(days=1)
    assert sum(counted) == len(games), "slices must partition the window"


def test_an_empty_offseason_slice_is_allowed() -> None:
    games = pd.DataFrame({"gameday": [pd.Timestamp("2024-10-22")]})
    # July is the offseason, so nothing there is missing data.
    assert ing._report_chunk_gaps(games, date(2024, 7, 1), date(2024, 9, 30)) == []
    assert ing._abort_on_empty_core_chunks([], date(2024, 7, 1),
                                           date(2024, 9, 30)) is None
    # June is an edge month: the finals end in mid-June, so a June slice is
    # often legitimately empty and must not stop the run.
    assert ing._report_chunk_gaps(games, date(2025, 6, 1), date(2025, 6, 30)) == []


def test_season_log_reuses_a_warm_cache(monkeypatch) -> None:
    path = ing._season_log_path("2024-25", ing.SEASON_TYPE_REGULAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(season_log_rows()).to_parquet(path, index=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a cached season must not be re-requested")

    monkeypatch.setattr(ing, "_get_json", refuse)
    frame = ing._fetch_season_log("2024-25", ing.SEASON_TYPE_REGULAR, 0.0)
    assert len(frame) == 2 * len(TEAMS)


def test_full_repull_ignores_a_cached_season(monkeypatch, tmp_path) -> None:
    path = ing._season_log_path("2024-25", ing.SEASON_TYPE_REGULAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(season_log_rows()).to_parquet(path, index=False)
    monkeypatch.setenv(ing.FULL_REPULL_ENV, "1")
    called: list[str] = []
    monkeypatch.setattr(ing, "_get_json", lambda url, **kw: (
        called.append(url), {"resultSets": [{"headers": ["GAME_ID"],
                                              "rowSet": [["x"]]}]})[1])
    ing._fetch_season_log("2024-25", ing.SEASON_TYPE_REGULAR, 0.0)
    assert called


# --------------------------------------------------------------------------
# load_ingested
# --------------------------------------------------------------------------


def test_the_loader_pulls_both_season_types_and_passes_the_gates(monkeypatch) -> None:
    calls = stub_season(monkeypatch)
    facts = ing.load_ingested()
    assert calls == ["2023-24|Regular Season", "2023-24|Playoffs",
                     "2024-25|Regular Season", "2024-25|Playoffs",
                     "2025-26|Regular Season", "2025-26|Playoffs"]
    assert len(facts.games) == len(TEAMS)
    assert set(facts.games.season) == {2024.0}
    assert len(facts.team_stats) == 2 * len(TEAMS)
    assert not ing._validate_ingested.__doc__ is None
    assert facts.manifest["source_id"] == ing.SOURCE_ID
    assert facts.manifest["window"] == {"start": "2024-01-01", "end": "2025-07-01"}
    assert facts.manifest["tables"]["games"] == len(TEAMS)
    assert ing._cache_paths()[0].exists()
    assert ing._cache_paths()[4].exists()


def test_the_loader_writes_a_reusable_cache(monkeypatch) -> None:
    stub_season(monkeypatch)
    ing.load_ingested()

    def refuse(*args, **kwargs):
        raise AssertionError("a cached run must not call the network")

    monkeypatch.setattr(ing, "_fetch_season_log", refuse)
    cached = ing.load_ingested(allow_download=False)
    assert len(cached.games) == len(TEAMS)
    assert not cached.team_stats.empty


def test_full_repull_ignores_the_cache(monkeypatch) -> None:
    stub_season(monkeypatch)
    ing.load_ingested()
    monkeypatch.setenv(ing.FULL_REPULL_ENV, "1")
    calls = stub_season(monkeypatch)
    facts = ing.load_ingested()
    assert calls  # the season log was re-read despite a warm cache
    assert len(facts.games) == len(TEAMS)


def test_disabled_pull_without_a_cache_fails_loudly(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="no cached games"):
        ing.load_ingested(allow_download=False)


def test_an_empty_window_is_reported_not_trained_on(monkeypatch) -> None:
    def empty(season, season_type, pause):
        return pd.DataFrame()

    monkeypatch.setattr(ing, "_fetch_season_log", empty)
    with pytest.raises(RuntimeError, match="returned no games"):
        ing.load_ingested()


def test_nothing_can_point_the_run_at_a_data_directory(capsys) -> None:
    """The pipeline reads live APIs; there is no longer a way to redirect it.

    The old contract let ``--source-path`` repoint the cache, which is how a
    run could be pointed at a directory of tables it never produced.  An
    argument that is accepted and ignored is worse than a missing one: it looks
    like a supported input, so the flag is gone from both surfaces.
    """
    import inspect

    for target in (ing.load_ingested, mp.run, ing.load_games,
                   ing.load_team_stats, ing.load_player_stats):
        assert "source" not in inspect.signature(target).parameters
    with pytest.raises(SystemExit):
        mp.main(["--source-path", str(BACKEND)])
    assert "unrecognized arguments" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def test_validate_rejects_a_window_without_eligible_seasons() -> None:
    games = pd.DataFrame([{
        "game_id": "1", "gameday": pd.Timestamp("2020-10-20"), "season": 2020.0,
        "home_team": "BOS", "away_team": "NYK", "home_score": 1.0,
        "away_score": 2.0, "game_type": 1,
    }])
    facts = ing.NBAFacts(games, pd.DataFrame(), pd.DataFrame(), {}, {})
    with pytest.raises(RuntimeError, match="no 2024-or-later season"):
        ing._validate_ingested(facts)


def test_validate_rejects_missing_current_team_coverage() -> None:
    games = pd.DataFrame([{
        "game_id": "1", "gameday": pd.Timestamp("2024-10-22"), "season": 2024.0,
        "home_team": "BOS", "away_team": "NYK", "home_score": 1.0,
        "away_score": 2.0, "game_type": 1,
    }])
    facts = ing.NBAFacts(games, pd.DataFrame(), pd.DataFrame(), {}, {})
    with pytest.raises(RuntimeError, match="missing current-team coverage"):
        ing._validate_ingested(facts)


def test_validate_rejects_missing_team_facts() -> None:
    games = pd.DataFrame([{
        "game_id": f"g{i}", "gameday": pd.Timestamp("2024-10-22"),
        "season": 2024.0, "home_team": TEAMS[i], "away_team": TEAMS[i + 1],
        "home_score": 110.0, "away_score": 100.0, "game_type": 1,
    } for i in range(len(TEAMS) - 1)])
    facts = pd.DataFrame([{"game_id": row.game_id, "team": row.home_team}
                          for row in games.itertuples()])
    facts = ing.NBAFacts(games, facts, pd.DataFrame(), {}, {})
    with pytest.raises(RuntimeError, match="missing team box scores"):
        ing._validate_ingested(facts)


def test_eligible_games_filters_season_and_type() -> None:
    games = pd.DataFrame([
        {"game_id": "a", "season": 2020.0, "gameday": "2020-01-01", "game_type": 1},
        {"game_id": "b", "season": 2024.0, "gameday": "2024-10-22", "game_type": 1},
        {"game_id": "c", "season": 2024.0, "gameday": "2024-10-22", "game_type": 2},
        {"game_id": "d", "season": 2024.0, "gameday": "2024-10-22", "game_type": 9},
    ])
    out = ing.eligible_games(games)
    assert sorted(out.game_id) == ["b", "c"]


def test_flags_read_booleans_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv(ing.PLAY_BY_PLAY_ENV, raising=False)
    assert ing._flag(ing.PLAY_BY_PLAY_ENV, True) is True
    for value in ("0", "false", "no", "off", "NO"):
        monkeypatch.setenv(ing.PLAY_BY_PLAY_ENV, value)
        assert ing._flag(ing.PLAY_BY_PLAY_ENV, True) is False
    monkeypatch.setenv(ing.PLAY_BY_PLAY_ENV, "yes")
    assert ing._flag(ing.PLAY_BY_PLAY_ENV, False) is True


# --------------------------------------------------------------------------
# 60-day slices, and a bar that only draws
#
# Two contracts.  The pull is cut into 60-day slices, the way MLB's Statcast
# pull is cut into ``statcast_chunk_days: 60``.  The bar that reports the walk
# is display only, and the guardrail on this work is exactly that: a run with
# the bar on is the same run as a run with it off.
# --------------------------------------------------------------------------


def test_the_pull_is_cut_into_60_day_slices_like_mlb(monkeypatch) -> None:
    assert ing.CDN_CHUNK_DAYS == 60
    start = date(2024, 10, 1)
    assert ing._chunk_bounds(start, date(2025, 7, 1), ing.CDN_CHUNK_DAYS) == (
        start, start + ing.timedelta(days=59))
    # A game 60 days in belongs to the second slice, not the first.
    assert ing._chunk_start(pd.Timestamp("2024-11-30"), start,
                            ing.CDN_CHUNK_DAYS) == start + ing.timedelta(days=60)
    # The override still wins, so an operator can widen or narrow a pull.
    monkeypatch.setenv(ing.CHUNK_DAYS_ENV, "15")
    assert ing._int_env(ing.CHUNK_DAYS_ENV, ing.CDN_CHUNK_DAYS) == 15


def test_the_gap_scan_is_not_coarsened_by_the_pull_width(monkeypatch) -> None:
    """The pull width and the scan width are two knobs on purpose.

    If the scan inherited the pull's 60 days, a fortnight of games that failed
    to arrive would share a slice with games that did arrive, the slice would
    not be empty, and the hole would be invisible.  Widening the pull must
    therefore leave the detector exactly where it was.
    """
    games = pd.DataFrame({"gameday": [pd.Timestamp("2024-10-05"),
                                      pd.Timestamp("2025-01-20")]})
    start, end = date(2024, 10, 1), date(2025, 3, 1)
    monkeypatch.setenv(ing.CHUNK_DAYS_ENV, "30")
    narrow = ing._report_chunk_gaps(games, start, end)
    for width in ("60", "200"):
        monkeypatch.setenv(ing.CHUNK_DAYS_ENV, width)
        assert ing._report_chunk_gaps(games, start, end) == narrow
    # The hole really is there: every slice between the two games is empty.
    assert any(name.startswith("2024-11") for name in narrow)
    assert narrow, "a core-season slice with no games must still be reported"


def test_the_slice_count_matches_the_loop_it_counts() -> None:
    """The bar's total is arithmetic, and arithmetic can disagree with a loop."""
    start, end = date(2024, 10, 1), date(2025, 7, 1)
    for days in (7, 30, 60, 91):
        cursor, walked = start, 0
        while cursor <= end:
            _, chunk_end = ing._chunk_bounds(cursor, end, days)
            walked += 1
            cursor = chunk_end + ing.timedelta(days=1)
        assert ing._slice_count(start, end, days) == walked


def test_the_bar_is_off_when_the_operator_says_so(monkeypatch) -> None:
    for value in ("0", "off", "no", "false"):
        monkeypatch.setenv(prog.ENV, value)
        assert prog.enabled() is False
    for value in ("1", "on", "yes", ""):
        monkeypatch.setenv(prog.ENV, value)
        assert prog._flag() is True
    # A bar needs somewhere to be drawn.  pytest and Kaggle both capture stderr,
    # and a bar redrawn into a captured log is noise, so the default is quiet.
    monkeypatch.delenv(prog.ENV, raising=False)
    monkeypatch.setattr(prog, "_drawable", lambda: False)
    assert prog.enabled() is False


def test_a_bar_without_tqdm_still_yields_the_same_work(monkeypatch) -> None:
    """tqdm is an accelerator, never a requirement.

    The Kaggle notebook is not known to install it, so the module has to be
    correct with the import failing and the run has to be correct either way.
    """
    monkeypatch.setattr(prog, "_drawable", lambda: True)
    monkeypatch.setattr(prog, "_tqdm", lambda: None)
    items = [f"00224000{n:04d}" for n in range(7)]
    assert list(prog.wrap(items, len(items), "demo", unit="game")) == items
    seen: list[int] = []
    with prog.track(3, desc="demo", unit="slice") as bar:
        for n in range(3):
            seen.append(n)
            bar.update(1)
        bar.set_postfix("x")
    assert seen == [0, 1, 2]


def test_a_bar_with_tqdm_still_yields_the_same_work(monkeypatch) -> None:
    """The installed path is the one a Kaggle run with tqdm actually takes."""
    monkeypatch.setattr(prog, "_drawable", lambda: True)
    if prog._tqdm() is None:  # pragma: no cover - depends on the environment
        pytest.skip("tqdm is not installed here")
    items = [1, 2, 3]
    assert list(prog.wrap(items, len(items), "demo", unit="thing")) == items


def test_a_failing_loop_still_closes_its_bar(monkeypatch) -> None:
    """A raised exception must not leave a half-drawn line in a captured log."""
    monkeypatch.setattr(prog, "_drawable", lambda: False)
    with pytest.raises(RuntimeError, match="boom"):
        with prog.track(3, desc="demo", unit="slice") as bar:
            bar.update(1)
            raise RuntimeError("boom")


def test_the_bar_does_not_change_what_the_gap_scan_reports(monkeypatch) -> None:
    """The no-functional-impact proof, at unit size.

    Same games, same window, bar drawing and bar not drawing: the reported
    holes and the log lines have to be identical.  If drawing could ever change
    the walk, this is where it shows.
    """
    games = pd.DataFrame({"gameday": [pd.Timestamp("2024-10-22"),
                                      pd.Timestamp("2025-02-02")]})
    start, end = date(2024, 10, 1), date(2025, 7, 1)

    def scan(drawable: bool) -> tuple[list[str], list[str]]:
        lines: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                lines.append(record.getMessage())

        handler = _Capture()
        previous = ing.logger.level
        ing.logger.addHandler(handler)
        ing.logger.setLevel(logging.INFO)
        monkeypatch.setattr(prog, "_drawable", lambda: drawable)
        try:
            return ing._report_chunk_gaps(games, start, end), lines
        finally:
            ing.logger.removeHandler(handler)
            ing.logger.setLevel(previous)

    with_bar, with_lines = scan(True)
    without_bar, without_lines = scan(False)
    assert with_bar == without_bar
    assert with_lines == without_lines
    assert len(without_lines) == ing._slice_count(start, end, ing.GAP_SCAN_DAYS)


def test_the_pipeline_names_every_phase_it_advances() -> None:
    """The run's phase list and the run's ``advance()`` calls must agree.

    A name that is added to ``PHASES`` without a matching ``advance`` leaves a
    bar that never reaches 100% and a phase nobody can name; an ``advance``
    without a name walks the list off the end.
    """
    source = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    assert source.count("prog.advance()") == len(mp.PHASES)
    assert len(set(mp.PHASES)) == len(mp.PHASES), "phase names must be unique"
    phases = prog.phases(mp.PHASES, desc="NBA pipeline")
    for _ in mp.PHASES:
        phases.advance()
    phases.close()


# --------------------------------------------------------------------------
# A refused host is a verdict, not a fact the run keeps re-buying
#
# The 2026-09-25 Kaggle run answered 1,723 box-score refusals, fell through to
# ESPN, and then asked the same blocked host again for 3,474 play-by-play
# files.  The refusals were cheap; asking for them was not.  These tests pin
# the two places the walk is allowed to stop.
# --------------------------------------------------------------------------


@pytest.fixture
def clean_refusal_state():
    """Refusal streaks are per-run state, and tests must not inherit them."""
    ing._HOST_REFUSALS.clear()
    yield
    ing._HOST_REFUSALS.clear()


def test_a_404_storm_never_condemns_a_host(clean_refusal_state) -> None:
    """Most probed game ids do not exist, and that is the host working.

    A season walk asks about 1,723 ids for roughly 1,230 games; the rest are
    404s by construction.  Counting those as refusals would condemn a host that
    is answering correctly, so only 403 moves the streak.
    """
    for _ in range(ing.REFUSAL_VERDICT_COUNT * 5):
        ing._note_refusal("cdn.nba.com", 404)
    assert ing._refusals_conclusive("cdn.nba.com") is False
    assert ing._known_blocked_host("cdn.nba.com") is None


def test_a_run_of_403s_is_conclusive_and_a_success_clears_it(
        clean_refusal_state) -> None:
    for _ in range(ing.REFUSAL_VERDICT_COUNT - 1):
        ing._note_refusal("cdn.nba.com", 403)
    assert ing._refusals_conclusive("cdn.nba.com") is False
    ing._note_refusal("cdn.nba.com", 403)
    assert ing._refusals_conclusive("cdn.nba.com") is True
    # A host that serves one request after a run of refusals is a host with
    # some absent artifacts, not a host that refuses this client.
    ing._note_success("cdn.nba.com")
    assert ing._refusals_conclusive("cdn.nba.com") is False


def test_a_refused_season_stops_in_sixteen_ids_not_1723(monkeypatch,
                                                        clean_refusal_state) -> None:
    """The walk must not buy 1,723 refusals to learn what 16 already said."""
    monkeypatch.setenv(ing.MAX_SEQUENCE_ENV, "400")
    ing._HEADER_RUNG.clear()
    asked: list[str] = []

    def refused(request, timeout=None):
        asked.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", refused)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    with pytest.raises(ing.CdnUnavailable):
        ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1), date(2025, 7, 1))
    # The re-probe is the one request that could still overturn this, so the
    # budget is the streak plus that request, not the whole season.
    assert len(asked) <= ing.REFUSAL_VERDICT_COUNT + 4, (
        f"{len(asked)} requests to conclude what 16 refusals settled")
    assert ing._known_blocked_host("cdn.nba.com") is not None, (
        "a host this run proved it cannot use must be recorded, or the next "
        "route pays for the same lesson")


def test_the_re_probe_still_gets_its_chance_before_a_verdict(
        monkeypatch, clean_refusal_state) -> None:
    """A client refused by one voice and served by another is not blocked.

    This is the case the short-circuit must not swallow: the streak is
    conclusive, but the season is still there for the asking, and one request
    in the next profile is what tells the two apart.
    """
    monkeypatch.setenv(ing.MAX_SEQUENCE_ENV, "40")
    ing._HEADER_RUNG.clear()
    asked: list[dict] = []

    def served_by_the_bare_client_only(request, timeout=None):
        headers = {k.lower(): v for k, v in request.header_items()}
        asked.append(headers)
        if "user-agent" in headers:
            raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)
        if request.full_url.endswith("0022400001.json"):
            return _Response(json.dumps(boxscore_payload(
                "0022400001", "20241022", TEAMS[0], TEAMS[1], 112, 104)).encode())
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen",
                        served_by_the_bare_client_only)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    frame = ing._pull_season_from_cdn(2024, 0.0, date(2024, 10, 1),
                                      date(2025, 7, 1))
    assert frame is not None and not frame.empty, "the season was written off"
    assert frame.game_id.nunique() == 1
    assert ing._known_blocked_host("cdn.nba.com") is None, (
        "a host that answered in the next voice is not condemned")


def test_play_by_play_is_not_asked_at_all_for_a_blocked_host(
        monkeypatch, clean_refusal_state) -> None:
    """The 3,474-request sweep, reduced to zero by an earlier verdict."""
    ing._record_host_verdict("cdn.nba.com", "answered and refused us (HTTP 403)")
    asked: list[str] = []

    def opener(request, timeout=None):
        asked.append(request.full_url)
        return _Response(b"{}")

    monkeypatch.setattr(ing.urllib.request, "urlopen", opener)
    games = pd.DataFrame({"game_id": [f"00224000{n:02d}" for n in range(50)],
                          "gameday": pd.Timestamp("2024-10-22")})
    out = ing._pull_play_by_play(games, date(2024, 10, 1), date(2025, 7, 1),
                                 config.CACHE_DIR / "play_by_play.parquet",
                                 enabled=True)
    assert asked == [], "a host with a fresh verdict must not be asked again"
    assert out.empty, "there is no cache to return yet, and none is invented"


def test_play_by_play_stops_after_a_conclusive_run_of_refusals(
        monkeypatch, clean_refusal_state) -> None:
    """Enrichment is not worth half an hour of requests that cannot succeed."""
    ing._HEADER_RUNG.clear()
    asked: list[str] = []

    def refused(request, timeout=None):
        asked.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", refused)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    games = pd.DataFrame({"game_id": [f"00224000{n:02d}" for n in range(3474)],
                          "gameday": pd.Timestamp("2024-10-22")})
    out = ing._pull_play_by_play(games, date(2024, 10, 1), date(2025, 7, 1),
                                 config.CACHE_DIR / "play_by_play.parquet",
                                 enabled=True)
    assert len(asked) <= ing.REFUSAL_VERDICT_COUNT + 1, (
        f"{len(asked)} play-by-play requests against a host refusing all of "
        "them; the run that shipped this bug asked for 3,474")
    assert out.empty
    assert ing._known_blocked_host("cdn.nba.com") is not None, (
        "the verdict must outlive the run that earned it")


# --------------------------------------------------------------------------
# ESPN's per-game box score, which is where the players come from now
#
# The 2026-09-25 Kaggle run reached the ESPN route because both NBA.com hosts
# refused it, and the ESPN route carried no player detail, so the run published
# artifacts whose every player-derived feature was a default. The schedules
# endpoint cannot fix that - but the per-game summary endpoint carries the box
# score, so that is where the player lines are fetched.
# --------------------------------------------------------------------------


def _espn_block(team: str, athlete_ids: list[str]) -> dict:
    return {
        "team": {"abbreviation": team},
        "statistics": [{
            "names": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO",
                      "STL", "BLK", "OREB", "DREB", "PF", "+/-"],
            "athletes": [
                {"athlete": {"id": pid, "fullName": f"Player {pid}"},
                 "stats": ["36", "24", "10-18", "2-5", "4-5", "7", "5", "3",
                           "1", "0", "2", "5", "3", "+6"]}
                for pid in athlete_ids
            ],
        }],
    }


def _espn_summary(home: str, away: str) -> dict:
    return {"boxscore": {"players": [_espn_block(home, ["101", "102"]),
                                     _espn_block(away, ["201"])]}}


def test_the_summary_box_score_becomes_player_lines() -> None:
    """One game's summary, in the normalized player_stats shape."""
    rows = ing._player_lines_from_espn_summary(
        _espn_summary("BOS", "NYK"), "401584693", "BOS", "NYK")
    assert len(rows) == 3
    home_row = next(r for r in rows if r["player_id"] == "101")
    assert home_row == {
        "game_id": "401584693", "player_id": "101", "player_name": "Player 101",
        "team": "BOS", "opponent": "NYK", "is_home": True, "win": None,
        "minutes": 36.0, "points": 24.0, "fgm": 10.0, "fga": 18.0,
        "fg3m": 2.0, "fg3a": 5.0, "ftm": 4.0, "fta": 5.0, "reb": 7.0,
        "ast": 5.0, "tov": 3.0, "stl": 1.0, "blk": 0.0, "oreb": 2.0,
        "dreb": 5.0, "pf": 3.0, "plus_minus": 6.0,
    }
    away_row = next(r for r in rows if r["player_id"] == "201")
    assert away_row["is_home"] is False and away_row["opponent"] == "BOS"


def test_the_espn_route_supplies_every_player_stat_the_model_reads(
        monkeypatch, tmp_path) -> None:
    """A fallback may not quietly drop the features it exists to rescue.

    The ESPN route is what serves a host that refuses NBA.com.  If it returns
    fewer player columns than the season log, the run still succeeds while
    every assists/rebounds/shooting feature trains on a default, which is the
    exact silent degradation the player gate exists to prevent — one layer up.
    """
    monkeypatch.setattr(ing, "_get_json", lambda url, **kw: _espn_summary("BOS", "NYK"))
    games = pd.DataFrame([{"game_id": "1", "gameday": "2024-10-22",
                           "home_team": "BOS", "away_team": "NYK",
                           "game_type": 1, "home_score": 112.0,
                           "away_score": 104.0}])
    produced = ing._pull_player_stats_from_espn(
        games, 0.0, tmp_path / "espn_player_stats.parquet")
    reference = set(ing._player_stats_frame(pd.DataFrame([{
        "game_id": "1", "gameday": pd.Timestamp("2024-10-22"),
        "player_id": "101", "player_name": "Player 101", "team": "BOS",
        "minutes": 36.0, "points": 24.0, "fgm": 10, "fga": 18,
        "fg3m": 2, "fg3a": 5, "ftm": 4, "fta": 5, "oreb": 2, "dreb": 5,
        "reb": 7, "ast": 5, "tov": 3, "stl": 1, "blk": 0, "pf": 3,
        "plus_minus": 6, "win": 1.0, "game_type": 1,
    }])).columns)
    missing = reference - set(produced.columns)
    assert not missing, f"the ESPN route omits player stats: {sorted(missing)}"
    # And the numbers have to survive, not just the column names.
    home = produced[produced.team == "BOS"].iloc[0]
    assert home.assists == 5.0 and home.rebounds == 7.0
    assert home.fg_pct == 10 / 18 and home.three_point_pct == 2 / 5
    assert home.win == 1.0


def test_the_shooting_percentages_are_derived_not_parsed() -> None:
    """Made-over-attempted is the rule; a bare cell is not a percentage."""
    frame = ing._derive_player_shooting(pd.DataFrame([
        {"fgm": 10.0, "fga": 18.0, "fg3m": 2.0, "fg3a": 5.0,
         "ftm": 4.0, "fta": 5.0, "reb": 7.0, "ast": 5.0},
        {"fgm": 0.0, "fga": 0.0, "fg3m": 0.0, "fg3a": 0.0,
         "ftm": 0.0, "fta": 0.0, "reb": 0.0, "ast": 0.0},
    ]))
    assert frame.fg_pct.tolist()[:1] == [10 / 18]
    assert frame.three_point_pct.iloc[0] == 2 / 5
    assert frame.free_throw_pct.iloc[0] == 4 / 5
    assert frame.rebounds.tolist() == [7.0, 0.0]
    assert frame.assists.tolist() == [5.0, 0.0]
    # Nobody who played took no shots; that is missing, not 0%.
    assert pd.isna(frame.fg_pct.iloc[1])


def test_win_is_read_from_the_score_the_caller_carries() -> None:
    """``win`` lives on the game, so the scores have to travel with the ask."""
    rows = ing._player_lines_from_espn_summary(
        _espn_summary("BOS", "NYK"), "1", "BOS", "NYK",
        home_score=112, away_score=104)
    assert next(r for r in rows if r["team"] == "BOS")["win"] == 1.0
    assert next(r for r in rows if r["team"] == "NYK")["win"] == 0.0
    lost = ing._player_lines_from_espn_summary(
        _espn_summary("BOS", "NYK"), "1", "BOS", "NYK",
        home_score=99, away_score=110)
    assert next(r for r in lost if r["team"] == "BOS")["win"] == 0.0


def test_stat_columns_are_read_by_name_not_position() -> None:
    """ESPN names its columns, so the parser must not assume an order."""
    payload = _espn_summary("BOS", "NYK")
    group = payload["boxscore"]["players"][0]["statistics"][0]
    group["names"] = ["REB", "AST", "MIN", "FG", "PTS"]
    for athlete in group["athletes"]:
        athlete["stats"] = ["7", "5", "36", "10-18", "24"]
    rows = ing._player_lines_from_espn_summary(payload, "1", "BOS", "NYK")
    assert rows[0]["minutes"] == 36.0 and rows[0]["points"] == 24.0
    assert rows[0]["reb"] == 7.0 and rows[0]["ast"] == 5.0
    assert rows[0]["fgm"] == 10.0 and rows[0]["fga"] == 18.0


def test_a_did_not_play_line_is_missing_not_zero() -> None:
    """A fabricated 0 is a real availability signal, so blanks stay missing."""
    payload = _espn_summary("BOS", "NYK")
    athletes = payload["boxscore"]["players"][0]["statistics"][0]["athletes"]
    athletes[0]["stats"] = [""] * len(
        payload["boxscore"]["players"][0]["statistics"][0]["names"])
    rows = ing._player_lines_from_espn_summary(payload, "1", "BOS", "NYK")
    assert rows[0]["minutes"] is None and rows[0]["points"] is None
    assert rows[0]["reb"] is None and rows[0]["ast"] is None
    assert rows[0]["fgm"] is None and rows[0]["fga"] is None


def test_a_shooting_zero_is_a_real_zero_not_a_missing_cell() -> None:
    """``0-0`` means the player shot nothing, which is not the same as blank."""
    made, attempted = ing._espn_split("0-0")
    assert (made, attempted) == (0.0, 0.0)
    assert ing._espn_split("13-22") == (13.0, 22.0)
    assert ing._espn_split("") == (None, None)
    assert ing._espn_split(None) == (None, None)
    assert ing._espn_split("5") == (None, None), "a bare cell is not a pair"


def test_a_summary_without_player_blocks_yields_nothing() -> None:
    assert ing._player_lines_from_espn_summary({}, "1", "BOS", "NYK") == []
    assert ing._player_lines_from_espn_summary(
        {"boxscore": {"players": []}}, "1", "BOS", "NYK") == []


def test_the_player_pull_asks_only_for_games_it_lacks(monkeypatch, tmp_path) -> None:
    """One request per game is expensive; re-asking is not allowed."""
    ing._HOST_REFUSALS.clear()
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    asked: list[str] = []

    def summary(event: str, **_kwargs):
        asked.append(event)
        return _espn_summary("BOS", "NYK")

    monkeypatch.setattr(ing, "_get_json", summary)
    cache = tmp_path / "espn_player_stats.parquet"
    games = pd.DataFrame([{"game_id": f"g{i}", "gameday": "2024-10-22",
                           "home_team": "BOS", "away_team": "NYK",
                           "game_type": 1} for i in range(3)])
    first = ing._pull_player_stats_from_espn(games, 0.0, cache)
    assert sorted(url.rsplit("=", 1)[-1] for url in asked) == ["g0", "g1", "g2"]
    assert len(first) == 9, "three players per game, three games"
    # Second pass: the cache answers, so nothing is asked again.
    asked.clear()
    second = ing._pull_player_stats_from_espn(games, 0.0, cache)
    assert asked == [], "a cached game was fetched again"
    assert len(second) == len(first)


def test_a_refused_summary_endpoint_stops_the_walk(monkeypatch, tmp_path) -> None:
    """Same discipline as every other per-game walk: bound it, then stop.

    The real ``_get_json`` has to run here.  Stubbing it would stub away the
    very guard under test, which is the mistake this test exists to prevent.
    """
    ing._HOST_REFUSALS.clear()
    ing._record_host_verdict("site.web.api.espn.com", "refused us")
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    asked: list[str] = []

    def opener(request, timeout=None):
        asked.append(request.full_url)
        return urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", opener)
    games = pd.DataFrame([{"game_id": f"g{i}", "gameday": "2024-10-22",
                           "home_team": "BOS", "away_team": "NYK",
                           "game_type": 1} for i in range(500)])
    out = ing._pull_player_stats_from_espn(games, 0.0,
                                           tmp_path / "p.parquet")
    assert asked == [], "a host with a fresh verdict was asked anyway"
    assert out.empty
