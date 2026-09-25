"""Offline contract tests for the NBA API ingestion path.

Every network call is stubbed.  The suite pins the parts that decide whether a
run trains on the right data: how a season log becomes games, team facts, and
player lines, how play-by-play is cached incrementally, and which conditions
must fail the run loudly.
"""
from __future__ import annotations

import json
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
    """Every test gets its own cache and a fixed window."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setenv(ing.START_DATE_ENV, "2024-01-01")
    monkeypatch.setenv(ing.END_DATE_ENV, "2025-07-01")
    monkeypatch.setenv(ing.FULL_REPULL_ENV, "0")
    monkeypatch.setenv(ing.PLAY_BY_PLAY_ENV, "0")
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
    assert "failed after 1 attempt" in str(exc.value)
    assert "/missing" in str(exc.value)


def test_stats_forbidden_is_not_retried_into_a_stall(monkeypatch) -> None:
    """stats.nba.com answers 403 to a rejected client; retrying only wastes time."""
    attempts = {"n": 0}

    def rejected(request, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(ing.urllib.request, "urlopen", rejected)
    monkeypatch.setattr(ing.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        ing._get_json("https://stats.nba.com/stats/LeagueGameLog?Season=2024-25",
                      headers=ing._STATS_HEADERS)
    assert attempts["n"] == 1


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
    """The season-log query is slow; a 45s ceiling is what killed the run."""
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
    assert seen["timeout"] >= 120


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
    wh = ing.load_dataset()
    assert len(wh.games) == len(TEAMS)
    assert set(wh.games.season) == {2024.0}


def test_a_total_season_failure_names_the_endpoint(monkeypatch) -> None:
    def all_failed(season, season_type, pause):
        raise ing.SeasonUnavailable(f"{season} {season_type}: timeout")

    monkeypatch.setattr(ing, "_fetch_season_log", all_failed)
    with pytest.raises(RuntimeError) as exc:
        ing.load_dataset()
    message = str(exc.value)
    assert "every season log failed" in message
    assert "stats.nba.com/stats/LeagueGameLog" in message


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
# load_dataset
# --------------------------------------------------------------------------


def test_load_dataset_pulls_both_season_types_and_passes_the_gates(monkeypatch) -> None:
    calls = stub_season(monkeypatch)
    wh = ing.load_dataset()
    assert calls == ["2023-24|Regular Season", "2023-24|Playoffs",
                     "2024-25|Regular Season", "2024-25|Playoffs",
                     "2025-26|Regular Season", "2025-26|Playoffs"]
    assert len(wh.games) == len(TEAMS)
    assert set(wh.games.season) == {2024.0}
    assert len(wh.team_stats) == 2 * len(TEAMS)
    assert not ing._validate_dataset.__doc__ is None
    assert wh.manifest["dataset_id"] == ing.SOURCE_ID
    assert wh.manifest["window"] == {"start": "2024-01-01", "end": "2025-07-01"}
    assert wh.manifest["tables"]["games"] == len(TEAMS)
    assert ing._cache_paths()[0].exists()
    assert ing._cache_paths()[4].exists()


def test_load_dataset_writes_a_reusable_cache(monkeypatch) -> None:
    stub_season(monkeypatch)
    ing.load_dataset()

    def refuse(*args, **kwargs):
        raise AssertionError("a cached run must not call the network")

    monkeypatch.setattr(ing, "_fetch_season_log", refuse)
    cached = ing.load_dataset(allow_download=False)
    assert len(cached.games) == len(TEAMS)
    assert not cached.team_stats.empty


def test_full_repull_ignores_the_cache(monkeypatch) -> None:
    stub_season(monkeypatch)
    ing.load_dataset()
    monkeypatch.setenv(ing.FULL_REPULL_ENV, "1")
    calls = stub_season(monkeypatch)
    wh = ing.load_dataset()
    assert calls  # the season log was re-read despite a warm cache
    assert len(wh.games) == len(TEAMS)


def test_disabled_pull_without_a_cache_fails_loudly(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="no cached games"):
        ing.load_dataset(allow_download=False)


def test_an_empty_window_is_reported_not_trained_on(monkeypatch) -> None:
    def empty(season, season_type, pause):
        return pd.DataFrame()

    monkeypatch.setattr(ing, "_fetch_season_log", empty)
    with pytest.raises(RuntimeError, match="returned no games"):
        ing.load_dataset()


def test_source_directory_becomes_the_cache_location(monkeypatch, tmp_path) -> None:
    stub_season(monkeypatch)
    target = tmp_path / "explicit-cache"
    target.mkdir()
    ing.load_dataset(source=str(target))
    assert (target / "games.parquet").exists()
    assert (target / "team_stats.parquet").exists()


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def test_validate_rejects_a_window_without_eligible_seasons() -> None:
    games = pd.DataFrame([{
        "game_id": "1", "gameday": pd.Timestamp("2020-10-20"), "season": 2020.0,
        "home_team": "BOS", "away_team": "NYK", "home_score": 1.0,
        "away_score": 2.0, "game_type": 1,
    }])
    wh = ing.Warehouse(games, pd.DataFrame(), pd.DataFrame(), {}, {})
    with pytest.raises(RuntimeError, match="no 2024-or-later season"):
        ing._validate_dataset(wh)


def test_validate_rejects_missing_current_team_coverage() -> None:
    games = pd.DataFrame([{
        "game_id": "1", "gameday": pd.Timestamp("2024-10-22"), "season": 2024.0,
        "home_team": "BOS", "away_team": "NYK", "home_score": 1.0,
        "away_score": 2.0, "game_type": 1,
    }])
    wh = ing.Warehouse(games, pd.DataFrame(), pd.DataFrame(), {}, {})
    with pytest.raises(RuntimeError, match="missing current-team coverage"):
        ing._validate_dataset(wh)


def test_validate_rejects_missing_team_facts() -> None:
    games = pd.DataFrame([{
        "game_id": f"g{i}", "gameday": pd.Timestamp("2024-10-22"),
        "season": 2024.0, "home_team": TEAMS[i], "away_team": TEAMS[i + 1],
        "home_score": 110.0, "away_score": 100.0, "game_type": 1,
    } for i in range(len(TEAMS) - 1)])
    facts = pd.DataFrame([{"game_id": row.game_id, "team": row.home_team}
                          for row in games.itertuples()])
    wh = ing.Warehouse(games, facts, pd.DataFrame(), {}, {})
    with pytest.raises(RuntimeError, match="missing team box scores"):
        ing._validate_dataset(wh)


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
