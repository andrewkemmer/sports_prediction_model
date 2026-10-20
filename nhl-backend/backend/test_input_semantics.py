"""Hockey-semantic fixtures independent of model tuning and live pulls."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
import features as feat

# Patch the exact module instances the feature engine consumes; package and
# direct-script imports can coexist during repository-root pytest collection.
config = feat.config
ing = feat.ingestion


@pytest.fixture(autouse=True)
def offline_features(monkeypatch):
    monkeypatch.setattr(feat, "_load_player_ratings", lambda: pd.DataFrame())
    monkeypatch.setattr(feat, "_combined_exclusions", lambda *args: (pd.DataFrame(), {}))
    monkeypatch.setattr(feat, "_load_team_rosters", lambda *args: None)


def official_sample():
    # Official game 2024020194: BUF 2/4 PP, OTT 0/3; 27/62 vs 35/62
    # faceoffs. Goalie PP shots (6 on each side) are NOT opportunities.
    box = {"id": 2024020194, "gameState": "OFF",
           "homeTeam": {"sog": 34}, "awayTeam": {"sog": 38},
           "playerByGameStats": {
               "homeTeam": {"forwards": [
                   {"powerPlayGoals": 2, "faceoffWinningPctg": 0},
                   {"powerPlayGoals": 0, "faceoffWinningPctg": .6}],
                   "goalies": [{"playerId": 8480045, "starter": True, "toi": "60:00",
                                "goalsAgainst": 1, "shotsAgainst": 38,
                                "powerPlayShotsAgainst": "6/6"}]},
               "awayTeam": {"forwards": [{"powerPlayGoals": 0, "faceoffWinningPctg": 0}],
                   "goalies": [{"playerId": 8476999, "starter": True, "toi": "60:00",
                                "goalsAgainst": 5, "shotsAgainst": 34,
                                "powerPlayShotsAgainst": "4/6"}]}}}
    rail = {"teamGameStats": [
        {"category": "powerPlay", "homeValue": "2/4", "awayValue": "0/3"},
        {"category": "faceoffWins", "homeValue": "27/62", "awayValue": "35/62"}]}
    return box, rail


def game(gid, day, season=2024, home="ANA", away="BUF", score=3):
    return dict(game_id=gid, gameday=pd.Timestamp(day), season=season,
                home_team=home, away_team=away, home_score=score,
                away_score=1 if pd.notna(score) else np.nan,
                start_time_utc=f"{day}T23:00:00Z")


def goalie_box(gid, goalie, name=None, ga=2):
    return dict(game_id=gid, home_goalie_id=goalie, home_goalie_name=name or goalie,
                home_goalie_toi=60, home_goals_against=ga, home_shots_against=25,
                away_goalie_id="buf", away_goalie_name="BUF goalie", away_goalie_toi=60,
                away_goals_against=3, away_shots_against=30)


def trade_history():
    clubs = ["ANA", "ANA", "ANA", "BOS", "BOS", "BOS", "BOS"]
    goalies = ["old", "current", "current", "old", "old", "old", "old"]
    games, boxes = [], []
    for i, (club, goalie) in enumerate(zip(clubs, goalies)):
        day = f"2024-10-{i + 1:02d}"
        games.append(game(str(i), day, home=club))
        boxes.append(goalie_box(str(i), goalie))
    target = pd.DataFrame([game("target", "2024-10-10", score=np.nan)])
    return pd.DataFrame(games), pd.DataFrame(boxes), target


def test_official_team_sample_not_player_or_goalie_proxies():
    box, rail = official_sample()
    row = ing._parse_boxscore(box, rail)
    assert row["home_pp_goals"] == 2
    assert row["home_pp_opportunities"] == 4
    assert row["away_pp_opportunities"] == 3
    assert row["home_faceoff_wins"] == 27
    assert row["home_faceoff_attempts"] == 62
    assert row["home_faceoff_pct"] == pytest.approx(27 / 62)
    assert row["away_faceoff_pct"] == pytest.approx(35 / 62)
    assert row["home_goalie_id"] == 8480045
    assert row["home_goals_against"] == 1


@pytest.mark.parametrize("bad", [None, "", "x", "2/", "-1/3", "4/2", "1.5/3", "1/2.5"])
def test_missing_or_malformed_counts_never_use_proxies(bad):
    box, _ = official_sample()
    rail = {"teamGameStats": [
        {"category": "powerPlay", "homeValue": bad},
        {"category": "faceoffWins", "homeValue": bad}]}
    row = ing._parse_boxscore(box, rail)
    for stat in ["pp_goals", "pp_opportunities", "faceoff_wins", "faceoff_attempts", "faceoff_pct"]:
        assert row[f"home_{stat}"] is None


@pytest.mark.parametrize("rail", [None, {}, {"teamGameStats": None}, {"teamGameStats": "bad"}])
def test_malformed_team_stats_payload_preserves_boxscore_facts(rail):
    box, _ = official_sample()
    row = ing._parse_boxscore(box, rail)
    assert row["home_sog"] == 34
    assert row["home_pp_opportunities"] is None
    assert row["home_faceoff_pct"] is None


def test_zero_opportunities_are_observed_counts_not_unknown():
    box, rail = official_sample()
    rail["teamGameStats"][0]["homeValue"] = "0/0"
    row = ing._parse_boxscore(box, rail)
    assert row["home_pp_goals"] == row["home_pp_opportunities"] == 0


def test_loader_fetches_right_rail_versions_cache_and_reuses_it(tmp_path, monkeypatch):
    box, rail = official_sample()
    calls = []
    old = tmp_path / "boxscore_v4_2024020194.parquet"
    old.write_bytes(b"old wrong count cache must not be read")
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing, "_progress_bar", lambda *args: None)
    def fetch(url):
        calls.append(url)
        return rail if url.endswith("right-rail") else box
    monkeypatch.setattr(ing, "_http_json", fetch)
    first = ing.load_boxscores(["2024020194"])
    second = ing.load_boxscores(["2024020194"])
    assert len(calls) == 2
    assert calls[0].endswith("boxscore") and calls[1].endswith("right-rail")
    assert (tmp_path / f"boxscore_{ing.BOXSCORE_CACHE_VERSION}_2024020194.parquet").is_file()
    assert old.read_bytes() == b"old wrong count cache must not be read"
    pd.testing.assert_frame_equal(first, second)
    assert first.home_pp_opportunities.iloc[0] == 4
    assert first.home_faceoff_wins.dtype == "float64"


def test_failed_right_rail_preserves_quality_but_retries_missing_counts(tmp_path, monkeypatch):
    box, rail = official_sample()
    attempts = []
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing, "_progress_bar", lambda *args: None)
    def fetch(url):
        if url.endswith("right-rail"):
            attempts.append(url)
            if len(attempts) == 1:
                raise RuntimeError("endpoint unavailable")
            return rail
        return box
    monkeypatch.setattr(ing, "_http_json", fetch)
    first = ing.load_boxscores(["2024020194"])
    assert first.home_sog.iloc[0] == 34
    assert first.home_goals_against.iloc[0] == 1
    assert pd.isna(first.home_pp_opportunities.iloc[0])
    assert not list(tmp_path.glob("*.parquet"))
    second = ing.load_boxscores(["2024020194"])
    assert second.home_pp_opportunities.iloc[0] == 4
    assert len(attempts) == 2
    assert len(list(tmp_path.glob("*.parquet"))) == 1


def test_recent_boxscores_refresh_for_feed_corrections(tmp_path, monkeypatch):
    """A cached boxscore inside BOXSCORE_REFRESH_DAYS re-pulls for feed fixes.

    2026-10-09 NHL log review: the cache write is gated ONLY on the right-rail
    team counts, so a boxscore cached before the feed finished posting
    goalie/SOG fields is stored incomplete and — in the mode where the cache
    is read at all (--skip-pull) — is never fetched again. The 2026-10-05 run
    shipped that signature (9 WARM nulls across the goalie families, healed
    only by NHL_FULL_REPULL=1).

    Three pinned behaviours, mirroring the score-page contract:
      (1) inside the window the game re-pulls and the correction lands in
          both the returned frame and the cache;
      (2) outside it a cache hit is final — no network call at all;
      (3) a FAILED refresh serves the cached row: staleness beats a hole.
    """
    from datetime import date, timedelta

    gid = "2024020194"
    box, rail = official_sample()
    corrected = {"teamGameStats": [
        {"category": "powerPlay", "homeValue": "2/4", "awayValue": "0/3"},
        {"category": "faceoffWins", "homeValue": "30/62", "awayValue": "32/62"}]}
    recent = pd.Timestamp(date.today() - timedelta(days=1))
    ancient = pd.Timestamp(date.today() -
                           timedelta(days=ing.BOXSCORE_REFRESH_DAYS + 400))
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing, "_progress_bar", lambda *args: None)

    state = {"rail": rail, "boom": False}

    def fetch(url):
        if state["boom"]:
            raise RuntimeError("network down")
        calls.append(url)
        return state["rail"] if url.endswith("right-rail") else box

    calls: list[str] = []
    monkeypatch.setattr(ing, "_http_json", fetch)
    cache_file = tmp_path / f"boxscore_{ing.BOXSCORE_CACHE_VERSION}_{gid}.parquet"

    # Seed: an uncached game fetches and writes a complete cached row.
    first = ing.load_boxscores([gid], use_cache=True, gameday_by_id={gid: recent})
    assert len(calls) == 2 and cache_file.is_file()
    assert int(first["home_faceoff_wins"].iloc[0]) == 27

    # (1) inside the refresh window: re-pull, correction lands, re-cached.
    state["rail"] = corrected
    second = ing.load_boxscores([gid], use_cache=True, gameday_by_id={gid: recent})
    assert len(calls) == 4, (
        "a cached boxscore inside BOXSCORE_REFRESH_DAYS was never re-pulled")
    assert int(second["home_faceoff_wins"].iloc[0]) == 30, (
        "the feed correction did not land in the returned frame")
    assert int(pd.read_parquet(cache_file)["home_faceoff_wins"].iloc[0]) == 30, (
        "the refreshed correction was not written back to the cache")

    # (2) outside the window: cache hit, no network, same values.
    third = ing.load_boxscores([gid], use_cache=True, gameday_by_id={gid: ancient})
    assert len(calls) == 4, "an old boxscore was re-pulled despite settlement"
    assert int(third["home_faceoff_wins"].iloc[0]) == 30
    assert len(third) == 1

    # (3) a failed refresh falls back to the cached row — never a hole.
    state["boom"] = True
    fourth = ing.load_boxscores([gid], use_cache=True, gameday_by_id={gid: recent})
    assert len(calls) == 4, "the fetch should have been attempted then failed"
    assert len(fourth) == 1, "a failed refresh dropped the game from the frame"
    assert int(fourth["home_faceoff_wins"].iloc[0]) == 30, (
        "a failed refresh must serve the cached row, not an empty one")


def test_refresh_window_needs_a_known_gameday(tmp_path, monkeypatch):
    """No mapping (or an undatable game) keeps the cache authoritative.

    The refresh is opt-in for dates we can date; without a gameday the age
    is unknowable, so falling back to 'no re-pull' is what keeps a daily
    run from re-fetching the whole history.
    """
    box, rail = official_sample()
    calls: list[str] = []
    monkeypatch.setattr(ing, "_cache_path", lambda name: tmp_path / name)
    monkeypatch.setattr(ing, "_progress_bar", lambda *args: None)

    def fetch(url):
        calls.append(url)
        return rail if url.endswith("right-rail") else box

    monkeypatch.setattr(ing, "_http_json", fetch)
    gid = "2024020194"
    ing.load_boxscores([gid], use_cache=True)               # seeds the cache
    assert len(calls) == 2
    ing.load_boxscores([gid], use_cache=True)               # no mapping
    ing.load_boxscores([gid], use_cache=True,
                       gameday_by_id={gid: None})           # undatable
    assert len(calls) == 2, "an undated cached boxscore was re-pulled"


def test_pp_and_faceoffs_pool_counts_and_exclude_current_game():
    games = pd.DataFrame([game(str(i), f"2024-10-{i + 1:02d}") for i in range(4)])
    rows = []
    for i, (wins, attempts, ppg, chances) in enumerate([(1, 2, 1, 2), (9, 10, 0, 1),
                                                         (0, 0, 0, 0), (100, 100, 4, 4)]):
        rows.append(dict(game_id=str(i), home_faceoff_wins=wins, home_faceoff_attempts=attempts,
                         away_faceoff_wins=attempts - wins, away_faceoff_attempts=attempts,
                         home_pp_goals=ppg, home_pp_opportunities=chances,
                         away_pp_goals=0, away_pp_opportunities=chances))
    frame = feat.build_game_features(games, pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())
    assert pd.isna(frame.faceoff_win_home.iloc[0])
    assert frame.faceoff_win_home.iloc[2] == pytest.approx(10 / 12)  # not mean(.5,.9)
    assert frame.faceoff_win_home.iloc[3] == pytest.approx(10 / 12)
    assert frame.faceoff_win_diff.iloc[3] == pytest.approx(8 / 12)
    assert frame.pp_success_home.iloc[3] == pytest.approx(1 / 3)
    assert frame.pp_success_diff.iloc[3] == pytest.approx(1 / 3)


def test_incomplete_count_pair_contributes_neither_numerator_nor_denominator():
    games = pd.DataFrame([game(str(i), f"2024-10-{i + 1:02d}") for i in range(3)])
    boxes = pd.DataFrame([
        dict(game_id="0", home_pp_goals=1., home_pp_opportunities=2.,
             home_faceoff_wins=1., home_faceoff_attempts=2.),
        dict(game_id="1", home_pp_goals=np.nan, home_pp_opportunities=100.,
             home_faceoff_wins=np.nan, home_faceoff_attempts=100.),
        dict(game_id="2", home_pp_goals=0., home_pp_opportunities=2.,
             home_faceoff_wins=1., home_faceoff_attempts=2.)])
    frame = feat.build_game_features(games, boxes, pd.DataFrame(), pd.DataFrame())
    assert frame.pp_success_home.iloc[2] == .5
    assert frame.faceoff_win_home.iloc[2] == .5


def test_shots_family_falls_back_to_settled_games_frame_sog():
    """A missing boxscore must not drop shots_for / shots_against.

    The settled score feed carries byte-identical team SOG, so those two
    served features resolve from ``games`` when no boxscore row exists, while
    PP/faceoff (no score-feed source) correctly stay undefined.
    """
    games = pd.DataFrame([
        {**game("0", "2024-10-01"), "home_sog": 34.0, "away_sog": 28.0},
        {**game("1", "2024-10-03"), "home_sog": 30.0, "away_sog": 31.0},
    ])
    # No boxscores at all: PP/faceoff/goalie have no source; shots do.
    frame = feat.build_game_features(games, pd.DataFrame(),
                                     pd.DataFrame(), pd.DataFrame())
    # shots-for/against resolve from the games frame's SOG (not all NaN).
    assert frame.shots_for_per_game_home.notna().sum() >= 1
    assert frame.shots_against_per_game_home.notna().sum() >= 1
    # Trailing strictly-prior window: game 1's home shots-for = game 0's 34,
    # shots-against = the opponent's (BUF) prior SOG of 28.
    assert frame.shots_for_per_game_home.iloc[1] == pytest.approx(34.0)
    assert frame.shots_against_per_game_home.iloc[1] == pytest.approx(28.0)
    # The team's first game has no prior -> NaN (never fabricated).
    assert pd.isna(frame.shots_for_per_game_home.iloc[0])
    # PP/faceoff stay undefined without a boxscore: the score feed lacks them.
    assert frame.pp_success_home.isna().all()
    assert frame.faceoff_win_home.isna().all()


@pytest.mark.parametrize("home_wins", [True, False])
def test_elo_positive_home_edge_and_rating_conservation(home_wins):
    g = pd.DataFrame([game("1", "2024-10-01", score=3 if home_wins else 0)])
    events, ratings = feat._elo_apply(feat.team_events(g))
    expected = 1 / (1 + 10 ** (-config.ELO_HOME_ADV / config.ELO_SCALE))
    # Adopted 2026-10-20: the margin mult scales K (both clubs open at the
    # prior, so the elo gap is exactly the home advantage). mov is +2 for a
    # 3-1 home win, -1 for a 0-1 home loss.
    mov = (3 if home_wins else 0) - 1
    mov_mult = ((abs(mov) + 3.0) ** 0.8) / (7.5 + 0.006 * config.ELO_HOME_ADV)
    delta = config.ELO_K * mov_mult * (int(home_wins) - expected)
    assert expected > .5
    assert ratings["ANA"] == pytest.approx(config.ELO_PRIOR + delta)
    assert ratings["BUF"] == pytest.approx(config.ELO_PRIOR - delta)
    assert events.elo_entering.eq(config.ELO_PRIOR).all()
    assert sum(ratings.values()) == pytest.approx(2 * config.ELO_PRIOR)


def test_elo_no_home_edge_is_symmetric(monkeypatch):
    monkeypatch.setattr(config, "ELO_HOME_ADV", 0)
    _, ratings = feat._elo_apply(feat.team_events(pd.DataFrame([game("1", "2024-10-01")])))
    # 3-1 home win: mov mult ((2+3)^0.8)/7.5 on the symmetric binary update.
    mov_mult = ((2.0 + 3.0) ** 0.8) / 7.5
    assert ratings["ANA"] == config.ELO_PRIOR + config.ELO_K * mov_mult / 2
    assert ratings["BUF"] == config.ELO_PRIOR - config.ELO_K * mov_mult / 2


def test_elo_margin_mult_flag_off_restores_binary(monkeypatch):
    """The adoption is a flag flip: False must restore the binary K update."""
    monkeypatch.setattr(config, "ELO_MARGIN_MULT", False)
    g = pd.DataFrame([game("1", "2024-10-01", score=3)])
    _, ratings = feat._elo_apply(feat.team_events(g))
    expected = 1 / (1 + 10 ** (-config.ELO_HOME_ADV / config.ELO_SCALE))
    delta = config.ELO_K * (1 - expected)
    assert ratings["ANA"] == pytest.approx(config.ELO_PRIOR + delta)
    assert ratings["BUF"] == pytest.approx(config.ELO_PRIOR - delta)
    assert sum(ratings.values()) == pytest.approx(2 * config.ELO_PRIOR)


def test_elo_pending_games_do_not_update_skill_and_opener_matches_history():
    prior = pd.DataFrame([game("1", "2024-10-01")])
    future = pd.DataFrame([game("2", "2025-10-01", 2025, score=np.nan),
                           game("3", "2025-10-03", 2025, score=np.nan)])
    _, old_ratings = feat._elo_apply(feat.team_events(prior))
    slate = feat.build_slate_features(pd.concat([prior, future]), player_ratings=pd.DataFrame(),
                                     stints=pd.DataFrame())
    expected = config.ELO_PRIOR + (old_ratings["ANA"] - config.ELO_PRIOR) * (1 - config.ELO_REVERT_FACTOR)
    assert slate.elo_home.tolist() == pytest.approx([expected, expected])
    decided = pd.concat([prior, future.iloc[:1].assign(home_score=3., away_score=1.)])
    history = feat.build_game_features(decided, player_ratings=pd.DataFrame(), stints=pd.DataFrame())
    assert history.elo_home.iloc[-1] == slate.elo_home.iloc[0]
    assert history.elo_away.iloc[-1] == slate.elo_away.iloc[0]


def test_goalie_workload_excludes_departed_club_and_keeps_earlier_history():
    games, boxes, target = trade_history()
    all_games = pd.concat([games, target])
    frame, _ = feat.goalie_state(boxes, target, all_games)
    assert frame.g_home_name.iloc[0] == "current"
    assert frame.goalie_starts_home.iloc[0] == 2
    # Future transfer cannot change the October 3 feature.
    earlier, _ = feat.goalie_state(boxes, games.iloc[[2]], all_games)
    before, _ = feat.goalie_state(boxes.iloc[:2], games.iloc[[2]], games.iloc[:3])
    pd.testing.assert_frame_equal(earlier, before)


def test_goalie_vote_counts_team_starts_not_all_club_starts():
    # Four earlier BOS starts cannot let a just-arrived ANA goalie beat ANA's
    # three-start incumbent. Player quality/season line can span both clubs.
    games, boxes = [], []
    for i, club in enumerate(["BOS"] * 4 + ["ANA"] * 4):
        gid = str(i)
        games.append(game(gid, f"2024-10-{i + 1:02d}", home=club))
        boxes.append(goalie_box(gid, "new" if i < 4 or i == 7 else "incumbent"))
    target = pd.DataFrame([game("t", "2024-10-10", score=np.nan)])
    frame, _ = feat.goalie_state(pd.DataFrame(boxes), target, pd.concat([pd.DataFrame(games), target]))
    assert frame.g_home_name.iloc[0] == "incumbent"


@pytest.mark.parametrize("capture,complete,expected", [
    ("2024-10-10T22:00:00Z", True, "current"),
    ("2024-10-10T23:00:00Z", True, "old"),  # exact boundary is not prior
    ("2024-10-11T00:00:00Z", True, "old"),
    ("2024-10-01T00:00:00Z", True, "old"),  # stale
    ("2024-10-10T22:00:00Z", False, "old"),  # absence isn't proof
])
def test_roster_absence_only_excludes_with_fresh_complete_prior_capture(capture, complete, expected):
    games = pd.DataFrame([game(str(i), f"2024-10-{i + 1:02d}") for i in range(4)])
    boxes = pd.DataFrame([goalie_box(str(i), "current" if i == 3 else "old") for i in range(4)])
    target = pd.DataFrame([game("t", "2024-10-10", score=np.nan)])
    roster = pd.DataFrame([dict(player_id="current", team="ANA", snapshot_at=capture, complete=complete)])
    frame, _ = feat.goalie_state(boxes, target, pd.concat([games, target]), rosters=roster)
    assert frame.g_home_name.iloc[0] == expected


def test_nullable_numeric_goalie_ids_match_central_roster_ids():
    games = pd.DataFrame([game("1", "2024-10-01"), game("2", "2024-10-02"),
                          game("3", "2024-10-03")])
    boxes = pd.DataFrame([goalie_box("1", 123., name="old"),
                         goalie_box("2", 123., name="old"),
                         goalie_box("3", 456., name="current")])
    target = pd.DataFrame([game("t", "2024-10-10", score=np.nan)])
    roster = pd.DataFrame([dict(player_id="123", team="BOS",
                               snapshot_at="2024-10-10T22:00:00Z", complete=False)])
    frame, _ = feat.goalie_state(boxes, target, pd.concat([games, target]), rosters=roster)
    assert frame.g_home_name.iloc[0] == "current"


def test_partial_roster_positive_transfer_knowledge_excludes_old_goalie():
    games = pd.DataFrame([game(str(i), f"2024-10-{i + 1:02d}") for i in range(4)])
    boxes = pd.DataFrame([goalie_box(str(i), "current" if i == 3 else "old") for i in range(4)])
    target = pd.DataFrame([game("t", "2024-10-10", score=np.nan)])
    roster = pd.DataFrame([dict(player_id="old", team="BOS", snapshot_at="2024-10-10T22:00:00Z",
                               complete=False)])
    frame, _ = feat.goalie_state(boxes, target, pd.concat([games, target]), rosters=roster)
    assert frame.g_home_name.iloc[0] == "current"


def test_future_goalie_names_and_current_boxscore_do_not_rewrite_prior_choice():
    games, boxes, target = trade_history()
    frame, _ = feat.goalie_state(boxes, games.iloc[[2]], games)
    dirty = boxes.copy()
    dirty.loc[dirty.game_id == "2", "home_goalie_name"] = "future name"
    changed, _ = feat.goalie_state(dirty, games.iloc[[2]], games)
    pd.testing.assert_frame_equal(frame, changed)


def test_goalie_roster_gate_is_wired_into_history_and_slate_builders(monkeypatch):
    games, boxes, target = trade_history()
    # Before any observed transfer, the old three-start leader is removed by
    # the captured roster. Capture is before exact puck drop, not gameday.
    games = games.iloc[:3]
    boxes = boxes.iloc[:3].copy()
    boxes.loc[boxes.game_id.isin(["0", "1"]), "home_goalie_id"] = "old"
    boxes.loc[boxes.game_id.isin(["0", "1"]), "home_goalie_name"] = "old"
    roster = pd.DataFrame([dict(player_id="current", team="ANA",
                               snapshot_at="2024-10-10T22:00:00Z", complete=True),
                           dict(player_id="buf", team="BUF",
                               snapshot_at="2024-10-10T22:00:00Z", complete=True)])
    monkeypatch.setattr(feat, "_load_team_rosters", lambda *args: roster)
    schedule = pd.concat([games, target])
    slate = feat.build_slate_features(schedule, boxes, pd.DataFrame(), pd.DataFrame())
    decided = schedule.copy()
    decided.loc[decided.game_id == "target", ["home_score", "away_score"]] = [3, 1]
    history = feat.build_game_features(decided, boxes, pd.DataFrame(), pd.DataFrame())
    assert slate.g_home_name.iloc[0] == "current"
    assert history.g_home_name.iloc[-1] == "current"
    assert slate.goalie_sv_pct_home.iloc[0] == history.goalie_sv_pct_home.iloc[-1]


def test_slate_pending_rows_carry_entering_records_not_null():
    # Regression (2026-10-07 feature review): records were attached from
    # decided-only events keyed by game_id, so every pending game_id missed
    # the reindex and the published moneyline board shipped
    # "home_record"/"away_record": null for the whole slate.
    decided = pd.DataFrame([game("1", "2024-10-01")])          # ANA 3-1 BUF
    pending = pd.DataFrame([game("2", "2024-10-05", score=np.nan)])
    slate = feat.build_slate_features(pd.concat([decided, pending]),
                                      player_ratings=pd.DataFrame(),
                                      stints=pd.DataFrame())
    row = slate.loc[slate.game_id == "2"].iloc[0]
    assert row.home_record == "1-0"   # ANA's ENTERING record that season
    assert row.away_record == "0-1"


def test_second_pending_game_never_reads_phantom_half_win():
    # Regression (2026-10-07 feature review): undecided rows used to
    # fabricate team_win=0.5, so a team's SECOND pending game read a
    # phantom .500 from the first pending row in its trailing win% window.
    p1 = pd.DataFrame([game("p1", "2024-10-05", score=np.nan)])
    p2 = pd.DataFrame([game("p2", "2024-10-07", home="COL", away="ANA",
                           score=np.nan)])
    # Undecided rows carry NaN team_win — never a fabricated half-win.
    ev = feat.team_events(pd.concat([p1, p2]))
    assert ev.team_win.isna().all()
    # Slate with NO decided games: every win% is honestly NaN (pre-fix the
    # second pending row served 0.5 from the first pending row).
    slate = feat.build_slate_features(pd.concat([p1, p2]),
                                      player_ratings=pd.DataFrame(),
                                      stints=pd.DataFrame())
    assert slate[["win_pct_home", "win_pct_away"]].isna().all().all()
    # With decided games in the window the numbers are real win%s: ANA lost
    # on 10-01 and BUF lost on 10-03 — pending p1 must not pollute ANA's
    # window to .25, and pending rows never contribute a fabricated 0.5.
    d1 = pd.concat([pd.DataFrame([game("d1", "2024-10-01", home="ANA",
                                       away="COL", score=0),       # ANA L
                                  game("d2", "2024-10-03", home="BUF",
                                       away="COL", score=0),       # BUF L
                                  ]),
                    p1, p2])
    slate2 = feat.build_slate_features(d1, player_ratings=pd.DataFrame(),
                                       stints=pd.DataFrame())
    r1 = slate2.loc[slate2.game_id == "p1"].iloc[0]
    assert r1.win_pct_home == 0.0      # ANA's only prior result: a loss
    assert r1.win_pct_away == 0.0      # BUF's only prior result: a loss
    r2 = slate2.loc[slate2.game_id == "p2"].iloc[0]
    assert r2.win_pct_home == 1.0      # COL won both decided games
    assert r2.win_pct_away == 0.0      # ANA: [loss, pending] -> 0.0, not .25


@pytest.mark.parametrize("kind", ["rolling", "ewm", "pooled"])
def test_unplayed_fixtures_never_consume_statistical_history(kind):
    # Two targets must see the same last three SETTLED games, even with an
    # unplayed fixture between results and another pending target in between.
    frame = pd.DataFrame({
        "team": ["ANA"] * 8,
        "for": [1., 5., 2., 6., np.nan, 8., np.nan, np.nan],
        "against": [2., 2., 2., 2., np.nan, 2., np.nan, np.nan],
        "value": [1., 5., 2., 6., np.nan, 8., np.nan, np.nan],
        "attempts": [2., 8., 4., 9., np.nan, 12., np.nan, np.nan],
    })
    def calculate(f):
        if kind == "rolling":
            return feat._trailing_per_team(f, "value", 3)
        if kind == "ewm":
            return feat._trailing_ewm(f, "value", 2)
        return feat._trailing_pooled_ratio(f, "value", "attempts", 3)
    clean = frame.drop(index=[4, 6]).reset_index(drop=True)
    expected = calculate(clean)[-1]
    actual = calculate(frame)
    np.testing.assert_allclose(actual[-2:], expected, rtol=0, atol=1e-14)
    # Even a malformed pending boxscore carrying a value must not advance it.
    poisoned = frame.copy()
    poisoned.loc[[4, 6, 7], ["value", "attempts"]] = 10000.
    np.testing.assert_allclose(calculate(poisoned), actual, equal_nan=True,
                               rtol=0, atol=1e-14)


@pytest.mark.parametrize("kind", ["rolling", "ewm", "pooled"])
def test_settled_trailing_arithmetic_is_unchanged_and_current_target_is_excluded(kind):
    frame = pd.DataFrame({
        "team": ["ANA"] * 6 + ["BUF"] * 6,
        "for": np.arange(12, dtype=float) + 1,
        "against": np.ones(12),
        "value": [1., np.nan, 2., 6., 8., 3.] * 2,
        "attempts": [2., np.nan, 4., 9., 12., 5.] * 2,
    })
    def old_roll(col, operation):
        grouped = frame.groupby("team", sort=False)[col]
        post = (grouped.ewm(halflife=2, min_periods=1).mean()
                if operation == "ewm" else
                getattr(grouped.rolling(3, min_periods=1), operation)())
        return post.groupby(level=0).shift(1).reset_index(level=0, drop=True).to_numpy()
    if kind == "rolling":
        actual = feat._trailing_per_team(frame, "value", 3)
        expected = old_roll("value", "mean")
    elif kind == "ewm":
        actual = feat._trailing_ewm(frame, "value", 2)
        expected = old_roll("value", "ewm")
    else:
        actual = feat._trailing_pooled_ratio(frame, "value", "attempts", 3)
        expected = old_roll("value", "sum") / old_roll("attempts", "sum")
    np.testing.assert_array_equal(actual, expected)


def test_slate_statistics_are_invariant_to_unplayed_intermediate_fixture():
    history = pd.DataFrame([
        game(str(i), f"2024-10-{i + 1:02d}", score=float(i + 2))
        for i in range(6)])
    target = pd.DataFrame([game("target", "2024-10-10", score=np.nan)])
    extra = pd.DataFrame([game("unplayed", "2024-10-08", score=np.nan)])
    kwargs = {"player_ratings": pd.DataFrame(), "stints": pd.DataFrame()}
    clean = feat.build_slate_features(pd.concat([history, target]), **kwargs)
    with_extra = feat.build_slate_features(pd.concat([history, extra, target]), **kwargs)
    columns = ["win_pct_home", "win_pct_away", "ewm_net_goals_home",
               "ewm_net_goals_away", "ewm_goal_share_home", "ga_per_game_home"]
    np.testing.assert_array_equal(
        clean.loc[clean.game_id == "target", columns].to_numpy(),
        with_extra.loc[with_extra.game_id == "target", columns].to_numpy())
    # Rest is schedule-derived and intentionally not part of this invariant.
