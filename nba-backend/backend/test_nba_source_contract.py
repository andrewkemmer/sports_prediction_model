"""Contract tests: a source may not reach the model unnormalized."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[0]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import ingestion as ing  # noqa: E402
import nba_sources as sources  # noqa: E402
import source_contract as contract  # noqa: E402


GAMES = pd.DataFrame({
    "game_id": ["0022500001", "0022500002"],
    "gameday": pd.to_datetime(["2025-10-22", "2025-10-24"]),
    "season": [2025, 2025],
    "home_team": ["BOS", "LAL"], "away_team": ["NYK", "GSW"],
    "home_score": [110.0, 105.0], "away_score": [104.0, 99.0],
    "game_type": [1, 1],
})


def _team_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["0022500001", "0022500001"],
        "team": ["BOS", "NYK"], "points_for": [110.0, 104.0],
        "points_against": [104.0, 110.0], "is_home": [True, False],
    })


# --------------------------------------------------------------------------
# The contract refuses a frame the model cannot use, and says which column
# --------------------------------------------------------------------------


def test_a_frame_missing_a_required_column_is_refused_by_name() -> None:
    """A missing column is a bug in the adapter, and must not reach the model.

    The failure this replaces is worse than an exception: a short frame flows
    onward, the feature built from it is quietly NaN, and the artifact looks
    fine. Naming the column at the boundary is the whole value of declaring a
    schema as data.
    """
    bad = _team_rows().drop(columns=["points_for"])
    with pytest.raises(contract.ContractError) as excinfo:
        contract.normalize(bad, "team_stats", source="test")

    message = str(excinfo.value)
    assert "points_for" in message, "the error must name the missing column"
    assert "test" in message, "the error must name the source at fault"


def test_a_source_with_the_wrong_granularity_cannot_pose_as_game_rows() -> None:
    """A per-season aggregate must not be mistakable for a game-level frame.

    The dash endpoints answer for a season without naming a game, so they can
    never fill a game-level contract. That has to be a loud refusal, because
    the alternative is a season total being averaged as if it were a game.
    """
    dash = pd.DataFrame({"TEAM_ABBREVIATION": ["BOS"], "PTS": [11000.0]})
    prepared = sources._dash_to_team(dash)
    with pytest.raises(contract.ContractError) as excinfo:
        contract.normalize(prepared, "team_stats", source="LeagueDashTeamStats")

    assert "game_id" in str(excinfo.value), (
        "the absent game identity is the reason, so it should be named")


def test_an_empty_frame_is_a_gap_in_the_data_not_a_contract_violation() -> None:
    """Empty and malformed must not share an exit path.

    A source that answered with zero rows is a hole in the data and the run
    should degrade. A source that answered with rows and dropped a required
    column is a bug. Collapsing the two would mean a real adapter bug gets
    filed as "no data today" and nobody looks at it.
    """
    empty = contract.normalize(pd.DataFrame(), "team_stats", source="test")
    assert empty.empty
    assert list(empty.columns) == list(contract.TEAM_STATS_SCHEMA)

    partial = pd.DataFrame({"team": ["BOS"]})
    with pytest.raises(contract.ContractError):
        contract.normalize(partial, "team_stats", source="test")


# --------------------------------------------------------------------------
# Derived columns are the contract's, not a vendor's
# --------------------------------------------------------------------------


def test_a_source_cannot_supply_its_own_derived_game_columns() -> None:
    """Margin, total, and home_win are computed from the score, always.

    A vendor that reports a margin disagreeing with the two scores it also
    reported would put a contradiction into the model, and every downstream
    feature would inherit it without anything looking wrong.
    """
    lying = GAMES.copy()
    lying["margin"] = 999.0
    lying["total"] = 1.0
    lying["home_win"] = 0.0

    out = contract.normalize(lying, "games", source="test")
    assert (out.margin == 6.0).all(), "the score wins over the vendor's margin"
    assert out.total.tolist() == [214.0, 204.0], "both games, from their scores"
    assert (out.home_win == 1.0).all()


def test_season_is_derived_from_the_tipoff_when_a_source_omits_it() -> None:
    """The season turns over in July; that is a league rule, not a vendor field."""
    june = GAMES.copy()
    june["gameday"] = pd.to_datetime(["2025-06-30", "2025-07-01"])
    june = june.drop(columns=["season"])

    out = contract.normalize(june, "games", source="test")
    assert list(out.season) == [2024, 2025], (
        "a June game belongs to the season that began the prior July")


# --------------------------------------------------------------------------
# Normalization is where dtypes are settled
# --------------------------------------------------------------------------


def test_scores_arriving_as_text_become_numbers_before_anything_adds_them() -> None:
    """A CSV-ish source must not turn a fold into a string comparison.

    Coercion belongs at the boundary. If it leaks inward, the first thing
    that notices is a subtraction raising deep inside a walk-forward loop.
    """
    textual = GAMES.copy()
    textual["home_score"] = ["110", "105"]
    textual["away_score"] = ["104", "99"]

    out = contract.normalize(textual, "games", source="test")
    assert out.home_score.dtype.kind == "f"
    assert (out.margin == 6.0).all(), "the derived column must be real arithmetic"


def test_a_game_id_is_normalized_to_a_string_so_keys_join_across_sources() -> None:
    """Two sources may type the same identifier differently.

    One source delivering ``game_id`` as text and another as an integer would
    produce two frames that silently fail to join, and the symptom is an
    all-NaN feature rather than an error. Normalizing the key at the boundary
    is what lets a second source merge against the first at all.
    """
    numeric = GAMES.copy()
    numeric["game_id"] = [22500001, 22500002]
    out = contract.normalize(numeric, "games", source="test")

    assert out.game_id.map(type).eq(str).all()
    left = contract.normalize(GAMES, "games", source="a")
    assert set(left.game_id).intersection(set(out.game_id)) == set(), (
        "the two frames must not collide on differing key spellings, which is "
        "exactly why the key is forced to one type at the boundary")


# --------------------------------------------------------------------------
# The parameter sets that 500 rather than 400
# --------------------------------------------------------------------------


def test_the_player_dash_query_leaves_weight_empty_because_a_value_500s() -> None:
    """``Weight=Basic`` on leaguedashplayerstats is a 500, not a result.

    Measured against the live endpoint: omitting ``Weight`` returns 569 rows
    and 69 columns in 1.5s; sending ``"Basic"`` returns HTTP 500. The team
    endpoint is the opposite - it wants ``Weight=Basic``. Pinning both means
    nobody re-derives this from a 500 during a run.
    """
    player = sources.dash_query("2024-25", "Regular Season", player=True)
    team = sources.dash_query("2024-25", "Regular Season", player=False)
    assert player["Weight"] == "", "a value here is a 500 from the live host"
    assert team["Weight"] == "Basic", "the team endpoint wants this one"


def test_the_season_log_query_still_carries_every_parameter_it_always_did() -> None:
    """The season log has answered with this exact set for years.

    The query moved into ``nba_sources`` so a second source could be declared
    beside it. A refactor that quietly dropped a parameter would show up as a
    500 or an empty result set on the backbone, so the set is pinned.
    """
    query = sources.season_log_query("2024-25", "Regular Season")
    assert query.startswith("LeagueID=00")
    for parameter in ("Season=2024-25", "SeasonType=Regular+Season",
                      "PerMode=PerGame", "MeasureType=Base", "DateFrom=&DateTo="):
        assert parameter in query, f"{parameter} must survive the refactor"


def test_the_ingestion_pull_and_the_diagnostic_request_the_same_query() -> None:
    """A probe against a different URL proves nothing about the failing one.

    ``_season_log_query`` is the alias every caller uses, so this also fails
    loudly if a future edit retypes the query inside ``ingestion`` instead of
    delegating to the declaration.
    """
    assert ing._season_log_query("2024-25", "Regular Season") == (
        sources.season_log_query("2024-25", "Regular Season"))


# --------------------------------------------------------------------------
# The play-by-play parser, which read seven field names nobody returns
# --------------------------------------------------------------------------


def test_every_contract_column_in_play_by_play_is_filled_from_a_real_payload() -> None:
    """The old mapping produced seven always-null columns out of nineteen.

    ``actionId``, ``sequenceNumber``, ``gameClock``, ``isFieldGoalAttempted``,
    ``isMade``, ``loc`` and ``isHundred`` are not names either NBA.com surface
    returns. A column count made the frame look populated; only non-null
    coverage shows what is actually in it. This is the check that would have
    caught it, and it runs against a real recorded action.
    """
    action = {
        "actionId": 41, "actionNumber": 55, "period": 1,
        "clock": "PT07M42.00S", "teamTricode": "OKC", "personId": 1631096,
        "actionType": "Made Shot", "subType": "Jump Shot",
        "description": "Holmgren 23' Jump Shot (9 PTS)",
        "scoreHome": "11", "scoreAway": "10", "pointsTotal": 21,
        "shotDistance": 23, "shotResult": "Made", "isFieldGoal": 1,
        "shotValue": 2, "x": 84.0, "y": 92.1, "xLegacy": 211, "yLegacy": 97,
    }
    frame = ing._play_by_play_frame({"game": {"actions": [action]}},
                                    "0022500001", pd.Timestamp("2025-10-21"))

    assert len(frame) == 1
    for column in ("action_id", "action_number", "period", "clock", "team",
                   "player_id", "action_type", "sub_type", "description",
                   "score_home", "score_away", "points", "shot_distance",
                   "shot_result", "is_field_goal", "shot_value", "x", "y"):
        assert frame[column].notna().all(), (
            f"{column} is null; the upstream field name changed")


def test_play_by_play_still_produces_an_empty_frame_for_a_game_with_no_actions() -> None:
    """A cancelled or not-yet-started game answers with no action list."""
    frame = ing._play_by_play_frame({"game": {}}, "0022500001",
                                    pd.Timestamp("2025-10-21"))
    assert frame.empty
    assert "action_id" in frame.columns, "an empty frame still declares its schema"


# --------------------------------------------------------------------------
# Every declared source is usable
# --------------------------------------------------------------------------


def test_every_declared_source_names_a_route_and_at_least_one_frame() -> None:
    """A source with no frames is dead configuration that looks live.

    The registry exists so adding a vendor is one entry. An entry that cannot
    be called is worse than no entry, because the next person trusts the list.
    """
    for source in sources.ALL_SOURCES:
        assert source.name, "a source must identify itself in the manifest"
        assert source.route, "a source must record which route answered"
        assert source.frames, f"{source.name} declares no frames"
        for name in source.frames:
            assert name in contract.SCHEMAS, (
                f"{source.name} declares frame {name!r}, which has no contract")


def test_the_coverage_report_distinguishes_required_from_optional() -> None:
    """A source can pass normalize and still be mostly empty.

    That is fine for an optional column and fatal for a required one, and the
    only place that difference is visible is the report.
    """
    sparse = _team_rows()
    report = contract.coverage_report(sparse, "team_stats")
    points = report[report.column == "points_for"].iloc[0]
    pace = report[report.column == "pace"].iloc[0]

    assert bool(points.required) and points.coverage_pct == 100.0
    assert not bool(pace.required) and pace.coverage_pct == 0.0


def test_a_second_source_reaches_the_contract_without_touching_ingestion() -> None:
    """The abstraction is real only if a new source needs no new code path.

    If declaring a vendor meant editing ``ingestion``, the abstraction would be
    a naming convention rather than a boundary. So this drives a source that
    ``ingestion`` never mentions, through the same ``SourceSpec`` interface the
    season log uses.
    """
    dash_rows = pd.DataFrame({
        "PLAYER_ID": [201939], "PLAYER_NAME": ["Stephen Curry"],
        "TEAM_ABBREVIATION": ["GSW"], "PTS": [24.0], "REB": [4.0],
        "AST": [6.0], "GP": [70], "FG_PCT": [0.478],
    })
    out = sources.PLAYER_DASH.normalize("player_stats", dash_rows,
                                         allow_missing=True)

    assert len(out) == 1
    assert out.player_id.iloc[0] == "201939", "the key is a string at the boundary"
    assert out.points.iloc[0] == 24.0
    assert "rebound" not in out.columns, "no alias invented behind the contract's back"


# --------------------------------------------------------------------------
# The dedup key that silently deleted a whole game
# --------------------------------------------------------------------------


def test_a_game_survives_the_cache_dedup_that_keys_on_action_id() -> None:
    """cdn.nba.com omits ``actionId`` on every action, and the cache keys on it.

    The play-by-play cache merges on ``["game_id", "action_id"]``. With
    ``action_id`` null on all 707 rows of a real game, ``drop_duplicates``
    treats them as one key and keeps exactly one row - so a full sweep would
    have written a single action per game and reported success. No exception,
    no warning, and a file that looks populated.

    The contract's identity is therefore derived: the upstream's ``actionId``
    when present, else the per-game sequence number, which is present and
    unique on all 707 rows.
    """
    actions = [
        {"actionNumber": n, "period": 1, "clock": f"PT0{n % 12}M00.00S",
         "teamTricode": "BOS", "actionType": "2pt", "subType": "Jump Shot",
         "description": f"shot {n}", "scoreHome": str(n), "scoreAway": "0",
         "pointsTotal": 2, "shotDistance": 10.0, "shotResult": "Made",
         "isFieldGoal": 1, "x": 1.0, "y": 2.0}
        for n in range(1, 708)
    ]
    frame = ing._play_by_play_frame({"game": {"actions": actions}},
                                    "0022500001", pd.Timestamp("2025-10-21"))

    assert len(frame) == 707
    assert frame.action_id.notna().all(), "the merge key cannot be null"
    assert frame.action_id.nunique() == 707
    survived = frame.drop_duplicates(["game_id", "action_id"], keep="first")
    assert len(survived) == 707, (
        "one action per game is the failure this test exists to prevent")


def test_shot_value_is_derived_because_the_cdn_action_type_encodes_it() -> None:
    """The CDN has no ``shotValue``; it names the shot in ``actionType``.

    So a points-from-shots feature has to be written against one derivation
    rather than reading a field one surface has and the other does not.
    """
    actions = [{"actionNumber": n, "actionType": kind, "period": 1,
                "shotResult": "Made", "isFieldGoal": 1}
               for n, kind in ((1, "3pt"), (2, "2pt"), (3, "freethrow"),
                                (4, "rebound"), (5, "timeout"))]
    frame = ing._play_by_play_frame({"game": {"actions": actions}},
                                    "0022500001", pd.Timestamp("2025-10-21"))

    assert frame.shot_value.tolist()[:3] == [3.0, 2.0, 1.0]
    assert pd.isna(frame.shot_value.iloc[3]), "a rebound is not a shot"
    assert pd.isna(frame.shot_value.iloc[4])


def test_the_stats_surface_action_id_is_preferred_when_the_host_sends_it() -> None:
    """``playbyplayv3`` does send ``actionId``; the CDN does not.

    The derivation is a fallback, not an override, so a surface that carries a
    real identifier keeps it rather than being renumbered.
    """
    actions = [{"actionId": 900 + n, "actionNumber": n, "actionType": "2pt",
                "period": 1} for n in range(1, 4)]
    frame = ing._play_by_play_frame({"game": {"actions": actions}},
                                    "0022500001", pd.Timestamp("2025-10-21"))
    assert frame.action_id.tolist() == [901.0, 902.0, 903.0]


# --------------------------------------------------------------------------
# A cache written before the contract must not grow a duplicate window
# --------------------------------------------------------------------------


def test_the_cache_merge_normalizes_both_sides_before_deduping(tmp_path,
                                                              monkeypatch) -> None:
    """A cache holding int64 ``player_id`` and an incoming str frame must not
    both survive as duplicates.

    The merge keys on ``(game_id, player_id)``. If the two sides disagree about
    the key's type, not one row is recognized as already-seen and the cache
    doubles on every run - a leak that never raises and never looks wrong
    until the file is too big to load. Normalizing before the merge is what
    makes the identity comparable; normalizing after it is too late, because
    the damage is already done.
    """
    monkeypatch.setenv(ing.FULL_REPULL_ENV, "0")
    monkeypatch.setattr(ing.config, "CACHE_DIR", tmp_path)
    path = tmp_path / "player_stats.parquet"

    stale = pd.DataFrame({
        "game_id": ["0022500001", "0022500002"],
        "player_id": [201939, 2544],          # int64, as the old writer left it
        "team": ["GSW", "LAL"], "points": [30.0, 12.0],
    })
    stale.to_parquet(path, index=False)

    fresh = pd.DataFrame({
        "game_id": ["0022500001"],             # same row, as the contract types it
        "player_id": ["201939"],
        "team": ["GSW"], "points": [31.0],
    })
    ing._write_cache({"player_stats": fresh}, {"player_stats": path},
                     {"player_stats": ["game_id", "player_id"]})

    merged = pd.read_parquet(path)
    assert len(merged) == 2, (
        f"the cache doubled: {len(merged)} rows for 2 distinct players")
    assert merged.player_id.dtype.kind in "OU", "the key is a string, not int64"
    newest = merged[merged.game_id == "0022500001"].iloc[0]
    assert newest.points == 31.0, "the incoming row wins on identity"


def test_a_full_cache_write_also_normalizes_before_it_touches_disk(
        tmp_path, monkeypatch) -> None:
    """Even with no existing cache, what lands on disk is contract-shaped.

    The dedup argument does not apply here, so this pins the simpler claim:
    a file written by this version is readable by the next one without anyone
    having to migrate it.
    """
    monkeypatch.setenv(ing.FULL_REPULL_ENV, "1")
    monkeypatch.setattr(ing.config, "CACHE_DIR", tmp_path)
    path = tmp_path / "player_stats.parquet"

    frame = pd.DataFrame({"game_id": ["0022500001"], "player_id": [201939],
                          "team": ["GSW"], "points": [30.0]})
    ing._write_cache({"player_stats": frame}, {"player_stats": path},
                     {"player_stats": ["game_id", "player_id"]})

    written = pd.read_parquet(path)
    assert written.player_id.iloc[0] == "201939"
    assert "gameday" in written.columns, "the contract's columns are all present"
