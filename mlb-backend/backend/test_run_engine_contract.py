"""Offline run-engine contract tests.

Run with: python mlb-backend/backend/test_run_engine_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from datetime import date

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import distributions as re
import data_ingestion as ingestion
from data_ingestion import build_upcoming_slate
from results import _attach_slate_lineup_keys, _count_evening_games


def _market_row(**overrides):
    row = {
        "game_pk": [101], "game_date": ["2026-09-18"],
        "home_expected_runs": [4.0], "away_expected_runs": [4.2],
    }
    row.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(row)


def _espn_event(event_id: str, timestamp: str, home: str = "NYY",
                away: str = "BOS") -> dict:
    """Minimal ESPN event shape for ET/UTC schedule boundary tests."""
    return {
        "id": event_id,
        "date": timestamp,
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": home}, "score": "0"},
                {"homeAway": "away", "team": {"abbreviation": away}, "score": "0"},
            ],
            "status": {"type": {"state": "pre", "detail": "Scheduled"}},
            "venue": {"fullName": "Test Park"},
        }],
    }


def test_contract_rejects_lambda_mismatch():
    oof = _market_row()
    markets = _market_row(home_expected_runs=4.1)
    result = re.build_artifact_contract(oof, markets)
    assert result["alignment_status"] == "misaligned"
    assert result["max_abs_home_expected_runs_diff"] == 0.1


def test_contract_rejects_date_mismatch():
    oof = _market_row()
    markets = _market_row(game_date="2026-09-19")
    result = re.build_artifact_contract(oof, markets)
    assert result["alignment_status"] == "misaligned"
    assert result["date_mismatch_rows"] == 1


def test_contract_accepts_post_edge_state():
    oof = _market_row()
    markets = _market_row(kind="oof")
    result = re.build_artifact_contract(oof, markets)
    assert result["alignment_status"] == "aligned"
    assert result["oof_rows"] == result["market_oof_rows"] == 1


def test_contract_rejects_duplicate_keys():
    oof = pd.concat([_market_row(), _market_row()], ignore_index=True)
    markets = _market_row(kind="oof")
    result = re.build_artifact_contract(oof, markets)
    assert result["alignment_status"] == "misaligned"
    assert result["duplicate_oof_keys"] == 1


def test_team_identity_is_persisted_in_oof_market_frame():
    # The market constructor's OOF frame must retain the same identity fields
    # used by the history dashboard; this prevents — @ — when the feature
    # snapshot lags the market artifact.
    assert "game_id" in re.MARKET_COLUMNS_V3
    assert "home_team" in re.MARKET_COLUMNS_V3
    assert "away_team" in re.MARKET_COLUMNS_V3


def test_upcoming_slate_carries_adopted_exp2_sources_pit_safely():
    """The adopted exp2 features must not disappear on the upcoming slate."""
    row = {
        "game_date": "2026-09-20", "start_time_utc": "2026-09-20T18:00:00",
        "game_id": "20260920_AWAY@HOME", "home_team": "HOME", "away_team": "AWAY",
        "home_win": 1.0, "home_score": 5, "away_score": 3,
        "home_starter_id": 101, "away_starter_id": 202,
        "sp_k9_home": 8.0, "sp_k9_away": 7.0,
        "team_k_rate_30g_home": 0.22, "team_k_rate_30g_away": 0.25,
        "opp_lefty_share_home": 0.40, "opp_lefty_share_away": 0.60,
        "sp_k_pct_fb_vs_l_home": 0.22, "sp_k_pct_fb_vs_l_away": 0.21,
        "sp_k_pct_fb_vs_r_home": 0.24, "sp_k_pct_fb_vs_r_away": 0.23,
        "team_k_pct_fb_vs_l_home": 0.19, "team_k_pct_fb_vs_l_away": 0.20,
        "team_k_pct_fb_vs_r_home": 0.21, "team_k_pct_fb_vs_r_away": 0.22,
        "league_k_pct": 0.23, "league_k_pct_fb_vs_l": 0.20,
        "league_k_pct_fb_vs_r": 0.21,
    }
    for cat, offset in (("fastball", 0.00), ("breaking", 0.02), ("offspeed", 0.04)):
        row[f"sp_k_pct_cat_{cat}_home"] = 0.22 + offset
        row[f"sp_k_pct_cat_{cat}_away"] = 0.21 + offset
        row[f"sp_xwoba_cat_{cat}_home"] = 0.31 + offset
        row[f"sp_xwoba_cat_{cat}_away"] = 0.30 + offset
        row[f"sp_usage_cat_{cat}_home"] = 0.50
        row[f"sp_usage_cat_{cat}_away"] = 0.50
        row[f"team_k_pct_cat_{cat}_home"] = 0.28 + offset
        row[f"team_k_pct_cat_{cat}_away"] = 0.26 + offset
        row[f"team_xwoba_cat_{cat}_home"] = 0.32 + offset
        row[f"team_xwoba_cat_{cat}_away"] = 0.30 + offset
        row[f"league_k_pct_cat_{cat}"] = 0.23 + offset / 2
        row[f"league_xwoba_cat_{cat}"] = 0.31 + offset / 2

    history = pd.DataFrame([row])
    history["game_date"] = pd.to_datetime(history["game_date"])
    history["start_time_utc"] = pd.to_datetime(history["start_time_utc"])
    schedule = pd.DataFrame([{
        "game_id": "20260921_AWAY@HOME", "game_date": "2026-09-21",
        "start_time_utc": "2026-09-21T18:00:00", "home_team": "HOME",
        "away_team": "AWAY", "venue": "Test Park", "sp_id_home": 101,
        "sp_id_away": 202, "sp_name_home": "Home Starter",
        "sp_name_away": "Away Starter",
    }])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"])
    slate = build_upcoming_slate(history, date(2026, 9, 21), schedule_df=schedule)

    assert len(slate) == 1
    # Sources are carried from the prior game and are not lost because the
    # target-date row has no completed-game data.
    assert slate.loc[0, "league_k_pct"] == 0.23
    assert pd.notna(slate.loc[0, "team_k_pct_cat_fastball_home"])
    assert pd.notna(slate.loc[0, "sp_k_pct_cat_fastball_home"])

    from features import add_exp2_features, EXP2_CANDIDATE_COLS
    enriched = add_exp2_features(slate)
    assert enriched[EXP2_CANDIDATE_COLS].notna().any(axis=1).all()


def test_upcoming_slate_tbd_away_pitcher_ships_nan_not_zero():
    """An unannounced away probable (TBD) must leave the away-side SP and
    exp2 features NaN — never fabricated 0, never a crash, and the home
    side still resolves (the 2026-09-29 shape: 2 home starters announced,
    all 4 away TBD). Silence here is how that run hid inside a green log;
    build_upcoming_slate now also WARNs per unresolved slot."""
    row = {
        "game_date": "2026-09-20", "start_time_utc": "2026-09-20T18:00:00",
        "game_id": "20260920_AWAY@HOME", "home_team": "HOME", "away_team": "AWAY",
        "home_win": 1.0, "home_score": 5, "away_score": 3,
        "home_starter_id": 101, "away_starter_id": 202,
        "sp_k9_home": 8.0, "sp_k9_away": 7.0,
        "team_k_rate_30g_home": 0.22, "team_k_rate_30g_away": 0.25,
        "league_k_pct": 0.23,
    }
    for cat in ("fastball", "breaking", "offspeed"):
        row[f"sp_k_pct_cat_{cat}_home"] = 0.22
        row[f"sp_k_pct_cat_{cat}_away"] = 0.21
        row[f"sp_usage_cat_{cat}_home"] = 0.50
        row[f"sp_usage_cat_{cat}_away"] = 0.50
        row[f"league_k_pct_cat_{cat}"] = 0.23
    history = pd.DataFrame([row])
    history["game_date"] = pd.to_datetime(history["game_date"])
    history["start_time_utc"] = pd.to_datetime(history["start_time_utc"])
    schedule = pd.DataFrame([{
        "game_id": "20260921_AWAY@HOME", "game_date": "2026-09-21",
        "start_time_utc": "2026-09-21T18:00:00", "home_team": "HOME",
        "away_team": "AWAY", "venue": "Test Park",
        "sp_id_home": 101, "sp_name_home": "Home Starter",
        # Away starter never announced: no id, placeholder name.
        "sp_id_away": np.nan, "sp_name_away": "TBD",
    }])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"])

    slate = build_upcoming_slate(history, date(2026, 9, 21), schedule_df=schedule)
    assert len(slate) == 1
    assert slate.loc[0, "sp_name_away"] == "TBD"
    # Away side carries NO pitcher state; home side carries its own.
    assert pd.isna(slate.loc[0, "sp_k9_away"])
    assert slate.loc[0, "sp_k9_home"] == 8.0

    from features import add_exp2_features
    enriched = add_exp2_features(slate)
    # Home-side half computes; away-side half and the diff propagate NaN.
    assert pd.notna(enriched.loc[0, "exp2_centered_k_home"])
    assert pd.isna(enriched.loc[0, "exp2_centered_k_away"])
    assert pd.isna(enriched.loc[0, "exp2_centered_k_diff"])


def test_slate_lineup_keys_are_carried_before_feature_join():
    """StatsAPI lineup identities must reach the slate feature join."""
    slate = pd.DataFrame({"game_id": ["g1", "g2"],
                          "home_team": ["HOME", "HOME2"]})
    lineup_rows = pd.DataFrame({"game_pk": [101, pd.NA],
                                "home_order": [[], None],
                                "away_order": [[], None]})
    out = _attach_slate_lineup_keys(slate, lineup_rows)
    assert list(out["game_pk"].astype("Int64")) == [101, pd.NA]
    assert "game_pk" not in slate.columns


def test_posted_lineup_smoke_populates_all_six_deltas():
    """A posted lineup with a resolved game_pk must enrich every delta."""
    import features
    from features import LINEUP_DELTA_COLS, add_lineup_delta_features

    day = pd.Timestamp("2026-09-21")
    features._lineup_cache.clear()
    features._lineup_cache.update({
        "lineups": pd.DataFrame(),
        "batter": pd.DataFrame({
            "season": [2026] * 4,
            "game_date": [day] * 4,
            "batter": [1, 2, 3, 4],
            "sd_woba": [0.34, 0.35, 0.36, 0.37],
            "prior_pa": [100] * 4,
        }),
        "team": pd.DataFrame({
            "season": [2026] * 2,
            "game_date": [day] * 2,
            "team": ["HOME", "AWAY"],
            "sd_woba": [0.32, 0.33],
            "top3_woba": [0.35, 0.36],
            "top5_ids": ["[1, 2, 3]", "[2, 3, 4]"],
        }),
    })
    try:
        slate = pd.DataFrame({
            "game_pk": pd.Series([101], dtype="Int64"),
            "game_date": [day],
            "home_team": ["HOME"],
            "away_team": ["AWAY"],
        })
        lineups = pd.DataFrame({
            "game_pk": [101],
            "home_order": [[1, 2]],
            "away_order": [[3, 4]],
        })
        out = add_lineup_delta_features(slate, lineups_override=lineups)
        assert out[LINEUP_DELTA_COLS].notna().all().all()
    finally:
        features._lineup_cache.clear()


def test_lineup_override_with_unresolved_pk_joins_without_crash():
    """Regression (2026-09-22): a lineup override keyed by a MIXED int/NA
    game_pk lands as object dtype, and pandas refuses the object-vs-Int64
    key merge with a hard ValueError — killing the entire slate build so
    zero dated artifacts shipped for the day. The enrichment must coerce
    the key and join cleanly: resolved games enrich, unresolved games ship
    NaN instead of crashing."""
    import features
    from features import LINEUP_DELTA_COLS, add_lineup_delta_features

    day = pd.Timestamp("2026-09-22")
    features._lineup_cache.clear()
    features._lineup_cache.update({
        "lineups": pd.DataFrame(),
        "batter": pd.DataFrame({
            "season": [2026] * 4,
            "game_date": [day] * 4,
            "batter": [1, 2, 3, 4],
            "sd_woba": [0.34, 0.35, 0.36, 0.37],
            "prior_pa": [100] * 4,
        }),
        "team": pd.DataFrame({
            "season": [2026] * 2,
            "game_date": [day] * 2,
            "team": ["HOME", "AWAY"],
            "sd_woba": [0.32, 0.33],
            "top3_woba": [0.35, 0.36],
            "top5_ids": ["[1, 2, 3]", "[2, 3, 4]"],
        }),
    })
    try:
        slate = pd.DataFrame({
            "game_pk": pd.Series([101, 102], dtype="Int64"),
            "game_date": [day, day],
            "home_team": ["HOME", "AWAY"],
            "away_team": ["AWAY", "HOME"],
        })
        # The exact production shape from _fetch_slate_lineups: one resolved
        # int mixed with one pd.NA — a bare DataFrame of this is object dtype.
        lineups = pd.DataFrame([
            {"game_pk": 101, "home_order": [1, 2], "away_order": [3, 4]},
            {"game_pk": pd.NA, "home_order": None, "away_order": None},
        ])
        assert lineups["game_pk"].dtype == object
        out = add_lineup_delta_features(slate, lineups_override=lineups)
        # Resolved game fully enriched from the posted order (team sd-wOBA
        # 0.32/0.33, batters 0.34/0.35 and 0.36/0.37). Unresolved game falls
        # to the team-baseline fallback and is NULLed downstream by the
        # slate's posted-lineup mask — the contract here is only: NO CRASH
        # on the mixed int/NA object-dtype key.
        assert abs(out.loc[0, "lineup_actual_woba_delta_home"] - 0.025) < 1e-9
        assert abs(out.loc[0, "lineup_actual_woba_delta_away"] - 0.035) < 1e-9
        assert len(out) == 2
    finally:
        features._lineup_cache.clear()


def test_et_schedule_probe_includes_next_utc_rollover():
    """An ET slate includes its 00:00Z game but not the next ET day."""
    target = date(2026, 9, 29)
    calls: list[date] = []

    def fake_scoreboard(query_date):
        calls.append(query_date)
        if query_date == target:
            return [_espn_event("target", "2026-09-29T23:00:00Z", "NYY", "BOS")]
        if query_date == date(2026, 9, 30):
            return [
                _espn_event("rollover", "2026-09-30T00:00:00Z", "TOR", "BOS"),
                _espn_event("next-et", "2026-09-30T04:00:00Z", "SEA", "TOR"),
            ]
        return []

    with patch.object(ingestion, "_fetch_espn_scoreboard", side_effect=fake_scoreboard), \
         patch.object(ingestion, "_fetch_statsapi_pitchers", return_value={}):
        schedule = ingestion.load_espn_schedule(target)

    assert calls == [target, date(2026, 9, 30)]
    assert set(schedule["game_id"]) == {
        "20260929_BOS@NYY", "20260929_BOS@TOR",
    }
    rollover = schedule[schedule["game_id"] == "20260929_BOS@TOR"].iloc[0]
    assert pd.Timestamp(rollover["start_time_utc"]).tz_convert(
        "America/New_York").strftime("%Y-%m-%d %H:%M") == "2026-09-29 20:00"
    assert _count_evening_games(schedule) == 2
    assert "20260930_TOR@SEA" not in set(schedule["game_id"])


def test_run_line_opposite_tail_is_not_home_dog_tail():
    # A deterministic draw matrix is unnecessary here: the two output arrays
    # have separate contracts and are populated independently by the MC path.
    assert "p_rl_1_5_away_favorite" == re.rl_col(1.5, "away_favorite")
    assert "p_rl_1_5_home_dog" == re.rl_col(1.5, "home_dog")
    assert "p_rl_1_5_away_favorite" in re.MARKET_COLUMNS_V3



class _LogRecorder:
    """Zero-dep logger stand-in so the contract runner can assert warnings."""
    def __init__(self):
        self.messages = []

    def warning(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else str(msg))

    def info(self, msg, *args, **kwargs):
        pass


def _invariant_board():
    return pd.DataFrame({
        "game_id": ["20260928_NYY@BOS", "20260928_LAD@SF",
                    "20260925_CHC@BOS_2", "20260926_TB@PHI", "20260927_SEA@TOR"],
        "game_date": ["2026-09-28", "2026-09-28", "2026-09-25",
                      "2026-09-26", "garbage"],
        "home_team": ["BOS", "SF", "BOS", "PHI", "TOR"],
        "away_team": ["NYY", "LAD", "CHC", "TB", "SEA"],
    })


def test_board_date_invariant_drops_foreign_and_unparseable_rows():
    """The 2026-09-28 regression contract: a board is EXACTLY its date.

    Rows dated otherwise (the recycled finals from 0925/0926 that shipped
    under a September 28 header) and rows whose game_date cannot be parsed
    must all be dropped, with a loud warning; same-date rows survive.
    """
    board = _invariant_board()
    rec = _LogRecorder()
    with patch.object(ingestion, "logger", rec):
        out = ingestion.enforce_board_date_invariant(board, date(2026, 9, 28))
    assert list(out["game_id"]) == ["20260928_NYY@BOS", "20260928_LAD@SF"]
    assert out is not board  # filtered result is a defensive copy
    assert len(rec.messages) == 1
    assert "2026-09-25" in rec.messages[0]
    assert "2026-09-26" in rec.messages[0]


def test_board_date_invariant_keeps_clean_board_and_empty_boards():
    """A fully same-date board passes through unfiltered; empty frames and
    frames without a game_date column are returned unchanged (an empty
    frame in, honest empty board out)."""
    board = _invariant_board()
    same_date = board.iloc[[0, 1]].reset_index(drop=True)
    rec = _LogRecorder()
    with patch.object(ingestion, "logger", rec):
        out = ingestion.enforce_board_date_invariant(same_date, date(2026, 9, 28))
    assert len(out) == 2
    assert rec.messages == []

    empty = pd.DataFrame(columns=["game_id", "game_date"])
    assert ingestion.enforce_board_date_invariant(empty, date(2026, 9, 28)).empty
    no_date_col = pd.DataFrame({"game_id": ["x"]})
    assert list(ingestion.enforce_board_date_invariant(
        no_date_col, date(2026, 9, 28))["game_id"]) == ["x"]


def _slate_history_row():
    row = {
        "game_date": "2026-09-20", "start_time_utc": "2026-09-20T18:00:00",
        "game_id": "20260920_AWAY@HOME", "home_team": "HOME", "away_team": "AWAY",
        "home_win": 1.0, "home_score": 5, "away_score": 3,
        "home_starter_id": 101, "away_starter_id": 202,
        "sp_k9_home": 8.0, "sp_k9_away": 7.0,
        "team_k_rate_30g_home": 0.22, "team_k_rate_30g_away": 0.25,
        "league_k_pct": 0.23,
    }
    for cat in ("fastball", "breaking", "offspeed"):
        row[f"sp_k_pct_cat_{cat}_home"] = 0.22
        row[f"sp_k_pct_cat_{cat}_away"] = 0.21
        row[f"sp_usage_cat_{cat}_home"] = 0.50
        row[f"sp_usage_cat_{cat}_away"] = 0.50
        row[f"league_k_pct_cat_{cat}"] = 0.23
    history = pd.DataFrame([row])
    history["game_date"] = pd.to_datetime(history["game_date"])
    history["start_time_utc"] = pd.to_datetime(history["start_time_utc"])
    return history


def test_upcoming_slate_rejects_foreign_date_schedule_rows():
    """Defense in depth for the 2026-09-28 regression: a schedule source
    that walks back to earlier dates must not let decided games masquerade
    as the target-date slate. Only target-date rows survive into features."""
    history = _slate_history_row()
    schedule = pd.DataFrame([
        {"game_id": "20260927_BOS@NYY", "game_date": "2026-09-27",
         "start_time_utc": "2026-09-27T23:00:00", "home_team": "NYY",
         "away_team": "BOS", "venue": "Test Park",
         "sp_id_home": 101, "sp_name_home": "Home Starter",
         "sp_id_away": np.nan, "sp_name_away": "TBD"},
        {"game_id": "20260925_CHC@BOS_2", "game_date": "2026-09-25",
         "start_time_utc": "2026-09-25T17:00:00", "home_team": "BOS",
         "away_team": "CHC", "venue": "Fenway Park",
         "sp_id_home": 303, "sp_name_home": "Decided Starter",
         "sp_id_away": 404, "sp_name_away": "Other Starter"},
    ])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"])

    rec = _LogRecorder()
    with patch.object(ingestion, "logger", rec):
        slate = build_upcoming_slate(history, date(2026, 9, 27),
                                     schedule_df=schedule)
    assert len(slate) == 1
    assert slate.iloc[0]["game_id"] == "20260927_BOS@NYY"
    assert "20260925_CHC@BOS_2" not in set(slate["game_id"])
    assert any("board-date invariant" in m for m in rec.messages)


def test_upcoming_slate_all_foreign_dates_ships_honest_empty_slate():
    """The off-day shape that produced the polluted board: when nothing is
    scheduled (or the schedule source only offers other dates), the slate
    is EMPTY -- never yesterday's finals re-dated as today."""
    history = _slate_history_row()
    schedule = pd.DataFrame([
        {"game_id": "20260925_CHC@BOS_2", "game_date": "2026-09-25",
         "start_time_utc": "2026-09-25T17:00:00", "home_team": "BOS",
         "away_team": "CHC", "venue": "Fenway Park",
         "sp_id_home": 303, "sp_name_home": "A", "sp_id_away": 404,
         "sp_name_away": "B"},
    ])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"])

    slate = build_upcoming_slate(history, date(2026, 9, 28),
                                 schedule_df=schedule)
    assert slate.empty


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\n{len(tests)} run-engine contract tests passed")
