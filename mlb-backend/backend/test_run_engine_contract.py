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



def test_xgboost_params_pin_depth2_and_causal_priors():
    """Pin the 2026-09-30 PM adoption: the shipped member is the joint
    depth-2 block (subsample 0.5406, gamma 4.0, lr 0.055) — owner-directed
    replacement of the depth-1 stump, sealed-holdout confirmed at the
    member level (seal folds 70-77: 0.68089/0.5861 vs stump 0.68107/0.5842,
    3 seeds; full-walk 3-seed refit 0.68662/0.5562 vs 0.68755/0.5528)
    while the walk regime's causal rounds stay the honest-mechanics
    transfer FLOORED at the static budget (probe-median ~19-30 rounds,
    every shipped model >= the pinned 50 since 2026-10-05).
    A silent retune that flips max_depth, the shrinkage axes, or the round
    priors must fail here and re-run BOTH operating-point experiments plus
    the sealed window documented in config.XGBOOST_PARAMS provenance —
    never re-hobble a baseline to manufacture an adoption."""
    import config
    assert config.XGBOOST_PARAMS["max_depth"] == 2
    assert config.XGBOOST_PARAMS["subsample"] == 0.5406
    assert config.XGBOOST_PARAMS["gamma"] == 4.0
    assert config.XGBOOST_PARAMS["learning_rate"] == 0.055
    assert config.XGBOOST_PARAMS["min_child_weight"] == 12
    assert config.XGBOOST_PARAMS["colsample_bytree"] == 0.6382
    assert config.XGBOOST_FOLD0_ROUNDS == 50
    assert config.XGBOOST_REFIT_ROUNDS == 50
    assert config.XGBOOST_FOLD_ROUNDS == 2000
    assert config.XGBOOST_EARLY_STOP == 20


def test_causal_xgb_rounds_prior_median_rule():
    """2026-09-30 PIT remediation: a fold's SHIPPED XGBoost round count is
    the median of PRIOR folds' measured best iterations — never the fold's
    own val-window early-stop result."""
    from training import _causal_xgb_rounds
    # no measurements -> fold-0 static prior
    assert _causal_xgb_rounds([]) == 50
    # odd count -> middle element
    assert _causal_xgb_rounds([30, 50, 90]) == 50
    # even count -> midpoint of the two central elements
    assert _causal_xgb_rounds([30, 50, 70, 90]) == 60
    # degenerate entries (0/None-ish) are ignored, not allowed to skew
    assert _causal_xgb_rounds([0, 40, 60]) == 50
    # all-degenerate falls back to the refit prior
    assert _causal_xgb_rounds([0]) == 50
    # the measurement list must exist and be clearable (walk-start reset)
    import training
    assert hasattr(training, "_LAST_XGB_BEST_ROUNDS")
    training._LAST_XGB_BEST_ROUNDS.clear()


def test_causal_xgb_rounds_fold_semantics():
    """Fold k consumes measurements strictly BEFORE k (the [:-1] slice at
    the trainer's call site); a refit consumes all of them."""
    from training import _causal_xgb_rounds
    meas = [44, 46, 52, 58]
    # fold 0 ships the prior, fold 1 sees [44], fold 3 sees [44,46,52]...
    assert _causal_xgb_rounds(meas[:0]) == 50
    assert _causal_xgb_rounds(meas[:1]) == 44
    assert _causal_xgb_rounds(meas[:2]) == 45
    assert _causal_xgb_rounds(meas[:3]) == 46
    # the deployed refit ships at the all-measurement median
    assert _causal_xgb_rounds(meas) == 49


# The 2026-10-05 production walk's measured XGBoost probe best iterations
# (fold order, 83 folds). Under the incumbent UNFLOORED transfer these
# shipped folds at 4/6/9/11/13/23... rounds and the refit at the
# 30-round all-median — the underfit behind the member's 0.0% earned
# blend weight (model_version_history 2026-10-05).
_PROD_XGB_PROBE_BESTS_20261005 = [
    4, 9, 13, 34, 45, 36, 131, 29, 1, 2, 4, 74, 52, 61, 27, 26, 67, 58,
    13, 41, 106, 13, 24, 43, 74, 37, 30, 3, 76, 36, 59, 180, 14, 15, 100,
    40, 4, 31, 39, 3, 71, 3, 1, 11, 134, 1, 77, 16, 15, 80, 64, 47, 11, 3,
    70, 21, 4, 1, 38, 24, 19, 65, 3, 10, 78, 33, 1, 4, 2, 66, 102, 42, 60,
    36, 27, 14, 79, 17, 44, 23, 152, 23, 8,
]


def test_shipped_xgb_rounds_floor_at_static_budget():
    """2026-10-05 run-log remediation: SHIPPED round counts go through
    training._shipped_xgb_rounds — the causal transfer FLOORED at the
    static no-evidence budget. The transfer itself (_causal_xgb_rounds)
    stays pure and pinned above; this pins the floor rule: below-budget
    transfers lift, above-budget transfers pass through (a floor, never a
    cap), and degenerate priors still land on the static."""
    import config
    from training import _shipped_xgb_rounds
    # no / tiny measurements -> static budget
    assert _shipped_xgb_rounds([], config.XGBOOST_FOLD0_ROUNDS) == 50
    assert _shipped_xgb_rounds([4, 8, 13], config.XGBOOST_FOLD0_ROUNDS) == 50
    # the floor is not a cap: a transfer above budget ships as measured
    assert _shipped_xgb_rounds([90, 120], config.XGBOOST_FOLD0_ROUNDS) == 105
    assert _shipped_xgb_rounds([70], config.XGBOOST_REFIT_ROUNDS) == 70
    # degenerate measurements fall back to the static prior (floored anyway)
    assert _shipped_xgb_rounds([0], config.XGBOOST_REFIT_ROUNDS) == 50


def test_every_production_fold_ships_at_the_static_budget():
    """Production pin: with the 2026-10-05 walk's 83 real probe bests the
    incumbent unfloored rule shipped early folds at 4/6/9/11/13/23 rounds
    (fold 0's noisy probe best became fold 1's whole prior) and the refit
    at the 30-round all-median. Under the floor every fold AND the
    deployed refit train at exactly the static 50 — the exact operating
    point the remediation walk measured (member AUC 0.5569 -> 0.5638,
    13.65% earned weight)."""
    import config
    from training import _causal_xgb_rounds, _shipped_xgb_rounds
    bests = _PROD_XGB_PROBE_BESTS_20261005
    assert len(bests) == 83
    # what the incumbent rule would have shipped for the first folds
    assert [_causal_xgb_rounds(bests[:k]) for k in range(1, 7)] == \
        [4, 6, 9, 11, 13, 23]
    # every fold (fold 0 uses no priors) floors to the static budget
    assert all(
        _shipped_xgb_rounds(bests[:k], config.XGBOOST_FOLD0_ROUNDS) == 50
        for k in range(len(bests) + 1))
    # the deployed refit: all-measurement transfer is 30 -> floors to 50
    assert _causal_xgb_rounds(bests) == 30
    assert _shipped_xgb_rounds(bests, config.XGBOOST_REFIT_ROUNDS) == 50


def test_fold_path_ships_xgb_at_floored_rounds_end_to_end():
    """Wiring pin: train_moneyline_ensemble's fold path must BUILD the
    XGBoost member at _shipped_xgb_rounds. Seeded probe priors of 4/8
    (which the incumbent rule ships at 4-6 rounds) must still produce a
    50-round fold model — a regression that drops the floor at the call
    site goes red here."""
    import config
    import training
    rng = np.random.default_rng(7)

    def _frame(n: int) -> pd.DataFrame:
        idx = np.arange(n)
        return pd.DataFrame({
            "game_date": pd.Timestamp("2024-03-01")
            + pd.to_timedelta(idx % 90, unit="D"),
            "home_win": (idx % 2).astype(float),
            "home_team": np.where(idx % 2 == 0, "NYY", "BOS"),
            "away_team": np.where(idx % 2 == 0, "BOS", "NYY"),
            "elo_diff": rng.normal(0.0, 100.0, n),
        })

    saved = list(training._LAST_XGB_BEST_ROUNDS)
    training._LAST_XGB_BEST_ROUNDS.clear()
    training._LAST_XGB_BEST_ROUNDS.extend([4, 8])  # incumbent -> ship 4/6
    try:
        models, _metrics = training.train_moneyline_ensemble(_frame(120),
                                                             _frame(60))
        assert {"xgboost", "lightgbm", "elasticnet"} <= set(models)
        assert models["xgboost"].get_params()["n_estimators"] == \
            config.XGBOOST_FOLD0_ROUNDS
    finally:
        training._LAST_XGB_BEST_ROUNDS.clear()
        training._LAST_XGB_BEST_ROUNDS.extend(saved)


def test_upcoming_slate_ships_total_runs_for_carried_finals():
    """2026-10-03 defect: every Final row in todays_games_*.csv shipped
    an empty total_runs because the row builder hard-coded NaN even when
    ESPN scores were carried. Finals must ship the derived total; an
    undecided row must still ship NaN (honest preview, never a guess)."""
    history = _slate_history_row()
    schedule = pd.DataFrame([
        {"game_id": "20261003_CHW@CLE", "game_date": "2026-10-03",
         "start_time_utc": "2026-10-03T23:10:00", "home_team": "CLE",
         "away_team": "CHW", "venue": "Test Park",
         "game_state": "post", "game_status_detail": "Final",
         "home_score": 3, "away_score": 0, "home_win": 0.0,
         "sp_id_home": 101, "sp_name_home": "Home Starter",
         "sp_id_away": 202, "sp_name_away": "Away Starter"},
        {"game_id": "20261003_NYY@TB", "game_date": "2026-10-03",
         "start_time_utc": "2026-10-03T22:10:00", "home_team": "TB",
         "away_team": "NYY", "venue": "Test Park 2",
         "game_state": "pre", "game_status_detail": "Scheduled",
         "sp_id_home": 303, "sp_name_home": "A",
         "sp_id_away": 404, "sp_name_away": "B"},
    ])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    schedule["start_time_utc"] = pd.to_datetime(schedule["start_time_utc"])

    slate = build_upcoming_slate(history, date(2026, 10, 3),
                                 schedule_df=schedule)
    assert len(slate) == 2
    final = slate.loc[slate["game_id"] == "20261003_CHW@CLE"].iloc[0]
    preview = slate.loc[slate["game_id"] == "20261003_NYY@TB"].iloc[0]
    assert final["total_runs"] == 3.0
    assert final["home_score"] == 3 and final["away_score"] == 0
    assert np.isnan(preview["total_runs"])
    assert preview["home_score"] is None or np.isnan(preview["home_score"])


def test_apply_official_results_matches_pkless_slate_by_date_and_teams():
    """Slate rows carry no StatsAPI game_pk; the overlay's documented
    fallback key (game_date, canonical home/away) must match a hydrated
    results frame — including the board's CHW vs StatsAPI CWS alias — and
    stamp authoritative scores, total_runs and game_state="post"."""
    from results import apply_official_results

    slate = pd.DataFrame([{
        "game_id": "20261003_CHW@CLE",
        "game_date": pd.Timestamp("2026-10-03"),
        "home_team": "CLE", "away_team": "CHW",
        "game_state": "final",
        "home_win": 0.0, "home_score": 3, "away_score": 0,
        "total_runs": np.nan,
    }])
    res = pd.DataFrame([{
        "game_pk": None, "game_date": "2026-10-03",
        "home_score": 3.0, "away_score": 0.0, "home_win": 0.0,
        "is_final": True,
        "home_team": "CLE", "away_team": "CWS",
    }])

    out = apply_official_results(slate, res)
    assert out.loc[0, "home_score"] == 3.0
    assert out.loc[0, "away_score"] == 0.0
    assert out.loc[0, "total_runs"] == 3.0
    assert out.loc[0, "game_state"] == "post"


def test_fetch_mlb_results_hydrates_and_ships_team_columns():
    """The schedule endpoint returns team objects WITHOUT abbreviations
    unless hydrated — which made the slate fallback key unbuildable.
    fetch_mlb_results must request team(abbreviation) and canonicalize
    the codes it gets (CHW→CWS via _canon_team)."""
    import results as results_mod
    from results import fetch_mlb_results

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"dates": [{"date": "2026-10-03", "games": [{
                "gamePk": 777,
                "status": {"abstractGameState": "Final"},
                "teams": {
                    "home": {"score": 3, "team": {"abbreviation": "CLE"}},
                    "away": {"score": 0, "team": {"abbreviation": "CHW"}},
                },
            }]}]}

    captured: dict = {}

    def _get(url, params=None, timeout=None):
        captured["params"] = params
        return _Resp()

    with patch.object(results_mod.requests, "get", side_effect=_get):
        df = fetch_mlb_results(date(2026, 10, 3), date(2026, 10, 3))

    assert captured["params"]["hydrate"] == "team(abbreviation)"
    assert "home_team" in df.columns and "away_team" in df.columns
    assert df.loc[0, "home_team"] == "CLE"
    assert df.loc[0, "away_team"] == "CWS"  # alias canonicalized
    assert df.loc[0, "home_score"] == 3.0 and df.loc[0, "is_final"]


def test_official_results_never_alias_doubleheader_legs():
    from results import apply_official_results
    games = pd.DataFrame([
        {"game_pk": np.nan, "game_date": "2026-09-01", "home_team": "NYY",
         "away_team": "BOS", "home_win": np.nan},
        {"game_pk": 999, "game_date": "2026-09-01", "home_team": "NYY",
         "away_team": "BOS", "home_win": np.nan}])
    results = pd.DataFrame([
        {"game_pk": 1, "game_date": "2026-09-01", "home_team": "NYY",
         "away_team": "BOS", "home_score": 2, "away_score": 1,
         "home_win": 1.0, "is_final": True},
        {"game_pk": 2, "game_date": "2026-09-01", "home_team": "NYY",
         "away_team": "BOS", "home_score": 1, "away_score": 2,
         "home_win": 0.0, "is_final": True}])
    out = apply_official_results(games, results)
    assert out.home_win.isna().all()
    keyed = games.copy()
    keyed["game_pk"] = [1, 2]
    out = apply_official_results(keyed, results)
    assert out.home_win.tolist() == [1.0, 0.0]


def test_slate_carries_bullpen_three_game_family_and_unknown_opener_rate():
    from features import add_diff_features
    history = pd.DataFrame([{
        "game_date": pd.Timestamp("2025-10-01"), "home_team": "NYY",
        "away_team": "BOS", "home_win": 1.0, "home_score": 3, "away_score": 1,
        "bullpen_whip_3g_home": 1.1, "bullpen_whip_3g_away": 1.4}])
    schedule = pd.DataFrame([{
        "game_date": pd.Timestamp("2026-03-25"), "home_team": "NYY",
        "away_team": "BOS", "start_time_utc": pd.Timestamp("2026-03-25T19:00:00")}])
    slate = build_upcoming_slate(history, date(2026, 3, 25), schedule_df=schedule)
    assert slate.home_win_pct.isna().all() and slate.away_win_pct.isna().all()
    assert slate.home_record.tolist() == ["0-0"]
    out = add_diff_features(slate)
    np.testing.assert_allclose(out.bullpen_whip_3g_diff, -0.3)


def test_slate_handedness_uses_opposing_probable_starter():
    from features import add_diff_features
    history = pd.DataFrame([{
        "game_date": pd.Timestamp("2026-09-01"), "home_team": "NYY",
        "away_team": "BOS", "home_win": 1.0, "home_score": 3, "away_score": 1,
        "home_starter_id": 101, "away_starter_id": 202,
        "home_starter_hand": "R", "away_starter_hand": "L",
        "lineup_ops_vs_l_home": 0.8, "lineup_ops_vs_r_home": 0.7,
        "lineup_ops_vs_l_away": 0.6, "lineup_ops_vs_r_away": 0.5}])
    schedule = pd.DataFrame([{
        "game_date": pd.Timestamp("2026-09-02"), "home_team": "NYY",
        "away_team": "BOS", "start_time_utc": pd.Timestamp("2026-09-02T19:00:00"),
        "sp_id_home": 101, "sp_id_away": 202}])
    slate = build_upcoming_slate(history, date(2026, 9, 2), schedule_df=schedule)
    np.testing.assert_allclose(add_diff_features(slate).lineup_handedness_matchup_advantage, 0.3)
    # Fresh probable-hand evidence takes priority over older observed starts.
    schedule["sp_hand_home"], schedule["sp_hand_away"] = "L", "R"
    slate = build_upcoming_slate(history, date(2026, 9, 2), schedule_df=schedule)
    np.testing.assert_allclose(add_diff_features(slate).lineup_handedness_matchup_advantage, 0.1)


def test_prior_filter_handles_nullable_strings_and_utc_boundaries():
    frame = pd.DataFrame({"start_time_utc": pd.Series(
        ["2026-09-01T18:59:59Z", "2026-09-01T19:00:00Z", None], dtype="string")})
    out = ingestion.filter_prior(frame, pd.Timestamp("2026-09-01T19:00:00Z"))
    assert out.index.tolist() == [0]


def test_probable_starter_categories_match_historical_identity():
    import training
    base = pd.DataFrame({"home_team": ["NYY"], "away_team": ["BOS"],
                         "home_starter_id": [101], "away_starter_id": [202]})
    historical = training._add_team_ids(base)
    slate = training._add_team_ids(base.rename(columns={
        "home_starter_id": "sp_id_home", "away_starter_id": "sp_id_away"}))
    for c in ("home_starter_cat_id", "away_starter_cat_id"):
        np.testing.assert_array_equal(historical[c], slate[c])
        assert int(slate[c].iloc[0]) != training.UNK_STARTER_ID


def test_legacy_pitch_cache_requires_observed_schema_rebuild(monkeypatch, tmp_path):
    import ingestion as source
    cache = tmp_path / "pitches.parquet"
    pd.DataFrame({"game_pk": [1], "game_date": ["2026-09-01"],
                  "at_bat_number": [1], "pitch_number": [1]}).to_parquet(cache, index=False)
    fresh = pd.DataFrame({"game_pk": [2], "game_date": ["2026-09-01"],
                          "at_bat_number": [1], "pitch_number": [1], "game_type": ["R"],
                          "post_home_score": [1], "post_away_score": [0], "launch_speed_angle": [6]})
    seen = []
    def chunks(start, end, chunk_days, pause):
        seen.append((start, end))
        return [fresh]
    monkeypatch.setattr(source, "_chunked_statcast", chunks)
    source.pull_statcast("2026-08-01", "2026-09-01", out_path=cache)
    assert seen == [(date(2026, 8, 1), date(2026, 9, 1))]
    assert pd.read_parquet(cache).game_pk.tolist() == [2]


def test_statcast_source_sql_uses_post_scores_and_observed_barrels(tmp_path):
    """Execute actual production stage SQL against discriminating events."""
    import ast
    import features
    source = (BACKEND / "features.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "_build_game_level")
    rows = pd.DataFrame([
        {"game_pk": 1, "game_date": "2026-09-01", "game_type": "R", "home_team": "NYY",
         "away_team": "BOS", "inning": 1, "inning_topbot": "Top", "at_bat_number": 1,
         "pitch_number": 1, "pitcher": 101, "batter": 201, "events": "home_run",
         "description": "hit_into_play", "home_score": 0, "away_score": 0,
         "post_home_score": 0, "post_away_score": 1, "launch_speed": 110,
         "launch_angle": 35, "launch_speed_angle": 6},
        {"game_pk": 1, "game_date": "2026-09-01", "game_type": "R", "home_team": "NYY",
         "away_team": "BOS", "inning": 1, "inning_topbot": "Bot", "at_bat_number": 2,
         "pitch_number": 1, "pitcher": 102, "batter": 202, "events": "home_run",
         "description": "hit_into_play", "home_score": 0, "away_score": 1,
         "post_home_score": 2, "post_away_score": 1, "launch_speed": 100,
         "launch_angle": 28, "launch_speed_angle": np.nan}])
    unknown_baseline = rows.iloc[[0]].copy()
    unknown_baseline["game_pk"], unknown_baseline["pitcher"] = 2, 103
    unknown_baseline[["home_score", "away_score"]] = np.nan
    rows = pd.concat([rows, unknown_baseline], ignore_index=True)
    path = tmp_path / "pitches.parquet"
    rows.to_parquet(path, index=False)
    con = features._connect(path)
    env = {"PA_END_EVENTS": features.PA_END_EVENTS, "_pa_events": features.PA_END_EVENTS,
           "_outs_fix_whens": "", "_batters_ok": False,
           "_batter_excl": features._batter_excl}
    try:
        for node in fn.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "execute":
                continue
            arg_source = ast.get_source_segment(source, call.args[0]) or ""
            if any(f"CREATE TABLE {table} AS" in arg_source for table in
                   ("game_winners", "pa_boundary", "team_contact_raw")):
                sql = eval(compile(ast.Expression(call.args[0]), "production_sql", "eval"), env)
                con.execute(sql)
        win = con.execute("SELECT home_score, away_score, home_win, total_runs FROM game_winners WHERE game_pk = 1").fetchone()
        assert win == (2, 1, 1.0, 3)
        pa = con.execute("SELECT pitcher, runs_on_pa, barrel_flag FROM pa_boundary ORDER BY pitcher").fetchall()
        assert pa == [(101, 1, 1.0), (102, 2, None), (103, None, 1.0)]
        contact = con.execute("SELECT batting_team, barrel_rate FROM team_contact_raw WHERE game_pk = 1 ORDER BY batting_team").fetchall()
        assert contact == [("BOS", 1.0), ("NYY", None)]
    finally:
        con.close()


def test_ingestion_retains_event_semantics_and_unique_aliases():
    from ingestion import _normalize_columns, UNUSED_COLS
    raw = pd.DataFrame({"game_type": ["R"], "game_date": ["2026-09-01"],
                        "launch_speed": [100.0], "exit_velocity": [99.0],
                        "launch_speed_angle": [6], "post_home_score": [4],
                        "post_away_score": [3]})
    out = _normalize_columns(raw)
    assert out.columns.is_unique and out.launch_speed.iloc[0] == 100.0
    assert not {"launch_speed_angle", "post_home_score", "post_away_score"} & set(UNUSED_COLS)
    # Multiple aliases when the canonical is absent must still yield one
    # column, with the first observation taking precedence and holes filled.
    aliases = _normalize_columns(pd.DataFrame({
        "exit_velocity": [100.0, np.nan], "exit_velo": [99.0, 101.0]}))
    assert aliases.columns.is_unique
    assert aliases.launch_speed.tolist() == [100.0, 101.0]


def test_master_full_universe_coverage_gate():
    """Run the exact master gate without executing notebook setup/push."""
    import ast
    import pytest
    import training
    tree = ast.parse((BACKEND / "master_pipeline.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "run_daily_pipeline")
    nodes = list(ast.walk(fn))
    assignments = [n for n in nodes if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in {"_absent", "_starved"}
                           for t in n.targets)]
    gate = next(n for n in nodes if isinstance(n, ast.If)
                and isinstance(n.test, ast.BoolOp)
                and {x.id for x in n.test.values if isinstance(x, ast.Name)}
                == {"_absent", "_starved"})
    code = compile(ast.Module(body=assignments + [gate], type_ignores=[]), "master_gate", "exec")
    cols = training.MONEYLINE_FEATURE_COLS
    complete = pd.DataFrame({c: [np.nan, 1.0] for c in cols})
    def run(frame):
        exec(code, {"pd": pd, "MONEYLINE_FEATURE_COLS": cols, "_decided_snapshot": frame})
    run(complete)  # Row-level warm NULLs are allowed; universe starvation is not.
    with pytest.raises(ValueError, match="missing="):
        run(complete.drop(columns=[cols[-1]]))
    starved = complete.copy()
    starved[cols[-1]] = np.nan
    with pytest.raises(ValueError, match="entirely_unobserved="):
        run(starved)


def test_source_schema_guard_rejects_legacy_engineering(tmp_path):
    import pytest
    import features
    path = tmp_path / "legacy.parquet"
    pd.DataFrame({"game_pk": [1]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Statcast source schema missing observed fields"):
        features.build_features(path, tmp_path / "features")


def test_load_features_preserves_false_start_provenance(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame({"game_date": ["2026-09-01"], "home_team": ["NYY"],
                  "away_team": ["BOS"], "home_win": [1.0],
                  "start_time_utc": ["2026-09-01T19:00:00Z"],
                  "start_time_observed": [False]}).to_csv(path, index=False)
    loaded = ingestion.load_game_features(path)
    assert not loaded.start_time_observed.any()
    assert str(loaded.start_time_utc.dtype) == "datetime64[us, UTC]" or \
        str(loaded.start_time_utc.dtype) == "datetime64[ns, UTC]"


if __name__ == "__main__":
    # Run through pytest so production contract cases receive isolated
    # tmp_path/monkeypatch fixtures in the documented direct CLI too.
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
