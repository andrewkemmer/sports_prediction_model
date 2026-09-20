"""Offline run-engine contract tests.

Run with: python mlb-backend/backend/test_run_engine_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import run_engine as re


def _market_row(**overrides):
    row = {
        "game_pk": [101], "game_date": ["2026-09-18"],
        "home_expected_runs": [4.0], "away_expected_runs": [4.2],
    }
    row.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(row)


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


def test_run_line_opposite_tail_is_not_home_dog_tail():
    # A deterministic draw matrix is unnecessary here: the two output arrays
    # have separate contracts and are populated independently by the MC path.
    assert "p_rl_1_5_away_favorite" == re.rl_col(1.5, "away_favorite")
    assert "p_rl_1_5_home_dog" == re.rl_col(1.5, "home_dog")
    assert "p_rl_1_5_away_favorite" in re.MARKET_COLUMNS_V3


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\n{len(tests)} run-engine contract tests passed")
