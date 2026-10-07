"""Offline diagnostic-harness tests; production defects are recorded, not fixed here."""
import numpy as np
import pandas as pd
import pytest

try:
    from backend.audit_model_quality import (
        metrics, paired_block_interval, reconstruct_history, semantic_probes)
except ImportError:  # direct script/import fallback
    from audit_model_quality import (
        metrics, paired_block_interval, reconstruct_history, semantic_probes)


def test_binary_metric_contract_and_reference_values():
    result = metrics([0, 1, 0, 1], [.2, .8, .3, .7])
    assert result["auc"] == 1
    assert result["brier"] == pytest.approx(.065)
    assert result["logloss"] == pytest.approx(-np.mean(np.log([.8, .8, .7, .7])))
    with pytest.raises(ValueError, match="binary targets"):
        metrics([.2, .8], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        metrics([0, 1], [np.nan, .7])
    with pytest.raises(ValueError, match="shapes"):
        metrics([0], [.2, .7])


def test_single_class_slice_retains_logloss_not_auc():
    result = metrics([1, 1], [.6, .8])
    assert result["auc"] is None
    assert result["logloss"] > 0


def test_reconstruction_and_duplicate_gate():
    rows = []
    for day in pd.date_range("2025-01-01", periods=8):
        for i in range(6):
            rows.append({"game_id": f"202402{len(rows) + 1:04d}", "game_date": day,
                         "home_win": i % 2, "home_win_prob_model": .5})
    history = pd.DataFrame(rows)
    result = reconstruct_history(history.sample(frac=1, random_state=42))
    assert result.groupby("fold_id").size().tolist() == [42, 6]
    assert result.grading.sum() == 42
    assert result.provisional.sum() == 6
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct_history(pd.concat([history, history.iloc[:1]]))


def test_block_bootstrap_identity_and_determinism():
    y = np.tile([0, 1], 30)
    p = np.where(y, .6, .4)
    dates = pd.date_range("2025-01-01", periods=len(y))
    a = paired_block_interval(y, p, p, dates, draws=20)
    b = paired_block_interval(y, p, p, dates, draws=20)
    assert a == b
    assert a["delta_logloss"] == pytest.approx(0)
    assert a["delta_logloss_ci95"] == pytest.approx([0, 0])
    assert a["delta_auc_ci95"] == pytest.approx([0, 0])


def test_semantic_probes_are_inspectable():
    result = semantic_probes()
    assert 0 < result["equal_elo_implied_home_expectation"] < 1
    assert result["positive_home_advantage_expected"] > .5
    assert result["goalie_trade_selected_name"] in {"departed", "current"}
    assert result["market_metric_correct_arguments"]["logloss"] == pytest.approx(.2899)
    assert result["market_metric_current_caller_arguments"]["logloss"] > 3
