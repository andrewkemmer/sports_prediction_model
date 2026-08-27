from __future__ import annotations

import numpy as np
import pandas as pd

from run_calibration_ablation import (
    _apply_candidate,
    _fit_candidate,
    candidate_names,
    metrics,
    replay,
)


def test_candidate_size_gates():
    assert candidate_names(299) == ["identity"]
    assert candidate_names(300) == ["identity", "platt"]
    assert candidate_names(999) == ["identity", "platt"]
    assert candidate_names(1000) == ["identity", "platt", "isotonic"]


def test_isotonic_predict_is_clipped_to_training_range():
    y = np.array([0, 0, 1, 1, 1, 0], dtype=float)
    p = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    fitted = _fit_candidate("isotonic", y, p)
    out = _apply_candidate("isotonic", np.array([0.0, 1.0]), fitted)
    assert np.isfinite(out).all()
    assert ((out >= 0) & (out <= 1)).all()


def test_replay_schema_and_gate_are_explicit():
    dates = pd.date_range("2026-01-01", periods=340, freq="D")
    df = pd.DataFrame({
        "game_date": dates,
        "home_win": (np.arange(len(dates)) % 2).astype(float),
        "home_win_prob_model": np.where(np.arange(len(dates)) % 2, 0.6, 0.4),
    })
    df.loc[len(df)] = [pd.Timestamp("2026-08-05"), 1.0, 0.7]
    df.loc[len(df)] = [pd.Timestamp("2026-08-06"), 0.0, 0.3]
    result = replay(df)
    assert result["schema"] == "calibration-ablation/v1"
    assert result["holdout_n"] == 23
    assert set(result["variants"]) == {"identity", "unconditional_platt", "conditional", "isotonic"}
    assert result["gate"]["verdict"] in {"ADOPT", "DON'T ADOPT"}


def test_metrics_is_finite():
    result = metrics(np.array([0, 1, 1, 0]), np.array([0.2, 0.7, 0.8, 0.3]))
    assert all(np.isfinite(v) for v in result.values())
