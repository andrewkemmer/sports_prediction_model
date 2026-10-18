"""Score-means stability section (2026-10-18): unit contract.

The pipeline summary must answer \"did the new mus break derived ML?\"
on common decided games every run — the forensics the 2026-10-17 ensemble
promotion required by hand (-2.8 bps AUC / +6.9 bps LL, benign but
unwatched). Never fatal: missing prior, no overlap, or any error yields
a status dict, never an exception.
"""
import numpy as np
import pandas as pd

import master_pipeline as mp


def _frame(n=300, seed=4, mu_shift=0.0):
    rng = np.random.default_rng(seed)
    h = rng.normal(113., 5., n) + mu_shift
    a = rng.normal(110., 5., n) + mu_shift
    y = (rng.normal(h - a, 11.) > 0).astype(float)
    p = 1. / (1. + np.exp(-(h - a) / 8.))
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)], "mu_h": h, "mu_a": a,
        "p_home_win_derived": np.clip(p, 1e-6, 1 - 1e-6),
        "y_home_win": y})


def test_common_games_delta_detects_a_mu_move(tmp_path):
    cur = _frame(mu_shift=1.0)
    _frame(mu_shift=0.0).to_csv(tmp_path / "nba_run_engine_markets_20261016.csv",
                                index=False)
    got = mp._score_means_stability(cur, tmp_path, "20261017")
    assert got["status"] == "ok"
    assert got["n_common"] == 300
    assert got["mean_abs_dmu_home"] == 1.0
    assert got["mean_abs_dmu_away"] == 1.0
    assert "derived_auc_delta_bps" in got
    assert "derived_logloss_delta_bps" in got


def test_no_prior_delivery_is_a_status_not_an_error(tmp_path):
    got = mp._score_means_stability(_frame(), tmp_path, "20261017")
    assert got == {"status": "no_prior_delivery"}


def test_insufficient_overlap_is_reported(tmp_path):
    _frame(n=10).to_csv(tmp_path / "nba_run_engine_markets_20261016.csv",
                        index=False)
    got = mp._score_means_stability(_frame(n=10), tmp_path, "20261017")
    assert got["status"] == "insufficient_overlap"


def test_missing_columns_never_raise(tmp_path):
    got = mp._score_means_stability(pd.DataFrame({"a": [1]}), tmp_path,
                                    "20261017")
    assert got["status"] == "missing_columns"
    assert mp._score_means_stability(None, tmp_path, "20261017") == \
        {"status": "no_current_oof"}
