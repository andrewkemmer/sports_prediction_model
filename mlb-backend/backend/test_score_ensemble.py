"""Tests for the distribution score ensemble (elasticnet + xgboost + lightgbm).

NBA structural port onto the strict-active-moneyline side view. Pins the
Poisson deviance math (including the y=0 edge), the log-space simplex
blender (positivity, NaN skipping, one-hot fallback), and the walk's
point-in-time property — fold k blends with weights earned on folds
strictly before k, verified by recomputation, not by trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from config import SCORE_ENSEMBLE_MEMBERS, SCORE_ENSEMBLE_WEIGHTS
import distributions as dist_mod
from training import (active_moneyline_feature_cols,
                      canonical_walk_forward_splits, RAW_PER_SIDE_COLS)

H_H = dist_mod._MLB_MU_H
H_A = dist_mod._MLB_MU_A


def _frame(n=140, seed=11):
    rng = np.random.default_rng(seed)
    cols = active_moneyline_feature_cols()
    data = {c: rng.normal(0, 1, n) for c in cols}
    data.update({
        "game_pk": np.arange(1000, 1000 + n),
        "game_date": pd.date_range("2024-04-01", periods=n, freq="D"),
        "home_team": ["NYY"] * n,
        "away_team": ["BOS"] * n,
        "game_type": ["R"] * n,
        "home_win": (rng.random(n) > 0.5).astype(int),
        "home_score": rng.poisson(5, n).astype(float),
        "away_score": rng.poisson(4, n).astype(float),
    })
    return pd.DataFrame(data)


def test_poisson_deviance_matches_hand_computation():
    y = np.array([5.0, 7.0, 0.0])
    mu = np.array([5.0, 5.0, 3.0])
    expected = 2.0 * np.mean([
        0.0,
        7.0 * np.log(7.0 / 5.0) - 2.0,
        3.0,
    ])
    assert dist_mod.poisson_deviance(y, mu) == pytest.approx(expected)
    assert dist_mod.poisson_deviance([5.0], [5.0]) == pytest.approx(0.0)
    assert np.isnan(dist_mod.poisson_deviance([], []))


def test_log_blend_is_geometric_positive_and_nan_skipping():
    P = np.array([[4.0, 9.0], [np.nan, 5.0], [np.nan, np.nan]])
    w = np.array([0.5, 0.5])
    out = dist_mod._log_blend_mus(P, w)
    assert out[0] == pytest.approx(np.sqrt(4.0 * 9.0))
    assert out[1] == pytest.approx(5.0)
    assert np.isnan(out[2])
    assert (out[np.isfinite(out)] > 0).all()
    solo = dist_mod._log_blend_mus(
        np.array([[4.0, 9.0]]), np.array([0.0, 1.0]))
    assert solo[0] == pytest.approx(9.0)


def test_score_weights_pay_the_near_perfect_member():
    rng = np.random.default_rng(3)
    y = rng.poisson(5, 300).astype(float)
    members = {
        "xgboost": y.copy(),
        "lightgbm": np.full_like(y, y.mean()),
        "elasticnet": np.full_like(y, y.mean()),
    }
    w = dist_mod.compute_score_weights(members, y)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert min(w.values()) >= 0.0
    assert w["xgboost"] == pytest.approx(1.0)
    assert dist_mod.compute_score_weights({"lightgbm": [5.0, 4.0]},
                                           [5.0, 4.0]) == {"lightgbm": 1.0}
    assert dist_mod.compute_score_weights({}, []) == {}


def test_ensemble_members_fit_and_predict_finite_positive():
    df = _frame()
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    for side in ("home", "away"):
        for name in SCORE_ENSEMBLE_MEMBERS:
            p = ens.predict_member(df.head(10), name, side)
            assert np.isfinite(p).all() and (p > 0).all(), (name, side)
    mu_h, mu_a = ens.predict(df.head(10))
    assert np.isfinite(mu_h).all() and (mu_h > 0).all()
    assert np.isfinite(mu_a).all() and (mu_a > 0).all()


def test_ensemble_walk_is_point_in_time_by_recomputation():
    df = _frame(n=210)
    result = dist_mod.walk_forward_oof_ensemble(
        df, min_val_games=5, min_train_days=0, progress_every=0)
    oof = result["oof"]
    assert {H_H, H_A}.issubset(oof.columns)
    for name in SCORE_ENSEMBLE_MEMBERS:
        assert f"{H_H}_{name}" in oof.columns
        assert f"{H_A}_{name}" in oof.columns
        assert f"weight_{name}" in oof.columns
    folds = sorted(oof.fold_idx.unique())
    assert len(folds) >= 3
    first = oof[oof.fold_idx == folds[0]].iloc[0]
    for name in SCORE_ENSEMBLE_MEMBERS:
        assert first[f"weight_{name}"] == pytest.approx(
            SCORE_ENSEMBLE_WEIGHTS[name])
    for fold in folds[1:]:
        prior = oof[(oof.fold_idx < fold) & oof.grades_blend]
        pooled = {}
        for name in SCORE_ENSEMBLE_MEMBERS:
            pooled[name] = (prior[f"{H_H}_{name}"].tolist()
                            + prior[f"{H_A}_{name}"].tolist())
        y = prior.home_score.tolist() + prior.away_score.tolist()
        if len(y) < 2:
            continue
        expected = dist_mod.compute_score_weights(pooled, np.asarray(y))
        if not expected:
            continue
        got = oof[oof.fold_idx == fold].iloc[0]
        for name in SCORE_ENSEMBLE_MEMBERS:
            assert got[f"weight_{name}"] == pytest.approx(
                expected.get(name, 0.0), abs=1e-9), (fold, name)


def test_en_diffs_only_flag_off_is_byte_identical_arm_b(monkeypatch):
    """Default-off: EN numeric array == full side-frame transform."""
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_DIFFS_ONLY", False)
    df = _frame()
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    assert ens.en_columns == {} and ens.en_scalers == {}
    for side in ("home", "away"):
        tree_frame, num = ens._matrices(df, side, for_elasticnet=True)
        assert list(ens.columns[side]) == list(tree_frame.columns)[:len(
            ens.columns[side])]
        assert num.shape[1] == len(ens.columns[side])


def test_en_diffs_only_adds_nothing_removes_raw_levels(monkeypatch):
    """Arm C view: raw per-side levels out of the EN input, trees untouched."""
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_DIFFS_ONLY", True)
    df = _frame()
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    for side in ("home", "away"):
        # Raw levels are side-routed: the home frame carries only the
        # home halves (+ shared env diffs), never the opponent's.
        side_raw = {c for c in RAW_PER_SIDE_COLS if c in ens.columns[side]}
        assert side_raw, f"{side} frame must carry same-side raw levels"
        assert not (set(ens.en_columns[side]) & set(RAW_PER_SIDE_COLS))
        assert side_raw <= set(ens.columns[side])  # trees keep them
        tree_frame, num = ens._matrices(df, side, for_elasticnet=True)
        assert num.shape[1] == len(ens.en_columns[side])
        p = ens.predict_member(df.head(10), "elasticnet", side)
        assert np.isfinite(p).all() and (p > 0).all(), side
    mu_h, mu_a = ens.predict(df.head(10))
    assert np.isfinite(mu_h).all() and (mu_h > 0).all()
    assert np.isfinite(mu_a).all() and (mu_a > 0).all()


def test_en_predict_reuses_fit_time_view_across_flag_toggle(monkeypatch):
    """Fit-time view freeze: toggling the flag between fit and predict
    must not change geometry (probed 2026-10-17: 74-wide EN model fed a
    41-wide frame raises in sklearn)."""
    df = _frame()
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_DIFFS_ONLY", False)
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    ref = ens.predict_member(df.head(10), "elasticnet", "home")
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_DIFFS_ONLY", True)
    got = ens.predict_member(df.head(10), "elasticnet", "home")
    np.testing.assert_allclose(got, ref, rtol=0)
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_DIFFS_ONLY", True)
    ens2 = dist_mod.ScoreEnsembleRegressor().fit(df)
    ref2 = ens2.predict_member(df.head(10), "elasticnet", "home")
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_DIFFS_ONLY", False)
    got2 = ens2.predict_member(df.head(10), "elasticnet", "home")
    np.testing.assert_allclose(got2, ref2, rtol=0)


def test_make_score_member_rejects_unknown_names():
    with pytest.raises(KeyError):
        dist_mod._make_score_member("nope")


def test_ensemble_mus_price_through_the_identical_variance_machinery():
    df = _frame(n=60)
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    mu_h, mu_a = ens.predict(df)
    frame = df.head(20).copy()
    frame[H_H] = mu_h[:20]
    frame[H_A] = mu_a[:20]
    priced = dist_mod._distribution_price(frame, {}, n_draws=200)
    for col in ("mu_margin", "mu_total"):
        assert col in priced.columns, col
        assert np.isfinite(priced[col].to_numpy(float)).all(), col


def test_walk_shares_geometry_with_canonical_splits():
    df = _frame(n=210)
    _, splits = canonical_walk_forward_splits(
        df, min_val_games=5, min_train_days=0)
    result = dist_mod.walk_forward_oof_ensemble(
        df, min_val_games=5, min_train_days=0, progress_every=0)
    assert len(result["oof"]) == sum(len(s["val_games"]) for s in splits)
