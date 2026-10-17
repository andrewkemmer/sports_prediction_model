"""Tests for the distribution score ensemble (elasticnet + xgboost + lightgbm).

NBA structural port. Pins the regression twins of the binary-moneyline
contract: Poisson deviance math (including the y=0 edge), the log-space
simplex blender (positivity, NaN skipping, one-hot fallback), and the
walk's point-in-time property — fold k blends with weights earned on
folds strictly before k, verified by recomputation, not by trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from backend import config
    from backend import distributions as dist_mod
    from backend import folds as folds_mod
except ImportError:
    import config
    import distributions as dist_mod
    import folds as folds_mod


def _frame(n=240, seed=11):
    rng = np.random.default_rng(seed)
    cols = config.active_moneyline_feature_cols()
    data = {c: rng.normal(0, 1, n) for c in cols}
    teams = ["KC", "BUF", "PHI", "DAL"]
    data.update({
        "game_id": [f"G{i}" for i in range(n)],
        "gameday": pd.date_range("2024-09-05", periods=n, freq="D"),
        "season": 2024,
        "game_type": ["Regular Season"] * n,
        "home_team": [teams[i % 4] for i in range(n)],
        "away_team": [teams[(i + 1) % 4] for i in range(n)],
        "home_score": rng.poisson(24, n).astype(float),
        "away_score": rng.poisson(21, n).astype(float),
    })
    return pd.DataFrame(data)


def test_poisson_deviance_matches_hand_computation():
    y = np.array([20.0, 24.0, 0.0])
    mu = np.array([20.0, 20.0, 10.0])
    expected = 2.0 * np.mean([
        0.0,
        24.0 * np.log(24.0 / 20.0) - 4.0,
        10.0,
    ])
    assert dist_mod.poisson_deviance(y, mu) == pytest.approx(expected)
    assert dist_mod.poisson_deviance([20.0], [20.0]) == pytest.approx(0.0)
    assert np.isnan(dist_mod.poisson_deviance([], []))


def test_log_blend_is_geometric_positive_and_nan_skipping():
    P = np.array([[21.0, 28.0], [np.nan, 24.0], [np.nan, np.nan]])
    w = np.array([0.5, 0.5])
    out = dist_mod._log_blend_mus(P, w)
    assert out[0] == pytest.approx(np.sqrt(21.0 * 28.0))
    assert out[1] == pytest.approx(24.0)
    assert np.isnan(out[2])
    assert (out[np.isfinite(out)] > 0).all()
    solo = dist_mod._log_blend_mus(
        np.array([[21.0, 28.0]]), np.array([0.0, 1.0]))
    assert solo[0] == pytest.approx(28.0)


def test_score_weights_pay_the_near_perfect_member():
    rng = np.random.default_rng(3)
    y = rng.poisson(24, 300).astype(float)
    members = {
        "xgboost": y.copy(),
        "lightgbm": np.full_like(y, y.mean()),
        "elasticnet": np.full_like(y, y.mean()),
    }
    w = dist_mod.compute_score_weights(members, y)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert min(w.values()) >= 0.0
    assert w["xgboost"] == pytest.approx(1.0)
    assert dist_mod.compute_score_weights({"lightgbm": [24.0, 21.0]},
                                           [24.0, 21.0]) == {"lightgbm": 1.0}
    assert dist_mod.compute_score_weights({}, []) == {}


def test_ensemble_members_fit_and_predict_finite_positive():
    df = _frame()
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    for side in ("home_score", "away_score"):
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            p = ens.predict_member(df.head(10), name, side)
            assert np.isfinite(p).all() and (p > 0).all(), (name, side)
    mu_h, mu_a = ens.predict(df.head(10))
    assert np.isfinite(mu_h).all() and (mu_h > 0).all()
    assert np.isfinite(mu_a).all() and (mu_a > 0).all()


def test_ensemble_walk_is_point_in_time_by_recomputation():
    df = _frame(n=300)
    folds = folds_mod.make_folds(df, "gameday")
    assert len(folds) >= 3
    result = dist_mod.walk_forward_oof_ensemble(
        df, fold_list=folds, progress_every=0)
    oof = result["oof"]
    assert {"mu_h", "mu_a"}.issubset(oof.columns)
    for name in config.SCORE_ENSEMBLE_MEMBERS:
        assert f"mu_h_{name}" in oof.columns and f"mu_a_{name}" in oof.columns
        assert f"weight_{name}" in oof.columns
    first = oof[oof.fold_id == folds[0].fold_id].iloc[0]
    for name in config.SCORE_ENSEMBLE_MEMBERS:
        assert first[f"weight_{name}"] == pytest.approx(
            config.SCORE_ENSEMBLE_WEIGHTS[name])
    for fold in folds[1:]:
        prior = oof[(oof.fold_id < fold.fold_id) & oof.grades_blend]
        pooled = {}
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            pooled[name] = (prior[f"mu_h_{name}"].tolist()
                            + prior[f"mu_a_{name}"].tolist())
        y = prior.home_score.tolist() + prior.away_score.tolist()
        if len(y) < 2:
            continue
        expected = dist_mod.compute_score_weights(pooled, np.asarray(y))
        if not expected:
            continue
        got = oof[oof.fold_id == fold.fold_id].iloc[0]
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            assert got[f"weight_{name}"] == pytest.approx(
                expected.get(name, 0.0), abs=1e-9), (fold.fold_id, name)


def test_make_score_member_rejects_unknown_names():
    with pytest.raises(KeyError):
        dist_mod._make_score_member("nope")


def test_ensemble_mus_price_through_the_identical_variance_machinery():
    df = _frame(n=60)
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    mu_h, mu_a = ens.predict(df)
    frame = df.head(20).copy()
    frame["mu_h"] = mu_h[:20]
    frame["mu_a"] = mu_a[:20]
    priced = dist_mod._distribution_price(frame, {}, n_draws=200)
    for col in ("mu_margin", "mu_total"):
        assert col in priced.columns, col
        assert np.isfinite(priced[col].to_numpy(float)).all(), col
    single = dist_mod.ScoreRegressor().fit(df)
    sh, sa = single.predict(df)
    frame2 = df.head(20).copy()
    frame2["mu_h"] = sh[:20]
    frame2["mu_a"] = sa[:20]
    priced2 = dist_mod._distribution_price(frame2, {}, n_draws=200)
    # Same pricer, same frame, single-model mus: the production-grid leg
    # contract is identical. Per-row adaptive fair-line extras (half[j]
    # medians) legitimately follow the mus, so only the deterministic
    # grid legs are pinned, not the exact column list.
    fixed = set()
    for kind, prefix, pp, under in (
            ("total", "p_over", "p_push_total", "p_under"),
            ("margin", "p_home_cover", "p_push", "p_away_cover")):
        original = (config.TOTAL_GRID if kind == "total"
                    else config.SPREAD_GRID)
        grid = set([*original, *[float(x) + .5 for x in original],
                    *([] if kind == "total" else config.HALF_STOP_LINES)])
        for line in grid:
            fixed |= {dist_mod._grid_key(prefix, line),
                      dist_mod._grid_key(pp, line),
                      dist_mod._grid_key(under, line)}
    assert fixed.issubset(set(priced.columns))
    assert fixed.issubset(set(priced2.columns))
    assert (set(priced.columns) ^ set(priced2.columns)).isdisjoint(fixed)


def test_elasticnet_full_view_flag_off_is_byte_identical_arm_b(monkeypatch):
    """Default-off challenger flag: elasticnet matrix == linear view exactly."""
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_FULL_VIEW", False)
    try:
        from backend import moneyline as ml_mod
    except ImportError:
        import moneyline as ml_mod
    df = _frame()
    ens = dist_mod.ScoreEnsembleRegressor()
    got = ens._matrix("elasticnet", df)
    want = ml_mod.member_matrix("elasticnet", df)
    assert list(got.columns) == list(want.columns)
    pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                  want.reset_index(drop=True))


def test_elasticnet_full_view_adds_raw_sides_not_team_ids(monkeypatch):
    """Arm C view: raw per-side columns in, team-ID categoricals out."""
    monkeypatch.setattr(config, "SCORE_EN_ELASTICNET_FULL_VIEW", True)
    df = _frame()
    ens = dist_mod.ScoreEnsembleRegressor()
    got = ens._matrix("elasticnet", df)
    raw = [c for c in config.RAW_PER_SIDE_COLS if c in df.columns]
    assert raw, "fixture must carry raw per-side columns for this test"
    assert set(raw) <= set(got.columns)
    for col in config.TREE_CATEGORICAL_COLS:
        assert col not in got.columns
    # Still fittable end to end through the linear preprocessor path.
    ens2 = dist_mod.ScoreEnsembleRegressor().fit(df)
    p = ens2.predict_member(df.head(5), "elasticnet", "home_score")
    assert np.isfinite(p).all() and (p > 0).all()
