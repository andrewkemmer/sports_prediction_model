"""Tests for the distribution score ensemble (elasticnet + xgboost + lightgbm).

Pins the regression twins of the binary-moneyline contract: Poisson
deviance math (including the y=0 edge), the log-space simplex blender
(positivity, NaN skipping, one-hot fallback), and the walk's
point-in-time property — fold k blends with weights earned on folds
strictly before k, verified by recomputation, not by trust.
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
    teams = ["BOS", "NYK", "LAL", "CHI"]
    data.update({
        "game_id": [f"G{i}" for i in range(n)],
        "gameday": pd.date_range("2024-10-22", periods=n, freq="D"),
        "season": 2024,
        "game_type": ["Regular Season"] * n,
        "home_team": [teams[i % 4] for i in range(n)],
        "away_team": [teams[(i + 1) % 4] for i in range(n)],
        "home_score": rng.poisson(112, n).astype(float),
        "away_score": rng.poisson(108, n).astype(float),
    })
    return pd.DataFrame(data)


def test_poisson_deviance_matches_hand_computation():
    y = np.array([100.0, 110.0, 0.0])
    mu = np.array([100.0, 100.0, 50.0])
    # y=mu contributes 0; y=0 contributes 2*mu; middle by formula.
    expected = 2.0 * np.mean([
        0.0,
        110.0 * np.log(110.0 / 100.0) - 10.0,
        50.0,
    ])
    assert dist_mod.poisson_deviance(y, mu) == pytest.approx(expected)
    assert dist_mod.poisson_deviance([100.0], [100.0]) == pytest.approx(0.0)
    assert np.isnan(dist_mod.poisson_deviance([], []))


def test_log_blend_is_geometric_positive_and_nan_skipping():
    P = np.array([[100.0, 121.0], [np.nan, 110.0], [np.nan, np.nan]])
    w = np.array([0.5, 0.5])
    out = dist_mod._log_blend_mus(P, w)
    assert out[0] == pytest.approx(np.sqrt(100.0 * 121.0))
    assert out[1] == pytest.approx(110.0)  # lone finite member carries
    assert np.isnan(out[2])
    assert (out[np.isfinite(out)] > 0).all()
    # Single positive weight reproduces that member exactly.
    solo = dist_mod._log_blend_mus(
        np.array([[100.0, 121.0]]), np.array([0.0, 1.0]))
    assert solo[0] == pytest.approx(121.0)


def test_score_weights_pay_the_near_perfect_member():
    rng = np.random.default_rng(3)
    y = rng.poisson(110, 300).astype(float)
    members = {
        "xgboost": y.copy(),  # deviance 0: must earn ~everything
        "lightgbm": np.full_like(y, y.mean()),
        "elasticnet": np.full_like(y, y.mean()),
    }
    w = dist_mod.compute_score_weights(members, y)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert min(w.values()) >= 0.0
    assert w["xgboost"] == pytest.approx(1.0)
    assert dist_mod.compute_score_weights({"lightgbm": [110.0, 112.0]},
                                           [110.0, 112.0]) == {"lightgbm": 1.0}
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
    """Fold k's blend weights equal a fresh fit on folds strictly before k.

    Recomputes compute_score_weights on pooled member OOF evidence cut at
    each fold boundary and requires exact agreement with the recorded
    weight_* columns — the PIT property as an equation, not a reading.
    """
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
    # First fold blends with the cold-start constants (no evidence yet).
    first = oof[oof.fold_id == folds[0].fold_id].iloc[0]
    for name in config.SCORE_ENSEMBLE_MEMBERS:
        assert first[f"weight_{name}"] == pytest.approx(
            config.SCORE_ENSEMBLE_WEIGHTS[name])
    # Every later fold: recompute on strictly-prior grading evidence.
    for fold in folds[1:]:
        prior = oof[(oof.fold_id < fold.fold_id) & oof.grades_blend]
        pooled = {}
        for name in config.SCORE_ENSEMBLE_MEMBERS:
            pooled[name] = (prior[f"mu_h_{name}"].tolist()
                            + prior[f"mu_a_{name}"].tolist())
        y = prior.home_score.tolist() + prior.away_score.tolist()
        if len(y) < 2:
            continue  # no evidence yet: weights carry forward unchanged
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
    """Structural identity: ensemble mus feed the same asymmetric two-sided
    skew variance family + calibration + Monte Carlo as the production LGBM
    mus. The pricer only reads mu_h/mu_a — never which member produced
    them — so each member is Gaussian-for-home/away-points through the
    identical downstream distribution, by construction rather than by
    convention."""
    df = _frame(n=60)
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    mu_h, mu_a = ens.predict(df)
    frame = df.head(20).copy()
    frame["mu_h"] = mu_h[:20]
    frame["mu_a"] = mu_a[:20]
    priced = dist_mod._distribution_price(frame, {}, n_draws=200)
    for col in ("mu_margin", "mu_total", "fair_spread", "fair_total"):
        assert col in priced.columns, col
        assert np.isfinite(priced[col].to_numpy(float)).all(), col
    # Same pricer, same frame, single-model mus: identical column contract.
    single = dist_mod.ScoreRegressor().fit(df)
    sh, sa = single.predict(df)
    frame2 = df.head(20).copy()
    frame2["mu_h"] = sh[:20]
    frame2["mu_a"] = sa[:20]
    priced2 = dist_mod._distribution_price(frame2, {}, n_draws=200)
    assert [c for c in priced.columns] == [c for c in priced2.columns]


# ---------------------------------------------------------------------------
# Level-2 mixture (Option B): per-member CDFs mixed on the union lattice
# with CRPS-earned simplex weights, strictly point-in-time.
# ---------------------------------------------------------------------------


def _step_cdf(support, y):
    y = np.asarray(y, dtype=float)
    return (support[None, :] >= y[:, None]).astype(float)


def test_mixture_of_identical_cdfs_is_identity():
    sup = np.arange(200, 251)
    y = np.array([220.0, 230.0])
    cdf = _step_cdf(sup, y)
    union, mix = dist_mod.mixture_cdf(
        {"a": (sup, cdf), "b": (sup, cdf.copy())}, {"a": 0.4, "b": 0.6})
    assert (union == sup).all()
    assert np.allclose(mix, cdf)


def test_mixture_weights_pay_the_perfect_member():
    sup = np.arange(200, 251)
    y = np.arange(205.0, 235.0)
    perfect = _step_cdf(sup, y)
    ramp = np.tile((sup - 200.0) / 50.0, (len(y), 1))
    blocks = [({"sharp": (sup, perfect), "dull": (sup, ramp)}, y)]
    w = dist_mod.compute_mixture_weights(blocks)
    assert w == {"sharp": 1.0, "dull": 0.0}
    assert dist_mod.compute_mixture_weights([]) == {}


def test_mixture_skips_nan_member_and_marks_empty_rows():
    sup = np.arange(200, 251)
    y = np.array([220.0, 230.0, 240.0])
    good = _step_cdf(sup, y)
    bad = np.full_like(good, np.nan)
    bad[2] = np.nan  # row 2 also missing for `good` below
    good2 = good.copy()
    good2[2] = np.nan
    union, mix = dist_mod.mixture_cdf(
        {"a": (sup, good2), "b": (sup, bad)}, {"a": 0.4, "b": 0.6})
    assert np.allclose(mix[:2], good[:2])  # renormalized onto `a`
    assert np.isnan(mix[2]).all()  # no member prices: NaN, not zeros
    with pytest.raises(ValueError):
        dist_mod.mixture_cdf({"a": (sup, good)}, {"a": 0.0})


def test_mixture_aligns_shifted_supports_preserving_mass():
    sup_a = np.arange(200, 251)
    sup_b = np.arange(210, 261)
    union, mix = dist_mod.mixture_cdf(
        {"a": (sup_a, _step_cdf(sup_a, [220.0])),
         "b": (sup_b, _step_cdf(sup_b, [230.0]))},
        {"a": 0.5, "b": 0.5})
    assert union[0] == 200 and union[-1] == 260
    pmf = np.diff(np.column_stack([np.zeros(1), mix]), axis=1)
    assert pmf.sum() == pytest.approx(1.0)  # still a distribution
    at = {v: i for i, v in enumerate(union)}
    assert mix[0, at[219]] == pytest.approx(0.0)
    # A has jumped (1) but B has not (0) between the two steps.
    assert mix[0, at[220]] == pytest.approx(0.5)
    assert mix[0, at[225]] == pytest.approx(0.5)
    assert mix[0, at[230]] == pytest.approx(1.0)
    assert bool((np.diff(mix, axis=1) >= -1e-12).all())  # monotone


def test_price_member_cdfs_skips_nonfinite_mus():
    frame = pd.DataFrame({"game_id": ["G1", "G2", "G3"]})
    mus = {
        "xgboost": (np.array([110.0, 112.0, np.nan]),
                      np.array([108.0, 109.0, 107.0])),
        "lightgbm": (np.array([110.0, 112.0, 111.0]),
                       np.array([108.0, 109.0, 107.0])),
    }
    priced = dist_mod.price_member_cdfs(frame, mus, {})
    assert set(priced) == {"total", "margin"}
    sup, cdf = priced["total"]["xgboost"]
    assert np.isfinite(cdf[:2]).all() and np.isnan(cdf[2]).all()
    _, cdf_b = priced["total"]["lightgbm"]
    assert np.isfinite(cdf_b).all()


def _mixture_oof(n_per=30, seed=7):
    rng = np.random.default_rng(seed)
    members = list(config.SCORE_ENSEMBLE_MEMBERS)
    parts = []
    for fold_id in range(3):
        n = n_per
        hs = rng.poisson(112, n).astype(float)
        aws = rng.poisson(108, n).astype(float)
        part = pd.DataFrame({
            "game_id": [f"F{fold_id}G{i}" for i in range(n)],
            "gameday": (pd.date_range("2024-10-01", periods=n, freq="D")
                         + pd.Timedelta(days=90 * fold_id)),
            "fold_id": fold_id,
            "home_score": hs,
            "away_score": aws,
            "mu_h": hs + rng.normal(0, 2, n),
            "mu_a": aws + rng.normal(0, 2, n),
            "grades_blend": True,
        })
        for m in members:
            part[f"mu_h_{m}"] = hs + rng.normal(0, 3, n)
            part[f"mu_a_{m}"] = aws + rng.normal(0, 3, n)
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    for c in [c for c in oof.columns if c.startswith("mu_")]:
        oof[c] = np.clip(oof[c].to_numpy(float), 1.0, None)
    bundle = {"origins": [{"maps": {}} for _ in range(3)]}
    return oof, bundle


def test_mixture_walk_is_point_in_time_by_recomputation():
    """Block k's mixture weights equal a fresh CRPS fit on blocks before k.

    Mirrors the level-1 PIT test: re-price prior-block member mus through
    the same origin maps, recompute the simplex on that evidence only, and
    require exact agreement with the recorded weight_mix_* columns.
    """
    oof, bundle = _mixture_oof()
    result = dist_mod.walk_forward_mixture(oof, bundle, progress_every=0)
    mix_oof = result["oof"]
    members = list(config.SCORE_ENSEMBLE_MEMBERS)
    for kind in ("total", "margin"):
        assert f"crps_single_{kind}" in mix_oof.columns
        assert f"crps_mix_{kind}" in mix_oof.columns
        for m in members:
            assert f"weight_mix_{kind}_{m}" in mix_oof.columns
    frame = oof.reset_index(drop=True)
    blocks = dist_mod._distribution_blocks(frame)
    assert len(blocks) == 3
    # First block mixes with equal weights (no evidence yet).
    first = mix_oof.iloc[np.flatnonzero(blocks[0][2])[0]]
    for kind in ("total", "margin"):
        for m in members:
            assert first[f"weight_mix_{kind}_{m}"] == pytest.approx(1.0 / 3)
    # Later blocks: recompute on strictly-prior grading evidence.
    for j in range(1, len(blocks)):
        for kind in ("total", "margin"):
            ev = []
            for pj in range(j):
                part = frame.iloc[np.flatnonzero(blocks[pj][2])]
                mus = {m: (part[f"mu_h_{m}"].to_numpy(float),
                           part[f"mu_a_{m}"].to_numpy(float))
                       for m in members}
                priced = dist_mod.price_member_cdfs(
                    part.reset_index(drop=True), mus, {})
                y = ((part.home_score + part.away_score).to_numpy(float)
                     if kind == "total"
                     else (part.home_score - part.away_score).to_numpy(float))
                g = part.grades_blend.to_numpy(bool) & np.isfinite(y)
                keep = np.flatnonzero(g)
                ev.append(({n: (s, c[keep]) for n, (s, c) in priced[kind].items()},
                           y[g]))
            expected = dist_mod.compute_mixture_weights(ev)
            got = mix_oof.iloc[np.flatnonzero(blocks[j][2])[0]]
            for m in members:
                assert got[f"weight_mix_{kind}_{m}"] == pytest.approx(
                    expected.get(m, 0.0) if expected else 1.0 / 3, abs=1e-9), \
                    (j, kind, m)
    # Mixture output is a valid CDF story: finite CRPS on both legs.
    for kind in ("total", "margin"):
        s = mix_oof[f"crps_single_{kind}"].to_numpy(float)
        m = mix_oof[f"crps_mix_{kind}"].to_numpy(float)
        assert np.isfinite(s).all() and np.isfinite(m).all()
    assert set(result["summary"]) == {"total", "margin"}


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
    rng = np.random.default_rng(7)
    df["elo_home"] = rng.normal(1500, 50, len(df))
    df["elo_away"] = rng.normal(1500, 50, len(df))
    df["home_team_id"] = 1
    df["away_team_id"] = 2
    ens = dist_mod.ScoreEnsembleRegressor()
    got = ens._matrix("elasticnet", df)
    assert "elo_home" in got.columns and "elo_away" in got.columns
    assert "home_team_id" not in got.columns
    assert "away_team_id" not in got.columns
    # Still fittable end to end through the linear preprocessor path.
    ens2 = dist_mod.ScoreEnsembleRegressor().fit(df)
    p = ens2.predict_member(df.head(5), "elasticnet", "home_score")
    assert np.isfinite(p).all() and (p > 0).all()


def test_ensemble_drift_views_use_member_matrices():
    """Promoted-model drift view: blend-weighted member importances."""
    df = _frame()
    ens = dist_mod.ScoreEnsembleRegressor().fit(df)
    ens.set_weights({"xgboost": 0.5, "lightgbm": 0.5, "elasticnet": 0.0})
    weights = dist_mod.distribution_feature_weights(ens)
    assert weights, "fitted tree members must yield a weight map"
    assert abs(sum(weights.values()) - 100.0) < 1e-6
    assert all(v >= 0.0 for v in weights.values())
    for col in config.TREE_CATEGORICAL_COLS:
        assert col not in weights
    union = set()
    for name in config.SCORE_ENSEMBLE_MEMBERS:
        union.update(ens.columns[name])
    assert set(weights) <= union
    cols = dist_mod.run_engine_feature_cols(ens)
    assert cols == [c for c in dict.fromkeys(
        [c for name in config.SCORE_ENSEMBLE_MEMBERS
         for c in ens.columns[name]])
        if c not in config.TREE_CATEGORICAL_COLS]


def test_ensemble_drift_views_degrade_honestly():
    """Unfitted / zero-weight ensembles: None + contract fallback, no crash."""
    ens = dist_mod.ScoreEnsembleRegressor()
    assert dist_mod.distribution_feature_weights(ens) is None
    assert dist_mod.run_engine_feature_cols(ens) == [
        c for c in config.active_moneyline_feature_cols()
        if c not in config.TREE_CATEGORICAL_COLS]
    df = _frame()
    ens.fit(df)
    ens.set_weights({"xgboost": 0.0, "lightgbm": 0.0, "elasticnet": 0.0})
    assert dist_mod.distribution_feature_weights(ens) is None
