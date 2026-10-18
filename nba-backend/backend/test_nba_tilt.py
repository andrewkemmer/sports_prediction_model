"""Totals pace-tilt trial (V3a, 2026-10-18): unit contract.

Default-off (no total_tilt block) prices byte-identical; flag-on fits a
per-origin slope + intercept on strictly-prior history, shifts ONLY the
total location (margin untouched), falls back on thin history, NBA-only.
"""
import numpy as np
import pytest

import gaussian_family as family


def _synthetic(n=600, seed=7, tilt=2.0):
    rng = np.random.default_rng(seed)
    h = rng.normal(113., 6., n)
    a = rng.normal(110., 6., n)
    tot = h + a + tilt * (h + a - 228.) + rng.normal(0., 13., n)
    mar = (h - a) + rng.normal(0., 11., n)
    return h, a, {"total": tot, "margin": mar}


def test_tilt_recovers_synthetic_slope_direction():
    h, a, out = _synthetic()
    got = family.fit_tilt(h, a, out, "nba")
    assert got["total_tilt"]["slope"] > 0.
    # Truth 2.0/pt raw = ~440 per pace-dev unit; accept a wide band.
    assert 100. < got["total_tilt"]["slope"] < 800.


def test_tilt_moves_totals_only():
    h, a, out = _synthetic()
    g = family.fit_variance(h, a, out, "nba")
    t = family.fit_tilt(h, a, out, "nba")
    _, c0 = family.raw_cdf(h, a, "total", "nba", g)
    _, c1 = family.raw_cdf(h, a, "total", "nba", dict(g, **t))
    assert not np.array_equal(c0, c1)
    _, m0 = family.raw_cdf(h, a, "margin", "nba", g)
    _, m1 = family.raw_cdf(h, a, "margin", "nba", dict(g, **t))
    np.testing.assert_array_equal(m0, m1)


def test_no_block_is_byte_identical():
    h, a, out = _synthetic()
    g = family.fit_variance(h, a, out, "nba")
    _, c0 = family.raw_cdf(h, a, "total", "nba", g)
    _, c1 = family.raw_cdf(h, a, "total", "nba", dict(g))
    np.testing.assert_array_equal(c0, c1)
    np.testing.assert_array_equal(family.tilt_shift(h, a, None, "nba"),
                                  np.zeros(len(h)))


def test_thin_history_falls_back():
    h, a, out = _synthetic(n=100)
    assert family.fit_tilt(h, a, out, "nba") == {}


def test_non_nba_has_no_tilt():
    h, a, out = _synthetic()
    assert family.fit_tilt(h, a, out, "nfl") == {}
    np.testing.assert_array_equal(
        family.tilt_shift(h, a, {"slope": 50., "intercept": 1.,
                                 "pace_mean": 0., "cap": 6.}, "nfl"),
        np.zeros(len(h)))


def test_adopted_fit_maps_carries_tilt_when_flag_on(monkeypatch):
    import distributions as dist
    assert dist.DISTRIBUTION_SPORT == "nba"
    # Patch the config object distributions.py actually reads (its own
    # module reference), not a same-named import that may be a different
    # module instance under pytest's import mode.
    h, a, out = _synthetic(n=500)
    monkeypatch.setattr(dist.config, "DISTRIBUTION_TOTAL_TILT_PACE", True)
    maps = dist._distribution_fit_maps(h, a, out)
    assert "total_tilt" in maps["variance"]
    assert maps["variance"]["total_tilt"]["slope"] > 0.
    monkeypatch.setattr(dist.config, "DISTRIBUTION_TOTAL_TILT_PACE", False)
    maps = dist._distribution_fit_maps(h, a, out)
    assert "total_tilt" not in maps["variance"]


def test_tilt_watch_panel_reports_slope_and_gap(monkeypatch):
    import pandas as pd
    import distributions as dist
    monkeypatch.setattr(dist.config, "DISTRIBUTION_TOTAL_TILT_PACE", True)
    rng = np.random.default_rng(3)
    n = 500
    h = rng.normal(113., 5., n)
    a = rng.normal(110., 5., n)
    frame = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "gameday": pd.date_range("2025-01-01", periods=n),
        "mu_h": h, "mu_a": a,
        "home_score": rng.poisson(np.clip(h, 1, None)),
        "away_score": rng.poisson(np.clip(a, 1, None)),
        "fold_id": (np.arange(n) // 60).astype(str)})
    priced, bundle = dist.causal_distribution_frame(frame, n_draws=100)
    assert "total_tilt" in bundle["maps"]["variance"]
    rep = dist.distribution_score_report(priced, bundle)
    watch = rep["tilt_watch"]
    assert set(watch) == {"tilt", "top_quintile_over_230"}
    assert set(watch["tilt"]) == {"slope", "intercept", "sign_ok",
                                   "origin_slopes", "origin_sign_flips"}
    assert watch["top_quintile_over_230"]["n"] >= 20


def test_shift_is_capped():
    h = np.full(10, 150.)
    a = np.full(10, 150.)
    tilt = {"slope": 500., "intercept": 0., "pace_mean": 0., "cap": 6.}
    np.testing.assert_array_equal(family.tilt_shift(h, a, tilt, "nba"),
                                  np.full(10, 6.))
