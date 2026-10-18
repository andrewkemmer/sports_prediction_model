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


def test_shift_is_capped():
    h = np.full(10, 150.)
    a = np.full(10, 150.)
    tilt = {"slope": 500., "intercept": 0., "pace_mean": 0., "cap": 6.}
    np.testing.assert_array_equal(family.tilt_shift(h, a, tilt, "nba"),
                                  np.full(10, 6.))
