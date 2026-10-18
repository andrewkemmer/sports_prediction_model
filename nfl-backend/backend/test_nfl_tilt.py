"""NFL totals pace-tilt trial (2026-10-18): unit contract.

Same mechanism as the NBA V3a adoption, opened to NFL: per-origin slope
+ intercept on pace deviation shifts ONLY the total location (margin
untouched), thin-history fallback, flag-gated production path.
"""
import numpy as np

import gaussian_family as family


def _synthetic(n=800, seed=11, tilt=0.35):
    rng = np.random.default_rng(seed)
    h = rng.normal(22., 4., n)
    a = rng.normal(22., 4., n)
    tot = h + a + tilt * (h + a - 45.) + rng.normal(0., 7., n)
    mar = (h - a) + rng.normal(0., 7., n)
    return h, a, {"total": tot, "margin": mar}


def test_tilt_fits_on_nfl_scale_with_positive_slope():
    h, a, out = _synthetic()
    got = family.fit_tilt(h, a, out, "nfl")
    assert got["total_tilt"]["slope"] > 0.
    # Truth 0.35/pt raw ~ 15 per pace-dev unit; accept a wide band.
    assert 3. < got["total_tilt"]["slope"] < 60.


def test_tilt_moves_totals_only():
    h, a, out = _synthetic()
    g = family.fit_variance(h, a, out, "nfl")
    t = family.fit_tilt(h, a, out, "nfl")
    _, c0 = family.raw_cdf(h, a, "total", "nfl", g)
    _, c1 = family.raw_cdf(h, a, "total", "nfl", dict(g, **t))
    assert not np.array_equal(c0, c1)
    _, m0 = family.raw_cdf(h, a, "margin", "nfl", g)
    _, m1 = family.raw_cdf(h, a, "margin", "nfl", dict(g, **t))
    np.testing.assert_array_equal(m0, m1)


def test_no_block_is_byte_identical():
    h, a, out = _synthetic()
    g = family.fit_variance(h, a, out, "nfl")
    _, c0 = family.raw_cdf(h, a, "total", "nfl", g)
    _, c1 = family.raw_cdf(h, a, "total", "nfl", dict(g))
    np.testing.assert_array_equal(c0, c1)


def test_thin_history_falls_back():
    h, a, out = _synthetic(n=100)
    assert family.fit_tilt(h, a, out, "nfl") == {}


def test_other_sports_have_no_tilt():
    h, a, out = _synthetic()
    assert family.fit_tilt(h, a, out, "mlb") == {}
    assert family.fit_tilt(h, a, out, "nhl") == {}


def test_adopted_fit_maps_carries_tilt_when_flag_on(monkeypatch):
    import distributions as dist
    assert dist.DISTRIBUTION_SPORT == "nfl"
    h, a, out = _synthetic()
    monkeypatch.setattr(dist.config, "DISTRIBUTION_TOTAL_TILT_PACE", True)
    maps = dist._distribution_fit_maps(h, a, out)
    assert "total_tilt" in maps["variance"]
    assert maps["variance"]["total_tilt"]["slope"] > 0.
    monkeypatch.setattr(dist.config, "DISTRIBUTION_TOTAL_TILT_PACE", False)
    maps = dist._distribution_fit_maps(h, a, out)
    assert "total_tilt" not in maps["variance"]
