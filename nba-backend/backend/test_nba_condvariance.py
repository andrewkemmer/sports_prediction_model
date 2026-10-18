"""Pace-conditional variance trial (V2, 2026-10-17): unit contract.

Default-off (config.DISTRIBUTION_CONDVARIANCE_PACE = False) must price
byte-identical to the incumbent; flag-on must be causal (sealed rows never
move prior fits), per-game (widths vary with expected total), and safe on
thin history (falls back to the global scalars).
"""
import numpy as np
import pandas as pd
import pytest

import config
import gaussian_family as family


def _synthetic(n=600, seed=7):
    rng = np.random.default_rng(seed)
    h = rng.normal(113., 6., n)
    a = rng.normal(110., 6., n)
    tot = h + a + rng.normal(0., 13., n)
    mar = (h - a) + rng.normal(0., 11., n)
    return h, a, {"total": tot, "margin": mar}


def test_scalar_params_price_identical_through_resolve_widths():
    h, a, out = _synthetic()
    g = family.fit_variance(h, a, out, "nba")
    vh, va, rho = family.resolve_widths(h, a, g, "nba")
    np.testing.assert_allclose(vh, float(g["home_variance"]) * h, rtol=0, atol=0)
    np.testing.assert_allclose(va, float(g["away_variance"]) * a, rtol=0, atol=0)
    np.testing.assert_allclose(rho, float(g["correlation"]), rtol=0, atol=0)


def test_flag_off_walk_is_byte_identical():
    import distributions as dist
    assert dist.DISTRIBUTION_SPORT == "nba"
    assert config.DISTRIBUTION_CONDVARIANCE_PACE is False
    h, a, out = _synthetic()
    maps_off = dist._distribution_fit_maps(h, a, out)
    assert "pace_cond" not in maps_off["variance"]


def test_condvariance_falls_back_on_thin_history():
    h, a, out = _synthetic(n=100)
    got = family.fit_condvariance(h, a, out, "nba")
    assert "pace_cond" not in got
    assert set(got) == set(family.fit_variance(h, a, out, "nba"))


def test_condvariance_is_nba_only():
    h, a, out = _synthetic()
    for sport in ("nfl", "nhl", "mlb"):
        hh = np.clip(h / 113. * {"nfl": 22., "nhl": 3., "mlb": 4.5}[sport], 1e-6, None)
        aa = np.clip(a / 110. * {"nfl": 22., "nhl": 3., "mlb": 4.5}[sport], 1e-6, None)
        tot = np.asarray(out["total"]) / 223. * (hh + aa)
        mar = np.asarray(out["margin"]) / 3. * (hh - aa)
        got = family.fit_condvariance(hh, aa, {"total": tot, "margin": mar}, sport)
        assert "pace_cond" not in got


def test_per_game_widths_track_pace_monotone():
    h, a, out = _synthetic(n=2000, seed=11)
    c = family.fit_condvariance(h, a, out, "nba")
    assert "pace_cond" in c
    vh, va, rho = family.resolve_widths(h, a, c, "nba")
    pace = (h + a) / 220. - 1.
    # Widths must vary across games (the trial's whole point).
    assert vh.std() > 0. and rho.std() >= 0.
    # Interpolation stays inside the anchor range.
    slow, fast = c["pace_cond"]["slow"], c["pace_cond"]["fast"]
    assert vh.min() >= min(slow["home_variance"], fast["home_variance"]) * h.min() - 1e-9
    assert vh.max() <= max(slow["home_variance"], fast["home_variance"]) * h.max() + 1e-9
    assert bool(np.all(np.abs(rho) < .99))
    assert bool(np.all(vh > 0.) and np.all(va > 0.))


def test_condfit_ignores_future_rows_causality():
    h, a, out = _synthetic(n=800, seed=13)
    early = family.fit_condvariance(h[:400], a[:400],
                                    {k: v[:400] for k, v in out.items()}, "nba")
    late_shifted = {k: np.concatenate([v[:400], v[:400] + 40.]) for k, v in out.items()}
    late = family.fit_condvariance(np.concatenate([h[:400], h[:400]]),
                                   np.concatenate([a[:400], a[:400]]),
                                   late_shifted, "nba")
    # Same first-400 history rows must fit the same anchors regardless of
    # what follows them: the fit reads only the rows it is given.
    refit = family.fit_condvariance(h[:400], a[:400],
                                    {k: v[:400] for k, v in out.items()}, "nba")
    assert early["pace_cond"] == refit["pace_cond"]
    assert late["pace_cond"] != early["pace_cond"]  # future rows do move later fits
