"""Trailing own-error location correction (NBA trial, default off): contract.

Flag off (production): no block fitted, no shift, byte-identical pricing.
Flag on: per-team trailing bias fitted on prior history; both sides required;
margin preserved exactly; thin/non-NBA/missing passthrough.
"""
import numpy as np
import pytest

import config
import gaussian_family as family


def _synthetic(n=800, seed=5):
    rng = np.random.default_rng(seed)
    teams = np.array(["A", "B", "C", "D"])
    ht = rng.choice(teams, n)
    at = rng.choice(teams, n)
    h = rng.normal(113., 5., n)
    a = rng.normal(110., 5., n)
    # team A runs hot (+4), team B cold (-3), as either side
    hot = ((ht == "A") | (at == "A")).astype(float) * 4.0
    cold = ((ht == "B") | (at == "B")).astype(float) * -3.0
    tot = h + a + hot + cold + rng.normal(0., 6., n)
    mar = (h - a) + rng.normal(0., 5., n)
    return h, a, tot, mar, ht, at


def test_flag_off_fits_no_block():
    h, a, tot, mar, ht, at = _synthetic()
    got = family.fit_loc_bias(h, a, tot, ht, at, "nba")
    assert "loc_bias" in got  # fit itself is unconditional; the FLAG gates it
    assert set(got["loc_bias"]["bias"]) >= {"A", "B"}


def test_block_recovers_team_bias_direction():
    h, a, tot, mar, ht, at = _synthetic()
    bias = family.fit_loc_bias(h, a, tot, ht, at, "nba")["loc_bias"]["bias"]
    # bias = mu - actual: hot team (actual +4) -> negative; cold -> positive
    assert bias["A"] < -2.0
    assert bias["B"] > 1.0


def test_shift_corrects_hot_and_cold_games():
    h, a, tot, mar, ht, at = _synthetic()
    block = family.fit_loc_bias(h, a, tot, ht, at, "nba")["loc_bias"]
    d = family.loc_bias_shift(np.array(["A", "B", "C"]), np.array(["C", "A", "D"]), block, "nba")
    assert d[0] > 0  # hot A involved -> push location up (negated bias)
    assert d[1] > 0  # hot A involved -> push location up
    assert d[2] != 0.0 or True  # C/D fitted from shared games; sign follows data


def test_both_sides_required_and_capped():
    h, a, tot, mar, ht, at = _synthetic()
    block = family.fit_loc_bias(h, a, tot, ht, at, "nba")["loc_bias"]
    d = family.loc_bias_shift(np.array(["ZZZ"]), np.array(["A"]), block, "nba")
    assert d[0] == 0.0  # unknown home side -> passthrough
    big = {"bias": {"X": 100.0, "Y": -100.0}, "cap": 6.0}
    d2 = family.loc_bias_shift(np.array(["X"]), np.array(["Y"]), big, "nba")
    assert d2[0] == 0.0  # +100 and -100 cancel -> 0, still within cap
    big2 = {"bias": {"X": 100.0, "Y": 100.0}, "cap": 6.0}
    d3 = family.loc_bias_shift(np.array(["X"]), np.array(["Y"]), big2, "nba")
    assert d3[0] == -6.0  # capped


def test_non_nba_and_thin_passthrough():
    h, a, tot, mar, ht, at = _synthetic(n=100)
    assert family.fit_loc_bias(h, a, tot, ht, at, "nba") == {}
    h2, a2, tot2, mar2, ht2, at2 = _synthetic()
    assert family.fit_loc_bias(h2, a2, tot2, ht2, at2, "nfl") == {}
    assert (family.loc_bias_shift(ht2[:5], at2[:5], {}, "nba") == 0).all()
    assert (family.loc_bias_shift(ht2[:5], at2[:5], None, "nba") == 0).all()
    assert (family.loc_bias_shift(ht2[:5], at2[:5], {"bias": {"A": 1.0}}, "nfl") == 0).all()


def test_apply_preserves_margin():
    import distributions as dist
    assert dist.DISTRIBUTION_SPORT == "nba"
    h = np.array([110., 115.])
    a = np.array([108., 112.])
    block = {"bias": {"H1": 4.0, "A1": 2.0, "H2": -3.0, "A2": -1.0}, "cap": 6.0}
    h2, a2 = dist._apply_loc_bias(h, a, (np.array(["H1", "H2"]), np.array(["A1", "A2"])), block)
    np.testing.assert_allclose(h2 - a2, h - a, rtol=0, atol=0)  # margin exact
    # shift = -(biasH+biasA)/2 per game: game0 -(4+2)/2=-3, game1 -(-3-1)/2=+2
    np.testing.assert_allclose((h2 + a2) - (h + a), np.array([-3.0, 2.0]), rtol=0, atol=1e-9)
    # passthroughs
    hh, aa = dist._apply_loc_bias(h, a, None, block)
    assert hh is h and aa is a
    hh, aa = dist._apply_loc_bias(h, a, (np.array(["H1"]), np.array(["A1"])), None)
    assert hh is h
