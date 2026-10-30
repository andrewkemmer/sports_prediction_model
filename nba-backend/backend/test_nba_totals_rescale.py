"""GPC served totals transform (adopted 2026-10-30): unit contract.

Default-on (SCALE=2.0, GATE=0.03, CAP=0.12) shrinks disagreement-regime
totals legs toward 0.5. Agreement-regime rows, push legs, zero-mass rows,
and median lines pass through byte-identical; every row still sums to 1.
"""
import numpy as np
import pytest
from scipy.special import ndtr, ndtri

import config
import distributions as dist


def _run(q, line, fair, push=0.0, scale=2.0, gate=0.03, cap=0.12, n=8):
    qa = np.full(n, float(q))
    over = (1 - push) * qa
    under = (1 - push) * (1 - qa)
    return dist.rescale_totals_legs_toward_half(
        over, np.full(n, push), under, line,
        np.full(n, float(fair)), scale, gate, cap)


def test_gate_keeps_agreement_slice_identical():
    # Line above the median favors the under side; |q-0.5|=0.01 stays shut.
    o, p, u = _run(0.49, 225.0, 220.0)
    np.testing.assert_array_equal(o, (1 - 0.0) * 0.49)
    np.testing.assert_array_equal(u, (1 - 0.0) * 0.51)
    np.testing.assert_array_equal(p, np.zeros(8))


def test_scale_pulls_disagreement_toward_half():
    # Coherent: line above fair -> under favored (q < 0.5).
    o, p, u = _run(0.30, 225.0, 220.0)
    # Manual: sigma=(225-220)/Phi^{-1}(0.7), q=1-Phi(5/(sigma*2)).
    sig = 5.0 / ndtri(0.7)
    expect = float(np.clip(1 - ndtr(5.0 / (sig * 2.0)), 0.38, 0.62))
    assert o[0] == pytest.approx(expect)
    assert 0.30 < o[0] < 0.5
    np.testing.assert_allclose(o + p + u, np.ones(8), rtol=0, atol=1e-12)


def test_cap_binds_extreme_tails():
    # Coherent extremes: far-below-median line loves the over side.
    o, _, _ = _run(0.80, 205.0, 220.0)
    assert o[0] == pytest.approx(0.62)
    o, _, _ = _run(0.20, 235.0, 220.0)
    assert o[0] == pytest.approx(0.38)


def test_push_legs_untouched_and_mass_conserved():
    o, p, u = _run(0.25, 225.0, 220.0, push=0.02)
    np.testing.assert_array_equal(p, np.full(8, 0.02))
    np.testing.assert_allclose(o + p + u, np.ones(8), rtol=0, atol=1e-12)


def test_median_line_and_empty_mass_pass_through():
    o, p, u = _run(0.50, 220.0, 220.0)  # d == 0: scale-invariant
    np.testing.assert_array_equal(o, np.full(8, 0.50))
    o2, p2, u2 = dist.rescale_totals_legs_toward_half(
        np.zeros(4), np.zeros(4), np.zeros(4), 225.0,
        np.full(4, 220.0), 2.0, 0.03, 0.12)
    np.testing.assert_array_equal(o2, np.zeros(4))
    np.testing.assert_array_equal(u2, np.zeros(4))


def test_scale_one_is_identity_when_cap_wide():
    o, p, u = _run(0.30, 225.0, 220.0, scale=1.0, cap=10.0)
    # scale 1 reproduces q up to the no-op clip; allow fp tolerance here
    # (the served no-op path skips the helper entirely when SCALE==1.0).
    np.testing.assert_allclose(o, np.full(8, 0.30), rtol=0, atol=1e-9)


def test_params_none_when_disabled_or_non_nba(monkeypatch):
    assert dist._totals_rescale_params() is not None  # adopted default-on
    monkeypatch.setattr(dist.config, "DISTRIBUTION_TOTALS_RESCALE_ENABLED", False)
    assert dist._totals_rescale_params() is None
    monkeypatch.setattr(dist.config, "DISTRIBUTION_TOTALS_RESCALE_ENABLED", True)
    monkeypatch.setattr(dist, "DISTRIBUTION_SPORT", "nfl")
    assert dist._totals_rescale_params() is None
