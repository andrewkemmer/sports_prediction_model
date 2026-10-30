"""Spreads GPC served transform (adopted 2026-10-30): unit contract.

Default-on (SCALE=6.0, GATE=0.06, CAP=0.12) shrinks disagreement-regime
spread legs toward 0.5. Agreement-regime rows (|q-0.5| <= 0.06), push legs,
zero-mass rows, and median lines pass through byte-identical; every row
still sums to 1. Same Normal-CDF core as the totals GPC.
"""
import numpy as np
import pytest
from scipy.special import ndtr, ndtri

import config
import distributions as dist


def _run(q, line, fair, push=0.0, scale=6.0, gate=0.06, cap=0.12, n=8):
    qa = np.full(n, float(q))
    home = (1 - push) * qa
    away = (1 - push) * (1 - qa)
    return dist.rescale_spread_legs_toward_half(
        home, np.full(n, push), away, line,
        np.full(n, float(fair)), scale, gate, cap)


def test_gate_keeps_agreement_slice_identical():
    # |q-0.5|=0.05 is inside the wider 0.06 spreads gate: outputs are the
    # EXACT input objects (np.where selection, no float touch).
    # (Under the totals 0.03 gate this row would rescale.)
    home_in = np.full(8, 0.55)
    away_in = np.full(8, 0.45)
    h, p, a = dist.rescale_spread_legs_toward_half(
        home_in, np.zeros(8), away_in, -4.5,
        np.full(8, -2.0), 6.0, 0.06, 0.12)
    np.testing.assert_array_equal(h, home_in)
    np.testing.assert_array_equal(a, away_in)
    np.testing.assert_array_equal(p, np.zeros(8))


def test_scale_pulls_disagreement_toward_half():
    # Coherent: line above fair spread -> away cover favored (q < 0.5).
    h, p, a = _run(0.30, 5.0, 2.0)
    sig = 3.0 / ndtri(0.7)
    expect = float(np.clip(1 - ndtr(3.0 / (sig * 6.0)), 0.38, 0.62))
    assert h[0] == pytest.approx(expect)
    assert 0.30 < h[0] < 0.5
    np.testing.assert_allclose(h + p + a, np.ones(8), rtol=0, atol=1e-12)


def test_cap_binds_extreme_tails():
    # With S=6 the cap needs |z| >= 1.83 (q >= 0.966): 0.90 never reaches it.
    h, _, _ = _run(0.97, -8.0, -2.0)
    assert h[0] == pytest.approx(0.62)
    h, _, _ = _run(0.03, 8.0, 2.0)
    assert h[0] == pytest.approx(0.38)


def test_push_legs_untouched_and_mass_conserved():
    h, p, a = _run(0.25, 5.0, 2.0, push=0.02)
    np.testing.assert_array_equal(p, np.full(8, 0.02))
    np.testing.assert_allclose(h + p + a, np.ones(8), rtol=0, atol=1e-12)


def test_median_line_and_empty_mass_pass_through():
    h, p, a = _run(0.50, 2.0, 2.0)  # d == 0: scale-invariant
    np.testing.assert_array_equal(h, np.full(8, 0.50))
    h2, p2, a2 = dist.rescale_spread_legs_toward_half(
        np.zeros(4), np.zeros(4), np.zeros(4), 5.0,
        np.full(4, 2.0), 6.0, 0.06, 0.12)
    np.testing.assert_array_equal(h2, np.zeros(4))
    np.testing.assert_array_equal(a2, np.zeros(4))


def test_scale_one_is_identity_when_cap_wide():
    h, p, a = _run(0.70, -1.5, -2.0, scale=1.0, cap=10.0)
    np.testing.assert_allclose(h, np.full(8, 0.70), rtol=0, atol=1e-9)


def test_params_none_when_disabled_or_non_nba(monkeypatch):
    assert dist._spread_rescale_params() is not None  # adopted default-on
    monkeypatch.setattr(dist.config, "DISTRIBUTION_SPREAD_RESCALE_ENABLED", False)
    assert dist._spread_rescale_params() is None
    monkeypatch.setattr(dist.config, "DISTRIBUTION_SPREAD_RESCALE_ENABLED", True)
    monkeypatch.setattr(dist, "DISTRIBUTION_SPORT", "nfl")
    assert dist._spread_rescale_params() is None
