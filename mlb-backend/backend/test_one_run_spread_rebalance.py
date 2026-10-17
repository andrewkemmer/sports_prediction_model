"""One-run spread rebalance contract (2026-10-16, MLB-only mechanism).

The MLB margin lattice ships a structural-only correction: the gated walk-off
home-win share re-splits each row's TIE mass, which cannot absorb the model's
own one-run-final split defect (20261016: P(M=+1) 0.1433 vs 0.1739 actual,
P(M=-1) 0.1255 vs 0.1089). The pre-registered rebalance in
``distributions.fit_one_run_spread_rebalance`` closes that gap with one
constant transfer whose amount, donor capacities and funding splits are all
read from the fitted evidence. These tests pin the contract: direction is
one-sided per bin, a donor the actuals show SHORT funds exact 0, the vector
is zero-sum and valid through ``lattice_correct``, the held-out gate and the
collateral guard veto, and the walk stores the disclosure (a refused
rebalance ships no delta, so prices are byte-identical).

Run: ``python -m pytest test_one_run_spread_rebalance.py -q``
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaussian_family as fam
import distributions as dist

SPORT = dist.DISTRIBUTION_SPORT
RECIPIENT = dist.ONE_RUN_REBALANCE_RECIPIENT
MIRROR = dist.ONE_RUN_REBALANCE_MIRROR
DONORS = dist.ONE_RUN_REBALANCE_DONORS


def _one_run_evidence(deficit=.04, n=3000, seed=11):
    """Smooth base model whose actuals pile M=+1 and give back at -1 and +3."""
    rng = np.random.default_rng(seed)
    support = np.arange(-25, 26)
    base = np.exp(-.5 * ((support - .5) / 6.) ** 2)
    base = base / base.sum()
    truth = base.copy()
    truth[support == RECIPIENT] += deficit
    truth[support == MIRROR] -= deficit * .6
    truth[support == 3] -= deficit * .4
    truth = truth / truth.sum()
    return (np.tile(base, (n, 1)), rng.choice(support, size=n, p=truth),
            support)


def _at(delta, support, value):
    return float(np.asarray(delta)[int(np.searchsorted(support, value))])


def test_amount_and_funding_are_read_from_the_evidence():
    """The transfer amount is the admitted share applied to the model's own
    one-run mass, and the donors fund in proportion to their actuals."""
    pmfs, outcomes, support = _one_run_evidence()
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes, support, .6)
    assert rep["estimator"] == "one_run_spread_rebalance"
    assert rep.get("significant") and delta is not None
    pbar = pmfs[:-int(fam.LATTICE_TEST_FRACTION * len(pmfs))].mean(axis=0)
    one_run_mass = (_at(pbar, support, RECIPIENT) + _at(pbar, support, MIRROR))
    assert rep["one_run_mass"] == pytest.approx(one_run_mass, abs=1e-6)
    # No constant anywhere: the amount is (share - 1/2) * half the one-run
    # mass of THIS evidence.
    assert rep["amount"] == pytest.approx((.6 - .5) * .5 * one_run_mass,
                                          abs=1e-6)
    assert _at(delta, support, RECIPIENT) == pytest.approx(rep["amount"],
                                                           abs=1e-5)
    # Direction: recipient up, donors down, zero-sum by construction.
    assert _at(delta, support, RECIPIENT) > 0.
    for donor in DONORS:
        assert _at(delta, support, donor) <= 0.
    assert abs(float(np.sum(delta))) < 1e-6
    assert rep["z"] >= fam.LATTICE_GATE_Z
    assert rep["funded_by_sink"] == pytest.approx(0., abs=1e-9)
    assert rep["funded_by_donors"] == pytest.approx(rep["amount"], abs=1e-6)
    rows = {r["bin"]: r for r in rep["bins"]}
    assert set(rows) == set(DONORS)
    for row in rows.values():
        assert row["capacity"] >= 0.
        assert row["drawn"] == pytest.approx(
            rep["amount"] * row["capacity"] / sum(r["capacity"]
                                                  for r in rows.values()),
            abs=1e-6)


def test_short_donor_funds_exact_zero_and_discloses_capacity():
    """A donor whose own actuals show a DEFICIT (model short) must fund 0 —
    the rebalance can never widen a bin its evidence already shows short."""
    rng = np.random.default_rng(5)
    support = np.arange(-25, 26)
    base = np.exp(-.5 * ((support - .5) / 6.) ** 2)
    base = base / base.sum()
    truth = base.copy()
    truth[support == RECIPIENT] += .04
    truth[support == 2] -= .04          # M=+2 long, M=-2 SHORT
    truth[support == -2] += .02
    truth[support == 3] -= .02
    truth = truth / truth.sum()
    pmfs = np.tile(base, (3000, 1))
    outcomes = rng.choice(support, size=3000, p=truth)
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes, support, .6)
    rows = {r["bin"]: r for r in rep["bins"]}
    assert rows[-2]["residual"] > 0.               # actuals show it short
    assert rows[-2]["capacity"] == 0.
    assert rows[-2]["drawn"] == 0.
    assert _at(delta, support, -2) == 0.           # never touched
    # The rest of the funding still lands (other donors + recipient).
    assert _at(delta, support, RECIPIENT) > 0.
    assert abs(float(np.sum(delta))) < 1e-6


def test_pure_noise_is_refused_with_a_verdict():
    """No one-run tilt in the actuals -> the transfer misaligns with the
    held-out residual and the unchanged generic bar refuses it."""
    rng = np.random.default_rng(3)
    support = np.arange(-25, 26)
    base = np.exp(-.5 * ((support - .5) / 6.) ** 2)
    base = base / base.sum()
    pmfs = np.tile(base, (3000, 1))
    outcomes = rng.choice(support, size=3000, p=base)
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes, support, .6)
    assert delta is None
    assert rep["reason"] in ("held_out_not_significant", "collateral_violation")
    assert "z" in rep or "collateral_max_excess" in rep


def test_no_admitted_share_is_a_no_op():
    pmfs, outcomes, support = _one_run_evidence()
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes, support,
                                                   None)
    assert delta is None and rep["reason"] == "no_admitted_walkoff_share"
    # A share at/below the equal split has no declared direction to apply.
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes, support, .5)
    assert delta is None and rep["reason"] == "no_positive_tilt"


def test_evidence_guards_refuse_short_and_misaligned_inputs():
    pmfs, outcomes, support = _one_run_evidence()
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs[:80], outcomes[:80],
                                                   support, .6)
    assert delta is None and rep["reason"] == "insufficient_held_out_evidence"
    delta, rep = dist.fit_one_run_spread_rebalance(None, outcomes, support, .6)
    assert delta is None and rep["reason"] == "misaligned_evidence"
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes,
                                                   support[:10], .6)
    assert delta is None and rep["reason"] == "misaligned_evidence"


def test_delta_prices_a_valid_cdf_and_keeps_the_forbidden_bin_empty():
    pmfs, outcomes, support = _one_run_evidence()
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes, support, .6)
    assert delta is not None
    x = np.asarray(support, float)
    pmf = np.exp(-.5 * ((x - .5) / 6.) ** 2)
    cdf = np.tile(np.cumsum(pmf / pmf.sum()), (2, 1))
    out = fam.lattice_correct(cdf, support, {"delta": delta}, SPORT, "margin")
    mass = np.diff(np.column_stack([np.zeros(len(cdf)), out]), axis=1)
    assert (mass >= 0.).all()
    np.testing.assert_allclose(mass.sum(axis=1), 1., atol=1e-12)
    assert (out[:, -1] == 1.).all()
    j = int(np.searchsorted(support, 0))
    assert (mass[:, j] == 0.).all()


def test_walk_stores_the_disclosure_and_composes_the_delta(monkeypatch):
    """The wiring stores the report in the margin spec and, when the gate
    admits, ships the transfer as the spec's delta (composed with the generic
    delta when that one also admitted)."""
    pmfs, outcomes, support = _one_run_evidence()
    evidence = {"margin": {"pmfs": pmfs, "outcomes": outcomes,
                           "support": support}}
    monkeypatch.setattr(fam, "fit_pooled_home_share",
                        lambda *a, **k: (.6, {"significant": True}))
    monkeypatch.setattr(fam, "fit_lattice_correction",
                        lambda *a, **k: {"significant": False, "target": a[4]})
    maps = dist._fit_lattice_maps(outcomes, {}, evidence)
    spec = maps["margin"]
    assert spec["ot_home_share"] == .6
    report = spec["one_run_rebalance"]
    assert report["estimator"] == "one_run_spread_rebalance"
    assert report.get("significant") and "delta" in spec
    assert spec["delta_source"] == "one_run_rebalance"
    assert np.asarray(spec["delta"])[int(np.searchsorted(support, RECIPIENT))] > 0.
    assert list(spec["support"]) == [int(support[0]), int(support[-1])]

    # Refused share -> the rebalance is inert and ships no delta at all.
    monkeypatch.setattr(fam, "fit_pooled_home_share",
                        lambda *a, **k: (None, {"reason": "x"}))
    spec = dist._fit_lattice_maps(outcomes, {}, evidence)["margin"]
    assert spec["one_run_rebalance"]["reason"] == "no_admitted_walkoff_share"
    assert "delta" not in spec and not spec.get("significant")


def test_composes_with_an_admitted_generic_delta(monkeypatch):
    pmfs, outcomes, support = _one_run_evidence()
    evidence = {"margin": {"pmfs": pmfs, "outcomes": outcomes,
                           "support": support}}
    generic = np.zeros(len(support))
    generic[int(np.searchsorted(support, -3))] = .004
    generic[int(np.searchsorted(support, 3))] = -.004
    monkeypatch.setattr(fam, "fit_pooled_home_share",
                        lambda *a, **k: (.6, {"significant": True}))
    monkeypatch.setattr(fam, "fit_lattice_correction",
                        lambda *a, **k: {"significant": True,
                                         "delta": [float(v) for v in generic],
                                         "support": [int(support[0]),
                                                     int(support[-1])]})
    spec = dist._fit_lattice_maps(outcomes, {}, evidence)["margin"]
    assert spec["delta_source"] == "generic+one_run_rebalance"
    delta = np.asarray(spec["delta"])
    assert delta[int(np.searchsorted(support, RECIPIENT))] > 0.
    assert abs(float(delta.sum())) < 1e-6
    assert spec["one_run_rebalance"]["significant"]
