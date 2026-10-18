"""NBA spread key-number transfer contract (2026-10-16, NBA-only mechanism).

The shipped spread lattice over-predicts M = -1, +1, -6, +14, +15, +18 and
sits under the actuals at M = -10, -9, -2, +6, +7, +11, +13, +16 — every
tooth smaller than its own 1-sigma binomial floor, so the generic localized
residual prices them shrink-limited and cannot make the donor -> recipient
trade its mass-proportional sink forbids. The pre-registered transfer in
``distributions.fit_keynumber_transfer`` moves mass from the registered
donors into the registered recipients, sized DYNAMICALLY from both sides of
the evidence and gated at the unchanged generic bar. These tests pin the
contract: sizing is read from the evidence (demand / capacity / amount /
pro-rata split), a donor the actuals show short funds exact 0, a recipient
the actuals show long demands exact 0, the vector is a zero-sum pure trade
that prices a valid CDF with the no-tie bin empty, the held-out gate and the
collateral guard veto, the walk stores the disclosure either way, and a
refused cold start ships no delta (prices byte-identical).

Run: ``python -m pytest test_keynumber_transfer.py -q``
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaussian_family as fam
import distributions as dist

SPORT = dist.DISTRIBUTION_SPORT
DONORS = dist.KEYNUMBER_TRANSFER_DONORS
RECIPIENTS = dist.KEYNUMBER_TRANSFER_RECIPIENTS


def _keynumber_evidence(shift=.008, n=8000, seed=11, flip=None):
    """Smooth base model whose actuals pile the recipient teeth and take the
    mass from the donor teeth. ``flip`` names a registered bin whose OWN
    actuals are moved the wrong way (model short there, for donor tests /
    model long there, for recipient tests)."""
    rng = np.random.default_rng(seed)
    support = np.arange(-30, 31)
    base = np.exp(-.5 * ((support - 1.) / 10.) ** 2)
    base = base / base.sum()
    truth = base.copy()
    for v in RECIPIENTS:
        truth[support == v] += shift
    for v in DONORS:
        truth[support == v] -= shift
    if flip in RECIPIENTS:                 # actuals even lower: no demand
        truth[support == flip] -= 2 * shift
    elif flip in DONORS:                   # actuals higher: model SHORT there
        truth[support == flip] += 2 * shift
    truth = np.clip(truth, 1e-9, None)
    truth = truth / truth.sum()
    return (np.tile(base, (n, 1)), rng.choice(support, size=n, p=truth),
            support)


def _at(delta, support, value):
    return float(np.asarray(delta)[int(np.searchsorted(support, value))])


def test_amount_and_sizing_are_read_from_the_evidence():
    """Demand, capacity, the amount and both pro-rata splits are all read
    from THIS evidence — no constant anywhere, and amount never exceeds
    either side of the trade."""
    pmfs, outcomes, support = _keynumber_evidence()
    delta, rep = dist.fit_keynumber_transfer(pmfs, outcomes, support)
    assert rep["estimator"] == "keynumber_spread_transfer"
    assert rep.get("significant") and delta is not None
    assert rep["bar"] == fam.LATTICE_GATE_Z          # generic bar, no structural claim
    assert rep["demand_relief"] == dist.KEYNUMBER_DEMAND_RELIEF
    # The claim is the declared demand; the template gives each ACTIVE
    # recipient the same unit — proportions carry no outcome data.
    assert rep["amount"] == pytest.approx(rep["demand_total"], abs=1e-6)
    assert rep["template"] == "equal_per_active_recipient"
    rows = {r["bin"]: r for r in rep["bins"]}
    assert set(rows) == set(DONORS) | set(RECIPIENTS)
    active = [v for v in RECIPIENTS if rows[v]["demand"] > 0.]
    assert active
    gains = sum(rows[v]["gain"] for v in RECIPIENTS)
    draws = sum(rows[v]["drawn"] for v in DONORS)
    for v in active:
        assert rows[v]["gain"] == pytest.approx(rep["amount"] / len(active),
                                                 abs=1e-6)
        assert _at(delta, support, v) == pytest.approx(rows[v]["gain"], abs=1e-5)
        assert _at(delta, support, v) > 0.            # declared direction: up
    for v in RECIPIENTS:
        if v not in active:
            assert rows[v]["gain"] == 0. and _at(delta, support, v) == 0.
    for v in DONORS:
        # Actuals-capped: a donor never funds beyond its own excess, and the
        # donor side scales pro rata over capacity.
        donor_scale = min(1., rep["amount"] / rep["capacity_total"])
        assert rows[v]["drawn"] == pytest.approx(
            rows[v]["capacity"] * donor_scale, abs=1e-6)
        assert rows[v]["drawn"] <= rows[v]["capacity"] + 1e-9
        assert _at(delta, support, v) == pytest.approx(-rows[v]["drawn"],
                                                       abs=1e-5)
        assert _at(delta, support, v) <= 0.           # declared direction: down
    # Funding: donors first (capped), the remainder falls to the
    # mass-proportional sink over bins outside the registered set.
    assert rep["funded_by_donors"] == pytest.approx(draws, abs=2e-5)
    assert rep["funded_by_donors"] == pytest.approx(
        min(rep["amount"], rep["capacity_total"]), abs=2e-5)
    assert rep["funded_by_sink"] == pytest.approx(
        max(0., rep["amount"] - rep["funded_by_donors"]), abs=2e-5)
    # Zero-sum by construction: claim == donor funding + sink funding.
    assert gains == pytest.approx(rep["amount"], abs=2e-5)
    assert (rep["funded_by_donors"] + rep["funded_by_sink"]
            == pytest.approx(rep["amount"], abs=2e-5))
    non_registered = [i for i in range(len(support))
                      if support[i] not in set(DONORS) | set(RECIPIENTS)]
    # The sink never touches a registered bin: donors receive exactly the
    # negative of their drawn figure (asserted above) and recipients the
    # unit gain; registered bins are therefore never sink-shifted.
    assert all(support[i] not in set(DONORS) | set(RECIPIENTS)
               for i in non_registered)
    assert abs(float(np.sum(delta))) < 1e-5           # zero-sum by construction
    assert rep["z"] >= fam.LATTICE_GATE_Z
    assert rep["collateral_max_excess"] <= 0.
    assert rep["n_test"] >= fam.LATTICE_MIN_TEST


def test_short_donor_funds_exact_zero_and_discloses_capacity():
    """A donor whose own actuals show a DEFICIT (model short) must fund 0 —
    mass is never taken from a bin the evidence does not over-predict."""
    short = DONORS[0]
    pmfs, outcomes, support = _keynumber_evidence(flip=short)
    delta, rep = dist.fit_keynumber_transfer(pmfs, outcomes, support)
    rows = {r["bin"]: r for r in rep["bins"]}
    assert rows[short]["role"] == "donor"
    assert rows[short]["residual"] > 0.               # actuals show it short
    assert rows[short]["capacity"] == 0.
    assert rows[short]["drawn"] == 0.
    assert _at(delta, support, short) == 0.           # never touched
    assert abs(float(np.sum(delta))) < 1e-5


def test_long_recipient_demands_exact_zero():
    """A recipient the actuals show LONG (model under it already) demands 0."""
    long_bin = RECIPIENTS[0]
    pmfs, outcomes, support = _keynumber_evidence(flip=long_bin)
    delta, rep = dist.fit_keynumber_transfer(pmfs, outcomes, support)
    rows = {r["bin"]: r for r in rep["bins"]}
    assert rows[long_bin]["role"] == "recipient"
    assert rows[long_bin]["residual"] < 0.
    assert rows[long_bin]["demand"] == 0.
    assert rows[long_bin]["gain"] == 0.
    assert _at(delta, support, long_bin) == 0.


def test_recipients_all_long_is_refused():
    """Declared direction reversed everywhere: no recipient under-fit, so no
    transfer exists to admit."""
    rng = np.random.default_rng(5)
    support = np.arange(-30, 31)
    base = np.exp(-.5 * ((support - 1.) / 10.) ** 2)
    base = base / base.sum()
    truth = base.copy()
    for v in RECIPIENTS:
        truth[support == v] -= .004
    for v in DONORS:
        truth[support == v] += .004
    truth = np.clip(truth, 1e-9, None)
    truth = truth / truth.sum()
    pmfs = np.tile(base, (3000, 1))
    outcomes = rng.choice(support, size=3000, p=truth)
    delta, rep = dist.fit_keynumber_transfer(pmfs, outcomes, support)
    assert delta is None and rep["reason"] == "no_recipient_underfit"
    assert not rep.get("significant")


def test_pure_noise_is_refused_with_a_verdict():
    """No tooth structure in the actuals and a sub-bar tail (seed 1) ->
    the transfer fails its held-out gate with a verdict. (At the unified
    1.5 bar, louder noise realizations can admit — the accepted cost of
    the cutoff.)"""
    rng = np.random.default_rng(1)
    support = np.arange(-30, 31)
    base = np.exp(-.5 * ((support - 1.) / 10.) ** 2)
    base = base / base.sum()
    pmfs = np.tile(base, (3000, 1))
    outcomes = rng.choice(support, size=3000, p=base)
    delta, rep = dist.fit_keynumber_transfer(pmfs, outcomes, support)
    assert delta is None
    assert rep["reason"] in ("held_out_not_significant", "collateral_violation",
                             "no_recipient_underfit", "no_donor_excess")
    assert "z" in rep or "collateral_max_excess" in rep


def test_evidence_guards_refuse_short_and_misaligned_inputs():
    pmfs, outcomes, support = _keynumber_evidence()
    delta, rep = dist.fit_keynumber_transfer(pmfs[:80], outcomes[:80], support)
    assert delta is None and rep["reason"] == "insufficient_held_out_evidence"
    delta, rep = dist.fit_keynumber_transfer(None, outcomes, support)
    assert delta is None and rep["reason"] == "misaligned_evidence"
    delta, rep = dist.fit_keynumber_transfer(pmfs, outcomes, support[:10])
    assert delta is None and rep["reason"] == "misaligned_evidence"


def test_delta_prices_a_valid_cdf_and_keeps_the_forbidden_bin_empty():
    pmfs, outcomes, support = _keynumber_evidence()
    delta, rep = dist.fit_keynumber_transfer(pmfs, outcomes, support)
    assert delta is not None
    assert (np.abs(np.asarray(delta)) > 0.).any()  # no clamp: ships whole
    x = np.asarray(support, float)
    pmf = np.exp(-.5 * ((x - 1.) / 9.) ** 2)
    cdf = np.tile(np.cumsum(pmf / pmf.sum()), (2, 1))
    out = fam.lattice_correct(cdf, support, {"delta": delta}, SPORT, "margin")
    mass = np.diff(np.column_stack([np.zeros(len(cdf)), out]), axis=1)
    assert (mass >= 0.).all()
    np.testing.assert_allclose(mass.sum(axis=1), 1., atol=1e-12)
    assert (out[:, -1] == 1.).all()
    j = int(np.searchsorted(support, 0))              # no-tie forbidden bin
    assert (mass[:, j] == 0.).all()


def test_walk_stores_the_disclosure_and_composes_the_delta(monkeypatch):
    """The wiring stores the report in the margin spec (admitted or refused)
    and, when the gate admits, ships the transfer as the spec's delta."""
    pmfs, outcomes, support = _keynumber_evidence()
    evidence = {"margin": {"pmfs": pmfs, "outcomes": outcomes,
                           "support": support}}
    monkeypatch.setattr(dist.family, "fit_lattice_correction",
                        lambda *a, **k: {"significant": False, "target": a[4]})
    maps = dist._fit_lattice_maps(outcomes, {}, evidence)
    spec = maps["margin"]
    report = spec["keynumber_report"]
    assert report["estimator"] == "keynumber_spread_transfer"
    assert report.get("significant") and "delta" in spec
    assert spec["delta_source"] == "keynumber_transfer"
    assert _at(spec["delta"], support, RECIPIENTS[3]) > 0.
    assert _at(spec["delta"], support, DONORS[0]) <= 0.
    assert list(spec["support"]) == [int(support[0]), int(support[-1])]
    # Totals is untouched by a margin-only mechanism.
    assert maps["totals"] == {}


def test_composes_with_an_admitted_generic_delta(monkeypatch):
    pmfs, outcomes, support = _keynumber_evidence()
    evidence = {"margin": {"pmfs": pmfs, "outcomes": outcomes,
                           "support": support}}
    generic = np.zeros(len(support))
    generic[int(np.searchsorted(support, -3))] = .004
    generic[int(np.searchsorted(support, 3))] = -.004
    monkeypatch.setattr(dist.family, "fit_lattice_correction",
                        lambda *a, **k: {
                            "significant": True,
                            "delta": [float(v) for v in generic],
                            "support": [int(support[0]), int(support[-1])]})
    spec = dist._fit_lattice_maps(outcomes, {}, evidence)["margin"]
    assert spec["delta_source"] == "generic+keynumber_transfer"
    delta = np.asarray(spec["delta"])
    assert delta[int(np.searchsorted(support, RECIPIENTS[0]))] > 0.
    assert abs(float(delta.sum())) < 1e-5
    assert spec["keynumber_report"]["significant"]


def test_refused_transfer_ships_no_delta_and_still_discloses(monkeypatch):
    """A refusal (gate, collateral, direction, cold start) leaves the spec
    delta-free — prices byte-identical — while the verdict stays visible."""
    pmfs, outcomes, support = _keynumber_evidence()
    evidence = {"margin": {"pmfs": pmfs, "outcomes": outcomes,
                           "support": support}}
    monkeypatch.setattr(dist, "fit_keynumber_transfer",
                        lambda *a, **k: (None, {"estimator":
                                                "keynumber_spread_transfer",
                                                "reason": "held_out_not_significant"}))
    monkeypatch.setattr(dist.family, "fit_lattice_correction",
                        lambda *a, **k: {"significant": False, "target": a[4]})
    spec = dist._fit_lattice_maps(outcomes, {}, evidence)["margin"]
    assert spec["keynumber_report"]["reason"] == "held_out_not_significant"
    assert "delta" not in spec and not spec.get("significant")


def test_cold_start_without_evidence_fits_nothing():
    maps = dist._fit_lattice_maps(np.zeros(0), {}, None)
    assert maps == {"totals": {}, "margin": {}}