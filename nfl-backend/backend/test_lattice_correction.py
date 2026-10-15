"""Integer-lattice correction contract (2026-10-14).

The generalized, sport-aware lattice correction replaces the NHL-only
parity fix as the discrete integer-bin layer of the pricing stack: a
sawtooth/"comb" residual r(k) = P_obs(k) - P_model(k) that any smooth
continuous density integrated over integer bins leaves behind (the
low-pass artifact). These tests pin the shared contract in every backend:
byte-identical passthrough when a correction is zero or unconfigured, CDF
validity guards, the structural rules per sport/target (NHL totals OT/SO
parity, no-tie margin bins, NFL ties), the causal fit with its held-out
significance gate, and the walk wiring that stores per-target corrections
in the maps.

Run: ``python -m pytest test_lattice_correction.py -q``
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaussian_family as fam
import distributions as dist

SPORT = dist.DISTRIBUTION_SPORT


def _smooth_cdf(support, loc, scale, rows=3):
    x = np.asarray(support, float)
    pmf = np.exp(-.5 * ((x - loc) / scale) ** 2)
    pmf /= pmf.sum()
    return np.tile(np.cumsum(pmf), (rows, 1))


def test_config_map_is_sport_aware():
    assert fam.lattice_config(SPORT, "totals") is not None
    assert fam.lattice_config(SPORT, "margin") is not None
    with pytest.raises(ValueError):
        fam.lattice_config(SPORT, "moneyline")
    margin = fam.lattice_config(SPORT, "margin")
    if SPORT == "nfl":
        # Ties are possible: P(M=0) stays valid, structural rule None.
        assert margin["structural"] is None and 0 not in margin["forbidden_bins"]
    else:
        assert 0 in margin["forbidden_bins"]
        assert margin["structural"] in ("no_tie", "ot_no_tie")
    if SPORT != "nhl":
        assert fam.lattice_config(SPORT, "totals")["structural"] is None


def test_passthrough_is_object_identical_when_disabled():
    support = fam.support_for("total", SPORT)
    cdf = _smooth_cdf(support, 6., 2.)
    for correction in (None, {}, {"delta": [0.] * len(support)},
                       {"alpha": 0.}, {"significant": False}):
        assert fam.lattice_correct(cdf, support, correction,
                                   SPORT, "totals") is cdf
    # Unconfigured sport/target pairs pass through even with a nonzero delta.
    assert fam.lattice_correct(cdf, support, {"delta": [0.01] * len(support)},
                               "unknown_sport", "totals") is cdf


def test_delta_guards_produce_a_valid_cdf():
    support = np.arange(0, 21)
    cdf = _smooth_cdf(support, 10., 3.)
    delta = np.where(np.asarray(support) % 2 == 1, .004, -.004)
    out = fam.lattice_correct(cdf, support, {"delta": delta.tolist()},
                              "nhl", "totals")
    pmf = np.diff(np.column_stack([np.zeros(len(cdf)), out]), axis=1)
    assert (pmf >= 0.).all()
    np.testing.assert_allclose(pmf.sum(axis=1), 1., atol=1e-12)
    assert (np.diff(out, axis=1) >= -1e-15).all()      # monotone
    assert (out[:, -1] == 1.).all()                    # exact-tail guard
    assert (out >= 0.).all() and (out <= 1.).all()


def test_forbidden_bins_stay_empty_after_the_delta():
    support = fam.support_for("margin", SPORT)
    cdf = _smooth_cdf(support, 0., 3.)
    delta = np.full(len(support), .001)
    out = fam.lattice_correct(cdf, support, {"delta": delta.tolist()},
                              SPORT, "margin")
    pmf = np.diff(np.column_stack([np.zeros(len(cdf)), out]), axis=1)
    j = int(np.searchsorted(support, 0))
    if SPORT != "nfl":
        assert (pmf[:, j] == 0.).all()


def test_nhl_totals_structural_path_is_exactly_parity_correct():
    if SPORT != "nhl":
        pytest.skip("NHL-only structural term")
    support = fam.support_for("total", SPORT)
    cdf = _smooth_cdf(support, 6., 2.)
    direct = fam.parity_correct(cdf, support, .36)
    via_lattice = fam.lattice_correct(cdf, support, {"alpha": .36},
                                      "nhl", "totals")
    assert (direct == via_lattice).all()


def test_default_no_tie_is_byte_identical_and_weighted_split_conserves():
    support = fam.support_for("margin", SPORT)
    cdf = _smooth_cdf(support, 0., 3.)
    if SPORT == "nfl":
        assert fam.no_tie(cdf, support, SPORT, .55) is cdf
        return
    assert (fam.no_tie(cdf, support, SPORT)
            == fam.no_tie(cdf, support, SPORT, .5)).all()
    weighted = fam.no_tie(cdf, support, SPORT, .55)
    pmf = np.diff(np.column_stack([np.zeros(len(cdf)), weighted]), axis=1)
    j = int(np.searchsorted(support, 0))
    assert (pmf[:, j] == 0.).all()
    np.testing.assert_allclose(pmf.sum(axis=1), 1., atol=1e-12)
    base = np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)
    t = base[:, j]
    np.testing.assert_allclose(pmf[:, j + 1], base[:, j + 1] + .55 * t)
    np.testing.assert_allclose(pmf[:, j - 1], base[:, j - 1] + .45 * t)


def _comb_evidence(amplitude=.02, n=900, seed=0):
    rng = np.random.default_rng(seed)
    support = np.arange(0, 41)
    base = np.exp(-.5 * ((support - 20) / 6.) ** 2)
    base = base / base.sum()
    truth = np.where(np.asarray(support) % 2 == 1, amplitude, -amplitude)
    p = np.clip(base + truth, 0., None)
    p = p / p.sum()
    outcomes = rng.choice(support, size=n, p=p)
    return np.tile(base, (n, 1)), outcomes, support


def test_fit_recovers_a_real_comb_and_gates_noise():
    pmfs, outcomes, support = _comb_evidence(.02)
    fitted = fam.fit_lattice_correction(pmfs, outcomes, support, SPORT,
                                        "totals")
    assert fitted["significant"] and "delta" in fitted
    delta = np.asarray(fitted["delta"])
    # zero-sum up to the stored 1e-6 rounding
    assert abs(delta.sum()) <= len(delta) * 5e-7 + 1e-9
    odd = np.asarray(support) % 2 == 1
    assert delta[odd].mean() > 0. > delta[~odd].mean()   # comb direction
    assert fitted["residual_mae_after"] < fitted["residual_mae_before"]
    # Uniform noise must not pass the held-out gate.
    rng = np.random.default_rng(1)
    flat = np.exp(-.5 * ((support - 20) / 6.) ** 2)
    flat = flat / flat.sum()
    noise = rng.choice(support, size=len(outcomes), p=flat)
    gated = fam.fit_lattice_correction(np.tile(flat, (len(outcomes), 1)),
                                       noise, support, "nfl", "totals")
    assert not gated["significant"] and "delta" not in gated


def test_fit_refuses_insufficient_history_and_misaligned_evidence():
    pmfs, outcomes, support = _comb_evidence()
    short = fam.fit_lattice_correction(pmfs[:50], outcomes[:50], support,
                                       SPORT, "margin")
    assert short["reason"] == "insufficient_history"
    assert "delta" not in short
    with pytest.raises(ValueError):
        fam.fit_lattice_correction(pmfs[:, :10], outcomes, support,
                                   SPORT, "totals")


def test_ot_home_share_is_admitted_only_on_held_out_evidence():
    rng = np.random.default_rng(2)
    n = 900
    support = np.arange(-20, 21)
    base = np.exp(-.5 * (support / 6.) ** 2)
    base = base / base.sum()
    pmfs = np.tile(base, (n, 1))
    tilt = np.where(support == 1, .15, np.where(support == -1, -.15, 0.))
    p = np.clip(base + tilt, 0., None)
    p = p / p.sum()
    outcomes = rng.choice(support, size=n, p=p)
    # No verifiable evidence -> equal split (byte-identical no-op).
    assert fam.fit_ot_home_share(outcomes) is None
    if SPORT not in ("nhl", "mlb"):
        # The gated one-goal home-share split is wired for the NHL (OT/SO)
        # and MLB (walk-off extras) margin lattices only.
        return
    admitted = fam.fit_ot_home_share(outcomes, pmfs=pmfs, support=support)
    assert admitted is not None and .35 <= admitted <= .6 and admitted > .5


def test_lattice_evidence_aligns_uneven_block_supports():
    small_pmf = np.array([[.5, .5]])
    wide_pmf = np.array([[.25, .5, .25]])
    aligned = fam.lattice_evidence([
        (small_pmf, np.array([1.]), np.arange(0, 2)),
        (wide_pmf, np.array([2.]), np.arange(0, 3))])
    assert list(aligned["support"]) == [0, 1, 2]
    np.testing.assert_allclose(aligned["pmfs"][0], [.5, .5, 0.])
    assert fam.lattice_evidence([]) is None


def _walk_frame(n=330, seed=3):
    rng = np.random.default_rng(seed)
    h = np.clip(rng.normal(2.6, 1.1, n), .5, 8.)
    a = np.clip(rng.normal(2.5, 1.1, n), .5, 8.)
    home = np.clip(rng.poisson(h), 0, 15)
    away = np.clip(rng.poisson(a), 0, 15)
    frame = pd.DataFrame({
        "game_id": np.arange(1000, 1000 + n),
        "gameday": pd.to_datetime("2026-01-01")
                    + pd.to_timedelta(np.repeat(np.arange(3), n // 3), unit="D"),
        "fold_id": np.repeat(np.arange(3), n // 3),
        "mu_h": h, "mu_a": a,
        "home_score": home.astype(float), "away_score": away.astype(float),
    })
    if SPORT == "mlb":
        frame = frame.rename(columns={"game_id": "game_pk",
                                      "gameday": "game_date",
                                      "mu_h": "home_expected_runs",
                                      "mu_a": "away_expected_runs"})
    return frame


def test_walk_stores_per_target_lattice_maps_and_prices_validly():
    frame = _walk_frame()
    priced, bundle = dist.causal_distribution_frame(frame, n_draws=500)
    assert "lattice" in bundle["maps"]
    assert set(bundle["maps"]["lattice"]) == {"totals", "margin"}
    assert set(bundle["lattice_summary"]) == {"totals", "margin"}
    for origin in bundle["origins"]:
        lattice = origin["maps"]["lattice"]
        assert set(lattice) == {"totals", "margin"}
        for spec in lattice.values():
            assert "delta" not in spec or spec.get("significant")
    for col in ("p_over_fair", "p_cover_fair", "distribution_mean_total",
                "distribution_mean_margin"):
        assert col in priced
    assert np.isfinite(priced["p_over_fair"]).all()


def test_pre_lattice_bundles_price_identically(monkeypatch):
    """A stored bundle that predates lattice fitting (no "lattice" key)
    must price through the exact same path as an empty-lattice bundle."""
    monkeypatch.setattr(fam, "fit_lattice_correction",
                        lambda *a, **k: {"significant": False})
    monkeypatch.setattr(fam, "fit_ot_home_share", lambda *a, **k: None)
    frame = _walk_frame()
    _, bundle = dist.causal_distribution_frame(frame, n_draws=500)
    maps = dict(bundle["maps"])
    a = dist._distribution_cdfs(frame, maps, sampled=False)
    legacy = {k: v for k, v in maps.items() if k != "lattice"}
    b = dist._distribution_cdfs(frame, legacy, sampled=False)
    for kind in ("total", "margin"):
        np.testing.assert_array_equal(a[kind][1], b[kind][1])
