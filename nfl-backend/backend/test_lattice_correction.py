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
import copy
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
    monkeypatch.setattr(fam, "fit_pooled_home_share",
                        lambda *a, **k: (None, {}))
    monkeypatch.setattr(fam, "fit_margin_spikes",
                        lambda *a, **k: (None, {"spikes": []}))
    frame = _walk_frame()
    _, bundle = dist.causal_distribution_frame(frame, n_draws=500)
    maps = dict(bundle["maps"])
    a = dist._distribution_cdfs(frame, maps, sampled=False)
    legacy = {k: v for k, v in maps.items() if k != "lattice"}
    b = dist._distribution_cdfs(frame, legacy, sampled=False)
    for kind in ("total", "margin"):
        np.testing.assert_array_equal(a[kind][1], b[kind][1])


def test_shared_gaussian_family_copies_are_byte_identical():
    """Drift guard: gaussian_family.py is shared infrastructure mirrored in
    all four backends — the copies must be byte-identical (sha256 equality).
    A one-copy edit silently forking behavior per sport already occurred
    once; it is a test failure now, not a smoke-time discovery."""
    import hashlib
    root = Path(__file__).resolve().parents[2]
    copies = [root / f"{sport}-backend" / "backend" / "gaussian_family.py"
              for sport in ("mlb", "nba", "nfl", "nhl")]
    missing = [str(p) for p in copies if not p.is_file()]
    assert not missing, f"missing shared copies: {missing}"
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in copies]
    assert all(d == digests[0] for d in digests), \
        "gaussian_family.py copies drifted: " + ", ".join(
            f"{p.parent.parent.name}={d[:12]}" for p, d in zip(copies, digests))


# ---------------------------------------------------------------------------
# Section 9 extensions: structural spikes, pooled share, collateral guard,
# compose contract (2026-10-16). This file ships in all four backends with
# SPORT bound per copy; spike/share mechanics are sport-agnostic math, so
# every backend pins the same contract.
# ---------------------------------------------------------------------------

def _spike_evidence(amplitude=.04, n=2500, seed=7):
    rng = np.random.default_rng(seed)
    support = np.arange(-30, 31)
    base = np.exp(-.5 * ((support - 1.5) / 13.) ** 2)
    base = base / base.sum()
    truth = base.copy()
    truth[support == 3] += amplitude
    truth[support == 7] += amplitude
    truth = truth / truth.sum()
    return np.tile(base, (n, 1)), rng.choice(support, size=n, p=truth), support


def test_structural_spikes_admit_primaries_individually_and_compose():
    pmfs, outcomes, support = _spike_evidence()
    delta, rep = fam.fit_margin_spikes(pmfs, outcomes, support, SPORT)
    assert rep["estimator"] == "key_number_spikes"
    assert rep.get("significant") and delta is not None
    by_bin = {s["bin"]: s for s in rep["spikes"]}
    assert by_bin[3]["admitted"] and by_bin[7]["admitted"]
    assert by_bin[3]["z"] >= fam.LATTICE_STRUCTURAL_Z
    # Bin 7's evidence sits between the bars -- exactly the structural
    # bar's jurisdiction: admitted here, rejected under the generic bar.
    assert fam.LATTICE_STRUCTURAL_Z <= by_bin[7]["z"] < fam.LATTICE_GATE_Z
    # Secondary key numbers carry no excess here: each gated individually,
    # each rejected with its own verdict row.
    for v in (1, 4, 6, 10, 14, 17):
        assert not by_bin[v]["admitted"] and "reason" in by_bin[v]
    for row in rep["spikes"]:
        assert {"bin", "admitted", "amplitude"} <= set(row)
        assert ("z" in row) if row["admitted"] else ("reason" in row)
    d = np.asarray(delta, float)
    assert abs(d.sum()) <= len(d) * 5e-7 + 1e-9              # zero-sum
    assert (np.abs(d) <= fam.LATTICE_DELTA_CAP + 1e-12).all()  # clamp
    assert d[support == 3] > 0. and d[support == 7] > 0.    # spike direction
    assert (d == np.round(d, fam.LATTICE_ROUND)).all()        # stored rounding
    # Compose keeps the contract: spike delta + zero generic residual gives
    # the spike vector back up to the stored rounding, still zero-sum.
    combo = np.asarray(fam.compose_lattice_deltas(
        [delta, [0.] * len(d)]), float)
    np.testing.assert_allclose(combo, d, atol=1e-6)
    assert abs(combo.sum()) <= len(combo) * 5e-7 + 1e-9
    with pytest.raises(ValueError):
        fam.compose_lattice_deltas([delta, None])


def test_spike_fit_rejects_pure_noise_with_verdicts():
    rng = np.random.default_rng(11)
    support = np.arange(-30, 31)
    base = np.exp(-.5 * ((support - 1.5) / 13.) ** 2)
    base = base / base.sum()
    outcomes = rng.choice(support, size=1500, p=base)
    delta, rep = fam.fit_margin_spikes(np.tile(base, (1500, 1)), outcomes,
                                       support, SPORT)
    assert delta is None and rep["reason"] == "no_spike_admitted"
    assert len(rep["spikes"]) == 8
    assert not any(s.get("admitted") for s in rep["spikes"])


def _mirror_spike_evidence(amplitude=.04, n=2500, seed=9):
    rng = np.random.default_rng(seed)
    support = np.arange(-30, 31)
    base = np.exp(-.5 * ((support + 1.5) / 13.) ** 2)
    base = base / base.sum()
    truth = base.copy()
    truth[support == -3] += amplitude
    truth[support == -7] += amplitude
    truth = truth / truth.sum()
    return np.tile(base, (n, 1)), rng.choice(support, size=n, p=truth), support


def test_structural_spikes_admit_mirror_bins_when_piled():
    """Away-side mirrors gate exactly like the home-side table: explicit
    candidates, one-sided structural bar, individually admitted, zero-sum
    composed. Pins the scoring-symmetry extension (spike_m3/spike_m7)."""
    pmfs, outcomes, support = _mirror_spike_evidence()
    delta, rep = fam.fit_margin_spikes(
        pmfs, outcomes, support, SPORT,
        primary=(3, 7, -3, -7),
        secondary=(1, 4, 6, 10, 14, 17, -1, -4, -6, -10, -14, -17))
    assert rep["estimator"] == "key_number_spikes"
    assert rep.get("significant") and delta is not None
    by_bin = {s["bin"]: s for s in rep["spikes"]}
    assert len(rep["spikes"]) == 16
    assert by_bin[-3]["admitted"] and by_bin[-7]["admitted"]
    assert by_bin[-3]["z"] >= fam.LATTICE_STRUCTURAL_Z
    assert by_bin[-7]["z"] >= fam.LATTICE_STRUCTURAL_Z
    d = np.asarray(delta, float)
    assert d[support == -3] > 0. and d[support == -7] > 0.  # spike direction
    assert abs(d.sum()) <= len(d) * 5e-7 + 1e-9              # zero-sum
    assert (np.abs(d) <= fam.LATTICE_DELTA_CAP + 1e-12).all()  # clamp


def test_preregistered_table_covers_nfl_mirror_spikes():
    """The disclosure record lists the away-side mirrors as rule-derived,
    one-sided, structural-bar mechanisms — a mechanism not listed here
    cannot take the structural bar."""
    for name in ("spike_m3", "spike_m7", "spike_secondary_mirror"):
        spec = fam.LATTICE_PREREGISTERED[("nfl", "margin", name)]
        assert spec["test"] == "one-sided"
        assert spec["direction"] == "amplitude > 0"
        assert spec["structural_bar"] == fam.LATTICE_STRUCTURAL_Z


def test_nfl_lattice_fit_passes_mirror_candidates_to_spikes():
    """Wiring: _fit_lattice_maps fits NFL margin spikes over the symmetric
    table, so piled -3/-7 evidence admits mirrors in the spike_report."""
    if SPORT != "nfl":
        pytest.skip("mirror candidates are an NFL-caller extension; "
                    "elsewhere the default 8-candidate table applies")
    pmfs, outcomes, support = _mirror_spike_evidence()
    maps = dist._fit_lattice_maps(
        {}, {}, {"margin": {"pmfs": pmfs, "outcomes": outcomes,
                             "support": support}})
    by_bin = {s["bin"]: s
                for s in maps["margin"]["spike_report"]["spikes"]}
    assert by_bin[-3]["admitted"] and by_bin[-7]["admitted"]


def _share_evidence(tilt=.15, n=900, seed=2):
    rng = np.random.default_rng(seed)
    support = np.arange(-20, 21)
    base = np.exp(-.5 * (support / 6.) ** 2)
    base = base / base.sum()
    p = np.clip(base + np.where(support == 1, tilt,
                                np.where(support == -1, -tilt, 0.)), 0., None)
    p = p / p.sum()
    return np.tile(base, (n, 1)), rng.choice(support, size=n, p=p), support


def test_pooled_share_admits_home_tilt_with_report():
    pmfs, outcomes, support = _share_evidence()
    share, rep = fam.fit_pooled_home_share(outcomes, pmfs=pmfs,
                                           support=support)
    assert share is not None and .35 <= share <= .60 and share > .5
    assert rep["estimator"] == "hierarchical_pooled"
    assert rep["structural"] is False and rep["bar"] == fam.LATTICE_GATE_Z
    assert rep["z"] >= fam.LATTICE_GATE_Z
    assert len(rep["blocks"]) >= 2 and "fallback" not in rep


def test_structural_bar_admits_what_generic_rejects_and_enforces_direction():
    pmfs, outcomes, support = _share_evidence(tilt=.03, n=500, seed=4)
    generic, rep_g = fam.fit_pooled_home_share(outcomes, pmfs=pmfs,
                                               support=support)
    assert generic is None
    assert rep_g["reason"] == "held_out_not_significant"
    assert fam.LATTICE_STRUCTURAL_Z <= rep_g["z"] < fam.LATTICE_GATE_Z
    structural, rep_s = fam.fit_pooled_home_share(
        outcomes, pmfs=pmfs, support=support,
        bar=fam.LATTICE_STRUCTURAL_Z, structural=True)
    assert structural is not None and structural > .5
    assert rep_s["structural"] is True and rep_s["bar"] == 2.5
    # Pre-registered direction enforced BEFORE the test: an away tilt is a
    # structural no-op, never a two-sided retry.
    pmfs_a, outcomes_a, _ = _share_evidence(tilt=-.15, n=900, seed=2)
    refused, rep_r = fam.fit_pooled_home_share(
        outcomes_a, pmfs=pmfs_a, support=support,
        bar=fam.LATTICE_STRUCTURAL_Z, structural=True)
    assert refused is None
    assert rep_r["reason"] == "structural_direction_failed"
    # The generic bar is two-sided by design: the same away tilt admits
    # below one-half (documents the generic/structural distinction).
    admitted, _ = fam.fit_pooled_home_share(outcomes_a, pmfs=pmfs_a,
                                            support=support)
    assert admitted is not None and admitted < .5


def test_pooled_share_partial_pooling_sanity_and_fallback():
    support = np.arange(-20, 21)
    base = np.exp(-.5 * (support / 6.) ** 2)
    base = base / base.sum()

    def _draws(tilt, n, seed):
        r = np.random.default_rng(seed)
        p = np.clip(base + np.where(support == 1, tilt,
                                    np.where(support == -1, -tilt, 0.)),
                    0., None)
        return r.choice(support, size=n, p=p / p.sum())
    # Nonstationary truth (strong tilt, then neutral): the pooled estimate
    # never extrapolates beyond the per-block extremes, even though the
    # held-out gate rightly refuses to admit it.
    out = np.concatenate([_draws(.15, 450, 5), _draws(0., 450, 6)])
    _, rep = fam.fit_pooled_home_share(out, pmfs=np.tile(base, (900, 1)),
                                      support=support)
    assert rep["reason"] == "held_out_not_significant"
    assert min(rep["blocks"]) <= rep["share"] <= max(rep["blocks"])
    # Thin evidence (<2 qualifying blocks): reported fallback to the raw
    # pooled train share -- never hidden, never admitted on noise.
    rng = np.random.default_rng(0)
    out = rng.choice(support, size=335, p=base)
    _, rep = fam.fit_pooled_home_share(out, pmfs=np.tile(base, (335, 1)),
                                      support=support)
    assert rep.get("fallback") == "raw_pooled_train_share"
    assert len(rep["blocks"]) < 2


def test_collateral_guard_measures_excess_over_the_noise_floor():
    r = np.array([.01, -.02, .005])
    d = np.array([.008, -.015, .02])
    se = np.array([.002, .002, .002])
    excess = fam._collateral_excess(r, d, se)
    assert (excess[:2] < 0.).all()            # improving bins: negative excess
    assert excess[2] == pytest.approx(.008)   # damaged bin: excess over floor
    np.testing.assert_allclose(               # zero delta: exactly -se
        fam._collateral_excess(r, np.zeros(3), se), -se)
    # The admitted comb fit stores a clean collateral verdict with the delta.
    pmfs, outcomes, support = _comb_evidence(.02)
    fitted = fam.fit_lattice_correction(pmfs, outcomes, support, SPORT,
                                        "totals")
    assert fitted["significant"] and "delta" in fitted
    assert fitted["collateral_ok"] and fitted["collateral_max_excess"] <= 0.


def test_lattice_targets_price_independently_both_directions():
    """Synthetic admitted deltas per target move disjoint priced-column
    sets (both directions non-vacuous), and enabling both moves exactly
    the union -- the pricing-path mirror of
    run_diagnostics/verify_target_isolation.py."""
    frame = _walk_frame()
    _, bundle = dist.causal_distribution_frame(frame, n_draws=200)
    maps0 = bundle["maps"]

    def _synth(kind):
        sup = fam.support_for("total" if kind == "totals" else "margin",
                              SPORT)
        dd = np.where(np.arange(len(sup)) % 2 == 0, .03, -.03)
        return {"significant": True, "delta": [float(v) for v in dd]}
    outs = {}
    for t, m in ((False, False), (True, False),
                 (False, True), (True, True)):
        maps = copy.deepcopy(maps0)
        maps["lattice"]["totals"] = _synth("totals") if t \
            else {"significant": False}
        maps["lattice"]["margin"] = _synth("margin") if m \
            else {"significant": False}
        outs[(t, m)] = dist._distribution_price(frame, maps, n_draws=200)
    cols = sorted(set().union(*[set(o.columns) for o in outs.values()]))

    def _vec(key, col):
        o = outs[key]
        if col not in o.columns:
            return np.full(len(o), np.nan)
        s = o[col]
        if pd.api.types.is_numeric_dtype(s):
            return np.asarray(s, float)
        return np.asarray(s, dtype=object)

    def _changed(key):
        out = []
        for col in cols:
            x, y = _vec((False, False), col), _vec(key, col)
            try:
                same = np.array_equal(np.nan_to_num(x, nan=-999.),
                                      np.nan_to_num(y, nan=-999.))
            except (TypeError, ValueError):
                same = np.array_equal(x, y)
            if not same:
                out.append(col)
        return out
    changed_t = _changed((True, False))
    changed_m = _changed((False, True))
    assert changed_t and changed_m                     # both non-vacuous
    assert not (set(changed_t) & set(changed_m))       # disjoint ownership
    assert sorted(_changed((True, True))) == sorted(set(changed_t)
                                                    | set(changed_m))


def test_support_drift_realigns_by_bin_value():
    # Dynamic Gaussian supports expand per-block with fitted skew/scale: a
    # delta fitted on the evidence-union lattice (here totals 0..250) must
    # still price a block whose support grew by one bin (0..251) instead of
    # raising. Bins the fit saw keep their value; the unseen bin gets 0.
    fit_support = np.arange(0, 251)
    delta = np.zeros(251)
    delta[100] = 0.004
    delta[50] = -0.001
    spec = {"target": "totals", "sport": SPORT, "significant": True,
            "delta": [float(v) for v in delta],
            "support": [int(fit_support[0]), int(fit_support[-1])]}
    wide = np.arange(0, 252)
    cdf = _smooth_cdf(wide, 44., 7., rows=2)
    out = fam.lattice_correct(cdf, wide, spec, SPORT, "totals")
    assert out.shape == cdf.shape
    assert np.all(out[:, -1] == 1.) and np.all(np.diff(out, axis=1) >= -1e-12)
    # Exact expectation: fitted bins keep their value, unseen bin 251 gets
    # 0, then the usual clip + renormalize (totals has no forbidden bins).
    base = np.diff(np.column_stack([np.zeros(2), cdf]), axis=1)
    got = np.diff(np.column_stack([np.zeros(2), out]), axis=1)
    realigned = np.zeros(252)
    realigned[:251] = delta
    expected = np.clip(base + realigned[None, :], 0., None)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.abs(got - expected).max() < 1e-12
    assert expected[0, -1] == base[0, -1]  # unseen bin uncorrected pre-norm


def test_support_drift_narrower_pricing_truncates_cleanly():
    # Pricing support narrower than the fitted lattice (union wider than the
    # current block): out-of-range fitted bins are dropped, pricing stays a
    # valid CDF.
    fit_support = np.arange(-250, 251)
    delta = np.zeros(501)
    delta[250] = 0.003
    spec = {"target": "margin", "sport": SPORT, "significant": True,
            "delta": [float(v) for v in delta],
            "support": [int(fit_support[0]), int(fit_support[-1])]}
    narrow = np.arange(-240, 241)
    cdf = _smooth_cdf(narrow, 0., 7., rows=2)
    out = fam.lattice_correct(cdf, narrow, spec, SPORT, "margin")
    assert out.shape == cdf.shape and np.all(out[:, -1] == 1.)


def test_support_drift_unknown_provenance_still_raises():
    # No stored fitted-lattice bounds: the strict alignment error stands
    # (never silently positional — margin lattices are symmetric, so a
    # positional map would shift bins).
    spec = {"target": "margin", "sport": SPORT, "significant": True,
            "delta": [0.001] * 11}
    cdf = _smooth_cdf(np.arange(-6, 7), 0., 3., rows=2)
    with pytest.raises(ValueError):
        fam.lattice_correct(cdf, np.arange(-6, 7), spec, SPORT, "margin")


def test_fitted_delta_stores_its_lattice_bounds():
    # Admitted corrections disclose the lattice they were fitted on, so
    # pricing-time realignment is by bin value, never by position.
    rng = np.random.default_rng(11)
    support = np.arange(0, 60)
    pmf = np.exp(-.5 * ((support - 30.) / 6.) ** 2)
    pmf /= pmf.sum()
    pmfs = np.tile(pmf, (700, 1))
    outcomes = np.repeat(np.arange(0, 60), 12)[:700].astype(float)
    # Inject a one-bin excess the generic gate can see.
    outcomes[:40] = 30.
    out = fam.fit_lattice_correction(pmfs, outcomes, support, SPORT, "totals")
    if out.get("delta") is not None:
        assert out["support"] == [int(support[0]), int(support[-1])]


def test_structural_lattice_fallback_is_nhl_margin_only_and_directional():
    # The empty-net mechanism is preregistered (disclosure record) with the
    # generic bar first and the one-sided structural bar as fallback.
    spec = fam.LATTICE_PREREGISTERED[("nhl", "margin", "empty_net_2_to_3")]
    assert spec["structural_bar"] == fam.LATTICE_STRUCTURAL_Z
    assert spec["generic_bar"] == fam.LATTICE_GATE_Z
    assert "empty_net_2_to_3" in fam.lattice_preregistered("nhl", "margin")
    assert fam.lattice_preregistered("nfl", "margin") != {}
    support = np.arange(-20, 21)
    r_train = np.zeros(len(support))
    r_train[int(np.searchsorted(support, 3))] = 0.01
    r_train[int(np.searchsorted(support, -3))] = 0.008
    out = {"residual_mae_before": 0.0008, "residual_mae_after": 0.0003,
           "collateral_ok": True}
    d = np.array([-0.002] * 400 + [0.0] * 400)  # mean < 0, z >> 2.5
    hit = fam._structural_lattice_fallback("nhl", "margin", support,
                                           r_train, d, 2.708, out)
    assert hit == {"structural": True, "bar": 2.5,
                   "mechanism": "empty_net_2_to_3"}
    # Direction enforced on train evidence: no excess at -3, no fallback.
    r_train[int(np.searchsorted(support, -3))] = -0.001
    assert fam._structural_lattice_fallback("nhl", "margin", support,
                                            r_train, d, 2.708, out) is None
    # Below the structural bar: no fallback even with the direction held.
    r_train[int(np.searchsorted(support, -3))] = 0.008
    assert fam._structural_lattice_fallback("nhl", "margin", support,
                                            r_train, d, 2.0, out) is None
    # Other pairs never take this path (NFL secondaries keep their own
    # spike gate; totals keep the generic bar).
    assert fam._structural_lattice_fallback("nfl", "margin", support,
                                            r_train, d, 2.708, out) is None
    assert fam._structural_lattice_fallback("nhl", "totals", support,
                                            r_train, d, 2.708, out) is None
    assert fam._structural_lattice_fallback("mlb", "margin", support,
                                            r_train, d, 2.708, out) is None


def test_margin_push_extension_ships_push_only_mass_to_the_blowout_tails():
    """Diagnostic margin-push extension (2026-10-16): the traded ±14 push
    legs plus push-only −30..+30 from the same sampled CDF — no cover/away
    legs outside the grid, and the shipped mass stays a sub-probability
    (the extension can only add shipped mass). The positive arm overlaps
    TOTAL_GRID at +24..+30 by design: those names carry MARGIN mass while a
    totals line resolves its own ``p_push_total_<U>`` namespace (asserted
    below), so the two families stay distinct columns."""
    import config
    assert config.MARGIN_PUSH_EXTENSION[0] == -30
    assert config.MARGIN_PUSH_EXTENSION[-1] == 30
    assert len(config.MARGIN_PUSH_EXTENSION) == 32   # 16 + 16 (both arms)
    assert set(config.MARGIN_PUSH_EXTENSION) & set(config.TOTAL_GRID) == \
        set(range(24, 31))
    frame = _walk_frame()
    priced, _ = dist.causal_distribution_frame(frame, n_draws=500)
    for k in config.MARGIN_PUSH_EXTENSION:
        col = dist._grid_key("p_push", float(k))
        assert col in priced.columns, col
        vals = priced[col].to_numpy(float)
        assert np.isfinite(vals).all() and (vals >= 0).all()
        assert dist._grid_key("p_home_cover", float(k)) not in priced.columns
        assert dist._grid_key("p_away_cover", float(k)) not in priced.columns
    # Writer hygiene (test_production.py's gate, asserted at the source): a
    # totals line owns its namespace. Outside the extension's overlap the
    # margin name must not exist at all; inside it BOTH columns ship and
    # carry different families, so a totals read resolving
    # ``p_push_total_<U>`` can never answer with margin mass.
    for U in config.TOTAL_GRID:
        if U in config.MARGIN_PUSH_EXTENSION:
            assert dist._grid_key("p_push", float(U)) in priced.columns, U
            assert dist._grid_key("p_push_total", float(U)) in priced.columns, U
        else:
            assert dist._grid_key("p_push", float(U)) not in priced.columns, U
    traded = [dist._grid_key("p_push", float(k)) for k in range(-14, 15)]
    ext = [dist._grid_key("p_push", float(k))
           for k in config.MARGIN_PUSH_EXTENSION]
    seam = priced[[*traded, *ext]].to_numpy(float)
    assert (seam.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (seam.sum(axis=1) >= priced[traded].to_numpy(float).sum(axis=1) - 1e-12).all()


def test_writer_persists_the_margin_push_extension(tmp_path):
    import config
    import serving
    frame = _walk_frame()
    priced, _ = dist.causal_distribution_frame(frame, n_draws=200)
    priced["kind"] = "oof"
    slate = priced.head(2).copy()
    slate["kind"] = "slate"
    out = serving.write_markets_csv(
        tmp_path / "markets.csv", tmp_path / "meta.json", priced, slate,
        {"v": 1})
    for k in config.MARGIN_PUSH_EXTENSION:
        label = f"m{-k}" if k < 0 else str(k)
        assert f"p_push_{label}" in out.columns, k
def test_margin_push_extension_mirrors_the_frontend_list():
    """The frontend keeps its own copy of the extension (the Distribution
    tab's modeled span). A one-sided edit would either hide shipped bins or
    plot bins the artifact never carries — the drift the shared-core guard
    above catches for the walk, caught here for the extension itself."""
    import ast
    import config

    root = Path(__file__).resolve().parents[2]
    src = (root / "frontend" / "nfl_market_diagnostics.py").read_text(
        encoding="utf-8")
    found = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MARGIN_PUSH_EXTENSION"
                for t in node.targets):
            found = ast.get_source_segment(src, node)
    assert found, "frontend copy of MARGIN_PUSH_EXTENSION not found"
    assert " ".join(found.split()) == \
        "MARGIN_PUSH_EXTENSION = [*range(-30, -14), *range(15, 31)]", found


def test_margin_push_extension_overlap_reads_the_totals_namespace():
    """Over the +24..+30 overlap the artifact ships BOTH families: the
    margin extension's push leg and the totals line's own push. The values
    must differ (different sampled masses) and the totals family is the one
    a totals read resolves — the mirror of the frontend's preference."""
    import config
    frame = _walk_frame()
    priced, _ = dist.causal_distribution_frame(frame, n_draws=500)
    seen_difference = False
    for U in range(24, 31):
        margin_p = priced[dist._grid_key("p_push", float(U))].to_numpy(float)
        totals_p = priced[dist._grid_key("p_push_total", float(U))].to_numpy(float)
        assert np.isfinite(margin_p).all() and np.isfinite(totals_p).all()
        assert (margin_p >= 0).all() and (totals_p >= 0).all()
        seen_difference = seen_difference or bool(
            np.abs(margin_p - totals_p).max() > 1e-9)
    assert seen_difference, "the two families shipped identical mass"


# ---------------------------------------------------------------------------
# Section 10: point-in-time core (2026-10-16). The distribution walk is ONE
# shared causal engine mirrored in all four backends. These guards pin (a)
# that the mirrored copies stay byte-identical, so a one-copy edit can never
# silently fork the point-in-time discipline per sport -- the same failure
# mode the gaussian_family drift guard above already caught once -- and (b)
# that THIS backend's walk leaks nothing backwards: not from features, not
# from outcomes, not from model structure (no origin is ever priced by a map
# fitted on itself, and the final serving maps never reprice issued rows).
# ---------------------------------------------------------------------------

PIT_CORE_FUNCTIONS = ("_distribution_blocks", "_distribution_inputs",
                      "_distribution_cdfs", "_distribution_fit_maps",
                      "causal_distribution_frame")


def _pit_core_digest(path):
    """sha256 of the point-in-time core's source, line-ending normalized."""
    import ast
    import hashlib
    import textwrap
    source = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    found = {node.name: ast.get_source_segment(source, node)
             for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef)}
    missing = [name for name in PIT_CORE_FUNCTIONS if not found.get(name)]
    assert not missing, f"{path.name} lost PIT core function(s): {missing}"
    body = "\n\n".join(textwrap.dedent(found[name])
                       for name in PIT_CORE_FUNCTIONS)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_shared_distribution_pit_core_copies_are_byte_identical():
    """The four distributions.py copies must share one causal core.

    A sport-local edit to the walk (a fit widened to the whole frame, an
    admission gate read on the origin being scored) would leak in that sport
    only and is exactly what this guard refuses.
    """
    root = Path(__file__).resolve().parents[2]
    copies = [root / f"{sport}-backend" / "backend" / "distributions.py"
              for sport in ("mlb", "nba", "nfl", "nhl")]
    absent = [str(p) for p in copies if not p.is_file()]
    assert not absent, f"missing shared copies: {absent}"
    digests = [_pit_core_digest(p) for p in copies]
    assert all(d == digests[0] for d in digests), \
        "distributions.py point-in-time core drifted: " + ", ".join(
            f"{p.parent.parent.name}={d[:12]}" for p, d in zip(copies, digests))


def test_distribution_walk_never_prices_from_the_future_or_its_own_origin():
    """Adversarial PIT contract of this backend's own distribution walk."""
    inputs = (("home_expected_runs", "away_expected_runs") if SPORT == "mlb"
              else ("mu_h", "mu_a"))
    date_col = "game_date" if SPORT == "mlb" else "gameday"
    frame = _walk_frame()
    priced, bundle = dist.causal_distribution_frame(frame, n_draws=200)
    cols = [c for c in priced if c.startswith("p_")]
    assert cols
    first = frame.index < 110

    # Later outcomes AND later feature inputs: neither may move an issued
    # price. Poisoning both covers feature leakage and outcome leakage.
    poisoned = frame.copy()
    poisoned.loc[~first, "home_score"] += 20
    poisoned.loc[~first, "away_score"] = 0
    poisoned.loc[~first, inputs[0]] = poisoned.loc[~first, inputs[0]] * 1.5
    poisoned.loc[~first, inputs[1]] = poisoned.loc[~first, inputs[1]] * 0.5
    changed, _ = dist.causal_distribution_frame(poisoned, n_draws=200)
    assert [c for c in cols if c in changed.columns] == cols,         "a later outcome or feature changed the priced column set: " +         f"{sorted(set(cols) - set(changed.columns))[:6]}"
    np.testing.assert_array_equal(
        priced.loc[first, cols].to_numpy(float),
        changed.loc[first, cols].to_numpy(float),
        err_msg="a later outcome or feature moved an already-issued price")

    # Row order is not evidence: the same rows in any order price identically.
    shuffled = frame.sample(frac=1., random_state=81)
    reordered, _ = dist.causal_distribution_frame(shuffled, n_draws=200)
    assert [c for c in cols if c in reordered.columns] == cols,         "row order changed the priced column set"
    np.testing.assert_allclose(
        priced[cols].to_numpy(float),
        reordered.reindex(frame.index)[cols].to_numpy(float),
        atol=1e-10, rtol=0.)

    # Model structure: the first origin holds no prior evidence, so it can
    # admit nothing and prices the identity base family; no origin anywhere is
    # priced by a map fitted on the origin it scores.
    opening = bundle["origins"][0]
    assert opening["n_prior"] == 0
    assert not any(opening["calibration_admitted"].values())
    for kind in ("total", "margin"):
        assert np.all(np.asarray(opening["maps"][kind], float) == 0.)
    assert all(o["n_prior"] < len(frame) for o in bundle["origins"])

    # The maps the walk issued are the maps that priced those rows (the final
    # serving fit, which may learn from every settled row, is not what the
    # issued OOF columns came from).
    tail = frame.index >= 220
    direct = dist._distribution_price(
        frame.loc[tail].drop(columns=["home_score", "away_score"]),
        bundle["origins"][-1]["maps"], n_draws=200)
    stable = [c for c in ("p_home_win_derived", "p_away_win_derived", "p_tie",
                          "p_over_fair", "p_cover_fair", "fair_total",
                          "fair_spread", "distribution_mean_total",
                          "distribution_mean_margin", "distribution_sd_total",
                          "distribution_sd_margin") if c in direct.columns]
    assert len(stable) >= 9
    np.testing.assert_allclose(priced.loc[tail, stable].to_numpy(float),
                               direct[stable].to_numpy(float),
                               atol=1e-12, rtol=0.)

    # An undated frame is refused, never silently claimed as point-in-time.
    with pytest.raises((ValueError, KeyError)):
        dist.causal_distribution_frame(frame.drop(columns=[date_col]),
                                       n_draws=200)
