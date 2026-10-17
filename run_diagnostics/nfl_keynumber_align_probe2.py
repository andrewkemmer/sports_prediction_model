"""NFL key-number alignment ladder v2 (READ-ONLY, 2026-10-17).

Complaint (delivered 20261016 artifact, pooled n=2586): the Spread view's
modeled line still sits below the observed towers at M=-3
(obs 0.0715 / mod 0.0528, +0.0187) and M=+3 (0.0777 / 0.0655, +0.0122).
v1 remediation (e3f90c46) closed the FINAL-FIT holdout (+3 z 0.07, -3 z
1.77) but the chart pools every causal origin, where:

  * rows 0..~330   : cold start — LATTICE_MIN_TEST(100)/0.3 needs 334
                     prior rows, so no correction exists at all
                     (mod(+3) ~0.026 flat); irreducible under causal
                     discipline, measured and reported, not touched.
  * rows 334..~1000: shrink(n)-limited ramp (se ~ 1/sqrt(n) keeps the
                     generic delta shallow) — reachable by DEEPER
                     registered relief.
  * late rows      : -3 persists ~+0.011 (drift: obs(-3) runs 0.067
                     train vs 0.081 holdout) — spike layer is
                     gate-ceilinged (holdout z at amp -> 2.5 around
                     amp 0.0236) and the generic layer's pass-through
                     attenuates spike gains.

Levers laddered here on the cached BASELINE causal-walk evidence
(same machinery as nfl_keynumber_relief_ladder.py, gates untouched):

  (A) registered generic margin relief boost {3, 5, 6} x quantile
      {0.75, 0.85} — boost 5/6 deepen the ramp and the -3/tie closes;
      the 0.85 quantile re-tests the +14 overshoot (z -3.38/-3.74 at
      q75) since a higher quantile may drop +14 from relief.
  (B) spike proposal fraction f in {shrink (shipped), 0.45, 0.60} of
      the TRAIN residual — the shrink proposal sits at f~0.29/-3 and
      f~0.40/+3; the -3 holdout gate ceiling caps f at ~0.48.

Reported per variant: per-spike verdicts (amp/z/admitted), generic
gate block, composed-at key bins, HOLDOUT and FULL-SAMPLE (chart
proxy) residuals at the key bins, band-outside counts, MAE.

READ-ONLY: nothing is written to artifacts, no production constant is
permanently changed (all patches restored in finally blocks).

Usage:  python run_diagnostics/nfl_keynumber_align_probe2.py
Runtime: seconds (cached evidence).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "nfl-backend" / "backend"
HOLDOUT_FRAC = 0.30
MARGIN_PLOTTED = range(-15, 16)
KEY_BINS = (-14, -9, -7, -3, -2, -1, 0, 1, 2, 3, 7, 14)
NFL_SPIKE_KW = {"primary": (3, 7, -3, -7),
                "secondary": (1, 4, 6, 10, 14, 17,
                              -1, -4, -6, -10, -14, -17)}
CACHE = ROOT / "run_diagnostics" / "nfl_keynumber_relief_ladder_ev.npz"


def load_evidence():
    z = np.load(CACHE)
    return {"margin": {"pmfs": z["margin_pmfs"], "outcomes": z["margin_y"],
                       "support": z["margin_support"]}}


def view(ev, delta, rows="holdout"):
    """Residual/z vectors for holdout tail or full sample (chart proxy)."""
    support = np.asarray(ev["support"])
    pmfs = np.asarray(ev["pmfs"], float)
    y = np.asarray(ev["outcomes"], float)
    n = len(y)
    if rows == "holdout":
        k = int(n * HOLDOUT_FRAC)
        y, pmfs = y[n - k:], pmfs[n - k:]
    inside = ((y >= support[0]) & (y <= support[-1]) & np.isfinite(y))
    y, pmfs = y[inside], pmfs[inside]
    idx = np.searchsorted(support, y).astype(int)
    obs = np.bincount(idx, minlength=len(support)).astype(float)
    obs /= max(len(idx), 1)
    model = pmfs.mean(axis=0) + (np.zeros(len(support)) if delta is None
                                 else np.asarray(delta, float))
    resid = obs - model
    pbar = np.clip(pmfs.mean(axis=0), 0, 1)
    se = np.sqrt(pbar * (1 - pbar) / max(len(idx), 1))
    return support, resid, se, float(np.abs(resid).mean())


def at(support, resid, se, bins):
    out = {}
    for b in bins:
        if support[0] <= b <= support[-1]:
            i = int(np.searchsorted(support, b))
            out[str(b)] = {"resid": round(float(resid[i]), 5),
                           "z": round(float(resid[i] / se[i]), 2)
                           if se[i] > 0 else 0.0}
    return out


def band_outside(support, resid, se, bar=2.0):
    return [int(b) for b in MARGIN_PLOTTED
            if support[0] <= b <= support[-1]
            and se[int(np.searchsorted(support, b))] > 0
            and abs(resid[int(np.searchsorted(support, b))]
                    / se[int(np.searchsorted(support, b))]) > bar]


def fit_spikes_rule(family, pmfs, outcomes, support, sport, rule="shrink",
                    frac=1.0, **kw):
    """fit_margin_spikes with a pluggable TRAIN-side amplitude proposal.

    Everything else is the production path: same per-bin structural gate
    on the held-out tail (one-sided, LATTICE_STRUCTURAL_Z), same sink
    removal, zero-sum, structural cap, rounding. rule='shrink' is the
    shipped proposal; rule='frac' proposes amp = frac * train residual.
    """
    support = np.asarray(support)
    pmfs = np.asarray(pmfs, float)
    outcomes = np.asarray(outcomes, float)
    report = {"estimator": "key_number_spikes", "sport": sport,
              "bar": float(family.LATTICE_STRUCTURAL_Z), "spikes": [],
              "proposal": rule, "frac": float(frac)}
    inside = ((outcomes >= support[0]) & (outcomes <= support[-1])
              & np.isfinite(outcomes))
    n_test = int(family.LATTICE_TEST_FRACTION * len(outcomes))
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    hold = (np.flatnonzero(inside[len(outcomes) - n_test:])  # GLOBAL rows: flatnonzero on a tail slice is slice-relative
                   + (len(outcomes) - n_test))
    primary = kw.get("primary", (3, 7))
    secondary = kw.get("secondary", (1, 4, 6, 10, 14, 17))
    candidates = ([(int(v), False) for v in primary]
                  + [(int(v), True) for v in secondary])
    candidates = [(v, s) for v, s in candidates if support[0] <= v <= support[-1]]
    n_train = max(int(len(train)), 1)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    se = np.sqrt(pbar * (1. - pbar) / n_train)
    obs = np.bincount(np.searchsorted(support, outcomes[train]),
                      minlength=len(support)).astype(float)
    obs /= n_train
    resid = obs - pmfs[train].mean(axis=0)
    sink = np.array([i for i in range(len(support))
                     if int(support[i]) not in [v for v, _ in candidates]])
    sink_mass = float(pbar[sink].sum()) if len(sink) else 0.
    if sink_mass <= 0.:
        report["reason"] = "no_sink_mass"
        return None, report
    idx_hold = np.searchsorted(support, outcomes[hold])
    bar = float(family.LATTICE_STRUCTURAL_Z)
    spikes = []
    for value, sec in candidates:
        i = int(np.searchsorted(support, value))
        mult = family.LATTICE_SPIKE_SECONDARY_SHRINK if sec else 1.
        shrink_amp = float(family.lattice_local_deltas(
            pbar[i:i + 1], resid[i:i + 1], se[i:i + 1], shrink_mult=mult,
            prior_mult=family.LATTICE_SPIKE_RELIEF_BOOST)[0])
        if rule == "frac":
            amp = float(frac) * float(resid[i])
            amp = max(amp, shrink_amp)      # never shallower than shipped
        else:
            amp = shrink_amp
        row = {"bin": value, "secondary": bool(sec),
               "amplitude": round(amp, 6),
               "shrink_amp": round(shrink_amp, 6)}
        if not amp > 0. or resid[i] <= 0.:
            row["admitted"] = False
            row["reason"] = "amplitude_not_positive"
            report["spikes"].append(row)
            continue
        trial = np.zeros(len(support))
        trial[i] = amp
        trial[sink] = -amp * pbar[sink] / sink_mass
        d = (-2. * trial[idx_hold] + 2. * (pmfs[hold] @ trial)
             + float(trial @ trial))
        tse = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        z = float(-d.mean() / tse) if tse > 0. else 0.
        row["z"] = round(z, 3)
        if d.mean() < 0. and z >= bar:
            row["admitted"] = True
            spikes.append((i, amp))
        else:
            row["admitted"] = False
            row["reason"] = "held_out_not_significant"
        report["spikes"].append(row)
    if not spikes:
        report["reason"] = "no_spike_admitted"
        return None, report
    delta = np.zeros(len(support))
    for i, amp in spikes:
        delta[i] += amp
        delta[sink] -= amp * pbar[sink] / sink_mass
    delta = family._zero_sum_mass_proportional(delta, pbar)
    delta = np.clip(delta, -family.LATTICE_SPIKE_CAP, family.LATTICE_SPIKE_CAP)
    delta[np.abs(delta) < family.LATTICE_DELTA_EPS] = 0.
    delta = np.round(delta, family.LATTICE_ROUND)
    report["significant"] = True
    report["cap"] = float(family.LATTICE_SPIKE_CAP)
    return [float(v) for v in delta], report


def variant(family, mev, label, rule="shrink", frac=1.0,
            relief_boost=None, quantile=None, prior_frac=None):
    real_relief = family.LATTICE_RELIEF
    real_pf = family.LATTICE_PRIOR_FRAC
    family.LATTICE_RELIEF = dict(real_relief)
    if prior_frac is not None:
        family.LATTICE_PRIOR_FRAC = float(prior_frac)
    if relief_boost is not None:
        family.LATTICE_RELIEF[("nfl", "margin")] = {
            "boost": float(relief_boost),
            "quantile": float(quantile if quantile is not None
                              else family.LATTICE_RELIEF_QUANTILE)}
    try:
        spike_delta, rep = fit_spikes_rule(
            family, mev["pmfs"], mev["outcomes"], mev["support"], "nfl",
            rule=rule, frac=frac, **NFL_SPIKE_KW)
        row = {"label": label, "rule": rule, "frac": frac,
               "relief_boost": relief_boost, "quantile": quantile,
               "prior_frac": prior_frac,
               "admitted": [s["bin"] for s in rep["spikes"]
                            if s.get("admitted")]}
        if spike_delta is None:
            row["reason"] = rep.get("reason")
            return row
        row["verdicts"] = {s["bin"]: {k: s.get(k) for k in
                                      ("amplitude", "z", "admitted", "reason")}
                           for s in rep["spikes"]}
        adjusted = np.clip(np.asarray(mev["pmfs"], float)
                           + np.asarray(spike_delta, float)[None, :], 0., None)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        gen = family.fit_lattice_correction(adjusted, mev["outcomes"],
                                            mev["support"], "nfl", "margin")
        row["generic"] = {k: gen.get(k) for k in
                          ("significant", "reason", "gate_z",
                           "residual_mae_before", "residual_mae_after",
                           "collateral_ok", "collateral_max_excess",
                           "delta_max")}
        if not gen.get("significant"):
            row["generic_rejected"] = True
            return row
        parts = [spike_delta]
        if gen.get("delta") is not None:
            parts.append(gen["delta"])
        weight = np.clip(np.asarray(mev["pmfs"], float).mean(axis=0), 0., None)
        sup = np.asarray(mev["support"])
        caps = np.full(len(sup), family.LATTICE_DELTA_CAP)
        for s in rep["spikes"]:
            if s.get("admitted"):
                j = int(np.searchsorted(sup, s["bin"]))
                if 0 <= j < len(sup) and int(sup[j]) == int(s["bin"]):
                    caps[j] = family.LATTICE_SPIKE_CAP
        composed = np.asarray(family.compose_lattice_deltas(
            parts, weight, caps=caps), float)
        row["composed_at"] = {str(b): round(float(composed[
            int(np.searchsorted(sup, b))]), 5)
            for b in (-14, -7, -3, -1, 0, 1, 2, 3, 7, 14)
            if sup[0] <= b <= sup[-1]}
        row["delta_max"] = round(float(np.abs(composed).max()), 6)
        # chart proxy: FULL-sample residuals (all evidence rows pooled)
        for tag, rows in (("full", "full"), ("holdout", "holdout")):
            s_, r_, se_, mae = view(mev, composed, rows=rows)
            row[f"{tag}_mae"] = mae
            row[f"{tag}_key_bins"] = at(s_, r_, se_, KEY_BINS)
            row[f"{tag}_band_outside"] = band_outside(s_, r_, se_)
        return row
    finally:
        family.LATTICE_RELIEF = real_relief
        family.LATTICE_PRIOR_FRAC = real_pf


def gate_decomposition(family, mev):
    """Why does -3's spike gate z collapse with amp while +3's holds?"""
    support = np.asarray(mev["support"])
    pmfs = np.asarray(mev["pmfs"], float)
    y = np.asarray(mev["outcomes"], float)
    n = len(y)
    n_test = int(family.LATTICE_TEST_FRACTION * n)
    train = np.flatnonzero(np.isfinite(y[:n - n_test]))
    tail = np.arange(n - n_test, n)
    hold = tail[np.isfinite(y[tail])]
    train = train[(y[train] >= support[0]) & (y[train] <= support[-1])]
    hold = hold[(y[hold] >= support[0]) & (y[hold] <= support[-1])]
    n_tr = len(train)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    se = np.sqrt(pbar * (1. - pbar) / n_tr)
    obs = np.bincount(np.searchsorted(support, y[train]),
                      minlength=len(support)).astype(float) / n_tr
    resid = obs - pmfs[train].mean(axis=0)
    candidates = [(int(v), False) for v in (3, 7, -3, -7)] + [
        (int(v), True) for v in (1, 4, 6, 10, 14, 17,
                                 -1, -4, -6, -10, -14, -17)]
    sink = np.array([i for i in range(len(support))
                     if int(support[i]) not in [v for v, _ in candidates]])
    sink_mass = float(pbar[sink].sum())
    out = {}
    for value in (3, -3, 7, -7):
        i = int(np.searchsorted(support, value))
        base_amp = float(family.lattice_local_deltas(
            pbar[i:i + 1], resid[i:i + 1], se[i:i + 1])[0])
        frac = base_amp / resid[i] if resid[i] > 0 else 0.
        lines = []
        for amp in (base_amp, 0.02, 0.03, resid[i]):
            trial = np.zeros(len(support))
            trial[i] = amp
            trial[sink] = -amp * pbar[sink] / sink_mass
            rec = {"amp": round(amp, 5)}
            for tag, rows in (("train", train), ("hold", hold)):
                idx = np.searchsorted(support, y[rows])
                lin = float(-2. * trial[idx].mean())
                mod = float(2. * (pmfs[rows] @ trial).mean())
                quad = float(trial @ trial)
                d = -2. * trial[idx] + 2. * (pmfs[rows] @ trial) + quad
                tse = float(d.std(ddof=1) / np.sqrt(len(d)))
                rec[tag] = {"lin": round(lin, 5), "mod": round(mod, 5),
                            "quad": round(quad, 5),
                            "mean_d": round(float(d.mean()), 5),
                            "z": round(-float(d.mean()) / tse, 3)}
            lines.append(rec)
        out[str(value)] = {"train_resid": round(float(resid[i]), 5),
                           "shrink_amp": round(base_amp, 5),
                           "shrink_frac": round(frac, 3),
                           "pbar_train": round(float(pbar[i]), 5),
                           "pbar_hold": round(float(pmfs[hold].mean(axis=0)[i]), 5),
                           "obs_train": round(float(
                               (np.searchsorted(support, y[train]) == i).mean()), 5),
                           "obs_hold": round(float(
                               (np.searchsorted(support, y[hold]) == i).mean()), 5),
                           "grid": lines}
    return out


def main() -> int:
    sys.path.insert(0, str(BACKEND))
    family = importlib.import_module("gaussian_family")
    ev = load_evidence()
    mev = ev["margin"]
    record = {"evidence": CACHE.name, "n": int(len(mev["outcomes"])),
              "read_only": True,
              "shipped_delivered_pooled": {
                  "-3": 0.0187, "3": 0.0122, "-7": 0.0070, "7": 0.0038,
                  "0": -0.0110},
              "gate_decomposition": gate_decomposition(family, mev)}
    record["variants"] = [
        variant(family, mev, "V0 shipped (relief3 q.75, shrink)",
                relief_boost=3., quantile=.75),
        variant(family, mev, "V1 relief5 q.75", relief_boost=5., quantile=.75),
        variant(family, mev, "V2 relief6 q.75", relief_boost=6., quantile=.75),
        variant(family, mev, "V3 relief5 q.85", relief_boost=5., quantile=.85),
        variant(family, mev, "V4 relief6 q.85", relief_boost=6., quantile=.85),
        variant(family, mev, "V5 relief5 f.45", rule="frac", frac=.45,
                relief_boost=5., quantile=.75),
        variant(family, mev, "V6 relief3 f.45", rule="frac", frac=.45,
                relief_boost=3., quantile=.75),
        # LATTICE_PRIOR_FRAC sweep: raises tau for BOTH layers (spike
        # proposal deepens too) at every mass bin, no quantile involved.
        variant(family, mev, "V7 pf.12 relief3", prior_frac=.12,
                relief_boost=3., quantile=.75),
        variant(family, mev, "V8 pf.12 relief5", prior_frac=.12,
                relief_boost=5., quantile=.75),
        variant(family, mev, "V9 pf.15 relief3", prior_frac=.15,
                relief_boost=3., quantile=.75),
        variant(family, mev, "V10 pf.15 relief5", prior_frac=.15,
                relief_boost=5., quantile=.75),
    ]
    # Shippable deep-amp lever: LATTICE_SPIKE_RELIEF_BOOST (prior_mult on
    # the spike proposal only) is byte-identical across the four families
    # but only NFL's compose site calls fit_margin_spikes in production.
    real_sb = family.LATTICE_SPIKE_RELIEF_BOOST
    for sb in (1.5, 2., 3.):
        family.LATTICE_SPIKE_RELIEF_BOOST = float(sb)
        try:
            for rb in (3., 4.):
                record["variants"].append(variant(
                    family, mev, f"V11 sb{sb} relief{rb}",
                    relief_boost=float(rb), quantile=.75))
        finally:
            family.LATTICE_SPIKE_RELIEF_BOOST = real_sb
    dest = ROOT / "run_diagnostics" / "nfl_keynumber_align_probe2.json"
    dest.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str))
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
