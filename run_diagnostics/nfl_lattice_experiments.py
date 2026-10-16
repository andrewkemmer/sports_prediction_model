"""NFL lattice experiments on causal-walk BASE evidence (READ-ONLY).

Re-runs the deterministic distribution walk with the lattice fits disabled
(the BASELINE of regenerate_lattice_artifacts.py), rebuilds the exact
pre-lattice BASE pmfs each origin issued, stacks them into the final
serving-fit evidence scope (all settled OOF rows, time-ordered), and runs:

  (a) cap sweep — refit the generic margin residual under
      LATTICE_DELTA_CAP in {.01, .02, .03, .05, .08}; report gate z,
      residual MAE before/after, delta L1/max, and the after-residual at
      the key bins. Production (.03) is refit first as a sanity check
      against the delivered monitor verdicts.
  (b) mirror spikes — the preregistered table covers +3/+7 only. Fit the
      symmetric mirror candidates (-3/-7 primary, -1/-4/-6/-10/-14/-17
      secondary) with the same one-sided structural bar on the same
      evidence. Verdicts only.
  (c) tie-hole accounting — baseline residual at M=0 plus the delta each
      cap level places there.
  (d) totals single-bin scan on BASE evidence — per-bin train residual,
      held-out one-sided z, and mass; structural bar + mass floor applied.

READ-ONLY: no artifact is written, no production constant is changed (the
cap override is a monkeypatch restored in a finally block), and nothing is
admitted anywhere. Output is a JSON verdict record for review.

Usage:  python run_diagnostics/nfl_lattice_experiments.py [--cap-list ...]
Runtime: one baseline causal walk (~4 min) + repricing; run unattended.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "nfl-backend" / "backend"
DELIVERY = ROOT / "nfl-backend" / "data_delivery"
HOLDOUT_FRAC = 0.30
STRUCTURAL_BAR = 2.5
MASS_FLOOR = 0.003


def newest_markets() -> Path:
    files = sorted(DELIVERY.glob("nfl_run_engine_markets_20*.csv"))
    if not files:
        raise SystemExit("no delivered NFL markets artifact")
    return files[-1]


def baseline_walk(dist, family, oof):
    """Causal walk with every admitting lattice path neutralized."""
    saved = (family.fit_lattice_correction, family.fit_ot_home_share,
             family.fit_pooled_home_share, family.fit_margin_spikes)
    family.fit_lattice_correction = lambda *a, **k: {
        "significant": False, "reason": "baseline_lattice_disabled"}
    family.fit_ot_home_share = lambda *a, **k: None
    family.fit_pooled_home_share = lambda *a, **k: (
        None, {"reason": "baseline_disabled"})
    family.fit_margin_spikes = lambda *a, **k: (
        None, {"reason": "baseline_disabled"})
    try:
        t0 = time.time()
        priced, bundle = dist.calibrate_market_frame(oof)
        return priced, bundle, round(time.time() - t0, 1)
    finally:
        (family.fit_lattice_correction, family.fit_ot_home_share,
         family.fit_pooled_home_share,
         family.fit_margin_spikes) = saved


def build_evidence(dist, family, priced, bundle):
    """Stack per-origin BASE pmfs in time order -> final-fit evidence."""
    out = {}
    blocks = dist._distribution_blocks(priced)
    origins = bundle["origins"]
    assert len(blocks) == len(origins), "blocks/origins misaligned"
    for (_, _, mask), origin in zip(blocks, origins):
        part = priced.iloc[np.flatnonzero(mask)]
        cdfs = dist._distribution_cdfs(part, origin["maps"], sampled=False)
        y = {
            "total": (part.home_score.to_numpy(float)
                      + part.away_score.to_numpy(float)),
            "margin": (part.home_score.to_numpy(float)
                       - part.away_score.to_numpy(float)),
        }
        for kind, (support, cdf) in cdfs.items():
            pmf = np.diff(np.column_stack([np.zeros(len(part)), cdf]),
                          axis=1)
            out.setdefault(kind, []).append((pmf, y[kind], support))
    return {kind: family.lattice_evidence(entries)
            for kind, entries in out.items()}


def after_residual_at(ev, delta, bins):
    support = np.asarray(ev["support"])
    pmfs = np.asarray(ev["pmfs"], float)
    y = np.asarray(ev["outcomes"], float)
    n = len(y)
    n_hold = int(n * HOLDOUT_FRAC)
    yh = y[n - n_hold:]
    ph = pmfs[n - n_hold:]
    inside = ((yh >= support[0]) & (yh <= support[-1]) & np.isfinite(yh))
    yh, ph = yh[inside], ph[inside]
    idx = np.searchsorted(support, yh).astype(int)
    obs = np.bincount(idx, minlength=len(support)).astype(float)
    obs /= max(len(idx), 1)
    resid = obs - (ph.mean(axis=0) + np.asarray(delta, float))
    pbar = np.clip(ph.mean(axis=0), 0, 1)
    se = np.sqrt(pbar * (1 - pbar) / max(len(idx), 1))
    got = {}
    for b in bins:
        if support[0] <= b <= support[-1]:
            i = int(np.searchsorted(support, b))
            got[str(b)] = {"resid_after": round(float(resid[i]), 5),
                           "z": round(float(resid[i] / se[i]), 3)
                           if se[i] > 0 else 0.0}
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="0.01,0.02,0.03,0.05,0.08")
    args = ap.parse_args()
    caps = [float(c) for c in args.caps.split(",")]
    sys.path.insert(0, str(BACKEND))
    dist = importlib.import_module("distributions")
    family = importlib.import_module("gaussian_family")

    markets = newest_markets()
    frame = pd.read_csv(markets, float_precision="round_trip")
    oof = frame[frame["kind"] == "oof"].copy()
    record: dict = {"artifact": markets.name, "n_oof": int(len(oof)),
                    "read_only": True}

    priced, bundle, secs = baseline_walk(dist, family, oof)
    priced = priced.reindex(oof.index)
    record["walk_seconds_baseline"] = secs
    ev = build_evidence(dist, family, priced, bundle)
    record["evidence_n"] = {k: int(len(v["outcomes"]))
                            for k, v in ev.items()}
    mev, tev = ev["margin"], ev["total"]

    # ---- sanity: production config refit vs delivered monitor -----------
    delivered = json.loads(
        (DELIVERY / markets.name.replace("markets", "monitor")
         .replace(".csv", ".json")).read_text())
    prod_lat = delivered["fit"]["lattice"]
    refit_prod = family.fit_lattice_correction(
        mev["pmfs"], mev["outcomes"], mev["support"], "nfl", "margin")
    record["sanity_production_refit_margin"] = {
        "delivered": {k: prod_lat["margin"].get(k) for k in
                      ("significant", "gate_z", "delta_l1", "delta_max",
                       "residual_mae_before", "residual_mae_after")},
        "refit_cap_0_03": {k: refit_prod.get(k) for k in
                           ("significant", "reason", "gate_z", "delta_l1",
                            "delta_max", "residual_mae_before",
                            "residual_mae_after")},
    }
    spikes_prod, spikes_prod_rep = family.fit_margin_spikes(
        mev["pmfs"], mev["outcomes"], mev["support"], "nfl")
    record["sanity_production_spikes"] = {
        "delivered_admitted": [s["bin"] for s in
                               prod_lat["margin"]["spike_report"]["spikes"]
                               if s.get("admitted")],
        "refit_admitted": [s["bin"] for s in spikes_prod_rep["spikes"]
                           if s.get("admitted")],
    }

    # ---- (a) cap sweep ---------------------------------------------------
    real_cap = family.LATTICE_DELTA_CAP
    sweep = {}
    try:
        for cap in caps:
            family.LATTICE_DELTA_CAP = cap
            fit = family.fit_lattice_correction(
                mev["pmfs"], mev["outcomes"], mev["support"],
                "nfl", "margin")
            delta = np.asarray(fit.get("delta") or [0.0], float)
            row = {"significant": bool(fit.get("significant")),
                   "reason": fit.get("reason"),
                   "gate_z": fit.get("gate_z"),
                   "mae_before": round(fit.get("residual_mae_before", 0), 6),
                   "mae_after": round(fit.get("residual_mae_after", 0), 6),
                   "delta_l1": fit.get("delta_l1"),
                   "delta_max": fit.get("delta_max"),
                   "after": after_residual_at(
                       mev, delta, (-7, -3, -1, 0, 1, 3, 7))}
            sweep[str(cap)] = row
    finally:
        family.LATTICE_DELTA_CAP = real_cap
    record["cap_sweep_margin"] = sweep

    # ---- (b) mirror spikes ------------------------------------------------
    _, mirror_rep = family.fit_margin_spikes(
        mev["pmfs"], mev["outcomes"], mev["support"], "nfl",
        primary=(-3, -7), secondary=(-1, -4, -6, -10, -14, -17))
    record["mirror_spikes"] = {
        "verdicts": mirror_rep["spikes"],
        "admitted": [s["bin"] for s in mirror_rep["spikes"]
                     if s.get("admitted")],
        "reading": "mirror bins clearing the one-sided structural bar are "
                   "preregistration PROPOSALS (rule-derived: scoring "
                   "symmetry) — nothing admitted by this script",
    }

    # ---- (c) tie-hole accounting ------------------------------------------
    support = np.asarray(mev["support"])
    pmfs = np.asarray(mev["pmfs"], float)
    y = np.asarray(mev["outcomes"], float)
    n = len(y)
    n_hold = int(n * HOLDOUT_FRAC)
    j0 = int(np.searchsorted(support, 0))
    tie = {}
    for label, sl in (("train", slice(0, n - n_hold)),
                      ("holdout", slice(n - n_hold, n))):
        yy, pp = y[sl], pmfs[sl]
        inside = ((yy >= support[0]) & (yy <= support[-1])
                  & np.isfinite(yy))
        obs0 = float((yy[inside] == 0).mean())
        mod0 = float(pp[inside][:, j0].mean())
        se0 = float(np.sqrt(max(mod0 * (1 - mod0), 0)
                            / max(int(inside.sum()), 1)))
        tie[label] = {"obs_push_0": round(obs0, 5),
                      "base_mod_push_0": round(mod0, 5),
                      "resid": round(obs0 - mod0, 5),
                      "z": round((obs0 - mod0) / se0, 3) if se0 else 0.0,
                      "n": int(inside.sum())}
    record["tie_hole_base_evidence"] = tie

    # ---- (d) totals single-bin scan on BASE evidence ----------------------
    tsup = np.asarray(tev["support"])
    tpmf = np.asarray(tev["pmfs"], float)
    ty = np.asarray(tev["outcomes"], float)
    tn = len(ty)
    tn_hold = int(tn * HOLDOUT_FRAC)
    tno = max(tn - tn_hold, 1)
    pbar_tr = np.clip(tpmf[:tn - tn_hold].mean(axis=0), 0, 1)
    obs_tr = np.bincount(
        np.searchsorted(tsup, ty[:tn - tn_hold]).astype(int),
        minlength=len(tsup)).astype(float) / tno
    r_tr = obs_tr - tpmf[:tn - tn_hold].mean(axis=0)
    yh, ph = ty[tn - tn_hold:], tpmf[tn - tn_hold:]
    inside = ((yh >= tsup[0]) & (yh <= tsup[-1]) & np.isfinite(yh))
    yh, ph = yh[inside], ph[inside]
    obs_h = np.bincount(np.searchsorted(tsup, yh).astype(int),
                        minlength=len(tsup)).astype(float) / max(len(yh), 1)
    pbar_h = np.clip(ph.mean(axis=0), 0, 1)
    se_h = np.sqrt(pbar_h * (1 - pbar_h) / max(len(yh), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_h = np.where(se_h > 0, (obs_h - pbar_h) / se_h, 0.0)
    rows = []
    for i in range(len(tsup)):
        if z_h[i] >= STRUCTURAL_BAR and abs(obs_h[i] - pbar_h[i]) >= MASS_FLOOR:
            rows.append({"k": int(tsup[i]),
                         "train_resid": round(float(r_tr[i]), 5),
                         "hold_resid": round(float(obs_h[i] - pbar_h[i]), 5),
                         "hold_z": round(float(z_h[i]), 3)})
    rows.sort(key=lambda r: -r["hold_z"])
    record["totals_base_scan"] = {
        "n_hold": int(len(yh)), "candidates": rows,
        "reading": "bins clearing bar+mass need a rule-derived mechanism "
                   "to take the structural bar; else generic path only",
    }

    dest = ROOT / "run_diagnostics" / "nfl_lattice_experiments.json"
    dest.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str))
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
