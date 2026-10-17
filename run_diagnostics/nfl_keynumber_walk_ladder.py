"""NFL key-number walk-level ladder (READ-ONLY, 2026-10-17).

The chart pools EVERY causal origin, and the final-fit evidence alone
cannot say what ships: early origins are shrink(n)-limited and the
cold start (rows 0..~330, LATTICE_MIN_TEST/0.3 = 334 prior rows) never
fits at all. This probe runs FULL production causal walks with each
candidate registration patched in (restored in finally), then stacks
per-origin BASE-AND-CORRECTED pmfs the same way the delivery does and
reports, per variant:

  * pooled (chart-proxy) key-bin obs/model/residual/z over all OOF rows
    plus the band-outside count over -15..+15 (n ~ 2574, bar 2.0) --
    the exact metric the Spread view renders;
  * the per-100-row ramp of model(+3)/model(-3) (cold start + onset);
  * per-origin admissions: how many origins admit each spike bin, how
    many origins' generic fit is VETOED by collateral, composed
    delta at +/-3 at the final origin.

Candidates (from the final-fit ladder in nfl_keynumber_align_probe2.json):
  S  shipped: LATTICE_RELIEF[('nfl','margin')] boost 3, PRIOR_FRAC .10
  V1 relief5: boost 5
  V7 pf.12 relief3: PRIOR_FRAC .12, boost 3
  V8 pf.12 relief5: PRIOR_FRAC .12, boost 5

READ-ONLY: no artifact written, every patch restored in finally.

Usage:  python run_diagnostics/nfl_keynumber_walk_ladder.py [variant...]
Runtime: one causal walk ~3-5 min each; run unattended.
"""
from __future__ import annotations

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
CHART_BINS = tuple(range(-15, 16))
KEY_BINS = (-14, -7, -3, 0, 3, 7, 14)
BAR = 2.0

VARIANTS = {
    "S":  {"relief": 3., "pf": None},
    "V1": {"relief": 5., "pf": None},
    "V7": {"relief": 3., "pf": .12},
    "V8": {"relief": 5., "pf": .12},
    # shippable deep-amp finalists (align probe2 V11): LATTICE_SPIKE_
    # RELIEF_BOOST deepens ONLY the spike proposal (NFL-only production
    # caller); relief 4 is the registered per-sport LATTICE_RELIEF entry.
    "W1": {"relief": 3., "pf": None, "sb": 2.0},
    "W2": {"relief": 4., "pf": None, "sb": 1.5},
    "W3": {"relief": 4., "pf": None, "sb": 2.0},
}


def newest_markets() -> Path:
    files = sorted(DELIVERY.glob("nfl_run_engine_markets_20*.csv"))
    if not files:
        raise SystemExit("no delivered NFL markets artifact")
    return files[-1]


def stack_pooled(dist, priced, bundle):
    """Per-origin corrected pmfs (delivery equivalent) -> pooled model."""
    blocks = dist._distribution_blocks(priced)
    origins = bundle["origins"]
    assert len(blocks) == len(origins), "blocks/origins misaligned"
    model = {"margin": [], "total": []}
    y = {"margin": [], "total": []}
    for (_, _, mask), origin in zip(blocks, origins):
        part = priced.iloc[np.flatnonzero(mask)]
        cdfs = dist._distribution_cdfs(part, origin["maps"], sampled=False)
        yy = {
            "total": (part.home_score.to_numpy(float)
                      + part.away_score.to_numpy(float)),
            "margin": (part.home_score.to_numpy(float)
                       - part.away_score.to_numpy(float)),
        }
        for kind, (support, cdf) in cdfs.items():
            pmf = np.diff(np.column_stack([np.zeros(len(part)), cdf]),
                          axis=1)
            model[kind].append((pmf, support))
            y[kind].append(yy[kind])
    out = {}
    for kind in ("margin", "total"):
        low = min(int(s[0]) for _, s in model[kind])
        high = max(int(s[-1]) for _, s in model[kind])
        support = np.arange(low, high + 1)
        wide, ys = [], []
        for (pmf, sup), yy in zip(model[kind], y[kind]):
            w = np.zeros((len(pmf), len(support)))
            j0 = int(sup[0]) - low
            w[:, j0:j0 + len(sup)] = pmf
            wide.append(w)
            ys.append(yy)
        out[kind] = {"pmfs": np.vstack(wide),
                     "outcomes": np.concatenate(ys),
                     "support": support}
    return out


def pooled_metrics(kind_ev, label):
    support = np.asarray(kind_ev["support"])
    pmfs = np.asarray(kind_ev["pmfs"], float)
    y = np.asarray(kind_ev["outcomes"], float)
    inside = ((y >= support[0]) & (y <= support[-1]) & np.isfinite(y))
    y, pmfs = y[inside], pmfs[inside]
    idx = np.searchsorted(support, y).astype(int)
    obs = np.bincount(idx, minlength=len(support)).astype(float)
    obs /= max(len(idx), 1)
    model = pmfs.mean(axis=0)
    resid = obs - model
    pbar = np.clip(model, 0, 1)
    se = np.sqrt(pbar * (1 - pbar) / max(len(idx), 1))
    key = {}
    for b in KEY_BINS:
        if support[0] <= b <= support[-1]:
            i = int(np.searchsorted(support, b))
            key[str(b)] = {"obs": round(float(obs[i]), 5),
                           "model": round(float(model[i]), 5),
                           "resid": round(float(resid[i]), 5),
                           "z": round(float(resid[i] / se[i]), 2)
                           if se[i] > 0 else 0.0}
    outside = [int(b) for b in CHART_BINS
               if support[0] <= b <= support[-1]
               and se[int(np.searchsorted(support, b))] > 0
               and abs(resid[int(np.searchsorted(support, b))]
                       / se[int(np.searchsorted(support, b))]) > BAR]
    return {"n": int(len(y)),
            "mae": round(float(np.abs(resid).mean()), 6),
            "key_bins": key, "band_outside": outside}


def ramp(kind_ev, label):
    """model(+3)/model(-3) per 100 rows -- cold start and onset shape."""
    support = np.asarray(kind_ev["support"])
    pmfs = np.asarray(kind_ev["pmfs"], float)
    j3 = int(np.searchsorted(support, 3))
    jm3 = int(np.searchsorted(support, -3))
    rows = []
    for s in range(0, len(pmfs), 100):
        e = min(s + 100, len(pmfs))
        rows.append([round(float(pmfs[s:e, j3].mean()), 4),
                     round(float(pmfs[s:e, jm3].mean()), 4)])
    return rows


def origin_timeline(bundle):
    """Per-origin spike admissions / generic verdicts / composed deltas."""
    spikes_admitted: dict = {}
    generic_sig = 0
    generic_veto = 0
    first_sig_idx = None
    last = None
    for i, origin in enumerate(bundle["origins"]):
        lattice = (origin["maps"] or {}).get("lattice") or {}
        margin = lattice.get("margin") or {}
        rep = margin.get("spike_report") or {}
        for s in rep.get("spikes", []):
            if s.get("admitted"):
                spikes_admitted[s["bin"]] = spikes_admitted.get(s["bin"], 0) + 1
        if margin.get("significant"):
            generic_sig += 1
            if first_sig_idx is None:
                first_sig_idx = i
        elif margin.get("reason") not in (None, "target_not_configured",
                                          "baseline_lattice_disabled",
                                          "insufficient_history",
                                          "insufficient_held_out_evidence",
                                          "no_evidence", "delta_not_significant"):
            generic_veto += 1
        last = margin
    comp = {}
    if last:
        sup = np.asarray(last.get("support", []), dtype=float)
        delta = np.asarray(last.get("delta") or [], float)
        for b in (-3, 3):
            if len(sup) and sup[0] <= b <= sup[-1]:
                comp[str(b)] = round(float(delta[
                    int(np.searchsorted(sup, b))]), 5)
        last = {"delta_max": last.get("delta_max"),
                "delta_l1": last.get("delta_l1"),
                "gate_z": last.get("gate_z"), "composed_at": comp}
    return {"n_origins": len(bundle["origins"]),
            "spike_admitted_origins": spikes_admitted,
            "generic_significant_origins": generic_sig,
            "generic_first_significant_origin": first_sig_idx,
            "generic_other_rejects": generic_veto,
            "final_origin": last}


def run_variant(dist, family, oof, label, cfg):
    real_relief = family.LATTICE_RELIEF
    real_pf = family.LATTICE_PRIOR_FRAC
    real_sb = family.LATTICE_SPIKE_RELIEF_BOOST
    family.LATTICE_RELIEF = dict(real_relief)
    family.LATTICE_RELIEF[("nfl", "margin")] = {
        "boost": float(cfg["relief"]),
        "quantile": family.LATTICE_RELIEF_QUANTILE}
    if cfg.get("pf") is not None:
        family.LATTICE_PRIOR_FRAC = float(cfg["pf"])
    if cfg.get("sb") is not None:
        family.LATTICE_SPIKE_RELIEF_BOOST = float(cfg["sb"])
    try:
        t0 = time.time()
        priced, bundle = dist.calibrate_market_frame(oof)
        secs = round(time.time() - t0, 1)
        priced = priced.reindex(oof.index)
        pooled = stack_pooled(dist, priced, bundle)
        return {"label": label, "cfg": cfg, "walk_seconds": secs,
                "margin": pooled_metrics(pooled["margin"], label),
                "ramp_mod3_m3": ramp(pooled["margin"], label),
                "origins": origin_timeline(bundle)}
    finally:
        family.LATTICE_RELIEF = real_relief
        family.LATTICE_PRIOR_FRAC = real_pf
        family.LATTICE_SPIKE_RELIEF_BOOST = real_sb


def main() -> int:
    want = sys.argv[1:] or list(VARIANTS)
    sys.path.insert(0, str(BACKEND))
    dist = importlib.import_module("distributions")
    family = importlib.import_module("gaussian_family")
    markets = newest_markets()
    frame = pd.read_csv(markets, float_precision="round_trip")
    oof = frame[frame["kind"] == "oof"].copy()
    record = {"artifact": markets.name, "n_oof": int(len(oof)),
              "read_only": True, "variants": []}
    # shipped delivered reference (the committed CSV)
    marg = oof.home_score - oof.away_score
    ref = {}
    for b in KEY_BINS:
        col = ("p_push_m" + str(-b)) if b < 0 else ("p_push_" + str(b))
        if col in oof.columns:
            ref[str(b)] = {"obs": round(float((marg == b).mean()), 5),
                           "model": round(float(oof[col].astype(float).mean()), 5)}
            ref[str(b)]["resid"] = round(ref[str(b)]["obs"]
                                         - ref[str(b)]["model"], 5)
    record["shipped_delivered_ref"] = ref
    for name in want:
        cfg = VARIANTS[name]
        print(f"=== walking {name} {cfg}", file=sys.stderr, flush=True)
        record["variants"].append(run_variant(dist, family, oof, name, cfg))
        dest = ROOT / "run_diagnostics" / "nfl_keynumber_walk_ladder.json"
        dest.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str))
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
