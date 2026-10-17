"""NFL key-number relief ladder on causal-walk BASE evidence (READ-ONLY).

The 20261016 shipped artifact still underfits the preregistered margin key
numbers (pooled post-lattice residuals from delivered columns:
M=-3 +0.0313 z 6.15, M=+3 +0.0281 z 5.31, M=-7 +0.0128 z 3.31) and the
totals teeth (T=51 +0.0135 z 3.42, T=37 +0.0097 z 2.63, T=44 +0.0078
z 2.09, troughs T=42 -0.0080 z -3.04, T=53 -0.0088 z -4.04). Two amplitude
levers are laddered here, both gated by the standing production gates:

  (a) margin spike relief x cap — LATTICE_SPIKE_RELIEF_BOOST in
      {1, 2, 3, 4, 6} x LATTICE_SPIKE_CAP in {0.03, 0.05}: per-spike
      one-sided held-out verdicts at the boosted amplitude, the generic
      fit on spike-adjusted pmfs (its own gate z / collateral / MAE), and
      the COMPOSED vector's held-out after-residual + z at the key bins.
  (b) totals registered relief — LATTICE_RELIEF[(nfl, totals)] boost in
      {1, 2, 3, 5}: generic totals gate must stay >= 3.0 two-sided with
      MAE improving and collateral clean; after-residuals + z at the
      totals focus bins and the band-outside count over the plotted
      range are reported per boost.

READ-ONLY: no artifact is written, no production constant is permanently
changed (patches restored in finally blocks), nothing is admitted. The
variants are PROPOSALS judged by the standing gates.

Usage:  python run_diagnostics/nfl_keynumber_relief_ladder.py
Runtime: one baseline causal walk (~4 min) + fast refits; run unattended.
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
HOLDOUT_FRAC = 0.30
MARGIN_FOCUS = (-14, -7, -3, -1, 0, 1, 3, 7, 14)
MARGIN_PLOTTED = range(-15, 16)
TOTALS_FOCUS = (33, 37, 40, 41, 44, 45, 47, 51, 53)
TOTALS_PLOTTED = range(20, 101)
NFL_SPIKE_KW = {"primary": (3, 7, -3, -7),
                "secondary": (1, 4, 6, 10, 14, 17,
                              -1, -4, -6, -10, -14, -17)}


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


def holdout_view(ev, delta):
    """Held-out pooled after-residual/z vectors + overall after-MAE."""
    support = np.asarray(ev["support"])
    pmfs = np.asarray(ev["pmfs"], float)
    y = np.asarray(ev["outcomes"], float)
    n = len(y)
    n_hold = int(n * HOLDOUT_FRAC)
    yh, ph = y[n - n_hold:], pmfs[n - n_hold:]
    inside = ((yh >= support[0]) & (yh <= support[-1]) & np.isfinite(yh))
    yh, ph = yh[inside], ph[inside]
    idx = np.searchsorted(support, yh).astype(int)
    obs = np.bincount(idx, minlength=len(support)).astype(float)
    obs /= max(len(idx), 1)
    model = ph.mean(axis=0) + (np.zeros(len(support)) if delta is None
                               else np.asarray(delta, float))
    resid = obs - model
    pbar = np.clip(ph.mean(axis=0), 0, 1)
    se = np.sqrt(pbar * (1 - pbar) / max(len(idx), 1))
    return round(float(np.abs(resid).mean()), 6), support, resid, se


def at_bins(_view, support, resid, se, bins):
    out = {}
    for b in bins:
        if support[0] <= b <= support[-1]:
            i = int(np.searchsorted(support, b))
            out[str(b)] = {"resid": round(float(resid[i]), 5),
                           "z": round(float(resid[i] / se[i]), 2)
                           if se[i] > 0 else 0.0}
    return out


def band_outside(support, resid, se, plotted, bar=2.0):
    out = []
    for b in plotted:
        if support[0] <= b <= support[-1]:
            i = int(np.searchsorted(support, b))
            if se[i] > 0 and abs(resid[i] / se[i]) > bar:
                out.append(int(b))
    return out


def margin_variant(family, mev, boost, spike_cap):
    """One (boost, cap) margin spike variant -> full verdict dict."""
    real_boost = family.LATTICE_SPIKE_RELIEF_BOOST
    real_cap = family.LATTICE_SPIKE_CAP
    family.LATTICE_SPIKE_RELIEF_BOOST = float(boost)
    family.LATTICE_SPIKE_CAP = float(spike_cap)
    try:
        spike_delta, rep = family.fit_margin_spikes(
            mev["pmfs"], mev["outcomes"], mev["support"], "nfl", **NFL_SPIKE_KW)
        row = {
            "boost": float(boost), "spike_cap": float(spike_cap),
            "admitted": [s["bin"] for s in rep["spikes"] if s.get("admitted")],
            "verdicts": {s["bin"]: {k: s.get(k) for k in
                                    ("amplitude", "z", "admitted", "reason")}
                         for s in rep["spikes"]},
        }
        if spike_delta is None:
            row["reason"] = rep.get("reason")
            return row
        adjusted = np.clip(np.asarray(mev["pmfs"], float)
                           + np.asarray(spike_delta, float)[None, :], 0., None)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        gen = family.fit_lattice_correction(adjusted, mev["outcomes"],
                                            mev["support"], "nfl", "margin")
        row["generic"] = {k: gen.get(k) for k in
                          ("significant", "reason", "gate_z",
                           "residual_mae_before", "residual_mae_after",
                           "collateral_ok", "delta_max")}
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
            for b in (-7, -3, 0, 3, 7) if sup[0] <= b <= sup[-1]}
        mae, vsup, vres, vse = holdout_view(mev, composed)
        row["holdout_mae_after"] = mae
        row["holdout_key_bins"] = at_bins(None, vsup, vres, vse,
                                          (-14, -7, -3, 0, 3, 7, 14))
        row["band_outside"] = band_outside(vsup, vres, vse, MARGIN_PLOTTED)
        return row
    finally:
        family.LATTICE_SPIKE_RELIEF_BOOST = real_boost
        family.LATTICE_SPIKE_CAP = real_cap


def totals_variant(family, tev, boost):
    """One registered totals relief boost -> generic-gate verdict dict."""
    real = family.LATTICE_RELIEF
    family.LATTICE_RELIEF = dict(real)
    if boost > 1:
        family.LATTICE_RELIEF[("nfl", "totals")] = {
            "boost": float(boost),
            "quantile": real.get(("nba", "margin"), {}).get(
                "quantile", family.LATTICE_RELIEF_QUANTILE)}
    try:
        fit = family.fit_lattice_correction(
            tev["pmfs"], tev["outcomes"], tev["support"], "nfl", "totals")
        delta = np.asarray(fit.get("delta") or [0.0], float)
        row = {
            "boost": float(boost),
            "significant": bool(fit.get("significant")),
            "reason": fit.get("reason"),
            "gate_z": fit.get("gate_z"),
            "mae_before": round(fit.get("residual_mae_before", 0), 6),
            "mae_after": round(fit.get("residual_mae_after", 0), 6),
            "delta_max": fit.get("delta_max"),
            "collateral_ok": fit.get("collateral_ok"),
        }
        mae, vsup, vres, vse = holdout_view(tev, delta)
        row["holdout_mae_after"] = mae
        row["holdout_key_bins"] = at_bins(None, vsup, vres, vse,
                                          TOTALS_FOCUS)
        row["band_outside"] = band_outside(vsup, vres, vse, TOTALS_PLOTTED)
        return row
    finally:
        family.LATTICE_RELIEF = real


def margin_relief_variant(family, mev, boost, spike_boost=1.,
                          spike_cap=0.05):
    """Registered generic margin relief on spike-adjusted pmfs.

    Spikes stay at the boost1/cap setting that keeps every key-number
    admission; the GENERIC layer then runs with
    LATTICE_RELIEF[(nfl, margin)] = boost (quantile over mass-bearing bins,
    same math as the NBA registration) — the second layer's shrinkage is
    what leaves the +/-3 leftovers (spike 36% + generic ~40% of their own
    residuals).
    """
    real_relief = family.LATTICE_RELIEF
    real_boost = family.LATTICE_SPIKE_RELIEF_BOOST
    real_cap = family.LATTICE_SPIKE_CAP
    family.LATTICE_SPIKE_RELIEF_BOOST = float(spike_boost)
    family.LATTICE_SPIKE_CAP = float(spike_cap)
    family.LATTICE_RELIEF = dict(real_relief)
    if boost > 1:
        family.LATTICE_RELIEF[("nfl", "margin")] = {
            "boost": float(boost),
            "quantile": family.LATTICE_RELIEF_QUANTILE}
    try:
        spike_delta, rep = family.fit_margin_spikes(
            mev["pmfs"], mev["outcomes"], mev["support"], "nfl", **NFL_SPIKE_KW)
        row = {"margin_relief_boost": float(boost),
               "spike_boost": float(spike_boost),
               "spike_cap": float(spike_cap),
               "admitted": [s["bin"] for s in rep["spikes"]
                            if s.get("admitted")]}
        if spike_delta is None:
            row["reason"] = rep.get("reason")
            return row
        adjusted = np.clip(np.asarray(mev["pmfs"], float)
                           + np.asarray(spike_delta, float)[None, :], 0., None)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        gen = family.fit_lattice_correction(adjusted, mev["outcomes"],
                                            mev["support"], "nfl", "margin")
        row["generic"] = {k: gen.get(k) for k in
                          ("significant", "reason", "gate_z",
                           "residual_mae_before", "residual_mae_after",
                           "collateral_ok", "delta_max",
                           "collateral_max_excess")}
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
        mae, vsup, vres, vse = holdout_view(mev, composed)
        row["holdout_mae_after"] = mae
        row["holdout_key_bins"] = at_bins(
            None, vsup, vres, vse, (-14, -9, -7, -3, -2, -1, 0, 1, 2, 3, 7, 14))
        row["band_outside"] = band_outside(vsup, vres, vse, MARGIN_PLOTTED)
        return row
    finally:
        family.LATTICE_RELIEF = real_relief
        family.LATTICE_SPIKE_RELIEF_BOOST = real_boost
        family.LATTICE_SPIKE_CAP = real_cap


def main() -> int:
    sys.path.insert(0, str(BACKEND))
    dist = importlib.import_module("distributions")
    family = importlib.import_module("gaussian_family")

    markets = newest_markets()
    frame = pd.read_csv(markets, float_precision="round_trip")
    oof = frame[frame["kind"] == "oof"].copy()
    record: dict = {"artifact": markets.name, "n_oof": int(len(oof)),
                    "read_only": True,
                    "shipped_postlattice_key_bins": {
                        "margin": {"-3": 0.03132, "3": 0.02807,
                                   "-7": 0.01279, "7": 0.01134},
                        "totals": {"51": 0.01347, "37": 0.00967,
                                   "44": 0.00783, "42": -0.00801,
                                   "53": -0.00882}}}

    cache = ROOT / "run_diagnostics" / "nfl_keynumber_relief_ladder_ev.npz"
    if cache.exists():
        z = np.load(cache)
        ev = {k: {"pmfs": z[f"{k}_pmfs"], "outcomes": z[f"{k}_y"],
                  "support": z[f"{k}_support"]}
              for k in ("margin", "total")}
        record["evidence_from_cache"] = str(cache.name)
        record["walk_seconds_baseline"] = "cached"
    else:
        priced, bundle, secs = baseline_walk(dist, family, oof)
        priced = priced.reindex(oof.index)
        record["walk_seconds_baseline"] = secs
        ev = build_evidence(dist, family, priced, bundle)
        np.savez_compressed(
            cache,
            **{f"{k}_{name}": arr for k, v in ev.items()
               for name, arr in (("pmfs", v["pmfs"]),
                                 ("y", v["outcomes"]),
                                 ("support", v["support"]))})
    mev, tev = ev["margin"], ev["total"]
    record["evidence_n"] = {k: int(len(v["outcomes"]))
                            for k, v in ev.items()}

    # baseline holdout residuals (delta=None) for reference
    mae, vsup, vres, vse = holdout_view(mev, None)
    record["margin_holdout_baseline"] = {
        "mae": mae,
        "key_bins": at_bins(None, vsup, vres, vse, MARGIN_FOCUS),
        "band_outside": band_outside(vsup, vres, vse, MARGIN_PLOTTED)}
    mae, vsup, vres, vse = holdout_view(tev, None)
    record["totals_holdout_baseline"] = {
        "mae": mae,
        "key_bins": at_bins(None, vsup, vres, vse, TOTALS_FOCUS),
        "band_outside": band_outside(vsup, vres, vse, TOTALS_PLOTTED)}

    record["margin_ladder"] = [
        margin_variant(family, mev, boost, cap)
        for boost in (1., 2., 3., 4., 6.)
        for cap in (0.03, 0.05)]
    record["totals_ladder"] = [
        totals_variant(family, tev, boost)
        for boost in (1., 2., 3., 5.)]
    # (c) registered generic MARGIN relief on spike-adjusted pmfs
    # (spikes fixed at the admissions-preserving boost1 / cap .05).
    record["margin_relief_ladder"] = [
        margin_relief_variant(family, mev, boost)
        for boost in (1., 2., 3., 5., 8.)]

    dest = ROOT / "run_diagnostics" / "nfl_keynumber_relief_ladder.json"
    dest.write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record, indent=1, default=str))
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
