"""Why the empty-net transfer barely moves the pooled chart, and which
gate/sizing policy actually admits (2026-10-17).

nhl16_origin_admissions_probe measured the shipped failure: 19/61 origins
admit, 29/99 pairs clear the structural bar, pooled +3 gap 0.0301 -> 0.0284
— and the admitting transfers are small (early, thin-evidence origins), so
the pooled chart can barely see them.

The held-out gate statistic for a zero-sum pair trial of size t is

    d_i = -2*trial[y_i] + 2*(pmf_i . trial) + |trial|^2
    mean(d) = -2*t*(gap_hold - t),   gap_hold = r_recipient - r_donor
    z(t) decreases LINEARLY in t      (se(d) is proportional to t)

so a DEEP transfer self-defeats its own gate: sizing on train evidence and
confirming on the held-out tail only works while t stays under the tail's
gap minus the bar's noise floor. This probe re-runs the causal walk ONCE
with the real estimator hooked, and inside the hook recomputes, from the
exact same per-origin evidence, four counterfactual policies:

  P0  current:   raw supply,  per-pair gates
  P1  shrunk supply (MLB rebalance 'capacity' precedent), per-pair gates
  P2  shrunk supply, ONE joint gate over the composed two-pair mechanism
      delta (the MLB single-composed-gate precedent; the preregistration
      declares ONE mechanism, empty_net_2_to_3)
  P3  raw supply, joint gate

For each policy it reports pairs/origins admitted, the sum of admitted
transfer mass weighted by each origin's priced rows (the pooled chart's
visibility), and the held-out gap distribution — then prints the REAL walk's
chart numbers (current policy) next to the delivered CSV.

Usage:  python run_diagnostics/nhl17_transfer_gate_policy_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nhl-backend" / "backend"))
import distributions as dist          # noqa: E402
import gaussian_family as family      # noqa: E402

ARTIFACT = max((ROOT / "nhl-backend" / "data_delivery").glob(
    "nhl_run_engine_markets_20*.csv"))   # newest delivered artifact
OUT = ROOT / "run_diagnostics" / "nhl17_transfer_gate_policy_probe.json"


def chart(priced: pd.DataFrame, label: str) -> None:
    d = priced.dropna(subset=["home_score", "away_score"]).copy()
    d["m"] = (d.home_score - d.away_score).astype(int)
    print(f"\n--- {label}: pooled chart numbers (n={len(d)}) ---")
    print(f"{'bin':>4} {'actual':>8} {'shipped':>8} {'gap':>9}")
    for v in (-4, -3, -2, -1, 1, 2, 3, 4):
        col = f"p_push_{v}" if v > 0 else f"p_push_m{-v}"
        if col not in d:
            continue
        o, m = float((d.m == v).mean()), float(d[col].mean())
        print(f"{v:>4} {o:>8.4f} {m:>8.4f} {o - m:>+9.4f}")


def _gate_z(trial, pmfs, hold, indices_hold, support):
    """Exact estimator gate statistic for an arbitrary trial vector."""
    d = (-2. * trial[indices_hold] + 2. * (pmfs[hold] @ trial)
         + float(trial @ trial))
    tse = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    z = float(-d.mean() / tse) if tse > 0 else 0.
    return z, float(d.mean())


def counterfactuals(pmfs, outcomes, support, pairs):
    """Recompute all four policies from the estimator's own split."""
    support = np.asarray(support)
    pmfs = np.asarray(pmfs, float)
    outcomes = np.asarray(outcomes, float)
    inside = ((outcomes >= support[0]) & (outcomes <= support[-1])
              & np.isfinite(outcomes))
    n_test = int(family.LATTICE_TEST_FRACTION * len(outcomes))
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    hold = np.arange(len(outcomes) - n_test, len(outcomes))
    hold = hold[inside[hold]]
    if len(train) < family.MIN_HISTORY or len(hold) < family.LATTICE_MIN_TEST:
        return None
    n_train = max(int(len(train)), 1)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    se = np.sqrt(pbar * (1. - pbar) / n_train)
    obs = np.bincount(np.searchsorted(support, outcomes[train]),
                      minlength=len(support)).astype(float)
    obs /= max(int(len(train)), 1)
    resid = obs - pbar
    # Held-out gap at the pair (what the gate actually has to beat).
    pbar_h = np.clip(pmfs[hold].mean(axis=0), 0., 1.)
    obs_h = np.bincount(np.searchsorted(support, outcomes[hold]),
                        minlength=len(support)).astype(float)
    obs_h /= max(int(len(hold)), 1)
    indices_hold = np.searchsorted(support, outcomes[hold])
    bar = family.LATTICE_STRUCTURAL_Z

    sized = {"raw": [], "shrunk": []}
    gaps = {}
    for donor, recipient in pairs:
        i2 = int(np.searchsorted(support, donor))
        i3 = int(np.searchsorted(support, recipient))
        if not (0 <= i2 < len(support) and 0 <= i3 < len(support)
                and int(support[i2]) == donor and int(support[i3]) == recipient):
            continue
        demand = float(family.lattice_local_deltas(
            dist.EMPTY_NET_DEMAND_RELIEF * pbar[i3:i3 + 1],
            resid[i3:i3 + 1], se[i3:i3 + 1])[0])
        supply_raw = float(np.clip(-resid[i2], 0., pbar[i2]))
        supply_shrunk = float(np.clip(
            -float(family.lattice_local_deltas(pbar[i2:i2 + 1],
                                               resid[i2:i2 + 1],
                                               se[i2:i2 + 1])[0]),
            0., pbar[i2]))
        gap = float((obs_h[i3] - pbar_h[i3]) - (obs_h[i2] - pbar_h[i2]))
        gaps[f"{donor}->{recipient}"] = round(gap, 5)
        for key, supply in (("raw", supply_raw), ("shrunk", supply_shrunk)):
            if demand > 0. and supply > 0.:
                t = float(min(demand, supply, dist.EMPTY_NET_SPIKE_CAP))
                trial = np.zeros(len(support))
                trial[i3], trial[i2] = t, -t
                z, mean = _gate_z(trial, pmfs, hold, indices_hold, support)
                sized[key].append({
                    "pair": f"{donor}->{recipient}", "t": t, "z": z,
                    "mean_neg": mean < 0.,
                    "ind_ok": mean < 0. and z >= bar,
                })
    out = {"n_evidence": int(len(outcomes)), "n_hold": int(len(hold)),
           "hold_gaps": gaps}
    for key in ("raw", "shrunk"):
        rows = sized[key]
        # Joint: compose every direction-passing pair's trial, one gate.
        joint_z = joint_mean = None
        if rows:
            trial = np.zeros(len(support))
            for r in rows:
                donor, recipient = (int(v) for v in r["pair"].split("->"))
                i3 = int(np.searchsorted(support, recipient))
                i2 = int(np.searchsorted(support, donor))
                trial[i3] += r["t"]
                trial[i2] -= r["t"]
            joint_z, joint_mean = _gate_z(trial, pmfs, hold, indices_hold,
                                          support)
        out[key] = {
            "pairs_ind_ok": sum(r["ind_ok"] for r in rows),
            "pairs_total": len(rows),
            "transfer_sum": round(sum(r["t"] for r in rows), 6),
            "joint_ok": bool(rows and joint_mean < 0 and joint_z >= bar),
            "joint_z": None if joint_z is None else round(joint_z, 3),
            "pair_rows": [{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in r.items()} for r in rows],
        }
    return out


def main() -> None:
    frame = pd.read_csv(ARTIFACT, float_precision="round_trip", low_memory=False)
    oof = frame[frame["kind"] == "oof"].copy()

    calls = []
    real = dist.fit_empty_net_transfer

    def hook(*a, **k):
        delta, rep = real(*a, **k)
        cf = None
        if a and a[1] is not None and len(a[1]):
            try:
                cf = counterfactuals(a[0], a[1], a[2],
                                     k.get("pairs", dist.EMPTY_NET_TRANSFER_PAIRS))
            except Exception as exc:                      # pragma: no cover
                cf = {"error": repr(exc)}
        pairs = rep.get("pairs") or []
        calls.append({
            "reason": rep.get("reason"),
            "delta_admitted": delta is not None,
            "pairs": [{"donor": p.get("donor"), "recipient": p.get("recipient"),
                       "demand": p.get("demand"), "supply": p.get("supply"),
                       "transfer": p.get("transfer"), "z": p.get("z"),
                       "admitted": p.get("admitted"),
                       "reason": p.get("reason")} for p in pairs],
            "cf": cf,
        })
        return delta, rep

    dist.fit_empty_net_transfer = hook
    t0 = time.time()
    priced, bundle = dist.calibrate_market_frame(oof)
    dist.fit_empty_net_transfer = real
    print("walk_seconds:", round(time.time() - t0, 1), flush=True)

    origins = bundle["origins"]
    # calls: one per origin's candidate fit + one final serving fit.
    print("\n--- baseline (current policy, real walk) ---")
    pair_rows = [p for c in calls for p in c["pairs"]]
    print("calls:", len(calls),
          "| delta admitted:", sum(c["delta_admitted"] for c in calls))
    zs = sorted(p["z"] for p in pair_rows if p.get("z") is not None)
    if zs:
        print("pair z >= 2.5:", sum(z >= 2.5 for z in zs), "/", len(zs),
              "| z deciles:", [round(z, 2) for z in zs[::max(1, len(zs) // 10)]])

    # Per-origin rows priced (for pooled visibility weighting).
    rows_scored = [o["n_scored"] for o in origins]
    with_any = with_transfer = 0
    for o in origins:
        lat = (o["maps"].get("lattice") or {}).get("margin") or {}
        if lat.get("delta"):
            with_any += 1
            if lat.get("empty_net_report", {}).get("significant"):
                with_transfer += 1
    print(f"origins shipping a margin delta: {with_any} / {len(origins)}"
          f" (transfer {with_transfer})")

    print("\n--- counterfactual policies (same evidence, exact gate math) ---")
    header = (f"{'policy':<28} {'pairs_ok':>9} {'origins':>8} "
              f"{'rows*transfer':>14}")
    print(header)
    policies = [
        ("P0 raw+per-pair (current)", "raw", "ind"),
        ("P1 shrunk+per-pair", "shrunk", "ind"),
        ("P2 shrunk+joint", "shrunk", "joint"),
        ("P3 raw+joint", "raw", "joint"),
    ]
    cf_calls = [c["cf"] for c in calls if c.get("cf") and "error" not in c["cf"]]
    # Match counterfactuals to origins: the serving fit is the LAST call.
    cf_origins = cf_calls[:len(origins)]
    summary = {}
    for label, key, mode in policies:
        pairs_ok = origins_ok = 0
        rows_x_t = 0.
        for i, cf in enumerate(cf_origins):
            p = cf[key]
            ok = (p["pairs_ind_ok"] if mode == "ind" else int(p["joint_ok"]))
            origins_ok += int(ok > 0)
            pairs_ok += p["pairs_ind_ok"] if mode == "ind" else ok
            if ok and i < len(rows_scored):
                rows_x_t += p["transfer_sum"] * rows_scored[i]
        n_rows = sum(rows_scored) or 1
        summary[label] = {"pairs_ok": pairs_ok, "origins_ok": origins_ok,
                          "pooled_shift_est": rows_x_t / n_rows}
        print(f"{label:<28} {pairs_ok:>9} {origins_ok:>8} "
              f"{rows_x_t / n_rows:>14.5f}")
    print("(pooled_shift_est = sum of admitted transfer x origin rows / total"
          " rows — the expected visible shift at each recipient/donor bin)")

    gaps = [cf["hold_gaps"] for cf in cf_origins if cf.get("hold_gaps")]
    if gaps:
        g1 = sorted(g.get("2->3", 0.) for g in gaps)
        g2 = sorted(g.get("-2->-3", 0.) for g in gaps)
        print("hold gap  2->3  deciles:",
              [round(g, 4) for g in g1[::max(1, len(g1) // 8)]])
        print("hold gap -2->-3 deciles:",
              [round(g, 4) for g in g2[::max(1, len(g2) // 8)]])

    chart(priced, "WALK (current policy, this run)")
    chart(oof, "DELIVERED CSV (newest)")

    OUT.write_text(json.dumps({"calls": calls, "summary": summary},
                              indent=1), encoding="utf-8")
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
