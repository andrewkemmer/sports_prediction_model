"""MLB margin one-run "spread spike" rebalance — evidence probe (2026-10-16).

Complaint (20261016 shipped artifact): the model's one-run-win spike at M=+1
sits BELOW the actual frequency (underfit to the residual: model 0.1433 vs
actual 0.1739) while the spike at M=-1 sits ABOVE it (overfit: model 0.1255
vs actual 0.1089). The ask: shift mass out of the neighbouring spread bins
(-2, +3) into the ±1 spikes, with every amount DYNAMICALLY DRIVEN BY THE
ACTUALS.

Phase A runs the real causal walk once (hooking fit_lattice_correction to
record every margin call) and saves the largest — final serving — margin
evidence (BASE equal-split pmfs + actuals) to /tmp/mlb_margin_evidence.pkl.
Phase B then:

  1. prints the train/held-out residual and the declared-direction verdict
     at every registered bin (recipient +1, donors -1/-2/+3),
  2. runs the SHIPPED mechanism — distributions.fit_one_run_spread_rebalance
     — on that evidence and prints its disclosure report plus the per-bin
     pooled residual before/after,
  3. sweeps the transfer amplitude to show the held-out gate ceiling (why the
     mechanism sizes itself off the admitted walk-off share and not off the
     full shrunk residual, which the unchanged generic bar refuses).

Run:  python run_diagnostics/mlb_pm1_spread_rebalance_probe.py [--recapture]
Log:  run_diagnostics/mlb_pm1_spread_rebalance.log
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlb-backend" / "backend"))
import distributions as dist          # noqa: E402
import gaussian_family as family      # noqa: E402

EVIDENCE = Path("/tmp/mlb_margin_evidence.pkl")
MARKETS = ROOT / "mlb-backend" / "data_delivery" / "run_engine_markets_20261016.csv"
SHARE = 0.6          # the served walk-off share on this artifact (gated q-hat)


def capture() -> dict:
    calls = []
    real = family.fit_lattice_correction

    def recording(*a, **k):
        out = real(*a, **k)
        try:
            if a[3] == "mlb" and a[4] == "margin":
                calls.append({"pmfs": np.asarray(a[0], float),
                              "outcomes": np.asarray(a[1], float),
                              "support": np.asarray(a[2])})
        except Exception:
            pass
        return out

    family.fit_lattice_correction = recording
    frame = pd.read_csv(MARKETS, float_precision="round_trip", low_memory=False)
    oof = frame[frame["kind"] == "oof"].copy()
    t0 = time.time()
    _priced, bundle = dist.calibrate_market_frame(oof)
    family.fit_lattice_correction = real
    print("walk_seconds:", round(time.time() - t0, 1), flush=True)
    final = max(calls, key=lambda c: len(c["outcomes"]))
    spec = (bundle.get("maps", {}).get("lattice", {}) or {}).get("margin", {}) or {}
    print("n_margin_calls:", len(calls), "final_n:", len(final["outcomes"]))
    print("serving margin spec:",
          json.dumps({k: v for k, v in spec.items() if k != "delta"},
                     default=str))
    with open(EVIDENCE, "wb") as f:
        pickle.dump(final, f)
    print("saved", EVIDENCE)
    return final


def report(ev: dict) -> None:
    pmfs, outcomes, support = ev["pmfs"], ev["outcomes"], ev["support"]
    index = {int(v): i for i, v in enumerate(support)}
    inside = (outcomes >= support[0]) & (outcomes <= support[-1])
    n_test = int(family.LATTICE_TEST_FRACTION * len(outcomes))
    train = np.flatnonzero(inside[:len(outcomes) - n_test])
    hold = np.flatnonzero(inside[len(outcomes) - n_test:])
    n_train, n_hold = max(len(train), 1), max(len(hold), 1)
    pbar = np.clip(pmfs[train].mean(axis=0), 0., 1.)
    obs = np.bincount(np.searchsorted(support, outcomes[train]),
                      minlength=len(support)).astype(float) / n_train
    obs_h = np.bincount(np.searchsorted(support, outcomes[hold]),
                        minlength=len(support)).astype(float) / n_hold
    pbar_h = np.clip(pmfs[hold].mean(axis=0), 0., 1.)
    shrunk = family.lattice_local_deltas(pbar, obs - pbar,
                                         np.sqrt(pbar * (1 - pbar) / n_train))
    print("\n--- registered bins: residual, stability, declared direction ---")
    print(f"{'bin':>4} {'train':>9} {'held-out':>9} {'shrunk':>9}")
    for v in (dist.ONE_RUN_REBALANCE_RECIPIENT, *dist.ONE_RUN_REBALANCE_DONORS):
        k = index[v]
        print(f"{v:>4} {obs[k] - pbar[k]:>+9.5f} "
              f"{obs_h[k] - pbar_h[k]:>+9.5f} {shrunk[k]:>+9.5f}"
              f"   {'recipient (up)' if v == dist.ONE_RUN_REBALANCE_RECIPIENT else 'donor (down, capacity-capped)'}")

    print("\n--- shipped mechanism: distributions.fit_one_run_spread_rebalance ---")
    delta, rep = dist.fit_one_run_spread_rebalance(pmfs, outcomes, support, SHARE)
    print(json.dumps(rep, indent=1))

    if delta is not None:
        d = np.asarray(delta)
        n_all = max(int(inside.sum()), 1)
        pooled = np.clip(pmfs[inside].mean(axis=0), 0., 1.)
        pooled_obs = np.bincount(np.searchsorted(support, outcomes[inside]),
                                 minlength=len(support)).astype(float) / n_all
        print(f"\n--- pooled residuals (BASE model -> after the rebalance) ---")
        print(f"{'bin':>4} {'actual':>8} {'model':>8} {'before':>9} {'after':>9}")
        for v in range(-4, 5):
            if v == 0:
                continue
            k = index[v]
            print(f"{v:>4} {pooled_obs[k]:>8.5f} {pooled[k]:>8.5f} "
                  f"{pooled_obs[k] - pooled[k]:>+9.5f} "
                  f"{pooled_obs[k] - pooled[k] - d[k]:>+9.5f}")
        print("  zero-sum:", round(float(d.sum()), 9),
              "MAE:", round(float(np.abs(pooled_obs - pooled).mean()), 6),
              "->", round(float(np.abs(pooled_obs - pooled - d).mean()), 6))

    print("\n--- gate ceiling: held-out z vs transfer amplitude (donor-funded) ---")
    print("  (why the mechanism is sized off the admitted share, not off the\n"
          "   full shrunk residual: the generic bar is unchanged and a deeper\n"
          "   proposal is refused by its own quadratic cost)")
    caps = {d: max(0., -float(shrunk[index[d]]))
            for d in dist.ONE_RUN_REBALANCE_DONORS}
    total = sum(caps.values())
    indices_hold = np.searchsorted(support, outcomes[hold])
    for amt in (.008, .0137, .0199, .025, float(abs(shrunk[index[1]]))):
        dv = np.zeros(len(support))
        dv[index[dist.ONE_RUN_REBALANCE_RECIPIENT]] = amt
        for d_, c in caps.items():
            dv[index[d_]] -= amt * c / total if total > 0. else 0.
        dv = family._zero_sum_mass_proportional(dv, pbar)
        dv = np.clip(dv, -family.LATTICE_DELTA_CAP, family.LATTICE_DELTA_CAP)
        dd = (-2. * dv[indices_hold] + 2. * (pmfs[hold] @ dv) + float(dv @ dv))
        se = float(dd.std(ddof=1) / np.sqrt(len(dd)))
        z = float(-dd.mean() / se) if se > 0. else 0.
        verdict = ("ADMIT" if z >= family.LATTICE_GATE_Z else
                   "structural-bar only" if z >= family.LATTICE_STRUCTURAL_Z
                   else "refuse")
        print(f"  amt {amt:.5f}  z {z:+.3f}  -> {verdict} "
              f"(generic {family.LATTICE_GATE_Z} / structural "
              f"{family.LATTICE_STRUCTURAL_Z})")


if __name__ == "__main__":
    if EVIDENCE.exists() and "--recapture" not in sys.argv:
        ev = pickle.load(open(EVIDENCE, "rb"))
        print("reusing", EVIDENCE, "(--recapture to refit the walk)")
    else:
        ev = capture()
    report(ev)
