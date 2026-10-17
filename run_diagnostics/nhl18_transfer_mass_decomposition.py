"""Where does the transfer mass go? Decompose pooled shipped p_push into
base + stored composed delta, and the stored delta into transfer vs generic
components (2026-10-17).

The joint gate admits 30/61 origins with rows*transfer = 0.0132, yet the
pooled chart moves ~0.001 at the recipient bins. This script runs the walk
once and reports, per bin (-4..+4, +/-1 as collateral watch):

  fitted_transfer = sum over origins of rows * (transfer delta at bin)
  fitted_generic  = sum over origins of rows * (generic delta at bin)
  stored          = sum over origins of rows * (COMPOSED stored delta at bin)
  shipped         = pooled mean p_push (what the chart draws)
  implied_base    = shipped - stored   (should be smooth in the teeth)

If fitted_transfer >> stored at the recipient bins, the compose/clip/round
or the generic component is cancelling it; if stored ~= fitted but
shipped - stored doesn't match a prior base run, pricing is not consuming
the stored delta.

Usage:  python run_diagnostics/nhl18_transfer_mass_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nhl-backend" / "backend"))
import distributions as dist          # noqa: E402

ARTIFACT = max((ROOT / "nhl-backend" / "data_delivery").glob(
    "nhl_run_engine_markets_20*.csv"))   # newest delivered artifact


def main() -> None:
    frame = pd.read_csv(ARTIFACT, float_precision="round_trip", low_memory=False)
    oof = frame[frame["kind"] == "oof"].copy()
    priced, bundle = dist.calibrate_market_frame(oof)
    origins = bundle["origins"]

    # Pooled shipped pmf at integer margin bins from p_push columns.
    d = priced.dropna(subset=["home_score", "away_score"]).copy()
    d["m"] = (d.home_score - d.away_score).astype(int)
    n_total = len(d)

    bins = [-4, -3, -2, -1, 1, 2, 3, 4]
    fitted_transfer = {v: 0. for v in bins}
    fitted_generic = {v: 0. for v in bins}
    stored = {v: 0. for v in bins}
    n_admit = 0
    detail = []
    for o in origins:
        rows = int(o["n_scored"])
        lat = (o["maps"].get("lattice") or {}).get("margin") or {}
        rep = lat.get("empty_net_report") or {}
        sup = np.asarray(lat.get("support") or [], dtype=int)
        delta = lat.get("delta")
        if delta is None:
            continue
        n_admit += 1
        delta = np.asarray(delta, float)
        if not len(sup) or len(sup) != 2:
            continue
        support = np.arange(int(sup[0]), int(sup[1]) + 1)
        if len(support) != len(delta):
            detail.append({"warn": "support_len_mismatch"})
            continue
        # Rebuild the transfer component from the report's admitted pairs.
        tcomp = np.zeros(len(support))
        for p in rep.get("pairs", []):
            if not p.get("admitted"):
                continue
            t = float(p.get("transfer") or 0.)
            for key, sign in (("recipient", 1.), ("donor", -1.)):
                j = int(np.searchsorted(support, int(p[key])))
                if 0 <= j < len(support) and int(support[j]) == int(p[key]):
                    tcomp[j] += sign * t
        # The compose re-zero-sums mass-proportionally; apply the same
        # post pipeline to the transfer component alone for comparability
        # (clip/eps/round at the composed caps is already reflected in
        # `stored`; here we want the raw fitted component).
        for v in bins:
            j = int(np.searchsorted(support, v))
            if 0 <= j < len(support) and int(support[j]) == v:
                fitted_transfer[v] += rows * float(tcomp[j])
                fitted_generic[v] += rows * (float(delta[j]) - float(tcomp[j]))
                stored[v] += rows * float(delta[j])
        detail.append({"rows": rows,
                       "transfer_at": {v: round(float(tcomp[int(np.searchsorted(support, v))])
                                                if 0 <= int(np.searchsorted(support, v)) < len(support)
                                                and int(support[int(np.searchsorted(support, v))]) == v else 0., 5)
                                       for v in (2, 3, -2, -3)},
                       "stored_at": {v: round(float(delta[int(np.searchsorted(support, v))])
                                              if 0 <= int(np.searchsorted(support, v)) < len(support)
                                              and int(support[int(np.searchsorted(support, v))]) == v else 0., 5)
                                     for v in (2, 3, -2, -3)},
                       "significant": lat.get("significant"),
                       "gate": (rep.get("gate") or {}).get("z")})

    print(f"origins with a stored margin delta: {n_admit} / {len(origins)}")
    print(f"\n{'bin':>4} {'shipped':>8} {'stored':>9} {'xfer':>9} {'generic':>9}"
          f" {'implied_base':>13} {'actual':>8}")
    for v in bins:
        col = f"p_push_{v}" if v > 0 else f"p_push_m{-v}"
        shipped = float(d[col].mean()) if col in d else float("nan")
        st = stored[v] / n_total
        actual = float((d["m"] == v).mean())
        print(f"{v:>4} {shipped:>8.4f} {st:>9.5f} {fitted_transfer[v] / n_total:>9.5f}"
              f" {fitted_generic[v] / n_total:>9.5f} {shipped - st:>13.4f}"
              f" {actual:>8.4f}")

    # Sanity: do the stored deltas reproduce the shipped columns? Compare
    # this run's p_push with a base (lattice-disabled) price of the same
    # rows using each origin's maps minus lattice.
    print("\nper-origin detail at mechanism bins (first 12 admits):")
    for row in [r for r in detail if "transfer_at" in r][:12]:
        print("  ", row)


if __name__ == "__main__":
    main()
