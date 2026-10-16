"""NFL dispersion probe (Step 3 — READ-ONLY, gated).

Question: do home/away residuals disperse differently? The margin family
prices from a shared variance structure; a side-specific spread the model
misses would show up as std(home_score - mu_h) != std(away_score - mu_a)
on strictly held-out (time-ordered tail) decided OOF rows.

Read-only: loads the latest committed markets artifact, does arithmetic,
writes a JSON verdict + log. NOTHING here touches the pipeline, admits a
parameter, or reprices a row — the verdict is the deliverable; any model
change would be a separate gated proposal.

Usage:  python run_diagnostics/nfl_dispersion_probe.py [--artifact PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_FRAC = 0.30  # time-ordered tail, mirrors LATTICE_TEST_FRACTION


def latest_artifact(pattern: str) -> Path:
    arts = sorted((REPO_ROOT / "nfl-backend" / "data_delivery").glob(pattern))
    if not arts:
        raise FileNotFoundError(f"no artifact matching {pattern}")
    return arts[-1]


def load_decided(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    dec = df[df.get("kind") == "oof"].copy()
    dec = dec[dec[["home_score", "away_score", "mu_h", "mu_a"]]
              .notna().all(axis=1)]
    dec = dec.sort_values(["gameday", "game_id"], kind="stable")
    return dec


def var_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    v = float(x.var(ddof=1)) if n > 1 else float("nan")
    # SE of the sample variance under normality: var * sqrt(2/(n-1)).
    se = float(v * np.sqrt(2.0 / max(n - 1, 1))) if n > 1 else float("nan")
    return {"n": int(n), "mean": round(float(x.mean()), 4),
            "std": round(float(np.sqrt(v)), 4) if v == v else None,
            "var": round(v, 4), "var_se": round(se, 4)}


def verdict(home: np.ndarray, away: np.ndarray) -> dict:
    sh, sa = var_stats(home), var_stats(away)
    vh, vah = sh["var"], sa["var"]
    se = float(np.sqrt(sh["var_se"] ** 2 + sa["var_se"] ** 2))
    z = float(abs(vh - vah) / se) if se > 0 else 0.0
    ratio = float(np.sqrt(vh / vah)) if vah > 0 else float("nan")
    return {"home": sh, "away": sa, "std_ratio_h_a": round(ratio, 4),
            "var_diff_z": round(z, 3),
            "signal": bool(z >= 3.0),
            "reading": ("side-specific dispersion SIGNAL (z>=3) — propose a "
                        "gated variance extension"
                        if z >= 3.0 else
                        "no side-specific dispersion signal — shared "
                        "variance stands")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    path = Path(args.artifact) if args.artifact else \
        latest_artifact("nfl_run_engine_markets_*.csv")
    dec = load_decided(path)
    res_h = (dec["home_score"] - dec["mu_h"]).to_numpy(float)
    res_a = (dec["away_score"] - dec["mu_a"]).to_numpy(float)
    n_hold = int(len(dec) * HOLDOUT_FRAC)
    cut = len(dec) - n_hold
    out = {"artifact": path.name, "n_decided": int(len(dec)),
           "n_holdout": int(n_hold),
           "full_sample": verdict(res_h, res_a),
           "held_out_tail": verdict(res_h[cut:], res_a[cut:]),
           "held_out_rule": "time-ordered tail 30%; signal bar z>=3 "
                            "(two-sided, same strictness as LATTICE_GATE_Z)"}
    print(json.dumps(out, indent=1))
    dest = REPO_ROOT / "run_diagnostics" / "nfl_dispersion.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
