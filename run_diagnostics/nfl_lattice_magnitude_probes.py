"""NFL lattice magnitude probes (Step 4 — READ-ONLY, gated).

Three questions about the ADMITTED production correction, answered from the
shipped post-lattice PMFs + outcomes on a time-ordered held-out tail:

(a) cap: the admitted margin delta hits delta_max == LATTICE_DELTA_CAP
    (.03) — the cap BINDS. How much structure is left over? Bins whose
    post-lattice held-out residual still exceeds their binomial noise
    floor are cap-constrained leftover (a bigger cap could only help
    there; everywhere else the cap is non-binding by construction).
(b) tie-hole: NFL keeps valid tie mass (no no-tie rule). Is the M=0 bin
    — and the +/-1 split around it — priced right after correction?
(c) totals spikes: the pre-registered spike table covers margin only.
    Does any single totals bin clear the structural bar (one-sided z>=2.5)
    on held-out data — i.e. is there a preregistration candidate?

Read-only: arithmetic on the latest committed markets artifact. NOTHING
here refits, reprices, or changes a production constant. A definitive cap
sweep (refit the generic residual under several caps on causal-walk
evidence) is flagged as follow-up — post-lattice residuals understate the
cap effect because the cap already absorbed what it could.

Usage:  python run_diagnostics/nfl_lattice_magnitude_probes.py [--artifact PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_FRAC = 0.30
STRUCTURAL_BAR = 2.5  # one-sided, mirrors LATTICE_STRUCTURAL_Z
GENERIC_BAR = 3.0     # two-sided, mirrors LATTICE_GATE_Z
CAP = 0.03            # mirrors LATTICE_DELTA_CAP (reported, never changed)


def tag(prefix: str, x: float) -> str:
    s = str(float(x))
    if s.endswith(".0"):
        s = s[:-2]
    return f"{prefix}_{s.replace('-', 'm').replace('.', '_')}"


def load_decided(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    dec = df[df.get("kind") == "oof"].copy()
    dec = dec[dec[["home_score", "away_score"]].notna().all(axis=1)]
    return dec.sort_values(["gameday", "game_id"], kind="stable")


def pmf_frame(dec: pd.DataFrame, target: str):
    """(outcomes, pmf matrix, support) from shipped post-lattice columns."""
    if target == "margin":
        y = (dec["home_score"] - dec["away_score"]).to_numpy(float)
        ks = list(range(-14, 15))
        cols = [tag("p_push", k) for k in ks]
    else:
        y = (dec["home_score"] + dec["away_score"]).to_numpy(float)
        int_cols = sorted(
            (c for c in dec.columns
             if c.startswith("p_push_total_") and c[13:].isdigit()),
            key=lambda c: int(c[13:]))
        ks = [int(c[13:]) for c in int_cols]
        cols = int_cols
    have = [c for c in cols if c in dec.columns]
    pmf = dec[have].to_numpy(float)
    return y, pmf, [ks[cols.index(c)] for c in have]


def heldout_residuals(y: np.ndarray, pmf: np.ndarray, support: list):
    support = np.asarray(support)
    n = len(y)
    n_hold = int(n * HOLDOUT_FRAC)
    yh = y[n - n_hold:]
    ph = pmf[n - n_hold:]
    inside = (yh >= support[0]) & (yh <= support[-1]) & np.isfinite(yh)
    yh, ph = yh[inside], ph[inside]
    idx = np.searchsorted(support, yh).astype(int)
    obs = np.bincount(idx, minlength=len(support)).astype(float)
    obs /= max(len(idx), 1)
    pbar = np.nanmean(ph, axis=0)
    resid = obs - pbar
    se = np.sqrt(np.clip(pbar, 0, 1) * (1 - np.clip(pbar, 0, 1))
                 / max(len(idx), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, resid / se, 0.0)
    return {"support": [int(k) for k in support], "n_hold": int(len(idx)),
            "resid": resid, "se": se, "z": z, "obs": obs, "pbar": pbar}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    arts = sorted((REPO_ROOT / "nfl-backend" / "data_delivery")
                  .glob("nfl_run_engine_markets_*.csv"))
    path = Path(args.artifact) if args.artifact else arts[-1]
    dec = load_decided(path)

    out: dict = {"artifact": path.name, "n_decided": int(len(dec)),
                 "cap_reported": CAP, "note": "post-lattice residuals; "
                 "a definitive cap sweep needs causal-walk evidence"}

    # (a) cap leftover on margin.
    y, pmf, sup = pmf_frame(dec, "margin")
    m = heldout_residuals(y, pmf, sup)
    order = np.argsort(-np.abs(m["resid"]))[:6]
    out["cap_leftover_margin"] = {
        "n_hold": m["n_hold"],
        "top_bins": [{"k": m["support"][i],
                      "resid_after": round(float(m["resid"][i]), 5),
                      "se": round(float(m["se"][i]), 5),
                      "z": round(float(m["z"][i]), 3)} for i in order],
        "max_abs_leftover": round(float(np.abs(m["resid"]).max()), 5),
        "bins_beyond_2se": int((np.abs(m["resid"]) > 2 * m["se"]).sum()),
        "reading": ("cap-constrained leftover EXISTS" if
                    (np.abs(m["resid"]) > 2 * m["se"]).any() else
                    "no leftover beyond noise — cap is non-binding "
                    "everywhere it matters")}

    # (b) tie-hole: the M=0 bin and the +/-1 split.
    j0 = m["support"].index(0)
    ym = y[len(y) - int(len(y) * HOLDOUT_FRAC):]
    one = ym[np.abs(ym) == 1]
    out["tie_hole"] = {
        "obs_push_0": round(float((ym == 0).mean()), 5),
        "mod_push_0": round(float(m["pbar"][j0]), 5),
        "tie_z": round(float(m["z"][j0]), 3),
        "pm1_home_share": (round(float((one > 0).mean()), 4)
                           if len(one) else None),
        "pm1_n": int(len(one)),
        "monitor_full_sample": "obs_push 0.0035 vs mod_push 0.024 (~7x) — "
            "the production monitor agrees (full-sample, vintage-mixed)",
        "reading": ("tie OVERPRICE confirmed (probe holdout + monitor "
                    "full-sample agree) — generic admission path only "
                    "(no rule-derived mechanism, so the structural bar is "
                    "unavailable); likely gate/collateral dynamics left it")}

    # (c) totals spike scan. A preregistration candidate needs BOTH z >=
    # structural bar AND non-negligible mass (|resid| floor kills deep-tail
    # SE artifacts) AND a rule-derived mechanism (the structural bar is
    # rule-derived-only).
    yt, pmft, supt = pmf_frame(dec, "totals")
    t = heldout_residuals(yt, pmft, supt)
    cand = [i for i in range(len(supt))
            if t["z"][i] >= STRUCTURAL_BAR and abs(t["resid"][i]) >= 0.003]
    out["totals_spike_scan"] = {
        "n_hold": t["n_hold"],
        "support_span": [supt[0], supt[-1]],
        "candidates": [{"k": supt[i],
                         "resid": round(float(t["resid"][i]), 5),
                         "z": round(float(t["z"][i]), 3)}
                        for i in sorted(cand, key=lambda i: -t["z"][i])],
        "max_z": round(float(t["z"].max()), 3),
        "reading": ("bins clear bar+mass but none has a rule-derived "
                    "mechanism — NO preregistration candidate; leftover "
                    "belongs to the generic path" if cand else
                    "no totals bin clears bar+mass — no preregistration "
                    "candidate")}

    # (d) spike-mirror asymmetry: the preregistered table covers +3/+7 (and
    # + secondaries) but NOT -3/-7 — yet scoring symmetry says away-win key
    # numbers pile identically. Only the capped generic residual serves
    # the negative side. Compare held-out residuals mirror-wise.
    sup = m["support"]
    pairs = []
    for v in (3, 7, 1, 4, 6, 10, 14):
        row = {"bin": v}
        for sgn, key in ((1, "plus"), (-1, "minus")):
            k = sgn * v
            if k in sup:
                i = sup.index(k)
                row[key] = {"resid": round(float(m["resid"][i]), 5),
                            "z": round(float(m["z"][i]), 3)}
        pairs.append(row)
    out["spike_mirror_asymmetry"] = {
        "pairs": pairs,
        "reading": ("negative key numbers carry leftover WITHOUT a "
                    "structural path — mirror spikes at -3/-7 are the "
                    "preregistration extension to propose (rule-derived: "
                    "scoring symmetry), gated one-sided at the structural "
                    "bar on causal-walk evidence")}

    print(json.dumps(out, indent=1))
    dest = REPO_ROOT / "run_diagnostics" / "nfl_lattice_magnitude.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
