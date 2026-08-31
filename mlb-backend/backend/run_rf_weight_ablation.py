"""RandomForest blend-weight ablation for the moneyline ensemble.

Question: is the RandomForest member's weight in the blend right? Production
blends with ADAPTIVE weights earned from pooled OOF AUC (softmax + floor/cap
projection; RF's static prior is 0.10 in ENSEMBLE_WEIGHTS). This ablation
measures (a) what RF actually earns adaptively, and (b) whether forcing RF to
a fixed weight — including ZERO (drop RF from the blend) — changes the blend
for better or worse.

Protocol (identical to run_weight_cap_ablation.py — the shared blend-gate
pattern): ONE fold loop trains the 5-member ensemble per fold and caches each
fold's member OOF probabilities (checkpointed under the temp dir, resumable;
shared by every arm — apples-to-apples); each arm's blend = sum(w * p_member)
for pooled OOF AND sealed holdout, with prequential Platt calibrated twins.
On top of the standard single sealed holdout, three sealed 21-day windows are
scored per arm (members refit once per window) so no single window drives the
verdict.

Arms:
  adaptive — production: training.compute_adaptive_weights (pooled-OOF AUC
             softmax + floor 0.05 + cap 0.45). Reports the RF weight
             production actually earns.
  prior    — static ENSEMBLE_WEIGHTS (RF = 0.10), the pre-adaptive fallback.
  rf_zero  — RF weight 0 (member dropped from the blend), others renormalized.
  rf_05 / rf_20 / rf_35 — RF pinned at 0.05 / 0.20 / 0.35, others
             renormalized proportionally from the priors.

Measured per arm: pooled OOF blend (raw + calibrated) and sealed-holdout
blend (raw + calibrated) on logloss/AUC/ECE, plus per-window sealed metrics.

DECISION RULE: change RF's blend weight ONLY if a fixed-weight arm beats the
production adaptive arm on the main sealed holdout logloss AND AUC without
degrading sealed ECE and without losing pooled OOF logloss. Otherwise keep
production adaptive weights.

Record: data_delivery/rf_weight_ablation_<date>.json. COMMITS NOTHING.
    python run_rf_weight_ablation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

# Windows stub for the POSIX-only `resource` module: features.py imports it at
# module level and only ever calls getrusage().ru_maxrss (peak-memory logging).
try:
    import resource  # noqa: F401
except ImportError:  # pragma: no cover - Windows only
    import types as _types
    _ru = _types.SimpleNamespace(ru_maxrss=0.0)
    _res = _types.ModuleType("resource")
    _res.RUSAGE_SELF = 0
    _res.getrusage = lambda _who: _ru
    sys.modules["resource"] = _res

import training  # noqa: E402
import run_margin_ablation as rma  # noqa: E402
import run_weight_cap_ablation as rwca  # noqa: E402
import run_stack_ablation as rsa  # noqa: E402
from config import (DATA_DELIVERY_DIR, RANDOM_SEED,  # noqa: E402
                    ENSEMBLE_WEIGHTS)
from calibration import MIN_OOF_FOR_FIT  # noqa: E402

EPS = 1e-7


def fixed_rf_weights(rf_weight: float) -> dict[str, float]:
    """Pin RF at rf_weight; renormalize the other priors to fill the rest."""
    others = {n: float(w) for n, w in ENSEMBLE_WEIGHTS.items() if n != "randomforest"}
    tot_other = sum(others.values())
    w = {n: (1.0 - rf_weight) * v / tot_other for n, v in others.items()}
    w["randomforest"] = rf_weight
    # round like production (sum to exactly 1.0)
    rounded = {n: round(v, 4) for n, v in w.items()}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(rounded, key=lambda n: w[n])
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def window_sealed(games: pd.DataFrame, end: pd.Timestamp, holdout_days: int,
                  arms: dict[str, dict[str, float]],
                  members: list[str]) -> dict[str, dict]:
    """Per-window sealed blend for every arm (members refit once per window)."""
    start = end - pd.Timedelta(days=holdout_days)
    hold_k = games[(games["game_date"] > start)
                   & (games["game_date"] <= end)].reset_index(drop=True)
    tune_k = games[games["game_date"] <= start].reset_index(drop=True)
    training._LAST_ADAPTIVE_WEIGHTS.clear()
    models, _ = training.train_moneyline_ensemble(tune_k)
    _blend, member_probs, _wts = training.ensemble_predict(models, hold_k)
    yh = hold_k["home_win"].values.astype(float)
    out: dict[str, dict] = {"n_hold": int(len(hold_k)),
                            "hold_start": str(start.date()),
                            "hold_end": str(end.date())}
    for label, w in arms.items():
        probs = np.clip(np.column_stack(
            [np.asarray(member_probs[n], dtype=float) for n in members])
            @ np.array([w[n] for n in members]), EPS, 1 - EPS)
        out[label] = training.compute_metrics(yh, probs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--windows", type=int, default=3,
                    help="additional sealed 21-day windows beyond the main "
                         "cached holdout")
    args = ap.parse_args()

    sha = rma.head_sha()
    (games, tune_enriched, hold_df, folds, _m, hold_margins, _rounds, _u) = \
        rma.prepare_data(holdout_days=21)
    hold_enriched = rma.attach(hold_df, hold_margins)

    print(f"commit={sha[:12]} games={len(games)} tuning={len(tune_enriched)} "
          f"holdout={len(hold_df)} folds={len(folds)} seed={RANDOM_SEED}",
          flush=True)
    print("running ONE fold loop (5-member ensemble per fold, checkpointed) "
          "...", flush=True)
    cache = rwca.cached_fold_loop(folds, tune_enriched, hold_enriched, sha)
    members = cache["members"]
    print(f"members: {members}", flush=True)

    y_pooled = np.concatenate([r["y"] for r in cache["fold_recs"]])
    pooled_members = {n: np.concatenate([r["members"][n]
                                         for r in cache["fold_recs"]]).tolist()
                      for n in members}

    # --- what production actually earns ---
    adaptive_w = training.compute_adaptive_weights(pooled_members, y_pooled)
    raw_w = rwca.raw_softmax_weights(pooled_members, y_pooled)
    member_metrics = {}
    for n in members:
        m = training.compute_metrics(y_pooled, np.asarray(pooled_members[n]))
        member_metrics[n] = {"logloss": m["logloss"], "auc": m["auc"],
                             "ece": m["ece"]}

    arms: dict[str, dict[str, float]] = {
        "adaptive": adaptive_w,
        "prior": {n: round(float(ENSEMBLE_WEIGHTS[n]), 4) for n in members},
        "rf_zero": fixed_rf_weights(0.0),
        "rf_05": fixed_rf_weights(0.05),
        "rf_20": fixed_rf_weights(0.20),
        "rf_35": fixed_rf_weights(0.35),
    }

    print("\n[member pooled OOF] " + "  ".join(
        f"{n}: ll={member_metrics[n]['logloss']:.4f} "
        f"auc={member_metrics[n]['auc']:.4f}" for n in members))
    print(f"[adaptive weights] {adaptive_w}")
    print(f"[raw softmax pref] { {n: round(v, 4) for n, v in raw_w.items()} }")

    arm_results: dict[str, dict] = {}
    for label, w in arms.items():
        oof_blend, oof_blend_cal, oof_y = rsa.prequential_blend(
            cache["fold_recs"], members, "adaptive", adaptive_w=w)
        blend_h, cal_h = rsa.holdout_blend(
            cache["X_hold"], cache["y_hold"], members, "adaptive",
            None, w, cache["fold_recs"], oof_blend, oof_y)
        pooled_raw = training.compute_metrics(
            np.asarray(oof_y), np.asarray(oof_blend, dtype=float))
        pooled_cal = training.compute_metrics(
            np.asarray(oof_y), np.asarray(oof_blend_cal, dtype=float))
        hold_raw = training.compute_metrics(
            cache["y_hold"], np.asarray(blend_h, dtype=float))
        hold_cal = training.compute_metrics(
            cache["y_hold"], np.asarray(cal_h, dtype=float))
        arm_results[label] = {
            "weights": {k: round(float(v), 4) for k, v in w.items()},
            "pooled": {"blend": pooled_raw, "blend_calibrated": pooled_cal},
            "holdout": {"blend": hold_raw, "blend_calibrated": hold_cal},
        }
        print(f"  {label:<8} rf_w={w.get('randomforest', 0.0):.2f} | "
              f"pooled ll={pooled_raw['logloss']:.4f} "
              f"auc={pooled_raw['auc']:.4f} ece={pooled_raw['ece']:.4f} | "
              f"sealed ll={hold_raw['logloss']:.4f} "
              f"auc={hold_raw['auc']:.4f} ece={hold_raw['ece']:.4f} "
              f"(cal {hold_cal['ece']:.4f})", flush=True)

    # --- additional sealed windows (members refit once per window) ---
    windows: dict[str, dict] = {}
    maxd = games["game_date"].max()
    for k in range(args.windows):
        end = maxd - pd.Timedelta(days=k * 21)
        windows[f"w{k}"] = window_sealed(games, end, 21, arms, members)
        row = windows[f"w{k}"]
        print(f"  w{k} [{row['hold_start']}..{row['hold_end']}] "
              f"n={row['n_hold']} | "
              + "  ".join(f"{l}: {row[l]['auc']:.4f}" for l in arms), flush=True)

    # --- decision rule ---
    a_h = arm_results["adaptive"]["holdout"]["blend"]
    a_p = arm_results["adaptive"]["pooled"]["blend"]
    best: str | None = None
    for label, res in arm_results.items():
        if label == "adaptive":
            continue
        h = res["holdout"]["blend"]
        p = res["pooled"]["blend"]
        beats = (h["logloss"] < a_h["logloss"] and h["auc"] > a_h["auc"]
                 and h["ece"] <= a_h["ece"]
                 and p["logloss"] <= a_p["logloss"] + 0.0005)
        if beats and (best is None or
                      h["auc"] > arm_results[best]["holdout"]["blend"]["auc"]):
            best = label
    verdict = (f"CHANGE RF WEIGHT -> {best} "
               f"({arm_results[best]['weights']['randomforest']:.2f})"
               if best else "KEEP PRODUCTION ADAPTIVE WEIGHTS")

    target = pd.Timestamp.now().date().isoformat().replace("-", "")
    record = {
        "schema": "rf-weight-ablation/v1",
        "commit_sha": sha,
        "date": target,
        "data": "data_delivery/game_level_features.csv",
        "seed": int(RANDOM_SEED),
        "members": members,
        "folds_executed": len(cache["fold_recs"]),
        "member_pooled_oof": member_metrics,
        "adaptive_weights": {k: round(float(v), 4) for k, v in adaptive_w.items()},
        "raw_softmax_preference": {n: round(float(v), 4)
                                   for n, v in raw_w.items()},
        "arms": arm_results,
        "windows": windows,
        "gate": {"verdict": verdict,
                 "rule": ("fixed-weight arm must beat adaptive on main sealed "
                          "logloss AND AUC without ECE degradation and without "
                          "losing pooled OOF logloss (>+0.0005)"),
                 "adaptive_holdout": a_h, "adaptive_pooled": a_p},
    }
    out = args.out or (DATA_DELIVERY_DIR / f"rf_weight_ablation_{target}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nGATE: {verdict}")
    print(f"record -> {out}")


if __name__ == "__main__":
    main()
