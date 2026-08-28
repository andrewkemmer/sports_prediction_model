"""Constrained stacking meta-learner ablation — STACK vs ADAPTIVE blend.

Trains an L2-regularized logistic regression meta-learner on the FIVE ensemble
members' pooled OOF probabilities (the same 45-fold walk-forward geometry as
production; leak-free — every member probability comes from a model trained
strictly before that game, so the blend weights are never fit on their own
predictions) and gates it against the CURRENT adaptive renormalization
(compute_adaptive_weights: pooled-OOF AUC softmax + floor/cap projection).

Design:
- Data: committed data_delivery/game_level_features.csv + production feature
  matrix (65 cols incl. run_margin_diff); run_margin_ablation.prepare_data
  provides the leak-free 45-fold geometry + sealed 284 holdout
  ([2026-08-05 .. 2026-08-25]) and the enriched frames.
- Fold cache: ONE fold loop trains the 5-member ensemble per fold and saves
  each fold's member OOF probabilities — shared by every arm (apples-to-
  apples; the fold loop is the expensive part and runs exactly once).
- ADAPTIVE arm (production baseline): weights from compute_adaptive_weights
  on the pooled OOF member probs; blend = sum(w*p) for pooled AND holdout.
- STACK arms (3 variants, scipy SLSQP — sklearn in this env lacks
  positive=True):
    unconstrained  : L2 logistic, intercept, no weight bounds
    nonneg         : L2 logistic, w >= 0, intercept allowed
    nonneg_sum1    : w >= 0, sum(w) = 1, no intercept (pure convex blend)
  Pooled OOF scoring is PREQUENTIAL: fold k's stack is fit on folds < k only,
  so every pooled stack point is out-of-sample for the stack (the same
  discipline as the prequential Platt twins). Sealed holdout: stack fit on
  ALL pooled OOF (strictly pre-holdout) and applied to the holdout members.
- Calibrated twins: prequential Platt on each arm's blend (fold k fit on
  prior folds' pairs); holdout calibrated = Platt fit on all pooled OOF pairs
  (never holdout information), matching run_margin_ablation exactly.
- Gate (task rule): ADOPT only if the best stack beats ADAPTIVE on SEALED
  logloss AND AUC without degrading sealed ECE, and pooled OOF logloss is not
  lost.

Emits data_delivery/stack_ablation_<date>.json (date-stamped). COMMITS
NOTHING.

Usage:
    python run_stack_ablation.py
    python run_stack_ablation.py --smoke   # 3 folds -> /tmp, gate skipped
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
import run_margin_ablation as rma  # noqa: E402
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402
from config import DATA_DELIVERY_DIR, RANDOM_SEED  # noqa: E402

EPS = 1e-7
# L2 strength on the STACK weights. Member OOF probs are highly correlated
# and near-constant (~0.45-0.55), so the logloss surface is flat in weight
# space and an unstandardized L2 fit collapses weights toward 0 (verified:
# L2=1.0 on raw probs -> intercept-only constant blend, all variants
# identical). The stack therefore STANDARDIZES member probs with FIT-TIME
# stats (leak-free: mu/sd from the fitting data only) and uses L2=0.01,
# which recovers the MLE-shaped weights (synthetic check matches sklearn).
L2 = 0.01
MIN_STACK_FIT = 300  # need at least this many prior OOF pairs to fit the stack

STACK_VARIANTS = ["unconstrained", "nonneg", "nonneg_sum1"]


def _log_loss_stable(y: np.ndarray, z: np.ndarray) -> float:
    """Mean binary log-loss from logits z (stable via logaddexp)."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    ll = np.where(y == 1, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))
    return float(np.mean(ll))


def fit_stack(X: np.ndarray, y: np.ndarray, variant: str,
              l2: float = L2) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit the constrained logistic meta-learner on member probs X -> y.

    Member probs are STANDARDIZED with fit-time stats (mu, sd) so the L2
    penalty acts on a unit-variance scale (raw near-constant probs collapse
    the fit — see module docstring). Returns (w, b, mu, sd) for predict-time
    standardization. variant:
      unconstrained  — L2, intercept, no bounds
      nonneg         — L2, w >= 0, intercept allowed
      nonneg_sum1    — w >= 0, sum(w)=1, NO intercept (pure convex blend)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    def _obj(vars_):
        if variant == "nonneg_sum1":
            w = vars_
            z = Xs @ w
        else:
            w, b = vars_[:m], vars_[m]
            z = Xs @ w + b
        pen = 0.5 * l2 * float(np.dot(w, w))
        return _log_loss_stable(y, z) + pen

    if variant == "nonneg_sum1":
        x0 = np.full(m, 1.0 / m)
        bounds = [(0.0, None)] * m
        cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    elif variant == "nonneg":
        x0 = np.zeros(m + 1)
        bounds = [(0.0, None)] * m + [(None, None)]
        cons = ()
    else:  # unconstrained
        x0 = np.zeros(m + 1)
        bounds = None
        cons = ()

    res = minimize(_obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 3000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"stack fit failed ({variant}): {res.message}")
    if variant == "nonneg_sum1":
        return np.clip(res.x, 0.0, None), 0.0, mu, sd
    return res.x[:m], float(res.x[m]), mu, sd


def stack_predict(X: np.ndarray, w: np.ndarray, b: float,
                  mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xs = (np.asarray(X, dtype=float) - mu) / sd
    return np.clip(expit(Xs @ w + b), EPS, 1 - EPS)


def _equal_weights(m: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Equal raw-prob blend (identity standardization): mean of member probs."""
    return np.full(m, 1.0 / m), 0.0, np.zeros(m), np.ones(m)


def build_member_matrix(recs: list[dict], names: list[str]) -> np.ndarray:
    return np.column_stack([recs["members"][name] for name in names])


def run_fold_cache(folds, tune_enriched, hold_enriched) -> dict:
    """One fold loop: per-fold member OOF probs + sealed-holdout member probs."""
    fold_recs = []
    names: list[str] | None = None
    for split in folds:
        train, val = split["train_games"], split["val_games"]
        try:
            models, _ = training.train_moneyline_ensemble(train, val)
        except Exception as e:
            print(f"  fold {split['fold_idx']} failed: {e}", flush=True)
            continue
        _blend, member_probs, _wts = training.ensemble_predict(models, val)
        if names is None:
            names = sorted(member_probs)
        fold_recs.append({
            "fold_idx": int(split["fold_idx"]),
            "y": val["home_win"].values.astype(float),
            "members": {n: np.asarray(member_probs[n], dtype=float)
                        for n in names if n in member_probs},
        })
    if names is None or len(names) < 3:
        raise RuntimeError(f"too few members trained: {names}")
    # Member order consistent across folds (intersection present everywhere).
    keep = [n for n in names if all(n in r["members"] for r in fold_recs)]
    if len(keep) < 3:
        raise RuntimeError(f"members missing across folds: {keep}")
    for r in fold_recs:
        r["members"] = {n: r["members"][n] for n in keep}

    models_hold, _ = training.train_moneyline_ensemble(tune_enriched)
    _blend_h, member_hold, _wts_h = training.ensemble_predict(models_hold,
                                                              hold_enriched)
    y_hold = hold_enriched["home_win"].values.astype(float)
    return {
        "members": keep,
        "fold_recs": fold_recs,
        "X_hold": np.column_stack([member_hold[n] for n in keep]),
        "y_hold": y_hold,
    }


def prequential_blend(fold_recs, members, arm: str,
                      stack_variant: str | None = None,
                      adaptive_w: dict | None = None) -> tuple[list, list, list]:
    """Pooled-OOF blend for an arm, PREQUENTIAL for the stack.

    Returns (oof_blend, oof_blend_cal, oof_y) aligned in fold order. Adaptive
    uses the pooled-earned weights directly (production semantics); stack fits
    fold k on folds < k only. Both get prequential Platt calibrated twins.
    """
    oof_y: list[float] = []
    oof_blend: list[float] = []
    oof_blend_cal: list[float] = []
    for k, rec in enumerate(fold_recs):
        X = np.column_stack([rec["members"][n] for n in members])
        y = rec["y"]
        if arm == "adaptive":
            w = np.array([adaptive_w[n] for n in members])
            b = 0.0
            blend = np.clip(X @ w, EPS, 1 - EPS)
        else:  # stack, prequential
            if k == 0 or sum(len(r["y"]) for r in fold_recs[:k]) < MIN_STACK_FIT:
                w, b, mu, sd = _equal_weights(len(members))
            else:
                X_prior = build_member_matrix(
                    {"members": {n: np.concatenate(
                        [r["members"][n] for r in fold_recs[:k]]) for n in members}},
                    members)
                y_prior = np.concatenate([r["y"] for r in fold_recs[:k]])
                w, b, mu, sd = fit_stack(X_prior, y_prior, stack_variant)
            blend = stack_predict(X, w, b, mu, sd)
        oof_y.extend(y.tolist())
        oof_blend.extend(blend.tolist())
        fold_cal = None
        if len(oof_y) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        oof_blend_cal.extend(
            np.asarray(apply_platt(np.asarray(blend, dtype=float), fold_cal),
                       dtype=float).tolist())
    return oof_blend, oof_blend_cal, oof_y


def holdout_blend(X_hold, y_hold, members, arm: str, stack_variant,
                  adaptive_w, fold_recs, oof_blend, oof_y) -> tuple[list, list]:
    """Sealed-holdout blend + calibrated twin (Platt on all pooled OOF pairs)."""
    if arm == "adaptive":
        w = np.array([adaptive_w[n] for n in members])
        blend = np.clip(X_hold @ w, EPS, 1 - EPS)
    else:
        X_all = build_member_matrix(
            {"members": {n: np.concatenate([r["members"][n] for r in fold_recs])
                         for n in members}}, members)
        y_all = np.concatenate([r["y"] for r in fold_recs])
        if len(y_all) < MIN_STACK_FIT:
            w, b, mu, sd = _equal_weights(len(members))
        else:
            w, b, mu, sd = fit_stack(X_all, y_all, stack_variant)
        blend = stack_predict(X_hold, w, b, mu, sd)
    cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
    return blend, np.asarray(apply_platt(np.asarray(blend, dtype=float), cal))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 folds -> /tmp, gate skipped")
    ap.add_argument("--variants", type=str, default=",".join(STACK_VARIANTS))
    args = ap.parse_args()

    sha = rma.head_sha()
    (games, tune_enriched, hold_df, folds, _m, hold_margins, _rounds, _u) = \
        rma.prepare_data(holdout_days=21)
    if args.smoke:
        folds = folds[:3]
    # The sealed holdout must be enriched exactly like run_margin_ablation does
    # (prepare_data returns the RAW holdout; the members need run_margin_diff).
    hold_enriched = rma.attach(hold_df, hold_margins)

    print(f"commit={sha[:12]} games={len(games)} tuning={len(tune_enriched)} "
          f"holdout={len(hold_df)} folds={len(folds)} seed={RANDOM_SEED} "
          f"l2={L2}", flush=True)
    print("running ONE fold loop (5-member ensemble per fold) ...", flush=True)
    cache = run_fold_cache(folds, tune_enriched, hold_enriched)
    members = cache["members"]
    print(f"members: {members}", flush=True)

    y_pooled = np.concatenate([r["y"] for r in cache["fold_recs"]])
    # compute_adaptive_weights expects plain lists (its emptiness check runs
    # `if not preds` on the values).
    pooled_members = {n: np.concatenate([r["members"][n]
                                         for r in cache["fold_recs"]]).tolist()
                      for n in members}

    # Production adaptive renormalization (pooled-OOF AUC softmax + floor/cap).
    adaptive_w = training.compute_adaptive_weights(pooled_members, y_pooled)
    print(f"adaptive weights: {adaptive_w}", flush=True)

    results = {
        "schema": "stack-ablation/v1",
        "commit_sha": sha,
        "data": "data_delivery/game_level_features.csv",
        "seed": int(RANDOM_SEED),
        "l2": L2,
        "members": members,
        "fold_geometry": "shared 45-fold walk-forward (MIN_VAL_FOLD_GAMES); "
                         "sealed 284 = [2026-08-05 .. 2026-08-25]",
        "folds_executed": len(cache["fold_recs"]),
        "stack_fit": "scipy SLSQP logistic on member OOF probs, STANDARDIZED "
                     "with fit-time stats (leak-free), L2=0.01; pooled scoring "
                     "is prequential (fold k fit on folds < k); sealed fit on "
                     "all pooled OOF (strictly pre-holdout)",
        "arms": {},
        "adaptive_weights": {k: round(float(v), 4)
                             for k, v in adaptive_w.items()},
    }

    arms = ["adaptive"] + [v for v in args.variants.split(",") if v.strip()]
    arm_results = {}
    for arm in arms:
        if arm == "adaptive":
            oof_blend, oof_blend_cal, oof_y = prequential_blend(
                cache["fold_recs"], members, "adaptive", adaptive_w=adaptive_w)
            blend_h, cal_h = holdout_blend(
                cache["X_hold"], cache["y_hold"], members, "adaptive",
                None, adaptive_w, cache["fold_recs"], oof_blend, oof_y)
            label = "adaptive"
        else:
            if arm not in STACK_VARIANTS:
                print(f"  unknown variant {arm}, skipping")
                continue
            oof_blend, oof_blend_cal, oof_y = prequential_blend(
                cache["fold_recs"], members, "stack", stack_variant=arm)
            blend_h, cal_h = holdout_blend(
                cache["X_hold"], cache["y_hold"], members, "stack", arm,
                adaptive_w, cache["fold_recs"], oof_blend, oof_y)
            label = f"stack_{arm}"

        pooled_raw = training.compute_metrics(
            np.asarray(oof_y), np.asarray(oof_blend, dtype=float))
        pooled_cal = training.compute_metrics(
            np.asarray(oof_y), np.asarray(oof_blend_cal, dtype=float))
        hold_raw = training.compute_metrics(
            cache["y_hold"], np.asarray(blend_h, dtype=float))
        hold_cal = training.compute_metrics(
            cache["y_hold"], np.asarray(cal_h, dtype=float))
        arm_results[label] = {
            "pooled": {"blend": pooled_raw, "blend_calibrated": pooled_cal},
            "holdout": {"blend": hold_raw, "blend_calibrated": hold_cal},
        }
        print(f"  {label}: pooled ll={pooled_raw['logloss']:.4f} "
              f"auc={pooled_raw['auc']:.4f} ece={pooled_raw['ece']:.4f} "
              f"(cal {pooled_cal['ece']:.4f}) | sealed ll={hold_raw['logloss']:.4f} "
              f"auc={hold_raw['auc']:.4f} ece={hold_raw['ece']:.4f} "
              f"(cal {hold_cal['ece']:.4f})", flush=True)
    results["arms"] = arm_results

    # ── Gate: best stack vs adaptive on the SEALED holdout ────────────────
    ad = arm_results["adaptive"]
    if not args.smoke and len(arm_results) > 1:
        ad_h = ad["holdout"]["blend"]
        ad_hc = ad["holdout"]["blend_calibrated"]
        best = None
        for label, r in arm_results.items():
            if label == "adaptive":
                continue
            h = r["holdout"]["blend"]
            hc = r["holdout"]["blend_calibrated"]
            beats = (h["logloss"] < ad_h["logloss"]
                     and h["auc"] >= ad_h["auc"]
                     and hc["ece"] <= ad_hc["ece"])
            if beats:
                if best is None or h["logloss"] < best[1]["holdout"]["blend"]["logloss"]:
                    best = (label, r)
        if best is None:
            verdict, reason = ("DON'T ADOPT",
                               "no stack variant beats adaptive on sealed "
                               "logloss AND AUC without ECE degradation")
            results["gate"] = {"verdict": verdict, "reason": reason,
                               "adaptive_sealed": ad_h,
                               "adaptive_sealed_cal": ad_hc}
        else:
            label, r = best
            h = r["holdout"]["blend"]
            hc = r["holdout"]["blend_calibrated"]
            po_ad = ad["pooled"]["blend"]["logloss"]
            po_st = r["pooled"]["blend"]["logloss"]
            pool_ok = po_st <= po_ad + 0.0005
            verdict = ("ADOPT" if pool_ok else "DON'T ADOPT")
            results["gate"] = {
                "verdict": verdict,
                "best_variant": label,
                "reason": (f"{label} beats adaptive on sealed logloss "
                           f"({h['logloss']} vs {ad_h['logloss']}) AND AUC "
                           f"({h['auc']} vs {ad_h['auc']}) with ECE-cal "
                           f"{hc['ece']} <= {ad_hc['ece']}"
                           + ("" if pool_ok else
                              "; pooled OOF logloss lost -> not adopted")),
                "adaptive_sealed": ad_h,
                "adaptive_sealed_cal": ad_hc,
                "stack_sealed": h,
                "stack_sealed_cal": hc,
            }
        print(f"\nGATE: {results['gate']['verdict']} — "
              f"{results['gate']['reason']}")

    target = pd.Timestamp.now().date().isoformat()
    compact = target.replace("-", "")
    out = args.out or (DATA_DELIVERY_DIR / f"stack_ablation_{compact}.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"record -> {out}")


if __name__ == "__main__":
    main()