"""Adaptive-weight CAP ablation for the moneyline blend.

Question: does the hard per-member cap on the adaptive blend weights
(ADAPTIVE_WEIGHT_CAP = 0.45) actively prevent overfit, or does it
silently hold the ensemble below what the members earn?

Protocol (identical to the prior blend-level gates — stack, calibration
flip, margin, edge correction): the shared 45-fold walk-forward geometry
+ sealed 284 holdout ([2026-08-05 .. 2026-08-25] on the 08-27 frame),
ONE fold loop trains the 5-member ensemble per fold and caches each
fold's member OOF probabilities (shared by every arm — apples-to-
apples), and each arm's blend = sum(w * p_member) for pooled OOF AND
holdout, with prequential Platt calibrated twins.

Arms:
  capped    — production: compute_adaptive_weights (pooled-OOF AUC
              softmax, temperature 0.03) + floor 0.05 + hard cap 0.45.
  uncapped  — same softmax/floor, cap = 1.0 (the cap fully removed).
  temperature — ONLY pursued if uncapped beats capped on sealed: the
              smooth substitute for the hard cap — the same softmax at
              a 2x temperature (0.06), then the same floor/cap
              projection (which now rarely binds). Hyperparameter-lean:
              one constant, no per-fold fitting.

Measured:
  (1) xgboost's RAW, unconstrained softmax weight preference (pre-
      projection) — is it trying to reach 48% or 85%?
  (2) how often across folds ANY member hits the 45% cap, and by how
      much (prequential weight path: weights earned from folds <= k).
      Does the cap even bind on the pooled production weights?
  (3) capped vs uncapped blend on sealed 284 + pooled OOF
      (logloss/AUC/ECE, raw + calibrated).

DECISION RULE (task):
  - uncapped WORSE on sealed than capped -> the hard cap is actively
    preventing overfit; keep the cap. Record as evidence; NO smart-cap
    pursuit.
  - uncapped BETTER on sealed -> pursue the ONE smart-cap variant
    (softmax temperature), gated through the standard blend protocol:
    ADOPT only if it beats capped on sealed logloss AND AUC without
    degrading sealed ECE, and pooled OOF logloss is not lost.

Record: data_delivery/weight_cap_ablation_<date>.json (date-stamped).
COMMITS NOTHING.

Usage:
    python run_weight_cap_ablation.py
    python run_weight_cap_ablation.py --smoke   # 3 folds -> /tmp
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
import run_margin_ablation as rma  # noqa: E402
import run_stack_ablation as rsa  # noqa: E402
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402
from config import (DATA_DELIVERY_DIR, RANDOM_SEED,  # noqa: E402
                    ADAPTIVE_WEIGHT_AUC_TEMPERATURE)

EPS = 1e-7
CAP_PRODUCTION = float(training.ADAPTIVE_WEIGHT_CAP)   # 0.45
CAP_REMOVED = 1.0
# Production temperature follows the configured metric: with
# ADAPTIVE_WEIGHT_METRIC="auc" the score is pooled-OOF AUC at
# ADAPTIVE_WEIGHT_AUC_TEMPERATURE (0.015); the logloss arm uses
# ADAPTIVE_WEIGHT_TEMPERATURE (0.03).
TEMP_PRODUCTION = float(
    ADAPTIVE_WEIGHT_AUC_TEMPERATURE
    if training.ADAPTIVE_WEIGHT_METRIC == "auc"
    else training.ADAPTIVE_WEIGHT_TEMPERATURE)
TEMP_SMART = 2.0 * TEMP_PRODUCTION                      # 0.03 (smooth cap)


# ── Weight machinery (replicated from training.compute_adaptive_weights) ────

def raw_softmax_weights(oof_members: dict[str, list[float]],
                        y_oof: np.ndarray,
                        temperature: float = TEMP_PRODUCTION) -> dict[str, float]:
    """The UNCONSTRAINED softmax preference (pre floor/cap projection).

    Mirrors compute_adaptive_weights exactly up to the projection:
    scores = pooled OOF AUC per member (missing AUC skipped), softmax at
    the given temperature against the best member, normalized to sum 1.
    """
    y = np.asarray(y_oof, dtype=float)
    scores: dict[str, float] = {}
    for name, preds in oof_members.items():
        p = np.asarray(preds, dtype=float)
        if p.size == 0 or p.size != y.size:
            continue
        m = training.compute_metrics(y, p)
        a = m.get("auc")
        if a is None or not np.isfinite(a):
            continue
        scores[name] = float(a)
    if not scores:
        return {}
    best = max(scores.values())
    exp_w = {n: float(np.exp((a - best) / temperature))
             for n, a in scores.items()}
    tot = sum(exp_w.values())
    return {n: float(v / tot) for n, v in exp_w.items()}


def project_weights(raw: dict[str, float], cap: float) -> dict[str, float]:
    """Floor/cap projection with renormalization (training's loop).

    cap = CAP_PRODUCTION -> the production hard cap; cap = CAP_REMOVED ->
    the unconstrained projection (floor only). Sums to exactly 1.0.
    """
    w = {n: float(v) for n, v in raw.items()}
    eff_cap = max(cap, 1.02 / max(len(w), 1))
    for _ in range(50):
        w = {n: max(v, training.ADAPTIVE_WEIGHT_FLOOR) for n, v in w.items()}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}
        w = {n: min(v, eff_cap) for n, v in w.items()}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}
    rounded = {n: round(v, 4) for n, v in w.items()}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(rounded, key=lambda n: w[n])
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


# ── Fold-loop cache (per-fold checkpoints; interrupted runs resume) ─────────

def _fold_cache_path(sha: str, n_folds: int) -> Path:
    return Path(tempfile.gettempdir()) / f"wcap_folds_{sha[:12]}_{n_folds}.parquet"


def _hold_cache_path(sha: str) -> Path:
    return Path(tempfile.gettempdir()) / f"wcap_hold_{sha[:12]}.parquet"


def _fold_recs_from_df(df: pd.DataFrame) -> list[dict]:
    members = [c for c in df.columns if c not in ("fold_idx", "y")]
    recs = []
    for idx in sorted(df["fold_idx"].unique()):
        g = df[df["fold_idx"] == idx]
        recs.append({"fold_idx": int(idx),
                     "y": g["y"].to_numpy(float),
                     "members": {n: g[n].to_numpy(float) for n in members}})
    return recs


def _save_fold_cache(p: Path, fold_recs: list[dict]) -> None:
    frames = []
    for r in fold_recs:
        d = {"fold_idx": r["fold_idx"], "y": r["y"]}
        d.update(r["members"])
        frames.append(pd.DataFrame(d))
    pd.concat(frames, ignore_index=True).to_parquet(p)


def cached_fold_loop(folds, tune_enriched, hold_enriched, sha: str) -> dict:
    """One fold loop with a per-fold /tmp checkpoint (resume-safe).

    Mirrors rsa.run_fold_cache exactly (same member extraction, same
    intersection filter) but writes the fold_recs to disk after EVERY
    fold, so a run killed by a terminal timeout resumes without
    retraining finished folds. The sealed-holdout member block is cached
    separately (it is a single full-frame train).
    """
    n_folds = len(folds)
    p = _fold_cache_path(sha, n_folds)
    fold_recs: list[dict] = []
    names: list[str] | None = None
    start = 0
    if p.exists():
        df = pd.read_parquet(p)
        done = int(df["fold_idx"].nunique())
        fold_recs = _fold_recs_from_df(df)
        if done >= n_folds:
            print(f"fold cache hit ({done}/{n_folds})", flush=True)
        else:
            print(f"resuming fold loop from {done}/{n_folds} "
                  f"(checkpoint)", flush=True)
            start = done
    t0 = time.time()
    for split in folds[start:]:
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
        _save_fold_cache(p, fold_recs)  # checkpoint after every fold
        print(f"  fold {split['fold_idx']} done ({len(fold_recs)}/{n_folds}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    if names is None or len(names) < 3:
        raise RuntimeError(f"too few members trained: {names}")
    keep = [n for n in names if all(n in r["members"] for r in fold_recs)]
    if len(keep) < 3:
        raise RuntimeError(f"members missing across folds: {keep}")
    for r in fold_recs:
        r["members"] = {n: r["members"][n] for n in keep}
    _save_fold_cache(p, fold_recs)

    hp = _hold_cache_path(sha)
    if hp.exists():
        hd = pd.read_parquet(hp)
        X_hold = hd[[f"m_{n}" for n in keep]].to_numpy(float)
        y_hold = hd["y"].to_numpy(float)
        print("holdout cache hit", flush=True)
    else:
        models_hold, _ = training.train_moneyline_ensemble(tune_enriched)
        _blend_h, member_hold, _wts_h = training.ensemble_predict(
            models_hold, hold_enriched)
        y_hold = hold_enriched["home_win"].values.astype(float)
        hdf = pd.DataFrame({"y": y_hold})
        for n in keep:
            hdf[f"m_{n}"] = np.asarray(member_hold[n], dtype=float)
        hdf.to_parquet(hp)
        X_hold = hdf[[f"m_{n}" for n in keep]].to_numpy(float)
    return {"members": keep, "fold_recs": fold_recs,
            "X_hold": X_hold, "y_hold": y_hold}


# ── Measurement helpers ─────────────────────────────────────────────────────

def prequential_weight_path(fold_recs: list[dict], members: list[str]) -> list[dict]:
    """Per-fold weight path: for fold k, weights earned from folds <= k.

    Returns one row per fold with raw softmax, capped, and uncapped
    weights — the data for "how often does the cap bind, and by how
    much" across the walk-forward.
    """
    path = []
    pooled_m: dict[str, list[float]] = {n: [] for n in members}
    pooled_y: list[float] = []
    for rec in fold_recs:
        for n in members:
            pooled_m[n].extend(rec["members"][n].tolist())
        pooled_y.extend(rec["y"].tolist())
        raw = raw_softmax_weights(pooled_m, np.asarray(pooled_y))
        capped = project_weights(raw, CAP_PRODUCTION)
        uncapped = project_weights(raw, CAP_REMOVED)
        eff_cap = max(CAP_PRODUCTION, 1.02 / max(len(members), 1))
        binds = {n: float(uncapped[n] - capped[n])
                 for n in members
                 if uncapped[n] > capped[n] + 1e-6}
        path.append({
            "fold": int(rec["fold_idx"]),
            "n_games": int(len(pooled_y)),
            "raw": {n: round(float(raw[n]), 4) for n in members},
            "capped": {n: round(float(capped[n]), 4) for n in members},
            "uncapped": {n: round(float(uncapped[n]), 4) for n in members},
            "eff_cap": round(eff_cap, 4),
            "binding": binds,
            "any_binding": bool(binds),
        })
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 folds -> /tmp, gate skipped")
    ap.add_argument("--target-date", type=str, default=None)
    args = ap.parse_args()

    sha = rma.head_sha()
    (games, tune_enriched, hold_df, folds, _m, hold_margins, _rounds, _u) = \
        rma.prepare_data(holdout_days=21)
    if args.smoke:
        folds = folds[:3]
    hold_enriched = rma.attach(hold_df, hold_margins)

    print(f"commit={sha[:12]} games={len(games)} tuning={len(tune_enriched)} "
          f"holdout={len(hold_df)} folds={len(folds)} seed={RANDOM_SEED} "
          f"cap={CAP_PRODUCTION} temp={TEMP_PRODUCTION}", flush=True)
    print("running ONE fold loop (5-member ensemble per fold, "
          "checkpointed) ...", flush=True)
    cache = cached_fold_loop(folds, tune_enriched, hold_enriched, sha)
    members = cache["members"]
    print(f"members: {members}", flush=True)

    y_pooled = np.concatenate([r["y"] for r in cache["fold_recs"]])
    pooled_members = {n: np.concatenate([r["members"][n]
                                         for r in cache["fold_recs"]]).tolist()
                      for n in members}

    # (1) RAW unconstrained preference + the three weight vectors.
    raw_w = raw_softmax_weights(pooled_members, y_pooled)
    capped_w = project_weights(raw_w, CAP_PRODUCTION)
    uncapped_w = project_weights(raw_w, CAP_REMOVED)
    # Consistency: the replicated projection must reproduce production.
    prod_w = training.compute_adaptive_weights(pooled_members, y_pooled)
    assert capped_w == prod_w, f"projection mismatch vs production: {capped_w} vs {prod_w}"

    # (2) per-fold weight path (prequential) — does the cap bind?
    weight_path = prequential_weight_path(cache["fold_recs"], members)
    n_binding_folds = sum(1 for r in weight_path if r["any_binding"])
    max_clip = max((max(r["binding"].values()) for r in weight_path
                    if r["binding"]), default=0.0)
    bound_members: dict[str, int] = {}
    for r in weight_path:
        for n in r["binding"]:
            bound_members[n] = bound_members.get(n, 0) + 1
    eff_cap = max(CAP_PRODUCTION, 1.02 / max(len(members), 1))
    pooled_binding = {n: round(float(uncapped_w[n] - capped_w[n]), 4)
                      for n in members if uncapped_w[n] > capped_w[n] + 1e-6}

    print("\n[1] RAW softmax preference (pooled OOF AUC):")
    for n in members:
        print(f"    {n:<12} raw={raw_w[n]:.4f} capped={capped_w[n]:.4f} "
              f"uncapped={uncapped_w[n]:.4f}")
    print(f"[2] cap binds on {n_binding_folds}/{len(weight_path)} folds "
          f"(max clip {max_clip:.4f}); pooled binding: {pooled_binding}")

    # (3) blends + scoring for each arm.
    arms = {
        "capped": capped_w,
        "uncapped": uncapped_w,
    }
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
        print(f"  {label}: pooled ll={pooled_raw['logloss']:.4f} "
              f"auc={pooled_raw['auc']:.4f} ece={pooled_raw['ece']:.4f} "
              f"(cal {pooled_cal['ece']:.4f}) | sealed ll={hold_raw['logloss']:.4f} "
              f"auc={hold_raw['auc']:.4f} ece={hold_raw['ece']:.4f} "
              f"(cal {hold_cal['ece']:.4f})", flush=True)

    # ── DECISION RULE ─────────────────────────────────────────────────────
    c_h = arm_results["capped"]["holdout"]["blend"]
    u_h = arm_results["uncapped"]["holdout"]["blend"]
    u_hc = arm_results["uncapped"]["holdout"]["blend_calibrated"]
    c_hc = arm_results["capped"]["holdout"]["blend_calibrated"]
    uncapped_worse = u_h["logloss"] >= c_h["logloss"]
    gate: dict = {
        "uncapped_sealed_deltas": {
            "logloss": round(float(u_h["logloss"] - c_h["logloss"]), 5),
            "auc": round(float(u_h["auc"] - c_h["auc"]), 5),
            "ece": round(float(u_h["ece"] - c_h["ece"]), 5),
            "ece_calibrated": round(float(u_hc["ece"] - c_hc["ece"]), 5),
        },
    }
    if uncapped_worse and not args.smoke:
        gate["verdict"] = "KEEP THE CAP"
        gate["reason"] = (
            f"uncapped sealed logloss {u_h['logloss']} >= capped "
            f"{c_h['logloss']} — the hard cap actively prevents overfit; "
            "any smart cap that permits weights >45% will likely lose too. "
            "Recorded as evidence; no smart-cap pursuit.")
    else:
        # Uncapped is better (or smoke): pursue the ONE smart-cap variant —
        # softmax temperature (smooth substitute for the hard cap).
        temp_raw = raw_softmax_weights(pooled_members, y_pooled,
                                       temperature=TEMP_SMART)
        temp_w = project_weights(temp_raw, CAP_PRODUCTION)
        oof_blend_t, oof_blend_cal_t, oof_y_t = rsa.prequential_blend(
            cache["fold_recs"], members, "adaptive", adaptive_w=temp_w)
        blend_h_t, cal_h_t = rsa.holdout_blend(
            cache["X_hold"], cache["y_hold"], members, "adaptive",
            None, temp_w, cache["fold_recs"], oof_blend_t, oof_y_t)
        arm_results["temperature"] = {
            "weights": {k: round(float(v), 4) for k, v in temp_w.items()},
            "pooled": {"blend": training.compute_metrics(
                np.asarray(oof_y_t), np.asarray(oof_blend_t, dtype=float)),
                "blend_calibrated": training.compute_metrics(
                    np.asarray(oof_y_t), np.asarray(oof_blend_cal_t, dtype=float))},
            "holdout": {"blend": training.compute_metrics(
                cache["y_hold"], np.asarray(blend_h_t, dtype=float)),
                "blend_calibrated": training.compute_metrics(
                    cache["y_hold"], np.asarray(cal_h_t, dtype=float))},
        }
        t_h = arm_results["temperature"]["holdout"]["blend"]
        t_hc = arm_results["temperature"]["holdout"]["blend_calibrated"]
        t_pool = arm_results["temperature"]["pooled"]["blend"]
        c_pool = arm_results["capped"]["pooled"]["blend"]
        print(f"  temperature: pooled ll={arm_results['temperature']['pooled']['blend']['logloss']:.4f} "
              f"| sealed ll={t_h['logloss']:.4f} auc={t_h['auc']:.4f} ece={t_h['ece']:.4f} "
              f"(cal {t_hc['ece']:.4f})", flush=True)
        if args.smoke:
            gate["verdict"] = "SMOKE — gate skipped"
            gate["reason"] = "3-fold smoke run; temperature arm computed for shape only."
        else:
            beats = (t_h["logloss"] < c_h["logloss"]
                     and t_h["auc"] >= c_h["auc"]
                     and t_hc["ece"] <= c_hc["ece"]
                     and t_pool["logloss"] <= c_pool["logloss"] + 0.0005)
            gate["verdict"] = "ADOPT temperature smart-cap" if beats else \
                "DON'T ADOPT temperature — keep the cap"
            gate["reason"] = (
                f"uncapped sealed logloss {u_h['logloss']} < capped "
                f"{c_h['logloss']} (uncapped better) -> smart-cap pursued; "
                f"temperature {TEMP_SMART} weights {temp_w} "
                f"sealed ll {t_h['logloss']} vs capped {c_h['logloss']}, "
                f"auc {t_h['auc']} vs {c_h['auc']}, ece-cal {t_hc['ece']} "
                f"vs {c_hc['ece']}; pooled ll {t_pool['logloss']} vs "
                f"{c_pool['logloss']}")

    target = (args.target_date
              or pd.Timestamp.now().date().isoformat()).replace("-", "")
    record = {
        "schema": "weight-cap-ablation/v1",
        "commit_sha": sha,
        "date": target,
        "data": "data_delivery/game_level_features.csv",
        "seed": int(RANDOM_SEED),
        "members": members,
        "fold_geometry": "shared 45-fold walk-forward (MIN_VAL_FOLD_GAMES); "
                         "sealed 284 = last 284 games by date",
        "folds_executed": len(cache["fold_recs"]),
        "weight_machinery": {
            "metric": training.ADAPTIVE_WEIGHT_METRIC,
            "temperature_production": TEMP_PRODUCTION,
            "temperature_smart": TEMP_SMART,
            "auc_temperature_config": ADAPTIVE_WEIGHT_AUC_TEMPERATURE,
            "floor": training.ADAPTIVE_WEIGHT_FLOOR,
            "cap_production": CAP_PRODUCTION,
            "cap_removed": CAP_REMOVED,
            "eff_cap": round(eff_cap, 4),
        },
        "q1_raw_preference": {n: round(float(raw_w[n]), 4) for n in members},
        "q2_cap_binding": {
            "n_folds": len(weight_path),
            "n_binding_folds": n_binding_folds,
            "binding_folds_pct": round(100 * n_binding_folds
                                       / max(len(weight_path), 1), 1),
            "max_clip": round(float(max_clip), 4),
            "member_binding_counts": bound_members,
            "pooled_production_binding": pooled_binding,
            "per_fold_path": weight_path,
        },
        "weights": {
            "raw": {n: round(float(raw_w[n]), 4) for n in members},
            "capped": {k: round(float(v), 4) for k, v in capped_w.items()},
            "uncapped": {k: round(float(v), 4) for k, v in uncapped_w.items()},
        },
        "arms": arm_results,
        "gate": gate,
        "acceptance": {
            "rule": ("uncapped WORSE on sealed -> KEEP the hard cap (it "
                     "prevents overfit). uncapped BETTER -> pursue softmax "
                     "temperature, ADOPT only if it beats capped on sealed "
                     "logloss AND AUC without degrading sealed ECE and "
                     "pooled OOF logloss is not lost."),
            "result": gate.get("verdict", ""),
        },
    }
    if args.smoke:
        out = Path("/tmp/weight_cap_ablation_smoke.json")
    else:
        out = args.out or (DATA_DELIVERY_DIR
                           / f"weight_cap_ablation_{target}.json")
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nGATE: {gate.get('verdict')} — {gate.get('reason', '')}")
    print(f"record -> {out}")


if __name__ == "__main__":
    main()
