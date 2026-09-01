"""Blend-level confirmation: tuned LightGBM params vs the current config.

Mirrors run_rf_tuned_blend_ablation.py's production-correct recipe, but for
the LightGBM member. CRITICAL DIFFERENCE from the RF harness: training.py
imports LIGHTGBM_PARAMS at MODULE level (line ~40) and binds it once, so the
per-variant swap must patch `training.LIGHTGBM_PARAMS`, NOT
`config.LIGHTGBM_PARAMS` (RF, by contrast, imports RF_PARAMS inside
train_moneyline_ensemble at call time).

1. run_margin_diff is ATTACHED exactly as production does — via
   training._attach_oof_run_margins over the whole decided frame (leakage-free
   OOF margins). Both arms sit on the real shipped baseline.
2. Each variant gets the full walk-forward pooled OOF (all folds, raw +
   prequential-calibrated) at the FULL-ENSEMBLE level: all 5 members + the
   static-prior blend, per-variant blend weights cleared.
3. The decision is evaluated on MULTIPLE sealed holdout windows (--windows),
   so a single 21-day slice can't drive the verdict.

Arms:
    CURRENT = config.LIGHTGBM_PARAMS verbatim (the deployed 50-round config)
    TUNED   = the tune_lightgbm_optuna.py winner (study
              lightgbm_moneyline_65col, 75 trials), scored at 50 rounds —
              exactly the candidate the tuner's sealed verification tested
              (median best_iter 14 clamped to the 50-round floor).

Gate (per window AND pooled): TUNED must beat CURRENT on holdout logloss AND
AUC. ECE and per-member movement are flagged in caveats, NOT part of the
verdict (policy: docs/model_tuning_policy.md).

Emits data_delivery/lgb_blend_ablation_<sha>.json (incremental). COMMITS NOTHING.
    python run_lgb_tuned_blend_ablation.py --windows 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
import config  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR, MIN_VAL_FOLD_GAMES, RANDOM_SEED, RETRAIN_CADENCE_DAYS,
)
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402

EPS = 1e-7

LGB_CURRENT = dict(config.LIGHTGBM_PARAMS)

# tune_lightgbm_optuna.py winner — study 'lightgbm_moneyline_65col', 75
# trials, pooled-OOF-logloss objective, margin-attached folds. Scored at 50
# rounds (the tuner's clamp: median best_iter 14 -> 50-round floor), matching
# the candidate the tuner's sealed verification tested.
LGB_TUNED = {
    "n_estimators": 50,
    "max_depth": 8,
    "num_leaves": 37,
    "min_child_samples": 83,
    "min_gain_to_split": 1.5790962151980537,
    "bagging_fraction": 0.5153966126860083,
    "bagging_freq": 2,
    "feature_fraction": 0.5054180971108294,
    "learning_rate": 0.09481182230869382,
    "lambda_l1": 0.022357477299229024,
    "lambda_l2": 0.0019816106771877785,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _run_pooled(lgb_params: dict, folds) -> tuple[dict, int]:
    training.LIGHTGBM_PARAMS = dict(lgb_params)
    training._LAST_ADAPTIVE_WEIGHTS.clear()

    oof_y: list[float] = []
    oof_blend: list[float] = []
    oof_members: dict[str, list[float]] = {}
    oof_blend_cal: list[float] = []
    oof_members_cal: dict[str, list[float]] = {}
    executed = 0
    for split in folds:
        train = split["train_games"]
        val = split["val_games"]
        try:
            models, _ = training.train_moneyline_ensemble(train, val)
        except Exception as e:
            print(f"  fold {split['fold_idx']} failed: {e}")
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        oof_y.extend(y_val.tolist())
        oof_blend.extend(np.asarray(blend, dtype=float).tolist())
        oof_blend_cal.extend(np.asarray(apply_platt(np.asarray(blend), fold_cal),
                                        dtype=float).tolist())
        for n, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(n, []).extend(pa.tolist())
            oof_members_cal.setdefault(n, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled = {
        "blend": training.compute_metrics(y_all, np.asarray(oof_blend, dtype=float)),
        "blend_calibrated": training.compute_metrics(
            y_all, np.asarray(oof_blend_cal, dtype=float)),
    }
    for n, plist in oof_members.items():
        pooled[n] = training.compute_metrics(y_all, np.asarray(plist, dtype=float))
        pooled[f"{n}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(n, []), dtype=float))
    return pooled, executed


def _sealed(tune_df: pd.DataFrame, hold_df: pd.DataFrame,
            lgb_params: dict) -> dict:
    training.LIGHTGBM_PARAMS = dict(lgb_params)
    training._LAST_ADAPTIVE_WEIGHTS.clear()
    models, _ = training.train_moneyline_ensemble(tune_df)
    blend, member_probs, _wts = training.ensemble_predict(models, hold_df)
    yh = hold_df["home_win"].values.astype(float)
    metrics = {"blend": training.compute_metrics(yh, np.asarray(blend))}
    for n, p in member_probs.items():
        metrics[n] = training.compute_metrics(yh, np.asarray(p, dtype=float))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, default=3,
                    help="number of sealed 21-day windows to evaluate")
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if config.LIGHTGBM_PARAMS != LGB_CURRENT:
        print("WARNING: config.LIGHTGBM_PARAMS != LGB_CURRENT. CURRENT arm "
              "will use the harness's LGB_CURRENT dict regardless.")

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = games.sort_values("game_date").reset_index(drop=True)

    splits = training.walk_forward_splits(games,
                                          retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    if "run_margin_diff" in training.FEATURE_COLS and splits:
        games, splits = training._attach_oof_run_margins(
            games, splits, MIN_VAL_FOLD_GAMES, 0, RETRAIN_CADENCE_DAYS, 0,
            decided_snapshot=games)
        cov = float(games["run_margin_diff"].notna().mean())
    else:
        games["run_margin_diff"] = np.nan
        cov = 0.0
    folds = [s for s in splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    print(f"lgb-blend commit={sha[:12]} data_sha={data_hash[:12]} "
          f"games={len(games)} margin_cov={cov:.3f} "
          f"folds={len(splits)}/{len(folds)}")

    variants = {"CURRENT": LGB_CURRENT, "TUNED": LGB_TUNED}
    out = args.out or (DATA_DELIVERY_DIR
                       / f"lgb_blend_ablation_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "lgb-blend-ablation/v1", "commit_sha": sha,
                   "data_sha256": data_hash, "windows": args.windows,
                   "holdout_days": args.holdout_days,
                   "folds_executed": len(folds), "margin_coverage": cov,
                   "clip_eps": EPS, "seed": int(RANDOM_SEED),
                   "lgb_current": LGB_CURRENT, "lgb_tuned": LGB_TUNED,
                   "variants": {}}

    maxd = games["game_date"].max()
    for name in ("CURRENT", "TUNED"):
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running pooled ...")
        pooled, executed = _run_pooled(variants[name], folds)
        variant = {"pooled": pooled, "executed": executed, "windows": {}}
        for k in range(args.windows):
            end = maxd - pd.Timedelta(days=k * args.holdout_days)
            start = end - pd.Timedelta(days=args.holdout_days)
            hold_k = games[(games["game_date"] > start)
                           & (games["game_date"] <= end)].reset_index(drop=True)
            tune_k = games[games["game_date"] <= start].reset_index(drop=True)
            hm = _sealed(tune_k, hold_k, variants[name])
            variant["windows"][f"w{k}"] = {
                "n_hold": int(len(hold_k)), "holdout": hm["blend"],
                "hold_start": str(start.date()),
                "hold_end": str(end.date()),
            }
            pb = pooled["blend"]
            hb = hm["blend"]
            print(f"    [{name}] pooled {pb['logloss']:.4f}/{pb['auc']:.4f} | "
                  f"w{k} [{start.date()}..{end.date()}] n={len(hold_k)} "
                  f"holdout {hb['logloss']:.4f}/{hb['auc']:.4f}/{hb['ece']:.4f}")
        results["variants"][name] = variant
        out.write_text(json.dumps(results, indent=2) + "\n")

    if "CURRENT" in results["variants"] and "TUNED" in results["variants"]:
        wo = results["variants"]["CURRENT"]
        wi = results["variants"]["TUNED"]
        wouts = {}
        any_win_gain = 0
        losses = []
        for k in range(args.windows):
            h0 = wo["windows"][f"w{k}"]["holdout"]; h1 = wi["windows"][f"w{k}"]["holdout"]
            wins = (h1["logloss"] < h0["logloss"]) and (h1["auc"] > h0["auc"])
            wouts[f"w{k}"] = {"verdict": "SHIP" if wins else "DON'T SHIP",
                              "holdout_current": h0, "holdout_tuned": h1}
            if wins:
                any_win_gain += 1
            else:
                losses.append(f"w{k}")
        p0 = wo["pooled"]["blend"]; p1 = wi["pooled"]["blend"]
        pooled_gain = (p1["logloss"] < p0["logloss"]) and (p1["auc"] > p0["auc"])
        overall = ("SHIP" if (any_win_gain == args.windows) and pooled_gain
                   else "DON'T SHIP")
        caveats = []
        for k in range(args.windows):
            h0 = wo["windows"][f"w{k}"]["holdout"]; h1 = wi["windows"][f"w{k}"]["holdout"]
            if h1["ece"] is not None and h0["ece"] is not None and h1["ece"] > h0["ece"]:
                caveats.append(
                    f"w{k} holdout ece declined {h0['ece']:.4f}->{h1['ece']:.4f}")
        results["gate"] = {
            "verdict": overall, "pooled_current": p0, "pooled_tuned": p1,
            "wins": any_win_gain, "total": args.windows,
            "loss_windows": losses, "windows": wouts,
            "ece_excluded_from_verdict": True, "caveats": caveats,
        }
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"GATE: pooled {p0['logloss']:.4f}/{p0['auc']:.4f} vs "
              f"{p1['logloss']:.4f}/{p1['auc']:.4f} -> "
              f"{'pooled-gain' if pooled_gain else 'no-pooled-gain'}; sealed "
              f"wins {any_win_gain}/{args.windows} -> {overall}")
        for c in caveats:
            print(f"  caveat: {c}")


if __name__ == "__main__":
    main()
