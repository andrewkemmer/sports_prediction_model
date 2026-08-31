"""Standalone-XGBoost lineup-delta ablation — full walk-forward, XGBoost ONLY.

Isolates XGBoost from the 5-member ensemble to answer one question: does adding
the six lineup_actual_* features help a SINGLE XGBoost moneyline model out of
sample? The ensemble context is removed so the per-member sample is the model.

Reproduces the production XGBoost member path EXACTLY (backend/training.py):
  - features = current FEATURE_COLS (WITHOUT) or + the 6 (WITH/WITHMASK)
  - _prepare_features -> _impute_median (train-fold medians only) ->
    _tree_dataframe (named numeric + pandas-Categorical team IDs)
  - walk-forward folds: XGBClassifier(**XGBOOST_PARAMS,
      n_estimators=XGBOOST_FOLD_ROUNDS, early_stopping_rounds=XGBOOST_EARLY_STOP)
      with eval_set on the val window; logloss metric
  - sealed refit: XGBClassifier(**XGBOOST_PARAMS) fit-only on all tune games
All vars the FUNCTIONS set module-globally (training.FEATURE_COLS) are reset
per variant so the width/order of the matrix matches the variant.

Arms:
  WITHOUT  = production FEATURE_COLS
  WITH     = + the 6, trained on REAL actuals
  WITHMASK = + the 6, randomly NULLed on ~mask_rate of TRAINING rows
      (simulates "lineup not yet posted at bet time"; float NaN so the
      imputation path and xgboost's NaN routing see it as genuinely missing)

Metrics: pooled OOF (all folds) raw + prequential-calibrated (fit_platt on
prior folds only); sealed-21-day holdout evaluated TWICE — real lineups and
all-six-NULL projected (bettor has no lineups).

TWO-SIDED GATE (same as run_lineup_ablation.py): a candidate ships only if it
BEATS WITHOUT on the real-actual holdout (logloss AND AUC — the boost leg) and
is NOT worse than WITHOUT on the projected-only holdout within tol (the
no-penalty leg). ECE/calibration flagged in caveats, not gated.

Emits data_delivery/lineup_ablation_xgb_<sha>.json (incremental). COMMITS
NOTHING.
  python run_xgboost_lineup_ablation.py                       # WITHOUT/WITH
  python run_xgboost_lineup_ablation.py --mask-train          # + WITHMASK
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
from features import add_lineup_delta_features, LINEUP_DELTA_COLS  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR, MIN_VAL_FOLD_GAMES, RANDOM_SEED, RETRAIN_CADENCE_DAYS,
    XGBOOST_PARAMS, XGBOOST_FOLD_ROUNDS, XGBOOST_EARLY_STOP,
)
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402

EPS = 1e-7


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


def build_variants() -> dict[str, list[str]]:
    base = [c for c in training.FEATURE_COLS if c not in LINEUP_DELTA_COLS]
    assert base == training.FEATURE_COLS
    assert len(LINEUP_DELTA_COLS) == 6
    return {"WITHOUT": base, "WITH": base + LINEUP_DELTA_COLS,
            "WITHMASK": base + LINEUP_DELTA_COLS}


def _mask_lineups(df: pd.DataFrame, rng, mask_rate: float) -> pd.DataFrame:
    out = df.copy()
    lc = [c for c in LINEUP_DELTA_COLS if c in out.columns]
    if not lc:
        return out
    hit = rng.random(len(out)) < mask_rate
    if hit.any():
        for c in lc:
            out.loc[hit, c] = np.nan
    return out


def _projected_holdout(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in LINEUP_DELTA_COLS:
        if c in out.columns:
            out[c] = np.nan
    return out


def _xgb_matrix(df: pd.DataFrame):
    """Mirror train_moneyline_ensemble's XGBoost input prep for one frame."""
    X, X_cat, y = training._prepare_features(df)
    X_lr, _ = training._impute_median(X)
    nc = list(training.FEATURE_COLS)
    X_xgb = training._tree_dataframe(X_lr, X_cat, nc)
    return X_xgb, y


def _xgb_fold_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    X_tr, y_tr = _xgb_matrix(train)
    X_va, y_va = _xgb_matrix(val)
    from xgboost import XGBClassifier
    m = XGBClassifier(**XGBOOST_PARAMS, n_estimators=XGBOOST_FOLD_ROUNDS,
                      early_stopping_rounds=XGBOOST_EARLY_STOP)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m.predict_proba(X_va)[:, 1]


def _xgb_refit_predict(tune: pd.DataFrame, infer: pd.DataFrame) -> np.ndarray:
    X_tr, y_tr = _xgb_matrix(tune)
    X_in, _ = _xgb_matrix(infer)
    from xgboost import XGBClassifier
    m = XGBClassifier(**XGBOOST_PARAMS)
    m.fit(X_tr, y_tr, verbose=False)
    return m.predict_proba(X_in)[:, 1]


def run_variant(cols: list[str], folds, tune_df, hold_df,
                mask_train: bool = False, mask_rate: float = 0.5) -> dict:
    training.FEATURE_COLS = list(cols)

    oof_y: list[float] = []
    oof_x: list[float] = []
    oof_x_cal: list[float] = []
    executed = 0
    for split in folds:
        train = split["train_games"]
        val = split["val_games"]
        if mask_train:
            train = _mask_lineups(
                train, np.random.default_rng(RANDOM_SEED + split["fold_idx"]),
                mask_rate)
        try:
            p = _xgb_fold_predict(train, val)
        except Exception as e:
            print(f"  fold {split['fold_idx']} failed: {e}")
            continue
        y = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_x) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_x))
        oof_y.extend(y.tolist())
        oof_x.extend(np.asarray(p, dtype=float).tolist())
        oof_x_cal.extend(np.asarray(apply_platt(np.asarray(p), fold_cal),
                                    dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled = {
        "xgboost": training.compute_metrics(y_all, np.asarray(oof_x, dtype=float)),
        "xgboost_calibrated": training.compute_metrics(
            y_all, np.asarray(oof_x_cal, dtype=float)),
    }

    refit_tune = tune_df
    if mask_train:
        refit_tune = _mask_lineups(
            tune_df, np.random.default_rng(RANDOM_SEED), mask_rate)
    models = _xgb_refit_predict(refit_tune, hold_df)
    models_proj = _xgb_refit_predict(refit_tune, _projected_holdout(hold_df))
    yh = hold_df["home_win"].values.astype(float)
    holdout = {"xgboost": training.compute_metrics(yh, np.asarray(models))}
    holdout_projected = {"xgboost": training.compute_metrics(
        yh, np.asarray(models_proj))}

    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout,
            "holdout_projected": holdout_projected}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--mask-train", action="store_true")
    ap.add_argument("--mask-rate", type=float, default=0.5)
    ap.add_argument("--no-penalty-tol", type=float, default=0.001)
    args = ap.parse_args()

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = add_lineup_delta_features(games)

    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    print(f"XGB-only commit={sha[:12]} data_sha={data_hash[:12]} "
          f"games={len(games)} tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} seed={RANDOM_SEED} "
          f"mask_train={args.mask_train} mask_rate={args.mask_rate}")

    variants = build_variants()
    out = args.out or (DATA_DELIVERY_DIR / f"lineup_ablation_xgb_{sha[:12]}.json")
    want = ((args.variants.split(",") if args.variants
             else (["WITHOUT", "WITH", "WITHMASK"] if args.mask_train
                   else ["WITHOUT", "WITH"])))
    want = [v.strip() for v in want if v.strip()]
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "lineup-ablation-xgb/v1", "commit_sha": sha,
                   "data_sha256": data_hash, "holdout_days": args.holdout_days,
                   "folds_declared": len(all_splits),
                   "folds_executed": len(folds), "clip_eps": EPS,
                   "seed": int(RANDOM_SEED), "mask_train": args.mask_train,
                   "mask_rate": args.mask_rate,
                   "no_penalty_tol": args.no_penalty_tol, "variants": {}}

    for name in want:
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        mask_train = (name == "WITHMASK") and args.mask_train
        print(f"  {name}: running ({len(variants[name])} cols, "
              f"mask_train={mask_train}) ...")
        r = run_variant(variants[name], folds, tune_df, hold_df,
                        mask_train=mask_train, mask_rate=args.mask_rate)
        r["cols"] = variants[name]
        results["variants"][name] = r
        out.write_text(json.dumps(results, indent=2) + "\n")
        p = r["pooled"]["xgboost"]
        h = r["holdout"]["xgboost"]
        hp = r["holdout_projected"]["xgboost"]
        print(f"    pooled {p['logloss']:.4f}/{p['auc']:.4f} ece {p['ece']:.4f} | "
              f"holdout {h['logloss']:.4f}/{h['auc']:.4f} | "
              f"holdout_projected {hp['logloss']:.4f}/{hp['auc']:.4f}")

    if "WITHOUT" in results["variants"]:
        wo = results["variants"]["WITHOUT"]["holdout"]["xgboost"]
        wo_proj = (results["variants"]["WITHOUT"].get("holdout_projected", {})
                   .get("xgboost", wo))
        gate_cands: dict[str, dict] = {}
        for name in ("WITH", "WITHMASK"):
            if name not in results["variants"]:
                continue
            wi = results["variants"][name]["holdout"]["xgboost"]
            wi_proj = (results["variants"][name].get("holdout_projected", {})
                       .get("xgboost"))
            boost = (wi["logloss"] < wo["logloss"]) and (wi["auc"] > wo["auc"])
            no_penalty = None
            if wi_proj is not None:
                no_penalty = (
                    wi_proj["logloss"] <= wo_proj["logloss"] + args.no_penalty_tol
                    and wi_proj["auc"] >= wo_proj["auc"] - args.no_penalty_tol)
            verdict = ("SHIP" if (boost and (no_penalty is not False))
                       else "DON'T SHIP")
            caveats: list[str] = []
            if wo.get("ece") is not None and wi.get("ece") is not None:
                if wi["ece"] > wo["ece"]:
                    caveats.append(
                        f"holdout calibration declined (ece {wo['ece']:.4f} -> "
                        f"{wi['ece']:.4f}, delta +{wi['ece'] - wo['ece']:.4f})")
            gate_cands[name] = {
                "verdict": verdict, "boost": boost, "no_penalty": no_penalty,
                "holdout_without": wo, "holdout_with": wi,
                "holdout_projected_with": wi_proj, "caveats": caveats,
            }
            print(f"GATE[{name}]: boost {wo['logloss']:.4f}/{wo['auc']:.4f} vs "
                  f"{wi['logloss']:.4f}/{wi['auc']:.4f} -> "
                  f"{'WINS' if boost else 'LOSES'}; no-penalty projected "
                  f"{wo_proj['logloss']:.4f}/{wo_proj['auc']:.4f} vs "
                  f"{wi_proj['logloss']:.4f}/{wi_proj['auc']:.4f} -> "
                  f"{'OK' if no_penalty else 'FAILS'} -> {verdict}")
            if caveats:
                print("  CAVEATS (not part of verdict):")
                for c in caveats:
                    print(f"    - {c}")
        results["gate"] = {
            "ece_excluded_from_verdict": True,
            "no_penalty_tol": args.no_penalty_tol,
            "candidates": gate_cands,
        }
        out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()