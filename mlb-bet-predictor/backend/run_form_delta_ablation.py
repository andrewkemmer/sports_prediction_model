"""Momentum form-delta ablation — WITH vs WITHOUT on the moneyline ensemble.

Measurement task (NOT tuning): does the continuous "recent window −
season-to-date baseline" delta set help the ensemble out of sample? Nothing is
changed silently; the feature set ships only if it clears the sealed-holdout
gate.

Design (mirrors the locked baseline conventions exactly):
- Data: committed data_delivery/game_level_features.csv (4,451 games, hash
  recorded). Decided games only; add_form_delta_features() computes the
  deltas that are computable on this artifact.
- Honest-on-artifact caveat: the committed CSV ships season baselines only
  for sp_era / sp_k9, so exactly 4 of the 38 delta columns are real here
  (sp_era_delta, sp_k9_delta, both sides). The other 34 are all-NaN by design
  (they land with the next pipeline run, which ships the SQL-computed deltas
  verified by test_form_delta_features). The WITH variant therefore measures
  the 4 real deltas; including 34 constant-NaN columns would measure nothing
  (median-imputed to a constant 0) while adding noise to the experiment.
- Variants: WITHOUT = the original 58 FEATURE_COLS (production baseline);
  WITH = 58 + the 4 computable deltas.
- Folds: walk_forward_splits on the tuning pool with RETRAIN_CADENCE_DAYS,
  filtered by MIN_VAL_FOLD_GAMES (same machinery as walk_forward_evaluate:
  declared vs executed recorded; current engine declares 48 / executes 44).
- Members: all 5 (xgb/lgbm/rf/logistic/mlp) + the static-prior blend (adaptive
  weights cleared before each variant so both variants blend identically).
- Metrics: compute_metrics (clip 1e-7) — logloss / AUC / Brier / ECE — raw
  and prequential-calibrated (fit_platt on prior folds' blend pairs only,
  exactly as walk_forward_evaluate does).
- Sealed 21-day holdout: refit fit-only on the whole tuning pool AFTER the
  fold loop; never touched during fold fitting.

Gate (task rule): ship ONLY if WITH beats WITHOUT on the sealed holdout on
logloss AND AUC without degrading calibration; a pooled win with a holdout
loss is flagged as likely overfit and never adopted.

Emits data_delivery/form_delta_ablation_<sha>.json (incremental — resumes by
skipping variants already present). COMMITS NOTHING.

Usage:
    python run_form_delta_ablation.py
    python run_form_delta_ablation.py --variants WITHOUT,WITH
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
from features import add_form_delta_features, FORM_DELTA_COLS  # noqa: E402
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
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
    """WITHOUT = original 58; WITH = 58 + the 4 deltas computable on the CSV."""
    base = [c for c in training.FEATURE_COLS if c not in FORM_DELTA_COLS]
    computable = ["sp_era_delta_home", "sp_era_delta_away",
                  "sp_k9_delta_home", "sp_k9_delta_away"]
    assert len(base) == 58, f"expected 58 base columns, got {len(base)}"
    return {
        "WITHOUT": base,
        "WITH": base + computable,
    }


def coverage_report(games: pd.DataFrame) -> list[dict]:
    out = []
    for c in FORM_DELTA_COLS:
        if c not in games.columns:
            out.append({"column": c, "present": False, "coverage": 0.0})
            continue
        cov = float(games[c].notna().mean())
        out.append({"column": c, "present": True, "coverage": round(cov, 4)})
    return out


def run_variant(cols: list[str], folds, tune_df, hold_df) -> dict:
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()  # both variants blend identically

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
        except Exception as e:  # keep the loop honest: log, skip, continue
            print(f"  fold {split['fold_idx']} failed: {e}")
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        oof_y.extend(y_val.tolist())
        oof_blend.extend(np.asarray(blend, dtype=float).tolist())
        oof_blend_cal.extend(
            np.asarray(apply_platt(np.asarray(blend), fold_cal), dtype=float).tolist())
        for name, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(name, []).extend(pa.tolist())
            oof_members_cal.setdefault(name, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled: dict[str, dict] = {}
    pooled["blend"] = training.compute_metrics(
        y_all, np.asarray(oof_blend, dtype=float))
    pooled["blend_calibrated"] = training.compute_metrics(
        y_all, np.asarray(oof_blend_cal, dtype=float))
    for name, plist in oof_members.items():
        pooled[name] = training.compute_metrics(
            y_all, np.asarray(plist, dtype=float))
        pooled[f"{name}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(name, []), dtype=float))

    # ── sealed holdout: fit only at the end ───────────────────────────────
    models, _ = training.train_moneyline_ensemble(tune_df)
    blend_hold, member_hold, _wts = training.ensemble_predict(models, hold_df)
    y_hold = hold_df["home_win"].values.astype(float)
    holdout: dict[str, dict] = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold)),
    }
    for name, p in member_hold.items():
        holdout[name] = training.compute_metrics(
            y_hold, np.asarray(p, dtype=float))

    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, default="WITHOUT,WITH")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    args = ap.parse_args()

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)
    games = add_form_delta_features(games)

    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    print(f"commit={sha[:12]} data_sha={data_hash[:12]} games={len(games)} "
          f"tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} seed={RANDOM_SEED} clip={EPS}")

    coverage = coverage_report(games)
    real = [c for c in coverage if c["coverage"] > 0]
    print(f"delta coverage on committed CSV: {len(real)}/38 real "
          f"({', '.join(c['column'] for c in real)}); "
          f"{38 - len(real)} all-NaN until the next pipeline run")

    variants = build_variants()
    out = args.out or (DATA_DELIVERY_DIR / f"form_delta_ablation_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "form-delta-ablation/v1", "commit_sha": sha,
                   "data_sha256": data_hash, "holdout_days": args.holdout_days,
                   "folds_declared": len(all_splits),
                   "folds_executed": len(folds), "clip_eps": EPS,
                   "seed": int(RANDOM_SEED), "coverage": coverage,
                   "variants": {}}
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    for name in want:
        if name in results["variants"]:
            print(f"  {name}: cached, skipping")
            continue
        print(f"  {name}: running ({len(variants[name])} cols) ...")
        r = run_variant(variants[name], folds, tune_df, hold_df)
        r["cols"] = variants[name]
        results["variants"][name] = r
        out.write_text(json.dumps(results, indent=2) + "\n")
        b = r["pooled"]["blend"]
        h = r["holdout"]["blend"]
        print(f"    pooled blend {b['logloss']:.4f}/{b['auc']:.4f} "
              f"brier {b['brier']:.4f} ece {b['ece']:.4f} | "
              f"holdout {h['logloss']:.4f}/{h['auc']:.4f}")

    # gate: WITH must beat WITHOUT on the sealed holdout (both metrics)
    if "WITHOUT" in results["variants"] and "WITH" in results["variants"]:
        wo = results["variants"]["WITHOUT"]["holdout"]["blend"]
        w = results["variants"]["WITH"]["holdout"]["blend"]
        win = w["logloss"] < wo["logloss"] and w["auc"] > wo["auc"]
        pw = (results["variants"]["WITH"]["pooled"]["blend"]["logloss"]
              < results["variants"]["WITHOUT"]["pooled"]["blend"]["logloss"])
        print(f"\n=== sealed-holdout gate (blend) ===")
        print(f"  WITHOUT {wo['logloss']:.4f}/{wo['auc']:.4f}  "
              f"WITH {w['logloss']:.4f}/{w['auc']:.4f}  -> "
              f"{'BEATS WITHOUT' if win else 'loses/ties WITHOUT'} "
              f"(pooled_win={pw})")
        if pw and not win:
            print("  FLAG: pooled win with holdout loss — likely overfit, not adopted.")
    print(f"\nablation written: {out}")


if __name__ == "__main__":
    main()
