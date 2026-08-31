"""Untapped-columns ablation — full ensemble, WITH vs WITHOUT on the moneyline.

Tests columns that are computed and SHIPPED in the frame but absent from
FEATURE_COLS (no new data needed):
    lineup_ops_vs_l_home/away, lineup_ops_vs_r_home/away,
    lineup_ops_vs_starter_hand_home/away, park_wind_factor
(lineup_handedness_matchup_advantage — already in the model — is built from the
starter-hand raws; this adds the raw per-side values alongside it, the same
pattern the model uses for other per-side raws.)

Full 5-member ensemble + static-prior blend, identical to the locked baseline:
    WITHOUT = current production FEATURE_COLS
    WITH    = baseline + the 7 untapped columns
Per-variant pooled OOF (all folds) raw + prequential-calibrated, per member;
sealed-21-day holdout (refit fit-only on the whole tuning pool, never touched
during folds).

Gate (task rule): ship ONLY if WITH clears the sealed holdout on logloss AND
AUC; a pooled win with a holdout loss is flagged overfit and never adopted.
ECE/calibration and per-member collapse are flagged in caveats, NOT part of
the verdict (policy, 2026-08).

Emits data_delivery/lineup_ablation_untapped_<sha>.json (incremental;
resumes by skipping variants already present). COMMITS NOTHING.
    python run_untapped_columns_ablation.py
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
from config import (  # noqa: E402
    DATA_DELIVERY_DIR, MIN_VAL_FOLD_GAMES, RANDOM_SEED, RETRAIN_CADENCE_DAYS,
)
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402

EPS = 1e-7

# Untapped columns computed/shipped but absent from FEATURE_COLS.
UNTAPPED_COLS = [
    "lineup_ops_vs_l_home", "lineup_ops_vs_l_away",
    "lineup_ops_vs_r_home", "lineup_ops_vs_r_away",
    "lineup_ops_vs_starter_hand_home", "lineup_ops_vs_starter_hand_away",
    "park_wind_factor",
]


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


def build_variants(add: list[str]) -> dict[str, list[str]]:
    base = [c for c in training.FEATURE_COLS if c not in add]
    assert base == training.FEATURE_COLS, (
        "an untapped column is already in FEATURE_COLS; it is not untapped")
    return {"WITHOUT": base, "WITH": base + add}


def coverage_report(games: pd.DataFrame) -> dict:
    out = {}
    for c in UNTAPPED_COLS:
        if c not in games.columns:
            out[c] = {"present": False, "coverage": 0.0}
        else:
            out[c] = {"present": True,
                      "coverage": round(float(games[c].notna().mean()), 4)}
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
        oof_blend_cal.extend(
            np.asarray(apply_platt(np.asarray(blend), fold_cal), dtype=float).tolist())
        for n, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(n, []).extend(pa.tolist())
            oof_members_cal.setdefault(n, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled: dict[str, dict] = {
        "blend": training.compute_metrics(y_all, np.asarray(oof_blend, dtype=float)),
        "blend_calibrated": training.compute_metrics(
            y_all, np.asarray(oof_blend_cal, dtype=float)),
    }
    for n, plist in oof_members.items():
        pooled[n] = training.compute_metrics(y_all, np.asarray(plist, dtype=float))
        pooled[f"{n}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(n, []), dtype=float))

    models, _ = training.train_moneyline_ensemble(tune_df)
    blend_hold, member_hold, _wts = training.ensemble_predict(models, hold_df)
    y_hold = hold_df["home_win"].values.astype(float)
    holdout: dict[str, dict] = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold))}
    for n, p in member_hold.items():
        holdout[n] = training.compute_metrics(y_hold, np.asarray(p, dtype=float))

    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--add", type=str, default=",".join(UNTAPPED_COLS),
                    help="comma-separated columns to add in the WITH arm")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    args = ap.parse_args()

    add = [c.strip() for c in args.add.split(",") if c.strip()]
    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)

    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    missing = [c for c in add if c not in games.columns]
    print(f"untapped ablation commit={sha[:12]} data_sha={data_hash[:12]} "
          f"games={len(games)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} seed={RANDOM_SEED} "
          f"add={len(add)} ({', '.join(add)})")
    if missing:
        print(f"  WARNING: not present in frame (auto-filled NULL): {missing}")
    print("untapped coverage:", coverage_report(games))

    variants = build_variants(add)
    out = args.out or (DATA_DELIVERY_DIR
                       / f"lineup_ablation_untapped_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "untapped-cols-ablation/v1", "commit_sha": sha,
                   "data_sha256": data_hash, "holdout_days": args.holdout_days,
                   "folds_declared": len(all_splits),
                   "folds_executed": len(folds), "clip_eps": EPS,
                   "seed": int(RANDOM_SEED), "add": add,
                   "coverage": coverage_report(games), "variants": {}}

    for name in ("WITHOUT", "WITH"):
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

    if "WITHOUT" in results["variants"] and "WITH" in results["variants"]:
        wo = results["variants"]["WITHOUT"]["holdout"]["blend"]
        wi = results["variants"]["WITH"]["holdout"]["blend"]
        wins = (wi["logloss"] < wo["logloss"]) and (wi["auc"] > wo["auc"])
        caveats: list[str] = []
        if wo.get("ece") is not None and wi.get("ece") is not None:
            if wi["ece"] > wo["ece"]:
                caveats.append(
                    f"holdout calibration declined (ece {wo['ece']:.4f} -> "
                    f"{wi['ece']:.4f}, delta +{wi['ece'] - wo['ece']:.4f})")
        pwo = results["variants"]["WITHOUT"].get("pooled", {})
        pwi = results["variants"]["WITH"].get("pooled", {})
        for m in ("logistic", "mlp"):
            if m in pwo and m in pwi:
                if pwi[m]["logloss"] > pwo[m]["logloss"] + 0.02:
                    caveats.append(
                        f"{m} pooled logloss collapsed {pwo[m]['logloss']:.4f} "
                        f"-> {pwi[m]['logloss']:.4f}")
        results["gate"] = {
            "verdict": "SHIP" if wins else "DON'T SHIP",
            "holdout_without": wo, "holdout_with": wi,
            "ece_excluded_from_verdict": True, "caveats": caveats,
        }
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"GATE: holdout WITHOUT {wo['logloss']:.4f}/{wo['auc']:.4f} vs "
              f"WITH {wi['logloss']:.4f}/{wi['auc']:.4f} -> "
              f"{results['gate']['verdict']}")
        if caveats:
            print("CAVEATS (not part of verdict):")
            for c in caveats:
                print(f"  - {c}")


if __name__ == "__main__":
    main()