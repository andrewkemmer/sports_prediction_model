"""Lineup-delta ablation — WITH vs WITHOUT on the moneyline ensemble.

Phase 2 measurement (NOT tuning): does the actual starting-9 wOBA delta set
(lineup_actual_woba_delta / lineup_actual_top3_delta / lineup_rest_count, per
side) help the ensemble out of sample? Nothing is changed silently; the feature
set ships only if it clears the sealed-holdout gate.

Design (mirrors the locked baseline conventions exactly — same as the
momentum ablation harness):
- Data: committed data_delivery/game_level_features.csv (6,992 games, hash
  recorded). Decided games only. add_lineup_delta_features() computes the 6
  columns from data_delivery/lineups.parquet (StatsAPI battingOrder, 100%
  of decided games) + batter/team point-in-time sd-wOBA tables — ALL SIX are
  real on this artifact.
- Variants:
    WITHOUT  = production FEATURE_COLS (the shipped baseline; the 6 stay OUT)
    WITH     = baseline + the 6 lineup-delta columns, trained on REAL actuals
    WITHMASK = baseline + the 6, but the 6 are randomly NULLed on ~mask_rate
               of TRAINING rows (posting-curve simulation of "lineup not yet
               posted at bet time"). The model is forced to learn BOTH
               regimes: actuals-present and lineups-missing. Serve already
               emits NULL when lineups aren't posted, so this aligns the
               training distribution with the deployment distribution.
- Folds: walk_forward_splits on the tuning pool with RETRAIN_CADENCE_DAYS,
  filtered by MIN_VAL_FOLD_GAMES (declared vs executed recorded).
- Members: all 5 (xgb/lgbm/rf/logistic/mlp) + the static-prior blend
  (adaptive weights cleared before each variant so both blend identically).
- Metrics: compute_metrics (clip 1e-7) — logloss / AUC / Brier / ECE — raw
  and prequential-calibrated (fit_platt on prior folds' blend pairs only).
- Sealed 21-day holdout: refit fit-only on the whole tuning pool AFTER the
  fold loop; never touched during fold fitting. Each variant is evaluated on
  the holdout TWICE:
      holdout (real)         — lineups available (the within-90-min regime)
      holdout_projected      — all 6 cols forced NULL (the hours-before regime)

TWO-SIDED GATE (2026-08): a feature ships only if it earns its keep in BOTH
regimes the user actually experiences:
  BOOST leg      — must BEAT WITHOUT on the real-actual holdout (logloss AND
                   AUC): when lineups are posted, the feature must help.
  NO-PENALTY leg — must NOT be worse than WITHOUT on the projected-only
                   holdout (within --no-penalty-tol): when lineups aren't
                   posted, the feature must cost nothing.
ECE/calibration and per-member collapse are FLAGGED in caveats but NOT part
of the verdict (policy, 2026-08): the verdict stays on the decision metrics.

Emits data_delivery/lineup_ablation_mask_<sha>.json with --mask-train
(incremental — resumes by skipping variants already present). COMMITS NOTHING.

Usage:
    python run_lineup_ablation.py                        # legacy WITHOUT/WITH
    python run_lineup_ablation.py --mask-train           # 3 arms, two-sided gate
    python run_lineup_ablation.py --mask-train --variants WITHMASK
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
    """WITHOUT = production FEATURE_COLS; WITH/WITHMASK = it + the 6 cols.

    Invariant: the lineup-delta family is REMOVED from production (train-serve
    skew fix), so WITHOUT is exactly the current FEATURE_COLS, and WITH re-adds
    the 6. A frozen count pin went stale as the frame gained run_margin_diff and
    friends, so we assert the structural invariant instead of a column count.
    WITHMASK uses the same columns as WITH but trains with the availability
    mask (see run_variant)."""
    base = [c for c in training.FEATURE_COLS if c not in LINEUP_DELTA_COLS]
    assert base == training.FEATURE_COLS, (
        "FEATURE_COLS unexpectedly contains lineup_actual_* columns that were "
        "removed for train-serve skew; the ablation compares a feature set "
        "that is no longer what ships")
    assert len(LINEUP_DELTA_COLS) == 6
    return {
        "WITHOUT": base,
        "WITH": base + LINEUP_DELTA_COLS,
        "WITHMASK": base + LINEUP_DELTA_COLS,
    }


def coverage_report(games: pd.DataFrame) -> list[dict]:
    out = []
    for c in LINEUP_DELTA_COLS:
        if c not in games.columns:
            out.append({"column": c, "present": False, "coverage": 0.0})
            continue
        cov = float(games[c].notna().mean())
        out.append({"column": c, "present": True, "coverage": round(cov, 4)})
    return out


def _mask_lineups(df: pd.DataFrame, rng, mask_rate: float) -> pd.DataFrame:
    """Randomly NULL the 6 lineup-delta cols on a COPY (training regime sim).

    Simulates "lineup not yet posted at bet time": each row is NULLed with
    probability ``mask_rate``. The rows themselves stay in training with all
    other features and outcomes — only the 6 cols go missing, so the model
    learns the no-lineup regime without losing games. float NaN (not pd.NA)
    because .to_numpy(dtype=float) rejects pandas' NAType."""
    out = df.copy()
    lc = [c for c in LINEUP_DELTA_COLS if c in out.columns]
    if not lc:
        return out
    hit = rng.random(len(out)) < mask_rate
    if not hit.any():
        return out
    for c in lc:
        out.loc[hit, c] = np.nan
    return out


def _projected_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Force ALL 6 cols NULL on a copy — the bettor has no lineups."""
    out = df.copy()
    for c in LINEUP_DELTA_COLS:
        if c in out.columns:
            out[c] = np.nan
    return out


def _eval(models, df: pd.DataFrame, name: str) -> dict:
    blend, member_probs, _wts = training.ensemble_predict(models, df)
    y = df["home_win"].values.astype(float)
    metrics = {"blend": training.compute_metrics(y, np.asarray(blend))}
    for n, p in member_probs.items():
        metrics[n] = training.compute_metrics(y, np.asarray(p, dtype=float))
    return metrics


def run_variant(cols: list[str], folds, tune_df, hold_df,
                mask_train: bool = False, mask_rate: float = 0.5) -> dict:
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
        if mask_train:
            # Availability regime sim on the TRAIN split only — val keeps real
            # actuals (pooled OOF measures the lineups-available regime).
            train = _mask_lineups(
                train, np.random.default_rng(RANDOM_SEED + split["fold_idx"]),
                mask_rate)
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
        for n, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(n, []).extend(pa.tolist())
            oof_members_cal.setdefault(n, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled: dict[str, dict] = {}
    pooled["blend"] = training.compute_metrics(
        y_all, np.asarray(oof_blend, dtype=float))
    pooled["blend_calibrated"] = training.compute_metrics(
        y_all, np.asarray(oof_blend_cal, dtype=float))
    for n, plist in oof_members.items():
        pooled[n] = training.compute_metrics(y_all, np.asarray(plist, dtype=float))
        pooled[f"{n}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(n, []), dtype=float))

    # ── sealed holdout: fit only at the end ───────────────────────────────
    refit_tune = tune_df
    if mask_train:
        refit_tune = _mask_lineups(
            tune_df, np.random.default_rng(RANDOM_SEED), mask_rate)
    models, _ = training.train_moneyline_ensemble(refit_tune)
    holdout = _eval(models, hold_df, "holdout")
    # The same sealed model seen through the projected-only lens: the bettor
    # has NO lineups. This is the no-penalty leg.
    holdout_projected = _eval(models, _projected_holdout(hold_df),
                              "holdout_projected")

    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout,
            "holdout_projected": holdout_projected}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--mask-train", action="store_true",
                    help="run the mixed-availability arm (WITHMASK): randomly "
                         "NULL the 6 cols on ~mask_rate of TRAINING rows so the "
                         "model learns the lineups-not-posted regime; evaluate "
                         "every variant on BOTH holdouts and apply the two-sided "
                         "gate (boost + no-penalty).")
    ap.add_argument("--mask-rate", type=float, default=0.5,
                    help="probability a training row's 6 lineup cols are NULLed "
                         "in the WITHMASK arm (default 0.5).")
    ap.add_argument("--no-penalty-tol", type=float, default=0.001,
                    help="allowed degradation vs WITHOUT on the projected-only "
                         "holdout for the no-penalty leg (logloss and AUC).")
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

    print(f"commit={sha[:12]} data_sha={data_hash[:12]} games={len(games)} "
          f"tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} seed={RANDOM_SEED} clip={EPS} "
          f"mask_train={args.mask_train} mask_rate={args.mask_rate}")

    coverage = coverage_report(games)
    real = [c for c in coverage if c["coverage"] > 0]
    print(f"lineup-delta coverage on committed CSV: {len(real)}/6 real "
          f"({', '.join(c['column'] for c in real)})")

    variants = build_variants()
    if args.mask_train:
        out = args.out or (DATA_DELIVERY_DIR
                           / f"lineup_ablation_mask_{sha[:12]}.json")
        want = (args.variants.split(",") if args.variants
                else ["WITHOUT", "WITH", "WITHMASK"])
    else:
        out = args.out or (DATA_DELIVERY_DIR / f"lineup_ablation_{sha[:12]}.json")
        want = (args.variants.split(",") if args.variants
                else ["WITHOUT", "WITH"])
    want = [v.strip() for v in want if v.strip()]
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {"schema": "lineup-ablation/v2", "commit_sha": sha,
                   "data_sha256": data_hash, "holdout_days": args.holdout_days,
                   "folds_declared": len(all_splits),
                   "folds_executed": len(folds), "clip_eps": EPS,
                   "seed": int(RANDOM_SEED), "coverage": coverage,
                   "mask_train": args.mask_train, "mask_rate": args.mask_rate,
                   "no_penalty_tol": args.no_penalty_tol,
                   "variants": {}}

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
        b = r["pooled"]["blend"]
        h = r["holdout"]["blend"]
        hp = r["holdout_projected"]["blend"]
        print(f"    pooled blend {b['logloss']:.4f}/{b['auc']:.4f} "
              f"brier {b['brier']:.4f} ece {b['ece']:.4f} | "
              f"holdout {h['logloss']:.4f}/{h['auc']:.4f} | "
              f"holdout_projected {hp['logloss']:.4f}/{hp['auc']:.4f}")

    # ── two-sided gate ────────────────────────────────────────────────────
    # BOOST: beat WITHOUT on the real-actual holdout (logloss AND AUC).
    # NO-PENALTY: not worse than WITHOUT on the projected-only holdout
    # (within tol). ECE/calibration + member collapse are FLAGGED only.
    if "WITHOUT" in results["variants"]:
        wo = results["variants"]["WITHOUT"]["holdout"]["blend"]
        wo_proj = (results["variants"]["WITHOUT"].get("holdout_projected", {})
                   .get("blend", wo))
        candidates = [n for n in ("WITH", "WITHMASK")
                      if n in results["variants"]]
        gate_cands: dict[str, dict] = {}
        for name in candidates:
            wi = results["variants"][name]["holdout"]["blend"]
            wi_proj = (results["variants"][name].get("holdout_projected", {})
                       .get("blend"))
            boost = (wi["logloss"] < wo["logloss"]) and (wi["auc"] > wo["auc"])
            no_penalty = None  # unknown if the file predates projected evals
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
            pwo = results["variants"]["WITHOUT"].get("pooled", {})
            pwi = results["variants"][name].get("pooled", {})
            for m in ("logistic", "mlp"):
                if m in pwo and m in pwi:
                    base_ll = pwo[m]["logloss"]
                    new_ll = pwi[m]["logloss"]
                    if new_ll > base_ll + 0.02:
                        caveats.append(
                            f"{m} pooled logloss collapsed "
                            f"{base_ll:.4f} -> {new_ll:.4f} (linear/DL "
                            "overfit marker)")
            gate_cands[name] = {
                "verdict": verdict,
                "boost": boost,
                "no_penalty": no_penalty,
                "holdout_without": wo,
                "holdout_with": wi,
                "holdout_projected_with": wi_proj,
                "caveats": caveats,
            }
            print(f"GATE[{name}]: boost holdout {wo['logloss']:.4f}/"
                  f"{wo['auc']:.4f} vs {wi['logloss']:.4f}/{wi['auc']:.4f} "
                  f"-> {'WINS' if boost else 'LOSES'}; no-penalty projected "
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