"""NFL per-side final-score engine — step-1 runner (candidate-only walk path).

Builds the two per-side mean regressors (mu_H on home_score, mu_A on
away_score — full-game final scores INCLUDING OT) on the moneyline's own
88-fold weekly geometry (pooled OOF 2021-24, sealed 2025). Record-only: no
wiring, no FEATURE_COLUMNS / 12-pool / moneyline / daily-pipeline change.

Outputs (all deterministic, same folds + seed):
- Per-game OOF predictions AND residuals for BOTH sides, keyed by game_id,
  persisted as data_delivery/nfl_per_side_oof_residuals_<frame_sha>.csv
  (the raw material for the joint covariance/variance/tie layer; the
  persistence guard fails loudly if the artifact is not written).
- data_delivery/nfl_per_side_<frame_sha>.json: per-side OOF + sealed
  MAE/CRPS vs naive baselines (per-team/season mean; home-field constant),
  family-recheck table (RF vs LGB vs XGB on the same folds — report-only,
  not a gate), coverage table, determinism pin (two identical walks
  byte-identical), residual-artifact location, scope pin.

CRPS note (Phase-1 finding): a point predictor's CRPS degenerates to MAE and
loses to the climatological full-distribution CRPS. Per-side CRPS here is
PRE-JOINT-LAYER — distributional CRPS only becomes meaningful after the
joint layer. Do not read these numbers as regression.

Usage:
    cd nfl-backend && python3 backend/run_nfl_per_side.py [--family lgb]
        [--skip-family-recheck] [--features <csv>] [--no-record]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_per_side_engine import (
    DEFAULT_FAMILY, FAMILIES, PRED_AWAY, PRED_COLS, PRED_HOME,
    RESID_AWAY, RESID_COLS, RESID_HOME, SIDE_FEATURES, SIDE_TARGETS,
    oof_per_side, persist_residuals, refit_per_side,
)
from nfl_moneyline import SEALED_SEASON, TRAIN_SEASONS, _valid_rows, generate_weekly_folds
from run_nfl_margin_ablation import _frame_sha256, load_features

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY / "nfl_game_level_features.csv"

SCOPE_PIN = (
    "Step 1 delivers per-side mean regressors + per-game OOF residuals. "
    "Joint layer — residual covariance between sides, per-side distribution "
    "family choice (NFL scores are not Poisson; NB-per-side like MLB is the "
    "wrong default — discretized-normal grid or similar is the candidate), "
    "discrete tie diagonal, and derived market probabilities with "
    "calibration — is the following build."
)


def _moneyline_folds(feats: pd.DataFrame) -> list[dict]:
    """The moneyline's OWN folds, exactly as the margin runner builds them:
    feature-valid 2019-24 rows → generate_weekly_folds (pooled OOF 2021-24)."""
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, SIDE_FEATURES)].copy()
    return generate_weekly_folds(preq_valid)


def _side_mae_crps(pred: np.ndarray, actual: np.ndarray) -> dict:
    """MAE + CRPS for a point forecast (CRPS == MAE by the degenerate-
    forecast identity — pre-joint-layer, documented in the record)."""
    err = np.abs(np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float))
    return {
        "mae": round(float(err.mean()), 3),
        "crps": round(float(err.mean()), 3),
        "n": int(len(err)),
    }


def _naive_team_season_mae(feats: pd.DataFrame, oof: pd.DataFrame) -> dict:
    """Constant per-team/season mean per side (descriptive benchmark on the
    same OOF-covered rows; includes the game itself — not a model)."""
    covered = feats[feats["game_id"].isin(set(oof["game_id"]))].copy()
    out = {}
    for side, col in SIDE_TARGETS.items():
        team_col = f"{side}_team"
        means = covered.groupby([team_col, "season"])[col].transform("mean")
        mae = float(np.abs(covered[col].astype(float) - means).mean())
        out[side] = {"mae": round(mae, 3), "n": int(len(covered))}
    return out


def _naive_homefield_mae(feats: pd.DataFrame, oof: pd.DataFrame) -> dict:
    """Home-field constant: global mean home score for the home side, global
    mean away score for the away side, on the same OOF-covered rows."""
    covered = feats[feats["game_id"].isin(set(oof["game_id"]))].copy()
    out = {}
    for side, col in SIDE_TARGETS.items():
        const = float(covered[col].astype(float).mean())
        mae = float(np.abs(covered[col].astype(float) - const).mean())
        out[side] = {"mae": round(mae, 3), "constant": round(const, 3),
                     "n": int(len(covered))}
    return out


def _climatological_crps(feats: pd.DataFrame, oof: pd.DataFrame) -> dict:
    """Climatological full-distribution CRPS (sigma / sqrt(pi)) per side on
    the covered rows — the reference a POINT forecast cannot beat."""
    covered = feats[feats["game_id"].isin(set(oof["game_id"]))].copy()
    out = {}
    for side, col in SIDE_TARGETS.items():
        sig = float(covered[col].astype(float).std(ddof=0))
        out[side] = {"sigma": round(sig, 4),
                     "naive_crps": round(sig / np.sqrt(np.pi), 3),
                     "n": int(len(covered))}
    return out


def _sealed_eval(feats: pd.DataFrame, rounds: dict,
                 family: str) -> tuple[pd.DataFrame, dict]:
    """Fit-only refill: fit 2019-24 at per-side median rounds, predict 2025.
    Returns (sealed frame with preds merged, per-side MAE/CRPS)."""
    preq = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    preq_valid = preq[_valid_rows(preq, SIDE_FEATURES)].copy()
    sld = feats[feats["season"] == SEALED_SEASON].copy()
    sld_valid = sld[_valid_rows(sld, SIDE_FEATURES)].copy()
    refit = refit_per_side(preq_valid, sld_valid, rounds, SIDE_FEATURES,
                           family=family)
    out = sld_valid.merge(refit, on="game_id", how="left")
    ev = {}
    for side, col in SIDE_TARGETS.items():
        m = out[[PRED_HOME if side == "home" else PRED_AWAY, col]].dropna()
        if len(m):
            ev[side] = _side_mae_crps(m.iloc[:, 0].to_numpy(float),
                                      m[col].to_numpy(float))
        else:
            ev[side] = {"mae": None, "crps": None, "n": 0}
    return out, ev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default=DEFAULT_FAMILY, choices=list(FAMILIES),
                    help="per-side regressor family (report-only recheck runs all)")
    ap.add_argument("--skip-family-recheck", action="store_true",
                    help="skip the RF/LGB/XGB family table (report-only)")
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    # Silence the lightgbm 4.7 sklearn eval_set deprecation (176 fits per walk).
    warnings.filterwarnings("ignore",
                            message="The argument 'eval_set' is deprecated")

    print("=" * 70)
    print("  NFL Per-Side Final-Score Engine — Step 1 (record-only)")
    print("=" * 70)

    t0 = time.time()
    feats = load_features(args.features)
    print(f"Loaded: {len(feats)} games, {feats['season'].min()}-{feats['season'].max()} "
          f"| frame_sha256={_frame_sha256(feats)}")

    folds = _moneyline_folds(feats)
    print(f"Moneyline folds: {len(folds)} (pooled OOF 2021-24, sealed {SEALED_SEASON})")

    # ---- Main family OOF walk + determinism pin (two identical walks) ----
    print(f"\n[1] OOF per-side walk (family={args.family})...")
    oof1, rounds, n_uncov = oof_per_side(folds, SIDE_FEATURES, feats,
                                         family=args.family)
    oof2, _r2, _u2 = oof_per_side(folds, SIDE_FEATURES, feats,
                                  family=args.family)
    a = oof1.sort_values("game_id").reset_index(drop=True)
    b = oof2.sort_values("game_id").reset_index(drop=True)
    deterministic = a.to_csv(index=False) == b.to_csv(index=False)
    print(f"  OOF n={len(oof1)} uncovered={n_uncov} rounds={rounds} "
          f"deterministic={deterministic}")

    # ---- Persist the residual artifact (loud-failure guard) ----
    frame_sha = hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]
    artifact = DATA_DELIVERY / f"nfl_per_side_oof_residuals_{frame_sha}.csv"
    persist_residuals(oof1, artifact)   # raises RuntimeError on any failure
    print(f"  Artifact: {artifact.name} ({len(oof1)} rows)")

    # Actuals for the eval (scores live on the decided frame, not the OOF
    # output); the artifact stays preds+residuals only.
    oof_ev_df = oof1.merge(
        feats[["game_id", "home_score", "away_score"]], on="game_id", how="left")

    # ---- Naive baselines + per-side MAE/CRPS on the covered OOF rows ----
    print("\n[2] Naive baselines (same covered rows)...")
    team_season = _naive_team_season_mae(feats, oof1)
    homefield = _naive_homefield_mae(feats, oof1)
    climato = _climatological_crps(feats, oof1)
    oof_ev = {}
    for side, col in SIDE_TARGETS.items():
        pred_col = PRED_HOME if side == "home" else PRED_AWAY
        oof_ev[side] = _side_mae_crps(oof_ev_df[pred_col].to_numpy(float),
                                      oof_ev_df[col].to_numpy(float))
        print(f"  {side}: model MAE/CRPS={oof_ev[side]['mae']} "
              f"vs team-season naive {team_season[side]['mae']} "
              f"vs homefield {homefield[side]['mae']} "
              f"| climatological CRPS {climato[side]['naive_crps']}")

    # ---- Sealed 2025 (fit-only refill) ----
    print(f"\n[3] Sealed {SEALED_SEASON} refill (fit-only, median rounds)...")
    sealed_frame, sealed_ev = _sealed_eval(feats, rounds, args.family)
    for side in SIDE_TARGETS:
        print(f"  {side}: sealed MAE/CRPS={sealed_ev[side]['mae']} "
              f"(n={sealed_ev[side]['n']})")

    # ---- Family recheck (report-only, same folds) ----
    family_table = {}
    if not args.skip_family_recheck:
        print("\n[4] Family recheck (RF vs LGB vs XGB, same folds — report-only)...")
        for fam in FAMILIES:
            if fam == args.family:
                fam_oof = oof_ev_df.copy()  # reuse the main walk (+ actuals)
            else:
                fam_oof, _fr, _fu = oof_per_side(folds, SIDE_FEATURES, feats,
                                                 family=fam)
                fam_oof = fam_oof.merge(
                    feats[["game_id", "home_score", "away_score"]],
                    on="game_id", how="left")
            row = {"family": fam, "n": int(len(fam_oof))}
            for side, col in SIDE_TARGETS.items():
                pred_col = PRED_HOME if side == "home" else PRED_AWAY
                m = _side_mae_crps(fam_oof[pred_col].to_numpy(float),
                                   fam_oof[col].to_numpy(float))
                row[f"mae_{side}"] = m["mae"]
                row[f"crps_{side}"] = m["crps"]
            family_table[fam] = row
            print(f"  {fam}: home MAE={row['mae_home']} away MAE={row['mae_away']}")

    # ---- Coverage (identical shape to the margin-engine record) ----
    pre_sealed = int((feats["season"] < SEALED_SEASON).sum())
    coverage = {
        "n_total": int(len(feats)),
        "n_pre_sealed_2019_24": pre_sealed,
        "n_sealed_2025": int(len(feats) - pre_sealed),
        "n_oof_covered": int(len(oof1)),
        "pct_covered_pre_sealed": round(len(oof1) / max(pre_sealed, 1) * 100, 1),
        "n_uncovered_pre_sealed": int(pre_sealed - len(oof1)),
        "n_sealed_refill": int(sealed_ev["home"]["n"]),
        "pct_covered_sealed": round(
            sealed_ev["home"]["n"] / max(len(feats) - pre_sealed, 1) * 100, 1),
        "imputation": ("uncovered = 2019-20 warmup (never in a fold's val "
                       "window) + playoff weeks with <5 val games → absent "
                       "from the residual artifact → NaN → tree-native "
                       "routing / train-median downstream; sealed 2025 = "
                       "fit-only refill at per-side median rounds (100% of "
                       "valid rows)"),
    }

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    record = {
        "record": "nfl_per_side_step1",
        "frame_sha": frame_sha,
        "frame_sha256": _frame_sha256(feats),
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "supersedes": "standalone margin-only sigma layer (dropped — the "
                      "per-side engine is the end-state: one joint "
                      "distribution prices spread, moneyline, totals, tie mass)",
        "scope": SCOPE_PIN,
        "geometry": {
            "seasons": sorted(feats["season"].unique().tolist()),
            "train_seasons": TRAIN_SEASONS,
            "sealed_season": SEALED_SEASON,
            "n_folds": len(folds),
            "targets": {"home": "home_score (full game incl. OT)",
                        "away": "away_score (full game incl. OT)"},
            "view": SIDE_FEATURES,
            "family": args.family,
            "family_is_parameter": True,
        },
        "coverage": coverage,
        "per_side_oof": {
            "model": oof_ev,
            "naive_team_season_mean": team_season,
            "naive_homefield_constant": homefield,
            "climatological_crps_reference": climato,
            "crps_note": ("point-forecast CRPS == MAE (degenerate); "
                          "climatological CRPS = sigma/sqrt(pi) full "
                          "distribution — a point predictor loses to it by "
                          "construction (Phase-1 finding). Per-side CRPS is "
                          "PRE-JOINT-LAYER; distributional CRPS is only "
                          "meaningful after the joint layer."),
        },
        "per_side_sealed": sealed_ev,
        "determinism_pin": {
            "two_identical_walks_byte_identical": bool(deterministic),
            "method": "two OOF walks, same folds + seed, CSV-bytes equal",
        },
        "family_recheck": {
            "note": ("report-only, NOT a gate — same folds, per-side targets "
                     "(integer support, floor at 0, right skew differ from "
                     "the margin target); standing interim decision = lgb"),
            "table": family_table,
        },
        "residual_artifact": {
            "path": str(artifact.relative_to(ROOT_DIR)),
            "n_rows": int(len(oof1)),
            "columns": ["game_id", "fold_idx"] + [PRED_HOME, PRED_AWAY,
                                                  RESID_HOME, RESID_AWAY]
                       + ["best_iter_home", "best_iter_away"],
            "guard": "persist_residuals raises RuntimeError if the artifact "
                     "cannot be written (missing columns / empty frame / "
                     "write failure / missing file after write)",
        },
        "feature_columns_untouched": True,
        "judgment_calls": {
            "targets_include_ot": ("full-game final scores INCLUDING OT so "
                                   "totals/spread calibration matches "
                                   "settlement; regulation-only targets only "
                                   "if P(OT) becomes a separate upset "
                                   "diagnostic — not in scope"),
            "family": "LightGBM interim (standing decision); family is a "
                      "parameter, rechecked on this target report-only",
            "no_moneyline_ablation": ("candidate-only walk path this step — "
                                      "per-side means are a feature producer, "
                                      "not a moneyline candidate yet"),
        },
    }

    if not args.no_record:
        record_path = DATA_DELIVERY / f"nfl_per_side_{frame_sha}.json"
        record_path.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {record_path.name}")
    else:
        print("\n[--no-record] record skipped")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())