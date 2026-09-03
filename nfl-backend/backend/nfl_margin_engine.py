"""NFL margin engine Phase 1 — single LightGBM regressor on point margin.

Mirror of the MLB run-engine's OOF-attach pattern (build_oof_margin.py),
simplified for NFL:

  Target:  home_margin = home_score − away_score  (continuous, ~N(0, σ≈14))
  Model:   single deterministic LightGBM regressor (objective='regression')
  Features: the 12-pool (FEATURE_COLUMNS minus the constant is_home anchor),
            identical to the moneyline's served view
  Folds:   weekly prequential over 2019–2024 (supplied by the moneyline
           harness); sealed 2025 = fit-only refit on 2019–2024 at the median
           fold round count
  Output:  pt_margin_diff — decorrelated from the binary win target by
           construction (continuous margin vs binary win, different model)

Design contract (Phase-1 spec, 2026-09-03):
- Folds are SUPPLIED BY THE CALLER — the moneyline harness's walk-forward
  splits (same weekly geometry, same VAL_SEASONS [2021–2024]). This module
  NEVER mutates those fold DataFrames (READ-ONLY producer).
- READ-ONLY: this module produces margin predictions but never consumes
  moneyline outputs; the moneyline's feature matrix gains pt_margin_diff as
  an input column. The two models never share targets, weights, or
  validation windows.
- Phase-1 view = the 12-pool as-is (cheapest clean first answer). No model
  outputs or probabilities in it — decorrelation comes from the different
  target (continuous margin vs binary win), the same reason MLB's
  heavily-overlapping run-engine view still adds signal. Phase-2 extension
  (NOT here): per-side offense/defense EWM splits.
- Uncovered rows (pre-2021 warmup folds → no fold ever validates them) →
  NaN → tree-native routing / train-fold-median imputation for the
  logistic+MLP members. Never zero-filled.
- Sealed 2025 = fit on 2019–2024, predict 2025. Slate 2026 = fit on all
  2019–2025 at final/median rounds → 100% coverage of board rows (fit-only
  refill, same as MLB's predict_slate_runs convention).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MARGIN_COL = "pt_margin_diff"
TARGET_MARGIN = "_home_margin"  # internal; home_score − away_score
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50

# Phase-1 view: the 12-pool (FEATURE_COLUMNS minus the constant is_home
# anchor, which the moneyline never feeds as a model column). No exclusions,
# no self-referencing (pt_margin_diff is NOT in this list).
MARGIN_FEATURES = [f for f in FEATURE_COLUMNS if f != "is_home"]


# ── Fold-aligned OOF margins ──────────────────────────────────────────────────

def _fit_regressor(params: dict, X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray
                   ) -> tuple[Any, np.ndarray, int]:
    """Train one LightGBM regressor, predict on val. Returns (model, preds, best_iter)."""
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    model = LGBMRegressor(**{**params, "n_estimators": MAX_ROUNDS})
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   log_evaluation(period=0)],
    )
    best = int(model.best_iteration_ or MAX_ROUNDS)
    preds = model.predict(X_va, num_iteration=best)
    return model, preds, best


def oof_margins(folds: list[dict], features: list[str],
                games: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
    """Compute out-of-fold margin predictions on the caller's folds.

    Args:
        folds: list of {week_start, train: DataFrame, val: DataFrame} —
               the moneyline harness's own weekly splits (never mutated).
        features: feature columns to use (Phase-1: the 12-pool).
        games: full decided frame (for the uncovered-game count).

    Returns:
        margins_df with [game_id, fold_idx, pt_margin_diff, best_iteration],
        median_best_rounds ({home, away} — kept for MLB parity), and
        n_uncovered_games (decided games in NO fold's val window → NaN →
        tree-native routing / train-median imputation downstream).

    Leakage assertion: max(train.gameday) < min(val.gameday) per fold, else
    AssertionError — mirror of MLB's oof_run_margins fold-boundary guard.
    Deterministic: identical folds + seed → identical table.
    """
    # Local copies only — the caller's fold DataFrames are READ-ONLY.
    games = games.copy()
    if TARGET_MARGIN not in games.columns:
        games[TARGET_MARGIN] = (games["home_score"].astype(float)
                                - games["away_score"].astype(float))

    # Intersect with what's actually available (never silently all-NaN).
    available = [f for f in features if f in games.columns]
    missing = [f for f in features if f not in games.columns]
    if missing:
        logger.warning("margin engine: %d requested feature(s) absent (%s%s)",
                       len(missing), ", ".join(missing[:6]),
                       " …" if len(missing) > 6 else "")
    if not available:
        raise ValueError("margin engine: no usable feature columns supplied")

    parts = []
    best_iters = []

    for i, f in enumerate(folds):
        tr = f["train"].copy()
        va = f["val"].copy()
        for df in (tr, va):
            if TARGET_MARGIN not in df.columns:
                df[TARGET_MARGIN] = (df["home_score"].astype(float)
                                     - df["away_score"].astype(float))

        # Leakage assertion: max(train.gameday) < min(val.gameday)
        tr_max = pd.to_datetime(tr["gameday"]).max()
        va_min = pd.to_datetime(va["gameday"]).min()
        if not (tr_max < va_min):
            raise AssertionError(
                f"fold {i}: train max {tr_max} not strictly before "
                f"val min {va_min} → leakage-safe split violated")

        id_cols = [c for c in ["game_id", "gameday"] if c in va.columns]
        tr_valid = tr[available + [TARGET_MARGIN]].dropna()
        va_valid = va[id_cols + available + [TARGET_MARGIN]].dropna()

        if len(tr_valid) < 30 or len(va_valid) < 5:
            logger.warning("margin engine: fold %d too small (tr=%d, va=%d), skipping",
                           i, len(tr_valid), len(va_valid))
            continue

        X_tr = tr_valid[available].to_numpy(float)
        y_tr = tr_valid[TARGET_MARGIN].to_numpy(float)
        X_va = va_valid[available].to_numpy(float)
        y_va = va_valid[TARGET_MARGIN].to_numpy(float)

        _model, preds, best = _fit_regressor(LGB_PARAMS, X_tr, y_tr, X_va, y_va)
        best_iters.append(best)

        rec = {
            "game_id": va_valid["game_id"].values,
            "fold_idx": i,
            MARGIN_COL: np.round(preds, 4),
            "best_iteration": best,
        }
        parts.append(pd.DataFrame(rec))

    margins = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["game_id", "fold_idx", MARGIN_COL, "best_iteration"])
    median_rounds = int(np.median(best_iters)) if best_iters else MAX_ROUNDS

    # Uncovered count is computed on the FULL decided frame (warmup games are
    # never in any fold's val window; small skipped folds add their games).
    covered = set(margins["game_id"]) if len(margins) else set()
    n_uncovered = int((~games["game_id"].isin(covered)).sum())

    if len(margins) == 0:
        logger.warning("margin engine: no folds produced margins")

    return margins, {"home": median_rounds, "away": median_rounds}, n_uncovered


def refit_margins(decided: pd.DataFrame, pred_df: pd.DataFrame,
                  n_rounds: int, features: list[str]) -> pd.DataFrame:
    """Fit-only refit on all decided games at fixed round count, predict pred_df.

    Used for sealed-holdout evaluation (fit 2019–2024 → predict 2025) and
    slate inference (fit all decided → predict board rows). Every predicted
    game is strictly future relative to the fit — cannot leak by
    construction (mirror of MLB's refit_run_margins).
    """
    decided = decided.copy()
    decided[TARGET_MARGIN] = (decided["home_score"].astype(float)
                              - decided["away_score"].astype(float))

    available = [f for f in features if f in decided.columns
                 and f in pred_df.columns]
    missing = [f for f in features if f not in available]
    if missing:
        logger.warning("margin engine refit: %d requested feature(s) absent (%s)",
                       len(missing), ", ".join(missing[:6]))

    valid_decided = decided[available + [TARGET_MARGIN]].dropna()
    valid_pred = pred_df[["game_id"] + available].dropna()

    X_tr = valid_decided[available].to_numpy(float)
    y_tr = valid_decided[TARGET_MARGIN].to_numpy(float)

    from lightgbm import LGBMRegressor, log_evaluation
    model = LGBMRegressor(**{**LGB_PARAMS, "n_estimators": int(n_rounds)})
    model.fit(X_tr, y_tr, callbacks=[log_evaluation(period=0)])

    X_pred = valid_pred[available].to_numpy(float)
    preds = model.predict(X_pred)

    result = valid_pred[["game_id"]].copy()
    result[MARGIN_COL] = np.round(preds, 4)
    return result


# ── Convenience: full OOF + sealed pipeline ──────────────────────────────────

def run_margin_oof(folds: list[dict], features: list[str],
                   games: pd.DataFrame,
                   sealed: pd.DataFrame | None = None
                   ) -> dict[str, Any]:
    """Full margin engine walk-forward + optional sealed evaluation.

    Returns dict with oof margins, sealed margins, coverage, median rounds.
    """
    oof, rounds, n_uncov = oof_margins(folds, features, games)
    n_rounds = rounds.get("home", MAX_ROUNDS)

    sealed_margins = pd.DataFrame()
    if sealed is not None and len(sealed) > 0:
        sealed_margins = refit_margins(games, sealed, n_rounds, features)

    coverage = {
        "n_oof_games": len(oof),
        "n_uncovered": n_uncov,
        "pct_oof": round(len(oof) / max(len(games), 1) * 100, 1),
        "n_sealed": len(sealed_margins),
    }

    return {
        "oof": oof,
        "sealed": sealed_margins,
        "rounds": rounds,
        "coverage": coverage,
        "features": features,
    }