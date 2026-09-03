"""NFL per-side final-score engine — step 1: per-side mean regressors.

Goal = final score per team (regulation or OT), so the coherent per-side
engine is the end-state: one joint distribution prices spread, moneyline,
totals, and tie mass. The standalone margin-only sigma layer is DROPPED
(superseded by this engine). THIS STEP builds the two per-side mean
regressors (mu_H on home_score, mu_A on away_score, full-game final scores
INCLUDING OT — matches how every market settles); the joint layer (residual
covariance between sides, per-side distribution family choice, discrete tie
diagonal, derived market probabilities + calibration) is the FOLLOWING build.

Geometry is identical to the Phase-1 margin engine (commit 4e766d3), keeping
every surface comparable:
- Same 12-pool PIT view (FEATURE_COLUMNS minus the constant is_home anchor).
- Same 88-fold weekly geometry: pooled OOF 2021-24, sealed 2025.
- Same fold-aligned leakage guard (max(train.gameday) < min(val.gameday),
  else AssertionError), mirror of MLB's oof_run_margins boundary guard.
- Same imputation discipline: uncovered rows (2019-20 warmup, never in any
  fold's val window; small skipped folds) → NaN → tree-native routing /
  train-median imputation downstream. Never zero-filled.
- Same fit-only refill at median rounds: sealed 2025 = fit on 2019-2024 at
  the median fold round count per side; slate = fit on all decided games.

Model family is a PARAMETER (``family`` in {"lgb", "rf", "xgb"}), never
hardcoded — the standing interim decision is LightGBM, re-measured on this
new target (integer support, floor at 0, right skew — different from the
margin target) before locking.

Per-game OOF predictions AND residuals are PERSISTED keyed by game_id for
BOTH sides — the raw material for the covariance/variance/tie layer. The
persistence guard raises loudly if the artifact cannot be written.

CRPS note: a point predictor's CRPS degenerates to MAE (and loses to the
climatological full-distribution CRPS, the Phase-1 finding). Per-side CRPS
here is therefore PRE-JOINT-LAYER — distributional CRPS is only meaningful
after the joint layer. Do not read these numbers as regression.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Same 12-pool PIT view as the margin engine (FEATURE_COLUMNS minus the
# constant is_home anchor, which the moneyline never feeds as a model column).
SIDE_FEATURES = [f for f in FEATURE_COLUMNS if f != "is_home"]

# Output column names (never collide with the served pool).
PRED_HOME = "pred_home"
PRED_AWAY = "pred_away"
RESID_HOME = "resid_home"
RESID_AWAY = "resid_away"

# Targets: full-game final scores INCLUDING OT (how every market settles).
SIDE_TARGETS = {"home": "home_score", "away": "away_score"}

FAMILIES = ("lgb", "rf", "xgb")
DEFAULT_FAMILY = "lgb"

MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50

# Family hyperparameters — read at call time so tests can shrink them.
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
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 5,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "mae",
    "tree_method": "hist",
}

RESID_COLS = [RESID_HOME, RESID_AWAY]
PRED_COLS = [PRED_HOME, PRED_AWAY]


# ── Per-side regressor fit (family as a parameter) ───────────────────────────

def _fit_side(family: str, X_tr: np.ndarray, y_tr: np.ndarray,
              X_va: np.ndarray, y_va: np.ndarray
              ) -> tuple[Any, np.ndarray, int]:
    """Train one per-side regressor of ``family``, predict on val.

    Returns (model, preds, best_iter). Boosting families early-stop on the
    fold's own strictly-future val rows (the moneyline fold loop's
    convention); RF has no early stopping and reports its full round count.
    Deterministic per family (fixed seeds, hist tree method for XGB).
    """
    family = (family or DEFAULT_FAMILY).strip().lower()
    if family == "lgb":
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        model = LGBMRegressor(**{**LGB_PARAMS, "n_estimators": MAX_ROUNDS})
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       log_evaluation(period=0)],
        )
        best = int(model.best_iteration_ or MAX_ROUNDS)
        preds = model.predict(X_va, num_iteration=best)
        return model, preds, best
    if family == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(**dict(XGB_PARAMS),
                             early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        # The wrapper's default predict uses its own early-stopped round
        # count (best_iteration is not reliably exposed in xgboost 3.2).
        preds = model.predict(X_va)
        best = int(getattr(model, "best_iteration", None) or MAX_ROUNDS)
        return model, preds, best
    if family == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**dict(RF_PARAMS))
        model.fit(X_tr, y_tr)
        best = int(RF_PARAMS.get("n_estimators", 300))
        preds = model.predict(X_va)
        return model, preds, best
    raise ValueError(
        f"unknown family {family!r} — expected one of {FAMILIES}")


# ── Fold-aligned OOF per-side predictions + residuals ────────────────────────

def oof_per_side(folds: list[dict], features: list[str],
                 games: pd.DataFrame, family: str = DEFAULT_FAMILY
                 ) -> tuple[pd.DataFrame, dict, int]:
    """Compute OOF per-side mean predictions + residuals on the caller's folds.

    Args:
        folds: the moneyline harness's own weekly splits (never mutated).
        features: feature columns (Phase-1: the 12-pool).
        games: full decided frame (for the uncovered-game count).
        family: regressor family — "lgb" (standing interim) / "rf" / "xgb".

    Returns:
        out: DataFrame with [game_id, fold_idx, pred_home, pred_away,
             resid_home, resid_away, best_iter_home, best_iter_away] — one
             row per covered game; residuals are actual − pred computed from
             the ROUNDED preds so the artifact is internally consistent
             (pred + resid == actual exactly).
        rounds: {"home": median best-iter, "away": median best-iter}.
        n_uncovered: decided games in NO fold's val window → NaN downstream.

    Leakage assertion per fold (max(train.gameday) < min(val.gameday), else
    AssertionError) — mirror of the margin engine's fold-boundary guard.
    Deterministic: identical folds + seed → byte-identical table.
    """
    games = games.copy()
    if not available_features(features, games.columns):
        raise ValueError("per-side engine: no usable feature columns supplied")

    parts: list[pd.DataFrame] = []
    best_home: list[int] = []
    best_away: list[int] = []

    for i, f in enumerate(folds):
        tr = f["train"].copy()
        va = f["val"].copy()

        # Leakage assertion: max(train.gameday) < min(val.gameday)
        tr_max = pd.to_datetime(tr["gameday"]).max()
        va_min = pd.to_datetime(va["gameday"]).min()
        if not (tr_max < va_min):
            raise AssertionError(
                f"fold {i}: train max {tr_max} not strictly before "
                f"val min {va_min} → leakage-safe split violated")

        id_cols = [c for c in ("game_id", "gameday") if c in va.columns]
        tr_valid = tr[features + ["home_score", "away_score"]].dropna()
        va_valid = va[id_cols + features + ["home_score", "away_score"]].dropna()

        if len(tr_valid) < 30 or len(va_valid) < 5:
            logger.warning("per-side engine: fold %d too small (tr=%d, va=%d), skipping",
                           i, len(tr_valid), len(va_valid))
            continue

        X_tr = tr_valid[features].to_numpy(float)
        X_va = va_valid[features].to_numpy(float)

        _mh, pred_h, bh = _fit_side(family, X_tr, tr_valid["home_score"].to_numpy(float),
                                    X_va, va_valid["home_score"].to_numpy(float))
        _ma, pred_a, ba = _fit_side(family, X_tr, tr_valid["away_score"].to_numpy(float),
                                    X_va, va_valid["away_score"].to_numpy(float))
        best_home.append(bh)
        best_away.append(ba)

        pred_h = np.round(pred_h, 4)
        pred_a = np.round(pred_a, 4)
        rec = {
            "game_id": va_valid["game_id"].values,
            "fold_idx": i,
            PRED_HOME: pred_h,
            PRED_AWAY: pred_a,
            RESID_HOME: np.round(va_valid["home_score"].to_numpy(float) - pred_h, 4),
            RESID_AWAY: np.round(va_valid["away_score"].to_numpy(float) - pred_a, 4),
            "best_iter_home": bh,
            "best_iter_away": ba,
        }
        parts.append(pd.DataFrame(rec))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["game_id", "fold_idx"] + PRED_COLS + RESID_COLS
                + ["best_iter_home", "best_iter_away"])
    rounds = {
        "home": int(np.median(best_home)) if best_home else MAX_ROUNDS,
        "away": int(np.median(best_away)) if best_away else MAX_ROUNDS,
    }

    covered = set(out["game_id"]) if len(out) else set()
    n_uncovered = int((~games["game_id"].isin(covered)).sum())

    if len(out) == 0:
        logger.warning("per-side engine: no folds produced predictions")
    return out, rounds, n_uncovered


def refit_per_side(decided: pd.DataFrame, pred_df: pd.DataFrame,
                   n_rounds: int | dict[str, int], features: list[str],
                   family: str = DEFAULT_FAMILY) -> pd.DataFrame:
    """Fit-only refit on all decided games at fixed round count, predict.

    Sealed-holdout evaluation (fit 2019-2024 → predict 2025) and slate
    inference (fit all decided → predict board rows). Every predicted game is
    strictly future relative to the fit — cannot leak by construction
    (mirror of the margin engine's refit_margins). ``n_rounds`` is either a
    single int (both sides) or the per-side median-round dict.
    """
    decided = decided.copy()
    available = [f for f in features if f in decided.columns
                 and f in pred_df.columns]
    missing = [f for f in features if f not in available]
    if missing:
        logger.warning("per-side refit: %d requested feature(s) absent (%s)",
                       len(missing), ", ".join(missing[:6]))

    valid_decided = decided[available + ["home_score", "away_score"]].dropna()
    valid_pred = pred_df[["game_id"] + available].dropna()

    X_tr = valid_decided[available].to_numpy(float)
    X_pred = valid_pred[available].to_numpy(float)

    result = valid_pred[["game_id"]].copy()
    for side, col in SIDE_TARGETS.items():
        n_fixed = int(n_rounds[side]) if isinstance(n_rounds, dict) \
            else int(n_rounds)
        y_tr = valid_decided[col].to_numpy(float)
        # Fixed round count — no early stopping (a val split would consume
        # rows the sealed holdout must never touch).
        if family == "lgb":
            from lightgbm import LGBMRegressor, log_evaluation
            model = LGBMRegressor(**{**LGB_PARAMS, "n_estimators": n_fixed})
            model.fit(X_tr, y_tr, callbacks=[log_evaluation(period=0)])
            preds = model.predict(X_pred)
        elif family == "xgb":
            from xgboost import XGBRegressor
            model = XGBRegressor(**{**XGB_PARAMS, "n_estimators": n_fixed})
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_pred)
        else:  # rf — n_rounds is a floor for RF; full forest is deterministic
            from sklearn.ensemble import RandomForestRegressor
            n = max(n_fixed, int(RF_PARAMS.get("n_estimators", 300)))
            model = RandomForestRegressor(**{**RF_PARAMS, "n_estimators": n})
            model.fit(X_tr, y_tr)
            preds = model.predict(X_pred)
        result[f"pred_{side}"] = np.round(preds, 4)
    return result


# ── Residual artifact persistence (loud-failure guard) ───────────────────────

def persist_residuals(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist per-game OOF predictions AND residuals keyed by game_id.

    FAILS LOUDLY: any failure to write (or a missing/empty artifact after the
    write) raises RuntimeError — the artifact is the raw material for the
    joint layer, so a silently-missing artifact must never pass as success.
    """
    path = Path(path)
    required = ["game_id", "fold_idx"] + PRED_COLS + RESID_COLS
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"residual artifact refused: missing columns {missing_cols}")
    if len(df) == 0:
        raise RuntimeError("residual artifact refused: empty frame")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:  # noqa: BLE001 — loud failure is the point
        raise RuntimeError(f"residual artifact write failed: {e}") from e
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            f"residual artifact missing/empty after write: {path}")
    logger.info("residual artifact written: %s (%d rows)", path, len(df))
    return path


# ── Shared helpers ────────────────────────────────────────────────────────────

def available_features(features: list[str], columns) -> list[str]:
    available = [f for f in features if f in set(columns)]
    missing = [f for f in features if f not in available]
    if missing:
        logger.warning("per-side engine: %d requested feature(s) absent (%s%s)",
                       len(missing), ", ".join(missing[:6]),
                       " …" if len(missing) > 6 else "")
    if not available:
        raise ValueError("per-side engine: no usable feature columns supplied")
    return available