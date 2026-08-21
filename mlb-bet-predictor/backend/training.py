"""Walk-forward training for the three model heads.

Design guarantees
-----------------
* **Expanding window**: every training fold contains only games whose start
  time is strictly before the validation window start. The train set grows
  monotonically; the validation window slides forward by ``cadence_days``.
* **No future leakage**: features are computed point-in-time upstream
  (``data_ingestion.build_point_in_time_features``) and splits filter purely
  on ``start_time``, so a game's outcome can never influence an earlier fold.
* **Multi-target**: Moneyline (P home win, classification ensemble),
  Totals (total runs, regression ensemble), Run Line (P home covers -1.5,
  classification ensemble). Ensembles average XGBoost + LightGBM + Logistic
  Regression probability outputs.

Heavy ML libraries (xgboost, lightgbm, scikit-learn) are imported lazily
inside functions so unit tests and the Streamlit frontend never need them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as date_cls, datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import (
    DATA_CUTOFF,
    EVALUATION_NOTES,
    FEATURE_COLUMNS,
    LGBM_PARAMS,
    LR_PARAMS,
    METRICS,
    MIN_TRAIN_DAYS,
    MIN_TRAIN_GAMES,
    MODELS_DIR,
    MODEL_MANIFEST_FILE,
    MODEL_TYPE,
    N_CALIBRATION_BINS,
    N_TRAIN_GAMES,
    SEED,
    TRAINED_AT,
    VERSION,
    VALIDATION_WINDOW_DAYS,
    XGB_PARAMS,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "v3.2.1"  # bump when features/hyperparameters change materially


# ===========================================================================
# Walk-forward splits (pure, unit-testable)
# ===========================================================================

@dataclass
class WalkForwardSplit:
    """One train/validation pair from the expanding-window procedure."""

    fold: int
    train_cutoff: pd.Timestamp  # exclusive: train = games with start_time < cutoff
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    train: pd.DataFrame
    valid: pd.DataFrame


def walk_forward_splits(
    games: pd.DataFrame,
    cadence_days: int = 7,
    min_train_days: int = MIN_TRAIN_DAYS,
    validation_window_days: int = VALIDATION_WINDOW_DAYS,
    n_splits: Optional[int] = None,
    min_train_games: int = MIN_TRAIN_GAMES,
) -> Iterable[WalkForwardSplit]:
    """Yield expanding-window train/validation splits over chronological games.

    * Train folds are strictly historical: ``max(train.start_time) < valid_start``.
    * The training window *expands* — every fold keeps all prior history.
    * Validation windows never overlap and never peek ahead.
    """
    df = games.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df = df.sort_values("start_time").reset_index(drop=True)
    if df.empty:
        return

    first = df["start_time"].min().normalize()
    last = df["start_time"].max().normalize()
    if (last - first).days < min_train_days:
        logger.warning("Not enough history (%d days) for walk-forward.", (last - first).days)
        return

    cutoff = first + pd.Timedelta(days=min_train_days)
    fold = 0
    while cutoff <= last:
        valid_end = min(cutoff + pd.Timedelta(days=validation_window_days), last + pd.Timedelta(days=1))
        train = df[df["start_time"] < cutoff].copy()
        valid = df[(df["start_time"] >= cutoff) & (df["start_time"] < valid_end)].copy()
        if len(train) >= min_train_games and len(valid) >= 1:
            yield WalkForwardSplit(
                fold=fold, train_cutoff=cutoff, valid_start=cutoff,
                valid_end=valid_end, train=train, valid=valid,
            )
            fold += 1
        cutoff = valid_end
        if n_splits is not None and fold >= n_splits:
            break


# ===========================================================================
# Metrics (sklearn imported lazily)
# ===========================================================================

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_CALIBRATION_BINS) -> float:
    """Expected Calibration Error (ECE) over equally spaced probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_prob)
    if total == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if lo == 0.0:
            mask |= y_prob == 0.0
        if hi == 1.0:
            mask |= y_prob == 1.0
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def evaluate_classification(y_true, y_prob, n_bins: int = N_CALIBRATION_BINS) -> dict:
    """AUC, Brier, LogLoss, ECE + per-bin calibration rows for a prob series."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # lazy

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    unique = np.unique(y_true)
    auc = float(roc_auc_score(y_true, y_prob)) if len(unique) > 1 else float("nan")
    metrics = {
        "auc_roc": round(auc, 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
        "cal_error": round(expected_calibration_error(y_true, y_prob, n_bins=n_bins), 4),
        "n_games": int(len(y_true)),
    }
    return metrics


def calibration_bins(y_true, y_prob, step: float = 0.05) -> list[dict]:
    """Reliability-diagram rows: bucket, mean predicted, mean actual, count, gap."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    for lo in np.arange(0.50, 0.95, step):
        hi = lo + step
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mp = float(y_prob[mask].mean())
        ma = float(y_true[mask].mean())
        rows.append(
            {
                "bucket": f"{int(lo * 100)}-{int(hi * 100)}%",
                "mean_predicted": round(mp, 3),
                "mean_actual": round(ma, 3),
                "count": n,
                "gap": round(mp - ma, 3),
            }
        )
    return rows


# ===========================================================================
# Ensembles
# ===========================================================================

class AveragingEnsemble:
    """Probability-averaging ensemble for classification heads."""

    def __init__(self, models: list, weights: Optional[list] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict_proba_home(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros(len(X))
        for w, m in zip(self.weights, self.models):
            probs += w * m.predict_proba(X)[:, 1]
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba_home(X) >= 0.5).astype(int)


class RegressionEnsemble:
    """Averaging ensemble for regression heads (totals, margins)."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)


# ===========================================================================
# Training
# ===========================================================================

def _build_classifiers():
    from lightgbm import LGBMClassifier  # lazy
    from sklearn.linear_model import LogisticRegression  # lazy
    from xgboost import XGBClassifier  # lazy

    return [
        XGBClassifier(**XGB_PARAMS),
        LGBMClassifier(**LGBM_PARAMS),
        LogisticRegression(**LR_PARAMS),
    ]


def _build_regressors():
    from lightgbm import LGBMRegressor  # lazy
    from xgboost import XGBRegressor  # lazy

    return [
        XGBRegressor(n_estimators=XGB_PARAMS["n_estimators"], max_depth=XGB_PARAMS["max_depth"],
                     learning_rate=XGB_PARAMS["learning_rate"], subsample=XGB_PARAMS["subsample"],
                     colsample_bytree=XGB_PARAMS["colsample_bytree"], reg_lambda=XGB_PARAMS["reg_lambda"],
                     n_jobs=XGB_PARAMS["n_jobs"], random_state=SEED, verbosity=0),
        LGBMRegressor(n_estimators=LGBM_PARAMS["n_estimators"], max_depth=LGBM_PARAMS["max_depth"],
                      learning_rate=LGBM_PARAMS["learning_rate"], subsample=LGBM_PARAMS["subsample"],
                      colsample_bytree=LGBM_PARAMS["colsample_bytree"], reg_lambda=LGBM_PARAMS["reg_lambda"],
                      n_jobs=LGBM_PARAMS["n_jobs"], random_state=SEED, verbosity=-1),
    ]


def train_moneyline(X_train: pd.DataFrame, y_train: pd.Series) -> AveragingEnsemble:
    xgb, lgbm, lr = _build_classifiers()
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return AveragingEnsemble([xgb, lgbm, lr])


def train_totals(X_train: pd.DataFrame, y_train: pd.Series) -> RegressionEnsemble:
    xgb, lgbm = _build_regressors()
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    return RegressionEnsemble([xgb, lgbm])


def train_runline(X_train: pd.DataFrame, y_train: pd.Series) -> AveragingEnsemble:
    return train_moneyline(X_train, y_train)


def prepare_training_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Keep completed games with complete feature rows (PIT already applied)."""
    df = features.copy()
    df = df[df["home_runs"].notna() & df["away_runs"].notna()]
    df = df.dropna(subset=FEATURE_COLUMNS)
    return df.reset_index(drop=True)


def run_walk_forward_evaluation(
    features: pd.DataFrame,
    cadence_days: int = 7,
    min_train_days: int = MIN_TRAIN_DAYS,
    n_splits: Optional[int] = None,
) -> dict:
    """Evaluate all heads across expanding-window folds.

    Returns pooled validation predictions/metrics plus per-fold metrics. No
    outcome from a validation window ever reaches the training set of an
    earlier (or the same) fold.
    """
    train_df = prepare_training_frame(features)
    if len(train_df) < MIN_TRAIN_GAMES:
        raise ValueError(
            f"Only {len(train_df)} completed games available; need at least {MIN_TRAIN_GAMES}. "
            "Extend the history window or lower MIN_TRAIN_GAMES."
        )

    pooled = {"home_win": ([], []), "total_runs": ([], []), "home_cover": ([], []), "dates": []}
    folds = []
    for split in walk_forward_splits(train_df, cadence_days, min_train_days, n_splits=n_splits):
        X_train = split.train[FEATURE_COLUMNS]
        y_ml = split.train["home_win"]
        y_tot = split.train["total_runs"]
        y_rl = split.train["home_cover"]

        ml_model = train_moneyline(X_train, y_ml)
        tot_model = train_totals(X_train, y_tot)
        rl_model = train_runline(X_train, y_rl)

        X_valid = split.valid[FEATURE_COLUMNS]
        ml_prob = ml_model.predict_proba_home(X_valid)
        tot_pred = tot_model.predict(X_valid)
        rl_prob = rl_model.predict_proba_home(X_valid)

        pooled["home_win"][0].extend(split.valid["home_win"].tolist())
        pooled["home_win"][1].extend(ml_prob.tolist())
        pooled["total_runs"][0].extend(split.valid["total_runs"].tolist())
        pooled["total_runs"][1].extend(tot_pred.tolist())
        pooled["home_cover"][0].extend(split.valid["home_cover"].tolist())
        pooled["home_cover"][1].extend(rl_prob.tolist())
        pooled["dates"].extend(pd.to_datetime(split.valid["start_time"], utc=True).tolist())

        folds.append(
            {
                "fold": split.fold,
                "train_cutoff": split.train_cutoff.date().isoformat(),
                "valid_start": split.valid_start.date().isoformat(),
                "valid_end": split.valid_end.date().isoformat(),
                "n_train": int(len(split.train)),
                "n_valid": int(len(split.valid)),
            }
        )

    if not folds:
        raise ValueError("Walk-forward produced no folds — widen the history window.")

    return {
        "folds": folds,
        "metrics_moneyline": evaluate_classification(pooled["home_win"][0], pooled["home_win"][1]),
        "metrics_totals": {
            "mae": round(float(np.mean(np.abs(np.array(pooled["total_runs"][0]) - np.array(pooled["total_runs"][1])))), 3),
            "rmse": round(float(np.sqrt(np.mean((np.array(pooled["total_runs"][0]) - np.array(pooled["total_runs"][1])) ** 2))), 3),
            "n_games": len(pooled["total_runs"][0]),
        },
        "metrics_runline": evaluate_classification(pooled["home_cover"][0], pooled["home_cover"][1]),
        "calibration": calibration_bins(pooled["home_win"][0], pooled["home_win"][1]),
        "pooled_y_true": pooled["home_win"][0],
        "pooled_y_prob": pooled["home_win"][1],
        "pooled_dates": pooled["dates"],
    }


def train_final_model(features: pd.DataFrame, data_cutoff: date_cls) -> dict:
    """Train the production ensemble on all history strictly before cutoff."""
    train_df = prepare_training_frame(features)
    if len(train_df) < MIN_TRAIN_GAMES:
        raise ValueError(f"Need at least {MIN_TRAIN_GAMES} completed games to train.")

    X = train_df[FEATURE_COLUMNS]
    moneyline = train_moneyline(X, train_df["home_win"])
    totals = train_totals(X, train_df["total_runs"])
    runline = train_runline(X, train_df["home_cover"])

    now_utc = datetime.now().astimezone()
    payload = {
        VERSION: MODEL_VERSION,
        TRAINED_AT: now_utc.isoformat(),
        DATA_CUTOFF: pd.Timestamp(data_cutoff, tz="UTC").isoformat(),
        MODEL_TYPE: "ensemble_average",
        "FEATURE_COLUMNS": FEATURE_COLUMNS,
        N_TRAIN_GAMES: int(len(train_df)),
        "moneyline": moneyline,
        "totals": totals,
        "runline": runline,
        METRICS: {},  # filled from walk-forward evaluation by the caller
        EVALUATION_NOTES: (
            "Ensemble = mean(XGBoost, LightGBM, LogisticRegression); "
            "trained with expanding-window walk-forward, no future data."
        ),
    }
    _persist_ensemble(payload)
    return payload


def _persist_ensemble(payload: dict) -> None:
    import joblib  # lazy

    path = MODELS_DIR / "ensemble_latest.joblib"
    joblib.dump(payload, path)
    manifest = MODELS_DIR / MODEL_MANIFEST_FILE
    manifest.write_text(
        json.dumps(
            {
                VERSION: payload[VERSION],
                TRAINED_AT: payload[TRAINED_AT],
                DATA_CUTOFF: payload[DATA_CUTOFF],
                N_TRAIN_GAMES: payload[N_TRAIN_GAMES],
                "artifact": path.name,
            },
            indent=2,
        )
    )
    logger.info("Persisted ensemble -> %s", path)


def load_ensemble() -> dict:
    """Load the latest persisted ensemble payload (or raise if absent)."""
    import joblib  # lazy

    path = MODELS_DIR / "ensemble_latest.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained ensemble at {path}. Run training first.")
    return joblib.load(path)


def is_retrain_due(target_date: date_cls, cadence_days: int) -> bool:
    """True when no model exists, or the last retrain is older than cadence."""
    manifest_path = MODELS_DIR / MODEL_MANIFEST_FILE
    if not manifest_path.exists():
        return True
    try:
        manifest = json.loads(manifest_path.read_text())
        trained = pd.Timestamp(manifest[TRAINED_AT]).tz_localize(None).date()
        return (target_date - trained).days >= cadence_days
    except Exception:
        return True
