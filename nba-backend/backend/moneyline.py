"""NBA binary moneyline ensemble with MLB/NHL-parity OOF semantics.

The module is intentionally self-contained: it does not import another
sport's backend.  Tree members retain named categorical team IDs and native
missing values; the elastic-net member receives median imputation and scaling
fitted on the training fold only.  Fold ``k`` is blended with weights learned
from folds strictly before ``k`` and is calibrated with a map fitted only on
prior OOF blend/outcome pairs.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import features as feat_mod
    from backend import folds as folds_mod
except ImportError:  # pragma: no cover - direct script/import fallback
    import config
    import features as feat_mod
    import folds as folds_mod

logger = logging.getLogger(__name__)
CLIP = 1e-7
LINEAR_MEMBERS = {"elasticnet"}


class TrainFoldPreprocessor:
    """Median imputation and scaling fitted on one training fold only."""

    def __init__(self) -> None:
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "TrainFoldPreprocessor":
        frame = X.astype(float)
        self.medians = frame.median(numeric_only=True).fillna(0.0)
        self.means = frame.mean(numeric_only=True).fillna(0.0)
        self.stds = frame.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.stds is None:
            raise RuntimeError("TrainFoldPreprocessor must be fitted before transform")
        frame = X.astype(float).fillna(self.medians)
        return ((frame - self.means) / self.stds).to_numpy(dtype=float)


def _make_member(
    name: str,
    n_estimators: int | None = None,
    early_stopping_rounds: int | None = None,
):
    """Construct one configured member without importing another sport."""
    if name == "xgboost":
        from xgboost import XGBClassifier
        params = dict(config.XGBOOST_PARAMS)
        if n_estimators is not None:
            params["n_estimators"] = int(n_estimators)
        if early_stopping_rounds is not None:
            # XGBoost 2.x/3.x both accept this in the sklearn constructor;
            # older wrappers expose it only through ``early_stopping_rounds``
            # at fit time.  Keep a constructor fallback so a version skew
            # cannot turn the whole ensemble into a missing member.
            params["early_stopping_rounds"] = int(early_stopping_rounds)
        try:
            return XGBClassifier(**params)
        except (TypeError, ValueError):
            params.pop("early_stopping_rounds", None)
            return XGBClassifier(**params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**config.LIGHTGBM_PARAMS)
    if name == "elasticnet":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**config.ELASTICNET_PARAMS)
    raise KeyError(f"unknown ensemble member {name!r}")


def member_matrix(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return the named feature representation for one model family."""
    return feat_mod.linear_view(df) if name in LINEAR_MEMBERS else feat_mod.tree_view(df)


def member_fit_input(
    name: str,
    X: pd.DataFrame,
    pre: TrainFoldPreprocessor | None,
):
    """Apply the one authoritative preprocessing path for a member."""
    if name in LINEAR_MEMBERS:
        if pre is None:
            raise ValueError("linear member requires a fitted train-fold preprocessor")
        return pre.transform(X)
    return X


def member_matrix_ndarray(
    name: str,
    df: pd.DataFrame,
    pre: TrainFoldPreprocessor | None = None,
) -> np.ndarray:
    """Return a plain matrix for explainers/tests without bypassing ``pre``."""
    return np.asarray(member_fit_input(name, member_matrix(name, df), pre))


def _predict(model, name: str, X: pd.DataFrame,
             pre: TrainFoldPreprocessor | None) -> np.ndarray:
    values = model.predict_proba(member_fit_input(name, X, pre))[:, 1]
    return np.clip(np.asarray(values, dtype=float), CLIP, 1 - CLIP)


def _binary_labels(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame.get("home_win"), errors="coerce")
    return values.where(values.isin([0, 1]), np.nan)


def _fit_member(name: str, train: pd.DataFrame, val: pd.DataFrame | None = None):
    """Fit one member on labeled training rows and optionally early-stop on val."""
    labels = _binary_labels(train)
    train_mask = labels.notna()
    train = train.loc[train_mask]
    labels = labels.loc[train_mask].astype(int)
    if len(train) < 2 or labels.nunique() < 2:
        raise ValueError(f"{name} requires two labeled training classes")

    X_train = member_matrix(name, train)
    pre = TrainFoldPreprocessor().fit(X_train) if name in LINEAR_MEMBERS else None
    y_train = labels.to_numpy(dtype=int)
    X_val = member_matrix(name, val) if val is not None else None
    val_labels = _binary_labels(val) if val is not None else pd.Series(dtype=float)
    val_mask = val_labels.notna() if val is not None else pd.Series(dtype=bool)
    if name == "xgboost" and X_val is not None and val_mask.any():

        model = _make_member(
            name,
            config.XGBOOST_FOLD_ROUNDS,
            config.XGBOOST_EARLY_STOP,
        )
        X_train_fit = member_fit_input(name, X_train, pre)
        X_val_fit = member_fit_input(
            name, X_val.loc[val_mask], pre
        )
        y_val = val_labels.loc[val_mask].astype(int).to_numpy()
        try:
            model.fit(
                X_train_fit,
                y_train,
                eval_set=[(X_val_fit, y_val)],
                verbose=False,
            )
        except TypeError:
            # Older sklearn wrappers do not accept ``verbose`` in fit.
            model.fit(X_train_fit, y_train, eval_set=[(X_val_fit, y_val)])
        except Exception as exc:  # noqa: BLE001
            # A few xgboost releases reject constructor-level early stopping
            # when the eval frame is categorical.  Preserve the configured
            # ceiling and fit without the optional stopping mechanism rather
            # than dropping the member.
            logger.warning("xgboost early stopping unavailable; refitting: %s", exc)
            fallback = _make_member(name, config.XGBOOST_FOLD_ROUNDS)
            fallback.fit(member_fit_input(name, X_train, pre), y_train)
            return fallback, pre
    else:
        model = _make_member(name)
        X_train_fit = member_fit_input(name, X_train, pre)
        fit_kwargs = {}
        if name == "lightgbm":
            categorical = [c for c in config.TREE_CATEGORICAL_COLS
                           if c in getattr(X_train_fit, "columns", [])]
            if categorical:
                fit_kwargs["categorical_feature"] = categorical
        try:
            model.fit(X_train_fit, y_train, **fit_kwargs)
        except Exception as exc:  # noqa: BLE001
            # LightGBM releases differ on pandas categorical routing.  The
            # numeric tree view remains valid; retry without the optional
            # keyword rather than dropping the member.
            if name == "lightgbm" and fit_kwargs:
                logger.warning("LightGBM categorical routing unavailable; retrying numerically: %s", exc)
                model.fit(X_train_fit, y_train)
            else:
                raise
    return model, pre


def _logit_blend_matrix(P: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted logit-space blend with per-row NaN-member skipping."""
    P = np.asarray(P, dtype=float)
    W = np.asarray(weights, dtype=float)
    if P.ndim != 2 or P.shape[1] != len(W):
        return np.full(len(P), np.nan)
    mask = np.isfinite(P)
    active = (W > 0)[None, :] & mask
    clipped = np.clip(np.where(mask, P, 0.5), CLIP, 1 - CLIP)
    logits = np.log(clipped / (1 - clipped))
    weighted = np.where(active, W[None, :], 0.0)
    denom = weighted.sum(axis=1)
    z = np.divide((logits * weighted).sum(axis=1), denom,
                  out=np.full(len(P), np.nan), where=denom > 0)
    result = 1.0 / (1.0 + np.exp(-z))
    return np.clip(np.where(denom > 0, result, np.nan), CLIP, 1 - CLIP)


def _blend(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    cols = [f"p_{name}" for name in config.ENSEMBLE_MEMBERS
            if f"p_{name}" in frame.columns]
    if not cols:
        return np.full(len(frame), np.nan)
    values = frame[cols].to_numpy(dtype=float)
    w = np.asarray([weights.get(c[2:], 0.0) for c in cols], dtype=float)
    return _logit_blend_matrix(values, w)


def compute_adaptive_weights(
    oof_members: dict[str, list[float]],
    y_oof,
) -> dict[str, float]:
    """Earn a deterministic simplex blend by pooled OOF log loss.

    The objective is evaluated in logit space with SLSQP.  A member is skipped
    on rows where it is unavailable; the remaining finite members are
    renormalized for that row.  An entirely unavailable member never erases
    healthy members, while a partially missing member can still earn a small
    weight from the observations where it produced a probability.
    """
    if str(getattr(config, "ADAPTIVE_WEIGHT_METRIC", "logloss")).lower() != "logloss":
        raise ValueError("NBA production ensemble optimization must use logloss")
    y = np.asarray(y_oof, dtype=float)
    if y.size == 0:
        return {}
    labeled = np.isfinite(y) & np.isin(y, [0.0, 1.0])
    if labeled.sum() < 2 or np.unique(y[labeled]).size < 2:
        return {}
    arrays: dict[str, np.ndarray] = {}
    for name, values in oof_members.items():
        if values is None:
            continue
        p = np.asarray(values, dtype=float)
        if p.ndim != 1 or len(p) != len(y):
            continue
        p = np.where(np.isfinite(p), p, np.nan)
        arrays[name] = p
    names = sorted(name for name, p in arrays.items()
                   if np.isfinite(p[labeled]).sum() >= 2)
    if not names:
        return {}
    yy = y[labeled]
    P = np.column_stack([arrays[name][labeled] for name in names])
    finite_any = np.isfinite(P).any(axis=1)
    if finite_any.sum() < 2:
        return {}
    P = P[finite_any]
    yy = yy[finite_any]
    P = np.clip(P, CLIP, 1 - CLIP)
    Z = np.log(P / (1 - P))

    def blend(w: np.ndarray) -> np.ndarray:
        active = np.isfinite(P) & (w[None, :] > 0)
        weights = np.where(active, w[None, :], 0.0)
        denom = weights.sum(axis=1)
        z = np.divide((np.where(active, Z, 0.0) * weights).sum(axis=1),
                      denom, out=np.full(len(P), np.nan), where=denom > 0)
        return 1.0 / (1.0 + np.exp(-z))

    def loss(p: np.ndarray) -> float:
        ok = np.isfinite(p)
        if not ok.any():
            return float("inf")
        p = np.clip(p[ok], CLIP, 1 - CLIP)
        target = yy[ok]
        return float(-(target * np.log(p) + (1 - target) * np.log(1 - p)).mean())

    def objective(w: np.ndarray) -> float:
        return loss(blend(w))

    w0 = np.full(len(names), 1.0 / len(names))
    if len(names) == 1:
        return {names[0]: 1.0}

    from scipy.optimize import minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
        options={"maxiter": 300, "ftol": 1e-9},
    )
    solo = {name: loss(P[:, i]) for i, name in enumerate(names)}
    best = min(solo, key=solo.get)
    if not result.success or not np.all(np.isfinite(result.x)):
        return {name: float(name == best) for name in names}

    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights = weights / weights.sum() if weights.sum() else w0
    if solo[best] < objective(weights) - 1e-12:
        return {name: float(name == best) for name in names}
    rounded = {name: round(float(value), 4)
               for name, value in zip(names, weights)}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(rounded, key=rounded.get)
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded



def walk_forward_oof(
    game_df: pd.DataFrame,
    date_col: str = "gameday",
    progress_every: int = 25,
    fold_list=None,
) -> dict:
    """Causal expanding OOF with prior-fold weights and calibration."""
    df = game_df.sort_values(date_col).reset_index(drop=True)
    folds = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    prior_weights = dict(config.ENSEMBLE_WEIGHTS)
    pooled: dict[str, list[float]] = {name: [] for name in config.ENSEMBLE_MEMBERS}
    pooled_y: list[float] = []
    prior_blend: list[float] = []
    # These are intentionally separate: the current fold may enter the next
    # fold's weight/calibration fit only after it has been scored.
    prior_y: list[float] = []

    for fold in folds:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        # Preserve the weights actually used to score this fold.  The
        # optimizer updates ``prior_weights`` only after the fold is appended;
        # recording the post-update vector would make fold 0 appear to have
        # already used OOF evidence from fold 0 itself.
        fold_weights = dict(prior_weights)
        train_labels = _binary_labels(train)
        if len(train_labels.dropna()) < 2 or train_labels.dropna().nunique() < 2:
            logger.warning("skipping moneyline fold %s: insufficient labels", fold.fold_id)
            continue

        member_p: dict[str, np.ndarray | None] = {}
        for name in config.ENSEMBLE_MEMBERS:
            try:
                model, pre = _fit_member(name, train, val)
                member_p[name] = _predict(model, name, member_matrix(name, val), pre)
            except Exception as exc:  # noqa: BLE001
                logger.warning("moneyline fold %s member %s failed: %s",
                               fold.fold_id, name, exc)
                member_p[name] = None

        y_val = _binary_labels(val)
        row = pd.DataFrame(
            {
                "game_id": val.game_id.to_numpy(),
                "gameday": pd.to_datetime(val[date_col]).to_numpy(),
                "season": val.season.to_numpy() if "season" in val else np.nan,
                "fold_id": fold.fold_id,
                "home_win": y_val.to_numpy(dtype=float),
            }
        )
        for name in config.ENSEMBLE_MEMBERS:
            p = member_p.get(name)
            row[f"p_{name}"] = (
                np.asarray(p, dtype=float) if p is not None
                else np.full(len(val), np.nan)
            )
        # This fold's blend uses only the prior weights.
        row["p_ensemble"] = _blend(row, prior_weights)
        prior_cal = moneyline_fit(
            np.asarray(prior_blend, dtype=float),
            np.asarray(prior_y, dtype=float),
        )
        calibrated = moneyline_apply(row.p_ensemble.to_numpy(float), prior_cal)
        row["p_ensemble_calibrated"] = np.where(
            np.isfinite(calibrated), calibrated, row.p_ensemble.to_numpy(float)
        )

        # Add the scored fold only after its prediction and calibration values
        # have been fixed.  These values can train the next fold.
        for name in config.ENSEMBLE_MEMBERS:
            pooled[name].extend(row[f"p_{name}"].tolist())
        pooled_y.extend(y_val.tolist())
        prior_blend.extend(row.p_ensemble.tolist())
        prior_y.extend(y_val.tolist())
        earned = compute_adaptive_weights(pooled, np.asarray(pooled_y, dtype=float))
        if earned:
            prior_weights = earned

        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "val_start": str(fold.val_start.date()),
                "val_end": str(fold.val_end.date()),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "is_partial_tail": bool(getattr(fold, "is_partial_tail", False)),
                "weights": fold_weights,
            }
        )
        parts.append(row)
        if progress_every and (fold.fold_id + 1) % progress_every == 0:
            logger.info("moneyline OOF fold %d/%d", fold.fold_id + 1, len(folds))

    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return {"oof": oof, "member_weights": prior_weights,
            "fold_table": pd.DataFrame(fold_rows)}


def fit_final_models(game_df: pd.DataFrame):
    """Fit all members on every labeled settled game for production serving."""
    labels = _binary_labels(game_df)
    frame = game_df.loc[labels.notna()].copy()
    if len(frame) < 2 or labels.loc[labels.notna()].nunique() < 2:
        raise ValueError("NBA final moneyline fit requires both outcome classes")
    models: dict[str, dict[str, Any]] = {}
    y = labels.loc[labels.notna()].astype(int).to_numpy()
    for name in config.ENSEMBLE_MEMBERS:
        X = member_matrix(name, frame)
        pre = TrainFoldPreprocessor().fit(X) if name in LINEAR_MEMBERS else None
        model = _make_member(name)
        X_fit = member_fit_input(name, X, pre)
        fit_kwargs = {}
        if name == "lightgbm":
            categorical = [c for c in config.TREE_CATEGORICAL_COLS
                           if c in getattr(X_fit, "columns", [])]
            if categorical:
                fit_kwargs["categorical_feature"] = categorical
        try:
            model.fit(X_fit, y, **fit_kwargs)
        except Exception as exc:  # noqa: BLE001
            if name == "lightgbm" and fit_kwargs:
                logger.warning("LightGBM categorical routing unavailable; retrying numerically: %s", exc)
                model.fit(X_fit, y)
            else:
                raise
        models[name] = {"model": model, "pre": pre}
    # Keep the historical two-tuple contract; per-member preprocessors live
    # in each entry and are used by predict_slate.
    return models, None


def predict_slate(models: dict, slate_df: pd.DataFrame,
                  weights: dict[str, float]) -> np.ndarray:
    """Predict and blend a pending slate with the deployed OOF weights."""
    predictions: list[np.ndarray] = []
    names: list[str] = []
    for name in config.ENSEMBLE_MEMBERS:
        entry = models.get(name)
        if not entry:
            continue
        try:
            predictions.append(
                _predict(entry["model"], name,
                         member_matrix(name, slate_df), entry.get("pre"))
            )
            names.append(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("serving member %s failed: %s", name, exc)
    if not predictions:
        return np.full(len(slate_df), np.nan)
    return _logit_blend_matrix(
        np.column_stack(predictions),
        np.asarray([weights.get(name, 0.0) for name in names], dtype=float),
    )


# Favored-space Platt calibration (the MLB/NHL production contract).
FAVORED_CALIBRATOR_METHOD = "favored_platt_floor"
FAVORED_PROBABILITY_FLOOR = 0.5
VALID_CALIBRATION_MODES = ("platt", "identity")


def get_calibration_mode() -> str:
    mode = str(os.environ.get("CALIBRATION_MODE") or config.CALIBRATION_MODE).strip().lower()
    if mode not in VALID_CALIBRATION_MODES:
        logger.warning("Calibration: unknown mode %r; using platt", mode)
        return "platt"
    return mode


def set_calibration_mode(mode: str) -> None:
    value = str(mode).strip().lower()
    if value not in VALID_CALIBRATION_MODES:
        raise ValueError(f"unknown calibration mode {value!r}")
    config.CALIBRATION_MODE = value


def is_identity(cal: dict | None) -> bool:
    if not cal or str(cal.get("method")) != FAVORED_CALIBRATOR_METHOD:
        return True
    try:
        return abs(float(cal.get("a", 1.0)) - 1.0) < 1e-9 and abs(float(cal.get("b", 0.0))) < 1e-9
    except (TypeError, ValueError):
        return True


def fit_platt(p_fav, y_fav):
    p = np.asarray(p_fav, dtype=float)
    y = np.asarray(y_fav, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y) & (p > 0) & (p < 1)
    p, y = p[ok], y[ok]
    if len(y) < config.MIN_OOF_FOR_FIT or len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    z = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y.astype(int))
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return {"method": "platt", "a": round(a, 6), "b": round(b, 6), "n": int(len(y))}


def _stable_sigmoid(z):
    values = np.asarray(z, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_z = np.exp(values[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def apply_platt(p, cal):
    values = np.asarray(p, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if cal and finite.any():
        clipped = np.clip(values[finite], CLIP, 1 - CLIP)
        z = np.log(clipped / (1 - clipped))
        out[finite] = _stable_sigmoid(
            float(cal.get("a", 1)) * z + float(cal.get("b", 0)))

    elif not cal:
        out[finite] = np.clip(values[finite], CLIP, 1 - CLIP)
    return np.clip(out, CLIP, 1 - CLIP)


def moneyline_fit(p_home, home_win):
    """Fit the calibration map in favored-team space."""
    if get_calibration_mode() == "identity":
        return None
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(home_win, dtype=float)
    favorite = p >= 0.5
    p_fav = np.where(favorite, p, 1 - p)
    y_fav = np.where(favorite, y, 1 - y)
    cal = fit_platt(p_fav, y_fav)
    if cal:
        cal.update({"method": FAVORED_CALIBRATOR_METHOD,
                    "floor": FAVORED_PROBABILITY_FLOOR})
    return cal


def moneyline_apply(p_home, cal):
    values = np.asarray(p_home, dtype=float)
    finite = np.isfinite(values)
    if not cal or get_calibration_mode() == "identity":
        return np.clip(values, 0.0, 1.0)
    if cal.get("method") != FAVORED_CALIBRATOR_METHOD:
        raise ValueError("legacy home-space calibration is unsupported")
    favorite = values >= 0.5
    p_fav = np.where(favorite, values, 1 - values)
    calibrated = np.maximum(0.5, apply_platt(p_fav, cal))
    out = np.where(favorite, calibrated, 1 - calibrated)
    return np.where(finite, out, np.nan)
