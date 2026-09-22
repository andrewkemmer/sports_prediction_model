"""Production NFL moneyline: XGBoost / LightGBM / elastic-net logistic
ensemble with expanding walk-forward OOF, causal per-fold logloss weights,
and Platt calibration fit ONLY on valid OOF predictions.

Determinism: explicit seeds on every member; preprocessing (median
imputation + scaling for linear/MLP) is fit on training data only.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from backend import config
    from backend import folds as folds_mod
    from backend import features as feat_mod
except ImportError:
    import config
    import folds as folds_mod
    import features as feat_mod

logger = logging.getLogger(__name__)

CLIP = 1e-7


# ---------------------------------------------------------------------------
# Preprocessing — fit on training data only
# ---------------------------------------------------------------------------
class TrainFoldPreprocessor:
    """Train-median imputation + standard scaling for linear/MLP members.

    Fit strictly on the training fold; the same fitted transform is applied
    to validation and serving frames. Trees consume NaN natively and get the
    raw matrix (no imputation, no scaling).
    """

    def __init__(self) -> None:
        self.medians: pd.Series | None = None
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "TrainFoldPreprocessor":
        self.medians = X.median(numeric_only=True)
        self.means = X.mean(numeric_only=True)
        self.stds = X.std(numeric_only=True).replace(0.0, 1.0)
        # Documented missing-value policy: train-median imputation; a column
        # with NO training observations (e.g. an unknown venue) imputes to
        # 0.0 deterministically — never a fabricated signal, just neutral.
        self.medians = self.medians.fillna(0.0)
        self.means = self.means.fillna(0.0)
        self.stds = self.stds.fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X.astype(float)
        Xf = Xf.fillna(self.medians)
        Xf = (Xf - self.means) / self.stds
        return Xf.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------
def _make_member(name: str):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**config.XGBOOST_PARAMS)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**config.LIGHTGBM_PARAMS)
    if name == "elasticnet":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**config.ELASTICNET_PARAMS)
    if name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**config.RF_PARAMS)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**config.MLP_PARAMS)
    raise KeyError(f"unknown ensemble member {name!r}")


LINEAR_MEMBERS = {"elasticnet", "mlp"}


def member_matrix(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The feature matrix a member consumes (model-family representation)."""
    return feat_mod.linear_view(df) if name in LINEAR_MEMBERS else feat_mod.tree_view(df)


def member_fit_input(name: str, X_raw: pd.DataFrame,
                     pre: "TrainFoldPreprocessor | None"):
    """The ONE authoritative representation handed to member.fit().

    Representation contract (mirrors MLB's model-specific routing):
      * linear members -> the fitted preprocessor's ndarray (imputed+scaled)
      * tree members   -> the NAMED tree-view DataFrame, feature names
        preserved. scikit-learn >= 1.6 + LightGBM < 4.6 warns
        "X does not have valid feature names ... fitted with feature names"
        on every ndarray predict (LightGBM auto-assigns Column_i names), so
        the tree family is pinned to the named frame at BOTH fit and
        predict. member_matrix()'s reindex guarantees identical column
        names/order at every call site.

    fit and predict MUST go through this pair of helpers — never convert to
    a raw ndarray at one site only.
    """
    if name in LINEAR_MEMBERS:
        return pre.transform(X_raw)
    return X_raw


def _member_predict_proba(model, name: str, X_raw: pd.DataFrame,
                          pre: TrainFoldPreprocessor | None) -> np.ndarray:
    X = member_fit_input(name, X_raw, pre)
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, CLIP, 1.0 - CLIP)


# ---------------------------------------------------------------------------
# Walk-forward OOF
# ---------------------------------------------------------------------------
def walk_forward_oof(game_df: pd.DataFrame,
                     date_col: str = "gameday",
                     progress_every: int = 25,
                     fold_list: list | None = None) -> dict:
    """Expanding walk-forward OOF for every ensemble member + the ensemble.

    Returns a dict with:
      oof: DataFrame (game_id, gameday, fold_id, per-member p_home, ensemble)
      member_weights: adaptive weights derived from pooled OOF AUC
      fold_table: per-fold diagnostics
    """
    df = game_df.sort_values(date_col).reset_index(drop=True)
    fold_list = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col=date_col)

    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    prior_weights = dict(config.ENSEMBLE_WEIGHTS)
    prior_losses: dict[str, list[float]] = {n: [] for n in config.ENSEMBLE_MEMBERS}

    for fold in fold_list:
        train = df.loc[fold.train_idx]
        val = df.loc[fold.val_idx]
        y_train = train["home_win"].astype(int).to_numpy()

        member_p: dict[str, np.ndarray] = {}
        for name in config.ENSEMBLE_MEMBERS:
            X_tr_raw = member_matrix(name, train)
            X_va_raw = member_matrix(name, val)
            if name in LINEAR_MEMBERS:
                pre = TrainFoldPreprocessor().fit(X_tr_raw)
            else:
                pre = None
            try:
                model = _make_member(name)
                model.fit(member_fit_input(name, X_tr_raw, pre), y_train)
                member_p[name] = _member_predict_proba(model, name, X_va_raw, pre)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fold %s member %s failed: %s",
                               fold.fold_id, name, exc)
                member_p[name] = None

        rows = pd.DataFrame({
            "game_id": val["game_id"].to_numpy(),
            "gameday": pd.to_datetime(val[date_col]).to_numpy(),
            "season": val["season"].to_numpy(),
            "fold_id": fold.fold_id,
            "home_win": val["home_win"].astype(float).to_numpy(),
        })
        for name in config.ENSEMBLE_MEMBERS:
            p = member_p.get(name)
            # A failed member is recorded as an all-NaN float column, never as
            # None: an object-dtype None column survives into the OOF frame
            # and poisons the numeric consumers that read it (a trial's
            # paired-loss arithmetic could no longer tell "no member scored"
            # from "this member failed").
            rows[f"p_{name}"] = (np.asarray(p, dtype=float) if p is not None
                                 else np.full(len(val), np.nan))
        # Blend this fold using only information earned before this fold.
        fold_weights = _weights_from_loss_history(prior_losses, prior_weights)
        rows["p_ensemble"] = _blend(rows, fold_weights)
        # Only after scoring the fold may its member losses affect the next
        # fold. This is the causal walk-forward weighting contract.
        from sklearn.metrics import log_loss
        for name in config.ENSEMBLE_MEMBERS:
            p_member = member_p.get(name)
            if p_member is not None and len(p_member) and len(np.unique(rows["home_win"])) > 1:
                prior_losses[name].append(float(log_loss(rows["home_win"], p_member, labels=[0, 1])))

        fold_rows.append({
            "fold_id": fold.fold_id,
            "val_start": str(fold.val_start.date()),
            "val_end": str(fold.val_end.date()),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
        })
        oof_parts.append(rows)
        if (fold.fold_id + 1) % progress_every == 0:
            logger.info("moneyline OOF fold %d/%d", fold.fold_id + 1,
                        len(fold_list))

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()

    weights = _weights_from_loss_history(prior_losses, prior_weights)
    return {"oof": oof, "member_weights": weights,
            "fold_table": pd.DataFrame(fold_rows)}


def _weights_from_loss_history(history: dict[str, list[float]], prior: dict[str, float]) -> dict[str, float]:
    losses = {n: float(np.mean(v)) for n, v in history.items() if v}
    if not losses:
        return dict(prior)
    inv = {n: 1.0 / max(losses.get(n, 1.0), 1e-6) for n in config.ENSEMBLE_MEMBERS}
    total = sum(inv.values())
    raw = {n: max(config.ADAPTIVE_WEIGHT_FLOOR,
                  min(config.ADAPTIVE_WEIGHT_CAP, v / total))
           for n, v in inv.items()}
    total = sum(raw.values())
    return {n: w / total for n, w in raw.items()}


def _adaptive_weights(oof: pd.DataFrame):
    """Causal-style weights from member logloss, lower loss earns more weight.
    The final artifact weight is diagnostics/serving weight; fold predictions
    are produced with the prior weights until prior OOF evidence exists."""
    prior = dict(config.ENSEMBLE_WEIGHTS)
    if oof is None or not len(oof):
        return prior
    y = _oof_targets(oof)
    if y is None or len(y) < 2 or len(np.unique(y)) < 2:
        return prior
    from sklearn.metrics import log_loss
    losses: dict[str, float] = {}
    for name in config.ENSEMBLE_MEMBERS:
        col = f"p_{name}"
        if col not in oof.columns:
            return prior
        p = oof[col].to_numpy(dtype=float)
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
            return prior
        losses[name] = float(log_loss(y[ok], p[ok], labels=[0, 1]))
    # Inverse-logloss weights are stable, interpretable, and preserve all
    # members instead of allowing a single noisy fold to dominate.
    inv = {n: 1.0 / max(v, 1e-6) for n, v in losses.items()}
    total = sum(inv.values())
    raw = {n: max(config.ADAPTIVE_WEIGHT_FLOOR,
                  min(config.ADAPTIVE_WEIGHT_CAP, v / total))
           for n, v in inv.items()}
    total = sum(raw.values())
    return {n: w / total for n, w in raw.items()}


def _oof_targets(oof: pd.DataFrame) -> np.ndarray | None:
    """home_win for OOF rows; None when unavailable."""
    if oof is None or not len(oof):
        return None
    if "home_win" in oof.columns:
        return oof["home_win"].astype(int).to_numpy()
    return None


def _blend(oof: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Weighted ensemble probability; NaN members skipped per row."""
    cols = [f"p_{n}" for n in config.ENSEMBLE_MEMBERS
            if f"p_{n}" in oof.columns]
    P = oof[cols].to_numpy(dtype=float)
    w = np.array([weights.get(c[2:], 0.0) for c in cols])
    mask = np.isfinite(P)
    wv = np.where(mask, w[None, :], 0.0)
    wsum = wv.sum(axis=1)
    out = np.divide((P * wv).sum(axis=1), wsum,
                    out=np.full(len(P), np.nan), where=wsum > 0)
    return np.clip(out, CLIP, 1.0 - CLIP)


# ---------------------------------------------------------------------------
# Final full-history refit (production serving models)
# ---------------------------------------------------------------------------
def fit_final_models(game_df: pd.DataFrame) -> tuple[dict, TrainFoldPreprocessor]:
    """Fit every member on ALL eligible settled history. Returns
    ({name: model}, fitted_preprocessor)."""
    models: dict = {}
    pre = TrainFoldPreprocessor()
    y = game_df["home_win"].astype(int).to_numpy()
    for name in config.ENSEMBLE_MEMBERS:
        X_raw = member_matrix(name, game_df)
        if name in LINEAR_MEMBERS:
            member_pre = TrainFoldPreprocessor().fit(X_raw)
            model = _make_member(name)
            model.fit(member_fit_input(name, X_raw, member_pre), y)
            models[name] = {"model": model, "pre": member_pre}
        else:
            model = _make_member(name)
            model.fit(member_fit_input(name, X_raw, None), y)
            models[name] = {"model": model, "pre": None}
    return models, pre


def predict_slate(models: dict, slate_df: pd.DataFrame,
                  weights: dict[str, float]) -> np.ndarray:
    """Production serving: per-member predictions on the slate, blended with
    the SAME weights the OOF derived (identical ensemble logic at serve)."""
    member_p: dict[str, np.ndarray | None] = {}
    for name in config.ENSEMBLE_MEMBERS:
        entry = models.get(name)
        if entry is None:
            member_p[name] = None
            continue
        X_raw = member_matrix(name, slate_df)
        member_p[name] = _member_predict_proba(entry["model"], name, X_raw,
                                               entry["pre"])
    cols = [n for n in config.ENSEMBLE_MEMBERS if member_p.get(n) is not None]
    if not cols:
        return np.full(len(slate_df), np.nan)
    P = np.column_stack([member_p[n] for n in cols])
    w = np.array([weights.get(n, 0.0) for n in cols])
    mask = np.isfinite(P)
    wv = np.where(mask, w[None, :], 0.0)
    wsum = wv.sum(axis=1)
    out = np.divide((P * wv).sum(axis=1), wsum,
                    out=np.full(len(P), np.nan), where=wsum > 0)
    return np.clip(out, CLIP, 1.0 - CLIP)


# ---------------------------------------------------------------------------
# Platt calibration — fit ONLY on valid OOF predictions
# ---------------------------------------------------------------------------
def fit_platt(oof_p: np.ndarray, y: np.ndarray) -> dict:
    """2-parameter logistic map p -> sigmoid(a*z + b) where z = logit(p).
    Deterministic (LBFGS, no randomness)."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(oof_p, CLIP, 1 - CLIP) /
               (1 - np.clip(oof_p, CLIP, 1 - CLIP)))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y.astype(int))
    # MLB presentation parity (calibration.fit_platt): persist the map at
    # 6-decimal precision so the artifact's Platt params render identically
    # on both sports' dashboards (a=0.931371, not a=0.9313710155066628).
    return {"a": round(float(lr.coef_[0][0]), 6),
            "b": round(float(lr.intercept_[0]), 6)}


def apply_platt(p: np.ndarray, cal: dict) -> np.ndarray:
    z = np.log(np.clip(p, CLIP, 1 - CLIP) / (1 - np.clip(p, CLIP, 1 - CLIP)))
    out = 1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"])))
    return np.clip(out, CLIP, 1.0 - CLIP)


def fit_favored_platt(p_home: np.ndarray, home_win: np.ndarray) -> dict:
    """Fit Platt in the same favored-team space shown to users."""
    favored_home = p_home >= 0.5
    p_fav = np.where(favored_home, p_home, 1.0 - p_home)
    y_fav = np.where(favored_home, home_win, 1.0 - home_win)
    cal = fit_platt(p_fav, y_fav)
    cal.update({"method": "favored_platt", "floor": 0.5})
    return cal


def apply_favored_platt(p_home: np.ndarray, cal: dict) -> np.ndarray:
    """Calibrate favored probability, floor it at 50%, convert home space back."""
    favored_home = p_home >= 0.5
    p_fav = np.where(favored_home, p_home, 1.0 - p_home)
    p_fav_cal = np.maximum(0.5, apply_platt(p_fav, cal))
    return np.where(favored_home, p_fav_cal, 1.0 - p_fav_cal)
