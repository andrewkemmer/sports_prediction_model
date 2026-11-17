"""NBA binary moneyline ensemble with MLB/NHL-parity OOF semantics.

The module is intentionally self-contained: it does not import another
sport's backend.  Tree members retain named categorical team IDs and native
missing values; the elastic-net member receives median imputation and scaling
fitted on the training fold only.  Fold ``k`` is blended with weights learned
from folds strictly before ``k`` and is calibrated with a map fitted only on
prior OOF blend/outcome pairs.

XGBoost probes a held-out chronological train tail for its round budget,
then refits on all training rows. Folds and final serving use this same policy;
no scored validation outcome can influence its own prediction.
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
# Tree-view members that still need the train-fold preprocessor. Random
# forest cannot ingest NaN (unlike the boosters' native missing handling)
# or pandas.Categorical (unlike their native categorical support), so its
# matrix codes the team IDs as plain ints and rides the preprocessor for
# median imputation; the scaling the preprocessor also applies is
# split-invariant, so tree geometry is unaffected. Dormant unless a trial
# adds "randomforest" to ENSEMBLE_MEMBERS.


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
    if name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**config.RANDOMFOREST_PARAMS)
    raise KeyError(f"unknown ensemble member {name!r}")


def member_matrix(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return the named feature representation for one model family."""
    if name in LINEAR_MEMBERS:
        return feat_mod.linear_view(df)
    out = feat_mod.tree_view(df)
    if name == "randomforest":
        out = out.copy()
        for col in config.TREE_CATEGORICAL_COLS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.astype(float)
    return out


def member_fit_input(
    name: str,
    X: pd.DataFrame,
    pre: TrainFoldPreprocessor | None = None,
):
    """Apply the one authoritative preprocessing path for a member."""
    if name in LINEAR_MEMBERS or name == "randomforest":
        if pre is None:
            raise ValueError(f"{name} requires a fitted train-fold preprocessor")
        return pre.transform(X)
    return X


def member_matrix_ndarray(
    name: str,
    df: pd.DataFrame,
    pre: TrainFoldPreprocessor | None = None,
) -> np.ndarray:
    """Return a plain matrix for explainers/tests without bypassing ``pre``.

    ``shap_explain`` builds each member's input through this one path, so the
    attribution matrix can never drift from the matrix the predict path
    builds. It is deliberately NOT inlined there: the whole point of the
    helper is that there is exactly one construction of a member's matrix.
    """
    return np.asarray(member_fit_input(name, member_matrix(name, df), pre))



def _market_logit(p) -> np.ndarray:
    """De-vigged probability -> log-odds, the offset every arm shares."""
    values = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


class _OffsetLinear:
    """Logistic member whose market logit enters with a FIXED coefficient.

    sklearn's LogisticRegression has no offset hook, so the linear member
    cannot start from the de-vigged price the way xgboost (``base_margin``)
    and lightgbm (``init_score``) can. Appending the logit as a column gives
    it a FREE coefficient, which on the NHL folds settled at 0.43 - a silent
    discount on the price. This holds the offset at 1.0 and learns only the
    residual: the same contract the trees are fitted under (NHL parity,
    2026-11-15).
    """

    def __init__(self, coef: np.ndarray, intercept: float):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)

    def decision(self, X, base_margin) -> np.ndarray:
        return (self.intercept_ + np.asarray(base_margin, dtype=float)
                + np.asarray(X, dtype=float) @ self.coef_)

    def predict_proba(self, X, base_margin) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision(X, base_margin)))
        return np.column_stack([1.0 - p, p])


def _fit_offset_linear(X, y, base_margin, *,
                       max_iter: int = 8000,
                       tol: float = 1e-9) -> _OffsetLinear:
    """FISTA on sklearn's elasticnet logistic objective with a fixed offset.

    Objective: mean log-loss at ``offset + b0 + X @ beta`` plus
    ``(l1_ratio/(C*n)) * |beta|_1 + 0.5*((1-l1_ratio)/(C*n)) * |beta|_2^2``,
    with an UNPENALISED intercept - the penalty sklearn applies for
    ``LogisticRegression(penalty='elasticnet', C=..., l1_ratio=...)``.
    Deterministic: no seed, no threads, so every fold reproduces
    bit-identically (the NHL implementation this mirrors is validated
    against sklearn at offset = 0).
    """
    params = config.ELASTICNET_PARAMS
    C = float(params.get("C", 1.0))
    rho = float(params.get("l1_ratio", 0.5))
    Z = np.asarray(X, dtype=float)
    off = np.asarray(base_margin, dtype=float)
    yv = np.asarray(y, dtype=float)
    n, p = Z.shape
    lam1 = rho / (C * n)
    lam2 = 0.5 * (1.0 - rho) / (C * n)
    beta = np.zeros(p)
    b0 = 0.0
    beta_z, b0_z = beta.copy(), b0
    momentum = 1.0
    lip = 0.25 * np.linalg.norm(Z, 2) ** 2 / n + lam2
    step = 1.0 / lip if lip > 0 else 1.0
    for _ in range(max_iter):
        pr = 1.0 / (1.0 + np.exp(-(off + b0_z + Z @ beta_z)))
        resid = pr - yv
        grad = Z.T @ resid / n + lam2 * beta_z
        grad0 = float(resid.mean())
        nxt = beta_z - step * grad
        nxt = np.sign(nxt) * np.maximum(np.abs(nxt) - step * lam1, 0.0)
        nxt0 = b0_z - step * grad0
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        beta_new = nxt + ((momentum - 1.0) / t_next) * (nxt - beta_z)
        b0_new = nxt0 + ((momentum - 1.0) / t_next) * (nxt0 - b0_z)
        converged = max(float(np.max(np.abs(nxt - beta))),
                        abs(nxt0 - b0)) < tol
        beta, b0 = nxt, nxt0
        beta_z, b0_z = beta_new, b0_new
        momentum = t_next
        if converged:
            break
    return _OffsetLinear(beta, b0)


def _predict(model, name: str, X: pd.DataFrame,
             pre: TrainFoldPreprocessor | None,
             base_margin: np.ndarray | None = None) -> np.ndarray:
    """Member probabilities; ``base_margin`` must match the fit convention.

    The offset must travel with the model: a member fitted from the market
    logit and predicted without it answers a different question (it would
    serve the residual alone, centred on 0.5).
    """
    feat = member_fit_input(name, X, pre)
    if isinstance(model, _OffsetLinear):
        if base_margin is None:
            raise ValueError(
                "offset linear member: base_margin is required at predict "
                "time - the fit was offset by the market logit")
        values = model.predict_proba(
            feat, np.asarray(base_margin, dtype=float))[:, 1]
    elif base_margin is not None and name == "xgboost":
        values = model.predict_proba(
            feat, base_margin=np.asarray(base_margin, dtype=float))[:, 1]
    elif base_margin is not None and name == "lightgbm":
        # The sklearn wrapper has no init_score at predict time: the raw
        # margin is the learned residual, so the offset is added back before
        # the sigmoid.
        raw = np.asarray(model.predict(feat, raw_score=True), dtype=float)
        values = 1.0 / (1.0 + np.exp(-(raw + np.asarray(base_margin,
                                                        dtype=float))))
    else:
        values = model.predict_proba(feat)[:, 1]
    return np.clip(np.asarray(values, dtype=float), CLIP, 1 - CLIP)


def _binary_labels(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame.get("home_win"), errors="coerce")
    return values.where(values.isin([0, 1]), np.nan)


def _early_stop_watch(X_train: pd.DataFrame, y_train: np.ndarray,
                      pre: TrainFoldPreprocessor | None,
                      ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    """The strictly-train chronological tail used to measure XGB rounds.

    Structure aligned with MLB's config-driven fold-fit shape
    (``XGBOOST_FOLD_ROUNDS`` / ``XGBOOST_EARLY_STOP``), with one deliberate
    divergence: the watch set is a tail of the TRAINING fold, not the
    validation window. The validation rows are the rows the fold is graded
    on, and a watch set drawn from them let each fold's OOF metric influence
    the round count of the model that produced it. A train-tail watch set
    is held out from the gradient (xgboost excludes the eval set from
    training) and is strictly before ``val_start`` by fold construction,
    so the fold's mechanics depend on nothing at or after the boundary.
    Folds too small to afford the minimum (``XGBOOST_EARLY_STOP_MIN_ROWS``)
    skip early stopping and fit the configured ceiling - the same fallback
    MLB's fold path already exercises when its eval frame is unusable.
    """
    rows = X_train.index
    if len(rows) < 2 * int(config.XGBOOST_EARLY_STOP_MIN_ROWS):
        return None
    tail_n = max(int(config.XGBOOST_EARLY_STOP_MIN_ROWS),
                 int(round(config.XGBOOST_EARLY_STOP_FRAC * len(rows))))
    tail_n = min(tail_n, len(rows) // 2)
    watch_rows = rows[-tail_n:]
    fit_rows = rows[:-tail_n]
    watch = pre.transform(X_train.loc[watch_rows]) if pre is not None \
        else X_train.loc[watch_rows]
    return (np.asarray(fit_rows), np.asarray(y_train[: len(fit_rows)]), watch)


def _fit_member(name: str, train: pd.DataFrame,
                base_margin=None
                ) -> tuple[Any, TrainFoldPreprocessor | None]:
    """Fit one member on labeled training rows only.

    The signature deliberately has no ``val`` parameter: the fold loop
    cannot hand the validation window to any member's fit, so "training
    stops at the fold's train end" is a property of the type, not a
    convention a call site can silently break.
    """
    train = folds_mod.canonical_sort(train)
    labels = _binary_labels(train)
    train_mask = labels.notna()
    train = train.loc[train_mask]
    labels = labels.loc[train_mask].astype(int)
    if len(train) < 2 or labels.nunique() < 2:
        raise ValueError(f"{name} requires two labeled training classes")

    # The offset rides a COLUMN, not a positional array or a re-indexed
    # Series: canonical_sort above re-sorts (and resets the index), so an
    # array handed in by position would attach the price to the wrong games
    # and a Series would reindex to NaN. A column travels with its row.
    bm: np.ndarray | None = None
    if base_margin is not None:
        # ``base_margin`` is the NAME of the margin column, read here and
        # therefore after the sort: an array handed in would still be in
        # pre-sort order and the price would attach to the wrong games.
        if base_margin not in train.columns:
            raise ValueError(
                f"{name}: market offset column {base_margin!r} is missing - "
                "the offset arm cannot fit an unpriced game")
        bm = pd.to_numeric(train[base_margin], errors="coerce").to_numpy(float)
        if not np.isfinite(bm).all():
            raise ValueError(
                f"{name}: market offset missing on a training row - the "
                "offset arm cannot fit an unpriced game")

    X_train = member_matrix(name, train)
    pre = (TrainFoldPreprocessor().fit(X_train)
           if name in LINEAR_MEMBERS or name == "randomforest" else None)
    y_train = labels.to_numpy(dtype=int)
    if bm is not None and name in LINEAR_MEMBERS:
        # A TRUE offset: the market logit is held at a coefficient of 1.0
        # and only the residual is learned, the same contract the trees are
        # fitted under (NHL parity).
        return (_fit_offset_linear(member_fit_input(name, X_train, pre),
                                   y_train, bm), pre)
    if name == "xgboost":
        # Early stopping watches a chronological tail of the TRAINING fold,
        # never the validation window - see _early_stop_watch. The tail is
        # carved out of the fit set, so the gradient never sees it and the
        # fit rows are strictly prior to the watch rows in game order.
        watch = _early_stop_watch(X_train, y_train, pre)
        if watch is not None:
            fit_rows, y_fit_rows, X_watch = watch
            model = _make_member(
                name,
                config.XGBOOST_FOLD_ROUNDS,
                config.XGBOOST_EARLY_STOP,
            )
            X_fit_set = member_fit_input(
                name, X_train.loc[fit_rows], pre)
            probe_kwargs = {"eval_set": [(X_watch,
                                          y_train[len(fit_rows):])]}
            if bm is not None:
                # The watch tail needs its own offset, or the probe measures
                # the residual's margin without the price and picks a round
                # count for a different problem.
                probe_kwargs["base_margin"] = bm[: len(fit_rows)]
                probe_kwargs["base_margin_eval_set"] = [bm[len(fit_rows):]]
            try:
                model.fit(X_fit_set, y_fit_rows, verbose=False,
                          **probe_kwargs)
            except TypeError:
                # Older sklearn wrappers do not accept ``verbose`` in fit.
                model.fit(X_fit_set, y_fit_rows, **probe_kwargs)
            # Probe labels are strictly prior. Refit on ALL training rows
            # with the measured budget so folds and serving share one policy.
            rounds = int(model.best_iteration) + 1
            model = _make_member(name, rounds)
            refit_kwargs = ({"base_margin": bm} if bm is not None else {})
            model.fit(member_fit_input(name, X_train, pre), y_train,
                      **refit_kwargs)
            _check_xgb_categories(model, X_train)
            return model, pre
        # Too small for a watch tail: fit the full training fold with the
        # configured ceiling and no stopping - still strictly train-only.
        model = _make_member(name, config.XGBOOST_FOLD_ROUNDS)
        small_kwargs = ({"base_margin": bm} if bm is not None else {})
        model.fit(member_fit_input(name, X_train, pre), y_train,
                  **small_kwargs)
        _check_xgb_categories(model, X_train)
        return model, pre
    # Every other member: fit on the whole training fold, no early stop.
    model = _make_member(name)
    X_train_fit = member_fit_input(name, X_train, pre)
    fit_kwargs = {}
    if name == "lightgbm":
        categorical = [c for c in config.TREE_CATEGORICAL_COLS
                       if c in getattr(X_train_fit, "columns", [])]
        if categorical:
            fit_kwargs["categorical_feature"] = categorical
        if bm is not None:
            # LightGBM's sklearn wrapper has no predict-time init_score: the
            # raw margin is the learned residual, so _predict adds the
            # offset back before the sigmoid.
            fit_kwargs["init_score"] = bm
    model.fit(X_train_fit, y_train, **fit_kwargs)
    return model, pre


def _check_xgb_categories(model, X: pd.DataFrame) -> None:
    booster = model.get_booster()
    types = dict(zip(booster.feature_names, booster.feature_types))
    if any(types.get(c) != "c" for c in config.TREE_CATEGORICAL_COLS if c in X):
        raise RuntimeError("XGBoost team columns must remain categorical")


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
    # Blend-policy bounds (diversity/robustness governance, see
    # config.ENSEMBLE_MEMBER_CAPS / _FLOORS).  Without a policy
    # these are the plain simplex bounds; with one, the optimiser
    # searches only the feasible slice.  The bounds are constants
    # fixed here, never tuned on the OOF frame, so a policy is
    # set before any out-of-fold evidence is seen.
    caps = dict(getattr(config, "ENSEMBLE_MEMBER_CAPS", None) or {})
    floors = dict(getattr(config, "ENSEMBLE_MEMBER_FLOORS", None) or {})
    lo = np.array([float(floors.get(name, 0.0)) for name in names])
    hi = np.array([min(float(caps.get(name, 1.0)), 1.0)
                   for name in names])
    # A policy whose floors cannot sum to one (or whose caps
    # cannot reach one), or that floors a member above its own
    # cap, has no feasible blend at all: honouring it would mean
    # returning an impossible weight vector.  The policy is
    # refused loudly and the plain simplex governs.
    if (lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9
            or (lo > hi + 1e-9).any()):
        logger.warning(
            "ensemble blend policy infeasible (floors sum %.4f, "
            "caps sum %.4f); ignoring caps/floors for this fit",
            lo.sum(), hi.sum())
        lo = np.zeros(len(names))
        hi = np.ones(len(names))
    if len(names) == 1:
        # The only feasible blend is the member itself.  A cap
        # below 1.0 on a lone member made the policy infeasible
        # and was refused above, so this can never publish an
        # infeasible 1.0.
        return {names[0]: 1.0}

    solo = {name: loss(P[:, i]) for i, name in enumerate(names)}
    best = min(solo, key=solo.get)
    # The solo-best (one-hot) fallback is only valid when the
    # one-hot solution is itself feasible under the policy: a
    # cap of 0.70 on the elastic net must not be defeated by
    # returning the pure elastic net, and tree floors must not
    # be defeated by returning zero weight for the trees.  Check
    # every member's bound against its one-hot value (1.0 for
    # the solo best, 0.0 for the rest).
    one_hot_feasible = all(
        lo[i] <= (1.0 if name == best else 0.0) <= hi[i]
        for i, name in enumerate(names))

    from scipy.optimize import minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=list(zip(lo.tolist(), hi.tolist())),
        constraints=({"type": "eq",
                      "fun": lambda w: float(w.sum() - 1.0)},),
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        if one_hot_feasible:
            return {name: float(name == best) for name in names}
        # The optimiser failed AND the pure best member violates
        # the policy: return the deterministic feasible point
        # (equal weights projected onto the policy box) rather
        # than an infeasible one-hot.
        fallback = np.full(len(names), 1.0 / len(names))
        for _ in range(100):
            fallback = np.clip(fallback, lo, hi)
            fallback = fallback / fallback.sum()
        return {name: round(float(value), 4)
                for name, value in zip(names, fallback)}

    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    weights = weights / weights.sum() if weights.sum() else w0
    if one_hot_feasible and solo[best] < objective(weights) - 1e-12:
        return {name: float(name == best) for name in names}
    rounded = {}
    for name, value in zip(names, weights):
        bound = (float(lo[names.index(name)]),
                 float(hi[names.index(name)]))
        # Clamp before rounding so the published vector can never
        # leave the policy box by a rounding step; the residue
        # fix below then redistributes at most ~1e-4.
        rounded[name] = round(
            min(max(float(value), bound[0]), bound[1]), 4)
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        # Put the rounding residue on a member with headroom
        # under its cap (the largest such member, preserving the
        # no-policy choice), so the fix that keeps the weights
        # summing to one cannot push a capped member over its
        # bound.
        at_cap = {name for name in names
                  if rounded[name] >= float(hi[names.index(name)])
                  - 1e-9}
        candidates = [name for name in names
                      if name not in at_cap] or list(names)
        top = max(candidates, key=rounded.get)
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded



def walk_forward_oof(
    game_df: pd.DataFrame,
    date_col: str = "gameday",
    progress_every: int = 25,
    fold_list=None,
    arm: str = "incumbent",
    market_col: str = "market_p_home",
) -> dict:
    """Causal expanding OOF with prior-fold weights and calibration.

    ``arm`` selects WHAT is being scored (NHL parity, 2026-11-15):
      ``incumbent`` - the shipped model, no market input at all;
      ``offset``    - the same members started from the de-vigged market
                      logit (``base_margin``), so they learn the RESIDUAL to
                      the price instead of re-deriving it;
      ``market``    - the de-vigged price itself, no model.
    The offset and market arms score only games that carry a price, so the
    three arms are compared on ONE population instead of three.
    """
    if arm not in ("incumbent", "offset", "market"):
        raise ValueError(f"unknown arm {arm!r}")
    df = folds_mod.canonical_sort(game_df, date_col)
    market = None
    if arm != "incumbent":
        if market_col not in df.columns:
            raise ValueError(
                f"arm={arm!r} needs a {market_col!r} column of de-vigged "
                "market probabilities")
        market = pd.to_numeric(df[market_col], errors="coerce")
        if not (market.notna() & market.gt(0) & market.lt(1)).any():
            raise ValueError(
                f"arm={arm!r}: {market_col!r} carries no priced game")
    folds = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col)
    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    prior_weights = dict(config.ENSEMBLE_WEIGHTS)
    pooled: dict[str, list[float]] = {name: [] for name in config.ENSEMBLE_MEMBERS}
    pooled_y: list[float] = []

    for fold in folds:
        train, val = df.loc[fold.train_idx], df.loc[fold.val_idx]
        bm_train: pd.Series | None = None
        bm_val: np.ndarray | None = None
        if market is not None:
            # Price coverage, not the calendar, decides which rows this arm
            # may score: an unpriced game has no offset and no market
            # probability, so it cannot be evidence for or against either.
            kept_tr = market.loc[train.index]
            kept_va = market.loc[val.index]
            ok_tr = kept_tr.notna() & kept_tr.gt(0) & kept_tr.lt(1)
            ok_va = kept_va.notna() & kept_va.gt(0) & kept_va.lt(1)
            train, val = train.loc[ok_tr], val.loc[ok_va]
            if len(val) == 0:
                logger.info(
                    "moneyline fold %s skipped: no priced game to score",
                    fold.fold_id)
                continue
            bm_val = _market_logit(kept_va.loc[ok_va].to_numpy(float))
            if arm == "offset":
                bm_train = pd.Series(
                    _market_logit(kept_tr.loc[ok_tr].to_numpy(float)),
                    index=train.index)
            # Carry the margin ON the frame: folds re-sort and re-index, and
            # a separate array silently desynchronises from its rows.
            val = val.assign(_offset_margin=bm_val)
            if bm_train is not None:
                train = train.assign(_offset_margin=bm_train.to_numpy(float))
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
            if arm == "market":
                member_p[name] = None
                continue
            try:
                model, pre = _fit_member(
                    name, train,
                    base_margin=("_offset_margin" if bm_train is not None
                                 else None))
                member_p[name] = _predict(model, name,
                                          member_matrix(name, val), pre,
                                          base_margin=bm_val)
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
        if market is not None:
            # Kept on the row so any arm can be paired with the price it
            # was measured against, without a second join.
            row["market_p_home"] = market.loc[val.index].to_numpy(float)
        val = val.drop(columns=["_offset_margin"], errors="ignore")

        # --- WHICH ROWS MAY GRADE THE MODEL -------------------------------
        # Policy (2026-10-03): only regular-season rows from windows at or
        # above MIN_VAL_FOLD_GAMES feed the blend optimizer and the in-loop
        # Platt map. Postseason rows and provisional (under-gate) windows are
        # still FIT and still SCORED -- they land in ``oof`` and therefore in
        # ``*_predictions_history`` -- but they are never evidence FOR the
        # model, only evidence ABOUT it. Two reasons, both observed:
        #   * the gate's stated purpose was to keep thin post-season folds out
        #     of pooled metrics, yet ``is_partial_tail`` re-admitted the single
        #     thinnest one (n=3 Finals), so a 3-game window graded the model
        #     while a 34-game playoff window did not;
        #   * postseason is trained on by every fold (train = strictly prior,
        #     no season-type filter) but was never scored, so the reported
        #     pooled metrics silently described only ~93% of the population.
        is_post = (folds_mod.postseason_flag(val["game_type"]).to_numpy()
                   if "game_type" in val
                   else (pd.to_numeric(val["is_playoffs"], errors="coerce")
                         .fillna(0).gt(0.5).to_numpy()
                         if "is_playoffs" in val
                         else np.zeros(len(val), dtype=bool)))
        fold_provisional = bool(getattr(fold, "provisional", False))
        grades = np.zeros(len(val), dtype=bool) if fold_provisional else ~is_post
        row["is_playoffs"] = is_post.astype(float)
        row["season_type"] = getattr(fold, "season_type", "regular")
        row["provisional"] = fold_provisional
        row["grades_pooled"] = grades

        # Never rewrite this causal blend using weights learned later.
        row["p_ensemble"] = (_blend(row, prior_weights) if arm != "market"
                             else row["market_p_home"].to_numpy(float))
        for name in config.ENSEMBLE_MEMBERS:
            row[f"weight_{name}"] = fold_weights.get(name, 0.0)

        # Add the scored fold only after its prediction has been fixed.
        # These values can train the next fold.
        for name in config.ENSEMBLE_MEMBERS:
            pooled[name].extend(row[f"p_{name}"].to_numpy(float)[grades].tolist())
        pooled_y.extend(row.home_win.to_numpy(float)[grades].tolist())
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
                "season_type": getattr(fold, "season_type", "regular"),
                "provisional": fold_provisional,
                "n_grading": int(grades.sum()),
                "weights": fold_weights,
            }
        )
        parts.append(row)
        if progress_every and (fold.fold_id + 1) % progress_every == 0:
            logger.info("moneyline OOF fold %d/%d", fold.fold_id + 1, len(folds))

    # NHL causal-evaluation parity: final weights replay only as a diagnostic.
    # Headline/calibration/history probabilities retain each origin's weights.
    gate_audit: list[dict] = []
    if parts:
        deployed_w = dict(prior_weights)
        # Uncapped denominators around the published-blend prequential
        # calibrator loop — the summary below reports the true fallback
        # count behind fit_platt's capped WARNING lines (2026-10-06 NBA
        # log review, MLB/NHL parity).
        _fit0 = getattr(fit_platt, "_fit_total", 0)
        _min0 = getattr(fit_platt, "_min_total", 0)
        _degen0 = getattr(fit_platt, "_degen_total", 0)
        for row in parts:
            row["p_ensemble_causal"] = row["p_ensemble"].to_numpy(float).copy()
            row["p_ensemble_retrospective"] = (
                row["p_ensemble"].to_numpy(float).copy() if arm == "market"
                else _blend(row, deployed_w))
        # Apply the same nested prior-evidence gate at every fold origin.
        pub_p: list[float] = []
        pub_y: list[float] = []
        for row in parts:
            p_k = row["p_ensemble"].to_numpy(float)
            if arm == "market":
                # A de-vigged price is already a probability. Re-fitting a
                # Platt map on it would grade a different object than "the
                # market", and the arm exists to be the market.
                row["p_ensemble_calibrated"] = p_k
                continue
            cal_k, gate = gated_calibrator(np.asarray(pub_p, dtype=float),
                                            np.asarray(pub_y, dtype=float))
            gate["fold_id"] = int(row.fold_id.iloc[0])
            gate_audit.append(gate)
            calibrated = moneyline_apply(p_k, cal_k)
            row["p_ensemble_calibrated"] = np.where(
                np.isfinite(calibrated), calibrated, p_k)
            gk = (row["grades_pooled"].astype(bool).to_numpy()
                  if "grades_pooled" in row.columns
                  else np.ones(len(row), dtype=bool))
            pub_p.extend(p_k[gk].tolist())
            pub_y.extend(row["home_win"].to_numpy(float)[gk].tolist())

        # Keep the causal headline and retrospective replay visibly distinct.
        logger.info(
            "Published blend: %d OOF rows — headline metrics grade the causal "
            "rolling blend; final weights %s replay only in "
            "p_ensemble_retrospective (NOT OOF)",
            sum(len(r) for r in parts),
            {k: f"{v:.1%}" for k, v in sorted(deployed_w.items())},
        )
        _fits = getattr(fit_platt, "_fit_total", 0) - _fit0
        _mins = getattr(fit_platt, "_min_total", 0) - _min0
        _degens = getattr(fit_platt, "_degen_total", 0) - _degen0
        if _mins or _degens:
            logger.info(
                "Calibration: %d of %d published-blend prequential Platt fits "
                "fell back to identity (%d below minimum or single-class, "
                "%d degenerate; the WARNING lines print only the first 3)",
                _mins + _degens, _fits + _mins, _mins, _degens)

    # No post-blend market shrink: the offset arm ships exactly as fitted
    # (offset at 1.0 -> blend -> Platt), matching the NHL mechanics.
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(oof):
        season_split = {
            "regular_rows": int((~oof.is_playoffs.astype(bool)).sum()),
            "postseason_rows": int(oof.is_playoffs.astype(bool).sum()),
            "provisional_rows": int(oof.provisional.astype(bool).sum()),
            "grading_rows": int(oof.grades_pooled.astype(bool).sum()),
        }
    else:
        season_split = {"regular_rows": 0, "postseason_rows": 0,
                        "provisional_rows": 0, "grading_rows": 0}
    return {"oof": oof, "arm": arm, "member_weights": prior_weights,
            "calibration_gate": {"folds": gate_audit},
            "season_split": season_split,
            "fold_table": pd.DataFrame(fold_rows)}


def fit_final_models(game_df: pd.DataFrame, offset_col: str | None = None):
    """Fit all members on every labeled settled game for production serving.

    ``offset_col`` is the market-offset column name (``market_logit``). When
    given, the members are fitted on the RESIDUAL to the de-vigged price and
    the population is restricted to priced games - an offset member cannot be
    fitted on a row with no price (NHL parity).
    """
    game_df = folds_mod.canonical_sort(game_df)
    labels = _binary_labels(game_df)
    frame = game_df.loc[labels.notna()].copy()
    if offset_col is not None:
        if offset_col not in frame.columns:
            raise ValueError(f"fit_final_models: offset column {offset_col!r} missing")
        frame = frame[frame[offset_col].notna()].copy()
    if len(frame) < 2 or _binary_labels(frame).nunique() < 2:
        raise ValueError("NBA final moneyline fit requires both outcome classes")
    models: dict[str, dict[str, Any]] = {}
    for name in config.ENSEMBLE_MEMBERS:
        model, pre = _fit_member(name, frame, base_margin=offset_col)
        models[name] = {"model": model, "pre": pre}
    # Keep the historical two-tuple contract; per-member preprocessors live
    # in each entry and are used by predict_slate.
    return models, None


def predict_slate(models: dict, slate_df: pd.DataFrame,
                  weights: dict[str, float],
                  base_margin=None) -> np.ndarray:
    """Predict and blend a pending slate with the deployed OOF weights.

    ``base_margin`` carries the slate's market logit when the deployed members
    were fitted with an offset; it must be supplied for every row, and rows
    without a price stay NaN rather than being filled in.
    """
    predictions: list[np.ndarray] = []
    names: list[str] = []
    for name in config.ENSEMBLE_MEMBERS:
        entry = models.get(name)
        if not entry:
            continue
        try:
            predictions.append(
                _predict(entry["model"], name,
                         member_matrix(name, slate_df), entry.get("pre"),
                         base_margin=base_margin)
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


def fit_platt(p_fav, y_fav):
    p = np.asarray(p_fav, dtype=float)
    y = np.asarray(y_fav, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y) & (p > 0) & (p < 1)
    p, y = p[ok], y[ok]
    if len(y) < config.MIN_OOF_FOR_FIT or len(np.unique(y)) < 2:
        # Evidence (2026-10-06 NBA log review, MLB/NHL parity): this
        # fallback used to be SILENT — the delivered log carried no
        # calibration line at all, so a reviewer could not tell a fitted
        # map from an identity fallback. Capped at 3 because the
        # published-blend prequential loop calls fit_platt once per fold
        # and early small folds miss the minimum repeatedly; the UNCAPPED
        # _min_total feeds the per-block summary in walk_forward_oof.
        _n = getattr(fit_platt, "_min_logged", 0)
        fit_platt._min_total = getattr(fit_platt, "_min_total", 0) + 1
        if _n < 3:
            if len(y) < config.MIN_OOF_FOR_FIT:
                logger.warning(
                    "Calibration: %d OOF games < %d minimum — identity map",
                    len(y), config.MIN_OOF_FOR_FIT)
            else:
                logger.warning(
                    "Calibration: single-class OOF labels (n=%d) — identity "
                    "map", len(y))
        fit_platt._min_logged = _n + 1
        return None
    # Uncapped attempt counter — denominator for the fallback summary
    # (incremented here so degenerate fits count as attempts too).
    fit_platt._fit_total = getattr(fit_platt, "_fit_total", 0) + 1
    from sklearn.linear_model import LogisticRegression
    z = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y.astype(int))
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        # WARNING (degenerate = real signal), rate-limited so the
        # prequential loop cannot flood the log on early small folds.
        # Two counters: _degen_logged drives the 3-line cap; the UNCAPPED
        # _degen_total feeds the summary so a reviewer can see HOW MANY
        # fits degenerated (2026-10-06 NBA log review, MLB/NHL parity: a
        # degenerate fit's sample size must be visible in the log).
        _degen_n = getattr(fit_platt, "_degen_logged", 0)
        fit_platt._degen_total = getattr(fit_platt, "_degen_total", 0) + 1
        if _degen_n < 3:
            logger.warning(
                "Calibration: degenerate Platt params (a=%s, n=%d) — identity map",
                a, len(y))
        fit_platt._degen_logged = _degen_n + 1
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


def gated_calibrator(p_home, home_win, grades=None) -> tuple[dict | None, dict]:
    """Use one nested chronological log-loss gate at every prediction origin."""
    p, y = np.asarray(p_home, float), np.asarray(home_win, float)
    if p.ndim != 1 or p.shape != y.shape:
        raise ValueError("calibration probabilities/targets must be aligned vectors")
    eligible = np.ones(len(p), bool) if grades is None else np.asarray(grades, bool)
    if eligible.shape != p.shape:
        raise ValueError("calibration grading mask must align with predictions")
    ok = eligible & np.isfinite(p) & np.isin(y, [0, 1]) & (p > 0) & (p < 1)
    p, y = p[ok], y[ok]
    rec = {"n_prior": len(p), "decision": "identity",
           "reason": "insufficient_prior_evidence", "raw_logloss": None,
           "nested_holdout_logloss": None}
    hold = max(config.CAL_GATE_MIN_HOLDOUT,
               int(round(config.CAL_GATE_HOLDOUT_FRAC * len(p))))
    n_fit = len(p) - hold
    if n_fit < config.MIN_OOF_FOR_FIT:
        return None, rec
    candidate = moneyline_fit(p[:n_fit], y[:n_fit])
    if candidate is None:
        rec["reason"] = "candidate_declined"
        return None, rec
    from sklearn.metrics import log_loss
    raw = float(log_loss(y[n_fit:], p[n_fit:], labels=[0, 1]))
    cal = float(log_loss(y[n_fit:], moneyline_apply(p[n_fit:], candidate), labels=[0, 1]))
    rec.update(raw_logloss=raw, nested_holdout_logloss=cal)
    if cal >= raw - config.CAL_GATE_EPS:
        rec["reason"] = "gated_no_gain"
        return None, rec
    final = moneyline_fit(p, y)
    rec.update(decision="fitted" if final else "identity",
               reason="accepted" if final else "final_fit_declined")
    return final, rec


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
