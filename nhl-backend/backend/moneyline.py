"""Production NHL moneyline: XGBoost / LightGBM / elastic-net logistic
ensemble with expanding walk-forward OOF, MLB-parity rolling blend weights
(simplex-constrained SLSQP minimizing pooled OOF log-loss in LOGIT space,
re-earned after every fold from strictly-prior evidence; no floor, no cap —
a member may earn 0% or 100%), and favored-space Platt calibration fit ONLY
on valid OOF predictions.

The ensemble is deliberately IDENTICAL to MLB's (same members, same tuned
params, same blending, same calibration contract). Determinism: explicit
seeds on every member; preprocessing (median imputation + scaling for
linear members) is fit on training data only.
"""
from __future__ import annotations

import logging
from typing import Any

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
    """Train-median imputation + standard scaling for the linear member.

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
        # with NO training observations imputes to 0.0 deterministically —
        # never a fabricated signal, just neutral.
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
# CAUSAL FOLD ROUNDS (ported from MLB's 2026-09-30 PIT remediation):
# each fold's early-stopped probe is a MEASUREMENT ONLY — its best_iteration
# enters this list and informs STRICTLY LATER folds. The SHIPPED fold model
# is refit without any eval_set at the median of PRIOR folds' measurements,
# so the window that gets scored never selects the model that scores it.
# 2a9554b measured the alternative on MLB's exact geometry: rounds selected
# on the scored window flattered the OOF by ~0.011 logloss / ~0.04 AUC.
_LAST_XGB_BEST_ROUNDS: list[int] = []


def causal_xgb_rounds(prior_bests: list[int]) -> int:
    """Shipped round count for a fold from PRIOR measurements only.

    Median of the given best-iteration list (even length: lower median,
    kept simple and deterministic); falls back to the config priors when no
    measurements exist. Pure function so the tests pin the selection rule.
    """
    if not prior_bests:
        return config.XGBOOST_FOLD0_ROUNDS
    s = sorted(int(b) for b in prior_bests if b and int(b) > 0)
    if not s:
        return config.XGBOOST_REFIT_ROUNDS
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2


def _make_member(name: str, fold: bool = False,
                 n_estimators: int | None = None,
                 causal: bool = False):
    """Construct one configured member, optionally with MLB fold settings.

    Fold XGBoost receives the same generous round ceiling and early-stopping
    window as MLB. The fit-only production refit deliberately uses the base
    parameter set and therefore has no validation dependency.

    ``n_estimators`` overrides the xgboost round budget (the causal fold
    refit ships the prior-folds' median of measured rounds instead of the
    2000-round early-stop ceiling); ``causal=True`` drops
    early_stopping_rounds entirely — the shipped fold model is a FIXED-budget
    fit with no eval_set dependency, so it cannot be steered by its own
    validation window.
    """
    if name == "xgboost":
        from xgboost import XGBClassifier
        params = dict(config.XGBOOST_PARAMS)
        if fold:
            params["n_estimators"] = (config.XGBOOST_FOLD_ROUNDS
                                      if n_estimators is None
                                      else int(n_estimators))
            if causal:
                params.pop("early_stopping_rounds", None)
            else:
                params["early_stopping_rounds"] = config.XGBOOST_EARLY_STOP
        elif n_estimators is not None:
            params["n_estimators"] = int(n_estimators)
        return XGBClassifier(**params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**config.LIGHTGBM_PARAMS)
    if name == "elasticnet":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**config.ELASTICNET_PARAMS)
    raise KeyError(f"unknown ensemble member {name!r}")


LINEAR_MEMBERS = {"elasticnet"}


def member_matrix(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The feature matrix a member consumes (model-family representation)."""
    return feat_mod.linear_view(df) if name in LINEAR_MEMBERS else feat_mod.tree_view(df)


def member_matrix_ndarray(name: str, df: pd.DataFrame,
                          pre: "TrainFoldPreprocessor | None" = None) -> np.ndarray:
    """The member's matrix as a plain ndarray (explainers, diagnostics).

    Tree members: the named tree_view frame (which carries the categorical
    team-ID pair) passed through member_fit_input unchanged. Linear members:
    the fitted preprocessor's imputed+scaled ndarray — callers MUST pass the
    member's own ``pre``.
    """
    return np.asarray(member_fit_input(name, member_matrix(name, df), pre))


def member_fit_input(name: str, X_raw: pd.DataFrame,
                     pre: "TrainFoldPreprocessor | None"):
    """The ONE authoritative representation handed to member.fit().

    Representation contract (mirrors MLB's model-specific routing):
      * linear members -> the fitted preprocessor's ndarray (imputed+scaled)
      * tree members   -> the NAMED tree-view DataFrame, feature names
        preserved (sklearn/LightGBM feature-name warnings are avoided by
        pinning the named frame at BOTH fit and predict).
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

    Rolling per-fold blend weighting (MLB structural parity): fold 0 blends
    on the static ENSEMBLE_WEIGHTS priors (1/3 each); after each fold the
    blend weights are re-earned by minimizing pooled OOF log-loss over the
    accumulated prior+current fold member predictions (LOGIT space,
    simplex-constrained SLSQP, no floor/cap), so the NEXT fold's blend is
    weighted by evidence strictly before it (causal — never sees what it
    scores). The final rolling update — the optimum over the whole
    walk-forward population — is the returned/shipped weight and feeds
    serving and the dashboard.

    Returns a dict with:
      oof: DataFrame (game_id, gameday, fold_id, per-member p_home, ensemble)
      member_weights: last rolling optimized blend weights
      fold_table: per-fold diagnostics
    """
    df = folds_mod.canonical_sort(game_df, date_col)
    fold_list = fold_list if fold_list is not None else folds_mod.make_folds(df, date_col=date_col)

    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    prior_weights = dict(config.ENSEMBLE_WEIGHTS)
    oof_members: dict[str, list[float]] = {n: [] for n in config.ENSEMBLE_MEMBERS}
    oof_y: list[float] = []
    _last_weights: dict[str, float] = dict(prior_weights)
    # Reset the causal fold-rounds ledger so repeated walks in one process
    # (tests, dashboards, retunes) never carry measurements across runs.
    _LAST_XGB_BEST_ROUNDS.clear()

    n_folds = len(fold_list)
    # Cadence and the final-fold guarantee live in folds.progress_checkpoints:
    # a fixed 25-fold cadence prints only "25/46" on a 46-fold run (50 never
    # arrives) and never says "done", which is what made the 2026-09-25 RFE
    # sweep print the same line 121 times over ~24 minutes.
    announce = set(folds_mod.progress_checkpoints(n_folds, progress_every))

    for fold in fold_list:
        train = df.loc[fold.train_idx]
        val = df.loc[fold.val_idx]
        y_train = train["home_win"].astype(int).to_numpy()

        member_p: dict[str, np.ndarray] = {}
        y_val = val["home_win"].astype(int).to_numpy()
        for name in config.ENSEMBLE_MEMBERS:
            X_tr_raw = member_matrix(name, train)
            X_va_raw = member_matrix(name, val)
            refit_rounds: int | None = None
            if name in LINEAR_MEMBERS:
                pre = TrainFoldPreprocessor().fit(X_tr_raw)
            else:
                pre = None
                if name == "xgboost":
                    # CAUSAL FOLD ROUNDS (MLB's 2026-09-30 PIT remediation,
                    # ported): the fold's own val window must never select
                    # the model that scores it. The shipped model below is
                    # refit at the median of PRIOR folds' measured rounds.
                    refit_rounds = causal_xgb_rounds(_LAST_XGB_BEST_ROUNDS)
            try:
                model = _make_member(name, fold=True,
                                     n_estimators=refit_rounds,
                                     causal=bool(refit_rounds is not None))
                X_tr = member_fit_input(name, X_tr_raw, pre)
                X_va = member_fit_input(name, X_va_raw, pre)
                fit_kwargs = {}
                if name == "xgboost":
                    # The SHIPPED fold model: FIXED round budget from strictly
                    # prior folds, no eval_set — it cannot be steered by the
                    # window it is about to be scored on.
                    fit_kwargs = {"verbose": False}
                elif name == "lightgbm":
                    # MLB supplies the same fold evaluation set to LightGBM.
                    # (Inert for round selection: LIGHTGBM_PARAMS has no
                    # early stopping, so nothing is tuned on the window.)
                    fit_kwargs = {
                        "eval_set": [(X_va, y_val)],
                        "categorical_feature": config.TREE_CATEGORICAL_COLS,
                    }
                model.fit(X_tr, y_train, **fit_kwargs)
                if name == "xgboost":
                    # The early-stopped probe runs AFTER the shipped fit and
                    # is a MEASUREMENT ONLY: its best_iteration enters the
                    # causal list and informs strictly LATER folds (never
                    # this one). The probe is never scored.
                    probe = _make_member(name, fold=True)
                    probe.fit(X_tr, y_train, eval_set=[(X_va, y_val)],
                              verbose=False)
                    # Missing best_iteration (a stubbed probe) is NOT a
                    # measurement: 0+1 would fabricate a 1-round fold and
                    # poison every later fold's median.
                    _best = getattr(probe, "best_iteration", None)
                    if _best is not None and int(_best) >= 0:
                        _LAST_XGB_BEST_ROUNDS.append(int(_best) + 1)
                    del probe
                member_p[name] = _member_predict_proba(model, name, X_va_raw, pre)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fold %s member %s failed: %s",
                               fold.fold_id, name, exc)
                member_p[name] = None

        # --- WHICH ROWS MAY GRADE THE MODEL -------------------------------
        # Policy (2026-10-03): only regular-season rows from windows at or
        # above MIN_VAL_FOLD_GAMES feed the blend optimizer. Postseason and
        # provisional (under-gate) windows are still FIT and still SCORED --
        # they land in ``oof`` and therefore in ``*_predictions_history`` --
        # but they are never evidence FOR the model, only evidence ABOUT it.
        # NHL retained every window on 2026-09-30; this makes that retention
        # meaningful by keeping a 7-game Stanley Cup week from carrying the
        # same SLSQP vote as a 70-game week.
        if "game_type" in val:
            is_post = folds_mod.postseason_flag(val["game_type"]).to_numpy()
        elif "is_playoffs" in val:
            is_post = (pd.to_numeric(val["is_playoffs"], errors="coerce")
                       .fillna(0).gt(0.5).to_numpy())
        else:
            is_post = np.zeros(len(val), dtype=bool)
        fold_provisional = bool(getattr(fold, "provisional", False))
        grades = (np.zeros(len(val), dtype=bool) if fold_provisional
                  else ~is_post)

        rows = pd.DataFrame({
            "game_id": val["game_id"].to_numpy(),
            "gameday": pd.to_datetime(val[date_col]).to_numpy(),
            "season": val["season"].to_numpy(),
            "fold_id": fold.fold_id,
            "home_win": val["home_win"].astype(float).to_numpy(),
        })
        rows["is_playoffs"] = is_post.astype(float)
        rows["season_type"] = getattr(fold, "season_type", "regular")
        rows["provisional"] = fold_provisional
        rows["grades_pooled"] = grades
        for name in config.ENSEMBLE_MEMBERS:
            p = member_p.get(name)
            # A failed member is recorded as an all-NaN float column, never
            # as None (an object-dtype None column poisons the numeric
            # consumers that read it).
            rows[f"p_{name}"] = (np.asarray(p, dtype=float) if p is not None
                                 else np.full(len(val), np.nan))
        # Blend this fold using only information earned before this fold
        # (fold 0 rides the static 1/3 priors).
        fold_weights = dict(_last_weights)
        rows["p_ensemble"] = _blend(rows, fold_weights)
        # Only after scoring the fold may its outcomes enter the weight
        # window: accumulate this fold's GRADING member predictions, then
        # re-earn the blend weights for the NEXT fold (causal walk-forward
        # weighting contract). Non-grading rows are excluded here and only
        # here -- they are still in ``rows``, so they are scored and shipped.
        for name in config.ENSEMBLE_MEMBERS:
            p_member = member_p.get(name)
            if p_member is not None and len(p_member):
                oof_members[name].extend(
                    np.asarray(p_member, dtype=float)[grades].tolist())
        oof_y.extend(rows["home_win"].astype(float).to_numpy()[grades].tolist())
        rolling = compute_adaptive_weights(oof_members, np.asarray(oof_y, dtype=float))
        if rolling:
            _last_weights = rolling

        fold_rows.append({
            "fold_id": fold.fold_id,
            "val_start": str(fold.val_start.date()),
            "val_end": str(fold.val_end.date()),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "season_type": getattr(fold, "season_type", "regular"),
            "provisional": fold_provisional,
            "n_grading": int(grades.sum()),
        })
        oof_parts.append(rows)
        if (fold.fold_id + 1) in announce:
            logger.info("moneyline OOF fold %d/%d", fold.fold_id + 1, n_folds)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()

    # Weight-earning evidence (2026-09-29 19:56 post-adoption review): the    # weights line alone cannot explain a zero. Log the pooled per-member
    # loglosses the SLSQP takeover rule compares, the rolling-blend pooled
    # logloss, and the causal fold-round budgets this walk measured — then
    # flag any zero-weight member loudly. Background: on a NEW machine,
    # xgboost re-buckets histograms in thread-order-dependent float
    # summation, so the tree member's OOF (and therefore its earned weight)
    # is machine-sensitive — lightgbm/elasticnet reproduce bit-identically,
    # xgboost does not (local 0.5942 AUC vs 0.5836 on the 19:56 run). The
    # 0% corner was the optimizer correctly reading THAT machine's honest
    # evidence; these lines make every future corner auditable from the
    # log alone instead of surprising anyone. (2026-09-29 refinement: the
    # nthread=1 pin removed the within-environment thread noise; lgbm and
    # elasticnet then reproduced bit-identically across machines AND library
    # versions, while xgb still tracks its build — Kaggle 3.2.0 ll=0.67625
    # vs local 3.4.0 ll=0.67382 on the identical frame. Weights are earned
    # where the model serves, so serving numbers stay honest either way.)
    # The logged pooled loglosses are the GRADING population's -- the rows
    # the optimizer actually saw. Reporting them over every scored row would
    # describe evidence the weights were never earned from.
    _grade = (oof[oof["grades_pooled"].astype(bool)]
              if len(oof) and "grades_pooled" in oof else oof)
    if len(oof) and "is_playoffs" in oof:
        season_split = {
            "regular_rows": int((~oof["is_playoffs"].astype(bool)).sum()),
            "postseason_rows": int(oof["is_playoffs"].astype(bool).sum()),
            "provisional_rows": int(oof["provisional"].astype(bool).sum()),
            "grading_rows": int(oof["grades_pooled"].astype(bool).sum()),
        }
    else:
        season_split = {"regular_rows": len(oof), "postseason_rows": 0,
                        "provisional_rows": 0, "grading_rows": len(oof)}
    _y_all = _grade["home_win"].to_numpy(dtype=float)
    _member_ll: dict[str, float] = {}
    for _m in config.ENSEMBLE_MEMBERS:
        _p = _grade[f"p_{_m}"].to_numpy(dtype=float)
        _ok = ~np.isnan(_p)
        if not _ok.any():
            _member_ll[_m] = float("nan")
            continue
        _pc = np.clip(_p[_ok], 1e-9, 1 - 1e-9)
        _yc = _y_all[_ok]
        _member_ll[_m] = float(-np.mean(
            _yc * np.log(_pc) + (1 - _yc) * np.log(1 - _pc)))
    _pe = _grade["p_ensemble"].to_numpy(dtype=float)
    _ok_e = ~np.isnan(_pe)
    _blend_ll = float("nan")
    if _ok_e.any():
        _pc = np.clip(_pe[_ok_e], 1e-9, 1 - 1e-9)
        _yc = _y_all[_ok_e]
        _blend_ll = float(-np.mean(
            _yc * np.log(_pc) + (1 - _yc) * np.log(1 - _pc)))
    _budgets = list(_LAST_XGB_BEST_ROUNDS)
    _budget_note = (
        f"median {sorted(_budgets)[len(_budgets) // 2]} over "
        f"{len(_budgets)} fold(s), range {min(_budgets)}-{max(_budgets)}"
        if _budgets else
        f"none recorded (fold-0 prior {config.XGBOOST_FOLD0_ROUNDS} only)")
    logger.info(
        "moneyline member OOF (weight-earning evidence): %s | "
        "rolling-blend pooled logloss %.5f | xgb causal fold budgets: %s",
        ", ".join(f"{m} ll={_member_ll[m]:.5f}"
                  for m in config.ENSEMBLE_MEMBERS),
        _blend_ll, _budget_note)
    for _m in config.ENSEMBLE_MEMBERS:
        if _last_weights.get(_m, 0.0) > 0.0:
            continue
        _others = [v for k, v in _member_ll.items()
                   if k != _m and np.isfinite(v)]
        _best_other = min(_others) if _others else float("nan")
        logger.warning(
            "moneyline blend: %s earned 0.000 weight (member pooled "
            "logloss %.5f vs best other %.5f) — no simplex blend of the "
            "members beats dropping it on this machine's OOF; the evidence "
            "lines above are the audit trail",
            _m, _member_ll.get(_m, float("nan")), _best_other)

    logger.info("moneyline OOF complete: %d fold(s), %d scored row(s), "
                "%d grading / %d postseason / %d provisional, "
                "final weights %s", n_folds, len(oof),
                season_split["grading_rows"], season_split["postseason_rows"],
                season_split["provisional_rows"],
                ", ".join(f"{k}={v:.3f}" for k, v in sorted(_last_weights.items())))

    # The last rolling update is the full-population optimum — the shipped
    # weight for serving and the dashboard.
    weights = dict(_last_weights)
    return {"oof": oof, "member_weights": weights,
            "season_split": season_split,
            "fold_table": pd.DataFrame(fold_rows)}


def compute_adaptive_weights(
    oof_members: dict[str, list[float]], y_oof: np.ndarray
) -> dict[str, float]:
    """Blend weights earned by out-of-sample performance (MLB structural
    parity with mlb-backend/backend/training.py).

    Simplex-constrained SLSQP minimizing POOLED OOF log-loss, applied in
    LOGIT space (clip -> logit -> weighted mean -> sigmoid). There is NO
    floor, cap, or temperature: a member may earn 0% or 100% of the weight.
    A member takes the ENTIRE weight only when its own pooled OOF log-loss
    beats the optimized blend's; otherwise the optimized weights stand. The
    caller re-earns these weights after every walk-forward fold (rolling
    per-fold weighting — each fold's blend is weighted by the PRIOR folds'
    OOF evidence only), so this function sees the accumulated prior+current
    OOF window; the last rolling update is the full-population optimum and
    is what serving and the dashboard ship. The result sums to exactly 1.0.

    A member whose prediction list is absent or shorter than ``y_oof``
    (it failed to predict on at least one fold in the window) is skipped
    entirely, exactly as MLB's optimizer skips members that failed.
    """
    if str(getattr(config, "ADAPTIVE_WEIGHT_METRIC", "logloss")).lower() != "logloss":
        raise ValueError("NHL production ensemble optimization must use logloss")
    y = np.asarray(y_oof, dtype=float)
    if len(y) == 0:
        return {}
    from sklearn.metrics import log_loss
    scores: dict[str, float] = {}
    for name, preds in oof_members.items():
        if preds is None:
            continue
        p = np.asarray(preds, dtype=float)
        if p.ndim != 1 or len(p) != len(y):
            continue
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
            continue
        ll = float(log_loss(y[ok], p[ok], labels=[0, 1]))
        if not np.isfinite(ll):
            continue
        scores[name] = ll
    if not scores:
        return {}

    # Optimized blend, no gates. Weights minimize pooled OOF log-loss over
    # the simplex (w >= 0, sum(w) = 1) — the same rehearsal window the
    # weights are graded on — and the blend is pooled in LOGIT space,
    # matching _blend / predict_slate at serve.
    names = sorted(scores)
    arrays = {n: np.asarray(oof_members[n], dtype=float) for n in names}

    def _logloss_of(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    Z = np.column_stack([
        np.log(np.clip(arrays[n], 1e-7, 1 - 1e-7)
               / (1 - np.clip(arrays[n], 1e-7, 1 - 1e-7)))
        for n in names])
    blend_loss = lambda w: _logloss_of(1.0 / (1.0 + np.exp(-(Z @ w))))  # noqa: E731

    if len(names) == 1:
        return {names[0]: 1.0}
    from scipy.optimize import minimize
    w0 = np.full(len(names), 1.0 / len(names))
    res = minimize(blend_loss, w0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * len(names),
                   constraints=({"type": "eq",
                                 "fun": lambda w: float(w.sum() - 1.0)}),
                   options={"maxiter": 300, "ftol": 1e-9})
    if not res.success or not np.all(np.isfinite(res.x)):
        # Fall back to the best single member rather than serve a
        # malformed weight vector.
        best = min(names, key=lambda n: scores[n])
        return {n: (1.0 if n == best else 0.0) for n in names}
    w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
    w = w / w.sum() if w.sum() > 0 else w0

    best_name = min(names, key=lambda n: scores[n])
    if scores[best_name] < blend_loss(w) - 1e-12:
        return {n: (1.0 if n == best_name else 0.0) for n in names}

    # Round without breaking the exact 1.0 total: give the rounding
    # remainder to the largest weight.
    rounded = {n: round(float(v), 4) for n, v in zip(names, w)}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(rounded, key=lambda n: rounded[n])
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def _logit_blend_matrix(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """LOGIT-space blend of a member-probability matrix: clip -> logit ->
    weighted mean -> sigmoid. Zero-weight members drop out; per-row NaN
    members are skipped (their weight renormalizes across active members),
    mirroring MLB's ensemble_predict pooling."""
    W = np.asarray(w, dtype=float)
    mask = np.isfinite(P)
    active = (W > 0)[None, :] & mask
    Pc = np.clip(np.where(mask, P, 0.5), 1e-7, 1 - 1e-7)
    Z = np.log(Pc / (1 - Pc))
    wv = np.where(active, W[None, :], 0.0)
    wsum = wv.sum(axis=1)
    z = np.divide((Z * wv).sum(axis=1), wsum,
                  out=np.full(len(P), np.nan), where=wsum > 0)
    out = 1.0 / (1.0 + np.exp(-z))
    return np.clip(np.where(wsum > 0, out, np.nan), CLIP, 1.0 - CLIP)


def _blend(oof: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Weighted ensemble probability in LOGIT space (the space the blend
    weights are optimized in — identical ensemble logic at serve); NaN
    members skipped per row."""
    cols = [f"p_{n}" for n in config.ENSEMBLE_MEMBERS
            if f"p_{n}" in oof.columns]
    P = oof[cols].to_numpy(dtype=float)
    w = np.array([weights.get(c[2:], 0.0) for c in cols])
    return _logit_blend_matrix(P, w)


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
    return _logit_blend_matrix(P, w)


# ---------------------------------------------------------------------------
# Platt calibration — fit ONLY on valid OOF predictions, FAVORED space only
# ---------------------------------------------------------------------------
# MLB structural parity:
#   * all fits/applications happen in FAVORED-team space (p_fav = max(p, 1-p),
#     the side with probability > 50%) — never home-team space;
#   * guardrails fall back to the identity map rather than a risky fit:
#     below MIN_OOF_FOR_FIT pooled games, single-class favored labels, or a
#     degenerate/non-positive slope;
#   * a CALIBRATION_MODE switch (platt/identity) gates the moneyline path;
#   * apply-time method-tag enforcement rejects legacy home-space maps.
FAVORED_CALIBRATOR_METHOD = "favored_platt_floor"
FAVORED_PROBABILITY_FLOOR = 0.5

VALID_CALIBRATION_MODES = ("platt", "identity")


def get_calibration_mode() -> str:
    """Active moneyline calibration mode ("platt" or "identity")."""
    import os
    mode = str(os.environ.get("CALIBRATION_MODE") or config.CALIBRATION_MODE).strip().lower()
    if mode not in VALID_CALIBRATION_MODES:
        logger.warning("Calibration: unknown CALIBRATION_MODE %r — falling back to platt", mode)
        return "platt"
    return mode


def fit_platt(p_fav: np.ndarray, y_fav: np.ndarray) -> dict | None:
    """Fit the 2-parameter logistic map p -> sigmoid(a*logit(p) + b) on
    FAVORED-space (p, y) pairs. Deterministic (LBFGS, no randomness).

    Returns {"method": "platt", "a", "b", "n"} or None when the data cannot
    support a fit (below MIN_OOF_FOR_FIT, single class, non-finite inputs,
    degenerate slope <= 0). None means the identity map everywhere.
    """
    y = np.asarray(y_fav, dtype=float)
    p = np.asarray(p_fav, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (p > 0) & (p < 1)
    y, p = y[ok], p[ok]
    n = len(y)
    if n < config.MIN_OOF_FOR_FIT:
        logger.debug("Calibration: %d OOF games < %d minimum — using identity map",
                     n, config.MIN_OOF_FOR_FIT)
        return None
    if len(np.unique(y)) < 2:
        logger.warning("Calibration: single-class OOF labels — identity map")
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        z = np.log(p / (1.0 - p))
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(z.reshape(-1, 1), y.astype(int))
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Calibration: Platt fit failed (%s) — identity map", exc)
        return None
    # A pathological fit (slope <= 0 would invert the favorite's ranking)
    # falls back to identity: ranking preservation matters more than ECE
    # cosmetics (MLB parity).
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        logger.warning("Calibration: degenerate Platt params (a=%s) — identity map", a)
        return None
    # MLB presentation parity: persist the map at 6-decimal precision.
    cal = {"method": "platt", "a": round(a, 6), "b": round(b, 6), "n": int(n)}
    logger.debug("Calibration: Platt fitted on %d OOF games (a=%.4f, b=%.4f)", n, a, b)
    return cal


def apply_platt(p: np.ndarray, cal: dict | None) -> np.ndarray:
    """Apply the fitted map; identity when cal is None/invalid."""
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
    if not cal:
        return p.copy() if isinstance(p, np.ndarray) else p
    try:
        a = float(cal["a"])
        b = float(cal["b"])
    except (KeyError, TypeError, ValueError):
        return p.copy() if isinstance(p, np.ndarray) else p
    z = np.log(p / (1.0 - p))
    return np.clip(1.0 / (1.0 + np.exp(-(a * z + b))), CLIP, 1.0 - CLIP)


def _favored_view(p_home: np.ndarray, home_win: np.ndarray):
    """Convert home-space predictions/outcomes into favored-team space."""
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(home_win, dtype=float)
    favored_home = p >= 0.5
    p_fav = np.where(favored_home, p, 1.0 - p)
    y_fav = np.where(favored_home, y, 1.0 - y)
    return p_fav, y_fav, favored_home


def moneyline_fit(p_home: np.ndarray, home_win: np.ndarray) -> dict | None:
    """Favored-space moneyline calibrator fit, gated by CALIBRATION_MODE."""
    if get_calibration_mode() == "identity":
        logger.info("Calibration: CALIBRATION_MODE=identity — moneyline publishes the raw blend (no Platt map)")
        return None
    p_fav, y_fav, _ = _favored_view(p_home, home_win)
    cal = fit_platt(p_fav, y_fav)
    if cal is None:
        return None
    cal["method"] = FAVORED_CALIBRATOR_METHOD
    cal["floor"] = FAVORED_PROBABILITY_FLOOR
    return cal


def moneyline_apply(p_home: np.ndarray, calibrator: dict | None) -> np.ndarray:
    """Apply the favored-space calibrator, gated by CALIBRATION_MODE.

    Identity mode returns p unchanged (calibrated == raw). A calibrator not
    tagged ``favored_platt_floor`` (e.g. a legacy home-space map) is rejected
    rather than silently applied (MLB parity).
    """
    p = np.clip(np.asarray(p_home, dtype=float), 0.0, 1.0)
    if get_calibration_mode() == "identity":
        return p
    if not calibrator:
        return p
    if calibrator.get("method") != FAVORED_CALIBRATOR_METHOD:
        raise ValueError(
            "legacy home-space moneyline calibration is unsupported; "
            "retrain to create a favored-space calibrator")
    return apply_favored_platt(p, calibrator)


def fit_favored_platt(p_home: np.ndarray, home_win: np.ndarray) -> dict | None:
    """Fit Platt in the same favored-team space shown to users."""
    return moneyline_fit(p_home, home_win)


def apply_favored_platt(p_home: np.ndarray, cal: dict | None) -> np.ndarray:
    """Calibrate favored probability, floor it at 50%, convert home space back.

    The favorite's calibrated probability never drops below 0.5 and the
    underdog mirrors 1 - p_fav_cal. Identity (None) returns the input.
    """
    p = np.asarray(p_home, dtype=float)
    if not cal:
        return np.clip(p, 0.0, 1.0)
    favored_home = p >= 0.5
    p_fav = np.where(favored_home, p, 1.0 - p)
    p_fav_cal = np.maximum(FAVORED_PROBABILITY_FLOOR, apply_platt(p_fav, cal))
    return np.where(favored_home, p_fav_cal, 1.0 - p_fav_cal)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean binary log loss on finite rows (same clipping as apply)."""
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def prequential_fold_calibrators(
        p_home: np.ndarray, home_win: np.ndarray,
        fold_ids: np.ndarray) -> tuple[dict[int, dict | None], dict]:
    """Per-fold favored-space calibrators, GATED by a nested prequential
    holdout (2026-10-01 remediation).

    Fold k's map is still fitted strictly on folds < k (the prequential
    contract is unchanged). What changes is the decision to USE that map:
    the candidate is fitted on the EARLIER part of the prior evidence and
    scored against the raw blend on the MOST RECENT slice of it. The OOF
    frame is walk-forward ordered, so the tail slice of the prior indices
    is the freshest evidence that exists before fold k — the only
    out-of-sample test available that cannot read the window it scores.
    When the candidate cannot beat the raw blend there (by more than
    CAL_GATE_EPS nats), the fold is served uncalibrated: a 2-parameter
    Platt on rolling near-calibrated evidence adds variance, not signal
    (the 2026-10-01 run's per-fold layer moved logloss 0.67583->0.67731
    and ECE 0.01036->0.01466 while the pooled calibrator sat at
    a=1.0124 b=-0.0069).

    Returns (calibrators {fold_id: map or None}, audit). The audit
    carries one record per fold (n_prior, decision, reason, both
    nested-holdout log losses when a candidate was fitted) plus counts.
    """
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(home_win, dtype=float)
    folds = np.asarray(fold_ids)
    ok = np.isfinite(p) & np.isfinite(y) & (p > 0.0) & (p < 1.0)

    calibrators: dict[int, dict | None] = {}
    fold_records: list[dict] = []
    for fold in np.unique(folds):
        fid = int(fold)
        prior_idx = np.flatnonzero(ok & (folds < fold))
        rec: dict[str, Any] = {
            "fold_id": fid, "n_prior": int(len(prior_idx)),
            "decision": "identity", "reason": "no_prior_evidence",
            "nested_holdout_logloss": None, "raw_logloss": None,
            "params": None,
        }
        if len(prior_idx) >= 2:
            # Temporal split of the strictly-prior evidence.
            n_prior = len(prior_idx)
            hold = max(config.CAL_GATE_MIN_HOLDOUT,
                       int(round(config.CAL_GATE_HOLDOUT_FRAC * n_prior)))
            fit_idx = prior_idx[:n_prior - hold] if n_prior > hold else []
            hold_idx = prior_idx[-hold:] if hold else prior_idx
            if (not len(fit_idx)
                    or len(fit_idx) < config.MIN_OOF_FOR_FIT
                    or len(hold_idx) < 2):
                rec["reason"] = "insufficient_prior_evidence"
            else:
                cand = moneyline_fit(p[fit_idx], y[fit_idx])
                if cand is None:
                    rec["reason"] = "candidate_declined"
                else:
                    ll_cal = _logloss(
                        apply_favored_platt(p[hold_idx], cand), y[hold_idx])
                    ll_raw = _logloss(p[hold_idx], y[hold_idx])
                    rec["nested_holdout_logloss"] = round(ll_cal, 6)
                    rec["raw_logloss"] = round(ll_raw, 6)
                    if ll_cal < ll_raw - config.CAL_GATE_EPS:
                        final = moneyline_fit(p[prior_idx], y[prior_idx])
                        if final is not None:
                            calibrators[fid] = final
                            rec.update(decision="fitted", reason="accepted",
                                       params={"a": final["a"],
                                               "b": final["b"]})
                        else:
                            rec["reason"] = "final_fit_declined"
                    else:
                        rec["reason"] = "gated_no_gain"
        fold_records.append(rec)
        calibrators.setdefault(fid, None)

    audit: dict[str, Any] = {
        "method": "prequential_platt_gated",
        "n_folds": len(fold_records),
        "n_fitted": sum(1 for r in fold_records if r["decision"] == "fitted"),
        "n_identity": sum(1 for r in fold_records if r["decision"] == "identity"),
        "n_gated": sum(1 for r in fold_records
                        if r["reason"] == "gated_no_gain"),
        "n_insufficient": sum(1 for r in fold_records
                               if r["reason"] == "insufficient_prior_evidence"),
        "n_declined": sum(1 for r in fold_records
                           if r["reason"] in ("candidate_declined",
                                               "final_fit_declined")),
        "holdout_frac": config.CAL_GATE_HOLDOUT_FRAC,
        "min_holdout": config.CAL_GATE_MIN_HOLDOUT,
        "eps": config.CAL_GATE_EPS,
        "folds": fold_records,
    }
    return calibrators, audit
