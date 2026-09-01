"""Run-engine challenger ablation — fix the base (Phase A), then establish
the best underlying model (Phase B).

Background (the λ-edge probe): the run engine retains only ~60% of true
margin spread (actual_margin ≈ 0.014 + 1.66·λ_edge), run-line −1.5 cover is
under-priced at the extremes (top-5% p_win: pred 0.614 vs actual 0.681;
top decile cover: pred 0.421 vs actual 0.483), totals (λ_home + λ_away) are
clean, and the compression is entirely in the run engine (the binary ML is
well-calibrated in the same bins). The current engine is two independent
per-side LGBM λ models (heavy regularization: RUN_LGBM_PARAMS with
num_leaves=8, min_child_samples=40, min_gain_to_split=0.5) + NB Monte Carlo.

PHASE A — fix the base (4 arms, same folds/seed as the run-engine
walk-forward, sealed 284-game holdout, fit-on-OOF / evaluate-on-sealed):
  C0  — current independent NB models (production RUN_LGBM_PARAMS).
  C1  — current architecture, RELAXED per-side params (num_leaves up,
        min_child_samples down, min_gain_to_split down) — is the
        compression purely regularization?
  C2  — C0 + symmetric LINEAR edge fix: λ'_H = μ + k(λ_H − μ),
        λ'_A = μ + k(λ_A − μ) with μ the per-game midpoint
        (λ_H + λ_A)/2 — the LEVEL (sum) is preserved exactly and the EDGE
        is scaled by k, fit on the PRE-sealed OOF only.
  C3  — C0 + superlinear edge correction, fit on pre-sealed OOF, evaluated
        on sealed. Preferred: ISOTONIC λ_edge → actual margin (data-driven,
        no hand-chosen breakpoints); the power form
        edge' = sign(edge)·|edge|^p (p>1, one parameter) is fitted and
        recorded as the alternative. No piecewise breakpoints are
        hand-selected from the observed bins (n≈53/n≈16 at the extremes).

PHASE B — underlying model, built on the Phase-A WINNER's base (never on
the compressed C0 unless C0 won):
  LEARNER_SWEEP — same per-side λ targets, same features, same folds:
        LGBM / XGB / HistGradientBoosting / Ridge / small MLP, each
        reasonably tuned on OOF only, priced through the SAME NB-MC.
  DIST_FORM     — the marginalization-step challengers: NB-MC (winner) vs
        a distributional NN outputting (λ_h, α_h, λ_a, α_a) jointly
        (numpy MLP, NB-NLL loss) vs DIRECT-margin regression + parametric
        residual (normal / Laplace), each priced into the same surfaces.
  ENSEMBLE      — weighted blend of the best λ-learners (adaptive weights
        by pooled OOF deviance, mirroring the binary side's structure),
        priced through the same MC.

SHARED GATE (every arm, pooled walk-forward AND sealed):
  MARGIN  — CRPS on the full margin distribution (PRIMARY); −1.5 cover
            calibration in the >0.65/>0.70 bins and across deciles; derived
            ML calibration + P(win) SD (target toward the binary model's
            ~0.066).
  TOTALS  — O/U calibration + CRPS on the sum distribution; must stay
            flat/within noise of C0 (the sum must not move).
  Per-line calibration tables (predicted vs actual by bin) for every arm.
Discipline: ALL parameters (k, p, isotonic map, residual scales, learner
hyperparams) fit on PRE-sealed OOF only; the sealed 284 games never touch
fitting or shape selection.

Usage (from mlb-backend/backend):
    python run_engine_challenger_ablation.py --arm C0        # one arm
    python run_engine_challenger_ablation.py --arm C1 --arm C2 --arm C3
    python run_engine_challenger_ablation.py --all           # all arms
    python run_engine_challenger_ablation.py --record        # write JSONs
Per-arm λ frames are cached under --cache-dir (default /tmp/challenger) so
arms can be computed in separate time-boxed invocations and the record can
be assembled later. Records: data_delivery/run_engine_challenger_<date>.json
(harness-only; NO production change until a winner is chosen).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import (ALPHA_FLOOR, MARGIN_PLUS1_HOME_SHARE,
                        RUN_LGBM_PARAMS, _rounded_total_line, alpha_of,
                        derive_run_features, nb_pmf_matrix,
                        select_alpha_curve)
from margin_reliability_diagnostic import _nb_score_pmf, _crps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

SEALED_N = 284                 # chronological last-N OOF games (edge-gate convention)
MARGIN_GRID = list(range(-60, 61))
TOTAL_GRID = list(range(0, 26))
COVER_BINS = (0.65, 0.70)      # the probe's extreme bins
COVER_DECILES = 10
TOTALS_TOL = 0.02              # totals CRPS must stay within this of C0 (sealed)
CRPS_IMPROVE = 0.0             # sealed margin CRPS must be strictly lower
MAX_ROUNDS = 1000
EARLY_STOP = 20

# Production vs RELAXED per-side params (Phase A C1).
RELAXED_LGBM_PARAMS = {
    **RUN_LGBM_PARAMS,
    "num_leaves": 31,           # 8 -> 31
    "min_child_samples": 10,    # 40 -> 10
    "min_gain_to_split": 0.05,  # 0.5 -> 0.05
}

SIDES = (("home", "home_score"), ("away", "away_score"))


# ---------------------------------------------------------------------------
# Frame + folds (same geometry as the run engine's walk-forward)
# ---------------------------------------------------------------------------
def load_decided_frame() -> pd.DataFrame:
    from data_ingestion import load_game_features
    from frames import get_decided_frame
    games = load_game_features(DATA / "game_level_features.csv")
    games["game_date"] = pd.to_datetime(games["game_date"])
    return get_decided_frame(games)


def build_folds(decided: pd.DataFrame) -> list[dict]:
    from config import MIN_VAL_FOLD_GAMES, RETRAIN_CADENCE_DAYS
    import training
    splits = training.walk_forward_splits(
        decided, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    return [s for s in splits if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]


def sealed_masks(oof: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(pre_mask, hold_mask) — chronological last SEALED_N games are sealed."""
    order = pd.to_datetime(oof["game_date"]).argsort(kind="stable")
    hold = np.zeros(len(oof), bool)
    hold[order[-SEALED_N:]] = True
    return ~hold, hold


# ---------------------------------------------------------------------------
# Per-side λ learner loop (mirrors run_oof / build_oof_margin geometry)
# ---------------------------------------------------------------------------
def run_learner_oof(decided: pd.DataFrame, folds: list[dict],
                    learner: str, params: dict | None = None,
                    cache: Path | None = None,
                    force: bool = False) -> pd.DataFrame:
    """Walk-forward OOF λs for one learner (per-side models, same folds as
    the run engine). Cached under ``cache`` (parquet) when given.

    Returns the OOF frame: game_pk, game_date, fold_idx, home_expected_runs,
    away_expected_runs, home_score, away_score (the run_engine contract).
    """
    if cache is not None and cache.exists() and not force:
        return pd.read_parquet(cache)
    params = dict(params or {})
    run_feats, _ = derive_run_features(
        list(__import__("training", fromlist=["FEATURE_COLS"]).FEATURE_COLS))
    frames = {side: _side_frame(decided, side, run_feats)
              for side, _t in SIDES}
    # sklearn learners (ridge/mlp) cannot take NaN — median-impute from the
    # TRAIN fold only (no leakage); LGBM/XGB/HGB route NaN natively.
    impute = learner in ("ridge", "mlp")
    out_rows: list[dict] = []
    best_iters: dict[str, list[int]] = {s: [] for s, _t in SIDES}
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        rec = {"game_pk": va["game_pk"].to_numpy(),
               "game_date": pd.to_datetime(
                   va["game_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
               "fold_idx": int(split["fold_idx"])}
        for side, target in SIDES:
            cols_all = frames[side][1]
            tr_f = tr.reindex(columns=cols_all).astype(float)
            va_f = va.reindex(columns=cols_all).astype(float)
            if impute:
                med = tr_f.median().fillna(0.0)
                tr_f = tr_f.fillna(med)
                va_f = va_f.fillna(med)
            y_tr = tr[target].to_numpy(float)
            y_va = va[target].to_numpy(float)
            lam, best = _fit_side(learner, params, tr_f, y_tr, va_f, y_va)
            best_iters[side].append(best)
            rec[f"{side}_expected_runs"] = np.round(lam, 4)
            rec[target] = y_va.astype(int)
        out_rows.append(pd.DataFrame(rec))
    oof = pd.concat(out_rows, ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(cache)
    return oof


def _side_frame(games: pd.DataFrame, side: str, run_feats: list[str]):
    from run_engine import build_side_frame
    return build_side_frame(games, side, run_features=run_feats,
                            dropped=[])


def _fit_side(learner: str, params: dict, tr_f: pd.DataFrame, y_tr: np.ndarray,
              va_f: pd.DataFrame, y_va: np.ndarray) -> tuple[np.ndarray, int]:
    """Fit one side's λ model, predict the val fold. Returns (λ, best_iter)."""
    if learner == "lgbm":
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        m = LGBMRegressor(**params)
        m.set_params(n_estimators=MAX_ROUNDS)
        m.fit(tr_f, y_tr, eval_set=[(va_f, y_va)],
              callbacks=[early_stopping(EARLY_STOP, verbose=False),
                         log_evaluation(period=0)])
        best = int(m.best_iteration_ or MAX_ROUNDS)
        return np.clip(m.predict(va_f, num_iteration=best), 1e-6, None), best
    if learner == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor(objective="count:poisson", max_depth=3,
                         learning_rate=0.05, n_estimators=MAX_ROUNDS,
                         min_child_weight=4, subsample=0.8,
                         colsample_bytree=0.8, random_state=42,
                         eval_metric="poisson-nloglik", early_stopping_rounds=EARLY_STOP)
        m.fit(tr_f, y_tr, eval_set=[(va_f, y_va)], verbose=False)
        bi = getattr(m, "best_iteration", None)
        best = int(bi) if bi is not None else MAX_ROUNDS
        # best_iteration is the 0-based index of the best round and can be 0
        # (first fold early-stops immediately); the model trained best+1
        # rounds, so iteration_range end = best + 1.
        return np.clip(m.predict(va_f, iteration_range=(0, best + 1)),
                       1e-6, None), best
    if learner == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(loss="poisson", max_iter=400,
                                          learning_rate=0.05, max_leaf_nodes=15,
                                          early_stopping=True, n_iter_no_change=20,
                                          validation_fraction=0.15,
                                          random_state=42)
        m.fit(tr_f, y_tr)
        return np.clip(m.predict(va_f), 1e-6, None), int(m.n_iter_)
    if learner == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0)
        m.fit(tr_f, np.log1p(y_tr))
        return np.clip(np.expm1(m.predict(va_f)), 1e-6, None), 1
    if learner == "mlp":
        # Poisson-deviance MLP with a RELU output on the RAW count target
        # (log1p-target + exp back-transform explodes on small early folds).
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(24, 12), alpha=0.01,
                         early_stopping=True, validation_fraction=0.15,
                         max_iter=600, learning_rate_init=0.001,
                         activation="relu", random_state=42)
        m.fit(tr_f, y_tr)
        lam = np.maximum(m.predict(va_f), 1e-6)
        best = int(getattr(m, "n_iter_", 0) or 1)
        return np.clip(lam, 1e-6, None), best
    raise ValueError(f"unknown learner {learner}")


# ---------------------------------------------------------------------------
# Phase A — edge corrections (C2 linear, C3 isotonic/power). All fit on the
# PRE-sealed window only; sum (λ_H + λ_A) is preserved exactly by re-centering
# on the per-game midpoint μ = (λ_H + λ_A)/2.
# ---------------------------------------------------------------------------
def _midpoint(lam_h: np.ndarray, lam_a: np.ndarray) -> np.ndarray:
    return (np.asarray(lam_h) + np.asarray(lam_a)) / 2.0


def fit_linear_k(pre_lam_h: np.ndarray, pre_lam_a: np.ndarray,
                 pre_margin: np.ndarray) -> float:
    """k = OLS slope of actual margin on the raw λ edge (pre-sealed only)."""
    d = np.asarray(pre_lam_h) - np.asarray(pre_lam_a)
    return float(np.polyfit(d, pre_margin, 1)[0])


def apply_linear_edge(lam_h: np.ndarray, lam_a: np.ndarray, k: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """λ' = μ + k(λ − μ) per side, μ the game midpoint — sum preserved."""
    mu = _midpoint(lam_h, lam_a)
    return mu + k * (np.asarray(lam_h) - mu), mu + k * (np.asarray(lam_a) - mu)


def fit_power_p(pre_lam_h: np.ndarray, pre_lam_a: np.ndarray,
                pre_margin: np.ndarray) -> float:
    """p = argmin_p Σ (margin − sign(d)|d|^p)² over p in (1, 3] (pre-sealed)."""
    from scipy.optimize import minimize_scalar
    d = np.asarray(pre_lam_h) - np.asarray(pre_lam_a)
    y = np.asarray(pre_margin)

    def loss(p: float) -> float:
        pred = np.sign(d) * np.abs(d) ** p
        return float(np.mean((y - pred) ** 2))

    res = minimize_scalar(loss, bounds=(1.0, 3.0), method="bounded")
    return float(res.x)


def fit_isotonic(pre_lam_h: np.ndarray, pre_lam_a: np.ndarray,
                 pre_margin: np.ndarray):
    """Isotonic λ_edge → actual margin (increasing), pre-sealed only."""
    from sklearn.isotonic import IsotonicRegression
    d = np.asarray(pre_lam_h) - np.asarray(pre_lam_a)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(d, pre_margin)
    return iso


def apply_isotonic_edge(lam_h: np.ndarray, lam_a: np.ndarray, iso,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Predict the corrected margin m̂(λ_edge) and re-split around μ so the
    sum is preserved: λ'_H = μ + m̂/2, λ'_A = μ − m̂/2."""
    d = np.asarray(lam_h) - np.asarray(lam_a)
    m_hat = np.asarray(iso.predict(d))
    mu = _midpoint(lam_h, lam_a)
    return mu + m_hat / 2.0, mu - m_hat / 2.0


# ---------------------------------------------------------------------------
# Phase B — distributional NN (joint λ, α) and direct-margin regressions
# ---------------------------------------------------------------------------
def run_dist_nn_oof(decided: pd.DataFrame, folds: list[dict],
                    cache: Path | None = None,
                    force: bool = False) -> pd.DataFrame:
    """Per-fold numpy MLP outputting (λ_h, α_h, λ_a, α_a) jointly under an
    NB-NLL loss. Cached parquet; returns the OOF frame + per-game α columns.
    """
    if cache is not None and cache.exists() and not force:
        return pd.read_parquet(cache)
    run_feats, _ = derive_run_features(
        list(__import__("training", fromlist=["FEATURE_COLS"]).FEATURE_COLS))
    frames = {side: _side_frame(decided, side, run_feats)
              for side, _t in SIDES}
    out_rows: list[dict] = []
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        Xtr, Xva, m_tr, s_tr = _nn_prep(frames, tr, va)
        ytr = np.column_stack([tr["home_score"].to_numpy(float),
                               tr["away_score"].to_numpy(float)])
        yva = np.column_stack([va["home_score"].to_numpy(float),
                               va["away_score"].to_numpy(float)])
        pred = _fit_dist_nn(Xtr, ytr, Xva, yva)
        rec = {"game_pk": va["game_pk"].to_numpy(),
               "game_date": pd.to_datetime(
                   va["game_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
               "fold_idx": int(split["fold_idx"])}
        lam_h = np.maximum(pred[:, 0], 1e-6)
        lam_a = np.maximum(pred[:, 2], 1e-6)
        rec["home_expected_runs"] = np.round(lam_h, 4)
        rec["away_expected_runs"] = np.round(lam_a, 4)
        rec["alpha_home"] = np.round(pred[:, 1], 4)
        rec["alpha_away"] = np.round(pred[:, 3], 4)
        rec["home_score"] = va["home_score"].to_numpy(int)
        rec["away_score"] = va["away_score"].to_numpy(int)
        out_rows.append(pd.DataFrame(rec))
    oof = pd.concat(out_rows, ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(cache)
    return oof


def _nn_prep(frames: dict, tr: pd.DataFrame, va: pd.DataFrame
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = frames["home"][1] + [c for c in frames["away"][1]
                                if c not in frames["home"][1]]
    tr_f = tr.reindex(columns=cols).astype(float)
    va_f = va.reindex(columns=cols).astype(float)
    med = tr_f.median().fillna(0.0)
    tr_i = tr_f.fillna(med)
    va_i = va_f.fillna(med)
    mu, sd = tr_i.mean().to_numpy(float), tr_i.std().to_numpy(float)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((tr_i.to_numpy(float) - mu) / sd,
            (va_i.to_numpy(float) - mu) / sd, mu, sd)


def _nb_logpmf(k: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vectorized NB log-pmf (mean μ, dispersion α), α floored."""
    from scipy.special import gammaln
    a = np.maximum(alpha, ALPHA_FLOOR)
    n = 1.0 / a
    p = n / (n + mu)
    return (gammaln(k + n) - gammaln(n) - gammaln(k + 1.0)
            + n * np.log(p) + k * np.log1p(-p))


def _fit_dist_nn(Xtr: np.ndarray, ytr: np.ndarray,
                 Xva: np.ndarray, yva: np.ndarray) -> np.ndarray:
    """Small numpy MLP (hidden 24, tanh) with NB-NLL; early-stops on the val
    fold's NLL. Outputs (λ_h, α_h, λ_a, α_a) via softplus. Returns val preds."""
    rng = np.random.default_rng(42)
    d_in, h = Xtr.shape[1], 24
    W1 = rng.normal(0, 0.1, (d_in, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0, 0.1, (h, 4))
    b2 = np.zeros(4)
    lr = 0.02
    best_ll, best = -np.inf, None
    for epoch in range(60):
        z1 = Xtr @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        out = np.log1p(np.exp(z2))
        lam_h = np.maximum(out[:, 0], 1e-6)
        al_h = np.maximum(out[:, 1], ALPHA_FLOOR)
        lam_a = np.maximum(out[:, 2], 1e-6)
        al_a = np.maximum(out[:, 3], ALPHA_FLOOR)
        nll = -(_nb_logpmf(ytr[:, 0], lam_h, al_h)
                + _nb_logpmf(ytr[:, 1], lam_a, al_a)).mean()
        # ---- backprop through the NB NLL (manual, scipy-free) ----
        g_mu_h, g_al_h = _nb_grad(ytr[:, 0], lam_h, al_h)
        g_mu_a, g_al_a = _nb_grad(ytr[:, 1], lam_a, al_a)
        d_z2 = np.column_stack([
            g_mu_h * (lam_h / (1 + np.exp(z2[:, 0]))),
            g_al_h * (al_h / (1 + np.exp(z2[:, 1]))),
            g_mu_a * (lam_a / (1 + np.exp(z2[:, 2]))),
            g_al_a * (al_a / (1 + np.exp(z2[:, 3]))),
        ]) / len(ytr)
        dW2 = a1.T @ d_z2
        db2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ W2.T
        d_z1 = d_a1 * (1 - a1 ** 2)
        dW1 = Xtr.T @ d_z1
        db1 = d_z1.sum(axis=0)
        for W, b, dW, db in ((W1, b1, dW1, db1), (W2, b2, dW2, db2)):
            W -= lr * dW
            b -= lr * db
        # val NLL for early stopping
        z1v = Xva @ W1 + b1
        a1v = np.tanh(z1v)
        outv = np.log1p(np.exp(a1v @ W2 + b2))
        lam_hv = np.maximum(outv[:, 0], 1e-6)
        al_hv = np.maximum(outv[:, 1], ALPHA_FLOOR)
        lam_av = np.maximum(outv[:, 2], 1e-6)
        al_av = np.maximum(outv[:, 3], ALPHA_FLOOR)
        vll = (_nb_logpmf(yva[:, 0], lam_hv, al_hv)
               + _nb_logpmf(yva[:, 1], lam_av, al_av)).mean()
        if vll > best_ll:
            best_ll, best = vll, (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    W1, b1, W2, b2 = best
    z1v = Xva @ W1 + b1
    outv = np.log1p(np.exp(np.tanh(z1v) @ W2 + b2))
    return outv


def _nb_grad(k: np.ndarray, mu: np.ndarray, alpha: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """d NLL/d λ and d NLL/d α for one side (numerically verified closed
    form; n = 1/α, p = n/(n+μ)):

        dLL/dμ   = k/μ − (n+k)/(n+μ)
        dLL/dα   = (ψ(k+n) − ψ(n))·(−1/α²) + log(p)·(−1/α²)
                 + n·(dp/dn·dn/dα)/p + k·(−dp/dn·dn/dα)/(1−p)
    """
    from scipy.special import digamma
    a = np.maximum(alpha, ALPHA_FLOOR)
    n = 1.0 / a
    dn_da = -1.0 / (a * a)
    p = n / (n + mu)
    dp_dn = mu / (n + mu) ** 2
    dp_da = dp_dn * dn_da
    d_log_dmu = k / mu - (n + k) / (n + mu)
    d_log_dalpha = (digamma(k + n) - digamma(n)) * dn_da \
        + np.log(p) * dn_da + n * (dp_da / p) + k * (-dp_da / (1.0 - p))
    return -d_log_dmu, -d_log_dalpha


def run_direct_margin_oof(decided: pd.DataFrame, folds: list[dict],
                          residual: str, cache: Path | None = None,
                          force: bool = False) -> pd.DataFrame:
    """Direct-margin regression + parametric residual (normal/laplace).

    Per fold: Ridge on the per-game (home-side − away-side) diff view +
    environment to predict (margin, total). The residual scale is fit on the
    PRE-sealed pooled OOF residuals only and applied to every game.
    Returns the OOF frame with margin_hat, total_hat + the residual params.
    """
    if cache is not None and cache.exists() and not force:
        return pd.read_parquet(cache)
    run_feats, _ = derive_run_features(
        list(__import__("training", fromlist=["FEATURE_COLS"]).FEATURE_COLS))
    frames = {side: _side_frame(decided, side, run_feats)
              for side, _t in SIDES}
    h_cols, a_cols = frames["home"][1], frames["away"][1]
    # Home/away side columns are symmetric pairs (*_home / *_away or
    # home_* / away_*); the shared environment columns appear identically on
    # both sides. Diff view: side pairs subtract the opponent's value;
    # environment columns stay as-is (shared, not doubled).
    env_h = [c for c in h_cols if not (c.endswith("_home")
                                       or c.startswith("home_"))]
    env_a = [c for c in a_cols if not (c.endswith("_away")
                                       or c.startswith("away_"))]
    pair_h = [c for c in h_cols if c.endswith("_home") or c.startswith("home_")]

    def _diff_view(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for hc in pair_h:
            ac = hc.replace("_home", "_away").replace("home_", "away_")
            if ac in df.columns and hc in df.columns:
                out[f"d_{hc}"] = df[hc].astype(float) - df[ac].astype(float)
        for c in env_h + env_a:
            if c in df.columns and c not in out.columns:
                out[c] = df[c].astype(float)
        return out

    out_rows: list[dict] = []
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        Xtr, Xva = _diff_view(tr), _diff_view(va)
        med = Xtr.median().fillna(0.0)
        Xtr, Xva = Xtr.fillna(med), Xva.fillna(med)
        ytr_m = (tr["home_score"].to_numpy(float)
                 - tr["away_score"].to_numpy(float))
        ytr_t = (tr["home_score"].to_numpy(float)
                 + tr["away_score"].to_numpy(float))
        from sklearn.linear_model import Ridge
        mm = Ridge(alpha=1.0).fit(Xtr, ytr_m)
        mt = Ridge(alpha=1.0).fit(Xtr, ytr_t)
        rec = {"game_pk": va["game_pk"].to_numpy(),
               "game_date": pd.to_datetime(
                   va["game_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
               "fold_idx": int(split["fold_idx"]),
               "margin_hat": mm.predict(Xva),
               "total_hat": mt.predict(Xva),
               "home_score": va["home_score"].to_numpy(int),
               "away_score": va["away_score"].to_numpy(int)}
        out_rows.append(pd.DataFrame(rec))
    oof = pd.concat(out_rows, ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(cache)
    return oof


# ---------------------------------------------------------------------------
# Shared gate — pricing + scoring
# ---------------------------------------------------------------------------
def price_arm(oof: pd.DataFrame, lam_h: np.ndarray, lam_a: np.ndarray,
              alpha_h: np.ndarray, alpha_a: np.ndarray,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Margin + totals PMFs for every game (exact NB convolution + tie-fix)."""
    pmf_m = _nb_score_pmf(lam_h, lam_a, alpha_h, alpha_a, MARGIN_GRID)
    # Totals PMF: convolution of the two NB marginals over TOTAL_GRID,
    # vectorized via the (n, T, T) outer product + anti-diagonal sums
    # (covers TOTAL_GRID exactly; the full-basis tail is renormalized).
    ks = np.arange(max(TOTAL_GRID) + 1)
    ph = nb_pmf_matrix(ks, np.maximum(lam_h, 1e-6)[:, None],
                       np.maximum(alpha_h, ALPHA_FLOOR)[:, None])[:, :len(TOTAL_GRID)]
    pa = nb_pmf_matrix(ks, np.maximum(lam_a, 1e-6)[:, None],
                       np.maximum(alpha_a, ALPHA_FLOOR)[:, None])[:, :len(TOTAL_GRID)]
    P = ph[:, :, None] * pa[:, None, :]
    idx = np.arange(len(TOTAL_GRID))
    pmf_t = np.stack(
        [P[:, idx[:t + 1], t - idx[:t + 1]].sum(axis=1)
         for t in range(len(TOTAL_GRID))], axis=1)
    pmf_t /= pmf_t.sum(axis=1, keepdims=True)
    return pmf_m, pmf_t


def cover_probs(pmf_m: np.ndarray) -> np.ndarray:
    return pmf_m[:, MARGIN_GRID.index(2):].sum(axis=1)


def win_probs(pmf_m: np.ndarray) -> np.ndarray:
    return pmf_m[:, MARGIN_GRID.index(1):].sum(axis=1)


def over_probs(pmf_t: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per-game assigned line (rounded total) and P(over) = P(total > line)."""
    lines = np.array([_rounded_total_line(lh, la)
                      for lh, la in zip(lam_h, lam_a)])
    ps = np.zeros(len(lam_h))
    for i, line in enumerate(lines):
        if line in TOTAL_GRID:
            j = TOTAL_GRID.index(line)
            ps[i] = pmf_t[i, j + 1:].sum() if j + 1 < len(TOTAL_GRID) else 0.0
        else:
            ps[i] = np.nan
    return lines, ps


def calibrate(p: np.ndarray, y: np.ndarray, bins: tuple = COVER_BINS,
              deciles: int = COVER_DECILES) -> dict:
    """Per-bin predicted-vs-actual table + decile table + overall delta."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    rows = []
    for lo, hi in zip([0.0] + list(np.linspace(0.1, 0.9, deciles - 1)),
                      list(np.linspace(0.1, 0.9, deciles - 1)) + [1.0]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                     "pred": round(float(p[m].mean()), 4),
                     "actual": round(float(y[m].mean()), 4),
                     "delta": round(float(p[m].mean() - y[m].mean()), 4)})
    ext = []
    for b in bins:
        m = p > b
        ext.append({"bin": f">{b:.2f}", "n": int(m.sum()),
                    "pred": round(float(p[m].mean()), 4) if m.sum() else None,
                    "actual": round(float(y[m].mean()), 4) if m.sum() else None,
                    "delta": (round(float(p[m].mean() - y[m].mean()), 4)
                              if m.sum() else None)})
    return {"deciles": rows, "extreme": ext,
            "overall_delta": round(float(p.mean() - y.mean()), 4)}


def score_arm(lam_h: np.ndarray, lam_a: np.ndarray,
              alpha_h: np.ndarray, alpha_a: np.ndarray,
              hs: np.ndarray, as_: np.ndarray, pre: np.ndarray, hold: np.ndarray,
              ) -> dict:
    """Full shared-gate score for one arm (pooled + sealed)."""
    margin = (hs - as_).astype(int)
    total = (hs + as_).astype(int)
    pmf_m, pmf_t = price_arm(None, lam_h, lam_a, alpha_h, alpha_a)
    pc, pw = cover_probs(pmf_m), win_probs(pmf_m)
    yc = (margin >= 2).astype(float)
    yw = (hs > as_).astype(float)
    lines, po = over_probs(pmf_t, lam_h, lam_a)
    yover = (total > lines).astype(float)

    def _crps_window(pmf: np.ndarray, grid: list, y: np.ndarray,
                     mask: np.ndarray) -> float:
        return round(float(_crps(pmf[mask], grid, y[mask])), 4)

    out = {
        "margin_crps_pooled": _crps_window(pmf_m, MARGIN_GRID, margin, pre),
        "margin_crps_sealed": _crps_window(pmf_m, MARGIN_GRID, margin, hold),
        "totals_crps_pooled": _crps_window(pmf_t, TOTAL_GRID, total, pre),
        "totals_crps_sealed": _crps_window(pmf_t, TOTAL_GRID, total, hold),
        "cover_cal_pooled": calibrate(pc[pre], yc[pre]),
        "cover_cal_sealed": calibrate(pc[hold], yc[hold]),
        "pwin_sd_pooled": round(float(pw[pre].std()), 4),
        "pwin_sd_sealed": round(float(pw[hold].std()), 4),
        "pwin_ece_pooled": round(float(_ece(yw[pre], pw[pre])), 4),
        "pwin_ece_sealed": round(float(_ece(yw[hold], pw[hold])), 4),
        "over_cal_pooled": _calibrate_lines(po[pre], yover[pre], lines[pre]),
        "over_cal_sealed": _calibrate_lines(po[hold], yover[hold], lines[hold]),
        "mean_lam_h": round(float(lam_h.mean()), 4),
        "mean_lam_a": round(float(lam_a.mean()), 4),
        "mean_lam_diff": round(float((lam_h - lam_a).mean()), 4),
    }
    return out


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    from run_engine import ece_score
    return ece_score(y, p, n_bins)


def _calibrate_lines(p: np.ndarray, y: np.ndarray, lines: np.ndarray) -> dict:
    """O/U calibration by assigned line (pred vs actual per line)."""
    df = pd.DataFrame({"p": p, "y": y, "line": lines})
    df = df[df["p"].notna() & df["line"].notna()]
    rows = []
    for line, g in df.groupby("line"):
        if len(g) < 10:
            continue
        rows.append({"line": float(line), "n": int(len(g)),
                     "pred": round(float(g["p"].mean()), 4),
                     "actual": round(float(g["y"].mean()), 4),
                     "delta": round(float(g["p"].mean() - g["y"].mean()), 4)})
    return {"by_line": rows,
            "overall_delta": (round(float(df["p"].mean() - df["y"].mean()), 4)
                              if len(df) else None)}


# ---------------------------------------------------------------------------
# Alpha curves (fit PRE-sealed only) + arm assembly
# ---------------------------------------------------------------------------
def fit_alphas_pre(oof: pd.DataFrame, pre: np.ndarray,
                   lam_h: np.ndarray, lam_a: np.ndarray,
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    curves = {}
    for side, tgt in (("home", "home_score"), ("away", "away_score")):
        lams = lam_h if side == "home" else lam_a
        curves[side], _ = select_alpha_curve(
            oof[tgt].to_numpy(float)[pre], lams[pre])
    return (alpha_of(lam_h, curves["home"]),
            alpha_of(lam_a, curves["away"]), curves)


def arm_lambdas(name: str, oof: pd.DataFrame, c1_oof: pd.DataFrame,
                pre: np.ndarray,
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-arm λ pair + fitted-param record. C2/C3 derive from C0's λs with
    the pre-sealed fit; everything else uses the OOF frame directly."""
    if name == "C1":
        oof = c1_oof
    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    margin = (oof["home_score"].to_numpy(float)
              - oof["away_score"].to_numpy(float))
    if name == "C0":
        return lam_h, lam_a, {"correction": "none"}
    if name == "C1":
        return lam_h, lam_a, {"correction": "none", "params": "relaxed"}
    if name == "C2":
        k = fit_linear_k(lam_h[pre], lam_a[pre], margin[pre])
        lh, la = apply_linear_edge(lam_h, lam_a, k)
        return lh, la, {"correction": "linear", "k": round(k, 4)}
    if name == "C3":
        iso = fit_isotonic(lam_h[pre], lam_a[pre], margin[pre])
        lh, la = apply_isotonic_edge(lam_h, lam_a, iso)
        p = fit_power_p(lam_h[pre], lam_a[pre], margin[pre])
        return lh, la, {"correction": "isotonic",
                        "power_alt_p": round(p, 4)}
    if name == "LEARNER_SWEEP":
        return lam_h, lam_a, {"correction": "none"}
    raise ValueError(name)


def linear_correction(oof: pd.DataFrame, pre: np.ndarray
                      ) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit + apply the C2 linear edge correction (λ' = μ + k(λ − μ)) on an
    OOF frame's own pre-sealed window. Returns (k, corrected_h, corrected_a).
    The correction is fit on THIS frame's scores — each Phase-B arm inherits
    the C2 structural base but with k re-fit to its own λs (same discipline)."""
    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    margin = (oof["home_score"].to_numpy(float)
              - oof["away_score"].to_numpy(float))
    k = fit_linear_k(lam_h[pre], lam_a[pre], margin[pre])
    lh, la = apply_linear_edge(lam_h, lam_a, k)
    return k, lh, la


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PHASE_A_ARMS = ("C0", "C1", "C2", "C3")
LEARNERS = ("lgbm", "xgb", "hgb", "ridge", "mlp")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", default=None,
                    choices=list(PHASE_A_ARMS) + ["DISTNN", "DIRECT_NORMAL",
                                                  "DIRECT_LAPLACE", "ENSEMBLE",
                                                  *LEARNERS],
                    help="compute/score specific arms (repeatable)")
    ap.add_argument("--all", action="store_true", help="run every arm")
    ap.add_argument("--record", action="store_true",
                    help="write the record JSON(s)")
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/challenger"),
                    help="per-arm λ cache directory")
    ap.add_argument("--phase-b-base", choices=("C0", "C2"), default="C2",
                    help="Phase-B base: the Phase-A winner's structural base "
                         "(C2 = linear edge layer; C0 = raw). Default C2.")
    ap.add_argument("--skip-phase-a", action="store_true",
                    help="skip (re)generating the C0/C1 OOF λ frames — use "
                         "the existing cached parquets (Phase-B-only scoring)")
    ap.add_argument("--force", action="store_true", help="ignore caches")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    decided = load_decided_frame()
    folds = build_folds(decided)
    base_oof = run_learner_oof(decided, folds, "lgbm",
                               params=dict(RUN_LGBM_PARAMS),
                               cache=args.cache_dir / "C0.parquet",
                               force=args.force and not args.skip_phase_a)
    pre, hold = sealed_masks(base_oof)
    hs = base_oof["home_score"].to_numpy(float)
    as_ = base_oof["away_score"].to_numpy(float)
    print(f"frame: {len(decided)} decided, {len(folds)} folds, "
          f"OOF {len(base_oof)} games (pre {int(pre.sum())} / sealed "
          f"{int(hold.sum())})")

    # C1 (relaxed params) gets its OWN OOF — the compression question is
    # exactly whether the params change the λ levels, so it must not reuse C0.
    c1_oof = run_learner_oof(decided, folds, "lgbm",
                             params=dict(RELAXED_LGBM_PARAMS),
                             cache=args.cache_dir / "C1.parquet",
                             force=args.force and not args.skip_phase_a)
    arms = {name: arm_lambdas(name, base_oof, c1_oof, pre)
            for name in PHASE_A_ARMS}
    # Phase B arms — built on the Phase-A winner's structural base.
    # Default base = C2 (the linear edge layer): every Phase-B arm is priced
    # through the SAME two-part shape (levels + edge) that won Phase A. With
    # --phase-b-base C0 the arms run raw (used only if C0 won Phase A).
    # NOTE: only the REQUESTED arms are computed; unrequested heavy arms
    # (e.g. the numpy-MLP DISTNN) are never touched.
    requested = list(args.arm or []) if not args.all else \
        [*PHASE_A_ARMS, *LEARNERS, "DISTNN", "DIRECT_NORMAL", "DIRECT_LAPLACE",
         "ENSEMBLE"]
    phase_b_any = any(a in requested
                      for a in (*LEARNERS, "DISTNN", "DIRECT_NORMAL",
                                "DIRECT_LAPLACE", "ENSEMBLE"))
    if phase_b_any:
        for ln in LEARNERS:
            # learner fits are always cached; skip ones not requested
            if ln not in requested and "ENSEMBLE" not in requested:
                continue
            oof = run_learner_oof(decided, folds, ln,
                                  params=(RELAXED_LGBM_PARAMS if ln == "lgbm"
                                          else None),
                                  cache=args.cache_dir / f"{ln}.parquet",
                                  force=args.force)
            if args.phase_b_base == "C2":
                k, lh, la = linear_correction(oof, pre)
                meta = {"correction": "linear", "k": round(k, 4),
                        "base": "C2"}
            else:
                lh, la = (oof["home_expected_runs"].to_numpy(float),
                          oof["away_expected_runs"].to_numpy(float))
                meta = {"correction": "none", "base": "C0"}
            arms[f"LEARN_{ln.upper()}"] = (lh, la, meta)
        if "DISTNN" in requested:
            nn = run_dist_nn_oof(decided, folds,
                                 cache=args.cache_dir / "DISTNN.parquet",
                                 force=args.force)
            if args.phase_b_base == "C2":
                k_nn, nn_h, nn_a = linear_correction(nn, pre)
                meta = {"correction": "linear", "k": round(k_nn, 4),
                        "alpha_from": "nn", "base": "C2"}
            else:
                nn_h = nn["home_expected_runs"].to_numpy(float)
                nn_a = nn["away_expected_runs"].to_numpy(float)
                meta = {"correction": "none", "alpha_from": "nn", "base": "C0"}
            arms["DISTNN"] = (nn_h, nn_a, meta)
        for res in ("normal", "laplace"):
            if f"DIRECT_{res.upper()}" not in requested:
                continue
            dm = run_direct_margin_oof(decided, folds, res,
                                       cache=args.cache_dir
                                       / f"DIRECT_{res.upper()}.parquet",
                                       force=args.force and not args.skip_phase_a)
            # DIRECT arms carry the (margin_hat, total_hat) frame; the
            # scoring loop routes them to _score_direct via the sentinel meta.
            arms[f"DIRECT_{res.upper()}"] = (dm, dm, {"direct": True})
        # ENSEMBLE: deviance-weighted λ blend of the learners, each corrected
        # through the Phase-B base layer (pre-sealed fit)
        if "ENSEMBLE" in requested:
            ens_h, ens_a, wts = _ensemble_lambdas(
                decided, folds, args.cache_dir, pre, base=args.phase_b_base,
                force=args.force and not args.skip_phase_a)
            arms["ENSEMBLE"] = (ens_h, ens_a, {"correction": "linear"
                                                if args.phase_b_base == "C2"
                                                else "none",
                                               "weights": wts,
                                               "base": args.phase_b_base})

    # ---- score every available arm on the shared gate ----
    results = {}
    for name, (lh, la, meta) in arms.items():
        oof_src = base_oof
        if name.startswith("LEARN_"):
            oof_src = pd.read_parquet(
                args.cache_dir / f"{name[len('LEARN_'):].lower()}.parquet")
        elif name == "DISTNN":
            oof_src = nn
        elif name.startswith("DIRECT_"):
            oof_src = dm if name == "DIRECT_LAPLACE" else \
                pd.read_parquet(args.cache_dir / f"{name}.parquet")
        if name.startswith("DIRECT_"):
            results[name] = _score_direct(oof_src, name)
            continue
        alpha_h, alpha_a, curves = fit_alphas_pre(
            oof_src, pre, lh, la) if name != "DISTNN" else \
            (np.maximum(nn["alpha_home"].to_numpy(float), ALPHA_FLOOR),
             np.maximum(nn["alpha_away"].to_numpy(float), ALPHA_FLOOR), {})
        results[name] = score_arm(lh, la, alpha_h, alpha_a, hs, as_, pre, hold)
        results[name]["meta"] = meta

    # ---- Phase A gate + leader ----
    if all(a in results for a in PHASE_A_ARMS):
        phase_a = _phase_a_verdict(results)
        print("\n=== PHASE A ===")
        for n in PHASE_A_ARMS:
            r = results[n]
            print(f"  {n}: sealed CRPS {r['margin_crps_sealed']} "
                  f"(pooled {r['margin_crps_pooled']}) | totals sealed "
                  f"{r['totals_crps_sealed']} | P(win) SD {r['pwin_sd_sealed']} "
                  f"| cover>0.70 {r['cover_cal_sealed']['extreme'][1]}")
        print(f"  PHASE-A WINNER: {phase_a['winner']} — {phase_a['reason']}")

    # ---- Phase B summary ----
    phase_b_names = sorted([n for n in results if n not in PHASE_A_ARMS])
    if phase_b_names:
        print("\n=== PHASE B (all on the Phase-A winner's base) ===")
        for n in phase_b_names:
            r = results[n]
            extra = ""
            if "k" in r.get("meta", {}):
                extra = f" k={r['meta']['k']}"
            if "weights" in r.get("meta", {}):
                extra = f" weights={r['meta']['weights']}"
            print(f"  {n}: sealed CRPS {r['margin_crps_sealed']} "
                  f"(pooled {r['margin_crps_pooled']}) | totals sealed "
                  f"{r['totals_crps_sealed']} | P(win) SD {r['pwin_sd_sealed']}{extra}")

    if args.record:
        rec = {
            "schema": "run-engine-challenger/v1",
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "frame": {"decided": int(len(decided)), "folds": len(folds),
                      "oof_games": int(len(base_oof)),
                      "sealed_n": SEALED_N},
            "probe_reproduced": {
                "note": "actual_margin ≈ 0.014 + 1.66·λ_edge (60% retention)",
                "pwin_sd_current": results.get("C0", {}).get("pwin_sd_sealed"),
                "pwin_sd_target": 0.066,
            },
            "arms": results,
            "phase_a": phase_a if all(a in results for a in PHASE_A_ARMS) else None,
        }
        DATA.mkdir(parents=True, exist_ok=True)
        out = DATA / f"run_engine_challenger_{datetime.now():%Y%m%d}.json"
        out.write_text(json.dumps(rec, indent=2))
        print(f"wrote {out}")
    return 0


def _ensemble_lambdas(decided, folds, cache_dir, pre, base="C2", force=False
                      ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deviance-weighted blend of the learner λs (weights fit pre-sealed).
    Each learner's λs are corrected through the Phase-B base layer (C2 =
    linear edge, k fit on that learner's own pre-sealed OOF) BEFORE the
    deviance is computed, so the blend weights the corrected learners."""
    lams_h, lams_a, dev = [], [], []
    for ln in LEARNERS:
        oof = run_learner_oof(decided, folds, ln,
                              params=(RELAXED_LGBM_PARAMS if ln == "lgbm"
                                      else None),
                              cache=cache_dir / f"{ln}.parquet", force=force)
        if base == "C2":
            _k, lh, la = linear_correction(oof, pre)
        else:
            lh = oof["home_expected_runs"].to_numpy(float)
            la = oof["away_expected_runs"].to_numpy(float)
        lams_h.append(lh)
        lams_a.append(la)
        dh = _deviance(oof["home_score"].to_numpy(float)[pre], lh[pre])
        da = _deviance(oof["away_score"].to_numpy(float)[pre], la[pre])
        dev.append((dh + da) / 2.0)
    inv = 1.0 / np.maximum(np.asarray(dev), 1e-9)
    w = inv / inv.sum()
    wts = {ln: round(float(wi), 4) for ln, wi in zip(LEARNERS, w)}
    lam_h = sum(wi * lh for wi, lh in zip(w, lams_h))
    lam_a = sum(wi * la for wi, la in zip(w, lams_a))
    return lam_h, lam_a, wts


def _deviance(y: np.ndarray, lam: np.ndarray) -> float:
    from run_engine import poisson_deviance
    return poisson_deviance(y, lam)


def _score_direct(oof: pd.DataFrame, name: str) -> dict:
    """Score a direct-margin arm: parametric margin/total distributions."""
    from scipy.stats import norm, laplace
    m = oof["margin_hat"].to_numpy(float)
    t = oof["total_hat"].to_numpy(float)
    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)
    pre, hold = sealed_masks(oof)
    margin = (hs - as_).astype(int)
    total = (hs + as_).astype(int)
    resid_m = margin - m
    resid_t = total - t
    if name == "DIRECT_NORMAL":
        sm, st = resid_m[pre].std(), resid_t[pre].std()
        pm = norm.pdf(np.asarray(MARGIN_GRID)[None, :], m[:, None], sm)
        pt = norm.pdf(np.asarray(TOTAL_GRID)[None, :], t[:, None], st)
    else:
        bm = np.abs(resid_m[pre]).mean()
        bt = np.abs(resid_t[pre]).mean()
        pm = laplace.pdf(np.asarray(MARGIN_GRID)[None, :], m[:, None], bm)
        pt = laplace.pdf(np.asarray(TOTAL_GRID)[None, :], t[:, None], bt)
    pm = pm / np.maximum(pm.sum(axis=1, keepdims=True), 1e-12)
    pt = pt / np.maximum(pt.sum(axis=1, keepdims=True), 1e-12)
    pc = pm[:, MARGIN_GRID.index(2):].sum(axis=1)
    pw = pm[:, MARGIN_GRID.index(1):].sum(axis=1)
    yc = (margin >= 2).astype(float)
    yw = (hs > as_).astype(float)
    return {
        "margin_crps_pooled": round(float(_crps(pm[pre], MARGIN_GRID,
                                                margin[pre])), 4),
        "margin_crps_sealed": round(float(_crps(pm[hold], MARGIN_GRID,
                                                margin[hold])), 4),
        "totals_crps_pooled": round(float(_crps(pt[pre], TOTAL_GRID,
                                                total[pre])), 4),
        "totals_crps_sealed": round(float(_crps(pt[hold], TOTAL_GRID,
                                                total[hold])), 4),
        "cover_cal_pooled": calibrate(pc[pre], yc[pre]),
        "cover_cal_sealed": calibrate(pc[hold], yc[hold]),
        "pwin_sd_pooled": round(float(pw[pre].std()), 4),
        "pwin_sd_sealed": round(float(pw[hold].std()), 4),
        "pwin_ece_pooled": round(float(_ece(yw[pre], pw[pre])), 4),
        "pwin_ece_sealed": round(float(_ece(yw[hold], pw[hold])), 4),
        "over_cal_pooled": {"overall_delta": None},
        "over_cal_sealed": {"overall_delta": None},
        "meta": {"residual": name.split("_")[1].lower()},
    }


def _phase_a_verdict(results: dict) -> dict:
    """Phase-A leader rule: sealed margin CRPS (primary), per-line cover
    calibration not degraded, totals within noise of C0, pooled corroboration."""
    c0 = results["C0"]
    best = "C0"
    reason = "no challenger beat C0 on sealed CRPS with pooled corroboration"
    for n in ("C1", "C2", "C3"):
        r = results[n]
        sealed_win = r["margin_crps_sealed"] < c0["margin_crps_sealed"] \
            - CRPS_IMPROVE
        pooled_ok = r["margin_crps_pooled"] < c0["margin_crps_pooled"]
        totals_ok = r["totals_crps_sealed"] <= c0["totals_crps_sealed"] \
            + TOTALS_TOL
        if sealed_win and pooled_ok and totals_ok:
            best = n
            reason = (f"{n} beats C0 on sealed margin CRPS "
                      f"({r['margin_crps_sealed']} < "
                      f"{c0['margin_crps_sealed']}), pooled corroborates, "
                      f"totals within noise")
            break
    return {"winner": best, "reason": reason,
            "sealed_crps": {n: results[n]["margin_crps_sealed"]
                            for n in PHASE_A_ARMS},
            "pooled_crps": {n: results[n]["margin_crps_pooled"]
                            for n in PHASE_A_ARMS}}


if __name__ == "__main__":
    raise SystemExit(main())
