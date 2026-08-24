"""Run engine Phase 1 — per-team expected-runs models (λ per side).

THE GOLDEN RULE: run models consume LEVELS + ENVIRONMENT only. Diff features
(sp_era_diff ≈ 0 for ace-vs-ace AND bad-vs-bad) carry no information about
scoring LEVEL, so they are excluded — EXCEPT park_factor_slug_diff, which is a
park-context term, and the four engineered interactions that are products of
excluded diffs. The kept list is DERIVED from FEATURE_COLS at call time so new
features flow in (and are logged); only the exclusion RULE lives here.

One regularized LightGBM regressor (objective="poisson") per side, trained on
the SAME fixed walk-forward folds as the moneyline pipeline. Pooled OOF scoring,
baseline comparison vs constant league-mean, Pearson chi-square/dispersion
probe for the Phase-2 Poisson-vs-negative-binomial decision.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from config import (
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)

logger = logging.getLogger(__name__)

# Excluded despite not ending in _diff: pure matchup term (no level info) and
# interactions whose factors are themselves excluded diffs.
RUN_EXTRA_EXCLUSIONS = {
    "lineup_handedness_matchup_advantage",
    "bullpen_meltdown_risk",          # pitches_diff × whip_diff
    "pitcher_regression_indicator",   # velo_diff × era_diff
    "lineup_depth_multiplier",        # woba_mean_diff × top3_diff
    "ace_efficiency_factor",          # k9_diff × whiff_diff
}
# The one sanctioned _diff survivor: a PARK context multiplier, not a matchup gap.
RUN_DIFF_EXCEPTION = "park_factor_slug_diff"

MAX_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 20  # matches the tuned LightGBM fold convention

RUN_LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 40,
    "min_gain_to_split": 0.5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Feature-view derivation
# ---------------------------------------------------------------------------
def derive_run_features(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Derive (run_features, dropped) from FEATURE_COLS by rule:

      drop  f  if f.endswith("_diff") and f != RUN_DIFF_EXCEPTION
         or  f in RUN_EXTRA_EXCLUSIONS

    Everything else flows in automatically (new level/env features included
    without touching this file). Returns both lists so callers log the drops.
    """
    run_feats, dropped = [], []
    for f in feature_cols:
        if f.endswith("_diff") and f != RUN_DIFF_EXCEPTION:
            dropped.append(f)
        elif f in RUN_EXTRA_EXCLUSIONS:
            dropped.append(f)
        else:
            run_feats.append(f)
    return run_feats, dropped


def split_side_view(run_features: list[str],
                    side: str) -> tuple[list[str], list[str]]:
    """Split run features into (side_view, env_shared) for 'home' or 'away'.

    Side columns: *_home / home_* (symmetrically for away). Shared environment
    = everything that belongs to NEITHER side (dome flag, park multiplier,
    weather, is_home) — never the opponent's levels."""
    other = "away" if side == "home" else "home"
    side_cols = [f for f in run_features
                 if f.endswith(f"_{side}") or f.startswith(f"{side}_")]
    other_cols = {f for f in run_features
                  if f.endswith(f"_{other}") or f.startswith(f"{other}_")}
    env_cols = [f for f in run_features
                if f not in side_cols and f not in other_cols]
    return side_cols, env_cols


def build_side_frame(games: pd.DataFrame, side: str,
                     run_features: Optional[list[str]] = None,
                     dropped: Optional[list[str]] = None
                     ) -> tuple[pd.DataFrame, list[str]]:
    """Materialize the side's model frame (levels + environment), preserving
    NaN (LightGBM routes it natively). Logs the derivation once per call."""
    from training import FEATURE_COLS

    feats = list(run_features) if run_features is not None else None
    if feats is None or dropped is None:
        feats, dropped = derive_run_features(list(FEATURE_COLS))
        logger.info("Run engine: %d/%d features kept; dropped %d: %s",
                    len(feats), len(FEATURE_COLS), len(dropped), dropped)
    side_cols, env_cols = split_side_view(feats, side)
    cols = side_cols + env_cols
    frame = games.reindex(columns=cols).astype(float)
    return frame, cols


# ---------------------------------------------------------------------------
# Training / OOF scoring
# ---------------------------------------------------------------------------
def _fit_side_model(params: dict, tr_frame: pd.DataFrame, y_tr: np.ndarray,
                    va_frame: pd.DataFrame, y_va: np.ndarray):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model = LGBMRegressor(**params)
    model.set_params(n_estimators=MAX_ROUNDS)
    model.fit(
        tr_frame, y_tr,
        eval_set=[(va_frame, y_va)],
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   log_evaluation(period=0)],
    )
    best = int(model.best_iteration_ or MAX_ROUNDS)
    lam = np.clip(model.predict(va_frame, num_iteration=best), 1e-6, None)
    return model, lam, best


def poisson_deviance(y: np.ndarray, lam: np.ndarray) -> float:
    """Mean unit deviance: 2·[y·ln(y/λ) − (y − λ)]; the y=0 term reduces to 2λ."""
    y = np.asarray(y, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-10, None)
    terms = np.empty_like(y)
    pos = y > 0
    terms[pos] = y[pos] * np.log(y[pos] / lam[pos]) - (y[pos] - lam[pos])
    terms[~pos] = lam[~pos]
    return float(2.0 * terms.mean())


def dispersion_ratio(y: np.ndarray, lam: np.ndarray) -> float:
    """Pearson chi-square / df. ≈1 → Poisson variance is adequate;
    clearly >1 → over-dispersion (negative-binomial upgrade territory)."""
    y = np.asarray(y, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-10, None)
    chi2 = float(((y - lam) ** 2 / lam).sum())
    df = max(int(len(y)) - 1, 1)
    return chi2 / df


def run_oof(games: pd.DataFrame,
            retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
            min_val_games: int = MIN_VAL_FOLD_GAMES,
            ) -> dict[str, Any]:
    """Walk-forward OOF for both side models on the moneyline pipeline's folds.

    Returns rows (one per decided game), per-side metrics, baseline metrics,
    and the dispersion probe."""
    from training import walk_forward_splits

    games = games[games["home_win"].notna()].reset_index(drop=True)
    folds = [
        s for s in walk_forward_splits(games, retrain_cadence_days=retrain_cadence_days)
        if len(s["val_games"]) >= min_val_games
    ]
    logger.info("Run engine: %d walk-forward folds, %d decided games",
                len(folds), len(games))

    frames = {side: build_side_frame(games, side) for side in ("home", "away")}
    params = dict(RUN_LGBM_PARAMS)

    out_rows: list[dict] = []
    metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}
    base_metrics: dict[str, dict[str, list[float]]] = {
        s: {"deviance": [], "rmse": [], "mae": []} for s in ("home", "away")}

    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        # game_id mirrors the moneyline predictions_history convention
        # ({YYYYMMDD}_{away}@{home}) so Phase 2 can merge against it.
        _d = pd.to_datetime(va["game_date"]).dt.strftime("%Y%m%d")
        rec_base = {"game_pk": va["game_pk"].to_numpy(),
                    "game_date": pd.to_datetime(va["game_date"]).dt.strftime("%Y-%m-%d"),
                    "fold_idx": split["fold_idx"],
                    "game_id": (va["game_id"] if "game_id" in va.columns else
                                _d + "_" + va["away_team"] + "@" + va["home_team"]) if
                    ("game_id" in va.columns or {"home_team", "away_team"}.issubset(va.columns))
                    else pd.Series([""] * len(va), index=va.index)}
        for side, target in (("home", "home_score"), ("away", "away_score")):
            _, cols_all = frames[side]
            tr_frame = tr.reindex(columns=cols_all).astype(float)
            va_frame = va.reindex(columns=cols_all).astype(float)
            y_tr = tr[target].to_numpy(dtype=float)
            y_va = va[target].to_numpy(dtype=float)
            _, lam, best = _fit_side_model(params, tr_frame, y_tr, va_frame, y_va)
            key = f"{side}_expected_runs"
            rec_base[key] = np.round(lam, 4)
            rec_base[target] = y_va.astype(int)
            m = metrics[side]
            m["deviance"].append(poisson_deviance(y_va, lam))
            m["rmse"].append(float(np.sqrt(((y_va - lam) ** 2).mean())))
            m["mae"].append(float(np.abs(y_va - lam).mean()))
            # Constant league-mean baseline (TRAIN mean — no val leakage).
            mu = float(y_tr.mean())
            bm = base_metrics[side]
            bm["deviance"].append(poisson_deviance(y_va, np.full(len(y_va), mu)))
            bm["rmse"].append(float(np.sqrt(((y_va - mu) ** 2).mean())))
            bm["mae"].append(float(np.abs(y_va - mu).mean()))
        out_rows.extend(pd.DataFrame(rec_base).to_dict(orient="records"))

    oof = pd.DataFrame(out_rows).sort_values(["game_date", "game_pk"])
    summary: dict[str, Any] = {"n_folds": len(folds), "n_games": len(oof)}
    for side in ("home", "away"):
        summary[f"{side}_model"] = {
            "poisson_deviance" if k == "deviance" else k:
                round(float(np.average(v)), 5)
            for k, v in metrics[side].items()}  # fold-mean; pooled below is exact
        summary[f"{side}_baseline"] = {
            "poisson_deviance" if k == "deviance" else k:
                round(float(np.average(v)), 5)
            for k, v in base_metrics[side].items()}
    # Pooled (exact) recomputation over all OOF rows — never fold averages.
    for side in ("home", "away"):
        y = oof[f"{side}_score"].to_numpy(float)
        lam = oof[f"{side}_expected_runs"].to_numpy(float)
        summary[f"{side}_pooled"] = {
            "poisson_deviance": round(poisson_deviance(y, lam), 5),
            "rmse": round(float(np.sqrt(((y - lam) ** 2).mean())), 5),
            "mae": round(float(np.abs(y - lam).mean()), 5),
        }
        summary[f"{side}_dispersion_ratio"] = round(dispersion_ratio(y, lam), 4)
    return {"oof": oof, "summary": summary}


# Persisted Phase-1 contract. fold_idx / game_id ride along internally
# (Phase 2's prequential calibration and agreement merge need them) but stay
# out of the CSV schema.
OOF_COLUMNS = ["game_pk", "game_date", "home_expected_runs",
               "away_expected_runs", "home_score", "away_score"]


def persist_oof(oof: pd.DataFrame, target_date_str: str,
                out_dir: Optional[Path] = None) -> Path:
    """run_engine_oof_<date>.csv — Phase 2's input contract."""
    out_path = (out_dir or DATA_DELIVERY_DIR) / f"run_engine_oof_{target_date_str}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in OOF_COLUMNS if c not in oof.columns]
    if missing:
        raise ValueError(f"OOF frame missing required columns: {missing}")
    tmp = out_path.with_suffix(".csv.tmp")
    oof[OOF_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(out_path)
    logger.info("Run engine OOF: %d rows -> %s", len(oof), out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# Phase 2 — market derivation (NB marginals + Monte Carlo; NEVER Skellam/Poisson:
# Phase 1 measured χ²/df ≈ 2.2–2.5, and the difference of two NBs has no closed
# form — Monte Carlo is the honest route).
# ---------------------------------------------------------------------------
MARKET_SEED = 42
MC_DRAWS = 10_000
TOTAL_LINE = 8.5          # over = home+away ≥ 9
RUN_LINE_MARGIN = 1.5     # home cover = home − away ≥ 2
ALPHA_FLOOR = 1e-6        # α below this ≈ Poisson; sample with huge n instead

MARKET_COLUMNS = ["game_pk", "game_date", "home_expected_runs",
                  "away_expected_runs", "p_over_8_5", "p_home_cover_1_5",
                  "p_home_win_derived", "home_score", "away_score",
                  "total_runs"]


def fit_alpha(y: np.ndarray, lam: np.ndarray) -> float:
    """Method-of-moments NB dispersion: var = λ̄ + α·λ̄²  ⇒
    α = max((var_obs − λ̄) / λ̄², 0)."""
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam_bar = float(lam.mean())
    var_obs = float(y.var(ddof=0))
    return round(max((var_obs - lam_bar) / (lam_bar ** 2), 0.0), 4)


def _nb_size_prob(mu: np.ndarray | float,
                  alpha: float) -> tuple[float, np.ndarray | float]:
    """numpy's NB(n, p): mean n(1−p)/p = μ ⇒ p = n/(n+μ); variance μ + μ²/n
    with n = 1/α. α below ALPHA_FLOOR → huge n (Poisson limit)."""
    n = 1.0 / alpha if alpha > ALPHA_FLOOR else 1e12
    p = n / (n + mu)
    return n, p


def nb_pmf(k: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """NB(μ, α) pmf at integer k ≥ 0 (scipy-free, log-space stable)."""
    from math import lgamma, exp, log
    n = 1.0 / alpha if alpha > ALPHA_FLOOR else 1e12
    out = np.empty(len(k), dtype=float)
    for i, ki in enumerate(k):
        out[i] = exp(lgamma(ki + n) - lgamma(n) - lgamma(ki + 1)
                     + n * log(n / (n + mu)) + ki * log(mu / (n + mu)))
    return out


def fit_check_table(actual: np.ndarray, lam: np.ndarray, alpha: float,
                    kmax: int = 12) -> list[dict]:
    """NB(λ̄, α) implied score distribution vs observed OOF distribution."""
    actual = np.asarray(actual, dtype=int)
    lam_bar = float(np.asarray(lam, dtype=float).mean())
    ks = np.arange(0, kmax + 1)
    modeled = nb_pmf(ks, lam_bar, alpha)
    observed = np.array([(actual == k).mean() for k in ks])
    tail_ge10_m = float(modeled[ks >= 10].sum())
    tail_ge10_o = float((actual >= 10).mean())
    le1_m = float(modeled[ks <= 1].sum())
    le1_o = float((actual <= 1).mean())
    rows = [{"k": int(k), "modeled_p": round(float(m), 4),
             "observed_p": round(float(o), 4)}
            for k, m, o in zip(ks, modeled, observed)]
    rows.append({"k": ">=10", "modeled_p": round(tail_ge10_m, 4),
                 "observed_p": round(tail_ge10_o, 4)})
    rows.append({"k": "<=1", "modeled_p": round(le1_m, 4),
                 "observed_p": round(le1_o, 4)})
    return rows


def derive_markets_mc(lam_home: np.ndarray, lam_away: np.ndarray,
                      alpha_home: float, alpha_away: float,
                      n_draws: int = MC_DRAWS,
                      seed: int = MARKET_SEED) -> dict[str, np.ndarray]:
    """Monte Carlo per game from INDEPENDENT NB(λ, α) marginals.

    Chunked over games to cap memory; identical seed ⇒ identical output.
    Returns p_over, p_home_cover, p_home_win, plus a per-game MC standard
    error for the totals market (binomial sqrt(p(1−p)/N))."""
    rng = np.random.default_rng(seed)
    n_games = len(lam_home)
    p_over = np.empty(n_games)
    p_cover = np.empty(n_games)
    p_win = np.empty(n_games)
    chunk = max(1, min(n_games, 2_000_000 // max(n_draws, 1)))
    for start in range(0, n_games, chunk):
        end = min(start + chunk, n_games)
        mu_h = np.maximum(np.asarray(lam_home[start:end], float), 1e-6)[:, None]
        mu_a = np.maximum(np.asarray(lam_away[start:end], float), 1e-6)[:, None]
        nh, ph_ = _nb_size_prob(mu_h, alpha_home)
        na, pa = _nb_size_prob(mu_a, alpha_away)
        # Explicit size: broadcast alone yields ONE draw per game.
        h = rng.negative_binomial(nh, ph_,
                                  size=(end - start, n_draws)).astype(np.int32)
        a = rng.negative_binomial(na, pa,
                                  size=(end - start, n_draws)).astype(np.int32)
        total = h + a
        diff = h - a
        p_over[start:end] = (total >= int(-(-TOTAL_LINE // 1))).mean(axis=1)
        p_cover[start:end] = (diff >= int(RUN_LINE_MARGIN) + 1).mean(axis=1)
        p_win[start:end] = (diff > 0).mean(axis=1)
    mc_se = np.sqrt(p_over * (1 - p_over) / n_draws)
    return {"p_over_8_5": p_over, "p_home_cover_1_5": p_cover,
            "p_home_win_derived": p_win, "mc_se_totals": mc_se}


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width binned expected calibration error."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return round(float(ece), 5)


def prequential_calibrate(y: np.ndarray, p: np.ndarray,
                          fold_idx: np.ndarray) -> np.ndarray:
    """Per-fold Platt maps fitted on PRIOR folds only (post-F2 discipline).
    Folds before MIN_OOF_FOR_FIT games accumulate keep raw probabilities.
    Never composes the global map on top of fold maps."""
    from calibration import apply_platt, fit_platt

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    folds = np.unique(fold_idx)
    history_y, history_p = [], []
    out = p.copy()
    for f in folds:
        m = fold_idx == f
        cal = fit_platt(np.concatenate(history_y) if history_y else [],
                        np.concatenate(history_p) if history_p else [])
        if cal is not None:
            out[m] = apply_platt(p[m], cal)
        history_y.append(y[m])
        history_p.append(p[m])
    return np.clip(out, 1e-6, 1 - 1e-6)


def derive_markets(oof: pd.DataFrame,
                   moneyline_probs: Optional[pd.DataFrame] = None,
                   n_draws: int = MC_DRAWS,
                   seed: int = MARKET_SEED) -> dict[str, Any]:
    """Full Phase-2 layer: fit α per side, NB fit-check, MC markets, honest
    pooled scoring vs base-rate baselines (raw + prequential calibrated),
    agreement groundwork vs the moneyline ensemble."""
    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)

    alpha_home = fit_alpha(hs, lam_h)
    alpha_away = fit_alpha(as_, lam_a)
    logger.warning(
        "Run engine Phase 2: fitted α_home=%.4f α_away=%.4f "
        "(Poisson rejected by Phase-1 dispersion)", alpha_home, alpha_away)

    summary: dict[str, Any] = {
        "alpha_home": alpha_home, "alpha_away": alpha_away,
        "n_draws": n_draws, "seed": seed,
        "total_line": TOTAL_LINE, "run_line_margin": RUN_LINE_MARGIN,
        "fit_check": {
            "home": fit_check_table(hs, lam_h, alpha_home),
            "away": fit_check_table(as_, lam_a, alpha_away),
        },
    }

    mc = derive_markets_mc(lam_h, lam_a, alpha_home, alpha_away,
                           n_draws=n_draws, seed=seed)
    total_runs = hs + as_
    y_over = (total_runs >= 9).astype(float)
    y_cover = ((hs - as_) >= 2).astype(float)
    y_win = (hs > as_).astype(float)

    # MC standard error on example games (metadata honesty).
    se = mc["mc_se_totals"]
    summary["mc_se_examples"] = {
        "first_game": round(float(se[0]), 6),
        "median": round(float(np.median(se)), 6),
        "max": round(float(se.max()), 6),
    }

    def score_market(name: str, p: np.ndarray, y: np.ndarray,
                     fold_idx: np.ndarray) -> dict:
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        base_p = float(np.asarray(y, float).mean())
        p_cal = prequential_calibrate(y, p, fold_idx)
        row = {
            "engine_logloss": round(log_loss(y, p), 5),
            "engine_brier": round(brier_score(y, p), 5),
            "engine_ece_raw": ece_score(y, p),
            "engine_ece_calibrated": ece_score(y, p_cal),
            "engine_logloss_calibrated": round(log_loss(y, p_cal), 5),
            "baseline_rate": round(base_p, 4),
            "baseline_logloss": round(log_loss(y, np.full(len(y), base_p)), 5),
            "baseline_brier": round(brier_score(y, np.full(len(y), base_p)), 5),
        }
        row["beats_baseline_logloss"] = bool(row["engine_logloss"] < row["baseline_logloss"])
        summary[f"market_{name}"] = row
        return row

    fold_idx = oof["fold_idx"].to_numpy()
    score_market("over_8_5", mc["p_over_8_5"], y_over, fold_idx)
    score_market("home_cover_1_5", mc["p_home_cover_1_5"], y_cover, fold_idx)
    score_market("derived_moneyline", mc["p_home_win_derived"], y_win, fold_idx)

    # Agreement groundwork vs the moneyline ensemble (if probabilities given).
    if moneyline_probs is not None and len(moneyline_probs):
        merged = pd.DataFrame({
            "game_id": oof["game_id"].to_numpy(),
            "p_run": mc["p_home_win_derived"],
        }).merge(
            moneyline_probs[["game_id", "home_win_prob_model"]],
            on="game_id", how="inner",
        )
        if len(merged) > 30:
            diff = (merged["p_run"] - merged["home_win_prob_model"]).abs()
            summary["agreement_vs_moneyline"] = {
                "n_merged": int(len(merged)),
                "correlation": round(float(merged["p_run"].corr(
                    merged["home_win_prob_model"])), 4),
                "mean_abs_diff": round(float(diff.mean()), 4),
                "share_gt_0_08": round(float((diff > 0.08).mean()), 4),
                "share_gt_0_10": round(float((diff > 0.10).mean()), 4),
            }

    markets = oof[["game_pk", "game_date", "home_expected_runs",
                   "away_expected_runs"]].copy()
    markets["p_over_8_5"] = np.round(mc["p_over_8_5"], 5)
    markets["p_home_cover_1_5"] = np.round(mc["p_home_cover_1_5"], 5)
    markets["p_home_win_derived"] = np.round(mc["p_home_win_derived"], 5)
    markets["home_score"] = hs.astype(int)
    markets["away_score"] = as_.astype(int)
    markets["total_runs"] = total_runs.astype(int)
    return {"markets": markets, "summary": summary}


def persist_markets(markets: pd.DataFrame, target_date_str: str,
                    summary: dict,
                    out_dir: Optional[Path] = None) -> Path:
    out_path = (out_dir or DATA_DELIVERY_DIR) / f"run_engine_markets_{target_date_str}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [c for c in MARKET_COLUMNS if c not in markets.columns]
    if missing:
        raise ValueError(f"markets frame missing required columns: {missing}")
    frame = markets[MARKET_COLUMNS]
    if frame.isna().any().any():
        raise ValueError("markets frame contains NaNs — refusing to persist")
    tmp = out_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path.with_suffix(".json.tmp"), "w") as f:
        json.dump(summary, f, indent=2)
    meta_path.with_suffix(".json.tmp").replace(meta_path)
    logger.info("Run engine markets: %d rows -> %s (+ meta json)", len(frame), out_path.name)
    return out_path


def load_moneyline_probs(csv_path: Path) -> Optional[pd.DataFrame]:
    """predictions_history columns needed for the agreement merge."""
    if not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path, usecols=lambda c: c in (
        "game_id", "home_win_prob_model"))
    return df if {"game_id", "home_win_prob_model"}.issubset(df.columns) else None


def _print_phase2(s: dict[str, Any]) -> None:
    print("\n========== PHASE 2 — NB MARGINALS ==========")
    print(f"α_home={s['alpha_home']}  α_away={s['alpha_away']}  "
          f"(N={s['n_draws']:,} draws, seed={s['seed']})")
    for side in ("home", "away"):
        print(f"\n{side.upper()} NB fit-check (NB(λ̄,{s[f'alpha_{side}']})):")
        print(f"{'k':>6}{'modeled':>10}{'observed':>10}")
        for row in s["fit_check"][side]:
            print(f"{str(row['k']):>6}{row['modeled_p']:>10.4f}{row['observed_p']:>10.4f}")
    print("\n========== MARKET SCORING (pooled OOF) ==========")
    print(f"{'market':<20}{'logloss':>9}{'brier':>8}{'ECE(raw)':>9}"
          f"{'ECE(cal)':>9}{'base LL':>9}{'ΔLL':>9}")
    for key, name in (("market_over_8_5", "over 8.5"),
                      ("market_home_cover_1_5", "home -1.5"),
                      ("market_derived_moneyline", "derived ML")):
        m = s[key]
        print(f"{name:<20}{m['engine_logloss']:>9.4f}{m['engine_brier']:>8.4f}"
              f"{m['engine_ece_raw']:>9.4f}{m['engine_ece_calibrated']:>9.4f}"
              f"{m['baseline_logloss']:>9.4f}"
              f"{m['engine_logloss'] - m['baseline_logloss']:>9.4f}")
    if "agreement_vs_moneyline" in s:
        a = s["agreement_vs_moneyline"]
        print(f"\nAgreement vs moneyline ensemble (n={a['n_merged']}): "
              f"corr={a['correlation']}, mean|Δ|={a['mean_abs_diff']}, "
              f">0.08: {100*a['share_gt_0_08']:.1f}%, >0.10: {100*a['share_gt_0_10']:.1f}%")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features.csv")
    ap.add_argument("--date", type=str, default=None,
                    help="Artifact date stamp (default: today)")
    args = ap.parse_args()
    import datetime
    date_stamp = args.date or datetime.date.today().strftime("%Y%m%d")

    df = pd.read_csv(args.data)
    result = run_oof(df)

    # Phase 2 consumes Phase 1's OOF λ directly (same process, no retrain).
    history_path = DATA_DELIVERY_DIR / f"predictions_history_{date_stamp}.csv"
    if not history_path.exists():
        history_path = DATA_DELIVERY_DIR / "predictions_history_latest.csv"
    ml_probs = load_moneyline_probs(history_path)

    mk = derive_markets(result["oof"], moneyline_probs=ml_probs)
    oof_path = persist_oof(result["oof"], date_stamp)
    mkt_path = persist_markets(mk["markets"], date_stamp, mk["summary"])

    s = result["summary"]
    print("\n================ RUN ENGINE PHASE 1 ================")
    print(f"folds={s['n_folds']}  oof games={s['n_games']}")
    print(f"\n{'metric':<18}{'home model':>12}{'away model':>12}{'baseline':>10}")
    for metric in ("poisson_deviance", "rmse", "mae"):
        print(f"{metric:<18}"
              f"{s['home_pooled'][metric]:>12.4f}"
              f"{s['away_pooled'][metric]:>12.4f}"
              f"{s['home_baseline'][metric]:>10.4f}"
              f"  (away base {s['away_baseline'][metric]:.4f})")
    for side in ("home", "away"):
        ratio = s[f"{side}_dispersion_ratio"]
        verdict = "Poisson OK" if ratio < 1.3 else (
            "MILD over-dispersion" if ratio < 1.8 else "STRONG over-dispersion")
        print(f"{side} dispersion ratio (Pearson χ²/df): {ratio:.3f} → {verdict}")
    print(f"\nOOF artifact: {oof_path}")
    _print_phase2(mk["summary"])
    print(f"\nMarkets artifact: {mkt_path}")


if __name__ == "__main__":
    main()
