"""
Walk-forward training for MLB Bet Predictor.

Implements expanding-window walk-forward splits, multi-target heads
(moneyline, totals, run line), evaluation metrics, and ensemble persistence.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from calibration import apply_platt, fit_platt, is_identity, MIN_OOF_FOR_FIT
from config import (
    ADAPTIVE_WEIGHT_AUC_TEMPERATURE,
    ADAPTIVE_WEIGHT_CAP,
    ADAPTIVE_WEIGHT_FLOOR,
    ADAPTIVE_WEIGHT_METRIC,
    ADAPTIVE_WEIGHT_TEMPERATURE,
    DATA_DELIVERY_DIR,
    DATE_FMT,
    ENSEMBLE_FILE,
    ENSEMBLE_WEIGHTS,
    LIGHTGBM_PARAMS,
    LIGHTGBM_REG_PARAMS,
    MIN_VAL_FOLD_GAMES,
    MODELS_DIR,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
    VERSION_KEY,
    TRAINED_AT_KEY,
    DATA_CUTOFF_KEY,
    XGBOOST_PARAMS,
    XGBOOST_REG_PARAMS,
)

logger = logging.getLogger(__name__)

# Features used for model input — all diff/computed features.
# Diff convention: home − away (positive = home advantage).
#
# Pruned 2026-08-23 (feature_audit):
#   wind/air_density factors         constant 0 (weather backfill covers only
#                                    ~7% of games; starved, not broken —
#                                    re-add when coverage improves)
#   bullpen_ip_diff                  r=0.87 with bullpen_pitches_diff, weaker
#
# Wave-2 ablation candidates (univariate |lift| < ~0.01, need walk-forward
# retrain to confirm): sp_fbpct_diff, team_barrel_diff, lineup_woba_std_diff,
# park_factor_slug_diff, closer_availability_diff, travel_fatigue_diff,
# bullpen_whip_3g_diff, ace_efficiency_factor, pitcher_regression_indicator.
FEATURE_COLS = [
    # 1. Baseline (home-field anchor; constant by construction)
    "is_home",
    # 2–4. Core pre-game diffs
    "win_pct_diff",
    "elo_diff",
    "rest_days_diff",
    # 5–8. Starting pitcher diffs (season-to-date + last-5-start)
    "sp_era_diff",
    "sp_era_5g_diff",
    "sp_k9_diff",
    "sp_k9_5g_diff",
    # 9–11. SP stuff diffs (trailing 3-game)
    "sp_fbvelo_diff",
    "sp_fbpct_diff",
    "sp_whiff_diff",
    # 10–11. SP xwOBA diffs
    "sp_xwoba_diff",
    "sp_xwoba_vs_l_diff",
    # 12–14. Lineup wOBA diffs
    "lineup_woba_mean_diff",
    "lineup_woba_top3_diff",
    "lineup_woba_std_diff",
    # 15. Team rolling wOBA diff
    "woba_30g_diff",
    # 16–18. Bullpen diffs
    "bullpen_whip_diff",
    "bullpen_whip_3g_diff",
    "bullpen_pitches_diff",
    # 19–21. Team contact form diffs (trailing 15g)
    "team_barrel_diff",
    "team_hardhit_diff",
    "team_exitvelo_diff",
    # 22. Lineup handedness matchup advantage (OPS vs tonight's starter hand)
    "lineup_handedness_matchup_advantage",
    # 23. Travel fatigue (timezone crossings, last 3 days)
    "travel_fatigue_diff",
    # 24. Closer availability
    "closer_availability_diff",
    # 25. Dome neutral flag
    "dome_is_neutral",
    # 26. Park factor
    "park_factor_slug_diff",
    # 27–28. Weather-driven interactions from REAL Open-Meteo observations
    # (validated 2026-08: varied, venue-sane values incl. Coors thin-air).
    # Missing observations stay NULL; dome wind is a valid neutral 0.
    "wind_advantage_flyball_factor",
    "air_density_velocity_boost",
    # 29–32. Derived interaction features
    "bullpen_meltdown_risk",
    "pitcher_regression_indicator",
    "lineup_depth_multiplier",
    "ace_efficiency_factor",
    # 33–56. Raw per-side inputs (home/away pre-differenced values).
    # Gives every member the raw home and away values alongside their diffs,
    # letting tree members discover side-specific thresholds and interactions
    # that a pure home-minus-away diff cannot express.
    # Elo
    "home_elo",
    "away_elo",
    # Win percentage
    "home_win_pct",
    "away_win_pct",
    # SP ERA (season-to-date)
    "sp_era_home",
    "sp_era_away",
    # SP K/9 (season-to-date)
    "sp_k9_home",
    "sp_k9_away",
    # SP xwOBA allowed (last 6 starts)
    "sp_xwoba_home",
    "sp_xwoba_away",
    # Lineup mean wOBA
    "lineup_woba_mean_home",
    "lineup_woba_mean_away",
    # Lineup top-3 wOBA
    "lineup_woba_top3_home",
    "lineup_woba_top3_away",
    # Team 30-game wOBA
    "woba_30g_home",
    "woba_30g_away",
    # Bullpen 10-game WHIP
    "bullpen_whip_10g_home",
    "bullpen_whip_10g_away",
    # Bullpen 3-game WHIP
    "bullpen_whip_3g_home",
    "bullpen_whip_3g_away",
    # Team barrel% (15-game)
    "team_barrel_15g_home",
    "team_barrel_15g_away",
    # Team exit velocity (15-game)
    "team_exitvelo_15g_home",
    "team_exitvelo_15g_away",
]
# Deduplicate (should already be unique but defensive)
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))


# ── Walk-forward splits ─────────────────────────────────────────────────────

def walk_forward_splits(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    max_eval_folds: int = 0,
    min_train_days: int = 0,
) -> list[dict[str, Any]]:
    """Generate expanding-window walk-forward train/val splits.

    Each validation window is `retrain_cadence_days` wide. The training set
    is all games strictly before the validation window start. Windows are
    non-overlapping and chronological.

    Args:
        min_train_days: Skip validation windows that start before this many
            calendar days of history. Prevents tiny-training-set folds from
            polluting pooled metrics with noise (default 0 = no warm-up).

    Returns a list of dicts with keys:
        train_games: DataFrame of training games
        val_games: DataFrame of validation games
        fold_idx: int
        val_start: datetime
        val_end: datetime
    """
    if "game_date" not in games.columns:
        raise ValueError("games must have a 'game_date' column")
    if "home_win" not in games.columns:
        logger.warning("walk_forward_splits: no 'home_win' column — cannot split")
        return []

    df = games.dropna(subset=["home_win"]).copy()
    if df.empty:
        logger.warning(
            "walk_forward_splits: all %d rows have NaN home_win — cannot split",
            len(games),
        )
        return []
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Normalize to date-only (strip time) so unique dates represent calendar days,
    # not individual timestamps. Without this, each game with a unique start time
    # becomes its own "date" and 7-day validation windows collapse to 1 game.
    df["game_date"] = df["game_date"].dt.normalize()
    df = df.sort_values("game_date").reset_index(drop=True)

    if df.empty:
        return []

    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) < retrain_cadence_days + 1:
        # Not enough data for even one split — use all as train, none as val
        logger.warning(
            "walk_forward_splits: only %d unique dates (need >= %d for cadence %d)",
            len(unique_dates), retrain_cadence_days + 1, retrain_cadence_days,
        )
        return []

    splits = []
    fold_idx = 0

    # Start validation windows after the warm-up period so every fold has
    # enough training history to produce meaningful predictions.
    val_start_idx = max(retrain_cadence_days, min_train_days)
    while val_start_idx < len(unique_dates):
        val_start = unique_dates[val_start_idx]
        val_end_idx = min(val_start_idx + retrain_cadence_days, len(unique_dates))
        val_end = unique_dates[val_end_idx - 1]

        # Training: everything strictly before val_start
        train_mask = df["game_date"] < val_start
        val_mask = (df["game_date"] >= val_start) & (df["game_date"] <= val_end)

        train_games = df[train_mask].copy()
        val_games = df[val_mask].copy()

        if not train_games.empty and not val_games.empty:
            splits.append({
                "train_games": train_games,
                "val_games": val_games,
                "fold_idx": fold_idx,
                "val_start": val_start,
                "val_end": val_end,
            })
            fold_idx += 1

        val_start_idx = val_end_idx

    # Limit to max_eval_folds (most recent folds)
    if max_eval_folds > 0 and len(splits) > max_eval_folds:
        splits = splits[-max_eval_folds:]

    return splits


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """Compute classification metrics: AUC, Brier, LogLoss, ECE."""
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)

    # Clip to avoid log(0)
    y_pred_prob = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)

    result = {}
    try:
        result["auc"] = round(float(roc_auc_score(y_true, y_pred_prob)), 4)
    except ValueError:
        result["auc"] = 0.5

    result["brier"] = round(float(brier_score_loss(y_true, y_pred_prob)), 4)
    result["logloss"] = round(float(log_loss(y_true, y_pred_prob)), 4)
    result["ece"] = round(float(_expected_calibration_error(y_true, y_pred_prob)), 4)

    return result


def _expected_calibration_error(
    y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred_prob >= bin_edges[i]) & (y_pred_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def calibration_buckets(
    y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Compute calibration bucket data for the dashboard.

    Observations are taken from the FAVORED team's perspective: each game
    contributes ONE point at probability max(p_home, p_away) ∈ [0.5, 1],
    labeled by whether the favorite actually won. This matches how the
    model is consumed (you bet the pick) and is information-equivalent
    to the home-side view, since (p, y) and (1 − p, 1 − y) are exact
    complements — every metric derived from these buckets mirrors the
    home-side version.
    """
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    fav_prob = np.maximum(y_pred_prob, 1.0 - y_pred_prob)
    fav_won = np.where(y_pred_prob >= 0.5, y_true, 1.0 - y_true)
    bin_edges = np.linspace(0.5, 1.0, max(n_bins // 2, 1) + 1)
    buckets = []
    for i in range(len(bin_edges) - 1):
        mask = (fav_prob >= bin_edges[i]) & (fav_prob < bin_edges[i + 1])
        if i == len(bin_edges) - 2:  # include 1.0 in the top bucket
            mask |= fav_prob == bin_edges[i + 1]
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = round(float(fav_prob[mask].mean()), 4)
        mean_actual = round(float(fav_won[mask].mean()), 4)
        gap = round(mean_pred - mean_actual, 4)
        buckets.append({
            "bucket": f"{bin_edges[i]*100:.0f}–{bin_edges[i+1]*100:.0f}%",
            "mean_predicted": mean_pred,
            "mean_actual": mean_actual,
            "count": count,
            "gap": gap,
        })
    return buckets


# ── Moneyline ensemble ─────────────────────────────────────────────────────

def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix preserving NaN.

    Missing observations stay NULL: XGBoost/LightGBM route them natively and
    zero-filling fabricated signal (a 0 mph fastball, a 0.000 wOBA). Only the
    logistic/MLP members — which cannot consume NaN — get train-median
    imputation, applied at predict time via the medians stored in the models
    dict. Team IDs route through a separate categorical path (LightGBM).
    """
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].to_numpy(dtype=float)


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract feature matrix, categorical matrix, and target.

    Returns (X_numeric, X_categorical, y). X_categorical carries team IDs
    for tree members that use native categorical support (LightGBM).
    """
    df = _add_team_ids(df)
    X = _feature_matrix(df)
    X_cat = _categorical_matrix(df)
    y = df["home_win"].values.astype(float)
    return X, X_cat, y


def _impute_median(
    X: np.ndarray, medians: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN with column medians (fit on train when medians is None).

    All-NaN columns fall back to 0.0 so the logistic member stays usable.
    Returns (imputed_matrix, medians_used).
    """
    X = np.asarray(X, dtype=float)
    if medians is None:
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
        medians = np.where(np.isnan(medians), 0.0, medians)
    X = X.copy()
    idx = np.isnan(X)
    X[idx] = np.take(np.asarray(medians, dtype=float), idx.nonzero()[1])
    return X, np.asarray(medians, dtype=float)


# Adaptive blend weights from the most recent walk-forward run. Empty until
# the first walk_forward_evaluate() completes; falls back to the static
# ENSEMBLE_WEIGHTS priors before that.
_LAST_ADAPTIVE_WEIGHTS: dict[str, float] = {}

# Post-hoc Platt calibrator from the most recent walk-forward run. Applied
# to live blended probabilities in predict_games(); restored from a cached
# bundle via set_calibration() so cached-model runs stay consistent.
_LAST_CALIBRATOR: dict | None = None

# ── Team ID mapping for tree-member categoricals ──────────────────────────

# Consistent integer IDs from 3-letter team abbreviations. Same team = same
# ID across seasons (verified: Statcast team_id is stable). Built lazily
# from observed data so expansion teams get IDs automatically.
_TEAM_ABBR_TO_ID: dict[str, int] = {}
_TEAM_ID_TO_ABBR: dict[int, str] = {}

def _team_id(abbr: str) -> int:
    """Convert a 3-letter team abbreviation to a stable integer ID.

    Unknown / invalid abbreviations (expansion teams, All-Star rosters,
    missing data) map to UNK_TEAM_ID — a dedicated category with
    near-zero training presence so trees learn a neutral weight for it
    instead of silently aliasing a real team (e.g., 0 = NYY).
    """
    if abbr in _TEAM_ABBR_TO_ID:
        return _TEAM_ABBR_TO_ID[abbr]
    if not isinstance(abbr, str) or len(abbr) < 2:
        return UNK_TEAM_ID  # semantic "unknown", not a real team
    tid = len(_TEAM_ABBR_TO_ID)
    _TEAM_ABBR_TO_ID[abbr] = tid
    _TEAM_ID_TO_ABBR[tid] = abbr
    return tid

# Tree-member-only categorical columns. NOT in FEATURE_COLS — they get
# native categorical handling in LightGBM. Logistic/MLP must not receive
# raw team IDs (one-hot would starve on ~4k rows with 30 categories).
TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]

# Dedicated "unknown team" category ID.  Never collides with a real team because
# it sits above the MLB team-space (~30 teams).  Unknown / invalid / predict-time
# expansion-team abbreviations map here instead of silently aliasing a real team
# (e.g. 0 = NYY).  With near-zero training presence, trees learn a neutral weight.
UNK_TEAM_ID = 99

# RF ablation toggle: set False to train RandomForest WITHOUT team IDs
# (for measuring the marginal benefit of team IDs on the RF member).
# Default True = team IDs included (the production config).
RF_WITH_TEAM_IDS = True

def _add_team_ids(df: "pd.DataFrame") -> "pd.DataFrame":
    """Attach stable integer team IDs for tree-member categorical routing."""
    import pandas as pd
    df = df.copy()
    df["home_team_id"] = df["home_team"].apply(_team_id)
    df["away_team_id"] = df["away_team"].apply(_team_id)
    return df

def _categorical_matrix(df: "pd.DataFrame") -> "np.ndarray":
    """Extract categorical-feature matrix (team IDs, for tree members)."""
    import numpy as np
    return df[TREE_CATEGORICAL_COLS].to_numpy(dtype=int)


def _tree_dataframe(
    X_num: "np.ndarray",
    X_cat: "np.ndarray",
    numeric_cols: list[str],
) -> "pd.DataFrame":
    """Build a DataFrame with named numeric + categorical columns.

    Numeric columns preserve their names from FEATURE_COLS. Team-ID columns
    are converted to pandas Categorical (XGBoost) / kept as int (LightGBM
    uses categorical_feature= by name). Unknown team IDs are mapped to
    UNK_TEAM_ID by _team_id — a dedicated category that never aliases a
    real team — so no NaN or silent misidentification reaches the trees.
    """
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(X_num, columns=numeric_cols)
    for i, c in enumerate(TREE_CATEGORICAL_COLS):
        vals = X_cat[:, i].copy()
        # Safety: clamp any lingering negatives / sub-0 values to UNK_TEAM_ID
        # instead of the first real team.  (Should never fire after the
        # _team_id fix, but belt-and-suspenders.)
        vals = np.where(vals < 0, UNK_TEAM_ID, vals)
        df[c] = pd.Categorical(vals)
    return df



def compute_adaptive_weights(
    oof_members: dict[str, list[float]], y_oof: np.ndarray
) -> dict[str, float]:
    """Blend weights earned by out-of-sample performance.

    Softmax over pooled OOF scores. With ADAPTIVE_WEIGHT_METRIC="auc"
    (default) the score is pooled OOF AUC — a member beating another by Δ
    earns exp(Δ / TEMPERATURE) times its weight, so the blend leans toward
    the members that actually separate winners from losers. With
    "logloss" it scores pooled OOF log-loss (lower is better) as before.
    FLOOR keeps every candidate alive for diversity; CAP prevents
    domination. The result sums to exactly 1.0 and feeds both prediction
    blending and reporting so the ensemble visibly self-corrects as
    features improve.
    """
    scores: dict[str, float] = {}
    y = np.asarray(y_oof, dtype=float)
    if len(y) == 0:
        return {}
    for name, preds in oof_members.items():
        if not preds or len(preds) != len(y):
            continue
        m = compute_metrics(y, np.asarray(preds, dtype=float))
        if ADAPTIVE_WEIGHT_METRIC == "auc":
            a = m.get("auc")
            if a is None or not np.isfinite(a):
                continue
            scores[name] = float(a)
        else:
            ll = m.get("logloss")
            if ll is None or not np.isfinite(ll):
                continue
            scores[name] = float(ll)
    if not scores:
        return {}

    if ADAPTIVE_WEIGHT_METRIC == "auc":
        _t = ADAPTIVE_WEIGHT_AUC_TEMPERATURE
        best = max(scores.values())
        exp_w = {n: np.exp((a - best) / _t)
                 for n, a in scores.items()}
    else:
        _t = ADAPTIVE_WEIGHT_TEMPERATURE
        best = min(scores.values())
        exp_w = {n: np.exp(-(ll - best) / _t)
                 for n, ll in scores.items()}
    tot = sum(exp_w.values())
    w = {n: float(v / tot) for n, v in exp_w.items()}

    # A per-member cap C is satisfiable only if n_members × C >= 1; widen
    # it slightly past 1/n for small rosters so the constraint set stays
    # feasible (with 2 members, 0.45 each is impossible).
    eff_cap = max(ADAPTIVE_WEIGHT_CAP, 1.02 / len(w))

    # Iterative floor/cap projection until both constraints hold
    for _ in range(50):
        w = {n: max(v, ADAPTIVE_WEIGHT_FLOOR) for n, v in w.items()}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}
        w = {n: min(v, eff_cap) for n, v in w.items()}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}

    # Round without breaking the exact 1.0 total: give the rounding
    # remainder to the largest weight.
    rounded = {n: round(v, 4) for n, v in w.items()}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(rounded, key=lambda n: w[n])
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def _member_weights(member_names: list[str]) -> dict[str, float]:
    """Normalized blend weights for the members that actually trained.

    Prefers adaptive weights earned from pooled OOF log-loss when available;
    falls back to static ENSEMBLE_WEIGHTS priors otherwise (e.g. mid-run or
    before the first full evaluation). Members that failed to train
    contribute 0% and the remainder renormalizes to exactly 1.0.
    """
    names = [n for n in member_names if n not in ("scaler", "impute_median")]
    source = _LAST_ADAPTIVE_WEIGHTS or ENSEMBLE_WEIGHTS
    raw = {n: float(source.get(n, 0.0)) for n in names}
    # A candidate with no earned weight still gets its static prior so it
    # can prove itself on the next OOF cycle instead of being locked out.
    zeroed = [n for n, v in raw.items() if v <= 0]
    for n in zeroed:
        prior = float(ENSEMBLE_WEIGHTS.get(n, 0.0))
        if prior > 0:
            raw[n] = min(prior, ADAPTIVE_WEIGHT_FLOOR * 2)
    total = sum(raw.values())
    if total <= 0:
        w = 1.0 / max(len(names), 1)
        return {n: w for n in names}
    return {n: v / total for n, v in raw.items()}


def feature_importance_weights(ml_models: dict[str, Any]) -> dict[str, float] | None:
    """Blend-weighted feature importance across ensemble members (sums to 100).

    Each member's importances are normalized internally, then averaged with
    the member's configured ENSEMBLE_WEIGHTS share — so the result answers
    "what fraction of the final blended model rides on this feature?"
    Tree members contribute split-gain importance; logistic contributes
    |coefficient|. Returns None when no member exposes importances.
    """
    members = {n: m for n, m in ml_models.items()
               if n not in ("scaler", "impute_median")}
    if not members:
        return None
    eff = _member_weights(list(members.keys()))
    raw = {n: float(eff.get(n, 0.0)) for n in members}
    total = sum(raw.values())
    if total <= 0:
        raw = {n: 1.0 / len(members) for n in members}
        total = 1.0

    agg = np.zeros(len(FEATURE_COLS))
    contributed = False
    for name, model in members.items():
        try:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float).ravel()
            elif hasattr(model, "coef_"):
                imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
            else:
                continue
        except Exception:
            continue
        # Tree members trained with team-ID categoricals have larger
        # feature-importance vectors; trim to numeric FEATURE_COLS only.
        nfc = len(FEATURE_COLS)
        if len(imp) >= nfc:
            imp = imp[:nfc]
        if len(imp) != nfc or imp.sum() <= 0:
            continue
        agg += (raw[name] / total) * (imp / imp.sum())
        contributed = True
    if not contributed or agg.sum() <= 0:
        return None
    return {f: round(float(w), 4) for f, w in zip(FEATURE_COLS, agg / agg.sum() * 100.0)}


def ensemble_predict(
    ml_models: dict[str, Any], games: pd.DataFrame
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Weighted-blend prediction plus per-member probabilities and weights.

    Returns (blended_prob, {member_name: prob_vector}, {member_name: weight}).
    Falls back to 0.5 when no member can predict.
    """
    games = _add_team_ids(games)
    X = _feature_matrix(games)
    X_cat = _categorical_matrix(games)
    X_tree = np.hstack([X, X_cat])
    scaler = ml_models.get("scaler")
    medians = ml_models.get("impute_median")

    members: dict[str, np.ndarray] = {}
    for name, model in ml_models.items():
        if name in ("scaler", "impute_median"):
            continue
        try:
            if name in ("logistic", "mlp"):
                Xi, _ = _impute_median(X, medians)
                Xuse = scaler.transform(Xi) if scaler is not None else Xi
            elif name == "xgboost":
                Xi, _ = _impute_median(X, medians)
                num_cols_in_data = [c for c in FEATURE_COLS if c in games.columns]
                Xuse = _tree_dataframe(Xi, X_cat, num_cols_in_data)
            elif name == "lightgbm":
                import pandas as pd
                num_cols_in_data = [c for c in FEATURE_COLS if c in games.columns]
                _df = pd.DataFrame(X, columns=num_cols_in_data)
                for i, c_ in enumerate(TREE_CATEGORICAL_COLS):
                    vals = np.where(X_cat[:, i] < 0, UNK_TEAM_ID, X_cat[:, i])
                    _df[c_] = vals.astype(int)
                Xuse = _df
            elif name == "randomforest":
                # If model was trained without team IDs (ablation), it expects
                # FEATURE_COLS dimensions. Detect from model's n_features_in_.
                if hasattr(model, "n_features_in_") and model.n_features_in_ == len(FEATURE_COLS):
                    Xuse = X  # ablation: numeric only
                else:
                    Xuse = X_tree  # production: numeric + int team IDs
            else:
                Xuse = X
            members[name] = model.predict_proba(Xuse)[:, 1]
        except Exception as e:
            logger.warning("Member %s failed to predict: %s", name, e)

    if not members:
        return np.full(len(games), 0.5), {}, {}

    weights = _member_weights(list(members.keys()))
    blend = np.zeros(len(games))
    for name, p in members.items():
        blend += weights[name] * p
    return blend, members, weights


# Candidate roster from the most recent walk_forward_evaluate() run:
# every candidate model with its blend weight and pooled out-of-sample
# AUC/Brier/LogLoss (None if it never produced predictions).
_LAST_ENSEMBLE_INFO: list[dict[str, Any]] = []


def last_ensemble_info() -> list[dict[str, Any]]:
    """Candidate-model report from the most recent walk-forward evaluation."""
    return [dict(e) for e in _LAST_ENSEMBLE_INFO]


def set_calibration(calibrator: dict | None) -> None:
    """Restore the post-hoc calibrator (e.g. from a persisted ensemble bundle)
    so published probabilities match the model that was actually evaluated."""
    global _LAST_CALIBRATOR
    _LAST_CALIBRATOR = dict(calibrator) if calibrator else None


def get_last_calibrator() -> dict | None:
    """Calibrator fitted on pooled OOF by the most recent walk-forward run."""
    return dict(_LAST_CALIBRATOR) if _LAST_CALIBRATOR else None


def set_adaptive_weights(weights: dict[str, float] | None) -> None:
    """Restore adaptive blend weights (e.g. from a persisted ensemble bundle)
    so prediction blending matches the model that was actually evaluated."""
    _LAST_ADAPTIVE_WEIGHTS.clear()
    if weights:
        _LAST_ADAPTIVE_WEIGHTS.update({k: float(v) for k, v in weights.items()})


def train_moneyline_ensemble(
    train: pd.DataFrame, val: Optional[pd.DataFrame] = None
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train the moneyline ensemble.

    ``val`` is supplied for walk-forward folds so the boosting members can
    evaluate against a strictly future holdout. When omitted, the function
    performs a fit-only refit on every decided game for the deployed bundle.
    """
    X_train, X_cat_train, y_train = _prepare_features(train)
    X_val = X_cat_val = y_val = None
    if val is not None:
        X_val, X_cat_val, y_val = _prepare_features(val)

    if len(X_train) == 0 or (X_val is not None and len(X_val) == 0):
        raise ValueError("Insufficient training or validation data")

    # Logistic cannot consume NaN — impute with TRAIN-fold medians only
    # (never val medians, which would leak).
    X_train_lr, impute_medians = _impute_median(X_train)
    X_val_scaled = None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lr)
    if X_val is not None:
        X_val_lr, _ = _impute_median(X_val, impute_medians)
        X_val_scaled = scaler.transform(X_val_lr)

    # Build tree-member feature matrices: numeric diffs + team-ID categoricals.
    # All three tree members receive team IDs. LightGBM uses native categorical
    # support; XGBoost and RandomForest treat them as integers (the small
    # cardinality — 30 teams — works fine as ordinal-like bins without one-hot).
    X_train_tree = np.hstack([X_train, X_cat_train])
    X_val_tree = np.hstack([X_val, X_cat_val]) if X_val is not None else None
    # Imputed-numeric + categorical (XGBoost/RF: impute NaN, keep team IDs).
    X_train_lr_tree = np.hstack([X_train_lr, X_cat_train])
    if X_val is not None:
        X_val_lr_tree = np.hstack([X_val_lr, X_cat_val])
    else:
        X_val_lr_tree = None

    models = {}

    # XGBoost — tuned config: train-median imputation + early stopping.
    # The raw NaN matrix that tree members used to consume natively is
    # replaced by the same train-fold-median-imputed matrix that logistic/MLP
    # use (no val leakage). Walk-forward folds get n_estimators=2000 +
    # early_stopping_rounds=20 on the val window (~50 median rounds at refit);
    # fit-only refits use XGBOOST_PARAMS directly with no early stopping.
    try:
        from xgboost import XGBClassifier
        from config import XGBOOST_FOLD_ROUNDS, XGBOOST_EARLY_STOP
        # XGBoost: named DataFrame with pd.Categorical team-ID columns.
        # enable_categorical=True (in XGBOOST_PARAMS) picks them up natively.
        num_cols_in_data = [c for c in FEATURE_COLS if c in train.columns]
        X_train_xgb = _tree_dataframe(X_train_lr, X_cat_train, num_cols_in_data)
        if X_val is not None:
            X_val_xgb = _tree_dataframe(X_val_lr, X_cat_val, num_cols_in_data)
            xgb = XGBClassifier(
                **XGBOOST_PARAMS,
                n_estimators=XGBOOST_FOLD_ROUNDS,
                early_stopping_rounds=XGBOOST_EARLY_STOP,
            )
            xgb.fit(
                X_train_xgb, y_train,
                eval_set=[(X_val_xgb, y_val)],
                verbose=False,
            )
        else:
            xgb = XGBClassifier(**XGBOOST_PARAMS)
            xgb.fit(X_train_xgb, y_train, verbose=False)
        models["xgboost"] = xgb
    except ImportError:
        logger.warning("xgboost not available, skipping XGB member")

    # LightGBM — native categorical support via named columns.
    # DataFrame with int team-ID columns + categorical_feature by NAME.
    try:
        from lightgbm import LGBMClassifier
        import pandas as pd
        num_cols_in_data = [c for c in FEATURE_COLS if c in train.columns]
        lgbm_cols = num_cols_in_data + TREE_CATEGORICAL_COLS
        X_train_lgbm = pd.DataFrame(X_train, columns=num_cols_in_data)
        for i, c in enumerate(TREE_CATEGORICAL_COLS):
            X_train_lgbm[c] = np.where(
                X_cat_train[:, i] < 0, UNK_TEAM_ID, X_cat_train[:, i]
            ).astype(int)
        lgbm = LGBMClassifier(**LIGHTGBM_PARAMS)
        if X_val is not None:
            X_val_lgbm = pd.DataFrame(X_val, columns=num_cols_in_data)
            for i, c in enumerate(TREE_CATEGORICAL_COLS):
                X_val_lgbm[c] = np.where(
                    X_cat_val[:, i] < 0, UNK_TEAM_ID, X_cat_val[:, i]
                ).astype(int)
            lgbm.fit(X_train_lgbm, y_train, eval_set=[(X_val_lgbm, y_val)],
                     categorical_feature=TREE_CATEGORICAL_COLS)
        else:
            lgbm.fit(X_train_lgbm, y_train,
                     categorical_feature=TREE_CATEGORICAL_COLS)
        models["lightgbm"] = lgbm
    except ImportError:
        logger.warning("lightgbm not available, skipping LGBM member")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train_scaled, y_train)
    models["logistic"] = lr
    models["scaler"] = scaler
    models["impute_median"] = impute_medians

    # Random Forest — bagged trees, decorrelated from boosting errors.
    # sklearn trees cannot consume NaN: use the train-median-imputed matrix.
    try:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        if RF_WITH_TEAM_IDS:
            rf.fit(X_train_lr_tree, y_train)
        else:
            rf.fit(X_train_lr, y_train)  # ablation: numeric only
        models["randomforest"] = rf
    except Exception as e:
        logger.warning("RandomForest member failed: %s", e)

    # MLP — small neural net with early stopping; diversity wildcard whose
    # weight is earned (or starved) by the adaptive blend.
    try:
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.01,
            early_stopping=True, validation_fraction=0.15,
            max_iter=300, random_state=RANDOM_SEED,
        )
        mlp.fit(X_train_scaled, y_train)
        models["mlp"] = mlp
    except Exception as e:
        logger.warning("MLP member failed: %s", e)

    # A fit-only refit has no honest holdout metric to report.
    if X_val is None:
        return models, {}

    # Weighted ensemble prediction (weights renormalized over trained members)
    weights = _member_weights(list(models.keys()))
    probs, wts = [], []
    for name, model in models.items():
        if name in ("scaler", "impute_median"):
            continue
        if name in ("logistic", "mlp"):
            Xuse = X_val_scaled
        elif name == "xgboost":
            Xuse = X_val_xgb  # DataFrame with pd.Categorical team IDs
        elif name == "randomforest":
            if RF_WITH_TEAM_IDS:
                Xuse = X_val_lr_tree
            else:
                Xuse = X_val_lr  # ablation: numeric only
        elif name == "lightgbm":
            Xuse = X_val_lgbm  # DataFrame with int team IDs + cat names
        else:
            Xuse = X_val
        probs.append(model.predict_proba(Xuse)[:, 1])
        wts.append(weights[name])

    ensemble_prob = np.average(probs, axis=0, weights=wts) if probs else np.full(len(y_val), 0.5)

    metrics = compute_metrics(y_val, ensemble_prob)
    return models, metrics


# ── Totals regression ───────────────────────────────────────────────────────

def train_totals_model(
    train: pd.DataFrame, val: pd.DataFrame
) -> dict[str, Any]:
    """Train XGBoost + LightGBM regression ensemble for total runs."""
    cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[cols].to_numpy(dtype=float)
    y_train = train["total_runs"].values.astype(float)
    X_val = val[cols].to_numpy(dtype=float)
    y_val = val["total_runs"].values.astype(float)

    models = {}

    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(**XGBOOST_REG_PARAMS)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models["xgboost_reg"] = xgb
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(**LIGHTGBM_REG_PARAMS)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models["lgbm_reg"] = lgbm
    except ImportError:
        pass

    # Predictions for metrics
    preds = []
    for name, model in models.items():
        preds.append(model.predict(X_val))

    if preds:
        ensemble_pred = np.mean(preds, axis=0)
        rmse = float(np.sqrt(np.mean((ensemble_pred - y_val) ** 2)))
        mae = float(np.mean(np.abs(ensemble_pred - y_val)))
    else:
        rmse = mae = float("nan")

    return {
        "models": models,
        "metrics": {"rmse": round(rmse, 4), "mae": round(mae, 4)},
    }


# ── Run-line classification ─────────────────────────────────────────────────

def train_run_line_model(
    train: pd.DataFrame, val: pd.DataFrame
) -> dict[str, Any]:
    """Train run-line cover probability classifier.

    Run-line cover: does the home team cover -1.5 run line?
    (i.e., win by 2+ runs)
    """
    train = train.copy()
    val = val.copy()
    train["run_line_cover"] = (train["home_win"] == 1) & (train.get("total_runs", 0) > 1)
    # Simplified: home covers if they win (since run_line is -1.5)
    train["run_line_cover"] = (train["home_win"] == 1).astype(float)
    val["run_line_cover"] = (val["home_win"] == 1).astype(float)

    cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[cols].to_numpy(dtype=float)
    y_train = train["run_line_cover"].values
    X_val = val[cols].to_numpy(dtype=float)
    y_val = val["run_line_cover"].values

    models = {}
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(**XGBOOST_PARAMS)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models["xgboost_rl"] = xgb
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(**LIGHTGBM_PARAMS)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models["lgbm_rl"] = lgbm
    except ImportError:
        pass

    # Predictions
    probs = []
    for name, model in models.items():
        probs.append(model.predict_proba(X_val)[:, 1])

    if probs:
        ensemble_prob = np.mean(probs, axis=0)
        metrics = compute_metrics(y_val, ensemble_prob)
    else:
        metrics = {"auc": 0.5, "brier": 0.25}

    return {"models": models, "metrics": metrics}


# ── Full walk-forward evaluation ────────────────────────────────────────────

def walk_forward_evaluate(
    games: pd.DataFrame,
    retrain_cadence_days: int = RETRAIN_CADENCE_DAYS,
    max_eval_folds: int = 0,
    force_retrain: bool = False,
    min_train_days: int = 0,
    min_val_games: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, float], pd.DataFrame]:
    """Run full walk-forward evaluation across all splits.

    Validation folds with fewer than ``min_val_games`` games are skipped
    (default MIN_VAL_FOLD_GAMES): tiny postseason/offseason-tail folds add
    high-variance metrics that pollute the pooled scores and the adaptive
    weights earned from them. Pass 0 to keep every fold (used by tests).

    Returns:
        (best_models, pooled_metrics, all_predictions)
    """
    if min_val_games is None:
        min_val_games = MIN_VAL_FOLD_GAMES

    # OOF scoring must start from the configured priors. Otherwise an earlier
    # run's adaptive weights can change the current run's fold predictions.
    _LAST_ADAPTIVE_WEIGHTS.clear()
    splits = walk_forward_splits(games, retrain_cadence_days, max_eval_folds, min_train_days)

    if not splits:
        logger.warning("No walk-forward splits generated; training on full data")
        # Fall back to train on everything
        splits = [{
            "train_games": games.dropna(subset=["home_win"]),
            "val_games": games.dropna(subset=["home_win"]).tail(min(50, len(games.dropna(subset=["home_win"])))),
            "fold_idx": 0,
            "val_start": games["game_date"].min(),
            "val_end": games["game_date"].max(),
        }]

    all_preds = []
    fold_metrics_list = []
    oof_members: dict[str, list[float]] = {}
    oof_y: list[float] = []
    # Blended raw probabilities and their PREQUENTIAL calibrated twins:
    # each fold's calibration comes from a Platt map fitted strictly on
    # PRIOR folds' OOF pairs, so every calibrated point stays honest.
    oof_blend: list[float] = []
    oof_blend_calibrated: list[float] = []

    for split in splits:
        train = split["train_games"]
        val = split["val_games"]

        if len(train) < 10 or len(val) < 5:
            continue
        if len(val) < min_val_games:
            logger.info(
                "Skipping fold %d [%s → %s]: only %d val games < %d minimum",
                split["fold_idx"], str(split["val_start"])[:10],
                str(split["val_end"])[:10], len(val), min_val_games,
            )
            continue

        try:
            ml_models, ml_metrics = train_moneyline_ensemble(train, val)
        except Exception as e:
            logger.warning("Fold %d moneyline training failed: %s", split["fold_idx"], e)
            continue

        logger.info(
            "Fold %d [%s → %s]: train=%d val=%d auc=%.4f brier=%.4f",
            split["fold_idx"],
            str(split["val_start"])[:10], str(split["val_end"])[:10],
            len(train), len(val), ml_metrics.get("auc", 0.5), ml_metrics.get("brier", 0.25),
        )

        # Weighted-blend prediction; keep each member's probabilities so we
        # can score candidates individually out of sample.
        ensemble_prob, member_probs, _wts = ensemble_predict(ml_models, val)
        y_val = val["home_win"].values.tolist()

        # Prequential calibration: fit on everything out-of-sample BEFORE
        # this fold, then transform this fold's predictions.
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(oof_y, oof_blend)
        fold_calibrated = apply_platt(ensemble_prob, fold_cal)

        oof_y.extend(y_val)
        oof_blend.extend(np.asarray(ensemble_prob, dtype=float).tolist())
        oof_blend_calibrated.extend(
            np.asarray(fold_calibrated, dtype=float).tolist()
        )
        for name, p in member_probs.items():
            oof_members.setdefault(name, []).extend(p.tolist())

        val_pred = val.copy()
        val_pred["home_win_prob_model"] = ensemble_prob
        val_pred["home_win_prob_model_calibrated"] = np.round(fold_calibrated, 4)
        val_pred["fold_idx"] = split["fold_idx"]
        all_preds.append(val_pred)
        fold_metrics_list.append(ml_metrics)

    # Pool metrics across folds
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        pooled = compute_metrics(combined["home_win"].values, combined["home_win_prob_model"].values)
    else:
        combined = pd.DataFrame()
        pooled = {"auc": 0.5, "brier": 0.25, "logloss": 0.69, "ece": 0.0}

    # Post-hoc calibration: fit the shipped Platt map on ALL pooled OOF
    # pairs, and score it against the raw blend using the prequential
    # calibrated predictions (fold k corrected only by folds < k — never
    # self-calibrated). Raw headline metrics stay untouched; calibrated
    # twins ride alongside so dashboards can show both.
    y_oof_all = np.asarray(oof_y, dtype=float) if oof_y else np.empty(0)
    p_raw_all = np.asarray(oof_blend, dtype=float) if oof_blend else np.empty(0)
    p_cal_prequential = (
        np.asarray(oof_blend_calibrated, dtype=float)
        if oof_blend_calibrated else np.empty(0)
    )
    final_calibrator = fit_platt(y_oof_all, p_raw_all)
    global _LAST_CALIBRATOR
    _LAST_CALIBRATOR = final_calibrator
    if p_cal_prequential.size == len(y_oof_all) and len(y_oof_all) > 0:
        m_cal = compute_metrics(y_oof_all, apply_platt(p_cal_prequential, final_calibrator))
        pooled["brier_calibrated"] = m_cal["brier"]
        pooled["logloss_calibrated"] = m_cal["logloss"]
        pooled["ece_calibrated"] = m_cal["ece"]
        logger.info(
            "Calibration (OOF, prequential): ECE %.4f → %.4f, log-loss %.4f → %.4f",
            pooled.get("ece", 0.0), m_cal["ece"],
            pooled.get("logloss", 0.0), m_cal["logloss"],
        )

    # Fit the deployed bundle on every decided game. The walk-forward folds
    # remain the only source of honest OOF metrics; no final validation holdout
    # is needed once evaluation is complete.
    full_train = games.dropna(subset=["home_win"])
    if len(full_train) >= 20:
        try:
            best_models, _ = train_moneyline_ensemble(full_train)
        except Exception:
            best_models = {}
    else:
        best_models = {}

    # Adaptive blend weights: earned from pooled OOF member scores
    # (AUC by default — see ADAPTIVE_WEIGHT_METRIC). These replace the
    # static ENSEMBLE_WEIGHTS priors for prediction blending (see
    # _member_weights) until the next evaluation, so the ensemble
    # self-corrects as features change.
    y_oof = np.asarray(oof_y, dtype=float)
    adaptive = compute_adaptive_weights(oof_members, y_oof)
    _LAST_ADAPTIVE_WEIGHTS.clear()
    _LAST_ADAPTIVE_WEIGHTS.update(adaptive)
    if adaptive:
        logger.info(
            "Adaptive ensemble weights: %s",
            {k: f"{v:.1%}" for k, v in sorted(adaptive.items())},
        )

    # Candidate-model report: every candidate that ever trained, its blend
    # weight in the deployed ensemble (adaptive when available; absent or
    # failed candidates report 0%), and its own pooled out-of-fold
    # AUC/Brier/LogLoss across all evaluation folds.
    final_members = {
        n for n in (best_models or {})
        if n not in ("scaler", "impute_median")
    }
    raw_w = {
        n: float(adaptive.get(n, 0.0)) if adaptive
        else float(ENSEMBLE_WEIGHTS.get(n, 0.0))
        for n in final_members
    }
    w_total = sum(raw_w.values()) or 1.0
    # Every configured candidate is reported — even ones that failed to train
    # this run (weight 0%, metrics null) — per the ensemble transparency rule.
    roster = list(dict.fromkeys(
        list(ENSEMBLE_WEIGHTS.keys()) + list(oof_members.keys()) + sorted(final_members)
    ))
    _LAST_ENSEMBLE_INFO.clear()
    for name in roster:
        entry: dict[str, Any] = {
            "name": name,
            "weight": round(raw_w[name] / w_total, 4) if name in final_members else 0.0,
        }
        preds = oof_members.get(name)
        if preds and len(preds) == len(y_oof) and len(y_oof) > 0:
            m = compute_metrics(y_oof, np.asarray(preds, dtype=float))
            entry.update({
                "auc": m.get("auc"),
                "brier": m.get("brier"),
                "logloss": m.get("logloss"),
                "n_eval": int(len(y_oof)),
            })
        else:
            entry.update({"auc": None, "brier": None, "logloss": None, "n_eval": 0})
        _LAST_ENSEMBLE_INFO.append(entry)

    return best_models, pooled, combined


# ── Persistence ─────────────────────────────────────────────────────────────

def persist_ensemble(
    models: dict[str, Any],
    metrics: dict[str, float],
    version: str = "v3.2.1",
    data_cutoff: Optional[str] = None,
) -> Path:
    """Save ensemble models and metadata to joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        VERSION_KEY: version,
        TRAINED_AT_KEY: datetime.now().isoformat(),
        DATA_CUTOFF_KEY: data_cutoff or datetime.now().strftime(DATE_FMT),
    }

    bundle = {
        "models": models,
        "metrics": metrics,
        "metadata": metadata,
        # Earned blend weights ride with the models so a cached-model run
        # predicts with exactly the weighting that was validated.
        "adaptive_weights": dict(_LAST_ADAPTIVE_WEIGHTS),
        # Post-hoc Platt calibrator fitted on pooled OOF; applied by
        # predict_games so published probabilities are calibrated.
        "calibrator": get_last_calibrator(),
    }

    path = MODELS_DIR / ENSEMBLE_FILE
    joblib.dump(bundle, path)
    logger.info("Ensemble persisted to %s", path)
    return path


def load_ensemble(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Load a persisted ensemble bundle."""
    path = path or (MODELS_DIR / ENSEMBLE_FILE)
    if not path.exists():
        return None
    return joblib.load(path)


def should_retrain(last_trained: Optional[datetime], cadence_days: int = RETRAIN_CADENCE_DAYS) -> bool:
    """Determine if retraining is needed based on cadence."""
    if last_trained is None:
        return True
    return (datetime.now() - last_trained).days >= cadence_days


def predict_games(
    models: dict[str, Any],
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Apply ensemble models to predict on a set of games.

    Adds columns: home_win_prob_model, away_win_prob_model, model_pick, edge_home, edge_away
    """
    if not models:
        return games

    blend, _members, _wts = ensemble_predict(models, games)
    # Post-hoc recalibration: correct blended probabilities before they
    # feed picks/edges. Identity (no-op) when no calibrator is loaded.
    calibrator = get_last_calibrator()
    if not is_identity(calibrator):
        blend = apply_platt(blend, calibrator)
    games["home_win_prob_model"] = np.round(blend, 4)

    games["away_win_prob_model"] = 1 - games["home_win_prob_model"]

    # Model pick
    games["model_pick"] = np.where(
        games["home_win_prob_model"] >= 0.5, games["home_team"], games["away_team"]
    )

    # Edge: model_prob - fair_market_prob (vig removed via two-way normalization)
    if "moneyline_home" in games.columns and games["moneyline_home"].notna().any():
        ml_home = games["moneyline_home"].fillna(-110).values
        ml_away = games["moneyline_away"].fillna(-110).values
        fair_home = np.where(ml_home < 0, -ml_home / (-ml_home + 100), 100 / (ml_home + 100))
        fair_away = np.where(ml_away < 0, -ml_away / (-ml_away + 100), 100 / (ml_away + 100))
        # Normalize (remove vig)
        total = fair_home + fair_away
        fair_home_norm = fair_home / total
        fair_away_norm = fair_away / total
        games["edge_home"] = np.round(games["home_win_prob_model"].values - fair_home_norm, 4)
        games["edge_away"] = np.round(games["away_win_prob_model"].values - fair_away_norm, 4)
    else:
        games["edge_home"] = 0.0
        games["edge_away"] = 0.0

    return games


def update_model_history(
    metrics: dict[str, float],
    version: str,
    notes: str = "",
) -> None:
    """Append a row to model_history.json for the Model Monitor page.

    One row per calendar day: a re-run on the same day REPLACES that day's
    row instead of appending, so the Version History table reflects real
    retrains rather than every debugging rerun.
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = DATA_DELIVERY_DIR / "model_history.json"

    history = []
    if history_path.exists():
        with open(history_path) as f:
            try:
                history = json.load(f)
            except ValueError:
                history = []

    today = datetime.now().strftime("%Y-%m-%d")
    history = [row for row in history if row.get("date") != today]
    history.append({
        "version": version,
        "date": today,
        "auc": metrics.get("auc", 0),
        "brier": metrics.get("brier", 0),
        "logloss": metrics.get("logloss", 0),
        "ece": metrics.get("ece", 0),
        **({"ece_calibrated": metrics["ece_calibrated"]}
           if "ece_calibrated" in metrics else {}),
        "notes": notes,
    })
    history.sort(key=lambda row: str(row.get("date", "")))

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
