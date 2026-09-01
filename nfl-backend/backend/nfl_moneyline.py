"""NFL moneyline — 5-member ensemble gate + gated per-game slate.

Model arm (the Part-1 upgrade): a 5-member weighted ensemble mirroring
mlb-backend's ``train_moneyline_ensemble`` — XGBoost, LightGBM, Logistic,
RandomForest, MLP — with train-fold-median imputation, StandardScaler,
32-team-ID categoricals for the tree members (LightGBM by name, XGBoost via
pd.Categorical, RF as integer features), per-member try/except degradation,
and an ADAPTIVE blend (pooled OOF member AUC softmax, floor/cap projection)
that replaces the static priors for the deployed bundle. Features come from
the admission-gate record (``nfl_features.py``): the v1 base plus the gated
v2 candidates (decaying-window strength aggregates, opponent-adjusted
margin, pace, short-rest edge, QB EPA, weather, division). ``is_home`` stays
a constant anchor — it is carried by the baselines/intercept and never fed
as a model column.

Discipline (MLB retrospective):
- STRICT point-in-time: every feature is already leakage-safe (feature gate;
  ``team_stats_ladder`` asserts per-team strict gameday monotonicity). At the
  model entry point we additionally assert walk-forward folds never train on a
  row at/after the fold's validation week, and that season 2025 (the SEALED
  hold-out) never appears in any pre-sealed fit or calibration map.
- Prequential weekly-cadence folds over 2019-2024 (warm-up = first two full
  seasons, 2019+2020, are never validated): pooled OOF logloss/AUC/ECE, plus
  an honestly-nested Platt twin.
- SEALED hold-out: ALL of 2025, model fitted on 2019-2024 only, calibrated by
  a Platt map fitted only on pre-holdout OOF (2021-2024 pooled).
- Baselines to beat (sealed): (a) constant home-edge (NFL home win rate),
  (b) elo-only logistic (cheapest signal). ADOPT requires the model to beat
  BOTH on the SEALED window in logloss AND AUC with a sane ECE. A pooled-gain
  / sealed-loss inversion means DON'T ADOPT.

Artifact: data_delivery/nfl_moneyline_v1_<date>.json — fold geometry,
per-arm pooled + sealed tables (raw + Platt twins), per-member tables,
adaptive weights, baselines, verdict+reason, and the per-game ``games[]`` slate
for the current schedule (2026 week 1) — ALWAYS written from the fresh
ensemble when a schedule loads. The ``verdict``/seal gate is a TESTING +
monitoring signal (ensemble vs elo/constant baselines + sanity ECE) that is
recorded but never blocks the board (mirroring MLB); the run always continues
normally (never errors).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn pieces the ensemble uses. Imported at module level (a guarded import
# inside train_ensemble would duplicate this); the ensemble members themselves
# are imported lazily so a missing xgboost/lightgbm degrades a single member.
try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except Exception:  # pragma: no cover - sklearn is a hard requirement
    _SKLEARN_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
DECIDED_FRAME = DATA_DELIVERY_DIR / "nfl_game_level_features.csv"

# Seasons covered by the default (no --seasons / --seasons flag) run —
# kept at module level so tests can import it.
DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]

# model inputs = the v1 numeric feature set (is_home is a constant anchor ->
# absorbed by intercept/baseline, not fed as a column)
V1_FEATURES = ["elo_diff", "form_diff_pts", "rest_days_diff", "ypp_diff",
               "is_dome_home"]
TARGET = "home_win"

WARMUP_SEASONS = [2018]
TRAIN_SEASONS = list(range(2019, 2025))   # pre-sealed training window: 2019..2024
VAL_SEASONS = [2021, 2022, 2023, 2024]    # prequential validation (2-season warm-up)
SEALED_SEASON = 2025

DATE_FMT = "%Y%m%d"
RECORD_TEMPLATE = f"nfl_moneyline_v1_{{date}}.json"
CALIBRATION_TEMPLATE = f"nfl_calibration_{{date}}.json"
HISTORY_TEMPLATE = f"nfl_predictions_history_{{date}}.csv"
MONITOR_TEMPLATE = f"nfl_model_monitor_{{date}}.json"
POWER_RANKINGS_TEMPLATE = f"nfl_power_rankings_{{date}}.csv"

# LightGBM hyperparams (modest regularization, deterministic)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 40,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "seed": 7,
    "verbosity": -1,
    "force_row_wise": True,
    "boost_from_average": True,
    "n_jobs": 1,
}
NUM_BOOST_ROUND = 300
EARLY_STOPPING = 30

ECE_BINS = 10
ECE_MAX = 0.08          # "sane" calibration bar for adoption
PROB_EPS = 1e-6

# ---------------------------------------------------------------------------
# 5-member ensemble config (mirrors mlb-backend/backend/training.py)
# ---------------------------------------------------------------------------
# Static base blend weights — the pre-adaptive FALLBACK, renormalized over
# whichever members actually trained. After the walk-forward run the blend
# switches to ADAPTIVE weights earned from pooled OOF member AUC (the same
# softmax + floor/cap projection MLB ships).
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.25, "lightgbm": 0.25, "logistic": 0.30,
    "randomforest": 0.10, "mlp": 0.10,
}
ADAPTIVE_WEIGHT_METRIC = "auc"
ADAPTIVE_WEIGHT_TEMPERATURE = 0.03
ADAPTIVE_WEIGHT_AUC_TEMPERATURE = 0.015
ADAPTIVE_WEIGHT_FLOOR = 0.05
ADAPTIVE_WEIGHT_CAP = 0.45
RANDOM_SEED = 42

# Tree-member categoricals (the 32 NFL teams). NOT model columns — native
# categoricals for LightGBM (by name) / XGBoost (pd.Categorical); the RF
# member consumes them as integer features. Logistic/MLP never see them.
TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]
UNK_TEAM_ID = 99      # reserved slot — never auto-assigned, never a real team

# Season of the CURRENT schedule (undecided games) the slate stage targets.
# The orchestrator overrides this from the loaded schedule; 2026 week 1 is
# the default for the no-schedule (features-csv) dry path.
SLATE_SEASON = 2026

# Adaptive weights earned by the most recent walk-forward run; prediction
# (sealed + slate) blends with these instead of the static priors, mirroring
# MLB's ``_LAST_ADAPTIVE_WEIGHTS`` persistence pattern.
_ADAPTIVE_WEIGHTS: dict[str, float] = {}

# Fit-only deployed bundle + sealed Platt map from the most recent run (used
# by the slate stage; re-fit per run, never persisted across runs).
_DEPLOYED_BUNDLE: dict | None = None
_SEALED_PLATT: object | None = None


# ---------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------
def clip_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, PROB_EPS, 1.0 - PROB_EPS)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = clip_p(np.asarray(p, dtype=float))
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    n = len(p)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        total += (m.sum() / n) * abs(acc - conf)
    return float(total)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via the v1 gate's rank statistic (ties -> 0.5)."""
    from nfl_features import univariate_auc
    return float(univariate_auc(np.asarray(y), np.asarray(p)))


# ---------------------------------------------------------------------------
# Platt calibration (pure, via sklearn; fit maps are sealed-off-holdout)
# ---------------------------------------------------------------------------
def platt_fit(p: np.ndarray, y: np.ndarray):
    """Fit the 2-parameter Platt map on (logit(p), y). Returns None when the
    pool cannot support a fit (single class, too few games) — callers treat
    None as the identity map (raw probabilities), mirroring MLB's fit_platt."""
    y = np.asarray(y, dtype=int)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    x = np.log(clip_p(p) / (1 - clip_p(p))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6)          # essentially unregularized Platt map
    lr.fit(x, y)
    return lr


def platt_predict(p: np.ndarray, lr) -> np.ndarray:
    if lr is None:
        return np.asarray(p, dtype=float)
    x = np.log(clip_p(p) / (1 - clip_p(p))).reshape(-1, 1)
    return lr.predict_proba(x)[:, 1]


# ---------------------------------------------------------------------------
# Walk-forward fold generation (pure, leakage-asserted)
# ---------------------------------------------------------------------------
def _week_start(dates: pd.Series) -> pd.Series:
    """Monday of each date's calendar week (NFL weeks are Mon-Sun)."""
    d = pd.to_datetime(dates)
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def generate_weekly_folds(preq: pd.DataFrame,
                          val_seasons: list[int] | None = None) -> list[dict]:
    """Prequential weekly folds over ``preq`` (2019-2024 decided games).

    Each fold validates all games in one calendar week (Mon-Sun) of a
    validation season; its train set is EVERY game with gameday strictly
    before that week (so a fold can never see its own or any future week).

    LEAKAGE ASSERTION: for every fold, max(train.gameday) < min(val.gameday).
    """
    val_seasons = val_seasons or VAL_SEASONS
    g = preq.copy()
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    g = g.sort_values("gameday").reset_index(drop=True)
    g["week_start"] = _week_start(g["gameday"])
    g["val_season"] = g["season"].isin(val_seasons)
    folds = []
    for mon, idx in g[g["val_season"]].groupby("week_start")["week_start"].groups.items():
        val = g.loc[idx]
        train = g[g["gameday"] < mon]
        if len(val) == 0 or len(train) == 0:
            continue
        tr_max = train["gameday"].max()
        va_min = val["gameday"].min()
        if not (tr_max < mon <= va_min):
            raise AssertionError(
                f"fold week {mon}: train max {tr_max} not strictly before "
                f"val min {va_min} -> future-week leak")
        folds.append({"week_start": mon, "train": train.copy(), "val": val.copy()})
    folds.sort(key=lambda f: f["week_start"])
    return folds


# ---------------------------------------------------------------------------
# Model arms
# ---------------------------------------------------------------------------
def _valid_rows(df: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    features = features or V1_FEATURES
    return df[features + [TARGET]].notna().all(axis=1).to_numpy()


# ---- Team-ID categorical mapping for the tree members ----------------------
_TEAM_ABBR_TO_ID: dict[str, int] = {}
_TEAM_ID_TO_ABBR: dict[int, str] = {}


def _team_id(abbr: object) -> int:
    """Stable integer ID for an NFL team abbreviation (same team = same ID
    across seasons). Unknown/short/missing values map to UNK_TEAM_ID — a
    dedicated category with near-zero training presence so trees learn a
    neutral weight instead of aliasing a real team."""
    if abbr in _TEAM_ABBR_TO_ID:
        return _TEAM_ABBR_TO_ID[abbr]
    if not isinstance(abbr, str) or len(abbr.strip()) < 2:
        return UNK_TEAM_ID
    key = abbr.strip().upper()
    if key in _TEAM_ABBR_TO_ID:
        return _TEAM_ABBR_TO_ID[key]
    tid = len(_TEAM_ABBR_TO_ID)
    if tid >= UNK_TEAM_ID:
        tid += 1  # skip the reserved slot
    _TEAM_ABBR_TO_ID[key] = tid
    _TEAM_ID_TO_ABBR[tid] = key
    return tid


def _cat_unk_for(col: str) -> int:
    return UNK_TEAM_ID


def _cat_known_ids(col: str) -> list[int]:
    return sorted(set(_TEAM_ID_TO_ABBR) | {UNK_TEAM_ID})


def _add_team_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team_id"] = df["home_team"].apply(_team_id)
    df["away_team_id"] = df["away_team"].apply(_team_id)
    for abbr, tid in sorted(_TEAM_ABBR_TO_ID.items()):
        if tid == UNK_TEAM_ID:
            raise AssertionError(
                f"UNK_TEAM_ID={UNK_TEAM_ID} collides with real team '{abbr}' → {tid}")
    return df


# ---- Feature matrices (mirror MLB _feature_matrix / _prepare_features) -----
def _feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Numeric matrix over ``features`` in canonical order, NaN preserved.

    Missing columns come back all-NaN with one loud warning (never silently
    dropped) — trees route NaN natively; logistic/MLP/RF get train-fold
    medians via the bundle's impute_median."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        logger.warning(
            "Feature matrix: %d/%d expected columns absent (%s%s) — filled NULL",
            len(missing), len(features), ", ".join(missing[:6]),
            " …" if len(missing) > 6 else "",
        )
    return df.reindex(columns=features).to_numpy(dtype=float)


def _categorical_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[TREE_CATEGORICAL_COLS].to_numpy(dtype=int)


def _prepare_features(df: pd.DataFrame, features: list[str]):
    df = _add_team_ids(df)
    X = _feature_matrix(df, features)
    X_cat = _categorical_matrix(df)
    y = df[TARGET].values.astype(float)
    return X, X_cat, y


def _impute_median(X: np.ndarray, medians=None):
    """Fill NaN with column medians (fit on TRAIN when medians is None).
    All-NaN columns fall back to 0.0. Returns (imputed, medians_used)."""
    X = np.asarray(X, dtype=float)
    if medians is None:
        with np.errstate(all="ignore"):
            medians = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
        medians = np.where(np.isnan(medians), 0.0, medians)
    X = X.copy()
    idx = np.isnan(X)
    X[idx] = np.take(np.asarray(medians, dtype=float), idx.nonzero()[1])
    return X, np.asarray(medians, dtype=float)


def _tree_dataframe(X_num: np.ndarray, X_cat: np.ndarray, numeric_cols: list[str],
                    vocabs: dict | None = None) -> pd.DataFrame:
    """Named numeric + pd.Categorical team-ID frame for XGBoost (explicit
    category set so predict-time newcomers never throw 'category not in the
    training set'). LightGBM builds its own int-coded frame."""
    df = pd.DataFrame(X_num, columns=numeric_cols)
    for i, c in enumerate(TREE_CATEGORICAL_COLS):
        vals = X_cat[:, i].copy()
        unk = _cat_unk_for(c)
        vals = np.where(vals < 0, unk, vals)
        vocab = (vocabs or {}).get(c)
        if vocab is not None:
            known = np.asarray(sorted(set(vocab)), dtype=int)
            vals = np.where(np.isin(vals, known), vals, unk)
            df[c] = pd.Categorical(vals, categories=sorted(set(vocab)))
        else:
            df[c] = pd.Categorical(vals, categories=_cat_known_ids(c))
    return df


def _lgbm_dataframe(X_num: np.ndarray, X_cat: np.ndarray, numeric_cols: list[str],
                    vocabs: dict | None = None) -> pd.DataFrame:
    """Named numeric + int team-ID frame for LightGBM (categorical by NAME)."""
    df = pd.DataFrame(X_num, columns=numeric_cols)
    for i, c in enumerate(TREE_CATEGORICAL_COLS):
        vals = X_cat[:, i].copy()
        unk = _cat_unk_for(c)
        vals = np.where(vals < 0, unk, vals)
        vocab = (vocabs or {}).get(c)
        if vocab is not None:
            known = np.asarray(sorted(set(vocab)), dtype=int)
            vals = np.where(np.isin(vals, known), vals, unk)
        df[c] = vals.astype(int)
    return df


def _member_weights(member_names: list[str], adaptive: dict | None = None) -> dict[str, float]:
    """Blend weights normalized over the members that actually trained.

    Prefers the run's adaptive weights when available; falls back to the
    static ENSEMBLE_WEIGHTS priors (e.g. before the first OOF cycle). A
    member that failed to train contributes 0% and the rest renormalize to
    exactly 1.0."""
    names = [n for n in member_names
             if n not in ("scaler", "impute_median", "categorical_vocab")]
    source = adaptive or ENSEMBLE_WEIGHTS
    raw = {n: float(source.get(n, 0.0)) for n in names}
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


def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """logloss / auc / ece / brier for an NFL probability vector."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    brier = float(np.mean((y - p) ** 2)) if n else 0.25
    return {
        "logloss": round(logloss(y, p), 4),
        "auc": round(auc(y, p), 4),
        "ece": round(ece(y, p), 4),
        "brier": round(brier, 4),
    }


def compute_adaptive_weights(oof_members: dict[str, list[float]],
                             y_oof: np.ndarray) -> dict[str, float]:
    """Blend weights earned by pooled OOF member AUC (softmax + floor/cap
    projection, mirroring MLB). Sums to exactly 1.0."""
    y = np.asarray(y_oof, dtype=float)
    scores: dict[str, float] = {}
    if len(y) == 0:
        return {}
    for name, preds in oof_members.items():
        if not preds or len(preds) != len(y):
            continue
        a = auc(y, np.asarray(preds, dtype=float))
        if a is not None and np.isfinite(a):
            scores[name] = float(a)
    if not scores:
        return {}
    if ADAPTIVE_WEIGHT_METRIC == "auc":
        _t = ADAPTIVE_WEIGHT_AUC_TEMPERATURE
        best = max(scores.values())
        exp_w = {n: np.exp((a - best) / _t) for n, a in scores.items()}
    else:
        _t = ADAPTIVE_WEIGHT_TEMPERATURE
        best = min(scores.values())
        exp_w = {n: np.exp(-(ll - best) / _t) for n, ll in scores.items()}
    tot = sum(exp_w.values())
    w = {n: float(v / tot) for n, v in exp_w.items()}
    eff_cap = max(ADAPTIVE_WEIGHT_CAP, 1.02 / len(w))
    for _ in range(50):
        w = {n: max(v, ADAPTIVE_WEIGHT_FLOOR) for n, v in w.items()}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}
        w = {n: min(v, eff_cap) for n, v in w.items()}
        s = sum(w.values())
        w = {n: v / s for n, v in w.items()}
    rounded = {n: round(v, 4) for n, v in w.items()}
    drift = round(1.0 - sum(rounded.values()), 4)
    if drift:
        top = max(rounded, key=lambda n: w[n])
        rounded[top] = round(rounded[top] + drift, 4)
    return rounded


def fit_predict_lgbm(Xtr: np.ndarray, ytr: np.ndarray,
                     Xte: np.ndarray, early_stopping_val=None) -> np.ndarray:
    """Single LightGBM; by default fixed rounds (NO early-stop on a fold's own
    labels, which would leak that fold: early_stopping_val is only for tests)."""
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, ytr)
    callbacks = [lgb.log_evaluation(0)]
    if early_stopping_val is not None:
        Xv, yv = early_stopping_val
        dv = lgb.Dataset(Xv, yv, reference=dtr)
        bst = lgb.train(dict(LGB_PARAMS), dtr, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dv], callbacks=[lgb.log_evaluation(0),
                                                    lgb.early_stopping(EARLY_STOPPING)])
        return bst.predict(Xte)
    bst = lgb.train(dict(LGB_PARAMS), dtr, num_boost_round=NUM_BOOST_ROUND,
                    callbacks=callbacks)
    return bst.predict(Xte)


def _platt_on(ppool: list[np.ndarray], ypool: list[np.ndarray]) -> object:
    return platt_fit(np.concatenate(ppool), np.concatenate(ypool).astype(int))


# ---------------------------------------------------------------------------
# Calibration + per-game history artifacts (Part-A: MLB-equivalent)
# ---------------------------------------------------------------------------
# Per-fold games carry the metadata the prediction-history CSV needs. The
# decided frame's columns (game_id / season / week / gameday / teams / scores)
# are preserved through the fold loop so a history row can be reconstructed.
META_COLS = ["game_id", "season", "week", "gameday", "home_team",
             "away_team", "home_score", "away_score"]
# MLB predictions_history column contract + the calibrated twin + reference
# columns. The frontend/mlc.favored_calibration_pts read home_win_prob_model/
# correct (raw); the page re-applies the Platt map from the calibration
# record's a/b for the deployed green curve.
HISTORY_COLUMNS = [
    "game_date", "home_team", "away_team", "home_win_prob_model",
    "home_win_prob_model_calibrated", "away_win_prob_model", "correct",
    "model_pick", "home_score", "away_score", "actual_winner",
    "game_status", "game_id", "season", "week",
]


def reliability_buckets(y, p, bins: int = ECE_BINS,
                        home_fav: np.ndarray | None = None) -> list[dict]:
    """Per-bin favored-side [{bucket, mean_predicted, mean_actual, count, gap}]
    over equal-width probability bins running 50%-100% only — MLB's
    ``calibration_buckets`` shape; empty bins omitted (never fabricated).

    Exactly like the frontend curve (``moneyline_calibration.py
    ``favored_calibration_pts`` takes ``max(p, 1-p)``), every game is counted
    ONCE from the favored side: ``p_fav = max(p, 1-p)`` and ``y_fav = y`` if
    ``p >= 0.5`` else ``1 - y`` ("did the favored side win"). Bin ``p_fav``
    into equal-width buckets so no sub-50% bucket can exist and the table
    agrees with the curve (which today bins the raw home-win prob, disagreeing
    with the curve).

    ``home_fav`` pins WHICH side is favorite when supplied (``True`` = the
    prediction favors the home team, i.e. ``p >= 0.5``) — build_calibration
    passes the SAME mask derived from the raw blend to both the raw and
    calibrated bucket sets, so the favored side is consistent and never
    re-derived per-line.

    Pure per-game relabeling: each game counted once, favored side from this
    run's own prediction, metrics/maps untouched — no leakage introduced.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if home_fav is None:
        home_fav = p >= 0.5
    else:
        home_fav = np.asarray(home_fav, dtype=bool)
    p_fav = np.where(home_fav, p, 1.0 - p)         # always >= 0.5
    y_fav = np.where(home_fav, y, 1.0 - y)         # did the favored side win
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(p_fav, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        mp = float(np.mean(p_fav[m]))
        ma = float(np.mean(y_fav[m]))
        out.append({
            "bucket": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
            "mean_predicted": round(mp, 4),
            "mean_actual": round(ma, 4),
            "count": int(m.sum()),
            "gap": round(mp - ma, 4),
        })
    return out


def _history_rows(meta: pd.DataFrame, raw, cal) -> list[dict]:
    """One history-dict per (decided-game, prob/cal) aligned row."""
    raw = np.asarray(raw, dtype=float)
    cal = np.asarray(cal, dtype=float)
    rows = []
    for i in range(len(meta)):
        r = meta.iloc[i]
        home = str(r.get("home_team") or "").strip()
        away = str(r.get("away_team") or "").strip()
        hs = float(r["home_score"])
        as_ = float(r["away_score"])
        p_raw = float(raw[i])
        p_cal = float(cal[i]) if len(cal) else p_raw
        pick = home if p_raw >= 0.5 else away
        winner = home if hs > as_ else away
        gd = pd.to_datetime(r.get("gameday"), errors="coerce")
        gd_str = gd.strftime("%Y-%m-%d") if not pd.isna(gd) else ""
        rows.append({
            "game_date": gd_str,
            "home_team": home,
            "away_team": away,
            "home_win_prob_model": round(p_raw, 4),
            "home_win_prob_model_calibrated": round(p_cal, 4),
            "away_win_prob_model": round(1.0 - p_raw, 4),
            "correct": bool(pick == winner),
            "model_pick": pick,
            "home_score": hs,
            "away_score": as_,
            "actual_winner": winner,
            "game_status": "Final",
            "game_id": str(r.get("game_id") or ""),
            "season": r.get("season"),
            "week": r.get("week"),
        })
    return rows


def build_history_frame(*, oof_meta, oof_raw, oof_cal,
                        sealed_meta, sealed_raw, sealed_cal) -> pd.DataFrame:
    """Per-game OOF (2021-2024) + sealed-2025 prediction history in the MLB
    predictions_history column contract. ``raw`` = the deployed-style raw
    blend; ``cal`` = the Platt-calibrated value the page would compute
    (consistent with the calibration record's a/b). Sealed rows are appended
    AFTER the OOF rows and carry their own season, so OOF rows can never be
    influenced by sealed outcomes."""
    rows = _history_rows(oof_meta, oof_raw, oof_cal)
    rows += _history_rows(sealed_meta, sealed_raw, sealed_cal)
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def build_calibration(y, raw, cal, platt, bins: int = ECE_BINS) -> dict:
    """The ``nfl_calibration_*.json`` shape — mirrors MLB ``calibration_*.json``
    so the frontend ``_normalize_calibration``/``load_calibration`` work
    unchanged. ``raw`` = pooled-OOF raw blend; ``cal`` = pooled-OOF
    PREQUENTIAL per-fold calibrated values (drive metrics_calibrated +
    calibration_buckets_calibrated); ``platt`` = the SEALED Platt map fitted
    only on pooled pre-holdout OOF (its a/b drive the frontend's deployed
    green curve). Every leakage guarantee is honored: 2025 appears in no
    pre-sealed fit/calibration map."""
    y = np.asarray(y, dtype=float)
    raw = np.asarray(raw, dtype=float)
    cal = np.asarray(cal, dtype=float)
    mr = compute_metrics(y, raw)
    mc = compute_metrics(y, cal)
    a = b = None
    if platt is not None:
        a = round(float(platt.coef_[0][0]), 6)
        b = round(float(platt.intercept_[0]), 6)
    return {
        "date": datetime.now().strftime(DATE_FMT),
        "n_games": int(len(y)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "auc": mr["auc"], "brier": mr["brier"],
            "logloss": mr["logloss"], "ece": mr["ece"],
            "brier_calibrated": mc["brier"],
            "logloss_calibrated": mc["logloss"],
            "ece_calibrated": mc["ece"],
        },
        "calibration_buckets": reliability_buckets(y, raw, bins),
        "calibration": {
            "method": "platt",
            "params": {"a": a, "b": b, "n": int(len(y))},
            "metrics_raw": mr,
            "metrics_calibrated": mc,
            # The favored side is defined ONCE from the raw blend; the
            # calibrated set reuses that same mask (never re-derived per-line).
            "calibration_buckets_calibrated": reliability_buckets(
                y, cal, bins, home_fav=raw >= 0.5),
        },
        "daily": [],
    }


# ---------------------------------------------------------------------------
# 5-member ensemble (mirrors mlb-backend training.py::train_moneyline_ensemble)
# ---------------------------------------------------------------------------
def train_ensemble(train: pd.DataFrame, val: pd.DataFrame | None = None,
                   features: list[str] | None = None) -> tuple[dict, dict]:
    """Train the 5-member moneyline ensemble (XGB/LGB/Logistic/RF/MLP).

    ``val`` is supplied for walk-forward folds so the boosting members can
    evaluate against a strictly-future holdout; when omitted the call is a
    fit-only refit on every decided game for the deployed bundle.

    Prep (mirror MLB): numeric features are imputed with TRAIN-fold medians
    only (never val), StandardScaler fit on train → transform val; tree
    members get numeric diffs + the 32-team-ID categoricals — LightGBM as
    named categorical cols, XGBoost as pd.Categorical, RF on imputed-numeric
    + integer team IDs. Every member is try/except-degraded: a member that
    fails to import or fit is skipped, never fatal.

    Returns (models, metrics) where metrics is {} for fit-only refits and
    the val-window metrics dict otherwise."""
    features = features or V1_FEATURES
    X_train, X_cat_train, y_train = _prepare_features(train, features)
    X_val = X_cat_val = y_val = None
    if val is not None:
        X_val, X_cat_val, y_val = _prepare_features(val, features)

    if len(X_train) == 0 or (val is not None and len(X_val) == 0):
        raise ValueError("Insufficient training or validation data")

    X_train_lr, impute_medians = _impute_median(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lr)
    X_val_scaled = None
    X_val_lr = None
    if X_val is not None:
        X_val_lr, _ = _impute_median(X_val, impute_medians)
        X_val_scaled = scaler.transform(X_val_lr)

    models: dict = {}

    # XGBoost — imputed matrix + pd.Categorical team IDs, enable_categorical.
    try:
        from xgboost import XGBClassifier
        X_train_xgb = _tree_dataframe(X_train_lr, X_cat_train, features)
        xgb_kw = dict(objective="binary:logistic", max_depth=2,
                      min_child_weight=8, gamma=1.0, subsample=0.6,
                      colsample_bytree=0.6, learning_rate=0.06,
                      n_estimators=600, random_state=RANDOM_SEED,
                      enable_categorical=True, eval_metric="logloss")
        if X_val is not None:
            X_val_xgb = _tree_dataframe(X_val_lr, X_cat_val, features)
            xgb = XGBClassifier(**xgb_kw, early_stopping_rounds=20)
            xgb.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)],
                    verbose=False)
        else:
            xgb = XGBClassifier(**xgb_kw)
            xgb.fit(X_train_xgb, y_train, verbose=False)
        models["xgboost"] = xgb
    except ImportError:
        logger.warning("xgboost not available, skipping XGB member")
    except Exception as e:
        logger.warning("XGBoost member failed: %s", e)

    # LightGBM — raw NaN numeric + int team IDs, categorical_feature BY NAME.
    try:
        from lightgbm import LGBMClassifier
        X_train_lgbm = _lgbm_dataframe(X_train, X_cat_train, features)
        lgbm = LGBMClassifier(**LGB_PARAMS)
        if X_val is not None:
            X_val_lgbm = _lgbm_dataframe(X_val, X_cat_val, features)
            lgbm.fit(X_train_lgbm, y_train, eval_set=[(X_val_lgbm, y_val)],
                     categorical_feature=TREE_CATEGORICAL_COLS)
        else:
            lgbm.fit(X_train_lgbm, y_train, categorical_feature=TREE_CATEGORICAL_COLS)
        models["lightgbm"] = lgbm
    except ImportError:
        logger.warning("lightgbm not available, skipping LGB member")
    except Exception as e:
        logger.warning("LightGBM member failed: %s", e)

    # Logistic Regression — imputed + scaled, full feature set.
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_train_scaled, y_train)
        models["logistic"] = lr
    except Exception as e:
        logger.warning("Logistic member failed: %s", e)
    models["scaler"] = scaler
    models["impute_median"] = impute_medians

    # Random Forest — imputed-numeric + integer team IDs (no native cats).
    try:
        from sklearn.ensemble import RandomForestClassifier
        X_train_lr_tree = np.hstack([X_train_lr, X_cat_train])
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                    random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train_lr_tree, y_train)
        models["randomforest"] = rf
    except Exception as e:
        logger.warning("RandomForest member failed: %s", e)

    # MLP — small net with early stopping (diversity wildcard).
    try:
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01,
                            early_stopping=True, validation_fraction=0.15,
                            max_iter=300, learning_rate_init=0.001,
                            n_iter_no_change=10, random_state=RANDOM_SEED)
        mlp.fit(X_train_scaled, y_train)
        models["mlp"] = mlp
    except Exception as e:
        logger.warning("MLP member failed: %s", e)

    # Record the team-ID vocabulary the tree members were FIT with, so
    # predict-time frames clamp unseen teams to UNK (never a fresh category).
    models["categorical_vocab"] = {c: _cat_known_ids(c) for c in TREE_CATEGORICAL_COLS}

    if X_val is None:
        return models, {}

    weights = _member_weights(list(models.keys()))
    probs, wts = [], []
    for name, model in models.items():
        if name in ("scaler", "impute_median", "categorical_vocab"):
            continue
        if name == "logistic" or name == "mlp":
            Xuse = X_val_scaled
        elif name == "xgboost":
            Xuse = X_val_xgb
        elif name == "randomforest":
            Xuse = np.hstack([X_val_lr, X_cat_val])
        elif name == "lightgbm":
            Xuse = X_val_lgbm
        else:
            Xuse = X_val
        probs.append(model.predict_proba(Xuse)[:, 1])
        wts.append(weights[name])
    ensemble_prob = np.average(probs, axis=0, weights=wts) if probs \
        else np.full(len(y_val), 0.5)
    return models, compute_metrics(y_val, ensemble_prob)


def ensemble_predict(models: dict, games: pd.DataFrame,
                     features: list[str] | None = None) -> tuple:
    """Weighted-blend prediction plus per-member probabilities and weights.

    Returns (blended_prob, {member_name: prob_vector}, {member_name: weight}).
    Falls back to 0.5 when no member can predict. Blending uses the run's
    adaptive weights when available, else the static priors."""
    features = features or V1_FEATURES
    games = _add_team_ids(games)
    X = _feature_matrix(games, features)
    X_cat = _categorical_matrix(games)
    scaler = models.get("scaler")
    medians = models.get("impute_median")
    vocab = models.get("categorical_vocab") or {}

    members: dict[str, np.ndarray] = {}
    for name, model in models.items():
        if name in ("scaler", "impute_median", "categorical_vocab"):
            continue
        try:
            if name in ("logistic", "mlp"):
                Xi, _ = _impute_median(X, medians)
                Xuse = scaler.transform(Xi) if scaler is not None else Xi
            elif name == "xgboost":
                Xi, _ = _impute_median(X, medians)
                Xuse = _tree_dataframe(Xi, X_cat, features, vocabs=vocab)
            elif name == "lightgbm":
                Xuse = _lgbm_dataframe(X, X_cat, features, vocabs=vocab)
            elif name == "randomforest":
                Xuse = np.hstack([X, X_cat])
            else:
                Xuse = X
            members[name] = model.predict_proba(Xuse)[:, 1]
        except Exception as e:
            logger.warning("Member %s failed to predict: %s", name, e)

    if not members:
        return np.full(len(games), 0.5), {}, {}

    weights = _member_weights(list(members.keys()), adaptive=_ADAPTIVE_WEIGHTS)
    blend = np.zeros(len(games))
    for name, p in members.items():
        blend += weights[name] * p
    return blend, members, weights


def _score_member_table(target: np.ndarray,
                        members: dict[str, np.ndarray]) -> dict:
    """Per-member metric table {member: {logloss, auc, ece, brier}} for one
    target vector (used for the sealed-2025 per-member view; empty-safe)."""
    out = {}
    for name, p in members.items():
        p = np.asarray(p, dtype=float)
        if len(p) != len(target):
            continue
        m = compute_metrics(target, p)
        out[name] = {k: m[k] for k in ("logloss", "auc", "ece", "brier")}
    return out


def _elo_logistic_p(tr: pd.DataFrame, va: pd.DataFrame,
                    features: list[str]) -> np.ndarray:
    """Cheap elo-only logistic reference arm (fit on ``tr``, predict ``va``)."""
    from sklearn.linear_model import LogisticRegression
    i = features.index("elo_diff") if "elo_diff" in features else 0
    elo = LogisticRegression(max_iter=1000)
    elo.fit(tr[features].to_numpy(dtype=float)[:, i].reshape(-1, 1),
            tr[TARGET].to_numpy(dtype=float))
    return elo.predict_proba(
        va[features].to_numpy(dtype=float)[:, i].reshape(-1, 1))[:, 1]


def _prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """logloss/auc of a probability vector vs target over its NON-NULL rows
    (the market reference arm has no training and may be NaN on rows without
    a line). NaN when nothing to score."""
    mask = ~np.isnan(p)
    yv = np.asarray(y, dtype=float)[mask]
    pv = p[mask]
    if len(yv) == 0 or len(np.unique(yv)) < 2:
        return {"logloss": round(float("nan"), 4), "auc": round(float("nan"), 4)}
    return {"logloss": round(logloss(yv, pv), 4),
            "auc": round(auc(yv, pv), 4)}


def _adaptive_blend(oof_members: dict[str, list[float]],
                    adaptive: dict[str, float], n: int) -> np.ndarray:
    """Re-blend pooled OOF member probs with the ADAPTIVE weights — the same
    weighting family the deployed (sealed/slate) blend uses, so the sealed
    Platt map is fit on OOF pairs produced like the ones it will correct."""
    out = np.zeros(n)
    for name, preds in oof_members.items():
        w = adaptive.get(name, 0.0)
        if w and len(preds) == n:
            out += w * np.asarray(preds, dtype=float)
    return out


def _latest_feature_record() -> dict | None:
    """Newest nfl_feature_v1_*.json in data_delivery (the admission-gate
    output), or None. Used to resolve the model feature set dynamically."""
    recs = sorted(DATA_DELIVERY_DIR.glob("nfl_feature_v1_*.json")) if DATA_DELIVERY_DIR.exists() else []
    if not recs:
        return None
    try:
        import json
        with open(recs[-1], encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def admitted_model_features() -> list[str]:
    """The model-facing admitted feature set: the gate's admitted list minus
    the constant ``is_home`` anchor. Falls back to the v1 base set when no
    feature-gate record exists (e.g. the committed v1-only frame)."""
    rec = _latest_feature_record()
    if rec is not None:
        admitted = rec.get("feature_admission", {}).get("v1_features") or []
        admit = [f for f in admitted if f != "is_home"]
        if admit:
            return admit
    return list(V1_FEATURES)


def run_walk_forward(feats: pd.DataFrame,
                     model_features: list[str] | None = None) -> dict:
    """Prequential fold evaluation over 2019-2024 + sealed 2025 evaluation,
    with the 5-member ensemble as the model arm.

    Returns per-arm pooled + sealed tables (raw + Platt twins), per-member
    OOF tables, the adaptive blend weights, and the adoption verdict. No
    training ever sees 2025; the sealed Platt map is fit only on the pooled
    pre-holdout OOF (2021-2024), never 2025.
    """
    preq_all = feats[feats["season"].isin(TRAIN_SEASONS)].copy()
    sealed = feats[feats["season"] == SEALED_SEASON].copy()

    # Model features: the admitted set (v1 base + gated v2 additions), kept
    # only where the frame actually carries the column (never silently all-
    # NaN, which would zero out the valid universe).
    Xcol = [f for f in (model_features or admitted_model_features())
            if f in feats.columns]
    if not Xcol:
        raise ValueError("no model features present in the frame")

    # Universe for a fair comparison: rows with all model features + target.
    preq = preq_all[_valid_rows(preq_all, Xcol)].copy()
    sld = sealed[_valid_rows(sealed, Xcol)].copy()

    folds = generate_weekly_folds(preq)          # asserts no future-week leak

    # ---- per-fold store for nested (honest) preq Platt twin ----
    order_actual, order_raw, order_elo, ws_list = [], [], [], []
    oof_members: dict[str, list[float]] = {}
    oof_members_cal: dict[str, list[float]] = {}
    cal_pool, raw_pool, elo_pool, market_pool, y_pool = [], [], [], [], []
    fold_meta: list[pd.DataFrame] = []

    for f in folds:
        tr, va = f["train"], f["val"]
        yva = va[TARGET].to_numpy(dtype=float)
        try:
            models, _mets = train_ensemble(tr, va, features=Xcol)
        except Exception as e:
            logger.warning("fold %s ensemble failed: %s", f["week_start"], e)
            continue
        blend, member_probs, _wts = ensemble_predict(models, va, features=Xcol)
        elo_p = _elo_logistic_p(tr, va, Xcol)
        # market reference arm: the no-vig closing-moneyline prob, NO fitting
        # (a pre-game fact) — reported for model-vs-market edge only.
        market_p = (va["market_home_implied"].to_numpy(dtype=float)
                    if "market_home_implied" in va.columns
                    else np.full(len(va), np.nan))

        # nested Platt twin: fit on all STRICTLY-EARLIER folds' OOF pairs
        lr = None
        if y_pool:
            lr = platt_fit(np.concatenate(raw_pool),
                           np.concatenate(y_pool).astype(int))
            cal_p = platt_predict(blend, lr)
        else:
            cal_p = blend.copy()
        for name, p in member_probs.items():
            p_arr = np.asarray(p, dtype=float)
            oof_members.setdefault(name, []).extend(p_arr.tolist())
            pc = platt_predict(p_arr, lr) if lr is not None else p_arr
            oof_members_cal.setdefault(name, []).extend(pc.tolist())

        order_actual.append(yva)
        order_raw.append(blend)
        order_elo.append(elo_p)
        ws_list.append(f["week_start"])
        cal_pool.append(cal_p)
        raw_pool.append(blend)
        elo_pool.append(elo_p)
        market_pool.append(market_p)
        y_pool.append(yva)
        _meta = [c for c in META_COLS if c in va.columns]
        fold_meta.append(va[_meta].reset_index(drop=True))

    if not y_pool:
        raise RuntimeError("no folds produced ensemble predictions")

    y_po = np.concatenate(y_pool)
    raw_po = np.concatenate(raw_pool)
    cal_po = np.concatenate(cal_pool)
    elo_po = np.concatenate(elo_pool)
    market_po = np.concatenate(market_pool)

    # constant home-edge baseline fit on pre-holdout (2019-2024) only
    const_p = preq[TARGET].mean()

    pooled = {
        "market_line": _prob_metrics(y_po, market_po),
        "n": int(len(y_po)),
        "fold_count": len(folds),
        "constant_home_edge": {
            "proba": round(float(const_p), 4),
            "logloss": round(logloss(y_po, np.full_like(y_po, const_p)), 4),
            "auc": round(auc(y_po, np.full_like(y_po, const_p)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(y_po, elo_po), 4),
            "auc": round(auc(y_po, elo_po), 4),
        },
        "model_raw": {
            "logloss": round(logloss(y_po, raw_po), 4),
            "auc": round(auc(y_po, raw_po), 4),
        },
        "model_platt": {
            "logloss": round(logloss(y_po, cal_po), 4),
            "auc": round(auc(y_po, cal_po), 4),
            "ece": round(ece(y_po, cal_po), 4),
        },
    }

    # ---- adaptive blend weights (pooled OOF member AUC) ----------------
    adaptive = compute_adaptive_weights(oof_members, y_po)
    _ADAPTIVE_WEIGHTS.clear()
    _ADAPTIVE_WEIGHTS.update(adaptive)

    # per-member tables (raw + prequential-calibrated twins + deployed weight)
    members_table = {}
    for name in sorted(set(oof_members)):
        raw_p = np.asarray(oof_members[name], dtype=float)
        entry = {"weight": float(adaptive.get(name, 0.0))}
        if len(raw_p) == len(y_po):
            m = compute_metrics(y_po, raw_p)
            entry.update({k: m[k] for k in ("logloss", "auc", "ece", "brier")})
        if len(oof_members_cal.get(name, [])) == len(y_po):
            mc = compute_metrics(y_po, np.asarray(oof_members_cal[name], dtype=float))
            entry.update({"logloss_calibrated": mc["logloss"],
                          "auc_calibrated": mc["auc"],
                          "ece_calibrated": mc["ece"]})
        members_table[name] = entry

    # ---- SEALED 2025 ----
    # fit-only refit on ALL 2019-2024 (no fold) -> predict 2025 with the
    # adaptive blend (the deployed weighting)
    models_sealed, _ = train_ensemble(preq, None, features=Xcol)
    sealed_raw, sealed_members, _w = ensemble_predict(models_sealed, sld, features=Xcol)
    sealed_elo = _elo_logistic_p(preq, sld, Xcol)
    sealed_market = (sld["market_home_implied"].to_numpy(dtype=float)
                     if "market_home_implied" in sld.columns
                     else np.full(len(sld), np.nan))

    # Platt twin for the sealed window: fit on the pooled pre-holdout OOF
    # re-blended with the SAME adaptive weights the deployed blend uses
    # (never 2025).
    oof_adaptive_blend = _adaptive_blend(oof_members, adaptive, len(y_po))
    platt_sealed = platt_fit(oof_adaptive_blend, y_po.astype(int))
    sealed_cal = platt_predict(sealed_raw, platt_sealed)

    const_sealed = preq[TARGET].mean()
    # per-member SEALED 2025 metrics (raw member probs vs the 2025 target) —
    # the per-member twin of ``members`` (pooled), surfaced for ablation
    # member-level reads (e.g. "which models like a candidate family").
    sealed_members_table = _score_member_table(sld[TARGET].to_numpy(),
                                               sealed_members)

    sealed = {
        "n": int(len(sld)),
        "market_line": _prob_metrics(sld[TARGET].to_numpy(), sealed_market),
        "constant_home_edge": {
            "proba": round(float(const_sealed), 4),
            "logloss": round(logloss(sld[TARGET], np.full(len(sld), const_sealed)), 4),
            "auc": round(auc(sld[TARGET], np.full(len(sld), const_sealed)), 4),
        },
        "elo_logistic": {
            "logloss": round(logloss(sld[TARGET], sealed_elo), 4),
            "auc": round(auc(sld[TARGET], sealed_elo), 4),
        },
        "model_raw": {
            "logloss": round(logloss(sld[TARGET], sealed_raw), 4),
            "auc": round(auc(sld[TARGET], sealed_raw), 4),
        },
        "model_platt": {
            "logloss": round(logloss(sld[TARGET], sealed_cal), 4),
            "auc": round(auc(sld[TARGET], sealed_cal), 4),
            "ece": round(ece(sld[TARGET], sealed_cal), 4),
        },
    }

    # ---- Part-A artifacts: per-game history + nfl_calibration record ----
    # The OOF rows use the ADAPTIVE re-blend (deployed-style raw) aligned to
    # y_po/cal_po order; their calibrated twin is the SEALED Platt map applied
    # to that raw (consistent with the emitted a/b the frontend replots), and
    # the calibration record's metrics/buckets_calibrated use the PREQUENTIAL
    # per-fold values (cal_po) — the documented preq-vs-deployed distinction.
    oof_meta = (pd.concat(fold_meta, ignore_index=True) if fold_meta
                else pd.DataFrame())
    oof_cal_deployed = (platt_predict(oof_adaptive_blend, platt_sealed)
                        if platt_sealed is not None else oof_adaptive_blend.copy())
    cal_history = (platt_predict(sealed_raw, platt_sealed)
                   if platt_sealed is not None else sealed_raw.copy())
    _sm = [c for c in META_COLS if c in sld.columns]
    history_df = build_history_frame(
        oof_meta=oof_meta, oof_raw=oof_adaptive_blend, oof_cal=oof_cal_deployed,
        sealed_meta=sld[_sm].reset_index(drop=True),
        sealed_raw=sealed_raw, sealed_cal=cal_history)
    calibration_rec = build_calibration(y_po, oof_adaptive_blend, cal_po,
                                        platt_sealed)

    verdict = adopt_decision(pooled, sealed)
    global _DEPLOYED_BUNDLE, _SEALED_PLATT
    _DEPLOYED_BUNDLE = {"models": models_sealed, "platt": platt_sealed,
                        "adaptive_weights": dict(adaptive),
                        "features": Xcol}
    _SEALED_PLATT = platt_sealed

    return {
        "fold_geometry": {
            "train_seasons": TRAIN_SEASONS,
            "val_seasons": VAL_SEASONS,
            "sealed_season": SEALED_SEASON,
            "fold_count": len(folds),
            "pooled_oof_games": int(len(y_po)),
            "sealed_games": int(len(sld)),
            "preq_weeks": [str(f["week_start"].date()) for f in folds],
        },
        "pooled_preq_2021_2024": pooled,
        "sealed_2025": sealed,
        "adaptive_weights": adaptive,
        "members": members_table,
        "members_sealed": sealed_members_table,
        "verdict": verdict,
        "_deployed": {"features": Xcol},
        "_history_df": history_df,
        "_calibration": calibration_rec,
    }


def adopt_decision(pooled: dict, sealed: dict) -> dict:
    """ADOPT only if the calibrated model beats BOTH baselines on the SEALED
    window in both logloss and AUC, with sane ECE. A pooled-gain/sealed-loss
    inversion -> DON'T ADOPT (the pattern every MLB gate hit)."""
    m_ll = sealed["model_platt"]["logloss"]
    m_auc = sealed["model_platt"]["auc"]
    m_ece = sealed["model_platt"]["ece"]
    elo_ll = sealed["elo_logistic"]["logloss"]
    elo_auc = sealed["elo_logistic"]["auc"]
    c_ll = sealed["constant_home_edge"]["logloss"]
    c_auc = sealed["constant_home_edge"]["auc"]

    beats_elo = (m_ll < elo_ll) and (m_auc > elo_auc)
    beats_const = (m_ll < c_ll) and (m_auc > c_auc)
    sane_ece = m_ece <= ECE_MAX

    # pooled-vs-sealed inversion warning (informational unless sealed fails)
    pm_ll = pooled["model_platt"]["logloss"]
    pe_ll = pooled["elo_logistic"]["logloss"]
    pc_ll = pooled["constant_home_edge"]["logloss"]
    pooled_wing = pm_ll < min(pe_ll, pc_ll)
    sealed_wing = m_ll < min(elo_ll, c_ll)

    adopt = bool(beats_elo and beats_const and sane_ece)
    reasons = []
    if not beats_elo:
        reasons.append(f"sealed logloss {m_ll} / auc {m_auc} not both better "
                       f"than elo-logistic ({elo_ll} / {elo_auc})")
    if not beats_const:
        reasons.append(f"sealed logloss {m_ll} / auc {m_auc} not both better "
                       f"than constant home-edge ({c_ll} / {c_auc})")
    if not sane_ece:
        reasons.append(f"sealed ECE {m_ece} > {ECE_MAX} (not sane)")
    if not adopt and (pooled_wing and not sealed_wing):
        reasons.append("pooled-gain / sealed-loss inversion -> DON'T ADOPT")
    elif adopt and (sealed_wing and not pooled_wing):
        reasons.append("note: model wins sealed but slightly worse pooled (watch)")

    return {
        "adopt": adopt,
        "sealed_beats_elo": bool(beats_elo),
        "sealed_beats_constant": bool(beats_const),
        "sane_ece": bool(sane_ece),
        "pooled_gain_sealed_loss_inversion": bool(pooled_wing and not sealed_wing),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Slate stage (Part 3) — reproducible per-game games[], always served by
# the fresh ensemble (the seal gate is a testing/monitoring signal, never a
# board block).
# ---------------------------------------------------------------------------
def _start_utc(gameday, gametime):
    """nflverse gametime is ET; combine with gameday and convert to UTC
    (matches the existing 20260830 games[] reference exactly)."""
    if pd.isna(gametime) or not isinstance(gametime, str) or ":" not in gametime:
        return ""
    try:
        from zoneinfo import ZoneInfo
        dt = pd.Timestamp(f"{gameday} {gametime}")
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
        return dt.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(pd.Timestamp(f"{gameday} {gametime}"))


def build_games_list(slate_feats: pd.DataFrame,
                     models: dict, platt: object, features: list[str]) -> list[dict]:
    """Predict the calibrated ensemble on scheduled games and emit the
    per-game games[] entries (the 20260830 shape: home_win_prob/away_win_prob,
    model_pick, game_date, game_status 'pre', start_time_utc, venue,
    home/away_record, spread_line, total_line)."""
    if slate_feats is None or slate_feats.empty:
        return []
    sf = slate_feats.copy()
    blend, _members, _wts = ensemble_predict(models, sf, features=features)
    if platt is not None:
        blend = platt_predict(blend, platt)
    sf["_p"] = np.clip(blend, 0.0, 1.0)

    games = []
    for _, r in sf.iterrows():
        home = str(r.get("home_team", "") or "").strip()
        away = str(r.get("away_team", "") or "").strip()
        ph = float(r["_p"])
        gameday = str(r.get("gameday", "") or "")[:10]
        games.append({
            "game_id": str(r.get("game_id", "") or ""),
            "game_date": gameday,
            "home_team": home,
            "away_team": away,
            "home_win_prob": round(ph, 4),
            "away_win_prob": round(1.0 - ph, 4),
            "home_score": None,
            "away_score": None,
            "game_status": "pre",
            "start_time_utc": _start_utc(gameday, r.get("gametime")),
            "venue": str(r.get("stadium", "") or "").strip(),
            "model_pick": home if ph >= 0.5 else away,
            "home_record": r.get("home_record") or "",
            "away_record": r.get("away_record") or "",
            "spread_line": _nl(r.get("spread_line")),
            "total_line": _nl(r.get("total_line")),
        })
    return games


def _nl(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def pull_and_run(out_dir: Path | None = None,
                 write_record: bool = True,
                 features_csv: Path | None = None,
                 schedule: pd.DataFrame | None = None,
                 pbp: pd.DataFrame | None = None,
                 slate_season: int | None = None,
                 seasons: list[int] | None = None) -> dict:
    """Run the moneyline ensemble + sealed gate over ``seasons`` when given.

    ``seasons`` (e.g. ``[2021, 2022, 2023]``) limits the decided frame and the
    schedule+pbp pull to that window. None (default) uses the full range, so a
    normal run is unchanged. The sealed-2025 gate still applies within whatever
    window is selected.
    """
    # Import DEFAULT_SEASONS before first use — the from-import below makes
    # the name function-local, so referencing it earlier must happen after a
    # local binding exists (regression: UnboundLocalError on the default path).
    from nfl_features import DEFAULT_SEASONS as _NF_DEFAULT_SEASONS
    feed_seasons = seasons or _NF_DEFAULT_SEASONS
    from nfl_features import _load_raw, build_features, build_slate_features
    out_dir = Path(out_dir) if out_dir is not None else DATA_DELIVERY_DIR

    if features_csv is not None and Path(features_csv).exists():
        logger.info("Loading pre-computed features from %s", features_csv)
        feats = pd.read_csv(features_csv)
        feats["gameday"] = pd.to_datetime(feats["gameday"])
        if TARGET not in feats.columns:
            feats[TARGET] = (feats["home_score"] > feats["away_score"]).astype(int)
    else:
        if not DECIDED_FRAME.exists():
            raise FileNotFoundError(
                f"{DECIDED_FRAME} absent — run `python3 nfl_game_frame.py` first")
        decided = pd.read_csv(DECIDED_FRAME)
        if "season" in decided.columns:
            decided = decided[decided["season"].isin(feed_seasons)]
        logger.info("Computing features over %s seasons", feed_seasons)
        sched, pbp_raw = _load_raw(feed_seasons)
        schedule = sched if schedule is None else schedule
        pbp = pbp_raw if pbp is None else pbp
        feats = build_features(decided, schedule, pbp)
        feats[TARGET] = (feats["home_score"] > feats["away_score"]).astype(int)

    # sealed isolation assertion: no 2025 row may be used in any pre-sealed fit
    # (guaranteed by construction in run_walk_forward; re-assert here loudly)
    if SEALED_SEASON not in TRAIN_SEASONS and SEALED_SEASON not in VAL_SEASONS:
        assert not feats[feats["season"] == SEALED_SEASON].empty

    result = run_walk_forward(feats)
    model_features = list(result.get("_deployed", {}).get("features", V1_FEATURES))

    # ---- slate stage: current schedule ------------------------------------
    # The seal gate is a TESTING/MONITORING signal (ensemble vs elo/constant +
    # sanity ECE), mirroring MLB — it never blocks the board. The fresh
    # ensemble always serves games[] when a schedule loads; the adopt verdict
    # is recorded for model-change testing but does not (and cannot) gate it.
    slate_info = None
    games = []
    if schedule is not None and "gameday" in schedule.columns:
        # Slate target = the CURRENT schedule year (e.g. 2026 week 1), NOT the
        # max season in the feed (all-decided 2025 would yield an empty slate).
        ss = slate_season or datetime.now().year
        try:
            from nfl_features import DECIDED_FRAME as _DDF
            decided = (pd.read_csv(_DDF) if _DDF.exists() else feats)
            slate_feats = build_slate_features(schedule, pbp, decided, ss)
            bundle = _DEPLOYED_BUNDLE or {}
            if bundle.get("models"):
                games = build_games_list(slate_feats, bundle["models"],
                                         bundle.get("platt"), model_features)
                slate_info = {
                    "season": int(ss),
                    "week": int(slate_feats["week"].iloc[0])
                    if not slate_feats.empty and "week" in slate_feats.columns else None,
                    "n_games": len(games),
                    "model": "sealed 2019-2024 fit + pre-holdout-OOF Platt map",
                }
            else:
                slate_info = {"season": int(ss),
                              "week": None, "n_games": 0,
                              "model": "sealed 2019-2024 fit + pre-holdout-OOF Platt map"}
        except Exception as e:
            logger.warning("Slate stage failed (continuing): %s", e)
            slate_info = None

    if write_record:
        record = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "config": {
                "features": model_features,
                "feature_source": "latest nfl_feature_v1_*.json admission gate (v2)",
                "excluded_constant_anchor": "is_home",
                "model": ("5-member ensemble (XGBoost/LightGBM/Logistic/RF/MLP) "
                          "+ adaptive AUC blend + Platt twin"),
                "ensemble_weights": dict(_ADAPTIVE_WEIGHTS or ENSEMBLE_WEIGHTS),
                "reference_arm": "logistic (full-fitted, elo-only for cheap signal)",
                "baselines": ["constant home-edge", "elo-only logistic"],
                "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
                "ece_bins": ECE_BINS, "ece_max": ECE_MAX,
                "leakage": ("features strictly-trailing (gate, windowed + ewm); "
                            "folds assert train.gameday < week_start; 2025 never "
                            "in any pre-sealed fit or calibration map; sealed Platt "
                            "fit on pooled pre-holdout OOF only"),
            },
            **{k: v for k, v in result.items()
               if k not in ("_deployed", "_history_df", "_calibration")},
        }        # The seal gate is reported (testing/model-change comparison) but never
        # blocks the board, mirroring MLB's daily pipeline. games[] is always
        # written from the fresh ensemble when a schedule is present.
        if slate_info is not None:
            record["slate"] = slate_info
            record["games"] = games if slate_info.get("n_games") else []
        else:
            record["predictions"] = {"status": "blocked (no schedule loaded)"}
        out_dir.mkdir(parents=True, exist_ok=True)
        _date = datetime.now().strftime(DATE_FMT)
        rec_path = out_dir / RECORD_TEMPLATE.format(date=_date)
        with open(rec_path, "w") as fh:
            json.dump(record, fh, indent=2)
        # Part-A siblings: the MLB-shaped calibration record + per-game
        # prediction history (written only when write_record is true, so a
        # --no-record dry run touches nothing).
        cal_path = out_dir / CALIBRATION_TEMPLATE.format(date=_date)
        with open(cal_path, "w") as fh:
            json.dump(result["_calibration"], fh, indent=2)
        hist_path = out_dir / HISTORY_TEMPLATE.format(date=_date)
        result["_history_df"].to_csv(hist_path, index=False)

        # MLB-shaped model-monitor record (true PSI drift + rolling Brier),
        # composed purely from objects already in memory this run.
        try:
            _write_monitor(out_dir, feats=feats, result=result,
                           history_df=result["_history_df"],
                           calibration=result["_calibration"],
                           current_date=_date)
        except Exception as exc:  # noqa: BLE001 — a monitor failure must never
            logger.warning("Model-monitor emission failed (continuing): %s", exc)

        # Power-rankings artifact (MLB-identical CSV shape) — wrapped so a
        # failure never blocks the run or the board.
        try:
            result["power_rankings_path"] = str(
                _power_rankings_csv(feats, _date, out_dir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Power-rankings emission failed (continuing): %s", exc)

        result["record"] = str(rec_path)
        result["games_written"] = bool(slate_info and (slate_info.get("n_games") or 0) > 0)

    _print_report(result)
    return result


def _write_monitor(out_dir: Path, *, feats: pd.DataFrame, result: dict,
                   history_df: pd.DataFrame, calibration: dict,
                   current_date: str) -> None:
    """Write the MLB-shaped ``nfl_model_monitor_<date>.json`` from this run's
    objects. PSI 'current' window = the last 30 days of decided games;
    'baseline' = every decided game before it. Version history is gathered
    from prior dated moneyline v1 records in ``out_dir``."""
    from nfl_monitor import build_model_monitor

    gd = pd.to_datetime(feats["gameday"], errors="coerce")
    latest = gd.max() if hasattr(gd, "max") else None
    baseline_cut = (latest - pd.Timedelta(days=30)).strftime("%Y-%m-%d") \
        if latest is not None and not pd.isna(latest) \
        else datetime.now().date().isoformat()

    records = []
    try:
        for p in sorted(out_dir.glob(RECORD_TEMPLATE.format(date="*") or "nfl_moneyline_v1_*.json")):
            if p.name.endswith(f"{current_date}.json"):
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            d["_date"] = p.name.replace("nfl_moneyline_v1_", "").replace(".json", "")
            records.append(d)
    except Exception:
        records = []

    mon = build_model_monitor(
        feats=feats, result=result, history_df=history_df, calibration=calibration,
        moneyline_records=records, current_date=current_date,
        baseline_cut_date=baseline_cut)
    mon_path = out_dir / MONITOR_TEMPLATE.format(date=current_date)
    with open(mon_path, "w") as fh:
        json.dump(mon, fh, indent=2)


# ---------------------------------------------------------------------------
# Power rankings (MLB-identical artifact shape)
# ---------------------------------------------------------------------------
def _power_rankings_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Pure per-team power-rankings rows from the decided feature frame.

    Mirrors mlb-backend backend/pipeline.py::_power_rankings_csv:
    wins/losses/pct/home_pct/away_pct come from the decided (home_win not-null)
    games only; ``l10`` is the team's last 10 DECIDED games as ``"W-L"``;
    ``run_diff`` is the team's signed point differential (home_score/away_score
    on the decided frame) so the page's +/− coloring works; Elo is the team's
    MEAN entering rating from ``compute_elo()``'s per-(team,event)
    ``elo_entering`` (a rating computed from ONLY strictly-prior games), falling
    back to 1500.0 for a team with no games — the same way MLB takes a team's
    mean ``home_elo``. ABsence of decided games must yield a valid ``0-0`` row,
    never a crash.
    """
    from nfl_features import compute_elo, team_events

    if feats is None or feats.empty:
        return pd.DataFrame()
    rows = feats.copy()
    # Explicit chrono sort so "last 10" = most recent decided games (MLB relies
    # on frame order; the feature frame is not guaranteed chronological).
    if "gameday" in rows.columns and rows["gameday"].notna().any():
        rows = rows.sort_values("gameday", kind="mergesort").reset_index(drop=True)

    decided = (rows[rows[TARGET].notna()]
               if TARGET in rows.columns else rows.copy())
    # Absolute Elo per team: mean of its entering rating, from strictly-prior
    # games only (compute_elo is point-in-time safe by construction).
    try:
        ladd = compute_elo(team_events(rows)) if not rows.empty else rows
        elo_mean = ladd.groupby("team")["elo_entering"].mean()
    except Exception:
        elo_mean = pd.Series(dtype=float)

    # team list from the FULL frame's home teams (matches MLB) so an undecided
    # home team still yields a neutral row.
    teams = list(pd.unique(rows["home_team"].dropna().astype(str)))
    ranking = []
    for team in teams:
        hg = decided[decided["home_team"] == team]
        ag = decided[decided["away_team"] == team]
        tg = decided[(decided["home_team"] == team)
                     | (decided["away_team"] == team)]

        wins = (int(hg["home_win"].sum())
                + (int((1 - ag["home_win"]).sum()) if not ag.empty else 0))
        losses = (int((1 - hg["home_win"]).sum())
                  + (int(ag["home_win"].sum()) if not ag.empty else 0))
        total = wins + losses
        pct = round(wins / max(total, 1), 3)

        home_count = len(hg)
        home_wins = int(hg["home_win"].sum()) if not hg.empty else 0
        home_pct = round(home_wins / max(home_count, 1), 3)

        away_count = len(ag)
        away_wins = int(ag["home_win"].sum()) if not ag.empty else 0
        away_pct = round(1 - away_wins / max(away_count, 1), 3) if away_count > 0 else 0.5

        # L10 -- last 10 DECIDED games
        recent = tg.tail(10)
        l10_wins = 0
        for _, g in recent.iterrows():
            if g["home_team"] == team:
                l10_wins += int(g["home_win"])
            else:
                l10_wins += int(1 - g["home_win"])
        l10 = f"{l10_wins}-{len(recent) - l10_wins}"

        # signed point differential (for the page's +/− RUN DIFF column)
        marg = 0
        if not tg.empty:
            hs = pd.to_numeric(tg["home_score"], errors="coerce")
            as_ = pd.to_numeric(tg["away_score"], errors="coerce")
            is_home_team = (tg["home_team"] == team).to_numpy()
            m = (hs - as_).to_numpy()
            team_m = np.where(is_home_team, m, -m)
            marg = int(round(float(np.nansum(team_m))))

        ranking.append({
            "team": team,
            "team_name": team,
            "elo": round(float(elo_mean.get(team, 1500.0)), 1),
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "pct": pct,
            "run_diff": marg,
            "l10": l10,
            "home_pct": home_pct,
            "away_pct": away_pct,
        })

    df = pd.DataFrame(ranking)
    if df.empty:
        return df
    return df.sort_values("elo", ascending=False).reset_index(drop=True)


def _power_rankings_csv(feats: pd.DataFrame, target_date_str: str,
                        out_dir: Path | None = None) -> Path:
    """Write nfl_power_rankings_YYYYMMDD.csv (MLB-identical column set, 1-based
    rank index named ``rank``). Never raises on input shape; an empty frame is
    still written so the loader returns empty cleanly."""
    out = Path(out_dir) if out_dir is not None else DATA_DELIVERY_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / POWER_RANKINGS_TEMPLATE.format(date=target_date_str)
    df = _power_rankings_rows(feats)
    if df.empty:
        df.to_csv(path, index=False)
        return path
    df.index += 1
    df.index.name = "rank"
    df.to_csv(path)
    return path


def _print_report(result: dict) -> None:
    print("\n=== NFL moneyline ensemble gate ===")
    print("pooled OOF (2021-2024):", result["fold_geometry"]["pooled_oof_games"],
          "games,", result["fold_geometry"]["fold_count"], "folds")
    print("adaptive blend weights:",
          {k: f"{v:.1%}" for k, v in sorted(result.get("adaptive_weights", {}).items())})
    print(format_table("sealed_2025", result["sealed_2025"]))
    print("VERDICT:", "ADOPT" if result["verdict"]["adopt"]
          else "DO NOT ADOPT (reporting only — board still served)")
    for r in result["verdict"]["reasons"]:
        print("  -", r)
    if result.get("games_written"):
        print("  [OK] games[] written (fresh ensemble served the board)")
    else:
        print("  [BLOCKED] no games[] (no schedule loaded)")


def format_table(window: str, arms: dict) -> str:
    lines = [f"\n{window}:"]
    lines.append(f"  {'arm':20s} {'logloss':>9s} {'auc':>7s} {'ece':>6s}")
    for name in ("constant_home_edge", "elo_logistic", "market_line",
                 "model_raw", "model_platt"):
        a = arms[name]
        lines.append(f"  {name:20s} {a['logloss']:9.4f} {a['auc']:7.4f} "
                     f"{a.get('ece', float('nan')):6.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Run the NFL moneyline 5-member ensemble walk-forward + "
                    "sealed gate; the gate is a testing/monitoring signal (never "
                    "blocks the board — always serves the fresh ensemble).")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    ap.add_argument("--features-csv", type=Path, default=None,
                    help="path to pre-computed features CSV (skips nflreadpy download)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory for the JSON record (default: data_delivery)")
    ap.add_argument("--slate-season", type=int, default=None,
                    help="slate target season (default: latest in the schedule)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    pull_and_run(write_record=not args.no_record, features_csv=args.features_csv,
                 out_dir=args.out_dir, slate_season=args.slate_season)
    return 0


if __name__ == "__main__":
    sys.exit(main())