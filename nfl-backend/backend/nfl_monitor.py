"""NFL Model & Data Drift Monitor emitter — produces an MLB-shaped
``nfl_model_monitor_<date>.json`` so the shared Streamlit ``model_monitor.py``
page (the exact one MLB renders) can run for NFL unchanged.

The page contract (mirrors ``mlb-backend``'s ``model_monitor_*.json``):

``last_retrained`` / ``next_retrain`` + notes  — health cards
``upset_note``                                — upset-monitoring callout
``feature_drift[]``                           — TRUE PSI drift matrix
      [{feature, current_mean, baseline_mean, psi, status, weight_pct(optional),
        n_baseline, n_current}]
``features_metadata{}``                       — feature -> {"tooltip"} for hover
``feature_coverage[]``                        — non-null/measured per window
      [{feature, window, n_games, pct_measured, pct_nonnull, n_default_zero, status}]
``ensemble[]``                                — per-member weight + pooled metrics
``rolling_brier[]`` + ``rolling_brier_meta``  — rolling mean-Brier timeline
``brier_baseline``/``brier_baseline_label``   — dashed reference line
``version_history[]``                         — one row per dated moneyline v1 record

All functions are PURE (no I/O, no Streamlit) so they are unit-testable in
isolation; ``build_model_monitor`` composes them from the objects a moneyline
run already has in memory (the decided feature frame, the walk-forward result,
the per-game prediction history, and prior dated moneyline records).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Drift / coverage gates (mirror MLB's semantics).
PSI_WARN = 0.10
PSI_ALERT = 0.25
MIN_DRIFT_SAMPLES = 30        # below -> INSUFFICIENT (PSI only informational)
COVERAGE_FLOOR = 0.80         # below CURRENT window -> LOW_COVERAGE / STARVED
RETRAIN_INTERVAL_DAYS = 7     # next-retrain heuristic

# The 5-member moneyline walk-forward roster the monitor page describes verbatim.
MEMBER_DESC = {
    "xgboost": "Gradient-boosted decision trees (logloss, early-stopped per fold).",
    "lightgbm": "Leaf-wise histogram gradient boosting, routes missing values natively.",
    "logistic": "L2-regularized linear anchor; keeps the blend calibrated when trees overfit.",
    "randomforest": "Bagged decision trees; errors decorrelate from the boosted members.",
    "mlp": "Small low-capacity net that picks up smooth nonlinearities trees split around.",
}


# ---------------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------------
def psi_score(baseline: np.ndarray, current: np.ndarray,
              bins: int = 10) -> float:
    """True PSI: sum over equi-width (in counts) bins of the baseline
    distribution of ``(a - e) * ln(a / e)`` for the current-vs-baseline
    probability mass, with a small floor on the expected share to avoid
    log(0). Constant/singular features return 0.0 (no shift measurable)."""
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    finite_b = b[np.isfinite(b)]
    finite_c = c[np.isfinite(c)]
    if len(finite_b) == 0 or len(finite_c) == 0:
        return 0.0
    lo, hi = float(np.min(finite_b)), float(np.max(finite_b))
    if hi == lo:                       # constant feature -> no distributional shift
        return 0.0
    # Bin edges over the BASELINE range so current mass is scored in reference
    # bins; widen slightly to catch values just at the boundary.
    edges = np.linspace(lo, hi, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    def _share(x: np.ndarray) -> np.ndarray:
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
        counts = np.bincount(idx, minlength=bins).astype(float)
        return counts / max(counts.sum(), 1.0)
    e, a = _share(finite_b), _share(finite_c)
    eps = 1e-6
    e = np.where(e < eps, eps, e)
    a = np.where(a < eps, eps, a)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_status(psi: float, n_current: int, min_samples: int = MIN_DRIFT_SAMPLES) -> str:
    """OK / WARN / ALERT at MLB's 0.10 / 0.25 thresholds; INSUFFICIENT when the
    current window is too small to judge."""
    if n_current < min_samples:
        return "INSUFFICIENT"
    if psi >= PSI_ALERT:
        return "ALERT"
    if psi >= PSI_WARN:
        return "WARN"
    return "OK"


def feature_drift_rows(feats: pd.DataFrame, columns: list[str],
                       baseline_mask: np.ndarray, current_mask: np.ndarray,
                       weight: dict[str, float] | None = None) -> list[dict]:
    """One row per feature: current/baseline mean, true PSI, status, sample
    sizes. ``weight`` (optional feature->share) is surfaced as ``weight_pct``
    only when provided — NFL doesn't compute blend importances, so it is
    omitted by default and the page simply hides the MODEL WEIGHT column."""
    rows = []
    for col in columns:
        if col not in feats.columns:
            continue
        ser = pd.to_numeric(feats[col], errors="coerce")
        base = ser[baseline_mask].to_numpy()
        cur = ser[current_mask].to_numpy()
        base = base[np.isfinite(base)]
        cur = cur[np.isfinite(cur)]
        psi = psi_score(base, cur)
        rows.append({
            "feature": col,
            "current_mean": (round(float(np.mean(cur)), 4) if len(cur) else None),
            "baseline_mean": (round(float(np.mean(base)), 4) if len(base) else None),
            "psi": round(psi, 4),
            "status": drift_status(psi, int(len(cur))),
            "weight_pct": (round(100.0 * float(weight[col]), 1) if weight and col in weight else None),
            "n_baseline": int(len(base)),
            "n_current": int(len(cur)),
        })
    return rows


def feature_coverage_rows(feats: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Non-null coverage per feature across the decided frame — the visual
    backstop for silent data starvation (MLB's measured-vs-default split). NFL
    has no default-fill, so measured == non-null and n_default_zero is 0."""
    n = len(feats)
    rows = []
    for col in columns:
        if col not in feats.columns:
            continue
        nn = int(pd.to_numeric(feats[col], errors="coerce").notna().sum())
        pct = 100.0 * nn / n if n else 0.0
        if pct < 0.25 * 100.0:
            status = "STARVED"
        elif pct < COVERAGE_FLOOR * 100.0:
            status = "LOW_COVERAGE"
        else:
            status = "OK"
        rows.append({
            "feature": col, "window": "decided pool",
            "n_games": n, "pct_measured": round(pct, 1),
            "pct_nonnull": round(pct, 1), "n_default_zero": 0, "status": status,
        })
    return rows


# ---------------------------------------------------------------------------
# Rolling Brier timeline (from per-game prediction history)
# ---------------------------------------------------------------------------
def _per_game_brier(history: pd.DataFrame) -> pd.DataFrame:
    """Standard 2-class Brier per decided game, aligned to ``game_date``.

    Brier = (p_home - y_home)^2 + (p_away - y_away)^2. Substituting y (1 hot on
    the winner) and p_winner = prob of the actual winner: this equals
    2 * (1 - p_winner)^2 in both cases, so the factor of 2 is applied here and
    in the constant-home-edge baseline for a consistent, standard-Brier scale.
    """
    p = pd.to_numeric(history.get("home_win_prob_model"), errors="coerce")
    winner = history.get("actual_winner")
    home = history.get("home_team")
    if p is None or winner is None or home is None:
        return pd.DataFrame(columns=["date", "brier", "y"])
    y = (winner.astype(str) == home.astype(str)).astype(float)
    is_home_w = y.astype(bool)
    p_winner = p.where(is_home_w, 1.0 - p)
    gd = history.get("game_date").astype(str)
    return pd.DataFrame({"date": gd, "brier": 2.0 * (1.0 - p_winner) ** 2,
                         "y": y})



def rolling_brier_rows(history: pd.DataFrame, window_days: int = 30,
                       min_games_per_day: int = 2) -> tuple[list, dict, float | None, str]:
    """Trailing-window mean-Brier series. Days with fewer than
    ``min_games_per_day`` are excluded. Baseline = mean Brier of a CONSTANT
    HOME-EDGE forecast (predict the pooled home-win rate for every game) —
    the dashed reference line on the same date span. Returns
    (rows, meta, baseline_brier, baseline_label)."""
    df = _per_game_brier(history)
    empty = ([], {"window_days": window_days, "min_games_per_day": min_games_per_day,
                  "excluded_sparse_days": 0, "calibrator_is_identity": False,
                  "map_scope_note": "post-calibration probabilities are the deployed Platt map σ(a·logit(p)+b)"},
             None, "Constant home-edge")
    if df.empty:
        return empty
    d = pd.to_datetime(df["date"], errors="coerce")
    if d.notna().all():
        df = df[d.notna()].copy()
        df["date"] = d[d.notna()]
        dates = pd.to_datetime(df["date"]).dropna()
        end = dates.max()
        start = end - pd.Timedelta(days=window_days - 1)
        win = df[(pd.to_datetime(df["date"]) >= start) & (pd.to_datetime(df["date"]) <= end)]
    else:
        win = df
    if win.empty:
        return empty
    byday = win.groupby(pd.to_datetime(win["date"])).agg(games=("brier", "size"),
                                                         brier=("brier", "mean"))
    dense = byday[byday["games"] >= min_games_per_day]
    excluded = int(len(byday) - len(dense))
    rows = [{"date": d.strftime("%Y-%m-%d"), "brier": round(float(b), 4)}
            for d, b in zip(dense.index, dense["brier"])]
    # Constant home-edge baseline across the SAME decided games: predict the
    # pooled home-win rate for home every game. True 2-class Brier for that
    # constant p over binary outcomes y is mean(2 * (p - y)^2).
    base = _per_game_brier(history)
    if not base.empty and base["brier"].notna().any():
        y = base["y"].astype(float)
        home_rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        base_brier = float(np.mean(2.0 * (home_rate - y) ** 2))
    else:
        base_brier = None
    meta = {"window_days": window_days, "min_games_per_day": min_games_per_day,
            "excluded_sparse_days": excluded, "calibrator_is_identity": False,
            "map_scope_note": "post-calibration probabilities are the deployed Platt map σ(a·logit(p)+b)"}
    return rows, meta, base_brier, "Constant home-edge"


# ---------------------------------------------------------------------------
# Ensemble + version history
# ---------------------------------------------------------------------------
def ensemble_rows(result: dict, history_len: int) -> list[dict]:
    """Per-member row from the walk-forward result: name, pooled metrics,
    n_eval. Weight = the member's base adaptive weight (0 when zero)."""
    out = []
    weights = result.get("adaptive_weights") or {}
    members = result.get("members") or {}
    for name in ("xgboost", "lightgbm", "logistic", "randomforest", "mlp"):
        m = members.get(name, {}) or {}
        out.append({
            "name": name, "weight": float(weights.get(name, 0.0) or 0.0),
            "auc": m.get("auc"), "brier": m.get("brier"),
            "logloss": m.get("logloss"), "n_eval": history_len,
        })
    return out


def _platt_ab(calibration: dict | None) -> dict:
    params = calibration.get("params") if calibration else None
    return {"a": params.get("a"), "b": params.get("b")} if params else {"a": None, "b": None}


def version_history_rows(moneyline_records: list[dict],
                         current_date: str) -> list[dict]:
    """One row per dated moneyline v1 record (oldest first): version = its
    filename date, pooled raw auc/logloss, calibrated ECE, the deployed Platt
    map, and the per-member adaptive weights."""
    rows = []
    for rec in moneyline_records:
        verdict = rec.get("verdict") or {}
        pooled = rec.get("pooled_preq_2021_2024") or {}
        plt = (pooled.get("model_platt") or {})
        cal = rec.get("calibration") or {}
        rows.append({
            "version": rec.get("_date") or current_date,
            "date": str(rec.get("date") or rec.get("_date") or current_date),
            "weights": rec.get("adaptive_weights") or {},
            "auc": plt.get("auc"), "logloss": plt.get("logloss"),
            "ecer_calibrated": plt.get("ece"),
            "ece_calibrated": plt.get("ece"),
            "calibration": _platt_ab(cal),
            "adopt": verdict.get("adopt"),
        })
    return rows


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------
def build_model_monitor(*, feats: pd.DataFrame, result: dict,
                        history_df: pd.DataFrame, calibration: dict,
                        moneyline_records: list[dict], current_date: str,
                        baseline_cut_date: str) -> dict:
    """Compose the MLB-shaped monitor record.

    ``feats``            : decided game frame (with ``gameday`` + model columns)
    ``result``           : ``run_walk_forward`` result (members / weights /
                           verdict / trained_at)
    ``history_df``       : per-game OOF+sealed prediction history
    ``calibration``      : ``nfl_calibration_*.json`` (Platt a/b)
    ``moneyline_records``: prior dated nfl_moneyline_v1 records for the table
    ``current_date``     : YYYYMMDD
    ``baseline_cut_date``: ISO date; games >= it are the 'current' drift window
    """
    cols = list(result.get("_deployed", {}).get("features", []))
    c_iso = f"{current_date[:4]}-{current_date[4:6]}-{current_date[6:8]}"

    gd = pd.to_datetime(feats["gameday"], errors="coerce")
    cut = pd.to_datetime(baseline_cut_date)
    current_mask = gd.notna() & (gd >= cut)
    baseline_mask = gd.notna() & (gd < cut)
    if not current_mask.any():                       # fallback: newest games
        newest = gd.max()
        current_mask = gd.notna() & (gd == newest)
    if not baseline_mask.any():                      # fallback: everything before
        baseline_mask = gd.notna() & (gd < gd.max())

    drift = feature_drift_rows(feats, cols, baseline_mask.to_numpy(),
                               current_mask.to_numpy())
    coverage = feature_coverage_rows(feats, cols)
    r_brier, rb_meta, base_brier, base_label = rolling_brier_rows(history_df)

    trained = result.get("trained_at", "")
    last_retrained = str(trained)[:10]
    next_retrain = (pd.to_datetime(c_iso) + pd.Timedelta(days=RETRAIN_INTERVAL_DAYS)
                    ).strftime("%Y-%m-%d")

    return {
        "last_retrained": last_retrained,
        "last_retrained_note": "Fresh model trained this run (sealed gate: "
                               f"{'ADOPT' if (result.get('verdict') or {}).get('adopt') else 'DO NOT ADOPT (reporting only)'})",
        "next_retrain": next_retrain,
        "next_retrain_note": f"retrain scheduled {RETRAIN_INTERVAL_DAYS} days out",
        "upset_note": f"Model upset rate over the decided pool — {len(history_df):,} games scored; see Calibration for the upset strip.",
        "feature_drift": drift,
        "features_metadata": {
            c: {"tooltip": f"{c} — defined/covered via the v1 feature admission gate."}
            for c in cols
        },
        "feature_coverage": coverage,
        "ensemble": ensemble_rows(result, int(history_df.shape[0])),
        "rolling_brier": r_brier,
        "rolling_brier_meta": rb_meta,
        "brier_baseline": base_brier,
        "brier_baseline_label": base_label,
        "version_history": version_history_rows(moneyline_records, current_date),
    }