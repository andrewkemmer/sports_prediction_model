"""NFL Model & Data Drift Monitor emitter — produces an MLB-shaped
``nfl_model_monitor_<date>.json`` so the shared Streamlit ``model_monitor.py``
page (the exact one MLB renders) can run for NFL unchanged.

The page contract (mirrors ``mlb-backend``'s ``model_monitor_*.json``):

``last_retrained`` / ``next_retrain`` + notes  — health cards
``upset_note``                                — upset-monitoring callout
``feature_drift[]``                           — TRUE PSI drift matrix
      [{feature, current_mean, baseline_mean, psi, status,
        weight_pct, n_baseline, n_current}]
         weight_pct = the MODEL WEIGHT column (blend-weighted importances,
         percent of the final blended ensemble, sums to ~100 across rows) —
         the SAME algorithm MLB ships (feature_importance_weights below,
         verbatim semantics of mlb-backend/backend/training.py).
``features_metadata{}``                       — feature -> {tooltip, definition}
                                                for hover + description
``feature_coverage[]``                        — non-null/measured per window
      [{feature, window, n_games, pct_measured, pct_nonnull, n_default_zero, status}]
``ensemble[]``                                — per-member weight + pooled metrics
``rolling_brier[]`` + ``rolling_brier_meta``  — rolling mean-Brier timeline
``brier_baseline``/``brier_baseline_label``   — dashed reference line
``version_history[]``                         — one row per dated moneyline v1 record

All functions are PURE (no I/O, no Streamlit) so they are unit-testable in
isolation; ``build_model_monitor`` composes them from the objects a moneyline
run already has in memory (the decided feature frame, the walk-forward result
— including the deployed re-fit's model objects under ``result["_models"]`` —
the per-game prediction history, and prior dated moneyline records).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Blend-weight constants for the MODEL WEIGHT column — the NFL ensemble is
# the same 5-member roster as MLB's, so the fallback/floor semantics are
# byte-identical (see _member_weight_share). Imported from nfl_moneyline for
# a single source of truth; nfl_moneyline only imports nfl_monitor lazily
# inside functions, so there is no circular import.
from nfl_moneyline import ADAPTIVE_WEIGHT_FLOOR, ENSEMBLE_WEIGHTS

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


def expand_drift_cut(gd: pd.Series, cut,
                     min_samples: int = MIN_DRIFT_SAMPLES):
    """Expand the 'current' window's lower bound ``cut`` backward so the
    [cut, latest] decided-game window holds at least ``min_samples`` games.

    The MLB drift window is a rolling 30 *calendar days*, which is always a
    healthy sample in a daily sport. NFL games cluster inside the season and
    the last-30-days can collapse to the few post-season games (~13), dropping
    every row to INSUFFICIENT. This pulls in strictly-earlier gamedays (prior
    season) until the floor is met, keeping the 30-day rolling anchor while
    guaranteeing a judgeable window. Unchanged when the window already clears
    the floor; passthrough when ``cut``/``gd`` cannot decide it."""
    g = pd.to_datetime(gd, errors="coerce")
    latest = g.max()
    if pd.isna(cut) or pd.isna(latest):
        return cut
    cut = pd.to_datetime(cut)
    n_cur = int(((g >= cut) & (g <= latest)).sum())
    if n_cur >= min_samples:
        return cut
    # Strictly-earlier gamedays, newest first; step back one game at a time
    # until the current window reaches the sample floor (or prior games run out).
    earlier = sorted(g[g.notna() & (g < cut)].unique(), reverse=True)
    if not earlier:
        return cut
    for cand in earlier:
        if int((g >= cand).sum()) >= min_samples:
            return cand
    return min(earlier)


def feature_drift_rows(feats: pd.DataFrame, columns: list[str],
                       baseline_mask: np.ndarray, current_mask: np.ndarray,
                       weight: dict[str, float] | None = None) -> list[dict]:
    """One row per feature: current/baseline mean, true PSI, status, sample
    sizes. ``weight`` (optional feature -> PERCENT share of the blended
    ensemble, summing to ~100 — the ``feature_importance_weights`` output) is
    surfaced as ``weight_pct`` rounded to 3 decimals, byte-matching MLB's own
    ``weight_pct`` row field. When omitted (no model objects available) the
    page hides the MODEL WEIGHT column, exactly like MLB with ``None``."""
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
            "weight_pct": (round(float(weight[col]), 3) if weight and col in weight else None),
            "n_baseline": int(len(base)),
            "n_current": int(len(cur)),
        })
    return rows


# ---------------------------------------------------------------------------
# Model weight — the MLB MODEL WEIGHT column (blend-weighted importances)
# ---------------------------------------------------------------------------
def _member_weight_share(adaptive_weights: dict | None,
                         member_names: list[str]) -> dict[str, float]:
    """Normalized blend weights for the members that actually trained.

    MLB-identical to ``mlb-backend/backend/training.py::_member_weights``:
    adaptive weights earned from pooled OOF AUC when available; static
    ENSEMBLE_WEIGHTS priors otherwise; a member with no earned weight still
    gets its static prior (capped at 2x the adaptive floor) so it can prove
    itself next cycle; survivors renormalize to exactly 1.0."""
    names = [n for n in member_names
             if n not in ("scaler", "impute_median", "categorical_vocab")]
    if not names:
        return {}
    source = adaptive_weights if adaptive_weights else ENSEMBLE_WEIGHTS
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


def feature_importance_weights(models: dict | None, feature_cols: list[str],
                               adaptive_weights: dict | None = None) -> dict[str, float] | None:
    """Blend-weighted feature importance across ensemble members (sums to 100).

    VERBATIM semantics of ``mlb-backend/backend/training.py::
    feature_importance_weights`` — the exact algorithm behind MLB's MODEL
    WEIGHT column, replicated line-for-line (not approximated):

    * per member, importances = split-gain ``feature_importances_`` for the
      tree members (XGB/LGB/RF) and |coefficient| for logistic — trimmed to
      the numeric feature columns only (the team-ID categoricals the trees
      are fit with sit AFTER the numeric columns and carry no served-pool
      interpretation on the drift table, matching MLB);
    * each member's importances are normalized internally;
    * the members' normalized vectors are averaged with their blend-weight
      share (adaptive when present, static priors otherwise);
    * the result answers "what fraction of the final blended model rides on
      this feature?" and sums to 100.

    This is NOT per-game SHAP. A member with neither ``feature_importances_``
    nor ``coef_`` (the MLP) contributes nothing and the remaining members
    renormalize to 100 — MLB's exact MLP handling. Returns None when no
    member exposes importances (the page then hides the column, as MLB does).
    """
    members = {n: m for n, m in (models or {}).items()
               if n not in ("scaler", "impute_median", "categorical_vocab")}
    if not members:
        return None
    eff = _member_weight_share(adaptive_weights, list(members.keys()))
    raw = {n: float(eff.get(n, 0.0)) for n in members}
    total = sum(raw.values())
    if total <= 0:
        raw = {n: 1.0 / len(members) for n in members}
        total = 1.0

    nfc = len(feature_cols)
    agg = np.zeros(nfc)
    contributed = False
    for name, model in members.items():
        try:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float).ravel()
            elif hasattr(model, "coef_"):
                imp = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
            else:
                continue   # MLP et al. — no importances surface
        except Exception:
            continue
        if len(imp) >= nfc:
            imp = imp[:nfc]
        if len(imp) != nfc or imp.sum() <= 0:
            continue
        agg += (raw[name] / total) * (imp / imp.sum())
        contributed = True
    if not contributed or agg.sum() <= 0:
        return None
    return {f: round(float(w), 4) for f, w in zip(feature_cols, agg / agg.sum() * 100.0)}


def feature_metadata_map(columns: list[str],
                         descriptions: dict | None = None) -> dict[str, dict]:
    """Per-feature description map for the drift table's hover + label.

    MLB shape: ``{feature: {definition, source, tooltip}}`` where the
    page renders ``tooltip`` and falls back to the raw name when the backend
    has no description for a feature. ``descriptions`` comes from the NFL
    feature builder's CANONICAL_SOURCE one-liners when a run provides them."""
    meta: dict[str, dict] = {}
    for col in columns:
        desc = (descriptions or {}).get(col) or col
        meta[col] = {
            "definition": desc,
            "source": "nfl feature engine (strictly-trailing per-team aggregates)",
            "tooltip": (f"What: {desc}\n"
                        f"Consumed by: the 5-member moneyline blend "
                        f"(see Model Ensemble)."),
        }
    return meta


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
    """The deployed Platt map, MLB-shaped: the "calibration" row field is
    OMITTED entirely when the record has no params (the renderer then shows
    'identity/none'). Emitting "a": null / "b": null instead crashes the
    shared version-history table (``float(None)``) — MLB never emits it, so
    neither does NFL."""
    params = calibration.get("params") if calibration else None
    if params and params.get("a") is not None and params.get("b") is not None:
        return {"a": params["a"], "b": params["b"]}
    return {}


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
        row = {
            "version": rec.get("_date") or current_date,
            "date": str(rec.get("date") or rec.get("_date") or current_date),
            "weights": rec.get("adaptive_weights") or {},
            "auc": plt.get("auc"), "logloss": plt.get("logloss"),
            "ecer_calibrated": plt.get("ece"),
            "ece_calibrated": plt.get("ece"),
            "adopt": verdict.get("adopt"),
        }
        cal_map = _platt_ab(cal)
        if cal_map:
            row["calibration"] = cal_map
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------
def build_model_monitor(*, feats: pd.DataFrame, result: dict,
                        history_df: pd.DataFrame, calibration: dict,
                        moneyline_records: list[dict], current_date: str,
                        baseline_cut_date: str,
                        feature_descriptions: dict | None = None) -> dict:
    """Compose the MLB-shaped monitor record.

    ``feats``            : decided game frame (with ``gameday`` + model columns)
    ``result``           : ``run_walk_forward`` result — carries the deployed
                           re-fit model objects under ``result["_models"]``
                           (MLB's ``best_models`` analog) plus members /
                           adaptive weights / verdict / trained_at. When the
                           models are absent the MODEL WEIGHT column is
                           omitted (weight_pct None), matching MLB's None path.
    ``history_df``       : per-game OOF+sealed prediction history
    ``calibration``      : ``nfl_calibration_*.json`` (Platt a/b)
    ``moneyline_records``: prior dated nfl_moneyline_v1 records for the table
    ``current_date``     : YYYYMMDD
    ``baseline_cut_date``: ISO date; games >= it are the 'current' drift window
    ``feature_descriptions``: {feature: one-line description} for the drift
                           table's description/hover (CANONICAL_SOURCE from
                           the feature builder when a run provides it)
    """
    cols = list(result.get("_deployed", {}).get("features", []))
    c_iso = f"{current_date[:4]}-{current_date[4:6]}-{current_date[6:8]}"

    gd = pd.to_datetime(feats["gameday"], errors="coerce")
    # MLB-style 30-day rolling boundary, widened backward into the prior
    # season when the last-30-days slice holds fewer than MIN_DRIFT_SAMPLES
    # decided games (so a <30-game post-season tail never drops to INSUFFICIENT).
    cut = expand_drift_cut(gd, baseline_cut_date)
    current_mask = gd.notna() & (gd >= cut)
    baseline_mask = gd.notna() & (gd < cut)
    if not current_mask.any():                       # fallback: newest games
        newest = gd.max()
        current_mask = gd.notna() & (gd == newest)
    if not baseline_mask.any():                      # fallback: everything before
        baseline_mask = gd.notna() & (gd < gd.max())

    # MODEL WEIGHT column: MLB-identical blend-weighted importances from the
    # deployed re-fit (models_sealed) blended with the run's adaptive weights.
    weights = feature_importance_weights(
        result.get("_models"), cols,
        adaptive_weights=result.get("adaptive_weights"))
    drift = feature_drift_rows(feats, cols, baseline_mask.to_numpy(),
                               current_mask.to_numpy(), weight=weights)
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
        "features_metadata": feature_metadata_map(cols, feature_descriptions),
        "feature_coverage": coverage,
        "ensemble": ensemble_rows(result, int(history_df.shape[0])),
        "rolling_brier": r_brier,
        "rolling_brier_meta": rb_meta,
        "brier_baseline": base_brier,
        "brier_baseline_label": base_label,
        "version_history": version_history_rows(moneyline_records, current_date),
    }