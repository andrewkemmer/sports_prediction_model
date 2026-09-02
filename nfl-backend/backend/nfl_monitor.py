"""NFL Model & Data Drift Monitor emitter — produces an MLB-shaped
``nfl_model_monitor_<date>.json`` so the shared Streamlit ``model_monitor.py``
page (the exact one MLB renders) can run for NFL unchanged.

The page contract (mirrors ``mlb-backend``'s ``model_monitor_*.json``):

``last_retrained`` / ``next_retrain`` + notes  — health cards
``upset_note``                                — upset-monitoring callout
``feature_drift[]``                           — TRUE PSI drift matrix, MLB-
                                                identical row fields:
      [{feature, current_mean, baseline_mean, psi, psi_adjusted,
        noise_floor, mean_shift, shift_se, location_shift, status,
        weight_pct, n_baseline, n_current}]
         status = NOISE-ADJUSTED psi + location-gate rule (MLB's
         compute_feature_drift verbatim: INSUFFICIENT when n_b<100 or
         n_c<30, else psi_status(psi_adjusted) only when the mean also
         moved — identical means with large raw PSI stay OK);
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

from datetime import datetime

import numpy as np
import pandas as pd

# Blend-weight constants for the MODEL WEIGHT column — the NFL ensemble is
# the same 5-member roster as MLB's, so the fallback/floor semantics are
# byte-identical (see _member_weight_share). Imported from nfl_moneyline for
# a single source of truth; nfl_moneyline only imports nfl_monitor lazily
# inside functions, so there is no circular import.
from nfl_moneyline import ADAPTIVE_WEIGHT_FLOOR, ENSEMBLE_WEIGHTS

# Drift / coverage gates (mirror MLB's semantics). The 0.10/0.25 thresholds
# apply to the NOISE-ADJUSTED psi (config.py PSI_WARN_THRESHOLD /
# PSI_ALERT_THRESHOLD) — raw PSI at these sample sizes averages ~0.07 from
# binning alone, so raw-threshold statuses constantly page (the 09-02
# artifact's pace_plays_min_diff PSI 2.038 on identical means).
PSI_WARN = 0.10
PSI_ALERT = 0.25
MIN_DRIFT_SAMPLES = 30        # current-window floor (expand_drift_cut)
COVERAGE_FLOOR = 0.80         # below CURRENT window -> LOW_COVERAGE / STARVED
# "Next expected run" heuristic for the NEXT RETRAIN card (retrain-every-
# run decision 2026-09-02, synced to MLB's corrected cadence): the ensemble
# is persisted on EVERY pipeline run, so the next expected run is ~1 day out.
# NOT scheduler-backed (no cron/next_run exists) — card text only. This is
# NOT the fold cadence (NFL folds are weekly via generate_weekly_folds).
RETRAIN_INTERVAL_DAYS = 1

# The 5-member moneyline walk-forward roster the monitor page describes verbatim.
MEMBER_DESC = {
    "xgboost": "Gradient-boosted decision trees (logloss, early-stopped per fold).",
    "lightgbm": "Leaf-wise histogram gradient boosting, routes missing values natively.",
    "logistic": "L2-regularized linear anchor; keeps the blend calibrated when trees overfit.",
    "randomforest": "Bagged decision trees; errors decorrelate from the boosted members.",
    "mlp": "Small low-capacity net that picks up smooth nonlinearities trees split around.",
}


# ---------------------------------------------------------------------------
# PSI (Population Stability Index) — MLB-identical replication
# ---------------------------------------------------------------------------
# Verbatim semantics of mlb-backend/backend/explainability.py::compute_psi /
# psi_status / psi_noise_floor and the compute_feature_drift status rule:
# quantile bins over the COMBINED sample, add-one-half smoothing, status on
# the NOISE-ADJUSTED psi AND only when the mean ALSO moved (location gate).
def psi_score(baseline: np.ndarray, current: np.ndarray,
              bins: int = 10) -> float:
    """True PSI, byte-identical to MLB's compute_psi: quantile bin edges of
    the COMBINED (deduplicated) sample, add-one-half smoothing instead of an
    epsilon, PSI = sum((c - e) * ln(c / e)) clipped at 0 and rounded 6.
    Constant/singular features return 0.0 (no shift measurable)."""
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    b = b[~np.isnan(b)]
    c = c[~np.isnan(c)]
    if len(b) == 0 or len(c) == 0:
        return 0.0
    combined = np.concatenate([b, c])
    lo, hi = float(combined.min()), float(combined.max())
    if lo == hi:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(combined, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[-1] = hi + 1e-10                    # include the right edge
    baseline_counts = np.histogram(b, bins=edges)[0].astype(float)
    current_counts = np.histogram(c, bins=edges)[0].astype(float)
    k = len(edges) - 1
    e = (baseline_counts + 0.5) / (baseline_counts.sum() + 0.5 * k)
    a = (current_counts + 0.5) / (current_counts.sum() + 0.5 * k)
    return round(max(float(np.sum((a - e) * np.log(a / e))), 0.0), 6)


def psi_status(psi_value: float) -> str:
    """OK / WARN / ALERT on the 0.10 / 0.25 thresholds (MLB's psi_status) —
    applied by the monitor to the NOISE-ADJUSTED psi."""
    if psi_value >= PSI_ALERT:
        return "ALERT"
    if psi_value >= PSI_WARN:
        return "WARN"
    return "OK"


def psi_noise_floor(n_baseline: int, n_current: int, n_bins: int = 10) -> float:
    """Expected PSI from sampling noise alone when both samples are drawn
    from the SAME distribution ((k-1)/2 * (1/n_base + 1/n_cur), MLB's
    psi_noise_floor verbatim). At NFL's drift-window sizes (~1,930 baseline /
    ~30 current) the floor is ~0.015 — identical distributions still page as
    non-zero raw PSI, so statuses must read the adjusted value."""
    if n_baseline <= 0 or n_current <= 0:
        return 0.0
    return (n_bins - 1) / 2.0 * (1.0 / n_baseline + 1.0 / n_current)


def noise_adjusted_drift(base: np.ndarray, cur: np.ndarray) -> dict:
    """MLB's per-feature noise-adjusted drift math (compute_feature_drift):
    raw psi, the sampling noise floor, psi_adjusted = max(psi - noise, 0),
    and the location gate — mean_shift judged against 2x shift_se, where
    shift_se = pooled_sd * sqrt(1/n_b + 1/n_c) * 1.5 (the 1.5 clustering
    inflation for ~7-start teams in a short window). Degenerate (zero pooled
    sd) falls back to location_shift = psi_adjusted > 0, exactly like MLB.
    """
    base = base[np.isfinite(base)]
    cur = cur[np.isfinite(cur)]
    nb, nc = int(len(base)), int(len(cur))
    psi = psi_score(base, cur)
    noise = psi_noise_floor(nb, nc)
    psi_adjusted = max(psi - noise, 0.0)
    mean_shift = float(np.mean(cur) - np.mean(base)) if nb and nc else 0.0
    if nb + nc > 2:
        pooled_sd = float(np.sqrt(
            ((nb - 1) * np.var(base, ddof=1) + (nc - 1) * np.var(cur, ddof=1))
            / (nb + nc - 2)))
    else:
        pooled_sd = 0.0
    if pooled_sd > 0:
        shift_se = float(pooled_sd * np.sqrt(1.0 / nb + 1.0 / nc) * 1.5)
        location_shift = abs(mean_shift) > 2.0 * shift_se
    else:
        shift_se = 0.0
        location_shift = psi_adjusted > 0
    return {
        "psi": psi, "noise_floor": noise, "psi_adjusted": psi_adjusted,
        "mean_shift": mean_shift, "shift_se": shift_se,
        "location_shift": bool(location_shift),
    }


def adjusted_status(psi_adjusted: float, location_shift: bool,
                    n_baseline: int, n_current: int) -> str:
    """MLB's compute_feature_drift status rule, verbatim: INSUFFICIENT when
    either window is below the sample floors (baseline 100, current 30 — raw
    PSI is meaningless at those sizes and must never page); otherwise the
    status is read from the NOISE-ADJUSTED psi AND a real mean location
    shift. Near-identical means with a large raw PSI (bin-boundary
    instability on tiny-scale near-constant features) therefore stay OK."""
    if n_baseline < 100 or n_current < 30:
        return "INSUFFICIENT"
    if not location_shift:
        return "OK"
    return psi_status(psi_adjusted)


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
    """One row per feature — the MLB drift-row field set, byte-for-byte:
    ``{feature, current_mean, baseline_mean, psi (raw), psi_adjusted,
    noise_floor, mean_shift, shift_se, location_shift, status, weight_pct,
    n_baseline, n_current}``. Status uses the noise-adjusted + location-gate
    rule (adjusted_status); ``current_mean``/``baseline_mean`` are 0.0 for
    an empty window and ``weight_pct`` mirrors MLB's percent, 3dp. When
    ``weight`` is None the page hides the MODEL WEIGHT column, exactly like
    MLB."""
    rows = []
    for col in columns:
        if col not in feats.columns:
            continue
        ser = pd.to_numeric(feats[col], errors="coerce")
        base = ser[baseline_mask].to_numpy()
        cur = ser[current_mask].to_numpy()
        base = base[np.isfinite(base)]
        cur = cur[np.isfinite(cur)]
        nb, nc = int(len(base)), int(len(cur))
        drift = noise_adjusted_drift(base, cur)
        rows.append({
            "feature": col,
            "current_mean": (round(float(np.mean(cur)), 4) if nc else 0.0),
            "baseline_mean": (round(float(np.mean(base)), 4) if nb else 0.0),
            "psi": drift["psi"],
            "psi_adjusted": round(drift["psi_adjusted"], 6),
            "noise_floor": round(drift["noise_floor"], 6),
            "mean_shift": round(drift["mean_shift"], 6),
            "shift_se": round(drift["shift_se"], 6),
            "location_shift": drift["location_shift"],
            "status": adjusted_status(drift["psi_adjusted"],
                                       drift["location_shift"], nb, nc),
            "weight_pct": (round(float(weight[col]), 3) if weight and col in weight else None),
            "n_baseline": nb,
            "n_current": nc,
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

    # LAST RETRAIN = the run's persist timestamp of the served ensemble; the
    # walk-forward result carries ``trained_at``, and when it is missing the
    # emitter falls back to now() — the identical MLB fallback
    # (pipeline.py::_model_monitor_json: ``last_retrained or now()``), so the
    # card shows a real date (this run -> today), never "—".
    trained = result.get("trained_at") or datetime.now().isoformat()
    last_retrained = str(trained)[:10]
    # NEXT RETRAIN = the next EXPECTED run, anchored at EMISSION time (now),
    # not the artifact target date — byte-matching MLB's corrected emitter so
    # both sports share (last=today, next=tomorrow) semantics.
    next_retrain = (datetime.now()
                    + pd.Timedelta(days=RETRAIN_INTERVAL_DAYS)).strftime("%Y-%m-%d")

    return {
        "last_retrained": last_retrained,
        "last_retrained_note": "Fresh model trained this run (sealed gate: "
                               f"{'ADOPT' if (result.get('verdict') or {}).get('adopt') else 'DO NOT ADOPT (reporting only)'})",
        "next_retrain": next_retrain,
        "next_retrain_note": f"next expected run in {RETRAIN_INTERVAL_DAYS} day(s) (retrains every run)",
        # Upset pool = the per-game walk-forward HISTORY (pooled OOF + sealed
        # 2025) — len(history_df) == pooled n + sealed n == the
        # nfl_predictions_history_*.csv row count (1,392 on the 09-02 frame),
        # NOT the decided pool (1,960) or the baseline window (1,930).
        "upset_note": ("Model upset rate over the walk-forward history "
                       f"(pooled OOF + sealed) — {len(history_df):,} games "
                       "scored; see Calibration for the upset strip."),

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