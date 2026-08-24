"""
Explainability for MLB Bet Predictor.

Provides per-game SHAP attributions (averaged across ensemble members)
and PSI (Population Stability Index) feature-drift computation.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import (
    DATA_DELIVERY_DIR,
    DATE_FMT,
    FEATURE_DRIFT,
    PSI_ALERT_THRESHOLD,
    PSI_WARN_THRESHOLD,
    RANDOM_SEED,
    SHAP_GAME,
)
from training import (
    FEATURE_COLS,
    TREE_CATEGORICAL_COLS,
    UNK_TEAM_ID,
    _add_team_ids,
    _categorical_matrix,
    _impute_median,
    _tree_dataframe,
)

logger = logging.getLogger(__name__)

_SHAP_XGB_SHIM_APPLIED = False

# Version ranges the shim + per-member SHAP routing were verified against
# (xgboost 3.2 / shap 0.49 with the decode shim active). Outside these
# ranges we still try — the shim is format-tolerant — but loudly.
_SHAP_TESTED_RANGE = ((0, 45), (0, 51))
_XGB_TESTED_RANGE = ((1, 7), (4, 0))


def _major_minor(version: Optional[str]) -> Optional[tuple[int, int]]:
    m = re.match(r"(\d+)\.(\d+)", str(version or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _warn_if_outside_tested_range() -> None:
    """Announce any untested shap/xgboost pairing before it can bite."""
    try:
        from importlib.metadata import version as _pkg_version
        shap_v = _major_minor(_pkg_version("shap"))
        xgb_v = _major_minor(_pkg_version("xgboost"))
    except Exception:
        return
    problems = []
    if shap_v and not (_SHAP_TESTED_RANGE[0] <= shap_v < _SHAP_TESTED_RANGE[1]):
        problems.append(f"shap {shap_v[0]}.{shap_v[1]} outside tested "
                        f"{_SHAP_TESTED_RANGE[0][0]}.{_SHAP_TESTED_RANGE[0][1]}–"
                        f"{_SHAP_TESTED_RANGE[1][0]}.{_SHAP_TESTED_RANGE[1][1]}")
    if xgb_v and not (_XGB_TESTED_RANGE[0] <= xgb_v < _XGB_TESTED_RANGE[1]):
        problems.append(f"xgboost {xgb_v[0]}.{xgb_v[1]} outside tested "
                        f"{_XGB_TESTED_RANGE[0][0]}.{_XGB_TESTED_RANGE[0][1]}–"
                        f"{_XGB_TESTED_RANGE[1][0]}.{_XGB_TESTED_RANGE[1][1]}")
    if problems:
        logger.warning(
            "Untested SHAP stack: %s — verify XGBoost attributions via the "
            "additivity check before trusting game explanations.",
            "; ".join(problems),
        )
    else:
        logger.info("SHAP stack in tested range (shap %s, xgboost %s)",
                    shap_v, xgb_v)


def _ensure_shap_xgb_compat() -> None:
    """Make shap's XGBoost loader parse xgboost ≥2 UBJSON dumps.

    xgboost 2+ serializes ``learner_model_param.base_score`` inside the raw
    UBJ model bytes as a bracketed string (e.g. ``'[5.25E-1]'``). shap's
    ``XGBTreeModelLoader`` does ``float(...)`` on it and crashes with
    ``ValueError: could not convert string to float``, killing every
    XGBoost attribution. Wrap the decoder once to normalize that field;
    idempotent and harmless for other boosters.
    """
    global _SHAP_XGB_SHIM_APPLIED
    if _SHAP_XGB_SHIM_APPLIED:
        return
    _SHAP_XGB_SHIM_APPLIED = True
    try:
        import shap.explainers._tree as st
    except Exception:
        return
    if getattr(st, "_mlb_base_score_shim", False):
        _warn_if_outside_tested_range()
        return
    if not hasattr(st, "decode_ubjson_buffer"):
        # shap internals changed upstream: our hook point is gone. Fail loud
        # here rather than letting every XGBoost attribution die quietly.
        logger.warning(
            "shap.explainers._tree.decode_ubjson_buffer is missing — shap "
            "internals changed; cannot apply the xgboost base_score shim. "
            "XGBoost SHAP will likely fail; pin a tested shap version.")
        _warn_if_outside_tested_range()
        return
    orig = st.decode_ubjson_buffer

    def _decode_fixed(fd):
        jm = orig(fd)
        p = jm.get("learner", {}).get("learner_model_param", {})
        bs = p.get("base_score")
        if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
            try:
                p["base_score"] = float(bs[1:-1])
            except ValueError:
                pass
        return jm

    st.decode_ubjson_buffer = _decode_fixed
    st._mlb_base_score_shim = True
    _warn_if_outside_tested_range()


# ── SHAP per-game attributions ──────────────────────────────────────────────

def _native_xgb_contribs(model: Any, Xin: Any) -> Optional[tuple[np.ndarray, float]]:
    """XGBoost attributions via the booster's NATIVE TreeSHAP.

    booster.predict(pred_contribs=True) is computed by xgboost itself: it
    honors native categorical split semantics exactly and satisfies
    Σφ + bias == margin to machine precision BY CONSTRUCTION. This removes
    shap's Python-side XGBoost tree parser (and its UBJSON base_score
    handling) from the primary attribution path — version drift between the
    resolved xgboost/shap pair showed up as a 2.63e-03 additivity violation
    (vs LightGBM's 1.78e-15) because shap walks categorical splits as plain
    numeric code thresholds while native predict uses category set-membership.
    Falls back to the shap explainer loudly when unavailable.
    """
    try:
        import xgboost as xgb_lib
        booster = model.get_booster()
        dm = xgb_lib.DMatrix(Xin, enable_categorical=True)
        contribs = booster.predict(dm, pred_contribs=True)
        arr = np.asarray(contribs, dtype=float)
        if arr.ndim == 3:      # (n, features+1, classes) — binary edge case
            arr = arr[:, :, -1]
        vec = np.asarray(arr[0, :-1], dtype=float).ravel()
        base = float(arr[0, -1])
        return vec, base
    except Exception as e:
        logger.warning(
            "Native XGBoost pred_contribs failed (%s) — falling back to "
            "shap.TreeExplainer for this member", e)
        return None


def _shap_vector(sv, n_cols: int):
    """Normalize one member's explainer output to a length-n_cols vector.

    Different explainers return different shapes for binary classification:
      - XGBoost/LightGBM: (1, n_features) or list [class0, class1]
      - sklearn RandomForest (recent shap): (1, n_features, 2)
    Averaging raw outputs of mixed shapes is what crashed the pipeline with
    'inhomogeneous shape'. Returns None when the output can't be reconciled.
    """
    if isinstance(sv, (list, tuple)):
        sv = sv[-1]  # binary classifiers: last element = class 1
    arr = np.asarray(sv, dtype=float)
    if arr.ndim == 3:            # (batch, features, classes)
        arr = arr[:, :, -1]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    vec = arr.ravel()
    if vec.size != n_cols:
        # LOUD failure: silent None here is exactly how the 58-vs-60 shape bug
        # slipped through — a member quietly vanished from attributions.
        logger.warning(
            "SHAP size mismatch: explainer returned %d values, expected %d — "
            "member input does not match its fit-time width/dtype; "
            "excluding it from attributions.",
            vec.size, n_cols,
        )
        return None
    return vec

def compute_shap_per_game(
    models: dict[str, Any],
    games: pd.DataFrame,
) -> None:
    """Compute and save SHAP attributions for each game.

    Averages TreeExplainer values across XGBoost and LightGBM members
    in log-odds space. If shap is unavailable, writes zero-attribution CSVs.

    Output: data_delivery/shap_game_<game_id>.csv
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import shap
        _ensure_shap_xgb_compat()
        has_shap = True
    except ImportError:
        has_shap = False
        logger.warning("shap not available; writing zero-attribution CSVs")

    # Full FEATURE_COLS width in canonical order — mirrors the training/
    # predict matrices (see _feature_matrix). A narrower matrix here is what
    # made SHAP attributions come back empty while logs looked healthy.
    missing = [c for c in FEATURE_COLS if c not in games.columns]
    if missing:
        logger.warning(
            "SHAP input: %d/%d expected columns absent (%s%s) — filled as NULL",
            len(missing), len(FEATURE_COLS), ", ".join(missing[:6]),
            " …" if len(missing) > 6 else "")
    cols = list(FEATURE_COLS)
    # Preserve NaN: tree explainers handle missing values natively and a
    # zero-fill would fabricate attributions for unobserved features.
    X = games.reindex(columns=FEATURE_COLS).to_numpy(dtype=float)

    # Tree members were trained on numeric features PLUS team-ID categorical
    # columns (58 + 2 = 60 wide). Feeding them a numeric-only matrix makes
    # LightGBM fatal-error with a shape mismatch and silently misaligns the
    # others. Build per-model inputs mirroring ensemble_predict's routing.
    if not {"home_team_id", "away_team_id"} <= set(games.columns):
        games = _add_team_ids(games)
    X_cat = _categorical_matrix(games)
    medians = models.get("impute_median")
    n_full = X.shape[1] + len(TREE_CATEGORICAL_COLS)

    def _model_input(name: str, i: int):
        """Row-slice input shaped exactly like that member's training matrix."""
        xn = X[i:i + 1]
        xc = X_cat[i:i + 1]
        if name == "xgboost":
            Xi = _impute_median(xn, medians)[0] if medians is not None else xn
            return _tree_dataframe(Xi, xc, cols)
        if name == "lightgbm":
            dfp = pd.DataFrame(xn, columns=cols)
            for j, c in enumerate(TREE_CATEGORICAL_COLS):
                dfp[c] = np.where(xc[:, j] < 0, UNK_TEAM_ID, xc[:, j]).astype(int)
            return dfp
        if name == "randomforest":
            rf = models.get("randomforest")
            n_feat = getattr(rf, "n_features_in_", None)
            if n_feat is not None and n_feat == len(FEATURE_COLS):
                return xn  # ablation RF trained without team IDs
            return np.hstack([xn, xc])
        return xn

    def _member_logit(name: str, model: Any, Xin: Any) -> Optional[float]:
        """Model's raw log-odds for one row (additivity-check target)."""
        if not hasattr(model, "predict_proba"):
            return None
        try:
            p = float(model.predict_proba(Xin)[0, 1])
        except Exception:
            return None
        p = min(max(p, 1e-12), 1 - 1e-12)
        return math.log(p) - math.log(1 - p)

    # Build explainers ONCE (was per-game-per-member) and capture each
    # member's base value for the Σφ + base ≈ log-odds additivity check.
    # XGBoost keeps its shap explainer ONLY as a fallback behind the native
    # pred_contribs path (see _native_xgb_contribs).
    explainers: dict[str, tuple[Any, Optional[float]]] = {}
    if has_shap:
        for name, model in models.items():
            if name in ("scaler", "logistic", "impute_median"):
                continue
            try:
                ex = shap.TreeExplainer(model)
                ev = getattr(ex, "expected_value", None)
                base = float(np.ravel(ev)[-1]) if ev is not None else None
                explainers[name] = (ex, base)
            except Exception as e:
                if name == "xgboost":
                    logger.info("shap TreeExplainer unavailable for xgboost (%s) — native pred_contribs remains primary", e)
                    continue
                logger.warning("TreeExplainer init failed for %s: %s", name, e)

    additivity_diffs: dict[str, list[float]] = {}
    shap_path_used: dict[str, str] = {}
    warned_no_members = False

    for idx, row in games.iterrows():
        game_id = row["game_id"]
        shap_values = {}
        perspective_team = home if (home := row.get("home_team")) else "HOME"

        if has_shap:
            # Collect SHAP from tree-based models
            tree_shaps = []
            for name, model in models.items():
                if name not in explainers:
                    continue
                try:
                    Xin = _model_input(name, idx)
                    sv = None
                    if name == "xgboost":
                        native = _native_xgb_contribs(model, Xin)
                        if native is not None:
                            sv, base = native
                            shap_path_used[name] = "native_pred_contribs"
                        else:
                            sv = None
                    if sv is None and name in explainers:
                        explainer, base = explainers[name]
                        sv = _shap_vector(explainer.shap_values(Xin), n_full)
                        if sv is not None and name == "xgboost":
                            shap_path_used[name] = "shap_TreeExplainer_fallback"
                    if sv is None:
                        continue  # loud logging already happened upstream
                    tree_shaps.append(sv)
                    # End-to-end additivity spot-check on margin-space members:
                    # Σφ + base must reconstruct the model's own log-odds.
                    if idx < 3 and name in ("xgboost", "lightgbm") and base is not None:
                        target = _member_logit(name, model, Xin)
                        if target is not None:
                            additivity_diffs.setdefault(name, []).append(
                                abs(float(sv.sum()) + base - target)
                            )
                except Exception as e:
                    logger.warning("SHAP failed for %s on model %s: %s", game_id, name, e)

            if not tree_shaps and not warned_no_members:
                warned_no_members = True
                logger.warning(
                    "No tree member produced SHAP values — attributions will be "
                    "written as zeros. Check member inputs vs fit-time shapes."
                )

            if tree_shaps:
                avg_shap = np.mean(tree_shaps, axis=0)
                # FAVORED-team perspective: the model outputs P(home win).
                # When the AWAY team is favored (p < 0.5), negate — shap
                # values then describe the favorite's win probability, with
                # positive = pushes the favorite toward winning. Consistent
                # with the calibration page's favored-side view.
                p_home = pd.to_numeric(pd.Series([row.get("home_win_prob_model")]),
                                       errors="coerce").iloc[0]
                if pd.notna(p_home) and float(p_home) < 0.5:
                    avg_shap = -avg_shap
                    away = row.get("away_team")
                    perspective_team = away if isinstance(away, str) and away else "AWAY"
                for i, col in enumerate(cols):
                    shap_values[col] = round(float(avg_shap[i]), 6)
                # Surface team-ID contributions: they carry real ensemble
                # weight, so omitting them would make the importance view lie.
                # IDs sit AFTER numeric cols in every member's input builder.
                for j, tcol in enumerate(TREE_CATEGORICAL_COLS):
                    pos = len(cols) + j
                    if pos < len(avg_shap):
                        shap_values[tcol] = round(float(avg_shap[pos]), 6)
            else:
                for col in cols:
                    shap_values[col] = 0.0
        else:
            for col in cols:
                shap_values[col] = 0.0

        # Sort by absolute SHAP value (descending)
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)

        rows = []
        for feat, val in sorted_features:
            rows.append({
                "feature": feat,
                "shap_value": val,
                "signed_effect": "positive" if val > 0 else "negative",
                "perspective_team": perspective_team,
            })

        df = pd.DataFrame(rows)
        out_path = DATA_DELIVERY_DIR / f"{SHAP_GAME}_{game_id}.csv"
        df.to_csv(out_path, index=False)

    for name, diffs in additivity_diffs.items():
        worst = max(diffs)
        log = logger.info if worst < 1e-4 else logger.warning
        log(
            "SHAP additivity [%s via %s]: worst |Σφ + base − log-odds| = %.2e "
            "over %d games%s",
            name, shap_path_used.get(name, "?"), worst, len(diffs),
            "" if worst < 1e-4 else " — INVESTIGATE input/dtype fidelity",
        )

    logger.info("SHAP attributions written for %d games", len(games))


# ── PSI (Population Stability Index) ────────────────────────────────────────

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))

    Implementation notes (fixed after false-alert investigation):
    - Bin edges are QUANTILES of the combined sample (deduplicated), not
      equal-width slices of the range. Equal-width bins let a single outlier
      stretch the range so most edge bins end up empty on one side.
    - Empty bins are handled with add-one-half smoothing instead of an
      epsilon of 1e-10. The old epsilon made one empty bin contribute ~+2.0
      to PSI by itself, flagging stable features as ALERT.

    Returns a non-negative float. 0 means identical distributions.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    # Remove NaN
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    combined = np.concatenate([baseline, current])
    min_val, max_val = combined.min(), combined.max()

    if min_val == max_val:
        return 0.0

    # Quantile bin edges from the combined sample; deduplicate so sparse or
    # heavily-discrete features don't produce zero-width bins.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(combined, quantiles))
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[-1] = max_val + 1e-10  # Include right edge

    baseline_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    # Add-one-half smoothing keeps empty bins bounded and the term-wise
    # contribution (c - b) * ln(c / b) always >= 0.
    k = len(bin_edges) - 1
    baseline_pct = (baseline_counts + 0.5) / (baseline_counts.sum() + 0.5 * k)
    current_pct = (current_counts + 0.5) / (current_counts.sum() + 0.5 * k)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    return round(max(psi, 0.0), 6)


def psi_status(psi_value: float) -> str:
    """Map PSI value to status: OK, WARN, or ALERT."""
    if psi_value >= PSI_ALERT_THRESHOLD:
        return "ALERT"
    elif psi_value >= PSI_WARN_THRESHOLD:
        return "WARN"
    return "OK"


def psi_noise_floor(n_baseline: int, n_current: int, n_bins: int = 10) -> float:
    """Expected PSI from sampling noise alone when both samples are drawn
    from the SAME distribution.

    For two independent samples the per-bin proportion error is O(1/sqrt(n)),
    giving E[PSI] ≈ (k−1)/2 · (1/n_base + 1/n_cur). At the drift step's
    adjacent-window sizes (~110 vs ~150 games) this is ≈0.07 — most of the
    way to the WARN threshold (0.10). Statuses must therefore be assigned on
    the NOISE-ADJUSTED PSI, or identical distributions page constantly.
    """
    if n_baseline <= 0 or n_current <= 0:
        return 0.0
    return (n_bins - 1) / 2.0 * (1.0 / n_baseline + 1.0 / n_current)


def compute_feature_drift(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    model_weights: dict | None = None,
) -> pd.DataFrame:
    """Compute PSI for each numeric feature and save feature_drift CSV.

    Output: data_delivery/feature_drift_YYYYMMDD.csv
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

    drift_rows = []
    for col in FEATURE_COLS:
        if col not in baseline_games.columns or col not in current_games.columns:
            continue

        baseline_vals = baseline_games[col].dropna().values
        current_vals = current_games[col].dropna().values

        n_b, n_c = len(baseline_vals), len(current_vals)

        if n_b == 0 or n_c == 0:
            drift_rows.append({
                "feature": col,
                "current_mean": round(float(current_vals.mean()), 4) if n_c > 0 else 0.0,
                "baseline_mean": round(float(baseline_vals.mean()), 4) if n_b > 0 else 0.0,
                "psi": 0.0,
                "psi_adjusted": 0.0,
                "noise_floor": 0.0,
                "mean_shift": 0.0,
                "shift_se": 0.0,
                "location_shift": False,
                "status": "INSUFFICIENT",
                "weight_pct": round(float(model_weights.get(col, 0.0)), 3)
                              if model_weights else None,
                "n_baseline": int(n_b),
                "n_current": int(n_c),
            })
            continue

        psi = compute_psi(baseline_vals, current_vals)
        noise = psi_noise_floor(len(baseline_vals), len(current_vals))
        psi_adjusted = max(psi - noise, 0.0)

        # Location gate. PSI responds to ANY distributional change — including
        # pure binning wiggle on heavily-tied/quantized features (win_pct and
        # hardhit% rounded to 2–3 decimals put many teams on one value, so a
        # quantile edge landing inside a tie cluster moves whole teams between
        # bins while the distribution is unchanged). Only escalate above OK
        # when the mean ALSO moved beyond its sampling noise. Games in a
        # 7-day window share teams (~7 starts each), so the naive SE
        # understates true variance — inflate by a clustering factor of 1.5.
        n_b, n_c = len(baseline_vals), len(current_vals)
        if n_b + n_c > 2:
            pooled_sd = np.sqrt(
                ((n_b - 1) * baseline_vals.var(ddof=1)
                 + (n_c - 1) * current_vals.var(ddof=1))
                / (n_b + n_c - 2)
            )
        else:
            pooled_sd = 0.0
        mean_shift = float(current_vals.mean() - baseline_vals.mean())
        if pooled_sd > 0:
            shift_se = float(pooled_sd * np.sqrt(1.0 / n_b + 1.0 / n_c) * 1.5)
            location_shift = abs(mean_shift) > 2.0 * shift_se
        else:
            shift_se = 0.0
            location_shift = psi_adjusted > 0  # degenerate: fall back to PSI

        # Small windows make PSI statistically meaningless — report them as
        # INSUFFICIENT rather than WARN/ALERT so they never page anyone.
        if n_b < 100 or n_c < 30:
            status = "INSUFFICIENT"
        else:
            # Threshold on the NOISE-ADJUSTED PSI: raw PSI between two
            # same-distribution samples of this size already averages ~0.07,
            # so near-equal means with raw PSI 0.10–0.30 are binning wiggle,
            # not regime change. Raw PSI stays in the CSV for transparency.
            status = psi_status(psi_adjusted) if location_shift else "OK"

        drift_rows.append({
            "feature": col,
            "current_mean": round(float(current_vals.mean()), 4),
            "baseline_mean": round(float(baseline_vals.mean()), 4),
            "psi": psi,
            "psi_adjusted": round(psi_adjusted, 6),
            "noise_floor": round(noise, 6),
            "mean_shift": round(mean_shift, 6),
            "shift_se": round(shift_se, 6),
            "location_shift": bool(location_shift),
            "status": status,
            "weight_pct": round(float(model_weights.get(col, 0.0)), 3)
                          if model_weights else None,
            "n_baseline": int(n_b),
            "n_current": int(n_c),
        })

    df = pd.DataFrame(drift_rows)
    out_path = DATA_DELIVERY_DIR / f"{FEATURE_DRIFT}_{target_date_str}.csv"
    df.to_csv(out_path, index=False)

    n_warns = (df["status"] == "WARN").sum()
    n_alerts = (df["status"] == "ALERT").sum()
    logger.info(
        "Feature drift: %d features, %d warnings, %d alerts "
        "(statuses on noise-adjusted PSI; mean noise floor %.3f)",
        len(df), n_warns, n_alerts,
        float(df["noise_floor"].mean()) if "noise_floor" in df.columns else float("nan"),
    )

    return df


def compute_feature_coverage(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
) -> pd.DataFrame:
    """Per-feature non-null coverage per drift window → coverage CSV.

    This is the visual backstop for the 'healthy logs, empty data' bug class:
    a fetcher can silently starve a season (the 2026 weather truncation did
    exactly that) while PSI rows show plausible-looking zeros. This table
    makes absence visible per feature × window.

    For the two weather-driven features it also separates MEASURED values
    from DEFAULT-filled ones, so legitimate zeros can't mask starvation:
      * wind_advantage_flyball_factor: the dome branch writes an exact 0.0
        without any observation — counted as ``default_zero``. Any other
        non-null value came from a fetched record (a real dome observation
        is wm×era ≈ tiny-but-nonzero float, so exact 0.0 + dome is a
        reliable default signature).
      * air_density_velocity_boost: only ever written from a fetched
        observation (domes stay NULL) — non-null ⇒ measured.
    All other features: non-null is reported as measured.
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

    def _window_rows(games: pd.DataFrame, window: str) -> list[dict]:
        rows: list[dict] = []
        if games is None or games.empty:
            return rows
        dome = pd.to_numeric(games.get("dome_is_neutral"), errors="coerce") \
            if "dome_is_neutral" in games.columns else pd.Series(np.nan, index=games.index)
        for col in FEATURE_COLS:
            if col not in games.columns:
                continue
            vals = pd.to_numeric(games[col], errors="coerce")
            n_total = int(len(vals))
            nonnull = vals.notna()
            n_nonnull = int(nonnull.sum())
            n_default = 0
            if col == "wind_advantage_flyball_factor":
                n_default = int((nonnull & (vals == 0.0) & (dome == 1)).sum())
            n_measured = n_nonnull - n_default
            pct_nonnull = round(100.0 * n_nonnull / n_total, 1) if n_total else 0.0
            pct_measured = round(100.0 * n_measured / n_total, 1) if n_total else 0.0
            status = "OK" if pct_measured >= 80.0 else (
                "LOW_COVERAGE" if pct_measured >= 25.0 else "STARVED")
            rows.append({
                "feature": col,
                "window": window,
                "n_games": n_total,
                "n_nonnull": n_nonnull,
                "pct_nonnull": pct_nonnull,
                "n_measured": n_measured,
                "pct_measured": pct_measured,
                "n_default_zero": n_default,
                "status": status,
            })
        return rows

    cov_rows = _window_rows(current_games, "current")
    cov_rows += _window_rows(baseline_games, "baseline")
    df = pd.DataFrame(cov_rows)
    out_path = DATA_DELIVERY_DIR / f"feature_coverage_{target_date_str}.csv"
    df.to_csv(out_path, index=False)

    starved = df[df["status"] != "OK"]
    if not starved.empty:
        worst = starved.sort_values("pct_measured").head(5)
        detail = "; ".join(
            f"{r.feature}/{r.window}={r.pct_measured:.0f}% measured"
            for r in worst.itertuples())
        logger.warning("Feature coverage gaps: %s", detail)
    else:
        logger.info("Feature coverage: all %d feature-window pairs OK", len(df))
    return df
