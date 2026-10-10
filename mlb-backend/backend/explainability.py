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
    DRIFT_PHASE_EXTENSION_MONTHS,
    FEATURE_DRIFT,
    PSI_ALERT_THRESHOLD,
    PSI_WARN_THRESHOLD,
    RANDOM_SEED,
    SHAP_GAME,
)
from training import (
    active_moneyline_feature_cols,
    TREE_CATEGORICAL_COLS,
    UNK_TEAM_ID,
    _add_team_ids,
    _categorical_matrix,
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

    # Full active-subset width in canonical order — mirrors the training/
    # predict matrices (see _feature_matrix). A narrower matrix here is what
    # made SHAP attributions come back empty while logs looked healthy.
    # SINGLE-LIST RULE: the served model's width (adopted RFE subset when
    # applied, else the full universe) is the ONLY enumeration here — the
    # matrices below must never be built from a different list than the
    # warning above announces (pre-2026-09-17 the universe was used while
    # the warning described the active subset, misaligning attributions
    # whenever the two widths diverged).
    active = active_moneyline_feature_cols()
    missing = [c for c in active if c not in games.columns]
    if missing:
        logger.warning(
            "SHAP input: %d/%d expected columns absent (%s%s) — filled as NULL",
            len(missing), len(active), ", ".join(missing[:6]),
            " …" if len(missing) > 6 else "")
    cols = list(active)
    # Preserve NaN: tree explainers handle missing values natively and a
    # zero-fill would fabricate attributions for unobserved features.
    X = games.reindex(columns=cols).to_numpy(dtype=float)

    # Tree members were trained on numeric features PLUS team-ID categorical
    # columns (58 + 2 = 60 wide). Feeding them a numeric-only matrix makes
    # LightGBM fatal-error with a shape mismatch and silently misaligns the
    # others. Build per-model inputs mirroring ensemble_predict's routing.
    if not {"home_team_id", "away_team_id"} <= set(games.columns):
        games = _add_team_ids(games)
    X_cat = _categorical_matrix(games)
    n_full = X.shape[1] + len(TREE_CATEGORICAL_COLS)

    def _model_input(name: str, i: int):
        """Row-slice input shaped exactly like that member's training matrix."""
        xn = X[i:i + 1]
        xc = X_cat[i:i + 1]
        if name == "xgboost":
            # Raw NaN row — the representation the XGB member is
            # fit AND served with (training.py routes the raw
            # frame to xgboost, NHL member_matrix parity); the
            # imputed row would misattribute the imputed cells.
            return _tree_dataframe(xn, xc, cols)
        if name == "lightgbm":
            dfp = pd.DataFrame(xn, columns=cols)
            for j, c in enumerate(TREE_CATEGORICAL_COLS):
                dfp[c] = np.where(xc[:, j] < 0, UNK_TEAM_ID, xc[:, j]).astype(int)
            return dfp
        if name == "randomforest":
            rf = models.get("randomforest")
            n_feat = getattr(rf, "n_features_in_", None)
            if n_feat is not None and n_feat == len(cols):
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
            # TreeExplainer only supports the tree members.  The linear
            # ensemble member and persisted metadata are intentionally not
            # explainable members; attempting them only emits misleading
            # "unsupported model" warnings and does not affect the XGB/LGBM
            # attribution paths below.
            if name in ("scaler", "logistic", "impute_median",
                        "categorical_vocab", "elasticnet"):
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
                        if sv is not None:
                            # Record the path for EVERY member, not just
                            # xgboost. lightgbm reaches TreeExplainer through
                            # this same branch (the `name == "xgboost"` gate
                            # above only routes the xgboost-native attempt),
                            # so gating the assignment on xgboost left
                            # shap_path_used empty for lightgbm and the
                            # additivity summary printed "via ?" on every
                            # run -- the label could never resolve.
                            # xgboost keeps its two historical labels; for
                            # every other member TreeExplainer IS the primary
                            # path, not a fallback.
                            shap_path_used[name] = (
                                "shap_TreeExplainer_fallback" if name == "xgboost"
                                else "shap_TreeExplainer")
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


def classify_drift_retention(gain: float, psi_adjusted: float,
                             noise_floor: float,
                             kept_median_gain: float) -> bool:
    """Flag a CULLED feature for retention (false-positive-cull test).

    Historical: the run engine's feature cull WAS a static name rule (the
    removed derive_run_features: *_diff except the restored matchup gaps and
    park_factor_slug_diff, the RUN_EXTRA_EXCLUSIONS composites,
    *_delta_home/away); since the 2026-08-30 gap restore the served view has
    been the FULL active moneyline list and the rule was deleted 2026-09-27.
    A feature is a FALSE-POSITIVE CULL when it was dropped by the historical
    rule yet (a) its gain importance is at least the kept-view median and
    (b) its noise-adjusted PSI is at or below its sampling noise floor — the
    cull removed predictive signal with no measured distributional change.
    Returns True → recommend retention. Both thresholds are INCLUSIVE (gain
    == median, psi_adjusted == noise_floor flag) so a borderline feature is
    never silently dropped.

    Not called by production monitors — selection is call-time resolution of
    the active moneyline list; this is the retention-backstop policy the
    cull diagnostic (and its fixture tests) pins.
    """
    return bool(gain >= kept_median_gain and psi_adjusted <= noise_floor)


def compute_feature_drift(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    model_weights: dict | None = None,
    feature_cols: Optional[list[str]] = None,
    out_name: Optional[str] = None,
    phase_frame: Optional[pd.DataFrame] = None,
    view: str = "moneyline",
) -> pd.DataFrame:
    """Compute PSI for each numeric feature and save feature_drift CSV.

    Output: data_delivery/feature_drift_YYYYMMDD.csv (or ``out_name``).
    ``feature_cols`` narrows the feature view (the run engine shares the
    moneyline contract through the same machinery on the same windows); the default
    enumerates the ACTIVE moneyline serving width (adopted RFE subset,
    else the universe) — SINGLE-LIST RULE: every monitor-facing surface
    reads exactly one list.

    ``view`` labels which model's monitoring surface produced this row
    set — the moneyline and run-engine wrappers run the SAME function on
    the SAME windows, and their unlabeled log lines were
    indistinguishable twins (2026-10-05 log review: "Feature drift: 109
    features..." printed twice with no way to tell which view reported).

    ``phase_frame`` (season-seam guard, 2026-09-30): the frame the
    prior-season phase windows are pulled from. MUST be the full decided
    frame — the production baseline is only the trailing ~250 games and
    contains no prior-season rows, so without this the OK-SEASONAL
    re-check could never fire. Callers that omit it get the old
    baseline-only behavior (the re-check simply finds no phase rows).
    """
    def _phase_matched_baseline(bgames: pd.DataFrame, cgames: pd.DataFrame,
                                col: str, months_back: tuple[int, ...]) -> list:
        """Prior-season same-calendar-phase values for one feature.        Current window's span = [min game_date, max game_date] of
        ``cgames``, padded ±7 calendar days (a one-week phase tolerance:
        the shifted window must hold enough rows for a stable mean-shift
        SE — the exact-span window can dip near the 100-row judge floor
        and flip borderline seasonal features back to ALERT, which the
        2026-09-29 bullpen_kbb_10g_diff replay demonstrated). For each k in
        ``months_back`` (negative ints), take bgames rows whose game_date
        falls in the padded same-phase window shifted k years — the
        season seam moves WITH the calendar instead of across it. Only
        rows already present in the drift frame qualify (no new data is
        fetched).
        """
        if "game_date" not in bgames.columns or "game_date" not in cgames.columns:
            return []
        try:
            cd = pd.to_datetime(cgames["game_date"])
        except Exception:
            return []
        if cd.notna().sum() == 0:
            return []
        lo, hi = cd.min(), cd.max()
        bd = pd.to_datetime(bgames["game_date"], errors="coerce")
        out: list = []
        for k in months_back:
            lo2 = (lo + pd.DateOffset(years=k) - pd.Timedelta(days=7))
            hi2 = (hi + pd.DateOffset(years=k) + pd.Timedelta(days=7))
            win = bgames.loc[(bd >= lo2) & (bd <= hi2), col].dropna()
            out.extend(win.tolist())
        return out

    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(feature_cols) if feature_cols is not None \
        else list(active_moneyline_feature_cols())

    drift_rows = []
    for col in cols:
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

        # Season-seam guard (2026-09-30): the ~21-day trailing baseline
        # spans the season boundary in every season's final week, so
        # late-September run regularly flags REGULAR seasonal movement
        # (playoff bullpen usage, eliminated-team call-ups) as drift. When
        # a location shift survives the trailing baseline, re-check the
        # same mean shift against the SAME calendar phase of prior
        # seasons (same months, one and two years back). A clean
        # re-check means the shift is seasonal, not a 2026 regime break.
        if status in ("WARN", "ALERT") and n_b >= 100 and n_c >= 30:
            phase_vals = _phase_matched_baseline(
                phase_frame if phase_frame is not None else baseline_games,
                current_games, col, tuple(DRIFT_PHASE_EXTENSION_MONTHS))
            if len(phase_vals) >= 100:
                pv = np.asarray(phase_vals, dtype=float)
                pooled2 = np.sqrt(
                    ((len(pv) - 1) * pv.var(ddof=1)
                     + (n_c - 1) * current_vals.var(ddof=1))
                    / (len(pv) + n_c - 2)) if len(pv) + n_c > 2 else 0.0
                shift2 = float(current_vals.mean() - pv.mean())
                se2 = (pooled2 * np.sqrt(1.0 / len(pv) + 1.0 / n_c) * 1.5
                       if pooled2 > 0 else 0.0)
                if se2 > 0 and abs(shift2) <= 2.0 * se2:
                    status = "OK-SEASONAL"

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
    out_path = DATA_DELIVERY_DIR / (out_name or
                                    f"{FEATURE_DRIFT}_{target_date_str}.csv")
    df.to_csv(out_path, index=False)

    n_warns = (df["status"] == "WARN").sum()
    n_alerts = (df["status"] == "ALERT").sum()
    has_status = "status" in df.columns
    n_seasonal = int((df["status"] == "OK-SEASONAL").sum()) if has_status else 0
    n_insufficient = (int((df["status"] == "INSUFFICIENT").sum())
                      if has_status else 0)
    n_evaluated = len(df) - n_insufficient
    # 2026-10-10 log review (second delivery): this line advertised
    # "0 warnings, 0 alerts" on seven consecutive runs during which NOT
    # ONE feature was evaluated — the trailing 7-day current window falls
    # under the 30-row judge floor once the regular season ends, so every
    # row is INSUFFICIENT and the summary still read as a clean result.
    # Publish how many rows were actually judged, and say so at WARNING
    # when nothing was: an inert monitor must not look like a healthy one.
    # INSUFFICIENT itself still never pages (it is a window-size
    # statement, not a defect) — only the all-unevaluated case does.
    logger.info(
        "Feature drift [%s]: %d features, %d evaluated, %d INSUFFICIENT, "
        "%d warnings, %d alerts, %d seasonal (statuses on noise-adjusted "
        "PSI; mean noise floor %.3f)",
        view,
        len(df), n_evaluated, n_insufficient, n_warns, n_alerts, n_seasonal,
        float(df["noise_floor"].mean()) if "noise_floor" in df.columns else float("nan"),
    )
    if len(df) and n_evaluated == 0:
        logger.warning(
            "Feature drift [%s]: 0 of %d features evaluable this run "
            "(every row INSUFFICIENT — current/baseline windows under the "
            "PSI sample floor) — drift is UNMEASURED, not clean",
            view, len(df),
        )

    return df


# ── Coverage-audit remediation (2026-10-09) ─────────────────────────────
# Declared missing-value policies for the two weather interactions — the
# only served features whose NULLs are produced by a documented policy
# gate rather than by a data hole:
#   * closed-roof rows are unconditionally zero-filled by
#     features.apply_indoor_neutral_fills (a POLICY zero that the table
#     correctly refuses to count as an observation), and
#   * open-air rows are NULL exactly when their governing SP input is
#     NULL — the SP staleness gate (a stale carry is nulled on purpose,
#     never zero-filled).
# On roof-heavy windows those two policies alone put % measured below the
# 80% OK line even when EVERY open-air game was observed, so the row used
# to stand amber forever — a permanent alarm that trained viewers to
# ignore it (2026-10-09 coverage audit). A weather OUTAGE has the opposite
# signature: open-air rows whose SP input IS present but whose value is
# NULL (the 2026-10-07 truncation class). Those rows are unexplained and
# keep the raw LOW_COVERAGE/STARVED thresholds, so sensitivity to real
# starvation is unchanged — STRUCTURAL is granted only when every single
# unmeasured row carries a declared reason (NBA 2026-10-08 semantics).
_DECLARED_POLICY_INPUTS = {
    "wind_advantage_flyball_factor": "sp_xfip_diff",
    "air_density_velocity_boost": "sp_fbvelo_diff",
}

# Statuses that still page from the coverage table. STRUCTURAL is NOT an
# alarm (it is absence by declared policy) and is summarized at INFO.
_COVERAGE_ALARM_STATUSES = ("STARVED", "LOW_COVERAGE", "MISSING_COLUMN")


def _declared_policy_reason(
    games: pd.DataFrame,
    col: str,
    vals: pd.Series,
    nonnull: pd.Series,
    default: pd.Series,
    dome: pd.Series,
) -> Optional[str]:
    """Explain EVERY unmeasured row of ``col``, or return None (fail open).

    Returns a human-readable reason only when all unmeasured rows are
    closed-roof policy zeros or open-air NULLs gated by this feature's
    declared SP input. One unexplained row (missing roof column, an
    observation gap, an inf, a broken indoor fill) keeps the real alarm.
    """
    input_col = _DECLARED_POLICY_INPUTS[col]
    unmeasured = ~nonnull | default
    n_unmeasured = int(unmeasured.sum())
    if not n_unmeasured:
        return None
    if input_col not in games.columns:
        # Without the governing input there is no proof of the gate — the
        # threshold alarm stays (fail-open, never fail-silent).
        return None
    inp = pd.to_numeric(games[input_col], errors="coerce")
    # Only a true NaN open-air value gated by a missing input is policy.
    # A dome row whose value is NULL (the indoor fill did not run), an
    # inf, or an open-air NULL beside a valid input are all unexplained.
    open_air_null = vals.isna() & ~dome.eq(1)
    gated = open_air_null & inp.isna()
    explained = default | gated
    if not bool(explained[unmeasured].all()):
        return None
    n_gated = int((unmeasured & gated).sum())
    n_defaults = int((unmeasured & default).sum())
    return (
        f"declared missing-value policy on all {n_unmeasured} unmeasured "
        f"row(s): {n_defaults} closed-roof policy zero(s) (indoor-neutral "
        f"fill, never an observation) + {n_gated} open-air NULL(s) gated "
        f"by a missing {input_col} (SP staleness gate); every open-air "
        "row with a valid input is observed"
    )


def compute_feature_coverage(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    feature_cols: Optional[list[str]] = None,
    out_name: Optional[str] = None,
    view: str = "moneyline",
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
      * air_density_velocity_boost: level-backed outdoor observations are
        measured; closed-roof policy zeros are reported as neutral defaults.
    All other features: finite non-null is reported as measured. Missing
    columns emit explicit MISSING_COLUMN rows; infinities never count as
    observations. Empty windows still write a readable schema. The default
    enumerates the ACTIVE moneyline serving width (adopted RFE subset,
    else the universe) — SINGLE-LIST RULE: every monitor-facing surface
    reads exactly one list.

    Status is OK / LOW_COVERAGE / STARVED on % measured, plus
    MISSING_COLUMN. A non-OK row of a DECLARED-POLICY feature
    (_DECLARED_POLICY_INPUTS) is downgraded to STRUCTURAL only when every
    unmeasured row is explained by that policy — never when even one row
    looks like an observation gap. STRUCTURAL is absence by declared
    design: it renders calm with its reason and is summarized at INFO,
    never WARNING'd.
    """
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(feature_cols) if feature_cols is not None \
        else list(active_moneyline_feature_cols())

    def _window_rows(games: pd.DataFrame, window: str) -> list[dict]:
        rows: list[dict] = []
        if games is None or games.empty:
            return rows
        dome_col = ("dome_is_neutral_game" if "dome_is_neutral_game" in games
                    else "dome_is_neutral")
        dome = pd.to_numeric(games[dome_col], errors="coerce") \
            if dome_col in games.columns else pd.Series(np.nan, index=games.index)
        for col in cols:
            present = col in games.columns
            vals = (pd.to_numeric(games[col], errors="coerce") if present
                    else pd.Series(np.nan, index=games.index, dtype=float))
            n_total = int(len(vals))
            nonnull = vals.notna() & np.isfinite(vals)
            n_invalid = int((vals.notna() & ~np.isfinite(vals)).sum())
            n_nonnull = int(nonnull.sum())
            # Exact-0.0 on a game-resolved closed roof is the dome branch's
            # POLICY signature (a real calm-wind observation outdoors can be
            # 0.0 too — hence dome==1 is part of the test), never counted as
            # an observation.
            default = nonnull & (vals == 0.0) & (dome == 1)
            n_default = int(default.sum()) if col in (
                "wind_advantage_flyball_factor",
                "air_density_velocity_boost") else 0
            n_measured = n_nonnull - n_default
            pct_nonnull = round(100.0 * n_nonnull / n_total, 1) if n_total else 0.0
            pct_measured = round(100.0 * n_measured / n_total, 1) if n_total else 0.0
            status = "OK" if pct_measured >= 80.0 else (
                "LOW_COVERAGE" if pct_measured >= 25.0 else "STARVED")
            # A non-OK row for a declared-policy feature is STRUCTURAL only
            # when every unmeasured row is explained (see the 2026-10-09
            # coverage-audit note above). A healthy family keeps its OK and
            # carries no reason — the monitor renders a reason as THE
            # finding, never a stray explanation beside a pass.
            structural_reason = None
            if status != "OK" and present \
                    and col in _DECLARED_POLICY_INPUTS:
                structural_reason = _declared_policy_reason(
                    games, col, vals, nonnull, default, dome)
            if structural_reason:
                status = "STRUCTURAL"
            rows.append({
                "feature": col,
                "window": window,
                "n_games": n_total,
                "n_nonnull": n_nonnull,
                "pct_nonnull": pct_nonnull,
                "n_measured": n_measured,
                "pct_measured": pct_measured,
                "n_default_zero": n_default,
                "status": "MISSING_COLUMN" if not present else status,
                "column_present": present,
                "n_invalid": n_invalid,
                "structural_reason": structural_reason,
            })
        return rows

    cov_rows = _window_rows(current_games, "current")
    cov_rows += _window_rows(baseline_games, "baseline")
    df = pd.DataFrame(cov_rows, columns=[
        "feature", "window", "n_games", "n_nonnull", "pct_nonnull",
        "n_measured", "pct_measured", "n_default_zero", "status",
        "column_present", "n_invalid", "structural_reason",
    ])
    out_path = DATA_DELIVERY_DIR / (out_name or
                                    f"feature_coverage_{target_date_str}.csv")
    df.to_csv(out_path, index=False)

    # STRUCTURAL rows are absence by DECLARED POLICY, not starvation —
    # they never page (cross-sport contract: NFL 2026-09-29, NBA
    # 2026-10-08). Only real alarms reach WARNING; structural rows are
    # summarized at INFO so the reasoning stays visible in the run log.
    alarms = df[df["status"].isin(_COVERAGE_ALARM_STATUSES)]
    structural = df[df["status"] == "STRUCTURAL"]
    if not alarms.empty:
        worst = alarms.sort_values("pct_measured").head(5)
        # Weather rows carry closed-roof POLICY zeros (counted as defaults,
        # never as observations), so their % measured can never reach the
        # 80% OK line on a roof-heavy window even when every open-air game
        # was observed. The 2026-10-09 review therefore prints the
        # open-air-only ratio alongside it, so a standing LOW_COVERAGE on
        # these two features is readable as "defaults + missing inputs",
        # not as a silent data outage (the alert itself is unchanged).
        def _detail(r) -> str:
            base = f"{r.feature}/{r.window}={r.pct_measured:.0f}% measured"
            n_open = int(r.n_games) - int(r.n_default_zero)
            if int(r.n_default_zero) and n_open > 0:
                base += (f" | open-air {100.0 * r.n_measured / n_open:.0f}% "
                         f"({n_open} rows, {int(r.n_default_zero)} "
                         f"closed-roof policy zeros excluded)")
            return base

        detail = "; ".join(_detail(r) for r in worst.itertuples())
        logger.warning("Feature coverage gaps [%s]: %s", view, detail)
        if not structural.empty:
            _log_structural(view, df, structural)
    elif not structural.empty:
        # 2026-10-10 log review: "all 218 ... OK (4 STRUCTURAL)" was
        # self-contradictory — the delivered CSV holds 214 OK + 4
        # STRUCTURAL, and a reader reconciling the two numbers could not
        # tell which one was wrong. Print the counts that match the CSV.
        logger.info(
            "Feature coverage [%s]: %d feature-window pairs — %d OK, "
            "%d STRUCTURAL by declared policy, 0 alarms",
            view, len(df), len(df) - len(structural), len(structural))
        _log_structural(view, df, structural)
    else:
        logger.info(
            "Feature coverage [%s]: %d feature-window pairs — %d OK, "
            "0 STRUCTURAL, 0 alarms", view, len(df), len(df))
    return df


def _log_structural(view: str, df: pd.DataFrame,
                    structural: pd.DataFrame) -> None:
    """INFO summary of STRUCTURAL coverage rows (never a WARNING)."""
    reasons = sorted({str(r) for r in structural["structural_reason"]
                      if r})
    pairs = ", ".join(
        f"{r.feature}/{r.window}={r.pct_measured:.0f}%"
        for r in structural.sort_values("pct_measured").itertuples())
    logger.info(
        "Feature coverage [%s]: STRUCTURAL (declared missing-value "
        "policy, not starvation): %s — %s",
        view, pairs, "; ".join(reasons))


# ---------------------------------------------------------------------------
# Run-engine feature view (same dynamic contract as moneyline)
# ---------------------------------------------------------------------------

# Content key of the last monitoring-view resolution already logged at INFO
# (2026-10-06 log review: the "Run-engine monitoring view" line printed twice
# per run — once for drift, once for coverage — byte-identical). Keyed on the
# RESOLVED CONTENT, not a call counter: a view that changes mid-process logs
# again as a distinct line (same idiom as distributions._log_resolved_view).
_LAST_LOGGED_MON_VIEW: Optional[tuple[str, ...]] = None


def _log_resolved_mon_view(feats: list) -> None:
    """Log the resolved run-engine monitoring view once per DISTINCT view.

    ``run_engine_feature_cols`` re-resolves on every call (drift + coverage
    each need the live contract) and used to log the byte-identical INFO
    both times — the same defect distributions._log_resolved_view fixed for
    ``build_side_frame``. First resolution of a view logs at INFO; repeats
    drop to DEBUG; a *different* view logs at INFO again.
    """
    global _LAST_LOGGED_MON_VIEW
    key = tuple(feats)
    if key == _LAST_LOGGED_MON_VIEW:
        logger.debug("Run-engine monitoring view: %d active moneyline features "
                     "(unchanged)", len(feats))
        return
    logger.info("Run-engine monitoring view: %d active moneyline features",
                len(feats))
    _LAST_LOGGED_MON_VIEW = key


def run_engine_feature_cols() -> list[str]:
    """Return the active run-engine inputs used by production monitoring.

    The run engine and binary moneyline share one dynamically resolved feature
    contract. This keeps PSI and coverage aligned automatically when the active
    moneyline RFE subset changes. Deferred import avoids a module cycle.
    """
    from training import active_moneyline_feature_cols
    feats = list(active_moneyline_feature_cols())
    # 2026-10-06 log review: this resolver runs twice per run (drift +
    # coverage) and logged the byte-identical INFO both times — the same
    # defect R7 fixed for distributions.build_side_frame.
    _log_resolved_mon_view(feats)
    return feats



def compute_run_engine_feature_drift(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
    model_weights: Optional[dict] = None,
    phase_frame: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """PSI over the run engine's own feature view on the SAME baseline /
    current windows as the moneyline drift — leakage-free.

    ``model_weights`` (2026-09-27) = the RUN LINE's own per-feature weights
    (pooled split-gain from its two side Poisson models, normalized to sum
    to 1.0; produced by predict_slate_runs and relayed through
    run_engine_daily's monitor block). The drift CSV's MODEL WEIGHT column
    then reports the run line model itself instead of the moneyline blend's
    weights the frontend used to borrow. Features absent from the map (e.g.
    the team-ID categoricals) render weight None — never borrowed.
    Writes data_delivery/run_engine_feature_drift_YYYYMMDD.csv."""
    weights = None
    if model_weights:
        # The drift column is percent-of-total (the moneyline's convention:
        # weight_pct sums to 100) — scale the run-line shares to match.
        weights = {k: 100.0 * v for k, v in model_weights.items()
                   if isinstance(v, (int, float))}
    return compute_feature_drift(
        baseline_games, current_games, target_date_str,
        model_weights=weights, feature_cols=run_engine_feature_cols(),
        out_name=f"run_engine_feature_drift_{target_date_str}.csv",
        phase_frame=phase_frame, view="run-engine")


def compute_run_engine_feature_coverage(
    baseline_games: pd.DataFrame,
    current_games: pd.DataFrame,
    target_date_str: str,
) -> pd.DataFrame:
    """Coverage over the run engine's shared feature contract on the SAME
    windows (the single-list rule keeps it byte-identical to the moneyline
    view). Writes data_delivery/run_engine_feature_coverage_YYYYMMDD.csv."""
    return compute_feature_coverage(
        baseline_games, current_games, target_date_str,
        feature_cols=run_engine_feature_cols(),
        out_name=f"run_engine_feature_coverage_{target_date_str}.csv",
        view="run-engine")


ROLLING_BRIER_WINDOW_DAYS = 30
ROLLING_BRIER_MIN_GAMES_PER_DAY = 5


def compute_rolling_brier(
    history_df: Optional[pd.DataFrame],
    target_date_str: str,
    calibrator: Optional[dict] = None,
    window_days: int = ROLLING_BRIER_WINDOW_DAYS,
    min_games_per_day: int = ROLLING_BRIER_MIN_GAMES_PER_DAY,
) -> dict:
    """Rolling trailing-window Brier series from walk-forward OOF history.

    Per game: brier = (p - y)^2 where p is the FINAL blended probability
    passed through the DEPLOYED Platt map — exactly the number the dashboard
    displays everywhere else (Today's Games win %, Prediction History MODEL
    PICK %). Raw blend or single-member probabilities are never used here.

    Daily Brier = mean over that game date; days with fewer than
    ``min_games_per_day`` decided games are excluded as sparse (counted in
    ``excluded_sparse_days``, never silently averaged in). The series point
    at each qualifying date d is the mean per-game Brier over ALL games in
    the trailing ``window_days`` calendar days [d - window_days + 1, d] —
    a game-count-free calendar window, so off-days between game dates simply
    contribute nothing instead of breaking or NaN-ing the series.

    Returns an artifact dict (also written to rolling_brier_<date>.json);
    ``series`` is [] when no data supports it (with a loud warning logged).
    """
    from calibration import moneyline_apply  # same adapter training.py deploys

    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "window_days": int(window_days),
        "min_games_per_day": int(min_games_per_day),
        "source_column": "home_win_prob_model",
        "calibration": "deployed moneyline calibration adapter (favored-space when available)",
        # The deployed map is fit on ALL OOF games, including recent ones, so
        # the newest ~window_days of points are mildly optimistic vs honest
        # prequential scoring (per-fold maps). Surfaced on the panel.
        "map_scope_note": (
            "Recent points use the deployed map (fit on all OOF games) and are "
            "not directly comparable to prequential-calibrated metrics like "
            "logloss_calibrated."
        ),
        "calibrator_is_identity": None,
        "n_points": 0,
        "n_games_total": 0,
        "excluded_sparse_days": 0,
        "history_mean_brier": None,
        "series": [],
    }

    if history_df is None or history_df.empty:
        logger.warning(
            "Rolling Brier: predictions history missing/empty — series will "
            "be empty (dashboard shows the empty state; do not fabricate data)"
        )
        _write_rolling_brier(result, target_date_str)
        return result

    dates = pd.to_datetime(history_df.get("game_date"), errors="coerce")
    y = pd.to_numeric(history_df.get("home_win"), errors="coerce")
    p_raw = pd.to_numeric(history_df["home_win_prob_model"], errors="coerce")
    ok = dates.notna() & y.notna() & p_raw.notna() & y.isin([0, 1])
    df = pd.DataFrame({
        "date": dates[ok].dt.normalize(),
        "y": y[ok].astype(int),
        "p_cal": moneyline_apply(p_raw[ok].to_numpy(dtype=float), calibrator),
    })
    if df.empty:
        logger.warning(
            "Rolling Brier: no DECIDED games with finite probabilities in "
            "history (%d rows scanned) — series empty", len(history_df)
        )
        _write_rolling_brier(result, target_date_str)
        return result

    # Identity-map detection for honest artifact metadata.
    try:
        from calibration import is_identity
        result["calibrator_is_identity"] = bool(is_identity(calibrator))
    except Exception:  # pragma: no cover - metadata only
        pass

    df["brier"] = (df["p_cal"] - df["y"]) ** 2
    daily = df.groupby("date")["brier"].agg(["mean", "size"])
    qualifying = daily[daily["size"] >= min_games_per_day]
    result["excluded_sparse_days"] = int((daily["size"] < min_games_per_day).sum())
    # Exclusion is consistent everywhere: sparse-day games never contribute
    # to a series point's trailing-window mean either.
    df_q = df[df["date"].isin(qualifying.index)]
    result["n_games_total"] = int(len(df))
    result["n_games_in_series"] = int(len(df_q))
    result["history_mean_brier"] = round(float(df["brier"].mean()), 6)

    span = pd.Timedelta(days=window_days - 1)
    series: list[dict] = []
    for day, row in qualifying.sort_index().iterrows():
        mask = (df_q["date"] >= day - span) & (df_q["date"] <= day)
        window_games = df_q[mask]
        if window_games.empty:  # defensive; qualifying ⊆ df by construction
            continue
        series.append({
            "date": day.strftime("%Y-%m-%d"),
            "brier": round(float(window_games["brier"].mean()), 6),
            "games": int(len(window_games)),
        })
    result["n_points"] = len(series)
    result["series"] = series

    if series:
        logger.info(
            "Rolling Brier: %d points (%s → %s), %d games, %d sparse days "
            "excluded (<%d games/day), mean %.4f",
            len(series), series[0]["date"], series[-1]["date"],
            result["n_games_total"], result["excluded_sparse_days"],
            min_games_per_day, result["history_mean_brier"],
        )
    else:
        logger.warning(
            "Rolling Brier: %d decided-game days but none reached the "
            "%d-game minimum — series empty",
            len(daily), min_games_per_day,
        )

    _write_rolling_brier(result, target_date_str)
    return result


def _write_rolling_brier(result: dict, target_date_str: str) -> None:
    out_path = DATA_DELIVERY_DIR / f"rolling_brier_{target_date_str}.json"
    try:
        import json
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    except OSError as exc:  # pragma: no cover - disk issues shouldn't kill run
        logger.error("Rolling Brier artifact write failed: %s", exc)
