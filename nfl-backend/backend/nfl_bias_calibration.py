"""NFL per-side mean-bias calibration — prediction-layer transform (record-only).

Context (commit 4c69cdb record): joint-layer G1 failed on the away leg
(pooled distributional CRPS improvement 2.94% < the 5% bar) and derived
totals ran hot (totals ECE 0.138, top bin pred 0.83 vs actual 0.55). Both
trace to the away per-side mean bias (pooled-OOF mean resid −1.49 vs home
−0.19). The joint machinery is sound. THIS STEP adds a leak-free LINEAR
recalibration of the per-side means (a prediction-layer transform), then
re-runs the joint chain through the EXISTING engine entrypoints —
``nfl_per_side_engine.py`` and ``nfl_joint_engine.py`` are NOT modified.

Transform (per side, independently): OLS actual ~ a*pred + b fitted on the
POOLED-OOF rows ONLY (n = 1,091); pred_cal = b + a*pred, applied to pooled
AND sealed 2025. Sealed rows are STRUCTURALLY excluded from the fit (the
cal dict carries the ``fit_on == "pooled_oof"`` marker that
``apply_calibration`` enforces, plus a season guard when the input carries
a season column). a != 1 means shrinkage (regression to the mean); the
spec's construction-change flag fires when |a − 1| > 0.15 (deeper
misspecification — side-anchored away features are the suspected cause —
recorded as a candidate for the future view-expansion work, NOT a blocker
here).

Diagnostics classify each side's bias as offset vs slope tilt vs curvature
vs time trend (report tables + advisory labels; thresholds documented
below). Sigma re-estimation is in scope because the OLD sigma was fit on
biased residuals: the chain re-run recomputes residuals from calibrated
preds and refits the joint params (family / sigma / rho) through
``nfl_joint_engine.fit_joint_params`` on the recalibrated pooled table —
the same entrypoint the joint runner uses, zero engine edits.

Judgment calls (flag if overridden):
  1. Linear over isotonic — n = 1,091 is thin; isotonic only if post-linear
     diagnostics show clear curvature.
  2. Per-side independent maps (home a/b, away a/b), not pooled.
  3. Calibration is a prediction-layer transform; module code untouched.
     Future market-layer wiring consumes the stored params JSON.
  4. Sigma re-estimation IS in scope (old sigma fit on biased residuals).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Column contract (mirrors the step-1 artifact) ───────────────────────────
PRED_COLS = {"home": "pred_home", "away": "pred_away"}
ACTUAL_COLS = {"home": "home_score", "away": "away_score"}
RESID_COLS = {"home": "resid_home", "away": "resid_away"}
# Calibrated outputs (added alongside — never overwrite the raw inputs).
CAL_PRED = {"home": "pred_home_cal", "away": "pred_away_cal"}
CAL_RESID = {"home": "resid_home_cal", "away": "resid_away_cal"}

FIT_ON = "pooled_oof"
Z = 1.96  # 95% normal CI (n = 1,091 pooled rows)

# Diagnostics thresholds (advisory classification, documented).
BIAS_OFFSET_PTS = 0.30       # |mean resid| >= this → offset label
SLOPE_TILT_DELTA = 0.05      # |a - 1| >= this → slope-tilt label
CONSTRUCTION_CHANGE_SLOPE = 0.15  # |a - 1| > this → construction-change flag
CURVATURE_SWING_PTS = 0.50   # quadratic swing over pred range → curvature label
TREND_CORR = 0.80            # |corr(season idx, season mean resid)| >= this
TREND_RANGE_PTS = 0.40       # and seasonal range >= this → time-trend label
DECILES = 10


# ── OLS map (actual ~ a*pred + b) ───────────────────────────────────────────

def ols_map(actual, pred) -> dict[str, float]:
    """Per-side linear map: a, b + 95% CIs, r2. a != 1 ⇒ shrinkage."""
    a_ = np.asarray(actual, dtype=float)
    p_ = np.asarray(pred, dtype=float)
    ok = np.isfinite(a_) & np.isfinite(p_)
    a_, p_ = a_[ok], p_[ok]
    if len(p_) < 30:
        raise ValueError(f"ols_map: need >=30 finite pairs, got {len(p_)}")
    res = stats.linregress(p_, a_)
    return {
        "a": round(float(res.slope), 4),
        "a_ci_low": round(float(res.slope) - Z * float(res.stderr), 4),
        "a_ci_high": round(float(res.slope) + Z * float(res.stderr), 4),
        "b": round(float(res.intercept), 4),
        "b_ci_low": round(float(res.intercept) - Z * float(res.intercept_stderr), 4),
        "b_ci_high": round(float(res.intercept) + Z * float(res.intercept_stderr), 4),
        "r2": round(float(res.rvalue ** 2), 4),
        "n": int(len(p_)),
    }


# ── Calibration fit + apply (the prediction-layer transform) ────────────────

def fit_calibration(pooled: pd.DataFrame, sealed_season: int = 2025
                    ) -> dict[str, Any]:
    """Fit per-side linear maps on POOLED-OOF rows ONLY (never sealed).

    Structural leak guards:
    - If ``pooled`` carries a ``season`` column, ANY row with
      season >= ``sealed_season`` raises ValueError (sealed rows can never
      enter the fit through this API).
    - The returned dict hardcodes ``fit_on == "pooled_oof"``;
      ``apply_calibration`` refuses to use anything else (mirror of the
      joint engine's sealed leak guard).
    """
    pooled = pooled.copy()
    if "season" in pooled.columns:
        bad = int((pooled["season"] >= sealed_season).sum())
        if bad:
            raise ValueError(
                f"fit_calibration: {bad} row(s) with season >= {sealed_season} "
                f"— calibration params must come from pooled OOF only")
    out: dict[str, Any] = {
        "method": "per-side OLS actual ~ a*pred + b on pooled OOF (n=1091); "
                  "pred_cal = b + a*pred",
        "fit_on": FIT_ON,
        "sealed_season": int(sealed_season),
        "home": None,
        "away": None,
    }
    for side, pcol in PRED_COLS.items():
        acol = ACTUAL_COLS[side]
        missing = [c for c in (pcol, acol) if c not in pooled.columns]
        if missing:
            raise ValueError(f"fit_calibration: missing columns {missing}")
        m = ols_map(pooled[acol], pooled[pcol])
        m["fit_on"] = FIT_ON
        out[side] = m
        logger.info("calibration %s: a=%s b=%s (n=%d)",
                    side, m["a"], m["b"], m["n"])
    return out


def apply_calibration(df: pd.DataFrame, cal: dict[str, Any]) -> pd.DataFrame:
    """Apply the pooled-OOF linear maps; add pred_*_cal / resid_*_cal columns.

    Raises ValueError if ``cal`` was not fitted on pooled OOF (sealed rows
    are structurally excluded from the fit). Raw pred/resid columns are left
    untouched — the *_cal columns ride alongside. Residuals are recomputed
    as actual − pred_cal (rounded to 4, the artifact convention).
    """
    if cal.get("fit_on") != FIT_ON:
        raise ValueError(
            f"apply_calibration: cal fit_on={cal.get('fit_on')!r} — only "
            f"{FIT_ON!r} calibration params may be applied")
    out = df.copy()
    for side, pcol in PRED_COLS.items():
        m = cal[side]
        if m is None or m.get("fit_on") != FIT_ON:
            raise ValueError(f"apply_calibration: side {side!r} not calibrated")
        if pcol not in out.columns:
            raise ValueError(f"apply_calibration: missing column {pcol}")
        pred_cal = m["b"] + m["a"] * out[pcol].to_numpy(float)
        out[CAL_PRED[side]] = np.round(pred_cal, 4)
        acol = ACTUAL_COLS[side]
        if acol in out.columns:
            resid_cal = out[acol].to_numpy(float) - pred_cal
            out[CAL_RESID[side]] = np.round(resid_cal, 4)
    return out


def engine_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calibrated frame under the joint engine's STANDARD column names.

    The chain re-run feeds the recalibrated pooled table to
    ``nfl_joint_engine.fit_joint_params``, which reads pred_home/pred_away/
    resid_home/resid_away/home_score/away_score. This helper builds that
    table from a df that has the *_cal columns (preds + residuals swapped in,
    actuals preserved). Fails loudly on a missing cal column — a silently
    uncalibrated frame must never pass as calibrated.
    """
    missing = [CAL_PRED[s] for s in PRED_COLS] + [CAL_RESID[s]
                                                  for s in PRED_COLS]
    missing = [c for c in missing if c not in df.columns]
    if missing:
        raise ValueError(f"engine_table: missing calibrated columns {missing}")
    out = df.copy()
    for side in PRED_COLS:
        out[PRED_COLS[side]] = out[CAL_PRED[side]]
        out[RESID_COLS[side]] = out[CAL_RESID[side]]
    return out


# ── Diagnostics (Step 1 — no retraining, report tables) ─────────────────────

def _side_stats(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    resid = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    return {
        "n": int(len(resid)),
        "mean_resid": round(float(resid.mean()), 4),
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 4),
        "mae": round(float(np.abs(resid).mean()), 4),
    }


def _decile_table(pred: np.ndarray, resid: np.ndarray) -> list[dict]:
    p = pd.Series(np.asarray(pred, dtype=float))
    r = pd.Series(np.asarray(resid, dtype=float))
    q = pd.qcut(p, DECILES, duplicates="drop")
    rows = []
    for lab, idx in q.groupby(q, observed=False).groups.items():
        # lab is an Interval; use its mid as the representative bin point
        mid = (lab.left + lab.right) / 2.0
        rows.append({
            "decile": str(lab),
            "pred_mid": round(float(mid), 2),
            "n": int(len(idx)),
            "mean_resid": round(float(r.loc[idx].mean()), 4),
        })
    return rows


def _curvature_swing(pred: np.ndarray, resid: np.ndarray) -> float:
    """Quadratic swing across the pred range from a degree-2 polyfit (pts).

    c2 * (max(pred) − min(pred))^2 / 8 approximates the peak-to-peak
    curvature contribution at the range edges vs the center.
    """
    p = np.asarray(pred, dtype=float)
    r = np.asarray(resid, dtype=float)
    ok = np.isfinite(p) & np.isfinite(r)
    if ok.sum() < 30:
        return 0.0
    c = np.polyfit(p[ok], r[ok], 2)
    span = float(np.ptp(p[ok]))
    return float(abs(c[0]) * span ** 2 / 8.0)


def diagnose(pooled: pd.DataFrame) -> dict[str, Any]:
    """Full Step-1 diagnostics on the pooled-OOF table (raw, uncalibrated).

    Expects a season column when the time-trend row is wanted (the pooled
    artifact joined to the decided frame). Returns per-side tables:
    by-season mean resid, by-prediction-decile mean resid, OLS a/b + CI, and
    an advisory classification (offset / slope tilt / curvature / time
    trend) with documented thresholds.
    """
    has_season = "season" in pooled.columns
    out: dict[str, Any] = {}
    for side in PRED_COLS:
        pcol, acol = PRED_COLS[side], ACTUAL_COLS[side]
        p_ = pooled[pcol].to_numpy(float)
        a_ = pooled[acol].to_numpy(float)
        r_ = a_ - p_
        stats_ = _side_stats(p_, a_)
        ols = ols_map(a_, p_)

        by_season = []
        if has_season:
            for s, g in pooled.groupby("season"):
                by_season.append({
                    "season": int(s), "n": int(len(g)),
                    "mean_resid": round(float((g[acol].to_numpy(float)
                                               - g[pcol].to_numpy(float)).mean()), 4),
                })
            by_season.sort(key=lambda d: d["season"])

        deciles = _decile_table(p_, r_)
        swing = _curvature_swing(p_, r_)

        labels: list[str] = []
        if abs(stats_["mean_resid"]) >= BIAS_OFFSET_PTS:
            labels.append("offset")
        if abs(ols["a"] - 1.0) >= SLOPE_TILT_DELTA:
            labels.append("slope tilt")
        if swing >= CURVATURE_SWING_PTS:
            labels.append("curvature")
        if has_season and len(by_season) >= 3:
            means = np.array([d["mean_resid"] for d in by_season])
            rho_t = float(np.corrcoef(np.arange(len(means)), means)[0, 1]) \
                if means.std() > 0 else 0.0
            if (abs(rho_t) >= TREND_CORR
                    and float(np.ptp(means)) >= TREND_RANGE_PTS):
                labels.append("time trend")
            trend = {"corr_season_idx": round(rho_t, 4),
                     "seasonal_range_pts": round(float(np.ptp(means)), 4)}
        else:
            trend = None

        out[side] = {
            "stats": stats_,
            "by_season": by_season if has_season else [],
            "by_pred_decile": deciles,
            "ols_actual_on_pred": ols,
            "curvature_swing_pts": round(swing, 4),
            "time_trend": trend,
            "classification": {
                "labels": labels,
                "note": ("advisory: offset |mean resid|>=0.30; slope tilt "
                         "|a-1|>=0.05 (construction-change candidate at "
                         f"|a-1|>{CONSTRUCTION_CHANGE_SLOPE}); curvature "
                         "quadratic swing >=0.50 pts; time trend |r|>=0.80 "
                         "on season means with >=0.40 pt seasonal range"),
            },
            "construction_change_flag": abs(ols["a"] - 1.0) > CONSTRUCTION_CHANGE_SLOPE,
            "construction_change_note": (
                "|a - 1| > 0.15 ⇒ suspected feature-orientation inversion "
                "(side-anchored away features) — recorded as a candidate "
                "for the future view-expansion work, NOT a blocker here"),
        }
    return out
