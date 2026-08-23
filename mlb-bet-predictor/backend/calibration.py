"""Post-hoc probability calibration (Platt scaling).

The walk-forward ensemble produces raw blended probabilities that are
measured (ECE) but never corrected before publishing. This module adds a
single post-hoc layer: a 2-parameter logistic map

    p_cal = sigmoid(a * logit(p_raw) + b)

fitted out-of-sample, applied after blending and before the published
probabilities feed picks/edge math.

Design constraints that keep the calibration honest:
- Fitted ONLY on pooled walk-forward OOF pairs — never on a member's own
  training data (trees memorize train folds; an in-sample Platt fit would
  learn an over-extreme correction).
- Per-fold evaluation uses a PREQUENTIAL scheme: fold k's calibrated
  predictions come from a calibrator fitted on folds 0..k-1 only.
- Guardrails: below MIN_OOF_FOR_FIT games, degenerate labels, or any fit
  failure the identity map is used (no correction) rather than a risky one.

The calibrator is a plain dict {"method","a","b","n"} so it survives
joblib round-trips inside ensemble_latest.joblib and JSON reporting.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Pooled OOF games required before trusting a fitted correction. Below
# this, a 2-param fit can chase noise; identity is the safer map.
MIN_OOF_FOR_FIT = 300

_EPS = 1e-6  # clip bound for logit(p); matches compute_metrics clipping spirit


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def is_identity(calibrator: dict | None) -> bool:
    """True when ``calibrator`` applies no correction (None or a≈1, b≈0)."""
    if not calibrator:
        return True
    if str(calibrator.get("method")) != "platt":
        return True
    try:
        a = float(calibrator.get("a", 1.0))
        b = float(calibrator.get("b", 0.0))
    except (TypeError, ValueError):
        return True
    return abs(a - 1.0) < 1e-9 and abs(b) < 1e-9


def fit_platt(y_true, y_prob) -> dict | None:
    """Fit the Platt map on pooled OOF (y, p) pairs.

    Returns {"method": "platt", "a": slope, "b": intercept, "n": n} or
    None when the data cannot support a fit (too few games, single class,
    non-finite inputs). The logistic fit uses negligible regularization so
    it converges to the classic 2-parameter Platt solution while staying
    numerically stable.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (p > 0) & (p < 1)
    y, p = y[ok], p[ok]
    n = len(y)
    if n < MIN_OOF_FOR_FIT:
        logger.info(
            "Calibration: %d OOF games < %d minimum — using identity map",
            n, MIN_OOF_FOR_FIT,
        )
        return None
    if len(np.unique(y)) < 2:
        logger.warning("Calibration: single-class OOF labels — identity map")
        return None

    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(_logit(p).reshape(-1, 1), y)
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
    except Exception as exc:
        logger.warning("Calibration: Platt fit failed (%s) — identity map", exc)
        return None

    # A pathological fit (slope <= 0 would invert the ranking) falls back
    # to identity: ranking preservation matters more than ECE cosmetics.
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        logger.warning("Calibration: degenerate Platt params (a=%s) — identity map", a)
        return None

    cal = {"method": "platt", "a": round(a, 6), "b": round(b, 6), "n": int(n)}
    logger.info(
        "Calibration: Platt fitted on %d OOF games (a=%.4f, b=%.4f)", n, a, b
    )
    return cal


def apply_platt(y_prob, calibrator: dict | None) -> np.ndarray:
    """Apply the fitted map; identity when calibrator is None/invalid."""
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if is_identity(calibrator):
        return p.copy()
    try:
        a = float(calibrator["a"])
        b = float(calibrator["b"])
    except (KeyError, TypeError, ValueError):
        return p.copy()
    return _sigmoid(a * _logit(p) + b)
