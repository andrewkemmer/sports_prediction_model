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

CALIBRATION_MODE switch (moneyline path only; run-engine calibration always
uses raw fit_platt/apply_platt and is untouched):
- "platt" (default): the shipped 2-parameter logistic map.
- "identity": publish the raw blend — moneyline_fit returns None and
  moneyline_apply returns p unchanged, so calibrated == raw everywhere.

The switch is reversible (CALIBRATION_MODE env var / set_calibration_mode)
and default stays "platt".

PINNED VERDICT (2026-08-27, blend-level gate run_calibration_flip_test.py,
45-fold geometry, run-engine read-only): DON'T ADOPT — keep platt. On the
sealed 284 holdout identity ties platt on logloss (0.6804) and AUC (0.5617)
and improves ECE-cal (0.0559 -> 0.0455), but pooled OOF ECE-cal DEGRADES
(0.0047 -> 0.0096 tune-only; 0.0066 -> 0.0083 incl. sealed) — the Platt map
still helps in-distribution pooled calibration, so the flip fails the gate.
Full table: data_delivery/calibration_flip_20260827.json.
"""

from __future__ import annotations

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)

# Pooled OOF games required before trusting a fitted correction. Below
# this, a 2-param fit can chase noise; identity is the safer map.
MIN_OOF_FOR_FIT = 300


VALID_CALIBRATION_MODES = ("platt", "identity")
FAVORED_CALIBRATOR_METHOD = "favored_platt_floor"
FAVORED_PROBABILITY_FLOOR = 0.5


def get_calibration_mode() -> str:
    """Active moneyline calibration mode ("platt" or "identity")."""
    mode = str(config.CALIBRATION_MODE).strip().lower()
    if mode not in VALID_CALIBRATION_MODES:
        logger.warning(
            "Calibration: unknown CALIBRATION_MODE %r — falling back to platt",
            config.CALIBRATION_MODE,
        )
        return "platt"
    return mode


def set_calibration_mode(mode: str) -> None:
    """Switch the moneyline calibration mode in-process (harness/test use).

    Affects ONLY the moneyline path via moneyline_fit/moneyline_apply;
    fit_platt/apply_platt (and therefore the run-engine calibration) are
    unchanged. "platt" is today's behavior; "identity" publishes raw.
    """
    m = str(mode).strip().lower()
    if m not in VALID_CALIBRATION_MODES:
        raise ValueError(
            f"unknown calibration mode {mode!r} (expected 'platt' or 'identity')")
    config.CALIBRATION_MODE = m


def moneyline_fit(y_true, y_prob):
    """Moneyline calibrator fit, gated by CALIBRATION_MODE.

    Identity mode skips the map entirely (returns None -> raw published
    probabilities); platt mode is today's behavior. Run-engine paths call
    fit_platt directly and are unaffected.
    """
    if get_calibration_mode() == "identity":
        logger.info(
            "Calibration: CALIBRATION_MODE=identity — moneyline publishes the "
            "raw blend (no Platt map)")
        return None
    return fit_favored_platt(y_true, y_prob)


def moneyline_apply(y_prob, calibrator: dict | None) -> np.ndarray:
    """Moneyline calibrator application, gated by CALIBRATION_MODE.

    Identity mode returns p unchanged (calibrated == raw); platt mode is
    today's behavior.
    """
    if get_calibration_mode() == "identity":
        return np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return apply_moneyline_calibration(y_prob, calibrator)

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
    if str(calibrator.get("method")) not in ("platt", FAVORED_CALIBRATOR_METHOD):
        return True
    try:
        a = float(calibrator.get("a", 1.0))
        b = float(calibrator.get("b", 0.0))
    except (TypeError, ValueError):
        return True
    return abs(a - 1.0) < 1e-9 and abs(b) < 1e-9


def _favored_view(y_true, y_prob) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert home-side predictions/outcomes into favored-team space."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    home_favorite = p >= 0.5
    favored_p = np.where(home_favorite, p, 1.0 - p)
    favored_won = np.where(home_favorite, y, 1.0 - y)
    return favored_won, favored_p, home_favorite


def fit_favored_platt(y_true, y_prob) -> dict | None:
    """Fit Platt scaling directly to the probability that the model-favored team wins."""
    favored_y, favored_p, _ = _favored_view(y_true, y_prob)
    cal = fit_platt(favored_y, favored_p)
    if cal is None:
        return None
    cal["method"] = FAVORED_CALIBRATOR_METHOD
    cal["floor"] = FAVORED_PROBABILITY_FLOOR
    return cal


def apply_moneyline_calibration(y_prob, calibrator: dict | None) -> np.ndarray:
    """Apply favored-space calibration and convert back to home-win space.

    Legacy home-space calibrators are deliberately rejected. Generic
    ``apply_platt`` remains available to the NB run-engine markets, but it is
    no longer a valid moneyline calibration path.
    """
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if not calibrator:
        return p.copy()
    if calibrator.get("method") != FAVORED_CALIBRATOR_METHOD:
        raise ValueError(
            "legacy home-space moneyline calibration is unsupported; "
            "retrain to create a favored-space calibrator"
        )
    home_favorite = p >= 0.5
    favored_p = np.maximum(p, 1.0 - p)
    favored_cal = apply_platt(favored_p, calibrator)
    favored_cal = np.maximum(favored_cal, float(calibrator.get("floor", FAVORED_PROBABILITY_FLOOR)))
    return np.where(home_favorite, favored_cal, 1.0 - favored_cal)


def fit_platt(y_true, y_prob) -> dict | None:
    """Fit the Platt map on pooled OOF (y, p) pairs.

    Returns a generic internal Platt map used by NB markets. The moneyline
    path wraps this result as ``favored_platt_floor``; generic ``method=platt``
    is not valid for moneyline serving. Returns {"method": "platt", "a": slope,
    "b": intercept, "n": n} or None when the data cannot support a fit (too
    few games, single class,
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
