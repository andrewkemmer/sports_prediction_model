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

NESTED CAUSAL GATE (2026-10-07, NHL parity): gated_moneyline_fit now uses
one policy at every fold and final origin: early prior evidence fits the
candidate; the recent prior tail must improve logloss by >0.005 nats before
refitting on all prior evidence. The historical aggregate harm-check helper
below remains for compatibility but no longer decides production deployment.

HISTORICAL DEPLOYED-CALIBRATOR GATE (2026-10-01): the flip-test verdict was a
static one-frame read, and the v2026.09.29 and v2026.09.30 runs then showed
the shipped map hurting on the honest prequential view both runs in a row
(v09.30: ECE 0.0068 -> 0.0088, log-loss 0.6843 -> 0.6847; the 09-30 run
still shipped the fitted map). walk_forward_evaluate now gates the DEPLOYED
map through should_gate_calibrator: when the prequential calibrated column
is worse than the raw blend on BOTH log-loss and ECE, the bundle ships the
identity map (raw blend) instead; mixed evidence keeps the fitted map. The
gate is causal (prequential = fit-on-prior-folds only) and self-reversing —
a later run whose prequential column improves ships the fitted map again.
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


def gated_moneyline_fit(y_true, y_prob) -> tuple[dict | None, dict]:
    """NHL-parity nested gate on chronological strictly-prior blend evidence.

    Fit on the early portion, require >0.005 nats logloss gain on the recent
    25% (at least 200 games), then refit on all prior evidence. The same
    policy runs at every OOF origin and final serving, never on future labels.
    """
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError("moneyline calibration requires aligned vectors")
    ok = np.isfinite(y) & np.isfinite(p) & (p > 0) & (p < 1)
    y, p = y[ok], p[ok]
    record = {"n_prior": len(p), "decision": "identity",
              "reason": "insufficient_prior_evidence"}
    hold = max(200, int(round(0.25 * len(p))))
    n_fit = len(p) - hold
    if n_fit < MIN_OOF_FOR_FIT:
        return None, record
    candidate = moneyline_fit(y[:n_fit], p[:n_fit])
    if candidate is None:
        record["reason"] = "candidate_declined"
        return None, record
    def loss(prob):
        prob = np.clip(np.asarray(prob, dtype=float), _EPS, 1 - _EPS)
        return float(-(y[n_fit:] * np.log(prob)
                       + (1 - y[n_fit:]) * np.log1p(-prob)).mean())
    raw_loss = loss(p[n_fit:])
    cal_loss = loss(moneyline_apply(p[n_fit:], candidate))
    record.update(raw_logloss=raw_loss, calibrated_logloss=cal_loss)
    if cal_loss >= raw_loss - 0.005:
        record["reason"] = "gated_no_gain"
        return None, record
    final = moneyline_fit(y, p)
    record.update(decision="fitted" if final else "identity",
                  reason="accepted" if final else "final_fit_declined")
    return final, record


def moneyline_apply(y_prob, calibrator: dict | None) -> np.ndarray:
    """Moneyline calibrator application, gated by CALIBRATION_MODE.

    Identity mode returns p unchanged (calibrated == raw); platt mode is
    today's behavior.
    """
    if get_calibration_mode() == "identity":
        return np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return apply_moneyline_calibration(y_prob, calibrator)

def should_gate_calibrator(
    raw_metrics: dict, cal_metrics: dict
) -> tuple[bool, str]:
    """Should the DEPLOYED moneyline calibrator be withheld this run?

    The shipped Platt map is fitted on ALL pooled OOF pairs, so the only
    honest rehearsal of how it behaves on unseen games is the PREQUENTIAL
    calibrated column (fold k corrected only by folds < k — exactly how the
    deployed map acts on tomorrow's slate). When that column is worse than
    the raw blend on BOTH log-loss and ECE, the map is hurting every
    headline metric and the identity map is the safer ship; on mixed
    evidence the fitted map stands (the 2026-08-27 flip-test status quo,
    which this gate generalizes into a per-run, self-reversing rule).
    Missing, empty, or non-finite metrics never fire the gate — it only
    responds to unambiguous harm, not to measurement gaps.

    Returns ``(gate_out, reason)``: ``gate_out=True`` means withhold the
    fitted map (ship identity); ``reason`` is a human-readable one-liner
    quoting both metric pairs for the run log.
    """
    if not raw_metrics or not cal_metrics:
        return False, ""
    try:
        raw_ll = float(raw_metrics["logloss"])
        raw_ece = float(raw_metrics["ece"])
        cal_ll = float(cal_metrics["logloss"])
        cal_ece = float(cal_metrics["ece"])
    except (KeyError, TypeError, ValueError):
        return False, ""
    if not all(np.isfinite(v) for v in (raw_ll, raw_ece, cal_ll, cal_ece)):
        return False, ""
    ll_worse = cal_ll > raw_ll
    ece_worse = cal_ece > raw_ece
    if not (ll_worse and ece_worse):
        return False, ""
    return True, (
        f"ECE {raw_ece:.4f} -> {cal_ece:.4f}, log-loss {raw_ll:.4f} -> "
        f"{cal_ll:.4f}")


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
        # DEBUG (not INFO): the prequential market loop calls this ~200x
        # per run (every fold x every market line) — at INFO it floods the
        # run log. Bundle/final params land in the markets meta at INFO.
        logger.debug(
            "Calibration: %d OOF games < %d minimum — using identity map",
            n, MIN_OOF_FOR_FIT,
        )
        return None
    if len(np.unique(y)) < 2:
        logger.warning("Calibration: single-class OOF labels — identity map")
        return None

    # Uncapped attempt counter — denominator for the degenerate summary
    # (incremented here so degenerate and failed fits count as attempts).
    fit_platt._fit_total = getattr(fit_platt, "_fit_total", 0) + 1

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
        # WARNING (degenerate = real signal), rate-limited so the
        # prequential loop cannot flood the log on early small folds.
        # Two counters: _degen_logged drives the 3-line cap; the UNCAPPED
        # _degen_total / _fit_total feed the per-block summary so a
        # reviewer can see HOW MANY fits degenerated (2026-10-06 log
        # review: three warnings were visible, the true count unknowable).
        _degen_n = getattr(fit_platt, "_degen_logged", 0)
        fit_platt._degen_total = getattr(fit_platt, "_degen_total", 0) + 1
        if _degen_n < 3:
            logger.warning(
                "Calibration: degenerate Platt params (a=%s, n=%d) — identity map",
                a, n)
        fit_platt._degen_logged = _degen_n + 1
        return None

    cal = {"method": "platt", "a": round(a, 6), "b": round(b, 6), "n": int(n)}
    # DEBUG: per-fold/prequential fit line (~200x per run); the final bundle
    # params land in the markets meta ("calibration") — that stays the
    # INFO-visible record.
    logger.debug(
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
