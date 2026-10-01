"""Pin the deployed-calibrator gate added reviewing the 2026-09-30 run.

That Kaggle run shipped a fitted favored-space Platt map even though its own
honest prequential rehearsal said the map hurts (ECE 0.0068 -> 0.0088,
log-loss 0.6843 -> 0.6847) — the second consecutive run with that inversion
(v2026.09.29: 0.0046 -> 0.0052 / 0.6844 -> 0.6849), while v2026.09.24–09.28
(xgb-heavy blends) had shown calibration helping. The gate exists so the
deployed map is withheld exactly when its causal rehearsal is unambiguously
harmful and ships again the moment the evidence flips back. Each test pins
one side of that boundary:

* both headline metrics worse (log-loss AND ECE)  -> gate fires, with a
  reason string quoting both pairs (the run-log evidence line);
* both better, or mixed evidence                  -> fitted map stands;
* equal values                                    -> not strictly worse, stands;
* missing keys / non-finite values                -> never fires.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import calibration  # noqa: E402


def _metrics(ll: float, ece: float) -> dict:
    return {"logloss": ll, "ece": ece, "brier": 0.245}


# ── 1. unambiguous harm fires the gate, with the log evidence line ──────────

def test_gate_fires_when_both_metrics_worse():
    raw = _metrics(0.6843, 0.0068)
    cal = _metrics(0.6847, 0.0088)
    gate, reason = calibration.should_gate_calibrator(raw, cal)
    assert gate is True
    # The reason quotes both pairs — it is the run-log evidence line.
    assert "0.0068" in reason and "0.0088" in reason
    assert "0.6843" in reason and "0.6847" in reason


def test_gate_fires_on_tiny_unambiguous_harm():
    gate, _ = calibration.should_gate_calibrator(
        _metrics(0.6800, 0.0040), _metrics(0.6801, 0.0041))
    assert gate is True


# ── 2. benefit or mixed evidence keeps the fitted map ───────────────────────

def test_gate_keeps_map_when_calibration_helps():
    # v2026.09.27 shape: ECE 0.0059 -> 0.0027, ll 0.6783 -> 0.6773.
    gate, reason = calibration.should_gate_calibrator(
        _metrics(0.6783, 0.0059), _metrics(0.6773, 0.0027))
    assert gate is False
    assert reason == ""


def test_gate_keeps_map_on_mixed_evidence_ll_worse_ece_better():
    gate, _ = calibration.should_gate_calibrator(
        _metrics(0.6843, 0.0068), _metrics(0.6850, 0.0050))
    assert gate is False


def test_gate_keeps_map_on_mixed_evidence_ll_better_ece_worse():
    gate, _ = calibration.should_gate_calibrator(
        _metrics(0.6843, 0.0068), _metrics(0.6830, 0.0090))
    assert gate is False


def test_gate_keeps_map_on_exact_ties():
    same = _metrics(0.6843, 0.0068)
    gate, _ = calibration.should_gate_calibrator(same, dict(same))
    assert gate is False


# ── 3. measurement gaps never fire the gate ─────────────────────────────────

def test_gate_never_fires_on_missing_metrics():
    assert calibration.should_gate_calibrator({}, {}) == (False, "")
    assert calibration.should_gate_calibrator(
        {"logloss": 0.68}, {"logloss": 0.70}) == (False, "")
    assert calibration.should_gate_calibrator(
        {"ece": 0.0068}, {"ece": 0.0088}) == (False, "")


def test_gate_never_fires_on_non_finite_values():
    bad = float("nan")
    assert calibration.should_gate_calibrator(
        _metrics(bad, 0.0068), _metrics(0.6847, 0.0088)) == (False, "")
    assert calibration.should_gate_calibrator(
        _metrics(0.6843, 0.0068), _metrics(bad, 0.0088)) == (False, "")
