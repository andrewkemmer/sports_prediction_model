"""Gated-calibrator note (2026-10-08 dashboard review).

The 2026-10-08 calibration dashboard (MLB, ``calibration_20261008.json``)
rendered KPI twins like ``0.2446 → 0.2446`` captioned "· after
calibration" with NO banner anywhere. Root cause: the prequential gate had
deployed an IDENTITY calibrator for that run (``calibrator_gated_out:
true``, ``calibration.method == "identity"``) — no map ever ran, so raw
and calibrated are the same probabilities by design. The identical values
are honest; presenting them as "after calibration" with no explanation was
the defect.

Pins, both directions:
  * identity runs render the Calibration Gate note, the KPI captions stop
    claiming a transformation, and the Platt banner stays away;
  * fitted-map runs (favored_platt_floor) keep the Post-Hoc Recalibration
    banner and the gate note stays away.

Stages fixtures at far-future dates (20991220 identity / 20991221 fitted)
so ``available_dates()[0]`` picks them, mirroring test_calibration_smoke's
``20991231`` trick without sharing its dates (or its cache keys).
"""
from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend"
NFL_DD = REPO_ROOT / "nfl-backend" / "data_delivery"

IDENTITY_DATE = "20991220"
FITTED_DATE = "20991221"

_BACKUPS: dict[Path, bytes] = {}
_WRITTEN: list[Path] = []

_COUNTS = [1030, 189, 88]          # favored-only buckets, sum == n_eval
_N_EVAL = sum(_COUNTS)


def _buckets() -> list[dict]:
    rows = []
    for bucket, mp, ma, n in [
        ("50–60%", 0.5431, 0.5352, _COUNTS[0]),
        ("60–70%", 0.6319, 0.6204, _COUNTS[1]),
        ("70%+", 0.7402, 0.7338, _COUNTS[2]),
    ]:
        rows.append({"bucket": bucket, "mean_predicted": mp,
                     "mean_actual": ma, "count": n,
                     "gap": round(mp - ma, 4)})
    return rows


def _record(gated: bool) -> dict:
    """A calibration_*.json in the CURRENT production shape.

    ``gated=True`` mirrors calibration_20261008.json exactly: identity map,
    raw == calibrated twins, ``calibrator_gated_out: true``. ``gated=False``
    mirrors the fitted runs (favored_platt_floor with a, b, n, floor).
    """
    raw = {"auc": 0.5719, "brier": 0.2446, "logloss": 0.6823, "ece": 0.0089}
    cal = dict(raw) if gated else {
        "auc": 0.5719, "brier": 0.2440, "logloss": 0.6816, "ece": 0.0067}
    metrics = dict(cal)
    metrics["brier_calibrated"] = cal["brier"]
    metrics["logloss_calibrated"] = cal["logloss"]
    metrics["ece_calibrated"] = cal["ece"]
    metrics["calibrator_gated_out"] = gated
    if gated:
        calibration = {
            "method": "identity",
            "params": None,
            "metrics_raw": {"brier": 0.2446, "logloss": 0.6823, "ece": 0.0089},
            "metrics_calibrated": {"auc": 0.5719, "brier": 0.2446,
                                   "logloss": 0.6823, "ece": 0.0089},
        }
    else:
        calibration = {
            "method": "favored_platt_floor",
            "params": {"method": "favored_platt_floor", "a": 1.0161,
                       "b": 0.0188, "n": _N_EVAL, "floor": 0.5},
            "metrics_raw": {"brier": 0.2446, "logloss": 0.6823, "ece": 0.0089},
            "metrics_calibrated": {"auc": 0.5719, "brier": 0.2440,
                                   "logloss": 0.6816, "ece": 0.0067},
        }
    return {
        "date": IDENTITY_DATE if gated else FITTED_DATE,
        "n_games": 4,             # slate-size legacy field (page uses n_eval)
        "n_eval": _N_EVAL,
        "league_total": 4,
        "trained_at": "2026-10-08T05:44:00.000000Z",
        "metrics": metrics,
        "calibration_buckets": _buckets(),
        "calibration": calibration,
        "daily": [],
    }


def _stage(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path not in _BACKUPS:
        _BACKUPS[path] = path.read_bytes()
    path.write_bytes(data)
    _WRITTEN.append(path)


def _restore() -> None:
    for p in _WRITTEN:
        try:
            if p in _BACKUPS:
                p.write_bytes(_BACKUPS.pop(p))
            else:
                p.unlink()
        except FileNotFoundError:
            pass
    _WRITTEN.clear()


def _render() -> AppTest:
    """Render the Calibration page for the staged NFL fixtures."""
    import streamlit as st
    # The union date set is per-sport cached (ttl 300s, shared across
    # AppTest runs in this process). NOTE: the per-function
    # ``_available_dates_cached.clear()`` does NOT bust the cache for an
    # in-script run on streamlit 1.63 — measured: after phase 1 the phase-2
    # render still saw ``['20991220', ...]`` through it, while the global
    # ``st.cache_data.clear()`` sees the freshly staged ``20991221``. Clear
    # the whole cache_data store before every phase; each phase re-reads
    # exactly what it just staged, so nothing else depends on cross-phase
    # cache state.
    st.cache_data.clear()
    at = AppTest.from_file(str(FRONTEND_DIR / "model_calibration.py"),
                           default_timeout=60)
    at.session_state["sport"] = "nfl"
    at.run()
    return at


def _text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "caption", "success", "error",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            try:
                chunks.append(str(el.value))
            except Exception:  # noqa: BLE001
                pass
    return "\n".join(chunks)


def test_identity_run_renders_the_gate_note_not_the_platt_banner():
    """Gated-out calibrator: explain the identical twins, claim no change."""
    _stage(NFL_DD / f"nfl_calibration_{IDENTITY_DATE}.json",
           json.dumps(_record(gated=True), indent=2).encode("utf-8"))
    try:
        at = _render()
        assert not at.exception, (
            "identity-run page raised:\n"
            + "\n".join(str(e.value) for e in at.exception))
        text = _text(at)

        # The note renders and says WHY the twins read identically.
        assert "Calibration Gate:" in text, "missing gated-calibrator note"
        assert "Gated out · identity" in text
        assert "no recalibration map" in text
        # The KPI captions stop claiming a calibration that never ran.
        assert "calibrator off (identity)" in text, (
            "KPI cards still caption 'after calibration' on an identity run")
        assert " · after calibration" not in text
        # The twins still show, honest and explained.
        assert "0.2446 → 0.2446" in text
        # The Platt banner belongs to fitted maps only.
        assert "Post-Hoc Recalibration" not in text, (
            "Platt banner rendered for an identity run")
    finally:
        _restore()


def test_fitted_map_run_keeps_the_platt_banner_without_the_note():
    """Non-vacuity: a deployed map renders its Post-Hoc banner and the gate
    note must not leak into fitted-map runs."""
    # Both dates staged so the newest (fitted) wins available_dates()[0];
    # the identity twin stays staged to prove the pick, not a fallback.
    _stage(NFL_DD / f"nfl_calibration_{IDENTITY_DATE}.json",
           json.dumps(_record(gated=True), indent=2).encode("utf-8"))
    _stage(NFL_DD / f"nfl_calibration_{FITTED_DATE}.json",
           json.dumps(_record(gated=False), indent=2).encode("utf-8"))
    try:
        at = _render()
        assert not at.exception, (
            "fitted-run page raised:\n"
            + "\n".join(str(e.value) for e in at.exception))
        text = _text(at)

        assert "Post-Hoc Recalibration" in text, (
            "fitted-map run lost its Platt recalibration banner")
        assert "Calibration Gate:" not in text, (
            "gate note leaked into a fitted-map run")
        assert "calibrator off (identity)" not in text
        assert " · after calibration" in text, (
            "fitted runs must keep the 'after calibration' caption")
    finally:
        _restore()
