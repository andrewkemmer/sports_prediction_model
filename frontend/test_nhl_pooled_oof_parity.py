"""NHL dashboard ↔ production pooled-OOF parity (2026-10-07 review).

The Model Calibration dashboard and the Model Monitor TOTAL row both render
the SAME headline block — the deployed blend's **pooled out-of-fold** scores
over the grading population (``oof_regular``): regular-season windows at or
above ``min_val_fold_games``, i.e. the rows the blend weights, the pooled
calibrator and the headline metrics are all fitted/earned on. ``oof_all``
(the whole OOF frame incl. provisional + postseason) is exposed separately
"so the split hides nothing", but it is NOT the headline.

This test pins that contract against the REAL shipped artifacts (no
synthetic fixtures): the calibration ``metrics`` block the dashboard reads
must equal (a) the Model Monitor ``metrics`` block (cross-artifact) and (b)
the pipeline's own ``fold_geometry.oof_regular`` pooled-OOF scores (the
production run), with ``n_games`` equal to the grading count and the
reliability buckets partitioning that same population exactly.

A dashboard that drifted onto ``oof_all`` (or a stale/partial metrics block)
would surface here as a number mismatch, not as a silently wrong KPI card.

Run from the frontend/ directory:
    python -m pytest test_nhl_pooled_oof_parity.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent if FRONTEND_DIR.name == "frontend" else FRONTEND_DIR
NHL_DD = REPO_ROOT / "nhl-backend" / "data_delivery"

# Half-unit of the COARSEST rounding the emitter applies to the headline
# block (auc/brier are stored at 3 decimals), so a real number and its
# stored rounding agree within this bound while a genuine drift (e.g. a
# dashboard on oof_all vs oof_regular) is far larger.
ROUND_TOL = 5e-4

_TWIN = {
    "auc": "auc_calibrated",
    "brier": "brier_calibrated",
    "logloss": "logloss_calibrated",
    "ece": "ece_calibrated",
}


def _latest(prefix: str) -> Path:
    files = sorted(NHL_DD.glob(f"{prefix}_*.json"))
    assert files, f"no {prefix}_*.json shipped under {NHL_DD}"
    return files[-1]


@pytest.fixture(scope="module")
def artifacts() -> tuple[dict, dict]:
    cal = json.loads(_latest("nhl_calibration").read_text(encoding="utf-8"))
    mon = json.loads(_latest("nhl_model_monitor").read_text(encoding="utf-8"))
    return cal, mon


def _oof_regular(mon: dict) -> dict:
    geo = mon.get("fold_geometry", {})
    oof = geo.get("oof_regular")
    assert isinstance(oof, dict) and oof.get("n"), \
        "model_monitor is missing fold_geometry.oof_regular (the pooled OOF run)"
    return oof


def test_calibration_and_monitor_share_one_headline_block(artifacts) -> None:
    """Both dashboard pages must read the SAME pooled-OOF metrics block."""
    cal, mon = artifacts
    cal_m, mon_m = cal.get("metrics", {}), mon.get("metrics", {})
    assert cal_m, "nhl_calibration is missing its metrics block"
    assert mon_m, "nhl_model_monitor is missing its metrics block"
    for k in ("auc", "brier", "logloss", "ece"):
        assert k in cal_m and k in mon_m, f"headline metric {k!r} missing"
        assert abs(float(cal_m[k]) - float(mon_m[k])) <= ROUND_TOL, (
            f"dashboard pages disagree on {k}: calibration={cal_m[k]} "
            f"vs model_monitor={mon_m[k]}")


def test_headline_metrics_are_the_pooled_oof_run(artifacts) -> None:
    """The dashboard KPIs must be the production pooled OOF (oof_regular)."""
    cal, mon = artifacts
    cal_m = cal["metrics"]
    oof = _oof_regular(mon)
    for k in ("auc", "brier", "logloss", "ece"):
        assert abs(float(cal_m[k]) - float(oof[k])) <= ROUND_TOL, (
            f"headline {k}={cal_m[k]} does not match the production pooled "
            f"OOF (oof_regular) {k}={oof[k]} — the dashboard is not showing "
            f"the pooled OOF run")


def test_n_games_is_the_grading_count(artifacts) -> None:
    """n = the grading population the pooled OOF scores, not a subset."""
    cal, mon = artifacts
    n = int(cal.get("n_games", 0))
    oof = _oof_regular(mon)
    assert n == int(oof["n"]), (
        f"dashboard n_games={n} != pooled-OOF grading n={oof['n']}")
    # The reliability buckets must partition that population exactly once.
    buckets = cal.get("calibration_buckets") or cal.get("calibration_curve") or []
    total = sum(int(b.get("count", 0) or 0) for b in buckets)
    if buckets:
        assert total == n, (
            f"reliability buckets cover {total} games but the pooled OOF "
            f"headline is n={n} — games are dropped or double-counted")


def test_identity_calibrator_twins_match_raw(artifacts) -> None:
    """When the calibrator is gated out (identity), the dashboard's raw →
    calibrated pairs must be equal — the ``0.241 → 0.241`` display."""
    cal, _ = artifacts
    m = cal["metrics"]
    if not m.get("calibrator_gated_out", False):
        pytest.skip("this run deployed a fitted calibrator (not identity)")
    for raw, cal_k in _TWIN.items():
        if raw in m and cal_k in m:
            assert abs(float(m[raw]) - float(m[cal_k])) <= ROUND_TOL, (
                f"calibrator_gated_out but {raw}={m[raw]} != "
                f"{cal_k}={m[cal_k]} — raw→calibrated display would lie")
    # Provenance: the calibration block itself must be the identity map.
    prov = cal.get("calibration", {}) or {}
    assert prov.get("method") in ("identity", None) or not prov, (
        f"calibrator_gated_out yet calibration.method={prov.get('method')!r}")
