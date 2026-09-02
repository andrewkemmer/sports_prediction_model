"""Smoke test — Model & Data Drift Monitor page, SPORT-DISPATCHED.

The NFL Model Monitor page now runs the SAME code as MLB (no sport-special
path). This test:

1. Writes a REPRESENTATIVE ``nfl_model_monitor_*.json`` (matching the exact
   shape the NFL backend ``nfl_monitor.build_model_monitor`` emits) into the
   real ``nfl-backend/data_delivery`` dir, so the page renders with real
   data (removed after the run).
2. Runs the ACTUAL ``model_monitor.py`` under ``sport=nfl`` and asserts the
   MLB-identical sections render: the last/next retrain + drift-alert health
   boxes, the upset-monitoring callout, the Feature Drift (PSI) matrix with
   status pills (incl. WARN), the Feature Coverage panel, the Model Ensemble
   table, the Rolling Brier timeline as a real Altair chart, and the Model
   Version History table — with no exceptions.
3. Runs the same page under ``sport=mlb`` and asserts it also runs clean
   (locally it warns on a missing MLB monitor artifact rather than crashing).

Run from the frontend/ directory:
    python -m test_monitor_smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent if FRONTEND_DIR.name == "frontend" else FRONTEND_DIR
NFL_DD = REPO_ROOT / "nfl-backend" / "data_delivery"

ARTIFACT_DATE = "20260831"
MONITOR_NAME = f"nfl_model_monitor_{ARTIFACT_DATE}.json"
MONITOR_PATH = NFL_DD / MONITOR_NAME

WRITTEN: list[Path] = []
# Path -> original bytes of a PRE-EXISTING (committed) artifact this test
# overwrites with a fixture; restored on cleanup, never deleted.
_BACKUPS: dict[Path, bytes] = {}


# ---------------------------------------------------------------------------
# Representative artifact construction (matches the emitted MLB-shaped schema)
# ---------------------------------------------------------------------------
def _monitor_record() -> dict:
    drift = [
        {"feature": "elo_diff", "current_mean": 22.4, "baseline_mean": 8.1,
         "psi": 0.31, "status": "ALERT", "weight_pct": 45.0,
         "n_baseline": 496, "n_current": 285},
        {"feature": "ewm_net_pts_diff", "current_mean": 3.2, "baseline_mean": 2.9,
         "psi": 0.12, "status": "WARN", "weight_pct": 21.0,
         "n_baseline": 496, "n_current": 285},
        {"feature": "div_game", "current_mean": 0.50, "baseline_mean": 0.51,
         "psi": 0.02, "status": "OK", "weight_pct": None,
         "n_baseline": 496, "n_current": 285},
    ]
    coverage = [
        {"feature": "elo_diff", "window": "decided pool", "n_games": 1960,
         "pct_measured": 100.0, "pct_nonnull": 100.0, "n_default_zero": 0,
         "status": "OK"},
        {"feature": "temp_f", "window": "decided pool", "n_games": 1960,
         "pct_measured": 3.0, "pct_nonnull": 3.0, "n_default_zero": 0,
         "status": "STARVED"},
    ]
    ensemble = [
        {"name": "xgboost", "weight": 0.45, "auc": 0.6911, "brier": 0.2040,
         "logloss": 0.6329, "n_eval": 1107},
        {"name": "lightgbm", "weight": 0.0, "auc": 0.61, "brier": 0.22,
         "logloss": 0.65, "n_eval": 1107},
    ]
    rolling_brier = [
        {"date": "2026-09-%02d" % d, "brier": round(0.205 + 0.002 * d, 4)}
        for d in range(1, 16)
    ]
    # Retrain-every-run semantics: last == the artifact date; next == +1 day
    # (the corrected MLB-shaped cadence, constant = 1, not the old +7).
    return {
        "last_retrained": "2026-08-31",
        "last_retrained_note": "Fresh model trained this run (sealed gate: ADOPT)",
        "next_retrain": "2026-09-01",
        "next_retrain_note": "next expected run in 1 day(s) (retrains every run)",
        "upset_note": "Model upset rate over the decided pool — 1,392 games scored; "
                      "see Calibration for the upset strip.",
        "feature_drift": drift,
        "features_metadata": {
            c["feature"]: {
                "definition": f"{c['feature']} — plain-language twin of "
                              "nfl_features.CANONICAL_SOURCE",
                "source": "nfl feature engine (strictly-trailing per-team "
                           "aggregates)",
                "tooltip": f"What: {c['feature']} — plain-language description.\n"
                           f"Consumed by: the 5-member moneyline blend.",
            }
            for c in drift
        },
        "feature_coverage": coverage,
        "ensemble": ensemble,
        "rolling_brier": rolling_brier,
        "rolling_brier_meta": {"window_days": 30, "min_games_per_day": 2,
                               "excluded_sparse_days": 0,
                               "calibrator_is_identity": False,
                               "map_scope_note": "Platt map deployed"},
        "brier_baseline": 0.23,
        "brier_baseline_label": "Constant home-edge",
        "version_history": [
            {"version": ARTIFACT_DATE, "date": "2026-08-31",
             "weights": {"xgboost": 0.45, "lightgbm": 0.0, "logistic": 0.0,
                         "randomforest": 0.0, "mlp": 0.0},
             "auc": 0.6911, "logloss": 0.6329, "ece_calibrated": 0.0290,
             "calibration": {"a": 1.233, "b": 0.130}}],
    }


def _stage(path: Path, data: bytes) -> None:
    """Write a fixture over ``path``, preserving any pre-existing (committed)
    artifact's bytes so cleanup can restore it rather than delete it."""
    if path.exists():
        _BACKUPS[path] = path.read_bytes()
    path.write_bytes(data)
    WRITTEN.append(path)


def _write_artifacts() -> None:
    NFL_DD.mkdir(parents=True, exist_ok=True)
    _stage(MONITOR_PATH, json.dumps(_monitor_record(), indent=2).encode("utf-8"))


def _remove_artifacts() -> None:
    for p in WRITTEN:
        try:
            if p in _BACKUPS:
                p.write_bytes(_BACKUPS.pop(p))  # restore committed artifact
            else:
                p.unlink()                       # fixture we created fresh
        except FileNotFoundError:
            pass
    WRITTEN.clear()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "caption", "success", "error",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            try:
                chunks.append(str(el.value))
            except Exception:
                pass
    return "\n".join(chunks)


def run() -> int:
    _write_artifacts()
    problems: list[str] = []
    try:
        at = AppTest.from_file(str(FRONTEND_DIR / "model_monitor.py"),
                               default_timeout=60)
        at.session_state["sport"] = "nfl"
        # Pin the page to the fixture date so the STAGED artifact renders
        # (the default selected date is the newest committed artifact, which
        # would silently bypass the fixture).
        at.session_state["selected_date"] = ARTIFACT_DATE
        at.run()

        if at.exception:
            problems.append("NFL PAGE RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in at.exception))

        text = _all_text(at)
        vcl = at.get("vega_lite_chart")

        # (1) header + health boxes
        for key, needle in [("header", "Model & Data Drift Monitor"),
                            ("last", "LAST RETRAIN"),
                            ("next", "NEXT RETRAIN"),
                            ("drift", "DRIFT ALERTS")]:
            if needle not in text:
                problems.append(f"missing [{key}] = {needle!r}")

        # (2) upset monitoring callout
        if "Upset Monitoring Note" not in text:
            problems.append("missing upset-monitoring callout")
        if "1,392 games scored" not in text:
            problems.append("upset note did not render its data")

        # (3) feature drift (PSI) matrix with WARN status pill
        if "Feature Drift Analysis (PSI Scores)" not in text:
            problems.append("missing feature-drift matrix section")
        if "ewm_net_pts_diff" not in text:
            problems.append("drift matrix missing a drift row")
        if "WARN" not in text:
            problems.append("drift matrix missing WARN status pill")

        # (3a) MLB-identical MODEL WEIGHT column: header + formatted weight +
        #      per-feature description label (sport-dispatched describe_feature)
        if "MODEL WEIGHT" not in text:
            problems.append("drift matrix missing the MODEL WEIGHT column header")
        if "45.00%" not in text:
            problems.append("drift matrix missing a formatted model-weight cell")
        if "Exponentially weighted" not in text and "point-in-time rating gap" not in text:
            problems.append("drift matrix missing the NFL feature description label")

        # (3b) retrain cards must never contradict: same-day persist renders
        #      "today" (not "0 days ago" / stale "7 days ago") and a +1-day
        #      next-retrain fires the "tonight" subtext.
        if "today" not in text:
            problems.append("LAST RETRAIN subtext missing the same-day 'today' suffix")
        if "0 days ago" in text or "7 days ago" in text:
            problems.append("LAST RETRAIN subtext shows a stale days-ago count")
        if "tonight" not in text:
            problems.append("NEXT RETRAIN subtext missing the 'tonight' suffix for +1 day")

        # (4) feature coverage panel (picks up the STARVED row)
        if "Feature Coverage (non-null / measured)" not in text:
            problems.append("missing feature-coverage section")
        if "STARVED" not in text:
            problems.append("coverage panel missing STARVED status")
        if "all windows healthy" in text:
            problems.append("coverage reported healthy despite a starved row")

        # (5) model ensemble table
        if "Model Ensemble" not in text:
            problems.append("missing model-ensemble section")
        if "XGBOOST" not in text.upper() and "xgboost" not in text.lower():
            problems.append("ensemble table missing the xgboost member row")

        # (6) rolling Brier timeline renders a real Altair chart
        if len(vcl) == 0:
            problems.append("rolling-Brier timeline did NOT render an Altair chart")
        if "Rolling Brier Score (Last 30 Days)" not in text:
            problems.append("missing rolling-Brier section")

        # (7) model version history table (populated row)
        if "Model Version History" not in text:
            problems.append("missing version-history section")
        if "No version history yet" in text:
            problems.append("version history empty instead of a populated row")

        if problems:
            print("MONITOR SMOKE TEST — FAIL (sport=nfl)")
            for p in problems:
                print("  -", p)
            return 1

        print("MONITOR SMOKE TEST — PASS (sport=nfl)")
        print(f"  - no exceptions; {len(vcl)} Altair chart(s) rendered")
        print("  - health boxes + upset callout + drift matrix (WARN) + coverage"
              " (STARVED) + ensemble + rolling Brier + version history")

        # sport=mlb must still run the SAME shared path, no exception.
        mlb = AppTest.from_file(str(FRONTEND_DIR / "model_monitor.py"),
                                default_timeout=60)
        mlb.session_state["sport"] = "mlb"
        mlb.run()
        if mlb.exception:
            prob = "\n  ".join(str(e.value) for e in mlb.exception)
            print("MONITOR SMOKE TEST — FAIL (sport=mlb)")
            print("  - mlb path raised:\n    " + prob)
            return 1
        print("  - sport=mlb path clean (no exception)")
        return 0
    finally:
        _remove_artifacts()


if __name__ == "__main__":
    sys.exit(run())