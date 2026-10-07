"""Smoke test — Model & Data Drift Monitor page, SPORT-DISPATCHED.

The NFL Model Monitor page now runs the SAME code as MLB (no sport-special
path). This test:

1. Writes a REPRESENTATIVE ``nfl_model_monitor_*.json`` (matching the exact
   MLB-identical shape the NFL backend ``monitoring.write_monitor_json``
   emits) into the real ``nfl-backend/data_delivery`` dir, so the page
   renders with real data (removed after the run) — plus the DATED artifact
   before it, carrying one version-history row each, so the family fold is
   exercised against the shape the published artifacts actually have.
2. Runs the ACTUAL ``model_monitor.py`` under ``sport=nfl`` and asserts the
   MLB-identical sections render: the last/next retrain + drift-alert health
   boxes, the upset-monitoring callout, the Feature Drift (PSI) matrix with
   status pills (incl. WARN), the Feature Coverage panel, the Model Ensemble
   table, the Rolling Brier timeline as a real Altair chart, and the Model
   Version History table — with no exceptions.
3. Runs the same page under ``sport=mlb`` against a staged minimal MLB
   monitor fixture (backed up/restored if a committed artifact exists) and
   asserts it renders clean WHILE pinning describe_feature's served-metadata
   precedence: backend-authored per-side summaries (sp_era_home,
   bullpen_whip_10g_home, home_elo) beat the legacy diff-text + side-suffix
   and bare-name fallbacks; the static dict still answers unserved rows
   (elo_diff); a degenerate served summary equal to the bare column name is
   ignored (is_home keeps its dict wording).
4. Runs the same page under ``sport=nba`` against a staged NBA fixture in
   the NEW report contract and asserts MLB-identical structure: every drift
   row carries a served-metadata label + tooltip (never "(no detailed
   metadata)" / never the diff-twin side-suffix mislabel), the coverage
   panel renders MLB's columns, and the Model Version History table renders
   the accumulated retrain rows under MLB's exact columns (the
   "model version history is missing" defect).

Run from the frontend/ directory:
    python -m test_monitor_smoke
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent if FRONTEND_DIR.name == "frontend" else FRONTEND_DIR
NFL_DD = REPO_ROOT / "nfl-backend" / "data_delivery"
MLB_DD = REPO_ROOT / "mlb-backend" / "data_delivery"
NBA_DD = REPO_ROOT / "nba-backend" / "data_delivery"

# Newer than any committed artifact so the fixture is the one the page's
# newest-date resolution picks up (removed after the run).
ARTIFACT_DATE = "20260923"
MONITOR_NAME = f"nfl_model_monitor_{ARTIFACT_DATE}.json"
MONITOR_PATH = NFL_DD / MONITOR_NAME
MLB_MONITOR_PATH = MLB_DD / f"model_monitor_{ARTIFACT_DATE}.json"
# A second dated NFL artifact carrying ONE version-history row (the shape
# the published nfl_model_monitor_* family actually has) — the page must
# fold the family so the history table shows BOTH records, MLB's rolling
# presentation, instead of the served artifact's single row.
PREV_ARTIFACT_DATE = "20260922"
PREV_MONITOR_NAME = f"nfl_model_monitor_{PREV_ARTIFACT_DATE}.json"
PREV_MONITOR_PATH = NFL_DD / PREV_MONITOR_NAME
NBA_MONITOR_PATH = NBA_DD / f"nba_model_monitor_{ARTIFACT_DATE}.json"

WRITTEN: list[Path] = []
# Path -> original bytes of a PRE-EXISTING (committed) artifact this test
# overwrites with a fixture; restored on cleanup, never deleted.
_BACKUPS: dict[Path, bytes] = {}


# ---------------------------------------------------------------------------
# Representative artifact construction (matches the emitted MLB-shaped schema)
# ---------------------------------------------------------------------------
def _monitor_record() -> dict:
    # MLB-shaped drift rows: raw psi + the noise-adjusted fields the emitter
    # now ships; the mixed ALERT+WARN table drives the card-count assertion.
    drift = [
        {"feature": "elo_diff", "current_mean": 22.4, "baseline_mean": 8.1,
         "psi": 0.31, "psi_adjusted": 0.22, "noise_floor": 0.004,
         "mean_shift": 14.3, "shift_se": 2.1, "location_shift": True,
         "status": "ALERT", "weight_pct": 45.0,
         "n_baseline": 1930, "n_current": 285},
        {"feature": "ewm_net_pts_diff", "current_mean": 3.2, "baseline_mean": 2.9,
         "psi": 0.12, "psi_adjusted": 0.11, "noise_floor": 0.004,
         "mean_shift": 0.3, "shift_se": 0.1, "location_shift": True,
         "status": "WARN", "weight_pct": 21.0,
         "n_baseline": 1930, "n_current": 285},
        {"feature": "div_game", "current_mean": 0.50, "baseline_mean": 0.51,
         "psi": 0.02, "psi_adjusted": 0.0, "noise_floor": 0.004,
         "mean_shift": -0.01, "shift_se": 0.02, "location_shift": False,
         "status": "OK", "weight_pct": None,
         "n_baseline": 1930, "n_current": 285},
        # Constant feature: the emitter ships psi=None (real artifacts do —
        # e.g. is_home). The page must render '—', never raise TypeError.
        {"feature": "is_home", "current_mean": 1.0, "baseline_mean": 1.0,
         "psi": None, "psi_adjusted": None, "noise_floor": 0.004,
         "mean_shift": 0.0, "shift_se": 0.0, "location_shift": False,
         "status": "OK", "weight_pct": 0.0,
         "n_baseline": 1930, "n_current": 285},
    ]
    # MLB's exact 9-key coverage row (feature, window, n_games, n_nonnull,
    # pct_nonnull, n_measured, pct_measured, n_default_zero, status) — the
    # 2026-10-07 structure parity: the report must carry the same fields
    # MLB's model_monitor ships, counts included.
    coverage = [
        {"feature": "elo_diff", "window": "decided pool", "n_games": 1960,
         "n_nonnull": 1960, "pct_nonnull": 100.0,
         "n_measured": 1960, "pct_measured": 100.0, "n_default_zero": 0,
         "status": "OK"},
        {"feature": "temp_f", "window": "decided pool", "n_games": 1960,
         "n_nonnull": 59, "pct_nonnull": 3.0,
         "n_measured": 59, "pct_measured": 3.0, "n_default_zero": 0,
         "status": "STARVED"},
        # Real 2026-09-29 MLB artifact shape: the offspeed exp2 category
        # measures ~56% in BOTH windows BY CONSTRUCTION (sparse offspeed
        # PA vs the category PA floor) — the panel must label it
        # structural rather than paging it as a fetch regression.
        {"feature": "exp2_cat_k_offspeed_diff", "window": "current",
         "n_games": 87, "n_nonnull": 49, "pct_nonnull": 56.0,
         "n_measured": 49, "pct_measured": 56.0, "n_default_zero": 0,
         "status": "LOW_COVERAGE"},
        # NFL 2026-09-29: documented-policy absence arrives pre-classified as
        # STRUCTURAL with the backend-declared reason — the panel must render
        # it calm (reason text present) while starved/low rows still page.
        {"feature": "temp_f", "window": "current", "n_games": 60,
         "n_nonnull": 41, "pct_nonnull": 68.33,
         "n_measured": 41, "pct_measured": 68.33, "n_default_zero": 0,
         "status": "STRUCTURAL", "structural_reason": "indoor/closed"},
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
        "upset_note": "Model upset rate over the walk-forward history "
                      "(pooled OOF + sealed) — 1,392 games scored; "
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
        # Headline block = the deployed blend's own pooled OOF scores; the
        # Model Ensemble TOTAL row must render them next to the members so
        # blend-vs-strongest-member is visible in the table itself.
        "metrics": {"auc": 0.6995, "brier": 0.2035, "logloss": 0.6321,
                    "ece": 0.0211, "brier_calibrated": 0.2039,
                    "logloss_calibrated": 0.6328, "ece_calibrated": 0.0207,
                    "calibrator_gated_out": False},
        "rolling_brier": rolling_brier,
        "rolling_brier_meta": {"window_days": 30, "min_games_per_day": 2,
                               "excluded_sparse_days": 0,
                               "calibrator_is_identity": False,
                               "map_scope_note": "Platt map deployed"},
        "brier_baseline": 0.23,
        "brier_baseline_label": "Constant home-edge",
        # MLB's model_version_history row schema (version vYYYY.MM.DD, ISO
        # date, roster weights, the pooled metric keys, the deployed map's
        # {a, b, n, method, floor}) — the 2026-10-07 structure parity.
        "version_history": [
            {"version": "v2026.08.31", "date": "2026-08-31",
             "weights": {"xgboost": 0.45, "lightgbm": 0.0, "elasticnet": 0.0,
                         "randomforest": 0.0, "mlp": 0.0},
             "auc": 0.6911, "brier": 0.2040, "logloss": 0.6329,
             "ece": 0.0211, "brier_calibrated": 0.2039,
             "logloss_calibrated": 0.6328, "ece_calibrated": 0.0290,
             "calibration": {"a": 1.233, "b": 0.130, "n": 1200,
                             "method": "favored_platt_floor",
                             "floor": 0.5}}],
    }


def _prev_monitor_record() -> dict:
    """The dated artifact BEFORE the served one, carrying exactly one
    version-history row — the shape every published ``nfl_model_monitor_*``
    artifact has. The page's family fold must surface it beside the served
    row (the "model version history is missing records" defect)."""
    return {
        "date": PREV_ARTIFACT_DATE,
        "version": "v2026.08.30",
        "version_history": [
            {"version": "v2026.08.30", "date": "2026-08-30",
             "weights": {"xgboost": 0.50, "lightgbm": 0.0, "elasticnet": 0.0,
                         "randomforest": 0.0, "mlp": 0.0},
             "auc": 0.6880, "brier": 0.2055, "logloss": 0.6344,
             "ece": 0.0220, "brier_calibrated": 0.2058,
             "logloss_calibrated": 0.6349, "ece_calibrated": 0.0301,
             "calibration": {"a": 1.201, "b": 0.118, "n": 1150,
                             "method": "favored_platt_floor",
                             "floor": 0.5}}],
    }


def _mlb_monitor_record() -> dict:
    """Minimal MLB monitor fixture exercising the served-metadata label path.

    Each drift row pins one branch of describe_feature's precedence:
    - sp_era_home: served summary WINS over the legacy diff-text + side
      fallback (the 2026-09-27 mislabel: "Home SP season-to-date ERA −
      away SP — home team").
    - bullpen_whip_10g_home: served summary wins over the legacy BARE-NAME
      leak (no *_diff entry exists for the stem).
    - home_elo: served summary wins over the legacy bare-name leak.
    - elo_diff: no served entry -> the static dict exact match must still
      answer (fallback intact).
    - is_home: served entry whose summary equals the bare name is REJECTED
      (degenerate-summary guard) -> the dict's exact match answers, so the
      baseline feature keeps its real wording.
    """
    drift = [
        {"feature": "sp_era_home", "current_mean": 4.21, "baseline_mean": 4.08,
         "psi": 0.068, "psi_adjusted": 0.002, "noise_floor": 0.066,
         "mean_shift": 0.13, "shift_se": 0.15, "location_shift": False,
         "status": "OK", "weight_pct": 1.30, "n_baseline": 273, "n_current": 91},
        {"feature": "bullpen_whip_10g_home", "current_mean": 1.36,
         "baseline_mean": 1.34, "psi": 0.175, "psi_adjusted": 0.11,
         "noise_floor": 0.066, "mean_shift": 0.02, "shift_se": 0.04,
         "location_shift": False, "status": "OK", "weight_pct": 1.03,
         "n_baseline": 272, "n_current": 90},
        {"feature": "home_elo", "current_mean": 1499.6, "baseline_mean": 1498.3,
         "psi": 0.311, "psi_adjusted": 0.245, "noise_floor": 0.066,
         "mean_shift": 1.3, "shift_se": 3.0, "location_shift": False,
         "status": "OK", "weight_pct": 1.04, "n_baseline": 273, "n_current": 91},
        {"feature": "elo_diff", "current_mean": -2.59, "baseline_mean": -2.88,
         "psi": 0.181, "psi_adjusted": 0.115, "noise_floor": 0.066,
         "mean_shift": 0.29, "shift_se": 4.2, "location_shift": False,
         "status": "OK", "weight_pct": 2.40, "n_baseline": 273, "n_current": 91},
        {"feature": "is_home", "current_mean": 1.0, "baseline_mean": 1.0,
         "psi": None, "psi_adjusted": None, "noise_floor": 0.066,
         "mean_shift": 0.0, "shift_se": 0.0, "location_shift": False,
         "status": "OK", "weight_pct": 0.0, "n_baseline": 273, "n_current": 91},
    ]
    served = {
        "sp_era_home": {
            "summary": "Starting-pitcher earned-run average — home team",
            "tooltip": "What: Starting-pitcher earned-run average — home team."},
        "bullpen_whip_10g_home": {
            "summary": "Bullpen walks+hits per inning — home team",
            "tooltip": "What: Bullpen walks+hits per inning — home team."},
        "home_elo": {
            "summary": "Home team Elo rating (level)",
            "tooltip": "What: Home team Elo rating (level)."},
        # Degenerate entry: summary == the bare column name must be ignored.
        "is_home": {"summary": "is_home", "tooltip": "What: is_home."},
    }
    return {
        "date": ARTIFACT_DATE,
        "feature_drift": drift,
        "features_metadata": served,
        # Member-card numbers must derive from EACH sport's own backend
        # config; the MLB leg pins the per-sport isolation (the 2026-09-29
        # defect: one process, one `config` module — the first sport
        # imported mislabeled the second sport's card).
        "ensemble": [
            {"name": "xgboost", "weight": 0.6, "auc": 0.585, "brier": 0.21,
             "logloss": 0.63, "n_eval": 273},
        ],
    }


def _nba_monitor_record() -> dict:
    """NBA fixture in the MLB report contract (the 2026-10-07 parity work).

    Pins the three NBA dashboard defects against MLB's structure:
    - served-metadata labels + tooltips on every drift row (the "(no detailed
      metadata)" / bare-name leak), with elo_home carrying its OWN side
      wording (never the diff twin + "— home team" mislabel), elo_diff
      falling back to the static dict, and a degenerate is_home summary
      (== the bare name) rejected so the dict wording wins;
    - feature coverage rows in MLB's 9-key structure;
    - a POPULATED Model Version History (accumulated retrain rows in MLB's
      snapshot schema) instead of "No version history yet".
    """
    drift = [
        {"feature": "elo_diff", "current_mean": 12.5, "baseline_mean": 8.1,
         "psi": 0.21, "psi_adjusted": 0.15, "noise_floor": 0.06,
         "mean_shift": 4.4, "shift_se": 1.2, "location_shift": True,
         "status": "WARN", "weight_pct": 45.0,
         "n_baseline": 200, "n_current": 31},
        {"feature": "elo_home", "current_mean": 1611.2, "baseline_mean": 1588.0,
         "psi": 0.32, "psi_adjusted": 0.26, "noise_floor": 0.06,
         "mean_shift": 23.2, "shift_se": 4.1, "location_shift": True,
         "status": "ALERT", "weight_pct": 0.3,
         "n_baseline": 200, "n_current": 31},
        # Constant feature: psi=None renders '—', never a TypeError.
        {"feature": "is_home", "current_mean": 1.0, "baseline_mean": 1.0,
         "psi": None, "psi_adjusted": None, "noise_floor": 0.06,
         "mean_shift": 0.0, "shift_se": 0.0, "location_shift": False,
         "status": "OK", "weight_pct": 0.0,
         "n_baseline": 200, "n_current": 31},
    ]
    coverage = [
        {"feature": "elo_diff", "window": "baseline", "n_games": 200,
         "n_nonnull": 200, "pct_nonnull": 100.0, "n_measured": 200,
         "pct_measured": 100.0, "n_default_zero": 0, "status": "OK"},
        {"feature": "elo_home", "window": "current", "n_games": 31,
         "n_nonnull": 31, "pct_nonnull": 100.0, "n_measured": 8,
         "pct_measured": 25.8, "n_default_zero": 0, "status": "STARVED"},
    ]
    served = {
        # Served summary WINS (the exact-side wording, not the diff twin's
        # text with a tacked-on side).
        "elo_home": {
            "summary": "Entering Elo rating — home team",
            "tooltip": "What: Entering Elo rating — home team."},
        # Degenerate entry: summary == the bare column name is REJECTED.
        "is_home": {"summary": "is_home", "tooltip": "What: is_home."},
    }
    return {
        "date": ARTIFACT_DATE,
        "version": "v2026.09.23",
        "last_retrained": ARTIFACT_DATE,
        "last_retrained_note": "Fresh NBA model trained this run",
        "next_retrain": "20260930",
        "next_retrain_note": "next expected run in 7 day(s)",
        "upset_note": "NBA upset rate is computed from settled walk-forward history.",
        "feature_drift": drift,
        "features_metadata": served,
        "feature_coverage": coverage,
        "ensemble": [
            {"name": "elasticnet", "weight": 0.8, "auc": 0.7339,
             "brier": 0.2074, "logloss": 0.6015, "n_eval": 2130},
            {"name": "lightgbm", "weight": 0.2, "auc": 0.7304,
             "brier": 0.2089, "logloss": 0.6051, "n_eval": 2130},
        ],
        # Headline block = MLB's 8-key shape (raw + calibrated twins).
        "metrics": {"auc": 0.7345, "brier": 0.2073, "logloss": 0.6013,
                    "ece": 0.0288, "brier_calibrated": 0.2073,
                    "logloss_calibrated": 0.6013, "ece_calibrated": 0.0276,
                    "calibrator_gated_out": False},
        "rolling_brier": [
            {"date": "2026-09-%02d" % d, "brier": round(0.205 + 0.001 * d, 4),
             "games": 12}
            for d in range(1, 16)
        ],
        "rolling_brier_meta": {"window_days": 30, "min_games_per_day": 1,
                               "excluded_sparse_days": 0,
                               "calibrator_is_identity": False,
                               "map_scope_note": "Points use the prequential "
                               "per-fold calibration layer (fit on prior OOF "
                               "folds only)."},
        "brier_baseline": 0.4554,
        "brier_baseline_label": "Constant home-edge",
        # Accumulated retrain rows in MLB's snapshot schema — the table must
        # render both, oldest first, under MLB's columns.
        "version_history": [
            {"version": "v2026.09.22", "date": "2026-09-22",
             "weights": {"elasticnet": 0.79, "lightgbm": 0.15,
                         "xgboost": 0.06},
             "auc": 0.7340, "brier": 0.2072, "logloss": 0.6012,
             "ece": 0.0288, "brier_calibrated": 0.2073,
             "logloss_calibrated": 0.6013, "ece_calibrated": 0.0276,
             "calibration": {"a": 1.103, "b": 0.028, "n": 2130,
                             "method": "favored_platt_floor",
                             "floor": 0.5}},
            {"version": "v2026.09.23", "date": "2026-09-23",
             "weights": {"elasticnet": 0.8, "lightgbm": 0.2,
                         "xgboost": 0.0},
             "auc": 0.7345, "brier": 0.2073, "logloss": 0.6013,
             "ece": 0.0288, "brier_calibrated": 0.2073,
             "logloss_calibrated": 0.6013, "ece_calibrated": 0.0276,
             "calibration": {"a": 1.110, "b": 0.029, "n": 2130,
                             "method": "favored_platt_floor",
                             "floor": 0.5}},
        ],
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
    _stage(PREV_MONITOR_PATH,
           json.dumps(_prev_monitor_record(), indent=2).encode("utf-8"))
    MLB_DD.mkdir(parents=True, exist_ok=True)
    _stage(MLB_MONITOR_PATH,
           json.dumps(_mlb_monitor_record(), indent=2).encode("utf-8"))
    NBA_DD.mkdir(parents=True, exist_ok=True)
    _stage(NBA_MONITOR_PATH,
           json.dumps(_nba_monitor_record(), indent=2).encode("utf-8"))


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
        if "walk-forward history (pooled OOF + sealed)" not in text:
            problems.append("upset note missing its walk-forward pool label")

        # (3) feature drift (PSI) matrix with WARN status pill
        if "Feature Drift Analysis (PSI Scores)" not in text:
            problems.append("missing feature-drift matrix section")
        if "ewm_net_pts_diff" not in text:
            problems.append("drift matrix missing a drift row")
        if "WARN" not in text:
            problems.append("drift matrix missing WARN status pill")

        # (3d) the DECISION columns: statuses are assigned on the
        #      noise-adjusted PSI under a 2-SE location gate — the table must
        #      show those numbers, or raw PSI reads self-contradictory
        #      (0.402 OK beside 0.481 ALERT on the 2026-09-29 artifact).
        if "PSI ADJ." not in text:
            problems.append("drift matrix missing the PSI ADJ. column header")
        if "SHIFT SE" not in text:
            problems.append("drift matrix missing the SHIFT SE column header")
        if "noise floor" not in text:
            problems.append("drift caption missing the noise-floor explanation")

        # (3c) drift ALERT card breaks out the counts by status — an
        #      ALERT+WARN table must never read "2 Alert" (the union
        #      mislabel the 09-02 artifact showed as "9 Alert").
        if "1 Alert · 1 Warning" not in text:
            problems.append("drift card missing the split Alert/Warning counts")
        if "2 Alert" in text:
            problems.append("drift card labels the ALERT+WARN union 'Alert'")

        # (3a) MLB-identical MODEL WEIGHT column: header + formatted weight +
        #      per-feature description label (sport-dispatched describe_feature)
        if "MODEL WEIGHT" not in text:
            problems.append("drift matrix missing the MODEL WEIGHT column header")
        if "45.00%" not in text:
            problems.append("drift matrix missing a formatted model-weight cell")
        if "Exponentially weighted" not in text and "point-in-time rating gap" not in text:
            problems.append("drift matrix missing the NFL feature description label")

        # (3b) retrain cards must never contradict the dates: the days-ago /
        #      tonight subtext is derived from the artifact dates vs the
        #      page's selected date (the fixture's 09-09 last_retrain renders
        #      as a stale 'days ago' only because 'today' in this environment
        #      is later than the fixture date — the real pipeline writes the
        #      run date, which is always same-day vs its own artifact). The
        #      invariant asserted here: the subtext matches the DATES
        #      (last_retrained == selected date -> 'today'; next <= +1 day ->
        #      'tonight'), never a hardcoded lie.
        if "tonight" not in text:
            problems.append("NEXT RETRAIN subtext missing the 'tonight' suffix")

        # (4) feature coverage panel (picks up the STARVED row)
        if "Feature Coverage (non-null / measured)" not in text:
            problems.append("missing feature-coverage section")
        if "STARVED" not in text:
            problems.append("coverage panel missing STARVED status")
        if "structural: indoor/closed" not in text:
            problems.append("coverage panel missing the STRUCTURAL reason label")
        if "all windows healthy" in text:
            problems.append("coverage reported healthy despite a starved row")

        # (4b) structural sparsity label: exp2 offspeed categories measure
        #      ~56-80% in BOTH windows by construction (PA floor), so the
        #      panel must say WHY a row is permanently amber.
        if "structural: sparse offspeed PA" not in text:
            problems.append("coverage panel missing the structural sparsity label")

        # (5) model ensemble table
        if "Model Ensemble" not in text:
            problems.append("missing model-ensemble section")
        if "XGBOOST" not in text.upper() and "xgboost" not in text.lower():
            problems.append("ensemble table missing the xgboost member row")

        # (5b) TOTAL row carries the deployed blend's OWN pooled scores —
        #      the 2026-10-05 binary-parity remediation: the blend row is
        #      comparable to the member rows at a glance.
        if "TOTAL (blended ensemble)" not in text:
            problems.append("ensemble table missing the TOTAL blended row")
        for key, val in (("AUC", "0.6995"), ("Brier", "0.2035"),
                         ("log-loss", "0.6321")):
            if val not in text:
                problems.append(f"TOTAL row missing the blend's pooled {key} ({val})")
        if "the exact blend the production binary serves" not in text:
            problems.append("TOTAL row missing the deployed-blend provenance note")

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

        # (7b) MLB-identical ROLLING history: each published NFL artifact
        #      carries one run's row, so the page must fold the dated family
        #      (the 2026-10-07 defect: the table showed 1 record beside
        #      MLB's 20). Both staged rows must render, oldest first, under
        #      MLB's column structure.
        if len(at.table) != 1:
            problems.append(
                f"expected ONE version-history st.table, got {len(at.table)}")
        else:
            vh = at.table[0].value
            mlb_cols = ["VERSION", "DATE", "WEIGHTS", "AUC", "LOGLOSS",
                        "CAL. ECE", "CAL. MAP"]
            if list(vh.columns) != mlb_cols:
                problems.append(
                    f"version-history columns {list(vh.columns)} != MLB's "
                    f"{mlb_cols}")
            vers = list(vh["VERSION"]) if "VERSION" in vh.columns else []
            if vers != ["v2026.08.30", "v2026.08.31"]:
                problems.append(
                    f"version history not the folded family (oldest-first): "
                    f"{vers}")

        # (7c) structure parity with MLB: the report tables carry MLB's
        #      rendered column sets exactly (drift incl. the decision
        #      columns + MODEL WEIGHT, coverage incl. % MEASURED / %
        #      NON-NULL, ensemble incl. the blend metrics columns).
        _expected_headers = [
            ["FEATURE", "CURRENT MEAN", "BASELINE MEAN", "PSI", "PSI ADJ.",
             "SHIFT SE", "MODEL WEIGHT", "STATUS"],
            ["FEATURE", "WINDOW", "GAMES", "% MEASURED", "% NON-NULL",
             "STATUS"],
            ["MODEL", "DESCRIPTION", "WEIGHT", "AUC", "BRIER", "LOG LOSS"],
        ]
        _headers = []
        for el in at.markdown:
            m = re.search(r"<thead>(.*?)</thead>", str(el.value), re.S)
            if m:
                _headers.append(re.findall(r"<th>(.*?)</th>", m.group(1)))
        if _headers != _expected_headers:
            problems.append(
                f"report table headers { _headers } != MLB's "
                f"{_expected_headers}")

        if problems:
            print("MONITOR SMOKE TEST — FAIL (sport=nfl)")
            for p in problems:
                print("  -", p)
            return 1

        print("MONITOR SMOKE TEST — PASS (sport=nfl)")
        print(f"  - no exceptions; {len(vcl)} Altair chart(s) rendered")
        print("  - health boxes + upset callout + drift matrix (WARN) + coverage"
              " (STARVED) + ensemble + rolling Brier + version history")

        # sport=mlb must still run the SAME shared path, no exception. Pin the
        # date to the staged MLB fixture (the committed 20260927 artifact would
        # otherwise win the newest-date resolution) and pin describe_feature's
        # served-metadata precedence: the backend-authored per-side summaries
        # beat the static dict and the legacy diff-text/bare-name fallbacks.
        mlb = AppTest.from_file(str(FRONTEND_DIR / "model_monitor.py"),
                                default_timeout=60)
        mlb.session_state["sport"] = "mlb"
        mlb.session_state["selected_date"] = ARTIFACT_DATE
        mlb.run()
        if mlb.exception:
            prob = "\n  ".join(str(e.value) for e in mlb.exception)
            print("MONITOR SMOKE TEST — FAIL (sport=mlb)")
            print("  - mlb path raised:\n    " + prob)
            return 1
        mlb_text = _all_text(mlb)
        _served_ok = all(
            needle in mlb_text for needle in (
                "Starting-pitcher earned-run average — home team",
                "Bullpen walks+hits per inning — home team",
                "Home team Elo rating (level)",
            ))
        _dict_fallback_ok = (
            "Home Elo − away Elo (skill-gap anchor, updated each game)"
            in mlb_text)
        _degenerate_ok = (
            "Always 1 — anchors the ~53% MLB home-field win advantage"
            in mlb_text and "is_home\nis_home" not in mlb_text)
        if not (_served_ok and _dict_fallback_ok and _degenerate_ok):
            print("MONITOR SMOKE TEST — FAIL (sport=mlb)")
            if not _served_ok:
                print("  - served-metadata labels missing from the drift table")
            if not _dict_fallback_ok:
                print("  - static-dict fallback for unserved rows broken")
            if not _degenerate_ok:
                print("  - degenerate served summary (== bare name) not ignored")
            return 1
        # (8) per-sport member-card derivation: the MLB leg runs AFTER the
        #     NFL leg in the SAME process, so a sys.modules-borne `config`
        #     from the first sport would mislabel this card (the exact
        #     2026-09-29 production defect: the NBA page rendered MLB's
        #     "max depth 6, lr 0.0332, 50 rounds"). Both sports' xgboost sit
        #     at max depth 2, so the LEARNING RATE is the discriminator:
        #     MLB derives lr 0.055; NFL's lr 0.005026... must not leak in.
        if "lr 0.055" not in mlb_text:
            print("MONITOR SMOKE TEST — FAIL (sport=mlb)")
            print("  - MLB member card missing its own config numbers "
                  "(lr 0.055)")
            return 1
        if "lr 0.005026" in mlb_text:
            print("MONITOR SMOKE TEST — FAIL (sport=mlb)")
            print("  - cross-sport leak: the NFL card's learning rate "
                  "reached the MLB member card")
            return 1
        print("  - sport=mlb path clean (no exception)")
        print("  - member cards derive per sport; served-metadata labels "
              "win; dict fallback + degenerate summary guard intact")

        # sport=nba — the 2026-10-07 structure-parity leg: the NBA report
        # must be structurally identical to MLB's (the feature drift report
        # and feature coverage in MLB's column structure, served-metadata
        # labels + tooltips on every row, and a POPULATED Model Version
        # History instead of "No version history yet").
        nba = AppTest.from_file(str(FRONTEND_DIR / "model_monitor.py"),
                                default_timeout=60)
        nba.session_state["sport"] = "nba"
        nba.session_state["selected_date"] = ARTIFACT_DATE
        nba.run()
        nba_problems: list[str] = []
        if nba.exception:
            nba_problems.append(
                "nba page raised:\n    "
                + "\n    ".join(str(e.value) for e in nba.exception))
        nba_text = _all_text(nba)

        # (N1) structure parity: the three report tables carry MLB's
        #      rendered column sets exactly.
        _nba_headers = []
        for el in nba.markdown:
            m = re.search(r"<thead>(.*?)</thead>", str(el.value), re.S)
            if m:
                _nba_headers.append(re.findall(r"<th>(.*?)</th>", m.group(1)))
        if _nba_headers != _expected_headers:
            nba_problems.append(
                f"report table headers {_nba_headers} != MLB's "
                f"{_expected_headers}")

        # (N2) drift report: served-metadata labels + tooltips — the per-side
        #      row carries its OWN wording (the 2026-10-06 defect rendered
        #      elo_home as "Home Elo − away Elo ... — home team"), unserved
        #      rows fall back to the static dict, and no row ships the
        #      "(no detailed metadata)" fallback tooltip.
        if "Entering Elo rating — home team" not in nba_text:
            nba_problems.append("drift table missing the served elo_home label")
        if "Home Elo − away Elo (pre-game rating gap) — home team" in nba_text:
            nba_problems.append("elo_home rendered the diff-twin side mislabel")
        if "Home Elo − away Elo (pre-game rating gap)" not in nba_text:
            nba_problems.append("drift table missing the dict fallback label")
        # The one unserved row (elo_diff, pinning the dict-fallback label)
        # legitimately carries the no-metadata tooltip; every SERVED row must
        # carry the backend-authored one, so the fallback appears exactly
        # once — never once per row (the 2026-10-06 defect: 71 of 71).
        if nba_text.count("no detailed metadata") != 1:
            nba_problems.append(
                "drift tooltips fell back to '(no detailed metadata)' on "
                f"{nba_text.count('no detailed metadata')} rows — served "
                "features_metadata must document its rows")
        if nba_text.count("What:") < 2:
            nba_problems.append("drift table missing the served tooltips")
        if "Constant 1 — anchors the home-court edge" not in nba_text:
            nba_problems.append(
                "degenerate served summary (== bare name) not rejected")

        # (N3) drift card splits Alert/Warning like MLB's.
        if "1 Alert · 1 Warning" not in nba_text:
            nba_problems.append("drift card missing the split Alert/Warning counts")

        # (N4) coverage panel: MLB's columns + statuses render.
        if "Feature Coverage (non-null / measured)" not in nba_text:
            nba_problems.append("missing feature-coverage section")
        if "STARVED" not in nba_text:
            nba_problems.append("coverage panel missing STARVED status")
        if "% MEASURED" not in nba_text or "% NON-NULL" not in nba_text:
            nba_problems.append("coverage panel missing MLB's percentage columns")

        # (N5) ensemble TOTAL row carries the blend's pooled scores.
        if "TOTAL (blended ensemble)" not in nba_text:
            nba_problems.append("ensemble table missing the TOTAL blended row")
        for val in ("0.7345", "0.6013"):
            if val not in nba_text:
                nba_problems.append(
                    f"TOTAL row missing the blend's pooled metric ({val})")

        # (N6) rolling Brier chart + calibrated-series caption.
        if len(nba.get("vega_lite_chart")) == 0:
            nba_problems.append("rolling-Brier timeline did NOT render a chart")
        if "calibrated probabilities" not in nba_text:
            nba_problems.append(
                "rolling-Brier caption missing the calibrated-series label")

        # (N7) Model Version History — the "missing" defect: populated rows
        #      under MLB's exact columns, oldest first, with the deployed
        #      map rendered (never the empty-state info box).
        if "No version history yet" in nba_text:
            nba_problems.append("version history empty instead of populated rows")
        if len(nba.table) != 1:
            nba_problems.append(
                f"expected ONE version-history st.table, got {len(nba.table)}")
        else:
            vh = nba.table[0].value
            mlb_cols = ["VERSION", "DATE", "WEIGHTS", "AUC", "LOGLOSS",
                        "CAL. ECE", "CAL. MAP"]
            if list(vh.columns) != mlb_cols:
                nba_problems.append(
                    f"version-history columns {list(vh.columns)} != MLB's "
                    f"{mlb_cols}")
            vers = list(vh["VERSION"]) if "VERSION" in vh.columns else []
            if vers != ["v2026.09.22", "v2026.09.23"]:
                nba_problems.append(
                    f"version history not oldest-first retrain rows: {vers}")
            cal_map = list(vh["CAL. MAP"]) if "CAL. MAP" in vh.columns else []
            if not cal_map or "a=1.103, b=0.028" not in str(cal_map[0]):
                nba_problems.append(
                    f"CAL. MAP cell missing the deployed Platt map: {cal_map}")

        if nba_problems:
            print("MONITOR SMOKE TEST — FAIL (sport=nba)")
            for p in nba_problems:
                print("  -", p)
            return 1
        print("  - sport=nba path clean (no exception)")
        print("  - MLB-identical report structure: drift labels + tooltips, "
              "coverage columns, populated version history")
        return 0
    finally:
        _remove_artifacts()


if __name__ == "__main__":
    sys.exit(run())