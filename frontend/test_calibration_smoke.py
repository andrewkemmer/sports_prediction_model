"""Smoke test — Calibration page, SPORT-DISPATCHED (steps 3 + Part B).

The NFL Calibration page now runs the SAME code as MLB (no sport-special
path). This test:

1. Writes REPRESENTATIVE ``nfl_calibration_*.json`` + ``nfl_predictions_\
   history_*.csv`` artifacts (matching the exact shape ``nfl_moneyline.py``
   Part-A emits) into the real ``nfl-backend/data_delivery`` dir, so the
   page renders with real data (they are removed after the run).
2. Runs the ACTUAL ``model_calibration.py`` under ``sport=nfl`` and asserts
   the MLB-identical seven sections: header pill, today's-record summary
   card, the four KPI cards (AUC/Brier/Log-Loss/Cal. Error raw→calibrated),
   the Platt recalibration banner, the per-1% calibration CURVE as a real
   Altair chart, the reliability table WITH rows + a TOTAL row, and the
   populated prediction-history table — with no exceptions.
3. Runs the same page under ``sport=mlb`` and asserts it also runs clean
   (the shared path; locally it halts/warns on missing MLB artifacts rather
   than crashing).

Run from the frontend/ directory:
    python -m test_calibration_smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent if FRONTEND_DIR.name == "frontend" else FRONTEND_DIR
NFL_DD = REPO_ROOT / "nfl-backend" / "data_delivery"

ARTIFACT_DATE = "20260831"
CALIBRATION_NAME = f"nfl_calibration_{ARTIFACT_DATE}.json"
HISTORY_NAME = f"nfl_predictions_history_{ARTIFACT_DATE}.csv"
CALIBRATION_PATH = NFL_DD / CALIBRATION_NAME
HISTORY_PATH = NFL_DD / HISTORY_NAME

WRITTEN: list[Path] = []
# Path -> original bytes of a PRE-EXISTING artifact this test overwrites with a
# fixture (e.g. a committed nfl_calibration_*.json). They are restored on
# cleanup — never deleted — so running the smoke test can't remove real
# committed artifacts from the tree.
_BACKUPS: dict[Path, bytes] = {}


# ---------------------------------------------------------------------------
# Representative artifact construction (matches the emitted Part-A schema)
# ---------------------------------------------------------------------------
def _calibration_record() -> dict:
    """A realistic nfl_calibration_*.json mirroring build_calibration.

    Buckets run the FAVORED view only (>= 50%), matching the backend
    reliability_buckets fix — the raw calibration_buckets / calibrated set
    never carry a sub-50% bucket."""
    seq = [0.52, 0.60, 0.68, 0.76, 0.84, 0.92]   # favored-only: 50%..100%
    counts = [130, 210, 260, 230, 170, 107]
    buckets, cal_buckets = [], []
    for i, mp in enumerate(seq):
        ma = min(0.99, mp + 0.02)
        cal_mp = min(0.99, mp + 0.035)
        buckets.append({"bucket": f"{int(mp * 100)}%-{int(mp * 100) + 8}%",
                        "mean_predicted": round(mp, 3),
                        "mean_actual": round(ma, 3),
                        "count": counts[i],
                        "gap": round(mp - ma, 3)})
        cal_buckets.append({"bucket": f"{int(mp * 100)}%-{int(mp * 100) + 8}%",
                            "mean_predicted": round(cal_mp, 3),
                            "mean_actual": round(ma, 3),
                            "count": counts[i],
                            "gap": round(cal_mp - ma, 3)})
    return {
        "date": ARTIFACT_DATE,
        "n_games": int(sum(counts)),
        "trained_at": "2026-08-31T01:00:00.000000Z",
        "metrics": {"auc": 0.6911, "brier": 0.2113, "logloss": 0.6329,
                    "ece": 0.0349, "brier_calibrated": 0.2040,
                    "logloss_calibrated": 0.6333, "ece_calibrated": 0.0290},
        "calibration_buckets": buckets,
        "calibration": {
            "method": "platt",
            "params": {"a": 2.5, "b": 0.1, "n": int(sum(counts))},
            "metrics_raw": {"brier": 0.2113, "logloss": 0.6329, "ece": 0.0349},
            "metrics_calibrated": {"brier": 0.2040, "logloss": 0.6333,
                                   "ece": 0.0290},
            "calibration_buckets_calibrated": cal_buckets,
        },
        "daily": [],
    }


def _history_frame(n: int = 320) -> pd.DataFrame:
    """A per-game decided prediction history (same column set the backend
    emits) spanning favored probs 0.51..0.95 so the 1% curve has points, and
    including some upsets (underdog winners) for the summary card."""
    rng = np.random.default_rng(7)
    ps = rng.uniform(0.51, 0.95, n)
    home = [f"H{i % 32:02d}" for i in range(n)]
    away = [f"A{(i + 9) % 32:02d}" for i in range(n)]
    correct = rng.random(n) < ps            # higher prob -> more correct
    pick = home                             # favored side for moneyline
    winner = [home[i] if correct[i] else away[i] for i in range(n)]
    rows = []
    for i in range(n):
        hs = 30 if winner[i] == home[i] else 13
        as_ = 13 if winner[i] == home[i] else 30
        rows.append({
            "game_date": f"2026-09-{i % 20 + 1:02d}",
            "home_team": home[i], "away_team": away[i],
            "home_win_prob_model": round(float(ps[i]), 4),
            "home_win_prob_model_calibrated": round(
                float(1.0 / (1.0 + np.exp(-(2.5 * np.log(ps[i] / (1 - ps[i])) + 0.1)))), 4),
            "away_win_prob_model": round(1.0 - float(ps[i]), 4),
            "correct": bool(correct[i]),
            "model_pick": pick[i],
            "home_score": hs, "away_score": as_,
            "actual_winner": winner[i],
            "game_status": "Final",
            "game_id": f"2026_W1_G{i:03d}",
            "season": 2026, "week": 1,
        })
    return pd.DataFrame(rows)


def _stage(path: Path, data: bytes) -> None:
    """Write a fixture over ``path``, preserving any pre-existing (committed)
    artifact's bytes so cleanup can restore it rather than delete it."""
    if path.exists():
        _BACKUPS[path] = path.read_bytes()
    path.write_bytes(data)
    WRITTEN.append(path)


def _write_artifacts() -> None:
    NFL_DD.mkdir(parents=True, exist_ok=True)
    cal = json.dumps(_calibration_record(), indent=2).encode("utf-8")
    hist = _history_frame().to_csv(index=False).encode("utf-8")
    _stage(CALIBRATION_PATH, cal)
    _stage(HISTORY_PATH, hist)


def _remove_artifacts() -> None:
    for p in WRITTEN:
        try:
            if p in _BACKUPS:
                # A committed artifact existed here — restore it byte-for-byte;
                # never delete real data.
                p.write_bytes(_BACKUPS.pop(p))
            else:
                # This was a fixture this test created fresh -> safe to remove.
                p.unlink()
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
        at = AppTest.from_file(str(FRONTEND_DIR / "model_calibration.py"),
                               default_timeout=60)
        at.session_state["sport"] = "nfl"
        at.run()

        if at.exception:
            problems.append("NFL PAGE RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in at.exception))

        text = _all_text(at)
        vcl = at.get("vega_lite_chart")

        # (1) header pill, record summary card
        for key, needle in [("header", "Model Calibration Dashboard"),
                            ("record", "Today's Record:"),
                            ("rec-completed", " completed games")]:
            if needle not in text:
                problems.append(f"missing [{key}] = {needle!r}")

        # (1b) NFL upset strip is capped to the most-surprising few (not all 50
        #     upsets from the lifetime pool) with a collapsed remainder tail.
        if " more upsets" not in text:  # tail renders as '· +40 more upsets'
            problems.append("NFL upset strip not collapsed to top-N + remainder")
        if text.count(" upset ") > 20:  # capped ~10 pills — not the ~50 full flood
            problems.append("NFL upset strip still floods (too many upset pills)")

        # (2) four KPI cards
        for key, needle in {"auc": "AUC-ROC", "brier": "BRIER SCORE",
                            "logloss": "LOG-LOSS", "ece": "CAL. ERROR"}.items():
            if needle not in text:
                problems.append(f"missing KPI [{key}] = {needle!r}")

        # (3) post-hoc Platt recalibration banner
        if "Post-Hoc Recalibration" not in text:
            problems.append("missing Platt recalibration banner")

        # (4) the per-1% curve must be a REAL Altair chart (not an info line)
        if len(vcl) == 0:
            problems.append("curve did NOT render an Altair chart (vega_lite_chart=0)")
        if "Calibration Curve" not in text:
            problems.append("missing calibration-curve section")
        if "Per-1% favored-team calibration curve ships when" in text:
            problems.append("curve section degraded (info line) instead of rendering")

        # (5) reliability table with rows + a TOTAL row
        if "Reliability Diagram" not in text:
            problems.append("missing reliability-diagram section")
        if "TOTAL" not in text:
            problems.append("reliability table missing TOTAL row")
        if "BUCKET" not in text:
            problems.append("reliability table missing BUCKET header/rows")

        # (5b) reliability table runs favored-only (>= 50%): the corrected
        #     backend never emits a sub-50% bucket, and the page must not show one.
        if "42%-50%" in text:
            problems.append("reliability table still shows a sub-50% bucket")
        if "52%-60%" not in text:
            problems.append("reliability table missing a >=50% favored bucket")

        # (6) prediction-history table populated (with real rows, not the empty info)
        if "Prediction History" not in text:
            problems.append("missing prediction-history section")
        if "No per-game prediction history" in text:
            problems.append("history table empty/info line instead of populated rows")

        if problems:
            print("CALIBRATION SMOKE TEST — FAIL (sport=nfl)")
            for p in problems:
                print("  -", p)
            return 1

        print("CALIBRATION SMOKE TEST — PASS (sport=nfl)")
        n_curves = len(vcl)
        print(f"  - no exceptions; {n_curves} Altair curve chart(s) rendered")
        print("  - record summary + 4 KPIs + Platt banner + reliability table"
              " (w/ TOTAL) + populated history table")

        # sport=mlb must still run the SAME shared path, no exception.
        mlb = AppTest.from_file(str(FRONTEND_DIR / "model_calibration.py"),
                                default_timeout=60)
        mlb.session_state["sport"] = "mlb"
        mlb.run()
        if mlb.exception:
            prob = "\n  ".join(str(e.value) for e in mlb.exception)
            print("CALIBRATION SMOKE TEST — FAIL (sport=mlb)")
            print("  - mlb path raised:\n    " + prob)
            return 1
        print("  - sport=mlb path clean (no exception)")
        return 0
    finally:
        _remove_artifacts()


if __name__ == "__main__":
    sys.exit(run())