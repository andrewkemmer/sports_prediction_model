"""Smoke test — NFL Calibration page (step 3).

Runs the real ``model_calibration.py`` page under ``sport=nfl`` through
Streamlit's official ``AppTest`` harness against the CURRENT committed
``nfl-backend/data_delivery/nfl_moneyline_v1_*.json`` artifacts. Confirms no
exceptions and that the aggregate sections render (header, ADOPT verdict,
KPIs, baselines, members, recalibration note) while the three per-game-OOF
conditional sections degrade to honest info lines (the v1 record carries no
per-game OOF history yet).

Run from the frontend/ directory:
    python -m test_calibration_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent


def _all_text(at: AppTest) -> str:
    """Concatenate every text-bearing element for cheap substring checks."""
    chunks = []
    for attr in ("markdown", "info", "warning", "caption", "success",
                 "error", "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            try:
                chunks.append(str(el.value))
            except Exception:
                pass
    return "\n".join(chunks)


def run() -> int:
    # The page reads artifacts from the local repo (no GitHub configured),
    # so the newest nfl_moneyline_v1_*.json under nfl-backend/data_delivery
    # is what renders. Run the ACTUAL page file (not a mock).
    at = AppTest.from_file(str(FRONTEND_DIR / "model_calibration.py"),
                           default_timeout=30)
    at.session_state["sport"] = "nfl"
    at.run()

    problems: list[str] = []

    # 1) No exceptions in the entire run.
    if at.exception:
        problems.append("PAGE RAISED EXCEPTIONS:\n  "
                        + "\n  ".join(str(e.value) for e in at.exception))

    text = _all_text(at)

    # 2) Aggregate sections must render.
    required = {
        "header": "Model Calibration Dashboard",
        "artif-date": "NFL",
        "adopt-banner": "ADOPT",
        "kpi-logloss": "LOG-LOSS",
        "kpi-auc": "AUC-ROC",
        "kpi-ece": "CAL. ERROR (ECE)",
        "baselines": "SEALED 2025",
        "members": "Ensemble Members",
        "recalib": "Post-Hoc Recalibration",
        "platt": "Platt",
    }
    for key, needle in required.items():
        if needle not in text:
            problems.append(f"missing aggregate marker [{key}] = {needle!r}")

    # 3) The conditional (per-game OOF) sections must DEGRADE to info lines —
    #    the v1 record is a schedule-only slate, so no curve/buckets/history.
    degrade = {
        "curve": "Per-1% favored-team calibration curve ships when the backend",
        "reliability": "Reliability diagram ships when the record emits binned",
        "history": "No per-game prediction history yet",
    }
    for key, needle in degrade.items():
        if needle not in text:
            problems.append(f"conditional section [{key}] did NOT degrade to an info line")

    # 4) The per-1% chart must NOT be rendered (no per-game OOF data).
    n_chart = len(at.get("altair_chart")) if hasattr(at, "get") else 0
    if n_chart:
        problems.append(f"expected NO altair chart under current v1 record, found {n_chart}")

    if problems:
        print("CALIBRATION SMOKE TEST — FAIL")
        for p in problems:
            print("  -", p)
        return 1

    print("CALIBRATION SMOKE TEST — PASS")
    print("  - no exceptions")
    print("  - aggregate sections rendered (header, ADOPT, KPIs, baselines, members, recalib note)")
    print("  - per-1% curve / reliability / history degraded to info lines (no per-game OOF in v1 record)")
    return 0


if __name__ == "__main__":
    sys.exit(run())