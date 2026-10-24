"""Off-day honesty — 2026-10-25 remediation.

2026-10-25 is a genuine league off-day: the pipeline shipped a header-only
``todays_games_20261025.csv`` ("never recycling decided games as today's
slate", verified 0 games via the schedule). The page used to treat that
existing-but-empty snapshot exactly like a CDN-lag fetch failure, walking
the recovery walk to the 10-23 pre-game slate and presenting an
already-decided game as tonight's pick under the Oct 25 header.

Contract under test (local committed artifacts, no network):

  A. ``utils.board_snapshot_exists_but_empty`` is True only for a resolved
     snapshot with zero data rows: 20261025 True; 20261023 (1 row) False;
     a missing date False; a polluted file with raw rows False.
  B. Page-level: selecting 20261025 renders the honest off-day state —
     "No games scheduled", NO "Recovery view" banner, and none of the
     10-23 slate's card content (Misiorowski / American Family Field).

Run from the frontend/ directory:
    python -m test_offday_no_recovery
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

if "pytest" not in sys.modules and (
        (getattr(sys.stdout, "encoding", "") or "").lower()
        not in ("utf-8", "utf8")):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

FRONTEND_DIR = Path(__file__).resolve().parent


def _harvest(at: AppTest) -> str:
    parts: list[str] = []
    for seq in (at.markdown, at.info, at.warning, at.caption,
                at.error, at.success, at.button, at.subheader, at.header):
        for el in seq:
            v = getattr(el, "value", None)
            if not isinstance(v, str):
                v = getattr(el, "label", "") or ""
            parts.append(str(v))
    return "\n".join(parts)


def _pin(at: AppTest, date_compact: str) -> None:
    at.session_state["selected_date"] = date_compact
    at.session_state["_nav_sport"] = "mlb"
    at.session_state["sport"] = "mlb"


def main() -> int:
    import utils

    st.cache_data.clear()

    # ---- A: helper unit proof ----
    assert utils.board_snapshot_exists_but_empty("20261025") is True, \
        "20261025 ships a header-only board — must read as an off-day"
    assert utils.board_snapshot_exists_but_empty("20261023") is False, \
        "20261023 carries the 1-game slate — must NOT read as an off-day"
    assert utils.board_snapshot_exists_but_empty("19990101") is False, \
        "a missing snapshot is a fetch miss, never an off-day"
    print("A PASS  board_snapshot_exists_but_empty: off-day True, "
          "slate/missing False")

    # ---- B: page-level — off-day dead-ends honestly, no recovery ----
    at = AppTest.from_file("todays_games.py", default_timeout=120)
    _pin(at, "20261025")
    at.run()
    assert not at.exception, f"page exception: {at.exception}"
    text = _harvest(at)
    assert "No games scheduled for Sunday, October 25, 2026" in text, \
        f"expected the honest off-day state, got: {text[:600]}"
    assert "Recovery view" not in text, \
        "off-day must NOT walk the recovery walk to another date's board"
    assert "Misiorowski" not in text, \
        "10-23 slate content leaked into the 10-25 off-day view"
    assert "American Family Field" not in text, \
        "10-23 venue leaked into the 10-25 off-day view"
    print("B PASS  20261025 renders the honest off-day state: no recovery "
          "banner, no 10-23 slate content")

    print("\nOFF-DAY NO-RECOVERY PROOF - PASS (2 scenarios)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
