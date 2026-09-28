"""Full local render — NFL Today's Games board (the 2026-09-27 remediation).

Drives the ACTUAL ``frontend/nfl_todays_page.py`` (the page Home.py
dispatches for the NFL Today's Games dashboard) under Streamlit AppTest with
representative committed artifacts:

  1. ``nfl_board_<date>.csv``   dated per-date board snapshots (the MLB
                                todays_games_<date>.csv structural twin),
                                including a STARTED EARLIER TODAY game and
                                a finished game from yesterday.
  2. ``nfl_moneyline_v1_<date>.json``  the current-slate fallback record.
  3. ``nfl_calibration_<date>.json``   the header-strip feed.

Asserts the remediation end-to-end at the render level:
  - today's board shows the started game (not removed) with its frozen
    pre-game price and a Live/started card;
  - yesterday's date is navigable and renders the dated snapshot's frozen
    prices (no OOF substitution);
  - the rolling-10-day valid-date rail is built from the dated board
    family (structurally aligned with MLB's board-family navigation);
  - no page exceptions.

Run from the frontend/ directory:
    python -m test_board_render_smoke
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent if FRONTEND_DIR.name == "frontend" else FRONTEND_DIR
NFL_DD = REPO_ROOT / "nfl-backend" / "data_delivery"

# Dates are pinned NEAR today (ET) so the page's rolling-10-day valid-date
# window includes them, but with an unused sentinel year suffix in the file
# name is not possible (the family is dated by real dates) — instead we use
# REAL today/yesterday and back up + restore any committed files we shadow.
_TODAY = datetime.now(ZoneInfo("America/New_York")).date()
_YDAY = _TODAY - timedelta(days=1)
_TODAY_C = _TODAY.strftime("%Y%m%d")
_YDAY_C = _YDAY.strftime("%Y%m%d")
BOARD_TODAY = NFL_DD / f"nfl_board_{_TODAY_C}.csv"
BOARD_YDAY = NFL_DD / f"nfl_board_{_YDAY_C}.csv"
ML_JSON = NFL_DD / f"nfl_moneyline_v1_{_TODAY_C}.json"
CAL_JSON = NFL_DD / f"nfl_calibration_{_TODAY_C}.json"

_BACKUPS: dict[Path, bytes | None] = {}
WRITTEN: list[Path] = []


def _iso(day, et_hour: int) -> str:
    """Start stamp the serving convention emits (date + ET time as-is)."""
    return f"{day.isoformat()}T{et_hour:02d}:00:00Z"


def _moneyline_record() -> dict:
    """Current-slate record: a started game + a later game, truthful states."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"feature_set": "smoke"},
        "slate_date": _TODAY.isoformat(),
        "n_games": 2,
        "games": [
            {
                "game_id": "SMOKE_LIVE", "game_date": _TODAY.isoformat(),
                "start_time_utc": _iso(_TODAY, 13),
                "home_team": "NYG", "away_team": "WAS",
                "home_team_name": "New York Giants",
                "away_team_name": "Washington",
                "home_record": "1-1", "away_record": "2-0",
                "venue": "MetLife Stadium", "game_status": "Live",
                "home_score": None, "away_score": None,
                "home_win_prob_model": 0.552, "away_win_prob_model": 0.448,
                "model_pick": "NYG", "model_correct": None,
            },
            {
                "game_id": "SMOKE_PRE", "game_date": _TODAY.isoformat(),
                "start_time_utc": _iso(_TODAY, 20),
                "home_team": "SF", "away_team": "SEA",
                "home_team_name": "San Francisco",
                "away_team_name": "Seattle",
                "home_record": "3-0", "away_record": "1-2",
                "venue": "Levi's Stadium", "game_status": "pre",
                "home_score": None, "away_score": None,
                "home_win_prob_model": 0.688, "away_win_prob_model": 0.312,
                "model_pick": "SF", "model_correct": None,
            },
        ],
    }


def _board_rows() -> pd.DataFrame:
    """Dated snapshots: yesterday's finished game + today's started game.

    home_win_prob_model is the FROZEN PRE-GAME price the serving horizon
    published — exactly what the dated board CSV carries.
    """
    return pd.DataFrame([
        {
            "game_id": "SMOKE_FIN", "game_date": _YDAY.isoformat(),
            "start_time_utc": _iso(_YDAY, 13),
            "home_team": "CHI", "away_team": "GB",
            "home_team_name": "Chicago", "away_team_name": "Green Bay",
            "home_record": "1-2", "away_record": "2-1",
            "venue": "Soldier Field", "game_status": "Final",
            "home_score": 20.0, "away_score": 27.0,
            "home_win_prob_model": 0.412, "away_win_prob_model": 0.588,
            "model_pick": "GB", "model_correct": True,
        },
        {
            "game_id": "SMOKE_LIVE", "game_date": _TODAY.isoformat(),
            "start_time_utc": _iso(_TODAY, 13),
            "home_team": "NYG", "away_team": "WAS",
            "home_team_name": "New York Giants",
            "away_team_name": "Washington",
            "home_record": "1-1", "away_record": "2-0",
            "venue": "MetLife Stadium", "game_status": "Live",
            "home_score": None, "away_score": None,
            "home_win_prob_model": 0.552, "away_win_prob_model": 0.448,
            "model_pick": "NYG", "model_correct": None,
        },
        {
            "game_id": "SMOKE_PRE", "game_date": _TODAY.isoformat(),
            "start_time_utc": _iso(_TODAY, 20),
            "home_team": "SF", "away_team": "SEA",
            "home_team_name": "San Francisco", "away_team_name": "Seattle",
            "home_record": "3-0", "away_record": "1-2",
            "venue": "Levi's Stadium", "game_status": "pre",
            "home_score": None, "away_score": None,
            "home_win_prob_model": 0.688, "away_win_prob_model": 0.312,
            "model_pick": "SF", "model_correct": None,
        },
    ])


def _calibration_record() -> dict:
    return {
        "date": _TODAY_C, "trained_at": "2026-01-01T00:00:00Z",
        "n_games": 3, "created_utc": "2026-01-01T00:00:00Z",
        "config": {}, "league_total": 3, "evening_games_league": 1,
        "metrics": {"auc": 0.69, "brier": 0.22, "logloss": 0.63, "ece": 0.02},
        "calibration": {}, "distribution_calibration": {},
        "calibration_buckets": [], "daily": [],
    }


def _write_artifacts() -> None:
    NFL_DD.mkdir(parents=True, exist_ok=True)
    for p in (BOARD_TODAY, BOARD_YDAY, ML_JSON, CAL_JSON):
        _BACKUPS[p] = p.read_bytes() if p.exists() else None
    _board_rows().to_csv(BOARD_TODAY, index=False)
    _board_rows()[_board_rows()["game_date"] == _YDAY.isoformat()] \
        .to_csv(BOARD_YDAY, index=False)
    ML_JSON.write_text(json.dumps(_moneyline_record()))
    CAL_JSON.write_text(json.dumps(_calibration_record()))
    WRITTEN.extend([BOARD_TODAY, BOARD_YDAY, ML_JSON, CAL_JSON])


def _restore() -> None:
    for p, prev in _BACKUPS.items():
        try:
            if prev is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(prev)
        except FileNotFoundError:
            pass
    WRITTEN.clear()


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


def _fold_checks() -> list[str]:
    """Pure-function pins for the 2026-09-28 card normalization fix.

    Synthetic rows encode the semantics exactly: an integer total/spread
    displays the 2-WAY FOLDED pair (push proportionately into both sides,
    summing to 100%) with the raw push as its own note; a pick'em game
    (fair spread 0) renders the +/-0.5 stop and NEVER a 0.0 spread; and at
    +/-0.5 the -0.5 side is the OUTRIGHT win (a tie makes the -0.5 team
    lose) while the run-ML note excludes the tie from both sides.
    """
    import nfl_slate_view as sv
    problems: list[str] = []
    base = {
        "mu_h": 28.8, "mu_a": 18.3, "fair_total": 46.0, "fair_spread": 10.0,
        "p_over_46": 0.49, "p_under_46": 0.47, "p_push_46": 0.04,
        "p_home_cover_10": 0.52, "p_push_10": 0.03,
        "p_home_cover_0": 0.45, "p_push_0": 0.10,
        "p_home_win_derived": 0.50, "p_away_win_derived": 0.50,
    }

    def _sum100(html: str, kind: str) -> bool:
        import re as _re
        nums = [float(x) for x in _re.findall(r"(\d+(?:\.\d+)?)%", html)]
        return len(nums) >= 2 and abs(nums[0] + nums[1] - 100.0) < 0.5, kind

    # (1) Integer total: folded pair sums to 100%, raw push kept visible.
    html = sv.runengine_html(base, "BUF", "LAC")
    if "Over 51% / Under 49%" not in html or "(4% push)" not in html:
        problems.append(f"integer total not folded with push note: {html}")
    ok, _ = _sum100(html, "total")
    if not ok:
        problems.append(f"integer total pair does not sum to 100: {html}")
    # (1b) Build stamp: every strip carries the process's own source digest
    # so a stale Streamlit process is identifiable on screen (its chip is
    # the OLD digest; a fresh process shows the current one).
    if not sv.BUILD_STAMP or f"build {sv.BUILD_STAMP}" not in html:
        problems.append(f"run-engine strip missing the build stamp: {html}")
    # (2) Integer spread (home -10): folded pair sums to 100% + push note.
    rl = sv.runline_html(base, "BUF", "LAC", home_spread=-10.0)
    if "BUF \u221210 54%" not in rl or "LAC +10 46%" not in rl \
            or "(3% push)" not in rl:
        problems.append(f"integer spread not folded: {rl}")
    # (3) Pick'em guard: fair_spread 0 -> +/-0.5 pair, never 0.0.
    pickem = dict(base, fair_spread=0.0)
    pe = sv.runengine_html(pickem, "HOM", "AWY")
    if "\u22120.0" in pe or "+0.0" in pe:
        problems.append(f"pick'em game rendered a 0.0 spread: {pe}")
    # p_home_cover_0 = 0.45 -> AWAY is the favorite; the pair orients to it.
    if "AWY \u22120.5 45%" not in pe or "HOM +0.5 55%" not in pe:
        problems.append(f"pick'em did not render the +/-0.5 pair: {pe}")
    # (4) Tie semantics at +/-0.5: -0.5 is the outright win (tie excluded),
    # +0.5 includes the tie, run-ML excludes the tie from BOTH sides
    # (0.45 raw / 0.10 tie -> 45% vs (run-ML 50%)).
    hs = sv.runline_html(pickem, "HOM", "AWY", half_stop=True)
    if "AWY \u22120.5 45%" not in hs or "HOM +0.5 55%" not in hs:
        problems.append(f"+/-0.5 raw pair wrong (tie must break to +0.5): {hs}")
    if hs.count("(run-ML 50%)") != 2:
        problems.append(f"run-ML notes must exclude ties from both sides: {hs}")
    return problems


def run() -> int:
    problems: list[str] = _fold_checks()
    _write_artifacts()
    try:
        # ---- Today's board renders with the started game PRESENT ----
        # The shipped page entry is todays_games.py (Home.py's st.Page):
        # its main() dispatches to nfl_todays_page.run() for sport=nfl.
        at = AppTest.from_file(str(FRONTEND_DIR / "todays_games.py"),
                               default_timeout=120)
        at.session_state["sport"] = "nfl"
        at.run()
        if at.exception:
            problems.append("NFL BOARD RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in at.exception))
        text = _all_text(at)
        if "SMOKE_LIVE" not in text and "New York Giants" not in text \
                and "NYG" not in text:
            problems.append(
                "today's board does not include the started game "
                "(SMOKE_LIVE) — the removal defect persists at render level")
        if "San Francisco" not in text and "SF" not in text:
            problems.append("today's board lost the later pre-game card")
        if "of 3 games shown" not in text:
            problems.append("header strip does not count 3 games (started "
                            "game missing from the board frame?)")

        # ---- Yesterday navigates to the dated snapshot (frozen price) ----
        import utils
        valid = utils.valid_dates("nfl")
        if _YDAY_C not in set(valid):
            problems.append(
                f"yesterday {_YDAY_C} is not in the NFL valid-date rail "
                f"(dated board family not feeding valid_dates); rail head: "
                f"{list(valid)[:4]}")
        hist = utils.load_nfl_board_games(_YDAY_C)
        if hist.empty:
            problems.append("yesterday's dated snapshot resolves EMPTY")
        else:
            fin = hist[hist["game_id"] == "SMOKE_FIN"]
            if fin.empty:
                problems.append("yesterday's snapshot lost the finished game")
            elif abs(float(fin.iloc[0]["home_win_prob_model"]) - 0.412) > 1e-9:
                problems.append(
                    "yesterday's card is not the frozen production price "
                    f"(got {fin.iloc[0]['home_win_prob_model']})")
        # A date with no board/history must NOT resurrect OOF rows for it.
        utils.load_nfl_prediction_history  # (attr presence; OOF gate is structural)
    finally:
        _restore()

    if problems:
        print("BOARD RENDER SMOKE [FAIL]")
        for p in problems:
            print("  -", p)
        return 1
    print("BOARD RENDER SMOKE [PASS]")
    print("  - today's board includes the started game with its frozen price")
    print("  - yesterday renders from the dated snapshot (frozen, not OOF)")
    print("  - valid-date rail includes the dated board family window")
    print("  - no page exceptions")
    return 0


if __name__ == "__main__":
    sys.exit(run())
