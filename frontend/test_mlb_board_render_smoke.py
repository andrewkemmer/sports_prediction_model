"""Full local render — MLB Today's Games board (the 2026-09-28 remediation).

Drives the ACTUAL ``frontend/todays_games.py`` under Streamlit AppTest
against the real committed data_delivery artifacts, reconstructing the
regression shape:

  A. 2026-09-28 (off-day, board deleted): landing on 0928 must NEVER show
     games "as today" — the page either recovers to the newest real board
     with an honest banner, or dead-ends honestly.
  B. Direct loader proof: a polluted board file (15 decided rows dated
     0925/0926 under the 0928 filename — the polluter's exact shape)
     yields ZERO rows from ``utils.load_todays_games``.
  D. Page-level proof: with that polluted file present, no 0925/0926
     content can reach the rendered page.
  E. With recovery disabled, 0928 dead-ends with the honest
     "No game board exists for Monday, September 28, 2026" warning.
  C. The legit board nearest the retention window still renders intact
     (its full card count, its own date, no recovery banner, its decided
     picks graded). The target is picked from TODAY's valid set — with
     decided rows, since a same-night board is still LIVE/pre-game —
     because the 10-day rolling MLB retention window legitimately drops
     2026-09-27 once ET today passes 2026-10-07; an out-of-retention
     board MUST fall back to recovery.

Run from the frontend/ directory:
    python -m test_mlb_board_render_smoke
"""
from __future__ import annotations

import io
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit.testing.v1 import AppTest

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
DD = REPO_ROOT / "mlb-backend" / "data_delivery"
BOARD_28 = DD / "todays_games_20260928.csv"
PROBE = FRONTEND_DIR / "_mlb_loader_probe.py"

ET_TODAY = datetime.now(ZoneInfo("America/New_York")).date()
assert ET_TODAY.isoformat() >= "2026-09-28", \
    f"recovery window assumes ET today >= 2026-09-28, got {ET_TODAY}"

# The polluter's exact shape: 2 finals from 0925 + 13 from 0926, all
# decided, under a todays_games_20260928.csv filename.
_POLL_PAIRS = [
    ("20260925_CHC@BOS_2", "2026-09-25", "CHC", "BOS", 3, 4),
    ("20260925_TB@PHI", "2026-09-25", "TB", "PHI", 0, 2),
    ("20260926_ATL@MIA", "2026-09-26", "ATL", "MIA", 2, 5),
    ("20260926_CIN@TOR", "2026-09-26", "CIN", "TOR", 4, 1),
    ("20260926_CLE@KC", "2026-09-26", "CLE", "KC", 6, 3),
    ("20260926_HOU@ATH", "2026-09-26", "HOU", "ATH", 5, 2),
    ("20260926_LAA@SEA", "2026-09-26", "LAA", "SEA", 1, 4),
    ("20260926_LAD@SF", "2026-09-26", "LAD", "SF", 7, 2),
    ("20260926_NYM@WSH", "2026-09-26", "NYM", "WSH", 3, 6),
    ("20260926_PIT@DET", "2026-09-26", "PIT", "DET", 2, 8),
    ("20260926_STL@MIL", "2026-09-26", "STL", "MIL", 4, 4),
    ("20260926_TEX@MIN", "2026-09-26", "TEX", "MIN", 9, 5),
    ("20260926_AZ@SD", "2026-09-26", "AZ", "SD", 1, 3),
    ("20260926_COL@CWS", "2026-09-26", "COL", "CWS", 0, 7),
    ("20260926_BAL@NYY", "2026-09-26", "BAL", "NYY", 5, 6),
]

POLLUTED_CSV = (
    "game_id,game_date,start_time_et,away_team,home_team,away_score,"
    "home_score,home_win,model_correct,game_status,venue,"
    "home_win_prob_model,model_pick\n"
    + "\n".join(
        f"{gid},{gd},13:05,{aw},{hm},{asc},{hsc},"
        f"{1.0 if hsc > asc else 0.0},{1 if hsc > asc else 0},Final,"
        f"Park {i},0.{55 + i % 40},{hm if hsc > asc else aw}"
        for i, (gid, gd, aw, hm, asc, hsc) in enumerate(_POLL_PAIRS)
    ) + "\n"
)

_BACKUPS: dict[Path, bytes | None] = {}


def _backup(path: Path) -> None:
    _BACKUPS.setdefault(path, path.read_bytes() if path.exists() else None)


def _clear_caches() -> None:
    st.cache_data.clear()


def _restore_all() -> None:
    for p, data in _BACKUPS.items():
        if data is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(data)
    _BACKUPS.clear()
    _clear_caches()


def _harvest(at: AppTest) -> str:
    parts: list[str] = []
    for seq in (at.markdown, at.info, at.warning, at.caption,
                at.error, at.success, at.button, at.subheader, at.header):
        for el in seq:
            v = getattr(el, "value", None)
            if not isinstance(v, str):
                # Buttons carry their text in .label (.value is the clicked
                # bool) — the date-nav field renders the board date there.
                v = getattr(el, "label", "") or ""
            parts.append(str(v))
    return "\n".join(parts)


def _pin(at: AppTest, date_compact: str) -> None:
    at.session_state["selected_date"] = date_compact
    at.session_state["_nav_sport"] = "mlb"
    at.session_state["sport"] = "mlb"


def _run_page(date_compact: str) -> tuple[AppTest, str]:
    at = AppTest.from_file("todays_games.py", default_timeout=120)
    _pin(at, date_compact)
    at.run()
    assert not at.exception, f"page exception: {at.exception}"
    return at, _harvest(at)


PROBE_SRC = (
    "import streamlit as st\n"
    "import utils\n"
    "df = utils.load_todays_games('20260928')\n"
    "st.session_state['_probe_rows'] = int(len(df))\n"
)


def main() -> int:
    _backup(BOARD_28)  # records the real state (absent) for restore
    print(f"ET today: {ET_TODAY}")

    # ---- A: 0928 with NO board file (real remediated state) ----
    # The recovery TARGET is date-dependent by design: the page recovers to
    # the most recent VALID board on or before ET-today (0927 while the run
    # is live on the 28th; the 0929 slate once the calendar rolls over past
    # midnight). The CONTRACT under test is date-independent: banner +
    # selected-date disclosure + recovery only to a valid, non-recycled
    # date — never 0925/0926 content under the 0928 header.
    BOARD_28.unlink(missing_ok=True)
    _clear_caches()
    at, text = _run_page("20260928")
    assert "Recovery view" in text, "expected the honest recovery banner"
    assert "September 28, 2026" in text
    import utils as _u
    _valid = [str(d) for d in _u.valid_dates("mlb") if str(d) != "20260928"]
    _expected = max((d for d in _valid
                     if d <= ET_TODAY.strftime("%Y%m%d")), default=None)
    assert _expected, "no valid recovery candidate exists for this ET day"
    _exp_long = (datetime.strptime(_expected, "%Y%m%d")
                 .strftime("%B %d, %Y").replace(" 0", " "))
    assert _exp_long in text, (
        f"recovery must land on the most recent valid board ({_expected})")
    assert "20260925" not in text, "recycled 0925 game leaked into 0928 view"
    assert "20260926" not in text, "recycled 0926 game leaked into 0928 view"
    assert "September 25, 2026" not in text
    assert "September 26, 2026" not in text
    print(f"A PASS  0928 lands on an honest recovery view of {_expected}; "
          "no 0925/0926 content anywhere")

    # ---- B: direct loader — polluted 0928 file -> zero rows ----
    BOARD_28.write_text(POLLUTED_CSV, encoding="utf-8")
    PROBE.write_text(PROBE_SRC, encoding="utf-8")
    _clear_caches()
    try:
        probe = AppTest.from_file("_mlb_loader_probe.py", default_timeout=60)
        probe.run()
        assert not probe.exception, f"probe exception: {probe.exception}"
        rows = probe.session_state["_probe_rows"]
        assert rows == 0, (
            f"row-level date honesty failed: {rows} foreign rows survived "
            "the loader under the 0928 header")
        print("B PASS  polluted 0928 board file loads as ZERO rows "
              "(15 foreign-date rows dropped)")
    finally:
        PROBE.unlink(missing_ok=True)

    # ---- D: page-level — polluted file present, no foreign content ----
    BOARD_28.write_text(POLLUTED_CSV, encoding="utf-8")
    _clear_caches()
    at, text = _run_page("20260928")
    assert "Recovery view" in text
    assert "September 28, 2026" in text
    assert "20260925" not in text, "polluter 0925 rows rendered under 0928"
    assert "20260926" not in text, "polluter 0926 rows rendered under 0928"
    assert "September 25, 2026" not in text
    assert "September 26, 2026" not in text
    print("D PASS  with the polluted file on disk, the 0928 page still "
          "never renders 0925/0926 content")

    # ---- E: recovery exhausted -> honest dead-end for the off-day ----
    # AppTest.from_file re-executes the page source in a fresh namespace, so
    # patching todays_games module attrs has no effect on the run — but the
    # page resolves `utils` through the SHARED imported module. Emptying
    # valid_dates makes every recovery candidate fail the valid-set gate
    # (_recovered_board refuses out-of-set candidates), exhausting the walk
    # into the honest dead-end warning.
    BOARD_28.unlink(missing_ok=True)
    _clear_caches()
    import utils
    _orig_valid = utils.valid_dates
    utils.valid_dates = lambda sport_key=None: ()
    try:
        at, text = _run_page("20260928")
        assert "No game board exists for Monday, September 28, 2026" in text
        assert "20260925" not in text and "20260926" not in text
        assert "Recovery view" not in text
        print("E PASS  with recovery exhausted, 0928 dead-ends honestly "
              "('No game board exists for Monday, September 28, 2026')")
    finally:
        utils.valid_dates = _orig_valid

    # ---- C: the legit board renders intact ----
    # Retention-aware target (see module docstring): prefer the original
    # 2026-09-27 remediation slate while it is still inside the rolling
    # 10-day valid window, otherwise the newest real local board that has
    # DECIDED rows — a same-night board is still LIVE/pre-game and renders
    # no accuracy line, so the accuracy contract needs a settled slate.
    # The CONTRACT under test is date-independent: a real board renders
    # under its own date with its full card count, no recovery banner,
    # and its decided picks graded.
    _clear_caches()
    import csv as _csv

    def _board_rows(date_str: str) -> list[dict]:
        with (DD / f"todays_games_{date_str}.csv").open(
                newline="", encoding="utf-8") as fh:
            return list(_csv.DictReader(fh))

    _real = sorted(
        (str(d) for d in _u.valid_dates("mlb")
         if (DD / f"todays_games_{str(d)}.csv").exists()),
        reverse=True)
    assert _real, "no real local board is valid today"
    _settled = [d for d in _real if any(r.get("home_win") for r in _board_rows(d))]
    assert _settled, "no valid local board has decided rows today"
    _c_date = "20260927" if "20260927" in _settled else _settled[0]
    _c_rows = len(_board_rows(_c_date))
    at, text = _run_page(_c_date)
    _c_long = (datetime.strptime(_c_date, "%Y%m%d")
               .strftime("%B %d, %Y").replace(" 0", " "))
    assert _c_long in text
    assert "Recovery view" not in text
    m = re.search(rf"{_c_rows} of (\d+) games shown", text)
    assert m, (f"expected '{_c_rows} of N games shown' strip for "
               f"{_c_date}, got: {text[:400]}")
    assert "20260925" not in text and "20260926" not in text
    assert "CORRECT PICK" in text or "Model Correct" in text or "accuracy" in text
    print(f"C PASS  {_c_date} intact: {_c_rows} cards, "
          f"league_total={m.group(1)}, no recovery banner, "
          "no foreign-date content")

    print("\nMLB BOARD RENDER PROOF - PASS (5 scenarios)")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        _restore_all()
        PROBE.unlink(missing_ok=True)
    sys.exit(code)
