"""Smoke test — Power Rankings page, SPORT-DISPATCHED.

The NFL Power Rankings page now runs the SAME code as MLB (no sport-special
path). This test:

1. Writes a REPRESENTATIVE ``{sport}_power_rankings_*.csv`` (matching the
   exact shape each backend writer emits — the MLB-identical 12-column
   contract, exercised through the loader's w/l fallback) into the real
   ``{sport}-backend/data_delivery`` dir, so the page renders with real data
   (restored after the run, never left behind).
2. Runs the ACTUAL ``power_rankings.py`` under ``sport=nfl`` and asserts the
   top-15 table renders with the exact 9 column headers — RANK | TEAM | ELO |
   W-L | PCT | RUN DIFF | L10 | HOME% | AWAY% — and the top-15 team rows are
   present (a below-top-15 team is NOT shown), with no exceptions.
3. Renders the SAME page under ``sport=mlb``, ``sport=nhl`` and ``sport=nba``
   and asserts each runs the shared path clean. All four artifacts carry
   MLB's exact column contract — the diff lives in ``run_diff`` everywhere —
   and NBA's page alone relabels that column header ``POINT DIFF``. The label
   is a header-text switch, not a column fork, so no sport-special code path
   exists anywhere in the page.

The point of (3) is structural alignment: all four sports go through one
renderer, and only the diff column's label/sources differ.

Run from the frontend/ directory:
    python -m test_power_rankings_smoke
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent if FRONTEND_DIR.name == "frontend" else FRONTEND_DIR
NFL_DD = REPO_ROOT / "nfl-backend" / "data_delivery"

ARTIFACT_DATE = "20260831"
ARTIFACT_NAME = f"nfl_power_rankings_{ARTIFACT_DATE}.csv"
ARTIFACT_PATH = NFL_DD / ARTIFACT_NAME

# Every sport the shared renderer must handle, and the diff header each
# publishes. All four share MLB's byte-identical column contract (the diff
# is ``run_diff``); only NBA's header text differs.
SPORTS = {
    "nfl": ("nfl-backend", "nfl_power_rankings_", "RUN DIFF"),
    "nhl": ("nhl-backend", "nhl_power_rankings_", "RUN DIFF"),
    "nba": ("nba-backend", "nba_power_rankings_", "POINT DIFF"),
    "mlb": ("mlb-backend", "power_rankings_", "RUN DIFF"),
}

WRITTEN: list[Path] = []
# Path -> original bytes of a PRE-EXISTING (committed) artifact this test
# overwrites with a fixture; restored on cleanup, never deleted.
_BACKUPS: dict[Path, bytes] = {}

# 17 teams with descending Elo: rank 1 = KC ... rank 17 = MIA. Top 15 should
# render; the 16th/17th (SEA, MIA) must be absent from the top-15 table.
_TEAMS = [
    ("KC", 1570.2, 8, 2, 7, 128), ("BUF", 1555.0, 9, 1, 8, 96),
    ("DET", 1539.4, 7, 3, 6, 74), ("PHI", 1527.8, 8, 2, 6, 61),
    ("BAL", 1519.9, 7, 3, 7, 55), ("GB", 1510.3, 6, 4, 6, 33),
    ("SF", 1504.6, 5, 5, 5, 27), ("MIN", 1498.1, 6, 4, 6, 12),
    ("TB", 1490.7, 5, 5, 4, -3), ("DAL", 1486.2, 5, 5, 5, -17),
    ("CIN", 1480.0, 4, 6, 3, -22), ("ATL", 1474.5, 4, 6, 5, -31),
    ("HOU", 1471.8, 4, 6, 3, -40), ("WAS", 1466.1, 3, 7, 2, -52),
    ("LAC", 1459.0, 3, 7, 4, -61), ("SEA", 1445.3, 2, 8, 2, -79),
    ("MIA", 1438.6, 2, 8, 1, -88),
]


def _rankings_frame(sport: str) -> pd.DataFrame:
    # Every sport publishes the diff under the same column name — NBA's
    # POINT DIFF is a header label applied by the page, never a CSV fork.
    diff_col = "run_diff"
    rows = []
    for i, (team, elo, w, l, home_wins, run_diff) in enumerate(_TEAMS, start=1):
        row = {"team": team, "team_name": team, "elo": round(float(elo), 1),
               "wins": w, "losses": l, "record": f"{w}-{l}",
               "pct": round(w / (w + l), 3), diff_col: run_diff,
               "l10": f"{max(0, 5 - i % 6)}-{min(5, i % 6)}",
               "home_pct": round(home_wins / max(w, 1), 3), "away_pct": 0.5}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.index += 1
    df.index.name = "rank"
    return df


def _stage(path: Path, data: bytes) -> None:
    """Write a fixture over ``path``, preserving any pre-existing (committed)
    artifact's bytes so cleanup can restore it rather than delete it."""
    if path.exists():
        _BACKUPS[path] = path.read_bytes()
    path.write_bytes(data)
    WRITTEN.append(path)


def _write_artifacts() -> None:
    for sport, (subdir, prefix, _hdr) in SPORTS.items():
        dd = REPO_ROOT / subdir / "data_delivery"
        dd.mkdir(parents=True, exist_ok=True)
        # ``rank`` is the frame's NAMED index, so write it to the CSV
        # (index=True), byte-for-byte the same as the backend's to_csv().
        buf = io.BytesIO()
        _rankings_frame(sport).to_csv(buf)
        _stage(dd / f"{prefix}{ARTIFACT_DATE}.csv", buf.getvalue())


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
        at = AppTest.from_file(str(FRONTEND_DIR / "power_rankings.py"),
                               default_timeout=60)
        at.session_state["sport"] = "nfl"
        at.session_state["selected_date"] = ARTIFACT_DATE
        at.run()

        if at.exception:
            problems.append("NFL PAGE RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in at.exception))

        text = _all_text(at)

        # heading
        if "Power Rankings" not in text:
            problems.append("missing 'Power Rankings' heading")

        # exact 9 column headers
        for h in ("RANK", "TEAM", "ELO", "W-L", "PCT", "RUN DIFF",
                  "L10", "HOME%", "AWAY%"):
            if h not in text:
                problems.append(f"missing column header {h!r}")

        # top-15 rows present, rank 16/17 absent
        top15 = [t for t, *_ in _TEAMS[:15]]
        for team in top15:
            if team not in text:
                problems.append(f"top-15 team {team!r} not rendered")
        for team in ("SEA", "MIA"):
            if team in text:
                problems.append(f"below-top-15 team {team!r} rendered (should be cut)")

        if problems:
            print("POWER-RANKINGS SMOKE TEST — FAIL (sport=nfl)")
            for p in problems:
                print("  -", p)
            return 1

        print("POWER-RANKINGS SMOKE TEST — PASS (sport=nfl)")
        print("  - no exceptions; top-15 table + 9 headers rendered; "
              "below-top-15 teams cut")

        # Every other sport must run the SAME shared path, no exception, and
        # publish its own diff header. Only the diff column's label differs.
        for sport, (_sub, _prefix, diff_header) in SPORTS.items():
            if sport == "nfl":
                continue  # already asserted in full above
            other = AppTest.from_file(str(FRONTEND_DIR / "power_rankings.py"),
                                      default_timeout=60)
            other.session_state["sport"] = sport
            other.session_state["selected_date"] = ARTIFACT_DATE
            other.run()
            if other.exception:
                prob = "\n  ".join(str(e.value) for e in other.exception)
                print(f"POWER-RANKINGS SMOKE TEST — FAIL (sport={sport})")
                print("  - path raised:\n    " + prob)
                return 1
            otext = _all_text(other)
            # the sport's own diff header must render; the other label must not
            if diff_header not in otext:
                print(f"POWER-RANKINGS SMOKE TEST — FAIL (sport={sport})")
                print(f"  - missing diff header {diff_header!r}")
                return 1
            wrong = "POINT DIFF" if diff_header == "RUN DIFF" else "RUN DIFF"
            if wrong in otext:
                print(f"POWER-RANKINGS SMOKE TEST — FAIL (sport={sport})")
                print(f"  - rendered the wrong diff header {wrong!r}")
                return 1
            print(f"  - sport={sport} path clean (no exception), "
                  f"diff header {diff_header!r}")
        print("POWER-RANKINGS SMOKE TEST — PASS (all sports)")
        return 0
    finally:
        _remove_artifacts()


if __name__ == "__main__":
    sys.exit(run())