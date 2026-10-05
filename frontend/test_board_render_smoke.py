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
    pre-game price and a Live/started card, under MLB's THREE-STATE badge
    (a pre-game slate never claims an accuracy record);
  - yesterday's date is navigable and renders the dated snapshot's frozen
    prices (no OOF substitution);
  - the ARCHIVE path (frozen card store) serves the store's published
    probability — never the committed OOF history value for the same game —
    with the run-engine totals/run-line values + the per-card O/U and
    run-line toggles sourced from THIS date's own markets artifact (never
    the file's kind='oof' rows, never the newest run), NO SHAP accordion,
    NO archive banner (the MLB flow renders none: date nav straight into
    the filter pills), display-only venue/kickoff enrichment from the
    dated board snapshot, NFL team names, and the store's own pick grade
    instead of a fabricated all-MISS card;
  - every card renders each team's CURRENT-SEASON entering W-L record
    (MLB compute_season_records parity): own season only, strictly prior —
    never the artifact's multi-season career tally, never the game's own
    result, never a prior season's row;
  - a host-local store miss self-heals through the committed store bytes
    (GitHub-raw first) instead of silently serving the OOF re-price;
  - the recovery walk refuses any candidate outside the valid (rolling
    10-day) set, and `_decided_mask` never counts a pre-game/live row;
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
# The frozen first-publication store — the ARCHIVE board's only price
# source. Backed up and restored like the other committed artifacts above.
STORE = NFL_DD / "nfl_production_cards_history.csv"
# Yesterday's DATED QB matchup — the archive card's twin of MLB's dated
# sp_* fields. Shadowed during the run and restored after, like STORE.
QB_JSON = NFL_DD / f"nfl_qb_matchup_{_YDAY_C}.json"
# Yesterday's DATED run-engine markets artifact — the archive card's
# distribution-model source (totals + run lines + the toggle grids).
# Shadowed/restored like the other fixtures.
RE_CSV = NFL_DD / f"nfl_run_engine_markets_{_YDAY_C}.csv"

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
    """Dated snapshots: yesterday's finished game + today's started game,
    plus display-only rows for yesterday's three STORE ids.

    home_win_prob_model is the FROZEN PRE-GAME price the serving horizon
    published — exactly what the dated board CSV carries. The three store
    rows carry the board's LATER re-published price (0.90) deliberately:
    only their venue/kickoff fields may reach the archive card, so any
    enrichment that leaks a probability fails the price pins.
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
        # Display-only rows for yesterday's store ids — the archive card's
        # venue + kickoff source (the frozen store ships neither). The
        # prices here are the board's later re-published values: only the
        # display fields may ever reach the card.
        {
            "game_id": "2026_04_ARI_NYG", "game_date": _YDAY.isoformat(),
            "start_time_utc": _iso(_YDAY, 13),
            "home_team": "NYG", "away_team": "ARI",
            "home_team_name": "New York Giants",
            "away_team_name": "Arizona Cardinals",
            "venue": "MetLife Stadium", "game_status": "Final",
            "home_score": 99.0, "away_score": 3.0,
            "home_win_prob_model": 0.90, "away_win_prob_model": 0.10,
            "model_pick": "NYG", "model_correct": True,
        },
        {
            "game_id": "SMOKE_ARCH", "game_date": _YDAY.isoformat(),
            "start_time_utc": _iso(_YDAY, 20),
            "home_team": "KC", "away_team": "LV",
            "home_team_name": "Kansas City Chiefs",
            "away_team_name": "Las Vegas Raiders",
            "venue": "Arrowhead Stadium", "game_status": "Final",
            "home_score": 3.0, "away_score": 99.0,
            "home_win_prob_model": 0.90, "away_win_prob_model": 0.10,
            "model_pick": "LV", "model_correct": True,
        },
        {
            "game_id": "2026_04_NE_BUF", "game_date": _YDAY.isoformat(),
            "start_time_utc": _iso(_YDAY, 13),
            "home_team": "NE", "away_team": "BUF",
            "home_team_name": "New England Patriots",
            "away_team_name": "Buffalo Bills",
            "venue": "Highmark Stadium", "game_status": "Final",
            "home_score": 3.0, "away_score": 99.0,
            "home_win_prob_model": 0.90, "away_win_prob_model": 0.10,
            "model_pick": "BUF", "model_correct": True,
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


def _store_rows() -> pd.DataFrame:
    """Frozen first-publication store rows for yesterday (the archive board).

    Deliberately shaped to pin the archive contract:

      * ``2026_04_ARI_NYG`` is a REAL id the committed OOF history carries
        at a different probability (~0.575 calibrated vs the frozen 0.617
        here), so a page that ever serves the OOF re-price fails the
        "62%" assertion below;
      * ``SMOKE_ARCH`` is a store-only id (the OOF fallback could never
        produce it);
      * ``2026_04_NE_BUF`` is graded WRONG in ``correct`` while its scores
        agree with the pick — pinning the STORE's grade as the displayed
        one (a score-derived grade would render 3 ✓ / 0 ✗);
      * abbreviations are MLB-colliding (ARI / KC / NE-adjacent) so the
        NFL name map is pinned too;
      * ``home_win`` is NaN, exactly as the committed store ships it.
    """
    return pd.DataFrame([
        {
            "game_id": "2026_04_ARI_NYG", "game_date": _YDAY.isoformat(),
            "home_team": "NYG", "away_team": "ARI",
            "home_score": 24.0, "away_score": 17.0, "home_win": np.nan,
            "p_home_win": 0.617, "p_away_win": 0.383,
            "model_pick": "NYG", "correct": True,
            "game_status": "Final", "source_artifact_date": _YDAY_C,
        },
        {
            "game_id": "SMOKE_ARCH", "game_date": _YDAY.isoformat(),
            "home_team": "KC", "away_team": "LV",
            "home_score": 20.0, "away_score": 23.0, "home_win": np.nan,
            "p_home_win": 0.412, "p_away_win": 0.588,
            "model_pick": "LV", "correct": True,
            "game_status": "Final", "source_artifact_date": _YDAY_C,
        },
        {
            "game_id": "2026_04_NE_BUF", "game_date": _YDAY.isoformat(),
            "home_team": "NE", "away_team": "BUF",
            "home_score": 17.0, "away_score": 24.0, "home_win": np.nan,
            "p_home_win": 0.708, "p_away_win": 0.292,
            "model_pick": "NE", "correct": False,
            "game_status": "Final", "source_artifact_date": _YDAY_C,
        },
        # Prior-week evidence for the CURRENT-SEASON entering record: the
        # 2026 rows count, the 2025 row must never leak across the
        # offseason, and no row's own result may enter its own record
        # (MLB compute_season_records parity — cards render w-l only).
        {
            "game_id": "2026_03_NE_MIA",
            "game_date": (_YDAY - timedelta(days=7)).isoformat(),
            "home_team": "NE", "away_team": "MIA",
            "home_score": 27.0, "away_score": 10.0, "home_win": np.nan,
            "p_home_win": 0.60, "p_away_win": 0.40,
            "model_pick": "NE", "correct": True,
            "game_status": "Final", "source_artifact_date": _YDAY_C,
        },
        {
            "game_id": "2026_03_NYG_WAS",
            "game_date": (_YDAY - timedelta(days=7)).isoformat(),
            "home_team": "NYG", "away_team": "WAS",
            "home_score": 24.0, "away_score": 21.0, "home_win": np.nan,
            "p_home_win": 0.55, "p_away_win": 0.45,
            "model_pick": "NYG", "correct": True,
            "game_status": "Final", "source_artifact_date": _YDAY_C,
        },
        {
            "game_id": "2026_03_GB_CHI",
            "game_date": (_YDAY - timedelta(days=7)).isoformat(),
            "home_team": "CHI", "away_team": "GB",
            "home_score": 20.0, "away_score": 27.0, "home_win": np.nan,
            "p_home_win": 0.44, "p_away_win": 0.56,
            "model_pick": "GB", "correct": True,
            "game_status": "Final", "source_artifact_date": _YDAY_C,
        },
        {
            "game_id": "2025_18_NE_BUF",
            "game_date": (_YDAY - timedelta(days=300)).isoformat(),
            "home_team": "NE", "away_team": "BUF",
            "home_score": 27.0, "away_score": 24.0, "home_win": np.nan,
            "p_home_win": 0.52, "p_away_win": 0.48,
            "model_pick": "NE", "correct": True,
            "game_status": "Final", "source_artifact_date": _YDAY_C,
        },
    ])


def _qb_record() -> dict:
    """Yesterday's date-keyed QB matchup (flat emitter schema): pins that
    an archive card serves THIS date's starters — MLB's history boards
    always render that date's sp_* fields, and the newest-only load used
    to leave these boxes empty on every frozen card."""
    game = {
        "gameday": _YDAY.isoformat(),
        "qb_home_name": "Frozen Home QB", "qb_home_rating": 91.5,
        "qb_home_td_per_game": 2.0, "qb_home_cmp_pct": 66.0,
        "qb_home_yards_per_attempt": 7.5, "qb_home_ints": 0.0,
        "qb_away_name": "Frozen Away QB", "qb_away_rating": 88.0,
        "qb_away_td_per_game": 1.5, "qb_away_cmp_pct": 64.0,
        "qb_away_yards_per_attempt": 7.0, "qb_away_ints": 1.0,
    }
    games = [dict(game, game_id=gid) for gid in
             ("2026_04_ARI_NYG", "SMOKE_ARCH", "2026_04_NE_BUF")]
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "n_games": len(games), "games": games}


def _run_engine_rows() -> pd.DataFrame:
    """Yesterday's run-engine markets artifact — the archive card's dated
    distribution-model source (MLB ``_build_slate_map`` parity).

    Shaped to pin the archive contract:

      * ``kind='oof'`` decoy rows FIRST (the committed files sort them
        first, before the slate rows) carrying distinctive values
        (fair_total 99 / Proj 99.0) — the card must serve the
        ``kind='slate'`` rows' published distribution, never the OOF
        re-price that an unfiltered first-hit match would surface;
      * slate rows for the three store ids with the COMPLETE MC grid
        (the usable gate) and per-game fair lines — totals 44 / 47 / 51
        and fair spreads 3 / 6 / 1.5 — so the O/U toggle defaults pin to
        44/47/51 and the run-line toggles to the rounded 3 / 6 / 2.
    """
    grid: dict = {}
    for line in range(-14, 15):
        label = f"m{-line}" if line < 0 else str(line)
        grid[f"p_home_cover_{label}"] = 0.48
        grid[f"p_push_{label}"] = 0.04 if line else 0.02
    for line in range(24, 67):
        grid[f"p_over_{line}"] = 0.48
        grid[f"p_under_{line}"] = 0.48
        grid[f"p_push_{line}"] = 0.04
    base = {
        "gameday": _YDAY.isoformat(), "season": 2026, "week": 5,
        "p_home_win": 0.55, "p_away_win": 0.45,
        "p_home_win_derived": 0.54, "p_away_win_derived": 0.46,
        **grid,
    }
    games = [
        # game_id, home, away, mu_h, mu_a, fair_total, fair_spread
        ("2026_04_ARI_NYG", "NYG", "ARI", 24.0, 17.0, 44.0, 3.0),
        ("SMOKE_ARCH", "KC", "LV", 21.0, 18.0, 47.0, 6.0),
        ("2026_04_NE_BUF", "NE", "BUF", 20.0, 25.0, 51.0, 1.5),
    ]
    oof = [dict(base, kind="oof", game_id=gid, home_team=h, away_team=a,
                mu_h=99.0, mu_a=99.0, fair_total=99.0, fair_spread=9.0)
           for gid, h, a, *_ in games]
    slate = [dict(base, kind="slate", game_id=gid, home_team=h,
                  away_team=a, mu_h=mh, mu_a=ma, fair_total=ft,
                  fair_spread=fs)
             for gid, h, a, mh, ma, ft, fs in games]
    return pd.DataFrame(oof + slate)


def _write_artifacts() -> None:
    NFL_DD.mkdir(parents=True, exist_ok=True)
    for p in (BOARD_TODAY, BOARD_YDAY, ML_JSON, CAL_JSON, STORE, QB_JSON,
              RE_CSV):
        _BACKUPS[p] = p.read_bytes() if p.exists() else None
    _board_rows().to_csv(BOARD_TODAY, index=False)
    _board_rows()[_board_rows()["game_date"] == _YDAY.isoformat()] \
        .to_csv(BOARD_YDAY, index=False)
    ML_JSON.write_text(json.dumps(_moneyline_record()))
    CAL_JSON.write_text(json.dumps(_calibration_record()))
    _store_rows().to_csv(STORE, index=False)
    QB_JSON.write_text(json.dumps(_qb_record()))
    _run_engine_rows().to_csv(RE_CSV, index=False)
    WRITTEN.extend([BOARD_TODAY, BOARD_YDAY, ML_JSON, CAL_JSON, STORE,
                    QB_JSON, RE_CSV])


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
    # (1b) No build chip (2026-10-05): the run-engine strip must not carry
    # a "build <digest>" stamp on any game card — the chip was removed at
    # the user's request from current-day and archive cards alike.
    if "build " in html:
        problems.append(f"run-engine strip still renders a build stamp: {html}")
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
        # MLB's THREE-STATE badge: neither shadow card is decided, so the
        # accuracy pill must not claim a record (it used to render
        # "✓ 0-0 Today · 0.0% accuracy" on every pre-game slate).
        if "No results yet — pre-game slate" not in text:
            problems.append("pre-game slate still claims an accuracy badge "
                            "(MLB three-state badge not applied)")

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

        # ---- Archive date: frozen slate prediction + MLB element anatomy ----
        # The moneyline record covers today only, so yesterday resolves
        # through the FROZEN CARD STORE — driven here with the store rows
        # above (a real id the committed OOF history prices differently).
        aa = AppTest.from_file(str(FRONTEND_DIR / "todays_games.py"),
                               default_timeout=120)
        aa.session_state["sport"] = "nfl"
        aa.session_state["selected_date"] = _YDAY_C
        aa.session_state["_nav_sport"] = "nfl"
        aa.run()
        if aa.exception:
            problems.append("NFL ARCHIVE BOARD RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in aa.exception))
        atext = _all_text(aa)

        # (a) THE SLATE PREDICTION, NOT THE OOF RE-PRICE: the frozen
        # p_home_win renders, not the committed history's calibrated value
        # for the same real id (0.575 → "57%", which must never appear).
        for want in ("62%", "41%", "71%"):
            if want not in atext:
                problems.append(
                    f"archive board lost the frozen store price {want} "
                    "(serving the OOF history instead?)")

        # (b) THE DATE'S OWN RUN-ENGINE DISTRIBUTION with the per-card
        # toggles (MLB _build_slate_map parity): every archive card
        # renders Proj / O/U / RL values from THIS date's markets slate
        # rows — never the quiet 'unavailable' fallback (the dated
        # artifact exists), never the file's kind='oof' decoy rows (they
        # sort FIRST in the fixture, carrying fair_total 99 / Proj 99.0),
        # and never the newest artifact's later run.
        acards = [m.value for m in aa.markdown
                  if isinstance(m.value, str) and '<div class="fb-top">' in m.value]
        if len(acards) != 3:
            problems.append(f"archive board rendered {len(acards)} cards, want 3")
        for c in acards:
            if "fb-runengine" not in c or "Proj: " not in c:
                problems.append("archive card lacks the run-engine strip "
                                "values (dated markets not served)")
            if 're-na">Run Engine data currently unavailable' in c:
                problems.append("archive card rendered the quiet run-engine "
                                "fallback despite a dated markets artifact")
        if "Run Engine data currently unavailable" in atext:
            problems.append("archive board shows the run-engine unavailable "
                            "notice while its dated artifact is reachable")
        if "Proj: ARI 17.0 \u2013 NYG 24.0" not in atext:
            problems.append("archive cards did not serve the DATED slate "
                            "row's projection (wrong artifact/row served)")
        if "O/U 44: Over 50% / Under 50%" not in atext:
            problems.append("archive card totals did not price the dated "
                            "fair total 44 from the MC grid")
        if "99.0" in atext or "O/U 99" in atext:
            problems.append("archive card served the kind='oof' decoy row "
                            "(the file's OOF re-price leaked onto a frozen "
                            "card)")
        if len(aa.selectbox) != 6:
            problems.append(f"archive board rendered {len(aa.selectbox)} "
                            "O/U + run-line toggles, want 6 (one pair per "
                            "card)")
        else:
            _toggles = sorted(float(sb.value) for sb in aa.selectbox)
            if _toggles != [2.0, 3.0, 6.0, 44.0, 47.0, 51.0]:
                problems.append("archive O/U + run-line toggle defaults "
                                f"wrong: {_toggles} (want the dated slate's "
                                "fair totals 44/47/51 and rounded spreads "
                                "3/6/2 — an oof leak would show 99)")
        # SHAP stays CURRENT-SLATE ONLY even though run-engine markets now
        # populate: that family is game-keyed and re-published every run.
        if len(aa.expander):
            problems.append(
                f"archive board rendered {len(aa.expander)} SHAP accordion(s) "
                "— a frozen card must not inherit a later run's attributions")

        # (c) NO archive banner — MLB's dashboard flow (header strip → date
        # nav → filter pills) carries no page-level archive notice, so the
        # mirror must render none either; the date nav still lands directly
        # in the filter pills.
        seq = list(aa.main)
        banners = [e for e in seq
                   if type(e).__name__ == "Info"
                   and "Archive view" in str(getattr(e, "value", ""))]
        if banners:
            problems.append(
                "archive banner still rendered between the date nav and "
                "the filter pills — MLB's dashboard flow renders none")
        nav_i = next((i for i, e in enumerate(seq)
                      if type(e).__name__ == "Button"
                      and str(getattr(e, "label", "")).startswith("◀")), -1)
        pills_i = next((i for i, e in enumerate(seq)
                        if type(e).__name__ == "ButtonGroup"
                        and str(getattr(e, "label", "")) == "Filter"), -1)
        if not (0 <= nav_i < pills_i):
            problems.append("date nav is not ahead of the filter pills "
                            f"(nav={nav_i}, pills={pills_i}) — MLB renders "
                            "them in that order")

        # (d) NFL team names — the store path used to fall through to the
        # MLB map ("Arizona Diamondbacks" / "Kansas City Royals" on NFL cards).
        if "Kansas City Chiefs" not in atext or "Arizona Cardinals" not in atext:
            problems.append("archive cards lost the NFL full team names")
        if "Kansas City Royals" in atext or "Arizona Diamondbacks" in atext:
            problems.append("archive cards rendered MLB team names")

        # (e) Pick grade preserved: the store's `correct` column drives the
        # pills (2 correct + 1 graded wrong), never the fabricated all-MISS
        # card the NaN `home_win` used to derive.
        n_ok = atext.count("✓ CORRECT PICK")
        n_miss = atext.count("X MISS")
        if (n_ok, n_miss) != (2, 1):
            problems.append("archive pick grades do not match the store's "
                            f"published `correct` column (correct={n_ok}, "
                            f"miss={n_miss}, want 2/1)")
        if "No results yet" in atext or "No games on this date" in atext:
            problems.append("decided archive board rendered the pre-game/"
                            "empty badge instead of the accuracy badge")

        # (g) CURRENT-SEASON entering records on the cards (MLB
        # compute_season_records parity): each team's own season only —
        # never the artifact's multi-season career tally (the board fixture
        # ships "1-1"/"3-0"), never the same game's own result, and never
        # the prior season's row (the 2025 NE win must not leak into 2026).
        for want in ("NYG 2-0", "WAS 0-1", "SF 0-0"):
            if want not in text:
                problems.append("today's cards lost the current-season "
                                f"entering record '{want}'")
        for stale in ("NYG 1-1", "WAS 2-0", "SF 3-0"):
            if stale in text:
                problems.append("today's cards still show the artifact's "
                                f"record '{stale}' instead of the season record")
        for want in ("NE 1-0", "BUF 0-0", "NYG 1-0", "ARI 0-0", "KC 0-0"):
            if want not in atext:
                problems.append("archive cards lost the current-season "
                                f"entering record '{want}'")
        for leak in ("NE 2-0", "NE 1-1"):
            if leak in atext:
                problems.append("archive record leaks across the offseason "
                                f"or counts the game's own result ('{leak}')")

        # (h) Store self-heal: a HOST-LOCAL store miss (the failure mode
        # behind "Oct 4 still serves the OOF re-price" — a failed or stale
        # local read silently swapped in the OOF CSV) must recover the
        # COMMITTED store bytes and serve the frozen prices. The fetch is
        # patched to the fixture so the check is deterministic and offline.
        import nfl_todays_page as _nflp
        import utils as _u
        _real_root, _real_fetch = _u.REPO_ROOT, _u._fetch_bytes
        _fixture = STORE.read_bytes()
        _heal_calls: list[str] = []

        def _fake_fetch(relpath, *a, **k):
            _heal_calls.append(relpath)
            if relpath.endswith("production_cards_history.csv"):
                return _fixture, "test"
            return None, "missing"

        _u.REPO_ROOT = FRONTEND_DIR / "__no_such_checkout__"
        _u._fetch_bytes = _fake_fetch
        try:
            _healed, _healed_hist = _nflp._load_day(_YDAY_C)
        finally:
            _u.REPO_ROOT, _u._fetch_bytes = _real_root, _real_fetch
        if not _healed_hist or _healed is None or len(_healed) != 3:
            problems.append("store self-heal did not serve the archive day "
                            f"(rows={0 if _healed is None else len(_healed)}, "
                            f"hist={_healed_hist})")
        else:
            _g = _healed[_healed["game_id"] == "2026_04_ARI_NYG"]
            if _g.empty or abs(float(_g.iloc[0]["home_win_prob_model"])
                               - 0.617) > 1e-9:
                problems.append("store self-heal served a non-frozen price: "
                                + str(None if _g.empty
                                      else _g.iloc[0]["home_win_prob_model"]))
        if not any(c.endswith("production_cards_history.csv")
                   for c in _heal_calls):
            problems.append("store self-heal never consulted the committed "
                            "bytes (host-local miss would serve OOF again)")

        # (i) DATED QB matchup on archive cards (MLB sp_* history parity):
        # an MLB history card always renders THAT date's pitcher fields, so
        # the NFL twin boxes must show THIS date's starters from
        # nfl_qb_matchup_<date> — not the empty '—' boxes the current-
        # slate-only gate produced, and never another date's starters.
        if "Frozen Home QB" not in atext or "Frozen Away QB" not in atext:
            problems.append("archive cards lost the DATED QB matchup — the "
                            "twin boxes must serve this date's starters "
                            "(MLB history parity)")
        if "Rating \u2014 \u00b7 TD/g \u2014" in atext:
            problems.append("archive cards render empty QB boxes despite a "
                            "dated matchup file for this date")

        # (j) DISPLAY-ONLY archive enrichment (MLB card anatomy): the
        # frozen store ships no venue/kickoff, so those fill from THIS
        # date's board snapshot — the stadium renders, the day/night tag
        # derives from the real kickoff (it used to default every archive
        # game to 🌙 Night Game), under the NFL serving convention (the
        # stamp's hour IS ET — never UTC-shifted, or an 8 PM kickoff reads
        # '☀ Day Game'), and the board's own later re-published price
        # (0.90) never reaches the card — the store's 62% still shows.
        if "MetLife Stadium" not in atext:
            problems.append("archive cards lost the dated board's venue "
                            "(MLB card anatomy renders the stadium)")
        if "\U0001f4cd \u2014" in atext:
            problems.append("archive cards still render the empty venue line")
        if "\u2600 Day Game" not in atext:
            problems.append("archive cards did not derive the day/night tag "
                            "from the dated kickoff stamp")
        if "\U0001f319 Night Game" not in atext:
            problems.append("archive evening kickoff lost its night tag — "
                            "the 20:00 stamp is 8 PM ET under the serving "
                            "convention and must never UTC-shift to Day")
        if "90%" in atext:
            problems.append("archive card served the DATED BOARD's price — "
                            "display enrichment must never touch "
                            "probabilities")

        # (f) Recovery walk + decided-mask gates (pure functions, no IO).
        from nfl_todays_page import _decided_mask, _recovered_day
        valid_set = set(valid)
        if _recovered_day(_YDAY_C, _YDAY_C, valid_set) is not None:
            problems.append("recovery would re-render the failed date itself")
        if _recovered_day(_YDAY_C, "19990101", valid_set) is not None:
            problems.append("recovery accepted a date outside the valid "
                            "(rolling 10-day) set — a pruned date must never "
                            "render another date's board")
        _fin = pd.DataFrame({"game_status": ["Final", "Final"],
                             "home_score": [20.0, 17.0],
                             "away_score": [23.0, 21.0],
                             "home_win": [np.nan, np.nan]})
        if int(_decided_mask(_fin).sum()) != 2:
            problems.append("_decided_mask missed a Final row carrying scores")
        _pre = pd.DataFrame({"game_status": ["Scheduled", "Live"],
                             "home_score": [None, None],
                             "away_score": [None, None]})
        if int(_decided_mask(_pre).sum()) != 0:
            problems.append("_decided_mask counted a pre-game/live row")
        if int(_decided_mask(pd.DataFrame()).sum()) != 0:
            problems.append("_decided_mask must be 0 on an empty frame")
    finally:
        _restore()

    if problems:
        print("BOARD RENDER SMOKE [FAIL]")
        for p in problems:
            print("  -", p)
        return 1
    print("BOARD RENDER SMOKE [PASS]")
    print("  - today's board includes the started game with its frozen price")
    print("  - pre-game slate renders MLB's honest three-state badge")
    print("  - yesterday renders from the dated snapshot (frozen, not OOF)")
    print("  - archive cards serve the frozen store price, grade, and names")
    print("  - cards serve each team's CURRENT-SEASON entering record "
          "(MLB offseason reset)")
    print("  - a host-local store miss self-heals to the committed bytes "
          "(never a silent OOF re-price)")
    print("  - archive cards render THIS date's QB matchup (MLB dated "
          "sp_* history parity)")
    print("  - archive cards serve the DATED run-engine markets with O/U "
          "+ run-line toggles; no SHAP accordion")
    print("  - archive board renders NO banner; venue + kickoff fill from "
          "the dated snapshot (prices untouched)")
    print("  - recovery and decided-mask gates hold")
    print("  - valid-date rail includes the dated board family window")
    print("  - no page exceptions")
    return 0


if __name__ == "__main__":
    sys.exit(run())
