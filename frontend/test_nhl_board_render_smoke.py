"""NHL board render smoke — AppTest over todays_games.py (sport=nhl).

Structural twin of ``test_board_render_smoke.py`` (the NFL mirror), pinning
the NHL Today's Games remediation:

  * TODAY's board renders the current-slate record with MLB's three-state
    badge (a pre-game slate never claims an accuracy record), full NHL team
    names (the card store has no ``*_team_name`` columns — a fall-through to
    MLB_TEAM_NAMES rendered "Philadelphia Phillies" for PHI) and each team's
    CURRENT-SEASON entering record instead of the artifact's multi-season
    career tally.
  * YESTERDAY's archive board serves THIS date's PUBLISHED slate rows from
    the ``nhl_moneyline_v1_*`` family (never the frozen store's price, never
    an OOF re-price), with the settled result overlaid display-only from
    the frozen card store.
  * The archive card carries the DATED run-engine distribution (kind='slate'
    rows of this date's markets artifact — never the file's kind='oof'
    decoy rows) with per-card O/U + run-line toggles on FINISHED games, the
    DATED goalie boxes (season-line era only) and NO SHAP accordion.
  * No archive banner; date nav ahead of the filter pills; recovery and
    decided-mask gates hold; the store self-heals through the committed
    bytes when the host-local read misses.

Artifacts are shadowed (backup -> fixture -> restore) exactly like the NFL
harness, so committed bytes are untouched after the run.
Run:  python3 -m test_nhl_board_render_smoke     (from frontend/)
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
NHL_DD = REPO_ROOT / "nhl-backend" / "data_delivery"

# Real today/yesterday (ET) so the page's rolling-10-day valid-date window
# includes them; every shadowed file is backed up and restored after.
_TODAY = datetime.now(ZoneInfo("America/New_York")).date()
_YDAY = _TODAY - timedelta(days=1)
_TODAY_C = _TODAY.strftime("%Y%m%d")
_YDAY_C = _YDAY.strftime("%Y%m%d")
ML_TODAY = NHL_DD / f"nhl_moneyline_v1_{_TODAY_C}.json"
ML_YDAY = NHL_DD / f"nhl_moneyline_v1_{_YDAY_C}.json"
CAL_TODAY = NHL_DD / f"nhl_calibration_{_TODAY_C}.json"
CAL_YDAY = NHL_DD / f"nhl_calibration_{_YDAY_C}.json"
STORE = NHL_DD / "nhl_production_cards_history.csv"
GOALIE_YDAY = NHL_DD / f"nhl_goalie_matchup_{_YDAY_C}.json"
RE_CSV = NHL_DD / f"nhl_run_engine_markets_{_YDAY_C}.csv"

# Era gate for the goalie family (utils._GOALIE_SEASON_LINE_ERA): records
# dated before it carry the retired career-window lookback and are never
# served. The fixture rides YDAY, so the goalie assertions hold whenever
# the suite runs at/after 2026-10-05 and degrade to a skip otherwise.
_GOALIE_ERA = "20261004"
_GOALIE_OK = _YDAY_C >= _GOALIE_ERA

_BACKUPS: dict[Path, bytes | None] = {}
WRITTEN: list[Path] = []

# Fixture prices — the PUBLISHED values the archive board must render, next
# to the frozen store's deliberately different decoy prices (78/67/33%) a
# serve-the-store/OOF regression would surface as the card probability.
FIN_PUB = {"SMOKE_NHL_FIN1": 0.620, "SMOKE_NHL_FIN2": 0.412,
           "SMOKE_NHL_FIN3": 0.710}
STORE_DECOY = {"SMOKE_NHL_FIN1": 0.777, "SMOKE_NHL_FIN2": 0.666,
               "SMOKE_NHL_FIN3": 0.333}


def _iso(day, et_hour: int) -> str:
    """Start stamp the serving convention emits (date + ET time as-is)."""
    return f"{day.isoformat()}T{et_hour:02d}:00:00Z"


def _calibration_record() -> dict:
    return {
        "date": _TODAY_C, "trained_at": "2026-01-01T00:00:00Z",
        "n_games": 3, "created_utc": "2026-01-01T00:00:00Z",
        "config": {}, "league_total": 3, "evening_games_league": 1,
        "today_record": {"wins": 2, "losses": 1, "completed": 3},
        "metrics": {"auc": 0.69, "brier": 0.22, "logloss": 0.63, "ece": 0.02},
        "calibration": {}, "distribution_calibration": {},
        "calibration_buckets": [], "daily": [],
    }


def _moneyline_today() -> dict:
    """Current-slate record: two pre-game games for TODAY.

    The career-tally records (the screenshot's TBL "102-76") and the
    deliberately wrong artifact team name ("Philadelphia Phillies" for PHI)
    are the regressions the card must NOT render: the season-record attach
    and the NHL display-name map both win over the record's own fields.
    """
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"feature_set": "smoke"},
        "slate_date": _TODAY.isoformat(),
        "n_games": 2,
        "games": [
            {
                "game_id": "SMOKE_NHL_T1", "game_date": _TODAY.isoformat(),
                "start_time_utc": _iso(_TODAY, 13),
                "home_team": "TBL", "away_team": "PHI",
                "home_team_name": "Tampa Bay Baseball Club",
                "away_team_name": "Philadelphia Phillies",
                "home_record": "102-76-5", "away_record": "88-79-3",
                "venue": "Amalie Arena", "game_status": "pre",
                "home_score": None, "away_score": None,
                "home_win_prob_model": 0.575, "away_win_prob_model": 0.425,
                "model_pick": "TBL", "model_correct": None,
            },
            {
                "game_id": "SMOKE_NHL_T2", "game_date": _TODAY.isoformat(),
                "start_time_utc": _iso(_TODAY, 20),
                "home_team": "COL", "away_team": "DAL",
                "home_team_name": "Colorado Avalanche",
                "away_team_name": "Dallas Stars",
                "home_record": "45-30-7", "away_record": "40-33-9",
                "venue": "Ball Arena", "game_status": "pre",
                "home_score": None, "away_score": None,
                "home_win_prob_model": 0.640, "away_win_prob_model": 0.360,
                "model_pick": "COL", "model_correct": None,
            },
        ],
    }


def _moneyline_yday() -> dict:
    """YDAY's PUBLISHED slate — the archive board's price source.

    Records freeze at ``pre`` (the serving layer's contract); the settled
    result overlays display-only from the frozen store. PHI's artifact name
    is the wrong MLB string on purpose: the NHL display-name map must win.
    """
    def _g(gid, home, away, ph, pick, venue, et_hour, wrong_away_name=None):
        return {
            "game_id": gid, "game_date": _YDAY.isoformat(),
            "start_time_utc": _iso(_YDAY, et_hour),
            "home_team": home, "away_team": away,
            "home_team_name": "", "away_team_name": wrong_away_name or "",
            "home_record": "0-0", "away_record": "0-0",
            "venue": venue, "game_status": "pre",
            "home_score": None, "away_score": None,
            "home_win_prob_model": ph, "away_win_prob_model": round(1 - ph, 6),
            "model_pick": pick, "model_correct": None,
        }

    games = [
        _g("SMOKE_NHL_FIN1", "TBL", "PHI", 0.620, "TBL",
           "Amalie Arena", 19, wrong_away_name="Philadelphia Phillies"),
        _g("SMOKE_NHL_FIN2", "OTT", "BOS", 0.412, "BOS",
           "Canadian Tire Centre", 19),
        _g("SMOKE_NHL_FIN3", "COL", "STL", 0.710, "COL",
           "Ball Arena", 21),
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"feature_set": "smoke"},
        "slate_date": _YDAY.isoformat(),
        "n_games": len(games),
        "games": games,
    }


_STORE_COLS = [
    "game_id", "game_date", "home_team", "away_team", "home_score",
    "away_score", "home_win", "home_win_prob_model",
    "home_win_prob_model_calibrated", "model_pick", "actual_winner",
    "correct", "source_artifact_date", "p_home_win", "p_away_win",
    "game_status",
]


def _row(gid, gdate, home, away, hs, as_, p_home, pick, actual, status,
         source):
    """One frozen store row (the committed CSV's 16-column schema)."""
    decided = hs is not None and as_ is not None
    if decided:
        winner = home if hs > as_ else (away if as_ > hs else "TIE")
        home_win = 1.0 if hs > as_ else (0.0 if as_ > hs else 0.5)
    else:
        winner, home_win = "", np.nan
    return {
        "game_id": gid, "game_date": gdate,
        "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "home_win": home_win,
        "home_win_prob_model": np.nan,
        "home_win_prob_model_calibrated": np.nan,
        "model_pick": pick, "actual_winner": actual or winner,
        "correct": (pick == (actual or winner)) if decided else False,
        "source_artifact_date": source,
        "p_home_win": p_home, "p_away_win": round(1.0 - float(p_home), 6),
        "game_status": status,
    }


def _store_rows() -> pd.DataFrame:
    """Frozen card-store fixture.

    Shaped to pin the archive contract:

      * YDAY's three finished rows carry the store's DECOY prices (77.7 /
        66.6 / 33.3%) — a serve-the-store/OOF regression fails the "62%"
        assertion below (and the ``78%`` absence pin);
      * two earlier 2026-season rows give TBL/BOS their entering records,
        and a 2025-season TBL row must never leak into them (the offseason
        reset: archive TBL is 1-0, never 2-0);
      * grades: FIN1 + FIN3 correct, FIN2 a MISS (2 ✓ / 1 ✗).
    """
    rows = [
        _row("SMOKE_PRIOR_A", "2026-10-01", "TBL", "NYR", 4, 1,
             0.600, "TBL", "TBL", "Final", "20261001"),
        _row("SMOKE_PRIOR_B", "2026-10-02", "MTL", "BOS", 1, 3,
             0.430, "MTL", "BOS", "Final", "20261002"),
        # Prior-SEASON row (Dec 2025 -> season 2025 under the July boundary):
        _row("SMOKE_PRIOR_C", "2025-12-15", "TBL", "BUF", 5, 2,
             0.550, "TBL", "TBL", "Final", "20251215"),
        _row("SMOKE_NHL_FIN1", _YDAY.isoformat(), "TBL", "PHI", 4, 1,
             STORE_DECOY["SMOKE_NHL_FIN1"], "TBL", "TBL", "Final",
             _YDAY_C),
        _row("SMOKE_NHL_FIN2", _YDAY.isoformat(), "OTT", "BOS", 3, 1,
             STORE_DECOY["SMOKE_NHL_FIN2"], "BOS", "OTT", "Final",
             _YDAY_C),
        _row("SMOKE_NHL_FIN3", _YDAY.isoformat(), "COL", "STL", 5, 2,
             STORE_DECOY["SMOKE_NHL_FIN3"], "COL", "COL", "Final",
             _YDAY_C),
    ]
    return pd.DataFrame(rows, columns=_STORE_COLS)


def _run_engine_rows() -> pd.DataFrame:
    """YDAY's dated run-engine markets artifact — the archive card's
    distribution-model source (MLB ``_build_slate_map`` parity).

    * ``kind='oof'`` decoy rows FIRST (the committed files sort them first)
      carrying fair_total 99 / Proj 99.0 — the card must serve the
      ``kind='slate'`` rows' published distribution, never the OOF re-price
      an unfiltered first-hit match would surface;
    * slate rows for the three finished games with the COMPLETE NHL MC grid
      (spreads −8..+8 with ``p_push_{label}``, totals 5/6/7 with
      ``p_push_total_{U}`` — ``_run_engine_grid_usable``'s NHL contract)
      and per-game fair lines: totals 6 / 5 / 7 (the O/U toggle defaults)
      and |fair spread| 1.5 / 1.5 / 2.4 (the run-line toggles land on ±2.0).
    """
    grid: dict = {}
    for line in range(-8, 9):
        label = f"m{-line}" if line < 0 else str(line)
        grid[f"p_home_cover_{label}"] = 0.52
        grid[f"p_push_{label}"] = 0.04 if line else 0.02
    for u in (5, 6, 7):
        grid[f"p_over_{u}"] = 0.48
        grid[f"p_under_{u}"] = 0.48
        grid[f"p_push_total_{u}"] = 0.04
    games = [
        # gid, home, away, mu_h, mu_a, fair_total, fair_spread, p_home
        ("SMOKE_NHL_FIN1", "TBL", "PHI", 3.4, 2.2, 6.0, 1.5, 0.620),
        ("SMOKE_NHL_FIN2", "OTT", "BOS", 2.6, 3.1, 5.0, -1.5, 0.412),
        ("SMOKE_NHL_FIN3", "COL", "STL", 3.8, 2.4, 7.0, 2.4, 0.710),
    ]
    base = {
        "gameday": _YDAY.isoformat(), "season": 2026,
        "p_home_win_derived": 0.50, "p_away_win_derived": 0.50,
        **grid,
    }
    oof = [dict(base, kind="oof", game_id=gid, home_team=h, away_team=a,
                p_home_win=ph, p_away_win=round(1 - ph, 6),
                mu_h=99.0, mu_a=99.0, fair_total=99.0, fair_spread=9.0)
           for gid, h, a, mh, ma, ft, fs, ph in games]
    slate = [dict(base, kind="slate", game_id=gid, home_team=h, away_team=a,
                  p_home_win=ph, p_away_win=round(1 - ph, 6),
                  mu_h=mh, mu_a=ma, fair_total=ft, fair_spread=fs)
             for gid, h, a, mh, ma, ft, fs, ph in games]
    return pd.DataFrame(oof + slate)


def _goalie_record() -> dict:
    """YDAY's dated goalie matchup (flat emitter schema) — the archive
    card's twin of MLB's dated sp_* fields, season-line era only."""
    games = []
    for i, (gid, home, away) in enumerate((
            ("SMOKE_NHL_FIN1", "TBL", "PHI"),
            ("SMOKE_NHL_FIN2", "OTT", "BOS"),
            ("SMOKE_NHL_FIN3", "COL", "STL")), start=1):
        games.append({
            "game_id": gid, "gameday": _YDAY.isoformat(),
            "home_team": home, "away_team": away,
            "g_home_name": f"Frozen Home Goalie {i}",
            "g_home_sv_pct": 0.921, "g_home_gaa": 2.14,
            "g_home_starts": 3.0,
            "g_away_name": f"Frozen Away Goalie {i}",
            "g_away_sv_pct": 0.913, "g_away_gaa": 2.45,
            "g_away_starts": 2.0,
        })
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "n_games": len(games), "games": games}


def _write_artifacts() -> None:
    NHL_DD.mkdir(parents=True, exist_ok=True)
    for p in (ML_TODAY, ML_YDAY, CAL_TODAY, CAL_YDAY, STORE,
              GOALIE_YDAY, RE_CSV):
        _BACKUPS[p] = p.read_bytes() if p.exists() else None
    ML_TODAY.write_text(json.dumps(_moneyline_today()))
    ML_YDAY.write_text(json.dumps(_moneyline_yday()))
    CAL_TODAY.write_text(json.dumps(_calibration_record()))
    CAL_YDAY.write_text(json.dumps(_calibration_record()))
    _store_rows().to_csv(STORE, index=False)
    GOALIE_YDAY.write_text(json.dumps(_goalie_record()))
    _run_engine_rows().to_csv(RE_CSV, index=False)
    WRITTEN.extend([ML_TODAY, ML_YDAY, CAL_TODAY, CAL_YDAY, STORE,
                    GOALIE_YDAY, RE_CSV])


def _restore() -> None:
    for p, prev in _BACKUPS.items():
        try:
            if prev is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(prev)
        except FileNotFoundError:
            pass
    _BACKUPS.clear()
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
    # Windows consoles default to cp1252; the success summary prints ✓/✗
    # (and would crash AFTER every assertion passed without this).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    problems: list[str] = []
    _write_artifacts()
    try:
        # ---- Today's board: current slate, honest badge, season records ----
        at = AppTest.from_file(str(FRONTEND_DIR / "todays_games.py"),
                               default_timeout=120)
        at.session_state["sport"] = "nhl"
        at.run()
        if at.exception:
            problems.append("NHL BOARD RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in at.exception))
        text = _all_text(at)
        if "SMOKE_NHL_T1" not in text and "TBL" not in text \
                and "Tampa Bay" not in text:
            problems.append("today's board lost the current-slate game cards")
        if "2 of 3 games shown" not in text:
            problems.append("header strip does not count 2 of 3 games "
                            "(current record not feeding the board frame?)")
        # MLB's THREE-STATE badge: both shadow cards are pre-game, so the
        # accuracy pill must not claim a record.
        if "No results yet — pre-game slate" not in text:
            problems.append("pre-game slate still claims an accuracy badge "
                            "(NHL three-state badge not applied)")
        # NHL display names: the record's wrong MLB name must never render.
        if "Philadelphia Flyers" not in text:
            problems.append("today's cards lost the NHL full team name for PHI")
        if "Philadelphia Phillies" in text or "Tampa Bay Baseball Club" in text:
            problems.append("today's cards rendered the artifact's WRONG team "
                            "name (NHL display-name map not applied)")
        # CURRENT-SEASON entering records replace the career tally.
        for want in ("TBL 2-0", "PHI 0-1", "COL 1-0", "DAL 0-0"):
            if want not in text:
                problems.append(f"today's cards lost the season record '{want}'")
        for stale in ("102-76-5", "88-79-3", "45-30-7", "40-33-9"):
            if stale in text:
                problems.append(f"today's cards still show the artifact's "
                                f"multi-season tally '{stale}'")
        if "57%" not in text or "64%" not in text:
            problems.append("today's cards lost the current-slate probabilities")

        # ---- Archive date: published slate, dated enrichment, no banner ----
        import utils
        valid = utils.valid_dates("nhl")
        for _d in (_YDAY_C, _TODAY_C):
            if _d not in set(valid):
                problems.append(
                    f"{_d} is not in the NHL valid-date rail "
                    f"(dated family not feeding valid_dates); head: "
                    f"{list(valid)[:4]}")

        aa = AppTest.from_file(str(FRONTEND_DIR / "todays_games.py"),
                               default_timeout=120)
        aa.session_state["sport"] = "nhl"
        aa.session_state["selected_date"] = _YDAY_C
        aa.session_state["_nav_sport"] = "nhl"
        aa.run()
        if aa.exception:
            problems.append("NHL ARCHIVE BOARD RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in aa.exception))
        atext = _all_text(aa)

        # (a) THE PUBLISHED SLATE PRICE, never the store/OOF re-price: the
        # frozen store fixture prices these ids at 77.7 / 66.6 / 33.3%, so
        # any serve-the-store regression fails both directions of this pin.
        for want in ("62%", "41%", "71%"):
            if want not in atext:
                problems.append(
                    f"archive board lost the published price {want} "
                    "(serving the frozen store / OOF history instead?)")
        for decoy in ("78%", "67%", "33%"):
            if decoy in atext:
                problems.append(
                    f"archive board rendered the store's decoy price {decoy} "
                    "— the published slate must win")

        # (b) THE DATE'S OWN RUN-ENGINE DISTRIBUTION with per-card toggles
        # on FINISHED games: Proj / O/U / RL from this date's kind='slate'
        # rows — never the quiet unavailable fallback, never the file's
        # kind='oof' decoy rows (fair_total 99 / Proj 99.0).
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
        if "Proj: PHI 2.2 \u2013 TBL 3.4" not in atext:
            problems.append("archive cards did not serve the DATED slate "
                            "row's projection (wrong artifact/row served)")
        # The push note rides its own <span class="re-na"> — assert the
        # quoted pair and the note independently (raw markdown text).
        if "O/U 6: Over 50% / Under 50%" not in atext:
            problems.append("archive card totals did not price the dated "
                            "fair total 6 from the MC grid")
        if "(4% push)" not in atext:
            problems.append("archive card totals lost the push note at the "
                            "integer fair line")
        if "99.0" in atext or "O/U 99" in atext:
            problems.append("archive card served the kind='oof' decoy row "
                            "(the file's OOF re-price leaked onto a card)")
        if "RL: TBL \u22122 54%" not in atext or "PHI +2 46%" not in atext:
            problems.append("archive card did not price the dated run line "
                            "at the fair ±2.0 default")
        if len(aa.selectbox) != 6:
            problems.append(f"archive board rendered {len(aa.selectbox)} "
                            "O/U + run-line toggles, want 6 (one pair per "
                            "card)")
        else:
            _toggles = sorted(float(sb.value) for sb in aa.selectbox)
            if _toggles != [2.0, 2.0, 2.0, 5.0, 6.0, 7.0]:
                problems.append("archive O/U + run-line toggle defaults "
                                f"wrong: {_toggles} (want totals 6/5/7 and "
                                "rounded spreads ±2/±2/±2 — an oof leak "
                                "would show 99)")
        # SHAP stays CURRENT-SLATE ONLY.
        if len(aa.expander):
            problems.append(
                f"archive board rendered {len(aa.expander)} SHAP accordion(s) "
                "— a frozen card must not inherit a later run's attributions")

        # (c) NO archive banner; date nav ahead of the filter pills (MLB's
        # rendered flow carries no page-level archive notice).
        seq = list(aa.main)
        banners = [e for e in seq
                   if type(e).__name__ == "Info"
                   and "Archive" in str(getattr(e, "value", ""))]
        if banners:
            problems.append("archive banner still rendered between the date "
                            "nav and the filter pills")
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

        # (d) THREE-STATE badge on the DECIDED archive board: results exist,
        # so the accuracy record shows and the pre-game wording must not.
        if "Today ·" not in atext or "accuracy" not in atext:
            problems.append("decided archive board did not render the "
                            "accuracy badge")
        if "No results yet — pre-game slate" in atext \
                or "No games on this date" in atext:
            problems.append("decided archive board rendered the pre-game/"
                            "empty badge instead of the accuracy badge")
        if "3 of 3 games shown" not in atext:
            problems.append("archive header strip does not count 3 of 3 games")

        # (e) Pick grades: the published pick against the overlaid result —
        # 2 correct (FIN1/FIN3) + 1 MISS (FIN2), never fabricated.
        n_ok = atext.count("✓ CORRECT PICK")
        n_miss = atext.count("X MISS")
        if (n_ok, n_miss) != (2, 1):
            problems.append("archive pick grades do not match the published "
                            f"picks (correct={n_ok}, miss={n_miss}, want 2/1)")

        # (f) NHL team names — the store path used to fall through to the
        # MLB map ("Philadelphia Phillies" on a PHI card).
        if "Philadelphia Flyers" not in atext or "Tampa Bay Lightning" not in atext:
            problems.append("archive cards lost the NHL full team names")
        if "Philadelphia Phillies" in atext:
            problems.append("archive cards rendered the MLB team name for PHI")

        # (g) CURRENT-SEASON entering records on the archive cards, strictly
        # prior and offseason-reset: TBL entered YDAY at 1-0 (its 2025 row
        # must not leak, and YDAY's own result must not count), BOS at 1-0,
        # PHI/OTT/COL/STL at 0-0.
        for want in ("TBL 1-0", "PHI 0-0", "BOS 1-0", "OTT 0-0",
                     "COL 0-0", "STL 0-0"):
            if want not in atext:
                problems.append("archive cards lost the current-season "
                                f"entering record '{want}'")
        if "TBL 2-0" in atext:
            problems.append("archive record leaks across the offseason or "
                            "counts the game's own result ('TBL 2-0')")
        if "102-76" in atext:
            problems.append("archive cards rendered a multi-season career "
                            "tally")

        # (h) DATED goalie boxes on archive cards (era-valid fixture):
        # THIS date's starters, never empty '—' boxes.
        if _GOALIE_OK:
            if "Frozen Home Goalie" not in atext or "Frozen Away Goalie" not in atext:
                problems.append("archive cards lost the DATED goalie matchup — "
                                "the twin boxes must serve this date's "
                                "starters (MLB dated sp_* history parity)")
            if "SV% \u2014 \u00b7 GAA \u2014" in atext:
                problems.append("archive cards render empty goalie boxes "
                                "despite a dated matchup file for this date")

        # (i) Loader-level pins (no AppTest): published-slate resolution,
        # dated markets/goalie contracts, name map, valid rail.
        day = utils.load_nhl_board_games(_YDAY_C)
        if len(day) != 3:
            problems.append(f"load_nhl_board_games({_YDAY_C}) returned "
                            f"{len(day)} rows, want 3")
        else:
            f1 = day[day["game_id"] == "SMOKE_NHL_FIN1"].iloc[0]
            if abs(float(f1["home_win_prob_model"]) - 0.620) > 1e-9:
                problems.append("published-slate loader served a non-published "
                                f"price: {f1['home_win_prob_model']}")
            if str(f1["game_status"]) != "Final" \
                    or pd.isna(f1["home_score"]) or int(f1["home_score"]) != 4:
                problems.append("published rows did not overlay the store's "
                                "settled result display-only")
            if f1["home_team_name"] != "Tampa Bay Lightning" \
                    or f1["away_team_name"] != "Philadelphia Flyers":
                problems.append(f"NHL display names wrong on the archive "
                                f"frame: {f1['home_team_name']!r} / "
                                f"{f1['away_team_name']!r}")
            if f1["home_record"] != "1-0" or f1["away_record"] != "0-0":
                problems.append("archive season records wrong: "
                                f"{f1['home_record']} / {f1['away_record']}")
        cur = utils.load_nhl_moneyline("nhl")
        if len(cur):
            t1 = cur[cur["game_id"] == "SMOKE_NHL_T1"]
            if not t1.empty:
                t1 = t1.iloc[0]
                if t1["home_record"] != "2-0" or t1["away_record"] != "0-1":
                    problems.append("today's season records wrong: "
                                    f"{t1['home_record']} / {t1['away_record']}")
                if t1["away_team_name"] != "Philadelphia Flyers":
                    problems.append("current-slate record's wrong MLB name "
                                    "reached the frame")
        mframe, mdate = utils.load_nhl_run_engine_markets(
            "nhl", date_str=_YDAY_C)
        if mframe.empty or mdate != _YDAY_C:
            problems.append(f"dated markets load missed (rows={len(mframe)}, "
                            f"date={mdate})")
        else:
            if set(mframe["kind"].astype(str)) != {"slate"}:
                problems.append("dated markets frame carries non-slate rows "
                                "(OOF rows leaked past the kind filter)")
            if set(mframe["gameday"].astype(str).str[:10]) \
                    != {_YDAY.isoformat()}:
                problems.append("dated markets frame carries a foreign gameday")
            if 99.0 in set(pd.to_numeric(mframe["fair_total"])):
                problems.append("dated markets frame served the oof decoy rows")
        if _GOALIE_OK:
            grow = utils.load_nhl_goalie_matchup("nhl", date_str=_YDAY_C)
            if len(grow) != 3 or not (grow["g_home_name"].astype(str)
                                      .str.startswith("Frozen")).any():
                problems.append("dated goalie load missed the fixture rows "
                                f"(rows={len(grow)})")
        names = utils._team_name_map("nhl")
        if len(names) != 32 or names.get("PHI") != "Philadelphia Flyers":
            problems.append("NHL team-name map is not 32 teams / PHI wrong")
        if utils._team_name_map("mlb").get("PHI") != "Philadelphia Phillies":
            problems.append("MLB team-name map changed unexpectedly")

        # (j) Store self-heal: a HOST-LOCAL store miss must recover the
        # COMMITTED bytes (fixture here) and serve the store's frozen price —
        # never a silent OOF fallback. Fetch patched for determinism.
        import nhl_todays_page as _nhlp
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
            _healed, _healed_hist = _nhlp._load_day(_YDAY_C)
        finally:
            _u.REPO_ROOT, _u._fetch_bytes = _real_root, _real_fetch
        if not _healed_hist or _healed is None or len(_healed) != 3:
            problems.append("store self-heal did not serve the archive day "
                            f"(rows={0 if _healed is None else len(_healed)}, "
                            f"hist={_healed_hist})")
        else:
            _g = _healed[_healed["game_id"] == "SMOKE_NHL_FIN1"]
            if _g.empty or abs(float(_g.iloc[0]["home_win_prob_model"])
                               - STORE_DECOY["SMOKE_NHL_FIN1"]) > 1e-9:
                problems.append("store self-heal served a non-frozen price: "
                                + str(None if _g.empty
                                      else _g.iloc[0]["home_win_prob_model"]))
        if not any(c.endswith("production_cards_history.csv")
                   for c in _heal_calls):
            problems.append("store self-heal never consulted the committed "
                            "bytes (host-local miss would serve OOF again)")

        # (k) Recovery walk + decided-mask gates (pure functions, no IO).
        from nhl_todays_page import _decided_mask, _recovered_day
        valid_set = set(valid)
        if _recovered_day(_YDAY_C, _YDAY_C, valid_set) is not None:
            problems.append("recovery would re-render the failed date itself")
        if _recovered_day(_YDAY_C, "19990101", valid_set) is not None:
            problems.append("recovery accepted a date outside the valid "
                            "(rolling 10-day) set — a pruned date must never "
                            "render another date's board")
        _fin = pd.DataFrame({"game_status": ["Final", "Final"],
                             "home_score": [4.0, 3.0],
                             "away_score": [1.0, 1.0],
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
        print("NHL BOARD RENDER SMOKE [FAIL]")
        for p in problems:
            print("  -", p)
        return 1
    print("NHL BOARD RENDER SMOKE [PASS]")
    print("  - today's board: current slate, honest three-state badge, "
          "NHL full team names, season records (no career tally)")
    print("  - archive board serves the PUBLISHED slate price "
          "(never the store/OOF re-price)")
    print("  - archive cards serve the DATED run-engine markets with O/U + "
          "run-line toggles on FINISHED games; oof decoy rows never leak")
    print("  - archive cards render the DATED goalie boxes (season-line era)")
    print("  - archive cards grade 2 ✓ / 1 ✗ from the published pick")
    print("  - archive board renders NO banner; date nav ahead of the pills")
    print("  - season records strictly prior + offseason reset; no SHAP on "
          "archive")
    print("  - store self-heals to the committed bytes (never a silent "
          "OOF re-price)")
    print("  - recovery and decided-mask gates hold")
    print("  - no page exceptions")
    return 0


if __name__ == "__main__":
    sys.exit(run())
