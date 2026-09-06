"""Starting-QB matchup emitter — `nfl_qb_matchup_<date>.json`.

Structural mirror of the run-engine slate-serve emitter's per-game feeds, for
the Today's Games QB-matchup card (the NFL substitute for MLB's
starting-pitcher matchup card).

DECISION (task step 2, option (a)): no existing artifact carries per-game QB
matchup stats, so this small dated emitter writes one. It reuses ONLY the
already-pulled seams — ``nfl_features._load_raw`` (nflreadpy schedule+pbp),
the depth-chart QB1 resolvers (``_dc_qb1_weekly`` / ``_dc_qb1_snapshots``)
and ``_expected_starter_for`` — with zero new pull logic.

LEAKAGE DISCIPLINE (mirrors compose_tier5_qb_features): the starting QB is
the team's PUBLISHED QB1 before kickoff (weekly depth chart for the game's
(season, week), else the latest rolling snapshot strictly before kickoff);
2026 board rows have no published 2026 charts, so they resolve through the
latest pre-season snapshot — the honest expected-starter state. Per-QB stats
(passer rating, TD/game, cmp%, Y/A, INTs) aggregate the passer's DECIDED
games STRICTLY BEFORE the target game's kickoff — never the target game
itself, never future games. A QB with no prior starts renders every stat as
null (the card shows '—'), never fabricated.

Outputs one dated record per run:
    nfl-backend/data_delivery/nfl_qb_matchup_<YYYYMMDD>.json
{
  "record": "nfl_qb_matchup",
  "target_date": "YYYYMMDD",
  "written_utc": "...",
  "frame_sha256": "...",            # the slate-serve frame pin (same source)
  "computed_through": "YYYY-MM-DD", # newest decided gameday aggregated
  "games": [ {"game_id", "gameday", "home_team", "away_team",
              "qb_home": {"name","gsis_id","passer_rating","td_per_game",
                          "cmp_pct","yards_per_attempt","ints","starts"},
              "qb_away": {...}} , ... ]
}
Pure stdlib + pandas — no streamlit. The harness entrypoint is ``main``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_features as NF  # noqa: E402  ( seams: _load_raw, depth charts )
from run_nfl_slate import CANONICAL_FRAME_SHA, DATA_DELIVERY, SLATE_SEASON  # noqa: E402

RECORD = "nfl_qb_matchup"


# ---------------------------------------------------------------------------
# Depth-chart loading (the harness seam the tier-5 ablation uses; kept pure —
# network access only inside this loader, never at render time)
# ---------------------------------------------------------------------------

def load_depth_charts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(weekly_charts, snapshot_charts) as pandas frames, empty when absent.

    The 2026 feed is the dated rolling-snapshot shape (``dt`` + ``team`` +
    ``player_name``); prior seasons keep the weekly-chart shape."""
    try:
        import nflreadpy
        try:
            weekly = nflreadpy.load_depth_charts(
                seasons=list(range(2001, SLATE_SEASON))).to_pandas()
        except Exception:
            weekly = pd.DataFrame()
        try:
            snaps = nflreadpy.load_depth_charts(
                seasons=[SLATE_SEASON]).to_pandas()
        except Exception:
            snaps = pd.DataFrame()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return weekly, snaps


def _resolve_qb_name(gsis_id: str, roster: pd.DataFrame) -> str:
    """Display name for a gsis_id from the depth charts (pure lookup —
    the charts carry the player name column; empty when unknown)."""
    if not gsis_id or roster is None or not len(roster):
        return ""
    hit = roster[roster["gsis_id"].astype(str) == str(gsis_id)]
    if not len(hit):
        return ""
    for col in ("display_name", "full_name", "player_name"):
        if col in hit.columns:
            v = hit.iloc[0][col]
            s = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
            if s:
                return s
    return ""


def _snapshot_name(gsis_id: str, frame: pd.DataFrame) -> str:
    """Player name straight off a depth-chart frame's own name column."""
    if not gsis_id or frame is None or not len(frame) or "player_name" not in frame.columns:
        return ""
    hit = frame[frame["gsis_id"].astype(str) == str(gsis_id)]
    if not len(hit):
        return ""
    v = hit.iloc[0]["player_name"]
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def _expected_starter(team: str, season: int, week, kickoff_utc,
                      weekly_map: dict, snap_map: dict) -> str | None:
    """The team's published QB1 before kickoff (gsis_id) — the same seam the
    tier-5 feature composer uses (``_expected_starter_for``)."""
    return NF._expected_starter_for(team, season, week, kickoff_utc,
                                    weekly_map, snap_map)


# ---------------------------------------------------------------------------
# Per-QB strictly-prior aggregates from the decided schedule + pbp
# ---------------------------------------------------------------------------

def load_pbp_wide(seasons: list[int]) -> pd.DataFrame:
    """Play-by-play with the passer-identity + passing-outcome columns the
    passer table needs (``_load_raw`` keeps a narrow team-level subset)."""
    import nflreadpy
    pbp = nflreadpy.load_pbp(seasons)
    need = ["game_id", "passer_id", "pass_attempt", "complete_pass",
            "passing_yards", "pass_touchdown", "interception"]
    keep = [c for c in need if c in pbp.columns]
    return pbp.select(keep).to_pandas()


def _passer_game_table(sched: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, passer_id) decided start: attempts, completions,
    yards, TDs, INTs and the NFL passer rating — from the play-by-play.

    Passer rating (the classic NFL formula) is computed per play-batch:
        a = (cmp/att - 0.3) * 5, b = (yds/att - 3) * 0.25,
        c = (td/att) * 20, d = 2.375 - (int/att) * 25
        rating = 100/6 * sum(clamp(x, 0, 2.375))
    Aggregated as TOTALS over the window (never an average of per-game
    ratings), which is the league's own season-rating convention."""
    need = {"passer_id", "pass_attempt", "complete_pass", "passing_yards",
            "pass_touchdown", "interception"}
    if pbp is None or not need.issubset(pbp.columns) or not len(pbp):
        return pd.DataFrame()
    num_cols = need - {"passer_id"}
    p = pbp.copy()
    for c in num_cols:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[p["passer_id"].notna() & (p["passer_id"].astype(str) != "")
          & (p["passer_id"].astype(str).str.lower() != "nan")]
    p = p[p["pass_attempt"] == 1]
    if p.empty:
        return pd.DataFrame()
    g = p.groupby(["game_id", "passer_id"], dropna=False).agg(
        att=("pass_attempt", "sum"),
        cmp=("complete_pass", "sum"),
        yds=("passing_yards", "sum"),
        td=("pass_touchdown", "sum"),
        ints=("interception", "sum"),
    ).reset_index()
    # attach the game's kickoff + teams so the strictly-prior filter is a
    # single vectorized comparison
    meta = sched[["game_id", "gameday", "home_team", "away_team",
                  "home_qb_id", "away_qb_id"]].copy()
    meta["gameday"] = pd.to_datetime(meta["gameday"], errors="coerce")
    g = g.merge(meta, on="game_id", how="inner")
    return g


def _qb_window_stats(games: pd.DataFrame, passer_id: str,
                     before: pd.Timestamp) -> dict[str, Any]:
    """Strictly-prior totals for one passer, keyed by kickoff."""
    if games is None or not len(games) or before is None:
        return _null_stats()
    sub = games[(games["passer_id"].astype(str) == str(passer_id))
                & (games["gameday"] < before)]
    if not len(sub):
        return _null_stats()
    att = float(sub["att"].sum())
    out = {
        "starts": int(len(sub)),
        "cmp_pct": round(float(sub["cmp"].sum()) / att * 100.0, 1) if att else None,
        "yards_per_attempt": round(float(sub["yds"].sum()) / att, 2) if att else None,
        "td_per_game": round(float(sub["td"].sum()) / len(sub), 2),
        "ints": int(sub["ints"].sum()),
        "passer_rating": _passer_rating(sub),
    }
    return out


def _passer_rating(sub: pd.DataFrame) -> float | None:
    att = float(sub["att"].sum())
    if att <= 0:
        return None
    cmp_ = float(sub["cmp"].sum())
    yds = float(sub["yds"].sum())
    td = float(sub["td"].sum())
    ints = float(sub["ints"].sum())
    a = (cmp_ / att - 0.3) * 5.0
    b = (yds / att - 3.0) * 0.25
    c = (td / att) * 20.0
    d = 2.375 - (ints / att) * 25.0
    clamped = [min(max(x, 0.0), 2.375) for x in (a, b, c, d)]
    return round(sum(clamped) / 6.0 * 100.0, 1)


def _null_stats() -> dict[str, Any]:
    return {"name": "", "gsis_id": "", "passer_rating": None,
            "td_per_game": None, "cmp_pct": None, "yards_per_attempt": None,
            "ints": None, "starts": 0}


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

def build_qb_matchup(target_date: str | None = None,
                     weekly: pd.DataFrame | None = None,
                     snaps: pd.DataFrame | None = None) -> dict[str, Any]:
    """The dated record dict (pure — no I/O). ``target_date`` is YYYYMMDD;
    None resolves to the SLATE_SEASON board's earliest undecided gameday."""
    # decided seasons only for the pull (pbp years are bounded by the data
    # source); the slate season's schedule is fetched separately and appended.
    hist_seasons = [s for s in range(2019, SLATE_SEASON) if s <= 2025]
    sched_hist, _team_pbp = NF._load_raw(hist_seasons)
    pbp = load_pbp_wide(hist_seasons)
    import nflreadpy  # noqa: PLC0415
    s_next = nflreadpy.load_schedules([SLATE_SEASON]).to_pandas()
    sched = pd.concat([sched_hist, s_next], ignore_index=True)

    if weekly is None or snaps is None:
        weekly, snaps = load_depth_charts()
    roster = weekly if len(weekly) else snaps

    decided = sched[sched["home_score"].notna()
                    & sched["away_score"].notna()].copy()
    decided["gameday"] = pd.to_datetime(decided["gameday"], errors="coerce")
    passer_games = _passer_game_table(sched, pbp)
    computed_through = decided["gameday"].max()
    weekly_map = NF._dc_qb1_weekly(weekly)
    snap_map = NF._dc_qb1_snapshots(snaps)

    board = sched[sched["home_score"].isna() | sched["away_score"].isna()].copy()
    board["gameday"] = pd.to_datetime(board["gameday"], errors="coerce")
    board = board.dropna(subset=["gameday"])
    if target_date is not None:
        td = pd.Timestamp(datetime.strptime(str(target_date), "%Y%m%d"))
        board = board[board["gameday"] == td]
    if board.empty:
        return {"record": RECORD, "target_date": target_date or "",
                "written_utc": datetime.now(timezone.utc).isoformat(),
                "frame_sha256": CANONICAL_FRAME_SHA,
                "computed_through": None, "games": []}

    games_out: list[dict[str, Any]] = []
    for _, r in board.sort_values("gameday").iterrows():
        kickoff = r["gameday"]
        rec: dict[str, Any] = {
            "game_id": str(r["game_id"]),
            "gameday": str(pd.Timestamp(kickoff).date()),
            "home_team": str(r["home_team"]),
            "away_team": str(r["away_team"]),
        }
        for side, qb_col, team_col in (("qb_home", "home_qb_id", "home_team"),
                                       ("qb_away", "away_qb_id", "away_team")):
            gsis = str(r.get(qb_col) or "").strip()
            if gsis.lower() in ("nan", "none", ""):
                # Future rows carry no recorded starter: fall back to the
                # team's PUBLISHED QB1 (the latest depth-chart snapshot
                # strictly before kickoff — the expected-starter state the
                # tier-5 seam resolves; leakage-free by construction).
                gsis = _expected_starter(
                    str(r.get(team_col) or ""), int(r.get("season") or 0),
                    r.get("week"), kickoff, weekly_map, snap_map) or ""
            stats = _qb_window_stats(passer_games, gsis, kickoff) if gsis else _null_stats()
            stats["gsis_id"] = gsis
            stats["name"] = (_resolve_qb_name(gsis, roster) or
                             _snapshot_name(gsis, snaps) or
                             _snapshot_name(gsis, weekly))
            rec[side] = stats
        games_out.append(rec)

    return {
        "record": RECORD,
        "target_date": target_date or "",
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "frame_sha256": CANONICAL_FRAME_SHA,
        "computed_through": (str(pd.Timestamp(computed_through).date())
                             if computed_through is not None else None),
        "games": games_out,
    }


def emit(target_date: str | None = None,
         out_dir: Path | None = None) -> Path:
    """Build + write ``nfl_qb_matchup_<date>.json`` (tmp-write + atomic
    replace, the persist_markets convention). Returns the written path."""
    rec = build_qb_matchup(target_date)
    out_dir = Path(out_dir) if out_dir is not None else DATA_DELIVERY
    stamp = rec.get("target_date") or datetime.now(
        timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"nfl_qb_matchup_{stamp}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    """Emit the QB-matchup record for the slate target date (or today)."""
    args = list(sys.argv[1:]) if argv is None else list(argv)
    target = str(args[0]) if args else None
    path = emit(target)
    rec = json.loads(path.read_text())
    print(f"  wrote {path.name}: {len(rec.get('games') or [])} games, "
          f"computed_through={rec.get('computed_through')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
