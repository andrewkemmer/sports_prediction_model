"""Backfill pre-game designations for a season and report the play rate by
designation. Reads the official NBA injury report archive.

    python backfill_injury_designations.py 2025-10-21 2026-04-12
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nba_injury_report as ir  # noqa: E402

CACHE = Path(os.path.expanduser(
    "~/.cache/sports_prediction_model/nba/injury_reports"))
LOGS = Path(os.path.expanduser(
    "~/.cache/sports_prediction_model/nba/season_logs"))


def box_scores() -> pd.DataFrame:
    """One row per player per game, de-duplicated.

    Read from the whole-season files only. The cache directory also holds
    60-day slices from earlier runs, and concatenating both counts every game
    twice - which is a measurement error, not a data defect, and one worth
    naming because it is easy to repeat.
    """
    frames = []
    for path in sorted(LOGS.glob("log_*_Regular_Season.parquet")):
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    log = pd.concat(frames, ignore_index=True)
    log = log.drop_duplicates(
        subset=["nba_game_id", "player_id", "SEASON_ID"], keep="first")
    log["gameday"] = pd.to_datetime(log["gameday"]).dt.normalize()
    return log


def flip(name: str) -> str:
    """'Last, First' from the report -> 'First Last' in the box score.

    A contains-test is not safe here: 'Harris' sits inside 'Harrison' and
    'Paul' is a first name, so substring matching invents violations that
    look like source errors.
    """
    if "," not in name:
        return name
    last, first = name.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def main(start: date, end: date) -> None:
    log = box_scores()
    seasons = log[log.gameday.between(pd.Timestamp(start), pd.Timestamp(end))]
    print(f"box scores: {len(seasons)} player-games, "
          f"{seasons.gameday.nunique()} dates, "
          f"{seasons.player_name.nunique()} players", flush=True)

    rows = []
    seen: dict[str, str] = {}
    day = start
    while day <= end:
        played = seasons[seasons.gameday == pd.Timestamp(day)]
        if not played.empty:
            slate = ir.pre_game_designations(day, CACHE)
            # Join to the box score on (day, team, player), not on the day
            # alone. A day-level test would credit a player who suited up for
            # a different club that night, which is exactly the kind of
            # over-generous join that hides a leak.
            played_names = set(zip(played.player_name, played.team))
            for matchup, records in slate.items():
                abbrs = matchup.split("@")
                # The report prints the away block before the home block, so
                # the abbreviation for a full club name can be read off the
                # order of first appearance - and then CHECKED, because a full
                # name that resolves to two different abbreviations on
                # different nights means the assumption broke.
                order: list[str] = []
                for r in records:
                    if r.team and r.team not in order:
                        order.append(r.team)
                for full, code in zip(order, abbrs):
                    seen.setdefault(full, code)
                    if seen[full] != code:
                        print(f"  !! team order conflict: {full!r} seen as "
                              f"{seen[full]} and {code} ({matchup})", flush=True)
                index = {full: i for i, full in enumerate(order)}
                for r in records:
                    canonical = flip(r.player)
                    abbr = abbrs[min(index.get(r.team, 0), len(abbrs) - 1)]
                    rows.append(dict(
                        gameday=day, matchup=matchup, team=abbr,
                        team_full=r.team, player_report=r.player,
                        player=canonical, status=r.status, reason=r.reason,
                        published_at=r.published_at.isoformat(),
                        played=(canonical, abbr) in played_names))
            slate_rows = len(rows)
            n = sum(len(v) for v in slate.values())
            print(f"  {day}  games={len(slate):2d} designations={n:3d} "
                  f"played={sum(1 for r in rows if r['gameday']==day and r['played'])}",
                  flush=True)
        day += timedelta(days=1)

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no designations collected")
        return
    out = CACHE.parent / f"nba_designations_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    frame.to_parquet(out, index=False)
    print(f"\nwrote {len(frame)} designations -> {out}")

    by = frame.groupby("status").agg(
        n=("played", "size"), played=("played", "sum"))
    by["play_rate_pct"] = (100 * by.played / by.n).round(1)
    by = by.sort_values("n", ascending=False)
    print("\nPIT pre-game designation -> did the player play that game")
    print(by.to_string())
    print("\ntotal designations:", len(frame))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "2025-10-21"
    b = sys.argv[2] if len(sys.argv) > 2 else "2026-04-12"
    main(date.fromisoformat(a), date.fromisoformat(b))
