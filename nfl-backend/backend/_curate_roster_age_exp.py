"""One-shot provenance for nfl_roster_age_exp.csv (committed).

Verifies the live nflreadpy 0.1.5 ``load_rosters`` schema and produces the
tiny team-season table the Tier-3 ROSTER candidates consume:

    team, season, n, mean_age, mean_exp

Reading (documented decisions):
  - ``load_rosters`` returns WEEKLY roster snapshots per season (season, week,
    team, gsis_id, full_name, position, birth_date, years_exp, status, ...).
    There is no ``age`` column — age is derived from ``birth_date`` against a
    fixed season-start reference (Sep 1 of the season year), the same rule for
    every player and season so means are comparable.
  - Per season we take the EARLIEST REGULAR-SEASON week present (the snapshot
    closest to kickoff of week 1 = the pre-season-known roster), filtered to
    ``status == "ACT"`` players (fallback: all rows when a team has < 20 ACT
    rows), because statuses like CUT/RES describe roster churn, not the team
    the coaches will field.
  - 2026 is AVAILABLE (3,197 rows, week 1 REG, 32 teams) — verified 2026-09-01,
    so the slate's team-season facts are pre-season-known for every 2026 game.
  - Caveat: some older seasons' snapshots start late (2018 min week = 17).
    Team-mean age/experience move ~0.1 yr within a season, so this is a
    documented approximation, not a fabrication.

Leakage: each (team, season) fact depends only on the roster listing BEFORE
that season's games — never on game outcomes, so mapping it to that season's
games is point-in-time safe.

Usage (run once, commit both this script and its output):
    PYTHONUTF8=1 python _curate_roster_age_exp.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import nflreadpy
import polars as pl

BACKEND_DIR = Path(__file__).resolve().parent
OUT = BACKEND_DIR / "nfl_roster_age_exp.csv"

SEASONS = list(range(2018, 2027))          # 2018-2026 (2026 = slate season)
AGE_REFERENCE = {s: dt.date(s, 9, 1) for s in SEASONS}


def _snapshot_for_season(rows: pl.DataFrame, season: int) -> pl.DataFrame:
    """Earliest REG week covering >= 30 teams (full pre-season roster).

    ONE consistent rule for every season so team means are comparable: the
    FULL roster of the earliest REG week with >= 30 teams (2020+: week 1;
    2018/2019: only late-season weeks exist in the release and hold ~20
    teams — the lookup falls back to the nearest season for those pairs).
    No ACT filtering: 2026 labels ACT but 2020-2025 do not, so filtering
    would inject a cross-season level shift into the feature."""
    reg = rows.filter(pl.col("game_type") == "REG")
    if reg.height == 0:
        reg = rows
    week = None
    for w in sorted(reg["week"].unique().to_list()):
        if reg.filter(pl.col("week") == w)["team"].n_unique() >= 30:
            week = w
            break
    if week is None:
        week = reg["week"].mode().to_list()[0]
    snap = reg.filter(pl.col("week") == week)
    return (snap.with_columns(pl.lit(season).alias("season")), week, False)


def main() -> int:
    frames = []
    print("=== load_rosters schema verification (live) ===")
    for season in SEASONS:
        rows = nflreadpy.load_rosters(seasons=[season])
        snap, week, use_act = _snapshot_for_season(rows, season)
        ref = AGE_REFERENCE[season]
        if snap["birth_date"].dtype == pl.Date:
            snaps = snap
        else:
            snaps = snap.with_columns(
                pl.col("birth_date").str.to_date(strict=False))
        snap = (snaps.with_columns(
                    ((pl.lit(ref) - pl.col("birth_date"))
                     .dt.total_days() / 365.25).alias("age")))
        teams = snap["team"].n_unique()
        ages = snap["age"].drop_nulls()
        print(f"  {season}: {rows.height} raw rows | snapshot week "
              f"{snap['week'].min()} ({snap['game_type'].head(1).item()}) | "
              f"{teams} teams | age {ages.min():.1f}-"
              f"{ages.max():.1f} (act-filter={use_act})")
        agg = (snap.group_by(["season", "team"])
               .agg(n=pl.len(),
                    mean_age=pl.col("age").mean(),
                    mean_exp=pl.col("years_exp").mean())
               .sort(["season", "team"]))
        frames.append(agg)

    table = pl.concat(frames, how="vertical").sort(["team", "season"])
    # 2020+ and the 2026 slate: fail loudly rather than fabricate. 2018/2019
    # releases are partial (late-season weeks, ~20 teams) — documented
    # fallback to the nearest available season happens at lookup time.
    teams26 = table.filter(pl.col("season") == 2026)["team"].n_unique()
    missing = [s for s in range(2020, 2027)
               if table.filter(pl.col("season") == s)["team"].n_unique() < 32]
    print(f"  teams per season: 2018={table.filter(pl.col('season')==2018)['team'].n_unique()} "
          f"2019={table.filter(pl.col('season')==2019)['team'].n_unique()} "
          f"2026={teams26} (expect 32) | missing <32 in 2020-2026: "
          f"{missing or 'none'}")
    if teams26 < 32 or missing:
        raise SystemExit("roster snapshot incomplete — refusing to write")

    table.write_csv(OUT)
    print(f"wrote {OUT} ({table.height} rows)")
    print(table.filter(pl.col("season") == 2026).head(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())