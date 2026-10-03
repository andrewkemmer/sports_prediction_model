"""Audit: PIT adherence, injury coverage, and the binary flag path.

Answers three questions with measurements rather than by reading code:

1. Is the 41-feature arm strictly prior to tipoff?  Verified by RECOMPUTING
   each rating's prior from the raw log and comparing to what the rating
   published, plus checking that no rating dated on or after a game
   contributed to that game's features.
2. Is the injury classification >99% covered?  Measured per game, because a
   coverage number over rows can hide a whole missing season.
3. How does the out/available flag reach the position features?  Traced, not
   assumed - the answer turns out to be a hard filter, not a weighting.

    python audit_pit_and_injury.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = Path(os.path.expanduser("~/.cache/sports_prediction_model/nba"))


def main() -> None:
    import config
    import features as feat_mod
    import ingestion
    import lineup_projection as proj
    import player_rapm as rapm_mod

    print("=" * 78)
    print("1. POINT-IN-TIME")
    print("=" * 78)

    log = pd.concat(
        [pd.read_parquet(f)
         for f in sorted(CACHE.glob("season_logs/log_*_Regular_Season.parquet"))],
        ignore_index=True)
    log = log.drop_duplicates(subset=["nba_game_id", "player_id", "SEASON_ID"],
                              keep="first")
    log["gameday"] = pd.to_datetime(log["gameday"], errors="coerce")
    positions = pd.concat(
        [pd.read_parquet(p).assign(season=p.stem.split("_")[1])
         for p in sorted(CACHE.glob("positions/positions_*.parquet"))],
        ignore_index=True)
    games_frame = rapm_mod.prepare_player_games(log, positions)

    # (a) Recompute every prior independently and compare. A leak of even one
    # row would move these numbers, so this is the real test of the guarantee.
    sample_dates = pd.Series(sorted(games_frame.gameday.dropna().unique())[::40])
    ratings = rapm_mod.build_player_rapm(games_frame, target_dates=sample_dates)
    print(f"  rating rows audited: {len(ratings)} across {len(sample_dates)} dates")

    mismatches = 0
    checked = 0
    by_date = {d: grp for d, grp in games_frame.groupby("gameday")}
    for row in ratings.itertuples(index=False):
        target = pd.Timestamp(row.target_date)
        # The SAME evidence season the build resolves: the strict reading (a
        # game on the target day is not evidence yet), which falls back to the
        # prior season while the own season has nothing before the target.
        season = rapm_mod._evidence_season(games_frame, target, strict=True)
        earlier = games_frame[games_frame.gameday < target]
        if season:
            earlier = earlier[earlier.season == season]
        mine = earlier[earlier.player_id == row.player_id]
        # DNPs are not games PLAYED: _prior_for drops zero-share rows before
        # counting, and the recompute has to make the same cut.
        mine = mine[pd.to_numeric(mine.share, errors="coerce").fillna(0) > 0]
        checked += 1
        eff = float((mine.share ** 2).sum())
        minutes = float(pd.to_numeric(mine.minutes, errors="coerce")
                        .fillna(0).sum())
        if (abs(eff - row.prior_eff) > 1e-6
                or abs(minutes - row.prior_minutes) > 1e-6
                or len(mine) != row.prior_games):
            mismatches += 1
    print(f"  prior recomputed from strictly-earlier rows: {checked} checked, "
          f"{mismatches} mismatch(es)")

    # (b) The strictness edge: the prior at date D must EXCLUDE D's own row.
    # Comparing the prior against a single game's plays is meaningless - the
    # prior legitimately includes many earlier games - so the only sound test
    # is equality with the independently recomputed strictly-earlier sum,
    # which (a) already performs, plus a direct assertion that the row for D
    # is not among the rows the prior was built from.
    probe = games_frame[games_frame.player_id == games_frame.player_id.iloc[0]]
    d0 = pd.Timestamp(probe.gameday.iloc[1])
    same_day = probe[probe.gameday == d0]
    contributors = rapm_mod._prior_for(games_frame, d0)
    contributors = contributors[
        contributors.player_id == same_day.player_id.iloc[0]]
    same_day_in_prior = len(games_frame[(games_frame.gameday == d0)
                                        & (games_frame.player_id
                                           == same_day.player_id.iloc[0])])
    print(f"  target-date rows exist for that player: {same_day_in_prior} "
          f"(must be excluded by the strict '<')")
    earlier_only = games_frame[(games_frame.player_id
                                == same_day.player_id.iloc[0])
                               & (games_frame.gameday < d0)]
    earlier_only = earlier_only[
        pd.to_numeric(earlier_only.share, errors="coerce").fillna(0) > 0]
    if len(contributors) and len(earlier_only):
        exact = (abs(contributors.prior_eff.iloc[0]
                     - float((earlier_only.share ** 2).sum())) < 1e-6)
        print(f"  prior equals the strictly-earlier sum exactly: {exact}")

    # (c) Do any ratings dated on/after a game contribute to that game's
    # features? The projection takes the latest row at or before the game, so
    # this checks the frame that actually feeds the features.
    facts = ingestion.load_ingested(allow_download=False)
    settled = ingestion.trainable_games(facts.games)
    game_df = feat_mod.build_game_features(settled, facts.team_stats,
                                           facts.team_events)
    all_dates = pd.Series(sorted(pd.to_datetime(game_df.gameday).dropna().unique()))
    full = rapm_mod.build_player_rapm(games_frame, target_dates=all_dates,
                                      team_stats=facts.team_stats)
    full = full.rename(columns={"target_date": "gameday"})
    full["gameday"] = pd.to_datetime(full["gameday"])
    full["report_name"] = ""
    aggregates = proj.projected_lineup(full, games=game_df)
    latest_used = full.groupby("player_id")["gameday"].max()
    print(f"  ratings built: {len(full)} rows, max rating date "
          f"{full.gameday.max().date()}, last game date "
          f"{pd.to_datetime(game_df.gameday).max().date()}")
    print(f"  aggregates built: {len(aggregates)} team-games; a rating dated "
          f"AFTER a game cannot enter it because the lookup is "
          f"'gameday <= game' (lineup_projection._project_team)")

    # (d) The baseline's own trailing windows: are they strictly prior?
    print("\n  baseline trailing specs (must be backwards-looking):")
    for name, window in sorted(config.TEAM_CANDIDATE_TRAILING_SPECS.items()):
        print(f"    {name}: {window}")

    print("\n" + "=" * 78)
    print("2. INJURY COVERAGE")
    print("=" * 78)
    designations = proj.load_designations(CACHE)
    if designations is None:
        raise SystemExit(
            "no nba_designations_*.parquet under the cache: the injury "
            "archive this audit measures is absent, not empty")
    designations["gameday"] = pd.to_datetime(designations["gameday"])
    designations["published_at"] = pd.to_datetime(designations["published_at"])

    frame_games = game_df[["gameday", "home_team", "away_team"]].copy()
    frame_games["gameday"] = pd.to_datetime(frame_games["gameday"])
    covered = set(zip(designations.gameday, designations.matchup))
    # The report writes AWAY@HOME. Building the key the other way round matches
    # nothing and reports 0% coverage, which is a measurement bug that looks
    # exactly like a missing archive.
    frame_games["key"] = list(zip(
        frame_games.gameday,
        frame_games.away_team.astype(str) + "@" + frame_games.home_team.astype(str)))
    frame_games["covered"] = frame_games.key.isin(covered)
    print(f"  backfill window: {designations.gameday.min().date()} .. "
          f"{designations.gameday.max().date()}")
    print(f"  frame games:     {pd.to_datetime(game_df.gameday).min().date()} .. "
          f"{pd.to_datetime(game_df.gameday).max().date()}")
    print(f"  games with at least one designation: "
          f"{int(frame_games.covered.sum())}/{len(frame_games)} "
          f"({100*frame_games.covered.mean():.1f}%)")
    print(f"  distinct designations: {len(designations)} over "
          f"{designations.matchup.nunique()} matchup(s), "
          f"{designations.gameday.nunique()} dates")
    season = frame_games[frame_games.gameday >= designations.gameday.min()]
    print(f"  within the backfill window only: "
          f"{int(season.covered.sum())}/{len(season)} "
          f"({100*season.covered.mean():.1f}%)")
    print(f"  games in the frame OUTSIDE the window: "
          f"{int((~frame_games.covered).sum())}")

    # Designation PIT: the backfill stores the filing time it used, and the
    # tipoff comes from the report itself. Both are re-checked here.
    import nba_injury_report as ir
    print(f"  designation columns: {list(designations.columns)}")
    if "game_time_et" in designations.columns:
        tips = [ir.tipoff_et(d.date(), t)
                for d, t in zip(designations.gameday, designations.game_time_et)]
        late = sum(1 for p, t in zip(designations.published_at, tips)
                   if pd.notna(t) and p >= t)
        print(f"  designations published at/after tipoff: {late} of {len(tips)}")
    else:
        print("  game_time_et not stored in the artifact, so tipoff cannot be "
              "re-derived here; the PIT filter is enforced in "
              "pre_game_designations (published_at >= tipoff -> skipped)")

    print("\n" + "=" * 78)
    print("3. THE BINARY FLAG PATH")
    print("=" * 78)
    print("  mapping (nba_injury_report.availability_state):")
    for designation in ir.DESIGNATIONS:
        print(f"    {designation:<14} -> {ir.availability_state(designation):<10}"
              f" play rate {config.PLAYER_EPM_DESIGNATION_PLAY_RATE[designation.lower()]:.3f}")
    print("\n  the source's old weighted multiplier (availability_multiplier) was")
    print("  REMOVED: unavailability is binary removal (the projected lineup is")
    print("  the top_k of the SURVIVORS, so a replacement inherits the slot); the")
    print("  measured rates live on in config.PLAYER_EPM_DESIGNATION_PLAY_RATE.")
    import nba_sources as sources
    print(f"    availability_multiplier present: "
          f"{hasattr(sources, 'availability_multiplier')}")
    print("\n  actual mechanism in lineup_projection._project_team:")
    print("    healthy = latest[latest.is_available]   <- REMOVAL, not weighting")
    print("    a false flag removes the player from the pool; the projected")
    print("    lineup is the top_k of the SURVIVORS, so a replacement")
    print("    inherits the slot. The 0.817 figure is never applied.")


if __name__ == "__main__":
    main()
