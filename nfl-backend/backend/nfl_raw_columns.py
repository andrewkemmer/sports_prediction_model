"""Raw per-side performance columns for the NFL raw-columns ablation.

The served 12-feature pool is encoded as DIFFS (home minus away) — e.g.
``elo_diff``, ``ewm_net_pts_diff``. That encoding collapses the two per-side
values into one axis, which can discard signal a tree would otherwise split
on (a team that is strong on BOTH sides of a lopsided matchup, absolute rest
levels, etc.). This module re-emits the per-side values the production
composition ALREADY computes inside ``nfl_features.team_stats_ladder`` before
it diffs them (verified read-only 2026-09-01: elo_entering, win_pct,
rest_days, short_rest, ewm_net_pts, ewm_ypp, pace_plays_min all exist in the
ladder), as composed-but-unregistered columns:

    elo_entering     -> elo_home / elo_away
    win_pct          -> win_pct_home / win_pct_away
    rest_days        -> rest_days_home / rest_days_away
    ewm_net_pts      -> ewm_net_pts_home / ewm_net_pts_away
    ewm_ypp          -> ewm_ypp_home / ewm_ypp_away
    pace_plays_min   -> pace_plays_min_home / pace_plays_min_away
    short_rest       -> rest_short_home / rest_short_away

Venue / schedule items (is_dome_home, altitude_home, prime_time, div_game)
and travel_miles_diff have no raw/diff split (they are per-side facts or a
pure distance) and are deliberately NOT emitted.

PIT discipline: the raw columns are computed with the EXACT production path
(``nfl_features.compute_elo`` -> ``team_events`` -> ``team_stats_ladder``),
so they inherit the same strictly-prior shift(1) rules and the ladder's
strict-monotonicity assertion — no new leak surface. Nothing here touches
``FEATURE_COLUMNS``; the raws stay composed-but-unregistered until the
raw-columns sealed ablation (``run_nfl_raw_ablation.py``) rules on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_features import (_decided_rows, _pbp_team_agg_engine, compute_elo,
                          team_events, team_stats_ladder)

# ladder column -> (home column, away column)
RAW_PER_SIDE: dict[str, tuple[str, str]] = {
    "elo_entering": ("elo_home", "elo_away"),
    "win_pct": ("win_pct_home", "win_pct_away"),
    "rest_days": ("rest_days_home", "rest_days_away"),
    "ewm_net_pts": ("ewm_net_pts_home", "ewm_net_pts_away"),
    "ewm_ypp": ("ewm_ypp_home", "ewm_ypp_away"),
    "pace_plays_min": ("pace_plays_min_home", "pace_plays_min_away"),
    "short_rest": ("rest_short_home", "rest_short_away"),
}

RAW_PER_SIDE_COLS: list[str] = [
    c for pair in RAW_PER_SIDE.values() for c in pair]


def compose_raw_columns(feats: pd.DataFrame,
                        schedule: pd.DataFrame | None,
                        pbp: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the raw per-side columns to a built feature frame.

    ``feats`` is the decided-game frame from ``nfl_features.build_features``;
    ``schedule`` is the full nflreadpy schedule (warmup + core seasons) used
    for the trailing ladder — exactly the inputs ``build_features`` consumes,
    so the raw values come from the SAME strictly-prior point-in-time ladder
    the diffs come from. Returns ``feats`` with the raw columns added (the
    diffs and every other existing column are untouched).
    """
    out = feats.copy()
    if schedule is None:
        raise ValueError("compose_raw_columns: schedule is required (the "
                         "trailing ladder spans warmup + core seasons)")
    full = _decided_rows(schedule)
    events = compute_elo(team_events(full))
    team_agg = None
    if pbp is not None and {"yards_gained", "posteam"}.issubset(pbp.columns):
        team_agg = _pbp_team_agg_engine(pbp)
    ladder = team_stats_ladder(events, team_agg)   # asserts strict monotonicity

    gids = out["game_id"]
    for base, (home_col, away_col) in RAW_PER_SIDE.items():
        if base not in ladder.columns:
            out[home_col] = np.nan
            out[away_col] = np.nan
            continue
        home = ladder[ladder["is_home"]].set_index("game_id")[base]
        away = ladder[~ladder["is_home"]].set_index("game_id")[base]
        out[home_col] = home.reindex(gids).to_numpy()
        out[away_col] = away.reindex(gids).to_numpy()
    return out


def raw_coverage(feats: pd.DataFrame) -> dict[str, float]:
    """Per-raw-column non-null coverage (%) on a frame (decided rows only)."""
    return {c: float(100.0 * feats[c].notna().mean())
            for c in RAW_PER_SIDE_COLS if c in feats.columns}
