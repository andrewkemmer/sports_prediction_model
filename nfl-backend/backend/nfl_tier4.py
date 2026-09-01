"""Tier-4 (v6) feature families — the market-independent model's first
genuinely new information axes beyond the 12-feature fundamentals pool.

Two families plus two conditional arms, ALL composed-but-unregistered (never
in ``FEATURE_COLUMNS``; the deployed pool changes only when the Tier-4
ablation harness — ``run_tier4_ablation.py`` — admits something through the
same sealed-2025 gate as every prior tier):

  GS     (game-script / context-filtered): ewm_qb_epa_play_diff_gs,
         ewm_net_pts_diff_gs, ewm_ypp_diff_gs — the existing EWM axes
         restricted to NON-GARBAGE-TIME plays. Garbage time uses the
         nflfastR definition, reconstructed (nflverse pbp no longer ships a
         ``garbage_time`` column): quarters 1-3 garbage when the current win
         probability ``wp`` is <= 0.01 or >= 0.99; the 4th quarter when
         ``wp`` <= 0.05 or >= 0.95; OT is never garbage. ``wp`` (the nflverse
         WP model) is used, NOT ``vegas_wp`` — identical coverage, but the
         vegas variant is derived from the pre-game betting line, which the
         market-independence policy forbids anywhere in the NFL pipeline.
         Missing ``wp`` (kicks/penalties/end-of-half noise) is treated as
         non-garbage so the per-game aggregates keep full coverage.

  OPPADJ (opponent-adjusted): ewm_qb_epa_play_diff_oppadj,
         ewm_net_pts_diff_oppadj, ewm_ypp_diff_oppadj — each trailing EWM
         value minus the trailing mean of the OPPONENT'S same-axis value
         entering each prior game (schedule-strength adjustment; the exact
         ``opp_adj_form`` pattern from ``team_stats_ladder``).

  DRIVE  (conditional arm): ewm_yds_per_drive_diff, ewm_epa_per_drive_diff,
         ewm_qb_epa_per_drive_diff — per-DRIVE efficiency (value / distinct
         drives) instead of per-play, ewm'd with the same halflife=2.

  QB     (conditional arm): ewm_qb_epa_starter_diff — trailing QB EPA/play
         restricted to the ANNOUNCED/RECORDED starter's plays (nflverse
         schedule ``home_qb_id`` / ``away_qb_id``, which match pbp
         ``passer_id`` format). Verified 2026-09-01: decided games carry the
         starter id 100% per season 2018-2025; PENDING games (e.g. the 2026
         slate) carry None — nflverse does not publish expected starters
         pre-kickoff. That only limits a CURRENT-game QB-matchup feature;
         this trailing feature conditions on PAST games' recorded starters
         (known), so it is slate-safe: pending rows get the strictly-prior
         starter-QB trailing values and are never faked with an invented
         starter.

Leak-safety: every aggregate here is a function of THAT game's plays only
(per-(game_id, team) sums / counts / nunique), so the strictly-prior shift in
``team_stats_ladder`` keeps them leak-safe exactly like the v2/v3 aggregates.
The starter id is a pre-game fact (the schedule's announced starter), so
deciding the starter from the schedule — never from the game's own plays —
introduces no retrospective leakage. A pending game's own starter aggregate
is NaN (no announced starter), but the trailing QB-conditional value uses
only prior games' recorded starters, so the slate rows are populated
honestly.

The per-game rollup is pandas-only (``tier4_team_agg``) and merged onto the
base rollup in ``build_features`` / ``build_slate_features``; the DuckDB
engine's parity contract covers ``TEAM_AGG_COLUMNS``, which this module does
not touch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Extra raw PBP source columns this family reads (beyond the base set and
# TIER1_NEEDS, which already carries defteam / drive / touchdown /
# field_goal_result). ``nfl_features._load_raw`` adds these to its keep list.
TIER4_PBP_NEEDS = ("wp", "qtr", "passer_id")

# Candidate column names (composed-but-unregistered — absent from
# FEATURE_COLUMNS by construction; the harness gates them).
TIER4_GS_FEATURES = [
    "ewm_qb_epa_play_diff_gs", "ewm_net_pts_diff_gs", "ewm_ypp_diff_gs",
]
TIER4_OPPADJ_FEATURES = [
    "ewm_qb_epa_play_diff_oppadj", "ewm_net_pts_diff_oppadj",
    "ewm_ypp_diff_oppadj",
]
TIER4_DRIVE_FEATURES = [
    "ewm_yds_per_drive_diff", "ewm_epa_per_drive_diff",
    "ewm_qb_epa_per_drive_diff",
]
TIER4_QB_FEATURES = ["ewm_qb_epa_starter_diff"]

TIER4_CANDIDATES = (TIER4_GS_FEATURES + TIER4_OPPADJ_FEATURES
                    + TIER4_DRIVE_FEATURES + TIER4_QB_FEATURES)

# Rollup output columns, in fixed order (absent source fields -> NaN, never
# fabricated, matching the ``_pbp_team_agg`` convention).
TIER4_AGG_COLUMNS = [
    "game_id", "team",
    "qb_epa_sum_gs", "qb_epa_n_gs",
    "epa_sum_gs", "epa_n_gs",
    "total_yards_gs", "n_plays_gs",
    "pts_scored_gs", "pts_allowed_gs",
    "n_drives", "yds_per_drive", "epa_per_drive", "qb_epa_per_drive",
    "qb_epa_sum_start", "qb_epa_n_start",
]


def non_garbage_mask(pbp: pd.DataFrame) -> pd.Series | None:
    """Boolean Series (same index as ``pbp``): True = play is NOT garbage time.

    nflfastR garbage definition, reconstructed: Q1-3 garbage when the current
    win probability is <= 0.01 or >= 0.99; Q4 garbage when ``wp`` is <= 0.05
    or >= 0.95; OT (qtr 5) is never garbage. Uses ``wp`` (the nflverse WP
    model) — NOT ``vegas_wp`` (market-derived via the pre-game line, banned
    by the market-independence policy). Missing ``wp`` -> kept (non-garbage),
    so aggregate coverage is maximal. Returns None when the source columns
    are absent.
    """
    if pbp is None or "wp" not in pbp.columns or "qtr" not in pbp.columns:
        return None
    wp = pbp["wp"].astype(float)
    qtr = pbp["qtr"]
    garbage = (
        (qtr.between(1, 3) & ((wp <= 0.01) | (wp >= 0.99)))
        | ((qtr == 4) & ((wp <= 0.05) | (wp >= 0.95)))
    )
    return ~garbage.fillna(False)


def qb_map_from_schedule(schedule: pd.DataFrame | None) -> dict:
    """(game_id, team) -> announced/recorded starter player id (nflverse
    ``home_qb_id`` / ``away_qb_id``). Rows without an id are skipped; a
    missing/None schedule yields an empty map (the QB arm degrades to NaN).
    """
    out: dict = {}
    if schedule is None or "home_qb_id" not in schedule.columns:
        return out
    s = schedule[["game_id", "home_team", "away_team",
                  "home_qb_id", "away_qb_id"]]
    for team_col, id_col in (("home_team", "home_qb_id"),
                             ("away_team", "away_qb_id")):
        sub = s[s[id_col].notna()]
        for gid, team, qid in zip(sub["game_id"], sub[team_col], sub[id_col]):
            out[(gid, team)] = qid
    return out


def tier4_team_agg(pbp: pd.DataFrame | None,
                   qb_map: dict | None = None) -> pd.DataFrame:
    """Per-(game_id, team) Tier-4 play aggregates (pandas-only seam).

    Columns (see ``TIER4_AGG_COLUMNS``): non-garbage-time sums/counts for
    QB-EPA / EPA / yards, non-garbage points scored (own) and allowed (the
    defending team), drive counts with per-drive rates, and the starter-QB
    EPA sum/count (restricted to the schedule's announced starter). Every
    column is a function of that game's plays only; absent source columns
    degrade to NaN. Byte-agnostic to the DuckDB engine by design (the parity
    contract pins only ``TEAM_AGG_COLUMNS``, untouched here).
    """
    cols = list(TIER4_AGG_COLUMNS)
    if pbp is None or "posteam" not in getattr(pbp, "columns", []):
        return pd.DataFrame(columns=cols)
    if "game_id" not in pbp.columns or "yards_gained" not in pbp.columns:
        return pd.DataFrame(columns=cols)
    p = pbp.copy()
    p = p.dropna(subset=["posteam"])
    has = set(pbp.columns)
    # Base per-(game_id, team) universe (the ``_pbp_team_agg`` convention):
    # every team with at least one play. Sub-aggregates merge onto it.
    g = p.groupby(["game_id", "posteam"], as_index=False).agg(
        n_plays=("yards_gained", "count")).rename(columns={"posteam": "team"})

    def _merge(sub: pd.DataFrame, key_col: str = "posteam") -> None:
        nonlocal g
        sub = sub.rename(columns={key_col: "team"})
        g = g.merge(sub, on=["game_id", "team"], how="outer")

    # --- non-garbage-time (GS) aggregates ---------------------------------
    # Mask computed on ``p`` (posteam-dropped copy) so it is index-aligned;
    # a play's garbage status is a function of its own wp/qtr row.
    ng = non_garbage_mask(p)
    gs = p[ng] if ng is not None else p.iloc[0:0]
    if "qb_epa" in has and not gs.empty:
        sub = gs.groupby(["game_id", "posteam"], as_index=False).agg(
            qb_epa_sum_gs=("qb_epa", "sum"), qb_epa_n_gs=("qb_epa", "count"))
        _merge(sub)
    else:
        g["qb_epa_sum_gs"] = np.nan
        g["qb_epa_n_gs"] = np.nan
    if "epa" in has and not gs.empty:
        sub = gs.groupby(["game_id", "posteam"], as_index=False).agg(
            epa_sum_gs=("epa", "sum"), epa_n_gs=("epa", "count"))
        _merge(sub)
    else:
        g["epa_sum_gs"] = np.nan
        g["epa_n_gs"] = np.nan
    if not gs.empty:
        sub = gs.groupby(["game_id", "posteam"], as_index=False).agg(
            total_yards_gs=("yards_gained", "sum"),
            n_plays_gs=("yards_gained", "count"))
        _merge(sub)
    else:
        g["total_yards_gs"] = np.nan
        g["n_plays_gs"] = np.nan
    # Non-garbage points scored (own) / allowed (defteam side).
    if ("touchdown" in has and "field_goal_result" in has and not gs.empty):
        p2 = gs.copy()
        p2["_pts"] = ((p2["touchdown"] == 1).astype(float) * 7.0
                      + (p2["field_goal_result"] == "made").astype(float) * 3.0)
        sub = p2.groupby(["game_id", "posteam"], as_index=False).agg(
            pts_scored_gs=("_pts", "sum"))
        _merge(sub)
        d = p2.dropna(subset=["defteam"])
        if not d.empty:
            sub = d.groupby(["game_id", "defteam"], as_index=False).agg(
                pts_allowed_gs=("_pts", "sum"))
            _merge(sub, key_col="defteam")
        else:
            g["pts_allowed_gs"] = np.nan
    else:
        g["pts_scored_gs"] = np.nan
        g["pts_allowed_gs"] = np.nan

    # --- drive-level aggregates (value / distinct drives) -----------------
    if "drive" in has:
        nd = p.groupby(["game_id", "posteam"], as_index=False).agg(
            n_drives=("drive", "nunique"))
        y = p.groupby(["game_id", "posteam"], as_index=False).agg(
            tot=("yards_gained", "sum"))
        nd = nd.merge(y, on=["game_id", "posteam"], how="left")
        nd["yds_per_drive"] = nd["tot"] / nd["n_drives"].replace(0, np.nan)
        if "epa" in has:
            e = p.groupby(["game_id", "posteam"], as_index=False).agg(
                epa_sum=("epa", "sum"))
            nd = nd.merge(e, on=["game_id", "posteam"], how="left")
            nd["epa_per_drive"] = nd["epa_sum"] / nd["n_drives"].replace(0, np.nan)
        else:
            nd["epa_per_drive"] = np.nan
        if "qb_epa" in has:
            q = p.groupby(["game_id", "posteam"], as_index=False).agg(
                qb_epa_sum=("qb_epa", "sum"))
            nd = nd.merge(q, on=["game_id", "posteam"], how="left")
            nd["qb_epa_per_drive"] = nd["qb_epa_sum"] / nd["n_drives"].replace(0, np.nan)
        else:
            nd["qb_epa_per_drive"] = np.nan
        _merge(nd.drop(columns=["tot", "epa_sum", "qb_epa_sum"]))
    else:
        for c in ("n_drives", "yds_per_drive", "epa_per_drive",
                  "qb_epa_per_drive"):
            g[c] = np.nan

    # --- QB-conditional: announced starter's plays only --------------------
    if "passer_id" in has and qb_map:
        qbt = pd.DataFrame([(gid, team, qid)
                            for (gid, team), qid in qb_map.items()],
                           columns=["game_id", "team", "qb_id"])
        q = p[p["passer_id"].notna()].merge(
            qbt, left_on=["game_id", "posteam"],
            right_on=["game_id", "team"], how="inner")
        q = q[q["passer_id"] == q["qb_id"]]
        if not q.empty:
            sub = q.groupby(["game_id", "team"], as_index=False).agg(
                qb_epa_sum_start=("qb_epa", "sum"),
                qb_epa_n_start=("qb_epa", "count"))
            _merge(sub, key_col="team")
        else:
            g["qb_epa_sum_start"] = np.nan
            g["qb_epa_n_start"] = np.nan
    else:
        g["qb_epa_sum_start"] = np.nan
        g["qb_epa_n_start"] = np.nan

    for c in cols:
        if c not in g.columns:
            g[c] = np.nan
    return g[cols]


# Ladder column names the diff composition reads (mirrors the tier4 names).
_TIER4_LADDER_COLS = {
    "ewm_qb_epa_play_diff_gs": "ewm_qb_epa_gs",
    "ewm_net_pts_diff_gs": "ewm_net_pts_gs",
    "ewm_ypp_diff_gs": "ewm_ypp_gs",
    "ewm_qb_epa_play_diff_oppadj": "ewm_qb_epa_oppadj",
    "ewm_net_pts_diff_oppadj": "ewm_net_pts_oppadj",
    "ewm_ypp_diff_oppadj": "ewm_ypp_oppadj",
    "ewm_yds_per_drive_diff": "ewm_yds_per_drive",
    "ewm_epa_per_drive_diff": "ewm_epa_per_drive",
    "ewm_qb_epa_per_drive_diff": "ewm_qb_epa_per_drive",
    "ewm_qb_epa_starter_diff": "ewm_qb_epa_starter",
}


def compose_tier4_features(df: pd.DataFrame,
                           schedule: pd.DataFrame | None = None,
                           pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the 10 Tier-4 candidate columns to a built feature frame.

    ``df`` is the output of ``nfl_features.build_features`` /
    ``build_slate_features`` (must carry game_id, teams, gameday, scores). The
    tier4 ladder values come from the SAME strictly-prior machinery
    (``team_stats_ladder``, which this module's aggregates feed): the
    per-(game_id, team) GS/drive/starter aggregates are merged onto the base
    PBP rollup, the ladder is re-run, and each candidate is home − away.

    This is the Tier-4 composition seam: ``run_tier4_ablation.py`` consumes it
    now, and the one-line integration into ``build_features`` /
    ``build_slate_features`` (``df = compose_tier4_features(df, schedule,
    pbp)`` before the venue/roster composition) is the follow-up that makes
    the production pipeline emit the columns. The candidates stay
    composed-but-unregistered either way — absent from FEATURE_COLUMNS until
    the sealed-2025 ablation admits them.
    """
    from nfl_features import (_decided_rows, _home_minus_away,
                              _pbp_team_agg_engine, team_events,
                              team_stats_ladder)
    for c in TIER4_CANDIDATES:
        df[c] = np.nan
    if schedule is None or pbp is None or "game_id" not in df.columns:
        return df
    if not {"game_id", "posteam"}.issubset(pbp.columns):
        return df
    t4 = tier4_team_agg(pbp, qb_map_from_schedule(schedule))
    if t4.empty:
        return df
    team_agg = _pbp_team_agg_engine(pbp)
    if team_agg is not None and not team_agg.empty:
        team_agg = team_agg.merge(t4, on=["game_id", "team"], how="outer")
    else:
        team_agg = t4
    # Ladder over the full decided timeline, then the scheduled (undecided)
    # rows appended so slate rows get trailing values from strictly-prior
    # DECIDED games only — the build_slate_features pattern. Pending rows
    # carry no qb_id, so their QB-conditional candidate stays NaN.
    events = team_events(_decided_rows(schedule))
    sched = schedule.copy()
    for c in ("home_score", "away_score"):
        if c in sched.columns:
            sched[c] = pd.to_numeric(sched[c], errors="coerce")
    pending = sched[sched["home_score"].isna() | sched["away_score"].isna()]
    if not pending.empty:
        events = pd.concat([events, team_events(pending)], ignore_index=True)
    ladder = team_stats_ladder(events, team_agg)
    gids = df["game_id"]
    for feat, ladder_col in _TIER4_LADDER_COLS.items():
        df[feat] = _home_minus_away(ladder, gids, ladder_col)
    return df
