"""
Pure DuckDB SQL feature engineering for MLB Statcast data.

ZERO pandas during feature engineering.  All rolling windows, shifted
features, groupby aggregations, and joins are DuckDB SQL window functions
operating on Parquet files on disk.

Architecture:
    pitches.parquet → DuckDB SQL → game_df.parquet + pbp_df.parquet

PIT compliance:
    All rolling metrics use LAG() first (shift), then ROWS BETWEEN N
    PRECEDING AND CURRENT ROW (rolling over shifted values).  This ensures
    no data leakage — Game T features use only data from games < T.
"""
from __future__ import annotations

import gc
import logging
import resource
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

VENUE_MAP = {
    "ARI": "Chase Field", "ATL": "Truist Park", "BAL": "Oriole Park at Camden Yards",
    "BOS": "Fenway Park", "CHC": "Wrigley Field", "CWS": "Rate Field",
    "CIN": "Great American Ball Park", "CLE": "Progressive Field",
    "COL": "Coors Field", "DET": "Comerica Park", "HOU": "Minute Maid Park",
    "KC": "Kauffman Stadium", "LAA": "Angel Stadium", "LAD": "Dodger Stadium",
    "MIA": "loanDepot park", "MIL": "American Family Field",
    "MIN": "Target Field", "NYM": "Citi Field", "NYY": "Yankee Stadium",
    "OAK": "Sutter Health Park", "PHI": "Citizens Bank Park",
    "PIT": "PNC Park", "SD": "Petco Park", "SF": "Oracle Park",
    "SEA": "T-Mobile Park", "STL": "Busch Stadium", "TB": "Steinbrenner Field",
    "TEX": "Globe Life Field", "TOR": "Rogers Centre", "WSH": "Nationals Park",
}

PA_END_EVENTS = (
    "'single', 'double', 'triple', 'home_run',"
    "'strikeout', 'strikeout_double_play',"
    "'walk', 'hit_by_pitch',"
    "'field_out', 'field_error', 'fielders_choice', 'fielders_choice_out',"
    "'grounded_into_double_play', 'double_play', 'triple_play',"
    "'sac_fly', 'sac_bunt', 'sac_fly_double_play',"
    "'catcher_interf', 'batter_interference',"
    "'force_out', 'sacrifice_bunt_double_play'"
)


def _connect(pitches_path: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB in-memory, load pitches from Parquet, tune for low RAM."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '4GB'")
    # Allow generous temp space for spilling to disk (critical for large queries)
    con.execute("SET temp_directory = '/tmp/duckdb_temp'")
    con.execute("SET max_temp_directory_size = '50GB'")

    con.execute(f"CREATE TABLE pitches AS SELECT * FROM '{pitches_path}'")

    con.execute("""
        ALTER TABLE pitches
        ALTER COLUMN game_date TYPE DATE
        USING CAST(game_date AS DATE)
    """)

    existing = {r[0] for r in con.execute("DESCRIBE pitches").fetchall()}

    # Drop genuinely unused columns
    for col in [
        "fielder_2", "fielder_3", "fielder_4", "fielder_5",
        "fielder_6", "fielder_7", "fielder_8", "fielder_9",
        "if_fielding_alignment", "of_fielding_alignment",
        "post_home_score", "post_away_score",
        "event", "type", "launch_speed_angle",
    ]:
        if col in existing:
            con.execute(f'ALTER TABLE pitches DROP COLUMN "{col}"')

    # Ensure ALL columns referenced by SQL queries exist (schema-robust).
    # Missing columns become NULL — never assume any column exists.
    _required = {
        "game_pk", "game_date", "game_type", "home_team", "away_team",
        "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
        "on_1b", "on_2b", "on_3b",
        "at_bat_number", "pitch_number", "pitcher", "batter",
        "p_throws", "stand",
        "pitch_type", "release_speed", "description", "events",
        "barrel", "hard_contact", "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle", "estimated_ba_using_speedangle",
        "zone", "home_score", "away_score", "spin_rate",
        "woba_value", "babip_value", "iso_value",
        "delta_home_win_exp", "delta_run_exp", "player_name",
        "hit_distance_sc", "release_pos_x", "release_pos_z",
        "release_spin_rate", "release_extension", "pfx_x", "pfx_z",
    }
    existing2 = {r[0] for r in con.execute("DESCRIBE pitches").fetchall()}
    # String columns must be VARCHAR, not DOUBLE — otherwise IN() clauses fail
    _str_cols = {
        "pitch_type", "events", "description", "player_name",
        "home_team", "away_team", "pitcher", "batter",
        "stand", "p_throws", "game_type", "inning_topbot",
    }
    for col in _required - existing2:
        dtype = "VARCHAR" if col in _str_cols else "DOUBLE"
        con.execute(f'ALTER TABLE pitches ADD COLUMN "{col}" {dtype}')
        con.execute(f'UPDATE pitches SET "{col}" = NULL')

    return con


# ── Game-level features ─────────────────────────────────────────────────────

def _tz_case_lines(indent: str = "                ") -> str:
    return "\n".join(
        f"{indent}WHEN '{k}' THEN {v}" for k, v in TEAM_TZ_OFFSETS.items()
    )


def _build_travel_features(con: duckdb.DuckDBPyConnection) -> None:
    """7g. Travel fatigue — timezone crossings across each team's last three
    games PRIOR to today (point-in-time: strictly past games only).

    The offset compared is that of each game's VENUE (the home team's park),
    NOT the travelling team's own city — a team's home timezone never
    changes, so comparing it to itself can never register a crossing.
    """
    tz_lines = _tz_case_lines()
    con.execute(f"""
        CREATE TABLE game_venue_tz AS
        SELECT DISTINCT CAST(game_date AS DATE) AS gd, game_pk,
               home_team, away_team,
               CASE home_team
{tz_lines}
                       ELSE 0 END AS venue_off
        FROM pitches
    """)
    con.execute("""
        CREATE TABLE team_travel_raw AS
        SELECT gd, game_pk, team, venue_off AS off FROM (
            SELECT gd, game_pk, home_team AS team, venue_off FROM game_venue_tz
            UNION ALL
            SELECT gd, game_pk, away_team AS team, venue_off FROM game_venue_tz
        )
    """)
    con.execute("""
        CREATE TABLE travel_seq AS
        SELECT *, LAG(off) OVER (PARTITION BY team ORDER BY gd, game_pk) AS prev_off
        FROM team_travel_raw
    """)
    con.execute("""
        CREATE TABLE travel_cross AS
        SELECT gd, team,
               CASE WHEN prev_off IS NOT NULL AND off != prev_off
                    THEN 1 ELSE 0 END AS crossed
        FROM travel_seq
    """)
    con.execute("""
        CREATE TABLE travel_fatigue AS
        WITH games AS (
            SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS gd,
                   home_team, away_team FROM pitches
        ),
        home_tz AS (
            SELECT g.game_pk, SUM(c.crossed) AS tz_crossed
            FROM games g JOIN travel_cross c
              ON c.team = g.home_team
             AND c.gd < g.gd AND c.gd >= g.gd - INTERVAL 3 DAY
            GROUP BY g.game_pk
        ),
        away_tz AS (
            SELECT g.game_pk, SUM(c.crossed) AS tz_crossed
            FROM games g JOIN travel_cross c
              ON c.team = g.away_team
             AND c.gd < g.gd AND c.gd >= g.gd - INTERVAL 3 DAY
            GROUP BY g.game_pk
        )
        SELECT g.game_pk,
               COALESCE(h.tz_crossed, 0) AS time_zones_crossed_last_3d_home,
               COALESCE(a.tz_crossed, 0) AS time_zones_crossed_last_3d_away
        FROM games g
        LEFT JOIN home_tz h ON g.game_pk = h.game_pk
        LEFT JOIN away_tz a ON g.game_pk = a.game_pk
    """)


def _build_closer_features(con: duckdb.DuckDBPyConnection) -> None:
    """7h. Closer availability — point-in-time high-leverage metric.

    For EVERY game, each team's closer is identified strictly from prior
    work: the reliever with the most cumulative late-inning (8th+) batters
    faced over the trailing 30 days. He is UNAVAILABLE entering tonight only
    when he pitched BOTH of the previous two days. Teams without an
    established closer default to available. The check runs for every game,
    whether or not the closer ends up pitching tonight — the previous
    implementation attached usage state only to games the closer appeared
    in, which collapsed the flag to ~0.8% nonzero.
    """
    con.execute(f"""
        CREATE TABLE late_relief AS
        SELECT CAST(p.game_date AS DATE) AS gd, p.game_pk,
               CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                    ELSE p.away_team END AS team,
               p.pitcher,
               COUNT(DISTINCT p.at_bat_number) AS tbf
        FROM pitches p
        JOIN starters s ON p.game_pk = s.game_pk
        WHERE p.inning >= 8
          AND (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
          AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
        GROUP BY 1, 2, 3, 4
    """)
    # Daily workload per reliever + cumulative workload BEFORE each date
    # (window excludes the current row → strictly prior, PIT-safe).
    con.execute("""
        CREATE TABLE rel_daily AS
        SELECT gd, pitcher, SUM(tbf) AS tbf FROM late_relief GROUP BY 1, 2
    """)
    con.execute("""
        CREATE TABLE rel_cum AS
        SELECT gd, pitcher,
            COALESCE(SUM(SUM(tbf)) OVER (
                PARTITION BY pitcher ORDER BY gd
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0.0) AS cum_before
        FROM rel_daily GROUP BY gd, pitcher
    """)
    con.execute("""
        CREATE TABLE rel_team AS
        SELECT pitcher, arg_max(team, gd) AS team
        FROM late_relief GROUP BY pitcher
    """)
    con.execute("""
        CREATE TABLE team_closer_pit AS
        WITH games AS (
            SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS gd,
                   home_team, away_team FROM pitches
        ),
        pairs AS (
            SELECT game_pk, gd, home_team AS team FROM games
            UNION ALL
            SELECT game_pk, gd, away_team AS team FROM games
        ),
        candidates AS (
            SELECT p.game_pk, p.gd, p.team, rc.pitcher, rc.cum_before,
                   ROW_NUMBER() OVER (PARTITION BY p.game_pk, p.team
                                      ORDER BY rc.cum_before DESC NULLS LAST,
                                               rc.pitcher) AS rn
            FROM pairs p
            JOIN rel_team tr ON tr.team = p.team
            JOIN rel_cum rc ON rc.pitcher = tr.pitcher
                 AND rc.gd < p.gd AND rc.gd >= p.gd - INTERVAL 30 DAY
        )
        SELECT game_pk, gd, team, pitcher FROM candidates WHERE rn = 1
    """)
    con.execute("""
        CREATE TABLE closer_avail AS
        WITH state AS (
            SELECT c.game_pk, c.team,
                   COALESCE(BOOL_OR(u.gd = c.gd - INTERVAL 1 DAY), FALSE) AS d1,
                   COALESCE(BOOL_OR(u.gd = c.gd - INTERVAL 2 DAY), FALSE) AS d2
            FROM team_closer_pit c
            LEFT JOIN rel_daily u
              ON u.pitcher = c.pitcher
             AND u.gd IN (c.gd - INTERVAL 1 DAY, c.gd - INTERVAL 2 DAY)
            GROUP BY 1, 2
        )
        SELECT g.game_pk,
            CASE WHEN sh.d1 AND sh.d2 THEN 0.0 ELSE 1.0 END AS closer_available_home,
            CASE WHEN sa.d1 AND sa.d2 THEN 0.0 ELSE 1.0 END AS closer_available_away
        FROM (SELECT DISTINCT game_pk, home_team, away_team FROM pitches) g
        LEFT JOIN state sh ON sh.game_pk = g.game_pk AND sh.team = g.home_team
        LEFT JOIN state sa ON sa.game_pk = g.game_pk AND sa.team = g.away_team
    """)



def _build_rest_days(con: duckdb.DuckDBPyConnection) -> None:
    """Season-scoped rest days per team-game, capped at REST_DAYS_CAP.

    A season opener has no prior game IN THAT SEASON → NULL (never the
    ~180-day October→March gap, which was an extreme outlier against a
    0–4 day in-season distribution).
    """
    # Rest days (days since each team's previous game).
    # Season-scoped: a season opener has no prior game IN THAT SEASON, so
    # rest is NULL (never the ~180-day October→March gap, which was an
    # extreme outlier against a 0–4 day in-season distribution). In-season
    # gaps are capped at REST_DAYS_CAP so All-Star breaks and long layoffs
    # don't fabricate outsized values.
    con.execute("""
        CREATE TABLE rest_days AS
        WITH team_games AS (
            SELECT DISTINCT game_date, game_pk, home_team AS team FROM pitches
            UNION
            SELECT DISTINCT game_date, game_pk, away_team AS team FROM pitches
        ),
        with_prev AS (
            SELECT *, LAG(game_date) OVER (
                       PARTITION BY team ORDER BY game_date, game_pk) AS prev_date
            FROM team_games
        )
        SELECT game_pk, team,
               CASE
                   WHEN prev_date IS NULL THEN NULL
                   WHEN YEAR(CAST(prev_date AS DATE)) != YEAR(CAST(game_date AS DATE))
                       THEN NULL
                   ELSE LEAST(
                       CAST(game_date AS DATE) - CAST(prev_date AS DATE),
                       %d)
               END AS rest_days
        FROM with_prev
    """ % REST_DAYS_CAP)



def _build_pitcher_stuff(con: duckdb.DuckDBPyConnection) -> None:
    """Last-3-start fastball/whiff form and season-to-date xwOBA splits.

    The xwOBA-vs-hand cumulative windows are PARTITIONED BY SEASON: an
    April start must never average in the prior October (the old
    unpartitioned window silently produced career-to-date values while the
    docs claimed "season to date").
    """
    # Starter stuff trends — last-3-start fastball velo/mix, whiff rate,
    # and CURRENT-SEASON-to-date xwOBA allowed vs left/right-handed
    # batters (windows are season-partitioned: April never averages in
    # the prior October).
    # All windows are LAG-shifted so the current game is excluded (point-in-time).
    con.execute(f"""
        CREATE TABLE pitcher_stuff_raw AS
        WITH per_start AS (
            SELECT CAST(game_date AS DATE) AS game_date,
                   YEAR(CAST(game_date AS DATE)) AS season, game_pk, pitcher,
                   AVG(CASE WHEN pitch_type IN ('FF','SI','FT')
                            THEN release_speed END) AS fb_velo,
                   AVG(CASE WHEN pitch_type IN ('FF','SI','FT')
                            THEN 1.0 ELSE 0.0 END) AS fb_pct,
                   COUNT(*) AS n_pitches,
                   SUM(CASE WHEN description IN
                            ('swinging_strike','swinging_strike_blocked')
                            THEN 1 ELSE 0 END) AS whiffs,
                   AVG(CASE WHEN events IN ({PA_END_EVENTS}) AND stand = 'L'
                            THEN estimated_woba_using_speedangle END) AS xwoba_vs_l,
                   AVG(CASE WHEN events IN ({PA_END_EVENTS}) AND stand = 'R'
                            THEN estimated_woba_using_speedangle END) AS xwoba_vs_r
            FROM pitches
            GROUP BY game_date, game_pk, pitcher
        )
        SELECT *,
            LAG(fb_velo, 1) OVER w AS _s_fb_velo,
            LAG(fb_pct, 1) OVER w AS _s_fb_pct,
            LAG(n_pitches, 1) OVER w AS _s_n,
            LAG(whiffs, 1) OVER w AS _s_whiffs,
            LAG(xwoba_vs_l, 1) OVER ws AS _s_xl,
            LAG(xwoba_vs_r, 1) OVER ws AS _s_xr
        FROM per_start
        WINDOW w AS (PARTITION BY pitcher ORDER BY game_date),
               ws AS (PARTITION BY pitcher, season ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE pitcher_stuff AS
        SELECT game_date, game_pk, pitcher,
            AVG(_s_fb_velo) OVER w3 AS sp_fbvelo_3g,
            AVG(_s_fb_pct) OVER w3 AS sp_fbpct_3g,
            SUM(_s_whiffs) OVER w3 / NULLIF(SUM(_s_n) OVER w3, 0) AS sp_whiff_3g,
            AVG(_s_xl) OVER wall AS sp_xwoba_vs_l,
            AVG(_s_xr) OVER wall AS sp_xwoba_vs_r
        FROM pitcher_stuff_raw
        WINDOW w3 AS (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
               wall AS (PARTITION BY pitcher, season ORDER BY game_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

def _build_game_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build game_level table via pure DuckDB SQL.  All intermediate tables
    are dropped after assembly to free RAM."""

    logger.info("Building game-level features...")

    # 1. Game winners (last pitch of each game)
    con.execute("""
        CREATE TABLE game_winners AS
        WITH last_pitch AS (
            SELECT game_pk, game_date, home_team, away_team,
                   home_score, away_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk
                       ORDER BY at_bat_number DESC, pitch_number DESC
                   ) AS rn
            FROM pitches
        )
        SELECT game_pk,
               CAST(game_date AS DATE) AS game_date,
               home_team, away_team,
               home_score, away_score,
               CASE WHEN home_score > away_score THEN 1.0
                    WHEN away_score > home_score THEN 0.0
                    ELSE NULL END AS home_win,
               (COALESCE(home_score, 0) + COALESCE(away_score, 0)) AS total_runs
        FROM last_pitch WHERE rn = 1
    """)

    # 2. Starting pitchers (first PA in inning 1)
    con.execute("""
        CREATE TABLE starters AS
        -- Top of the 1st: AWAY team bats, so the HOME team's starter pitches.
        -- These were swapped before (home SP stats were being read off the
        -- away starter and vice versa on every card and feature).
        WITH first_pa_top AS (
            SELECT game_pk, pitcher AS home_starter_id, p_throws AS home_starter_hand,
                   ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Top'
        ),
        first_pa_bot AS (
            SELECT game_pk, pitcher AS away_starter_id, p_throws AS away_starter_hand,
                   ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Bot'
        )
        SELECT DISTINCT p.game_pk,
               CAST(p.game_date AS DATE) AS game_date,
               p.home_team, p.away_team,
               t.home_starter_id, t.home_starter_hand,
               b.away_starter_id, b.away_starter_hand
        FROM (SELECT DISTINCT game_pk, game_date, home_team, away_team FROM pitches) p
        LEFT JOIN first_pa_top t ON p.game_pk = t.game_pk AND t.rn = 1
        LEFT JOIN first_pa_bot b ON p.game_pk = b.game_pk AND b.rn = 1
    """)

    # 3. Venues
    venue_lines = "\n".join(f"            WHEN '{k}' THEN '{v}'" for k, v in VENUE_MAP.items())
    con.execute(f"""
        CREATE TABLE venues AS
        SELECT DISTINCT game_pk, home_team,
            CASE home_team
{venue_lines}
                ELSE 'Unknown'
            END AS venue
        FROM pitches
    """)

    _build_rest_days(con)

    # 5. Pitcher rolling features (PIT-compliant: LAG first, then rolling SUM)
    #
    # Innings pitched and runs allowed are computed from real game state, not
    # proxies: outs come from a per-PA out-event map and runs from the score
    # progression across PA boundaries (attributed to the pitcher who threw
    # the final pitch of the scoring PA). The previous proxy (PA count / 3 as
    # "IP" and hits+BB+HBP-HR as "runs") produced ERA ~7.5 and K/9 ~6.0 —
    # numbers that don't exist in Major League Baseball.
    con.execute(f"""
        CREATE TABLE pa_boundary AS
        WITH lastp AS (
            SELECT CAST(game_date AS DATE) AS game_date,
                   game_pk, inning, inning_topbot, at_bat_number,
                   pitcher, events,
                   launch_speed, launch_angle,
                   COALESCE(home_score, 0) + COALESCE(away_score, 0) AS tot_score,
                   estimated_woba_using_speedangle AS xwoba_val,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk, inning, inning_topbot, at_bat_number
                       ORDER BY pitch_number DESC
                   ) AS rn
            FROM pitches
            WHERE events IN ({PA_END_EVENTS})
        ),
        lp AS (SELECT * FROM lastp WHERE rn = 1),
        seq AS (
            SELECT *,
                LAG(tot_score) OVER (
                    -- at_bat_number is globally monotone within a game_pk, so this
                    -- is true chronological PA order. The previous ordering
                    -- (all Top half-innings, then all Bottom) made tot_score carry
                    -- across half-innings, so LAG jumps credited the OTHER team's
                    -- runs to this half's pitcher — double-counting every run
                    -- (verified on game_pk 824317: 20 attributed vs 10 actual),
                    -- which inflated all ERA-family features ~2x.
                    PARTITION BY game_pk
                    ORDER BY at_bat_number
                ) AS prev_tot
            FROM lp
        )
        SELECT game_date, game_pk, pitcher, events,
               -- First PA of the game: score starts 0-0, so its runs = tot_score.
               CASE WHEN prev_tot IS NULL THEN tot_score
                    ELSE GREATEST(tot_score - prev_tot, 0) END AS runs_on_pa,
               CASE events
                    WHEN 'field_out' THEN 1
                    WHEN 'strikeout' THEN 1
                    WHEN 'strikeout_double_play' THEN 2
                    WHEN 'grounded_into_double_play' THEN 2
                    WHEN 'double_play' THEN 2
                    WHEN 'triple_play' THEN 3
                    WHEN 'sac_fly' THEN 1
                    WHEN 'sac_bunt' THEN 1
                    WHEN 'fielders_choice_out' THEN 1
                    WHEN 'sac_fly_double_play' THEN 2
                    WHEN 'sacrifice_bunt_double_play' THEN 2
                    ELSE 0 END AS outs_on_pa,
               xwoba_val,
               -- Barrel proxy: EV>=98 with LA in [22,36]. Calibrated on live
               -- pulls: the official core band (LA 26-30) alone yields ~2.3%
               -- of BBE (real league rate is ~8%), while this widened band
               -- lands at ~7% here. The EV+LA sum-rule overcounts (~19%) in
               -- our data, so the angle-band form is used.
               CASE WHEN launch_speed >= 98 AND launch_angle BETWEEN 22 AND 36
                    THEN 1.0 ELSE 0.0 END AS barrel_flag,
               CASE WHEN launch_speed >= 95 THEN 1.0 ELSE 0.0 END AS hard_flag
        FROM seq
    """)
    con.execute(f"""
        CREATE TABLE pitcher_game_stats AS
        SELECT game_date, game_pk, pitcher,
               COUNT(*) AS n_batters_faced,
               SUM(outs_on_pa) / 3.0 AS ip,
               SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS ks,
               SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bbs,
               SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbps,
               SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END) AS hits_allowed,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs_allowed,
               SUM(runs_on_pa) AS runs,
               AVG(xwoba_val) AS xwoba,
               AVG(barrel_flag) AS barrel_rate,
               AVG(hard_flag) AS hard_contact_rate
        FROM pa_boundary
        GROUP BY game_date, game_pk, pitcher
    """)

    con.execute("""
        CREATE TABLE pitcher_shifted AS
        SELECT *,
            LAG(ip, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_ip,
            LAG(runs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_runs,
            LAG(ks, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_ks,
            LAG(bbs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_bbs,
            LAG(hits_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hits,
            LAG(hrs_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hrs,
            LAG(hbps, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hbps,
            LAG(xwoba, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_xwoba
        FROM pitcher_game_stats
    """)

    con.execute("""
        CREATE TABLE pitcher_rolling AS
        SELECT game_date, game_pk, pitcher,
            SUM(_s_runs) OVER w AS _roll_runs,
            SUM(_s_ks) OVER w AS _roll_ks,
            SUM(_s_bbs) OVER w AS _roll_bbs,
            SUM(_s_hits) OVER w AS _roll_hits,
            SUM(_s_hrs) OVER w AS _roll_hrs,
            SUM(_s_hbps) OVER w AS _roll_hbps,
            SUM(_s_ip) OVER w AS _roll_ip,
            AVG(_s_xwoba) OVER w AS _roll_xwoba
        FROM pitcher_shifted
        WINDOW w AS (PARTITION BY pitcher ORDER BY game_date
                     ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
    """)

    # Season-to-date ERA / K/9 (strictly in-season, via season-partitioned
    # LAGs so the prior October never leaks into a new season's cumulative)
    # plus last-5-start ERA / K/9 (ACROSS seasons — no season-start gap: an
    # April start rolls over the prior season's tail). All point-in-time
    # safe: the row holds the previous game's stats via LAG, so the current
    # game never enters its own feature.
    con.execute("""
        CREATE TABLE pitcher_shifted_season AS
        SELECT *,
            LAG(ip, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_ip,
            LAG(runs, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_runs,
            LAG(ks, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_ks
        FROM (SELECT *, EXTRACT(YEAR FROM game_date) AS season FROM pitcher_game_stats)
    """)

    con.execute("""
        CREATE TABLE pitcher_season_rolling AS
        SELECT game_date, game_pk, pitcher,
            SUM(_s_runs) OVER w_season AS _s_runs_s,
            SUM(_s_ks)  OVER w_season AS _s_ks_s,
            SUM(_s_ip)  OVER w_season AS _s_ip_s
        FROM pitcher_shifted_season
        WINDOW w_season AS (PARTITION BY pitcher, season ORDER BY game_date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

    # Last-5-start window across ALL seasons (no season partition): built on
    # the cross-season LAGs in pitcher_shifted, so a season's first starts
    # roll over the prior season's tail — no gap at the season boundary. A
    # career debut (no prior start at all) yields NULL naturally.
    con.execute("""
        CREATE TABLE pitcher_5g_rolling AS
        SELECT game_date, game_pk, pitcher,
            SUM(_s_runs) OVER w5 AS _roll5_runs,
            SUM(_s_ks)  OVER w5 AS _roll5_ks,
            SUM(_s_ip)  OVER w5 AS _roll5_ip
        FROM pitcher_shifted
        WINDOW w5 AS (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
    """)

    con.execute("""
        CREATE TABLE pitcher_season_features AS
        SELECT psr.game_date, psr.game_pk, psr.pitcher,
            -- True season-to-date ERA / K/9 (through the prior in-season
            -- starts only; NULL for a season's opening start).
            psr._s_runs_s / NULLIF(psr._s_ip_s, 0) * 9.0 AS sp_era,
            psr._s_ks_s / NULLIF(psr._s_ip_s, 0) * 9.0 AS sp_k9,
            -- Last-5-start ERA / K/9 across seasons (no start-count guard).
            p5._roll5_runs / NULLIF(p5._roll5_ip, 0) * 9.0 AS sp_era_5g,
            p5._roll5_ks / NULLIF(p5._roll5_ip, 0) * 9.0 AS sp_k9_5g
        FROM pitcher_season_rolling psr
        LEFT JOIN pitcher_5g_rolling p5
               ON psr.game_pk = p5.game_pk AND psr.pitcher = p5.pitcher
    """)

    con.execute("""
        CREATE TABLE pitcher_features AS
        SELECT game_date, game_pk, pitcher,
            _roll_bbs / NULLIF(_roll_ip, 0) * 9.0 AS sp_bb9_30g,
            (_roll_bbs + _roll_hits) / NULLIF(_roll_ip, 0) AS sp_whip_30g,
            (13 * _roll_hrs + 3 * (_roll_bbs + _roll_hbps) - 2 * _roll_ks)
                / NULLIF(_roll_ip, 0) AS sp_fip_30g,
            _roll_xwoba AS sp_xwoba_30g
        FROM pitcher_rolling
    """)

    # 6. Team offense rolling features
    con.execute(f"""
        CREATE TABLE team_offense_raw AS
        WITH pa_events AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS batting_team,
                   events
            FROM pitches WHERE events IN ({PA_END_EVENTS})
        ),
        game_agg AS (
            SELECT game_date, game_pk, batting_team,
                   COUNT(*) AS n_pa,
                   SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS singles,
                   SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS doubles,
                   SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS triples,
                   SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
                   SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
                   SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
                   SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS ks
            FROM pa_events GROUP BY game_date, game_pk, batting_team
        )
        SELECT *,
            (0.690*bb + 0.722*hbp + 0.878*singles + 1.242*doubles
             + 1.568*triples + 2.007*hrs) / NULLIF(n_pa, 0) AS team_woba_game,
            (doubles + 2*triples + 3*hrs) / NULLIF(n_pa - bb - hbp, 0) AS team_iso_game,
            ks::DOUBLE / NULLIF(n_pa, 0) AS team_k_rate_game,
            bb::DOUBLE / NULLIF(n_pa, 0) AS team_bb_rate_game
        FROM game_agg
    """)

    con.execute("""
        CREATE TABLE team_off_shifted AS
        SELECT *,
            LAG(team_woba_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_woba,
            LAG(team_iso_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_iso,
            LAG(team_k_rate_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_krate,
            LAG(team_bb_rate_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date) AS _s_bbrate
        FROM team_offense_raw
    """)
    con.execute("""
        CREATE TABLE team_offense_rolling AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_woba) OVER w AS team_woba_30g,
            AVG(_s_iso) OVER w AS team_iso_30g,
            AVG(_s_krate) OVER w AS team_k_rate_30g,
            AVG(_s_bbrate) OVER w AS team_bb_rate_30g
        FROM team_off_shifted
        WINDOW w AS (PARTITION BY batting_team ORDER BY game_date
                     ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)    # 7. Bullpen rolling features

    # Reliever workload is attributed to the FIELDING team (Top half → home
    # team pitchers, Bottom half → away team), and BOTH starting pitchers are
    # excluded. The old first_pitchers CTE had one row per PITCH and only
    # excluded the home starter, so every join fanned out ~300x (a team's
    # 3-day "bullpen pitch count" read 69,901) and the away starter's warmup-
    # free pitches were billed to the home bullpen.
    con.execute(f"""
        CREATE TABLE bullpen_raw AS
        WITH reliever_events AS (
            SELECT CAST(p.game_date AS DATE) AS game_date,
                   p.game_pk,
                   CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                        ELSE p.away_team END AS fielding_team,
                   p.events
            FROM pitches p
            JOIN starters s ON p.game_pk = s.game_pk
            WHERE (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
              AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
              AND p.events IN ('single','double','triple','home_run',
                  'strikeout','strikeout_double_play','walk','hit_by_pitch',
                  'field_out','field_error','fielders_choice','fielders_choice_out',
                  'grounded_into_double_play','double_play','triple_play',
                  'sac_fly','sac_bunt','sac_fly_double_play',
                  'catcher_interf','batter_interference',
                  'force_out','sacrifice_bunt_double_play')
        )
        SELECT game_date, game_pk, fielding_team AS team,
            SUM(CASE events
                    WHEN 'field_out' THEN 1
                    WHEN 'strikeout' THEN 1
                    WHEN 'strikeout_double_play' THEN 2
                    WHEN 'grounded_into_double_play' THEN 2
                    WHEN 'double_play' THEN 2
                    WHEN 'triple_play' THEN 3
                    WHEN 'sac_fly' THEN 1
                    WHEN 'sac_bunt' THEN 1
                    WHEN 'fielders_choice_out' THEN 1
                    WHEN 'sac_fly_double_play' THEN 2
                    WHEN 'sacrifice_bunt_double_play' THEN 2
                    ELSE 0 END) / 3.0 AS bullpen_ip,
            SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS bullpen_ks,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bullpen_bbs,
            SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS bullpen_hits,
            SUM(CASE WHEN events IN ('single','double','triple','home_run','walk','hit_by_pitch') THEN 1 ELSE 0 END) AS bullpen_runs
        FROM reliever_events
        GROUP BY game_date, game_pk, fielding_team
    """)
    con.execute("""
        CREATE TABLE bp_daily AS
        SELECT CAST(p.game_date AS DATE) AS day,
               CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                    ELSE p.away_team END AS team,
               COUNT(*) AS pitches,
               SUM(CASE p.events
                    WHEN 'field_out' THEN 1
                    WHEN 'strikeout' THEN 1
                    WHEN 'strikeout_double_play' THEN 2
                    WHEN 'grounded_into_double_play' THEN 2
                    WHEN 'double_play' THEN 2
                    WHEN 'triple_play' THEN 3
                    WHEN 'sac_fly' THEN 1
                    WHEN 'sac_bunt' THEN 1
                    WHEN 'fielders_choice_out' THEN 1
                    WHEN 'sac_fly_double_play' THEN 2
                    WHEN 'sacrifice_bunt_double_play' THEN 2
                    ELSE 0 END) / 3.0 AS ip
        FROM pitches p
        JOIN starters s ON p.game_pk = s.game_pk
        WHERE (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
          AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE TABLE bp_fatigue AS
        WITH games AS (
            SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS gd,
                   home_team, away_team FROM pitches
        ),
        home_load AS (
            SELECT g.game_pk, SUM(d.pitches) AS pitches_3d, SUM(d.ip) AS ip_3d
            FROM games g JOIN bp_daily d
              ON d.team = g.home_team
             AND d.day < g.gd AND d.day >= g.gd - INTERVAL 3 DAY
            GROUP BY g.game_pk
        ),
        away_load AS (
            SELECT g.game_pk, SUM(d.pitches) AS pitches_3d, SUM(d.ip) AS ip_3d
            FROM games g JOIN bp_daily d
              ON d.team = g.away_team
             AND d.day < g.gd AND d.day >= g.gd - INTERVAL 3 DAY
            GROUP BY g.game_pk
        )
        SELECT g.game_pk,
               h.pitches_3d AS bullpen_pitches_3d_home,
               h.ip_3d AS bullpen_ip_3d_home,
               a.pitches_3d AS bullpen_pitches_3d_away,
               a.ip_3d AS bullpen_ip_3d_away
        FROM games g
        LEFT JOIN home_load h ON g.game_pk = h.game_pk
        LEFT JOIN away_load a ON g.game_pk = a.game_pk
    """)
    con.execute("""
        CREATE TABLE bullpen_shifted AS
        SELECT *,
            LAG(bullpen_bbs, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_bbs,
            LAG(bullpen_hits, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_hits,
            LAG(bullpen_ip, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_ip,
            LAG(bullpen_runs, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_runs
        FROM bullpen_raw
    """)
    con.execute("""
        CREATE TABLE bullpen_rolling AS
        SELECT game_date, game_pk, team,
            (SUM(_s_bbs) OVER w10 + SUM(_s_hits) OVER w10)
                / NULLIF(SUM(_s_ip) OVER w10, 0) AS bullpen_whip_10g,
            SUM(_s_runs) OVER w10 / NULLIF(SUM(_s_ip) OVER w10, 0) * 9.0 AS bullpen_era_10g,
            (SUM(_s_bbs) OVER w3 + SUM(_s_hits) OVER w3)
                / NULLIF(SUM(_s_ip) OVER w3, 0) AS bullpen_whip_3g
        FROM bullpen_shifted
        WINDOW w10 AS (PARTITION BY team ORDER BY game_date
                       ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
               w3 AS (PARTITION BY team ORDER BY game_date
                      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    """)

    _build_pitcher_stuff(con)

    # 7c. Team contact-form — barrel rate / hard-hit rate / avg exit velo over
    # the trailing 15 games (LAG-shifted, excludes current game).
    # Balls in play ONLY: launch_speed is also populated on foul balls
    # (~76 mph avg vs ~87 on BIP), which dragged the old mean to ~83.
    # Barrel/hard-hit are derived from the official Statcast definitions —
    # the `barrel`/`hard_contact` columns don't exist in real Statcast pulls,
    # so the old AVG() over them silently produced all-NULL features.
    con.execute("""
        CREATE TABLE team_contact_raw AS
        WITH bip AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   CASE WHEN launch_speed >= 98 AND launch_angle BETWEEN 26 AND 30
                        THEN 1.0 ELSE 0.0 END AS barrel_flag,
                   CASE WHEN launch_speed >= 95 THEN 1.0 ELSE 0.0 END AS hard_flag,
                   launch_speed
            FROM pitches
            WHERE description = 'hit_into_play' AND launch_speed IS NOT NULL
        )
        SELECT game_date, game_pk, batting_team,
               AVG(barrel_flag) AS barrel_rate,
               AVG(hard_flag) AS hardhit_rate,
               AVG(launch_speed) AS exitvelo
        FROM bip
        GROUP BY game_date, game_pk, batting_team
    """)
    con.execute("""
        CREATE TABLE team_contact_shifted AS
        SELECT *,
            LAG(barrel_rate, 1) OVER w AS _s_barrel,
            LAG(hardhit_rate, 1) OVER w AS _s_hardhit,
            LAG(exitvelo, 1) OVER w AS _s_exitvelo
        FROM team_contact_raw
        WINDOW w AS (PARTITION BY batting_team ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE team_contact_rolling AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_barrel) OVER w15 AS team_barrel_15g,
            AVG(_s_hardhit) OVER w15 AS team_hardhit_15g,
            AVG(_s_exitvelo) OVER w15 AS team_exitvelo_15g
        FROM team_contact_shifted
        WINDOW w15 AS (PARTITION BY batting_team ORDER BY game_date
                       ROWS BETWEEN 14 PRECEDING AND CURRENT ROW)
    """)

    # 7d. Opposing-lineup handedness share — fraction of a team's PAs taken by
    # left-handed batters over the trailing 30 games. Paired with each
    # starter's xwOBA-vs-L/R splits, tree models can learn platoon-fit
    # interactions without raw player-name encoding.
    con.execute(f"""
        CREATE TABLE team_hand_raw AS
        WITH pa AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   stand
            FROM pitches WHERE events IN ({PA_END_EVENTS})
        )
        SELECT game_date, game_pk, batting_team,
               AVG(CASE WHEN stand = 'L' THEN 1.0 ELSE 0.0 END) AS lefty_share
        FROM pa
        GROUP BY game_date, game_pk, batting_team
    """)
    con.execute("""
        CREATE TABLE team_hand_shifted AS
        SELECT *, LAG(lefty_share, 1) OVER w AS _s_lefty
        FROM team_hand_raw
        WINDOW w AS (PARTITION BY batting_team ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE team_hand_rolling AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_lefty) OVER w30 AS lineup_lefty_share_30g
        FROM team_hand_shifted
        WINDOW w30 AS (PARTITION BY batting_team ORDER BY game_date
                       ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)

    # 7e. Lineup composition — every hitter gets his own statistical
    # assumption: a per-player trailing-30g wOBA shrunk toward the
    # point-in-time league mean by PA count (empirical Bayes), then
    # aggregated into expected-lineup features (top-9 by playing time).
    con.execute(f"""
        CREATE TABLE batter_game_stats AS
        WITH pa AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   batter, events
            FROM pitches WHERE events IN ({PA_END_EVENTS})
        )
        SELECT game_date, game_pk, batting_team, batter,
               COUNT(*) AS pa,
               SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
               SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
               SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS s,
               SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS d,
               SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS t,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play')
                        THEN 1 ELSE 0 END) AS k
        FROM pa
        GROUP BY game_date, game_pk, batting_team, batter
    """)
    con.execute("""
        CREATE TABLE batter_shifted AS
        SELECT *,
            LAG(pa, 1) OVER w AS _pa,
            LAG(bb, 1) OVER w AS _bb,
            LAG(hbp, 1) OVER w AS _hbp,
            LAG(s, 1) OVER w AS _s,
            LAG(d, 1) OVER w AS _d,
            LAG(t, 1) OVER w AS _t,
            LAG(hr, 1) OVER w AS _hr
        FROM batter_game_stats
        WINDOW w AS (PARTITION BY batter ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE batter_rolling AS
        SELECT game_date, game_pk, batting_team, batter,
            SUM(0.690*_bb + 0.722*_hbp + 0.878*_s + 1.242*_d
                + 1.568*_t + 2.007*_hr) OVER w30 AS _woba_num,
            SUM(_pa - _bb - _hbp) OVER w30 AS _ab,
            SUM(_pa) OVER w30 AS _pa30
        FROM batter_shifted
        WINDOW w30 AS (PARTITION BY batter ORDER BY game_date
                       ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)
    # League mean wOBA, cumulative through each date — built ONLY from
    # already-shifted (prior-game) stats, so it stays point-in-time safe.
    con.execute("""
        CREATE TABLE batter_league AS
        SELECT game_date,
            SUM(_woba_num) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
              / NULLIF(SUM(_ab) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
              AS lg_woba
        FROM (
            SELECT game_date, SUM(_woba_num) AS _woba_num, SUM(_ab) AS _ab
            FROM batter_rolling GROUP BY game_date
        )
    """)
    con.execute("""
        CREATE TABLE batter_ratings AS
        SELECT r.game_date, r.game_pk, r.batting_team, r.batter,
               CASE WHEN COALESCE(r._ab, 0) > 0 THEN
                   (r._woba_num + COALESCE(l.lg_woba, 0.315) * 120)
                   / (r._ab + 120)
               END AS shrunk_woba,
               r._pa30
        FROM batter_rolling r
        LEFT JOIN batter_league l USING (game_date)
    """)
    con.execute("""
        CREATE TABLE lineup_agg AS
        WITH ranked AS (
            SELECT game_date, game_pk, batting_team, shrunk_woba,
                   ROW_NUMBER() OVER (PARTITION BY game_pk, batting_team
                                      ORDER BY _pa30 DESC) AS rn
            FROM batter_ratings WHERE shrunk_woba IS NOT NULL
        ),
        top9 AS (SELECT * FROM ranked WHERE rn <= 9)
        SELECT game_pk, batting_team,
               AVG(shrunk_woba) AS lineup_woba_mean,
               AVG(CASE WHEN rn <= 3 THEN shrunk_woba END) AS lineup_woba_top3,
               STDDEV(shrunk_woba) AS lineup_woba_std
        FROM top9
        GROUP BY game_pk, batting_team
    """)

    # 7f. Lineup OPS split by opposing pitching hand — every hitter's
    # trailing-30g OPS vs left-handed and right-handed pitchers separately
    # (LAG-shifted, excludes the current game), aggregated over the expected
    # top-9 by playing time per hand.  Paired with tonight's opposing starter
    # throwing hand at assembly time this yields lineup_ops_vs_starter_hand.
    con.execute(f"""
        CREATE TABLE batter_hand_game AS
        WITH pa AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   batter, p_throws, events
            FROM pitches WHERE events IN ({PA_END_EVENTS})
        )
        SELECT game_date, game_pk, batting_team, batter, p_throws,
               COUNT(*) AS pa_n,
               SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
               SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
               SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS s,
               SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS d,
               SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS t,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr
        FROM pa GROUP BY 1, 2, 3, 4, 5
    """)
    con.execute("""
        CREATE TABLE batter_hand_shifted AS
        SELECT *,
            LAG(pa_n, 1) OVER w AS _pa_n,
            LAG(bb, 1) OVER w AS _bb,
            LAG(hbp, 1) OVER w AS _hbp,
            LAG(s, 1) OVER w AS _s,
            LAG(d, 1) OVER w AS _d,
            LAG(t, 1) OVER w AS _t,
            LAG(hr, 1) OVER w AS _hr
        FROM batter_hand_game
        WINDOW w AS (PARTITION BY batter, p_throws ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE batter_hand_rolling AS
        SELECT game_date, game_pk, batting_team, batter, p_throws,
            SUM(_s + _d + _t + _hr) OVER w30 AS _h30,
            SUM(_s + 2 * _d + 3 * _t + 4 * _hr) OVER w30 AS _tb30,
            SUM(_bb + _hbp + _s + _d + _t + _hr) OVER w30 AS _onb_num,
            SUM(_pa_n - _bb - _hbp) OVER w30 AS _ab30,
            SUM(_pa_n) OVER w30 AS _pa30
        FROM batter_hand_shifted
        WINDOW w30 AS (PARTITION BY batter, p_throws ORDER BY game_date
                       ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE lineup_ops_agg AS
        WITH ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY game_pk, batting_team, p_throws
                                   ORDER BY _pa30 DESC) AS rn
            FROM batter_hand_rolling WHERE _ab30 > 0
        ),
        top9 AS (
            SELECT game_pk, batting_team, p_throws,
                   (_onb_num::DOUBLE / _ab30) + (_tb30::DOUBLE / _ab30) AS ops
            FROM ranked WHERE rn <= 9
        )
        SELECT game_pk, batting_team,
               AVG(CASE WHEN p_throws = 'L' THEN ops END) AS lineup_ops_vs_l,
               AVG(CASE WHEN p_throws = 'R' THEN ops END) AS lineup_ops_vs_r
        FROM top9
        GROUP BY game_pk, batting_team
    """)

    # 7g/7h — travel fatigue + closer availability (helpers above).
    _build_travel_features(con)
    _build_closer_features(con)

    # 8. Assemble game_level via LEFT JOINs
    con.execute("""
        CREATE TABLE game_level AS
        SELECT
            w.game_pk, w.game_date, w.home_team, w.away_team,
            w.home_score, w.away_score, w.home_win, w.total_runs,
            s.home_starter_id, s.away_starter_id, v.venue,
            rh.rest_days AS rest_days_home, ra.rest_days AS rest_days_away,
            sh.sp_era AS sp_era_home, sh.sp_k9 AS sp_k9_home,
            ph.sp_bb9_30g AS sp_bb9_home, ph.sp_whip_30g AS sp_whip_home,
            ph.sp_fip_30g AS sp_fip_home, ph.sp_xwoba_30g AS sp_xwoba_home,
            sa.sp_era AS sp_era_away, sa.sp_k9 AS sp_k9_away,
            pa.sp_bb9_30g AS sp_bb9_away, pa.sp_whip_30g AS sp_whip_away,
            pa.sp_fip_30g AS sp_fip_away, pa.sp_xwoba_30g AS sp_xwoba_away,
            -- Last-5-start SP twins under their own names (feature 6/8)
            sh.sp_era_5g AS sp_era_5g_home, sh.sp_k9_5g AS sp_k9_5g_home,
            sa.sp_era_5g AS sp_era_5g_away, sa.sp_k9_5g AS sp_k9_5g_away,
            th.team_woba_30g AS team_woba_30g_home, th.team_iso_30g AS team_iso_30g_home,
            th.team_k_rate_30g AS team_k_rate_30g_home, th.team_bb_rate_30g AS team_bb_rate_30g_home,
            ta.team_woba_30g AS team_woba_30g_away, ta.team_iso_30g AS team_iso_30g_away,
            ta.team_k_rate_30g AS team_k_rate_30g_away, ta.team_bb_rate_30g AS team_bb_rate_30g_away,
            bh.bullpen_whip_10g AS bullpen_whip_10g_home, bh.bullpen_era_10g AS bullpen_era_10g_home,
            ba.bullpen_whip_10g AS bullpen_whip_10g_away, ba.bullpen_era_10g AS bullpen_era_10g_away,
            bh.bullpen_whip_3g AS bullpen_whip_3g_home,
            ba.bullpen_whip_3g AS bullpen_whip_3g_away,
            hst.sp_fbvelo_3g AS sp_fbvelo_3g_home, hst.sp_fbpct_3g AS sp_fbpct_3g_home,
            hst.sp_whiff_3g AS sp_whiff_3g_home,
            hst.sp_xwoba_vs_l AS sp_xwoba_vs_l_home, hst.sp_xwoba_vs_r AS sp_xwoba_vs_r_home,
            ast.sp_fbvelo_3g AS sp_fbvelo_3g_away, ast.sp_fbpct_3g AS sp_fbpct_3g_away,
            ast.sp_whiff_3g AS sp_whiff_3g_away,
            ast.sp_xwoba_vs_l AS sp_xwoba_vs_l_away, ast.sp_xwoba_vs_r AS sp_xwoba_vs_r_away,
            ch.team_barrel_15g AS team_barrel_15g_home, ch.team_hardhit_15g AS team_hardhit_15g_home,
            ch.team_exitvelo_15g AS team_exitvelo_15g_home,
            ca.team_barrel_15g AS team_barrel_15g_away, ca.team_hardhit_15g AS team_hardhit_15g_away,
            ca.team_exitvelo_15g AS team_exitvelo_15g_away,
            -- opp_lefty_share_* = the LINEUP THE OPPOSING STARTER FACES:
            -- home col uses the AWAY team's batters (facing home starter).
            hd.lineup_lefty_share_30g AS opp_lefty_share_home,
            ha.lineup_lefty_share_30g AS opp_lefty_share_away,
            bf.bullpen_pitches_3d_home, bf.bullpen_ip_3d_home,
            bf.bullpen_pitches_3d_away, bf.bullpen_ip_3d_away,
            loh.lineup_ops_vs_l AS lineup_ops_vs_l_home,
            loh.lineup_ops_vs_r AS lineup_ops_vs_r_home,
            loa.lineup_ops_vs_l AS lineup_ops_vs_l_away,
            loa.lineup_ops_vs_r AS lineup_ops_vs_r_away,
            -- Feature 25: each lineup's OPS vs the hand of the starter it faces
            CASE WHEN s.away_starter_hand = 'L' THEN loh.lineup_ops_vs_l
                 WHEN s.away_starter_hand = 'R' THEN loh.lineup_ops_vs_r
                 ELSE NULL END AS lineup_ops_vs_starter_hand_home,
            CASE WHEN s.home_starter_hand = 'L' THEN loa.lineup_ops_vs_l
                 WHEN s.home_starter_hand = 'R' THEN loa.lineup_ops_vs_r
                 ELSE NULL END AS lineup_ops_vs_starter_hand_away,
            tf.time_zones_crossed_last_3d_home, tf.time_zones_crossed_last_3d_away,
            cl.closer_available_home, cl.closer_available_away,
            lh.lineup_woba_mean AS lineup_woba_mean_home,
            lh.lineup_woba_top3 AS lineup_woba_top3_home,
            lh.lineup_woba_std AS lineup_woba_std_home,
            la.lineup_woba_mean AS lineup_woba_mean_away,
            la.lineup_woba_top3 AS lineup_woba_top3_away,
            la.lineup_woba_std AS lineup_woba_std_away
        FROM game_winners w
        LEFT JOIN starters s ON w.game_pk = s.game_pk
        LEFT JOIN venues v ON w.game_pk = v.game_pk
        LEFT JOIN rest_days rh ON w.game_pk = rh.game_pk AND w.home_team = rh.team
        LEFT JOIN rest_days ra ON w.game_pk = ra.game_pk AND w.away_team = ra.team
        LEFT JOIN pitcher_season_features sh ON w.game_pk = sh.game_pk AND sh.pitcher = s.home_starter_id
        LEFT JOIN pitcher_season_features sa ON w.game_pk = sa.game_pk AND sa.pitcher = s.away_starter_id
        LEFT JOIN pitcher_features ph ON w.game_pk = ph.game_pk AND ph.pitcher = s.home_starter_id
        LEFT JOIN pitcher_features pa ON w.game_pk = pa.game_pk AND pa.pitcher = s.away_starter_id
        LEFT JOIN team_offense_rolling th ON w.game_pk = th.game_pk AND w.home_team = th.batting_team
        LEFT JOIN team_offense_rolling ta ON w.game_pk = ta.game_pk AND w.away_team = ta.batting_team
        LEFT JOIN bullpen_rolling bh ON w.game_pk = bh.game_pk AND w.home_team = bh.team
        LEFT JOIN bullpen_rolling ba ON w.game_pk = ba.game_pk AND w.away_team = ba.team
        LEFT JOIN pitcher_stuff hst ON w.game_pk = hst.game_pk AND hst.pitcher = s.home_starter_id
        LEFT JOIN pitcher_stuff ast ON w.game_pk = ast.game_pk AND ast.pitcher = s.away_starter_id
        LEFT JOIN team_contact_rolling ch ON w.game_pk = ch.game_pk AND w.home_team = ch.batting_team
        LEFT JOIN team_contact_rolling ca ON w.game_pk = ca.game_pk AND w.away_team = ca.batting_team
        LEFT JOIN team_hand_rolling hd ON w.game_pk = hd.game_pk AND w.away_team = hd.batting_team
        LEFT JOIN team_hand_rolling ha ON w.game_pk = ha.game_pk AND w.home_team = ha.batting_team
        LEFT JOIN bp_fatigue bf ON w.game_pk = bf.game_pk
        LEFT JOIN travel_fatigue tf ON w.game_pk = tf.game_pk
        LEFT JOIN closer_avail cl ON w.game_pk = cl.game_pk
        LEFT JOIN lineup_agg lh ON w.game_pk = lh.game_pk AND w.home_team = lh.batting_team
        LEFT JOIN lineup_agg la ON w.game_pk = la.game_pk AND w.away_team = la.batting_team
        LEFT JOIN lineup_ops_agg loh ON w.game_pk = loh.game_pk AND w.home_team = loh.batting_team
        LEFT JOIN lineup_ops_agg loa ON w.game_pk = loa.game_pk AND w.away_team = loa.batting_team
    """)

    n = con.execute("SELECT COUNT(*) FROM game_level").fetchone()[0]
    logger.info("game_level built: %d games", n)

    for tbl in (
        "game_winners", "starters", "venues", "rest_days",
        "pa_boundary", "pitcher_game_stats", "pitcher_shifted", "pitcher_rolling",
        "pitcher_shifted_season", "pitcher_season_rolling", "pitcher_5g_rolling",
        "pitcher_season_features", "pitcher_features",
        "team_offense_raw", "team_off_shifted", "team_offense_rolling",
        "bullpen_raw", "bullpen_shifted", "bullpen_rolling",
        "bp_daily", "bp_fatigue",
        "pitcher_stuff_raw", "pitcher_stuff",
        "team_contact_raw", "team_contact_shifted", "team_contact_rolling",
        "team_hand_raw", "team_hand_shifted", "team_hand_rolling",
        "batter_game_stats", "batter_shifted", "batter_rolling",
        "batter_league", "batter_ratings", "lineup_agg",
        "batter_hand_game", "batter_hand_shifted", "batter_hand_rolling", "lineup_ops_agg",
        "game_venue_tz", "team_travel_raw", "travel_seq", "travel_cross", "travel_fatigue",
        "late_relief", "rel_daily", "rel_cum", "rel_team", "team_closer_pit",
        "closer_avail",
    ):
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    gc.collect()


# ── PBP-level features ──────────────────────────────────────────────────────

def _build_pbp_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build pbp_level table via pure DuckDB SQL."""

    logger.info("Building PBP-level features...")

    con.execute("""
        CREATE TABLE pbp_level AS
        WITH game_feats AS (
            SELECT game_pk,
                   home_win, total_runs, venue,
                   rest_days_home, rest_days_away,
                   sp_era_home, sp_k9_home, sp_bb9_home, sp_whip_home, sp_fip_home, sp_xwoba_home,
                   sp_era_away, sp_k9_away, sp_bb9_away, sp_whip_away, sp_fip_away, sp_xwoba_away,
                   team_woba_30g_home, team_iso_30g_home, team_k_rate_30g_home, team_bb_rate_30g_home,
                   team_woba_30g_away, team_iso_30g_away, team_k_rate_30g_away, team_bb_rate_30g_away,
                   bullpen_whip_10g_home, bullpen_era_10g_home,
                   bullpen_whip_10g_away, bullpen_era_10g_away
            FROM game_level
        )
        SELECT
            p.game_pk, p.game_date, p.game_type, p.home_team, p.away_team,
            p.inning, p.inning_topbot, p.outs_when_up, p.balls, p.strikes,
            p.on_1b, p.on_2b, p.on_3b, p.at_bat_number, p.pitch_number,
            p.pitcher, p.batter, p.p_throws, p.stand,
            p.pitch_type, p.release_speed, p.description, p.events,
            p.barrel, p.hard_contact, p.launch_speed, p.launch_angle,
            p.estimated_woba_using_speedangle, p.estimated_ba_using_speedangle,
            p.zone, p.home_score, p.away_score, p.spin_rate,
            p.woba_value, p.babip_value, p.iso_value,
            p.delta_home_win_exp, p.delta_run_exp, p.player_name,
            p.hit_distance_sc, p.release_pos_x, p.release_pos_z,
            p.release_spin_rate, p.release_extension, p.pfx_x, p.pfx_z,
            -- Situational
            (COALESCE(p.on_1b IS NOT NULL, FALSE)::INT
             + COALESCE(p.on_2b IS NOT NULL, FALSE)::INT
             + COALESCE(p.on_3b IS NOT NULL, FALSE)::INT) AS bases_loaded,
            (COALESCE(p.on_2b IS NOT NULL, FALSE)::INT
             + COALESCE(p.on_3b IS NOT NULL, FALSE)::INT) AS runners_in_scoring_position,
            CASE WHEN (COALESCE(p.on_2b IS NOT NULL, FALSE)::INT
                       + COALESCE(p.on_3b IS NOT NULL, FALSE)::INT) > 0
                 THEN TRUE ELSE FALSE END AS is_risp,
            COALESCE(p.home_score, 0) - COALESCE(p.away_score, 0) AS score_diff,
            CASE WHEN p.inning_topbot = 'Top' THEN p.away_team ELSE p.home_team END AS batting_team,
            ROW_NUMBER() OVER (
                PARTITION BY p.game_pk, p.at_bat_number ORDER BY p.pitch_number
            ) AS ab_pitch_count,
            CEIL(p.at_bat_number / 9.0)::INT AS times_through_order,
            CASE WHEN p.stand = p.p_throws THEN 'same' ELSE 'opposite' END AS lr_matchup,
            p.barrel AS is_barrel,
            p.hard_contact AS is_hard_hit,
            p.launch_speed AS exit_velocity,
            p.launch_angle AS launch_angle_f,
            CASE
                WHEN p.pitch_type IN ('FF','FT','SI','FC','FS','FO') THEN 'fastball'
                WHEN p.pitch_type IN ('SL','CU','KC','CS','SV','WR') THEN 'breaking'
                WHEN p.pitch_type IN ('CH','EP','SC','KN','UN','PO') THEN 'offspeed'
                ELSE 'unknown'
            END AS pitch_category,
            -- Game-level features
            gf.home_win, gf.total_runs, gf.venue,
            gf.rest_days_home, gf.rest_days_away,
            gf.sp_era_home, gf.sp_k9_home, gf.sp_bb9_home, gf.sp_whip_home, gf.sp_fip_home, gf.sp_xwoba_home,
            gf.sp_era_away, gf.sp_k9_away, gf.sp_bb9_away, gf.sp_whip_away, gf.sp_fip_away, gf.sp_xwoba_away,
            gf.team_woba_30g_home, gf.team_iso_30g_home, gf.team_k_rate_30g_home, gf.team_bb_rate_30g_home,
            gf.team_woba_30g_away, gf.team_iso_30g_away, gf.team_k_rate_30g_away, gf.team_bb_rate_30g_away,
            gf.bullpen_whip_10g_home, gf.bullpen_era_10g_home,
            gf.bullpen_whip_10g_away, gf.bullpen_era_10g_away
        FROM pitches p
        LEFT JOIN game_feats gf ON p.game_pk = gf.game_pk
        ORDER BY p.game_date, p.game_pk, p.inning, p.at_bat_number, p.pitch_number
    """)

    n = con.execute("SELECT COUNT(*) FROM pbp_level").fetchone()[0]
    logger.info("pbp_level built: %d pitches", n)

    con.execute("DROP TABLE IF EXISTS pitches")
    con.execute("DROP TABLE IF EXISTS game_level")
    gc.collect()


# ── Diff features ───────────────────────────────────────────────────────────

# Statcast SLG park factors for 2025, indexed to 100 (league average).
# Source: Baseball Savant custom leaderboard (year=2025, type=park, xslg).
# >100 = hitters park, <100 = pitchers park.
PARK_FACTORS_SLG = {
    # Keys MUST match Statcast team codes (AZ/CWS, not ARI/CHW).
    "ARI": 101, "AZ": 101, "ATL": 97, "BAL": 99, "BOS": 103, "CHC": 100,
    "CHW": 97, "CWS": 97, "CIN": 103, "CLE": 98, "COL": 116, "DET": 98,
    "HOU": 99, "KC": 100, "LAA": 101, "LAD": 101, "MIA": 97,
    "MIL": 101, "MIN": 100, "NYM": 96, "NYY": 102, "OAK": 99,
    "PHI": 101, "PIT": 98, "SD": 95, "SF": 96, "SEA": 99,
    "STL": 97, "TB": 100, "TEX": 102, "TOR": 100, "WSH": 101,
    "ATH": 99,
}

# Dome/closed-roof flag: 1 if fixed dome, 0 if open-air.
# Prevents the model from hallucinating weather impacts indoors.
DOME_STATUS = {
    # Keys MUST match Statcast team codes (AZ/CWS, not ARI/CHW).
    # 1 = roof typically closed (fixed or retractable): ARI/HOU/MIA/MIL/
    # SEA/TB/TEX/TOR.  Citi Field (NYM) and the A's parks (SAC/OAK) are
    # OPEN-AIR — previously mislabeled 1, which nulled weather features.
    "ARI": 1, "AZ": 1, "ATL": 0, "BAL": 0, "BOS": 0, "CHC": 0,
    "CHW": 0, "CWS": 0, "CIN": 0, "CLE": 0, "COL": 0, "DET": 0,
    "HOU": 1, "KC": 0, "LAA": 0, "LAD": 0, "MIA": 1,
    "MIL": 1, "MIN": 1, "NYM": 0, "NYY": 0, "OAK": 0,
    "PHI": 0, "PIT": 0, "SD": 0, "SF": 0, "SEA": 1,
    "STL": 0, "TB": 1, "TEX": 1, "TOR": 1, "WSH": 0,
    "ATH": 0,
}

# Approximate home-plate UTC offsets per stadium (standard-time style).
# Used for travel fatigue: a crossing is counted each time consecutive
# games are played in different time zones within the trailing window.
# Cap for rest_days: days since a team's previous game are clipped here so
# All-Star breaks and long layoffs don't fabricate outliers (season openers
# carry NULL instead — see the rest_days table above).
REST_DAYS_CAP = 6

# The 30 team codes exactly as they appear in Statcast data.  Lookup dicts
# below MUST cover every one of these — an unmatched code silently degrades
# to a default (offset 0 / NaN) instead of failing loudly.
REAL_TEAM_CODES = frozenset({
    "ATH", "AZ", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS",
    "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
})

TEAM_TZ_OFFSETS = {
    # Keys MUST match Statcast team codes: AZ/CWS/LAA/NYM (not ARI/CHW) —
    # an unmatched code silently maps to offset 0.
    # Eastern (-5)
    "NYY": -5, "NYM": -5, "BOS": -5, "BAL": -5, "TB": -5, "TOR": -5,
    "WSH": -5, "ATL": -5, "MIA": -5, "PHI": -5, "PIT": -5,
    "CLE": -5, "DET": -5, "CIN": -5,
    # Central (-6)
    "CHC": -6, "CWS": -6, "KC": -6, "MIN": -6,
    "HOU": -6, "TEX": -6, "MIL": -6, "STL": -6,
    # Mountain (-7)
    "COL": -7, "AZ": -7,
    # Pacific (-8)
    "LAD": -8, "SD": -8, "SF": -8, "SEA": -8, "LAA": -8,
    "ATH": -8,
}

# Shrinkage weight for early-season win% smoothing (feature 2):
# smoothed = (wins + K/2) / (games + K) -> exactly .500 at game 0.
WIN_PCT_SHRINKAGE_GAMES = 30.0


def _smoothed_win_pct(wins: pd.Series, losses: pd.Series) -> pd.Series:
    """Win pct shrunk toward .500 by games played (early-season smoothing).

    (wins + K/2) / (games + K): equals exactly 0.500 before a team plays,
    heavily smoothed to .500 early season, converges to the raw win pct
    as the season matures.
    """
    wins = pd.to_numeric(wins, errors="coerce")
    losses = pd.to_numeric(losses, errors="coerce")
    games = (wins + losses).clip(lower=0)
    return (wins + 0.5 * WIN_PCT_SHRINKAGE_GAMES) / (games + WIN_PCT_SHRINKAGE_GAMES)



# Historical column-name aliases: canonical raw name -> alternate names that
# have appeared for the same underlying stat in different ingestion eras.
# add_diff_features resolves raw inputs through these so renamed columns are
# still sourced; anything that remains missing yields NaN (never 0).
RAW_COLUMN_ALIASES: dict[str, list[str]] = {
    "sp_xwoba_vs_l_home": ["sp_xwoba_vs_l_home", "sp_xwoba_l_home"],
    "sp_xwoba_vs_l_away": ["sp_xwoba_vs_l_away", "sp_xwoba_l_away"],
    "bullpen_ip_3d_home": ["bullpen_ip_3d_home", "bullpen_innings_3d_home"],
    "bullpen_ip_3d_away": ["bullpen_ip_3d_away", "bullpen_innings_3d_away"],
    "lineup_ops_vs_l_home": ["lineup_ops_vs_l_home", "lineup_ops_l_home"],
    "lineup_ops_vs_l_away": ["lineup_ops_vs_l_away", "lineup_ops_l_away"],
    "lineup_ops_vs_r_home": ["lineup_ops_vs_r_home", "lineup_ops_r_home"],
    "lineup_ops_vs_r_away": ["lineup_ops_vs_r_away", "lineup_ops_r_away"],
    "time_zones_crossed_last_3d_home": ["time_zones_crossed_last_3d_home",
                                        "travel_zones_crossed_last_3d_home"],
    "time_zones_crossed_last_3d_away": ["time_zones_crossed_last_3d_away",
                                        "travel_zones_crossed_last_3d_away"],
}


def add_diff_features(
    game_df: pd.DataFrame,
    weather_data: dict | None = None,
) -> pd.DataFrame:
    """Compute all 35 model features from the raw home/away columns.

    Exact feature layout (order matters — mirrors the spec sheet):

         1. is_home            always 1 (anchors baseline home-field edge)
         2. win_pct_diff       home_win_pct − away_win_pct
                               (smoothed to 0.500 if early season)
         3. elo_diff           home_elo − away_elo
         4. rest_days_diff     rest_days_home − rest_days_away
         5. sp_era_diff        home_sp_era − away_sp_era
                               (true season-to-date ERA, per-season)
         6. sp_era_5g_diff     home_sp_era_5g − away_sp_era_5g
                               (last 5 starts, across seasons)
         7. sp_k9_diff         home_sp_k9 − away_sp_k9
                               (true season-to-date K/9, per-season)
         8. sp_k9_5g_diff      home_sp_k9_5g − away_sp_k9_5g
                               (last 5 starts, across seasons)
         9. sp_fbvelo_diff     home_sp_fbvelo_3g − away_sp_fbvelo_3g
        10. sp_fbpct_diff      home_sp_fbpct_3g − away_sp_fbpct_3g
        11. sp_whiff_diff      home_sp_whiff_3g − away_sp_whiff_3g
        12. sp_xwoba_diff      home_sp_xwoba − away_sp_xwoba
                               (trailing 6-start xwOBA allowed — NOT season-to-date)
        13. sp_xwoba_vs_l_diff home_sp_xwoba_vs_l − away_sp_xwoba_vs_l
                               (current-season-to-date xwOBA vs LHB)
        14. lineup_woba_mean_diff
        15. lineup_woba_top3_diff
        16. lineup_woba_std_diff
        17. woba_30g_diff      home_woba_30g − away_woba_30g
        18. bullpen_whip_diff  home_bullpen_whip_10g − away_bullpen_whip_10g
        19. bullpen_whip_3g_diff
        20. bullpen_pitches_diff  (home_bullpen_pitches_3d − away_…)
        21. bullpen_ip_diff       (home_bullpen_ip_3d − away_…)
        22. team_barrel_diff   (trailing 15g barrel rate)
        23. team_hardhit_diff  (trailing 15g hard-hit rate)
        24. team_exitvelo_diff (trailing 15g avg exit velo)
        25. lineup_handedness_matchup_advantage
                               home_lineup_ops_vs_starter_hand
                               − away_lineup_ops_vs_starter_hand
                               (replaces opp_lefty_share_diff)
        26. travel_fatigue_diff
                               home_time_zones_crossed_last_3d
                               − away_time_zones_crossed_last_3d
        27. closer_availability_diff
                               home_closer_available − away_closer_available
        28. dome_is_neutral    binary flag (1 fixed dome/closed roof)
        29. park_factor_slug_diff  home_park_slug_factor × lineup_woba_top3_diff
        30. wind_advantage_flyball_factor
                               wind_direction_multiplier (Out=1, In=-1, Dome=0)
                               × sp_era_diff
        31. air_density_velocity_boost  stadium_air_density × sp_fbvelo_diff
        32. bullpen_meltdown_risk       bullpen_pitches_diff × bullpen_whip_diff
        33. pitcher_regression_indicator  sp_fbvelo_diff × sp_era_5g_diff
        34. lineup_depth_multiplier      lineup_woba_mean_diff × lineup_woba_top3_diff
        35. ace_efficiency_factor        sp_k9_5g_diff × sp_whiff_diff

    All diff features follow the convention: home − away (positive = home
    advantage).  Interaction features (29–35) are built from the diff
    features, so the model sees relative strengths directly.

    Args:
        game_df: DataFrame with raw home/away columns.
        weather_data: Optional dict keyed by game_id → weather dict with
            air_density and wind_multiplier (from weather.fetch_day_weather).
            When provided, features 30–31 use real weather data.

    Returns a copy of game_df with the 35 new columns appended.
    """
    df = game_df.copy()
    n = len(df)
    logger.info("Computing %d diff features for %d games...", 35, n)

    def _resolve_col(col: str) -> str | None:
        """First present column for a canonical raw name (aliases included)."""
        for name in [col, *RAW_COLUMN_ALIASES.get(col, [])]:
            if name in df.columns:
                return name
        return None

    def _diff(out: str, h_col: str, a_col: str) -> None:
        """home − away diff.

        Missing observations are NULL (NaN) — never a fabricated 0.  Column
        names may vary across ingestion eras; aliases are tried first.
        """
        h = _resolve_col(h_col)
        a = _resolve_col(a_col)
        if h is None or a is None:
            df[out] = np.nan
        else:
            df[out] = (pd.to_numeric(df[h], errors="coerce")
                       - pd.to_numeric(df[a], errors="coerce"))

    # ── 1. is_home (always 1 — anchors the baseline home-field advantage)
    df["is_home"] = 1.0

    # ── 2. win_pct_diff: home_win_pct − away_win_pct
    # Smoothed to 0.500 if early season: when W/L counts are available the
    # raw rates are shrunk toward .500 by games played; otherwise fall back
    # to the precomputed win-pct columns.
    has_records = all(c in df.columns for c in
                      ("home_wins", "home_losses", "away_wins", "away_losses"))
    if has_records:
        df["win_pct_diff"] = (_smoothed_win_pct(df["home_wins"], df["home_losses"])
                              - _smoothed_win_pct(df["away_wins"], df["away_losses"]))
    elif "home_win_pct" in df.columns and "away_win_pct" in df.columns:
        df["win_pct_diff"] = (pd.to_numeric(df["home_win_pct"], errors="coerce")
                              - pd.to_numeric(df["away_win_pct"], errors="coerce"))
    else:
        df["win_pct_diff"] = np.nan
        logger.warning("win_pct_diff: record/win-pct columns missing, set to NaN")

    # ── 3–24. Straight home − away diffs (exact spec-sheet order)
    simple_diffs = [
        ("elo_diff", "home_elo", "away_elo"),                                    # 3
        ("rest_days_diff", "rest_days_home", "rest_days_away"),                  # 4
        ("sp_era_diff", "sp_era_home", "sp_era_away"),                           # 5
        ("sp_era_5g_diff", "sp_era_5g_home", "sp_era_5g_away"),                   # 6
        ("sp_k9_diff", "sp_k9_home", "sp_k9_away"),                              # 7
        ("sp_k9_5g_diff", "sp_k9_5g_home", "sp_k9_5g_away"),                     # 8
        ("sp_fbvelo_diff", "sp_fbvelo_3g_home", "sp_fbvelo_3g_away"),            # 9
        ("sp_fbpct_diff", "sp_fbpct_3g_home", "sp_fbpct_3g_away"),               # 10
        ("sp_whiff_diff", "sp_whiff_3g_home", "sp_whiff_3g_away"),               # 11
        ("sp_xwoba_diff", "sp_xwoba_home", "sp_xwoba_away"),                     # 12
        ("sp_xwoba_vs_l_diff", "sp_xwoba_vs_l_home", "sp_xwoba_vs_l_away"),      # 13
        ("lineup_woba_mean_diff", "lineup_woba_mean_home", "lineup_woba_mean_away"),  # 14
        ("lineup_woba_top3_diff", "lineup_woba_top3_home", "lineup_woba_top3_away"),  # 15
        ("lineup_woba_std_diff", "lineup_woba_std_home", "lineup_woba_std_away"),     # 16
        ("woba_30g_diff", "woba_30g_home", "woba_30g_away"),                     # 17
        ("bullpen_whip_diff", "bullpen_whip_10g_home", "bullpen_whip_10g_away"),      # 18
        ("bullpen_whip_3g_diff", "bullpen_whip_3g_home", "bullpen_whip_3g_away"),     # 19
        ("bullpen_pitches_diff", "bullpen_pitches_3d_home", "bullpen_pitches_3d_away"),  # 20
        ("bullpen_ip_diff", "bullpen_ip_3d_home", "bullpen_ip_3d_away"),         # 21
        ("team_barrel_diff", "team_barrel_15g_home", "team_barrel_15g_away"),    # 22
        ("team_hardhit_diff", "team_hardhit_15g_home", "team_hardhit_15g_away"), # 23
        ("team_exitvelo_diff", "team_exitvelo_15g_home", "team_exitvelo_15g_away"),  # 24
    ]
    for out, h_col, a_col in simple_diffs:
        _diff(out, h_col, a_col)

    # ── 25. lineup_handedness_matchup_advantage
    # Each lineup's OPS against the hand of the starter it faces (assembled
    # in SQL from per-batter L/R splits + tonight's starter throwing hand).
    # Replaces the old opp_lefty_share_diff.
    _diff("lineup_handedness_matchup_advantage",
          "lineup_ops_vs_starter_hand_home", "lineup_ops_vs_starter_hand_away")

    # ── 26. travel_fatigue_diff (new schedule metric)
    _diff("travel_fatigue_diff",
          "time_zones_crossed_last_3d_home", "time_zones_crossed_last_3d_away")

    # ── 27. closer_availability_diff (new high-leverage metric)
    _diff("closer_availability_diff",
          "closer_available_home", "closer_available_away")

    # ── 28. dome_is_neutral: binary flag (1 if fixed dome/closed roof)
    home_team = df["home_team"].astype(str).str.upper().str.strip()
    df["dome_is_neutral"] = home_team.map(DOME_STATUS).astype(float)  # NaN = unknown

    # ── 29. park_factor_slug_diff: home_park_slug_factor × lineup_woba_top3_diff
    # Maps out when a power-heavy lineup gets to exploit a small ballpark.
    pf_raw = home_team.map(PARK_FACTORS_SLG).astype(float)  # NaN = unknown park
    pf = (pf_raw - 100.0) / 100.0  # center at 0: +0.05 = 5% more SLG than avg
    df["park_factor_slug_diff"] = pf * pd.to_numeric(
        df["lineup_woba_top3_diff"], errors="coerce")

    # ── 30. wind_advantage_flyball_factor
    # wind_direction_multiplier(Out=1, In=-1, Dome=0) × sp_era_diff.
    # Flags when mistake-prone pitchers are at risk of wind-blown home runs.
    # NULL when weather is missing or the SP diff is missing — never 0.
    df["wind_advantage_flyball_factor"] = np.nan

    # ── 31. air_density_velocity_boost: stadium_air_density × sp_fbvelo_diff
    # Adjusts for how cold or thin air alters raw pitching velocity.
    # NULL when weather is missing or the SP diff is missing — never 0.
    df["air_density_velocity_boost"] = np.nan

    # Standard sea-level air density ≈ 1.225 kg/m³ — center so neutral = 0
    SEA_LEVEL_RHO = 1.225
    if weather_data:
        wind_mults = []
        air_dens = []
        dome_mask = []
        for _, row in df.iterrows():
            gid = row.get("game_id", "")
            w = weather_data.get(gid, {}) if isinstance(weather_data, dict) else {}
            dome = row.get("dome_is_neutral")
            dome_mask.append(pd.notna(dome) and float(dome) == 1)
            if w.get("available"):
                wind_mults.append(w.get("wind_multiplier", np.nan))
                air_dens.append(w.get("air_density", np.nan))
            else:
                wind_mults.append(np.nan)
                air_dens.append(np.nan)
        wm = pd.Series(wind_mults, index=df.index, dtype="float64")
        ad = pd.Series(air_dens, index=df.index, dtype="float64")
        dome = pd.Series(dome_mask, index=df.index)
        df["wind_advantage_flyball_factor"] = (
            wm * pd.to_numeric(df["sp_era_diff"], errors="coerce"))
        df["air_density_velocity_boost"] = (
            (ad - SEA_LEVEL_RHO) * pd.to_numeric(df["sp_fbvelo_diff"], errors="coerce"))
        # Dome games: wind and air density are genuinely neutral indoors —
        # a real, valid 0 (not a fabricated default) — but only when the
        # underlying diff input is present; a dome game with a missing ERA
        # diff stays NULL (never a fabricated 0).
        _era_ok = pd.to_numeric(df["sp_era_diff"], errors="coerce").notna()
        _velo_ok = pd.to_numeric(df["sp_fbvelo_diff"], errors="coerce").notna()
        df.loc[dome & _era_ok, "wind_advantage_flyball_factor"] = 0.0
        df.loc[dome & _velo_ok, "air_density_velocity_boost"] = 0.0
        n_weather = int((wm.notna() & ad.notna()).sum())
        logger.info("Weather applied to %d/%d games", n_weather, len(df))
    else:
        # No weather fetched: dome games (KNOWN dome status) get a valid
        # neutral 0 — but only where the underlying diff input exists; every
        # other game stays NULL until real weather exists.
        dome = df["dome_is_neutral"] == 1
        _era_ok = pd.to_numeric(df["sp_era_diff"], errors="coerce").notna()
        _velo_ok = pd.to_numeric(df["sp_fbvelo_diff"], errors="coerce").notna()
        df.loc[dome & _era_ok, "wind_advantage_flyball_factor"] = 0.0
        df.loc[dome & _velo_ok, "air_density_velocity_boost"] = 0.0

    # ── 32. bullpen_meltdown_risk: bullpen_pitches_diff × bullpen_whip_diff
    # Overworked + low quality bullpen = elevated meltdown risk.
    df["bullpen_meltdown_risk"] = df["bullpen_pitches_diff"] * df["bullpen_whip_diff"]

    # ── 33. pitcher_regression_indicator: sp_fbvelo_diff × sp_era_5g_diff
    # Physical velocity drop vs surface-level last-5-start ERA results —
    # flags regression candidates before the ERA fully catches up to the stuff.
    df["pitcher_regression_indicator"] = df["sp_fbvelo_diff"] * df["sp_era_5g_diff"]

    # ── 34. lineup_depth_multiplier: lineup_woba_mean_diff × lineup_woba_top3_diff
    # Star power vs complete batting order depth.
    df["lineup_depth_multiplier"] = df["lineup_woba_mean_diff"] * df["lineup_woba_top3_diff"]

    # ── 35. ace_efficiency_factor: sp_k9_5g_diff × sp_whiff_diff
    # Last-5-start strikeout volume driven by raw swing-and-miss stuff —
    # the true-ace differentiator.
    df["ace_efficiency_factor"] = df["sp_k9_5g_diff"] * df["sp_whiff_diff"]

    logger.info("Diff features complete: %d columns added", 35)
    return df

# ── Public API ──────────────────────────────────────────────────────────────

def _mem_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def build_features(
    pitches_path: str | Path,
    output_dir: str | Path = ".",
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run pure-DuckDB feature engineering on a pitches Parquet file.

    Reads pitches.parquet, builds game_df + pbp_df via SQL, writes them to
    disk, and loads into pandas ONLY for model training.

    Args:
        pitches_path: Path to pitches.parquet (from ingestion.py).
        output_dir:   Where to write output Parquet files.
        validate:     Reserved for future validation checks.

    Returns:
        (game_df, pbp_df) as pandas DataFrames.
    """
    pitches_path = Path(pitches_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    game_out = out_dir / "game_level_features.parquet"
    pbp_out = out_dir / "pbp_level_features.parquet"

    logger.info("=== DuckDB Feature Engineering ===")
    logger.info("[MEM] Start: %.0f MB", _mem_mb())

    con = _connect(pitches_path)
    logger.info("[MEM] After DuckDB load: %.0f MB", _mem_mb())

    try:
        _build_game_level(con)
        logger.info("[MEM] After game_level: %.0f MB", _mem_mb())

        # Export game_level BEFORE pbp_level (pbp_level drops game_level)
        con.execute(f"COPY (SELECT * FROM game_level) TO '{game_out}' (FORMAT PARQUET)")
        logger.info("Exported game_level: %.1f MB", game_out.stat().st_size / 1e6)

        _build_pbp_level(con)
        logger.info("[MEM] After pbp_level: %.0f MB", _mem_mb())

        con.execute(f"COPY (SELECT * FROM pbp_level) TO '{pbp_out}' (FORMAT PARQUET)")
        logger.info("Exported pbp_level: %.1f MB", pbp_out.stat().st_size / 1e6)

        con.execute("DROP TABLE IF EXISTS pbp_level")
        gc.collect()
    finally:
        con.close()

    logger.info("[MEM] After DuckDB close: %.0f MB", _mem_mb())

    # Phase 3: Load into pandas (model-ready only)
    game_df = pd.read_parquet(game_out)
    pbp_df = pd.read_parquet(pbp_out)

    # ── Official results overlay ────────────────────────────────────────
    # Statcast pitch rows derive scores from the last cached pitch — a
    # partial crawl freezes wrong finals forever, and mid-game snapshots
    # become bogus training labels.  The MLB StatsAPI schedule endpoint
    # gives authoritative scores + game state per game_pk, so we overlay
    # them here: final scores are corrected and non-final labels are nulled
    # so the model never trains on a game that hasn't actually finished.
    try:
        from results import fetch_mlb_results, apply_official_results
        if len(game_df) > 0 and "game_date" in game_df.columns:
            gd = game_df["game_date"].dropna()
            start = pd.to_datetime(gd.min()).date()
            end = pd.to_datetime(gd.max()).date()
            res = fetch_mlb_results(start, end)
            if not res.empty:
                game_df = apply_official_results(game_df, res)
    except Exception as exc:
        logger.warning("Official results overlay failed — keeping pitch-derived scores: %s", exc)

    for df in [game_df, pbp_df]:
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            if df[col].max() < 32767 and df[col].min() >= -32768:
                df[col] = df[col].astype("int16")

    logger.info("[MEM] After pandas load: %.0f MB", _mem_mb())

    # ── Diff features: 34 model inputs from home/away pairs ──────────
    game_df = add_diff_features(game_df)

    logger.info("=== Complete: %d games, %d pitches ===", len(game_df), len(pbp_df))

    return game_df, pbp_df
