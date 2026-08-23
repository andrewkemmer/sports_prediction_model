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
            SELECT game_pk, pitcher AS home_starter_id,
                   ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Top'
        ),
        first_pa_bot AS (
            SELECT game_pk, pitcher AS away_starter_id,
                   ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Bot'
        )
        SELECT DISTINCT p.game_pk,
               CAST(p.game_date AS DATE) AS game_date,
               p.home_team, p.away_team,
               t.home_starter_id, b.away_starter_id
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

    # 4. Rest days (days since each team's previous game)
    con.execute("""
        CREATE TABLE rest_days AS
        WITH team_games AS (
            SELECT DISTINCT game_date, game_pk, home_team AS team FROM pitches
            UNION
            SELECT DISTINCT game_date, game_pk, away_team AS team FROM pitches
        ),
        with_prev AS (
            SELECT *, LAG(game_date) OVER (PARTITION BY team ORDER BY game_date) AS prev_date
            FROM team_games
        )
        SELECT game_pk, team,
               CAST(game_date AS DATE) - CAST(prev_date AS DATE) AS rest_days
        FROM with_prev
    """)

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

    con.execute("""
        CREATE TABLE pitcher_features AS
        SELECT game_date, game_pk, pitcher,
            _roll_runs / NULLIF(_roll_ip, 0) * 9.0 AS sp_era_30g,
            _roll_ks / NULLIF(_roll_ip, 0) * 9.0 AS sp_k9_30g,
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
            (SUM(_s_bbs) OVER w + SUM(_s_hits) OVER w)
                / NULLIF(SUM(_s_ip) OVER w, 0) AS bullpen_whip_10g,
            SUM(_s_runs) OVER w / NULLIF(SUM(_s_ip) OVER w, 0) * 9.0 AS bullpen_era_10g
        FROM bullpen_shifted
        WINDOW w AS (PARTITION BY team ORDER BY game_date
                     ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
    """)

    # 7b. Starter stuff trends — last-3-start fastball velo/mix, whiff rate,
    # and season-to-date xwOBA allowed vs left/right-handed batters.
    # All windows are LAG-shifted so the current game is excluded (point-in-time).
    con.execute(f"""
        CREATE TABLE pitcher_stuff_raw AS
        WITH per_start AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk, pitcher,
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
            LAG(xwoba_vs_l, 1) OVER w AS _s_xl,
            LAG(xwoba_vs_r, 1) OVER w AS _s_xr
        FROM per_start
        WINDOW w AS (PARTITION BY pitcher ORDER BY game_date)
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
               wall AS (PARTITION BY pitcher ORDER BY game_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

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

    # 8. Assemble game_level via LEFT JOINs
    con.execute("""
        CREATE TABLE game_level AS
        SELECT
            w.game_pk, w.game_date, w.home_team, w.away_team,
            w.home_score, w.away_score, w.home_win, w.total_runs,
            s.home_starter_id, s.away_starter_id, v.venue,
            rh.rest_days AS rest_days_home, ra.rest_days AS rest_days_away,
            ph.sp_era_30g AS sp_era_home, ph.sp_k9_30g AS sp_k9_home,
            ph.sp_bb9_30g AS sp_bb9_home, ph.sp_whip_30g AS sp_whip_home,
            ph.sp_fip_30g AS sp_fip_home, ph.sp_xwoba_30g AS sp_xwoba_home,
            pa.sp_era_30g AS sp_era_away, pa.sp_k9_30g AS sp_k9_away,
            pa.sp_bb9_30g AS sp_bb9_away, pa.sp_whip_30g AS sp_whip_away,
            pa.sp_fip_30g AS sp_fip_away, pa.sp_xwoba_30g AS sp_xwoba_away,
            th.team_woba_30g AS team_woba_30g_home, th.team_iso_30g AS team_iso_30g_home,
            th.team_k_rate_30g AS team_k_rate_30g_home, th.team_bb_rate_30g AS team_bb_rate_30g_home,
            ta.team_woba_30g AS team_woba_30g_away, ta.team_iso_30g AS team_iso_30g_away,
            ta.team_k_rate_30g AS team_k_rate_30g_away, ta.team_bb_rate_30g AS team_bb_rate_30g_away,
            bh.bullpen_whip_10g AS bullpen_whip_10g_home, bh.bullpen_era_10g AS bullpen_era_10g_home,
            ba.bullpen_whip_10g AS bullpen_whip_10g_away, ba.bullpen_era_10g AS bullpen_era_10g_away,
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
        LEFT JOIN lineup_agg lh ON w.game_pk = lh.game_pk AND w.home_team = lh.batting_team
        LEFT JOIN lineup_agg la ON w.game_pk = la.game_pk AND w.away_team = la.batting_team
    """)

    n = con.execute("SELECT COUNT(*) FROM game_level").fetchone()[0]
    logger.info("game_level built: %d games", n)

    for tbl in (
        "game_winners", "starters", "venues", "rest_days",
        "pa_boundary", "pitcher_game_stats", "pitcher_shifted", "pitcher_rolling", "pitcher_features",
        "team_offense_raw", "team_off_shifted", "team_offense_rolling",
        "bullpen_raw", "bullpen_shifted", "bullpen_rolling",
        "bp_daily", "bp_fatigue",
        "pitcher_stuff_raw", "pitcher_stuff",
        "team_contact_raw", "team_contact_shifted", "team_contact_rolling",
        "team_hand_raw", "team_hand_shifted", "team_hand_rolling",
        "batter_game_stats", "batter_shifted", "batter_rolling",
        "batter_league", "batter_ratings", "lineup_agg",
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
    "ARI": 101, "ATL": 97, "BAL": 99, "BOS": 103, "CHC": 100,
    "CHW": 97, "CIN": 103, "CLE": 98, "COL": 116, "DET": 98,
    "HOU": 99, "KC": 100, "LAA": 101, "LAD": 101, "MIA": 97,
    "MIL": 101, "MIN": 100, "NYM": 96, "NYY": 102, "OAK": 99,
    "PHI": 101, "PIT": 98, "SD": 95, "SF": 96, "SEA": 99,
    "STL": 97, "TB": 100, "TEX": 102, "TOR": 100, "WSH": 101,
    "ATH": 99,
}

# Dome/closed-roof flag: 1 if fixed dome, 0 if open-air.
# Prevents the model from hallucinating weather impacts indoors.
DOME_STATUS = {
    "ARI": 1, "ATL": 0, "BAL": 0, "BOS": 0, "CHC": 0,
    "CHW": 0, "CIN": 0, "CLE": 0, "COL": 0, "DET": 0,
    "HOU": 1, "KC": 0, "LAA": 0, "LAD": 0, "MIA": 1,
    "MIL": 1, "MIN": 1, "NYM": 1, "NYY": 0, "OAK": 1,
    "PHI": 0, "PIT": 0, "SD": 0, "SF": 0, "SEA": 1,
    "STL": 0, "TB": 1, "TEX": 1, "TOR": 1, "WSH": 0,
    "ATH": 1,
}



def add_diff_features(
    game_df: pd.DataFrame,
    weather_data: dict | None = None,
) -> pd.DataFrame:
    """Compute all 34 model features from the raw home/away columns.

    All diff features follow the convention: home − away (positive = home
    advantage).  Interaction features (26–34) are built from the diff
    features, so the model sees relative strengths directly.

    Args:
        game_df: DataFrame with raw home/away columns.
        weather_data: Optional dict keyed by game_id → weather dict with
            air_density and wind_multiplier (from weather.fetch_day_weather).
            When provided, features 28-29 use real weather data.

    Returns a copy of game_df with the 34 new columns appended.
    """
    df = game_df.copy()
    n = len(df)
    logger.info("Computing %d diff features for %d games...", 34, n)

    # ── 1. is_home (always 1 — anchors the baseline home-field advantage)
    df["is_home"] = 1.0

    # ── 2. win_pct_diff: home_win_pct − away_win_pct
    if "home_win_pct" in df.columns and "away_win_pct" in df.columns:
        df["win_pct_diff"] = pd.to_numeric(df["home_win_pct"], errors="coerce") \
            - pd.to_numeric(df["away_win_pct"], errors="coerce")
    else:
        df["win_pct_diff"] = 0.0
        logger.warning("win_pct_diff: home_win_pct/away_win_pct columns missing, set to 0")

    # ── 3. elo_diff: home_elo − away_elo
    if "home_elo" in df.columns and "away_elo" in df.columns:
        df["elo_diff"] = pd.to_numeric(df["home_elo"], errors="coerce") \
            - pd.to_numeric(df["away_elo"], errors="coerce")
    else:
        df["elo_diff"] = 0.0
        logger.warning("elo_diff: home_elo/away_elo columns missing, set to 0")

    # ── 4. rest_days_diff: rest_days_home − rest_days_away
    if "rest_days_home" in df.columns and "rest_days_away" in df.columns:
        df["rest_days_diff"] = pd.to_numeric(df["rest_days_home"], errors="coerce") \
            - pd.to_numeric(df["rest_days_away"], errors="coerce")
    else:
        df["rest_days_diff"] = 0.0

    # ── 5–8. Starting pitcher diffs (career-level metrics from DuckDB)
    for pair in [
        ("sp_era_home", "sp_era_away", "sp_era_diff"),
        ("sp_k9_home", "sp_k9_away", "sp_k9_diff"),
    ]:
        h, a, out = pair
        if h in df.columns and a in df.columns:
            df[out] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
        else:
            df[out] = 0.0

    # ── 9–11. SP stuff diffs (trailing 3-game fastball metrics)
    for pair in [
        ("sp_fbvelo_3g_home", "sp_fbvelo_3g_away", "sp_fbvelo_diff"),
        ("sp_fbpct_3g_home", "sp_fbpct_3g_away", "sp_fbpct_diff"),
        ("sp_whiff_3g_home", "sp_whiff_3g_away", "sp_whiff_diff"),
    ]:
        h, a, out = pair
        if h in df.columns and a in df.columns:
            df[out] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
        else:
            df[out] = 0.0

    # ── 12–13. SP xwOBA diffs (season-to-date contact quality)
    for pair in [
        ("sp_xwoba_home", "sp_xwoba_away", "sp_xwoba_diff"),
        ("sp_xwoba_vs_l_home", "sp_xwoba_vs_l_away", "sp_xwoba_vs_l_diff"),
    ]:
        h, a, out = pair
        if h in df.columns and a in df.columns:
            df[out] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
        else:
            df[out] = 0.0

    # ── 14–16. Lineup wOBA diffs (mean, top-3 star power, dispersion)
    for pair in [
        ("lineup_woba_mean_home", "lineup_woba_mean_away", "lineup_woba_mean_diff"),
        ("lineup_woba_top3_home", "lineup_woba_top3_away", "lineup_woba_top3_diff"),
        ("lineup_woba_std_home", "lineup_woba_std_away", "lineup_woba_std_diff"),
    ]:
        h, a, out = pair
        if h in df.columns and a in df.columns:
            df[out] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
        else:
            df[out] = 0.0

    # ── 17. woba_30g_diff: team 30-game rolling wOBA
    if "woba_30g_home" in df.columns and "woba_30g_away" in df.columns:
        df["woba_30g_diff"] = pd.to_numeric(df["woba_30g_home"], errors="coerce") \
            - pd.to_numeric(df["woba_30g_away"], errors="coerce")
    else:
        df["woba_30g_diff"] = 0.0

    # ── 18–21. Bullpen diffs (WHIP 10g, WHIP 3g, pitches 3d, IP 3d)
    for pair in [
        ("bullpen_whip_10g_home", "bullpen_whip_10g_away", "bullpen_whip_diff"),
        ("bullpen_pitches_3d_home", "bullpen_pitches_3d_away", "bullpen_pitches_diff"),
        ("bullpen_ip_3d_home", "bullpen_ip_3d_away", "bullpen_ip_diff"),
    ]:
        h, a, out = pair
        if h in df.columns and a in df.columns:
            df[out] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
        else:
            df[out] = 0.0

    # 19. bullpen_whip_3g_diff — compute from 3-day pitches + IP if available
    # (the raw DuckDB output doesn't have a pre-built 3g WHIP column,
    # so derive it from the raw bullpen workload data if available)
    if "bullpen_whip_3g_diff" not in df.columns:
        df["bullpen_whip_3g_diff"] = 0.0  # placeholder until 3g WHIP is added to DuckDB

    # ── 22–24. Team contact form diffs (barrel%, hard-hit%, avg EV — trailing 15g)
    for pair in [
        ("team_barrel_15g_home", "team_barrel_15g_away", "team_barrel_diff"),
        ("team_hardhit_15g_home", "team_hardhit_15g_away", "team_hardhit_diff"),
        ("team_exitvelo_15g_home", "team_exitvelo_15g_away", "team_exitvelo_diff"),
    ]:
        h, a, out = pair
        if h in df.columns and a in df.columns:
            df[out] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
        else:
            df[out] = 0.0

    # ── 25. opp_lefty_share_diff: opposing lineup lefty share vs each starter
    if "opp_lefty_share_home" in df.columns and "opp_lefty_share_away" in df.columns:
        df["opp_lefty_share_diff"] = pd.to_numeric(df["opp_lefty_share_home"], errors="coerce") \
            - pd.to_numeric(df["opp_lefty_share_away"], errors="coerce")
    else:
        df["opp_lefty_share_diff"] = 0.0

    # ── 26. dome_is_neutral: binary flag (1 if fixed dome/closed roof)
    # Prevents models from hallucinating weather impacts indoors.
    home_team = df["home_team"].astype(str).str.upper().str.strip()
    df["dome_is_neutral"] = home_team.map(DOME_STATUS).fillna(0).astype(float)

    # ── 27. park_factor_slug_diff: park_factor × lineup_woba_top3_diff
    # Maps out when a power-heavy lineup gets to exploit a small ballpark.
    pf_raw = home_team.map(PARK_FACTORS_SLG).fillna(100).astype(float)
    pf = (pf_raw - 100.0) / 100.0  # center at 0: +0.05 = 5% more SLG than avg
    df["park_factor_slug_diff"] = pf * df["lineup_woba_top3_diff"]

    # ── 28. wind_advantage_flyball_factor
    # wind_direction_multiplier(Out=1, In=-1, Dome=0) × sp_era_diff.
    # Flags when mistake-prone pitchers are at risk of wind-blown home runs.
    df["wind_advantage_flyball_factor"] = 0.0
    # ── 29. air_density_velocity_boost
    # stadium_air_density × sp_fbvelo_diff. Adjusts for how cold or thin air
    # alters raw pitching velocity.
    df["air_density_velocity_boost"] = 0.0
    # Fill from weather data when available
    if weather_data:
        wind_mults = []
        air_dens = []
        for _, row in df.iterrows():
            gid = row.get("game_id", "")
            w = weather_data.get(gid, {})
            wind_mults.append(w.get("wind_multiplier", 0.0) if w.get("available") else 0.0)
            ad = w.get("air_density", np.nan) if w.get("available") else np.nan
            air_dens.append(ad if ad is not None and not np.isnan(ad) else np.nan)
        df["wind_advantage_flyball_factor"] = [
            wm * sp if not np.isnan(wm) and not np.isnan(sp) else 0.0
            for wm, sp in zip(wind_mults, df["sp_era_diff"].values)
        ]
        # Standard sea-level air density ≈ 1.225 kg/m³ — center so neutral = 0
        SEA_LEVEL_RHO = 1.225
        df["air_density_velocity_boost"] = [
            (ad - SEA_LEVEL_RHO) * sp if not np.isnan(ad) and not np.isnan(sp) else 0.0
            for ad, sp in zip(air_dens, df["sp_fbvelo_diff"].values)
        ]
        n_weather = sum(1 for gid in df["game_id"] if gid in weather_data and weather_data[gid].get("available"))
        logger.info("Weather applied to %d/%d games", n_weather, len(df))
    else:
        # Dome games always get 0
        df.loc[df["dome_is_neutral"] == 1, "wind_advantage_flyball_factor"] = 0.0
        df.loc[df["dome_is_neutral"] == 1, "air_density_velocity_boost"] = 0.0

    # ── 30. bullpen_meltdown_risk: bullpen_pitches_diff × bullpen_whip_diff
    # Overworked + low quality bullpen = elevated meltdown risk.
    df["bullpen_meltdown_risk"] = df["bullpen_pitches_diff"] * df["bullpen_whip_diff"]

    # ── 31. platoon_exploit_edge: opp_lefty_share_diff × sp_xwoba_vs_l_diff
    # Lineup matching against pitcher platoon flaw.
    df["platoon_exploit_edge"] = df["opp_lefty_share_diff"] * df["sp_xwoba_vs_l_diff"]

    # ── 32. pitcher_regression_indicator: sp_fbvelo_diff × sp_era_30g_diff
    # Physical velocity drop vs surface-level ERA results — flags regression candidates.
    df["pitcher_regression_indicator"] = df["sp_fbvelo_diff"] * df["sp_era_diff"]

    # ── 33. lineup_depth_multiplier: lineup_woba_mean_diff × lineup_woba_top3_diff
    # Star power vs complete batting order depth.
    df["lineup_depth_multiplier"] = df["lineup_woba_mean_diff"] * df["lineup_woba_top3_diff"]

    # ── 34. ace_efficiency_factor: sp_k9_30g_diff × sp_whiff_diff
    # High strikeout volume driven by raw stuff vs command — ace differentiator.
    df["ace_efficiency_factor"] = df["sp_k9_diff"] * df["sp_whiff_diff"]

    logger.info("Diff features complete: %d columns added", 34)
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
