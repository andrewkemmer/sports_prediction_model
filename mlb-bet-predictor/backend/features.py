"""
Pure DuckDB SQL feature engineering for MLB Statcast data.

ZERO pandas during feature engineering.  All rolling windows, shifted
features, groupby aggregations, and joins are DuckDB SQL window functions
operating on Parquet files on disk.

Architecture:
    pitches.parquet → DuckDB SQL → game_df.parquet + pbp_df.parquet

PIT compliance:
    All rolling metrics use ROWS BETWEEN N PRECEDING AND 1 PRECEDING
    (shifted rolling = no data leakage).  LAG() for previous-game context.
    ORDER BY game_date ensures chronological processing.
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


# ── Connection helper ───────────────────────────────────────────────────────

def _connect(pitches_path: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB, load pitches from Parquet, tune for low RAM."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = false")

    con.execute(f"CREATE TABLE pitches AS SELECT * FROM '{pitches_path}'")

    # Ensure game_date is DATE type
    con.execute("""
        ALTER TABLE pitches
        ALTER COLUMN game_date TYPE DATE
        USING CAST(game_date AS DATE)
    """)

    # Drop unused columns
    existing = {r[0] for r in con.execute("DESCRIBE pitches").fetchall()}
    for col in [
        "fielder_2", "fielder_3", "fielder_4", "fielder_5",
        "fielder_6", "fielder_7", "fielder_8", "fielder_9",
        "if_fielding_alignment", "of_fielding_alignment",
        "post_home_score", "post_away_score",
        "event", "type", "launch_speed_angle",
    ]:
        if col in existing:
            con.execute(f'ALTER TABLE pitches DROP COLUMN "{col}"')

    return con


# ── Game-level features ─────────────────────────────────────────────────────

def _build_game_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build game_level table via pure DuckDB SQL."""

    logger.info("Building game-level features via DuckDB SQL...")

    # 1. Game winners
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
               CASE WHEN home_score > away_score THEN 1.0 ELSE 0.0 END AS home_win,
               (COALESCE(home_score, 0) + COALESCE(away_score, 0)) AS total_runs
        FROM last_pitch WHERE rn = 1
    """)

    # 2. Starting pitchers
    con.execute("""
        CREATE TABLE starters AS
        WITH first_pa_top AS (
            SELECT game_pk, pitcher AS away_starter_id,
                   ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Top'
        ),
        first_pa_bot AS (
            SELECT game_pk, pitcher AS home_starter_id,
                   ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) AS rn
            FROM pitches WHERE inning = 1 AND inning_topbot = 'Bot'
        )
        SELECT DISTINCT p.game_pk,
               CAST(p.game_date AS DATE) AS game_date,
               p.home_team, p.away_team,
               t.away_starter_id, b.home_starter_id
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

    # 4. Rest days
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

    # 5. Pitcher rolling features (PIT-compliant: LAG first, then rolling)
    con.execute(f"""
        CREATE TABLE pitcher_game_stats AS
        WITH pa_events AS (
            SELECT game_date, game_pk, pitcher, events,
                   estimated_woba_using_speedangle AS xwoba_val,
                   barrel AS barrel_val, hard_contact AS hard_val
            FROM pitches
            WHERE events IN ({PA_END_EVENTS})
        ),
        game_agg AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk, pitcher,
                   COUNT(*) AS n_batters_faced,
                   COUNT(*) / 3.0 AS ip_approx,
                   SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS ks,
                   SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bbs,
                   SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbps,
                   SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END) AS hits_allowed,
                   SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs_allowed,
                   AVG(xwoba_val) AS xwoba,
                   AVG(barrel_val) AS barrel_rate,
                   AVG(hard_val) AS hard_contact_rate
            FROM pa_events
            GROUP BY CAST(game_date AS DATE), game_pk, pitcher
        )
        SELECT *, hits_allowed + bbs + hbps - hrs_allowed AS runs
        FROM game_agg
    """)

    # 5a. LAG all stats
    con.execute("""
        CREATE TABLE pitcher_shifted AS
        SELECT *,
            LAG(ip_approx, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_ip,
            LAG(runs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_runs,
            LAG(ks, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_ks,
            LAG(bbs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_bbs,
            LAG(hits_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hits,
            LAG(hrs_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hrs,
            LAG(hbps, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_hbps,
            LAG(xwoba, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _s_xwoba
        FROM pitcher_game_stats
    """)

    # 5b. Rolling SUM/AVG over lagged columns
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

    # 5c. Derive pitcher features
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
            (0.690 * bb + 0.722 * hbp + 0.878 * singles + 1.242 * doubles
             + 1.568 * triples + 2.007 * hrs) / NULLIF(n_pa - bb - hbp + bb + hbp, 0) AS team_woba_game,
            (doubles + 2 * triples + 3 * hrs) / NULLIF(n_pa - bb - hbp, 0) AS team_iso_game,
            ks::DOUBLE / NULLIF(n_pa - bb - hbp, 0) AS team_k_rate_game,
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
    """)

    # 7. Bullpen rolling features
    con.execute("""
        CREATE TABLE bullpen_raw AS
        WITH first_pitchers AS (
            SELECT game_pk,
                   FIRST_VALUE(pitcher) OVER (
                       PARTITION BY game_pk ORDER BY at_bat_number, pitch_number
                   ) AS starter_id
            FROM pitches
        ),
        reliever_events AS (
            SELECT CAST(p.game_date AS DATE) AS game_date,
                   p.game_pk, p.home_team, p.away_team, p.events
            FROM pitches p
            JOIN first_pitchers fp ON p.game_pk = fp.game_pk
            WHERE p.pitcher != fp.starter_id
              AND p.events IN ('single', 'double', 'triple', 'home_run',
                  'strikeout', 'strikeout_double_play', 'walk', 'hit_by_pitch',
                  'field_out', 'field_error', 'fielders_choice', 'fielders_choice_out',
                  'grounded_into_double_play', 'double_play', 'triple_play',
                  'sac_fly', 'sac_bunt', 'sac_fly_double_play',
                  'catcher_interf', 'batter_interference',
                  'force_out', 'sacrifice_bunt_double_play')
        )
        SELECT game_date, game_pk, home_team, away_team,
            COUNT(*) / 3.0 AS bullpen_ip,
            SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS bullpen_ks,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bullpen_bbs,
            SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END) AS bullpen_hits,
            SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch') THEN 1 ELSE 0 END) AS bullpen_runs
        FROM reliever_events
        GROUP BY game_date, game_pk, home_team, away_team
    """)

    con.execute("""
        CREATE TABLE bullpen_team AS
        SELECT game_date, game_pk, home_team AS team,
               bullpen_ip, bullpen_ks, bullpen_bbs, bullpen_hits, bullpen_runs
        FROM bullpen_raw
        UNION ALL
        SELECT game_date, game_pk, away_team AS team,
               bullpen_ip, bullpen_ks, bullpen_bbs, bullpen_hits, bullpen_runs
        FROM bullpen_raw
    """)
    con.execute("""
        CREATE TABLE bullpen_shifted AS
        SELECT *,
            LAG(bullpen_bbs, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_bbs,
            LAG(bullpen_hits, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_hits,
            LAG(bullpen_ip, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_ip,
            LAG(bullpen_runs, 1) OVER (PARTITION BY team ORDER BY game_date) AS _s_runs
        FROM bullpen_team
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

    # 8. Assemble game_level
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
            ba.bullpen_whip_10g AS bullpen_whip_10g_away, ba.bullpen_era_10g AS bullpen_era_10g_away
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
    """)

    logger.info("game_level built: %d games", con.execute("SELECT COUNT(*) FROM game_level").fetchone()[0])

    # Drop intermediate tables
    for tbl in (
        "game_winners", "starters", "venues", "rest_days",
        "pitcher_game_stats", "pitcher_shifted", "pitcher_rolling", "pitcher_features",
        "team_offense_raw", "team_off_shifted", "team_offense_rolling",
        "bullpen_raw", "bullpen_team", "bullpen_shifted", "bullpen_rolling",
    ):
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    gc.collect()


# ── PBP-level features ──────────────────────────────────────────────────────

def _build_pbp_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build pbp_level table via pure DuckDB SQL."""

    logger.info("Building PBP-level features via DuckDB SQL...")

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
            -- Situational features
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
                WHEN p.pitch_type IN ('FF', 'FT', 'SI', 'FC', 'FS', 'FO') THEN 'fastball'
                WHEN p.pitch_type IN ('SL', 'CU', 'KC', 'CS', 'SV', 'WR') THEN 'breaking'
                WHEN p.pitch_type IN ('CH', 'EP', 'SC', 'KN', 'UN', 'PO') THEN 'offspeed'
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

    logger.info("pbp_level built: %d pitches", con.execute("SELECT COUNT(*) FROM pbp_level").fetchone()[0])

    # Free pitches + game_level
    con.execute("DROP TABLE IF EXISTS pitches")
    con.execute("DROP TABLE IF EXISTS game_level")
    gc.collect()


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

    Phase 2 only — reads pitches.parquet, builds game_df + pbp_df via SQL,
    writes them to disk, and loads into pandas ONLY for model training.

    Args:
        pitches_path: Path to pitches.parquet (from ingestion.py).
        output_dir:   Where to write game_df.parquet and pbp_df.parquet.
        validate:     Run validation checks.

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

    # Phase 2: Pure DuckDB SQL
    con = _connect(pitches_path)
    logger.info("[MEM] After DuckDB load: %.0f MB", _mem_mb())

    try:
        _build_game_level(con)
        logger.info("[MEM] After game_level: %.0f MB", _mem_mb())

        _build_pbp_level(con)
        logger.info("[MEM] After pbp_level: %.0f MB", _mem_mb())

        # Export to Parquet
        con.execute(f"COPY (SELECT * FROM game_level) TO '{game_out}' (FORMAT PARQUET)")
        con.execute(f"COPY (SELECT * FROM pbp_level) TO '{pbp_out}' (FORMAT PARQUET)")
        logger.info("Exported game_level: %.1f MB", game_out.stat().st_size / 1e6)
        logger.info("Exported pbp_level: %.1f MB", pbp_out.stat().st_size / 1e6)

        # Drop DuckDB tables before loading into pandas
        con.execute("DROP TABLE IF EXISTS game_level")
        con.execute("DROP TABLE IF EXISTS pbp_level")
        gc.collect()
    finally:
        con.close()

    logger.info("[MEM] After DuckDB close: %.0f MB", _mem_mb())

    # Phase 3: Load into pandas (model-ready only)
    game_df = pd.read_parquet(game_out)
    pbp_df = pd.read_parquet(pbp_out)

    # Downcast for memory
    for df in [game_df, pbp_df]:
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            if df[col].max() < 32767 and df[col].min() >= -32768:
                df[col] = df[col].astype("int16")

    logger.info("[MEM] After pandas load: %.0f MB", _mem_mb())
    logger.info("=== Feature Engineering complete: %d games, %d pitches ===",
                len(game_df), len(pbp_df))

    return game_df, pbp_df
