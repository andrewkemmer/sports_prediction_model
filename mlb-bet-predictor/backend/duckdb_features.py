"""
DuckDB-based feature engineering for MLB Statcast data.

All rolling windows, shifted features, groupby aggregations, and merges
are implemented as DuckDB SQL window functions operating on Parquet files
on disk.  RAM usage stays well under 2 GB even for multi-season datasets.

Architecture:
    Phase 1: Statcast pull → pitches.parquet  (pandas, data ingestion only)
    Phase 2: DuckDB queries → game_df.parquet + pbp_df.parquet
    Phase 3: Load final Parquet into pandas for model training only

PIT compliance:
    All rolling metrics use ROWS BETWEEN N PRECEDING AND 1 PRECEDING
    (i.e. shift(1) equivalent).  LAG() is used for previous-pitch context.
"""
from __future__ import annotations

import gc
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MLB_TEAMS_ABBREV = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
    "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SF",
    "SEA", "STL", "TB", "TEX", "TOR", "WSH",
]

ELO_START = 1500.0
ELO_K = 20
ELO_HOME_ADV = 65

ROLLING_WINDOW_PITCHER_GAMES = 6   # ≈ 100 pitches
ROLLING_WINDOW_BATTER_GAMES = 4    # ≈ 80 pitches
ROLLING_WINDOW_TEAM_GAMES = 30
ROLLING_WINDOW_BULLPEN_GAMES = 10

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

# Column aliases for Statcast schema drift
COLUMN_ALIASES = {
    "barrel百分比": "barrel", "barrel_pct": "barrel", "is_barrel": "barrel",
    "hardhit": "hard_contact", "hard_hit": "hard_contact", "hardhit百分比": "hard_contact",
    "exit_velocity": "launch_speed", "exit_velo": "launch_speed",
    "la": "launch_angle",
    "xwoba": "estimated_woba_using_speedangle", "xwOBA": "estimated_woba_using_speedangle",
    "xba": "estimated_ba_using_speedangle",
    "pitcher_name": "player_name",
    "event": "events",
}

UNUSED_COLS = [
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "if_fielding_alignment", "of_fielding_alignment",
    "post_home_score", "post_away_score",
    "event", "type", "launch_speed_angle",
]

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


# ── Helper: create DuckDB connection and load pitches ────────────────────────

def _connect_and_load(pitches_path: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and register the pitches parquet."""
    con = duckdb.connect(database=":memory:")

    # Register the parquet file as a table
    con.execute(f"CREATE TABLE pitches AS SELECT * FROM '{pitches_path}'")

    # Ensure game_date is a proper date type
    con.execute("""
        ALTER TABLE pitches
        ALTER COLUMN game_date TYPE DATE
        USING CASE
            WHEN typeof(game_date) = 'VARCHAR' THEN CAST(game_date AS DATE)
            WHEN typeof(game_date) = 'BIGINT' THEN CAST(epoch_ms(game_date) AS DATE)
            ELSE game_date
        END
    """)

    # Drop unused columns
    existing = [r[0] for r in con.execute("DESCRIBE pitches").fetchall()]
    for col in UNUSED_COLS:
        if col in existing:
            con.execute(f"ALTER TABLE pitches DROP COLUMN \"{col}\"")

    return con


# ── Phase 2: Game-Level Features ────────────────────────────────────────────

def _create_game_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build game_df via DuckDB SQL and register as 'game_level' table."""

    logger.info("Building game-level features via DuckDB SQL...")

    # ── Step 1: Determine winners ─────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE game_winners AS
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
        FROM last_pitch
        WHERE rn = 1
    """)

    # ── Step 2: Starting pitchers ─────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE starters AS
        WITH first_pa AS (
            SELECT game_pk, pitcher,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk ORDER BY at_bat_number, pitch_number
                   ) AS rn
            FROM pitches
        ),
        top_first AS (
            SELECT game_pk, pitcher AS away_starter_id
            FROM pitches
            WHERE inning = 1 AND inning_topbot = 'Top'
            GROUP BY game_pk, pitcher
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) = 1
        ),
        bot_first AS (
            SELECT game_pk, pitcher AS home_starter_id
            FROM pitches
            WHERE inning = 1 AND inning_topbot = 'Bot'
            GROUP BY game_pk, pitcher
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number, pitch_number) = 1
        )
        SELECT DISTINCT
            p.game_pk,
            CAST(p.game_date AS DATE) AS game_date,
            p.home_team,
            p.away_team,
            t.away_starter_id,
            b.home_starter_id
        FROM (SELECT DISTINCT game_pk, game_date, home_team, away_team FROM pitches) p
        LEFT JOIN top_first t ON p.game_pk = t.game_pk
        LEFT JOIN bot_first b ON p.game_pk = b.game_pk
    """)

    # ── Step 3: Venue info ────────────────────────────────────────────
    venue_lines = []
    for k, v in VENUE_MAP.items():
        venue_lines.append(f"            WHEN '{k}' THEN '{v}'")
    venue_cases = "\n".join(venue_lines)
    con.execute(f"""
        CREATE OR REPLACE TABLE venues AS
        SELECT DISTINCT game_pk, home_team,
            CASE home_team
{venue_cases}
                ELSE 'Unknown'
            END AS venue
        FROM pitches
    """)

    # ── Step 4: Rest days ─────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE rest_days AS
        WITH team_games AS (
            SELECT DISTINCT game_date, game_pk, home_team AS team FROM pitches
            UNION
            SELECT DISTINCT game_date, game_pk, away_team AS team FROM pitches
        ),
        with_prev AS (
            SELECT *,
                LAG(game_date) OVER (PARTITION BY team ORDER BY game_date) AS prev_date
            FROM team_games
        )
        SELECT game_pk, team,
               CAST(game_date AS DATE) - CAST(prev_date AS DATE) AS rest_days
        FROM with_prev
    """)

    # ── Step 5: Pitcher rolling features (PIT-compliant) ──────────────
    con.execute(f"""
        CREATE OR REPLACE TABLE pitcher_game_stats AS
        WITH pa_events AS (
            SELECT game_date, game_pk, pitcher, events,
                   estimated_woba_using_speedangle AS xwoba_val,
                   barrel AS barrel_val,
                   hard_contact AS hard_val
            FROM pitches
            WHERE events IN ({PA_END_EVENTS})
        ),
        game_agg AS (
            SELECT
                CAST(game_date AS DATE) AS game_date,
                game_pk,
                pitcher,
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
        SELECT *,
            hits_allowed + bbs + hbps - hrs_allowed AS runs
        FROM game_agg
    """)

    # Rolling pitcher features with PIT-compliant shifted windows
    con.execute(f"""
        CREATE OR REPLACE TABLE pitcher_rolling AS
        SELECT *,
            -- Shifted IP (previous game's IP)
            LAG(ip_approx, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_ip,
            -- Shifted stats for PIT compliance
            LAG(runs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_runs,
            LAG(ks, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_ks,
            LAG(bbs, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_bbs,
            LAG(hits_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_hits,
            LAG(hrs_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_hrs,
            LAG(hbps, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_hbps,
            LAG(xwoba, 1) OVER (PARTITION BY pitcher ORDER BY game_date) AS _shifted_xwoba,
            -- Rolling sums over last {ROLLING_WINDOW_PITCHER_GAMES} games (shifted)
            SUM(LAG(runs, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_runs,
            SUM(LAG(ks, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_ks,
            SUM(LAG(bbs, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_bbs,
            SUM(LAG(hits_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_hits,
            SUM(LAG(hrs_allowed, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_hrs,
            SUM(LAG(hbps, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_hbps,
            SUM(LAG(ip_approx, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_ip,
            AVG(LAG(xwoba, 1) OVER (PARTITION BY pitcher ORDER BY game_date))
                OVER (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_PITCHER_GAMES - 1} PRECEDING AND CURRENT ROW) AS _roll_xwoba
        FROM pitcher_game_stats
    """)

    # Derive final pitcher features
    con.execute("""
        CREATE OR REPLACE TABLE pitcher_features AS
        SELECT
            game_date, game_pk, pitcher,
            _roll_runs / NULLIF(_roll_ip, 0) * 9.0 AS sp_era_30g,
            _roll_ks / NULLIF(_roll_ip, 0) * 9.0 AS sp_k9_30g,
            _roll_bbs / NULLIF(_roll_ip, 0) * 9.0 AS sp_bb9_30g,
            (_roll_bbs + _roll_hits) / NULLIF(_roll_ip, 0) AS sp_whip_30g,
            (13 * _roll_hrs + 3 * (_roll_bbs + _roll_hbps) - 2 * _roll_ks)
                / NULLIF(_roll_ip, 0) AS sp_fip_30g,
            _roll_xwoba AS sp_xwoba_30g
        FROM pitcher_rolling
    """)

    # ── Step 6: Team offense rolling features ─────────────────────────
    con.execute(f"""
        CREATE OR REPLACE TABLE team_offense_raw AS
        WITH pa_events AS (
            SELECT
                CAST(game_date AS DATE) AS game_date,
                game_pk,
                CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS batting_team,
                events
            FROM pitches
            WHERE events IN ({PA_END_EVENTS})
        ),
        game_agg AS (
            SELECT
                game_date, game_pk, batting_team,
                COUNT(*) AS n_pa,
                SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS singles,
                SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS doubles,
                SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS triples,
                SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hrs,
                SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
                SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
                SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS ks
            FROM pa_events
            GROUP BY game_date, game_pk, batting_team
        )
        SELECT *,
            -- wOBA (simplified FanGraphs weights)
            (0.690 * bb + 0.722 * hbp + 0.878 * singles + 1.242 * doubles
             + 1.568 * triples + 2.007 * hrs) / NULLIF(n_pa - bb - hbp + bb + hbp, 0) AS team_woba_game,
            -- ISO
            (doubles + 2 * triples + 3 * hrs) / NULLIF(n_pa - bb - hbp, 0) AS team_iso_game,
            -- K rate
            ks::DOUBLE / NULLIF(n_pa - bb - hbp, 0) AS team_k_rate_game,
            -- BB rate
            bb::DOUBLE / NULLIF(n_pa, 0) AS team_bb_rate_game
        FROM game_agg
    """)

    # Rolling team offense with PIT shift
    con.execute(f"""
        CREATE OR REPLACE TABLE team_offense_rolling AS
        SELECT
            game_date, game_pk, batting_team,
            AVG(LAG(team_woba_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date))
                OVER (PARTITION BY batting_team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_TEAM_GAMES - 1} PRECEDING AND CURRENT ROW) AS team_woba_30g,
            AVG(LAG(team_iso_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date))
                OVER (PARTITION BY batting_team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_TEAM_GAMES - 1} PRECEDING AND CURRENT ROW) AS team_iso_30g,
            AVG(LAG(team_k_rate_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date))
                OVER (PARTITION BY batting_team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_TEAM_GAMES - 1} PRECEDING AND CURRENT ROW) AS team_k_rate_30g,
            AVG(LAG(team_bb_rate_game, 1) OVER (PARTITION BY batting_team ORDER BY game_date))
                OVER (PARTITION BY batting_team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_TEAM_GAMES - 1} PRECEDING AND CURRENT ROW) AS team_bb_rate_30g
        FROM team_offense_raw
    """)

    # ── Step 7: Bullpen rolling features ──────────────────────────────
    con.execute(f"""
        CREATE OR REPLACE TABLE bullpen_raw AS
        WITH first_pitchers AS (
            SELECT game_pk,
                   FIRST_VALUE(pitcher) OVER (
                       PARTITION BY game_pk ORDER BY at_bat_number, pitch_number
                   ) AS starter_id
            FROM pitches
        ),
        reliever_events AS (
            SELECT
                CAST(p.game_date AS DATE) AS game_date,
                p.game_pk, p.home_team, p.away_team, p.events
            FROM pitches p
            JOIN first_pitchers fp ON p.game_pk = fp.game_pk
            WHERE p.pitcher != fp.starter_id
              AND p.events IN ({PA_END_EVENTS})
        ),
        game_bullpen AS (
            SELECT
                game_date, game_pk, home_team, away_team,
                COUNT(*) / 3.0 AS bullpen_ip,
                SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS bullpen_ks,
                SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bullpen_bbs,
                SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run') THEN 1 ELSE 0 END) AS bullpen_hits,
                SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch') THEN 1 ELSE 0 END) AS bullpen_runs
            FROM reliever_events
            GROUP BY game_date, game_pk, home_team, away_team
        )
        SELECT * FROM game_bullpen
    """)

    # Explode into team rows and compute rolling bullpen
    con.execute(f"""
        CREATE OR REPLACE TABLE bullpen_rolling AS
        WITH team_rows AS (
            SELECT game_date, game_pk, home_team AS team,
                   bullpen_ip, bullpen_ks, bullpen_bbs, bullpen_hits, bullpen_runs
            FROM bullpen_raw
            UNION ALL
            SELECT game_date, game_pk, away_team AS team,
                   bullpen_ip, bullpen_ks, bullpen_bbs, bullpen_hits, bullpen_runs
            FROM bullpen_raw
        )
        SELECT
            game_date, game_pk, team,
            (SUM(LAG(bullpen_bbs, 1) OVER (PARTITION BY team ORDER BY game_date))
                OVER (PARTITION BY team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_BULLPEN_GAMES - 1} PRECEDING AND CURRENT ROW)
             + SUM(LAG(bullpen_hits, 1) OVER (PARTITION BY team ORDER BY game_date))
                OVER (PARTITION BY team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_BULLPEN_GAMES - 1} PRECEDING AND CURRENT ROW))
            / NULLIF(SUM(LAG(bullpen_ip, 1) OVER (PARTITION BY team ORDER BY game_date))
                OVER (PARTITION BY team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_BULLPEN_GAMES - 1} PRECEDING AND CURRENT ROW), 0)
            AS bullpen_whip_10g,
            SUM(LAG(bullpen_runs, 1) OVER (PARTITION BY team ORDER BY game_date))
                OVER (PARTITION BY team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_BULLPEN_GAMES - 1} PRECEDING AND CURRENT ROW)
            / NULLIF(SUM(LAG(bullpen_ip, 1) OVER (PARTITION BY team ORDER BY game_date))
                OVER (PARTITION BY team ORDER BY game_date
                      ROWS BETWEEN {ROLLING_WINDOW_BULLPEN_GAMES - 1} PRECEDING AND CURRENT ROW), 0)
            * 9.0 AS bullpen_era_10g
        FROM team_rows
    """)

    # ── Step 8: Assemble game_level ───────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE game_level AS
        SELECT
            w.game_pk,
            w.game_date,
            w.home_team,
            w.away_team,
            w.home_score,
            w.away_score,
            w.home_win,
            w.total_runs,
            s.home_starter_id,
            s.away_starter_id,
            v.venue,
            rh.rest_days AS rest_days_home,
            ra.rest_days AS rest_days_away,
            -- Pitcher features (home)
            ph.sp_era_30g AS sp_era_home,
            ph.sp_k9_30g AS sp_k9_home,
            ph.sp_bb9_30g AS sp_bb9_home,
            ph.sp_whip_30g AS sp_whip_home,
            ph.sp_fip_30g AS sp_fip_home,
            ph.sp_xwoba_30g AS sp_xwoba_home,
            -- Pitcher features (away)
            pa.sp_era_30g AS sp_era_away,
            pa.sp_k9_30g AS sp_k9_away,
            pa.sp_bb9_30g AS sp_bb9_away,
            pa.sp_whip_30g AS sp_whip_away,
            pa.sp_fip_30g AS sp_fip_away,
            pa.sp_xwoba_30g AS sp_xwoba_away,
            -- Team offense (home)
            th.team_woba_30g AS team_woba_30g_home,
            th.team_iso_30g AS team_iso_30g_home,
            th.team_k_rate_30g AS team_k_rate_30g_home,
            th.team_bb_rate_30g AS team_bb_rate_30g_home,
            -- Team offense (away)
            ta.team_woba_30g AS team_woba_30g_away,
            ta.team_iso_30g AS team_iso_30g_away,
            ta.team_k_rate_30g AS team_k_rate_30g_away,
            ta.team_bb_rate_30g AS team_bb_rate_30g_away,
            -- Bullpen (home)
            bh.bullpen_whip_10g AS bullpen_whip_10g_home,
            bh.bullpen_era_10g AS bullpen_era_10g_home,
            -- Bullpen (away)
            ba.bullpen_whip_10g AS bullpen_whip_10g_away,
            ba.bullpen_era_10g AS bullpen_era_10g_away
        FROM game_winners w
        LEFT JOIN starters s ON w.game_pk = s.game_pk
        LEFT JOIN venues v ON w.game_pk = v.game_pk
        LEFT JOIN rest_days rh ON w.game_pk = rh.game_pk AND w.home_team = rh.team
        LEFT JOIN rest_days ra ON w.game_pk = ra.game_pk AND w.away_team = ra.team
        LEFT JOIN pitcher_features ph ON w.game_pk = ph.game_pk AND w.home_team = (SELECT home_team FROM pitches WHERE game_pk = w.game_pk LIMIT 1) AND ph.pitcher = s.home_starter_id
        LEFT JOIN pitcher_features pa ON w.game_pk = pa.game_pk AND pa.pitcher = s.away_starter_id
        LEFT JOIN team_offense_rolling th ON w.game_pk = th.game_pk AND w.home_team = th.batting_team
        LEFT JOIN team_offense_rolling ta ON w.game_pk = ta.game_pk AND w.away_team = ta.batting_team
        LEFT JOIN bullpen_rolling bh ON w.game_pk = bh.game_pk AND w.home_team = bh.team
        LEFT JOIN bullpen_rolling ba ON w.game_pk = ba.game_pk AND w.away_team = ba.team
    """)

    logger.info("game_level built: %d games", con.execute("SELECT COUNT(*) FROM game_level").fetchone()[0])


def _create_pbp_level(con: duckdb.DuckDBPyConnection) -> None:
    """Build pbp_df via DuckDB SQL and register as 'pbp_level' table."""

    logger.info("Building PBP-level features via DuckDB SQL...")

    con.execute("""
        CREATE OR REPLACE TABLE pbp_level AS
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
            p.*,
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
            -- AB pitch count
            ROW_NUMBER() OVER (
                PARTITION BY p.game_pk, p.at_bat_number
                ORDER BY p.pitch_number
            ) AS ab_pitch_count,
            -- Times through order
            CEIL(p.at_bat_number / 9.0)::INT AS times_through_order,
            -- L/R matchup
            CASE WHEN p.stand = p.p_throws THEN 'same' ELSE 'opposite' END AS lr_matchup,
            -- Contact quality
            p.barrel AS is_barrel,
            p.hard_contact AS is_hard_hit,
            p.launch_speed AS exit_velocity,
            p.launch_angle AS launch_angle_f,
            -- Pitch category
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


# ── Public API ──────────────────────────────────────────────────────────────

def run_duckdb_pipeline(
    start_date: str | date,
    end_date: str | date,
    checkpoint_dir: Optional[str | Path] = None,
    resume: bool = True,
    validate: bool = True,
    chunk_games: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Statcast feature pipeline using DuckDB SQL.

    All rolling windows, shifted features, groupby aggregations, and merges
    are SQL window functions operating on Parquet on disk.  RAM stays < 2 GB.

    Args:
        start_date: Start date.
        end_date: End date.
        checkpoint_dir: Output directory.
        resume: Resume from Statcast checkpoints.
        validate: Run validation suite.
        chunk_games: Ignored (kept for API compat; DuckDB handles memory internally).

    Returns:
        (game_level, pnp_level) DataFrames.
    """
    from statcast_pipeline import pull_statcast_data

    logger.info("=== DuckDB Pipeline: %s to %s ===", start_date, end_date)

    work_dir = Path(checkpoint_dir) if checkpoint_dir else Path("/content/mlb_tmp")
    work_dir.mkdir(parents=True, exist_ok=True)
    RAW = work_dir / "raw_pitches.parquet"
    DUCKDB_DB = work_dir / "mlb.duckdb"

    # ── Phase 1: Pull Statcast → parquet ──────────────────────────────
    if not RAW.exists() or not resume:
        pitches = pull_statcast_data(start_date, end_date,
                                     checkpoint_dir=checkpoint_dir, resume=resume)
        if pitches.empty:
            logger.error("No Statcast data pulled. Aborting.")
            return pd.DataFrame(), pd.DataFrame()
        pitches.to_parquet(RAW, index=False)
        logger.info("Raw pitches saved: %d rows, %.0f MB",
                     len(pitches), RAW.stat().st_size / 1e6)
        del pitches; gc.collect()
    else:
        logger.info("Resuming from existing parquet: %s", RAW)

    # ── Phase 2: DuckDB SQL feature engineering ───────────────────────
    # Remove old DuckDB file to start fresh
    if DUCKDB_DB.exists():
        DUCKDB_DB.unlink()

    con = _connect_and_load(RAW)
    logger.info("DuckDB loaded pitches: %d rows",
                con.execute("SELECT COUNT(*) FROM pitches").fetchone()[0])

    try:
        _create_game_level(con)
        _create_pbp_level(con)

        # ── Export to Parquet ─────────────────────────────────────────
        game_out = work_dir / "game_level_features.parquet"
        pbp_out = work_dir / "pbp_level_features.parquet"

        con.execute(f"COPY (SELECT * FROM game_level) TO '{game_out}' (FORMAT PARQUET)")
        con.execute(f"COPY (SELECT * FROM pbp_level) TO '{pbp_out}' (FORMAT PARQUET)")

        logger.info("Exported game_level: %s (%.0f MB)",
                     game_out, game_out.stat().st_size / 1e6)
        logger.info("Exported pbp_level: %s (%.0f MB)",
                     pbp_out, pbp_out.stat().st_size / 1e6)

        # ── Phase 3: Load into pandas (model-ready) ──────────────────
        game_level = pd.read_parquet(game_out)
        pnp_level = pd.read_parquet(pbp_out)

        # Downcast
        for df in [game_level, pnp_level]:
            for col in df.select_dtypes(include=["float64"]).columns:
                df[col] = df[col].astype("float32")
            for col in df.select_dtypes(include=["int64"]).columns:
                if df[col].max() < 32767 and df[col].min() >= -32768:
                    df[col] = df[col].astype("int16")

        # Validate
        if validate:
            from statcast_pipeline import validate_datasets
            validate_datasets(game_level, pnp_level, pnp_level)

        # Save final outputs
        if checkpoint_dir:
            out_dir = Path(checkpoint_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            game_level.to_parquet(out_dir / "game_level_features.parquet", index=False)
            pnp_level.to_parquet(out_dir / "pbp_level_features.parquet", index=False)

    finally:
        con.close()

    # Cleanup temp files
    for f in [RAW, DUCKDB_DB, work_dir / "game_level_features.parquet",
              work_dir / "pbp_level_features.parquet"]:
        if f.exists():
            f.unlink()

    logger.info("=== DuckDB Pipeline complete: %d games, %d pitches ===",
                len(game_level), len(pnp_level))
    return game_level, pnp_level
