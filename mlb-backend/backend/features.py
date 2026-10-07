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
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import resource  # POSIX-only stdlib (ru_maxrss); absent on Windows
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

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

# ── Expected-lineup availability (OUT/IR injury flag) ─────────────────────
# An OUT/IR designation removes a player from the game-eligible nine, but
# his own trailing rating (shrunk_re24) is untouched: it is built strictly
# from games he actually played and stays available to any side he is
# eligible for.
# The IL tables are LOCAL-RUNTIME CACHES (build_il_stints.py, rebuilt as
# Phase 1.5 of every daily run) that live OUTSIDE the git repo — the
# pipeline points MLB_IL_STINTS_DIR at the run cache and this module
# falls back to data_delivery only for legacy checkouts. Absence must
# degrade loudly to the participant behavior, never silently look correct.
IL_STINTS_FILE = "il_stints.parquet"
IL_STINTS_PITCHERS_FILE = "il_stints_pitchers.parquet"
# StatsAPI full-season roster position map (batter, season, pos) — the
# pl_<pos>_xwoba_* family's pool key. Lives beside the IL ledger (same
# runtime cache; .adhoc/mlb_fetch_positions.py rebuilds both copies).
PLAYER_POSITIONS_FILE = "player_positions.parquet"
# Announced batting orders (game_pk, home_order/away_order of 9 MLBAM ids,
# complete_* flags) — membership's tier-1 source: tonight's nine. Lives in
# the repo's data_delivery (backfill_lineups.py extends it; the daily run
# refreshes today's slate). Absence degrades to EMPTY, never a build failure.
LINEUPS_FILE = "lineups.parquet"
# A stale IL table is the silent inclusion failure: a player PLACED on
# the IL after the table's last transaction date stays in the projected
# nine (his frozen _pa30 keeps him ranked high — the 2026-09-28 Aaron
# Judge audit measured exactly that shape: rank 5 of 14 without the
# filter). Beyond this lag vs the data horizon, _register_il_stints
# warns loudly that new placements are invisible.
IL_STINTS_MAX_LAG_DAYS = 7
IL_STINTS_META_FILE = "il_stints.meta.json"
# How far back a team member's rating row may sit and still make him a
# candidate for the next game. Rating rows live on dates the team played,
# so 10 days comfortably spans one skipped game plus a rainout.
LINEUP_POOL_LOOKBACK_DAYS = 10

# The OUT/IR predicate, kept as its own constant so the shipped SQL and the
# tests pin the SAME text: a stint opened on or before the game date and not
# closed by it (an open stint has a NULL end and is still injured).
_IL_EXISTS_PREDICATE = """CASE WHEN EXISTS (
                   SELECT 1 FROM il_stints i
                   WHERE i.batter = p.batter
                     AND i.il_start <= p.game_date
                     AND (i.il_end IS NULL OR i.il_end > p.game_date))
               THEN 1 ELSE 0 END"""

# The TEAM-side availability filter (2026-10-02): the same batter stint
# semantics as _IL_EXISTS_PREDICATE, generalized to any row alias/date so
# every team-side SOURCE table can drop an unavailable batter's rows BEFORE
# any window/season aggregate forms — the bullpen_raw precedent (source-level
# filter, served family inherits it):
#   team_offense_raw  -> team_woba/iso/k_rate/bb_rate_30g (+ season, deltas)
#   team_contact_raw  -> team_barrel/hardhit/exitvelo_15g (+ season, deltas)
#   team_hand_raw     -> lineup_lefty_share_30g -> opp_lefty_share
#   exp2_team_cat_*   -> team_k_pct_cat, team_xwoba_cat
#   exp2_team_fbhand_* -> team_k_pct_fb_vs, team_pa_fb_vs
# Interpolated ONLY when the ledger registered (_batters_ok); absent cache →
# literal TRUE (the exact pre-availability behavior — _register_il_stints
# already warned loudly). Coverage contract: season-to-date/ASOF totals only
# shrink slightly (a batter has no PAs while his stint is open, so the filter
# catches edge rows only) — xwoba/K% coverage is preserved, never gated.
_BATTER_LEDGER_EXCLUDE_ROW_SQL = """NOT EXISTS (
              SELECT 1 FROM il_stints i
              WHERE i.batter = {alias}.batter
                AND i.il_start <= CAST({date} AS DATE)
                AND (i.il_end IS NULL OR i.il_end > CAST({date} AS DATE)))"""


def _batter_excl(alias: str, date: str, ledger_ok: bool) -> str:
    """Row-level batter-availability exclusion for team-side source tables.

    Returns the NOT-EXISTS fragment bound to ``il_stints`` when the ledger
    is registered; literal TRUE when it is not — a missing cache degrades to
    the exact pre-availability semantics (never a missing-table error).
    """
    if not ledger_ok:
        return "TRUE"
    return _BATTER_LEDGER_EXCLUDE_ROW_SQL.format(alias=alias, date=date)

# ── Membership: tonight's nine (2026-10-03 membership audit) ────────────────
# The pools previously widened to every healthy roster member with a rating
# row in the lookback — a 26-man roster average priced on guys with no path
# into the game. The agreed membership set is the EFFECTIVE nine per
# team-game, resolved in lineup_effective:
#   tier 1  the announced batting order (lineups.parquet, complete nine)
#   tier 2  projected: each slot's MODAL starter over the last {lookback}
#           days of announced orders, conditioned on the opposing starter's
#           handedness (starters table) and IL-filtered as of the target date
#   (no row) full roster — the exact pre-audit semantics, so an absent or
#           unreadable lineup cache never shrinks coverage below what shipped
#           before (2024's horizon, lineups.parquet does not cover)
# The clause below is hardcoded into BOTH pool templates so the lineup_* and
# pl_* families can never drift apart on membership; an EMPTY
# lineup_effective table (the degrade path, or a test fixture) makes
# NOT EXISTS true for every row and restores the full-roster pool.
_LINEUP_MEMBERSHIP_SQL = """(
    NOT EXISTS (
        SELECT 1 FROM lineup_effective e
        WHERE e.game_pk = g.game_pk AND e.team = g.batting_team)
    OR r.batter IN (
        SELECT e2.batter FROM lineup_effective e2
        WHERE e2.game_pk = g.game_pk AND e2.team = g.batting_team))"""

# Empty degrade tables — the loud no-data shape every consumer binds to.
_EMPTY_LINEUPS_RAW_SQL = """CREATE OR REPLACE TEMP TABLE lineups_raw AS
    SELECT CAST(NULL AS BIGINT) AS game_pk,
           CAST(NULL AS DATE) AS game_date,
           CAST(NULL AS VARCHAR) AS home_team,
           CAST(NULL AS VARCHAR) AS away_team,
           CAST(NULL AS BIGINT[]) AS home_order,
           CAST(NULL AS BIGINT[]) AS away_order,
           CAST(NULL AS BOOLEAN) AS complete_home,
           CAST(NULL AS BOOLEAN) AS complete_away,
           CAST(NULL AS VARCHAR) AS home_starter_hand,
           CAST(NULL AS VARCHAR) AS away_starter_hand
    WHERE false"""
_EMPTY_LINEUP_EFFECTIVE_SQL = """CREATE OR REPLACE TABLE lineup_effective AS
    SELECT CAST(NULL AS BIGINT) AS game_pk,
           CAST(NULL AS VARCHAR) AS team,
           CAST(NULL AS BIGINT) AS batter,
           CAST(NULL AS INTEGER) AS tier
    WHERE false"""

# Tier-2 availability filter: a projected slot must not hand a roster spot
# to a batter whose IL stint is open as of the TARGET game (bindings reuse
# the _IL_EXISTS_PREDICATE semantics, re-aliased to n/s).
_EFFECTIVE_IL_FILTER = """NOT EXISTS (
            SELECT 1 FROM il_stints i
            WHERE i.batter = s.batter
              AND i.il_start <= n.game_date
              AND (i.il_end IS NULL OR i.il_end > n.game_date))"""

# ── Pool universe: the team-games every candidate pool aggregates ──────────
# Structural alignment (2026-10-03): the pools derived their game grid from
# batter_ratings alone — a frame that only contains PLAYED games — so
# tonight's team-game never entered the grid and the 3-tier chain
# (announced nine / hand-conditioned projection / roster degrade) could not
# run for the slate. Slate pl_* shipped as an unmarked stale carry-forward
# of each team's last completed game instead. pool_universe fixes the grid
# at the SOURCE so history and the slate resolve through the SAME SQL:
#   hist   — batter_ratings team-games (identical set to the old inline
#            subquery, so every historical row is unchanged)
#   slate  — lineups_raw rows with no played frame yet (today's scheduled
#            games, captured pre-game by fetch_scheduled_lineups), expanded
#            to both sides; codes are normalized at capture and NULLIF'd so
#            the 2026-10-03 legacy empty-string rows (schedule captured
#            without hydrate) can never fabricate a '' team.
# is_slate flags the rows _export_slate_pl ships to the serving slate.
_SLATE_TEAMGAMES_SQL = """
    SELECT l.game_pk, l.game_date, l.home_team AS batting_team
    FROM lineups_raw l
    WHERE NULLIF(l.home_team, '') IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM pitches p WHERE p.game_pk = l.game_pk)
    UNION
    SELECT l.game_pk, l.game_date, l.away_team AS batting_team
    FROM lineups_raw l
    WHERE NULLIF(l.away_team, '') IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM pitches p WHERE p.game_pk = l.game_pk)
"""


def _build_pool_universe(con: "duckdb.DuckDBPyConnection") -> None:
    """Create ``pool_universe`` — the team-game grid membership and the
    position pools both bind. Never raises; always leaves a usable table.

    Degrades loudly: a fixture/checkout without ``lineups_raw`` or
    ``pitches`` keeps the historical grid only (the slate then serves the
    MARKED carry in build_upcoming_slate), and a checkout without
    ``batter_ratings`` gets the empty grid (every pool falls to its own
    no-data path — never a missing-table crash).
    """
    hist = ("SELECT DISTINCT game_date, game_pk, batting_team,\n"
            "              false AS is_slate\n"
            "           FROM batter_ratings")
    try:
        con.execute(f"""CREATE OR REPLACE TABLE pool_universe AS
            {hist}
            UNION ALL
            SELECT game_date, game_pk, batting_team, true AS is_slate
            FROM ({_SLATE_TEAMGAMES_SQL}) s""")
    except Exception as e:  # noqa: BLE001 — absent input tables in fixtures
        logger.warning("pool_universe slate extension failed (%s) — historical "
                       "team-games only (slate pl_* serves the marked carry)", e)
        try:
            con.execute(f"CREATE OR REPLACE TABLE pool_universe AS {hist}")
        except Exception as e2:  # noqa: BLE001
            logger.warning("pool_universe build failed (%s) — empty grid; pools "
                           "degrade to their no-data paths", e2)
            con.execute("""CREATE OR REPLACE TABLE pool_universe AS
                SELECT CAST(NULL AS DATE) AS game_date,
                       CAST(NULL AS BIGINT) AS game_pk,
                       CAST(NULL AS VARCHAR) AS batting_team,
                       false AS is_slate
                WHERE false""")
    n, ns = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE is_slate) "
        "FROM pool_universe").fetchone()
    logger.info("pool_universe: %d team-games (%d upcoming/slate)", n, ns)


_LINEUP_EFFECTIVE_SQL = """
    CREATE OR REPLACE TABLE lineup_effective AS
    WITH uni AS (
        -- the exact team-games the pools aggregate: historical
        -- (batter_ratings) PLUS tonight's slate — one grid, one membership
        -- chain (see _build_pool_universe).
        SELECT DISTINCT game_pk, batting_team AS team, game_date
        FROM pool_universe
    ),
    order_sides AS (
        -- Announced orders exploded to (game, side, slot, batter), carrying
        -- the feed's own team codes alongside — slots maps them into the
        -- pool vocabulary below. game_pk + home/away side is the stable
        -- identity.
        --
        -- ENGINE PORTABILITY (2026-10-03 production incident): the first
        -- shape used UNNEST(...) WITH ORDINALITY, which the Kaggle DuckDB
        -- build does not implement ("WITH ORDINALITY not implemented") —
        -- membership silently fell back to the full roster for every run.
        -- range(1, 10) + list_extract is the spelling every DuckDB line we
        -- run supports, and it pins the slot deterministically instead of
        -- trusting UNNEST's output order under parallel execution.
        SELECT l.game_pk, l.game_date, 'home' AS side,
               l.home_team, l.away_team,
               list_extract(l.home_order, r.slot) AS batter, r.slot AS slot
        FROM lineups_raw l, range(1, 10) AS r(slot)
        WHERE l.complete_home
          AND list_extract(l.home_order, r.slot) IS NOT NULL
        UNION ALL
        SELECT l.game_pk, l.game_date, 'away' AS side,
               l.home_team, l.away_team,
               list_extract(l.away_order, r.slot) AS batter, r.slot AS slot
        FROM lineups_raw l, range(1, 10) AS r(slot)
        WHERE l.complete_away
          AND list_extract(l.away_order, r.slot) IS NOT NULL
    ),
    slots AS (
        -- Team vocabulary: PITCHES' codes first — the feed's own
        -- abbreviations differ from the pool key (measured 2026-10-03: the
        -- feed returned OAK for all 162 of the Athletics' 2025 games while
        -- pitches carried ATH; a code-keyed join silently dropped that side
        -- to the tier-3 roster fallback, 1,458 missing members). The feed's
        -- capture-normalized codes are used ONLY when the game has no played
        -- frame yet — tonight's slate is absent from pitches by definition,
        -- and pool_universe keys those rows by the SAME lineups_raw codes,
        -- so tier 1 binds there too (the slate half of the 2026-10-03
        -- structural alignment).
        SELECT o.game_pk, o.game_date, o.side,
               COALESCE(CASE o.side WHEN 'home' THEN p.home_team
                                    ELSE p.away_team END,
                        CASE o.side WHEN 'home' THEN o.home_team
                                    ELSE o.away_team END) AS team,
               o.batter AS batter, o.slot AS slot
        FROM order_sides o
        LEFT JOIN (SELECT DISTINCT game_pk, home_team, away_team FROM pitches) p
          ON p.game_pk = o.game_pk
    ),
    -- tier 1: tonight's nine, exactly as announced
    t1 AS (
        SELECT u.game_pk, u.team, s.batter
        FROM uni u
        JOIN slots s ON s.game_pk = u.game_pk AND s.team = u.team
    ),
    -- team-games with no announced nine need a projection. The target row
    -- carries the opposing starter's hand (home team bats vs the AWAY
    -- starter, and vice versa); a missing/NULL hand cannot condition and
    -- falls back to the unconditioned window below.
    need AS (
        -- Target's opposing-starter hand: the STARTERS table's real throw
        -- hand for played games; the lineups_raw PROBABLE-starter hand for
        -- games with no played frame (tonight's slate — starters is
        -- pitches-derived and has no row there), which is what makes the
        -- projection hand-conditioned on the slate exactly as in history.
        -- COALESCE also covers a played game whose inning-1 PA row is
        -- missing. NULL on both sides cannot condition and falls back to
        -- the unconditioned window below (loud, never fabricated).
        SELECT u.game_pk, u.team, u.game_date,
               COALESCE(
                   CASE WHEN u.team = s.home_team THEN s.away_starter_hand
                        ELSE s.home_starter_hand END,
                   CASE WHEN u.team = l.home_team THEN l.away_starter_hand
                        ELSE l.home_starter_hand END) AS opp_hand
        FROM uni u
        LEFT JOIN (SELECT DISTINCT game_pk, team FROM t1) a
          ON a.game_pk = u.game_pk AND a.team = u.team
        LEFT JOIN starters s ON s.game_pk = u.game_pk
        LEFT JOIN lineups_raw l ON l.game_pk = u.game_pk
        WHERE a.team IS NULL
    ),
    -- prior announced lineups of that team inside the 10-day window,
    -- tagged with the opposing hand THEY actually faced. The IL filter
    -- binds as of the TARGET date (availability must never look ahead).
    hist AS (
        SELECT n.game_pk, n.team, n.opp_hand, s.batter, s.slot,
               s.game_date AS hist_date,
               COALESCE(
                   CASE WHEN s.team = hs.home_team THEN hs.away_starter_hand
                        ELSE hs.home_starter_hand END,
                   CASE WHEN s.team = hl.home_team THEN hl.away_starter_hand
                        ELSE hl.home_starter_hand END) AS hist_hand
        FROM need n
        JOIN slots s
          ON s.team = n.team
         AND s.game_date < n.game_date
         AND s.game_date >= n.game_date - INTERVAL {lookback} DAY
        LEFT JOIN starters hs ON hs.game_pk = s.game_pk
        LEFT JOIN lineups_raw hl ON hl.game_pk = s.game_pk
        WHERE {il_filter}
    ),
    votes AS (
        SELECT game_pk, team, slot, batter,
               count(*) AS votes, max(hist_date) AS last_seen
        FROM hist
        WHERE hist_hand = opp_hand
           OR hist_hand IS NULL OR opp_hand IS NULL
        GROUP BY game_pk, team, slot, batter
    ),
    -- one batter per slot: most votes, then most recent appearance
    t2 AS (
        SELECT game_pk, team, batter FROM (
            SELECT game_pk, team, slot, batter,
                   ROW_NUMBER() OVER (PARTITION BY game_pk, team, slot
                                      ORDER BY votes DESC, last_seen DESC,
                                               batter) AS rn
            FROM votes)
        WHERE rn = 1
    )
    SELECT game_pk, team, batter, 1 AS tier FROM t1
    UNION ALL
    SELECT game_pk, team, batter, 2 AS tier FROM t2
"""


def _pl_agg_coverage(con: "duckdb.DuckDBPyConnection") -> float:
    """Share of pos_agg's pool cells that are non-NULL across all 9 pools.

    1.0 is healthy; 0.0 means the position map never bound, so every
    served pl_* column ships NULL and is median-imputed downstream. The
    2026-10-03 remote run measured 0.0 here while the run still reported
    "ok" and the drift gate raised nothing — this number is the honest
    signal, so it is logged on EVERY build (healthy value included).
    """
    cells = " + ".join(f"(pl_{p}_xwoba IS NOT NULL)::INT"
                       for p in PL_POSITIONS)
    row = con.execute(
        f"SELECT avg(({cells})::DOUBLE) / {len(PL_POSITIONS)} "
        "FROM pos_agg").fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def _warn_pl_coverage(con: "duckdb.DuckDBPyConnection") -> None:
    cov = _pl_agg_coverage(con)
    served = len(PL_SERVED_POSITIONS)
    if cov >= PL_COVERAGE_FLOOR:
        logger.info("pl_* pool coverage: %.1f%% non-NULL across %d pools "
                    "(%d served)", cov * 100, len(PL_POSITIONS), served)
        return
    logger.warning(
        "pl_* pool coverage %.1f%% — below the %.0f%% floor. The position map "
        "did not bind, so all %d served pl_* universe columns ship NULL and "
        "are median-imputed as constants (a missing feature the drift gate "
        "cannot see). Check the player_positions warning above.",
        cov * 100, PL_COVERAGE_FLOOR * 100, 3 * served)


def _export_slate_pl(con: "duckdb.DuckDBPyConnection",
                     base: "Path | None" = None) -> int:
    """Ship tonight's resolved pl_* pools to ``pl_slate_<YYYYMMDD>.parquet``.

    Serving half of the 2026-10-03 slate structural alignment: while history
    resolves pl_* through the 3-tier chain (tier 1 announced nine, tier 2
    hand-conditioned 10-day projection, tier 3 full-roster degrade) + the
    position pools + the strictly-prior league fallback, the slate used to
    ship an UNMARKED stale carry-forward of each team's last completed game
    (lineup_effective/pos_agg never saw tonight's game_pk — batter_ratings
    only contains played games). This artifact closes that gap: one file per
    upcoming game_date with, per team-game:

      pl_<pos>_xwoba         the SAME pos_agg values history serves (8 served)
      pl_tier                membership tier that bound (1/2/3 degrade)
      pl_prior_positions     served pools that resolved to the league-by-
                             position prior instead of real pool data

    build_upcoming_slate reads it and marks every side (pool-t1/t2/t3 vs
    carry), so resolved-vs-carried is inspectable wherever the frame ships.
    Writes NOTHING when no upcoming team-games exist (every side then serves
    the marked carry — never a fabricated value). Never raises: a failure
    degrades the slate to the marked carry under a loud warning.
    Returns the number of team-game rows written.
    """
    served = list(PL_SERVED_POSITIONS)
    try:
        cols = ", ".join(f"a.pl_{p}_xwoba" for p in served)
        rows = con.execute(f"""
            SELECT u.game_date, u.game_pk, u.batting_team AS team,
                   {cols},
                   COALESCE((SELECT MAX(e.tier) FROM lineup_effective e
                             WHERE e.game_pk = u.game_pk
                               AND e.team = u.batting_team), 3) AS pl_tier
            FROM pool_universe u
            JOIN pos_agg a
              ON a.game_pk = u.game_pk AND a.batting_team = u.batting_team
            WHERE u.is_slate
            ORDER BY u.game_date, u.game_pk, u.batting_team
        """).fetchall()
        if not rows:
            logger.info("slate pl_*: no upcoming team-games resolved — the "
                        "serving slate marks every side 'carry'")
            return 0
        # Which served pools actually held pool data for each team-game;
        # everything else in the served set resolved via the prior fallback
        # (the pos_agg COALESCE) — named so the artifact stays honest about
        # "real pool vs league prior" even though both are resolved tonight.
        pool_pos: dict[tuple[int, str], set] = {}
        for pk, tm, pos in con.execute(
                f"SELECT game_pk, batting_team, {PL_POS_SQL} AS pos "
                "FROM pos_pool WHERE on_il = 0 "
                "GROUP BY game_pk, batting_team, pos "
                "HAVING SUM(_pa30) > 0").fetchall():
            pool_pos.setdefault((int(pk), str(tm)), set()).add(str(pos))
        col_names = (["game_date", "game_pk", "team"]
                     + [f"pl_{p}_xwoba" for p in served] + ["pl_tier"])
        df = pd.DataFrame(rows, columns=col_names)
        df["pl_prior_positions"] = [
            ",".join(p for p in served
                     if p not in pool_pos.get((int(pk), str(tm)), set()))
            for pk, tm in zip(df["game_pk"], df["team"])]
        base_dir = Path(base) if base is not None else _lineup_base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for d, grp in df.groupby("game_date"):
            p = base_dir / f"pl_slate_{pd.Timestamp(d):%Y%m%d}.parquet"
            grp.drop_duplicates(subset=["game_pk", "team"]).to_parquet(
                p, index=False)
            files.append(p.name)
        tiers = {int(k): int(v)
                 for k, v in df["pl_tier"].value_counts().items()}
        n_prior = int((df["pl_prior_positions"] != "").sum())
        logger.info("slate pl_* pools: %d team-games resolved across %d date(s) "
                    "(membership tiers %s; %d with prior-fallback pools) → %s",
                    len(df), df["game_date"].nunique(), tiers, n_prior,
                    ", ".join(files))
        return len(df)
    except Exception as e:  # noqa: BLE001
        logger.warning("slate pl_* export failed (%s) — the serving slate "
                       "falls back to the marked carry", e)
        return 0


def _build_lineup_effective(con: "duckdb.DuckDBPyConnection",
                            batters_ok: bool) -> None:
    """Resolve ``lineup_effective`` — the membership set both pools bind to.

    Never raises: a failed or empty build creates the EMPTY table under a
    loud warning so every pool degrades to the full-roster semantics
    (never a missing-table error, never a fabricated nine).
    """
    try:
        # The grid membership resolves against (and the position pools will
        # aggregate) — built HERE so lineup_effective and pos_pool can never
        # disagree on which team-games exist: history AND tonight's slate.
        _build_pool_universe(con)
        con.execute(_LINEUP_EFFECTIVE_SQL.format(
            lookback=LINEUP_POOL_LOOKBACK_DAYS,
            il_filter=(_EFFECTIVE_IL_FILTER if batters_ok else "TRUE")))
        n, ngames = con.execute(
            "SELECT count(*), count(DISTINCT game_pk) "
            "FROM lineup_effective").fetchone()
    except Exception as e:  # noqa: BLE001
        logger.warning("lineup_effective build failed (%s) — pools fall back "
                       "to the full-roster membership", e)
        con.execute(_EMPTY_LINEUP_EFFECTIVE_SQL)
        return
    if not n:
        logger.warning("lineup_effective is EMPTY — no announced or "
                       "projected nine anywhere in the frame; every pool "
                       "falls back to the full-roster membership")
        return
    logger.info("lineup_effective: %d members across %d team-games "
                "(announced + projected)", n, ngames)


# Candidate pool over ONE schema for both paths — same columns, same names,
# so every downstream consumer is indifferent to which path ran.
#   IL path   : widen to team members with a rating row in the last
#               {lookback} days; flag OUT/IR per the predicate.
#   Fallback  : restrict to that game's own rating rows (the participant
#               pool) and hardwire the flag to 0 — byte-equivalent to the
#               pre-IL feature.
# Both paths additionally bind membership (2026-10-03): only batters in the
# team-game's effective nine pool — see _LINEUP_MEMBERSHIP_SQL.
_LINEUP_POOL_SQL = """
    CREATE TABLE lineup_pool AS
    WITH pool AS (
        SELECT g.game_date, g.game_pk, g.batting_team,
               CAST(r.batter AS BIGINT) AS batter,
               r.shrunk_re24, r._pa30
        FROM (SELECT DISTINCT game_date, game_pk, batting_team
              FROM batter_ratings WHERE shrunk_re24 IS NOT NULL) g
        JOIN batter_ratings r
          ON r.batting_team = g.batting_team
         AND r.shrunk_re24 IS NOT NULL
         AND r.game_date <= g.game_date
         AND r.game_date >= g.game_date - INTERVAL {lookback} DAY
         {restrict}
         AND """ + _LINEUP_MEMBERSHIP_SQL + """
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY g.game_pk, g.batting_team, r.batter
            ORDER BY r.game_date DESC) = 1
    )
    SELECT p.game_date, p.game_pk, p.batting_team,
           p.batter, p.shrunk_re24, p._pa30,
           {on_il} AS on_il
    FROM pool p
"""

# Game-eligible aggregate — PARTICIPATION-WEIGHTED (2026-09-30). The
# widen-then-filter healthy pool IS the participation set, so the
# expected-lineup average weights every on_il = 0 member by his frozen
# trailing-30g PA (opportunity to actually take an at-bat) instead of an
# unweighted top-9-by-PA mean. Dropping the arbitrary nine-cut also
# RAISES coverage: a value whenever >=1 healthy member exists (a
# depleted side's real, thin participation set — never padded). Offline
# study (648 team-dates, Apr-Sep 2026, strict PIT + IL filter):
# rho vs same-day actual runs 0.1573 vs 0.1374 for the top-9 mean
# (bootstrap delta +0.020, P(delta>0)=0.93). top3 stays a top-3-BY-PA
# unweighted mean (its identity is the top of the order, not the pool);
# the dispersion column becomes the PA-weighted std over the pool.
_LINEUP_AGG_SQL = """
    CREATE TABLE lineup_agg AS
    WITH ranked AS (
        SELECT game_date, game_pk, batting_team, shrunk_re24, _pa30,
               ROW_NUMBER() OVER (PARTITION BY game_pk, batting_team
                                  ORDER BY _pa30 DESC) AS rn
        FROM lineup_pool WHERE on_il = 0
    ),
    pooled AS (
        SELECT game_date, game_pk, batting_team,
               SUM(_pa30) AS pa_pool,
               SUM(_pa30 * shrunk_re24) AS w_sum,
               SUM(_pa30 * shrunk_re24 * shrunk_re24) AS w2_sum,
               AVG(CASE WHEN rn <= 3 THEN shrunk_re24 END) AS top3
        FROM ranked
        GROUP BY game_date, game_pk, batting_team
    )
    SELECT game_date, game_pk, batting_team,
           w_sum / NULLIF(pa_pool, 0) AS lineup_re24_mean,
           top3 AS lineup_re24_top3,
           CASE WHEN pa_pool > 0 THEN
               SQRT(GREATEST(w2_sum / pa_pool
                             - POWER(w_sum / pa_pool, 2), 0))
           ELSE NULL END AS lineup_re24_std
    FROM pooled
"""

# OUT/IR availability flag on the projected (game-eligible) nine: 1 when at
# least one of the nine is on the IL as of the game date, else 0. An IL
# player outside the top-9 by playing time does NOT trip the flag.
_LINEUP_IL_FLAG_SQL = """
    CREATE TABLE lineup_il_flag AS
    WITH ranked AS (
        SELECT game_date, game_pk, batting_team, on_il,
               ROW_NUMBER() OVER (PARTITION BY game_pk, batting_team
                                  ORDER BY _pa30 DESC) AS rn
        FROM lineup_pool
    )
    SELECT game_date, game_pk, batting_team,
           MAX(CASE WHEN rn <= 9 THEN on_il ELSE 0 END) AS lineup_il_flag
    FROM ranked
    GROUP BY game_date, game_pk, batting_team
"""

# ── Position-pool xwOBA family (pl_<pos>_xwoba_*, 2026-10-02) ─────────────
# The MLB counterpart of NHL's pl_<metric>_<pos> family: instead of one
# team-wide lineup average, the healthy roster's trailing-30g SHRUNK xwOBA
# rating is pooled PER POSITION and served as pl_<pos>_xwoba_{home,away,diff}
# (9 pools x 3 reps = 27 columns; pitcher rows excluded from the pools).
# Structure mirrors the lineup family one-for-one: candidate pool (same
# 10-day lookback, one row per batter, OUT/IR flagged via the same
# predicate), PA-weighted healthy-pool aggregate, then a strictly-prior
# league-by-position mean used ONLY when a position's pool is empty for a
# team-game — the NHL _POSITION_PRIOR fallback, measured necessary here:
# season-level StatsAPI positions leave DH pools empty in ~64% of
# team-games because most clubs list their DH at a defensive position.
# A missing/unreadable player_positions.parquet degrades LOUDLY to an
# empty pos_agg (every pl_* ships NULL) — never a build failure, never a
# fabricated value (see _register_player_positions).
PL_POSITIONS = ("c", "fb", "sb", "ss", "tb", "rf", "cf", "lf", "dh")
# StatsAPI roster abbrev -> served pool suffix: fb/sb/tb carry the owner's
# first/second/third-base spelling (1B/2B/3B), everything else lowercases
# (C/SS/LF/CF/RF/DH). P stays out of the pools; TWP already folded to DH
# at fetch time.
PL_POS_SQL = ("CASE pos WHEN '1B' THEN 'fb' WHEN '2B' THEN 'sb' "
              "WHEN '3B' THEN 'tb' ELSE LOWER(pos) END")
# The 8 pools the 2026-10-03 plan SERVES (3 cols each = 24 universe
# members). pl_dh is still computed and pooled — season-level StatsAPI
# positions put most clubs' DH at a defensive spot — it simply never
# enters the serving universe, so neither the slate carry nor the
# coverage tripwire below counts it.
PL_SERVED_POSITIONS = tuple(p for p in PL_POSITIONS if p != "dh")
# pos_agg must fill at least this share of its pool cells or the pl_*
# family is effectively absent. A missing position map produces 0% and is
# INVISIBLE to the drift gate (a median-imputed constant column has
# PSI ~0) — this is the only signal that says "these 24 columns are dead".
PL_COVERAGE_FLOOR = 0.90

# Candidate pool — structurally identical to _LINEUP_POOL_SQL (same
# lookback/QUALIFY/ledger semantics + the same membership clause) plus the
# position-map join, so the two families degrade and refresh together.
_POS_POOL_SQL = """
    CREATE TABLE pos_pool AS
    WITH pool AS (
        SELECT g.game_date, g.game_pk, g.batting_team,
               CAST(r.batter AS BIGINT) AS batter,
               r.shrunk_xwoba, r._pa30,
               {pos_case} AS pos
        -- pool_universe = historical + tonight's slate team-games
        -- (_build_pool_universe): tonight's game_pk aggregates through the
        -- SAME membership/lookback/prior chain every historical game uses.
        FROM pool_universe g
        JOIN batter_ratings r
          ON r.batting_team = g.batting_team
         AND r.shrunk_xwoba IS NOT NULL
         AND r.game_date <= g.game_date
         AND r.game_date >= g.game_date - INTERVAL {lookback} DAY
         {restrict}
         AND """ + _LINEUP_MEMBERSHIP_SQL + """
        JOIN player_positions pp
          ON pp.batter = r.batter
         AND pp.season = EXTRACT(YEAR FROM r.game_date)
         AND pp.pos IN ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY g.game_pk, g.batting_team, r.batter
            ORDER BY r.game_date DESC) = 1
    )
    SELECT p.game_date, p.game_pk, p.batting_team,
           p.batter, p.shrunk_xwoba, p._pa30, p.pos,
           {on_il} AS on_il
    FROM pool p
"""

# League-by-position prior — the healthy-pool PA-weighted mean over every
# OTHER date (strictly prior: UNBOUNDED PRECEDING .. 1 PRECEDING), pivoted
# to one row per date so pos_agg binds it with a single join. Day 1 of the
# data horizon has no prior date and ships NULL — the same first-day
# behavior the rating chain itself has.
_POS_LEAGUE_SQL = """
    CREATE TABLE pos_league AS
    WITH daily AS (
        SELECT game_date, pos,
               SUM(_pa30 * shrunk_xwoba) AS num,
               SUM(_pa30) AS den
        FROM pos_pool WHERE on_il = 0
        GROUP BY game_date, pos
    ),
    -- Grid of every pool date x every pool position, so each position
    -- carries a PRIOR row on days its own pool was inactive (a zero-filled
    -- day contributes nothing to the expanding sums — the prior is the
    -- carry-forward of the last known value, still strictly prior).
    grid AS (
        SELECT d.game_date, p.pos,
               COALESCE(daily.num, 0) AS num,
               COALESCE(daily.den, 0) AS den
        FROM (SELECT DISTINCT game_date FROM pos_pool) d
        CROSS JOIN (SELECT DISTINCT pos FROM pos_pool) p
        LEFT JOIN daily
          ON daily.game_date = d.game_date AND daily.pos = p.pos
    )
    SELECT game_date,
           MAX(CASE WHEN pos = 'c'  THEN prior END) AS pr_c,
           MAX(CASE WHEN pos = 'fb' THEN prior END) AS pr_fb,
           MAX(CASE WHEN pos = 'sb' THEN prior END) AS pr_sb,
           MAX(CASE WHEN pos = 'ss' THEN prior END) AS pr_ss,
           MAX(CASE WHEN pos = 'tb' THEN prior END) AS pr_tb,
           MAX(CASE WHEN pos = 'rf' THEN prior END) AS pr_rf,
           MAX(CASE WHEN pos = 'cf' THEN prior END) AS pr_cf,
           MAX(CASE WHEN pos = 'lf' THEN prior END) AS pr_lf,
           MAX(CASE WHEN pos = 'dh' THEN prior END) AS pr_dh
    FROM (
        SELECT game_date, pos,
               SUM(num) OVER w / NULLIF(SUM(den) OVER w, 0) AS prior
        FROM grid
        WINDOW w AS (PARTITION BY pos ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ) e
    GROUP BY game_date
"""

# Game-eligible per-position aggregate — PA-weighted mean of the healthy
# pool (same participation weighting as _LINEUP_AGG_SQL), pivoted to one
# row per team-game with all 9 pool columns, then the empty-pool fallback:
# COALESCE(pool mean, league-by-position prior) so a thin/depleted position
# ships the honest prior instead of padding the pool or going NULL.
_POS_AGG_SQL = """
    CREATE TABLE pos_agg AS
    WITH pooled AS (
        SELECT game_date, game_pk, batting_team, pos,
               SUM(_pa30 * shrunk_xwoba) / NULLIF(SUM(_pa30), 0) AS px
        FROM pos_pool WHERE on_il = 0
        GROUP BY game_date, game_pk, batting_team, pos
    ),
    wide AS (
        SELECT game_date, game_pk, batting_team,
               MAX(CASE WHEN pos = 'c'  THEN px END) AS pl_c_xwoba,
               MAX(CASE WHEN pos = 'fb' THEN px END) AS pl_fb_xwoba,
               MAX(CASE WHEN pos = 'sb' THEN px END) AS pl_sb_xwoba,
               MAX(CASE WHEN pos = 'ss' THEN px END) AS pl_ss_xwoba,
               MAX(CASE WHEN pos = 'tb' THEN px END) AS pl_tb_xwoba,
               MAX(CASE WHEN pos = 'rf' THEN px END) AS pl_rf_xwoba,
               MAX(CASE WHEN pos = 'cf' THEN px END) AS pl_cf_xwoba,
               MAX(CASE WHEN pos = 'lf' THEN px END) AS pl_lf_xwoba,
               MAX(CASE WHEN pos = 'dh' THEN px END) AS pl_dh_xwoba
        FROM pooled
        GROUP BY game_date, game_pk, batting_team
    )
    SELECT w.game_date, w.game_pk, w.batting_team,
           COALESCE(w.pl_c_xwoba,  lg.pr_c)  AS pl_c_xwoba,
           COALESCE(w.pl_fb_xwoba, lg.pr_fb) AS pl_fb_xwoba,
           COALESCE(w.pl_sb_xwoba, lg.pr_sb) AS pl_sb_xwoba,
           COALESCE(w.pl_ss_xwoba, lg.pr_ss) AS pl_ss_xwoba,
           COALESCE(w.pl_tb_xwoba, lg.pr_tb) AS pl_tb_xwoba,
           COALESCE(w.pl_rf_xwoba, lg.pr_rf) AS pl_rf_xwoba,
           COALESCE(w.pl_cf_xwoba, lg.pr_cf) AS pl_cf_xwoba,
           COALESCE(w.pl_lf_xwoba, lg.pr_lf) AS pl_lf_xwoba,
           COALESCE(w.pl_dh_xwoba, lg.pr_dh) AS pl_dh_xwoba
    FROM wide w
    LEFT JOIN pos_league lg ON lg.game_date = w.game_date
"""

# Position-segmented league xwOBA prior — the shrinkage TARGET for each
# batter's trailing-30g rating (2026-10-02: the prior must be the batter's
# own position's average, not the overall league average — a catcher's
# thin sample shrinks toward catcher quality, a DH's toward DH quality).
# Structurally identical to batter_league: per-date sums of the SAME
# already-shifted rolling quantities (point-in-time safe — each value
# predates its own row's game), aggregated per position, cumulative
# through the current date within the position. Batters without a listed
# position simply never appear here; their rating falls back to the
# overall lg_xwoba via COALESCE at rating time.
_BATTER_LEAGUE_POS_SQL = """
    CREATE TABLE batter_league_pos AS
    SELECT game_date, pos,
        SUM(_xwoba_num30) OVER w
          / NULLIF(SUM(_xwoba_den30) OVER w, 0) AS lg_xwoba_pos
    FROM (
        SELECT b.game_date, pp.pos,
               SUM(b._xwoba_num30) AS _xwoba_num30,
               SUM(b._xwoba_den30) AS _xwoba_den30
        FROM batter_rolling b
        JOIN player_positions pp
          ON pp.batter = b.batter
         AND pp.season = EXTRACT(YEAR FROM b.game_date)
         AND pp.pos IN ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH')
        GROUP BY b.game_date, pp.pos
    ) t
    WINDOW w AS (PARTITION BY pos ORDER BY game_date
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
"""

# ── Bullpen availability (2026-09-30) ────────────────────────────────────
# The lineup family prices tonight's ROSTER (unavailable batters are
# filtered out of lineup_re24_* before the average). The bullpen family
# priced only INNINGS ALREADY PITCHED: an arm placed on the IL today
# kept contributing his pre-IL innings to every WHIP/ERA/pitch-count the
# model reads, and nothing told the model the pen is thinner than its own
# numbers. Fix (same directive as the lineup pool — NO new features):
# innings thrown by an unavailable arm are EXCLUDED from bullpen_raw
# before any window/season aggregate is formed, so the whole served
# family (bullpen_whip_10g_home/away, bullpen_whip_10g_diff,
# bullpen_whip_3g_*, bullpen_era_10g_*, bullpen_pitches_3d_* and the
# meltdown twins) inherits tonight's real availability.
#
# Ledger: the generalized availability table (build_il_stints.py →
# il_stints_pitchers.parquet) — IL, paternity, bereavement/family
# medical, restricted, administrative leave, suspension,
# optioned/reassigned — the same taxonomy the lineup filter consumes.
#
# PIT rule (pitcher-specific): innings ON the placement date stay
# available (the appearance is the announcement, thrown before it) and
# innings ON the return date stay available (the builder closes the
# stint AT the return appearance). So an arm is unavailable strictly
# BETWEEN his stint's dates:
#   unavailable(game_date) = il_start < game_date
#                            AND (il_end IS NULL OR game_date < il_end)
#
# Missing ledger degrades LOUDLY to all-pitched-innings semantics (the
# exact pre-availability behavior) — never a silent identity change.
# ── Bullpen readiness (2026-09-30) ──────────────────────────────────────
# Player-level availability LIKELIHOOD, derived from measured rest-usage
# physics (51,487 reliever outings 2024-2026; P(reliever appears in his
# team's next game | last-outing pitch count); year-stable within 0.02):
#   < 20 pitches -> 0.25 | 20-34 -> 0.13 | 35+ -> 0.007 (near-certain sit)
# a monotone staircase, recomputable from any season's data. Arms on the
# availability ledger are 0 by definition. bp_ready_share = the pen's
# last-2-day pitch-weighted readiness, NORMALIZED to [0, 1] by the light
# ceiling (1.0 = every recent arm threw <20 pitches = fully fresh; ~0.05 =
# all arms heavy = tomorrow's pen is spent; 0 = all ledgered); a pen with
# no recent outings has nothing known -> NULL (never a fake 0 or 1).
_BP_READY_PITCH_BUCKETS = (20, 35)   # (light ceiling, heavy floor)
_BP_READY_P_LIGHT = 0.25             # P(appear next day) for < 20 pitches
_BP_READY_P_MEDIUM = 0.13            # 20-34 pitches
_BP_READY_P_HEAVY = 0.007            # 35+ pitches (sits tomorrow)

# Over-budget exclusion (2026-09-30, same mechanism as the availability
# filter): a reliever appearance within 2 days AFTER a 35+ pitch outing is
# near-certainly an unavailability exception (P(appear) = 0.007 at 1 day
# rest, 0.066 at 2 - both below the 0.13 doubtful bar), so those rows are
# EXCLUDED from bullpen_raw before the WHIP/ERA windows form. Every served
# bullpen quality feature (bullpen_whip_10g home/away/diff, whip_3g,
# era_10g and the meltdown twins) therefore prices only innings from arms
# in a tonight-ready state. The fatigue channel (bullpen_pitches_3d) stays
# raw by design: the budget IS its signal. Window is STRICTLY the
# availability fact (arms return 18%+ by day 3, so day 3+ is a manager-
# trust question, not an availability fact); revisit via the member gate.
# REGRESSION NOTE (2026-09-30, Kaggle run): the 4d->2d tighten commit
# accidentally deleted the _BP_SPENT_PITCHES line below, which only
# surfaces at build time (the bp_spent f-string evaluates the name when
# _build_game_level runs) — imports and py_compile stay green. Restored
# verbatim from e0b5003; test_bp_fstring_constants_resolve now pins every
# _BP_* placeholder so this class of deletion cannot ship silently again.
_BP_SPENT_PITCHES = 35               # prior-outing ceiling for the spent
                                     # exclusion (P(appear) = 0.007 at 1d)
_BP_SPENT_LOOKBACK_DAYS = 2          # strict availability rule: P(appear)
                                     # 0.007 @ 1d / 0.066 @ 2d, both below the
                                     # 0.13 doubtful bar; arms return 18%+ by
                                     # day 3, so day 3+ is NOT an availability
                                     # fact (re-assess via the member gate)

# ── Bullpen opportunity-weighted shrinkage (2026-09-30) ───────────────
# The NBA player_ts / NHL player-rating convention, structurally aligned:
# a rate estimated from few opportunities is pulled toward the league
# reliever prior with prior weight k = 20% of the mean PLAYER-SEASON
# opportunity (pitches). Measured 2024-2026 on the availability+spent-
# filtered reliever population: mean ~464 pitches -> k ~= 93 pitches ~=
# 6.0 IP at 15.5 pitches/IP; ~24% of arm-seasons sit below k, so the
# prior genuinely bites on thin samples. k is derived from the frame
# each build (season-stable; derivation logged) with a hard floor — the
# portable 20% fraction is the point, not any one season's mean. RATES
# ONLY: bullpen_whip_10g, bullpen_era_10g, bullpen_whip_3g and the
# *_std season baselines are shrunk as w*rate + (1-w)*prior with
# w = window_IP/(window_IP + k_IP); workloads (bullpen_pitches_3d/
# ip_3d, budget_2d, ready_share) are COUNTS, not rate estimates, and
# stay raw — exactly why the NBA shrinks TS but never point totals.
# The 10g/3g windows roll across seasons (existing convention, shared
# with the SP 5g windows), so the blend weight (not a season partition)
# is what tames thin April windows. The league prior is strictly point-
# in-time (cumulative over prior CALENDAR dates, season-as-of with an
# all-history fallback) and drawn from the SAME availability+spent-
# filtered bullpen_raw population the windows see. Degenerate frames
# (no bp_outing rows) leave k NULL and the CASE guard ships raw rates —
# never a silent all-NULL.
# ADOPTED 2026-09-30 by owner decision (full-family arm): the formal
# 3-seed member gate on the production walk-forward measured the full
# arm at -0.00018 selection logloss vs baseline (fallback no-3g arm
# -0.00013; sealed holdout and the Mar-Apr slice agreed in sign) —
# below the +0.001 bar, adopted anyway for structural alignment with
# the NFL epa / NBA ts / NHL rating families. Measurement record:
# %TEMP%/bp_shrink_gate.py + bp_shrink_gate_results.csv (2026-09-30).
_BP_SHRINK_FRACTION = 0.20
_BP_PITCHES_PER_IP = 15.5    # measured pool constant (2024-2026)
_BP_MIN_SEASON_PITCHES = 30  # cameo player-seasons excluded from k's mean
_BP_K_FLOOR = 20.0           # pitch floor: k can never collapse to ~0

# ── Batter-rating shrinkage arm (A/B, 2026-10-02) ─────────────────────────
# Owner intent: shrinkage exists to hold back THIN player-level data, not
# to compress established separation forever. Two arms, selected per run:
#
#   bayesian (default, shipped)  w = n / (n + 120)
#       the previously-shipped empirical-Bayes form; the league prior
#       never fully washes out (50% own weight at n = 120).
#   ramp (candidate, opt-in via MLB_SHRINK_ARM=ramp)  w = min(n / 120, 1)
#       full own weight once 20%-of-a-season opportunity (120 PA/AB) is
#       reached, raw forever after. Continuous at the threshold — the
#       119 -> 120 step is only (mu - r_bar)/120, so a player crossing
#       it shows no wobble — and past the midpoint it keeps more of the
#       player's own data than bayesian does (n = 60: ramp 50%, bayes 33%).
#
# Both arms evaluate value = w * r_bar + (1 - w) * mu with the SAME
# point-in-time league prior (lg_woba default 0.315; lg_re24 default 0.0,
# centered on expectation by construction) and the SAME NULL gate: a
# rating row with zero opportunities stays NULL so pool joins behave
# identically under either arm.
#
# SCOPE: batter ratings only (shrunk_woba + shrunk_re24 -> lineup_*).
# The bullpen, sp_era_5g, win% and exp2 shrinkers are NOT in this A/B.
# Measured record — .adhoc shrink A/B on the production frame (7387
# decided games; sealed 21-day holdout 2026-09-11..2026-10-01): ramp
# holdout blend logloss 0.66578 vs bayesian 0.66564 (delta +0.00014,
# inside the ±0.001 adoption bar); walk OOF marginally favored ramp
# (-0.00019, 46/76 folds; lightgbm -0.00057, xgboost +0.00020) — a wash;
# lineup_* carries only ~2% of model weight either way. A feature-level
# re-gate the same date (paired nested logloss, treatment slice) found
# no difference either (all |z| <= 1.1).
# DEFAULT HELD AT BAYESIAN (2026-10-02): the adoption commit was
# un-committed so the code default keeps matching the shipped
# artifacts (the production frame was built bayesian at 11:20, before
# the ramp decision). Ramp stays one env var away —
# MLB_SHRINK_ARM=ramp — and can be re-adopted once production has
# actually shipped a ramp-built frame. test_shrink_arm.py pins both
# weight schedules.
MLB_SHRINK_ARM = os.getenv("MLB_SHRINK_ARM", "bayesian").strip().lower() or "bayesian"

BATTER_SHRINK_K = 120  # PA/AB — 20% of a ~600-opportunity season


def batter_rating_sql(arm: str) -> tuple[str, str]:
    """Return ``(shrunk_woba_expr, shrunk_re24_expr)`` SQL for ``arm``.

    Expressions reference ``r.*`` (batter_rolling) and ``l.*``
    (batter_league) — the aliases of the CREATE TABLE batter_ratings
    block — so the production SQL itself is what test_shrink_arm.py
    evaluates against single-row fixtures. Unknown arms raise: a typo
    must never silently ship the default.
    """
    k = BATTER_SHRINK_K
    if arm == "bayesian":
        return (
            "CASE WHEN COALESCE(r._ab, 0) > 0 THEN\n"
            f"    (r._woba_num + COALESCE(l.lg_woba, 0.315) * {k})\n"
            f"    / (r._ab + {k})\n"
            "END",
            "CASE WHEN COALESCE(r._pa30, 0) > 0 THEN\n"
            f"    (r._re24_num + COALESCE(l.lg_re24, 0.0) * {k})\n"
            f"    / (r._pa30 + {k})\n"
            "END",
        )
    if arm == "ramp":
        return (
            "CASE WHEN COALESCE(r._ab, 0) > 0 THEN\n"
            f"    CASE WHEN r._ab >= {k} THEN r._woba_num / r._ab\n"
            f"         ELSE (r._woba_num + ({k} - r._ab)\n"
            "               * COALESCE(l.lg_woba, 0.315))\n"
            f"         / {k}.0\n"
            "    END\n"
            "END",
            "CASE WHEN COALESCE(r._pa30, 0) > 0 THEN\n"
            f"    CASE WHEN r._pa30 >= {k} THEN r._re24_num / r._pa30\n"
            f"         ELSE (r._re24_num + ({k} - r._pa30)\n"
            "               * COALESCE(l.lg_re24, 0.0))\n"
            f"         / {k}.0\n"
            "    END\n"
            "END",
        )
    raise ValueError(
        f"MLB_SHRINK_ARM must be 'bayesian' or 'ramp', got {arm!r}")


def batter_xwoba_rating_sql(arm: str) -> str:
    """Return the ``shrunk_xwoba`` SQL for ``arm`` (the pl_* pool rating).

    Same two-arm contract as :func:`batter_rating_sql`: blend the
    trailing-30g non-null xwOBA numerator (Statcast
    ``estimated_woba_using_speedangle`` over PA-ending events — the exp2
    non-null-only convention) toward a point-in-time league xwOBA with
    prior weight ``k = BATTER_SHRINK_K``.

    ZERO-OPPORTUNITY ROWS SHIP THE PRIOR (2026-10-03 membership audit):
    the old NULL gate dropped a zero-PA rating row from every pool — a
    season debut, or a window whose only PA-ending events carried no
    measured estimate — which is precisely the membership hole the
    effective-nine fix closes. With zero observations the honest estimate
    IS the prior, so both arms COALESCE their window sums and return it
    instead of NULL. Under the PA-weighted pool aggregates a zero-PA row
    carries weight 0 anyway; what changes is that the row EXISTS, so the
    QUALIFY dedupe, the membership clause, and coverage see the batter.

    The prior is POSITION-SEGMENTED (2026-10-02): first the batter's own
    position's league average (``lp.lg_xwoba_pos`` from
    ``batter_league_pos``), falling back to the overall ``l.lg_xwoba``
    when the batter has no listed position, then the 0.315 literal on
    day one — so a catcher's thin sample shrinks toward catcher quality,
    not the DH average. References ``r.*`` (batter_rolling), ``l.*``
    (batter_league) and ``lp.*`` (batter_league_pos); unknown arms raise
    (a typo must never ship the default).
    """
    k = BATTER_SHRINK_K
    prior = "COALESCE(lp.lg_xwoba_pos, l.lg_xwoba, 0.315)"
    if arm == "bayesian":
        return (
            "    (COALESCE(r._xwoba_num30, 0) + "
            f"{prior} * {k})\n"
            f"    / (COALESCE(r._xwoba_den30, 0) + {k})"
        )
    if arm == "ramp":
        return (
            f"CASE WHEN COALESCE(r._xwoba_den30, 0) >= {k}\n"
            "     THEN COALESCE(r._xwoba_num30, 0) / r._xwoba_den30\n"
            f"     ELSE (COALESCE(r._xwoba_num30, 0)\n"
            f"           + ({k} - COALESCE(r._xwoba_den30, 0)) * {prior})\n"
            f"          / {k}.0\n"
            "END"
        )
    raise ValueError(
        f"MLB_SHRINK_ARM must be 'bayesian' or 'ramp', got {arm!r}")

# Recent pitcher runs/9 is blended toward the same pitcher's non-overlapping
# older appearance history using 30 pseudo innings. This is an explicit,
# conservative first-pass regularizer; unlike bullpen shrinkage, it has not yet
# been selected by a full walk-forward search. Sparse/no older history falls
# back to a strictly prior league runs/9 rate, then to the raw recent rate.
_SP_ERA_5G_SHRINK_IP = 30.0

# Zero-prior fallback for SP pitch-category cells (2026-10-01 exp2 coverage
# remediation; mirrors the 6b3b3e2 shrinkage philosophy at the feature layer).
# The exp2 matchup products multiply SP deviation-from-league factors, so a
# category with zero prior tracked PAs renders as league-average rate (SP
# deviation exactly 0 = no information) and zero tracked usage — the honest
# "no matchup edge" — instead of NaN riding per-fold median imputation, which
# fabricated "league-average offspeed" for ~43% of offspeed rows. Team-side
# gaps are NOT backfilled here: the opponent's history is real information,
# never fabricated. Templates so unit tests can assert the exact emitted SQL.
_EXP2_SP_CAT_RATE_SQL = "COALESCE(%s, %s)"
_EXP2_SP_USAGE_SQL = "COALESCE(%s, 0.0)"

# Zero-prior SP category priors (module constants so the emitted SQL is
# unit-testable against tiny fabricated inputs — see
# test_exp2_no_history_fallback.py).
_EXP2_SP_CAT_PRELIM_SQL = f"""
        CREATE TABLE exp2_sp_cat_prelim AS
        SELECT g.game_date, g.pitcher, g.season, g.pitch_cat,
            {_EXP2_SP_CAT_RATE_SQL % ("c.k_thru / NULLIF(c.pa_thru, 0)",
                                      "lg.league_k_pct_cat")}
                AS sp_k_pct_cat,
            {_EXP2_SP_CAT_RATE_SQL % ("c.xwo_thru / NULLIF(c.xwon_thru, 0)",
                                      "lg.league_xwoba_cat")}
                AS sp_xwoba_cat,
            c.pa_thru AS sp_pa_cat_cum
        FROM (
            SELECT d.game_date, d.pitcher, d.season, cats.pitch_cat
            FROM (SELECT DISTINCT game_date, pitcher, season
                  FROM exp2_sp_cat_daily) d
            CROSS JOIN (VALUES ('fastball'), ('breaking'), ('offspeed'))
                   AS cats(pitch_cat)
        ) g
        ASOF LEFT JOIN exp2_sp_cat_cum c
          ON g.pitcher = c.pitcher AND g.season = c.season
         AND g.pitch_cat = c.pitch_cat AND g.game_date > c.game_date
        ASOF LEFT JOIN exp2_league_cat lg
          ON g.pitch_cat = lg.pitch_cat AND g.game_date >= lg.game_date
    """

_EXP2_SP_CAT_SQL = f"""
        CREATE TABLE exp2_sp_cat AS
        SELECT g.game_date, g.pitcher, g.pitch_cat,
            g.sp_k_pct_cat,
            g.sp_xwoba_cat,
            -- Usage: prior PAs of this category / prior PAs across all three
            -- tracked categories (sums to 1 across categories, same as-of
            -- date). A zero-prior category is a KNOWN ZERO of tracked usage,
            -- not a missing measurement: ships 0.0 so the exp2 product
            -- collapses to no-edge (2026-10-01) instead of NaN→imputation
            -- which told the model the pitcher throws league-average
            -- offspeed. Day-1 rows stay NaN via 0 * NULL rate propagation.
            {_EXP2_SP_USAGE_SQL % "g.sp_pa_cat_cum / NULLIF(t.pa_thru, 0)"}
                AS sp_usage_cat
        FROM exp2_sp_cat_prelim g
        ASOF LEFT JOIN exp2_sp_cat_tot_cum t
          ON g.pitcher = t.pitcher AND g.season = t.season
         AND g.game_date > t.game_date
    """

_EXP2_SP_FBHAND_SQL = f"""
        CREATE TABLE exp2_sp_fbhand AS
        SELECT g.game_date, g.pitcher, g.stand,
            -- Zero-prior handedness cell: the strictly-prior league rate for
            -- that stand (same no-edge fallback as exp2_sp_cat above).
            {_EXP2_SP_CAT_RATE_SQL % ("c.k_thru / NULLIF(c.pa_thru, 0)",
                                      "lg.league_k_pct_fb_vs")}
                AS sp_k_pct_fb_vs,
            c.pa_thru AS sp_pa_fb_vs
        FROM (
            SELECT d.game_date, d.pitcher, d.season, h.st AS stand
            FROM (SELECT DISTINCT game_date, pitcher, season
                  FROM exp2_sp_fbhand_daily) d
            CROSS JOIN (VALUES ('L'), ('R')) AS h(st)
        ) g
        ASOF LEFT JOIN exp2_sp_fbhand_cum c
          ON g.pitcher = c.pitcher AND g.season = c.season
         AND g.stand = c.stand AND g.game_date > c.game_date
        ASOF LEFT JOIN exp2_league_fbhand lg
          ON g.stand = lg.stand AND g.game_date >= lg.game_date
    """

_BP_ARM_UNAVAILABLE_SQL = """EXISTS (
              SELECT 1 FROM il_stints_pitchers i
              WHERE i.batter = p.pitcher
                AND CAST(p.game_date AS DATE) > CAST(i.il_start AS DATE)
                AND (i.il_end IS NULL OR CAST(p.game_date AS DATE)
                     < CAST(i.il_end AS DATE)))"""

# Parameterized ledger predicates so EVERY consumer binds availability
# conditionally on the registered ledger. 2026-09-30: the retention sweep
# deleted il_stints_pitchers.parquet from the repo; the guarded paths
# degraded loudly as designed, but bp_day2 and bp_shrink_prior carried
# hard table references and crashed the Kaggle build with a
# CatalogException. Fragments are interpolated only when _pitchers_ok —
# absent-ledger builds fall back to pre-availability semantics (loud),
# never a missing-table error.
_BP_LEDGER_EXCLUDE_ROW_SQL = """NOT EXISTS (
              SELECT 1 FROM il_stints_pitchers i
              WHERE i.batter = {alias}.pitcher
                AND CAST({date} AS DATE) > CAST(i.il_start AS DATE)
                AND (i.il_end IS NULL OR CAST({date} AS DATE)
                     < CAST(i.il_end AS DATE)))"""

_BP_LEDGER_EXCLUDE_ASOF_SQL = """EXISTS (
              SELECT 1 FROM il_stints_pitchers i
              WHERE i.batter = {alias}.pitcher
                AND {ref} > CAST(i.il_start AS DATE)
                AND (i.il_end IS NULL OR {ref} < CAST(i.il_end AS DATE)))"""

# ── SP staleness gate (2026-09-30) ──────────────────────────────────────
# Same directive and mechanism as the bullpen availability filter, applied
# to the STARTING-PITCHER stat chain. The SP family prices each starter's
# OWN prior appearances (LAG-shifted), so a starter returning from a long
# availability stint shipped his PRE-STINT cumulative form with nothing
# marking the interruption. Measured 2024-2026 (appearance-level, ledger
# reconciled): return appearances after a >=14d stint-crossing gap —
# K/9 8.49 / WHIP 1.450 / xwOBA .318 vs 8.82 / 1.317 / .301 on normal 3-7d
# rotation rest; at MATCHED 8-13d rest, stint-returners K/9 7.94 vs 8.70
# without a stint and WHIP +0.066 vs matched >=14d no-stint controls (the
# stint, not the rest, drives the drop); 331 of 4,858 2026 starter slots
# carried a >=14d gap, 201 of them availability returns (in-season long
# gaps are ~94% returns). Fix: an appearance whose gap to the pitcher's
# previous appearance is >= _SP_STALE_GAP_DAYS AND overlaps an availability
# stint is STALE — every SP stat derived from that appearance ships NULL
# (the model prices that start on team-level evidence); all other rows are
# byte-identical. Gates the per-pitcher SOURCES (pitcher_season_features,
# pitcher_features, pitcher_stuff) so the whole served SP family inherits
# it: sp_era/k9 (season to date), sp_era_5g/k9_5g (last 5),
# sp_bb9/whip/fip/xwoba (trailing-6-appearance window), sp_fbvelo/fbpct/
# whiff_3g and sp_xwoba_vs_l/r. The exp2 SP-category chain is date-level
# ASOF season-to-date feeding candidate-only columns — out of scope.
# Ledger taxonomy: the SAME generalized availability definition as the
# bullpen filter (IL + paternity + bereavement/family medical + restricted
# + administrative leave + suspension + optioned/reassigned) — non-medical
# leaves included by construction.
# 10 days: the measured quality jump sits at >=14d gaps; 7-day IL and
# paternity stints resume within the normal rotation band (5-day turns,
# 8-13d holds), so 10 separates availability returns from routine turns
# without gating them.
# Missing ledger degrades LOUDLY to no-gating (exact pre-gate semantics).
_SP_STALE_GAP_DAYS = 10


def _register_sp_staleness_gate(con) -> bool:
    """Flag per-appearance rows whose prior-appearance gap crosses an
    availability stint (see _SP_STALE_GAP_DAYS). Builds the TEMP table
    ``pitcher_stale`` (game_date, game_pk, pitcher, sp_stale) consumed by
    the per-pitcher feature tables; False (loudly) when the ledger is
    absent — consumers then join an EMPTY gate table, i.e. no gating."""
    ok = _register_il_stints_pitchers(con)
    if not ok:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE pitcher_stale AS
            SELECT * FROM (SELECT
                CAST(NULL AS DATE) AS game_date,
                CAST(NULL AS BIGINT) AS game_pk,
                CAST(NULL AS BIGINT) AS pitcher,
                FALSE AS sp_stale) WHERE FALSE
        """)
        return False
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pitcher_stale AS
        WITH app AS (
            SELECT game_date, game_pk, pitcher,
                LAG(game_date) OVER wprev AS prev_date,
                date_diff('day', LAG(game_date) OVER wprev, game_date)
                    AS gap_days
            FROM pitcher_game_stats
            WINDOW wprev AS (PARTITION BY pitcher
                             ORDER BY game_date, game_pk)
        )
        SELECT game_date, game_pk, pitcher,
            COALESCE(gap_days >= {_SP_STALE_GAP_DAYS}, FALSE)
              AND EXISTS (SELECT 1 FROM il_stints_pitchers i
                          WHERE i.batter = app.pitcher
                            AND CAST(i.il_start AS DATE) <= app.game_date
                            AND (i.il_end IS NULL
                                 OR CAST(i.il_end AS DATE) >= app.prev_date)
                         ) AS sp_stale
        FROM app
    """)
    n, n_stale = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN sp_stale THEN 1 ELSE 0 END) "
        "FROM pitcher_stale").fetchone()
    logger.info("SP staleness gate: %d appearance rows, %d flagged stale "
                "(gap >= %dd overlapping an availability stint)",
                n, n_stale or 0, _SP_STALE_GAP_DAYS)
    return True



def export_bullpen_availability(con, day) -> "pd.DataFrame | None":
    """Per-pitcher availability card for the upcoming slate (2026-09-30).

    For every arm who pitched in the last 3 days: his next-day appearance
    likelihood from the measured workload staircase (see _BP_READY_*),
    ledger status, last-outing workload, and the team's last played date.
    Runs on the CALLER's connection (reads the bp_outing table the build
    just produced); returns None when that table is absent (no pitches
    loaded) — never raises. The pipeline writes the frame as
    bullpen_availability_<date>.csv next to the other slate artifacts.
    """
    try:
        df = con.execute("""
            SELECT o.pitcher, o.team, o.game_date AS last_outing,
                   o.n_pitches AS last_outing_pitches,
                   CASE WHEN EXISTS (SELECT 1 FROM il_stints_pitchers i
                                     WHERE i.batter = o.pitcher
                                       AND CAST(? AS DATE) > CAST(i.il_start AS DATE)
                                       AND (i.il_end IS NULL
                                            OR CAST(? AS DATE)
                                               < CAST(i.il_end AS DATE)))
                        THEN 'ON_LEDGER'
                        WHEN o.n_pitches >= 35 THEN 'UNLIKELY (0.007)'
                        WHEN o.n_pitches >= 20 THEN 'DOUBTFUL (0.13)'
                        ELSE 'LIKELY (0.25)' END AS next_day_status,
                   CASE WHEN EXISTS (SELECT 1 FROM il_stints_pitchers i
                                     WHERE i.batter = o.pitcher
                                       AND CAST(? AS DATE) > CAST(i.il_start AS DATE)
                                       AND (i.il_end IS NULL
                                            OR CAST(? AS DATE)
                                               < CAST(i.il_end AS DATE)))
                        THEN 0 WHEN o.n_pitches >= 35 THEN 0.007
                        WHEN o.n_pitches >= 20 THEN 0.13 ELSE 0.25 END
                        AS p_available,
                   MAX(o.game_date) OVER (PARTITION BY o.team)
                        AS team_last_played
            FROM bp_outing o
            WHERE o.game_date >= ?
            ORDER BY o.team, p_available, o.n_pitches DESC
        """, [pd.Timestamp(day), pd.Timestamp(day),
                 pd.Timestamp(day), pd.Timestamp(day),
                 pd.Timestamp(day) - pd.Timedelta(days=3)]).df()
        return df if not df.empty else None
    except Exception as e:  # noqa: BLE001
        logger.warning("export_bullpen_availability unavailable: %s", e)
        return None


def il_stints_freshness() -> dict:
    """Coverage horizon of the committed IL table, for staleness checks.

    Prefers the builder's provenance meta (``window_end`` = the last
    transaction date the fetch saw); falls back to the latest stint
    boundary inside the parquet itself (max of coalesce(il_end,
    il_start) — an activation recorded later than any start is the
    later knowledge date). Returns {} when neither is readable; the
    tripwire must never break feature building.
    """
    path = il_stints_dir() / IL_STINTS_FILE
    out: dict = {}
    try:
        import json
        meta = json.loads(
            (il_stints_dir() / IL_STINTS_META_FILE).read_text())
        if meta.get("window_end"):
            out["window_end"] = str(meta["window_end"])
    except Exception:
        pass
    try:
        import duckdb
        lit = str(path).replace("\\", "/")
        c = duckdb.connect(database=":memory:")
        sig = c.execute(
            f"SELECT max(COALESCE(il_end, il_start)) FROM "
            f"read_parquet('{lit}')").fetchone()[0]
        c.close()
        if sig is not None:
            out["parquet_max_stint_date"] = str(pd.Timestamp(sig).date())
    except Exception:
        pass
    return out


def _register_il_stints_pitchers(con) -> bool:
    """Load the PITCHER availability ledger; False (loudly) when absent."""
    p = il_stints_dir() / IL_STINTS_PITCHERS_FILE
    if not p.exists():
        logger.warning(
            "il_stints_pitchers.parquet missing: bullpen availability "
            "degrades to exposure 0 / count 0 (pre-availability semantics)")
        return False
    try:
        con.execute(
            "CREATE OR REPLACE TEMP TABLE il_stints_pitchers AS "
            "SELECT * FROM read_parquet(?)", [str(p)])
        n = con.execute(
            "SELECT count(*), count(DISTINCT batter) "
            "FROM il_stints_pitchers").fetchone()
        logger.info("pitcher availability ledger: %d stints, %d pitchers",
                    n[0], n[1])
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("il_stints_pitchers.parquet unreadable (%s); "
                       "bullpen availability degrades to exposure 0", e)
        return False


def _register_il_stints(con: "duckdb.DuckDBPyConnection") -> bool:
    """Load the IL cache into ``con``; False (loudly) when unavailable.

    Returns False (never raises) so a missing/stale cache degrades the
    feature to the participant pool instead of failing the whole build.
    """
    path = il_stints_dir() / IL_STINTS_FILE
    if not path.exists():
        logger.warning(
            "%s not found in %s — expected lineups fall back to the participant "
            "pool (no OUT/IR eligibility filter, lineup_il_flag_* forced 0). "
            "Run build_il_stints.py to restore the filter.",
            IL_STINTS_FILE, _lineup_base_dir())
        return False
    try:
        lit = str(path).replace("\\", "/")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE il_stints AS
            SELECT CAST(batter AS BIGINT) AS batter,
                   CAST(il_start AS DATE) AS il_start,
                   CAST(il_end AS DATE) AS il_end
            FROM read_parquet('{lit}')
        """)
        n, nb = con.execute(
            "SELECT count(*), count(DISTINCT batter) FROM il_stints").fetchone()
    except Exception as e:
        logger.warning("il_stints.parquet present but unreadable (%s); "
                       "expected lineups fall back to the participant pool", e)
        return False
    if not n:
        logger.warning("il_stints.parquet is EMPTY; expected lineups fall "
                       "back to the participant pool")
        return False
    # STALENESS TRIPWIRE (2026-09-28 Aaron Judge audit): the table is a
    # CACHE of transactions, rebuilt out-of-band. A player placed on the
    # IL AFTER the table's coverage horizon is invisible to the
    # eligibility filter, and his frozen _pa30 keeps him ranked inside
    # the projected nine — the inclusion this table exists to prevent.
    # Compare the horizon against the data horizon of the frame being
    # built (the pitches table) and warn loudly past the lag budget.
    horizon = il_stints_freshness()
    signal = horizon.get("window_end") or horizon.get(
        "parquet_max_stint_date")
    try:
        ref = con.execute(
            "SELECT max(game_date) FROM pitches").fetchone()[0]
    except Exception:
        ref = None
    if signal and ref is not None:
        lag = (pd.Timestamp(ref).normalize()
               - pd.Timestamp(signal)).days
        logger.info("injured list: %d stints, %d batters; coverage "
                    "through %s (lag %dd vs data horizon %s)",
                    n, nb, signal, lag, pd.Timestamp(ref).date())
        if lag > IL_STINTS_MAX_LAG_DAYS:
            logger.warning(
                "il_stints.parquet coverage stops %s — %d days behind the "
                "data horizon (%s). A player PLACED on the IL after %s "
                "stays in the projected nine (frozen _pa30 keeps him "
                "ranked high — the 2026-09-28 Aaron Judge audit failure "
                "mode). Run: python build_il_stints.py --end <today> "
                "--refresh", signal, lag, pd.Timestamp(ref).date(),
                signal)
    else:
        logger.info("injured list: %d stints, %d batters", n, nb)
    return True


def _player_positions_candidates() -> list:
    """Where the position map may live, most authoritative first.

    The run cache (``MLB_IL_STINTS_DIR``) wins when present — that is what a
    local refresh writes — but nothing in the DAILY run produces it there,
    so the committed repo copy has to be the fallback. Measured on the
    2026-10-03 remote run: the run cache had no map, the repo copy was
    complete (4,381 rows, seasons 2024-2026), and because we only ever
    looked at the cache, all 24 served ``pl_*`` columns shipped at 0%
    coverage and were median-imputed as constants.
    """
    out: list[Path] = []
    for base in (il_stints_dir(), _lineup_base_dir()):
        p = base / PLAYER_POSITIONS_FILE
        if p not in out:
            out.append(p)
    return out


def _warn_player_positions_staleness(con) -> None:
    """Loud when the position map does not reach the data horizon's season.

    A map covering 2024-2026 prices 2024-2026. At the 2027 opener every
    ``pl_*`` pool would find no rows and the same 24 columns would go
    median-imputed — the 2026-10-03 failure shape, one season early.
    """
    try:
        max_season = con.execute(
            "SELECT max(season) FROM player_positions").fetchone()[0]
        horizon = con.execute("SELECT max(game_date) FROM pitches").fetchone()[0]
    except Exception:  # noqa: BLE001 — the tripwire never breaks a build
        return
    if max_season is None or horizon is None:
        return
    hs = pd.Timestamp(horizon).year
    if int(max_season) < hs:
        logger.warning(
            "player_positions covers through season %s but the data horizon "
            "is %d — the current season has no position map, so every pl_* "
            "pool for those games is EMPTY and 24 served columns ship "
            "median-imputed. Refresh the map before betting this season.",
            int(max_season), hs)


def _register_player_positions(con: "duckdb.DuckDBPyConnection") -> bool:
    """Load the StatsAPI position map into ``con``; False (loudly) when absent.

    ``player_positions.parquet`` (batter, season, pos) keys the pl_*
    position pools. Resolution order is _player_positions_candidates().
    Absence never fails the build: the caller creates an empty-but-well-
    formed ``pos_agg`` so every game_level LEFT JOIN stays bound while the
    pl_* family ships NULL under the warnings below.
    """
    candidates = _player_positions_candidates()
    present = [p for p in candidates if p.exists()]
    if not present:
        logger.warning(
            "%s not found in any of %s — pl_* position-pool xwOBA features "
            "degrade to NULL (empty pos_agg), i.e. 24 of the 109 served "
            "columns ship median-imputed constants. Run "
            ".adhoc/mlb_fetch_positions.py to restore the position map.",
            PLAYER_POSITIONS_FILE,
            ", ".join(str(p) for p in candidates))
        return False
    last_err = None
    for path in present:
        try:
            lit = str(path).replace("\\", "/")
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE player_positions AS
                SELECT CAST(batter AS BIGINT) AS batter,
                       CAST(season AS INTEGER) AS season,
                       UPPER(TRIM(pos)) AS pos
                FROM read_parquet('{lit}')
            """)
            n, nb = con.execute(
                "SELECT count(*), count(DISTINCT batter) "
                "FROM player_positions").fetchone()
        except Exception as e:  # noqa: BLE001
            logger.warning("%s unreadable (%s) — trying the next source",
                           path, e)
            last_err = e
            continue
        if not n:
            logger.warning("%s is EMPTY — trying the next source", path)
            continue
        _warn_player_positions_staleness(con)
        logger.info("player positions: %d rows, %d batters (source: %s)",
                    n, nb, path)
        return True
    logger.warning(
        "no readable %s among %s (last error: %s) — pl_* degrades to NULL",
        PLAYER_POSITIONS_FILE, ", ".join(str(p) for p in present), last_err)
    return False


def _register_lineups(con: "duckdb.DuckDBPyConnection") -> bool:
    """Load the announced-batting-order cache into ``con``; False when absent.

    ``lineups.parquet`` (game_pk, game_date, home/away_team, complete
    home_order/away_order of 9 MLBAM ids) is tier 1 of
    lineup_effective — tonight's nine. Absent/unreadable degrades LOUDLY
    to an empty ``lineups_raw`` so lineup_effective comes up without
    announced orders and every pool falls back to the pre-audit semantics
    (never a build failure, never a fabricated lineup).
    """
    path = _lineup_base_dir() / LINEUPS_FILE
    if not path.exists():
        logger.warning("%s not found in %s — membership degrades to "
                       "projected-only then full-roster (no announced nine). "
                       "Run backfill_lineups.py to restore tier 1.",
                       LINEUPS_FILE, _lineup_base_dir())
        return False
    try:
        lit = str(path).replace("\\", "/")
        # Column-existence probe BEFORE the SELECT: read_parquet raises on a
        # column that is absent (it does NOT return NULL), and the committed
        # cache predates the hand columns. A legacy cache must still load —
        # degrade to NULL hands (tier 2 runs unconditioned, loudly) rather
        # than fail the whole membership build over two optional booleans.
        have = set()
        try:
            have = {r[0] for r in con.execute(
                f"SELECT name FROM (DESCRIBE SELECT * FROM read_parquet('{lit}'))")}
        except Exception:  # noqa: BLE001 — unreadable cache handled below
            have = set()
        _hand_cols = ("home_starter_hand, away_starter_hand"
                      if {"home_starter_hand", "away_starter_hand"} <= have
                      else "CAST(NULL AS VARCHAR) AS home_starter_hand, "
                          "CAST(NULL AS VARCHAR) AS away_starter_hand")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE lineups_raw AS
            SELECT CAST(game_pk AS BIGINT) AS game_pk,
                   CAST(game_date AS DATE) AS game_date,
                   home_team, away_team,
                   home_order, away_order,
                   complete_home, complete_away,
                   {_hand_cols}
            FROM read_parquet('{lit}')
        """)
        n, n_complete = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE complete_home "
            "OR complete_away) FROM lineups_raw").fetchone()
    except Exception as e:  # noqa: BLE001
        logger.warning("%s present but unreadable (%s) — membership degrades "
                       "to projected-only then full-roster", LINEUPS_FILE, e)
        return False
    if not n:
        logger.warning("%s is EMPTY — membership degrades to "
                       "projected-only then full-roster", LINEUPS_FILE)
        return False
    logger.info("lineups: %d games (%d with a complete nine)", n, n_complete)
    return True

# Experiment-only superset (audit_pitcher_era_k9.py): keeps intent_walk /
# truncated_pa PAs as pa_boundary rows so they stop vanishing from
# pitcher_game_stats (their runs no longer leak into the NEXT PA's score
# delta). Used ONLY when build_features(corrected_outs=True) is requested
# by the pitcher-stats ablation; the production path uses PA_END_EVENTS.
#
# GATE VERDICT 2026-08-31: REJECT (DON'T ADOPT). Full-ensemble walk-forward
# on production-exact frames (data_delivery/pitcher_ensemble_gate_81aea53.json)
# showed pooled-neutral blend with degraded member ECE (LGB 0.0295->0.0329,
# XGB 0.0085->0.0125) and a sealed raw-ECE spike (+0.0075); the sealed AUC
# gain (+0.0058) did not survive pooled. The flag stays OFF by default.
PA_END_EVENTS_CORRECTED = (
    "'single', 'double', 'triple', 'home_run',"
    "'strikeout', 'strikeout_double_play',"
    "'walk', 'hit_by_pitch',"
    "'field_out', 'field_error', 'fielders_choice', 'fielders_choice_out',"
    "'grounded_into_double_play', 'double_play', 'triple_play',"
    "'sac_fly', 'sac_bunt', 'sac_fly_double_play',"
    "'catcher_interf', 'batter_interference',"
    "'force_out', 'sacrifice_bunt_double_play',"
    "'intent_walk', 'truncated_pa'"
)


def _connect(pitches_path: Path) -> duckdb.DuckDBPyConnection:
    """Open DuckDB in-memory, load pitches from Parquet, tune for low RAM."""
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '4GB'")
    # Allow generous temp space for spilling to disk (critical for large
    # queries). The temp dir must be portable: a hardcoded POSIX '/tmp' path
    # fails to create on Windows (verified live), so use the system temp dir.
    _duckdb_tmp = os.path.join(tempfile.gettempdir(), "duckdb_temp").replace("\\", "/")
    con.execute(f"SET temp_directory = '{_duckdb_tmp}'")
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
        "event", "type",
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
        "post_home_score", "post_away_score", "launch_speed_angle",
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
            LAG(xwoba_vs_r, 1) OVER ws AS _s_xr,
            -- Season-partitioned twins for the season-to-date baselines
            -- (a season opener's LAG is NULL, so the prior October never
            -- leaks into the expanding season mean).
            LAG(fb_velo, 1) OVER ws AS _s_fb_velo_s,
            LAG(fb_pct, 1) OVER ws AS _s_fb_pct_s,
            LAG(n_pitches, 1) OVER ws AS _s_n_s,
            LAG(whiffs, 1) OVER ws AS _s_whiffs_s
        FROM per_start
        WINDOW w AS (PARTITION BY pitcher ORDER BY game_date),
               ws AS (PARTITION BY pitcher, season ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE pitcher_stuff AS
        SELECT pitcher_stuff_raw.game_date, pitcher_stuff_raw.game_pk,
               pitcher_stuff_raw.pitcher,
            -- SP staleness gate: same NULLing rule as the other SP sources.
            -- sp_xwoba_vs_l/r (season-partitioned cumulative) and the
            -- season-to-date *_std stuff baselines are left ungated: they
            -- feed candidate-only momentum/diff plumbing, not the served
            -- per-side SP columns.
            CASE WHEN st.sp_stale THEN NULL
                 ELSE AVG(_s_fb_velo) OVER w3 END AS sp_fbvelo_3g,
            CASE WHEN st.sp_stale THEN NULL
                 ELSE AVG(_s_fb_pct) OVER w3 END AS sp_fbpct_3g,
            CASE WHEN st.sp_stale THEN NULL
                 ELSE SUM(_s_whiffs) OVER w3
                      / NULLIF(SUM(_s_n) OVER w3, 0) END AS sp_whiff_3g,
            AVG(_s_xl) OVER wall AS sp_xwoba_vs_l,
            AVG(_s_xr) OVER wall AS sp_xwoba_vs_r,
            -- Season-to-date stuff baselines (season-partitioned LAG twins,
            -- so an opener's row is NULL and never averages the prior
            -- October) — momentum companion for the 3g windows.
            AVG(_s_fb_velo_s) OVER wall AS sp_fbvelo_std,
            AVG(_s_fb_pct_s) OVER wall AS sp_fbpct_std,
            SUM(_s_whiffs_s) OVER wall / NULLIF(SUM(_s_n_s) OVER wall, 0) AS sp_whiff_std
        FROM pitcher_stuff_raw
        LEFT JOIN pitcher_stale st
               ON pitcher_stuff_raw.game_date = st.game_date
              AND pitcher_stuff_raw.game_pk = st.game_pk
              AND pitcher_stuff_raw.pitcher = st.pitcher
        WINDOW w3 AS (PARTITION BY pitcher_stuff_raw.pitcher
                      ORDER BY pitcher_stuff_raw.game_date
                      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
               wall AS (PARTITION BY pitcher_stuff_raw.pitcher,
                              pitcher_stuff_raw.season
                        ORDER BY pitcher_stuff_raw.game_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

def _build_game_level(con: duckdb.DuckDBPyConnection,
                       corrected_outs: bool = False) -> None:
    """Build game_level table via pure DuckDB SQL.  All intermediate tables
    are dropped after assembly to free RAM.

    corrected_outs: EXPERIMENTAL pitcher-stat semantics behind a flag
    (measured by run_pitcher_stats_ablation.py, NEVER shipped silently).
    Fixes the outs_on_pa map (force_out and fielders_choice credit their
    outs; batter_interference too) and keeps intent_walk / truncated_pa
    PAs as boundaries in pa_boundary so they stop vanishing from
    pitcher_game_stats. Everything downstream (sp_era, sp_k9, bb9, whip,
    bullpen family) re-derives from the corrected pitcher_game_stats.
    When False, the SQL produced is byte-identical to the pre-flag
    production path.
    """

    logger.info("Building game-level features (corrected_outs=%s)...",
                corrected_outs)

    # 1. Game winners (last pitch of each game)
    con.execute("""
        CREATE TABLE game_winners AS
        WITH last_pitch AS (
            SELECT game_pk, game_date, game_type, home_team, away_team,
                   post_home_score AS home_score,
                   post_away_score AS away_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk
                       ORDER BY at_bat_number DESC, pitch_number DESC
                   ) AS rn
            FROM pitches
        )
        SELECT game_pk,
               CAST(game_date AS DATE) AS game_date,
               game_type,
               home_team, away_team,
               home_score, away_score,
               CASE WHEN home_score > away_score THEN 1.0
                    WHEN away_score > home_score THEN 0.0
                    ELSE NULL END AS home_win,
               (home_score + away_score) AS total_runs
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
    _pa_events = PA_END_EVENTS_CORRECTED if corrected_outs else PA_END_EVENTS
    _outs_fix_whens = (
        "WHEN 'force_out' THEN 1\n"
        "                    WHEN 'fielders_choice' THEN 1\n"
        "                    WHEN 'batter_interference' THEN 1\n"
        "                    " if corrected_outs else "")
    con.execute(f"""
        CREATE TABLE pa_boundary AS
        WITH lastp AS (
            SELECT CAST(game_date AS DATE) AS game_date,
                   game_pk, inning, inning_topbot, at_bat_number,
                   pitcher, events,
                   launch_speed, launch_angle,
                   home_score + away_score AS pre_tot_score,
                   post_home_score + post_away_score AS tot_score,
                   launch_speed_angle,
                   estimated_woba_using_speedangle AS xwoba_val,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_pk, inning, inning_topbot, at_bat_number
                       ORDER BY pitch_number DESC
                   ) AS rn
            FROM pitches
            WHERE events IN ({_pa_events})
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
               -- Scores are POST-pitch, not the pre-pitch snapshot. Missing
               -- post scores remain unknown; never shift scoring to the next PA.
               CASE WHEN tot_score IS NULL OR COALESCE(prev_tot, pre_tot_score) IS NULL
                    THEN NULL
                    ELSE GREATEST(tot_score - COALESCE(prev_tot, pre_tot_score), 0)
                    END AS runs_on_pa,
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
                    {_outs_fix_whens}
                    ELSE 0 END AS outs_on_pa,
               xwoba_val,
               -- Savant's observed classification: 6 = Barrel. Missing
               -- classification is unknown, not a non-barrel observation.
               CASE WHEN launch_speed_angle BETWEEN 1 AND 6
                    THEN CASE WHEN launch_speed_angle = 6 THEN 1.0 ELSE 0.0 END
                    ELSE NULL END AS barrel_flag,
               CASE WHEN launch_speed IS NULL THEN NULL
                    WHEN launch_speed >= 95 THEN 1.0 ELSE 0.0 END AS hard_flag
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

    # SP staleness gate (see _SP_STALE_GAP_DAYS): flags appearance rows whose
    # prior-appearance gap crosses an availability stint, BEFORE any SP
    # window forms. Built from pitcher_game_stats, which is exactly the row
    # set every SP feature LAGs over.
    _register_sp_staleness_gate(con)

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
    # plus recent runs/9 and K/9 over the prior five appearances (ACROSS
    # seasons — no season-start gap: an April appearance rolls over the prior
    # season's tail). All point-in-time
    # safe: the row holds the previous game's stats via LAG, so the current
    # game never enters its own feature.
    con.execute("""        CREATE TABLE pitcher_shifted_season AS
        SELECT *,
            LAG(ip, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_ip,
            LAG(runs, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_runs,
            LAG(ks, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_ks,
            LAG(bbs, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_bbs,
            LAG(hits_allowed, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_hits,
            LAG(hrs_allowed, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_hrs,
            LAG(hbps, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_hbps,
            LAG(xwoba, 1) OVER (PARTITION BY pitcher, season ORDER BY game_date) AS _s_xwoba
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

    # Last-five-appearance window across ALL seasons (no season partition).
    # The all-history sum minus the recent-five sum is the same pitcher's older
    # career baseline, with no overlap between baseline and recent sample.
    # A career debut has NULL recent innings and remains NULL naturally.
    con.execute("""
        CREATE TABLE pitcher_5g_rolling AS
        SELECT game_date, game_pk, pitcher,
            SUM(_s_runs) OVER w5 AS _roll5_runs,
            SUM(_s_ks)  OVER w5 AS _roll5_ks,
            SUM(_s_ip)  OVER w5 AS _roll5_ip,
            COALESCE(SUM(_s_runs) OVER wall, 0.0)
                - COALESCE(SUM(_s_runs) OVER w5, 0.0) AS _older_runs,
            COALESCE(SUM(_s_ip) OVER wall, 0.0)
                - COALESCE(SUM(_s_ip) OVER w5, 0.0) AS _older_ip
        FROM pitcher_shifted
        WINDOW w5 AS (PARTITION BY pitcher ORDER BY game_date
                      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
               wall AS (PARTITION BY pitcher ORDER BY game_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

    # A date-level league fallback for pitchers without older personal innings.
    # Publish only the cumulative value through the PREVIOUS date so none of
    # today's games, including another game in a doubleheader, affect it.
    con.execute("""
        CREATE TABLE pitcher_era_league AS
        WITH daily AS (
            SELECT CAST(game_date AS DATE) AS game_date,
                   SUM(runs) AS runs, SUM(ip) AS ip
            FROM pitcher_game_stats
            GROUP BY CAST(game_date AS DATE)
        ), cumulative AS (
            SELECT game_date,
                   SUM(runs) OVER w AS runs_thru,
                   SUM(ip) OVER w AS ip_thru
            FROM daily
            WINDOW w AS (ORDER BY game_date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        SELECT game_date,
               LAG(runs_thru) OVER (ORDER BY game_date)
                 / NULLIF(LAG(ip_thru) OVER (ORDER BY game_date), 0) * 9.0
                 AS league_era_prior
        FROM cumulative
    """)

    con.execute(f"""
        CREATE TABLE pitcher_season_features AS
        SELECT psr.game_date, psr.game_pk, psr.pitcher,
            -- True season-to-date ERA / K/9 (through the prior in-season
            -- appearances only; NULL for a season's opening appearance).
            -- SP staleness gate keeps its stronger NULL semantics.
            CASE WHEN st.sp_stale THEN NULL
                 ELSE psr._s_runs_s / NULLIF(psr._s_ip_s, 0) * 9.0 END AS sp_era,
            CASE WHEN st.sp_stale THEN NULL
                 ELSE psr._s_ks_s / NULLIF(psr._s_ip_s, 0) * 9.0 END AS sp_k9,
            -- Recent runs/9 across the prior five appearances, blended with
            -- non-overlapping older pitcher history. _SP_ERA_5G_SHRINK_IP is
            -- pseudo prior innings; league prior is only a cold-start fallback.
            CASE WHEN st.sp_stale THEN NULL
                 WHEN p5._roll5_ip > 0 THEN
                    (9.0 * p5._roll5_runs
                     + {_SP_ERA_5G_SHRINK_IP} * COALESCE(
                         9.0 * p5._older_runs / NULLIF(p5._older_ip, 0),
                         lg.league_era_prior,
                         9.0 * p5._roll5_runs / NULLIF(p5._roll5_ip, 0))
                    ) / (p5._roll5_ip + {_SP_ERA_5G_SHRINK_IP})
            END AS sp_era_5g,
            CASE WHEN st.sp_stale THEN NULL
                 ELSE p5._roll5_ks / NULLIF(p5._roll5_ip, 0) * 9.0 END AS sp_k9_5g
        FROM pitcher_season_rolling psr
        LEFT JOIN pitcher_5g_rolling p5
               ON psr.game_pk = p5.game_pk AND psr.pitcher = p5.pitcher
        LEFT JOIN pitcher_era_league lg
               ON psr.game_date = lg.game_date
        LEFT JOIN pitcher_stale st
               ON psr.game_date = st.game_date AND psr.game_pk = st.game_pk
              AND psr.pitcher = st.pitcher
    """)

    # Season-to-date SP bb9/WHIP/xwOBA baselines — the momentum companion
    # terms for the 30g SP windows. Built from the SAME shifted per-game
    # stats as sp_era/sp_k9 above, season-partitioned cumulative so an
    # April start never averages in the prior October (point-in-time safe:
    # rows hold the PREVIOUS game's stats via LAG).
    con.execute("""
        CREATE TABLE pitcher_season_full AS
        SELECT p.game_date, p.game_pk, p.pitcher,
            SUM(_s_bbs) OVER w AS _s_bbs_s,
            SUM(_s_hits) OVER w AS _s_hits_s,
            SUM(_s_hrs) OVER w AS _s_hrs_s,
            SUM(_s_hbps) OVER w AS _s_hbps_s,
            SUM(_s_ks) OVER w AS _s_ks_s,
            SUM(_s_ip) OVER w AS _s_ip_s,
            SUM(_s_runs) OVER w AS _s_runs_s,
            AVG(_s_xwoba) OVER w AS _s_xwoba_s
        FROM pitcher_shifted_season p
        WINDOW w AS (PARTITION BY pitcher, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute("""
        CREATE TABLE pitcher_season_std AS
        SELECT game_date, game_pk, pitcher,
            _s_bbs_s / NULLIF(_s_ip_s, 0) * 9.0 AS sp_bb9_std,
            (_s_bbs_s + _s_hits_s) / NULLIF(_s_ip_s, 0) AS sp_whip_std,
            _s_xwoba_s AS sp_xwoba_std
        FROM pitcher_season_full
    """)

    con.execute("""
        CREATE TABLE pitcher_features AS
        SELECT pitcher_rolling.game_date, pitcher_rolling.game_pk,
               pitcher_rolling.pitcher,
            -- SP staleness gate: same NULLing rule as pitcher_season_features.
            CASE WHEN st.sp_stale THEN NULL
                 ELSE _roll_bbs / NULLIF(_roll_ip, 0) * 9.0 END AS sp_bb9_30g,
            CASE WHEN st.sp_stale THEN NULL
                 ELSE (_roll_bbs + _roll_hits) / NULLIF(_roll_ip, 0) END AS sp_whip_30g,
            CASE WHEN st.sp_stale THEN NULL
                 ELSE (13 * _roll_hrs + 3 * (_roll_bbs + _roll_hbps) - 2 * _roll_ks)
                      / NULLIF(_roll_ip, 0) END AS sp_fip_30g,
            CASE WHEN st.sp_stale THEN NULL
                 ELSE _roll_xwoba END AS sp_xwoba_30g
        FROM pitcher_rolling
        LEFT JOIN pitcher_stale st
               ON pitcher_rolling.game_date = st.game_date
              AND pitcher_rolling.game_pk = st.game_pk
              AND pitcher_rolling.pitcher = st.pitcher
    """)

    # Batter availability ledger, registered ONCE before the FIRST
    # team-side source table (2026-10-02) so the whole team_* family can
    # filter unavailable batters at the source (see _batter_excl). The
    # expected-lineup pool at 7e-bis consumes the same registration.
    # Missing cache -> _batters_ok False -> every fragment binds TRUE
    # (pre-availability semantics) under _register_il_stints's loud warning.
    _batters_ok = _register_il_stints(con)
    # Position map registers BEFORE the batter chain (2026-10-02): the
    # pl_* shrinkage prior is position-segmented — batter_league_pos and
    # the shrunk_xwoba rating itself join it at rating time. Absence
    # degrades to an EMPTY well-formed temp table so every downstream
    # join/pool SQL still binds: ratings fall back to the overall league
    # xwOBA prior, pools come up empty, pl_* ships NULL — loud, never a
    # missing-table error.
    _pos_ok = _register_player_positions(con)
    if not _pos_ok:
        con.execute("CREATE OR REPLACE TEMP TABLE player_positions "
                    "(batter BIGINT, season INTEGER, pos VARCHAR)")
    # Announced batting orders register with the other membership caches
    # (2026-10-03): lineups_raw is tier 1 of lineup_effective — tonight's
    # nine. Absence degrades to an EMPTY table under _register_lineups's
    # warning, so the membership clause below never hits a missing table.
    _lineups_ok = _register_lineups(con)
    if not _lineups_ok:
        con.execute(_EMPTY_LINEUPS_RAW_SQL)

    # 6. Team offense rolling features
    con.execute(f"""
        CREATE TABLE team_offense_raw AS
        WITH pa_events AS (
            SELECT CAST(p.game_date AS DATE) AS game_date, p.game_pk,
                   CASE WHEN p.inning_topbot = 'Top' THEN p.away_team ELSE p.home_team END AS batting_team,
                   p.events
            FROM pitches p WHERE p.events IN ({PA_END_EVENTS})
              AND {_batter_excl('p', 'p.game_date', _batters_ok)}
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
    """)

    # Season-to-date team offense baselines (momentum companion for the 30g
    # windows). The LAG shift is season-partitioned so the 2026 opener's row
    # is NULL and the expanding mean excludes all 2025 games.
    con.execute("""
        CREATE TABLE team_off_season AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_woba) OVER w AS team_woba_std,
            AVG(_s_iso) OVER w AS team_iso_std,
            AVG(_s_krate) OVER w AS team_k_rate_std,
            AVG(_s_bbrate) OVER w AS team_bb_rate_std
        FROM (SELECT game_date, game_pk, batting_team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(team_woba_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_woba,
                     LAG(team_iso_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_iso,
                     LAG(team_k_rate_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_krate,
                     LAG(team_bb_rate_game, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_bbrate
              FROM team_offense_raw)
        WINDOW w AS (PARTITION BY batting_team, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

    # 7. Bullpen rolling features

    # Reliever workload is attributed to the FIELDING team (Top half → home
    # team pitchers, Bottom half → away team), and BOTH starting pitchers are
    # excluded. The old first_pitchers CTE had one row per PITCH and only
    # excluded the home starter, so every join fanned out ~300x (a team's
    # 3-day "bullpen pitch count" read 69,901) and the away starter's warmup-
    # free pitches were billed to the home bullpen.
    # Bullpen availability ledger (same taxonomy as the lineup's OUT/IR
    # filter). Missing → loud degradation to all-innings semantics.
    _pitchers_ok = _register_il_stints_pitchers(con)
    _arm_out = (_BP_ARM_UNAVAILABLE_SQL if _pitchers_ok else "FALSE")
    # Per-reliever outing ledger (pitches per reliever-game; starters
    # excluded by the production starters table, >= 3 pitches to drop
    # position-player cameos). Built BEFORE bullpen_raw: it backs both the
    # readiness rollup and the over-budget exclusion below.
    con.execute("""
        CREATE TABLE bp_outing AS
        SELECT CAST(p.game_date AS DATE) AS game_date,
               p.game_pk,
               CASE WHEN p.inning_topbot = 'Top' THEN p.home_team
                    ELSE p.away_team END AS team,
               p.pitcher,
               COUNT(*) AS n_pitches
        FROM pitches p
        JOIN starters s ON p.game_pk = s.game_pk
        WHERE (s.home_starter_id IS NULL OR p.pitcher != s.home_starter_id)
          AND (s.away_starter_id IS NULL OR p.pitcher != s.away_starter_id)
          AND p.pitcher IS NOT NULL
        GROUP BY 1, 2, 3, 4
        HAVING COUNT(*) >= 3
    """)

    # Over-budget (spent) rows: appearances within _BP_SPENT_LOOKBACK_DAYS
    # of a prior >= _BP_SPENT_PITCHES outing. Degenerate when bp_outing is
    # empty.
    con.execute(f"""
        CREATE TABLE bp_spent AS
        -- AS-OF heavy outings: (team, pitcher, game_date) rows whose PRIOR
        -- outing (same team) was >= _BP_SPENT_PITCHES within the lookback.
        -- An outing row in this set was thrown by an arm who — per the
        -- calibration — was near-certainly unavailable that day (P(appear)
        -- 0.007 at 1d rest, 0.066 at 2d): keeping those rows in the WHIP
        -- windows would price tonight's pen on innings its overworked arms
        -- threw against their own fatigue (1,921 rows / 63k = 3.1% of the
        -- ledger). Latest-only matching was a proven no-op: an arm who
        -- pitches again supersedes his heavy outing as latest.
        SELECT o.team, o.pitcher, o.game_date AS game_date,
               h.last_heavy AS heavy_outing
        FROM bp_outing o
        JOIN (
            SELECT team, pitcher, game_date,
                   MAX(CASE WHEN n_pitches >= {_BP_SPENT_PITCHES}
                            THEN game_date END) OVER (
                       PARTITION BY team, pitcher ORDER BY game_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                       AS last_heavy
            FROM bp_outing
        ) h ON h.team = o.team
           AND h.pitcher = o.pitcher
           AND h.game_date = o.game_date
        WHERE h.last_heavy IS NOT NULL
          AND o.game_date > h.last_heavy
          AND o.game_date <= CAST(h.last_heavy AS DATE)
              + INTERVAL '{_BP_SPENT_LOOKBACK_DAYS} DAY'
    """)

    con.execute(f"""
        CREATE TABLE bullpen_raw AS
        WITH reliever_events AS (
            SELECT CAST(p.game_date AS DATE) AS game_date,
                   p.game_pk,
                   p.pitcher,
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
              AND NOT {_arm_out}
              AND NOT EXISTS (
                  SELECT 1 FROM bp_spent sp
                  WHERE sp.team = (CASE WHEN p.inning_topbot = 'Top'
                                  THEN p.home_team ELSE p.away_team END)
                    AND sp.pitcher = p.pitcher
                    AND sp.game_date = CAST(p.game_date AS DATE))
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

    # Opportunity-weighted shrinkage prior (2026-09-30, structural alignment
    # with the NBA player_ts / NHL player-rating convention; see the
    # _BP_SHRINK_FRACTION block). Two parts, both strictly PIT:
    #   k  = 20% of the mean reliever-season pitch count over the SAME
    #        availability+spent-filtered population, cameo floor 30 pitches
    #        — the portable fraction is the point, not any one season's mean.
    #   mu = league reliever rate per (season, event-date): cumulative
    #        counts over PRIOR dates only, so April blends toward April
    #        league reality, never the season's own final numbers. Rows
    #        with no prior-season data fall back to all-history-through-
    #        prior-date (opening day of the first season).
    # Degenerate (no reliever rows): k NULL -> every consumer's CASE guard
    # ships the raw rate; loud log, never a silent all-NULL.
    # Missing ledger: the availability predicate interpolates as FALSE —
    # the k derivation runs on the unfiltered population (loud degradation
    # consistent with the rest of the chain), never a missing-table error.
    _k_arm_out = (_BP_LEDGER_EXCLUDE_ROW_SQL.format(alias="o",
                                                    date="o.game_date")
                  if _pitchers_ok else "FALSE")
    con.execute(f"""
        CREATE TABLE bp_shrink_prior AS
        WITH arm_season AS (
            -- Same population the rates see: availability-filtered (ledger)
            -- and spent-arm-excluded outing rows only.
            SELECT o.pitcher, EXTRACT(YEAR FROM o.game_date) AS season,
                   SUM(o.n_pitches) AS pitches
            FROM bp_outing o
            WHERE {_k_arm_out}
              AND NOT EXISTS (
                      SELECT 1 FROM bp_spent sp
                      WHERE sp.team = o.team AND sp.pitcher = o.pitcher
                        AND sp.game_date = o.game_date)
            GROUP BY 1, 2
            HAVING SUM(n_pitches) >= {_BP_MIN_SEASON_PITCHES}),
        k AS (
            -- COALESCE: an all-cameo frame (every arm under the season
            -- floor — opening week) still yields the floor k, so the
            -- prior table is never empty and rates never NULL out.
            SELECT GREATEST(
                       {_BP_K_FLOOR},
                       {_BP_SHRINK_FRACTION} * COALESCE(
                           (SELECT AVG(pitches) FROM arm_season),
                           {_BP_K_FLOOR})) AS k_pitches,
                   GREATEST(
                       {_BP_K_FLOOR},
                       {_BP_SHRINK_FRACTION} * COALESCE(
                           (SELECT AVG(pitches) FROM arm_season),
                           {_BP_K_FLOOR}))
                       / {_BP_PITCHES_PER_IP} AS k_ip
            FROM (SELECT 1) anchor),
        daily AS (
            SELECT game_date,
                   EXTRACT(YEAR FROM game_date) AS season,
                   SUM(bullpen_bbs) AS bbs, SUM(bullpen_hits) AS hits,
                   SUM(bullpen_runs) AS runs, SUM(bullpen_ip) AS ip
            FROM bullpen_raw GROUP BY 1, 2),
        thru AS (
            SELECT season, game_date,
                SUM(bbs) OVER w AS bbs, SUM(hits) OVER w AS hits,
                SUM(runs) OVER w AS runs, SUM(ip) OVER w AS ip
            FROM daily
            WINDOW w AS (PARTITION BY season ORDER BY game_date
                         ROWS BETWEEN UNBOUNDED PRECEDING
                              AND 1 PRECEDING)),
        allthru AS (
            SELECT game_date,
                SUM(bbs) OVER w AS bbs, SUM(hits) OVER w AS hits,
                SUM(runs) OVER w AS runs, SUM(ip) OVER w AS ip
            FROM daily
            WINDOW w AS (ORDER BY game_date
                         ROWS BETWEEN UNBOUNDED PRECEDING
                              AND 1 PRECEDING))
        SELECT t.season, t.game_date, k.k_pitches, k.k_ip,
            CASE WHEN t.ip > 0
                 THEN (t.bbs + t.hits) / t.ip
                 ELSE (a.bbs + a.hits) / NULLIF(a.ip, 0) END AS lg_whip,
            CASE WHEN t.ip > 0
                 THEN t.runs / t.ip * 9.0
                 ELSE a.runs / NULLIF(a.ip, 0) * 9.0 END AS lg_era
        FROM thru t
        LEFT JOIN allthru a ON a.game_date = t.game_date
        CROSS JOIN k
    """)
    _krow = con.execute(
        "SELECT k_pitches, k_ip FROM bp_shrink_prior LIMIT 1").fetchone()
    if _krow and _krow[0] is not None:
        logger.info("Bullpen shrinkage: k = %.1f pitches (%.2f IP) = %.0f%% "
                    "of the mean reliever-season (%.1f pitch floor)",
                    _krow[0], _krow[1], _BP_SHRINK_FRACTION * 100,
                    _BP_K_FLOOR)
    else:
        logger.warning("Bullpen shrinkage: degenerate (no reliever rows) — "
                       "raw unshrunk rates ship")
    # Readiness rollup per team-day: the prior-2-day pitch budget plus the
    # ready-weighted share of those pitches (0.25/0.13/0.007 staircase by
    # last-outing length; ledger arms are 0). The team's MOST RECENT game
    # is included (an arm who threw yesterday is exactly the fatigue case);
    # the CURRENT day is not (it does not exist yet at prediction time).
    # Missing ledger: the as-of predicate interpolates as FALSE and the
    # ready staircase runs on raw workloads (loud degradation, consistent
    # with the rest of the chain) — never a missing-table error.
    _d2_ledger = (_BP_LEDGER_EXCLUDE_ASOF_SQL.format(alias="o", ref="g.gd")
                  if _pitchers_ok else "FALSE")
    con.execute(f"""
        CREATE TABLE bp_day2 AS
        WITH d2 AS (
            -- One row per (consuming game_pk, side): each outing feeds
            -- exactly one game-day (o.game_date = g.gd - 1 DAY, the slice
            -- the outer attach reads) instead of being counted once per
            -- EVERY following game in the prior-2-day window. The old
            -- per-(team, outing-date) grouping multiplied a back-to-back
            -- team's budget by its following-game count (x2 on every
            -- two-games-in-three-days stretch, x3 before a twin bill) and
            -- re-added the same pitches once per doubleheader leg.
            SELECT g.game_pk, g.gd AS ref_day, o.team,
                   SUM(o.n_pitches) AS pitches_2d,
                   SUM(o.n_pitches *
                       CASE
                         -- Ledger as of the REFERENCE game date: an arm
                         -- placed on the IL AFTER he pitched (the normal
                         -- reconciliation fact) is unavailable TONIGHT, so
                         -- his recent pitches carry no ready capacity.
                         WHEN {_d2_ledger}
                             THEN 0.0
                         WHEN o.n_pitches < 20 THEN 0.25
                         WHEN o.n_pitches < 35 THEN 0.13
                         ELSE 0.007 END) AS ready_pitches_2d,
                   COUNT(*) AS n_arms
            FROM bp_outing o
            JOIN (SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS gd,
                         home_team, away_team FROM pitches) g
              ON (o.team = g.home_team OR o.team = g.away_team)
             AND o.game_date = g.gd - INTERVAL 1 DAY
            GROUP BY 1, 2, 3
        )
        SELECT g2.gd AS ref_day, g2.game_pk, g2.home_team, g2.away_team,
               hh.pitches_2d AS home_pitches_2d,
               hh.ready_pitches_2d AS home_ready_2d,
               aa.pitches_2d AS away_pitches_2d,
               aa.ready_pitches_2d AS away_ready_2d
        FROM (SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS gd,
                     home_team, away_team FROM pitches) g2
        LEFT JOIN d2 hh ON hh.game_pk = g2.game_pk AND hh.team = g2.home_team
        LEFT JOIN d2 aa ON aa.game_pk = g2.game_pk AND aa.team = g2.away_team
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
               a.ip_3d AS bullpen_ip_3d_away,
               d2.home_pitches_2d AS bullpen_budget_2d_home,
               d2.away_pitches_2d AS bullpen_budget_2d_away,
               CASE WHEN COALESCE(d2.home_pitches_2d, 0) > 0
                    THEN d2.home_ready_2d / d2.home_pitches_2d
                         / 0.25 END
                    AS bp_ready_share_home,
               CASE WHEN COALESCE(d2.away_pitches_2d, 0) > 0
                    THEN d2.away_ready_2d / d2.away_pitches_2d
                         / 0.25 END
                    AS bp_ready_share_away
        FROM games g
        LEFT JOIN home_load h ON g.game_pk = h.game_pk
        LEFT JOIN away_load a ON g.game_pk = a.game_pk
        -- bp_day2 is ONE row per game_pk carrying both sides' values
        -- (home_* / away_*), so a single game_pk join. The old
        -- (ref_day, team) key matched BOTH legs of a same-opponent
        -- doubleheader per side and fanned game_level x4 per twin bill
        -- (164 game_pks x 4 duplicate rows shipped to training).
        LEFT JOIN bp_day2 d2 ON d2.game_pk = g.game_pk
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
        SELECT bullpen_shifted.game_date, bullpen_shifted.game_pk,
               bullpen_shifted.team,
            -- Opportunity-weighted shrinkage (NBA player_ts alignment):
            -- shrunk = w*rate + (1-w)*league_prior, w = window_IP/(window_IP+k).
            -- Thin windows (April 3g median ~11 IP) blend hardest toward the
            -- PIT league reliever prior; k comes from bp_shrink_prior
            -- (data-derived, see the constants block). Rates only —
            -- workloads never shrink.
            CASE WHEN p.k_ip > 0 THEN
                (SUM(_s_ip) OVER w10 / (SUM(_s_ip) OVER w10 + p.k_ip))
                * ((SUM(_s_bbs) OVER w10 + SUM(_s_hits) OVER w10)
                   / NULLIF(SUM(_s_ip) OVER w10, 0))
                + (p.k_ip / (SUM(_s_ip) OVER w10 + p.k_ip)) * p.lg_whip
            ELSE (SUM(_s_bbs) OVER w10 + SUM(_s_hits) OVER w10)
                 / NULLIF(SUM(_s_ip) OVER w10, 0) END AS bullpen_whip_10g,
            CASE WHEN p.k_ip > 0 THEN
                (SUM(_s_ip) OVER w10 / (SUM(_s_ip) OVER w10 + p.k_ip))
                * (SUM(_s_runs) OVER w10
                   / NULLIF(SUM(_s_ip) OVER w10, 0) * 9.0)
                + (p.k_ip / (SUM(_s_ip) OVER w10 + p.k_ip)) * p.lg_era
            ELSE SUM(_s_runs) OVER w10
                 / NULLIF(SUM(_s_ip) OVER w10, 0) * 9.0 END AS bullpen_era_10g,
            CASE WHEN p.k_ip > 0 THEN
                (SUM(_s_ip) OVER w3 / (SUM(_s_ip) OVER w3 + p.k_ip))
                * ((SUM(_s_bbs) OVER w3 + SUM(_s_hits) OVER w3)
                   / NULLIF(SUM(_s_ip) OVER w3, 0))
                + (p.k_ip / (SUM(_s_ip) OVER w3 + p.k_ip)) * p.lg_whip
            ELSE (SUM(_s_bbs) OVER w3 + SUM(_s_hits) OVER w3)
                 / NULLIF(SUM(_s_ip) OVER w3, 0) END AS bullpen_whip_3g
        FROM bullpen_shifted
        JOIN bp_shrink_prior p
               ON p.season = EXTRACT(YEAR FROM bullpen_shifted.game_date)
              AND p.game_date = CAST(bullpen_shifted.game_date AS DATE)
        WINDOW w10 AS (PARTITION BY bullpen_shifted.team
                       ORDER BY bullpen_shifted.game_date
                       ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
               w3 AS (PARTITION BY bullpen_shifted.team
                      ORDER BY bullpen_shifted.game_date
                      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    """)

    # NOTE: no separate availability layer — unavailable arms' innings are
    # already excluded at bullpen_raw (see _BP_ARM_UNAVAILABLE_SQL), so every
    # served bullpen feature inherits tonight's availability. No extra
    # columns are emitted (2026-09-30 directive: no new features).

    # Season-to-date bullpen baselines (momentum companion for the 10g/3g
    # windows) — season-partitioned cumulative sums; the LAG shift is
    # season-partitioned so a season opener never averages the prior season.
    con.execute("""
        CREATE TABLE bullpen_season AS
        SELECT b.game_date, b.game_pk, b.team,
            -- Opportunity-weighted shrinkage on the season baselines too:
            -- an April season-to-date WHIP (a handful of IP) is noise-
            -- dominated; the season-as-of league prior takes the slack.
            CASE WHEN p.k_ip > 0 THEN
                (SUM(_s_ip) OVER w / (SUM(_s_ip) OVER w + p.k_ip))
                * ((SUM(_s_bbs) OVER w + SUM(_s_hits) OVER w)
                   / NULLIF(SUM(_s_ip) OVER w, 0))
                + (p.k_ip / (SUM(_s_ip) OVER w + p.k_ip)) * p.lg_whip
            ELSE (SUM(_s_bbs) OVER w + SUM(_s_hits) OVER w)
                 / NULLIF(SUM(_s_ip) OVER w, 0) END AS bullpen_whip_std,
            CASE WHEN p.k_ip > 0 THEN
                (SUM(_s_ip) OVER w / (SUM(_s_ip) OVER w + p.k_ip))
                * (SUM(_s_runs) OVER w / NULLIF(SUM(_s_ip) OVER w, 0) * 9.0)
                + (p.k_ip / (SUM(_s_ip) OVER w + p.k_ip)) * p.lg_era
            ELSE SUM(_s_runs) OVER w
                 / NULLIF(SUM(_s_ip) OVER w, 0) * 9.0 END AS bullpen_era_std
        FROM (SELECT game_date, game_pk, team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(bullpen_bbs, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_bbs,
                     LAG(bullpen_hits, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_hits,
                     LAG(bullpen_ip, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_ip,
                     LAG(bullpen_runs, 1) OVER (
                         PARTITION BY team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_runs
              FROM bullpen_raw) AS b
        JOIN bp_shrink_prior p
               ON p.season = b.season
              AND p.game_date = CAST(b.game_date AS DATE)
        WINDOW w AS (PARTITION BY b.team, b.season ORDER BY b.game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

    _build_pitcher_stuff(con)

    # 7c. Team contact-form — barrel rate / hard-hit rate / avg exit velo over
    # the trailing 15 games (LAG-shifted, excludes current game).
    # Balls in play ONLY: launch_speed is also populated on foul balls
    # (~76 mph avg vs ~87 on BIP), which dragged the old mean to ~83.
    # Barrel uses Savant's observed launch_speed_angle classification (6),
    # not the narrow 98-mph/26-30-degree proxy. Missing classifications stay
    # NULL. Hard-hit uses measured exit velocity >=95 mph.
    con.execute(f"""
        CREATE TABLE team_contact_raw AS
        WITH bip AS (
            SELECT CAST(p.game_date AS DATE) AS game_date, p.game_pk,
                   CASE WHEN p.inning_topbot = 'Top' THEN p.away_team
                        ELSE p.home_team END AS batting_team,
                   CASE WHEN p.launch_speed_angle BETWEEN 1 AND 6
                        THEN CASE WHEN p.launch_speed_angle = 6 THEN 1.0 ELSE 0.0 END
                        ELSE NULL END AS barrel_flag,
                   CASE WHEN p.launch_speed >= 95 THEN 1.0 ELSE 0.0 END AS hard_flag,
                   p.launch_speed
            FROM pitches p
            WHERE p.description = 'hit_into_play' AND p.launch_speed IS NOT NULL
              AND {_batter_excl('p', 'p.game_date', _batters_ok)}
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

    # Season-to-date contact baselines (momentum companion for the 15g
    # windows) — season-partitioned expanding mean of the same shifted
    # per-game barrel/hard-hit/exit-velo stats.
    con.execute("""
        CREATE TABLE team_contact_season AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_barrel) OVER w AS team_barrel_std,
            AVG(_s_hardhit) OVER w AS team_hardhit_std,
            AVG(_s_exitvelo) OVER w AS team_exitvelo_std
        FROM (SELECT game_date, game_pk, batting_team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(barrel_rate, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_barrel,
                     LAG(hardhit_rate, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_hardhit,
                     LAG(exitvelo, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_exitvelo
              FROM team_contact_raw)
        WINDOW w AS (PARTITION BY batting_team, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

    # 7d. Opposing-lineup handedness share — fraction of a team's PAs taken by
    # left-handed batters over the trailing 30 games. Paired with each
    # starter's xwOBA-vs-L/R splits, tree models can learn platoon-fit
    # interactions without raw player-name encoding.
    con.execute(f"""
        CREATE TABLE team_hand_raw AS
        WITH pa AS (
            SELECT CAST(p.game_date AS DATE) AS game_date, p.game_pk,
                   CASE WHEN p.inning_topbot = 'Top' THEN p.away_team
                        ELSE p.home_team END AS batting_team,
                   p.stand
            FROM pitches p WHERE p.events IN ({PA_END_EVENTS})
              AND {_batter_excl('p', 'p.game_date', _batters_ok)}
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
    """)    # 7e. Lineup composition — every hitter gets his own statistical
    # assumption: a per-player trailing-30g RE24 shrunk toward the
    # point-in-time league mean by PA count (empirical Bayes, 120-PA prior),
    # then aggregated into expected-lineup features. The wOBA machinery
    # (_woba_num/_ab/lg_woba/shrunk_woba) is retained alongside; the served
    # lineup_* features consume the RE24 rating (2026-10-02 swap).
    con.execute(f"""
        CREATE TABLE batter_game_stats AS
        WITH allp AS (
            SELECT CAST(game_date AS DATE) AS game_date, game_pk,
                   CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS batting_team,
                   batter, events, delta_run_exp,
                   estimated_woba_using_speedangle
            FROM pitches
        )
        SELECT game_date, game_pk, batting_team, batter,
               SUM(CASE WHEN events IN ({PA_END_EVENTS}) THEN 1 ELSE 0 END) AS pa,
               SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
               SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
               SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS s,
               SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS d,
               SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS t,
               SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play')
                        THEN 1 ELSE 0 END) AS k,
               -- RE24 (2026-10-02): Statcast's per-pitch run-expectancy
               -- change summed over EVERY pitch the batter saw in the game —
               -- the telescoping sum is exactly his runs-above-expectation
               -- vs the base-out states (delta_run_exp already ships in the
               -- pitches frame; no 24-state matrix needed). The PA-ending
               -- counters above keep their exact old semantics (conditional
               -- sums over event rows). COALESCE: a game whose pitches all
               -- lack the column counts 0 (league-average value), never
               -- NULL — NULL here would drop the rating row and cost pool
               -- coverage.
               COALESCE(SUM(delta_run_exp), 0.0) AS re24,
               -- xwOBA (2026-10-02 position-pool A/B): Statcast's
               -- estimated_woba_using_speedangle over PA-ending events
               -- with a measured estimate — the exp2 non-null-only
               -- convention, so sac bunts / intent walks / interference
               -- (no estimate) drop out of BOTH numerator and denominator
               -- instead of counting as fake 0s.
               SUM(CASE WHEN events IN ({PA_END_EVENTS})
                         AND estimated_woba_using_speedangle IS NOT NULL
                        THEN estimated_woba_using_speedangle
                        ELSE 0 END) AS xwoba_num,
               SUM(CASE WHEN events IN ({PA_END_EVENTS})
                         AND estimated_woba_using_speedangle IS NOT NULL
                        THEN 1 ELSE 0 END) AS xwoba_den
        FROM allp
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
            LAG(hr, 1) OVER w AS _hr,
            LAG(re24, 1) OVER w AS _re24,
            LAG(xwoba_num, 1) OVER w AS _xwoba_num,
            LAG(xwoba_den, 1) OVER w AS _xwoba_den
        FROM batter_game_stats
        WINDOW w AS (PARTITION BY batter ORDER BY game_date)
    """)
    con.execute("""
        CREATE TABLE batter_rolling AS
        SELECT game_date, game_pk, batting_team, batter,
            SUM(0.690*_bb + 0.722*_hbp + 0.878*_s + 1.242*_d
                + 1.568*_t + 2.007*_hr) OVER w30 AS _woba_num,
            SUM(_pa - _bb - _hbp) OVER w30 AS _ab,
            SUM(_pa) OVER w30 AS _pa30,
            SUM(_re24) OVER w30 AS _re24_num,
            SUM(_xwoba_num) OVER w30 AS _xwoba_num30,
            SUM(_xwoba_den) OVER w30 AS _xwoba_den30
        FROM batter_shifted
        WINDOW w30 AS (PARTITION BY batter ORDER BY game_date
                       ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
    """)
    # League mean wOBA, cumulative through each date — built ONLY from
    # already-shifted (prior-game) stats, so it stays point-in-time safe.
    # lg_re24 mirrors it exactly: pooled trailing-window RE24 numerator over
    # pooled trailing-window PA (the same per-date windowed quantities
    # summed across batters — identical construction to lg_woba).
    con.execute("""
        CREATE TABLE batter_league AS
        SELECT game_date,
            SUM(_woba_num) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
              / NULLIF(SUM(_ab) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
              AS lg_woba,
            SUM(_re24_num) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
              / NULLIF(SUM(_pa30) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
              AS lg_re24,
            SUM(_xwoba_num30) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
              / NULLIF(SUM(_xwoba_den30) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
              AS lg_xwoba
        FROM (
            SELECT game_date, SUM(_woba_num) AS _woba_num, SUM(_ab) AS _ab,
                   SUM(_re24_num) AS _re24_num, SUM(_pa30) AS _pa30,
                   SUM(_xwoba_num30) AS _xwoba_num30,
                   SUM(_xwoba_den30) AS _xwoba_den30
            FROM batter_rolling GROUP BY game_date
        )
    """)
    # Position-segmented prior for the pl_* shrinkage target — must exist
    # BEFORE batter_ratings joins it (an empty player_positions leaves it
    # empty; ratings then fall back to lg_xwoba via COALESCE).
    con.execute(_BATTER_LEAGUE_POS_SQL)
    _woba_expr, _re24_expr = batter_rating_sql(MLB_SHRINK_ARM)
    _xwoba_expr = batter_xwoba_rating_sql(MLB_SHRINK_ARM)
    logger.info("Batter shrinkage arm: %s (k=%d PA/AB)",
                MLB_SHRINK_ARM, BATTER_SHRINK_K)
    con.execute(f"""
        CREATE TABLE batter_ratings AS
        SELECT r.game_date, r.game_pk, r.batting_team, r.batter,
               {_woba_expr} AS shrunk_woba,
               {_re24_expr} AS shrunk_re24,
               {_xwoba_expr} AS shrunk_xwoba,
               r._pa30
        FROM batter_rolling r
        LEFT JOIN batter_league l USING (game_date)
        LEFT JOIN player_positions pp
          ON pp.batter = r.batter
         AND pp.season = EXTRACT(YEAR FROM r.game_date)
        LEFT JOIN batter_league_pos lp
          ON lp.game_date = r.game_date AND lp.pos = pp.pos
    """)
    # 7e-bis. Game-eligible candidate pool: widen from "batted in this game"
    # to "team member with a rating row in the last {lookback} days" (one row
    # per batter: the most recent at or before the game date, LAG-shifted like
    # every window here), then mark who is OUT/IR as of the game date. The
    # rating itself is untouched by the flag — only game eligibility is.
    # When the IL cache is absent the pool degrades to the participant pool
    # (that game's own rating rows) with the flag hardwired to 0 — the exact
    # pre-IL behavior — under a loud warning, never a silent identity change.
    # Membership resolve — tonight's nine per team-game (tier 1
    # announced order, tier 2 hand-conditioned 10-day projection). Runs
    # AFTER batter_ratings (its universe) and starters/il_stints (its
    # conditioning inputs) exist, and BEFORE either pool so both bind the
    # same membership set. Degrades to an empty table — full-roster pools —
    # on any failure (loud, never a crash).
    _build_lineup_effective(con, _batters_ok)
    if _batters_ok:
        con.execute(_LINEUP_POOL_SQL.format(
            lookback=LINEUP_POOL_LOOKBACK_DAYS, restrict="",
            on_il=_IL_EXISTS_PREDICATE))
        _ov = con.execute(
            "SELECT count(DISTINCT i.batter) FROM il_stints i WHERE EXISTS "
            "(SELECT 1 FROM batter_ratings r WHERE r.batter = i.batter)").fetchone()[0]
        if not _ov:
            logger.warning(
                "il_stints loaded but 0 batters intersect batter_ratings — the "
                "OUT/IR eligibility filter cannot bind (id-space break). "
                "Rebuild data_delivery/il_stints.parquet before trusting the "
                "expected-lineup features.")
    else:
        con.execute(_LINEUP_POOL_SQL.format(
            lookback=LINEUP_POOL_LOOKBACK_DAYS,
            restrict="AND r.game_pk = g.game_pk", on_il="0"))
    con.execute(_LINEUP_AGG_SQL)
    con.execute(_LINEUP_IL_FLAG_SQL)

    # 7e-ter. Position-pool xwOBA family (pl_<pos>_xwoba_*). Built from
    # the SAME batter_ratings rows as the lineup family (LAG-shifted,
    # shrunk toward the position-segmented prior, IL-flagged) plus the
    # StatsAPI position map registered before the batter chain. The chain
    # is UNCONDITIONAL: with an absent/empty map every table binds but
    # comes up empty, so pos_agg has well-formed zero rows and game_level's
    # LEFT JOINs ship NULL pl_* (loud) — never a crash.
    con.execute(_POS_POOL_SQL.format(
        lookback=LINEUP_POOL_LOOKBACK_DAYS,
        pos_case=PL_POS_SQL,
        restrict=("" if _batters_ok else "AND r.game_pk = g.game_pk"),
        on_il=(_IL_EXISTS_PREDICATE if _batters_ok else "0")))
    con.execute(_POS_LEAGUE_SQL)
    con.execute(_POS_AGG_SQL)
    # 2026-10-03 production incident: a missing position map silently
    # emptied pos_agg and the run still reported "ok" with 24 dead
    # universe columns. Name the production symptom, not the cause.
    _warn_pl_coverage(con)
    # Serving artifact: tonight's resolved pools + provenance, written while
    # batter_ratings / pos_agg / lineup_effective still exist. The slate
    # build reads it to REPLACE the stale unmarked carry — the serving half
    # of the 3-tier structural alignment (never raises: absence keeps the
    # marked-carry path).
    _export_slate_pl(con)

    # Season-to-date lineup baselines (momentum companion for today's
    # projected-lineup RE24) — expanding mean of the team's PRIOR games'
    # expected-lineup RE24, season-partitioned (LAG-shifted: the current
    # game's lineup never enters its own baseline).
    con.execute("""
        CREATE TABLE lineup_season AS
        SELECT game_date, game_pk, batting_team,
            AVG(_s_lwm) OVER w AS lineup_re24_mean_std,
            AVG(_s_lwt) OVER w AS lineup_re24_top3_std
        FROM (SELECT game_date, game_pk, batting_team,
                     EXTRACT(YEAR FROM game_date) AS season,
                     LAG(lineup_re24_mean, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_lwm,
                     LAG(lineup_re24_top3, 1) OVER (
                         PARTITION BY batting_team, EXTRACT(YEAR FROM game_date)
                         ORDER BY game_date) AS _s_lwt
              FROM lineup_agg)
        WINDOW w AS (PARTITION BY batting_team, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

    # 7f. Lineup OPS split by opposing pitching hand — every hitter's
    # trailing-30g OPS vs left-handed and right-handed pitchers separately
    # (LAG-shifted, excludes the current game), aggregated over the expected
    # top-9 by playing time per hand.  Paired with tonight's opposing starter
    # throwing hand at assembly time this yields lineup_ops_vs_starter_hand.
    # 2026-10-02: unavailable batters' PAs are excluded at the SOURCE (the
    # team-family availability filter — _batter_excl), so the hand splits and
    # the OPS aggregates price tonight's real roster like every other
    # team-side family; absent ledger binds TRUE (pre-availability behavior).
    con.execute(f"""
        CREATE TABLE batter_hand_game AS
        WITH pa AS (
            SELECT CAST(p.game_date AS DATE) AS game_date, p.game_pk,
                   CASE WHEN p.inning_topbot = 'Top' THEN p.away_team
                        ELSE p.home_team END AS batting_team,
                   p.batter, p.p_throws, p.events
            FROM pitches p WHERE p.events IN ({PA_END_EVENTS})
              AND {_batter_excl('p', 'p.game_date', _batters_ok)}
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

    # 7i. Experiment #2 SOURCE layer — point-in-time pitch-category inputs
    # (data-layer only; consumed by future research candidates, never by
    # MONEYLINE_FEATURE_COLS). All windows are LAG-shifted so the current game never
    # enters its own features; expanding season windows are season-partitioned
    # like sp_era/sp_k9; league priors are per-date cumulative from already-
    # shifted stats (same construction as batter_league.lg_woba).
    #
    # Taxonomy (production pitch_category, defined here and in pbp_level):
    #   fastball: FF FT SI FC FS FO · breaking: SL CU KC CS SV WR ·
    #   offspeed: CH EP SC KN UN PO · everything else → NULL (excluded).
    # xwOBA cells average estimated_woba_using_speedangle over PA-ending
    # pitches with a non-null value (Statcast-native; not all PAs carry a
    # value — that is the production aggregation convention everywhere).
    # League priors/usage/K% are pooled K-rate or wOBA-numerator ratios
    # (SUM/SUM), NOT averages of rates — small-sample stable.
    con.execute(f"""
        CREATE TABLE exp2_pa AS
        WITH lastp AS (
            SELECT CAST(game_date AS DATE) AS game_date,
                   game_pk, pitcher, batter, stand, p_throws,
                   -- Each PA's OWN batting side (2026-10-02): the team-side
                   -- builders must attribute a PA to the team that took it;
                   -- deriving it downstream from a game-level join fanned
                   -- every PA out to both sides (see exp2_team_cat_game).
                   CASE WHEN inning_topbot = 'Top' THEN away_team
                        ELSE home_team END AS batting_team,
                   CASE WHEN pitch_type IN ('FF','FT','SI','FC','FS','FO') THEN 'fastball'
                        WHEN pitch_type IN ('SL','CU','KC','CS','SV','WR') THEN 'breaking'
                        WHEN pitch_type IN ('CH','EP','SC','KN','UN','PO') THEN 'offspeed'
                        ELSE NULL END AS pitch_cat,
                   CASE WHEN events IN ('strikeout','strikeout_double_play')
                        THEN 1.0 ELSE 0.0 END AS k_flag,
                   estimated_woba_using_speedangle AS xwoba_val
            FROM pitches
            WHERE events IN ({PA_END_EVENTS})
        )
        SELECT game_date, game_pk, pitcher, batter, stand, batting_team,
               pitch_cat,
               k_flag,
               CASE WHEN pitch_cat IS NOT NULL THEN 1.0 ELSE 0.0 END AS n_flag,
               CASE WHEN xwoba_val IS NOT NULL THEN 1.0 ELSE 0.0 END AS xwoba_n,
               COALESCE(xwoba_val, 0.0) AS xwoba_summand
        FROM lastp
    """)

    # League K% (all-category): per-date pooled K / PA. The published value
    # for date D is the cumulative through the PREVIOUS date with data
    # (outer LAG by date), so same-day games never enter — strictly prior.
    con.execute(f"""
        CREATE TABLE exp2_league_k AS
        WITH daily AS (
            SELECT game_date,
                SUM(_k) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                  / NULLIF(SUM(_pa) OVER (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0)
                  AS k_pct_thru
            FROM (
                SELECT game_date, SUM(k_flag) AS _k, COUNT(*) AS _pa
                FROM exp2_pa GROUP BY game_date
            )
        )
        SELECT game_date,
               LAG(k_pct_thru) OVER (ORDER BY game_date) AS league_k_pct
        FROM daily
    """)

    # League K% and wOBA per pitch category (same per-date cumulative,
    # published as-of the previous date; first observed date → NULL).
    con.execute(f"""
        CREATE TABLE exp2_league_cat AS
        WITH daily AS (
            SELECT game_date, pitch_cat,
                SUM(_k) OVER w / NULLIF(SUM(_pa) OVER w, 0) AS k_thru,
                SUM(_xwoba_num) OVER w / NULLIF(SUM(_xwoba_n) OVER w, 0) AS xwoba_thru
            FROM (
                SELECT game_date, pitch_cat, SUM(k_flag) AS _k, COUNT(*) AS _pa,
                       SUM(xwoba_summand) AS _xwoba_num, SUM(xwoba_n) AS _xwoba_n
                FROM exp2_pa WHERE pitch_cat IS NOT NULL
                GROUP BY game_date, pitch_cat
            )
            WINDOW w AS (PARTITION BY pitch_cat ORDER BY game_date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        SELECT game_date, pitch_cat,
               LAG(k_thru) OVER wp AS league_k_pct_cat,
               LAG(xwoba_thru) OVER wp AS league_xwoba_cat
        FROM daily
        WINDOW wp AS (PARTITION BY pitch_cat ORDER BY game_date)
    """)

    # League fastball K% by batter handedness (per-date cumulative published
    # as-of the previous date — strictly prior, same as exp2_league_k).
    # 2026-10-01: hoisted ABOVE the exp2_sp_cat/_fbhand builders — the
    # zero-prior fallback joins this table there, and creation order must
    # match dependency order (the nightly run died on exactly that).
    con.execute(f"""
        CREATE TABLE exp2_league_fbhand AS
        WITH daily AS (
            SELECT game_date, stand,
                SUM(_k) OVER w / NULLIF(SUM(_pa) OVER w, 0) AS k_thru
            FROM (
                SELECT game_date, stand, SUM(k_flag) AS _k, SUM(n_flag) AS _pa
                FROM exp2_pa WHERE pitch_cat = 'fastball'
                GROUP BY game_date, stand
            )
            WINDOW w AS (PARTITION BY stand ORDER BY game_date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        SELECT game_date, stand,
               LAG(k_thru) OVER wp AS league_k_pct_fb_vs
        FROM daily
        WINDOW wp AS (PARTITION BY stand ORDER BY game_date)
    """)

    # Starter per-category raw per-game counts (PAs ending on each category).
    con.execute(f"""
        CREATE TABLE exp2_sp_cat_game AS
        SELECT game_date, game_pk, pitcher, pitch_cat,
               SUM(k_flag) AS k_n, SUM(n_flag) AS pa_n,
               SUM(xwoba_summand) AS xwoba_num, SUM(xwoba_n) AS xwoba_n
        FROM exp2_pa WHERE pitch_cat IS NOT NULL
        GROUP BY game_date, game_pk, pitcher, pitch_cat
    """)
    # Starter fastball-by-opposing-batter-hand raw per-game counts.
    con.execute(f"""
        CREATE TABLE exp2_sp_fbhand_game AS
        SELECT game_date, game_pk, pitcher, stand,
               SUM(k_flag) AS k_n, SUM(n_flag) AS pa_n
        FROM exp2_pa WHERE pitch_cat = 'fastball'
        GROUP BY game_date, game_pk, pitcher, stand
    """)
    # Starter per-category — DATE-LEVEL season-to-date priors. Daily raw
    # counts are aggregated per (pitcher, season, pitch_cat, DATE) — doubleheader
    # same-date games merge, so a target game never sees its date's PAs — then
    # a cumulative is carried and ASOF-joined so every pitching date gets the
    # prior-total even when that date itself had no PAs of the category.
    # Season-partitioned (prior October never leaks — sp_era/sp_k9 semantics).
    # NULL until the pitcher has a prior in-season start.
    con.execute(f"""
        CREATE TABLE exp2_sp_cat_daily AS
        SELECT game_date, pitcher,
               EXTRACT(YEAR FROM game_date) AS season, pitch_cat,
               SUM(k_n) AS k_n, SUM(pa_n) AS pa_n,
               SUM(xwoba_num) AS xwoba_num, SUM(xwoba_n) AS xwoba_n
        FROM exp2_sp_cat_game
        GROUP BY game_date, pitcher, season, pitch_cat
    """)
    con.execute(f"""
        CREATE TABLE exp2_sp_cat_cum AS
        SELECT *,
            SUM(k_n) OVER w AS k_thru,
            SUM(pa_n) OVER w AS pa_thru,
            SUM(xwoba_num) OVER w AS xwo_thru,
            SUM(xwoba_n) OVER w AS xwon_thru
        FROM exp2_sp_cat_daily
        WINDOW w AS (PARTITION BY pitcher, season, pitch_cat ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute(f"""
        CREATE TABLE exp2_sp_cat_tot_daily AS
        SELECT game_date, pitcher, season, SUM(pa_n) AS pa_n
        FROM exp2_sp_cat_daily
        GROUP BY game_date, pitcher, season
    """)
    con.execute(f"""
        CREATE TABLE exp2_sp_cat_tot_cum AS
        SELECT *, SUM(pa_n) OVER w AS pa_thru
        FROM exp2_sp_cat_tot_daily
        WINDOW w AS (PARTITION BY pitcher, season ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute(_EXP2_SP_CAT_PRELIM_SQL)
    con.execute(_EXP2_SP_CAT_SQL)

    # Starter fastball K% by opposing-batter handedness — DATE-LEVEL
    # season-to-date priors (same daily + ASOF construction as exp2_sp_cat).
    con.execute(f"""
        CREATE TABLE exp2_sp_fbhand_daily AS
        SELECT game_date, pitcher,
               EXTRACT(YEAR FROM game_date) AS season, stand,
               SUM(k_n) AS k_n, SUM(pa_n) AS pa_n
        FROM exp2_sp_fbhand_game
        GROUP BY game_date, pitcher, season, stand
    """)
    con.execute(f"""
        CREATE TABLE exp2_sp_fbhand_cum AS
        SELECT *,
            SUM(k_n) OVER w AS k_thru,
            SUM(pa_n) OVER w AS pa_thru
        FROM exp2_sp_fbhand_daily
        WINDOW w AS (PARTITION BY pitcher, season, stand ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute(_EXP2_SP_FBHAND_SQL)

    # Offense per-category raw per-game counts (team-aggregated).
    # 2026-10-02 FIX (double-attribution): exp2_pa now carries each PA's OWN
    # batting_team, so no join is needed. The old join to the game's
    # DISTINCT (game_pk, inning_topbot) rows fanned EVERY PA out to BOTH
    # sides — each team's per-game counts equaled the game TOTAL (~half
    # opponent PAs; 158/158 team-game cells wrong on the 2026-06-16..21
    # window). Same availability filter as the rest of the team family.
    con.execute(f"""
        CREATE TABLE exp2_team_cat_game AS
        SELECT game_date, game_pk, batting_team, pitch_cat,
               SUM(k_flag) AS k_n, SUM(n_flag) AS pa_n,
               SUM(xwoba_summand) AS xwoba_num, SUM(xwoba_n) AS xwoba_n
        FROM exp2_pa a
        WHERE a.pitch_cat IS NOT NULL
          AND {_batter_excl('a', 'a.game_date', _batters_ok)}
        GROUP BY game_date, game_pk, batting_team, pitch_cat
    """)
    # Offense per-category K%/xwOBA — DATE-LEVEL season-to-date priors (same
    # daily + ASOF construction), so a target game never sees its own date's
    # PAs (doubleheader-safe) and every batting date carries the prior total
    # even on a category-less day.
    con.execute(f"""
        CREATE TABLE exp2_team_cat_daily AS
        SELECT game_date, batting_team,
               EXTRACT(YEAR FROM game_date) AS season, pitch_cat,
               SUM(k_n) AS k_n, SUM(pa_n) AS pa_n,
               SUM(xwoba_num) AS xwoba_num, SUM(xwoba_n) AS xwoba_n
        FROM exp2_team_cat_game
        GROUP BY game_date, batting_team, season, pitch_cat
    """)
    con.execute(f"""
        CREATE TABLE exp2_team_cat_cum AS
        SELECT *,
            SUM(k_n) OVER w AS k_thru,
            SUM(pa_n) OVER w AS pa_thru,
            SUM(xwoba_num) OVER w AS xwo_thru,
            SUM(xwoba_n) OVER w AS xwon_thru
        FROM exp2_team_cat_daily
        WINDOW w AS (PARTITION BY batting_team, season, pitch_cat ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute(f"""
        CREATE TABLE exp2_team_cat AS
        SELECT g.game_date, g.batting_team, g.pitch_cat,
            c.k_thru / NULLIF(c.pa_thru, 0) AS team_k_pct_cat,
            c.xwo_thru / NULLIF(c.xwon_thru, 0) AS team_xwoba_cat
        FROM (
            SELECT d.game_date, d.batting_team, d.season, cats.pitch_cat
            FROM (SELECT DISTINCT game_date, batting_team, season
                  FROM exp2_team_cat_daily) d
            CROSS JOIN (VALUES ('fastball'), ('breaking'), ('offspeed'))
                   AS cats(pitch_cat)
        ) g
        ASOF LEFT JOIN exp2_team_cat_cum c
          ON g.batting_team = c.batting_team AND g.season = c.season
         AND g.pitch_cat = c.pitch_cat AND g.game_date > c.game_date
    """)

    # Offense K% vs fastballs by the BATTER's handedness (season-to-date).
    # 2026-10-02: own-batting-team attribution + availability filter, same
    # as exp2_team_cat_game above (the old game-level join double-counted).
    con.execute(f"""
        CREATE TABLE exp2_team_fbhand_game AS
        SELECT game_date, game_pk, batting_team, stand,
               SUM(k_flag) AS k_n, SUM(n_flag) AS pa_n
        FROM exp2_pa a
        WHERE a.pitch_cat = 'fastball'
          AND {_batter_excl('a', 'a.game_date', _batters_ok)}
        GROUP BY game_date, game_pk, batting_team, stand
    """)
    con.execute(f"""
        CREATE TABLE exp2_team_fbhand_daily AS
        SELECT game_date, batting_team,
               EXTRACT(YEAR FROM game_date) AS season, stand,
               SUM(k_n) AS k_n, SUM(pa_n) AS pa_n
        FROM exp2_team_fbhand_game
        GROUP BY game_date, batting_team, season, stand
    """)
    con.execute(f"""
        CREATE TABLE exp2_team_fbhand_cum AS
        SELECT *,
            SUM(k_n) OVER w AS k_thru,
            SUM(pa_n) OVER w AS pa_thru
        FROM exp2_team_fbhand_daily
        WINDOW w AS (PARTITION BY batting_team, season, stand ORDER BY game_date
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)
    con.execute(f"""
        CREATE TABLE exp2_team_fbhand AS
        SELECT g.game_date, g.batting_team, g.stand,
            c.k_thru / NULLIF(c.pa_thru, 0) AS team_k_pct_fb_vs,
            c.pa_thru AS team_pa_fb_vs
        FROM (
            SELECT d.game_date, d.batting_team, d.season, h.st AS stand
            FROM (SELECT DISTINCT game_date, batting_team, season
                  FROM exp2_team_fbhand_daily) d
            CROSS JOIN (VALUES ('L'), ('R')) AS h(st)
        ) g
        ASOF LEFT JOIN exp2_team_fbhand_cum c
          ON g.batting_team = c.batting_team AND g.season = c.season
         AND g.stand = c.stand AND g.game_date > c.game_date
    """)

    # 7g/7h — travel fatigue + closer availability (helpers above).
    _build_travel_features(con)
    _build_closer_features(con)

    # 8. Assemble game_level via LEFT JOINs
    con.execute("""
        CREATE TABLE game_level AS
        SELECT
            w.game_pk, w.game_date, w.game_type,
            w.home_team, w.away_team,
            w.home_score, w.away_score, w.home_win, w.total_runs,
            s.home_starter_id, s.away_starter_id, v.venue,
            rh.rest_days AS rest_days_home, ra.rest_days AS rest_days_away,
            sh.sp_era AS sp_era_home, sh.sp_k9 AS sp_k9_home,
            ph.sp_bb9_30g AS sp_bb9_home, ph.sp_whip_30g AS sp_whip_home,
            ph.sp_fip_30g AS sp_fip_home, ph.sp_xwoba_30g AS sp_xwoba_home,
            sa.sp_era AS sp_era_away, sa.sp_k9 AS sp_k9_away,
            pa.sp_bb9_30g AS sp_bb9_away, pa.sp_whip_30g AS sp_whip_away,
            pa.sp_fip_30g AS sp_fip_away, pa.sp_xwoba_30g AS sp_xwoba_away,
            -- Recent-form SP twins under their existing model-contract names.
            sh.sp_era_5g AS sp_era_5g_home, sh.sp_k9_5g AS sp_k9_5g_home,
            sa.sp_era_5g AS sp_era_5g_away, sa.sp_k9_5g AS sp_k9_5g_away,
            th.team_woba_30g AS team_woba_30g_home, th.team_iso_30g AS team_iso_30g_home,
            th.team_k_rate_30g AS team_k_rate_30g_home, th.team_bb_rate_30g AS team_bb_rate_30g_home,
            ta.team_woba_30g AS team_woba_30g_away, ta.team_iso_30g AS team_iso_30g_away,
            ta.team_k_rate_30g AS team_k_rate_30g_away, ta.team_bb_rate_30g AS team_bb_rate_30g_away,
            bh.bullpen_whip_10g AS bullpen_whip_10g_home,
            bh.bullpen_era_10g AS bullpen_era_10g_home,

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
            bf.bullpen_budget_2d_home, bf.bullpen_budget_2d_away,
            bf.bp_ready_share_home, bf.bp_ready_share_away,
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
            lh.lineup_re24_mean AS lineup_re24_mean_home,
            lh.lineup_re24_top3 AS lineup_re24_top3_home,
            lh.lineup_re24_std AS lineup_re24_std_home,
            la.lineup_re24_mean AS lineup_re24_mean_away,
            la.lineup_re24_top3 AS lineup_re24_top3_away,
            la.lineup_re24_std AS lineup_re24_std_away,
            -- Position-pool xwOBA (pl_* family, 2026-10-02): one level
            -- pair per position pool, same home/away aliasing as the
            -- lineup family; the _diff twins form in add_diff_features.
            posh.pl_c_xwoba AS pl_c_xwoba_home,
            posa.pl_c_xwoba AS pl_c_xwoba_away,
            posh.pl_fb_xwoba AS pl_fb_xwoba_home,
            posa.pl_fb_xwoba AS pl_fb_xwoba_away,
            posh.pl_sb_xwoba AS pl_sb_xwoba_home,
            posa.pl_sb_xwoba AS pl_sb_xwoba_away,
            posh.pl_ss_xwoba AS pl_ss_xwoba_home,
            posa.pl_ss_xwoba AS pl_ss_xwoba_away,
            posh.pl_tb_xwoba AS pl_tb_xwoba_home,
            posa.pl_tb_xwoba AS pl_tb_xwoba_away,
            posh.pl_rf_xwoba AS pl_rf_xwoba_home,
            posa.pl_rf_xwoba AS pl_rf_xwoba_away,
            posh.pl_cf_xwoba AS pl_cf_xwoba_home,
            posa.pl_cf_xwoba AS pl_cf_xwoba_away,
            posh.pl_lf_xwoba AS pl_lf_xwoba_home,
            posa.pl_lf_xwoba AS pl_lf_xwoba_away,
            posh.pl_dh_xwoba AS pl_dh_xwoba_home,
            posa.pl_dh_xwoba AS pl_dh_xwoba_away,
            COALESCE(fh.lineup_il_flag, 0) AS lineup_il_flag_home,
            COALESCE(fa.lineup_il_flag, 0) AS lineup_il_flag_away,
            -- Momentum form deltas: recent window − season-to-date baseline,
            -- per side. Continuous (no binary flags); the model learns its
            -- own thresholds. Computed here from the SAME shifted per-game
            -- stats as the levels above (no parallel computation path).
            sh.sp_era_5g - sh.sp_era AS sp_era_delta_home,
            sh.sp_k9_5g - sh.sp_k9 AS sp_k9_delta_home,
            ph.sp_bb9_30g - phs.sp_bb9_std AS sp_bb9_delta_home,
            ph.sp_whip_30g - phs.sp_whip_std AS sp_whip_delta_home,
            ph.sp_xwoba_30g - phs.sp_xwoba_std AS sp_xwoba_delta_home,
            hst.sp_fbvelo_3g - hst.sp_fbvelo_std AS sp_fbvelo_delta_home,
            hst.sp_fbpct_3g - hst.sp_fbpct_std AS sp_fbpct_delta_home,
            hst.sp_whiff_3g - hst.sp_whiff_std AS sp_whiff_delta_home,
            th.team_woba_30g - tosh.team_woba_std AS woba_delta_home,
            th.team_iso_30g - tosh.team_iso_std AS team_iso_delta_home,
            th.team_k_rate_30g - tosh.team_k_rate_std AS team_k_rate_delta_home,
            th.team_bb_rate_30g - tosh.team_bb_rate_std AS team_bb_rate_delta_home,
            ch.team_barrel_15g - tcsh.team_barrel_std AS team_barrel_delta_home,
            ch.team_hardhit_15g - tcsh.team_hardhit_std AS team_hardhit_delta_home,
            ch.team_exitvelo_15g - tcsh.team_exitvelo_std AS team_exitvelo_delta_home,
            bh.bullpen_whip_10g - bpsh.bullpen_whip_std AS bullpen_whip_delta_home,
            bh.bullpen_era_10g - bpsh.bullpen_era_std AS bullpen_era_delta_home,
            lh.lineup_re24_mean - lsh.lineup_re24_mean_std AS lineup_re24_mean_delta_home,
            lh.lineup_re24_top3 - lsh.lineup_re24_top3_std AS lineup_re24_top3_delta_home,
            sa.sp_era_5g - sa.sp_era AS sp_era_delta_away,
            sa.sp_k9_5g - sa.sp_k9 AS sp_k9_delta_away,
            pa.sp_bb9_30g - pas.sp_bb9_std AS sp_bb9_delta_away,
            pa.sp_whip_30g - pas.sp_whip_std AS sp_whip_delta_away,
            pa.sp_xwoba_30g - pas.sp_xwoba_std AS sp_xwoba_delta_away,
            ast.sp_fbvelo_3g - ast.sp_fbvelo_std AS sp_fbvelo_delta_away,
            ast.sp_fbpct_3g - ast.sp_fbpct_std AS sp_fbpct_delta_away,
            ast.sp_whiff_3g - ast.sp_whiff_std AS sp_whiff_delta_away,
            ta.team_woba_30g - tosa.team_woba_std AS woba_delta_away,
            ta.team_iso_30g - tosa.team_iso_std AS team_iso_delta_away,
            ta.team_k_rate_30g - tosa.team_k_rate_std AS team_k_rate_delta_away,
            ta.team_bb_rate_30g - tosa.team_bb_rate_std AS team_bb_rate_delta_away,
            ca.team_barrel_15g - tcsa.team_barrel_std AS team_barrel_delta_away,
            ca.team_hardhit_15g - tcsa.team_hardhit_std AS team_hardhit_delta_away,
            ca.team_exitvelo_15g - tcsa.team_exitvelo_std AS team_exitvelo_delta_away,
            ba.bullpen_whip_10g - bpsa.bullpen_whip_std AS bullpen_whip_delta_away,
            ba.bullpen_era_10g - bpsa.bullpen_era_std AS bullpen_era_delta_away,
            la.lineup_re24_mean - lsa.lineup_re24_mean_std AS lineup_re24_mean_delta_away,
            la.lineup_re24_top3 - lsa.lineup_re24_top3_std AS lineup_re24_top3_delta_away,
            -- Experiment #2 source layer (see the exp2 block above). Side map:
            -- *_home = the HOME starter's priors / the AWAY lineup's priors
            -- (the batters the home starter faces), mirroring opp_lefty_share.
            s.home_starter_hand,
            s.away_starter_hand,
            lgk.league_k_pct,
            -- League per-category K% / xwOBA priors (date-level; identical
            -- for both sides like league_k_pct itself).
            lkcf.league_k_pct_cat AS league_k_pct_cat_fastball,
            lkcb.league_k_pct_cat AS league_k_pct_cat_breaking,
            lkco.league_k_pct_cat AS league_k_pct_cat_offspeed,
            lxkf.league_xwoba_cat AS league_xwoba_cat_fastball,
            lxkb.league_xwoba_cat AS league_xwoba_cat_breaking,
            lxko.league_xwoba_cat AS league_xwoba_cat_offspeed,
            lgl.league_k_pct_fb_vs AS league_k_pct_fb_vs_l,
            lgr.league_k_pct_fb_vs AS league_k_pct_fb_vs_r,
            hsc.sp_k_pct_cat AS sp_k_pct_cat_fastball_home,
            asc_.sp_k_pct_cat AS sp_k_pct_cat_fastball_away,
            bsc.sp_k_pct_cat AS sp_k_pct_cat_breaking_home,
            bsa.sp_k_pct_cat AS sp_k_pct_cat_breaking_away,
            osc.sp_k_pct_cat AS sp_k_pct_cat_offspeed_home,
            osa.sp_k_pct_cat AS sp_k_pct_cat_offspeed_away,
            hsu.sp_usage_cat AS sp_usage_cat_fastball_home,
            asu.sp_usage_cat AS sp_usage_cat_fastball_away,
            bsu.sp_usage_cat AS sp_usage_cat_breaking_home,
            bsau.sp_usage_cat AS sp_usage_cat_breaking_away,
            osu.sp_usage_cat AS sp_usage_cat_offspeed_home,
            osau.sp_usage_cat AS sp_usage_cat_offspeed_away,
            hsx.sp_xwoba_cat AS sp_xwoba_cat_fastball_home,
            asx.sp_xwoba_cat AS sp_xwoba_cat_fastball_away,
            bsx.sp_xwoba_cat AS sp_xwoba_cat_breaking_home,
            bsax.sp_xwoba_cat AS sp_xwoba_cat_breaking_away,
            osx.sp_xwoba_cat AS sp_xwoba_cat_offspeed_home,
            osax.sp_xwoba_cat AS sp_xwoba_cat_offspeed_away,
            tkc.team_k_pct_cat AS team_k_pct_cat_fastball_home,
            akc.team_k_pct_cat AS team_k_pct_cat_fastball_away,
            tkb.team_k_pct_cat AS team_k_pct_cat_breaking_home,
            akb.team_k_pct_cat AS team_k_pct_cat_breaking_away,
            tko.team_k_pct_cat AS team_k_pct_cat_offspeed_home,
            ako.team_k_pct_cat AS team_k_pct_cat_offspeed_away,
            txk.team_xwoba_cat AS team_xwoba_cat_fastball_home,
            axk.team_xwoba_cat AS team_xwoba_cat_fastball_away,
            txb.team_xwoba_cat AS team_xwoba_cat_breaking_home,
            axb.team_xwoba_cat AS team_xwoba_cat_breaking_away,
            txo.team_xwoba_cat AS team_xwoba_cat_offspeed_home,
            axo.team_xwoba_cat AS team_xwoba_cat_offspeed_away,
            hfb.sp_k_pct_fb_vs AS sp_k_pct_fb_vs_l_home,
            afb.sp_k_pct_fb_vs AS sp_k_pct_fb_vs_l_away,
            hfr.sp_k_pct_fb_vs AS sp_k_pct_fb_vs_r_home,
            afr.sp_k_pct_fb_vs AS sp_k_pct_fb_vs_r_away,
            hfb.sp_pa_fb_vs AS sp_pa_fb_vs_l_home,
            afb.sp_pa_fb_vs AS sp_pa_fb_vs_l_away,
            hfr.sp_pa_fb_vs AS sp_pa_fb_vs_r_home,
            afr.sp_pa_fb_vs AS sp_pa_fb_vs_r_away,
            tfb.team_k_pct_fb_vs AS team_k_pct_fb_vs_l_home,
            atb.team_k_pct_fb_vs AS team_k_pct_fb_vs_l_away,
            tfr.team_k_pct_fb_vs AS team_k_pct_fb_vs_r_home,
            atr.team_k_pct_fb_vs AS team_k_pct_fb_vs_r_away,
            tfb.team_pa_fb_vs AS team_pa_fb_vs_l_home,
            atb.team_pa_fb_vs AS team_pa_fb_vs_l_away,
            tfr.team_pa_fb_vs AS team_pa_fb_vs_r_home,
            atr.team_pa_fb_vs AS team_pa_fb_vs_r_away
        FROM game_winners w
        LEFT JOIN starters s ON w.game_pk = s.game_pk
        LEFT JOIN venues v ON w.game_pk = v.game_pk
        LEFT JOIN rest_days rh ON w.game_pk = rh.game_pk AND w.home_team = rh.team
        LEFT JOIN rest_days ra ON w.game_pk = ra.game_pk AND w.away_team = ra.team
        LEFT JOIN pitcher_season_features sh ON w.game_pk = sh.game_pk AND sh.pitcher = s.home_starter_id
        LEFT JOIN pitcher_season_features sa ON w.game_pk = sa.game_pk AND sa.pitcher = s.away_starter_id
        LEFT JOIN pitcher_features ph ON w.game_pk = ph.game_pk AND ph.pitcher = s.home_starter_id
        LEFT JOIN pitcher_features pa ON w.game_pk = pa.game_pk AND pa.pitcher = s.away_starter_id
        LEFT JOIN pitcher_season_std phs ON w.game_pk = phs.game_pk AND phs.pitcher = s.home_starter_id
        LEFT JOIN pitcher_season_std pas ON w.game_pk = pas.game_pk AND pas.pitcher = s.away_starter_id
        LEFT JOIN team_offense_rolling th ON w.game_pk = th.game_pk AND w.home_team = th.batting_team
        LEFT JOIN team_offense_rolling ta ON w.game_pk = ta.game_pk AND w.away_team = ta.batting_team
        LEFT JOIN team_off_season tosh ON w.game_pk = tosh.game_pk AND w.home_team = tosh.batting_team
        LEFT JOIN team_off_season tosa ON w.game_pk = tosa.game_pk AND w.away_team = tosa.batting_team
        LEFT JOIN bullpen_rolling bh ON w.game_pk = bh.game_pk AND w.home_team = bh.team
        LEFT JOIN bullpen_rolling ba ON w.game_pk = ba.game_pk AND w.away_team = ba.team
        LEFT JOIN bullpen_season bpsh ON w.game_pk = bpsh.game_pk AND w.home_team = bpsh.team
        LEFT JOIN bullpen_season bpsa ON w.game_pk = bpsa.game_pk AND w.away_team = bpsa.team
        LEFT JOIN pitcher_stuff hst ON w.game_pk = hst.game_pk AND hst.pitcher = s.home_starter_id
        LEFT JOIN pitcher_stuff ast ON w.game_pk = ast.game_pk AND ast.pitcher = s.away_starter_id
        LEFT JOIN team_contact_rolling ch ON w.game_pk = ch.game_pk AND w.home_team = ch.batting_team
        LEFT JOIN team_contact_rolling ca ON w.game_pk = ca.game_pk AND w.away_team = ca.batting_team
        LEFT JOIN team_contact_season tcsh ON w.game_pk = tcsh.game_pk AND w.home_team = tcsh.batting_team
        LEFT JOIN team_contact_season tcsa ON w.game_pk = tcsa.game_pk AND w.away_team = tcsa.batting_team
        LEFT JOIN team_hand_rolling hd ON w.game_pk = hd.game_pk AND w.away_team = hd.batting_team
        LEFT JOIN team_hand_rolling ha ON w.game_pk = ha.game_pk AND w.home_team = ha.batting_team
        LEFT JOIN bp_fatigue bf ON w.game_pk = bf.game_pk
        LEFT JOIN travel_fatigue tf ON w.game_pk = tf.game_pk
        LEFT JOIN closer_avail cl ON w.game_pk = cl.game_pk
        LEFT JOIN lineup_agg lh ON w.game_pk = lh.game_pk AND w.home_team = lh.batting_team
        LEFT JOIN lineup_agg la ON w.game_pk = la.game_pk AND w.away_team = la.batting_team
        LEFT JOIN pos_agg posh ON w.game_pk = posh.game_pk AND w.home_team = posh.batting_team
        LEFT JOIN pos_agg posa ON w.game_pk = posa.game_pk AND w.away_team = posa.batting_team
        LEFT JOIN lineup_il_flag fh ON w.game_pk = fh.game_pk AND w.home_team = fh.batting_team
        LEFT JOIN lineup_il_flag fa ON w.game_pk = fa.game_pk AND w.away_team = fa.batting_team
        LEFT JOIN lineup_season lsh ON w.game_pk = lsh.game_pk AND w.home_team = lsh.batting_team
        LEFT JOIN lineup_season lsa ON w.game_pk = lsa.game_pk AND w.away_team = lsa.batting_team
        LEFT JOIN lineup_ops_agg loh ON w.game_pk = loh.game_pk AND w.home_team = loh.batting_team
        LEFT JOIN lineup_ops_agg loa ON w.game_pk = loa.game_pk AND w.away_team = loa.batting_team
        LEFT JOIN exp2_league_k lgk ON w.game_date = lgk.game_date
        LEFT JOIN exp2_league_cat lkcf ON w.game_date = lkcf.game_date AND lkcf.pitch_cat = 'fastball'
        LEFT JOIN exp2_league_cat lkcb ON w.game_date = lkcb.game_date AND lkcb.pitch_cat = 'breaking'
        LEFT JOIN exp2_league_cat lkco ON w.game_date = lkco.game_date AND lkco.pitch_cat = 'offspeed'
        LEFT JOIN exp2_league_cat lxkf ON w.game_date = lxkf.game_date AND lxkf.pitch_cat = 'fastball'
        LEFT JOIN exp2_league_cat lxkb ON w.game_date = lxkb.game_date AND lxkb.pitch_cat = 'breaking'
        LEFT JOIN exp2_league_cat lxko ON w.game_date = lxko.game_date AND lxko.pitch_cat = 'offspeed'
        LEFT JOIN exp2_league_fbhand lgl ON w.game_date = lgl.game_date AND lgl.stand = 'L'
        LEFT JOIN exp2_league_fbhand lgr ON w.game_date = lgr.game_date AND lgr.stand = 'R'
        LEFT JOIN exp2_sp_cat hsc ON w.game_date = hsc.game_date AND hsc.pitcher = s.home_starter_id AND hsc.pitch_cat = 'fastball'
        LEFT JOIN exp2_sp_cat asc_ ON w.game_date = asc_.game_date AND asc_.pitcher = s.away_starter_id AND asc_.pitch_cat = 'fastball'
        LEFT JOIN exp2_sp_cat bsc ON w.game_date = bsc.game_date AND bsc.pitcher = s.home_starter_id AND bsc.pitch_cat = 'breaking'
        LEFT JOIN exp2_sp_cat bsa ON w.game_date = bsa.game_date AND bsa.pitcher = s.away_starter_id AND bsa.pitch_cat = 'breaking'
        LEFT JOIN exp2_sp_cat osc ON w.game_date = osc.game_date AND osc.pitcher = s.home_starter_id AND osc.pitch_cat = 'offspeed'
        LEFT JOIN exp2_sp_cat osa ON w.game_date = osa.game_date AND osa.pitcher = s.away_starter_id AND osa.pitch_cat = 'offspeed'
        LEFT JOIN exp2_sp_cat hsu ON w.game_date = hsu.game_date AND hsu.pitcher = s.home_starter_id AND hsu.pitch_cat = 'fastball'
        LEFT JOIN exp2_sp_cat asu ON w.game_date = asu.game_date AND asu.pitcher = s.away_starter_id AND asu.pitch_cat = 'fastball'
        LEFT JOIN exp2_sp_cat bsu ON w.game_date = bsu.game_date AND bsu.pitcher = s.home_starter_id AND bsu.pitch_cat = 'breaking'
        LEFT JOIN exp2_sp_cat bsau ON w.game_date = bsau.game_date AND bsau.pitcher = s.away_starter_id AND bsau.pitch_cat = 'breaking'
        LEFT JOIN exp2_sp_cat osu ON w.game_date = osu.game_date AND osu.pitcher = s.home_starter_id AND osu.pitch_cat = 'offspeed'
        LEFT JOIN exp2_sp_cat osau ON w.game_date = osau.game_date AND osau.pitcher = s.away_starter_id AND osau.pitch_cat = 'offspeed'
        LEFT JOIN exp2_sp_cat hsx ON w.game_date = hsx.game_date AND hsx.pitcher = s.home_starter_id AND hsx.pitch_cat = 'fastball'
        LEFT JOIN exp2_sp_cat asx ON w.game_date = asx.game_date AND asx.pitcher = s.away_starter_id AND asx.pitch_cat = 'fastball'
        LEFT JOIN exp2_sp_cat bsx ON w.game_date = bsx.game_date AND bsx.pitcher = s.home_starter_id AND bsx.pitch_cat = 'breaking'
        LEFT JOIN exp2_sp_cat bsax ON w.game_date = bsax.game_date AND bsax.pitcher = s.away_starter_id AND bsax.pitch_cat = 'breaking'
        LEFT JOIN exp2_sp_cat osx ON w.game_date = osx.game_date AND osx.pitcher = s.home_starter_id AND osx.pitch_cat = 'offspeed'
        LEFT JOIN exp2_sp_cat osax ON w.game_date = osax.game_date AND osax.pitcher = s.away_starter_id AND osax.pitch_cat = 'offspeed'
        LEFT JOIN exp2_team_cat tkc ON w.game_date = tkc.game_date AND w.away_team = tkc.batting_team AND tkc.pitch_cat = 'fastball'
        LEFT JOIN exp2_team_cat akc ON w.game_date = akc.game_date AND w.home_team = akc.batting_team AND akc.pitch_cat = 'fastball'
        LEFT JOIN exp2_team_cat tkb ON w.game_date = tkb.game_date AND w.away_team = tkb.batting_team AND tkb.pitch_cat = 'breaking'
        LEFT JOIN exp2_team_cat akb ON w.game_date = akb.game_date AND w.home_team = akb.batting_team AND akb.pitch_cat = 'breaking'
        LEFT JOIN exp2_team_cat tko ON w.game_date = tko.game_date AND w.away_team = tko.batting_team AND tko.pitch_cat = 'offspeed'
        LEFT JOIN exp2_team_cat ako ON w.game_date = ako.game_date AND w.home_team = ako.batting_team AND ako.pitch_cat = 'offspeed'
        LEFT JOIN exp2_team_cat txk ON w.game_date = txk.game_date AND w.away_team = txk.batting_team AND txk.pitch_cat = 'fastball'
        LEFT JOIN exp2_team_cat axk ON w.game_date = axk.game_date AND w.home_team = axk.batting_team AND axk.pitch_cat = 'fastball'
        LEFT JOIN exp2_team_cat txb ON w.game_date = txb.game_date AND w.away_team = txb.batting_team AND txb.pitch_cat = 'breaking'
        LEFT JOIN exp2_team_cat axb ON w.game_date = axb.game_date AND w.home_team = axb.batting_team AND axb.pitch_cat = 'breaking'
        LEFT JOIN exp2_team_cat txo ON w.game_date = txo.game_date AND w.away_team = txo.batting_team AND txo.pitch_cat = 'offspeed'
        LEFT JOIN exp2_team_cat axo ON w.game_date = axo.game_date AND w.home_team = axo.batting_team AND axo.pitch_cat = 'offspeed'
        LEFT JOIN exp2_sp_fbhand hfb ON w.game_date = hfb.game_date AND hfb.pitcher = s.home_starter_id AND hfb.stand = 'L'
        LEFT JOIN exp2_sp_fbhand afb ON w.game_date = afb.game_date AND afb.pitcher = s.away_starter_id AND afb.stand = 'L'
        LEFT JOIN exp2_sp_fbhand hfr ON w.game_date = hfr.game_date AND hfr.pitcher = s.home_starter_id AND hfr.stand = 'R'
        LEFT JOIN exp2_sp_fbhand afr ON w.game_date = afr.game_date AND afr.pitcher = s.away_starter_id AND afr.stand = 'R'
        LEFT JOIN exp2_team_fbhand tfb ON w.game_date = tfb.game_date AND w.away_team = tfb.batting_team AND tfb.stand = 'L'
        LEFT JOIN exp2_team_fbhand atb ON w.game_date = atb.game_date AND w.home_team = atb.batting_team AND atb.stand = 'L'
        LEFT JOIN exp2_team_fbhand tfr ON w.game_date = tfr.game_date AND w.away_team = tfr.batting_team AND tfr.stand = 'R'
        LEFT JOIN exp2_team_fbhand atr ON w.game_date = atr.game_date AND w.home_team = atr.batting_team AND atr.stand = 'R'
    """)

    n = con.execute("SELECT COUNT(*) FROM game_level").fetchone()[0]
    logger.info("game_level built: %d games", n)

    for tbl in (
        "game_winners", "starters", "venues", "rest_days",
        "pa_boundary",        "pitcher_game_stats", "pitcher_stale", "pitcher_shifted", "pitcher_rolling",
        "pitcher_shifted_season", "pitcher_season_rolling", "pitcher_5g_rolling",
        "pitcher_era_league", "pitcher_season_features", "pitcher_features",
        "team_offense_raw", "team_off_shifted", "team_offense_rolling",
        "team_off_season",
        "bullpen_raw", "bp_shrink_prior", "bullpen_shifted", "bullpen_rolling", "bullpen_season",
        "bp_daily", "bp_fatigue", "il_stints_pitchers",
        "bp_outing", "bp_day2", "bp_spent",
        "pitcher_stuff_raw", "pitcher_stuff",
        "pitcher_season_full", "pitcher_season_std",
        "team_contact_raw", "team_contact_shifted", "team_contact_rolling",
        "team_contact_season",
        "lineup_agg_shifted", "lineup_season", "lineup_pool", "lineup_il_flag",
        "pos_pool", "pos_league", "pos_agg",
        "team_hand_raw", "team_hand_shifted", "team_hand_rolling",
        "batter_game_stats", "batter_shifted", "batter_rolling",
        "batter_league", "batter_ratings", "lineup_agg",
        "batter_league_pos",
        "batter_hand_game", "batter_hand_shifted", "batter_hand_rolling", "lineup_ops_agg",
        "exp2_pa", "exp2_league_k", "exp2_league_cat", "exp2_sp_cat_game",
        "exp2_sp_fbhand_game", "exp2_sp_cat_daily", "exp2_sp_cat_cum",
        "exp2_sp_cat_tot_daily", "exp2_sp_cat_tot_cum", "exp2_sp_cat_prelim",
        "exp2_sp_cat", "exp2_sp_fbhand_daily", "exp2_sp_fbhand_cum",
        "exp2_sp_fbhand", "exp2_team_cat_game", "exp2_team_cat_daily",
        "exp2_team_cat_cum", "exp2_team_cat", "exp2_team_fbhand_game",
        "exp2_team_fbhand_daily", "exp2_team_fbhand_cum", "exp2_team_fbhand",
        "exp2_league_fbhand",
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

# Retractable-roof venues: venue-level DOME_STATUS=1 is only correct when the
# roof is actually CLOSED. For these teams roof state is resolved PER GAME
# from the StatsAPI feed condition ("Roof Open"/"Roof Closed"/"Indoor").
# Fixed domes (TB) are always closed. MIN is a known data-quality anomaly:
# Target Field has been open-air since 2010 yet DOME_STATUS says 1.
RETRACTABLE_ROOF_TEAMS = frozenset(
    {"ARI", "AZ", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR"})
FIXED_DOME_TEAMS = frozenset({"TB"})
OPEN_AIR_MISLABELED = frozenset({"MIN"})  # flagged dome=1 but open-air

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
#
# INDEPENDENT A/B ARM (2026-10-02): win_pct_diff carries ~8.65% of model
# weight — ~4x the entire lineup family — so this schedule is gated on
# its own (MLB_WINPCT_SHRINK_ARM), default bayesian until the gate rules:
#   bayesian (default)  w = G/(G+30)      never fully raw (84% own at G=162)
#   ramp                w = min(G/30, 1)  linear own weight below 30 games,
#                       raw from game 30 (~mid-May); continuous at the
#                       threshold, monotone in season-to-date G (crosses
#                       once, resets every April).
# Independent of MLB_SHRINK_ARM (batter ratings, adopted ramp 2026-10-02):
# per-family switches, per-family gates, per-family adoption. The level
# twins (home_win_pct/away_win_pct) are RAW under every arm.
WIN_PCT_SHRINKAGE_GAMES = 30.0
MLB_WINPCT_SHRINK_ARM = (os.getenv("MLB_WINPCT_SHRINK_ARM", "bayesian")
                         .strip().lower() or "bayesian")


def _smoothed_win_pct(wins: pd.Series, losses: pd.Series,
                      arm: str | None = None) -> pd.Series:
    """Win pct shrunk toward .500 by games played (early-season smoothing).

    bayesian (default): ``(wins + K/2) / (games + K)`` — equals exactly
    0.500 before a team plays, heavily smoothed early season, converging
    toward (but never reaching) the raw win pct as the season matures.

    ramp: own weight ``w = min(G/K, 1)`` — the linear batter-arm
    schedule; raw win pct at/above K games, continuous at the threshold.

    ``arm=None`` uses the module default (``MLB_WINPCT_SHRINK_ARM``);
    NaN inputs stay NaN under both arms; an unknown arm raises so a typo
    can never silently ship the default.
    """
    wins = pd.to_numeric(wins, errors="coerce")
    losses = pd.to_numeric(losses, errors="coerce")
    games = (wins + losses).clip(lower=0)
    k = WIN_PCT_SHRINKAGE_GAMES
    chosen = MLB_WINPCT_SHRINK_ARM if arm is None else arm
    if chosen == "bayesian":
        return (wins + 0.5 * k) / (games + k)
    if chosen == "ramp":
        below = games < k
        # below k: w*p + (1-w)*.5 with w = G/k, algebraically
        # (wins + (k - G)*.5)/k; at/above k: raw (denominator masked to
        # NaN below k so the division never sees a small/zero G).
        blended = (wins + (k - games) * 0.5) / k
        raw = wins / games.where(~below)
        return blended.where(below, raw)
    raise ValueError(
        f"MLB_WINPCT_SHRINK_ARM must be 'bayesian' or 'ramp', "
        f"got {chosen!r}")



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


def roof_state_from_condition(condition) -> str | None:
    """Map a StatsAPI gameData.weather.condition string to open/closed/None.

    Retractable parks report conditions like "Roof Closed", "Roof Open",
    "Indoor". Anything else (clear/cloudy/missing) is NOT a roof statement
    -> None (unknown), which callers must treat loudly, never as closed.
    """
    if condition is None:
        return None
    text = str(condition).strip().lower()
    if not text:
        return None
    if "roof closed" in text or text == "indoor" or "closed roof" in text:
        return "closed"
    if "roof open" in text or "open roof" in text:
        return "open"
    return None



def load_roof_cache(path) -> dict:
    """Load the StatsAPI roof-state cache JSON -> {int(game_pk): "open"|"closed"|None}.

    Deduplicates by int key (last wins) and warns loudly on duplicates.
    Used by _fetch_roofs.py and the pipeline's roof-state loader.
    """
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    cache: dict = {}
    dups: int = 0
    for k, v in raw.items():
        pk = int(k)
        if pk in cache:
            dups += 1
        cache[pk] = v
    if dups:
        logger.warning("load_roof_cache: %d duplicate keys in %s (last wins)",
                        dups, p.name)
    return cache


def refine_dome_game_level(df: pd.DataFrame,
                           roof_states: dict | None = None) -> pd.DataFrame:
    """Game-accurate roof flag column: dome_is_neutral_game.

    Venue-level dome_is_neutral is kept UNTOUCHED. The refined column:
      * fixed domes           -> 1 (always closed)
      * retractable + known   -> 0 when OPEN (real weather applies),
                                 1 when CLOSED
      * retractable + unknown -> venue value + LOUD WARNING (never silently
                                 assumed closed)
      * open-air venues       -> 0 (includes the MIN mislabel correction)
    ``roof_states`` maps game_pk -> "open"|"closed" (StatsAPI cache).
    """
    from collections import Counter

    df = df.copy()
    roof_states = roof_states or {}
    refined = []
    unknown_retractable = 0
    unknown_teams = Counter()
    for _, row in df.iterrows():
        team = str(row.get("home_team", "")).upper().strip()
        venue_dome = row.get("dome_is_neutral")
        venue_dome = float(venue_dome) if pd.notna(venue_dome) else np.nan
        if team in FIXED_DOME_TEAMS:
            refined.append(1.0)
        elif team in RETRACTABLE_ROOF_TEAMS:
            state = roof_states.get(row.get("game_pk")) \
                or roof_state_from_condition(row.get("statsapi_condition"))
            if state == "open":
                refined.append(0.0)
            elif state == "closed":
                refined.append(1.0)
            else:
                unknown_retractable += 1
                unknown_teams[team] += 1
                refined.append(venue_dome if pd.notna(venue_dome) else 1.0)
        elif team in OPEN_AIR_MISLABELED:
            refined.append(0.0)
        else:
            refined.append(venue_dome if pd.notna(venue_dome) else 0.0)
    df["dome_is_neutral_game"] = pd.Series(refined, index=df.index,
                                           dtype="float64")
    n_open = int((df["dome_is_neutral_game"] == 0).sum())
    log = logger.warning if unknown_retractable else logger.info
    log(
        "Dome refinement: %d/%d games resolved; %d retractable games have "
        "UNKNOWN roof state (teams: %s) - venue fallback applied, never "
        "silently treated as closed",
        len(df) - unknown_retractable, len(df), unknown_retractable,
        dict(unknown_teams) if unknown_teams else "{}")
    logger.info(
        "Dome refinement: %d games play under real weather after refinement",
        n_open)
    return df


def add_env_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standalone environment-LEVEL columns for the run engine (additive).

    - park_wind_factor: wind speed × sign(cos(wind_dir − park_bearing)),
      derived from the weather cache raw fields (wind_speed_kmh,
      wind_direction_deg, stadium_bearing). Dome/closed-roof = 0.
    - air_density_level: raw air density (kg/m³) from the weather cache
      (computed from temp_c, rh_pct, pressure_hpa, altitude_m).
    - park_factor_slug: pure park SLG factor from PARK_FACTORS_SLG.

    Coverage depends on the weather cache. The weather cache should cover
    ~90%+ of decided games. NULLs mean no observation was available — never
    fabricated. Missing sources produce NULLs plus loud warnings.
    """
    import os
    from pathlib import Path as _P

    df = df.copy()
    home_team = df["home_team"].astype(str).str.upper().str.strip()

    # ── park_factor_slug (static, from PARK_FACTORS_SLG) ──────────────
    pf_raw = home_team.map(PARK_FACTORS_SLG).astype(float)
    df["park_factor_slug"] = (pf_raw - 100.0) / 100.0
    missing_park = int(pf_raw.isna().sum())
    if missing_park:
        logger.warning(
            "Env-level features: %d/%d rows have no PARK_FACTORS_SLG entry "
            "for home_team - park_factor_slug left NULL",
            missing_park, len(df))

    # ── park_wind_factor & air_density_level from weather cache ────────
    # The weather cache stores per-game raw observations keyed by game_pk.
    # We fill NULL level columns from it — no division of interactions.
    if "park_wind_factor" not in df.columns:
        df["park_wind_factor"] = np.nan
    if "air_density_level" not in df.columns:
        df["air_density_level"] = np.nan

    _need_fill_w = df["park_wind_factor"].isna().any()
    _need_fill_a = df["air_density_level"].isna().any()
    if _need_fill_w or _need_fill_a:
        cache_path = _P(__file__).resolve().parent.parent / "data_delivery" / "weather_history.parquet"
        if not cache_path.exists() and os.getenv("MLB_CACHE_DIR"):
            cache_path = _P(os.getenv("MLB_CACHE_DIR")) / "weather_history.parquet"
        if cache_path.exists():
            try:
                wx_cache = pd.read_parquet(cache_path)
                if "game_pk" in df.columns and "game_pk" in wx_cache.columns:
                    wx_cache = wx_cache.copy()
                    wx_cache["game_pk"] = pd.to_numeric(wx_cache["game_pk"], errors="coerce").astype("Int64")
                    df["_gpk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
                    wx_map = wx_cache.set_index("game_pk")

                    if _need_fill_w:
                        # park_wind_factor = wind_multiplier (already scaled
                        # by speed, sign from bearing). Domes genuinely = 0.
                        wm_series = pd.Series(np.nan, index=df.index, dtype=float)
                        has_wx = df["_gpk"].isin(wx_map.index)
                        matched = df.loc[has_wx, "_gpk"].map(wx_map["wind_multiplier"])
                        wm_series[has_wx] = pd.to_numeric(matched, errors="coerce")
                        # Dome closed-roof → genuinely 0 wind (valid observation)
                        dome_col = "dome_is_neutral_game" if "dome_is_neutral_game" in df.columns else "dome_is_neutral"
                        dome_mask = pd.to_numeric(df.get(dome_col), errors="coerce") == 1
                        wm_series = wm_series.where(~(dome_mask & wm_series.isna()), 0.0)
                        # Closed roof forces 0 ALWAYS - even when the cache
                        # carries an outdoor wind value fetched at the stadium's
                        # coordinates: the roof state, not the outdoor reading,
                        # decides the wind.
                        df["park_wind_factor"] = (
                            df["park_wind_factor"].fillna(wm_series)
                            .where(~dome_mask, 0.0)
                        )

                    if _need_fill_a:
                        # air_density_level = air_density (kg/m³ from raw fields)
                        ad_series = pd.Series(np.nan, index=df.index, dtype=float)
                        has_wx = df["_gpk"].isin(wx_map.index)
                        matched = df.loc[has_wx, "_gpk"].map(wx_map["air_density"])
                        ad_series[has_wx] = pd.to_numeric(matched, errors="coerce")
                        df["air_density_level"] = df["air_density_level"].fillna(ad_series)

                    df.drop(columns=["_gpk"], inplace=True, errors="ignore")
            except Exception as exc:
                logger.warning("Weather cache load failed for env-level features: %s", exc)
        else:
            logger.info("Env-level features: weather cache not found at %s; "
                        "level columns rely on apply_weather_features fill", cache_path)

    # Closed-roof/dome games are genuinely neutral for BOTH interaction
    # components: outdoor readings fetched at a dome's coordinates must
    # never price wind/air advantage indoors. The interaction columns were
    # built earlier with the VENUE flag (or carried stale pre-cache
    # values); correct them here with the GAME-level roof state so the
    # run engine never trains on non-zero dome wind. Missing diff inputs
    # stay NULL (never a fabricated 0).
    dome_col2 = ("dome_is_neutral_game"
                 if "dome_is_neutral_game" in df.columns
                 else "dome_is_neutral")
    dome_flag = pd.to_numeric(df.get(dome_col2), errors="coerce") == 1
    if "wind_advantage_flyball_factor" in df.columns:
        _era_ok = pd.to_numeric(df.get("sp_era_diff"),
                               errors="coerce").notna()
        df.loc[dome_flag & _era_ok, "wind_advantage_flyball_factor"] = 0.0
    if "air_density_velocity_boost" in df.columns:
        _velo_ok = pd.to_numeric(df.get("sp_fbvelo_diff"),
                                 errors="coerce").notna()
        _density_ok = pd.to_numeric(df["air_density_level"], errors="coerce").notna()
        df.loc[dome_flag & _velo_ok & _density_ok, "air_density_velocity_boost"] = 0.0
        df.loc[dome_flag & ~_density_ok, "air_density_velocity_boost"] = np.nan

    n_w = int(df["park_wind_factor"].notna().sum())
    n_a = int(df["air_density_level"].notna().sum())
    n_p = int(df["park_factor_slug"].notna().sum())
    logger.info(
        "Env-level features (raw-derived): park_wind_factor %d/%d, "
        "air_density_level %d/%d, park_factor_slug %d/%d",
        n_w, len(df), n_a, len(df), n_p, len(df))
    if n_w < 0.5 * len(df):
        logger.warning("Env-level features: park_wind_factor only %.0f%% populated "
                       "— weather cache coverage may be low", 100 * n_w / len(df))
    if n_a < 0.5 * len(df):
        logger.warning("Env-level features: air_density_level only %.0f%% populated "
                       "— weather cache coverage may be low", 100 * n_a / len(df))
    return df


# ── Momentum form-delta features ─────────────────────────────────────────────
# Continuous "recent window − season-to-date baseline" per side, per stat
# family. The model learns its own thresholds (a binary ">10% improvement"
# flag is strictly weaker — the trees can derive it from the continuous
# version). Single computation point: the DuckDB layer ships the delta
# columns in game_level; this helper is the fallback that computes them from
# the shipped recent/season columns on frames that predate the refresh (e.g.
# the committed CSV). Idempotent: an existing delta column is never
# overwritten (the SQL-computed value is authoritative).

# ABLATION VERDICT (2026-08): NOT SHIPPED into the moneyline. The WITH-vs-
# WITHOUT measurement on the committed CSV (run_form_delta_ablation.py) lost
# BOTH pooled OOF (0.6895/0.5494 vs 0.6867/0.5540) and the sealed 21-day
# holdout (0.6829/0.5437 vs 0.6814/0.5529), so MONEYLINE_FEATURE_COLS excludes them.
# The columns still ship in the artifact — re-test on a refreshed artifact
# before re-enabling.

# (delta_base, recent_col_base, season_col_base, window_label)
FORM_DELTA_SPECS: list[tuple[str, str, str, str]] = [
    ("sp_era_delta",        "sp_era_5g",       "sp_era",             "last 5 starts − season to date"),
    ("sp_k9_delta",         "sp_k9_5g",        "sp_k9",              "last 5 starts − season to date"),
    ("sp_bb9_delta",        "sp_bb9",          "sp_bb9_std",         "last 30 starts − season to date"),
    ("sp_whip_delta",       "sp_whip",         "sp_whip_std",        "last 30 starts − season to date"),
    ("sp_xwoba_delta",      "sp_xwoba",        "sp_xwoba_std",       "last 30 starts − season to date"),
    ("sp_fbvelo_delta",     "sp_fbvelo_3g",    "sp_fbvelo_std",      "last 3 starts − season to date"),
    ("sp_fbpct_delta",      "sp_fbpct_3g",     "sp_fbpct_std",       "last 3 starts − season to date"),
    ("sp_whiff_delta",      "sp_whiff_3g",     "sp_whiff_std",       "last 3 starts − season to date"),
    ("woba_delta",          "woba_30g",        "team_woba_std",      "last 30 games − season to date"),
    ("team_iso_delta",      "team_iso_30g",    "team_iso_std",       "last 30 games − season to date"),
    ("team_k_rate_delta",   "team_k_rate_30g", "team_k_rate_std",    "last 30 games − season to date"),
    ("team_bb_rate_delta",  "team_bb_rate_30g","team_bb_rate_std",   "last 30 games − season to date"),
    ("team_barrel_delta",   "team_barrel_15g", "team_barrel_std",    "last 15 games − season to date"),
    ("team_hardhit_delta",  "team_hardhit_15g","team_hardhit_std",   "last 15 games − season to date"),
    ("team_exitvelo_delta", "team_exitvelo_15g","team_exitvelo_std", "last 15 games − season to date"),
    ("bullpen_whip_delta",  "bullpen_whip_10g","bullpen_whip_std",   "last 10 games − season to date"),
    ("bullpen_era_delta",   "bullpen_era_10g", "bullpen_era_std",    "last 10 games − season to date"),
    ("lineup_re24_mean_delta", "lineup_re24_mean", "lineup_re24_mean_std", "today's lineup − season-to-date lineup"),
    ("lineup_re24_top3_delta", "lineup_re24_top3", "lineup_re24_top3_std", "today's lineup − season-to-date lineup"),
]

# All 38 column names, canonical order (family-major, side-minor).
FORM_DELTA_COLS: list[str] = [
    f"{base}_{side}"
    for base, *_ in FORM_DELTA_SPECS
    for side in ("home", "away")
]


def add_form_delta_features(game_df: pd.DataFrame,
                            inplace: bool = False) -> pd.DataFrame:
    """Compute momentum form-delta columns (recent − season-to-date), per side.

    For every spec in FORM_DELTA_SPECS, if the delta column is already present
    (shipped by the DuckDB layer — authoritative) it is left untouched. If
    missing, it is computed as ``recent_col_<side> − season_col_<side>`` when
    BOTH source columns exist in the frame; otherwise the column is created as
    all-NaN (thin early-season coverage is handled by the existing imputation
    path — never fabricated).

    On the committed pre-refresh CSV only the SP era/K9 deltas are computable
    (those season baselines are shipped); the rest stay NaN until the next
    pipeline run ships the season columns / deltas.

    Returns the frame (same object when inplace=True, else a copy).
    """
    df = game_df if inplace else game_df.copy()
    created: list[str] = []
    for base, recent, season, _window in FORM_DELTA_SPECS:
        for side in ("home", "away"):
            col = f"{base}_{side}"
            if col in df.columns:
                continue  # SQL-shipped value is authoritative
            r = f"{recent}_{side}"
            s = f"{season}_{side}"
            if r in df.columns and s in df.columns:
                df[col] = pd.to_numeric(df[r], errors="coerce") - pd.to_numeric(df[s], errors="coerce")
            else:
                df[col] = np.nan
            created.append(col)
    if created:
        logger.info("add_form_delta_features: computed %d delta column(s) missing from the frame", len(created))
    return df


# ── Experiment #2 candidate features (SHIPPED 2026-09-07) ──────────────────
# The 8 frozen exp2 matchup candidates (C+E moneyline / D+F run-line decision,
# run_exp2_feature_test registry exp2_feature_test_20260907.json), built as
# pure arithmetic on the PIT-safe source columns shipped by the Experiment #2
# source layer (league priors, category K%/xwOBA, platoon splits). Every
# input is a point-in-time season-to-date aggregate (source_game_date <
# target_game_date, doubleheader-safe date-level ASOF), so the arithmetic
# here adds no new temporal exposure. Coverage gaps (offspeed 0.56,
# breaking 0.79) stay NULL — the feature matrix preserves NaN (tree members
# route it; logistic/mlp impute). The gaps are a SOURCE-AVAILABILITY floor,
# not an artifact: a starter with no tracked offspeed/breaking usage has no
# per-side value, and the twins inherit exactly that floor (a twin dominates
# its diff, so the diff can never be better covered than its halves).
#
# 2026-09-27 structural expansion: each diff also serves its raw per-side
# halves under final names (exp2_*_home / exp2_*_away) — the very scratch
# this function always computed and then dropped. Same arithmetic, same
# sources, so home − away == diff by construction; 8 diffs + 16 twins =
# 24 created columns.
EXP2_CANDIDATE_COLS: list[str] = [
    "exp2_centered_k_diff",
    "exp2_cat_k_fastball_diff", "exp2_cat_k_breaking_diff",
    "exp2_cat_k_offspeed_diff",
    "exp2_cat_xwoba_fastball_diff", "exp2_cat_xwoba_breaking_diff",
    "exp2_cat_xwoba_offspeed_diff",
    "exp2_cat_platoon_k_fastball_diff",
]
_EXP2_CATEGORIES = ("fastball", "breaking", "offspeed")
# Per-side twins of the candidates, in canonical order (the frame's scratch
# values served under final names; tree-only routing per the level rule).
EXP2_TWIN_COLS: list[str] = [
    "exp2_centered_k_home", "exp2_centered_k_away",
    "exp2_cat_k_fastball_home", "exp2_cat_k_fastball_away",
    "exp2_cat_k_breaking_home", "exp2_cat_k_breaking_away",
    "exp2_cat_k_offspeed_home", "exp2_cat_k_offspeed_away",
    "exp2_cat_xwoba_fastball_home", "exp2_cat_xwoba_fastball_away",
    "exp2_cat_xwoba_breaking_home", "exp2_cat_xwoba_breaking_away",
    "exp2_cat_xwoba_offspeed_home", "exp2_cat_xwoba_offspeed_away",
    "exp2_cat_platoon_k_fastball_home", "exp2_cat_platoon_k_fastball_away",
]


def add_exp2_features(game_df: pd.DataFrame,
                      inplace: bool = False) -> pd.DataFrame:
    """Compute the 8 frozen exp2 matchup candidates from source columns.

    Formulas are FROZEN from run_exp2_feature_test.add_candidates (the
    experiment that produced the adoption decision) — identical arithmetic,
    identical column names, no alternate windows/transformations:

      centered_k(side)     = (sp_k9 − lg_k)·(team_k_rate_30g − lg_k)
      cat_k_cat(side)      = sp_usage_cat·(sp_k_cat − lg_k_cat)·(opp_k_cat − lg_k_cat)
      cat_xwoba_cat(side)  = sp_usage_cat·(sp_xwoba_cat − lg_x_cat)·(opp_x_cat − lg_x_cat)
      platoon(side)        = Σ_hand share_hand·(SP_fb_vs_hand − lg_fb_hand)
                             · Σ_hand share_hand·(opp_fb_vs_hand − lg_fb_hand)
                             · sp_usage_fastball      (share = opposing lineup L/R)

    each shipped as home − away. NOTE (recorded in the experiment registry,
    frozen before results): candidate 1 mixes denominators — SP K/9 vs
    opponent/league K/PA — by design. Missing inputs propagate as NaN, never
    a fabricated 0.

    2026-09-27: the per-side halves are ALSO served under final names
    (exp2_*_home / exp2_*_away) — the same arithmetic's scratch values,
    kept instead of dropped, so every served diff exposes its raw halves
    (home − away == diff by construction). 24 columns created in total.

    Returns the frame (same object when inplace=True, else a copy).
    """
    df = game_df if inplace else game_df.copy()

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    missing_src = sorted({c for c in (
        ["league_k_pct", "league_k_pct_fb_vs_l", "league_k_pct_fb_vs_r",
         "opp_lefty_share_home", "opp_lefty_share_away"]
        + [f"sp_k9_{s}" for s in ("home", "away")]
        + [f"team_k_rate_30g_{s}" for s in ("home", "away")]
        + [f"{p}_{c}_{s}" for c in _EXP2_CATEGORIES
           for p in ("sp_usage_cat", "sp_k_pct_cat", "team_k_pct_cat",
                     "sp_xwoba_cat", "team_xwoba_cat")
           for s in ("home", "away")]
        # league category priors are DATE-level (unsuffixed, identical for
        # both sides — same convention as league_k_pct itself)
        + [f"league_k_pct_cat_{c}" for c in _EXP2_CATEGORIES]
        + [f"league_xwoba_cat_{c}" for c in _EXP2_CATEGORIES]
        + [f"sp_k_pct_fb_vs_{h}_{s}" for h in ("l", "r") for s in ("home", "away")]
        + [f"team_k_pct_fb_vs_{h}_{s}" for h in ("l", "r") for s in ("home", "away")]
        + [f"sp_usage_cat_fastball_{s}" for s in ("home", "away")]
    ) if c not in df.columns})
    if missing_src:
        for c in [*EXP2_CANDIDATE_COLS, *EXP2_TWIN_COLS]:
            df[c] = np.nan
        logger.warning(
            "add_exp2_features: %d source columns absent (%s…) — all 8 "
            "candidates + 16 per-side twins ship as NaN (thin/partial frame)",
            len(missing_src), ", ".join(missing_src[:5]))
        return df

    lg_k = _num("league_k_pct")

    # 1. centered K — production K-rate columns: SP = sp_k9 (K/9), opp =
    #    team_k_rate_30g (offense K/PA), league = league_k_pct (K/PA).
    for side in ("home", "away"):
        sp_c = _num(f"sp_k9_{side}") - lg_k
        opp_c = _num(f"team_k_rate_30g_{side}") - lg_k
        df[f"_exp2_side_{side}_centered_k"] = sp_c * opp_c
    df["exp2_centered_k_diff"] = (df["_exp2_side_home_centered_k"]
                                  - df["_exp2_side_away_centered_k"])

    # 2-4 / 5-7: category K and xwOBA (directionality: higher xwOBA = worse
    # for the pitcher; product is the same amplification form — no sign flip).
    for cat in _EXP2_CATEGORIES:
        lg_kc = _num(f"league_k_pct_cat_{cat}")
        lg_xc = _num(f"league_xwoba_cat_{cat}")
        for side in ("home", "away"):
            df[f"_exp2_side_{side}_cat_k_{cat}"] = (
                _num(f"sp_usage_cat_{cat}_{side}")
                * (_num(f"sp_k_pct_cat_{cat}_{side}") - lg_kc)
                * (_num(f"team_k_pct_cat_{cat}_{side}") - lg_kc))
            df[f"_exp2_side_{side}_cat_xwoba_{cat}"] = (
                _num(f"sp_usage_cat_{cat}_{side}")
                * (_num(f"sp_xwoba_cat_{cat}_{side}") - lg_xc)
                * (_num(f"team_xwoba_cat_{cat}_{side}") - lg_xc))
        df[f"exp2_cat_k_{cat}_diff"] = (df[f"_exp2_side_home_cat_k_{cat}"]
                                        - df[f"_exp2_side_away_cat_k_{cat}"])
        df[f"exp2_cat_xwoba_{cat}_diff"] = (
            df[f"_exp2_side_home_cat_xwoba_{cat}"]
            - df[f"_exp2_side_away_cat_xwoba_{cat}"])

    # 8: platoon fastball K — for side s, the OPPOSING lineup is the other
    #    team's offense: its L share is opp_lefty_share_<other> and its
    #    K-vs-FB by hand is team_k_pct_fb_vs_<hand>_<other>.
    def _platoon(side: str) -> pd.Series:
        other = "away" if side == "home" else "home"
        lsh = _num(f"opp_lefty_share_{other}")
        rsh = 1.0 - lsh
        lg_l = _num("league_k_pct_fb_vs_l")
        lg_r = _num("league_k_pct_fb_vs_r")
        sp_c = (lsh * (_num(f"sp_k_pct_fb_vs_l_{side}") - lg_l)
                + rsh * (_num(f"sp_k_pct_fb_vs_r_{side}") - lg_r))
        opp_c = (lsh * (_num(f"team_k_pct_fb_vs_l_{other}") - lg_l)
                 + rsh * (_num(f"team_k_pct_fb_vs_r_{other}") - lg_r))
        return sp_c * opp_c * _num(f"sp_usage_cat_fastball_{side}")

    # Per-side twins served under FINAL names — the exact scratch values the
    # diffs were computed from (home − away == diff by construction). The
    # internal _exp2_side_* aliases remain until the cleanup below so the
    # frozen arithmetic is untouched.
    df["exp2_centered_k_home"] = df["_exp2_side_home_centered_k"]
    df["exp2_centered_k_away"] = df["_exp2_side_away_centered_k"]
    for cat in _EXP2_CATEGORIES:
        df[f"exp2_cat_k_{cat}_home"] = df[f"_exp2_side_home_cat_k_{cat}"]
        df[f"exp2_cat_k_{cat}_away"] = df[f"_exp2_side_away_cat_k_{cat}"]
        df[f"exp2_cat_xwoba_{cat}_home"] = df[f"_exp2_side_home_cat_xwoba_{cat}"]
        df[f"exp2_cat_xwoba_{cat}_away"] = df[f"_exp2_side_away_cat_xwoba_{cat}"]
    df["exp2_cat_platoon_k_fastball_home"] = _platoon("home")
    df["exp2_cat_platoon_k_fastball_away"] = _platoon("away")

    df["exp2_cat_platoon_k_fastball_diff"] = (_platoon("home")
                                              - _platoon("away"))

    df.drop(columns=[c for c in df.columns
                     if c.startswith("_exp2_side_")], inplace=True)
    _exp2_all = [*EXP2_CANDIDATE_COLS, *EXP2_TWIN_COLS]
    cov = {c: round(float(df[c].notna().mean()), 3) for c in _exp2_all}
    logger.info("add_exp2_features: 8 candidates + 16 twins computed, "
                "coverage %s", cov)
    return df


# ── Lineup-delta features (Phase 2, moneyline-only) ── RETIRED 2026-09-26 ────
# No longer on the daily run path: the daily orchestrator (now
# master_pipeline.py, ex-pipeline.py) stopped calling
# add_lineup_delta_features, and build_batter_woba.py + its two caches were
# deleted. Retained as the documented shape of a correct re-implementation
# (one built from PROJECTED lineups on both sides, not post-game actuals).
# mean wOBA of tonight's ACTUAL starting 9 (and its top-3) minus the team's
# season-to-date wOBA, per side. The model otherwise only sees season-average
# lineup stats, so resting-star days are invisible. Sources:
#   data_delivery/lineups.parquet      StatsAPI battingOrder per game (4,451/4,451)
#   data_delivery/batter_woba.parquet  point-in-time batter sd-wOBA (prior games only)
#   data_delivery/team_woba.parquet    point-in-time team sd-wOBA + top-3/top-5 regulars
# Moneyline-only: these columns are excluded from MONEYLINE_FEATURE_COLS (the
# run engine serves that list verbatim), so they stay out of the run view.
LINEUP_DELTA_COLS: list[str] = [
    "lineup_actual_woba_delta_home", "lineup_actual_woba_delta_away",
    "lineup_actual_top3_delta_home", "lineup_actual_top3_delta_away",
    "lineup_rest_count_home", "lineup_rest_count_away",
]
LINEUP_MIN_PA = 20   # batters below this use the team season mean (never a 3-PA wOBA swing)
LINEUP_REST_PA = 50  # "regular" floor for the top-5 rest-count
LINEUP_TOP5_K = 5

# RETIRED 2026-09-26. The six lineup_actual_* / lineup_rest_count_* columns
# left MONEYLINE_FEATURE_COLS on 2026-08-29 (training.py) as a train-serve
# skew fix — populated from post-game ACTUAL lineups in the decided frame but
# always NULL at bet time — and the daily orchestrator (now
# master_pipeline.py) stopped calling this module's enrichment the same day. build_batter_woba.py and its two caches are
# deleted: that builder read pbp_chunks/, whose producer (fetch_pbp_chunks.py)
# was removed in ff372c3, so it could never again produce current data.
# Nothing on the daily run reads any of it. lineups.parquet and
# backfill_lineups.py survive for a future correct re-implementation and stay
# exact-name protected from the Phase 6 cleanup.
_LINEUP_REQUIRED_FILES = ("lineups.parquet",)
# Tripwire threshold: how far the caches may trail the decided frame before
# the loader warns. The caches are rebuilt only by the standalone builders, so
# without this a silent freeze reads as "the feature has no signal" instead of
# "the feature has no data". Offseason legitimately runs ~4-5 months ahead of
# the last decided game, so the bar is a month, not a day.
LINEUP_CACHE_MAX_LAG_DAYS = 31
_lineup_cache: dict = {}


def _lineup_base_dir():
    from pathlib import Path as _P
    return _P(__file__).resolve().parent.parent / "data_delivery"


def il_stints_dir():
    """Directory holding the IL/availability ledger caches.

    ``MLB_IL_STINTS_DIR`` wins when set — the daily pipeline points it at
    the run cache (beside pitches.parquet, outside the git repo) so the
    ledgers are runtime inputs that never ship as artifacts. Without the
    env var the legacy in-repo data_delivery location is used so older
    checkouts and existing tests keep working.
    """
    env = os.environ.get("MLB_IL_STINTS_DIR", "").strip()
    if env:
        return Path(env)
    return _lineup_base_dir()


def _missing_lineup_artifacts() -> list[str]:
    """Names of the required lineup/wOBA artifacts absent from data_delivery."""
    base = _lineup_base_dir()
    return [n for n in _LINEUP_REQUIRED_FILES if not (base / n).exists()]


def lineup_cache_staleness(max_date: "pd.Timestamp | None" = None) -> dict[str, Any]:
    """How far the lineup/wOBA caches trail the decided frame, in days.

    The 2026-08-29 removal note blamed train-serve skew for the lineup-delta
    features looking weak, but the real reason they looked weak for months was
    simpler: ``lineups.parquet`` froze at 2026-08-24 and ``batter_woba`` /
    ``team_woba`` followed it, so every 2024 game and all 316 September 2026
    games silently shipped NULL. Nothing failed, nothing warned, and the
    feature was simply absent for 40% of history — which then made an A/B of
    it read as noise (or worse, as a coverage artifact).

    These caches are only rebuilt by the standalone builders
    (backfill_lineups.py / build_pbp_chunks.py / build_batter_woba.py), so
    nothing in the daily run notices them falling behind. This check is the
    missing tripwire: compare each cache's max game_date against the decided
    frame's and report the lag. Returns {} when the caches are absent or
    unreadable (the missing-file path already reports that separately).

    ``max_date`` defaults to the decided frame on disk, so callers that
    already hold the frame can pass its max and skip the CSV re-read.
    """
    base = _lineup_base_dir()
    if any(not (base / n).exists() for n in _LINEUP_REQUIRED_FILES):
        return {}
    if max_date is None:
        csv_path = base / "game_level_features.csv"
        if not csv_path.exists():
            return {}
        try:
            max_date = pd.to_datetime(
                pd.read_csv(csv_path, usecols=["game_date"])["game_date"],
                errors="coerce").max()
        except Exception:
            return {}
    if pd.isna(max_date):
        return {}
    ref = pd.Timestamp(max_date).normalize()

    out: dict[str, Any] = {"reference_max_game_date": str(ref.date())}
    worst = 0
    for name, key in (("lineups.parquet", "lineups"),
                      ("batter_woba.parquet", "batter"),
                      ("team_woba.parquet", "team")):
        try:
            cache_max = pd.to_datetime(
                pd.read_parquet(base / name, columns=["game_date"])["game_date"],
                errors="coerce").max()
        except Exception:
            out[key] = None
            continue
        if pd.isna(cache_max):
            out[key] = None
            continue
        lag = int((ref - pd.Timestamp(cache_max).normalize()).days)
        out[key] = {"max_game_date": str(pd.Timestamp(cache_max).date()),
                    "lag_days": lag}
        worst = max(worst, lag)
    out["max_lag_days"] = worst
    return out


def _load_lineup_cache() -> dict:
    """Lazy-load the three lineup/wOBA artifacts (cached across calls).

    On missing artifacts, degrades to {} WITH a warning naming the files —
    the caller decides whether that is acceptable (require_caches=True on the
    training path turns the same condition into a loud error instead).

    Also warns when the caches are merely STALE: present, loadable, and
    quietly short of the decided frame. See lineup_cache_staleness()."""
    if _lineup_cache:
        return _lineup_cache
    base = _lineup_base_dir()
    missing = [n for n in _LINEUP_REQUIRED_FILES if not (base / n).exists()]
    if missing:
        logger.warning(
            "add_lineup_delta_features: lineup artifacts unavailable (%s); "
            "columns stay NaN — the shipped lineup-delta feature is DEAD on "
            "this run (training path should pass require_caches=True to fail "
            "loud instead)", ", ".join(missing))
        _lineup_cache.clear()
        return {}
    try:
        _lineup_cache["lineups"] = pd.read_parquet(base / "lineups.parquet")
        _lineup_cache["batter"] = pd.read_parquet(base / "batter_woba.parquet")
        _lineup_cache["team"] = pd.read_parquet(base / "team_woba.parquet")
    except Exception as e:  # present but unreadable: degrade to NaN
        logger.warning(
            "add_lineup_delta_features: lineup artifacts failed to load (%s); "
            "columns stay NaN", e)
        _lineup_cache.clear()
        return {}
    try:
        stale = lineup_cache_staleness()
    except Exception as e:  # the tripwire must never break feature building
        logger.debug("lineup staleness check failed (ignored): %s", e)
        stale = {}
    if stale and stale.get("max_lag_days", 0) > LINEUP_CACHE_MAX_LAG_DAYS:
        detail = ", ".join(
            f"{k}={v['max_game_date']} (lag {v['lag_days']}d)"
            for k, v in stale.items()
            if isinstance(v, dict))
        logger.warning(
            "add_lineup_delta_features: lineup caches are %d days STALE "
            "vs the decided frame (%s) — every game after those dates ships "
            "NULL lineup-delta features. Re-run backfill_lineups.py.",
            stale["max_lag_days"], detail)
    return _lineup_cache


def add_lineup_delta_features(game_df: pd.DataFrame,
                              inplace: bool = False,
                              lineups_override: pd.DataFrame | None = None,
                              require_caches: bool = False) -> pd.DataFrame:
    """Add the 6 lineup-delta columns (actual starting-9 wOBA vs team season).

    Per side (home/away), per game:
      lineup_actual_woba_delta = mean(actual 9's sd-wOBA) − team season sd-wOBA
      lineup_actual_top3_delta = mean(top-3 of the actual 9) − team top-3 regulars
      lineup_rest_count      = # of the team's top-5 wOBA regulars not in the
                               actual 9 (resting-star days)

    All inputs are point-in-time (batter/team wOBA through games strictly
    before the game date — no lookahead, season-partitioned). Batters below
    LINEUP_MIN_PA use the team season mean, so a rookie's 3-PA wOBA never
    swings the feature; when the team baseline is itself missing the cell is
    NaN and the existing imputation path handles it. Games without a lineup
    row (no battingOrder in the backfill) get NaN — never fabricated.

    Idempotent: existing columns are left untouched (SQL/other-shipped values
    are authoritative). Returns the frame (same object when inplace=True).

    ``require_caches=True`` (the TRAINING path) fails LOUD instead of
    degrading to NaN: (1) FileNotFoundError naming the missing artifact(s)
    when lineups/batter_woba/team_woba parquet are absent; (2) all-NaN
    placeholder columns shipped by a prior broken run are recomputed and
    overwritten (real values stay authoritative); (3) RuntimeError after
    enrichment if any of the 6 columns is still absent/all-NaN — stale
    caches or an empty game_pk join must never train the feature dead.
    The slate path (require_caches=False) keeps the graceful NaN fallback:
    projected-only mornings are by design.
    """
    df = game_df if inplace else game_df.copy()
    missing = [c for c in LINEUP_DELTA_COLS if c not in df.columns]
    if require_caches:
        # Dead-placeholder guard (the v26 "computed 0 column(s)" incident):
        # a prior run that shipped the 6 columns as ALL-NaN (the rebind bug
        # re-saved them into game_level_features.csv) must NOT be treated as
        # already enriched — recompute and overwrite them on the training
        # path. Real values stay authoritative (idempotence preserved).
        dead = [c for c in LINEUP_DELTA_COLS
                if c in df.columns and bool(df[c].isna().all())]
        missing = sorted(set(missing) | set(dead))
    if not missing:
        return df
    for c in missing:
        df[c] = np.nan
    if not {"game_pk", "game_date", "home_team", "away_team"}.issubset(df.columns):
        logger.warning("add_lineup_delta_features: frame lacks game_pk/game_date/teams; columns stay NaN")
        return df
    if require_caches:
        # NOTE: use a SEPARATE name here — rebinding `missing` to the file-
        # absence list used to blank the assignment guards below, so the
        # columns were computed but never written (the v26 "computed 0").
        absent_files = _missing_lineup_artifacts()
        if absent_files:
            raise FileNotFoundError(
                "lineup-delta feature: REQUIRED committed runtime inputs "
                "missing from data_delivery: " + ", ".join(absent_files) +
                ". Without them the 6 lineup_actual_* columns stay NaN and the "
                "shipped moneyline feature trains DEAD. Restore these files "
                "(they are protected from Phase 6 cleanup) or rebuild them via "
                "backfill_lineups.py before training.")
    caches = _load_lineup_cache()
    if not caches:
        if require_caches:
            raise FileNotFoundError(
                "lineup-delta feature: cache load failed (see warnings above); "
                "refusing to train without the shipped lineup-delta inputs")
        return df
    batters, teams = caches["batter"].copy(), caches["team"].copy()
    lineups = (lineups_override if lineups_override is not None
               else caches["lineups"])
    if lineups is not None and "game_pk" in lineups.columns:
        # dtype-safe by construction: an override keyed by a mixed int/NA
        # column lands as object dtype, and pandas refuses an object-vs-Int64
        # key merge with a hard ValueError (the 2026-09-22 slate crash).
        # Coerce once here so ANY caller's override joins cleanly.
        lineups = lineups.copy()
        lineups["game_pk"] = pd.to_numeric(
            lineups["game_pk"], errors="coerce").astype("Int64")

    date = pd.to_datetime(df["game_date"])
    df["game_date"] = date
    df["_season"] = date.dt.year
    df["_gpk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    batters = batters.rename(columns={"season": "_season"})
    teams = teams.rename(columns={"season": "_season"})
    # Unify the join key name: every merge below uses the typed _gpk column
    # (game_pk-vs-_gpk merge keys are the exact 2026-09-22 crash surface).
    lineups = lineups.rename(columns={"game_pk": "_gpk"})
    batters["batter"] = pd.to_numeric(batters["batter"], errors="coerce").astype("Int64")

    for side, team_col, order_col, woba_col, top3_col, rest_col in (
        ("home", "home_team", "home_order", "lineup_actual_woba_delta_home",
         "lineup_actual_top3_delta_home", "lineup_rest_count_home"),
        ("away", "away_team", "away_order", "lineup_actual_woba_delta_away",
         "lineup_actual_top3_delta_away", "lineup_rest_count_away"),
    ):
        # game-level: team sd-wOBA + top-3/top-5 baselines as of the date
        g = df[["_gpk", "_season", "game_date", team_col]].rename(
            columns={team_col: "team"})
        g = g.merge(teams, on=["_season", "game_date", "team"], how="left")
        g = g.merge(lineups[["_gpk", order_col]].rename(columns={order_col: "order"}),
                    on="_gpk", how="left")

        # explode to batter level; join point-in-time batter wOBA
        exp = g[["_gpk", "_season", "game_date", "team", "sd_woba", "top3_woba",
                 "top5_ids", "order"]].explode("order")
        exp["batter"] = pd.to_numeric(exp["order"], errors="coerce").astype("Int64")
        exp = exp.merge(
            batters[["_season", "game_date", "batter", "sd_woba", "prior_pa"]],
            on=["_season", "game_date", "batter"], how="left",
            suffixes=("", "_b"))
        # min-PA rule: effective wOBA = batter sd-wOBA, or the team season mean
        eff = exp["sd_woba_b"].where(
            exp["sd_woba_b"].notna() & (exp["prior_pa"] >= LINEUP_MIN_PA),
            exp["sd_woba"])
        exp["_eff"] = eff

        # per-game aggregates
        agg = exp.groupby("_gpk", as_index=False).agg(
            actual_mean=("_eff", "mean"),
            top3_mean=("_eff", lambda s: s.nlargest(3).mean()),
            n_in_lineup=("batter", "count"),
        )
        g2 = g.merge(agg, on="_gpk", how="left")
        # rest count: team's top-5 regulars not in tonight's 9
        lineup_sets = (exp.groupby("_gpk")["batter"]
                       .apply(lambda s: set(s.dropna().astype(int))))
        rest = {}
        for _, row in g2.iterrows():
            gpk = row["_gpk"]
            ids = row["top5_ids"]
            if not isinstance(ids, str) or not ids.strip():
                rest[gpk] = 0 if pd.notna(row["top5_ids"]) else np.nan
                continue
            try:
                top5 = json.loads(ids)
            except Exception:
                rest[gpk] = np.nan
                continue
            cur = lineup_sets.get(gpk, set())
            rest[gpk] = sum(1 for b in top5 if b not in cur)
        g2["_rest"] = g2["_gpk"].map(rest)

        if woba_col in missing:
            df[woba_col] = g2["actual_mean"] - g2["sd_woba"]
        if top3_col in missing:
            df[top3_col] = g2["top3_mean"] - g2["top3_woba"]
        if rest_col in missing:
            df[rest_col] = g2["_rest"]

    df = df.drop(columns=["_season", "_gpk"])
    if require_caches:
        # Zero-column sentinel (the missing guard): if ANY of the 6 shipped
        # columns is absent or ALL-NaN after enrichment, the caches loaded
        # but produced nothing (stale artifacts / empty game_pk join). Fail
        # LOUD — never a quiet "computed 0" log on the training path.
        dead = [c for c in LINEUP_DELTA_COLS
                if c not in df.columns or bool(df[c].isna().all())]
        if dead:
            raise RuntimeError(
                "lineup-delta feature: TRAINING-path enrichment produced NO "
                "live values for " + ", ".join(dead) +
                " even though the committed caches loaded (lineups.parquet, "
                "batter_woba.parquet, team_woba.parquet) — the artifacts are "
                "stale or the game_pk join is empty. Refusing to train with "
                "the shipped moneyline feature DEAD; refresh/rebuild the "
                "caches before training.")
    logger.info("add_lineup_delta_features: computed %d column(s) for %d games",
                len(missing), len(df))
    return df


def add_diff_features(
    game_df: pd.DataFrame,
    weather_data: dict | None = None,
    require_records: bool = False,
) -> pd.DataFrame:
    """Compute all 56 model features from the raw home/away columns.

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
        14. lineup_re24_mean_diff
        15. lineup_re24_top3_diff
        16. lineup_re24_std_diff
        16b. lineup_il_flag_diff  home_lineup_il_flag − away_lineup_il_flag
                               (OUT/IR availability signal, one side's
                               projected nine missing a player)
        16c. pl_<pos>_xwoba_diff ×9  home pool − away pool for each of the
                               9 position pools (c/fb/sb/ss/tb/rf/cf/lf/dh)
                               — the position-pool xwOBA family; the level
                               halves are the game_level pl_*_{home,away}
                               columns, so home − away == diff by
                               construction. Served only by an explicit
                               arm (computed, not in the default universe).
        17. woba_30g_diff      home_woba_30g − away_woba_30g
        18. bullpen_whip_10g_diff  home_bullpen_whip_10g − away_bullpen_whip_10g
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
        29. park_factor_slug_diff  home_park_slug_factor × lineup_re24_top3_diff
        30. wind_advantage_flyball_factor
                               wind_direction_multiplier (Out=1, In=-1, Dome=0)
                               × sp_era_diff
        31. air_density_velocity_boost  stadium_air_density × sp_fbvelo_diff
        32. bullpen_meltdown_risk_diff  bullpen_pitches_diff × bullpen_whip_10g_diff
        33. pitcher_regression_indicator_diff
                                        sp_fbvelo_diff × sp_era_5g_diff
        34. lineup_depth_multiplier_diff
                                        lineup_re24_mean_diff × lineup_re24_top3_diff
        35. ace_efficiency_factor_diff   sp_k9_5g_diff × sp_whiff_diff

    56 model features in total: the numbered run 1-35 plus 16b, plus the
    raw per-side twins documented below (12 level + 6 interaction; the 2
    travel twins are ingest-layer frame columns that are only coerced
    here). On a real frame 42 of those are NEW columns — the 14 level and
    travel twins arrive as this function's own diff inputs and are
    coerced in place — while frames lacking the raw inputs see all 56
    created (the missing twins ship NULL, the same pattern as their
    diffs).

    All diff features follow the convention: home − away (positive = home
    advantage).  Interaction features (29–35) are built from the diff
    features, so the model sees relative strengths directly.

    Raw per-side twins (2026-09-27 structural expansion): every served diff
    family above also exposes its raw home/away halves for the tree view.
    The level families' twins ARE this function's own diff inputs (the same
    strictly-prior source, so home − away == diff holds by construction and
    a twin can never drift from its gap); the interaction features 33–35
    ship the within-side product of their factors per side. Twins route
    tree-only: the logistic member keeps its diffs-only slice and the run
    engine's λ view carries levels but not matchup composites.

    Args:
        game_df: DataFrame with raw home/away columns.
        weather_data: Optional dict keyed by game_id → weather dict with
            air_density and wind_multiplier (from weather.fetch_day_weather).
            When provided, features 30–31 use real weather data.

    Returns a copy of game_df with the new columns appended.
    """
    df = game_df.copy()
    n = len(df)
    # Snapshot the input columns so both log lines report the MEASURED
    # number of columns this function actually creates. These used to be
    # hardcoded 35, which went stale the moment lineup_il_flag_diff was
    # added: the function created 36 while every log line and the
    # docstring below still claimed 35. A self-reported count cannot rot.
    _cols_before = set(df.columns)
    logger.info("Computing diff features for %d games...", n)

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
    elif require_records:
        df["win_pct_diff"] = np.nan
        logger.warning(
            "win_pct_diff: record/win-pct columns missing on FINAL computation "
            "— feature ships as NaN; verify official-results overlay ran")
    else:
        df["win_pct_diff"] = np.nan
        logger.debug(
            "win_pct_diff: record/win-pct columns not present yet at this stage "
            "— expected pre-overlay; diffs are recomputed after official results")

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
        ("lineup_re24_mean_diff", "lineup_re24_mean_home", "lineup_re24_mean_away"),  # 14
        ("lineup_re24_top3_diff", "lineup_re24_top3_home", "lineup_re24_top3_away"),  # 15
        ("lineup_re24_std_diff", "lineup_re24_std_home", "lineup_re24_std_away"),     # 16
        # 16c. Position-pool xwOBA diffs (pl_* family, 2026-10-02 A/B).
        # Frames lacking the pl_* levels see all 9 created as NULL — the
        # exact _diff contract (never a fabricated 0).
        ("pl_c_xwoba_diff", "pl_c_xwoba_home", "pl_c_xwoba_away"),
        ("pl_fb_xwoba_diff", "pl_fb_xwoba_home", "pl_fb_xwoba_away"),
        ("pl_sb_xwoba_diff", "pl_sb_xwoba_home", "pl_sb_xwoba_away"),
        ("pl_ss_xwoba_diff", "pl_ss_xwoba_home", "pl_ss_xwoba_away"),
        ("pl_tb_xwoba_diff", "pl_tb_xwoba_home", "pl_tb_xwoba_away"),
        ("pl_rf_xwoba_diff", "pl_rf_xwoba_home", "pl_rf_xwoba_away"),
        ("pl_cf_xwoba_diff", "pl_cf_xwoba_home", "pl_cf_xwoba_away"),
        ("pl_lf_xwoba_diff", "pl_lf_xwoba_home", "pl_lf_xwoba_away"),
        ("pl_dh_xwoba_diff", "pl_dh_xwoba_home", "pl_dh_xwoba_away"),
        ("lineup_il_flag_diff", "lineup_il_flag_home", "lineup_il_flag_away"),        # 16b availability
        ("woba_30g_diff", "woba_30g_home", "woba_30g_away"),                     # 17
        ("bullpen_whip_10g_diff", "bullpen_whip_10g_home", "bullpen_whip_10g_away"),  # 18 (RENAMED 2026-09-30)
        ("bullpen_whip_3g_diff", "bullpen_whip_3g_home", "bullpen_whip_3g_away"),     # 19
        ("bullpen_pitches_diff", "bullpen_pitches_3d_home", "bullpen_pitches_3d_away"),  # 20
        ("bullpen_ip_diff", "bullpen_ip_3d_home", "bullpen_ip_3d_away"),         # 21
        ("team_barrel_diff", "team_barrel_15g_home", "team_barrel_15g_away"),    # 22
        ("team_hardhit_diff", "team_hardhit_15g_home", "team_hardhit_15g_away"), # 23
        ("team_exitvelo_diff", "team_exitvelo_15g_home", "team_exitvelo_15g_away"),  # 24
    ]
    for out, h_col, a_col in simple_diffs:
        _diff(out, h_col, a_col)

    # ── Raw per-side level twins (tree view) ───────────────────────────────
    # These ARE the diff inputs above — the same strictly-prior source, so
    # home − away == diff by construction and no second derivation exists
    # to drift. Coerced to float (or created NULL when a side's observation
    # is absent, the identical NULL pattern the diff inherits). Travel
    # twins are assembled by the ingest layer, not here; they ride the same
    # coercion so the serving matrix sees one dtype contract.
    for h_col, a_col in (
        ("rest_days_home", "rest_days_away"),
        ("sp_era_5g_home", "sp_era_5g_away"),
        ("sp_fbvelo_3g_home", "sp_fbvelo_3g_away"),
        ("lineup_re24_std_home", "lineup_re24_std_away"),
        ("bullpen_pitches_3d_home", "bullpen_pitches_3d_away"),
        ("team_hardhit_15g_home", "team_hardhit_15g_away"),
        ("time_zones_crossed_last_3d_home", "time_zones_crossed_last_3d_away"),
    ):
        for col in (h_col, a_col):
            res = _resolve_col(col)
            if res is not None:
                df[res] = pd.to_numeric(df[res], errors="coerce")
                if res != col:
                    df[col] = df[res]
            else:
                df[col] = np.nan

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

    # ── 29. park_factor_slug_diff: home_park_slug_factor × lineup_re24_top3_diff
    # Maps out when a power-heavy lineup gets to exploit a small ballpark.
    pf_raw = home_team.map(PARK_FACTORS_SLG).astype(float)  # NaN = unknown park
    pf = (pf_raw - 100.0) / 100.0  # center at 0: +0.05 = 5% more SLG than avg
    df["park_factor_slug_diff"] = pf * pd.to_numeric(
        df["lineup_re24_top3_diff"], errors="coerce")

    # Capture observed weather LEVELS before recomputing interactions. Saved
    # frames / RFE must not erase valid weather or freeze products of stale SP
    # diffs. Only level-backed observations can be reconstructed safely.
    _wind_level = (pd.to_numeric(df["park_wind_factor"], errors="coerce")
                   if "park_wind_factor" in df else pd.Series(np.nan, index=df.index))
    _air_level = (pd.to_numeric(df["air_density_level"], errors="coerce")
                  if "air_density_level" in df else pd.Series(np.nan, index=df.index))

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
            # Fetchers key results by game_pk (falling back to game_id);
            # accept raw/string/int variants of either so a direct
            # weather_data= call resolves identically to the production
            # apply_weather_features path.
            candidates = []
            for v in (row.get("game_pk"), row.get("game_id")):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                candidates.extend([v, str(v)])
                s = str(v)
                if s.isdigit():
                    candidates.append(int(s))
            w = {}
            if isinstance(weather_data, dict):
                for c in candidates:
                    if c in weather_data:
                        w = weather_data[c] or {}
                        break
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
        # This pass writes levels too, so repeated calls share the same
        # representation as weather.apply_weather_features.
        df["park_wind_factor"] = wm.combine_first(_wind_level)
        df["air_density_level"] = ad.combine_first(_air_level)
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

    # Recompute from current factors, preserving absent-from-fetch observed
    # levels. Air density is not neutral merely because a roof is closed.
    _wm = pd.to_numeric(df.get("park_wind_factor", _wind_level), errors="coerce")
    _ad = pd.to_numeric(df.get("air_density_level", _air_level), errors="coerce")
    df["wind_advantage_flyball_factor"] = (
        _wm * pd.to_numeric(df["sp_era_diff"], errors="coerce")
    ).combine_first(df["wind_advantage_flyball_factor"])
    df["air_density_velocity_boost"] = (
        (_ad - SEA_LEVEL_RHO) * pd.to_numeric(df["sp_fbvelo_diff"], errors="coerce")
    )
    _dome_col = "dome_is_neutral_game" if "dome_is_neutral_game" in df else "dome_is_neutral"
    _closed = pd.to_numeric(df[_dome_col], errors="coerce").eq(1)
    # Preserve the production indoor-neutral policy, but never claim density
    # is observed when the input level is absent. Coverage labels these zeros.
    df.loc[_closed & df["air_density_velocity_boost"].notna(),
           "air_density_velocity_boost"] = 0.0
    df.loc[_closed & df["sp_era_diff"].notna(),
           "wind_advantage_flyball_factor"] = 0.0

    # ── 32. bullpen_meltdown_risk_diff (RENAMED 2026-09-30): the family's
    # cross-side form, bullpen_pitches_diff × bullpen_whip_10g_diff.
    # Overworked + low quality bullpen = elevated meltdown risk.
    df["bullpen_meltdown_risk_diff"] = (
        df["bullpen_pitches_diff"] * df["bullpen_whip_10g_diff"])

    # Interaction twins: each side's OWN product of the interaction's
    # factors (the within-side form the cross-side gap only summarizes).
    # home − away of a twin pair equals the diff only up to cross terms, so
    # the pair is served as the interaction's raw representation rather
    # than as its halves. RENAMED 2026-09-27: every model-side feature ends
    # in _diff; the old bare names are retired everywhere in the same
    # commit (universe, routing, metadata, adopted RFE state).
    def _twin(out: str, f1: str, f2: str) -> None:
        """Within-side product of the interaction's own factor columns.

        Missing factor columns (thin frame) ship NULL — the same pattern
        as the diffs, never a fabricated 0. Aliases resolve like _diff."""
        c1 = _resolve_col(f1)
        c2 = _resolve_col(f2)
        if c1 is None or c2 is None:
            df[out] = np.nan
        else:
            df[out] = (pd.to_numeric(df[c1], errors="coerce")
                       * pd.to_numeric(df[c2], errors="coerce"))

    # ── 33. pitcher_regression_indicator_diff: sp_fbvelo_diff × sp_era_5g_diff
    # Physical velocity drop vs shrunk recent runs/9 results — flags
    # regression candidates before the recent results fully catch up to stuff.
    df["pitcher_regression_indicator_diff"] = (
        df["sp_fbvelo_diff"] * df["sp_era_5g_diff"])
    _twin("pitcher_regression_indicator_home", "sp_fbvelo_3g_home", "sp_era_5g_home")
    _twin("pitcher_regression_indicator_away", "sp_fbvelo_3g_away", "sp_era_5g_away")

    # ── 34. lineup_depth_multiplier_diff:
    #        lineup_re24_mean_diff × lineup_re24_top3_diff
    # Star power vs complete batting order depth.
    df["lineup_depth_multiplier_diff"] = (
        df["lineup_re24_mean_diff"] * df["lineup_re24_top3_diff"])
    _twin("lineup_depth_multiplier_home", "lineup_re24_mean_home", "lineup_re24_top3_home")
    _twin("lineup_depth_multiplier_away", "lineup_re24_mean_away", "lineup_re24_top3_away")
    # Bullpen meltdown per-side twins (2026-09-30): the within-side product
    # of the family's own factors — 3-day pitch count x 10-game WHIP.
    _twin("bullpen_meltdown_risk_home", "bullpen_pitches_3d_home",
          "bullpen_whip_10g_home")
    _twin("bullpen_meltdown_risk_away", "bullpen_pitches_3d_away",
          "bullpen_whip_10g_away")

    # ── 35. ace_efficiency_factor_diff: sp_k9_5g_diff × sp_whiff_diff
    # Last-5-start strikeout volume driven by raw swing-and-miss stuff —
    # the true-ace differentiator.
    df["ace_efficiency_factor_diff"] = (
        df["sp_k9_5g_diff"] * df["sp_whiff_diff"])
    _twin("ace_efficiency_factor_home", "sp_k9_5g_home", "sp_whiff_3g_home")
    _twin("ace_efficiency_factor_away", "sp_k9_5g_away", "sp_whiff_3g_away")

    _added = len(set(df.columns) - _cols_before)
    logger.info("Diff features complete: %d columns added", _added)
    return df

# ── Public API ──────────────────────────────────────────────────────────────

def _mem_mb() -> float:
    """Current resident set size in MB — NOT the process peak.

    ru_maxrss is a high-water mark: once the run touches it, every later
    reading is identical, so all six [MEM] lines reported the same
    number (2026-10-05 log review: 5955 MB six times — flat and
    meaningless). /proc/self/statm's second field is the LIVE resident
    page count, so each checkpoint reads what the run actually holds
    right then. Non-Linux platforms fall back to the peak (at least
    monotone-honest).
    """
    try:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        pass
    if os.name == "nt":
        # Windows has no /proc or resource; query the LIVE working set.
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize / 1e6)
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def build_features(
    pitches_path: str | Path,
    output_dir: str | Path = ".",
    validate: bool = True,
    corrected_outs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run pure-DuckDB feature engineering on a pitches Parquet file.

    Reads pitches.parquet, builds game_df + pbp_df via SQL, writes them to
    disk, and loads into pandas ONLY for model training.

    Args:
        pitches_path: Path to pitches.parquet (from ingestion.py).
        output_dir:   Where to write output Parquet files.
        validate:     Enforce observed source-schema and unique game contracts.
        corrected_outs: EXPERIMENTAL (pitcher-stats ablation only). When
            True, pa_boundary credits force_out / fielders_choice /
            batter_interference outs and keeps intent_walk / truncated_pa
            PAs as boundaries (see _build_game_level). Default False = the
            byte-identical production path.

    Returns:
        (game_df, pbp_df) as pandas DataFrames.

        ``game_df`` does NOT carry the ``*_diff`` columns. Callers MUST
        call ``add_diff_features()`` on it themselves, after their own
        inputs are final — master_pipeline attaches Elo/season records
        first, so deriving diffs before that point would ship NaN
        elo/win_pct/woba_30g diffs.
    """
    pitches_path = Path(pitches_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    game_out = out_dir / "game_level_features.parquet"
    pbp_out = out_dir / "pbp_level_features.parquet"

    if validate:
        import pyarrow.parquet as pq
        source_cols = set(pq.read_schema(pitches_path).names)
        required = {"game_pk", "game_date", "at_bat_number", "pitch_number",
                    "home_team", "away_team", "post_home_score",
                    "post_away_score", "launch_speed_angle"}
        if not required <= source_cols:
            raise ValueError("Statcast source schema missing observed fields "
                             f"{sorted(required - source_cols)}; re-pull historical "
                             "pitches before building corrected features")

    logger.info("=== DuckDB Feature Engineering ===")
    logger.info("[MEM] Start: %.0f MB", _mem_mb())

    con = _connect(pitches_path)
    logger.info("[MEM] After DuckDB load: %.0f MB", _mem_mb())

    try:
        _build_game_level(con, corrected_outs=corrected_outs)
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
    if validate and game_df["game_pk"].duplicated().any():
        raise ValueError("Feature engineering fanned game_pk identities; refusing training")

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

    # Diff features are NOT computed here. They are derived from the raw
    # home/away columns, and master_pipeline re-runs add_diff_features
    # AFTER enrich_elo_and_records() attaches Elo/season records (spec
    # features 2, 3, 17) — so computing them here too built all 35 columns
    # twice per run, discarded them once, and logged "Computing 35 diff
    # features" twice back to back with no way to tell which pass was
    # authoritative. Each caller now derives diffs exactly once, after its
    # own raw inputs are final. Callers MUST call add_diff_features()
    # themselves on the returned game frame.

    logger.info("=== Complete: %d games, %d pitches ===", len(game_df), len(pbp_df))

    return game_df, pbp_df
