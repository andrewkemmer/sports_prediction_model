"""DuckDB feature-engineering engine for the NFL backend — mirrors
``mlb-backend/backend/features.py::_connect``'s low-RAM pattern.

MLB keeps the giant raw Statcast table in DuckDB and computes the heavy
aggregations in SQL with a tight ``memory_limit`` + a generous disk ``temp_directory``
(so DuckDB spills to disk instead of holding the whole table in RAM), narrows /
drops unused columns, and exports SMALL frames that pandas then diffs.

The NFL analogue of "the heavy raw-table step" is the play-by-play rollup
(``nfl_features._pbp_team_agg`` over the 2019-2025 PBP). This module owns that
aggregation in DuckDB with the same spill settings, produces a frame that is
byte-equivalent to the pandas rollup (answer-key tested), and exposes an
additive, leak-safe ``pbp_aggregate`` seam so new PBP-backed features route
through SQL instead of growing pandas memory.

Ownership split (matches MLB): DuckDB owns the raw PBP + heavy SQL
aggregations; pandas still computes ELO / ``team_events`` / ``team_stats_ladder``
and the ``*_diff`` features on the small (~1,960-row) game frame. This module
deliberately does NOT port that sequential logic to SQL — DuckDB buys nothing
there (row-sequential rolling state) and it would hurt clarity.

Leak-safety contract for the aggregate seam: every registered aggregate is a
function of that GAME's plays only (a sum, count or mean over ``(game_id,
posteam)``). Downstream ``team_stats_ladder`` shifts these to strictly-prior
games, so surfacing anything other than a per-game aggregate would leak future
play. The registry asserts this contract structurally (suffix guard) and a unit
test proves a future game's plays never reach a trailing value.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DuckDB connection tuning — mirrored verbatim from mlb-backend/backend/features.py::_connect.
DUCKDB_THREADS = 1
DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_TEMP_DIR = "/tmp/duckdb_temp"
DUCKDB_MAX_TEMP = "50GB"

# Output columns of the rollup — matches ``nfl_features._pbp_team_agg`` exactly.
TEAM_AGG_COLUMNS = [
    "game_id", "team", "total_yards", "n_plays",
    "epa_sum", "epa_n", "qb_epa_sum", "qb_epa_n", "elapsed_min",
]

# PBP columns the rollup understands (the ones _pbp_team_agg reads).
_PBP_GROUP = ("game_id", "posteam")
_PBP_SUM_COLS = ("yards_gained", "epa", "qb_epa")


def duckdb_available() -> bool:
    """True when the DuckDB module is importable (used for graceful fallback)."""
    try:
        import duckdb  # noqa: F401  (imported for the side effect only)
        return True
    except Exception:
        return False


@contextmanager
def duckdb_engine() -> Iterator["DuckDBPyConnection"]:
    """In-memory DuckDB tuned for low RAM, mirroring MLB features._connect.

    ``memory_limit`` + ``temp_directory`` (with a large ``max_temp_directory_size``)
    lets DuckDB spill intermediate results to disk instead of holding the full
    PBP table in RAM. ``threads=1`` + ``preserve_insertion_order=false`` match
    MLB's deterministic, RAM-conscious settings.
    """
    import duckdb
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{DUCKDB_TEMP_DIR}'")
    con.execute(f"SET max_temp_directory_size = '{DUCKDB_MAX_TEMP}'")
    try:
        yield con
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Additive aggregate seam
# ---------------------------------------------------------------------------
@dataclass
class PBPAggregate:
    """A per-(game_id, posteam) SQL aggregate surfaced as an extra rollup column.

    ``expr`` is a DuckDB expression evaluated over the rows of one game+team
    (e.g. ``"SUM(epa*epa)"``). It MUST be a per-game aggregate (sum/count/mean)
    — never a raw per-play value or a window spanning other games — so the
    strictly-trailing shift in ``team_stats_ladder`` remains leak-safe.
    ``needs`` lists extra source PBP columns the expression requires (beyond the
    base yardage/EPA/QB-EPA/clock set); they are added to the registered table.
    """

    name: str
    expr: str
    result_col: str
    needs: tuple[str, ...] = field(default_factory=tuple)


_PBP_AGGREGATES: dict[str, PBPAggregate] = {}


# Leak-safety contract: only per-``(game_id, posteam)`` scalar aggregates may
# be registered. A raw expression (or a window spanning other games) would leak
# future play through the strictly-trailing shift in team_stats_ladder.
_AGG_FUNCTION_PREFIXES = ("SUM", "COUNT", "AVG", "MIN", "MAX",
                          "VARIANCE", "STDDEV", "MEDIAN", "CORR", "COVAR")


def _validate_leak_safe(expr: str) -> None:
    s = expr.lstrip()
    if not s.startswith(_AGG_FUNCTION_PREFIXES):
        raise ValueError(
            "pbp_aggregate expr must be a per-game scalar aggregate starting with "
            f"one of {_AGG_FUNCTION_PREFIXES}; got {expr!r} — a raw per-play "
            "expression would leak future play through the trailing shift")


def register_pbp_aggregate(agg: PBPAggregate) -> PBPAggregate:
    """Register a {PBPAggregate} so ``pbp_team_agg(..., extra_names=(name,))``
    appends its column. Re-registering the same name replaces it."""
    _validate_leak_safe(agg.expr)
    _PBP_AGGREGATES[agg.name] = agg
    return agg


def pbp_aggregate(name, expr, result_col, needs=()):
    """Convenience factory: build + register a {PBPAggregate}."""
    return register_pbp_aggregate(PBPAggregate(name, expr, result_col, tuple(needs)))


# ---------------------------------------------------------------------------
# DuckDB registration + rollup
# ---------------------------------------------------------------------------
def register_pbp(con, pbp: pd.DataFrame, table: str = "nfl_pbp") -> set[str]:
    """Register the (already-narrowed) PBP frame as ``table``.

    Only the columns the rollup / registered aggregates need are carried; the
    pandas frame passed in is already column-narrowed upstream, so DuckDB never
    sees the ~370-col raw PBP. Returns the set of columns present.
    """
    keep = list(dict.fromkeys(list(_PBP_GROUP) + list(_PBP_SUM_COLS)
                              + ["game_seconds_remaining"]))
    for agg in _PBP_AGGREGATES.values():
        for c in agg.needs:
            if c not in keep:
                keep.append(c)
    cols = [c for c in keep if c in pbp.columns]
    sub = pbp[cols]
    # Keep numeric types clean; object/None coercion handled by DuckDB.
    con.register(table, sub)
    return set(cols)


def _elapsed_expr(table: str, has_clock: bool) -> str:
    if not has_clock:
        return "NULL::DOUBLE"
    # Per game, elapsed minutes from the final play — mirrors pandas
    # (3600 - min(game_seconds_remaining)) / 60, attached to every team row.
    return (f"(SELECT (3600.0 - MIN(t.game_seconds_remaining)) / 60.0 "
            f"FROM {table} t WHERE t.game_id = g.game_id "
            f"AND t.game_seconds_remaining IS NOT NULL)")


def pbp_team_agg(con, pbp: pd.DataFrame,
                 extra_names: Sequence[str] = (),
                 table: str = "nfl_pbp") -> pd.DataFrame:
    """DuckDB rollup byte-equivalent to ``nfl_features._pbp_team_agg``.

    Groups the PBP by ``(game_id, posteam)`` and emits the same columns
    (``game_id, team, total_yards, n_plays, epa_sum/epa_n, qb_epa_sum/qb_epa_n,
    elapsed_min``) plus any registered additive aggregates named in
    ``extra_names``. Empty/malformed inputs return an empty frame with the full
    schema, exactly like the pandas function.
    """
    cols = list(TEAM_AGG_COLUMNS)
    cols.extend([_PBP_AGGREGATES[n].result_col
                 for n in extra_names if n in _PBP_AGGREGATES])
    if pbp is None or getattr(pbp, "columns", None) is None:
        return pd.DataFrame(columns=cols)
    if "posteam" not in pbp.columns or \
            "game_id" not in pbp.columns or "yards_gained" not in pbp.columns:
        return pd.DataFrame(columns=cols)

    present = register_pbp(con, pbp, table)
    # COALESCE(SUM(...), 0) matches pandas' skipna groupby.sum(), which returns
    # 0.0 for an all-NaN slice; DuckDB SUM over all-NULL returns NULL. The
    # absent-column branch stays strict NULL (pandas sets those columns to NaN).
    parts = ["COALESCE(SUM(yards_gained), 0) AS total_yards",
             "COUNT(yards_gained) AS n_plays"]
    refs = ["g.total_yards", "g.n_plays"]
    for col, s_col, n_col in (("epa", "epa_sum", "epa_n"),
                              ("qb_epa", "qb_epa_sum", "qb_epa_n")):
        if col in present:
            parts.append(f"COALESCE(SUM(CAST({col} AS DOUBLE)), 0) AS {s_col}")
            refs.append(f"g.{s_col}")
            parts.append(f"COUNT({col}) AS {n_col}")
            refs.append(f"g.{n_col}")
        else:
            parts.append(f"NULL::DOUBLE AS {s_col}")
            refs.append(f"g.{s_col}")
            parts.append(f"NULL::BIGINT AS {n_col}")
            refs.append(f"g.{n_col}")
    sel_extras = []
    for n in extra_names:
        agg = _PBP_AGGREGATES.get(n)
        if agg is not None:
            sel_extras.append(f"{agg.expr} AS {agg.result_col}")
            refs.append(f"g.{agg.result_col}")
    select_list = ",\n  ".join(parts + sel_extras)
    ref_list = ",\n  ".join(refs)

    sql = (
        "SELECT\n"
        "  g.game_id, g.posteam AS team,\n"
        f"  {ref_list},\n"
        f"  {_elapsed_expr(table, 'game_seconds_remaining' in present)} AS elapsed_min\n"
        "FROM (\n"
        f"  SELECT game_id, posteam, {select_list}\n"
        f"  FROM {table}\n"
        "  WHERE posteam IS NOT NULL\n"
        "  GROUP BY game_id, posteam\n"
        ") g\n"
        "ORDER BY g.game_id, g.posteam"
    )
    out = con.execute(sql).df()
    if out.empty:
        return pd.DataFrame(columns=cols)
    res = out[cols].copy()
    # Absent-column branches carry SQL NULL -> pandas nullable Int64 <NA>; the
    # pandas rollup sets those columns to float NaN. Normalize so parity holds.
    _idents = {"game_id", "team"}
    for c in res.columns:
        # Absent-column branches surface as DuckDB nullable Int64 <NA>; map
        # every numeric column to a plain float64 (NA -> NaN) so the frame
        # matches the pandas rollup's float-NaN representation exactly.
        if c not in _idents:
            res[c] = res[c].to_numpy(dtype=np.float64, na_value=np.nan)
    return res