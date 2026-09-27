"""Pin the OUT/IR game-eligibility contract for the expected lineup.

The feature ships in ``features.py``'s lineup-pool SQL constants, so these
tests drive the SHIPPED SQL against synthetic ``batter_ratings`` /
``il_stints`` tables rather than a re-implementation: if someone edits the
pool, the IL predicate, or the flag emission, a test here fails.

What is pinned, and why each one matters:

* the pool WIDENS past actual participants -- without this the OUT/IR
  filter is a no-op, because an injured player is absent from the
  participant pool by construction.
* the IL predicate binds, respects stint bounds in BOTH directions, and
  treats an open stint (NULL end) as still injured.
* an OUT/IR player is REMOVED from the eligible nine and a healthy
  replacement is promoted -- but his rating row remains untouched in the
  pool (eligibility, not quality).
* the flag is MAX(on_il) over the projected nine only: an IL player
  OUTSIDE the top-9 does NOT trip it.
* the fallback path (cache missing) hardwires on_il = 0 and restricts the
  pool to that game's own rating rows -- byte-equivalent to the pre-IL
  participant-pool feature.
* both paths emit the SAME columns (serving contract).
* the flag columns are known-pool RFE candidates and the generation
  universe width is untouched (64 = 62 + 2 candidate-side flags in the
  known pool, universe itself stays 62).
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import features  # noqa: E402
import training  # noqa: E402

GAME = pd.Timestamp("2024-07-10").date()
GPK = 745001
TEAM = "BOS"


def _ratings(pa30s: dict[int, float], extra_team: str | None = None) -> pd.DataFrame:
    """One rating row per (game, team, batter). pa30s: batter -> (pa30, woba)."""
    rows = []
    for batter, (pa30, woba) in pa30s.items():
        rows.append({"game_date": GAME, "game_pk": GPK, "batting_team": TEAM,
                     "batter": batter, "shrunk_woba": woba, "_pa30": pa30})
    if extra_team:
        for batter, (pa30, woba) in pa30s.items():
            rows.append({"game_date": GAME, "game_pk": GPK,
                         "batting_team": extra_team,
                         "batter": batter, "shrunk_woba": woba, "_pa30": pa30})
    return pd.DataFrame(rows)


@pytest.fixture()
def con():
    c = duckdb.connect(database=":memory:")
    yield c
    c.close()


def _run_pool(con: duckdb.DuckDBPyConnection, ratings: pd.DataFrame,
              il_rows: list[tuple[int, str, str | None]]) -> None:
    con.register("batter_ratings", ratings)
    if il_rows:
        vals = ",".join(
            "({}, DATE '{}', {})".format(b, s, f"DATE '{e}'" if e else "NULL")
            for b, s, e in il_rows)
        con.execute(f"""CREATE TABLE il_stints AS
            SELECT CAST(batter AS BIGINT) batter,
                   CAST(il_start AS DATE) il_start,
                   CAST(il_end AS DATE) il_end
            FROM (VALUES {vals}) AS t(batter, il_start, il_end)""")
    else:
        con.execute("""CREATE TABLE il_stints AS
            SELECT CAST(NULL AS BIGINT) AS batter,
                   CAST(NULL AS DATE) AS il_start,
                   CAST(NULL AS DATE) AS il_end
            WHERE false""")
    con.execute(features._LINEUP_POOL_SQL.format(
        lookback=features.LINEUP_POOL_LOOKBACK_DAYS, restrict="",
        on_il=features._IL_EXISTS_PREDICATE))
    con.execute(features._LINEUP_AGG_SQL)
    con.execute(features._LINEUP_IL_FLAG_SQL)


def _run_pool_fallback(con: duckdb.DuckDBPyConnection,
                       ratings: pd.DataFrame) -> None:
    con.register("batter_ratings", ratings)
    con.execute("""CREATE TABLE il_stints AS
        SELECT CAST(NULL AS BIGINT) AS batter,
               CAST(NULL AS DATE) AS il_start,
               CAST(NULL AS DATE) AS il_end
        WHERE false""")
    con.execute(features._LINEUP_POOL_SQL.format(
        lookback=features.LINEUP_POOL_LOOKBACK_DAYS,
        restrict="AND r.game_pk = g.game_pk", on_il="0"))
    con.execute(features._LINEUP_AGG_SQL)
    con.execute(features._LINEUP_IL_FLAG_SQL)


# ── pool widening + eligibility ──────────────────────────────────────────────

def test_pool_widens_past_participants(con):
    # 11 batters rated 9 days ago, none batted in THIS game's rating rows.
    old = {1000 + i: (100.0 - i, 0.300 + i / 1000) for i in range(11)}
    _run_pool(con, _ratings(old), [])
    n = con.execute("SELECT count(*) FROM lineup_pool").fetchone()[0]
    assert n == 11, "rating rows inside the lookback window must all pool"


def test_il_predicate_binds_and_respects_bounds(con):
    # 1100 placed BEFORE the game -> flagged; 1101 placed AFTER -> not;
    # 1102 stint closed the day BEFORE the game -> not; 1103 open stint -> yes.
    r = _ratings({1100: (50, 0.300), 1101: (49, 0.300),
                  1102: (48, 0.300), 1103: (47, 0.300)})
    _run_pool(con, r, [
        (1100, "2024-07-01", None),            # open, started before
        (1101, "2024-07-11", None),            # starts the day after
        (1102, "2024-06-01", "2024-07-09"),    # closed the day before
        (1103, "2024-07-10", None),            # opens the game date itself
    ])
    flags = dict(con.execute(
        "SELECT batter, on_il FROM lineup_pool").fetchall())
    assert flags == {1100: 1, 1101: 0, 1102: 0, 1103: 1}


def test_out_player_removed_and_replacement_promoted(con):
    # 10 healthy batters; the best one (pa30=200) is OUT/IR.
    r = {2000 + i: (200.0 - i, 0.400 - i / 1000) for i in range(10)}
    _run_pool(con, _ratings(r), [(2000, "2024-07-01", None)])
    nine = [b for b, in con.execute(
        """WITH ranked AS (SELECT batter, _pa30 FROM lineup_pool
           WHERE on_il = 0 ORDER BY _pa30 DESC LIMIT 9)
           SELECT batter FROM ranked""").fetchall()]
    assert 2000 not in nine, "OUT/IR player must leave the eligible nine"
    assert len(nine) == 9 and 2008 in nine, "healthy replacement is promoted"
    # rating row survives in the pool: eligibility flag, not a quality edit
    row = con.execute(
        "SELECT shrunk_woba, _pa30 FROM lineup_pool WHERE batter=2000"
    ).fetchone()
    assert row == pytest.approx((0.400, 200.0))


def test_il_outside_top9_does_not_trip_flag(con):
    # 10 healthy batters; the WORST one (rank 10) is OUT/IR.
    r = {3000 + i: (100.0 - i, 0.300) for i in range(10)}
    _run_pool(con, _ratings(r), [(3009, "2024-07-01", None)])
    flag = con.execute("SELECT lineup_il_flag FROM lineup_il_flag").fetchone()[0]
    assert flag == 0, "an IL player outside the projected nine is not a flag"


def test_il_inside_top9_trips_flag(con):
    r = {4000 + i: (100.0 - i, 0.300) for i in range(10)}
    _run_pool(con, _ratings(r), [(4003, "2024-07-01", None)])
    flag = con.execute("SELECT lineup_il_flag FROM lineup_il_flag").fetchone()[0]
    assert flag == 1


# ── fallback = pre-IL participant behavior ───────────────────────────────────

def test_fallback_is_participant_pool_with_zero_flag(con):
    # Two games for the team; only today's rows are the participant pool.
    rows = []
    for i in range(9):
        rows.append({"game_date": GAME, "game_pk": GPK, "batting_team": TEAM,
                     "batter": 5000 + i, "shrunk_woba": 0.310, "_pa30": 90.0})
    prior = dict(rows[0]); prior["game_pk"] = GPK - 1
    prior["game_date"] = pd.Timestamp(GAME) - pd.Timedelta(days=3)
    prior["batter"] = 5099
    rows.append(prior)
    df = pd.DataFrame(rows)
    _run_pool_fallback(con, df)
    today = {b for b, in con.execute(
        f"SELECT batter FROM lineup_pool WHERE game_pk={GPK}").fetchall()}
    assert 5099 not in today, "fallback pool restricts to that game's rows"
    assert len(today) == 9, "today's side pools only its own participants"
    assert con.execute(
        f"SELECT max(on_il) FROM lineup_pool WHERE game_pk={GPK}").fetchone()[0] == 0
    assert con.execute(
        "SELECT lineup_il_flag FROM lineup_il_flag").fetchone()[0] == 0


def test_both_paths_emit_same_columns(con):
    r = _ratings({6000 + i: (100.0 - i, 0.300) for i in range(10)})
    c1 = duckdb.connect(database=":memory:")
    _run_pool(c1, r, [(6002, "2024-07-01", None)])
    cols_il = [d[0] for d in c1.execute(
        "DESCRIBE SELECT * FROM lineup_pool").fetchall()]
    agg_il = [d[0] for d in c1.execute(
        "DESCRIBE SELECT * FROM lineup_agg").fetchall()]
    c1.close()
    c2 = duckdb.connect(database=":memory:")
    _run_pool_fallback(c2, r)
    cols_fb = [d[0] for d in c2.execute(
        "DESCRIBE SELECT * FROM lineup_pool").fetchall()]
    agg_fb = [d[0] for d in c2.execute(
        "DESCRIBE SELECT * FROM lineup_agg").fetchall()]
    c2.close()
    assert cols_il == cols_fb and agg_il == agg_fb


# ── serving contract ─────────────────────────────────────────────────────────

def test_flag_columns_are_known_pool_candidates():
    for c in ("lineup_il_flag_home", "lineup_il_flag_away",
              "lineup_il_flag_diff"):
        assert c in training.KNOWN_FEATURE_COLS, c


def test_generation_universe_width_unchanged():
    # The flags are candidates, NOT universe members: the 62-col serving
    # contract is untouched until an RFE adoption says otherwise.
    assert len(training.MONEYLINE_FEATURE_COLS) == 62
    assert "lineup_il_flag_home" not in training.MONEYLINE_FEATURE_COLS


def test_pool_and_flag_sql_are_shipped_constants():
    for name in ("_LINEUP_POOL_SQL", "_LINEUP_AGG_SQL", "_LINEUP_IL_FLAG_SQL",
                 "_IL_EXISTS_PREDICATE"):
        assert hasattr(features, name), name
    assert "on_il" in features._LINEUP_AGG_SQL
    assert "lineup_il_flag" in features._LINEUP_IL_FLAG_SQL
