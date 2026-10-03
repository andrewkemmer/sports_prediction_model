"""Position-pool xwOBA family (pl_<pos>_xwoba_*, 2026-10-02 re24 A/B).

The pl_* family is the MLB counterpart of NHL's pl_<metric>_<pos> features:
per-batter trailing-30g shrunk xwOBA (Statcast
estimated_woba_using_speedangle over PA-ending events, the exp2 non-null
convention), pooled per position over the healthy roster, served as
pl_<pos>_xwoba_{home,away,diff}. The frame computes all 9 pools; the
2026-10-03 plan serves 8 (pl_dh computes but never enters the universe).

Pinned here:
  * batter_xwoba_rating_sql: both shrinkage arms evaluate to the shipped
    formula on single-row fixtures, zero-opportunity rows ship the
    position prior (2026-10-03 — the old NULL gate dropped them from
    every pool), and an unknown arm raises (mirrors test_shrink_arm.py);
  * the shrinkage target is POSITION-SEGMENTED (2026-10-02): the prior
    is the batter's own position's point-in-time league xwOBA
    (batter_league_pos), falling back to the overall lg_xwoba for an
    unmapped batter, then the 0.315 literal on day one;
  * the pool -> league-prior -> agg chain executed end-to-end on synthetic
    tables: PA-weighted pool means, and the empty-pool fallback resolving
    to the strictly-prior league-by-position mean (the NHL _POSITION_PRIOR
    analog — measured necessary: DH pools are empty in ~64% of team-games);
  * a missing position map degrades to False (loudly) — never raises;
  * add_diff_features creates all 9 pl diffs as home - away, and ships NULL
    when the level inputs are absent (never a fabricated 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import features  # noqa: E402
from features import (  # noqa: E402
    BATTER_SHRINK_K,
    PL_POSITIONS,
    PL_POS_SQL,
    _POS_AGG_SQL,
    _POS_LEAGUE_SQL,
    _POS_POOL_SQL,
    batter_xwoba_rating_sql,
)


def _eval_xwoba(expr: str, *, num: float | None = 0.0, den: int | None = 0,
                lg_xwoba: float = 0.315,
                lg_xwoba_pos: float | None = None) -> float:
    """Run one production shrunk_xwoba expression on an r/l/lp fixture.

    ``lp`` mirrors batter_league_pos: NULL lg_xwoba_pos models a batter
    with no listed position (the documented fallback chain). None num/den
    models a first-ever rating row (all-NULL window sums).
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE r AS SELECT ?::DOUBLE AS _xwoba_num30,"
        " ?::BIGINT AS _xwoba_den30",
        [None if num is None else float(num),
         None if den is None else int(den)])
    con.execute("CREATE TABLE l AS SELECT ?::DOUBLE AS lg_xwoba",
                [float(lg_xwoba)])
    con.execute("CREATE TABLE lp AS SELECT ?::DOUBLE AS lg_xwoba_pos",
                [None if lg_xwoba_pos is None else float(lg_xwoba_pos)])
    return con.execute(f"SELECT {expr} FROM r, l, lp").fetchone()[0]


def test_unknown_arm_raises_instead_of_silently_defaulting():
    with pytest.raises(ValueError, match="MLB_SHRINK_ARM"):
        batter_xwoba_rating_sql("linear")
    with pytest.raises(ValueError, match="MLB_SHRINK_ARM"):
        batter_xwoba_rating_sql("")


def test_k_is_the_shared_batter_prior_constant():
    assert BATTER_SHRINK_K == 120


def test_bayesian_matches_shipped_formula():
    got = _eval_xwoba(batter_xwoba_rating_sql("bayesian"),
                      num=36.0, den=120, lg_xwoba=0.315)
    assert got == pytest.approx((36.0 + 0.315 * 120) / (120 + 120))
    # thin sample leans hard on the prior
    got = _eval_xwoba(batter_xwoba_rating_sql("bayesian"),
                      num=3.0, den=10, lg_xwoba=0.315)
    assert got == pytest.approx((3.0 + 0.315 * 120) / (10 + 120))


def test_ramp_washes_out_prior_at_k():
    got = _eval_xwoba(batter_xwoba_rating_sql("ramp"),
                      num=48.0, den=120, lg_xwoba=0.315)
    assert got == pytest.approx(48.0 / 120)
    got = _eval_xwoba(batter_xwoba_rating_sql("ramp"),
                      num=16.5, den=60, lg_xwoba=0.315)
    assert got == pytest.approx((16.5 + 60 * 0.315) / 120)


def test_zero_opportunity_ships_the_prior_under_both_arms():
    """2026-10-03 membership audit: a zero-PA rating row ships the
    shrinkage prior instead of NULL. With zero observations the honest
    estimate IS the prior — and the row must EXIST so the QUALIFY dedupe,
    the membership clause, and coverage see the batter (a season debut's
    rating row used to vanish from every pool)."""
    for arm in ("bayesian", "ramp"):
        # zero measured PAs -> the 0.315 day-one prior
        assert _eval_xwoba(batter_xwoba_rating_sql(arm)) == pytest.approx(0.315)
        # NULL window sums (a batter's very first row, pre-LAG) behave the
        # same — the rating is never NULL under either arm
        assert _eval_xwoba(batter_xwoba_rating_sql(arm),
                           num=None, den=None) == pytest.approx(0.315)
        # the position-segmented prior still wins at zero opportunity
        assert _eval_xwoba(batter_xwoba_rating_sql(arm),
                           num=None, den=None,
                           lg_xwoba_pos=0.340) == pytest.approx(0.340)


def test_position_prior_is_the_shrinkage_target():
    """2026-10-02: the prior is the batter's POSITION league average,
    not the overall league average — DH quality for a DH, C quality for
    a catcher — on both arms."""
    got = _eval_xwoba(batter_xwoba_rating_sql("bayesian"),
                      num=20.0, den=100, lg_xwoba=0.315,
                      lg_xwoba_pos=0.340)
    assert got == pytest.approx((20.0 + 0.340 * 120) / (100 + 120))
    got = _eval_xwoba(batter_xwoba_rating_sql("ramp"),
                      num=16.5, den=60, lg_xwoba=0.315,
                      lg_xwoba_pos=0.300)
    assert got == pytest.approx((16.5 + 60 * 0.300) / 120)


def test_position_prior_falls_back_to_overall_then_literal():
    # no listed position -> lp NULL -> overall lg_xwoba
    got = _eval_xwoba(batter_xwoba_rating_sql("bayesian"),
                      num=36.0, den=120, lg_xwoba=0.320,
                      lg_xwoba_pos=None)
    assert got == pytest.approx((36.0 + 0.320 * 120) / (120 + 120))
    # day one: both priors NULL -> the 0.315 literal
    con = duckdb.connect()
    con.execute("CREATE TABLE r AS SELECT 36.0::DOUBLE AS _xwoba_num30,"
                " 120::BIGINT AS _xwoba_den30")
    con.execute("CREATE TABLE l AS SELECT NULL::DOUBLE AS lg_xwoba")
    con.execute("CREATE TABLE lp AS SELECT NULL::DOUBLE AS lg_xwoba_pos")
    got = con.execute(
        f"SELECT {batter_xwoba_rating_sql('bayesian')} FROM r, l, lp"
    ).fetchone()[0]
    assert got == pytest.approx((36.0 + 0.315 * 120) / (120 + 120))


def test_league_pos_prior_is_position_segmented_and_point_in_time():
    """batter_league_pos aggregates the same shifted rolling quantities
    per position: distinct targets per pos, cumulative through the
    current date, unmapped batters excluded."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE batter_rolling AS
        SELECT * FROM (VALUES
            (DATE '2026-06-01', 100, 30.0, 100),
            (DATE '2026-06-01', 200, 66.0, 200),
            (DATE '2026-06-02', 100, 36.0, 120),
            (DATE '2026-06-02', 300, 40.0, 100)
        ) t(game_date, batter, _xwoba_num30, _xwoba_den30)
    """)
    con.execute("""
        CREATE TABLE player_positions AS
        SELECT * FROM (VALUES (100, 2026, 'C'), (200, 2026, 'DH'))
        t(batter, season, pos)
    """)
    con.execute(features._BATTER_LEAGUE_POS_SQL)
    rows = {(pd.Timestamp(r[0]), r[1]): float(r[2])
            for r in con.execute(
                "SELECT game_date, pos, lg_xwoba_pos "
                "FROM batter_league_pos").fetchall()}
    # day 1: each position its OWN average (C 0.30 vs DH 0.33)
    assert rows[(pd.Timestamp('2026-06-01'), 'C')] == pytest.approx(0.30)
    assert rows[(pd.Timestamp('2026-06-01'), 'DH')] == pytest.approx(0.33)
    # day 2: C accumulates (30+36)/(100+120); DH has no row that day
    assert rows[(pd.Timestamp('2026-06-02'), 'C')] == pytest.approx(66 / 220)
    assert (pd.Timestamp('2026-06-02'), 'DH') not in rows
    # the unmapped batter (300) contributes to NO position prior
    assert all(pos in ('C', 'DH') for (_, pos) in rows)


def test_position_suffix_map_matches_the_served_names():
    # owner spelling: fb/sb/tb = first/second/third base; DH is the 9th pool
    assert PL_POSITIONS == ("c", "fb", "sb", "ss", "tb", "rf", "cf", "lf", "dh")
    con = duckdb.connect()
    con.execute("CREATE TABLE p AS SELECT * FROM (VALUES ('C'), ('1B'),"
                " ('2B'), ('3B'), ('SS'), ('LF'), ('CF'), ('RF'), ('DH'),"
                " ('P')) t(pos)")
    mapped = {r[0] for r in con.execute(
        f"SELECT {PL_POS_SQL} FROM p").fetchall()}
    assert mapped == set(PL_POSITIONS) | {"p"}  # P maps but never enters pools


def _pool_con(effective=None) -> duckdb.DuckDBPyConnection:
    """Synthetic batter_ratings + player_positions across two team-games.

    Day 1 (2026-06-01) has a C and a DH pool; day 2 (2026-06-15) has ONLY a
    C (the DH's last rating row falls outside the 10-day lookback), so the
    day-2 DH column must resolve to the position prior from day 1.

    ``effective`` optionally seeds lineup_effective — (game_pk, team,
    batter, tier) rows; None uses the SHIPPED empty degrade table (the
    pre-audit full-roster pool every historical fixture pins).
    """
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE batter_ratings AS
        SELECT * FROM (VALUES
            (DATE '2026-06-01', 1, 'AAA', 100, 0.30, 30),
            (DATE '2026-06-01', 1, 'AAA', 200, 0.40, 90),
            (DATE '2026-06-14', 2, 'AAA', 100, 0.33, 60)
        ) t(game_date, game_pk, batting_team, batter, shrunk_xwoba, _pa30)
    """)
    con.execute("""
        CREATE TABLE player_positions AS
        SELECT * FROM (VALUES (100, 2026, 'C'), (200, 2026, 'DH'))
        t(batter, season, pos)
    """)
    if effective:
        vals = ", ".join("(%d, '%s', %d, %d)" % (g, t, b, tier)
                         for (g, t, b, tier) in effective)
        con.execute(f"""CREATE OR REPLACE TABLE lineup_effective AS
            SELECT * FROM (VALUES {vals})
            AS t(game_pk, team, batter, tier)""")
    else:
        con.execute(features._EMPTY_LINEUP_EFFECTIVE_SQL)
    con.execute(_POS_POOL_SQL.format(
        lookback=10, pos_case=PL_POS_SQL, restrict="", on_il="0"))
    con.execute(_POS_LEAGUE_SQL)
    con.execute(_POS_AGG_SQL)
    return con


def test_pool_chain_builds_pa_weighted_means_and_fallback():
    con = _pool_con()
    cols = [r[0] for r in con.execute("DESCRIBE pos_agg").fetchall()]
    assert cols == (["game_date", "game_pk", "batting_team"]
                    + [f"pl_{p}_xwoba" for p in PL_POSITIONS])
    rows = con.execute(
        "SELECT game_pk, pl_c_xwoba, pl_dh_xwoba FROM pos_agg"
        " ORDER BY game_pk").fetchall()
    assert len(rows) == 2
    # game 1: both pools real (PA-weighted means of their single members)
    assert rows[0] == (1, pytest.approx(0.30), pytest.approx(0.40))
    # game 2: C pool real; empty DH pool falls back to the strictly-prior
    # league-by-position mean (game 1's healthy DH pool: 0.40)
    assert rows[1][0] == 2
    assert rows[1][1] == pytest.approx(0.33)
    assert rows[1][2] == pytest.approx(0.40)


def test_pool_is_point_in_time_and_one_row_per_batter():
    con = _pool_con()
    # one row per (game, team, batter): the QUALIFY dedupe — batter 100's
    # C pool appears in BOTH games, batter 200's DH pool only in game 1
    n, nb = con.execute(
        "SELECT count(*), count(DISTINCT batter) FROM pos_pool").fetchone()
    assert (n, nb) == (3, 2)
    # game 2's C pool uses batter 100's LATEST row <= game date (0.33)
    px = con.execute(
        "SELECT shrunk_xwoba FROM pos_pool WHERE game_pk = 2 AND pos = 'c'"
    ).fetchone()[0]
    assert float(px) == pytest.approx(0.33)


def test_membership_restricts_position_pools_to_the_effective_nine():
    """2026-10-03: the pl_* family binds the SAME effective nine as the
    lineup family. Game 1's nine lists only the C (100) — the DH (200) is
    left off — so the DH pool is EMPTY and resolves to the position prior
    (NULL on day one) instead of shipping 0.40 from a guy not in the game.
    Game 2 has no membership rows at all (no announced order, no
    projection) and therefore degrades to the pre-audit roster pool."""
    con = _pool_con(effective=[(1, "AAA", 100, 1)])
    try:
        row1 = con.execute(
            "SELECT pl_c_xwoba, pl_dh_xwoba FROM pos_agg WHERE game_pk = 1"
        ).fetchone()
        assert row1[0] == pytest.approx(0.30)  # tonight's C pools normally
        assert row1[1] is None, "a batter outside membership must not pool"
        row2 = con.execute(
            "SELECT pl_c_xwoba FROM pos_agg WHERE game_pk = 2").fetchone()
        assert row2[0] == pytest.approx(0.33)
    finally:
        con.close()


def test_slate_carries_the_served_pl_pools_forward():
    """2026-10-03: build_upcoming_slate carried ZERO pl_* inputs (grep found
    no pl hits in data_ingestion) — every slate row shipped NaN levels, so
    add_diff_features could not compute any of the 24 model columns at serve
    time. The 8 served pools must ride the same _RAW_CARRY/_RAW_INPUTS path
    as every other diff family, and pl_dh must NOT be carried (it generates
    but never serves)."""
    from datetime import date

    from data_ingestion import build_upcoming_slate
    from features import add_diff_features

    target = date(2025, 6, 5)
    hist = pd.DataFrame({
        "game_date": ["2025-06-01"],
        "game_pk": [1],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "home_win": [1.0], "home_score": [5], "away_score": [3],
        "total_runs": [8],
        "pl_c_xwoba_home": [0.34],
        "pl_c_xwoba_away": [0.30],
    })
    sched = pd.DataFrame({
        "game_date": [pd.Timestamp(target)],
        "game_id": ["20250605_BOS@NYY"],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "start_time_utc": [pd.Timestamp("2025-06-05 23:05")],
    })
    slate = build_upcoming_slate(hist, target, schedule_df=sched)
    assert len(slate) == 1
    # each team's own latest value rides ITS slot (home value -> home slot)
    assert float(slate["pl_c_xwoba_home"].iloc[0]) == pytest.approx(0.34)
    assert float(slate["pl_c_xwoba_away"].iloc[0]) == pytest.approx(0.30)
    out = add_diff_features(slate)
    assert float(out["pl_c_xwoba_diff"].iloc[0]) == pytest.approx(0.04)
    # the 16 served level inputs all exist on the slate row...
    for p in ("c", "fb", "sb", "ss", "tb", "rf", "cf", "lf"):
        for side in ("home", "away"):
            assert f"pl_{p}_xwoba_{side}" in slate.columns, (p, side)
    # ...while pl_dh is never carried (not a served pool): its level inputs
    # are absent and its diff ships NaN through the normal missing-input path
    assert "pl_dh_xwoba_home" not in slate.columns
    assert "pl_dh_xwoba_diff" in out.columns
    assert out["pl_dh_xwoba_diff"].isna().all()


def test_missing_position_map_degrades_loudly(monkeypatch):
    monkeypatch.setattr(features, "PLAYER_POSITIONS_FILE", "__absent__.parquet")
    con = duckdb.connect()
    assert features._register_player_positions(con) is False


def test_add_diff_features_creates_the_nine_pl_diffs():
    df = pd.DataFrame({
        "home_team": ["NYY"],
        "pl_c_xwoba_home": [0.34],
        "pl_c_xwoba_away": [0.30],
        "pl_dh_xwoba_home": [np.nan],
        "pl_dh_xwoba_away": [0.41],
    })
    out = features.add_diff_features(df)
    assert out["pl_c_xwoba_diff"].iloc[0] == pytest.approx(0.04)
    assert np.isnan(out["pl_dh_xwoba_diff"].iloc[0])  # NULL propagates
    # absent level inputs -> all 9 diffs created NULL, never 0
    out2 = features.add_diff_features(pd.DataFrame({"home_team": ["NYY"]}))
    for p in PL_POSITIONS:
        assert out2[f"pl_{p}_xwoba_diff"].isna().all(), p
