"""Lineup hand-split availability binding (2026-10-26).

``lineup_ops_vs_l/r`` used to average the trailing top-9 by PA per hand —
whoever banked the most PAs, with no notion of tonight's nine. An IL bat
kept his slot; a call-up starting tonight might not make the nine. These
tests pin the membership-bound replacement (same ``lineup_effective``
membership the lineup_* and pl_* pools bind):

* the announced nine decides who pools (a rated batter left off the order
  does not), PA-weighted within the nine;
* the grid is pool_universe (history AND the slate), so tonight's splits
  resolve instead of serving last game's population;
* with no membership rows anywhere the game's own rows reproduce the
  exact pre-availability population (the rn <= 9 cut is vacuous when
  membership binds: the nine IS the population);
* the slate artifact overrides the carry per team-date with marked
  pool-tN/carry provenance, and the probable-hand matchup selection picks
  over the resolved levels.
"""

from datetime import date

import duckdb
import pandas as pd
import pytest

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import features  # noqa: E402


def _universe(con, slate=()):
    con.execute("""CREATE TABLE pool_universe AS
        SELECT * FROM (VALUES
            (DATE '2026-06-01', 1, 'AAA', false)
        ) AS t(game_date, game_pk, batting_team, is_slate)""")
    for game_pk, game_date, team in slate:
        con.execute(
            "INSERT INTO pool_universe VALUES "
            f"(DATE '{game_date}', {game_pk}, '{team}', true)")


def _rolling(con, rows):
    # (game_date, game_pk, batting_team, batter, p_throws,
    #  _onb_num, _tb30, _ab30, _pa30)
    vals = ", ".join(
        f"(DATE '{d}', {pk}, '{tm}', {b}, '{h}', {onb}, {tb}, {ab}, {pa})"
        for (d, pk, tm, b, h, onb, tb, ab, pa) in rows)
    con.execute(f"""CREATE TABLE batter_hand_rolling AS
        SELECT * FROM (VALUES {vals})
        AS t(game_date, game_pk, batting_team, batter, p_throws,
             _onb_num, _tb30, _ab30, _pa30)""")


def _effective(con, members=()):
    if members:
        vals = ", ".join("(%d, '%s', %d, %d)" % m for m in members)
        con.execute(f"""CREATE TABLE lineup_effective AS
            SELECT * FROM (VALUES {vals})
            AS t(game_pk, team, batter, tier)""")
    else:
        con.execute(features._EMPTY_LINEUP_EFFECTIVE_SQL)


def _ops(con):
    con.execute(features._LINEUP_OPS_AGG_SQL)
    return con.execute(
        "SELECT lineup_ops_vs_l, lineup_ops_vs_r FROM lineup_ops_agg"
    ).fetchone()


def test_ops_pool_binds_the_effective_nine_pa_weighted():
    """Announced nine decides; PA-weighted mean; no future leakage.

    Batters 100/200 are tonight's nine (tier 1); rated batter 300 (ops
    2.0 over 300 PA) is left off the order and must not pool. Batter 100
    also has a poisoned LATER row (game 2, 06-15) that must not leak into
    game 1 through the member leg's date window.
    """
    con = duckdb.connect()
    try:
        _universe(con)
        _rolling(con, [
            ("2026-06-01", 1, "AAA", 100, "R", 10, 15, 30, 100),  # ops .8333
            ("2026-06-01", 1, "AAA", 200, "R", 6, 6, 20, 20),      # ops .60
            ("2026-06-01", 1, "AAA", 300, "R", 30, 30, 30, 300),   # ops 2.0
            ("2026-06-01", 1, "AAA", 100, "L", 9, 12, 30, 100),    # ops .70
            ("2026-06-15", 2, "AAA", 100, "R", 300, 300, 30, 3000),  # poison
        ])
        _effective(con, [(1, "AAA", 100, 1), (1, "AAA", 200, 1)])
        got_l, got_r = _ops(con)
        # R: (100*.8333 + 20*.60) / 120 — batter 300 (ops 2.0) excluded,
        # later row excluded by the strictly-PIT date cap.
        assert got_r == pytest.approx((100 * (10 / 30 + 15 / 30)
                                       + 20 * (6 / 20 + 6 / 20)) / 120)
        assert got_r == pytest.approx(0.794444, abs=1e-4)
        assert got_l == pytest.approx(0.70)
    finally:
        con.close()


def test_ops_degrade_without_membership_uses_participants():
    """No membership rows anywhere: the game's own rows pool (PA-weighted).

    The exact pre-availability population (participants), only the
    combination changes (PA-weighted mean instead of the simple mean) —
    coverage never shrinks below what shipped before.
    """
    con = duckdb.connect()
    try:
        _universe(con)
        _rolling(con, [
            ("2026-06-01", 1, "AAA", 100, "R", 10, 15, 30, 100),
            ("2026-06-01", 1, "AAA", 300, "R", 30, 30, 30, 300),
        ])
        _effective(con)
        _, got_r = _ops(con)
        assert got_r == pytest.approx((100 * (10 / 30 + 15 / 30)
                                       + 300 * 2.0) / 400)
    finally:
        con.close()


def test_export_slate_lineup_ships_resolved_splits_with_tier(tmp_path):
    """Slate team-game resolves through the member leg (no own rows exist).

    Tonight's nine (tier 2 here) prices trailing data; the left-off batter
    (300) does not pool; the hand with no member data stays NULL (honest,
    never the carry).
    """
    from features import _export_slate_lineup

    con = duckdb.connect()
    try:
        _universe(con, slate=[(9001, "2026-06-20", "AAA")])
        _rolling(con, [
            ("2026-06-15", 1, "AAA", 100, "R", 10, 15, 30, 100),
            ("2026-06-15", 1, "AAA", 200, "R", 6, 6, 20, 20),
            ("2026-06-15", 1, "AAA", 300, "R", 30, 30, 30, 300),
        ])
        _effective(con, [(9001, "AAA", 100, 2), (9001, "AAA", 200, 2)])
        con.execute(features._LINEUP_OPS_AGG_SQL)
        n = _export_slate_lineup(con, base=tmp_path)
        assert n == 1
        art = pd.read_parquet(tmp_path / "lineup_slate_20260620.parquet")
        assert len(art) == 1
        r = art.iloc[0]
        assert str(r["team"]) == "AAA"
        assert int(r["lineup_tier"]) == 2
        assert float(r["lineup_ops_vs_r"]) == pytest.approx(0.794444, abs=1e-4)
        assert pd.isna(r["lineup_ops_vs_l"])
    finally:
        con.close()


def test_slate_lineup_artifact_overrides_carry_and_marks_source(tmp_path,
                                                                monkeypatch):
    """Resolved splits override the carry; uncovered sides stay marked.

    NYY's artifact row (tier 2) wins over the carried levels; a NULL
    artifact split stays NaN (never re-inherits the stale carry). BOS is
    absent from the artifact — its carry stands, MARKED.
    """
    import data_ingestion as di
    from data_ingestion import build_upcoming_slate

    target = date(2025, 6, 5)
    art = pd.DataFrame([{
        "game_date": pd.Timestamp(target),
        "game_pk": 9001,
        "team": "NYY",
        "lineup_ops_vs_l": 0.75,
        "lineup_ops_vs_r": None,
        "lineup_tier": 2,
    }])
    art.to_parquet(tmp_path / "lineup_slate_20250605.parquet", index=False)
    monkeypatch.setattr(di, "DATA_DELIVERY_DIR", tmp_path)

    hist = pd.DataFrame({
        "game_date": ["2025-06-01"],
        "game_pk": [1],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "home_win": [1.0], "home_score": [5], "away_score": [3],
        "total_runs": [8],
        "lineup_ops_vs_l_home": [0.80],
        "lineup_ops_vs_l_away": [0.60],
        "lineup_ops_vs_r_home": [0.70],
        "lineup_ops_vs_r_away": [0.50],
    })
    sched = pd.DataFrame({
        "game_date": [pd.Timestamp(target)],
        "game_id": ["20250605_BOS@NYY"],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "start_time_utc": [pd.Timestamp("2025-06-05 23:05")],
    })
    slate = build_upcoming_slate(hist, target, schedule_df=sched)
    assert len(slate) == 1
    # home (NYY): resolved splits win; NULL stays NaN, never the carry.
    assert float(slate["lineup_ops_vs_l_home"].iloc[0]) == pytest.approx(0.75)
    assert pd.isna(slate["lineup_ops_vs_r_home"].iloc[0])
    assert slate["lineup_source_home"].iloc[0] == "pool-t2"
    # away (BOS): absent from the artifact — the carry stands, MARKED.
    assert float(slate["lineup_ops_vs_l_away"].iloc[0]) == pytest.approx(0.60)
    assert float(slate["lineup_ops_vs_r_away"].iloc[0]) == pytest.approx(0.50)
    assert slate["lineup_source_away"].iloc[0] == "carry"


def test_slate_advantage_selects_probable_hand_over_resolved_levels(tmp_path,
                                                                   monkeypatch):
    """The probable-hand matchup selection picks over tonight's population.

    NYY hosts; the opposing (away) probable hand is L, so the advantage
    level must be the RESOLVED L split (0.75), not the carried R split.
    """
    import data_ingestion as di
    from data_ingestion import build_upcoming_slate
    from features import add_diff_features

    target = date(2025, 6, 5)
    art = pd.DataFrame([{
        "game_date": pd.Timestamp(target),
        "game_pk": 9001,
        "team": "NYY",
        "lineup_ops_vs_l": 0.75,
        "lineup_ops_vs_r": 0.65,
        "lineup_tier": 1,
    }])
    art.to_parquet(tmp_path / "lineup_slate_20250605.parquet", index=False)
    monkeypatch.setattr(di, "DATA_DELIVERY_DIR", tmp_path)

    hist = pd.DataFrame({
        "game_date": ["2025-06-01"],
        "game_pk": [1],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "home_win": [1.0], "home_score": [5], "away_score": [3],
        "total_runs": [8],
        "home_starter_id": [101], "away_starter_id": [202],
        "home_starter_hand": ["R"], "away_starter_hand": ["L"],
        "lineup_ops_vs_l_home": [0.80],
        "lineup_ops_vs_l_away": [0.60],
        "lineup_ops_vs_r_home": [0.70],
        "lineup_ops_vs_r_away": [0.50],
    })
    sched = pd.DataFrame({
        "game_date": [pd.Timestamp(target)],
        "game_id": ["20250605_BOS@NYY"],
        "home_team": ["NYY"], "away_team": ["BOS"],
        "start_time_utc": [pd.Timestamp("2025-06-05 23:05")],
        "sp_id_home": [101], "sp_id_away": [202],
    })
    slate = build_upcoming_slate(hist, target, schedule_df=sched)
    # opposing hand for the home side is L → resolved 0.75, not carry 0.70.
    assert float(slate["lineup_ops_vs_starter_hand_home"].iloc[0]) == \
        pytest.approx(0.75)
    assert slate["lineup_source_home"].iloc[0] == "pool-t1"
    out = add_diff_features(slate)
    assert slate["lineup_source_home"].iloc[0] == "pool-t1"
    assert float(out["lineup_ops_vs_starter_hand_home"].iloc[0]) == \
        pytest.approx(0.75)
