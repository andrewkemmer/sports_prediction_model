"""Exp2 team-input availability binding (2026-10-26).

The team-side exp2 builders counted every PA taken — including subs'
PAs and PAs by bats with no path into the game. These tests pin the
membership-bound replacement (same ``lineup_effective`` membership the
lineup_* / pl_* / ops pools bind):

* game cells count only effective-nine PAs (team lefty share, per-category
  K/xwOBA counts, fastball-by-hand counts);
* with no membership rows anywhere the counts are byte-identical to the
  pre-availability build;
* the slate artifact ships tonight's resolved date-level priors with tier
  provenance, and build_upcoming_slate overrides the carry per team with
  marked pool-tN/carry provenance — through the opponent-slot mapping
  (each slot carries the OPPOSING lineup's state).
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
from features import (  # noqa: E402
    _EXP2_MEMBERSHIP_SQL,
    _EXP2_TEAM_CAT_GAME_SQL,
    _EXP2_TEAM_FBHAND_GAME_SQL,
    _TEAM_HAND_RAW_SQL,
    PA_END_EVENTS,
)


def _member(a="pa_all", pk=None, tm=None):
    pk = pk or f"{a}.game_pk"
    tm = tm or f"{a}.batting_team"
    return _EXP2_MEMBERSHIP_SQL.format(a=a, pk=pk, tm=tm)


def _pitches(con):
    # AAA bats in the Top (away); one game, three batters:
    # 100 (L) x2 PA, 200 (R) x1 PA, 300 (R, sub) x1 PA.
    con.execute(
        "CREATE TABLE pitches AS "
        "SELECT * FROM (VALUES "
        "(DATE '2026-06-01', 1, 11, 100, 'L', 'R', 'Top', 'AAA', 'BBB', "
        "'FF', 'single', 0.3), "
        "(DATE '2026-06-01', 1, 11, 100, 'L', 'R', 'Top', 'AAA', 'BBB', "
        "'FF', 'strikeout', 0.1), "
        "(DATE '2026-06-01', 1, 11, 200, 'R', 'R', 'Top', 'AAA', 'BBB', "
        "'SL', 'single', 0.3), "
        "(DATE '2026-06-01', 1, 11, 300, 'R', 'R', 'Top', 'AAA', 'BBB', "
        "'FF', 'single', 0.3) "
        ") AS t(game_date, game_pk, pitcher, batter, stand, p_throws, "
        "inning_topbot, away_team, home_team, pitch_type, events, "
        "estimated_woba_using_speedangle)"
    )


def _effective(con, members=()):
    if members:
        vals = ", ".join("(%d, '%s', %d, %d)" % m for m in members)
        con.execute(
            "CREATE TABLE lineup_effective AS "
            f"SELECT * FROM (VALUES {vals}) "
            "AS t(game_pk, team, batter, tier)"
        )
    else:
        con.execute(features._EMPTY_LINEUP_EFFECTIVE_SQL)


def test_team_hand_share_counts_only_effective_nine_pas():
    """Lefty share prices the nine: the sub's R PAs must not move it."""
    con = duckdb.connect()
    try:
        _pitches(con)
        _effective(con, [(1, "AAA", 100, 1), (1, "AAA", 200, 1)])
        con.execute(
            _TEAM_HAND_RAW_SQL.format(
                pa_events=PA_END_EVENTS,
                batter_excl="TRUE",
                member=_member(),
            )
        )
        (share,) = con.execute("SELECT lefty_share FROM team_hand_raw").fetchone()
        # members 100 (2 L PAs) + 200 (1 R PA): 2/3 — the sub's R PA is
        # excluded (2/4 unbound).
        assert float(share) == pytest.approx(2 / 3)
    finally:
        con.close()


def _exp2_pa(con):
    con.execute(
        "CREATE TABLE exp2_pa AS "
        "SELECT * FROM (VALUES "
        "(DATE '2026-06-01', 1, 11, 100, 'L', 'AAA', 'fastball', "
        "1.0, 1.0, 0.1, 1.0, 0.1), "
        "(DATE '2026-06-01', 1, 11, 200, 'R', 'AAA', 'fastball', "
        "0.0, 1.0, 0.3, 1.0, 0.3), "
        "(DATE '2026-06-01', 1, 11, 300, 'R', 'AAA', 'fastball', "
        "0.0, 1.0, 0.5, 1.0, 0.5) "
        ") AS t(game_date, game_pk, pitcher, batter, stand, batting_team, "
        "pitch_cat, k_flag, n_flag, xwoba_val, xwoba_n, xwoba_summand)"
    )


def test_team_cat_counts_exclude_non_member_pas():
    """Category priors count the nine: sub PAs leave k_n and pa_n."""
    con = duckdb.connect()
    try:
        _exp2_pa(con)
        _effective(con, [(1, "AAA", 100, 1), (1, "AAA", 200, 1)])
        con.execute(
            _EXP2_TEAM_CAT_GAME_SQL.format(
                batter_excl="TRUE",
                member=_EXP2_MEMBERSHIP_SQL.format(
                    a="a", pk="a.game_pk", tm="a.batting_team"
                ),
            )
        )
        k, pa, xn, xw = con.execute(
            "SELECT k_n, pa_n, xwoba_num, xwoba_n FROM exp2_team_cat_game"
        ).fetchone()
        # members: 100 (K) + 200 (non-K); the sub's PA is excluded
        # (unbound: k_n 1, pa_n 3).
        assert (k, pa) == (1, 2)
        assert float(xn) == pytest.approx(0.1 + 0.3)
        assert float(xw) == pytest.approx(2.0)
    finally:
        con.close()


def test_fbhand_counts_exclude_non_member_pas():
    """Hand-split fastball counts bind the same nine."""
    con = duckdb.connect()
    try:
        _exp2_pa(con)
        _effective(con, [(1, "AAA", 100, 1), (1, "AAA", 200, 1)])
        con.execute(
            _EXP2_TEAM_FBHAND_GAME_SQL.format(
                batter_excl="TRUE",
                member=_EXP2_MEMBERSHIP_SQL.format(
                    a="a", pk="a.game_pk", tm="a.batting_team"
                ),
            )
        )
        rows = con.execute(
            "SELECT stand, k_n, pa_n FROM exp2_team_fbhand_game ORDER BY stand"
        ).fetchall()
        assert rows == [("L", 1, 1), ("R", 0, 1)]
    finally:
        con.close()


def test_degrade_without_membership_counts_everything():
    """No membership rows: the gate is vacuous — byte-identical counts."""
    con = duckdb.connect()
    try:
        _pitches(con)
        _effective(con)
        con.execute(
            _TEAM_HAND_RAW_SQL.format(
                pa_events=PA_END_EVENTS,
                batter_excl="TRUE",
                member=_member(),
            )
        )
        (share,) = con.execute("SELECT lefty_share FROM team_hand_raw").fetchone()
        assert float(share) == pytest.approx(2 / 4)
    finally:
        con.close()


def test_export_slate_exp2_ships_resolved_priors_with_tier(tmp_path):
    """Slate team-game resolves the latest strictly-prior same-season rows."""
    from features import _export_slate_exp2

    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE pool_universe AS "
            "SELECT * FROM (VALUES "
            "(DATE '2026-06-20', 9001, 'AAA', true) "
            ") AS t(game_date, game_pk, batting_team, is_slate)"
        )
        con.execute(
            "CREATE TABLE exp2_team_cat AS "
            "SELECT * FROM (VALUES "
            "(DATE '2026-06-15', 'AAA', 'fastball', 0.25, 0.32), "
            "(DATE '2026-06-15', 'AAA', 'breaking', 0.28, 0.30), "
            "(DATE '2026-06-15', 'AAA', 'offspeed', 0.20, 0.34), "
            "(DATE '2025-09-01', 'AAA', 'fastball', 0.99, 0.99) "
            ") AS t(game_date, batting_team, pitch_cat, "
            "team_k_pct_cat, team_xwoba_cat)"
        )
        con.execute(
            "CREATE TABLE exp2_team_fbhand AS "
            "SELECT * FROM (VALUES "
            "(DATE '2026-06-15', 'AAA', 'L', 0.22), "
            "(DATE '2026-06-15', 'AAA', 'R', 0.26) "
            ") AS t(game_date, batting_team, stand, team_k_pct_fb_vs)"
        )
        con.execute(
            "CREATE TABLE team_hand_rolling AS "
            "SELECT * FROM (VALUES "
            "(DATE '2026-06-15', 1, 'AAA', 0.40) "
            ") AS t(game_date, game_pk, batting_team, lineup_lefty_share_30g)"
        )
        _effective(con, [(9001, "AAA", 100, 2)])
        n = _export_slate_exp2(con, base=tmp_path)
        assert n == 1
        art = pd.read_parquet(tmp_path / "exp2_slate_20260620.parquet")
        assert len(art) == 1
        r = art.iloc[0]
        assert str(r["team"]) == "AAA"
        assert int(r["exp2_tier"]) == 2
        assert float(r["team_k_pct_cat_fastball"]) == pytest.approx(0.25)
        assert float(r["team_xwoba_cat_breaking"]) == pytest.approx(0.30)
        assert float(r["team_k_pct_fb_vs_l"]) == pytest.approx(0.22)
        assert float(r["opp_lefty_share"]) == pytest.approx(0.40)
    finally:
        con.close()


def test_slate_exp2_override_uses_opponent_mapping_and_marks(tmp_path, monkeypatch):
    """Resolved team state flows through the opponent-slot mapping, marked.

    NYY (home) resolves (tier 2); BOS is absent from the artifact. The
    home slot carries the AWAY team's state (BOS: carry, marked carry);
    the away slot carries the HOME team's state (NYY: resolved, pool-t2).
    A NULL artifact input stays NaN — never the stale carry.
    """
    import data_ingestion as di
    from data_ingestion import build_upcoming_slate

    target = date(2025, 6, 5)
    art = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp(target),
                "game_pk": 9001,
                "team": "NYY",
                "team_k_pct_cat_fastball": 0.25,
                "team_k_pct_cat_breaking": 0.28,
                "team_k_pct_cat_offspeed": 0.20,
                "team_xwoba_cat_fastball": None,
                "team_xwoba_cat_breaking": 0.30,
                "team_xwoba_cat_offspeed": 0.34,
                "team_k_pct_fb_vs_l": 0.22,
                "team_k_pct_fb_vs_r": 0.26,
                "opp_lefty_share": 0.40,
                "exp2_tier": 2,
            }
        ]
    )
    art.to_parquet(tmp_path / "exp2_slate_20250605.parquet", index=False)
    monkeypatch.setattr(di, "DATA_DELIVERY_DIR", tmp_path)

    hist = pd.DataFrame(
        {
            "game_date": ["2025-06-01"],
            "game_pk": [1],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
            "home_win": [1.0],
            "home_score": [5],
            "away_score": [3],
            "total_runs": [8],
            "team_k_pct_cat_fastball_home": [0.30],
            "team_k_pct_cat_fastball_away": [0.20],
            "team_xwoba_cat_fastball_home": [0.33],
            "team_xwoba_cat_fastball_away": [0.31],
            "team_k_pct_fb_vs_l_home": [0.21],
            "team_k_pct_fb_vs_l_away": [0.19],
            "opp_lefty_share_home": [0.35],
            "opp_lefty_share_away": [0.45],
        }
    )
    sched = pd.DataFrame(
        {
            "game_date": [pd.Timestamp(target)],
            "game_id": ["20250605_BOS@NYY"],
            "home_team": ["NYY"],
            "away_team": ["BOS"],
            "start_time_utc": [pd.Timestamp("2025-06-05 23:05")],
        }
    )
    slate = build_upcoming_slate(hist, target, schedule_df=sched)
    assert len(slate) == 1
    # home slot = AWAY team's state (BOS, uncovered): the carry stands —
    # the hist *_home cols describe the away lineup, so BOS carries 0.30.
    assert float(slate["team_k_pct_cat_fastball_home"].iloc[0]) == pytest.approx(
        0.30
    )
    assert slate["exp2_source_home"].iloc[0] == "carry"
    # away slot = HOME team's state (NYY, resolved): override wins;
    # the NULL xwoba split stays NaN instead of re-inheriting 0.33.
    assert float(slate["team_k_pct_cat_fastball_away"].iloc[0]) == pytest.approx(
        0.25
    )
    assert pd.isna(slate["team_xwoba_cat_fastball_away"].iloc[0])
    assert slate["exp2_source_away"].iloc[0] == "pool-t2"
