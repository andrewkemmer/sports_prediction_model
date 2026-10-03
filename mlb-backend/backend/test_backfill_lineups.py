"""Lineup cache maintenance (2026-10-03 membership audit).

lineups.parquet is membership TIER 1 (tonight's nine): features.py resolves
lineup_effective from it, so the cache's write rules decide what the pools
can price. Pinned here:

  * order_row — pure feed -> row shape; complete_* is TRUE only on an
    exact 9-man side, and a missing feed is an honest incomplete row;
  * upsert_lineups — latest-write merge by game_pk with the NO-DOWNGRADE
    rule: an incomplete pre-game post can never erase a known nine;
  * complete_game_pks — the COMPLETE-only resume set, so fetch_error /
    partial rows retry every run instead of freezing a game at
    projected-only membership forever.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import backfill_lineups as bl  # noqa: E402


def _feed(home_ids, away_ids, state="Final"):
    return {
        "gameData": {"status": {"abstractGameState": state}},
        "liveData": {"boxscore": {"teams": {
            "home": {"battingOrder": list(home_ids)},
            "away": {"battingOrder": list(away_ids)},
        }}},
    }


_NINE_H = [301, 302, 303, 304, 305, 306, 307, 308, 309]
_NINE_A = [101, 102, 103, 104, 105, 106, 107, 108, 109]


def test_order_row_requires_an_exact_nine_per_side():
    row = bl.order_row(42, "2025-06-05", "NYY", "BOS",
                       _feed(_NINE_H, _NINE_A))
    assert row["complete_home"] and row["complete_away"]
    assert row["home_order"] == _NINE_H and row["away_order"] == _NINE_A
    assert row["state"] == "Final"
    # short side -> honest INCOMPLETE row (home_order withheld, not padded)
    row = bl.order_row(43, "2025-06-05", "NYY", "BOS",
                       _feed(_NINE_H[:-1], _NINE_A))
    assert row["complete_home"] is False and row["complete_away"] is True
    assert row["home_order"] is None
    # no feed (fetch error) -> nothing fabricated, caller's state kept
    row = bl.order_row(44, "2025-06-05", "NYY", "BOS", None,
                       state="fetch_error: timeout")
    assert row["complete_home"] is False and row["complete_away"] is False
    assert row["home_order"] is None and row["away_order"] is None
    assert row["state"] == "fetch_error: timeout"


def test_order_row_never_raises_on_malformed_feed():
    row = bl.order_row(45, "2025-06-05", "NYY", "BOS",
                       {"liveData": {"boxscore": {"teams": {
                           "home": {"battingOrder": "not-a-list"}}}}})
    assert row["complete_home"] is False
    assert row["state"] == "unknown"


def _path(tmp_path) -> Path:
    return tmp_path / "lineups.parquet"


def test_upsert_appends_new_games(tmp_path):
    p = _path(tmp_path)
    n = bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
        bl.order_row(2, "2025-06-02", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
    ]), out_path=p)
    assert n == 2
    df = pd.read_parquet(p)
    assert set(df.game_pk) == {1, 2}
    assert df.complete_home.all() and df.complete_away.all()


def test_upsert_never_downgrades_a_known_nine(tmp_path):
    p = _path(tmp_path)
    bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
    ]), out_path=p)
    # a pre-game re-fetch BEFORE the sides post must not erase the nine
    n = bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", None,
                     state="Preview"),
    ]), out_path=p)
    assert n == 0
    df = pd.read_parquet(p)
    row = df.loc[df.game_pk == 1].iloc[0]
    assert row.complete_home and row.complete_away
    assert list(row.home_order) == _NINE_H
    assert row.state == "Final"


def test_upsert_upgrades_an_incomplete_row(tmp_path):
    p = _path(tmp_path)
    bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", None, state="Preview"),
    ]), out_path=p)
    assert bl.complete_game_pks(p) == set()
    n = bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
    ]), out_path=p)
    assert n == 1
    assert bl.complete_game_pks(p) == {1}


def test_complete_resume_set_excludes_partial_and_error_rows(tmp_path):
    p = _path(tmp_path)
    bl.upsert_lineups(pd.DataFrame([
        bl.order_row(1, "2025-06-01", "NYY", "BOS", _feed(_NINE_H, _NINE_A)),
        bl.order_row(2, "2025-06-02", "NYY", "BOS", _feed(_NINE_H, [])),
        bl.order_row(3, "2025-06-03", "NYY", "BOS", None,
                     state="fetch_error: timeout"),
    ]), out_path=p)
    assert bl.complete_game_pks(p) == {1}, \
        "only a 9+9 row may leave the resume set; the rest retry next run"


def test_missing_cache_resume_set_is_empty(tmp_path):
    assert bl.complete_game_pks(_path(tmp_path)) == set()
