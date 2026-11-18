"""The price-history refresh must look back from NOW (2026-11-18).

The 2026-11-17 NHL run enumerated its window from the cursor, which sits at
the furthest scheduled round (94 = Dec 31), so the walk began in December and
every November round was skipped: 3 price requests, +3 new, 0 moved, and the
11 games dated that night kept an 11:09Z morning capture.
"""
from __future__ import annotations

import pandas as pd

import odds_history as oh


def _m(mid, date):
    return {"SIMatchID": mid,
            "MatchData": {"Date": date, "HomeTeam": {"Abv": "H"},
                          "AwayTeam": {"Abv": "A"}}}


def test_the_lookback_trails_now_not_the_furthest_scheduled_round(monkeypatch):
    rounds = {
        10: [_m("oct", "2026-10-10T00:00:00Z")],
        50: [_m("nov", "2026-11-10T00:00:00Z")],
        94: [_m("dec", "2026-12-31T23:00:00Z")],
    }
    monkeypatch.setattr(oh, "matches_for",
                        lambda season, rnd: rounds.get(rnd, []))
    monkeypatch.setattr(oh, "_now",
                        lambda: pd.Timestamp("2026-11-18T00:00:00Z"))
    monkeypatch.setattr(oh, "EMPTY_LIMIT", 200)
    games, last = oh.enumerate_recent(2026, 94, 30)
    ids = {g["match_id"] for g in games}
    assert "nov" in ids          # inside the 30-day lookback
    assert "dec" in ids          # still scheduled
    assert "oct" not in ids      # outside it
    assert last == 94


def test_missing_unstamped_and_pre_game_rows_all_get_requested():
    now = pd.Timestamp("2026-11-18T00:00:00Z")
    base = pd.DataFrame({
        "match_id": ["known_today", "closing", "unstamped"],
        "fetched_utc": ["2026-11-17T12:00:00Z", "2026-11-10T23:00:00Z", None]})
    games = [
        {"match_id": "known_today", "date": "2026-11-18T23:00:00Z"},
        {"match_id": "closing", "date": "2026-11-10T00:00:00Z"},
        {"match_id": "unstamped", "date": "2026-11-10T00:00:00Z"},
        {"match_id": "missing", "date": "2026-11-10T00:00:00Z"},
    ]
    got = {g["match_id"] for g in oh.plan_fetch(games, base, now, 3)}
    # A row with no capture stamp is not a proven closing line, even when it
    # is older than the re-price window; the stamped closing row is left.
    assert got == {"known_today", "unstamped", "missing"}
