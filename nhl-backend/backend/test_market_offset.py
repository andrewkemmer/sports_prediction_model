"""Serve-time market ids and the robustness of a failed fetch (2026-11-16).

``attach_live`` used to look a match id up on (home, away) alone. A club meets
the same opponent twice inside the lookback window routinely -- 124 consecutive
NHL meetings sit 4 days or less apart -- so when tonight's row had not been
scraped yet, the pair's most recent stored row was the EARLIER game and its
closing price was served for tonight's, with no NaN to show for it. These pin
the date key that stops it, and the schema guarantee that keeps a dead fetch
from raising through the serving build.
"""
from __future__ import annotations

import pandas as pd
import pytest

try:
    from backend import market_offset as mo
except ImportError:  # pragma: no cover - direct script/import fallback
    import market_offset as mo  # type: ignore


def _hist(tmp_path, rows, name="hist.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _row(mid, date, home="BOS", away="NYK", h=1.80, a=2.10):
    return dict(match_id=mid, date=date, home=home, away=away,
                tab_homeodds=h, tab_awayodds=a)


def _slate(days, pairs):
    return pd.DataFrame({"gameday": pd.to_datetime(days),
                         "home_team": [p[0] for p in pairs],
                         "away_team": [p[1] for p in pairs]})


def test_a_different_meeting_is_never_served(tmp_path, monkeypatch):
    # The archive holds the Jan 4 meeting; the slate wants Jan 6.
    # (00:00Z starts land on the PREVIOUS Eastern date, so a 7pm ET game on
    # Jan 6 is stamped Jan 7 in UTC - the fixtures respect that.)
    path = _hist(tmp_path, [_row("OLD", "2026-01-05T00:00:00Z")])
    fetched = []
    monkeypatch.setattr(mo, "fetch_live", lambda ids: fetched.extend(ids) or pd.DataFrame())
    out = mo.attach_live(_slate(["2026-01-06"], [("BOS", "NYK")]),
                         lookback_days=3650, path=path)
    assert fetched == [] or "OLD" not in fetched
    assert "market_p_home" not in out.columns or out["market_p_home"].isna().all()


def test_the_same_date_meeting_is_served(tmp_path, monkeypatch):
    path = _hist(tmp_path, [_row("TONIGHT", "2026-01-07T00:00:00Z")])
    seen = []

    def fake(ids):
        seen.extend(ids)
        return pd.DataFrame([{"match_id": ids[0], "market_p_home": 0.61,
                              "market_logit": 0.45, "price_utc": None,
                              "stale_minutes": 30.0, "market_home_odds": 1.9,
                              "market_away_odds": 1.9, "market_book": "novig",
                              "market_price_utc": None}])

    monkeypatch.setattr(mo, "fetch_live", fake)
    out = mo.attach_live(_slate(["2026-01-06"], [("BOS", "NYK")]),
                         lookback_days=3650, path=path)
    assert seen == ["TONIGHT"]
    assert out["market_p_home"].iloc[0] == pytest.approx(0.61)


def test_a_back_to_back_serves_tonights_row_not_the_earlier_one(tmp_path, monkeypatch):
    # Jan 4 and Jan 6 both stored: the Jan 6 slate must take the Jan 6 id.
    path = _hist(tmp_path, [_row("OLD", "2026-01-05T00:00:00Z"),
                            _row("TONIGHT", "2026-01-07T00:00:00Z")])
    seen = []

    def fake(ids):
        seen.extend(ids)
        return pd.DataFrame([{"match_id": m, "market_p_home": 0.5,
                              "market_logit": 0.0, "price_utc": None,
                              "stale_minutes": 1.0, "market_home_odds": 1.9,
                              "market_away_odds": 1.9, "market_book": "tab",
                              "market_price_utc": None} for m in ids])

    monkeypatch.setattr(mo, "fetch_live", fake)
    mo.attach_live(_slate(["2026-01-06"], [("BOS", "NYK")]),
                   lookback_days=3650, path=path)
    assert seen == ["TONIGHT"]


def test_a_failed_fetch_leaves_the_game_unpriced(tmp_path, monkeypatch):
    path = _hist(tmp_path, [_row("TONIGHT", "2026-01-07T00:00:00Z")])
    monkeypatch.setattr(mo, "fetch_live", lambda ids: pd.DataFrame(
        [{"match_id": m, "home": None, "away": None, "market_p_home": float("nan"),
          "market_logit": float("nan"), "price_utc": None, "stale_minutes": float("nan"),
          "market_home_odds": float("nan"), "market_away_odds": float("nan"),
          "market_book": None, "market_price_utc": None} for m in ids]))
    out = mo.attach_live(_slate(["2026-01-06"], [("BOS", "NYK")]),
                         lookback_days=3650, path=path)
    assert out["market_p_home"].isna().all()


def test_fetch_live_keeps_its_schema_when_a_request_fails(monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    live = mo.fetch_live(["NHL_2099_1_BOS_NYK"])
    assert "home" in live.columns and "away" in live.columns
    assert live["market_p_home"].isna().all()
    assert live["error"].notna().all()



# ── live round-listing fallback (2026-11-18) ─────────────────────────────
def _live(mid, p=0.61):
    return pd.DataFrame([{"match_id": mid, "market_p_home": p,
                          "market_logit": 0.45, "price_utc": None,
                          "stale_minutes": 30.0, "market_home_odds": 1.9,
                          "market_away_odds": 1.9, "market_book": "novig",
                          "market_price_utc": None}])


def test_a_missing_history_still_resolves_ids_from_the_round_listing(
        tmp_path, monkeypatch):
    """The history is a fast path, not the only id source (NBA parity)."""
    day = pd.Timestamp("2026-01-06")
    monkeypatch.setattr(mo, "live_match_index", lambda **k: (
        {("BOS", "NYK", day): "LIVE"}, {("BOS", "NYK"): "LIVE"}))
    seen = []
    monkeypatch.setattr(mo, "fetch_live",
                        lambda ids: seen.extend(ids) or _live(ids[0]))
    out = mo.attach_live(_slate(["2026-01-06"], [("BOS", "NYK")]),
                         path=str(tmp_path / "missing.csv"))
    assert seen == ["LIVE"]
    assert out["market_p_home"].iloc[0] == pytest.approx(0.61)


def test_an_empty_recent_window_falls_back_too(tmp_path, monkeypatch):
    path = _hist(tmp_path, [_row("OLD", "2026-01-05T00:00:00Z")])
    day = pd.Timestamp("2026-01-06")
    monkeypatch.setattr(mo, "live_match_index", lambda **k: (
        {("BOS", "NYK", day): "LIVE"}, {("BOS", "NYK"): "LIVE"}))
    seen = []
    monkeypatch.setattr(mo, "fetch_live",
                        lambda ids: seen.extend(ids) or _live(ids[0]))
    mo.attach_live(_slate(["2026-01-06"], [("BOS", "NYK")]), path=path,
                   lookback_days=1)
    assert seen == ["LIVE"]


def test_the_live_listing_maps_si_codes_and_eastern_dates(monkeypatch):
    """SI writes NJ/TB; the board writes NJD/TBL, and the day is Eastern."""
    def listing(season, rnd):
        if rnd != 7:
            return []
        return [{"MatchData": {"SIMatchID": "M7",
                               "Date": "2026-01-07T00:00:00Z",
                               "HomeTeam": {"Abv": "NJ"},
                               "AwayTeam": {"Abv": "TB"}}}]

    monkeypatch.setattr(mo, "_round_listing", listing)
    by_day, by_pair = mo.live_match_index(season=2026, max_round=10,
                                          workers=2)
    assert by_day[("NJD", "TBL", pd.Timestamp("2026-01-06"))] == "M7"
    assert by_pair[("NJD", "TBL")] == "M7"



def test_the_sweep_reaches_the_end_of_a_season(monkeypatch):
    """Past NHL seasons end at round 251-259, so the cap must clear 240."""
    def listing(season, rnd):
        if rnd != 240:
            return []
        return [{"MatchData": {"SIMatchID": "M240",
                               "Date": "2027-03-02T00:00:00Z",
                               "HomeTeam": {"Abv": "LA"},
                               "AwayTeam": {"Abv": "SJ"}}}]

    monkeypatch.setattr(mo, "_round_listing", listing)
    by_day, _by_pair = mo.live_match_index(season=2026, workers=4)
    assert by_day[("LAK", "SJS", pd.Timestamp("2027-03-01"))] == "M240"
