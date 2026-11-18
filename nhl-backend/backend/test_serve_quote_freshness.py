"""Serve-time quote freshness (2026-11-18).

The 2026-11-17 NHL run attached BOS/LAK at 23:21:03 and published the board at
23:34:13 - thirteen minutes of drift behind a card that said "priced live" and
carried no age (the NBA board had the same hole the same night). The feed's
own stamp is what ``market_price_utc`` records (the display book's), so a run can
fetch seconds ago and still serve a quarter-hour-old number.

These pin the three pieces that make that impossible to hide:

* ``quote_age_minutes`` measures the age of the TAB quote that FEEDS THE
  OFFSET (``price_utc``), not the no-vig display stamp and not the age of
  our request - TAB is the training series, so ``market_logit`` is TAB's
  de-vig and that is the number whose staleness matters.
* ``needs_quote_refresh`` re-pulls an unpriced slate, a slate with no usable
  stamp, and one whose OLDEST quote is past the bar.
* ``keep_prior_quotes`` keeps the prior number (and its stale stamp) for any
  game the refresh cannot price, instead of blanking a card that had a price.
"""
from __future__ import annotations

import pandas as pd
import pytest

from market_offset import (
    keep_prior_quotes,
    needs_quote_refresh,
    quote_age_minutes,
)


def _frame():
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame(
        {
            "game_id": ["2026020333", "2026020334"],
            "market_p_home": [0.576, 0.45],
            "market_logit": [0.306, -0.2],
            # TAB is the offset book and the gate's clock; a priced frame
            # always carries its stamp (attach_live always sets the column).
            "price_utc": [
                (now - pd.Timedelta(minutes=13.2)).isoformat(),
                (now - pd.Timedelta(minutes=2)).isoformat(),
            ],
            "market_price_utc": [
                (now - pd.Timedelta(minutes=13.2)).isoformat(),
                (now - pd.Timedelta(minutes=2)).isoformat(),
            ],
            "market_home_odds": [1.65289, 2.2],
            "market_away_odds": [2.5, 1.72],
            "market_book": ["novig", "novig"],
        }
    )


def test_quote_age_reads_the_feed_stamp_not_our_request():
    age = quote_age_minutes(_frame())
    assert age.iloc[0] == pytest.approx(13.2, abs=0.5)
    assert age.iloc[1] == pytest.approx(2.0, abs=0.5)


def test_quote_age_reads_price_utc_when_it_is_the_only_stamp():
    now = pd.Timestamp.now(tz="UTC")
    frame = pd.DataFrame({"price_utc": [(now - pd.Timedelta(minutes=4)).isoformat()]})
    assert quote_age_minutes(frame).iloc[0] == pytest.approx(4.0, abs=0.5)


def test_the_tab_offset_stamp_wins_over_the_novig_display_stamp():
    # TAB feeds market_logit (the offset input); novig is display only and
    # moves on its own clock. Gating on the display stamp would refresh on
    # the wrong book - or miss a stale offset input entirely.
    now = pd.Timestamp.now(tz="UTC")
    frame = pd.DataFrame({
        "price_utc": [(now - pd.Timedelta(minutes=14)).isoformat()],
        "market_price_utc": [(now - pd.Timedelta(minutes=1)).isoformat()],
    })
    assert quote_age_minutes(frame).iloc[0] == pytest.approx(14.0, abs=0.5)


def test_quote_age_is_empty_without_a_stamp():
    assert quote_age_minutes(pd.DataFrame({"game_id": ["x"]})).empty
    assert quote_age_minutes(None).empty


def test_untouched_quotes_are_reused():
    assert needs_quote_refresh(True, 2.0, 10.0) is False


def test_a_stale_oldest_quote_forces_a_refresh():
    # The 2026-11-17 case: 13.2 minutes old at serve, bar of 10.
    assert needs_quote_refresh(True, 13.2, 10.0) is True


def test_an_unpriced_or_unstamped_slate_is_refreshed():
    assert needs_quote_refresh(False, 2.0, 10.0) is True
    assert needs_quote_refresh(True, float("nan"), 10.0) is True


def test_refresh_keeps_the_prior_number_where_the_feed_is_silent():
    prior = _frame()
    fresh = pd.DataFrame(
        {
            "game_id": ["2026020333", "2026020334"],
            "market_p_home": [0.60, float("nan")],
            "price_utc": [pd.Timestamp.now(tz="UTC").isoformat(), None],
            "market_price_utc": [pd.Timestamp.now(tz="UTC").isoformat(), None],
            "market_book": ["novig", None],
        }
    )
    merged = keep_prior_quotes(fresh, prior).set_index("game_id")

    # Re-priced game takes the new number.
    assert merged.loc["2026020333", "market_p_home"] == pytest.approx(0.60)
    # Game the feed dropped keeps its price AND its old stamp, so the
    # staleness stays visible instead of being disguised as fresh.
    assert merged.loc["2026020334", "market_p_home"] == pytest.approx(0.45)
    assert merged.loc["2026020334", "market_book"] == "novig"
    age = quote_age_minutes(merged.reset_index())
    assert age.iloc[1] == pytest.approx(2.0, abs=0.5)



# ── two books, two clocks (2026-11-18) ───────────────────────────────────
def test_the_card_stamps_and_ages_each_book_at_publish(tmp_path):
    import numpy as np
    import serving

    now = pd.Timestamp.now(tz="UTC")
    tab = (now - pd.Timedelta(minutes=20)).isoformat()
    nov = (now - pd.Timedelta(minutes=4)).isoformat()
    slate = pd.DataFrame([{
        "game_id": "g1", "gameday": pd.Timestamp("2026-10-30"),
        "home_team": "BOS", "away_team": "LAK",
        "game_state": "pre", "game_status_detail": "7:30 PM ET",
        "start_time_utc": "2099-01-01T00:00:00Z",
        "home_win_prob_model": 0.6,
        "market_home_odds": 1.8, "market_away_odds": 2.1,
        "market_book": "novig", "price_utc": tab, "market_price_utc": nov}])
    rec = serving.write_moneyline_json(
        tmp_path / "c.json", slate, np.array([0.6]), np.array([0.6]),
        {"BOS": "Bruins", "LAK": "Kings"}, {}, run_date="2026-10-30")
    g = rec["games"][0]
    assert g["market_price_utc"].startswith(nov[:19])
    assert g["offset_price_utc"].startswith(tab[:19])
    assert g["market_stale_minutes"] == pytest.approx(4.0, abs=0.5)
    assert g["offset_stale_minutes"] == pytest.approx(20.0, abs=0.5)
