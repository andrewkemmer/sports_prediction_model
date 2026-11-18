"""Serve-time quote freshness (2026-11-18).

The 2026-11-17 evening run attached prices before the scoring block, then
served them ~13 minutes later while logging them as "live": MIN shipped at
54.9% no-vig from a quote stamped ``00:03:39Z`` while the market had moved to
~61.5%. ``price_utc`` is the FEED's last-updated instant, so a run can fetch
seconds ago and still serve a quarter-hour-old number.

These pin the three pieces that make that impossible to hide:

* ``_quote_age_minutes`` measures the age of the number a reader would see,
  not the age of our request.
* ``_needs_quote_refresh`` re-pulls an unpriced slate, a slate with no usable
  stamp, and one whose OLDEST quote is past the bar.
* ``_keep_prior_quotes`` keeps the prior number (and its stale stamp) for any
  game the refresh cannot price, instead of blanking a card that had a price.
"""
from __future__ import annotations

import pandas as pd
import pytest

try:  # run from nba-backend/backend, as the pipeline's own docstring says
    from master_pipeline import (
        _keep_prior_quotes,
        _needs_quote_refresh,
        _quote_age_minutes,
    )
    from serving import started_mask
except ImportError:  # run from nba-backend
    from backend.master_pipeline import (
        _keep_prior_quotes,
        _needs_quote_refresh,
        _quote_age_minutes,
    )
    from backend.serving import started_mask


def _frame():
    now = pd.Timestamp.now(tz="UTC")
    return pd.DataFrame(
        {
            "game_id": ["401910003", "401910004"],
            "market_p_home": [0.548743, 0.61],
            "market_logit": [-0.183, 0.447],
            "price_utc": [
                (now - pd.Timedelta(minutes=12.75)).isoformat(),
                (now - pd.Timedelta(minutes=2)).isoformat(),
            ],
            "market_home_odds": [1.83333, 1.62],
            "market_away_odds": [2.17, 2.45],
            "market_book": ["novig", "novig"],
        }
    )


def test_quote_age_reads_the_feed_stamp_not_our_request():
    age = _quote_age_minutes(_frame())
    assert age.iloc[0] == pytest.approx(12.75, abs=0.5)
    assert age.iloc[1] == pytest.approx(2.0, abs=0.5)


def test_quote_age_is_empty_without_a_stamp():
    assert _quote_age_minutes(pd.DataFrame({"game_id": ["x"]})).empty
    assert _quote_age_minutes(None).empty


def test_untouched_quotes_are_reused():
    assert _needs_quote_refresh(True, 2.0, 10.0) is False


def test_a_stale_oldest_quote_forces_a_refresh():
    # The 2026-11-17 case: 12.75 minutes old at serve, bar of 10.
    assert _needs_quote_refresh(True, 12.75, 10.0) is True


def test_an_unpriced_or_unstamped_slate_is_refreshed():
    assert _needs_quote_refresh(False, 2.0, 10.0) is True
    assert _needs_quote_refresh(True, float("nan"), 10.0) is True


def test_refresh_keeps_the_prior_number_where_the_feed_is_silent():
    prior = _frame()
    fresh = pd.DataFrame(
        {
            "game_id": ["401910003", "401910004"],
            "market_p_home": [0.615, float("nan")],
            "price_utc": [
                pd.Timestamp.now(tz="UTC").isoformat(),
                None,
            ],
            "market_book": ["novig", None],
        }
    )
    merged = _keep_prior_quotes(fresh, prior).set_index("game_id")

    # Re-priced game takes the new number.
    assert merged.loc["401910003", "market_p_home"] == pytest.approx(0.615)
    # Game the feed dropped keeps its price AND its old stamp, so the
    # staleness stays visible instead of being disguised as fresh.
    assert merged.loc["401910004", "market_p_home"] == pytest.approx(0.61)
    assert merged.loc["401910004", "market_book"] == "novig"
    age = _quote_age_minutes(merged.reset_index())
    assert age.iloc[1] == pytest.approx(2.0, abs=0.5)


# ── tipoff is the lock boundary, not first publication ───────────────────
# The lock itself lives in the writers (serving.write_moneyline_json /
# write_markets_csv), NHL-style: they read the card they are replacing, so
# "locked" means "the last number published before tip" - a value that is
# always on disk. This predicate is shared by both writers.

def _slate(state, detail="7:00 PM ET", tip="2026-11-18T01:00:00Z"):
    return pd.DataFrame(
        {
            "game_id": ["401910003"],
            "gameday": ["2026-11-17"],
            "p_ensemble_calibrated": [0.61],
            "p_ensemble": [0.61],
            "home_win_prob_model": [0.61],
            "game_state": [state],
            "game_status_detail": [detail],
            "start_time_utc": [tip],
        }
    )


def test_a_row_with_no_state_and_no_tip_counts_as_started():
    # NHL parity: the conservative default. Re-pricing a live game is the
    # failure the lock exists to prevent, so an unplaceable row is frozen.
    assert bool(started_mask(_slate(None, "", tip=None))[0]) is True


def test_state_beats_the_clock_but_the_clock_covers_legacy_rows():
    # A delayed game still reads `pre` past its tip: not started.
    assert bool(started_mask(_slate("pre"))[0]) is False
    # No state at all: the clock decides.
    legacy = _slate(None, "", tip="2020-01-01T00:00:00Z")
    assert bool(started_mask(legacy)[0]) is True
    future = _slate(None, "", tip="2099-01-01T00:00:00Z")
    assert bool(started_mask(future)[0]) is False
