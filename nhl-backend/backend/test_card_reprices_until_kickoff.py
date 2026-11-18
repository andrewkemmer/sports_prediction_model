"""The card re-prices until the puck drops (2026-11-18).

``write_moneyline_json`` keeps a PRIOR row's published probability only once
the game has started (``_kickoff_passed``); a pre-game card always takes the
newest run's number. That is the "no freeze until the start of the game"
contract the board promises in its banner, and it is what the NBA board had
wrong until 2026-11-18 (it froze every previously published game, so a
pre-tip re-run could not move a number).

These pin the gate itself, since it is the single decision the writer makes
between "publish this run's price" and "keep the locked pre-game record".
"""
from __future__ import annotations

import pandas as pd
import pytest

from serving import _kickoff_passed

NOW = pd.Timestamp("2026-11-18T00:10:00Z")
PAST = "2026-11-18T00:00:00Z"
FUTURE = "2026-11-18T01:00:00Z"


def test_a_future_kickoff_reprices_even_when_a_prior_row_exists():
    row = {"game_state": "FUT", "start_time_utc": FUTURE}
    prior = {"game_state": "FUT", "start_time_utc": FUTURE}
    assert _kickoff_passed(row, prior, NOW) is False


def test_a_delayed_game_still_reprices_past_its_scheduled_time():
    # State beats the clock: a slipped puck drop is past its start instant
    # but has not begun, so the card must keep moving.
    row = {"game_state": "PRE", "start_time_utc": PAST}
    assert _kickoff_passed(row, {}, NOW) is False


def test_a_started_game_freezes_at_its_last_pregame_number():
    row = {"game_state": "LIVE", "start_time_utc": PAST}
    assert _kickoff_passed(row, {}, NOW) is True


def test_without_a_state_the_scheduled_instant_decides():
    assert _kickoff_passed({"start_time_utc": FUTURE}, {}, NOW) is False
    assert _kickoff_passed({"start_time_utc": PAST}, {}, NOW) is True


def test_with_neither_state_nor_clock_the_row_counts_as_started():
    # Repricing a live game is the failure this guard exists to prevent.
    assert _kickoff_passed({}, {}, NOW) is True


@pytest.mark.parametrize("state", ["FUT", "PRE", "PPD"])
def test_every_not_started_state_reprices(state):
    assert _kickoff_passed({"game_state": state, "start_time_utc": PAST},
                           {}, NOW) is False
