"""Unit tests for the NFL and NHL arm-C rating construction.

These pin the two things the old C arm got wrong: NFL summed units (not a
four-way mean) with the defence converted from EPA/play to EPA/game, and NHL
weights that follow who is actually on the ice instead of averaging the eight
position/situation cells equally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sport_abc as sa


def _frame(n: int = 2, **columns) -> pd.DataFrame:
    base = {"gameday": pd.date_range("2025-09-01", periods=n, freq="D")}
    base.update({k: np.full(n, v, dtype=float) if np.isscalar(v) else v
                 for k, v in columns.items()})
    return pd.DataFrame(base)


def _nfl_frame(**over) -> pd.DataFrame:
    cols = {f"pl_dvoa_{p}_{s}": 0.0
            for p in ("qb", "wr", "te", "rb") for s in ("home", "away")}
    cols.update({"pbp_def_epa_play_ewm_home": 0.0,
                 "pbp_def_epa_play_ewm_away": 0.0,
                 "pace_plays_min_home": 1.0, "pace_plays_min_away": 1.0})
    cols.update(over)
    return _frame(**cols)


def _nhl_frame(**over) -> pd.DataFrame:
    cols = {f"pl_{sit}_{p}_{side}": 0.0
            for sit in ("evo", "ppo") for p in ("c", "l", "r", "d")
            for side in ("home", "away")}
    cols.update({"goalie_sv_pct_home": 0.900, "goalie_sv_pct_away": 0.900,
                 "shots_against_per_game_home": 0.0,
                 "shots_against_per_game_away": 0.0})
    cols.update(over)
    return _frame(**cols)


# ---------------------------------------------------------------- NFL

def test_nfl_sums_the_four_dvoa_legs():
    """A 10-EPA quarterback must not be diluted to one quarter of himself."""
    frame = _nfl_frame(pl_dvoa_qb_home=10.0, pl_dvoa_wr_home=1.0,
                       pl_dvoa_te_home=1.0, pl_dvoa_rb_home=1.0)
    out = sa.attach_team_rating(frame, "nfl")
    assert out["team_rating_home"].iloc[0] == 13.0
    assert out["team_rating_home"].iloc[0] != 3.25  # the old mean()


def test_nfl_defensive_epa_play_becomes_epa_per_game():
    frame = _nfl_frame(pbp_def_epa_play_ewm_home=0.1,
                       pace_plays_min_home=1.0, pace_plays_min_away=1.0)
    out = sa.attach_team_rating(frame, "nfl")
    # 0.1 EPA/play over 60 plays is 6.0 EPA/game, not 0.1.
    assert out["team_rating_home"].iloc[0] == -6.0


def test_nfl_defence_volume_is_the_opponents_pace():
    frame = _nfl_frame(pbp_def_epa_play_ewm_home=0.1,
                       pbp_def_epa_play_ewm_away=0.1,
                       pace_plays_min_home=1.0, pace_plays_min_away=1.2)
    out = sa.attach_team_rating(frame, "nfl")
    assert np.isclose(out["team_rating_home"].iloc[0], -0.1 * 72.0)
    # The away defence faces the home offence's 60 plays.
    assert np.isclose(out["team_rating_away"].iloc[0], -0.1 * 60.0)


def test_nfl_rating_diff_is_home_minus_away():
    frame = _nfl_frame(pl_dvoa_qb_home=3.0, pl_dvoa_qb_away=1.0)
    out = sa.attach_team_rating(frame, "nfl")
    assert np.allclose(out["team_rating_diff"],
                       out["team_rating_home"] - out["team_rating_away"])


# ---------------------------------------------------------------- NHL

def test_nhl_uniform_rates_give_five_skaters_worth():
    cols = {f"pl_{sit}_{p}_{side}": 1.0
            for sit in ("evo", "ppo") for p in ("c", "l", "r", "d")
            for side in ("home", "away")}
    out = sa.attach_team_rating(_nhl_frame(**cols), "nhl")
    # Five skaters on the ice, all at 1.0 xPM/60, no goalie edge.
    assert np.isclose(out["team_rating_home"].iloc[0], 5.0)


def test_nhl_power_play_does_not_get_half_the_weight():
    over = {**{f"pl_evo_{p}_home": 0.0 for p in ("c", "l", "r", "d")},
            **{f"pl_ppo_{p}_home": 1.0 for p in ("c", "l", "r", "d")}}
    out = sa.attach_team_rating(_nhl_frame(**over), "nhl")
    got = out["team_rating_home"].iloc[0]
    assert np.isclose(got, 0.12 * 5.0)
    assert not np.isclose(got, 0.5)  # what the equal-weight mean produced


def test_nhl_defence_is_two_fifths_of_even_strength():
    over = {f"pl_evo_{p}_home": 0.0 for p in ("c", "l", "r")}
    over["pl_evo_d_home"] = 1.0
    out = sa.attach_team_rating(_nhl_frame(**over), "nhl")
    # 5 skaters x (2/5 defence weight) x 1.0, at 88% of the ice.
    assert np.isclose(out["team_rating_home"].iloc[0], 0.88 * 5.0 * 0.4)


def test_nhl_goalie_goals_saved_above_league():
    over = {"goalie_sv_pct_home": [0.900, 0.930],
            "goalie_sv_pct_away": [0.900, 0.930],
            "shots_against_per_game_home": 30.0}
    out = sa.attach_team_rating(_nhl_frame(**over), "nhl")
    # First date has no prior league history, so it falls back to 0.900.
    assert np.isclose(out["team_rating_home"].iloc[0], 0.0)
    # Second date's league mean is the first date's 0.900.
    assert np.isclose(out["team_rating_home"].iloc[1], 0.03 * 30.0)


def test_nhl_availability_flows_through_the_served_starter():
    """The goalie term must move when the expected starter changes."""
    over = {"goalie_sv_pct_home": [0.900, 0.930],
            "goalie_sv_pct_away": [0.900, 0.880],
            "shots_against_per_game_home": 30.0,
            "shots_against_per_game_away": 30.0}
    out = sa.attach_team_rating(_nhl_frame(**over), "nhl")
    assert out["team_rating_home"].iloc[1] > out["team_rating_away"].iloc[1]
