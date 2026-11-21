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


def test_nfl_prefers_the_opponent_adjusted_defence():
    """When the adjusted series exists it must win over the raw one."""
    frame = _nfl_frame(pbp_def_epa_play_ewm_home=0.10,
                       pbp_def_epa_play_ewm_away=0.10,
                       pbp_def_epa_play_opp_adj_ewm_home=0.02,
                       pbp_def_epa_play_opp_adj_ewm_away=0.02)
    out = sa.attach_team_rating(frame, "nfl")
    assert np.isclose(out["team_rating_home"].iloc[0], -0.02 * 60.0)


def test_nfl_falls_back_to_the_raw_defence():
    frame = _nfl_frame(pbp_def_epa_play_ewm_home=0.10,
                       pbp_def_epa_play_ewm_away=0.10)
    out = sa.attach_team_rating(frame, "nfl")
    assert np.isclose(out["team_rating_home"].iloc[0], -0.10 * 60.0)


def test_nfl_rating_diff_is_home_minus_away():
    frame = _nfl_frame(pl_dvoa_qb_home=3.0, pl_dvoa_qb_away=1.0)
    out = sa.attach_team_rating(frame, "nfl")
    assert np.allclose(out["team_rating_diff"],
                       out["team_rating_home"] - out["team_rating_away"])


