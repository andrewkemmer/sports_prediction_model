"""Harness tests for the NFL arm-C fitted scoring rating.

The old arm C recombined four served DVOA legs arithmetically.  The adopted
arm C is the causal two-sided points model in
``nfl-backend/backend/team_score_ratings.py``; these tests pin the harness
surface (seven columns, identity, routing) while the backend test file owns
the causal-fit details.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sport_abc as sa


def _nfl_frame(n: int = 240) -> pd.DataFrame:
    """A small, continuous league with enough decided games to fit."""
    teams = ("A", "B", "C", "D")
    dates = pd.date_range("2020-09-01", periods=n, freq="D")
    rows = []
    for i, date in enumerate(dates):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        # Vary the scoring so the design is not rank-deficient; the exact
        # values are irrelevant to the identity and routing assertions.
        rows.append({
            "game_id": f"g{i:04d}",
            "gameday": date,
            "season": 2020,
            "home_team": home,
            "away_team": away,
            "home_score": 20 + (i % 7),
            "away_score": 17 + ((i * 3) % 7),
        })
    return pd.DataFrame(rows)


def test_nfl_attaches_the_seven_adopted_columns():
    out = sa.attach_team_rating(_nfl_frame(), "nfl")
    expected = {
        "off_rating_diff", "off_rating_home", "off_rating_away",
        "def_rating_diff", "def_rating_home", "def_rating_away",
        "net_rating_diff",
    }
    assert expected <= set(out.columns)
    assert not {"team_rating_diff", "team_rating_home",
                "team_rating_away"} & set(out.columns)
    assert out["net_rating_diff"].notna().any()


def test_nfl_off_def_identity_and_diffs():
    out = sa.attach_team_rating(_nfl_frame(), "nfl")
    rated = out.dropna(subset=["net_rating_diff"])
    assert len(rated) > 0
    net = rated["off_rating_home"] + rated["def_rating_home"]
    assert np.allclose(rated["net_rating_diff"],
                       net - (rated["off_rating_away"]
                              + rated["def_rating_away"]),
                       atol=1e-9)
    assert np.allclose(rated["off_rating_diff"],
                       rated["off_rating_home"] - rated["off_rating_away"],
                       atol=1e-9)
    assert np.allclose(rated["def_rating_diff"],
                       rated["def_rating_home"] - rated["def_rating_away"],
                       atol=1e-9)


def test_nfl_arm_c_is_the_fitted_block_and_a_reconstructs_incumbent():
    mods = sa.load_modules("nfl")
    arms = sa.arm_columns(mods, "nfl")
    fitted = set(sa._fitted_rating_cols("nfl"))
    assert fitted == {
        "off_rating_diff", "off_rating_home", "off_rating_away",
        "def_rating_diff", "def_rating_home", "def_rating_away",
        "net_rating_diff",
    }
    assert fitted <= set(arms["C"])
    assert fitted.isdisjoint(arms["A"])
    assert set(sa._retired_strength_cols("nfl")) <= set(arms["A"])
    assert len(arms["A"]) == len(arms["B"]) + 6
    assert len(arms["C"]) == len(arms["B"]) + 7
