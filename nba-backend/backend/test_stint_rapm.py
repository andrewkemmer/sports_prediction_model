"""Tests for stint_rapm: point-in-time cuts, O/D identification, SEs, prior."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stint_rapm as sr


def _stints():
    # Two games, fixed fives: H1..H5 always outscore A1..A5.
    rows = []
    for day, hs, aws in [("2024-01-01", 12.0, 8.0),
                         ("2024-01-02", 14.0, 9.0)]:
        rows.append({
            "game_key": f"g{day}", "gameday": pd.Timestamp(day),
            "period": 1, "home_team": "H", "away_team": "A",
            "home_5": ["H1", "H2", "H3", "H4", "H5"],
            "away_5": ["A1", "A2", "A3", "A4", "A5"],
            "minutes": 10.0, "home_pts": hs, "away_pts": aws})
    return pd.DataFrame(rows)


def test_two_rows_per_stint_with_rates():
    rows = sr.stint_rows(_stints())
    assert len(rows) == 4
    # 12 pts / 10 min * 48 = 57.6 for the home attack row.
    home = rows[rows.side == 1].iloc[0]
    assert home.y == abs(home.y) and abs(home.y - 57.6) < 1e-9
    assert home.w == abs(home.w) and abs(home.w - 10.0) < 1e-9
    assert set(home.off_cols) == {"H1", "H2", "H3", "H4", "H5"}
    assert set(home.def_cols) == {"A1", "A2", "A3", "A4", "A5"}


def test_zero_minute_stints_skipped():
    st = _stints()
    st.loc[0, "minutes"] = 0.0
    rows = sr.stint_rows(st)
    assert len(rows) == 2  # only the second stint's pair survives


def test_advance_is_strictly_prior():
    rows = sr.stint_rows(_stints())
    design = sr._StintDesign(rows, lam=4.0)
    design.advance(pd.Timestamp("2024-01-02"))
    assert design.nrows == 2  # only the day-1 pair, never the target day
    design.advance(pd.Timestamp("2024-01-03"))
    assert design.nrows == 4


def _varied_stints():
    # Lineup VARIATION is what identifies offense: H2-H5 only appear in
    # the 20-point outburst, H6-H9 only in the 4-point dud (same
    # opponents, same minutes). A design that cannot separate them
    # from the intercept is the game-level collinearity trap.
    away = ["A1", "A2", "A3", "A4", "A5"]
    return pd.DataFrame([
        {"game_key": "g1", "gameday": pd.Timestamp("2024-01-01"),
         "period": 1, "home_team": "H", "away_team": "A",
         "home_5": ["H1", "H2", "H3", "H4", "H5"], "away_5": away,
         "minutes": 10.0, "home_pts": 20.0, "away_pts": 5.0},
        {"game_key": "g1", "gameday": pd.Timestamp("2024-01-01"),
         "period": 2, "home_team": "H", "away_team": "A",
         "home_5": ["H1", "H6", "H7", "H8", "H9"], "away_5": away,
         "minutes": 10.0, "home_pts": 4.0, "away_pts": 5.0},
    ])


def test_offense_identified_through_lineup_variation():
    rows = sr.stint_rows(_varied_stints())
    design = sr._StintDesign(rows, lam=0.5)
    design.advance(None)
    solved = design.solve()
    assert solved is not None
    hot = solved["off"].loc[["H2", "H3", "H4", "H5"]].mean()
    cold = solved["off"].loc[["H6", "H7", "H8", "H9"]].mean()
    assert hot > cold
    # Net sorts the same way: the outburst five outranks the dud five.
    assert (solved["net"].loc[["H2", "H3", "H4", "H5"]].mean()
            > solved["net"].loc[["H6", "H7", "H8", "H9"]].mean())


def test_standard_errors_finite_and_shrink_with_evidence():
    rows = sr.stint_rows(_stints())
    design = sr._StintDesign(rows, lam=4.0)
    design.advance(None)
    s1 = design.solve()
    assert np.isfinite(s1["off_se"]).all()
    assert np.isfinite(s1["net_se"]).all()
    assert (s1["off_se"] > 0).all()
    # Ten times the evidence: the same betas, tighter bars.
    big = pd.concat([rows] * 10, ignore_index=True)
    design2 = sr._StintDesign(big, lam=4.0)
    design2.advance(None)
    s2 = design2.solve()
    assert (s2["off_se"] < s1["off_se"]).all()


def test_carryover_blends_prev_beta_with_position_mean():
    rows = sr.stint_rows(_stints())
    designs = {"2023-24": sr._StintDesign(rows, lam=1.0)}
    pos_of = {p: ("G" if p.startswith("H") else "C")
              for p in designs["2023-24"].players}
    m, info = sr._carryover_prior(designs, rows, "2024-25",
                                  carry_w=0.5, pos_of=pos_of)
    assert info["from_season"] == "2023-24"
    assert len(m) == 2 * len(designs["2023-24"].players)
    # A rookie (no previous evidence) centers exactly on his position mean.
    assert "G" in info["pos_off"] and "C" in info["pos_def"]


def test_prior_centers_the_ridge():
    # With overwhelming lambda the betas collapse onto the prior mean.
    rows = sr.stint_rows(_stints())
    design = sr._StintDesign(rows, lam=1e9)
    design.set_prior(np.full(2 * design.n, 3.0))
    design.advance(None)
    solved = design.solve()
    assert np.allclose(solved["off"].to_numpy(), 3.0, atol=1e-3)
    assert np.allclose(solved["def"].to_numpy(), 3.0, atol=1e-3)
