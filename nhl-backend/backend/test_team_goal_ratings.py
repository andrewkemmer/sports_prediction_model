"""Tests for the fitted NHL two-sided goal-rating panel.

The properties that matter are causal and dimensional: a rating attached to a
game must never see that game, the target must be goals per 60 minutes of team
ice time, and the seven columns must satisfy the documented identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import team_goal_ratings as tgr


def _frame(seasons=(2024,), per_season: int = 120, teams: int = 8,
           strong_team: str = "T0", strong_goals: float = 6.0) -> pd.DataFrame:
    names = [f"T{i}" for i in range(teams)]
    rng = np.random.default_rng(13)
    rows = []
    pk = 0
    for season in seasons:
        start = pd.Timestamp(f"{season}-10-08")
        for day in range(per_season // 2):
            for pair in range(teams // 2):
                home = names[(pair * 2 + day) % teams]
                away = names[(pair * 2 + day + 1) % teams]
                if home == away:
                    continue
                pk += 1
                hg = strong_goals if home == strong_team else 3.0
                ag = strong_goals if away == strong_team else 3.0
                rows.append({
                    "game_pk": pk,
                    "gameday": start + pd.Timedelta(days=day),
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_score": float(hg + rng.normal(0, 0.3)),
                    "away_score": float(ag + rng.normal(0, 0.3)),
                    "rest_days_home": 2.0, "rest_days_away": 2.0,
                    "back_to_back_home": 0.0, "back_to_back_away": 0.0,
                    "is_playoffs": 0.0,
                })
    return pd.DataFrame(rows)


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    return tgr.build_causal_panel(frame, refit_days=7, min_train_games=40)


def test_target_is_goals_per_sixty():
    frame = _frame(per_season=4)
    rows = tgr.build_side_rows(frame)
    first = rows[rows["pos"] == 0].iloc[0]
    # A regulation game is 60 minutes, so per-60 equals goals scored here.
    assert np.isclose(first["rating"], frame["home_score"].iloc[0])


def test_toi_column_actually_normalises():
    frame = _frame(per_season=4)
    frame["toi_minutes_home"] = 50.0
    rows = tgr.build_side_rows(frame)
    first = rows[rows["pos"] == 0].iloc[0]
    assert np.isclose(first["rating"], 60.0 * frame["home_score"].iloc[0] / 50.0)


def test_toi_clip_falls_back_to_regulation():
    frame = _frame(per_season=4)
    frame["toi_minutes_home"] = 5.0          # implausible, gets clipped
    minutes = tgr._team_minutes(frame, "home")
    assert np.allclose(minutes.to_numpy(), tgr.REGULATION_MINUTES)


def test_game_level_playoff_flag_is_used_once_not_per_side():
    frame = _frame(per_season=4)
    frame["is_playoffs"] = 1.0
    rows = tgr.build_side_rows(frame)
    assert "is_playoffs" in rows.columns
    assert rows["is_playoffs"].notna().all()
    assert np.allclose(rows["is_playoffs"].to_numpy(), 1.0)


def test_seven_columns_and_identity():
    panel = _panel(_frame())
    assert list(panel.columns) == tgr.GOAL_RATING_COLS
    assert len(tgr.GOAL_RATING_COLS) == 7
    got = panel.dropna(subset=["net_rating_diff"])
    assert len(got) > 0
    err = (got["off_rating_diff"] + got["def_rating_diff"]
           - got["net_rating_diff"]).abs().max()
    assert err < 1e-9


def test_future_games_cannot_change_past_ratings():
    frame = _frame(per_season=200)
    cut = pd.Timestamp("2024-12-01")
    assert (frame["gameday"] < cut).any() and (frame["gameday"] >= cut).any()
    full = _panel(frame)
    kept = frame[frame["gameday"] < cut].reset_index(drop=True)
    truncated = _panel(kept)
    past = full.loc[(frame["gameday"] < cut).to_numpy()].reset_index(drop=True)
    assert len(truncated) == len(past)
    for column in tgr.GOAL_RATING_COLS:
        assert np.allclose(past[column].to_numpy(),
                           truncated[column].to_numpy(),
                           equal_nan=True), column


def test_fit_refuses_training_rows_at_or_after_the_anchor():
    frame = _frame(per_season=40)
    rows = tgr.build_side_rows(frame)
    anchor = pd.Timestamp(rows["gameday"].iloc[10])
    leaky = rows.copy()
    leaky.loc[leaky.index[0], "gameday"] = anchor
    with pytest.raises(ValueError, match="anchor"):
        tgr._fit_once(leaky, ["T0", "T1"], anchor, tgr.ALPHA, tgr.HALF_LIFE_DAYS)


def test_strongest_offence_gets_the_highest_rating():
    frame = _frame(per_season=200, strong_team="T0", strong_goals=6.0)
    panel = _panel(frame)
    means = {}
    for idx, row in frame.iterrows():
        if np.isfinite(panel.loc[idx, "off_rating_home"]):
            means.setdefault(row["home_team"], []).append(
                panel.loc[idx, "off_rating_home"])
            means.setdefault(row["away_team"], []).append(
                panel.loc[idx, "off_rating_away"])
    means = {t: float(np.mean(v)) for t, v in means.items() if len(v) >= 10}
    assert means
    assert max(means, key=means.get) == "T0", means


def test_missing_score_column_raises():
    frame = _frame(per_season=4).drop(columns=["home_score"])
    with pytest.raises(ValueError, match="score column"):
        tgr.build_side_rows(frame)


def test_attach_keeps_row_order():
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=9).reset_index(drop=True)
    out = tgr.attach_goal_ratings(shuffled, refit_days=7, min_train_games=40)
    assert np.array_equal(out["game_pk"].to_numpy(),
                          shuffled["game_pk"].to_numpy())
    assert set(tgr.GOAL_RATING_COLS) <= set(out.columns)
