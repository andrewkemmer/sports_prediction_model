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


def test_eight_columns_and_identity():
    panel = _panel(_frame())
    assert list(panel.columns) == tgr.GOAL_RATING_COLS
    assert len(tgr.GOAL_RATING_COLS) == 8
    got = panel.dropna(subset=["net_rating_diff"])
    assert len(got) > 0
    err = (got["off_rating_diff"] + got["def_rating_diff"]
           - got["net_rating_diff"]).abs().max()
    assert err < 1e-9
    # The adopted game-level sum is symmetric and built from the same panel.
    sum_err = (got["off_rating_home"] + got["def_rating_home"]
               + got["off_rating_away"] + got["def_rating_away"]
               - got["net_rating_sum"]).abs().max()
    assert sum_err < 1e-9


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


# ---------------------------------------------------------------------------
# The two parity blocks added for the MLB comparison: the opposing goaltender
# and the shrunk prior-season goals.  Their defaults must be ON; ``blocks=()``
# must reproduce the thin design the first NHL A/B measured.


def _with_goalies(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for side, sv, gaa, starts in (("home", 0.900, 3.00, 40.0),
                                  ("away", 0.930, 2.20, 12.0)):
        out[f"goalie_sv_pct_{side}"] = sv
        out[f"goalie_gaa_{side}"] = gaa
        out[f"goalie_starts_{side}"] = starts
    return out


def test_opposing_goalie_block_reads_the_opponent():
    rows = tgr.build_side_rows(_with_goalies(_frame(per_season=4)))
    # ``pos`` repeats across the home and away halves of the concat, so the
    # side flag is what selects the row, not the position alone.
    home = rows[(rows["pos"] == 0) & (rows["is_home"] == 1.0)].iloc[0]
    away = rows[(rows["pos"] == 0) & (rows["is_home"] == 0.0)].iloc[0]
    # The scoring team sees the OTHER side's goalie.
    assert np.isclose(home["opp_goalie_sv_pct"], 0.930)
    assert np.isclose(home["opp_goalie_gaa"], 2.20)
    assert np.isclose(home["opp_goalie_starts"], 12.0)
    assert np.isclose(away["opp_goalie_sv_pct"], 0.900)
    assert np.isclose(away["opp_goalie_starts"], 40.0)


def test_prior_block_is_last_season_shrunk_toward_the_league():
    frame = _frame(seasons=(2024, 2025), per_season=120)
    # Make T0 a scoring outlier in 2024 only, so its 2025 prior must stand out.
    frame.loc[(frame["season"] == 2024) & (frame["home_team"] == "T0"),
              "home_score"] += 3.0
    frame.loc[(frame["season"] == 2024) & (frame["away_team"] == "T0"),
              "away_score"] += 3.0
    prior = tgr._prior_goals(frame)
    # Each played season contributes a prior for the FOLLOWING season; the
    # 2026 rows are simply never joined to anything.
    assert set(prior["season"].unique()) == {2025.0, 2026.0}
    t0 = prior[(prior["season"] == 2025.0) & (prior["team"] == "T0")].iloc[0]
    other = prior[(prior["season"] == 2025.0)
                  & (prior["team"] == "T1")].iloc[0]
    assert t0["own_p_gf"] > other["own_p_gf"]

    rows = tgr.build_side_rows(frame)
    assert rows.loc[rows["season"] == 2024, "own_p_gf"].isna().all()
    got = rows[(rows["season"] == 2025) & (rows["team"] == "T0")]
    assert np.isclose(got["own_p_gf"].iloc[0], t0["own_p_gf"])
    assert np.isclose(got["own_p_ga"].iloc[0], t0["own_p_ga"])


def test_blocks_control_the_numeric_design():
    rows = tgr.build_side_rows(_with_goalies(_frame(per_season=8)))
    thin = tgr._numeric_columns(rows, ())
    full = tgr._numeric_columns(rows, tgr.DESIGN_BLOCKS)
    assert thin == ["rest_days", "back_to_back", "is_playoffs", "is_home"]
    added = set(full) - set(thin)
    assert {f"off_{n}" for n in tgr.OFF_COMPONENTS} <= added
    assert "off_pool_xpm" in added
    assert {"opp_goalie_sv_pct", "opp_goalie_gaa", "opp_goalie_starts"} <= added
    assert {"own_p_gf", "own_p_ga", "opp_p_gf", "opp_p_ga"} <= added
    assert set(tgr.DESIGN_BLOCKS) == {"components", "opp_goalie", "prior"}


def test_full_default_design_is_the_mlb_shape():
    rows = tgr.build_side_rows(_with_goalies(_frame(per_season=8)))
    full = tgr._numeric_columns(rows, tgr.DESIGN_BLOCKS)
    # 9 offence-form columns (8 counters + the pool mean), 3 opposing goalie,
    # 4 prior-season, 4 schedule/home context -- MLB's block counts, with
    # hockey's schedule context standing in for park/weather.
    assert len(full) == 9 + 3 + 4 + 4


def test_components_block_reads_the_teams_own_offence():
    frame = _frame(per_season=4)
    for name, home_value, away_value in (("ewm_net_goals", 0.8, -0.3),
                                         ("ewm_goal_share", 0.55, 0.45),
                                         ("shots_for_per_game", 32.0, 27.0),
                                         ("pp_success", 0.24, 0.16),
                                         ("nhl_pp_goals_pg_ewm", 0.8, 0.5),
                                         ("faceoff_win", 0.52, 0.48),
                                         ("nhl_giveaways_pg_ewm", 8.0, 11.0),
                                         ("nhl_takeaways_pg_ewm", 9.0, 6.0)):
        frame[f"{name}_home"] = home_value
        frame[f"{name}_away"] = away_value
    frame["pl_xpm_home"] = 0.6
    frame["pl_xpm_away"] = -0.4
    rows = tgr.build_side_rows(frame)
    home = rows[(rows["pos"] == 0) & (rows["is_home"] == 1.0)].iloc[0]
    away = rows[(rows["pos"] == 0) & (rows["is_home"] == 0.0)].iloc[0]
    assert np.isclose(home["off_ewm_net_goals"], 0.8)
    assert np.isclose(away["off_shots_for_per_game"], 27.0)
    assert np.isclose(home["off_pool_xpm"], 0.6)
    assert np.isclose(away["off_pool_xpm"], -0.4)


def test_pool_xpm_falls_back_to_position_cells():
    frame = _frame(per_season=4)
    cells = {"c": 0.4, "l": 0.2, "r": 0.0, "d": -0.2}
    for pos, value in cells.items():
        frame[f"pl_xpm_{pos}_home"] = value
        frame[f"pl_xpm_{pos}_away"] = value + 1.0
    rows = tgr.build_side_rows(frame)
    home = rows[(rows["pos"] == 0) & (rows["is_home"] == 1.0)].iloc[0]
    assert np.isclose(home["off_pool_xpm"], 0.1)


def test_own_goalie_block_is_opt_in():
    rows = tgr.build_side_rows(_with_goalies(_frame(per_season=8)))
    off = tgr._numeric_columns(rows, tgr.DESIGN_BLOCKS, own_goalie=False)
    on = tgr._numeric_columns(rows, tgr.DESIGN_BLOCKS, own_goalie=True)
    assert set(on) - set(off) == {f"own_{n}" for n in tgr.OWN_GOALIE_COMPONENTS}
    home = rows[(rows["pos"] == 0) & (rows["is_home"] == 1.0)].iloc[0]
    assert np.isclose(home["own_goalie_sv_pct"], 0.900)
    assert tgr.OWN_GOALIE is False


def test_poisson_likelihood_runs_the_whole_panel():
    frame = _with_goalies(_frame(seasons=(2024, 2025), per_season=120))
    panel = tgr.build_causal_panel(frame, refit_days=7, min_train_games=40,
                                   likelihood="poisson")
    assert panel["net_rating_diff"].notna().mean() > 0.5
    assert np.isfinite(panel["net_rating_diff"].dropna()).all()


def test_unknown_likelihood_is_rejected():
    rows = tgr.build_side_rows(_frame(per_season=8))
    teams = sorted(set(rows["team"]))
    anchor = pd.Timestamp(rows["gameday"].quantile(0.7))
    train = rows[rows["gameday"] < anchor]
    with pytest.raises(ValueError, match="likelihood"):
        tgr._fit_once(train, teams, anchor, 50.0, 120.0, likelihood="tweedie")


def test_defaults_are_the_shipped_configuration():
    frame = _with_goalies(_frame(seasons=(2024, 2025), per_season=120))
    default = tgr.build_causal_panel(frame, refit_days=7, min_train_games=40)
    explicit = tgr.build_causal_panel(
        frame, refit_days=7, min_train_games=40,
        alpha=tgr.ALPHA, half_life_days=tgr.HALF_LIFE_DAYS,
        blocks=tgr.DESIGN_BLOCKS, likelihood=tgr.LIKELIHOOD,
        own_goalie=tgr.OWN_GOALIE)
    for column in tgr.GOAL_RATING_COLS:
        assert np.allclose(default[column].to_numpy(),
                           explicit[column].to_numpy(), equal_nan=True), column


def test_delivery_frame_shape_attaches_without_gameday_or_season():
    """Production's frame names the date game_date and carries no season."""
    frame = _with_goalies(_frame(seasons=(2024, 2025))).rename(
        columns={"gameday": "game_date"}).drop(columns=["season"])
    out = tgr.attach_goal_ratings(frame, refit_days=7, min_train_games=40)
    assert out["net_rating_diff"].notna().mean() > 0.5
    reference = tgr.attach_goal_ratings(
        _with_goalies(_frame(seasons=(2024, 2025))), refit_days=7,
        min_train_games=40)
    assert np.allclose(out["net_rating_diff"].to_numpy(),
                       reference["net_rating_diff"].to_numpy(),
                       equal_nan=True)


def test_default_panel_is_the_full_design():
    frame = _with_goalies(_frame(seasons=(2024, 2025), per_season=120))
    default = tgr.build_causal_panel(frame, refit_days=7, min_train_games=40)
    explicit = tgr.build_causal_panel(
        frame, refit_days=7, min_train_games=40, blocks=tgr.DESIGN_BLOCKS)
    thin = tgr.build_causal_panel(frame, refit_days=7, min_train_games=40,
                                  blocks=())
    for column in tgr.GOAL_RATING_COLS:
        assert np.allclose(default[column].to_numpy(),
                           explicit[column].to_numpy(), equal_nan=True), column
    assert not np.allclose(default["net_rating_diff"].fillna(0.0).to_numpy(),
                           thin["net_rating_diff"].fillna(0.0).to_numpy())


def test_goals_target_matches_per_60_on_regulation_frames():
    """The raw count and the per-60 rate are the same target at 60 minutes."""
    frame = _with_goalies(_frame(seasons=(2024, 2025), per_season=120))
    per_60 = tgr.build_causal_panel(frame, refit_days=7, min_train_games=40)
    goals = tgr.build_causal_panel(frame, refit_days=7, min_train_games=40,
                                   target="goals")
    for column in tgr.GOAL_RATING_COLS:
        assert np.allclose(per_60[column].to_numpy(), goals[column].to_numpy(),
                           equal_nan=True), column


def test_unknown_target_is_rejected():
    frame = _with_goalies(_frame(seasons=(2024,), per_season=120))
    with pytest.raises(ValueError, match="unknown target"):
        tgr.build_causal_panel(frame, refit_days=7, min_train_games=40,
                               target="goals_per_9")


def test_chance_block_is_available_but_not_default():
    """The MLB contact-quality counterpart exists, off until measured."""
    assert "chance" not in tgr.DESIGN_BLOCKS
    frame = _with_goalies(_frame(seasons=(2024, 2025), per_season=120))
    teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
    for side in ("home", "away"):
        rank = {t: float(i) for i, t in enumerate(teams)}
        frame[f"xg_for_15g_{side}"] = frame[f"{side}_team"].map(rank) * 0.05 + 2.3
        frame[f"xg_against_15g_{side}"] = frame[f"{side}_team"].map(rank) * 0.04 + 2.1
    base = tgr.build_causal_panel(frame, refit_days=7, min_train_games=40)
    chance = tgr.build_causal_panel(
        frame, refit_days=7, min_train_games=40,
        blocks=tgr.DESIGN_BLOCKS + ("chance",))
    assert not np.allclose(base["net_rating_diff"].fillna(0.0).to_numpy(),
                           chance["net_rating_diff"].fillna(0.0).to_numpy())
