"""Tests for the two-sided run model's rating panel.

The properties that matter are causal, not statistical: a rating attached to a
game must never be a function of that game or anything later, the seven columns
must satisfy the documented identity, and the fit must rank offence correctly
on a frame where the answer is known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import team_run_ratings as trr


def _frame(seasons=(2024, 2025), per_season: int = 120, teams: int = 8,
           strong_team: str = "T0", strong_runs: float = 9.0) -> pd.DataFrame:
    """A synthetic league: balanced schedule, one team that cannot stop scoring."""
    names = [f"T{i}" for i in range(teams)]
    rng = np.random.default_rng(7)
    rows = []
    pk = 0
    for season in seasons:
        start = pd.Timestamp(f"{season}-04-01")
        for day in range(per_season // 2):
            for pair in range(teams // 2):
                home = names[(pair * 2 + day) % teams]
                away = names[(pair * 2 + day + 1) % teams]
                if home == away:
                    continue
                pk += 1
                home_runs = strong_runs if home == strong_team else 4.2
                away_runs = strong_runs if away == strong_team else 4.2
                rows.append({
                    "game_pk": pk,
                    "gameday": start + pd.Timedelta(days=day),
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_score": float(home_runs + rng.normal(0, 0.3)),
                    "away_score": float(away_runs + rng.normal(0, 0.3)),
                })
    return pd.DataFrame(rows)


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    return trr.build_causal_panel(frame, refit_days=7, min_train_games=40)


def test_seven_columns_and_identity():
    frame = _frame()
    panel = _panel(frame)
    assert list(panel.columns) == trr.RUN_RATING_COLS
    assert len(trr.RUN_RATING_COLS) == 7
    got = panel.dropna(subset=["net_rating_diff"])
    assert len(got) > 0
    err = (got["off_rating_diff"] + got["def_rating_diff"]
           - got["net_rating_diff"]).abs().max()
    assert err < 1e-9
    # net is the matchup number.  def_rating is signed positive-is-good, so
    # the matchup margin is off_home + def_home - off_away - def_away, which
    # is the same expression as home offence against away pitching minus the
    # reverse.
    matchup = ((got["off_rating_home"] + got["def_rating_home"])
               - (got["off_rating_away"] + got["def_rating_away"]))
    assert np.allclose(matchup, got["net_rating_diff"], atol=1e-9)


def test_future_games_cannot_change_past_ratings():
    frame = _frame()
    cut = pd.Timestamp("2025-05-01")
    assert (frame["gameday"] < cut).any() and (frame["gameday"] >= cut).any()
    full = _panel(frame)
    kept = frame[frame["gameday"] < cut].reset_index(drop=True)
    truncated = _panel(kept)
    past = full.loc[(frame["gameday"] < cut).to_numpy()].reset_index(drop=True)
    assert len(truncated) == len(past)
    for column in trr.RUN_RATING_COLS:
        assert np.allclose(past[column].to_numpy(),
                           truncated[column].to_numpy(),
                           equal_nan=True), column


def test_fit_refuses_training_rows_at_or_after_the_anchor():
    frame = _frame(seasons=(2024,), per_season=40)
    rows = trr.build_side_rows(frame)
    anchor = pd.Timestamp("2024-04-20")
    leaky = rows.copy()
    # Force one row onto the anchor itself.
    leaky.loc[leaky.index[0], "gameday"] = anchor
    with pytest.raises(ValueError, match="anchor"):
        trr._fit_once(leaky, ["T0", "T1"], anchor, trr.ALPHA, trr.HALF_LIFE_DAYS)


def test_strongest_offence_gets_the_highest_rating():
    frame = _frame(strong_team="T0", strong_runs=9.0)
    panel = _panel(frame)
    by_team = {}
    for _, row in frame.iterrows():
        idx = frame.index[frame["game_pk"] == row["game_pk"]][0]
        if np.isfinite(panel.loc[idx, "off_rating_home"]):
            by_team.setdefault(row["home_team"], []).append(
                panel.loc[idx, "off_rating_home"])
            by_team.setdefault(row["away_team"], []).append(
                panel.loc[idx, "off_rating_away"])
    means = {t: float(np.mean(v)) for t, v in by_team.items() if len(v) >= 10}
    assert means, "no team accumulated ratings"
    ranked = sorted(means, key=means.get, reverse=True)
    assert ranked[0] == "T0", means


def test_attach_keeps_row_order_and_reports_coverage():
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = trr.attach_run_ratings(shuffled, refit_days=7, min_train_games=40)
    assert len(out) == len(shuffled)
    assert set(trr.RUN_RATING_COLS) <= set(out.columns)
    # Attaching must not reorder or reindex the caller's frame.
    assert np.array_equal(out["game_pk"].to_numpy(), shuffled["game_pk"].to_numpy())
    assert out["net_rating_diff"].notna().mean() > 0.5


def test_empty_frame_raises():
    with pytest.raises(ValueError):
        trr.attach_run_ratings(pd.DataFrame({"gameday": [], "home_team": [],
                                             "away_team": []}))


# ---------------------------------------------------------------------------
# The opt-in knobs used by the optimisation study.  Their defaults must stay
# exactly the shipped configuration, or the study would be measuring a
# different model from the one in the A/B/C harness.


def test_design_blocks_shrink_the_numeric_block():
    rows = trr.build_side_rows(_frame(per_season=8))
    full = trr._numeric_columns(rows, trr.DESIGN_BLOCKS, False)
    bare = trr._numeric_columns(rows, (), False)
    context = trr._numeric_columns(rows, ("context",), False)
    assert bare == ["is_home"]
    assert len(context) > len(bare)
    assert len(full) > len(context)


def test_own_starter_block_is_opt_in():
    frame = _frame(per_season=8)
    for name in trr.OWN_STARTER_COMPONENTS:
        frame[f"{name}_home"] = 4.0
        frame[f"{name}_away"] = 4.5
    rows = trr.build_side_rows(frame)
    off = trr._numeric_columns(rows, trr.DESIGN_BLOCKS, False)
    on = trr._numeric_columns(rows, trr.DESIGN_BLOCKS, True)
    assert set(on) - set(off) == {f"own_{n}" for n in trr.OWN_STARTER_COMPONENTS}
    # The home side row must carry the HOME starter, not the visitor's.
    home = rows[rows["pos"] == 0].iloc[0]
    assert np.isclose(home["own_sp_xfip_5g"], 4.0)


def test_poisson_likelihood_runs_the_whole_panel():
    frame = _frame(per_season=150)
    panel = trr.build_causal_panel(frame, refit_days=7, min_train_games=40,
                                   likelihood="poisson")
    assert panel["net_rating_diff"].notna().mean() > 0.5
    assert np.isfinite(panel["net_rating_diff"].dropna()).all()


def test_unknown_likelihood_is_rejected():
    rows = trr.build_side_rows(_frame(per_season=8))
    teams = sorted(set(rows["team"]))
    anchor = pd.Timestamp(rows["gameday"].quantile(0.7))
    train = rows[rows["gameday"] < anchor]
    with pytest.raises(ValueError, match="likelihood"):
        trr._fit_once(train, teams, anchor, 50.0, 120.0, likelihood="tweedie")


def test_defaults_are_the_shipped_configuration():
    frame = _frame(per_season=150)
    default = trr.build_causal_panel(frame, refit_days=7, min_train_games=40)
    explicit = trr.build_causal_panel(
        frame, refit_days=7, min_train_games=40,
        alpha=trr.ALPHA, half_life_days=trr.HALF_LIFE_DAYS,
        blocks=trr.DESIGN_BLOCKS, likelihood=trr.LIKELIHOOD,
        own_starter=trr.OWN_STARTER)
    for column in trr.RUN_RATING_COLS:
        assert np.allclose(default[column].to_numpy(), explicit[column].to_numpy(),
                           equal_nan=True), column


# ---------------------------------------------------------------------------
# Production contract (adopted 2026-11-21, MLB scenario C): the seven fitted
# columns are the served strength block, and the daily pipeline must be able
# to attach them to the delivered frame shape (game_date, no gameday/season).


def test_production_universe_serves_the_run_ratings():
    import training

    for column in trr.RUN_RATING_COLS:
        assert column in training.MONEYLINE_FEATURE_COLS, column
    for retired in ("elo_diff", "win_pct_diff", "home_elo", "away_elo",
                    "home_win_pct", "away_win_pct"):
        assert retired not in training.MONEYLINE_FEATURE_COLS, retired
        # Retired from serving, NOT from the pool: RFE may re-trial them and
        # a bundle recording the old width must still validate.
        assert retired in training.KNOWN_FEATURE_COLS, retired
    assert set(trr.SIDE_RATING_COLS) <= set(training.RAW_PER_SIDE_COLS)
    logistic = set(training._logistic_feature_cols())
    assert set(trr.SIDE_RATING_COLS).isdisjoint(logistic)
    assert {"off_rating_diff", "def_rating_diff",
            "net_rating_diff"} <= logistic


def test_delivery_frame_shape_attaches_without_gameday_or_season():
    """Production's frame names the date game_date and carries no season."""
    frame = _frame().rename(columns={"gameday": "game_date"})
    frame = frame.drop(columns=["season"])
    out = trr.attach_run_ratings(frame, refit_days=7, min_train_games=40)
    assert out["net_rating_diff"].notna().mean() > 0.5
    reference = trr.attach_run_ratings(_frame(), refit_days=7,
                                       min_train_games=40)
    assert np.allclose(out["net_rating_diff"].to_numpy(),
                       reference["net_rating_diff"].to_numpy(),
                       equal_nan=True)


def test_an_appended_slate_row_is_priced_from_history():
    """A target-date row with NULL scores gets a rating and changes nothing."""
    frame = _frame()
    target = pd.Timestamp(frame["gameday"].max()) + pd.Timedelta(days=1)
    slate = pd.DataFrame([{
        "game_pk": 10 ** 9, "gameday": target, "season": target.year,
        "home_team": "T0", "away_team": "T1",
        "home_score": np.nan, "away_score": np.nan,
    }])
    combined = trr.attach_run_ratings(
        pd.concat([frame, slate], ignore_index=True),
        refit_days=7, min_train_games=40)
    row = combined.iloc[-1]
    for column in trr.RUN_RATING_COLS:
        assert np.isfinite(row[column]), column
    history = combined.iloc[:-1].reset_index(drop=True)
    reference = trr.attach_run_ratings(frame, refit_days=7,
                                       min_train_games=40)
    for column in trr.RUN_RATING_COLS:
        assert np.allclose(history[column].to_numpy(),
                           reference[column].to_numpy(),
                           equal_nan=True), column
