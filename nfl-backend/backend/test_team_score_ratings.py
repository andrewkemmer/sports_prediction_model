"""Tests for the fitted NFL offence/defence scoring-rating panel.

The properties that matter are causal and structural: a rating attached to a
game must never see that game or anything later, the off/def columns must stay
additive while the net carries the Elo tilt, and the fit must recover a known
strong offence and a known strong defence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import features
import team_score_ratings as tsr


def _frame(seasons=(2024, 2025), per_season: int = 240, teams: int = 8,
           strong_off: str = "T0", strong_def: str = "T1") -> pd.DataFrame:
    """A synthetic league with one strong offence and one strong defence."""
    names = [f"T{i}" for i in range(teams)]
    rng = np.random.default_rng(11)
    rows = []
    pk = 0
    for season in seasons:
        start = pd.Timestamp(f"{season}-09-05")
        for day in range(per_season // 4):
            for pair in range(teams // 2):
                home = names[(pair * 2 + day) % teams]
                away = names[(pair * 2 + day + 1) % teams]
                if home == away:
                    continue
                pk += 1
                home_pts = 21.0 + (7.0 if home == strong_off else 0.0) \
                    - (7.0 if away == strong_def else 0.0) + 2.0
                away_pts = 21.0 + (7.0 if away == strong_off else 0.0) \
                    - (7.0 if home == strong_def else 0.0)
                rows.append({
                    "game_id": f"{season}_{pk:05d}",
                    "gameday": start + pd.Timedelta(days=day),
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_score": float(home_pts + rng.normal(0, 0.8)),
                    "away_score": float(away_pts + rng.normal(0, 0.8)),
                })
    return pd.DataFrame(rows)


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    return tsr.build_causal_panel(frame, refit_days=1, min_train_games=40)


def test_seven_columns_and_decomposition():
    frame = _frame()
    rng = np.random.default_rng(3)
    frame["elo_home"] = 1500.0 + rng.normal(0, 60, len(frame))
    frame["elo_away"] = 1500.0 + rng.normal(0, 60, len(frame))
    panel = _panel(frame)
    assert list(panel.columns) == tsr.SCORE_RATING_COLS
    assert len(tsr.SCORE_RATING_COLS) == 7
    # The net is published for the whole history (the Elo prior covers the
    # warmup), while the fitted components need the 40-game minimum.
    assert (panel["net_rating_diff"].notna().mean()
            >= panel["off_rating_diff"].notna().mean())
    got = panel.dropna(subset=["net_rating_diff", "off_rating_diff",
                               "def_rating_diff"])
    assert len(got) > 0
    # The side levels are the pure fitted scoring components.
    assert np.allclose(got["off_rating_diff"],
                       got["off_rating_home"] - got["off_rating_away"])
    assert np.allclose(got["def_rating_diff"],
                       got["def_rating_home"] - got["def_rating_away"])
    # The net is a blended strength index, not the sum of the components, so
    # the linear member gets a direction the decomposition cannot carry.
    implied = got["off_rating_diff"] + got["def_rating_diff"]
    assert np.corrcoef(implied, got["net_rating_diff"])[0, 1] > 0.8
    assert (implied - got["net_rating_diff"]).abs().max() > 1e-6


def test_future_games_cannot_change_past_ratings():
    frame = _frame(seasons=(2024, 2025), per_season=240)
    cut = pd.Timestamp("2025-10-01")
    assert (frame["gameday"] < cut).any() and (frame["gameday"] >= cut).any()
    full = _panel(frame)
    kept = frame[frame["gameday"] < cut].reset_index(drop=True)
    truncated = _panel(kept)
    past = full.loc[(frame["gameday"] < cut).to_numpy()].reset_index(drop=True)
    assert len(truncated) == len(past)
    for column in tsr.SCORE_RATING_COLS:
        assert np.allclose(past[column].to_numpy(),
                           truncated[column].to_numpy(),
                           equal_nan=True), column


def test_fit_refuses_training_rows_at_or_after_the_anchor():
    frame = _frame(seasons=(2024,), per_season=80)
    rows = tsr.build_side_rows(frame)
    anchor = pd.Timestamp(rows["gameday"].iloc[10])
    leaky = rows.copy()
    leaky.loc[leaky.index[0], "gameday"] = anchor
    with pytest.raises(ValueError, match="anchor"):
        tsr._fit_once(leaky, ["T0", "T1"], anchor, tsr.ALPHA,
                      tsr.HALF_LIFE_DAYS)


def test_strongest_offence_and_defence_rank_first():
    frame = _frame(strong_off="T0", strong_def="T1")
    panel = _panel(frame)
    off, dfn = {}, {}
    for _, row in frame.iterrows():
        idx = frame.index[frame["game_id"] == row["game_id"]][0]
        if np.isfinite(panel.loc[idx, "off_rating_home"]):
            off.setdefault(row["home_team"], []).append(
                panel.loc[idx, "off_rating_home"])
            off.setdefault(row["away_team"], []).append(
                panel.loc[idx, "off_rating_away"])
            dfn.setdefault(row["home_team"], []).append(
                panel.loc[idx, "def_rating_home"])
            dfn.setdefault(row["away_team"], []).append(
                panel.loc[idx, "def_rating_away"])
    off_mean = {t: float(np.mean(v)) for t, v in off.items() if len(v) >= 10}
    def_mean = {t: float(np.mean(v)) for t, v in dfn.items() if len(v) >= 10}
    assert off_mean and def_mean
    assert max(off_mean, key=off_mean.get) == "T0", off_mean
    assert max(def_mean, key=def_mean.get) == "T1", def_mean


def test_attach_keeps_row_order_and_reports_coverage():
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = tsr.attach_score_ratings(
        shuffled, refit_days=1, min_train_games=40)
    assert len(out) == len(shuffled)
    assert set(tsr.SCORE_RATING_COLS) <= set(out.columns)
    assert np.array_equal(out["game_id"].to_numpy(),
                          shuffled["game_id"].to_numpy())
    assert out["net_rating_diff"].notna().mean() > 0.5


def test_short_frame_degrades_to_nan_columns():
    frame = _frame(seasons=(2024,), per_season=12)
    out = tsr.attach_score_ratings(frame, min_train_games=200)
    assert set(tsr.SCORE_RATING_COLS) <= set(out.columns)
    assert out["net_rating_diff"].isna().all()


def test_design_blocks_shrink_the_numeric_block():
    frame = _frame(per_season=8)
    for name in tsr.OFF_COMPONENTS:
        for side in ("home", "away"):
            frame[f"{name}_{side}"] = 1.0
    for name in tsr.OPP_COMPONENTS:
        for side in ("home", "away"):
            frame[f"{name}_{side}"] = 1.0
    rows = tsr.build_side_rows(frame)
    full = tsr._numeric_columns(rows, ("components", "context", "prior"))
    bare = tsr._numeric_columns(rows, ())
    prior = tsr._numeric_columns(rows, ("prior",))
    assert bare == ["is_home"]
    assert len(prior) > len(bare)
    assert len(full) > len(prior)
    # Adopted design (2026-11-25): the trailing component panel sits beside
    # the prior bridge; schedule context remains a trial block.
    assert tsr.DESIGN_BLOCKS == ("components", "prior")
    shipped = tsr._numeric_columns(rows, tsr.DESIGN_BLOCKS)
    terms = ([f"off_{name}" for name in tsr.OFF_COMPONENTS]
             + [f"opp_{name}" for name in tsr.OPP_COMPONENTS])
    assert shipped[:len(terms)] == terms
    assert shipped[len(terms):] == prior


def test_component_block_changes_the_fit_and_is_a_noop_when_missing():
    frame = _frame(seasons=(2024, 2025), per_season=240)
    rng = np.random.default_rng(13)
    for name in tsr.OFF_COMPONENTS:
        for side in ("home", "away"):
            frame[f"{name}_{side}"] = rng.normal(0.0, 1.0, len(frame))
    for name in tsr.OPP_COMPONENTS:
        for side in ("home", "away"):
            frame[f"{name}_{side}"] = rng.normal(0.0, 1.0, len(frame))
    shipped = _panel(frame)
    prior = tsr.build_causal_panel(frame, refit_days=1, min_train_games=40,
                                   blocks=("prior",))
    assert not np.allclose(shipped["net_rating_diff"].to_numpy(),
                           prior["net_rating_diff"].to_numpy(),
                           equal_nan=True)
    # A frame with no component columns standardises them to zero, so the
    # shipped fit degrades exactly to the prior-only bridge, not to NaN.
    bare = _frame(seasons=(2024, 2025), per_season=240)
    assert np.allclose(_panel(bare)["net_rating_diff"].to_numpy(),
                       tsr.build_causal_panel(bare, refit_days=1,
                                              min_train_games=40,
                                              blocks=("prior",)
                                              )["net_rating_diff"].to_numpy(),
                       equal_nan=True)


def test_delivery_frame_shape_attaches_without_gameday_or_season():
    frame = _frame().rename(columns={"gameday": "game_date"})
    frame = frame.drop(columns=["season"])
    out = tsr.attach_score_ratings(
        frame, refit_days=1, min_train_games=40)
    assert out["net_rating_diff"].notna().mean() > 0.5
    reference = tsr.attach_score_ratings(
        _frame(), refit_days=1, min_train_games=40)
    assert np.allclose(out["net_rating_diff"].to_numpy(),
                       reference["net_rating_diff"].to_numpy(),
                       equal_nan=True)


def test_prior_season_is_shifted_one_year():
    frame = _frame(seasons=(2024, 2025), per_season=40)
    prior = tsr._prior_points(frame)
    assert 2025 in set(prior["season"])
    assert 2026 in set(prior["season"])
    # The 2025 season's prior is the completed 2024 season.
    row = prior[(prior["season"] == 2025) & (prior["team"] == "T0")].iloc[0]
    raw = frame[(frame["season"] == 2024)
                & ((frame["home_team"] == "T0")
                   | (frame["away_team"] == "T0"))]
    assert np.isfinite(row["own_p_pf"])
    assert len(raw) > 0


def test_slate_helper_maps_history_panel_to_pending_game():
    history = _frame(seasons=(2024, 2025), per_season=240)
    pending = history.iloc[[-1]].copy()
    pending["game_id"] = "PENDING_GAME"
    pending["gameday"] = (history["gameday"].max()
                           + pd.Timedelta(days=7))
    pending["home_score"] = np.nan
    pending["away_score"] = np.nan
    source = pd.concat([history, pending], ignore_index=True, sort=False)
    out = features._attach_score_ratings(pending, source=source)
    assert len(out) == 1
    assert np.isfinite(out["net_rating_diff"].iloc[0])
    assert np.isfinite(out["off_rating_diff"].iloc[0])
    assert np.isfinite(out["def_rating_diff"].iloc[0])


def test_production_contract_serves_the_score_ratings():
    import config

    for column in tsr.SCORE_RATING_COLS:
        assert column in config.MONEYLINE_FEATURE_COLS, column
    for retired in config.RETIRED_STRENGTH_COLS:
        assert retired not in config.MONEYLINE_FEATURE_COLS, retired
        assert retired in config.KNOWN_FEATURE_COLS, retired
    assert set(tsr.SIDE_SCORE_RATING_COLS) <= set(config.RAW_PER_SIDE_COLS)
    # The net carries the Elo prior, so the three diffs are not collinear and
    # every family receives the full decomposition; only the side levels stay
    # tree-only.
    linear = set(features.linear_feature_columns())
    assert {"off_rating_diff", "def_rating_diff",
            "net_rating_diff"} <= linear
    assert set(tsr.SIDE_SCORE_RATING_COLS).isdisjoint(linear)
    # The component panel reads only columns the production frame already
    # serves, so adopting the block needs no new ingestion or contract width.
    component_inputs = {f"{name}_{side}"
                        for name in tsr.OFF_COMPONENTS + tsr.OPP_COMPONENTS
                        for side in ("home", "away")}
    assert component_inputs <= set(config.MONEYLINE_FEATURE_COLS)


def test_elo_prior_is_used_and_causal():
    frame = _frame(seasons=(2024, 2025), per_season=240)
    rng = np.random.default_rng(7)
    frame["elo_home"] = 1500.0 + rng.normal(0, 60, len(frame))
    frame["elo_away"] = 1500.0 + rng.normal(0, 60, len(frame))
    with_prior = _panel(frame)
    without = _panel(frame.drop(columns=["elo_home", "elo_away"]))
    got = with_prior["net_rating_diff"].notna().to_numpy()
    assert got.any()
    assert not np.allclose(with_prior.loc[got, "net_rating_diff"],
                           without.loc[got, "net_rating_diff"])
    # A future Elo change must not move a past net.
    cut = pd.Timestamp("2025-10-01")
    tampered = frame.copy()
    later = (tampered["gameday"] >= cut).to_numpy()
    tampered.loc[later, ["elo_home", "elo_away"]] += 400.0
    after = _panel(tampered)
    past = (frame["gameday"] < cut).to_numpy()
    assert np.allclose(with_prior.loc[past, "net_rating_diff"].to_numpy(),
                       after.loc[past, "net_rating_diff"].to_numpy(),
                       equal_nan=True)


def test_side_rows_canonicalise_relocation_labels():
    frame = _frame(per_season=8)
    frame["home_team"] = "OAK"
    frame["away_team"] = "SD"
    rows = tsr.build_side_rows(frame)
    assert set(rows["team"]) == {"LV", "LAC"}
    assert set(rows["opp"]) == {"LV", "LAC"}
