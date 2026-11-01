"""O-PTS arm: additive expected-points aggregates and the six-column attach.

Pins the methodology, not the verdict:
* per-player rate math (x_pts_100 x pace, pace fallback chain, NaN rate);
* the team SUM identity (additive, not a mean) over final weights;
* absorbed minutes repriced at the survivor's rate (never vanishing);
* star-out points read the same pre-availability top-three as impact;
* the attach publishes sides + diff with the always-create rule;
* config registration (KNOWN, RAW_PER_SIDE_COLS, subset gate).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import lineup_projection as proj_mod


def _latest(rows):
    return pd.DataFrame(rows)


def test_pts_per_minute_math_and_pace_fallback():
    frame = pd.DataFrame([
        {"x_pts_100": 120.0, "x_pace": 100.0},  # 1.2 * 100/48
        {"x_pts_100": 110.0, "x_pace": np.nan},  # team-mean pace fill
        {"x_pts_100": np.nan, "x_pace": 100.0},  # unpriced -> NaN
    ])
    ppm = proj_mod._pts_per_minute(frame)
    assert abs(ppm.iloc[0] - 1.2 * 100.0 / 48.0) < 1e-9
    assert abs(ppm.iloc[1] - 1.1 * 100.0 / 48.0) < 1e-9
    assert np.isnan(ppm.iloc[2])


def test_pts_per_minute_league_pace_fallback():
    frame = pd.DataFrame([{"x_pts_100": 96.0, "x_pace": np.nan}])
    ppm = proj_mod._pts_per_minute(frame)
    assert abs(ppm.iloc[0] - 0.96 * 100.0 / 48.0) < 1e-9


def test_team_o_pts_is_a_sum_over_final_weights():
    healthy = pd.DataFrame({"_w": [30.0, 20.0, 10.0]})
    ppm = pd.Series([2.0, 1.0, 0.5], index=healthy.index)
    assert proj_mod._team_o_pts(healthy, ppm) == 30 * 2.0 + 20 * 1.0 + 10 * 0.5


def test_team_o_pts_refuses_when_nothing_priced():
    healthy = pd.DataFrame({"_w": [30.0, 20.0]})
    ppm = pd.Series([np.nan, np.nan], index=healthy.index)
    assert np.isnan(proj_mod._team_o_pts(healthy, ppm))
    assert np.isnan(proj_mod._team_o_pts(healthy, None))


def test_team_o_pts_reprices_absorbed_minutes_at_survivor_rate():
    # An out star's 30 expected minutes absorbed into the survivor's _w:
    # all 40 price at the survivor's 1.5 ppm, never at the star's 2.5.
    healthy = pd.DataFrame({"_w": [40.0]})
    ppm = pd.Series([1.5], index=healthy.index)
    assert proj_mod._team_o_pts(healthy, ppm) == 60.0


def _ref_pool(out_idx=()):
    rows = []
    for i, (mpg, rate) in enumerate([(34.0, 2.5), (30.0, 2.0),
                                     (28.0, 1.5), (10.0, 1.0)]):
        rows.append({
            "player_id": f"p{i}",
            "prior_minutes_per_game": mpg,
            "prior_minutes": mpg * 50,
            "rapm_shrunk": 1.0,
            "is_available": i not in set(out_idx),
            "x_pts_100": rate * 48.0,  # pace defaults to 100 -> ppm = rate
            "x_pace": 100.0,
        })
    return _latest(rows)


def test_star_out_pts_same_reference_set_as_impact():
    latest = _ref_pool(out_idx=(0,))
    ppm = proj_mod._pts_per_minute(latest)
    impact, share = proj_mod._star_out_aggregates(latest)
    pts = proj_mod._star_out_pts(latest, ppm)
    # Out member is the top-weight member; both legs read him.
    assert share > 0
    assert pts == 34.0 * 2.5
    assert impact == impact  # finite: the DPM leg prices the same member


def test_star_out_pts_zero_when_nobody_out_nan_when_empty():
    latest = _ref_pool()
    assert proj_mod._star_out_pts(latest, proj_mod._pts_per_minute(latest)) == 0.0
    assert np.isnan(proj_mod._star_out_pts(
        latest.iloc[0:0], proj_mod._pts_per_minute(latest.iloc[0:0])))


def test_attach_o_pts_sides_and_diff():
    slate = pd.DataFrame([{
        "gameday": pd.Timestamp("2026-01-15"),
        "home_team": "BOS", "away_team": "NYK",
    }])
    aggs = pd.DataFrame([
        {"gameday": pd.Timestamp("2026-01-15"), "team": "BOS",
         "team_o_pts": 115.0, "star_out_pts": 20.0},
        {"gameday": pd.Timestamp("2026-01-15"), "team": "NYK",
         "team_o_pts": 110.0, "star_out_pts": 0.0},
    ])
    out = proj_mod.attach_o_pts_features(slate, aggs)
    assert out["pl_o_pts_home"].iloc[0] == 115.0
    assert out["pl_o_pts_away"].iloc[0] == 110.0
    assert out["pl_o_pts_diff"].iloc[0] == 5.0
    assert out["star_out_pts_home"].iloc[0] == 20.0
    assert out["star_out_pts_diff"].iloc[0] == 20.0


def test_attach_o_pts_always_creates_columns():
    out = proj_mod.attach_o_pts_features(pd.DataFrame(), pd.DataFrame())
    for c in config.O_PTS_FEATURE_COLS:
        assert c in out.columns


def test_config_registration():
    assert len(config.O_PTS_FEATURE_COLS) == 6
    assert set(config.O_PTS_FEATURE_COLS) <= set(config.KNOWN_FEATURE_COLS)
    assert "pl_o_pts_home" in config.RAW_PER_SIDE_COLS
    assert "pl_o_pts_away" in config.RAW_PER_SIDE_COLS
    assert "star_out_pts_home" in config.RAW_PER_SIDE_COLS
    assert "star_out_pts_away" in config.RAW_PER_SIDE_COLS
    assert "pl_o_pts_diff" not in config.RAW_PER_SIDE_COLS
    assert "star_out_pts_diff" not in config.RAW_PER_SIDE_COLS
    config.set_feature_subset(list(config.O_PTS_FEATURE_COLS))
    try:
        assert set(config.active_moneyline_feature_cols()) >= set(
            config.O_PTS_FEATURE_COLS)
    finally:
        config.reset_feature_subset()
    # The C arm's replaced eleven are all live contract columns.
    from ab_o_pts import REPLACE_11
    assert len(REPLACE_11) == 11
    assert set(REPLACE_11) <= set(config.MONEYLINE_FEATURE_COLS)
    assert set(REPLACE_11).isdisjoint(config.O_PTS_FEATURE_COLS)


def test_o_pts_is_contract_and_linear_safe():
    assert set(config.O_PTS_FEATURE_COLS) <= set(config.MONEYLINE_FEATURE_COLS)
    import features as feat_mod
    linear = feat_mod.linear_feature_columns()
    for c in config.O_PTS_FEATURE_COLS:
        if c.endswith("_diff"):
            assert c in linear, c
        else:
            assert c not in linear, c
