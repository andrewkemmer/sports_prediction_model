"""FG arm: minutes-weighted mean expected FG% and the three-column attach.

Pins the methodology, not the verdict:
* the team aggregate is a MEAN over priced members only (a rate, not a sum);
* unpriced members leave numerator AND denominator (no 0% fill);
* absorbed minutes reprice at the survivor's rate;
* the attach publishes sides + diff with the always-create rule;
* config registration (KNOWN, RAW_PER_SIDE_COLS, subset gate, trial-only).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import lineup_projection as proj_mod


def test_team_fg_pct_is_a_weighted_mean_over_final_weights():
    healthy = pd.DataFrame({"_w": [30.0, 20.0, 10.0]})
    fg = pd.Series([0.50, 0.40, 0.30], index=healthy.index)
    assert proj_mod._team_fg_pct(healthy, fg) == (15.0 + 8.0 + 3.0) / 60.0


def test_team_fg_pct_excludes_unpriced_from_denominator():
    # The 20-minute member carries no projection: (15 + 3) / 40, not / 60.
    # A 0% fill would punish thinly-projected pools; league-average fill
    # would invent information. Documented refusal of both.
    healthy = pd.DataFrame({"_w": [30.0, 20.0, 10.0]})
    fg = pd.Series([0.50, np.nan, 0.30], index=healthy.index)
    assert proj_mod._team_fg_pct(healthy, fg) == 18.0 / 40.0


def test_team_fg_pct_refuses_when_nothing_priced():
    healthy = pd.DataFrame({"_w": [30.0, 20.0]})
    fg = pd.Series([np.nan, np.nan], index=healthy.index)
    assert np.isnan(proj_mod._team_fg_pct(healthy, fg))
    assert np.isnan(proj_mod._team_fg_pct(healthy, None))
    assert np.isnan(proj_mod._team_fg_pct(healthy.iloc[0:0], fg.iloc[0:0]))


def test_team_fg_pct_ignores_zero_weight_members():
    healthy = pd.DataFrame({"_w": [30.0, 0.0]})
    fg = pd.Series([0.50, 0.90], index=healthy.index)
    assert proj_mod._team_fg_pct(healthy, fg) == 0.50


def test_team_fg_pct_reprices_absorbed_minutes_at_survivor_rate():
    # An out star's 30 expected minutes absorbed into the survivor's _w:
    # all 40 price at the survivor's 0.45, never at the star's 0.55.
    healthy = pd.DataFrame({"_w": [40.0]})
    fg = pd.Series([0.45], index=healthy.index)
    assert proj_mod._team_fg_pct(healthy, fg) == 0.45


def test_attach_fg_pct_sides_and_diff():
    slate = pd.DataFrame([{
        "gameday": pd.Timestamp("2026-01-15"),
        "home_team": "BOS", "away_team": "NYK",
    }])
    aggs = pd.DataFrame([
        {"gameday": pd.Timestamp("2026-01-15"), "team": "BOS",
         "team_fg_pct": 0.472},
        {"gameday": pd.Timestamp("2026-01-15"), "team": "NYK",
         "team_fg_pct": 0.458},
    ])
    out = proj_mod.attach_fg_pct_features(slate, aggs)
    assert out["pl_fg_pct_home"].iloc[0] == 0.472
    assert out["pl_fg_pct_away"].iloc[0] == 0.458
    assert out["pl_fg_pct_diff"].iloc[0] == 0.472 - 0.458


def test_attach_fg_pct_always_creates_columns():
    out = proj_mod.attach_fg_pct_features(pd.DataFrame(), pd.DataFrame())
    for c in config.FG_PCT_FEATURE_COLS:
        assert c in out.columns


def test_config_registration():
    assert len(config.FG_PCT_FEATURE_COLS) == 3
    assert set(config.FG_PCT_FEATURE_COLS) <= set(config.KNOWN_FEATURE_COLS)
    assert "pl_fg_pct_home" in config.RAW_PER_SIDE_COLS
    assert "pl_fg_pct_away" in config.RAW_PER_SIDE_COLS
    assert "pl_fg_pct_diff" not in config.RAW_PER_SIDE_COLS
    config.set_feature_subset(list(config.FG_PCT_FEATURE_COLS))
    try:
        assert set(config.active_moneyline_feature_cols()) >= set(
            config.FG_PCT_FEATURE_COLS)
    finally:
        config.reset_feature_subset()
    # Trial-only: never in the served contract until a verdict promotes it.
    assert set(config.FG_PCT_FEATURE_COLS).isdisjoint(
        set(config.MONEYLINE_FEATURE_COLS))
