"""Distribution-objective parity with NBA (2026-10-13 structure alignment).

The split-skew challenger mechanics (gaussian_family, calibration, Monte
Carlo) were already byte-shared across the four backends. What NBA carried
alone was the OBJECTIVE layer around the side means and the diagnostics that
read them. These tests pin the NHL side of that alignment: the Poisson
default stays the default, the CRPS location trees stay opt-in, and the
team-ID pair keeps the categorical routing NHL's OOF walk and refit share.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import distributions as dist


def test_gaussian_crps_objective_matches_the_numerical_gradient():
    objective = dist.GaussianCRPSObjective(4.)
    y = np.array([1., 2., 3.])
    pred = np.array([1.5, 2.5, 3.5])
    grad, hess = objective(y, pred)
    eps = 1e-6
    numerical = (dist.gaussian_crps(y, pred + eps, 4.)
                 - dist.gaussian_crps(y, pred - eps, 4.)) / (2 * eps)
    np.testing.assert_allclose(grad, numerical, rtol=1e-5)
    assert np.all(hess > 0.)


def test_gaussian_crps_is_minimised_at_the_truth_and_refuses_zero_scale():
    y = np.array([0., 5., 10.])
    assert np.all(dist.gaussian_crps(y, y, 3.) < dist.gaussian_crps(y, y + 2., 3.))
    with pytest.raises(ValueError, match="positive"):
        dist.gaussian_crps(y, y, 0.)


def test_side_regressor_keeps_the_poisson_default_and_exposes_the_opt_in():
    default = dist.ScoreRegressor()
    assert default.crps_location is False
    assert default._declare_categoricals is True
    assert default.home_model.objective == "poisson"
    challenger = dist.ScoreRegressor(crps_location=True)
    assert challenger.crps_location is True
    assert isinstance(challenger.home_model.objective,
                      dist.GaussianCRPSObjective)


def test_fit_requires_at_least_two_settled_games(monkeypatch):
    frame = pd.DataFrame({"home_score": [3.], "away_score": [2.]})
    reg = dist.ScoreRegressor()
    monkeypatch.setattr(reg, "_matrix", lambda df: df[["home_score"]])
    with pytest.raises(ValueError, match="two settled games"):
        reg.fit(frame)


def test_grid_key_is_the_shared_nba_column_builder():
    assert dist._grid_key("p_over", 6) == "p_over_6"
    assert dist._grid_key("p_home_cover", -1.5) == "p_home_cover_m1_5"


def test_run_line_fit_check_reports_both_sides_against_the_baseline():
    oof = pd.DataFrame({"home_score": [4., 7., 2.],
                        "away_score": [3., 6., 5.],
                        "mu_h": [3.5, 7.2, 2.4],
                        "mu_a": [3.2, 5.8, 4.6]})
    check = dist.run_line_fit_check(oof)
    assert set(check) == {"home", "away"}
    for side in ("home", "away"):
        assert check[side]["n"] == 3
        assert np.isfinite(check[side]["pearson"])
        assert check[side]["rmse_model"] < check[side]["rmse_baseline"]


def test_distribution_feature_weights_degrade_honestly_when_unfitted():
    assert dist.distribution_feature_weights(None) is None
    assert dist.distribution_feature_weights(dist.ScoreRegressor()) is None


def test_run_engine_feature_cols_excludes_the_team_id_pair():
    cols = dist.run_engine_feature_cols()
    assert cols
    assert not set(cols) & set(config.TREE_CATEGORICAL_COLS)
