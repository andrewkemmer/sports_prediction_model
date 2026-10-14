"""Distribution-objective parity with NBA (2026-10-13 structure alignment).

MLB's run engine keeps its own mean-training architecture (``run_oof`` /
``_fit_side_model`` with the fixed-round policy -- the structure the other
three sports cite as "MLB parity"). What it did not carry from NBA was the
OBJECTIVE layer around those side means, plus the ``_grid_key`` builder the
shared pricing body resolves through. These tests pin that alignment and the
promise that the Poisson default still ships unchanged.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def test_make_reg_ships_poisson_by_default_and_the_crps_objective_on_opt_in():
    assert dist._make_reg().objective == "poisson"
    assert dist._make_reg(7.).objective == "poisson"
    challenger = dist._make_reg(7., crps=True)
    assert isinstance(challenger.objective, dist.GaussianCRPSObjective)
    # the opt-in never rewrites the shared incumbent dict
    assert dist.RUN_LGBM_PARAMS["objective"] == "poisson"


def _tiny_side_frames(n: int = 300):
    # Enough rows that the incumbent's min_child_samples (40) can actually
    # split: on a frame smaller than that LightGBM builds an empty tree and
    # its custom-objective path fails the num_features check -- an artifact
    # of the fixture, not of the crps switch.
    rng = np.random.default_rng(11)
    tr = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    va = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    y_tr = rng.poisson(4., n).astype(float)
    y_va = rng.poisson(4., 50).astype(float)
    return tr, y_tr, va, y_va


def test_fit_side_model_default_path_is_still_poisson_and_positive():
    tr, y_tr, va, y_va = _tiny_side_frames()
    model, lam, best = dist._fit_side_model(dict(dist.RUN_LGBM_PARAMS),
                                            tr, y_tr, va, y_va,
                                            fixed_rounds=8)
    assert model.objective == "poisson"
    assert best == 8 and len(lam) == len(va) and np.all(lam > 0)


def test_fit_side_model_crps_opt_in_fits_and_predicts_positive_means():
    tr, y_tr, va, y_va = _tiny_side_frames()
    model, lam, best = dist._fit_side_model(dict(dist.RUN_LGBM_PARAMS),
                                            tr, y_tr, va, y_va,
                                            fixed_rounds=8, crps=True)
    assert isinstance(model.objective, dist.GaussianCRPSObjective)
    assert np.all(np.isfinite(lam)) and np.all(lam > 0)


def test_grid_key_is_the_shared_nba_column_builder():
    assert dist._grid_key("p_over", 8) == "p_over_8"
    assert dist._grid_key("p_home_cover", -1.5) == "p_home_cover_m1_5"


def test_run_line_fit_check_reads_the_mlb_mu_aliases():
    oof = pd.DataFrame({"home_score": [4., 7., 2.],
                        "away_score": [3., 6., 5.],
                        "home_expected_runs": [3.5, 7.2, 2.4],
                        "away_expected_runs": [3.2, 5.8, 4.6]})
    check = dist.run_line_fit_check(oof)
    assert set(check) == {"home", "away"}
    for side in ("home", "away"):
        assert check[side]["n"] == 3
        assert np.isfinite(check[side]["pearson"])
        assert check[side]["rmse_model"] < check[side]["rmse_baseline"]


def test_poisson_adequacy_probe_sees_overdispersion():
    rng = np.random.default_rng(7)
    mu = np.full(500, 3.)
    y = rng.poisson(mu) * rng.choice([1., 3.], size=500)   # over-dispersed
    assert dist.pearson_poisson_adequacy(y, mu) > 1.
