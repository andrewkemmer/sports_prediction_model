"""Distribution-family mathematical and public-serving contracts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distributions as dist
import gaussian_family as family


def test_split_normal_is_normal_at_zero_skew_and_preserves_moments():
    grid = np.linspace(-150., 150., 30001)
    np.testing.assert_allclose(family.split_cdf(grid, 3., 8., 0.), ndtr((grid - 3.) / 8.))
    for skew in (-.6, .6):
        cdf = family.split_cdf(grid, 3., 8., skew)
        mass = np.diff(cdf)
        mid = (grid[1:] + grid[:-1]) / 2
        assert np.all(mass >= 0.)
        assert abs(mass.sum() - 1.) < 1e-10
        mean = np.sum(mid * mass)
        variance = np.sum((mid - mean) ** 2 * mass)
        assert abs(mean - 3.) < 1e-6
        assert abs(variance - 64.) < .001


def test_split_calibration_is_monotone_and_asymmetric():
    raw = np.linspace(0., 1., 1001)[None, :]
    beta = np.array([0., 0., 0., -.4, 0., 0., .4, 0., 0.])
    cdf = family.transform(raw, np.array([[1., 0., 0.]]), beta)
    assert np.all(np.diff(cdf) >= 0.)
    assert cdf[0, 0] == 0. and cdf[0, -1] == 1.
    assert not np.isclose(cdf[0, 100], 1. - cdf[0, 900])


def test_crps_fit_beats_identity_on_calibration_training_population():
    # This checks the optimizer, not a claimed out-of-sample improvement.
    rng = np.random.default_rng(912)
    h, a = np.full(400, 110.), np.full(400, 106.)
    y = np.round(226. + rng.normal(0., 22., len(h)))
    p = family.defaults('nba')
    support, raw = family.raw_cdf(h, a, 'total', 'nba', p)
    beta = family.fit_map(h, a, y, 'total', 'nba', p)
    calibrated = family.transform(raw, family.design(h, a, 'nba'), beta)
    assert family.crps(calibrated, support, y).mean() < family.crps(raw, support, y).mean()


def test_sampler_precision_stable_identity_streams_and_mass():
    support = np.arange(-20, 21)
    cdf = np.tile(ndtr((support + .5) / 3.), (2, 1))
    a, meta = family.sample_cdf(cdf, support, ['a', 'b'], 'margin', 'nba')
    b, _ = family.sample_cdf(cdf[::-1], support, ['b', 'a'], 'margin', 'nba')
    np.testing.assert_array_equal(a, b[::-1])
    assert np.max(np.abs(a - cdf)) < .015
    np.testing.assert_allclose(a[:, -1], 1.)
    assert meta['n_draws'] == 50000
    with pytest.raises(ValueError, match='draw count'):
        family.sample_cdf(cdf, support, ['a', 'b'], 'margin', 'nba', n_draws=3)


def test_public_pricing_is_outcome_free_and_arbitrary_lines_match_grid():
    reference = family.SPORT_PRIORS[dist.DISTRIBUTION_SPORT][0] / 2
    frame = pd.DataFrame({'game_id': ['a', 'b'], 'game_pk': ['a', 'b'],
                          'mu_h': [reference + 1., reference], 'mu_a': [reference, reference + 1.],
                          'home_expected_runs': [reference + 1., reference],
                          'away_expected_runs': [reference, reference + 1.]})
    bundle = {'method': family.METHOD, 'maps': {}, 'monte_carlo': {'requested_draws': 50000, 'seed': 42}}
    priced = dist.apply_market_calibration(frame, bundle)
    poisoned = frame.assign(home_score=999, away_score=0)
    second = dist.apply_market_calibration(poisoned, bundle)
    cols = [c for c in priced if c.startswith('p_')]
    np.testing.assert_array_equal(priced[cols], second[cols])
    result = dist.distribution_probability(frame, bundle, 'margin', 0.)
    np.testing.assert_allclose(result['win'], priced.p_home_win_derived)
    np.testing.assert_allclose(result['push'], priced.p_tie)
    np.testing.assert_allclose(result['loss'], priced.p_away_win_derived)
    np.testing.assert_allclose(result['win'] + result['push'] + result['loss'], 1.)
    if dist.DISTRIBUTION_SPORT == 'nfl':
        assert np.all(priced.p_tie > 0.)
    else:
        np.testing.assert_allclose(priced.p_tie, 0.)
