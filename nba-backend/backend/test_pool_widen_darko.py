"""Unit tests for DARKO-widened pool membership (pool/Darko remediation).

A player Darko projects for THIS game (same-day x_minutes >=
PLAYER_POOL_WIDEN_MIN_X) is admitted even when the trailing gates refuse
him. Strictly additive: the rating is untouched, availability still binds,
and below-floor / missing-x players stay refused.
"""
import pandas as pd
import pytest

import lineup_projection as proj

# NOTE: mutate the config object the module under test actually reads.
# Under pytest this package imports as backend.lineup_projection (whose
# `from backend import config` is a distinct module object from top-level
# `import config`); via proj.config both regimes hit the same object.
config = proj.config


def _frame():
    day = pd.Timestamp('2026-01-15')
    rows = [
        # player_id, rapm, minutes, mpg, avail, pos, gap, x
        ('vet_g', 1.0, 600.0, 25.0, True, 'G', 1.0, 24.0),
        ('vet_c', 0.5, 500.0, 20.0, True, 'C', 1.0, 22.0),
        # Rookie: floor refuses (10 < 50), recency exempts NaN; x admits.
        ('rookie_f', -0.5, 10.0, 10.0, True, 'F', float('nan'), 20.0),
        # Returnee: floor passes, recency refuses (45 > 30); x admits.
        ('fringe_g', 2.0, 60.0, 12.0, True, 'G', 45.0, 18.0),
        # Phantom: floor passes, recency refuses, no x -> stays refused.
        ('phantom_c', 0.0, 60.0, 12.0, True, 'C', 45.0, float('nan')),
        # Scrub: x below the 5.0 floor -> stays refused.
        ('scrub_f', 3.0, 5.0, 5.0, True, 'F', float('nan'), 2.0),
        # Out rookie: widened into the pool but removed as unavailable.
        ('out_g', 9.9, 8.0, 8.0, False, 'G', float('nan'), 22.0),
    ]
    return pd.DataFrame([{
        'team': 'T', 'gameday': day, 'player_id': pid, 'rapm_shrunk': rapm,
        'prior_minutes': mins, 'prior_minutes_per_game': mpg,
        'is_available': avail, 'position': pos,
        'days_since_appearance': gap, 'darko_x_minutes': x,
    } for pid, rapm, mins, mpg, avail, pos, gap, x in rows])


def _run(frame, **kwargs):
    return proj._project_team(
        frame, 'T', pd.Timestamp('2026-01-15'), None,
        30, 50.0, 8, 126.0, 5, **kwargs)


@pytest.fixture()
def widen_on():
    old_w = config.PLAYER_POOL_WIDEN_DARKO
    old_e = config.PLAYER_USE_DARKO_EXPECTED_MINUTES
    config.PLAYER_POOL_WIDEN_DARKO = True
    config.PLAYER_USE_DARKO_EXPECTED_MINUTES = True
    yield
    config.PLAYER_POOL_WIDEN_DARKO = old_w
    config.PLAYER_USE_DARKO_EXPECTED_MINUTES = old_e


def test_widened_members_price_their_segments(widen_on):
    row = _run(_frame())
    assert row['pool_size'] == 7
    assert row['healthy_size'] == 4  # vets + rookie + fringe; out_g removed
    # F = rookie only (scrub refused): -0.5 at weight 20.
    assert row['pl_dpm_f'] == pytest.approx(-0.5)
    assert row['pl_dpm_f_eligible'] == 1.0
    # G = vet (24 x 1.0) + fringe (18 x 2.0), out_g removed: 60/42.
    assert row['pl_dpm_g'] == pytest.approx(60.0 / 42.0)
    # C = vet only (phantom refused): 0.5.
    assert row['pl_dpm_c'] == pytest.approx(0.5)


def test_flag_off_restores_trailing_gates_only(widen_on):
    config.PLAYER_POOL_WIDEN_DARKO = False
    row = _run(_frame())
    assert row['healthy_size'] == 2  # vet_g + vet_c only
    assert 'pl_dpm_f' not in row  # refused segment emits no value (NaN upstream)
    assert row['pl_dpm_f_eligible'] == 0.0
    assert row['pl_dpm_g'] == pytest.approx(1.0)
    assert row['pl_dpm_c'] == pytest.approx(0.5)


def test_ignore_floor_admits_scrub_but_not_phantom(widen_on):
    # scrub_f (5 min, gap NaN, x=2): floor 0 admits him (recency NaN-exempt),
    # widening cannot (x < 5) - so he distinguishes floor-off from widening.
    # phantom_c (60 min, gap 45, no x): recency still refuses him.
    proj.config.PLAYER_POOL_IGNORE_FLOOR = True
    try:
        row = _run(_frame())
    finally:
        proj.config.PLAYER_POOL_IGNORE_FLOOR = False
    assert row['healthy_size'] == 5  # vets + rookie + fringe + scrub
    # F = rookie (x=20 x -0.5) + scrub (x=2 x 3.0): (-10 + 6) / 22.
    assert row['pl_dpm_f'] == pytest.approx(-4.0 / 22.0)
    assert row['pl_dpm_c'] == pytest.approx(0.5)  # phantom still out


def test_ignore_recency_admits_phantom_but_not_scrub(widen_on):
    # phantom_c (60 min, gap 45, no x): recency-off admits him (floor passes,
    # widening cannot - no x). scrub_f (5 min): floor still refuses him.
    proj.config.PLAYER_POOL_IGNORE_RECENCY = True
    try:
        row = _run(_frame())
    finally:
        proj.config.PLAYER_POOL_IGNORE_RECENCY = False
    assert row['healthy_size'] == 5  # vets + rookie + fringe + phantom
    # C = vet (22 x 0.5) + phantom (weight 0: no x, trailing mpg 12? no -
    # phantom HAS trailing mpg 12.0, so weight 12, rating 0.0): 11/34.
    assert row['pl_dpm_c'] == pytest.approx(11.0 / 34.0)
    assert row['pl_dpm_f'] == pytest.approx(-0.5)  # scrub still out


def test_both_gates_off_admits_everyone_rated(widen_on):
    proj.config.PLAYER_POOL_IGNORE_FLOOR = True
    proj.config.PLAYER_POOL_IGNORE_RECENCY = True
    try:
        row = _run(_frame())
    finally:
        proj.config.PLAYER_POOL_IGNORE_FLOOR = False
        proj.config.PLAYER_POOL_IGNORE_RECENCY = False
    assert row['healthy_size'] == 6  # all but out_g (unavailable)


def test_below_floor_and_missing_x_stay_refused(widen_on):
    row = _run(_frame())
    # scrub_f (x=2) and phantom_c (x NaN) never enter any segment:
    # F has exactly the rookie's rating, C exactly the vet's.
    assert row['pl_dpm_f'] == pytest.approx(-0.5)
    assert row['pl_dpm_c'] == pytest.approx(0.5)
