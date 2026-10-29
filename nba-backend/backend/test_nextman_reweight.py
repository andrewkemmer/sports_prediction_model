"""Unit tests for the next_man_up reweight challenger.

The literal proportional-rescale reading of the fork is a no-op (every
blend divides by its weight sum), so the challenger donates each removed
member's full weight to the top same-position survivor. These tests pin
that behavior on a synthetic pool: the legacy arm must drop the out
player's weight, the challenger must move it onto exactly one survivor,
and a donor with no same-position survivor must vanish as in legacy.
"""
import pandas as pd

import lineup_projection as proj


def _pool():
    day = pd.Timestamp('2026-01-15')
    rows = [
        # player_id, rapm, minutes, mpg, available, position
        ('c_out', 2.0, 300.0, 30.0, False, 'C'),
        ('c1', 1.0, 200.0, 20.0, True, 'C'),
        ('c2', -1.0, 100.0, 10.0, True, 'C'),
        ('g1', 5.0, 340.0, 34.0, True, 'G'),
    ]
    return pd.DataFrame([{
        'team': 'T', 'gameday': day, 'player_id': pid, 'rapm_shrunk': rapm,
        'prior_minutes': mins, 'prior_minutes_per_game': mpg,
        'is_available': avail, 'position': pos,
    } for pid, rapm, mins, mpg, avail, pos in rows])


def _run(frame, **kwargs):
    return proj._project_team(
        frame, 'T', pd.Timestamp('2026-01-15'), None,
        30, 1.0, 8, 100.0, 5, **kwargs)


def test_legacy_drops_out_weight():
    row = _run(_pool())
    # Survivor mean only: (20*1 + 10*-1) / 30.
    assert abs(row['pl_dpm_c'] - (10.0 / 30.0)) < 1e-9
    # Guard segment untouched by the centre absence.
    assert abs(row['pl_dpm_g'] - 5.0) < 1e-9


def test_next_man_up_moves_weight_to_top_survivor():
    diag = {}
    row = _run(_pool(), reweight='next_man_up', diagnostics=diag)
    # c1 absorbs c_out's 30: (50*1 + 10*-1) / 60.
    assert abs(row['pl_dpm_c'] - (40.0 / 60.0)) < 1e-9
    assert abs(row['pl_dpm_g'] - 5.0) < 1e-9
    assert diag['donors'] == 1
    assert abs(diag['absorbed_weight'] - 30.0) < 1e-9
    assert diag['vanished_no_survivor'] == 0


def test_donor_without_same_position_survivor_vanishes():
    frame = _pool()
    frame.loc[frame.player_id == 'c_out', 'position'] = 'Z'
    diag = {}
    row = _run(frame, reweight='next_man_up', diagnostics=diag)
    assert abs(row['pl_dpm_c'] - (10.0 / 30.0)) < 1e-9
    assert diag['vanished_no_survivor'] == 1


def test_unknown_reweight_mode_matches_legacy():
    row = _run(_pool(), reweight='not_a_mode')
    assert abs(row['pl_dpm_c'] - (10.0 / 30.0)) < 1e-9


def _waterfall_pool():
    day = pd.Timestamp('2026-01-15')
    rows = [
        # c_out's 30 fills c1 (30 -> cap 40) then spills 20 onto c2 (10 -> 30).
        ('c_out', 2.0, 300.0, 30.0, False, 'C'),
        ('c1', 1.0, 300.0, 30.0, True, 'C'),
        ('c2', -1.0, 100.0, 10.0, True, 'C'),
        ('g1', 5.0, 340.0, 34.0, True, 'G'),
    ]
    return pd.DataFrame([{
        'team': 'T', 'gameday': day, 'player_id': pid, 'rapm_shrunk': rapm,
        'prior_minutes': mins, 'prior_minutes_per_game': mpg,
        'is_available': avail, 'position': pos,
    } for pid, rapm, mins, mpg, avail, pos in rows])


def test_waterfall_caps_and_spills_to_next_survivor():
    diag = {}
    row = _run(_waterfall_pool(), reweight='waterfall', diagnostics=diag)
    # (40*1 + 30*-1) / 70: diluted vs winner-take-all, moved vs legacy 0.5.
    assert abs(row['pl_dpm_c'] - (10.0 / 70.0)) < 1e-9
    assert abs(row['pl_dpm_g'] - 5.0) < 1e-9
    assert diag['donors'] == 1
    assert abs(diag['absorbed_weight'] - 30.0) < 1e-9
    assert diag['spills'] >= 1
    assert diag['vanished_no_survivor'] == 0


def test_waterfall_never_exceeds_cap():
    # A 39-mpg survivor can take only 1 more; the rest must spill or vanish.
    frame = _waterfall_pool()
    frame.loc[frame.player_id == 'c1', 'prior_minutes_per_game'] = 39.0
    frame.loc[frame.player_id == 'c1', 'prior_minutes'] = 390.0
    diag = {}
    row = _run(frame, reweight='waterfall', diagnostics=diag)
    # c1 39->40, c2 10->39: (40*1 + 39*-1) / 79.
    assert abs(row['pl_dpm_c'] - (1.0 / 79.0)) < 1e-9
    assert abs(diag['absorbed_weight'] - 30.0) < 1e-9
