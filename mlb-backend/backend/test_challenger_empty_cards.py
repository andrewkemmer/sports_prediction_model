from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distributions as dist


def test_empty_home_lean_cards_are_valid_without_fabricated_scores():
    frame = pd.DataFrame({"game_pk": [1, 2], "game_date": ["2026-01-01", "2026-01-02"],
                          "kind": "oof", "home_score": [3, 2], "away_score": [4, 5],
                          "total_runs": [7, 7], "home_expected_runs": [3., 3.],
                          "away_expected_runs": [4., 4.], "p_home_cover_1_5": [.2, .3],
                          "p_home_win_derived": [.4, .3], "ml_win_prob": [.4, .3]})
    cards = dist.compute_winner_cards(frame, frame)
    for key in ('run_line', 'derived_ml'):
        assert cards[key]['n'] == 0
        assert cards[key]['win_rate'] is None
        assert cards[key]['brier'] is None
        assert cards[key]['logloss'] is None
    empty = dist._winner_card_stats(np.array([]), np.array([]), None, np.array([], dtype=bool))
    assert empty['n'] == 0 and empty['ece_calibrated'] is None


def test_winner_card_auc_ranks_the_full_pool_not_the_lean_gate():
    """AUC is a ranking metric and must rank every graded row (2026-10-13).

    The run_line / derived_ml cards grade only home-lean rows so their
    calibration metrics describe the population a bet lands on — but
    gating AUC to P > 0.5 truncates the score to (0.5, 1] and discards
    every pair spanning the 50/50 line, which is where the discrimination
    lives. On NBA's 2026-10-13 artifact that read as derived-ML AUC
    0.6435 instead of 0.7304, and 0.6435 is also the old PICK-SIDE number
    — hence the "still using the pick side?" misread. ``auc_n`` states the
    pool so it is never read as the card's ``n``.

    These fixtures are single-class once gated, so a gated AUC would be
    undefined while the full pool separates perfectly.
    """
    frame = pd.DataFrame({
        "game_pk": [1, 2, 3, 4],
        "game_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "kind": ["oof"] * 4,
        # margins 2, -3, 2, 5 -> home covers -1.5 in every game but #2
        "home_score": [10, 6, 11, 12],
        "away_score": [8, 9, 9, 7],
        "total_runs": [18, 15, 20, 19],
        "home_expected_runs": [4., 4., 4., 4.],
        "away_expected_runs": [4., 4., 4., 4.],
        # The only below-50% price is also the only home loss / non-cover.
        "p_home_cover_1_5": [.60, .30, .55, .70],
        "p_home_win_derived": [.62, .35, .55, .70],
        "ml_win_prob": [.60, .40, .55, .70],
    })
    cards = dist.compute_winner_cards(frame, frame)

    ml, rl = cards['derived_ml'], cards['run_line']
    # Calibration keeps the documented lean-gated populations.
    assert ml['n'] == 3 and ml['actual_win_rate'] == 1.0
    assert rl['n'] == 3 and rl['actual_win_rate'] == 1.0
    # AUC ranks all four graded rows instead — and separates them
    # perfectly, which the single-class gated pools could never do.
    assert ml['auc_n'] == 4 and ml['auc'] == 1.0
    assert rl['auc_n'] == 4 and rl['auc'] == 1.0
    assert ml['auc_n'] >= ml['n'] and rl['auc_n'] >= rl['n']
    # The control: rank the gated pool and there is nothing to rank.
    gated = frame['p_home_win_derived'].to_numpy(float) > 0.5
    won = (frame['home_score'] > frame['away_score']).to_numpy(float)
    assert set(won[gated]) == {1.0}
    assert dist._safe_auc(frame['p_home_win_derived'].to_numpy(float)[gated],
                          won[gated]) is None
