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
