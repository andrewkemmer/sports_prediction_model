import unittest
from pathlib import Path

import pandas as pd

from training import walk_forward_splits
from pipeline import _attach_drift_run_margins


class TestDriftArtifactAlignment(unittest.TestCase):
    def test_saved_artifacts_require_feature_frame_as_canonical_row_set(self):
        root = Path(__file__).parents[1] / "data_delivery"
        features = pd.read_csv(root / "game_level_features.csv")
        history = pd.read_csv(root / "predictions_history_20260826.csv")
        self.assertEqual(len(features), 4466)
        self.assertEqual(len(features) - len(features[features.game_id.isin(history.game_id)]), 450)
        self.assertEqual(features.game_id.duplicated().sum(), 46)
        self.assertEqual(history.game_id.duplicated().sum(), 44)
        # A history-derived frame cannot be the drift input: it loses rows and
        # changes fold boundaries. The pipeline must enrich `games` directly.
        self.assertNotEqual(len(features), len(features[features.game_id.isin(history.game_id)]))

    def test_canonical_splits_are_repeatable_and_drift_attach_does_not_desync(self):
        first = walk_forward_splits(features := pd.read_csv(Path(__file__).parents[1] / "data_delivery" / "game_level_features.csv"), retrain_cadence_days=7)
        second = walk_forward_splits(features, retrain_cadence_days=7)
        self.assertEqual(len(first), 51)
        self.assertEqual([(s["val_start"], s["val_end"], s["val_games"]["game_pk"].tolist()) for s in first],
                         [(s["val_start"], s["val_end"], s["val_games"]["game_pk"].tolist()) for s in second])
        self.assertEqual(str(first[-1]["val_start"])[:10], "2026-08-22")
        self.assertEqual(str(first[-1]["val_end"])[:10], "2026-08-25")
        enriched = _attach_drift_run_margins(features)
        self.assertIn("run_margin_diff", enriched.columns)


if __name__ == "__main__":
    unittest.main()
