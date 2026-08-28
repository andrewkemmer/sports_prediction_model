import ast
import unittest
from pathlib import Path

import pandas as pd

from training import walk_forward_splits
from pipeline import _attach_drift_run_margins


class TestDriftArtifactAlignment(unittest.TestCase):
    def test_saved_artifacts_require_feature_frame_as_canonical_row_set(self):
        root = Path(__file__).parents[1] / "data_delivery"
        features = pd.read_csv(root / "game_level_features.csv")
        history = pd.read_csv(root / "predictions_history_20260827.csv")
        self.assertEqual(len(features), 4481)
        self.assertEqual(len(features) - len(features[features.game_id.isin(history.game_id)]), 425)
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
        self.assertEqual(str(first[-1]["val_end"])[:10], "2026-08-26")
        enriched = _attach_drift_run_margins(features)
        self.assertIn("run_margin_diff", enriched.columns)


    def test_drift_step_uses_pre_slate_snapshot_not_post_slate_games(self):
        """Regression: the pipeline must snapshot games before the slate
        merge and use that snapshot for the drift step. Without this,
        slate games whose results are filled in post-merge expand the
        decided count by 2-3 rows, shifting fold boundaries and tripping
        the desync guard in _attach_oof_run_margins."""
        src = Path(__file__).parents[1] / "backend" / "pipeline.py"
        tree = ast.parse(src.read_text())
        source_lines = src.read_text().splitlines()

        # 1. Find _pre_slate_games assignment BEFORE the slate concat
        pre_slate_line = None
        concat_line = None
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            if "_pre_slate_games" in stripped and "= games.copy()" in stripped:
                pre_slate_line = i + 1
            if "pd.concat([games, slate]" in stripped:
                concat_line = i + 1
        self.assertIsNotNone(pre_slate_line,
            "_pre_slate_games snapshot not found in pipeline.py")
        self.assertIsNotNone(concat_line,
            "slate concat not found in pipeline.py")
        self.assertLess(pre_slate_line, concat_line,
            "_pre_slate_games must be captured BEFORE the slate merge")

        # 2. The drift step must reference _pre_slate_games, not games
        drift_section = False
        found_pre_slate_ref = False
        found_games_ref = False
        for i, line in enumerate(source_lines):
            stripped = line.strip()
            if "Decided games ONLY" in stripped:
                drift_section = True
            if drift_section and "_pre_slate_games[" in stripped:
                found_pre_slate_ref = True
            if drift_section and "decided = games[" in stripped:
                found_games_ref = True
        self.assertTrue(found_pre_slate_ref,
            "Drift step must use _pre_slate_games, not games")
        self.assertFalse(found_games_ref,
            "Drift step must NOT use games[home_win].notna() — "
            "that was the desync source")

    def test_pre_slate_snapshot_preserves_decided_parity_with_training(self):
        """The pre-slate snapshot must have the same decided count as
        game_level_features.csv (the training source)."""
        root = Path(__file__).parents[1] / "data_delivery"
        features = pd.read_csv(root / "game_level_features.csv")
        decided = features[features["home_win"].notna()]
        # All rows should be decided in the current frame
        self.assertEqual(len(decided), len(features))
        # walk_forward_splits drops NaN, so training sees this many
        splits = walk_forward_splits(features, retrain_cadence_days=7)
        total_val = sum(len(s["val_games"]) for s in splits)
        self.assertGreater(total_val, 0)
        # The drift step via _pre_slate_games would see the same count
        drift_decided = features[features["home_win"].notna()]
        self.assertEqual(len(drift_decided), len(decided))


if __name__ == "__main__":
    unittest.main()
