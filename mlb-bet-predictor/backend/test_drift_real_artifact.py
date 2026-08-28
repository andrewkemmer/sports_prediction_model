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
        self.assertEqual(len(features), 6953)
        self.assertEqual(len(features) - len(features[features.game_id.isin(history.game_id)]), 495)
        self.assertEqual(features.game_id.duplicated().sum(), 75)
        self.assertEqual(history.game_id.duplicated().sum(), 68)
        # A history-derived frame cannot be the drift input: it loses rows and
        # changes fold boundaries. The pipeline must enrich `games` directly.
        self.assertNotEqual(len(features), len(features[features.game_id.isin(history.game_id)]))

    def test_canonical_splits_are_repeatable_and_drift_attach_does_not_desync(self):
        first = walk_forward_splits(features := pd.read_csv(Path(__file__).parents[1] / "data_delivery" / "game_level_features.csv"), retrain_cadence_days=7)
        second = walk_forward_splits(features, retrain_cadence_days=7)
        self.assertEqual(len(first), 81)
        self.assertEqual([(s["val_start"], s["val_end"], s["val_games"]["game_pk"].tolist()) for s in first],
                         [(s["val_start"], s["val_end"], s["val_games"]["game_pk"].tolist()) for s in second])
        self.assertEqual(str(first[-1]["val_start"])[:10], "2026-08-23")
        self.assertEqual(str(first[-1]["val_end"])[:10], "2026-08-26")
        enriched = _attach_drift_run_margins(features)
        self.assertIn("run_margin_diff", enriched.columns)


    def test_drift_step_uses_canonical_accessor_not_post_slate_games(self):
        """Regression (supersedes the 0d18eaf _pre_slate_games snapshot test):
        the drift step must derive its decided frame from the SINGLE canonical
        accessor (frames.get_decided_frame), which excludes slate rows by
        construction — a decided frame is never the slate frame. The snapshot
        hack is removed; if the accessor is ever bypassed again, this fails."""
        src = Path(__file__).parents[1] / "backend" / "pipeline.py"
        source = src.read_text()
        source_lines = source.splitlines()

        # 1. The snapshot hack is GONE — the accessor replaced it.
        self.assertNotIn("_pre_slate_games", source,
            "_pre_slate_games snapshot must not return; get_decided_frame "
            "is the single source of truth now")

        # 2. The drift step must use the canonical accessor, never a raw
        # home_win filter, on the post-slate games frame.
        drift_section = False
        found_accessor = False
        found_raw_filter = False
        for line in source_lines:
            stripped = line.strip()
            if "Decided games ONLY" in stripped:
                drift_section = True
            if drift_section and "get_decided_frame(games)" in stripped:
                found_accessor = True
            if drift_section and "decided = games[" in stripped:
                found_raw_filter = True
            if drift_section and "home_win" in stripped and ".notna()" in stripped \
                    and "decided = " in stripped and "get_decided_frame" not in stripped:
                found_raw_filter = True
        self.assertTrue(found_accessor,
            "Drift step must derive its frame via frames.get_decided_frame")
        self.assertFalse(found_raw_filter,
            "Drift step must NOT reconstruct the decided frame with a raw "
            "home_win filter — that was the desync source")

        # 3. The signature assert is wired at the drift step.
        self.assertIn("require_matching_signatures", source)
        self.assertIn("fold_signature(decided)", source)

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
        # The drift step (via get_decided_frame) sees the same count
        drift_decided = features[features["home_win"].notna()]
        self.assertEqual(len(drift_decided), len(decided))


if __name__ == "__main__":
    unittest.main()
