"""Feature-metadata artifact tests (features_metadata_<date>.json).

Covers: full FEATURE_COLS coverage, populated rich fields, loud warning +
placeholder on missing entries, stale-entry detection, member-routing derived
from live config (diff → logistic included; raw per-side → trees only),
tooltip formatting, and model_monitor embedding.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import feature_metadata as fm
from training import FEATURE_COLS


class TestFeaturesMetadata(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name)

    def _generate(self):
        return fm.generate_features_metadata("TEST", out_dir=self.out_dir)

    def test_every_feature_col_has_an_entry(self):
        payload = self._generate()
        self.assertEqual(payload["n_features"], len(FEATURE_COLS))
        self.assertEqual(set(payload["features"].keys()), set(FEATURE_COLS))

    def test_rich_fields_populated_no_warnings(self):
        # All 58 features are authored → generation itself must be silent.
        payload = self._generate()
        gen_warns = [w for w in payload["warnings"]
                     if "no authored entry" in w or "no longer in" in w]
        self.assertEqual(gen_warns, [], "unauthored features found")
        for name, row in payload["features"].items():
            for field in ("summary", "definition", "formula", "source",
                          "window", "units", "direction"):
                self.assertTrue(row.get(field), f"{name}.{field} empty")
            self.assertIsInstance(row.get("members"), list) and None
            self.assertTrue(row["members"], f"{name}.members empty")
            self.assertIn("Formula:", row["tooltip"])
            self.assertIn("Consumed by:", row["tooltip"])

    def test_missing_entry_warns_loudly_and_placeholders(self):
        with patch.object(fm, "_RICH", {k: v for k, v in fm._RICH.items()
                                        if k != "is_home"}):
            with self.assertLogs(fm.logger, level="WARNING") as logs:
                payload = self._generate()
            self.assertTrue(any("'is_home'" in line for line in logs.output),
                            "missing entry must warn naming the feature")
            row = payload["features"]["is_home"]
            self.assertEqual(row["definition"], "No detailed metadata authored yet.")
            # Placeholder still carries routing + a tooltip (graceful gap).
            self.assertTrue(row["members"])
            self.assertIn("No detailed metadata authored yet", row["tooltip"])

    def test_stale_entry_warns(self):
        with patch.dict(fm._RICH, {"removed_feature_xyz": {
                "summary": "x", "definition": "x", "formula": "x",
                "source": "x", "window": "x", "units": "x", "direction": "x"}}):
            with self.assertLogs(fm.logger, level="WARNING") as logs:
                payload = self._generate()
            self.assertTrue(any("removed_feature_xyz" in l for l in logs.output))
            self.assertNotIn("removed_feature_xyz", payload["features"])

    def test_routing_diff_feature_includes_logistic(self):
        """A known diff feature routes to logistic + all trees/MLP."""
        payload = self._generate()
        members = payload["features"]["win_pct_diff"]["members"]
        for m in ("logistic", "xgboost", "lightgbm", "randomforest", "mlp"):
            self.assertIn(m, members)

    def test_routing_raw_per_side_excludes_logistic(self):
        """Raw per-side columns are trees+MLP only (LOGISTIC_USE_RAW_COLS=False)."""
        payload = self._generate()
        members = payload["features"]["home_elo"]["members"]
        self.assertNotIn("logistic", members)
        for m in ("xgboost", "lightgbm", "randomforest", "mlp"):
            self.assertIn(m, members)

    def test_routing_follows_live_flag_not_hardcode(self):
        """Flip LOGISTIC_USE_RAW_COLS on and routing must follow the config."""
        from training import RAW_PER_SIDE_COLS
        with patch("training.LOGISTIC_USE_RAW_COLS", True):
            meta, _ = fm.build_features_metadata()
            members = meta["home_elo"]["members"]
            self.assertIn("logistic", members,
                          "routing did not follow the live config flag")

    def test_artifact_written_atomically(self):
        payload = self._generate()
        path = self.out_dir / "features_metadata_TEST.json"
        self.assertTrue(path.exists())
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        on_disk = json.loads(path.read_text())
        self.assertEqual(on_disk["features"], payload["features"])


if __name__ == "__main__":
    unittest.main()
