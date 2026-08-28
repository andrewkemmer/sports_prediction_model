"""Tests for model_version_history.json snapshots (update_model_version_history).

Covers: snapshot row correctness (fixture), append/merge by version, cap at
VERSION_HISTORY_CAP, atomic-write leftovers, partial-input refusal, monitor-JSON
embedding under the frontend's `version_history` key.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import pipeline as pl
from training import (
    VERSION_HISTORY_CAP,
    VERSION_HISTORY_FILE,
    update_model_version_history,
)

_METRICS = {
    "auc": 0.5542, "brier": 0.2484, "logloss": 0.6901, "ece": 0.0322,
    "brier_calibrated": 0.2476, "logloss_calibrated": 0.6883,
    "ece_calibrated": 0.0164,
}
_ROSTER = [
    {"name": "xgboost", "weight": 0.45},
    {"name": "logistic", "weight": 0.20},
    {"name": "lightgbm", "weight": 0.07},
]
_CAL = {"method": "platt", "a": 0.57497, "b": 0.011677, "n": 4016}


class TestVersionHistory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        for mod_name in ("training", "pipeline"):
            patcher = patch(f"{mod_name}.DATA_DELIVERY_DIR", self.data_dir)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _path(self) -> Path:
        return self.data_dir / VERSION_HISTORY_FILE

    def test_snapshot_row_correctness(self):
        update_model_version_history(_METRICS, "v2026.08.24",
                                     ensemble_info=_ROSTER, calibrator=_CAL)
        rows = json.loads(self._path().read_text())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["version"], "v2026.08.24")
        self.assertEqual(row["weights"],
                         {"xgboost": 0.45, "logistic": 0.2, "lightgbm": 0.07})
        for k, v in _METRICS.items():
            self.assertAlmostEqual(row[k], v)
        self.assertEqual(row["calibration"], {"a": 0.57497, "b": 0.011677, "n": 4016})

    def test_merge_replaces_same_version(self):
        update_model_version_history(_METRICS, "v2026.08.24",
                                     ensemble_info=_ROSTER, calibrator=_CAL)
        rerun = dict(_METRICS, auc=0.5555)
        update_model_version_history(rerun, "v2026.08.24",
                                     ensemble_info=_ROSTER, calibrator=_CAL)
        rows = json.loads(self._path().read_text())
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["auc"], 0.5555)

    def test_append_and_cap_at_n(self):
        class _FakeNow:
            def __init__(self, day):
                self._day = day

            def now(self):
                class _T:
                    def strftime(_self, *_a):
                        return f"2026-08-{self._day:02d}"
                return _T()

        for day in range(1, VERSION_HISTORY_CAP + 5):
            with patch("training.datetime", _FakeNow(day)):
                update_model_version_history(
                    dict(_METRICS, auc=0.50 + day / 100), f"v2026.{day:02d}.01",
                    ensemble_info=_ROSTER, calibrator=_CAL)
        rows = json.loads(self._path().read_text())
        self.assertEqual(len(rows), VERSION_HISTORY_CAP)
        versions = [r["version"] for r in rows]
        self.assertNotIn("v2026.01.01", versions)  # oldest evicted
        self.assertIn(f"v2026.{VERSION_HISTORY_CAP + 4:02d}.01", versions)

    def test_partial_inputs_refused(self):
        # No roster → no row (weights must come from the run, never hardcode).
        with self.assertLogs("training", level="WARNING"):
            out = update_model_version_history(_METRICS, "v2026.08.24",
                                               ensemble_info=None)
        self.assertIsNone(out)
        self.assertFalse(self._path().exists())

        # No metrics → no row.
        with self.assertLogs("training", level="WARNING"):
            out = update_model_version_history({}, "v2026.08.24",
                                               ensemble_info=_ROSTER)
        self.assertIsNone(out)
        self.assertFalse(self._path().exists())

    def test_atomic_write_leaves_no_tmp(self):
        update_model_version_history(_METRICS, "v2026.08.24",
                                     ensemble_info=_ROSTER, calibrator=_CAL)
        tmp = self._path().with_suffix(".json.tmp")
        self.assertFalse(tmp.exists(), "atomic rename left a temp file behind")
        json.loads(self._path().read_text())  # valid JSON

    def test_monitor_json_embeds_version_history(self):
        update_model_version_history(_METRICS, "v2026.08.24",
                                     ensemble_info=_ROSTER, calibrator=_CAL)
        path = pl._model_monitor_json(dict(_METRICS), pd.DataFrame(), "20260825")
        mon = json.loads(Path(path).read_text())
        vh = mon["version_history"]
        self.assertEqual(len(vh), 1)
        self.assertEqual(vh[0]["version"], "v2026.08.24")
        self.assertIn("weights", vh[0])

    def test_monitor_json_falls_back_to_legacy_rows(self):
        legacy = [{"version": "v2026.08.01", "date": "2026-08-01",
                   "auc": 0.55, "brier": 0.25, "notes": "old run"}]
        (self.data_dir / "model_history.json").write_text(json.dumps(legacy))
        path = pl._model_monitor_json({"auc": 0.55}, pd.DataFrame(), "20260825")
        mon = json.loads(Path(path).read_text())
        self.assertEqual(mon["version_history"], legacy)


if __name__ == "__main__":
    unittest.main()
