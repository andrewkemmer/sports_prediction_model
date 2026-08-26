"""Tests for the run-engine monitor artifact (pipeline + run_engine).

Covers: JSON schema (per-line + rolling + fit + persisted flag),
persist-failure path returns flags without crashing, _run_engine_monitor_json
produces valid JSON, rolling history folding, and guardrails.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestMonitorSchema(unittest.TestCase):
    """The JSON written by _run_engine_monitor_json has the required shape."""

    def _minimal_block(self) -> dict:
        """Synthetic monitor_block with all required keys."""
        base_card = {
            "engine_logloss": 0.68, "engine_brier": 0.247,
            "engine_ece_raw": 0.019, "baseline_rate": 0.45,
            "baseline_logloss": 0.693, "baseline_brier": 0.25,
            "engine_logloss_calibrated": 0.67,
            "engine_ece_calibrated": 0.014,
            "predicted_mean": 0.452,
            "n": 4000,
            "holdout": {"n": 280, "engine_logloss": 0.70,
                        "engine_brier": 0.26,
                        "engine_ece_raw": 0.05,
                        "engine_logloss_calibrated": 0.69,
                        "engine_ece_calibrated": 0.048,
                        "baseline_rate": 0.44,
                        "baseline_logloss": 0.693,
                        "baseline_brier": 0.25,
                        "predicted_mean": 0.445,
                        "beats_baseline_logloss": True},
            "beats_baseline_logloss": True,
        }
        metrics = {}
        for name in ("over_7_5", "over_8_5", "over_9_5",
                      "home_cover_1_5", "home_cover_2_5", "derived_moneyline"):
            metrics[name] = dict(base_card)
        return {
            "market_metrics": metrics,
            "alpha_home": {"alpha_hat": 1.2, "lambda_grid": [1, 2, 3],
                           "alpha_curve": [1.1, 1.2, 1.3]},
            "alpha_away": {"alpha_hat": 1.3, "lambda_grid": [1, 2, 3],
                           "alpha_curve": [1.2, 1.3, 1.4]},
            "fit_check_alpha_lambda": {
                "home": [{"k": 0, "observed": 0.02, "modeled": 0.018},
                         {"k": 1, "observed": 0.08, "modeled": 0.075},
                         {"k": 10, "observed": 0.01, "modeled": 0.012}],
                "away": [{"k": 0, "observed": 0.025, "modeled": 0.022},
                         {"k": 1, "observed": 0.09, "modeled": 0.085},
                         {"k": 10, "observed": 0.008, "modeled": 0.01}],
            },
            "variance_check": {
                "home_raw_var": 9.5, "home_nb_var": 8.2,
                "away_raw_var": 9.1, "away_nb_var": 7.8,
            },
            "mc_meta": {"n_samples": 10000, "seed": 42},
            "holdout_gate": {"n_pre": 4000, "n_holdout": 280},
            "phase1": {
                "dispersion_ratio": {"home": 2.3, "away": 2.1},
            },
            "line_grid": [7.5, 8.0, 8.5, 9.0, 9.5],
        }

    def test_schema_has_all_top_level_keys(self):
        from pipeline import _run_engine_monitor_json
        block = self._minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826",
                                               True, None)
            data = json.loads(path.read_text())
        self.assertEqual(data["schema"], "run-engine-monitor/v1")
        self.assertEqual(data["date"], "20260826")
        self.assertTrue(data["markets_persisted"])
        self.assertIsNone(data["markets_persist_error"])
        self.assertIn("per_line", data)
        self.assertIn("rolling", data)
        self.assertIn("fit", data)

    def test_per_line_has_all_six_markets(self):
        from pipeline import _run_engine_monitor_json
        block = self._minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826",
                                               True, None)
            data = json.loads(path.read_text())
        expected = {"over_7_5", "over_8_5", "over_9_5",
                    "home_cover_1_5", "home_cover_2_5", "derived_moneyline"}
        self.assertEqual(set(data["per_line"].keys()), expected)

    def test_per_line_card_has_required_fields(self):
        from pipeline import _run_engine_monitor_json
        block = self._minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826",
                                               True, None)
            data = json.loads(path.read_text())
        card = data["per_line"]["over_8_5"]
        for key in ("n", "base_rate", "predicted_mean", "ece_raw",
                     "ece_calibrated", "brier", "logloss"):
            self.assertIn(key, card, f"missing per_line field: {key}")
        self.assertIn("holdout", card)
        h = card["holdout"]
        # Holdout sub-block is passthrough from run_engine with engine_ prefix
        for key in ("n", "engine_ece_raw", "engine_ece_calibrated",
                     "engine_brier", "engine_logloss"):
            self.assertIn(key, h, f"missing holdout field: {key}")

    def test_fit_has_required_sections(self):
        from pipeline import _run_engine_monitor_json
        block = self._minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826",
                                               True, None)
            data = json.loads(path.read_text())
        fit = data["fit"]
        self.assertIn("alpha_home", fit)
        self.assertIn("alpha_away", fit)
        self.assertIn("dispersion_chi2_per_df", fit)
        self.assertIn("home", fit["dispersion_chi2_per_df"])
        self.assertIn("away", fit["dispersion_chi2_per_df"])
        self.assertIn("fit_tables", fit)
        self.assertIn("home", fit["fit_tables"])
        self.assertIn("away", fit["fit_tables"])
        self.assertIn("variance_check", fit)
        self.assertIn("mc_meta", fit)

    def test_rolling_starts_with_current_when_no_prior(self):
        """With a non-empty block and no prior monitor files, each line has
        exactly 1 entry (today)."""
        from pipeline import _run_engine_monitor_json
        block = self._minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826",
                                               True, None)
            data = json.loads(path.read_text())
        rolling = data["rolling"]
        self.assertIsInstance(rolling, dict)
        for key in ("over_7_5", "over_8_5", "over_9_5",
                     "home_cover_1_5", "home_cover_2_5", "derived_moneyline"):
            self.assertIn(key, rolling)
            self.assertEqual(len(rolling[key]), 1)
            self.assertEqual(rolling[key][0]["date"], "2026-08-26")


class TestPersistFailurePath(unittest.TestCase):
    """markets_persisted=False is reflected in the monitor JSON."""

    def test_persist_failure_flagged(self):
        from pipeline import _run_engine_monitor_json
        block = {
            "market_metrics": {},
            "alpha_home": {}, "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {},
            "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(
                    block, "20260826", False,
                    "ValueError: markets frame contains NaNs")
            data = json.loads(path.read_text())
        self.assertFalse(data["markets_persisted"])
        self.assertIn("NaNs", data["markets_persist_error"])


class TestRollingHistoryFolding(unittest.TestCase):
    """Prior monitor files are folded into the rolling series."""

    def test_prior_monitors_folded(self):
        from pipeline import _run_engine_monitor_json
        block = {
            "market_metrics": {
                "over_8_5": {
                    "engine_logloss": 0.68, "engine_brier": 0.247,
                    "engine_ece_raw": 0.019, "baseline_rate": 0.45,
                    "baseline_logloss": 0.693, "baseline_brier": 0.25,
                    "engine_logloss_calibrated": 0.67,
                    "engine_ece_calibrated": 0.014,
                    "predicted_mean": 0.452, "n": 4000,
                    "holdout": {"n": 280, "engine_logloss": 0.70,
                                "engine_brier": 0.26,
                                "engine_ece_raw": 0.05,
                                "engine_logloss_calibrated": 0.69,
                                "engine_ece_calibrated": 0.048,
                                "baseline_rate": 0.44,
                                "baseline_logloss": 0.693,
                                "baseline_brier": 0.25,
                                "predicted_mean": 0.445,
                                "beats_baseline_logloss": True},
                    "beats_baseline_logloss": True,
                },
            },
            "alpha_home": {}, "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {},
            "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        # Create a prior monitor file
        prior = {
            "schema": "run-engine-monitor/v1",
            "date": "20260825",
            "per_line": {
                "over_8_5": {
                    "n": 3900, "ece_calibrated": 0.015,
                    "brier": 0.250, "logloss": 0.685,
                    "predicted_mean": 0.448, "base_rate": 0.449,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            prior_path = p / "run_engine_monitor_20260825.json"
            prior_path.write_text(json.dumps(prior))
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(
                    block, "20260826", True, None)
            data = json.loads(path.read_text())
        rolling = data["rolling"]["over_8_5"]
        self.assertEqual(len(rolling), 2)  # prior + today
        self.assertEqual(rolling[0]["date"], "2026-08-25")
        self.assertEqual(rolling[1]["date"], "2026-08-26")

    def test_rolling_trims_to_45_days(self):
        from pipeline import _run_engine_monitor_json
        block = {
            "market_metrics": {
                "over_8_5": {
                    "engine_logloss": 0.68, "engine_brier": 0.247,
                    "engine_ece_raw": 0.019, "baseline_rate": 0.45,
                    "baseline_logloss": 0.693, "baseline_brier": 0.25,
                    "engine_logloss_calibrated": 0.67,
                    "engine_ece_calibrated": 0.014,
                    "predicted_mean": 0.452, "n": 4000,
                    "holdout": {"n": 280, "engine_logloss": 0.70,
                                "engine_brier": 0.26,
                                "engine_ece_raw": 0.05,
                                "engine_logloss_calibrated": 0.69,
                                "engine_ece_calibrated": 0.048,
                                "baseline_rate": 0.44,
                                "baseline_logloss": 0.693,
                                "baseline_brier": 0.25,
                                "predicted_mean": 0.445,
                                "beats_baseline_logloss": True},
                    "beats_baseline_logloss": True,
                },
            },
            "alpha_home": {}, "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {},
            "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        # Create 50 prior monitor files
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(50):
                d = f"2026070{i % 10}" if i < 10 else f"202607{i}"
                # Fix the date format
                from datetime import date, timedelta
                dt = date(2026, 8, 26) - timedelta(days=50 - i)
                d = dt.strftime("%Y%m%d")
                prior = {
                    "schema": "run-engine-monitor/v1",
                    "date": d,
                    "per_line": {"over_8_5": {"n": 3900,
                                               "ece_calibrated": 0.015,
                                               "brier": 0.250,
                                               "logloss": 0.685,
                                               "predicted_mean": 0.448,
                                               "base_rate": 0.449}},
                }
                (p / f"run_engine_monitor_{d}.json").write_text(
                    json.dumps(prior))
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(
                    block, "20260826", True, None)
            data = json.loads(path.read_text())
        rolling = data["rolling"]["over_8_5"]
        self.assertLessEqual(len(rolling), 46)  # 45 prior + 1 today max


class TestRunWithRealArtifact(unittest.TestCase):
    """If a real run_engine_markets CSV exists, the monitor can be built."""

    def test_build_from_real_csv_if_available(self):
        """Smoke test: the monitor builder does not crash with a synthetic
        block that mirrors the real artifact shape."""
        from pipeline import _run_engine_monitor_json
        # Minimal block that exercises all code paths
        block = {
            "market_metrics": {},
            "alpha_home": {},
            "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {},
            "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(
                    block, "20260826", True, None)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertEqual(data["schema"], "run-engine-monitor/v1")


if __name__ == "__main__":
    unittest.main()
