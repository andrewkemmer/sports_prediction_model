"""Tests for the run-engine monitor artifact (pipeline + run_engine), v2.

Covers: JSON schema (three winner cards + rolling + fit + persisted flag),
persist-failure path returns flags without crashing, _run_engine_monitor_json
produces valid JSON, v2 rolling history folding (renamed field), and
guardrails.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _winner_card(auc=None, n: int = 4000) -> dict:
    return {
        "n": n,
        "actual_win_rate": 0.5414,
        "win_rate": 0.5414,
        "predicted_mean": 0.5310,
        "auc": auc,
        "ece_raw": 0.019,
        "ece_calibrated": 0.014,
        "brier": 0.247,
        "logloss": 0.680,
        "logloss_calibrated": 0.670,
        "beats_baseline_logloss": True,
        "holdout": {"n": 275, "actual_win_rate": 0.5891, "win_rate": 0.5891,
                    "predicted_mean": 0.5600, "ece_raw": 0.05,
                    "ece_calibrated": 0.048, "brier": 0.26, "logloss": 0.70,
                    "baseline_rate": 0.44, "baseline_logloss": 0.693,
                    "beats_baseline_logloss": True},
    }


def _minimal_block() -> dict:
    """Synthetic monitor_block with the v2 winner-cards shape."""
    metrics = {
        "over_8_5": {"auc": 0.5505, "n": 4369},
        "home_cover_1_5": {"auc": 0.5432, "n": 4369},
        "derived_moneyline": {"auc": 0.5545, "n": 4369},
    }
    return {
        "winner_cards": {
            "over_under": _winner_card(auc=0.5505),
            "run_line": _winner_card(auc=0.5432),
            "derived_ml": _winner_card(auc=0.5545),
        },
        "market_metrics": metrics,
        # Fit block in the REAL run-engine schema (mirrors run_engine_daily's
        # monitor-embed dict and data_delivery/run_engine_monitor_*.json):
        # alpha curves with selection bins, fit-check tables with
        # observed_p/modeled_p + unicode tail labels, per-side variance
        # implied/observed, mc_meta.n_draws.
        "alpha_home": {
            "form": "piecewise", "direction": "rising",
            "lam": [4.1, 4.4, 4.8], "alpha": [0.22, 0.27, 0.28],
            "selection": {"chosen": "piecewise", "bins": [
                {"count": 573, "mean_lam": 4.1577, "alpha": 0.2165},
                {"count": 590, "mean_lam": 4.3623, "alpha": 0.2686},
                {"count": 580, "mean_lam": 4.4264, "alpha": 0.3113}]},
            "fitted_on": "pre-holdout OOF only", "cap": 2.0,
            "min_bin_count": 250,
        },
        "alpha_away": {
            "form": "linear", "a": 0.6218, "b": -0.0633,
            "selection": {"chosen": "linear", "bins": [
                {"count": 582, "mean_lam": 3.8069, "alpha": 0.4196},
                {"count": 581, "mean_lam": 4.1625, "alpha": 0.3708},
                {"count": 581, "mean_lam": 4.2918, "alpha": 0.3522}]},
            "fitted_on": "pre-holdout OOF only", "cap": 2.0,
            "min_bin_count": 250,
        },
        "fit_check_alpha_lambda": {
            "home": [{"k": 0, "observed_p": 0.0597, "modeled_p": 0.0534},
                     {"k": 1, "observed_p": 0.108, "modeled_p": 0.108},
                     {"k": "≤1", "observed_p": 0.1678,
                      "modeled_p": 0.1613},
                     {"k": "≥10", "observed_p": 0.0691,
                      "modeled_p": 0.0733}],
            "away": [{"k": 0, "observed_p": 0.0732, "modeled_p": 0.0712},
                     {"k": 1, "observed_p": 0.1176, "modeled_p": 0.1224},
                     {"k": "≤1", "observed_p": 0.1909,
                      "modeled_p": 0.1936},
                     {"k": "≥10", "observed_p": 0.0838,
                      "modeled_p": 0.078}],
        },
        "variance_check": {
            "home": {"implied_var": 9.938, "observed_var": 9.886,
                      "phase2_implied_var": 9.992},
            "away": {"implied_var": 10.902, "observed_var": 11.03,
                      "phase2_implied_var": 11.054},
        },
        "mc_meta": {"n_draws": 10000, "requested_draws": 10000,
                    "reason": "default", "mc_se_totals_max": 0.005},
        "holdout_gate": {"n_pre": 4000, "n_holdout": 280},
        "phase1": {"dispersion_ratio": {"home": 2.3, "away": 2.1}},
        "line_grid": [7.5, 8.0, 8.5, 9.0, 9.5],
    }


def _write_monitor(block: dict, date_str: str, tmp: Path,
                   persisted: bool = True, error=None):
    from pipeline import _run_engine_monitor_json
    with patch("pipeline.DATA_DELIVERY_DIR", tmp):
        path = _run_engine_monitor_json(block, date_str, persisted, error)
    return json.loads(path.read_text())


class TestMonitorSchema(unittest.TestCase):
    """The JSON written by _run_engine_monitor_json has the v2 shape."""

    def test_schema_has_all_top_level_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        self.assertEqual(data["schema"], "run-engine-monitor/v2")
        self.assertEqual(data["date"], "20260826")
        self.assertTrue(data["markets_persisted"])
        self.assertIsNone(data["markets_persist_error"])
        self.assertIn("winner_cards", data)
        self.assertIn("rolling", data)
        self.assertIn("fit", data)

    def test_winner_cards_has_exactly_three_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        self.assertEqual(set(data["winner_cards"].keys()),
                         {"over_under", "run_line", "derived_ml"})

    def test_winner_card_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        card = data["winner_cards"]["over_under"]
        for key in ("n", "actual_win_rate", "win_rate", "predicted_mean",
                    "auc", "ece_raw", "ece_calibrated", "brier", "logloss"):
            self.assertIn(key, card, f"missing winner-card field: {key}")
        h = card["holdout"]
        for key in ("n", "actual_win_rate", "win_rate", "ece_raw",
                    "ece_calibrated", "brier", "logloss"):
            self.assertIn(key, h, f"missing holdout field: {key}")

    def test_auc_passthrough_from_reference_line(self):
        """auc rides through from the winner card (already attached from the
        fixed reference line by run_engine.compute_winner_cards)."""
        block = _minimal_block()
        block["winner_cards"]["over_under"]["auc"] = 0.55051
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(block, "20260826", Path(tmp))
        self.assertAlmostEqual(
            data["winner_cards"]["over_under"]["auc"], 0.55051)

    def test_fit_has_required_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        fit = data["fit"]
        for key in ("alpha_home", "alpha_away", "dispersion_chi2_per_df",
                    "fit_tables", "variance_check", "mc_meta"):
            self.assertIn(key, fit)
        self.assertIn("home", fit["dispersion_chi2_per_df"])
        self.assertIn("away", fit["dispersion_chi2_per_df"])
        self.assertIn("home", fit["fit_tables"])
        self.assertIn("away", fit["fit_tables"])

    def test_rolling_starts_with_current_when_no_prior(self):
        """With a non-empty block and no prior monitor files, each card has
        exactly 1 entry (today)."""
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        rolling = data["rolling"]
        self.assertIsInstance(rolling, dict)
        for key in ("over_under", "run_line", "derived_ml"):
            self.assertIn(key, rolling)
            self.assertEqual(len(rolling[key]), 1)
            self.assertEqual(rolling[key][0]["date"], "2026-08-26")

    def test_empty_block_is_graceful(self):
        """A block with no winner cards still writes a valid v2 file."""
        block = _minimal_block()
        block["winner_cards"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(block, "20260826", Path(tmp))
        self.assertEqual(data["schema"], "run-engine-monitor/v2")
        self.assertEqual(data["winner_cards"], {})
        for key in ("over_under", "run_line", "derived_ml"):
            self.assertEqual(data["rolling"][key], [])


class TestPersistFailurePath(unittest.TestCase):
    """markets_persisted=False is reflected in the monitor JSON."""

    def test_persist_failure_flagged(self):
        block = _minimal_block()
        block["winner_cards"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(
                block, "20260826", Path(tmp), persisted=False,
                error="ValueError: markets frame contains NaNs")
        self.assertFalse(data["markets_persisted"])
        self.assertIn("NaNs", data["markets_persist_error"])


class TestRollingHistoryFolding(unittest.TestCase):
    """Prior v2 monitor files are folded into the rolling series."""

    def test_prior_v2_monitors_folded_with_renamed_field(self):
        block = _minimal_block()
        prior = {
            "schema": "run-engine-monitor/v2",
            "date": "20260825",
            "winner_cards": {
                "over_under": {"n": 3900, "actual_win_rate": 0.5390,
                               "win_rate": 0.5390, "predicted_mean": 0.5300,
                               "auc": 0.55, "ece_calibrated": 0.015,
                               "brier": 0.250, "logloss": 0.685},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260825.json").write_text(
                json.dumps(prior))
            data = _write_monitor(block, "20260826", p)
        rolling = data["rolling"]["over_under"]
        self.assertEqual(len(rolling), 2)  # prior + today
        self.assertEqual(rolling[0]["date"], "2026-08-25")
        self.assertEqual(rolling[0]["n"], 3900)
        self.assertEqual(rolling[0]["ece_calibrated"], 0.015)
        self.assertEqual(rolling[1]["date"], "2026-08-26")

    def test_v1_prior_files_mapped_into_rolling(self):
        """A v1 per_line file maps onto the v2 cards (over_8_5 -> over_under,
        home_cover_1_5 -> run_line, derived_moneyline -> derived_ml) so the
        rolling history stays continuous across the cutover; the base_rate
        field is simply not carried (rolling points never carried it)."""
        block = _minimal_block()
        prior = {
            "schema": "run-engine-monitor/v1",
            "date": "20260825",
            "per_line": {
                "over_8_5": {"n": 3900, "ece_calibrated": 0.015,
                             "brier": 0.250, "logloss": 0.685,
                             "predicted_mean": 0.448, "base_rate": 0.449},
                "home_cover_1_5": {"n": 3901, "ece_calibrated": 0.016,
                                   "brier": 0.251, "logloss": 0.686,
                                   "predicted_mean": 0.449,
                                   "base_rate": 0.450},
                "derived_moneyline": {"n": 3902, "ece_calibrated": 0.017,
                                      "brier": 0.252, "logloss": 0.687,
                                      "predicted_mean": 0.450,
                                      "base_rate": 0.451},
                # Unmappable v1 line (not a winner card) -> skipped, no crash.
                "over_7_5": {"n": 3800, "ece_calibrated": 0.02},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260825.json").write_text(
                json.dumps(prior))
            data = _write_monitor(block, "20260826", p)
        for card, n_prior in (("over_under", 3900), ("run_line", 3901),
                              ("derived_ml", 3902)):
            rolling = data["rolling"][card]
            self.assertEqual(len(rolling), 2, f"{card}: prior + today")
            self.assertEqual(rolling[0]["date"], "2026-08-25")
            self.assertEqual(rolling[0]["n"], n_prior)
            self.assertEqual(rolling[1]["date"], "2026-08-26")

    def test_v1_v2_real_fixtures_fold_continuous(self):
        """The REAL v1 monitor fixture + the REAL locally-built v2 monitor
        fold into a continuous per-card series (v1 08-26 -> v2 08-27 ->
        today) — the migration path the dashboard renders through."""
        _here = Path(__file__).resolve().parent
        v1_fixture = _here / "fixtures" / "run_engine_monitor_v1_20260826.json"
        v2_accept = _here.parent / "data_delivery" \
            / "run_engine_monitor_20260827.json"
        if not v1_fixture.exists() or not v2_accept.exists():
            self.skipTest("v1 fixture / v2 acceptance artifact missing")
        block = _minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260826.json").write_text(
                v1_fixture.read_text())
            (p / "run_engine_monitor_20260827.json").write_text(
                v2_accept.read_text())
            data = _write_monitor(block, "20260828", p)
        for card in ("over_under", "run_line", "derived_ml"):
            rolling = data["rolling"][card]
            dates = [r["date"] for r in rolling]
            self.assertEqual(dates, ["2026-08-26", "2026-08-27",
                                     "2026-08-28"], f"{card} continuous")
            self.assertIsInstance(rolling[0]["n"], int)  # v1-mapped point
            self.assertIsInstance(rolling[1]["n"], int)  # v2 point

    def test_rolling_trims_to_45_days(self):
        from datetime import date, timedelta
        block = _minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(50):
                dt = date(2026, 8, 26) - timedelta(days=50 - i)
                d = dt.strftime("%Y%m%d")
                prior = {
                    "schema": "run-engine-monitor/v2",
                    "date": d,
                    "winner_cards": {
                        "over_under": {"n": 3900, "actual_win_rate": 0.5390,
                                       "win_rate": 0.5390,
                                       "predicted_mean": 0.5300, "auc": 0.55,
                                       "ece_calibrated": 0.015,
                                       "brier": 0.250, "logloss": 0.685},
                    },
                }
                (p / f"run_engine_monitor_{d}.json").write_text(
                    json.dumps(prior))
            data = _write_monitor(block, "20260826", p)
        rolling = data["rolling"]["over_under"]
        self.assertLessEqual(len(rolling), 46)  # 45 prior + 1 today max


if __name__ == "__main__":
    unittest.main()
