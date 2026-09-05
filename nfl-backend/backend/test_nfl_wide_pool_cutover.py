"""Tests: wide-pool serving cutover decoupling + legacy export removal at this commit.

Purpose (per task): (1) assert the localized legacy constants defined in the
decoupling pin (nfl_run_engine_legacy_windows) equal the recorded pre-cutover
nfl_moneyline legacy values; (2) assert nfl_moneyline does NOT export
TRAIN_SEASONS / VAL_SEASONS / SEALED_SEASON (nor the legacy
generate_weekly_folds) at this commit, so downstream legacy behavior only
comes from the localized pin; (3) assert no legacy run-engine consumer
imports the three names or generate_weekly_folds from nfl_moneyline anymore;
(4) verify the dead-code removal (no wandered platt_fit copy in _valid_rows)
and the wide-pool constants (WARMUP_SEASONS=[2018], CORE_SEASONS=2019-2025,
POSTSEASON_GAME_TYPES, PLATT_SEED_FLOOR=300).

These are targeted pin/scope tests for this cutover; they do NOT exercise the
full pipeline emission (that is the separate determinism task). They run at
this commit and must stay green when the cutover record ships.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import nfl_moneyline as nm  # noqa: E402
import nfl_run_engine_legacy_windows as pin  # noqa: E402

# Legacy consumers that must now read the three names + the legacy fold
# generator from the localized pin, NOT from nfl_moneyline.
# run_nfl_wide_pool_baseline.py is deliberately EXCLUDED: it is the validated
# wide-pool reference harness and already consumes the wide-pool constants
# from nfl_features (pre-cutover scope).
LEGACY_CONSUMERS = [
    "run_nfl_era.py",
    "run_nfl_joint.py",
    "run_nfl_joint_rebaseline.py",
    "run_nfl_margin_ablation.py",
    "run_nfl_market.py",
    "run_nfl_markets_backfill.py",
    "run_nfl_mean_bias_calibration.py",
    "run_nfl_per_side.py",
    "run_nfl_raw_ablation.py",
    "run_nfl_sigma.py",
    "run_nfl_window_ablation.py",
    "run_nfl_window_gate.py",
    "run_nfl_xgb_reg_ablation.py",
]


class TestLocalizedLegacyPin(unittest.TestCase):
    EXPECTED_LEGACY_TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
    EXPECTED_LEGACY_VAL_SEASONS = [2021, 2022, 2023, 2024]
    EXPECTED_SEALED_SEASON = 2025

    def test_pin_train_seasons_equal_pre_cutover_legacy(self):
        self.assertEqual(pin.TRAIN_SEASONS, self.EXPECTED_LEGACY_TRAIN_SEASONS)

    def test_pin_val_seasons_equal_pre_cutover_legacy(self):
        self.assertEqual(pin.VAL_SEASONS, self.EXPECTED_LEGACY_VAL_SEASONS)

    def test_pin_sealed_season_equal_pre_cutover_legacy(self):
        self.assertEqual(pin.SEALED_SEASON, self.EXPECTED_SEALED_SEASON)

    def test_pin_keeps_legacy_fold_generator(self):
        self.assertTrue(callable(pin.generate_weekly_folds))

    def test_pin_docstring_states_legacy_scope(self):
        doc = pin.__doc__
        self.assertIsNotNone(doc)
        self.assertIn("legacy", doc.lower())
        self.assertIn("pre-cutover", doc.lower())
        self.assertIn("d1 alignment", doc.lower())


class TestNflMoneylineNoLegacyExportAtThisCommit(unittest.TestCase):
    LEGACY_NAMES = ("TRAIN_SEASONS", "VAL_SEASONS", "SEALED_SEASON",
                    "generate_weekly_folds")

    def test_nfl_moneyline_does_not_export_legacy_gate_tokens(self):
        for name in self.LEGACY_NAMES:
            self.assertFalse(
                hasattr(nm, name),
                msg=f"nfl_moneyline still exports {name} at this commit; "
                "legacy gate tokens must be removed from the serving module.",
            )


class TestNoLegacyImportsFromNflMoneyline(unittest.TestCase):
    LEGACY_TOKENS = ("TRAIN_SEASONS", "VAL_SEASONS", "SEALED_SEASON",
                     "generate_weekly_folds")

    def test_legacy_consumers_import_the_names_only_from_the_pin(self):
        for fn in LEGACY_CONSUMERS:
            src = (BACKEND / fn).read_text(encoding="utf-8")
            for tok in self.LEGACY_TOKENS:
                self.assertNotIn(
                    f"from nfl_moneyline import (SEALED_SEASON, TRAIN_SEASONS",
                    src,
                    msg=f"{fn} still imports legacy tokens from nfl_moneyline "
                    f"({tok}); re-point to nfl_run_engine_legacy_windows.",
                )
        # Spot-check the split imports landed in every legacy consumer.
        for fn in LEGACY_CONSUMERS:
            src = (BACKEND / fn).read_text(encoding="utf-8")
            self.assertIn("from nfl_run_engine_legacy_windows import", src,
                          msg=f"{fn} does not import from the localized pin.")

    def test_no_legacy_consumer_imports_the_wide_pool_constants(self):
        wide_tokens = ("CORE_SEASONS", "WARMUP_SEASONS", "generate_week_id_folds")
        for fn in LEGACY_CONSUMERS:
            src = (BACKEND / fn).read_text(encoding="utf-8")
            for tok in wide_tokens:
                self.assertNotIn(tok, src,
                                 msg=f"{fn} references wide-pool token {tok}; "
                                 "legacy consumers must stay on the old "
                                 "methodology this cycle.")


class TestDeadCodeRemoved(unittest.TestCase):
    def test_valid_rows_has_no_wandered_platt_copy(self):
        import inspect
        body = inspect.getsource(nm._valid_rows)
        self.assertNotIn("platt_fit", body,
                         msg="wandered platt_fit copy after `return` in "
                         "_valid_rows must be removed.")
        self.assertNotIn("platt_predict", body)


class TestWidePoolConstantsPresentWhereExpected(unittest.TestCase):
    EXPECTED_WARMUP = [2018]
    EXPECTED_CORE = list(range(2019, 2026))

    def test_warmup_seasons_present(self):
        self.assertEqual(nm.WARMUP_SEASONS, self.EXPECTED_WARMUP)

    def test_core_seasons_present(self):
        self.assertEqual(nm.CORE_SEASONS, self.EXPECTED_CORE)

    def test_postseason_game_types_excluded(self):
        self.assertIn("WC", nm.POSTSEASON_GAME_TYPES)
        self.assertIn("DIV", nm.POSTSEASON_GAME_TYPES)
        self.assertIn("CON", nm.POSTSEASON_GAME_TYPES)
        self.assertIn("SB", nm.POSTSEASON_GAME_TYPES)

    def test_platt_seed_floor_present_and_300(self):
        self.assertTrue(hasattr(nm, "PLATT_SEED_FLOOR"))
        self.assertEqual(nm.PLATT_SEED_FLOOR, 300)


if __name__ == "__main__":
    unittest.main()