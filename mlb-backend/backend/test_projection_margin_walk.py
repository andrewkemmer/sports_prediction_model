"""Unit tests for the run-engine projection margin-walk gate
(run_projection_margin_walk.py).

Pins:
  * Gate legs compute correctly from price surfaces (sealed CRPS noise
    band, pooled tolerance, totals/covers ECE, P(win) SD non-shrink,
    away-fav SP-band non-degradation, determinism).
  * Verdicts: ADOPT (all legs + worth-having), RE_TEST_CANDIDATE (one
    secondary leg over tolerance by <= half its tolerance), DON'T_ADOPT
    (leg 1 fails or a leg blows past tolerance).
  * Determinism check flags byte-level λ drift.
  * Arm spec: C0 = production view unchanged; P1/P2 route the projection
    ONLY through sp_projection.py columns on the decided frame (no inline
    re-derivation); served FEATURE_COLS untouched.

Run:  python -m unittest test_projection_margin_walk -v
      (from mlb-backend/backend)
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if "resource" not in sys.modules:  # POSIX-only module, stub on Windows
    _res = types.ModuleType("resource")
    _ru = types.SimpleNamespace(ru_maxrss=0)
    _res.getrusage = lambda *_: _ru
    _res.RUSAGE_SELF = 0
    sys.modules["resource"] = _res

from run_projection_margin_walk import (  # noqa: E402
    CRPS_NOISE,
    decide,
    determinism_check,
    gate_legs,
    verify_producer,
)
from sp_projection import PROJ_HI_BETTER, PROJ_LO_BETTER  # noqa: E402

CSV = _BACKEND_DIR.parent / "data_delivery" / "game_level_features.csv"


def _surf(sealed_crps, pooled_crps, totals_ece_p, totals_ece_s,
          covers_ece_p, covers_ece_s, pwin_sd, away_gap):
    return {
        "margin_crps_sealed": sealed_crps,
        "margin_crps_pooled": pooled_crps,
        "totals": {"metrics_pooled": {"ece": totals_ece_p},
                   "metrics_sealed": {"ece": totals_ece_s}},
        "run_line_minus_1_5": {"metrics_pooled": {"ece": covers_ece_p},
                               "metrics_sealed": {"ece": covers_ece_s}},
        "derived_ml": {"pwin_sd_sealed": pwin_sd},
        "pooled_sp_band_gaps": {"away_fav": {"gap": away_gap, "n": 1800}},
    }


# b7eed32 C0 pins as the baseline fixture.
C0 = _surf(2.53856, 2.46847, 0.02474, 0.02859, 0.00467, 0.02977, 0.0409,
           0.0325)


class TestGateLegs(unittest.TestCase):
    def test_p1_style_candidate_passes_all_legs(self):
        # b7eed32 P1 numbers — sealed CRPS -0.021, totals ECE within tol,
        # covers ECE within tol, pwin SD grows, away-fav gap improves.
        p1 = _surf(2.51736, 2.46701, 0.02639, 0.02965, 0.00585, 0.03168,
                   0.0546, 0.0305)
        legs = gate_legs(p1, C0, det=None)
        self.assertTrue(all(v["ok"] is True for v in legs.values()), legs)
        self.assertLessEqual(legs["1_sealed_crps_improves"]["delta"],
                             -CRPS_NOISE)

    def test_leg1_fails_when_sealed_gain_inside_noise(self):
        bad = _surf(C0["margin_crps_sealed"] - 0.0015, 2.46800, 0.02500,
                    0.02900, 0.00500, 0.03000, 0.0450, 0.0330)
        legs = gate_legs(bad, C0, det=None)
        self.assertIs(legs["1_sealed_crps_improves"]["ok"], False)
        self.assertIs(legs["2_pooled_crps_not_regress"]["ok"], True)

    def test_pooled_crps_regression_fails_leg2(self):
        bad = _surf(2.52000, C0["margin_crps_pooled"] + 0.004, 0.02500,
                    0.02900, 0.00500, 0.03000, 0.0450, 0.0330)
        legs = gate_legs(bad, C0, det=None)
        self.assertIs(legs["2_pooled_crps_not_regress"]["ok"], False)

    def test_totals_ece_regression_fails_leg3(self):
        bad = _surf(2.52000, 2.46800, 0.02474 + 0.003, 0.02859, 0.00500,
                    0.03000, 0.0450, 0.0330)
        legs = gate_legs(bad, C0, det=None)
        self.assertIs(legs["3_totals_ece"]["ok"], False)
        self.assertIs(legs["4_covers_ece"]["ok"], True)

    def test_pwin_sd_shrink_fails_leg5(self):
        bad = _surf(2.52000, 2.46800, 0.02500, 0.02900, 0.00500, 0.03000,
                    0.0395, 0.0330)  # SD 0.0395 < 0.0409 - 0.001
        legs = gate_legs(bad, C0, det=None)
        self.assertIs(legs["5_pwin_sd_not_shrink"]["ok"], False)

    def test_away_fav_band_regression_fails_leg6(self):
        bad = _surf(2.52000, 2.46800, 0.02500, 0.02900, 0.00500, 0.03000,
                    0.0450, 0.0325 + 0.015)
        legs = gate_legs(bad, C0, det=None)
        self.assertIs(legs["6_sp_band_strata"]["ok"], False)


class TestVerdicts(unittest.TestCase):
    def test_adopt(self):
        p1 = _surf(2.51736, 2.46701, 0.02639, 0.02965, 0.00585, 0.03168,
                   0.0546, 0.0305)
        legs = gate_legs(p1, C0, det=None)
        v = decide(legs)
        self.assertEqual(v["verdict"], "ADOPT")
        self.assertEqual(v["legs_failed"], [])

    def test_dont_adopt_when_leg1_fails(self):
        bad = _surf(C0["margin_crps_sealed"] - 0.0015, 2.46800, 0.02500,
                    0.02900, 0.00500, 0.03000, 0.0450, 0.0330)
        v = decide(gate_legs(bad, C0, det=None))
        self.assertEqual(v["verdict"], "DON'T_ADOPT")
        self.assertIn("1_sealed_crps_improves", v["legs_failed"])

    def test_retest_on_borderline_secondary_leg(self):
        # Leg 1 passes; totals ECE pooled regresses +0.003 (excess 0.001 =
        # ECE_TOL/2) — a genuine near-miss → RE_TEST_CANDIDATE.
        near = _surf(2.51736, 2.46701, 0.02474 + 0.003, 0.02859, 0.00500,
                     0.03000, 0.0450, 0.0330)
        legs = gate_legs(near, C0, det=None)
        v = decide(legs)
        self.assertEqual(v["verdict"], "RE_TEST_CANDIDATE")

    def test_dont_adopt_on_clear_secondary_leg_fail(self):
        # Leg 1 passes; covers ECE pooled regresses +0.01 (excess >> tol/2).
        bad = _surf(2.51736, 2.46701, 0.02500, 0.02900, 0.00467 + 0.01,
                    0.03000, 0.0450, 0.0330)
        v = decide(gate_legs(bad, C0, det=None))
        self.assertEqual(v["verdict"], "DON'T_ADOPT")

    def test_determinism_leg(self):
        ok = {"identical_walk": True, "crps_sealed_equal": True,
              "max_lambda_abs_diff": 0.0, "rows_a": 5, "rows_b": 5}
        broken = dict(ok, max_lambda_abs_diff=1e-5, identical_walk=False)
        p1 = _surf(2.51736, 2.46701, 0.02639, 0.02965, 0.00585, 0.03168,
                   0.0546, 0.0305)
        self.assertIs(
            gate_legs(p1, C0, det=ok)["7_determinism"]["ok"], True)
        self.assertIs(
            gate_legs(p1, C0, det=broken)["7_determinism"]["ok"], False)


class TestProducerVerification(unittest.TestCase):
    def test_pins_pass_on_record_fit(self):
        meta = {"home": {"era_on_proj_slope": -1.2213,
                         "coverage_pre": 0.9736,
                         "coverage_sealed": 0.9866},
                "away": {"era_on_proj_slope": -1.2138,
                         "coverage_pre": 0.9742,
                         "coverage_sealed": 0.9933}}
        v = verify_producer(meta, "7bec561aa0391920")
        self.assertTrue(v["pins"]["_all_ok"])

    def test_pins_fail_on_drifted_fit(self):
        meta = {"home": {"era_on_proj_slope": -0.9,
                         "coverage_pre": 0.9736,
                         "coverage_sealed": 0.9866},
                "away": {"era_on_proj_slope": -1.2138,
                         "coverage_pre": 0.9742,
                         "coverage_sealed": 0.9933}}
        v = verify_producer(meta, "7bec561aa0391920")
        self.assertFalse(v["pins"]["_all_ok"])
        self.assertFalse(v["pins"]["era_on_proj_slope_home"]["ok"])


class TestDeterminismCheck(unittest.TestCase):
    def test_byte_identical(self):
        oof_a = pd.DataFrame({"home_expected_runs": [4.5, 4.2],
                              "away_expected_runs": [4.1, 3.9]})
        oof_b = oof_a.copy()
        res = {"margin_crps_sealed": 2.5, "margin_crps_pooled": 2.4}
        d = determinism_check(oof_a, oof_b, res, dict(res))
        self.assertTrue(d["identical_walk"])
        self.assertEqual(d["max_lambda_abs_diff"], 0.0)

    def test_drift_detected(self):
        oof_a = pd.DataFrame({"home_expected_runs": [4.5, 4.2],
                              "away_expected_runs": [4.1, 3.9]})
        oof_b = pd.DataFrame({"home_expected_runs": [4.5, 4.2 + 1e-5],
                              "away_expected_runs": [4.1, 3.9]})
        d = determinism_check(oof_a, oof_b, {"margin_crps_sealed": 2.5},
                              {"margin_crps_sealed": 2.5})
        self.assertFalse(d["identical_walk"])
        self.assertGreater(d["max_lambda_abs_diff"], 0.0)


class TestArmSpec(unittest.TestCase):
    @unittest.skipUnless(CSV.exists(), "committed frame not present")
    def test_projection_routed_through_sp_projection_columns(self):
        from data_ingestion import load_game_features
        from frames import get_decided_frame
        import run_projection_margin_walk as rw
        import run_sp_sensitivity as sps

        games = load_game_features(CSV)
        decided, proj_meta, _gl = rw.attach_on_decided(games)
        self.assertIn("sp_proj_era_home", decided.columns)
        self.assertIn("sp_proj_era_away", decided.columns)
        # Producer pins hold on the current frame.
        self.assertAlmostEqual(proj_meta["home"]["era_on_proj_slope"],
                               -1.2213, delta=0.0006)
        # P1/P2 extras reference ONLY the producer's columns, present in the
        # decided frame; C0 keeps the production view (no per-side extras).
        _, c0_per_side = sps.arm_params_and_frames("C0", decided)
        self.assertIsNone(c0_per_side)
        for arm in ("P1", "P2"):
            _params, per_side = sps.arm_params_and_frames(arm, decided)
            for side in ("home", "away"):
                opp = "away" if side == "home" else "home"
                self.assertIn(f"sp_proj_era_{opp}", per_side[side])
                for c in per_side[side]:
                    if c.startswith("sp_proj"):
                        self.assertIn(c, decided.columns)

    def test_producer_component_palette_unchanged(self):
        # Locked verbatim to b7eed32 (the runner consumes the producer, which
        # owns these constants — a palette drift would break comparability).
        self.assertEqual(list(PROJ_LO_BETTER),
                         ["sp_fip", "sp_xwoba", "sp_whip", "sp_bb9"])
        self.assertEqual(list(PROJ_HI_BETTER),
                         ["sp_k9_5g", "sp_whiff_3g", "sp_fbvelo_3g"])

    def test_served_feature_cols_untouched(self):
        import training
        before = list(training.FEATURE_COLS)
        import run_projection_margin_walk as rw  # noqa: F401 (import only)
        self.assertEqual(list(training.FEATURE_COLS), before)


if __name__ == "__main__":
    unittest.main()
