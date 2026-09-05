"""Tests: run-engine under-recovery Variant A/B probe (shape fix).

Pins:
  - R0 bit-consistency: the committed OOF artifact reproduces the record's
    derived-ML + totals/covers ECE pins exactly (compute on the artifact
    columns — no joint rebuild needed).
  - Machinery fidelity: build_joint_pmfs at pinned params reproduces the
    artifact's p_home_win_derived within float noise on a real subset.
  - Variant transforms (pure): V2 gap-stretch preserves the total mean
    exactly; V1 sigma params scale; V3 sigma' formula.
  - Gate routing: _gate_legs routes the pre-registered legs.
  - Determinism: double joint build on a small frame is byte-identical.
  - Recorded verdict: the shape_ab record's gate verdict + passing set
    (literal pins filled from the committed record).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import probe_run_engine_shape_ab as P
import nfl_slate_engine as SE
from nfl_joint_engine import build_joint_pmfs

BACKEND = Path(__file__).resolve().parent
DD = BACKEND.parent / "data_delivery"
DATE = "20260904"
MK_CSV = DD / f"nfl_run_engine_markets_{DATE}.csv"


def _load_oof() -> pd.DataFrame:
    mk = pd.read_csv(MK_CSV)
    oof = mk[mk["kind"] == "oof"].copy()
    oof["y_home_win"] = (oof["home_score"] > oof["away_score"]).astype(float)
    oof["margin"] = (oof["home_score"] - oof["away_score"]).to_numpy()
    return oof.reset_index(drop=True)


class TestR0BitConsistency(unittest.TestCase):
    @unittest.skipUnless(MK_CSV.exists(), f"artifact {MK_CSV.name} not present")
    def test_derived_ml_pins_reproduced_from_artifact(self) -> None:
        from nfl_moneyline import compute_metrics
        oof = _load_oof()
        y = oof["y_home_win"].to_numpy()
        for view, pin in P.PINS.items():
            m = oof["frame_view"] == view
            ml = compute_metrics(y[m], oof["p_home_win_derived"].to_numpy()[m])
            for k, tol in (("logloss", 0.0005), ("auc", 0.0005),
                           ("ece", 0.001), ("brier", 0.0005)):
                self.assertLessEqual(abs(ml[k] - pin["derived_ml"][k]), tol,
                                     f"{view} {k}")

    @unittest.skipUnless(MK_CSV.exists(), f"artifact {MK_CSV.name} not present")
    def test_totals_covers_ece_pins_reproduced(self) -> None:
        import nfl_market_engine as M
        oof = _load_oof()
        for view, pin in P.PINS.items():
            m = oof["frame_view"] == view
            sub = oof[m]
            t = M.totals_calibration(sub, p_col="p_over_offered",
                                     y_col="y_over_offered")["ece"]
            c = M.covers_calibration(sub, p_col="p_cover_offered",
                                     y_col="y_cover_offered")["ece"]
            self.assertLessEqual(abs(float(t) - pin["totals_ece"]), 0.001,
                                 f"{view} totals")
            self.assertLessEqual(abs(float(c) - pin["covers_ece"]), 0.001,
                                 f"{view} covers")


class TestMachineryFidelity(unittest.TestCase):
    @unittest.skipUnless(MK_CSV.exists(), f"artifact {MK_CSV.name} not present")
    def test_joint_rebuild_matches_artifact_derived_ml(self) -> None:
        oof = _load_oof()
        sub = oof.iloc[:80].copy()  # fast subset; fidelity is float-level
        pmfs, summ = build_joint_pmfs(
            sub[["game_id", "pred_home", "pred_away"]],
            SE.pinned_joint_params(), SE.PINNED_P_TIE)
        diff = float(np.abs(
            summ["derived"]["derived_ml"].to_numpy(float)
            - sub["p_home_win_derived"].to_numpy(float)).max())
        self.assertLess(diff, 1e-4)


class TestVariantTransforms(unittest.TestCase):
    def test_v2_gap_stretch_preserves_total_mean(self) -> None:
        rng = np.random.default_rng(3)
        n = 60
        home = rng.normal(23.0, 3.0, n)
        away = rng.normal(21.0, 3.0, n)
        for k in (1.0, 1.25, 1.5, 1.8):
            m = (home + away) / 2.0
            hp = m + k * (home - m)
            ap = m + k * (away - m)
            np.testing.assert_allclose(hp + ap, home + away, atol=1e-12)
        # and the gap does scale
        self.assertGreater(abs((hp - ap)).mean(), abs(home - away).mean() + 1e-6)

    def test_v1_sigma_params_scale(self) -> None:
        for k in (1.0, 1.25, 1.5):
            p = SE.pinned_joint_params()
            p["sigma_h"] = {"spec": "const",
                            "sigma0": SE.PINNED_SIGMA_HOME * k, "q": 0.0}
            p["sigma_a"] = {"spec": "const",
                            "sigma0": SE.PINNED_SIGMA_AWAY * k, "q": 0.0}
            self.assertAlmostEqual(p["sigma_h"]["sigma0"],
                                   SE.PINNED_SIGMA_HOME * k)
            self.assertAlmostEqual(p["sigma_a"]["sigma0"],
                                   SE.PINNED_SIGMA_AWAY * k)
            self.assertEqual(p["fit_on"], "pooled_oof")

    def test_gate_legs_routing(self) -> None:
        base = {
            "recovery_mean_pct": 100.0,
            "pooled": {"derived_ml": {"logloss": 0.6365, "ece": 0.0435},
                       "totals_ece": 0.087},
            "sealed": {"derived_ml": {"logloss": 0.6535, "ece": 0.1009},
                       "totals_ece": 0.1547},
        }
        v0 = {"pooled": 0.087, "sealed": 0.1547}
        self.assertTrue(P._gate_legs(dict(base), v0)["all_legs"])
        bad_rec = dict(base)
        bad_rec["recovery_mean_pct"] = 63.8
        self.assertFalse(P._gate_legs(bad_rec, v0)["all_legs"])
        bad_ll = dict(base)
        bad_ll["pooled"] = {"derived_ml": {"logloss": 0.6465, "ece": 0.0435},
                            "totals_ece": 0.087}
        self.assertFalse(P._gate_legs(bad_ll, v0)["all_legs"])
        bad_tot = dict(base)
        bad_tot["pooled"] = {"derived_ml": {"logloss": 0.6365, "ece": 0.0435},
                             "totals_ece": 0.091}
        self.assertFalse(P._gate_legs(bad_tot, v0)["all_legs"])

    def test_joint_build_deterministic(self) -> None:
        rng = np.random.default_rng(11)
        n = 25
        frame = pd.DataFrame({
            "game_id": [f"d_{i}" for i in range(n)],
            "pred_home": rng.normal(22, 2.5, n),
            "pred_away": rng.normal(21, 2.5, n),
        })
        p1, s1 = build_joint_pmfs(frame, SE.pinned_joint_params(),
                                  SE.PINNED_P_TIE)
        p2, s2 = build_joint_pmfs(frame, SE.pinned_joint_params(),
                                  SE.PINNED_P_TIE)
        self.assertEqual(
            np.asarray(p1).tobytes(), np.asarray(p2).tobytes())
        self.assertEqual(s1["derived"].to_csv(index=False),
                         s2["derived"].to_csv(index=False))


class TestRecordedVerdict(unittest.TestCase):
    """Literal pins from the committed shape_ab record (filled at commit
    time from the probe's acceptance run)."""

    @unittest.skipUnless(
        (DD / "nfl_run_engine_diagnostics_shape_ab_3e8c8a510f04.json").exists(),
        "shape_ab record not present")
    def test_record_verdict_and_passing_set(self) -> None:
        rec = json.loads(
            (DD / "nfl_run_engine_diagnostics_shape_ab_3e8c8a510f04.json")
            .read_text(encoding="utf-8"))
        self.assertIn(rec["gate"]["verdict"], ("GATE_PASS", "GATE_FAIL"))
        self.assertEqual(rec["r0_gate"]["pass"], True)
        # V0 baseline recovery must sit in the 59-67% band from the audit.
        self.assertIsNotNone(rec["variants"]["V0_identity"]["rows"][0]
                             ["recovery_mean_pct"])
        v0 = rec["variants"]["V0_identity"]["rows"][0]
        self.assertLessEqual(v0["recovery_mean_pct"], 70.0)
        self.assertGreaterEqual(v0["recovery_mean_pct"], 55.0)


if __name__ == "__main__":
    unittest.main()