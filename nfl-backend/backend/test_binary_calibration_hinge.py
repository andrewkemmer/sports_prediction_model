"""Binary moneyline serving-axis arm test — hinge-map pins (spec 2026-09-05).

Follow-on to e3aeece (DO_NOT_REFIT for pchip). Pins the Phase-1 diagnostic
(pchip's +0.0129 pooled regression is fold-variance overfit: it concentrates
IN-BAND raw>0.70 where the per-fold pchip fits explode — std 0.118, outputs
>1.0 at input 0.85) and the Phase-2 arm verdict (KEEP_PLATT: R1 raw fails
pooled-ll +0.0030 / pooled-ECE / sealed 70-80 band 0.0366 >= 0.03 / sealed
80+ adjacent +0.0956; R2 hinge fails the sealed 70-80 band 0.1866 and
worth-having — its pooled gain is a -0.0001 tie and the pooled-fitted slope
a~3.3 over-sharpens sealed rows). Production Platt twin UNCHANGED.

Artifact tests recompute through ``probe_binary_calibration_hinge``
(read-only; double-run byte-identical) over the committed 20260904 CSV
family; skipped when the artifacts are absent (repo real-artifact
convention).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_binary_calibration_hinge as H  # noqa: E402

DD = Path(__file__).resolve().parent.parent / "data_delivery"
HIST_CSV = DD / f"nfl_predictions_history_{H.DATE}.csv"
HAVE_ARTIFACTS = HIST_CSV.exists()

_PROBE: dict | None = None


def _run_once() -> bytes:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        H.main()
    return buf.getvalue().encode("utf-8")


def _probe() -> dict:
    global _PROBE
    if _PROBE is None:
        _PROBE = json.loads(_run_once())
    return _PROBE


# ---------------------------------------------------------------------------
# Phase 1.1/1.2 — pchip regression = fold-variance overfit (the e3aeece
# hypothesis, now with mechanism)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestPerBandAndFoldStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()

    def test_pchip_regression_concentrates_in_band(self):
        """In-band (raw>0.70, n=162) carries the pchip damage: pchip is
        +0.069 logloss over nested platt there (vs only +0.003 out-of-band)
        — the regression is NOT spread evenly, ruling out a smooth global
        discrimination loss and pointing at the top-tail fold variance."""
        pb = self.o["phase1_per_band_logloss"]
        in_b = pb["in_band_raw_gt_0_70"]
        out_b = pb["out_of_band_raw_le_0_70"]
        self.assertEqual(in_b["n"], 162)
        self.assertGreater(in_b["pchip_nested"] - in_b["platt_nested"], 0.05)
        self.assertLess(out_b["pchip_nested"] - out_b["platt_nested"], 0.01)

    def test_fold_stability_pchip_variance_explodes_at_top(self):
        """At the constant input 0.85, per-fold pchip fits spread with std
        0.118 and a mean output ABOVE 1.0 (un-clipped extrapolation) — 2-5x
        platt's spread at every input. Logloss is convex: this per-fold
        prediction variance is the Jensen cost behind pchip's pooled +0.0129
        (the e3aeece overfit hypothesis, now measured directly)."""
        stab = self.o["phase1_fold_stability"]["families"]
        pchip_85 = stab["pchip_nested"]["input_0.85"]
        platt_85 = stab["platt_nested"]["input_0.85"]
        self.assertGreater(pchip_85["std"], 0.05)
        self.assertGreater(pchip_85["std"], 2.0 * platt_85["std"])
        self.assertGreater(pchip_85["mean"], 1.0)      # out of [0,1]
        self.assertLess(platt_85["max"], 1.0)


# ---------------------------------------------------------------------------
# Phase 1.3 — raw ECE direct (no nan)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestRawEceDirect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()["phase1_raw_ece_direct"]

    def test_raw_ece_pooled_and_sealed(self):
        self.assertAlmostEqual(self.o["pooled"]["overall_ece"], 0.0425,
                               delta=0.0005)
        self.assertAlmostEqual(self.o["sealed"]["overall_ece"], 0.0593,
                               delta=0.0005)

    def test_raw_beats_platt_at_the_sealed_70_80_band(self):
        """Identity's sealed 70-80 band ECE (0.0366, n=42) is far below the
        published Platt axis (0.1337, n=39) — the e3aeece Branch-B
        over-stretch is an out-of-sample cost of the global logistic."""
        self.assertEqual(self.o["sealed"]["band_70_80"]["n"], 42)
        self.assertAlmostEqual(self.o["sealed"]["band_70_80"]["band_ece"],
                               0.0366, delta=0.0005)
        self.assertAlmostEqual(
            self.o["sealed"]["reference_platt_sealed"]["band_70_80"]["band_ece"],
            0.1337, delta=0.0005)

    def test_no_nan_fields(self):
        for view in ("pooled", "sealed"):
            self.assertIsNotNone(self.o[view]["overall_ece"])
            self.assertIsNotNone(self.o[view]["band_70_80"]["band_ece"])


# ---------------------------------------------------------------------------
# Phase 1.4 — hinge scan
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestHingeScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()["phase1_hinge_scan"]

    def test_scan_surface_is_flat_and_argmin_is_0_66(self):
        surf = self.o["surface"]
        self.assertEqual(len(surf), 9)
        self.assertEqual(self.o["chosen_h"], 0.66)
        # the hinge barely moves pooled nested logloss anywhere on the scan
        lls = [r["pooled_logloss"] for r in surf]
        self.assertLess(max(lls) - min(lls), 0.005)

    def test_band_ece_best_near_0_69(self):
        surf = self.o["surface"]
        best = min(surf, key=lambda r: r["band_70_80_ece"])
        self.assertIn(best["h"], (0.69, 0.70))


# ---------------------------------------------------------------------------
# Phase 2 — arm table + gate + verdict routing
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestArmVerdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2 = _probe()["phase2"]

    def test_r0_bit_consistency(self):
        """R0 (deployed protocol) reproduces the published sealed axis
        exactly: ll 0.6249 / ECE 0.0745 == the moneyline record + CSV."""
        r0 = self.p2["sealed_deployed"]["R0_platt"]
        self.assertEqual(r0["logloss"], 0.6249)
        self.assertEqual(r0["ece"], 0.0745)

    def test_r1_fails_the_pooled_ll_leg(self):
        """RAW identity regresses pooled nested logloss +0.0030 beyond the
        +/-0.001 bar (platt is genuinely better in-sample: pooled above-0.70
        raw rows realize 0.852 vs raw 0.759 — a real 9pp under-confidence the
        logistic stretch corrects). The +/-0.001 leg is NOT loosened."""
        l1 = self.p2["legs"]["R1_raw"]["pooled_ll_no_regression_beyond_0_001_vs_R0"]
        self.assertGreater(l1["delta"], 0.001)
        self.assertFalse(l1["pass"])

    def test_r1_fails_sealed_band_and_adjacent(self):
        l1 = self.p2["legs"]["R1_raw"]
        self.assertFalse(l1["sealed_70_80_band_lt_0_03"]["pass"])   # 0.0366 >= 0.03
        self.assertFalse(l1["sealed_adjacent_no_regression_gt_0_01"]["pass"])

    def test_r2_fails_sealed_band_and_worth_having(self):
        """The pooled-fitted hinge (slope a~3.3 at h=0.66) over-sharpens the
        sealed 0.66-0.75 rows: sealed 70-80 band ECE 0.1866 (n=16) and a
        +0.0064 sealed-logloss regression; pooled gain is a -0.0001 tie."""
        l2 = self.p2["legs"]["R2_hinge"]
        band = self.p2["sealed_deployed"]["R2_hinge"]["bands"]["70%-80%"]
        self.assertGreaterEqual(band["band_ece"], 0.03)
        self.assertFalse(l2["sealed_70_80_band_lt_0_03"]["pass"])
        self.assertFalse(l2["worth_having"]["pass"])
        self.assertGreater(
            self.p2["sealed_deployed"]["R2_hinge"]["logloss"]
            - self.p2["sealed_deployed"]["R0_platt"]["logloss"], 0.0)

    def test_s2_arbiter_favors_raw_over_platt(self):
        """R1's S2 (arbiter half) improves on R0 materially (ECE 0.1296 ->
        0.0808, ll 0.6387 -> 0.6329) — the record must state this honestly:
        the serving-axis question is a real pooled-vs-sealed/S2 trade that
        the pooled legs resolve against changing."""
        s2 = self.p2["sealed_s1_s2"]["S2_second"]
        self.assertLess(s2["R1"]["ece"], s2["R0"]["ece"] - 0.02)
        self.assertLess(s2["R1"]["ll"], s2["R0"]["ll"])
        self.assertTrue(self.p2["legs"]["R1_raw"]["sealed_s2_arbiter"]["pass"])

    def test_auc_flat_contract(self):
        """Monotone arms leave pooled AUC within +/-0.001 of the raw axis
        (small cross-fold-map rank effects, not an AUC gain/loss claim)."""
        aucs = self.p2["pooled_auc"]
        for arm in ("R0", "R1", "R2"):
            self.assertLessEqual(abs(aucs[arm] - aucs["R1"]), 0.001)

    def test_verdict_keep_platt(self):
        self.assertFalse(self.p2["R1_passes_all"])
        self.assertFalse(self.p2["R2_passes_all"])
        self.assertEqual(self.p2["verdict"], "KEEP_PLATT")


# ---------------------------------------------------------------------------
# Pure map-property pins (synthetic — no artifacts needed)
# ---------------------------------------------------------------------------
class TestHingeMapProperties(unittest.TestCase):
    def test_hinge_is_monotone_and_continuous_at_h(self):
        rng = np.random.default_rng(11)
        n = 300
        p = np.sort(rng.uniform(0.30, 0.92, n))
        y = rng.binomial(1, np.clip(p * 1.05, 0.05, 0.95)).astype(float)
        for h in (0.66, 0.70, 0.74):
            hi = p > h
            a = H._fit_hinge(p[hi], y[hi], h)
            mapped = H._hinge_apply(p, a, h)
            self.assertTrue(np.all(np.diff(mapped) >= -1e-12), f"h={h} monotone")
            # continuity at the hinge
            eps = 1e-6
            self.assertAlmostEqual(
                H._hinge_apply(np.array([h + eps]), a, h)[0],
                H._hinge_apply(np.array([h]), a, h)[0], places=5)
            self.assertAlmostEqual(H._hinge_apply(np.array([h]), a, h)[0], h)
            # outputs stay in [0, 1]
            self.assertGreaterEqual(mapped.min(), 0.0)
            self.assertLessEqual(mapped.max(), 1.0)

    def test_nested_fit_on_strictly_earlier_weeks_only(self):
        """Protocol purity spy: a fold's hinge slope is fit on strictly-
        earlier folds' above-h rows only — never its own or any future row."""
        rng = np.random.default_rng(7)
        n = 400
        week = np.repeat(np.arange(8), n // 8)
        raw = np.clip(0.45 + 0.25 * np.sin(week * 0.7)
                      + rng.normal(0, 0.08, n), 0.05, 0.95)
        y = rng.binomial(1, np.clip(raw * 1.1 - 0.02, 0.05, 0.95))
        fitted_pairs = []
        h = 0.68
        for w in range(8):
            prior = week < w
            if prior.sum() >= 10:
                hi = raw[prior] > h
                if hi.sum() >= 20 and len(np.unique(y[prior][hi])) >= 2:
                    H._fit_hinge(raw[prior][hi], y[prior][hi], h)
                    fitted_pairs.append((w, int(week[prior].max())))
        self.assertGreaterEqual(len(fitted_pairs), 3)
        for fold_w, seen in fitted_pairs:
            self.assertEqual(seen, fold_w - 1)   # strictly-earlier folds only
            self.assertLess(seen, fold_w)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestDeterminism(unittest.TestCase):
    def test_double_run_byte_identical(self):
        self.assertEqual(_run_once(), _run_once())


if __name__ == "__main__":
    unittest.main()
