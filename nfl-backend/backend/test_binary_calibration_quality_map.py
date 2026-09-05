"""Binary moneyline quality-stratum LOCAL recalibration test (spec 2026-09-05).

Follow-on to e3aeece (DO_NOT_REFIT) + fc4f4bd (KEEP_PLATT): a correction
applied ONLY in the stratum (raw > h) where the Platt 70-80 band
over-stretch lives, leaving the global Platt map untouched outside. Pins:
  - R0 bit-consistency: deployed global Platt reproduces the published map
    (a/b to 3e-5) and the sealed ll 0.6249 / ECE 0.0745 exactly.
  - Stratum characterization: raw>0.68 n=257 (18.7%), ~half quality-extreme
    (top-quartile |elo_diff|), elevated |binary - derived| — the over-stretch
    concentrates on the audit's quality-extreme/high-confidence games.
  - Selection rule: identity-in-stratum is EXCLUDED (the seam vs Platt is
    discontinuous -> breaks the global-monotonicity / AUC-flat contract by
    construction); among the continuous families (L anchored logistic /
    P anchored pchip) the rule picks min nested pooled ll among 70-80 band
    ECE <= 0.03 -> h*=0.72, P_local_pchip.
  - Gate legs (deployed protocol): GATE_FAIL — S2 70-80 band 0.1685 not
    < 0.03; nested pooled ll +0.0092 > +/-0.001; sealed 80+ adjacent
    +0.0658 (n=48); worth-having fails (band gain 0.0086 < noise/3). AUC-flat
    holds EXACTLY (0.0) and no-bleed passes — the continuous families are
    contract-clean; they just cannot fix the band without paying elsewhere.
  - Verdict routing + determinism (double run byte-identical).

Artifact tests recompute through ``probe_binary_calibration_quality_map``
(read-only; deterministic) over the committed 20260904 CSV family; skipped
when the artifacts are absent (repo real-artifact convention).
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
import probe_binary_calibration_quality_map as Q  # noqa: E402

DD = Path(__file__).resolve().parent.parent / "data_delivery"
HIST_CSV = DD / f"nfl_predictions_history_{Q.DATE}.csv"
HAVE_ARTIFACTS = HIST_CSV.exists()

_PROBE: dict | None = None


def _run_once() -> bytes:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        Q.main()
    return buf.getvalue().encode("utf-8")


def _probe() -> dict:
    global _PROBE
    if _PROBE is None:
        _PROBE = json.loads(_run_once())
    return _PROBE


# ---------------------------------------------------------------------------
# R0 + stratum characterization
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestR0AndStratum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()

    def test_r0_bit_consistency(self):
        """Deployed global Platt reproduces the published map (a/b to 3e-5)
        and the moneyline record's sealed metrics exactly."""
        r0 = self.o["r0_gate"]
        self.assertTrue(r0["pass"])
        self.assertLess(r0["a_err"], 3e-5)
        self.assertLess(r0["b_err"], 3e-5)
        self.assertEqual(r0["sealed_ll"], 0.6249)
        self.assertEqual(r0["sealed_ece"], 0.0745)

    def test_universe_is_the_shared_1376(self):
        u = self.o["universe"]
        self.assertEqual(u["oof_rows"], 1376)
        self.assertEqual(u["joined_with_binary"], 1376)

    def test_stratum_is_quality_extreme_overlap(self):
        """raw>0.68 n=257 with an elevated mean |binary - derived| gap vs the
        audit's 0.1041 overall — the over-stretch concentrates on the
        high-confidence games. The quality-extreme overlap (top-quartile
        |elo_diff|) is pinned when the feature cache resolves; the CSV-only
        pins always hold."""
        sc = self.o["stratum_char"]
        self.assertEqual(sc["raw_gt_0.68"]["n"], 257)
        self.assertGreater(sc["raw_gt_0.68"]["mean_abs_binary_derived_gap"], 0.11)
        if sc["quality_extreme_n"] is not None:
            self.assertGreaterEqual(sc["raw_gt_0.68"]["share_quality_extreme"], 0.45)
            self.assertEqual(sc["quality_extreme_n"], 344)


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()

    def test_selection_is_continuous_family_not_identity(self):
        """The served axis must be globally monotone: identity-in-stratum is
        excluded from selection (its seam vs Platt is discontinuous). The
        rule lands on P_local_pchip at h=0.72."""
        sel = self.o["selection"]
        self.assertEqual(sel["h_star"], 0.72)
        self.assertEqual(sel["family_star"], "P_local_pchip")
        self.assertNotEqual(sel["family_star"], "I_identity")
        self.assertIn("discontinuous seam", sel["rule"])

    def test_identity_reference_row_present_in_surface(self):
        surf = self.o["nested_scan"]["surface"]
        self.assertIn("h0.72_I_identity", surf)


# ---------------------------------------------------------------------------
# Gate legs + verdict
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestGateAndVerdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe()
        cls.g = _probe()["gate_legs"]

    def test_leg1_s2_band_fails(self):
        """S2 (arbiter) 70-80 band ECE 0.1685 > 0.03: the anchored pchip
        vacates the band (in-stratum served >= Platt(0.72) ~ 0.795), leaving
        the over-stretched Platt sliver (raw 0.64-0.72) untouched on sealed."""
        l = self.g["leg1_sealed_s2_70_80_band_ece_le_0_03"]
        self.assertFalse(l["pass"])
        self.assertGreater(l["served"], 0.03)

    def test_leg2_pooled_ll_fails(self):
        """Nested pooled cal-logloss +0.0092 > +/-0.001 (9x the bar). The
        deployed single fit actually improves in-sample (-0.0021) — the
        per-fold pchip fits carry fold variance (Jensen, cf. e3aeece) that
        shows up in the record's nested protocol. Bar not loosened."""
        l = self.g["leg2_pooled_cal_logloss_within_0_001"]
        self.assertFalse(l["pass"])
        self.assertGreater(l["delta"], 0.001)
        self.assertLess(l["deployed_in_sample_delta"], 0.0)

    def test_leg3_sealed_adjacent_fails(self):
        """The pooled-fitted pchip's steep in-stratum tail over-predicts
        sealed's mild top: sealed 80+ band ECE +0.0658 (n=48) > +0.01."""
        l = self.g["leg3_sealed_adjacent_bands_no_regression"]
        self.assertFalse(l["pass"])
        self.assertGreater(l["80_plus"]["delta_served_minus_platt"], 0.01)

    def test_leg4_auc_flat_contract_holds_exactly(self):
        """Continuous monotone local maps are rank-invariant: served AUC ==
        raw AUC exactly (0.0 delta) — the identity seam was the only thing
        that could move AUC, and it is excluded from serving."""
        l = self.g["leg4_auc_flat_within_0_001"]
        self.assertTrue(l["pass"])
        self.assertEqual(l["max_abs_delta"], 0.0)

    def test_leg6_no_bleed_passes(self):
        """The chosen map stays in [0,1], is monotone on the local segment,
        and is seam-continuous with the global Platt (jump 0.0) — the
        fc4f4bd >1.0 mode is absent and the identity seam is avoided."""
        l = self.g["leg6_no_bleed_monotone"]
        self.assertTrue(l["pass"])
        self.assertLessEqual(l["served_max"], 1.0)
        self.assertGreaterEqual(l["served_min"], 0.0)
        self.assertAlmostEqual(l["seam_jump_served_minus_platt_at_h"], 0.0,
                               places=6)

    def test_verdict_gate_fail(self):
        self.assertEqual(self.o["verdict"]["result"], "GATE_FAIL")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 history CSV not present")
class TestDeterminism(unittest.TestCase):
    def test_double_run_byte_identical(self):
        self.assertEqual(_run_once(), _run_once())


# ---------------------------------------------------------------------------
# Pure map-property pins (synthetic — no artifacts needed)
# ---------------------------------------------------------------------------
class TestLocalMapProperties(unittest.TestCase):
    def test_anchored_logistic_continuous_at_h_and_monotone(self):
        """The anchored local logistic passes through (h, anchor) and is
        monotone non-decreasing on [h, 1] — the served axis is continuous
        with the global Platt at the seam when anchor = Platt(h)."""
        rng = np.random.default_rng(4)
        n = 400
        p = np.sort(rng.uniform(0.40, 0.95, n))
        y = rng.binomial(1, np.clip(p, 0.05, 0.95)).astype(float)
        for h, anchor in ((0.68, 0.74), (0.72, 0.795), (0.74, 0.806)):
            hi = p > h
            a = Q._fit_local_logistic(p[hi], y[hi], h, anchor)
            mapped = Q._apply_local_logistic(a, p, h, anchor)
            self.assertTrue(np.all(np.diff(mapped) >= -1e-12), f"h={h}")
            self.assertAlmostEqual(
                Q._apply_local_logistic(a, np.array([h]), h, anchor)[0],
                anchor, places=5)
            self.assertGreaterEqual(mapped.min(), 0.0)
            self.assertLessEqual(mapped.max(), 1.0)

    def test_anchored_pchip_continuous_at_h_and_clipped(self):
        rng = np.random.default_rng(9)
        n = 400
        p = np.sort(rng.uniform(0.40, 0.95, n))
        y = rng.binomial(1, np.clip(p * 1.02, 0.05, 0.95)).astype(float)
        for h, anchor in ((0.68, 0.74), (0.72, 0.795)):
            hi = p > h
            m = Q._fit_local_pchip(p[hi], y[hi], h, anchor)
            mapped = Q._apply_local_pchip(m, p)
            # outputs hard-clipped to [0,1]
            self.assertGreaterEqual(mapped.min(), 0.0)
            self.assertLessEqual(mapped.max(), 1.0)
            if m is not None:
                # seam knot (h, anchor) honored -> continuity with Platt
                self.assertAlmostEqual(
                    Q._apply_local_pchip(m, np.array([h]))[0], anchor,
                    places=4)

    def test_identity_seam_is_discontinuous(self):
        """Why I_identity is the reference, not the serving map: served =
        where(raw>h, raw, Platt(raw)) drops at the seam (Platt(h) ~ 0.79 ->
        raw just above h ~ 0.72), breaking global monotonicity and the
        AUC-flat contract."""
        p = np.linspace(0.60, 0.85, 501)
        # emulate the deployed Platt map (published a/b)
        a_, b_ = Q.PIN_A, Q.PIN_B
        platt = Q._logistic(a_ * Q._logit(p) + b_)
        h = 0.72
        served = np.where(p > h, p, platt)
        # find the largest downward step across the sorted axis
        drops = np.diff(served)
        self.assertLess(drops.min(), -0.05)      # the seam discontinuity
        self.assertGreater(platt[p == h][0], h)  # Platt(h) > h (the stretch)


if __name__ == "__main__":
    unittest.main()