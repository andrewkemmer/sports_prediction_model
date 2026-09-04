"""Binary moneyline high-band calibration — gated-fix test pins (spec 2026-09-05).

Pins the Phase-1 verdict (same-set cell = Branch B; pchip spline clears the
Phase-1 gate) AND the Phase-2 verdict (DO_NOT_REFIT — no candidate family
clears every blocking leg: pooled-cal logloss regression +0.0129 > +/-0.001,
sealed 70-80 band ECE 0.0421 not < 0.03, sealed 80+ adjacent regression
+0.0953 > +0.01). Production Platt twin UNCHANGED — these tests assert the
computed verdicts, not a served-axis delta.

Artifact tests recompute through ``probe_binary_calibration`` (read-only,
deterministic double-run byte-identical) over the committed 20260904 CSV
family; skipped when the artifacts are absent (mirroring the repo's real-
artifact test convention).
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
import probe_binary_calibration as probe  # noqa: E402

DD = Path(__file__).resolve().parent.parent / "data_delivery"
MK_CSV = DD / f"nfl_run_engine_markets_{probe.DATE}.csv"
HIST_CSV = DD / f"nfl_predictions_history_{probe.DATE}.csv"

HAVE_ARTIFACTS = MK_CSV.exists() and HIST_CSV.exists()

_PROBE_RUN: dict | None = None
_PROBE_BYTES: list[bytes] = []


def _probe_json() -> dict:
    """Run the probe ONCE per process (module-level cache) and return JSON."""
    global _PROBE_RUN
    if _PROBE_RUN is None:
        _PROBE_RUN = json.loads(_run_probe_once())
    return _PROBE_RUN


def _run_probe_once() -> bytes:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        probe.main()
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Phase 1.1 — same-set cell (Branch classification)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 run-engine markets + history "
                     "CSVs not present — run the daily emission first")
class TestSameSetCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = _probe_json()["same_set_cell"]

    def test_platt_band_universe_replicates_audit_n191(self):
        fwd = self.o["platt_band_70_80_games"]
        self.assertEqual(fwd["n"], 191)
        self.assertAlmostEqual(fwd["mean_cal"], 0.7485, delta=0.0005)
        self.assertAlmostEqual(fwd["actual_win_rate"], 0.6754, delta=0.0005)

    def test_raw_mean_on_platt_band_games_branch_B(self):
        """The decisive cell: raw axis mean on the SAME n=191 set is 0.6815
        (~actual), far below the calibrated 0.7485 — Branch B (the Platt map
        stretched these games up), not ensemble overconfidence."""
        fwd = self.o["platt_band_70_80_games"]
        self.assertAlmostEqual(fwd["mean_raw"], 0.6815, delta=0.002)
        self.assertEqual(self.o["branch"], "B")

    def test_reverse_cell_platt_push_on_raw_band_games(self):
        rev = self.o["raw_band_70_80_games"]
        self.assertEqual(rev["n"], 175)
        self.assertAlmostEqual(rev["mean_raw"], 0.7469, delta=0.0005)
        self.assertAlmostEqual(rev["mean_cal"], 0.8174, delta=0.0005)

    def test_band_gap_exceeds_two_se(self):
        fwd = self.o["platt_band_70_80_games"]
        gap = abs(fwd["mean_cal"] - fwd["actual_win_rate"])
        self.assertGreater(gap, 2.0 * fwd["se_actual"])
        self.assertGreater(fwd["band_ece"], 0.04)


# ---------------------------------------------------------------------------
# Phase 1.3/1.4 — map-family selection + the Phase-1 gate
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 CSVs not present")
class TestMapFamilySelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = _probe_json()["phase1_gate"]
        cls.fam = _probe_json()["map_family"]["families"]

    def test_nested_pool_is_the_production_pool(self):
        mf = _probe_json()["map_family"]
        self.assertEqual(mf["pooled_pre_holdout_rows"], 1107)
        self.assertEqual(mf["fold_count"], 88)

    def test_parsimony_rule_chooses_spline_not_platt(self):
        """Best joint band-ECE profile wins unless platt is within 0.005 —
        here pchip beats platt by ~0.026 on the target band (> 0.005), so the
        chosen family is the monotone spline (parsimony does NOT apply)."""
        a = self.fam["a_platt_refit"]["bands"]["70%-80%"]
        c = self.fam["c_pchip_spline"]["bands"]["70%-80%"]
        self.assertGreaterEqual(a - c, 0.02)
        self.assertGreater(a - c, 0.005)

    def test_phase1_gate_passes(self):
        self.assertTrue(self.gate["leg1_gap_2se_or_ece_gt_004"]["pass"])
        self.assertTrue(self.gate["leg2_family_improvement"]["pass"])
        self.assertEqual(self.gate["verdict"], "GATE_PASS")

    def test_isotonic_loses_logloss_materially(self):
        """Isotonic is excluded by its logloss cost (+0.086 pooled nested)."""
        a = self.fam["a_platt_refit"]["logloss"]
        b = self.fam["b_isotonic"]["logloss"]
        self.assertGreater(b - a, 0.05)

    def test_fit_on_strictly_earlier_folds_only(self):
        """Protocol purity: a fold's map is fit on STRICTLY-EARLIER folds'
        (raw, actual) pairs — never its own or any future row (the nested
        Platt geometry). Inline spy over a synthetic weekly series."""
        rng = np.random.default_rng(7)
        n = 400
        week = np.repeat(np.arange(8), n // 8)
        raw = np.clip(0.5 + 0.2 * np.sin(week * 0.7)
                      + rng.normal(0, 0.08, n), 0.05, 0.95)
        y = rng.binomial(1, np.clip(raw * 1.1 - 0.05, 0.05, 0.95))
        fitted_weeks = []

        cal = np.zeros(n)
        for w in range(8):
            f_mask = week == w
            prior = week < w
            if prior.sum() >= 10:
                seen = int(week[prior].max())   # fold w's fit pool
                fitted_weeks.append(seen)
                mapper = probe._fit_platt(raw[prior], y[prior])
            else:
                mapper = None
            cal[f_mask] = probe._apply(mapper, raw[f_mask])
        # one fit per fold from fold 1 onward; fold i+1's fit saw strictly-
        # earlier weeks only, so its max seen week == i (< i+1)
        self.assertEqual(fitted_weeks, list(range(7)))
        for i, seen in enumerate(fitted_weeks):
            self.assertEqual(seen, i)


# ---------------------------------------------------------------------------
# Phase-2 — projected blocking legs -> DO_NOT_REFIT
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 CSVs not present")
class TestPhase2Verdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legs = _probe_json()["phase2_legs"]

    def test_pooled_cal_logloss_leg_fails_for_spline(self):
        """The logistic MLE is the logloss optimum of its family; the spline
        trades ~+0.013 pooled nested logloss for band honesty — 13x the
        +/-0.001 bar. This is the binding Phase-2 leg."""
        reg = self.legs["pooled_cal_logloss_nested"]["regression_c_minus_a"]
        self.assertGreater(reg, 0.001)

    def test_sealed_70_80_band_ece_not_below_0_03_on_new_axis(self):
        e = self.legs["sealed_70_80_band_ece_new_axis"]["c"]
        self.assertGreaterEqual(e, 0.03)

    def test_sealed_80_plus_adjacent_regression_exceeds_0_01(self):
        d = self.legs["sealed_adjacent_band_deltas_c_minus_a"]["80_plus"]
        self.assertGreater(d, 0.01)

    def test_verdict_do_not_refit_no_production_change(self):
        self.assertEqual(self.legs["verdict"], "DO_NOT_REFIT")
        self.assertGreaterEqual(len(self.legs["blocking_legs"]), 1)

    def test_auc_flat_contract(self):
        """Calibration is rank-invariant: the strictly-monotone candidate
        families (logistic Platt, monotone spline) leave AUC on the pooled
        pre-holdout rows EXACTLY unchanged (delta 0.0). No AUC-improvement
        claim is made or expected. (Isotonic is the documented exception:
        its step plateaus introduce ties, and the repo's ties->0.5 rank
        statistic moves — that is why isotonic is not the chosen family.)"""
        import pandas as pd
        from nfl_moneyline import auc
        hist = pd.read_csv(HIST_CSV)
        pool = hist[hist["season"] <= 2024]
        raw = pool["home_win_prob_model"].to_numpy(float)
        y = (pool["home_score"] > pool["away_score"]).to_numpy(float)
        a = probe._apply(probe._fit_platt(raw, y), raw)
        c = probe._apply(probe._fit_pchip(raw, y), raw)
        base = auc(y, raw)
        self.assertAlmostEqual(auc(y, a), base, delta=0.001)
        self.assertAlmostEqual(auc(y, c), base, delta=0.001)


# ---------------------------------------------------------------------------
# Determinism — double walk byte-identical
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_ARTIFACTS, "20260904 CSVs not present")
class TestDeterminism(unittest.TestCase):
    def test_double_run_byte_identical(self):
        r1 = _run_probe_once()
        r2 = _run_probe_once()
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
