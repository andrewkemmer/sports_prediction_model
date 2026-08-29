"""Tests for the margin-distribution diagnostic record (read-only).

The harness (run_margin_distribution_diagnostic.py) writes
data_delivery/margin_distribution_diagnostic_<date>.json and modifies
nothing else. These tests assert the record is well-formed and
internally consistent: per-margin probabilities sum to 1.0, the tables
have the expected rows, the verdict names a mechanism, and the harness
leaves the OOF/markets artifacts untouched (mtime hash check on
regeneration is out of scope; the harness performs no writes by
construction — verified by reading its source for write calls).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data_delivery"


class TestMarginDistributionRecord(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        hits = sorted(DATA.glob("margin_distribution_diagnostic_*.json"))
        if not hits:
            raise unittest.SkipTest(
                "no margin_distribution_diagnostic_*.json record committed")
        cls.record = json.loads(hits[-1].read_text())

    def test_record_parses_and_has_expected_tables(self):
        d = self.record
        for key in ("per_margin", "per_total", "correlation",
                    "total_bucket_split", "run_line_minus1_decomposition",
                    "verdict"):
            self.assertIn(key, d, f"record missing {key}")
        self.assertEqual(len(d["per_margin"]), 13)   # −6 .. +6
        self.assertEqual(len(d["per_total"]), 7)     # 6 .. 12
        self.assertEqual([r["margin"] for r in d["per_margin"]],
                         list(range(-6, 7)))
        self.assertEqual([r["total"] for r in d["per_total"]],
                         list(range(6, 13)))
        self.assertEqual(d["n_games"], 6812)

    def test_per_margin_probabilities_sum_to_one(self):
        """The per_margin table is a −6..+6 window; the out-of-window tail
        is recorded explicitly in margin_pmf_window, so window + tail must
        sum to 1.0 (± rounding)."""
        w = self.record["margin_pmf_window"]
        total = (sum(r["pred_p"] for r in self.record["per_margin"])
                 + w["pred_tail_beyond"])
        self.assertAlmostEqual(total, 1.0, places=3,
                               msg="predicted per-margin PMF (window + "
                                   "tail) must sum to 1.0")
        total_act = (sum(r["actual_p"] for r in self.record["per_margin"])
                     + w["actual_tail_beyond"])
        self.assertAlmostEqual(total_act, 1.0, places=3,
                               msg="actual per-margin frequencies "
                                   "(window + tail) must sum to 1.0")

    def test_margin_zero_discrepancy_recorded(self):
        """margin=0 is impossible in baseball — after the tie fix the
        persisted distribution has P(margin=0) = 0 exactly, and the record
        must show the raw pre-fix tie mass for evidence."""
        by_m = {r["margin"]: r for r in self.record["per_margin"]}
        self.assertEqual(by_m[0]["actual_p"], 0.0)
        self.assertEqual(by_m[0]["pred_p"], 0.0)
        w = self.record["margin_pmf_window"]
        self.assertGreater(w["raw_pre_fix_tie_mass"], 0.05)
        self.assertEqual(w["tie_handling"],
                         "margin distribution conditioned on no tie "
                         "(P(margin=0)=0; mass rescaled by 1/(1-P0)) — "
                         "the run-engine tie fix")

    def test_home_one_run_asymmetry_recorded(self):
        """The one-run gap is asymmetric: actual +1 >> actual −1."""
        by_m = {r["margin"]: r for r in self.record["per_margin"]}
        self.assertGreater(by_m[1]["actual_p"], by_m[-1]["actual_p"] + 0.03)
        self.assertLess(abs(by_m[1]["pred_p"] - by_m[-1]["pred_p"]), 0.01)

    def test_verdict_present_and_evidence_nonempty(self):
        v = self.record["verdict"]
        self.assertIn("most_likely_mechanism", v)
        self.assertIn("summary", v)
        self.assertTrue(v["evidence"])
        for e in v["evidence"]:
            self.assertIsInstance(e, str) and self.assertTrue(e)

    def test_harness_writes_only_the_record(self):
        """The harness must not modify any artifact — source-level check:
        the only write path is the diagnostic JSON output."""
        src = (Path(__file__).resolve().parents[0]
               / "run_margin_distribution_diagnostic.py").read_text()
        # No to_csv / write_text / open(...,'w') except the record itself.
        self.assertNotIn(".to_csv(", src)
        self.assertNotIn(".write_text(", src)
        writes = [ln.strip() for ln in src.splitlines()
                  if "open(" in ln and ("\"w\"" in ln or "'w'" in ln)]
        self.assertEqual(len(writes), 1,
                         f"harness must write exactly one file, got: {writes}")

    def test_artifacts_untouched(self):
        """The OOF and markets artifacts are still readable and unchanged in
        shape (the harness is read-only over them)."""
        oof = pd.read_csv(DATA / "run_engine_oof_20260829.csv")
        mk = pd.read_csv(DATA / "run_engine_markets_20260829.csv")
        self.assertEqual(len(oof), 6812)
        self.assertEqual(len(mk[mk["kind"] == "oof"]), 6812)


if __name__ == "__main__":
    unittest.main()
