#!/usr/bin/env python3
"""Tests for the NFL wide-pool validation rearchitecture (nfl_wide_pool_).

Validity-harness change only — production full-history refit/serving path
UNTOUCHED. The tests verify the re-baseline harness produced correct artifacts
(record + rolling brier + binary OOF + decided store) and that the protected-
delivery prefix is registered.

Runs against the committed artifacts in nfl-backend/data_delivery (produced by
run_nfl_wide_pool_baseline.py). No network in tests.
"""
from __future__ import annotations
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DD_STR = "nfl-backend/data_delivery"
DD = Path(__file__).resolve().parent.parent / "data_delivery"
RECORD = DD / "nfl_wide_pool_rearchitecture_3e8c8a510f04.json"
ROLLING_BRIER = DD / "nfl_rolling_brier_3e8c8a510f04.csv"
BINARY_OOF = DD / "nfl_binary_oof_3e8c8a510f04.csv"
DECIDED_STORE = DD / "nfl_decided_store_rs_2018_2025.csv"
LEGACY_FIXTURE = DD / "nfl_wide_pool_legacy_pins_3e8c8a510f04.json"

# ---- frame pins (committed canonical frame; production path UNTOUCHED) ----
CANONICAL_FRAME_SHA = "3e8c8a510f04"   # nfl_game_level_features.csv sha
LEGACY_POOL = {                          # OLD 2021-2024 pool (88 calendar-week folds)
    "geometry": "88 calendar-week folds, 2021-2024 pooled + 2025 sealed",
    "pooled_oof_n": 1107,
    "sealed_n": 285,
    "binary": {"platt_a": 1.276336, "platt_b": 0.121988,
               "pooled_platt": {"logloss": 0.6249, "auc": 0.6950, "ece": 0.0745}},
    "run_engine": {
        "pooled": {"totals_ece_offered": 0.087, "covers_ece_offered": 0.078,
                   "derived_ml": {"logloss": 0.6365, "auc": 0.695, "ece": 0.0435}},
        "sealed": {"totals_ece_offered": 0.1547, "covers_ece_offered": 0.1145,
                   "derived_ml": {"logloss": 0.6535, "auc": 0.6782, "ece": 0.1009}},
    },
    "recovery": {"elo_diff": 0.638, "ewm_net_pts_diff": 0.590,
                 "win_pct_diff": 0.649, "ewm_ypp_diff": 0.665},
}


class TestWidePoolRecordExists(unittest.TestCase):
    """The re-baseline record + artifacts exist and are non-trivial."""

    def test_record_exists_and_valid_json(self):
        self.assertTrue(RECORD.exists(), "record missing")
        blob = RECORD.read_text(encoding="utf-8")
        r = json.loads(blob)
        self.assertEqual(r["record"], "nfl_wide_pool_rearchitecture")
        self.assertEqual(r["frame_sha256"], CANONICAL_FRAME_SHA)

    def test_record_size_non_trivial(self):
        self.assertGreater(RECORD.stat().st_size, 5000,
                           "record too small to be a real re-baseline")

    def test_rolling_brier_exists(self):
        self.assertTrue(ROLLING_BRIER.exists())
        self.assertGreater(ROLLING_BRIER.stat().st_size, 1000)

    def test_binary_oof_exists(self):
        self.assertTrue(BINARY_OOF.exists())
        self.assertGreater(BINARY_OOF.stat().st_size, 50000)

    def test_decided_store_exists(self):
        self.assertTrue(DECIDED_STORE.exists())
        self.assertGreater(DECIDED_STORE.stat().st_size, 50000)


class TestWidePoolRecordContent(unittest.TestCase):
    """The record's content matches the spec architecture."""

    @classmethod
    def setUpClass(cls):
        cls.r = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_frame_sha_pin(self):
        self.assertEqual(self.r["frame_sha256"], CANONICAL_FRAME_SHA)

    def test_committed_frame_untouched_note(self):
        note = self.r["committed_frame_note"]
        self.assertIn("UNTOUCHED", note)
        self.assertIn("production full-history refit/serving path untouched",
                      note.lower())

    def test_step1_rs_decided_store_counts(self):
        s1 = self.r["step1_rs_decided_store"]
        self.assertEqual(s1["total"], 2127)       # 256 warmup + 1871 scored
        self.assertEqual(s1["scored"], 1871)
        self.assertEqual(s1["warmup"], 256)
        ps = s1["per_season"]
        self.assertEqual(int(ps["2018"]), 256)
        self.assertEqual(int(ps["2019"]), 256)
        self.assertEqual(int(ps["2020"]), 256)
        self.assertEqual(int(ps["2021"]), 272)
        self.assertEqual(int(ps["2022"]), 271)   # verified: 271 not 272
        self.assertEqual(int(ps["2023"]), 272)
        self.assertEqual(int(ps["2024"]), 272)
        self.assertEqual(int(ps["2025"]), 272)
        self.assertTrue(s1["playoffs_stripped"])

    def test_step2_fold_builder(self):
        s2 = self.r["step2_fold_builder"]
        self.assertIn("week-ID folds", s2["method"])
        self.assertEqual(s2["n_folds"], 124)
        self.assertEqual(s2["n_folds_full"], 124)
        self.assertEqual(s2["first_scored"], {"season": 2019, "week": 1})
        self.assertEqual(s2["last_scored"], {"season": 2025, "week": 18})
        wps = s2["weeks_per_season"]
        self.assertIn("2019", wps)
        self.assertEqual(int(wps["2019"]), 17)
        self.assertEqual(int(wps["2020"]), 17)
        for y in ("2021", "2022", "2023", "2024", "2025"):
            self.assertEqual(int(wps[y]), 18)

    def test_step3_calibration(self):
        s3 = self.r["step3_calibration"]
        self.assertEqual(s3["seed"], 300)
        self.assertIn("identity map", s3["method"])
        self.assertEqual(s3["raw_segment"]["2019_per_season_label"], "raw")
        self.assertIn("calibrated_pool", s3)

    def test_step4_binary_pooled_metrics(self):
        b = self.r["step4_binary_baseline"]["pooled"]
        self.assertEqual(b["n_oof"], 1871)
        self.assertEqual(b["n_folds"], 124)
        self.assertEqual(b["calib_seed"], 300)
        self.assertIn("raw", b)
        self.assertIn("calibrated", b)
        self.assertIn("platt_a", b)
        self.assertIn("platt_b", b)
        # The re-baseline's pooled metrics (full 124-fold walk).
        self.assertAlmostEqual(b["raw"]["logloss"], 0.6282, places=4)
        self.assertAlmostEqual(b["calibrated"]["logloss"], 0.6294, places=4)
        self.assertAlmostEqual(b["raw"]["auc"], 0.6977, places=4)
        self.assertAlmostEqual(b["calibrated"]["auc"], 0.6964, places=4)

    def test_step4_binary_per_season(self):
        ps = self.r["step4_binary_baseline"]["per_season"]
        self.assertEqual(len(ps), 7)              # 2019-2025
        seasons = [p["season"] for p in ps]
        self.assertEqual(seasons, list(range(2019, 2026)))
        # 2019 = raw segment; 2020-2025 = cal.
        self.assertEqual(ps[0]["segment"], "raw")
        for p in ps[1:]:
            self.assertEqual(p["segment"], "cal")

    def test_step4_binary_rolling_brier(self):
        rb = self.r["step4_binary_baseline"]["rolling_brier"]
        self.assertEqual(len(rb), 124)           # one point per scored fold
        # raw segment = 2019 games (first ~19 points, until seed reached).
        raw_pts = [p for p in rb if p["segment"] == "raw"]
        self.assertGreater(len(raw_pts), 0)
        self.assertLessEqual(len(raw_pts), 19)   # 300 seed reached ~fold 19
        # every point has the required fields.
        for p in rb:
            self.assertIn("season", p)
            self.assertIn("week", p)
            self.assertIn("n", p)
            self.assertIn("brier", p)
            self.assertIn("segment", p)

    def test_step4_run_engine_pooled_metrics(self):
        r4 = self.r["step4_run_engine_baseline"]["pooled"]
        self.assertEqual(r4["n_oof"], 1871)
        self.assertEqual(r4["n_folds"], 124)
        self.assertIn("derived_ml", r4)
        self.assertIn("totals_ece_offered", r4)
        self.assertIn("covers_ece_offered", r4)
        self.assertIn("totals_ece_fair", r4)
        self.assertIn("covers_ece_fair", r4)
        self.assertAlmostEqual(r4["derived_ml"]["logloss"], 0.6364, places=4)
        self.assertAlmostEqual(r4["derived_ml"]["auc"], 0.6936, places=4)
        self.assertAlmostEqual(r4["derived_ml"]["ece"], 0.0292, places=4)

    def test_step4_run_engine_per_season(self):
        ps = self.r["step4_run_engine_baseline"]["per_season"]
        self.assertEqual(len(ps), 7)
        for p in ps:
            self.assertIn("season", p)
            self.assertIn("n", p)
            self.assertIn("derived_ml", p)

    def test_guardrails(self):
        g = self.r["guardrails"]
        self.assertTrue(g["production_serving_untouched"])
        self.assertTrue(g["no_production_prediction_change"])
        self.assertTrue(g["production_canonical_frame_untouched"])
        self.assertTrue(g["historical_records_never_edited"])

    def test_data_notes(self):
        notes = self.r["data_notes"]
        self.assertGreater(len(notes), 0)
        joined = " ".join(notes).lower()
        self.assertIn("2022", joined)
        self.assertIn("covid", joined)
        self.assertIn("2018", joined)


class TestWidePoolDeterminism(unittest.TestCase):
    """Byte-identical-prediction: the OOF CSV re-derives the record's pooled
    metrics exactly (the re-baseline walk is deterministic)."""

    def _load_oof(self):
        import pandas as pd
        return pd.read_csv(BINARY_OOF)

    def _compute_metrics(self, y, p):
        from nfl_moneyline import compute_metrics
        return compute_metrics(y, p)

    def _rec(self):
        return json.loads(RECORD.read_text(encoding="utf-8"))

    def test_oof_row_count_matches_record(self):
        oof = self._load_oof()
        self.assertEqual(len(oof), self._rec()["step4_binary_baseline"]["pooled"]["n_oof"])

    def test_raw_logloss_matches_record(self):
        oof = self._load_oof()
        y = oof["home_win"].to_numpy(float)
        raw = oof["home_win_prob_raw"].to_numpy(float)
        mr = self._compute_metrics(y, raw)
        np = __import__("numpy")
        np.testing.assert_allclose(
            mr["logloss"],
            self._rec()["step4_binary_baseline"]["pooled"]["raw"]["logloss"],
            atol=1e-6)

    def test_cal_logloss_matches_record(self):
        oof = self._load_oof()
        y = oof["home_win"].to_numpy(float)
        cal = oof["home_win_prob_calibrated"].to_numpy(float)
        mc = self._compute_metrics(y, cal)
        np = __import__("numpy")
        np.testing.assert_allclose(
            mc["logloss"],
            self._rec()["step4_binary_baseline"]["pooled"]["calibrated"]["logloss"],
            atol=1e-6)

    def test_raw_auc_matches_record(self):
        oof = self._load_oof()
        y = oof["home_win"].to_numpy(float)
        raw = oof["home_win_prob_raw"].to_numpy(float)
        mr = self._compute_metrics(y, raw)
        np = __import__("numpy")
        np.testing.assert_allclose(
            mr["auc"], self._rec()["step4_binary_baseline"]["pooled"]["raw"]["auc"],
            atol=1e-4)

    def test_cal_auc_matches_record(self):
        oof = self._load_oof()
        y = oof["home_win"].to_numpy(float)
        cal = oof["home_win_prob_calibrated"].to_numpy(float)
        mc = self._compute_metrics(y, cal)
        np = __import__("numpy")
        np.testing.assert_allclose(
            mc["auc"], self._rec()["step4_binary_baseline"]["pooled"]["calibrated"]["auc"],
            atol=1e-4)

    def test_raw_ece_matches_record(self):
        oof = self._load_oof()
        y = oof["home_win"].to_numpy(float)
        raw = oof["home_win_prob_raw"].to_numpy(float)
        mr = self._compute_metrics(y, raw)
        np = __import__("numpy")
        np.testing.assert_allclose(
            mr["ece"], self._rec()["step4_binary_baseline"]["pooled"]["raw"]["ece"],
            atol=1e-4)

    def test_cal_ece_matches_record(self):
        oof = self._load_oof()
        y = oof["home_win"].to_numpy(float)
        cal = oof["home_win_prob_calibrated"].to_numpy(float)
        mc = self._compute_metrics(y, cal)
        np = __import__("numpy")
        np.testing.assert_allclose(
            mc["ece"], self._rec()["step4_binary_baseline"]["pooled"]["calibrated"]["ece"],
            atol=1e-4)

    def test_oof_home_win_distribution(self):
        oof = self._load_oof()
        y = oof["home_win"].to_numpy(float)
        self.assertAlmostEqual(y.mean(), 0.5, places=1)  # home wins ~half

    def test_oof_segment_labels(self):
        oof = self._load_oof()
        seg = oof["segment"].value_counts()
        self.assertIn("raw", seg)
        self.assertIn("cal", seg)
        # 2019 = raw segment (first 300 scored games, ~256 rows).
        self.assertGreater(seg["raw"], 0)


class TestWidePoolDecidedStore(unittest.TestCase):
    """The committed RS-only 2018-2025 decided store has the right shape."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.df = pd.read_csv(DECIDED_STORE)

    def test_row_count(self):
        self.assertEqual(len(self.df), 2127)

    def test_season_counts(self):
        per = self.df.groupby("season").size()
        self.assertEqual(per[2018], 256)
        self.assertEqual(per[2019], 256)
        self.assertEqual(per[2020], 256)
        self.assertEqual(per[2021], 272)
        self.assertEqual(per[2022], 271)
        self.assertEqual(per[2023], 272)
        self.assertEqual(per[2024], 272)
        self.assertEqual(per[2025], 272)

    def test_reg_only(self):
        self.assertTrue((self.df["game_type"] == "REG").all())

    def test_no_playoff_rows(self):
        postseason = {"WC", "DIV", "CON", "SB"}
        self.assertFalse(self.df["game_type"].isin(postseason).any())

    def test_has_required_columns(self):
        for c in ("game_id", "season", "week", "gameday",
                  "home_score", "away_score", "total"):
            self.assertIn(c, self.df.columns)

    def test_2018_source_note(self):
        r = json.loads(RECORD.read_text(encoding="utf-8"))
        s1 = r["step1_rs_decided_store"]
        self.assertIn("schedule-only", s1.get("2018_source", ""))
        self.assertIn("no PBP", s1.get("2018_source", ""))


class TestWidePoolLegacyFixture(unittest.TestCase):
    """The legacy 88-fold geometry pins are archived (not edited into
    historical records)."""

    @classmethod
    def setUpClass(cls):
        cls.r = json.loads(RECORD.read_text(encoding="utf-8"))

    def test_legacy_fixture_pointer(self):
        s5 = self.r["step5_legacy_pins_fixture"]
        self.assertIn("path", s5)
        self.assertIn("note", s5)
        self.assertIn("pins", s5)
        self.assertIn("old_pool", s5["pins"])

    def test_legacy_pins_content(self):
        pins = self.r["step5_legacy_pins_fixture"]["pins"]
        old = pins["old_pool"]
        self.assertEqual(old["pooled_oof_n"], 1107)
        self.assertEqual(old["sealed_n"], 285)
        self.assertEqual(old["fold_count"], 88)
        self.assertAlmostEqual(old["binary"]["platt_a"], 1.276336, places=5)
        self.assertAlmostEqual(old["binary"]["platt_b"], 0.121988, places=5)

    def test_legacy_pins_note(self):
        note = self.r["step5_legacy_pins_fixture"]["note"]
        self.assertIn("archived", note.lower())
        self.assertIn("never edited", note)


class TestWidePoolPrefixProtection(unittest.TestCase):
    """The nfl_wide_pool_ prefix is registered in master_pipeline's
    _PROTECTED_DELIVERY_PREFIXES (targeted, never a broad nfl_)."""

    def test_nfl_wide_pool_prefix_registered(self):
        from master_pipeline import _PROTECTED_DELIVERY_PREFIXES as P
        self.assertIn("nfl_wide_pool_", P)

    def test_no_broad_nfl_prefix(self):
        from master_pipeline import _PROTECTED_DELIVERY_PREFIXES as P
        self.assertNotIn("nfl_", P)

    def test_record_name_is_protected(self):
        from master_pipeline import _is_protected_name
        rel = f"{DD_STR}/nfl_wide_pool_rearchitecture_3e8c8a510f04.json"
        self.assertTrue(_is_protected_name(rel))

    def test_record_classifies_protected(self):
        from master_pipeline import classify_stale, _is_protected_name
        EMPTY: set[str] = set()
        rel = f"{DD_STR}/nfl_wide_pool_rearchitecture_3e8c8a510f04.json"
        self.assertEqual(classify_stale(rel, EMPTY, EMPTY, EMPTY), "protected")

    def test_dated_moneyline_untouched_by_addition(self):
        from master_pipeline import _is_protected_name
        rel = f"{DD_STR}/nfl_moneyline_v1_20260830.json"
        self.assertFalse(_is_protected_name(rel))


if __name__ == "__main__":
    unittest.main()
