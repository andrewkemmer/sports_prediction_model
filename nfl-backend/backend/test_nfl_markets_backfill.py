"""NFL run-engine decided-history backfill tests (run_nfl_markets_backfill).

Pins the Phase-A connector: the decided OOF store (pooled 2021-24 n=1,091 +
sealed 2025 n=285 = 1,376 rows) emitted through the slate emitter's schema
with actuals + honest outcomes, one dated markets artifact carrying BOTH
kinds (oof + slate), the deterministic E2 regeneration verified against the
committed records, and the market-free / served-pool invariants.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import nfl_slate_engine as SE  # noqa: E402
import run_nfl_markets_backfill as B  # noqa: E402
from nfl_features import DECIDED_FRAME, FEATURE_COLUMNS  # noqa: E402


def _synth_rows(n: int, seed: int = 0, tag: str = "a") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({
        "game_id": [f"{tag}_{i:02d}_syn" for i in range(n)],
        "pred_home": rng.uniform(18.0, 28.0, n),
        "pred_away": rng.uniform(16.0, 26.0, n),
        "home_score": rng.integers(10, 45, n),
        "away_score": rng.integers(10, 45, n),
    })
    return rows


class TestDecidedStore(unittest.TestCase):
    def _store(self):
        pooled = _synth_rows(8, seed=1, tag="p")
        pooled["frame_view"] = "pooled"
        sealed = _synth_rows(5, seed=2, tag="s")
        sealed["frame_view"] = "sealed"
        lines = pd.DataFrame({
            "game_id": pd.concat([pooled, sealed])["game_id"].values,
            "spread_line": [-3.0] * 13,
            "total_line": [46.0] * 13,
        })
        meta = pd.DataFrame({
            "game_id": pd.concat([pooled, sealed])["game_id"].values,
            "season": [2021] * 8 + [2025] * 5,
            "week": list(range(1, 14)),
            "gameday": pd.date_range("2021-09-09", periods=13).strftime(
                "%Y-%m-%d"),
            "home_team": ["H"] * 13,
            "away_team": ["A"] * 13,
        })
        return B.build_decided_store(
            pooled, sealed, SE.pinned_joint_params(), SE.PINNED_P_TIE,
            lines, meta)

    def test_store_schema_and_actuals(self):
        oof, pins = self._store()
        self.assertEqual(len(oof), 13)
        self.assertTrue((oof["kind"] == "oof").all())
        self.assertTrue((oof["decided"] == 1).all())
        # actuals present and consistent with the input scores
        self.assertTrue(oof["home_score"].notna().all())
        self.assertTrue(oof["away_score"].notna().all())
        self.assertTrue((oof["total"]
                         == oof["home_score"] + oof["away_score"]).all())
        self.assertTrue((oof["margin"]
                         == oof["home_score"] - oof["away_score"]).all())
        # outcomes are binary and complement
        self.assertTrue(oof["y_over_fair"].isin([0.0, 1.0]).all())
        self.assertTrue(np.allclose(oof["y_over_fair"] + oof["y_under_fair"]
                                    + oof["y_push_fair"], 1.0))
        self.assertTrue(np.allclose(oof["y_cover_fair"]
                                    + (oof["margin"] <= oof["fair_spread"])
                                    .astype(float), 1.0))
        # offered-level outcomes at the quoted lines
        self.assertTrue(np.allclose(
            oof["y_over_offered"]
            + oof["y_under_offered"] + oof["y_push_total_offered"], 1.0))
        # derived-ML pair complements
        self.assertTrue(np.allclose(oof["p_home_win_derived"]
                                    + oof["p_away_win_derived"], 1.0))
        # both views represented
        self.assertEqual(set(oof["frame_view"]), {"pooled", "sealed"})
        self.assertTrue(set(pins) == {"pooled", "sealed"})

    def test_store_deterministic_double_walk(self):
        a, _ = self._store()
        b, _ = self._store()
        self.assertEqual(a.drop(columns=["kind"]).to_csv(index=False),
                         b.drop(columns=["kind"]).to_csv(index=False))

    def test_store_uses_pinned_joint_params(self):
        # The store is built with the PINNED constants (never refit) — the
        # p_tie diagonal + mass conservation follow from the engine.
        oof, _ = self._store()
        self.assertAlmostEqual(float(oof["p_tie"].mean()), SE.PINNED_P_TIE,
                               places=4)


class TestRegenerationVerification(unittest.TestCase):
    def test_verify_raises_on_drift(self):
        pooled = _synth_rows(8, seed=3)
        sealed = _synth_rows(5, seed=4)
        rounds = dict(SE.MEDIAN_ROUNDS)
        params = SE.pinned_joint_params()
        # wrong round counts must raise
        with self.assertRaises(RuntimeError):
            B.verify_regeneration(pooled, sealed, {"home": 1, "away": 2},
                                  params, SE.PINNED_P_TIE)
        with self.assertRaises(RuntimeError):
            B.verify_regeneration(pooled.iloc[:5], sealed, rounds, params,
                                  SE.PINNED_P_TIE)
        with self.assertRaises(RuntimeError):
            B.verify_regeneration(pooled, sealed.iloc[:2], rounds, params,
                                  SE.PINNED_P_TIE)


class TestRealArtifacts(unittest.TestCase):
    """Round-trip the dated backfill artifact (when present in
    data_delivery): both kinds, actuals keyed by game_id, leakage, honest
    monitor state."""

    DD = BACKEND.parent / "data_delivery"

    def _latest(self, prefix: str, ext: str) -> Path | None:
        cands = sorted(self.DD.glob(f"{prefix}*{ext}"))
        return cands[-1] if cands else None

    def test_markets_csv_both_kinds_with_actuals(self):
        path = self._latest("nfl_run_engine_markets_", ".csv")
        if path is None:
            self.skipTest("no markets artifact committed")
        df = pd.read_csv(path)
        self.assertTrue(set(df["kind"]).issubset({"slate", "oof"}))
        if "oof" not in set(df["kind"]):
            self.skipTest("artifact predates the decided-history backfill")
        oof = df[df["kind"] == "oof"]
        self.assertEqual(len(oof), B.POOLED_N + B.SEALED_N)
        # actuals keyed by game_id against the decided frame
        decided = pd.read_csv(DECIDED_FRAME)
        merged = oof[["game_id", "home_score", "away_score"]].merge(
            decided[["game_id", "home_score", "away_score"]],
            on="game_id", how="left", suffixes=("", "_decided"))
        self.assertEqual(len(merged), len(oof))
        self.assertTrue((merged["home_score"] == merged["home_score_decided"]
                         ).all())
        self.assertTrue((merged["away_score"] == merged["away_score_decided"]
                         ).all())
        # outcomes present + fair lines populated
        for c in ("margin", "y_home_win", "y_over_fair", "y_cover_fair",
                  "p_over_fair", "p_cover_fair", "fair_spread",
                  "fair_total", "derived_ml"):
            self.assertIn(c, oof.columns)
        self.assertTrue(oof["y_home_win"].isin([0.0, 1.0]).all())
        # 100% offered-line coverage on the decided rows
        self.assertTrue(oof["spread_line"].notna().all())
        self.assertTrue(oof["total_line"].notna().all())
        self.assertTrue(oof["p_cover_offered"].notna().all())
        # grid columns populated on EVERY row (both kinds)
        grid_cols = ([f"p_home_cover_{SE._fname(float(L))}"
                      for L in SE.SPREAD_INT_LINES]
                     + [f"p_over_{SE._fname(float(U))}"
                        for U in SE.TOTAL_INT_LINES])
        self.assertFalse(df[grid_cols].isna().any().any())
        # board rows carry no decided-targets (undecided by definition)
        slate = df[df["kind"] == "slate"]
        self.assertTrue(slate["home_score"].isna().all())
        self.assertTrue(slate["margin"].isna().all())

    def test_leakage_no_future_info(self):
        path = self._latest("nfl_run_engine_markets_", ".csv")
        if path is None:
            self.skipTest("no markets artifact committed")
        df = pd.read_csv(path)
        oof = df[df["kind"] == "oof"]
        if not len(oof):
            self.skipTest("artifact predates the decided-history backfill")
        days = pd.to_datetime(oof["gameday"], errors="coerce")
        # pooled 2021-24 + sealed 2025 rows only (the 2025 season runs
        # through its final game — Super Bowl LX on 2026-02-08); sealed is
        # fit on 2019-24 strictly-prior rows by construction
        self.assertLessEqual(oof["season"].max(), 2025)
        self.assertGreaterEqual(days.min(), pd.Timestamp("2021-01-01"))
        # board rows are strictly later than every decided row
        slate = df[df["kind"] == "slate"]
        if len(slate):
            self.assertGreater(
                pd.to_datetime(slate["gameday"]).min(), days.max())

    def test_monitor_honest_empty_state(self):
        path = self._latest("nfl_run_engine_monitor_", ".json")
        if path is None:
            self.skipTest("no monitor artifact committed")
        data = json.loads(path.read_text())
        base = data["oof_baseline_research_pinned"]
        self.assertEqual(base["covers_ece_pooled"], 0.078)
        self.assertEqual(base["totals_ece_pooled_own"], 0.087)
        # slate history still empty — no served-slate outcomes exist
        self.assertEqual(data["slate_history"], [])
        # backfill-computed calibration present (when the store shipped)
        if "oof_decided_store_backfill_computed" in data:
            cal = data["oof_decided_store_backfill_computed"]["calibration"]
            self.assertEqual(cal["pooled"]["n"], B.POOLED_N)
            self.assertEqual(cal["sealed"]["n"], B.SEALED_N)

    def test_meta_provenance_and_treatment(self):
        path = self._latest("nfl_run_engine_markets_", ".meta.json")
        if path is None:
            self.skipTest("no meta artifact committed")
        meta = json.loads(path.read_text())
        self.assertFalse(meta["treatment"]["shrink_applied"])
        self.assertEqual(meta["treatment"]["mode"],
                         "own-line quoting both sides with honest ECE")
        self.assertEqual(meta["provenance_records"][0],
                         "nfl_era_3e8c8a510f04.json")
        self.assertTrue(meta["engines_modified"] is False)
        self.assertTrue(meta["moneyline_pool_untouched"] is True)


class TestScopePins(unittest.TestCase):
    def test_served_pool_untouched(self):
        # 12 market-free features + the is_home anchor (13 total)
        self.assertEqual(len(FEATURE_COLUMNS), 13)
        self.assertIn("is_home", FEATURE_COLUMNS)
        self.assertNotIn("market_home_implied", FEATURE_COLUMNS)
        self.assertNotIn("spread_line", FEATURE_COLUMNS)
        self.assertNotIn("total_line", FEATURE_COLUMNS)

    def test_backfill_source_uses_pinned_constants(self):
        src = (BACKEND / "run_nfl_markets_backfill.py").read_text()
        # the store must be built from the PINNED joint params (never refit)
        self.assertIn("pinned_joint_params", src)
        self.assertIn("PINNED_P_TIE", src)
        self.assertNotIn("fit_joint_params", src.replace(
            "pinned_joint_params", ""))


if __name__ == "__main__":
    unittest.main()