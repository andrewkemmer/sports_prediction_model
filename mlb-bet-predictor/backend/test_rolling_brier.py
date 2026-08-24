"""Tests for the Rolling Brier (Last 30 Days) series.

Covers the compute_rolling_brier contract:
- hand-computed fixture values (identity calibrator),
- trailing CALENDAR window semantics (off-days contribute nothing, no NaN),
- min-games-per-day sparsity threshold,
- empty / all-sparse / missing-column inputs → loud warning + empty series,
- deployed Platt map actually applied to per-game Brier,
- pipeline embedding: _model_monitor_json carries the series for the frontend.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import explainability
from explainability import compute_rolling_brier


def _row(d: str, p: float, y: int) -> dict:
    return {"game_date": d, "home_win": y, "home_win_prob_model": p}


def _fixture() -> pd.DataFrame:
    """3 qualifying days separated by off-days + 1 sparse day.

    Day A (2026-06-01, 5 games): briers .04/.36/.25/.25/.36 → mean 0.252
    Day B (2026-06-02, 2 games): sparse — excluded (<5 games)
    Day C (2026-06-29, 5 games): all (0.5, 1) → 0.25 each
    Day D (2026-08-15, 5 games): all (0.0, 0) → 0.0 each; A and C fall
        outside its trailing 30-day window ([07-17 .. 08-15]).
    """
    rows = [
        _row("2026-06-01", 0.8, 1), _row("2026-06-01", 0.6, 0),
        _row("2026-06-01", 0.5, 1), _row("2026-06-01", 0.5, 0),
        _row("2026-06-01", 0.4, 1),
        # Sparse day B:
        _row("2026-06-02", 0.9, 1), _row("2026-06-02", 0.2, 0),
        # Day C:
        *[_row("2026-06-29", 0.5, 1) for _ in range(5)],
        # Day D:
        *[_row("2026-08-15", 0.0, 0) for _ in range(5)],
    ]
    return pd.DataFrame(rows)


class TestRollingBrierComputation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(explainability, "DATA_DELIVERY_DIR",
                               Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_hand_computed_values_identity_calibrator(self):
        out = compute_rolling_brier(_fixture(), "20260824", calibrator=None)

        self.assertEqual(out["n_points"], 3)  # A, C, D — sparse B excluded
        pts = {p["date"]: p for p in out["series"]}

        # Day A window contains only day A's five games.
        self.assertAlmostEqual(pts["2026-06-01"]["brier"], 0.252, places=6)
        self.assertEqual(pts["2026-06-01"]["games"], 5)

        # Day C's trailing window spans the June off-days but NOT day D.
        self.assertAlmostEqual(pts["2026-06-29"]["brier"],
                               (5 * 0.252 + 5 * 0.25) / 10, places=6)
        self.assertEqual(pts["2026-06-29"]["games"], 10)

        # Day D is alone in its window (A/C older than 30 calendar days).
        self.assertAlmostEqual(pts["2026-08-15"]["brier"], 0.0, places=6)
        self.assertEqual(pts["2026-08-15"]["games"], 5)

        self.assertEqual(out["excluded_sparse_days"], 1)
        self.assertEqual(out["n_games_total"], 17)
        # History mean over ALL decided games (incl. the sparse day).
        expected_mean = (1.26 + 0.05 + 5 * 0.25 + 0.0) / 17
        self.assertAlmostEqual(out["history_mean_brier"], round(expected_mean, 6),
                               places=6)
        # No NaN ever leaks into the series.
        self.assertTrue(all(np.isfinite(p["brier"]) for p in out["series"]))
        # Artifact written and consistent with the return value.
        art = Path(self._tmp.name) / "rolling_brier_20260824.json"
        self.assertTrue(art.exists())
        on_disk = json.loads(art.read_text())
        self.assertEqual(on_disk["series"], out["series"])
        self.assertEqual(on_disk["window_days"], 30)
        self.assertEqual(on_disk["min_games_per_day"], 5)

    def test_deployed_platt_map_is_applied(self):
        one = pd.DataFrame([_row("2026-06-01", 0.8, 1)] * 6)
        cal = {"method": "platt", "a": 0.5, "b": 0.0}
        out = compute_rolling_brier(one, "20260824", calibrator=cal)
        from calibration import apply_platt
        p_cal = float(apply_platt(np.array([0.8]), cal)[0])
        self.assertAlmostEqual(out["series"][0]["brier"],
                               round((p_cal - 1) ** 2, 6), places=6)
        self.assertFalse(out["calibrator_is_identity"])

    def test_semantics_quantity3_deployed_map_not_prequential(self):
        """Task-A guardrail: rolling Brier must use quantity (3), the deployed
        global map σ(a·logit(p_raw)+b) — NOT the per-fold prequential column.
        Recomputed here with an INDEPENDENT sigmoid/logit implementation."""
        df = _fixture()
        cal = {"method": "platt", "a": 0.57497, "b": 0.011677}
        out = compute_rolling_brier(df, "20260824", calibrator=cal)
        # Independent reimplementation of the deployed map:
        p_raw = pd.to_numeric(df["home_win_prob_model"], errors="coerce")
        logit = np.log(p_raw / (1 - p_raw)).to_numpy(float)
        p_dep = 1.0 / (1.0 + np.exp(-(cal["a"] * logit + cal["b"])))
        y = pd.to_numeric(df["home_win"], errors="coerce").to_numpy(float)
        briers = pd.Series((p_dep - y) ** 2)
        dates = pd.to_datetime(df["game_date"]).dt.normalize()
        day_a = briers[dates == pd.Timestamp("2026-06-01")]
        pts = {p["date"]: p for p in out["series"]}
        self.assertAlmostEqual(pts["2026-06-01"]["brier"],
                               round(float(day_a.mean()), 6), places=5)
        # Metadata pins the semantics so regressions are visible in artifacts.
        self.assertEqual(out["source_column"], "home_win_prob_model")
        self.assertIn("deployed Platt map", out["calibration"])
        self.assertIn("not directly comparable", out["map_scope_note"])

    def test_custom_window_and_threshold(self):
        out = compute_rolling_brier(_fixture(), "20260824",
                                    window_days=7, min_games_per_day=2,
                                    calibrator=None)
        pts = {p["date"]: p for p in out["series"]}
        # With min_games=2, day B qualifies; its 7-day trailing window still
        # contains day A ([May 27 .. Jun 2]), so it averages both days.
        self.assertIn("2026-06-02", pts)
        self.assertAlmostEqual(pts["2026-06-02"]["brier"],
                               (5 * 0.252 + 0.05) / 7, places=6)
        self.assertAlmostEqual(pts["2026-06-29"]["brier"], 0.25, places=6)
        self.assertEqual(out["min_games_per_day"], 2)
        self.assertEqual(out["window_days"], 7)

    def test_empty_input_warns_and_returns_empty_series(self):
        with self.assertLogs(explainability.logger, level="WARNING") as logs:
            out = compute_rolling_brier(pd.DataFrame(), "20260824")
        self.assertEqual(out["series"], [])
        self.assertTrue(any("Rolling Brier" in line for line in logs.output))

    def test_none_input_warns_and_returns_empty_series(self):
        with self.assertLogs(explainability.logger, level="WARNING"):
            out = compute_rolling_brier(None, "20260824")
        self.assertEqual(out["series"], [])

    def test_all_sparse_input_yields_empty_not_nan(self):
        df = pd.DataFrame([
            _row("2026-06-01", 0.5, 1),
            _row("2026-06-03", 0.5, 0),
            _row("2026-06-05", 0.5, 1),
        ])
        with self.assertLogs(explainability.logger, level="WARNING") as logs:
            out = compute_rolling_brier(df, "20260824")
        self.assertEqual(out["series"], [])
        self.assertEqual(out["excluded_sparse_days"], 3)
        self.assertTrue(any("minimum" in line for line in logs.output))

    def test_undecided_and_malformed_rows_are_dropped(self):
        df = pd.DataFrame([
            *_fixture().to_dict(orient="records"),
            _row("2026-07-01", 0.7, np.nan),   # undecided slate row
            _row("2026-07-02", np.nan, 1),     # missing probability
            _row("not-a-date", 0.5, 1),        # malformed date
            {"game_date": "2026-07-03"},       # missing columns entirely
        ])
        out = compute_rolling_brier(df, "20260824")
        self.assertTrue(all(np.isfinite(p["brier"]) for p in out["series"]))
        # Only the four fixture game-days survive.
        self.assertEqual({p["date"] for p in out["series"]},
                         {"2026-06-01", "2026-06-29", "2026-08-15"} |
                         set())  # day A/C/D qualify; sparse B never appears
        self.assertNotIn("2026-07-01", {p["date"] for p in out["series"]})


class TestModelMonitorEmbedding(unittest.TestCase):
    """Pipeline-level wiring: the series reaches model_monitor_<date>.json."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for mod_name in ("pipeline", "explainability"):
            patcher = patch(f"{mod_name}.DATA_DELIVERY_DIR",
                            Path(self._tmp.name))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_monitor_json_embeds_series_and_meta(self):
        import pipeline as pl

        drift_df = pd.DataFrame(columns=["status"])
        rb = {
            "window_days": 30,
            "min_games_per_day": 5,
            "excluded_sparse_days": 3,
            "calibrator_is_identity": False,
            "n_games_total": 4001,
            "history_mean_brier": 0.2451,
            "n_points": 40,
            "series": [{"date": "2026-08-23", "brier": 0.2400, "games": 90}],
        }
        path = pl._model_monitor_json(
            {"auc": 0.55}, drift_df, "20260824", rolling_brier=rb,
        )
        mon = json.loads(Path(path).read_text())
        self.assertEqual(mon["rolling_brier"], rb["series"])
        self.assertEqual(mon["brier_baseline"], 0.2451)
        self.assertEqual(mon["brier_baseline_label"], "History mean (4001 games)")
        self.assertEqual(mon["rolling_brier_meta"]["window_days"], 30)
        self.assertEqual(mon["rolling_brier_meta"]["excluded_sparse_days"], 3)

    def test_monitor_json_empty_state_is_clean(self):
        import pipeline as pl

        drift_df = pd.DataFrame(columns=["status"])
        path = pl._model_monitor_json({"auc": 0.55}, drift_df, "20260824",
                                      rolling_brier=None)
        mon = json.loads(Path(path).read_text())
        self.assertEqual(mon["rolling_brier"], [])
        self.assertIsNone(mon["brier_baseline"])
        self.assertEqual(mon["rolling_brier_meta"], {})

    def test_end_to_end_compute_then_embed(self):
        """compute_rolling_brier output feeds _model_monitor_json unchanged."""
        import pipeline as pl

        rb = compute_rolling_brier(_fixture(), "20260824", calibrator=None)
        path = pl._model_monitor_json({}, pd.DataFrame(), "20260824",
                                      rolling_brier=rb)
        mon = json.loads(Path(path).read_text())
        self.assertEqual(mon["rolling_brier"], rb["series"])
        self.assertGreater(len(mon["rolling_brier"]), 0)


if __name__ == "__main__":
    unittest.main()
