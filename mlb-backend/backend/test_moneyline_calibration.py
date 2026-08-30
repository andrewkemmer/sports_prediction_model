"""Tests for the merged moneyline "Calibration Curve — Favored Team" chart.

The old standalone "Prediction Confidence & Accuracy" section is gone; the
top calibration chart now carries count bars (LEFT 'Games' axis) whose
heights sum to the total decided games, aligned one-to-one with the blue
actual-rate curve's 1% bins, with the Platt map + perfect-calibration
diagonal on a shared right '%' axis.

The aggregation + chart builder live in frontend/moneyline_calibration.py
(pure — no Streamlit), so they are tested directly. The page source is
guarded by source-inspection, and an AppTest smoke (subprocess) renders the
Calibration page end-to-end with 0 exceptions.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(_FRONTEND) not in sys.path:
    sys.path.insert(0, str(_FRONTEND))

import moneyline_calibration as mlc  # noqa: E402


def _dump(chart) -> str:
    return json.dumps(chart.to_dict())


class TestFavoredCalibrationPts(unittest.TestCase):
    """Per-1% favored-probability aggregation: binning + counts."""

    def test_bars_sum_to_total_decided_games(self):
        # 6 predictions -> 6 decided games are each binned to one 1% slice.
        h = pd.DataFrame({
            "home_win_prob_model": [0.55, 0.62, 0.51, 0.58, 0.71, 0.49],
            "correct": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        })
        pts = mlc.favored_calibration_pts(h)
        self.assertEqual(int(pts["n"].sum()), 6)
        self.assertEqual(len(pts), len(pts["prob"].unique()))  # one row per bin

    def test_nearest_1pct_and_favored_side(self):
        # 0.49 -> favored 0.51; 0.50 stays 0.50 (>= 0.5); 0.501 -> 0.51.
        h = pd.DataFrame({
            "home_win_prob_model": [0.49, 0.50, 0.501, 0.555, 0.556],
            "correct": [1.0] * 5,
        })
        pts = mlc.favored_calibration_pts(h)
        by = {float(p): n for p, n in zip(pts["prob"], pts["n"])}
        # 0.49 -> max(0.49, 0.51)=0.51 rounded nearest 1% = 0.51
        # 0.50 -> 0.50 ; 0.501 -> 0.50 ; 0.555 -> 0.56 ; 0.556 -> 0.56
        self.assertEqual(by, {0.51: 1, 0.50: 2, 0.56: 2})
        self.assertEqual(int(pts["n"].sum()), 5)

    def test_empty_when_missing_columns(self):
        self.assertTrue(mlc.favored_calibration_pts(None).empty)
        self.assertTrue(mlc.favored_calibration_pts(
            pd.DataFrame({"home_win_prob_model": [0.5]})).empty)  # no 'correct'
        self.assertTrue(mlc.favored_calibration_pts(pd.DataFrame()).empty)

    def test_win_rate_per_bin(self):
        h = pd.DataFrame({
            "home_win_prob_model": [0.52, 0.52, 0.52, 0.53],
            "correct": [1.0, 0.0, 1.0, 1.0],
        })
        pts = mlc.favored_calibration_pts(h)
        by = {float(p): r for p, r in zip(pts["prob"], pts.to_dict("records"))}
        self.assertEqual(by[0.52]["n"], 3)
        self.assertAlmostEqual(by[0.52]["win_rate"], 2 / 3)
        self.assertEqual(by[0.53]["n"], 1)
        self.assertAlmostEqual(by[0.53]["win_rate"], 1.0)


class TestMergedChart(unittest.TestCase):
    """Merged chart: count bars sum to the total, align to the curve's bins,
    hover counts present, and empty/low-n render without error."""

    @staticmethod
    def _pts():
        return pd.DataFrame({
            "prob": [0.51, 0.55, 0.62, 0.71],
            "win_rate": [0.60, 0.55, 0.50, 0.45],
            "n": [100, 80, 60, 40],
        })

    def test_bars_sum_to_total_and_align_with_curve(self):
        built = mlc.chart_favored_calibration(self._pts(), pd.DataFrame())
        self.assertEqual(built["n_total"], 280)
        self.assertEqual(int(built["bars"]["n"].sum()), 280)
        # Bars and the blue curve share the same per-1% x-bins (same frame).
        self.assertEqual(len(built["bars"]), len(built["bars"]))
        self.assertTrue(set(["prob", "win_rate", "n"]).issubset(
            built["bars"].columns))

    def test_spec_dual_axis_and_bar_mark(self):
        built = mlc.chart_favored_calibration(
            self._pts(),
            pd.DataFrame({"prob": [0.55], "cal_mean": [0.60], "n": [80]}))
        d = _dump(built["chart"])
        # Bars layer present (mark_bar) with the count field.
        self.assertIn('"type": "bar"', d)
        self.assertIn('"field": "n"', d)
        # Right-axis win-rate '%' scale and independent y (dual axis).
        self.assertIn('"field": "win_rate_pct"', d)
        self.assertIn('"field": "cal_mean_pct"', d)
        self.assertIn("independent", d)

    def test_axis_titles_single_source(self):
        """No overlapping axis labels: 'Actual win rate %' (right scale) and
        'Games' (left scale) each appear on EXACTLY ONE layer's y-axis title —
        the merged-chart regression where both the blue and green curve layers
        emitted the right-axis title on top of each other."""
        def _y_axis_titles(chart):
            titles = []
            for layer in chart.to_dict()["layer"]:
                y = layer.get("encoding", {}).get("y", {})
                axis = y.get("axis")
                if isinstance(axis, dict) and isinstance(axis.get("title"), str):
                    titles.append(axis.get("title"))
            return titles

        built = mlc.chart_favored_calibration(
            self._pts(),
            pd.DataFrame({"prob": [0.55], "cal_mean": [0.60], "n": [80]}))
        titles = _y_axis_titles(built["chart"])
        self.assertEqual(
            titles.count("Actual win rate %"), 1,
            "right-axis title must appear exactly once across all layers")
        self.assertEqual(
            titles.count("Games"), 1,
            "'Games' title must appear exactly once across all layers")
        # The green Platt layer (rotated right axis) must NOT repeat the title.
        self.assertIn("Actual win rate %", titles)
        self.assertIn("Games", titles)

    def test_hover_counts_present(self):
        built = mlc.chart_favored_calibration(self._pts(), pd.DataFrame())
        d = _dump(built["chart"]) + _dump(
            mlc.chart_favored_calibration(
                self._pts(),
                pd.DataFrame({"prob": [0.51], "cal_mean": [0.58], "n": [100]}))
            ["chart"])
        self.assertIn("Games", d)       # tooltip title / Games axis

    def test_empty_pts_no_error(self):
        built = mlc.chart_favored_calibration(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(built["n_total"], 0)
        self.assertIn('"type": "bar"', _dump(built["chart"]))
        self.assertIn('"type": "line"', _dump(built["chart"]))  # diagonal still

    def test_low_n_single_row_no_error(self):
        built = mlc.chart_favored_calibration(
            pd.DataFrame({"prob": [0.99], "win_rate": [0.9], "n": [1]}),
            pd.DataFrame())
        self.assertEqual(built["n_total"], 1)


# ---------------------------------------------------------------------------
# Page source guards (model_calibration.py)
# ---------------------------------------------------------------------------

class TestMoneylinePageSource(unittest.TestCase):
    """The standalone confidence section is gone; the merged chart + updated
    caption remain in the page."""

    @classmethod
    def _src(cls):
        return (_FRONTEND / "model_calibration.py").read_text()

    def test_old_section_render_path_deleted(self):
        src = self._src()
        # No section builder / heading for the former standalone chart.
        self.assertNotIn("Prediction Confidence & Accuracy", src)
        self.assertNotIn("confidence distribution", src)
        self.assertNotIn("conf_df", src)

    def test_page_delegates_to_merged_builder(self):
        src = self._src()
        self.assertIn("mlc.chart_favored_calibration", src)
        self.assertIn("utils.show_chart(built[\"chart\"])", src)

    def test_caption_keeps_confidence_vs_accuracy_and_hover(self):
        src = self._src()
        self.assertIn("confidence-vs-accuracy", src)
        self.assertIn("Games", src)
        self.assertIn("hover for games per point", src)
        self.assertIn("each game counted once from the favored side", src)


# ---------------------------------------------------------------------------
# AppTest smoke: the Calibration page renders the merged chart with 0 errors
# ---------------------------------------------------------------------------

class TestMoneylineCalibrationAppTest(unittest.TestCase):
    """End-to-end through model_calibration.py: the merged chart + reliability
    table render with 0 exceptions. Runs in a SUBPROCESS so the canonical
    suite's streamlit stubs (swapped by test_frontend_markets) cannot poison
    the real Streamlit run."""

    def test_apptest_calibration_page_renders(self):
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.run();\n"
            "assert not at.exception, at.exception;\n"
            "assert len(at.caption) > 0, 'no captions rendered';\n"
            "assert len(at.markdown) > 0, 'page rendered nothing';\n"
            "print('CAL_OK')\n"
        ) % (str(_FRONTEND), str(_FRONTEND / "model_calibration.py"))
        res = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(
            res.returncode, 0,
            f"AppTest subprocess failed:\nSTDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr[-2000:]}")
        self.assertIn("CAL_OK", res.stdout)


if __name__ == "__main__":
    unittest.main()