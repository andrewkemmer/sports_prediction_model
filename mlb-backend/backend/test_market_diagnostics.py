"""Tests for the Markets-page diagnostics module (frontend/market_diagnostics.py).

Pure-function fixtures: offset logit interpolation, calibration binning,
favored-side bucket accuracy, totals distribution fit-check, empty/undecided
inputs (no NaN, loud warning), probability bounds — plus render smoke tests
proving each of the six chart builders produces a non-empty altair chart.
Read-only over artifacts; no model/config/metric changes.
"""
from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# frontend/ moved to the repository root (multi-sport restructure, Phase B)
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

import market_diagnostics as diag  # noqa: E402

DATA_DELIVERY = Path(__file__).resolve().parents[1] / "data_delivery"


def _latest_markets_artifact() -> Path:
    """Newest committed run_engine_markets_*.csv (never the *_rl bridge copy
    — the canonical file is what the frontend fetches). Skips gracefully
    when no artifact is present locally."""
    cands = sorted(DATA_DELIVERY.glob("run_engine_markets_*.csv"),
                   reverse=True)
    for c in cands:
        if "_rl." not in c.name:
            return c
    return Path()


def make_grid_df(n=50, lam_total=9.0, seed=0):
    """Artifact-shaped frame: full totals grid around a smooth curve."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        exp_t = float(np.clip(rng.normal(lam_total, 0.8), 5.5, 13.0))
        row = {"game_pk": i, "kind": "oof",
               "home_expected_runs": exp_t / 2, "away_expected_runs": exp_t / 2,
               "alpha_home": 0.27, "alpha_away": 0.35}
        # Monotone-decreasing toy grid centered near p=0.5 at the game's own
        # expected total. p_under mirrors p_over (denom = 1, no push band) so
        # the fair-line re-scaled 2-way confidence is well-defined in tests.
        for g in diag.TOTAL_GRID:
            shift = (exp_t - g) * 0.25
            po = float(1 / (1 + math.exp(-4 * shift)))
            row[f"p_over_{str(g).replace('.', '_')}"] = po
            row[f"p_under_{str(g).replace('.', '_')}"] = 1.0 - po
        rows.append(row)
    return pd.DataFrame(rows)


def add_outcomes(df, seed=1):
    rng = np.random.default_rng(seed)
    out = df.copy()
    # Sample totals from the toy model's own central probability at the
    # projected line so outcomes are consistent, not random noise.
    p = out[[f"p_over_{str(g).replace('.', '_')}" for g in diag.TOTAL_GRID]].mean(axis=1)
    base = np.round(out["home_expected_runs"] + out["away_expected_runs"])
    over = rng.random(len(out)) < p
    out["total_runs"] = np.where(over, base + 2, np.maximum(base - 3, 0))
    out["total_runs"] = out["total_runs"].astype(int)
    out["home_score"] = (out["total_runs"] // 2).astype(int)
    out["away_score"] = (out["total_runs"] - out["home_score"]).astype(int)
    # Run-line market column (independent of the totals grid).
    out["p_home_cover_1_5"] = rng.uniform(0.35, 0.68, len(out))
    return out


class TestOffsetInterpolation(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({"game_pk": [0], "kind": ["oof"]})
        # Flat 0.5 grid, then hand-set two adjacent columns:
        # logit(0.5)=0, logit(0.731058)=1.0
        for g in diag.TOTAL_GRID:
            df[f"p_over_{str(g).replace('.', '_')}"] = 0.5
        df["p_over_8_5"] = 0.5
        df["p_over_9_0"] = 0.7310585786300049
        self.df = df

    def test_logit_midpoint_is_exact(self):
        got = diag.over_prob_at_lines(self.df, np.array([8.75]))
        self.assertAlmostEqual(float(got[0]), 1 / (1 + math.exp(-0.5)), places=6)

    def test_interpolation_monotone_in_line(self):
        lines = np.array([8.5, 8.6, 8.75, 8.9, 9.0])
        p = diag.over_prob_at_lines(self.df, lines)
        self.assertTrue((np.diff(p) > 0).all())

    def test_clamps_outside_grid(self):
        lines = np.array([2.0, 20.0])
        p = diag.over_prob_at_lines(self.df, lines)
        # Both clamp to edge COLUMN pairs; this fixture is flat there → 0.5.
        self.assertAlmostEqual(float(p[0]), 0.5, places=9)
        self.assertAlmostEqual(float(p[1]), 0.5, places=9)

    def test_missing_columns_raise_loudly(self):
        with self.assertRaises(ValueError):
            diag.over_prob_at_lines(pd.DataFrame({"a": [1]}), np.array([8.5]))


class TestCalibrationBinning(unittest.TestCase):
    def test_perfect_calibration_hand_fixture(self):
        centers = (np.arange(20) + 0.5) / 20          # one per equal-width bin
        pairs = pd.DataFrame({"p": centers,
                              "y": (centers > 0.5).astype(float)})
        out = diag.calibration_curve(pairs, n_bins=20, min_count=1)
        self.assertEqual(out["warning"], None)
        self.assertEqual(out["n_pairs"], 20)
        self.assertEqual(len(out["bins"]), 20)
        for b in out["bins"]:
            self.assertEqual(b["count"], 1)
            self.assertAlmostEqual(b["bin_center"], b["mean_pred"], places=3)
            self.assertIn(b["mean_actual"], (0.0, 1.0))

    def test_min_count_drops_sparse_bins(self):
        pairs = pd.DataFrame({"p": [0.1, 0.9], "y": [0.0, 1.0]})
        out = diag.calibration_curve(pairs, n_bins=20, min_count=30)
        self.assertEqual(out["bins"], [])
        self.assertEqual(out["n_dropped_bins"], 20)
        self.assertIsNotNone(out["warning"])

    def test_empty_pairs_warning(self):
        out = diag.calibration_curve(pd.DataFrame(columns=["p", "y"]))
        self.assertEqual(out["bins"], [])
        self.assertIsNotNone(out["warning"])


class TestRelativizedPairs(unittest.TestCase):
    def test_spread_and_bounds_on_synthetic(self):
        decided = add_outcomes(make_grid_df(n=200))
        pairs = diag.relativized_pairs(decided)
        self.assertEqual(len(pairs), 200 * len(diag.OFFSET_EDGES))
        self.assertTrue(((pairs["p"] >= 0) & (pairs["p"] <= 1)).all())
        self.assertFalse(pairs.isna().any().any())
        # The whole point: pooled predictions must span a wide range.
        self.assertLess(pairs["p"].min(), 0.15)
        self.assertGreater(pairs["p"].max(), 0.85)

    def test_offset_sign_flips_probability_direction(self):
        decided = add_outcomes(make_grid_df(n=30))
        neg = diag.relativized_pairs(decided, offsets=[-1.0])
        pos = diag.relativized_pairs(decided, offsets=[+1.0])
        self.assertEqual(len(neg), 30)
        self.assertEqual(len(pos), 30)
        self.assertGreater(neg["p"].mean(), pos["p"].mean(),
                           "lower line must carry HIGHER p_over")

    def test_all_undecided_no_nan(self):
        df = make_grid_df(n=10)     # no total_runs column at all
        pairs = diag.relativized_pairs(df)
        self.assertEqual(len(pairs), 0)


class TestPickBuckets(unittest.TestCase):
    def test_hand_computed_buckets(self):
        p = np.array([0.52, 0.57, 0.62, 0.77])
        hit = np.array([1, 0, 1, 1], float)
        out = diag.pick_buckets(p, hit)
        by_label = {b["bucket"]: b for b in out["buckets"]}
        self.assertEqual(by_label["50-55"]["count"], 1)
        self.assertAlmostEqual(by_label["50-55"]["accuracy"], 100.0)
        self.assertEqual(by_label["55-60"]["count"], 1)
        self.assertAlmostEqual(by_label["55-60"]["accuracy"], 0.0)
        self.assertEqual(by_label["60-65"]["accuracy"], 100.0)
        self.assertEqual(by_label["65-70"]["count"], 0)
        self.assertIsNone(by_label["65-70"]["accuracy"])
        self.assertAlmostEqual(by_label["75+"]["accuracy"], 100.0)

    def test_under_favored_side_counts_as_pick(self):
        # p_over=0.45 → pick UNDER with confidence 0.55; under hits when the
        # total stays under — encoded via favored-side max(p, 1-p).
        p = np.array([0.45, 0.44])
        hit = np.array([1.0, 0.0])       # first under hit, second missed
        out = diag.pick_buckets(np.maximum(p, 1 - p), hit)
        b = out["buckets"][1]            # 55-60 bucket holds both
        self.assertEqual(b["count"], 2)
        self.assertAlmostEqual(b["accuracy"], 50.0)

    def test_nan_input_filtered(self):
        out = diag.pick_buckets(np.array([0.6, np.nan]), np.array([1.0, 0.0]))
        self.assertEqual(sum(b["count"] for b in out["buckets"]), 1)

    def test_empty_input_warns(self):
        out = diag.pick_buckets(np.array([]), np.array([]))
        self.assertIsNotNone(out["warning"])
        self.assertEqual(out["buckets"], [])


class TestRoundToHalfAndGrid(unittest.TestCase):
    """Explicit rounding rule: nearest 0.5, ties round half up."""

    def test_rounding_rule(self):
        cases = [(9.3, 9.5), (9.4, 9.5), (9.6, 9.5), (9.8, 10.0),
                 (8.25, 8.5), (8.0, 8.0), (8.75, 9.0), (9.2709, 9.5)]
        for raw, expected in cases:
            self.assertEqual(diag.round_to_half(raw), expected,
                             f"round_to_half({raw}) != {expected}")

    def test_half_tie_rounds_up(self):
        # 8.25*2 = 16.5 → ties away from zero → 17 → 8.5 (NOT banker's 8.0)
        self.assertEqual(diag.round_to_half(8.25), 8.5)

    def test_clamp_to_grid_edges(self):
        self.assertEqual(diag.clamp_to_grid(6.0), (6.5, True))
        self.assertEqual(diag.clamp_to_grid(13.0), (12.5, True))
        self.assertEqual(diag.clamp_to_grid(9.5), (9.5, False))

    def test_grid_column_names(self):
        self.assertEqual(diag.grid_over_under_cols(9.5),
                         ("p_over_9_5", "p_under_9_5"))
        self.assertEqual(diag.grid_over_under_cols(10.0),
                         ("p_over_10_0", "p_under_10_0"))


class TestRoundedTotalPairs(unittest.TestCase):
    def test_lines_are_own_rounded_totals(self):
        df = make_grid_df(n=40)
        df["total_runs"] = 9
        stats = diag.push_stats(df)
        pairs = diag.rounded_total_pairs(df)
        # One pair per NON-PUSH game (some games round to 9.0 == total → push)
        self.assertEqual(len(pairs), stats["n_games"] - stats["n_pushes"])
        exp = (df["home_expected_runs"] + df["away_expected_runs"])
        want_all = [diag.round_to_half(v) for v in exp]
        mask = (df["total_runs"].to_numpy(float) != np.array(want_all))
        want = [w for w, keep in zip(want_all, mask) if keep]
        self.assertEqual(pairs["line"].tolist(), want)
        self.assertTrue(((pairs["p"] >= 0) & (pairs["p"] <= 1)).all())
        self.assertTrue(pairs["y"].isin((0.0, 1.0)).all())

    def test_outcome_is_strictly_over_line(self):
        df = pd.DataFrame({
            "game_pk": [0, 1], "kind": ["oof", "oof"],
            "home_expected_runs": [4.5, 4.6], "away_expected_runs": [4.5, 4.4],
            "total_runs": [10, 8],
        })
        for g in diag.TOTAL_GRID:
            df[f"p_over_{str(g).replace('.', '_')}"] = 0.5
        # Both round to 9.0; over at 9.0 requires total >= 9.5 (10+)
        pairs = diag.rounded_total_pairs(df)
        self.assertEqual(pairs["line"].tolist(), [9.0, 9.0])
        self.assertEqual(pairs["y"].tolist(), [1.0, 0.0])

    def test_missing_grid_column_rows_skipped(self):
        # With NO p_under columns the fair 2-way grid can't be priced, so
        # rounded_total_pairs falls back to the rounded-to-half line; a game
        # whose rounded line column is missing is then SKIPPED (not dropped
        # silently or fabricated).
        rows = []
        for i in range(5):
            exp_t = 9.0 + 0.5 * i   # 9.0, 9.5, 10.0, 10.5, 11.0
            rows.append({"game_pk": i, "kind": "oof",
                         "home_expected_runs": exp_t / 2,
                         "away_expected_runs": exp_t / 2,
                         "total_runs": int(exp_t) + (2 if i % 2 == 0 else 0)})
        for r in rows:
            for g in diag.TOTAL_GRID:
                shift = (r["home_expected_runs"] + r["away_expected_runs"]
                         - g) * 0.25
                r[f"p_over_{str(g).replace('.', '_')}"] = float(
                    1 / (1 + math.exp(-4 * shift)))
        df = pd.DataFrame(rows).drop(columns=["p_over_11_0"])
        pairs = diag.rounded_total_pairs(df)
        # Only the 11.0 game (rounded line column dropped) is skipped
        self.assertEqual(len(pairs), 4)
        self.assertTrue((pairs["line"] != 11.0).all())

    def test_empty_input_loud_warning(self):
        pairs = diag.rounded_total_pairs(pd.DataFrame())
        self.assertEqual(len(pairs), 0)
        pairs = diag.rounded_total_pairs(
            make_grid_df(n=5).drop(columns=["home_expected_runs"]))
        self.assertEqual(len(pairs), 0)


class TestGameTotalCurveMoneylineGrammar(unittest.TestCase):
    """The Game Total Lines tab is now a single moneyline-style chart
    (``chart_game_total_curve``): continuous probability x, left 'Games' count
    bars, right '%' observed + win-rate curves, dashed perfect-calibration
    diagonal and amber pooled marker rolled into ONE chart (no separate
    scatter), and each axis title on exactly one layer — the same grammar as
    the moneyline Calibration Curve page."""

    @staticmethod
    def _decided():
        rows = []
        for k in range(10):            # bin 40-45 (n=10 < LOW_N=30)
            over = k < 4
            rows.append({"game_pk": k, "total_runs": 9 if over else 8,
                         "p_over_8_5": 0.42, "p_under_8_5": 0.58})
        for k in range(50):            # bin 55-60 (n=50 >= LOW_N)
            over = k < 29
            rows.append({"game_pk": 100 + k, "total_runs": 10 if over else 8,
                         "p_over_8_5": 0.57, "p_under_8_5": 0.43})
        return pd.DataFrame(rows)

    def _built(self, line=8.5, title="Calibration Curve — Over 8.5"):
        out = diag.game_total_calibration(self._decided(), line)
        return out, diag.chart_game_total_curve(out, title)

    def test_returns_chart_and_table_no_scatter(self):
        out, built = self._built()
        self.assertIsNone(out["warning"])
        self.assertEqual(sorted(built.keys()), ["chart", "table"])
        self.assertNotIn("scatter", built,
                         "no separate observed-vs-predicted scatter")
        self.assertFalse(built["table"].empty)
        self.assertEqual(built["table"].iloc[-1]["bin"], "Total")
        self.assertEqual(built["table"].iloc[-1]["share_pct"], 100.0)

    def test_continuous_probability_x_not_categorical(self):
        _, built = self._built()
        d = _spec_dump(built["chart"])
        # Continuous bin-center x on a probability scale, not categorical bins.
        self.assertIn('"bin_center"', d)
        # bin_center is the x field used on the continuous probability scale
        # (never a categorical bin axis).
        self.assertIn('"field": "bin_center"', d)
        # Both axes: left count ('Games') + right pct (%).
        self.assertIn('"field": "count"', d)
        self.assertIn('"field": "pct"', d)
        # Diagonal + amber pooled marker + series legend.
        self.assertIn("#64748B", d)
        self.assertIn("#F59E0B", d)
        self.assertIn('"shape": "diamond"', d)
        self.assertIn("#8B5CF6", d)

    def test_static_no_zoom_selection(self):
        """The chart is static again - no zoom/pan selection in the spec."""
        _, built = self._built()
        spec = built["chart"].to_dict()
        self.assertIsNone(spec.get("params"),
                          "no interactive()/zoom selection expected")
        self.assertNotIn("interactive", _spec_dump(built["chart"]))
        self.assertNotIn('"bind": "scales"', _spec_dump(built["chart"]))

    @staticmethod
    def _gtb(center, count=40, low_n=False, mean=None, observed=0.5):
        return {"bin_center": center, "bin": f"{int(round(center * 100))}",
                "count": count,
                "mean_pred": (mean if mean is not None else center),
                "observed": observed, "win_rate": observed,
                "low_n": low_n}

    @staticmethod
    def _gtable(bins, **kw):
        return {"bins": bins, "pooled_pred": kw.get("pooled_pred", 0.5),
                "pooled_observed": kw.get("pooled_observed", 0.5),
                "pooled_winrate": 0.5, "pooled_ece": 0.02,
                "pooled_brier": 0.25}

    def test_fixed_domain_0_25_0_75(self):
        # x-axis domain is FIXED at [0.25, 0.75] regardless of the data.
        t = self._gtable([self._gtb(0.50), self._gtb(0.51), self._gtb(0.52)])
        built = diag.chart_game_total_curve(t, "t")
        self.assertIn('"domain": [0.25, 0.75]', _spec_dump(built["chart"]))

    def test_low_n_or_empty_data_still_fixed_domain_no_error(self):
        # Even a low-n-only / degenerate table renders with the fixed domain.
        t = self._gtable([self._gtb(0.50, count=10, low_n=True)])
        built = diag.chart_game_total_curve(t, "t")
        d = _spec_dump(built["chart"])
        self.assertIn('"domain": [0.25, 0.75]', d)
        self.assertNotIn("NaN", d)

    def test_all_and_fixed_line_share_same_fixed_domain(self):
        # No adaptive domain: all selections emit the identical [0.25, 0.75].
        all_t = self._gtable([self._gtb(0.50), self._gtb(0.51)])
        fixed_t = self._gtable([self._gtb(0.10), self._gtb(0.90)])
        da = _spec_dump(diag.chart_game_total_curve(all_t, "t")["chart"])
        df = _spec_dump(diag.chart_game_total_curve(fixed_t, "t")["chart"])
        self.assertIn('"domain": [0.25, 0.75]', da)
        self.assertIn('"domain": [0.25, 0.75]', df)

    def test_out_of_domain_bins_clipped_no_error(self):
        # A low-n bin outside the fixed domain renders clipped, no error.
        t = self._gtable([self._gtb(0.50), self._gtb(0.51),
                          self._gtb(0.90, count=12, low_n=True)])
        built = diag.chart_game_total_curve(t, "t")
        d = _spec_dump(built["chart"])
        self.assertIn('"domain": [0.25, 0.75]', d)
        self.assertNotIn("NaN", d)
        self.assertGreater(len(built["chart"].to_dict()), 1)

    def test_each_axis_title_exactly_once(self):
        _, built = self._built()
        titles = []

        def _walk(node):
            if isinstance(node, dict):
                # Collect y-axis title if this is an encoding-bearing layer.
                y = node.get("encoding", {}).get("y")
                if isinstance(y, dict):
                    ax = y.get("axis")
                    t = ax.get("title") if isinstance(ax, dict) else None
                    if isinstance(t, str):
                        titles.append(t)
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(built["chart"].to_dict())
        self.assertEqual(titles.count("Observed % (2-way, no push)"), 1,
                         "right '%' axis title must appear exactly once")
        self.assertEqual(titles.count("Games"), 1,
                         "left 'Games' axis title must appear exactly once")

    def test_title_is_dynamic(self):
        _, built = self._built(
            title="Calibration Curve — Over (All = own fair line)")
        # The spec dump JSON-escapes the em-dash (\u2014); match ASCII-safe
        # substrings so we never depend on the dump's escaping.
        self.assertIn("Calibration Curve \\u2014 Over (All = own fair line)",
                      _spec_dump(built["chart"]))
        _, built8 = self._built(line=8.5, title="Calibration Curve — Over 8.5")
        self.assertIn("Over 8.5", _spec_dump(built8["chart"]))

    def test_hover_metadata_and_low_n_suppression(self):
        _, built = self._built()
        d = _spec_dump(built["chart"])
        # Hover carries games / mean predicted / observed / win rate.
        self.assertIn('"Games"', d)
        self.assertIn("Mean predicted", d)
        self.assertIn("Win rate", d)
        # low-n bin (40-45, n=10) rendered as a gray bar; its curve point
        # dropped; non-low-n (55-60) point kept.
        self.assertIn("#94A3B8", d)
        pts = diag._gtl_line_points(diag.game_total_calibration(
            self._decided(), 8.5))
        self.assertIn(0.575, list(pts["bin_center"]))
        self.assertNotIn(0.425, list(pts["bin_center"]))

    def test_all_branch_uses_own_line_and_renders(self):
        decided = add_outcomes(make_grid_df(n=200, seed=11))
        out = diag.game_total_calibration(decided, None)
        self.assertIsNone(out["warning"])
        built = diag.chart_game_total_curve(
            out, "Calibration Curve — Over (All = own fair line)")
        self.assertIn("chart", built)
        self.assertNotIn("scatter", built)
        self.assertGreater(len(built["chart"].to_dict()), 1)


class TestFixedLineCalibration(unittest.TestCase):
    """The Diagnostics fixed-line tab: ALL decided games priced at ONE line
    (default 8.5), predicted = re-scaled 2-way P(over), observed = over
    frequency on the same no-push basis, 5-pt buckets with empty bins kept."""

    def test_line_selector_changes_bucket_distribution(self):
        decided = add_outcomes(make_grid_df(n=200, seed=3))
        c85 = diag.game_total_calibration(decided, 8.5)
        c95 = diag.game_total_calibration(decided, 9.5)
        self.assertIsNone(c85["warning"])
        self.assertIsNone(c95["warning"])
        dist85 = [b["count"] for b in c85["bins"]]
        dist95 = [b["count"] for b in c95["bins"]]
        self.assertNotEqual(dist85, dist95,
                            "changing the line must re-bucket the games")
        self.assertNotAlmostEqual(c85["pooled_pred"], c95["pooled_pred"])

    def test_counts_sum_to_non_push_games_at_that_line(self):
        decided = add_outcomes(make_grid_df(n=150, seed=5))
        for line in (8.5, 9.0, 9.5):
            out = diag.game_total_calibration(decided, line)
            self.assertEqual(sum(b["count"] for b in out["bins"]),
                             out["n_games"] - out["n_pushes"],
                             "pushes excluded from the calibration population")
            self.assertEqual(out["n_games"], len(decided))
            self.assertEqual(len(out["bins"]), 20)   # 5-pt buckets, all kept

    def test_whole_line_pushes_excluded_2way_hand_computed(self):
        # Line 9.0: g0 total==9 is a PUSH (excluded from count AND observed);
        # g1/g3 over (10, 11), g2/g4 under (8, 7). pred2 = po/(po+pu).
        rows = [
            {"game_pk": 0, "total_runs": 9, "p_over_9_0": 0.45,
             "p_under_9_0": 0.45},
            {"game_pk": 1, "total_runs": 10, "p_over_9_0": 0.60,
             "p_under_9_0": 0.30},
            {"game_pk": 2, "total_runs": 8, "p_over_9_0": 0.55,
             "p_under_9_0": 0.40},
            {"game_pk": 3, "total_runs": 11, "p_over_9_0": 0.70,
             "p_under_9_0": 0.20},
            {"game_pk": 4, "total_runs": 7, "p_over_9_0": 0.50,
             "p_under_9_0": 0.45},
        ]
        out = diag.game_total_calibration(pd.DataFrame(rows), 9.0)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_games"], 5)
        self.assertEqual(out["n_pushes"], 1)
        self.assertAlmostEqual(out["push_rate"], 0.2)
        by = {b["bin"]: b for b in out["bins"]}
        # 50-55 holds ONLY g4 (pred2 0.5263, under) — the push game is
        # excluded from the count, observed and share denominator
        self.assertEqual(by["50-55"]["count"], 1)
        self.assertEqual(by["50-55"]["observed"], 0.0)
        self.assertAlmostEqual(by["50-55"]["mean_pred"], 0.5 / 0.95,
                               places=4)
        self.assertAlmostEqual(by["50-55"]["share_pct"], 25.0)  # 1/4 obs
        self.assertEqual(by["65-70"]["count"], 1)   # g1 over
        self.assertEqual(by["65-70"]["observed"], 1.0)
        self.assertEqual(by["75-80"]["count"], 1)   # g3 over
        self.assertEqual(by["75-80"]["observed"], 1.0)
        self.assertEqual(by["55-60"]["count"], 1)   # g2 under
        self.assertEqual(by["55-60"]["observed"], 0.0)
        # Pooled pred/observed over the 4 NON-PUSH games
        exp_pred = (0.6 / 0.9 + 0.55 / 0.95 + 0.7 / 0.9 + 0.5 / 0.95) / 4
        self.assertAlmostEqual(out["pooled_pred"], exp_pred, places=4)
        self.assertAlmostEqual(out["pooled_observed"], 0.5)

    def test_share_pct_hand_computed(self):
        # 3 games in 0-5, 1 in 5-10 -> shares 75% / 25%, sum 100
        rows = [{"game_pk": i, "total_runs": 8 if i % 2 else 10,
                 "p_over_8_5": p, "p_under_8_5": 1.0 - p}
                for i, p in enumerate((0.02, 0.04, 0.03, 0.06))]
        out = diag.game_total_calibration(pd.DataFrame(rows), 8.5)
        self.assertIsNone(out["warning"])
        by = {b["bin"]: b for b in out["bins"]}
        self.assertEqual(by["0-5"]["count"], 3)
        self.assertAlmostEqual(by["0-5"]["share_pct"], 75.0, places=2)
        self.assertEqual(by["5-10"]["count"], 1)
        self.assertAlmostEqual(by["5-10"]["share_pct"], 25.0, places=2)
        self.assertAlmostEqual(sum(b["share_pct"] for b in out["bins"]),
                               100.0, places=1)

    def test_half_line_never_pushes(self):
        decided = add_outcomes(make_grid_df(n=120, seed=7))
        for line in (6.5, 8.5, 9.5, 12.5):
            out = diag.game_total_calibration(decided, line)
            self.assertEqual(out["n_pushes"], 0,
                             f"half-line {line} cannot push")
            self.assertIsNone(out["warning"])

    def test_all_branch_uses_own_line_1pt_buckets(self):
        decided = add_outcomes(make_grid_df(n=200, seed=11))
        out = diag.game_total_calibration(decided, None)
        self.assertIsNone(out["warning"])
        self.assertEqual([b["bin"] for b in out["bins"]],
                         diag.OWN_LINE_LABELS)
        self.assertEqual(out["bins"][-1]["bin"], "60+")
        self.assertEqual(sum(b["count"] for b in out["bins"]),
                         out["n_games"] - out["n_pushes"])
        # Own-line re-scaled P(over) hugs 50% by construction (the fair
        # line is the 50/50 grid argmin) — never a wide spread
        self.assertGreaterEqual(out["pooled_pred"], 0.45)
        self.assertLessEqual(out["pooled_pred"], 0.6)
        self.assertGreaterEqual(out["pooled_observed"], 0.0)
        self.assertLessEqual(out["pooled_observed"], 1.0)

    def test_all_whole_line_fat_push_band_uses_rescaled_p_over(self):
        """The All branch prices the re-scaled P(over) = p_over/(p_over +
        p_under) — the SAME value the Today's Games card displays — never
        the picked-side max. p_over=0.36 / p_under=0.48 at own line 9.0
        → rso = 0.36/0.84 ≈ 0.4286 → 42-43 bucket (not a bogus 57-58)."""
        rows = [{"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
                 "away_expected_runs": 4.5, "total_runs": 8,
                 "p_over_9_0": 0.36, "p_under_9_0": 0.48}]
        out = diag.game_total_calibration(pd.DataFrame(rows), None)
        by = {b["bin"]: b for b in out["bins"]}
        self.assertEqual(by["42-43"]["count"], 1)
        self.assertEqual(by["57-58"]["count"], 0)
        self.assertAlmostEqual(out["pooled_pred"], 0.4286, places=4)  # 4dp rounded
        self.assertEqual(out["pooled_observed"], 0.0)  # 8 < 9.5 → under

    def test_total_row_present_and_weighted_hand_computed(self):
        """The bin table ends with a 'Total' summary row: count = sum of
        bucket counts (empty buckets contribute 0), mean_pred / observed =
        the pooled count-weighted values, share_pct = 100.00."""
        rows = [
            {"game_pk": 0, "total_runs": 8, "p_over_9_0": 0.50,
             "p_under_9_0": 0.50},
            {"game_pk": 1, "total_runs": 10, "p_over_9_0": 0.50,
             "p_under_9_0": 0.50},
            {"game_pk": 2, "total_runs": 10, "p_over_9_0": 0.80,
             "p_under_9_0": 0.20},
            # PUSH: total 9 == whole line 9.0 → excluded from both sides
            {"game_pk": 3, "total_runs": 9, "p_over_9_0": 0.80,
             "p_under_9_0": 0.20},
        ]
        out = diag.game_total_calibration(pd.DataFrame(rows), 9.0)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_pushes"], 1)
        built = diag.chart_game_total_curve(out, "t")
        tab = built["table"]
        self.assertEqual(tab.iloc[-1]["bin"], "Total")
        self.assertEqual(len(tab), len(out["bins"]) + 1)  # bins + Total
        tot = tab.iloc[-1]
        self.assertEqual(tot["count"], 3)          # 2 + 1 non-push games
        self.assertAlmostEqual(tot["mean_pred"], 0.6, places=4)   # (0.5+0.5+0.8)/3
        self.assertAlmostEqual(tot["observed"], 0.6667, places=4)  # 2 over / 3, 4dp
        self.assertEqual(tot["share_pct"], 100.0)
        self.assertAlmostEqual(tot["mean_pred"], out["pooled_pred"], places=9)
        self.assertAlmostEqual(tot["observed"], out["pooled_observed"], places=9)
        # Empty buckets contribute 0 to the Total count
        self.assertEqual(tot["count"],
                         sum(b["count"] for b in out["bins"]))

    def test_total_row_empty_buckets_contribute_zero(self):
        """With only one populated bin, the Total row still equals the sum
        (the 19 empty buckets add nothing) and renders without error."""
        rows = [{"game_pk": i, "total_runs": 10 if i % 2 else 8,
                 "p_over_9_0": 0.52, "p_under_9_0": 0.47}
                for i in range(60)]
        out = diag.game_total_calibration(pd.DataFrame(rows), 9.0)
        built = diag.chart_game_total_curve(out, "t")
        tab = built["table"]
        self.assertEqual(tab.iloc[-1]["bin"], "Total")
        self.assertEqual(tab.iloc[-1]["count"],
                         sum(b["count"] for b in out["bins"]))
        self.assertAlmostEqual(tab.iloc[-1]["share_pct"], 100.0, places=6)

    def test_empty_bins_kept_and_chart_builds(self):
        # All pred2 cluster in 0.50-0.55 -> only one populated bin; the rest
        # keep count 0 / None stats and both charts still render.
        rows = [{"game_pk": i, "total_runs": 10 if i % 2 else 8,
                 "p_over_9_0": 0.52, "p_under_9_0": 0.47}
                for i in range(60)]
        out = diag.game_total_calibration(pd.DataFrame(rows), 9.0)
        self.assertEqual(out["n_pushes"], 0)
        populated = [b for b in out["bins"] if b["count"] > 0]
        self.assertEqual(len(populated), 1)
        empty = [b for b in out["bins"] if b["count"] == 0]
        self.assertEqual(len(empty), 19)
        for b in empty:
            self.assertIsNone(b["observed"])
            self.assertIsNone(b["mean_pred"])
            self.assertEqual(b["share_pct"], 0.0)
        built = diag.chart_game_total_curve(out, "t")
        for k in ("chart", "table"):
            self.assertIn(k, built)
        self.assertNotIn("scatter", built, "no separate scatter chart")
        self.assertFalse(built["table"].empty)
        self.assertGreater(len(built["chart"].to_dict()), 1)

    def test_missing_columns_warn_not_crash(self):
        out = diag.game_total_calibration(pd.DataFrame(), 8.5)
        self.assertIsNotNone(out["warning"])
        out = diag.game_total_calibration(
            make_grid_df(n=5).drop(columns=["p_under_8_5"]), 8.5)
        self.assertIsNotNone(out["warning"])
        self.assertEqual(out["bins"], [])
        out = diag.game_total_calibration(
            make_grid_df(n=5).drop(columns=["home_expected_runs"]), None)
        self.assertIsNotNone(out["warning"])

    def test_real_artifact_fixed_line_spread_and_sum(self):
        """On the shipped artifact a single line spreads predicted P(over)
        widely (the point of the view); buckets + shares sum over the
        non-push population."""
        m_path = _latest_markets_artifact()
        if not m_path.exists():
            self.skipTest("run-engine artifact absent")
        out = diag.game_total_calibration(
            diag.decided_rows(pd.read_csv(m_path)), 8.5)
        self.assertIsNone(out["warning"])
        self.assertEqual(sum(b["count"] for b in out["bins"]),
                         out["n_games"] - out["n_pushes"])
        self.assertAlmostEqual(sum(b["share_pct"] for b in out["bins"]),
                               100.0, places=1)
        mn = min(b["mean_pred"] for b in out["bins"] if b["mean_pred"])
        mx = max(b["mean_pred"] for b in out["bins"] if b["mean_pred"])
        self.assertGreater(mx - mn, 0.20,
                           "fixed-line predicted spread must be wide")
        self.assertLess(mn, 0.40)
        self.assertGreater(mx, 0.55)

    def test_real_artifact_all_55plus_empty(self):
        """The 'All' own-line view on the shipped artifact: predicted is the
        re-scaled P(over) (card basis), so the band spans BOTH sides of
        50% (47-48 … 52-53) and 53+ is empty; buckets sum to the non-push
        population; pooled pred/obs match the card's pooled calibration
        (0.500/0.505)."""
        m_path = _latest_markets_artifact()
        if not m_path.exists():
            self.skipTest("run-engine artifact absent")
        out = diag.game_total_calibration(
            diag.decided_rows(pd.read_csv(m_path)), None)
        self.assertIsNone(out["warning"])
        by = {b["bin"]: b for b in out["bins"]}
        self.assertEqual(sum(b["count"] for b in out["bins"]),
                         out["n_games"] - out["n_pushes"])
        # rso semantics: populated buckets on BOTH sides of 50%
        self.assertGreater(by["47-48"]["count"], 0)
        self.assertGreater(by["48-49"]["count"], 0)
        self.assertGreater(by["51-52"]["count"], 0)
        self.assertGreater(by["52-53"]["count"], 0)
        hi = sum(by[k]["count"] for k in
                 ("53-54", "54-55", "55-56", "56-57", "57-58",
                  "58-59", "59-60", "60+"))
        self.assertEqual(hi, 0)
        self.assertAlmostEqual(out["pooled_pred"], 0.500, places=2)
        self.assertAlmostEqual(out["pooled_observed"], 0.505, places=2)


class TestAllViewMatchesCardDefaultLine(unittest.TestCase):
    """PIN: the 'All' (own-line) view's per-game predicted P(over) is the
    SAME value the Today's Games card displays at its default line —
    re-scaled 2-way p_over/(p_over+p_under) at the fair line. Verified
    identical on the shipped artifact (30/30 + hand-picked, max |diff| 0.0);
    this locks the equivalence so the two code paths (fair_total_lines in
    the diagnostics branch vs fair_total_line_row in run_engine_card_bits)
    can never drift."""

    def _slate_map(self, decided):
        return {str(r["game_pk"]): r for _, r in decided.iterrows()}

    def _assert_game_matches_card(self, decided):
        slate = self._slate_map(decided)
        fair_lines = diag.fair_total_lines(decided)
        for i, (_, row) in enumerate(decided.iterrows()):
            pk = str(row["game_pk"])
            bits = diag.run_engine_card_bits(pk, slate, line=None)
            self.assertIsNotNone(bits)
            fair = diag.fair_total_line_row(row)
            self.assertIsNotNone(fair)
            self.assertEqual(bits["total_line"], fair,
                             f"card default line == fair line for {pk}")
            over_col, under_col = diag.grid_over_under_cols(fair)
            po = float(row[over_col])
            pu = float(row[under_col])
            rso = po / (po + pu)
            self.assertAlmostEqual(bits["p_over"], rso, places=6,
                                   msg=f"card Over% == All-view pred for {pk}")
            # All branch resolves the identical fair line
            self.assertEqual(fair_lines[i], fair,
                             f"All branch line == fair line for {pk}")

    def test_hand_picked_whole_half_and_push_band_rows(self):
        rows = [
            # whole-number fair line with a FAT push band (p_over+p_under
            # < 1) — the push-fold path: rso = 0.36/0.84 ≈ 0.4286
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 8,
             "p_over_9_0": 0.36, "p_under_9_0": 0.48,
             "p_over_8_5": 0.45, "p_push_9_0": 0.09,
             "p_home_cover_1_5": 0.5},
            # half-line fair line, no push band (denom = 1)
            {"game_pk": 2, "kind": "oof", "home_expected_runs": 4.2,
             "away_expected_runs": 4.3, "total_runs": 9,
             "p_over_8_5": 0.53, "p_under_8_5": 0.47,
             "p_home_cover_1_5": 0.5},
            # half-line fair line below 50%
            {"game_pk": 3, "kind": "oof", "home_expected_runs": 5.4,
             "away_expected_runs": 5.1, "total_runs": 11,
             "p_over_10_5": 0.49, "p_under_10_5": 0.51,
             "p_home_cover_1_5": 0.5},
            # grid-edge fair line (upper boundary taken verbatim)
            {"game_pk": 4, "kind": "oof", "home_expected_runs": 6.4,
             "away_expected_runs": 6.1, "total_runs": 13,
             "p_over_12_5": 0.51, "p_under_12_5": 0.49,
             "p_home_cover_1_5": 0.5},
        ]
        self._assert_game_matches_card(pd.DataFrame(rows))

    def test_synthetic_grid_30_games_whole_and_half(self):
        """make_grid_df prices every line with p_under mirrors (denom = 1,
        no push band) — 30 games spanning whole AND half fair lines must
        all match the card at the default line."""
        decided = add_outcomes(make_grid_df(n=30, seed=5))
        self._assert_game_matches_card(decided)
        # Sanity: the fixture actually exercises both line kinds
        fairs = diag.fair_total_lines(decided)
        self.assertTrue(any(f == round(f) for f in fairs))
        self.assertTrue(any(f != round(f) for f in fairs))


class TestPushExclusion(unittest.TestCase):
    """Whole-number-line pushes (total == rounded line) are neither wins
    nor losses — excluded from win rates, reported in push_rate. The card's
    Over/Under display is untouched (still exactly two, summing to 1)."""

    def _decided(self, rows):
        return pd.DataFrame(rows)

    def test_all_branch_excludes_push(self):
        rows = [
            # PUSH: total 9 == own fair line 9.0 (4.5 + 4.5)
            {"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 9, "p_over_9_0": 0.52,
             "p_under_9_0": 0.47},
            # over pick wins: 10 >= 9.5
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 10, "p_over_9_0": 0.52,
             "p_under_9_0": 0.47},
            # over pick loses: 8 < 9.5
            {"game_pk": 2, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 8, "p_over_9_0": 0.52,
             "p_under_9_0": 0.47},
        ]
        out = diag.game_total_calibration(self._decided(rows), None)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_games"], 3)
        self.assertEqual(out["n_pushes"], 1)
        self.assertAlmostEqual(out["push_rate"], 1 / 3, places=4)
        # Push excluded from the calibration population (2 non-push games)
        self.assertEqual(sum(b["count"] for b in out["bins"]), 2)
        self.assertAlmostEqual(out["pooled_observed"], 0.5)  # 1 win / 2

    def test_pairs_exclude_push_rows(self):
        rows = [
            {"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 9, "p_over_9_0": 0.52},
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 10, "p_over_9_0": 0.52},
        ]
        pairs = diag.rounded_total_pairs(self._decided(rows))
        self.assertEqual(len(pairs), 1)             # push row dropped
        self.assertEqual(pairs.iloc[0]["y"], 1.0)
        self.assertEqual(pairs.iloc[0]["line"], 9.0)

    def test_half_line_never_pushes(self):
        # Line 9.5 cannot push: integer totals never equal 9.5. Total 9 at
        # line 9.5 is an OVER MISS (9 < 10), correctly counted — not a push.
        rows = [{"game_pk": 0, "kind": "oof", "home_expected_runs": 4.7,
                 "away_expected_runs": 4.8, "total_runs": 9, "p_over_9_5": 0.62,
                 "p_under_9_5": 0.31}]
        out = diag.game_total_calibration(self._decided(rows), 9.5)
        self.assertEqual(out["n_pushes"], 0)
        self.assertEqual(out["n_games"], 1)
        self.assertAlmostEqual(out["pooled_observed"], 0.0)  # over pick missed

    def test_push_stats_helper(self):
        rows = [
            {"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 9},     # push
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 10},
            {"game_pk": 2, "kind": "oof", "home_expected_runs": 4.7,
             "away_expected_runs": 4.8, "total_runs": 10},    # line 9.5
        ]
        stats = diag.push_stats(self._decided(rows))
        self.assertEqual(stats["n_games"], 3)
        self.assertEqual(stats["n_pushes"], 1)
        self.assertAlmostEqual(stats["push_rate"], 1 / 3, places=4)
        # Empty input → zeros, no crash
        self.assertEqual(diag.push_stats(pd.DataFrame())["n_pushes"], 0)


class TestOversAndRunlineTables(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 120
        p85 = rng.uniform(0.35, 0.72, n)
        pc15 = rng.uniform(0.35, 0.68, n)
        total = np.where(rng.random(n) < p85, 10, 7)
        diff = np.where(rng.random(n) < pc15, 3, -2)
        self.decided = pd.DataFrame({
            "kind": "oof", "total_runs": total,
            "home_score": np.maximum(diff, 0), "away_score": np.maximum(-diff, 0),
            "p_over_8_5": p85, "p_home_cover_1_5": pc15,
            "home_expected_runs": 4.6, "away_expected_runs": 4.4,
            "alpha_home": 0.27, "alpha_away": 0.35,
        })

    def test_overs_table_shape_and_rule(self):
        out = diag.overs_pick_table(self.decided, line=8.5)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_games"], 120)
        self.assertEqual(sum(b["count"] for b in out["buckets"]), 120)
        # Recompute one bucket by hand from raw arrays.
        p = self.decided["p_over_8_5"].to_numpy()
        hit = ((p >= 0.5).astype(float)
               == (self.decided["total_runs"].to_numpy() >= 9).astype(float)).astype(float)
        pct = np.maximum(p, 1 - p) * 100
        m = (pct >= 50) & (pct < 55)
        row = next(b for b in out["buckets"] if b["bucket"] == "50-55")
        self.assertEqual(row["count"], int(m.sum()))

    def test_runline_table_symmetric_side(self):
        out = diag.runline_pick_table(self.decided)
        self.assertIsNone(out["warning"])
        self.assertEqual(sum(b["count"] for b in out["buckets"]), 120)
        self.assertIn("away +1.5", out["pick_rule"])

    def test_missing_columns_warn_not_crash(self):
        out = diag.overs_pick_table(self.decided.drop(columns=["p_over_8_5"]))
        self.assertIsNotNone(out["warning"])
        out = diag.runline_pick_table(self.decided.drop(columns=["home_score"]))
        self.assertIsNotNone(out["warning"])


class TestTotalDistribution(unittest.TestCase):
    def test_callouts_present_and_mass_conserved(self):
        decided = add_outcomes(make_grid_df(n=150))
        dist = diag.total_distribution(decided)
        self.assertIsNone(dist["warning"])
        self.assertEqual(len(dist["ks"]), 16)
        self.assertEqual(len(dist["observed"]), 16)
        obs_sum = sum(dist["observed"])
        self.assertLessEqual(obs_sum, 1.0 + 1e-9)
        self.assertIn("P(total<=1)", dist["callouts"])
        self.assertIn("P(total>=10)", dist["callouts"])
        for v in dist["modeled"]:
            self.assertGreaterEqual(v, 0.0)

    def test_empty_input_loud_warning(self):
        dist = diag.total_distribution(pd.DataFrame())
        self.assertIsNotNone(dist["warning"])
        self.assertEqual(dist["observed"], [])


class TestRealArtifactPushSmoke(unittest.TestCase):
    """Read-only smoke over the shipped OOF artifact: pushes detected,
    excluded from win rates, reported in push_rate."""

    def test_real_oof_push_stats_and_win_rate(self):
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        m_path = dd / "run_engine_markets_20260824.csv"
        if not m_path.exists():
            self.skipTest("local run-engine artifact absent in this workspace")
        markets = pd.read_csv(m_path)
        decided = diag.decided_rows(markets)
        self.assertGreater(len(decided), 4000)
        stats = diag.push_stats(decided)
        self.assertEqual(stats["n_games"], len(decided))
        # Whole-number lines are common (9.0), so pushes must exist on the
        # real artifact — and win rate must exclude them.
        self.assertGreater(stats["n_pushes"], 0)
        self.assertLess(stats["push_rate"], 0.10)
        tp = diag.game_total_calibration(decided, None)
        self.assertIsNone(tp["warning"])
        self.assertEqual(tp["n_pushes"], stats["n_pushes"])
        self.assertEqual(tp["n_games"], stats["n_games"])
        self.assertAlmostEqual(tp["push_rate"], stats["push_rate"], places=4)
        # Pushes excluded from the calibration population (2-way convention)
        self.assertEqual(sum(b["count"] for b in tp["bins"]),
                         tp["n_games"] - tp["n_pushes"])
        # Pairs: every push excluded, one pair per non-push game.
        pairs = diag.rounded_total_pairs(decided)
        self.assertEqual(len(pairs), stats["n_games"] - stats["n_pushes"])


def _spec_dump(chart) -> str:
    import json
    return json.dumps(chart.to_dict())


class TestAccuracyAxisAlwaysIndependent(unittest.TestCase):
    """The accuracy line must ALWAYS get its own independent right-side
    y-axis with a real scale/domain — compiled-spec assertions, no
    Streamlit. A regression passed scale=None on the default path, which
    serializes to "scale": null in vega-lite ("disable the scale, drop the
    axis"), so the Run-line picks line rendered on the count scale with no
    accuracy axis."""

    @classmethod
    def setUpClass(cls):
        cls.decided = add_outcomes(make_grid_df(n=120))

    @staticmethod
    def _units(spec):
        out = []

        def walk(layers):
            for ly in layers:
                if "layer" in ly:
                    walk(ly["layer"])
                else:
                    out.append(ly)
        walk(spec.get("layer", []))
        return out

    def _accuracy_unit(self, built):
        spec = built["chart"].to_dict()
        units = self._units(spec)
        acc = [u for u in units
               if u.get("encoding", {}).get("y", {}).get("field") == "accuracy"]
        self.assertEqual(len(acc), 1, "exactly one accuracy unit")
        return spec, acc[0]["encoding"]["y"]

    def test_default_build_two_independent_scales_accuracy_field(self):
        built = diag.chart_pick_buckets(
            diag.runline_pick_table(self.decided), "Run-line picks")
        spec, y = self._accuracy_unit(built)
        # TWO independent y-scales: count (bars) + accuracy (line).
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")
        y_fields = [u["encoding"]["y"]["field"] for u in self._units(spec)
                    if "y" in u.get("encoding", {})]
        self.assertIn("count", y_fields)
        self.assertIn("accuracy", y_fields)
        # The line plots ACCURACY (not count) with a REAL scale — never
        # "scale": null.
        self.assertEqual(y["field"], "accuracy")
        self.assertIsInstance(y.get("scale"), dict)
        self.assertIsNotNone(y["scale"].get("domain"))

    def test_default_build_accuracy_domain_covers_bucket_range(self):
        built = diag.chart_pick_buckets(
            diag.runline_pick_table(self.decided), "Run-line picks")
        _, y = self._accuracy_unit(built)
        domain = y["scale"]["domain"]
        table_acc = built["table"]["accuracy"].dropna().tolist()
        self.assertGreaterEqual(domain[1], max(table_acc),
                                "accuracy axis must not clip the buckets")
        self.assertGreaterEqual(domain[1], 90.0)   # full 0–100% scale

    def test_no_null_scale_in_any_build(self):
        table = diag.pick_buckets(
            np.array([0.52, 0.54, 0.57, 0.51, 0.55, 0.53] * 20),
            np.array([1, 0, 1, 1, 0, 1] * 20, float),
            labels=diag.OWN_LINE_LABELS, edges=diag.OWN_LINE_EDGES)
        for kwargs in ({}, {"total_line": True, "acc_y_max": 75.0}):
            built = diag.chart_pick_buckets(table, "t", **kwargs)
            dump = _spec_dump(built["chart"])
            self.assertNotIn('"scale": null', dump,
                             f"null scale leaks into spec with {kwargs}")


class TestRealArtifactRunlineAxis(unittest.TestCase):
    """On the shipped artifact, the Run-line picks accuracy line must rise
    across confidence buckets (52.7 → 93.8 on the refreshed 20260825
    artifact; was 50.0 → 92.9 before the pipeline's 2026-08-26 re-run) and
    sit on its own independent right-side accuracy axis — the regression
    rendered it on the count scale with no accuracy axis."""

    def test_runline_line_rises_with_own_axis(self):
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        m_path = dd / "run_engine_markets_20260825.csv"
        if not m_path.exists():
            self.skipTest("local run-engine artifact absent")
        decided = diag.decided_rows(pd.read_csv(m_path))
        rp = diag.runline_pick_table(decided)
        accs = [b["accuracy"] for b in rp["buckets"]]
        # The table's accuracy rises ~52.7 → ~93.8 across buckets (exact
        # bucket-0 value tracks the artifact snapshot — this is a data
        # assertion, not the chart contract under test).
        self.assertAlmostEqual(accs[0], 52.7, places=1)
        self.assertGreaterEqual(accs[-1], 92.0)
        self.assertEqual(accs, sorted(accs), "line must rise, not fall")
        # The chart must plot that accuracy on its own independent axis
        # with a domain that includes the bucket range.
        built = diag.chart_pick_buckets(rp, "Run-line picks at −1.5")
        spec = built["chart"].to_dict()
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")
        units = []

        def walk(layers):
            for ly in layers:
                if "layer" in ly:
                    walk(ly["layer"])
                else:
                    units.append(ly)
        walk(spec.get("layer", []))
        acc_y = [u["encoding"]["y"] for u in units
                 if u.get("encoding", {}).get("y", {}).get("field") == "accuracy"]
        self.assertEqual(len(acc_y), 1)
        self.assertEqual(acc_y[0]["field"], "accuracy")
        self.assertIsInstance(acc_y[0].get("scale"), dict)
        domain = acc_y[0]["scale"]["domain"]
        self.assertGreaterEqual(domain[1], max(accs))


class TestRenderSmoke(unittest.TestCase):
    """Each chart builder returns a NON-EMPTY altair object for a fixture."""

    @classmethod
    def setUpClass(cls):
        import altair as alt
        cls.alt = alt
        cls.decided = add_outcomes(make_grid_df(n=120))

    def _assert_chart_has_data(self, chart):
        # Layered charts carry Undefined .data — validate the compiled spec.
        spec = chart.to_dict()
        self.assertIsInstance(spec, dict)
        self.assertTrue(len(spec) > 1, f"chart spec unexpectedly empty: {spec}")

    def test_1_distribution_renders(self):
        dist = diag.total_distribution(self.decided)
        self._assert_chart_has_data(diag.chart_distribution(dist))

    def test_2_relativized_renders_with_real_spread(self):
        curve = diag.calibration_curve(
            diag.relativized_pairs(self.decided), n_bins=20, min_count=10)
        self.assertGreaterEqual(len(curve["bins"]), 5)
        xs = [b["mean_pred"] for b in curve["bins"]]
        self.assertLess(min(xs), 0.25)
        self.assertGreater(max(xs), 0.75)
        self._assert_chart_has_data(diag.chart_calibration(curve, "t"))

    def test_3_pooled_fixed_lines_renders(self):
        curve = diag.calibration_curve(
            diag.fixed_line_pairs(self.decided, (7.5, 8.5, 9.5, 10.5)),
            n_bins=20, min_count=10)
        self._assert_chart_has_data(diag.chart_calibration(curve, "t"))

    def test_4_money_line_renders_narrow_blob(self):
        rel = diag.calibration_curve(
            diag.relativized_pairs(self.decided), n_bins=20, min_count=10)
        xs_rel = [b["mean_pred"] for b in rel["bins"]]
        m85c2 = diag.calibration_curve(
            diag.fixed_line_pairs(self.decided, (8.5,)), n_bins=20,
            min_count=10)
        xs85 = [b["mean_pred"] for b in m85c2["bins"]]
        # The money-line blob must be strictly NARROWER than relativized.
        self.assertLess(max(xs85) - min(xs85), max(xs_rel) - min(xs_rel))
        chart = diag.chart_calibration(m85c2, "t", x_domain=[0.30, 0.70])
        self._assert_chart_has_data(chart)

    def test_4b_rounded_money_line_renders_one_pair_per_game(self):
        curve = diag.calibration_curve(
            diag.rounded_total_pairs(self.decided), n_bins=20, min_count=10)
        self.assertEqual(curve["n_pairs"], len(self.decided))
        self.assertGreaterEqual(len(curve["bins"]), 2)
        self._assert_chart_has_data(
            diag.chart_calibration(curve, "Per-game rounded total"))

    @staticmethod
    def _synthetic_own_line_table():
        """Synthetic pick-buckets table (chart_pick_buckets contract) for
        the total_line/axis tests — the deleted Totals picks tab no longer
        provides a real fixture."""
        p = np.array([0.52, 0.54, 0.57, 0.51, 0.55, 0.53] * 20)
        hit = np.array([1, 0, 1, 1, 0, 1] * 20, float)
        table = diag.pick_buckets(p, hit, labels=diag.OWN_LINE_LABELS,
                                  edges=diag.OWN_LINE_EDGES)
        table["n_games"] = len(p)
        table["win_rate"] = round(float(hit.mean()), 4)
        return table

    def test_7_game_total_lines_renders(self):
        for line in (None, 8.5):
            out = diag.game_total_calibration(self.decided, line)
            self.assertIsNone(out["warning"])
            built = diag.chart_game_total_curve(out, "t")
            self._assert_chart_has_data(built["chart"])
            self.assertFalse(built["table"].empty)
            # New table contract: share_pct replaces the redundant observed_pct
            self.assertIn("share_pct", built["table"].columns)
            self.assertNotIn("observed_pct", built["table"].columns)

    def test_pick_buckets_total_line_at_pooled_rate(self):
        import json
        tp = self._synthetic_own_line_table()
        built = diag.chart_pick_buckets(
            tp, "Pick accuracy", total_line=True)
        dump = json.dumps(built["chart"].to_dict())
        # Constant horizontal line at the POOLED win rate, not a bucket point
        self.assertIn('"rule"', dump)
        self.assertIn(f'"y": {tp["win_rate"] * 100}', dump)
        # Labeled with n + pooled rate
        self.assertIn(
            f"Total (n={tp['n_games']:,}): {tp['win_rate'] * 100:.1f}%",
            dump)

    def test_accuracy_axis_floor_not_pinned_to_top(self):
        import json
        tp = self._synthetic_own_line_table()
        built = diag.chart_pick_buckets(
            tp, "Pick accuracy", total_line=True, acc_y_max=75.0)
        dump = json.dumps(built["chart"].to_dict())
        # Accuracy axis is in PERCENT units; domain max >= 75 (the floor),
        # extended with headroom above the largest visible value.
        data_max = max(b["accuracy"] for b in tp["buckets"]
                       if b["accuracy"] is not None)
        y_max = max(75.0, max(data_max, tp["win_rate"] * 100) + 5.0)
        self.assertGreaterEqual(y_max, 75.0)
        self.assertIn(f'"domain": [0.0, {y_max}]', dump)
        # The pooled-rate rule sits INSIDE the visible domain
        self.assertLess(tp["win_rate"] * 100, y_max)
        self.assertIn(f'"y": {tp["win_rate"] * 100}', dump)

    def test_accuracy_axis_never_clips_real_values(self):
        import json
        # Five picks at 0.80 confidence, 4/5 hit → a genuine 80% bucket
        p = np.full(5, 0.8)
        hit = np.array([1, 1, 1, 1, 0], float)
        table = diag.pick_buckets(p, hit, labels=diag.OWN_LINE_LABELS,
                                  edges=diag.OWN_LINE_EDGES)
        table["n_games"] = 5
        table["win_rate"] = 0.8
        built = diag.chart_pick_buckets(
            table, "t", total_line=True, acc_y_max=75.0)
        dump = json.dumps(built["chart"].to_dict())
        # max(75, 80 + 5) = 85 — the 80% value is NOT clipped
        self.assertIn('"domain": [0.0, 85.0]', dump)
        self.assertIn('"y": 80.0', dump)

    def test_5_overs_picks_renders(self):
        built = diag.chart_pick_buckets(
            diag.overs_pick_table(self.decided, 8.5), "Overs")
        self._assert_chart_has_data(built["chart"])
        self.assertFalse(built["table"].empty)

    def test_6_runline_picks_renders(self):
        built = diag.chart_pick_buckets(
            diag.runline_pick_table(self.decided), "Run line")
        self._assert_chart_has_data(built["chart"])
        self.assertFalse(built["table"].empty)




class TestTotalsCalibrationMetrics(unittest.TestCase):
    """Win rate + ECE/Brier + low-n on the totals calibration chart (fixed
    line AND All own-line views). Mirrors the moneyline calibration card:
    pick over if P(over) > 50% else under; win rate = W/(W+L) per bin on the
    no-push 2-way basis (a 'V' around 50%; bins below 50% give 1 - observed);
    ECE = |mean_pred - observed|; Brier = mean((pred - outcome)^2). low_n
    (n < LOW_N) bins are flagged and their chart points suppressed."""

    @staticmethod
    def _metrics_decided():
        """Line 8.5 (half-line, no pushes). bin 40-45: 10 games @ 0.42,
        4 over/6 under. bin 55-60: 50 games @ 0.57, 29 over/21 under."""
        rows = []
        for k in range(10):
            over = k < 4
            rows.append({"game_pk": len(rows), "kind": "oof",
                         "home_expected_runs": 4.2, "away_expected_runs": 4.3,
                         "total_runs": 9 if over else 8,
                         "p_over_8_5": 0.42, "p_under_8_5": 0.58})
        for k in range(50):
            over = k < 29
            rows.append({"game_pk": len(rows), "kind": "oof",
                         "home_expected_runs": 4.2, "away_expected_runs": 4.3,
                         "total_runs": 10 if over else 8,
                         "p_over_8_5": 0.57, "p_under_8_5": 0.43})
        return pd.DataFrame(rows)

    def test_win_rate_line_hand_computed_v_shape(self):
        out = diag.game_total_calibration(self._metrics_decided(), 8.5)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_pushes"], 0)          # half-line never pushes
        by = {b["bin"]: b for b in out["bins"]}
        # 40-45 (pred 0.42 < 0.5 -> UNDER pick): win rate = 1 - observed 0.40
        self.assertEqual(by["40-45"]["count"], 10)
        self.assertAlmostEqual(by["40-45"]["observed"], 0.40, places=4)
        self.assertAlmostEqual(by["40-45"]["win_rate"], 0.60, places=4)
        # 55-60 (pred 0.57 > 0.5 -> OVER pick): win rate = observed 0.58
        self.assertEqual(by["55-60"]["count"], 50)
        self.assertAlmostEqual(by["55-60"]["observed"], 0.58, places=4)
        self.assertAlmostEqual(by["55-60"]["win_rate"], 0.58, places=4)

    def test_per_bin_ece_and_brier_hand_computed(self):
        out = diag.game_total_calibration(self._metrics_decided(), 8.5)
        by = {b["bin"]: b for b in out["bins"]}
        b40 = by["40-45"]
        self.assertAlmostEqual(b40["mean_pred"], 0.42, places=4)
        self.assertAlmostEqual(b40["ece"], 0.02, places=4)       # |0.42-0.40|
        # Brier over 10 games (4 over @(0.42-1)^2, 6 under @0.42^2) = 2.404/10
        self.assertAlmostEqual(b40["brier"], 0.2404, places=4)
        b55 = by["55-60"]
        self.assertAlmostEqual(b55["ece"], 0.01, places=4)       # |0.57-0.58|
        # (29*0.1849 + 21*0.3249)/50 = 12.185/50
        self.assertAlmostEqual(b55["brier"], 0.2437, places=4)

    def test_pooled_aggregates_weighted(self):
        out = diag.game_total_calibration(self._metrics_decided(), 8.5)
        self.assertAlmostEqual(out["pooled_pred"], (4.2 + 28.5) / 60, places=4)
        self.assertAlmostEqual(out["pooled_observed"], 33 / 60, places=4)
        # wins = 6 (under bin) + 29 (over bin) = 35 / 60
        self.assertAlmostEqual(out["pooled_winrate"], 35 / 60, places=4)
        # pooled ECE = (10/60)*0.02 + (50/60)*0.01, rounded to 4dp
        self.assertAlmostEqual(
            out["pooled_ece"],
            round((10 / 60) * 0.02 + (50 / 60) * 0.01, 4), places=9)
        # pooled Brier = mean of squared errors over all pairs, rounded 4dp
        po = np.array([0.42] * 10 + [0.57] * 50)
        ev = np.array([1.0] * 4 + [0.0] * 6 + [1.0] * 29 + [0.0] * 21)
        self.assertAlmostEqual(
            out["pooled_brier"],
            round(float(((po - ev) ** 2).mean()), 4), places=9)

    def test_low_n_bins_flagged_and_points_suppressed(self):
        out = diag.game_total_calibration(self._metrics_decided(), 8.5)
        built = diag.chart_game_total_curve(out, "t")
        tab = built["table"]
        row40 = tab[tab["bin"] == "40-45"].iloc[0]
        row55 = tab[tab["bin"] == "55-60"].iloc[0]
        self.assertTrue(bool(row40["low_n"]))    # n=10 < 30
        self.assertFalse(bool(row55["low_n"]))   # n=50 >= 30
        # Line/point spec keeps ONLY non-low-n populated bins (55-60); the
        # low-n bin (0.42) is dropped from the win-rate/observed points.
        dump = _spec_dump(built["chart"])
        self.assertIn("Observed", dump)    # curves present
        self.assertGreater(len(built["chart"].to_dict()), 1)

    def test_all_branch_flat_own_line_win_rate_renders(self):
        decided = add_outcomes(make_grid_df(n=200, seed=11))
        out = diag.game_total_calibration(decided, None)
        self.assertIsNone(out["warning"])
        populated = [b for b in out["bins"] if b["count"] > 0]
        self.assertTrue(populated)
        for b in populated:
            self.assertIsNotNone(b["win_rate"])
            self.assertGreaterEqual(b["win_rate"], 0.0)
            self.assertLessEqual(b["win_rate"], 1.0)
        # Flat own-line win rate (the documented finding, not a bug)
        self.assertGreaterEqual(out["pooled_winrate"], 0.35)
        self.assertLessEqual(out["pooled_winrate"], 0.65)
        built = diag.chart_game_total_curve(out, "t")
        self.assertGreater(len(built["chart"].to_dict()), 1)

    def test_empty_bins_render_zero_none_no_error(self):
        # All preds cluster in 0.50-0.55 at line 9.0 -> one populated bin.
        rows = [{"game_pk": i, "total_runs": 10 if i % 2 else 8,
                 "p_over_9_0": 0.52, "p_under_9_0": 0.47}
                for i in range(60)]
        out = diag.game_total_calibration(pd.DataFrame(rows), 9.0)
        built = diag.chart_game_total_curve(out, "t")
        empty = [b for b in out["bins"] if b["count"] == 0]
        self.assertTrue(empty)
        for b in empty:
            self.assertIsNone(b["observed"])
            self.assertIsNone(b["mean_pred"])
            self.assertIsNone(b["win_rate"])
            self.assertIsNone(b["ece"])
            self.assertIsNone(b["brier"])
            self.assertFalse(b["low_n"])
            self.assertEqual(b["share_pct"], 0.0)
        for k in ("chart", "table"):
            self.assertIn(k, built)
        self.assertNotIn("scatter", built, "no separate scatter view")
        self.assertFalse(built["table"].empty)
        # table carries the new columns
        for col in ("win_rate", "ece", "brier", "low_n"):
            self.assertIn(col, built["table"].columns)




class TestGameTotalLinesDiagnosticsAppTest(unittest.TestCase):
    """End-to-end through frontend/markets.py (the Diagnostics page): the
    'Game Total Lines' tab (All own-line / fixed line) renders the merged
    moneyline-style calibration chart + pooled table with 0 exceptions, on
    the current committed artifact. Runs in a SUBPROCESS so the canonical
    suite's streamlit stubs (swapped by test_frontend_markets) cannot poison
    the real Streamlit run."""

    def test_diagnostics_game_total_lines_tab_renders(self):
        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "from streamlit.testing.v1 import AppTest;\n"
            "at = AppTest.from_file(%r, default_timeout=60);\n"
            "at.run();\n"
            "labels = [t.proto.label for t in at.tabs];\n"
            "assert 'Game Total Lines' in labels, labels;\n"
            "idx = labels.index('Game Total Lines');\n"
            "at.tabs[idx].run();\n"
            "assert not at.exception, at.exception;\n"
            "assert len(at.caption) > 0, 'no captions rendered';\n"
            "# Smoke the line selector at All, 8.0 and 8.5 (the reported 8.0\n"
            "# degenerate-domain strip must not appear in any).\n"
            "for _val in ['All', '8.0', '8.5']:\n"
            "    at.tabs[idx].selectbox[0].set_value(_val);\n"
            "    at.tabs[idx].run();\n"
            "    assert not at.exception, (_val, at.exception);\n"
            "print('DIAG_GT_OK')\n"
        ) % (str(FRONTEND), str(FRONTEND / "markets.py"))
        res = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(
            res.returncode, 0,
            f"AppTest subprocess failed:\nSTDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr[-2000:]}")
        self.assertIn("DIAG_GT_OK", res.stdout)


class TestResolveSlateAcrossArtifacts(unittest.TestCase):
    """Run-engine slate resolution across dated run_engine_markets frames.

    ``resolve_slate_across_artifacts`` keys a card's run-engine lookup by
    game_pk across the available dated artifacts (instead of the exact
    game-date file alone): exact-date wins when it holds the id; otherwise
    the newest artifact with an EXACT game_pk; otherwise a same-matchup row
    (the GMT-rollover case where the Aug 30 run priced the Aug 29 evening
    game under the next-day prefix). Absent ids never fabricate a row.
    """

    @staticmethod
    def _frame(*rows):
        return pd.DataFrame(rows)

    @staticmethod
    def _slate(game_pk, proj=4.5):
        return {"game_pk": game_pk, "kind": "slate",
                "home_expected_runs": proj, "away_expected_runs": 4.0}

    def test_game_only_in_newer_artifact_resolves_rollover(self):
        # Aug 29 card; the Aug 29 file was deleted; the Aug 30 run priced the
        # SAME evening game under the next-day prefix (GMT rollover).
        f29 = self._frame()
        f30 = self._frame(self._slate("20260830_BAL@ATH", 4.44))
        res = diag.resolve_slate_across_artifacts(
            {"20260829": f29, "20260830": f30}, ["20260829_BAL@ATH"])
        self.assertIn("20260829_BAL@ATH", res,
                      "rollover card must resolve via the newer artifact")
        self.assertEqual(res["20260829_BAL@ATH"]["game_pk"], "20260830_BAL@ATH")

    def test_exact_date_file_wins_over_newer(self):
        # Both the exact-date file and a newer run price the id: exact wins.
        f29 = self._frame(self._slate("20260829_BAL@ATH", 4.1))
        f30 = self._frame(self._slate("20260830_BAL@ATH", 4.44))
        res = diag.resolve_slate_across_artifacts(
            {"20260829": f29, "20260830": f30}, ["20260829_BAL@ATH"])
        self.assertEqual(res["20260829_BAL@ATH"]["home_expected_runs"], 4.1)

    def test_absent_everywhere_graceful_no_crash(self):
        res = diag.resolve_slate_across_artifacts(
            {"20260830": self._frame(self._slate("20260830_MIA@WSH"))},
            ["20260829_ARI@SF_2_2"])
        self.assertEqual(res, {}, "unresolvable id -> absent (fallback)")

    def test_newest_first_tiebreak_when_multiple_have_id(self):
        # No exact-date file; two artifacts carry the same EXACT id -> the
        # newest wins (tier b newest-first).
        f28 = self._frame(self._slate("20260830_BAL@ATH", 4.0))
        f31 = self._frame(self._slate("20260830_BAL@ATH", 5.0))
        res = diag.resolve_slate_across_artifacts(
            {"20260828": f28, "20260831": f31}, ["20260830_BAL@ATH"])
        self.assertEqual(res["20260830_BAL@ATH"]["home_expected_runs"], 5.0)

    def test_later_run_same_id_general_case(self):
        # A game whose EXACT id lives only in a LATER-dated artifact (a game
        # on Aug 30 priced by the Aug 31 run) resolves that row.
        f30 = self._frame()
        f31 = self._frame(self._slate("20260830_MIA@WSH", 4.38))
        res = diag.resolve_slate_across_artifacts(
            {"20260830": f30, "20260831": f31}, ["20260830_MIA@WSH"])
        self.assertEqual(res["20260830_MIA@WSH"]["home_expected_runs"], 4.38)

    def test_present_row_is_the_raw_slate_record(self):
        # When the exact-date file holds the id, the map carries the RAW row
        # unchanged -- render is byte-identical to current behavior.
        row = self._slate("20260829_PHI@LAA", 4.36)
        f29 = self._frame(row)
        res = diag.resolve_slate_across_artifacts(
            {"20260829": f29}, ["20260829_PHI@LAA"])
        self.assertEqual(res["20260829_PHI@LAA"], row)

    def test_invalid_and_blank_game_pk_fall_back(self):
        f30 = self._frame(self._slate("20260830_MIA@WSH"))
        res = diag.resolve_slate_across_artifacts(
            {"20260830": f30}, ["", "nonsense"])
        self.assertEqual(res, {}, "blank/non-date ids never fabricate")

    def test_ood_rows_and_non_slate_ignored(self):
        # kind != 'slate' (OOF rows) and NaN game_pk never match a card id.
        oof = pd.DataFrame([{"game_pk": 778485, "kind": "oof",
                             "home_expected_runs": 4.5}])
        res = diag.resolve_slate_across_artifacts(
            {"20260830": oof}, ["778485", "20260830_MIA@WSH"])
        self.assertEqual(res, {})
        # Matchup helper strips the date prefix only for dated ESPN ids.
        self.assertEqual(diag._espy_matchup("20260830_BAL@ATH"), "BAL@ATH")
        self.assertEqual(diag._espy_matchup("20260829_ARI@SF_2_2"), "ARI@SF_2_2")
        self.assertEqual(diag._espy_matchup("778485"), "778485")


class TestGameTotalFixedXDomain(unittest.TestCase):
    """The Game Total Lines chart uses a FIXED x-domain of [0.25, 0.75].

    Replaces the adaptive/dynamic domain (bdf477d, b4a0562) entirely: the
    domain is constant for ALL selections, so the degenerate-domain failure
    class (NaN / reversed / collapsed strip) cannot recur. Assertions target
    the emitted spec: the x-domain is exactly [0.25, 0.75] regardless of the
    selected line, and nothing NaN leaks into the spec.
    """

    @staticmethod
    def _b(center, count=40, mean=None, observed=0.5):
        return {"bin_center": center, "bin": f"{int(round(center * 100))}",
                "count": count,
                "mean_pred": center if mean is None else mean,
                "observed": observed, "win_rate": observed,
                "low_n": count < 30 and count > 0}

    def test_fixture_domain_exactly_fixed(self):
        built = diag.chart_game_total_curve({"bins": [self._b(0.50)]}, "t")
        self.assertIn('"domain": [0.25, 0.75]', _spec_dump(built["chart"]))

    def test_low_n_and_nan_do_not_change_fixed_domain(self):
        # Even a degenerate bin (NaN mean_pred) keeps the fixed domain -- no
        # min/max/padding over the data, so nothing can drive it off-course.
        bins = [self._b(0.5, mean=float("nan")), self._b(0.9, count=10)]
        built = diag.chart_game_total_curve({"bins": bins}, "t")
        d = _spec_dump(built["chart"])
        self.assertIn('"domain": [0.25, 0.75]', d)
        self.assertNotIn("NaN", d)

    def test_empty_bins_render_no_error(self):
        # Empty bin table: the guarded early-return still yields a chart.
        built = diag.chart_game_total_curve({"bins": []}, "t")
        self.assertIn("chart", built)
        self.assertNotIn("NaN", _spec_dump(built["chart"]))

    def test_real_artifact_fixed_domain_all_lines(self):
        # The fixed domain holds at every selection on the real artifact.
        m_path = _latest_markets_artifact()
        if not m_path.exists():
            self.skipTest("run-engine artifact absent")
        decided = diag.decided_rows(pd.read_csv(m_path))
        for line in (None, 8.0, 8.5):   # None = 'All'
            out = diag.game_total_calibration(decided, line)
            built = diag.chart_game_total_curve(out, "t")
            d = _spec_dump(built["chart"])
            self.assertIn('"domain": [0.25, 0.75]', d, f"line={line}")
            self.assertNotIn("NaN", d, f"line={line}")


if __name__ == "__main__":
    unittest.main()
