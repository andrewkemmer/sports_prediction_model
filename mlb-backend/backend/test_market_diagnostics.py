"""Tests for the Markets-page diagnostics module (frontend/market_diagnostics.py).

Pure-function fixtures: offset logit interpolation, calibration binning,
favored-side bucket accuracy, totals distribution fit-check, empty/undecided
inputs (no NaN, loud warning), probability bounds — plus render smoke tests
proving each of the six chart builders produces a non-empty altair chart.
Read-only over artifacts; no model/config/metric changes.
"""
from __future__ import annotations

import math
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


class TestFixedLineCalibration(unittest.TestCase):
    """The Diagnostics fixed-line tab: ALL decided games priced at ONE line
    (default 8.5), predicted = re-scaled 2-way P(over), observed = over
    frequency on the same no-push basis, 5-pt buckets with empty bins kept."""

    def test_line_selector_changes_bucket_distribution(self):
        decided = add_outcomes(make_grid_df(n=200, seed=3))
        c85 = diag.fixed_line_calibration(decided, 8.5)
        c95 = diag.fixed_line_calibration(decided, 9.5)
        self.assertIsNone(c85["warning"])
        self.assertIsNone(c95["warning"])
        dist85 = [b["count"] for b in c85["bins"]]
        dist95 = [b["count"] for b in c95["bins"]]
        self.assertNotEqual(dist85, dist95,
                            "changing the line must re-bucket the games")
        self.assertNotAlmostEqual(c85["pooled_pred"], c95["pooled_pred"])

    def test_counts_sum_to_decided_games_at_that_line(self):
        decided = add_outcomes(make_grid_df(n=150, seed=5))
        for line in (8.5, 9.0, 9.5):
            out = diag.fixed_line_calibration(decided, line)
            self.assertEqual(sum(b["count"] for b in out["bins"]),
                             out["n_games"])
            self.assertEqual(out["n_games"], len(decided))
            self.assertEqual(len(out["bins"]), 20)   # 5-pt buckets, all kept

    def test_whole_line_pushes_excluded_2way_hand_computed(self):
        # Line 9.0: g0 total==9 is a PUSH (excluded from observed); g1/g3
        # over (10, 11), g2/g4 under (8, 7). pred2 = po/(po+pu).
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
        out = diag.fixed_line_calibration(pd.DataFrame(rows), 9.0)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_games"], 5)
        self.assertEqual(out["n_pushes"], 1)
        self.assertAlmostEqual(out["push_rate"], 0.2)
        by = {b["bin"]: b for b in out["bins"]}
        # 50-55 holds the push (pred2 0.5, observed None) + g4 (pred2
        # 0.5263, under) -> observed 0.0
        self.assertEqual(by["50-55"]["count"], 2)
        self.assertEqual(by["50-55"]["observed"], 0.0)
        self.assertAlmostEqual(by["50-55"]["mean_pred"],
                               (0.5 + 0.5 / 0.95) / 2, places=4)
        self.assertEqual(by["65-70"]["count"], 1)   # g1 over
        self.assertEqual(by["65-70"]["observed"], 1.0)
        self.assertEqual(by["75-80"]["count"], 1)   # g3 over
        self.assertEqual(by["75-80"]["observed"], 1.0)
        self.assertEqual(by["55-60"]["count"], 1)   # g2 under
        self.assertEqual(by["55-60"]["observed"], 0.0)
        # Pooled: pred over all 5 priced games; observed 2/4 excluding push
        exp_pred = (0.5 + 0.6 / 0.9 + 0.55 / 0.95 + 0.7 / 0.9
                    + 0.5 / 0.95) / 5
        self.assertAlmostEqual(out["pooled_pred"], exp_pred, places=4)
        self.assertAlmostEqual(out["pooled_observed"], 0.5)

    def test_half_line_never_pushes(self):
        decided = add_outcomes(make_grid_df(n=120, seed=7))
        for line in (6.5, 8.5, 9.5, 12.5):
            out = diag.fixed_line_calibration(decided, line)
            self.assertEqual(out["n_pushes"], 0,
                             f"half-line {line} cannot push")
            self.assertIsNone(out["warning"])

    def test_empty_bins_kept_and_chart_builds(self):
        # All pred2 cluster in 0.50-0.55 -> only one populated bin; the rest
        # keep count 0 / observed None and both charts still render.
        rows = [{"game_pk": i, "total_runs": 10 if i % 2 else 8,
                 "p_over_9_0": 0.52, "p_under_9_0": 0.47}
                for i in range(60)]
        out = diag.fixed_line_calibration(pd.DataFrame(rows), 9.0)
        self.assertEqual(out["n_pushes"], 0)
        populated = [b for b in out["bins"] if b["count"] > 0]
        self.assertEqual(len(populated), 1)
        empty = [b for b in out["bins"] if b["count"] == 0]
        self.assertEqual(len(empty), 19)
        for b in empty:
            self.assertIsNone(b["observed"])
            self.assertIsNone(b["mean_pred"])
        built = diag.chart_fixed_line(out, "t")
        for k in ("chart", "scatter", "table"):
            self.assertIn(k, built)
        self.assertFalse(built["table"].empty)
        self.assertGreater(len(built["chart"].to_dict()), 1)
        self.assertGreater(len(built["scatter"].to_dict()), 1)

    def test_missing_columns_warn_not_crash(self):
        out = diag.fixed_line_calibration(pd.DataFrame(), 8.5)
        self.assertIsNotNone(out["warning"])
        out = diag.fixed_line_calibration(
            make_grid_df(n=5).drop(columns=["p_under_8_5"]), 8.5)
        self.assertIsNotNone(out["warning"])
        self.assertEqual(out["bins"], [])

    def test_real_artifact_fixed_line_spread_and_sum(self):
        """On the shipped artifact a single line spreads predicted P(over)
        widely (the point of the view) and buckets sum to all decided games."""
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        m_path = dd / "run_engine_markets_20260829.csv"
        if not m_path.exists():
            self.skipTest("run-engine artifact absent")
        out = diag.fixed_line_calibration(
            diag.decided_rows(pd.read_csv(m_path)), 8.5)
        self.assertIsNone(out["warning"])
        self.assertEqual(sum(b["count"] for b in out["bins"]),
                         out["n_games"])
        mn = min(b["mean_pred"] for b in out["bins"] if b["mean_pred"])
        mx = max(b["mean_pred"] for b in out["bins"] if b["mean_pred"])
        self.assertGreater(mx - mn, 0.20,
                           "fixed-line predicted spread must be wide")
        self.assertLess(mn, 0.40)
        self.assertGreater(mx, 0.55)


class TestPushExclusion(unittest.TestCase):
    """Whole-number-line pushes (total == rounded line) are neither wins
    nor losses — excluded from win rates, reported in push_rate. The card's
    Over/Under display is untouched (still exactly two, summing to 1)."""

    def _decided(self, rows):
        return pd.DataFrame(rows)

    def test_pick_table_excludes_push(self):
        rows = [
            # PUSH: total 9 == fair/rounded line 9.0 (4.5 + 4.5)
            {"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 9, "p_over_9_0": 0.52,
             "p_under_9_0": 0.47},
            # over hit: 10 >= 9.5
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 10, "p_over_9_0": 0.52,
             "p_under_9_0": 0.47},
            # over miss: 8 < 9.5
            {"game_pk": 2, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 8, "p_over_9_0": 0.52,
             "p_under_9_0": 0.47},
        ]
        out = diag.totals_pick_table(self._decided(rows))
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_games"], 2)        # push dropped
        self.assertEqual(out["n_pushes"], 1)
        self.assertAlmostEqual(out["push_rate"], 1 / 3, places=4)
        self.assertAlmostEqual(out["win_rate"], 0.5)  # 1 hit / 2 non-push
        self.assertEqual(sum(b["count"] for b in out["buckets"]), 2)

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
        out = diag.totals_pick_table(self._decided(rows))
        self.assertEqual(out["n_pushes"], 0)
        self.assertEqual(out["n_games"], 1)
        self.assertAlmostEqual(out["win_rate"], 0.0)   # over pick missed

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


class TestTotalsPickTable(unittest.TestCase):
    def test_hand_computed_buckets_and_win_rate(self):
        # Each game carries ONLY its target line's columns, so the fair line
        # resolves to that line deterministically. re-scaled P(over) =
        # po/(po+pu); pick = the side with re-scaled P >= 0.5; confidence =
        # picked side's re-scaled P; 1% buckets (50-51 … 60+).
        rows = [
            # A: line 9.0, rso=0.54/0.94=0.5745 over, total 10 -> hit, 57-58
            {"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 10, "p_over_9_0": 0.54,
             "p_under_9_0": 0.40},
            # B: same line/prob, total 8 -> over MISS (8 < 9.5), 57-58
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 8, "p_over_9_0": 0.54,
             "p_under_9_0": 0.40},
            # C: line 9.5, rso=0.60, over, total 9 -> MISS (9 < 10), 60+
            {"game_pk": 2, "kind": "oof", "home_expected_runs": 4.7,
             "away_expected_runs": 4.8, "total_runs": 9, "p_over_9_5": 0.60,
             "p_under_9_5": 0.40},
            # D: line 8.5, rso=0.45 -> PICK UNDER conf 0.55, total 7 -> hit,
            #    55-56
            {"game_pk": 3, "kind": "oof", "home_expected_runs": 4.2,
             "away_expected_runs": 4.3, "total_runs": 7, "p_over_8_5": 0.45,
             "p_under_8_5": 0.55},
            # E: line 9.5, rso=0.51 over, total 12 -> hit (12 >= 10), 51-52
            {"game_pk": 4, "kind": "oof", "home_expected_runs": 4.7,
             "away_expected_runs": 4.8, "total_runs": 12, "p_over_9_5": 0.51,
             "p_under_9_5": 0.49},
        ]
        decided = pd.DataFrame(rows)
        out = diag.totals_pick_table(decided)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_games"], 5)
        by = {b["bucket"]: b for b in out["buckets"]}
        # A over@0.5745 hit (10 >= 9.5) + B over@0.5745 MISS (8 < 9.5)
        self.assertEqual(by["57-58"]["count"], 2)
        self.assertAlmostEqual(by["57-58"]["accuracy"], 50.0)
        # C over@0.60 MISS (9 < 10)
        self.assertEqual(by["60+"]["count"], 1)
        self.assertAlmostEqual(by["60+"]["accuracy"], 0.0)
        # D under@0.55 hit (7 < 9)
        self.assertEqual(by["55-56"]["count"], 1)
        self.assertAlmostEqual(by["55-56"]["accuracy"], 100.0)
        # E over@0.51 hit (12 >= 10)
        self.assertEqual(by["51-52"]["count"], 1)
        self.assertAlmostEqual(by["51-52"]["accuracy"], 100.0)
        # Empty 1% buckets present with count 0, never dropped
        self.assertEqual(by["50-51"]["count"], 0)
        self.assertEqual(by["58-59"]["count"], 0)
        self.assertEqual(sum(b["count"] for b in out["buckets"]), 5)
        self.assertAlmostEqual(out["win_rate"], 0.6)  # 3/5 pooled
        self.assertIn("fair line", out["pick_rule"])

    def test_missing_columns_warn_not_crash(self):
        out = diag.totals_pick_table(
            make_grid_df(n=5).drop(columns=["home_expected_runs"]))
        self.assertIsNotNone(out["warning"])
        self.assertEqual(out["buckets"], [])
        out = diag.totals_pick_table(pd.DataFrame())
        self.assertIsNotNone(out["warning"])
        # No p_under anywhere → re-scaled 2-way can't be priced → empty, not
        # a crash (fair-line confidence is always 2-way re-normalized).
        bare = pd.DataFrame({
            "game_pk": [0], "kind": ["oof"],
            "home_expected_runs": [4.5], "away_expected_runs": [4.5],
            "total_runs": [10], "p_over_9_0": [0.52],
        })
        out = diag.totals_pick_table(bare)
        self.assertIsNotNone(out["warning"])
        self.assertEqual(out["buckets"], [])

    def test_confidence_is_rescaled_not_raw_over(self):
        """A whole-line pick must use the re-scaled 2-way P, never raw
        max(p_over, 1 - p_over) (which folds the push band into the under
        complement and inflates confidence). Here p_over=0.36 with a fat
        push band (p_under=0.48): raw max = 0.64 → a bogus 64% bucket;
        re-scaled picked side = 0.48/0.84 = 0.571 → 57-58 bucket."""
        rows = [{"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
                 "away_expected_runs": 4.5, "total_runs": 8,
                 "p_over_9_0": 0.36, "p_under_9_0": 0.48}]
        out = diag.totals_pick_table(pd.DataFrame(rows))
        by = {b["bucket"]: b for b in out["buckets"]}
        # Under pick, conf = 0.48/(0.36+0.48) = 0.5714 → 57-58
        self.assertEqual(by["57-58"]["count"], 1)
        # No game lands in 60+/any bucket above 59-60 from the raw impulse
        self.assertEqual(by["60+"]["count"], 0)
        self.assertEqual(sum(b["count"] for b in out["buckets"]), 1)

    def test_conf_buckets_are_1pct_increments(self):
        self.assertEqual(diag.TOTALS_PICK_LABELS,
                         ["50-51", "51-52", "52-53", "53-54", "54-55",
                          "55-56", "56-57", "57-58", "58-59", "59-60",
                          "60+"])
        self.assertEqual(diag.TOTALS_PICK_EDGES,
                         [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 101])

    def test_real_artifact_fairline_shift_and_sum(self):
        """On the shipped artifact the fair-line pick is re-scaled, so the
        former 55-60 population lands in the 50-53 bands: buckets sum to the
        priced non-push games and nothing sits at 55+ (max re-scaled conf ≈
        52.7)."""
        dd = Path(__file__).resolve().parents[1] / "data_delivery"
        m_path = dd / "run_engine_markets_20260829.csv"
        if not m_path.exists():
            self.skipTest("run-engine artifact absent")
        out = diag.totals_pick_table(diag.decided_rows(pd.read_csv(m_path)))
        self.assertIsNone(out["warning"])
        by = {b["bucket"]: b for b in out["buckets"]}
        # Buckets sum exactly to the priced non-push games (no double count)
        self.assertEqual(sum(b["count"] for b in out["buckets"]),
                         out["n_games"])
        # 55+ is empty on this artifact — the max re-scaled conf is ~52.7
        hi = sum(by[k]["count"] for k in
                 ("55-56", "56-57", "57-58", "58-59", "59-60", "60+"))
        self.assertEqual(hi, 0)
        self.assertEqual(by["60+"]["count"], 0)


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
        tp = diag.totals_pick_table(decided)
        self.assertIsNone(tp["warning"])
        self.assertEqual(tp["n_pushes"], stats["n_pushes"])
        self.assertEqual(tp["n_games"] + tp["n_pushes"],
                         stats["n_games"],
                         "win-rate denominator excludes pushes exactly")
        self.assertAlmostEqual(tp["push_rate"], stats["push_rate"], places=4)
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
        for kwargs in ({}, {"total_line": True, "acc_y_max": 75.0}):
            built = diag.chart_pick_buckets(
                diag.totals_pick_table(self.decided), "t", **kwargs)
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

    def test_7_totals_picks_renders(self):
        built = diag.chart_pick_buckets(
            diag.totals_pick_table(self.decided), "Totals picks")
        self._assert_chart_has_data(built["chart"])
        self.assertFalse(built["table"].empty)
        # Labels follow the 4-bucket convention (50–55 … 65+)
        self.assertEqual(built["table"]["bucket"].tolist(),
                         diag.TOTALS_PICK_LABELS)
        # Default build: no reference line (that's the total_line=True
        # amber rule), but the accuracy axis is ALWAYS present with an
        # explicit domain (the fix for the Run-line picks regression —
        # scale=None serialized to "scale": null and dropped the axis).
        default_dump = _spec_dump(built["chart"])
        self.assertNotIn('"rule"', default_dump)
        self.assertIn('"domain": [0.0, 100.0]', default_dump)

    def test_totals_picks_total_line_at_pooled_rate(self):
        import json
        tp = diag.totals_pick_table(self.decided)
        built = diag.chart_pick_buckets(
            tp, "Totals picks", total_line=True)
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
        tp = diag.totals_pick_table(self.decided)
        built = diag.chart_pick_buckets(
            tp, "Totals picks", total_line=True, acc_y_max=75.0)
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
        table = diag.pick_buckets(p, hit, labels=diag.TOTALS_PICK_LABELS,
                                  edges=diag.TOTALS_PICK_EDGES)
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


if __name__ == "__main__":
    unittest.main()
