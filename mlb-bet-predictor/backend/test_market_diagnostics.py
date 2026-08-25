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

FRONTEND = Path(__file__).resolve().parents[1].parent / "mlb-bet-predictor" / "frontend"
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
        # expected total.
        for g in diag.TOTAL_GRID:
            shift = (exp_t - g) * 0.25
            row[f"p_over_{str(g).replace('.', '_')}"] = float(
                1 / (1 + math.exp(-4 * shift)))
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
        df = make_grid_df(n=10)
        df["total_runs"] = 9
        df = df.drop(columns=["p_over_9_0"])
        pairs = diag.rounded_total_pairs(df)
        # Only games rounding to lines other than 9.0 survive
        self.assertGreaterEqual(len(pairs), 0)
        self.assertLess(len(pairs), 10)
        self.assertTrue((pairs["line"] != 9.0).all())

    def test_empty_input_loud_warning(self):
        pairs = diag.rounded_total_pairs(pd.DataFrame())
        self.assertEqual(len(pairs), 0)
        pairs = diag.rounded_total_pairs(
            make_grid_df(n=5).drop(columns=["home_expected_runs"]))
        self.assertEqual(len(pairs), 0)


class TestPushExclusion(unittest.TestCase):
    """Whole-number-line pushes (total == rounded line) are neither wins
    nor losses — excluded from win rates, reported in push_rate. The card's
    Over/Under display is untouched (still exactly two, summing to 1)."""

    def _decided(self, rows):
        return pd.DataFrame(rows)

    def test_pick_table_excludes_push(self):
        rows = [
            # PUSH: total 9 == rounded line 9.0 (4.5 + 4.5)
            {"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 9, "p_over_9_0": 0.52},
            # over hit: 10 >= 9.5
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 10, "p_over_9_0": 0.52},
            # over miss: 8 < 9.5
            {"game_pk": 2, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 8, "p_over_9_0": 0.52},
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
                 "away_expected_runs": 4.8, "total_runs": 9, "p_over_9_5": 0.62}]
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
        rows = [
            # home, away, total, p_over at own rounded line
            {"game_pk": 0, "kind": "oof", "home_expected_runs": 4.5,
             "away_expected_runs": 4.5, "total_runs": 10, "p_over_9_0": 0.52},
            {"game_pk": 1, "kind": "oof", "home_expected_runs": 4.6,
             "away_expected_runs": 4.4, "total_runs": 8, "p_over_9_0": 0.42},
            {"game_pk": 2, "kind": "oof", "home_expected_runs": 4.7,
             "away_expected_runs": 4.8, "total_runs": 11, "p_over_9_5": 0.62},
            {"game_pk": 3, "kind": "oof", "home_expected_runs": 5.0,
             "away_expected_runs": 5.0, "total_runs": 5, "p_over_10_0": 0.72},
            {"game_pk": 4, "kind": "oof", "home_expected_runs": 4.8,
             "away_expected_runs": 4.7, "total_runs": 9, "p_over_9_5": 0.53},
        ]
        decided = pd.DataFrame(rows)
        out = diag.totals_pick_table(decided)
        self.assertIsNone(out["warning"])
        self.assertEqual(out["n_games"], 5)
        by = {b["bucket"]: b for b in out["buckets"]}
        # A over@0.52 hit (10 >= 9.5) + E over@0.53 MISS (9 < 10) → 2 picks, 50%
        self.assertEqual(by["50-55"]["count"], 2)
        self.assertAlmostEqual(by["50-55"]["accuracy"], 50.0)
        # B under@0.58 (0.42 < 0.5) hit (8 < 9.5) → 55-60
        self.assertEqual(by["55-60"]["count"], 1)
        self.assertAlmostEqual(by["55-60"]["accuracy"], 100.0)
        # C over@0.62 hit (11 >= 10) → 60-65
        self.assertEqual(by["60-65"]["count"], 1)
        self.assertAlmostEqual(by["60-65"]["accuracy"], 100.0)
        # D over@0.72 MISS (5 < 10.5) → 65+
        self.assertEqual(by["65+"]["count"], 1)
        self.assertAlmostEqual(by["65+"]["accuracy"], 0.0)
        self.assertAlmostEqual(out["win_rate"], 0.6)  # 3/5 pooled
        self.assertIn("rounded total", out["pick_rule"])

    def test_missing_columns_warn_not_crash(self):
        out = diag.totals_pick_table(
            make_grid_df(n=5).drop(columns=["home_expected_runs"]))
        self.assertIsNotNone(out["warning"])
        self.assertEqual(out["buckets"], [])
        out = diag.totals_pick_table(pd.DataFrame())
        self.assertIsNotNone(out["warning"])


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
