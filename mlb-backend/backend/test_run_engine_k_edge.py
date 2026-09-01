"""Tests for the C2 k-edge expansion (run_engine_k_edge.py) + its rollout gate
(run_engine_k_edge_gate.py).

The k-edge module monkey-patches run_engine's derive_markets_v3 /
predict_slate_runs / run_engine_daily at import. These tests assert:

  - apply_k_edge preserves the LEVEL (λ'_H + λ'_A == λ_H + λ_A) and is the
    identity at k = 1.0.
  - fit_k_edge fits on the MASKED (pre-holdout) games only — corrupting a
    sealed game's margin never moves k; it recovers a known slope.
  - k_edge_holdout_mask mirrors the pre-holdout discipline (chronological
    last HOLDOUT_DAYS sealed).
  - k_edge_meta emits the drift band [ref−0.2, ref+0.2] and raises
    drift_alert outside it.
  - derive_markets_v3 (wrapper): no expansion + no k_edge key when k_edge is
    None; expansion + k logged when k_edge is given; the daily seam
    (_K_EDGE_ACTIVE) expands the OOF even with k_edge=None (the OOF and
    slate must price the SAME k); k=1.0 stays identity but still logs.
  - predict_slate_runs (wrapper): seam inactive -> output untouched;
    seam active -> λ columns expanded and grid re-priced.
  - patch()/unpatch() are idempotent and restore the originals.
  - Gate helpers: _bin_rows deciles/extremes, _totals_rows push-exclusion,
    _pwin_gap bucket math.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_engine as re
import run_engine_k_edge as ke
from run_engine_k_edge_gate import _bin_rows, _pwin_gap, _totals_rows


def _oof_like(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lh = rng.uniform(2.5, 5.5, n)
    la = rng.uniform(2.5, 5.5, n)
    return pd.DataFrame({
        "game_pk": np.arange(n),
        "game_date": pd.date_range("2021-04-01", periods=n, freq="2D"),
        "fold_idx": np.repeat(np.arange(20), n // 20),
        "home_expected_runs": lh,
        "away_expected_runs": la,
        "home_score": np.round(lh * 1.1),
        "away_score": np.round(la),
    })


class TestApplyFit(unittest.TestCase):

    def test_apply_preserves_sum_and_scales_edge(self):
        rng = np.random.default_rng(1)
        lh = rng.uniform(2.0, 6.0, 200)
        la = rng.uniform(2.0, 6.0, 200)
        for k in (1.3, 1.5, 1.5306, 1.7):
            lh2, la2 = ke.apply_k_edge(lh, la, k)
            np.testing.assert_allclose(lh2 + la2, lh + la, atol=1e-12)
            np.testing.assert_allclose(lh2 - la2, k * (lh - la), atol=1e-12)

    def test_identity_at_k_1(self):
        lh = np.array([3.5, 4.1, 2.9])
        la = np.array([3.9, 3.2, 4.4])
        h1, a1 = ke.apply_k_edge(lh, la, 1.0)
        np.testing.assert_allclose(h1, lh)
        np.testing.assert_allclose(a1, la)

    def test_fit_recovers_known_slope(self):
        rng = np.random.default_rng(7)
        n = 400
        lh = rng.uniform(2.5, 5.5, n)
        la = rng.uniform(2.5, 5.5, n)
        margin = 1.53 * (lh - la) + rng.normal(0, 0.2, n)
        k = ke.fit_k_edge(lh, la, margin, np.ones(n, bool))
        self.assertAlmostEqual(k, 1.53, delta=0.02)

    def test_fit_uses_masked_games_only(self):
        """Corrupting the UNMASKED (sealed) margins must not move k."""
        rng = np.random.default_rng(3)
        n = 500
        lh = rng.uniform(2.5, 5.5, n)
        la = rng.uniform(2.5, 5.5, n)
        margin = 1.53 * (lh - la) + rng.normal(0, 0.2, n)
        mask = np.zeros(n, bool)
        mask[:n // 2] = True
        k_a = ke.fit_k_edge(lh, la, margin, mask)
        margin_corrupt = margin.copy()
        margin_corrupt[~mask] = 999.0
        k_b = ke.fit_k_edge(lh, la, margin_corrupt, mask)
        self.assertAlmostEqual(k_a, k_b, places=9)

    def test_fit_bails_to_identity_on_low_variance(self):
        lh = np.full(120, 4.0)
        la = np.full(120, 4.0)
        margin = np.zeros(120)
        k = ke.fit_k_edge(lh, la, margin, np.ones(120, bool))
        self.assertEqual(k, 1.0)

    def test_holdout_mask_seals_last_holdout_days(self):
        oof = _oof_like(300)
        mask = ke.k_edge_holdout_mask(oof)
        dates = pd.to_datetime(oof["game_date"])
        cutoff = dates.max() - pd.Timedelta(days=re.HOLDOUT_DAYS)
        self.assertTrue((dates[mask] < cutoff).all())
        self.assertTrue((dates[~mask] >= cutoff).all())
        self.assertGreater(int(mask.sum()), 0)
        self.assertGreater(int((~mask).sum()), 0)

    def test_k_edge_meta_drift_band(self):
        m = ke.k_edge_meta(1.5)
        self.assertEqual(m["drift_band"], [1.33, 1.73])
        self.assertFalse(m["drift_alert"])
        m = ke.k_edge_meta(1.8)
        self.assertTrue(m["drift_alert"])
        m = ke.k_edge_meta(1.3)
        self.assertTrue(m["drift_alert"])
        # k=1.0 (identity / disabled) is 0.53 from the 1.53 reference —
        # outside the band by design, so it MUST alert.
        m = ke.k_edge_meta(1.0)
        self.assertTrue(m["drift_alert"])


class TestV3Wrapper(unittest.TestCase):

    def _orig(self, oof, **kw):
        return {"summary": {"n_pre": 100, "n_holdout": 100},
                "markets": oof.copy()}

    def test_none_no_expansion_no_meta(self):
        oof = _oof_like()
        with patch.object(ke, "_orig_derive_markets_v3",
                          side_effect=self._orig) as orig:
            res = ke.derive_markets_v3(oof, k_edge=None)
        oof_in = orig.call_args.args[0]
        np.testing.assert_allclose(
            oof_in["home_expected_runs"], oof["home_expected_runs"])
        self.assertNotIn("k_edge", res["summary"])

    def test_k_edge_expands_and_logs(self):
        oof = _oof_like()
        with patch.object(ke, "_orig_derive_markets_v3",
                          side_effect=self._orig) as orig:
            res = ke.derive_markets_v3(oof, k_edge=1.53)
        oof_in = orig.call_args.args[0]
        spread0 = (oof["home_expected_runs"] - oof["away_expected_runs"])
        spread1 = (oof_in["home_expected_runs"]
                   - oof_in["away_expected_runs"])
        self.assertGreater(np.std(spread1), np.std(spread0))
        np.testing.assert_allclose(
            oof_in["home_expected_runs"] + oof_in["away_expected_runs"],
            oof["home_expected_runs"] + oof["away_expected_runs"],
            atol=1e-3)  # wrapper rounds λ to 4dp
        self.assertIn("k_edge", res["summary"])
        self.assertEqual(res["summary"]["k_edge"]["k"], 1.53)
        self.assertFalse(res["summary"]["k_edge"]["drift_alert"])

    def test_daily_seam_expands_even_with_none(self):
        oof = _oof_like()
        with patch.object(ke, "_orig_derive_markets_v3",
                          side_effect=self._orig) as orig:
            ke._K_EDGE_ACTIVE = 1.53
            try:
                res = ke.derive_markets_v3(oof, k_edge=None)
            finally:
                ke._K_EDGE_ACTIVE = None
        oof_in = orig.call_args.args[0]
        spread0 = (oof["home_expected_runs"] - oof["away_expected_runs"])
        spread1 = (oof_in["home_expected_runs"]
                   - oof_in["away_expected_runs"])
        self.assertGreater(np.std(spread1), np.std(spread0))
        self.assertIn("k_edge", res["summary"])
        self.assertEqual(res["summary"]["k_edge"]["k"], 1.53)

    def test_k_1_identity_but_logged(self):
        oof = _oof_like()
        with patch.object(ke, "_orig_derive_markets_v3",
                          side_effect=self._orig) as orig:
            res = ke.derive_markets_v3(oof, k_edge=1.0)
        oof_in = orig.call_args.args[0]
        np.testing.assert_allclose(
            oof_in["home_expected_runs"], oof["home_expected_runs"])
        self.assertIn("k_edge", res["summary"])
        self.assertEqual(res["summary"]["k_edge"]["k"], 1.0)


class TestSlateWrapper(unittest.TestCase):

    def _slate(self, n=6):
        rng = np.random.default_rng(5)
        return pd.DataFrame({
            "game_pk": np.arange(n),
            "home_expected_runs": rng.uniform(3.0, 5.0, n),
            "away_expected_runs": rng.uniform(3.0, 5.0, n),
            "alpha_home": np.full(n, 0.15),
            "alpha_away": np.full(n, 0.15),
            "p_over_8_5": np.full(n, 0.5),
            "p_home_cover_1_5": np.full(n, 0.5),
            "p_home_win_derived": np.full(n, 0.52),
            "p_away_win_derived": np.full(n, 0.48),
        })

    def test_seam_inactive_output_untouched(self):
        out = self._slate()
        with patch.object(ke, "_orig_predict_slate_runs",
                          return_value=out.copy()):
            res = ke.predict_slate_runs(None, None, {}, {})
        pd.testing.assert_frame_equal(
            res, out, check_exact=False, atol=1e-9)

    def test_seam_active_expands_and_reprices(self):
        out = self._slate()
        called = {}

        def fake_mc(lh, la, ah, aa, n_draws=ke._re.MC_DRAWS, seed=ke._re.MARKET_SEED):
            called["edge"] = float(np.std(lh - la))
            n = len(lh)
            return {
                "p_over_grid": np.full((n, 13), 0.51),
                "p_push_grid": np.full((n, 13), 0.02),
                "p_cover_grid": np.full((n, 4), 0.55),
                "p_rl_home_grid": np.full((n, 7), 0.54),
                "p_rl_push_grid": np.full((n, 7), 0.02),
                "p_rl_away_grid": np.full((n, 7), 0.44),
                "p_home_win_derived": np.full(n, 0.60),
            }

        with patch.object(ke, "_orig_predict_slate_runs",
                          return_value=out.copy()), \
             patch.object(ke._re, "derive_markets_mc",
                          side_effect=fake_mc), \
             patch.object(ke._re, "alpha_of",
                          return_value=np.full(len(out), 0.15)):
            ke._K_EDGE_ACTIVE = 1.5
            try:
                res = ke.predict_slate_runs(None, None, {}, {"home": {},
                                                             "away": {}})
            finally:
                ke._K_EDGE_ACTIVE = None
        spread0 = np.std(out["home_expected_runs"]
                         - out["away_expected_runs"])
        self.assertGreater(np.std(res["home_expected_runs"]
                                  - res["away_expected_runs"]), spread0)
        # grid re-priced from the expanded λ (stub values land)
        self.assertAlmostEqual(float(res["p_home_cover_1_5"].iloc[0]), 0.55)
        self.assertAlmostEqual(float(res["p_home_win_derived"].iloc[0]), 0.60)
        self.assertGreater(called["edge"], spread0)


class TestPatch(unittest.TestCase):

    def test_patch_idempotent_and_unpatch_restores(self):
        orig_d = ke._orig_derive_markets_v3
        orig_s = ke._orig_predict_slate_runs
        orig_da = ke._orig_run_engine_daily
        self.assertIs(re.derive_markets_v3, ke.derive_markets_v3)  # patched
        ke.patch()
        ke.patch()
        self.assertIs(re.derive_markets_v3, ke.derive_markets_v3)
        ke.unpatch()
        self.assertIs(re.derive_markets_v3, orig_d)
        self.assertIs(re.predict_slate_runs, orig_s)
        self.assertIs(re.run_engine_daily, orig_da)
        ke.patch()


class TestGateHelpers(unittest.TestCase):

    def test_bin_rows_deciles_and_extremes(self):
        rng = np.random.default_rng(9)
        p = np.clip(rng.normal(0.5, 0.12, 2000), 0.05, 0.95)
        y = (rng.random(2000) < p).astype(float)
        tbl = _bin_rows(p, y, np.ones(2000, bool))
        self.assertGreater(len(tbl["deciles"]), 3)
        self.assertEqual(len(tbl["extreme"]), 2)
        self.assertIsNotNone(tbl["overall_delta"])
        # extreme bins: last one is >= 0.70
        self.assertEqual(tbl["extreme"][-1]["bin"], ">=0.70")

    def test_totals_rows_push_excluded(self):
        rng = np.random.default_rng(2)
        n = 300
        lh = rng.uniform(3.5, 5.5, n)
        la = rng.uniform(3.5, 5.5, n)
        total = np.round(lh + la)
        grid = np.tile(np.linspace(0.2, 0.8, 13), (n, 1))
        ps, ys, idx = _totals_rows(grid, lh, la, total)
        # every row must be non-push (total != line) and within the grid
        lines = np.asarray([re._rounded_total_line(lh[i], la[i])
                            for i in range(n)])
        for i in idx:
            self.assertNotEqual(total[i], lines[i])
        self.assertGreater(len(ps), 0)
        self.assertEqual(len(ps), len(ys))
        self.assertEqual(len(ps), len(idx))

    def test_pwin_gap_bucket(self):
        p = np.array([0.56, 0.57, 0.58, 0.30, 0.62, 0.55])
        y = np.array([1, 1, 0, 1, 1, 0])
        gap = _pwin_gap(p, y, np.ones(6, bool), 0.55, 0.60)
        self.assertEqual(gap["n"], 4)  # 0.56, 0.57, 0.58, 0.55
        self.assertAlmostEqual(gap["pred"], 0.565)
        self.assertAlmostEqual(gap["actual"], 0.5)


if __name__ == "__main__":
    unittest.main()
