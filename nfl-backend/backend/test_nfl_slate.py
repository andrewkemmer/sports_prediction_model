"""NFL slate-serve pricing module tests (nfl_slate_engine / run_nfl_slate).

Covers: pinned-params constants (DN/sigma/rho/tie/(c,d)), board-era-center
mirror equivalence with the era module (compute_centers), price_board
semantics (fair-vs-offer separation, grid monotonicity + complementarity,
offer NaN honesty, shrink flags), FEATURE_COLUMNS / no-moneyline-import
pins, and — when the real artifacts exist — schema round-trip vs the
mapping table + first-run monitor empty-state honesty.

Heavy runner paths (full 2026 board pricing, refit walks) are exercised by
the runner itself (run_nfl_slate.py gates) and by the light real-artifact
checks here; no synthetic full-chain runs.
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
from nfl_era_features import CENTER_COLS, compute_centers  # noqa: E402
from nfl_joint_engine import cover_prob  # noqa: E402


def _synth_decided(seed: int = 0) -> pd.DataFrame:
    """Small multi-week decided frame (two teams, scores with drift)."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-09-01", periods=10, freq="7D")
    teams = ["KC", "BUF"]
    gid = 0
    for i, d in enumerate(dates):
        for home in teams:
            away = teams[1] if home == teams[0] else teams[0]
            gid += 1
            rows.append({
                "game_id": f"2024_{i + 1:02d}_{away}_{home}",
                "season": 2024, "gameday": d,
                "home_score": float(rng.normal(24 + i * 0.1, 6)),
                "away_score": float(rng.normal(20, 6)),
            })
    return pd.DataFrame(rows)


def _synth_board(decided: pd.DataFrame, n: int = 6,
                 after_days: int = 40) -> pd.DataFrame:
    """Board rows strictly after the decided timeline (no scores)."""
    last = decided["gameday"].max()
    teams = sorted(set(decided["home_team"] if "home_team" in decided
                       else ["KC", "BUF"]))
    rows = []
    for i in range(n):
        rows.append({
            "game_id": f"2026_{i + 1:02d}_A_B",
            "gameday": last + pd.Timedelta(days=after_days + i * 7),
        })
    return pd.DataFrame(rows)


def _synth_refit(board: pd.DataFrame, mu_h: float = 24.0,
                 mu_a: float = 20.0) -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": board["game_id"],
        "pred_home": np.full(len(board), mu_h),
        "pred_away": np.full(len(board), mu_a),
    })


def _lines(board: pd.DataFrame, n_offers: int | None = None) -> pd.DataFrame:
    n = n_offers if n_offers is not None else len(board)
    return pd.DataFrame({
        "game_id": board["game_id"].iloc[:n],
        "spread_line": [-3.0] * n,
        "total_line": [46.0] * n,
    })


class TestPinnedParams(unittest.TestCase):
    """The slate builder uses EXACTLY the research-pinned constants — never
    refit at slate time (in-sample dispersion would re-create hot totals)."""

    def test_pinned_constants(self):
        self.assertEqual(SE.PINNED_SIGMA_HOME, 9.663)
        self.assertEqual(SE.PINNED_SIGMA_AWAY, 9.0789)
        self.assertEqual(SE.PINNED_RHO, 0.0076)
        self.assertEqual(SE.PINNED_P_TIE, 0.00275)
        self.assertEqual(SE.ERA_SPEC, "ewm_2w")
        self.assertEqual(SE.MEDIAN_ROUNDS, {"home": 20, "away": 23})
        self.assertEqual(SE.TOTALS_CD, (-0.3599, 0.3472))
        self.assertEqual(SE.SPREAD_CD, (0.446165, 0.307486))
        self.assertEqual(SE.FEED_PRESENT, False)

    def test_pinned_params_shape_and_fit_on(self):
        p = SE.pinned_joint_params()
        self.assertEqual(p["family"], "dn")
        self.assertEqual(p["fit_on"], "pooled_oof")
        self.assertEqual(p["sigma_h"]["spec"], "const")
        self.assertEqual(p["sigma_a"]["spec"], "const")
        self.assertEqual(p["sigma_h"]["sigma0"], 9.663)
        self.assertEqual(p["sigma_a"]["sigma0"], 9.0789)
        self.assertEqual(p["rho"], 0.0076)
        # build_joint_pmfs' sealed leak guard accepts the pinned params
        # (fit_on == pooled_oof) — refitting on slate data is structurally
        # impossible through this API.
        from nfl_joint_engine import build_joint_pmfs
        r = pd.DataFrame({"game_id": ["g1", "g2"],
                          "pred_home": [24.0, 20.0],
                          "pred_away": [20.0, 24.0]})
        pmfs, _ = build_joint_pmfs(r, p, SE.PINNED_P_TIE)
        self.assertEqual(pmfs.shape[0], 2)
        self.assertTrue(np.allclose(pmfs.sum(axis=(1, 2)), 1.0, atol=1e-9))


class TestBoardEraCenters(unittest.TestCase):
    """board_era_centers must reproduce the era module's day recursion EXACTLY
    (strictly-prior decided scores, same-day excluded, decay semantics)."""

    def _ref_on_day(self, decided, day, ids):
        """compute_centers' center for fake decided games at ONE day (state
        before that day — nothing after it, so no contamination)."""
        rows = [{"game_id": g, "season": 2026, "gameday": day,
                 "home_score": 21.0, "away_score": 21.0} for g in ids]
        combined = pd.concat([decided, pd.DataFrame(rows)],
                             ignore_index=True)
        ref = compute_centers(combined, SE.ERA_SPEC)
        ref = ref[ref["game_id"].isin(set(ids))].reset_index(drop=True)
        return ref.sort_values("game_id").reset_index(drop=True)

    def test_mirror_equivalence_with_compute_centers(self):
        decided = _synth_decided(1)
        board = _synth_board(decided, n=6)
        # All board rows on ONE future day — the mirror's center for that day
        # (state after every strictly-prior decided game) must equal what
        # compute_centers assigns a decided game on the same day (the fake's
        # own score cannot contaminate anything after it — nothing follows).
        board["gameday"] = decided["gameday"].max() + pd.Timedelta(days=40)
        ref = self._ref_on_day(decided, board["gameday"].iloc[0],
                               board["game_id"])
        got = SE.board_era_centers(decided, board, SE.ERA_SPEC)
        got = got.sort_values("game_id").reset_index(drop=True)
        self.assertEqual(len(got), len(board))
        pd.testing.assert_frame_equal(got[CENTER_COLS].round(6),
                                      ref[CENTER_COLS].round(6))

    def test_same_day_excluded_later_included(self):
        decided = _synth_decided(2)
        last_day = decided["gameday"].max()
        later_day = last_day + pd.Timedelta(days=7)
        board = pd.DataFrame({"game_id": ["SAME", "LATER"],
                              "gameday": [last_day, later_day]})
        got = SE.board_era_centers(decided, board, SE.ERA_SPEC)
        same = float(got.loc[got["game_id"] == "SAME",
                             "era_center_home"].iloc[0])
        later = float(got.loc[got["game_id"] == "LATER",
                              "era_center_home"].iloc[0])
        # SAME is on the last decided day: same-day games are excluded, so its
        # center is the state BEFORE that day's scores. LATER (+7d) includes
        # them (strictly prior) — pinned against compute_centers directly.
        ref_same = self._ref_on_day(decided, last_day, ["SAME"])
        ref_later = self._ref_on_day(decided, later_day, ["LATER"])
        self.assertAlmostEqual(
            same, float(ref_same["era_center_home"].iloc[0]), places=4)
        self.assertAlmostEqual(
            later, float(ref_later["era_center_home"].iloc[0]), places=4)
        self.assertNotAlmostEqual(same, later, places=2)
        self.assertTrue(np.isfinite(same) and np.isfinite(later))


class TestPriceBoard(unittest.TestCase):
    def _frame(self, n_offers: int | None = None, n: int = 4):
        board = _synth_board(_synth_decided(3), n=n)
        refit = _synth_refit(board)
        return SE.price_board(refit, SE.pinned_joint_params(),
                              SE.PINNED_P_TIE,
                              lines=_lines(board, n_offers))

    def test_offerless_rows_still_quote_fair_lines(self):
        out = self._frame(n_offers=0)
        self.assertTrue(out["spread_line"].isna().all())
        self.assertTrue(out["p_cover_offered"].isna().all())
        # Fair (model) lines are always quoted; never conflated with offers.
        self.assertTrue(out["fair_spread"].notna().all())
        self.assertTrue(out["fair_total"].notna().all())
        self.assertTrue(out["derived_ml"].notna().all())

    def test_offered_line_quotes_present(self):
        out = self._frame(n_offers=2, n=4)
        self.assertEqual(int(out["spread_line"].notna().sum()), 2)
        self.assertEqual(int(out["total_line"].notna().sum()), 2)
        sub = out[out["spread_line"].notna()]
        self.assertTrue(sub["p_cover_offered"].between(0, 1).all())
        self.assertTrue(sub["p_over_offered"].between(0, 1).all())

    def test_grid_monotonicity_and_complementarity(self):
        out = self._frame(n=6)
        for _, row in out.iterrows():
            covs = [row[f"p_home_cover_{SE._fname(float(L))}"]
                    for L in SE.SPREAD_INT_LINES]
            self.assertEqual(covs, sorted(covs, reverse=True))
            for U in SE.TOTAL_INT_LINES:
                p_o = row[f"p_over_{SE._fname(float(U))}"]
                p_u = row[f"p_under_{SE._fname(float(U))}"]
                p_p = row[f"p_push_{SE._fname(float(U))}"]
                self.assertAlmostEqual(p_o + p_u + p_p, 1.0, places=5)

    def test_raw_pair_and_derived_pair(self):
        out = self._frame(n_offers=3, n=3)
        sub = out[out["spread_line"].notna()].reset_index(drop=True)
        for _, r in sub.iterrows():
            self.assertAlmostEqual(
                r["p_home_cover_minus_half"] + r["p_away_cover_minus_half"],
                1.0, places=5)
            self.assertAlmostEqual(
                r["p_home_cover_plus_half"] + r["p_away_cover_plus_half"],
                1.0, places=5)
            self.assertAlmostEqual(r["p_home_win_derived"]
                                   + r["p_away_win_derived"], 1.0, places=6)
        self.assertTrue(sub["derived_ml"].between(0, 1).all())

    def test_shrink_columns_additive_and_flagged(self):
        out = self._frame(n_offers=3, n=3)
        self.assertTrue((out["shrink_applied"] == 0).all())
        self.assertTrue(out["fair_spread_shrunk"].notna().all())
        self.assertTrue(out["fair_total_shrunk"].notna().all())
        self.assertTrue(out["derived_ml_shrunk"].between(0, 1).all())
        # Offer-dependent shrink quotes are NaN only without an offer.
        self.assertEqual(int(out["p_cover_shrunk"].notna().sum()), 3)

    def test_determinism_double_walk(self):
        board = _synth_board(_synth_decided(4), n=5)
        refit = _synth_refit(board)
        a = SE.price_board(refit, SE.pinned_joint_params(), SE.PINNED_P_TIE,
                           lines=_lines(board))
        b = SE.price_board(refit, SE.pinned_joint_params(), SE.PINNED_P_TIE,
                           lines=_lines(board))
        self.assertEqual(a.to_csv(index=False), b.to_csv(index=False))


class TestScopePins(unittest.TestCase):
    def test_feature_columns_untouched(self):
        import nfl_features as nf
        # The slate engine must not register anything into the served pool.
        self.assertNotIn("market_home_implied", nf.FEATURE_COLUMNS)
        self.assertNotIn("spread_line", nf.FEATURE_COLUMNS)
        self.assertNotIn("total_line", nf.FEATURE_COLUMNS)

    def test_no_moneyline_import_in_engine(self):
        src = (BACKEND / "nfl_slate_engine.py").read_text()
        self.assertNotIn("import nfl_moneyline", src)
        self.assertNotIn("from nfl_moneyline", src)

    def test_market_free_board_slate_frame(self):
        import nfl_features as nf
        # build_slate_features drops market columns from the model frame
        # (market-independence policy); the offers are a separate feed.
        self.assertNotIn("FEED_PRESENT", dir(nf))


class TestRealArtifacts(unittest.TestCase):
    """Round-trip the dated artifacts (when present in data_delivery) against
    the mapping table + first-run monitor empty-state honesty."""

    ROOT = BACKEND.parents[0] if BACKEND.name == "backend" else None
    DD = BACKEND.parent / "data_delivery"

    def _latest(self, prefix: str, ext: str) -> Path | None:
        cands = sorted(self.DD.glob(f"{prefix}*{ext}"))
        return cands[-1] if cands else None

    def test_markets_csv_schema_round_trip(self):
        path = self._latest("nfl_run_engine_markets_", ".csv")
        if path is None:
            self.skipTest("no nfl_run_engine_markets_* artifact committed")
        df = pd.read_csv(path)
        self.assertTrue((df["kind"] == "slate").all())
        for col in ("game_id", "gameday", "home_team", "away_team",
                    "mu_h", "mu_a", "fair_spread", "fair_total",
                    "derived_ml", "p_home_win_derived", "p_away_win_derived",
                    "shrink_applied"):
            self.assertIn(col, df.columns)
        # grid columns present per the mapping table
        grid_cols = ([f"p_home_cover_{SE._fname(float(L))}"
                      for L in SE.SPREAD_INT_LINES]
                     + [f"p_push_{SE._fname(float(L))}"
                        for L in SE.SPREAD_INT_LINES]
                     + [f"p_over_{SE._fname(float(U))}"
                        for U in SE.TOTAL_INT_LINES]
                     + [f"p_under_{SE._fname(float(U))}"
                        for U in SE.TOTAL_INT_LINES]
                     + ["fair_spread", "fair_total", "derived_ml",
                        "mu_h", "mu_a"])
        for c in grid_cols:
            self.assertIn(c, df.columns)
        # derived pair complements to 1
        self.assertTrue(np.allclose(df["p_home_win_derived"]
                                    + df["p_away_win_derived"], 1.0))
        # NaN only allowed on offer-level columns (no grid NaNs)
        self.assertFalse(df[grid_cols].isna().any().any())

    def test_monitor_first_run_empty_state(self):
        path = self._latest("nfl_run_engine_monitor_", ".json")
        if path is None:
            self.skipTest("no nfl_run_engine_monitor_* artifact committed")
        data = json.loads(path.read_text())
        # Research-pinned OOF baseline present with provenance.
        base = data["oof_baseline_research_pinned"]
        self.assertEqual(base["covers_ece_pooled"], 0.078)
        self.assertEqual(base["totals_ece_pooled_own"], 0.087)
        self.assertEqual(base["derived_ml"]["logloss"], 0.6365)
        self.assertEqual(base["derived_ml"]["auc"], 0.695)
        # Empty accumulating slate-history — nothing fabricated on the first
        # run (no served-slate outcomes exist yet).
        self.assertEqual(data["slate_history"], [])
        self.assertTrue(data["markets_persisted"])

    def test_meta_json_treatment_flags(self):
        path = self._latest("nfl_run_engine_markets_", ".meta.json")
        if path is None:
            self.skipTest("no meta artifact committed")
        meta = json.loads(path.read_text())
        self.assertFalse(meta["treatment"]["shrink_applied"])
        self.assertEqual(meta["treatment"]["mode"],
                         "own-line quoting both sides with honest ECE")
        self.assertEqual(meta["treatment"]["shrink_params"]["spread_cd"],
                         [0.446165, 0.307486])


if __name__ == "__main__":
    unittest.main()
