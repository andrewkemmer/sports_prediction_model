"""NFL market layer — regression pins (pure, no-network).

Pins the market layer's claims (nfl_market_engine.py) without touching the
network, /tmp era dumps, or the joint walk:

 1. Leak-safety of the second-level walk-forward: fold k's (c_k, d_k) fit
    set is the val rows of STRICTLY-prior folds (weeks < k) only; every
    pooled row is assigned its own fold's (c, d).
 2. Warmup no-shrink: folds with < min_prior_rows prior rows get d=1, c=0
    -> the shrink arm is IDENTICAL to the own-line arm on those rows
    (never fabricate shrinkage).
 3. delta/2 margin-invariance: shifting both means by delta/2 leaves the
    margin PMF / cover / tie / derived-ML unchanged (the spread side stays
    untouched); negative control: shifting only ONE side moves the margin
    materially.
 4. Determinism: two identical arm builds byte-identical (G4).
 5. Median-of-fold arithmetic: sealed transfer coefficients are the median
    over FITTED folds only (warmup folds excluded).
 6. Market-frame Step-0 asserts: every row needs mu + offered line; a
    missing line raises.
 7. House pins: no FEATURE_COLUMNS mutation / no nfl_moneyline import in
    the engine source; engine files byte-identical to the committed
    re-baseline record's hashes (5eb7d5c).
 8. Emitter contract: market_record_table emits the full honest-ECE column
    set.
 9. fair_total = discrete median of the total PMF.

Run: python -m unittest test_nfl_market -v   (no network needed)
"""
from __future__ import annotations

import ast
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_joint_engine as je  # noqa: E402
import nfl_market_engine as M  # noqa: E402

# Joint params for the pure tests (const sigma / rho 0 / pooled-OOF marker).
PARAMS = {
    "family": "dn",
    "sigma_h": {"spec": "const", "sigma0": 9.0, "q": 0.0},
    "sigma_a": {"spec": "const", "sigma0": 9.0, "q": 0.0},
    "rho": 0.0,
    "fit_on": "pooled_oof",
    "grid_max": je.GRID_MAX,
}
P_TIE = 0.005


def _synth_market(n_weeks: int = 6, rows_per_week: int = 12,
                  seed: int = 7, prefix: str = "") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, n_weeks + 1):
        for _ in range(rows_per_week):
            mu_h = float(rng.uniform(20, 27))
            mu_a = float(rng.uniform(16, 24))
            line = float(round(rng.uniform(42, 50), 1))
            y_h = float(np.clip(round(rng.normal(mu_h, 9.0)), 0, 75))
            y_a = float(np.clip(round(rng.normal(mu_a, 9.0)), 0, 75))
            rows.append({
                "game_id": f"{prefix}g{w}_{len(rows)}", "week_start": w,
                "season": 2021,
                "pred_home": round(mu_h, 4), "pred_away": round(mu_a, 4),
                "mu_T_hat": round(mu_h + mu_a, 4),
                "total_line": line, "spread_line": round(mu_h - mu_a, 1),
                "total": y_h + y_a, "margin": y_h - y_a,
                "home_score": y_h, "away_score": y_a,
            })
    return pd.DataFrame(rows)


class LeakSafetyTests(unittest.TestCase):
    # ── 1. second-level walk-forward leak safety ───────────────────────────

    def test_fold_fit_set_is_strictly_prior(self) -> None:
        mk = _synth_market(n_weeks=6, rows_per_week=12)
        walk = M.fit_fold_disciplined_cd(mk, min_prior_rows=20)
        self.assertTrue(walk["leak_safe"])
        self.assertEqual(walk["n_folds"], 6)
        for f in walk["folds"]:
            prior_n = int((mk["week_start"] < f["week_start"]).sum())
            self.assertEqual(f["n_prior"], prior_n)
            if not f["warmup"]:
                prior_weeks = mk.loc[mk["week_start"] < f["week_start"],
                                     "week_start"]
                self.assertTrue((prior_weeks < f["week_start"]).all())
        # every row assigned its own fold's (c, d)
        for _, r in mk.iterrows():
            f = next(x for x in walk["folds"]
                     if x["week_start"] == r["week_start"])
            self.assertEqual(walk["used_cd"][r["game_id"]], (f["c"], f["d"]))

    def test_warmup_folds_are_first_and_no_shrink(self) -> None:
        mk = _synth_market(n_weeks=4, rows_per_week=10)  # prior 0/10/20/30
        walk = M.fit_fold_disciplined_cd(mk, min_prior_rows=25)
        # weeks 1-3 (0/10/20 prior rows) are warmup; week 4 (30) is fitted
        warm = [f for f in walk["folds"] if f["warmup"]]
        self.assertEqual(len(warm), 3)
        for f in warm:
            self.assertEqual((f["c"], f["d"]), (0.0, 1.0))
        self.assertEqual(walk["n_fitted"], 1)


class WarmupNoShrinkTests(unittest.TestCase):
    # ── 2. warmup (d=1, c=0) => shrink arm == own-line arm exactly ─────────

    def test_warmup_shrink_identical_to_own(self) -> None:
        mk = _synth_market(n_weeks=3, rows_per_week=10)  # 30 rows < 50
        walk = M.fit_fold_disciplined_cd(mk, min_prior_rows=50)
        self.assertEqual(walk["n_warmup"], 3)
        cd = {f["week_start"]: (f["c"], f["d"]) for f in walk["folds"]}
        own = M.build_arm(mk, PARAMS, P_TIE, "none")
        shr = M.build_arm(mk, PARAMS, P_TIE, "fold", cd_by_week=cd)
        for col in ("fair_total", "p_over", "p_cover", "derived_ml",
                    "p_home_win", "p_away_win", "p_tie"):
            pd.testing.assert_series_equal(own[col], shr[col],
                                           check_names=False)


class MarginInvarianceTests(unittest.TestCase):
    # ── 3. delta/2 keeps the margin side untouched ─────────────────────────

    def test_delta_half_shift_leaves_margin_invariant(self) -> None:
        mu_h, mu_a, delta = 24.0, 20.0, 2.0
        J0 = je.joint_pmf_copula(mu_h, mu_a, PARAMS)
        J1 = je.joint_pmf_copula(mu_h + delta / 2, mu_a + delta / 2, PARAMS)
        m0, m1 = je.margin_pmf_from_joint(J0), je.margin_pmf_from_joint(J1)
        self.assertLess(float(np.max(np.abs(m0 - m1))), 5e-3)
        # the margin CENTER (mean) is the discriminating invariant: delta/2
        # cancels, so E[margin] moves by ~0 (residual is the DN clamp),
        # whereas a one-sided shift moves it by ~delta (see negative
        # control).
        grid = np.arange(-75, 76)
        d_mean = abs(float((grid * m1).sum()) - float((grid * m0).sum()))
        self.assertLess(d_mean, 0.05)
        d0, d1 = je.derived_from_joint(J0), je.derived_from_joint(J1)
        self.assertLess(abs(d0["p_tie"] - d1["p_tie"]), 5e-3)
        self.assertLess(abs(d0["p_home_win"] - d1["p_home_win"]), 5e-3)
        self.assertLess(abs(d0["derived_ml"] - d1["derived_ml"]), 5e-3)
        self.assertLess(abs(je.cover_prob(m0, -3.5)
                            - je.cover_prob(m1, -3.5)), 5e-3)

    def test_negative_control_one_sided_shift_moves_margin(self) -> None:
        # shifting only ONE side moves the margin center by ~delta — the
        # invariance is specific to the delta/2 construction.
        mu_h, mu_a, delta = 24.0, 20.0, 2.0
        J0 = je.joint_pmf_copula(mu_h, mu_a, PARAMS)
        J1 = je.joint_pmf_copula(mu_h + delta, mu_a, PARAMS)  # wrong arm
        m0, m1 = je.margin_pmf_from_joint(J0), je.margin_pmf_from_joint(J1)
        grid = np.arange(-75, 76)
        d_mean = abs(float((grid * m1).sum()) - float((grid * m0).sum()))
        self.assertGreater(d_mean, 1.0)  # ~2.0 measured


class DeterminismTests(unittest.TestCase):
    # ── 4. byte-identical double build (G4) ────────────────────────────────

    def test_determinism_byte_identical(self) -> None:
        mk = _synth_market(n_weeks=4, rows_per_week=8, seed=3)
        walk = M.fit_fold_disciplined_cd(mk, min_prior_rows=12)
        cd = {f["week_start"]: (f["c"], f["d"]) for f in walk["folds"]}
        a1 = M.market_record_table(
            M.build_arm(mk, PARAMS, P_TIE, "none"),
            M.build_arm(mk, PARAMS, P_TIE, "fold", cd_by_week=cd)).to_csv(
            index=False)
        a2 = M.market_record_table(
            M.build_arm(mk, PARAMS, P_TIE, "none"),
            M.build_arm(mk, PARAMS, P_TIE, "fold", cd_by_week=cd)).to_csv(
            index=False)
        self.assertEqual(a1, a2)


class MedianOfFoldTests(unittest.TestCase):
    # ── 5. sealed transfer = median over FITTED folds only ─────────────────

    def test_median_excludes_warmup_folds(self) -> None:
        mk = _synth_market(n_weeks=4, rows_per_week=10, seed=11)
        walk = M.fit_fold_disciplined_cd(mk, min_prior_rows=25)
        fitted = [f for f in walk["folds"] if not f["warmup"]]
        self.assertEqual(walk["n_warmup"], 3)
        self.assertEqual(walk["n_fitted"], 1)
        self.assertAlmostEqual(
            walk["median_c"], float(np.median([f["c"] for f in fitted])))
        self.assertAlmostEqual(
            walk["median_d"], float(np.median([f["d"] for f in fitted])))


class MarketFrameTests(unittest.TestCase):
    # ── 6. Step-0 asserts: mu + offered line on every row ──────────────────

    def test_missing_line_raises(self) -> None:
        pooled = _synth_market(n_weeks=2, rows_per_week=5)
        sealed = _synth_market(n_weeks=1, rows_per_week=3, seed=2,
                               prefix="s")
        feats = pd.concat([pooled[["game_id", "season"]],
                           sealed[["game_id", "season"]]],
                          ignore_index=True)
        week_map = {r["game_id"]: r["week_start"]
                    for _, r in pooled.iterrows()}
        lines = pooled[["game_id", "spread_line", "total_line"]].copy()
        lines.loc[lines.index[0], "total_line"] = np.nan
        with self.assertRaises(RuntimeError):
            M.build_market_frame(pooled, sealed, feats, week_map, lines)

    def test_happy_path(self) -> None:
        pooled = _synth_market(n_weeks=2, rows_per_week=5)
        sealed = _synth_market(n_weeks=1, rows_per_week=3, seed=2,
                               prefix="s")
        feats = pd.concat([pooled[["game_id", "season"]],
                           sealed[["game_id", "season"]]],
                          ignore_index=True)
        week_map = {r["game_id"]: r["week_start"]
                    for _, r in pooled.iterrows()}
        lines = pd.concat([pooled[["game_id", "spread_line", "total_line"]],
                           sealed[["game_id", "spread_line",
                                   "total_line"]]], ignore_index=True)
        mp, ms = M.build_market_frame(pooled, sealed, feats, week_map, lines)
        self.assertEqual(len(mp), len(pooled))
        self.assertEqual(len(ms), len(sealed))
        self.assertTrue(mp[["pred_home", "pred_away", "spread_line",
                            "total_line"]].notna().all().all())
        self.assertTrue(ms[["pred_home", "pred_away", "spread_line",
                            "total_line"]].notna().all().all())


class HousePinTests(unittest.TestCase):
    # ── 7. FEATURE_COLUMNS / no-moneyline / engine byte-identity pins ──────

    def test_no_feature_columns_mutation_and_no_moneyline_import(self) -> None:
        src = Path(M.__file__).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) \
                    else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == "FEATURE_COLUMNS":
                        self.fail("nfl_market_engine mutates FEATURE_COLUMNS")
        self.assertNotIn("nfl_moneyline", src)

    def test_engine_byte_identity_pin(self) -> None:
        # Engine files must be byte-identical to the committed re-baseline
        # record's hashes (commit 5eb7d5c) — the market layer runs against
        # unchanged committed engines.
        rec = json_load_rebaseline()
        hashes = rec["geometry"]["engine_files_sha256"]
        for name, expected in hashes.items():
            path = Path(__file__).resolve().parent / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            self.assertEqual(actual, expected, f"engine changed: {name}")


class EmitterTests(unittest.TestCase):
    # ── 8. per-game market record emitter contract ─────────────────────────

    def test_emitter_columns(self) -> None:
        mk = _synth_market(n_weeks=3, rows_per_week=6, seed=5)
        walk = M.fit_fold_disciplined_cd(mk, min_prior_rows=12)
        cd = {f["week_start"]: (f["c"], f["d"]) for f in walk["folds"]}
        own = M.build_arm(mk, PARAMS, P_TIE, "none")
        shr = M.build_arm(mk, PARAMS, P_TIE, "fold", cd_by_week=cd)
        tbl = M.market_record_table(own, shr)
        self.assertEqual(list(tbl.columns), M.MARKET_RECORD_COLUMNS)
        self.assertEqual(len(tbl), len(mk))
        self.assertTrue(tbl[["p_over_own", "p_over_shrunk",
                             "p_cover_own", "p_cover_shrunk",
                             "derived_ml_own", "derived_ml_shrunk",
                             "fair_total_own", "fair_total_shrunk",
                             "y_over", "y_cover", "y_home_win",
                             "used_c", "used_d"]].notna().all().all())


class FairTotalTests(unittest.TestCase):
    # ── 9. fair total = discrete median of the total PMF ───────────────────

    def test_fair_total_median(self) -> None:
        J = je.joint_pmf_copula(24.0, 23.0, PARAMS)  # E[total] ~ 47
        tot = je.total_pmf_from_joint(J)
        ft = M.fair_total(tot)
        self.assertGreaterEqual(ft, 45)
        self.assertLessEqual(ft, 49)
        # median property: CDF(ft-1) < 0.5 <= CDF(ft)
        cdf = np.cumsum(tot)
        self.assertLess(cdf[int(ft) - 1], 0.5)
        self.assertGreaterEqual(cdf[int(ft)], 0.5)


def json_load_rebaseline() -> dict:
    path = (Path(__file__).resolve().parent.parent / "data_delivery"
            / "nfl_joint_rebaseline_3e8c8a510f04.json")
    if not path.exists():
        raise RuntimeError(f"re-baseline record missing: {path} "
                           "(engine byte-identity pin needs it)")
    import json
    return json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)