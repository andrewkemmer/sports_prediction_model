"""NFL per-side joint layer — step-1 tests (pure, no-network).

Covers ``nfl_joint_engine`` (correlated integer-score joint PMF) and the
step-1 runner guards:
- Grid mass conserved per game < 1e-4 (raw copula AND post-IPF).
- Marginals reproduced after IPF to 1e-9 (the G3 gate).
- rho = 0 reduction: joint factors to the product of the marginals.
- Symmetric game (mu_H == mu_A, sigma_H == sigma_A): P(home) == P(away),
  symmetric joint, derived ML == 0.5, tie mass == the IPF target.
- Tie calibration hits the base rate (G2); D_raw vs calibrated reported.
- P(home covers -L) monotone non-increasing in L; P(over) monotone in U.
- CRPS via two independent methods agrees on synthetic families; the
  degenerate point-mass identity CRPS == |m - x|.
- Sealed leak guard: build_joint_pmfs refuses params not fitted on pooled
  OOF (fit_on marker); fit_joint_params structurally cannot fit on sealed.
- Determinism pin: two identical builds byte-identical (the G5 pin).
- Artifact loud-failure guard: missing columns / wrong row count /
  duplicate game_ids all raise RuntimeError.
- Family selection: near-normal data → DN (data decides; NB does not
  auto-win); rho CI contains rho; sigma params sane.
- FEATURE_COLUMNS untouched; no moneyline import (feature-producer only).

Run: python -m unittest test_nfl_joint_layer -v   (no network needed)
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_features as nf
import nfl_joint_engine as je

GRID = je.GRID


def _synth_pooled(n: int = 220, rho_true: float = 0.3,
                  seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic pooled-OOF residual table (normal-ish scores,
    heteroskedastic residuals, true cross-side correlation rho_true)."""
    rng = np.random.default_rng(seed)
    mu_h = rng.uniform(15, 30, n)
    mu_a = rng.uniform(13, 28, n)
    sig = lambda m: 6.0 * (m / 22.0) ** 0.4
    r_h = rng.normal(0, sig(mu_h))
    r_a = 0.3 * r_h + np.sqrt(1 - 0.09) * rng.normal(0, sig(mu_a))
    r_a = (rho_true / 0.3) * 0.3 * r_h + np.sqrt(
        1 - (rho_true ** 2)) * rng.normal(0, sig(mu_a))
    y_h = np.clip(np.round(mu_h + r_h), 0, 75)
    y_a = np.clip(np.round(mu_a + r_a), 0, 75)
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "fold_idx": np.arange(n) % 88,
        "pred_home": np.round(mu_h, 4),
        "pred_away": np.round(mu_a, 4),
        "resid_home": np.round(y_h - mu_h, 4),
        "resid_away": np.round(y_a - mu_a, 4),
        "best_iter_home": 30, "best_iter_away": 30,
        "home_score": y_h, "away_score": y_a,
    })


def _params(pooled: pd.DataFrame) -> dict:
    return je.fit_joint_params(pooled)


def _game_marginals(mu_h: float, mu_a: float, params: dict):
    sig_h = je.sigma_callable(params["sigma_h"])(mu_h)
    sig_a = je.sigma_callable(params["sigma_a"])(mu_a)
    return (je.marginal_pmf(mu_h, sig_h, params["family"]),
            je.marginal_pmf(mu_a, sig_a, params["family"]))


class JointEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pooled = _synth_pooled()
        self.params = _params(self.pooled)

    # ── grid mass + IPF marginals (G3) ────────────────────────────────────

    def test_grid_mass_conserved_raw(self) -> None:
        for mu_h, mu_a in [(22.5, 20.0), (10.0, 30.0), (30.0, 12.0)]:
            J = je.joint_pmf_copula(mu_h, mu_a, self.params)
            self.assertLess(abs(float(J.sum()) - 1.0), 1e-4)

    def test_grid_mass_conserved_calibrated(self) -> None:
        p_tie = 0.004
        pmfs, _summary = je.build_joint_pmfs(self.pooled, self.params, p_tie)
        for J in pmfs[:20]:
            self.assertLess(abs(float(J.sum()) - 1.0), 1e-4)

    def test_ipf_preserves_marginals_1e9(self) -> None:
        p_tie = 0.004
        pmfs, summary = je.build_joint_pmfs(self.pooled, self.params, p_tie)
        err = summary["summary"]["max_marginal_err_post_ipf"]
        self.assertIsNotNone(err)
        self.assertLessEqual(err, 1e-9)

    # ── rho = 0 factorization ─────────────────────────────────────────────

    def test_rho_zero_factors(self) -> None:
        p0 = dict(self.params, rho=0.0)
        for mu_h, mu_a in [(22.5, 20.0), (12.0, 25.0)]:
            J = je.joint_pmf_copula(mu_h, mu_a, p0)
            mh, ma = _game_marginals(mu_h, mu_a, p0)
            np.testing.assert_allclose(J, np.outer(mh, ma), atol=1e-9)

    # ── symmetric game ────────────────────────────────────────────────────

    def test_symmetric_game(self) -> None:
        # Force EXACTLY equal sigma on both sides (fitted sigma_h != sigma_a
        # by a hair, which would break the symmetry premise).
        same_sig = {"spec": "const", "sigma0": 6.0, "q": 0.0}
        p_sym = dict(self.params, sigma_h=same_sig, sigma_a=same_sig)
        mu = 22.0
        sig = je.sigma_callable(p_sym["sigma_h"])(mu)
        J = je.joint_pmf_copula(mu, mu, p_sym)
        np.testing.assert_allclose(J, J.T, atol=1e-9)    # symmetric tie mass
        m = je.marginal_pmf(mu, sig, p_sym["family"])
        Jc = je.calibrate_tie_diagonal(J, m, m, p_tie=0.004)
        d = je.derived_from_joint(Jc)
        self.assertAlmostEqual(d["p_home_win"], d["p_away_win"], places=4)
        self.assertAlmostEqual(d["derived_ml"], 0.5, places=4)
        self.assertAlmostEqual(d["p_tie"], 0.004, places=6)

    # ── tie calibration (G2) ──────────────────────────────────────────────

    def test_tie_calibration_hits_base_rate(self) -> None:
        p_tie = 0.004
        pmfs, summary = je.build_joint_pmfs(self.pooled, self.params, p_tie)
        means = np.mean([np.trace(p) for p in pmfs])
        self.assertAlmostEqual(means, p_tie, places=6)
        self.assertGreater(summary["summary"]["d_raw_mean"],
                           summary["summary"]["d_calibrated_mean"])
        self.assertAlmostEqual(summary["summary"]["d_calibrated_mean"],
                               p_tie, places=6)

    # ── monotonicity of derived cover/over probabilities ──────────────────

    def test_cover_over_monotone(self) -> None:
        J = je.joint_pmf_copula(22.5, 20.0, self.params)
        mp = je.margin_pmf_from_joint(J)
        tp = je.total_pmf_from_joint(J)
        ls = [1.5, 3.5, 6.5, 10.5, 14.5]
        covs = [je.cover_prob(mp, L) for L in ls]
        self.assertEqual(covs, sorted(covs, reverse=True))
        us = [41.5, 45.5, 49.5, 53.5]
        overs = [je.over_prob(tp, U) for U in us]
        self.assertEqual(overs, sorted(overs, reverse=True))

    # ── CRPS ──────────────────────────────────────────────────────────────

    def test_crps_two_methods_agree(self) -> None:
        cases = [je.dn_pmf(22.0, 6.0), je.dn_pmf(10.0, 3.0),
                 je.nb_pmf(20.0, 7.0)]
        for pmf in cases:
            for x in (14, 22, 27):
                a = je.crps_discrete(pmf, x)
                b = je.crps_discrete_pairs(pmf, x)
                self.assertLess(abs(a - b), 1e-9)

    def test_crps_degenerate_identity(self) -> None:
        for m in (10, 20, 45):
            pmf = np.zeros(76)
            pmf[m] = 1.0
            for x in (5, m, 60):
                self.assertAlmostEqual(je.crps_discrete(pmf, x), abs(m - x),
                                       places=9)

    # ── sealed leak guard ─────────────────────────────────────────────────

    def test_sealed_leak_guard(self) -> None:
        bad = dict(self.params, fit_on="sealed_2025")
        with self.assertRaises(ValueError):
            je.build_joint_pmfs(self.pooled, bad, p_tie=0.004)
        good = dict(self.params, fit_on="pooled_oof")
        pmfs, _s = je.build_joint_pmfs(self.pooled, good, p_tie=0.004)
        self.assertEqual(len(pmfs), len(self.pooled))
        self.assertEqual(self.params["fit_on"], "pooled_oof")
        # fit_joint_params takes only a residual table (no y args) — the API
        # is structurally incapable of fitting on sealed rows.
        import inspect
        sig = inspect.signature(je.fit_joint_params)
        self.assertEqual(list(sig.parameters), ["pooled"])

    # ── determinism (G5) ──────────────────────────────────────────────────

    def test_determinism_byte_identical(self) -> None:
        pmfs1, s1 = je.build_joint_pmfs(self.pooled, self.params, 0.004)
        pmfs2, s2 = je.build_joint_pmfs(self.pooled, self.params, 0.004)
        self.assertEqual(s1["derived"].to_csv(index=False),
                         s2["derived"].to_csv(index=False))
        np.testing.assert_array_equal(pmfs1, pmfs2)

    # ── artifact loud-failure guard ───────────────────────────────────────

    def test_artifact_loud_failure(self, tmp_path="/tmp") -> None:
        bad_cols = self.pooled.drop(columns=["resid_away"])
        p1 = Path(tmp_path) / "j_art_badcols.csv"
        bad_cols.to_csv(p1, index=False)
        with self.assertRaises(RuntimeError):
            je.load_residual_artifact(p1)
        # wrong row count
        p2 = Path(tmp_path) / "j_art_wrongn.csv"
        self.pooled.head(50).to_csv(p2, index=False)
        with self.assertRaises(RuntimeError):
            je.load_residual_artifact(p2, expected_n=je.ARTIFACT_N_EXPECTED)
        # duplicate game_ids
        dup = pd.concat([self.pooled.head(3), self.pooled.head(1)],
                        ignore_index=True)
        p3 = Path(tmp_path) / "j_art_dup.csv"
        dup.to_csv(p3, index=False)
        with self.assertRaises(RuntimeError):
            je.load_residual_artifact(p3, expected_n=len(dup))
        # empty frame
        p4 = Path(tmp_path) / "j_art_empty.csv"
        self.pooled.head(0).to_csv(p4, index=False)
        with self.assertRaises(RuntimeError):
            je.load_residual_artifact(p4)

    # ── family / sigma / rho selection ────────────────────────────────────

    def test_family_selection_near_normal_data(self) -> None:
        # Synthetic near-normal scores → DN should win (data decides; NB does
        # not auto-win).
        self.assertEqual(self.params["family"], "dn")

    def test_sigma_params_sane(self) -> None:
        for side in ("sigma_h", "sigma_a"):
            spec = self.params[side]
            self.assertIn(spec["spec"], ("power", "const"))
            self.assertGreater(spec["sigma0"], 0.0)
            if spec["spec"] == "power":
                self.assertGreater(spec["q"], -0.5)
                self.assertLess(spec["q"], 1.5)

    def test_rho_ci_contains_rho(self) -> None:
        self.assertLessEqual(self.params["rho_ci"]["low"], self.params["rho"])
        self.assertGreaterEqual(self.params["rho_ci"]["high"],
                                self.params["rho"])
        # true rho = 0.3 on 220 pairs → CI should comfortably contain it
        self.assertGreater(self.params["rho"], 0.1)

    def test_integer_ll_peaks_at_truth(self) -> None:
        y = 22
        ll_true = je.integer_ll(je.dn_pmf(22.0, 6.0), y)
        ll_far = je.integer_ll(je.dn_pmf(30.0, 6.0), y)
        self.assertGreater(ll_true, ll_far)

    # ── pins ──────────────────────────────────────────────────────────────

    def test_no_moneyline_import(self) -> None:
        src = Path(je.__file__).read_text()
        self.assertNotIn("nfl_moneyline", src)
        self.assertNotIn("nfl_margin_engine", src)

    def test_feature_columns_untouched(self) -> None:
        feats_before = list(nf.FEATURE_COLUMNS)
        self.assertEqual(list(nf.FEATURE_COLUMNS), feats_before)
        # engine never mutates the served pool
        src = Path(je.__file__).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == "FEATURE_COLUMNS":
                        self.fail("nfl_joint_engine mutates FEATURE_COLUMNS")


if __name__ == "__main__":
    unittest.main()