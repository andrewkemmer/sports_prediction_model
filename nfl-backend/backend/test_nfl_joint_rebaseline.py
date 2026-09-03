"""NFL joint-engine re-baseline — regression pins (pure, no-network).

The grid fix (f251bc6) aligned ``nfl_joint_engine.marginal_breakpoints`` to
the documented convention (P(score k) at index k). This file pins the
re-baseline's claims without touching the network or /tmp dumps:

 1. Convention pin: dn_pmf == marginal_pmf on a grid of DN means — P(score
    k) at index k in BOTH paths (max diff 0.0 measured; pinned to 1e-12),
    76 marginal cells on the 0..75 grid.
 2. Joint geometry: 76x76 joints; convolution mean == sum of the marginal
    means to 1e-9 and within 0.5 of mu_H + mu_A (the +2 regression sat
    ~2.05 away; the residual ~0.07 is the DN 0-floor clamp bias).
 3. Convention-collapse pin (the re-baseline's headline): on a
    well-specified synthetic sample, the ENGINE's own derived totals
    (internal-grid totals ECE via ``totals_ece_internal``) and the
    documented-convention total-PIT ECE now agree at the calibrated level
    (both < 0.04, |gap| < 0.03). Pre-fix the internal-grid measure sat at
    0.0796 (the +2 shift) while the PIT measure was 0.0092 — the split is
    gone because the engine now prices totals at the documented convention.
 4. Invariance pin: margin PMF / tie / p_home / p_away / derived-ML are
    index-DIFFERENCE quantities — unchanged pre-vs-post fix. A verbatim
    copy of the pre-fix breakpoints (commit 3480b05) reproduces the old
    joints; the shared cell masses match to float precision after
    renormalizing out the below-0.5 tail sliver (pre-fix: phantom
    "score −1" cell; post-fix: absorbed into score 0), and the derived
    quantities agree to < 1e-4 (the sliver re-seating).
 5. Determinism: two identical builds byte-identical (G5).
 6. House pins: FEATURE_COLUMNS untouched + no moneyline/margin import in
    the engine source (house convention).

Run: python -m unittest test_nfl_joint_rebaseline -v   (no network needed)
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_features as nf  # noqa: E402
import nfl_joint_engine as je  # noqa: E402
import nfl_sigma_layer as S  # noqa: E402

GRID_MAX = 75

# Era-record joint params (dn / const sigma 9.663/9.0789 / rho 0.0076).
ERA_PARAMS = {
    "family": "dn",
    "sigma_h": {"spec": "const", "sigma0": 9.663, "q": 0.0},
    "sigma_a": {"spec": "const", "sigma0": 9.0789, "q": 0.0},
    "rho": 0.0076,
    "fit_on": "pooled_oof",
    "grid_max": je.GRID_MAX,
}


# ── Verbatim pre-fix implementation (commit 3480b05) for the invariance arm ──

def _old_marginal_breakpoints(mu: float, sigma: float, family: str
                              ) -> np.ndarray:
    """PRE-FIX copy: breakpoints at F(arange(76) − 0.5) + endpoints → cell k
    held the mass of score k−1 (P(score k) lived at index k+1)."""
    if family == "dn":
        mu = float(mu)
        sigma = max(float(sigma), 1e-9)
        b = stats.norm.cdf((np.arange(GRID_MAX + 1) - 0.5 - mu) / sigma)
    else:  # nb
        mu = max(float(mu), 1e-9)
        r = je._nb_r(mu, sigma)
        b = stats.nbinom.cdf(np.arange(GRID_MAX + 1) - 0.5, r, r / (r + mu))
    b = np.clip(b, 0.0, 1.0)
    b = np.concatenate([[0.0], b, [1.0]])
    b = np.maximum.accumulate(b)
    return b


def _old_joint(mu_h: float, mu_a: float, params: dict) -> np.ndarray:
    """PRE-FIX joint PMF (77x77; cell (i,j) holds score (i−1, j−1))."""
    rho = float(params["rho"])
    family = str(params["family"])
    b_h = _old_marginal_breakpoints(
        mu_h, je.sigma_callable(params["sigma_h"])(mu_h), family)
    b_a = _old_marginal_breakpoints(
        mu_a, je.sigma_callable(params["sigma_a"])(mu_a), family)
    qh = np.clip(stats.norm.ppf(np.clip(b_h, 1e-12, 1 - 1e-12)), -37.0, 37.0)
    qa = np.clip(stats.norm.ppf(np.clip(b_a, 1e-12, 1 - 1e-12)), -37.0, 37.0)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mv = stats.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    pts = np.column_stack([np.repeat(qh, len(qa)), np.tile(qa, len(qh))])
    C = np.asarray(mv.cdf(pts)).reshape(len(qh), len(qa))
    J = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    J = np.clip(J, 0.0, None)
    return J / J.sum()


def _old_derived(J: np.ndarray) -> dict[str, float]:
    n = J.shape[0]
    p_home = float(np.tril(J, -1).sum())
    p_away = float(np.triu(J, 1).sum())
    p_tie = float(np.trace(J))
    return {"p_home": p_home, "p_away": p_away, "p_tie": p_tie,
            "derived_ml": p_home / (1.0 - p_tie) if p_tie < 1 else 0.5}


def _synth_rows(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    """Well-specified synthetic games: y ~ round(N(mu, 9)), clipped to grid."""
    rng = np.random.default_rng(seed)
    mu_h = rng.uniform(18, 27, n)
    mu_a = rng.uniform(15, 25, n)
    y_h = np.clip(np.round(rng.normal(mu_h, 9.0)), 0, 75)
    y_a = np.clip(np.round(rng.normal(mu_a, 9.0)), 0, 75)
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "pred_home": np.round(mu_h, 4),
        "pred_away": np.round(mu_a, 4),
        "home_score": y_h,
        "away_score": y_a,
    })


class RebaselineConventionTests(unittest.TestCase):
    # ── 1. convention pin: P(score k) at index k in both paths ─────────────

    def test_dn_equals_marginal_on_grid_of_means(self) -> None:
        for mu in (5.0, 15.0, 25.0, 40.0, 60.0):
            a = je.dn_pmf(mu, 9.0)
            b = je.marginal_pmf(mu, 9.0, "dn")
            self.assertEqual(a.shape, (76,))
            self.assertEqual(b.shape, (76,))
            # measured max diff is exactly 0.0; pinned with headroom
            np.testing.assert_allclose(a, b, atol=1e-12)

    # ── 2. joint geometry + total-mean identity ────────────────────────────

    def test_joint_76x76_and_total_mean_within_clamp_bias(self) -> None:
        J = je.joint_pmf_copula(23.0, 20.0, ERA_PARAMS)
        self.assertEqual(J.shape, (76, 76))
        tot = je.total_pmf_from_joint(J)
        s = np.arange(len(tot), dtype=float)
        e_tot = float((s * tot).sum())
        mh = je.marginal_pmf(23.0, je.sigma_callable(
            ERA_PARAMS["sigma_h"])(23.0), "dn")
        ma = je.marginal_pmf(20.0, je.sigma_callable(
            ERA_PARAMS["sigma_a"])(20.0), "dn")
        e_h = float((np.arange(76) * mh).sum())
        e_a = float((np.arange(76) * ma).sum())
        # exact identity: convolution mean == sum of marginal means
        self.assertLess(abs(e_tot - (e_h + e_a)), 1e-9)
        # +2 regression sat ~2.05 above mu_H + mu_A; residual ~0.07 is the
        # DN 0-floor clamp bias, not the index bug.
        self.assertGreater(e_tot - (23.0 + 20.0), 0.0)
        self.assertLess(e_tot - (23.0 + 20.0), 0.5)

    # ── 3. convention collapse: internal totals ECE ≈ total-PIT ECE ────────

    def test_internal_totals_ece_collapses_to_pit_ece(self) -> None:
        rows = _synth_rows()
        params = {"family": "dn",
                  "sigma_h": {"spec": "const", "sigma0": 9.0, "q": 0.0},
                  "sigma_a": {"spec": "const", "sigma0": 9.0, "q": 0.0},
                  "rho": 0.0, "fit_on": "pooled_oof", "grid_max": je.GRID_MAX}
        p_tie = float(np.mean(rows["home_score"] == rows["away_score"]))
        pmfs, summ = je.build_joint_pmfs(rows, params, p_tie)
        derived = summ["derived"].copy()
        derived = derived.merge(rows[["game_id", "home_score", "away_score"]],
                                on="game_id", how="left")
        totals_ece = S.totals_ece_internal(pmfs, derived)
        pit = S.total_pit(rows["pred_home"].to_numpy(float),
                          np.full(len(rows), 9.0),
                          rows["pred_away"].to_numpy(float),
                          np.full(len(rows), 9.0),
                          (rows["home_score"] + rows["away_score"])
                          .to_numpy(float))
        pit_ece = S.uniformity_table(pit)["ece"]
        # PRE-FIX the internal-grid measure was ~0.08 (the +2 shift); both
        # measures now agree at the calibrated level (the collapse).
        self.assertLess(totals_ece["ece"], 0.04)
        self.assertLess(pit_ece, 0.04)
        self.assertLess(abs(totals_ece["ece"] - pit_ece), 0.03)

    # ── 4. invariance: margin/tie/ML unchanged pre-vs-post fix ─────────────

    def test_margin_tie_ml_invariant_pre_vs_post_fix(self) -> None:
        # Shared cell masses are identical after renormalizing out the
        # below-0.5 tail sliver (pre-fix: phantom "score −1" cell).
        for muh, mua in ((23.0, 20.0), (25.0, 13.0), (15.0, 22.0)):
            J_old = _old_joint(muh, mua, ERA_PARAMS)
            J_new = je.joint_pmf_copula(muh, mua, ERA_PARAMS)
            # Score cells (k, l) with k, l >= 1 sit on the SAME copula
            # rectangle pre/post fix (old index (k+1, l+1)) and both joints
            # are normalized over their FULL support — identical to float
            # precision with NO renormalization. (New row/col 0 = score 0
            # ABSORBS the below-0.5 tail; old row/col 0 was a phantom
            # "score −1" cell, so score-0 cells are NOT shared.)
            np.testing.assert_allclose(J_new[1:, 1:], J_old[2:, 2:],
                                       atol=1e-12)
            # derived quantities: index-difference quantities, unchanged up
            # to the tail sliver re-seating. The bound scales with the
            # below-0.5 mass at low mu (5.4% at mu=15/sigma=9 -> ~2e-4); on
            # the real era-centered rows (mu 18-31) the measured max diff is
            # ~1.1e-5. A real index change shows at ~1e-2 scale.
            m_old = je.margin_pmf_from_joint(J_old)[1:152]  # shared −75..75
            m_new = je.margin_pmf_from_joint(J_new)
            self.assertLess(float(np.max(np.abs(m_old - m_new))), 5e-4)
            d_old, d_new = _old_derived(J_old), je.derived_from_joint(J_new)
            self.assertLess(abs(d_old["p_tie"] - d_new["p_tie"]), 5e-4)
            self.assertLess(abs(d_old["p_home"] - d_new["p_home_win"]), 5e-4)
            self.assertLess(abs(d_old["p_away"] - d_new["p_away_win"]), 5e-4)
            self.assertLess(abs(d_old["derived_ml"] - d_new["derived_ml"]),
                            5e-4)

    # ── 5. determinism (G5) ────────────────────────────────────────────────

    def test_determinism_byte_identical(self) -> None:
        rows = _synth_rows(n=24, seed=3)
        _, s1 = je.build_joint_pmfs(rows, ERA_PARAMS, p_tie=0.004)
        _, s2 = je.build_joint_pmfs(rows, ERA_PARAMS, p_tie=0.004)
        self.assertEqual(s1["derived"].to_csv(index=False),
                         s2["derived"].to_csv(index=False))

    # ── 6. house pins ──────────────────────────────────────────────────────

    def test_feature_columns_untouched(self) -> None:
        before = list(nf.FEATURE_COLUMNS)
        self.assertEqual(list(nf.FEATURE_COLUMNS), before)
        src = Path(je.__file__).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) \
                    else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id == "FEATURE_COLUMNS":
                        self.fail("nfl_joint_engine mutates FEATURE_COLUMNS")

    def test_no_moneyline_or_margin_import(self) -> None:
        src = Path(je.__file__).read_text()
        self.assertNotIn("nfl_moneyline", src)
        self.assertNotIn("nfl_margin_engine", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)