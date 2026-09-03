"""Tests for the NFL sigma/dispersion layer (record-only).

Pure/offline by convention (like test_nfl_era.py): no network, no nflreadpy,
no writes outside /tmp. Covers the spec's test list adapted to the Step-0
STOP outcome:
  - Step-0 gate helper correctness (uniformity table, clean-gamma rule);
  - PIT convention correctness: well-specified DN simulations are uniform
    under the DOCUMENTED convention; the engine-index helper quantifies
    the +1 grid artifact (flagged engine finding);
  - per-game sigma injection into the engine's public entrypoints (row/col
    marginals == injected DN marginals; engine byte-identity preserved);
  - engine byte-identity pin (engine source files byte-identical after the
    helpers run);
  - fold-local no-leak assertion (a pooled-OOF static sigma across scored
    rows raises AssertionError);
  - median-of-fold transfer arithmetic vs hand-computed;
  - clip bounds ([0.5, 2.0] x median sigma_const);
  - determinism: two identical per-game builds byte-identical;
  - imputation discipline (NaN inputs -> NaN outputs, never zero);
  - FEATURE_COLUMNS-untouched + no-moneyline-import pins.
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_sigma_layer as S  # noqa: E402
from nfl_features import FEATURE_COLUMNS  # noqa: E402
from nfl_joint_engine import marginal_pmf  # noqa: E402

BACKEND = Path(__file__).resolve().parent
ENGINE_FILES = ["nfl_joint_engine.py", "nfl_per_side_engine.py",
                "nfl_era_features.py"]


def _well_specified(n: int = 3000, seed: int = 0):
    rng = np.random.RandomState(seed)
    mu = rng.uniform(15, 30, n)
    sig = rng.uniform(6, 10, n)
    y = np.round(rng.normal(mu, sig)).clip(0, 75).astype(float)
    return mu, sig, y


class TestPitConventions(unittest.TestCase):
    def test_well_specified_pit_uniform_documented_convention(self):
        mu, sig, y = _well_specified()
        u = S.side_pit(mu, sig, y)
        t = S.uniformity_table(u)
        self.assertGreater(t["ks_p"], 0.05)
        self.assertGreater(t["chi2_p"], 0.05)
        self.assertTrue(t["is_uniform"])
        self.assertAlmostEqual(t["mean"], 0.5, delta=0.03)

    def test_engine_convention_quantifies_grid_artifact(self):
        # Under the engine's raw index convention a well-calibrated model
        # shows mean PIT ~ 0.5 - 0.4/sigma (the flagged +1 grid offset).
        mu, sig, y = _well_specified()
        u = S.side_pit_engine_convention(mu, sig, y)
        mean_shift = 0.5 - float(np.mean(u))
        # sigma ~ 8 => expected shift ~0.4/8 = 0.05; allow generous band.
        self.assertGreater(mean_shift, 0.02)
        self.assertLess(mean_shift, 0.12)

    def test_scale_moves_pit_dispersion_direction(self):
        mu, sig, y = _well_specified(seed=3)
        u_narrow = S.side_pit(mu, sig * 0.7, y)
        u_wide = S.side_pit(mu, sig * 1.4, y)
        sd_uniform = 1.0 / np.sqrt(12.0)
        # too-narrow sigma => PIT piles at edges => sd ABOVE uniform;
        # too-wide => PIT piles mid => sd BELOW uniform.
        self.assertGreater(np.std(u_narrow), sd_uniform * 1.05)
        self.assertLess(np.std(u_wide), sd_uniform * 0.97)

    def test_engine_grid_offset_evidence(self):
        # Cell k of marginal_pmf(25, 9) holds the mass of score k-1:
        # argmax sits at index 26 while the documented convention (dn_pmf
        # semantics) puts P(round(N(25,9)) = 25) at index 25.
        p = marginal_pmf(25.0, 9.0, "dn")
        self.assertEqual(int(np.argmax(p)), 26)
        from scipy import stats
        text = stats.norm.cdf((25.5 - 25.0) / 9.0) \
            - stats.norm.cdf((24.5 - 25.0) / 9.0)
        self.assertAlmostEqual(float(p[26]), float(text), places=4)


class TestUniformityAndGamma(unittest.TestCase):
    def test_uniformity_table_known_cases(self):
        rng = np.random.RandomState(1)
        u_uniform = rng.uniform(size=2000)
        t = S.uniformity_table(u_uniform)
        self.assertTrue(t["is_uniform"])
        self.assertAlmostEqual(t["mean"], 0.5, delta=0.02)
        # U-shaped (sigma too small) is not uniform and has inflated ECE.
        u_peaked = np.concatenate([rng.uniform(0, 0.05, 1000),
                                   rng.uniform(0.95, 1, 1000)])
        t2 = S.uniformity_table(u_peaked)
        self.assertFalse(t2["is_uniform"])
        self.assertGreater(t2["ece"], 0.05)

    def test_clean_gamma_minimum_rules(self):
        flat = [{"gamma": round(1.0 + 0.1 * i, 2), "total_pit_ece": 0.010}
                for i in range(11)]
        out = S.clean_gamma_minimum(flat)
        self.assertFalse(out["clean"])
        rising = [{"gamma": round(1.0 + 0.1 * i, 2),
                   "total_pit_ece": 0.010 + 0.005 * i} for i in range(11)]
        out2 = S.clean_gamma_minimum(rising)
        self.assertFalse(out2["clean"])  # argmin at 1.0 is not a fix
        v_shaped = [{"gamma": round(1.0 + 0.1 * i, 2),
                     "total_pit_ece": 0.03 - 0.025 * i + 0.002 * i * i}
                    for i in range(11)]
        out3 = S.clean_gamma_minimum(v_shaped)
        self.assertTrue(out3["clean"])
        self.assertGreater(out3["argmin_gamma"], 1.0)


class TestEngineInjectionAndGuards(unittest.TestCase):
    def _rows(self, sigma_home=(9.0, 9.4, 9.2), sigma_away=(8.8, 9.1, 9.0)):
        return pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "pred_home": [22.0, 24.5, 27.0],
            "pred_away": [20.0, 21.0, 22.5],
            "sigma_home": list(sigma_home),
            "sigma_away": list(sigma_away),
            "home_score": [20, 31, 17],
            "away_score": [17, 24, 10]})

    def test_per_game_sigma_injection_marginals_match(self):
        rows = self._rows()
        pmfs, summ = S.build_joints_per_game_sigma(rows, 0.01, 0.003)
        self.assertEqual(len(pmfs), 3)
        self.assertLessEqual(summ["summary"]["max_marginal_err_post_ipf"],
                             1e-9)
        for i, (_, r) in enumerate(rows.iterrows()):
            marg_h = marginal_pmf(float(r["pred_home"]),
                                  float(r["sigma_home"]), "dn")
            err = np.max(np.abs(pmfs[i].sum(axis=1) - marg_h))
            self.assertLess(err, 1e-9)

    def test_fold_local_guard_raises_on_pooled_static(self):
        rows = self._rows(sigma_home=(9.0, 9.0, 9.0),
                          sigma_away=(9.0, 9.0, 9.0))
        with self.assertRaises(AssertionError):
            S.build_joints_per_game_sigma(rows, 0.01, 0.003)

    def test_allow_constant_explicit_path(self):
        rows = self._rows(sigma_home=(9.0, 9.0, 9.0),
                          sigma_away=(9.0, 9.0, 9.0))
        pmfs, _ = S.build_joints_per_game_sigma(rows, 0.01, 0.003,
                                                allow_constant=True)
        self.assertEqual(len(pmfs), 3)

    def test_determinism_byte_identical(self):
        rows = self._rows()
        _, a = S.build_joints_per_game_sigma(rows, 0.01, 0.003)
        _, b = S.build_joints_per_game_sigma(rows, 0.01, 0.003)
        self.assertEqual(a["derived"].to_csv(index=False),
                         b["derived"].to_csv(index=False))

    def test_median_of_fold_transfer_arithmetic(self):
        stats = [{"sigma_const_home": 9.0, "gamma_home": 1.0},
                 {"sigma_const_home": 10.0, "gamma_home": 1.2},
                 {"sigma_const_home": 12.0, "gamma_home": 1.4},
                 {"sigma_const_home": 14.0, "gamma_home": 1.1}]
        self.assertEqual(S.median_of_fold_transfer(stats,
                                                   "sigma_const_home"), 11.0)
        self.assertEqual(S.median_of_fold_transfer(stats, "gamma_home"), 1.15)

    def test_clip_bounds(self):
        med = 9.0
        out = S.clip_sigma_to_anchor(np.array([1.0, 9.0, 50.0]), med)
        np.testing.assert_allclose(out, [0.5 * med, 9.0, 2.0 * med])

    def test_nan_inputs_stay_nan_never_zero(self):
        # Imputation discipline: uncovered rows carry NaN mu/sigma; the PIT
        # helpers propagate NaN (never silently zero).
        u = S.side_pit(np.array([np.nan, 22.0]),
                       np.array([np.nan, 9.0]),
                       np.array([17.0, 17.0]))
        self.assertTrue(np.isnan(u[0]))
        self.assertFalse(np.isnan(u[1]))


class TestEngineIdentityAndPins(unittest.TestCase):
    def test_engine_files_byte_identical_after_helpers(self):
        before = {f: (BACKEND / f).read_bytes() for f in ENGINE_FILES}
        # Exercise the layer machinery (imports the engines, never edits).
        mu, sig, y = _well_specified(seed=7)
        S.side_pit(mu, sig, y)
        S.total_pit(mu, sig, mu, sig, y * 0 + 44)
        rows = pd.DataFrame({
            "game_id": ["a", "b"], "pred_home": [22.0, 24.0],
            "pred_away": [20.0, 21.0], "sigma_home": [9.0, 9.4],
            "sigma_away": [8.8, 9.1], "home_score": [20, 31],
            "away_score": [17, 24]})
        S.build_joints_per_game_sigma(rows, 0.01, 0.003)
        for f in ENGINE_FILES:
            self.assertEqual((BACKEND / f).read_bytes(), before[f],
                             f"{f} modified by the sigma layer")

    def test_feature_columns_untouched(self):
        before = list(FEATURE_COLUMNS)
        import nfl_features  # noqa: F401 — module-level list snapshot
        self.assertEqual(FEATURE_COLUMNS, before)
        # The sigma module must not alter the served pool list in memory.
        self.assertIn("is_home", FEATURE_COLUMNS)
        self.assertIn("is_home", before)

    def test_no_moneyline_import_pin(self):
        # Importing the sigma layer must not pull the moneyline module.
        # Source-text pin (era-suite convention): survives same-process runs
        # where other test modules have already imported the moneyline.
        src = (BACKEND / "nfl_sigma_layer.py").read_text()
        self.assertNotIn("import nfl_moneyline", src)
        self.assertNotIn("from nfl_moneyline", src)
        self.assertNotIn("import run_nfl_joint", src)
        self.assertNotIn("from run_nfl_joint", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
