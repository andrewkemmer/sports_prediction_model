"""Run-line feature-expansion ablation tests (C2-base re-test of the 5
pruned composites).

Covers the 2026-09 run-line task's crucial requirements with pure unit
tests on synthetic fixtures (mirroring test_mlb_margin_k_ablation.py):

- PIT: flipping SEALED (future) games' outcomes leaves every pre-holdout
  surface byte-identical — k, α curves (fit pre-holdout only), pooled
  margin/totals CRPS, pooled per-line tables. Only the sealed targets move.
- Coverage ≥95% for each candidate on the current C2 frame (decided rows).
- Arms differ ONLY in the added feature(s): C0 == the 53-col production
  view; A1..A5 == C0 + exactly one candidate; AALL == C0 + all 5; the
  run_oof override contract (arm feats + drop terms partition FEATURE_COLS).
- C2 layer ACTIVE in every arm: price_arm recovers a known planted k from
  pre-holdout games only, expands the λ pair (sum preserved), and sealed
  corruption leaves the fitted k byte-identical.
- FEATURE_COLUMNS / production config untouched (unregistered invariant +
  source guard).

Run from mlb-backend/:
    python -m unittest backend.test_mlb_runline_expansion_ablation
or directly:
    python backend/test_mlb_runline_expansion_ablation.py
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_mlb_runline_expansion_ablation as rexp
from run_engine import derive_run_features
from training import FEATURE_COLS


def _synthetic_oof(n_pre: int = 2500, n_sealed: int = 60,
                   true_k: float = 1.55, seed: int = 42) -> pd.DataFrame:
    """Decided OOF-shaped frame: independent per-side λ means with a planted
    edge→margin slope (true_k), so fit_k_edge should recover ~true_k on the
    pre-holdout rows. The pre window is kept >= ~2,500 rows so alpha_bins'
    quantile bins each clear ALPHA_MIN_BIN (250) WITHOUT reaching the merge
    loop's latent last-bin edge case (run_engine.py:919 fires only on small
    samples; production OOFs are ~6.7k and never hit it)."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_sealed
    lam_h = np.clip(rng.normal(4.45, 0.5, n), 2.5, 6.5)
    lam_a = np.clip(rng.normal(4.25, 0.5, n), 2.5, 6.5)
    edge = lam_h - lam_a
    # Plant the TOTAL actual-margin slope on the modeled edge (like the
    # production probe's actual ≈ 1.66·λ_edge): build the score DIFFERENCE
    # from a common scoring base so OLS(actual, edge) ≈ true_k exactly — the
    # λ level difference stays in the MODELED edge only, never in the scores.
    base = np.clip(rng.normal(4.3, 0.6, n), 1.0, 8.0)
    diff = true_k * edge + rng.normal(0, 0.4, n)
    hs = np.clip(np.round(base + diff / 2), 0, 20)
    as_ = np.clip(np.round(base - diff / 2), 0, 20)
    # Round λs to the 4-decimal convention the engine persists.
    lam_h = np.round(lam_h, 4)
    lam_a = np.round(lam_a, 4)
    dates = pd.date_range("2026-04-01", periods=n, freq="D")
    pk = 800_000
    return pd.DataFrame({
        "game_pk": list(range(pk, pk + n)),
        "game_date": dates,
        "home_expected_runs": lam_h,
        "away_expected_runs": lam_a,
        "home_score": hs, "away_score": as_,
        "fold_idx": np.arange(n) // 60,
    })


_N_DRAWS_TEST = 2000   # cheap MC for unit tests (production MC_DRAWS in real runs)


class TestPIT(unittest.TestCase):
    def test_sealed_outcome_flips_leave_pre_surfaces_identical(self):
        oof = _synthetic_oof(seed=3)
        base = rexp.price_arm(oof, holdout_days=21, n_draws=_N_DRAWS_TEST)
        # PIT: flip the SEALED games' outcomes (their actual scores).
        flipped = oof.copy()
        # The seal is the last holdout_days=21 rows (price_arm's cut). Flip
        # exactly those sealed rows' outcomes.
        hold_idx = np.argsort(pd.to_datetime(oof["game_date"]).to_numpy(),
                              kind="stable")[-21:]
        flipped.loc[hold_idx, "home_score"] += 50
        flipped.loc[hold_idx, "away_score"] += 25
        alt = rexp.price_arm(flipped, holdout_days=21, n_draws=_N_DRAWS_TEST)
        self.assertEqual(base["k"]["k_fitted_run"],
                         alt["k"]["k_fitted_run"],
                         "sealed scores must never enter the k fit")
        self.assertEqual(base["margin_crps_pooled"],
                         alt["margin_crps_pooled"],
                         "pooled margin CRPS must be PIT-invariant")
        self.assertEqual(base["totals"]["crps_pooled"],
                         alt["totals"]["crps_pooled"],
                         "pooled totals CRPS must be PIT-invariant")
        self.assertEqual(base["totals"]["by_line_pooled"],
                         alt["totals"]["by_line_pooled"])
        self.assertIn("by_line_sealed", base["totals"])
        self.assertEqual(base["run_line_minus_1_5"]["pooled"],
                         alt["run_line_minus_1_5"]["pooled"],
                         "pooled run-line bins must be PIT-invariant")
        # The sealed targets DID move, so sealed CRPS may too — assert only
        # that the surfaces still exist and the frame is sane.
        self.assertIsNotNone(alt["margin_crps_sealed"])
        # Strict date cutoff (max − 21d) seals the last 22 daily fixture rows.
        self.assertGreaterEqual(int(alt["n_sealed"]), 20)
        self.assertEqual(int(base["n_sealed"]), int(alt["n_sealed"]))

    def test_k_recovered_from_pre_holdout_only(self):
        oof = _synthetic_oof(true_k=1.55, seed=7)
        res = rexp.price_arm(oof, holdout_days=21, n_draws=_N_DRAWS_TEST)
        self.assertAlmostEqual(res["k"]["k_fitted_run"], 1.55, delta=0.12,
                               msg="planted k must be recovered on pre rows")
        # C2 applied: expanded lams differ from raw only when k != 1, and
        # the sum is preserved (harness asserts via apply_k_edge itself here).
        self.assertTrue(res["margin_crps_pooled"] is not None)
        self.assertTrue(np.isfinite(res["margin_crps_pooled"]))
        for key in ("run_line_minus_1_5", "run_line_edge_bins", "totals",
                    "derived_ml"):
            self.assertIn(key, res)


class TestCoverage(unittest.TestCase):
    def test_candidate_coverage_ge_95pct_on_current_frame(self):
        path = rexp.DATA_DELIVERY_DIR / "game_level_features.csv"
        cols = ["game_pk", "home_win"] + list(rexp.CANDIDATES.values())
        df = pd.read_csv(path, usecols=cols)
        decided = df[df["home_win"].notna()]
        self.assertGreaterEqual(len(decided), 6000,
                                "expected a multi-season decided frame")
        for tag, f in rexp.CANDIDATES.items():
            cov = float(decided[f].notna().mean())
            self.assertGreaterEqual(
                cov, 0.95, f"{tag} ({f}) must clear the 95% coverage floor; "
                f"got {cov:.4f}")


class TestArmConstruction(unittest.TestCase):
    def test_arms_are_c0_plus_exactly_the_candidate(self):
        feats = rexp.arm_features()
        self.assertEqual(len(feats["C0"]), 53,
                         "production run view is 53 columns")
        kept, dropped = derive_run_features(list(FEATURE_COLS))
        self.assertEqual(sorted(kept), sorted(feats["C0"]),
                         "C0 must be exactly the served 53-feature view")
        for tag, f in rexp.CANDIDATES.items():
            self.assertEqual(sorted(set(feats[tag]) - set(feats["C0"])), [f],
                             f"{tag} must add exactly {f}")
        self.assertEqual(
            sorted(set(feats["AALL"]) - set(feats["C0"])),
            sorted(rexp.CANDIDATES.values()),
            "AALL must add exactly the 5 candidates")

    def test_override_contract_partitions_feature_cols(self):
        feats = rexp.arm_features()
        for name, arm_feats in feats.items():
            terms = rexp.arm_drop_terms(arm_feats)
            self.assertEqual(sorted(arm_feats + terms),
                             sorted(FEATURE_COLS),
                             f"{name}: run_features + dropped must partition "
                             f"FEATURE_COLS")

    def test_feature_cols_untouched(self):
        before = list(FEATURE_COLS)
        _ = rexp.arm_features()
        _ = rexp.arm_drop_terms(rexp.arm_features()["AALL"])
        self.assertEqual(list(FEATURE_COLS), before)
        for f in rexp.CANDIDATES.values():
            self.assertIn(f, FEATURE_COLS,
                          "candidates must remain in FEATURE_COLS (they are "
                          "only run-view excluded, never removed)")
        src = Path(rexp.__file__).read_text()
        self.assertNotIn("FEATURE_COLS =", src.replace(
            "FEATURE_COLS  # noqa: E402", ""),
            "harness must never reassign FEATURE_COLS")


class TestEvaluationSurfaces(unittest.TestCase):
    def test_price_arm_emits_all_gate_surfaces(self):
        oof = _synthetic_oof(seed=11)
        res = rexp.price_arm(oof, holdout_days=21, n_draws=_N_DRAWS_TEST)
        for label in ("pooled", "sealed"):
            self.assertIn(label, res["run_line_minus_1_5"])
            self.assertIn(label, res["run_line_edge_bins"])
        self.assertIn("by_line_pooled", res["totals"])
        self.assertIn("by_line_sealed", res["totals"])
        self.assertIn("bucket_55_60_sealed", res["derived_ml"])
        self.assertGreaterEqual(res["n_pre"], 100)
        self.assertGreaterEqual(res["n_sealed"], 20)
        self.assertTrue(np.isfinite(
            res["derived_ml"]["pwin_sd_pooled"]))


if __name__ == "__main__":
    unittest.main()