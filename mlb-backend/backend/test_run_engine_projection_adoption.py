"""P1 projection-input ADOPTION pins (gate 7e4c529 -> engine change 2026-09-05).

The run-engine production seam that wired the measured P1 arm into
production:
  * sp_projection.py gained a fit/apply stats API (attach_projection_cols
    byte-identical) for the cross-frame decided->slate transform.
  * run_engine.build_side_frame appends the OPPONENT's sp_proj_era_<opp>
    column per side WHEN PRESENT in the frame (present-in-frame => P1 view;
    absent => the exact pre-adoption C0 column list).
  * run_engine.attach_projection_levels is the production seam (fit stats on
    the decided pre-holdout rows, apply to decided + slate) and degrades to
    the legacy view (loud log, no raise) when the frame lacks components.
  * run_engine_daily / run_engine.main enrich decided + slate before the
    walk/pricing; run_engine_k_edge's daily fallback enriches so the k fit
    sees the same P1 lambda basis it prices.

These pins assert the WIRING properties (col lists, PIT fit-on-pre-only,
determinism, slate same-stats, schema round-trip, served set untouched).
The full C0/P1 reproduction gate over the real 75-fold OOF lives in
run_projection_adoption_check.py + its record.
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_engine as re_
import run_engine_k_edge as ke  # noqa: F401  (patches run_engine — wrapper-identical)
from run_engine import (
    MARKET_COLUMNS_V3,
    NULLABLE_MARKET_COLUMNS,
    attach_projection_levels,
    build_side_frame,
    derive_run_features,
    predict_slate_runs,
    run_oof,
)
from training import FEATURE_COLS

# Projection component columns (sp_projection producer palette).
_LO = ["sp_fip", "sp_xwoba", "sp_whip", "sp_bb9"]
_HI = ["sp_k9_5g", "sp_whiff_3g", "sp_fbvelo_3g"]


def _component_values(rng: np.random.Generator, side: str) -> dict:
    return {
        f"sp_era_{side}": float(rng.uniform(2.5, 6.0)),
        f"sp_fip_{side}": float(rng.uniform(2.5, 6.0)),
        f"sp_xwoba_{side}": float(rng.uniform(0.26, 0.38)),
        f"sp_whip_{side}": float(rng.uniform(0.95, 1.55)),
        f"sp_bb9_{side}": float(rng.uniform(1.5, 5.5)),
        f"sp_k9_5g_{side}": float(rng.uniform(6.0, 12.0)),
        f"sp_whiff_3g_{side}": float(rng.uniform(0.18, 0.36)),
        f"sp_fbvelo_3g_{side}": float(rng.uniform(90.0, 96.0)),
    }


def make_games(n_days: int = 80, per_day: int = 6, seed: int = 7,
               with_components: bool = True) -> pd.DataFrame:
    """Synthetic decided frame (FEATURE_COLS + core cols + optional
    projection component columns), mirroring test_run_engine's fixtures."""
    rng = np.random.default_rng(seed)
    abbrs = ["NYY", "BOS", "LAD", "SF", "ATL", "HOU"]
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=d)
        for g in range(per_day):
            ht, at = abbrs[(d + g) % 6], abbrs[(d + g + 3) % 6]
            lam_h = np.exp(0.4 + 0.05 * rng.normal())
            lam_a = np.exp(0.3 + 0.05 * rng.normal())
            hs, as_ = rng.poisson(lam_h), rng.poisson(lam_a)
            row = {c: float(rng.normal()) for c in FEATURE_COLS}
            row.update({"game_pk": 200000 + d * per_day + g,
                        "game_date": date, "home_team": ht, "away_team": at,
                        "home_win": float(hs > as_),
                        "home_score": int(hs), "away_score": int(as_)})
            if with_components:
                row.update(_component_values(rng, "home"))
                row.update(_component_values(rng, "away"))
            rows.append(row)
    return pd.DataFrame(rows)


class TestSideViewProjectionAppend(unittest.TestCase):
    def setUp(self):
        self.raw = make_games(n_days=60, per_day=4, seed=3,
                              with_components=False)
        # frame WITH components but NOT attached == raw (components are not
        # FEATURE_COLS, so they never enter the C0 view by themselves)
        self.raw_comps = make_games(n_days=60, per_day=4, seed=3,
                                    with_components=True)
        self.att = make_games(n_days=60, per_day=4, seed=3,
                              with_components=True)
        self.att, _, meta = attach_projection_levels(self.att)
        self.assertTrue(meta["attached"])

    def test_c0_column_list_byte_identical_without_projection(self):
        """No sp_proj columns in the frame -> exactly the pre-adoption
        view (nothing appended, nothing reordered). Adding the component
        columns alone (no attach) must not perturb the view either."""
        for side in ("home", "away"):
            _, cols = build_side_frame(self.raw, side)
            self.assertNotIn("sp_proj_era_home", cols)
            self.assertNotIn("sp_proj_era_away", cols)
            _, cols2 = build_side_frame(self.raw_comps, side)
            self.assertEqual(cols, cols2)
        # the derived keep-list is the 53-feature production view
        keep, _ = derive_run_features(list(FEATURE_COLS))
        self.assertEqual(len(keep), 53)

    def test_p1_appends_opponent_level_only_when_present(self):
        c0_cols = {}
        p1_cols = {}
        for side in ("home", "away"):
            _, c0 = build_side_frame(self.raw, side)
            _, p1 = build_side_frame(self.att, side)
            c0_cols[side], p1_cols[side] = c0, p1
        # home model sees the AWAY starter's projection; never its own.
        self.assertIn("sp_proj_era_away", p1_cols["home"])
        self.assertNotIn("sp_proj_era_home", p1_cols["home"])
        self.assertIn("sp_proj_era_home", p1_cols["away"])
        self.assertNotIn("sp_proj_era_away", p1_cols["away"])
        # diff vs C0 is EXACTLY the one appended column, appended last.
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            diff = [c for c in p1_cols[side]
                    if c not in c0_cols[side]]
            self.assertEqual(diff, [f"sp_proj_era_{opp}"])
            self.assertEqual(c0_cols[side] + diff, p1_cols[side])

    def test_served_feature_set_untouched(self):
        keep, dropped = derive_run_features(list(FEATURE_COLS))
        self.assertNotIn("sp_proj_era_home", FEATURE_COLS)
        self.assertNotIn("sp_proj_era_away", FEATURE_COLS)
        self.assertNotIn("sp_proj_era_home", keep)
        self.assertNotIn("sp_proj_era_away", keep)
        self.assertEqual(len(keep), 53)
        # the projection columns must never ride into the moneyline's view
        self.assertNotIn("sp_proj_era_home", dropped + keep)


class TestAttachProjectionLevels(unittest.TestCase):
    def test_skips_without_raise_when_components_missing(self):
        raw = make_games(n_days=50, per_day=4, seed=11,
                         with_components=False)
        dec, slate, meta = attach_projection_levels(
            raw, slate=raw.head(3).copy())
        self.assertFalse(meta["attached"])
        self.assertIn("absent", meta["reason"])
        self.assertNotIn("sp_proj_era_home", dec.columns)
        # empty decided also degrades
        e, _, m = attach_projection_levels(pd.DataFrame())
        self.assertFalse(m["attached"])

    def test_attach_pit_fit_on_pre_rows_only(self):
        df = make_games(n_days=80, per_day=5, seed=21, with_components=True)
        dates = pd.to_datetime(df["game_date"])
        pre = (dates < dates.max() - pd.Timedelta(days=21)).to_numpy()
        d1, _, meta1 = attach_projection_levels(df)
        self.assertTrue(meta1["attached"])
        # Perturb every SEALED row's components to extreme values; the PRE
        # rows' projections must be bit-identical (stats exclude sealed).
        df2 = df.copy()
        sealed = ~pre
        for c in ([f"{x}_{s}" for s in ("home", "away") for x in _LO + _HI]
                  + [f"sp_era_{s}" for s in ("home", "away")]):
            df2.loc[sealed, c] = 999.0
        d2, _, _ = attach_projection_levels(df2)
        for s in ("home", "away"):
            col = f"sp_proj_era_{s}"
            a = d1.loc[pre, col].to_numpy(float)
            b = d2.loc[pre, col].to_numpy(float)
            ok = np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True)
            self.assertTrue(ok, f"{col} pre projections moved when sealed "
                                f"rows changed (PIT violation)")
            # ... and the SEALED rows DID move (the transform applies to all)
            self.assertFalse(np.allclose(
                d1.loc[sealed, col].to_numpy(float),
                d2.loc[sealed, col].to_numpy(float), equal_nan=True))

    def test_deterministic_and_slate_uses_decided_stats(self):
        df = make_games(n_days=90, per_day=5, seed=31, with_components=True)
        slate = df.head(4).copy()
        d1, s1, m1 = attach_projection_levels(df, slate=slate)
        d2, s2, m2 = attach_projection_levels(df, slate=slate)
        for col in ("sp_proj_era_home", "sp_proj_era_away"):
            self.assertTrue((d1[col] == d2[col]).all())
            self.assertTrue((s1[col] == s2[col]).all())
            # slate row with identical components as a decided row gets the
            # identical projection (row-wise transform on shared stats)
            self.assertTrue(np.allclose(
                s1[col].to_numpy(float), d1[col].iloc[:4].to_numpy(float),
                atol=1e-12))
        self.assertEqual(m1["coverage"], m2["coverage"])
        self.assertEqual(m1["slopes"], m2["slopes"])
        # attach is idempotent on an already-attached frame
        d3, _, m3 = attach_projection_levels(d1)
        for col in ("sp_proj_era_home", "sp_proj_era_away"):
            self.assertTrue((d1[col] == d3[col]).all())
        self.assertEqual(m3["slopes"], m1["slopes"])


class TestWalkWiring(unittest.TestCase):
    def test_p1_walk_deterministic_and_differs_from_c0(self):
        df = make_games(n_days=90, per_day=6, seed=41, with_components=True)
        # C0 walk: raw frame (no projection columns).
        r1 = run_oof(df, min_val_games=5)["oof"]
        r2 = run_oof(df, min_val_games=5)["oof"]
        self._assert_lambda_equal(r1, r2, "C0 double walk")
        # P1 walk: attach then run through the SAME production run_oof.
        att, _, m = attach_projection_levels(df.copy())
        self.assertTrue(m["attached"])
        p1a = run_oof(att, min_val_games=5)["oof"]
        p1b = run_oof(att, min_val_games=5)["oof"]
        self._assert_lambda_equal(p1a, p1b, "P1 double walk")
        # The wiring actually changes the lambda basis (P1 != C0 somewhere)
        # and every row is priced on both views.
        self.assertEqual(len(p1a), len(r1))
        diff = (p1a["home_expected_runs"].to_numpy(float)
                - r1["home_expected_runs"].to_numpy(float))
        self.assertGreater(float(np.abs(diff).max()), 0.0,
                           "P1 walk must differ from C0 walk")

    def _assert_lambda_equal(self, a, b, label):
        a = a.sort_values("game_pk").reset_index(drop=True)
        b = b.sort_values("game_pk").reset_index(drop=True)
        self.assertEqual(len(a), len(b), label)
        for col in ("home_expected_runs", "away_expected_runs"):
            self.assertEqual(
                a[col].to_numpy(float).tolist(),
                b[col].to_numpy(float).tolist(),
                f"{label}: {col} drift")

    def test_slate_priced_through_p1_schema_unchanged(self):
        df = make_games(n_days=90, per_day=6, seed=51, with_components=True)
        decided, _, m = attach_projection_levels(df.copy())
        self.assertTrue(m["attached"])
        # Pseudo-slate: tail rows with outcomes dropped (same game_pk space
        # is fine for this schema check — predict_slate_runs prices rows).
        slate = df.tail(3).drop(columns=["home_win", "home_score",
                                         "away_score"]).copy()
        slate["game_pk"] = [900001, 900002, 900003]
        decided, slate, _ = attach_projection_levels(decided, slate=slate)
        self.assertIn("sp_proj_era_home", slate.columns)
        oof = run_oof(decided, min_val_games=5)
        rounds = oof["summary"]["final_fit_rounds"]
        curve = {"form": "linear", "a": 0.25, "b": 0.01}
        out = predict_slate_runs(decided, slate, rounds,
                                 {"home": curve, "away": curve},
                                 n_draws=500, seed=1)
        self.assertEqual(len(out), 3)
        self.assertTrue((out["kind"] == "slate").all())
        for col in MARKET_COLUMNS_V3:
            if col not in NULLABLE_MARKET_COLUMNS and col not in (
                    "home_score", "away_score", "total_runs"):
                self.assertIn(col, out.columns, col)
                self.assertFalse(out[col].isna().any(), col)
        self.assertTrue((out["home_expected_runs"] > 1).all())

    def test_k_edge_daily_fallback_attaches_projection(self):
        """The k-edge daily seam fits k on the same P1 basis it prices: its
        fallback OOF (cache-empty) must be walked on projection-attached
        decided."""
        df = make_games(n_days=60, per_day=5, seed=61, with_components=True)
        seen = {}

        def fake_oof(decided, **kwargs):
            seen["decided"] = decided
            # minimal non-empty OOF: fit_k_edge returns 1.0 below n=100.
            oof = pd.DataFrame({
                "home_expected_runs": [4.5, 4.2, 4.8, 4.4, 4.6],
                "away_expected_runs": [4.3, 4.1, 4.7, 4.0, 4.4],
                "home_score": [5, 3, 6, 2, 4],
                "away_score": [2, 4, 1, 5, 3],
                "game_date": pd.to_datetime(
                    ["2026-06-01", "2026-06-02", "2026-06-03",
                     "2026-06-04", "2026-06-05"]),
            })
            return {"oof": oof, "summary": {}}

        orig_cache = ke._DAILY_OOF_CACHE
        ke._DAILY_OOF_CACHE = None
        try:
            with unittest.mock.patch.object(ke, "_orig_run_oof",
                                            side_effect=fake_oof), \
                 unittest.mock.patch.object(ke, "_orig_run_engine_daily",
                                            return_value={"block": None}):
                ke.run_engine_daily(df, df.head(2), "20260905",
                                    decided_snapshot=df.copy(), k_edge=None)
        finally:
            ke._DAILY_OOF_CACHE = orig_cache
        self.assertIn("sp_proj_era_home", seen["decided"].columns)
        self.assertIn("sp_proj_era_away", seen["decided"].columns)


if __name__ == "__main__":
    unittest.main()
