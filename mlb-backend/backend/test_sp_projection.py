"""Tests for the candidate SP projection-composite producer (sp_projection).

Pins (record-only spec, verbatim b7eed32 discipline):
  * Coverage: ~97-99% of OOF rows per season-partial, >= 0.95 on the current
    frame (2024 partial may sit ~0.94-0.95).
  * PIT: z-statistics + OLS scale fit exclusively on the pre-holdout rows —
    mutating POST-holdout component values never changes pre-holdout outputs.
  * Served set untouched: the producer never modifies training.FEATURE_COLS
    and never imports the run engine (no engine change by any path).
  * Determinism: attaching twice to the same frame is bit-identical.
  * The composite is produced from the frame's own PIT-safe trailing
    Statcast-derived columns (sp_fip/sp_xwoba/sp_whip/sp_bb9 lower-better;
    sp_k9_5g/sp_whiff_3g/sp_fbvelo_3g higher-better); no pitch-level store,
    no market data, no re-fit per game.

Run:  python -m unittest test_sp_projection -v   (from mlb-backend/backend)
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if "resource" not in sys.modules:  # POSIX-only module, stub on Windows
    _res = types.ModuleType("resource")
    _ru = types.SimpleNamespace(ru_maxrss=0)
    _res.getrusage = lambda *_: _ru
    _res.RUSAGE_SELF = 0
    sys.modules["resource"] = _res

from sp_projection import (  # noqa: E402
    MIN_PROJ_COMPONENTS,
    PROJ_HI_BETTER,
    PROJ_LO_BETTER,
    SP_JUNK_ERA,
    attach_projection_cols,
)

CSV = _BACKEND_DIR.parent / "data_delivery" / "game_level_features.csv"
HOLDOUT_DAYS = 21

COMPONENT_COLS = ([f"{c}_home" for c in PROJ_LO_BETTER + PROJ_HI_BETTER]
                  + [f"{c}_away" for c in PROJ_LO_BETTER + PROJ_HI_BETTER]
                  + ["sp_era_home", "sp_era_away"])


def _load_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.dropna(subset=["home_win"]).reset_index(drop=True)


def _pre_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["game_date"]
            < df["game_date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()


def _synthetic_frame(n: int = 400) -> pd.DataFrame:
    """Small self-contained frame (no CSV) with all component columns."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    df = pd.DataFrame({"game_date": dates,
                       "home_win": rng.integers(0, 2, n).astype(float)})
    for c in COMPONENT_COLS:
        df[c] = (rng.normal(4.0, 1.0, n) if c.startswith("sp_era")
                 else rng.normal(0.0, 1.0, n))
    return df


class TestCompositeDefinition(unittest.TestCase):
    def test_component_sets_and_floors(self):
        self.assertGreaterEqual(MIN_PROJ_COMPONENTS, 3)
        self.assertEqual(len(PROJ_LO_BETTER), 4)
        self.assertEqual(len(PROJ_HI_BETTER), 3)
        # Lower-is-better set must not overlap higher-is-better.
        self.assertFalse(set(PROJ_LO_BETTER) & set(PROJ_HI_BETTER))
        self.assertEqual(SP_JUNK_ERA, 15.0)


class TestCoverageAndFitPins(unittest.TestCase):
    @unittest.skipUnless(CSV.exists(), "committed frame not present")
    def test_coverage_pins_on_current_frame(self):
        df = _load_frame()
        pre = _pre_mask(df)
        out, meta = attach_projection_cols(df, pre)
        for side in ("home", "away"):
            self.assertGreaterEqual(meta[side]["coverage_pre"], 0.95, meta[side])
            self.assertGreaterEqual(meta[side]["coverage_sealed"], 0.95,
                                    meta[side])
            # Composite in ERA-equivalent quality units: ERA ~ composite slope
            # is negative (better composite -> lower ERA) and ~-1.2 on this
            # frame (record pin: -1.2213 / -1.2138).
            self.assertTrue(-1.35 <= meta[side]["era_on_proj_slope"] <= -1.05,
                            meta[side])
        # Per season-partial coverage (2024 partial is the thinnest).
        out["season"] = out["game_date"].dt.year
        for season, g in out.groupby("season"):
            cov = float(g[["sp_proj_era_home", "sp_proj_era_away"]]
                        .notna().mean().mean())
            self.assertGreaterEqual(cov, 0.94,
                                    f"season {season} coverage {cov:.4f}")


class TestPitDiscipline(unittest.TestCase):
    def test_pit_fit_uses_pre_rows_only(self):
        df = _synthetic_frame()
        pre = _pre_mask(df)
        out, _ = attach_projection_cols(df, pre)
        pre_idx = np.where(pre)[0]

        # Corrupt every POST-holdout component value; pre-row outputs must not
        # move at all (stats + OLS scale fit exclusively on pre rows).
        df2 = df.copy()
        post = ~pre
        for c in COMPONENT_COLS:
            df2.loc[post, c] = 999.0
        out2, meta2 = attach_projection_cols(df2, pre)

        pd.testing.assert_series_equal(
            out["sp_proj_era_home"].iloc[pre_idx],
            out2["sp_proj_era_home"].iloc[pre_idx])
        pd.testing.assert_series_equal(
            out["sp_proj_era_away"].iloc[pre_idx],
            out2["sp_proj_era_away"].iloc[pre_idx])
        self.assertGreater(meta2["home"]["n_cal_rows"], 0)

    def test_post_rows_use_pre_fit_stats(self):
        """Post rows are transformed with PRE-fit stats — dropping post rows
        from the frame entirely leaves pre-row values identical."""
        df = _synthetic_frame()
        pre = _pre_mask(df)
        out_full, _ = attach_projection_cols(df, pre)
        out_pre, _ = attach_projection_cols(df[pre].reset_index(drop=True),
                                            np.ones(int(pre.sum()), dtype=bool))
        pd.testing.assert_series_equal(
            out_full["sp_proj_era_home"].iloc[np.where(pre)[0]]
            .reset_index(drop=True),
            out_pre["sp_proj_era_home"])


class TestIsolationAndDeterminism(unittest.TestCase):
    def test_served_feature_cols_untouched(self):
        import training
        before = list(training.FEATURE_COLS)
        df = _synthetic_frame()
        pre = _pre_mask(df)
        attach_projection_cols(df, pre)
        self.assertEqual(list(training.FEATURE_COLS), before)
        # None of the candidate columns are served today.
        for c in ("sp_proj_era_home", "sp_proj_era_away", "sp_proj_era_diff"):
            self.assertNotIn(c, training.FEATURE_COLS)

    def test_no_run_engine_touch(self):
        """The producer must not import the run engine or training
        (record-only by construction)."""
        src = (_BACKEND_DIR / "sp_projection.py").read_text(encoding="utf-8")
        for banned in ("import run_engine", "from run_engine",
                       "import training", "from training"):
            self.assertNotIn(banned, src,
                             f"sp_projection.py must not import {banned!r}")
        # Importing the producer must not ADD run_engine/training to
        # sys.modules (earlier test modules may already have imported them —
        # the pin is that sp_projection itself never pulls them in).
        before = set(sys.modules)
        import sp_projection
        added = set(sys.modules) - before
        self.assertFalse(added & {"run_engine", "training"},
                         f"sp_projection import pulled in: {sorted(added)}")
        self.assertFalse(hasattr(sp_projection, "run_engine"))

    def test_determinism(self):
        df = _synthetic_frame()
        pre = _pre_mask(df)
        out1, meta1 = attach_projection_cols(df, pre)
        out2, meta2 = attach_projection_cols(df, pre)
        pd.testing.assert_frame_equal(out1, out2)
        self.assertEqual(meta1, meta2)

    def test_requires_all_component_columns(self):
        df = _synthetic_frame().drop(columns=["sp_fip_home"])
        pre = _pre_mask(df)
        with self.assertRaisesRegex(ValueError, "sp_fip_home"):
            attach_projection_cols(df, pre)

    def test_pre_mask_length_must_match_frame(self):
        df = _synthetic_frame()
        with self.assertRaisesRegex(ValueError, "pre_mask length"):
            attach_projection_cols(df, np.ones(10, dtype=bool))


if __name__ == "__main__":
    unittest.main()
