"""NFL Calibration artifacts (``nfl_moneyline.py`` Part-A) — tests.

Network-free unit tests around the MLB-equivalent artifacts the moneyline
record now emits as siblings:

- ``reliability_buckets`` — pooled-OOF reliability-bin math (bucket shape,
  count / mean_predicted / mean_actual / gap), empty-bin omission, tiny fold.
- ``build_calibration`` — the ``nfl_calibration_*.json`` shape: metrics w/
  calibrated twins, calibration map (method / params a,b,n / metrics_raw /
  metrics_calibrated / calibration_buckets_calibrated), preq-vs-deployed
  distinction.
- ``build_history_frame`` — the per-game OOF + sealed prediction-history CSV
  (exact column contract), correctness of pick/winner/correct, and the
  LEAK-FREE assertion that OOF rows never carry the sealed season (2025).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nfl_moneyline import (  # noqa: E402
    ECE_BINS,
    HISTORY_COLUMNS,
    SEALED_SEASON,
    build_calibration,
    build_history_frame,
    platt_fit,
    reliability_buckets,
)


def _oof_meta(seasons=(2021, 2022), n=6):
    """A small decided-game frame (no network) shaped like a fold val set."""
    rows = []
    for i in range(n):
        s = seasons[i % len(seasons)]
        rows.append({
            "game_id": f"g{i}", "season": s, "week": (i % 18) + 1,
            "gameday": pd.Timestamp(f"{s}-09-{i % 20 + 1:02d}"),
            "home_team": f"H{i % 4}", "away_team": f"A{(i + 1) % 4}",
            "home_score": 24 if i % 2 == 0 else 10,
            "away_score": 10 if i % 2 == 0 else 27,
        })
    return pd.DataFrame(rows)


class TestReliabilityBuckets(unittest.TestCase):
    def test_bucket_math_favored_view(self):
        """Every game is binned from the favored side (>= 50%) with the
        outcome flipped on away-favored rows; exact count/means/gap per bin."""
        # p: [home-fav, home-fav, away-fav -> fav=0.4?s no: 0.4 -> fav 0.6, away-fav]
        # p_fav = max(p, 1-p): [0.6, 0.9, 0.6, 0.9]; y_fav = y when home-fav,
        # else 1-y: [1, 0, 0, 1]
        y = np.array([1, 0, 1, 0])
        p = np.array([0.6, 0.9, 0.4, 0.1])
        b = reliability_buckets(y, p, bins=2)
        for x in b:  # no sub-50% bucket
            lo = int(x["bucket"].split("-")[0].rstrip("%"))
            self.assertGreaterEqual(lo, 50)
        low = next(x for x in b if x["bucket"].startswith("50%"))
        high = next(x for x in b if x["bucket"].startswith("75%"))
        self.assertEqual(low["count"], 2)
        self.assertAlmostEqual(low["mean_predicted"], 0.6, places=4)
        self.assertAlmostEqual(low["mean_actual"], 0.5, places=4)
        self.assertAlmostEqual(low["gap"], 0.6 - 0.5, places=4)
        self.assertEqual(high["count"], 2)
        self.assertAlmostEqual(high["mean_predicted"], 0.9, places=4)
        self.assertAlmostEqual(high["mean_actual"], 0.5, places=4)

    def test_favored_flip_parity_with_frontend_curve(self):
        """Parity with the frontend flip np.maximum(p, 1-p): bins run >= 50%
        (no sub-50%) and mean_predicted/mean_actual/gap match the expected
        favored-side grouping exactly."""
        y = np.array([1, 0, 1, 1, 0])
        p = np.array([0.2, 0.5, 0.7, 0.9, 0.3])
        bins = 10
        b = reliability_buckets(y, p, bins=bins)
        for x in b:
            lo = int(x["bucket"].split("-")[0].rstrip("%"))
            self.assertGreaterEqual(lo, 50)
        # the frontend curve's flip: favored pred + flipped favored outcome
        p_fav = np.maximum(p, 1.0 - p)
        y_fav = np.where(p >= 0.5, y, 1.0 - y)
        edges = np.linspace(0.5, 1.0, bins + 1)
        idx = np.clip(np.digitize(p_fav, edges[1:-1]), 0, bins - 1)
        got = {x["bucket"]: x for x in b}
        for i in range(bins):
            m = idx == i
            if not m.any():
                continue
            label = f"{edges[i]:.0%}-{edges[i + 1]:.0%}"
            g = got[label]
            self.assertEqual(g["count"], int(m.sum()))
            self.assertAlmostEqual(g["mean_predicted"],
                                   float(np.mean(p_fav[m])), places=3)
            self.assertAlmostEqual(g["mean_actual"],
                                   float(np.mean(y_fav[m])), places=3)
            self.assertAlmostEqual(g["gap"],
                                   g["mean_predicted"] - g["mean_actual"],
                                   places=4)

    def test_shared_mask_parameter_from_raw(self):
        """When home_fav is passed (the calibrated set), the FAVORED side is the
        same one the raw blend defined — never re-derived per-line."""
        raw = np.array([0.6, 0.6, 0.4, 0.4])        # home-fav for 0,1; away 2,3
        raw_fav = raw >= 0.5
        # calibrated values stay on the raw side (Platt monotone in practice)
        cal = np.array([0.58, 0.62, 0.42, 0.38])
        b = reliability_buckets(np.array([1, 0, 1, 1]), cal, bins=2,
                                home_fav=raw_fav)
        p_fav = np.where(raw_fav, cal, 1.0 - cal)   # [0.58,0.62,0.58,0.62]
        self.assertEqual([x["count"] for x in b], [4])
        self.assertAlmostEqual(b[0]["mean_predicted"],
                               float(np.mean(p_fav)), places=4)

    def test_empty_bin_omitted(self):
        """No fabricated points: a bin with zero observations is absent."""
        y = np.array([1, 1, 0, 0])
        p = np.array([0.99, 0.99, 0.01, 0.01])
        b = reliability_buckets(y, p, bins=10)
        self.assertLess(len(b), 10)
        for x in b:
            self.assertGreaterEqual(x["count"], 1)
            self.assertIn("mean_predicted", x)
            self.assertIn("mean_actual", x)
            self.assertIn("gap", x)

    def test_bucket_keys_present(self):
        b = reliability_buckets(np.array([1, 0, 1]), np.array([0.6, 0.6, 0.4]),
                                bins=2)
        self.assertTrue(all({"bucket", "mean_predicted", "mean_actual",
                             "count", "gap"} <= set(x) for x in b))


class TestBuildCalibration(unittest.TestCase):
    def test_shape_mirrors_mlb_calibration(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        raw = np.array([0.7, 0.3, 0.7, 0.3, 0.7, 0.3])
        cal = np.array([0.65, 0.35, 0.65, 0.35, 0.65, 0.35])
        platt = platt_fit(np.array([0.7, 0.3] * 10),
                          np.array([1, 0] * 10))
        c = build_calibration(y, raw, cal, platt, bins=2)

        self.assertIn("date", c)
        self.assertEqual(c["n_games"], len(y))
        self.assertEqual(c["daily"], [])
        # metrics + calibrated twins
        for k in ("auc", "brier", "logloss", "ece",
                  "brier_calibrated", "logloss_calibrated",
                  "ece_calibrated"):
            self.assertIn(k, c["metrics"])
        # calibration map
        cal_sec = c["calibration"]
        self.assertEqual(cal_sec["method"], "platt")
        self.assertAlmostEqual(cal_sec["params"]["a"],
                               float(platt.coef_[0][0]), places=4)
        self.assertAlmostEqual(cal_sec["params"]["b"],
                               float(platt.intercept_[0]), places=4)
        self.assertEqual(cal_sec["params"]["n"], len(y))
        for k in ("metrics_raw", "metrics_calibrated"):
            self.assertIn("ece", cal_sec[k])
            self.assertIn("brier", cal_sec[k])
            self.assertIn("logloss", cal_sec[k])
        # calibration_buckets + calibrated-bucket presence
        self.assertTrue(c["calibration_buckets"])
        self.assertGreaterEqual(len(cal_sec["calibration_buckets_calibrated"]), 1)

    def test_calibrated_bucket_present(self):
        """calibration_buckets_calibrated is emitted alongside the raw curve."""
        y = np.array([1, 0, 1, 0, 1, 0])
        c = build_calibration(y, np.full(6, 0.5), np.full(6, 0.5), None, bins=2)
        self.assertTrue(c["calibration_buckets"])
        self.assertTrue(
            c["calibration"]["calibration_buckets_calibrated"])
        # platt None -> params a/b carry None (deployed map absent), still shape-correct
        self.assertIsNone(c["calibration"]["params"]["a"])
        self.assertIsNone(c["calibration"]["params"]["b"])
        self.assertEqual(c["calibration"]["method"], "platt")

    def test_preq_vs_deployed_metrics_are_distinct_pools(self):
        """metrics_calibrated uses the PREQUENTIAL per-fold values passed in
        (cal), and must differ from the raw metrics when the two pools differ."""
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        # raw probabilities perfectly separated, cal flatter -> ece differs
        raw = np.array([0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01])
        cal = np.full(8, 0.5)
        c = build_calibration(y, raw, cal, None, bins=2)
        self.assertNotEqual(c["metrics"]["ece"], c["metrics"]["ece_calibrated"])

    def test_raw_and_calibrated_share_favored_mask(self):
        """Both bucket sets run >= 50% and bin the SAME favored side (mask from
        the raw blend), so their per-bin counts/labels line up."""
        y = np.array([1, 0, 1, 0, 1, 0])
        raw = np.array([0.6, 0.6, 0.7, 0.4, 0.4, 0.9])
        # calibrated values stay on the raw side (Platt monotone in practice)
        cal = np.array([0.58, 0.62, 0.72, 0.42, 0.38, 0.88])
        c = build_calibration(y, raw, cal, None, bins=2)
        raw_b = c["calibration_buckets"]
        cal_b = c["calibration"]["calibration_buckets_calibrated"]
        for x in raw_b + cal_b:
            lo = int(x["bucket"].split("-")[0].rstrip("%"))
            self.assertGreaterEqual(lo, 50)
        # same mask (from raw) -> identical bucket grouping/counts
        self.assertEqual([x["bucket"] for x in raw_b],
                         [x["bucket"] for x in cal_b])
        self.assertEqual([x["count"] for x in raw_b],
                         [x["count"] for x in cal_b])
        self.assertTrue(raw_b)  # non-empty favored curve


class TestBuildHistoryFrame(unittest.TestCase):
    def test_exact_column_contract(self):
        oof = _oof_meta()
        h = build_history_frame(oof_meta=oof, oof_raw=np.full(len(oof), 0.6),
                                oof_cal=np.full(len(oof), 0.62),
                                sealed_meta=pd.DataFrame(),
                                sealed_raw=np.array([]), sealed_cal=np.array([]))
        self.assertEqual(list(h.columns), HISTORY_COLUMNS)
        # required frontend/mlc columns all present
        for col in ("game_date", "home_team", "away_team",
                    "home_win_prob_model", "correct", "model_pick",
                    "home_score", "away_score", "actual_winner", "game_status"):
            self.assertIn(col, h.columns)

    def test_pick_winner_correct_and_final(self):
        oof = _oof_meta(n=2)
        oof.loc[0, "home_team"], oof.loc[0, "away_team"] = "H0", "A0"
        oof.loc[0, "home_score"], oof.loc[0, "away_score"] = 24, 10  # H0 wins
        oof.loc[1, "home_team"], oof.loc[1, "away_team"] = "H1", "A1"
        oof.loc[1, "home_score"], oof.loc[1, "away_score"] = 3, 28     # A1 wins
        h = build_history_frame(oof_meta=oof, oof_raw=np.array([0.9, 0.3]),
                                oof_cal=np.array([0.88, 0.32]),
                                sealed_meta=pd.DataFrame(),
                                sealed_raw=np.array([]), sealed_cal=np.array([]))
        self.assertEqual(h.iloc[0]["model_pick"], "H0")
        self.assertEqual(h.iloc[0]["actual_winner"], "H0")
        self.assertTrue(bool(h.iloc[0]["correct"]))
        self.assertEqual(h.iloc[1]["model_pick"], "A1")
        self.assertEqual(h.iloc[1]["actual_winner"], "A1")
        self.assertTrue(bool(h.iloc[1]["correct"]))
        self.assertEqual(set(h["game_status"]), {"Final"})

    def test_leakfree_oof_excludes_sealed_season(self):
        """OOF rows (2021-2024) never carry the sealed 2025 season; sealed
        rows are appended separately and tagged 2025."""
        oof = _oof_meta(seasons=(2021, 2022, 2023, 2024))
        sealed = _oof_meta(seasons=(SEALED_SEASON,))
        h = build_history_frame(oof_meta=oof, oof_raw=np.full(len(oof), 0.5),
                                oof_cal=np.full(len(oof), 0.5),
                                sealed_meta=sealed,
                                sealed_raw=np.full(len(sealed), 0.5),
                                sealed_cal=np.full(len(sealed), 0.5))
        oof_rows = h[h["season"] != SEALED_SEASON]
        sealed_rows = h[h["season"] == SEALED_SEASON]
        self.assertEqual(len(oof_rows), len(oof))
        self.assertEqual(set(oof_rows["season"]), {2021, 2022, 2023, 2024})
        self.assertEqual(len(sealed_rows), len(sealed))
        self.assertEqual(set(sealed_rows["season"]), {SEALED_SEASON})
        # whole-history count == OOF + sealed
        self.assertEqual(len(h), len(oof) + len(sealed))

    def test_correct_flag_when_pick_misses(self):
        oof = _oof_meta(n=1)
        oof.loc[0, "home_team"], oof.loc[0, "away_team"] = "H0", "A0"
        oof.loc[0, "home_score"], oof.loc[0, "away_score"] = 10, 35  # A0 wins
        h = build_history_frame(oof_meta=oof, oof_raw=np.array([0.9]),  # pick H0
                                oof_cal=np.array([0.9]), sealed_meta=pd.DataFrame(),
                                sealed_raw=np.array([]), sealed_cal=np.array([]))
        self.assertEqual(h.iloc[0]["model_pick"], "H0")
        self.assertEqual(h.iloc[0]["actual_winner"], "A0")
        self.assertFalse(bool(h.iloc[0]["correct"]))


if __name__ == "__main__":
    unittest.main()