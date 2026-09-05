"""Tests for the NFL window-extension ablation (run_nfl_window_ablation.py).

Pins, per the task spec:
  - the served market-free 12-pool (DEPLOYED_12 — no market features anywhere);
  - arm compositions (W2019/W2016/W2014 boundary + window mapping) and the
    constant sealed-2025 hold-out;
  - the survey metrics + the coverage-floor gating that decides which
    boundaries become arms;
  - the window-parameterized walk-forward on synthetic data (a wider window
    than production, without touching the network), incl. no-2025-in-pool
    leakage and no market columns in the model input.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import run_nfl_window_ablation as w


def _synth_sched_pbp(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small synthetic schedule + pbp for one season (survey metrics)."""
    rows = []
    for i in range(6):
        rows.append({
            "season": season, "game_id": f"{season}_G{i}",
            "away_score": 17 if i % 2 == 0 else None,
            "home_score": 20 if i % 2 == 0 else None,
            "stadium": "Stadium A" if i != 5 else None,
            "roof": "dome" if i < 3 else "outdoors",
            "gametime": "13:00" if i != 4 else None,
        })
    sched = pd.DataFrame(rows)
    pbp_rows = []
    for gi in range(6):
        for p in range(4):
            pbp_rows.append({
                "season": season, "game_id": f"{season}_G{gi}",
                "epa": None if (gi * 4 + p) % 3 == 0 else 0.1,
                "qb_epa": None if (gi * 4 + p) % 4 == 0 else 0.05,
                "game_seconds_remaining": None if p == 0 else 1800.0,
            })
    return sched, pd.DataFrame(pbp_rows)


class TestDeployedPool(unittest.TestCase):
    def test_depleted_12_is_the_served_market_free_pool(self):
        self.assertEqual(len(w.DEPLOYED_12), 12)
        self.assertEqual(w.DEPLOYED_12, [
            "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
            "ewm_net_pts_diff", "ewm_ypp_diff",
            "pace_plays_min_diff", "rest_short_diff", "div_game",
            "travel_miles_diff", "altitude_home", "prime_time",
        ])
        for c in w.DEPLOYED_12:
            self.assertNotIn("market", c.lower())
            self.assertNotIn("line", c.lower())
            self.assertNotIn("implied", c.lower())

    def test_arm_features_filters_to_deployed_only(self):
        feats = pd.DataFrame([{**{c: 0.0 for c in w.DEPLOYED_12},
                               "market_home_implied": 0.62,
                               "spread_line": 3.0, "home_win": 1}])
        self.assertEqual(w.arm_features(feats), w.DEPLOYED_12)


class TestBoundaryGeometry(unittest.TestCase):
    def test_boundary_map(self):
        self.assertEqual(w.BOUNDARIES,
                         {"W2019": 2019, "W2016": 2016, "W2014": 2014})
        self.assertEqual(w.SEALED_SEASON, 2025)
        self.assertEqual(w.TRAIN_END, 2024)
        self.assertEqual(w.VAL_SEASONS, [2021, 2022, 2023, 2024])

    def test_train_windows(self):
        for name, b in w.BOUNDARIES.items():
            train = list(range(b, w.TRAIN_END + 1))
            self.assertEqual(train[0], b)
            self.assertEqual(train[-1], 2024)
            self.assertEqual(len(train), 2024 - b + 1)

    def test_no_feature_source_points_at_market(self):
        for c in w.DEPLOYED_12:
            self.assertIn(c, w.FEATURE_SOURCE)
        for src in w.FEATURE_SOURCE.values():
            self.assertIn(src, w.FLOORS)


class TestSurvey(unittest.TestCase):
    def test_survey_season_metrics(self):
        sched, pbp = _synth_sched_pbp(2020)
        row = w.survey_season(2020, sched, pbp)
        # 6 scheduled, 3 decided (i even) -> 0.5
        self.assertEqual(row["sched_games"], 6)
        self.assertEqual(row["decided_games"], 3)
        self.assertEqual(row["decided_rate"], 0.5)
        self.assertEqual(row["pbp_games"], 6)
        # epa non-null: 16/24 (8 nulls when (gi*4+p)%3==0) — survey rounds to 1dp
        self.assertEqual(row["epa_pct"], round(100.0 * 16 / 24, 1))
        # qb_epa null when p==0 (6 nulls / 24); pace null when p==0 (6 / 24)
        self.assertEqual(row["qb_epa_pct"], round(100.0 * 18 / 24, 1))
        self.assertEqual(row["pace_pct"], round(100.0 * 18 / 24, 1))
        # stadium null at i==5 -> 5/6; roof all present; gametime null i==4
        self.assertEqual(row["stadium_pct"], round(100.0 * 5 / 6, 1))
        self.assertEqual(row["roof_pct"], 100.0)
        self.assertEqual(row["gametime_pct"], round(100.0 * 5 / 6, 1))

    def test_survey_table_aggregates(self):
        rows = [w.survey_season(s, *_synth_sched_pbp(s)) for s in (2016, 2017)]
        per_season, agg = w.survey_table(rows)
        self.assertEqual(len(per_season), 2)
        a2016 = agg["W2016"]
        self.assertEqual(a2016["decided_games"], 3 + 3)      # both seasons
        self.assertEqual(a2016["pbp_games"], 6 + 6)
        self.assertEqual(a2016["pbp_games_rate"], 1.0)
        self.assertEqual(a2016["sched_games"], 12)
        # core = the seasons PRESENT in the table at/after the boundary
        self.assertEqual(len(a2016["core_seasons"]), 2)
        self.assertEqual(a2016["core_seasons"], [2016, 2017])
        self.assertEqual(a2016["warmup_decided"], 0)         # no 2015 rows
        self.assertEqual(a2016["decided_rate"], 0.5)
        # no rows >= 2019 in the table -> the W2019 boundary has no core data
        self.assertIsNone(agg["W2019"])

    def test_survey_verdict_gates_boundaries(self):
        good = {k: v for k, v in w.FLOORS.items()}
        ok, reasons = w.survey_verdict(good)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])
        bad = dict(good)
        bad["epa_pct"] = 0.5
        ok, reasons = w.survey_verdict(bad)
        self.assertFalse(ok)
        self.assertTrue(any("epa_pct" in r for r in reasons))
        self.assertFalse(w.survey_verdict(None)[0])

    def test_survey_feature_source_buildability(self):
        """Every one of the 12 features maps to a measured survey metric."""
        for feature, src in w.FEATURE_SOURCE.items():
            self.assertIn(feature, w.DEPLOYED_12)
            self.assertIn(src, w.FLOORS)


class TestWindowWalkForward(unittest.TestCase):
    def _synth_window_feats(self) -> pd.DataFrame:
        from test_nfl_moneyline import _synth_fold_frame
        feats = _synth_fold_frame(seasons=list(range(2016, 2026)))
        rng = np.random.default_rng(7)
        extras = {
            "win_pct_diff": 0.5, "ewm_net_pts_diff": 0.0,
            "ewm_ypp_diff": 0.0, "pace_plays_min_diff": 0.0,
            "rest_short_diff": 0.0, "div_game": 0.0,
            "travel_miles_diff": 0.0, "altitude_home": 0.0,
            "prime_time": 0.0,
        }
        for c, val in extras.items():
            feats[c] = rng.normal(size=len(feats)) + val
            # train-side missingness only (2025 fully observed, like the sealed
            # rows in real frames — decisions stay clean for the hold-out)
            pre25 = feats.index[feats["season"] < 2025]
            feats.loc[pre25[::7], c] = np.nan
        return feats

    def test_wider_window_runs_and_keeps_2025_sealed(self):
        feats = self._synth_window_feats()
        train = list(range(2016, 2025))       # 2016..2024 — wider than prod
        res = w.run_walk_forward_window(feats, w.DEPLOYED_12, train)
        geo = res["fold_geometry"]
        self.assertEqual(geo["train_seasons"], train)
        self.assertEqual(geo["sealed_season"], 2025)
        self.assertEqual(geo["val_seasons"], [2021, 2022, 2023, 2024])
        self.assertGreater(geo["pooled_oof_games"], 0)
        self.assertEqual(geo["sealed_games"], 144)   # 18 weeks x 8 games x 2025
        for key in ("sealed_2025", "pooled_preq_2021_2024"):
            m = res[key]["model_platt"]
            for k in ("logloss", "auc", "ece"):
                self.assertIn(k, m)
        # market-absence in the deployed input set
        self.assertEqual(res["_deployed"]["features"], w.DEPLOYED_12)

    def test_warmup_larger_than_production_matches_production_geometry(self):
        """W2019 (production window) uses the SAME val/sealed split as prod."""
        from nfl_run_engine_legacy_windows import TRAIN_SEASONS
        feats = self._synth_window_feats()
        res = w.run_walk_forward_window(feats, w.DEPLOYED_12, TRAIN_SEASONS)
        self.assertEqual(res["fold_geometry"]["train_seasons"], TRAIN_SEASONS)
        self.assertEqual(res["fold_geometry"]["sealed_games"], 144)


if __name__ == "__main__":
    unittest.main()