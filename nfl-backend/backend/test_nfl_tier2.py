"""Tier-2 (venue/travel/schedule) candidate tests — pure, no-network.

Covers the six static candidates composed by ``nfl_features``:
travel_miles_diff, timezone_diff, altitude_home, turf_home, prime_time,
neutral_site. Invariants pinned:
- haversine correctness on known coordinate pairs + NaN propagation.
- >=95% coverage on a representative decided frame built from the REAL
  committed venue table (real stadium names).
- missing source fields / unknown stadiums degrade to NaN (never fabricated).
- end-to-end: ``build_features`` emits all six candidates.
- harness smoke: run_tier2_ablation arms compose as 10/16/13 and
  ``run_walk_forward`` executes with the new columns on synthetic data.
"""
import unittest

import numpy as np
import pandas as pd

from nfl_features import (
    VENUE_FEATURES, _compose_venue_candidates, _haversine_miles,
    _team_home_stadium_map, _venue_facts, build_features, build_slate_features,
)
from run_tier2_ablation import VENUE_3_FEATURES, build_arms


def _representative_decided_frame(n_per_team: int = 3) -> pd.DataFrame:
    """A decided frame over the real 32-team home-venue universe: every game
    at a real stadium with real surface/gametime/location values."""
    team_home = _team_home_stadium_map()
    teams = sorted(team_home)
    home = {t: team_home[t] for t in teams}
    away = {t: next(team_home[u] for u in teams if u != t) for t in teams}
    rows = []
    gid = 0
    for t in teams:
        for k in range(n_per_team):
            away_team = teams[(teams.index(t) + k + 1) % len(teams)]
            rows.append({
                "game_id": f"G{gid}", "season": 2024, "week": (k % 18) + 1,
                "gameday": pd.Timestamp("2024-09-08") + pd.Timedelta(days=7 * k),
                "home_team": t, "away_team": away_team,
                "home_score": 24, "away_score": 17,
                "stadium": home[t],
                "surface": "grass" if k % 2 == 0 else "fieldturf",
                "gametime": "13:00" if k % 2 == 0 else "20:15",
                "location": "Neutral" if (k == 1) else "Home",
            })
            gid += 1
    return pd.DataFrame(rows)


class TestHaversine(unittest.TestCase):
    def test_known_pair(self):
        """NYC -> Los Angeles is ~2446 miles; within 1% of the great-circle value."""
        d = _haversine_miles([40.7128], [-74.0060], [34.0522], [-118.2437])
        self.assertAlmostEqual(float(d[0]), 2446.0, delta=25.0)

    def test_zero_distance(self):
        d = _haversine_miles([40.0], [-75.0], [40.0], [-75.0])
        self.assertAlmostEqual(float(d[0]), 0.0, places=6)

    def test_nan_propagates(self):
        d = _haversine_miles([np.nan], [-75.0], [40.0], [-75.0])
        self.assertTrue(np.isnan(d[0]))

    def test_vectorized(self):
        d = _haversine_miles([40.0, 40.0], [-75.0, -75.0], [41.0, 41.0], [-74.0, -74.0])
        self.assertEqual(len(d), 2)
        self.assertTrue(np.all(d > 50.0))   # ~62 mi NY metro


class TestVenueTable(unittest.TestCase):
    def test_all_32_teams_have_home_venues(self):
        # real committed table covers every team that appears in the leagues
        team_home = _team_home_stadium_map()
        for t in ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
                  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
                  "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
                  "NYJ", "OAK", "PHI", "PIT", "SEA", "SF", "TB", "TEN",
                  "WAS"]:
            self.assertIn(t, team_home, f"{t} must map to a real home venue")
            self.assertIn(team_home[t], _venue_facts())

    def test_real_facts_spot_check(self):
        """Mile High ~5,236 ft and Arrowhead in America/Chicago — real data."""
        f = _venue_facts()
        self.assertAlmostEqual(f["Empower Field at Mile High"]["altitude_ft"],
                               5236.0, delta=80.0)
        self.assertEqual(f["GEHA Field at Arrowhead Stadium"]["tz"],
                         "America/Chicago")


class TestCoverage(unittest.TestCase):
    def test_coverage_ge_95pct_representative_frame(self):
        df = _representative_decided_frame()
        out = _compose_venue_candidates(df, None)
        for f in VENUE_FEATURES:
            cov = float(out[f].notna().mean())
            self.assertGreaterEqual(cov, 0.95, f"{f} coverage {cov:.1%} < 95%")


class TestDegradation(unittest.TestCase):
    def test_unknown_stadium_and_missing_fields_all_nan(self):
        df = pd.DataFrame([{
            "game_id": "G1", "season": 2024, "week": 1,
            "gameday": "2024-09-08", "home_team": "ZZ", "away_team": "YY",
            "home_score": 20, "away_score": 10,
            "stadium": "Mystery Bowl", "surface": "", "gametime": None,
            "location": None,
        }])
        out = _compose_venue_candidates(df, None)
        for f in VENUE_FEATURES:
            self.assertTrue(pd.isna(out[f].iloc[0]), f"{f} must degrade to NaN")

    def test_home_game_zero_travel_sign(self):
        """Home team plays at its home venue -> home leg 0, so the diff is the
        NEGATIVE away travel (signed, like the ELO diffs); stadium-share pairs
        (SoFi, MetLife) read exactly 0.0. Unknown venues still NaN."""
        df = _representative_decided_frame(n_per_team=1)
        out = _compose_venue_candidates(df, None)
        t = out["travel_miles_diff"]
        self.assertTrue((t <= 0).all())
        self.assertTrue((t < -1000).sum() >= 10)      # real cross-country legs
        self.assertEqual(float(t[out["home_team"] == "LA"].iloc[0]), 0.0)
        self.assertEqual(float(t[out["home_team"] == "NYG"].iloc[0]), 0.0)


class TestComposeEndToEnd(unittest.TestCase):
    def test_build_features_emits_all_six(self):
        rows = [
            dict(game_id="2019_01_KC_JAX", season=2019, week=1,
                 gameday="2019-09-08", home_team="KC", away_team="JAX",
                 home_score=26, away_score=40, roof="outdoors", temp=70.0,
                 wind=8.0, div_game=0, stadium="GEHA Field at Arrowhead Stadium",
                 surface="grass", gametime="16:25", location="Home"),
            dict(game_id="2019_01_GB_CHI", season=2019, week=1,
                 gameday="2019-09-05", home_team="CHI", away_team="GB",
                 home_score=3, away_score=10, roof="closed", temp=72.0,
                 wind=6.0, div_game=1, stadium="Soldier Field",
                 surface="fieldturf", gametime="20:20", location="Home"),
        ]
        decided = pd.DataFrame(rows).drop(columns=["roof", "stadium",
                                                  "surface", "gametime",
                                                  "location"])
        schedule = pd.DataFrame(rows)
        feats = build_features(decided, schedule, None)
        for f in VENUE_FEATURES:
            self.assertIn(f, feats.columns, f"{f} missing from build_features")
        kc = feats[feats["game_id"] == "2019_01_KC_JAX"].iloc[0]
        chi = feats[feats["game_id"] == "2019_01_GB_CHI"].iloc[0]
        # prime flag: 16:25 -> day, 20:20 -> evening
        self.assertEqual(kc["prime_time"], 0.0)
        self.assertEqual(chi["prime_time"], 1.0)
        # turf mapping: grass -> 0, fieldturf -> 1
        self.assertEqual(kc["turf_home"], 0.0)
        self.assertEqual(chi["turf_home"], 1.0)
        # KC home venue == game venue -> home leg 0 -> diff strictly negative
        self.assertLess(kc["travel_miles_diff"], 0.0)
        self.assertGreater(chi["altitude_home"], 0.0)

    def test_build_slate_features_emits_all_six(self):
        rows = [
            dict(game_id="2024_01_KC_BAL", season=2024, week=1,
                 gameday="2024-09-05", home_team="KC", away_team="BAL",
                 home_score=27, away_score=20, roof="outdoors", temp=75.0,
                 wind=5.0, div_game=0, stadium="GEHA Field at Arrowhead Stadium",
                 surface="grass", gametime="20:20", location="Home"),
            dict(game_id="2026_01_KC_CLE", season=2026, week=1,
                 gameday="2026-09-10", home_team="KC", away_team="CLE",
                 home_score=None, away_score=None, roof="outdoors", temp=76.0,
                 wind=6.0, div_game=0, stadium="GEHA Field at Arrowhead Stadium",
                 surface="grass", gametime="13:00", location="Home"),
        ]
        sched = pd.DataFrame(rows)
        decided = sched[sched["home_score"].notna()].copy()
        slate = build_slate_features(sched, None, decided, 2026)
        self.assertEqual(len(slate), 1)
        row = slate.iloc[0]
        for f in VENUE_FEATURES:
            self.assertIn(f, slate.columns)
            self.assertTrue(pd.notna(row[f]), f"{f} must resolve pre-game")


class TestHarnessSmoke(unittest.TestCase):
    def test_arms_compose_as_10_16_13(self):
        feats = pd.DataFrame([{**{c: 0.0 for c in (
            "elo_diff", "win_pct_diff", "rest_days_diff", "is_dome_home",
            "ewm_net_pts_diff", "ewm_qb_epa_play_diff", "ewm_ypp_diff",
            "pace_plays_min_diff", "rest_short_diff", "div_game")},
            **{c: 1.0 for c in VENUE_FEATURES}, "game_id": "G1"}])
        arms = build_arms(feats)
        self.assertEqual(len(arms["WITHOUT"]), 10)
        self.assertEqual(len(arms["VENUE"]), 16)
        self.assertEqual(len(arms["VENUE_3"]), 13)
        self.assertEqual(set(VENUE_3_FEATURES), {"travel_miles_diff",
                                                 "altitude_home", "prime_time"})

    def test_run_walk_forward_with_venue_columns(self):
        """The production walk-forward executes with the 16-column VENUE arm on
        synthetic data (network-free, ~30s)."""
        from nfl_run_engine_legacy_windows import TRAIN_SEASONS, SEALED_SEASON
        from nfl_moneyline import run_walk_forward
        from test_nfl_moneyline import _synth_fold_frame
        feats = _synth_fold_frame(seasons=TRAIN_SEASONS + [SEALED_SEASON])
        feats = feats[pd.notna(feats["elo_diff"])].copy()
        rng = np.random.default_rng(3)
        for c in VENUE_FEATURES:
            feats[c] = rng.normal(size=len(feats))
            feats.loc[::5, c] = np.nan   # some missingness, like real data
        res = run_walk_forward(feats, model_features=build_arms(feats)["VENUE"])
        self.assertIn("sealed_2025", res)
        self.assertIn("pooled_preq_2021_2024", res)
        for key in ("sealed_2025", "pooled_preq_2021_2024"):
            m = res[key]["model_platt"]
            for k in ("logloss", "auc", "ece"):
                self.assertIn(k, m)


if __name__ == "__main__":
    unittest.main()