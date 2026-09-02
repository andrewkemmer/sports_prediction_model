"""Tests for the run-line DEFENSIVE-expansion ablation harness
(run_mlb_runline_defense_ablation.py): arm construction over the 53-feature
C2 base, family column wiring, and the point-in-time (PIT) ladder rules
(game_date < target date; F3 trend = short minus long).

Run from the mlb-backend directory:
    python -m unittest backend.test_runline_defense_ablation -v
"""
from __future__ import annotations

import sys
import types
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if "resource" not in sys.modules:  # POSIX-only module, stub on Windows
    _res = types.ModuleType("resource")
    _res.getrusage = lambda *_: types.SimpleNamespace(ru_maxrss=0)
    _res.RUSAGE_SELF = 0
    sys.modules["resource"] = _res

import run_mlb_runline_defense_ablation as m  # noqa: E402
from run_engine import derive_run_features  # noqa: E402
from training import FEATURE_COLS  # noqa: E402


N_DAYS = 12  # must exceed F1's min-prior guard (10 prior games)
WIDE_COLS = ["game_pk", "game_date", "home_team", "away_team",
             "inning_topbot", "events", "launch_speed", "launch_angle",
             "hit_location"]


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two teams, N_DAYS consecutive days for the SAME matchup pair; a team's
    day-T feature must reflect only games strictly before T. Each prior day
    contributes exactly ONE scoring event against HOU as the fielding side
    (a Top-half single), so HOU's runs-allowed ladder at day k equals k-1."""
    games = []
    pbp_rows = []
    day0 = date(2026, 4, 1)
    for i in range(N_DAYS):
        d = day0 + timedelta(days=i)
        pk = 100 + i
        games.append({"game_pk": pk, "game_date": pd.Timestamp(d),
                      "home_team": "HOU", "away_team": "TEX",
                      "home_win": 0.0 if i % 2 == 0 else 1.0,
                      "home_score": 3 + i, "away_score": 2})
        pbp_rows.append({"game_pk": pk, "game_date": pd.Timestamp(d),
                         "home_team": "HOU", "away_team": "TEX",
                         "inning_topbot": "Top", "events": "single",
                         "launch_speed": 101.0, "launch_angle": 12.0,
                         "hit_location": 7.0})
        pbp_rows.append({"game_pk": pk, "game_date": pd.Timestamp(d),
                         "home_team": "HOU", "away_team": "TEX",
                         "inning_topbot": "Bottom", "events": "field_out",
                         "launch_speed": 88.0, "launch_angle": 30.0,
                         "hit_location": 6.0})
    return (pd.DataFrame(pbp_rows), pd.DataFrame(games))


class TestArmConstruction(unittest.TestCase):
    def test_c0_is_the_production_53_feature_view(self):
        kept, _ = derive_run_features(list(FEATURE_COLS))
        arms = m.arm_features()
        self.assertEqual(len(kept), 53)
        self.assertEqual(set(arms["C0"]), set(kept))

    def test_family_arms_add_only_their_per_side_columns(self):
        arms = m.arm_features()
        base = set(arms["C0"])
        self.assertEqual(sorted(set(arms["F1"]) - base),
                         sorted(m.ARM_LABELS["F1"]))
        self.assertEqual(sorted(set(arms["F2"]) - base),
                         sorted(m.ARM_LABELS["F2"]))
        self.assertEqual(sorted(set(arms["F3"]) - base),
                         sorted(m.ARM_LABELS["F3"]))
        self.assertEqual(sorted(set(arms["F4"]) - base),
                         sorted(m.ARM_LABELS["F4"]))

    def test_every_family_column_is_side_routed(self):
        # run_oof's split_side_view routes *_home/*_away to the right side;
        # no family column may be a bare (non-suffixed) feature.
        all_fam = {c for tag in ("F1", "F2", "F3", "F4")
                   for c in m.ARM_LABELS[tag]}
        bad = [c for c in all_fam
               if not (c.endswith("_home") or c.endswith("_away"))]
        self.assertEqual(bad, [])

    def test_drop_terms_never_include_defense_columns(self):
        arms = m.arm_features()
        for tag in ("C0", "F1", "F2", "F3", "F4"):
            dropped = set(m.arm_drop_terms(arms[tag]))
            self.assertTrue(dropped.issubset(set(FEATURE_COLS)))
        # Defense columns ride in run_features; dropped must stay FEATURE_COLS-only.
        for tag in ("F1", "F2", "F3", "F4"):
            dropped = set(m.arm_drop_terms(arms[tag]))
            self.assertFalse(dropped & set(m.ARM_LABELS[tag]))


class TestPitLadders(unittest.TestCase):
    def test_f1_uses_only_prior_games_and_f3_is_short_minus_long(self):
        pbp, games = _synthetic_frames()
        # The last day's features may only aggregate strictly-prior rows.
        f135 = m.build_f1_f3_f5(pbp, games)
        last = games["game_pk"].iloc[-1]
        idx = f135["game_pk"] == last
        f10 = float(f135.loc[idx, "team_runs_allowed_10g_home"].iloc[0])
        # Rolling MEAN: HOU allowed exactly 1 scoring event per prior day,
        # so the 10g/30g window averages to 1.0 and never leaks day T.
        self.assertEqual(f10, 1.0,
                         "ladder must average exactly the prior days' events")
        # F3 trend = 10g minus 30g of the same core (identical priors here).
        f30 = float(f135.loc[idx, "team_runs_allowed_30g_home"].iloc[0])
        self.assertAlmostEqual(f10, f30)
        # The first day has zero prior games -> guarded NaN (min-prior 10).
        first = games["game_pk"].iloc[0]
        self.assertTrue(np.isnan(
            f135.loc[f135["game_pk"] == first,
                     "team_runs_allowed_10g_home"].iloc[0]))

    def test_ladders_attach_columns_with_valid_empty_wide(self):
        pbp, games = _synthetic_frames()
        # Column-complete but row-empty wide cache: ladders must attach with
        # NaN (never crash on missing columns).
        wide = pd.DataFrame(columns=WIDE_COLS)
        out = m.build_defense_ladders(games, wide)
        for tag in ("F1", "F2", "F3", "F4"):
            cols = [c for c in m.ARM_LABELS[tag] if c in out.columns]
            self.assertTrue(len(cols) > 0, f"{tag} must attach columns")
        self.assertIn("game_pk", out.columns)
        self.assertEqual(len(out), len(games))


class TestF5Excluded(unittest.TestCase):
    def test_f5_reason_documents_coverage_floor(self):
        self.assertIn("61.6%", m.F5_EXCLUDED_REASON)
        self.assertNotIn("F5", m.ARM_LABELS)


if __name__ == "__main__":
    unittest.main(verbosity=2)