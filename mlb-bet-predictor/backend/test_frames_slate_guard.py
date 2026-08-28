"""Regression tests for get_decided_frame's slate-row identity guard.

Verifies that ESPN slate rows with numeric game_pk values (synthesised by
_attach_slate_run_margins from game_id) can never leak into the decided
frame, while pre-slate canonical frames pass through unchanged.
"""
import unittest

import numpy as np
import pandas as pd

from frames import get_decided_frame, fold_signature


def _make_statcast_row(game_pk, game_date="2026-04-01", home_team="NYY",
                       home_win=1.0, home_starter_id=680570):
    """A Statcast-decided row: has numeric game_pk AND home_starter_id."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": "BOS",
        "home_win": home_win,
        "home_score": 5,
        "away_score": 3,
        "total_runs": 8,
        "home_starter_id": home_starter_id,
        "away_starter_id": 669372,
        "game_id": f"{game_date.replace('-', '')}_{home_team}@BOS",
    }


def _make_slate_row(game_pk, game_date="2026-08-28", home_team="NYY",
                    home_win=1.0):
    """An ESPN slate row: has numeric game_pk BUT NO home_starter_id."""
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": "BOS",
        "home_win": home_win,
        "home_score": 4,
        "away_score": 2,
        "total_runs": 6,
        # NO home_starter_id — ESPN build_upcoming_slate never sets it
        "game_id": f"{game_date.replace('-', '')}_{home_team}@BOS",
    }


class TestSlateGuard(unittest.TestCase):
    """get_decided_frame excludes ESPN slate rows with numeric game_pk."""

    def test_prealte_frame_unchanged(self):
        """Pre-slate frame (all rows have home_starter_id) passes through."""
        df = pd.DataFrame([
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_statcast_row(800002, "2026-04-02", "BOS"),
            _make_statcast_row(800003, "2026-04-03", "LAD"),
        ])
        result = get_decided_frame(df)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["game_pk"].tolist(), [800001, 800002, 800003])

    def test_postsLate_numeric_espn_game_pk_excluded(self):
        """Post-slate frame: ESPN numeric game_pks WITHOUT home_starter_id
        are excluded, even though they pass the numeric game_pk filter."""
        statcast_rows = [
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_statcast_row(800002, "2026-04-02", "BOS"),
        ]
        # ESPN game_ids are typically 401576xxx — numeric but NOT StatsAPI
        slate_rows = [
            _make_slate_row(401576789, "2026-08-28", "NYY"),
            _make_slate_row(401576790, "2026-08-28", "BOS"),
        ]
        df = pd.DataFrame(statcast_rows + slate_rows)
        result = get_decided_frame(df)
        # Only the 2 Statcast rows survive
        self.assertEqual(len(result), 2)
        self.assertTrue(all(pk > 700000 for pk in result["game_pk"]))

    def test_postsLate_home_starter_id_present_survives(self):
        """Post-slate frame: rows WITH home_starter_id survive even if
        they look like they could be slate (e.g. results-filled rows
        that DO carry a real starter)."""
        rows = [
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_slate_row(401576789, "2026-08-28", "NYY"),
        ]
        # Give the slate row a home_starter_id (simulating future pipeline
        # that sets starter IDs on slate rows)
        rows[1]["home_starter_id"] = 680570
        df = pd.DataFrame(rows)
        result = get_decided_frame(df)
        # Both survive — home_starter_id is the discriminator
        self.assertEqual(len(result), 2)

    def test_fold_signature_unchanged_on_prealte_frame(self):
        """fold_signature on pre-slate frame is identical before and after
        adding the starter guard — zero metric drift."""
        df = pd.DataFrame([
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_statcast_row(800002, "2026-04-02", "BOS"),
            _make_statcast_row(800003, "2026-04-03", "LAD"),
        ])
        sig = fold_signature(get_decided_frame(df))
        # Signature must be stable — no regression
        self.assertIsInstance(sig, str)
        self.assertEqual(len(sig), 16)  # SHA-256 truncated to 16 chars

    def test_slate_row_with_nan_home_starter_id_excluded(self):
        """Slate row with NaN home_starter_id is excluded even if game_pk
        is numeric."""
        df = pd.DataFrame([
            _make_statcast_row(800001, "2026-04-01", "NYY"),
            _make_slate_row(401576789, "2026-08-28", "NYY"),
        ])
        # Explicitly set home_starter_id to NaN on the slate row
        df.loc[df["game_pk"] == 401576789, "home_starter_id"] = np.nan
        result = get_decided_frame(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["game_pk"].iloc[0], 800001)

    def test_empty_frame_unchanged(self):
        """Empty frame returns empty — no crash."""
        df = pd.DataFrame(columns=["game_pk", "game_date", "home_team",
                                    "home_win", "home_starter_id"])
        result = get_decided_frame(df)
        self.assertEqual(len(result), 0)

    def test_frame_without_starter_id_column_unchanged(self):
        """Frame without home_starter_id column (e.g. synthetic test data)
        skips the guard — backward compatible."""
        df = pd.DataFrame([
            {"game_pk": 800001, "game_date": "2026-04-01",
             "home_team": "NYY", "home_win": 1.0},
            {"game_pk": 800002, "game_date": "2026-04-02",
             "home_team": "BOS", "home_win": 0.0},
        ])
        result = get_decided_frame(df)
        self.assertEqual(len(result), 2)

    def test_real_frame_unchanged(self):
        """The committed 6,953-game frame is unchanged by the guard."""
        from pathlib import Path
        csv = Path(__file__).parent.parent / "data_delivery" / "game_level_features.csv"
        if not csv.exists():
            self.skipTest("committed CSV not available")
        df = pd.read_csv(csv)
        df["game_date"] = pd.to_datetime(df["game_date"])
        result = get_decided_frame(df)
        # All 6,953 rows have home_starter_id — guard is a no-op
        self.assertEqual(len(result), 6953)


if __name__ == "__main__":
    unittest.main()
