"""Tests for the StatsAPI results overlay and ESPN game status fix."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from results import (
    apply_official_results,
    fetch_mlb_results,
    merge_result_cache,
)


class TestApplyOfficialResults(unittest.TestCase):
    """apply_official_results should fix frozen finals and null live labels."""

    def _make_games(self):
        return pd.DataFrame([
            # Aug 21 STL@PHI: Statcast had frozen partial (6-5 STL)
            {"game_pk": 823420, "game_date": "2026-08-21",
             "home_team": "PHI", "away_team": "STL",
             "home_score": 5, "away_score": 6, "home_win": 0.0,
             "total_runs": 11},
            # Aug 21 WSH@MIA: Statcast had 2-2 tie (partial)
            {"game_pk": 823830, "game_date": "2026-08-21",
             "home_team": "MIA", "away_team": "WSH",
             "home_score": 2, "away_score": 2, "home_win": None,
             "total_runs": 4},
            # Aug 22 PIT@LAD: live mid-game (0-7)
            {"game_pk": 824010, "game_date": "2026-08-22",
             "home_team": "LAD", "away_team": "PIT",
             "home_score": 7, "away_score": 0, "home_win": 1.0,
             "total_runs": 7},
        ])

    def _make_results(self):
        return pd.DataFrame([
            # PHI won 7-6 in 10 innings (official final)
            {"game_pk": 823420, "game_date": "2026-08-21",
             "home_score": 7.0, "away_score": 6.0,
             "home_win": 1.0, "is_final": True},
            # MIA won 3-2 (official final)
            {"game_pk": 823830, "game_date": "2026-08-21",
             "home_score": 3.0, "away_score": 2.0,
             "home_win": 1.0, "is_final": True},
            # PIT@LAD still in progress
            {"game_pk": 824010, "game_date": "2026-08-22",
             "home_score": np.nan, "away_score": np.nan,
             "home_win": np.nan, "is_final": False},
        ])

    def test_frozen_final_corrected(self):
        """PHI-STL: frozen 5-6 → official 7-6 (PHI wins, not STL)."""
        df = apply_official_results(self._make_games(), self._make_results())
        phi = df[df["game_pk"] == 823420].iloc[0]
        self.assertEqual(phi["home_score"], 7.0)
        self.assertEqual(phi["away_score"], 6.0)
        self.assertEqual(phi["home_win"], 1.0)  # PHI won

    def test_tie_resolved_to_final(self):
        """WSH-MIA: frozen 2-2 → official 3-2 (MIA wins)."""
        df = apply_official_results(self._make_games(), self._make_results())
        mia = df[df["game_pk"] == 823830].iloc[0]
        self.assertEqual(mia["home_score"], 3.0)
        self.assertEqual(mia["away_score"], 2.0)
        self.assertEqual(mia["home_win"], 1.0)

    def test_live_game_label_nulled(self):
        """PIT-LAD: live game → home_win set to NaN (never a training label)."""
        df = apply_official_results(self._make_games(), self._make_results())
        lad = df[df["game_pk"] == 824010].iloc[0]
        self.assertTrue(pd.isna(lad["home_win"]))

    def test_duplicate_game_pk_rows_do_not_crash(self):
        """StatsAPI lists rescheduled games under multiple dates → the same
        gamePk appears twice. This crashed set_index().to_dict('index') and
        silently disabled the overlay on every real run."""
        results = pd.concat([
            self._make_results(),
            # Duplicate listing of 823420 under a different date (postponed
            # then played), same final.
            pd.DataFrame([{
                "game_pk": 823420, "game_date": "2026-08-22",
                "home_score": 7.0, "away_score": 6.0,
                "home_win": 1.0, "is_final": True,
            }]),
        ], ignore_index=True)
        df = apply_official_results(self._make_games(), results)
        phi = df[df["game_pk"] == 823420].iloc[0]
        self.assertEqual(phi["home_score"], 7.0)
        self.assertEqual(phi["home_win"], 1.0)

    def test_empty_results_returns_original(self):
        """No results → games unchanged."""
        games = self._make_games()
        df = apply_official_results(games, pd.DataFrame(columns=[
            "game_pk", "game_date", "home_score", "away_score",
            "home_win", "is_final"]))
        pd.testing.assert_frame_equal(df, games)

    def test_missing_game_pk_returns_original(self):
        """Games with no game_pk → results can't match."""
        games = pd.DataFrame([{"home_team": "X", "away_team": "Y",
                               "home_win": 0.0}])
        res = pd.DataFrame([{"game_pk": 999, "home_score": 1, "away_score": 0,
                             "home_win": 1.0, "is_final": True,
                             "game_date": "2026-08-22"}])
        df = apply_official_results(games, res)
        self.assertEqual(df.iloc[0]["home_win"], 0.0)  # unchanged

    def test_fallback_matches_by_date_and_teams(self):
        """Slate rows without game_pk are overlaid via (date, teams)."""
        games = pd.DataFrame([
            # frozen mid-game final on the ESPN-built slate (no game_pk)
            {"game_id": "20260821_PIT@LAD", "game_date": "2026-08-21",
             "home_team": "LAD", "away_team": "PIT",
             "home_score": 4, "away_score": 4, "home_win": None,
             "total_runs": 8, "game_state": "in"},
            # live game must keep home_win nulled
            {"game_id": "20260822_SF@BOS", "game_date": "2026-08-22",
             "home_team": "BOS", "away_team": "SF",
             "home_score": 0, "away_score": 3, "home_win": 1.0,
             "total_runs": 3, "game_state": "in"},
        ])
        res = pd.DataFrame([
            {"game_pk": 823911, "game_date": "2026-08-21",
             "home_team": "LAD", "away_team": "PIT",
             "home_score": 5.0, "away_score": 4.0,
             "home_win": 1.0, "is_final": True},
            {"game_pk": 824718, "game_date": "2026-08-22",
             "home_team": "BOS", "away_team": "SF",
             "home_score": np.nan, "away_score": np.nan,
             "home_win": np.nan, "is_final": False},
        ])
        df = apply_official_results(games, res)
        lad = df[df["game_id"] == "20260821_PIT@LAD"].iloc[0]
        self.assertEqual(lad["home_score"], 5.0)
        self.assertEqual(lad["away_score"], 4.0)
        self.assertEqual(lad["home_win"], 1.0)
        self.assertEqual(lad["total_runs"], 9)
        bos = df[df["game_id"] == "20260822_SF@BOS"].iloc[0]
        self.assertTrue(pd.isna(bos["home_win"]))

    def test_fallback_canonicalizes_team_codes(self):
        """CHW/OAK/ARI alias to the canonical Statcast codes when joining."""
        games = pd.DataFrame([{"game_id": "g1", "game_date": "2026-08-21",
                               "home_team": "CWS", "away_team": "NYM",
                               "home_score": 4, "away_score": 4,
                               "home_win": None, "total_runs": 8}])
        res = pd.DataFrame([{"game_pk": 1, "game_date": "2026-08-21",
                             "home_team": "CHW", "away_team": "NYM",
                             "home_score": 6.0, "away_score": 4.0,
                             "home_win": 1.0, "is_final": True}])
        df = apply_official_results(games, res)
        self.assertEqual(df.iloc[0]["home_score"], 6.0)
        self.assertEqual(df.iloc[0]["home_win"], 1.0)


class TestFetchMlbResults(unittest.TestCase):
    """fetch_mlb_results should parse StatsAPI correctly."""

    def test_empty_date_range_returns_empty(self):
        """Off-season date returns empty frame with correct columns."""
        df = fetch_mlb_results(date(2025, 12, 1), date(2025, 12, 2))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("game_pk", df.columns)
        self.assertIn("is_final", df.columns)

    def test_regular_season_date_has_games(self):
        """A mid-season date should return some completed games."""
        df = fetch_mlb_results(date(2026, 7, 15), date(2026, 7, 15))
        if not df.empty:
            self.assertIn("game_pk", df.columns)
            self.assertIn("is_final", df.columns)
            # All returned games should be from the requested date
            self.assertTrue(all(d == "2026-07-15" for d in df["game_date"]))

    def test_multi_year_range_fetched_in_yearly_chunks(self):
        """The schedule endpoint silently truncates long ranges (a full
        2-season query returned ~3,000 games vs ~5,900 per-year), so the
        fetcher must split by calendar year and combine the chunks."""
        from unittest.mock import patch, MagicMock
        import backend.results as results_mod

        calls = []

        def _fake_get(url, params=None, timeout=None):
            calls.append((params["startDate"], params["endDate"]))
            resp = MagicMock()
            resp.status_code = 200
            year = params["startDate"][:4]
            resp.json.return_value = {"dates": [{
                "date": f"{year}-07-15",
                "games": [{
                    "gamePk": int(year) * 100000,
                    "status": {"abstractGameState": "Final"},
                    "teams": {
                        "home": {"score": 5},
                        "away": {"score": 3},
                    },
                }],
            }]}
            return resp

        with patch.object(results_mod.requests, "get", side_effect=_fake_get):
            df = fetch_mlb_results(date(2025, 3, 1), date(2026, 8, 23))

        self.assertEqual(len(calls), 2, f"expected 2 yearly chunks, got {calls}")
        self.assertEqual(calls[0], ("2025-03-01", "2025-12-31"))
        self.assertEqual(calls[1], ("2026-01-01", "2026-08-23"))
        self.assertEqual(sorted(df["game_pk"].tolist()), [202500000, 202600000])

    def test_suspended_game_keeps_scored_listing(self):
        """Suspended/resumed games appear under two dates — the original
        listing can be Final-with-no-score. The scored listing must win the
        dedupe, else the overlay goes blind to the game entirely."""
        from unittest.mock import patch, MagicMock
        import backend.results as results_mod

        def _listing(pk, date_str, hs, asc, hw):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"dates": [{
                "date": date_str,
                "games": [{
                    "gamePk": pk,
                    "status": {"abstractGameState": "Final"},
                    "teams": {
                        "home": {"score": hs},
                        "away": {"score": asc},
                    },
                    # home_win derived by parser; emulate via scores only
                }],
            }]}
            return resp

        # NOTE: fetch_mlb_results parses scores itself; a None-score listing
        # is emulated with missing score keys. Both listings arrive in ONE
        # payload (same year-chunk), as the real endpoint returns them.
        dates_payload = []
        for date_str, hs, asc in [("2025-04-29", None, None),   # suspended
                                  ("2025-04-30", 0, 6)]:        # completion
            teams_home = {"score": hs} if hs is not None else {}
            teams_away = {"score": asc} if asc is not None else {}
            dates_payload.append({
                "date": date_str,
                "games": [{
                    "gamePk": 778134,
                    "status": {"abstractGameState": "Final"},
                    "teams": {"home": teams_home, "away": teams_away},
                }],
            })
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"dates": dates_payload}

        with patch.object(results_mod.requests, "get", return_value=resp):
            df = fetch_mlb_results(date(2025, 4, 29), date(2025, 4, 30))

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["home_score"], 0.0)
        self.assertEqual(row["away_score"], 6.0)
        self.assertEqual(row["home_win"], 0.0)


class TestMergeResultCache(unittest.TestCase):
    """merge_result_cache should dedupe keeping the newest/final copy."""

    def test_merges_new_games(self):
        cached = pd.DataFrame([{
            "game_pk": 1, "game_date": "2026-08-21",
            "home_score": 5, "away_score": 6, "home_win": 0.0,
            "is_final": True}])
        fresh = pd.DataFrame([{
            "game_pk": 2, "game_date": "2026-08-21",
            "home_score": 3, "away_score": 2, "home_win": 1.0,
            "is_final": True}])
        merged = merge_result_cache(cached, fresh)
        self.assertEqual(len(merged), 2)

    def test_final_overwrites_partial(self):
        """A final result should overwrite a partial (non-final) one."""
        cached = pd.DataFrame([{
            "game_pk": 1, "game_date": "2026-08-21",
            "home_score": np.nan, "away_score": np.nan,
            "home_win": np.nan, "is_final": False}])
        fresh = pd.DataFrame([{
            "game_pk": 1, "game_date": "2026-08-21",
            "home_score": 7, "away_score": 6,
            "home_win": 1.0, "is_final": True}])
        merged = merge_result_cache(cached, fresh)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["home_score"], 7)
        self.assertTrue(merged.iloc[0]["is_final"])


class TestEspnGameState(unittest.TestCase):
    """ESPN parser should only set home_win for final games."""

    def _make_event(self, state="post", home_score="7", away_score="6"):
        return {
            "date": "2026-08-22T17:05Z",
            "competitions": [{
                "status": {"type": {"state": state, "detail": "Final" if state == "post" else "In Progress"}},
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": "PHI"},
                     "score": home_score},
                    {"homeAway": "away", "team": {"abbreviation": "STL"},
                     "score": away_score},
                ],
            }],
        }

    def test_post_sets_home_win(self):
        from data_ingestion import _parse_espn_event
        result = _parse_espn_event(self._make_event("post"))
        self.assertEqual(result["home_win"], 1.0)
        self.assertEqual(result["game_state"], "post")

    def test_in_nulls_home_win(self):
        from data_ingestion import _parse_espn_event
        result = _parse_espn_event(self._make_event("in"))
        self.assertIsNone(result["home_win"])
        self.assertEqual(result["game_state"], "in")

    def test_pre_nulls_home_win(self):
        from data_ingestion import _parse_espn_event
        result = _parse_espn_event(self._make_event("pre", "", ""))
        self.assertIsNone(result["home_win"])
        self.assertEqual(result["game_state"], "pre")

    def test_live_scores_carry_through(self):
        """Live game should carry scores for display but no home_win."""
        from data_ingestion import _parse_espn_event
        result = _parse_espn_event(self._make_event("in", "0", "7"))
        self.assertEqual(result["home_score"], 0)
        self.assertEqual(result["away_score"], 7)
        self.assertIsNone(result["home_win"])
        self.assertEqual(result["game_state"], "in")


if __name__ == "__main__":
    unittest.main()
