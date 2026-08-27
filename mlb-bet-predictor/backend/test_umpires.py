"""Tests for backend/umpires.py — maintained home-plate umpire data access.

Covers:
  1. Officials-payload parser: HP umpire extraction (id + name), null-umpire
     handling for missing/empty officials and crews without a Home Plate.
  2. Incremental maintenance: new seasons appended; seasons already present
     in the map are NEVER re-fetched; a failed fetch keeps the map intact.
  3. Real-frame coverage: 100% of the committed 4,481-game frame resolves an
     HP umpire from the maintained map (the data-access capability the
     scoping step established).
  4. Per-umpire diagnostics table shape (called-pitch placeholders null).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from umpires import (
    MAP_COLS,
    build_umpire_stats,
    fetch_season_umpires,
    load_umpire_map,
    maintain_umpire_map,
    parse_schedule_officials,
)

ROOT = Path(__file__).resolve().parents[1]


def _game(game_pk: int, day: str = "2026-08-20", season: int = 2026,
          game_type: str = "R", officials: list[dict] | None = None) -> dict:
    return {
        "gamePk": game_pk,
        "gameType": game_type,
        "season": season,
        "officialDate": day,
        "teams": {
            "home": {"team": {"abbreviation": "HOM"}},
            "away": {"team": {"abbreviation": "AWY"}},
        },
        "officials": officials if officials is not None else [],
    }


def _hp_official(official_id: int, name: str, off_type: str = "Home Plate") -> dict:
    return {"official": {"id": official_id, "fullName": name},
            "officialType": off_type}


def _payload(*games: dict) -> dict:
    return {"dates": [{"games": list(games)}]}


class TestParseScheduleOfficials(unittest.TestCase):
    def test_extracts_home_plate_umpire(self):
        payload = _payload(
            _game(778563, officials=[
                _hp_official(482631, "Mike Estabrook", "First Base"),
                _hp_official(427344, "Bill Miller", "Home Plate"),
                _hp_official(483561, "Lance Barrett", "Second Base"),
            ]))
        rows = parse_schedule_officials(payload)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["game_pk"], 778563)
        self.assertEqual(r["game_date"], "2026-08-20")
        self.assertEqual(r["home_team"], "HOM")
        self.assertEqual(r["away_team"], "AWY")
        self.assertEqual(r["hp_umpire_id"], 427344)
        self.assertEqual(r["hp_umpire_name"], "Bill Miller")
        self.assertEqual(r["season"], 2026)

    def test_missing_officials_null_umpire_no_crash(self):
        """A game with no officials array yields a null-umpire row."""
        payload = _payload(_game(778564, officials=None))
        rows = parse_schedule_officials(payload)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["hp_umpire_id"])
        self.assertIsNone(rows[0]["hp_umpire_name"])

    def test_empty_officials_list_null_umpire(self):
        payload = _payload(_game(778565, officials=[]))
        rows = parse_schedule_officials(payload)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["hp_umpire_id"])

    def test_crew_without_home_plate_null_umpire(self):
        payload = _payload(_game(778566, officials=[
            _hp_official(1, "A", "First Base"),
            _hp_official(2, "B", "Second Base"),
        ]))
        rows = parse_schedule_officials(payload)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["hp_umpire_id"])

    def test_live_payload_teams_fallback_to_name(self):
        """Real schedule payloads carry NO team ``abbreviation`` (only id /
        name / link) — the parser must fall back to the full name so the
        home_team / away_team columns are never blank."""
        game = _game(778568, officials=[_hp_official(427344, "Bill Miller")])
        game["teams"] = {
            "home": {"team": {"id": 116, "name": "Detroit Tigers"}},
            "away": {"team": {"id": 147, "name": "New York Yankees"}},
        }
        rows = parse_schedule_officials(_payload(game))
        self.assertEqual(rows[0]["home_team"], "Detroit Tigers")
        self.assertEqual(rows[0]["away_team"], "New York Yankees")

    def test_multiple_games_and_types_kept(self):
        """Regular + postseason games are all kept (the frame includes the
        2025 postseason)."""
        payload = _payload(
            _game(778567, officials=[_hp_official(10, "R1")]),
            _game(813027, day="2025-10-24", season=2025, game_type="W",
                  officials=[_hp_official(11, "W1")]),
            _game(999, day="2026-03-01", game_type="S", officials=[]),
        )
        rows = parse_schedule_officials(payload)
        self.assertEqual(len(rows), 3)
        types = {r["game_pk"]: r["hp_umpire_id"] for r in rows}
        self.assertEqual(types[813027], 11)   # postseason kept
        self.assertIsNone(types[999])          # spring kept, null ump


class TestMaintainUmpireMap(unittest.TestCase):
    def _season_df(self, season: int, n: int = 5) -> pd.DataFrame:
        rows = [{
            "game_pk": season * 1000 + i,
            "game_date": f"{season}-08-{10+i:02d}",
            "home_team": "HOM", "away_team": "AWY",
            "hp_umpire_id": 427344, "hp_umpire_name": "Bill Miller",
            "season": season,
        } for i in range(n)]
        return pd.DataFrame(rows, columns=MAP_COLS)

    def test_new_season_appends_existing_not_refetched(self):
        """A map that already has 2025 must NOT re-fetch 2025 — only the
        missing 2026 season is pulled and appended; 2025 rows preserved."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            existing = self._season_df(2025)
            existing.to_csv(base / "umpire_map.csv", index=False)

            with patch("umpires.fetch_season_umpires",
                       side_effect=lambda s: self._season_df(s)) as mock:
                res = maintain_umpire_map(date(2026, 8, 27), base=base)

            self.assertEqual(mock.call_count, 1)
            self.assertEqual(mock.call_args[0][0], 2026)
            self.assertEqual(res["seasons_fetched"], ["2026"])
            self.assertEqual(res["rows_added"], 5)
            m = load_umpire_map(base)
            self.assertEqual(len(m), 10)                    # 2025 + 2026
            self.assertEqual(set(m["season"]), {2025, 2026})
            # 2025 rows byte-identical (never re-fetched)
            self.assertEqual(len(m[m["season"] == 2025]), 5)

    def test_empty_map_fetches_all_seasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch("umpires.fetch_season_umpires",
                       side_effect=lambda s: self._season_df(s)) as mock:
                res = maintain_umpire_map(date(2026, 8, 27), base=base)
            self.assertEqual(mock.call_count, 2)
            fetched = [c[0][0] for c in mock.call_args_list]
            self.assertEqual(fetched, [2025, 2026])
            self.assertEqual(res["rows_added"], 10)
            m = load_umpire_map(base)
            self.assertEqual(len(m), 10)

    def test_frame_gap_triggers_season_refetch_appends_and_patches(self):
        """A decided frame game missing from a PRESENT season re-fetches only
        that season (zero fetches for gap-free seasons), appends the new
        game_pk, and patches a null-umpire row whose crew was assigned after
        the initial pull. Existing rows are never overwritten."""
        # Both seasons present: 2025 (never re-fetched) + 2026 with a
        # crew-less row (assigned later) and no row for the frame's newest
        # game 2026002 -- the ONLY fetch should be the 2026 gap-fill.
        existing_2026 = pd.DataFrame([
            {"game_pk": 2025000, "game_date": "2025-08-01",
             "home_team": "HOM", "away_team": "AWY",
             "hp_umpire_id": 427344, "hp_umpire_name": "Bill Miller",
             "season": 2025},
            {"game_pk": 2026000, "game_date": "2026-08-01",
             "home_team": "HOM", "away_team": "AWY",
             "hp_umpire_id": 427344, "hp_umpire_name": "Bill Miller",
             "season": 2026},
            # crew not yet assigned when the season was first fetched
            {"game_pk": 2026001, "game_date": "2026-08-02",
             "home_team": "HOM", "away_team": "AWY",
             "hp_umpire_id": None, "hp_umpire_name": None, "season": 2026},
        ], columns=MAP_COLS)
        fresh_2026 = pd.concat([
            existing_2026,
            pd.DataFrame([{"game_pk": 2026002, "game_date": "2026-08-03",
                           "home_team": "HOM", "away_team": "AWY",
                           "hp_umpire_id": 482631,
                           "hp_umpire_name": "Mike Estabrook",
                           "season": 2026}], columns=MAP_COLS),
        ], ignore_index=True)
        # the fresh payload assigns the crew for game 2026001
        fresh_2026.loc[fresh_2026["game_pk"] == 2026001,
                       ["hp_umpire_id", "hp_umpire_name"]] = (427344, "Bill Miller")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            existing_2026.to_csv(base / "umpire_map.csv", index=False)
            req = pd.DataFrame({
                "game_pk": [2026000, 2026001, 2026002],
                "game_date": ["2026-08-01", "2026-08-02", "2026-08-03"],
            })
            with patch("umpires.fetch_season_umpires",
                       side_effect=lambda s: fresh_2026) as mock:
                res = maintain_umpire_map(date(2026, 8, 27), base=base,
                                          required_games=req)
            # 2026 is present, so no season-cache fetch; the frame gap for
            # game 2026002 triggers exactly ONE gap-fill fetch of 2026.
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(mock.call_args[0][0], 2026)
            self.assertEqual(res["gap_filled_seasons"], ["2026"])
            self.assertEqual(res["rows_added"], 1)
            m = load_umpire_map(base)
            self.assertEqual(len(m), 4)
            self.assertEqual(set(m["game_pk"]),
                             {2025000, 2026000, 2026001, 2026002})
            # patched null row now resolves; untouched row unchanged
            patched = m[m["game_pk"] == 2026001].iloc[0]
            self.assertEqual(patched["hp_umpire_id"], 427344)
            kept = m[m["game_pk"] == 2026000].iloc[0]
            self.assertEqual(kept["hp_umpire_name"], "Bill Miller")

    def test_fetch_failure_keeps_existing_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            existing = self._season_df(2025)
            existing.to_csv(base / "umpire_map.csv", index=False)
            with patch("umpires.fetch_season_umpires",
                       side_effect=RuntimeError("network down")) as mock:
                res = maintain_umpire_map(date(2026, 8, 27), base=base)
            self.assertEqual(mock.call_count, 1)   # tried 2026 only
            self.assertIn("2026", res["errors"])
            self.assertEqual(res["rows_added"], 0)
            m = load_umpire_map(base)
            self.assertEqual(len(m), 5)            # map untouched
            self.assertEqual(set(m["season"]), {2025})

    def test_empty_fetch_leaves_season_unpresent(self):
        """A season returning no rows is NOT marked fetched — next run
        retries it (a real season always returns rows)."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch("umpires.fetch_season_umpires",
                       return_value=pd.DataFrame(columns=MAP_COLS)):
                res = maintain_umpire_map(date(2026, 8, 27), base=base)
            self.assertIn("2026", res["errors"])
            self.assertFalse((base / "umpire_map.csv").exists())


class TestFetchSeasonUmpires(unittest.TestCase):
    def test_fetch_returns_schema_on_http_error(self):
        """Fail-safe: an HTTP error yields an empty MAP_COLS frame, no raise."""
        with patch("umpires._get_with_retry") as mock:
            mock.return_value.status_code = 500
            mock.return_value.ok = False
            df = fetch_season_umpires(2026)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), MAP_COLS)

    def test_fetch_parses_live_shaped_payload(self):
        payload = _payload(_game(778563, officials=[
            _hp_official(427344, "Bill Miller")]))
        with patch("umpires._get_with_retry") as mock:
            mock.return_value.ok = True
            mock.return_value.status_code = 200
            mock.return_value.json = lambda: payload
            df = fetch_season_umpires(2026)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["hp_umpire_id"], 427344)
        self.assertEqual(df.iloc[0]["season"], 2026)


class TestRealFrameCoverage(unittest.TestCase):
    def test_all_frame_games_resolve_hp_umpire(self):
        """The committed maintained map covers 100% of the real frame with
        an HP umpire ID (the scoping capability, preserved)."""
        gl = pd.read_csv(ROOT / "data_delivery" / "game_level_features.csv")
        m = pd.read_csv(ROOT / "data_delivery" / "umpire_map.csv",
                        dtype={"game_pk": "int64"})
        frame_pks = set(gl["game_pk"].astype(int))
        sub = m[m["game_pk"].isin(frame_pks)]
        self.assertEqual(len(sub), len(frame_pks))       # 100% present
        self.assertEqual(sub["hp_umpire_id"].notna().mean(), 1.0)
        self.assertGreaterEqual(sub["hp_umpire_id"].nunique(), 90)
        self.assertEqual(len(sub), 4481)


class TestUmpireStats(unittest.TestCase):
    def test_stats_table_shape_and_placeholders(self):
        rows = [
            {"game_pk": 1, "game_date": "2026-08-01", "season": 2026,
             "hp_umpire_id": 427344, "hp_umpire_name": "Bill Miller"},
            {"game_pk": 2, "game_date": "2026-08-02", "season": 2026,
             "hp_umpire_id": 427344, "hp_umpire_name": "Bill Miller"},
            {"game_pk": 3, "game_date": "2026-08-03", "season": 2026,
             "hp_umpire_id": 482631, "hp_umpire_name": "Mike Estabrook"},
        ]
        m = pd.DataFrame(rows)
        games = pd.DataFrame({
            "game_pk": [1, 2, 3],
            "total_runs": [9.0, 7.0, 5.0],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out = build_umpire_stats(m, games, base=Path(tmp))
        self.assertEqual(len(out), 2)                     # 2 umpires
        bill = out[out["hp_umpire_id"] == 427344].iloc[0]
        self.assertEqual(bill["season"], 2026)
        self.assertEqual(bill["games_worked"], 2)
        self.assertAlmostEqual(bill["mean_runs_game"], 8.0)
        # called-pitch placeholders present and null until pitch data lands
        self.assertIsNone(bill["called_pitch_n"])
        self.assertIsNone(bill["called_strike_rate"])
        for col in ("hp_umpire_id", "hp_umpire_name", "season",
                    "games_worked", "mean_runs_game", "trailing_runs_game",
                    "called_pitch_n", "called_strike_rate"):
            self.assertIn(col, out.columns)


if __name__ == "__main__":
    unittest.main()
