"""NFL master_pipeline stale-cleanup predicates (``master_pipeline.py``) —
focused pure-function tests for the board-backed retention rule.

Covered:
- The canonical decided frame (``nfl_game_level_features.csv``, a dateless
  name) is exact-name protected — never pruned, no matter the board dates or
  retention window.
- A dated ``nfl_moneyline_v1_<d>.json`` / ``nfl_feature_v1_<d>.json`` record
  is KEPT when ``<d>`` still renders a board (appears as a distinct game_date
  in the moneyline games[]), even outside the retention window.
- It is PRUNED when ``<d>`` has no board (the reverse direction is allowed).
- Unrelated dated files (monitor/calibration-like) are unaffected by the
  board-backed rule — they stay on the plain retention window.
- ``board_dates_from_records`` unions game_dates across ALL moneyline records
  (a blocked run's record has no games[] and must not drop protection).
- Staged files always win (this run's regenerated artifacts are never stale).

No network, no git — pure classification + a temp-dir JSON read.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from master_pipeline import (  # noqa: E402
    _artifact_date,
    _board_backed_keep,
    _is_protected_name,
    board_dates_from_records,
    classify_stale,
    parse_args,
)

DD = "nfl-backend/data_delivery"
EMPTY: set[str] = set()


class TestCanonicalFrameProtected(unittest.TestCase):
    def test_canonical_frame_never_pruned(self):
        """The dateless canonical decided frame must survive cleanup no matter
        what — empty board dates and an empty retention window included."""
        rel = f"{DD}/nfl_game_level_features.csv"
        self.assertEqual(classify_stale(rel, EMPTY, EMPTY, EMPTY), "protected")
        self.assertTrue(_is_protected_name(rel))

    def test_staged_canonical_frame_still_wins(self):
        """'staged files win' ordering: a regenerated file is never stale even
        when its name is also protected (the ordering intact)."""
        rel = f"{DD}/nfl_game_level_features.csv"
        self.assertEqual(classify_stale(rel, {rel}, EMPTY, EMPTY), "staged")


class TestBoardBackedRecords(unittest.TestCase):
    def test_record_kept_when_date_still_renders_board(self):
        """nfl_moneyline_v1_<d> and nfl_feature_v1_<d> are kept when <d> is a
        distinct game_date in the moneyline games[] — even with an empty
        retention window (this is the MLB regression fix: a board date never
        loses the record it renders)."""
        board = {"20260909", "20260910"}
        for name in ("nfl_moneyline_v1_20260909.json", "nfl_feature_v1_20260909.json"):
            rel = f"{DD}/{name}"
            self.assertEqual(classify_stale(rel, EMPTY, board, EMPTY), "keep",
                             name)
            self.assertTrue(_board_backed_keep(rel, board))

    def test_record_pruned_when_date_has_no_board(self):
        """A record whose filename date has no board is stale (the reverse
        direction is explicitly allowed), unless the retention window keeps it."""
        rel = f"{DD}/nfl_moneyline_v1_20260830.json"
        board = {"20260909"}          # board lives on 09-09, not 08-30
        retention = {"20260901"}      # 08-30 outside the window
        self.assertEqual(classify_stale(rel, EMPTY, board, retention), "stale")
        self.assertFalse(_board_backed_keep(rel, board))

    def test_record_kept_by_retention_when_no_board(self):
        """Without a board date the plain retention window still saves a
        recent record — the board-backed rule is ADDITIVE, not replacing."""
        rel = f"{DD}/nfl_moneyline_v1_20260830.json"
        self.assertEqual(classify_stale(rel, EMPTY, {"20260909"},
                                        {"20260830"}), "keep")


class TestUnrelatedDatedFiles(unittest.TestCase):
    def test_monitor_like_files_unaffected_by_board_rule(self):
        """A dated monitor/calibration-like file is never saved by the
        board-backed rule — it stays on the plain retention window."""
        rel = f"{DD}/nfl_model_monitor_20260909.json"
        board = {"20260909"}                     # board exists, but rule is
        self.assertFalse(_board_backed_keep(rel, board))  # scoped to records
        self.assertEqual(classify_stale(rel, EMPTY, board, EMPTY), "stale")
        # ... and the retention window still applies to it normally
        self.assertEqual(classify_stale(rel, EMPTY, board, {"20260909"}), "keep")

    def test_other_dated_artifact_within_retention(self):
        rel = f"{DD}/nfl_pbp_chunks_20260830.parquet"
        self.assertEqual(classify_stale(rel, EMPTY, {"20260909"},
                                        {"20260830"}), "keep")
        self.assertEqual(classify_stale(rel, EMPTY, {"20260909"},
                                        {"20260901"}), "stale")


class TestBoardDateSet(unittest.TestCase):
    def test_union_across_all_records(self):
        """S = distinct game_dates across EVERY moneyline record's games[] —
        a blocked (no games[]) record must not drop protection for the still-
        live boards its predecessors wrote."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "nfl_moneyline_v1_20260830.json").write_text(json.dumps({
                "verdict": {"adopt": True},
                "games": [{"game_date": "2026-09-09"}, {"game_date": "2026-09-10"}],
            }), encoding="utf-8")
            # a later blocked run: no games[] -> contributes nothing
            (d / "nfl_moneyline_v1_20260901.json").write_text(json.dumps({
                "verdict": {"adopt": False},
                "predictions": {"status": "blocked (not adopted)"},
            }), encoding="utf-8")
            self.assertEqual(board_dates_from_records(d),
                             {"20260909", "20260910"})

    def test_empty_dir_yields_empty_board(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(board_dates_from_records(Path(td)), set())

    def test_malformed_record_never_crashes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "nfl_moneyline_v1_20260830.json").write_text(
                "{ not json", encoding="utf-8")
            self.assertEqual(board_dates_from_records(d), set())


class TestArtifactDate(unittest.TestCase):
    def test_date_extraction(self):
        self.assertEqual(_artifact_date(f"{DD}/nfl_moneyline_v1_20260830.json"),
                         "20260830")
        self.assertEqual(_artifact_date(f"{DD}/nfl_feature_v1_20260909.json"),
                         "20260909")
        self.assertIsNone(_artifact_date(f"{DD}/nfl_game_level_features.csv"))
        self.assertIsNone(_artifact_date(f"{DD}/models/bundle.joblib"))


class TestSeasonWindow(unittest.TestCase):
    """parse_args -> .window: the data/feature season window from
    --start-season / --end-season or NFL_START_SEASON / NFL_END_SEASON.
    Default (no overrides) is None -> each module keeps its full range."""

    def _window(self, argv, env=None):
        import os
        saved = os.environ.copy()
        env = env or {}
        for k, v in env.items():
            os.environ[k] = v
        try:
            return parse_args(argv).window
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_default_is_none(self):
        """No --start/--end-season -> window is None (unchanged default)."""
        self.assertIsNone(self._window([]))
        self.assertIsNone(self._window(["--slate-season", "2026"]))

    def test_both_cli(self):
        self.assertEqual(self._window(["--start-season", "2021",
                                       "--end-season", "2023"]),
                         [2021, 2022, 2023])

    def test_only_start_closes_to_last(self):
        self.assertEqual(self._window(["--start-season", "2022"]),
                         [2022, 2023, 2024, 2025])

    def test_only_end_closes_to_first(self):
        self.assertEqual(self._window(["--end-season", "2022"]),
                         [2019, 2020, 2021, 2022])

    def test_env_vars(self):
        self.assertEqual(self._window([], {"NFL_START_SEASON": "2019",
                                           "NFL_END_SEASON": "2020"}),
                         [2019, 2020])

    def test_env_takes_back_seat_to_cli(self):
        self.assertEqual(self._window(["--start-season", "2022",
                                       "--end-season", "2023"],
                                      {"NFL_START_SEASON": "2019",
                                       "NFL_END_SEASON": "2020"}),
                         [2022, 2023])

    def test_invalid_order_raises(self):
        with self.assertRaises(ValueError):
            parse_args(["--start-season", "2024", "--end-season", "2022"])


if __name__ == "__main__":
    unittest.main()
