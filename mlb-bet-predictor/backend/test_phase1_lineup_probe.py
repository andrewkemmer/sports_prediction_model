"""Tests for the Phase 1 lineup-coverage probe's pure parser.

Covers parse_batting_orders + classify_order/classify_game on canned
StatsAPI live-feed shapes: complete 9-per-side, missing boxscore, missing
battingOrder key, empty arrays, short/over-length orders, non-numeric
entries, and ID resolution against the same feed's players dict. No
network — fixtures are inline dicts.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_lineup_coverage import (  # noqa: E402
    classify_game,
    classify_order,
    parse_batting_orders,
)

_NINE = [650402, 701678, 805808, 693307, 680664, 656716, 679529, 595879, 703601]


def _feed(boxscore: dict | None = None, state: str = "Final") -> dict:
    """Canned live-feed shape: gameData.status + liveData.boxscore."""
    teams = {"home": {"battingOrder": list(_NINE),
                      "players": {f"{i}:P": {"person": {"fullName": f"Player{i}"}}
                                  for i in _NINE}},
             "away": {"battingOrder": list(_NINE),
                      "players": {f"{i}:B": {"person": {"fullName": f"Away{i}"}}
                                  for i in _NINE}}}
    box = {"teams": teams}
    if boxscore is not None:
        box = boxscore
    return {
        "gameData": {"status": {"abstractGameState": state},
                     "teams": {"home": {"abbreviation": "HOM"},
                               "away": {"abbreviation": "AWY"}}},
        "liveData": {"boxscore": box},
    }


class TestParseBattingOrders(unittest.TestCase):
    def test_complete_9_per_side(self):
        p = parse_batting_orders(_feed())
        self.assertEqual(len(p["home"]), 9)
        self.assertEqual(len(p["away"]), 9)
        self.assertEqual(p["home"][0], 650402)

    def test_players_dict_resolves_ids(self):
        p = parse_batting_orders(_feed())
        for i in _NINE:
            self.assertIn(i, p["home_players"])
            self.assertEqual(p["home_players"][i], f"Player{i}")

    def test_missing_boxscore_is_nan_safe(self):
        p = parse_batting_orders({"gameData": {}})
        self.assertEqual(p["home"], [])
        self.assertEqual(p["away"], [])
        self.assertEqual(p["home_players"], {})
        # classify_game names the failure instead of fabricating an order
        row = classify_game({"gameData": {"status": {"abstractGameState": "Final"}},
                             "liveData": {}}, 745266)
        self.assertEqual(row["failure"], "null_boxscore")

    def test_missing_batting_order_key(self):
        p = parse_batting_orders(_feed(boxscore={"teams": {}}))
        self.assertEqual(p["home"], [])

    def test_non_numeric_entries_dropped(self):
        box = {"teams": {"home": {"battingOrder": [650402, "abc", None, 701678]},
                         "away": {"battingOrder": []}}}
        p = parse_batting_orders(_feed(boxscore=box))
        self.assertEqual(p["home"], [650402, 701678])

    def test_classify_order_labels(self):
        self.assertEqual(classify_order(list(_NINE)), ("complete_9", 9))
        self.assertEqual(classify_order([]), ("empty_array", 0))
        self.assertEqual(classify_order([1, 2]), ("short_n", 2))
        self.assertEqual(classify_order(list(range(11))), ("over_9", 11))

    def test_short_and_over_orders_surface(self):
        box = {"teams": {"home": {"battingOrder": [650402, 701678]},
                         "away": {"battingOrder": list(range(11))}}}
        row = classify_game(_feed(boxscore=box), 1)
        self.assertEqual(row["home"], ("short_n", 2))
        self.assertEqual(row["away"], ("over_9", 11))

    def test_id_resolution_counts(self):
        row = classify_game(_feed(), 745266)
        self.assertEqual(row["order_ids_resolved"], {"home": 9, "away": 9})

    def test_postponed_state_preserved(self):
        row = classify_game(_feed(state="Postponed"), 1)
        self.assertEqual(row["state"], "Postponed")
        self.assertEqual(row["home"], ("complete_9", 9))


if __name__ == "__main__":
    unittest.main()
