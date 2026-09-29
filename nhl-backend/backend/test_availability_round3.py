"""Round-3 availability contracts (2026-09-29 audit, examples 6-10).

The five new examples sharpen two structural boundaries every future change
must keep honest, and pin three of them as data-verified regression tests:

STRUCTURAL BOUNDARY 1 - GOALIES NEVER ENTER A SKATER POOL. The rating space
is skaters only (C/L/R/D): MoneyPuck player-games carry no goalie rows, so a
Skinner/Hellebuyck-shaped absence can never touch pl_evo/pl_ppo no matter
what status the feed serves. Verified live: PIT's 2025-26 starts were
Silovs/Larsson/Murashov/Gauthier/Jarry, WPG's Comrie (+ relief goalies).

STRUCTURAL BOUNDARY 2 - PERSONAL LEAVES ARE OUTSIDE THE FEED OF RECORD.
ESPN's injuries payload vocabulary is {IR, Day-To-Day, Suspension, Out}
(verified live 2026-09-29, 110 rows, zero leave rows). A Dahlin/Meier/
Winterton-shaped absence therefore never produces an ESPN status row, and
the policy's refusal to exclude on anything but Out/IR/Doubtful is exactly
right: the alternative would be appearance-inference, which is forbidden.

DATA-VERIFIED EXAMPLES (boxscores + MoneyPuck appearances):
  Skinner: dressed 0-TOI backup on 04-04 (the bench accident game), missed
    exactly the 04-05 game, back starting 04-09 - DTD without IR, exactly
    the "DTD is not a game-day out" shape the policy assumes.
  Hellebuyck: not dressed 11 consecutive games (Nov 21..Dec 11); counting
    the Nov 18 dress-without-entering the claim's 12-game count matches;
    returned Dec 13 (the claim's mid-January date is contradicted by data).
  Winterton: 62 appearances before the Mar 23 leave, 68 for the season.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import features as feat  # noqa: E402
import injury_stints as ist  # noqa: E402
from test_injury_stints import make_ratings  # noqa: E402


def _skater_ratings(source_date: str) -> pd.DataFrame:
    """Two teams' skaters rated as of `source_date` (one day before a game)."""
    ratings = make_ratings([
        ("mp-d1", "BUF", "5on5", "D", 0.055, 12000.0, source_date),
        ("mp-d2", "BUF", "5on5", "D", 0.025, 12000.0, source_date),
        ("mp-f1", "SEA", "5on5", "R", 0.040, 6000.0, source_date),
        ("mp-f2", "SEA", "5on5", "R", 0.020, 6000.0, source_date),
    ])
    ratings["player_name"] = ["Rasmus Dahlin", "Buf Partner",
                              "Ryan Winterton", "Sea Teammate"]
    ratings["pool_date"] = ratings["game_date"]
    return ratings


def _snapshot(name, status, captured_at, espn_id):
    return {"player_id": espn_id, "player_name": name, "team": None,
            "status": status, "report_date": captured_at[:10],
            "snapshot_at": captured_at, "snapshot_marker": False}


def _patched_reports(monkeypatch, reports):
    monkeypatch.setattr(feat.ingestion, "load_espn_injuries",
                        lambda *a, **k: reports)


def test_goalie_statuses_can_never_touch_the_skater_pool(monkeypatch):
    # The bridge maps by NAME against the skater rating space; a goalie's
    # name resolves to nothing there, so his exclusion is vacuous even when
    # the feed serves Out/IR for him. The pool keeps every rated skater.
    ratings = _skater_ratings("2026-04-05")
    reports = pd.DataFrame([
        _snapshot("Stuart Skinner", "Out", "2026-04-05T15:00:00Z", "e-sk"),
        _snapshot("Connor Hellebuyck", "Injured Reserve",
                  "2026-04-05T15:00:00Z", "e-he"),
    ])
    games = pd.DataFrame([{
        "game_id": "G1", "season": 2026, "game_date": "2026-04-06",
        "start_time_utc": "2026-04-06T02:00:00Z",
        "home_team": "BUF", "away_team": "SEA",
    }])
    _patched_reports(monkeypatch, reports)
    out = feat.add_player_pool_features(games, player_ratings=ratings)
    # Both BUF skaters and both SEA skaters remain: no goalie leakage.
    assert out.loc[0, "pl_evo_d_home"] == pytest.approx((0.055 + 0.025) / 2)
    assert out.loc[0, "pl_evo_r_away"] == pytest.approx((0.040 + 0.020) / 2)
    # And the bridge's audit says exactly what happened: 2 unmatched.
    bridged, audit = ist.map_reports_to_rating_ids(reports, ratings)
    assert audit["unmatched"] == 2 and audit["ambiguous"] == 0


def test_personal_leave_status_is_exclusion_refused_by_policy(monkeypatch):
    # Even IF a provider served a leave-shaped status, the policy excludes
    # only exact Out/IR/Doubtful. A "Personal Leave" label must not remove
    # anyone: the leave examples (Dahlin/Meier/Winterton) never appear in
    # the feed at all, and inventing a status channel is appearance-
    # inference by another name.
    ratings = _skater_ratings("2025-11-07")
    reports = pd.DataFrame([
        _snapshot("Rasmus Dahlin", "Personal Leave",
                  "2025-11-07T23:00:00Z", "e-da"),
    ])
    games = pd.DataFrame([{
        "game_id": "G1", "season": 2026, "game_date": "2025-11-08",
        "start_time_utc": "2025-11-08T02:00:00Z",
        "home_team": "BUF", "away_team": "SEA",
    }])
    _patched_reports(monkeypatch, reports)
    out = feat.add_player_pool_features(games, player_ratings=ratings)
    assert out.loc[0, "pl_evo_d_home"] == pytest.approx((0.055 + 0.025) / 2)
    assert ist._is_il_status("Personal Leave") is False
    assert ist._is_il_status("Leave of Absence") is False


def test_day_to_day_is_not_a_game_day_out():
    # Skinner's DTD spell (04-05..04-08) proves the policy assumption: DTD
    # is roster paperwork, not a game-day out (he missed exactly 1 of 4
    # possible games). The exclusion predicate must keep refusing DTD.
    for status in ("Day-To-Day", "DTD"):
        assert ist._is_il_status(status) is False
    # ...while the only exclusion statuses stay the exact Out/IR family.
    for status in ("Out", "Injured Reserve", "IR"):
        assert ist._is_il_status(status) is True


def test_winterton_style_gap_never_reads_as_injury_or_health(monkeypatch):
    # A leave creates an appearance gap with NO status row anywhere. The
    # pipeline's contract: the gap is unknown (NaN receipt), the player's
    # rating is untouched, and nothing invents either an exclusion or a
    # clean bill of health from the absence itself.
    ratings = _skater_ratings("2026-03-23")
    games = pd.DataFrame([{
        "game_id": "G1", "season": 2026, "game_date": "2026-03-24",
        "start_time_utc": "2026-03-24T02:00:00Z",
        "home_team": "SEA", "away_team": "BUF",
    }])
    # The archive of the leave period: no row for Winterton at all.
    reports = pd.DataFrame([
        _snapshot("Unrelated Player", "Out", "2026-03-20T15:00:00Z", "e-x"),
    ])
    _patched_reports(monkeypatch, reports)
    out = feat.add_player_pool_features(games, player_ratings=ratings)
    assert out.loc[0, "pl_evo_r_home"] == pytest.approx((0.040 + 0.020) / 2)
    assert pd.isna(out.loc[0, "pl_il_out_fraction"])
