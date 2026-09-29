"""Tests for the league player-availability audit's verdicts.

The audit must fail on the three ways an availability layer can be silently
wrong (pool-eligible names missed by the bridge, look-ahead into pre-capture
games, ambiguous refusals) and stay silent on the shapes that are merely the
league being the league (roster moves, exclusion-refused statuses, a young
archive) unless --strict asks for the noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import audit_player_availability as apa  # noqa: E402
import features as feat  # noqa: E402
import injury_stints as ist  # noqa: E402
from test_injury_stints import make_ratings  # noqa: E402


def _ratings() -> pd.DataFrame:
    """Two teams' skaters over three days ending 2026-04-15 (pre-capture)."""
    ratings = make_ratings([
        ("mp-a1", "ANA", "5on5", "C", 0.060, 9000.0, "2026-04-15"),
        ("mp-a2", "ANA", "5on5", "C", 0.020, 9000.0, "2026-04-15"),
        ("mp-a3", "ANA", "5on5", "L", 0.050, 9000.0, "2026-04-15"),
        ("mp-b1", "STL", "5on5", "C", 0.040, 9000.0, "2026-04-15"),
        ("mp-b2", "STL", "5on5", "C", 0.010, 9000.0, "2026-04-15"),
    ])
    ratings["player_name"] = ["Ana Center One", "Ana Center Two",
                              "Ana Left Wing", "Stl Center One",
                              "Stl Center Two"]
    ratings["pool_date"] = ratings["game_date"]
    return ratings


def _reports(rows: list[dict]) -> pd.DataFrame:
    base = {"team": None, "report_date": None, "return_date": None,
            "injury_type": None, "detail": None, "snapshot_marker": False}
    return pd.DataFrame([{**base, **row} for row in rows])


CAPTURE = "2026-09-28T21:50:57Z"


def test_clean_archive_passes_with_soft_findings_only():
    hard, soft, details = apa.audit(_ratings(), _reports([
        {"player_id": "e-1", "player_name": "Ana Left Wing",
         "status": "Injured Reserve", "snapshot_at": CAPTURE},
        {"player_id": "e-2", "player_name": "Ghost Goalie",
         "status": "Out", "snapshot_at": CAPTURE},
        {"player_id": None, "player_name": None, "status": None,
         "snapshot_at": CAPTURE, "snapshot_marker": True},
    ]))
    assert hard == []
    assert any("1 captured snapshot" in s for s in soft)
    assert details["bridge"]["out_ir_matched"] == 1
    assert details["bridge"]["out_ir_eligible"] == 2  # all Out/IR rows counted


def test_unmatched_names_are_outside_the_rating_space_and_vacuous():
    # A name the bridge leaves unmatched is BY CONSTRUCTION absent from the
    # skater rating space (the bridge resolves on the same normalisation the
    # space is built from), so its exclusion can never bind: not a miss.
    # One matched row keeps the bind gate satisfied (an archive that shared
    # NO vocabulary with the ratings would rightly hard-fail).
    hard, soft, details = apa.audit(_ratings(), _reports([
        {"player_id": "e-1", "player_name": "Ana Left Wing",
         "status": "Injured Reserve", "snapshot_at": CAPTURE},
        {"player_id": "e-2", "player_name": "Ghost Goalie",
         "status": "Out", "snapshot_at": CAPTURE},
        {"player_id": "e-3", "player_name": "Never Dressed Prospect",
         "status": "Injured Reserve", "snapshot_at": CAPTURE},
    ]))
    assert hard == []
    assert details["bridge"]["out_ir_matched"] == 1
    assert details["bridge"]["out_ir_eligible"] == 3
    assert details["bridge"]["unmatched"] == 2


def test_pool_predicate_binds_only_after_the_exact_capture_time():
    # The predicate-level retroactivity proof: capture lands 04-16T00:30Z —
    # AFTER the last source rating (04-15) but BEFORE the probe's 02:00Z
    # puck drop on 04-16. The probe pool forms (source < target, within the
    # lookback) and loses exactly one player: binding starts the instant
    # the capture exists, and nothing earlier was touched.
    ratings = _ratings()
    reports = _reports([
        {"player_id": "e-1", "player_name": "Ana Center One",
         "status": "Out", "snapshot_at": "2026-04-16T00:30:00Z"},
    ])
    bridged, _ = ist.map_reports_to_rating_ids(reports, ratings)
    stints, saudit = ist.build_stint_intervals(bridged)
    stints.attrs["snapshot_times"] = [pd.Timestamp("2026-04-16T00:30:00Z")]
    stints.attrs["snapshot_based"] = bool(saudit.get("snapshot_based"))
    stints.attrs["window_end"] = pd.Timestamp("2026-04-16T00:30:00Z")
    hist = pd.DataFrame([
        {"game_date": pd.Timestamp("2026-04-16"), "team": t,
         "start_time_utc": "2026-04-16T02:00:00Z"}
        for t in sorted(ratings["team"].unique())])
    pool, paudit = ist.team_game_rates(ratings, stints=stints, games=hist)
    assert len(pool) > 0
    assert paudit["dropped_unavailable"] == 1


def test_the_audit_lookahead_branch_fires_on_report_date_semantics():
    # Snapshot semantics make a pre-capture removal doubly unreachable: a
    # stint STARTS at its capture AND the predicate requires a capture before
    # puck drop. The audit's hard branch guards the REGRESSION to report-date
    # semantics — verify it by injecting exactly that shape: a capture just
    # before the probe (so a snapshot exists pre-decision) with the stint
    # backdated to the provider's report date (before that capture).
    ratings = _ratings()
    reports = _reports([
        {"player_id": "e-1", "player_name": "Ana Center One",
         "status": "Out", "snapshot_at": "2026-04-16T00:30:00Z",
         "report_date": "2026-04-15"},
    ])
    real_build = ist.build_stint_intervals

    def backdated(reports_frame, **kwargs):
        stints, saudit = real_build(reports_frame, **kwargs)
        if len(stints):
            stints = stints.copy()
            # The report-date semantics leak: the interval opens at the
            # provider's report date, not at the capture instant.
            stints[ist.STINT_START] = pd.Timestamp("2026-04-15T12:00:00Z")
        return stints, saudit

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(apa.ist, "build_stint_intervals", backdated)
    try:
        hard, _, details = apa.audit(ratings, reports)
    finally:
        monkeypatched.undo()
    assert any("PRE-capture" in h for h in hard)


def test_ambiguous_bridge_refusal_is_a_hard_failure():
    ratings = _ratings()
    # Two rating ids share one normalised name: the bridge must refuse.
    dup = ratings[ratings.player_name == "Ana Center One"].copy()
    dup["player_id"] = "mp-a1-shadow"
    ratings = pd.concat([ratings, dup], ignore_index=True)
    hard, _, details = apa.audit(ratings, _reports([
        {"player_id": "e-1", "player_name": "Ana Center One",
         "status": "Out", "snapshot_at": CAPTURE},
    ]))
    assert details["bridge"]["ambiguous"] == 1
    assert any("ambiguous" in h for h in hard)


def test_suspension_and_dtd_are_reported_never_failed():
    reports = _reports([
        {"player_id": "e-1", "player_name": "Ana Center One",
         "status": "Suspension", "snapshot_at": CAPTURE},
        {"player_id": "e-2", "player_name": "Ana Center Two",
         "status": "Day-To-Day", "snapshot_at": CAPTURE},
    ])
    hard, soft, _ = apa.audit(_ratings(), reports)
    assert hard == []
    assert any("exclusion-refused" in s for s in soft)
    hard_strict, _, _ = apa.audit(_ratings(), reports, strict=True)
    assert any("exclusion-refused" in h for h in hard_strict)


def test_roster_move_labels_are_soft_unless_strict():
    # The archived player's rating id sits on a DIFFERENT team than the
    # archive label (an off-season move): informational, never hard.
    reports = _reports([
        {"player_id": "e-1", "player_name": "Ana Center One",
         "team": "Toronto Maple Leafs", "status": "Out",
         "snapshot_at": CAPTURE},
    ])
    hard, soft, details = apa.audit(_ratings(), reports)
    assert hard == []
    assert details["roster_move_labels"] == 1
    hard_strict, _, _ = apa.audit(_ratings(), reports, strict=True)
    assert any("roster-move" in h for h in hard_strict)


def test_main_reports_the_league_audit_and_exits_zero(monkeypatch, capsys):
    ratings = _ratings()
    reports = _reports([
        {"player_id": "e-1", "player_name": "Ana Left Wing",
         "status": "Injured Reserve", "snapshot_at": CAPTURE},
    ])
    monkeypatch.setattr(feat, "_load_player_ratings", lambda: ratings)
    monkeypatch.setattr(apa.feat, "_load_player_ratings", lambda: ratings)
    monkeypatch.setattr(apa.ing, "_injury_history", lambda: reports)
    rc = apa.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no hard findings" in out
    assert "[bridge]" in out and "[shadow]" in out


def test_main_fails_closed_without_any_history(monkeypatch, capsys):
    monkeypatch.setattr(apa.ing, "_injury_history", lambda: None)
    assert apa.main([]) == 1
    assert "no captured injury history" in capsys.readouterr().out
