"""Tests for the team-staleness remediation (2026-10-03).

The bug: the pool's as-of join is keyed on team, so a rating row kept serving
a player to his OLD club after a trade — up to POOL_LOOKBACK_DAYS in-season
(Kane: 42 days served to Chicago), and through opener-window carries after an
off-season move no rating row can disprove. Two independent guards:

1. TEAM-CHANGE GUARD — row-derived: a side may serve a player only while his
   own newest prior row still names that team. Needs no feed; applies to
   every game, backtest included.
2. ROSTER GATING — a captured roster snapshot (player → team, capture strictly
   before puck drop) re-keys membership BOTH ways: leavers leave the old side
   before they ever appear for the new one, joiners join immediately. Applied
   only where the capture is point-in-time legal; everything else fails closed
   to row-keyed membership.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import features as feat  # noqa: E402
import ingestion as ing  # noqa: E402
import injury_stints as ist  # noqa: E402
from test_injury_stints import make_ratings  # noqa: E402


def _strict(rows):
    """Production-shaped ratings: ``pool_date`` = source game date (strict)."""
    frame = make_ratings(rows)
    frame["pool_date"] = frame["game_date"]
    return frame


def _side(game_id, team, date="2026-10-15", start="2026-10-15T23:00:00Z"):
    row = {"game_id": game_id, "game_date": date, "season": 2026,
           "team": team}
    if start is not None:
        row["start_time_utc"] = start
    return row


def _roster(rows, capture="2026-10-14T12:00:00Z"):
    """rows: (player_id, team) observed at one capture time."""
    return pd.DataFrame([
        {"player_id": p, "team": t, "snapshot_at": capture, "complete": True}
        for p, t in rows
    ])


# ---------------------------------------------------------------------------
# Guard 1: the team-change guard — row-derived, no feed required
# ---------------------------------------------------------------------------
class TestTeamChangeGuard:
    def test_a_traded_player_stops_serving_his_old_club(self):
        """THE 45-day bug: CHI rows are fresh, but he now plays for NYR.

        The join keyed on team would serve his 2026-09-28 CHI row to CHI's
        2026-10-15 game (17 days old, well inside the window) even though
        his 2026-10-05 row names NYR. The guard drops him from CHI and leaves
        him intact on NYR.
        """
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.060, 9000.0, "2026-09-20"),
            ("mp-p", "CHI", "5on5", "C", 0.060, 9000.0, "2026-09-28"),
            ("mp-p", "NYR", "5on5", "C", 0.060, 9000.0, "2026-10-05"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-28"),
        ])
        games = pd.DataFrame([
            _side("G1", "CHI"), _side("G1", "NYR"),
        ])
        pool, audit = ist.team_game_rates(ratings, stints=None, games=games)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        nyr = pool[(pool["team"] == "NYR") & (pool["situation"] == "5on5")]
        assert chi["n_players"].iloc[0] == 1
        assert chi["rate"].iloc[0] == pytest.approx(0.020)  # mp-c alone
        assert nyr["n_players"].iloc[0] == 1
        assert nyr["rate"].iloc[0] == pytest.approx(0.060)
        assert audit["dropped_wrong_team"] == 1

    def test_without_a_newer_row_the_old_club_keeps_him(self):
        """The honest limit: an off-season move leaves no row to prove it.

        He has only CHI rows — no NYR game has been played yet — so row-
        derived knowledge cannot show the move and the guard abstains. That
        gap is roster gating's job (below), not a reason to guess here.
        """
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.060, 9000.0, "2026-09-20"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-28"),
        ])
        games = pd.DataFrame([_side("G1", "CHI")])
        pool, audit = ist.team_game_rates(ratings, stints=None, games=games)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        assert chi["n_players"].iloc[0] == 2
        assert audit["dropped_wrong_team"] == 0

    def test_same_day_two_team_rows_prove_no_ordering(self):
        """Rows naming two teams on ONE date cannot be ordered within the day.

        The guard must abstain — on both sides. Falling back to the older
        unambiguous date and asserting a team would claim exactly the
        knowledge the data refuses to give.
        """
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.060, 9000.0, "2026-09-20"),
            ("mp-p", "CHI", "5on5", "C", 0.060, 9000.0, "2026-09-28"),
            ("mp-p", "NYR", "5on5", "C", 0.060, 9000.0, "2026-09-28"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-28"),
        ])
        games = pd.DataFrame([_side("G1", "CHI"), _side("G1", "NYR")])
        pool, audit = ist.team_game_rates(ratings, stints=None, games=games)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        nyr = pool[(pool["team"] == "NYR") & (pool["situation"] == "5on5")]
        assert chi["n_players"].iloc[0] == 2   # kept on CHI: abstained
        assert nyr["n_players"].iloc[0] == 1   # kept on NYR: abstained
        assert audit["dropped_wrong_team"] == 0

    def test_a_single_team_player_and_the_boundary_carry_are_untouched(self):
        """Opener-window carrying a prior-season rating must survive intact —
        the guard is about WHICH team, never about freshness."""
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.060, 9000.0, "2026-05-01"),
        ])
        games = pd.DataFrame([_side("G1", "CHI")])
        pool, audit = ist.team_game_rates(ratings, stints=None, games=games)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        assert len(chi) == 1
        assert chi["rate"].iloc[0] == pytest.approx(0.060)
        assert audit["dropped_wrong_team"] == 0
        assert audit["season_boundary_carried_rows"] == 1


# ---------------------------------------------------------------------------
# Guard 2: roster gating — capture-stamped membership, applied PIT
# ---------------------------------------------------------------------------
class TestRosterGating:
    def test_a_captured_roster_moves_a_leaver_and_his_joiner(self):
        """Off-season move: only CHI rows exist, roster says NYR.

        Row-keyed membership keeps serving him to CHI (nothing disproves it).
        The capture — taken before puck drop — proves otherwise: he must
        leave CHI's pool AND appear in NYR's pool before his debut there.
        """
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.050, 9000.0, "2026-09-25"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-25"),
        ])
        rosters = _roster([("mp-p", "NYR"), ("mp-c", "CHI")])
        games = pd.DataFrame([_side("G1", "CHI"), _side("G1", "NYR")])
        pool, audit = ist.team_game_rates(
            ratings, stints=None, games=games, rosters=rosters)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        nyr = pool[(pool["team"] == "NYR") & (pool["situation"] == "5on5")]
        assert chi["n_players"].iloc[0] == 1
        assert chi["rate"].iloc[0] == pytest.approx(0.020)
        assert nyr["n_players"].iloc[0] == 1
        assert nyr["rate"].iloc[0] == pytest.approx(0.050)
        assert audit["roster_covered_games"] == 1
        assert audit["roster_dropped_players"] == 1
        assert audit["roster_added_players"] == 1
        assert audit["dropped_wrong_team"] == 0  # row proof alone can't move him

    def test_a_snapshot_captured_after_puck_drop_never_binds(self):
        """PIT contract: a post-game capture cannot alter the game's pool —
        the same rule the injury snapshots obey. Row-keyed membership stands."""
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.050, 9000.0, "2026-09-25"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-25"),
        ])
        rosters = _roster([("mp-p", "NYR"), ("mp-c", "CHI")],
                          capture="2026-10-16T12:00:00Z")
        games = pd.DataFrame([_side("G1", "CHI")])
        pool, audit = ist.team_game_rates(
            ratings, stints=None, games=games, rosters=rosters)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        assert chi["n_players"].iloc[0] == 2  # he stays: capture too late
        assert audit["roster_covered_games"] == 0
        assert audit["roster_dropped_players"] == 0

    def test_a_game_without_a_start_time_never_binds(self):
        """No exact puck drop → no PIT comparison → no roster application."""
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.050, 9000.0, "2026-09-25"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-25"),
        ])
        rosters = _roster([("mp-p", "NYR"), ("mp-c", "CHI")])
        games = pd.DataFrame([_side("G1", "CHI", start=None)])
        pool, audit = ist.team_game_rates(
            ratings, stints=None, games=games, rosters=rosters)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        assert chi["n_players"].iloc[0] == 2
        assert audit["roster_covered_games"] == 0

    def test_an_unfetched_side_keeps_its_row_keyed_players(self):
        """A partial capture proves only what it saw: a player MAPPED away
        leaves (positive knowledge), an ABSENT player on an unfetched side
        stays (absence there means nothing — his roster was never fetched)."""
        ratings = _strict([
            ("mp-t", "TOR", "5on5", "C", 0.040, 9000.0, "2026-09-25"),
            ("mp-x", "TOR", "5on5", "C", 0.060, 9000.0, "2026-09-25"),
        ])
        # Snapshot covers NYR and CHI — TOR was never fetched.
        rosters = _roster([("mp-x", "NYR"), ("mp-c", "CHI")])
        games = pd.DataFrame([_side("G1", "TOR"), _side("G1", "NYR")])
        pool, audit = ist.team_game_rates(
            ratings, stints=None, games=games, rosters=rosters)
        tor = pool[(pool["team"] == "TOR") & (pool["situation"] == "5on5")]
        nyr = pool[(pool["team"] == "NYR") & (pool["situation"] == "5on5")]
        assert tor["n_players"].iloc[0] == 1     # mp-t stays (TOR unseen)
        assert tor["rate"].iloc[0] == pytest.approx(0.040)
        assert nyr["n_players"].iloc[0] == 1     # mp-x moved (NYR seen)
        assert nyr["rate"].iloc[0] == pytest.approx(0.060)
        assert audit["roster_dropped_players"] == 1
        assert audit["roster_added_players"] == 1

    def test_the_guard_still_binds_when_no_roster_is_supplied(self):
        """Row-keyed membership with the guard is the complete fallback —
        a run whose egress is blocked loses the off-season half, not both."""
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.060, 9000.0, "2026-09-28"),
            ("mp-p", "NYR", "5on5", "C", 0.060, 9000.0, "2026-10-05"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-28"),
        ])
        games = pd.DataFrame([_side("G1", "CHI")])
        pool, audit = ist.team_game_rates(
            ratings, stints=None, games=games, rosters=None)
        chi = pool[(pool["team"] == "CHI") & (pool["situation"] == "5on5")]
        assert chi["n_players"].iloc[0] == 1
        assert audit["dropped_wrong_team"] == 1
        assert audit["roster_covered_games"] == 0


# ---------------------------------------------------------------------------
# The wiring: add_player_pool_features serves roster-gated pl_* columns
# ---------------------------------------------------------------------------
class TestPoolFeatureWiring:
    @staticmethod
    def _games():
        return pd.DataFrame([{
            "game_id": "G1", "season": 2026, "game_date": "2026-10-15",
            "start_time_utc": "2026-10-15T23:00:00Z",
            "home_team": "NYR", "away_team": "CHI",
        }])

    @staticmethod
    def _ratings():
        ratings = _strict([
            ("mp-p", "CHI", "5on5", "C", 0.050, 9000.0, "2026-09-25"),
            ("mp-c", "CHI", "5on5", "C", 0.020, 9000.0, "2026-09-25"),
        ])
        ratings["player_name"] = ["Departed Center", "Stayed Center"]
        return ratings

    def test_pl_columns_follow_the_roster_not_the_old_club_rows(self):
        """Row-keyed, CHI's 5on5-C pool would be mean(0.05, 0.02)=0.035 and
        NYR would read the prior. Roster-gated: CHI serves 0.02 alone, NYR
        serves the joiner's 0.05 — both columns move with the roster."""
        rosters = _roster([("mp-p", "NYR"), ("mp-c", "CHI")])
        with patch.object(feat.ingestion, "load_team_roster_snapshot",
                          return_value=rosters), \
             patch.object(feat.ingestion, "load_espn_injuries",
                          return_value=None):
            out = feat.add_player_pool_features(
                self._games(), player_ratings=self._ratings())
        assert out.loc[0, "pl_evo_c_away"] == pytest.approx(0.020)
        assert out.loc[0, "pl_evo_c_home"] == pytest.approx(0.050)

    def test_without_a_snapshot_the_old_behaviour_stands(self):
        """No capture (egress blocked, never fetched) → row-keyed pool with
        the guard — the run degrades, it does not fabricate."""
        with patch.object(feat.ingestion, "load_team_roster_snapshot",
                          return_value=None), \
             patch.object(feat.ingestion, "load_espn_injuries",
                          return_value=None):
            out = feat.add_player_pool_features(
                self._games(), player_ratings=self._ratings())
        assert out.loc[0, "pl_evo_c_away"] == pytest.approx(0.035)
        prior_home = feat._POSITION_PRIOR["EVO"]["C"]
        assert out.loc[0, "pl_evo_c_home"] == pytest.approx(prior_home)


# ---------------------------------------------------------------------------
# The loader: capture stamps, TTL gating, failure contracts
# ---------------------------------------------------------------------------
class TestRosterLoader:
    @staticmethod
    def _payload(player_id=8474141, team="CHI"):
        return {"forwards": [{"id": player_id}],
                "defensemen": [], "goalies": []}

    @pytest.fixture()
    def cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ing, "CACHE_DIR", tmp_path)
        return tmp_path

    def test_a_capture_is_fetched_once_per_ttl_and_never_restamped(
            self, cache_dir, monkeypatch):
        calls = []

        def fake_http(url, **kwargs):
            calls.append(url)
            return self._payload()

        monkeypatch.setattr(ing, "_http_json", fake_http)
        first = ing.load_team_roster_snapshot(["CHI"])
        second = ing.load_team_roster_snapshot(["CHI"])
        assert first is not None and second is not None
        assert len(calls) == 1  # TTL gate: the second call reuses the capture
        # Never re-stamped: the cached capture keeps its own instant (the
        # read-back path normalises to UTC timestamps).
        s1 = pd.to_datetime(first["snapshot_at"], errors="coerce", utc=True)
        s2 = pd.to_datetime(second["snapshot_at"], errors="coerce", utc=True)
        assert (s1 == s2).all()
        stamp = s2.iloc[0]
        assert (pd.Timestamp.now(tz="UTC") - stamp) < pd.Timedelta(hours=1)

    def test_a_failed_fetch_returns_none_and_suppresses_retries(
            self, cache_dir, monkeypatch):
        calls = []

        def boom(url, **kwargs):
            calls.append(url)
            raise RuntimeError("egress blocked")

        monkeypatch.setattr(ing, "_http_json", boom)
        assert ing.load_team_roster_snapshot(["CHI"]) is None
        first_wave = len(calls)
        assert first_wave >= 1
        # The failed attempt is stamped: the TTL must suppress the next call
        # instead of re-blocking every feature build with fresh retries.
        assert ing.load_team_roster_snapshot(["CHI"]) is None
        assert len(calls) == first_wave

    def test_a_partial_capture_forces_a_refetch_when_a_team_is_missing(
            self, cache_dir, monkeypatch):
        """Coverage honesty: a cache that never saw BOS must not let BOS
        players' absence read as 'not on that roster'."""
        state = {"ok": True}

        def fake_http(url, **kwargs):
            if "bos" in url and state["ok"]:
                raise RuntimeError("one team down")
            return self._payload()

        monkeypatch.setattr(ing, "_http_json", fake_http)
        partial = ing.load_team_roster_snapshot(["chi", "bos"])
        assert partial is not None
        assert set(partial["team"]) == {"chi"}          # BOS missing
        assert not bool(partial["complete"].all())
        # Same requested set: TTL-usable, no refetch needed for coverage.
        calls = {"n": 0}

        def counting(url, **kwargs):
            calls["n"] += 1
            return self._payload()

        monkeypatch.setattr(ing, "_http_json", counting)
        again = ing.load_team_roster_snapshot(["chi"])
        assert calls["n"] == 0 and again is not None
        # Asking for the team the cache never saw forces a fresh capture.
        state["ok"] = False
        complete = ing.load_team_roster_snapshot(["chi", "bos"])
        assert calls["n"] == 2
        assert set(complete["team"]) == {"chi", "bos"}
