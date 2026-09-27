"""Tests for the NHL player pool and the timestamped binary injury policy.

Production injury removals come only from captured Out/IR status snapshots.
The old appearance-derived interval tests below are explicitly diagnostic;
nonparticipation is not used as injury evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import injury_stints as ist  # noqa: E402
import ingestion as ing  # noqa: E402
import features as feat  # noqa: E402


def make_reports(rows):
    """rows: (player_id, status, report_date, return_date|None)."""
    return pd.DataFrame([
        {"player_id": p, "status": s, "report_date": d, "return_date": r}
        for p, s, d, r in rows
    ])


def make_ratings(rows):
    """rows: (player_id, team, situation, position, rate, ice, game_date)."""
    return pd.DataFrame([
        {"player_id": p, "player_name": p, "team": t, "situation": sit,
         "position": pos, "shrunk_rate_per60": r, "prior_ice_seconds": ice,
         "game_date": pd.Timestamp(d)}
        for p, t, sit, pos, r, ice, d in rows
    ])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class TestClassification:
    @pytest.mark.parametrize("status,is_il", [
        ("Out", True),
        ("Injured Reserve", True),
        ("IR", True),
        ("out", True),
        ("Day-To-Day", False),
        ("Suspension", False),
        ("SUSP", False),
        ("DTD", False),
        ("Unknown Provider Status", False),
        ("Healthy", False),
        (None, False),
        ("", False),
    ])
    def test_only_an_il_placement_opens_an_interval(self, status, is_il):
        # Only exact normalized Out / Injured Reserve / IR designations set
        # injury. Every other status—including DTD, suspension, and unknown
        # provider labels—is available under the approved production policy.
        assert ist._is_il_status(status) is is_il


# ---------------------------------------------------------------------------
# Interval construction
# ---------------------------------------------------------------------------
class TestBuildStintIntervals:
    def test_a_single_il_report_leaves_an_open_stint(self):
        rep = make_reports([("1", "Out", "2026-01-05", None)])
        st, audit = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_START] == pd.Timestamp("2026-01-05")
        assert pd.isna(st.iloc[0][ist.STINT_END])
        assert audit["open_stints"] == 1

    def test_a_later_clean_report_closes_the_interval(self):
        rep = make_reports([("1", "Out", "2026-01-05", None),
                            ("1", "Healthy", "2026-02-10", None)])
        st, audit = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_END] == pd.Timestamp("2026-02-10")
        assert audit["closed_on_clean_report"] == 1

    def test_projected_return_date_does_not_close_a_legacy_interval(self):
        rep = make_reports([("1", "Injured Reserve", "2026-01-05",
                             "2026-01-20")])
        st, audit = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert pd.isna(st.iloc[0][ist.STINT_END])
        assert audit["closed_on_return_date"] == 0

    def test_day_to_day_never_opens_an_interval(self):
        rep = make_reports([("1", "Day-To-Day", "2026-01-05", None)])
        st, audit = ist.build_stint_intervals(rep)
        assert len(st) == 0
        assert audit["non_il_reports"] == 1

    def test_a_stint_in_the_past_closes_at_the_first_later_report(self):
        rep = make_reports([("1", "Out", "2026-01-05", None),
                            ("1", "Out", "2026-01-09", None),
                            ("1", "Healthy", "2026-01-20", None)])
        st, _ = ist.build_stint_intervals(rep)
        # A repeat IL report is an escalation, not a new stint.
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_START] == pd.Timestamp("2026-01-05")
        assert st.iloc[0][ist.STINT_END] == pd.Timestamp("2026-01-20")

    def test_players_are_tracked_independently(self):
        rep = make_reports([("1", "Out", "2026-01-05", None),
                            ("2", "Healthy", "2026-01-06", None)])
        st, _ = ist.build_stint_intervals(rep)
        assert set(st[ist.OUT_PLAYER]) == {"1"}
        assert len(st) == 1

    def test_tz_aware_report_dates_are_accepted(self):
        rep = pd.DataFrame([{"player_id": "1", "status": "Out",
                             "report_date": "2026-01-05T20:08Z",
                             "return_date": None}])
        st, _ = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_START] == pd.Timestamp("2026-01-05")

    def test_empty_input(self):
        st, audit = ist.build_stint_intervals(pd.DataFrame())
        assert len(st) == 0
        assert audit["report_rows"] == 0

    def test_return_date_never_closes_a_snapshot_interval(self):
        rep = pd.DataFrame([{
            "player_id": "1", "status": "Out",
            "report_date": "2026-01-01", "return_date": "2026-01-02",
            "snapshot_at": "2026-01-05T20:00:00Z", "snapshot_marker": False,
        }])
        st, _ = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_START] == pd.Timestamp("2026-01-05 20:00:00")
        assert pd.isna(st.iloc[0][ist.STINT_END])


class TestSnapshotIntervals:
    @staticmethod
    def _snapshot(at, rows=(), marker=False):
        if marker:
            return [{"player_id": None, "status": None,
                     "report_date": None, "return_date": None,
                     "snapshot_at": at, "snapshot_marker": True}]
        return [{"player_id": player, "status": status,
                 "report_date": report_date, "return_date": return_date,
                 "snapshot_at": at, "snapshot_marker": False}
                for player, status, report_date, return_date in rows]

    def test_intervals_use_capture_times_not_provider_dates_or_return_dates(self):
        rep = pd.DataFrame(
            self._snapshot("2026-01-05T20:00:00Z",
                           [("1", "Out", "2025-12-01", "2026-01-06")])
            + self._snapshot("2026-01-08T20:00:00Z",
                             [("1", "Day-To-Day", "2026-01-05", None)]))
        st, audit = ist.build_stint_intervals(rep)
        assert audit["snapshot_based"] is True
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_START] == pd.Timestamp("2026-01-05 20:00:00")
        assert st.iloc[0][ist.STINT_END] == pd.Timestamp("2026-01-08 20:00:00")

    @pytest.mark.parametrize("status", ["Day-To-Day", "Suspension", "DTD",
                                          "Unknown Provider Status", None])
    def test_every_nonapproved_snapshot_status_is_available(self, status):
        rep = pd.DataFrame(
            self._snapshot("2026-01-05T18:00:00Z",
                           [("1", "Out", "2026-01-05", None)])
            + self._snapshot("2026-01-06T18:00:00Z",
                             [("1", status, "2026-01-06", None)]))
        st, _ = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_END] == pd.Timestamp("2026-01-06 18:00:00")

    def test_a_successful_empty_snapshot_closes_the_open_out_interval(self):
        rep = pd.DataFrame(
            self._snapshot("2026-01-05T20:00:00Z",
                           [("1", "IR", "2026-01-05", None)])
            + self._snapshot("2026-01-08T20:00:00Z", marker=True))
        st, audit = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_END] == pd.Timestamp("2026-01-08 20:00:00")
        assert audit["closed_on_snapshot"] == 1

    def test_report_received_after_puck_drop_does_not_retroactively_exclude(self):
        st, _ = ist.build_stint_intervals(pd.DataFrame(
            self._snapshot("2026-01-05T20:00:00Z",
                           [("1", "Out", "2026-01-04", None)])))
        assert ist.is_unavailable(st, "1", "2026-01-05T19:59:59Z",
                                  strict_start=True) is False
        assert ist.is_unavailable(st, "1", "2026-01-05T20:00:00Z",
                                  strict_start=True) is False
        assert ist.is_unavailable(st, "1", "2026-01-05T20:00:01Z",
                                  strict_start=True) is True

    def test_clear_snapshot_at_puck_drop_does_not_clear_prior_known_out(self):
        st, _ = ist.build_stint_intervals(pd.DataFrame(
            self._snapshot("2026-01-05T18:00:00Z",
                           [("1", "Out", "2026-01-05", None)])
            + self._snapshot("2026-01-05T20:00:00Z",
                             [("1", "Active", "2026-01-05", None)])))
        assert ist.is_unavailable(st, "1", "2026-01-05T19:59:59Z",
                                  strict_start=True) is True
        assert ist.is_unavailable(st, "1", "2026-01-05T20:00:00Z",
                                  strict_start=True) is True
        assert ist.is_unavailable(st, "1", "2026-01-05T20:00:01Z",
                                  strict_start=True) is False

    def test_stale_last_snapshot_does_not_claim_a_known_clear_or_out(self):
        rep = pd.DataFrame(self._snapshot(
            "2026-01-05T18:00:00Z", [("1", "Out", "2026-01-05", None)]))
        st, _ = ist.build_stint_intervals(rep)
        assert ist.is_unavailable(
            st, "1", "2026-03-01T18:00:00Z", strict_start=True) is False

    def test_missing_exact_puck_drop_time_cannot_apply_snapshot_interval(self):
        ratings = make_ratings([
            ("1", "BOS", "5on5", "C", 0.060, 9000.0, "2026-04-15"),
            ("2", "BOS", "5on5", "C", 0.020, 9000.0, "2026-04-15"),
        ])
        games = pd.DataFrame([{"game_date": "2026-10-15", "team": "BOS",
                               "season": 2026}])
        st, _ = ist.build_stint_intervals(pd.DataFrame(self._snapshot(
            "2026-10-14T18:00:00Z", [("1", "Out", "2026-10-14", None)])))
        pool, audit = ist.team_game_rates(ratings, stints=st, games=games)
        assert audit["dropped_unavailable"] == 0
        assert pool["n_players"].iloc[0] == 2
        assert audit["missing_decision_time"] == 2


# ---------------------------------------------------------------------------
# The exclusion predicate — MLB's NOT EXISTS, verbatim
# ---------------------------------------------------------------------------
class TestIsUnavailable:
    @pytest.fixture
    def stints(self):
        return pd.DataFrame([
            {"player_id": "1", "stint_start": "2026-01-05",
             "stint_end": "2026-01-20"},
        ])

    @pytest.mark.parametrize("date,expected", [
        ("2026-01-04", False),   # before the stint
        ("2026-01-05", True),    # on the placement date
        ("2026-01-12", True),    # inside
        ("2026-01-20", False),   # activated ON the day -> available for it
        ("2026-01-21", False),   # after
    ])
    def test_interval_edges(self, stints, date, expected):
        assert ist.is_unavailable(stints, "1", date) is expected

    def test_open_stint_covers_every_later_date(self):
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": pd.NaT}])
        assert ist.is_unavailable(stints, "1", "2027-06-01") is True

    def test_another_player_is_unaffected(self, stints):
        assert ist.is_unavailable(stints, "2", "2026-01-12") is False

    def test_no_table_means_nobody_is_unavailable(self):
        assert ist.is_unavailable(pd.DataFrame(), "1", "2026-01-12") is False

    def test_mask_is_per_row_not_per_player(self):
        # The answer is a function of the date, not a property of the player.
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": "2026-01-20"}])
        ratings = make_ratings([
            ("1", "T", ist.SITUATION_EVO, "C", 0.02, 20000.0, "2026-01-10"),
            ("1", "T", ist.SITUATION_EVO, "C", 0.02, 20000.0, "2026-02-01"),
        ])
        mask = ist.build_unavailable_mask(ratings, stints)
        assert list(mask) == [True, False]


# ---------------------------------------------------------------------------
# Reconciliation against observed appearances
# ---------------------------------------------------------------------------
class TestReconcile:
    def test_an_appearance_ends_the_stint_early(self):
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": pd.NaT}])
        apps = pd.DataFrame([{"player_id": "1",
                              "game_date": "2026-01-15"}])
        out, audit = ist.reconcile_against_appearances(stints, apps)
        assert out.iloc[0][ist.STINT_END] == pd.Timestamp("2026-01-15")
        assert audit["stints_shortened"] == 1

    def test_a_same_date_appearance_is_the_announcement_not_a_return(self):
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": pd.NaT}])
        apps = pd.DataFrame([{"player_id": "1", "game_date": "2026-01-05"}])
        out, audit = ist.reconcile_against_appearances(stints, apps)
        assert pd.isna(out.iloc[0][ist.STINT_END])
        assert audit["stints_shortened"] == 0

    def test_games_before_the_return_stay_flagged(self):
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": pd.NaT}])
        apps = pd.DataFrame([{"player_id": "1", "game_date": "2026-01-15"}])
        out, _ = ist.reconcile_against_appearances(stints, apps)
        assert ist.is_unavailable(out, "1", "2026-01-06") is True
        assert ist.is_unavailable(out, "1", "2026-01-16") is False

    def test_an_already_shorter_stint_is_not_lengthened(self):
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": "2026-01-10"}])
        apps = pd.DataFrame([{"player_id": "1", "game_date": "2026-01-20"}])
        out, audit = ist.reconcile_against_appearances(stints, apps)
        assert out.iloc[0][ist.STINT_END] == pd.Timestamp("2026-01-10")
        assert audit["stints_shortened"] == 0

    def test_no_appearances_changes_nothing(self):
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": pd.NaT}])
        out, audit = ist.reconcile_against_appearances(stints, pd.DataFrame())
        assert pd.isna(out.iloc[0][ist.STINT_END])
        assert audit["appearances"] == 0


# ---------------------------------------------------------------------------
# The pool -> team aggregate
# ---------------------------------------------------------------------------
class TestTeamGameRates:
    @pytest.fixture
    def two_teams(self):
        return make_ratings([
            ("1", "BOS", ist.SITUATION_EVO, "C", 0.030, 30000.0, "2026-01-10"),
            ("2", "BOS", ist.SITUATION_EVO, "C", 0.010, 30000.0, "2026-01-10"),
            ("3", "TOR", ist.SITUATION_EVO, "C", 0.020, 30000.0, "2026-01-10"),
        ])

    def test_baseline_pool_averages_the_healthy_players(self, two_teams):
        out, audit = ist.team_game_rates(two_teams, stints=None)
        bos = out[out.team == "BOS"].iloc[0]
        assert bos["rate"] == pytest.approx(0.020)
        assert bos["n_players"] == 2
        assert audit["il_filter_active"] is False

    def test_an_open_stint_removes_the_player_from_the_pool(self, two_teams):
        """THE load-bearing test: the exclusion must actually bite.

        The walk-forward has no PIT injury archive for the seasons it scores,
        so this synthetic interval is the only proof the filter is not a
        silent no-op.
        """
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-01",
                                "stint_end": pd.NaT}])
        out, audit = ist.team_game_rates(two_teams, stints=stints)
        bos = out[out.team == "BOS"].iloc[0]
        # Player 1 (0.030) is gone; the pool is now just player 2 (0.010).
        assert bos["rate"] == pytest.approx(0.010)
        assert bos["n_players"] == 1
        assert audit["dropped_unavailable"] == 1
        # And the OTHER team is untouched.
        tor = out[out.team == "TOR"].iloc[0]
        assert tor["n_players"] == 1

    def test_the_rating_itself_is_never_modified(self, two_teams):
        """MLB filters the POOL; it does not discount the rating."""
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-01",
                                "stint_end": pd.NaT}])
        out, _ = ist.team_game_rates(two_teams, stints=stints)
        assert set(out.columns) >= {"rate", "n_players", "evidence"}
        assert "shrunk_rate_per60" not in out.columns

    def test_a_closed_stint_does_not_remove_anyone(self, two_teams):
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2025-12-01",
                                "stint_end": "2025-12-20"}])
        out, audit = ist.team_game_rates(two_teams, stints=stints)
        assert audit["dropped_unavailable"] == 0
        assert out[out.team == "BOS"].iloc[0]["n_players"] == 2

    def test_the_minimum_games_gate_drops_a_thin_player(self):
        ratings = make_ratings([
            ("1", "BOS", ist.SITUATION_EVO, "C", 0.500, 30000.0, "2026-01-10"),
            ("2", "BOS", ist.SITUATION_EVO, "C", 0.010, 200.0, "2026-01-10"),
        ])
        out, audit = ist.team_game_rates(ratings, stints=None)
        assert audit["dropped_below_min_ice"] == 1
        # Without the gate, player 2's wild rate from 200 seconds of ice would
        # drag the mean; the gate is what makes the mean trustworthy.
        assert out.iloc[0]["n_players"] == 1
        assert out.iloc[0]["rate"] == pytest.approx(0.500)

    def test_the_gate_can_be_relaxed_for_a_sparse_situation(self):
        ratings = make_ratings([
            ("1", "BOS", ist.SITUATION_PPO, "D", 0.010, 500.0, "2026-01-10"),
        ])
        _, strict = ist.team_game_rates(ratings, stints=None)
        assert strict["dropped_below_min_ice"] == 1
        out, loose = ist.team_game_rates(ratings, stints=None,
                                         min_prior_ice_seconds=100.0)
        assert loose["dropped_below_min_ice"] == 0
        assert len(out) == 1

    def test_a_player_with_no_rate_is_dropped(self):
        ratings = make_ratings([
            ("1", "BOS", ist.SITUATION_EVO, "C", np.nan, 30000.0, "2026-01-10"),
            ("2", "BOS", ist.SITUATION_EVO, "C", 0.010, 30000.0, "2026-01-10"),
        ])
        out, audit = ist.team_game_rates(ratings, stints=None)
        assert audit["dropped_no_rate"] == 1
        assert out.iloc[0]["n_players"] == 1

    def test_the_most_recent_rating_per_player_wins(self):
        ratings = make_ratings([
            ("1", "BOS", ist.SITUATION_EVO, "C", 0.010, 30000.0, "2025-04-17"),
            ("1", "BOS", ist.SITUATION_EVO, "C", 0.050, 30000.0, "2026-04-16"),
        ])
        out, audit = ist.team_game_rates(ratings, stints=None)
        assert audit["pool_rows"] == 1
        assert out.iloc[0]["rate"] == pytest.approx(0.050)

    def test_an_empty_pool_returns_an_empty_frame_not_a_crash(self):
        ratings = make_ratings([
            ("1", "BOS", ist.SITUATION_EVO, "C", 0.010, 10.0, "2026-01-10"),
        ])
        out, audit = ist.team_game_rates(ratings, stints=None)
        assert len(out) == 0
        assert audit["pool_rows"] == 0

    def test_no_stints_means_the_fallback_pool(self, two_teams):
        """MLB's participant-pool fallback, kept reachable and explicit."""
        with_stints, a1 = ist.team_game_rates(two_teams, stints=None)
        without, a2 = ist.team_game_rates(
            two_teams, stints=pd.DataFrame())
        assert a1["dropped_unavailable"] == 0 and a2["dropped_unavailable"] == 0
        pd.testing.assert_frame_equal(with_stints, without)


# ---------------------------------------------------------------------------
# The per-game grid -- the fix that lets the exclusion bind at all
# ---------------------------------------------------------------------------
class TestPerGameGrid:
    """The exclusion is a function of the GAME date, not the rating's date.

    Season-grain ratings carry one date per season, so evaluating ``NOT
    EXISTS`` at the rating date asks "was he hurt on the last day of last
    season?" for every game in the season. These tests pin the per-game
    behaviour, which is what MLB's ``p.game_date`` predicate actually does.
    """

    @staticmethod
    def _world():
        # Two centres with different rates, both rated 2026-04-15 (the end of
        # the 2025-26 season, which serves the 2026-27 season -- labelled 2026
        # because seasons here are named by the year they BEGIN), plus a
        # deliberately awful third centre who is about to be injured.
        ratings = make_ratings([
            ("1", "BOS", "5on5", "C", 0.060, 9000.0, "2026-04-15"),
            ("2", "BOS", "5on5", "C", 0.020, 9000.0, "2026-04-15"),
            ("3", "BOS", "5on5", "C", 0.900, 9000.0, "2026-04-15"),
        ])
        games = pd.DataFrame([
            {"game_date": pd.Timestamp("2026-10-10"), "team": "BOS",
             "season": 2026},
            {"game_date": pd.Timestamp("2026-12-05"), "team": "BOS",
             "season": 2026},
        ])
        return ratings, games

    def _stint(self, start, end=None):
        return pd.DataFrame([{"player_id": "3", "stint_start": start,
                              "stint_end": end}])

    def _covering(self, end="2026-12-10"):
        """A stint that covers the 2026-12-05 game and not the 2026-10-10 one.

        A window falling BETWEEN the two games is not a test of anything: it
        excludes nobody, which is the same answer as no stints at all.
        """
        return self._stint("2026-11-25", end)

    #: All three centres available.
    FULL = pytest.approx((0.060 + 0.020 + 0.900) / 3)
    #: The $0.90 player removed, the two healthy centres left.
    DEPLETED = pytest.approx((0.060 + 0.020) / 2)

    def _rates(self, ratings, games, stints):
        pool, _ = ist.team_game_rates(ratings, stints=stints, games=games)
        return pool.set_index("game_date")["rate"]

    def test_a_stint_covering_the_later_game_drops_him_from_that_game_only(
            self):
        ratings, games = self._world()
        r = self._rates(ratings, games, self._covering())
        assert r[pd.Timestamp("2026-10-10")] == self.FULL
        assert r[pd.Timestamp("2026-12-05")] == self.DEPLETED

    def test_an_open_stint_behaves_the_same_as_one_that_closes(self):
        """An interval with a NULL end is still an interval."""
        ratings, games = self._world()
        open_ = self._rates(ratings, games, self._stint("2026-11-25"))
        closed = self._rates(ratings, games, self._covering())
        pd.testing.assert_series_equal(open_, closed)

    def test_the_played_game_keeps_the_player_the_out_game_drops_him(self):
        """The distinction the rating-date evaluation cannot express."""
        ratings, games = self._world()
        pool, audit = ist.team_game_rates(ratings, stints=self._covering(),
                                          games=games)
        assert audit["dropped_unavailable"] == 1
        n = pool.set_index(["game_date", "position"])["n_players"]
        assert n[(pd.Timestamp("2026-10-10"), "C")] == 3
        assert n[(pd.Timestamp("2026-12-05"), "C")] == 2

    def test_a_stint_that_covers_neither_game_is_a_no_op(self):
        ratings, games = self._world()
        base = self._rates(ratings, games, self._stint("2027-01-01"))
        out = self._rates(ratings, games, self._stint("2026-10-01", "2026-10-05"))
        pd.testing.assert_series_equal(base, out)

    def test_a_strictly_rating_date_evaluation_would_drop_both_games(self):
        """Why the grid is mandatory: the old semantics, pinned as a mutant.

        At the rating's own date (2026-04-15) a November stint matches nothing
        and a March stint matches everything. Either way the answer is
        constant across the season, so a player cannot be out for one game and
        in for the next -- which is the entire job.
        """
        ratings, games = self._world()
        per_game = self._rates(ratings, games, self._covering())
        assert per_game.nunique() == 2  # the correct answer varies
        flat, _ = ist.team_game_rates(ratings, stints=self._covering())
        assert flat["rate"].nunique() == 1  # the old answer could not vary

    def test_no_grid_falls_back_to_the_rating_date(self):
        """Documented degradation, not a crash: it cannot bind, by design."""
        ratings, games = self._world()
        stints = self._covering()
        grid, _ = ist.team_game_rates(ratings, stints=stints, games=games)
        flat, _ = ist.team_game_rates(ratings, stints=stints)
        # Flat: one row dated 2026-04-15, and the November stint misses it.
        assert len(flat) == 1
        assert flat["game_date"].iloc[0] == pd.Timestamp("2026-04-15")
        assert flat["n_players"].iloc[0] == 3
        assert grid["n_players"].sum() == 5

    def test_the_minimum_games_gate_runs_before_the_exclusion(self):
        """Order matters: a thin player is not 'injured', he was never in.

        Player 4 is BOTH under the ice-time minimum AND on the IL. He must be
        counted by the minimum-games counter, because a player who never
        qualified was never removed -- counting him as an injury would hide a
        broken evidence gate behind a working injury filter.
        """
        ratings, games = self._world()
        ratings = pd.concat([ratings, make_ratings([
            ("4", "BOS", "5on5", "C", 0.900, 10.0, "2026-04-15")])],
            ignore_index=True)
        stints = pd.DataFrame([{"player_id": "4", "stint_start": "2026-11-25",
                                "stint_end": "2026-12-10"}])
        pool, audit = ist.team_game_rates(ratings, stints=stints, games=games)
        assert audit["dropped_below_min_ice"] == 1
        assert audit["dropped_unavailable"] == 0
        assert pool["n_players"].max() == 3

    def test_a_rating_does_not_serve_the_season_it_measured(self):
        """Point-in-time discipline: a 2026-04 rating cannot serve 2025.

        The rating closes the 2025-26 season, so the season it can serve is
        2026 (named by the year it begins). A 2025 game predates it.
        """
        ratings, _ = self._world()
        past = pd.DataFrame([{"game_date": pd.Timestamp("2025-11-01"),
                              "team": "BOS", "season": 2025}])
        pool, audit = ist.team_game_rates(ratings, stints=None, games=past)
        assert len(pool) == 0 and audit["pool_rows"] == 0

    def test_the_serve_offset_is_zero(self):
        """The bug this whole class exists to prevent, pinned as a number.

        Seasons are named by the year they BEGIN, so an October 2026 game is
        season 2026, and the rating that serves it is the one stamped
        2026-04-15 -- its own year. Reading the offset as the obvious ``+ 1``
        still produces a live pool for the wrong season, so nothing raises.
        """
        assert ist.POOL_SERVE_OFFSET_YEARS == 0
        ratings, games = self._world()
        pool, _ = ist.team_game_rates(ratings, stints=None, games=games)
        assert len(pool) == 2
        wrong = games.assign(season=games["season"] + 1)
        off_by_one, _ = ist.team_game_rates(ratings, stints=None, games=wrong)
        assert len(off_by_one) == 0

    def test_the_season_label_is_derived_when_the_grid_omits_it(self):
        """Oct-Apr seasons, named by the year they begin."""
        ratings, games = self._world()
        derived = games.drop(columns=["season"])
        a, _ = ist.team_game_rates(ratings, stints=None, games=games)
        b, _ = ist.team_game_rates(ratings, stints=None, games=derived)
        pd.testing.assert_frame_equal(a, b)

    def test_the_optional_game_lookback_expires_a_rating_mid_season(self):
        ratings, games = self._world()
        wide, _ = ist.team_game_rates(ratings, games=games)
        tight, _ = ist.team_game_rates(ratings, games=games,
                                        game_lookback_days=60)
        assert len(wide) == 2 and len(tight) == 0

    def test_a_team_with_no_grid_row_is_absent_not_zero(self):
        """"Unknown" must not read as "no offence"; the caller falls back."""
        ratings, games = self._world()
        pool, _ = ist.team_game_rates(ratings, games=games)
        assert set(pool["team"]) == {"BOS"}
        assert "NYR" not in set(pool["team"])


# ---------------------------------------------------------------------------
# The fail-closed PIT gate
# ---------------------------------------------------------------------------
class TestPitGate:
    """The gate requires id binding, fresh capture history, and snapshot provenance."""

    @staticmethod
    def _good():
        ratings = make_ratings([
            ("1", "BOS", "5on5", "C", 0.060, 9000.0, "2026-01-10"),
            ("2", "BOS", "5on5", "C", 0.020, 9000.0, "2026-01-10"),
        ])
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-05",
                                "stint_end": "2026-01-08"}])
        stints.attrs["snapshot_based"] = True
        stints.attrs["snapshot_history"] = True
        stints.attrs["snapshot_times"] = [pd.Timestamp("2026-01-05"),
                                           pd.Timestamp("2026-01-09")]
        return ratings, stints

    def _assert(self, **over):
        kw = dict(decided_max_date="2026-01-10", window_end="2026-01-09",
                  snapshot_based=True)
        kw.update(over)
        ratings, stints = self._good()
        return ist.assert_pit(ratings, stints, **kw)

    def test_a_sound_filter_passes_and_reports_every_condition(self):
        v = self._assert()
        assert set(v) == set(ist.PIT_CONDITIONS)
        assert all(d["ok"] for d in v.values())

    def test_bind_failure_raises_and_names_the_bridge(self):
        ratings, stints = self._good()
        stints = stints.assign(player_id="999")
        with pytest.raises(ist.PitViolation, match=r"\[bind\]"):
            ist.assert_pit(ratings, stints, decided_max_date="2026-01-10",
                           window_end="2026-01-09", snapshot_based=True)

    def test_a_disjoint_id_space_fails_even_with_everything_else_satisfied(self):
        """The live failure: the gate is not bypassed by good intentions."""
        ratings, _ = self._good()
        stints = pd.DataFrame([{"player_id": "999", "stint_start": "2026-01-05",
                                "stint_end": "2026-01-08"}])
        stints.attrs["snapshot_based"] = True
        stints.attrs["snapshot_times"] = [pd.Timestamp("2026-01-05"),
                                           pd.Timestamp("2026-01-09")]
        with pytest.raises(ist.PitViolation):
            ist.assert_pit(ratings, stints, decided_max_date="2026-01-10",
                           window_end="2026-01-09", snapshot_based=True)

    def test_an_unknown_window_raises_rather_than_passing(self):
        """"No sidecar" must not read as "fresh"."""
        with pytest.raises(ist.PitViolation, match=r"\[fresh\]"):
            self._assert(window_end=None)

    def test_a_stale_archive_raises(self):
        with pytest.raises(ist.PitViolation, match=r"\[fresh\]"):
            self._assert(window_end="2025-10-01")

    def test_a_projected_return_must_never_be_the_window(self):
        """Why the sidecar rule exists, pinned as a mutant.

        A months-dead archive, fetched through 2025-10-01, is reported fresh by
        the staleness bar when the window handed in is the club's PROJECTED
        return instead of the fetch date — the lag goes negative, so the bar
        can never fire. The gate cannot detect this on its own; the test can.
        """
        ratings, stints = self._good()
        dead_fetch = "2025-10-01"
        projected = pd.to_datetime(stints[ist.STINT_END]).max()

        # Truth: the archive really is 101 days stale. The gate must say so.
        with pytest.raises(ist.PitViolation, match=r"\[fresh\]"):
            ist.assert_pit(ratings, stints, decided_max_date="2026-01-10",
                           window_end=dead_fetch, snapshot_based=True)

        # The trap: the same dead archive, judged against a projected return,
        # reports a negative lag and sails through.
        fooled = ist.staleness_report(stints, "2026-01-10", projected)
        assert fooled["lag_days"] == 2 and fooled["stale"] is False

    def test_legacy_report_date_intervals_raise(self):
        with pytest.raises(ist.PitViolation, match=r"\[snapshot_based\]"):
            self._assert(snapshot_based=False)

    def test_non_snapshot_intervals_raise_even_when_ids_bind(self):
        ratings, stints = self._good()
        stints.attrs["snapshot_times"] = [pd.Timestamp("2026-01-05"),
                                           pd.Timestamp("2026-01-09")]
        with pytest.raises(ist.PitViolation, match=r"\[snapshot_based\]"):
            ist.assert_pit(ratings, stints, decided_max_date="2026-01-10",
                           window_end="2026-01-09", snapshot_based=False)

    def test_conditions_fail_in_order_so_the_first_is_the_real_problem(self):
        """All conditions broken: binding is the first failure."""
        ratings, stints = self._good()
        stints = stints.assign(player_id="999")
        with pytest.raises(ist.PitViolation) as e:
            ist.assert_pit(ratings, stints, decided_max_date="2026-01-10",
                           window_end=None, snapshot_based=False)
        assert "[bind]" in str(e.value)

    def test_the_violation_is_not_an_assertion_error(self):
        """A data failure must be catchable without swallowing logic bugs."""
        assert not issubclass(ist.PitViolation, AssertionError)
        assert issubclass(ist.PitViolation, RuntimeError)

    def test_the_gate_does_not_mutate_its_inputs(self):
        ratings, stints = self._good()
        before_r, before_s = ratings.copy(), stints.copy()
        self._assert()
        pd.testing.assert_frame_equal(ratings, before_r)
        pd.testing.assert_frame_equal(stints, before_s)


# ---------------------------------------------------------------------------
# Appearance -> PIT intervals
# ---------------------------------------------------------------------------
class TestSeasonServing:
    """A rating must serve the WHOLE season it is stamped for.

    The games frame labels a season by the year it BEGINS, so a game on
    2025-01-15 is season 2024 and the rating dated 2024-04-17 is the one that
    serves it. Read the calendar year instead and that game looks for a rating
    stamped 2025-04-17 -- three months in its own future -- which the
    served-recency filter correctly rejects, leaving the side with no pool.

    The failure has no exception and no null: the caller falls back to the
    position prior, the columns freeze, and the injury flag has an empty pool
    to remove anyone from. So the assertion is about COVERAGE of a whole
    season, not about one value.
    """

    TEAM = "BOS"

    def _ratings(self, served_date="2024-04-17", players=("P1", "P2", "P3")):
        return pd.DataFrame([
            {"player_id": p, "player_name": p, "team": self.TEAM,
             "situation": "5on5", "position": "C", "shrunk_rate_per60": 0.05,
             "prior_ice_seconds": 9000.0,
             "game_date": pd.Timestamp(served_date)}
            for p in players])

    def _season(self, dates):
        return pd.DataFrame([
            {"game_date": pd.Timestamp(d), "team": self.TEAM, "season": 2024}
            for d in dates])

    def test_a_january_game_is_served_by_the_rating_behind_it(self):
        games = self._season(["2024-10-01", "2025-01-15", "2025-04-10"])
        pool, _ = ist.team_game_rates(self._ratings(), games=games)
        assert len(pool) == 3, (
            "the second half of the season has no pool: the season column was "
            "not the one the expansion reads")
        assert set(pd.to_datetime(pool.game_date)) == {
            pd.Timestamp(d) for d in games.game_date}

    def test_the_calendar_year_is_not_a_season(self):
        """A frame WITHOUT a season column cannot invent one from the date.

        This is the trap the production grid fell into: it renamed the season to
        ``_season``, so the expansion never saw it and derived one from the
        calendar year. So the fallback has to be a season rule, not a year rule.
        """
        games = self._season(["2024-10-01", "2025-01-15", "2025-04-10"])
        pool, _ = ist.team_game_rates(self._ratings(), games=games)
        nov = pool[pool.game_date == pd.Timestamp("2024-10-01")]
        apr = pool[pool.game_date == pd.Timestamp("2025-04-10")]
        assert len(nov) and len(apr)
        # Same season, same players, same rate -- a 6-month gap must not
        # change the answer.
        assert float(nov.rate.iloc[0]) == float(apr.rate.iloc[0])

    def test_a_rating_stamped_after_the_game_never_serves_it(self):
        """Recency still bites, on the rating's own clock."""
        games = self._season(["2024-01-15"])
        pool, _ = ist.team_game_rates(self._ratings(), games=games)
        assert len(pool) == 0, (
            "a rating dated after the game was allowed into that game's pool")


class TestAppearanceAvailabilityDiagnostics:
    """Legacy roster-availability diagnostics only; never injury evidence.

    Full coverage, and no interval may reach outside what was knowable.

    Two claims, one test each per way they could silently break.

    PIT: a player who misses a game is still in the pool FOR THAT GAME,
    because nobody knew at puck drop. The far edge is the same rule pointed
    the other way -- a return is not available to the game it happened in.

    COVERAGE: with an appearance record behind every game in the window, the
    flag has a value everywhere, and it must not INVENT one where the evidence
    is missing. The last group is the one that catches a truncated fetch
    masquerading as a league-wide injury epidemic.
    """

    # BOS plays on 5 dates. A 12-man SIDE, because the truncation floor is a
    # real number and a two-man fixture would test the floor instead of the
    # rule: every side in a toy roster reads as partial and nothing is ever
    # evaluated. P00 dresses in 0, 1, 3, 4 and misses game 2 ONLY.
    GAMES = ["2026-10-01", "2026-10-03", "2026-10-06", "2026-10-09", "2026-10-12"]
    TEAM = "BOS"
    ROSTER = tuple(f"P{i:02d}" for i in range(12))
    OUT_P = "P00"

    def _ratings(self, players=None):
        return pd.DataFrame([
            {"player_id": p, "player_name": p, "team": self.TEAM,
             "situation": "5on5", "position": "C", "shrunk_rate_per60": 0.05,
             "prior_ice_seconds": 9000.0,
             "game_date": pd.Timestamp("2026-04-15")}
            for p in (self.ROSTER if players is None else players)])

    def _games(self, served=2026):
        return pd.DataFrame([{"game_date": pd.Timestamp(d), "team": self.TEAM,
                              "season": served} for d in self.GAMES])

    def _dressed(self, rows):
        out = []
        for pid, idxs in rows:
            for i in idxs:
                out.append({"game_date": pd.Timestamp(self.GAMES[i]),
                            "team": self.TEAM, "player_id": pid})
        return pd.DataFrame(out)

    def _everyone_dresses(self, idxs):
        return self._dressed([(p, idxs) for p in self.ROSTER])

    def _one_miss(self):
        """P00 misses game 2 only; every other skater plays everything."""
        st, _ = ist.appearances_to_stints(
            self._with_one_out([0, 1, 3, 4]), self._ratings(), self._games())
        return st

    def _everyone_dressing(self, idxs, roster=None):
        return self._dressed([(p, idxs) for p in (roster or self.ROSTER)])

    def _with_one_out(self, out_idxs):
        """A full side every game, with OUT_P dressing only on ``out_idxs``."""
        rows = [(p, range(5)) for p in self.ROSTER if p != self.OUT_P]
        rows.append((self.OUT_P, out_idxs))
        return self._dressed(rows)

    # ------------------------------------------------------------------ PIT
    def test_a_missed_game_is_still_played(self):
        """The core PIT claim, as an assertion.

        P00 missed game 2, so he MUST still be in the pool for game 2: his
        absence was unknowable before it. Opening the interval on the missed
        date instead would be a look-ahead, and would let a retroactive
        injury remove a player from a game he was available for.
        """
        st = self._one_miss()
        assert ist.is_unavailable(st, self.OUT_P, pd.Timestamp(self.GAMES[2])) is False, (
            "P00 was removed from the game he missed -- a look-ahead")
        for i in (0, 1):
            assert ist.is_unavailable(
                st, self.OUT_P, pd.Timestamp(self.GAMES[i])) is False

    def test_the_interval_opens_on_the_next_game_not_the_missed_one(self):
        st = self._one_miss()
        s = st[st[ist.OUT_PLAYER] == self.OUT_P].iloc[0]
        assert pd.Timestamp(s[ist.STINT_START]) == pd.Timestamp(self.GAMES[3]), (
            "the first game the absence was public for is the one after it")

    def test_the_interval_does_not_close_on_the_return_game(self):
        """The far edge is the rule that keeps the OOF run honest.

        P00 dressed in game 3. A feature computed BEFORE game 3 cannot know
        that, so the flag must still be set for game 3 -- and it may close on
        game 4, whose boxscore is what would have revealed the return.

        Closing on game 3 instead would reconcile perfectly against the
        appearance record and would leak: for every game in the window the
        flag would know whether the man who played in it played.
        """
        st = self._one_miss()
        assert ist.is_unavailable(st, self.OUT_P, pd.Timestamp(self.GAMES[3])) is True, (
            "the return was used to score the very game it was observed in")
        assert ist.is_unavailable(st, self.OUT_P, pd.Timestamp(self.GAMES[4])) is False, (
            "P00 is not released until a LATER game says he is back")
        s = st[st[ist.OUT_PLAYER] == self.OUT_P].iloc[0]
        assert pd.notna(s[ist.STINT_END]), "the far edge is an observation"
        assert pd.Timestamp(s[ist.STINT_END]) == pd.Timestamp(self.GAMES[4])

    def test_a_player_who_never_missed_is_never_excluded(self):
        st = self._one_miss()
        for d in self.GAMES:
            assert ist.is_unavailable(st, "P01", pd.Timestamp(d)) is False

    def test_a_multi_game_run_is_one_interval_not_several(self):
        st, _ = ist.appearances_to_stints(
            self._with_one_out([0]), self._ratings(), self._games())
        assert len(st) == 1
        # misses 1..4 -> excluded from game 2 onward; no return observed
        assert pd.isna(st[ist.STINT_END].iloc[0]), (
            "an unobserved return is not an end date")
        assert pd.Timestamp(st[ist.STINT_START].iloc[0]) == pd.Timestamp(self.GAMES[2])
        assert ist.is_unavailable(st, self.OUT_P, pd.Timestamp(self.GAMES[1])) is False
        assert ist.is_unavailable(st, self.OUT_P, pd.Timestamp(self.GAMES[2])) is True

    def test_a_player_out_for_every_game_yields_no_future_exclusion(self):
        """He missed game 4, the last one, so no later game exists to exclude."""
        st, _ = ist.appearances_to_stints(
            self._with_one_out([0, 1, 2, 3]), self._ratings(), self._games())
        for d in self.GAMES:
            assert ist.is_unavailable(st, self.OUT_P, pd.Timestamp(d)) is False, (
                f"P00 wrongly excluded from {d}")

    def test_the_first_game_of_a_window_is_never_excluded(self):
        """There is no prior game to read, so the state is 'no evidence'."""
        st, _ = ist.appearances_to_stints(
            self._with_one_out([1, 2, 3, 4]), self._ratings(), self._games())
        assert ist.is_unavailable(st, self.OUT_P, pd.Timestamp(self.GAMES[0])) is False

    # -------------------------------------------------------------- coverage
    def test_every_side_is_evaluated_not_just_the_absent_ones(self):
        app = self._everyone_dressing(range(5))
        _, a = ist.appearances_to_stints(app, self._ratings(), self._games())
        assert a["sides"] == 5
        assert a["sides_observed"] == 5
        assert a["sides_with_absence"] == 0
        assert a["stints"] == 0

    def test_a_full_dressed_roster_produces_no_stints(self):
        app = self._everyone_dressing(range(5))
        st, _ = ist.appearances_to_stints(app, self._ratings(), self._games())
        assert len(st) == 0

    def test_a_player_with_no_rating_is_never_an_absence(self):
        """No rating row means no pool row means nothing to remove."""
        app = self._everyone_dressing(range(5))
        _, a = ist.appearances_to_stints(app, self._ratings((self.OUT_P,)),
                                         self._games())
        assert a["sides_with_absence"] == 0

    def test_the_derivation_binds_to_the_ratings(self):
        st = self._one_miss()
        v = ist.bind_check(self._ratings(), st)
        assert v["ok"] is True and v["matched"] == 1

    def test_end_to_end_the_flag_removes_a_player_from_later_games_only(self):
        st = self._one_miss()
        on, a2 = ist.team_game_rates(self._ratings(), stints=st,
                                      games=self._games())
        by_game = on.set_index("game_date")["n_players"]
        n = len(self.ROSTER)
        for i in (0, 1, 2):
            assert int(by_game.loc[pd.Timestamp(self.GAMES[i])]) == n, (
                f"the pool shrank for game {i}, which the flag cannot know")
        assert int(by_game.loc[pd.Timestamp(self.GAMES[3])]) == n - 1
        assert int(by_game.loc[pd.Timestamp(self.GAMES[4])]) == n
        assert a2["dropped_unavailable"] == 1

    # ------------------------------------------------- no evidence, no flag
    def test_empty_inputs_do_not_raise(self):
        for app in (pd.DataFrame(), None):
            st, a = ist.appearances_to_stints(app, self._ratings(), self._games())
            assert len(st) == 0, (
                "absent appearance data marked the whole league unavailable")

    def test_a_side_with_no_appearance_rows_is_not_evaluated(self):
        """One missing boxscore must not read as 'nobody dressed'."""
        app = self._everyone_dressing([0, 1, 3, 4])
        st, a = ist.appearances_to_stints(app, self._ratings(), self._games())
        assert len(st) == 0
        assert a["sides_unobserved"] == 1 and a["sides_observed"] == 4

    def test_a_truncated_roster_is_below_the_floor_and_ignored(self):
        """A side that reports fewer than the floor did not go out with nobody.

        A decided game dresses 18 a side; the floor is a truncation guard set
        well below that, so a real game is never discarded and a partial fetch
        always is.
        """
        thin = self._everyone_dressing([0, 1, 3, 4])
        thin = pd.concat([thin, self._everyone_dressing([2], self.ROSTER[:3])],
                         ignore_index=True)
        st, a = ist.appearances_to_stints(thin, self._ratings(), self._games())
        assert a["sides_observed"] == 4 and a["sides_unobserved"] == 1
        assert len(st) == 0, (
            "a three-skater side was read as nine absences")

    def test_a_appearance_frame_missing_its_columns_yields_nothing(self):
        app = self._everyone_dressing(range(5)).drop(columns=["player_id"])
        st, a = ist.appearances_to_stints(app, self._ratings(), self._games())
        assert len(st) == 0 and "reason" in a


class TestProductionInjuryLoader:
    def test_loader_uses_timestamped_status_reports_not_appearance_absences(self):
        ratings = pd.DataFrame([
            {"player_id": "mp-1", "player_name": "Alex Lee"},
            {"player_id": "mp-2", "player_name": "Bo Nix"},
        ])
        reports = pd.DataFrame([
            {"player_id": "espn-1", "player_name": "Alex Lee", "status": "Out",
             "snapshot_at": "2026-10-01T18:00:00Z", "snapshot_marker": False},
            {"player_id": "espn-2", "player_name": "Bo Nix", "status": "Suspension",
             "snapshot_at": "2026-10-01T18:00:00Z", "snapshot_marker": False},
        ])
        with (patch.object(feat.ingestion, "load_espn_injuries", return_value=reports),
              patch.object(ist, "appearances_to_stints",
                           side_effect=AssertionError("appearance proxy called"))):
            stints, sources = feat._load_injury_stints(ratings)

        assert set(stints[ist.OUT_PLAYER]) == {"mp-1"}
        assert stints.attrs["snapshot_based"] is True
        assert sources["espn"]["snapshot_times"] == [pd.Timestamp("2026-10-01 18:00:00")]

    def test_pool_filter_uses_capture_time_against_exact_game_start(self):
        ratings = make_ratings([
            ("mp-1", "BOS", "5on5", "C", 0.060, 9000.0, "2026-04-15"),
            ("mp-2", "BOS", "5on5", "C", 0.020, 9000.0, "2026-04-15"),
            ("mp-3", "TOR", "5on5", "C", 0.030, 9000.0, "2026-04-15"),
        ])
        ratings["player_name"] = ["Alex Lee", "Bo Nix", "Other Player"]
        games = pd.DataFrame([{
            "game_id": "G1", "season": 2026, "game_date": "2026-10-15",
            "start_time_utc": "2026-10-15T19:00:00Z",
            "home_team": "BOS", "away_team": "TOR",
        }])
        reports = pd.DataFrame([{
            "player_id": "espn-1", "player_name": "Alex Lee", "status": "Out",
            "report_date": "2026-10-15", "snapshot_at": "2026-10-15T18:00:00Z",
            "snapshot_marker": False,
        }])
        with patch.object(feat.ingestion, "load_espn_injuries", return_value=reports):
            out = feat.add_player_pool_features(games, player_ratings=ratings)

        assert out.loc[0, "pl_evo_c_home"] == pytest.approx(0.020)
        assert out.loc[0, "pl_evo_c_away"] == pytest.approx(0.030)
        assert pd.notna(out.loc[0, "pl_il_out_fraction"])
        assert out.loc[0, "pl_il_out_fraction"] > 0

    def test_post_puckdrop_snapshot_does_not_change_pool_and_coverage_is_unknown(self):
        ratings = make_ratings([
            ("mp-1", "BOS", "5on5", "C", 0.060, 9000.0, "2026-04-15"),
            ("mp-2", "BOS", "5on5", "C", 0.020, 9000.0, "2026-04-15"),
        ])
        ratings["player_name"] = ["Alex Lee", "Bo Nix"]
        games = pd.DataFrame([{
            "game_id": "G1", "season": 2026, "game_date": "2026-10-15",
            "start_time_utc": "2026-10-15T19:00:00Z",
            "home_team": "BOS", "away_team": "TOR",
        }])
        reports = pd.DataFrame([{
            "player_id": "espn-1", "player_name": "Alex Lee", "status": "Out",
            "snapshot_at": "2026-10-15T20:00:00Z", "snapshot_marker": False,
        }])
        with patch.object(feat.ingestion, "load_espn_injuries", return_value=reports):
            out = feat.add_player_pool_features(games, player_ratings=ratings)

        assert out.loc[0, "pl_evo_c_home"] == pytest.approx(0.040)
        assert pd.isna(out.loc[0, "pl_il_out_fraction"])


# ---------------------------------------------------------------------------
# The id bridge: ESPN athlete id -> MoneyPuck rating id
# ---------------------------------------------------------------------------
class TestIdBridge:
    """Without this the exclusion compares two disjoint vocabularies.

    ESPN athlete ids and MoneyPuck player ids share no space, so an unbound
    filter returns False for every player on every date and every "healthy"
    pool is silently the unfiltered pool. These pin the fold and, more
    importantly, the refusals.
    """

    @staticmethod
    def _ratings(rows):
        return pd.DataFrame([
            {"player_id": p, "player_name": n, "team": "BOS",
             "situation": "5on5", "position": "C",
             "shrunk_rate_per60": 0.05, "prior_ice_seconds": 9000.0,
             "game_date": pd.Timestamp("2026-04-15")}
            for p, n in rows])

    @staticmethod
    def _reports(rows):
        return pd.DataFrame([
            {"player_id": f"ESPN{i}", "player_name": n, "status": "Out",
             "report_date": "2026-09-20", "return_date": "2026-10-01"}
            for i, (_, n) in enumerate(rows)])

    def test_an_exact_name_maps_to_the_rating_id(self):
        r = self._ratings([("mp1", "A.J. Greer")])
        rep = self._reports([("ESPN0", "A.J. Greer")])
        out, a = ist.map_reports_to_rating_ids(rep, r)
        assert out[ist.OUT_PLAYER].tolist() == ["mp1"]
        assert a["matched"] == 1 and a["ambiguous"] == 0

    @pytest.mark.parametrize("feed,rating", [
        ("Adam Edstrom", "Adam Edström"),   # diacritics
        ("Owen Tippett", "Owen  Tippett"),   # whitespace
        ("Dylan Cozens", "Dylan Cozens Jr."),  # generational suffix
        ("Alex Barre-Boulet", "Alex Barre Boulet"),  # hyphen
    ])
    def test_forms_that_differ_still_match(self, feed, rating):
        r = self._ratings([("mp1", rating)])
        rep = self._reports([("ESPN0", feed)])
        out, a = ist.map_reports_to_rating_ids(rep, r)
        assert out[ist.OUT_PLAYER].tolist() == ["mp1"], f"{feed} vs {rating}"
        assert a["matched"] == 1

    def test_an_ambiguous_name_is_refused_not_guessed(self):
        """A wrong exclusion is strictly worse than no exclusion."""
        r = self._ratings([("mp1", "Alex Lee"), ("mp2", "Alex Lee")])
        rep = self._reports([("ESPN0", "Alex Lee")])
        out, a = ist.map_reports_to_rating_ids(rep, r)
        assert a["ambiguous"] == 1 and a["matched"] == 0
        # Left carrying its ESPN id, which will simply never bind.
        assert out[ist.OUT_PLAYER].tolist() == ["ESPN0"]

    def test_an_unmatched_name_is_left_alone(self):
        r = self._ratings([("mp1", "A.J. Greer")])
        rep = self._reports([("ESPN0", "Some Rookie")])
        out, a = ist.map_reports_to_rating_ids(rep, r)
        assert a["unmatched"] == 1 and a["matched"] == 0
        assert out[ist.OUT_PLAYER].tolist() == ["ESPN0"]

    def test_the_bridge_makes_the_filter_bind(self):
        """The end-to-end claim: a bridged stint removes its player."""
        ratings = make_ratings([
            ("mp1", "BOS", "5on5", "C", 0.060, 9000.0, "2026-04-15"),
            ("mp2", "BOS", "5on5", "C", 0.900, 9000.0, "2026-04-15"),
        ])
        ratings["player_name"] = ["Alex Lee", "Bo Nix"]
        rep = self._reports([("ESPN0", "Alex Lee")])
        rep["report_date"] = "2026-10-01"
        rep["return_date"] = "2026-11-01"
        rep = rep.drop(columns=["status"])
        rep["status"] = "Out"
        bridged, audit = ist.map_reports_to_rating_ids(rep, ratings)
        stints, _ = ist.build_stint_intervals(bridged)
        assert ist.bind_check(ratings, stints)["ok"] is True
        games = pd.DataFrame([{"game_date": pd.Timestamp("2026-10-15"),
                               "team": "BOS", "season": 2026}])
        off, _ = ist.team_game_rates(ratings, stints=None, games=games)
        on, a2 = ist.team_game_rates(ratings, stints=stints, games=games)
        assert a2["dropped_unavailable"] == 1
        assert off["n_players"].iloc[0] == 2 and on["n_players"].iloc[0] == 1

    def test_unbridged_the_same_stint_removes_nobody(self):
        """The failure the bridge exists to prevent, pinned as a mutant."""
        ratings = make_ratings([
            ("mp1", "BOS", "5on5", "C", 0.060, 9000.0, "2026-04-15"),
            ("mp2", "BOS", "5on5", "C", 0.900, 9000.0, "2026-04-15"),
        ])
        raw = pd.DataFrame([{"player_id": "ESPN0", "player_name": "Alex Lee",
                             "status": "Out", "report_date": "2026-10-01",
                             "return_date": "2026-11-01"}])
        stints, _ = ist.build_stint_intervals(raw)
        assert ist.bind_check(ratings, stints)["ok"] is False
        games = pd.DataFrame([{"game_date": pd.Timestamp("2026-10-15"),
                               "team": "BOS", "season": 2026}])
        _, a = ist.team_game_rates(ratings, stints=stints, games=games)
        assert a["dropped_unavailable"] == 0

    def test_normalisation_never_shortens_to_initials(self):
        """'Alex Lee' and 'Anthony Lee' must not collide."""
        assert ist.normalise_player_name("Alex Lee") != \
            ist.normalise_player_name("Anthony Lee")
        assert ist.normalise_player_name("A.J. Greer") == "a j greer"

    def test_empty_inputs_are_handled_not_raised(self):
        r = self._ratings([("mp1", "A.J. Greer")])
        for rep in (pd.DataFrame(), None):
            out, a = ist.map_reports_to_rating_ids(rep, r)
            assert a["matched"] == 0


# ---------------------------------------------------------------------------
# Tripwires
# ---------------------------------------------------------------------------
class TestTripwires:
    def test_bind_check_passes_when_ids_overlap(self):
        ratings = make_ratings([("1", "BOS", "5on5", "C", 0.02, 9000.0,
                                 "2026-01-10")])
        stints = pd.DataFrame([{"player_id": "1", "stint_start": "2026-01-01",
                                "stint_end": pd.NaT}])
        v = ist.bind_check(ratings, stints)
        assert v["ok"] is True and v["matched"] == 1

    def test_bind_check_fails_on_a_disjoint_id_space(self):
        """The silent failure MLB logs an ERROR for."""
        ratings = make_ratings([("1", "BOS", "5on5", "C", 0.02, 9000.0,
                                 "2026-01-10")])
        stints = pd.DataFrame([{"player_id": "999", "stint_start": "2026-01-01",
                                "stint_end": pd.NaT}])
        v = ist.bind_check(ratings, stints)
        assert v["ok"] is False
        assert "cannot bind" in v["reason"]

    def test_bind_check_is_ok_when_there_are_no_stints(self):
        ratings = make_ratings([("1", "BOS", "5on5", "C", 0.02, 9000.0,
                                 "2026-01-10")])
        v = ist.bind_check(ratings, pd.DataFrame())
        assert v["ok"] is True and v["reason"] == "no stints"

    def test_staleness_fires_past_the_bar(self):
        v = ist.staleness_report(pd.DataFrame(), "2026-04-16", "2026-01-01")
        assert v["stale"] is True
        assert v["lag_days"] > ist.STINT_MAX_LAG_DAYS

    def test_staleness_quiet_inside_the_bar(self):
        v = ist.staleness_report(pd.DataFrame(), "2026-01-20", "2026-01-01")
        assert v["stale"] is False

    def test_staleness_without_a_window_is_not_a_second_failure_mode(self):
        assert ist.staleness_report(pd.DataFrame(), "2026-01-20", None) == {}


# ---------------------------------------------------------------------------
# Game-grain pool: strictly-prior source dates (production v1.1)
# ---------------------------------------------------------------------------
def make_pool_ratings(rows):
    """rows: (player_id, team, position, rate, source_date).

    Production candidate ratings carry ``pool_date`` = the source game date,
    so ``team_game_rates`` takes the strict game-grain path: a candidate is
    eligible for a target game only when its source date is STRICTLY earlier
    (``merge_asof`` with ``allow_exact_matches=False``), because date-grain
    source feeds cannot prove within-day publish ordering.
    """
    return pd.DataFrame([
        {"player_id": p, "player_name": p, "team": t,
         "situation": ist.SITUATION_EVO, "position": pos,
         "shrunk_rate_per60": r, "prior_ice_seconds": 9000.0,
         "game_date": pd.Timestamp(d), "pool_date": pd.Timestamp(d)}
        for p, t, pos, r, d in rows])


def make_grid(rows):
    """rows: (game_id, date, team, season, start_time_utc)."""
    return pd.DataFrame([
        {"game_id": g, "game_date": pd.Timestamp(d), "team": t, "season": s,
         "start_time_utc": st}
        for g, d, t, s, st in rows])


class TestStrictSourceDatePool:
    """The date gate that keeps a target game's own row out of its pool.

    An inclusive trailing window means each rating row INCLUDES the stats of
    the game it is stamped on. PIT therefore lives entirely in this join: the
    same-day candidate must never serve the game it was built from.
    """

    GRID = [("G1", "2026-10-10", "BOS", 2026, "2026-10-10T23:00:00Z")]

    @staticmethod
    def _grid(dates=("2026-10-10",)):
        return make_grid([(f"G{i}", d, "BOS", 2026, f"{d}T23:00:00Z")
                          for i, d in enumerate(dates)])

    def test_same_date_candidate_is_excluded_and_prior_date_is_included(self):
        ratings = make_pool_ratings([
            ("1", "BOS", "C", 0.060, "2026-10-10"),   # same-day source
            ("2", "BOS", "C", 0.020, "2026-10-08"),   # strictly prior
        ])
        pool, audit = ist.team_game_rates(
            ratings, stints=None, games=self._grid())
        assert audit["pool_rows"] == 1
        assert pool["n_players"].iloc[0] == 1
        assert pool["rate"].iloc[0] == pytest.approx(0.020)

    def test_the_latest_strictly_prior_source_date_wins(self):
        ratings = make_pool_ratings([
            ("1", "BOS", "C", 0.060, "2026-10-01"),
            ("1", "BOS", "C", 0.050, "2026-10-08"),
            ("1", "BOS", "C", 0.900, "2026-10-10"),  # same-day, ineligible
        ])
        pool, _ = ist.team_game_rates(
            ratings, stints=None, games=self._grid())
        assert pool["rate"].iloc[0] == pytest.approx(0.050)

    def test_a_target_between_source_dates_serves_the_older_one(self):
        ratings = make_pool_ratings([
            ("1", "BOS", "C", 0.060, "2026-10-01"),
            ("1", "BOS", "C", 0.900, "2026-10-08"),
        ])
        pool, _ = ist.team_game_rates(
            ratings, stints=None, games=self._grid(dates=("2026-10-05",)))
        assert pool["rate"].iloc[0] == pytest.approx(0.060)

    def test_candidates_older_than_the_lookback_default_expire(self):
        ratings = make_pool_ratings(
            [("1", "BOS", "C", 0.060, "2026-08-01")])   # 70 days stale
        pool, _ = ist.team_game_rates(
            ratings, stints=None, games=self._grid())
        assert len(pool) == 0

    def test_a_fresh_enough_candidate_survives_the_default_window(self):
        ratings = make_pool_ratings(
            [("1", "BOS", "C", 0.060, "2026-09-05")])   # 35 days
        pool, _ = ist.team_game_rates(
            ratings, stints=None, games=self._grid())
        assert len(pool) == 1

    def test_game_lookback_days_overrides_the_default_window(self):
        ratings = make_pool_ratings(
            [("1", "BOS", "C", 0.060, "2026-10-01")])   # 9 days
        grid = self._grid()
        wide, _ = ist.team_game_rates(ratings, stints=None, games=grid)
        tight, _ = ist.team_game_rates(ratings, stints=None, games=grid,
                                       game_lookback_days=7)
        assert len(wide) == 1 and len(tight) == 0

    def test_game_id_partitions_the_pool_and_the_audit(self):
        ratings = make_pool_ratings(
            [("1", "BOS", "C", 0.060, "2026-10-08")])
        grid = self._grid(dates=("2026-10-10", "2026-10-12"))
        pool, audit = ist.team_game_rates(ratings, stints=None, games=grid)
        assert len(pool) == 2
        assert set(pool["game_id"]) == {"G0", "G1"}
        assert audit["games"] == 2


class TestDoubtfulIsAnInjury:
    """Doubtful is an approved exact designation: the same binary policy as
    Out / Injured Reserve / IR, applied only through the pool exclusion."""

    def test_the_status_set_includes_doubtful_exactly(self):
        assert ist._is_il_status("Doubtful") is True
        assert ist._is_il_status("doubtful") is True
        assert ist._is_il_status("  Doubtful  ") is True
        assert ist.IL_STATUSES == frozenset(
            {"out", "injured reserve", "ir", "doubtful"})

    def test_doubtful_snapshot_opens_an_interval(self):
        rep = pd.DataFrame([
            {"player_id": "1", "status": "Doubtful",
             "report_date": "2026-01-05", "return_date": None,
             "snapshot_at": "2026-01-05T18:00:00Z", "snapshot_marker": False},
        ])
        st, audit = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_START] == pd.Timestamp("2026-01-05 18:00:00")
        assert pd.isna(st.iloc[0][ist.STINT_END])
        assert audit["snapshot_based"] is True

    def test_a_later_dtd_snapshot_closes_the_doubtful_interval(self):
        rep = pd.DataFrame([
            {"player_id": "1", "status": "Doubtful",
             "report_date": "2026-01-05", "return_date": None,
             "snapshot_at": "2026-01-05T18:00:00Z", "snapshot_marker": False},
            {"player_id": "1", "status": "Day-To-Day",
             "report_date": "2026-01-08", "return_date": None,
             "snapshot_at": "2026-01-08T18:00:00Z", "snapshot_marker": False},
        ])
        st, _ = ist.build_stint_intervals(rep)
        assert len(st) == 1
        assert st.iloc[0][ist.STINT_END] == pd.Timestamp("2026-01-08 18:00:00")

    def test_a_doubtful_snapshot_before_puck_drop_removes_the_player(self):
        ratings = make_pool_ratings([
            ("1", "BOS", "C", 0.060, "2026-10-08"),
            ("2", "BOS", "C", 0.020, "2026-10-08"),
        ])
        grid = make_grid([("G1", "2026-10-10", "BOS", 2026,
                           "2026-10-10T23:00:00Z")])
        rep = pd.DataFrame([
            {"player_id": "1", "status": "Doubtful",
             "report_date": "2026-10-10", "return_date": None,
             "snapshot_at": "2026-10-10T18:00:00Z", "snapshot_marker": False},
        ])
        st, _ = ist.build_stint_intervals(rep)
        pool, audit = ist.team_game_rates(ratings, stints=st, games=grid)
        assert audit["dropped_unavailable"] == 1
        assert pool["n_players"].iloc[0] == 1
        assert pool["rate"].iloc[0] == pytest.approx(0.020)
        assert pool["n_unavailable"].iloc[0] == 1

    def test_a_doubtful_snapshot_after_puck_drop_excludes_nobody(self):
        ratings = make_pool_ratings([
            ("1", "BOS", "C", 0.060, "2026-10-08"),
            ("2", "BOS", "C", 0.020, "2026-10-08"),
        ])
        grid = make_grid([("G1", "2026-10-10", "BOS", 2026,
                           "2026-10-10T19:00:00Z")])
        rep = pd.DataFrame([
            {"player_id": "1", "status": "Doubtful",
             "report_date": "2026-10-10", "return_date": None,
             "snapshot_at": "2026-10-10T20:00:00Z", "snapshot_marker": False},
        ])
        st, _ = ist.build_stint_intervals(rep)
        pool, audit = ist.team_game_rates(ratings, stints=st, games=grid)
        assert audit["dropped_unavailable"] == 0
        assert pool["n_players"].iloc[0] == 2

    def test_the_rating_of_the_excluded_player_is_never_modified(self):
        """Exclusion removes the candidate from ONE target game's pool; the
        rolling rating he earned from games he played stays intact."""
        ratings = make_pool_ratings([
            ("1", "BOS", "C", 0.060, "2026-10-08"),
            ("2", "BOS", "C", 0.020, "2026-10-08"),
        ])
        grid = make_grid([("G1", "2026-10-10", "BOS", 2026,
                           "2026-10-10T23:00:00Z")])
        rep = pd.DataFrame([
            {"player_id": "1", "status": "Doubtful",
             "report_date": "2026-10-10", "return_date": None,
             "snapshot_at": "2026-10-10T18:00:00Z", "snapshot_marker": False},
        ])
        st, _ = ist.build_stint_intervals(rep)
        ist.team_game_rates(ratings, stints=st, games=grid)
        # The input frame is untouched: still two candidates at 0.060/0.020.
        assert len(ratings) == 2
        assert set(ratings["shrunk_rate_per60"]) == {0.060, 0.020}
