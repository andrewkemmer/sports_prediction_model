"""Tests for the injury-stint table and the projected-lineup pool.

The PIT guardrail gets the first test in the file on purpose. Everything else
here is a correctness question that a reviewer can reason about; the
point-in-time rule is the one that must be true, and the one that fails
quietly - a leak does not raise, it just produces a slightly better number than
the truth, which is the hardest kind of bug to notice from the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import injury_stints as stints
import lineup_projection as proj


def _roster(player, team="BOS", status="Out", published="2026-03-01T12:00Z"):
    return {"player_id": player, "team": team, "status": status,
            "raw_status": status, "published_at": published}


# ===========================================================================
# The guardrail: strictly point-in-time prior to game start
# ===========================================================================


class TestPointInTimeIsStrict:
    def _stint(self, start, end=pd.NaT):
        return pd.DataFrame([{"player_id": "1", "team": "BOS",
                              "il_start": pd.Timestamp(start),
                              "il_end": pd.Timestamp(end) if end is not pd.NaT
                              else pd.NaT,
                              "status": "out"}])

    def test_a_status_published_before_tipoff_applies(self):
        frame = self._stint("2026-03-01T12:00Z")
        assert stints.out_at(frame, "1", "2026-03-05T19:30Z") is True

    def test_a_status_published_exactly_at_tipoff_does_not_apply(self):
        """The strict edge, and the one that is easiest to get wrong.

        A status timestamped at the instant of tipoff is not knowable to
        anyone placing a bet at that instant, so it cannot gate that game. A
        ``<=`` here silently admits information from the future, and it does so
        by one record per game, which is invisible in the output.
        """
        frame = self._stint("2026-03-05T19:30Z")
        assert stints.out_at(frame, "1", "2026-03-05T19:30Z") is False

    def test_the_same_record_applies_to_a_later_game(self):
        """Strictness is per game, not a global skip.

        The identical record that cannot gate a 19:30 game absolutely can gate
        tomorrow's, and a filter that dropped the record entirely would lose
        the absence where it matters.
        """
        frame = self._stint("2026-03-05T19:30Z")
        assert stints.out_at(frame, "1", "2026-03-06T19:30Z") is True

    def test_a_return_published_before_tipoff_ends_the_absence(self):
        frame = self._stint("2026-03-01T12:00Z", "2026-03-04T09:00Z")
        assert stints.out_at(frame, "1", "2026-03-03T19:30Z") is True
        assert stints.out_at(frame, "1", "2026-03-05T19:30Z") is False

    def test_an_open_stint_covers_every_later_tipoff(self):
        """The NULL is handled INSIDE the predicate.

        MLB documents the trap: an open stint has a NULL ``il_end``, so a
        ``LEFT JOIN ... IS NULL`` emptiness test flags 59.6% of all
        participants as injured. Here the NULL can only mean "not closed yet".
        """
        frame = self._stint("2026-03-01T12:00Z")
        assert stints.out_at(frame, "1", "2030-01-01T19:30Z") is True

    def test_a_return_published_exactly_at_tipoff_does_not_clear_it(self):
        """Symmetric strictness at the far edge.

        A recovery published at tipoff is not knowable at tipoff either, so the
        player is still out for that game and available from the next one.
        """
        frame = self._stint("2026-03-01T12:00Z", "2026-03-05T19:30Z")
        assert stints.out_at(frame, "1", "2026-03-05T19:30Z") is True
        assert stints.out_at(frame, "1", "2026-03-06T19:30Z") is False

    def test_a_timezone_aware_record_matches_a_naive_tipoff(self):
        """Mixed awareness raises rather than coercing, which is the safer
        failure, but ESPN publishes ``Z`` timestamps so the common path has to
        work."""
        frame = self._stint(pd.Timestamp("2026-03-01T12:00Z"))
        assert stints.out_at(frame, "1", "2026-03-05T19:30") is True

    def test_no_table_means_nobody_is_out(self):
        assert stints.out_at(pd.DataFrame(), "1", "2026-03-05") is False
        assert stints.out_at(None, "1", "2026-03-05") is False


# ===========================================================================
# Records and the state machine
# ===========================================================================


class TestStintConstruction:
    def test_repeated_absent_records_collapse_to_one_interval(self):
        """THE bug MLB's builder documents, and the reason for a state machine.

        The feed republishes the same status on every snapshot, so a two-week
        absence emits a long run of identical records. Pairing an opening
        against a closing leaves every intermediate record as its own
        permanently-open stint, suppressing the player for the rest of time.
        """
        records = stints.records_from_rosters([("2026-03-10", pd.DataFrame([
            _roster("1", published="2026-03-01T12:00Z"),
            _roster("1", published="2026-03-05T12:00Z"),
            _roster("1", published="2026-03-09T12:00Z"),
        ]))])
        built = stints.build_stints(records)
        assert len(built) == 1
        assert built.iloc[0].il_start == pd.Timestamp("2026-03-01T12:00Z").tz_localize(None)
        assert pd.isna(built.iloc[0].il_end)

    def test_a_healthy_observation_closes_the_stint(self):
        records = stints.records_from_rosters([("2026-03-10", pd.DataFrame([
            _roster("1", status="Out", published="2026-03-01T12:00Z"),
            _roster("1", status="Active", published="2026-03-05T12:00Z"),
        ]))])
        # The healthy row is not itself an event, so it is not a record; the
        # close has to be carried by the observation rather than the absence.
        assert len(records) == 1

    def test_healthy_records_are_not_stored_at_all(self):
        """Storing them would make "no record" ambiguous.

        A missing record has to mean "no absence was ever observed", which is
        different from "healthy", so only absent records are kept.
        """
        records = stints.records_from_rosters([("2026-03-10", pd.DataFrame([
            _roster("1", status="Active", published="2026-03-05T12:00Z"),
        ]))])
        assert records.empty

    def test_a_doubtful_absence_is_an_absence(self):
        # "Doubtful" is the NBA's own word and is an absence on evidence
        # (0 players in 5 filings). "Questionable" is NOT - it played 34 of 52
        # and is pooled with the available designations.
        records = stints.records_from_rosters([("2026-03-10", pd.DataFrame([
            _roster("1", status="Doubtful", published="2026-03-01T12:00Z"),
        ]))])
        assert len(records) == 1
        assert records.iloc[0].status == "doubtful"
        assert stints.build_stints(records).iloc[0].status == "doubtful"

    def test_a_questionable_record_is_not_recorded_as_an_absence(self):
        """A player the league merely flagged uncertain still gets projected."""
        records = stints.records_from_rosters([("2026-03-10", pd.DataFrame([
            _roster("1", status="Questionable", published="2026-03-01T12:00Z"),
        ]))])
        assert len(records) == 0

    def test_a_missing_record_date_falls_back_to_the_snapshot_and_says_so(self):
        """The fallback is recorded, because it is an approximation.

        Without the ``dated`` flag a caller cannot tell a real publication time
        from an assumed one, and an assumed one is exactly where a PIT leak
        would come from.
        """
        records = stints.records_from_rosters([("2026-03-10", pd.DataFrame([
            {"player_id": "1", "team": "BOS", "status": "out",
             "raw_status": "Out", "published_at": None},
        ]))])
        assert records.iloc[0].dated is False or not records.iloc[0].dated
        assert records.iloc[0].published_at == pd.Timestamp("2026-03-10")

    def test_no_records_yields_a_well_formed_empty_frame(self):
        built = stints.build_stints(stints.records_from_rosters([]))
        assert built.empty
        assert list(built.columns) == stints.STINT_COLUMNS


# ===========================================================================
# Reconciliation against observed appearances
# ===========================================================================


class TestAppearanceReconciliation:
    def _stint(self, end=pd.NaT):
        return pd.DataFrame([{"player_id": "1", "team": "BOS",
                              "il_start": pd.Timestamp("2026-03-01"),
                              "il_end": pd.Timestamp(end) if end is not pd.NaT
                              else pd.NaT, "status": "out"}])

    def test_an_open_stint_closes_at_the_first_appearance_after_it(self):
        """MLB's highest-value rule, and the one that made the feature safe.

        A player with a player-game row was in the game, so no absence may span
        it. A status the feed never clears would otherwise suppress a player
        indefinitely.
        """
        appearances = pd.DataFrame({"player_id": ["1", "1"],
                                   "gameday": ["2026-03-04", "2026-03-06"]})
        out = stints.reconcile_with_appearances(self._stint(), appearances)
        assert out.iloc[0].il_end == pd.Timestamp("2026-03-04")

    def test_an_appearance_on_the_start_date_is_not_a_return(self):
        """The feed publishes an injury the same day it is reported.

        A same-day appearance is the announcement, not a return, so it must not
        close the stint - otherwise the very act of filing the injury clears it.
        """
        appearances = pd.DataFrame({"player_id": ["1"],
                                   "gameday": ["2026-03-01"]})
        out = stints.reconcile_with_appearances(self._stint(), appearances)
        assert pd.isna(out.iloc[0].il_end)

    def test_a_later_appearance_does_not_shorten_an_earlier_close(self):
        appearances = pd.DataFrame({"player_id": ["1"],
                                   "gameday": ["2026-03-20"]})
        out = stints.reconcile_with_appearances(
            self._stint("2026-03-10"), appearances)
        assert out.iloc[0].il_end == pd.Timestamp("2026-03-10")

    def test_the_rule_is_strictly_subtractive(self):
        """It can only shorten an interval, so it cannot invent an absence.

        A player who demonstrably played on a date is not released back into
        the projection on any date where he was genuinely absent.
        """
        appearances = pd.DataFrame({"player_id": ["1"],
                                   "gameday": ["2026-03-04"]})
        out = stints.reconcile_with_appearances(self._stint(), appearances)
        # Still out before the appearance.
        assert stints.out_at(out, "1", "2026-03-02") is True
        # No longer out after it.
        assert stints.out_at(out, "1", "2026-03-05") is False

    def test_no_appearances_leaves_the_table_untouched(self):
        assert len(stints.reconcile_with_appearances(self._stint(),
                                                     pd.DataFrame())) == 1


# ===========================================================================
# The pool: widen, then filter, then rank
# ===========================================================================


def _bos(aggregates):
    """The 2026-03-01 BOS row.

    The fixture also carries a 2026-01-01 rating for an out-of-lookback
    player, which sorts first, so selecting by team alone picks the wrong game.
    """
    rows = aggregates[aggregates.team == "BOS"]
    return rows[rows.gameday == pd.Timestamp("2026-03-01")].iloc[0]


def _rating(player, team, day, ts, plays, available=True,
            days_since_appearance=None):
    return {"player_id": player, "team": team, "gameday": day,
            "ts_shrunk": ts, "prior_plays": plays, "is_available": available,
            "days_since_appearance": days_since_appearance}


class TestProjectedLineup:
    def _ratings(self, **overrides):
        rows = [
            _rating("a", "BOS", "2026-03-01", 0.60, 400),
            _rating("b", "BOS", "2026-03-01", 0.58, 380),
            _rating("c", "BOS", "2026-03-01", 0.55, 300),
            _rating("d", "BOS", "2026-03-01", 0.40, 120, available=False),
            _rating("e", "BOS", "2026-03-01", 0.30, 10),
            _rating("f", "BOS", "2026-01-01", 0.70, 900),
        ]
        rows = [dict(row, **overrides.get(row["player_id"], {})) for row in rows]
        return pd.DataFrame(rows)

    def test_an_injured_player_leaves_the_pool_entirely(self):
        """Removal, not zero-weighting.

        Multiplying his rating by zero would leave him IN the mean dragging it
        toward zero, instead of letting a replacement inherit the slot.
        """
        out = proj.projected_lineup(self._ratings())
        row = _bos(out)
        # a, b, c are healthy; d is removed entirely rather than scored 0.0.
        # Blend weights are each member's own prior_plays (opportunity):
        # (400*.60 + 380*.58 + 300*.55) / 1080.
        assert row.healthy_size == 3
        assert row.lineup_ts_mean == pytest.approx(
            (400 * 0.60 + 380 * 0.58 + 300 * 0.55) / 1080)

    def test_the_pool_is_widened_before_it_is_filtered(self):
        """The step that makes the filter bind at all.

        A pool drawn only from players who appeared recently cannot subtract an
        injured player, because being injured is what removed them from that
        pool. Here the injured player IS in the pool and IS removed from it.
        """
        ratings = self._ratings()
        out = proj.projected_lineup(ratings)
        row = _bos(out)
        assert row.pool_size == 5   # a-e are inside the lookback
        assert row.healthy_size == 3

    def test_a_player_outside_the_lookback_is_not_a_candidate(self):
        """``f`` last rated in January and is more than 10 days back."""
        ratings = self._ratings()
        out = proj.projected_lineup(ratings)
        assert _bos(out).pool_size == 5

    def test_the_min_plays_floor_gates_membership_not_the_rating(self):
        """``e`` has 10 plays - under the floor - and is excluded from the pool.

        Shrinkage already handles a thin RATING; this is about whether the
        player is a candidate for tonight's lineup at all.
        """
        out = proj.projected_lineup(self._ratings())
        assert _bos(out).healthy_size == 3

    def test_a_short_handed_roster_is_not_padded(self):
        """Padding would fabricate full strength from a depleted roster."""
        ratings = pd.DataFrame([
            _rating("a", "BOS", "2026-03-01", 0.60, 400),
            _rating("b", "BOS", "2026-03-01", 0.55, 300, available=False),
        ])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        assert row.healthy_size == 1
        assert row.lineup_ts_mean == pytest.approx(0.60)
        assert row.lineup_ts_std != row.lineup_ts_std  # NaN: one player


    def test_ranking_is_by_participation_not_by_rating(self):
        """Ordering by rating would answer "who scores" instead of "who plays".

        The highest-rated player in the fixture is also the most established,
        so the ranking is checked by giving a low-volume, high-rating player
        the inside track on ordering while staying out of the average.
        """
        ratings = pd.DataFrame([
            _rating("low_vol_high_rating", "BOS", "2026-03-01", 0.70, 60),
            _rating("a", "BOS", "2026-03-01", 0.55, 500),
            _rating("b", "BOS", "2026-03-01", 0.56, 480),
        ])
        out = proj.projected_lineup(ratings, top_k=2)
        row = _bos(out)
        # top_k=2 takes the two highest prior_plays: a and b, not the 0.70.
        # The blend prices them by opportunity: (500*.55 + 480*.56) / 980.
        assert row.lineup_ts_mean == pytest.approx(
            (500 * 0.55 + 480 * 0.56) / 980)

    def test_the_stint_table_removes_a_player_the_flag_calls_healthy(self):
        """The table is evaluated PIT and ANDed with the carried flag."""
        ratings = self._ratings(**{"a": {"is_available": True}})
        table = pd.DataFrame([{"player_id": "a", "team": "BOS",
                               "il_start": pd.Timestamp("2026-02-01"),
                               "il_end": pd.NaT, "status": "out"}])
        out = proj.projected_lineup(ratings, stints=table)
        # a leaves on the table's evidence; d was already out on the flag.
        assert _bos(out).healthy_size == 2

    def test_the_two_sources_are_anded_never_replaced(self):
        """A source may only ever REMOVE a player, never add one back.

        Letting the stint table overwrite the carried flag would resurrect a
        player the roster marks unavailable but who has no interval yet - the
        exact state a fresh absence is in before the next snapshot closes it.
        """
        ratings = self._ratings(**{"a": {"is_available": False}})
        table = pd.DataFrame([{"player_id": "a", "team": "BOS",
                               "il_start": pd.Timestamp("2026-02-01"),
                               "il_end": pd.NaT, "status": "out"}])
        without = proj.projected_lineup(ratings)
        with_table = proj.projected_lineup(ratings, stints=table)
        assert _bos(without).healthy_size == _bos(with_table).healthy_size

    def test_no_ratings_yields_a_well_formed_empty_frame(self):
        for value in (None, pd.DataFrame()):
            out = proj.projected_lineup(value)
            assert out is not None and out.empty
            assert "lineup_ts_mean" in out.columns


class TestRecencyGate:
    """A pool member whose evidence has gone stale is a phantom.

    Rows are emitted per target date for every player who EVER appeared in
    the season, so a player who stopped appearing (injury never filed,
    quiet shutdown, roster cut) keeps riding the pool on evidence that only
    LOOKS fresh - no designation will ever remove him. The audit's worst
    case sat in a pool 217 days after his last game. The gate reads the
    player's actual last appearance; NaN (season not started for him) is
    the season-start carryover, not staleness, and stays eligible.
    """

    def test_a_player_whose_appearance_gap_exceeds_the_gate_is_excluded(self):
        # "ghost" appeared 40 days before the target: past the 30-day gate.
        ratings = pd.DataFrame([
            _rating("a", "BOS", "2026-03-01", 0.60, 400,
                    days_since_appearance=1),
            _rating("ghost", "BOS", "2026-03-01", 0.90, 900,
                    days_since_appearance=40),
        ])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        # The ghost is WIDENED into the pool (that step predates the gates)
        # but excluded from membership, so he cannot enter the lineup or the
        # mean - exactly the phantom the audit found riding pools for months.
        assert row.pool_size == 2
        assert row.healthy_size == 1
        assert row.lineup_ts_mean == pytest.approx(0.60)

    def test_a_player_inside_the_gate_stays_in_the_pool(self):
        ratings = pd.DataFrame([
            _rating("a", "BOS", "2026-03-01", 0.60, 400,
                    days_since_appearance=30),
            _rating("b", "BOS", "2026-03-01", 0.55, 300,
                    days_since_appearance=3),
        ])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        assert row.pool_size == 2 and row.healthy_size == 2

    def test_a_nan_gap_is_carryover_not_staleness_and_stays_eligible(self):
        """The season has not started for this player: the min-plays floor
        governs him, not the recency gate."""
        ratings = pd.DataFrame([
            _rating("a", "BOS", "2026-03-01", 0.60, 400,
                    days_since_appearance=None),
        ])
        out = proj.projected_lineup(ratings)
        assert out[out.team == "BOS"].iloc[0].pool_size == 1

    def test_the_rating_frame_carries_the_appearance_gap(self, monkeypatch):
        """build_player_ts must compute the gap from the player's actual
        last appearance in the rated season - the input the gate reads."""
        import player_ts as ts_mod
        games = pd.DataFrame({
            "player_id": ["p1", "p1", "p2"],
            "gameday": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-01-05"]),
            "season": ["2025-26"] * 3,
            "team": ["BOS", "BOS", "BOS"],
            "points": [20.0, 22.0, 18.0],
            "fga": [15.0, 16.0, 14.0],
            "fta": [4.0, 5.0, 3.0],
            "plays": [19.0, 21.0, 17.0],
            "position": ["G", "G", "F"],
        })
        ratings = ts_mod.build_player_ts(
            games, target_dates=pd.Series([pd.Timestamp("2026-02-10")]))
        row_p1 = ratings[ratings.player_id == "p1"].iloc[0]
        row_p2 = ratings[ratings.player_id == "p2"].iloc[0]
        assert row_p1.days_since_appearance == 9    # last app 2026-02-01
        assert row_p2.days_since_appearance == 36   # last app 2026-01-05

    def test_a_low_workload_member_cannot_price_half_the_lineup(self):
        """The NFL v9.4 guard, NBA face: the blend is combined shrunk TS over
        combined opportunities, so discarding a 21-play bench piece moves the
        projection onto the starter's own rating instead of leaving him at
        half weight. The plain mean kept the absent piece priced equally."""
        ratings = pd.DataFrame([
            _rating("starter", "BOS", "2026-03-01", 0.55, 500,
                    days_since_appearance=1),
            _rating("bench", "BOS", "2026-03-01", 0.90, 21,
                    days_since_appearance=1),
        ])
        out = proj.projected_lineup(ratings)
        full = out[out.team == "BOS"].iloc[0]
        # Weighted: (500*.55 + 21*.90) / 521 - the starter dominates.
        assert full.lineup_ts_mean == pytest.approx(
            (500 * 0.55 + 21 * 0.90) / 521)
        # The plain mean priced the 21-play bench piece at half: 0.725.
        assert full.lineup_ts_mean < (0.55 + 0.90) / 2
        # Without him, the blend IS the starter's own rating - not the mean
        # of the remaining members at equal weight.
        out2 = proj.projected_lineup(ratings.iloc[[0]])
        solo = out2[out2.team == "BOS"].iloc[0]
        assert solo.lineup_ts_mean == pytest.approx(0.55)

    def test_weights_are_the_strictly_prior_opportunity_totals(self):
        """The weight source is prior_plays - the same PIT quantity the
        ranking reads - so the blend leaks nothing: it reuses evidence
        already point-in-time before the game."""
        ratings = pd.DataFrame([
            _rating("a", "BOS", "2026-03-01", 0.60, 400,
                    days_since_appearance=2),
            _rating("b", "BOS", "2026-03-01", 0.58, 380,
                    days_since_appearance=1),
        ])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        # Ratio identity: equal ratings blend to themselves regardless of
        # weights, so equal-weight-vs-weighted divergence needs unequal
        # ratings. Pin the weighted form on both aggregates - top3 blends
        # the same two members here, so it equals the lineup blend.
        weighted = (400 * 0.60 + 380 * 0.58) / 780
        assert row.lineup_ts_top3 == pytest.approx(weighted)
        assert row.lineup_ts_mean == pytest.approx(weighted)

    def test_a_nan_rating_contributes_no_weight_and_no_rating(self):
        """Removal, not zeroing: a NaN-rated member's plays vanish from the
        denominator too, and a family with no finite-weight member is NaN."""
        ratings = pd.DataFrame([
            _rating("a", "BOS", "2026-03-01", 0.60, 400),
            _rating("nan_guy", "BOS", "2026-03-01", np.nan, 300),
        ])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        assert row.lineup_ts_mean == pytest.approx(0.60)
        assert row.lineup_ts_std != row.lineup_ts_std  # one finite member

    def test_a_fallback_evidence_season_carries_nan_not_a_cross_season_gap(
            self):
        """Season-start carryover, the real-data face the 2026-09-29 Kaggle
        run exposed: an October slate target rates from the LAST COMPLETED
        season (the evidence fallback), so every carryover member's gap
        computed across the fallback read ~150 days and the recency gate
        emptied every projected lineup (mean healthy 0.0, all nine pl_ts
        columns empty on the slate). The gap is only defined within the
        target's OWN season; a fallback-evidence target carries NaN, which
        the gate treats as carryover governed by the min-plays floor - the
        same semantics a mid-season first-game player gets."""
        import player_ts as ts_mod
        games = pd.DataFrame({
            "player_id": ["p1", "p1"],
            "gameday": pd.to_datetime(["2026-01-10", "2026-01-20"]),
            "season": ["2025-26", "2025-26"],
            "team": ["BOS", "BOS"],
            "points": [20.0, 22.0],
            "fga": [15.0, 16.0],
            "fta": [4.0, 5.0],
            "plays": [19.0, 21.0],
            "position": ["G", "G"],
        })
        ratings = ts_mod.build_player_ts(
            games, target_dates=pd.Series([pd.Timestamp("2026-10-28")]))
        assert len(ratings)
        row = ratings.iloc[0]
        # The target's own season (2026-27) has no evidence, so the rating
        # falls back to 2025-26 - and the gap MUST be NaN, not ~292 days.
        assert pd.isna(row.days_since_appearance)

    def test_the_column_is_documented_in_the_emitted_contract(self):
        """The column exists in build_player_ts's declared output set, so a
        caller reading the frame by contract sees it."""
        import inspect
        import player_ts as ts_mod
        src = inspect.getsource(ts_mod.build_player_ts)
        assert "days_since_appearance" in src


class TestStaleness:
    def test_lag_is_reported_so_a_frozen_table_is_visible(self):
        """A table that stops updating keeps every open stint open.

        The failure mode is a slowly growing set of permanently-suppressed
        players, invisible until someone counts them.
        """
        table = pd.DataFrame([{"player_id": "1", "team": "BOS",
                               "il_start": pd.Timestamp("2026-01-01"),
                               "il_end": pd.NaT, "status": "out"}])
        report = stints.staleness(table, decided_max_date="2026-06-01")
        assert report["lag_days"] == 151
        assert report["open_at_window_end"] == 1

    def test_no_table_reports_nothing_rather_than_raising(self):
        report = stints.staleness(pd.DataFrame())
        assert report["stints"] == 0 and report["lag_days"] is None


class TestConfigMirrorsMlb:
    @pytest.mark.parametrize("nba,mlb,why", [
        (config.PLAYER_TS_MIN_PLAYS, 20, "LINEUP_MIN_PA"),
        (config.PLAYER_TS_POOL_LOOKBACK_DAYS, 10, "LINEUP_POOL_LOOKBACK_DAYS"),
        (config.PLAYER_TS_REST_PLAYS, 50, "LINEUP_REST_PA"),
        (config.PLAYER_TS_TOP5_K, 5, "LINEUP_TOP5_K"),
        (config.PLAYER_TS_MAX_LAG_DAYS, 45, "IL_STINT_MAX_LAG_DAYS"),
    ])
    def test_the_guards_match_their_mlb_namesakes(self, nba, mlb, why):
        """Pinned so a change here is a deliberate edit, not a drift."""
        assert nba == mlb, f"{why} analogue has drifted"


class TestPitDesignationFilter:
    """The point-in-time removal contract, and its one-game scope.

    A designation is a fact about ONE game. The rating is a fact about the
    games a player actually played. These tests pin both halves, because
    the failure modes are opposite and both plausible-looking: over-removal
    (a player erased from games he was cleared for) and under-removal (an
    Out player silently left in tonight's pool).
    """

    def _designations(self, *rows):
        return pd.DataFrame(
            [{"gameday": g, "team": t, "player": p, "status": s}
             for g, t, p, s in rows])

    def _ratings(self):
        # Two games, same three players. "a" is Out for game 2 only.
        return pd.DataFrame([
            _rating("a", "BOS", "2026-03-01", 0.60, 400),
            _rating("b", "BOS", "2026-03-01", 0.58, 380),
            _rating("c", "BOS", "2026-03-01", 0.55, 300),
            _rating("a", "BOS", "2026-03-05", 0.60, 420),
            _rating("b", "BOS", "2026-03-05", 0.58, 400),
            _rating("c", "BOS", "2026-03-05", 0.55, 330),
        ])

    def test_out_removal_binds_on_exactly_the_designated_game(self):
        designations = self._designations(
            ("2026-03-05", "BOS", "Alice", "Out"))
        out = proj.apply_pit_designations(
            self._ratings(), designations,
            name_by_id={"a": "Alice", "b": "Bob", "c": "Carl"})
        early = out[out.gameday == pd.Timestamp("2026-03-01")]
        assert bool(early.is_available.all())
        late = out[(out.gameday == pd.Timestamp("2026-03-05"))
                   & (out.player_id == "a")]
        assert not bool(late.is_available.iloc[0])

    def test_the_player_keeps_his_rating_on_every_other_game(self):
        """Removal is one-game; the rolling rating survives everywhere else."""
        designations = self._designations(
            ("2026-03-05", "BOS", "Alice", "Out"))
        out = proj.apply_pit_designations(
            self._ratings(), designations,
            name_by_id={"a": "Alice", "b": "Bob", "c": "Carl"})
        # His 2026-03-01 row is untouched AND his rating value is untouched.
        row = out[(out.gameday == pd.Timestamp("2026-03-01"))
                  & (out.player_id == "a")].iloc[0]
        assert bool(row.is_available) and row.ts_shrunk == 0.60

    def test_doubtful_and_recovery_remove_questionable_does_not(self):
        designations = self._designations(
            ("2026-03-05", "BOS", "Bob", "Doubtful"),
            ("2026-03-05", "BOS", "Carl", "Recovery"),
            ("2026-03-05", "BOS", "Alice", "Questionable"))
        out = proj.apply_pit_designations(
            self._ratings(), designations,
            name_by_id={"a": "Alice", "b": "Bob", "c": "Carl"})
        day = out[out.gameday == pd.Timestamp("2026-03-05")]
        assert not bool(day[day.player_id == "b"].is_available.iloc[0])
        assert not bool(day[day.player_id == "c"].is_available.iloc[0])
        # Questionable is a play-rate bucket, not an absence.
        assert bool(day[day.player_id == "a"].is_available.iloc[0])

    def test_an_unresolvable_name_removes_nothing(self):
        """A designation that cannot name a player suppresses nothing."""
        designations = self._designations(
            ("2026-03-05", "BOS", "Mystery Player", "Out"))
        out = proj.apply_pit_designations(
            self._ratings(), designations,
            name_by_id={"a": "Alice", "b": "Bob", "c": "Carl"})
        assert bool(out.is_available.all())

    def test_removal_means_the_replacement_inherits_the_slot(self):
        designations = self._designations(
            ("2026-03-05", "BOS", "Alice", "Out"))
        ratings = proj.apply_pit_designations(
            self._ratings(), designations,
            name_by_id={"a": "Alice", "b": "Bob", "c": "Carl"})
        out = proj.projected_lineup(ratings)
        row = out[(out.team == "BOS")
                  & (out.gameday == pd.Timestamp("2026-03-05"))].iloc[0]
        # b and c cover the slots; Alice is not zero-weighted into the mean.
        # Weights are the March-5 rows' own prior_plays (b 400, c 330).
        assert row.healthy_size == 2
        assert row.lineup_ts_mean == pytest.approx(
            (400 * 0.58 + 330 * 0.55) / 730)
        # And the March 1 projection is unchanged by a March 5 designation.
        early = out[(out.team == "BOS")
                    & (out.gameday == pd.Timestamp("2026-03-01"))].iloc[0]
        assert early.healthy_size == 3


class TestPositionTsFeatures:
    """The nine published columns and their MLB alignment."""

    def test_the_nine_columns_exist_in_the_contract(self):
        expected = [f"pl_ts_{p}_{s}" for p in ("c", "f", "g")
                    for s in ("away", "home", "diff")]
        for col in expected:
            assert col in config.MONEYLINE_FEATURE_COLS, col
        assert config.PLAYER_TS_POSITION_FEATURE_COLS == expected

    def test_sides_are_retained_not_diffed_away(self):
        """MLB keeps lineup_woba_mean_home AND _away; so does the NBA mirror."""
        assert proj.POSITION_TS_FEATURES == [
            "pl_ts_c_away", "pl_ts_c_home", "pl_ts_c_diff",
            "pl_ts_f_away", "pl_ts_f_home", "pl_ts_f_diff",
            "pl_ts_g_away", "pl_ts_g_home", "pl_ts_g_diff",
        ]

    def test_attach_produces_all_nine_columns_even_when_empty(self):
        slate = pd.DataFrame({
            "gameday": pd.to_datetime(["2026-03-05"]),
            "home_team": ["BOS"], "away_team": ["NYK"],
            "game_id": ["g1"],
        })
        empty_agg = pd.DataFrame({c: pd.Series(dtype="float64")
                                  for c in proj.AGG_COLUMNS})
        empty_agg["gameday"] = pd.Series(dtype="datetime64[ns]")
        empty_agg["team"] = pd.Series(dtype="str")
        out = proj.attach_position_ts(slate, empty_agg)
        for col in proj.POSITION_TS_FEATURES:
            assert col in out.columns and out[col].isna().all()

    def test_diff_is_home_minus_away(self):
        slate = pd.DataFrame({
            "gameday": pd.to_datetime(["2026-03-05"]),
            "home_team": ["BOS"], "away_team": ["NYK"],
            "game_id": ["g1"],
        })
        aggregates = pd.DataFrame([{
            "gameday": pd.Timestamp("2026-03-05"), "team": "BOS",
            "pl_ts_c": 0.62, "pl_ts_f": 0.55, "pl_ts_g": 0.58,
        }, {
            "gameday": pd.Timestamp("2026-03-05"), "team": "NYK",
            "pl_ts_c": 0.60, "pl_ts_f": 0.57, "pl_ts_g": 0.54,
        }])
        out = proj.attach_position_ts(slate, aggregates)
        assert out.pl_ts_c_diff.iloc[0] == pytest.approx(0.62 - 0.60)
        assert out.pl_ts_g_home.iloc[0] == pytest.approx(0.58)
        assert out.pl_ts_g_away.iloc[0] == pytest.approx(0.54)


class TestPlayerTsIsStrictlyPriorPerGame:
    def test_a_ratings_target_date_excludes_that_dates_games(self):
        """The core PIT edge: the rated game never rates itself.

        Verified independently: recompute the prior from raw rows with a
        strict ``<`` and compare to what the rating published.
        """
        import player_ts as ts_mod
        log = pd.DataFrame([
            {"player_id": 1.0, "gameday": "2026-03-01", "points": 20,
             "fga": 12, "fta": 4, "player_name": "A", "team": "BOS"},
            {"player_id": 1.0, "gameday": "2026-03-05", "points": 30,
             "fga": 15, "fta": 2, "player_name": "A", "team": "BOS"},
            {"player_id": 2.0, "gameday": "2026-03-01", "points": 10,
             "fga": 8, "fta": 0, "player_name": "B", "team": "BOS"},
            {"player_id": 2.0, "gameday": "2026-03-05", "points": 12,
             "fga": 9, "fta": 1, "player_name": "B", "team": "BOS"},
        ])
        positions = pd.DataFrame([
            {"player_id": 1.0, "season": "2025-26", "position": "G"},
            {"player_id": 2.0, "season": "2025-26", "position": "F"},
        ])
        games = ts_mod.prepare_player_games(log, positions)
        ratings = ts_mod.build_player_ts(
            games, target_dates=pd.Series(["2026-03-05"]))
        row = ratings[ratings.player_id == "1"].iloc[0]
        # Strictly the 03-01 game: 20 points / (2*(12 + 0.44*4)) plays.
        assert row.prior_points == 20
        assert row.prior_plays == pytest.approx(12 + 0.44 * 4)
        assert row.prior_games == 1


def _shard(path, rows):
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return frame


class TestDesignationShardUnion:
    """Every backfilled window counts, not whichever file sorts last.

    The production path read ``sorted(glob(...))[-1]``, so with shards for
    2025-10-21..2026-04-12 (13,168 records) and 2026-01-08..2026-01-12 (400) the
    run applied the 400-record shard: the injury removal bound on five days of a
    six-month window and every other game looked healthy. A filename sort is not
    a coverage policy.
    """

    def test_every_shard_is_read_not_just_the_last(self, tmp_path):
        _shard(tmp_path / "nba_designations_20251021_20260412.parquet",
               [{"gameday": "2025-11-01", "team": "BOS", "player_report": "A, B",
                 "status": "Out", "published_at": "2025-11-01T18:00Z"}])
        _shard(tmp_path / "nba_designations_20260108_20260112.parquet",
               [{"gameday": "2026-01-08", "team": "LAL", "player_report": "C, D",
                 "status": "Doubtful", "published_at": "2026-01-08T18:00Z"}])
        out = proj.load_designations(tmp_path)
        assert len(out) == 2
        assert set(out.player_report) == {"A, B", "C, D"}

    def test_an_overlapping_window_is_idempotent(self, tmp_path):
        """A designation is a (date, team, player) removal read as a set, so a
        record in two shards must remove once, not twice."""
        row = {"gameday": "2026-01-08", "team": "LAL", "player_report": "C, D",
               "status": "Out", "published_at": "2026-01-08T18:00Z"}
        _shard(tmp_path / "nba_designations_20251021_20260412.parquet", [row])
        _shard(tmp_path / "nba_designations_20260108_20260112.parquet", [row])
        assert len(proj.load_designations(tmp_path)) == 1

    def test_no_archive_is_none_not_an_empty_healthy_league(self, tmp_path):
        assert proj.load_designations(tmp_path) is None
        assert proj.load_designations(None) is None

    def test_one_unreadable_shard_does_not_discard_the_rest(self, tmp_path,
                                                            caplog):
        _shard(tmp_path / "nba_designations_20251021_20260412.parquet",
               [{"gameday": "2025-11-01", "team": "BOS", "player_report": "A, B",
                 "status": "Out", "published_at": "2025-11-01T18:00Z"}])
        (tmp_path / "nba_designations_20260108_20260112.parquet").write_text(
            "not a parquet file")
        with caplog.at_level("WARNING"):
            out = proj.load_designations(tmp_path)
        assert len(out) == 1
        assert "unreadable" in caplog.text


class TestPositionTsBuildSurvivesAColdCache:
    """The production run's failure, reproduced and fixed.

    A fresh cache - every full repull, every new Kaggle kernel - left this
    phase reading an empty positions directory, and ``pd.concat([])`` answered
    "no position table" with ``ValueError: No objects to concatenate``. The
    caller caught it and degraded the nine promoted features to NaN, so the run
    published a 41-feature contract in which nine were empty.
    """

    @staticmethod
    def _facts():
        """Two games of history before the targets, because the pool's
        membership floor is 20 prior plays and a fixture with no history is
        honestly unrateable rather than a failure of the build."""
        import ingestion
        days = ["2026-02-24", "2026-02-26", "2026-03-01", "2026-03-02"]
        games = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(len(days) + 1)],
            "gameday": pd.to_datetime(days + ["2026-03-08"]),
            "home_team": ["BOS"] * (len(days) + 1),
            "away_team": ["NYK"] * len(days) + ["LAL"],
            "home_score": [110.0] * len(days) + [None],
            "away_score": [100.0] * len(days) + [None],
        })
        rows = []
        for day, (b_pts, n_pts) in zip(days, ((20, 10), (30, 12), (24, 11),
                                              (30, 12))):
            rows.append({"player_id": 1.0, "gameday": day, "points": b_pts,
                         "fga": 12, "fta": 4, "player_name": "A", "team": "BOS"})
            rows.append({"player_id": 2.0, "gameday": day, "points": n_pts,
                         "fga": 9, "fta": 1, "player_name": "B", "team": "NYK"})
        log = pd.DataFrame(rows)
        return ingestion.NBAFacts(
            games=games, team_stats=pd.DataFrame(), player_stats=log,
            team_events=pd.DataFrame(), play_by_play=pd.DataFrame(),
            team_names={}, manifest={}), games

    @staticmethod
    def _positions(monkeypatch):
        """Patch the accessor on the module the build actually calls.

        ``master_pipeline`` reaches ingestion as ``backend.ingestion`` when the
        suite is collected as a package, so patching a bare ``import ingestion``
        can land on a second copy of the module and change nothing - a test
        that silently stops testing, and reads the developer's real cache
        instead of the fixture. The module object under test is the one to
        patch.
        """
        import master_pipeline as mp
        frame = pd.DataFrame([
            {"player_id": 1.0, "season": "2025-26", "position": "G"},
            {"player_id": 2.0, "season": "2025-26", "position": "F"},
        ])
        monkeypatch.setattr(mp.ingestion, "_fetch_positions",
                            lambda season, use_cache=True: frame)
        return frame

    def test_an_empty_positions_directory_builds_instead_of_raising(
            self, monkeypatch, tmp_path):
        import master_pipeline as mp
        self._positions(monkeypatch)
        facts, games = self._facts()
        frame, slate = mp._build_position_ts_features(facts, games, cache_dir=tmp_path)
        assert frame is not None and len(frame)
        for col in config.PLAYER_TS_POSITION_FEATURE_COLS:
            assert col in frame.columns
        assert frame[config.PLAYER_TS_POSITION_FEATURE_COLS].notna().any().any()
        assert slate is not None and len(slate) == 1   # the unplayed game
        # The slate is the case the production run lost: the nine columns have
        # to be present AND valued on the game being served, not merely there.
        assert slate[config.PLAYER_TS_POSITION_FEATURE_COLS].notna().any().any()

    def test_no_concat_of_nothing_is_a_frame_not_an_exception(self):
        import master_pipeline as mp
        assert mp._concat_frames([], "positions").empty
        assert len(mp._concat_frames([pd.DataFrame(), pd.DataFrame({"a": [1]})],
                                     "positions")) == 1

    def test_a_missing_injury_archive_is_reported_not_assumed_healthy(
            self, monkeypatch, tmp_path, caplog):
        import master_pipeline as mp
        import lineup_projection as proj
        self._positions(monkeypatch)
        facts, games = self._facts()
        # The build falls back to the DELIVERY copy of the archive when the
        # machine cache is empty - the path a cloud run actually takes. The
        # loader is stubbed to answer None for the delivery root so the
        # warning path itself is what gets exercised; the published-archive
        # path has its own test in TestDesignationShardUnion.
        real_loader = proj.load_designations

        def _scoped_loader(root):
            base = str(root)
            if "data_delivery" in base.replace("\\", "/"):
                return None
            return real_loader(root)

        monkeypatch.setattr(proj, "load_designations", _scoped_loader)
        with caplog.at_level("WARNING"):
            mp._build_position_ts_features(facts, games, cache_dir=tmp_path)
        assert "UNFILTERED" in caplog.text

    def test_a_contract_column_no_row_carries_is_named(self):
        import master_pipeline as mp
        frame = pd.DataFrame({
            "pl_ts_c_away": [0.5, None], "pl_ts_f_away": [None, None],
            "rest": [1, 2]})
        assert mp._empty_contract_columns(frame) == ["pl_ts_f_away"]
        assert mp._empty_contract_columns(None) == []
        assert mp._empty_contract_columns(pd.DataFrame()) == []
