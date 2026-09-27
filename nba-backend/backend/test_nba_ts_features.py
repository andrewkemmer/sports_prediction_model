"""Tests for the Tier 3 projected-lineup features and the status machinery.

Three things here earn their own tests because each fails quietly:

* the seven diff features, which must be populated or absent TOGETHER rather
  than only on the days they happen to be computable;
* the PIT cutoff, where a mode change silently changes which games a status
  can gate;
* the status classification, where an unrecognised word that defaults to
  "available" would project a player the feed said something about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    # Deliberately the SAME import form every production module uses.
    # ``import config`` here would create a SECOND module object for the same
    # file alongside ``backend.config``, and a monkeypatch applied to it would
    # be silently ignored by the code under test - the test would then fail (or
    # worse, pass) for a reason that has nothing to do with the change.
    from backend import config
except ImportError:
    import config

import injury_stints as stints
import lineup_projection as proj


def _agg(day, team, **kw):
    row = {"gameday": pd.Timestamp(day), "team": team, "pool_size": 18,
           "healthy_size": 17, "lineup_out_count": 1,
           "lineup_healthy_frac": 17 / 18, "lineup_ts_concentration": 1.02,
           "lineup_ts_mean": 0.58, "lineup_ts_top3": 0.60,
           "lineup_ts_std": 0.03, "lineup_ts_rest_count": 0}
    row.update(kw)
    return row


def _slate(rows):
    return pd.DataFrame(rows)


class TestDiffFeatures:
    def test_all_seven_are_always_created(self):
        """Populated or not, every column exists.

        A feature that appears only on the days it could be computed is a
        feature whose train-serve behaviour depends on the calendar, which is
        the skew that retired MLB's six ``lineup_actual_*`` columns.
        """
        out = proj.attach_to_slate(_slate([{"gameday": "2026-03-01",
                                            "home_team": "BOS",
                                            "away_team": "MIA"}]), None)
        assert list(proj.DIFF_FEATURES) == [
            "lineup_ts_mean_diff", "lineup_ts_top3_diff",
            "lineup_ts_std_diff", "lineup_ts_rest_count_diff",
            "lineup_out_count_diff", "lineup_healthy_frac_diff",
            "lineup_ts_concentration_diff"]
        for column in proj.DIFF_FEATURES:
            assert column in out.columns
            assert out[column].isna().all()

    def test_a_diff_is_home_minus_away(self):
        aggs = pd.DataFrame([_agg("2026-03-01", "BOS", lineup_ts_mean=0.60),
                             _agg("2026-03-01", "MIA", lineup_ts_mean=0.55)])
        out = proj.attach_to_slate(
            _slate([{"gameday": "2026-03-01", "home_team": "BOS",
                     "away_team": "MIA"}]), aggs)
        assert out.iloc[0].lineup_ts_mean_diff == pytest.approx(0.05)

    def test_every_diff_is_computed_not_just_the_first(self):
        """A dict-iteration bug that filled one column and left six NaN would
        pass any single-feature test, so all seven are checked at once."""
        aggs = pd.DataFrame([
            _agg("2026-03-01", "BOS", lineup_ts_mean=0.60, lineup_ts_top3=0.62,
                 lineup_ts_std=0.05, lineup_ts_rest_count=2,
                 lineup_out_count=3, lineup_healthy_frac=0.8,
                 lineup_ts_concentration=1.10),
            _agg("2026-03-01", "MIA", lineup_ts_mean=0.55, lineup_ts_top3=0.57,
                 lineup_ts_std=0.02, lineup_ts_rest_count=1,
                 lineup_out_count=1, lineup_healthy_frac=0.9,
                 lineup_ts_concentration=1.01)])
        out = proj.attach_to_slate(
            _slate([{"gameday": "2026-03-01", "home_team": "BOS",
                     "away_team": "MIA"}]), aggs)
        row = out.iloc[0]
        assert row.lineup_ts_mean_diff == pytest.approx(0.05)
        assert row.lineup_ts_top3_diff == pytest.approx(0.05)
        assert row.lineup_ts_std_diff == pytest.approx(0.03)
        assert row.lineup_ts_rest_count_diff == pytest.approx(1)
        assert row.lineup_out_count_diff == pytest.approx(2)
        assert row.lineup_healthy_frac_diff == pytest.approx(-0.1)
        assert row.lineup_ts_concentration_diff == pytest.approx(0.09)
        for column in proj.DIFF_FEATURES:
            assert pd.notna(row[column]), f"{column} was not populated"

    def test_a_team_with_no_aggregate_yields_nan_not_a_borrowed_value(self):
        """Degrades honestly rather than reading another team's game."""
        aggs = pd.DataFrame([_agg("2026-03-01", "BOS")])
        out = proj.attach_to_slate(
            _slate([{"gameday": "2026-03-01", "home_team": "BOS",
                     "away_team": "MIA"}]), aggs)
        assert pd.isna(out.iloc[0].lineup_ts_mean_diff)

    def test_the_scratch_side_columns_are_dropped(self):
        aggs = pd.DataFrame([_agg("2026-03-01", "BOS")])
        out = proj.attach_to_slate(
            _slate([{"gameday": "2026-03-01", "home_team": "BOS",
                     "away_team": "MIA"}]), aggs)
        assert not [c for c in out.columns
                    if c.startswith("_home_") or c.startswith("_away_")]


class TestAvailabilityFeatures:
    def test_out_count_and_fraction_are_both_reported(self):
        """A COUNT and a FRACTION answer different questions.

        Two players missing from an eighteen-man pool is a tenth of the
        roster, not a third, and the two move the line for different reasons.
        """
        ratings = pd.DataFrame([
            {"player_id": p, "team": "BOS", "gameday": "2026-03-01",
             "ts_shrunk": 0.5 + 0.01 * i, "prior_plays": 300,
             "is_available": available}
            for i, (p, available) in enumerate(
                [("a", True), ("b", True), ("c", False), ("d", False)])])
        out = proj.projected_lineup(ratings, top_k=3)
        row = out[out.team == "BOS"].iloc[0]
        assert row.pool_size == 4
        assert row.healthy_size == 2
        assert row.lineup_out_count == 2
        assert row.lineup_healthy_frac == pytest.approx(0.5)

    def test_concentration_is_the_top3_share_of_the_mean(self):
        ratings = pd.DataFrame([
            {"player_id": "a", "team": "BOS", "gameday": "2026-03-01",
             "ts_shrunk": 0.60, "prior_plays": 500, "is_available": True},
            {"player_id": "b", "team": "BOS", "gameday": "2026-03-01",
             "ts_shrunk": 0.50, "prior_plays": 400, "is_available": True},
            {"player_id": "c", "team": "BOS", "gameday": "2026-03-01",
             "ts_shrunk": 0.50, "prior_plays": 300, "is_available": True}])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        assert row.lineup_ts_concentration == pytest.approx(
            row.lineup_ts_top3 / row.lineup_ts_mean)

    def test_an_empty_pool_does_not_divide_by_zero(self):
        ratings = pd.DataFrame([
            {"player_id": "a", "team": "BOS", "gameday": "2026-03-01",
             "ts_shrunk": 0.5, "prior_plays": 1, "is_available": True}])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        assert row.healthy_size == 0
        assert pd.isna(row.lineup_healthy_frac) or row.lineup_healthy_frac == 0.0


class TestFeatureCoverage:
    def test_a_constant_feature_is_flagged(self):
        """The other failure a slot-consuming column can have."""
        slate = _slate([{"lineup_ts_mean_diff": 0.0},
                        {"lineup_ts_mean_diff": 0.0}])
        report = proj.feature_coverage(slate)
        row = report[report.feature == "lineup_ts_mean_diff"].iloc[0]
        assert bool(row.constant) is True
        assert row.distinct_values == 1
        assert row.coverage == 1.0

    def test_an_unpopulated_feature_shows_zero_coverage(self):
        slate = _slate([{"lineup_ts_mean_diff": np.nan}] * 4)
        report = proj.feature_coverage(slate)
        row = report[report.feature == "lineup_ts_mean_diff"].iloc[0]
        assert row.coverage == 0.0
        assert row.populated == 0

    def test_a_missing_column_is_reported_not_raised(self):
        report = proj.feature_coverage(_slate([{"gameday": "2026-03-01"}]))
        assert len(report) == len(proj.DIFF_FEATURES)
        assert (report.coverage == 0.0).all()


class TestStatusClassification:
    def test_the_vocabulary_is_the_official_filing(self):
        """Pinned because the mapping below implies more than we have.

        This assertion used to be the OPPOSITE: it pinned the vocabulary to
        ESPN's two statuses and refused Doubtful / Questionable / Probable as
        "NFL words". That reasoning was sound - do not encode a vocabulary you
        do not have - but it was based on the only source then available, and
        the official NBA filing publishes all six. The rule survives; the
        evidence behind it changed.

        What must not come back is ``day_to_day``. That is ESPN's word for this
        league, and keeping it would let a mapper treat a two-status snapshot
        and a six-status official filing as the same thing.
        """
        treatment = config.PLAYER_TS_STATUS_TREATMENT
        for designation in ("out", "doubtful", "recovery", "questionable",
                            "available", "probable"):
            assert designation in treatment
        assert "day_to_day" not in treatment

    def test_an_unlisted_status_is_unknown_not_available(self):
        """The safe direction: never silently project a player the feed
        flagged, and never silently suppress a healthy one either."""
        assert stints.treatment_of("Felt A Little Under The Weather") \
            == "unknown"

    def test_a_doubtful_status_is_still_an_absence(self):
        """Doubtful is an absence on evidence - 0 players in 5 filings."""
        assert stints.is_absent("Doubtful") is True
        assert stints.is_absent("Out") is True
        assert stints.is_absent("Active") is False

    def test_questionable_is_not_an_absence(self):
        """Pooled into the available bucket. It played 34 of 52, so calling
        it an absence deleted a third of a real player's appearances."""
        assert stints.treatment_of("Questionable") == "available"
        assert stints.is_absent("Questionable") is False

    def test_the_espn_word_is_not_silently_treated_as_absent(self):
        """``Day-To-Day`` is not an NBA designation. Dropping it rather than
        mapping it is the safe direction: an unrecognised status must not
        delete a player from a projection."""
        assert stints.treatment_of("Day-To-Day") == "unknown"
        assert stints.is_absent("Day-To-Day") is False

    def test_the_observed_vocabulary_is_counted(self):
        seen = stints.observed_vocabulary(["Out", "Day-To-Day", "Out", ""])
        assert seen == {"Out": 2, "Day-To-Day": 1}


class TestPITCutoff:
    def _stint(self, start, end=pd.NaT):
        return pd.DataFrame([{"player_id": "1", "team": "BOS",
                              "il_start": pd.Timestamp(start),
                              "il_end": pd.Timestamp(end) if end is not pd.NaT
                              else pd.NaT, "status": "out"}])

    def test_tipoff_mode_reads_at_the_game(self, monkeypatch):
        monkeypatch.setattr(config, "PLAYER_TS_INJURY_CUTOFF", "tipoff")
        assert stints.cutoff_for("2026-03-05T19:30Z") == pd.Timestamp(
            "2026-03-05T19:30")
        # Published the morning of the game: it CAN gate tonight.
        assert stints.out_at(self._stint("2026-03-05T09:00Z"),
                             "1", "2026-03-05T19:30Z") is True

    def test_prior_end_of_day_mode_cannot_be_gated_by_game_day_news(
            self, monkeypatch):
        """The conservative reading: nothing published on game day counts."""
        monkeypatch.setattr(config, "PLAYER_TS_INJURY_CUTOFF",
                            "prior_end_of_day")
        assert stints.out_at(self._stint("2026-03-05T09:00Z"),
                             "1", "2026-03-05T19:30Z") is False
        # Published the day before: it counts.
        assert stints.out_at(self._stint("2026-03-04T09:00Z"),
                             "1", "2026-03-05T19:30Z") is True

    def test_both_modes_agree_on_an_absence_days_old(self, monkeypatch):
        frame = self._stint("2026-03-01T09:00Z")
        for mode in ("tipoff", "prior_end_of_day"):
            monkeypatch.setattr(config, "PLAYER_TS_INJURY_CUTOFF", mode)
            assert stints.out_at(frame, "1", "2026-03-05T19:30Z") is True

    def test_the_default_is_the_finest_granularity(self):
        assert config.PLAYER_TS_INJURY_CUTOFF == "tipoff"


class TestPlayRateReporter:
    def _history(self, rows):
        return pd.DataFrame(rows, columns=["snapshot_date", "player_id",
                                           "team", "status", "raw_status",
                                           "published_at"])

    def test_no_history_says_so_instead_of_reporting_a_number(self):
        """The honest state, and the one that motivates keeping the history."""
        report = stints.play_rates(pd.DataFrame(), pd.DataFrame())
        assert report["sufficient"] is False
        assert "unanswerable" in report["note"] or "no injury history" in \
            report["note"]

    def test_a_small_sample_is_not_reported_as_sufficient(self):
        history = self._history([
            ("2026-03-01", "1", "BOS", "day_to_day", "Day-To-Day", None),
            ("2026-03-02", "1", "BOS", "day_to_day", "Day-To-Day", None)])
        appearances = pd.DataFrame({"player_id": ["1"],
                                    "gameday": ["2026-03-01"]})
        report = stints.play_rates(history, appearances)
        assert report["sufficient"] is False
        assert "not yet findings" in report["note"]

    def test_a_large_sample_reports_the_rate_with_its_size(self):
        rows = []
        appearances = {"player_id": [], "gameday": []}
        for day in range(1, 29):
            player = f"{day}"
            played = day % 4 != 0        # plays on 3 of every 4 days
            rows.append(("2026-03-01", player, "BOS", "day_to_day",
                         "Day-To-Day", None))
            appearances["player_id"].append(player)
            appearances["gameday"].append(f"2026-03-{day:02d}")
            if not played:
                # Not appearing on the snapshot day is the negative case; the
                # join is exact-day, so a different date is not a "did not play".
                appearances["player_id"].pop()
                appearances["gameday"].pop()
                rows[-1] = (f"2026-03-{day:02d}", player, "BOS", "day_to_day",
                            "Day-To-Day", None)
        history = self._history(rows)
        report = stints.play_rates(history, pd.DataFrame(appearances))
        entry = report["by_status"]["day_to_day"]
        assert entry["total"] >= 20
        assert 0.0 <= entry["play_rate"] <= 1.0
        assert report["sufficient"] is True

    def test_a_status_with_no_appearances_reads_as_zero_not_missing(self):
        history = self._history([
            ("2026-03-01", f"p{i}", "BOS", "out", "Out", None)
            for i in range(25)])
        report = stints.play_rates(history, pd.DataFrame(
            {"player_id": [], "gameday": []}))
        # No appearances at all is handled before the join, so this is the
        # no-log path rather than a measured zero.
        assert report["sufficient"] is False
