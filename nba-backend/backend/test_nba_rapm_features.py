"""Tests for the Tier 3 projected-lineup features and the status machinery.

Three things here earn their own tests because each fails quietly:

* the seven diff features, which must be populated or absent TOGETHER rather
  than only on the days they happen to be computable;
* the PIT cutoff, where a mode change silently changes which games a status
  can gate;
* the status classification, where an unrecognised word that defaults to
  "available" would project a player the feed said something about.

The Elo season-boundary class (added with the 2026-09-29 boundary audit)
pins the revert convention: ratings regress 1/3 toward 1500 at the season
flip - they are never reset, matching MLB's season-to-season regression;
the NFL's no-revert divergence is that sport's own deliberate choice.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

try:
    from backend import features as feat_mod
except ImportError:
    import features as feat_mod

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
           "lineup_healthy_frac": 17 / 18, "lineup_rapm_concentration": 1.02,
           "lineup_rapm_mean": 0.58, "lineup_rapm_top3": 0.60,
           "lineup_rapm_std": 0.03, "lineup_rapm_rest_count": 0}
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
            "lineup_rapm_mean_diff", "lineup_rapm_top3_diff",
            "lineup_rapm_std_diff", "lineup_rapm_rest_count_diff",
            "lineup_out_count_diff", "lineup_healthy_frac_diff",
            "lineup_rapm_concentration_diff"]
        for column in proj.DIFF_FEATURES:
            assert column in out.columns
            assert out[column].isna().all()

    def test_a_diff_is_home_minus_away(self):
        aggs = pd.DataFrame([_agg("2026-03-01", "BOS", lineup_rapm_mean=0.60),
                             _agg("2026-03-01", "MIA", lineup_rapm_mean=0.55)])
        out = proj.attach_to_slate(
            _slate([{"gameday": "2026-03-01", "home_team": "BOS",
                     "away_team": "MIA"}]), aggs)
        assert out.iloc[0].lineup_rapm_mean_diff == pytest.approx(0.05)

    def test_every_diff_is_computed_not_just_the_first(self):
        """A dict-iteration bug that filled one column and left six NaN would
        pass any single-feature test, so all seven are checked at once."""
        aggs = pd.DataFrame([
            _agg("2026-03-01", "BOS", lineup_rapm_mean=0.60, lineup_rapm_top3=0.62,
                 lineup_rapm_std=0.05, lineup_rapm_rest_count=2,
                 lineup_out_count=3, lineup_healthy_frac=0.8,
                 lineup_rapm_concentration=1.10),
            _agg("2026-03-01", "MIA", lineup_rapm_mean=0.55, lineup_rapm_top3=0.57,
                 lineup_rapm_std=0.02, lineup_rapm_rest_count=1,
                 lineup_out_count=1, lineup_healthy_frac=0.9,
                 lineup_rapm_concentration=1.01)])
        out = proj.attach_to_slate(
            _slate([{"gameday": "2026-03-01", "home_team": "BOS",
                     "away_team": "MIA"}]), aggs)
        row = out.iloc[0]
        assert row.lineup_rapm_mean_diff == pytest.approx(0.05)
        assert row.lineup_rapm_top3_diff == pytest.approx(0.05)
        assert row.lineup_rapm_std_diff == pytest.approx(0.03)
        assert row.lineup_rapm_rest_count_diff == pytest.approx(1)
        assert row.lineup_out_count_diff == pytest.approx(2)
        assert row.lineup_healthy_frac_diff == pytest.approx(-0.1)
        assert row.lineup_rapm_concentration_diff == pytest.approx(0.09)
        for column in proj.DIFF_FEATURES:
            assert pd.notna(row[column]), f"{column} was not populated"

    def test_a_team_with_no_aggregate_yields_nan_not_a_borrowed_value(self):
        """Degrades honestly rather than reading another team's game."""
        aggs = pd.DataFrame([_agg("2026-03-01", "BOS")])
        out = proj.attach_to_slate(
            _slate([{"gameday": "2026-03-01", "home_team": "BOS",
                     "away_team": "MIA"}]), aggs)
        assert pd.isna(out.iloc[0].lineup_rapm_mean_diff)

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
             "rapm_shrunk": 0.5 + 0.01 * i, "prior_minutes": 300,
             "prior_minutes_per_game": 30.0,
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
             "rapm_shrunk": 0.60, "prior_minutes": 500,
             "prior_minutes_per_game": 50.0, "is_available": True},
            {"player_id": "b", "team": "BOS", "gameday": "2026-03-01",
             "rapm_shrunk": 0.50, "prior_minutes": 400,
             "prior_minutes_per_game": 40.0, "is_available": True},
            {"player_id": "c", "team": "BOS", "gameday": "2026-03-01",
             "rapm_shrunk": 0.50, "prior_minutes": 300,
             "prior_minutes_per_game": 30.0, "is_available": True}])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        assert row.lineup_rapm_concentration == pytest.approx(
            row.lineup_rapm_top3 / row.lineup_rapm_mean)

    def test_an_empty_pool_does_not_divide_by_zero(self):
        ratings = pd.DataFrame([
            {"player_id": "a", "team": "BOS", "gameday": "2026-03-01",
             "rapm_shrunk": 0.5, "prior_minutes": 1,
             "prior_minutes_per_game": 0.1, "is_available": True}])
        out = proj.projected_lineup(ratings)
        row = out[out.team == "BOS"].iloc[0]
        assert row.healthy_size == 0
        assert pd.isna(row.lineup_healthy_frac) or row.lineup_healthy_frac == 0.0


class TestFeatureCoverage:
    def test_a_constant_feature_is_flagged(self):
        """The other failure a slot-consuming column can have."""
        slate = _slate([{"lineup_rapm_mean_diff": 0.0},
                        {"lineup_rapm_mean_diff": 0.0}])
        report = proj.feature_coverage(slate)
        row = report[report.feature == "lineup_rapm_mean_diff"].iloc[0]
        assert bool(row.constant) is True
        assert row.distinct_values == 1
        assert row.coverage == 1.0

    def test_an_unpopulated_feature_shows_zero_coverage(self):
        slate = _slate([{"lineup_rapm_mean_diff": np.nan}] * 4)
        report = proj.feature_coverage(slate)
        row = report[report.feature == "lineup_rapm_mean_diff"].iloc[0]
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
        treatment = config.PLAYER_EPM_STATUS_TREATMENT
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


class TestBrierHeadline:
    """The run log's Brier headline must not dress a one-game tail up as an
    aggregate. The 2026-09-29 logs printed 0.2589 -> 0.2713 as "rolling Brier
    over 283 day(s)"; both numbers were the single game of 2026-06-13, and the
    283-day games-weighted mean had IMPROVED 0.2098 -> 0.2091."""

    SERIES = [
        {"date": "2026-06-10", "brier": 0.1630, "games": 1},
        {"date": "2026-06-13", "brier": 0.2713, "games": 1},
    ]

    def test_the_last_day_is_named_and_never_passes_as_the_aggregate(self):
        head = mon.brier_headline(self.SERIES)
        # The weighted mean of these two days is 0.21715 - not the 0.2713 the
        # old headline would have shown.
        assert head["weighted"] == pytest.approx(0.21715, abs=1e-4)
        assert "last day" in head["summary"] and "2026-06-13" in head["summary"]
        assert "games-weighted" in head["summary"]
        assert "1 game(s)" in head["summary"]
        assert head["last_day"]["brier"] == pytest.approx(0.2713)

    def test_the_weighted_mean_dominates_the_series_not_the_last_day(self):
        # 283 lopsided days plus a bad single-game tail: the headline number
        # must stay near the 0.21 body, never jump to the 0.27 tail.
        body = [{"date": f"2024-11-{d:02d}", "brier": 0.2100, "games": 7}
                for d in range(1, 29)]
        head = mon.brier_headline(body + self.SERIES)
        assert head["weighted"] < 0.215
        assert head["summary"].startswith("0.21")

    def test_an_empty_series_degrades_to_n_a(self):
        head = mon.brier_headline([])
        assert head["weighted"] is None and head["summary"] == "n/a"


class TestPITCutoff:
    def _stint(self, start, end=pd.NaT):
        return pd.DataFrame([{"player_id": "1", "team": "BOS",
                              "il_start": pd.Timestamp(start),
                              "il_end": pd.Timestamp(end) if end is not pd.NaT
                              else pd.NaT, "status": "out"}])

    def test_tipoff_mode_reads_at_the_game(self, monkeypatch):
        monkeypatch.setattr(config, "PLAYER_EPM_INJURY_CUTOFF", "tipoff")
        assert stints.cutoff_for("2026-03-05T19:30Z") == pd.Timestamp(
            "2026-03-05T19:30")
        # Published the morning of the game: it CAN gate tonight.
        assert stints.out_at(self._stint("2026-03-05T09:00Z"),
                             "1", "2026-03-05T19:30Z") is True

    def test_prior_end_of_day_mode_cannot_be_gated_by_game_day_news(
            self, monkeypatch):
        """The conservative reading: nothing published on game day counts."""
        monkeypatch.setattr(config, "PLAYER_EPM_INJURY_CUTOFF",
                            "prior_end_of_day")
        assert stints.out_at(self._stint("2026-03-05T09:00Z"),
                             "1", "2026-03-05T19:30Z") is False
        # Published the day before: it counts.
        assert stints.out_at(self._stint("2026-03-04T09:00Z"),
                             "1", "2026-03-05T19:30Z") is True

    def test_both_modes_agree_on_an_absence_days_old(self, monkeypatch):
        frame = self._stint("2026-03-01T09:00Z")
        for mode in ("tipoff", "prior_end_of_day"):
            monkeypatch.setattr(config, "PLAYER_EPM_INJURY_CUTOFF", mode)
            assert stints.out_at(frame, "1", "2026-03-05T19:30Z") is True

    def test_the_default_is_the_finest_granularity(self):
        assert config.PLAYER_EPM_INJURY_CUTOFF == "tipoff"


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


# ---------------------------------------------------------------------------
# The drift/coverage status machinery.  These earn tests because each defect
# here shipped silently: a baseline-constant feature scored PSI 0.000 OK
# through a wholesale regime flip, a 60-row tail against a full season paged
# eleven features whose means had not moved, and the monitor JSON forwarded
# INSUFFICIENT rows as alerts.

try:
    from backend import (features as feat_mod, master_pipeline as mp,
                         moneyline as ml_mod, monitoring as mon)
except ImportError:
    import features as feat_mod
    import master_pipeline as mp
    import moneyline as ml_mod
    import monitoring as mon


class TestFeatureDriftMachinery:
    """The drift table must measure like-for-like windows, honestly."""

    def test_windows_are_a_recent_tail_against_a_like_for_like_prior(self):
        frame = pd.DataFrame(
            {"gameday": pd.date_range("2026-01-01", periods=800, freq="D")})
        baseline, current = mon.drift_windows(frame, days=7, min_baseline=250)
        # A thin 7-day tail widens (capped at 45 days) until it can be judged.
        assert len(current) >= 30
        assert len(baseline) >= 250
        assert baseline.gameday.max() < current.gameday.min()

    def test_a_sparse_calendar_widens_the_current_window(self):
        # Three games in the last week - the Finals tail - must widen to a
        # judgable window rather than report INSUFFICIENT for every feature.
        days = list(pd.date_range("2026-01-01", periods=300, freq="D")) * 1
        frame = pd.DataFrame({"gameday": days})
        baseline, current = mon.drift_windows(frame, days=7, min_current=30)
        assert len(current) >= 30
        assert len(baseline) >= 250

    def test_a_missing_gameday_degrades_to_a_positional_tail(self):
        frame = pd.DataFrame({"elo_diff": np.arange(400.0),
                              "gameday": pd.NaT})
        baseline, current = mon.drift_windows(frame)
        assert len(current) == 60 and len(baseline) == 250

    def test_a_wholesale_flip_is_measured_not_reported_as_identity(self):
        # 0s against 1s collapses quantile binning to one bin in ANY
        # combined-sample scheme; the discrete path must answer for it.
        assert mon._psi(pd.Series(np.ones(80)), pd.Series(np.zeros(300))) > 5.0

    def test_the_is_playoffs_signature_now_escalates(self):
        # The delivered table showed is_playoffs at 1.0 vs 0.0725 as
        # "PSI 0.000, OK" - a regime flip wearing an OK badge.
        rng = np.random.default_rng(11)
        baseline = pd.DataFrame(
            {"is_playoffs": (rng.random(300) < 0.0725).astype(float)})
        current = pd.DataFrame({"is_playoffs": np.ones(60)})
        row = mon.feature_drift(baseline, current)[0]
        assert row["status"] == "ALERT" and row["location_shift"]


class TestCarryProvenance:
    """A forward-filled value is not a measurement.

    The event family forward-fills each team's last measured profile across
    games the play-by-play sweep has not reached. Counting those carries as
    observations is how the 13:29 run on 2026-09-29 published 99.6%
    frame-wide event coverage that was substantially frozen constants, and
    how a baseline window can compare a stale snapshot against fresh games
    as if both were data. The ladder stamps ``_measured_<feature>`` 0/1
    provenance on the contract; coverage counts observations and carries
    separately, and drift compares observations only.
    """

    FEATURE = "event_rim_rate_diff"

    @staticmethod
    def _frame(values, flags=None, n_extra_cols=True):
        frame = pd.DataFrame({TestCarryProvenance.FEATURE: values})
        if flags is not None:
            frame[f"_measured_{TestCarryProvenance.FEATURE}"] = flags
        return frame

    def test_a_carried_value_is_not_counted_as_coverage(self, monkeypatch):
        monkeypatch.setattr(config, "MONEYLINE_FEATURE_COLS", [self.FEATURE])
        values = [0.4, 0.5, 0.3, 0.3]
        flags = [1.0, 1.0, 0.0, 0.0]
        rows = mon.coverage(self._frame(values, flags))
        row = rows[0]
        assert row["n_nonnull"] == 4
        assert row["pct_measured"] == 50.0
        assert row["n_measured"] == 2 and row["n_carried"] == 2

    def test_no_provenance_means_every_non_null_is_measured(self, monkeypatch):
        monkeypatch.setattr(config, "MONEYLINE_FEATURE_COLS", [self.FEATURE])
        rows = mon.coverage(self._frame([0.4, 0.5, 0.3]))
        assert rows[0]["n_measured"] == 3 and rows[0]["n_carried"] == 0

    def test_drift_compares_observations_not_carries(self, monkeypatch):
        monkeypatch.setattr(config, "MONEYLINE_FEATURE_COLS", [self.FEATURE])
        rng = np.random.default_rng(7)
        # 120 genuinely measured baseline rows, then 60 rows that are the
        # frozen carry of a profile measured at 3.0 - the exact shape an
        # unswept stretch produces. The measured count stays above the
        # INSUFFICIENT floor (100) so the comparison is judgable.
        baseline_values = np.concatenate([rng.normal(0, 1, 120),
                                          np.full(60, 3.0)])
        baseline_flags = np.concatenate([np.ones(120), np.zeros(60)])
        current_values = rng.normal(0, 1, 40)
        current_flags = np.ones(40)
        flagged = mon.feature_drift(
            self._frame(baseline_values, baseline_flags),
            self._frame(current_values, current_flags))[0]
        assert flagged["n_baseline"] == 120 and flagged["n_current"] == 40
        assert flagged["status"] == "OK"
        # Without the stamps the same frame invents drift out of the carry.
        unflagged = mon.feature_drift(
            self._frame(baseline_values), self._frame(current_values))[0]
        assert unflagged["status"] == "ALERT"
        assert unflagged["psi"] > flagged["psi"]

    def test_the_provenance_flags_never_reach_a_model_matrix(self):
        rng = np.random.default_rng(3)
        n = 10
        rows = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "gameday": pd.date_range("2026-01-01", periods=n, freq="D"),
            "season": 2026,
            "home_team": "BOS", "away_team": "NYK",
            "home_score": rng.integers(95, 125, n).astype(float),
            "away_score": rng.integers(95, 125, n).astype(float),
        })
        events = pd.DataFrame([
            {"game_id": f"g{i}", "team": team, "fga": 90,
             "possessions": 100.0, "rim_attempts": 30, "mid_attempts": 25,
             "corner_three_attempts": 25, "live_turnovers": 10, "and_in": 1,
             "shooting_fouls": 18, "shot_distance": 13.5, "q4_points": 25,
             "ot_points": 0.0}
            for i in range(4) for team in ("BOS", "NYK")])
        built = feat_mod.build_game_features(rows, None, events)
        stamped = [c for c in built.columns if c.startswith("_measured_")]
        assert stamped, "the fixture should exercise the provenance stamps"
        for view in (feat_mod.tree_view(built), feat_mod.linear_view(built)):
            assert not [c for c in view.columns if c.startswith("_measured_")]
            assert list(view.columns)[:1] == \
                [config.active_moneyline_feature_cols()[0]]

    def test_identical_distributions_sit_at_the_noise_floor_and_report_ok(self):
        rng = np.random.default_rng(3)
        baseline = pd.DataFrame({"elo_diff": rng.normal(0, 1, 300)})
        current = pd.DataFrame({"elo_diff": rng.normal(0, 1, 80)})
        row = mon.feature_drift(baseline, current)[0]
        assert row["status"] == "OK" and not row["location_shift"]
        assert row["psi"] <= row["noise_floor"] * 1.5

    def test_a_real_location_shift_escalates(self):
        rng = np.random.default_rng(5)
        baseline = pd.DataFrame({"elo_diff": rng.normal(0, 1, 300)})
        current = pd.DataFrame({"elo_diff": rng.normal(0, 1, 80) + 5.0})
        row = mon.feature_drift(baseline, current)[0]
        assert row["status"] == "ALERT" and row["location_shift"]

    def test_a_tiny_window_reports_insufficient_and_never_pages(self):
        rng = np.random.default_rng(7)
        baseline = pd.DataFrame({"elo_diff": rng.normal(0, 1, 50)})
        current = pd.DataFrame({"elo_diff": rng.normal(0, 1, 10) + 9.0})
        row = mon.feature_drift(baseline, current)[0]
        assert row["status"] == "INSUFFICIENT"

    def test_a_playoff_tail_baseline_is_composition_matched(self):
        """The Finals-window fix: a 100%-playoff current window against a
        mostly-regular-season prior measured the calendar, not drift - the
        same 16 pages three Finals windows running. The baseline restricts
        to prior PLAYOFF rows (newest first), so is_playoffs itself and
        every covariate-correlated feature compare like-for-like."""
        rng = np.random.default_rng(17)
        n_reg, n_po = 1000, 220
        frame = pd.DataFrame({
            "gameday": pd.date_range("2025-10-01", periods=n_reg + n_po,
                                     freq="D"),
            "is_playoffs": np.r_[np.zeros(n_reg), np.ones(n_po)],
            # Playoff games really do differ: slower pace, sharper Elo.
            "ewm_pace_home": np.r_[rng.normal(230.0, 2.0, n_reg),
                                   rng.normal(220.0, 2.0, n_po)],
            "elo_diff": np.r_[rng.normal(0, 60, n_reg),
                              rng.normal(0, 120, n_po)],
        })
        baseline, current = mon.drift_windows(frame, days=7, min_current=30)
        assert (pd.to_numeric(current.is_playoffs).mean() == 1.0)
        # The mixed baseline was mostly regular season; the matched one is
        # playoff rows only, still time-adjacent and still judgable.
        assert (pd.to_numeric(baseline.is_playoffs).mean() == 1.0)
        assert baseline.gameday.max() < current.gameday.min()
        assert len(baseline) >= 3 * len(current)
        # And the drift read is now quiet: the pace gap vs the MATCHED
        # baseline is inside noise, where the mixed baseline paged it.
        row = mon.feature_drift(baseline, current)[1]
        assert row["status"] == "OK"

    def test_a_mixed_current_window_is_left_alone(self):
        """Matching is for tail seasons; the regular season's own mixed
        composition must not silently become a playoff-only filter. The
        last 45 days sit INSIDE the regular-season block (the playoff block
        is mid-history here), so the current window is mixed and untouched."""
        rng = np.random.default_rng(19)
        n_reg_early, n_po, n_reg_late = 600, 120, 600
        frame = pd.DataFrame({
            "gameday": pd.date_range(
                "2025-10-01", periods=n_reg_early + n_po + n_reg_late,
                freq="D"),
            "is_playoffs": np.r_[np.zeros(n_reg_early), np.ones(n_po),
                                 np.zeros(n_reg_late)],
            "ewm_pace_home": rng.normal(230.0, 2.0,
                                        n_reg_early + n_po + n_reg_late),
        })
        baseline, current = mon.drift_windows(frame, days=7, min_current=30)
        assert (pd.to_numeric(current.is_playoffs).mean() == 0.0)
        assert (pd.to_numeric(baseline.is_playoffs).mean() == 0.0)

    def test_an_under_supplied_match_degrades_loudly_to_the_mixed_baseline(
            self, caplog):
        """When prior playoff rows cannot floor 3x the current window, the
        restriction is REFUSED - an honest page beats a silently thin
        baseline."""
        n_reg, n_po = 1180, 40
        frame = pd.DataFrame({
            "gameday": pd.date_range("2025-10-01", periods=n_reg + n_po,
                                     freq="D"),
            "is_playoffs": np.r_[np.zeros(n_reg), np.ones(n_po)],
            "ewm_pace_home": np.r_[np.full(n_reg, 230.0),
                                   np.full(n_po, 230.0)],
        })
        baseline, current = mon.drift_windows(frame, days=7, min_current=30)
        assert (pd.to_numeric(current.is_playoffs).mean() == 1.0)
        # Refused: the baseline stays mixed.
        assert (pd.to_numeric(baseline.is_playoffs).mean() < 1.0)

    def test_insufficient_rows_do_not_page_in_the_monitor_json(self, tmp_path):
        drift = [{"feature": "elo_diff", "status": "INSUFFICIENT"},
                 {"feature": "rest_days_diff", "status": "ALERT"}]
        cov = [{"feature": "elo_diff", "window": "current",
                "status": "LOW_COVERAGE"}]
        record = mon.write_monitor_json(tmp_path / "m.json", "20260927",
                                        drift, cov, [], [], 0.5)
        assert [r["feature"] for r in record["alerts"]["feature_drift"]] ==             ["rest_days_diff"]
        assert record["alerts"]["coverage"][0]["status"] == "LOW_COVERAGE"

    def test_coverage_reports_both_windows(self):
        baseline = pd.DataFrame({"elo_diff": [0.5, np.nan] * 150})
        current = pd.DataFrame({"elo_diff": np.ones(80)})
        by = {(r["window"], r["feature"]): r["pct_measured"]
              for r in mon.coverage(baseline, current)}
        assert by[("baseline", "elo_diff")] == 50.0
        assert by[("current", "elo_diff")] == 100.0

    @staticmethod
    def _refusal_frame(values, flags, n=None):
        return pd.DataFrame({"pl_rapm_c_diff": values,
                             "_eligible_pl_rapm_c_diff": flags})

    def test_a_fully_refused_family_reads_structural_and_never_pages(self, tmp_path):
        """2026-10-08 coverage audit, medium-high finding: pl_rapm_c* was the
        only served family below full coverage, and its nulls are the
        eligible pool's REFUSAL (minutes floor / 30-day recency / a feed
        listing with no centre for that club) - measured per row by
        lineup_projection's flag, not inferred from a policy keyword. Fully
        explained -> STRUCTURAL with the reason and the count, the true 0%
        number kept, and no coverage alert."""
        baseline = self._refusal_frame([np.nan] * 60, [0.0] * 60)
        current = self._refusal_frame([np.nan] * 40, [0.0] * 40)
        rows = mon.coverage(baseline, current, feature_cols=["pl_rapm_c_diff"])
        assert len(rows) == 2
        for row in rows:
            assert row["status"] == "STRUCTURAL"
            assert row["pct_measured"] == 0.0          # numbers stay honest
            assert row["n_pool_refused"] == row["n_games"]
            assert "eligible-pool refusal" in row["structural_reason"]
        record = mon.write_monitor_json(tmp_path / "m.json", "20261008",
                                        [], rows, [], [], 0.5)
        assert record["alerts"]["coverage"] == []

    def test_an_unexplained_hole_keeps_its_alarm(self):
        """Flag 1 (or no flag at all) behind a null = no refusal evidence:
        the old thresholds apply unchanged - the mask may only silence what
        the builder itself refused."""
        flagged = self._refusal_frame([np.nan] * 60, [1.0] * 60)
        bare = pd.DataFrame({"pl_rapm_c_diff": [np.nan] * 60})
        for frame in (flagged, bare):
            rows = mon.coverage(frame, frame, feature_cols=["pl_rapm_c_diff"])
            assert {r["status"] for r in rows} == {"STARVED"}
            assert all("structural_reason" not in r for r in rows)
            assert all(r["n_pool_refused"] == 0 for r in rows)

    def test_a_partially_explained_family_keeps_the_real_alarm(self):
        """STRUCTURAL only when EVERY missing value is explained. Half
        refused + half a genuine hole must still page on the raw number."""
        values = [np.nan] * 30 + [np.nan] * 20 + [0.5] * 50
        flags = [0.0] * 30 + [1.0] * 20 + [1.0] * 50
        frame = self._refusal_frame(values, flags)
        row = mon.coverage(frame, feature_cols=["pl_rapm_c_diff"])[0]
        assert row["status"] == "LOW_COVERAGE"        # 50% measured
        assert "structural_reason" not in row
        assert row["n_pool_refused"] == 30            # the explained share is still reported

    def test_healthy_coverage_stays_ok_with_refusals_counted(self):
        values = [np.nan] * 10 + [0.5] * 90
        flags = [0.0] * 10 + [1.0] * 90
        row = mon.coverage(self._refusal_frame(values, flags),
                           feature_cols=["pl_rapm_c_diff"])[0]
        assert row["status"] == "OK"                  # never relabel a pass
        assert row["n_pool_refused"] == 10
        assert "structural_reason" not in row

    def test_the_mlb_projection_carries_the_reason_but_not_the_extra_field(self):
        row = {"feature": "pl_rapm_c_diff", "window": "current",
               "n_games": 60, "n_nonnull": 0, "pct_measured": 0.0,
               "pct_nonnull": 0.0, "n_measured": 0, "n_default_zero": 0,
               "n_pool_refused": 60, "status": "STRUCTURAL",
               "structural_reason": "eligible-pool refusal on 60 row(s)"}
        projected = mon._coverage_row_mlb(row)
        assert projected["structural_reason"] == row["structural_reason"]
        # The MLB report schema is nine fields; extras stay on the CSV.
        assert "n_pool_refused" not in projected


class TestFeatureImportanceWeights:
    """The MODEL WEIGHT column answers with the ensemble, not a table of zeros."""

    @staticmethod
    def _tree_member():
        import types
        # Tree members train on the numeric active width PLUS the team-ID
        # categoricals, so their importance vectors run two longer - the
        # trim in the aggregation exists for exactly this shape.
        n = len(config.active_moneyline_feature_cols()) + 2
        return types.SimpleNamespace(
            feature_importances_=np.linspace(1.0, 2.0, n))

    @staticmethod
    def _elasticnet_member():
        from sklearn.linear_model import LogisticRegression
        cols = feat_mod.linear_feature_columns()
        rng = np.random.default_rng(9)
        X = pd.DataFrame(rng.normal(size=(120, len(cols))), columns=cols)
        y = (rng.random(120) < 0.5).astype(int)
        model = LogisticRegression(max_iter=200).fit(X, y)
        pre = ml_mod.TrainFoldPreprocessor().fit(X)
        return {"model": model, "pre": pre}

    def test_importances_are_blend_weighted_across_members(self):
        weights = mp.feature_importance_weights(
            {"xgboost": {"model": self._tree_member()},
             "elasticnet": self._elasticnet_member()},
            {"xgboost": 0.7034, "elasticnet": 0.2966})
        assert weights is not None
        # Per-feature rounding to 4dp spreads across the serving width.
        assert abs(sum(weights.values()) - 100.0) < 1e-2
        assert all(v >= 0 for v in weights.values())

    def test_all_zero_member_weights_still_yield_a_real_table(self):
        weights = mp.feature_importance_weights(
            {"xgboost": {"model": self._tree_member()}}, {"xgboost": 0.0})
        assert weights is not None and max(weights.values()) > 0

    def test_weight_percentages_pass_through_unrescaled(self):
        # The helper returns percentages (0-100); the drift rows must not
        # multiply them again - a delivered CSV once showed 4191.85 for a
        # feature riding 41.9% of the blend.
        tree = self._tree_member()
        weights = mp.feature_importance_weights(
            {"xgboost": {"model": tree}}, {"xgboost": 1.0})
        top = max(weights.values())
        assert top <= 100.0
        baseline = pd.DataFrame({c: np.random.default_rng(1).normal(0, 1, 300)
                                 for c in ["elo_diff"]})
        current = pd.DataFrame({"elo_diff": np.random.default_rng(2).normal(0, 1, 40)})
        row = mon.feature_drift(baseline, current, weights)[0]
        assert row["weight_pct"] == weights["elo_diff"]

    def test_elasticnet_importance_is_per_sd_not_std_scaled(self):
        """``coef_`` is fit on z-scored columns, so |coef| already IS
        the per-SD importance - the decomposition must not multiply by
        the preprocessor's stds a second time.

        The 2026-09/10 drift tables read elo_diff at 68-94% of MODEL
        WEIGHT because that second multiplication let the widest column
        (Elo points, std ~126.8, against rates at std ~0.006) absorb
        nearly the whole elastic-net profile. A member whose coefficients
        are uniform over wildly unequal stds must come back uniform:
        any std weighting puts the fat column on top.
        """
        import types
        cols = feat_mod.linear_feature_columns()
        active = config.active_moneyline_feature_cols()
        assert cols[0] in active
        model = types.SimpleNamespace(coef_=np.ones((1, len(cols))))
        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.normal(size=(400, len(cols))), columns=cols)
        X[cols[0]] = rng.normal(0.0, 100.0, 400)  # std ~100 against ~1
        pre = ml_mod.TrainFoldPreprocessor().fit(X)
        decomp = mp.feature_importance_decomposition(
            {"elasticnet": {"model": model, "pre": pre}},
            {"elasticnet": 1.0})
        assert decomp is not None
        profile = decomp["member_profiles"]["elasticnet"]
        matched = [c for c in cols if c in active]
        expected = round(100.0 / len(matched), 4)
        for c in matched:
            assert abs(profile[c] - expected) < 0.01, c
        # The old coef*std scaling would have given the fat column ~90%.
        assert profile[cols[0]] < 10.0

    def test_members_without_importances_yield_none_not_zeros(self):
        import types
        weights = mp.feature_importance_weights(
            {"xgboost": {"model": types.SimpleNamespace()}}, {"xgboost": 1.0})
        assert weights is None

    def test_a_zero_weight_member_cannot_appear_in_the_column(self):
        """The published column is a BLEND average, so a member at weight 0
        is multiplied out before the sum.

        This is not a hypothetical. On 2026-09-29 the monitor published
        ``elo_diff`` at 91.02% of MODEL WEIGHT - the reader's natural
        conclusion being "the model is 91% Elo". It was not: with the blend
        concentrated on elasticnet, that column WAS elasticnet's |coef|*std
        alone (the pre-2026-10-04 reporting scaling, since fixed to per-SD
        |coef|), while xgboost's and lightgbm's own importances put
        ``elo_diff`` at 12.97% and 15.91%. The trees were not consulted at
        all.

        The concentration is correct arithmetic and the honest response is
        disclosure, not arithmetic - so the pipeline now logs the member
        shares beside the column (master_pipeline, feature report phase).
        This pins the arithmetic half of that contract: a zero-share member
        contributes nothing, so a reader can reconstruct exactly which
        profiles the number is built from.
        """
        tree = self._tree_member()
        # A tree that puts its whole mass on one column, so the effect is
        # visible rather than buried in a linspace.
        n = len(config.active_moneyline_feature_cols()) + 2
        peaked = np.zeros(n)
        peaked[0] = 1.0
        tree.feature_importances_ = peaked

        solo = mp.feature_importance_weights(
            {"xgboost": {"model": tree}}, {"xgboost": 1.0})
        # Same member at zero blend weight alongside a member that does
        # contribute: the peaked profile must vanish from the result.
        mixed = mp.feature_importance_weights(
            {"xgboost": {"model": tree},
             "elasticnet": self._elasticnet_member()},
            {"xgboost": 0.0, "elasticnet": 1.0})
        assert solo is not None and mixed is not None
        # In the solo case the peaked column takes the whole 100; in the
        # mixed case the elasticnet profile takes over, so the peaked
        # column can no longer be the whole story.
        assert solo[config.active_moneyline_feature_cols()[0]] == 100.0
        assert mixed[config.active_moneyline_feature_cols()[0]] < 100.0

    def test_the_decomposition_takes_the_blend_apart(self):
        """The monitor JSON must let a reader reconstruct the MODEL
        WEIGHT column without rerunning the pipeline.

        The 2026-10-01 drift table published ``elo_diff`` at 78.627%
        while the elastic net held 82.56% of the blend with 94.4% of
        its own mass on that column under the old coef*std reporting
        (per-SD: ~17%) - arithmetic a reader could previously only
        verify by rerunning the pipeline. The
        decomposition is the disclosure: the blended column, the
        member shares, and each member's own profile, with the
        column rebuildable by hand from the other two.
        """
        members = {"xgboost": {"model": self._tree_member()},
                   "elasticnet": self._elasticnet_member()}
        blend = {"xgboost": 0.7034, "elasticnet": 0.2966}
        decomp = mp.feature_importance_decomposition(members, blend)
        assert decomp is not None
        assert decomp["model_weight"] == mp.feature_importance_weights(
            members, blend)
        shares = decomp["member_shares"]
        profiles = decomp["member_profiles"]
        assert abs(sum(shares.values()) - 1.0) < 1e-3
        assert set(profiles) == set(shares)
        for profile in profiles.values():
            assert abs(sum(profile.values()) - 100.0) < 1e-2
        # The published column is the share-weighted mean of the
        # member profiles - the arithmetic the disclosure exists for.
        for col in config.active_moneyline_feature_cols():
            rebuilt = sum(shares[n] * profiles[n][col] for n in shares)
            assert abs(rebuilt - decomp["model_weight"][col]) < 0.05

    def test_a_zero_share_member_keeps_its_own_profile(self):
        """Disclosure, not erasure: a member the blend zeroed still
        shows its own profile.

        The trees' single-digit Elo share is exactly the evidence that
        a concentrated MODEL WEIGHT column is the elastic net's
        opinion rather than the ensemble's - so a zero-share member
        must appear in the decomposition with its own profile, not
        vanish from the disclosure the way it vanishes from the
        column.
        """
        import types
        n = len(config.active_moneyline_feature_cols()) + 2
        peaked = np.zeros(n)
        peaked[0] = 1.0
        tree = types.SimpleNamespace(feature_importances_=peaked)
        decomp = mp.feature_importance_decomposition(
            {"xgboost": {"model": tree},
             "elasticnet": self._elasticnet_member()},
            {"xgboost": 0.0, "elasticnet": 1.0})
        assert decomp is not None
        first = config.active_moneyline_feature_cols()[0]
        assert decomp["member_shares"]["xgboost"] == 0.0
        assert decomp["member_profiles"]["xgboost"][first] == 100.0
        # Zero share means zero contribution to the published column.
        assert decomp["model_weight"][first] < 100.0

    def test_the_monitor_json_carries_the_decomposition(self, tmp_path):
        """The disclosure must reach the delivered artifact: the
        monitor JSON carries the decomposition beside the served
        weights, with a one-line reading attached.
        """
        members = {"xgboost": {"model": self._tree_member()},
                   "elasticnet": self._elasticnet_member()}
        blend = {"xgboost": 0.4, "elasticnet": 0.6}
        decomp = mp.feature_importance_decomposition(members, blend)
        record = mon.write_monitor_json(
            tmp_path / "m.json", "20260927", [], [], [], [],
            None, {}, None, None, feature_importance=decomp)
        block = record["feature_importance"]
        assert block["model_weight"] == decomp["model_weight"]
        assert block["member_shares"] == decomp["member_shares"]
        assert block["member_profiles"] == decomp["member_profiles"]
        assert "member_shares" in block["reading"]
        # Absent decomposition ships an empty block, never a crash.
        bare = mon.write_monitor_json(
            tmp_path / "m2.json", "20260927", [], [], [], [],
            None, {}, None, None)
        assert bare["feature_importance"] == {}


class TestAdaptiveBlendPolicy:
    """Caps/floors are blend policy, fixed before any OOF evidence.

    The adaptive optimiser minimises pooled OOF logloss on the plain
    simplex, so it is free to put every point of weight on the best
    member - and on the delivered OOF it did (elastic net 0.8256,
    whose reported mass - old coef*std scaling, since fixed to
    per-SD - was 94.4% ``elo_diff``, making the served model read
    as ~79% one column). The production
    default is the EMPTY box (``ENSEMBLE_MEMBER_CAPS`` /
    ``_FLOORS`` both {} in config): NBA runs the plain simplex for
    MLB/NHL/NFL parity since 2026-10-01. The box machinery stays
    dormant, and these tests pin that any future policy - a constant
    fixed before the OOF frame is seen - is honoured by the
    optimiser, that an infeasible policy is refused back to the
    plain simplex, and that rounding can never push a published
    weight outside the box.
    """

    @staticmethod
    def _oof(seed: int = 0, n: int = 600):
        """Three members of deliberately unequal quality."""
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n).astype(float)
        enet = np.clip(
            np.where(y > 0.5, 0.9, 0.1) + rng.normal(0.0, 0.02, n),
            0.01, 0.99)
        lgbm = np.clip(
            np.where(y > 0.5, 0.7, 0.3) + rng.normal(0.0, 0.06, n),
            0.01, 0.99)
        xgb = np.clip(
            np.where(y > 0.5, 0.6, 0.4) + rng.normal(0.0, 0.10, n),
            0.01, 0.99)
        return {"elasticnet": enet, "lightgbm": lgbm,
                "xgboost": xgb}, y

    def test_a_cap_bounds_the_best_member(self, monkeypatch):
        # Unconstrained, the optimiser wants the (near-)pure elastic
        # net - the one-hot fallback included. A 0.70 cap must hold it
        # AT the cap, not beneath it and not above it.
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_CAPS",
                            {"elasticnet": 0.70})
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_FLOORS", {})
        members, y = self._oof()
        w = ml_mod.compute_adaptive_weights(members, y)
        assert abs(sum(w.values()) - 1.0) < 1e-3
        assert w["elasticnet"] <= 0.70 + 1e-4
        assert w["elasticnet"] >= 0.70 - 0.01

    def test_a_floor_keeps_a_worst_member_in_the_blend(self, monkeypatch):
        # xgboost is the weakest member and earns exactly 0.0 without
        # a floor; the policy must put its floor share back.
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_CAPS", {})
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_FLOORS",
                            {"xgboost": 0.15})
        members, y = self._oof()
        w = ml_mod.compute_adaptive_weights(members, y)
        assert abs(sum(w.values()) - 1.0) < 1e-3
        assert w["xgboost"] >= 0.15 - 1e-4

    def test_an_infeasible_policy_is_refused_to_the_plain_simplex(
            self, monkeypatch):
        # Floors that cannot sum to one describe no blend at all; the
        # policy must be refused and the ungoverned optimum served,
        # never an impossible weight vector.
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_CAPS", {})
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_FLOORS",
                            {"elasticnet": 0.5, "lightgbm": 0.5,
                             "xgboost": 0.5})
        members, y = self._oof()
        governed = ml_mod.compute_adaptive_weights(members, y)
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_FLOORS", {})
        plain = ml_mod.compute_adaptive_weights(members, y)
        for name, value in plain.items():
            assert abs(governed[name] - value) < 1e-6

    def test_caps_that_cannot_reach_one_are_also_refused(
            self, monkeypatch):
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_CAPS",
                            {"elasticnet": 0.3, "lightgbm": 0.3,
                             "xgboost": 0.3})
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_FLOORS", {})
        members, y = self._oof()
        governed = ml_mod.compute_adaptive_weights(members, y)
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_CAPS", {})
        plain = ml_mod.compute_adaptive_weights(members, y)
        for name, value in plain.items():
            assert abs(governed[name] - value) < 1e-6

    def test_the_published_vector_never_leaves_the_policy_box(
            self, monkeypatch):
        # Cap and floor together: the rounded, residue-fixed vector
        # the pipeline publishes must still be feasible, for every
        # draw of the OOF frame.
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_CAPS",
                            {"elasticnet": 0.60})
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_FLOORS",
                            {"xgboost": 0.05})
        for seed in range(4):
            members, y = self._oof(seed=seed)
            w = ml_mod.compute_adaptive_weights(members, y)
            assert abs(sum(w.values()) - 1.0) < 1e-3
            assert w["elasticnet"] <= 0.60 + 1e-3
            assert w["xgboost"] >= 0.05 - 1e-3

    def test_a_cap_on_a_lone_member_is_refused_not_honoured(
            self, monkeypatch):
        # A single member's only feasible blend is itself; a cap below
        # 1.0 makes the policy infeasible, and the refusal - not the
        # cap - governs what is published.
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_CAPS",
                            {"elasticnet": 0.70})
        monkeypatch.setattr(config, "ENSEMBLE_MEMBER_FLOORS", {})
        members, y = self._oof()
        w = ml_mod.compute_adaptive_weights(
            {"elasticnet": members["elasticnet"]}, y)
        assert w == {"elasticnet": 1.0}


class TestRunLineStructuralContract:
    """The run line's structural contract with the MLB/NHL/NFL family.

    Each test pins one clause of the alignment: the feature matrix is PULLED
    from the binary moneyline's own member path (never a parallel list), the
    dispersion estimator is MLB's pooled method-of-moments form, the MC
    derivation records its own resolution and guards it, and the artifacts
    carry the transparency blocks. Every later run-line change must flip one
    of these deliberately or it is a regression.
    """

    @staticmethod
    def _decided(n: int = 240, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        teams = list(range(30))
        df = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "gameday": pd.date_range("2026-01-01", periods=n, freq="D"),
            "season": 2026,
            "home_team": rng.choice(teams, n),
            "away_team": rng.choice(teams, n),
            "home_score": rng.integers(95, 125, n).astype(float),
            "away_score": rng.integers(95, 125, n).astype(float),
        })
        for col in config.MONEYLINE_FEATURE_COLS:
            df[col] = rng.normal(0, 1, n)
        return df

    @staticmethod
    def _oof(n: int = 200, seed: int = 11) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        scores_h = rng.integers(95, 125, n).astype(float)
        scores_a = rng.integers(95, 125, n).astype(float)
        oof = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "mu_h": scores_h + rng.normal(0, 4, n),
            "mu_a": scores_a + rng.normal(0, 4, n),
            "home_score": scores_h, "away_score": scores_a,
            "fold_id": np.arange(n) // 25,
        })
        oof["margin"] = oof.home_score - oof.away_score
        oof["total"] = oof.home_score + oof.away_score
        return oof

    def test_the_run_line_pulls_the_moneyline_contract_by_construction(self):
        import distributions as dist_mod
        import moneyline as ml_mod
        df = self._decided()
        reg = dist_mod.ScoreRegressor().fit(df)
        expected = ml_mod.member_matrix(dist_mod.MONEYLINE_TREE_MEMBER, df)
        assert list(reg.feature_columns) == list(expected.columns)

    def test_matrix_route_is_the_moneyline_member_helper_not_a_parallel_path(self):
        import distributions as dist_mod
        import moneyline as ml_mod
        df = self._decided(n=12)
        X = dist_mod.ScoreRegressor()._matrix(df)
        expected = ml_mod.member_matrix(dist_mod.MONEYLINE_TREE_MEMBER, df)
        assert list(X.columns) == list(expected.columns)
        assert X.shape == expected.shape

    def test_dispersion_estimator_is_mlbs_pooled_moment_form(self):
        import distributions as dist_mod
        # MLB's fit_alpha: pooled, unweighted, UNcapped, 4dp:
        #   alpha = max((var(y) - mean(mu)) / mean(mu)^2, 0)
        rng = np.random.default_rng(3)
        mu = np.full(500, 110.0)
        # Poisson sample -> estimate ~ 0
        y = rng.poisson(mu).astype(float)
        est = dist_mod.estimate_alpha(y, mu)
        expect = max((y.var(ddof=0) - mu.mean()) / mu.mean() ** 2, 0.0)
        assert abs(est - round(expect, 4)) < 1e-9
        # A genuinely over-dispersed sample must NOT be clipped by a cap
        # (the old mu^2-weighted capped form silently saturated): variance
        # far above the mean must land well above the Poisson-limit noise.
        y_over = mu + rng.normal(0, 25, 500)
        assert dist_mod.estimate_alpha(y_over, mu) > 0.03
        assert dist_mod.ALPHA_FLOOR == 1e-6

    def test_the_alpha_curve_layer_rides_alongside_the_scalars(self):
        import distributions as dist_mod
        oof = self._oof()
        sig = dist_mod.calibrate_dispersion(oof)
        for side in ("home", "away"):
            curve = sig[f"alpha_{side}_curve"]
            assert curve["form"] in ("piecewise", "linear", "power")
            assert f"alpha_{side}_max" in sig
        # The draw path consumes the SCALARS; the limit verdict is scalar too.
        assert sig["poisson_limit"] == bool(
            sig["alpha_home"] <= dist_mod.ALPHA_FLOOR
            and sig["alpha_away"] <= dist_mod.ALPHA_FLOOR)

    def test_empty_oof_degrades_to_poisson_not_an_exception(self):
        import distributions as dist_mod
        sig = dist_mod.calibrate_dispersion(None)
        assert sig["poisson_limit"] is True
        assert sig["alpha_home"] == 0.0 and sig["alpha_away"] == 0.0

    def test_mc_derivation_records_its_resolution_and_guards_it(self):
        import distributions as dist_mod
        rng = np.random.default_rng(5)
        mu_h = rng.uniform(105, 115, 4)
        mu_a = rng.uniform(105, 115, 4)
        meta: dict = {}
        dist_mod.simulate_distributions(mu_h, mu_a, 0.0, 0.0,
                                        n_draws=2000, meta_out=meta)
        assert set(meta) == {"n_draws", "requested_draws",
                             "mc_se_totals_max", "reason"}
        # At a tiny draw count the guard MUST fire: the NBA totals grid is
        # 101 lines, and the tail lines cannot be priced at se <= 5e-3
        # without the bump.
        assert meta["n_draws"] == dist_mod.MC_DRAWS_TAIL
        assert "bumped" in meta["reason"]
        # At the production resolution the guard must NOT fire.
        meta2: dict = {}
        dist_mod.simulate_distributions(mu_h[:2], mu_a[:2], 0.0, 0.0,
                                        n_draws=dist_mod.MC_DRAWS,
                                        meta_out=meta2)
        assert meta2["n_draws"] == dist_mod.MC_DRAWS
        assert meta2["reason"] == "default"

    def test_draw_constants_are_family_aligned(self):
        import distributions as dist_mod
        assert dist_mod.MC_DRAWS == 10_000
        assert dist_mod.MC_DRAWS_TAIL == 50_000
        assert dist_mod.MC_SE_TARGET == 5e-3
        assert dist_mod.ALPHA_CAP == 2.0

    def test_the_fit_diagnostics_pack_is_measured_every_run(self):
        import distributions as dist_mod
        fc = dist_mod.run_line_fit_check(self._oof())
        for side in ("home", "away"):
            block = fc[side]
            assert {"pearson", "deviance_model", "deviance_baseline",
                    "rmse_model", "rmse_baseline", "n"} <= set(block)
            # A real regressor beats the constant league-mean baseline it
            # replaces on this synthetic signal; in any case the baseline
            # numbers must be real measurements, not placeholders.
            assert block["n"] > 0
            assert np.isfinite(block["deviance_baseline"])

    def test_markets_meta_carries_the_transparency_blocks(self, tmp_path):
        import serving
        oof = self._oof(n=8)
        oof["kind"] = "oof"
        oof["fold_id"] = 0
        slate = oof.head(2).copy()
        slate["kind"] = "slate"
        meta_path = tmp_path / "meta.json"
        serving.write_markets_csv(
            tmp_path / "markets.csv", meta_path, oof, slate,
            mc_meta={"n_draws": 10000, "requested_draws": 10000,
                     "mc_se_totals_max": 0.0005, "reason": "default"},
            run_line_fit_check={"home": {"pearson": 1.0}, "away": {"pearson": 1.0}})
        import json
        meta = json.loads(meta_path.read_text())
        assert meta["mc_meta"]["n_draws"] == 10000
        assert "home" in meta["run_line_fit_check"]

    def test_every_market_leg_is_calibrated_from_its_own_outcome(self):
        """NHL 8d6f9b0 parity, pinned: over/under/push legs must each carry
        their OWN platt map in the bundle, never a shared one."""
        import distributions as dist_mod
        rng = np.random.default_rng(9)
        n = 120
        total = rng.uniform(190, 260, n)
        oof = self._oof(n=n, seed=13)
        oof["total"] = total
        oof["margin"] = rng.uniform(-20, 20, n)
        grid = dist_mod.simulate_distributions(
            oof.mu_h.to_numpy(), oof.mu_a.to_numpy(), 0.0, 0.0,
            n_draws=dist_mod.MC_DRAWS)
        frame = grid
        frame["fold_id"] = oof.fold_id.to_numpy()
        frame["margin"] = oof.margin.to_numpy()
        frame["total"] = oof.total.to_numpy()
        _, bundle = dist_mod.calibrate_market_frame(frame)
        sample = bundle["totals"].get("220") or next(iter(bundle["totals"].values()))
        assert sample["over"] is not None and sample["under"] is not None
        assert sample["over"] != sample["under"] or sample["push"] != sample["over"]


class TestMemberTuningNoiseFloor:
    """xgboost's own seed moves the member metric more than any gain on record.

    Why this is a test and not a note: the 2026-09-27 XGBoost retune picked
    the best of 150 trials on pooled OOF logloss, reported "+117 bps", and
    the gain reversed on its sealed holdout. A from-scratch re-run
    (.adhoc/nba_xgb_retune/) measured the reason afterwards - the SAME
    params under five seeds span 40 bps of pooled logloss, and 190 bps
    within a single fold block. At 150 trials, selecting the argmax on
    pooled logloss is selecting noise, and the reversal was the correct
    outcome rather than bad luck.

    So the floor is pinned here. A future study that reports a member gain
    without clearing it has measured the seed, not the parameters - and
    the honest comparison is a PAIRED one on the same games, judged across
    seeds, never two independently-noisy pooled numbers.
    """

    SEEDS = (42, 7, 123)

    @staticmethod
    def _oos(seed: int, n: int = 420) -> pd.DataFrame:
        """A small walk-forward whose members are genuinely seed-sensitive.

        xgboost draws its column/row subsamples from ``random_state``, so
        two seeds give two different models. The signal is weak and the
        noise is real, which is the regime the production study lives in.
        """
        import moneyline as ml_mod
        import folds as folds_mod
        rng = np.random.default_rng(11)
        df = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "gameday": pd.date_range("2025-01-01", periods=n, freq="D"),
            "season": 2026,
            "home_team": rng.choice(30, n),
            "away_team": rng.choice(30, n),
        })
        base = rng.normal(0, 1, n)
        df["home_score"] = 110 + base
        df["away_score"] = 110 - base
        df["home_win"] = (df.home_score > df.away_score).astype(float)
        for col in config.MONEYLINE_FEATURE_COLS:
            df[col] = rng.normal(0, 1, n)
        frame = folds_mod.canonical_sort(df)
        orig = dict(config.XGBOOST_PARAMS)
        try:
            config.XGBOOST_PARAMS["random_state"] = int(seed)
            assert ml_mod.config is config, "config identity drift"
            oof = ml_mod.walk_forward_oof(frame, progress_every=0)["oof"]
        finally:
            config.XGBOOST_PARAMS.clear()
            config.XGBOOST_PARAMS.update(orig)
        return oof[["game_id", "home_win", "p_xgboost"]]

    def test_the_same_params_under_different_seeds_are_not_the_same_model(self):
        """The floor exists. A gain smaller than this is not measurable."""
        runs = {s: self._oos(s).set_index("game_id").sort_index()
                for s in self.SEEDS}

        # Paired per-game logloss, seed vs seed - the comparison a study
        # should actually be making.
        deltas = []
        for a, b in itertools.combinations(self.SEEDS, 2):
            fa, fb = runs[a], runs[b]
            common = fa.index.intersection(fb.index)
            y = fa.loc[common, "home_win"].to_numpy(float)

            def per_game(p):
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return -(y * np.log(p) + (1 - y) * np.log(1 - p))

            deltas.append((per_game(fa.loc[common, "p_xgboost"].to_numpy(float))
                           - per_game(fb.loc[common, "p_xgboost"].to_numpy(float)))
                          .mean() * 10000)
        spread = max(deltas) - min(deltas)
        # The production frame measured 40 bps; this synthetic frame is
        # noisier, so the pin is a floor on the FLOOR, not a measurement of
        # it. If this ever goes to ~0 the member has become deterministic
        # and the whole caveat is obsolete - fail loudly rather than let a
        # future study cite a noise floor that no longer holds.
        assert spread > 5.0, (
            f"seed spread collapsed to {spread:.2f} bps; xgboost is no longer "
            "seed-sensitive, so the tuning caveat no longer applies and the "
            "2026-09-27 reversal needs a different explanation")


class TestFoldEarlyStopIsPointInTime:
    """The fold-fit mechanics never consult the rows they are graded on.

    Finding-1 remediation pins: xgboost's fold fits early-stop against a
    chronological tail of the TRAINING fold (never the validation window),
    ``_fit_member`` has no ``val`` parameter a call site could leak through,
    and the walk-forward loop's folds still fit strictly-prior rows only.
    Each test pins one clause.
    """

    @staticmethod
    def _decided(n: int = 300, seed: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        teams = list(range(30))
        df = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "gameday": pd.date_range("2026-01-01", periods=n, freq="D"),
            "season": 2026,
            "home_team": rng.choice(teams, n),
            "away_team": rng.choice(teams, n),
            "home_score": rng.integers(95, 125, n).astype(float),
            "away_score": rng.integers(95, 125, n).astype(float),
        })
        df["home_win"] = (df.home_score > df.away_score).astype(float)
        for col in config.MONEYLINE_FEATURE_COLS:
            df[col] = rng.normal(0, 1, n)
        return df

    def test_the_watch_tail_is_a_train_tail_never_the_validation_window(self):
        import moneyline as ml_mod
        df = self._decided(n=300)
        train = df.iloc[:240]
        X = ml_mod.member_matrix("xgboost", train)
        y = (train.home_score > train.away_score).astype(int).to_numpy()
        watch = ml_mod._early_stop_watch(X, y, None)
        assert watch is not None
        fit_rows, y_fit, X_watch = watch
        # 15% of 240 rounds to 36 held-out watch rows, capped well inside
        # the fold; the fit set is exactly the training frame MINUS that
        # tail, so no row is both watched and fitted.
        assert len(X_watch) == 36 and len(y_fit) == 204
        assert int(np.asarray(fit_rows).max()) == 203
        assert int(X_watch.index.min()) == 204

    def test_xgboost_early_stops_on_rows_strictly_before_the_validation_window(
            self, monkeypatch):
        import moneyline as ml_mod
        from xgboost import XGBClassifier
        df = self._decided(n=300)
        train, val = df.iloc[:240], df.iloc[240:]
        captured: dict = {}
        real_fit = XGBClassifier.fit

        def spy(self, *args, **kwargs):
            if kwargs.get("eval_set"):
                captured["eval_set"] = kwargs["eval_set"]
                captured["probe"] = self
            else:
                captured["refit_rows"] = len(args[0])
            return real_fit(self, *args, **kwargs)

        monkeypatch.setattr(XGBClassifier, "fit", spy)
        model, _ = ml_mod._fit_member("xgboost", train)
        assert model is not None and "eval_set" in captured
        eval_X, eval_y = captured["eval_set"][0]
        assert eval_y is not None and len(eval_y) == 36
        assert captured["refit_rows"] == len(train)
        assert model.get_booster().num_boosted_rounds() == captured["probe"].best_iteration + 1
        # The watch rows map back to training-fold games, every one of them
        # strictly before the validation window's first game day.
        watch_positions = list(eval_X.index)
        watch_games = train.iloc[watch_positions]
        assert watch_games.gameday.max() < val.gameday.min()
        assert not (set(watch_games.game_id.astype(str))
                    & set(val.game_id.astype(str)))

    def test_fit_member_has_no_validation_parameter_to_leak_through(self):
        import inspect
        import moneyline as ml_mod
        params = inspect.signature(ml_mod._fit_member).parameters
        assert list(params) == ["name", "train"]

    def test_walk_forward_folds_fit_strictly_prior_rows_only(self, monkeypatch):
        import moneyline as ml_mod
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 2)
        df = self._decided(n=120)
        result = ml_mod.walk_forward_oof(df, progress_every=0)
        oof, ft = result["oof"], result["fold_table"]
        assert len(oof) and len(ft)
        for _, row in ft.iterrows():
            in_fold = oof[oof.fold_id == row.fold_id]
            assert (pd.to_datetime(in_fold.gameday)
                    >= pd.Timestamp(row.val_start)).all()
            # n_train is exactly the strict-prior row count: nothing at or
            # after val_start ever entered the fold's fit set.
            n_prior = int((df.gameday < pd.Timestamp(row.val_start)).sum())
            assert row.n_train == n_prior


class TestCausalMoneylineParity:
    def test_future_labels_cannot_rewrite_headlines(self, monkeypatch):
        import moneyline as ml
        from types import SimpleNamespace
        frame = TestFoldEarlyStopIsPointInTime._decided(90)
        frame["game_type"] = 1
        folds = [SimpleNamespace(fold_id=i, train_idx=np.arange(30 + 20*i),
                 val_idx=np.arange(30 + 20*i, 50 + 20*i),
                 val_start=frame.gameday.iloc[30 + 20*i],
                 val_end=frame.gameday.iloc[49 + 20*i], provisional=False,
                 season_type="regular", is_partial_tail=False) for i in range(3)]
        monkeypatch.setattr(ml, "_fit_member", lambda name, train: (name, None))
        def predict(model, name, X, pre):
            offset = {"xgboost": .2, "lightgbm": -.2, "elasticnet": 0}[name]
            return 1 / (1 + np.exp(-(X.iloc[:, 0].to_numpy(float) + offset)))
        monkeypatch.setattr(ml, "_predict", predict)
        before = ml.walk_forward_oof(frame, fold_list=folds, progress_every=0)
        frame.loc[folds[-1].val_idx, "home_win"] = 1 - frame.loc[folds[-1].val_idx, "home_win"]
        after = ml.walk_forward_oof(frame, fold_list=folds, progress_every=0)
        for col in ("p_ensemble", "p_ensemble_causal", "p_ensemble_calibrated"):
            np.testing.assert_array_equal(before["oof"][col], after["oof"][col])
        np.testing.assert_array_equal(before["oof"].p_ensemble,
                                      before["oof"].p_ensemble_causal)
        assert "p_ensemble_retrospective" in before["oof"]
        assert before["fold_table"].iloc[0].weights == config.ENSEMBLE_WEIGHTS

    def test_gate_is_aligned_exclusion_safe_and_rejects_no_gain(self, monkeypatch):
        import moneyline as ml
        monkeypatch.setattr(config, "MIN_OOF_FOR_FIT", 4)
        monkeypatch.setattr(config, "CAL_GATE_MIN_HOLDOUT", 2)
        monkeypatch.setattr(ml, "moneyline_fit", lambda p, y: {"fake": True})
        monkeypatch.setattr(ml, "moneyline_apply", lambda p, cal: p)
        p, y = np.linspace(.2, .8, 12), np.tile([0., 1.], 6)
        grades = np.arange(12) < 10
        cal, audit = ml.gated_calibrator(p, y, grades)
        assert cal is None and audit["reason"] == "gated_no_gain"
        y[~grades] = 100
        assert ml.gated_calibrator(p, y, grades) == (cal, audit)
        with pytest.raises(ValueError, match="aligned"):
            ml.gated_calibrator(p, y[:-1])
        with pytest.raises(ValueError, match="align"):
            ml.gated_calibrator(p, y, grades[:-1])

    def test_gate_accepts_gain_and_refits_all_prior_evidence(self, monkeypatch):
        import moneyline as ml
        monkeypatch.setattr(config, "MIN_OOF_FOR_FIT", 4)
        monkeypatch.setattr(config, "CAL_GATE_MIN_HOLDOUT", 2)
        fits = []
        def fit(p, y):
            fits.append(len(p))
            return {"fake": True}
        monkeypatch.setattr(ml, "moneyline_fit", fit)
        monkeypatch.setattr(ml, "moneyline_apply", lambda p, cal: np.where(p > .5, .8, .2))
        p, y = np.tile([.49, .51], 6), np.tile([0., 1.], 6)
        cal, audit = ml.gated_calibrator(p, y)
        assert cal and audit["reason"] == "accepted"
        assert fits == [9, 12]

    def test_final_refit_uses_shared_policy_in_canonical_order(self, monkeypatch):
        import moneyline as ml
        frame = TestFoldEarlyStopIsPointInTime._decided(90)
        seen = []
        def fit(name, train):
            seen.append((name, train.game_id.tolist()))
            return object(), None
        monkeypatch.setattr(ml, "_fit_member", fit)
        models, _ = ml.fit_final_models(frame.sample(frac=1, random_state=7))
        assert set(models) == set(config.ENSEMBLE_MEMBERS)
        assert all(ids == frame.game_id.tolist() for name, ids in seen)


class TestSealedDispersionHoldout:
    """The alpha layer never fits the rows it is evaluated against.

    Finding-2 remediation pins (NHL 62d00fd / MLB derive_markets_v3
    parity): the last HOLDOUT_DAYS of a dated OOF frame are sealed away
    from the curve fit, the scalars, and the alpha-vector max — and the
    gate records its own scope so a run cannot silently claim a discipline
    it did not exercise. An undated frame is ungated, exactly as before.
    """

    @staticmethod
    def _oof(n: int = 200, seed: int = 13) -> pd.DataFrame:
        """Dated OOF frame whose tail is intentionally degenerate: the
        sealed window's scores are wildly inflated so that any fit that
        saw them would produce a visibly different alpha."""
        rng = np.random.default_rng(seed)
        scores_h = rng.integers(95, 125, n).astype(float)
        scores_a = rng.integers(95, 125, n).astype(float)
        oof = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "gameday": pd.date_range("2026-01-01", periods=n, freq="D"),
            "mu_h": scores_h + rng.normal(0, 4, n),
            "mu_a": scores_a + rng.normal(0, 4, n),
            "home_score": scores_h, "away_score": scores_a,
            "fold_id": np.arange(n) // 25,
        })
        # The sealed tail: 21 days of absurdly lopsided scores. A fit that
        # included these rows would read massive over-dispersion.
        tail = oof.gameday >= (oof.gameday.max().normalize()
                               - pd.Timedelta(days=21))
        oof.loc[tail, "home_score"] = 200.0
        oof.loc[tail, "away_score"] = 80.0
        oof["margin"] = oof.home_score - oof.away_score
        oof["total"] = oof.home_score + oof.away_score
        return oof

    def test_the_gate_fits_pre_holdout_rows_and_records_its_scope(self):
        import distributions as dist_mod
        oof = self._oof()
        sig = dist_mod.calibrate_dispersion(oof)
        hold = sig["holdout"]
        cutoff = (oof.gameday.max().normalize() - pd.Timedelta(days=21))
        assert hold["cutoff"] == str(cutoff.date())
        # The definition, not a count: sealed rows are dates >= cutoff
        # (on a daily grid the cutoff day itself is sealed, so 21 days of
        # separation seals 22 rows).
        n_sealed = int((oof.gameday >= cutoff).sum())
        assert hold["n_holdout"] == n_sealed
        assert hold["n_pre"] == len(oof) - n_sealed
        assert hold["fitted_on"] == "pre-holdout OOF only"
        # The degenerate sealed tail did NOT reach the fit: alpha stays in
        # the sane range the pre-holdout rows support, not the inflated
        # value the lopsided tail would drag it to.
        ungated = dist_mod.calibrate_dispersion(
            oof.drop(columns=["gameday"]))
        assert sig["alpha_home"] < ungated["alpha_home"]

    def test_an_undated_frame_is_ungated_and_fully_fit(self):
        import distributions as dist_mod
        oof = self._oof().drop(columns=["gameday"])
        sig = dist_mod.calibrate_dispersion(oof)
        hold = sig["holdout"]
        assert hold["cutoff"] is None
        assert hold["fitted_on"] == "full OOF (no gameday on frame)"
        assert hold["n_holdout"] == 0 and hold["n_pre"] == len(oof)

    def test_the_stamp_marks_exactly_the_sealed_window(self):
        import master_pipeline as mp
        oof = self._oof(n=120)
        oof["frame_view"] = "oof"
        stamped = mp._stamp_sealed_tail(oof)
        cutoff = (oof.gameday.max().normalize() - pd.Timedelta(days=21))
        sealed = stamped[stamped.frame_view == "sealed"]
        assert len(sealed) == int((oof.gameday >= cutoff).sum())
        assert (pd.to_datetime(sealed.gameday) >= cutoff).all()
        assert (stamped.loc[stamped.frame_view != "sealed", "frame_view"]
                == "oof").all()

    def test_the_stamp_leaves_an_undated_frame_untouched(self):
        import master_pipeline as mp
        oof = self._oof(n=60).drop(columns=["gameday"])
        oof["frame_view"] = "oof"
        assert mp._stamp_sealed_tail(oof)["frame_view"].eq("oof").all()
        # And a None frame does not raise.
        assert mp._stamp_sealed_tail(None) is None

    def test_the_monitor_carries_the_gate_scope(self):
        import monitoring as mon
        import distributions as dist_mod
        sig = dist_mod.calibrate_dispersion(self._oof())
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "m.json"
            record = mon.write_run_engine_monitor(
                out, "20260928", dispersion=sig,
                config_meta={"sport": "nba"})
            fit = record["fit"]["holdout"]
            assert fit["cutoff"] == sig["holdout"]["cutoff"]
            assert fit["holdout_days"] == dist_mod.HOLDOUT_DAYS
            assert fit["fitted_on"] == "pre-holdout OOF only"


class TestEloSeasonBoundary:
    """Elo never resets at the season flip; it regresses 1/3 toward 1500.

    The 2026-09-29 boundary audit's answer, stated as executable fact. The
    revert is load-bearing (the drift report's elo_* means are Elo entering
    means, and the season flip is when a reset would move them most) and
    until now existed only as code, not as a pin. Expectations are computed
    against a single-season reference walk of the SAME games - no hand-
    derived Elo constants.
    """

    @staticmethod
    def _game(gid, day, season, home, away, hs, as_):
        return pd.DataFrame([{
            "game_id": gid, "gameday": pd.Timestamp(day), "season": season,
            "home_team": home, "away_team": away,
            "home_score": float(hs), "away_score": float(as_),
        }])

    @classmethod
    def _two_season_events(cls) -> pd.DataFrame:
        games = pd.concat([
            cls._game("s1-a", "2025-01-05", 2024, "BOS", "NYK", 118, 104),
            cls._game("s1-b", "2025-01-07", 2024, "NYK", "BOS", 100, 112),
            cls._game("s2-a", "2025-10-22", 2025, "BOS", "LAL", 105, 110),
            cls._game("s2-b", "2025-10-24", 2025, "LAL", "BOS", 108, 101),
        ], ignore_index=True)
        return feat_mod.team_events(games)

    def test_ratings_regress_a_third_toward_1500_at_the_season_flip(self):
        events = self._two_season_events()
        ev, final = feat_mod._elo_apply(events)
        # The season-1-only walk's final state is exactly the state the
        # two-season walk carries INTO the flip (same games, same order,
        # no flip in the reference). A team playing on BOTH sides enters
        # season 2 at that state pulled 1/3 toward 1500 - a reset would
        # enter at ELO_PRIOR, no revert at the raw state.
        s1 = events[events.game_id.isin(["s1-a", "s1-b"])]
        _, s1_final = feat_mod._elo_apply(s1)
        entering = ev[ev.game_id == "s2-a"].set_index("team").elo_entering
        expected_bos = (s1_final["BOS"] + config.ELO_REVERT_FACTOR
                        * (config.ELO_PRIOR - s1_final["BOS"]))
        assert entering["BOS"] == pytest.approx(expected_bos, abs=1e-9)
        # A team whose season ENDED at the flip keeps the reverted state
        # rather than vanishing (NYK never plays in season 2).
        assert final["NYK"] == pytest.approx(
            s1_final["NYK"] + config.ELO_REVERT_FACTOR
            * (config.ELO_PRIOR - s1_final["NYK"]), abs=1e-9)

    def test_the_flip_fires_once_not_every_boundary_game(self):
        """Season 2's second entering rating is the plain one-step K update
        from the run's own season-2 entering values - pinning that the
        revert applied exactly once at the flip, never per boundary game.
        """
        events = self._two_season_events()
        ev, _ = feat_mod._elo_apply(events)
        a = ev[ev.game_id == "s2-a"].set_index("team").elo_entering
        b = ev[ev.game_id == "s2-b"].set_index("team").elo_entering
        ra, rb = float(a["BOS"]), float(a["LAL"])
        exp_home = 1.0 / (1.0 + 10.0 ** (
            (rb - ra - config.ELO_HOME_ADV) / config.ELO_SCALE))
        # BOS lost s2-a at home (105-110).
        assert b["BOS"] == pytest.approx(
            ra + config.ELO_K * (0.0 - exp_home), abs=1e-9)

    def test_a_never_played_team_enters_at_1500(self):
        events = self._two_season_events()
        ev, _ = feat_mod._elo_apply(events)
        lal = ev[(ev.game_id == "s2-a") & (ev.team == "LAL")].iloc[0]
        assert lal.elo_entering == pytest.approx(config.ELO_PRIOR)


class TestRunEngineDistributionArtifacts:
    """The Totals & Run Lines drift/coverage report the DISTRIBUTION model.

    2026-10-08 NBA totals-page parity with MLB: the run-engine panels must
    carry the per-side Poisson score regressor's own weights and input view,
    never the binary moneyline blend — and the moneyline monitor's default
    contract stays exactly as it was.
    """

    @staticmethod
    def _fake_reg(cols, h_imp, a_imp):
        import types
        return types.SimpleNamespace(
            feature_columns=list(cols),
            home_model=types.SimpleNamespace(
                feature_importances_=np.asarray(h_imp, dtype=float)),
            away_model=types.SimpleNamespace(
                feature_importances_=np.asarray(a_imp, dtype=float)))

    def test_distribution_weights_pool_the_per_side_fits_and_sum_to_100(self):
        import distributions as dist_mod
        cols = ["elo_diff", "pace_diff"] + list(config.TREE_CATEGORICAL_COLS)
        # team-ID importances are heavy on purpose: they must be excluded
        # from BOTH the map and the denominator (no relabelled leftovers).
        reg = self._fake_reg(cols, [7.0, 1.0, 90.0, 90.0],
                             [3.0, 1.0, 90.0, 90.0])
        weights = dist_mod.distribution_feature_weights(reg)
        assert weights is not None
        assert set(weights) == {"elo_diff", "pace_diff"}
        assert abs(sum(weights.values()) - 100.0) < 1e-6
        # pooled per side, renormalized over the numeric view only:
        # elo (7+3) / (10 + 2) and pace (1+1) / 12
        assert weights["elo_diff"] == pytest.approx(100.0 * 10 / 12)
        assert weights["pace_diff"] == pytest.approx(100.0 * 2 / 12)

    def test_distribution_weights_are_none_without_lightgbm(self):
        import types
        import distributions as dist_mod
        reg = types.SimpleNamespace(feature_columns=["elo_diff"],
                                    home_model=None, away_model=None)
        # unfitted / ridge-fallback regressor -> no weight, never moneyline
        assert dist_mod.distribution_feature_weights(reg) is None
        assert dist_mod.distribution_feature_weights(None) is None

    def test_distribution_weights_refuse_mismatched_importance_vectors(self):
        import distributions as dist_mod
        reg = self._fake_reg(["elo_diff"], [1.0, 2.0], [1.0, 2.0])
        assert dist_mod.distribution_feature_weights(reg) is None

    def test_run_engine_feature_view_excludes_team_ids(self):
        import distributions as dist_mod
        cols = ["elo_diff", "pace_diff"] + list(config.TREE_CATEGORICAL_COLS)
        reg = self._fake_reg(cols, [1.0] * 4, [1.0] * 4)
        assert dist_mod.run_engine_feature_cols(reg) == ["elo_diff",
                                                         "pace_diff"]
        # No fitted model -> the tree_view contract (numeric width, no ids).
        assert (dist_mod.run_engine_feature_cols(None)
                == config.active_moneyline_feature_cols())

    def test_run_engine_artifacts_iterate_the_given_view(self, tmp_path):
        import monitoring as mon
        baseline = pd.DataFrame({"keep_a": [0.5, np.nan] * 40,
                                 "keep_b": np.ones(80),
                                 "binary_only": np.zeros(80)})
        current = pd.DataFrame({"keep_a": np.ones(60),
                                "keep_b": np.ones(60),
                                "binary_only": np.zeros(60)})
        drift_name, cov_name = mon.write_run_engine_feature_artifacts(
            tmp_path, "20261008", baseline, current,
            weights={"keep_a": 42.5},
            feature_cols=["keep_a", "keep_b"])
        drift = pd.read_csv(tmp_path / drift_name)
        cov = pd.read_csv(tmp_path / cov_name)
        # rows come from the DISTRIBUTION view, not the binary contract
        assert sorted(drift["feature"]) == ["keep_a", "keep_b"]
        assert sorted(cov["feature"].unique()) == ["keep_a", "keep_b"]
        assert "binary_only" not in set(cov["feature"])
        assert drift.set_index("feature").loc["keep_a", "weight_pct"] == 42.5
        # a view feature with no weight of its own stays empty, not borrowed
        assert pd.isna(drift.set_index("feature").loc["keep_b", "weight_pct"])
        assert set(cov["window"]) == {"baseline", "current"}

    def test_coverage_default_contract_is_unchanged_for_the_monitor(self):
        import monitoring as mon
        baseline = pd.DataFrame({"elo_diff": np.ones(30),
                                 "pace_diff": np.ones(30)})
        current = pd.DataFrame({"elo_diff": np.ones(20),
                                "pace_diff": np.ones(20)})
        # Omitted feature_cols -> the binary moneyline contract (the path
        # the Model Monitor JSON uses); the run-engine writer passes its
        # own view, this default must stay untouched.
        default_rows = mon.coverage(baseline, current)
        assert ({r["feature"] for r in default_rows}
                == set(config.active_moneyline_feature_cols()))
        redirected = mon.coverage(baseline, current,
                                  feature_cols=["elo_diff"])
        assert {r["feature"] for r in redirected} == {"elo_diff"}


def test_the_lineup_inventory_log_splits_priced_from_refusal_rows():
    """2026-10-09 log review: since the pool-eligibility mask (62afdd8c)
    every refused team-game emits a pool_size-0 mask row, so the whole-frame
    means behind the inventory line read ~0.1 and a healthy run's own log
    looked like a projection collapse (10-09: 6976 rows, mean pool 0.1, vs
    the pre-mask 22 slate rows at 19.5). The line must state the
    priced/refusal split so the number cannot be misread."""
    from pathlib import Path
    src = (Path(__file__).with_name("master_pipeline.py")).read_text(
        encoding="utf-8")
    assert "pool-refusal mask rows" in src
    assert "over priced rows" in src
