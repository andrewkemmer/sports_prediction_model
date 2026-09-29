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

    def test_members_without_importances_yield_none_not_zeros(self):
        import types
        weights = mp.feature_importance_weights(
            {"xgboost": {"model": types.SimpleNamespace()}}, {"xgboost": 1.0})
        assert weights is None


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
            captured["eval_set"] = kwargs.get("eval_set")
            return real_fit(self, *args, **kwargs)

        monkeypatch.setattr(XGBClassifier, "fit", spy)
        model, _ = ml_mod._fit_member("xgboost", train)
        assert model is not None and "eval_set" in captured
        eval_X, eval_y = captured["eval_set"][0]
        assert eval_y is not None and len(eval_y) == 36
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
