"""Fold retention, season-type tagging and the grading split (2026-10-03).

The contract these pin down:

* ``make_folds`` RETAINS every observed window. ``MIN_VAL_FOLD_GAMES`` no
  longer discards one -- it marks it ``provisional``, which removes the
  window from pooled metrics, blend weights and calibration while still
  fitting and scoring it. Before this change the gate dropped 15 of 56
  candidate windows on the production frame, 13 of them postseason (~195
  playoff games), with no log line at all.
* ``Fold.season_type`` classifies a window by row-level ``game_type``
  (regular / postseason / mixed). It is metadata, never a rejection reason.
* Only regular-season rows from non-provisional windows enter ``pooled`` and
  therefore the SLSQP blend and the in-loop Platt map. Postseason and
  provisional rows are scored -- they land in ``oof`` and in
  ``*_predictions_history`` -- but they never grade the model.
* ``is_partial_tail`` no longer re-admits a sub-gate window. That exemption
  was the structural contradiction: a 3-game Finals window graded the model
  while a 34-game playoff window did not.
* ``postseason_flag`` must agree with the ``is_playoffs`` FEATURE the model
  is fed, or the gate and the model would disagree about what "playoff"
  means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from backend import config, folds as folds_mod, moneyline as ml_mod
except ImportError:  # pragma: no cover - direct script/import fallback
    import config
    import folds as folds_mod
    import moneyline as ml_mod


def _frame(spec, season: int = 2026) -> pd.DataFrame:
    """Build a decided frame from ``(date, n_games, game_type)`` triples."""
    rows = []
    for day, count, game_type in spec:
        for i in range(count):
            rows.append({
                "game_id": f"{day}-{i}",
                "gameday": pd.Timestamp(day),
                "season": season,
                "game_type": game_type,
                "home_team": i % 30,
                "away_team": (i + 1) % 30,
                "home_score": 100.0 + i,
                "away_score": 99.0 + i,
            })
    df = pd.DataFrame(rows)
    df["home_win"] = (df.home_score > df.away_score).astype(float)
    return df


def _week_series(start: str, days: int, per_day: int, game_type: int = 1):
    """``days`` consecutive observed dates, ``per_day`` games each."""
    dates = pd.date_range(start, periods=days, freq="D")
    return [(str(d.date()), per_day, game_type) for d in dates]


class TestRetainDontDrop:
    def test_sub_gate_windows_are_retained_as_provisional(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        # 7 dates x 1 game = 7 rows per window: far under the 40-game gate.
        df = folds_mod.canonical_sort(_frame(_week_series("2026-01-01", 60, 1)))
        folds = folds_mod.make_folds(df)
        assert folds, "windows must be retained even when they are thin"
        assert all(f.provisional for f in folds)
        assert all(not f.grades_pooled for f in folds)

    def test_windows_at_or_above_the_gate_grade(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        # 93 observed dates: index 30 onward is exactly 9 full 7-date windows,
        # so no window is partial and none falls under the gate.
        df = folds_mod.canonical_sort(_frame(_week_series("2026-01-01", 93, 8)))
        folds = folds_mod.make_folds(df)
        assert len(folds) == 9, [str(f.val_start.date()) for f in folds]
        assert not any(f.provisional for f in folds)
        assert all(f.grades_pooled for f in folds)

    def test_partial_tail_under_the_gate_is_no_longer_re_admitted(self,
                                                                  monkeypatch):
        """The gate's tail contradiction: exempt tails used to grade.

        55 observed dates from index 30 leaves a 4-date final window, which
        is both a partial tail and far under the gate. The legacy filter kept
        it anyway (``len(val) >= minimum or is_partial_tail``); the grading
        split must not.
        """
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        df = folds_mod.canonical_sort(_frame(_week_series("2026-01-01", 55, 8)))
        folds = folds_mod.make_folds(df)
        tail = [f for f in folds if f.is_partial_tail]
        assert tail, "the short final window must still exist as a fold"
        assert all(f.provisional for f in tail)
        assert not any(f.grades_pooled for f in tail)

    def test_fold_summary_publishes_coverage_instead_of_silence(self,
                                                                monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        df = folds_mod.canonical_sort(
            _frame(_week_series("2026-01-01", 60, 1)
                   + _week_series("2026-06-01", 60, 8)))
        diag: dict = {}
        folds = folds_mod.make_folds(df, diagnostics=diag)
        summary = folds_mod.fold_summary(folds)
        # Nothing is dropped any more, and that is asserted rather than
        # assumed: a regression back to silent dropping shows up here.
        assert summary["dropped_windows"] == 0
        assert summary["dropped_games"] == 0
        assert summary["provisional_windows"] == diag["provisional_windows"] > 0
        assert summary["provisional_games"] == diag["provisional_games"] > 0
        assert summary["n_candidates"] == len(folds)


class TestSeasonTypeClassification:
    def test_playoff_rows_land_in_exactly_one_fold(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        spec = (_week_series("2026-01-01", 60, 8, game_type=1)
                + _week_series("2026-04-01", 35, 4, game_type=2))
        df = folds_mod.canonical_sort(_frame(spec))
        folds = folds_mod.make_folds(df)

        post_total = int(folds_mod.postseason_flag(df.game_type).sum())
        assert post_total > 0
        seen: list = []
        for fold in folds:
            seen.extend(fold.val_idx.tolist())
        assert len(seen) == len(set(seen)), "validation windows must not overlap"
        post_scored = int(folds_mod.postseason_flag(
            df.game_type.iloc[seen]).sum())
        assert post_scored == post_total, "no playoff game may be dropped"

    def test_classifies_regular_postseason_and_mixed(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        # 41 regular dates then playoffs: the observed-date windows start at
        # index 30 and step 7, so window [37..43] straddles the seam at 41
        # and must classify as mixed rather than being forced one way.
        spec = (_week_series("2026-01-01", 41, 8, game_type=1)
                + _week_series("2026-02-15", 30, 8, game_type=2))
        df = folds_mod.canonical_sort(_frame(spec))
        folds = folds_mod.make_folds(df)
        kinds = {f.season_type for f in folds}
        assert kinds == {"regular", "postseason", "mixed"}, kinds

    def test_missing_game_type_column_keeps_fixture_behaviour(self,
                                                              monkeypatch):
        """Unit-test frames without ``game_type`` must classify as regular."""
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        df = _frame(_week_series("2026-01-01", 60, 8)).drop(
            columns=["game_type"])
        df = folds_mod.canonical_sort(df)
        folds = folds_mod.make_folds(df)
        assert folds
        assert {f.season_type for f in folds} == {"regular"}

    def test_fold_table_exposes_the_split(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        spec = (_week_series("2026-01-01", 60, 8, game_type=1)
                + _week_series("2026-04-01", 35, 4, game_type=2))
        df = folds_mod.canonical_sort(_frame(spec))
        table = folds_mod.fold_table(df, folds_mod.make_folds(df))
        for col in ("season_type", "provisional", "grades_pooled"):
            assert col in table.columns
        assert set(table.season_type) <= {"regular", "postseason", "mixed"}
        assert (table.grades_pooled == ~table.provisional).all()

    def test_postseason_flag_matches_the_is_playoffs_expression(self):
        """The gate and the model must agree on what a playoff row is.

        ``features.build_*`` derives ``is_playoffs`` from ``game_type`` with
        an OR over textual spellings (``features.py:472``); the fold
        classifier carries its own copy of that expression so a change to
        one cannot silently split the grading population from the feature
        the ensemble is fed. This pins the formula itself.
        """
        raw = pd.Series([1, 2, 3, "playoff", "POSTSEASON", "regular", None],
                        index=range(7))
        type_num = pd.to_numeric(raw, errors="coerce")
        type_text = raw.astype("string").str.lower()
        feature = ((type_num == config.GAME_TYPE_POST)
                   | type_text.str.contains("play|post", regex=True,
                                            na=False)).astype(float)
        assert folds_mod.postseason_flag(raw).astype(float).equals(feature)
        # Numeric codes: NBA uses GAME_TYPE_REG=1 / GAME_TYPE_POST=2.
        got = folds_mod.postseason_flag(pd.Series([1, 2, 1, 2])).tolist()
        assert got == [False, True, False, True]


class TestGradingSplit:
    @staticmethod
    def _graded_frame(regular_days: int = 150, playoff_days: int = 30,
                      per_day: int = 8, seed: int = 3) -> pd.DataFrame:
        """Regular + playoff series with windows large enough to grade.

        ``per_day`` must be >= 6 so a 7-date window clears MIN_VAL_FOLD_GAMES
        and the split is exercised on season type alone, not on size.
        """
        spec = (_week_series("2026-01-01", regular_days, per_day, game_type=1)
                + _week_series("2026-06-01", playoff_days, per_day,
                               game_type=2))
        rng = np.random.default_rng(seed)
        df = _frame(spec)
        df["home_score"] = rng.integers(95, 125, len(df)).astype(float)
        df["away_score"] = rng.integers(95, 125, len(df)).astype(float)
        df["home_win"] = (df.home_score > df.away_score).astype(float)
        for col in config.MONEYLINE_FEATURE_COLS:
            df[col] = rng.normal(0, 1, len(df))
        return df

    @staticmethod
    def _run(df: pd.DataFrame) -> dict:
        return ml_mod.walk_forward_oof(df, progress_every=0)

    def test_only_regular_non_provisional_rows_grade(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        result = self._run(self._graded_frame())
        oof = result["oof"]
        split = result["season_split"]
        assert len(oof)
        assert split["postseason_rows"] > 0, "playoff rows must be scored"
        assert split["grading_rows"] > 0, "regular rows must still grade"
        assert split["grading_rows"] < len(oof), \
            "playoff rows must not reach the optimizer"

        graded = oof[oof.grades_pooled.astype(bool)]
        assert len(graded) == split["grading_rows"]
        assert not graded.is_playoffs.astype(bool).any()
        assert not graded.provisional.astype(bool).any()
        # Every row is accounted for: graded, or explicitly marked otherwise.
        assert split["regular_rows"] + split["postseason_rows"] == len(oof)

    def test_postseason_rows_are_scored_but_never_grade(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        oof = self._run(self._graded_frame())["oof"]
        post = oof[oof.is_playoffs.astype(bool)]
        assert len(post) > 0
        assert not post.grades_pooled.astype(bool).any()

    def test_fold_table_records_the_grading_count(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        result = self._run(self._graded_frame())
        oof, table = result["oof"], result["fold_table"]
        assert len(table)
        for col in ("season_type", "provisional", "n_grading"):
            assert col in table.columns
        assert (table.n_grading <= table.n_val).all()
        # A provisional window contributes no grading rows by construction.
        assert table.loc[table.provisional, "n_grading"].eq(0).all()
        # And the per-fold counts sum to the frame's grading total.
        assert int(table.n_grading.sum()) == int(
            oof.grades_pooled.astype(bool).sum())


@pytest.mark.parametrize("labels", [[0., 1., 1., 0.], [1., 1., 1.], [0., 0.]])
def test_monitor_constant_home_edge_is_a_probability_brier_score(labels, tmp_path):
    # Exercise the actual master's expression without executing a daily run
    # (which fetches data and persists trained models). Then round-trip the
    # monitor writer used by the frontend, not merely an independent formula.
    import ast
    from pathlib import Path
    import monitoring
    source = Path(__file__).with_name("master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "baseline_brier"
                              for t in n.targets))
    oof = pd.DataFrame({"home_win": labels})
    value = eval(compile(ast.Expression(assignment.value), "master_baseline", "eval"),
                 {"np": np, "config": config, "ml_oof": oof})
    probability = 1.0 / (1.0 + 10.0 ** (-config.ELO_HOME_ADV / config.ELO_SCALE))
    expected = float(np.mean((np.asarray(labels) - probability) ** 2))
    assert value == pytest.approx(expected)
    record = monitoring.write_monitor_json(
        tmp_path / "monitor.json", "20261010", [], [], [], [], value)
    assert record["baseline_brier"] == pytest.approx(expected)
    assert record["brier_baseline"] == pytest.approx(expected)
    assert record["brier_baseline_label"] == "Constant home-edge"
    assert eval(compile(ast.Expression(assignment.value), "master_baseline", "eval"),
                {"np": np, "config": config, "ml_oof": pd.DataFrame()}) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


class TestMarketArms:
    """Arm A (incumbent) / B (offset) / C (market), NHL parity 2026-11-16.

    The offset arm is only meaningful if the price enters with a FIXED
    coefficient and travels with the model into predict. Both halves are
    cheap to get wrong and silent when wrong: a free coefficient discounts
    the price, and a missing margin at predict time serves the residual alone.
    """

    def test_offset_linear_leaves_the_price_it_was_given_alone(self):
        rng = np.random.default_rng(7)
        n = 600
        mkt = rng.uniform(0.2, 0.8, n)
        y = (rng.random(n) < mkt).astype(float)
        folds = rng.normal(size=(n, 4))
        base = ml_mod._market_logit(mkt)
        model = ml_mod._fit_offset_linear(folds, y, base)
        p = model.predict_proba(folds, base)[:, 1]
        # Uninformative features: the fit must leave the price essentially
        # where it found it rather than fitting the noise in the labels.
        assert np.abs(p - mkt).mean() < 0.05

    def test_the_offset_enters_with_a_coefficient_of_exactly_one(self):
        # The contract that separates an offset from a feature: shifting the
        # market logit must shift the decision by exactly that much. A free
        # coefficient (what appending the logit as a column gives the linear
        # member) cannot satisfy this - on the NHL folds it earned 0.43, a
        # silent discount on the price.
        rng = np.random.default_rng(11)
        n = 400
        mkt = rng.uniform(0.25, 0.75, n)
        y = (rng.random(n) < mkt).astype(float)
        X = rng.normal(size=(n, 5))
        base = ml_mod._market_logit(mkt)
        model = ml_mod._fit_offset_linear(X, y, base)
        delta = 0.37

        def _logit(p):
            return np.log(p / (1.0 - p))

        moved = _logit(model.predict_proba(X, base + delta)[:, 1])
        here = _logit(model.predict_proba(X, base)[:, 1])
        assert np.allclose(moved - here, delta, atol=1e-9)
        # And the features are not quietly standing in for the price: an
        # uninformative X learns an almost flat residual.
        assert np.abs(model.coef_).max() < 0.5

    def test_the_offset_arms_require_a_priced_column(self):
        frame = _frame([("2026-01-01", 4, 1), ("2026-01-08", 4, 1)])
        for arm in ("offset", "market"):
            with pytest.raises(ValueError, match="market_p_home"):
                ml_mod.walk_forward_oof(frame, arm=arm)
        with pytest.raises(ValueError, match="unknown arm"):
            ml_mod.walk_forward_oof(frame, arm="closing_line")

    def test_a_single_price_widens_the_population_only_where_priced(self):
        # A column that exists but carries no in-range price is not a priced
        # population: the arm must refuse rather than score a column of NaN.
        frame = _frame([("2026-01-01", 4, 1), ("2026-01-08", 4, 1)])
        frame["market_p_home"] = np.nan
        with pytest.raises(ValueError, match="no priced game"):
            ml_mod.walk_forward_oof(frame, arm="offset")

    def test_offset_fit_survives_the_fold_sort(self):
        # Regression (2026-11-16): canonical_sort re-sorts AND resets the
        # index, so a margin aligned by row index reindexed to NaN and every
        # member of the offset arm failed behind a warning - the arm scored
        # 0 rows on the real frame. The margin rides a COLUMN, read after the
        # sort, so reordering cannot detach a price from its game.
        rng = np.random.default_rng(3)
        n = 120
        frame = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(n)],
            "gameday": pd.to_datetime("2026-01-01")
                       + pd.to_timedelta(rng.permutation(n), unit="D"),
            "home_team": rng.integers(0, 8, n).astype(str),
            "away_team": rng.integers(0, 8, n).astype(str),
            "home_win": rng.integers(0, 2, n).astype(float),
        })
        for col in config.MONEYLINE_FEATURE_COLS:
            if col not in frame.columns:
                frame[col] = rng.normal(size=n)
        frame["_offset_margin"] = rng.normal(size=n)
        scrambled = frame.sample(frac=1.0, random_state=0)
        model, _ = ml_mod._fit_member("elasticnet", scrambled,
                                      base_margin="_offset_margin")
        assert isinstance(model, ml_mod._OffsetLinear)
        # A missing margin must fail loudly instead of fitting an unoffset
        # model that would then be compared as if it had seen the price.
        with pytest.raises(ValueError, match="market offset"):
            ml_mod._fit_member("elasticnet", scrambled,
                               base_margin="_missing_margin")

    def test_arm_metrics_grade_one_shared_population(self):
        from backend import evaluation as eval_mod
        a = pd.DataFrame({"game_id": ["1", "2", "3"],
                          "home_win": [1.0, 0.0, 1.0],
                          "p_ensemble_calibrated": [0.6, 0.4, 0.7]})
        b = pd.DataFrame({"game_id": ["1", "2", "3", "4"],
                          "home_win": [1.0, 0.0, 1.0, 0.0],
                          "p_ensemble_calibrated": [0.5, 0.5, 0.5, 0.5]})
        out = eval_mod.arm_metrics({"incumbent": a, "market": b}).set_index("arm")
        assert set(out["n"]) == {3}, "the extra game must not grade either arm"
        assert list(out.columns) == ["n", "auc", "logloss", "brier", "ece"]


class TestMarketLive:
    """Serve-time prices onto a slate (NHL parity, 2026-11-16).

    ``attach_live`` must price from the CURRENT quote, never the archive, and
    must key on (away, home, Eastern tip date) so tonight's meeting cannot
    borrow last month's number.
    """

    def _slate(self, rows):
        return pd.DataFrame(rows)

    def _hist(self, tmp_path, rows):
        path = tmp_path / "hist.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    @staticmethod
    def _live(mid, p=0.62):
        return pd.DataFrame([{"match_id": mid, "market_p_home": p,
                              "market_logit": float(np.log(p / (1 - p))),
                              "overround": 1.05, "price_utc": "2026-11-16T12:00:00Z",
                              "stale_minutes": 30.0, "pregame_minutes": 120.0,
                              "market_home_odds": 1.9, "market_away_odds": 1.9,
                              "market_book": "novig"}])

    def test_live_attach_prices_from_the_fetched_quote(self, tmp_path):
        from backend import market_rundown as mr
        hist = self._hist(tmp_path, [dict(sim_match_id="s1", away_abv="NYK",
                                          home_abv="BOS",
                                          tipoff_utc="2026-01-04T00:30:00Z")])
        slate = self._slate([dict(game_id="g1", gameday=pd.Timestamp("2026-01-03"),
                                  away_team="NYK", home_team="BOS")])
        seen = []

        def fetch(ids):
            seen.extend(ids)
            return self._live(ids[0], 0.62)

        out = mr.attach_live(slate, path=hist, fetcher=fetch, lookback_days=3650)
        assert seen == ["s1"]
        assert np.isclose(out.market_p_home.iloc[0], 0.62)
        assert np.isclose(out.stale_minutes.iloc[0], 30.0)

    def test_live_attach_discards_a_price_already_on_the_slate(self, tmp_path):
        from backend import market_rundown as mr
        hist = self._hist(tmp_path, [dict(sim_match_id="s1", away_abv="NYK",
                                          home_abv="BOS",
                                          tipoff_utc="2026-01-04T00:30:00Z")])
        slate = self._slate([dict(game_id="g1", gameday=pd.Timestamp("2026-01-03"),
                                  away_team="NYK", home_team="BOS",
                                  market_p_home=0.90, stale_minutes=999.0)])
        out = mr.attach_live(slate, path=hist,
                             fetcher=lambda ids: self._live(ids[0], 0.55),
                             lookback_days=3650)
        assert np.isclose(out.market_p_home.iloc[0], 0.55)
        assert np.isclose(out.stale_minutes.iloc[0], 30.0)

    def test_a_game_the_feed_cannot_price_stays_nan(self, tmp_path):
        from backend import market_rundown as mr
        hist = self._hist(tmp_path, [dict(sim_match_id="s1", away_abv="NYK",
                                          home_abv="BOS",
                                          tipoff_utc="2026-01-04T00:30:00Z")])
        slate = self._slate([dict(game_id="g1", gameday=pd.Timestamp("2026-01-03"),
                                  away_team="NYK", home_team="BOS"),
                             dict(game_id="g2", gameday=pd.Timestamp("2026-01-03"),
                                  away_team="LAL", home_team="DEN")])
        out = mr.attach_live(slate, path=hist,
                             fetcher=lambda ids: self._live(ids[0], 0.62),
                             lookback_days=3650)
        assert np.isclose(out.market_p_home.iloc[0], 0.62)
        assert pd.isna(out.market_p_home.iloc[1])

    def test_de_vig_rejects_a_placeholder_pair(self):
        from backend import market_rundown as mr
        p, _, ov = mr.de_vig(1.001, 1.001)
        assert np.isnan(p) and ov > mr.MAX_OVERROUND
        p, a, ov = mr.de_vig(1.9, 1.9)
        assert np.isclose(p, 0.5) and np.isclose(a, 0.5)


class TestMarketAttach:
    """Stats Insider prices onto the decided frame (2026-11-16).

    A DIRECT join on the tip-off instant. The feed's publish timestamp sits a
    day either side of tip-off and its finals can be a point off the box
    score, so anything but an exact tip-off key is an inferred match. The
    tests pin that down: shift the date by one day and the price must not
    attach; offer two prices for one game and neither may.
    """

    def _frame(self, gameday="2026-01-03"):
        return pd.DataFrame({
            "game_id": ["g1"],
            "gameday": pd.to_datetime([gameday]),
            "season": [2025],
            "home_team": ["BOS"],
            "away_team": ["NYK"],
            "home_score": [110.0],
            "away_score": [105.0],
        })

    def _si(self, tmp_path, rows):
        path = tmp_path / "si.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    @staticmethod
    def _row(**kw):
        base = dict(sim_match_id="s1", season=2025, away_abv="NYK",
                    home_abv="BOS", date_utc="2026-01-01T12:00:00+00:00",
                    tipoff_utc="2026-01-04T00:30:00Z", home_odds_dec=1.8,
                    away_odds_dec=2.0, final_home_score=110.0,
                    final_away_score=105.0)
        base.update(kw)
        return base

    def test_the_tipoff_instant_defines_the_join(self, tmp_path):
        from backend import market_rundown as mr
        # 00:30Z on Jan 4 is 19:30 Eastern on Jan 3 - the schedule's date.
        out = mr.attach_market(self._frame("2026-01-03"),
                               self._si(tmp_path, [self._row()]))
        assert np.isclose(out.market_p_home.iloc[0], 1/1.8 / (1/1.8 + 1/2.0))

    def test_a_one_day_shift_does_not_attach(self, tmp_path):
        from backend import market_rundown as mr
        out = mr.attach_market(self._frame("2026-01-04"),
                               self._si(tmp_path, [self._row()]))
        assert out.market_p_home.isna().all(), \
            "an off-by-one date is not a match; no tolerance is allowed"

    def test_two_prices_for_one_game_are_refused(self, tmp_path):
        from backend import market_rundown as mr
        rows = [self._row(sim_match_id="s1"),
                self._row(sim_match_id="s2", home_odds_dec=1.9,
                          away_odds_dec=1.9)]
        out = mr.attach_market(self._frame("2026-01-03"),
                               self._si(tmp_path, rows))
        assert out.market_p_home.isna().all(), \
            "ambiguity must be dropped, never broken by a tiebreak"

    def test_a_placeholder_quote_is_refused(self, tmp_path):
        from backend import market_rundown as mr
        out = mr.attach_market(
            self._frame("2026-01-03"),
            self._si(tmp_path, [self._row(home_odds_dec=1.001,
                                          away_odds_dec=1.001)]))
        assert out.market_p_home.isna().all(), \
            "an overround of 1.998 is a placeholder, not a price"

    def test_the_final_score_is_not_used_to_place_a_price(self, tmp_path):
        from backend import market_rundown as mr
        out = mr.attach_market(
            self._frame("2026-01-04"),
            self._si(tmp_path, [self._row(final_home_score=110.0,
                                          final_away_score=105.0)]))
        assert out.market_p_home.isna().all(), \
            "a matching score must not rescue an unmatched date"

    def test_the_newest_quote_for_a_match_wins(self, tmp_path):
        # NHL parity: Stats Insider keeps serving the last pre-game quote, so
        # the newest book stamp IS the closing line and must supersede an
        # earlier capture of the same match.
        from backend import market_rundown as mr
        rows = [
            self._row(sim_match_id="s1", home_odds_dec=1.8, away_odds_dec=2.0,
                      book_last_updated="2025-10-29T12:00:00Z",
                      fetched_utc="2026-11-16T00:00:00+00:00"),
            self._row(sim_match_id="s1", home_odds_dec=2.0, away_odds_dec=1.8,
                      book_last_updated="2026-01-03T23:05:00Z",
                      fetched_utc="2026-11-16T00:05:00+00:00"),
        ]
        out = mr.attach_market(self._frame("2026-01-03"),
                               self._si(tmp_path, rows))
        assert np.isclose(out.market_p_home.iloc[0],
                          (1/2.0) / (1/2.0 + 1/1.8))

    def test_staleness_is_measured_against_this_games_tipoff(self, tmp_path):
        from backend import market_rundown as mr
        row = self._row(book_last_updated="2026-01-03T22:00:00Z")
        out = mr.attach_market(self._frame("2026-01-03"),
                               self._si(tmp_path, [row]))
        # tip-off 2026-01-04T00:30Z minus a 22:00Z stamp = 150 minutes.
        assert np.isclose(out.pregame_minutes.iloc[0], 150.0)

    def test_an_odds_file_without_tipoff_is_rejected(self, tmp_path):
        from backend import market_rundown as mr
        row = self._row()
        row.pop("tipoff_utc")
        with pytest.raises(ValueError, match="tipoff_utc"):
            mr.attach_market(self._frame(), self._si(tmp_path, [row]))
