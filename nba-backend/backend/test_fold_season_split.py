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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
