"""NHL fold classification and the grading split (2026-10-03).

NHL retained every observed window on 2026-09-30 but "retained" only meant
the games were scored -- they still fed the blend optimizer and the headline
pooled metrics alongside ordinary regular-season windows, so a 7-game
Stanley Cup week carried the same SLSQP vote as a 70-game week. These pin the
part that was still missing: classification, and a grading split that keeps
retention (nothing hidden, everything scored) while separating evidence FOR
the model from evidence ABOUT it.
"""
from __future__ import annotations

import pandas as pd
import pytest

try:
    from backend import config, folds as folds_mod
except ImportError:  # pragma: no cover - direct script/import fallback
    import config
    import folds as folds_mod


def _frame(spec, season: int = 2025) -> pd.DataFrame:
    """Build a decided frame from ``(date, n_games, game_type)`` triples."""
    rows = []
    for day, count, game_type in spec:
        for i in range(count):
            rows.append({
                "game_id": f"{day}-{i}",
                "gameday": pd.Timestamp(day),
                "season": season,
                "game_type": game_type,
                "home_team": i % 32,
                "away_team": (i + 1) % 32,
                "home_score": 3.0 + i % 5,
                "away_score": 2.0 + (i + 1) % 5,
            })
    df = pd.DataFrame(rows)
    df["home_win"] = (df.home_score > df.away_score).astype(float)
    return df


def _days(start: str, n: int, per_day: int, game_type: int = 2):
    """``n`` consecutive observed dates with ``per_day`` games each."""
    return [(str(d.date()), per_day, game_type)
            for d in pd.date_range(start, periods=n, freq="D")]


class TestNHLFoldClassification:
    def test_every_observed_window_is_retained(self, monkeypatch):
        """NHL's 2026-09-30 contract: thin windows are disclosed, not dropped."""
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        df = folds_mod.canonical_sort(_frame(_days("2025-01-01", 60, 1)))
        diag: dict = {}
        folds = folds_mod.make_folds(df, diagnostics=diag)
        assert folds
        assert diag["dropped_windows"] == 0
        assert diag["dropped_games"] == 0
        assert diag["n_candidates"] == len(folds)

    def test_thin_windows_grade_nothing_but_still_exist(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        df = folds_mod.canonical_sort(_frame(_days("2025-01-01", 60, 1)))
        folds = folds_mod.make_folds(df)
        assert all(f.provisional for f in folds)
        assert not any(f.grades_pooled for f in folds)

    def test_windows_at_or_above_the_gate_grade(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        # 100 observed dates from index 30 is 10 full 7-date windows.
        df = folds_mod.canonical_sort(_frame(_days("2025-01-01", 100, 8)))
        folds = folds_mod.make_folds(df)
        assert folds
        assert not any(f.provisional for f in folds)
        assert all(f.grades_pooled for f in folds)

    def test_classifies_postseason_and_mixed(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        # 41 regular dates then playoffs: windows start at index 30 and step
        # 7, so [37..43] straddles the seam and must classify as mixed.
        spec = (_days("2025-01-01", 41, 8, game_type=2)
                + _days("2025-03-01", 30, 8, game_type=3))
        df = folds_mod.canonical_sort(_frame(spec))
        kinds = {f.season_type for f in folds_mod.make_folds(df)}
        assert kinds == {"regular", "postseason", "mixed"}, kinds

    def test_no_playoff_game_is_dropped(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        spec = (_days("2025-01-01", 60, 8, game_type=2)
                + _days("2025-04-01", 30, 4, game_type=3))
        df = folds_mod.canonical_sort(_frame(spec))
        folds = folds_mod.make_folds(df)
        post_total = int(folds_mod.postseason_flag(df.game_type).sum())
        assert post_total > 0
        seen = [i for f in folds for i in f.val_idx.tolist()]
        assert len(seen) == len(set(seen)), "windows must not overlap"
        assert int(folds_mod.postseason_flag(
            df.game_type.iloc[seen]).sum()) == post_total

    def test_postseason_flag_matches_the_is_playoffs_feature(self):
        """The gate and the model must agree on what a playoff row is.

        ``features.py:1230`` derives ``is_playoffs`` from ``game_type`` with
        ``== GAME_TYPE_POST``; the fold classifier must use the identical
        expression or the grading population and the feature would diverge.
        """
        raw = pd.Series([2, 3, 1, 3, None, 2])
        expected = (pd.to_numeric(raw, errors="coerce")
                    == config.GAME_TYPE_POST).astype(float).fillna(0.0)
        got = folds_mod.postseason_flag(raw).astype(float)
        assert got.reset_index(drop=True).equals(
            expected.astype(float).reset_index(drop=True))

    def test_fold_table_exposes_the_split(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 40)
        df = folds_mod.canonical_sort(
            _frame(_days("2025-01-01", 60, 8, game_type=2)
                   + _days("2025-04-01", 30, 4, game_type=3)))
        table = folds_mod.fold_table(df, folds_mod.make_folds(df))
        for col in ("season_type", "provisional", "grades_pooled"):
            assert col in table.columns
        assert (table.grades_pooled == ~table.provisional).all()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
