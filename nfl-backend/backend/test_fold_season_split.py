"""NFL fold classification and the grading split (2026-10-03).

NFL was the worst-covered sport: ~758 of 2,431 core games (31%) were never
validated, because a 7-CALENDAR-day block that straddled an NFL week fell
under MIN_VAL_FOLD_GAMES and was silently discarded -- with no skip log line
anywhere in the pipeline. Every postseason game in ten seasons was among
them: the OOF contained no February date at all (no Super Bowl, ever) and
each January count equalled regular-season Weeks 17/18 exactly.

These pin the replacement contract: every non-empty window is a fold, a thin
one is ``provisional`` (fit and scored, never grading), and a window with no
training rows is not a fold at all -- the guard the old value gate had been
providing by accident.
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


def _frame(spec, season: int = 2023) -> pd.DataFrame:
    """Build a decided frame from ``(date, n_games, game_type)`` triples."""
    rows = []
    for day, count, game_type in spec:
        for i in range(count):
            rows.append({
                "game_id": f"{day}-{i}",
                "gameday": pd.Timestamp(day),
                "season": season,
                "game_type": game_type,
                "home_team": f"T{i % 32:02d}",
                "away_team": f"T{(i + 1) % 32:02d}",
                "home_score": 20.0 + i % 15,
                "away_score": 17.0 + (i + 1) % 15,
            })
    df = pd.DataFrame(rows)
    df["home_win"] = (df.home_score > df.away_score).astype(float)
    return df


def _days(start: str, n: int, per_day: int, game_type: str = "REG"):
    """``n`` consecutive calendar days with ``per_day`` games each."""
    return [(str(d.date()), per_day, game_type)
            for d in pd.date_range(start, periods=n, freq="D")]


class TestRetainDontDrop:
    def test_sub_gate_windows_are_retained_as_provisional(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        # 1 game per 7-day block: far under the gate. Before 2026-10-03 this
        # frame produced exactly ONE fold (the exempt final window).
        df = folds_mod.canonical_sort(_frame(_days("2023-09-01", 70, 1)))
        diag: dict = {}
        folds = folds_mod.make_folds(df, diagnostics=diag)
        assert len(folds) > 1, "thin windows must be retained, not skipped"
        assert all(f.provisional for f in folds)
        assert not any(f.grades_pooled for f in folds)
        assert diag["dropped_windows"] == 0
        assert diag["dropped_games"] == 0
        assert diag["n_candidates"] == len(folds)

    def test_windows_at_or_above_the_gate_grade(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        df = folds_mod.canonical_sort(_frame(_days("2023-09-01", 70, 4)))
        folds = folds_mod.make_folds(df)
        assert folds
        assert not any(f.provisional for f in folds)
        assert all(f.grades_pooled for f in folds)

    def test_a_window_with_no_training_rows_is_not_a_fold(self, monkeypatch):
        """The guard the old value gate was providing by accident.

        The very first window of a frame with no pre-core rows has an empty
        train set, so no member can fit or predict for it (lightgbm: "input
        data must be 2 dimensional and non empty"; elasticnet: "0 samples").
        Keeping the value gate hid this; removing it must not.
        """
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        df = folds_mod.canonical_sort(_frame(_days("2023-09-01", 70, 4)))
        folds = folds_mod.make_folds(df)
        assert folds
        assert all(len(f.train_idx) > 0 for f in folds)

    def test_pre_oof_seasons_never_validate(self, monkeypatch):
        """Warmup = season < OOF_FIRST_SEASON, expressed by core_mask alone."""
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        warm = _frame(_days("2016-09-01", 40, 4, game_type="REG"),
                      season=2016)
        core = _frame(_days("2023-09-01", 60, 4, game_type="REG"),
                      season=config.OOF_FIRST_SEASON)
        df = folds_mod.canonical_sort(pd.concat([warm, core],
                                                ignore_index=True))
        folds = folds_mod.make_folds(df)
        assert folds
        for f in folds:
            assert set(df.loc[f.val_idx, "season"]) == {config.OOF_FIRST_SEASON}


class TestSeasonTypeClassification:
    def test_no_playoff_game_is_dropped(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        spec = (_days("2023-09-01", 60, 4, "REG")
                + _days("2024-01-10", 40, 4, "POST"))
        df = folds_mod.canonical_sort(_frame(spec))
        folds = folds_mod.make_folds(df)
        post_total = int(folds_mod.postseason_flag(df.game_type).sum())
        assert post_total > 0
        seen = [i for f in folds for i in f.val_idx.tolist()]
        assert len(seen) == len(set(seen)), "windows must not overlap"
        assert int(folds_mod.postseason_flag(
            df.game_type.iloc[seen]).sum()) == post_total

    def test_classifies_regular_postseason_and_mixed(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        # Windows are 7 CALENDAR days stepped from the first core date, so a
        # seam placed mid-window yields a mixed classification.
        spec = (_days("2023-09-01", 30, 4, "REG")
                + _days("2023-10-01", 30, 4, "POST"))
        df = folds_mod.canonical_sort(_frame(spec))
        kinds = {f.season_type for f in folds_mod.make_folds(df)}
        assert kinds <= {"regular", "postseason", "mixed"}
        assert "postseason" in kinds, kinds

    def test_postseason_flag_reads_the_nflverse_spelling(self):
        raw = pd.Series(["REG", "POST", "reg", "post", None, " PRE"])
        got = folds_mod.postseason_flag(raw).tolist()
        assert got == [False, True, False, True, False, False]

    def test_fold_table_exposes_the_split(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        df = folds_mod.canonical_sort(
            _frame(_days("2023-09-01", 60, 4, "REG")
                   + _days("2024-01-10", 30, 4, "POST")))
        table = folds_mod.fold_table(df, folds_mod.make_folds(df))
        for col in ("season_type", "provisional", "grades_pooled"):
            assert col in table.columns
        assert (table.grades_pooled == ~table.provisional).all()


class TestGradingSplit:
    @staticmethod
    def _feats(n_weeks: int = 12, start: str = "2023-09-01",
               season: int = 2023) -> pd.DataFrame:
        import features as feat_mod
        rng = np.random.default_rng(config.RANDOM_SEED)
        teams = [f"T{i:02d}" for i in range(8)]
        rows, day, gid = [], pd.Timestamp(start), 0
        for wk in range(n_weeks):
            order = teams.copy()
            rng.shuffle(order)
            for i in range(0, len(order), 2):
                rows.append({
                    "game_id": f"G{season}-{gid:04d}", "season": season,
                    "week": wk + 1,
                    "gameday": (day + pd.Timedelta(days=wk * 7 + (gid % 3))
                                ).strftime("%Y-%m-%d"),
                    "home_team": order[i], "away_team": order[i + 1],
                    "home_score": int(rng.integers(0, 45)),
                    "away_score": int(rng.integers(0, 45)),
                    "game_type": "REG", "roof": rng.choice(["outdoors", "dome"]),
                    "div_game": 0, "stadium": "Unknown Stadium",
                    "gametime": "13:00",
                })
                gid += 1
        games = pd.DataFrame(rows)
        out = feat_mod.build_game_features(games, pbp=None)
        return out.sort_values("gameday").reset_index(drop=True)

    def test_walk_forward_reports_a_season_split(self, monkeypatch):
        # Under the gate: every window is provisional, so nothing grades but
        # everything is still scored.
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 15)
        result = ml_mod.walk_forward_oof(self._feats())
        oof, split = result["oof"], result["season_split"]
        assert len(oof)
        assert split["regular_rows"] + split["postseason_rows"] == len(oof)
        assert split["provisional_rows"] == len(oof)
        assert split["grading_rows"] == 0
        assert not oof.grades_pooled.astype(bool).any()

    def test_windows_at_the_gate_grade_every_row(self, monkeypatch):
        # ...and with a gate every window clears, all rows grade.
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 3)
        result = ml_mod.walk_forward_oof(self._feats())
        oof, split = result["oof"], result["season_split"]
        assert len(oof)
        assert split["provisional_rows"] == 0
        assert split["grading_rows"] == len(oof)
        assert oof.grades_pooled.astype(bool).all()

    def test_fold_table_records_the_grading_count(self, monkeypatch):
        monkeypatch.setattr(config, "MIN_VAL_FOLD_GAMES", 3)
        result = ml_mod.walk_forward_oof(self._feats())
        oof, table = result["oof"], result["fold_table"]
        assert len(table)
        for col in ("season_type", "provisional", "n_grading"):
            assert col in table.columns
        assert (table.n_grading <= table.n_val).all()
        assert int(table.n_grading.sum()) == len(oof)
        assert int(oof.grades_pooled.astype(bool).sum()) == len(oof)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
