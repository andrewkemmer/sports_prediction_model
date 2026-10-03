"""Season-split walk-forward tests (2026-10-03 remediation).

Covers the MLB port of the shared season-split design:
  * every non-empty window is RETAINED (no size-based skipping),
  * sub-gate windows are marked provisional — fit + scored, not graded,
  * the grading population is regular-season rows of non-provisional folds,
  * four published blocks (regular/postseason/provisional/all) reconcile
    against the headline metrics,
  * the margin-split helpers keep one unconditional fold set (the
    margin-fold-sync assertion's precondition).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import training  # noqa: E402
from training import (  # noqa: E402
    _regenerate_splits,
    canonical_walk_forward_splits,
    get_last_season_split,
    postseason_flag,
    walk_forward_splits,
)

CADENCE = 7


def _frame(n_days: int = 60, games_per_day: int = 4, post_from: int = 51
           ) -> pd.DataFrame:
    """Deterministic decided frame: daily games, postseason from ``post_from``.

    With cadence 7 and val_start_idx = 7 the windows are days 7-48 (all R),
    49-55 (mixed R+W), and the 56-59 partial tail (all W) — exercising all
    three season_type labels plus a sub-gate tail.
    """
    dates = pd.date_range("2024-03-01", periods=n_days, freq="D")
    rows = []
    pk = 1000
    for d_i, d in enumerate(dates):
        for _ in range(games_per_day):
            pk += 1
            rows.append({
                "game_pk": pk,
                "game_date": d,
                "home_win": float(pk % 2),
                "game_type": "R" if d_i < post_from else "W",
                "home_team": "HOM",
                "away_team": "AWY",
            })
    return pd.DataFrame(rows)


def _patch_trainers(monkeypatch) -> None:
    """Bypass the heavy ensemble; evaluate's own accounting stays real."""

    def _fake_train(train, val=None, *args, **kwargs):
        return {"m1": object()}, {"auc": 0.6, "brier": 0.3,
                                  "logloss": 0.5, "ece": 0.01}

    def _fake_predict(models, games, *args, **kwargs):
        n = len(games)
        p = np.linspace(0.35, 0.65, n) if n > 1 else np.array([0.5])
        return p, {"m1": p.copy()}, {"m1": 1.0}

    monkeypatch.setattr(training, "train_moneyline_ensemble", _fake_train)
    monkeypatch.setattr(training, "ensemble_predict", _fake_predict)


def test_postseason_flag_recognizes_round_codes_and_text():
    got = postseason_flag(["R", "F", "D", "L", "W", "P",
                           "playoff", "POSTSEASON", "S"])
    assert got.tolist() == [False, True, True, True, True, True,
                            True, True, False]
    assert postseason_flag([]).tolist() == []


def test_walk_forward_splits_stamps_season_type():
    splits = walk_forward_splits(_frame(), retrain_cadence_days=CADENCE)
    assert splits, "synthetic frame must produce windows"
    labels = {s["season_type"] for s in splits}
    assert {"regular", "mixed", "postseason"} <= labels, labels
    mixed = next(s for s in splits if s["season_type"] == "mixed")
    gt = set(mixed["val_games"]["game_type"])
    assert "R" in gt and "W" in gt
    # A frame without game_type never crashes and reads regular.
    bare = _frame().drop(columns=["game_type"])
    assert all(s["season_type"] == "regular"
               for s in walk_forward_splits(bare, CADENCE))


def test_canonical_retains_every_window_and_stamps_provisional():
    decided, kept = canonical_walk_forward_splits(
        _frame(), retrain_cadence_days=CADENCE, min_val_games=10_000)
    raw = walk_forward_splits(decided, retrain_cadence_days=CADENCE)
    # The old kept-filter dropped every window under the gate (the 8
    # October windows of the diagnosis). Retention is now unconditional.
    assert len(kept) == len(raw)
    assert all(s["provisional"] for s in kept)
    assert all(s.get("season_type") for s in kept)

    _, kept_low = canonical_walk_forward_splits(
        _frame(), retrain_cadence_days=CADENCE, min_val_games=1)
    assert len(kept_low) == len(raw)
    assert not any(s["provisional"] for s in kept_low)


def test_evaluate_scores_every_window_and_splits_the_grade(
        monkeypatch, caplog):
    import logging
    _patch_trainers(monkeypatch)
    with caplog.at_level(logging.INFO, logger="training"):
        _, pooled, combined = training.walk_forward_evaluate(
            _frame(), retrain_cadence_days=CADENCE, min_val_games=20)

    # Retention: every val row of every window is in the OOF frame —
    # 6 regular windows x 28 + mixed 28 + tail 16.
    assert len(combined) == 212
    assert int(combined["provisional"].astype(bool).sum()) == 16
    assert int(combined["is_playoffs"].astype(bool).sum()) == 36
    assert int(combined["grades_pooled"].astype(bool).sum()) == 176
    # Grading population = regular-season rows of non-provisional folds.
    assert (combined["grades_pooled"].astype(bool).to_numpy()
            == (combined["game_type"] == "R").to_numpy()).all()

    # The headline pools ONLY the grading rows.
    g = combined["grades_pooled"].astype(bool).to_numpy()
    expected = training.compute_metrics(
        combined["home_win"].values[g],
        combined["home_win_prob_model"].values[g])
    for k, v in expected.items():
        assert pooled.get(k) == v, (k, pooled.get(k), v)

    rep = get_last_season_split()
    assert rep["counts"] == {"regular_rows": 176, "postseason_rows": 36,
                             "provisional_rows": 16, "grading_rows": 176,
                             "oof_rows": 212}
    blocks = rep["blocks"]
    assert blocks["oof_regular"]["n"] == 176
    assert blocks["oof_postseason"]["n"] == 36
    assert blocks["oof_provisional"]["n"] == 16
    assert blocks["oof_all"]["n"] == 212
    # oof_regular reconciles with the headline exactly.
    for k in ("auc", "brier", "logloss", "ece"):
        assert blocks["oof_regular"][k] == pooled[k]
    # sufficiency is disclosed, not assumed.
    assert blocks["oof_regular"]["sufficient"] is True
    assert blocks["oof_provisional"]["sufficient"] is False  # 16 < 20

    # Every scored row still ships to the CSV/history schema.
    assert {"is_playoffs", "season_type", "provisional", "grades_pooled",
            "home_win_prob_model", "home_win_prob_model_calibrated",
            "fold_idx"}.issubset(combined.columns)
    # The old size-based skip is gone; sub-gate windows log as provisional.
    assert "Skipping fold" not in caplog.text
    assert "PROVISIONAL" in caplog.text


def test_evaluate_keeps_every_row_when_all_windows_are_provisional(
        monkeypatch, caplog):
    import logging
    _patch_trainers(monkeypatch)
    with caplog.at_level(logging.INFO, logger="training"):
        _, pooled, combined = training.walk_forward_evaluate(
            _frame(), retrain_cadence_days=CADENCE, min_val_games=30)

    # OLD behavior: every non-partial window fell below the gate and was
    # dropped, leaving only the partial tail pooled into the headline.
    # NEW: all 212 rows retained — fit and scored — and none of them grade.
    assert len(combined) == 212
    assert combined["provisional"].astype(bool).all()
    assert not combined["grades_pooled"].astype(bool).any()
    assert pooled == {"auc": 0.5, "brier": 0.25, "logloss": 0.69,
                      "ece": 0.0}

    rep = get_last_season_split()
    assert rep["counts"]["grading_rows"] == 0
    assert rep["counts"]["oof_rows"] == 212
    assert rep["blocks"]["oof_provisional"]["n"] == 212
    assert rep["blocks"]["oof_regular"]["n"] == 0
    assert rep["blocks"]["oof_regular"]["sufficient"] is False
    assert "Skipping fold" not in caplog.text


def test_margin_split_helpers_keep_one_unconditional_fold_set():
    frame = _frame()
    splits = walk_forward_splits(frame, retrain_cadence_days=CADENCE)
    # Even an absurd gate retains everything: the regenerated folds must
    # match the margin-build folds one-for-one or the fold-sync assertion
    # in _attach_oof_run_margins would fire in production.
    regen = _regenerate_splits(frame, splits, 999_999, CADENCE, 0, 0)
    assert [s["fold_idx"] for s in regen] == [s["fold_idx"] for s in splits]
    assert len(regen) == len(splits)
