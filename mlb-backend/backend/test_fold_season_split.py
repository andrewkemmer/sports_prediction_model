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


def _fake_train_multi(train, val=None, *args, **kwargs):
    return {"m1": object(), "m2": object()}, {
        "auc": 0.6, "brier": 0.3, "logloss": 0.5, "ece": 0.01}


_PRED_CALLS: list[dict[int, float]] = []


def _fake_predict_multi(models, games, *args, **kwargs):
    """Row-stable member probs + WILDLY varying per-fold weights.

    The per-fold blend alternates between pure-m1 and pure-m2 across
    folds, so any published column that carries fold-time weights is
    trivially distinguishable from the deployed-weights blend.
    """
    pk = games["game_pk"].to_numpy(float)
    p1 = 0.25 + (pk % 89) / 200.0
    p2 = 0.65 - (pk % 61) / 200.0
    k = len(_PRED_CALLS)
    if k % 2 == 0:
        w, blend = {"m1": 1.0, "m2": 0.0}, p1
    else:
        w, blend = {"m1": 0.0, "m2": 1.0}, p2
    _PRED_CALLS.append(dict(zip(games["game_pk"].tolist(),
                                np.asarray(blend, float).tolist())))
    return blend, {"m1": p1.copy(), "m2": p2.copy()}, w


def test_published_blend_is_causal_and_replay_is_explicit(monkeypatch, tmp_path):
    """NHL parity: headline is causal; final-weight replay is NOT OOF.

    Persist both policies explicitly so future-serving weights cannot be
    mistaken for weights that were available at historical origins.
    """
    import joblib
    from calibration import MIN_OOF_FOR_FIT, moneyline_apply, gated_moneyline_fit

    _PRED_CALLS.clear()
    monkeypatch.setattr(training, "train_moneyline_ensemble", _fake_train_multi)
    monkeypatch.setattr(training, "ensemble_predict", _fake_predict_multi)
    # Big enough that the prequential map fits part-way through (graded
    # rows > MIN_OOF_FOR_FIT), with labels carrying REAL signal so the
    # Platt fit is non-degenerate and the calibrated column is NOT
    # trivially the rounded raw column (pure-noise labels make fit_platt
    # reject a<0 and fall back to identity).
    frame = _frame(n_days=110, games_per_day=5, post_from=101)
    pk_f = frame["game_pk"].to_numpy(float)
    p1_f = 0.25 + (pk_f % 89) / 200.0
    p2_f = 0.65 - (pk_f % 61) / 200.0
    u = (pk_f * 0.6180339887) % 1.0
    frame["home_win"] = (u < 0.5 * (p1_f + p2_f)).astype(float)
    _, pooled, combined = training.walk_forward_evaluate(
        frame, retrain_cadence_days=CADENCE, min_val_games=20)

    deployed = dict(training._LAST_ADAPTIVE_WEIGHTS)
    assert deployed, "the walk must leave the deployed earning weights"

    # Final-weight replay is deliberately separate from causal headlines.
    pk = combined["game_pk"].to_numpy(float)
    p1 = 0.25 + (pk % 89) / 200.0
    p2 = 0.65 - (pk % 61) / 200.0
    expected = training._pool_member_probs(
        {"m1": p1, "m2": p2}, training._member_weights(["m1", "m2"]))
    got = combined["home_win_prob_model"].to_numpy(float)
    assert np.allclose(combined["home_win_prob_model_retrospective"],
                       expected, atol=1e-12, rtol=0)
    assert np.array_equal(got, combined["home_win_prob_model_causal"])

    # (2) Discriminating: the published column is NOT the fold-time blend
    # (the fakes alternate pure-m1 / pure-m2 per fold).
    fold_blend_map: dict[int, float] = {}
    for call in _PRED_CALLS:
        fold_blend_map.update(call)
    fold_blend = np.array([fold_blend_map[int(k_)]
                           for k_ in combined["game_pk"].tolist()])
    np.testing.assert_array_equal(got, fold_blend)
    assert not np.allclose(got, expected, atol=1e-9), \
        "fixture must distinguish causal headlines from final-weight replay"

    # (3) Prequential honesty kept: fold k's calibrated column comes from
    # a map fitted strictly on folds < k's published pairs.
    g = combined["grades_pooled"].astype(bool).to_numpy()
    assert int(g.sum()) > MIN_OOF_FOR_FIT, "fixture must exercise the map path"
    acc_y: list[float] = []
    acc_p: list[float] = []
    exp_cal = np.zeros(len(combined))
    for _, frows in combined.groupby("fold_idx", sort=False):
        blend_k = frows["home_win_prob_model"].to_numpy(float)
        cal, _ = gated_moneyline_fit(np.asarray(acc_y, float), np.asarray(acc_p, float))
        exp_cal[frows.index] = np.round(moneyline_apply(blend_k, cal), 4)
        gk = frows["grades_pooled"].astype(bool).to_numpy()
        acc_y.extend(frows["home_win"].to_numpy(float)[gk].tolist())
        acc_p.extend(blend_k[gk].tolist())
    assert np.allclose(
        combined["home_win_prob_model_calibrated"].to_numpy(float),
        exp_cal, atol=5e-5, rtol=0)
    # The nested gate may honestly decline a map; fit/gate behavior has
    # separate positive and negative regression cases.

    # (4) The persisted bundle ("the artifact") is self-consistent: its
    # metrics are the pooled OOF of ITS OWN blend — the acceptance that
    # the artifact matches the production binary's blend pooled OOF.
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path)
    training.persist_ensemble({"m1": object(), "m2": object()}, pooled)
    bundle = joblib.load(tmp_path / training.ENSEMBLE_FILE)
    assert bundle["adaptive_weights"] == deployed
    # Replay through the SERVING path (restore + _member_weights) exactly
    # as predict_games would blend a future game.
    training.set_adaptive_weights(bundle["adaptive_weights"])
    y_g = combined["home_win"].to_numpy(float)[g]
    p_g = training._pool_member_probs(
        {"m1": p1[g], "m2": p2[g]}, training._member_weights(["m1", "m2"]))
    replay = training.compute_metrics(y_g, p_g)
    for k in ("auc", "brier", "logloss", "ece"):
        causal = training.compute_metrics(y_g, got[g])
        assert bundle["metrics"][k] == causal[k] == pooled[k], k
        assert bundle["season_split"]["retrospective_metrics_not_oof"][k] == replay[k]
    assert bundle["evaluation_policy"] == "causal_rolling_blend"


def test_future_labels_cannot_rewrite_causal_headlines(monkeypatch):
    monkeypatch.setattr(training, "train_moneyline_ensemble", _fake_train_multi)
    def predict(models, games):
        pk = games["game_pk"].to_numpy(float)
        members = {"m1": 0.25 + (pk % 89) / 200,
                   "m2": 0.65 - (pk % 61) / 200}
        weights = training._member_weights(list(members))
        return training._pool_member_probs(members, weights), members, weights
    monkeypatch.setattr(training, "ensemble_predict", predict)
    frame = _frame(n_days=180, games_per_day=5, post_from=999)
    _, _, before = training.walk_forward_evaluate(frame, min_val_games=20)
    last = before.loc[before.fold_idx == before.fold_idx.max(), "game_pk"]
    poisoned = frame.copy()
    mask = poisoned.game_pk.isin(last)
    poisoned.loc[mask, "home_win"] = 1 - poisoned.loc[mask, "home_win"]
    _, _, after = training.walk_forward_evaluate(poisoned, min_val_games=20)
    for col in ["home_win_prob_model", "home_win_prob_model_calibrated",
                "weight_m1", "weight_m2"]:
        np.testing.assert_array_equal(before[col], after[col], err_msg=col)


def test_tiny_validation_rows_are_scored_not_silently_lost(monkeypatch):
    _patch_trainers(monkeypatch)
    frame = _frame(n_days=22, games_per_day=1, post_from=999)
    _, _, oof = training.walk_forward_evaluate(frame, min_train_days=14)
    assert len(oof) == 8
    assert oof.groupby("fold_idx").size().tolist() == [7, 1]
    assert oof.provisional.all()


def test_infinite_features_fail_before_model_fit():
    import pytest
    frame = pd.DataFrame({"is_home": [np.inf]})
    with pytest.raises(ValueError, match="infinite observations"):
        training._feature_matrix(frame)


def test_cached_bundle_cannot_score_corrected_schema_without_refit():
    import pytest
    import config
    training.reset_feature_subset()
    try:
        with pytest.raises(ValueError, match="older feature schema"):
            training.apply_bundle_feature_cols({"feature_cols": training.MONEYLINE_FEATURE_COLS})
        training.apply_bundle_feature_cols({
            "feature_cols": training.MONEYLINE_FEATURE_COLS,
            "feature_schema_version": config.FEATURE_SCHEMA_VERSION})
        assert training.active_moneyline_feature_cols() == training.MONEYLINE_FEATURE_COLS
    finally:
        training.reset_feature_subset()


def test_no_history_refuses_in_sample_oof():
    import pytest
    with pytest.raises(ValueError, match="strictly-prior"):
        training.walk_forward_evaluate(_frame(n_days=3))


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
