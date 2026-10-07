"""Causal moneyline evidence and next-origin fit policy, offline and bounded."""
import ast
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
import moneyline as ml

config = ml.config
folds = ml.folds_mod


def frame(n=72):
    rng = np.random.default_rng(73)
    df = pd.DataFrame({"game_id": [str(i) for i in range(n)],
                       "gameday": pd.date_range("2024-10-01", periods=n),
                       "season": 2024, "home_win": np.arange(n) % 2,
                       "home_team": np.where(np.arange(n) % 2, "ANA", "BOS"),
                       "away_team": np.where(np.arange(n) % 3, "BUF", "TOR")})
    for col in config.active_moneyline_feature_cols():
        df[col] = rng.normal(size=n)
    return df


class FixedModel:
    def __init__(self, name):
        self.name = name
        self.best_iteration = 6

    def fit(self, X, y, **kwargs):
        return self

    def predict_proba(self, X):
        p = np.full(len(X), {"xgboost": .8, "lightgbm": .2, "elasticnet": .5}[self.name])
        return np.column_stack([1 - p, p])


def test_future_outcomes_cannot_rewrite_headline_or_earlier_calibration(monkeypatch):
    df = frame(72 * 8)
    df["gameday"] = pd.date_range("2024-10-01", periods=72).repeat(8)
    fold_list = folds.make_folds(df)
    assert len(fold_list) >= 3
    monkeypatch.setattr(ml, "_make_member", lambda name, **kwargs: FixedModel(name))
    # Deliberately outcome-sensitive weight learning, unlike a fixed mocked
    # vector which could not expose retrospective-overwrite leakage.
    def earn(members, y):
        return {"xgboost": float(y[-1] == 1), "lightgbm": float(y[-1] == 0), "elasticnet": 0.}
    monkeypatch.setattr(ml, "compute_adaptive_weights", earn)
    original = ml.walk_forward_oof(df, fold_list=fold_list)
    poisoned = df.copy()
    poisoned.loc[fold_list[-1].val_idx, "home_win"] = 1 - poisoned.loc[fold_list[-1].val_idx, "home_win"]
    changed = ml.walk_forward_oof(poisoned, fold_list=fold_list)
    a, b = original["oof"], changed["oof"]
    assert original["member_weights"] != changed["member_weights"]
    np.testing.assert_array_equal(a.p_ensemble, b.p_ensemble)
    np.testing.assert_array_equal(a.p_ensemble, a.p_ensemble_causal)
    assert not np.allclose(a.p_ensemble_retrospective, b.p_ensemble_retrospective)
    for name in config.ENSEMBLE_MEMBERS:
        np.testing.assert_array_equal(a[f"weight_{name}"], b[f"weight_{name}"])
    ca, aa = ml.prequential_fold_calibrators(a.p_ensemble, a.home_win, a.fold_id, a.grades_pooled)
    cb, ab = ml.prequential_fold_calibrators(b.p_ensemble, b.home_win, b.fold_id, b.grades_pooled)
    assert ca == cb and aa == ab
    np.testing.assert_array_equal(original["fold_table"].xgb_rounds,
                                  [config.XGBOOST_FOLD0_ROUNDS] + [7] * (len(fold_list) - 1))
    assert original["final_xgb_rounds"] == 7
    assert original["xgb_best_rounds"] == [7] * len(fold_list)


def test_calibration_ignores_excluded_rows_and_shares_next_origin_gate(monkeypatch):
    monkeypatch.setattr(config, "MIN_OOF_FOR_FIT", 4)
    monkeypatch.setattr(config, "CAL_GATE_MIN_HOLDOUT", 2)
    monkeypatch.setattr(config, "CAL_GATE_HOLDOUT_FRAC", .25)
    p = np.full(24, .9)
    y = np.tile([0., 1.], 12)
    fids = np.repeat(np.arange(4), 6)
    grades = np.tile([True, True, True, False, False, False], 4)
    calls = []
    def fit(prob, target):
        calls.append((prob.copy(), target.copy()))
        return {"a": 1., "b": 0., "method": ml.FAVORED_CALIBRATOR_METHOD}
    monkeypatch.setattr(ml, "moneyline_fit", fit)
    monkeypatch.setattr(ml, "moneyline_apply", lambda prob, cal: np.full(len(prob), .5))
    ca, aa = ml.prequential_fold_calibrators(p, y, fids, grades)
    reference_calls = [(a.tolist(), b.tolist()) for a, b in calls]
    calls.clear()
    dirty_p, dirty_y = p.copy(), y.copy()
    dirty_p[~grades] = .01
    dirty_y[~grades] = 1 - dirty_y[~grades]
    cb, ab = ml.prequential_fold_calibrators(dirty_p, dirty_y, fids, grades)
    assert ca == cb and aa == ab
    assert reference_calls == [(a.tolist(), b.tolist()) for a, b in calls]
    assert [r["n_prior"] for r in aa["folds"]] == [0, 3, 6, 9]
    final, final_audit = ml.gated_calibrator(p, y, grades)
    # Adding an empty next origin exposes EXACTLY the final-serving policy.
    next_maps, next_audit = ml.prequential_fold_calibrators(
        np.append(p, .7), np.append(y, 1), np.append(fids, 4), np.append(grades, False))
    assert next_maps[4] == final
    assert {k: v for k, v in next_audit["folds"][-1].items() if k != "fold_id"} == final_audit


def test_final_calibration_does_not_bypass_no_gain_gate(monkeypatch):
    monkeypatch.setattr(config, "MIN_OOF_FOR_FIT", 4)
    monkeypatch.setattr(config, "CAL_GATE_MIN_HOLDOUT", 2)
    monkeypatch.setattr(ml, "moneyline_fit", lambda *args: {"a": 1., "b": 0., "method": ml.FAVORED_CALIBRATOR_METHOD})
    monkeypatch.setattr(ml, "moneyline_apply", lambda prob, cal: prob.copy())
    cal, audit = ml.gated_calibrator(np.full(12, .7), np.tile([0., 1.], 6))
    assert cal is None and audit["reason"] == "gated_no_gain"


@pytest.mark.parametrize("kind", ["probability", "fold", "grades"])
def test_calibration_rejects_misaligned_vectors(kind):
    p, y, fids, grades = np.full(8, .7), np.tile([0., 1.], 4), np.repeat([0, 1], 4), np.ones(8, bool)
    if kind == "probability":
        p = p[:-1]
    elif kind == "fold":
        fids = fids[:-1]
    else:
        grades = grades[:-1]
    with pytest.raises(ValueError, match="align"):
        ml.prequential_fold_calibrators(p, y, fids, grades)


def test_xgb_category_vocabulary_is_static_with_unknown_slot():
    df = frame(8)
    training = ml.member_fit_input("xgboost", ml.member_matrix("xgboost", df), None)
    slate = df.iloc[:2].copy()
    slate["home_team"] = ["TOR", "NEW_UNKNOWN_TEAM"]
    serving = ml.member_fit_input("xgboost", ml.member_matrix("xgboost", slate), None)
    for col in config.TREE_CATEGORICAL_COLS:
        assert isinstance(training[col].dtype, pd.CategoricalDtype)
        assert training[col].cat.categories.tolist() == serving[col].cat.categories.tolist()
    assert serving.home_team_id.tolist() == [config.NHL_TEAM_ID["TOR"], config.UNK_TEAM_ID]
    assert not ({"home_team_id", "away_team_id"} & set(ml.member_matrix("elasticnet", df)))


def test_shared_fold_final_fit_arguments_and_explicit_round_ledger(monkeypatch):
    captures = []
    class Capture(FixedModel):
        def fit(self, X, y, **kwargs):
            captures.append((self.name, X.copy(), y.copy(), kwargs))
            return self
    constructors = []
    def make(name, **kwargs):
        constructors.append((name, kwargs))
        return Capture(name)
    monkeypatch.setattr(ml, "_make_member", make)
    df = frame(12)
    for name in config.ENSEMBLE_MEMBERS:
        ml._fit_member(name, ml.member_matrix(name, df), df.home_win.to_numpy(), fold=True, xgb_rounds=23)
    model, pre = ml.fit_final_models(df.sample(frac=1, random_state=19), xgb_best_rounds=[19, 27])
    for fold_capture, final_capture in zip(captures[:3], captures[3:]):
        assert fold_capture[0] == final_capture[0]
        np.testing.assert_array_equal(fold_capture[1], final_capture[1])
        np.testing.assert_array_equal(fold_capture[2], final_capture[2])
        assert fold_capture[3] == final_capture[3]
        assert "eval_set" not in final_capture[3]
    final_xgb = [kw for name, kw in constructors if name == "xgboost"][-1]
    assert final_xgb == {"fold": False, "n_estimators": 23, "causal": True}
    assert pre is model["elasticnet"]["pre"]
    # A previous walk's mutable ledger MUST NOT steer an independent refit.
    monkeypatch.setattr(ml, "_LAST_XGB_BEST_ROUNDS", [999])
    ml.fit_final_models(df)
    assert [kw for name, kw in constructors if name == "xgboost"][-1]["n_estimators"] == config.XGBOOST_FOLD0_ROUNDS


def test_real_estimators_refit_serialize_and_predict_unseen_clubs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "XGBOOST_PARAMS", {**config.XGBOOST_PARAMS, "max_depth": 2,
                         "gamma": 0., "min_child_weight": 1, "colsample_bytree": 1., "subsample": 1.})
    monkeypatch.setattr(config, "LIGHTGBM_PARAMS", {**config.LIGHTGBM_PARAMS, "n_estimators": 5,
                         "min_child_samples": 2, "min_gain_to_split": 0., "n_jobs": 1})
    df = frame(64)
    models, _ = ml.fit_final_models(df, xgb_best_rounds=[5, 7])
    xgb = models["xgboost"]["model"].get_booster()
    assert xgb.num_boosted_rounds() == 6
    types = dict(zip(xgb.feature_names, xgb.feature_types))
    assert all(types[col] == "c" for col in config.TREE_CATEGORICAL_COLS)
    lgb = models["lightgbm"]["model"].booster_.dump_model()
    assert all(lgb["feature_infos"][col]["values"] for col in config.TREE_CATEGORICAL_COLS)
    slate = df.iloc[:3].copy()
    slate["home_team"] = ["TOR", "SEA", "NOT_A_TEAM"]
    weights = dict(config.ENSEMBLE_WEIGHTS)
    expected = ml.predict_slate(models, slate, weights)
    assert np.isfinite(expected).all() and ((expected > 0) & (expected < 1)).all()
    bundle = tmp_path / "models.joblib"
    joblib.dump(models, bundle)
    replay = ml.predict_slate(joblib.load(bundle), slate, weights)
    np.testing.assert_array_equal(expected, replay)
    # Real fold model uses precisely the final representation and fit args.
    fold_entry = ml._fit_member("xgboost", ml.member_matrix("xgboost", df),
                                df.home_win.to_numpy(), fold=True, xgb_rounds=6)
    np.testing.assert_array_equal(
        models["xgboost"]["model"].predict_proba(ml.member_fit_input("xgboost", ml.member_matrix("xgboost", slate), None)),
        fold_entry["model"].predict_proba(ml.member_fit_input("xgboost", ml.member_matrix("xgboost", slate), None)))


def test_master_routes_causal_mask_gate_and_own_round_ledger():
    source = (BACKEND / "master_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)]
    def call(name):
        return next(node for node in calls if node.func.attr == name)
    for name in ("gated_calibrator", "prequential_fold_calibrators"):
        node = call(name)
        assert ast.unparse(node.args[0]) == "p_ens"
        assert {kw.arg: ast.unparse(kw.value) for kw in node.keywords}["grades"] == "_grading"
    node = call("fit_final_models")
    assert {kw.arg: ast.unparse(kw.value) for kw in node.keywords}["xgb_best_rounds"] == "ml['xgb_best_rounds']"
    assert '"moneyline_retrospective_not_oof": retrospective_m' in source
    assert '"headline_view": "causal_rolling_blend"' in source
    assert '"xgb_best_rounds": ml["xgb_best_rounds"]' in source
    # Geometry is written before fitting; the later write MUST persist the
    # earned budget/weight columns rather than losing them in memory.
    merge = next(node for node in calls if node.func.attr == "merge"
                 and any(kw.arg == "validate" and ast.literal_eval(kw.value) == "one_to_one"
                         for kw in node.keywords))
    assert ast.unparse(merge.func.value) == "fold_tbl"
    assert 'ml[\'fold_table\']' in ast.unparse(merge.args[0])
    assert 'c == "xgb_rounds" or c.startswith("weight_")' in source
