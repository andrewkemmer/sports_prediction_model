"""Regression test: LightGBM fit/predict feature-name consistency.

Background (the bug): the production OOF pipeline emitted

    UserWarning: X does not have valid feature names, but LGBMClassifier
    was fitted with feature names

because fit and prediction constructed their feature matrices at separate
call sites. A model fitted with a pandas DataFrame (feature names recorded
in ``feature_name_``) could be predicted with a NumPy array — or vice
versa — and sklearn validates the representation mismatch at predict time.

The fix routes every LightGBM (and ensemble-member) fit and predict through
a SINGLE ndarray builder so both sides always receive the identical
representation. This test proves the invariant holds and would catch a
regression that reintroduces split fit/predict construction.

Run:  python3 test_feature_name_consistency.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail and not cond else ""))


# ---------------------------------------------------------------------------
def _synthetic_view_df(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """A small frame with every tree-view/linear-view feature column filled,
    so member_matrix / tree_view produce real (non-degenerate) matrices."""
    import config
    import features as feat_mod

    rng = np.random.default_rng(seed)
    tree_cols = [c for c in config.FEATURE_COLUMNS if c != "is_home"]
    side_pool = sorted(set(
        c[:-5] for c in config.FEATURE_COLUMNS
        if c.endswith("_home") and c[:-5] + "_away" in config.FEATURE_COLUMNS
    ))
    df = pd.DataFrame({
        "elo_home": rng.normal(1500, 40, n),
        "elo_away": rng.normal(1500, 40, n),
        "win_pct_home": rng.uniform(0.2, 0.8, n),
        "win_pct_away": rng.uniform(0.2, 0.8, n),
        "ewm_net_pts_home": rng.normal(0, 5, n),
        "ewm_net_pts_away": rng.normal(0, 5, n),
    })
    for c in tree_cols:
        if c not in df.columns:
            df[c] = rng.normal(0, 1, n)
    for base in side_pool:
        for suffix in ("_home", "_away"):
            if base + suffix not in df.columns:
                df[base + suffix] = rng.normal(0, 1, n)
    # Ensure every view column exists and is numeric
    for view in (feat_mod.tree_view(df), feat_mod.linear_view(df)):
        for c in view.columns:
            if c not in df.columns:
                df[c] = rng.normal(0, 1, n)
    return df


def _assert_no_feature_name_warning(model, X) -> None:
    """Predict and fail if sklearn emits the feature-name mismatch warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            if hasattr(model, "predict_proba"):
                model.predict_proba(X)
            else:
                model.predict(X)
        except UserWarning as exc:
            raise AssertionError(f"feature-name warning leaked: {exc}")


# ---------------------------------------------------------------------------
print("\n== 1. Moneyline ensemble member fit/predict representation ==")
import config  # noqa: E402
import features as feat_mod  # noqa: E402
import moneyline as ml_mod  # noqa: E402
import distributions as dist_mod  # noqa: E402

df = _synthetic_view_df()
y = (rng := np.random.default_rng(config.RANDOM_SEED)).integers(0, 2, len(df))

for name in config.ENSEMBLE_MEMBERS:
    X_raw = ml_mod.member_matrix(name, df)
    pre = (ml_mod.TrainFoldPreprocessor().fit(X_raw)
           if name in ml_mod.LINEAR_MEMBERS else None)
    X_fit = ml_mod.member_matrix_ndarray(name, X_raw, pre)
    model = ml_mod._make_member(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X_fit, y)
    check(f"{name}: fit accepts the member ndarray (no warnings)", True)
    # The EXACT ndarray a prediction call would build must match fit-time.
    X_pred = ml_mod.member_matrix_ndarray(name, X_raw, pre)
    check(f"{name}: fit/predict matrices identical representation",
          type(X_fit) is type(X_pred) is np.ndarray
          and X_fit.shape == X_pred.shape)
    try:
        _assert_no_feature_name_warning(model, X_pred)
        check(f"{name}: predict emits no feature-name warning", True)
    except AssertionError as exc:
        check(f"{name}: predict emits no feature-name warning", False, str(exc))

# Linear members through their fitted preprocessor — same builder both sides.
for name in ("logistic", "mlp"):
    X_raw = ml_mod.member_matrix(name, df)
    pre = ml_mod.TrainFoldPreprocessor().fit(X_raw)
    X_fit = ml_mod.member_matrix_ndarray(name, X_raw, pre)
    model = ml_mod._make_member(name)
    model.fit(X_fit, y)
    X_pred = ml_mod.member_matrix_ndarray(name, X_raw, pre)
    try:
        _assert_no_feature_name_warning(model, X_pred)
        check(f"{name} (fitted pre): predict emits no warning", True)
    except AssertionError as exc:
        check(f"{name} (fitted pre): predict emits no warning", False, str(exc))

# ---------------------------------------------------------------------------
print("\n== 2. ScoreRegressor (LGBMRegressor) fit/predict consistency ==")
scores = pd.DataFrame({
    "home_score": rng.normal(24, 7, len(df)).round().astype(float),
    "away_score": rng.normal(21, 7, len(df)).round().astype(float),
})
reg_df = pd.concat([df.reset_index(drop=True),
                    scores.reset_index(drop=True)], axis=1)
reg = dist_mod.ScoreRegressor()
with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    reg.fit(reg_df)
check("ScoreRegressor.fit accepts the _matrix ndarray (no warnings)", True)

mu_h, mu_a = None, None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mu_h, mu_a = reg.predict(reg_df)
    check("ScoreRegressor.predict emits no feature-name warning", True)
except AssertionError as exc:
    check("ScoreRegressor.predict emits no feature-name warning", False, str(exc))
check("ScoreRegressor.predict returns finite per-side means",
      mu_h is not None and np.all(np.isfinite(mu_h))
      and np.all(np.isfinite(mu_a)))

# ---------------------------------------------------------------------------
print("\n== 3. Static cause check: the old split fit/predict pattern is gone ==")
import inspect  # noqa: E402

ml_src = inspect.getsource(ml_mod.walk_forward_oof)
check("walk_forward_oof fits members via member_matrix_ndarray (single builder)",
      "member_matrix_ndarray(name, train, pre)" in ml_src
      and ".to_numpy(dtype=np.float64), y_train" not in ml_src)
final_src = inspect.getsource(ml_mod.fit_final_models)
check("fit_final_models fits members via member_matrix_ndarray",
      "member_matrix_ndarray" in final_src
      and "X_raw.to_numpy(dtype=np.float64), y)" not in final_src)
pred_src = inspect.getsource(ml_mod._member_predict_proba)
check("_member_predict_proba routes through member_matrix_ndarray",
      "member_matrix_ndarray" in pred_src)

dist_src = inspect.getsource(dist_mod.ScoreRegressor)
check("ScoreRegressor fit and predict share _matrix (single builder)",
      dist_src.count("self._matrix(") == 2)  # once in fit, once in predict

# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED checks:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("ALL FEATURE-NAME CONSISTENCY CHECKS PASSED")
