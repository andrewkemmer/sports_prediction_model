"""Feature-name consistency regression tests (production call paths).

Background: scikit-learn >= 1.6 + LightGBM < 4.6 emits

    UserWarning: X does not have valid feature names, but LGBMClassifier
    was fitted with feature names

on EVERY ndarray predict(), because LightGBM auto-assigns Column_i names
even when fit() received a plain ndarray (lightgbm-org/LightGBM#6798). The
production Phase 5 moneyline walk-forward OOF printed 124 such warnings
(one per fold) plus 124 LGBMRegressor warnings in the distribution OOF and
two more in the Phase 11 slate serving path.

Fix contract (mirrors MLB's model-specific routing):
  * tree members (LightGBM / XGBoost) are FIT and PREDICTED on the NAMED
    tree-view DataFrame — one authoritative matrix builder per family
    (moneyline.member_fit_input / features.tree_view), never a raw
    ndarray at one end.
  * linear members (logistic / MLP) stay on the preprocessor's ndarray at
    both ends (no feature names either way — consistent).

These tests escalate the exact production UserWarning to an ERROR and run
the REAL production call chains (walk_forward_oof -> member fit/predict,
fit_final_models -> predict_slate, ScoreRegressor.fit/predict). A future
regression that reintroduces a DataFrame/ndarray mismatch at either end
fails here — no helper-only shortcut.

Run:  python3 test_feature_name_consistency.py
"""
from __future__ import annotations

import contextlib
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


import config  # noqa: E402
import features as feat_mod  # noqa: E402
import moneyline as ml_mod  # noqa: E402
import distributions as dist_mod  # noqa: E402

WARNING_MSG = "X does not have valid feature names"


@contextlib.contextmanager
def warnings_as_errors():
    """Escalate the production feature-name warning to an exception."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=WARNING_MSG)
        yield


# ---------------------------------------------------------------------------
# Synthetic decided-game frame with the full production feature engine.
# (Same generator discipline as test_production.py — pure python, no network.)
# ---------------------------------------------------------------------------
def _synthetic_games(n_weeks: int = 12, start="2023-09-01",
                     season: int = 2023) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    teams = [f"T{i:02d}" for i in range(8)]
    rows = []
    day = pd.Timestamp(start)
    gid = 0
    for wk in range(n_weeks):
        order = teams.copy()
        rng.shuffle(order)
        for i in range(0, len(order), 2):
            h, a = order[i], order[i + 1]
            rows.append({
                "game_id": f"G{season}-{gid:04d}", "season": season,
                "week": wk + 1,
                "gameday": (day + pd.Timedelta(days=wk * 7 + (gid % 3))
                            ).strftime("%Y-%m-%d"),
                "home_team": h, "away_team": a,
                "home_score": int(rng.integers(0, 45)),
                "away_score": int(rng.integers(0, 45)),
                "game_type": "REG",
                "roof": rng.choice(["outdoors", "dome"]),
                "div_game": 0, "stadium": "Unknown Stadium",
                "gametime": "13:00",
            })
            gid += 1
    return pd.DataFrame(rows)


print("\n== 1. Moneyline walk-forward OOF (production path, warnings-as-errors) ==")
games = pd.concat([
    _synthetic_games(n_weeks=10, start="2018-09-01", season=2018),   # warmup
    _synthetic_games(n_weeks=12, start="2023-09-01", season=2023),   # core
], ignore_index=True)
feats = feat_mod.build_game_features(games, pbp=None)
feats = feats.sort_values("gameday").reset_index(drop=True)

oof = None
with warnings_as_errors():
    try:
        ml = ml_mod.walk_forward_oof(feats)
        oof = ml["oof"]
        check("walk_forward_oof completes with zero feature-name warnings", True)
    except Warning as exc:
        check("walk_forward_oof completes with zero feature-name warnings",
              False, f"warning escalated: {exc}")
    except Exception as exc:  # noqa: BLE001
        check("walk_forward_oof completes with zero feature-name warnings",
              False, f"error: {exc}")

core_n = int((pd.to_numeric(feats["season"]) >= config.OOF_FIRST_SEASON).sum())
check("OOF rows == core-season games (warmup never evaluated)",
      oof is not None and len(oof) == core_n,
      f"oof={len(oof) if oof is not None else None} core={core_n}")
p_cols = [f"p_{n}" for n in config.ENSEMBLE_MEMBERS]
check("all member probability columns present",
      oof is not None and all(c in oof.columns for c in p_cols))
if oof is not None:
    ok = all(np.isfinite(oof[c].to_numpy(float)).all() for c in p_cols)
    check("all member OOF probabilities finite", ok)

# ---------------------------------------------------------------------------
print("\n== 2. Representation contract ==")
tree_cols = list(feat_mod.tree_view(feats).columns)
lin_cols = list(feat_mod.linear_view(feats).columns)

# The lightgbm member of the LAST fold's fit must carry the named tree view.
# Re-fit one fold exactly as the production loop does and assert the
# fitted feature names match the tree-view columns.
train = feats.iloc[:-8]
model = ml_mod._make_member("lightgbm")
X_tr = ml_mod.member_matrix("lightgbm", train)
model.fit(ml_mod.member_fit_input("lightgbm", X_tr, None),
          train["home_win"].astype(int).to_numpy())
fitted_names = list(getattr(model, "feature_names_in_", []))
check("lightgbm fitted with tree-view feature names",
      fitted_names == tree_cols,
      f"fitted={fitted_names[:4]}... expected={tree_cols[:4]}...")

# member_fit_input keeps trees on the named frame and linear members on
# the preprocessor ndarray — at BOTH fit and predict.
check("tree member_fit_input returns named DataFrame",
      isinstance(ml_mod.member_fit_input("lightgbm", X_tr, None), pd.DataFrame))
pre = ml_mod.TrainFoldPreprocessor().fit(ml_mod.member_matrix("logistic", train))
lin_in = ml_mod.member_fit_input("logistic",
                                 ml_mod.member_matrix("logistic", train), pre)
check("linear member_fit_input returns ndarray", isinstance(lin_in, np.ndarray))
check("linear ndarray width == linear_view width",
      lin_in.shape[1] == len(lin_cols))

# ---------------------------------------------------------------------------
print("\n== 3. Final full-history fit + slate prediction (warnings-as-errors) ==")
models, _ = None, None
with warnings_as_errors():
    try:
        models, _pre = ml_mod.fit_final_models(feats)
        check("fit_final_models completes with zero feature-name warnings", True)
    except Warning as exc:
        check("fit_final_models completes with zero feature-name warnings",
              False, f"warning escalated: {exc}")

slate = feats.tail(8).copy()  # stand-in slate rows with identical schema
with warnings_as_errors():
    try:
        p = ml_mod.predict_slate(models, slate, ml["member_weights"])
        check("predict_slate completes with zero feature-name warnings", True)
    except Warning as exc:
        check("predict_slate completes with zero feature-name warnings",
              False, f"warning escalated: {exc}")
check("slate predictions finite and in (0,1)",
      np.isfinite(p).all() and ((p > 0) & (p < 1)).all())

# fitted feature names on every tree member match the tree view
for name in ("lightgbm", "xgboost"):
    fitted = list(getattr(models[name]["model"], "feature_names_in_", []))
    check(f"{name} final-fit feature names == tree_view columns",
          fitted == tree_cols)

# ---------------------------------------------------------------------------
print("\n== 4. ScoreRegressor (distribution model, warnings-as-errors) ==")
with warnings_as_errors():
    try:
        reg = dist_mod.ScoreRegressor().fit(feats)
        mu_h, mu_a = reg.predict(feats)
        check("ScoreRegressor fit+predict with zero feature-name warnings", True)
    except Warning as exc:
        check("ScoreRegressor fit+predict with zero feature-name warnings",
              False, f"warning escalated: {exc}")
check("mu_h/mu_a finite", np.isfinite(mu_h).all() and np.isfinite(mu_a).all())
lgb_names = list(getattr(reg.away_model, "feature_names_in_", []))
check("LGBMRegressor fitted with tree-view feature names", lgb_names == tree_cols)

# ---------------------------------------------------------------------------
print("\n== 5. Model-output invariance (ndarray vs named-DataFrame fit) ==")
# The representation change must not alter model outputs: LightGBM fits on
# an ndarray and on the matching named DataFrame must agree bit-for-bit.
Xa = X_tr.to_numpy(dtype=np.float64)
m_arr = ml_mod._make_member("lightgbm").fit(Xa, train["home_win"].astype(int).to_numpy())
m_df = ml_mod._make_member("lightgbm").fit(X_tr, train["home_win"].astype(int).to_numpy())
p_arr = m_arr.predict_proba(Xa)[:, 1]
p_df = m_df.predict_proba(X_tr)[:, 1]
check("lightgbm ndarray-fit == DataFrame-fit predictions",
      np.array_equal(p_arr, p_df),
      f"max abs diff {np.nanmax(np.abs(p_arr - p_df)):.3g}")

# ---------------------------------------------------------------------------
print("\n== 6. Guard sanity: a deliberate mismatch IS caught ==")
# Fit on the named frame, predict on a raw ndarray — exactly the mismatch
# class that produced the production warnings. The harness must convert the
# sklearn UserWarning into an exception here, proving section 1/3/4 would
# fail if the production paths ever regress to it.
mismatch_model = ml_mod._make_member("lightgbm").fit(
    X_tr, train["home_win"].astype(int).to_numpy())
caught = False
with warnings_as_errors():
    try:
        mismatch_model.predict_proba(X_tr.to_numpy(dtype=np.float64))
    except Warning:
        caught = True
check("warnings-as-errors guard fires on a real df/ndarray mismatch", caught)

# ---------------------------------------------------------------------------

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
