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
  * linear members (elastic-net / MLP) stay on the preprocessor's ndarray at
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

# Import-shape pin (the quirk test_leave_events._patch_dd documents): when
# the sport root is importable (pytest from nfl-backend/, python -m, etc.),
# ``backend`` resolves as a namespace package and the production modules'
# ``from backend import config`` binds the ``backend.config`` module object —
# a DIFFERENT object from this file's ``import config``. The subset state
# these checks mutate must land on the instance the production views read,
# or the views silently answer from the un-subset contract.
config = feat_mod.config

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
# Ordinary under-minimum calendar windows are intentionally skipped; the
# accepted OOF population is the union of the generated fold validation rows.
expected_oof_n = int(ml["fold_table"]["n_val"].sum()) if ml and len(ml.get("fold_table", [])) else 0
check("OOF rows match accepted fold population (warmup never evaluated)",
      oof is not None and len(oof) == expected_oof_n,
      f"oof={len(oof) if oof is not None else None} accepted={expected_oof_n} core={core_n}")
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

# The tree view appends the categorical team-ID pair AFTER the served
# contract; the linear view never sees it (MLB TREE_CATEGORICAL_COLS parity).
check("tree view appends home/away team-ID pair last",
      tree_cols[-len(config.TREE_CATEGORICAL_COLS):] == config.TREE_CATEGORICAL_COLS,
      f"tail={tree_cols[-3:]}")
check("linear view excludes the team-ID pair",
      all(c not in lin_cols for c in config.TREE_CATEGORICAL_COLS))
check("tree numeric prefix is the served contract projection",
      tree_cols[:-len(config.TREE_CATEGORICAL_COLS)]
      == [c for c in config.active_moneyline_feature_cols() if c in feats.columns])
check("team IDs are stable ints with UNK fallback",
      # 2026-10-08 coverage audit, HIGH: historical franchise spellings
      # resolve to their MODERN identity (one franchise, one ID) instead of
      # the unknown bucket; only genuinely unrecognized labels hit UNK.
      config.team_category_id("KC") == config.NFL_TEAM_ID["KC"]
      and config.team_category_id("OAK") == config.NFL_TEAM_ID["LV"]
      and config.team_category_id("SD") == config.NFL_TEAM_ID["LAC"]
      and config.team_category_id("JST") == config.UNK_TEAM_ID
      and config.team_category_id(None) == config.UNK_TEAM_ID)
_ids = feat_mod.team_category_ids(feats)
check("team_category_ids maps home/away from the frame",
      list(_ids.columns) == config.TREE_CATEGORICAL_COLS
      and set(_ids["home_team_id"].unique())
      <= set(config.NFL_TEAM_ID.values()) | {config.UNK_TEAM_ID})

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
pre = ml_mod.TrainFoldPreprocessor().fit(ml_mod.member_matrix("elasticnet", train))
lin_in = ml_mod.member_fit_input("elasticnet",
                                 ml_mod.member_matrix("elasticnet", train), pre)
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
        check("ScoreRegressor preserves native LightGBM NaN routing",
              not hasattr(reg, "feature_medians"))
        check("ScoreRegressor carries the team-ID pair (run-line parity)",
              list(reg.feature_columns) == feat_mod.tree_numeric_columns()
              + config.TREE_CATEGORICAL_COLS)
        # Guardrail: the run line and the binary moneyline must consume the
        # SAME feature set (MLB strict-parity structure). The run line owns no
        # list of its own, so this is checked, not assumed.
        _act = list(config.active_moneyline_feature_cols())
        check("run line consumes the binary moneyline contract, in order",
              list(reg.feature_columns[:len(_act)]) == _act,
              f"run-line={len(reg.feature_columns)} contract={len(_act)}")
        check("run line appends exactly the team-ID pair, nothing else",
              list(reg.feature_columns[len(_act):]) == list(config.TREE_CATEGORICAL_COLS))
        check("run-line and moneyline contracts are the same length set",
              set(reg.feature_columns) == set(_act) | set(config.TREE_CATEGORICAL_COLS))
        # A divergent run-line list must be rejected, not silently accepted.
        _diverged = list(_act)[:-1] + ["not_a_moneyline_feature"]
        try:
            dist_mod._assert_run_line_contract(_diverged)
            check("a diverged run-line feature set is rejected at build time", False,
                  "guard did not raise")
        except RuntimeError:
            check("a diverged run-line feature set is rejected at build time", True)
        try:
            dist_mod._assert_run_line_contract(_act + ["home_team_id"])
            check("a run-line set missing the away ID is rejected", False,
                  "guard did not raise")
        except RuntimeError:
            check("a run-line set missing the away ID is rejected", True)
    except Warning as exc:
        check("ScoreRegressor fit+predict with zero feature-name warnings",
              False, f"warning escalated: {exc}")
check("mu_h/mu_a finite", np.isfinite(mu_h).all() and np.isfinite(mu_a).all())
lgb_names = list(getattr(reg.away_model, "feature_names_in_", []))
lgb_names = list(getattr(reg.away_model, "feature_names_in_", []))
check("LGBMRegressor fitted with the numeric contract + team-ID pair",
      lgb_names == feat_mod.tree_numeric_columns() + config.TREE_CATEGORICAL_COLS)
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
print("\n== 7. One master list (contract, routing, candidates) ==")
# The binary moneyline defines config.MONEYLINE_FEATURE_COLS and every other
# consumer PULLS it (run line via features.tree_view, the RFE via
# _trial_space, monitoring/manifest via the same accessor). These checks
# assert the structural properties that make duplicate columns and drift
# impossible — nothing here is duplicate PROTECTION; the views have none.


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


contract = list(config.MONEYLINE_FEATURE_COLS)
check("contract is declared once with unique names",
      len(contract) > 0 and len(set(contract)) == len(contract),
      f"n={len(contract)}")
check("raw per-side routing is a subset of the TRIABLE pool",
      set(config.RAW_PER_SIDE_COLS) <= set(config.KNOWN_FEATURE_COLS),
      f"n_raw={len(config.RAW_PER_SIDE_COLS)} n_pool={len(config.KNOWN_FEATURE_COLS)}")
check("raw per-side routing is pool-served or declared candidates",
      all(f in contract or f in config.RFE_CANDIDATE_COLS
          for f in config.RAW_PER_SIDE_COLS),
      "served raws route now; candidate raws route after adoption")
check("candidate manifest documents every declared candidate",
      not __import__("manifest").validate(),
      "manifest.validate() includes candidate parity")
check("pool == contract + declared candidates (derived, never re-listed)",
      config.KNOWN_FEATURE_COLS == list(dict.fromkeys(
          contract + [c for c in config.RFE_CANDIDATE_COLS if c not in contract])))
config.set_feature_subset(["elo_diff", "elo_diff", "is_home", "elo_away"])
subset = list(config.active_moneyline_feature_cols())
check("subsets rebuild canonically from the pool (duplicates impossible)",
      subset == [c for c in config.KNOWN_FEATURE_COLS
                 if c in {"elo_diff", "is_home", "elo_away"}],
      f"got {subset}")
sub_tree = list(feat_mod.tree_view(feats).columns)
sub_lin = list(feat_mod.linear_view(feats).columns)
config.reset_feature_subset()
exp_tree = ([c for c in subset if c in feats.columns]
            + config.TREE_CATEGORICAL_COLS)
exp_lin = [c for c in subset
           if c not in config.RAW_PER_SIDE_COLS and c in feats.columns]
check("subset views stay unique and pool-ordered",
      len(set(sub_tree)) == len(sub_tree) and len(set(sub_lin)) == len(sub_lin)
      and sub_tree == exp_tree
      and sub_lin == exp_lin,
      f"tree {sub_tree} != {exp_tree} | lin {sub_lin} != {exp_lin}")
check("unknown subset names are rejected, not silently intersected",
      _raises(lambda: config.set_feature_subset(["not_a_feature"])))
check("active contract returns to the full list after reset",
      config.active_moneyline_feature_cols() == contract)

# ---------------------------------------------------------------------------

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    if __name__ == "__main__":
        sys.exit(1)
    # Imported (pytest): sys.exit here would abort COLLECTION with an
    # INTERNALERROR instead of naming the failing check — raise so the
    # failure surfaces as a normal collection error.
    raise AssertionError(f"feature-name consistency checks failed: {FAIL}")


# CRPS whole-CDF production contract (2026-10-11). These tests exercise the
# same public interfaces the pipelines call, including outcome poisoning.
def _crps_test_frame():
    import numpy as np
    import pandas as pd
    import distributions as dist
    rng = np.random.default_rng(817)
    n = 360
    h = rng.uniform(100., 120., n) if dist.DISTRIBUTION_SPORT == "nba" else rng.uniform(2., 5., n)
    a = rng.uniform(100., 120., n) if dist.DISTRIBUTION_SPORT == "nba" else rng.uniform(2., 5., n)
    hs, aws = rng.poisson(h), rng.poisson(a)
    if dist.DISTRIBUTION_SPORT != "nfl": hs = hs + (hs == aws)
    frame = pd.DataFrame({"home_score": hs, "away_score": aws})
    if dist.DISTRIBUTION_SPORT == "mlb":
        frame["home_expected_runs"], frame["away_expected_runs"] = h, a
        frame["game_date"] = pd.date_range("2025-01-01", periods=n)
        frame["fold_idx"] = np.arange(n) // 60
    else:
        frame["mu_h"], frame["mu_a"] = h, a
        frame["gameday"] = pd.date_range("2025-01-01", periods=n)
        frame["fold_id"] = np.arange(n) // 60
    return frame


def test_crps_point_mass_matches_absolute_error_and_correct_cdf_orientation():
    import distributions as dist
    import numpy as np
    support = np.arange(-3, 4)
    cdf = (support[None, :] >= np.array([0., 1., -1.])[:, None]).astype(float)
    np.testing.assert_allclose(dist.lattice_crps(cdf, support, [2., 1., -2.]), [2., 0., 1.])
    # Reversing I(y <= k) would incorrectly score a perfect point mass.
    np.testing.assert_allclose(dist.lattice_crps(cdf, support, [0., 1., -1.]), 0.)


def test_crps_full_pipeline_future_poisoning_and_row_order_invariance():
    import distributions as dist
    import numpy as np
    frame = _crps_test_frame()
    priced, bundle = dist.causal_distribution_frame(frame)
    poisoned = frame.copy()
    poisoned.loc[poisoned.index >= 300, "home_score"] += 20
    changed, _ = dist.causal_distribution_frame(poisoned)
    cols = [c for c in priced if c.startswith("p_")]
    np.testing.assert_allclose(priced.loc[:299, cols], changed.loc[:299, cols], atol=0., rtol=0.)
    shuffled = frame.sample(frac=1., random_state=81)
    reordered, _ = dist.causal_distribution_frame(shuffled)
    np.testing.assert_allclose(priced[cols], reordered.reindex(frame.index)[cols], atol=1e-10)
    assert bundle["method"] == "causal_lattice_crps"
    assert bundle["scope"] == "whole_cdf"
    assert bundle["origins"][0]["n_prior"] == 0
    assert all(o["n_prior"] < len(frame) for o in bundle["origins"])


def test_crps_cdf_pushes_moneyline_fair_half_and_slate_agree():
    import distributions as dist
    import numpy as np
    frame = _crps_test_frame()
    priced, bundle = dist.causal_distribution_frame(frame)
    np.testing.assert_allclose(priced.p_home_win_derived + priced.p_away_win_derived + priced.p_tie, 1., atol=1e-12)
    if dist.DISTRIBUTION_SPORT != "nfl": np.testing.assert_allclose(priced.p_tie, 0.)
    assert np.all(np.mod(priced.fair_total, 1.) == .5)
    assert np.all(np.mod(priced.fair_spread, 1.) == .5)
    last = bundle["origins"][-1]
    forecast = dist._distribution_price(frame.iloc[300:].drop(columns=["home_score", "away_score"]), last["maps"])
    cols = [c for c in forecast if c.startswith("p_")]
    np.testing.assert_allclose(priced.iloc[300:][cols], forecast[cols], atol=1e-12)
    for kind, (support, cdf) in dist._distribution_cdfs(frame, bundle["maps"]).items():
        assert np.all(np.diff(cdf, axis=1) >= -1e-12)
        assert np.all((cdf >= 0.) & (cdf <= 1.))
        assert np.all(cdf[:, -1] >= 1 - 1e-8)
    if dist.DISTRIBUTION_SPORT != "mlb":
        # Fair aliases must refer to their recalibrated CDF, not stale integers.
        for row in priced.itertuples():
            assert abs(getattr(row, dist._grid_key("p_over", row.fair_total)) - row.p_over_fair) < 1e-12
            assert abs(getattr(row, dist._grid_key("p_home_cover", row.fair_spread)) - row.p_cover_fair) < 1e-12
        z = dist._grid_key("p_home_cover", 0)
        np.testing.assert_allclose(priced[z], priced.p_home_win_derived, atol=1e-12)


def test_crps_refuses_overlapping_origins_missing_dates_and_invalid_scores():
    import distributions as dist
    import pytest
    frame = _crps_test_frame()
    date = "game_date" if dist.DISTRIBUTION_SPORT == "mlb" else "gameday"
    bad = frame.copy(); bad.loc[60, date] = bad.loc[10, date]
    with pytest.raises(ValueError, match="overlap"):
        dist.causal_distribution_frame(bad)
    bad = frame.copy(); bad.loc[0, date] = None
    with pytest.raises(ValueError, match="dates"):
        dist.causal_distribution_frame(bad)
    bad = frame.copy()
    key = "home_expected_runs" if dist.DISTRIBUTION_SPORT == "mlb" else "mu_h"
    bad.loc[0, key] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        dist.causal_distribution_frame(bad)
