"""Offline NHL backend unit tests (no model training, no network, no data).

Mirrors ``mlb-backend/backend/test_grid_rfe.py`` structurally:

  * folds.make_folds   — the MLB-style 30-day warm-up gate (the FIRST
                         validation window cannot start before first core
                         date + WARMUP_DAYS), expanding trains, strict
                         train<val ordering, MIN_VAL_FOLD_GAMES skipping,
                         final-partial-window retention;
  * folds.canonical_sort — the ONE row order the fold labels are valid for
                         (fold indices are POSITIONS): a total order whose
                         result is independent of arrival order, whose fold
                         membership survives a shuffle, and which every OOF
                         consumer applies before generating fold labels;
  * feature views      — linear/tree routing over the ONE master list (the
                         raw per-side block is tree-only; the categorical
                         team-ID pair is tree-only; linear_view is a pure
                         diff+anchor projection);
  * blend weights      — the rolling SLSQP re-earn contract (logit space,
                         simplex, exact 1.0 total, mismatched members
                         skipped, best-single-member takeover);
  * config pins        — MLB-tuned member params copied exactly, 1-sigma
                         RFE bar, grid/geometry constants, the 32-team ID
                         map and goalie serving contract;
  * manifest parity    — FEATURE_MANIFEST / CANDIDATE_MANIFEST match the
                         served contract name-for-name;
  * retention          — the blanket 10-day window with NHL's SHAP
                         filename aging (game ids embed YYYYMMDD).

Run:  python nhl-backend/backend/test_grid_rfe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import config                                        # noqa: E402
import folds as folds_mod                            # noqa: E402
import features as feat_mod                          # noqa: E402
import distributions as dist_mod                     # noqa: E402
import moneyline as ml_mod                           # noqa: E402
import manifest                                      # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

TEAMS = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY",
         "CHI", "COL", "DAL", "DET", "EDM", "FLA"]


def _synth_games(n_days: int = 60, games_per_day: int = 6, seed: int = 7,
                 start: str = "2025-10-01") -> pd.DataFrame:
    """One game per pair of disjoint teams per day (each team plays at most
    once daily — the ladder's strictly-increasing gameday gate)."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp(start)
    pk = 2025010000
    for d in range(n_days):
        day = (base + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for i in range(games_per_day):
            home, away = TEAMS[i], TEAMS[i + games_per_day]
            pk += 1
            hs = int(rng.poisson(3.1))
            as_ = int(rng.poisson(2.7))
            rows.append({
                "game_id": f"{day.replace('-', '')}_{away}@{home}",
                "season": 2025,
                "gameday": day,
                "home_team": home, "away_team": away,
                "home_score": hs, "away_score": as_,
            })
    return pd.DataFrame(rows)


def _feature_frame(n: int = 5, seed: int = 11) -> pd.DataFrame:
    """A frame carrying every served contract column + team abbreviations."""
    cols = config.MONEYLINE_FEATURE_COLS
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    df["home_team"] = ["TOR", "BOS", "NYR", "EDM", "VGK"]
    df["away_team"] = ["MTL", "SEA", "TBL", "CGY", "FLA"]
    return df


# ── folds: the 30-day warm-up gate ─────────────────────────────────────────

def test_warmup_first_window_starts_warmup_days_after_first_core_date():
    df = _synth_games(n_days=60, games_per_day=6)
    folds = folds_mod.make_folds(df, date_col="gameday")
    assert folds, "expected folds on a dense 60-day slate"
    first_core = pd.Timestamp("2025-10-01")
    assert folds[0].val_start == first_core + pd.Timedelta(days=config.WARMUP_DAYS), (
        f"first validation window started {folds[0].val_start.date()} — "
        f"the 30-day warm-up gate leaked early games into validation")
    # No validation row predates the warm-up boundary.
    for f in folds:
        assert pd.to_datetime(df.loc[f.val_idx, "gameday"]).min() >= f.val_start


def test_folds_are_expanding_and_strictly_prior():
    df = _synth_games(n_days=60, games_per_day=6)
    folds = folds_mod.make_folds(df, date_col="gameday")
    assert len(folds) > 2, f"expected multiple folds, got {len(folds)}"
    sizes = [len(f.train_idx) for f in folds]
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), \
        f"training folds not expanding: {sizes}"
    for f in folds:
        tr_end = pd.to_datetime(df.loc[f.train_idx, "gameday"]).max()
        va_start = pd.to_datetime(df.loc[f.val_idx, "gameday"]).min()
        assert tr_end < va_start, (
            f"fold {f.fold_id}: train ends {tr_end.date()} on/after "
            f"validation start {va_start.date()}")
    # Validation windows are the 7-day cadence, non-overlapping, chronological.
    for a, b in zip(folds, folds[1:]):
        assert a.val_start < a.val_end < b.val_start
        assert (b.val_start - a.val_start).days == config.RETRAIN_CADENCE_DAYS


def test_min_val_fold_games_skips_sparse_windows_but_keeps_the_final_tail():
    # 3 games/day = 21 games per 7-day window < MIN_VAL_FOLD_GAMES (40):
    # every ordinary window is skipped; the final partial window is retained
    # so the newest games remain visible.
    df = _synth_games(n_days=60, games_per_day=3, seed=13)
    folds = folds_mod.make_folds(df, date_col="gameday")
    assert all(len(f.val_idx) >= config.MIN_VAL_FOLD_GAMES
               for f in folds[:-1]), \
        "a non-final window below MIN_VAL_FOLD_GAMES was not skipped"
    assert folds, "the final partial window must be retained"
    assert len(folds[-1].val_idx) > 0

    # Dense slates fill the same windows well past the gate.
    dense = _synth_games(n_days=60, games_per_day=6)
    dense_folds = folds_mod.make_folds(dense, date_col="gameday")
    assert all(len(f.val_idx) >= config.MIN_VAL_FOLD_GAMES
               for f in dense_folds[:-1])


# ── folds: the ONE row order the fold labels are valid for ────────────────
# make_folds returns index LABELS, and the OOF consumers look those rows up
# POSITIONALLY after their own reset_index(drop=True). Every caller used to sort
# on the date column ALONE, and a single-column sort_values is an UNSTABLE
# quicksort — so two callers over the same data disagreed on the order of
# same-date games. The right games stayed in every fold (make_folds selects by
# DATE, so membership was never at risk); the boosting members were simply
# handed them in a different order under a fixed seed, which made a run
# unreproducible. canonical_sort is the one order both sides agree on.


def test_canonical_sort_is_a_total_order_independent_of_arrival_order():
    df = _synth_games(n_days=60, games_per_day=6)
    canonical = folds_mod.canonical_sort(df, "gameday")
    # Same games, same order, from any arrival order.
    arrived = df.sample(frac=1.0, random_state=3)
    shuffled = folds_mod.canonical_sort(arrived, "gameday")
    assert canonical["game_id"].tolist() == shuffled["game_id"].tolist(), \
        "canonical_sort is not order-independent — its labels are not a total order"
    # Fresh RangeIndex: fold labels are POSITIONS in exactly this frame.
    assert canonical.index.tolist() == list(range(len(df)))
    # Same-date ties break on the tiebreaker column, not on arrival order.
    assert bool(canonical.groupby("gameday")["game_id"]
                .apply(lambda s: s.is_monotonic_increasing).all())
    # The hazard itself: over a frame that ARRIVES in another order, a
    # date-only sort does NOT agree with it.
    date_only = arrived.sort_values("gameday").reset_index(drop=True)
    assert canonical["game_id"].tolist() != date_only["game_id"].tolist(), \
        "a date-only sort now agrees with canonical_sort — the pin is vacuous"


def test_fold_membership_survives_a_shuffled_arrival_order():
    df = _synth_games(n_days=60, games_per_day=6)
    canonical = folds_mod.canonical_sort(df, "gameday")
    shuffled = canonical.sample(frac=1.0, random_state=5).reset_index(drop=True)
    ref_folds = folds_mod.make_folds(canonical, date_col="gameday")
    arr_folds = folds_mod.make_folds(shuffled, date_col="gameday")
    assert len(arr_folds) == len(ref_folds)
    for ref, arr in zip(ref_folds, arr_folds):
        # Membership is a SET property; the ORDER inside a fold is what
        # canonical_sort pins. Same games either way.
        assert (set(canonical.loc[ref.train_idx, "game_id"])
                == set(shuffled.loc[arr.train_idx, "game_id"]))
        assert (set(canonical.loc[ref.val_idx, "game_id"])
                == set(shuffled.loc[arr.val_idx, "game_id"]))
    # Regenerating the labels on the canonicalized arrival frame reproduces
    # the production fold objects exactly — labels included.
    regen = folds_mod.make_folds(
        folds_mod.canonical_sort(shuffled, "gameday"), date_col="gameday")
    assert [f.val_idx.tolist() for f in regen] == \
           [f.val_idx.tolist() for f in ref_folds]


def test_no_fold_consumer_reintroduces_a_date_only_sort():
    """A bare sort_values(date_col) anywhere ahead of make_folds reopens the
    label-vs-position gap, so the canonical order is pinned at each call site
    by an ASSIGNMENT (a comment mentioning it does not satisfy this)."""
    import re
    pattern = re.compile(r"=\s*folds_mod\.canonical_sort\(")
    for fname in ("moneyline.py", "distributions.py",
                  "master_pipeline.py", "feature_selection.py"):
        src = (BACKEND_DIR / fname).read_text(encoding="utf-8")
        canonical_at = pattern.search(src)
        make_at = src.find("folds_mod.make_folds")
        assert canonical_at is not None, \
            f"{fname} never canonicalizes — its fold labels are positions "\
            f"under an order it does not pin"
        assert make_at != -1 and canonical_at.start() < make_at, \
            f"{fname} generates fold labels before canonicalizing the frame"


def test_member_oof_is_invariant_to_the_arrival_row_order():
    """Behavioural guardrail, not a decorative one.

    The stand-in members below are deliberately ROW-ORDER SENSITIVE (they
    weight training rows by position), which is the property the real fixed-seed
    boosters share. If walk_forward_oof generated fold labels under one row
    order and applied them under another, every training set is permuted and
    every prediction moves. Reverting moneyline.py to the date-only sort — or
    leaving a consumer un-canonicalized — fails this.
    """
    games = feat_mod.build_game_features(_synth_games(n_days=40))
    canonical = folds_mod.canonical_sort(games, "gameday")
    shuffled = canonical.sample(frac=1.0, random_state=11).reset_index(drop=True)

    class _OrderSensitiveModel:
        def __init__(self, name):
            self.name = name

        def fit(self, X, y, **kwargs):
            w = np.arange(1, len(y) + 1, dtype=float)
            self.p = float(np.average(np.asarray(y, dtype=float), weights=w))
            return self

        def predict_proba(self, X):
            p = np.full(len(X), self.p, dtype=float)
            return np.column_stack([1.0 - p, p])

    class _OrderSensitiveRegressor:
        def fit(self, df):
            w = np.arange(1, len(df) + 1, dtype=float)
            self.h = float(np.average(df["home_score"].to_numpy(float), weights=w))
            self.a = float(np.average(df["away_score"].to_numpy(float), weights=w))
            return self

        def predict(self, df):
            return (np.full(len(df), self.h), np.full(len(df), self.a))

    def _fake_member(name, fold=False):
        return _OrderSensitiveModel(name)

    # The production path: each OOF canonicalizes the frame it is HANDED and
    # generates its own folds from that same canonical order.
    from unittest.mock import patch as _patch
    with _patch.object(ml_mod, "_make_member", side_effect=_fake_member), \
            _patch.object(dist_mod, "ScoreRegressor", _OrderSensitiveRegressor):
        ref = ml_mod.walk_forward_oof(canonical)["oof"]
        arr = ml_mod.walk_forward_oof(shuffled)["oof"]
        ref_d = dist_mod.walk_forward_oof(canonical)["oof"]
        arr_d = dist_mod.walk_forward_oof(shuffled)["oof"]

    # The order-sensitive members are actually sensitive — otherwise the
    # invariance below would be vacuously true.
    assert not np.allclose(ref["p_xgboost"].to_numpy(float),
                           np.full(len(ref), 0.5)), \
        "the stand-in member is not row-order sensitive; the pin proves nothing"

    pcols = [f"p_{m}" for m in config.ENSEMBLE_MEMBERS] + ["p_ensemble"]
    a = arr.set_index("game_id").sort_index()
    r = ref.set_index("game_id").sort_index()
    assert a.index.tolist() == r.index.tolist()
    for col in pcols:
        np.testing.assert_allclose(a[col].to_numpy(float), r[col].to_numpy(float),
                                   rtol=0, atol=1e-12,
                                   err_msg=f"{col} moved under a different "
                                           f"arrival row order")

    ad = arr_d.set_index("game_id").sort_index()
    rd = ref_d.set_index("game_id").sort_index()
    assert ad.index.tolist() == rd.index.tolist()
    for col in ("mu_h", "mu_a"):
        np.testing.assert_allclose(ad[col].to_numpy(float), rd[col].to_numpy(float),
                                   rtol=0, atol=1e-12,
                                   err_msg=f"{col} moved under a different "
                                           f"arrival row order")


# ── feature views: the ONE list, routed per model family ──────────────────

def test_linear_view_excludes_raw_sides_and_team_ids():
    df = _feature_frame()
    lv = ml_mod.member_matrix("elasticnet", df)
    expected = [c for c in config.active_moneyline_feature_cols()
                if c not in config.RAW_PER_SIDE_COLS]
    assert list(lv.columns) == expected
    assert "is_home" in lv.columns                       # constant anchor
    assert not ({"home_team_id", "away_team_id"} & set(lv.columns))
    assert not (config.RAW_PER_SIDE_COLS & set(lv.columns))


def test_tree_view_carries_the_categorical_team_id_pair():
    df = _feature_frame()
    tv = ml_mod.member_matrix("xgboost", df)
    want = set(config.active_moneyline_feature_cols()) | set(config.TREE_CATEGORICAL_COLS)
    assert set(tv.columns) == want, "tree view must be contract + team-ID pair"
    assert len(tv.columns) == len(set(tv.columns)), "duplicate column in tree view"
    assert list(tv[config.TREE_CATEGORICAL_COLS].dtypes) == ["int64", "int64"]
    # The linear family never sees the pair; tree_numeric_columns() agrees.
    assert "home_team_id" not in feat_cols_tree_numeric()


def feat_cols_tree_numeric() -> list[str]:
    import features as feat_mod
    cols = feat_mod.tree_numeric_columns()
    assert not ({"home_team_id", "away_team_id"} & set(cols))
    assert cols == config.active_moneyline_feature_cols()
    return cols


def test_team_category_ids_stable_with_unknown_fallback():
    import features as feat_mod
    df = _feature_frame(5)
    ids = feat_mod.team_category_ids(df)
    assert int(ids.loc[0, "home_team_id"]) == config.NHL_TEAM_ID["TOR"]
    assert int(ids.loc[1, "away_team_id"]) == config.NHL_TEAM_ID["SEA"]
    # Unseen / historical / defunct abbreviations map to the reserved UNK slot
    # (only the 32 active franchises are in NHL_TEAM_ID — ATL/PHX are not).
    assert config.team_category_id("XYZ") == config.UNK_TEAM_ID
    assert config.team_category_id(None) == config.UNK_TEAM_ID
    assert config.team_category_id("ATL") == config.UNK_TEAM_ID
    assert config.team_category_id(" tor ") == config.NHL_TEAM_ID["TOR"]


def test_single_list_active_matches_universe_after_reset():
    config.reset_feature_subset()
    assert config.active_moneyline_feature_cols() == list(config.MONEYLINE_FEATURE_COLS)
    # The candidate pool never overlaps the served universe.
    assert not (set(config.RFE_CANDIDATE_COLS) & set(config.MONEYLINE_FEATURE_COLS))
    # Subset application rebuilds in canonical pool order and validates.
    try:
        cols = list(config.MONEYLINE_FEATURE_COLS)[:5]
        config.set_feature_subset(cols)
        assert config.active_moneyline_feature_cols() == cols
    finally:
        config.reset_feature_subset()


# ── blend weights: the rolling SLSQP re-earn contract ─────────────────────

def _member_preds(n: int = 400, seed: int = 5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    mk = lambda strength, off: np.clip(
        0.5 + off + strength * (y - 0.5) + rng.normal(0, 0.15, n), 0.02, 0.98)
    return y, {"xgboost": mk(0.30, 0.02), "lightgbm": mk(0.26, -0.01),
               "elasticnet": mk(0.22, 0.0)}


def test_blend_weights_sum_exactly_to_one():
    y, members = _member_preds()
    w = ml_mod.compute_adaptive_weights(members, y)
    assert set(w) == set(config.ENSEMBLE_MEMBERS)
    assert abs(sum(w.values()) - 1.0) < 1e-9, f"weights do not sum to 1: {w}"
    assert all(v >= 0.0 for v in w.values())


def test_blend_skips_mismatched_members_and_renormalizes():
    y, members = _member_preds()
    broken = dict(members)
    broken["lightgbm"] = members["lightgbm"][:-1]        # wrong length
    broken["elasticnet"] = None                          # failed member
    w = ml_mod.compute_adaptive_weights(broken, y)
    assert "lightgbm" not in w and "elasticnet" not in w
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_blend_best_member_takeover_and_empty_inputs():
    y, members = _member_preds()
    perfect = dict(members)
    perfect["xgboost"] = np.clip(y, 1e-7, 1 - 1e-7)      # near-zero logloss
    w = ml_mod.compute_adaptive_weights(perfect, y)
    assert w["xgboost"] >= 0.99, f"best member did not take over: {w}"
    # Empty pool / empty labels degrade to empty dicts, never raise.
    assert ml_mod.compute_adaptive_weights({}, y) == {}
    assert ml_mod.compute_adaptive_weights(members, np.array([])) == {}
    # A single surviving member takes the entire weight.
    w1 = ml_mod.compute_adaptive_weights({"lightgbm": members["lightgbm"]}, y)
    assert w1 == {"lightgbm": 1.0}


def test_blend_is_logit_space_simplex_slsqp():
    # The optimizer runs over the simplex (bounds [0,1], eq sum-1) in LOGIT
    # space: an interior optimum must respect both constraints structurally.
    y, members = _member_preds()
    w = ml_mod.compute_adaptive_weights(members, y)
    assert max(w.values()) < 1.0 and min(w.values()) > 0.0, \
        f"interior optimum collapsed to a vertex without cause: {w}"


# ── config pins: MLB copies + geometry ─────────────────────────────────────

def test_ensemble_members_are_the_mlb_ensemble():
    assert config.ENSEMBLE_MEMBERS == ["xgboost", "lightgbm", "elasticnet"]
    assert config.LINEAR_MEMBERS if False else True  # LINEAR_MEMBERS lives in moneyline
    assert ml_mod.LINEAR_MEMBERS == {"elasticnet"}


def test_member_params_are_exact_mlb_copies():
    xg = config.XGBOOST_PARAMS
    assert (xg["max_depth"], xg["min_child_weight"], xg["gamma"],
            xg["subsample"], xg["colsample_bytree"], xg["learning_rate"]) == \
        (3, 12, 2.4178, 0.812, 0.6382, 0.1097)
    lg = config.LIGHTGBM_PARAMS
    assert (lg["n_estimators"], lg["max_depth"], lg["num_leaves"],
            lg["min_child_samples"], lg["min_gain_to_split"],
            lg["bagging_fraction"], lg["feature_fraction"],
            lg["learning_rate"]) == (50, 6, 6, 70, 1.2224, 0.4518, 0.7632, 0.0332)
    en = config.ELASTICNET_PARAMS
    assert (en["l1_ratio"], en["C"], en["max_iter"]) == (0.5, 0.03, 4000)
    # Numeric parity against the MLB source of truth, verified in-tree.
    mlb_config_path = BACKEND_DIR.parents[1] / "mlb-backend" / "backend" / "config.py"
    assert mlb_config_path.exists()
    text = mlb_config_path.read_text(encoding="utf-8", errors="replace")
    for token in ("2.4178", "0.6382", "0.1097", "1.2224", "0.4518",
                  "0.7632", "0.0332"):
        assert token in text, f"MLB tuned param {token} drifted in-tree"


def test_config_pins_one_sigma_bar():
    """The NHL RFE commit bar is 1 standard error of the PAIRED per-game
    logloss difference — numeric parity with the MLB RFE bar
    (RFE_NOISE_SIGMA = 1.0, same 0.0005 floor and AUC/ECE guards)."""
    assert config.RFE_NOISE_SIGMA == 1.0
    assert config.RFE_COMMIT_SE_MULTIPLE == 1.0
    mlb_config_path = BACKEND_DIR.parents[1] / "mlb-backend" / "backend" / "config.py"
    text = mlb_config_path.read_text(encoding="utf-8", errors="replace")
    import re as _re
    m = _re.search(r"RFE_NOISE_SIGMA\s*=\s*([0-9.]+)", text)
    assert m and float(m.group(1)) == 1.0, "MLB RFE bar drifted from 1σ"


def test_config_geometry_and_contract_pins():
    assert config.NHL_FIRST_SEASON == 2024
    assert config.WARMUP_DAYS == 30
    assert config.MIN_VAL_FOLD_GAMES == 40
    assert config.RETRAIN_CADENCE_DAYS == 7
    assert config.ELO_K == 20.0 and config.ELO_HOME_ADV == 65.0
    assert abs(config.ELO_REVERT_FACTOR - 1 / 3) < 1e-12
    assert config.SPREAD_GRID == list(range(-8, 9))
    assert config.TOTAL_GRID == list(range(4, 13))
    assert config.HALF_STOP_LINES == [-0.5, 0.5]
    assert config.RUN_ENGINE_FIXED_TOTALS == (5, 6, 7)
    assert config.RUN_ENGINE_CANONICAL_SPREADS == (1, 2)
    assert len(config.NHL_TEAM_ID) == 32
    assert set(config.NHL_TEAM_ID) == {
        "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
        "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
        "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
        "WPG", "WSH"}
    assert config.UNK_TEAM_ID == 99
    assert config.TREE_CATEGORICAL_COLS == ["home_team_id", "away_team_id"]
    assert config.GOALIE_FIELDS == [
        "g_home_name", "g_home_sv_pct", "g_home_gaa", "g_home_starts",
        "g_away_name", "g_away_sv_pct", "g_away_gaa", "g_away_starts"]


# ── manifest parity ────────────────────────────────────────────────────────

def test_manifest_validate_parity():
    problems = manifest.validate()
    assert problems == [], f"FEATURE_MANIFEST out of parity: {problems}"


def test_candidate_manifest_parity():
    problems = config.assert_candidate_manifest_parity()
    assert problems == [], f"CANDIDATE_MANIFEST out of parity: {problems}"
    # Every documented candidate carries the tree-only raw-side routing.
    from manifest import CANDIDATE_MANIFEST
    for name, meta in CANDIDATE_MANIFEST.items():
        if name.endswith("_home") or name.endswith("_away"):
            assert meta["model_family_availability"] == ["tree"]
        else:
            assert set(meta["model_family_availability"]) == {"linear", "tree"}


# ── grid keys ──────────────────────────────────────────────────────────────

def test_grid_key_formatting():
    assert dist_mod._grid_key("p_over", 8) == "p_over_8"
    assert dist_mod._grid_key("p_over", 8.5) == "p_over_8_5"
    assert dist_mod._grid_key("p_home_cover", -2) == "p_home_cover_m2"
    assert dist_mod._grid_key("p_home_cover", -0.5) == "p_home_cover_m0_5"
    assert dist_mod._grid_key("p_home_cover", 0.5) == "p_home_cover_0_5"


# ── retention: blanket 10-day window, NHL filename-aged SHAP ──────────────

def test_retention_board_families_age_out_on_the_10_day_window():
    """Blanket 10-day window (anchor -10 .. anchor): dated board families
    keep inside and go stale together beyond it; the frontend serves a
    rolling 10 days, never a historical archive. The SHAP entries here use
    the LEGACY date-embedded id (YYYYMMDD_AWAY@HOME); official NHL numeric
    ids carry no date and are covered in test_run_engine_pit."""
    import retention_policy as rp

    keep = {"current", "seen", "protected"}
    retention_dates = {"20260914", "20260920", "20260923"}
    for rel in (
        "nhl-backend/data_delivery/nhl_moneyline_v1_20260920.json",
        "nhl-backend/data_delivery/nhl_run_engine_markets_20260920.csv",
        "nhl-backend/data_delivery/nhl_run_engine_markets_20260920.meta.json",
        "nhl-backend/data_delivery/nhl_predictions_history_20260920.csv",
        "nhl-backend/data_delivery/nhl_goalie_matchup_20260920.json",
        "nhl-backend/data_delivery/nhl_shap_game_20260920_TOR@MTL.csv",
    ):
        verdict = rp.classify_artifact(
            rel, seen=set(), retention_dates=retention_dates,
            recent_dates=set(), board_dates=set(), anchor_date="20260923")
        assert verdict in keep, f"{rel} classified {verdict} — in-window keep broken"

    # Older than the window: every card family is stale TOGETHER.
    for rel in (
        "nhl-backend/data_delivery/nhl_moneyline_v1_20260911.json",
        "nhl-backend/data_delivery/nhl_run_engine_markets_20260911.csv",
        "nhl-backend/data_delivery/nhl_run_engine_markets_20260911.meta.json",
        "nhl-backend/data_delivery/nhl_predictions_history_20260911.csv",
        "nhl-backend/data_delivery/nhl_shap_game_20260911_TOR@MTL.csv",
    ):
        verdict = rp.classify_artifact(
            rel, seen=set(), retention_dates=set(), recent_dates=set(),
            board_dates=set(), anchor_date="20260923")
        assert verdict == "stale", f"{rel} classified {verdict} — survives the window"

    # A board STILL TRACKED keeps its companions even outside the window.
    with_board = rp.classify_artifact(
        "nhl-backend/data_delivery/nhl_run_engine_markets_20260911.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates={"20260911"}, anchor_date="20260923")
    assert with_board == "current"

    # Backfill safety: artifacts NEWER than the run's anchor keep.
    backfill = rp.classify_artifact(
        "nhl-backend/data_delivery/nhl_moneyline_v1_20260930.json",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set(), anchor_date="20260923")
    assert backfill == "current"

    # Series / never-delete families are untouched by the window.
    for rel in (
        "nhl-backend/data_delivery/nhl_run_engine_monitor_20260911.json",
        "nhl-backend/data_delivery/models/nhl_ensemble_latest.joblib",
        "nhl-backend/data_delivery/nhl_production_cards_history.csv",
        "nhl-backend/data_delivery/nhl_production_cards_history.meta.json",
        "nhl-backend/data_delivery/nhl_feature_selection_state.json",
        "nhl-backend/data_delivery/nhl_pipeline_summary.json",
        "nhl-backend/data_delivery/nhl_oof_moneyline.csv",
    ):
        verdict = rp.classify_artifact(
            rel, seen=set(), retention_dates=set(), recent_dates=set(),
            board_dates=set(), anchor_date="20260923")
        assert verdict == "protected", f"{rel} classified {verdict} — must be protected"

    # An unresolvable SHAP id is never guessed — protected.
    unresolvable = rp.classify_artifact(
        "nhl-backend/data_delivery/nhl_shap_game_TOR@MTL.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set(), anchor_date="20260923")
    assert unresolvable == "protected"


def test_shap_age_uses_the_embedded_game_date():
    """Legacy date-embedded ids age by filename; official NHL numeric ids
    must NOT be truncated into a bogus date."""
    import retention_policy as rp
    rel = "nhl-backend/data_delivery/nhl_shap_game_20260921_TOR@MTL.csv"
    assert rp.shap_game_date(rel) == "20260921"
    assert rp.artifact_date(rel) == "20260921"
    assert rp.is_never_delete(rel) is False
    numeric = "nhl-backend/data_delivery/nhl_shap_game_2026020001.csv"
    assert rp.shap_game_date(numeric) is None
    assert rp.artifact_date(numeric) is None, \
        "a 10-digit NHL game id must never parse as a truncated date"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(fns)} grid/RFE tests passed (offline, no data needed)")
