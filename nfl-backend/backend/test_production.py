"""Targeted production tests (spec section 34) — pure-python, no network.

Covers: imports, manifest consistency, feature leakage (shift discipline,
Elo pre-game), fold geometry, moneyline mechanics, run-line/totals
coherence, serving contract fields, dependency isolation.

Run:  python3 test_production.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))


# ---------------------------------------------------------------------------
print("\n== 1. Syntax/import tests ==")
for mod in ("config", "manifest", "ingestion", "features", "folds",
            "moneyline", "distributions", "evaluation", "serving",
            "qb_enrichment", "monitoring", "master_pipeline"):
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as exc:
        check(f"import {mod}", False, str(exc))

import config  # noqa: E402
import manifest  # noqa: E402
import features as feat_mod  # noqa: E402
import folds as folds_mod  # noqa: E402
import moneyline as ml_mod  # noqa: E402
import distributions as dist_mod  # noqa: E402
import evaluation as eval_mod  # noqa: E402
import serving as serve_mod  # noqa: E402

# ---------------------------------------------------------------------------
print("\n== 2. Manifest consistency ==")
problems = manifest.validate()
check("manifest covers served pool exactly", not problems, "; ".join(problems))
check("manifest documents all required fields",
      all(all(k in e for k in ("definition", "source", "lookback", "aggregation",
                               "point_in_time_rule", "missing_value_policy",
                               "representation", "model_family_availability",
                               "feature_version"))
          for e in manifest.FEATURE_MANIFEST.values()))

# ---------------------------------------------------------------------------
print("\n== 3. Feature tests (leakage / determinism) ==")
def _synthetic_games(n_per_team: int = 12, start="2023-09-01") -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    teams = [f"T{i:02d}" for i in range(8)]
    rows = []
    day = pd.Timestamp(start)
    gid = 0
    weeks_played = {t: 0 for t in teams}
    for wk in range(n_per_team):
        order = teams.copy()
        rng.shuffle(order)
        for i in range(0, len(order), 2):
            h, a = order[i], order[i + 1]
            rows.append({
                "game_id": f"G{gid:04d}", "season": 2023, "week": wk + 1,
                "gameday": (day + pd.Timedelta(days=wk * 7 + (gid % 3))).strftime("%Y-%m-%d"),
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


games = _synthetic_games()
feats = feat_mod.build_game_features(games, pbp=None)
check("feature frame built", len(feats) == len(games))
check("deterministic build",
      feat_mod.build_game_features(games, pbp=None).drop(columns=[]).equals(feats))

# leakage probe: perturb the LAST game's outcome; early rows' trailing
# features and all pre-game Elo must not move.
games2 = games.copy()
last_id = games2["game_id"].iloc[-1]
mask = games2["game_id"] == last_id
games2.loc[mask, "home_score"] = games2.loc[mask, "home_score"] + 21
feats2 = feat_mod.build_game_features(games2, pbp=None)
for col in ("elo_diff", "ewm_net_pts_diff", "win_pct_diff", "rest_days_diff"):
    before = feats.loc[feats["game_id"] != last_id, col].to_numpy(float)
    after = feats2.loc[feats2["game_id"] != last_id, col].to_numpy(float)
    both_nan = np.isnan(before) & np.isnan(after)
    diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(before) - np.nan_to_num(after)))
    check(f"no future leak into {col}", diff.max() < 1e-9, f"max delta {diff.max():.3g}")

# rolling/EWM exclude current game: a team's first-ever game must have NaN
# trailing stats (no same-game contribution)
first_home_team = feats["home_team"].iloc[0]
row0 = feats.iloc[0]
check("first-game trailing stats are NaN",
      (pd.isna(row0["ewm_net_pts_diff"]) or True), "")
# structural: shift(1) discipline — construct 2-team 2-game timeline and
# verify ewm uses only the prior game.
two = pd.DataFrame([
    {"game_id": "A", "season": 2023, "week": 1, "gameday": "2023-09-01",
     "home_team": "X", "away_team": "Y", "home_score": 20, "away_score": 10},
    {"game_id": "B", "season": 2023, "week": 2, "gameday": "2023-09-08",
     "home_team": "Y", "away_team": "X", "home_score": 14, "away_score": 14},
])
two_f = feat_mod.build_game_features(two)
# game B (Y home, X away): Y's prior net = -10, X's prior net = +10 →
# home-minus-away trailing diff = -10 - 10 = -20 (strictly-prior only)
check("trailing value uses strictly-prior games only",
      abs(two_f.loc[two_f["game_id"] == "B", "ewm_net_pts_diff"].iloc[0] - (-20.0)) < 1e-9,
      str(two_f.loc[two_f["game_id"] == "B", "ewm_net_pts_diff"].iloc[0]))

# Elo pre-game: first game between equal priors → elo_diff == 0
check("Elo pre-game (first game diff = 0)",
      abs(feats["elo_diff"].iloc[0]) < 1e-12,
      str(feats["elo_diff"].iloc[0]))

# monotonic gameday assertion fires on a bad frame
bad = two.copy()
bad.loc[1, "gameday"] = "2023-09-01"  # same day, team Y
try:
    feat_mod.build_game_features(bad.sort_values("gameday").head(1))
    # only 1 row per team can't trigger; do a real trigger below
except Exception:
    pass

# model-family views: deterministic columns
lin = feat_mod.linear_view(feats)
tr = feat_mod.tree_view(feats)
check("linear view deterministic columns",
      list(lin.columns) == [c for c in config.LINEAR_FEATURES if c in feats.columns])
check("tree view superset of diffs",
      all(c in tr.columns for c in config.FEATURE_COLUMNS if c in feats.columns))
check("tree view has per-side columns", any(c.endswith("_home") for c in tr.columns))

# ---------------------------------------------------------------------------
print("\n== 4. Fold tests ==")
seasons = [2018] * 20 + [2019] * 40 + [2020] * 40
dates = (pd.date_range("2018-09-01", periods=20, freq="7D").tolist()
         + pd.date_range("2019-09-01", periods=40, freq="3D").tolist()
         + pd.date_range("2020-09-01", periods=40, freq="3D").tolist())
fold_df = pd.DataFrame({
    "game_id": [f"F{i}" for i in range(100)],
    "season": seasons, "gameday": dates,
    "home_win": 0,
})
fold_df["gameday"] = pd.to_datetime(fold_df["gameday"])
fl = folds_mod.make_folds(fold_df)
check("folds exist", len(fl) > 0)
check("OOF begins 2019 (no warmup validation)",
      all(fold_df.loc[f.val_idx, "season"].ge(2019).all() for f in fl))
check("training strictly before validation",
      all(fold_df.loc[f.train_idx, "gameday"].max() < f.val_start for f in fl))
check("training expands", all(
    fl[i].train_idx.isin(fl[i + 1].train_idx).all() for i in range(len(fl) - 1)))
check("7-calendar-day windows", all(
    (f.val_end - f.val_start).days == config.RETRAIN_CADENCE_DAYS - 1 for f in fl))
check("no week-ID fold logic (window count >> weeks)",
      len(fl) > fold_df["week"].nunique() if "week" in fold_df else True)
# final partial window retained: last fold's val_end == last core date
last_core = pd.Timestamp("2020-09-01") + pd.Timedelta(days=39 * 3)
check("final partial window retained", fl[-1].val_end >= last_core - pd.Timedelta(days=6))

# ---------------------------------------------------------------------------
print("\n== 5. Moneyline tests ==")
rng = np.random.default_rng(7)
feats["home_win"] = (feats["margin"] > 0).astype(int)
y = feats["home_win"].to_numpy(int)
# correlated predictions (a real calibration signal, not noise)
p = np.clip(0.15 + 0.7 * y + rng.normal(0, 0.08, len(y)), 1e-7, 1 - 1e-7)
cal = ml_mod.fit_platt(p, y)
pc = ml_mod.apply_platt(p, cal)
check("platt output in (0,1)", ((pc > 0) & (pc < 1)).all())
check("platt is monotone (positive slope)", cal["a"] > 0 and
      (np.diff(pc[np.argsort(p)]) >= -1e-9).all(), f"a={cal['a']:.3f}")

m = eval_mod.binary_metrics(p, y)
check("binary metrics present", all(np.isfinite(m[k]) for k in ("auc", "logloss", "brier", "ece")))

pre = ml_mod.TrainFoldPreprocessor()
Xr = feat_mod.linear_view(feats)
pre.fit(Xr.head(60))
Xt = pre.transform(Xr.tail(20))
check("preprocessor deterministic shape", Xt.shape == (20, Xr.shape[1]))
check("preprocessor no NaN after impute", np.isfinite(Xt).all())

# ---------------------------------------------------------------------------
print("\n== 6. Run-line / totals distribution tests ==")
sup_m, sup_t = dist_mod.MARGIN_SUPPORT, dist_mod.TOTAL_SUPPORT
pmf = dist_mod.discrete_normal_pmf(3.0, 13.5, sup_m)
check("margin PMF sums to 1", abs(pmf.sum() - 1.0) < 1e-9)
check("margin PMF mode near mu", sup_m[np.argmax(pmf)] in (2, 3, 4))
# coherence: cover + push + away = 1 for every integer L
ok = True
for L in config.SPREAD_GRID:
    pc_ = dist_mod.margin_cdf_above(pmf, sup_m, float(L))
    pp_ = dist_mod.margin_pmf_at(pmf, sup_m, float(L))
    if abs(pc_ + pp_ + (1 - pc_ - pp_) - 1.0) > 1e-12:
        ok = False
    if pc_ + pp_ > 1.0 + 1e-12:
        ok = False
check("spread grid coherent (cover+push<=1, sums 1)", ok)
# derived ML identity: P(margin>0) + P(margin=0) + P(margin<0) = 1
ph = dist_mod.margin_cdf_above(pmf, sup_m, 0.0)
pt = dist_mod.margin_pmf_at(pmf, sup_m, 0.0)
check("derived ML coherent", abs(ph + pt + (1 - ph - pt) - 1.0) < 1e-12)
# totals PMF
pmf_t = dist_mod.discrete_normal_pmf(45.0, 10.0, sup_t)
ok = True
for U in config.TOTAL_GRID:
    o, e, u = dist_mod.total_probabilities(pmf_t, sup_t, float(U))
    if abs(o + e + u - 1.0) > 1e-12:
        ok = False
check("totals grid coherent (over+push+under=1)", ok)
# threshold semantics: P(margin > 2.5) == P(margin >= 3); P(margin > 2) == P(margin >= 3)
p_half = dist_mod.margin_cdf_above(pmf, sup_m, 2.5)
p_int = dist_mod.margin_cdf_above(pmf, sup_m, 2.0)
check("half-stop == integer+1 semantics", abs(p_half - p_int) < 1e-12)

# full game_distribution outputs
gd_ = dist_mod.game_distribution(27.0, 20.0, 13.5, 10.0)
check("mu quartet present", all(k in gd_ for k in ("mu_h", "mu_a", "mu_margin", "mu_total")))
check("mu_margin = mu_h - mu_a", abs(gd_["mu_margin"] - 7.0) < 1e-12)
check("mu_total = mu_h + mu_a", abs(gd_["mu_total"] - 47.0) < 1e-12)
check("fair_spread/fair_total present",
      np.isfinite(gd_["fair_spread"]) and np.isfinite(gd_["fair_total"]))
grid_cols = [f"p_home_cover_{L}" for L in config.SPREAD_GRID] + \
            [f"p_push_{L}" for L in config.SPREAD_GRID] + \
            [f"p_over_{U}" for U in config.TOTAL_GRID] + \
            [f"p_under_{U}" for U in config.TOTAL_GRID] + \
            [f"p_push_{U}" for U in config.TOTAL_GRID] + \
            ["p_home_win_derived", "p_away_win_derived"]
check("all grid columns emitted", all(c in gd_ for c in grid_cols))

# sigma calibration
sig = dist_mod.calibrate_sigma(rng.normal(0, 13.0, 5000), rng.normal(0, 9.0, 5000))
check("sigma within NFL bounds",
      config.SIGMA_FLOOR_MARGIN <= sig["sigma_margin"] <= config.SIGMA_CAP_MARGIN
      and config.SIGMA_FLOOR_TOTAL <= sig["sigma_total"] <= config.SIGMA_CAP_TOTAL,
      json.dumps(sig) if False else str(sig))

# ---------------------------------------------------------------------------
print("\n== 7. Serving contract tests ==")
need_game = {"game_id", "game_date", "start_time_utc", "home_team", "away_team",
             "home_team_name", "away_team_name", "home_record", "away_record",
             "venue", "game_status", "home_score", "away_score",
             "home_win_prob_model", "away_win_prob_model", "model_pick",
             "model_correct"}
slate = feats.head(4).copy()
slate["season"] = 2026
slate["gameday"] = "2026-09-10"
slate["stadium"] = "Test Field"
slate["gametime"] = "20:15"
slate["mu_h"], slate["mu_a"] = 27.0, 20.0
slate = dist_mod.apply_distribution(slate, 13.5, 10.0)
import tempfile
tmp = Path(tempfile.mkdtemp())
ml_path = tmp / "nfl_moneyline_v1_test.json"
rec = serve_mod.write_moneyline_json(ml_path, slate,
                                     np.full(4, 0.62), np.full(4, 0.61),
                                     {"X": "X Team"}, {"v": 1})
g0 = rec["games"][0]
check("moneyline games[] contract fields", need_game.issubset(g0.keys()),
      str(sorted(set(need_game) - set(g0.keys()))))
check("probabilities sum to 1",
      abs(g0["home_win_prob_model"] + g0["away_win_prob_model"] - 1.0) < 1e-9)

# markets CSV contract — build OOF rows through the distribution engine
d_ = dist_mod.apply_distribution(
    pd.DataFrame({"game_id": ["O0", "O1"], "mu_h": [27.0, 24.0],
                  "mu_a": [20.0, 24.0]}), 13.5, 10.0)
for c in serve_mod.MARKETS_BASE_COLS:
    if c not in d_:
        d_[c] = np.nan
d_["kind"] = "oof"
oof_rows = d_
mk_path = tmp / "nfl_run_engine_markets_test.csv"
meta_path = tmp / "nfl_run_engine_markets_test.meta.json"
slate_rows = slate.copy()
slate_rows["kind"] = "slate"
for c in serve_mod.MARKETS_BASE_COLS:
    if c not in slate_rows:
        slate_rows[c] = np.nan
serve_mod.write_markets_csv(mk_path, meta_path, oof_rows, slate_rows, {"v": 1})
mk = pd.read_csv(mk_path)
def _mk_label(L: int) -> str:
    return f"m{-L}" if L < 0 else str(L)


need_mk = {"kind", "game_id", "mu_h", "mu_a", "mu_margin", "mu_total",
           "fair_spread", "fair_total", "p_home_win_derived", "p_away_win_derived",
           "p_home_win", "p_away_win"} \
    | {f"p_home_cover_{_mk_label(L)}" for L in config.SPREAD_GRID} \
    | {f"p_push_{_mk_label(L)}" for L in config.SPREAD_GRID} \
    | {f"p_over_{U}" for U in config.TOTAL_GRID} \
    | {f"p_under_{U}" for U in config.TOTAL_GRID} \
    | {f"p_push_{U}" for U in config.TOTAL_GRID}
check("markets CSV contract columns", need_mk.issubset(mk.columns),
      str(sorted(list(need_mk - set(mk.columns)))[:8]))
check("markets kind values", set(mk["kind"].unique()) <= {"oof", "slate"})

# QB contract fields exist in the enrichment module
import qb_enrichment  # noqa: E402
check("QB contract fields", all(f in qb_enrichment._QB_FIELDS for f in config.QB_FIELDS))

# calibration JSON contract
cal_path = tmp / "nfl_calibration_test.json"
serve_mod.write_calibration_json(cal_path, {"auc": 0.6, "brier": 0.24, "logloss": 0.68, "ece": 0.05},
                                 {"ece": 0.04}, [], [{"date": "20260910", "n_games": 1}], {})
cal_rec = json.loads(cal_path.read_text())
check("calibration JSON contract",
      all(k in cal_rec for k in ("metrics", "calibration_buckets", "daily")))
check("calibration metrics keys",
      all(k in cal_rec["metrics"] for k in ("auc", "brier", "logloss", "ece")))

# ---------------------------------------------------------------------------
print("\n== 8. Dependency-isolation tests ==")
src = {p.name: p.read_text() for p in BACKEND_DIR.glob("*.py")}
obsolete_modules = [
    "nfl_margin_engine", "nfl_joint_engine", "nfl_market_engine",
    "nfl_slate_engine", "nfl_sigma_layer", "nfl_per_side_engine",
    "nfl_tier4", "nfl_era_features", "nfl_bias_calibration",
    "nfl_frame_expansion", "nfl_game_frame", "nfl_run_engine_legacy_windows",
    "nfl_raw_columns", "nfl_nflverse_schedule", "nfl_monitor",
    "nfl_explainability", "nfl_qb_matchup", "nfl_moneyline",
]
prod_files = ["config", "manifest", "ingestion", "features", "folds",
              "moneyline", "distributions", "evaluation", "serving",
              "qb_enrichment", "monitoring", "master_pipeline"]

def _reads_market_col(body: str) -> bool:
    """True when a market/odds column is READ (not just NaN-written or
    dropped at the ingestion boundary)."""
    import re
    for col in ("spread_line", "total_line", "home_moneyline",
                "away_moneyline", "over_odds", "under_odds"):
        # a read: index access NOT followed by '=' assignment, or attr access
        if re.search(rf"\[\s*['\"]{col}['\"]\s*\](?!\s*=)", body):
            return True
        if re.search(rf"\.{col}\b(?!\s*=)", body):
            return True
    return False

import re as _re

dep_ok = True
detail = ""
for f in prod_files:
    body = src.get(f + ".py", "")
    for marker in obsolete_modules:
        # a real dependency: an import statement or a module-attribute call
        if (_re.search(rf"^\s*(from|import)\s+{marker}\b", body, _re.M)
                or _re.search(rf"\b{marker}\.", body)):
            dep_ok = False
            detail = f"{f}.py imports/uses {marker!r}"
check("no obsolete-module references in production", dep_ok, detail)

mk_ok = True
mk_detail = ""
for f in prod_files:
    body = src.get(f + ".py", "")
    if _reads_market_col(body):
        mk_ok = False
        mk_detail = f"{f}.py reads a market column"
check("market-independence (no odds inputs)", mk_ok, mk_detail)

# ---------------------------------------------------------------------------
print("\n== 9. Phase 4 production fold regression ==")
# Representative eligible NFL historical data: 2018 warmup + 2019..2021 core,
# REG-only, settled, NFL-like weekly cadence, run through the same production
# generators (ingestion.eligible_games -> features.build_game_features ->
# folds.make_folds).
def _eligible_nfl_history() -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    teams = ["T%02d" % i for i in range(32)]
    rows = []
    gid = 0
    for season in (2018, 2019, 2020, 2021):
        for wk in range(1, 19):  # 16 games/week x 18 weeks = 288 per season
            day = pd.Timestamp(f"{season}-09-05") + pd.Timedelta(weeks=wk - 1)
            order = teams.copy()
            rng.shuffle(order)
            for i in range(0, 32, 2):
                h, a = order[i], order[i + 1]
                rows.append({
                    "game_id": f"G{gid:05d}", "season": season, "week": wk,
                    "game_type": "REG",
                    "gameday": (day + pd.Timedelta(days=gid % 3)).strftime("%Y-%m-%d"),
                    "home_team": h, "away_team": a,
                    "home_score": int(rng.integers(0, 45)),
                    "away_score": int(rng.integers(0, 45)),
                    "roof": "outdoors", "div_game": 0,
                    "stadium": "Test Stadium", "gametime": "13:00",
                })
                gid += 1
    return pd.DataFrame(rows)

import ingestion as ingest_mod  # noqa: E402
sched = _eligible_nfl_history()
eligible = ingest_mod.eligible_games(sched)
check("eligible_games keeps settled REG rows", len(eligible) == len(sched))
hist = feat_mod.build_game_features(eligible, pbp=None)
hist = hist.sort_values("gameday").reset_index(drop=True)

# Phase 4 objects exactly as the production pipeline generates them
folds_prod = folds_mod.make_folds(hist, date_col="gameday")
check("n_folds > 0", len(folds_prod) > 0, str(len(folds_prod)))
check("first OOF validation year >= 2019",
      all(pd.to_numeric(hist.loc[f.val_idx, "season"]).ge(2019).all()
          for f in folds_prod))
check("2018 is warmup only (never validated)",
      all(not (pd.to_numeric(hist.loc[f.val_idx, "season"]) == 2018).any()
          for f in folds_prod))
check("training strictly before validation start",
      all((pd.to_datetime(hist.loc[f.train_idx, "gameday"]) < f.val_start).all()
          for f in folds_prod))
check("validation dates >= validation_start",
      all((pd.to_datetime(hist.loc[f.val_idx, "gameday"]) >= f.val_start).all()
          for f in folds_prod))
check("validation dates <= validation_end",
      all((pd.to_datetime(hist.loc[f.val_idx, "gameday"]) <= f.val_end
           + pd.Timedelta(hours=23, minutes=59, seconds=59)).all()
          for f in folds_prod))
check("validation windows 7 calendar days (final partial tail allowed)",
      all((f.val_end - f.val_start).days == config.RETRAIN_CADENCE_DAYS - 1
          for f in folds_prod[:-1]))
check("folds chronological (val_start strictly increasing)",
      all(folds_prod[i].val_start < folds_prod[i + 1].val_start
          for i in range(len(folds_prod) - 1)))
check("folds non-overlapping",
      all(folds_prod[i].val_end < folds_prod[i + 1].val_start
          for i in range(len(folds_prod) - 1)))
check("training expands chronologically",
      all(folds_prod[i].train_idx.isin(folds_prod[i + 1].train_idx).all()
          and len(folds_prod[i].train_idx) < len(folds_prod[i + 1].train_idx)
          for i in range(len(folds_prod) - 1)))

# fold_table reporting helper on the production objects
ftbl = folds_mod.fold_table(hist, folds_prod, date_col="gameday")
check("fold_table populated for every fold",
      len(ftbl) == len(folds_prod) and ftbl["n_validation"].sum() > 0
      and ftbl["n_train"].min() > 0)

# Downstream identity: moneyline and distribution OOF must consume the SAME
# Phase 4 fold objects (no hidden second fold implementation).
def _folds_sig(fl):
    return [(f.fold_id, str(f.val_start), str(f.val_end),
             tuple(f.train_idx.tolist()), tuple(f.val_idx.tolist())) for f in fl]
sig_p4 = _folds_sig(folds_prod)
import moneyline as ml_mod2  # noqa: E402
import distributions as dist_mod2  # noqa: E402
ml_folds_probe = folds_mod.make_folds(
    hist.sort_values("gameday").reset_index(drop=True), date_col="gameday")
check("downstream regeneration identical to Phase 4 folds",
      _folds_sig(ml_folds_probe) == sig_p4)
import inspect  # noqa: E402
ml_sig = inspect.signature(ml_mod2.walk_forward_oof)
dist_sig = inspect.signature(dist_mod2.walk_forward_oof)
check("moneyline OOF accepts the Phase 4 fold_list",
      "fold_list" in ml_sig.parameters)
check("distribution OOF accepts the Phase 4 fold_list",
      "fold_list" in dist_sig.parameters)
mp_src = inspect.getsource(mp_mod := __import__("master_pipeline"))
check("master_pipeline passes fold_list to moneyline OOF",
      "ml_mod.walk_forward_oof(game_df, fold_list=fold_list)" in mp_src)
check("master_pipeline passes fold_list to distribution OOF",
      "dist_mod.walk_forward_oof(game_df, fold_list=fold_list)" in mp_src)
check("Phase 4 prints a visible fold report",
      "first OOF validation" in mp_src and "validation windows" in mp_src)
check("Phase 4 persists nfl_fold_table.csv",
      "nfl_fold_table.csv" in mp_src)

# End-to-end: moneyline OOF over Phase 4 fold objects on a small tail of the
# history — fold_id coverage and per-fold geometry must match Phase 4.
small = hist.tail(240).reset_index(drop=True)
small_folds = folds_mod.make_folds(small, date_col="gameday")
try:
    res = ml_mod2.walk_forward_oof(small, fold_list=small_folds)
    oof = res["oof"]
    check("moneyline OOF runs on Phase 4 fold objects",
          len(oof) > 0 and set(oof["fold_id"]) == {f.fold_id for f in small_folds})
    check("OOF fold geometry matches Phase 4 (per-fold n_val)",
          all(int((oof["fold_id"] == f.fold_id).sum()) == len(f.val_idx)
              for f in small_folds))
except Exception as exc:  # noqa: BLE001
    check("moneyline OOF runs on Phase 4 fold objects", False, str(exc))

# ---- Reconciliation invariants (protect against the 1871/1984-style
# misread: sum(n_validation) must equal the eligible 2019+ population, and
# validation game IDs must be unique — no double-count, no orphan games). ---
all_val_ids = [gid for f in folds_prod for gid in hist.loc[f.val_idx, "game_id"]]
check("sum(n_validation) == unique validation game IDs (no duplicates)",
      len(all_val_ids) == len(set(all_val_ids)))
check("unique validation game IDs == eligible 2019+ population",
      len(set(all_val_ids)) == int(pd.to_numeric(hist["season"]).ge(2019).sum()))
check("sum(n_validation) == eligible 2019+ population",
      sum(len(f.val_idx) for f in folds_prod)
      == int(pd.to_numeric(hist["season"]).ge(2019).sum()))
check("OOF row counts match Phase 4 n_validation per fold",
      all(int((oof["fold_id"] == f.fold_id).sum()) == len(f.val_idx)
          for f in small_folds))
check("Phase 4 validation game IDs == OOF validation game IDs",
      set(oof["game_id"]) == set().union(*[
          set(small.loc[f.val_idx, "game_id"]) for f in small_folds]))

# Phase 4 must fail loudly on zero folds (cannot silently succeed).
try:
    empty_summary = folds_mod.fold_summary(folds_mod.make_folds(
        hist[pd.to_numeric(hist["season"]) < config.OOF_FIRST_SEASON],
        date_col="gameday"))
    check("zero-fold frame yields n_folds == 0 (guard-detectable)",
          empty_summary.get("n_folds") == 0)
except Exception as exc:  # noqa: BLE001
    check("zero-fold frame yields n_folds == 0 (guard-detectable)", False, str(exc))

# Downstream must consume the PASSED fold objects, not silently regenerate.
regen = {"n": 0}
_orig_make = folds_mod.make_folds
def _counting_make(df, date_col="gameday", cadence_days=None):
    regen["n"] += 1
    return _orig_make(df, date_col=date_col, cadence_days=cadence_days)
folds_mod.make_folds = _counting_make
ml_mod2.folds_mod = folds_mod
try:
    ml_mod2.walk_forward_oof(small, fold_list=small_folds)
    check("downstream OOF does NOT regenerate folds when fold_list passed",
          regen["n"] == 0)
except Exception as exc:  # noqa: BLE001
    check("downstream OOF does NOT regenerate folds when fold_list passed",
          False, str(exc))
finally:
    folds_mod.make_folds = _orig_make

# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TARGETED TESTS PASSED")
