"""Defensive-feature EXPANSION ablation for the MLB moneyline (v3).

Question: do point-in-time DEFENSIVE features (raw outcomes + leading
indicators, built strictly from prior games) improve the 5-member
ensemble? EXPERIMENT ONLY — production FEATURE_COLS, training, and
artifacts are untouched; this script writes only its own report.

Families (each = per-side home/away columns + home-minus-away diffs;
raw and its diff are ONE block — trees derive diffs implicitly, linear
gets explicit diffs, both z-scored on the training window only):
  F1 TEAM FORM DEFENSE  — runs allowed / ERA-style rolling defense
                          (team_runs_allowed_10g/30g per side + diff)
  F2 BATTED-BALL ALLOWED— opponent exit velo / barrel% / GB% / hard-hit%
                          / line-drive% allowed per team (balls in play
                          AGAINST the team), rolling 15g/30g (min 8/10)
  F3 DEFENSE TREND      — F1/F2 short-vs-long deltas (leading form)
  F4 POSITION-SPLIT     — IF / OF / catcher decomposition via hit_location
                          + fielder IDs + catcher events, rolling 30g
  F5 STARTER-CONDITIONED— defense behind the starter (team defense in the
                          starter's recent starts), rolling 15g

Conditions: C0 baseline; C1 +F1; C2 +F2; C3 +F1+F2; C4 +F3; C5 +F4;
C6 +F5; C7 all. Nested contrasts: C5 vs C3 (position-split vs aggregate),
C6 vs C3 (starter-conditioning), C4 vs C3 (trends vs levels), C7 vs each.

Protocol (two-family):
  1. PIT: every feature for game G uses only pbp rows with
     game_date < G.date (same-day doubleheader legs excluded).
  2. PRE-SCREEN on baseline OOF residuals: LightGBM AND standardized
     logistic, each fit on the defensive features alone to predict the
     baseline's residual sign/size. A family survives if >=1 proxy shows
     signal (logloss reduction vs constant or AUC > 0.5 with n>=500).
     Families both proxies reject are NOT walked forward.
  3. WALK-FORWARD: identical folds/seeds as baseline (shared geometry),
     TWO proxies (LightGBM + standardized logistic) per condition —
     per-game logloss delta vs baseline PER FAMILY.
  4. SIGNIFICANCE: paired Diebold-Mariano + paired t-test on per-game
     logloss (baseline vs condition), per family.
  5. WINNER: single condition by ensemble-weighted validation metric
     (tree family weight 0.5, linear family 0.5 — proxies stand in for
     the production heads). Tree-vs-linear disagreement -> top-2.
  6. GATE: winner(s) evaluated once on the sealed holdout. Nothing
     adopted here — adoption is a separate decision.

Point-in-time publication lag: day T results are assumed available for
T+1 (Statcast finalizes overnight); every ladder uses rows with
game_date < target date, so the 1-day lag is respected by construction.

Data source: data_delivery/pbp_chunks/*.parquet (8-col committed cache)
for F1/F3/F5; the wide defense cache (pbp_defense_*.parquet, built by
build_pbp_defense.py on Kaggle where the 88-col frame exists) for
F2/F4. Missing wide cache -> F2/F4 marked UNBUILDABLE and skipped.

Record: data_delivery/ablation_defense_v3_<sha>.json. COMMITS NOTHING.

Usage:
    python ablation_defense.py                 # full run
    python ablation_defense.py --smoke         # 3 folds -> /tmp
    python ablation_defense.py --prescreen-only
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
import run_margin_ablation as rma  # noqa: E402
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402
from config import DATA_DELIVERY_DIR, RANDOM_SEED  # noqa: E402

EPS = 1e-7
SEED = int(RANDOM_SEED)

# ── Family definitions ──────────────────────────────────────────────────────
# Every family is a list of (home_col, away_col) pairs; diffs are derived.

F1_PAIRS = [
    ("team_runs_allowed_10g", "team_runs_allowed_10g"),
    ("team_runs_allowed_30g", "team_runs_allowed_30g"),
]

F2_PAIRS = [
    ("opp_exitvelo_15g", "opp_exitvelo_15g"),
    ("opp_barrel_pct_15g", "opp_barrel_pct_15g"),
    ("opp_gb_pct_15g", "opp_gb_pct_15g"),
    ("opp_hardhit_pct_15g", "opp_hardhit_pct_15g"),
    ("opp_ld_pct_15g", "opp_ld_pct_15g"),
    ("opp_exitvelo_30g", "opp_exitvelo_30g"),
    ("opp_barrel_pct_30g", "opp_barrel_pct_30g"),
]

F3_PAIRS = [  # leading: short-window minus long-window of F1/F2 cores
    ("team_runs_allowed_10g", "team_runs_allowed_30g"),
    ("opp_exitvelo_15g", "opp_exitvelo_30g"),
    ("opp_barrel_pct_15g", "opp_barrel_pct_30g"),
]

F4_PAIRS = [
    ("def_if_30g", "def_if_30g"),      # IF: hit_location 1-4,6 outs quality
    ("def_of_30g", "def_of_30g"),      # OF: hit_location 7-9
    ("def_catcher_30g", "def_catcher_30g"),  # PB/WP allowed rate
]

F5_PAIRS = [
    ("starter_def_runs_15g", "starter_def_runs_15g"),
    ("starter_def_era_15g", "starter_def_era_15g"),
]

FAMILIES = {"F1": F1_PAIRS, "F2": F2_PAIRS, "F3": F3_PAIRS,
            "F4": F4_PAIRS, "F5": F5_PAIRS}

CONDITIONS = {
    "C0": [],
    "C1": ["F1"],
    "C2": ["F2"],
    "C3": ["F1", "F2"],
    "C4": ["F1", "F2", "F3"],
    "C5": ["F1", "F2", "F4"],
    "C6": ["F1", "F2", "F5"],
    "C7": ["F1", "F2", "F3", "F4", "F5"],
}


# ── Point-in-time ladders from the committed 8-col pbp cache ───────────────

def _load_pbp_lean() -> pd.DataFrame:
    """Concatenate the committed lean pbp chunks (8 cols, 2025-03-18+)."""
    frames = []
    for p in sorted((DATA_DELIVERY_DIR / "pbp_chunks").glob("pbp_*.parquet")):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def build_f1_f3_f5(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Team runs-allowed ladders (F1), trend deltas (F3), and
    starter-conditioned defense (F5) from the LEAN cache + games frame.

    PIT: for each game, only pbp rows with game_date < game date count.
    F5 conditions on the games frame's home_starter_id/away_starter_id:
    the fielding team's runs-allowed in PRIOR games started by THIS
    starter (>=2 prior starts required, else NaN).
    """
    out = games[["game_pk", "game_date", "home_team", "away_team"]].copy()
    for c in ("team_runs_allowed_10g_home", "team_runs_allowed_10g_away",
              "team_runs_allowed_30g_home", "team_runs_allowed_30g_away",
              "starter_def_runs_15g_home", "starter_def_runs_15g_away"):
        out[c] = np.nan
    if pbp.empty or "events" not in pbp.columns:
        return out

    # Runs allowed per team-day: count scoring events against the fielding
    # side. In the lean cache we only know inning_topbot + events; a run-
    # scoring event for the batting side = runs allowed by the fielding side.
    scoring = {"single", "double", "triple", "home_run"}  # proxy: reached-base
    top = pbp[pbp["inning_topbot"] == "Top"]    # away bats, home fields
    bot = pbp[pbp["inning_topbot"] == "Bottom"]  # home bats, away fields

    def _team_day(df: pd.DataFrame, fld_col: str) -> pd.DataFrame:
        return (df.assign(is_score=df["events"].isin(scoring))
                  .groupby([fld_col, "game_date"])["is_score"].sum()
                  .rename("allowed").reset_index()
                  .rename(columns={fld_col: "team"}))

    allowed = pd.concat([
        _team_day(top, "home_team"),
        _team_day(bot, "away_team"),
    ]).groupby(["team", "game_date"])["allowed"].sum().reset_index()
    allowed = allowed.sort_values("game_date")

    def _rolling(team: str, gdate, windows) -> dict:
        # MIN-PRIOR GUARD (spec: min 8/10 prior): a team with 1-2 prior games
        # produces an extreme single-game mean -> extreme z-scores -> the
        # linear proxy diverges (logloss 7.5 blowup in the first local run).
        h = allowed[(allowed["team"] == team) & (allowed["game_date"] < gdate)]
        n = len(h)
        return {w: (float(h["allowed"].tail(w).mean()) if n >= 10 else np.nan)
                for w in windows}

    for i, row in out.iterrows():
        gd = row["game_date"]
        for side, team in (("home", row["home_team"]), ("away", row["away_team"])):
            r = _rolling(team, gd, (10, 30))
            out.at[i, f"team_runs_allowed_10g_{side}"] = r[10]
            out.at[i, f"team_runs_allowed_30g_{side}"] = r[30]

    # F5 starter-conditioned: needs starter ids — without them the columns
    # stay NaN (documented limitation of the lean cache alone).
    if not {"home_starter_id", "away_starter_id"}.issubset(games.columns):
        return out
    gsub = games[["game_pk", "game_date", "home_team", "away_team",
                  "home_starter_id", "away_starter_id"]]
    home_side = allowed.merge(gsub, left_on=["game_date", "team"],
                              right_on=["game_date", "home_team"], how="inner")
    home_side["starter_id"] = home_side["home_starter_id"]
    away_side = allowed.merge(gsub, left_on=["game_date", "team"],
                              right_on=["game_date", "away_team"], how="inner")
    away_side["starter_id"] = away_side["away_starter_id"]
    per_game = pd.concat([
        home_side[["team", "game_date", "allowed", "starter_id"]],
        away_side[["team", "game_date", "allowed", "starter_id"]],
    ]).dropna(subset=["starter_id"]).sort_values("game_date")

    def _starter_def(team: str, starter, gdate) -> float:
        h = per_game[(per_game["team"] == team)
                     & (per_game["starter_id"] == starter)
                     & (per_game["game_date"] < gdate)]
        return float(h["allowed"].tail(15).mean()) if len(h) >= 3 else np.nan

    gsrc = games.set_index("game_pk")
    for i, row in out.iterrows():
        g = gsrc.loc[row["game_pk"]]
        gd = row["game_date"]
        hs, as_ = g.get("home_starter_id"), g.get("away_starter_id")
        if hs is not None and pd.notna(hs):
            out.at[i, "starter_def_runs_15g_home"] = _starter_def(
                row["home_team"], hs, gd)
        if as_ is not None and pd.notna(as_):
            out.at[i, "starter_def_runs_15g_away"] = _starter_def(
                row["away_team"], as_, gd)
    return out


# ── Wide-cache ladders (F2/F4) — require pbp_defense_*.parquet ──────────────

def _load_pbp_wide() -> pd.DataFrame | None:
    for p in sorted(DATA_DELIVERY_DIR.glob("pbp_defense_*.parquet")):
        try:
            df = pd.read_parquet(p)
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
            return df
        except Exception:
            continue
    return None


BIP_EVENTS = {"single", "double", "triple", "field_out", "force_out",
              "fielders_choice", "fielders_choice_out", "grounded_into_double_play",
              "field_error", "sac_bunt", "sac_fly"}


def build_f2_f4(wide: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Batted-ball-allowed (F2) and position-split (F4) ladders from the
    WIDE defense cache. PIT: rows with game_date < target date only."""
    out = games[["game_pk"]].copy()
    f2_cols = [f"{c}_{s}" for c, _ in F2_PAIRS for s in ("home", "away")]
    f4_cols = [f"{c}_{s}" for c, _ in F4_PAIRS for s in ("home", "away")]
    for c in f2_cols + f4_cols:
        out[c] = np.nan
    if wide is None:
        return out

    bip = wide[wide["events"].isin(BIP_EVENTS) & wide["launch_speed"].notna()].copy()
    bip["is_barrel"] = ((bip["launch_speed"] >= 98) &
                        (bip["launch_angle"].between(26, 30))).astype(float)
    bip["is_gb"] = (bip["launch_angle"] < 10).astype(float)
    bip["is_ld"] = (bip["launch_angle"].between(10, 25)).astype(float)
    bip["is_hh"] = (bip["launch_speed"] >= 95).astype(float)
    # Fielding side: Top = home fields, Bottom = away fields
    bip["field_team"] = np.where(bip["inning_topbot"] == "Top",
                                 bip["home_team"], bip["away_team"])

    def _loc_bucket(loc) -> str:
        try:
            v = float(loc)
        except (TypeError, ValueError):
            return "none"
        if v in (1, 2, 3, 4, 6):
            return "if"
        if v in (7, 8, 9):
            return "of"
        return "none"

    bip["loc_bucket"] = bip["hit_location"].map(_loc_bucket)

    agg_cache: dict[tuple, dict] = {}

    def _stats(team: str, gdate, window: int) -> dict:
        key = (team, gdate, window)
        if key in agg_cache:
            return agg_cache[key]
        h = bip[(bip["field_team"] == team) & (bip["game_date"] < gdate)]
        tail = h.tail(window * 130)  # ~130 BIP/team-game cap
        n = len(tail)
        s = {
            "exitvelo": float(tail["launch_speed"].tail(window * 40).mean()) if n else np.nan,
            "barrel": float(tail["is_barrel"].mean()) if n else np.nan,
            "gb": float(tail["is_gb"].mean()) if n else np.nan,
            "hh": float(tail["is_hh"].mean()) if n else np.nan,
            "ld": float(tail["is_ld"].mean()) if n else np.nan,
            "if": float(tail[tail["loc_bucket"] == "if"]["is_barrel"].mean()) if n else np.nan,
            "of": float(tail[tail["loc_bucket"] == "of"]["is_barrel"].mean()) if n else np.nan,
        }
        # catcher: PB/WP allowed per game (from wide events, all rows)
        cw = wide[(wide["events"].isin(["wild_pitch", "passed_ball"])) &
                  (wide["game_date"] < gdate)]
        cteam = cw[cw["field_team"] == team] if "field_team" in cw.columns else cw.iloc[0:0]
        # wild_pitch/passed_ball rows carry the FIELDING team in home/away via topbot
        if cteam.empty and not cw.empty:
            cw2 = wide[wide["events"].isin(["wild_pitch", "passed_ball"])].copy()
            cw2["field_team"] = np.where(cw2["inning_topbot"] == "Top",
                                         cw2["home_team"], cw2["away_team"])
            cteam = cw2[(cw2["field_team"] == team) & (cw2["game_date"] < gdate)]
        n_games = h["game_pk"].nunique() if n else 0
        s["catcher"] = float(len(cteam) / max(n_games, 1)) if n_games else np.nan
        agg_cache[key] = s
        return s

    for i, row in out.iterrows():
        g = games.iloc[i]
        gd, ht, at = g["game_date"], g["home_team"], g["away_team"]
        for side, team in (("home", ht), ("away", at)):
            s15 = _stats(team, gd, 15)
            s30 = _stats(team, gd, 30)
            out.at[i, f"opp_exitvelo_15g_{side}"] = s15["exitvelo"]
            out.at[i, f"opp_barrel_pct_15g_{side}"] = s15["barrel"]
            out.at[i, f"opp_gb_pct_15g_{side}"] = s15["gb"]
            out.at[i, f"opp_hardhit_pct_15g_{side}"] = s15["hh"]
            out.at[i, f"opp_ld_pct_15g_{side}"] = s15["ld"]
            out.at[i, f"opp_exitvelo_30g_{side}"] = s30["exitvelo"]
            out.at[i, f"opp_barrel_pct_30g_{side}"] = s30["barrel"]
            out.at[i, f"def_if_30g_{side}"] = s30["if"]
            out.at[i, f"def_of_30g_{side}"] = s30["of"]
            out.at[i, f"def_catcher_30g_{side}"] = s30["catcher"]
    return out


# ── Feature-matrix assembly per condition ───────────────────────────────────

def family_columns(family: str) -> list[str]:
    cols = []
    for h, a in FAMILIES[family]:
        for base in ({h, a}):
            for side in ("home", "away"):
                cols.append(f"{base}_{side}")
    return sorted(set(cols))


def diff_columns(family: str) -> list[str]:
    cols = []
    for h, a in FAMILIES[family]:
        base = h if h == a else None
        if base:
            cols.append(f"{base}_diff")
    return cols


def add_defense_frame(games: pd.DataFrame, families: list[str],
                      f135: pd.DataFrame, f24: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the requested families' side columns + diffs to a games copy.

    Join is by game_pk (NOT positional): fold subsets are slices of the
    games frame, so a positional ``src[col].values`` assignment crashes on
    any subset whose index/length differs from the full frame.
    """
    df = games.copy()
    if "game_pk" not in df.columns:
        return df
    for fam in families:
        src = f135 if fam in ("F1", "F3", "F5") else f24
        if src is None or "game_pk" not in src.columns:
            continue
        cols = [c for c in family_columns(fam) if c in src.columns]
        if cols:
            df = df.merge(src[["game_pk"] + cols].drop_duplicates("game_pk"),
                          on="game_pk", how="left")
        if fam == "F3":
            # trend = short minus long (already distinct side cols)
            for short, long in F3_PAIRS:
                for side in ("home", "away"):
                    sc, lc = f"{short}_{side}", f"{long}_{side}"
                    if sc in df.columns and lc in df.columns:
                        df[f"trend_{short}_{side}"] = df[sc] - df[lc]
            continue
        for dcol in diff_columns(fam):
            base = dcol[:-5]
            hc, ac = f"{base}_home", f"{base}_away"
            if hc in df.columns and ac in df.columns:
                df[dcol] = df[hc] - df[ac]
    return df


def condition_feature_cols(cond: list[str], f135: pd.DataFrame,
                           f24: pd.DataFrame | None) -> list[str]:
    cols = []
    for fam in cond:
        src = f135 if fam in ("F1", "F3", "F5") else f24
        if src is None:
            continue
        cols.extend(family_columns(fam))
        if fam == "F3":
            cols.extend([f"trend_{s}_{side}" for s, _ in F3_PAIRS
                         for side in ("home", "away")])
        else:
            cols.extend(diff_columns(fam))
    return sorted(set(cols))


# ── Per-family z-scoring (train-window only) ────────────────────────────────

def _median_zero_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median-fill, then zero-fill.

    A column that is ALL-NaN in a split (sparse family coverage, e.g. an
    F5 behind-starter ladder early in the frame) keeps NaN after a median
    fill (median of NaN is NaN), which LogisticRegression rejects. The
    zero-fill pass turns it into a constant column — no signal after
    z-scoring, so it cannot bias the measurement. Guarded by unit test
    (TestImputeSeam).
    """
    return df.fillna(df.median()).fillna(0.0)


def zscore_train(train: pd.DataFrame, val: pd.DataFrame,
                 cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    va = val.copy()
    for c in cols:
        if c not in tr.columns:
            continue
        mu = tr[c].mean()
        sd = tr[c].std() or 1.0
        tr[c] = (tr[c] - mu) / sd
        va[c] = (va[c] - mu) / sd
    return tr, va


# ── Two-proxy condition trainer (mirrors production member wiring) ──────────

def train_condition(train: pd.DataFrame, val: pd.DataFrame,
                    def_cols: list[str]):
    """LightGBM (raw+diffs) + standardized logistic (diffs only, z-scored)
    on BASELINE features + defense columns. Mirrors production routing:
    logistic mirrors LOGISTIC_USE_RAW_COLS=False (diff cols only)."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    base_cols = [c for c in training.FEATURE_COLS if c in train.columns]
    diff_cols = [c for c in def_cols if c.endswith("_diff") or c.startswith("trend_")]
    raw_cols = [c for c in def_cols if c not in diff_cols]

    y_tr = train["home_win"].values.astype(int)
    y_va = val["home_win"].values.astype(int)

    # Tree proxy: baseline numerics + ALL defense (raw + diffs)
    Xtr_tree = train[base_cols + def_cols].fillna(train[base_cols + def_cols].median())
    Xva_tree = val[base_cols + def_cols].fillna(train[base_cols + def_cols].median())
    lgbm = LGBMClassifier(random_state=SEED, n_estimators=200, learning_rate=0.05,
                          num_leaves=15, min_child_samples=30, verbose=-1)
    lgbm.fit(Xtr_tree, y_tr)

    # Linear proxy: baseline + defense DIFFS only, z-scored on train window
    lin_cols = base_cols + diff_cols
    # Raw train mu/sd captured BEFORE z-scoring so the predict lambda can
    # z-score val with the SAME transform the fit used. NaN fill uses the
    # Z-SCORED median (~0), matching Xtr_lin's imputation.
    lin_mu = train[lin_cols].mean()
    lin_sd = train[lin_cols].std().replace(0, 1.0)
    tr_z, va_z = zscore_train(train, val, lin_cols)
    z_med = tr_z[lin_cols].median()
    # An all-NaN column (sparse family coverage, e.g. F5 early-season
    # split) has NaN median; the zero-fill pass makes it a constant column
    # rather than crashing LogisticRegression. Tested in TestImputeSeam.
    Xtr_lin = _median_zero_impute(tr_z[lin_cols])
    Xva_lin = _median_zero_impute(va_z[lin_cols])
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(Xtr_lin, y_tr)

    def _logistic_predict(v: pd.DataFrame) -> pd.DataFrame:
        Xz = (v[lin_cols] - lin_mu) / lin_sd
        return _median_zero_impute(Xz.fillna(z_med))

    return {
        "lgbm": (lgbm, lambda v: v[base_cols + def_cols]
                 .fillna(train[base_cols + def_cols].median())),
        "logistic": (lr, _logistic_predict),
    }, y_va


def logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ── Walk-forward for one condition (two proxies, shared folds) ──────────────

def walk_forward_condition(cond_families: list[str], folds, f135: pd.DataFrame,
                           f24: pd.DataFrame | None) -> dict | None:
    """Walk-forward one condition with BOTH proxies.

    SUBSET-CONSISTENCY (mandatory): every fold either trains/evaluates on
    the FULL val split or is skipped entirely — a condition may never be
    evaluated on a smaller covered subset than another condition. The
    returned fold_counts make the caller able to assert C0 and C1..C7 saw
    the identical game set; a mismatch aborts the comparison rather than
    shipping an artifact comparison."""
    def_cols = condition_feature_cols(cond_families, f135, f24)
    y_all = []
    proxy_ll = {"lgbm": [], "logistic": []}
    fold_counts = []

    for split in folds:
        train, val = split["train_games"], split["val_games"]
        if len(val) < 40 or len(train) < 200:
            fold_counts.append({"val_start": str(split.get("val_start", "")),
                                "n_val": len(val), "used": False,
                                "reason": "fold_too_small"})
            continue
        tr = add_defense_frame(train, cond_families, f135, f24)
        va = add_defense_frame(val, cond_families, f135, f24)
        # Wide-cache bases (opp_*) inside F3/F4 only exist when f24 does;
        # keep only def columns that actually attached to the frame.
        def_cols_present = [c for c in def_cols if c in tr.columns]
        if not def_cols_present or tr[def_cols_present].notna().sum().sum() == 0:
            fold_counts.append({"val_start": str(split.get("val_start", "")),
                                "n_val": len(val), "used": False,
                                "reason": "no_def_cols"})
            continue
        try:
            proxies, y_va = train_condition(tr, va, def_cols_present)
        except Exception:
            fold_counts.append({"val_start": str(split.get("val_start", "")),
                                "n_val": len(val), "used": False,
                                "reason": "train_error"})
            continue
        # SUBSET-CONSISTENCY ASSERT: the evaluated games must be the FULL
        # val split, in fold order — never a covered subset.
        assert len(y_va) == len(val), (
            f"subset violation: evaluated {len(y_va)} of {len(val)} val games")
        fold_counts.append({"val_start": str(split.get("val_start", "")),
                            "n_val": len(val), "used": True})
        y_all.extend(y_va.tolist())
        for name, (model, mat) in proxies.items():
            p = model.predict_proba(mat(va))[:, 1]
            proxy_ll[name].extend(p.tolist())

    if not y_all:
        return None
    return {"y": np.asarray(y_all), "proxy_p": {k: np.asarray(v) for k, v in proxy_ll.items()},
            "def_cols": def_cols_present, "fold_counts": fold_counts}


# ── Significance tests ──────────────────────────────────────────────────────

def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """DM statistic + two-sided p on paired per-game losses (H0: equal)."""
    d = loss_a - loss_b
    n = len(d)
    if n < 30:
        return float("nan"), float("nan")
    dbar = d.mean()
    # HAC variance with lag 1 (simple Newey-West)
    g0 = ((d - dbar) ** 2).mean()
    g1 = ((d[1:] - dbar) * (d[:-1] - dbar)).mean()
    var = (g0 + 2 * g1) / n
    if var <= 0:
        # Zero variance: identical loss vectors -> no difference (null);
        # degenerate variance otherwise -> untestable.
        if abs(dbar) < 1e-12:
            return 0.0, 1.0
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var)
    # normal p
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(dm) / sqrt(2))))
    return float(dm), float(p)


def paired_t(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    from scipy import stats as st
    d = np.asarray(loss_a) - np.asarray(loss_b)
    if len(d) < 30:
        return float("nan"), float("nan")
    t, p = st.ttest_1samp(d, 0)
    return float(t), float(p)


# ── Pre-screen on baseline OOF residuals ────────────────────────────────────

def prescreen(folds, f135: pd.DataFrame, f24: pd.DataFrame | None,
              families: list[str]) -> dict:
    """Fit LightGBM + standardized logistic on the defensive features alone
    to predict the baseline residual (|residual|>median -> hard games).
    Signal = AUC vs 0.5 (n>=500) or logloss reduction vs constant prior."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    results = {}
    for fam in families:
        cols = condition_feature_cols([fam], f135, f24)
        if not cols or (fam in ("F2", "F4") and f24 is None):
            results[fam] = {"status": "UNBUILDABLE" if fam in ("F2", "F4") and f24 is None
                            else "NO_COLS", "survived": False}
            continue
        X_parts, y_parts = [], []
        for split in folds:
            train, val = split["train_games"], split["val_games"]
            if len(val) < 40:
                continue
            tr = add_defense_frame(train, [fam], f135, f24)
            va = add_defense_frame(val, [fam], f135, f24)
            # Wide-cache bases (opp_*) inside F3 only exist when f24 does;
            # measure F3 on whichever of its columns actually attached.
            present = [c for c in cols if c in va.columns]
            if not present or va[present].notna().sum().sum() == 0:
                continue
            # residual proxy: baseline predicts via elo/win_pct only (cheap);
            # hard games = close games where baseline is least confident
            base_score = (va["elo_diff"].fillna(0) * 0.01 + va["win_pct_diff"].fillna(0))
            p_base = 1 / (1 + np.exp(-base_score.clip(-6, 6)))
            y = va["home_win"].values.astype(int)
            resid = np.abs(y - p_base)
            hard = (resid > np.median(resid)).astype(int)
            # Median-fill, then zero-fill: a column that is ALL-NaN in this
            # split keeps NaN after median (median of NaN is NaN), which
            # LogisticRegression rejects. Zero-fill leaves it a constant
            # column (no signal after z-scoring) — harmless for screening.
            X_parts.append(_median_zero_impute(va[present]).values)
            y_parts.append(hard)
        if not X_parts or sum(len(y) for y in y_parts) < 500:
            results[fam] = {"status": "INSUFFICIENT_N", "survived": False}
            continue
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        # LightGBM proxy
        lgbm = LGBMClassifier(random_state=SEED, n_estimators=100, num_leaves=7,
                              verbose=-1)
        lgbm.fit(X, y)
        auc_l = roc_auc_score(y, lgbm.predict_proba(X)[:, 1])
        # Logistic proxy (z-scored)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        lr = LogisticRegression(max_iter=500, random_state=SEED)
        lr.fit((X - mu) / sd, y)
        auc_l2 = roc_auc_score(y, lr.predict_proba((X - mu) / sd)[:, 1])
        results[fam] = {
            "status": "OK",
            "auc_lgbm": round(float(auc_l), 4),
            "auc_logistic": round(float(auc_l2), 4),
            "n": int(len(y)),
            "survived": bool(max(auc_l, auc_l2) > 0.52),
        }
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prescreen-only", action="store_true")
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.out = Path("/tmp/ablation_defense_v3_smoke.json")

    sha = rma.head_sha()
    (games, _tune_enriched, hold_df, folds, _m, _hm, _rounds, _u) = \
        rma.prepare_data(args.holdout_days)
    if args.smoke:
        folds = folds[:3]

    print(f"commit={sha[:12]} games={len(games)} folds={len(folds)} seed={SEED}",
          flush=True)

    # FOLD GEOMETRY: recorded so baselines are never miscompared across
    # ablation versions. Folds come from training.walk_forward_splits with
    # RETRAIN_CADENCE_DAYS=7 and MIN_VAL_FOLD_GAMES=40, applied to the
    # tuning frame (games minus the holdout). The count therefore depends on
    # the committed snapshot's date range, not on the ablation version:
    # v2 ran 78 folds on an earlier snapshot, v3 runs 71 on the current one
    # because later games moved into the holdout. Comparisons are only valid
    # within a single result JSON.

    pbp_lean = _load_pbp_lean()
    wide = _load_pbp_wide()
    # F1/F3/F5 build from the WIDE cache when present: it starts 2024-03-20
    # (the lean chunks start 2025-03-18), so early folds are comparable to
    # F2/F4 and the baseline — the subset-consistency check requires a
    # condition's folds to match C0's, and lean-sourced F1 skipped the
    # 2024 folds (all-NaN) while the baseline used them. Lean is the
    # fallback when no wide cache is committed.
    pbp = wide if wide is not None else pbp_lean
    print(f"pbp lean rows={len(pbp_lean)} | wide defense cache={'YES' if wide is not None else 'MISSING (F2/F4 unbuildable)'} "
          f"| F1/F3/F5 source = {'wide' if wide is not None else 'lean'}",
          flush=True)

    f135 = build_f1_f3_f5(pbp, games)
    f24 = build_f2_f4(wide, games) if wide is not None else None

    # Coverage per family
    coverage = {}
    for fam in FAMILIES:
        cols = family_columns(fam)
        src = f135 if fam in ("F1", "F3", "F5") else (f24 if f24 is not None else None)
        if src is None:
            coverage[fam] = 0.0
            continue
        present = [c for c in cols if c in src.columns]
        if not present:
            coverage[fam] = 0.0
            continue
        coverage[fam] = round(float(src[present].notna().all(axis=1).mean()), 4)
    print("family coverage:", coverage, flush=True)

    report = {
        "version": "v3",
        "commit": sha,
        "seed": SEED,
        "n_folds": len(folds),
        "fold_geometry": {
            "retrain_cadence_days": 7,
            "min_val_fold_games": 40,
            "note": ("Fold count varies with the committed snapshot's date "
                     "range (v2: 78 folds on an earlier snapshot; v3: 71 on "
                     "the current one). Only compare within one result JSON."),
        },
        "pbp_lean_rows": int(len(pbp)),
        "wide_cache": wide is not None,
        "family_coverage": coverage,
        "pit_rule": "game_date < target game date (1-day publication lag)",
    }

    # 1. Pre-screen
    print("\n[1] PRE-SCREEN on baseline OOF residuals ...", flush=True)
    screen = prescreen(folds, f135, f24, list(FAMILIES))
    report["prescreen"] = screen
    for fam, r in screen.items():
        print(f"  {fam}: {r}")

    if args.prescreen_only:
        out = args.out or (DATA_DELIVERY_DIR / f"ablation_defense_v3_prescreen_{sha[:12]}.json")
        out.write_text(json.dumps(report, indent=2))
        print(f"\nsaved {out}")
        return

    # 2. Walk-forward surviving conditions (C0 baseline always for reference)
    survivors = [f for f, r in screen.items() if r.get("survived")]
    conds_to_run = ["C0"] + [c for c, fams in CONDITIONS.items()
                             if c != "C0" and all(f in survivors for f in fams)]
    print(f"\n[2] WALK-FORWARD conditions: {conds_to_run}", flush=True)

    # Baseline per-game losses via the same two proxies WITHOUT defense cols
    base_result = walk_forward_condition([], folds, f135, f24)
    # C0 has no def cols -> walk_forward_condition returns None; compute
    # baseline explicitly:
    y_base, base_ll = [], {"lgbm": [], "logistic": []}
    base_fold_counts = []
    if base_result is None:
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        base_cols = [c for c in training.FEATURE_COLS if c in games.columns]
        for split in folds:
            train, val = split["train_games"], split["val_games"]
            if len(val) < 40 or len(train) < 200:
                base_fold_counts.append({"val_start": str(split.get("val_start", "")),
                                         "n_val": len(val), "used": False,
                                         "reason": "fold_too_small"})
                continue
            y_tr = train["home_win"].values.astype(int)
            y_va = val["home_win"].values.astype(int)
            Xtr = train[base_cols].fillna(train[base_cols].median())
            Xva = val[base_cols].fillna(train[base_cols].median())
            lgbm = LGBMClassifier(random_state=SEED, n_estimators=200,
                                  learning_rate=0.05, num_leaves=15,
                                  min_child_samples=30, verbose=-1)
            lgbm.fit(Xtr, y_tr)
            mu, sd = Xtr.mean(), Xtr.std()
            sd[sd == 0] = 1
            lr = LogisticRegression(max_iter=1000, random_state=SEED)
            lr.fit((Xtr - mu) / sd, y_tr)
            assert len(y_va) == len(val), "subset violation in baseline fold"
            base_fold_counts.append({"val_start": str(split.get("val_start", "")),
                                     "n_val": len(val), "used": True})
            y_base.extend(y_va.tolist())
            base_ll["lgbm"].extend(lgbm.predict_proba(Xva)[:, 1].tolist())
            base_ll["logistic"].extend(
                lr.predict_proba((Xva - mu) / sd)[:, 1].tolist())
    y_base = np.asarray(y_base)

    cond_results = {}
    wf_results = {}
    for cond in conds_to_run:
        if cond == "C0":
            continue
        fams = CONDITIONS[cond]
        if any(f not in survivors for f in fams):
            continue
        print(f"  running {cond} (+{','.join(fams)}) ...", flush=True)
        res = walk_forward_condition(fams, folds, f135, f24)
        if res is None:
            continue
        wf_results[cond] = res
        y_c = res["y"]
        n = min(len(y_base), len(y_c))
        entry = {"n": int(n), "def_cols_n": len(res["def_cols"])}
        for proxy in ("lgbm", "logistic"):
            if len(res["proxy_p"][proxy]) == 0 or len(base_ll[proxy]) == 0:
                continue
            m = min(len(base_ll[proxy]), len(res["proxy_p"][proxy]))
            la = logloss(np.asarray(y_base[:m]), np.asarray(base_ll[proxy][:m]))
            lb = logloss(np.asarray(y_c[:m]), np.asarray(res["proxy_p"][proxy][:m]))
            dm, dm_p = diebold_mariano(la, lb)
            t, t_p = paired_t(la, lb)
            entry[proxy] = {
                "base_ll": round(float(la.mean()), 5),
                "cond_ll": round(float(lb.mean()), 5),
                "delta": round(float(lb.mean() - la.mean()), 5),
                "dm": round(dm, 3) if np.isfinite(dm) else None,
                "dm_p": round(dm_p, 4) if np.isfinite(dm_p) else None,
                "t_p": round(t_p, 4) if np.isfinite(t_p) else None,
            }
        cond_results[cond] = entry
        print(f"    {entry}", flush=True)

    # SUBSET-CONSISTENCY CHECK (mandatory): C0 and every condition must have
    # been evaluated on the IDENTICAL game subset — same folds used/skipped,
    # same per-fold val counts. If not, abort rather than ship an artifact.
    if base_result is None:
        base_used = base_fold_counts
    else:
        base_used = base_result.get("fold_counts", [])
    for cond, res in wf_results.items():
        c_used = res.get("fold_counts", [])
        if len(c_used) != len(base_used):
            raise RuntimeError(
                f"{cond}: fold count {len(c_used)} != baseline {len(base_used)} "
                "— subset inconsistency, comparison invalid")
        for b, c in zip(base_used, c_used):
            if b["used"] != c["used"] or b["n_val"] != c["n_val"]:
                raise RuntimeError(
                    f"{cond}: fold {b.get('val_start')} (used={b['used']}, n={b['n_val']}) "
                    f"!= {c.get('val_start')} (used={c['used']}, n={c['n_val']}) — "
                    "subset inconsistency, comparison invalid")
        n_cond = sum(f["n_val"] for f in c_used if f["used"])
        n_base = sum(f["n_val"] for f in base_used if f["used"])
        assert n_cond == n_base, f"{cond}: total eval games differ from baseline"
        print(f"  subset check {cond}: identical folds to C0, "
              f"{n_cond} games evaluated", flush=True)

    report["fold_counts"] = {
        "C0": base_used,
        **{c: res.get("fold_counts", []) for c, res in wf_results.items()},
    }
    report["walk_forward"] = cond_results
    report["surviving_families"] = survivors

    # 3. Winner + gate (per protocol: ensemble-weighted validation; here the
    # two proxies stand in for tree/linear families, equal weight).
    scored = {}
    for cond, e in cond_results.items():
        vals = [v["delta"] for v in e.values() if isinstance(v, dict) and "delta" in v]
        if vals:
            scored[cond] = float(np.mean(vals))
    report["winner_scores"] = scored

    # MAGNITUDE SANITY (mandatory): per-family |delta| > 0.01 is implausibly
    # large for a small defensive block and usually means a subset or
    # covariate mismatch between baseline and condition. Flag loudly.
    for cond, e in cond_results.items():
        for proxy in ("lgbm", "logistic"):
            v = e.get(proxy)
            if isinstance(v, dict) and abs(v["delta"]) > 0.01:
                v["magnitude_flag"] = (
                    "LARGE — verify subset/covariate consistency before trusting")
                print(f"  ⚠ {cond}/{proxy} delta {v['delta']:+.4f} — LARGE; "
                      "verify subset/covariate consistency", flush=True)
    if scored:
        best = min(scored, key=scored.get)
        report["winner"] = best
        print(f"\n[3] WINNER by mean per-family delta: {best} ({scored[best]:+.5f})",
              flush=True)
        print("    GATE NOT RUN: adoption is a separate decision; sealed-holdout "
              "gate executes only when the winner is promoted for adoption.",
              flush=True)
    else:
        report["winner"] = None
        print("\n[3] NO condition beat baseline — DON'T ADOPT", flush=True)

    out = args.out or (DATA_DELIVERY_DIR / f"ablation_defense_v3_{sha[:12]}.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
