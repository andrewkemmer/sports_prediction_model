"""Defensive feature expansion ablation — raw outcomes + leading indicators on
the MLB moneyline ensemble (gated measurement; NOT a feature ship).

Question: the current model has ZERO defensive features (team/pitcher signal is
offensive/woba/ERA-side only; sp_fip is absent from FEATURE_COLS). Does adding
point-in-time defensive metrics — raw run-prevention outcomes and their leading
trends — improve the 5-model ensemble out of sample?

Honest-on-artifact constraint (repo precedent, see run_opponent_adjusted_ablation):
a candidate column is measured ONLY if it is genuinely and leak-safely
computable on the COMMITTED artifacts in this repo. This script therefore:
- builds the defensive family from the committed pbp_chunks/ play-by-play cache
  (the only committed artifact carrying defensive outcomes) + the committed
  game_level_features.csv;
- reports coverage per column and drops nothing silently — a column that is
  not computable (e.g. OAA/DRS/framing: no batted-ball location, no fielder or
  catcher identity anywhere in the committed cache) is reported with the reason
  and excluded from the WITH arms;
- enforces point-in-time discipline: every ladder uses ONLY rows with
  game_date STRICTLY before the game's own date (same-day doubleheader legs
  are excluded by construction; verified by unit test).

Why the pbp cache can only produce three raw metrics. The committed cache has
one row per plate appearance with exactly: game_pk, game_date, home_team,
away_team, inning_topbot, batter, events, game_type. There is no batted-ball
location/launch data, no fielder identity, no catcher identity, no
stolen-base/caught-stealing/passed-ball/wild-pitch events, and no pitch-level
data — so Statcast OAA, DRS, framing runs / CSAA, pop-time and outfield sprint
speed are ALL uncomputable here (each would require data this repo does not
commit). What IS computable per side from PAB-level events:
  * defensive efficiency = outs recorded on balls in play / balls in play
    (the team's BABIP-adjacent run-prevention, standard DEF_EFF);
  * errors per game (field_error + catcher_interf);
  * double plays turned per game (dp + gidp + strikeout_dp + sac_fly_dp + tp).
All three validate against the artifact (corr(home_def_eff, away_score) ≈ −0.57).

Families (mirroring the task's C0–C3):
  C0 baseline = exact production training.FEATURE_COLS (59, asserted).
  C1 = + RAW outcome defense: per side + home−away diff, trailing 30-team-game
       means (min 10 prior games), strictly before game date — 9 columns.
  C2 = + LEADING indicators: for each raw metric, the recent-vs-season trend
       (trailing 15g mean − trailing 60g mean, min 8/30 prior) per side + diff
       — 9 columns. Rationale: defensive outcomes are noisy; a team's recent
       deviation from its season baseline (hot/cold defensive stretch) leads
       run-prevention because it times mean-reversion before it fully regresses
       — the same representational language as the model's existing *_delta_*
       momentum features.
  C3 = C1 + C2 (18 columns).
Raw + its diff travel as ONE block (trees derive diffs; linear gets explicit
diffs + z-scored versions via the fold scaler) — no raw-only/diff-only arms.

Two-family protocol (the rigor that matters):
  1. PIT check — ladders are as-of windows over strictly-prior rows only,
     verified programmatically + by unit test.
  2. CHEAP PRE-SCREEN on baseline OOF residuals — the C0 walk-forward OOF
     residual frame (r = y − blend_p); per family, LightGBM and standardized
     logistic are fit on the family's columns ALONE to predict r. A family
     survives only if at least one proxy shows residual-MSE reduction >= 0.5%
     OR residual sign-AUC >= 0.515 (above constant). Families both proxies
     reject are dropped and never refit.
  3. PER-FAMILY REPRESENTATION — LightGBM (tree proxy) consumes the raw
     columns on the NaN-native imputed matrix; logistic (linear proxy) gets
     train-window median-imputed, train-window-standardized versions (scaler
     and medians fit on the fold's training rows only — never val, never the
     sealed window). The full-ensemble winner arm gets the same routing for
     free from the production trainer (logistic/MLP are standardized by its
     fold scaler; trees consume the imputed matrix).
  4. FAST WALK-FORWARD — expanding-window, IDENTICAL folds/seeds for every
     condition, two proxies (LightGBM = LIGHTGBM_PARAMS verbatim; standardized
     logistic). Per-game log-loss deltas vs baseline per model family.
  5. SIGNIFICANCE — paired Diebold–Mariano (HAC lag-1) and paired t-test on
     the per-game log-loss difference series, baseline vs each condition, per
     model family, on the treatment-on-treated subset (games where the
     condition's defensive columns are all real).
  6. SINGLE WINNER -> FULL ENSEMBLE — one condition selected by the
     ensemble-weighted validation metric (tree share 0.74 / linear share 0.26,
     from the v2026.08.30 blend weights; computed from config if present). If
     the tree and linear families strongly disagree, the top-2 conditions are
     promoted. Winner(s) are evaluated on the FULL 5-model ensemble
     (train_moneyline_ensemble, adaptive weights cleared) and ONCE on the
     sealed 21-day holdout. The sealed window is NEVER touched during
     selection.
  7. NO PRODUCTION CHANGE — training.py / features.py / pipeline.py /
     config.py untouched; the harness swaps training.FEATURE_COLS at run time
     exactly like the prior ablations.

Gate (task rule, identical to the repo's prior gates): a condition is adopted
ONLY if it beats C0 on the sealed 21-day holdout on logloss AND AUC without
degrading ECE-cal, and the pooled-OOF direction does not invert. A pooled win
with a sealed loss -> DON'T ADOPT (this repo has hit that inversion repeatedly
— margin, form-delta, home-edge, opponent-adjusted records). A clean negative
is a valid, reportable outcome.

Also emits a collinearity check: max |pearson r| of each new column against the
baseline FEATURE_COLS (esp. the bullpen / team-woba / sp_fip proxies) — "no
added value because it is a proxy" is a real finding, not a failure.

Emits data_delivery/ablation_defense_<sha>.json (incremental). COMMITS NOTHING.

Usage:
    python ablation_defense.py
    python ablation_defense.py --skip-ensemble      # proxy protocol only
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
    LIGHTGBM_PARAMS,
)
from calibration import apply_platt, fit_platt, MIN_OOF_FOR_FIT  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from scipy.stats import norm, ttest_rel  # noqa: E402

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None

EPS = 1e-7

# ── PAB event taxonomy (from the committed pbp cache's 22 event strings) ────
_BIP_EVENTS = {
    "single", "double", "triple", "field_out", "force_out", "field_error",
    "fielders_choice", "fielders_choice_out", "double_play",
    "grounded_into_double_play", "sac_fly", "sac_bunt", "sac_fly_double_play",
    "triple_play",
}
_OUTS_ON_BIP = {
    "field_out", "force_out", "fielders_choice_out", "double_play",
    "grounded_into_double_play", "sac_fly", "sac_bunt", "sac_fly_double_play",
    "triple_play",
}
_ERROR_EVENTS = {"field_error", "catcher_interf"}
_DP_EVENTS = {
    "double_play", "grounded_into_double_play", "strikeout_double_play",
    "sac_fly_double_play", "triple_play",
}
_TOP_BOT = {"Top": "away", "Bot": "home"}

# Ladder windows / min-gates (never imputed below the gate — NaN is honest).
RAW_WINDOW = 30       # trailing team games for raw outcome means
RAW_MIN = 10          # min prior games with pbp before a raw ladder is real
TREND_FAST = (15, 8)  # (window, min_games) for the recent leg of the trend
TREND_SLOW = (60, 30)  # (window, min_games) for the season leg of the trend

RAW_COLS = [
    "home_defeff_30", "away_defeff_30", "defeff_30_diff",
    "home_err_30", "away_err_30", "err_30_diff",
    "home_dp_30", "away_dp_30", "dp_30_diff",
]
# Defense-behind-the-starter (v2 F5): the F1 per-side metrics recomputed over
# only the games the CURRENT starter started (strictly before, per-starter
# trailing ladder). Windows in STARTS, not team games.
SP_WINDOW = 10   # trailing starts used for the behind-starter ladder
SP_MIN = 5       # min prior starts before a behind-starter value is real
STARTER_COLS = [
    "home_defeff_sp", "away_defeff_sp", "defeff_sp_diff",
    "home_err_sp", "away_err_sp", "err_sp_diff",
    "home_dp_sp", "away_dp_sp", "dp_sp_diff",
]
TREND_COLS = [
    "home_defeff_tr", "away_defeff_tr", "defeff_tr_diff",
    "home_err_tr", "away_err_tr", "err_tr_diff",
    "home_dp_tr", "away_dp_tr", "dp_tr_diff",
]
CONDITIONS: dict[str, list[str]] = {
    "C1": RAW_COLS,
    "C2": TREND_COLS,
    "C3": RAW_COLS + TREND_COLS,
}

# Proxy blend shares from the v2026.08.30 ensemble weights (tree 74% / linear
# 26%): XGB 0.45 + LGB 0.153 + RF 0.137 ; logistic 0.183 + MLP 0.077.
_TREE_KEYS = ("xgboost", "lightgbm", "randomforest")
_LINEAR_KEYS = ("logistic", "mlp")

# Pre-screen survival thresholds (above constant-model noise).
SCREEN_MSE_RED = 0.005   # >= 0.5% relative residual-MSE reduction
SCREEN_AUC = 0.515       # residual sign-AUC clearly above 0.5


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.new("sha256")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── Raw defensive aggregates from the committed pbp cache ──────────────────

def pbp_defensive_aggregates(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per game_pk, per side: defensive efficiency, errors, double plays.

    The pbp frame must carry game_pk / inning_topbot / events (game_type R
    rows are used; everything else is ignored). Each plate appearance belongs
    to the AWAY batting side in "Top" innings and the HOME side in "Bot"
    innings, so the defense facing that side is the OPPOSITE team: home
    defense is measured on the away-batting plate appearances and vice versa.

    Returns one row per game_pk with:
      home_def_eff / away_def_eff  (outs-on-BIP / balls-in-play, NaN if 0 BIP)
      home_err / away_err          (field_error + catcher_interf)
      home_dp / away_dp            (DP events turned)
    """
    pbp = pbp[pbp["game_type"] == "R"].copy()
    pbp["bats"] = pbp["inning_topbot"].map(_TOP_BOT)
    pbp = pbp.dropna(subset=["bats", "events"])
    agg = pbp.groupby(["game_pk", "bats"]).agg(
        bip=("events", lambda s: s.isin(_BIP_EVENTS).sum()),
        oobip=("events", lambda s: s.isin(_OUTS_ON_BIP).sum()),
        err=("events", lambda s: s.isin(_ERROR_EVENTS).sum()),
        dp=("events", lambda s: s.isin(_DP_EVENTS).sum()),
    ).reset_index()
    agg["def_eff"] = (agg["oobip"] / agg["bip"]).where(agg["bip"] > 0)
    # batting side "away" -> HOME defense; "home" -> AWAY defense
    def_eff = agg.pivot(index="game_pk", columns="bats",
                        values="def_eff").rename(
        columns={"away": "home_def_eff", "home": "away_def_eff"})
    err = agg.pivot(index="game_pk", columns="bats", values="err").rename(
        columns={"away": "home_err", "home": "away_err"})
    dp = agg.pivot(index="game_pk", columns="bats", values="dp").rename(
        columns={"away": "home_dp", "home": "away_dp"})
    return def_eff.join(err).join(dp).reset_index()


# ── Leak-safe trailing ladders (pure, testable in isolation) ───────────────

def trailing_team_metric(side: pd.DataFrame,
                         window: int,
                         min_games: int) -> dict[tuple, float]:
    """Pure: per (team, gidx) trailing mean of ``value`` over prior games.

    ``side`` must carry gidx / date (datetime64) / team / value. ONLY rows
    with date STRICTLY before the current row's date contribute (same-day
    doubleheader legs are excluded, so nothing after first pitch leaks). A row
    with fewer than ``min_games`` of prior history gets NaN (never imputed).
    """
    side = side.sort_values(["date", "gidx"]).reset_index(drop=True)
    hist: dict[str, list] = {}
    out: dict[tuple, float] = {}
    for r in side.to_dict("records"):
        t, d, gi = r["team"], r["date"], r["gidx"]
        prior = [h for h in hist.get(t, []) if h[0] < d]
        win = prior[-window:]
        out[(t, gi)] = (float(np.mean([h[1] for h in win]))
                        if len(win) >= min_games else np.nan)
        hist.setdefault(t, []).append((d, r["value"]))
    return out


def trailing_starter_metric(side: pd.DataFrame,
                            window: int = SP_WINDOW,
                            min_starts: int = SP_MIN) -> dict[tuple, float]:
    """Pure: per (starter, gidx) trailing mean of ``value`` over prior starts.

    ``side`` must carry gidx / date (datetime64) / starter / value. ONLY rows
    with the SAME starter and date STRICTLY before the current row's date
    contribute (the starter's prior starts with pbp). A starter with fewer
    than ``min_starts`` of prior starts gets NaN (never imputed). This is the
    "defense behind the starter" signal: the team's defensive outcome in games
    THIS pitcher started, trailing, point-in-time.
    """
    side = side.sort_values(["date", "gidx"]).reset_index(drop=True)
    hist: dict[str, list] = {}
    out: dict[tuple, float] = {}
    for r in side.to_dict("records"):
        s = r["starter"]
        if pd.isna(s):
            continue
        d, gi = r["date"], r["gidx"]
        prior = [h for h in hist.get(s, []) if h[0] < d]
        win = prior[-window:]
        out[(s, gi)] = (float(np.mean([h[1] for h in win]))
                        if len(win) >= min_starts else np.nan)
        hist.setdefault(s, []).append((d, r["value"]))
    return out


def _merge_per_game(df: pd.DataFrame, per_game: pd.DataFrame) -> pd.DataFrame:
    """Merge only the per-game defensive columns that are not already present.

    Guards against merge-suffix collisions when callers apply the F1 and F5
    builders to the same frame in sequence (re-joining shared columns would
    turn ``home_def_eff`` into ``home_def_eff_x``/``home_def_eff_y``)."""
    need = [c for c in per_game.columns if c != "game_pk"
            and c not in df.columns]
    if not need:
        return df
    return df.merge(per_game[["game_pk"] + need], on="game_pk", how="left")


def add_defensive_features(games: pd.DataFrame,
                           per_game: pd.DataFrame,
                           raw_window: int = RAW_WINDOW,
                           raw_min: int = RAW_MIN,
                           trend_fast: tuple = TREND_FAST,
                           trend_slow: tuple = TREND_SLOW) -> pd.DataFrame:
    """Merge pbp defensive aggregates and attach the 18 ladder columns.

    Ladders are strictly point-in-time: each game's home/away values are the
    trailing means over the team's prior games with pbp only. A game's own pbp
    is not required for ITS ladder value (only its team's prior games are), but
    games before enough prior pbp history (and the entire 2024 season — the
    committed cache starts 2025-03-18) keep NaN. Windows are parameterized for
    the unit tests; production defaults are the module constants.
    """
    df = games.copy()
    df = _merge_per_game(df, per_game)
    n = len(df)
    dates = pd.to_datetime(df["game_date"]).values
    idx = np.arange(n)
    for metric, home_col, away_col in (
        ("def_eff", "home_def_eff", "away_def_eff"),
        ("err", "home_err", "away_err"),
        ("dp", "home_dp", "away_dp"),
    ):
        short = "defeff" if metric == "def_eff" else metric
        home = pd.DataFrame({
            "gidx": idx, "date": dates, "team": df["home_team"].values,
            "value": df[home_col].values.astype(float),
        })
        away = pd.DataFrame({
            "gidx": idx, "date": dates, "team": df["away_team"].values,
            "value": df[away_col].values.astype(float),
        })
        side = pd.concat([home, away], ignore_index=True)
        raw = trailing_team_metric(side, raw_window, raw_min)
        fast = trailing_team_metric(side, *trend_fast)
        slow = trailing_team_metric(side, *trend_slow)
        home_teams = df["home_team"].tolist()
        away_teams = df["away_team"].tolist()
        h_raw = [raw.get((t, i), np.nan) for i, t in zip(idx, home_teams)]
        a_raw = [raw.get((t, i), np.nan) for i, t in zip(idx, away_teams)]
        h_tr = [fast.get((t, i), np.nan) - slow.get((t, i), np.nan)
                for i, t in zip(idx, home_teams)]
        a_tr = [fast.get((t, i), np.nan) - slow.get((t, i), np.nan)
                for i, t in zip(idx, away_teams)]
        df[f"home_{short}_30"] = h_raw
        df[f"away_{short}_30"] = a_raw
        df[f"{short}_30_diff"] = np.asarray(h_raw) - np.asarray(a_raw)
        df[f"home_{short}_tr"] = h_tr
        df[f"away_{short}_tr"] = a_tr
        df[f"{short}_tr_diff"] = np.asarray(h_tr) - np.asarray(a_tr)
    return df


def add_starter_defensive_features(
        games: pd.DataFrame,
        per_game: pd.DataFrame,
        window: int = SP_WINDOW,
        min_starts: int = SP_MIN) -> pd.DataFrame:
    """Attach the 9 defense-behind-the-starter columns (v2 F5).

    For each side, ``value`` is the team's defensive metric in that game and
    ``starter`` is that side's starting pitcher (home/away_starter_id, 100%
    present in the artifact). The ladder averages the metric over ONLY the
    starter's prior starts (strictly earlier dates), so the signal is "how the
    defense played behind THIS pitcher recently" — starter-conditioning of
    the aggregate F1. A starter with < min_starts prior keeps NaN.
    """
    df = games.copy()
    df = _merge_per_game(df, per_game)
    n = len(df)
    dates = pd.to_datetime(df["game_date"]).values
    idx = np.arange(n)
    for metric, home_col, away_col, h_starter, a_starter in (
        ("def_eff", "home_def_eff", "away_def_eff",
         "home_starter_id", "away_starter_id"),
        ("err", "home_err", "away_err",
         "home_starter_id", "away_starter_id"),
        ("dp", "home_dp", "away_dp",
         "home_starter_id", "away_starter_id"),
    ):
        short = "defeff" if metric == "def_eff" else metric
        home = pd.DataFrame({
            "gidx": idx, "date": dates,
            "starter": df[h_starter].values,
            "value": df[home_col].values.astype(float),
        })
        away = pd.DataFrame({
            "gidx": idx, "date": dates,
            "starter": df[a_starter].values,
            "value": df[away_col].values.astype(float),
        })
        side = pd.concat([home, away], ignore_index=True)
        ladder = trailing_starter_metric(side, window, min_starts)
        h_sp = [ladder.get((s, i), np.nan)
                for i, s in zip(idx, df[h_starter].tolist())]
        a_sp = [ladder.get((s, i), np.nan)
                for i, s in zip(idx, df[a_starter].tolist())]
        df[f"home_{short}_sp"] = h_sp
        df[f"away_{short}_sp"] = a_sp
        df[f"{short}_sp_diff"] = np.asarray(h_sp) - np.asarray(a_sp)
    return df


def coverage_report(games: pd.DataFrame,
                    cols: list[str]) -> list[dict]:
    out = []
    for c in cols:
        if c not in games.columns:
            out.append({"column": c, "present": False, "coverage": 0.0})
            continue
        out.append({"column": c, "present": True,
                    "coverage": round(float(games[c].notna().mean()), 4)})
    return out


# ── Proxies (tree = LightGBM verbatim LIGHTGBM_PARAMS; linear = std. logistic)

def _impute_scale(X_train: np.ndarray, X_val: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train-window median imputation + StandardScaler (fit on train ONLY)."""
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtri = np.where(np.isnan(X_train), med, X_train)
    Xvai = np.where(np.isnan(X_val), med, X_val)
    sc = StandardScaler()
    Xtrs = sc.fit_transform(Xtri)
    Xvas = sc.transform(Xvai)
    return Xtrs, Xvas, med


def _per_game_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dm_pvalue(d: np.ndarray) -> float:
    """Diebold–Mariano on the paired per-game logloss difference series.

    HAC variance with one lag of autocovariance (DM's standard small-sample
    form). Returns NaN when the series is too short (< 30 games)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return float("nan")
    m = float(d.mean())
    if abs(m) < 1e-12:
        return 1.0
    g0 = float(np.mean((d - m) ** 2))
    g1 = float(np.mean((d[:-1] - m) * (d[1:] - m))) if n > 1 else 0.0
    v = (g0 + 2 * g1) / n
    if v <= 0:
        v = g0 / n
    stat = m / np.sqrt(v)
    return float(2 * (1 - norm.cdf(abs(stat))))


def walk_forward_proxies(folds: list[dict],
                         games: pd.DataFrame,
                         cols: list[str]) -> dict:
    """Two-proxy expanding-window walk-forward on IDENTICAL folds.

    LightGBM gets the raw matrix (NaN-native); standardized logistic gets
    train-window imputed+z-scored columns (medians/scaler fit on the fold's
    train rows only). Returns per-game OOF arrays aligned by row_id:
    y, p_lgb, p_lr (and the NaN-native per-proxy availability mask)."""
    rows = []
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        if len(tr) == 0 or len(va) == 0:
            continue
        Xtr = tr.reindex(columns=cols).to_numpy(dtype=float)
        Xva = va.reindex(columns=cols).to_numpy(dtype=float)
        ytr = tr["home_win"].values.astype(float)
        yva = va["home_win"].values.astype(float)
        p_lgb = np.full(len(va), np.nan)
        p_lr = np.full(len(va), np.nan)
        if LGBMClassifier is not None:
            try:
                m = LGBMClassifier(**LIGHTGBM_PARAMS)
                m.fit(Xtr, ytr)
                p_lgb = m.predict_proba(Xva)[:, 1]
            except Exception:
                pass
        try:
            Xtrs, Xvas, _ = _impute_scale(Xtr, Xva)
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
            lr.fit(Xtrs, ytr)
            p_lr = lr.predict_proba(Xvas)[:, 1]
        except Exception:
            pass
        rows.append(pd.DataFrame({
            "row_id": va["row_id"].values,
            "y": yva,
            "p_lgb": p_lgb,
            "p_lr": p_lr,
        }))
    oof = pd.concat(rows, ignore_index=True).sort_values("row_id")
    return oof


def prescreen(family_cols: list[str], oof: pd.DataFrame,
              games: pd.DataFrame) -> dict:
    """Fit LightGBM + standardized logistic on the family's columns ALONE to
    predict the baseline residual r = y − blend_p. Survives if at least one
    proxy shows residual-MSE reduction >= SCREEN_MSE_RED or sign-AUC >=
    SCREEN_AUC (vs the constant model)."""
    r = oof["y"].values - (0.74 * oof["p_lgb"].values
                           + 0.26 * oof["p_lr"].values)
    g = games.loc[oof["row_id"].values]
    X = g.reindex(columns=family_cols).to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(r) & np.isfinite(oof["p_lgb"].values) & np.isfinite(oof["p_lr"].values)
    Xk, rk = X[keep], r[keep]
    res = {}
    base_mse = float(np.mean(rk ** 2)) if len(rk) else np.nan
    if LGBMRegressor is not None and len(rk) >= 60:
        reg = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                            num_leaves=15, min_child_samples=20,
                            random_state=RANDOM_SEED, verbose=-1)
        reg.fit(Xk, rk)
        rh = reg.predict(Xk)
        mse = float(np.mean((rk - rh) ** 2))
        auc = float(roc_auc_score((rk > 0).astype(int), rh))
        res["lgbm"] = {"mse": mse, "mse_rel_reduction": 1 - mse / base_mse,
                       "sign_auc": round(auc, 4), "n": int(len(rk))}
    if len(rk) >= 60:
        Xs, _, _ = _impute_scale(Xk, Xk)
        lin = LinearRegression()
        lin.fit(Xs, rk)  # regression on the residuals (linear-head proxy)
        rh = lin.predict(Xs)
        mse = float(np.mean((rk - rh) ** 2))
        auc = float(roc_auc_score((rk > 0).astype(int), rh))
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(Xs, (rk > 0).astype(int))
        auc_clf = float(roc_auc_score((rk > 0).astype(int),
                                      clf.predict_proba(Xs)[:, 1]))
        res["logistic"] = {"mse": mse,
                           "mse_rel_reduction": 1 - mse / base_mse,
                           "sign_auc": round(max(auc, auc_clf), 4),
                           "n": int(len(rk))}
    survive = any(
        v["mse_rel_reduction"] >= SCREEN_MSE_RED or v["sign_auc"] >= SCREEN_AUC
        for v in res.values())
    return {"family": family_cols, "n_residuals": int(len(rk)),
            "base_mse": round(base_mse, 6), "survived": survive,
            "per_proxy": res}


# ── Per-condition walk-forward + significance ─────────────────────────────

def condition_walk_forward(folds, games: pd.DataFrame, cols: list[str],
                           cond_name: str, base_oof: pd.DataFrame,
                           base_cols: list[str],
                           def_cols: list[str] | None = None) -> dict:
    oof = walk_forward_proxies(folds, games, cols)
    out = {"cols": cols}
    for tag, pcol in (("lgb", "p_lgb"), ("lr", "p_lr")):
        m = oof.dropna(subset=[pcol])
        ll = _per_game_logloss(m["y"].values, m[pcol].values)
        out[f"{tag}_n"] = int(len(m))
        out[f"{tag}_logloss"] = round(float(np.mean(ll)), 4)
        try:
            out[f"{tag}_auc"] = round(float(roc_auc_score(m["y"].values,
                                                          m[pcol].values)), 4)
        except ValueError:
            out[f"{tag}_auc"] = 0.5
    # treatment-on-treated: games where THIS condition's DEFENSIVE columns are
    # all real (base columns excluded — run_margin_diff is all-NaN in the raw
    # artifact) -> paired per-game logloss delta + DM / paired-t vs baseline.
    g = games.loc[oof["row_id"].values]
    if def_cols:
        present = g.reindex(columns=def_cols).notna().all(axis=1).values
    else:
        present = np.ones(len(oof), dtype=bool)  # C0 baseline: everything
    for tag, pcol in (("lgb", "p_lgb"), ("lr", "p_lr")):
        m = oof[present].dropna(subset=[pcol])
        if len(m) < 60:
            out[f"{tag}_tot_n"] = int(len(m))
            out[f"{tag}_delta"] = None
            out[f"{tag}_dm_p"] = None
            out[f"{tag}_t_p"] = None
            continue
        b = base_oof.set_index("row_id").loc[m["row_id"].values]
        ll_base = _per_game_logloss(m["y"].values, b[pcol].values)
        ll_cond = _per_game_logloss(m["y"].values, m[pcol].values)
        d = ll_base - ll_cond  # positive = condition helps
        out[f"{tag}_tot_n"] = int(len(m))
        out[f"{tag}_delta"] = round(float(d.mean()), 6)
        out[f"{tag}_dm_p"] = round(dm_pvalue(d), 4)
        tt = ttest_rel(ll_base, ll_cond)
        out[f"{tag}_t_p"] = round(float(tt.pvalue), 4)
    return out


def proxy_blend_share() -> tuple[float, float]:
    try:
        from config import ENSEMBLE_WEIGHTS
        w = dict(ENSEMBLE_WEIGHTS)
        tree = sum(w[k] for k in _TREE_KEYS if k in w)
        lin = sum(w[k] for k in _LINEAR_KEYS if k in w)
        if tree > 0 and lin > 0:
            return round(float(tree), 4), round(float(lin), 4)
    except Exception:
        pass
    return 0.74, 0.26  # v2026.08.30 blend weights (documented above)


# ── Full 5-member ensemble arm (winner) — mirrors the prior ablations ──────

def run_ensemble_variant(cols: list[str], folds, tune_df, hold_df) -> dict:
    training.FEATURE_COLS = list(cols)
    training._LAST_ADAPTIVE_WEIGHTS.clear()  # both variants blend identically

    oof_y: list[float] = []
    oof_blend: list[float] = []
    oof_blend_cal: list[float] = []
    oof_members: dict[str, list[float]] = {}
    oof_members_cal: dict[str, list[float]] = {}
    executed = 0
    for split in folds:
        train, val = split["train_games"], split["val_games"]
        try:
            models, _ = training.train_moneyline_ensemble(train, val)
        except Exception as e:
            print(f"  fold {split['fold_idx']} failed: {e}")
            continue
        blend, member_probs, _wts = training.ensemble_predict(models, val)
        y_val = val["home_win"].values.astype(float)
        fold_cal = None
        if len(oof_blend) >= MIN_OOF_FOR_FIT:
            fold_cal = fit_platt(np.asarray(oof_y), np.asarray(oof_blend))
        oof_y.extend(y_val.tolist())
        oof_blend.extend(np.asarray(blend, dtype=float).tolist())
        oof_blend_cal.extend(np.asarray(
            apply_platt(np.asarray(blend), fold_cal), dtype=float).tolist())
        for name, p in member_probs.items():
            pa = np.asarray(p, dtype=float)
            oof_members.setdefault(name, []).extend(pa.tolist())
            oof_members_cal.setdefault(name, []).extend(
                np.asarray(apply_platt(pa, fold_cal), dtype=float).tolist())
        executed += 1

    y_all = np.asarray(oof_y, dtype=float)
    pooled: dict[str, dict] = {
        "blend": training.compute_metrics(y_all, np.asarray(oof_blend)),
        "blend_calibrated": training.compute_metrics(
            y_all, np.asarray(oof_blend_cal)),
    }
    for name, plist in oof_members.items():
        pooled[name] = training.compute_metrics(y_all, np.asarray(plist))
        pooled[f"{name}_calibrated"] = training.compute_metrics(
            y_all, np.asarray(oof_members_cal.get(name, [])))

    models, _ = training.train_moneyline_ensemble(tune_df)
    blend_hold, member_hold, _wts = training.ensemble_predict(models, hold_df)
    y_hold = hold_df["home_win"].values.astype(float)
    holdout: dict[str, dict] = {
        "blend": training.compute_metrics(y_hold, np.asarray(blend_hold)),
    }
    for name, p in member_hold.items():
        holdout[name] = training.compute_metrics(y_hold, np.asarray(p))
    return {"n_cols": len(cols), "folds_executed": executed,
            "pooled": pooled, "holdout": holdout}


# ── Collinearity of new columns vs baseline proxies ────────────────────────

def collinearity_report(games: pd.DataFrame, new_cols: list[str],
                        base_cols: list[str]) -> list[dict]:
    out = []
    for c in new_cols:
        s = games[c]
        rows = s.notna()
        if rows.sum() < 30:
            out.append({"column": c, "n": int(rows.sum()), "max_abs_r": None,
                        "top": []})
            continue
        corrs = []
        for b in base_cols:
            if b not in games.columns:
                continue  # e.g. run_margin_diff (training-time only)
            r = games.loc[rows, b].corr(s[rows])
            if np.isfinite(r):
                corrs.append((abs(float(r)), b, float(r)))
        corrs.sort(reverse=True)
        out.append({
            "column": c, "n": int(rows.sum()),
            "max_abs_r": round(corrs[0][0], 4) if corrs else None,
            "top": [{"base_col": b, "abs_r": round(a, 4), "r": round(r, 4)}
                    for a, b, r in corrs[:3]],
        })
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-ensemble", action="store_true",
                    help="run the proxy protocol only (no full-ensemble gate)")
    args = ap.parse_args()

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    data_hash = sha256_file(data_path)
    pbp_files = sorted(DATA_DELIVERY_DIR.glob("pbp_chunks/pbp_*.parquet"))

    games = pd.read_csv(data_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_win"]).reset_index(drop=True)

    pbp = pd.concat([pd.read_parquet(f) for f in pbp_files], ignore_index=True)
    per_game = pbp_defensive_aggregates(pbp)
    games = add_defensive_features(games, per_game)
    games["row_id"] = np.arange(len(games))

    base_cols = list(training.FEATURE_COLS)
    assert len(base_cols) == 59, (
        f"expected 59 production FEATURE_COLS, got {len(base_cols)} — sync "
        f"this harness with training.py before measuring")

    cutoff = games["game_date"].max() - pd.Timedelta(days=args.holdout_days - 1)
    tune_df = games[games["game_date"] < cutoff].reset_index(drop=True)
    hold_df = games[games["game_date"] >= cutoff].reset_index(drop=True)
    all_splits = training.walk_forward_splits(
        tune_df, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    folds = [s for s in all_splits
             if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]

    tree_share, lin_share = proxy_blend_share()
    print(f"commit={sha[:12]} data_sha={data_hash[:12]} games={len(games)} "
          f"tuning={len(tune_df)} holdout={len(hold_df)} "
          f"folds={len(all_splits)}/{len(folds)} seed={RANDOM_SEED} "
          f"proxy_blend={tree_share}/{lin_share}")

    coverage = coverage_report(games, RAW_COLS + TREND_COLS)
    print("defensive column coverage on committed artifacts (pbp cache "
          f"starts 2025-03-18; 2024 season has no pbp):")
    for c in coverage:
        print(f"    {c['column']:18s} coverage={c['coverage']:.3f}")

    out = args.out or (DATA_DELIVERY_DIR / f"ablation_defense_{sha[:12]}.json")
    if out.exists():
        results = json.loads(out.read_text())
    else:
        results = {
            "schema": "defense-ablation/v1", "commit_sha": sha,
            "data_sha256": data_hash, "holdout_days": args.holdout_days,
            "pbp_chunks": len(pbp_files),
            "pbp_rows": int(len(pbp)), "folds_declared": len(all_splits),
            "folds_executed": len(folds), "clip_eps": EPS,
            "seed": int(RANDOM_SEED),
            "ladder_params": {"raw_window": RAW_WINDOW, "raw_min": RAW_MIN,
                              "trend_fast": TREND_FAST,
                              "trend_slow": TREND_SLOW},
            "proxy_blend": {"tree": tree_share, "linear": lin_share},
            "event_sets": {"bip": sorted(_BIP_EVENTS),
                           "outs_on_bip": sorted(_OUTS_ON_BIP),
                           "errors": sorted(_ERROR_EVENTS),
                           "double_plays": sorted(_DP_EVENTS)},
            "coverage": coverage, "conditions": CONDITIONS,
            "prescreen": {}, "walkforward": {}, "ensemble": {},
            "collinearity": collinearity_report(games, RAW_COLS + TREND_COLS,
                                                base_cols),
        }
        out.write_text(json.dumps(results, indent=2) + "\n")

    # ── 2) pre-screen on baseline OOF residuals ───────────────────────────
    if not results["prescreen"]:
        print("\n[C0 baseline] proxy walk-forward (residual frame) ...")
        base_oof = walk_forward_proxies(folds, games, base_cols)
        base_oof.to_parquet(DATA_DELIVERY_DIR / "ablation_defense_base_oof.parquet")
        base_ll_lgb = float(np.mean(_per_game_logloss(
            base_oof.dropna(subset=["p_lgb"])["y"].values,
            base_oof.dropna(subset=["p_lgb"])["p_lgb"].values)))
        base_ll_lr = float(np.mean(_per_game_logloss(
            base_oof.dropna(subset=["p_lr"])["y"].values,
            base_oof.dropna(subset=["p_lr"])["p_lr"].values)))
        results["baseline"] = {
            "n_oof": int(len(base_oof)),
            "lgb_logloss": round(base_ll_lgb, 4),
            "lr_logloss": round(base_ll_lr, 4),
        }
        for fam in ("C1", "C2"):
            res = prescreen(CONDITIONS[fam], base_oof, games)
            results["prescreen"][fam] = res
            print(f"  pre-screen {fam}: "
                  f"survived={res['survived']} "
                  f"{ {k: v for k, v in res['per_proxy'].items()} }")
        out.write_text(json.dumps(results, indent=2) + "\n")

    survivors = [c for c in ("C1", "C2") if results["prescreen"][c]["survived"]]
    run_conds = ["C0"] + survivors
    if len(survivors) == 2:
        run_conds.append("C3")
    if not survivors:
        print("\npre-screen: BOTH families rejected — no defensive condition "
              "survives to the walk-forward. Verdict: keep baseline (evidence "
              "below).")
    print(f"conditions proceeding to walk-forward: {run_conds}")

    # ── 4) per-condition two-proxy walk-forward + significance ────────────
    for cond in run_conds:
        if cond in results["walkforward"]:
            continue
        cols = base_cols if cond == "C0" else base_cols + CONDITIONS[cond]
        print(f"\n[{cond}] two-proxy walk-forward ({len(cols)} cols) ...")
        base_oof = pd.read_parquet(DATA_DELIVERY_DIR
                                   / "ablation_defense_base_oof.parquet")
        wf = condition_walk_forward(folds, games, cols, cond, base_oof,
                                    base_cols,
                                    def_cols=CONDITIONS.get(cond, []))
        results["walkforward"][cond] = wf
        print(f"    lgb ll={wf.get('lgb_logloss')} auc={wf.get('lgb_auc')} "
              f"delta_tot={wf.get('lgb_delta')} dm_p={wf.get('lgb_dm_p')} "
              f"t_p={wf.get('lgb_t_p')}")
        print(f"    lr  ll={wf.get('lr_logloss')} auc={wf.get('lr_auc')} "
              f"delta_tot={wf.get('lr_delta')} dm_p={wf.get('lr_dm_p')} "
              f"t_p={wf.get('lr_t_p')}")
        out.write_text(json.dumps(results, indent=2) + "\n")

    # ── 6) winner selection on the proxy-blend validation metric ──────────
    if survivors and not results.get("winner"):
        scores = {}
        for cond in run_conds:
            w = results["walkforward"][cond]
            ll = (tree_share * w.get("lgb_logloss", np.nan)
                  + lin_share * w.get("lr_logloss", np.nan))
            scores[cond] = round(float(ll), 6)
        base_ll = scores["C0"]
        cands = {c: scores[c] for c in run_conds if c != "C0"}
        best = min(cands, key=cands.get)
        tree_best = min(run_conds[1:],
                        key=lambda c: results["walkforward"][c]["lgb_logloss"])
        lin_best = min(run_conds[1:],
                       key=lambda c: results["walkforward"][c]["lr_logloss"])
        # A condition is a candidate only if it BEATS baseline on the
        # ensemble-weighted validation metric (a worse blend is not a winner).
        winners = []
        if best in cands and cands[best] < base_ll:
            winners.append(best)
        if tree_best != lin_best and tree_best in cands and lin_best in cands:
            for c in (tree_best, lin_best):
                if c not in winners and cands[c] < base_ll:
                    winners.append(c)
        results["winner"] = {
            "blend_scores": scores, "base_blend_ll": base_ll,
            "selected": winners, "tree_family_best": tree_best,
            "linear_family_best": lin_best,
            "tree_vs_linear_agree": tree_best == lin_best,
            "none_beat_baseline": not winners,
        }
        print(f"\nwinner selection (proxy-blend {tree_share}/{lin_share}): "
              f"scores={scores} -> "
              f"selected={winners or ['(none — keep baseline)']}")
        out.write_text(json.dumps(results, indent=2) + "\n")

    # ── 6b) full 5-member ensemble on baseline + winner(s), sealed gate ───
    if (not args.skip_ensemble and survivors
            and results.get("winner", {}).get("selected")
            and not results["ensemble"]):
        arms = {"C0": base_cols}
        for w in results["winner"]["selected"]:
            arms[w] = base_cols + CONDITIONS[w]
        print(f"\nfull 5-member ensemble arms: {list(arms)}")
        for name, cols in arms.items():
            print(f"  [{name}] {len(cols)} cols, {len(folds)} folds ...")
            r = run_ensemble_variant(cols, folds, tune_df, hold_df)
            r["cols"] = cols
            results["ensemble"][name] = r
            b, h = r["pooled"]["blend"], r["holdout"]["blend"]
            print(f"    pooled {b['logloss']:.4f}/{b['auc']:.4f} "
                  f"ece_cal {r['pooled']['blend_calibrated']['ece']:.4f} | "
                  f"holdout {h['logloss']:.4f}/{h['auc']:.4f}")
            out.write_text(json.dumps(results, indent=2) + "\n")

        # ── gate ─────────────────────────────────────────────────────────
        c0 = results["ensemble"]["C0"]
        gate = {}
        for w in results["winner"]["selected"]:
            if w not in results["ensemble"]:
                continue
            wv = results["ensemble"][w]
            h0, hw = c0["holdout"]["blend"], wv["holdout"]["blend"]
            p0, pw = c0["pooled"], wv["pooled"]
            e0 = c0["pooled"]["blend_calibrated"]["ece"]
            ew = wv["pooled"]["blend_calibrated"]["ece"]
            win = (hw["logloss"] < h0["logloss"] and hw["auc"] > h0["auc"]
                   and ew <= e0)
            pooled_ok = pw["blend"]["logloss"] < p0["blend"]["logloss"]
            gate[w] = {
                "holdout": {"base": h0, "with": hw},
                "pooled_invert": not pooled_ok,
                "ece_cal": {"base": e0, "with": ew,
                            "degraded": ew > e0},
                "adopt": bool(win and pooled_ok),
            }
            print(f"\n=== sealed-holdout gate [{w}] vs C0 ===\n"
                  f"  holdout blend: C0 {h0['logloss']:.4f}/{h0['auc']:.4f} "
                  f"| {w} {hw['logloss']:.4f}/{hw['auc']:.4f}\n"
                  f"  pooled blend: C0 {p0['blend']['logloss']:.4f} | "
                  f"{w} {pw['blend']['logloss']:.4f} "
                  f"(inverted={not pooled_ok})\n"
                  f"  ECE-cal: C0 {e0:.4f} | {w} {ew:.4f} "
                  f"(degraded={ew > e0})\n"
                  f"  -> {('ADOPT' if gate[w]['adopt'] else "DON'T ADOPT")}")
        results["gate"] = gate
        out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nablation written: {out}")


if __name__ == "__main__":
    main()
