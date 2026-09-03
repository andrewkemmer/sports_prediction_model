"""NFL era/conditional-mean layer — league-level scoring centers (record-only).

Context (mean-bias record 56893d3 / W2016 record 1a42c9f): the away per-side
mean shows a by-season bias that is a SEASONAL/FOLD-EPOCH artifact, not a
stable offset — away mean resid (actual − pred) 2021 −1.71 / 2022 −1.63 /
2023 −2.27 / 2024 −0.40 / sealed 2025 −0.14. The live hypothesis: the
per-side marginals see only diff/relative features (no absolute scoring
level), so when the LEAGUE-LEVEL scoring environment shifts by season the
absolute-score predictions drift (the regressors anchor near their training
window's mean away scoring, which carried the 2020 spike ~24.7 vs the
2021-23 away collapse ~20.4-22.1). This module adds a leakage-safe,
side-specific league "era center" and the centered-target arm (E2) that
removes the level by construction.

SCOPE (verbatim): this addresses LEAGUE-LEVEL scoring-environment drift,
not team-specific offseason structural change (roster/defense overhaul stays
invisible until games confirm it — do not overclaim).

Center definitions (all functions of ONLY decided games with
game_date strictly before the target game's game_date — NO same-day games,
regardless of kickoff time; source = decided scores + game_date only):
  ps      — prior-season final mean: for a game in season S, the mean of the
            side's score over ALL decided games of season S−1.
  ewm_Xw  — trailing exponential weighted mean over strictly-prior decided
            games with weight 0.5^((days_lag)/halflife), halflife = X*7 days
            (X ∈ {2, 4, 8}). Same-day games are excluded by construction
            (only strictly-earlier days contribute).

Rows with no strictly-prior history at all (the 2019 season's opening games
— warmup rows, never in a fold's val window) are filled with the documented
neutral constant 21.0 (spec judgment call 1c: "arbitrary neutral constant is
fine — 2019-20 are warmup, never scored"). A per-column constant carries no
information to a tree and keeps the E1/E2 training sets identical to C0's.

Arms (same 88-fold geometry, pooled OOF 2021-24 / sealed 2025):
  C0 — current per-side marginals on raw targets (unchanged engine call).
  E1 — 12-pool view + [era_center_home, era_center_away] macro columns,
       raw targets (comparison only — trees must find the level).
  E2 — centered targets: each marginal fits (target − center_side) and adds
       the center back at prediction (PRIMARY arm — fixes level by
       construction, leaves the 12-feature mapping untouched, cannot be
       ignored by the trees).

The centered OOF walk and fit-only refill below MIRROR the per-side engine's
leakage assertion, rounding, per-side median rounds, NaN-not-zero imputation
and fold-aligned geometry exactly — but they are the era module's own code
path because the engine's targets are hardcoded to home_score/away_score
(nfl_per_side_engine.py and nfl_joint_engine.py are NOT modified).

Judgment calls (flag if overridden):
  1. Linear center specs only (ps + 3 EWM halflives); "blend" specs omitted —
     each candidate is a clean falsifiable level estimate and the CV
     selection is thin (n=1,091); a blend is only warranted if the CV table
     shows no clean winner.
  2. League-level (not team-level) centers: team-specific strength is already
     carried by the 12-pool; the era layer isolates the common level. Roster
     turnover stays invisible until games confirm it (scope note).
  3. Constant 21.0 neutral fallback for the first-ever rows (warmup only).
  4. Sealed 2025 rows' centers may use strictly-prior 2025 games (the center
     is a pure time-series of outcomes; its inputs are never the row's own
     or later labels) — mirror of how OOF val rows use same-season prior
     weeks. The model is never FIT on 2025 rows.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Center column names (never collide with the served pool / 12-pool).
CENTER_HOME = "era_center_home"
CENTER_AWAY = "era_center_away"
CENTER_COLS = [CENTER_HOME, CENTER_AWAY]

# Neutral first-row fallback per judgment call 3 (spec 1c). Warmup rows only:
# 2019-season opening days for the EWM specs and all 2019 rows for ps (no
# 2018 season in the decided frame). Constant ⇒ information-free for trees.
NEUTRAL_CENTER = 21.0

# Center spec registry: ps (prior-season anchor) + trailing EWM halflives.
EWM_HALFLIFE_DAYS = {"ewm_2w": 14, "ewm_4w": 28, "ewm_8w": 56}
SPECS = ["ps"] + sorted(EWM_HALFLIFE_DAYS)
DEFAULT_SPEC = "ewm_2w"  # provisional; chosen empirically on 2021-24 CV

SIDES = ("home", "away")
SCORE_COL = {"home": "home_score", "away": "away_score"}
PRED_COL = {"home": "pred_home", "away": "pred_away"}
RESID_COL = {"home": "resid_home", "away": "resid_away"}

# Residual convention (identical to the step-1 artifact): resid = actual − pred.
# Negative mean resid ⇒ predictions run HIGH (overprediction).


# ── Center construction (leakage-safe) ───────────────────────────────────────

def _ewm_centers(sorted_df: pd.DataFrame, hw_days: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Trailing EWM centers over strictly-prior days.

    ``sorted_df`` must be sorted by [gameday, game_id]. Day-level rolling
    sums: the center for every game on day t is the decayed weighted mean of
    side scores over games on days < t (same-day games NEVER contribute —
    they are added to the state only after the day's centers are assigned).
    Weight of a game at lag Δ days = 0.5^(Δ / hw_days).
    """
    hs = sorted_df["home_score"].to_numpy(float)
    as_ = sorted_df["away_score"].to_numpy(float)
    day = sorted_df["gameday"].dt.normalize()
    ords = (day - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    uniq, idx = np.unique(ords, return_inverse=True)
    out_h = np.full(len(sorted_df), np.nan)
    out_a = np.full(len(sorted_df), np.nan)
    dec = 0.5 ** (1.0 / hw_days)
    for s_arr, out in ((hs, out_h), (as_, out_a)):
        S = W = 0.0
        prev_o = None
        for i, o in enumerate(uniq):
            m = idx == i
            out[m] = S / W if W > 0 else np.nan
            if prev_o is not None:
                diff = int(o - prev_o)
                S *= dec ** diff
                W *= dec ** diff
            S += float(s_arr[m].sum())
            W += float(m.sum())
            prev_o = o
    return out_h, out_a


def _ps_centers(sorted_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prior-season final mean per side (mean over season S−1 decided games)."""
    means = sorted_df.groupby("season")[["home_score", "away_score"]].mean()
    season = sorted_df["season"].to_numpy()
    out_h = np.full(len(sorted_df), np.nan)
    out_a = np.full(len(sorted_df), np.nan)
    for i, s in enumerate(season):
        if s - 1 in means.index:
            out_h[i] = float(means.loc[s - 1, "home_score"])
            out_a[i] = float(means.loc[s - 1, "away_score"])
    return out_h, out_a


def compute_centers(decided: pd.DataFrame, spec: str = DEFAULT_SPEC
                    ) -> pd.DataFrame:
    """Leakage-safe era centers per game for one spec.

    Args:
        decided: the decided frame (one row per decided game) with columns
            game_id, season, gameday, home_score, away_score. Center for a
            game g is a function of ONLY games with gameday strictly before
            g's gameday (same-day games excluded regardless of kickoff).
        spec: "ps" | "ewm_2w" | "ewm_4w" | "ewm_8w".

    Returns:
        DataFrame [game_id, era_center_home, era_center_away], one row per
        decided game in input order. Rows with no strictly-prior history are
        filled with NEUTRAL_CENTER (documented constant; warmup rows only —
        never scored, information-free for trees).
    """
    if spec not in SPECS:
        raise ValueError(
            f"compute_centers: unknown spec {spec!r} — expected {SPECS}")
    need = {"game_id", "season", "gameday", "home_score", "away_score"}
    missing = [c for c in need if c not in decided.columns]
    if missing:
        raise ValueError(f"compute_centers: missing columns {missing}")
    df = decided.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    if df["gameday"].isna().any():
        raise ValueError("compute_centers: NaN gameday present")
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    if spec == "ps":
        ch, ca = _ps_centers(df)
    else:
        ch, ca = _ewm_centers(df, float(EWM_HALFLIFE_DAYS[spec]))

    ch = np.where(np.isfinite(ch), ch, NEUTRAL_CENTER)
    ca = np.where(np.isfinite(ca), ca, NEUTRAL_CENTER)
    out = pd.DataFrame({
        "game_id": df["game_id"].values,
        CENTER_HOME: np.round(ch, 4),
        CENTER_AWAY: np.round(ca, 4),
    })
    if out[CENTER_COLS].isna().any().any():
        raise RuntimeError("compute_centers: NaN centers survived the fill")
    return out


def attach_centers(feats: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    """Merge center columns onto the feature frame by game_id (no reindex)."""
    out = feats.merge(centers, on="game_id", how="left")
    missing = int(out[CENTER_COLS].isna().sum().sum())
    if missing:
        raise RuntimeError(
            f"attach_centers: {missing} feature rows lack a center — "
            "frame/center mismatch (every decided game must carry one)")
    return out


# ── Centered-target OOF walk (E2), mirror of oof_per_side ────────────────────

def oof_centered_per_side(folds: list[dict], features: list[str],
                          frame: pd.DataFrame, family: str = "lgb"
                          ) -> tuple[pd.DataFrame, dict, int]:
    """Fold-aligned OOF walk on (target − center_side); center added back.

    Mirror of ``nfl_per_side_engine.oof_per_side`` — same leakage assertion
    (max(train.gameday) < min(val.gameday), else AssertionError), same
    small-fold skip rules, same 4-decimal rounding, same per-side median
    rounds and uncovered counting. The ONLY differences: targets are centered
    per row by the row's own era center (frame must carry CENTER_COLS) and
    the val prediction adds the val row's center back.

    Returns (out, rounds, n_uncovered) with the artifact schema
    [game_id, fold_idx, pred_home, pred_away, resid_home, resid_away,
     best_iter_home, best_iter_away]; resid = actual − pred (artifact
    convention), so pred + resid == actual exactly on covered rows.
    """
    from nfl_per_side_engine import _fit_side  # noqa: PLC0415 — reuse fitter
    missing_cols = [c for c in CENTER_COLS if c not in frame.columns]
    if missing_cols:
        raise ValueError(
            f"oof_centered_per_side: frame missing center columns "
            f"{missing_cols} — run compute_centers/attach_centers first")

    parts: list[pd.DataFrame] = []
    best_home: list[int] = []
    best_away: list[int] = []
    folds = [f.copy() for f in folds]

    for i, f in enumerate(folds):
        tr, va = f["train"].copy(), f["val"].copy()
        tr_max = pd.to_datetime(tr["gameday"]).max()
        va_min = pd.to_datetime(va["gameday"]).min()
        if not (tr_max < va_min):
            raise AssertionError(
                f"fold {i}: train max {tr_max} not strictly before "
                f"val min {va_min} → leakage-safe split violated")

        id_cols = [c for c in ("game_id", "gameday") if c in va.columns]
        cols = (features + [SCORE_COL["home"], SCORE_COL["away"]]
                + CENTER_COLS)
        tr_valid = tr[cols].dropna()
        va_valid = va[id_cols + cols].dropna()
        if len(tr_valid) < 30 or len(va_valid) < 5:
            logger.warning("era engine: fold %d too small (tr=%d, va=%d), skipping",
                           i, len(tr_valid), len(va_valid))
            continue

        X_tr = tr_valid[features].to_numpy(float)
        X_va = va_valid[features].to_numpy(float)
        rec: dict[str, Any] = {"game_id": va_valid["game_id"].values,
                               "fold_idx": i}
        for side in SIDES:
            sc, cc = SCORE_COL[side], CENTER_COLS[0 if side == "home" else 1]
            y_tr = tr_valid[sc].to_numpy(float) - tr_valid[cc].to_numpy(float)
            y_va = va_valid[sc].to_numpy(float) - va_valid[cc].to_numpy(float)
            _m, pred_c, best = _fit_side(family, X_tr, y_tr, X_va, y_va)
            pred = np.round(pred_c + va_valid[cc].to_numpy(float), 4)
            rec[PRED_COL[side]] = pred
            rec[RESID_COL[side]] = np.round(
                va_valid[sc].to_numpy(float) - pred, 4)
            rec[f"best_iter_{side}"] = best
            (best_home if side == "home" else best_away).append(best)
        parts.append(pd.DataFrame(rec))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["game_id", "fold_idx"] + [PRED_COL[s] for s in SIDES]
                + [RESID_COL[s] for s in SIDES]
                + ["best_iter_home", "best_iter_away"])
    rounds = {"home": int(np.median(best_home)) if best_home else 2000,
              "away": int(np.median(best_away)) if best_away else 2000}
    covered = set(out["game_id"]) if len(out) else set()
    n_uncovered = int((~frame["game_id"].isin(covered)).sum())
    return out, rounds, n_uncovered


def refit_centered_per_side(decided: pd.DataFrame, pred_df: pd.DataFrame,
                            n_rounds: int | dict[str, int],
                            features: list[str], family: str = "lgb"
                            ) -> pd.DataFrame:
    """Fit-only refill on centered targets at fixed per-side rounds.

    Mirror of ``nfl_per_side_engine.refit_per_side``: fit (target − center)
    once on all ``decided`` rows at the per-side median rounds (no early
    stopping — a val split would consume sealed rows), predict centered then
    add each predicted row's center back. Used for sealed-2025 evaluation
    (fit 2019-24 → predict 2025) with the E2 arm's rounds.
    """
    missing_cols = [c for c in CENTER_COLS if c not in decided.columns
                    or c not in pred_df.columns]
    if missing_cols:
        raise ValueError(
            f"refit_centered_per_side: missing center columns {missing_cols}")
    from nfl_per_side_engine import (  # noqa: PLC0415 — reuse the family params
        LGB_PARAMS, RF_PARAMS, XGB_PARAMS)
    available = [f for f in features
                 if f in decided.columns and f in pred_df.columns]
    cols = available + [SCORE_COL[s] for s in SIDES] + CENTER_COLS
    cols = [c for c in cols if c in decided.columns]
    valid_decided = decided[cols].dropna()
    pred_cols = ["game_id"] + available + CENTER_COLS
    valid_pred = pred_df[pred_cols].dropna()
    X_tr = valid_decided[available].to_numpy(float)
    X_pred = valid_pred[available].to_numpy(float)
    result = valid_pred[["game_id"]].copy()
    for side in SIDES:
        n_fixed = int(n_rounds[side]) if isinstance(n_rounds, dict) \
            else int(n_rounds)
        sc = SCORE_COL[side]
        cc = CENTER_COLS[0 if side == "home" else 1]
        y_tr = valid_decided[sc].to_numpy(float) \
            - valid_decided[cc].to_numpy(float)
        if family == "lgb":
            from lightgbm import LGBMRegressor, log_evaluation  # noqa: PLC0415
            model = LGBMRegressor(**{**LGB_PARAMS, "n_estimators": n_fixed})
            model.fit(X_tr, y_tr, callbacks=[log_evaluation(period=0)])
            preds = model.predict(X_pred)
        elif family == "xgb":
            from xgboost import XGBRegressor  # noqa: PLC0415
            model = XGBRegressor(**{**XGB_PARAMS, "n_estimators": n_fixed})
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_pred)
        else:
            from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415
            n = max(n_fixed, int(RF_PARAMS.get("n_estimators", 300)))
            model = RandomForestRegressor(**{**RF_PARAMS, "n_estimators": n})
            model.fit(X_tr, y_tr)
            preds = model.predict(X_pred)
        result[PRED_COL[side]] = np.round(
            preds + valid_pred[cc].to_numpy(float), 4)
    return result


# ── Step-0 diagnostic helpers (tables for the record + gate) ─────────────────

def season_fact_table(frame: pd.DataFrame) -> list[dict]:
    """0a: per-season n, mean home/away score, mean total, home win rate."""
    rows = []
    for s, g in frame.groupby("season"):
        rows.append({
            "season": int(s), "n": int(len(g)),
            "mean_home_score": round(float(g["home_score"].mean()), 3),
            "mean_away_score": round(float(g["away_score"].mean()), 3),
            "mean_total": round(float(g["total"].mean()), 3),
            "home_win_rate": round(float((g["home_score"]
                                          > g["away_score"]).mean()), 4),
            "n_final_ties": int((g["home_score"] == g["away_score"]).sum()),
        })
    return sorted(rows, key=lambda d: d["season"])


def mean_resid_stats(sub: pd.DataFrame, pred_col: str, actual_col: str
                     ) -> dict:
    """resid = actual − pred (artifact convention; negative ⇒ prediction
    runs HIGH)."""
    r = sub[actual_col].to_numpy(float) - sub[pred_col].to_numpy(float)
    return {"n": int(len(r)), "mean_resid": round(float(r.mean()), 4),
            "rmse": round(float(np.sqrt(np.mean(r ** 2))), 4)}


def bias_by_season(sub: pd.DataFrame, pred_col: str, actual_col: str
                   ) -> list[dict]:
    """0b: per-season mean resid (actual − pred), pinned sign convention."""
    out = []
    for s, g in sub.groupby("season"):
        st = mean_resid_stats(g, pred_col, actual_col)
        st["season"] = int(s)
        out.append(st)
    return sorted(out, key=lambda d: d["season"])


def week_half_split(sub: pd.DataFrame, pred_col: str, actual_col: str,
                    reg_week_max: int = 18) -> list[dict]:
    """0c: mean resid in weeks 1-5 vs 6+ (regular weeks only)."""
    reg = sub[sub["week"] <= reg_week_max].copy()
    out = []
    for s, g in reg.groupby("season"):
        a = g[g["week"] <= 5]
        b = g[g["week"] > 5]
        row = {"season": int(s),
               "w1_5_n": int(len(a)),
               "w1_5_mean_resid": (mean_resid_stats(a, pred_col, actual_col)
                                   ["mean_resid"] if len(a) else None),
               "w6_plus_n": int(len(b)),
               "w6_plus_mean_resid": (mean_resid_stats(b, pred_col, actual_col)
                                      ["mean_resid"] if len(b) else None)}
        out.append(row)
    return sorted(out, key=lambda d: d["season"])


def center_bias_by_season(frame: pd.DataFrame,
                          centers: dict[str, pd.DataFrame]
                          ) -> dict[str, Any]:
    """0d: model-free per-spec bias (center − actual) by season, both sides.

    ``centers`` maps spec → the compute_centers output DataFrame. Returns
    per-side per-season mean bias per spec + the LGB-free framing numbers
    used by the Step-0 gate (mean |bias| 2021-23 per spec).
    """
    out: dict[str, Any] = {"per_side": {}, "summary": {}}
    for side in SIDES:
        actual_col = SCORE_COL[side]
        act = {gid: float(v) for gid, v in
               zip(frame["game_id"], frame[actual_col].to_numpy(float))}
        tbl = {}
        for spec, cdf in centers.items():
            ccol = CENTER_COLS[0 if side == "home" else 1]
            cvals = dict(zip(cdf["game_id"], cdf[ccol].to_numpy(float)))
            seasons = []
            for s, g in frame.groupby("season"):
                if int(s) < 2021:
                    continue
                b = [cvals[gid] - act[gid] for gid in g["game_id"]
                     if gid in cvals]
                seasons.append({"season": int(s), "n": int(len(b)),
                                "mean_bias": round(float(np.mean(b)), 4)
                                if b else None})
            seasons.sort(key=lambda d: d["season"])
            m2123 = [d["mean_bias"] for d in seasons
                     if d["season"] in (2021, 2022, 2023)
                     and d["mean_bias"] is not None]
            tbl[spec] = {
                "by_season": seasons,
                "mean_abs_bias_2021_23": round(
                    float(np.mean(np.abs(m2123))), 4) if m2123 else None,
            }
        out["per_side"][side] = tbl
    return out
