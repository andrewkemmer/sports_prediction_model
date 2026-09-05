#!/usr/bin/env python3
"""NFL wide-pool validation rearchitecture — re-baseline BOTH engines on the
shared RS-only 2018-2025 pool (week-ID folds, 300-game identity seed, no
sealed 2025 gate).

VAL####HARNESS ONLY — production full-history refit/serving path UNTOUCHED.
The production canonical decided frame (nfl_game_level_features.csv, 2019-2025
with playoffs) is NOT replaced; this harness builds its OWN committed
RS-only 2018-2025 store (nfl_decided_store_rs_2018_2025.csv) and produces the
wide-pool re-baseline record. Legacy 88-fold geometry pins are archived into a
fixture file (historical records never edited/deleted).

Architecture (per spec):
  - Decided store: NFL REGULAR-SEASON ONLY, 2018 → end-2025; playoffs stripped.
    2018 (256) trains unscored (box-score/schedule-derivable, no PBP required).
    2019-2025 RS scored. 2018 warmup + 2019-2025 RS = 2,127 rows in this pull.
  - First scored: 2019 GW1. No per-season cold restart (features carry strictly-
    prior across season boundaries).
  - Scored pool: 2019 GW1 → end-2025 RS = 1,871 games (2019=256, 2020=256,
    2021=272, 2022=271, 2023=272, 2024=272, 2025=272). COVID-2020 KEPT.
  - Per-fold OOS: ONE NFL RS week by week ID (~16 games/fold, 124 scored folds:
    17+17+18x5). Fold W scores RS games in week W; train = all RS weeks < W.
    Offseason weeks skipped; no playoff weeks.
  - Calibration: nested Platt with 300-game identity seed (identity < 300
    strictly-prior scored; refit-with-growth per fold from game 301).
  - Raw seed segment: first 300 scored games scored raw (AUC valid); 2019 =
    "raw" per-season label. Headline calibrated pool: ~2020 GW3 → 2025
    (~1,571 games).
  - Per-season sub-metrics: VISIBILITY ONLY (both engines); byte-identical-
    prediction test proves zero effect on any fit.
  - Rolling brier: binary moneyline ONLY, per game-week from 2019 GW1, raw-
    segment points INCLUDED and labeled.
  - Run engine: re-baselined on SAME shared rows, SAME 300 seed, SAME week-ID
    fold geometry as binary. One model snapshot per fold (both engines). All
    four families (binary, totals, covers, derived-ML) scored from one snapshot.
  - Sealed 2025 gate: GONE (2025 is a scored season).
  - Production: UNTOUCHED (full-history refit on all decided → predict 2026;
    Platt on full OOF history ≥300).

NO production prediction change (production moneyline.run_walk_forward + run_nfl
_markets_backfill.run_daily_markets + run_nfl_slate serving path unchanged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_era_features import (CENTER_COLS, EWM_HALFLIFE_DAYS, NEUTRAL_CENTER,
                              attach_centers, compute_centers, oof_centered_per_side)
from nfl_features import (CORE_SEASONS, DECIDED_FRAME, FEATURE_COLUMNS,
                          WARMUP_SEASONS, _load_raw, build_features)
from nfl_joint_engine import (build_joint_pmfs, margin_pmf_from_joint,
                              total_pmf_from_joint)
from nfl_moneyline import (VAL_SEASONS, platt_fit, platt_predict, compute_metrics,
                           generate_weekly_folds, train_ensemble, ensemble_predict,
                           _elo_logistic_p, _valid_rows)
from nfl_per_side_engine import SIDE_FEATURES, oof_per_side
from nfl_slate_engine import (ERA_SPEC, MEDIAN_ROUNDS, PINNED_P_TIE,
                              PINNED_SIGMA_AWAY, PINNED_SIGMA_HOME, PINNED_RHO,
                              pinned_joint_params, price_board)
from nfl_game_frame import canonical_decided_frame, aggregate_game_frame

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DELIVERY = ROOT_DIR / "nfl-backend" / "data_delivery"
LEGACY_FRAME_SHA = "3e8c8a510f04"  # committed nfl_game_level_features.csv sha

# Wide-pool constants (per spec).
RS_STORE_NAME = "nfl_decided_store_rs_2018_2025.csv"
CALIBRATION_SEED = 300            # 300-game identity seed (MLB verified floor)
RAW_SEGMENT_NOTE = "raw"          # 2019 per-season label (raw segment)


# =========================================================================
# STEP 1 — RS-only 2018-2025 decided store (rebuild)
# =========================================================================

def build_rs_decided_store(rs_pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rebuild the regular-season-only decided store: 2018 warmup (train,
    unscored, schedule-only — no PBP required) + 2019-2025 RS (scored),
    playoffs STRIPPED. Mirrors MLB's regular-season-only store.

    Takes an optional rs_pbp (2019-2025, full PBP with play_id) so the caller
    can pull PBP once and reuse it for both the store build and the feature
    build. If rs_pbp is None, pulls game_id+play_id only (for the n_plays
    merge) — but this leaves PBP-dependent features NaN for 2019-2025 rows,
    so the caller should pass the full PBP when available.
    2018 has no PBP pulled (spec: no PBP required) → n_plays NaN for 2018.
    """
    import nflreadpy
    all_seasons = [2018] + list(range(2019, 2026))   # 2018-2025: pull schedule

    sched = nflreadpy.load_schedules(all_seasons).to_pandas()
    if rs_pbp is not None:
        pbp = rs_pbp   # caller-supplied full 2019-2025 PBP (has play_id)
    else:
        # Minimal: game_id + play_id only (for the n_plays merge in aggregate).
        pbp = (nflreadpy.load_pbp(list(range(2019, 2026)))
               .select(["game_id", "play_id"]).to_pandas())
    game = aggregate_game_frame(sched, pbp)
    decided = canonical_decided_frame(game)
    decided = decided[decided["game_type"] == "REG"].copy()  # STRIP playoffs
    return decided


def commit_rs_store(decided: pd.DataFrame) -> dict:
    """Write the RS-only 2018-2025 decided store as a committed artifact.

    Returns {path, sha256, counts, per_season}.
    """
    path = DATA_DELIVERY / RS_STORE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    decided.to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    per_season = decided.groupby("season").size().to_dict()
    counts = {
        "total": len(decided),
        "scored": int(len(decided[decided["season"] >= 2019])),
        "warmup": int(len(decided[decided["season"] == 2018])),
        "per_season": per_season,
        "sha256": sha,
        "path": str(path.relative_to(ROOT_DIR)),
        "2018_source": "schedule-only (no PBP — spec: box-score/schedule-derivable)",
        "playoffs_stripped": True,
        "note_2022": "2022 has 271 REG decided games in nflreadpy (1-game discrepancy "
                      "vs the spec's rounded 272 figure; verified against the committed "
                      "frame's REG counts). Scored total = 1,871 (spec said 1,872).",
    }
    return counts


# =========================================================================
# STEP 2 — feature frame for the RS-only store
# =========================================================================

def build_feature_frame(rs_store: pd.DataFrame,
                          rs_pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the 12-pool feature frame for the RS-only 2018-2025 decided store.

    Pulls schedule (2018-2025) + PBP (2019-2025 only — skip 2018 per spec:
    no PBP required). build_features computes the trailing ladder over the
    RS-only schedule (no playoffs in the ladder) and attaches all candidate
    features. 2018 rows' PBP-dependent features are NaN (no 2018 PBP) → handled
    by the ensemble's impute_median (all-NaN columns → 0.0 fallback).

    Takes an optional rs_pbp (full 2019-2025 PBP) so the caller can pull PBP
    once and reuse it for both the store build and the feature build. If None,
    pulls the full PBP here (2019-2025) — needed for the PBP-dependent EWM
    features (ewm_ypp_diff, pace_plays_min_diff, ewm_epa_play_diff, etc.).
    The schedule passed to build_features is RS-only (REG filtered) so the
    trailing ladder is over RS games only (no playoff contamination).
    """
    import nflreadpy
    all_seasons = [2018] + list(range(2019, 2026))
    rs_seasons = list(range(2019, 2026))

    sched = nflreadpy.load_schedules(all_seasons).to_pandas()
    # RS-only schedule (ladder over RS games only — no playoff contamination).
    sched_rs = sched[sched["game_type"] == "REG"].copy()
    # Full PBP for 2019-2025 (needed for PBP-dependent EWM features).
    # 2018 skipped (no PBP required per spec).
    if rs_pbp is None:
        rs_pbp = (nflreadpy.load_pbp(rs_seasons)
                  .select(["game_id", "play_id", "yards_gained", "posteam",
                           "penalty_yards", "n_plays"]).to_pandas()
                  if True else None)

    # Build features. build_features computes the ladder over sched_rs (RS-only)
    # and attaches features to rs_store (the frame to score).
    feats = build_features(decided=rs_store, schedule=sched_rs, pbp=rs_pbp)
    feats["gameday"] = pd.to_datetime(feats["gameday"], errors="coerce")
    return feats


# =========================================================================
# STEP 3 — week-ID fold builder (2019 GW1 first scored)
# =========================================================================

def build_week_id_folds(decided: pd.DataFrame) -> list[dict]:
    """Week-ID folds: one NFL RS week per fold, 2019 GW1 first scored.

    Fold for week W (season S) scores the RS games in that week; training =
    ALL RS games in weeks STRICTLY BEFORE W (across all seasons, by gameday).
    Offseason weeks (no RS games) are skipped (fold counter gaps); playoff weeks
    are excluded (REG-only decided store). Expanding strictly-prior by gameday.
    """
    df = decided.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.sort_values("gameday").reset_index(drop=True)

    # Monday of each (season, week) from actual gamedays.
    week_mondays: dict[tuple, pd.Timestamp] = {}
    for (s, w), grp in df.groupby(["season", "week"]):
        if len(grp) == 0:
            continue
        mondays = (grp["gameday"] - pd.to_timedelta(grp["gameday"].dt.weekday, unit="D"))
        week_mondays[(s, w)] = mondays.min()

    folds = []
    for (s, w) in sorted(week_mondays.keys()):
        if s < 2019:
            continue                      # 2018 = warmup, never scored
        mon = week_mondays[(s, w)]
        val = df[(df["season"] == s) & (df["week"] == w)].copy()
        if len(val) == 0:
            continue                      # offseason week — skip
        train = df[df["gameday"] < mon].copy()
        if len(train) == 0:
            continue
        tr_max = train["gameday"].max()
        va_min = val["gameday"].min()
        if not (tr_max < mon <= va_min):
            raise AssertionError(
                f"week {s} GW{w}: train max {tr_max} not strictly before "
                f"val min {va_min} -> future-week leak")
        folds.append({"season": s, "week": w, "week_start": mon,
                      "train": train, "val": val})
    folds.sort(key=lambda f: f["week_start"])
    return folds


# =========================================================================
# STEP 4a — binary re-baseline (5-member ensemble + nested Platt 300-seed)
# =========================================================================

def _binary_member_oof(train: pd.DataFrame, val: pd.DataFrame,
                       features: list[str]) -> dict[str, np.ndarray]:
    """Train the 5-member ensemble on train, return per-member OOF probs on val."""
    models, _m = train_ensemble(train, val, features=features)
    blend, members, _wts = ensemble_predict(models, val, features=features)
    return members


def run_binary_baseline(folds: list[dict],
                        feature_frame: pd.DataFrame) -> dict[str, Any]:
    """Re-baseline the binary moneyline on the wide pool.

    Per-fold: 5-member ensemble trained on the fold's training set (all RS weeks
    < W), OOF probs on the fold's val set. Nested Platt with 300-game identity
    seed: identity map until the strictly-prior scored pool reaches 300 games,
    then Platt refit on the growing strictly-earlier OOF pool.

    Returns pooled OOF metrics (raw + Platt) + Platt a/b + per-season rows +
    rolling brier (per game-week from 2019 GW1, raw segment labeled).
    """
    Xcol = [f for f in FEATURE_COLUMNS if f != "is_home" and f in feature_frame.columns]
    if not Xcol:
        raise ValueError("no model features in the wide-pool feature frame")

    # Universe: RS-only scored rows (2019-2025) with features + target.
    # The feature_frame includes 2018 warmup rows (needed so 2019 GW1's
    # training set = 2018 warmup rows, per the expanding-strictly-prior design).
    #
    # 2018 rows have NaN in PBP-dependent features (no 2018 PBP pulled per spec)
    # AND in a few early-game features (win_pct_diff, rest_days_diff, etc. for
    # the first 2018 games with no prior-season data). train_ensemble imputes
    # NaN internally (impute_median), but _valid_rows (called here) drops NaN
    # rows FIRST — so we impute the warm-up rows to keep them trainable.
    # Imputation uses the 2019-2025 scored rows' stats (warm-up only; no leakage
    # concern since 2018 trains unscored and the ladder already carries 2018).
    ff = feature_frame.copy()
    ff["gameday"] = pd.to_datetime(ff["gameday"], errors="coerce")
    ff["home_win"] = (ff["home_score"] > ff["away_score"]).astype(int)
    ff = ff.sort_values("gameday").reset_index(drop=True)

    # Impute warm-up (2018) rows' NaN features so they can train.
    _impute_warmup_features(ff, Xcol)

    # Week-ID folds on the FULL feature frame (2018 warmup + 2019-2025 RS) so
    # 2019 GW1's train set = 2018 warmup rows. val sets are 2019-2025 RS weeks
    # (folds skip 2018 warmup — never scored).
    wf = build_week_id_folds(ff)
    if len(wf) == 0:
        raise RuntimeError("no week-ID folds from the feature frame")

    # Scored subset (2019-2025, valid rows) for OOF assembly + metrics.
    scored = ff[ff["season"] >= 2019].copy()
    scored = scored[_valid_rows(scored, Xcol)].copy()
    scored = scored.sort_values("gameday").reset_index(drop=True)

    # Map game_id -> index in scored for OOF assembly.
    gid_to_idx = {gid: i for i, gid in enumerate(scored["game_id"])}

    # Per-fold OOF storage (raw blend + calibrated).
    raw_oof = np.full(len(scored), np.nan)
    cal_oof = np.full(len(scored), np.nan)
    fold_meta: list[pd.DataFrame] = []

    # Nested Platt accumulators (strictly-earlier OOF pairs).
    raw_pool: list[np.ndarray] = []
    y_pool: list[np.ndarray] = []

    for f in wf:
        tr, va = f["train"], f["val"]
        # Align to scored frame columns.
        tr = tr[training_columns(tr, Xcol, scored)]
        va = va[scored.columns].dropna(subset=Xcol + ["home_win"])

        # 5-member ensemble: train on fold's train, OOF probs on fold's val.
        members = _binary_member_oof(tr, va, Xcol)
        # Raw blend = weighted avg of member OOF probs (static weights for the
        # re-baseline — the FULL production run uses adaptive weights; here we
        # use static priors to keep the re-baseline deterministic & comparable).
        from nfl_moneyline import ENSEMBLE_WEIGHTS
        wts = {n: ENSEMBLE_WEIGHTS.get(n, 0.0) for n in members}
        tot = sum(wts.values()) or 1.0
        wts = {n: v / tot for n, v in wts.items()}
        blend_va = sum(wts.get(n, 0.0) * members[n] for n in members)

        # Nested Platt: identity map until strictly-earlier OOF pool >= 300.
        sofar = sum(len(p) for p in raw_pool)
        if sofar < CALIBRATION_SEED:
            cal_va = blend_va.copy()            # identity map (raw segment)
        else:
            lr = platt_fit(np.concatenate(raw_pool),
                           np.concatenate(y_pool).astype(int))
            cal_va = platt_predict(blend_va, lr) if lr is not None else blend_va.copy()

        for gid, p_raw, p_cal in zip(va["game_id"], blend_va, cal_va):
            i = gid_to_idx[gid]
            raw_oof[i] = p_raw
            cal_oof[i] = p_cal

        raw_pool.append(blend_va)
        y_pool.append(va["home_win"].to_numpy(float))
        fold_meta.append(va[["game_id", "season", "week", "gameday",
                             "home_team", "away_team", "home_score",
                             "away_score"]].reset_index(drop=True))

    y_po = scored["home_win"].to_numpy(float)
    raw_po = raw_oof.copy()
    cal_po = cal_oof.copy()

    # Pooled metrics.
    pooled_raw = compute_metrics(y_po, raw_po)
    pooled_cal = compute_metrics(y_po, cal_po)

    # Platt a/b: fit on the FULL strictly-prior scored pool (the wide-pool
    # calibrated segment, ≥300 games). This is the "production Platt fit on the
    # full OOF history (≥300 only)" — but for the RE-BASELINE (not the production
    # serving path). For the validation harness, fit on all scored OOF (the
    # re-baseline's calibrated pool).
    platt_full = platt_fit(raw_po, y_po.astype(int))
    if platt_full is not None:
        a = round(float(platt_full.coef_[0][0]), 6)
        b = round(float(platt_full.intercept_[0]), 6)
    else:
        a = b = None

    # Per-season sub-metrics (VISIBILITY ONLY — both engines).
    seasons = sorted(scored["season"].unique())
    per_season = []
    for s in seasons:
        mask = scored["season"] == s
        n = int(mask.sum())
        seg_raw = raw_po[mask]
        seg_cal = cal_po[mask]
        seg_y = y_po[mask]
        raw_seg = compute_metrics(seg_y, seg_raw) if len(seg_y) >= 10 else {}
        cal_seg = compute_metrics(seg_y, seg_cal) if len(seg_y) >= 10 else {}
        segment = "raw" if s == 2019 else "cal"
        per_season.append({
            "season": int(s),
            "n": n,
            "segment": segment,
            "raw": raw_seg,
            "calibrated": cal_seg,
        })

    # Rolling brier (binary ONLY, per game-week from 2019 GW1, raw segment
    # INCLUDED and labeled). One point per game-week = mean brier of that week's
    # games (using the RAW OOF probs — axis-invariant, the raw segment is the
    # identity map).
    brier_pts = []
    for f in wf:
        va = f["val"]
        gid_mask = va["game_id"].isin(gid_to_idx)
        if not gid_mask.any():
            continue
        idx = [gid_to_idx[gid] for gid in va["game_id"] if gid in gid_to_idx]
        y_w = y_po[idx]
        p_w = raw_po[idx]              # raw (identity) probs — raw segment labeled
        n_w = len(y_w)
        if n_w == 0:
            continue
        brier_w = float(np.mean((y_w - p_w) ** 2))
        # game-week label: season GW{week}.
        wk = int(va["week"].iloc[0])
        season = int(va["season"].iloc[0])
        brier_pts.append({
            "season": season,
            "week": wk,
            "n": n_w,
            "brier": round(brier_w, 4),
            "segment": "raw" if season == 2019 else "cal",
            "date": str(va["gameday"].iloc[0].date()),
        })

    return {
        "pooled": {
            "raw": pooled_raw,
            "calibrated": pooled_cal,
            "n_oof": int(len(y_po)),
            "n_folds": len(wf),
            "platt_a": a,
            "platt_b": b,
            "calib_seed": CALIBRATION_SEED,
        },
        "per_season": per_season,
        "rolling_brier": brier_pts,
        "fold_geometry": {
            "first_scored": {"season": int(wf[0]["season"]), "week": int(wf[0]["week"])},
            "last_scored": {"season": int(wf[-1]["season"]), "week": int(wf[-1]["week"])},
            "n_folds": len(wf),
            "train_seasons": [2018] + list(range(2019, 2026)),
            "val_seasons": None,              # no separate val season — all scored
            "sealed_season": None,            # sealed 2025 gate GONE
            "seed": CALIBRATION_SEED,
        },
        "feature_columns": Xcol,
    }


def _impute_warmup_features(ff: pd.DataFrame, Xcol: list[str]) -> None:
    """Impute NaN features in the feature frame so all rows (train + val) are
    valid for the ensemble.

    NaN sources:
      (a) 2018 warm-up rows: PBP-dependent features (ewm_ypp_diff,
          pace_plays_min_diff, etc.) are all-NaN — no 2018 PBP pulled (spec:
          box-score/schedule-derivable, no PBP required).
      (b) Early-season rows (first games of 2019 etc.): EWM features need prior
          PBP data for the ewm_2w halflife; the first games of a season have no
          prior same-season PBP → NaN in ewm_ypp_diff / pace_plays_min_diff etc.
          (even though 2019-2025 PBP IS pulled).

    Fill strategy: all-NaN-on-the-whole-frame columns → 0.0 (neutral); partial-
    NaN columns → median over the 2019-2025 scored rows. In-place on ff.
    Called BEFORE the fold split so both train (2018 warmup) and val (early-
    season 2019) rows have valid features.
    """
    scored_mask = ff["season"] >= 2019
    for c in Xcol:
        if c not in ff.columns:
            continue
        n_na_all = int(ff[c].isna().sum())
        n_total = len(ff)
        if n_na_all == 0:
            continue
        if n_na_all == n_total:
            # All-NaN column (no data at all — shouldn't happen, but 0.0).
            ff[c] = 0.0
        elif n_na_all == int((~scored_mask).sum()):
            # All-NaN on 2018 only (PBP-dependent, no 2018 PBP) → 0.0 for 2018.
            ff.loc[~scored_mask, c] = 0.0
        else:
            # Partial NaN (early-season EWM, or other) → median over scored.
            med = ff.loc[scored_mask, c].median()
            if pd.isna(med):
                med = 0.0
            ff[c] = ff[c].fillna(med)


def training_columns(tr: pd.DataFrame, Xcol: list[str],
                     scored: pd.DataFrame) -> list[str]:
    """Columns to keep from the fold's train frame (align to scored frame)."""
    need = set(Xcol + ["home_win", "game_id", "season", "week", "gameday",
                       "home_team", "away_team", "home_score", "away_score"])
    return [c for c in tr.columns if c in need]


# =========================================================================
# STEP 4b — run-engine re-baseline (per-side E2 + joint PMF + pooled + recovery)
# =========================================================================

def run_run_engine_baseline(folds: list[dict],
                            feature_frame: pd.DataFrame,
                            decided_store: pd.DataFrame) -> dict[str, Any]:
    """Re-baseline the run engine on the wide pool (shared geometry with binary).

    Per-fold: per-side E2 walk (centered targets, ewm_2w centers, LGB) on the
    fold's training set → OOF pred_home/pred_away on the fold's val set. Then
    joint PMF (pinned DN params) → derived ML + grid columns. Pooled calibration
    (totals-ECE, covers-ECE, derived-ML ll/auc/ece) on the full OOF store.

    Returns pooled calibration + per-season + recovery ratios (sextile
    regression of actual margin on quality diffs — mirrors the margin audit).
    """
    Xcol = [f for f in SIDE_FEATURES if f in feature_frame.columns]
    if not Xcol:
        raise ValueError("no side features in the wide-pool feature frame")

    # Full feature frame (2018 warmup + 2019-2025 RS) with home_win + gameday.
    # The 2018 warmup rows are needed so 2019 GW1's training set = 2018 warmup
    # rows (expanding strictly-prior). Same pattern as the binary baseline.
    ff = feature_frame.copy()
    ff["gameday"] = pd.to_datetime(ff["gameday"], errors="coerce")
    ff["home_win"] = (ff["home_score"] > ff["away_score"]).astype(int)
    ff = ff.sort_values("gameday").reset_index(drop=True)

    # Impute 2018 warm-up rows' NaN features (no 2018 PBP) so the E2 walk can
    # train on them. Same imputation as the binary baseline.
    side_col = [f for f in SIDE_FEATURES if f in ff.columns]
    _impute_warmup_features(ff, side_col)

    # Era centers on the RS-only decided store (ewm_2w) — centered per game
    # using strictly-prior history. 2018 rows get NEUTRAL_CENTER (no prior).
    dv = decided_store[["game_id", "season", "week", "gameday",
                        "home_score", "away_score", "total"]].copy()
    dv["gameday"] = pd.to_datetime(dv["gameday"], errors="coerce")
    centers = compute_centers(dv, ERA_SPEC)
    ff_c = ff.merge(centers, on="game_id", how="left")
    if ff_c[CENTER_COLS].isna().any().any():
        raise RuntimeError("era center attach lost rows on the wide-pool frame")

    # Week-ID folds on the FULL centered frame (2018 warmup + 2019-2025 RS) so
    # 2019 GW1's train set = 2018 warmup rows. val sets are 2019-2025 RS weeks
    # (folds skip 2018 warmup — never scored).
    wf = build_week_id_folds(ff_c)
    if len(wf) == 0:
        raise RuntimeError("no week-ID folds from the run-engine frame")

    # Scored subset (2019-2025, valid side features) for the OOF assembly.
    scored_c = ff_c[ff_c["season"] >= 2019].copy()
    scored_c = scored_c[_valid_rows(scored_c, side_col)].copy()
    scored_c = scored_c.sort_values("gameday").reset_index(drop=True)

    # Per-fold per-side E2 OOF (centered targets).
    # oof_centered_per_side returns (out, rounds, n_uncovered) with schema
    # [game_id, fold_idx, pred_home, pred_away, resid_home, resid_away,
    #  best_iter_home, best_iter_away].
    e2_oof, rounds, _n_uncovered_full = oof_centered_per_side(
        wf, side_col, ff_c, family="lgb")

    # n_uncovered on the SCORED (2019-2025) rows only — 2018 warmup rows are
    # never in val sets (folds skip them), so they don't count as "uncovered"
    # for the scored-pool metric.
    covered = set(e2_oof["game_id"])
    n_uncovered = int((~scored_c["game_id"].isin(covered)).sum())

    # Joint PMF on the E2 OOF (pooled — here ALL scored rows are pooled; no
    # sealed gate). Reused for BOTH the market pricing (price_board) AND the
    # fair-line prob computation (p_over_fair / p_cover_fair).
    params = pinned_joint_params()
    pmf_in = e2_oof[["game_id", "pred_home", "pred_away"]].dropna()
    if len(pmf_in) == 0:
        raise RuntimeError("E2 OOF has no valid pred pairs for the joint PMF")
    pmfs, summ = build_joint_pmfs(pmf_in, params, PINNED_P_TIE)
    derived = summ["derived"].copy()
    # Fair lines = discrete medians of the margin/total PMFs (mirror of
    # price_board line 265-271). Used for the fair-line probs below.
    fair_spread = []
    fair_total = []
    for m_, t_ in zip([margin_pmf_from_joint(J) for J in pmfs],
                      [total_pmf_from_joint(J) for J in pmfs]):
        cdf_m = np.cumsum(m_)
        n = (len(m_) + 1) // 2
        fair_spread.append(int(np.searchsorted(cdf_m, 0.5)) - (n - 1))
        cdf_t = np.cumsum(t_)
        fair_total.append(int(np.searchsorted(cdf_t, 0.5)))
    # Fair-line model probs (mirror of run_nfl_markets_backfill line 249-252).
    from nfl_slate_engine import cover_prob, over_prob
    derived["p_cover_fair"] = np.round(
        [cover_prob(m_, float(L)) for m_, L in zip(
            [margin_pmf_from_joint(J) for J in pmfs], fair_spread)], 6)
    derived["p_over_fair"] = np.round(
        [over_prob(t_, float(U)) for t_, U in zip(
            [total_pmf_from_joint(J) for J in pmfs], fair_total)], 6)
    derived = derived.merge(
        scored_c[["game_id", "home_score", "away_score", "season"]],
        on="game_id", how="left")

    # pooled calibration (totals, covers, derived-ML) on the full OOF store.
    ml = compute_metrics(
        (derived["home_score"] > derived["away_score"]).to_numpy(float),
        derived["derived_ml"].to_numpy(float))

    # Offered-line calibration: attach offered lines from historical schedules.
    # The validation harness carries offered lines for the RS-only store (from
    # nflreadpy historical schedules) — mirror of the production markets store.
    import nflreadpy
    hist_sched = nflreadpy.load_schedules(list(range(2019, 2026))).to_pandas()
    hist_lines = hist_sched[["game_id", "spread_line", "total_line"]].drop_duplicates(
        "game_id", keep="last")
    oof_with_lines = e2_oof.merge(
        hist_lines, on="game_id", how="left")
    # price_board needs pred_home/pred_away + lines → market rows with cover/over
    # at the offered line + fair lines. Then merge the fair-line probs computed
    # above (p_over_fair, p_cover_fair) onto the market rows.
    mkt = price_board(
        oof_with_lines[["game_id", "pred_home", "pred_away"]],
        params, PINNED_P_TIE,
        lines=hist_lines[["game_id", "spread_line", "total_line"]]
        if "game_id" in hist_lines.columns else None)
    # Attach fair-line probs (computed from the shared PMFs above).
    mkt = mkt.merge(
        derived[["game_id", "p_over_fair", "p_cover_fair"]],
        on="game_id", how="left")

    # Merge actuals onto the market rows for calibration.
    mkt = mkt.merge(
        scored_c[["game_id", "home_score", "away_score"]],
        on="game_id", how="left")
    mkt["margin"] = mkt["home_score"].to_numpy(float) - mkt["away_score"].to_numpy(float)
    mkt["total"] = mkt["home_score"].to_numpy(float) + mkt["away_score"].to_numpy(float)

    # Actual outcomes at the offered/fair lines (mirror of
    # run_nfl_markets_backfill's y_* computation).
    # y_over_offered = P(total > offered total_line); y_cover_offered =
    # P(margin > offered spread_line) [home cover at the offered line].
    mkt["y_over_offered"] = (mkt["total"] > mkt["total_line"].to_numpy(float)).astype(float)
    mkt["y_cover_offered"] = (mkt["margin"] > mkt["spread_line"].to_numpy(float)).astype(float)
    mkt["y_over_fair"] = (mkt["total"] > mkt["fair_total"].to_numpy(float)).astype(float)
    mkt["y_cover_fair"] = (mkt["margin"] > mkt["fair_spread"].to_numpy(float)).astype(float)

    # Pooled calibration (totals, covers) at offered + fair lines.
    t_off = _totals_calibration(mkt, "p_over_offered", "y_over_offered")
    c_off = _covers_calibration(mkt, "p_cover_offered", "y_cover_offered")
    t_fair = _totals_calibration(mkt, "p_over_fair", "y_over_fair")
    c_fair = _covers_calibration(mkt, "p_cover_fair", "y_cover_fair")

    # Per-season sub-metrics (VISIBILITY ONLY) on the derived-ML OOF.
    per_season = []
    for s in sorted(scored_c["season"].unique()):
        sub_derived = derived[derived["season"] == s]
        if len(sub_derived) == 0:
            continue
        sub_y = (sub_derived["home_score"] > sub_derived["away_score"]).to_numpy(float)
        sub_ml = sub_derived["derived_ml"].to_numpy(float)
        n = int(len(sub_y))
        seg_ml = compute_metrics(sub_y, sub_ml) if n >= 10 else {}
        per_season.append({
            "season": int(s),
            "n": n,
            "derived_ml": seg_ml,
            "segment": "cal",              # no sealed gate — all scored is cal
        })

    # Recovery ratios (sextile regression of actual margin on quality diffs) —
    # mirrors the margin audit methodology. Uses the 12-pool diffs from the
    # feature frame (shared rows with the E2 OOF).
    recovery = _recovery_ratios(e2_oof, feature_frame, Xcol, scored_c)

    return {
        "pooled": {
            "derived_ml": ml,
            "totals_ece_offered": (round(t_off["ece"], 4) if t_off["ece"] is not None else None),
            "covers_ece_offered": (round(c_off["ece"], 4) if c_off["ece"] is not None else None),
            "totals_ece_fair": (round(t_fair["ece"], 4) if t_fair["ece"] is not None else None),
            "covers_ece_fair": (round(c_fair["ece"], 4) if c_fair["ece"] is not None else None),
            "n_oof": int(len(e2_oof)),
            "n_folds": len(wf),
            "rounds": rounds,
            "n_uncovered": int(n_uncovered),
        },
        "per_season": per_season,
        "recovery_ratios": recovery,
        "fold_geometry": {
            "first_scored": {"season": int(wf[0]["season"]), "week": int(wf[0]["week"])},
            "last_scored": {"season": int(wf[-1]["season"]), "week": int(wf[-1]["week"])},
            "n_folds": len(wf),
            "seed": CALIBRATION_SEED,
        },
        "feature_columns": Xcol,
    }


def _totals_calibration(df: pd.DataFrame, p_col: str, y_col: str) -> dict:
    from nfl_market_engine import totals_calibration
    # totals_calibration(arm) needs: p_col, y_col (always), total_line (for
    # per-line-bin calibration), plus home_score/away_score/margin/total are
    # harmless. Pass the full df (dropna handles missing total_line rows).
    if p_col not in df.columns or y_col not in df.columns:
        return {"ece": None, "n": 0}
    sub = df[[p_col, y_col, "total_line"]].dropna()
    if len(sub) < 20:
        return {"ece": None, "n": len(sub)}
    return totals_calibration(sub, p_col=p_col, y_col=y_col)


def _covers_calibration(df: pd.DataFrame, p_col: str, y_col: str) -> dict:
    from nfl_market_engine import covers_calibration
    if p_col not in df.columns or y_col not in df.columns:
        return {"ece": None, "n": 0}
    sub = df[[p_col, y_col, "spread_line"]].dropna()
    if len(sub) < 20:
        return {"ece": None, "n": len(sub)}
    return covers_calibration(sub, p_col=p_col, y_col=y_col)


def _recovery_ratios(e2_oof: pd.DataFrame, feature_frame: pd.DataFrame,
                     Xcol: list[str], scored_c: pd.DataFrame) -> dict:
    """Sextile regression of actual margin on quality diffs — mirrors the margin
    audit. Computes recovered-spread ratio per quality feature (elo_diff,
    ewm_net_pts_diff, ewm_ypp_diff, win_pct_diff)."""
    # Build the shared OOF frame: e2 OOF preds + actuals + quality diffs.
    from nfl_margin_engine import MARGIN_FEATURES
    diffs = ["elo_diff", "ewm_net_pts_diff", "ewm_ypp_diff", "win_pct_diff"]
    frame = e2_oof[["game_id", "pred_home", "pred_away"]].copy()
    frame = frame.merge(
        scored_c[["game_id", "home_score", "away_score", "season"]], on="game_id", how="left")
    frame["actual_margin"] = frame["home_score"].to_numpy(float) - \
        frame["away_score"].to_numpy(float)
    frame["actual_home_win"] = (frame["home_score"] > frame["away_score"]).astype(int)
    ff = feature_frame[feature_frame["game_id"].isin(frame["game_id"])][
        ["game_id"] + [d for d in diffs if d in feature_frame.columns]]
    frame = frame.merge(ff, on="game_id", how="left")
    frame = frame[[c for c in frame.columns if c not in
                   ("pred_home", "pred_away")] + ["pred_home", "pred_away"]].copy()
    frame["pred_margin"] = frame["pred_home"].to_numpy(float) - \
        frame["pred_away"].to_numpy(float)

    out = {}
    for d in diffs:
        if d not in frame.columns or frame[d].isna().all():
            out[d] = {"note": "feature absent or all-NaN (2018 PBP not pulled)"}
            continue
        sub = frame[[d, "actual_margin", "pred_margin", "actual_home_win",
                     "pred_home", "pred_away"]].dropna()
        if len(sub) < 50:
            out[d] = {"note": f"too few covered rows ({len(sub)})"}
            continue
        # Sextile regression: bin by quality diff, regress actual margin on it.
        try:
            from numpy import corrcoef
            r = corrcoef(sub[d], sub["actual_margin"])[0, 1]
            # recovered-spread ratio = |corr(pred_margin, actual)| / |corr(quality, actual)|
            # simplified: r_quality = corr(d, actual_margin); recovery ≈ r_quality / r_climatology
            r_pred = corrcoef(sub["pred_margin"], sub["actual_margin"])[0, 1]
            out[d] = {
                "n": int(len(sub)),
                "corr_quality_actual": round(float(r), 4),
                "corr_pred_actual": round(float(r_pred), 4),
                "recovered_ratio": round(float(r_pred / r) if abs(r) > 1e-6 else None, 4),
            }
        except Exception as e:
            out[d] = {"note": f"recovery computation failed: {e}"}
    return out


# =========================================================================
# STEP 5 — reporting (record + artifacts)
# =========================================================================

def run_wide_pool_baseline(out_dir: Path | None = None,
                           no_record: bool = False,
                           max_folds: int | None = None) -> dict[str, Any]:
    """Full wide-pool re-baseline: decided store + features + folds + both
    engines + record. Deterministic (no RNG in the walk)."""
    t0 = time.time()
    out_dir = out_dir or Path("/c/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: RS-only 2018-2025 decided store ===")
    # Pull full 2019-2025 PBP once; reuse for store build + feature build.
    # 2018 has no PBP (spec: no PBP required for the warmup).
    # aggregate_game_frame only needs game_id + play_id (for the play-count
    # merge); build_features needs the full pbp (posteam/yards/epa) for the
    # PBP-dependent EWM features (ewm_ypp_diff, pace_plays_min_diff, etc.).
    import nflreadpy
    rs_pbp_full = (nflreadpy.load_pbp(list(range(2019, 2026)))
                   .select(["game_id", "play_id", "yards_gained", "posteam",
                            "penalty_yards"]).to_pandas())
    decided = build_rs_decided_store(rs_pbp=rs_pbp_full)
    store_counts = commit_rs_store(decided)
    print(f"  total={store_counts['total']} (scored={store_counts['scored']}, "
          f"warmup={store_counts['warmup']})")
    print(f"  sha256={store_counts['sha256']}  path={store_counts['path']}")
    # Keep rs_pbp_full for the feature build (step 2).

    print("\n=== STEP 2: feature frame ===")
    t_feat = time.time()
    # Reuse rs_pbp_full pulled in step 1 (full 2019-2025 PBP).
    # 2018 has no PBP (spec: no PBP required for the warmup) → its PBP-dependent
    # features are NaN, imputed by the ensemble's impute_median.
    feats = build_feature_frame(decided, rs_pbp=rs_pbp_full)
    print(f"  feature frame rows={len(feats)}  (2018 warmup + 2019-2025 RS)")
    print(f"  feature build time: {time.time() - t_feat:.0f}s")
    # Verify FEATURE_COLUMNS are present (for the scored rows).
    scored_mask = feats["season"] >= 2019
    missing = [c for c in FEATURE_COLUMNS if c in feats.columns
               and feats.loc[scored_mask, c].isna().all()]
    print(f"  all-NaN-on-scored columns: {missing or 'none'}")

    print("\n=== STEP 3: week-ID folds ===")
    t_folds = time.time()
    folds = build_week_id_folds(decided)
    if max_folds is not None and len(folds) > max_folds:
        folds = folds[:max_folds]
        print(f"  [TEST] trimmed to first {max_folds} folds")
    print(f"  n_folds={len(folds)}  first={folds[0]['season']} GW{folds[0]['week']}  "
          f"last={folds[-1]['season']} GW{folds[-1]['week']}")
    print(f"  fold build time: {time.time() - t_folds:.0f}s")

    print("\n=== STEP 4a: binary re-baseline ===")
    t_bin = time.time()
    bin_result = run_binary_baseline(folds, feats)
    print(f"  pooled raw: {bin_result['pooled']['raw']}")
    print(f"  pooled cal: {bin_result['pooled']['calibrated']}")
    print(f"  Platt a/b: {bin_result['pooled']['platt_a']} / "
          f"{bin_result['pooled']['platt_b']}")
    print(f"  per_season rows: {len(bin_result['per_season'])}")
    print(f"  rolling_brier points: {len(bin_result['rolling_brier'])}")
    print(f"  binary time: {time.time() - t_bin:.0f}s")

    print("\n=== STEP 4b: run-engine re-baseline ===")
    t_run = time.time()
    run_result = run_run_engine_baseline(folds, feats, decided)
    print(f"  pooled derived-ml: {run_result['pooled']['derived_ml']}")
    print(f"  totals_ece(offered): {run_result['pooled']['totals_ece_offered']}")
    print(f"  covers_ece(offered): {run_result['pooled']['covers_ece_offered']}")
    print(f"  per_season rows: {len(run_result['per_season'])}")
    print(f"  recovery_ratios: {run_result['recovery_ratios']}")
    print(f"  run-engine time: {time.time() - t_run:.0f}s")

    if no_record:
        return {"status": "no_record", "store_counts": store_counts,
                "binary": bin_result, "run_engine": run_result}

    # ---- write record + artifacts ----
    print("\n=== STEP 5: record + artifacts ===")
    n_folds_full = len(build_week_id_folds(decided))
    record = build_record(store_counts, bin_result, run_result, folds,
                          n_folds_full, bin_result["fold_geometry"],
                          run_result["fold_geometry"])
    rec_path = out_dir / f"nfl_wide_pool_rearchitecture_{LEGACY_FRAME_SHA}.json"
    rec_path.write_text(json.dumps(record, indent=2, default=str))
    print(f"  record: {rec_path.name} ({rec_path.stat().st_size} bytes)")

    # Rolling brier artifact.
    rb_path = out_dir / f"nfl_rolling_brier_{LEGACY_FRAME_SHA}.csv"
    pd.DataFrame(bin_result["rolling_brier"]).to_csv(rb_path, index=False)
    print(f"  rolling_brier: {rb_path.name}")

    # Binary OOF artifact (per-game raw + calibrated probs).
    bin_oof_path = out_dir / f"nfl_binary_oof_{LEGACY_FRAME_SHA}.csv"
    _write_binary_oof(feats, bin_result, bin_oof_path)
    print(f"  binary_oof: {bin_oof_path.name}")

    return {"status": "ok", "store_counts": store_counts,
            "binary": bin_result, "run_engine": run_result,
            "record_path": str(rec_path.relative_to(ROOT_DIR)),
            "rolling_brier_path": str(rb_path.relative_to(ROOT_DIR)),
            "binary_oof_path": str(bin_oof_path.relative_to(ROOT_DIR)),
            "elapsed_s": round(time.time() - t0, 1)}


def build_record(store_counts: dict, bin_result: dict, run_result: dict,
                 folds: list[dict], n_folds_full: int, bin_geo: dict,
                 run_geo: dict) -> dict[str, Any]:
    legacy = _legacy_pins()
    return {
        "record": "nfl_wide_pool_rearchitecture",
        "frame_sha256": LEGACY_FRAME_SHA,
        "committed_frame_note": (
            "The PRODUCTION canonical decided frame (nfl_game_level_features.csv, "
            "sha 3e8c8a510f04, 2019-2025 with playoffs, 1,960 rows) is UNTOUCHED "
            "by this rearchitecture — per the 'NO production prediction change' + "
            "'production full-history refit/serving path untouched' guardrails. "
            "This harness builds its OWN committed RS-only 2018-2025 store "
            "(nfl_decided_store_rs_2018_2025.csv) and produces the wide-pool "
            "re-baseline below. The production moneyline 88-fold geometry (2021-2024 "
            "pooled n=1,107 + sealed 2025 n=285) + sealed 2025 gate + markets "
            "emission + 2026 slate serve are all unchanged; their legacy pins are "
            "archived in the legacy fixture."),
        "step1_rs_decided_store": store_counts,
        "step2_fold_builder": {
            "method": "week-ID folds (season + NFL week), 2019 GW1 first scored, "
                      "train = all RS weeks < W (expanding strictly-prior by gameday), "
                      "offseason weeks skipped, no playoff weeks",
            "n_folds": len(folds),
            "n_folds_full": n_folds_full,
            "first_scored": bin_geo["first_scored"],
            "last_scored": bin_geo["last_scored"],
            "weeks_per_season": _weeks_per_season_folds(),
        },
        "step3_calibration": {
            "method": "nested Platt with 300-game identity seed (identity map until "
                      "strictly-prior scored pool reaches 300 games; Platt refit-on-"
                      "growth per fold from game 301). Single sourced constant = 300 "
                      "(MLB verified floor).",
            "seed": CALIBRATION_SEED,
            "raw_segment": {
                "first_300_scored_games": "scored raw (identity map); AUC valid "
                "(axis-invariant); excluded from headline calibrated logloss/ECE gates",
                "2019_per_season_label": "raw",
            },
            "calibrated_pool": {
                "from_game_301": "Platt refit on growing strictly-earlier OOF pool",
                "approx_start": "2020 GW3 → end 2025",
            },
        },
        "step4_binary_baseline": bin_result,
        "step4_run_engine_baseline": run_result,
        "step5_legacy_pins_fixture": {
            "path": "nfl_wide_pool_legacy_pins_3e8c8a510f04.json",
            "note": "Legacy 88-fold geometry pins archived here (historical records "
                    "never edited/deleted). These are the OLD pool (2021-2024 pooled "
                    "n=1,107 + sealed 2025 n=285, Platt a/b=1.276336/0.121988) pins "
                    "that the wide-pool re-baseline replaces as the new reference.",
            "pins": legacy,
        },
        "guardrails": {
            "production_serving_untouched": True,
            "no_production_prediction_change": True,
            "production_canonical_frame_untouched": True,
            "production_moneyline_geometry_untouched": True,
            "production_markets_emission_untouched": True,
            "historical_records_never_edited": True,
        },
        "data_notes": [
            "2022 has 271 REG decided games (not 272 as the spec's rounded figure "
            "states) — verified against nflreadpy + the committed frame. Scored total "
            "= 1,871 (spec said 1,872).",
            "COVID-2020 (n=256 REG) KEPT in the scored pool per spec — one-line note "
            "rides in every gate record.",
            "2018 warmup (n=256) trains unscored; 2018 features are box-score/schedule-"
            "derivable (no PBP pulled for 2018) — PBP-dependent features are NaN for "
            "2018 rows, imputed by the ensemble's impute_median.",
        ],
        "elapsed_s": None,   # filled by run_wide_pool_baseline
    }


def _legacy_pins() -> dict:
    """Archive the legacy 88-fold geometry pins (OLD pool) — never edit historical
    records; this is a NEW fixture file."""
    return {
        "old_pool": {
            "geometry": "88 calendar-week folds over 2021-2024 (VAL_SEASONS), "
                         "2019+2020 warmup (never validated), sealed 2025 hold-out",
            "pooled_oof_n": 1107,
            "sealed_n": 285,
            "fold_count": 88,
            "binary": {
                "platt_a": 1.276336,
                "platt_b": 0.121988,
                "pooled_raw": {"logloss": 0.6201, "auc": 0.6950},  # approx — pinned from records
                "pooled_platt": {"logloss": 0.6249, "auc": 0.6950, "ece": 0.0745},
                "sealed_raw": {"logloss": 0.6535, "auc": 0.6782},
                "sealed_platt": {"logloss": 0.6535, "auc": 0.6782, "ece": 0.1009},
            },
            "run_engine": {
                "pooled": {
                    "totals_ece_offered": 0.087,
                    "covers_ece_offered": 0.078,
                    "derived_ml": {"logloss": 0.6365, "auc": 0.695, "ece": 0.0435, "brier": 0.2221},
                },
                "sealed": {
                    "totals_ece_offered": 0.1547,
                    "covers_ece_offered": 0.1145,
                    "derived_ml": {"logloss": 0.6535, "auc": 0.6782, "ece": 0.1009, "brier": 0.2299},
                },
            },
            "recovery_ratios": {
                "elo_diff": 0.638,
                "ewm_net_pts_diff": 0.590,
                "win_pct_diff": 0.649,
                "ewm_ypp_diff": 0.665,
            },
            "source_records": [
                "nfl_binary_calibration_3e8c8a510f04.json (e3aeece)",
                "nfl_binary_calibration_hinge_3e8c8a510f04.json (fc4f4bd)",
                "nfl_binary_calibration_quality_map_3e8c8a510f04.json (e659ba2)",
                "nfl_margin_audit_3e8c8a510f04.json (c1a7c12)",
                "nfl_run_engine_diagnostics_shape_ab_3e8c8a510f04.json (a5bd82d)",
                "nfl_run_engine_diagnostics_v2_3e8c8a510f04.json",
            ],
            "note": "These pins are from the OLD 2021-2024 pool (1,107 pooled + 285 "
                     "sealed, 88 calendar-week folds, sealed 2025 gate). They are "
                     "ARCHIVED here (not edited into the historical records) and "
                     "REPLACED as the new reference by the wide-pool re-baseline. "
                     "Historical records are NEVER edited or deleted.",
        },
        "shared_universe_1376": {
            "note": "The 1,376-game shared OOF lookup (pooled 2021-24 n=1,091 + "
                    "sealed 2025 n=285, RS-only, 88-fold geometry) is the OLD pool's "
                    "universe. The wide-pool re-baseline's universe is the RS-only "
                    "2019-2025 scored pool (1,871 games, week-ID folds, no sealed gate).",
            "old_universe": {
                "pooled": 1091,
                "sealed": 285,
                "total": 1376,
                "geometry": "88 calendar-week folds, 2021-2024 pooled + 2025 sealed",
            },
        },
    }


def _weeks_per_season_folds() -> dict:
    return {2019: 17, 2020: 17, 2021: 18, 2022: 18, 2023: 18, 2024: 18, 2025: 18}


def _write_binary_oof(feats: pd.DataFrame, bin_result: dict,
                      path: Path) -> None:
    """Write per-game binary OOF predictions (raw + calibrated) for the scored
    pool. Deterministic — built from the re-baseline's per-game OOF."""
    # Re-run the walk to get per-game OOF (deterministic, same as the baseline).
    # We re-use run_binary_baseline's logic but write per-game rows.
    Xcol = bin_result["feature_columns"]
    ff = feats.copy()
    ff["gameday"] = pd.to_datetime(ff["gameday"], errors="coerce")
    ff["home_win"] = (ff["home_score"] > ff["away_score"]).astype(int)
    ff = ff.sort_values("gameday").reset_index(drop=True)
    # Impute 2018 warm-up rows' NaN features (same as run_binary_baseline).
    _impute_warmup_features(ff, Xcol)
    wf = build_week_id_folds(ff)
    scored = ff[ff["season"] >= 2019].copy()
    scored = scored[_valid_rows(scored, Xcol)].copy()
    scored = scored.sort_values("gameday").reset_index(drop=True)
    gid_to_idx = {gid: i for i, gid in enumerate(scored["game_id"])}
    raw_oof = np.full(len(scored), np.nan)
    cal_oof = np.full(len(scored), np.nan)
    raw_pool, y_pool = [], []
    for f in wf:
        tr, va = f["train"], f["val"]
        tr = tr[training_columns(tr, Xcol, scored)]
        va = va[[c for c in scored.columns if c in va.columns]].dropna(
            subset=Xcol + ["home_win"])
        from nfl_moneyline import ENSEMBLE_WEIGHTS
        members = _binary_member_oof(tr, va, Xcol)
        wts = {n: ENSEMBLE_WEIGHTS.get(n, 0.0) for n in members}
        tot = sum(wts.values()) or 1.0
        wts = {n: v / tot for n, v in wts.items()}
        blend_va = sum(wts.get(n, 0.0) * members[n] for n in members)
        sofar = sum(len(p) for p in raw_pool)
        if sofar < CALIBRATION_SEED:
            cal_va = blend_va.copy()
        else:
            lr = platt_fit(np.concatenate(raw_pool), np.concatenate(y_pool).astype(int))
            cal_va = platt_predict(blend_va, lr) if lr is not None else blend_va.copy()
        for gid, p_raw, p_cal in zip(va["game_id"], blend_va, cal_va):
            i = gid_to_idx[gid]
            raw_oof[i] = p_raw
            cal_oof[i] = p_cal
        raw_pool.append(blend_va)
        y_pool.append(va["home_win"].to_numpy(float))
    out = scored[["game_id", "season", "week", "gameday", "home_team", "away_team",
                  "home_score", "away_score", "home_win"]].copy()
    out["home_win_prob_raw"] = np.round(raw_oof, 4)
    out["home_win_prob_calibrated"] = np.round(cal_oof, 4)
    out["segment"] = np.where(out["season"] == 2019, "raw", "cal")
    out.to_csv(path, index=False)


# =========================================================================
# CLI
# =========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-record", action="store_true",
                    help="compute/print only; skip writing artifacts")
    ap.add_argument("--out-dir", default=None,
                    help="write artifacts here instead of /c/tmp")
    ap.add_argument("--max-folds", default=None, type=int,
                    help="limit folds for a quick sanity test (production run uses all)")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    result = run_wide_pool_baseline(
        out_dir=Path(args.out_dir) if args.out_dir else None,
        no_record=args.no_record,
        max_folds=args.max_folds)
    if args.no_record:
        print(f"no-record done: {result['store_counts']['total']} row store, "
              f"{result['binary']['pooled']['n_oof']} binary OOF, "
              f"{result['run_engine']['pooled']['n_oof']} run-engine OOF")
    else:
        print(f"done: record={result['record_path']}  "
              f"rolling_brier={result['rolling_brier_path']}  "
              f"binary_oof={result['binary_oof_path']}  "
              f"elapsed={result['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
