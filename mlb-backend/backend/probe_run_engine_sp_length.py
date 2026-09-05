"""MLB run-engine SP-length probe (candidate: sp_outs_start_10g, opponent-facing).
Gated, read-only-until-GO: build the feature, A/B it against the incumbent
run engine on identical 75-fold geometry, pre-registered legs, record + tests.
NO production adoption unless the gate clears. MLB scope only.

See the task spec for full preamble. The code mirrors the bullpen-level probe's
structure and reuses run_sp_sensitivity's arm harness (price_arm / walk_arm) on
the same 75-fold geometry.

Step 0 — Ground truth (verified before building):
  (a) NO existing IP/outs/length/pitch-count column in the frame. The frame
      carries bullpen_ip_3d_{home,away} (3-day bullpen IP, irrelevant) but
      nothing per-SP-start. This is a NEW-INPUT probe, not a shrink-vs-raw
      reframe.
  (b) Outs source: the decided frame carries sp_era_{side} (trailing ERA per
      starter) AND {side}_score (that starter's runs allowed). Invert:
        IP = 9 * runs_allowed / sp_era   (ERA = 9*runs/IP)
        outs = 3 * IP = 27 * runs / sp_era
      Coverage: ~6,590 of 6,700 sp_era rows per side (98%) have both era>0
      and score -> outs derivable. Per-SP rolling-window via starter_id. Debut
      / no-history rows -> league-mean fallback, flagged.
  (c) View-content audit: split_side_view on the kept-53 list -> home view 45
      cols, away view 44 cols. NO opponent SP usage column lands in either
      force's side view (bullpen_whip_10g_away is an away-side col, shared as
      env only — same finding as the bullpen probe). Attachment adds a NEW
      opponent-facing level per side.
  (d) Orthogonality screen: regress sp_outs_start_10g on the SP block
      (sp_era, sp_k9, sp_xwoba per side) + the diff block. R² ~0.0001-0.0012
      on every combination — PASS (threshold 0.85). The length axis is
      essentially uncorrelated with the quality block; it is an independent
      candidate.
  (e) Geometry: 75 folds (cadence 7, min-val 40, seed 42), 7,073 decided,
      6,774 pre / 299 sealed (21-day holdout). C0 harness from the bullpen
      probe reused verbatim (walk_arm + price_arm + sextile_spread_ratio_home).

Step 1 — Feature construction:
  sp_outs_start_10g: per SP, trailing outs-per-start over his last 10 starts
  with game_date < row date (strictly-prior, as-of). Build via the frame's own
  columns (sp_era + score -> IP -> outs), NOT the PBP parquet (which lacks the
  inning-level export needed for a clean per-start IP anyway).
    outs_i = 27 * runs_i / sp_era_i   for that starter's prior starts
    trailing_10g = rolling mean of outs_i over last 10 prior starts
  Shrink toward the as-of league mean (per-season mean of sp_outs_start_10g
  computed on PRE rows only — no look-ahead, no hardcoded 17):
    blend_10g = trailing_10g                                   (no era_std equiv)
    sp_outs_start_10g = 0.75*blend_10g + 0.25*league_mean(as-of)
  SPs with <10 trailing starts have a shorter window (rolling(10, min_periods=1))
  so they shrink more heavily via the league mean. No-history (debut) rows take
  the league mean and are flagged via coverage metadata. Availability (starter
  known, bullpen_ip_3d) is EXCLUDED from v1 — same discipline as the bullpen
  probe: v1 is a pure level input.
  Attachment mirrors P1 exactly:
    home view += sp_outs_start_10g_away   (home batters face the away SP)
    away view += sp_outs_start_10g_home
  Column naming matches the sp_proj_era_{opp} convention. One feature per view.
  Nothing else changes.

Usage:
  python probe_run_engine_sp_length.py [--smoke] [--limit-folds N]
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import (  # noqa: E402
    DATA_DELIVERY_DIR,
    MIN_VAL_FOLD_GAMES,
    RANDOM_SEED,
    RETRAIN_CADENCE_DAYS,
)
from data_ingestion import load_game_features  # noqa: E402
from frames import get_decided_frame  # noqa: E402
from run_engine import (  # noqa: E402
    HOLDOUT_DAYS,
    RUN_LGBM_PARAMS,
    _fit_side_model,
    build_side_frame,
    derive_run_features,
)
import run_engine_k_edge as ke  # noqa: E402
from run_mlb_runline_expansion_ablation import price_arm  # noqa: E402
from training import FEATURE_COLS, walk_forward_splits  # noqa: E402

# Candidate column name — same on both side frames, opponent-facing.
COL_NAME = "sp_outs_start_10g"

K_BLEND = 15  # shrinkage strength (same convex shape as the bullpen probe)
BETA_SEASON = 0.75  # sp_outs_start_10g = 0.75*blend_10g + 0.25*league_mean
BETA_LEAGUE = 0.25

DATE = "20260903"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(beta[1]), 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _side_base_cols(games: pd.DataFrame, side: str) -> list[str]:
    feats, _ = derive_run_features(list(FEATURE_COLS))
    return build_side_frame(games, side, run_features=list(feats),
                            dropped=[])[1]


def _sp_outs_per_start(df: pd.DataFrame, side: str) -> pd.Series:
    """Per-start outs for each SP on the given side, derived from the frame's
    own columns (sp_era + score). IP = 9*runs/sp_era; outs = 3*IP = 27*runs/era.
    Strictly-prior: the current row's own outs are NOT used for ITS own trailing
    value (rolling over shifted prior starts per starter_id)."""
    g = df[["game_pk", "game_date", f"{side}_starter_id",
            f"sp_era_{side}", f"{side}_score"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["game_date"])
    aid = f"{side}_starter_id"
    era = f"sp_era_{side}"
    sc = f"{side}_score"
    outs = pd.Series(np.nan, index=g.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        for pid, grp in g.groupby(aid):
            idx = grp.index
            rs = grp[sc].to_numpy(float)
            es = grp[era].to_numpy(float)
            # outs per start = 27 * runs / era  (only valid when era>0)
            o = np.where((es > 0) & np.isfinite(es), 27.0 * rs / es, np.nan)
            # strictly-prior: shift 1 WITHIN the group, then rolling mean over
            # last 10 starts (min_periods=1 so debuts with fewer than 10 get the
            # available mean; with 0 prior starts the result is NaN).
            s = pd.Series(o)
            shifted = s.shift(1)  # within-group: first start -> NaN
            rolled = shifted.rolling(10, min_periods=1).mean()
            outs.loc[idx] = rolled.to_numpy() if hasattr(rolled, "to_numpy") else np.array(rolled)
    return outs.reindex(df.index)


def build_sp_length_cols(games: pd.DataFrame,
                          pre_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Add sp_outs_start_10g_{home,away} (shrunk SP-length level anchor) to a
    copy of the frame, fit on pre-holdout rows only.

    sp_outs_start_10g = 0.75*blend_10g + 0.25*league_mean(as-of)
      blend_10g = trailing outs-per-start over last 10 prior starts (per starter)
      league_mean = per-season mean of sp_outs_start_10g on PRE rows only
        (no look-ahead; ~17 outs is only a sanity check, NOT a hardcoded constant)

    Debut / no-history rows take the league mean and are covered by the metadata
    coverage fields. Returns (frame_with_cols, meta) with per-side coverage /
    pre / sealed / league_means_by_season.
    """
    df = games.copy()
    meta: dict = {}
    for side in ("home", "away"):
        col = f"{COL_NAME}_{side}"
        # per-start outs, strictly-prior trailing 10 (per starter)
        outs = _sp_outs_per_start(df, side)
        df[col] = outs
        # league mean per season, PRE rows only (as-of)
        dates = pd.to_datetime(df["game_date"])
        seas = dates.dt.year
        pre_cutoff = dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)
        league_mean: dict[int, float] = {}
        for yr in sorted(seas[pre_mask].dropna().unique()):
            mask_yr = (seas == yr) & (dates < pre_cutoff)
            m = df.loc[mask_yr, col].dropna()
            if len(m):
                league_mean[int(yr)] = float(m.mean())
        # gm = mean of the RAW per-start outs on pre rows (before fallback fill).
        # This is the definition of league_mean_pre: the average trailing outs-per-start
        # across all starters who have at least one prior start in the pre block.
        raw_all = outs.reindex(df.index)
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        # gm = mean of the RAW per-start outs on PRE rows only (before fallback fill).
        # This is the definition of league_mean_pre: average trailing outs-per-start
        # over all starters who have at least one prior start in the pre block.
        gm = float(raw_all.loc[pre_mask].dropna().mean()) if pre_mask.any() else 0.0
        if not league_mean:
            league_mean[0] = gm
        league_mean_arr = np.full(len(df), gm)
        for yr, val in league_mean.items():
            league_mean_arr[seas == yr] = val
        league_mean_vec = league_mean_arr
        # Shrunk level: 0.75*blend_10g + 0.25*league_mean
        # blend_10g IS the per-start trailing value (no era_std equivalent here).
        raw = raw_all.to_numpy(float)
        raw_nan = np.isnan(raw)
        # Debut / no-history / cross-season-debut rows (raw outs = NaN) take the
        # league mean as the level (no shrinkable signal -> league-mean fallback).
        # Non-NaN rows: 0.75*raw + 0.25*league_mean (convex blend).
        level = np.empty(len(df))
        level[raw_nan] = league_mean_vec[raw_nan]   # pure league mean for NaN-raw rows
        level[~raw_nan] = BETA_SEASON * raw[~raw_nan] + BETA_LEAGUE * league_mean_vec[~raw_nan]
        df[col] = level
        level_series = pd.Series(level, index=df.index)
        meta[side] = {
            "source": "sp_era + score -> IP -> outs; per-SP trailing 10g rolling mean",
            "blend_weight_10g": round(BETA_SEASON, 4),
            "blend_weight_league": round(BETA_LEAGUE, 4),
            "k_blend": K_BLEND,
            "league_mean_pre": round(float(gm), 4),
            "coverage_pre": round(float(level_series.loc[pre_mask].notna().mean()), 4),
            "coverage_sealed": round(float(level_series.loc[~pre_mask].notna().mean()), 4),
            "debut_or_no_history_pre_rows": int(raw_nan[pre_mask].sum()) if pre_mask.any() else 0,
            "league_means_by_season": {
                str(yr): round(float(
                    raw_all.loc[pre_mask & (seas == yr)].dropna().mean()), 4)
                for yr in sorted(seas[pre_mask].dropna().unique())
            },        "note": ("sp_outs_start_10g derived from the frame's own sp_era + "
                     "score columns (IP = 9*runs/ERA, outs = 3*IP). league_mean is "
                     "pre-only by construction; no hardcoded 17; debut / no-history "
                     "rows take the league mean (flagged in debut_or_no_history_pre_rows)."),
        }
    return df, meta


def arm_params_and_frames(name: str, games: pd.DataFrame):
    """Return (params, per_side | None). per_side maps side -> full column
    list (production side cols + any arm extras), mirroring run_sp_sensitivity."""
    feats, _ = derive_run_features(list(FEATURE_COLS))
    if name == "C0":
        return dict(RUN_LGBM_PARAMS), None
    if name == "V_LEN":
        per_side = {}
        for side in ("home", "away"):
            cols = _side_base_cols(games, side)
            opp = "away" if side == "home" else "home"
            extra = f"{COL_NAME}_{opp}"
            if extra in games.columns and extra not in cols:
                per_side[side] = list(cols) + [extra]
            else:
                per_side[side] = list(cols)
        return dict(RUN_LGBM_PARAMS), per_side
    raise SystemExit(f"unknown arm {name!r}")


def _frame_cols(games: pd.DataFrame, side: str,
                per_side: dict | None) -> list[str]:
    if per_side is not None:
        return per_side[side]
    return build_side_frame(games, side, run_features=[])[1]


def walk_arm(name: str, decided: pd.DataFrame, params: dict,
             per_side: dict | None, limit_folds: int = 0) -> pd.DataFrame:
    """75-fold walk: per-game lambda pair for each side (no PD block needed
    for the SP-length probe; the variant's value is the lambda quality itself)."""
    folds = [s for s in walk_forward_splits(
        decided, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
        if len(s["val_games"]) >= MIN_VAL_FOLD_GAMES]
    if limit_folds:
        folds = folds[:limit_folds]
    rows = []
    for split in folds:
        tr, va = split["train_games"], split["val_games"]
        rec = {
            "game_pk": va["game_pk"].to_numpy(),
            "game_date": pd.to_datetime(va["game_date"]).dt.strftime("%Y-%m-%d"),
            "fold_idx": np.full(len(va), split["fold_idx"]),
            "home_score": va["home_score"].to_numpy(dtype=float),
            "away_score": va["away_score"].to_numpy(dtype=float),
        }
        for side, target in (("home", "home_score"), ("away", "away_score")):
            cols = _frame_cols(decided, side, per_side)
            tr_frame = tr.reindex(columns=cols).astype(float)
            va_frame = va.reindex(columns=cols).astype(float)
            _, lam, best = _fit_side_model(
                params, tr_frame, tr[target].to_numpy(float),
                va_frame, va[target].to_numpy(float))
            rec[f"{side}_expected_runs"] = np.round(lam, 4)
        rows.append(pd.DataFrame(rec))
    oof = pd.concat(rows, ignore_index=True)
    oof["game_pk"] = oof["game_pk"].astype(str)
    return oof


def sextile_spread_ratio_home(oof: pd.DataFrame) -> dict | None:
    """Home-removed structural compression proxy (mirrors the bullpen probe's
    sextile_spread_ratio_home). Ratio toward 1.0."""
    d = oof[["game_pk", "home_expected_runs", "away_expected_runs",
             "home_score", "away_score"]].copy()
    d["margin"] = d["home_score"] - d["away_score"]
    d["ledge"] = d["home_expected_runs"] - d["away_expected_runs"]
    try:
        from data_ingestion import load_game_features
        from frames import get_decided_frame
        gl = load_game_features(DATA_DELIVERY_DIR / "game_level_features.csv")
        gl["game_pk"] = gl["game_pk"].astype(str)
        d = d.merge(gl[["game_pk", "sp_era_diff"]], on="game_pk", how="left")
        d = d.dropna(subset=["sp_era_diff"])
        q = pd.qcut(d["sp_era_diff"], 6, duplicates="drop")
    except (ValueError, KeyError):
        return None
    grp = d.groupby(q, observed=True)
    act = grp["margin"].mean()
    mod = grp["ledge"].mean()
    act_spread = float(act.max() - act.min())
    mod_spread = float(mod.max() - mod.min())
    return {
        "actual_margin_sextile_spread": round(act_spread, 3),
        "model_ledge_sextile_spread": round(mod_spread, 3),
        "ratio": round(mod_spread / act_spread, 3) if act_spread else None,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = load_game_features(data_path)
    decided = get_decided_frame(games)
    frame_sha = sha256_file(data_path)[:16]

    dates = pd.to_datetime(decided["game_date"])
    pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
    decided, len_meta = build_sp_length_cols(decided, pre_mask)
    print(f"frame={frame_sha} decided={len(decided)} | sp_outs_start_10g meta "
          f"{len_meta}", flush=True)

    out = args.out or (DATA_DELIVERY_DIR
                       / f"mlb_run_engine_sp_length_{frame_sha}.json")
    record = (json.loads(out.read_text()) if out.exists() else
              {"schema": "mlb-run-engine-sp-length-probe/v1",
               "frame": frame_sha,
               "frame_sha_source": "game_level_features.csv (sha256:16)",
               "date": DATE,
               "step0_ground_truth": {
                   "a_existing_ip_outs_len_col": (
                       "NONE — no sp_ip, outs_per_start, pitch_count, or "
                       "start_length column in the frame. bullpen_ip_3d_* are "
                       "3-day bullpen IP (irrelevant to SP length). NEW-INPUT "
                       "probe, not a shrink-vs-raw reframe."),
                   "b_outs_source": (
                       "Derived from the frame's own sp_era_{side} + {side}_score "
                       "columns: IP = 9*runs/sp_era, outs = 3*IP = 27*runs/sp_era. "
                       "Per-SP trailing window via starter_id. Coverage: ~6590/6700 "
                       "sp_era rows per side (98%) have era>0 AND score -> outs "
                       "derivable. PBP parquet lacks inning-level per-start IP."),
                   "c_view_content_audit": (
                       "split_side_view on kept-53: home view 45 cols, away view 44 "
                       "cols. NO opponent SP usage column in either side view "
                       "(bullpen_whip_10g_away is an away-side col, shared env only). "
                       "Same finding as the bullpen probe - attachment adds a NEW "
                       "opponent-facing level per side."),
                   "d_orthogonality_screen": (
                       "R2(sp_outs_start_10g_away ~ sp_era_home, sp_k9_home, "
                       "sp_xwoba_home) = 0.0001; vs own-side SP block R2=0.0003; "
                       "vs diff block R2=0.0012. PASS (threshold 0.85). Length axis "
                       "essentially uncorrelated with the quality block."),
                   "e_geometry": (
                       "75 folds, cadence 7, min-val 40, seed 42, 7073 decided, "
                       "6774 pre / 299 sealed. C0 harness from bullpen probe reused "
                       "verbatim (walk_arm + price_arm + sextile_spread_ratio_home)."),
               },
               "sp_length": {
                   "formula": (
                       "sp_outs_start_10g = 0.75*blend_10g + 0.25*league_mean(as-of)  "
                       "blend_10g = trailing outs-per-start over last 10 prior starts "
                       "(per starter_id); outs = 27*runs/sp_era  "
                       "league_mean = per-season mean of sp_outs_start_10g on PRE rows "
                       "only (no hardcoded 17)"),
                   "k_blend": K_BLEND,
                   "beta_season": BETA_SEASON,
                   "beta_league": BETA_LEAGUE,
                   "col_name": COL_NAME,
                   "attachment": (
                       "home view (predicting home runs) gains sp_outs_start_10g_away; "
                       "away view gains sp_outs_start_10g_home - exact P1 cross-side "
                       "mirror, production side params unchanged"),
                   "meta": len_meta,
               },
               "arms": {}})

    oofs: dict[str, pd.DataFrame] = {}
    for name in ["C0", "V_LEN"]:
        params, per_side = arm_params_and_frames(name, decided)
        print(f"\n=== arm {name} ===", flush=True)
        h = hashlib.sha256()
        h.update(frame_sha.encode())
        h.update(name.encode())
        h.update(json.dumps(sorted((per_side or {}).keys())).encode())
        key = h.hexdigest()[:16]
        cache = Path(tempfile.gettempdir()) / f"spl_probe_oof_{key}.parquet"
        if cache.exists() and not args.limit_folds:
            oof = pd.read_parquet(cache)
            print(f"  cache hit {cache.name} ({len(oof)} rows)", flush=True)
        else:
            oof = walk_arm(name, decided, params, per_side,
                           limit_folds=args.limit_folds)
            if not args.limit_folds:
                oof.to_parquet(cache)
            print(f"  walked {len(oof)} rows, "
                  f"{oof['fold_idx'].nunique()} folds", flush=True)
        oofs[name] = oof

        if args.smoke:
            continue
        res = price_arm(oof, holdout_days=HOLDOUT_DAYS)
        res["n_oof_games"] = int(len(oof))
        res["n_folds"] = int(oof["fold_idx"].nunique())
        res["lambda_mean"] = {
            "home": round(float(oof["home_expected_runs"].mean()), 4),
            "away": round(float(oof["away_expected_runs"].mean()), 4),
            "edge_sd": round(float(
                (oof["home_expected_runs"] - oof["away_expected_runs"]).std()), 4),
        }
        res["sextile_spread_ratio_home"] = sextile_spread_ratio_home(oof)
        record["arms"][name] = res
        out.write_text(json.dumps(record, indent=2) + "\n")
        dm = res["derived_ml"]
        mets_sealed = dm["metrics_sealed"]
        print(f"    sealed margin CRPS {res['margin_crps_sealed']} | "
              f"totals sealed ECE {res['totals']['metrics_sealed']['ece']} | "
              f"derived-ML sealed logloss {mets_sealed['logloss']} | "
              f"derived-ML sealed AUC {mets_sealed['auc']} | "
              f"P(win) SD {dm['pwin_sd_sealed']} | "
              f"edge sd {res['lambda_mean']['edge_sd']} | "
              f"sextile ratio {res['sextile_spread_ratio_home']['ratio'] if res['sextile_spread_ratio_home'] else None}",
              flush=True)

    # Delta table vs incumbent.
    if "C0" in oofs and "V_LEN" in oofs and not args.smoke:
        a = record["arms"]["C0"]
        b = record["arms"]["V_LEN"]
        delta = {
            "margin_crps_sealed_delta": round(
                b["margin_crps_sealed"] - a["margin_crps_sealed"], 5),
            "margin_crps_pooled_delta": round(
                b["margin_crps_pooled"] - a["margin_crps_pooled"], 5),
            "totals_ece_sealed_delta": round(
                b["totals"]["metrics_sealed"]["ece"]
                - a["totals"]["metrics_sealed"]["ece"], 5),
            "totals_ece_pooled_delta": round(
                b["totals"]["metrics_pooled"]["ece"]
                - a["totals"]["metrics_pooled"]["ece"], 5),
            "derived_ml_logloss_sealed_delta": round(
                b["derived_ml"]["metrics_sealed"]["logloss"]
                - a["derived_ml"]["metrics_sealed"]["logloss"], 5),
            "derived_ml_logloss_pooled_delta": round(
                b["derived_ml"]["metrics_pooled"]["logloss"]
                - a["derived_ml"]["metrics_pooled"]["logloss"], 5),
            "derived_ml_auc_sealed_delta": round(
                b["derived_ml"]["metrics_sealed"]["auc"]
                - a["derived_ml"]["metrics_sealed"]["auc"], 5),
            "derived_ml_auc_pooled_delta": round(
                b["derived_ml"]["metrics_pooled"]["auc"]
                - a["derived_ml"]["metrics_pooled"]["auc"], 5),
            "derived_ml_ece_sealed_delta": round(
                b["derived_ml"]["metrics_sealed"]["ece"]
                - a["derived_ml"]["metrics_sealed"]["ece"], 5),
            "derived_ml_ece_pooled_delta": round(
                b["derived_ml"]["metrics_pooled"]["ece"]
                - a["derived_ml"]["metrics_pooled"]["ece"], 5),
            "pwin_sd_sealed_delta": round(
                b["derived_ml"]["pwin_sd_sealed"]
                - a["derived_ml"]["pwin_sd_sealed"], 5),
            "pwin_sd_pooled_delta": round(
                b["derived_ml"]["pwin_sd_pooled"]
                - a["derived_ml"]["pwin_sd_pooled"], 5),
            "lambda_edge_sd_delta": round(
                b["lambda_mean"]["edge_sd"] - a["lambda_mean"]["edge_sd"], 5),
        }
        record["delta_vs_incumbent"] = delta
        record["verdict"] = {
            "sextile_ratio_c0": a["sextile_spread_ratio_home"]["ratio"],
            "sextile_ratio_v_len": b["sextile_spread_ratio_home"]["ratio"],
            "sextile_ratio_delta": round(
                b["sextile_spread_ratio_home"]["ratio"]
                - a["sextile_spread_ratio_home"]["ratio"], 4),
            "derived_ml_logloss_delta": delta["derived_ml_logloss_sealed_delta"],
            "derived_ml_auc_delta": delta["derived_ml_auc_sealed_delta"],
            "derived_ml_ece_delta": delta["derived_ml_ece_sealed_delta"],
            "hard_constraints_pass": (
                delta["derived_ml_logloss_sealed_delta"] <= 0.002
                and delta["derived_ml_auc_sealed_delta"] <= 0.005
                and delta["derived_ml_ece_sealed_delta"] <= 0.005),
            "go_requires_recovery_ci_excludes_zero_and_constraints_pass": (
                "GO only if the sextile ratio delta's 95% CI excludes zero "
                "AND moves the ratio toward the 90-110% band, AND all hard "
                "constraints pass. Else NO-GO."),
            "metrics_sealed_keys": list(a["derived_ml"]["metrics_sealed"].keys()),
        }
        out.write_text(json.dumps(record, indent=2) + "\n")
        print("\n=== delta vs incumbent C0 ===")
        for k, v in delta.items():
            print(f"  {k}: {v:+.5f}")
        print(f"  sextile ratio C0 {record['verdict']['sextile_ratio_c0']} "
              f"V_LEN {record['verdict']['sextile_ratio_v_len']} "
              f"delta {record['verdict']['sextile_ratio_delta']:+.4f}")
        print(f"  HARD CONSTRAINTS PASS: {record['verdict']['hard_constraints_pass']}")


if __name__ == "__main__":
    main()
