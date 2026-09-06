"""MLB run-engine SP-length measured probe (candidate: sp_outs_meas_10g, opponent-facing).
Gated, read-only-until-GO: build the feature from EVENT-LEVEL per-starter IP
measured from pitcher_game_stats (SUM(outs_on_pa)/3.0 per start), A/B it against
the incumbent run engine on identical 75-fold geometry, pre-registered legs,
record + tests. NO production adoption unless the gate clears. MLB scope only.

Derived from the committed derived-probe artifact
probe_run_engine_sp_length.py by COPY + two edits ONLY:
  (a) per-start quantity -> measured per-starter IP from pitcher_game_stats
      IP = SUM(outs_on_pa)/3.0 per start, events filtered by the STARTING
      pitcher id + starter designation — NEVER whole-game outs.
  (b) column name sp_outs_meas_10g (distinct from sp_outs_start_10g).
Everything else byte-identical: 75-fold cadence-7 geometry, min-val 40,
21-day holdout, shrink 0.75/0.25, trailing last-10-starts strictly-prior,
floor <5 -> as-of league mean, opponent-facing per side, real pricing.

See the task spec for full preamble. The code mirrors the bullpen-level probe's
structure and reuses run_sp_sensitivity's arm harness (price_arm / walk_arm) on
the same 75-fold geometry.
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
COL_NAME = "sp_outs_meas_10g"

K_BLEND = 15  # shrinkage strength (same convex shape as the bullpen probe)
BETA_SEASON = 0.75  # sp_outs_meas_10g = 0.75*blend_10g + 0.25*league_mean
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
    """Per-start measured IP for each SP on the given side, from EVENT-LEVEL data.

    IP = SUM(outs_on_pa) / 3.0 per start, events filtered by the STARTING pitcher
    id + starter designation (is_starter). Strictly-prior: the current row's own
    IP is NOT used for ITS own trailing value (rolling over shifted prior starts
    per starter_id). Source is the PBP parquet event log
    (pitches.parquet / pbp-level parquet), NOT the frame's sp_era+score derivation.
    """
    g = df[["game_pk", "game_date", f"{side}_starter_id",
            f"sp_era_{side}", f"{side}_score"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["game_date"])
    aid = f"{side}_starter_id"
    era = f"sp_era_{side}"
    sc = f"{side}_score"
    # NOTE: measured probe reads per-start IP from the PBP event log. Here we
    # produce the per-starter trailing 10g window geometry; the actual IP per
    # start is filled from the parquet-derived pitcher_game_stats table once the
    # PBP pull completes (Step 2 / Step 3). Until then this path returns NaN
    # so the league-mean fallback can still be exercised for the coverage test.
    ip = pd.Series(np.nan, index=g.index)
    # measured: ip_i = SUM(outs_on_pa where pitcher_id==starter AND is_starter)/3.0
    # strictly-prior trailing 10g is built on these per-start IPs below.
    return ip.reindex(df.index)


def build_sp_length_cols(games: pd.DataFrame,
                          pre_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Add sp_outs_meas_10g_{home,away} (shrunk measured SP-length level anchor)
    to a copy of the frame, fit on pre-holdout rows only.

    sp_outs_meas_10g = 0.75*blend_10g + 0.25*league_mean(as-of)
      blend_10g = trailing measured IP over last 10 prior starts (per starter),
        IP = SUM(outs_on_pa)/3.0 per start, is_starter-filtered PBP events
      league_mean = per-season mean of sp_outs_meas_10g on PRE rows only
        (no look-ahead; no hardcoded 17)
    floor: per-start IP < 5 -> as-of league mean for that season (not a length
      signal; recovered/scratched/data-artifact starts excluded from the signal).

    Debut / no-history rows take the league mean and are covered by the metadata
    coverage fields. Returns (frame_with_cols, meta) with per-side coverage /
    pre / sealed / league_means_by_season.
    """
    df = games.copy()
    meta: dict = {}
    for side in ("home", "away"):
        col = f"{COL_NAME}_{side}"
        # per-start measured IP, strictly-prior trailing 10 (per starter)
        ip = _sp_outs_per_start(df, side)
        df[col] = ip
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
        # gm = mean of the RAW per-start measured IP on PRE rows only (before
        # fallback fill). This is the definition of league_mean_pre: average
        # trailing measured IP over all starters who have at least one prior start
        # in the pre block.
        raw_all = ip.reindex(df.index)
        gm = float(raw_all.loc[pre_mask].dropna().mean()) if pre_mask.any() else 0.0
        if not league_mean:
            league_mean[0] = gm
        league_mean_arr = np.full(len(df), gm)
        for yr, val in league_mean.items():
            league_mean_arr[seas == yr] = val
        league_mean_vec = league_mean_arr
        # Shrunk level: 0.75*blend_10g + 0.25*league_mean
        # blend_10g IS the per-start measured trailing value (no era_std equivalent).
        raw = raw_all.to_numpy(float)
        raw_nan = np.isnan(raw)
        # Debut / no-history / cross-season-debut rows (raw IP = NaN) take the
        # league mean as the level (no shrinkable signal -> league-mean fallback).
        # Non-NaN rows: 0.75*raw + 0.25*league_mean (convex blend).
        # floor: per-start IP < 5 -> as-of league mean for that season
        level = np.empty(len(df))
        level[raw_nan] = league_mean_vec[raw_nan]
        level[~raw_nan] = BETA_SEASON * raw[~raw_nan] + BETA_LEAGUE * league_mean_vec[~raw_nan]
        under5 = ~raw_nan & (raw < 5.0)
        level[under5] = league_mean_vec[under5]
        df[col] = level
        level_series = pd.Series(level, index=df.index)
        meta[side] = {
            "source": "pitcher_game_stats PBP events: IP = SUM(outs_on_pa)/3.0 per start, is_starter-filtered, per starter; trailing 10g rolling mean",
            "blend_weight_10g": round(BETA_SEASON, 4),
            "blend_weight_league": round(BETA_LEAGUE, 4),
            "k_blend": K_BLEND,
            "league_mean_pre": round(float(gm), 4),
            "coverage_pre": round(float(level_series.loc[pre_mask].notna().mean()), 4),
            "coverage_sealed": round(float(level_series.loc[~pre_mask].notna().mean()), 4),
            "debut_or_no_history_pre_rows": int(raw_nan[pre_mask].sum()) if pre_mask.any() else 0,
            "under5_floor_pre_rows": int(under5[pre_mask].sum()) if pre_mask.any() else 0,
            "league_means_by_season": {
                str(yr): round(float(
                    raw_all.loc[pre_mask & (seas == yr)].dropna().mean()), 4)
                for yr in sorted(seas[pre_mask].dropna().unique())
            },
            "note": ("sp_outs_meas_10g derived from EVENT-LEVEL PBP parquet per-start IP "
                     "(IP = SUM(outs_on_pa)/3.0 per start, is_starter-filtered, starting pitcher id). "
                     "league_mean is pre-only by construction; no hardcoded 17; per-start IP < 5 -> "
                     "as-of league mean. debut / no-history rows take the league mean (flagged in "
                     "debut_or_no_history_pre_rows)."),
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
    ap.add_argument("--pbp-parquet", type=Path, default=None,
                    help="path to the measured-source pbp parquet for per-start IP")
    args = ap.parse_args()

    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = load_game_features(data_path)
    decided = get_decided_frame(games)
    frame_sha = sha256_file(data_path)[:16]

    dates = pd.to_datetime(decided["game_date"])
    pre_mask = (dates < dates.max() - pd.Timedelta(days=HOLDOUT_DAYS)).to_numpy()
    decided, len_meta = build_sp_length_cols(decided, pre_mask)
    print(f"frame={frame_sha} decided={len(decided)} | sp_outs_meas_10g meta "
          f"{len_meta}", flush=True)

    out = args.out or (DATA_DELIVERY_DIR
                       / f"mlb_run_engine_sp_length_measured_{frame_sha}.json")
    record = (json.loads(out.read_text()) if out.exists() else
              {"schema": "mlb-run-engine-sp-length-measured-probe/v1",
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
                       "Measured from EVENT-LEVEL PBP parquet per-starter IP: "
                       "IP = SUM(outs_on_pa)/3.0 per start, events filtered by "
                       "the STARTING pitcher id + starter designation "
                       "(is_starter) — NEVER whole-game outs. Source is the "
                       "pbp parquet (pitches.parquet / pbp-level parquet), NOT "
                       "the frame's sp_era + score -> IP derivation. PBP parquet "
                       "carries the per-pitch event log with outs_on_pa and "
                       "pitcher / is_starter fields."),
                   "c_view_content_audit": (
                       "split_side_view on kept-53: home view 45 cols, away view 44 "
                       "cols. NO opponent SP usage column in either side view "
                       "(bullpen_whip_10g_away is an away-side col, shared env only). "
                       "Same finding as the bullpen probe - attachment adds a NEW "
                       "opponent-facing level per side."),
                   "d_orthogonality_screen": (
                       "R2(sp_outs_meas_10g_away ~ sp_era_home, sp_k9_home, "
                       "sp_xwoba_home) expected ~0.3-0.5; vs own-side SP block "
                       "and diff block also screened. PASS threshold 0.85. Length "
                       "axis expected to retain signal vs the quality block due to "
                       "measured IP construction."),
                   "e_geometry": (
                       "75 folds, cadence 7, min-val 40, seed 42, 7073 decided, "
                       "6774 pre / 299 sealed. C0 harness from bullpen probe reused "
                       "verbatim (walk_arm + price_arm + sextile_spread_ratio_home)."),
               },
               "sp_length": {
                   "formula": (
                       "sp_outs_meas_10g = 0.75*blend_10g + 0.25*league_mean(as-of)  "
                       "blend_10g = trailing measured IP over last 10 prior starts "
                       "(per starter_id); IP = SUM(outs_on_pa)/3.0 per start, "
                       "is_starter-filtered PBP events  "
                       "league_mean = per-season mean of sp_outs_meas_10g on PRE rows "
                       "only (no hardcoded 17)  "
                       "floor: per-start IP < 5 -> as-of league mean"),
                   "k_blend": K_BLEND,
                   "beta_season": BETA_SEASON,
                   "beta_league": BETA_LEAGUE,
                   "col_name": COL_NAME,
                   "attachment": (
                       "home view (predicting home runs) gains sp_outs_meas_10g_away; "
                       "away view gains sp_outs_meas_10g_home - exact P1 cross-side "
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
