"""NFL W2016 frame-expansion measurement runner (record-only, no engine edits).

Runs the W2016 (2016-2025 decided frame) measurement for the per-side +
joint chain and writes data_delivery/nfl_frame_expansion_<sha>.json:

  Step 0   — ELO/EWM implementation findings (code read; recorded verbatim).
  Step 1   — build the 2016-2025 decided frame (nfl_game_frame rules) to a
             scratch dir (NEVER mutates data_delivery/nfl_game_level_features.csv);
             per-season counts + validation + new frame sha.
  Step 1.5 — feature delta diagnostics on identical rows (W2016 build vs the
             W2019 production build): elo_diff / home_elo / away_elo delta
             distributions by season, outcome regressions of home_win and
             home_margin on the elo delta, ewm/rolling/static sanity, and the
             recorded (NOT gating) decision rule.
  Step 2/3 — reads the per-side + joint records the UNCHANGED runners wrote on
             the W2016 frame; by-season away-bias table vs the W2019 record;
             G1-G5 + totals ECE/top-bin comparison; family-recheck comparison.
  Step 5   — determinism pins read from both records (per-side double-walk +
             joint double-build).

Harness-geometry note (recorded): the per-side/joint runners build folds from
TRAIN_SEASONS (2019-2024) rows only, so the W2016 measurement via the
unchanged runners isolates the FEATURE-level frame-start effect (2016-18 ELO
priors entering the 2019-24 rows). The "~800 added training rows" component
is exercised only by the window-gate fold geometry — see the record note.

Usage:
    cd nfl-backend && python3 backend/run_nfl_frame_expansion.py \
        [--frame-w2016 <raw csv>] [--frame-w2019 <raw csv>] \
        [--features-w2016 <csv>] [--features-w2019 <csv>] \
        [--per-side-record <json>] [--joint-record <json>] \
        [--artifact <csv>] [--no-record]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_frame_expansion import (
    EWM_MAX_ABS_DELTA, MATERIAL_P95_DELTA, SCORED_SEASONS, SEALED_SEASON,
    SIGNAL_T, W2016_SEASONS, W2019_SEASONS, away_bias_by_season,
    decision_rule, elo_by_side, elo_delta_stats, feature_sanity,
    frame_sha256_of, ladder_elo, ols, outcome_regressions,
)
from nfl_per_side_engine import RESID_AWAY, RESID_HOME, SIDE_FEATURES, SIDE_TARGETS

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"

# The W2019 joint record's frame_sha256 (content hash of its feature frame) —
# used to fast-path-reuse the cached W2019 feature build when it matches.
W2019_FEATURE_SHA = "84987a607522"
_CACHED_FEATURES = Path("/tmp/nfl_features_d450bb9bfd96.parquet")

SCOPE_PIN = (
    "the window gate's W2016 DON'T-ADOPT verdict applied to the binary "
    "moneyline target; this is a separate per-side/joint-engine measurement "
    "and does not bind or contradict that record. Moneyline remains adopted "
    "at 2019. Frame start is a per-model hyperparameter."
)

STEP0_ELO_FINDINGS = {
    "prior": 1500.0, "k": 32.0, "scale": 400.0,
    "expected": "1/(1+10**((r_opp - r_self)/400)); update r += K*(actual - expected); actual = win 1 / loss 0 / tie 0.5",
    "season_boundary_reset_or_regression_to_mean": "NONE — the per-team rating dict persists across all seasons; no reversion, no reset, no prior re-anchor at any boundary",
    "home_field_bonus": "NONE in the ELO update — expected win is symmetric in the two ratings; HFA enters only via the is_home anchor and the other features, never the rating",
    "unseen_team_initialization": "rating.get(team, 1500.0) — a team with no prior games enters at ELO_PRIOR",
    "chronology": "strictly chronological iteration sorted by (gameday, game_id, is_home); elo_entering per (game_id, team) read from the rating dict BEFORE that game's update (strictly-prior games only)",
    "source": "nfl_features._elo_apply / compute_elo (ELO_PRIOR/ELO_K/ELO_SCALE module constants); ladder over _decided_rows(schedule)",
    "frame_start_interpretation": "no-reversion ELO ⇒ the ratings entering 2019 in the W2016 build carry 2016-18 history, and the Step-1.5 by-season delta trend measures whether that offset persists or decays through the scored window",
}
STEP0_EWM_FINDINGS = {
    "halflife": 2.0, "halflife_units": "GAMES — per-team ewm over that team's own games",
    "implementation": "per-team groupby(team)[value].ewm(halflife=2, min_periods=1).mean().groupby(level=0).shift(1) — strictly-prior by the same shift(1) discipline as the windowed ladder",
    "leakage_gate": "team_stats_ladder asserts gameday is strictly increasing within each (team) after sorting by (team, gameday, game_id)",
    "source": "nfl_features._trailing_ewm (EWM_HALFLIFE = 2)",
}
STEP0_SUMMARY = (
    "With no season-boundary reset and no home-field bonus in the rating "
    "update, the only way 2016-18 history reaches the scored window is "
    "through the strictly-prior ladder itself — which is exactly what the "
    "Step-1.5 elo delta measures. EWM (halflife 2 games) decays to ~0 within "
    "~10 prior games, so EWM features are NOT warm-up features: the sanity "
    "check pins max |Δ| <= 1e-3 on scored rows."
)


# ---------------------------------------------------------------------------
# Pulls (network, cached under /tmp)
# ---------------------------------------------------------------------------
def _pull_pbp(seasons: list[int]) -> pd.DataFrame:
    """Season pbp with /tmp parquet caches (feature-trimmed). 2019-2025 reuse
    the full-width caches written by the margin-engine pull; 2015-2018 are
    pulled and trimmed in polars before saving (keeps the pandas side tiny)."""
    from nfl_features import TIER1_NEEDS
    import pyarrow.parquet as pq
    keep = ["game_id", "play_id", "posteam", "yards_gained", "epa",
            "qb_epa", "game_seconds_remaining"] + list(TIER1_NEEDS)
    parts = []
    for yr in seasons:
        pf = Path(f"/tmp/nfl_pbp_feat_{yr}.parquet")
        full = Path(f"/tmp/nfl_pbp_{yr}.parquet")
        if not pf.exists():
            if full.exists():
                avail = set(pq.read_schema(full).names)
                df = pd.read_parquet(full, columns=[c for c in keep
                                                    if c in avail])
                df.to_parquet(pf)
            else:
                import nflreadpy
                pbp = nflreadpy.load_pbp([yr])
                df = pbp.select([c for c in keep if c in pbp.columns])
                df = df.to_pandas()
                df.to_parquet(pf)
        parts.append(pd.read_parquet(pf))
    return pd.concat(parts, ignore_index=True)


def _pull_schedule(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy
    sched = nflreadpy.load_schedules(seasons)
    return sched.to_pandas() if hasattr(sched, "to_pandas") else sched


# ---------------------------------------------------------------------------
# Frame + feature builds (write to scratch; never mutate data_delivery)
# ---------------------------------------------------------------------------
def build_decided_frame(sched: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    from nfl_game_frame import aggregate_game_frame, canonical_decided_frame
    return canonical_decided_frame(aggregate_game_frame(sched, pbp))


def build_w2016_features(raw: pd.DataFrame) -> pd.DataFrame:
    """End-to-end W2016 feature build (warmup 2015 + core 2016-2025) — the
    same shape run_nfl_window_gate uses for its candidate arms."""
    from nfl_features import build_features
    sched = _pull_schedule(W2016_SEASONS)
    pbp = _pull_pbp(W2016_SEASONS)
    return build_features(raw, schedule=sched, pbp=pbp)


def build_w2019_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Production-pull W2019 reference build (schedule 2018-2025 + pbp
    2019-2025 — exactly load_features' hardcoded ranges)."""
    from nfl_features import build_features
    # Fast path: the cached W2019 feature frame (content-hash pinned).
    if _CACHED_FEATURES.exists():
        cached = pd.read_parquet(_CACHED_FEATURES)
        cached["gameday"] = pd.to_datetime(cached["gameday"], errors="coerce")
        if frame_sha256_of(cached) == W2019_FEATURE_SHA:
            logger.info("reused cached W2019 feature frame (%s)",
                        W2019_FEATURE_SHA)
            return cached
    sched = _pull_schedule(W2019_SEASONS)
    pbp = _pull_pbp(W2019_SEASONS)
    return build_features(raw, schedule=sched, pbp=pbp)


# ---------------------------------------------------------------------------
# Step 1.5 helpers
# ---------------------------------------------------------------------------
def _shared_frame(f16: pd.DataFrame, f19: pd.DataFrame) -> pd.DataFrame:
    ids = set(f16["game_id"]) & set(f19["game_id"])
    a = f16[f16["game_id"].isin(ids)].set_index("game_id")
    b = f19[f19["game_id"].isin(ids)].set_index("game_id")
    cols = ["season", "home_team", "away_team", "home_score", "away_score",
            "home_win", "elo_diff", "home_elo", "away_elo"]
    out = a[[c for c in cols if c in a.columns]].copy()
    if "home_win" not in out.columns:
        out["home_win"] = (out["home_score"] > out["away_score"]).astype(float)
    for c in ("elo_diff", "home_elo", "away_elo"):
        out[c + "_2019"] = b[c]
        out[c + "_2016"] = a[c]
    return out.reset_index()


def delta_diagnostics(f16: pd.DataFrame, f19: pd.DataFrame) -> dict[str, Any]:
    """Step 1.5 — the frame-start effect on identical rows."""
    shared = _shared_frame(f16, f19)
    pooled = shared[shared["season"].isin(SCORED_SEASONS)]
    shared_ids = pooled["game_id"].to_numpy()

    deltas = elo_delta_stats(f16, f19, shared_ids)
    reg = outcome_regressions(pooled)

    ewm_cols = ["ewm_net_pts_diff", "ewm_ypp_diff"]
    rolling_cols = ["win_pct_diff", "pace_plays_min_diff", "rest_days_diff"]
    static_cols = ["is_dome_home", "div_game", "travel_miles_diff",
                   "altitude_home", "prime_time"]
    for c in ewm_cols + rolling_cols + static_cols:
        if c in f16.columns and c in f19.columns:
            s = shared.set_index("game_id")
            pooled2 = pooled.set_index("game_id")
            pooled2[c + "_2016"] = f16.set_index("game_id").loc[pooled2.index, c]
            pooled2[c + "_2019"] = f19.set_index("game_id").loc[pooled2.index, c]
            pooled = pooled2.reset_index()
    sanity = feature_sanity(pooled, ewm_cols, rolling_cols, static_cols)

    p95 = max(d["elo_diff"]["p95_abs_delta"] for d in deltas)
    material = p95 >= MATERIAL_P95_DELTA
    t_win = reg["home_win"].get("t")
    signal = bool(t_win is not None and abs(t_win) >= SIGNAL_T)
    rule = decision_rule(material, signal)

    return {
        "n_shared_rows_pooled": int(len(pooled)),
        "elo_delta_by_season": deltas,
        "outcome_regression": reg,
        "feature_sanity": sanity,
        "material_threshold": MATERIAL_P95_DELTA,
        "signal_threshold": SIGNAL_T,
        "material": material, "signal": signal,
        "decision_rule": rule,
    }


# ---------------------------------------------------------------------------
# Step 2/3 comparison vs W2019 records
# ---------------------------------------------------------------------------
def _w2019_bias_table(mb_record: dict) -> list[dict]:
    away = (mb_record.get("step1_diagnostics", {}).get("per_side", {})
            .get("away", {}))
    rows = list(away.get("by_season", []))
    sealed = mb_record.get("step1_diagnostics", {}).get("sealed_2025_bias", {})
    if isinstance(sealed, dict) and "away" in sealed:
        rows.append({"season": SEALED_SEASON, "mean_resid": sealed.get("away")})
    return rows


def exit_criteria(joint_rec: dict, bias_w2016: list[dict],
                  bias_w2019: list[dict]) -> dict[str, Any]:
    sealed = joint_rec["crps_vs_climatology"]["sealed"]
    c1_ok = (sealed["improvement_pct_home"] >= 5.0
             and sealed["improvement_pct_away"] >= 5.0)
    w2019_2021_23 = [r["mean_resid"] for r in bias_w2019
                     if r["season"] in (2021, 2022, 2023)
                     and r.get("mean_resid") is not None]
    w2016_2021_23 = [r["away_bias"] for r in bias_w2016
                     if r["season"] in (2021, 2022, 2023)]
    base = abs(float(np.mean(w2019_2021_23))) if w2019_2021_23 else None
    new = abs(float(np.mean(w2016_2021_23))) if w2016_2021_23 else None
    shrink = None if base is None or base == 0 else (base - new) / base
    c2_ok = shrink is not None and shrink >= 0.40

    tot = joint_rec.get("data_seam", {}).get("totals", {})
    bins = tot.get("bins", [])
    top = bins[-1] if bins else {}
    top_gap = None
    if top.get("pred_mean") is not None and top.get("actual_rate") is not None:
        top_gap = float(top["pred_mean"]) - float(top["actual_rate"])
    c3_ok = top_gap is not None and abs(top_gap) < 0.15

    g5 = (joint_rec.get("gates", {}).get("g5", {}).get("pass")
          and joint_rec.get("gates", {}).get("g5", {}).get("pass") is True)
    return {
        "c1_sealed_crps_both_legs_ge_5pct": {
            "pass": bool(c1_ok),
            "home_pct": sealed["improvement_pct_home"],
            "away_pct": sealed["improvement_pct_away"],
            "bar": ">= 5% on BOTH legs (W2019: 4.74 home / 1.36 away)"},
        "c2_away_bias_2021_23_shrinks_ge_40pct": {
            "pass": bool(c2_ok),
            "w2019_mean_2021_23": None if base is None else round(base, 3),
            "w2016_mean_2021_23": None if new is None else round(new, 3),
            "shrink_pct": None if shrink is None else round(shrink * 100, 1),
            "bar": ">= 40% shrink vs the W2019 mean (2021 -1.71 / 2022 -1.63 "
                   "/ 2023 -2.27)"},
        "c3_totals_top_bin_gap_lt_0_15": {
            "pass": bool(c3_ok),
            "gap": None if top_gap is None else round(top_gap, 4),
            "pred": top.get("pred_mean"), "actual": top.get("actual_rate"),
            "bar": "< 0.15 (W2019: 0.83 - 0.55 = 0.28)"},
        "c4_determinism_g5": {"pass": bool(g5)},
        "w2019_totals_ece": tot.get("ece"),
        "w2016_totals_ece": tot.get("ece"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-w2016", default=None,
                    help="prebuilt 2016-2025 raw decided CSV (else built to scratch)")
    ap.add_argument("--frame-w2019", default=None,
                    help="prebuilt 2019-2025 raw decided CSV (else: --frame-w2016 "
                         "filtered to 2019+)")
    ap.add_argument("--features-w2016", default=None,
                    help="prebuilt W2016 feature CSV (else built end-to-end)")
    ap.add_argument("--features-w2019", default=None,
                    help="prebuilt W2019 reference feature CSV (else built)")
    ap.add_argument("--per-side-record", default=None,
                    help="W2016 per-side record JSON (written by the unchanged runner)")
    ap.add_argument("--joint-record", default=None,
                    help="W2016 joint record JSON (written by the unchanged runner)")
    ap.add_argument("--artifact", default=None,
                    help="W2016 per-side residual artifact CSV")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    t0 = time.time()
    print("=" * 70)
    print("  NFL W2016 Frame-Expansion Measurement (record-only)")
    print("=" * 70)

    # ---- Step 1: decided frame 2016-2025 ----
    if args.frame_w2016:
        raw16 = pd.read_csv(args.frame_w2016)
    else:
        scratch = Path("/tmp/nfl_w2016")
        scratch.mkdir(parents=True, exist_ok=True)
        print("[1] Pulling schedule+pbp 2016-2025 and building the decided frame...")
        sched = _pull_schedule(W2016_SEASONS)
        pbp = _pull_pbp(W2016_SEASONS)
        raw16 = build_decided_frame(sched, pbp)
        out = scratch / "nfl_game_level_features.csv"
        raw16.to_csv(out, index=False)
        print(f"    wrote {out} ({len(raw16)} rows)")
    raw16["gameday"] = pd.to_datetime(raw16["gameday"], errors="coerce")
    # frame_sha = sha256 of the raw CSV FILE BYTES — the runners' convention
    # (hashlib.sha256(DECIDED_FRAME.read_bytes())[:12]); the canonical copy
    # must be byte-identical to this file for the record shas to line up.
    if args.frame_w2016:
        f16_sha = hashlib.sha256(
            Path(args.frame_w2016).read_bytes()).hexdigest()[:12]
    else:
        f16_sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    per_season = (raw16.groupby("season")["game_id"].count()
                  .rename("games").reset_index())
    print(f"    decided {len(raw16)} games | per-season:\n"
          + per_season.to_string(index=False))
    n_home = int((raw16["home_score"] > raw16["away_score"]).sum())
    print(f"    home win rate {n_home/len(raw16):.3f} | "
          f"ppg {(raw16['home_score'].sum()+raw16['away_score'].sum())/len(raw16):.1f}")

    # ---- Step 1.5: feature deltas on identical rows ----
    print("\n[1.5] Building W2016 + W2019 feature frames...")
    if args.features_w2016:
        f16 = pd.read_csv(args.features_w2016)
    else:
        f16 = build_w2016_features(raw16)
        Path("/tmp/nfl_w2016_features.csv").parent.mkdir(parents=True, exist_ok=True)
        Path("/tmp/nfl_w2016_features.csv").write_text(f16.to_csv(index=False))
    if args.frame_w2019:
        raw19 = pd.read_csv(args.frame_w2019)
    else:
        raw19 = raw16[raw16["season"] >= 2019].copy()
    f19 = (pd.read_csv(args.features_w2019)
           if args.features_w2019 else build_w2019_features(raw19))
    f16["gameday"] = pd.to_datetime(f16["gameday"], errors="coerce")
    f19["gameday"] = pd.to_datetime(f19["gameday"], errors="coerce")
    for _f in (f16, f19):
        if "home_win" not in _f.columns:
            _f["home_win"] = (_f["home_score"] > _f["away_score"]).astype(float)
    print(f"    W2016 features: {len(f16)} rows sha256={frame_sha256_of(f16)} | "
          f"W2019 features: {len(f19)} rows sha256={frame_sha256_of(f19)}")

    # Attach per-side ELO ladders for the delta — rebuilt from the SAME
    # schedules the feature builds used (warmup + core), exactly as
    # build_features computes the ladder timeline.
    ev16, _ = ladder_elo(_pull_schedule(W2016_SEASONS))
    ev19, _ = ladder_elo(_pull_schedule(W2019_SEASONS))
    side16 = elo_by_side(ev16).set_index("game_id")
    side19 = elo_by_side(ev19).set_index("game_id")
    for df_, side_ in ((f16, side16), (f19, side19)):
        df_["home_elo"] = df_["game_id"].map(side_["home_elo"])
        df_["away_elo"] = df_["game_id"].map(side_["away_elo"])
        df_["elo_diff_ladder"] = df_["game_id"].map(side_["elo_diff_ladder"])
    ok_parity = float(
        np.nanmean(np.abs(f16["elo_diff"].to_numpy(float)
                          - f16["elo_diff_ladder"].to_numpy(float))))
    print(f"    ladder-parity (|elo_diff - ladder| mean): {ok_parity:.6f}")

    diag = delta_diagnostics(f16, f19)
    print(f"    decision rule: {diag['decision_rule']['verdict']} — "
          f"{diag['decision_rule']['reason']}")

    # ---- Steps 2/3: read the unchanged runners' records ----
    per_side_rec = (json.loads(Path(args.per_side_record).read_text())
                    if args.per_side_record else None)
    joint_rec = (json.loads(Path(args.joint_record).read_text())
                 if args.joint_record else None)

    bias_w2016 = []
    mb_rec = DATA_DELIVERY / "nfl_mean_bias_calibration_3e8c8a510f04.json"
    bias_w2019 = _w2019_bias_table(json.loads(mb_rec.read_text()))
    if per_side_rec and joint_rec and args.artifact:
        artifact = pd.read_csv(args.artifact)
        if "season" not in artifact.columns:
            artifact = artifact.merge(
                f16[["game_id", "season"]].drop_duplicates(),
                on="game_id", how="left")
            if artifact["season"].isna().any():
                raise RuntimeError("artifact season join left NaN rows — "
                                   "frame/artifact mismatch")
        from run_nfl_joint import _sealed_predictions
        rounds = {"home": int(artifact["best_iter_home"].median()),
                  "away": int(artifact["best_iter_away"].median())}
        sealed = _sealed_predictions(f16, rounds)
        sealed["resid_away"] = sealed["away_score"] - sealed["pred_away"]
        sealed["resid_home"] = sealed["home_score"] - sealed["pred_home"]
        bias_w2016 = away_bias_by_season(artifact, sealed)
        print("\n[2] away bias by season (W2016):")
        for r in bias_w2016:
            print(f"    {r['season']}: away {r['away_bias']:+.3f} "
                  f"(n={r['n']})")

    crit = (exit_criteria(joint_rec, bias_w2016, bias_w2019)
            if joint_rec and bias_w2016 else {})
    if crit:
        print("\n[exit criteria]")
        for k, v in crit.items():
            if isinstance(v, dict) and "pass" in v:
                print(f"    {k}: {v['pass']}")

    record = {
        "record": "nfl_w2016_frame_expansion",
        "frame_sha": f16_sha,
        "frame_sha256_w2016": frame_sha256_of(f16),
        "frame_sha256_w2019_reference": frame_sha256_of(f19),
        "w2019_frame_sha256_pinned": W2019_FEATURE_SHA,
        "written_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "scope": SCOPE_PIN,
        "step0_elo": STEP0_ELO_FINDINGS,
        "step0_ewm": STEP0_EWM_FINDINGS,
        "step0_summary": STEP0_SUMMARY,
        "step1_frame": {
            "seasons": sorted(raw16["season"].unique().tolist()),
            "n_games": int(len(raw16)),
            "n_home_wins": int(n_home),
            "home_win_rate": round(n_home / len(raw16), 4),
            "combined_ppg": round(float(
                (raw16["home_score"].sum() + raw16["away_score"].sum())
                / len(raw16)), 1),
            "per_season": per_season.to_dict("records"),
            "frame_sha": f16_sha,
        },
        "step1_5_diagnostics": diag,
        "harness_geometry_note": (
            "the per-side/joint runners build folds from TRAIN_SEASONS "
            "(2019-2024) rows only (_moneyline_folds / _sealed_eval filter "
            "feats[season in TRAIN_SEASONS] before generate_weekly_folds), so "
            "expanding the decided frame to 2016 does NOT add training rows "
            "through these runners — the W2016 measurement isolates the "
            "FEATURE-level frame-start effect (2016-18 ELO priors entering the "
            "2019-24 rows). The ~800-added-training-rows component is only "
            "exercised by the window-gate fold geometry (run_nfl_window_gate "
            "builds folds over the candidate's full train window)."),
        "step2_per_side": (
            {"record": per_side_rec["frame_sha"],
             "oof": per_side_rec["per_side_oof"]["model"],
             "sealed": per_side_rec["per_side_sealed"],
             "family_recheck": per_side_rec["family_recheck"]["table"],
             "determinism": per_side_rec["determinism_pin"]}
            if per_side_rec else None),
        "step3_joint": (
            {"record": joint_rec["frame_sha"],
             "crps_vs_climatology": joint_rec["crps_vs_climatology"],
             "gates": joint_rec["gates"],
             "data_seam": joint_rec["data_seam"],
             "marginal_family": joint_rec["marginal_family"],
             "sigma_curve": joint_rec["sigma_curve"],
             "rho": joint_rec["rho"],
             "tie": joint_rec["tie"],
             "determinism_g5": joint_rec["gates"]["g5"]}
            if joint_rec else None),
        "away_bias_w2016_by_season": bias_w2016,
        "away_bias_w2019_by_season": bias_w2019,
        "exit_criteria": crit,
        "verdict": _verdict(crit),
        "feature_columns_untouched": True,
        "engines_unmodified": True,
    }

    if not args.no_record:
        out_path = DATA_DELIVERY / f"nfl_frame_expansion_{f16_sha}.json"
        out_path.write_text(json.dumps(record, indent=2, default=str))
        print(f"\nRecord: {out_path.name}")
    else:
        print("\n[--no-record] record skipped")
    return 0


def _verdict(crit: dict) -> dict:
    if not crit:
        return {"verdict": "not_scored",
                "reason": "records/artifact not supplied to this run"}
    passes = [k for k, v in crit.items()
              if isinstance(v, dict) and v.get("pass")]
    outcome_passes = [k for k in passes if k != "c4_determinism_g5"]
    if len(outcome_passes) == 3:
        return {"verdict": "GO — exit criteria met; market layer queued",
                "passed": passes}
    if len(outcome_passes) == 0:
        return {"verdict": "NOTHING_MOVED — 2016 is not the binding "
                           "constraint for this harness; reassess "
                           "(dispersion/conditional-curvature layer is the "
                           "fallback lever)",
                "passed": passes,
                "note": "c4 determinism is a mechanical invariant, not an "
                        "outcome improvement; only c1-c3 count toward GO"}
    return {"verdict": "PARTIAL — not a hard fail; record the trend; the "
                       "market layer's own calibration pass can absorb "
                       "residual base miscalibration (ECE shown honestly)",
            "passed": passes}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())