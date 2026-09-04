"""NFL run-engine decided-history backfill — decided OOF rows through the
slate emitter's own schema + actuals, one dated markets artifact.

Mirrors MLB's single-store design: ``run_engine_markets_YYYYMMDD.csv``
carries BOTH kinds — decided OOF rows (``kind == "oof"``, predictions AND
actuals) and scheduled rows (``kind == "slate"``) — regenerated each run and
read by the frontend loader as the newest dated file.

The decided OOF store is the market-layer walk's rows:
  pooled 2021-24  n = 1,091   (88-fold weekly geometry, strictly-prior val)
  sealed 2025     n =   285   (fit 2019-24 at median rounds -> predict 2025)
scored through the SLATE EMITTER's schema (``nfl_slate_engine.price_board``:
fair spread/total, full grid, derived-ML pair, raw ±0.5 pair, shrink columns
flagged) plus actuals and honest outcomes/ECE columns.

Chain (all pinned from committed records — NOTHING retrained new):
  era      nfl_era_3e8c8a510f04.json   E2 spec ewm_2w, median rounds 20/23
  joint    (same record)               DN const sigma 9.663/9.0789, rho
                                       0.0076, tie 0.275%
  market   nfl_market_3e8c8a510f04.json totals median-of-fold (c,d) =
                                       (-0.3599, 0.3472); own-line OOF pins
                                       (totals ECE 0.087 / covers 0.078 /
                                       derived-ML 0.6365/0.695/0.0435 pooled;
                                       0.1547 / 0.1145 / 0.6535/0.6782/0.1009
                                       sealed) — the determinism check
  adoption nfl_adoption_decision_3e8c8a510f04.json  spread (c,d) =
                                       (0.446165, 0.307486)

The decided OOF predictions are REGENERATED deterministically (canonical
frame sha 3e8c8a510f04, the SAME ``generate_weekly_folds`` geometry, the
SAME seeded LGB per-side engine, the SAME ``nfl_era_features`` entrypoints
the era runner used — no /tmp dump dependency) and verified against the
records' pinned figures BEFORE anything is emitted. If the regeneration
diverges, the runner stops (never emits with unverified numbers).

Emitted for target_date = run date (America/New_York):
  nfl_run_engine_markets_{date}.csv        board rows (kind == slate) +
                                          decided OOF rows (kind == oof)
  nfl_run_engine_markets_{date}.meta.json  summary incl. the decided-store
                                          calibration figures + provenance
  nfl_run_engine_monitor_{date}.json       research-pinned OOF baseline +
                                          backfill-computed OOF calibration +
                                          EMPTY accumulating slate-history
                                          (no served-slate outcomes exist
                                          yet — nothing fabricated)
  nfl_slate_serve_{date}.json              run gates + mapping table (record)

Retention: dated artifacts are TRACKED-AND-ACCUMULATING (MLB mirror).

Usage:
    cd nfl-backend && python3 backend/run_nfl_markets_backfill.py [--no-record]

Deterministic (no RNG): identical pull -> byte-identical artifacts (g2
double-walk assert).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nfl_market_engine as M  # noqa: E402
import nfl_slate_engine as SE  # noqa: E402
from nfl_era_features import (attach_centers, compute_centers,  # noqa: E402
                              mean_resid_stats, oof_centered_per_side)
from nfl_features import DECIDED_FRAME  # noqa: E402
from nfl_joint_engine import (build_joint_pmfs, cover_prob,  # noqa: E402
                              margin_pmf_from_joint, over_prob,
                              total_pmf_from_joint)
from nfl_moneyline import (DEFAULT_SEASONS, SEALED_SEASON,  # noqa: E402
                           TRAIN_SEASONS, compute_metrics)
from nfl_per_side_engine import SIDE_FEATURES  # noqa: E402
from run_nfl_era import _folds_for, _sealed_era_eval  # noqa: E402
from run_nfl_margin_ablation import load_features  # noqa: E402
from run_nfl_slate import (DATA_DELIVERY, MAPPING_TABLE,  # noqa: E402
                           SLATE_SEASON, _frame_sha,
                           build_board_inputs, price_board_rows)

logger = logging.getLogger(__name__)

CANONICAL_FRAME_SHA = "3e8c8a510f04"

POOLED_N = 1091
SEALED_N = 285

# Record pins the regeneration must reproduce before anything is emitted
# (market record step2_arms own-line figures on the SAME rows).
RECORD_PINS = {
    "pooled": {"totals_ece": 0.087, "covers_ece": 0.078,
               "derived_ml": {"logloss": 0.6365, "auc": 0.695,
                              "ece": 0.0435, "brier": 0.2221}},
    "sealed": {"totals_ece": 0.1547, "covers_ece": 0.1145,
               "derived_ml": {"logloss": 0.6535, "auc": 0.6782,
                              "ece": 0.1009, "brier": 0.2299}},
}
PIN_TOLS = {"ece": 0.001, "ll": 0.0005, "auc": 0.0005, "brier": 0.0005}

# Columns unique to the decided OOF rows (board rows are NaN here —
# undecided by definition, mirror of MLB's target-col exemption).
DECIDED_COLS = ["home_score", "away_score", "total", "margin",
                "p_over_fair", "p_cover_fair",
                "y_over_fair", "y_under_fair", "y_push_fair",
                "y_cover_fair", "y_push_spread_fair",
                "y_over_offered", "y_under_offered", "y_push_total_offered",
                "y_cover_offered", "y_push_spread_offered",
                "y_home_win", "decided", "frame_view"]


def regenerate_era_e2(feats: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Deterministic E2 pooled walk + sealed eval (the era runner's own
    entrypoints, same folds, same seeded LGB). Returns (pooled, sealed,
    rounds) with game_id/pred_home/pred_away/home_score/away_score."""
    decided = pd.read_csv(DECIDED_FRAME)
    decided["gameday"] = pd.to_datetime(decided["gameday"], errors="coerce")
    dv = decided[["game_id", "season", "week", "gameday", "home_score",
                  "away_score", "total"]].copy()
    centers = compute_centers(dv, SE.ERA_SPEC)
    f_chosen = attach_centers(feats, centers)
    folds = _folds_for(f_chosen, SIDE_FEATURES)
    if len(folds) != 88:
        raise RuntimeError(f"E2 fold geometry {len(folds)} != 88 — STOP")
    e2_oof, rounds, _ = oof_centered_per_side(folds, SIDE_FEATURES, f_chosen)
    pooled = e2_oof.merge(f_chosen[["game_id", "season", "home_score",
                                    "away_score"]], on="game_id", how="left")
    sealed = _sealed_era_eval(f_chosen, rounds, SIDE_FEATURES)
    return pooled, sealed, rounds


def _away_abs_resid_2021_23(pooled: pd.DataFrame) -> float:
    """Era record's own definition (run_nfl_era._mean_abs_resid_2021_23):
    mean over 2021-23 of |per-season away mean resid|."""
    sub = pooled[pooled["season"].isin([2021, 2022, 2023])]
    if len(sub) == 0:
        return float("nan")
    vals = []
    for _, g in sub.groupby("season"):
        st = mean_resid_stats(g, "pred_away", "away_score")
        vals.append(abs(st["mean_resid"]))
    return round(float(np.mean(vals)), 4)


def verify_regeneration(pooled: pd.DataFrame, sealed: pd.DataFrame,
                        rounds: dict[str, int],
                        params: dict[str, Any], p_tie: float) -> dict[str, Any]:
    """Machine-verify the regenerated walk against the committed records.
    Raises RuntimeError on any pin breach — the runner never emits with
    unverified numbers."""
    if len(pooled) != POOLED_N or len(sealed) != SEALED_N:
        raise RuntimeError(
            f"regeneration row counts {len(pooled)}/{len(sealed)} != "
            f"{POOLED_N}/{SEALED_N} — STOP")
    if rounds != SE.MEDIAN_ROUNDS:
        raise RuntimeError(f"regenerated rounds {rounds} != "
                           f"record {SE.MEDIAN_ROUNDS} — STOP")
    away_2123 = _away_abs_resid_2021_23(pooled)
    if abs(away_2123 - 0.4083) > 0.01:
        raise RuntimeError(f"away |bias| 21-23 {away_2123:.4f} != record "
                           "0.4083 — regeneration drift; STOP")

    checks: dict[str, Any] = {}
    for name, df in (("pooled", pooled), ("sealed", sealed)):
        pmfs, _s = build_joint_pmfs(
            df[["game_id", "pred_home", "pred_away"]], params, p_tie)
        derived = _s["derived"].copy()
        derived = derived.merge(df[["game_id", "home_score", "away_score"]],
                                on="game_id", how="left")
        ml = compute_metrics(
            (derived["home_score"] > derived["away_score"]).to_numpy(float),
            derived["derived_ml"].to_numpy(float))
        pin = RECORD_PINS[name]["derived_ml"]
        for k, tol in (("logloss", PIN_TOLS["ll"]), ("auc", PIN_TOLS["auc"]),
                       ("ece", PIN_TOLS["ece"]), ("brier", PIN_TOLS["brier"])):
            if abs(ml[k] - pin[k]) > tol:
                raise RuntimeError(
                    f"{name} derived-ML {k} {ml[k]:.4f} != record "
                    f"{pin[k]:.4f} — regeneration drift; STOP")
        checks[name] = {"derived_ml": {k: round(float(ml[k]), 4)
                                       for k in ("logloss", "auc", "ece",
                                                 "brier")}}
    checks["away_abs_resid_2021_23"] = round(away_2123, 4)
    return checks


def build_decided_store(pooled: pd.DataFrame, sealed: pd.DataFrame,
                        params: dict[str, Any], p_tie: float,
                        lines: pd.DataFrame,
                        decided_meta: pd.DataFrame) -> tuple[pd.DataFrame,
                                                            dict[str, Any]]:
    """Price every decided OOF row through the slate emitter's schema and
    attach actuals + honest outcomes. Returns (oof_frame, pins) where pins
    holds the offered-line calibration figures for the record gates."""
    pooled = pooled.copy()
    pooled["frame_view"] = "pooled"
    sealed = sealed.copy()
    sealed["frame_view"] = "sealed"
    allp = pd.concat([pooled, sealed], ignore_index=True)
    # The real-run count guard (1,376) lives in the gates (g1) — this
    # function is pure pricing and stays testable on small frames.

    need = {"game_id", "pred_home", "pred_away", "home_score", "away_score"}
    missing = [c for c in need if c not in allp.columns]
    if missing:
        raise RuntimeError(f"decided store missing {missing}")

    mkt = SE.price_board(allp[["game_id", "pred_home", "pred_away"]],
                         params, p_tie,
                         lines=lines[["game_id", "spread_line",
                                      "total_line"]])
    if len(mkt) != len(allp):
        raise RuntimeError(f"price coverage {len(mkt)}/{len(allp)}")

    # Fair-line probabilities computed directly from the rebuilt PMFs
    # (exact at the fair lines; the grid is integer-anchored and the fair
    # medians always fall inside it, but direct computation is exact).
    pmfs, _s = build_joint_pmfs(
        allp[["game_id", "pred_home", "pred_away"]], params, p_tie)
    margs = [margin_pmf_from_joint(J) for J in pmfs]
    tots = [total_pmf_from_joint(J) for J in pmfs]
    fair_s = mkt["fair_spread"].to_numpy(float)
    fair_t = mkt["fair_total"].to_numpy(float)
    mkt["p_cover_fair"] = np.round(
        [cover_prob(m_, float(L)) for m_, L in zip(margs, fair_s)], 6)
    mkt["p_over_fair"] = np.round(
        [over_prob(t_, float(U)) for t_, U in zip(tots, fair_t)], 6)

    meta = decided_meta[["game_id", "season", "week", "gameday",
                         "home_team", "away_team"]].copy()
    meta["gameday"] = meta["gameday"].astype(str)
    out = mkt.merge(meta, on="game_id", how="left")
    if len(out) != len(mkt) or out["gameday"].isna().any():
        raise RuntimeError("decided identity merge lost rows")

    hs = allp["home_score"].to_numpy(float)
    as_ = allp["away_score"].to_numpy(float)
    total = hs + as_
    margin = hs - as_
    sl = out["spread_line"].to_numpy(float)
    tl = out["total_line"].to_numpy(float)

    def _push(vals: np.ndarray, line: np.ndarray) -> np.ndarray:
        return (vals == line).astype(float)

    out["home_score"] = hs.astype(int)
    out["away_score"] = as_.astype(int)
    out["total"] = total.astype(int)
    out["margin"] = margin.astype(int)
    out["y_over_fair"] = (total > fair_t).astype(float)
    out["y_push_fair"] = _push(total, fair_t)
    out["y_under_fair"] = (total < fair_t).astype(float)
    out["y_cover_fair"] = (margin > fair_s).astype(float)
    out["y_push_spread_fair"] = _push(margin, fair_s)
    out["y_over_offered"] = (total > tl).astype(float)
    out["y_push_total_offered"] = _push(total, tl)
    out["y_under_offered"] = (total < tl).astype(float)
    out["y_cover_offered"] = (margin > sl).astype(float)
    out["y_push_spread_offered"] = _push(margin, sl)
    out["y_home_win"] = (margin > 0).astype(float)
    out["decided"] = 1
    out["frame_view"] = allp["frame_view"].to_numpy()
    # The unrounded per-side mu pair (MLB mirror: home_expected_runs /
    # away_expected_runs) — carried so the g2 double-walk re-prices the
    # store byte-identically and the frontend can show the raw projections.
    out["pred_home"] = allp["pred_home"].to_numpy()
    out["pred_away"] = allp["pred_away"].to_numpy()
    out.insert(0, "kind", "oof")

    pins: dict[str, Any] = {}
    for view in ("pooled", "sealed"):
        sub = out[out["frame_view"] == view]
        t_off = M.totals_calibration(sub, p_col="p_over_offered",
                                     y_col="y_over_offered")
        c_off = M.covers_calibration(sub, p_col="p_cover_offered",
                                     y_col="y_cover_offered")
        t_fair = M.totals_calibration(sub, p_col="p_over_fair",
                                      y_col="y_over_fair")
        c_fair = M.covers_calibration(sub, p_col="p_cover_fair",
                                      y_col="y_cover_fair")
        ml = compute_metrics(sub["y_home_win"].to_numpy(float),
                             sub["p_home_win_derived"].to_numpy(float))
        # ece is None below 20 valid rows (reliability_table's guard) —
        # small-sample views report None honestly; the real run (1,091 /
        # 285) always has figures, and g3 verifies them against the records.
        pins[view] = {
            "n": int(len(sub)),
            "totals_ece_offered": (round(float(t_off["ece"]), 4)
                                    if t_off["ece"] is not None else None),
            "covers_ece_offered": (round(float(c_off["ece"]), 4)
                                    if c_off["ece"] is not None else None),
            "totals_ece_fair": (round(float(t_fair["ece"]), 4)
                                 if t_fair["ece"] is not None else None),
            "covers_ece_fair": (round(float(c_fair["ece"]), 4)
                                 if c_fair["ece"] is not None else None),
            "derived_ml": {k: round(float(ml[k]), 4)
                           for k in ("logloss", "auc", "ece", "brier")},
        }
    return out, pins


def _gates(oof: pd.DataFrame, board_out: pd.DataFrame, pins: dict[str, Any],
           checks: dict[str, Any], frame_sha: str) -> dict[str, Any]:
    g1 = {
        "pass": bool(len(oof) == POOLED_N + SEALED_N
                      and oof["home_score"].notna().all()
                      and oof["away_score"].notna().all()
                      and oof["spread_line"].notna().all()
                      and oof["total_line"].notna().all()),
        "rule": "1,376 decided rows, actuals + 100% offered-line coverage"}
    # g2 determinism: re-price the decided store, byte-identical on the
    # price_board-derived columns (the identity/outcome columns are attached
    # by the store, not by the pricing walk).
    params = SE.pinned_joint_params()
    lines = oof[["game_id", "spread_line", "total_line"]].copy()
    allp = oof[["game_id", "pred_home", "pred_away"]].copy()
    mkt2 = SE.price_board(allp, params, SE.PINNED_P_TIE,
                          lines=lines[["game_id", "spread_line",
                                       "total_line"]])
    common = [c for c in mkt2.columns if c in oof.columns]
    g2 = {"pass": bool(oof[common].to_csv(index=False)
                        == mkt2[common].to_csv(index=False)),
           "rule": "byte-identical double pricing walk on the priced "
                    "columns (no RNG)"}
    # g3 record pins: offered-line calibration reproduces the market record.
    g3_rows = []
    for view, pin in RECORD_PINS.items():
        got = pins[view]
        if abs(got["totals_ece_offered"] - pin["totals_ece"]) > PIN_TOLS["ece"]:
            g3_rows.append(f"{view} totals {got['totals_ece_offered']} vs "
                           f"{pin['totals_ece']}")
        if abs(got["covers_ece_offered"] - pin["covers_ece"]) > PIN_TOLS["ece"]:
            g3_rows.append(f"{view} covers {got['covers_ece_offered']} vs "
                           f"{pin['covers_ece']}")
        for k, tol in (("logloss", PIN_TOLS["ll"]), ("auc", PIN_TOLS["auc"]),
                       ("ece", PIN_TOLS["ece"])):
            if abs(got["derived_ml"][k] - pin["derived_ml"][k]) > tol:
                g3_rows.append(f"{view} derived_ml.{k} "
                               f"{got['derived_ml'][k]} vs {pin['derived_ml'][k]}")
    g3 = {"pass": bool(not g3_rows), "mismatches": g3_rows,
           "rule": "offered-line calibration + derived-ML reproduce the "
                    "market record pins"}
    # g4 leakage: the sealed 2025 frame runs through the season's final game
    # (Super Bowl LX, Feb 8 2026) — the prior-ness guarantee is that the fit
    # windows are strictly prior BY CONSTRUCTION (pooled: per-fold walk
    # assert train < val; sealed: refit on 2019-24 rows only) and the 2026
    # board is strictly later than EVERY decided row.
    oof_days = pd.to_datetime(oof["gameday"], errors="coerce")
    board_days = pd.to_datetime(board_out["gameday"], errors="coerce")
    g4 = {
        "pass": bool(oof["season"].max() <= SEALED_SEASON
                      and oof_days.min() >= pd.Timestamp("2021-01-01")
                      and board_days.min() > oof_days.max()),
        "oof_max_day": str(oof_days.max().date()),
        "oof_min_day": str(oof_days.min().date()),
        "board_min_day": str(board_days.min().date()),
        "rule": "pooled 2021-24 + sealed 2025 rows only; the 2026 board is "
                 "strictly later; fit windows prior by construction"}
    # g5 schema: grid columns populated on every row; only offer-level +
    # decided-target columns may be NaN on board rows; OOF rows carry full
    # offers + actuals (stadium/gametime/records absent for OOF by design).
    grid_cols = ([f"p_home_cover_{SE._fname(float(L))}"
                  for L in SE.SPREAD_INT_LINES]
                 + [f"p_push_{SE._fname(float(L))}"
                    for L in SE.SPREAD_INT_LINES]
                 + [f"p_over_{SE._fname(float(U))}"
                    for U in SE.TOTAL_INT_LINES]
                 + [f"p_under_{SE._fname(float(U))}"
                    for U in SE.TOTAL_INT_LINES])
    # decided/frame_view are FLAGS (0/1), not targets — the board rows must
    # be NaN on the target/outcome columns only.
    target_cols = [c for c in DECIDED_COLS
                   if c not in ("decided", "frame_view")]
    g5 = {
        "pass": bool(not oof[grid_cols].isna().any().any()
                      and not board_out[grid_cols].isna().any().any()
                      and oof[["p_cover_offered", "p_over_offered"]]
                      .notna().all().all()
                      and board_out[target_cols].isna().all().all()),
        "rule": "grid populated everywhere; OOF rows carry offers; board "
                 "rows carry no decided-targets"}
    # g6 served pool untouched: the 12 market-free features + the is_home
    # anchor (13 total), zero market columns.
    import nfl_features as nf  # noqa: PLC0415
    g6 = {"pass": bool(len(nf.FEATURE_COLUMNS) == 13
                        and "is_home" in nf.FEATURE_COLUMNS
                        and "market_home_implied" not in nf.FEATURE_COLUMNS
                        and "spread_line" not in nf.FEATURE_COLUMNS
                        and "total_line" not in nf.FEATURE_COLUMNS),
           "rule": "FEATURE_COLUMNS untouched (12 market-free + is_home "
                    "anchor), market-free"}
    gates = {"g1_actuals_and_coverage": g1, "g2_determinism": g2,
             "g3_record_pins": g3, "g4_leakage": g4, "g5_schema": g5,
             "g6_pool_untouched": g6}
    for k, v in gates.items():
        print(f"  {k}: pass={v['pass']}")
    return gates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-record", action="store_true",
                    help="compute/print only; skip writing artifacts")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    warnings.filterwarnings("ignore",
                            message="The argument 'eval_set' is deprecated")
    t0 = time.time()

    frame_sha = _frame_sha()
    if frame_sha != CANONICAL_FRAME_SHA:
        print(f"FATAL: frame sha {frame_sha} != canonical "
              f"{CANONICAL_FRAME_SHA} — STOP")
        return 1
    print(f"frame_sha256={frame_sha}")

    # =====================================================================
    # STEP 0 — record config pins (never guessed)
    # =====================================================================
    print("\n[Step 0] config pins...")
    params = SE.pinned_joint_params()
    p_tie = SE.PINNED_P_TIE
    print(f"  era spec={SE.ERA_SPEC} rounds={SE.MEDIAN_ROUNDS} | "
          f"sigma_h/a={SE.PINNED_SIGMA_HOME}/{SE.PINNED_SIGMA_AWAY} "
          f"rho={SE.PINNED_RHO} p_tie={p_tie} | "
          f"totals (c,d)={SE.TOTALS_CD} spread (c,d)={SE.SPREAD_CD}")

    # =====================================================================
    # STEP 1 — deterministic E2 regeneration + record verification
    # =====================================================================
    print("\n[Step 1] E2 pooled walk + sealed eval (deterministic)...")
    feats = load_features(None)
    pooled, sealed, rounds = regenerate_era_e2(feats)
    checks = verify_regeneration(pooled, sealed, rounds, params, p_tie)
    print(f"  pooled n={len(pooled)} sealed n={len(sealed)} "
          f"rounds={rounds} away|bias|21-23="
          f"{checks['away_abs_resid_2021_23']}")
    for v in ("pooled", "sealed"):
        print(f"  {v} derived-ML ll/auc/ece/brier = "
              f"{checks[v]['derived_ml']}")

    # =====================================================================
    # STEP 2 — board path (shared with the slate runner) + decided store
    # =====================================================================
    print("\n[Step 2] board + decided store...")
    bi = build_board_inputs()
    decided, dv = bi["decided"], bi["dv"]
    board, lines, impute_rate = bi["board"], bi["lines"], bi["impute_rate"]
    preds, mkt_board = price_board_rows(bi)

    # The DECIDED rows' offered lines come from the historical schedules
    # (2019-2025) — the board's 2026 lines never cover them. Same source the
    # market-layer walk used (100% coverage asserted by the store's gates).
    hist_lines = M.load_offered_lines()
    decided_meta = decided[["game_id", "season", "week", "gameday",
                            "home_team", "away_team"]].copy()
    oof, pins = build_decided_store(pooled, sealed, params, p_tie,
                                    hist_lines, decided_meta)
    for v in ("pooled", "sealed"):
        p = pins[v]
        print(f"  {v} n={p['n']} totals_ECE(offered)={p['totals_ece_offered']} "
              f"covers_ECE(offered)={p['covers_ece_offered']} "
              f"totals_ECE(fair)={p['totals_ece_fair']} "
              f"covers_ECE(fair)={p['covers_ece_fair']} "
              f"derived_ml={p['derived_ml']}")
    print(f"  decided store: {len(oof)} rows | board: {len(board)} rows")

    # =====================================================================
    # STEP 3 — combined out frame (kind slate + kind oof) + gates
    # =====================================================================
    print("\n[Step 3] combined frame + gates...")
    identity = board[["game_id", "gameday", "season", "week", "home_team",
                      "away_team", "stadium", "gametime", "home_record",
                      "away_record"]].copy()
    identity["gameday"] = identity["gameday"].dt.strftime("%Y-%m-%d")
    out_b = mkt_board.merge(identity, on="game_id", how="left")
    if len(out_b) != len(mkt_board):
        raise RuntimeError("board identity merge lost market rows")
    out_b.insert(0, "kind", "slate")
    for c in DECIDED_COLS:
        out_b[c] = np.nan
    out_b["decided"] = 0
    out_b["frame_view"] = np.nan
    # Per-side mu pair on every row (MLB mirror: home_expected_runs /
    # away_expected_runs) — the oof rows carry the unrounded pair so the
    # g2 double-walk re-prices them byte-identically.
    out_b["pred_home"] = np.nan
    out_b["pred_away"] = np.nan

    cols = (["kind", "game_id", "gameday", "season", "week", "home_team",
             "away_team", "stadium", "gametime", "home_record", "away_record"]
            + [c for c in out_b.columns
               if c not in ("kind", "game_id", "gameday", "season", "week",
                            "home_team", "away_team", "stadium", "gametime",
                            "home_record", "away_record")])
    out_b = out_b[cols]
    # cols already carries every DECIDED_COLS (they were added to the board
    # frame above as NaN) — selecting oof by cols gives one unique column
    # set; concat then aligns the kinds cleanly.
    oof = oof[[c for c in cols if c in oof.columns]]
    out = pd.concat([out_b, oof], ignore_index=True, sort=False)
    out = out[cols]

    gates = _gates(oof, out_b, pins, checks, frame_sha)
    ok = all(v["pass"] for v in gates.values())
    if not ok:
        print("FATAL: backfill gates failed — no artifacts emitted")
        return 2

    # =====================================================================
    # STEP 4 — emitters (markets CSV + meta + monitor + serve record)
    # =====================================================================
    print("\n[Step 4] emitters...")
    target = datetime.now(ZoneInfo("America/New_York"))
    date_str = target.strftime("%Y%m%d")
    as_of_utc = target.astimezone(ZoneInfo("UTC")).isoformat()

    oof_baseline = {
        "covers_ece_pooled": 0.078,
        "totals_ece_pooled_own": 0.087,
        "derived_ml": {"logloss": 0.6365, "auc": 0.695, "ece": 0.0435},
        "provenance": [
            "nfl_era_3e8c8a510f04.json (era record: seam covers ECE, G4)",
            "nfl_market_3e8c8a510f04.json (totals ECE own vs shrink)",
            "nfl_adoption_decision_3e8c8a510f04.json (spread adoption)"],
        "note": ("research-pinned pooled-OOF figures from the committed "
                 "records — the backfill-computed store below verifies them "
                 "deterministically on every run"),
    }
    meta = {
        "artifact": f"nfl_run_engine_markets_{date_str}.csv",
        "target_date": date_str,
        "as_of_utc": as_of_utc,
        "frame_sha256": frame_sha,
        "board": {"season": SLATE_SEASON,
                  "n_games": int(len(out_b)),
                  "weeks": [int(w) for w in sorted(
                      out_b["week"].dropna().unique())]},
        "decided_store": {
            "n_pooled": POOLED_N, "n_sealed": SEALED_N,
            "n_total": int(len(oof)),
            "method": ("deterministic E2 regeneration (canonical frame, 88 "
                       "weekly folds, seeded LGB) + slate-emitter schema + "
                       "actuals; verified against the market record pins "
                       "before emission"),
            "calibration": pins,
            "regeneration_checks": checks,
            "line_vintage_caveat": ("nflreadpy schedule-line vintage "
                                    "(closing vs early) UNCONFIRMED — "
                                    "offered-line columns and shrink params "
                                    "carry this caveat; fair-line quotes are "
                                    "model-only"),
        },
        "line_vintage_status": ("nflreadpy schedule-line vintage (closing vs "
                                "early) UNCONFIRMED — shrink columns are "
                                "computed and flagged shrink_applied=false; "
                                "own-line quoting is the served mode"),
        "treatment": {
            "mode": "own-line quoting both sides with honest ECE",
            "shrink_applied": False,
            "one_feed_governs_both_sides": True,
            "shrink_params": {"totals_cd": list(SE.TOTALS_CD),
                              "spread_cd": list(SE.SPREAD_CD)}},
        "impute_rates": impute_rate,
        "oof_baseline": oof_baseline,
        "provenance_records": ["nfl_era_3e8c8a510f04.json",
                               "nfl_market_3e8c8a510f04.json",
                               "nfl_adoption_decision_3e8c8a510f04.json"],
        "engines_modified": False,
        "moneyline_pool_untouched": True,
    }
    monitor = {
        "artifact": f"nfl_run_engine_monitor_{date_str}.json",
        "target_date": date_str,
        "as_of_utc": as_of_utc,
        "oof_baseline_research_pinned": oof_baseline,
        "oof_decided_store_backfill_computed": {
            "n_pooled": POOLED_N, "n_sealed": SEALED_N,
            "method": ("backfill-computed from the decided OOF store in this "
                       "artifact (kind == 'oof' rows): totals/covers ECE at "
                       "the fair lines and at the offered lines, derived-ML "
                       "ll/auc/ece, per view"),
            "calibration": pins,
            "regeneration_checks": checks,
        },
        "slate_history": [],   # accumulating; still empty — no served-slate
                               # outcomes exist yet (honest)
        "markets_persisted": True,
        "markets_path": f"nfl_run_engine_markets_{date_str}.csv",
        "gates": gates,
    }

    if not args.no_record:
        DATA_DELIVERY.mkdir(parents=True, exist_ok=True)
        csv_path = DATA_DELIVERY / f"nfl_run_engine_markets_{date_str}.csv"
        tmp = csv_path.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(csv_path)
        meta_path = DATA_DELIVERY / f"nfl_run_engine_markets_{date_str}.meta.json"
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        tmp.replace(meta_path)
        mon_path = DATA_DELIVERY / f"nfl_run_engine_monitor_{date_str}.json"
        tmp = mon_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(monitor, indent=2, default=str))
        tmp.replace(mon_path)
        rec = {
            "record": "nfl_markets_backfill",
            "target_date": date_str,
            "written_utc": pd.Timestamp.now("UTC").isoformat(),
            "frame_sha256": frame_sha,
            "scope": ("run-engine decided-history connector: decided OOF "
                      "store through the slate emitter's schema + actuals, "
                      "one dated markets artifact (kinds slate + oof), MLB "
                      "mirror. Record + dated artifacts only; NOTHING wired "
                      "into master_pipeline; FEATURE_COLUMNS / served "
                      "12-pool untouched"),
            "config": {"era_spec": SE.ERA_SPEC, "median_rounds": dict(
                SE.MEDIAN_ROUNDS), "pinned_joint": {
                    "sigma_h": SE.PINNED_SIGMA_HOME,
                    "sigma_a": SE.PINNED_SIGMA_AWAY, "rho": SE.PINNED_RHO,
                    "p_tie": SE.PINNED_P_TIE},
                       "market_params": {"totals_cd": list(SE.TOTALS_CD),
                                         "spread_cd": list(SE.SPREAD_CD)},
                       "view": "12-pool per-side PIT (SIDE_FEATURES)"},
            "regeneration_checks": checks,
            "decided_calibration": pins,
            "gates": gates,
            "n_board": int(len(out_b)),
            "n_oof": int(len(oof)),
            "mapping_table": MAPPING_TABLE,
            "retention_decision": ("dated artifacts tracked-and-accumulating "
                                   "(MLB mirror); repo cleanup may never "
                                   "delete a committed file (tracked-file "
                                   "guard)"),
            "judgment_calls": {
                "1_deterministic_regeneration": ("the era E2 pooled walk + "
                                                 "sealed eval are "
                                                 "re-generated with the era "
                                                 "runner's own entrypoints "
                                                 "and pinned frame — no "
                                                 "/tmp dump dependency, "
                                                 "verified against the "
                                                 "market record pins before "
                                                 "emission"),
                "2_one_file_both_kinds": ("MLB mirror: the dated markets "
                                          "CSV carries kind==oof (decided, "
                                          "with actuals) AND kind==slate "
                                          "(2026 board) rows so the newest "
                                          "dated file is the complete "
                                          "store"),
                "3_fair_vs_offered": ("fair-line probabilities computed "
                                      "directly from the rebuilt PMFs; "
                                      "offered-line quotes carry the "
                                      "line-vintage caveat; never "
                                      "conflated"),
                "4_slate_history_still_empty": ("no served-slate outcomes "
                                                "exist yet — the monitor's "
                                                "slate_history stays empty; "
                                                "the backfill-computed OOF "
                                                "calibration is labeled as "
                                                "such, not as slate "
                                                "history"),
            },
        }
        rec_path = DATA_DELIVERY / f"nfl_slate_serve_{date_str}.json"
        tmp = rec_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2, default=str))
        tmp.replace(rec_path)
        print(f"  wrote {csv_path.name} ({len(out_b)} slate + "
              f"{len(oof)} oof rows)")
        print(f"  wrote {meta_path.name}")
        print(f"  wrote {mon_path.name}")
        print(f"  wrote {rec_path.name}")
    else:
        print("  [--no-record] artifacts skipped")
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    raise SystemExit(main())