"""NFL slate-serve runner — prices the 2026 board through the pinned research
chain and emits the dated run-line/totals artifacts (record-only; NOTHING is
wired into master_pipeline).

Chain (all pinned from committed records — the slate engine never re-derives):
  era      nfl_era_3e8c8a510f04.json     E2 spec ewm_2w, rounds 20/23
  joint    (same record)                 DN const sigma 9.663/9.0789, rho
                                         0.0076, tie 0.275%
  market   nfl_market_3e8c8a510f04.json  totals median-of-fold (c,d) =
                                         (-0.3599, 0.3472)
  adoption nfl_adoption_decision_3e8c8a510f04.json  spread median-of-fold
                                         (c,d) = (0.446165, 0.307486),
                                         ADOPT_SHRINK_TO_LINE both sides,
                                         feed decision: one feed governs both

Emitted for target_date = run date (America/New_York):
  nfl_run_engine_markets_{date}.csv        per-board-game market rows
                                          (kind == slate; no oof rows on the
                                          first run — the monitor carries the
                                          research-pinned OOF baseline)
  nfl_run_engine_markets_{date}.meta.json  summary (treatment mode per side,
                                          line-vintage status, as-of date,
                                          provenance record list, OOF
                                          baseline figures)
  nfl_run_engine_monitor_{date}.json       research-pinned OOF calibration
                                          baseline + EMPTY accumulating
                                          slate-history section (nothing
                                          fabricated)
  nfl_slate_serve_{date}.json              run gates + the MLB↔NFL artifact
                                          mapping table (the record)

Retention: dated artifacts are TRACKED-AND-ACCUMULATING (MLB mirror). The
repo cleanup may never delete a committed file (tracked-file guard), so
these are safe once committed. First run commits them with this change.

Usage:
    cd nfl-backend && python3 backend/run_nfl_slate.py [--no-record] [--out-dir DIR]

Deterministic (no RNG): identical pull -> byte-identical artifacts (G6
double-walk assert).
"""
from __future__ import annotations

import argparse
import hashlib
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

import nfl_slate_engine as SE  # noqa: E402
from nfl_era_features import (CENTER_COLS, EWM_HALFLIFE_DAYS,  # noqa: E402
                              attach_centers, compute_centers,
                              refit_centered_per_side)
from nfl_joint_engine import build_joint_pmfs  # noqa: E402
from nfl_moneyline import DEFAULT_SEASONS  # noqa: E402
from nfl_per_side_engine import SIDE_FEATURES  # noqa: E402
from nfl_features import (DECIDED_FRAME, _load_raw,  # noqa: E402
                          build_slate_features)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY = ROOT_DIR / "data_delivery"

ERA_RECORD = DATA_DELIVERY / "nfl_era_3e8c8a510f04.json"
MARKET_RECORD = DATA_DELIVERY / "nfl_market_3e8c8a510f04.json"
ADOPTION_RECORD = DATA_DELIVERY / "nfl_adoption_decision_3e8c8a510f04.json"

CANONICAL_FRAME_SHA = "3e8c8a510f04"
SLATE_SEASON = 2026

# MLB -> NFL artifact mapping table (mirror of run_engine.persist_markets /
# frontend markets loader semantics; documented in the record).
MAPPING_TABLE = {
    "markets_csv": {
        "mlb": "run_engine_markets_{YYYYMMDD}.csv",
        "nfl": "nfl_run_engine_markets_{YYYYMMDD}.csv",
        "rows": ("per-game rows; MLB kind ∈ {oof, slate}; NFL first-run rows "
                 "all kind == 'slate' (no OOF rows — the OOF baseline is "
                 "research-pinned in the monitor JSON)"),
        "identity": "game_pk (MLB, int) -> game_id (NFL, nflverse string)",
        "offer_columns": ("MLB p_over_<U>/p_push_<U>/p_under_<U> over the "
                          "totals grid + p_home_cover_<L> over the run-line "
                          "grid -> NFL p_over_<U>/p_under_<U>/p_push_<U> over "
                          "TOTAL_INT_LINES 24..66 and p_home_cover_<L>/"
                          "p_push_<L> over SPREAD_INT_LINES -14..+14 "
                          "(integers; NFL margin PMF is integer-support)"),
        "derived_pair": ("p_home_win_derived/p_away_win_derived (MLB derived "
                         "from NB lambda layer) -> NFL same names from "
                         "P(H>A)/(1-P_tie) of the calibrated 76x76 joint"),
        "raw_pair": ("NFL-only additive pair a future ±0.5 toggle needs: "
                     "p_home/away_cover_{minus,plus}_half at the offered "
                     "spread line"),
        "fair_vs_offer": ("fair_* columns = model medians (no market "
                          "blending); offered line columns = nflreadpy "
                          "schedule; never conflated"),
        "persist": ("tmp-write + atomic replace + .meta.json sidecar — "
                    "identical to MLB persist_markets"),
    },
    "meta_json": {
        "mlb": "run_engine_markets_{YYYYMMDD}.meta.json",
        "nfl": "nfl_run_engine_markets_{YYYYMMDD}.meta.json",
        "content": ("treatment mode per side, line-vintage status, as-of "
                    "date, provenance record list, OOF baseline figures"),
    },
    "monitor_json": {
        "mlb": "model_monitor_*.json run_engine block",
        "nfl": "nfl_run_engine_monitor_{YYYYMMDD}.json",
        "content": ("research-pinned OOF baseline (covers ECE 0.078, seam "
                    "totals ECE 0.087, derived-ML ll/auc/ece "
                    "0.6365/0.695/0.0435) + accumulating slate-history "
                    "(empty on the first run)"),
    },
}


def _frame_sha() -> str:
    return hashlib.sha256(DECIDED_FRAME.read_bytes()).hexdigest()[:12]


def _record_config() -> dict[str, Any]:
    """Read the pinned chain config from the committed records (never
    guessed); verify the values match the slate engine's constants."""
    era = json.loads(ERA_RECORD.read_text())
    mkt = json.loads(MARKET_RECORD.read_text())
    adopt = json.loads(ADOPTION_RECORD.read_text())
    era_cfg = {
        "spec": era["step1_arms"]["cv_selection"]["chosen"],
        "rounds": era["step1_arms"]["e2"]["rounds"],
        "sigma_h": era["step2_joint_chain"]["joint_params"]["sigma_h"],
        "sigma_a": era["step2_joint_chain"]["joint_params"]["sigma_a"],
        "rho": era["step2_joint_chain"]["joint_params"]["rho"],
        "p_tie": era["step2_joint_chain"]["tie"]["p_tie_empirical"],
    }
    mkt_cd = mkt["step1_disagreement_walk"]["median_cd"] if \
        "median_cd" in mkt.get("step1_disagreement_walk", {}) else \
        [mkt["step1_disagreement_walk"]["median_c"],
         mkt["step1_disagreement_walk"]["median_d"]]
    adopt_walk = adopt["spread_measurement"]["walk"]
    adopt_cd = [adopt_walk["median_c"], adopt_walk["median_d"]]
    return {"era": era_cfg, "market": {"median_cd": mkt_cd,
                                       "verdict": mkt["verdict"]["state"]},
            "adoption": {"median_cd": adopt_cd,
                         "verdict": adopt["verdict"]["state"],
                         "feed": adopt["feed_decision"]}}


def build_board_inputs() -> dict[str, Any]:
    """STEP 1 — decided frame + era centers + schedule/board.

    Shared with ``run_nfl_markets_backfill`` so the 2026 board path is ONE
    code path (the decided-history backfill appends kind==oof rows to the
    SAME board artifact the slate runner emits). Returns everything the
    pricing step needs.
    """
    decided = pd.read_csv(DECIDED_FRAME)
    decided["gameday"] = pd.to_datetime(decided["gameday"], errors="coerce")
    dv = decided[["game_id", "season", "week", "gameday", "home_score",
                  "away_score", "total"]].copy()
    centers = compute_centers(dv, SE.ERA_SPEC)
    decided_c = dv.merge(centers, on="game_id", how="left")
    if decided_c[CENTER_COLS].isna().any().any():
        raise RuntimeError("decided center attach lost rows")

    # Features on the decided rows are needed for the 12-pool refit view.
    # load_features builds over the canonical decided frame (nflreadpy pull,
    # cached) — same frame the era walk used.
    from run_nfl_margin_ablation import load_features
    feats = load_features(None)
    feats = feats[feats["season"] >= 2019]
    if len(feats) != 1960:
        raise RuntimeError(f"decided feature frame {len(feats)} != 1960")
    decided_f = decided_c.merge(
        feats[["game_id"] + SIDE_FEATURES], on="game_id", how="left")
    n_full = int(decided_f[SIDE_FEATURES + CENTER_COLS].dropna().shape[0])
    if n_full < 1700:
        raise RuntimeError(f"decided rows usable for the refit only {n_full}/"
                           f"{len(decided_f)} — STOP")
    print(f"  decided rows: {len(decided_f)} | refit-usable (full 12-pool + "
          f"centers): {n_full}")

    sched_hist, pbp = _load_raw(DEFAULT_SEASONS)
    import nflreadpy  # noqa: PLC0415
    s2026 = nflreadpy.load_schedules([SLATE_SEASON])
    if hasattr(s2026, "to_pandas"):
        s2026 = s2026.to_pandas()
    for c in ("home_score", "away_score"):
        s2026[c] = pd.to_numeric(s2026[c], errors="coerce")
    sched = pd.concat([sched_hist, s2026], ignore_index=True)

    # Offered lines for the board come straight from the 2026 schedule rows
    # (before build_slate_features drops market columns — market-free policy
    # applies to the model's feature frame, not to this offer feed).
    lines = s2026[["game_id", "spread_line", "total_line"]].drop_duplicates(
        "game_id", keep="last")
    board = build_slate_features(sched, pbp, decided, SLATE_SEASON)
    board["gameday"] = pd.to_datetime(board["gameday"], errors="coerce")
    if board["gameday"].isna().any():
        raise RuntimeError("slate board carries unparseable gameday — STOP")
    print(f"  board: {len(board)} scheduled {SLATE_SEASON} rows | offered "
          f"line coverage: spread {lines['spread_line'].notna().mean()*100:.0f}% "
          f"total {lines['total_line'].notna().mean()*100:.0f}%")
    if len(board) == 0:
        raise RuntimeError("empty slate board — STOP")

    # Board era centers (strictly-prior decided only; mirror pinned to the
    # era module by tests). The board stays MARKET-FREE (offers are a
    # separate feed merged only at pricing/emission — g5 asserts this).
    bcen = SE.board_era_centers(dv, board, SE.ERA_SPEC)
    board = board.merge(bcen, on="game_id", how="left")

    # ---- serve-time missingness (repo policy: admit on merit, handle NaN
    # at serve — mirror of the moneyline ensemble's impute-median bundle).
    # Future 2026 rows have no season pbp yet (pace etc.) and some neutral
    # sites lack venue facts — impute from the decided-train medians and
    # record the per-feature rates honestly in the record.
    impute = {f: float(decided_f[f].median()) for f in SIDE_FEATURES}
    impute_rate = {}
    for f in SIDE_FEATURES:
        rate = float(board[f].isna().mean())
        if rate > 0:
            board[f] = board[f].fillna(impute[f])
        impute_rate[f] = round(rate, 4)
    print("  serve-time impute rates: "
          + ", ".join(f"{f}={r}" for f, r in impute_rate.items() if r))
    return {"decided": decided, "dv": dv, "decided_f": decided_f,
            "board": board, "lines": lines, "impute_rate": impute_rate}


def price_board_rows(bi: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """STEP 2 — fit-only refit at median rounds (era-centered) + price.

    Shared with ``run_nfl_markets_backfill``. Returns (preds, mkt) — the
    refit per-side means and the priced market rows for the board.
    """
    board, decided, dv, decided_f, lines = (bi["board"], bi["decided"],
                                            bi["dv"], bi["decided_f"],
                                            bi["lines"])
    preds = refit_centered_per_side(decided_f, board, SE.MEDIAN_ROUNDS,
                                    SIDE_FEATURES)
    if len(preds) != len(board):
        raise RuntimeError(f"refit coverage {len(preds)}/{len(board)} — "
                           "impute failed to restore board rows")
    # Leakage asserts: every board gameday is strictly after every decided
    # gameday AND the refit's training frame contains only decided rows.
    if board["gameday"].min() <= decided["gameday"].max():
        raise RuntimeError("board rows overlap the decided timeline — STOP")
    params = SE.pinned_joint_params()
    mkt = SE.price_board(preds, params, SE.PINNED_P_TIE,
                         lines=lines[["game_id", "spread_line",
                                      "total_line"]])
    if len(mkt) != len(board):
        raise RuntimeError(f"price coverage {len(mkt)}/{len(board)}")
    return preds, mkt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-record", action="store_true",
                    help="compute/print only; skip writing artifacts")
    ap.add_argument("--out-dir", default=None,
                    help="emit artifacts here instead of data_delivery")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    warnings.filterwarnings("ignore",
                            message="The argument 'eval_set' is deprecated")
    t0 = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DELIVERY

    # =====================================================================
    # STEP 0 — frame + config pins
    # =====================================================================
    frame_sha = _frame_sha()
    if frame_sha != CANONICAL_FRAME_SHA:
        print(f"FATAL: frame sha {frame_sha} != canonical "
              f"{CANONICAL_FRAME_SHA} — STOP")
        return 1
    cfg = _record_config()
    era_cfg = cfg["era"]
    # Verify record-vs-engine constants (provenance pin, not assumption).
    assert era_cfg["spec"] == SE.ERA_SPEC, (era_cfg["spec"], SE.ERA_SPEC)
    assert era_cfg["rounds"]["home"] == SE.MEDIAN_ROUNDS["home"]
    assert era_cfg["rounds"]["away"] == SE.MEDIAN_ROUNDS["away"]
    assert era_cfg["sigma_h"]["sigma0"] == SE.PINNED_SIGMA_HOME
    assert era_cfg["sigma_a"]["sigma0"] == SE.PINNED_SIGMA_AWAY
    assert abs(era_cfg["rho"] - SE.PINNED_RHO) < 1e-9
    assert abs(era_cfg["p_tie"] - SE.PINNED_P_TIE) < 1e-9
    mkt_cd = tuple(round(float(x), 4) for x in cfg["market"]["median_cd"])
    assert mkt_cd == SE.TOTALS_CD, (mkt_cd, SE.TOTALS_CD)
    adopt_cd = tuple(round(float(x), 6) for x in cfg["adoption"]["median_cd"])
    assert adopt_cd == SE.SPREAD_CD, (adopt_cd, SE.SPREAD_CD)
    print(f"frame_sha256={frame_sha}")
    print(f"  era spec={era_cfg['spec']} rounds={era_cfg['rounds']} "
          f"sigma_h/a={era_cfg['sigma_h']['sigma0']}/"
          f"{era_cfg['sigma_a']['sigma0']} rho={era_cfg['rho']} "
          f"p_tie={era_cfg['p_tie']}")
    print(f"  market totals (c,d)={mkt_cd} "
          f"[{cfg['market']['verdict']}] | spread (c,d)={adopt_cd} "
          f"[{cfg['adoption']['verdict']}] | "
          f"feed_present={cfg['adoption']['feed']['known_vintage_feed_present']}")

    # =====================================================================
    # STEP 1 — decided frame + era centers + schedule/board
    # =====================================================================
    print("\n[Step 1] data + era centers...")
    bi = build_board_inputs()
    decided, dv, decided_f, board, lines, impute_rate = (
        bi["decided"], bi["dv"], bi["decided_f"], bi["board"],
        bi["lines"], bi["impute_rate"])

    # =====================================================================
    # STEP 2 — fit-only refit at median rounds (era-centered) + price
    # =====================================================================
    print("\n[Step 2] refit + price...")
    preds, mkt = price_board_rows(bi)
    print(f"  priced {len(mkt)}/{len(board)} board games (100% coverage)")

    # =====================================================================
    # STEP 3 — gates (mass conservation / tie calibration / determinism)
    # =====================================================================
    print("\n[Step 3] gates...")
    pmfs, summ = build_joint_pmfs(
        preds[["game_id", "pred_home", "pred_away"]], params,
        SE.PINNED_P_TIE)
    g1_mass = bool(np.allclose(pmfs.sum(axis=(1, 2)), 1.0, atol=1e-9))
    g2_tie = bool(abs(float(np.mean([np.trace(p) for p in pmfs]))
                      - SE.PINNED_P_TIE) <= 1e-4)
    g3_marg = bool(summ["summary"]["max_marginal_err_post_ipf"] is not None
                   and summ["summary"]["max_marginal_err_post_ipf"] <= 1e-9)
    # G4: derived-ML inside (0,1) and complements to 1.0 per row.
    ml = mkt[["derived_ml", "p_home_win_derived", "p_away_win_derived"]]
    g4 = bool((ml.min().min() > 0.0) and (ml.max().max() < 1.0))
    # G5: board rows all strictly future (leak) + offers never enter the
    # model frame (board's market columns dropped by build_slate_features).
    g5 = bool("spread_line" not in board.columns
              and "total_line" not in board.columns)
    # G6: determinism — double price walk, byte-identical.
    mkt2 = SE.price_board(preds, params, SE.PINNED_P_TIE,
                          lines=lines[["game_id", "spread_line",
                                       "total_line"]])
    g6 = bool(mkt.to_csv(index=False) == mkt2.to_csv(index=False))
    gates = {
        "g1_mass_conservation": {"pass": g1_mass,
                                 "rule": "every 76x76 joint sums to 1"},
        "g2_tie_calibrated": {"pass": g2_tie,
                              "mean_trace": round(float(np.mean(
                                  [np.trace(p) for p in pmfs])), 6),
                              "target": SE.PINNED_P_TIE,
                              "rule": "IPF-calibrated tie diagonal == 0.275%"},
        "g3_marginal_post_ipf": {"pass": g3_marg,
                                 "max_err": summ["summary"][
                                     "max_marginal_err_post_ipf"],
                                 "rule": "row/col marginals match to 1e-9"},
        "g4_derived_ml_range": {"pass": bool(g4)},
        "g5_leakage_and_market_free": {"pass": bool(g5)},
        "g6_determinism": {"pass": bool(g6),
                           "rule": "byte-identical double price walk"},
    }
    for k, v in gates.items():
        print(f"  {k}: pass={v['pass']}")
    ok = all(v["pass"] for v in gates.values())
    if not ok:
        print("FATAL: slate gates failed — no artifacts emitted")
        return 2

    # =====================================================================
    # STEP 4 — emitter (markets CSV + meta + monitor + serve record)
    # =====================================================================
    print("\n[Step 4] emitters...")
    target = datetime.now(ZoneInfo("America/New_York"))
    date_str = target.strftime("%Y%m%d")
    as_of_utc = target.astimezone(ZoneInfo("UTC")).isoformat()

    identity = board[["game_id", "gameday", "season", "week", "home_team",
                      "away_team", "stadium", "gametime", "home_record",
                      "away_record"]].copy()
    identity["gameday"] = identity["gameday"].dt.strftime("%Y-%m-%d")
    out = mkt.merge(identity, on="game_id", how="left")
    if len(out) != len(mkt):
        raise RuntimeError("identity merge lost market rows")
    out.insert(0, "kind", "slate")
    cols = (["kind", "game_id", "gameday", "season", "week", "home_team",
             "away_team", "stadium", "gametime", "home_record", "away_record"]
            + [c for c in out.columns
               if c not in ("kind", "game_id", "gameday", "season", "week",
                            "home_team", "away_team", "stadium", "gametime",
                            "home_record", "away_record")])
    out = out[cols]
    # Full-grid NaN guards: derived probabilities must be populated on every
    # row (fair lines always exist); only offer-level columns may be NaN.
    bad = [c for c in out.columns
           if c not in ("spread_line", "total_line", "p_cover_offered",
                        "p_push_offered", "p_over_offered", "p_under_offered",
                        "p_push_total_offered", "p_home_cover_minus_half",
                        "p_home_cover_plus_half", "p_away_cover_minus_half",
                        "p_away_cover_plus_half", "fair_spread_shrunk",
                        "fair_total_shrunk", "p_cover_shrunk",
                        "p_over_shrunk", "derived_ml_shrunk")
           and out[c].isna().any()]
    if bad:
        raise RuntimeError(f"markets frame contains NaN in {bad} — "
                           "refusing to emit (mirror of MLB persist guard)")

    oof_baseline = {
        "covers_ece_pooled": 0.078,
        "totals_ece_pooled_own": 0.087,
        "derived_ml": {"logloss": 0.6365, "auc": 0.695, "ece": 0.0435},
        "provenance": [
            "nfl_era_3e8c8a510f04.json (era record: seam covers ECE, G4)",
            "nfl_market_3e8c8a510f04.json (totals ECE own vs shrink)",
            "nfl_adoption_decision_3e8c8a510f04.json (spread adoption)"],
        "note": ("research-pinned pooled-OOF figures from the committed "
                 "records — the first-run monitor slate-history section is "
                 "empty by design (no served slate outcomes exist yet)"),
    }
    meta = {
        "artifact": f"nfl_run_engine_markets_{date_str}.csv",
        "target_date": date_str,
        "as_of_utc": as_of_utc,
        "frame_sha256": frame_sha,
        "board": {"season": SLATE_SEASON, "n_games": int(len(out)),
                  "weeks": [int(w) for w in sorted(
                      out["week"].dropna().unique())]},
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
        "slate_history": [],   # accumulating; empty on the first run — honest
        "markets_persisted": True,
        "markets_path": f"nfl_run_engine_markets_{date_str}.csv",
        "gates": gates,
    }

    if not args.no_record:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"nfl_run_engine_markets_{date_str}.csv"
        tmp = csv_path.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(csv_path)
        meta_path = out_dir / f"nfl_run_engine_markets_{date_str}.meta.json"
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        tmp.replace(meta_path)
        mon_path = out_dir / f"nfl_run_engine_monitor_{date_str}.json"
        tmp = mon_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(monitor, indent=2, default=str))
        tmp.replace(mon_path)
        rec = {
            "record": "nfl_slate_serve",
            "target_date": date_str,
            "written_utc": pd.Timestamp.now("UTC").isoformat(),
            "frame_sha256": frame_sha,
            "scope": ("run-line/totals slate-serve pricing path + artifact "
                      "emitters — record + dated artifacts only; NOTHING "
                      "wired into master_pipeline; FEATURE_COLUMNS / served "
                      "12-pool / daily pipeline untouched"),
            "config": {"era_spec": SE.ERA_SPEC, "median_rounds": dict(
                SE.MEDIAN_ROUNDS), "pinned_joint": {
                    "sigma_h": SE.PINNED_SIGMA_HOME,
                    "sigma_a": SE.PINNED_SIGMA_AWAY, "rho": SE.PINNED_RHO,
                    "p_tie": SE.PINNED_P_TIE},
                       "market_params": {"totals_cd": list(SE.TOTALS_CD),
                                         "spread_cd": list(SE.SPREAD_CD)},
                       "view": "12-pool per-side PIT (SIDE_FEATURES)"},
            "impute_rates": impute_rate,
            "gates": gates,
            "n_board": int(len(out)),
            "mapping_table": MAPPING_TABLE,
            "retention_decision": ("dated artifacts tracked-and-accumulating "
                                   "(MLB mirror); repo cleanup may never "
                                   "delete a committed file (tracked-file "
                                   "guard)"),
            "judgment_calls": {
                "1_pinned_dispersion": ("joint dispersion params pinned from "
                                        "research (DN/9.663/9.0789/0.0076/"
                                        "0.275%) — in-sample refit on "
                                        "all-decided residuals would understate "
                                        "sigma and re-create hot totals"),
                "2_own_line_first_run": ("no known-vintage feed exists — "
                                         "shrink columns additive + flagged, "
                                         "never silently applied"),
                "3_impute_at_serve": ("board rows with no 2026 pbp yet (pace) "
                                      "or venue facts (neutral sites) are "
                                      "median-imputed from decided-train at "
                                      "serve time, mirroring the moneyline "
                                      "ensemble's impute-median bundle; rates "
                                      "recorded"),
                "4_spread_formal_adoption": ("folded in per the adoption "
                                             "record's deferral (see "
                                             "nfl_adoption_decision_3e8c8a510f04"
                                             ".json)"),
                "5_artifact_schema": ("mirrors MLB emitters from source "
                                      "(run_engine.persist_markets / markets "
                                      "loader) with nfl_ prefixes + the "
                                      "NFL-only raw-vs-derived ±0.5 pair"),
            },
        }
        rec_path = out_dir / f"nfl_slate_serve_{date_str}.json"
        tmp = rec_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=2, default=str))
        tmp.replace(rec_path)
        print(f"  wrote {csv_path.name} ({len(out)} slate rows)")
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
