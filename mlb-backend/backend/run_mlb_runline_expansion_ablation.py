"""Run-line feature expansion ablation — re-test the 5 pruned run-engine
features on the corrected C2 base.

Background (2026-09): the run engine serves 53/59 features. The 5 dropped
candidates are the engineered composites in RUN_EXTRA_EXCLUSIONS: they were
excluded by the 2026-08-30 feature-restore decision
(run_engine_cull_diagnostic_20260830.json, git 730c34b) — a GROUP test
(FULL = 53 + all 5) was WORSE on the derived-ML leg than B (53) on the
then-current frame (AUC 0.5641 vs 0.5682, holdout ll 0.6847 vs 0.6846), so
they stayed out. Their interaction FACTORS are now restored (the 24
matchup-gap diffs) and the corrected C2 base (k-edge expansion, per-run
refit, drift band [1.33, 1.73], live since d274651) prices extremes
honestly. Question: does any pruned feature add signal on the corrected
base? (run_margin_diff is EXCLUDED by design — a moneyline feature, run
engine READ-ONLY; closed NULL by the fold-local margin-k ablation 1d683cd.)

Candidates (interaction of two restored diffs, coverage on the fresh C2
frame — 7,018 decided games, game_level_features.csv 2026-09-01):
    A1 lineup_handedness_matchup_advantage  (pure matchup term, PSI 0.139>
        floor 0.055 on 08-30 frame — rule-excluded + drift)
    A2 bullpen_meltdown_risk       bullpen_pitches_diff × whip_diff
    A3 pitcher_regression_indicator sp_fbvelo_diff × sp_era_5g_diff
    A4 lineup_depth_multiplier     woba_mean_diff × woba_top3_diff
    A5 ace_efficiency_factor       sp_k9_5g_diff × sp_whiff_diff
  (A2–A5: rule-excluded composites; low global-fit gain, no/within-noise
   drift on the 08-30 frame; AALL = the group test those five were last
   measured as.)

Arms, same geometry as the run-engine walk-forward (74-fold geometry,
RETRAIN_CADENCE_DAYS=7, MIN_VAL_FOLD_GAMES=40, RANDOM_SEED=42, decided
frame ~7,018, sealed-21d holdout, fit-on-OOF / evaluate-on-sealed):
    C0   = current 53-feature C2 base (baseline)
    A1..A5 = C0 + one candidate each
    AALL = C0 + all 5 (58)

C2 layer ACTIVE in EVERY arm (the whole point — measuring on the corrected
base): per arm, k is REFIT on that arm's pre-holdout OOF only
(ke.fit_k_edge + ke.k_edge_holdout_mask — the production per-run refit
policy; sealed never sees k), the λ pair is expanded with
ke.apply_k_edge, and the α(λ) curves are fit on the arm's EXPANDED
pre-holdout λs (select_alpha_curve, the derive_markets_v3 path) before
NB-MC pricing and CRPS.

Per arm (pooled + sealed):
  MARGIN (primary): CRPS on the full margin distribution (_nb_score_pmf +
  _crps over MARGIN_GRID); run-line −1.5 cover calibration in p-deciles,
  >0.65/>0.70 p-bins, and |λ_edge| >=0.5/0.70/0.90 bins (the probe's
  surface: "run-line −1.5 cover under-priced at extremes").
  TOTALS: O/U calibration by assigned line + ECE + CRPS on the sum (must
  stay flat within noise of C0 — the sum must not move).
  DERIVED ML: calibration + P(win) SD (target 0.066) + [0.55,0.60) gap.
  Per-line calibration tables for every arm.

Gate (task discipline): a feature ADOPTs iff it wins C0 on sealed CRPS
AND per-line calibration (extreme-bin |delta| closes to within noise) AND
totals stay flat within tolerance AND pooled corroborates. A feature that
only helps pooled is NOT adopted.

Record: data_delivery/mlb_runline_expansion_<frame>.json (frame = data
hash), written after EACH arm (resumable). Per-arm OOF cached under
/tmp/runline_oof_<frame>_<arm>.parquet. COMMITS NOTHING.

Usage:
    python run_mlb_runline_expansion_ablation.py --arms C0
    python run_mlb_runline_expansion_ablation.py --arms C0,A1,A2,A3,A4,A5,AALL
    python run_mlb_runline_expansion_ablation.py --smoke   # 6 folds, /tmp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))
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
from margin_reliability_diagnostic import _crps  # noqa: E402
from run_engine import (  # noqa: E402
    MC_DRAWS,
    MARKET_SEED,
    HOLDOUT_DAYS,
    TOTAL_LINE_GRID,
    alpha_of,
    derive_markets_mc,
    derive_run_features,
    run_oof,
    select_alpha_curve,
)
import run_engine_k_edge as ke  # noqa: E402
from run_engine import _rounded_total_line  # noqa: E402
from run_engine_challenger_ablation import (  # noqa: E402
    MARGIN_GRID,
    TOTAL_GRID,
    price_arm as challenger_price_arm,
)
from run_engine_k_edge_gate import (  # noqa: E402
    _bin_rows,
    _edge_bin_rows,
    _metrics,
    _pwin_gap,
    _totals_rows,
)
from training import FEATURE_COLS  # noqa: E402

ECE_TOL = 0.005                 # totals/run-line ECE degradation tolerance
EXTREME_BINS = (0.65, 0.70)
PWIN_BUCKET = (0.55, 0.60)
PWIN_SD_TARGET = 0.066
MIN_DECILE_N = 20
ALPHA_SEED_OFFSET = {"home": 1, "away": 2}

CANDIDATES = {
    "A1": "lineup_handedness_matchup_advantage",
    "A2": "bullpen_meltdown_risk",
    "A3": "pitcher_regression_indicator",
    "A4": "lineup_depth_multiplier",
    "A5": "ace_efficiency_factor",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
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


def arm_features() -> dict[str, list[str]]:
    """Arm feature lists: C0 = production 53-view; A1..A5 = +1 composite;
    AALL = + all 5 (58). Derived from derive_run_features so C0 is exactly
    the served pool today."""
    kept, dropped = derive_run_features(list(FEATURE_COLS))
    assert len(kept) == 53, f"production run view must be 53, got {len(kept)}"
    arms: dict[str, list[str]] = {"C0": list(kept)}
    for tag, f in CANDIDATES.items():
        assert f in FEATURE_COLS, f"{f} must exist in FEATURE_COLS"
        arms[tag] = list(kept) + [f]
    arms["AALL"] = list(kept) + [CANDIDATES[a] for a in
                                 ("A1", "A2", "A3", "A4", "A5")]
    return arms


def arm_drop_terms(arm_feats: list[str]) -> list[str]:
    """The 'dropped' list run_oof expects when run_features is given: every
    FEATURE_COLS col not in the arm (the derivation would produce the same
    for the base; keep it explicit for the override contract)."""
    return [c for c in FEATURE_COLS if c not in arm_feats]


def price_arm(oof: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS,
              n_draws: int = MC_DRAWS, seed: int = MARKET_SEED) -> dict:
    """C2-layered pricing of one arm's OOF: refit k on the arm's pre-holdout
    OOF only, expand the λ pair (sum preserved), fit α(λ) curves on the
    EXPANDED pre-holdout λs (the derive_markets_v3 path), then NB-MC table
    + margin/totals PMFs for CRPS. Returns every surface the gate reads.

    Sealed discipline: pre_mask = dates < max − holdout_days (derive_markets_
    v3's own cutoff). k + α + binning NEVER see sealed rows.
    """
    dates = pd.to_datetime(oof["game_date"])
    cutoff = dates.max() - pd.Timedelta(days=holdout_days)
    pre = (dates < cutoff).to_numpy()
    hold = ~pre

    lam_h = oof["home_expected_runs"].to_numpy(float)
    lam_a = oof["away_expected_runs"].to_numpy(float)
    hs = oof["home_score"].to_numpy(float)
    as_ = oof["away_score"].to_numpy(float)
    margin = hs - as_
    total = hs + as_
    home_covers = (margin >= 2).astype(float)   # −1.5 line cover
    home_won = (margin > 0).astype(float)       # derived-ML target
    edge_c0 = lam_h - lam_a                     # shared |edge| binning basis

    # C2 layer: per-arm refit on the pre-holdout OOF only (production
    # per-run refit policy). Sealed never sees k.
    k = ke.fit_k_edge(lam_h, lam_a, margin, pre)
    lh2, la2 = ke.apply_k_edge(lam_h, lam_a, k)

    # α(λ) curves fit on the EXPANDED pre-holdout λs (derive_markets_v3
    # path — same seed offsets, same form selection).
    curves = {}
    alpha_cols = {}
    for side, off in ALPHA_SEED_OFFSET.items():
        lam = lh2 if side == "home" else la2
        y = hs if side == "home" else as_
        curve, _diag = select_alpha_curve(y[pre], lam[pre],
                                          seed=seed + off)
        curves[side] = curve
        alpha_cols[side] = alpha_of(lam, curve)

    mc = derive_markets_mc(lh2, la2, alpha_cols["home"], alpha_cols["away"],
                           n_draws=n_draws, seed=seed)
    pmf_m, pmf_t = challenger_price_arm(oof, lh2, la2,
                                        alpha_cols["home"],
                                        alpha_cols["away"])

    def _crps_window(pmf: np.ndarray, grid: list, y: np.ndarray,
                     mask: np.ndarray) -> float:
        if not mask.any():
            return None
        return round(float(_crps(pmf[mask], grid, y[mask])), 5)

    # Totals grid per game (assigned rounded line — the O/U surface).
    tp, ty, tidx = _totals_rows(mc["p_over_grid"], lh2, la2, total)
    pwin = mc["p_home_win_derived"]
    pcov = mc["p_home_cover_1_5"]

    # Strictly-prior discipline: "pooled" = the pre-holdout window ONLY
    # (challenger convention), so every pooled surface is PIT-invariant to
    # sealed flips. Sealed = the holdout window.
    rl = {
        "pooled": _bin_rows(pcov, home_covers, pre),
        "sealed": _bin_rows(pcov, home_covers, hold),
        "metrics_pooled": _metrics(pcov, home_covers, pre),
        "metrics_sealed": _metrics(pcov, home_covers, hold),
    }
    rl_edge = {
        "pooled": _edge_bin_rows(pcov, home_covers, edge_c0, pre),
        "sealed": _edge_bin_rows(pcov, home_covers, edge_c0, hold),
    }
    totals = {
        "metrics_pooled": _metrics(tp, ty, pre[tidx]),
        "metrics_sealed": _metrics(tp, ty, hold[tidx]),
        "crps_pooled": _crps_window(pmf_t, TOTAL_GRID, total, pre),
        "crps_sealed": _crps_window(pmf_t, TOTAL_GRID, total, hold),
        "by_line_pooled": _totals_by_line(mc["p_over_grid"], lh2, la2,
                                           total, pre),
        "by_line_sealed": _totals_by_line(mc["p_over_grid"], lh2, la2,
                                           total, hold),
    }
    derived_ml = {
        "metrics_pooled": _metrics(pwin, home_won, pre),
        "metrics_sealed": _metrics(pwin, home_won, hold),
        "pwin_sd_pooled": round(float(pwin[pre].std(ddof=1)), 4),
        "pwin_sd_sealed": round(float(pwin[hold].std(ddof=1)), 4),
        "bucket_55_60_pooled": _pwin_gap(pwin, home_won, pre,
                                         *PWIN_BUCKET),
        "bucket_55_60_sealed": _pwin_gap(pwin, home_won, hold,
                                         *PWIN_BUCKET),
    }
    return {
        "k": {"k_fitted_run": round(float(k), 4),
              "fit": "pre-holdout OOF only (per-run refit)", "edge": "C2"},
        "n_pre": int(pre.sum()), "n_sealed": int(hold.sum()),
        "margin_crps_pooled": _crps_window(pmf_m, MARGIN_GRID,
                                           margin, pre),
        "margin_crps_sealed": _crps_window(pmf_m, MARGIN_GRID,
                                           margin, hold),
        "run_line_minus_1_5": rl,
        "run_line_edge_bins": rl_edge,
        "totals": totals,
        "derived_ml": derived_ml,
    }


def _totals_by_line(p_over_grid: np.ndarray, lam_h: np.ndarray,
                    lam_a: np.ndarray, total: np.ndarray,
                    mask: np.ndarray | None = None) -> dict:
    """Per-line pred-vs-actual over table (assigned rounded line, push
    excluded) — the O/U per-line calibration table mapped back onto the
    full grid (mirrors the gate's _totals_rows, line-grouped). ``mask``
    restricts to a window (strictly-prior / sealed)."""
    ps: list[float] = []
    ys: list[float] = []
    lines: list[float] = []
    for i in range(len(lam_h)):
        if mask is not None and not mask[i]:
            continue
        line = _rounded_total_line(lam_h[i], lam_a[i])
        if line not in TOTAL_LINE_GRID:
            continue
        j = TOTAL_LINE_GRID.index(line)
        p = float(p_over_grid[i, j])
        if np.isnan(p) or total[i] == line:      # push (whole numbers only)
            continue
        ps.append(p)
        ys.append(float(total[i] > line))
        lines.append(float(line))
    if not ps:
        return {"by_line": [], "overall_delta": None}
    df = pd.DataFrame({"p": ps, "y": ys, "line": lines})
    rows = []
    for line, g in df.groupby("line"):
        if len(g) < 10:
            continue
        rows.append({"line": float(line), "n": int(len(g)),
                     "pred": round(float(g["p"].mean()), 4),
                     "actual": round(float(g["y"].mean()), 4),
                     "delta": round(float(g["p"].mean()
                                          - g["y"].mean()), 4)})
    return {"by_line": rows,
            "overall_delta": round(float(df["p"].mean() - df["y"].mean()), 4)
            if len(df) else None}


def _se_band(n: int) -> float:
    return float(np.sqrt(0.05 * 0.95 / max(n, 1)))


def _edge_gap_close(arm: dict, which: str) -> bool | None:
    rows = arm["run_line_edge_bins"][which]["extreme"]
    gs = [abs(r["delta"]) for r in rows
          if r.get("delta") is not None and (r.get("n") or 0) >= 10]
    return bool(gs and max(gs) <= 0.05)


def run_arm(oof: pd.DataFrame) -> dict:
    return price_arm(oof)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=str,
                    default="C0,A1,A2,A3,A4,A5,AALL")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="6 folds, /tmp output, gate skipped")
    args = ap.parse_args()
    if args.smoke:
        # 40 folds ≈ 3,600 OOF rows: alpha_bins' quantile bins stay above
        # ALPHA_MIN_BIN without entering the small-sample merge loop, and
        # the walk-forward cost is the same as a real arm (limit only trims
        # the priced tail).
        args.limit_folds = min(args.limit_folds or 40, 40)
        args.out = Path("/tmp/mlb_runline_expansion_smoke.json")
        args.arms = "C0,A1"

    sha = head_sha()
    data_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    games = load_game_features(data_path)
    decided = get_decided_frame(games)
    frame_sha = sha256_file(data_path)
    frame = frame_sha[:16]
    print(f"commit={sha[:12]} frame={frame} decided_games={len(decided)} "
          f"holdout_days={args.holdout_days}")

    feats = arm_features()
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in want:
        if a not in feats:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(feats)}")

    out = args.out or (DATA_DELIVERY_DIR /
                       f"mlb_runline_expansion_{frame}.json")
    if out.exists():
        record = json.loads(out.read_text())
    else:
        record = {"schema": "mlb-runline-expansion/v1",
                  "commit_sha": sha, "frame": frame,
                  "data_sha256": frame_sha,
                  "holdout_days": args.holdout_days,
                  "seed": int(RANDOM_SEED),
                  "mc_draws": int(MC_DRAWS),
                  "n_decided": int(len(decided)),
                  "candidates": CANDIDATES,
                  "c2": "ACTIVE in every arm (per-run k refit on the arm's "
                        "pre-holdout OOF; α curves fit on expanded pre-holdout "
                        "λs)",
                  "arms": {}}
        out.write_text(json.dumps(record, indent=2) + "\n")

    for name in want:
        if name in record["arms"]:
            print(f"  arm {name} already recorded — skipping")
            continue
        arm_feats = feats[name]
        print(f"\n  {name}: {len(arm_feats)} features "
              f"(base + {sorted(set(arm_feats) - set(feats['C0']))}) — "
              f"run-engine walk-forward + C2 pricing ...", flush=True)

        # Per-arm OOF cache (frame + arm-keyed; deterministic walk-forward).
        h = hashlib.sha256()
        h.update(frame.encode())
        h.update(json.dumps(arm_feats).encode())
        key = h.hexdigest()[:16]
        cache = (Path(tempfile.gettempdir())
                 / f"runline_oof_{key}.parquet")
        if cache.exists():
            oof = pd.read_parquet(cache)
        else:
            oof = run_oof(decided, run_features=arm_feats,
                          dropped=arm_drop_terms(arm_feats),
                          retrain_cadence_days=RETRAIN_CADENCE_DAYS,
                          min_val_games=MIN_VAL_FOLD_GAMES)["oof"]
            oof.to_parquet(cache)   # ALWAYS the FULL walk-forward OOF
        if args.limit_folds:
            # Trim only for pricing (smoke): never cache a trimmed OOF.
            oof = oof.tail(min(len(oof), args.limit_folds * 90))
        res = price_arm(oof, holdout_days=args.holdout_days)
        res["n_folds_oof"] = int(oof["fold_idx"].nunique())
        res["n_oof_games"] = int(len(oof))
        res["features_added"] = sorted(set(arm_feats) - set(feats["C0"]))

        record["arms"][name] = res
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"    sealed margin CRPS {res['margin_crps_sealed']} "
              f"(pooled {res['margin_crps_pooled']}) | totals CRPS "
              f"{res['totals']['crps_sealed']} | P(win) SD "
              f"{res['derived_ml']['pwin_sd_sealed']} | "
              f"k={res['k']['k_fitted_run']}", flush=True)

    # ---- gate: per-feature verdict vs C0 ----
    if not args.smoke and "C0" in record["arms"]:
        c0 = record["arms"]["C0"]
        adopt, reject = [], []
        for name in [a for a in ("A1", "A2", "A3", "A4", "A5")
                     if a in record["arms"]]:
            a = record["arms"][name]
            scaled_win = (a["margin_crps_sealed"] is not None and
                          c0["margin_crps_sealed"] is not None and
                          a["margin_crps_sealed"] < c0["margin_crps_sealed"])
            pooled_corr = (a["margin_crps_pooled"] is not None and
                           c0["margin_crps_pooled"] is not None and
                           a["margin_crps_pooled"] < c0["margin_crps_pooled"])
            line_ok = _edge_gap_close(a, "pooled") and \
                _edge_gap_close(a, "sealed") is not False
            totals_ok = (abs(a["totals"]["metrics_pooled"]["ece"]
                             - c0["totals"]["metrics_pooled"]["ece"])
                         <= ECE_TOL)
            ml_ok = a["derived_ml"]["pwin_sd_sealed"] >= \
                c0["derived_ml"]["pwin_sd_sealed"]
            verdict = bool(scaled_win and pooled_corr and line_ok and
                           totals_ok and ml_ok)
            (adopt if verdict else reject).append(name)
            record.setdefault("gate", {})[name] = {
                "feature": CANDIDATES[name],
                "verdict": "ADOPT" if verdict else "DON'T ADOPT",
                "sealed_crps_delta": round(
                    (a["margin_crps_sealed"] - c0["margin_crps_sealed"])
                    if None not in (a["margin_crps_sealed"],
                                    c0["margin_crps_sealed"]) else None, 5),
                "pooled_crps_delta": round(
                    (a["margin_crps_pooled"] - c0["margin_crps_pooled"])
                    if None not in (a["margin_crps_pooled"],
                                    c0["margin_crps_pooled"]) else None, 5),
                "sealed_win": bool(scaled_win),
                "pooled_corroborates": bool(pooled_corr),
                "line_extreme_ok": bool(line_ok),
                "totals_ece_delta": round(
                    a["totals"]["metrics_pooled"]["ece"]
                    - c0["totals"]["metrics_pooled"]["ece"], 5),
                "totals_ok": bool(totals_ok),
                "derived_ml_pwin_sd": a["derived_ml"]["pwin_sd_sealed"],
                "derived_ml_ok": bool(ml_ok),
            }
        record.setdefault("gate", {})["summary"] = {
            "adopted": adopt,
            "rejected": reject,
            "rule": "ADOPT iff wins sealed CRPS AND pooled corroborates AND "
                    "run-line extreme-edge bins close to within noise AND "
                    "totals ECE flat within 0.005 AND derived-ML P(win) SD "
                    "not regressed; pooled-only wins never adopt"}
        out.write_text(json.dumps(record, indent=2) + "\n")
        print("\n================= GATE =================")
        print("adopted:", adopt or "none", "| rejected:", reject)
        for name, g in record["gate"].items():
            if isinstance(g, dict) and "feature" in g:
                print(f"  {name} {CANDIDATES[name]}: {g['verdict']} "
                      f"(sealed ΔCRPS {g['sealed_crps_delta']}, "
                      f"pooled ΔCRPS {g['pooled_crps_delta']})")


if __name__ == "__main__":
    main()