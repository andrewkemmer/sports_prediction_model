"""Consolidated unified-gate CONFIRMATION ablation for the NFL moneyline.

WHY THIS HARNESS EXISTS
-----------------------
Policy 2026-09-02 (commit d7c3ffa) unified every NFL ablation harness onto
the ONE shared MLB-shaped gate rule (``nfl_moneyline.tolerance_verdict`` —
ll/auc/ece x pooled/sealed, each of the six conditions blocking, nothing
else). That unification RE-DERIVED 6 arms from OLD record data to ADOPT:

    corr  WITHOUT_YPP     (removal vs WITH_13 pool)
    corr  WITHOUT_BOTH    (removal vs WITH_13 pool)
    raw   RAW_ADDED       (C0 + 14 raw per-side cols, per-member masks)
    tier3 ROSTER          (WITHOUT + roster_age_diff / roster_exp_diff)
    tier4 QB              (WITHOUT + ewm_qb_epa_starter_diff)
    xgb_reg C1 (OPPADJ)   (C0 + 3 opponent-adjusted cols, regularized xgb)

Those are RE-DERIVED verdicts (rule changed, numbers unchanged), NOT fresh
measurements. This harness re-measures every one of them on the CURRENT
frame (same pull, same 12-pool market-free base, same walk-forward geometry
as the production gate: pooled 2021-2024 weekly folds, sealed-2025) and
writes the fresh per-arm tables. Adoption is NOT wired here — the adoption
commit is a separate, deliberate decision AFTER review of these tables.

IT ALSO COMPLETES THE TIER-1 RECORD: ``nfl_tier1_ablation_<sha>.json`` was
written on Kaggle and never committed (pooled AUC/ECE never entered the
review table). The tier-1 arms (T1_WITHOUT / T1_WITH / T1_WITH_ADMITTED /
T1_WITH_SUBSET / T1_TIER1_ONLY, same design as run_tier1_ablation.py) are
re-measured here and their record is written with pooled AUC/ECE included.

ARMS AND THEIR OWN WITHIN-RUN BASELINES
--------------------------------------
Each arm is measured vs ITS OWN WITHOUT baseline re-trained within-run
(the unified harness convention — the production gate's incumbent baseline
only exists for the served candidate). The current-geometry served base is
the 12-pool (DEPLOYED_12 == FEATURE_COLUMNS minus the is_home anchor):

  key            runner    candidate cols                     baseline
  corr WITH_12   plain     12-pool (shared served base)       —
  corr WITHOUT_YPP  plain  WITH_12 - ewm_ypp_diff             WITH_12
  corr WITH_QBEPA plain  WITH_12 + ewm_qb_epa_play_diff      —   (== the
                           historical WITH_13 pool, for removal-baseline
                           parity with the re-derived corr verdicts)
  corr WITHOUT_BOTH plain WITH_QBEPA - ypp - qbepa            WITH_QBEPA
                           (same 11-column candidate as WITHOUT_YPP —
                           measured once, verdict differs via baseline)
  raw  C0          plain  WITH_12                             —
  raw  RAW_ADDED   masked WITH_12 + 14 raws (trees/mlp);      C0
                           logistic stays on C0 only
  tier3 WITHOUT_13 plain  WITHOUT_FEATURES + VENUE_3 (13)     —   (the
                           historical tier-3 geometry == WITH_QBEPA)
  tier3 ROSTER_13  plain  WITHOUT_13 + roster (15)            WITHOUT_13
  tier3 ROSTER     plain  WITH_12 + roster (14)               WITH_12
                           (current-geometry variant — what adoption
                            would actually change)
  tier4 QB         plain  WITH_12 + ewm_qb_epa_starter_diff   WITH_12
  xgb  C0_REG      reg    WITH_12 (regularized xgb)           —
  xgb  C1          reg    WITH_12 + 3 OPPADJ (logistic C0)    C0_REG
  tier1 T1_WITHOUT plain  WITHOUT_FEATURES (10)               —
  tier1 T1_WITH    plain  T1_WITHOUT + all 9 Tier-1           T1_WITHOUT
  tier1 T1_WITH_ADMITTED plain T1_WITHOUT + the 7 admitted   T1_WITHOUT
  tier1 T1_WITH_SUBSET   plain T1_WITHOUT + the 3 strongest   T1_WITHOUT
  tier1 T1_TIER1_ONLY    plain the 7 admitted Tier-1 alone    T1_WITHOUT

Removal arms (corr): the removal arm is the CANDIDATE; adopt=True means the
removal pool is WITHIN TOLERANCE of the pool that includes the feature(s) —
the same tolerance_verdict rule, candidate = removal arm (the
run_feature_corr_ablation convention).

RUNNERS (identical to their origin harnesses, never re-invented)
----------------------------------------------------------------
  plain   -> nfl_moneyline.run_walk_forward(feats, model_features=cols)
  masked  -> run_nfl_raw_ablation.run_walk_forward_masked(tree, logi)
  reg     -> run_nfl_xgb_reg_ablation.run_walk_forward_reg(tree, logi)

Column-identical arms SHARE one walk-forward (cache key = content hash of
runner + cols + logistic_cols — stable across processes) — e.g.
corr.WITH_QBEPA and tier3.WITHOUT_13 are the same 13-column list, so one
measurement serves both arms; raw.C0 / tier4.WITH_12 / corr.WITH_12 share
the 12-pool.

USAGE (Kaggle — network + nflreadpy needed for the pull):
    python3 run_nfl_unified_confirm_ablation.py                  # all arms
    python3 run_nfl_unified_confirm_ablation.py --features <f.csv>
    python3 run_nfl_unified_confirm_ablation.py --arms QB,ROSTER --cache /tmp/c.json
    python3 run_nfl_unified_confirm_ablation.py --assemble-only --cache /tmp/c.json
Artifacts: data_delivery/nfl_unified_confirm_<sha>.json AND
data_delivery/nfl_tier1_ablation_<sha>.json (same schema as
run_tier1_ablation.py — this fills the never-committed tier-1 record).
The frame SHA is the content hash of the feature frame (same helper as
every other harness), so the record ties back to exactly one pull.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_features import TIER3_ROSTER_FEATURES, run_feature_gate
from nfl_raw_columns import RAW_PER_SIDE_COLS
from nfl_tier4 import TIER4_OPPADJ_FEATURES, TIER4_QB_FEATURES
from run_feature_winpct_ablation import DEPLOYED_12
from run_tier1_ablation import (TIER1_ADMITTED, TIER1_FEATURES, TIER1_SUBSET,
                                WITHOUT_FEATURES, _frame_sha256,
                                _member_metrics, adopt_verdict)
from run_tier2_ablation import VENUE_3_FEATURES

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

QBEPA = "ewm_qb_epa_play_diff"
YPP = "ewm_ypp_diff"

# Current-geometry served base: FEATURE_COLUMNS minus the is_home anchor.
BASE_12 = list(DEPLOYED_12)


def load_features(features_csv: str | None) -> pd.DataFrame:
    """Feature frame: a provided CSV, else the nflreadpy pull + build.

    The CSV path must carry EVERY candidate column the arms need (the raw
    per-side columns AND the Tier-4 / roster candidates). The pull path
    composes them the same way the origin harnesses do — build_features +
    compose_tier4_features + compose_raw_columns on the same decided frame.
    """
    if features_csv and Path(features_csv).exists():
        feats = pd.read_csv(features_csv)
        feats["gameday"] = pd.to_datetime(feats["gameday"])
    else:
        import nflreadpy
        from nfl_features import (DEFAULT_SEASONS, TIER1_NEEDS, _decided_rows,
                                  build_features)
        from nfl_moneyline import DECIDED_FRAME
        from nfl_raw_columns import compose_raw_columns
        from nfl_tier4 import TIER4_PBP_NEEDS, compose_tier4_features
        seasons = DEFAULT_SEASONS
        sched = nflreadpy.load_schedules(seasons).to_pandas()
        pbp = nflreadpy.load_pbp(seasons)
        keep = [c for c in (("game_id", "posteam", "yards_gained", "epa",
                             "qb_epa", "game_seconds_remaining")
                            + TIER1_NEEDS + TIER4_PBP_NEEDS)
                if c in pbp.columns]
        pbp = pbp.select(keep).to_pandas()
        decided = pd.read_csv(DECIDED_FRAME)
        decided = decided[decided["season"].isin(seasons)]
        feats = build_features(decided, sched, pbp)
        feats = compose_tier4_features(feats, sched, pbp)
        feats = compose_raw_columns(feats, sched, pbp)
        feats["_decided"] = feats["game_id"].isin(
            set(_decided_rows(sched)["game_id"]))
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


def _only(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns present in the frame — never silently all-NaN inputs."""
    return [c for c in cols if c in df.columns]


def build_arms(feats: pd.DataFrame) -> dict[str, dict]:
    """Arm table: key -> {cols, logistic_cols (masked/reg runners), baseline}."""

    def P(cols):  # plain runner cols
        return _only(feats, cols)

    base12 = P(BASE_12)
    without10 = P(WITHOUT_FEATURES)
    without13 = P(WITHOUT_FEATURES + VENUE_3_FEATURES)
    with_qbepa = P(BASE_12 + [QBEPA])
    tier1 = P(TIER1_FEATURES)
    admitted = P(TIER1_ADMITTED)
    subset = P(TIER1_SUBSET)
    roster = P(TIER3_ROSTER_FEATURES)
    qb = P(TIER4_QB_FEATURES)
    oppadj = P(TIER4_OPPADJ_FEATURES)
    raws = P(RAW_PER_SIDE_COLS)

    arms: dict[str, dict] = {}

    def arm(key: str, cols: list[str], runner: str = "plain",
            base: str | None = None,
            logistic_cols: list[str] | None = None):
        arms[key] = {
            "cols": cols, "runner": runner, "baseline": base,
            "logistic_cols": logistic_cols,
        }

    # corr-pair removal arms (removal = candidate)
    arm("WITH_12", base12)
    arm("WITHOUT_YPP", [c for c in base12 if c != YPP], base="WITH_12")
    arm("WITH_QBEPA", with_qbepa)
    both = [c for c in with_qbepa if c not in (YPP, QBEPA)]
    arm("WITHOUT_BOTH", both, base="WITH_QBEPA")

    # raw per-side (masked: trees/mlp raws+diffs; logistic diffs only)
    arm("C0", base12)
    arm("RAW_ADDED", base12 + raws, runner="masked",
        base="C0", logistic_cols=base12)

    # tier-3 roster (historical 13-col geometry + current-geometry variant)
    arm("WITHOUT_13", without13)
    arm("ROSTER_13", without13 + roster, base="WITHOUT_13")
    arm("ROSTER", base12 + roster, base="WITH_12")

    # tier-4 QB-conditional
    arm("QB", base12 + qb, base="WITH_12")

    # xgb-reg OPPADJ (regularized xgb; logistic on C0 both arms)
    arm("C0_REG", base12, runner="reg")
    arm("C1", base12 + oppadj, runner="reg", base="C0_REG",
        logistic_cols=base12)

    # tier-1 arms
    arm("T1_WITHOUT", without10)
    arm("T1_WITH", without10 + tier1, base="T1_WITHOUT")
    arm("T1_WITH_ADMITTED", without10 + admitted, base="T1_WITHOUT")
    arm("T1_WITH_SUBSET", without10 + subset, base="T1_WITHOUT")
    arm("T1_TIER1_ONLY", admitted, base="T1_WITHOUT")

    return arms


def _cache_key(runner: str, cols: list[str],
               logistic_cols: list[str] | None = None) -> str:
    """Content-stable cache key (sha1 of runner + sorted columns) — safe
    across processes (never Python's randomized builtin hash)."""
    c = sorted(set(cols))
    l = sorted(set(logistic_cols)) if logistic_cols is not None else c
    h = hashlib.sha1(
        (runner + "|" + json.dumps(c) + "|" + json.dumps(l)).encode()
    ).hexdigest()[:12]
    return f"{runner}|{h}"


def _run_arm(runner: str, feats: pd.DataFrame, cols: list[str],
             logistic_cols: list[str] | None) -> dict:
    """Dispatch to the SAME runner each origin harness uses (never a re-implementation)."""
    if runner == "plain":
        from nfl_moneyline import run_walk_forward
        return run_walk_forward(feats, model_features=cols)
    if runner == "masked":
        from run_nfl_raw_ablation import run_walk_forward_masked
        # same-set default: masks only differ when an arm declares them
        return run_walk_forward_masked(
            feats, tree_cols=cols,
            logistic_cols=logistic_cols if logistic_cols is not None else cols)
    if runner == "reg":
        from run_nfl_xgb_reg_ablation import run_walk_forward_reg
        # reg harness requires explicit equal lists for the same-set arm
        return run_walk_forward_reg(
            feats, tree_cols=cols,
            logistic_cols=logistic_cols if logistic_cols is not None else cols)
    raise ValueError(f"unknown runner {runner}")


def _m(rec: dict) -> dict:
    """{logloss, auc, ece} from a within-run block (rounded for the record)."""
    out = {}
    for k in ("logloss", "auc", "ece"):
        v = rec.get(k)
        out[k] = v if v is None else round(float(v), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--arms", default=None,
                    help="comma list of arm keys to run (default: ALL keys)")
    ap.add_argument("--cache", default="/tmp/nfl_unified_confirm_cache.json",
                    help="JSON cache of per-arm walk-forward results (default "
                         "/tmp/nfl_unified_confirm_cache.json, never committed)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="load the cache, print tables + write both records")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON records")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cache_path = Path(args.cache)

    if args.assemble_only:
        if not cache_path.exists():
            print(f"cache {cache_path} not found — run arms first",
                  file=sys.stderr)
            return 2
        return _assemble(cache_path, write_record=not args.no_record)

    feats = load_features(args.features)
    arms = build_arms(feats)
    frame_sha = _frame_sha256(feats)
    print(f"decided games: {len(feats)} | frame sha256: {frame_sha}",
          flush=True)
    for key, a in arms.items():
        flag = {"plain": "P", "masked": "M", "reg": "R"}[a["runner"]]
        print(f"  {key:15s} [{flag}] {len(a['cols']):2d} cols "
              f"-> {a['cols']}", flush=True)

    want = set((args.arms or "").split(",")) - {""}
    if not want:
        want = set(arms)

    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    if not cache.get("frame_sha256") == frame_sha:
        # different frame => stale runs must not be mixed in
        cache = {"frame_sha256": frame_sha, "runs": {}}

    for key in sorted(want):
        if key not in arms:
            print(f"unknown arm {key!r} (skip)", file=sys.stderr)
            continue
        a = arms[key]
        ck = _cache_key(a["runner"], a["cols"], a["logistic_cols"])
        if ck in cache["runs"]:
            print(f"  [{key}] reusing cached run {ck}", flush=True)
            continue
        print(f"\n=== running arm {key} (runner={a['runner']}) ===", flush=True)
        res = _run_arm(a["runner"], feats, a["cols"], a["logistic_cols"])
        cache["runs"][ck] = {
            "runner": a["runner"],
            "cols": a["cols"],
            "logistic_cols": a["logistic_cols"],
            "fold_geometry": res.get("fold_geometry"),
            "pooled_model_platt": _m(res["pooled_preq_2021_2024"]["model_platt"]),
            "sealed_model_platt": _m(res["sealed_2025"]["model_platt"]),
            "members": _member_metrics(res, "members"),
            "members_sealed": _member_metrics(res, "members_sealed"),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"  cached {len(cache['runs'])} unique runs -> {cache_path}",
              flush=True)

    # Tier-1 admission-gate report (informational; the walk-forward verdicts
    # are the arbiter — parity with run_tier1_ablation.py).
    try:
        gate = run_feature_gate(feats)
        cache["tier1_gate"] = {
            c: gate["reasons"].get(c)
            or "reverted: not in production candidate pool (FEATURE_COLUMNS)"
            for c in TIER1_FEATURES
        }
        cache_path.write_text(json.dumps(cache, indent=2))
    except Exception as e:  # noqa: BLE001 — informational only
        print(f"tier1_gate skipped: {e!r}", file=sys.stderr)

    print(f"\ndone; {len(cache['runs'])} unique runs cached in {cache_path}\n"
          f"run with --assemble-only to write the records", flush=True)
    return 0


def _assemble(cache_path: Path, write_record: bool = True) -> int:
    cache = json.loads(cache_path.read_text())
    frame_sha = cache.get("frame_sha256", "?")

    all_cols = set()
    for r in cache["runs"].values():
        all_cols.update(r["cols"])
        if r.get("logistic_cols"):
            all_cols.update(r["logistic_cols"])
        all_cols.update({"home_win", "home_score", "away_score"})
    carrier = pd.DataFrame({c: [0.0] for c in sorted(all_cols)})
    arms = build_arms(carrier)

    def key_of(key: str) -> str:
        a = arms[key]
        return _cache_key(a["runner"], a["cols"], a["logistic_cols"])

    def metrics(key: str) -> tuple[dict, dict]:
        r = cache["runs"][key_of(key)]
        return r["pooled_model_platt"], r["sealed_model_platt"]

    def verdict_named(cand: str, base: str, baseline_name: str) -> dict:
        """tolerance_verdict via the shared adopt_verdict wrapper on canned
        within-run blocks; baseline_name names the pool for reason strings."""
        from nfl_moneyline import tolerance_verdict
        pc, sc = metrics(cand)
        pb, sb = metrics(base)
        return tolerance_verdict(pooled_cand=pc, pooled_base=pb,
                                 sealed_cand=sc, sealed_base=sb,
                                 baseline_name=baseline_name)

    pairs = [
        ("WITHOUT_YPP", "WITH_12", "WITH_12 pool (served 12)", "corr removal"),
        ("WITHOUT_BOTH", "WITH_QBEPA", "WITH_QBEPA pool (hist. WITH_13)",
         "corr removal"),
        ("RAW_ADDED", "C0", "C0 (served 12)", "raw per-side"),
        ("ROSTER_13", "WITHOUT_13", "WITHOUT_13 (hist. tier-3 base)",
         "tier3 roster"),
        ("ROSTER", "WITH_12", "WITH_12 pool (current geometry)",
         "tier3 roster"),
        ("QB", "WITH_12", "WITH_12 pool (current geometry)",
         "tier4 QB-conditional"),
        ("C1", "C0_REG", "C0_REG (regularized xgb, served 12)",
         "xgb-reg OPPADJ"),
        ("T1_WITH", "T1_WITHOUT", "T1_WITHOUT arm", "tier1"),
        ("T1_WITH_ADMITTED", "T1_WITHOUT", "T1_WITHOUT arm", "tier1"),
        ("T1_WITH_SUBSET", "T1_WITHOUT", "T1_WITHOUT arm", "tier1"),
        ("T1_TIER1_ONLY", "T1_WITHOUT", "T1_WITHOUT arm", "tier1"),
    ]
    verdicts = {c: verdict_named(c, b, n) for c, b, n, _ in pairs}

    print(f"\n=== unified-gate confirmation — per-arm tables "
          f"(frame {frame_sha}) ===")
    present = [k for k in arms if key_of(k) in cache["runs"]]
    print("arm            runner  pooled_ll  pooled_auc  pooled_ece"
          "  sealed_ll  sealed_auc  sealed_ece")
    for key in sorted(present):
        r = cache["runs"][key_of(key)]
        p, s = r["pooled_model_platt"], r["sealed_model_platt"]
        print(f"{key:15s} {r['runner']:6s} {str(p['logloss']):>9s} "
              f"{str(p['auc']):>10s} {str(p['ece']):>10s} "
              f"{str(s['logloss']):>9s} {str(s['auc']):>10s} "
              f"{str(s['ece']):>9s}")

    print("\n=== verdicts (tolerance_verdict: ll/auc/ece x pooled/sealed,"
          " each blocking) ===")
    for cand, base, name, note in pairs:
        if key_of(cand) not in cache["runs"] or key_of(base) not in cache["runs"]:
            print(f"{cand:16s} vs {base:12s} | SKIPPED (run missing)")
            continue
        v = verdicts[cand]
        tag = "ADOPT" if v["adopt"] else "DON'T ADOPT"
        print(f"{cand:16s} vs {base:12s} | {tag:10s} | {note}")
        for r in v["reasons"]:
            print(f"    - {r}")

    if not write_record:
        return 0

    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": frame_sha,
        "rule": ("tolerance_verdict (nfl_moneyline, THE one shared helper): "
                 "ll_ok = cand <= base + TOL_LL; auc_ok = cand >= base "
                 "- TOL_AUC; ece_ok = cand <= base + ECE_TOL — each on "
                 "pooled AND sealed, each blocking; adopt = all six. "
                 "Baseline per arm = the arm's own WITHIN-RUN base re-trained "
                 "in the same geometry (removal arms: candidate = removal, "
                 "baseline = the pool WITH the feature(s))."),
        "tol": {"ll": 0.012, "auc": 0.016, "ece": 0.01},
        "environment": "local confirmation run 2026-09-02 (pandas 2.3.3)",
        "arms": {},
        "verdicts": {},
        "notes": [
            "FRESH MEASUREMENT of the 6 arms the unified-gate review "
            "(nfl_harness_review_unified_gate.json) re-derived to ADOPT from "
            "OLD record data — this is a real gate pass on the current "
            "geometry (same pull, same 12-pool market-free base, same "
            "walk-forward: pooled 2021-2024 weekly folds, sealed 2025).",
            "corr.WITH_QBEPA == the historical corr WITH_13 pool (13 cols); "
            "corr.WITHOUT_BOTH's 11-col candidate == WITH_12 minus ypp "
            "(ewm_qb_epa_play_diff is already unserved), so WITHOUT_YPP and "
            "WITHOUT_BOTH share one walk-forward measurement — their "
            "verdicts differ only via the baselines (WITH_12 vs WITH_QBEPA).",
            "tier3.WITHOUT_13 == corr.WITH_QBEPA (13 cols: 10 v1/v2 + "
            "VENUE_3, incl. the unserved qbepa twin) — one shared run. "
            "ROSTER (current geometry) is measured against the served "
            "12-pool; ROSTER_13 preserves the historical tier-3 geometry for "
            "record-to-record comparison.",
            "RAW_ADDED / C1 use the per-member masks of their origin "
            "harnesses: logistic sees only the diffs+flags base; trees and "
            "mlp see the arm's columns. C1 additionally uses the "
            "REGULARIZED xgb (REGULARIZED_XGB_PARAMS) applied IDENTICALLY "
            "to C0_REG and C1.",
            "NO adoption wiring in this session: nfl_features.FEATURE_COLUMNS "
            "and the served pool are untouched.",
        ],
    }
    for key in sorted(present):
        a = arms[key]
        r = cache["runs"][key_of(key)]
        record["arms"][key] = {
            "features": a["cols"],
            "logistic_features": a["logistic_cols"],
            "runner": a["runner"],
            "baseline": a["baseline"],
            "baseline_features": (arms[a["baseline"]]["cols"]
                                  if a["baseline"] else None),
            "fold_geometry": r.get("fold_geometry"),
            "pooled_model_platt": r["pooled_model_platt"],
            "sealed_model_platt": r["sealed_model_platt"],
            "members": r["members"],
            "members_sealed": r["members_sealed"],
        }
    record["verdicts"] = {c: verdicts[c] for c, _, _, _ in pairs}

    # ---- tier-1 record (schema parity with run_tier1_ablation.py) --------
    def _t1(key):
        r = cache["runs"][key_of(key)]
        return {"features": arms[key]["cols"],
                "sealed_model_platt": r["sealed_model_platt"],
                "pooled_model_platt": r["pooled_model_platt"],
                "members": r["members"],
                "members_sealed": r["members_sealed"]}

    tier1_record = {
        "created_utc": record["created_utc"],
        "frame_sha256": frame_sha,
        "arms": {k: _t1(k) for k in ("T1_WITHOUT", "T1_WITH",
                                     "T1_WITH_ADMITTED", "T1_WITH_SUBSET",
                                     "T1_TIER1_ONLY")},
        "tier1_gate": cache.get("tier1_gate") or {},
        "verdict_with": verdicts["T1_WITH"],
        "verdict_with_admitted": verdicts["T1_WITH_ADMITTED"],
        "verdict_with_subset": verdicts["T1_WITH_SUBSET"],
        "verdict_tier1_only": verdicts["T1_TIER1_ONLY"],
    }

    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        DATA_DELIVERY_DIR / f"nfl_unified_confirm_{frame_sha}.json",
        DATA_DELIVERY_DIR / f"nfl_tier1_ablation_{frame_sha}.json",
    ]
    for path, rec in zip(paths, (record, tier1_record)):
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())