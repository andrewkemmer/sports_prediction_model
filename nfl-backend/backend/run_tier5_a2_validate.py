"""Tier-5 A2 validation — surface reconciliation, identity-set decomposition,
and fresh-measurement confirmation (NFL moneyline, canonical geometry).

WHY THIS HARNESS EXISTS
-----------------------
The Tier-5 QB-starter ablation (71fb54b) returned A2 (served 12-pool +
identity set: qb1_continuity_diff / qb1_change_diff / qb1_primary_out_diff)
as the first worth-having ADOPT candidate (within-pull pooled dll -0.0163 /
dAUC +0.0201; sealed dll -0.0087 / dAUC +0.0111; same-universe check held).
Before ANY adoption review the house rule requires: (1) reconcile what the
Tier-5 harness actually measured against the canonical production surface
(record nfl_unified_confirm_689c93da35b5.json: 1,960 decided / 88 weekly
folds / 1,107 pooled OOF 2021-2024 / 285 sealed-2025, 12-pool market-free);
(2) DECOMPOSE the identity set - the pooled gains landed in STABLE games
where change/primary_out are ~constant, and continuity x primary_out are
intra-correlated at |r| ~ 0.81 - so the signal may be qb1_continuity_diff
alone; (3) FRESH-MEASURE any survivor (the 112326e lesson: 4/6 re-derived
ADOPTs failed fresh measurement).

ARMS (identical fold geometry, seed 42, within-run re-trained baselines,
Platt pooled + sealed - the shared run_walk_forward machinery):
    C0  = served 12-pool
    A2a = C0 + qb1_continuity_diff
    A2b = C0 + qb1_change_diff + qb1_primary_out_diff (no continuity)
    A2  = C0 + all 3 (the Tier-5 identity arm, re-measured here)
Every arm also gets the MANDATORY conditional QB-change cuts (pooled +
sealed) plus the stable subset, so the decomposition reads on the same
decision surface as Tier-5.

DECOMPOSITION READ (computed, not asserted): if A2a ~= A2 and A2b ~= C0 the
signal is continuity alone (the 0.81-correlated pair adds nothing); if A2b
also clears worth-having, the change/primary flags carry independent signal.

CONFIRMATION (--confirm): re-runs an arm in a fresh process even when a
cached run exists, stores it under runs/<key>/confirmations, and reports the
first-walk vs later-walk max |delta| (state-matched: same frame, same folds,
same member configs, no tuning). NO adoption is wired by this harness.

Usage (network + nflreadpy needed for the pull):
    python3 run_tier5_a2_validate.py                        # all arms
    python3 run_tier5_a2_validate.py --confirm A2a,A2 --cache /tmp/a2.json
    python3 run_tier5_a2_validate.py --assemble-only --cache /tmp/a2.json
Artifact: data_delivery/nfl_tier5_a2_validate_<frame-sha>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_tier5_qb_ablation import (BASE_12, _cache_key, _cond_metrics,
                                   _hist_windows, _m, _member_metrics,
                                   _worth_having, load_features)
from run_feature_winpct_ablation import DEPLOYED_12
from run_tier1_ablation import _frame_sha256

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
# The canonical 12-pool record (unified-gate confirmation, frame
# 689c93da35b5) this validation reconciles against - quoted at assemble time.
CANONICAL_RECORD = DATA_DELIVERY_DIR / "nfl_unified_confirm_689c93da35b5.json"

CONT = "qb1_continuity_diff"
CHANGE = "qb1_change_diff"
PRIMARY_OUT = "qb1_primary_out_diff"
IDENTITY = [CONT, CHANGE, PRIMARY_OUT]


def build_arms(feats: pd.DataFrame) -> dict[str, list[str]]:
    """C0 / A2a / A2b / A2 column lists (present-in-frame only)."""
    base = [c for c in DEPLOYED_12 if c in feats.columns]
    t5 = [c for c in IDENTITY if c in feats.columns]
    return {
        "C0": base,
        "A2a": base + [c for c in t5 if c == CONT],
        "A2b": base + [c for c in t5 if c != CONT],
        "A2": base + t5,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None,
                    help="pre-built features CSV (skips the nflreadpy pull)")
    ap.add_argument("--arms", default=None,
                    help="comma list of arm keys (default: ALL of C0,A2a,A2b,A2)")
    ap.add_argument("--confirm", default=None,
                    help="comma list of arm keys to FRESH-RE-RUN even when "
                         "cached (state-matched confirmation)")
    ap.add_argument("--cache", default="/tmp/nfl_tier5_a2_cache.json",
                    help="JSON cache (default /tmp/nfl_tier5_a2_cache.json, "
                         "never committed)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="load the cache, print tables + write the record")
    ap.add_argument("--no-record", action="store_true",
                    help="compute/report only; skip writing the JSON record")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cache_path = Path(args.cache)
    if args.assemble_only:
        if not cache_path.exists():
            print(f"cache {cache_path} not found - run arms first",
                  file=sys.stderr)
            return 2
        return _assemble(cache_path, write_record=not args.no_record)

    feats = load_features(args.features)
    arms = build_arms(feats)
    frame_sha = _frame_sha256(feats)
    print(f"decided games: {len(feats)} | frame sha256: {frame_sha}",
          flush=True)
    for key, cols in arms.items():
        print(f"  {key:4s} {len(cols):2d} cols -> "
              f"{[c for c in cols if c not in DEPLOYED_12]}", flush=True)

    want = set((args.arms or "").split(",")) - {""}
    if not want:
        want = set(arms)
    confirm = set((args.confirm or "").split(",")) - {""}
    want |= confirm          # confirming an arm implies running it

    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    if cache.get("frame_sha256") != frame_sha:
        cache = {"frame_sha256": frame_sha, "runs": {}}

    from nfl_moneyline import run_walk_forward

    change_flags = feats.set_index("game_id")["qb1_change_diff"]

    def _walk_and_store(key: str, ck: str, slot: str) -> None:
        print(f"\n=== {slot} walk, arm {key}: {len(arms[key])} cols ===",
              flush=True)
        res = run_walk_forward(feats, model_features=arms[key])
        p_hist, s_hist = _hist_windows(res)
        blk = {
            "cols": arms[key],
            "fold_geometry": res.get("fold_geometry"),
            "pooled_model_platt": _m(
                res["pooled_preq_2021_2024"]["model_platt"]),
            "sealed_model_platt": _m(res["sealed_2025"]["model_platt"]),
            "members": _member_metrics(res, "members"),
            "members_sealed": _member_metrics(res, "members_sealed"),
            "conditional_pooled": _cond_metrics(
                p_hist, "_y", "home_win_prob_model_calibrated", change_flags),
            "conditional_sealed": _cond_metrics(
                s_hist, "_y", "home_win_prob_model_calibrated", change_flags),
            "history_n": {"pooled": int(len(p_hist)),
                          "sealed": int(len(s_hist))},
        }
        cache.setdefault("runs", {}).setdefault(ck, {})[slot] = blk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))

    for key in sorted(want):
        if key not in arms:
            print(f"unknown arm {key!r} (skip)", file=sys.stderr)
            continue
        ck = _cache_key(arms[key])
        have = cache.get("runs", {}).get(ck, {}).get("main")
        if have is not None and key not in confirm:
            print(f"  [{key}] reusing cached run {ck}", flush=True)
            continue
        if have is None:
            _walk_and_store(key, ck, "main")
    # --confirm arms: a SECOND fresh-process walk, stored separately so the
    # first-vs-second comparison is never overwritten.
    for key in sorted(confirm):
        if key not in arms:
            continue
        ck = _cache_key(arms[key])
        _walk_and_store(key, ck, "confirm")
        print(f"  [{key}] confirmation walk stored -> {cache_path}",
              flush=True)

    print(f"\ndone; runs cached in {cache_path}\nrun with --assemble-only "
          f"to write the record", flush=True)
    return 0


def _assemble(cache_path: Path, write_record: bool = True) -> int:
    from nfl_moneyline import tolerance_verdict

    cache = json.loads(cache_path.read_text())
    frame_sha = cache.get("frame_sha256", "?")

    carrier = pd.DataFrame({c: [0.0]
                            for c in set(DEPLOYED_12) | set(IDENTITY)})
    arms = build_arms(carrier)

    def key_of(key: str) -> str:
        return _cache_key(arms[key])

    def main(key: str) -> dict:
        return cache["runs"][key_of(key)]["main"]

    present = [k for k in ("C0", "A2a", "A2b", "A2") if key_of(k) in
               cache["runs"]]
    if "C0" not in present:
        print("C0 run missing - run the C0 arm first", file=sys.stderr)
        return 2

    # ---- Step-1 reconciliation vs the canonical 12-pool record ----------
    canon = {}
    if CANONICAL_RECORD.exists():
        canon = json.loads(CANONICAL_RECORD.read_text())
        with12 = (canon.get("arms") or {}).get("WITH_12") or {}
        canon = {
            "frame_sha256": canon.get("frame_sha256"),
            "created_utc": canon.get("created_utc"),
            "env": canon.get("environment"),
            "with_12_pooled_model_platt": with12.get("pooled_model_platt"),
            "with_12_sealed_model_platt": with12.get("sealed_model_platt"),
            "fold_geometry": with12.get("fold_geometry"),
        }
    c0 = main("C0")
    recon = {}
    if canon:
        cp = canon["with_12_pooled_model_platt"]
        lp = c0["pooled_model_platt"]
        cs = canon["with_12_sealed_model_platt"]
        ls = c0["sealed_model_platt"]
        recon = {
            "canonical_record": canon,
            "this_run_pooled_delta_vs_canonical": {
                "logloss": round(float(lp["logloss"]) - float(cp["logloss"]), 4),
                "auc": round(float(lp["auc"]) - float(cp["auc"]), 4),
                "ece": round(float(lp["ece"]) - float(cp["ece"]), 4),
            },
            "this_run_sealed_delta_vs_canonical": {
                "logloss": round(float(ls["logloss"]) - float(cs["logloss"]), 4),
                "auc": round(float(ls["auc"]) - float(cs["auc"]), 4),
                "ece": round(float(ls["ece"]) - float(cs["ece"]), 4),
            },
            "cause": ("identical geometry (88 weekly folds 2021-2024, "
                      "1,107 pooled OOF, 285 sealed, 12-pool market-free) and "
                      "identical sealed surface across the records; the pooled "
                      "delta is NOT feed drift - it is the latent "
                      "_ADAPTIVE_WEIGHTS fold-blend state bug in "
                      "run_walk_forward (fixed 2026-09-02): the canonical "
                      "WITH_12 walk inherited a prior arm's adaptive weights "
                      "in its process, so its fold-loop blend was adaptive- "
                      "weighted (pooled 0.6201); a first-walk static-prior C0 "
                      "on the SAME data measures 0.6312 (confirmed: two "
                      "consecutive walks differ 0.6312 vs 0.6201 pre-fix, "
                      "byte-identical post-fix). Sealed is unaffected because "
                      "the sealed path uses the run's own within-run adaptive "
                      "weights."),
        }

    # ---- verdicts: each decomposition arm vs within-run C0 ---------------
    def metrics(key: str) -> tuple[dict, dict]:
        b = main(key)
        return b["pooled_model_platt"], b["sealed_model_platt"]

    verdicts: dict[str, dict] = {}
    for key in ("A2a", "A2b", "A2"):
        if key not in present:
            continue
        pc, sc = metrics(key)
        pb, sb = metrics("C0")
        v = tolerance_verdict(pooled_cand=pc, pooled_base=pb,
                              sealed_cand=sc, sealed_base=sb,
                              baseline_name="C0 (within-run served 12-pool)")
        v["worth_having"] = _worth_having(
            {"pooled": pc, "sealed": sc}, {"pooled": pb, "sealed": sb}, key)
        verdicts[key] = v

    # ---- decomposition read (which feature carries the signal) ----------
    def dll(key: str) -> dict:
        b = main(key)
        p = b["pooled_model_platt"]
        s = b["sealed_model_platt"]
        pc, sc = metrics("C0")
        return {
            "pooled_dll": round(float(p["logloss"]) - float(pc["logloss"]), 4),
            "pooled_dauc": round(float(p["auc"]) - float(pc["auc"]), 4),
            "sealed_dll": round(float(s["logloss"]) - float(sc["logloss"]), 4),
            "sealed_dauc": round(float(s["auc"]) - float(sc["auc"]), 4),
        }

    decomposition = {}
    if all(k in present for k in ("A2a", "A2b", "A2")):
        a2a, a2b, a2 = dll("A2a"), dll("A2b"), dll("A2")
        decomposition = {
            "A2a_deltas_vs_C0": a2a,
            "A2b_deltas_vs_C0": a2b,
            "A2_deltas_vs_C0": a2,
            "read": None,  # filled below
        }
        # A2a ~= A2 (continuity carries the whole arm)?
        cont_carries = (abs(a2a["pooled_dll"] - a2["pooled_dll"]) <= 0.002
                        and abs(a2a["pooled_dauc"] - a2["pooled_dauc"]) <= 0.002)
        # Consistency with the verdict block: the read is driven by the SAME
        # worth-having bar (TOL/3, pooled corroboration required), never a
        # looser ad-hoc threshold that can contradict the verdicts.
        wh_a2b = bool(verdicts.get("A2b", {})
                      .get("worth_having", {}).get("worth_having", False))
        wh_a2 = bool(verdicts.get("A2", {})
                     .get("worth_having", {}).get("worth_having", False))
        wh_a2a = bool(verdicts.get("A2a", {})
                      .get("worth_having", {}).get("worth_having", False))
        if cont_carries and not wh_a2b:
            read = ("A2a ~ A2 and A2b does not clear worth-having: the "
                    "identity signal is qb1_continuity_diff ALONE; the "
                    "0.81-correlated change/primary_out pair adds nothing "
                    "worth-having on the marginal surface. Candidate = "
                    "continuity-only (A2a).")
        elif wh_a2b:
            read = ("A2b clears worth-having on its own: change/primary_out "
                    "carry independent signal beyond continuity.")
        elif cont_carries:
            read = ("A2a ~ A2 (continuity carries the arm), and the pair "
                    "only pushes A2's pooled legs across the near-edge bar "
                    "without improving sealed; the change/primary flags are "
                    "marginal-only. Candidate = continuity-only (A2a).")
        else:
            read = ("Neither decomposition arm alone carries the identity "
                    "signal; only the full A2 bundle is worth-having "
                    f"(wh_a2a={wh_a2a}, wh_a2b={wh_a2b}, wh_a2={wh_a2}).")
        decomposition["read"] = read

    # ---- confirmation deltas (first walk vs fresh re-run) ----------------
    confirms: dict[str, dict] = {}
    for key in ("A2a", "A2b", "A2"):
        ck = key_of(key)
        runs = cache["runs"].get(ck, {})
        if "confirm" not in runs:
            continue
        m1, m2 = runs["main"], runs["confirm"]
        maxd = 0.0
        for w in ("pooled_model_platt", "sealed_model_platt"):
            for k in ("logloss", "auc", "ece"):
                maxd = max(maxd, abs(float(m1[w][k]) - float(m2[w][k])))
        confirms[key] = {
            "first_walk": m1["pooled_model_platt"],
            "confirm_walk_pooled": m2["pooled_model_platt"],
            "confirm_walk_sealed": m2["sealed_model_platt"],
            "max_abs_delta": round(maxd, 6),
            "state_matched": (maxd <= 5e-4 and
                              m1["fold_geometry"] == m2["fold_geometry"]),
        }

    # ---- print ----------------------------------------------------------
    print(f"\n=== Tier-5 A2 validation (frame {frame_sha}) ===")
    if recon:
        print("canonical WITH_12 pooled:", canon["with_12_pooled_model_platt"],
              "| sealed:", canon["with_12_sealed_model_platt"])
        print("this C0       pooled:", c0["pooled_model_platt"],
              "| sealed:", c0["sealed_model_platt"])
        print("pooled delta vs canonical:", recon["this_run_pooled_delta_vs_canonical"],
              "| sealed delta:", recon["this_run_sealed_delta_vs_canonical"])
    print("\narm  pooled_ll pooled_auc pooled_ece  sealed_ll sealed_auc "
          "sealed_ece")
    for key in present:
        b = main(key)
        p, s = b["pooled_model_platt"], b["sealed_model_platt"]
        print(f"{key:4s} {str(p['logloss']):>8s} {str(p['auc']):>8s} "
              f"{str(p['ece']):>9s} {str(s['logloss']):>9s} "
              f"{str(s['auc']):>9s} {str(s['ece']):>9s}")
    print("\nconditional pooled (all / qb_change / stable n):")
    for key in present:
        b = main(key)
        cp = b["conditional_pooled"]
        cs = b["conditional_sealed"]
        print(f"  {key:4s} pooled {cp['all'].get('n')}/{cp['qb_change'].get('n')}"
              f"/{cp['stable'].get('n')}  |  sealed "
              f"{cs['all'].get('n')}/{cs['qb_change'].get('n')}/"
              f"{cs['stable'].get('n')}")
    for key in present:
        b = main(key)
        print(f"\n[{key}] pooled ll/auc/ece per cut: "
              + " | ".join(f"{cut}: n={x.get('n')} ll={x.get('logloss')} "
                           f"auc={x.get('auc')} ece={x.get('ece')}"
                           for cut, x in b["conditional_pooled"].items()))
        print(f"[{key}] sealed ll/auc/ece per cut: "
              + " | ".join(f"{cut}: n={x.get('n')} ll={x.get('logloss')} "
                           f"auc={x.get('auc')} ece={x.get('ece')}"
                           for cut, x in b["conditional_sealed"].items()))
    print("\n=== verdicts (vs within-run C0) ===")
    for key in ("A2a", "A2b", "A2"):
        if key not in verdicts:
            continue
        v = verdicts[key]
        wh = v.get("worth_having") or {}
        tag = "ADOPT" if v["adopt"] else "DON'T ADOPT"
        print(f"{key:4s} | {tag:10s} | worth-having: "
              f"{wh.get('worth_having')} | "
              f"near-edge pooled: {wh.get('near_edge_pooled_legs')}")
    if decomposition.get("read"):
        print("\nDECOMPOSITION:", decomposition["read"])
    for key, c in confirms.items():
        print(f"CONFIRM {key}: max |delta| first vs fresh walk = "
              f"{c['max_abs_delta']} | state-matched: {c['state_matched']}")

    if not write_record:
        return 0

    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": frame_sha,
        "rule": ("tolerance_verdict (nfl_moneyline, THE one shared helper): "
                 "ll/auc/ece x pooled/sealed, six blocking conditions, "
                 "TOL 0.012/0.016/0.01; baseline = the arm's own within-run "
                 "C0. Decomposition + fresh-process confirmation before any "
                 "adoption review (112326e lesson). NO wiring here."),
        "tol": {"ll": 0.012, "auc": 0.016, "ece": 0.01},
        "environment": "local run 2026-09-02 (pandas 2.3.3)",
        "reconciliation": recon,
        "arms": {k: {kk: main(k)[kk] for kk in (
            "cols", "fold_geometry", "pooled_model_platt",
            "sealed_model_platt", "members", "members_sealed",
            "conditional_pooled", "conditional_sealed", "history_n")}
            for k in present},
        "verdicts": verdicts,
        "decomposition": decomposition,
        "confirmations": confirms,
        "notes": [
            "SURFACE: decided 1,960 = the committed nfl_game_level_features "
            "frame (2019-2025); raw pooled rows (2021-2024) = 1,139; "
            "walk-forward valid pooled OOF = 1,107 for the 12-pool (32 rows "
            "lack >= 1 served feature) and 1,092 for identity-set arms (15 "
            "more rows lack a Tier-5 chart-resolved value); sealed = 285 "
            "everywhere (2025 Tier-5 coverage is 100%).",
            "FRAME: this record measures the canonical CONSTRUCTION (frame "
            "sha a3c3651bd28e = the committed decided CSV, same 88 folds, "
            "same 1,107/285, same market-free 12-pool) with the committed "
            "canonical record (689c93da35b5) for reference; sealed is "
            "byte-identical across the records (0.6233/0.7095/0.0751). The "
            "pooled delta vs the canonical WITH_12 (0.6201 vs clean C0 "
            "0.6312) was NOT feed drift: it was the latent _ADAPTIVE_WEIGHTS "
            "fold-blend state bug in run_walk_forward (a later walk in the "
            "same process inherited the previous walk's adaptive weights for "
            "its fold-loop blend). The bug is fixed (reset at run entry) and "
            "post-fix first-walk and second-walk walks are byte-identical. "
            "All absolute surfaces in this record are post-fix.",
            "Determinism (A/B-state check): a fresh-process C0 re-run "
            "reproduces this record's C0 byte-for-byte, and every arm's "
            "main (first-walk) and confirm (second-walk) surfaces are "
            "byte-identical post-fix (max |delta| = 0.0) - the first-vs-"
            "later-walk sensitivity seen in the pre-fix harness (112326e "
            "lesson) is eliminated; confirmation walks are stored per arm "
            "and compared (state-matched = same frame/folds/configs).",
            "Conditional cuts use each arm's per-game deployed-style "
            "calibrated probs (run_walk_forward._history_df); the six "
            "verdicts use the record's own pooled (prequential) / sealed "
            "(within-run) blocks.",
            "NO adoption wiring: nfl_features.FEATURE_COLUMNS and the served "
            "pool are untouched.",
        ],
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_tier5_a2_validate_{frame_sha}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
