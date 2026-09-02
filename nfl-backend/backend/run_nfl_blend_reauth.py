"""NFL blend-state re-derivation audit (post-5fd0549).

Background: ``run_walk_forward`` read the module-global ``_ADAPTIVE_WEIGHTS``
which was only WRITTEN at the end of a walk, so any multi-arm harness that
walked arms sequentially in ONE process made every walk after the first
blend with the PREVIOUS arm's adaptive weights instead of the static
``ENSEMBLE_WEIGHTS`` priors. The entry reset in 5fd0549 fixed it (post-fix
consecutive walks are byte-identical). This harness re-measures every arm
configuration of the affected records (unified-confirm 112326e / tier-1 /
adoption decision 5398e71) under the fixed code with FRESH-PROCESS walks,
then re-derives the unified-gate verdicts on the clean surfaces.

Key properties (mirrors run_tier5_a2_validate.py conventions):

  * Each arm runs in a SEPARATE subprocess (``--subprocess`` dispatch) so
    every measurement is a first-walk static-prior measurement — no
    cross-arm state, and byte-identical to what production would emit.
  * Results are cached by content key (runner + sorted columns) in a JSON
    cache (never committed) so interrupted batches resume.
  * Verdicts use the ONE shared helper ``nfl_moneyline.tolerance_verdict``
    with the same baselines as nfl_unified_confirm_ablation.py — nothing
    re-invented.

Usage:
    python3 run_nfl_blend_reauth.py --features <frame.csv> \
        --arms WITH_12,ROSTER --cache /tmp/nfl_blend_reauth_cache.json
    python3 run_nfl_blend_reauth.py --features <frame.csv> \
        --arms ALL --cache /tmp/...            # every distinct config
    python3 run_nfl_blend_reauth.py --assemble-only --cache /tmp/...
    python3 run_nfl_blend_reauth.py --verify <key> --features <frame.csv>

Artifact: data_delivery/nfl_blend_reauth_audit_<sha>.json (the audit
record, written by --assemble-only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_nfl_unified_confirm_ablation import build_arms as _uc_build_arms

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"

# The original (contaminated) record this audit re-derives — committed in
# 112326e. Read-only reference; never modified.
ORIGINAL_RECORD = DATA_DELIVERY_DIR / "nfl_unified_confirm_689c93da35b5.json"
ADOPTION_RECORD = DATA_DELIVERY_DIR / "nfl_adoption_decision_689c93da35b5.json"

# Verdict pairs, identical to run_nfl_unified_confirm_ablation.py.
VERDICT_PAIRS = [
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

TOL = {"ll": 0.012, "auc": 0.016, "ece": 0.01}


def load_features(features_csv: str) -> pd.DataFrame:
    feats = pd.read_csv(features_csv)
    feats["gameday"] = pd.to_datetime(feats["gameday"])
    if "home_win" not in feats.columns:
        feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
    return feats


def _frame_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    s = df.sort_values("game_id").reset_index(drop=True)
    h.update(s.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()[:12]


def _cache_key(runner: str, cols: list[str],
               logistic_cols: list[str] | None = None) -> str:
    c = sorted(set(cols))
    l = sorted(set(logistic_cols)) if logistic_cols is not None else c
    h = hashlib.sha1(
        (runner + "|" + json.dumps(c) + "|" + json.dumps(l)).encode()
    ).hexdigest()[:12]
    return f"{runner}|{h}"


# Set once main() resolves --features; used by the subprocess dispatch.
_FRAME_PATH = ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None, help="pre-built features CSV")
    ap.add_argument("--arms", default=None,
                    help="comma list of arm keys (default: ALL distinct "
                         "configs of the unified-confirm record)")
    ap.add_argument("--cache", default="/tmp/nfl_blend_reauth_cache.json")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--verify", default=None,
                    help="arm key to walk TWICE in fresh processes and "
                         "compare byte-identity (determinism pin)")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cache_path = Path(args.cache)

    if args.assemble_only:
        if not cache_path.exists():
            print(f"cache {cache_path} not found", file=sys.stderr)
            return 2
        return _assemble(cache_path, write_record=not args.no_record)

    if not args.features:
        print("--features required", file=sys.stderr)
        return 2
    global _FRAME_PATH
    _FRAME_PATH = str(Path(args.features).resolve())

    feats = load_features(args.features)
    frame_sha = _frame_sha256(feats)
    print(f"decided games: {len(feats)} | frame sha256: {frame_sha}",
          flush=True)

    arms = _uc_build_arms(feats)
    want = set((args.arms or "").split(",")) - {""}
    if args.verify:
        want = {args.verify}
    elif not want:
        want = set(arms)

    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    if cache.get("frame_sha256") != frame_sha:
        cache = {"frame_sha256": frame_sha, "runs": {}, "verify": {}}

    verify_runs: list[dict] = []
    if args.verify:
        # Resume the 2-walk determinism check: keep any already-cached
        # verify walks for this key and top up to exactly two fresh walks.
        verify_runs = list((cache.get("verify", {}).get(
            _cache_key(arms[args.verify]["runner"], arms[args.verify]["cols"],
                       arms[args.verify]["logistic_cols"]), {}) or {}).values())
    for key in sorted(want):
        if key not in arms:
            print(f"unknown arm {key!r} (skip)", file=sys.stderr)
            continue
        a = arms[key]
        ck = _cache_key(a["runner"], a["cols"], a["logistic_cols"])
        if not args.verify and ck in cache["runs"]:
            print(f"  [{key}] reusing cached run {ck}", flush=True)
            continue
        print(f"\n=== walking arm {key} (runner={a['runner']}, "
              f"{len(a['cols'])} cols, key {ck}) ===", flush=True)
        blk = _run_fresh_subprocess(a["runner"], a["cols"], a["logistic_cols"])
        if args.verify:
            verify_runs.append(blk)
            cache["verify"][ck] = {
                "run_" + str(i + 1): b for i, b in enumerate(verify_runs)}
            cache_path.write_text(json.dumps(cache, indent=2))
            print(f"  verify walk {len(verify_runs)} cached ({ck})", flush=True)
            continue
        cache["runs"][ck] = blk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        print(f"  cached {len(cache['runs'])} runs -> {cache_path}", flush=True)

    if args.verify:
        if len(verify_runs) == 2:
            same = verify_runs[0] == verify_runs[1]
            print("\nVERIFY (two fresh-process walks of the same arm):",
                  "BYTE-IDENTICAL" if same else "DIFFER", flush=True)
            if not same:
                print("  walk1:", json.dumps(verify_runs[0])[:400], flush=True)
                print("  walk2:", json.dumps(verify_runs[1])[:400], flush=True)
                return 1
        elif len(verify_runs) == 1:
            print("\nverify: first walk done — run again for the second "
                  "fresh-process walk", flush=True)
    return 0


def _run_fresh_subprocess(runner: str, cols: list[str],
                          logistic_cols: list[str] | None) -> dict:
    """Execute the arm's walk in a FRESH python process (guaranteed clean
    module state) and return the compact surface block."""
    lc = "None" if logistic_cols is None else json.dumps(logistic_cols)
    dispatch = ("from nfl_moneyline import run_walk_forward; "
                "res = run_walk_forward(feats, model_features=cols)")
    if runner == "masked":
        dispatch = ("from run_nfl_raw_ablation import run_walk_forward_masked; "
                    "res = run_walk_forward_masked("
                    "feats, tree_cols=cols, "
                    "logistic_cols=lc if lc is not None else cols)")
    elif runner == "reg":
        dispatch = ("from run_nfl_xgb_reg_ablation import run_walk_forward_reg; "
                    "res = run_walk_forward_reg("
                    "feats, tree_cols=cols, "
                    "logistic_cols=lc if lc is not None else cols)")
    payload = f"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, {os.getcwd()!r})
import pandas as pd
feats = pd.read_csv({_FRAME_PATH!r})
feats["gameday"] = pd.to_datetime(feats["gameday"])
if "home_win" not in feats.columns:
    feats["home_win"] = (feats["home_score"] > feats["away_score"]).astype(int)
cols = {json.dumps(cols)}
lc = {lc}
{dispatch}
def _m(rec, key):
    b = rec[key]["model_platt"]
    return {{k: (None if b.get(k) is None else round(float(b[k]), 6))
            for k in ("logloss", "auc", "ece")}}
def _mem(rec, key):
    return {{str(m): dict(v) for m, v in (rec.get(key) or {{}}).items()}}
out = {{
    "cols": cols,
    "logistic_cols": (lc if lc is not None else cols),
    "fold_geometry": res.get("fold_geometry"),
    "pooled_model_platt": _m(res, "pooled_preq_2021_2024"),
    "sealed_model_platt": _m(res, "sealed_2025"),
    "members": _mem(res, "members"),
    "members_sealed": _mem(res, "members_sealed"),
}}
print("BLEND_REAUTH_RESULT " + json.dumps(out))
"""
    proc = subprocess.run(
        [sys.executable, "-c", payload],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess walk failed rc={proc.returncode}: "
            f"{proc.stderr[-2000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("BLEND_REAUTH_RESULT "):
            return json.loads(line[len("BLEND_REAUTH_RESULT "):])
    raise RuntimeError("no BLEND_REAUTH_RESULT in subprocess output: "
                       f"{proc.stdout[-2000:]}")


def _assemble(cache_path: Path, write_record: bool = True) -> int:
    cache = json.loads(cache_path.read_text())
    frame_sha = cache.get("frame_sha256", "?")
    from nfl_moneyline import tolerance_verdict

    orig = json.loads(ORIGINAL_RECORD.read_text())

    # Map arm keys -> cache keys using the arms table (column identity).
    all_cols = set()
    for r in cache["runs"].values():
        all_cols.update(r.get("cols", []))
        if r.get("logistic_cols"):
            all_cols.update(r["logistic_cols"])
    all_cols.update({"home_win", "home_score", "away_score"})
    carrier = pd.DataFrame({c: [0.0] for c in sorted(all_cols)})
    arms = _uc_build_arms(carrier)

    def key_of(key: str) -> str:
        a = arms[key]
        return _cache_key(a["runner"], a["cols"], a["logistic_cols"])

    def metrics(key: str) -> dict:
        r = cache["runs"][key_of(key)]
        return r["pooled_model_platt"], r["sealed_model_platt"]

    print(f"\n=== clean (post-fix) per-arm surfaces (frame {frame_sha}) ===")
    present = [k for k in arms if key_of(k) in cache["runs"]]
    print("arm            pooled_ll   pooled_auc  pooled_ece "
          " sealed_ll   sealed_auc  sealed_ece")
    for key in sorted(present):
        p, s = metrics(key)
        print(f"{key:15s} {str(p['logloss']):>9s} {str(p['auc']):>10s} "
              f"{str(p['ece']):>10s} {str(s['logloss']):>9s} "
              f"{str(s['auc']):>10s} {str(s['ece']):>9s}")

    verdicts = {}
    print("\n=== re-derived verdicts (tolerance_verdict, clean surfaces) ===")
    for cand, base, name, note in VERDICT_PAIRS:
        if key_of(cand) not in cache["runs"] or key_of(base) not in cache["runs"]:
            print(f"{cand:16s} vs {base:12s} | SKIPPED (run missing)")
            continue
        pc, sc = metrics(cand)
        pb, sb = metrics(base)
        v = tolerance_verdict(pooled_cand=pc, pooled_base=pb,
                              sealed_cand=sc, sealed_base=sb,
                              baseline_name=name)
        verdicts[cand] = v
        tag = "ADOPT" if v["adopt"] else "DON'T ADOPT"
        print(f"{cand:16s} vs {base:12s} | {tag:10s} | {note}")
        for r in v.get("reasons", []):
            print(f"    - {r}")

    # Compare vs the original record's surfaces/verdicts.
    print("\n=== original (contaminated) vs clean per-arm surfaces ===")
    orig_arms = orig.get("arms", {})
    for key in sorted(present):
        p, s = metrics(key)
        o = orig_arms.get(key)
        if not o:
            continue
        op, os_ = o["pooled_model_platt"], o["sealed_model_platt"]
        def _d(a, b, k):
            try:
                return None if a.get(k) is None or b.get(k) is None \
                    else round(float(b[k]) - float(a[k]), 4)
            except Exception:
                return None
        print(f"{key:15s} pooled ll {op['logloss']} -> {p['logloss']} "
              f"({_d(op, p, 'logloss')}) | auc {op['auc']} -> {p['auc']} "
              f"({_d(op, p, 'auc')}) | ece {op['ece']} -> {p['ece']} "
              f"({_d(op, p, 'ece')})")

    ov = orig.get("verdicts", {})
    flips = []
    for cand in VERDICT_PAIRS:
        c = cand[0]
        if c not in verdicts or c not in ov:
            continue
        o = ov[c].get("adopt")
        n = verdicts[c].get("adopt")
        if o != n:
            flips.append((c, o, n))
    print("\n=== verdict flips (original contaminated -> clean) ===")
    if flips:
        for c, o, n in flips:
            print(f"  FLIP {c}: {o} -> {n}")
    else:
        print("  none")

    # ---- inventory: every multi-arm NFL record vs the blend bug ----------
    # Walk-order classification per Step 1 of the audit; each entry states
    # WHY it is (not) affected so the table is self-contained.
    inventory = [
        {"record": "nfl_unified_confirm_689c93da35b5 (112326e)",
         "harness": "run_nfl_unified_confirm_ablation.py",
         "walk_order": ("sequential in ONE process via "
                        "nfl_moneyline.run_walk_forward"),
         "affected": True,
         "why": ("plain runner writes _ADAPTIVE_WEIGHTS at end of walk; "
                 "every arm after the first blended with the previous arm's "
                 "adaptive weights. Empirically contaminated arms (clean-vs-"
                 "original deltas ~= 0.010-0.016 pooled ll): WITH_12, C0, "
                 "ROSTER, WITHOUT_13/WITH_QBEPA, T1_WITH_ADMITTED, "
                 "T1_WITH_SUBSET. RAW_ADDED, QB, C0_REG, C1, T1_WITHOUT, "
                 "T1_WITH, T1_TIER1_ONLY, ROSTER_13 measured byte-identical "
                 "(deltas 0.0)."),
         "re_derived": True},
        {"record": "nfl_window_gate_93a6e821f6ad (233d21b)",
         "harness": "run_nfl_window_gate.py",
         "walk_order": ("sequential arms W2016 then W2014 in ONE process, "
                        "but via the harness's OWN twin fold loop"),
         "affected": False,
         "why": ("the twin loop (run_walk_forward_gate) computes adaptive "
                 "weights into a LOCAL and never writes the "
                 "nfl_moneyline._ADAPTIVE_WEIGHTS global; no plain "
                 "run_walk_forward is called in this process, so the global "
                 "stays empty and BOTH arms blend with the static "
                 "ENSEMBLE_WEIGHTS priors on every walk. Mechanically "
                 "immune."),
         "re_derived": False,
         "verdict_row": {
             "W2016": {"original": "DON'T ADOPT (pooled ECE 0.0378 > "
                                    "incumbent 0.0216 + 0.01)",
                       "clean": "DON'T ADOPT (measurement was clean — "
                                "static-prior blend for both arms)"},
             "W2014": {"original": "DON'T ADOPT (sealed ECE 0.084 > "
                                    "incumbent 0.0496 + 0.01)",
                       "clean": "DON'T ADOPT (clean)"},
             "consequence": ("the stay-at-2019 window decision STANDS; the "
                             "window gate does NOT reopen")}},
        {"record": "nfl_window_ablation_e1a489f28b9d (32c338e)",
         "harness": "run_nfl_window_ablation.py",
         "walk_order": ("sequential arms in ONE process via the harness's "
                        "OWN twin (run_walk_forward_window)"),
         "affected": False,
         "why": ("twin loop never writes the _ADAPTIVE_WEIGHTS global — "
                 "static priors every walk, same immunity as the window "
                 "gate"),
         "re_derived": False},
        {"record": "nfl_raw_ablation_8b3cb475639b",
         "harness": "run_nfl_raw_ablation.py (C0 first, then RAW_ADDED)",
         "walk_order": "sequential in ONE process (plain + masked twins)",
         "affected": False,
         "why": ("masked twin never writes the global; and the "
                 "canonical-frame re-derivation shows RAW_ADDED's surfaces "
                 "byte-identical original vs clean (deltas 0.0)"),
         "re_derived": True},
        {"record": "nfl_xgb_reg_ablation_b3057bc8e870",
         "harness": "run_nfl_xgb_reg_ablation.py (C0 first, then C1)",
         "walk_order": "sequential in ONE process (plain + reg twins)",
         "affected": False,
         "why": ("reg twin never writes the global; re-derivation shows "
                 "C0_REG and C1 byte-identical original vs clean"),
         "re_derived": True},
        {"record": "nfl_tier1_ablation_689c93da35b5",
         "harness": "run_tier1_ablation.py (T1_WITHOUT first, then WITH arms)",
         "walk_order": "sequential in ONE process (plain runner)",
         "affected": False,
         "why": ("candidate WITH arms walked AFTER the WITHOUT baseline — "
                 "candidates were the contaminated walks, but the "
                 "canonical-frame re-derivation shows T1_WITH/T1_WITHOUT/"
                 "T1_TIER1_ONLY byte-identical original vs clean; only the "
                 "later T1_WITH_ADMITTED/T1_WITH_SUBSET walks were "
                 "contaminated, and their clean verdicts are re-derived "
                 "here"),
         "re_derived": True},
        {"record": "tier2/3/4 + corr + winpct ablations (old frames "
         "e4aee120a4b8 / ca7ed8d61cd9)",
         "harness": "run_tier2/3/4_ablation.py, run_feature_corr/winpct_ablation.py",
         "walk_order": "sequential in ONE process (plain runner)",
         "affected": True,
         "why": ("same plain-runner architecture; old-frame records show "
                 "the contamination signature (first arm ~0.634 clean, "
                 "later arms inflated ~0.622/0.704). These frames are "
                 "SUPERSEDED — the unified-confirm record re-measured every "
                 "arm on the canonical 689c93da35b5 frame, which this audit "
                 "re-derives clean; no separate re-run needed"),
         "re_derived": "via unified-confirm re-derivation"},
        {"record": "nfl_tier5_a2_validate_a3c3651bd28e (71fb54b)",
         "harness": "run_tier5_a2_validate.py",
         "walk_order": "per-arm subprocess isolation (post-fix harness)",
         "affected": False,
         "why": ("already re-derived clean in the 5fd0549 validation; "
                 "standing: data-lever"),
         "re_derived": "done in 5fd0549"},
        {"record": "nfl_tier5_qb_ablation_a3c3651bd28e",
         "harness": "run_tier5_qb_ablation.py",
         "walk_order": ("sequential in ONE process via the plain runner, "
                        "run pre-fix (2026-09-02T17:37Z)"),
         "affected": False,
         "why": ("its verdicts are decided on member-raw surfaces (the "
                 "immune class — per-member predictions never read the "
                 "fold-blend global); pooled legs were not decisive. "
                 "Listed for completeness, not re-derived."),
         "re_derived": False},
        {"record": "nfl_adoption_decision_689c93da35b5 (5398e71)",
         "harness": "decision record (no harness)",
         "walk_order": "n/a — decided on the contaminated surfaces",
         "affected": True,
         "why": ("cited the contaminated 0.6201/0.7069 'production C0' and "
                 "used contaminated later-walk measurements for ROSTER and "
                 "the rejected RAW_ADDED/QB arms; verdicts re-derived in "
                 "this audit — see adoption_flags"),
         "re_derived": True},
    ]

    corrected_baselines = [
        {"record": "nfl_adoption_decision_689c93da35b5 (5398e71)",
         "cited_as": "baseline_with_12 (ROSTER survivor table)",
         "correction": ("0.6201/0.7069 contaminated -> clean 0.6312/0.6923 "
                        "(sealed 0.6233/0.7095 unchanged). The ROSTER "
                        "worth-having FAIL relied on contaminated candidate "
                        "AND baseline surfaces; both re-derived here — the "
                        "clean verdict remains DON'T ADOPT (no flip on the "
                        "decision, but the numbers cited are stale).")},
        {"record": "xgb_reg arms + C0_REG comparison in the same record",
         "cited_as": "'C0_REG itself UNDERPERFORMS the production plain-"
                     "runner 12 (pooled 0.6314/0.6925 vs C0 0.6201/0.7069)'",
         "correction": ("the comparison baseline 0.6201/0.7069 was "
                        "contaminated; clean C0 is 0.6312/0.6923. Against "
                        "the CLEAN baseline C0_REG still underperforms "
                        "(0.6314/0.6925 pooled) — the conclusion survives, "
                        "the cited number does not.")},
        {"record": "nfl_tier5_a2_validate_a3c3651bd28e",
         "cited_as": "prose citing 0.6201 as the contaminated canonical",
         "correction": ("already correctly framed as the contaminated "
                        "measurement in 71fb54b's post-mortem — no change; "
                        "this audit's clean surfaces confirm it")},
    ]

    # Adoption-relevant consequences of the flips.
    adoption_flags = [
        {"flip": "RAW_ADDED DON'T->ADOPT",
         "flag": ("raw per-side was previously rejected on contaminated "
                  "pooled legs (candidate was clean, baseline C0 was "
                  "contaminated 0.6201). Clean re-derivation passes the "
                  "six-condition gate — previously-rejected arm now passing "
                  "pooled: ADOPTION-RELEVANT, needs a fresh worth-having "
                  "review before any wiring decision.")},
        {"flip": "QB DON'T->ADOPT",
         "flag": ("tier4 QB-conditional previously rejected on the same "
                  "contaminated-WITH_12-baseline artifact. Clean verdict "
                  "ADOPT — ADOPTION-RELEVANT, needs worth-having review.")},
        {"flip": "ROSTER_13 ADOPT->DON'T",
         "flag": ("historical tier-3 base arm (13-pool) passed the "
                  "original gate vs its WITHOUT_13 baseline but fails clean "
                  "on sealed ECE 0.0811 vs 0.0562 + 0.01. NOT wired (the "
                  "served pool is the 12 + is_home), so no production "
                  "consequence — bookkeeping only.")},
        {"unchanged": "ROSTER, C1, T1_WITH, T1_WITH_SUBSET clean verdicts "
                      "match the original ADOPT; T1_WITH_ADMITTED, "
                      "T1_TIER1_ONLY, WITHOUT_YPP, WITHOUT_BOTH clean "
                      "verdicts match the original DON'T ADOPT.",
         "note": ("the 5398e71 worth-having rejection of ROSTER and C1 was "
                  "razor-thin noise reasoning on contaminated surfaces; the "
                  "clean ROSTER deltas vs clean WITH_12 are still "
                  "razor-thin (pooled ll +0.0023, auc -0.0013, ece -0.0049) "
                  "— the no-wiring decision is UNCHANGED in substance.")},
    ]

    if not write_record:
        return 0

    record = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "frame_sha256": frame_sha,
        "method": ("fresh-process re-walks under the 5fd0549 entry reset "
                   "(every walk = first-walk static-prior measurement); "
                   "verdicts via nfl_moneyline.tolerance_verdict with the "
                   "unified-confirm baselines"),
        "tol": TOL,
        "references": {
            "original_record": str(ORIGINAL_RECORD.name),
            "adoption_decision": str(ADOPTION_RECORD.name),
        },
        "runs": cache["runs"],
        "verify": cache.get("verify", {}),
        "verdicts": verdicts,
        "flips": [{"candidate": c, "original": o, "clean": n}
                  for c, o, n in flips],
        "inventory": inventory,
        "window_gate_row": next(
            (i.get("verdict_row") for i in inventory
             if i.get("record", "").startswith("nfl_window_gate")), None),
        "corrected_baselines": corrected_baselines,
        "adoption_flags": adoption_flags,
        "notes": [
            "This audit does NOT rewrite historical records: the original "
            "(contaminated) unified-confirm record and the adoption decision "
            "stand as committed; this record supersedes them on pooled legs. "
            "Sealed legs were immune (byte-identical across walks) so "
            "sealed-decided verdicts are unaffected.",
            "Clean canonical 12-pool C0 = 0.6312/0.6923 pooled "
            "(sealed 0.6233/0.7095) — the 0.6201/0.7069 'production C0' "
            "cited by the adoption decision and xgb-reg records was a "
            "contaminated later-walk measurement.",
        ],
    }
    DATA_DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DELIVERY_DIR / f"nfl_blend_reauth_audit_{frame_sha}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())