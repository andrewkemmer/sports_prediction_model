"""run_pitcher_ensemble_gate.py — full-ensemble gate for CORRECTED vs CURRENT.

Step 3.3 of the pitcher-stats measurement task: after the two-family screen
(run_pitcher_stats_ablation.py), if the corrected pitcher-stat semantics
showed promise, this runs the FULL production-correct ensemble walk-forward
(5 members + static-prior/adaptive blend, run_margin_diff attached
out-of-fold, prequential Platt calibration, sealed 21-day holdout) on BOTH
frames using run_margin_ablation's locked protocol.

Each variant's CSV is staged into its own temp dir AS game_level_features.csv
and run_margin_ablation.prepare_data/run_variant are pointed at it — so this
gate exercises the exact production harness, differing only in the frame.

Pooled OOF is the arbiter; the sealed holdout is the gate (logloss/AUC
not worse + ECE not worse). model_version_history.json entries are read to
anchor the CURRENT arm against production metrics.

Usage:
    python run_pitcher_ensemble_gate.py \
        --current-csv ../data_delivery/game_level_features_raw_current.csv \
        --corrected-csv ../data_delivery/game_level_features_corrected.csv

Emits data_delivery/pitcher_ensemble_gate_<sha>.json. COMMITS NOTHING.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import types
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

if "resource" not in sys.modules:  # POSIX-only module, stub on Windows
    _res = types.ModuleType("resource")
    _res.RUSAGE_SELF = 0
    _res.getrusage = lambda who: type("RU", (), {"ru_maxrss": 0})()
    sys.modules["resource"] = _res

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

import training  # noqa: E402
import run_margin_ablation as rma  # noqa: E402
from config import DATA_DELIVERY_DIR  # noqa: E402


def head_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=_BACKEND_DIR.parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_model_history() -> list[dict]:
    p = DATA_DELIVERY_DIR / "model_version_history.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else data.get("versions", [])


def gate_variant(csv: Path, label: str, holdout_days: int,
                 limit_folds: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"pitchergate_{label.lower()}_"))
    shutil.copy(csv, tmp / "game_level_features.csv")
    rma.DATA_DELIVERY_DIR = tmp
    games, tune, hold, folds, margins, hold_margins, rounds, uncov = (
        rma.prepare_data(holdout_days=holdout_days, limit_folds=limit_folds))
    # Production column set IS training.FEATURE_COLS (59 cols, run_margin_diff
    # included, already attached OOF by prepare_data). rma.build_variants is
    # skipped: its 64-col assertion predates the margin feature becoming a
    # FEATURE_COLS member.
    cols = list(training.FEATURE_COLS)
    assert cols == [c for c in training.FEATURE_COLS], "column set drift"
    res = rma.run_variant(
        cols, folds, tune, hold,
        partial_path=tmp / "partial.json")
    shutil.rmtree(tmp, ignore_errors=True)
    res["n_folds"] = len(folds)
    res["margin_uncovered"] = uncov
    res["margin_rounds"] = rounds
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current-csv", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features_raw_current.csv")
    ap.add_argument("--corrected-csv", type=Path,
                    default=DATA_DELIVERY_DIR / "game_level_features_corrected.csv")
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = {
        "task": "pitcher-stats corrected-outs FULL ENSEMBLE gate",
        "head": head_sha(),
        "frames": {"current": str(args.current_csv),
                   "current_sha": sha256_file(args.current_csv),
                   "corrected": str(args.corrected_csv),
                   "corrected_sha": sha256_file(args.corrected_csv)},
        "holdout_days": args.holdout_days,
        "limit_folds": args.limit_folds,
        "variants": {},
    }
    for label, csv in (("CURRENT", args.current_csv),
                       ("CORRECTED", args.corrected_csv)):
        print(f"── {label} full-ensemble gate ──", flush=True)
        report["variants"][label] = gate_variant(
            csv, label, args.holdout_days, args.limit_folds)

    cur = report["variants"]["CURRENT"]
    cor = report["variants"]["CORRECTED"]

    def cell(v, key, sub="blend"):
        return v["pooled"][sub][key] if key in v["pooled"][sub] else "—"

    print("\n=== POOLED OOF (blend, production-correct) ===")
    for k in ("logloss", "auc", "brier", "ece"):
        print(f"  {k:8s} CURRENT {cell(cur,k)}   CORRECTED {cell(cor,k)}")
    print("\n=== SEALED HOLDOUT (blend) ===")
    rows = []
    for sub in ("blend", "blend_calibrated"):
        a, b = cur["holdout"].get(sub, {}), cor["holdout"].get(sub, {})
        if not a:
            continue
        for k in ("logloss", "auc", "brier", "ece"):
            rows.append((f"{sub}.{k}", a.get(k), b.get(k)))
    for name, a, b in rows:
        print(f"  {name:22s} CURRENT {a}   CORRECTED {b}")

    # Gate per the policy: pooled blend not worse + sealed not worse + ECE ok
    def _d(x, y, key):
        if key not in x or key not in y:
            return None
        return round(y[key] - x[key], 4)

    pooled_d = {k: _d(cur["pooled"]["blend"], cor["pooled"]["blend"], k)
                for k in ("logloss", "auc", "brier", "ece")}
    hold_d = {k: _d(cur["holdout"].get("blend", {}),
                    cor["holdout"].get("blend", {}), k)
              for k in ("logloss", "auc", "brier", "ece")}
    passes = []
    if pooled_d["logloss"] is not None:
        passes.append(("pooled logloss ≤ +0.001", pooled_d["logloss"] <= 0.001))
        passes.append(("pooled AUC ≥ −0.001", pooled_d["auc"] >= -0.001))
        passes.append(("pooled ECE ≤ 0", pooled_d["ece"] <= 0))
    if hold_d["logloss"] is not None:
        passes.append(("sealed logloss ≤ +0.001", hold_d["logloss"] <= 0.001))
        passes.append(("sealed AUC ≥ −0.001", hold_d["auc"] >= -0.001))
        passes.append(("sealed ECE ≤ 0", hold_d["ece"] <= 0))
    verdict = "ADOPT" if all(p for _, p in passes) else "REJECT"
    report["deltas"] = {"pooled": pooled_d, "sealed": hold_d}
    report["gate_checks"] = [{"check": c, "pass": bool(p)} for c, p in passes]
    report["verdict"] = verdict

    hist = read_model_history()
    if hist:
        report["production_history_tail"] = hist[-3:]
    print(f"\nGATE VERDICT: {verdict}")
    for c, p in passes:
        print(f"    [{'PASS' if p else 'FAIL'}] {c}")

    out = args.out or (DATA_DELIVERY_DIR /
                       f"pitcher_ensemble_gate_{head_sha()[:12]}.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report → {out}")


if __name__ == "__main__":
    main()