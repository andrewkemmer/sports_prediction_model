"""Model-tuning admission gate for the moneyline ensemble.

Executable checklist for docs/model_tuning_policy.md. Uniformly applies the
two gates to ANY member (xgboost / lightgbm / logistic / randomforest / mlp):

  MEMBER GATE — candidate must beat the current member on ALL SIX conditions
  (pooled OOF and sealed holdout x logloss/AUC/ECE), on identical
  folds/seeds. Strict: ll <, AUC >, ECE <= on both views.

  BLEND GATE — production-correct, adaptive weights, multi-window. ADOPT
  requires ALL of:
    pooled blend AUC    >= current - 0.001
    pooled blend logloss <= current + 0.001
    pooled blend ECE    <= current            (ECE is a priority: strict)
    member ECE improves (candidate < current, pooled view)
  Sealed windows are CONTEXT ONLY (pooled OOF is the arbiter per repo
  convention); a 0/N window record is counter-evidence, not a rejection.

Decision: ADOPT iff both gates pass. Any failure -> REJECT (recorded; config
untouched). No per-model exceptions; adopting over a gate failure requires
amending docs/model_tuning_policy.md first.

The script reads evidence JSONs (schema "tuning-gate-evidence/v1"; see
data_delivery/rf_tuning_gate_20260831.json) rather than trusting a recorded
verdict — it RE-DERIVES the verdict from the numbers, so every candidate is
evaluated by the same code path.

Usage:
    python run_tuning_gate.py                          # RF precedent (default)
    python run_tuning_gate.py --evidence path.json     # any candidate
    python run_tuning_gate.py --check-config           # verify config matches
                                                        # the adopted candidate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

# Policy thresholds (docs/model_tuning_policy.md sections 3-4).
BLEND_AUC_TOL = 0.001        # candidate AUC may not fall more than this
BLEND_LOGLOSS_TOL = 0.001    # candidate logloss may not rise more than this
# ECE is a stated priority: strict (candidate <= current) on pooled blend,
# and the member's ECE must IMPROVE (strictly <) for adoption.

CONFIG_DIR = _BACKEND_DIR.parent / "data_delivery"
DEFAULT_EVIDENCE = CONFIG_DIR / "rf_tuning_gate_20260831.json"


def _num(d: dict, *keys) -> float | None:
    cur: dict | float | None = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return float(cur)


def member_gate(ev: dict) -> tuple[bool, list[str]]:
    """Six strict conditions. Returns (passed, checklist rows)."""
    mg = ev["member_gate"]
    rows: list[str] = []
    ok = True
    checks = [
        ("pooled_oof", "logloss", "<", lambda c, k: c < k),
        ("pooled_oof", "auc", ">", lambda c, k: c > k),
        ("pooled_oof", "ece", "<=", lambda c, k: c <= k),
        ("sealed_21d", "logloss", "<", lambda c, k: c < k),
        ("sealed_21d", "auc", ">", lambda c, k: c > k),
        ("sealed_21d", "ece", "<=", lambda c, k: c <= k),
    ]
    for view, metric, op, cond in checks:
        c = _num(mg[view]["current"], metric)
        k = _num(mg[view]["candidate"], metric)
        if c is None or k is None:
            rows.append(f"  {view}.{metric}: MISSING DATA")
            ok = False
            continue
        passed = cond(k, c)
        ok = ok and passed
        rows.append(
            f"  {view:<10} {metric:<8} current={c:.4f} candidate={k:.4f} "
            f"({op}) -> {'PASS' if passed else 'FAIL'}")
    return ok, rows


def blend_gate(ev: dict) -> tuple[bool, list[str], dict]:
    """Four ADOPT criteria + sealed-window context. Returns (passed, rows, deltas)."""
    bg = ev["blend_gate"]
    rows: list[str] = []
    ok = True
    c = bg["pooled_blend"]["current"]
    k = bg["pooled_blend"]["candidate"]
    d_auc = _num(k, "auc") - _num(c, "auc")
    d_ll = _num(k, "logloss") - _num(c, "logloss")
    d_ece = _num(k, "ece") - _num(c, "ece")
    m_ece_c = _num(ev["member_gate"]["pooled_oof"]["current"], "ece")
    m_ece_k = _num(ev["member_gate"]["pooled_oof"]["candidate"], "ece")

    criteria = [
        ("pooled blend AUC    >= current - 0.001",
         d_auc is not None and d_auc >= -BLEND_AUC_TOL),
        ("pooled blend logloss <= current + 0.001",
         d_ll is not None and d_ll <= BLEND_LOGLOSS_TOL),
        ("pooled blend ECE    <= current (strict)",
         d_ece is not None and d_ece <= 0.0),
        ("member ECE improves (candidate < current)",
         m_ece_c is not None and m_ece_k is not None and m_ece_k < m_ece_c),
    ]
    for label, passed in criteria:
        rows.append(f"  {label:<45} -> {'PASS' if passed else 'FAIL'}")
        ok = ok and passed

    wins = int(_num(bg["sealed_windows"], "wins") or 0)
    total = int(_num(bg["sealed_windows"], "total") or 0)
    rows.append(
        f"  sealed windows (context only): {wins}/{total} — "
        f"counter-evidence, NOT a rejection per policy")
    deltas = {"auc": d_auc, "logloss": d_ll, "ece": d_ece}
    return ok, rows, deltas


def verdict_for(evidence_path: Path) -> tuple[str, list[str]]:
    ev = json.loads(evidence_path.read_text())
    out: list[str] = []
    member_ok, mrows = member_gate(ev)
    blend_ok, brows, _deltas = blend_gate(ev)
    out.append(f"evidence     : {evidence_path}")
    out.append(f"candidate    : {ev.get('candidate', {}).get('label', '?')}")
    out.append("")
    out.append("MEMBER GATE (all six must pass):")
    out.extend(mrows)
    out.append(f"  => {'PASS' if member_ok else 'FAIL'}")
    out.append("")
    out.append("BLEND GATE (all four must pass; sealed = context):")
    out.extend(brows)
    out.append(f"  => {'PASS' if blend_ok else 'FAIL'}")
    out.append("")
    verdict = "ADOPT" if (member_ok and blend_ok) else "REJECT"
    if verdict == "ADOPT":
        out.append("GATE VERDICT: ADOPT — update config.*_PARAMS with a "
                   "provenance block citing docs/model_tuning_policy.md and "
                   "this evidence file.")
    else:
        failed = [r for r in mrows + brows if "FAIL" in r]
        out.append(f"GATE VERDICT: REJECT ({len(failed)} failed condition(s)) — "
                   "config untouched; record the evidence.")
    return verdict, out


def check_config(evidence_path: Path) -> list[str]:
    """Verify config.*_PARAMS matches the adopted candidate params."""
    import config
    ev = json.loads(evidence_path.read_text())
    cand = ev["candidate"]
    member = cand["member"]
    expected = dict(cand["params"])
    # member -> config attr map (config.RF_PARAMS, not RANDOMFOREST_PARAMS).
    attr = {"randomforest": "RF_PARAMS",
            "xgboost": "XGBOOST_PARAMS",
            "lightgbm": "LIGHTGBM_PARAMS",
            "logistic": "LOGISTIC_PARAMS",
            "mlp": "MLP_PARAMS"}.get(member)
    if attr is None:
        raise ValueError(f"unknown member '{member}'")
    actual = dict(getattr(config, attr))
    rows = ["CONFIG CHECK:"]
    # random_state / n_jobs are forced by the training path; compare the rest.
    mismatches = []
    for k, v in expected.items():
        if k in ("random_state", "n_jobs"):
            continue
        if actual.get(k) != v:
            mismatches.append(f"{k}: config={actual.get(k)} expected={v}")
    if mismatches:
        rows.append(f"  MISMATCH for {member}: {mismatches}")
    else:
        rows.append(f"  config.{attr} matches the adopted "
                    f"candidate ({', '.join(f'{k}={v}' for k, v in expected.items())})")
    # Construct-and-fit smoke: the params must be consumable end-to-end.
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    try:
        model = RandomForestClassifier(**actual)
        rng = np.random.RandomState(0)
        X = rng.rand(150, 61)
        y = (rng.rand(150) > 0.5).astype(int)
        model.fit(X, y)
        _ = model.predict_proba(X)
        rows.append("  construct-and-fit smoke: OK (RF consumes config params)")
    except Exception as e:  # pragma: no cover
        rows.append(f"  construct-and-fit smoke: FAILED ({e})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE,
                    help="tuning-gate-evidence/v1 JSON (default: the RF "
                         "2026-08-31 precedent)")
    ap.add_argument("--check-config", action="store_true",
                    help="also verify config.*_PARAMS == adopted candidate "
                         "params (construct-and-fit smoke)")
    args = ap.parse_args()

    verdict, out = verdict_for(args.evidence)
    print("\n".join(out))
    if args.check_config:
        print()
        print("\n".join(check_config(args.evidence)))
    sys.exit(0 if verdict == "ADOPT" else 1)


if __name__ == "__main__":
    main()
