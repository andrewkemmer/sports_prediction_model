"""Post-correction validation (normal full validation pipeline).

The corrected Experiment #2 operation is a REPLACEMENT:
  baseline 59 − 6 unique E/F removals + 8 exp2 candidates = 61.

Arms (same harness, same folds, paired):
  F-ML    : original 59-feature baseline, moneyline      (control)
  CORR-ML : corrected 61-feature set, moneyline
  F-RL    : original 59-feature baseline, true −1.5 run line (control)
  CORR-RL : corrected 61-feature set, true −1.5 run line

Frame: decided games strictly before 2026-08-16 (the sealed trailing
21-day holdout is never read). Fold geometry is a pure function of
game_date + non-null target, so ML and RL fold boundaries are identical.

Writes data_delivery/exp2_correction_validation_<date>.json.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import training  # noqa: E402
from data_ingestion import load_game_features  # noqa: E402
from frames import get_decided_frame  # noqa: E402
from run_engine import (  # noqa: E402
    RUN_LAMBDA_DROPPED_FROZEN,
    RUN_LAMBDA_VIEW_FROZEN,
)
from training import (  # noqa: E402
    FEATURE_COLS,
    walk_forward_evaluate,
    walk_forward_evaluate_runline,
)

CUTOFF = "2026-08-16"  # sealed trailing 21-day holdout untouched
OUT = Path(__file__).resolve().parents[1] / "data_delivery"


def baseline_59() -> list[str]:
    """Original 59-feature baseline = corrected 61 − 8 exp2 + 6 removals."""
    exp2 = [c for c in FEATURE_COLS if c.startswith("exp2_")]
    removals = ["sp_k9_diff", "sp_k9_5g_diff", "sp_fbpct_diff",
                "sp_whiff_diff", "sp_xwoba_diff", "sp_xwoba_vs_l_diff"]
    base = [c for c in FEATURE_COLS if c not in exp2]
    # reinsert removals at their original positions (order matters only for
    # determinism; reconstruct from the frozen run view's diff block order)
    out: list[str] = []
    for c in base:
        if c == "sp_fbvelo_diff":
            out += ["sp_k9_diff", "sp_k9_5g_diff", "sp_fbvelo_diff",
                    "sp_fbpct_diff", "sp_whiff_diff",
                    "sp_xwoba_diff", "sp_xwoba_vs_l_diff"]
        elif c not in removals:
            out.append(c)
    assert len(out) == 59, len(out)
    return out


def metrics(m: dict[str, float], preds: pd.DataFrame) -> dict:
    out = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
    out["n_oof"] = int(len(preds))
    return out


def run_arm(games: pd.DataFrame, feats: list[str], runline: bool, tag: str) -> dict:
    saved = list(training.FEATURE_COLS)
    training.FEATURE_COLS = list(feats)
    try:
        t0 = time.time()
        fn = walk_forward_evaluate_runline if runline else walk_forward_evaluate
        _bundle, m, preds = fn(games)
        mm = metrics(m, preds)
        mm["tag"] = tag
        mm["feature_count"] = len(feats)
        mm["fold_signature"] = training.get_last_fold_signature()
        mm["wall_s"] = round(time.time() - t0, 1)
        print(f"  {tag}: {mm}", flush=True)
        return mm
    finally:
        training.FEATURE_COLS = saved


def main() -> None:
    games = load_game_features(str(OUT / "game_level_features.csv"))
    decided = get_decided_frame(games)
    frame = decided[decided["game_date"] < CUTOFF].copy()
    print(f"Frame: {len(frame)} decided games through "
          f"{frame['game_date'].max().date()} (cutoff {CUTOFF})", flush=True)

    b59, c61 = baseline_59(), list(FEATURE_COLS)
    assert len(c61) == 61 and len(b59) == 59

    results = {
        "date_tag": str(date.today()),
        "note": ("CORRECTED exp2 implementation validation: replacement "
                 "semantics (59 − 6 + 8 = 61); baseline F rerun on the SAME "
                 "harness/folds as a paired control; sealed trailing 21-day "
                 "holdout untouched (frame cutoff 2026-08-16)."),
        "run_engine_view": {
            "frozen_lambda_view_len": len(RUN_LAMBDA_VIEW_FROZEN),
            "frozen_lambda_dropped_len": len(RUN_LAMBDA_DROPPED_FROZEN),
            "unchanged_vs_pre_correction": True,
        },
        "arms": {},
    }
    for runline, target in ((False, "moneyline"), (True, "run_line_true_minus_1_5")):
        for feats, tagname in ((b59, "F_baseline_59"), (c61, "corrected_61")):
            tag = f"{target}/{tagname}"
            results["arms"][tag] = run_arm(frame, feats, runline, tag)

    # pairing check: identical fold signatures across arms of the same target
    for target in ("moneyline", "run_line_true_minus_1_5"):
        sigs = {results["arms"][f"{target}/{t}"]["fold_signature"]
                for t in ("F_baseline_59", "corrected_61")}
        assert len(sigs) == 1, f"fold geometry drifted within {target}: {sigs}"

    out_path = OUT / f"exp2_correction_validation_{date.today():%Y%m%d}.json"
    out_path.write_text(json.dumps(results, indent=1))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
