"""Build the TRUE 284-game sealed-holdout replay CSV for the calibration ablation.

predictions_history_<date>.csv only carries walk-forward OOF games that were
decided when the run's Step-3 frame was built (max 2026-08-23). The canonical
moneyline frame (game_level_features.csv) holds 4,466 decided games through
2026-08-25, so the sealed window [2026-08-05 .. 2026-08-25] contains 284 games
-- 259 in the saved history plus 25 (2026-08-24: 10, 2026-08-25: 15) that never
received OOF predictions.

This script recomputes honest OOF predictions for those 25 games with the
SAME machinery the run used:

  * deterministic walk_forward_splits on the canonical 4,466-row frame
    (51 folds; the 25 games sit in fold 50, val [2026-08-22 .. 2026-08-25],
    train strictly < 2026-08-22);
  * run_margin_diff attached from the run's saved run-engine OOF artifact
    (run_engine_oof_<date>.csv -- per-game expected runs are the OOF lambdas,
    so the diff is leakage-free per game);
  * train_moneyline_ensemble(train < 08-22, val = the 25 games) and
    ensemble_predict for the raw OOF blend, exactly like a walk-forward fold.

Sanity check: the same fold-50 model predicts the 30 games on 08-22/08-23 and
is compared against the run's stored probabilities (same train cutoff; only
the frame's row set differs).

Writes <date>-stamped replay CSV (data_delivery/sealed_holdout_replay_<date>.csv)
with columns game_date, home_win, home_win_prob_model (+ provenance columns
source, game_id, game_pk, fold_idx).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import (  # noqa: E402
    MARGIN_COL,
    MIN_VAL_FOLD_GAMES,
    RETRAIN_CADENCE_DAYS,
    ensemble_predict,
    train_moneyline_ensemble,
    walk_forward_splits,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_delivery"

HOLDOUT_START = pd.Timestamp("2026-08-05")
HOLDOUT_END = pd.Timestamp("2026-08-25")


def attach_margins(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Attach run_margin_diff from the saved run-engine OOF (λ_home − λ_away)."""
    o = oof[["game_pk", "home_expected_runs", "away_expected_runs"]].copy()
    o["game_pk"] = pd.to_numeric(o["game_pk"], errors="coerce")
    o[MARGIN_COL] = o["home_expected_runs"] - o["away_expected_runs"]
    out = frame.copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
    out = out.drop(columns=[MARGIN_COL] if MARGIN_COL in out.columns else [])
    out = out.merge(o[["game_pk", MARGIN_COL]], on="game_pk", how="left")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Pipeline target date YYYY-MM-DD (default today)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    target = args.date or date.today().isoformat()
    stamp = target.replace("-", "")

    features = pd.read_csv(DATA / "game_level_features.csv")
    history = pd.read_csv(DATA / f"predictions_history_{stamp}.csv")
    reoof = pd.read_csv(DATA / f"run_engine_oof_{stamp}.csv")

    features["game_date"] = pd.to_datetime(features["game_date"])
    history["game_date"] = pd.to_datetime(history["game_date"])

    # --- canonical fold geometry ---
    splits = walk_forward_splits(features, RETRAIN_CADENCE_DAYS)
    fold50 = [s for s in splits if s["fold_idx"] == 50]
    if not fold50:
        raise SystemExit("fold 50 [2026-08-22..2026-08-25] not found in canonical splits")
    fold = fold50[0]
    train = fold["train_games"].copy()
    val = fold["val_games"].copy()

    # --- the 25 missing games ---
    missing = val[val["game_date"].between("2026-08-24", "2026-08-25")].copy()
    missing = missing[missing["game_pk"].isin(set(reoof["game_pk"]))]
    if len(missing) != 25:
        raise SystemExit(f"expected 25 missing games, got {len(missing)}")

    # --- attach margins (leakage-free: OOF expected runs per game) ---
    train_m = attach_margins(train, reoof)
    val_m = attach_margins(missing, reoof)

    # --- train fold-50 model exactly like a walk-forward fold ---
    models, _ = train_moneyline_ensemble(train_m, val_m)
    prob, _member, _wts = ensemble_predict(models, val_m)
    val_m = val_m.copy()
    val_m["home_win_prob_model"] = np.asarray(prob, dtype=float)

    # --- sanity check against stored predictions on 08-22/08-23 ---
    sanity = attach_margins(val[val["game_date"].between("2026-08-22", "2026-08-23")], reoof)
    sprob, _, _ = ensemble_predict(models, sanity)
    hist_map = {
        str(r["game_id"]): r["home_win_prob_model"]
        for _, r in history.iterrows()
        if pd.notna(r.get("game_id"))
    }
    diffs = []
    for i, (_, r) in enumerate(sanity.iterrows()):
        stored = hist_map.get(str(r["game_id"]))
        if stored is not None:
            diffs.append(abs(float(stored) - float(sprob[i])))
    sanity_note = {
        "games": len(diffs),
        "max_abs_diff": float(np.max(diffs)) if diffs else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
    }

    # --- assemble replay frame: tuning + 259 stored holdout + 25 recomputed ---
    h = history.copy()
    h["source"] = "predictions_history"
    h["fold_idx"] = np.nan
    h["game_pk"] = pd.to_numeric(h.get("game_pk"), errors="coerce")
    replay = pd.DataFrame({
        "game_date": h["game_date"],
        "game_id": h.get("game_id"),
        "game_pk": h.get("game_pk"),
        "home_win": pd.to_numeric(h["home_win"], errors="coerce"),
        "home_win_prob_model": pd.to_numeric(h["home_win_prob_model"], errors="coerce"),
        "source": "predictions_history",
        "fold_idx": np.nan,
    })
    new_rows = pd.DataFrame({
        "game_date": val_m["game_date"],
        "game_id": val_m.get("game_id"),
        "game_pk": val_m["game_pk"],
        "home_win": pd.to_numeric(val_m["home_win"], errors="coerce"),
        "home_win_prob_model": val_m["home_win_prob_model"],
        "source": "recomputed_fold50",
        "fold_idx": 50,
    })
    replay = pd.concat([replay, new_rows], ignore_index=True)
    replay = replay.dropna(subset=["home_win", "home_win_prob_model"]).copy()

    hold = replay[replay["game_date"].between(HOLDOUT_START, HOLDOUT_END)]
    tuning = replay[replay["game_date"] < HOLDOUT_START]
    print(f"tuning games: {len(tuning)}")
    print(f"sealed holdout games: {len(hold)}  "
          f"(history: {len(hold[hold['source'] == 'predictions_history'])}, "
          f"recomputed fold-50: {len(hold[hold['source'] == 'recomputed_fold50'])})")

    out = args.out or (DATA / f"sealed_holdout_replay_{stamp}.csv")
    replay.to_csv(out, index=False)
    meta = {
        "target_date": target,
        "holdout_n": int(len(hold)),
        "holdout_history_n": int((hold["source"] == "predictions_history").sum()),
        "holdout_recomputed_n": int((hold["source"] == "recomputed_fold50").sum()),
        "tuning_n": int(len(tuning)),
        "fold50": {"val_start": str(fold["val_start"].date()),
                   "val_end": str(fold["val_end"].date()),
                   "train_cutoff": str(train["game_date"].max().date()),
                   "train_n": int(len(train_m))},
        "sanity_vs_stored_0822_0823": sanity_note,
        "sources": {
            "259": "predictions_history_%s.csv (run walk-forward OOF)" % stamp,
            "25": "recomputed: fold-50 model trained strictly < 2026-08-22, "
                  "run_margin_diff from run_engine_oof_%s.csv" % stamp,
        },
    }
    (DATA / f"sealed_holdout_replay_{stamp}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"replay CSV -> {out}")
    print(f"meta -> sealed_holdout_replay_{stamp}.meta.json")


if __name__ == "__main__":
    main()
