"""Build the v2 run-engine monitor JSON locally from the real saved artifacts.

The on-disk monitor JSONs predate the winner-card rework (v1, per-line cards).
This bridge derives the three v2 WINNER cards from the run's real artifacts —
per-game probabilities (run_engine_oof_YYYYMMDD.csv) + per-game line
assignment/outcomes (run_engine_markets_YYYYMMDD.csv) — reuses the v1 file's
distributional-fit block (format-agnostic), and writes the v2 monitor JSON
through the SAME writer the pipeline uses (so the rolling fold over prior v1
monitor files exercises the production v1->v2 mapping).

Usage:
    PYTHONPATH=backend:. python3 backend/build_v2_monitor_local.py [YYYYMMDD]

One-off bridge until the next pipeline run emits the genuine v2 artifact;
re-runnable and idempotent (overwrites data_delivery/run_engine_monitor_<d>.json).

Note: the persisted OOF/markets CSVs do not carry fold_idx, so this local
build's predicted_mean/ece_calibrated are raw-based (prequential calibration
requires the in-memory fold assignment; the next genuine pipeline run emits
the calibrated numbers). n, win rates, pushes, and AUC are exact.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from config import RETRAIN_CADENCE_DAYS
from explainability import (compute_run_engine_feature_coverage,
                            compute_run_engine_feature_drift)
from pipeline import DATA_DELIVERY_DIR, _run_engine_monitor_json
from run_engine import brier_score, compute_winner_cards, ece_score
from training import walk_forward_splits

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# Fixed-reference lines per winner card (never a mixed-line rank): the AUC
# each card shows is roc_auc over the SAME OOF y/p vectors score_at scores.
# Keys are the market_metrics ref names compute_winner_cards looks up
# (WINNER_CARD_AUC_REF): over_8_5 / home_cover_1_5 / derived_moneyline.
_REFERENCE_LINES = {
    "over_8_5": ("p_over_8_5",
                  lambda df: (df["home_score"] + df["away_score"] >= 9)),
    "home_cover_1_5": ("p_home_cover_1_5",
                       lambda df: (df["home_score"] - df["away_score"] >= 2)),
    "derived_moneyline": ("p_home_win_derived",
                          lambda df: df["home_score"] > df["away_score"]),
}


def _reference_aucs(markets: pd.DataFrame) -> dict:
    """Fixed-reference-line AUC per market_metrics ref from the real OOF
    rows (mirrors score_at: same y/p vectors; single-class y -> None, never
    a crash)."""
    df = markets[markets.get("kind") == "oof"].copy()
    out: dict[str, dict] = {}
    for ref, (pcol, yfn) in _REFERENCE_LINES.items():
        if pcol not in df.columns:
            out[ref] = {}
            continue
        p = df[pcol].to_numpy(float)
        y = yfn(df).to_numpy(float)
        ok = np.isfinite(p)
        p, y = p[ok], y[ok]
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = None
        if auc is None or not np.isfinite(auc):
            out[ref] = {"auc": None}
        else:
            out[ref] = {"auc": round(auc, 5)}
    return out


def _market_metrics(markets: pd.DataFrame) -> dict:
    """Per-line engine OOF metrics for the model card, scored on the SAME
    OOF y/p vectors score_at uses (the markets CSV's 'oof' rows carry the
    per-game line probabilities). No fold_idx on disk -> calibrated == raw
    here; the genuine pipeline run emits the prequentially-calibrated rows.
    """
    df = markets[markets.get("kind") == "oof"].copy()
    total = df["home_score"] + df["away_score"]
    margin = df["home_score"] - df["away_score"]
    lines = [
        ("over_7_5", "p_over_7_5", (total >= 8).astype(float)),
        ("over_8_5", "p_over_8_5", (total >= 9).astype(float)),
        ("over_9_5", "p_over_9_5", (total >= 10).astype(float)),
        ("home_cover_1_5", "p_home_cover_1_5", (margin >= 2).astype(float)),
        ("home_cover_2_5", "p_home_cover_2_5", (margin >= 3).astype(float)),
        ("derived_moneyline", "p_home_win_derived",
         (df["home_score"] > df["away_score"]).astype(float)),
    ]
    out: dict[str, dict] = {}
    for name, pcol, y in lines:
        if pcol not in df.columns:
            out[name] = {}
            continue
        p = np.clip(df[pcol].to_numpy(float), 1e-6, 1 - 1e-6)
        yv = y.to_numpy(float)
        base = float(yv.mean())
        try:
            auc = float(roc_auc_score(yv, p))
        except ValueError:
            auc = None
        row = {
            "engine_logloss": round(float(log_loss(yv, p)), 5),
            "engine_brier": round(float(brier_score(yv, p)), 5),
            "engine_ece_raw": ece_score(yv, p),
            "engine_logloss_calibrated": round(float(log_loss(yv, p)), 5),
            "engine_ece_calibrated": ece_score(yv, p),
            "auc": (round(auc, 5) if auc is not None and np.isfinite(auc)
                    else None),
            "baseline_rate": round(base, 4),
            "baseline_logloss": round(float(log_loss(
                yv, np.full(len(yv), base))), 5),
            "n": int(len(yv)),
        }
        out[name] = row
    return out


def _phase1_geometry(date_str: str) -> dict:
    """Walk-forward geometry for the model card: fold count via the SAME
    deterministic walk_forward_splits the run engine uses on the canonical
    decided frame (run_engine.run_oof), plus per-side dispersion. Median
    rounds are NOT persisted anywhere on disk -> absent (renderer shows '--'
    until a genuine pipeline run emits phase1.final_fit_rounds)."""
    gl_path = DATA_DELIVERY_DIR / "game_level_features.csv"
    if not gl_path.exists():
        return {"n_folds": None, "n_games": None}
    gl = pd.read_csv(gl_path)
    decided = gl[gl["home_win"].notna()].sort_values("game_date")
    decided = decided.reset_index(drop=True)
    folds = walk_forward_splits(
        decided, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    return {"n_folds": int(len(folds)), "n_games": int(len(decided))}


def _build_drift_coverage(date_str: str) -> tuple[Path, Path]:
    """Build the run-engine drift + coverage CSVs on the SAME baseline /
    current windows the pipeline drift step uses (last 7 days vs an adjacent
    season-local window of ~3x, min 250) — the acceptance artifacts for the
    run-line monitor's drift/coverage sections."""
    gl = pd.read_csv(DATA_DELIVERY_DIR / "game_level_features.csv")
    decided = gl[gl["home_win"].notna()].sort_values("game_date")
    gd = pd.to_datetime(decided["game_date"])
    cutoff = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    cutoff -= pd.Timedelta(days=7)
    current = decided[gd >= cutoff]
    prior = decided[gd < cutoff]
    baseline = (prior.tail(max(3 * len(current), 250))
                if not prior.empty else prior)
    drift = compute_run_engine_feature_drift(baseline, current, date_str)
    cov = compute_run_engine_feature_coverage(baseline, current, date_str)
    d_path = DATA_DELIVERY_DIR / f"run_engine_feature_drift_{date_str}.csv"
    c_path = DATA_DELIVERY_DIR / f"run_engine_feature_coverage_{date_str}.csv"
    print(f"drift: {len(drift)} features -> {d_path.name}")
    print(f"coverage: {len(cov)} feature-window pairs -> {c_path.name}")
    return d_path, c_path


def _block_from_v1_monitor(v1: dict, winner_cards: dict) -> dict:
    """Reconstruct the raw monitor block shape _run_engine_monitor_json
    expects, carrying the v1 file's distributional-fit content (identical in
    v1 and v2) plus the freshly-derived winner cards."""
    fit = v1.get("fit") or {}
    return {
        "winner_cards": winner_cards,
        "alpha_home": fit.get("alpha_home") or {},
        "alpha_away": fit.get("alpha_away") or {},
        "phase1": {"dispersion_ratio": (fit.get("dispersion_chi2_per_df")
                                        or {"home": None, "away": None})},
        "fit_check_alpha_lambda": fit.get("fit_tables") or {},
        "variance_check": fit.get("variance_check") or {},
        "mc_meta": fit.get("mc_meta") or {},
        "line_grid": fit.get("line_grid") or [],
        "holdout_gate": fit.get("holdout_gate") or {},
    }


def main() -> None:
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime(
        "%Y%m%d")
    oof_path = DATA_DELIVERY_DIR / f"run_engine_oof_{date_str}.csv"
    markets_path = DATA_DELIVERY_DIR / f"run_engine_markets_{date_str}.csv"
    v1_path = DATA_DELIVERY_DIR / f"run_engine_monitor_{date_str}.json"
    # The v1 SOURCE of truth is the preserved fixture — this builder writes
    # the v2 JSON over the same data_delivery path, so re-runs must never
    # re-read their own (v2) output as the v1 input.
    fixture = FIXTURES_DIR / f"run_engine_monitor_v1_{date_str}.json"
    v1_path = fixture if fixture.exists() else v1_path
    for p in (oof_path, markets_path, v1_path):
        if not p.exists():
            raise SystemExit(f"missing artifact: {p}")
    print(f"load: {oof_path.name} / {markets_path.name} / {v1_path.name}")

    markets = pd.read_csv(markets_path)
    aucs = _reference_aucs(markets)
    winner_cards = compute_winner_cards(markets, market_metrics=aucs)
    n_cards = len(winner_cards)
    print(f"winner cards derived: {sorted(winner_cards)} ({n_cards})")
    for card, c in sorted(winner_cards.items()):
        print(f"  {card}: n={c.get('n')} win_rate={c.get('win_rate'):.4f} "
              f"pred={c.get('predicted_mean'):.4f} auc={c.get('auc')}")

    v1 = json.loads(v1_path.read_text())
    block = _block_from_v1_monitor(v1, winner_cards)
    block["market_metrics"] = _market_metrics(markets)
    # Merge — never overwrite: the v1-derived phase1 already carries
    # dispersion_ratio (which feeds fit.dispersion_chi2_per_df).
    block["phase1"] = {**(block.get("phase1") or {}),
                       **_phase1_geometry(date_str)}
    d_path, c_path = _build_drift_coverage(date_str)
    out = _run_engine_monitor_json(
        block,
        date_str,
        bool(v1.get("markets_persisted")),
        v1.get("markets_persist_error"),
    )
    data = json.loads(out.read_text())
    print(f"wrote: {out}  schema={data['schema']}")
    print("market_metrics lines:", sorted(data.get("market_metrics") or {}))
    print("phase1:", data.get("phase1"))
    print("artifacts:", d_path.name, c_path.name)
    for card, series in data["rolling"].items():
        pts = ", ".join(f"{p['date']}(n={p['n']})" for p in series)
        print(f"  rolling {card}: {pts}")


if __name__ == "__main__":
    main()
