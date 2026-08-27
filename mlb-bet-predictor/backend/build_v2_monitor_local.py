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
from sklearn.metrics import roc_auc_score

from pipeline import DATA_DELIVERY_DIR, _run_engine_monitor_json
from run_engine import compute_winner_cards

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
    out = _run_engine_monitor_json(
        block,
        date_str,
        bool(v1.get("markets_persisted")),
        v1.get("markets_persist_error"),
    )
    data = json.loads(out.read_text())
    print(f"wrote: {out}  schema={data['schema']}")
    for card, series in data["rolling"].items():
        pts = ", ".join(f"{p['date']}(n={p['n']})" for p in series)
        print(f"  rolling {card}: {pts}")


if __name__ == "__main__":
    main()
