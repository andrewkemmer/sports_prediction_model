# NFL Production Backend

Clean, unified, production-grade NFL prediction backend. One authoritative
pipeline, point-in-time features, expanding walk-forward OOF, and three
first-class production models (moneyline, run-line/margin distribution,
totals distribution) serving the existing NFL frontend contract.

## Architecture

```
nfl-backend/backend/
  config.py         Central configuration: seeds, windows, grids, members
  manifest.py       Authoritative feature manifest (single source of truth)
  ingestion.py      nflverse pulls + parquet caching (outside the git tree)
  features.py       Point-in-time feature engine (Elo, trailing, EWM, venue)
  folds.py          Expanding 7-calendar-day walk-forward fold geometry
  moneyline.py      Binary ML ensemble + adaptive weights + Platt calibration
  distributions.py  Joint score distribution (margin PMF + total PMF)
  evaluation.py     OOF metrics: AUC/LogLoss/Brier/ECE + distributional
  serving.py        Frontend-contract artifact writers
  qb_enrichment.py  QB serving fields (display only, never fabricated)
  monitoring.py     Drift/coverage/ensemble/rolling-Brier monitor record
  master_pipeline.py  THE single production entry point
  test_production.py  Targeted test suite (imports/leakage/folds/contracts)
```

## Methodology

- **Historical eligibility** — 2018 is warmup only (trailing priors); the
  OOF population is every settled regular-season game from 2019 onward,
  including settled 2026 games. No preseason, no postseason, no cutoffs.
- **Walk-forward OOF** — expanding, chronological, non-overlapping
  7-calendar-day validation windows keyed on game dates (never week IDs);
  training is strictly prior; the final partial window is retained.
- **Leakage prevention** — Elo updates only after a game settles; every
  trailing/rolling/EWM statistic uses a per-team `shift(1)`; venue facts are
  static pre-game attributes. Asserted structurally and tested.
- **Market independence** — no sportsbook data anywhere: no spreads,
  totals, moneylines, implied probabilities, or edges enter any model.
- **Moneyline** — XGBoost / LightGBM / Logistic / Random Forest / MLP
  ensemble; adaptive weights derived from pooled OOF AUC; Platt calibration
  fit only on OOF predictions; final full-history refit for serving.
- **Run line** — joint score distribution: per-side mu regressions, margin
  PMF (discrete normal, sigma calibrated on pooled OOF residuals), fair
  spread, derived ML, and coherent cover/push/away probabilities for every
  integer spread −14…+14 plus ±0.5 half-stops.
- **Totals** — same joint distribution's total PMF: fair total and coherent
  over/push/under probabilities for every integer total 24…66.
- **Determinism** — explicit seeds on NumPy/sklearn/XGBoost/LightGBM/RF/MLP.

## Running

```bash
cd nfl-backend/backend
python3 test_production.py      # targeted tests (no network)
python3 master_pipeline.py      # full production run (pulls nflverse)
python3 master_pipeline.py --skip-pull   # cached pulls only
```

## Artifacts (nfl-backend/data_delivery/, generated per run)

| Family | Purpose |
|---|---|
| `nfl_moneyline_v1_<date>.json` | Today's Games card contract (`games[]`) |
| `nfl_run_engine_markets_<date>.csv` + `.meta.json` | mu quartet, fair lines, full spread/total grids, OOF + slate rows |
| `nfl_qb_matchup_<date>.json` | QB enrichment contract |
| `nfl_calibration_<date>.json` | Calibration page metrics/daily record |
| `nfl_predictions_history_<date>.csv` | Decided-game predicted-vs-actual |
| `nfl_power_rankings_<date>.csv` | Elo power rankings board |
| `nfl_model_monitor_<date>.json` | Monitoring record (drift/coverage/ensemble) |
| `nfl_feature_v1_<date>.json` | Feature manifest + coverage + fold geometry |
| `nfl_oof_moneyline.csv` / `nfl_oof_distribution.csv` | OOF stores |
| `models/nfl_ensemble_latest.joblib` | Persisted production model bundle |

The frontend is never modified by this backend; it consumes these artifact
families through its existing readers.

## Production / research separation

The production pipeline imports only the modules above. If every research
or legacy file were deleted, `master_pipeline.py` still functions from
scratch. The retired sealed-2025 holdout, candidate/incumbent gates,
tolerance verdicts, adoption decisions, wide-pool gates, week-ID folds,
Phase-3b market backfill, and separate 2026 regime no longer exist anywhere
in production.
