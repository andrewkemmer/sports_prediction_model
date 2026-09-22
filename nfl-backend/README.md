# NFL Production Backend

Clean, unified, production-grade NFL prediction backend. One authoritative
pipeline, point-in-time features, expanding walk-forward OOF, and three
first-class production models (moneyline, run-line/margin distribution,
totals distribution) serving the existing NFL frontend contract.

## Architecture

```
nfl-backend/backend/
  config.py         The one MONEYLINE_FEATURE_COLS contract + seeds/windows/grids
  manifest.py       Documentation of that contract (validated name-for-name)
  ingestion.py      nflverse pulls + parquet caching (outside the git tree)
  features.py       Point-in-time feature engine (Elo, trailing, EWM, venue)
  folds.py          Expanding 7-calendar-day walk-forward fold geometry
  moneyline.py      Binary ML ensemble + adaptive weights + Platt calibration
  distributions.py  Joint score distribution (margin PMF + total PMF)
  evaluation.py     OOF metrics: AUC/LogLoss/Brier/ECE + distributional
  serving.py        Frontend-contract artifact writers
  qb_enrichment.py  QB serving fields (display only, never fabricated)
  monitoring.py     Drift/coverage/ensemble/rolling-Brier monitor record
  feature_selection.py  RFE sweep (record-only; adoption is an explicit step)
  feature_workbook.py   RFE decision workbook (.xlsx) writer
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
- **One feature contract (MLB parity)** — the binary moneyline defines
  `config.MONEYLINE_FEATURE_COLS`; every other consumer pulls it and none
  declares or synthesizes its own list: the run-line/totals regressors read
  `features.tree_view()` of the same list, the RFE triallist comes from the
  same list plus the declared `config.RFE_CANDIDATE_COLS`, and monitoring /
  the manifest / the workbook read the same accessor. Member routing is a
  rule (`config.RAW_PER_SIDE_COLS`), not a second list, so views are pure
  projections and duplicate columns are impossible by construction.
- **Moneyline** — XGBoost / LightGBM / elastic-net logistic
  ensemble; adaptive weights derived from pooled OOF log loss; Platt calibration
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
| `nfl_feature_selection_<date>[_targeted].json` | RFE sweep trace (record-only; never mutates the served contract) |
| `nfl_feature_workbook_<date>.xlsx` | RFE decision workbook (summary + per-trial/per-fold detail + feature pool) |
| `models/nfl_ensemble_latest.joblib` | Persisted production model bundle |

## Feature selection (RFE) run options

The sweep is opt-in and record-only: it scores trials, it never changes the
served contract. Adoption stays an explicit step
(`feature_selection.adopt()` / `--adopt`).

| Env var | Effect |
|---|---|
| `NFL_RFE_FORCE=1` | run the sweep at the end of the pipeline |
| `NFL_RFE_RETRIAL=1` | after the sweep, re-offer every pool feature: a feature the sweep committed out is re-tested and can come back |
| `NFL_RFE_ADDITION_MONEYLINE_LIST` / `NFL_RFE_REMOVAL_MONEYLINE_LIST` | comma-separated explicit add/remove targets (targeted mode) |
| `NFL_RFE_ADDITION_REMOVAL_GRID_MODE=1` | targeted cross-product grid, capped by `RFE_GRID_MAX_STATES` (default 16) |

The RFE owns NO feature list. Its trial space is pulled from the one contract
(`config.py`):

- **removals** — the incumbent serving width (`active_moneyline_feature_cols`),
  which is the master list unless an adopted subset is active;
- **additions** — only `config.RFE_CANDIDATE_COLS`, the candidate list declared
  once next to the contract (PIT-safe, pre-game, not already in the universe).
  It is empty today, so a non-targeted sweep is removals-only over the
  contract; add a candidate by declaring it in that one place (never by
  deriving one from a frame, which is how the 2026-09-22 run ended up
  trialing view-synthesized columns);
- **pool** — `config.KNOWN_FEATURE_COLS` = contract + candidates, the only
  names `set_feature_subset` accepts (an adopted record may promote
  candidates into serving width).

A completed sweep writes the trace JSON; the workbook is generated from that
trace in the same run. `openpyxl` (see `backend/requirements.txt`) is
required for the workbook — without it the step reports a non-fatal warning
and no `.xlsx` is produced.

The frontend is never modified by this backend; it consumes these artifact
families through its existing readers.

## Production / research separation

The production pipeline imports only the modules above. If every research
or legacy file were deleted, `master_pipeline.py` still functions from
scratch. The retired sealed-2025 holdout, candidate/incumbent gates,
tolerance verdicts, adoption decisions, wide-pool gates, week-ID folds,
Phase-3b market backfill, and separate 2026 regime no longer exist anywhere
in production.
