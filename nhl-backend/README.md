# NHL Backend

Binary moneyline + run-line prediction backend for the NHL — a structural
mirror of `nfl-backend/` (MLB lineage). One authoritative entry point:

```
python3 backend/master_pipeline.py              # full production run
python3 backend/master_pipeline.py --skip-pull  # use the .nhl_cache/ pulls
```

## Data

* Official NHL API (`https://api-web.nhle.com/v1`) — `/score/{date}` per-date
  schedule/score pages and `/gamecenter/{id}/boxscore` per-game boxscores,
  cached per date/game under `.nhl_cache/` beside the repo.
* MoneyPuck shots data is OPTIONAL enrichment (`load_moneypuck_shots`); the
  pipeline never requires it.
* Market-free: no sportsbook data is ingested or used anywhere.

## Model family (identical to MLB/NFL)

* **Moneyline**: XGBoost + LightGBM + elastic-net ensemble, MLB-tuned member
  params, rolling SLSQP blend-weight re-earning in logit space (no floor/cap),
  favored-space Platt calibration.
* **Run line**: per-side LightGBM-Poisson goal regressors → negative-binomial
  Monte Carlo → full spread (−8..+8) / total (4..12) grids + per-line
  prequential Platt calibration.
* **Walk-forward**: expanding 7-day folds, MLB-style 30-day warm-up, data
  starting season 2024 (2024-25), `MIN_VAL_FOLD_GAMES = 40`.

## Artifacts (`data_delivery/`)

`nhl_moneyline_v1_<date>.json`, `nhl_calibration_<date>.json`,
`nhl_predictions_history_<date>.csv`, `nhl_power_rankings_<date>.csv`,
`nhl_run_engine_markets_<date>.csv` (+ `.meta.json`),
`nhl_run_engine_monitor_<date>.json`, `nhl_goalie_matchup_<date>.json`,
`nhl_feature_v1_<date>.json`, `nhl_model_monitor_<date>.json`,
`nhl_shap_game_<game_id>.csv`, `models/nhl_ensemble_latest.joblib`,
`nhl_production_cards_history.csv`, `nhl_oof_moneyline.csv`,
`nhl_oof_distribution.csv`, `nhl_pipeline_summary.json`, `nhl_fold_table.csv`,
`run_engine_feature_drift_/coverage_<date>.csv`.

Game ids embed the date (`YYYYMMDD_AWAY@HOME` — the MLB convention), so SHAP
files age by filename.

## Environment knobs

| Variable | Meaning (default) |
| --- | --- |
| `NHL_START_DATE` / `NHL_END_DATE` | run window (else today ET) |
| `NHL_NO_PUSH=1` | skip the data-delivery git sync |
| `NHL_RFE_FORCE=1` | run the record-only feature selection sweep |
| `NHL_RFE_ADDITION_MONEYLINE_LIST` / `NHL_RFE_REMOVAL_MONEYLINE_LIST` | targeted RFE trial lists |
| `NHL_RFE_ADDITION_REMOVAL_GRID_MODE=1` | silent 16-state grid trials |
| `RFE_GRID_MAX_STATES` | grid window cap (16) |
| `CALIBRATION_MODE` | `platt` (default) or `identity` |

## Tests

```
python -m pytest nhl-backend/backend/     # from the repo root
```

`test_grid_rfe.py` — offline pins (warm-up fold gate, feature-view routing,
blend-weight contract, config parity, manifest parity, retention).
`test_run_engine_pit.py` — point-in-time pins (grid coherence, prequential
per-line Platt, leakage gates, NB dispersion, monitor line pairs).
