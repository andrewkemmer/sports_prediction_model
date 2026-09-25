# NBA Production Backend

Standalone NBA moneyline and totals/spread backend structurally mirroring the
MLB/NFL/NHL production contracts. The canonical historical input is the free
`wyattowalsh/basketball` Kaggle warehouse export.

## Run

Attach or download the pinned Kaggle dataset, then run:

```bash
cd nba-backend/backend
python master_pipeline.py --source-path /path/to/basketball
```

On Kaggle, the backend also discovers mounted warehouse exports below
`/kaggle/input` (including generated dataset slugs and nested Parquet/CSV
partitions). The `kaggle_nba_run.ipynb` notebook resolves the source before
starting the pipeline; it never passes a missing value as `--source-path`.
The normalized cache is stored outside the repository (by default under
`~/.cache/sports_prediction_model/nba`; set `NBA_CACHE_DIR` to override it).
Set `NBA_KAGGLE_DATASET_PATH` in Kaggle, or pass `--source-path` explicitly.
The production graph does not import another sport's backend.

The model uses the exact MLB XGBoost/LightGBM/elastic-net member parameters,
seed 42, seven-day expanding walk-forward folds, 30-day warm-up, season-2024
eligibility, and rolling SLSQP blend optimization in logit space. The run-line
engine uses two Poisson score regressors and a seeded negative-binomial joint
score sampler for NBA totals and point-spread grids. No sportsbook data is
used.

The default Kaggle notebook is `kaggle_nba_run.ipynb` at repository root. It is
orchestration only; all backend logic lives here.
