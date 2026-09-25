# NBA Production Backend

Standalone NBA moneyline and totals/spread backend structurally mirroring the
MLB/NFL/NHL production contracts.

## Run

The canonical historical input is the free `wyattowalsh/basketball` Kaggle
warehouse export, but **this backend has no reader for that export's raw
format.** It loads a *normalized* warehouse: a directory holding
`games.parquet`, `team_stats.parquet` and `player_stats.parquet`. If you have
one, run:

```bash
cd nba-backend/backend
python master_pipeline.py --source-path /path/to/normalized-warehouse
```

The pipeline will find that directory itself if you do not pass it. It looks,
in order, at `NBA_KAGGLE_DATASET_PATH`, then at `/kaggle/input` and
`/kaggle/working/nba-warehouse` (including Kaggle's generated dataset-slug
directories), and finally it will download the pinned export with the Kaggle
CLI into `NBA_KAGGLE_DOWNLOAD_DIR` (default `/kaggle/working/nba-warehouse`)
provided a Kaggle token is present in `KAGGLE_USERNAME`/`KAGGLE_KEY` or
`~/.kaggle/kaggle.json`. Set `NBA_KAGGLE_AUTO_DOWNLOAD=0` to disable that last
step; it is skipped entirely when there is nothing to authenticate with.

If a raw `nba.duckdb`, `.sqlite` or similar export is found, the run stops and
says so by name. That is a missing feature rather than a misconfiguration:
deriving the normalized tables from the pinned export needs that export's
table and column names, which are not in this repository.

Without a normalized warehouse the pipeline pulls from NBA.com. Those routes
are blocked from Kaggle, and none of them can supply player lines — so a run
that falls back to them is refused rather than published, because a window with
no player detail silently trains every player-derived feature on its default.

The normalized cache is stored outside the repository (by default under
`~/.cache/sports_prediction_model/nba`; set `NBA_CACHE_DIR` to override it).
The production graph does not import another sport's backend.

The model uses the exact MLB XGBoost/LightGBM/elastic-net member parameters,
seed 42, seven-day expanding walk-forward folds, 30-day warm-up, season-2024
eligibility, and rolling SLSQP blend optimization in logit space. The run-line
engine uses two Poisson score regressors and a seeded negative-binomial joint
score sampler for NBA totals and point-spread grids. No sportsbook data is
used.

The default Kaggle notebook is `kaggle_nba_run.ipynb` at repository root. It is
orchestration only; all backend logic lives here.
