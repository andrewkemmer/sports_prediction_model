# NBA Production Backend

Standalone NBA moneyline and totals/spread backend structurally mirroring the
MLB/NFL/NHL production contracts.

## Run

```bash
cd nba-backend/backend
python master_pipeline.py
```

The backend has no external input, no mounted volume and no download step. It
reads public, unauthenticated APIs and normalizes them itself. There is no
`--source-path`: the flag is gone rather than accepted-and-ignored, so a run
cannot be pointed at a directory of tables it never produced.

## Sources

Three routes, tried in order. The first that can serve the whole window wins,
and the manifest records which one answered.

1. **`stats.nba.com/stats/LeagueGameLog`** — the backbone. One request returns a
   whole season of player game logs, so the game index, the game type, and every
   player line arrive in the same call. Regular season and playoffs are separate
   calls, which is where the game type comes from; nothing is inferred from
   dates. ~10 requests for any window, re-read every run.
2. **`cdn.nba.com`** — the same JSON NBA.com serves its own site: the league
   schedule, plus per-game box scores and play-by-play. Used to rebuild the
   window when the season log cannot be read.
3. **`site.web.api.espn.com`** — the last resort. The ESPN scoreboard for the
   schedule and one `summary` box score per game. Cloud hosts block both NBA.com
   surfaces at the IP level; ESPN is what serves the window there. It costs
   ~5,700 requests for a full window against ~10 for the season log, so the
   player lines it yields are cached to `espn_player_stats.parquet` and a later
   run asks only for the games the cache has not seen.

No key, quota, or paid tier is involved on any route.

A host that answers with refusals is detected, not retried into a loop: after
`REFUSAL_VERDICT_COUNT` consecutive 403s the walk stops and the run falls
through to the next route rather than spending its whole request budget being
told no.

## Player detail is required

The pipeline refuses to publish a window that has games but no player lines,
rather than training every player-derived feature on its default. That gate is
the reason the ESPN route exists: on a host where NBA.com is blocked, a run that
could only read schedules would otherwise look successful and be silently wrong.

## Cache

The normalized cache lives outside the repository (by default under
`~/.cache/sports_prediction_model/nba`; set `NBA_CACHE_DIR` to override it, and
Kaggle/CI can point it at a mounted volume). It holds `games.parquet`,
`team_stats.parquet`, `player_stats.parquet`, `play_by_play.parquet` and
`run_manifest.json`, which records the source that actually answered and the
row counts. `--skip-pull` serves from it without touching the network.

Window and rebuild controls are environment variables: `NBA_START_DATE` /
`NBA_END_DATE` bound the window, `NBA_FULL_REPULL=1` ignores every cache,
`NBA_FETCH_PLAY_BY_PLAY=0` skips play-by-play, and `NBA_PULL_DEADLINE_SEC`
bounds the season-log pull.

The production graph does not import another sport's backend.

## Model

The exact MLB XGBoost/LightGBM/elastic-net member parameters, seed 42, seven-day
expanding walk-forward folds, 30-day warm-up, season-2024 eligibility, and
rolling SLSQP blend optimization in logit space. The run-line engine uses two
Poisson score regressors and a seeded negative-binomial joint score sampler for
NBA totals and point-spread grids. No sportsbook data is used.

## Notebook

`kaggle_nba_run.ipynb` at repository root is orchestration only: it clones the
repo, installs `requirements-kaggle.txt`, runs `master_pipeline.py`, and stages
`nba-backend/data_delivery` for human review without committing or pushing. All
backend logic lives in this directory.
