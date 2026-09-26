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

Three upstreams, each asked for the one thing it is actually good at, with no
fallback route between them. This is the same division of labour MLB already
runs, and it is the reason the schedule and the features come from different
companies: no single NBA vendor has both.

| Frame | Upstream | Why that one |
|---|---|---|
| schedule, sides, results | **ESPN scoreboard** | the only one that can report a game nobody has played yet, which is what a pending slate *is*. stats.nba.com reports no future games. |
| features (box score, player lines) | **stats.nba.com `LeagueGameLog`** | one request per season returns every player line of that season — 26,306 rows for 2024-25. No per-game API reaches the same data in fewer requests, because a per-game API is one request per game. |
| play-by-play | **stats.nba.com `playbyplayv3`** | one request per game, ~0.1s. Counted into a per-team event rollup that the feature ladder reads. |

No key, quota, or paid tier is involved on any of them.

`cdn.nba.com` is gone. It was a second hostname for the same vendor wearing a
fallback's clothes, and it was the thing that made a blocked host look like a
flaky pipeline rather than an unreachable one.

### Identity: the schedule owns it

`game_id` is ESPN's event id, and stats.nba.com's own game id rides alongside in
`nba_game_id` because that is the key the play-by-play endpoint is addressed by.
The two meet on `(date, the unordered pair of teams)` — the only fact both
upstreams agree on.

The key is deliberately orientation-free. The season log spells each matchup
from that row's own team's perspective, and for 5 games in the 2024-25 log it
uses `@` on both sides, so `WAS @ MIA` appears on WAS's rows even though WAS is
home. A join on the ordered pair silently inverts the sides of some games, and
a schedule with swapped venues still looks like a schedule. The schedule
supplies home/away; the log contributes only the NBA game id.

### The play-by-play rollup is checked against the box score

The rollup deliberately re-counts the box score's own columns. Two independent
counts of the same game that agree are evidence the pull is right, and two that
disagree mean one of them is wrong — a signal nothing else produces. The
comparison is published in the run manifest and in `nba_pipeline_summary.json`.

On a full 2024-25 window (2,628 team-games) `fga`, `fgm`, `fg3a`, `fg3m`, `fta`,
`ftm`, `tov` and `points` match **exactly, every game**. Rebounds and fouls
differ slightly and are advisory only: the box score is a sum over player lines
while the event stream is a narration, and the model reads the box score for
those columns. The model reads the rollup only for what an event stream knows
and a box score does not — shot location, live-ball turnovers, and-ones, the
quarter split.

Two things the feed encodes only in prose, which is why the rollup reads text
at all:

* A free throw's make is a **missing leading `MISS`**. `shotResult` is empty on
  every free throw, and the trailing `(N PTS)` is cumulative across a trip, not
  the points on the shot.
* A rebound's side is in **`REBOUND (Off:1 Def:0)`**, and those are the
  *player's* running totals — a team's total is the sum of each player's
  maximum, never the team maximum and never the sum of the readings. A rebound
  the feed attributes to a team (`Hawks Rebound`) has no team on the row at
  all, so it is counted per game rather than forced onto a side.

There is no assist, steal or block action in this feed. Assists are counted from
the made shot's text, which names the assister; they are never attributed to a
player id, because the action's own `personId` is the shooter.

### The User-Agent is bare on purpose

Both upstreams refuse a full browser impersonation. `stats.nba.com` accepts the
connection and then never answers — the read times out with no status line,
which is what an edge that has decided to tarpit a client looks like from
Python. `site.api.espn.com` answers HTTP 403 immediately. With a bare
`Mozilla/5.0` both serve in about 0.15s.

So the request is not the problem and neither host is blocked; the costume was.
A test asserts this, because anything that re-adds a browser string will
reintroduce a failure that presents as an unreachable host.

## Player detail is required

The pipeline refuses to publish a window that has games but no player lines,
rather than training every player-derived feature on its default.

It also refuses a window with no settled games, a schedule that joined to no
team facts, or a frame with a duplicated game — each naming what is wrong. A
joined-but-empty team frame is the failure to look for: it is what a team-name
mismatch produces, and it is silent, because the unmatched side is simply
absent rather than wrong.

## Cache

The normalized cache lives outside the repository (by default under
`~/.cache/sports_prediction_model/nba`; set `NBA_CACHE_DIR` to override it, and
Kaggle/CI can point it at a mounted volume). It is **per-source and per-key**
rather than one file per run: `schedule/YYYYMMDD.parquet` per day,
`season_logs/log_<season>_<type>.parquet` per season, and
`play_by_play/pbp_<nba_game_id>.parquet` per game. A partial or failed sweep
therefore re-fetches only what it did not get, which is what makes a
1,300-game play-by-play sweep affordable to run incrementally.

`--skip-pull` serves from the cache without touching the network, so a host that
can reach neither upstream can still run the model against a window that was
fetched elsewhere.

Window and sweep controls are environment variables: `NBA_START_DATE` /
`NBA_END_DATE` bound the window; `NBA_FULL_REPULL=1` ignores every cache;
`NBA_FETCH_PLAY_BY_PLAY=0` skips the play-by-play sweep; `NBA_PBP_MAX_GAMES`,
`NBA_PBP_BUDGET_SEC`, `NBA_PBP_PAUSE_SEC` and `NBA_PBP_LOOKBACK_DAYS` size it;
`NBA_SCHEDULE_BUDGET_SEC` bounds the day-by-day schedule sweep; and
`NBA_REQUEST_TIMEOUT_SEC` / `NBA_REQUEST_ATTEMPTS` bound a single request.

A cold play-by-play sweep is the long pole: ~0.25s per game, so a 1,300-game
window takes about five minutes once and nothing after. The default cap is
sized to cover a real share of the window on purpose — the event features are
trailing, so a sweep that reaches 8% of the games leaves 92% of the rows with no
event history at all, and the run manifest reports the share that was reached.

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
