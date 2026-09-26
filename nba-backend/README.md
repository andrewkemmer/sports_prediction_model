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
| features (box score, player lines) | **stats.nba.com `LeagueGameLog`** | one request per 60-day slice returns every player line in that slice. No per-game API reaches the same data in fewer requests, because a per-game API is one request per game. |
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
`season_logs/log_<season>_<type>_<from>_<to>.parquet` per 60-day slice, and
`play_by_play/pbp_<nba_game_id>.parquet` per game. A partial or failed sweep
therefore re-fetches only what it did not get, which is what makes a
1,300-game play-by-play sweep affordable to run incrementally.

The slice width is part of the key on purpose. A whole-season file and a
60-day file are not interchangeable, and a key loose enough to let one stand in
for the other would let a run assemble its window from whichever granularity
happened to be on disk. One consequence to expect: the first run after this
change re-pulls the season log into slices and leaves the old
`log_<season>_<type>.parquet` files behind, unused.

`--skip-pull` serves from the cache without touching the network, so a host that
can reach neither upstream can still run the model against a window that was
fetched elsewhere.

Window and sweep controls are environment variables: `NBA_START_DATE` /
`NBA_END_DATE` bound the window; `NBA_FULL_REPULL=1` ignores every cache;
`NBA_SLICE_DAYS` sets the season-log slice width (default 60) and the width of
the windows the schedule sweep reports in;
`NBA_FETCH_PLAY_BY_PLAY=0` skips the play-by-play sweep; `NBA_PBP_MAX_GAMES`,
`NBA_PBP_BUDGET_SEC`, `NBA_PBP_PAUSE_SEC` and `NBA_PBP_LOOKBACK_DAYS` size it;
`NBA_SCHEDULE_BUDGET_SEC` bounds the day-by-day schedule sweep; and
`NBA_REQUEST_TIMEOUT_SEC` / `NBA_REQUEST_ATTEMPTS` bound a single request.
`NBA_PROGRESS=0` silences the progress display without changing any result.

### How the run reports itself

A run prints a banner per phase and a `✅` line with the numbers each one
produced, MLB's idiom, and every sweep carries a `tqdm` bar: the schedule
days, the season-log slices, and the play-by-play games.

**The bars draw in a captured log too**, and that is the whole point of the
shape. MLB's are not MLB's code — `pybaseball.statcast` wraps its per-day
sub-requests in `tqdm(total=len(date_range))`, and `tqdm` writes to a pipe, a
file and a Kaggle cell exactly as it writes to a terminal. The MLB log this
mirrors reads:

```
  Chunk: 2024-03-01 → 2024-04-29
  0%|          | 0/46 [00:00<?, ?it/s]
100%|██████████| 46/46 [00:52<00:00,  1.15s/it]
    → 164216 pitches
```

Three things per window, in that order: the line naming it, a bar that walks
`0/N` to `N/N` while the work happens, and the line reporting what it got. The
schedule sweep does the same over `NBA_SLICE_DAYS`-day windows, one request per
day, so every tick is a day actually walked:

```
  Chunk: 2024-01-01 -> 2024-02-29
schedule:   0%|          | 0/60 [00:00<?, ?day/s]
schedule: 100%|##########| 60/60 [00:31<00:00,  1.93s/day]
    -> 407 game(s) over 60 day(s) (0 fetched, 60 from cache)
```

Two details make the capture readable, and both are about keeping the bar line
short. The bar names no window and carries no running totals, because the
`Chunk:` line above it names the window and the `-> N games` line below it
reports the result; and every bar is pinned to `position=0`, because a bar that
tqdm stacks above another rewinds the cursor on every redraw, which is invisible
on a terminal and an `ESC[A` per refresh in a log. The season-log bar counts
slices (one request each) rather than days for the same reason: 34 requests is
the honest total, and a bar that counted 2,040 days would be measuring a
different sweep than the one that runs.

`NBA_PROGRESS=0` silences the display without changing any result, and a
`requirements.txt` without `tqdm` falls back to a heartbeat log line every ten
seconds carrying the count, rate and ETA. The bars are display only — with them
on, off, or unavailable, the run returns byte-identical artifacts.

That fallback exists, but it should not be what you see. The first version of
this module refused to draw a bar unless `sys.stderr.isatty()`, on the theory
that a redrawn bar in a log file is noise; on Kaggle, where stderr is captured
and never a tty, that meant a run that spent ten minutes walking 1,024 schedule
days printed nothing at all until it finished — indistinguishable from a hang,
and a stricter rule than `tqdm` itself keeps. The gate is gone. `tqdm` can
suppress itself on a non-terminal (`std.py`: `if disable is None and not
file.isatty()`), but `disable` defaults to `False`, not `None`, so that branch
is unreachable unless a caller asks for it.

One rule matters if you add a sweep: **tick the counter on the way out of the
unit, not on the way in.** `bar.item()` exists for this and is the thing to
wrap a loop body in. Ticking at the top of the body — the obvious way to write
it — makes the count lead the work, and the 2026-09-26 Kaggle run showed exactly
what that costs: the schedule sweep closed on `1023 fetched` and then summarised
`1024 fetched`, and the play-by-play sweep closed on `608 fetched` and then
summarised `609 fetched`. Two numbers for one fact, a few lines apart, in the
log an operator is relying on. A budget `break` taken before the block leaves
the count short of the total, which is the honest reading: those units were
never asked about.

### How much play-by-play a run actually gets

The event features — offensive and defensive rebounds, fouls, possessions — are
counted from per-game play-by-play, and the sweep is deliberately partial. It
takes the most recent `NBA_PBP_LOOKBACK_DAYS` (240) days of played games, capped
at `NBA_PBP_MAX_GAMES` (1,500) and bounded by `NBA_PBP_BUDGET_SEC` (5,400).

**The lookback is what binds, not the cap or the budget.** Measured on the
2026-09-26 window: 2,768 games carry an NBA game id, of which 609 fall inside
240 days, 1,315 inside 400, and all 2,768 inside 900 — while the 1,500 cap and
the 5,400s budget are both slack (the whole 609-game sweep took 2.6s warm and
283s cold). So `team_events` covers roughly 18% of settled games, and the
event-derived features are forward-filled or absent for the rest.

That is a policy choice rather than a defect, and it is the obvious lever if the
event features are worth more than the fetch time: raising the lookback to cover
the window would fill `team_events` for every training game, at roughly 2.2
games/s. On a cold Kaggle run that is ~26 minutes added to a ~10 minute run, so
it is a deliberate trade rather than something to change silently.

### 60 days where the endpoint allows it, one day where it does not

The season log is pulled in 60-day slices, which is MLB's number
(`results.SCHEDULE_CHUNK_DAYS`, `statcast_chunk_days`) and is measured here
too: a 60-day slice of 2023-24 returned 8,627 player rows across 405 games in
1.98s against 26,401 rows in 2.84s for the whole season, and each slice's game
set was a strict subset of the season's. A 1,024-day window over three seasons
is 28 requests instead of 6, each a third of the size, each individually
restartable, and the run's model output is unchanged.

The schedule is **not** sliced, and that is not an oversight. MLB can chunk its
schedule because StatsAPI's `schedule` endpoint takes `startDate`/`endDate`.
ESPN's scoreboard has no equivalent: given a range it either refuses outright
(HTTP 400) or ignores the date and answers with whatever slate it currently
holds. Measured 2026-09-26, a request spanning January 2024 came back `200`
with one game dated October 2026.

The second behaviour is the dangerous one, because nothing complains. A 60-day
schedule sweep would issue 18 requests, receive 18 copies of the same game, and
report a plausible 18-game schedule with no error raised. So
`_answered_a_different_day` treats a response whose every game falls outside the
day requested as a refusal, which routes it into the same consecutive-failure
breaker a 403 does. A single stray game does not trip it — ESPN's `date` is UTC
and the frame is Eastern, so a late start legitimately lands a game on the next
day — which is why the check asks whether *every* row is off-day.

### A host that refuses this client

Both upstreams can and do refuse a given client, and the symptom is expensive
to diagnose by sweeping: a refusal is cheap per request and ruinous in
aggregate. The schedule sweep asks ESPN one question per day, so a 1,024-day
window is 1,024 requests; a 403 returns instantly and the sweep finished the
whole window before reporting that the schedule was *missing*, which names the
wrong culprit. Against a host that accepts the connection and then stops
answering — what `stats.nba.com` does — the same code would have spent roughly
25 hours at the 90s request timeout proving one thing.

So refusal is detected before it is suffered:

* **A preflight probe.** The first time a process needs to ask a host for
  something it has not cached, it makes one cheap request against the same
  endpoint and headers the sweep uses, with a 20s ceiling. A refusal raises
  immediately, naming the host, the reason, and what continuing would have
  cost. A fully cached run never probes, so it can never fail here.
* **A consecutive-failure breaker.** Each sweep stops after a short run of
  failures — 3 days for the schedule, 2 season logs, 5 games of play-by-play —
  because a single failure is a blip and a run of them is a decision about this
  client rather than about the network. `NBA_SCHEDULE_MAX_CONSECUTIVE_FAILURES`
  and `NBA_PBP_MAX_CONSECUTIVE_FAILURES` tune the thresholds. The play-by-play
  breaker stops the sweep *without* failing the run: that feature is
  forward-filled across games without it, so no play-by-play is a thinner model
  rather than a wrong one, and the run manifest records that the sweep tripped.

A cold play-by-play sweep is the long pole: ~0.25s per game, so a 1,300-game
window takes about five minutes once and nothing after. The default cap is
sized to cover a real share of the window on purpose — the event features are
trailing, so a sweep that reaches 8% of the games leaves 92% of the rows with no
event history at all, and the run manifest reports the share that was reached.

The production graph does not import another sport's backend.

## Games that are not league games

The schedule adapter drops any event with a side that is not one of the 30
franchises, using the same alias table the join uses, so a team ESPN spells
differently (`GS`, `NO`, `NY`, `SA`, `UTAH`, `WSH`) is still recognised.

This exists because the all-star event cannot be caught any other way. ESPN
files it as `regular-season`, so the season-type check cannot see it, and in
2026 the all-star became a four-game tournament on a single date that played the
same two squads twice. That made the orientation-free date/team-pair join key
ambiguous — it is unique only because a team plays at most once a day — and
raised `MergeError` for the entire pipeline, over four games that have no player
lines in the season log and so could never have been trained on. The drop is
logged by name rather than done silently, and the join also resolves a
duplicate key with a warning rather than failing, because the uniqueness it
relies on is an assumption about the world and not about the data.

## Model

The exact MLB XGBoost/LightGBM/elastic-net member parameters, seed 42, seven-day
expanding walk-forward folds, 30-day warm-up, season-2024 eligibility, and
rolling SLSQP blend optimization in logit space. The run-line engine uses two
Poisson score regressors and a seeded negative-binomial joint score sampler for
NBA totals and point-spread grids. No sportsbook data is used.

## Notebook

`kaggle_nba_run.ipynb` at repository root is orchestration only: it clones the
repo, installs `requirements-kaggle.txt`, and runs `master_pipeline.py`. All
backend logic lives in this directory.

### Delivery

The pipeline publishes its own artifacts, in its final phase, to
`nba-backend/data_delivery` on `main` — the same boundary MLB, NFL and NHL
publish on. It uses a throwaway clone rather than the run's own checkout, so a
delivery problem cannot disturb the code it just ran, and it pushes with
retries and then verifies the paths on the remote tree: a push that reports
success and delivers nothing would otherwise be indistinguishable from a good
run, because both report success.

Publication requires `GITHUB_TOKEN` in the environment. Without one the run
skips delivery and says so in the `sync` block of its summary, with the reason;
it does not silently succeed. `NBA_PUSH=0` forces the skip, and `NBA_PUSH=1`
forces the attempt so a missing token fails loudly instead of quietly skipping.
The remote comes from `GITHUB_REPO_URL` when set, otherwise from the checkout's
own `origin`.

This phase used to be a no-op that returned `staged_files: []` and the message
"automatic Git staging/commit/push disabled". On Kaggle, where the working
directory is ephemeral, that meant every run wrote its artifacts and then lost
them at session end, and `nba-backend/data_delivery` stayed empty on `main`
permanently while the notebook went on reporting that artifacts had been
pushed. NBA was the only sport not publishing.

### Retention: what the dashboard will serve

The board offers a **rolling 10-day window**, the same shape MLB's frontend
enforces (the backend half of MLB's policy is `retention_policy.py`; the
frontend half is the valid-date filter in `frontend/utils.py`). The NBA's is
enforced in that same place, and it is the only thing bounding which dates a
card can be rendered for.

**The anchor is the newest date the board can serve, not today.** That is the
one place the NBA cannot copy MLB verbatim, and it is a property of the data
rather than a preference. MLB's boards are same-day slates, so "today" and "the
newest board" are the same day. The NBA run publishes the slate it is *about to
play*: measured on the deployed branch at 2026-09-26, the moneyline was
`slate_date 2026-10-20` (24 days ahead) while the newest played game in the
retained history was `2026-06-13` (105 days behind) — the season does not exist
between them. A today-anchored window would hold neither and the board would
render empty for the whole offseason and pre-season.

So the window runs from the newest served date back ten days, which is MLB's
own rule read precisely ("keeps the run's anchor date and the 10 days before
it", where the anchor is the run's window end, which *is* the newest slate).
It buys a guarantee the today-anchored version does not have: the newest date
is inside its own window by construction, so the truncation can never empty the
board it applies to. The production slate also wins the anchor over history on
purpose — a history date past the slate must not push the slate out of the
window and land the board back on an archive card.

Measured before and after, against the deployed artifacts:

| | dates offered | range |
|---|---|---|
| before | 284 | 2024-11-22 .. 2026-10-20 |
| after | 1 | 2026-10-20 |

Every dropped date predates the window (newest dropped: `2026-06-13`) and no
in-window date was lost.

This is what retires the archive card. The board's render gate already refused
a date outside its valid set, so bounding the valid set means a date that is no
longer served cannot reach a card by any path — including the history fallback
that rebuilds a card from the OOF prediction-history CSV. An NBA dashboard
therefore never shows a card for a date whose published prediction has aged out,
which is the invariant the same rule buys MLB.

Two things are deliberately **not** done here:

- **No backend pruning.** MLB's policy has a second half that `git rm`s dated
  artifacts outside the window in its delivery phase. The NBA has no equivalent,
  so `nba-backend/data_delivery` keeps accumulating dated artifacts that the
  dashboard will never serve. Adding it is a backend change and is out of scope
  for this one; the frontend window already makes those files invisible, at the
  cost of repository growth.
- **No change to the board itself.** The card layout, the date rail, the
  prev/next stepping, the calendar and the history fallback are all untouched.
  The frozen-cards store is still consulted first for a served date, and the
  OOF CSV is still the last resort — MLB keeps that same ladder, and MLB's
  invariant rests on the window rather than on removing the fallback.
