# NHL run-log review — 2026-10-08 (the 21:34 ET delivery run)

**Subject:** the run trained 2026-10-08 21:34:34 → 23:03:14 ET (5,320 s),
artifacts commit `5a892d75`, run log `nhl_pipeline_run_log.txt` = 433 lines,
`NHL_FULL_REPULL=1`, window 2024-01-01 → 2026-10-08 (674 score dates,
2,847 boxscores, 0 cache hits by design), 14 artifacts written.

**Verdict:** delivery is clean — all **11 Phase-13 gates PASS**, coverage is
100.000% measured with **no warm nulls** anywhere, the board carries the new
slate, and the headline matches the committed calibration artifact. Two
findings were remediated in this review: the season-seam drift re-check
**could never fire** (6 ALERTs page every October), and the boxscore cache
has **no refresh window** — the boxscore half of the score-page fix that
landed on 10-08 was committed as an orphaned comment and no code. One
suspicious log number (the injury bridge's 26% unmatched) was measured and
cleared as a non-defect. Tuning for both tree members was completed under the
MLB gate doctrine: **no parameter adopted** (`nhl_tune_20261008.json`).

## What was validated

### Gates, geometry, headline

All 11 Phase-13 gates PASS: `ml_probability_bounds`, `oof_population`,
`spread_grid_coherent`, `totals_grid_coherent`, `markets_has_slate`,
`slate_contract_fields`, `fold_geometry`, `season_openers_ingested`,
`oof_reach`, `sealed_holdout_gate`, `delivery_consistency`. (The 10-05 run
carried only 10 — `season_openers_ingested` is new and holds.)

| measure | run log (Phase 5/8/9) | committed artifact |
|---|---|---|
| AUC (oof_regular, n=2465) | 0.5959569634 | `nhl_calibration_20261008.json` 0.596 |
| log-loss | 0.6747233520 | 0.6747 |
| Brier | 0.2409625056 | 0.241 |
| ECE | 0.0091339743 | 0.0091 |

Fold geometry: 60 expanding windows (7-observed-date, non-overlapping),
2,655 OOF rows = 2,465 grading + 168 postseason + 179 provisional, reach
93.3%; the 14 thin windows are the usual playoff weeks/season ramps and are
excluded from pooled metrics by design (the run's only WARNING).
Members: xgboost 0.67563 / lightgbm 0.67632 / elasticnet 0.67614, adaptive
weights 0.329 / 0.228 / 0.443. The pooled calibrator gated out
(`identity`, `gated_no_gain`) — raw is served, which is why raw and
calibrated twins read identical (explained on the dashboard by the MLB 10-08
T2 note).

### Board and coverage

- `valid_dates("nhl")` → `['20261008', '20261007', …]` (11 dates): the 10-07
  board fix holds and today's slate is on the rail.
- Coverage: **68/68 features, 100.000% measured** in all three windows
  (baseline / current / serving slate), `no warm nulls` in every one. The
  2026-10-05 run warned `coverage [current]: 9 feature(s) with WARM nulls —
  goalie_sv_pct_diff(2), goalie_gaa_diff(2), goalie_starts_diff(2), …`; that
  signature is gone. Cold nulls (159, current window) are team debuts by
  design.

## Finding 1 — the season-seam drift re-check could never fire (DEFECT, fixed)

Phase 14 logged `68 features, 1 warnings, 6 alerts, 0 seasonal`:

| status | feature | current mean | baseline mean | ψ_adj |
|---|---|---|---|---|
| ALERT | rest_days_home | 42.467 | 2.256 | 0.403 |
| ALERT | rest_days_away | 43.350 | 2.148 | 0.399 |
| ALERT | goalie_starts_home | 6.000 | 45.732 | 8.141 |
| ALERT | goalie_starts_away | 6.364 | 45.704 | 8.148 |
| ALERT | ewm_net_goals_home | −0.441 | 0.195 | 0.704 |
| ALERT | ewm_goal_share_home | 0.457 | 0.518 | 0.456 |
| WARN | win_pct_home | 0.465 | 0.534 | 0.163 |

Every one is a **season-seam artifact**: the current window is the league's
first six weeks (teams have played 6 games, rest is off the charts, goalies
have started 6 times), and the baseline is the full history. The
`b698f90b` seam guard was supposed to relabel these `OK-SEASONAL` after
comparing against the same-calendar-phase window of prior years — it did not
(`0 seasonal`).

**Root cause.** The MLB port kept its fixed floor:

```python
if len(phase_vals) >= 100:   # monitoring.feature_drift
```

At the NHL seam those phase windows (late Sep → mid Oct of the prior years,
±7 days) hold only **70–110 rows** — the league has barely started — so the
re-check was *structurally* skipped every October, and the alerts paged with
no seasonal explanation.

**Fix (`nhl-backend/backend/monitoring.py`).** Require a phase sample at
least as large as the window it judges: `len(phase_vals) >= max(30, n_c)`
(`n_c >= 30` is already guaranteed by the guard above). The 2-SE test below
widens as the phase sample shrinks, so the smaller floor makes the re-check
**harder** to pass, never easier — the 100-floor was suppressing the test,
not protecting it.

**Verification.**

- `test_season_seam_recheck_reaches_sub_100_phase_samples` (new, in
  `test_run_engine_pit.py`): a 90-row phase sample against a 45-row current
  window relabels (the old floor blocked it forever), while a 40-row phase
  sample — smaller than the window it judges — still pages.
- Real-frame replay (`.freebuff/nhl_seam_verify.py`, log
  `.freebuff/nhl_seam_verify.log`) on the rebuilt 2,847×148 frame, Phase 14's
  exact call: **68 features → 57 OK, 7 OK-SEASONAL, 4 INSUFFICIENT, 0
  flagged** (pre-fix: 6 ALERT + 1 WARN). Re-check arithmetic on the six
  alerts: phase n 76–95 (all below the old 100), z = 0.31 … 1.68.

## Finding 2 — injury id bridge 26% unmatched: measured, not a defect

`injury id bridge: 3274/4454 reports matched (73.5%), 0 ambiguous, 1180
unmatched` reads like a quarter of the injury feed silently failing to bind.
Decomposed over the whole archive (`.freebuff/nhl_bridge_diag.py`, 4,691
report rows at review time):

| class | rows | share |
|---|---|---|
| matched (exactly one rating id) | 3,459 | 73.7% |
| ambiguous (several rating ids → refused by design) | 0 | 0% |
| name present in MoneyPuck but missing from the ratings frame | 0 | 0% |
| **name absent from the MoneyPuck name space entirely** | **1,232** | **26.3%** |

All 1,232 unmatched rows collapse to **40 distinct names**, every one absent
from MoneyPuck's player-game table: fifteen are goalies (Lyon, Hellebuyck,
Vladar, Merzlikins, Gustavsson, Andersen, Oettinger, Korpisalo, Annunen,
Keyser, Ullmark, Montembeault, Demko, Miner, Luukkonen) — MoneyPuck carries
skater ice time, so a goalie can never have a rating row — and the rest are
skaters with no NHL game in the archive (AHL/call-ups: Musty, Mews,
Stillman, Webber, Bear, …).

Neither group can enter the player pool, so no `pl_*` availability feature
can be affected: the exclusion predicate compares vocabularies that are
disjoint *for a structural reason*, not because matching is broken.
**No change made** — the measurement is recorded so a future reviewer does
not spend a day on it. (If goalie injuries should ever feed the expected-
goalie channel, that is a feature-design request, not a matching repair.)

## Finding 3 — the boxscore cache has no refresh window (INGESTION, fixed)

`SCORE_REFRESH_DAYS = 3` (commit `3ac49fe2`) taught *score pages* to re-pull
inside a 3-day window, because a settled page is otherwise never re-fetched
and the stale `away_sog 28/28` would feed trailing shots features forever.
The same comment block was committed a **second time with no constant and no
code after it** — the boxscore half of that work was scaffolded and never
written.

Why it matters:

- The boxscore cache write is gated **only** on the right-rail team counts
  (PP/faceoff). Goalie and SOG completeness is not gated, so a boxscore
  fetched before the feed finished posting those fields is stored incomplete.
- In `--skip-pull` mode — the only mode where the cache is read
  (`use_cache = args.skip_pull and not full_repull`) — that row is then
  served forever.
- Evidence it happened: the 2026-10-05 run's
  `coverage [current]: 9 feature(s) with WARM nulls — goalie_sv_pct_diff(2),
  goalie_gaa_diff(2), goalie_starts_diff(2), goalie_sv_pct_home/away(1 each),
  goalie_gaa_home/away(1 each), goalie_starts_home/away(1 each)`.
  Only `NHL_FULL_REPULL=1` healed it — the 10-08 run shows none.

**Fix (`nhl-backend/backend/ingestion.py`).** `BOXSCORE_REFRESH_DAYS = 3`;
the orphaned duplicate comment is replaced by the boxscore-specific one. A
cached game whose gameday is inside the window re-pulls like an uncached
one, and **every** failure mode of that refresh serves the cached row:

| refresh outcome | behaviour |
|---|---|
| network/HTTP error | cached row served + WARN |
| state not final | cached row served + WARN |
| right-rail unavailable | cached row served + WARN (PP/faceoff counts kept) |
| fresh row has incomplete counts | cached row served + WARN |
| fresh row complete | fresh row wins, cache overwritten |
| gameday unknown / no mapping | cache stays authoritative (no re-pull) |

Staleness on a bad network beats a hole in the feature frame — the same
contract the score-page refresh already made.

**Tests** (in `test_input_semantics.py`): 
`test_recent_boxscores_refresh_for_feed_corrections` pins all three
behaviours (inside the window → re-pull, correction lands in frame *and*
cache; outside → cache hit with no network call; failed refresh → cached row
served, never dropped), and `test_refresh_window_needs_a_known_gameday` pins
the undated fallback.

## Tuning — XGBoost and LightGBM (record `nhl_tune_20261008.json`)

Protocol (MLB `tune_members` doctrine, ported to the NHL harness): determinism
blueprint regime, the exact 60-fold production walk, seal = last 4 folds
(47 grading rows) excluded from all selection, 3 paired seeds (42/7/2026)
with the incumbent re-run at the same seed as each candidate.

**Stage 2 — noise floor** (new; the screens previously had no reference):

| member | log-loss by seed | floor range |
|---|---|---|
| xgboost | 0.67658 / 0.67764 / 0.67625 | **0.001390** |
| lightgbm | 0.67632 / 0.67718 / 0.67929 | **0.002967** |
| elasticnet | 0.67614 / 0.67614 / 0.67614 | 0.000000 (seed-independent) |
| blend | 0.67477 / 0.67557 / 0.67525 | **0.000803** |

Seed 42 reproduces the blueprint run bit-for-bit (0.6765829365126542), which
is the self-check that the floor runs are the same regime as the reference.

**Stage 4 — verify** (top-2 screen candidates per member, full walk × 3
seeds; gates = member *all seeds better* **and** mean Δ < −floor, blend
mean Δ ≤ floor, seal mean Δ ≤ 0):

| candidate | member mean Δ (floor) | blend mean Δ (floor) | seal mean Δ | gates | verdict |
|---|---|---|---|---|---|
| xgboost c1 (`cand_xgb.json`) | +0.000767 (0.001390) | −0.000159 (0.000803) | +0.003574 | member ✗ blend ✓ seal ✗ | **REJECT** |
| xgboost c2 | −0.000699 (0.001390) | +0.000059 (0.000803) | +0.003664 | member ✗ blend ✓ seal ✗ | **REJECT** |
| lightgbm c1 (`cand_lgbm.json`) | −0.001828 (0.002967) | −0.001067 (0.000803) | −0.017989 | member ✗ blend ✓ seal ✓ | **REJECT** |
| lightgbm c2 | −0.000455 (0.002967) | +0.000013 (0.000803) | −0.011158 | member ✗ blend ✓ seal ✓ | **REJECT** |

**`adopted: {}` — `nhl-backend/backend/config.py` is untouched.**

**Record:** `nhl-backend/data_delivery/nhl_tune_20261008.json` (47 KB:
floor runs, per-seed verify blocks, gate arithmetic). It follows MLB's
stance — `data_delivery/nhl_tune_*` is training residue, regenerated by the
harness and **never committed** (the rule was added to
`nhl-backend/.gitignore` alongside this review); the verdict tables above
are the durable evidence.

Reading, and it is the MLB L9 story again:

- The **narrow-surface artifact** struck both screens. XGB c1 was the best
  screen trial (0.67540 vs 0.67658 incumbent on the single-member surface)
  and *lost* on the full walk at two of three seeds (+0.0013, +0.0022) —
  its blended gain is a weight reallocation, not a stronger member.
- **LightGBM's own seed floor (0.002967) is wider than any screen gain.**
  Bagging at `bagging_fraction 0.47` with a different `random_state` moves
  the member more than the candidates move it, so no trial could clear the
  bar honestly. c1 is the one real lead: better on all 3 seeds, better blend
  (−0.0011), better seal (−0.0180 on 47 rows) — recorded as the candidate to
  revisit if the floor ever narrows (e.g. after a `bagging_freq`/seed
  stability pass), never by lowering the gate.
- Both members are therefore **confirmed**, not improved: the production
  config stays.

## Ops observations (recorded, not changed)

- **Duration 5,320 s** vs 4,414 s (10-08 10:42 run) vs 2,268 s (10-05 run).
  The delta is feed latency, not code: boxscores pulled at **1.19 game/s**
  vs **~3.5 game/s** on 10-05 (same 2,847 games, same code path), score
  pages 3.1–5.4/s vs 3.6–9/s. Phase 3 (PIT feature engine) is 80 of the 89
  minutes: 39 min of boxscore fetch + ~41 min of frame build.
- **Two full rebuilds in one day** (10:42 and 21:34, both `FULL_REPULL=1`,
  ≈2.2 h combined). The committed Kaggle notebook still pins
  `NHL_FULL_REPULL = "1"` under the comment *"set once, then remove"*. With
  `SCORE_REFRESH_DAYS` and `BOXSCORE_REFRESH_DAYS` now in place an
  incremental run is the cheaper default — flagged for the notebook owner
  (the notebook is Kaggle-owned per `nhl_model_audit_20261006.md`), not
  edited here.

## Verification

- `python -m pytest nhl-backend/backend/ -q --ignore=…/test_run_engine_pit.py`
  — **455 passed** (includes the 2 new boxscore-refresh tests).
- `python -m pytest nhl-backend/backend/test_run_engine_pit.py -q -k "drift or seam"`
  — **9 passed** (includes the new sub-100 phase-sample test);
  `-k "boxscore or refresh or cache"` — **12 passed**.
- `python -m pytest frontend/test_nhl_frontend.py
  frontend/test_nhl_pooled_oof_parity.py
  frontend/test_nhl_board_render_smoke.py -q` — **11 passed**.
- `python -m pytest nhl-backend/backend/test_run_engine_pit.py -q` (the
  slow model-training suite) — **151 passed in 8:14**, run separately on the
  final tree. NHL backend total: **606 passed**.
- Real-frame seam replay: `.freebuff/nhl_seam_verify.py` → 0 flagged rows
  across all 68 features.

## Known limitations

- The production pipeline was **not** re-run (Kaggle environment). Every
  conclusion comes from the committed run log and artifacts, local replays
  on the locally rebuilt frame (`.freebuff/nhl_game_frame_20261007.parquet`,
  2,847×148 — the Kaggle frame's exact shape), and live code execution.
- The seam replay and the tuning frame are the *local* rebuild; they match
  the run's shape and fold count (60) but were not produced on the remote
  machine.
- The boxscore refresh acts only where the cache is read
  (`--skip-pull`); `NHL_FULL_REPULL=1` runs bypass it entirely, as before.
- Scratch harnesses live in `.freebuff/` (untracked by policy):
  `nhl_seam_verify.py`, `nhl_bridge_diag.py`, `nhl_tune.py`,
  `nhl_build_frame.py`, `nhl_drift_diag.py`, plus their logs.
