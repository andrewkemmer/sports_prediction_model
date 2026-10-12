# NFL closing-line archive — implementation plan (free ESPN source)

Goal: a de-vigged market benchmark for every NFL game back to **2016-01-01**,
built free from ESPN's core API, plus a forward daily collector — the
measurement substrate for model-vs-market grading and CLV. Paid alternatives
were evaluated this session and cannot serve the 2016 window at any price:
The Odds API starts 2020-06-06 (no Pinnacle), SportsGameOdds "varies by
league", Unabated is live-only ($799–3,000/mo), OpticOdds ~2020s (~$5k/mo).
ESPN's archive is the only multi-book source reaching 2016 — verified live.

## Evidence this plan is built on (probed 2026-10-11, all exit 0)

- **Depth:** odds present 100% of sampled games 2014 → 2026; real books start
  2014; 2016-01-09 Wild Card and Super Bowl 50 both complete; SB51–57, SB59–60
  complete; **SB58 (2024-02-04) is a zero-odds placeholder** ("NFC at AFC").
- **Book panels per game:** 2016 ≈ 8–9 complete (5Dimes, BetOnline,
  SportsBetting, Bovada*, Westgate, Pinnacle (playoffs), consensus*, …);
  2017–23 = 6→13 (Caesars family, Wynn, CG, Unibet, DK from 2019, consensus
  **prices** 2017→mid-2023 only — None/None in 2016); 2024 = Bet 365 + ESPN
  BET; 2025 = ESPN BET only; 2026 = DraftKings only.
- **Both ML sides present** on frozen closes in every era → two-way de-vig
  computable throughout. Measured holds: 2.0–3.4% (2016 sharp books), 2.4%
  Pinnacle, 4.6% ESPN BET, 4.3% DK.
- **Same-game cross-book dispersion** of de-vigged fair prob: 2.9 pts (2016 WC,
  Bovada outlier) and ~1.0 pt (2016 WK1) — book choice ≈ edge-sized, hence
  median-across-books + dispersion as a stored uncertainty.
- **Hygiene traps confirmed:** "ESPN BET - Live Odds"/"Caesars … - Live Odds"
  twins zero out post-game; `Opening` is not a close; numberfire /
  teamrankings / accuscore / Betradar are model feeds, not books; Bovada/BetUS
  items zero on some games; Caesars state-suffixed duplicates (CO/NJ/TN) are
  one book counted many times.

## Architecture (fits repo hygiene: production graph + documented regeneration builders)

Three pieces, in `nfl-backend/backend/`, mirroring the `backfill_lineups.py`
precedent (allowlisted regeneration builder) and the MLB ESPN fetch pattern:

### 1. `build_odds_history.py` — one-time backfill (allowlisted builder)

- Walk every NFL event 2015-09 → present via the ESPN scoreboard endpoint
  (`site.api.espn.com/…/nfl/scoreboard?dates=YYYYMMDD`), then the per-game
  core endpoint (`sports.core.api.espn.com/…/events/{id}/competitions/{id}/odds`).
- Extract every provider item: provider name, home/away ML, spread, total,
  `lastUpdate`, fetched_at. Write raw rows FIRST (append-only parquet,
  `data_delivery/nfl_odds_history.parquet`) — raw-first means re-processing
  the de-vig logic never requires re-hitting ESPN.
- Pace 0.2s/request (≈3k requests, one evening); bounded-retry on transport
  errors only; on persistent failure: skip + count — never fabricate.
- Emits a **coverage report** (`data_delivery/nfl_odds_coverage_<date>.json`):
  per season × provider → % games present / % both-sides-complete. This
  converts every sampled claim above into a measured fact, per the repo's
  "coverage is reported, never fabricated" doctrine. It also re-pins the
  boundaries: consensus price start (2016-09 → 2017-09 gap), Pinnacle's
  exact season span, SB58.

### 2. `collect_closing_lines()` — daily production phase (master_pipeline step)

- Each run, for games whose event state is `post`: fetch the odds endpoint
  once (the served line on a completed game IS the close) and append to the
  history parquet. ~15–17 requests/day.
- Second snapshot at pick publication (~pre-game, morning-of): this is the
  **CLV enabler** — pick-time line vs close, both from the same provider.
  ~15 more requests/day. Total ~30/day, trivially free.
- PIT-safe by construction: post snapshots only after `state == post`; every
  row carries `fetched_at_utc`; failures degrade to a gap count, like every
  other pipeline phase.

### 3. `market_benchmark.py` — de-vig + benchmark (pure function of the parquet)

Per game, deterministic and re-runnable:

- **Filter:** drop Live-Odds twins, `Opening`, model feeds
  (numberfire/teamrankings/accuscore/Betradar), items with missing/zero sides;
  dedupe Caesars state variants into one family (keep family median).
- **De-vig:** two-way multiplicative, pinned as a config constant with
  provenance comment (method consistency > method choice); power-method
  published as a sensitivity column (Unabated's method — free cross-check).
- **Benchmark:** median of de-vigged home-win probabilities across complete
  real books, storing per game: `devig_home_median`, `n_books`, provider
  list, per-book holds, cross-book dispersion (max−min), `era`
  (`multi_book` ≥3 / `thin` 2 / `single_book` 1, computed — never
  hard-coded dates), `source="espn_core_api"`, `consensus_ml` when present
  (2017+ overlap validation), and NULL + gap-count when absent (SB58 class).
- **Artifact:** `data_delivery/nfl_market_benchmark_<date>.csv` per run;
  rolling history stays in the parquet (retention: never delete, repo rule).

## Grading integration (measurement only — no model changes in this phase)

- In the NFL walk-forward reporting: per-fold and pooled **market**
  logloss/Brier/AUC on the same grading rows, paired model−market delta with
  a **block-bootstrap SE (7-day blocks)**, and pick-time→close CLV on
  published picks — the deep dive's rec #1. Market *features* (rec #4) stay a
  separate owner decision; keep measurement and modeling decoupled so the
  benchmark's credibility is never entangled with a model change.
- Report per era segment and pooled-with-flag; the 2024–25 single-retail-book
  segment is disclosed as a lower-quality yardstick, never silently pooled.

## Tests (pytest pins, repo style)

1. De-vig math on known pairs (e.g., −150/+130 → fair 0.5917; hold math).
2. Filters: Live-Odds twin, `Opening`, model feed, zeroed Bovada-style item,
   missing consensus sides — each excluded with counts disclosed.
3. Caesars state-variant dedup → one family vote in the median.
4. Median + dispersion + era computation on a synthetic multi-book game;
   single-book era flags `single_book`.
5. PIT: post snapshot only for `state == post`; fetched_at recorded.
6. Hole handling: SB58-class game → NULL benchmark + gap count, never a
   fabricated price.
7. Coverage-report integrity: provider presence and completeness computed per
   game from raw rows (no assumptions from season labels).

## Sequencing & effort

- **Session A:** Phase 0 probe is subsumed by the backfill itself (it emits
  the coverage report); build `build_odds_history.py`, run backfill, review
  coverage + sanity checks (consensus 2017+ vs our median ≤ ~1pt agreement;
  Pinnacle subset on 2016 playoffs; hold distributions match the measured
  table).
- **Session B:** `market_benchmark.py` + benchmark artifact + tests.
- **Session C:** daily collector wired into `master_pipeline` (allowlist
  update in `check_production_graph.py`) + CLV columns + grading report
  integration + pytest.

Cost: $0 (ESPN core API, keyless, ~3k one-time + ~30/day requests at 0.2s
pacing — the same host family the pipelines already poll for scores).
Risk: undocumented endpoint, no SLA — mitigated by raw-first storage (one
fetch, forever reprocessable) and gap-count disclosure everywhere.
