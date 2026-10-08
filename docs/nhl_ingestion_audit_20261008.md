# NHL ingestion & feature-engineering audit — 2026-10-08

Deep-dive validating the NHL feature pipeline against actual game data,
covering the 2026-10-07 board outage, ingestion correctness, and
computational integrity of the feature engineering. Scratch harnesses
live in `.freebuff/` (untracked by policy): `nhl_official_vs_cache_audit.py`,
`nhl_full_feature_validation.py`, `nhl_leakage_check.py`,
`probe_nhl_oct7_board.py`.

## 1. Board bug: no slate predictions for Oct 7

**Symptom.** On 2026-10-08 the Today's Games board defaulted to Oct 6 and
offered no Oct 7 date, although `nhl_moneyline_v1_20261007.json` (3 priced
games) and `nhl_run_engine_markets_20261007.csv` were committed.

**Root cause.** `utils.valid_dates("nhl")` derived its dates from exactly
two sources: the strict current-slate resolver (which rejects any record
once `slate_date < today ET` — by design, a past slate is not a live
board) and the retained prediction history (which only gains a game day
when a later run republishes decided rows for it). After midnight ET the
Oct 7 slate fell out of the first source while history still ended at
Oct 6, so the date vanished from the rail — and the board's render gate
refuses any date outside `valid`. This is the same class as the NFL's
2026-09-30 dated-board regression and the backend's "late-September slate
never ingested" review.

**Fix (`frontend/utils.py`).** The NHL now folds its dated families
(`nhl_moneyline_v1_*.json` + `nhl_run_engine_markets_*.csv`) into the
date sources, the way the NFL folds `nfl_board_*.csv`; the existing 10-day
retention window still bounds the rail, so old slates age out unchanged.

**Verification.** `valid_dates("nhl")` → `['20261007', '20261006', ...]`;
a no-fixture AppTest render of `todays_games.py` for `selected_date=20261007`
raises nothing and serves the archive board (PIT@WSH/COL@WPG/EDM@ANA
cards, published 59/40/58% prices, run-engine strips). NFL/NBA rails
unchanged.

## 2. Ingestion vs the official feed

Independent re-parse of `api-web.nhle.com/v1/score/{date}` compared
field-by-field (teams, scores, SOG, start time, venue, state, outcome,
season, game type) against the pipeline's `score_v2_*` cache pages:

| window | dates | games | missing pages | missing finals | field mismatches |
|---|---|---|---|---|---|
| 2026-09-01..2026-10-08 | 38 | 130 | 10-07, 10-08 (provisional, by design) | 0 | 2 |
| historical sample (2024-11..2026-06) | 35 | 138 | 0 | 0 | 0 |

The two mismatches were the settled 2026-10-06 page holding
`away_sog` 28/28 while the feed's corrected values are 27/30 — a settled
page was **never** re-fetched, so the stale shots values would have fed
trailing `shots_for/against` features forever.

**Remediation (`nhl-backend/backend/ingestion.py`).**
`SCORE_REFRESH_DAYS = 3`: a settled score page stays refreshable for
three days after its game date (re-pulled like a provisional page and
re-cached when every game is final); a failed refresh falls back to the
settled cache so a network blip can never drop a recent date. Beyond the
window the permanent-settlement contract holds. The stale page itself was
re-pulled (now 27/30, matching the feed). Regression test:
`test_recent_settled_score_pages_refresh_for_feed_corrections`.

The Oct 7 page (3 finals) and Oct 8 page (unplayed) were correctly not
cached at audit time; the in-flight run fetches the Oct 7 page now that
all three games are final.

## 3. Feature-engineering computational correctness

- **From-scratch recompute** (`nhl_full_feature_validation.py`, 2,844
  decided games, no `features.py` logic reused): `win_pct`, `rest_days`,
  `back_to_back`, `ga_per_game`, `shots_for/against` trailing groups all
  match at 100.00% with max error 0.00000; every served
  `diff == home − away` identity is exact (max |err| = 0 across 14
  families, n≈2,824).
- **Point-in-time leakage** (`nhl_leakage_check.py`): rebuilding with the
  last 25 games removed leaves all 70 active feature columns
  byte-identical for the 2,819 shared games — no feature consumes the
  future tail.
- **Coverage**: the boxscore-only families (PP / faceoff / goalie) span
  the 93 cached boxscores (2026-05-01 → 2026-10-06) — 2.4% of history by
  design — but **100% measured across every serving window** in
  `run_engine_feature_coverage_20261007.csv` (250-game baseline, current,
  serving slate). Score-derived families cover ≥99.5% of history.

## 4. Known transient

The pipeline running during this audit (started 2026-10-07 23:16) loaded
pre-fix code, so its 20261008 run-engine artifacts may briefly reflect
the previous behavior; the next run picks up the committed refresh +
weight-source code. The NHL board render smoke shadows production
artifacts with fixtures and was therefore NOT re-run while the pipeline
was live; its 2026-10-08 failure list reproduced identically at clean
HEAD before any of these changes (pre-existing, rail-gated).
