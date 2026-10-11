# Pipeline run-log review — MLB / NFL / NHL / NBA — 2026-10-11

**Scope:** the four tracked run logs pushed with the pipeline deliveries:
`mlb-backend/data_delivery/mlb_pipeline_run_log.txt` (456 lines),
`nfl-backend/data_delivery/nfl_pipeline_run_log.txt` (472),
`nhl-backend/data_delivery/nhl_pipeline_run_log.txt` (408),
`nba-backend/data_delivery/nba_pipeline_run_log.txt` (254).

**Verdict:** all four runs completed (`DONE`) with **zero errors, exceptions or
tracebacks** (grep count 0 per log). Every remaining warning is classified
below; none indicates a computational defect requiring code remediation.

## Disposition table

| Log | Entry | Disposition | Action |
|---|---|---|---|
| MLB | `WARNING MLB_END_DATE=2026-10-10 is stale … extended to 2026-10-11` | Self-heal guard working as designed (`mlb-backend/backend/config.py` stale-pin guard): window extended so the daily slate cannot freeze. The pin lives in the **Kaggle notebook** (Kaggle-owned, T7 2026-10-06), outside this repo; behavior pinned by `test_log_review_20261006`. | Operator action on the notebook (re-date or unset), per the log's own instruction. No code change — same disposition as `mlb_log_review_20261010.md`. |
| MLB | `SP staleness gate: 2400 appearance rows flagged stale`; `SP slate staleness gate: 146 pitcher(s) … SP lines withheld` | Gate working: stale-prior pitcher form is withheld rather than served; disclosed per slate. | None. |
| MLB | Feature coverage `STRUCTURAL (declared missing-value policy, not starvation)` on `wind_advantage_flyball_factor` / `air_density_velocity_boost` (48 closed-roof policy zeros + 15 open-air NULLs gated by missing `sp_*_diff`) | Declared policy, exhaustively enumerated; every open-air row with a valid input is observed. | None. |
| MLB | Log tail references remote tip `6a1fdd4e80` | The notebook's sync-time remote tip, not a repository discrepancy: `git fetch` confirms local `main` == `origin/main` at review time. | None. |
| NFL | `retention: pruned 11 stale artifact(s)` (2026-09-29 vintage) | Retention policy operating normally; run completed `DONE — 238 artifacts in 1594s`. | None. |
| NHL | `coverage [baseline/current/serving slate]: pl_ppo_l_* / pl_ppo_r_* carry default-filled values counted as UNMEASURED — a warm gap, not a warm-up` | Coverage gate (`nhl-backend/backend/master_pipeline.py:103`) working as designed: default-filled values on regular-season rows are counted UNMEASURED and disclosed per population. This is the position-pool power-play-offense family (`pl_ppo_{c,l,r,d}_*`); the L/R pools lack recent observations. Upstream data-coverage limitation, not a join or computation failure — `audit_player_availability.py` verifies the pool entry exclusions bind point-in-time. | None in code; gaps close as pools accumulate observations. Disclosure is the correct remediation arm. |
| NBA | `2 finished game(s) have no season-log player lines … excluded from training (postponed, neutral-site and all-star games, not a join failure)` | Disclosed structural exclusion with named games (2024-12-17 MIL@OKC, 2025-12-16 SAS@NYK). | None. |
| NBA | `pl_rapm coverage gap: 195/3461 decided rows lack position-rapm features … early-season rows are structural, not a build failure` | Declared by the coverage gate; 104 rows inside the 10-day pool lookback, 19 within 30-day recency, 72 beyond — consistent with the `nba-prod-v2.8-rapm-strict-pregame` contract's strictly-prior pools. | None. |
| NBA | `✅ status ok, 3332.77s`, 10/10 phases | Healthy run. | None. |

## Remediation summary

- **No code remediation was warranted**: every warning is a working guard, a
  working disclosure gate, or an operator-owned pin. Intervening on the gates
  would suppress accurate disclosure rather than fix anything.
- The one open action is external: re-date or unset the `MLB_END_DATE` pin in
  the Kaggle notebook (recurring since 2026-10-07; self-heal keeps slates
  unfrozen meanwhile).

## Cross-checks performed

- Error/exception/traceback scan of all four logs: 0 hits each.
- Warning/anomaly scan with context: all hits classified above.
- `git fetch` + `git status -sb`: local `main` in sync with `origin/main`; no
  unreviewed remote log state.
- Coverage-gate semantics confirmed against `nhl-backend/backend/master_pipeline.py`
  and `monitoring.py` (`MEASURED` counts real observations only; default fills
  are not silently counted as measured).
