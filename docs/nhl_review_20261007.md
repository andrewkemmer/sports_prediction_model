# NHL model review — 2026-10-07 (dashboard ↔ pooled-OOF parity + flat-metric diagnosis)

Scope: continue the model-performance review (mechanics, feature validation,
cleanup, integrity), **ensure the dashboard matches the production pooled OOF
run**, and explain why AUC / Log-loss have not noticeably improved across the
initiative. Ends with a smoke test and a committable parity guard.

## 1. Dashboard matches the production pooled OOF run ✅

The Model Calibration dashboard and the Model Monitor TOTAL row both render the
deployed blend's **pooled out-of-fold** scores over the grading population
(`fold_geometry.oof_regular`). Verified against the 20261007 artifacts:

| Dashboard KPI | Shown | `oof_regular` (production pooled OOF) |
|---|---|---|
| AUC-ROC       | 0.596   | 0.5959569634 (`n`=2465) |
| Brier         | 0.241   | 0.2409625056 |
| Log-loss      | 0.6747  | 0.6747233520 |
| Cal. Error    | 0.0091  | 0.0091339743 |

* Header `n = 2,465 games` **is** the grading count (`oof_regular.n`).
* The four KPI cards are exactly `oof_regular`, rounded at the emitter
  (auc/brier 3dp, logloss/ece 4dp). Raw → calibrated pairs show `X → X` because
  the calibrator was gated out (identity map).
* Reliability buckets partition the same 2,465 games (1714 + 664 + 87).
* `oof_all` (`n`=2652, AUC 0.5942, logloss 0.6755) is the *whole* OOF frame
  (adds provisional + postseason). It is exposed "so the split hides nothing"
  but is deliberately **not** the headline. The dashboard is on `oof_regular`
  by design, and the calibration + monitor artifacts agree block-for-block.

A new regression test locks this: `frontend/test_nhl_pooled_oof_parity.py`
asserts the dashboard `metrics` block == `oof_regular` (within emitter
rounding), `n_games` == grading n == bucket sum, and calibration == monitor.
It is non-vacuous: `oof_all` differs from `oof_regular` by 0.0018 AUC /
0.00077 logloss, well outside the 5e-4 tolerance, so a drift onto `oof_all`
would fail.

## 2. The two round-2 "record" FAILs were false alarms ✅

`.freebuff/nhl_round2_pressure.log` reported `markets records` and
`moneyline board records` mismatches (5201/5314) against its own
"independent recompute". That checker **undercounts** prior games. A clean
independent recompute from the raw cached scores (strictly-prior,
season-scoped, eligible game_types 2+3, decided games only) reproduces the
**pipeline's** records exactly:

| game | pipeline / my recompute | round-2 checker |
|---|---|---|
| 2026020051 VGK@SEA | home **2-1**, away **2-1** | 1-0 / 1-0 |
| 2024020194 OTT@BUF | home **4-8**, away **6-5** | 3-4 / 1-4 |

Comprehensive sweep over the served slate: **5,308 shipped market rows
checked, 0 mismatches.** The pipeline's `_record_frame` (season-scoped
groupby `[team, season]`, strictly-prior cumsum with the row's own flag
subtracted, ties/undecided excluded) is correct. `season` is start-year
(Jan 2025 → season 2024), so a season groups whole — no calendar-year split.
game_type 9/19/20 (All-Star/special) and preseason (1) are correctly
excluded. There is **no record-integrity defect**; the FAILs are an artifact
of the untracked throwaway checker.

## 3. Why AUC / Log-loss are flat — the model is at a feature-driven ceiling

The integrity/mechanics work established the **honest** metric; it did not and
could not create new predictive signal. Evidence:

* **Real skill, not a bug.** `oof_regular` AUC 0.596 is ~8 SE above 0.5
  (n=2465), beats the base-rate baselines (constant 0.5428 home-win:
  logloss 0.6887, Brier 0.2482 → model 0.6747 / 0.2410). Predictions align
  with outcomes (round-2 `history AUC == log oof_all`,
  `markets p_home_win == history calibrated` maxdiff 0.0).
* **Hard ceiling across every config.** `docs/nhl_model_audit_20261006/
  exploratory_experiments.csv` sweeps calibration method, engine-weight and
  tie-handling: all land at AUC 0.586–0.599 / logloss 0.674–0.692. Nothing
  beats ~0.599 / 0.6743.
* **The ML engine adds little over the Elo/derived baseline.** The
  engine-weight sweep is best at weight 0.0–0.1 (AUC ≈0.599) and *worse* at
  1.0 (AUC 0.587). The three ML members score 0.6756–0.6763 logloss; the
  deployed blend (0.6747) is only marginally better than the best member.
  So the remaining signal lives in **features**, not model mechanics.
* **Features are healthy** (not a suppressor): `feature_coverage` 204/204 OK
  (100% measured); weights concentrate on real hockey signal —
  `shots_against_per_game_diff` 13.5%, `back_to_back_diff` 8.1%,
  `pl_evo_d_diff` 6.4%, `elo_diff` 5.6%. (`is_home`, `is_playoffs`,
  `back_to_back_home/away` at 0% are correct: constant or redundant with
  their diffs.) Drift ALERTs are on low-weight features (`goalie_starts_*`
  0.46%).

**Conclusion:** the flat AUC/Log-loss is the model's true skill level, reached
by correctly removing leakage and making evaluation causal (which *lowers*
inflated pre-fix numbers toward the honest value). 0.596 AUC / 0.6747
log-loss is within the normal band for published NHL moneyline models
(≈0.58–0.62). Further gains require **new feature signal**, not more
mechanics/integrity work. Highest-leverage candidates: richer goalie /
pre-game availability (the PIT + injury channels already in place), schedule
& travel/zone-time features, and — if the goal is market-beating rather than
raw AUC — anchoring to closing-line implied probability.

## 4. Smoke tests

* `frontend`: `test_calibration_smoke`, `test_monitor_smoke`,
  `test_nhl_frontend`, `test_nhl_board_render_smoke`,
  `test_nhl_pooled_oof_parity` — pass (13 tests).
* `nhl-backend/backend`: 452 pass (all except the slow
  `test_run_engine_pit.py`), plus the focused integrity set
  (`test_input_semantics`, `test_causal_evaluation`, `test_fold_season_split`,
  `test_audit_model_quality`) — 57 pass. `test_run_engine_pit.py` runs
  separately (slow model-training suite).
