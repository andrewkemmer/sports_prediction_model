# MLB modeling deep dive — is the walk-forward structure the right one?

Date: 2026-10-11. Scope: the moneyline walk-forward training stack
(`mlb-backend/backend/training.py`, `calibration.py`, `feature_selection.py`,
`config.py` provenance, `master_pipeline.py` call sites) and the delivered
performance artifacts (`data_delivery/calibration_20261011.json`,
`mlb_pipeline_run_log.txt`). Question under review: **does the walk-forward
model structure promote high predictive signal — in-training logloss
minimization that translates into high AUC and low logloss / Brier on
out-of-fold predictions?**

## Verdict

**Yes — the fold architecture itself is correct and is among the best
practice setups for this translation. Keep it.** Expanding-window, strictly-prior,
weekly non-overlapping folds with *prequential* (fold-k-scored-by-folds-<k)
blend weights, calibration and boosting-round transfer is exactly the
structure that turns an in-training logloss objective into honest OOF
logloss/Brier/AUC. The MLB implementation is more disciplined than most
production systems: it closes subtleties that leak in weaker setups (fold's
own validation window selecting its own model, weights fitted on scored
labels, calibration double-application).

The gaps that hold the model back are **not** in the fold geometry. They are:

1. **The headline OOF number is not fully out-of-sample with respect to
   model selection.** Hyperparameters, roster and feature set were chosen by
   repeatedly watching the same pooled walk-forward that produces the
   headline (evidence below). The folds are honest; the *procedure* that
   produced the config is not accounted for in the reported estimate.
2. **The objective stops short of the betting objective.** The market is
   absent from both the feature universe and the grading artifact, so the
   one comparison that matters for a betting model — model probabilities vs
   vig-removed market probabilities on the same games — is never measured.
3. **No uncertainty on the headline, and pooling across 84 folds averages
   yesterday's model with today's.**

Current measured performance (delivered 2026-10-11, grading population
n = 6,879 regular-season OOF rows, 84 expanding weekly folds):

| Quantity | Model | Constant-base-rate baseline | Skill |
|---|---|---|---|
| AUC | 0.5750 | 0.5 | — |
| LogLoss | 0.6815 | 0.6910 (home view) | +0.0095 nats |
| Brier | 0.2443 | 0.2489 | +1.8% |
| ECE | 0.0078 | — | excellent *in-distribution* |

That is real but thin signal — the expected shape for MLB sides, where the
market is efficient and home-win base rate is 0.533. The model's probabilities
correctly squeeze toward the base rate (5,515 of 6,879 rows inside the
50–60% bucket); under a logloss objective with weak signal that squeeze is
optimal behavior, not a defect. One tail note: the 70–80% bucket
(n = 57, predicted 0.7223, actual 0.8421) suggests the model's rare
high-confidence calls have been *under*-confident — small n, but worth
tracking.

## What the structure gets right (keep all of it)

- **Geometry** (`training.walk_forward_splits`, training.py:561): validation
  window = one week; training = entire history strictly before the window
  start (`game_date < val_start`); windows non-overlapping and chronological;
  warm-up (`min_train_days=30`, master_pipeline.py:2826) keeps tiny-train
  folds out. No row ever trains on data at or after its validation window.
- **No fold scores itself.** XGBoost's early-stopped fit on the fold's own
  validation window is a *measurement only*; the shipped fold model is
  refit at the causal transfer of PRIOR folds' measured rounds, floored at
  a static budget (training.py:1690-1720, `_shipped_xgb_rounds`). LightGBM
  is a fixed-budget learner with no scored labels in fit. This kills the
  most common walk-forward leak — per-fold early stopping on the scoring
  window.
- **Prequential everything.** Blend weights are re-earned after each fold
  from strictly-prior OOF member logloss (rolling, simplex-constrained
  SLSQP in logit space, training.py:1243); the deployed bundle carries the
  last fold's earned vector (elasticnet 43.8% / lightgbm 30.2% /
  xgboost 26.0%). The calibrated column is fold-k corrected by a Platt map
  fitted on folds < k only; the final deployed map passes the same nested
  gate (fit on early prior evidence, >0.005 nats improvement required on
  the recent prior tail, then refit on all prior). The double-apply
  artifact (brier 0.2473 vs honest 0.2715) was caught and documented.
- **Honest populations.** Postseason rows and provisional (thin) folds are
  fit and scored but never grade; the log reconciles partition vs overlap
  line-by-line. Metrics never come from data the model saw.
- **Selection defenses that already exist.** RFE commits only on a
  logloss gain that also clears a standard-error noise bar (verified on
  synthetic pure-noise targets) and must reproduce on an alternate fold
  geometry before adoption; member/blend changes are gate-tested and
  frequently rejected (stacked meta-learner, calibration flip, MLP/RF
  removals); tuning rounds use a measured 3-seed noise floor.

## Where the signal chain breaks — the four real risks

### R1. Selection bias in the headline OOF (moderate, quantifiable)

The pooled walk-forward logloss is simultaneously (a) the tuning objective
and (b) the headline metric. `config.py` provenance documents this openly:
ElasticNet C/l1_ratio chosen by a grid "on the 84-fold walk-forward"
(config.py:294); LightGBM by Optuna "on the 44-fold walk-forward"
(config.py:574); XGB rounds floors chosen by a "37-fold walk-forward
replay" (config.py:556); the L5/L6/L8 re-tunes all measured on the same
expanding walk; roster changes (RF/MLP out) were graded on the same folds.
Each decision was individually well-defended, but the reported
0.6815/0.575 is the best-of-many-configs number on the folds that chose the
config — an optimistic estimate of what the *procedure* generalizes at.

Related: the "sealed 284 holdout" (2026-08-03→08-25 windows) has been
consulted in at least six different decisions (config.py:576, 582, 592,
747; calibration flip; stack ablation). It is no longer sealed — it is a
second grading set under selection fatigue. Newer practice in the repo is
better (the 2026-10-10 budget review used "verify window folds 70–82, never
used for selection") — formalize that and retire the August windows.

**Magnitude estimate:** one nested replay (tune inside folds < k, evaluate
on fold k, for the last ~20 folds) would measure the optimism directly. If
nested logloss lands at ~0.683–0.684 vs pooled 0.6815, the bias is ~0.002–
0.003 nats — material relative to the model's entire 0.0095-nats skill, but
not fatal.

### R2. The betting objective is unmeasured (highest-impact gap)

`MONEYLINE_FEATURE_COLS` (training.py:132-470) contains no market quantity —
no implied probability, no line, no vig-removed price. Worse: **the lines
that do flow through the pipeline are synthetic.** Production Step 2 calls
generate_synthetic_market_lines (master_pipeline.py:2333; data_ingestion.py:590
fabricates "fair" probability from a fixed home-advantage term plus Gaussian
noise and a random vig), so the delivered `moneyline_home/away` and the
`edge_home/edge_away` columns (master_pipeline.py:2505) are computed from
fabricated prices. The PIT as-of merge plumbing exists and is correct
(data_ingestion.attach_market_lines rejects lines posted at/after first
pitch) — it has simply never been fed a real odds feed. Two consequences:

- **No head-to-head grading.** The calibration artifact reports model vs
  realized outcomes, never model vs market on the same rows. For a betting
  model the decision-relevant metrics are: market logloss/Brier/AUC on the
  grading population, the paired model−market delta with a block-bootstrap
  SE, and closing-line value on published picks. A model at 0.6815 is, by
  standard MLB market numbers (~0.675–0.678 vig-removed logloss),
  probably slightly *behind* the market's own probabilities — and the repo
  currently has no artifact that would reveal either that fact or a future
  improvement past it.
- **The strongest available feature is unused.** Vig-removed implied
  probability typically carries AUC ~0.60 on its own. Feeding it (or the
  model's logit residual around it) is the standard professional
  construction: your number becomes a *correction to* the market rather
  than an independent estimate competing with it. Expected effect if the
  sports-side features carry any genuine incremental information: pooled
  logloss toward ~0.676–0.679, AUC up, and the residual signal becomes
  directly auditable. The cost is philosophical — the product stops being a
  "pure sports model" and the market-mirror failure mode must be guarded
  (RFE gate + a monitored model−market spread floor). This is an owner
  decision, not a defect fix.

### R3. No uncertainty, and pooled-all-folds dilutes the deployed model

Headline AUC/Brier/logloss are point estimates over 84 folds treating ~6,879
correlated games (same teams, overlapping rotations inside a week) as iid.
Fold-to-fold spread in the delivered log is wide (fold 80 AUC 0.6896, fold 81
0.6290, early folds far lower). Two improvements:

- **Block bootstrap by fold (7-day blocks) for CIs** on every headline
  metric. The repo already measures a 3-seed *tuning* noise floor
  (0.00144 pooled OOF, config.py:605) — the headline deserves the same
  treatment so 0.001-level claims are interpretable.
- **A trailing-window headline next to pooled-all.** Pooled grading averages
  the 2025 model (trained on ~40% of the data) with the 2026 model. The
  deployed bundle is only the newest. Publishing "pooled all" +
  "trailing 26 folds" (≈2 quarters) makes the deployed model's expected
  performance legible without changing the honest all-history number.
  (`max_eval_folds` exists; make the trailing view a first-class artifact
  rather than a manual knob.)

### R4. Early-fold dilution and calibration tails (minor)

- Folds trained on ~30 days of history contribute near-base-rate predictions
  to the grading pool, mechanically compressing pooled AUC. This is honest
  for "the whole procedure from cold start" but irrelevant to the deployed
  model. Solved by the R3 trailing view; alternatively raise the *grading*
  warm-up (keep fitting from day 30) — but the trailing view is the cleaner
  estimand.
- Calibration is excellent in-distribution (ECE 0.0078) largely because
  probabilities live near 0.53–0.55. The interesting calibration region is
  the 60%+ tail where n is small (1,307 and 57). Keep the existing
  prequential gate; add tail-bucket monitoring (favorite-view reliability
  for ≥60% calls) so a confidence-creep regression is visible before it
  costs bets. The run-engine block's 273/6,140 degenerate Platt fits
  (negative slopes, correctly falling back to identity) are the expected
  signature of near-zero-signal windows — no action.

## Ranked recommendations

1. **Grade against the market** (R2, measurement): wire a real odds feed
   into the existing as-of merge (replacing the synthetic generator), add
   vig-removed market implied probabilities to the grading artifact —
   per-fold and pooled market logloss/Brier/AUC, paired model−market delta
   with a block-bootstrap SE, and CLV tracking on published picks. Pure
   measurement, no modeling risk; it converts every future gate from
   "better than our past" into "better than the price". **Feasibility
   verified 2026-10-11:** ESPN's core API
   (sports.core.api.espn.com/…/competitions/{id}/odds) serves frozen
   closing lines for completed MLB games free and keyless — 15/15 games on
   2026-09-15 (DraftKings) and 10/10 on 2024-04-25 (ESPN BET) sampled —
   covering the entire graded frame for backfill plus one snapshot per
   game for the daily close, on the same ESPN dependency the pipeline
   already has for scores. It is a single US soft book's close, not the
   sharp Pinnacle close; provider varies by season (ESPN BET through
   2025 — where each game ALSO carries an "ESPN BET - Live Odds" twin
   that zeroes out post-game and must be filtered; DraftKings-only from
   2026, single item per game).
2. **Formalize a verification block** (R1): a rolling never-tuned window
   (e.g. most recent 8 folds + an alternate cadence-5 geometry) that no
   Optuna search, grid, roster or RFE decision may ever see; retire the
   burned August sealed windows. Then run ONE nested replay to quantify
   current selection optimism and publish it next to the headline.
3. **Headline + trailing metrics with CIs** (R3): block-bootstrap bands on
   pooled and trailing-26-fold AUC/logloss/Brier in the calibration
   artifact and dashboard.
4. **Market-anchored member or feature** (R2, modeling — owner decision):
   either add vig-removed implied prob as a feature, or add a dedicated
   "market residual" member to the roster. Either routes through the
   existing RFE/gate machinery, so adoption is evidence-gated, not
   conviction-gated.
5. **Margin→probability member** (roster diversity): the run engine already
   produces OOF run-margin distributions on the same canonical folds; a
   member that maps predicted run margin to win probability adds genuine
   structural diversity (the blend's linear member currently dominates at
   43.8%, telling us the signal is mostly additive-linear). Past roster
   additions failed gates on learner diversity alone — this one differs
   because it shares no functional form with the current three.
6. **Do not**: chase focal/class-weighted losses (they distort probability
   targets and would trade the model's best property — calibration — for
   rank metrics); deepen trees (the 2026-10-10 budget review already
   located the capacity frontier and found it flat); add members via the
   same pooled-OOF gates (the stack ablation's pooled-gain/sealed-loss
   inversion is the pattern).

## Bottom line

The walk-forward structure is the right spine and is already stronger than
industry default — the honest-OOF machinery is not where predictive signal
is being lost. Signal is lost at the *edges of the objective*: the model is
tuned against its own grading folds (inflating the headline slightly) and
graded against nothing but itself (so its betting-relevant value is
unmeasured). Fix the measurement first (market benchmark + verification
block + CIs); those two artifacts will tell you, within one retrain cycle,
whether the next modeling bet should be market-anchoring, roster diversity,
or simply more feature work — with numbers instead of conviction.
