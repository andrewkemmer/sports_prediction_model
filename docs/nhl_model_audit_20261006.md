# NHL model-quality deep dive

**Reviewed artifact vintage:** 2026-10-06 (generated 2026-10-07 UTC).
**Scope:** market-free NHL moneyline ensemble, goal-distribution head, feature construction, calibration, walk-forward geometry, model explanations, and selection governance.
**Decision:** repair semantic and evaluation defects before another hyperparameter sweep. No production parameters, models, features, delivery artifacts, or notebook were changed by this audit.

## Executive verdict

The current ensemble has useful but modest discrimination: published regular-season AUC **0.59886**, log loss **0.67447**, Brier **0.24083**, on **2,418** grading games. However, the published blend is **not a fully out-of-sample ensemble estimate**: final weights were optimized against those same outcomes and then applied retrospectively. The causal rolling-blend log loss in the production log is **0.67549**, approximately **0.00102** worse. That gap combines retrospective weight fitting and the cost of learning weights over time; it is not a pure estimate of selection optimism or a deployable improvement.

The strongest opportunities are upstream of model tuning:

1. **Restore genuinely causal ensemble evaluation and train/serve parity.** Final LightGBM categorical handling and XGBoost round budgets differ from the evaluated fold models.
2. **Correct Elo, power-play, and faceoff semantics.** These are verified defects, not speculative features.
3. **Make goalies and player availability representative of what is known before each game.** A reproducible trade scenario selects a departed goalie; most historical injury coverage is absent.
4. **Repair run-engine metrics and probability contracts.** Current per-line log losses are corrupted by reversed arguments; calibrated grids violate cross-line monotonicity.
5. **Then test NHL-native strength, defensive, special-teams, schedule, and lineup features in a nested temporal experiment.** Do not tune around corrupted inputs.

No new policy demonstrated a robust AUC/log-loss win in the available exploratory experiments. Small blend gains are insufficient to justify adoption; generic home-space calibration also failed on the full population.

## 1. Evidence and reproducibility

### Deliverables

- [Read-only audit CLI](../nhl-backend/backend/audit_model_quality.py)
- [Audit tests](../nhl-backend/backend/test_audit_model_quality.py)
- [Machine-readable diagnostics](nhl_model_audit_20261006/diagnostics.json)
- [Performance slices](nhl_model_audit_20261006/metrics_by_slice.csv)
- [Per-model feature weights](nhl_model_audit_20261006/feature_weights.csv)
- [Exploratory experiments](nhl_model_audit_20261006/exploratory_experiments.csv)
- [Official API semantic comparison](nhl_model_audit_20261006/official_api_semantic_sample.json)
- [Full test results](nhl_model_audit_20261006/test_results.txt)

```bash
PYTHONUTF8=1 python nhl-backend/backend/audit_model_quality.py \
  --date 20261006 --output docs/nhl_model_audit_20261006
PYTHONUTF8=1 python -m pytest nhl-backend/backend/ -q
```

The CLI performs no network pulls, refits, adoption, or production writes. It reconciles reconstructed folds against published counts and headline log loss, rejects malformed targets/probabilities, inspects bundle structure, and hashes its main inputs. The official API sample was fetched separately and saved.

### Available versus missing evidence

Available: 2,635 game-level blend predictions; 2,635 OOF distribution rows plus nine slate rows; 59-fold summaries; member aggregate metrics; final bundle; feature coverage/drift; older RFE correlation evidence; injury archive; production run log.

**Missing:** raw settled/boxscore cache, current training feature matrix, player-rating input archive, member-level OOF predictions, per-fold earned weight vectors, and per-fold feature importances. Those diagnostic stores are local/gitignored and absent from this checkout. Consequently:

- Current full feature correlations, VIF/condition numbers, member residual correlations, grouped OOF permutation importance, and full refit ablations **could not be measured**.
- Bundle gain/coefficient rankings below are **training-model diagnostics, not causal predictive importance**.
- The exploratory calibration/blend tests inherit retrospectively selected base weights. They are not unbiased policy backtests.
- Bundle load produced version warnings: production XGBoost 3.2.0/sklearn 1.6.1 versus local 3.4.0/1.9.0, with other stack differences. No predictions were made using the cross-version loaded bundle.

## 2. Current models and blends

| Model/view | Final weight | AUC | Log loss | Brier | Population |
|---|---:|---:|---:|---:|---:|
| XGBoost | 34.61% | 0.59453 | 0.67620 | 0.24163 | 2,418 grading games |
| LightGBM | 22.45% | 0.58892 | 0.67751 | 0.24230 | same |
| Elastic net | 42.94% | 0.59080 | 0.67644 | 0.24179 | same |
| Retrospective final-weight blend | — | 0.59886 | 0.67447 | 0.24083 | same |
| Gated prequential calibration of that blend | — | 0.59781 | 0.67485 | 0.24099 | same |
| Causal rolling-weight blend | — | unavailable | **0.67549** | unavailable | production log, rounded |

**Interpretation:** XGBoost is the best individual ranking model, but elastic net earns the largest log-loss weight. That is plausible: optimal mixture weights depend on conditional complementarity, not simply member AUC or standalone log loss. LightGBM's weaker standalone score does not prove it should be removed. Member-level residual correlation and causal leave-one-member-out experiments are needed first.

The retrospective blend gains 0.00433 AUC and 0.00173 log loss over XGBoost, but its weights are selected on the grading population. The causal log-loss benefit over XGBoost is only approximately 0.00071 from available log evidence. Do not count the full retrospective improvement as proven generalization.

Historical final weights changed sharply across artifact vintages, including near-total XGBoost allocation in older runs. Those runs also changed feature/evaluation protocols. They are **not clean temporal weight-stability evidence** and cannot be interpreted as natural day-to-day drift.

### Recommended blend experiments

After obtaining honest member OOF stores, preregister a small comparison set:

- Fixed thirds versus expanding causal SLSQP (current causal baseline).
- Simplex weights regularized toward thirds or the prior weight vector; tune strength inside earlier chronological folds only.
- Expanding versus trailing 500/1,000/1,500-game or exponentially decayed meta-training windows, with minimum evidence and shrinkage in early folds.
- Probability averaging versus logit averaging. Neither is inherently superior; pooling space changes confidence and must be evaluated.
- Regularized nonnegative logistic stacking with intercept/temperature, tested nested rather than fitted and scored on the same member OOF rows.

Optimize proper scoring loss; assess AUC separately as a co-primary outcome. Do not force diversity floors or weight caps without evidence. AUC-only optimization is not a substitute for calibration.

## 3. Walk-forward and evaluation integrity

### What is sound

[Fold construction](../nhl-backend/backend/folds.py#L109-L185) uses canonical ordering, non-overlapping seven-observed-date windows, expanding training before the validation date, and a 30-observed-date initial warmup. Median imputation/scaling are fitted on training only. XGBoost's early-stop probe informs strictly later folds rather than selecting its own scored model. These are important strengths.

### A. Retrospective meta-learner evaluation — confirmed

[The end of the fold walk](../nhl-backend/backend/moneyline.py#L440-L476) preserves `p_ensemble_causal`, then replaces the headline `p_ensemble` with the final weights optimized over all grading outcomes. Base learners are OOF; the final weight vector is not OOF relative to those rows.

**Solution:** publish three explicitly separate views: causal pipeline backtest; retrospective final-bundle replay/diagnostic; timestamped live forecast performance. Use causal predictions for selection and headline generalization claims. Keep final all-prior weights for future serving. For a frozen-weight candidate, earn weights on an earlier meta-training interval and score untouched later intervals.

### B. Calibration gates are only locally causal — confirmed

[Calibration](../nhl-backend/backend/master_pipeline.py#L510-L587) uses the retrospective blend as its input. Prior-fold-only calibrator fitting cannot undo that upstream selection dependence. Additionally, `prequential_fold_calibrators` accepts no grading mask: it pools prior postseason/provisional rows, while the final serving map uses grading rows only.

**Solution:** base the evaluation on causal blend inputs, pass eligibility explicitly, and align the calibration-training population with serving policy. Decide transparently whether all regular games or a restricted subset informs calibration.

### C. Evaluated models do not match the final refit — confirmed

[Fold fit](../nhl-backend/backend/moneyline.py#L234-L285) versus [final refit](../nhl-backend/backend/moneyline.py#L603-L625):

- XGBoost folds use a prior-fold budget; the final model receives no explicit `n_estimators` and ships **100** trees. The production probe ledger's median is **127**, range **1–871**. A stable round-policy comparison is warranted, but the immediate requirement is that the final refit follow the evaluated policy.
- LightGBM folds explicitly pass the categorical column names; the final refit does not. The final bundle shows team metadata with no categorical values. The serving model therefore does not implement the categorical treatment whose OOF evidence earns its weight.
- XGBoost's `enable_categorical=True` is insufficient: its view uses integer columns. The persisted feature types contain **zero categorical features**, with both team IDs typed `int`. Alphabetical IDs become ordinal numeric thresholds.

**Solution:** one shared fit path for folds/refit, explicit learned/pinned round policy, stable pandas categorical schema, and assertions on actual booster feature types/categorical splits. Unknown category handling must be identical at fit and serve.

### D. Thin-window filtering changes the target population

59 folds score 2,635 games. Headline metrics grade 2,418 games. There are 168 playoff rows and 206 provisional rows, **overlapping categories**, so those counts must not be summed as disjoint exclusions. Exactly 49 regular-season games are outside the grading set; the 35 current-season games are all provisional and do not inform the headline.

The source comments justify exclusion partly by claiming a seven-game week gets the same optimizer vote as a 70-game week. The actual SLSQP objective is a **pooled per-game mean**; seven games get one-tenth the mass, not equal fold weight. Regime separation is defensible; that weighting explanation is not.

**Solution:** keep separate regular/playoff reporting, but evaluate all eligible regular games, including season ramps. If thinness motivates estimator shrinkage, shrink the update rather than erase the games from the target population. State the estimand before selecting exclusions.

### E. Observed-date cadence and sealed-tail power

Seven observed dates are not seven elapsed days. Windows can cross long gaps or season boundaries. Expanding fitting includes playoffs even though grading excludes them. The 21-calendar-day sealed distribution tail contains **35 games**, all October 1–5. It is too small for reliable tuning admission, particularly per-line push-excluded subsets.

**Solution:** retain exact PIT cutoffs, compare calendar cadence and season-aware windows, and define sealed windows using both minimum duration and minimum game count (e.g. at least 300–500 regular games). Add rolling-origin evaluation across several seasons, with genuinely unused future data as the final gate. Previously inspected historical tails are no longer pristine holdouts.

## 4. Wrong or suboptimal feature data

### A. Elo home-advantage sign — confirmed semantic defect

[The expectation formula](../nhl-backend/backend/features.py#L123-L156) applies positive advantage in the wrong direction. Equal ratings imply home expectation **0.407534** instead of **0.592466**. A synthetic home win updates its rating by 11.8493 rather than 8.1507.

**Solution:** use `E_home = 1/(1 + 10**((R_away - R_home - H)/400))`; away expectation is its complement. Rebuild the full rating timeline and re-evaluate all models/weights; do not just flip a stored feature. Assert equal-rating expectations, symmetry at H=0, and rating conservation. Historical/serving season reversion also needs parity: the pending-slate path maps the last settled ratings directly, whereas historical Elo reverts at a season transition.

After correctness, tune NHL-specific K, home advantage, and offseason carryover nested in time. Consider a rating that separates regulation and OT/shootout information rather than treating every one-goal outcome as equally informative.

### B. Power-play denominator — verified against official data

[Ingestion](../nhl-backend/backend/ingestion.py#L705-L762) equates goalie power-play shots faced with team power-play opportunities. Those are different quantities. It also reads only one goalie, missing relief-shot volume if a shooting-rate feature were intended.

Official game **2024020194**:

| Statistic | Current parser | Official team statistic |
|---|---:|---:|
| BUF PP opportunities | 6 | **4** (`2/4`) |
| OTT PP opportunities | 6 | **3** (`0/3`) |
| BUF PP conversion | 2/6 = 33.3% | **2/4 = 50.0%** |

The existing official API's `/right-rail` response provides team `powerPlay` values; evidence URLs and observed payload fields are preserved in the sample. Cache these historical postgame facts and shift them before target games. Use pooled goals/opportunities, and add opportunity volume/shrunken rates to distinguish a one-chance success from sustained quality. Preserve a separately named PP shooting percentage only if it proves useful.

### C. Faceoff rate — verified aggregation error

[Parser aggregation](../nhl-backend/backend/ingestion.py#L787-L824) averages individual skater percentages, including zero-valued nonparticipants. For the same game: BUF parser **15.72%**, official **43.55%**; OTT parser **26.11%**, official **56.45%**. Current full-history per-side means near 18% are not plausible team faceoff win rates.

**Solution:** ingest team wins/attempts (`27/62`, `35/62`) or the official team rate; pool counts across the trailing window. Do not average player percentages without attempts. Add actual-API fixtures: current synthetic parser tests encode the incorrect denominator and cannot validate hockey semantics.

### D. Goalie identity/workload — confirmed synthetic defect plus information gap

[Expected-starter logic](../nhl-backend/backend/features.py#L808-L843) scans goalies who have ever started for a team, then ranks them by **all-team** starts that season. No team-specific, as-of current roster gate exists in this vote. A synthetic goalie with one ANA start and four later BOS starts defeats ANA's current goalie with two ANA starts; the selected player is the departed goalie with five global starts.

The production rule also always picks the season workhorse rather than estimating the probability of a backup start. Recent goalie-diff coverage is only **70%** versus about 98% across full history. Early-season quality resets to missing despite available prior-season performance.

**Solution:** timestamped confirmed-start announcements when available; otherwise a team/roster-aware probabilistic starter model using recent workload, goalie rest, back-to-back schedule, injuries, and prior starter rotation. Average candidate quality over starter probabilities; report uncertainty. Keep multi-season goalie skill with prior-season shrinkage while modeling seasonal workload separately. Quality upgrades should use GSAx per shot with strong exposure shrinkage if the source supports a PIT history, rather than a three-start EWM of raw SV%/GAA alone.

### E. Player availability and roster provenance — confirmed gap

The run log explicitly reports the frozen pregame-availability channel **unpopulated**. The carried injury archive starts **2026-09-28**, while OOF evaluation starts in November 2024. Thus most historical player-pool ratings do not represent known game-day availability. Unknown is correctly not called healthy, but missing knowledge still leaves absent players in the historical candidate pool.

[Pool fallback priors](../nhl-backend/backend/features.py#L956-L981) are fixed constants documented as remeasured from the 2026 run's distribution. This is a limited empirical-prior lookahead risk when used for earlier missing pools, not a demonstrated large outcome leak. Player-rating shrinkage itself uses strictly prior league evidence and completed seasons; that good discipline should extend to pool fallbacks.

**Solution:** frozen timestamped pregame events and as-of roster membership; never infer historical injuries from a player's eventual nonappearance. Until coverage exists, disclose separate availability-aware and unknown-availability evaluations. Record per-team missingness, fallback rate, roster source/capture time, rating age, eligible players, total projected ice, and removals. Non-null `pl_*` is **not** proof of measured coverage: defaults can yield 100% non-null output.

### F. Rest, season transitions, and stale form — observed design weakness

Rest-day differences have standard deviation **11.28 days**; per-side rest standard deviations exceed 16 days because offseason gaps remain uncapped. The elasticity of a 100-day offseason is not the same as one extra day of recovery. The linear rest coefficient is negative and small partly because that broad scale dilutes ordinary rest variation.

**Solution:** nonlinear/binned rest, a modest capped recovery range, separate season-opener/offseason indicators, back-to-back plus three-in-four/four-in-six density, travel distance/time-zone change, and prior game's OT workload. Tune short/medium/long form windows by family rather than assigning many noisy quantities the same five-game/three-game-half-life memory. Avoid simply deleting inconvenient October errors.

### G. Multi-game lookahead can contaminate schedule features — synthetic reproduction

The pending-slate builder combines settled history with every pending row. `team_events` labels rows with null scores as `team_win=0.5`, then the trailing ladder includes them as if an outcome existed. With one settled ANA win followed by two pending ANA games, entering win percentage falls from **1.00** for the first pending game to **0.75** for the second solely because the earlier unplayed game contributes an artificial half-win. Rest also references an unplayed scheduled game. This matters when the serving horizon includes several upcoming games for the same team; it does not imply every single-day card is affected.

**Solution:** separate observed-outcome history from planned schedule history. Freeze performance features at each prediction's available-data cutoff; use future schedule rows only for explicitly planned workload features. Add a future-unplayed-row invariance test and verify all scheduled targets resolve from the same settled cutoff without fabricated results.

### H. Coverage/drift can conceal defects

[Cold-start classification](../nhl-backend/backend/monitoring.py#L258-L282) determines debut from each sliced window's own first-seen teams, not the full historical timeline or goalie-specific season exposure. Every team in the latest 60-game window existed earlier. The reported 16 cold goalie-diff nulls therefore cannot be assumed genuinely cold solely from that label.

Current drift mixes 25 playoff games and a new-season ramp. Goalie-start alerts largely reflect seasonal resets. Fix provenance first, then compare season/regime-matched windows and separate missingness from value-distribution drift. Mean-shift gating can also miss variance-only/tail changes.

## 5. Feature weights, correlations, and model-specific implications

### Actual feature routing

68 master numeric features; tree width **70** including two IDs. Linear width **40**, not a purely diff-only set: all **24 player home/away/diff** features survive routing. The exclusion set removes raw team levels but not player levels.

### Top final-model weights

| Elastic net: standardized log-odds coefficient | Value |
|---|---:|
| shots_against_per_game_diff | −0.17203 |
| back_to_back_diff | −0.12839 |
| pl_evo_d_diff | +0.08209 |
| pl_evo_l_home | +0.08150 |
| elo_diff | +0.07080 |
| goalie_starts_diff | −0.05355 |
| pl_evo_c_home | +0.05199 |
| pl_evo_r_diff | +0.05173 |

23 of 40 linear coefficients are nonzero. Win percentage, net-goal EWM, goal-share EWM, goalie GAA, playoff flag, and constant `is_home` are zero in the final fit. That does not prove they have no OOF value; regularization reallocates correlated signal. Goalie SV% is slightly negative, a warning to investigate identity/collinearity—not evidence that poorer goalies cause winning.

XGBoost's top total-gain feature is shots-against difference (**5.90%**), followed by Elo difference (**3.68%**) and several EVO player features. LightGBM concentrates **13.57%** of total gain on shots-against difference, followed by EVO right-wing home (**4.87%**), EVO defense away (**4.63%**), and shots-against home (**4.00%**). Player features comprise roughly **41–42%** of total tree gain and **43.4%** of absolute linear coefficients. These shares describe model reliance, not unique predictive contribution, and use incomparable importance definitions across model families.

### Correlation/redundancy findings

**Exact algebraic redundancy, current contract:** for each player situation/position, `diff = home − away`; eight triples are linearly dependent whenever measured. Linear regularization then chooses among arbitrary equivalent representations. A diff-only linear view would reduce width from 40 to 24; test that against matchup diff plus sum/strength level where a genuine nonlinearity is expected. Do not interpret unstable individual coefficients as independent effects.

**Historical measured correlations, NOT a current full-matrix measurement:** the older 2,792-row RFE trace reports:

- net-goal EWM difference versus goal-share difference: **r=0.9142**;
- served net-goal EWM versus candidate net-goal EWM: **r=1.0000**;
- shots-for rolling difference versus candidate SOG rolling difference: **r=1.0000**;
- shots-for rolling versus SOG EWM difference: **r=0.9338**;
- faceoff flat versus EWM difference: **r=0.9474**.

Some are duplicate definitions, not new signals. Remove aliases from the candidate search before spending trials. High correlation alone is not enough to drop a feature: nonlinear learners may use levels and differences differently.

**Required next diagnostics:** per-training-fold Pearson/Spearman correlations with pair counts; grouped correlation clusters; missingness correlations; rank/condition number after imputation; fold-specific coefficient sign stability; grouped blocked OOF permutation and leave-family-out retraining. Permute home/away/diff families together to avoid generating impossible matchups. Record correlations by season rather than fit screening thresholds on future validation data.

### Explanation limitations

[SHAP implementation](../nhl-backend/backend/shap_explain.py#L1-L16) omits the **42.94%** linear member and renormalizes tree weights to 100%. Those files explain a tree subensemble, not the full deployed probability. The tree renormalization scales contributions by about 1/0.5706 = **1.75** versus their actual ensemble weight.

Add exact linear contributions in the transformed space and preserve original member weights, intercept/base values, perspective, and calibration scope. Verify base plus summed contributions reconstructs the blended raw logit. Binary class-output normalization also needs an additivity check; subtracting symmetric class outputs can double a logit contribution depending on the explainer version. Do not use these current files as evidence for full-ensemble feature removal.

## 6. Temporal stability and calibration

| Grading season | n | AUC | Log loss | Actual home-win rate | Mean p(home) |
|---|---:|---:|---:|---:|---:|
| 2024–25 | 1,113 | 0.60999 | 0.66752 | 56.42% | 54.96% |
| 2025–26 | 1,305 | 0.59060 | 0.68040 | 52.34% | 55.08% |

Discrimination falls about 0.0194 and log loss worsens 0.0129 across these slices; uncertainty and training-size differences preclude declaring a regime break. The second season has **2.74 percentage points of home overprediction**, so time-varying home intercept/season adjustment deserves a preregistered experiment after correcting Elo and parity.

The existing calibration gate accepted only folds 18–20, changing 164 predictions. Its aggregate log-loss delta is **+0.000374**, with conditional four-calendar-week moving-block bootstrap 95% interval **[−0.000216, +0.001554]**. The observed degradation is real in this artifact, but is not statistically decisive; this interval is conditional on selected inputs and does not adjust for prior tuning.

A fixed prior-grading-only home-space Platt candidate (minimum 500 prior rows, C=1) yielded:

- Full grading set: AUC **0.59731**, loss **0.67481** versus **0.59886 / 0.67447** raw: **do not adopt**.
- Last 500 grading games: loss 0.67527 → 0.67461, but AUC 0.60657 → 0.60563: interesting temporal context, not a full win.

A single global increasing map preserves rankings except for ties introduced by flooring; separate maps across folds can reorder pooled predictions. Favored-space calibration cannot freely correct asymmetric home/away bias, and flooring can collapse near-even games. Test identity, regularized temperature, regularized home-space Platt, and home/season intercept adjustments inside a common causal pipeline. Do not increase confidence merely to cosmetically improve a reliability plot.

All-row favorite buckets are near calibration in the 50–60% range and underconfident around 60–70%, but evidence above 70% is sparse. One 93.25% prediction, VGK–LAK on 2025-10-08, lost. Investigate its source feature vector, offseason rest, imputation, and member disagreement before adding arbitrary clipping.

## 7. Goal-distribution head: do not blend it into moneyline yet

### Corrupt monitoring — confirmed

`_run_engine_line_pairs` returns `(p, y)`; [the metric caller](../nhl-backend/backend/monitoring.py#L757-L785) unpacks it as `(y, p)`. Corrected from delivered probabilities:

| Line | n | Correct AUC | Correct log loss | Current reported log loss |
|---|---:|---:|---:|---:|
| Derived ML alias | 2,635 | 0.58549 | 0.69252 | 6.7951 |
| Over 5, push excluded | 2,032 | 0.51333 | 0.59803 | 5.7492 |
| Over 6, push excluded | 2,344 | 0.50216 | 0.69564 | 6.9053 |
| Over 7, push excluded | 2,107 | 0.49879 | 0.61777 | 5.7141 |
| Home margin >1, push excluded | 2,065 | 0.59518 | 0.66686 | 6.5395 |
| Home margin >2, push excluded | 2,364 | 0.58002 | 0.55598 | 5.1375 |

Brier's squared difference is symmetric, so it survives the swap; log loss, ECE, means, and AUC do not. Validate label support `{0,1}`, use keyword arguments, and compare independent sklearn calculations. Per-line totals discrimination is currently near random; low loss at asymmetric thresholds is not necessarily skill versus a base-rate predictor.

### Hockey probability contract mismatch — confirmed

The simulated score head models final goal counts, yet raw `derived_ml` is `P(home score > away score)` and retains tie mass. For NHL full-game moneyline there is no final tie. Mean distribution tie mass is **13.95%**; mean raw derived home probability is **45.98%** versus actual home wins **54.27%**. A scorer trained on final OT/SO-resolved scores cannot simply reinterpret its independent Poisson tie mass as a regulation-tie forecast.

Further, `derived_ml` is not refreshed after derived calibration: its maximum difference from `p_home_win_derived` is **0.17769**. Calibrated home/away/tie sums can miss one by **0.02602**, because away is computed using an older tie probability before tie gets overwritten.

**Solution:** explicitly separate regulation score distribution, regulation tie probability, and conditional OT/SO home win probability:

`P(home full-game win) = P(home regulation win) + P(regulation tie) * P(home wins OT/SO | tie)`.

If only final-score data are modeled, define a justified final-winner conditioning/allocation policy rather than claim the tie is regulation. Audit shootout-added scoreboard goals and empty-net effects separately. Refresh all aliases together and enforce sum/bounds/event definitions across calibrated grids.

### Cross-line incoherence — measured

Independent line-specific calibration breaks required survival ordering:

- **143 / 2,635** rows have a totals probability increasing at a higher line; largest upward jump **0.05778**.
- **1,176 / 2,635** rows have home-cover probability increasing at a harder margin; largest upward jump **0.28444**.

Per-line normalization is insufficient. Use a shared coherent discrete PMF/CDF or monotone distribution calibration, then derive every line from that same distribution. Whole-line pushes and half-lines must correspond to the same events.

### Blend experiments: no adoption

On the 2,418 grading rows, an exploratory 10% logit blend with raw derived alias gives loss **0.67430**, AUC **0.59915**—tiny changes of −0.00017 and +0.00029. Higher weights get worse; the raw alias is semantically wrong for full-game ML. A 10% no-tie-conditioned grid blend gives loss **0.67444**, AUC **0.59906**, essentially noise. A 50/50 tie allocation is also neutral at low weight and worse at higher weights.

These sweeps use the whole inspected population, not sealed selection. The lesson is **do not add this head just for diversity**; repair its targets/contracts and test causal residual complementarity first.

## 8. Selection governance and monitoring defects

[Current RFE scoring](../nhl-backend/backend/feature_selection.py#L92-L110) evaluates retrospective `p_ensemble` over **all** finite rows, not the production grading population. Its gate can allow AUC deterioration up to **0.003**, contrary to a strict goal of increasing both AUC and loss quality. It is adaptive repeated search with a roughly one-standard-error threshold, not multiplicity-controlled evidence.

The existing dated RFE artifact is stale: **28** base features, 120 trials, old scores/protocol. Its one accepted feature gained 0.0020866 with paired SE 0.0020724—only about one SE. Do not use its apparent AUC 0.619 as proof a candidate beats today's 68-feature/causal system.

Additional implementation issue: each successive trial computes paired SE against the original baseline losses, while gain is compared against the evolving best subset. Those uncertainty and effect comparisons must reference the same incumbent and exact game IDs.

**Solution:** use causal ensemble predictions, explicit population masks, aligned per-game incumbent losses, family-level trials, nested selection, and a small preregistered candidate budget. Store all rejections. Require independent later-period confirmation rather than selecting a winner from 120 noisy comparisons. Zero/failed member rows must be aligned by game ID; the current member accumulator's length-mismatch fallback can silently discard a model after a failed fold.

The monitor's Brier baseline is **0.45731**, computed as `1 − mean(home_win)`. That is the error/Brier score of an **always-home deterministic forecast**, not a realistic constant home-probability benchmark. A descriptive constant observed-rate benchmark is **0.24818**. Replace the label or calculate a train-prior home-rate baseline per fold. Also, the function named rolling 30-day Brier currently emits per-day means rather than a 30-day rolling calculation; the unused window argument is a reporting defect, not a signal of model gain.

## 9. Prioritized improvement program

### Phase 0 — Trustworthy baseline and correctness

**Do first:** causal-versus-retrospective reporting; shared model fit/refit contracts; correct metric arguments; Elo expectation sign; PP and faceoff definitions; goalie team/roster gate; probability alias/coherence checks; honest coverage provenance.

Archive a reproducible feature snapshot and member OOF ledger before refitting. Rebuild features from raw cached facts under a new feature version. Refit all members, weights, and calibration together. Correction can make historical scores temporarily worse by removing optimism; that is a more trustworthy baseline, not a reason to retain the defect.

### Phase 1 — Higher-quality NHL signal

**Highest-priority feature families to evaluate:**

1. Opponent-adjusted, venue/score-state-aware 5-on-5 team xGF/xGA rates, xG share, unblocked attempts, and shot quality; multi-scale strength rather than only final goals and five-game SOG.
2. Player-pool defense and playmaking alongside individual offensive xG; projected ice-weighted lineup strength, replacement-level lost value, and PP1/PK unit composition. Current EVO/PPO individual shot-generation rates omit defensive skill and puck distribution.
3. Probabilistic/confirmed goalies, exposure-shrunken GSAx/SV skill, rest/workload, and starter uncertainty.
4. Real PP/PK conversion and expected-goal rates plus penalties drawn/taken and opportunity volume; matchup interactions between offense and opposing penalty kill.
5. Travel/schedule load, season-opening carryover, and team/offseason roster changes.

These need source-level PIT availability, not merely a historical archive that includes information published after the prediction cutoff. The current policy is market-free; no sportsbook inputs are proposed here. An external benchmark would be evaluation-only and require a separate policy decision.

### Phase 2 — Representation and regularization

- Deduplicate aliases; compare diff-only versus diff-plus-strength-level linear features, not all redundant triples.
- Normalize/clamp rest meaningfully, shrink noisy rates by exposure, and preserve missingness provenance.
- Retune elastic-net C/L1 mix after representation fixes; compare ridge with less arbitrary selection among correlated inputs.
- Retune NHL tree depth/leaves/regularization and stable round budgets only after feature repair; compare feature-family subsampling rather than an unstructured 68-column search.
- Use exposure-aware missingness indicators only when the missingness process is stable at serve; a scrape failure should not become a winning signal.

### Phase 3 — Blend and calibration

Run the finite blend candidate set from Section 2, causally re-earning weights for every candidate. Test identity/temperature/regularized calibration using the same base predictions and eligibility as serving. Include season/early-ramp diagnostics and member-disagreement-based uncertainty. Do not lower a calibration gate until an independent temporal test supports it.

### Acceptance gates

These are proposed **preregistered thresholds**, not claimed gains:

- **Correctness:** all semantic fixtures and game-ID/population alignments pass; no scored outcome changes its own or earlier inputs; every price is bounded; grids are cross-line monotone; full-game home+away=1; regulation three-way sums=1; fit/serve feature types and round policies match.
- **Primary success:** paired causal candidate log loss improves by at least **0.001** and AUC improves by at least **0.002** on the prespecified aggregate evaluation. Show week-block intervals and season/team robustness; point estimates alone are not enough to advertise success. If intervals remain inconclusive, keep the candidate in shadow evaluation.
- **Non-regression:** no prespecified season/regime meaningfully worsens (suggested disclosure bounds +0.002 loss/−0.003 AUC); inspect calibration slope/intercept, Brier, reliability uncertainty, and high-confidence tails. Fixed-bin ECE alone is noisy and not an admission oracle.
- **Confirmation:** multiple rolling-origin outer windows with enough games; one genuinely untouched later interval or live shadow ledger. Retain near-opening and playoff reporting even when their models are separate.
- **Reproducibility:** data/feature/schema hashes, library lock, seed, feature order, member rounds, failed-fit status, weights, calibrator params, capture/source timestamps, and forecast cutoff persisted. Track fit/inference cost without claiming an unmeasured speedup.

## 10. Verification status and limits

The audit CLI was run on the delivered CSV/JSON interface, and reconstructed **59 folds / 2,635 scored / 2,418 grading** exactly. Published raw log loss was independently reproduced. Five new harness tests passed, and both new Python files compiled. The official API comparison and synthetic Elo/goalie/metric probes were executed. A full NHL regression run is recorded in the linked test results.

The first full-suite run had **3 failures / 537 passes**: two Windows default-cp1252 text failures and one existing notebook dependency assertion. The final run with `PYTHONUTF8=1` (including the five new tests) finished **544 passed / 1 failed**. The remaining failure is `test_the_kaggle_notebook_installs_tqdm_or_there_is_no_bar_to_draw`: the Kaggle notebook does not install `tqdm`. This unrelated pre-existing defect was left unchanged; the notebook is documented as Kaggle-owned. No tests were skipped or weakened. [Initial run log](nhl_model_audit_20261006/test_results_initial_locale.txt) and the final results file preserve both runs.

No repository typechecker configuration was found; compilation and runtime/pytest verification were used. The production Kaggle pipeline was **not** rerun: raw feature inputs/member OOF stores are absent and a fresh full pull would not recreate missing historical pregame availability. No AUC/log-loss improvement from retraining is claimed. This delivery is a diagnosis and experiment/admission plan, with production untouched.
