# MLB custom Pitching+: build and validation verdict

## Update 2026-10-25: the family is now the production contract

Promoted at the owner's direction. The game-model replacement ablation was run
after this report (`data_delivery/mlb_pitchingplus_replacement_ab_20261025.json`,
6,879 graded games, 16 incumbent columns out / 18 pitchingplus columns in):
AUC **0.5773 → 0.5758**, logloss **0.6810 → 0.6814**, Brier 0.2440 → 0.2443,
ECE **0.0121 → 0.0084**. The owner accepted the small AUC/logloss cost for the
markedly better calibration and for stuff/location inputs over results-based
xFIP/K-BB%. `PITCHINGPLUS_PROMOTION_APPROVED` is now `True`; the guard itself
remains, so setting it back to `False` restores shadow-only behavior.

What this does **not** change: the 2025 external-agreement shortfall, the
sub-0.90 Stuff+/Location+ agreement, and the missing untouched-new-data OOS all
still stand as open items in this report. The promoted columns are named with
the `_pitchingplus` infix (`sp_pitchingplus_5g_*`,
`pitcher_regression_indicator_pitchingplus_*`,
`wind_advantage_flyball_factor_pitchingplus_*`, `bullpen_pitchingplus_10g_*`,
`bullpen_pitchingplus_3g_*`, `bullpen_meltdown_risk_pitchingplus_*`); the 16
replaced columns keep being generated and returned to the RFE candidate pool.
Serving width is 116 (114 − 16 + 18; the wind interaction gained its home/away
legs). The dashboard artifacts (`features_metadata`, `model_monitor`, drift and
coverage tables) follow the serving list on the next pipeline run, so artifacts
written before that run still describe the 114-column contract.

## Refined verdict at the time of the 2026-10-24 report: production blocked

This report describes the refined candidate; the original validation remains in
`run_diagnostics/pitchingplus/validation.json`. Refined artifacts are isolated in
`run_diagnostics/pitchingplus/refined/`.

The learned model and game-feature plumbing are implemented and tested, but the
requested FanGraphs agreement has **not** been achieved. Do **not** switch the
production xFIP/K-BB%/interaction features yet. This is a completed validation of
the current candidate, **not** a claim that the requested production migration
is finished.

| Regular season | Eligible pitchers | All-pitcher Pearson r | SP r | RP r |
|---|---:|---:|---:|---:|
| 2025 (previously inspected development comparison) | 522 | 0.8802 | 0.8923 (221 pitchers) | 0.8708 (301 pitchers) |
| 2026 (previously inspected temporal comparison) | 510 | 0.9193 | 0.9282 (210 pitchers) | 0.9127 (300 pitchers) |

Eligibility: at least 400 local regular-season pitch rows and at least 95% scored;
join by MLBAM pitcher ID and season; SP means GS/G >= 0.5, RP otherwise. The
comparison uses pitch-usage means to compare like-for-like pitcher-season
summaries, **not** the IP-weighted game predictors. Pearson r is correlation,
not accuracy or percent variance explained. No training labels are FanGraphs
grades, and there is no regression-to-FanGraphs or rescaling fitted on this
benchmark. Sensitivity at 800 and 1,000 pitches is recorded in the refined
validation JSON; it does not replace the fixed 400-pitch eligibility gate.

2025 all-pitcher 95% Fisher interval: **0.8593–0.8982**; 2026:
**0.9047–0.9318**. The 2026 combined-model groups pass 0.90, but every 2025
group still fails. The external snapshots contain 873 (2025) and 868 (2026)
rows, all matched before eligibility filtering. Both snapshots had already
been inspected: neither is an untouched external holdout, and 2026 was not
used for candidate selection.

### Measured improvement over the original learned candidate

| Year | Combined r before → after | MAE before → after | Custom season SD before → after | FanGraphs SD |
|---|---:|---:|---:|---:|
| 2025 | 0.7634 → 0.8802 | 6.38 → 3.30 | 0.82 → 7.73 | 8.65 |
| 2026 | 0.8122 → 0.9193 | 6.21 → 2.77 | 0.82 → 8.01 | 8.59 |

The corrected scale is based only on prior calibration pitcher means, not
FanGraphs. It addresses severe compression without claiming proprietary-scale
replication. Independently undoing each quarterly affine grade transform gives
raw-prediction agreement of **0.8808/0.9190** (2025/2026), showing that the
correlation improvement is not merely display rescaling. MAE/spread improvements
reflect both the learned model and the changed scale; they are not a pure
model-only ablation.

Stuff+ all-pitcher agreement is 0.8313/0.8717 and Location+ is 0.7724/0.7633.
These components are not claimed to meet 0.90 either.

## Model structure

Production feature generation now uses three separate causal nonlinear
LightGBM run-value models, not the old fixed-weight composite:

- **Stuff+**: pitch type, throwing hand, velocity, spin, separate horizontal and
  vertical movement, release coordinates/extension, secondary-to-primary-fastball
  velocity/movement differences, and a disclosed spin-axis/movement angular
  disagreement proxy. Added measurements include circular spin-axis inputs,
  arm angle, throwing-side acceleration, vertical/forward acceleration,
  effective speed, and extension-to-release-height ratio. The angular proxy
  now uses the correct catcher-view horizontal orientation. No plate location
  enters this component. The angular proxy is **not** a reconstructed
  active-spin or seam-shifted-wake measure.
- **Location+**: pitch type, count, batter/throwing hand, horizontal location,
  raw and batter-zone-normalized vertical location. No physical pitch traits.
- **Pitching+**: a third combined model using physical and location/context
  inputs. It is **not** `(Stuff+ + Location+)/2`, and a missing combined-model
  prediction never falls back to one component.

The starting label is observed `delta_pitcher_run_exp`, positive for runs
prevented. For balls in play, a training-only count-intercept regression maps
observed wOBA to run value, then substitutes Savant expected-contact wOBA.
Missing expected contact or mapping falls back to observed run value; missing
observed labels remain missing. Contact outcomes enter training labels only,
never inference. Stuff+ removes train-estimated count means; Location+ and the
combined model observe count and fit the expected-contact-adjusted target.
Pitcher/batter identities, game results, roles and FanGraphs grades are not
learned inputs. IDs only group calibration and output measurements. Primary
fastball references use historical outing pitch usage; whole-outing measurements
are used only in subsequent games' features. Pitch keys have deterministic order.

A bounded seven-candidate comparison trained before September 2024 and selected
by September–October 2024 **raw observed run-value MSE**, not FanGraphs agreement.
The selected expanded-tracking/expected-contact/1,000-tree model reduced that
holdout MSE from 0.046182 to 0.046016 (0.36% relative). This is development
selection evidence, not an untouched final test. The combined model uses 63
leaves and learning rate 0.035; the component models retain 180 trees.

Models refit at quarterly origins using at most two prior years. The last
**30 observed regular-season dates** before the origin form the calibration
period. The reference population is equal-weight pitcher means with at least
100 measured pitches each and at least 30 qualifying arms. On that prior
population, mean/SD are mapped to 100/10; this is not pitch-level SD or a
FanGraphs-fitted transform. This handles offseason gaps without fabricating
opening spring scores. Fit, calibration, and scored periods are disjoint; a
score's own quarter never trains or calibrates its model. Initial warmup and
insufficient references remain NULL. Location+ is centered within pitch type;
the other models are centered globally. This scaling is distinct from blending
an individual pitcher toward 100: **there is no such blending in the learned
model's game features.**

[FanGraphs' primer](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/)
supports separate physical/location/combined models, contextual location,
and pitch-level plus scaling. Its exact model, target transforms, fitting
parameters, and aggregation are proprietary. This implementation follows the
public structure but does not claim to reproduce the proprietary model.

## Exact per-game normalization and weights

For an historical appearance, average the measured combined-model pitch grades:

`Q_appearance = mean(Pitching+ over scored pitches)`

For a starter's target game:

`SP_Q_5g = SUM(IP_appearance * Q_appearance) / SUM(scored IP_appearance)`

using the last five appearances on dates strictly before the target date.
`IP` is the existing canonical recorded-outs / 3 value from
`pitcher_game_stats`, not baseball decimal notation, pitch-implied innings, or
a nine-inning total. Missing/zero IP receives no weight. There are no pseudo-IP
or league-prior blends. Existing starter staleness gates remain. Multiplying
all observed IP by the same constant leaves the quality average unchanged.

For the bullpen, apply the same numerator/denominator over reliever appearances
in the prior **three or ten team games**. Each reliever is weighted by his own
recorded IP. Complete games/no-eligible-relief games still consume a team-game
window slot. The target date's IL ledger and recent heavy-outing exclusions
filter historical arms; returned pitchers can rejoin, while a currently absent
arm's earlier healthy innings cannot continue contributing. No blend to 100.
This is exposure-weighting of observed staff, not a projected tonight's-roster
or projected-innings model; trades/current roster membership are not separately
reconstructed.

Same-date games are excluded from both SP and BP histories because `game_pk`
is not a verified start timestamp. Both legs of a doubleheader receive the
same pre-date history; the target game's pitches never score its own feature.

### What the incumbent code actually weights

The assumption that all existing bullpen quality is IP-weighted is **not true**:

- `bullpen_kbb_10g/3g` pools `(SUM(K) - SUM(BB)) / SUM(PA)`, then applies
  the existing opportunity/league-prior shrinkage. Its denominator is **PA**.
- `bullpen_xfip_10g` uses pooled counts per recorded **IP**, plus its existing
  shrinkage. It is not a simple equal-weight pitcher average.
- `bullpen_pitches_3d` is a **summed three-calendar-day workload**, not quality.
  It remains unchanged, per the agreed decision to preserve fatigue separately.
- The preexisting trial proxy used pitch-count weights and blending. It is now
  explicitly labeled a legacy diagnostic comparator, not the active learned
  feature-generation path.

## Feature mapping and interactions

The families below were generated in parallel with the incumbents on
2026-10-24; since the 2026-10-25 promotion the right-hand column **is** the
serving contract (the `_pp` names in this table's original form were renamed to
`_pitchingplus` for naming consistency before promotion):

| Replaced family | Promoted column (now served) |
|---|---|
| `sp_xfip_5g_{diff,home,away}` | `sp_pitchingplus_5g_{diff,home,away}` |
| `pitcher_regression_indicator_{diff,home,away}` | `pitcher_regression_indicator_pitchingplus_{diff,home,away}` |
| wind interaction plus new home/away legs | `wind_advantage_flyball_factor_pitchingplus_{diff,home,away}` |
| `bullpen_kbb_10g_{diff,home,away}` | `bullpen_pitchingplus_10g_{diff,home,away}` |
| `bullpen_kbb_3g_{diff,home,away}` | `bullpen_pitchingplus_3g_{diff,home,away}` |
| `bullpen_pitches_3d_{diff,home,away}` | unchanged workload |
| `bullpen_meltdown_risk_{diff,home,away}` | `bullpen_meltdown_risk_pitchingplus_{diff,home,away}` |

Raw quality differences are home minus away. Interaction structures are kept:

- Regression: velocity-difference × SP-quality-difference; within-side twins
  use that side's velocity × that side's SP grade.
- Wind: observed multiplier × matching SP grade/difference. New home/away
  legs exist in the shadow family. Outdoor missing weather stays NULL; known
  closed roofs neutralize wind to zero under the existing roof policy.
- Meltdown: workload-difference × bullpen-quality-difference; within-side
  twins use that side's three-day workload × ten-game bullpen grade.

These interactions are per-game derived values, **not** all 100-centered
Pitching+ grades. A within-side product has different units and scale than a
home-minus-away product. Higher Pitching+ reverses xFIP's direction but has the
**same** direction as K-BB%; no inherited coefficient sign is assumed.

`PITCHINGPLUS_PROMOTION_APPROVED` gates both writing an adopted RFE subset
containing these features and applying such a saved subset. It was `False` at
this report's time and is `True` after the 2026-10-25 promotion; either state
leaves the columns generated, so shadow measurement is always possible. The
adopted feature state was re-issued to the 116-column contract; serialized
models were untouched and are replaced by the next run's fit. No report JSON
unlocks promotion — it is a config decision, not an artifact flag.

## Verification and measurements

Local input: 2,178,102 cached pitch rows, 2024-03-20 through 2026-10-18.
Full SQL feature assembly produced 7,412 unique game rows and 63,624 pitcher
appearances. The refined complete assembly took **464.0 seconds**, including
439.9 seconds for the learned pitch-model stage. Independent pitch-model
validation took **449.3 seconds**. These are local runtime measurements, **not**
claims of a speedup; the deeper model costs more than the original candidate.

Across 1,799,975 strictly out-of-time pitches in ten scored quarterly periods,
combined-model **raw observed** run-value MSE was **0.048582**, versus
**0.050479** for the training-mean constant baseline: **3.76% lower MSE**.
The original candidate's MSE was 0.048726: the refined candidate reduces that
by about 0.30% relative. Location+ improved 3.51% versus its constant baseline;
Stuff+ improved 0.10% on its count-neutralized target.
This demonstrates limited pitch-level predictive signal, **not** a proven gain
in game winner/score prediction. Full game-model A/B retraining was not run,
because the external agreement gate failed first.

For 2025–2026 game rows, SP-side coverage is approximately 98.97%/98.99%,
with 100% coverage for the checked bullpen windows in this cache. Initial
2024 model warmup explains most missing rows in full-history coverage. All
nonmissing quality columns are finite, and the refined scale has materially
wider pitcher-level spread. All raw-quality diff and within-side interaction
identities were exercised.

The raw no-weather assembly correctly has only roof-neutral wind values.
A separate one-to-one join to existing cached weather factors exercised
4,400 outdoor wind-difference games, 4,460 home legs and 4,469 away legs, with
both positive and negative outputs and exact multiplier identities. No weather
history was fetched.

**Final targeted regression battery: 235 passed, 1 skipped, 8 warnings (33.59 s).**
The skip is the existing real-symlink test blocked by Windows privileges;
its mocked positive case passes. Warnings cover an all-NaN training fixture,
a sklearn deprecation, and DataFrame fragmentation. New tests exercise causal
outcome perturbation, component input separation, offseason calibration,
recorded-IP versus pitch weights, target-date IL/spent availability, complete
game slots, five-appearance limits, same-date isolation, NULL propagation,
weather formulas, deterministic input order, metadata, and promotion rejection.
Python compilation and `git diff --check` are recorded in the refined final
checks JSON. There is no configured
standalone static typechecker in this backend; compilation is not a substitute
for a typecheck.

### Limitations that remain blocking or unverified

1. All 2025 combined-model groups remain below 0.90, including relievers;
   2026 passing does not override this. Both external periods were already
   inspected and a genuinely new evaluation period is still needed.
2. Corrected prior-only scaling removes the severe compression but does not
   establish exact FanGraphs scale equivalence. Stuff+/Location+ agreement is
   also below 0.90.
3. Local pitcher/batter IL stint artifacts are absent. Full real-data assembly
   reports this loudly and uses existing degraded availability semantics.
   Target-date availability logic passes synthetic contract tests, but the
   real-ledger integration cannot be claimed as verified on this checkout.
4. The game-model walk-forward replacement ablation was run on 2026-10-25
   (results in the update section at the top) and the family was promoted by the
   owner; exact upcoming-slate model parity has still not been certified.
   Existing starter-state
   carry is not treated as proof that tonight's grade incorporates the latest
   completed start; a direct as-of slate binding is required before adoption.
5. Full-backend/all-sport suites were not run. Only the relevant MLB suites and
   full historical game-feature assembly were exercised.

## Frozen 2025 disagreement diagnosis

The follow-on analysis reads only 2025 pitch rows and the cached 2025 FanGraphs
response. It **does not fit models, tune on 2026, fetch data, or change serving**.
The same 522 eligible pitchers reproduce r=0.8802408960 and MAE=3.3021 exactly.
The signed gap below is **custom minus FanGraphs**; negative means lower custom
quality relative to the external snapshot. This is post-hoc descriptive evidence,
not causal attribution, a new holdout, or a license to recalibrate on FanGraphs.

### Where season-level disagreement is concentrated

- Overall mean gap: **-0.97 points**; RMSE: **4.21 points**.
- Relievers: **301/522 pitchers, 66.8% of squared error**, MAE 3.67 and mean
  gap -1.17. Starters: MAE 2.80 and mean gap -0.69.
- The worst 10 pitchers contribute **16.8%** of squared error; the worst 25
  contribute **31.2%**. Errors are not confined to a few extraordinary cases:
  **115 pitchers** have absolute gaps of at least five points, and 12 exceed ten.
- Removing the worst 10 *after seeing their errors* would raise r only to
  0.8991; removing 25 gives 0.9088. These are explicitly selection-biased
  sensitivity figures, **not valid eligibility changes or promotion evidence**.
- Among 102 pitchers with at least 2,000 pitches, r is 0.9122 and MAE 2.61;
  among 176 with 400–799 pitches, r is 0.8473 and MAE 3.61. Volume is not the
  whole explanation: the 1,200–1,999 group also has r=0.8331 (76 pitchers).
- Pitchers with >60% usage of one pitch have MAE **4.45** (35 pitchers), versus
  **3.09** for maximum usage <=40% (227 pitchers). These are overlapping
  role/mix/volume effects, not controlled estimates.

External-grade strata show compression/ranking mismatch: mean custom gaps are
+2.06 for FanGraphs below 90, +0.42 for 90–<100, -2.21 for 100–<110 and -3.62
for 110+. Because the residual itself subtracts the external grade, this
association is partly mechanical and **does not justify a fitted correction**.
The combined residual correlates 0.515 with Stuff+ disagreement and 0.283 with
Location+ disagreement. The combined model is nonlinear and independently
trained, so these are associations, not additive or causal component shares.

### Pitch-mix results

These are **pitcher–pitch-type pairs** among the same 522 pitchers, with at least
100 scored local pitches of that type. Each pair receives equal weight. The
external per-type rows have no independent pitch-count field, so the threshold
is local only. Exact labels may still differ across vendors.

| Exact pitch label | Pairs | Combined r | Mean custom–FG gap | MAE |
|---|---:|---:|---:|---:|
| FF (four-seam) | 434 | 0.8920 | +0.02 | 4.21 |
| SI (sinker) | 275 | 0.8578 | **-5.40** | **6.78** |
| FC (cutter) | 159 | 0.8205 | -1.75 | 4.91 |
| SL (slider) | 278 | **0.7743** | -1.27 | 5.47 |
| CH (changeup) | 219 | 0.8687 | -0.65 | 5.60 |
| CU (curve) | 153 | 0.8268 | -0.91 | 5.41 |
| KC (knuckle curve) | 32 | 0.8298 | -2.91 | 5.62 |
| FS (splitter) | 71 | 0.8066 | +0.87 | **6.60** |

The strongest broad directional issue is **sinker undergrading**. Sinker-dominant
pitchers average -2.97 overall, with MAE 4.01 (95 pitchers). The mean gap persists
in both starters (-3.12, 35 pitchers) and relievers (-2.89, 60 pitchers), so it is
not solely the relief/starter mix. This remains descriptive, not proof of a
specific missing physical variable. Sliders show weaker exact-label ranking;
splitters show high dispersion rather than a uniform signed shift. Forkballs
have only one qualifying exact-label pair and cannot support a group conclusion.

### Taxonomy and aggregation must be separated from model disagreement

FanGraphs' cached response has no separate ST/SV grades. Two diagnostic mappings
are kept separate:

1. **Exact labels only:** 91.30% of eligible scored pitches have a matching
   external per-type grade.
2. **Pooling hypothesis:** ST→SL, SV→CU, FT→SI increases this to 99.34%.
   These are **unconfirmed** mappings, not corrections to either model. With
   ST pooled into SL, slider agreement is 0.7976 across 402 qualifying pairs
   and MAE is 5.08; this does not change the frozen overall comparison.

For every pitcher, the script checks the accounting identity:

`season gap = SUM(w_type × (custom_type − FG_type))`
`           + SUM(w_type × (FG_type − FG_overall))`
`           + SUM_unpaired(w_type × (custom_type − FG_overall))`

Weights are shares of **scored local pitches**, without renormalizing the
matched subset. The middle term is an aggregation/taxonomy bridge, not model
error; the final term is explicitly unpaired, not an imputed external grade.
The identity closes to <3e-14 points for both mappings. It does not establish
that either vendor's proprietary overall aggregation is wrong.

Among 502 pitchers with >=99% paired usage under the pooling hypothesis, the
median absolute bridge is **0.13 points**, but 16 exceed three points. Thus
aggregation differences are usually small but can dominate individual outliers.

Examples below use the pooled hypothesis when needed; exact-label results are
also retained in the diagnostic tables:

| Pitcher | Overall gap | Main diagnostic observation |
|---|---:|---|
| Carson Palmquist | +14.99 | FF gap +15.28 contributes +8.02 season points; ST→SL gap +22.48 contributes +7.08. Both model disagreement and taxonomy remain relevant. |
| Dylan Dodd | -13.75 | Local SI usage 35.3%; external SI grade absent, while external FA usage is 35.3%. Taxonomy prevents a clean sinker comparison; do not treat the entire gap as a learned-model failure. |
| Anthony Molina | -13.52 | Local sinker grade is unpaired; paired-type contribution -3.51, aggregation bridge -7.35, unpaired centered contribution -2.66. |
| Roki Sasaki | +12.82 | Local FO usage 33.3% but no external FO grade; external FS usage is 33.5%. FF/SL pairing alone cannot explain the full gap. No FO→FS remap is silently assumed. |
| Chris Martin | -11.55 | Paired-type contribution -3.04 and bridge -8.51; this is mainly an overall-versus-type aggregation/taxonomy discrepancy under this mapping. |
| Brenan Hanifee | -11.53 | SI gap -15.14 contributes -9.47 season points; bridge is only +0.01, a cleaner matched-type undergrading case. |
| Jonathan Loaisiga | -10.95 | SI gap -12.55 contributes -7.26 season points; bridge is +0.01. |
| Felix Bautista | -10.07 | FS grade 73.45 versus external 110.15 contributes -10.50 season points; SI agrees within +0.24. |
| Logan Webb | -9.31 | CH gap -22.59 contributes -5.45 season points; SI contributes -2.71. |
| Kenley Jansen | -9.29 | FC gap -10.19 at 81.4% of scored usage contributes -8.29 season points. |

Input coverage is not an obvious global explanation: median scored-minus-external
pitch count is -1, and total local-minus-external count differs by at most 23
pitches. Signed disagreement correlations with total-count difference (-0.039),
scored fraction (0.065), and arm-angle
missingness (-0.003) are small. This does **not** rule out individual missing-data
or taxonomy problems. Monthly local means are saved only as custom-score drift
versus a season benchmark, **not time-matched external errors**.

### Diagnostic verification and next investigation boundary

The runner verifies all 2025 pitch identities and saved pitcher/component means,
recovers frozen raw predictions (r=0.8807553510), checks role/volume/mix/SSE totals,
and reconciles the two pitch-type decompositions. SHA-256 guards confirm frozen
inputs, production feature/model code, adopted feature state, and the notebook
remain unchanged. No full training/test battery is needed to claim a diagnostic
result; this follow-on changes only diagnostic code/artifacts and this report.

The evidence supports investigating **sinker-relative target behavior and
physical/context interactions**, reviewing **splitter-specific outliers**, and
resolving **vendor taxonomy/overall aggregation** before tuning. It does not
support FanGraphs-fitted offsets, deleting difficult pitchers, or production
promotion. Any future modeling experiment still needs prior-period selection
and genuinely new evaluation data.

- [Diagnostic runner](../run_diagnostics/pitchingplus/diagnose_2025.py)
- [Machine-readable 2025 diagnosis](../run_diagnostics/pitchingplus/refined/diagnosis_2025.json)
- [Ranked pitcher diagnostics and mix shares](../run_diagnostics/pitchingplus/refined/diagnosis_2025_pitchers.csv)
- [Pitch-type summary](../run_diagnostics/pitchingplus/refined/diagnosis_2025_type_summary.csv)
- [Exact-label pitcher/type pairs](../run_diagnostics/pitchingplus/refined/diagnosis_2025_types_exact_labels.csv)
- [Pooled-hypothesis pitcher/type pairs](../run_diagnostics/pitchingplus/refined/diagnosis_2025_types_pooled_hypothesis.csv)
- [Monthly custom-score means](../run_diagnostics/pitchingplus/refined/diagnosis_2025_monthly.csv)
- [Diagnostic verification log](../run_diagnostics/pitchingplus/refined/diagnosis_2025.log)

## Sinker remediation follow-up: measured, bounded, and rejected candidates

A follow-on investigation addressed the sinker gap and the report's guidance to
study sinker-relative target behavior and physical/context interactions. Every
measurement uses 2024–2025 rows only; **2026 rows are never loaded** and no
FanGraphs grade enters any fit, offset, or selection. Selection is strictly by
Sep–Dec 2024 raw **observed** run-value MSE on a fixed holdout (the report's
"Sep–Oct 2024" window; no MLB games exist after the World Series), with every
candidate trained Jan–Aug 2024 under the production combined-model budget.
Each selection verdict is written to disk **before** any FanGraphs file is
opened; FG numbers afterward are descriptive only.

### Where the sinker gap does and does not come from

The target-behavior probe decomposes the model's out-of-time residual by pitch
type against raw observed run value and prices the expected-contact
substitution directly:

- The expected-contact substitution **credits** sinkers rather than charging
  them: on 2025 BIP the mean substitution shift is **+0.0083** raw RV for SI
  (realized wOBA 0.3640 exceeds xwOBA 0.3554), versus **−0.0030** for FF.
  Sinker contact outcomes are genuinely better for pitchers than fastball
  contact (2025 mean observed BIP RV −0.021 for SI vs −0.058 for FF), and the
  target construction preserves that. Target pricing is **not** the cause of
  the −5.40 season gap.
- Against observed outcomes the model underrates sinker run prevention only
  slightly: holdout SI bias is −0.00139 (SE 0.00157, **within noise**) and
  2025 SI bias is −0.00117, versus −0.00055 and +0.00079 for FF. At the
  frozen quarterly scale these magnitudes are a small fraction of the −5.40
  external season gap, which is dominated by external scale/aggregation and
  taxonomy differences documented above.
- Splitter residuals are **two-sided**, not a systematic FS shift: among 75
  pitchers with ≥100 FS rows, 40 bias negative and 35 positive (mean
  +0.0013). The report's flagship FS case, Felix Bautista (−0.038 raw RV over
  179 rows), is the most negative individual but not evidence of a group-level
  defect.

### Bounded candidate comparison — nothing beat the baseline

Seven pre-declared candidates were compared under identical budgets in two
rounds. All failed the pre-registered rule (strictly lower holdout raw
observed MSE than the refined baseline, 0.04601412):

| Round | Candidate | Added inputs | Holdout raw MSE |
|---|---|---|---:|
| 1 | baseline | (refined spec) | **0.04601412** |
| 1 | type_relative | per-type relative velo/movement (fit-period means) | 0.04604298 |
| 1 | interactions_raw | movement × batter hand, movement × zone height | 0.04602147 |
| 1 | type_relative_context | type-relative + its interactions | 0.04604919 |
| 1 | all_features | union | 0.04603593 |
| 2 | approach_angles | geometric VAA/HAA | 0.04603532 |
| 2 | approach_context | VAA/HAA + zone/batter interactions | 0.04603238 |

Paired per-pitch loss differentials on 125,513 holdout pitches confirm the
verdict is not a knife-edge: interactions_raw is statistically
indistinguishable from baseline (t = 0.66), while type_relative_context
(t = 2.81) and approach_angles (t = 2.01) are genuinely worse. The
sinker-specific bias barely moves across candidates (−0.00139 baseline vs
−0.00118 best) and remains within its standard error. Round 2 is explicitly a
second look at the same holdout; even a pass would have required genuinely new
evaluation data before any claim.

FG descriptive agreement on 2025 pitcher means moves from 0.8499 (baseline,
single-model comparison) to at most 0.8557 (interactions_raw), but these
numbers are previously inspected, single-origin comparisons and **did not and
cannot select a candidate** — integrating on that basis would be exactly the
FG-fitted tuning the verdict forbids.

### Verdict

**No enhancement is integrated.** The shadow build, its pinned tests, adopted
feature state, serving features, and frozen artifacts are all unchanged;
`PITCHINGPLUS_PROMOTION_APPROVED` remains False. The measured conclusion is
that the sinker gap is predominantly external scale/taxonomy disagreement, not
a fixable target or feature defect within this model class and budget. Further
candidate search on the same holdout would be selection on noise; the report's
standing requirement — genuinely new evaluation data — remains the gating
item, alongside vendor taxonomy/overall aggregation resolution.

- [Target-behavior probe](../run_diagnostics/pitchingplus/refined/sinker_target_probe.json)
- [Round-1 selection verdict](../run_diagnostics/pitchingplus/enhance_selection.json)
- [Round-1 full verdict](../run_diagnostics/pitchingplus/enhance_verdict.json)
- [Paired noise check](../run_diagnostics/pitchingplus/enhance_noise_check.json)
- [Round-2 selection verdict](../run_diagnostics/pitchingplus/enhance_round2_selection.json)
- [Round-2 full verdict](../run_diagnostics/pitchingplus/enhance_round2_verdict.json)

## Location+ remediation follow-up: measurement side closed, target side closed

Two bounded remediation steps addressed the Location+ gap (0.7724/0.7633 versus
FanGraphs in 2025/2026). Neither changed a serving file, adopted feature state,
frozen artifact or the promotion gate.

### Taxonomy/aggregation audit (diagnostic only, no model fit)

The audit recomputes pitcher-season Location+ agreement under alternative
measurement conventions on the same frozen per-pitch scores and cached
FanGraphs responses. The current method reproduces the frozen comparison
exactly (verified to 1e-12 against `validation.json`), so every difference
below is attributable to the variant alone.

| Variant (Location+ r) | 2025 | 2026 | vs baseline |
|---|---:|---:|---:|
| pitch-weighted, all pitches (current) | 0.7724 | 0.7633 | — |
| PA-weighted | 0.7378 | 0.7370 | −0.035 / −0.026 |
| appearance-weighted | 0.7548 | 0.7340 | −0.018 / −0.029 |
| paired exact labels only | 0.7415 | 0.7568 | −0.031 / −0.006 |
| paired pooled labels only | 0.7621 | 0.7634 | −0.010 / +0.000 |
| like-for-like, exact | 0.7839 | 0.7754 | +0.012 / +0.012 |
| like-for-like, pooled | 0.7881 | 0.7638 | +0.016 / +0.001 |

Season-aggregation weighting is not the problem: pitch-usage weighting is
already the best of the three conventions, and PA-weighting introduces a
+4.15-point mean gap. Taxonomy pooling helps only slider-heavy 2025 (type-level
SL 0.674 → 0.722) and does nothing for 2026, where sinkers are the worst type
(SI 0.638). Pricing both vendors' type grades with identical local usage
weights — removing FanGraphs' proprietary overall aggregation entirely — is
the only consistently favorable variant, and it reaches only 0.788/0.775.
Measurement-side factors cannot close the gap.

### Pre-registered Location+ target-realignment trial

Protocol registered at 2026-10-25T03:42:53Z, before any run, with the runner's
SHA-256 and every guarded input hash. Five pre-declared training targets under
the production location-model budget (180 trees, 31 leaves, identical for
every candidate); train Jan–Aug 2024, selection strictly by Sep–Dec 2024 raw
observed run-value MSE on one shared 125,513-row holdout set; 2026 rows never
loaded; the verdict is written to disk before any FanGraphs file is opened.

| Candidate (target) | Holdout raw MSE | Paired t vs baseline |
|---|---:|---:|
| baseline (expected-contact) | **0.04612661** | — |
| raw observed run value | 0.04613378 | +0.69 (indistinguishable, not better) |
| expected-contact, non-BIP rows only | 0.04705731 | +16.11 |
| contact-neutral BIP (no xwOBA term) | 0.05141163 | +39.73 |
| strike-minus-ball (+1/−1) | 0.64535561 | +401.67 |

**Verdict: no_integration.** Every challenger is worse than the production
expected-contact target; none is selected. Descriptive FanGraphs agreement
(read only after the verdict) ranks candidates differently: raw observed 0.7915
versus baseline 0.7699, contact-neutral 0.6545, strike-minus-ball 0.5391. The
raw-target candidate therefore tracks external Location+ better while tying on
run prevention — selecting it on that basis would be exactly the FanGraphs-
fitted tuning both earlier verdicts forbid, and its holdout bias is slightly
worse (−0.0016 versus −0.0010). It is recorded as descriptive evidence only.

### Standing conclusion

Within this model class and budget the Location+ gap is not remediable through
measurement-side taxonomy/aggregation conventions or through training-target
realignment. Remaining untested levers are location-specific input
formulations (for example the in-repo empirical-location-value comparator)
and genuinely new evaluation data: the completed World Series window plus a
fresh external snapshot. `PITCHINGPLUS_PROMOTION_APPROVED` remains False.

- [Taxonomy/aggregation audit runner](../run_diagnostics/pitchingplus/location_taxonomy_aggregation_audit.py)
- [Audit results](../run_diagnostics/pitchingplus/refined/location_taxonomy_aggregation_audit.json)
- [Registered target-trial protocol](../run_diagnostics/pitchingplus/location_target_protocol.json)
- [Target-trial runner](../run_diagnostics/pitchingplus/location_target_trial.py)
- [Target-trial selection verdict](../run_diagnostics/pitchingplus/location_target_trial_verdict.json)
- [Post-verdict FanGraphs description](../run_diagnostics/pitchingplus/location_target_trial_fg_descriptive.json)

### Empirical location-value challenger (pre-registered, no integration)

The ready in-repo comparator was tested as-is: strictly prior-date pitch-type ×
count-bucket × batter-side × 2-inch-cell prices, 3×3 expansion smoothing, the
existing context floor and shrinkage. A protocol with runner/source hashes was
registered before the run. Training used only rows before Sep. 1, 2024; selection
used the Sep.–Dec. 2024 holdout, common raw observed run-value rows, and the
production Location+ LightGBM budget. No FanGraphs response was read, and no
2025/2026 pitch rows were loaded.

| Method | Common holdout raw-RV MSE | Gain vs train-mean constant |
|---|---:|---:|
| Learned Location+ | **0.04610756** | **3.67%** |
| Empirical location value (`context_mean + PIT cell credit`) | 0.04686354 | 2.09% |
| Context mean only | 0.04786361 | 0.00% |

The empirical comparator is worse than learned Location+ by **0.000756 MSE**;
paired per-pitch loss difference is +21.77 SE. It captures useful location signal
versus the constant, but does not clear the pre-registered challenger gate.
**Verdict: no integration.** The runner and [registered protocol](../run_diagnostics/pitchingplus/location_empirical_protocol.json)
produce the [selection result](../run_diagnostics/pitchingplus/location_empirical_trial_verdict.json).

### New-data evaluation status

The frozen one-shot World Series fetch contains only 524 rows over Oct. 19 and
Oct. 23 (260 LCS, 264 WS); evaluation scored 523 pitches and correctly reported
**insufficient sample** under the registered 2,000-row minimum. The fetched
window was explicitly incomplete. The authoritative MLB schedule still lists
Game 2 on Oct. 24 and Games 3–7 as potentially upcoming ([2026 postseason schedule](https://www.mlb.com/news/2026-mlb-playoff-and-world-series-schedule)),
so the final permitted completed-window rerun is not yet available. The local
2026 FanGraphs response is the Oct. 24 snapshot; this enhancement has not fetched
a fresh snapshot. Do not refresh external data or refit against it before the
pre-registered final OOS rerun. No external-agreement conclusion or promotion
claim follows from the interim postseason sample.

- [Empirical trial runner](../run_diagnostics/pitchingplus/location_empirical_trial.py)
- [Empirical trial protocol](../run_diagnostics/pitchingplus/location_empirical_protocol.json)
- [Empirical trial verdict](../run_diagnostics/pitchingplus/location_empirical_trial_verdict.json)
- [World Series fetch provenance](../run_diagnostics/pitchingplus/oos/fetch_provenance.json)
- [Interim OOS results](../run_diagnostics/pitchingplus/oos/oos_evaluation.json)

## Reproduce and inspect

From the repository root, without a pitch repull:

```bash
python run_diagnostics/pitchingplus/validate.py
python run_diagnostics/pitchingplus/integration.py
python run_diagnostics/pitchingplus/diagnose_2025.py
python run_diagnostics/pitchingplus/sinker_target_probe.py
python run_diagnostics/pitchingplus/enhance.py
python run_diagnostics/pitchingplus/enhance_noise_check.py
python run_diagnostics/pitchingplus/enhance_round2.py
python run_diagnostics/pitchingplus/location_taxonomy_aggregation_audit.py
python run_diagnostics/pitchingplus/location_target_trial.py
python run_diagnostics/pitchingplus/location_empirical_trial.py
python -m pytest -q -rs \
  mlb-backend/backend/test_pitchingplus_trial.py \
  mlb-backend/backend/test_bullpen_availability.py \
  mlb-backend/backend/test_per_side_twins.py \
  mlb-backend/backend/test_run_engine_contract.py \
  mlb-backend/backend/test_grid_rfe.py \
  mlb-backend/backend/test_run_log_observability.py \
  mlb-backend/backend/test_log_review_20261007.py \
  mlb-backend/backend/test_log_review_20261008.py
```

The integration runner disables unrelated production artifact writers and
writes only diagnostic outputs. Cached FanGraphs responses are read only by
validation; normal feature generation never fetches them. The bounded development
experiments are recorded separately. The refined candidate uses expanded
tracking and expected-contact labels; it still fails the full promotion gate
and was not silently promoted.

- [Refined validation metrics](../run_diagnostics/pitchingplus/refined/validation.json)
- [2025 refined comparison](../run_diagnostics/pitchingplus/refined/final_comparison_2025.csv)
- [2026 refined comparison](../run_diagnostics/pitchingplus/refined/final_comparison_2026.csv)
- [Final model-origin diagnostics](../run_diagnostics/pitchingplus/refined/final_model_diagnostics.json)
- [Game-feature coverage, ranges, runtime](../run_diagnostics/pitchingplus/refined/integration.json)
- [Weather and notebook checks](../run_diagnostics/pitchingplus/refined/final_checks.json)
- [Final test log](../run_diagnostics/pitchingplus/refined_tests_final.log)
- [Bounded candidate-selection metrics](../run_diagnostics/pitchingplus/refine_metrics.json)
- [Original learned-candidate validation](../run_diagnostics/pitchingplus/validation.json)
- [2025 external request provenance](../run_diagnostics/pitchingplus/fangraphs_2025_source.json)
- [2026 external request provenance](../run_diagnostics/pitchingplus/fangraphs_2026_source.json)

The Kaggle notebook is byte-identical to the captured baseline (SHA-256
`4e1b0edd4d66cac7b1d6d87c107bfdc1ae3240d15f6df3c18d625a8cf5ce4be0`). Its
existing `MLB_FULL_REPULL` control was not changed. No new historical pitch pull,
production model retrain, Git commit, push, or deployment was performed.
