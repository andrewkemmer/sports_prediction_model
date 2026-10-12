# NBA run-log review — 2026-10-11

Scope: the delivered `nba-backend/data_delivery/nba_pipeline_run_log.txt` — 254
lines, run **2026-10-11 05:20:34 → 06:11:35 (3,061s)**, `NBA_FULL_REPULL=1`,
delivered by `8f6f4b2d` ("Update NBA features + predictions") + `323aceac`
("final delivery").

**Verdict: 0 ERROR / 0 Traceback; 14 WARNING lines, all classified below; 2
findings remediated (the drift monitor's off-season window went silently
stale; the dispersion line's "(curve max)" parenthetical reads as a violated
bound); 1 pre-existing environment-dependent test failure repaired; every
material claim reconciled against the delivered artifacts.**

## Log line review — all 14 WARNINGs

| Line | Message (abridged) | Verdict | Disposition |
|---|---|---|---|
| 8× across chunks | `dropping <id> (EAST vs WEST / CHK vs KEN / STARS vs WORLD …): a side is not one of the 30 franchises` | By design — the four 2024 All-Star, three 2025 All-Star (skills-challenge) and four 2026 All-Star-format games are not scoreable by this model; the drop names the reason. | None. |
| schedule | `27 scheduled game(s) have no season-log match …` and `27 game(s) carry no NBA game id …` | Same 27 games seen from two angles; reconciles exactly: **11 postponed** (excluded from the slate, listed at 06:01:19) **+ 2 finished without player lines** (MIL@OKC 2024-12-17, SAS@NYK 2025-12-16 — excluded from training, disclosed as postponed/neutral-site, not a join failure) **+ 14 upcoming slate games** (2026-10-20/21, obviously not yet played). 11+2+14 = 27. | None. |
| position index ×2 | `covers only 135 of 572 2023-24 players (24%, floor 90%); this season keeps the G/F/C convention` (and 2024-25 at 23%) | By design — the position archive predates those seasons; the fallback convention is disclosed rather than silently applied. 2025-26 reads 100%. | None. |
| designations | `no filing lists DAL@GSW before tipoff on 2024-01-19: game uncovered` | Disclosed archive gap for one 2024 game; the 14 upcoming games' `pre-submission` lines are by design (filing windows close near tipoff). | None. |
| folds | `OOF validation population is thin on 16 of 56 window(s) (< MIN_VAL_FOLD_GAMES=40) … PROVISIONAL` | Working as designed — playoff weeks and season ramps; provisional rows never grade the blend. | None. |

## Claims reconciliation — every material count against the artifacts

| Log claim | Against | Verdict |
|---|---|---|
| `NBA facts: 3488 games … 6922 event rows`; `projected lineups: 6976 team-game(s)` | 3488 × 2 = 6976; event rows = 3461 decided × 2 = 6922 | ✓ |
| `pl_rapm features attached: 3266/3461`; `195 lack … 104 + 19 + 72` | 3266 + 195 = 3461; 104+19+72 = 195 | ✓ exact |
| `Published blend: 2400 OOF rows … weights elasticnet 65.2 / lightgbm 30.2 / xgboost 4.6`; retrospective replay `n=2130` | monitor JSON ensemble (0.6522/0.3021/0.0457, n_eval 2130); 2400 = 2130 grading + 169 postseason + 101 provisional | ✓ |
| `sealed holdout gate: … 2389 pre / 11 sealed rows` | 2389 + 11 = 2400 | ✓ |
| `feature drift: 71 features, 0 warning(s), 10 alert(s) [alerts: …]` | `nba_run_engine_feature_drift_20261011.csv`: 71 = 61 OK + 10 ALERT — counts **sum exactly** (no hidden statuses; the T15 class is absent) | ✓ (see Finding 1 for the window) |
| `SHAP attributions written for 14 game(s)`; `14 upcoming game(s) scored` | slate = the 2026-10-20/21 games in the 27-list | ✓ |
| `✅ status ok, 3061.11s` vs log span 05:20:34 → 06:11:35 | 3,061s | ✓ |
| `rolling Brier: 0.2091 … 2400 game(s)`; `designation archive published: 37211 record(s)` = loaded 37,211; `event rollup: 6922` | monitor JSON / archives | ✓ |

Also reviewed, no defect: `alpha_home 0.0040 (curve max 0.0034)` — reconciles
once read as the two different estimators it is (pooled MoM scalar vs
per-row curve diagnostic; see Finding 2 for the wording fix); the lineup
features' `11/3488 rows … CONSTANT` disclosure (the feature is young and
mostly pool-refused historically — disclosed, not silent); the drift
window's widening + composition-matching (both logged, both careful).

## Finding 1 (remediated): the drift monitor's "current" window was silently frozen

The run printed:

```
drift window: only 3 decided game(s) in the last 7 day(s); widened to 37 day(s) for a current window of 31
drift window: current window is 100% is_playoffs against a 22% prior; baseline composition-matched to 220 is_playoffs row(s) …
feature drift: 71 features, 0 warning(s), 10 alert(s); …
```

The window builder did its job — but the window's **newest decided game was
2026-06-13, 120 days before the run**. The same 10 ALERTs (`elo_home`,
`elo_away`, three `ewm_*`, `event_live_tov_rate_home`, four `pl_rapm_*`)
had been printed identically for weeks: the delivered CSVs for
20261006–20261011 carry the same alert set. Nothing in the log said the
comparison was frozen, so a reader reasonably took "10 alerts" as a
statement about the October slate being served — a comparison the June
tail structurally cannot make (off-season roster and trade movement is
invisible to it). Same disclosure family as MLB's T15 / NHL's hidden
INSUFFICIENT: the line omitted the fact that changes its interpretation.

**Remediation** (`nba-backend/backend/monitoring.py`, disclosure only —
statuses, windows and thresholds untouched): `drift_windows` now takes an
optional `as_of` and, when the newest decided game is more than
`DRIFT_STALE_DAYS = 14` before the run, warns:

```
WARNING drift window: newest decided game is 2026-06-13 — 120 day(s) before this run (staleness floor 14); statuses measure that frozen 31-row tail against its baseline, NOT the slate being served. Off-season roster and trade movement is UNMEASURED until play resumes, and recurring alerts repeat the same frozen comparison — they are not new signal
```

14 days clears every in-season gap (the All-Star break is ~1 week; the
longest gap between decided NBA games is days), so the line fires only
when the clock has genuinely stopped.

**Pins** (`nba-backend/backend/test_log_review_20261011.py`, 4 tests):
the stale disclosure (window end, age, "NOT the slate being served",
"UNMEASURED", "not new signal"; windows unchanged); a fresh window stays
quiet; the boundary is exactly `DRIFT_STALE_DAYS` (quiet) vs +1 (warns);
and the default `as_of` is the wall clock.

## Finding 2 (remediated): "(curve max)" reads as a violated bound

`calibrated NB dispersion …: alpha_home 0.0040 (curve max 0.0034)` puts a
number *above* its own parenthetical, which reads as "the fit exceeded its
allowed range". The two numbers are different estimators — the shipped
scalar is the pooled method-of-moments estimate the draw path consumes;
`alpha_*_max` is the per-row maximum under the selected diagnostic curve
(different rows, different λ). Reworded in
`nba-backend/backend/master_pipeline.py` to `(per-row curve max …)` so the
parenthetical cannot be misread as a cap. NBA-only: MLB and NHL print no
such parenthetical (grep-verified).

## Finding 3 (repaired, pre-existing): the torn-line tee pin could not pass on Windows

`test_run_log_tee.py::test_torn_plain_line_waits_for_its_terminator`
failed in isolation on this machine: the test opened its capture file with
the locale default (cp1252 on Windows), and writing `"✅"` raised
`UnicodeEncodeError` — a `ValueError` subclass that `_Tee.write`'s
disk-failure guard swallows by design — so the file stayed empty and the
pin asserted against nothing. Production (`install_run_log_tee`) opens the
tee file `encoding="utf-8", errors="replace"`, so Kaggle deliveries are
unaffected; only the test diverged from the production contract. Fixed by
constructing the file and reading it back under `encoding="utf-8"`. The
same test exists only in NBA's suite (grep-verified across the other three
leagues).

## Verification

- `python -m pytest -q` in `nba-backend/backend` → **629 passed, exit 0**
  (625 baseline + 4 new pins), re-run after the final edit.
- The reconciliation tables above were executed against the delivered
  artifacts (pandas/JSON), not eyeballed.
- The staleness behavior was exercised against the real `drift_windows`
  with a synthetic decided frame and explicit `as_of` clocks — no live
  pipeline run.

## Deliberately not changed

- Drift statuses, judge floors, window construction, composition matching —
  disclosure only; the 10 alerts are a legitimate year-over-year
  playoff-tail signal once the window's age is disclosed.
- The 27-game disclosures listing upcoming slate games — noisy but honest;
  the "not swept" wording is correct for games without play-by-play.
- OOF provisional policy, calibration gates, lineup pool-refusal, the
  position-index fallback — all documented behavior.
- `mlb/nfl/nhl-backend` each carry their own copy of `drift_windows`; the
  same staleness gap exists there (their off-seasons will freeze the
  window too). Flagged for those leagues' next reviews, not fixed here.

## Residual risk / next-run expectations

- From the next run until the 2026-27 schedule produces decided games
  (~2026-10-21+), the log carries the `drift window: newest decided game …
  UNMEASURED` WARNING on every run — correct by design, not a regression.
  Once the season starts, the newest game is days old and the line
  disappears.
- If the same 10 alerts persist INTO the new season with a fresh window,
  that is real signal about the 2026-27 population and belongs to the
  model owner, not this review.


---

# Four-sport distribution-model deep dive and CRPS remediation — 2026-10-11

## Verdict and acceptance boundaries

**The prior distributions were not professional-grade CDFs. This change repairs
causal distribution mechanics and coherent pricing, implements CRPS-driven
conditional calibration, and exposes the remaining accuracy deficit. It does
NOT certify positive ROI, high offered-line AUC, or universal CRPS improvement.**

Constraints honored: only existing production/test/report files changed; no new
program committed, no new SDK/dependency, no binary moneyline feature adoption,
no automatic production pipeline/delivery push. Historical first-publication
stores and existing delivery artifacts were not rewritten. A full daily rerun
is required to publish new-model artifacts; source push is not model deployment.

Acceptance interfaces are the existing `ScoreRegressor`, `walk_forward_oof`,
`apply_distribution`, `calibrate_market_frame`, `apply_market_calibration`,
`nb_distribution_metrics`, and MLB `derive_markets_v3`/`predict_slate_runs`/
`persist_markets`. Existing integer grid names stay available; half grids,
origin provenance, full-support CRPS and reliability metrics are additive.
A new function INSIDE each existing distribution module,
`distribution_probability(frame, bundle, kind, line)`, prices any margin/total
threshold directly from the same CDF. Home handicap -3.5 means margin threshold
+3.5. Integer prices return strict win / push / loss, not push-inclusive wins.

## Measured defects in delivered 2026-10-11 artifacts

| Sport | OOF rows | Totals games with CDF crossings | Spread games with crossings | Largest totals / spread upward jump |
|---|---:|---:|---:|---:|
| NBA | 2,400 | 2,242 | 2,341 | .44464 / .23836 |
| NFL | 2,567 | 565 | 2,536 | .25169 / .30765 |
| NHL | 2,683 | 127 | 2,630 | .04704 / .42062 |

Increasing a line MUST NOT increase its strict exceedance probability. These
are not harmless rounding differences. NHL derived home + away + tie also
missed one by up to .02612. NBA/NFL derived-ML calibration fitted all OOF labels
then rewrote the same OOF probabilities, unlike the line calibrators. NHL's
favored-space correction was prequential but could disconnect ML from the
spread CDF and produce negative/incoherent residuals.

Other defects and limitations:

* Poisson deviance training was not CRPS optimization; independent side draws
  impose Cov(home, away)=0, so total and spread variances cannot adapt separately.
* Global pre-holdout dispersion reused on ALL OOF rows sees future outcomes
  relative to older rows. A 21-day sealed tail does not make pooled OOF causal.
* Alpha-form selection used random A/B halves and reverse-half fits. It now
  uses forward prefix-to-next-block selection for retained LEGACY diagnostics.
* NBA's legacy NB validation used `kmax=80` against scores around 110 and
  `tail_k=10`; this cannot validate basketball tails. That diagnostic is not
  used for new production prices. Legacy pooled alpha includes predictable
  between-game mean variation; the new count baseline uses OOF residuals.
* Old fair lines were integers, sometimes a grid boundary, and calibration did
  not reliably refresh the fair line / fair probability / outcome aliases.
* Old evaluators mostly reported binary metrics, not full-distribution CRPS;
  pushes were sometimes coded as losses or removed without conditioning prices.
* NHL accepted obsolete sigma arguments while ignoring them. They remain
  compatibility-only on old diagnostic APIs, not claimed as active parameters.

## Production architecture and causality

1. Preserve the existing resolved, market-free moneyline TREE feature contract
   and existing score-mean regressors. Genuine NaNs retain native tree routing.
   Supplied train/validation folds must be separated by normalized game dates;
   violations raise. NBA no longer silently substitutes ridge after a fit error.
2. Use outcome-independent fixed integration supports: NBA 0..600 totals,
   +/-600 margin; NFL 0..250 / +/-250; NHL 0..100 / +/-100; MLB 0..160 / +/-160.
   Check represented tail mass and refuse an insufficient support. Targets
   outside support receive boundary tail error; observations aren't clipped.
3. NBA: discretized, nonnegative-normal total and margin BASE CDFs. Each is
   independently corrected, removing the forced equal-variance relationship.
   Count sports: exact Poisson/NB score-PMF convolutions; residual dispersion
   estimated on PRIOR OOF only, floor at Poisson, bounded .5. NFL lumpy/key-score
   structure is still a limitation; no claim this count base is uniquely optimal.
4. Optimize a six-parameter monotone CDF map separately for totals and margin:
   `F*(k)=sigmoid(exp(X beta_scale)*logit(F(k))+X beta_location)`, with endpoint
   preservation. X contains intercept, expected-total level, absolute expected
   margin. Width and location vary by pre-game scoring/favorite scenario.
   Ridge .01 and fixed bounds are predeclared, not tuned on reported test rows.
   This is a compact conditional CDF calibrator, NOT a full feature-conditioned
   variance learner. It can correct distribution mean as well as scale.
5. Fit by unit-lattice CRPS, `sum_k (F(k)-1[y<=k])^2`, with analytic gradients.
   One increasing map for the ENTIRE CDF guarantees no crossings/nonnegative
   pushes. All over/under/cover/ML prices are CDF differences. NBA/NHL/MLB
   final-score ties are resolved symmetrically to +/-1 before scoring; NFL
   keeps ties and reports non-tie-conditional ML binary metrics separately.
   NHL shootout-final-score convention should be validated against any future
   bookmaker's settlement rule; regulation-only markets are NOT interchangeable.
6. Forecast origins are chronological, indivisible blocks, ordered by actual
   dates (NOT numeric fold labels or arrival order). Reject missing dates,
   overlapping origins, non-finite/nonpositive means, duplicate indices,
   unsettled/fractional/negative final scores. The first 200 prior forecasts
   use identity calibration. No fitted global alpha/CDF map is replayed on OOF.
7. Admit calibration at each origin only if its paired CRPS on PREVIOUSLY
   issued origin forecasts beats identity in mean, with >=200 prior outcomes.
   This selection is itself prequential; the current origin never grades its
   own correction. This is an adaptive admission rule, NOT a statistical
   significance guarantee or a general promise of future improvement.
8. Final serving map may use all settled past OOF evidence and admission
   results. It is never applied back to old OOF rows. Artifact metadata names
   the method/objective; fold ids and fit-end/count are retained for exact
   future replay. Calibrated mean/SD are separate from raw score-mean aliases.
9. Half-point nearest-50% fair lines are computed from the actual CDF, even
   outside displayed grids. Half stops have zero push. All outcome aliases
   are refreshed after correction. Any offered line can use the exact API.
10. Retain legacy MC/Platt APIs for compatibility diagnostics only; undated
    fixtures are not labeled PIT. Legacy alpha and Pearson monitor entries
    describe legacy SCORE diagnostics, not the new CRPS predictive CDF.

The four backends run independently in Kaggle. The compact layer is mirrored
in their existing modules instead of adding a shared runtime program. Tests pin
common properties and sport-specific settlement/supports. This duplication is
a maintenance trade-off imposed by the no-new-program guardrail.

## NBA architecture experiment — rejection, not optimistic adoption

A direct total/margin LightGBM location challenger trained by analytic Gaussian
CRPS, with train-only initial mean/scale and correct gradient/Hessian, was run
on all 56 existing shared folds / 2,400 OOF games. It produced total CRPS
10.97378, margin CRPS 8.61267 and derived-ML AUC .69091 / logloss .63281 /
Brier .22127. The existing score means with whole-CDF correction were better
(total 10.84870 / margin 8.37864 before nested admission; derived ML AUC .71155 /
logloss .61770). **Direct CRPS tree location training is REJECTED as default**;
its optional existing-module implementation remains available for controlled
research. Objective finite-difference tests pass. No retrain-on-validation early
stopping or metric cherry-picking was used to rescue the challenger.

The existing scratch NBA margin-feature ablation also showed no binary lift:
A baseline LL .6112 / AUC .7194; B (margin + home/away mean) LL .6118 / AUC .7181;
C (also total) LL .6113 / AUC .7189. Therefore binary feature adoption remains
unchanged. A new causal bridge ablation is necessary after any admitted
mean-model improvement; a distribution cleanup alone is not proof of blend lift.

## Empirical protocol and limits

Latest delivered OOF means/outcomes were replayed at chronological blocks of
seven observed game dates, matching the intended family cadence. Old market
CSVs and MLB OOF CSV omitted fold IDs, so these are **reconstructed calibration
origins**, not a certified byte-identical training replay. New artifacts persist
origin IDs. NBA direct-location challenger alone used the actual shared folds
and the cached 2,629-row feature frame (71 numeric features + categoricals).
The cached frame is an earlier production feature snapshot; current feature
contract/timestamp provenance still requires the next full pipeline replay.

These are retrospective software/conditional-distribution evaluations, not a
prospectively untouched final acceptance season. Architecture choices were
informed by these results; an independent chronological acceptance season is
still required. Confidence intervals resample the same paired forecast-origin
blocks 2,000 times (seed 451), preserving within-block games. They do not cover
all model-selection or source-revision uncertainty. CRPS units differ by sport.

### Final gated model vs SAME raw base CDF (calibration-only attribution)

| Sport | N / origin blocks | Total CRPS base -> gated (95% delta CI) | Margin CRPS base -> gated (95% delta CI) |
|---|---|---|---|
| NBA | 2400 / 56 | 10.96251 -> 10.86332 (-0.14012, -0.05594) | 8.37449 -> 8.37623 (+0.00000, +0.00568) |
| NFL | 2567 / 80 | 7.69656 -> 7.69900 (-0.00118, +0.00667) | 7.44919 -> 7.43868 (-0.02331, +0.00340) |
| NHL | 2683 / 60 | 1.29150 -> 1.29013 (-0.00368, +0.00110) | 1.44397 -> 1.44346 (-0.00309, +0.00228) |
| MLB | 7032 / 84 | 2.46561 -> 2.46599 (-0.00113, +0.00192) | 2.46907 -> 2.46972 (+0.00000, +0.00164) |

Interpretation: NBA totals calibration has a measurable paired improvement;
NBA margin correction is essentially neutral/slightly harmful and the FINAL
serving gate rejects it. NFL total/MLB total+margin FINAL corrections are also
rejected. NFL margin and NHL corrections have mixed/weak evidence (CIs include
zero), not demonstrated generalizable superiority. Adaptively admitted early
corrections can still hurt later realized CRPS; admission does not erase losses.

Comparing a different family baseline (prior-residual NB independent-score CDF)
produced NBA total 10.86466 / margin 8.43404 vs new 10.86332 / 8.37623. That
comparison reflects BOTH the marginal family and correction changes; do not
attribute the entire difference to calibration or claim a large totals gain.
NFL/MLB new whole-CDF forecasts did not beat that prior-NB baseline universally.

### Binary diagnostics at nearest half-line to each game's 50% probability

| Sport | Total AUC / LL / Brier; actual over rate | Spread AUC / LL / Brier; actual cover rate | Derived ML AUC / LL / Brier |
|---|---|---|---|
| NBA | 0.5033 / 0.69316 / 0.25001; 0.5029 | 0.5071 / 0.69308 / 0.24996; 0.5029 | 0.7119 / 0.61717 / 0.21438 |
| NFL | 0.5120 / 0.69287 / 0.24986; 0.5060 | 0.4825 / 0.69391 / 0.25038; 0.5029 | 0.6986 / 0.63060 / 0.21999 |
| NHL | 0.5396 / 0.69131 / 0.24908; 0.5114 | 0.5798 / 0.68351 / 0.24522; 0.4957 | 0.5888 / 0.67731 / 0.24220 |
| MLB | 0.5181 / 0.69293 / 0.24989; 0.5011 | 0.5486 / 0.68943 / 0.24814; 0.5219 | 0.5710 / 0.68444 / 0.24568 |

AUC at a MODEL-SELECTED 50% threshold is not a meaningful high-AUC objective:
probabilities are intentionally nearly constant and the per-game event changes
with the model. Even a correctly specified forecast often has AUC near .5 here,
LL near log(2), Brier near .25. Correct calibration is predicted .5 vs actual .5,
plus uncertainty and conditional strata—not a fabricated ranking win. AUC
must ALSO be reported at common fixed thresholds and actual offered lines on
the SAME settled games. CRPS scores the whole CDF; proper threshold logloss/
Brier diagnose local misspecification. High fixed-line AUC alone does not prove
an edge against a sportsbook whose line already embeds the signal.

### Retrospective archived bookmaker totals-line diagnostic (NOT PIT ROI)

| Sport | Matched / nonpush | AUC | Conditional logloss | Brier | Mean predicted / actual over |
|---|---|---:|---:|---:|---|
| NBA | 1510 / 1510 | 0.5027 | 0.73645 | 0.26876 | 0.5126 / 0.4980 |
| NFL | 2563 / 2546 | 0.5116 | 0.71570 | 0.25979 | 0.4872 / 0.4874 |
| NHL | 2613 / 2613 | 0.5095 | 0.70404 | 0.25529 | 0.4880 / 0.4971 |
| MLB | 6548 / 6434 | 0.5092 | 0.70304 | 0.25469 | 0.4905 / 0.4949 |

Books are deduplicated by family (state variants count once), live/opening/model
feeds excluded, median total line used, pushed outcomes removed and probability
conditioned on no push. **Every sport is worse than the .69315 LL / .25 Brier
50/50 baseline in this diagnostic. No standalone totals betting edge is proven.**
A lack of ranking power at offered lines is materially more important than
apparently good fixed-threshold AUC. Do not use this diagnostic as a betting
backtest: archives were fetched after settlement, provider update timestamps
are null, spread quotes are absent, total over/under prices are absent, and
median lines across books are not necessarily executable quotes.

For genuine ROI, archive as-issued forecasts and exact bookmaker line+decimal
prices+quote timestamp BEFORE start; match settlement conventions; freeze edge,
bet selection and stake policies on past data; hold out a future season; report
win/push/loss counts, net profit/stake ROI, drawdown, CLV, coverage, paired
blocked uncertainty and bookmaker/line/regime strata. With decimal price d,
unit EV is p(win)*(d-1)-p(loss); push contributes zero. Two-sided de-vig and
no-push conditional probabilities must use compatible conventions. No invented
-110 price, post-result line or future-selected best edge is admissible.

## Feature PIT audit and unresolved provenance

The score model uses the production feature resolver, never outcomes, market
lines, closing probabilities or postgame facts as raw predictors. NBA ladders
use shifted trailing/EWM states, prior-settlement rest dates, and Elo updated
after the current prediction; team categories use the fixed UNK vocabulary.
Existing tests cover future outcome mutation and injury/lineup as-of binding.
OOF score training now explicitly rejects same-day/future train rows even for
caller-supplied folds. No OOF validation labels determine early stopping for
these score models; MLB production uses its existing fixed-round policy.

**Not fully certified here:** original source publication timestamps vs later
API revisions, historical injury/player-rating archives for every feature,
and globally adopted feature-subset selection dates. A feature can be shifted
correctly but still be reconstructed from revised statistics or a subset chosen
using a future season. A full bitemporal input ledger and feature-selection
freeze/replay are required for a complete strictly-PIT certification. A date
split is not proof that every label had settled before an intraday forecast;
the current workflow conservatively uses prior game DATES, not same-day results.

## Sport-specific remaining remediations / promotion gates

* **NBA:** admit only improvements that beat causal raw-NB AND outcome-free
  normal baselines in both full CRPS and offered-line LL/Brier; do not promote
  the rejected direct-location architecture. Investigate pregame pace, lineup
  continuity, rest and officiating information for total location; separately
  model uncertainty by pace/lineup/mismatch regime. Match historic lineup/injury
  publication times rather than actual starters. Negative market-line results
  mean the current totals model is research-only for ROI, even though its CDF
  is now valid. At the most recent 21-day tail there are only 11 NBA games—use
  broader seasons/multiple origins rather than claim decisive tail acceptance.
* **NFL:** the negative-binomial integer base is not a key-number/scoring-drive
  model. Candidate drive/possession or empirical residual mixtures must capture
  3/7-point mass, overtime ties and weather/tempo variance, gated chronologically
  against count baselines. Final total correction was rejected; offered-line
  discrimination remains weak. No ROI approval.
* **NHL:** explicitly separate final incl. OT/shootout from regulation markets;
  goalie confirmation/empty-net dynamics can change tails. Current small CRPS
  improvements do not clear block-bootstrap uncertainty. The symmetric tie
  resolution is a coherent approximation, not an independently proven overtime
  winner model. No automatic transfer of NBA parameters/thresholds.
* **MLB:** the count base earns much larger residual dispersion than NHL; retain
  that distinction. Both final CDF corrections are rejected. Require a new
  paired chronological count-family/score-mean gate before substantive model
  promotion; investigate starter/bullpen run variance and correlation separately.
  Do not equate the old home-weighted tie-resolution constant or margin-feature
  ablation with certified moneyline/ROI improvement.
* **All:** retain every regime in reporting; publish season/provisional/playoff
  strata, threshold reliability, calibrated mean/SD and origin metadata. A joint
  score-pair law is NOT reconstructed by separately corrected marginal total/
  margin CDFs; correlated parlays/joint bets are out of scope. Fair median is
  not mean; separate aliases now make the distinction visible.

## Verification and operational notes

Python 3.14.3, NumPy 2.4.3, pandas 3.0.1, LightGBM 4.7.0. Verification uses
public production interfaces plus actual CSV writers/readback, not just builds.
All four real historical market artifacts produced zero checked CDF crossings,
finite line prices and coherent three-way ML after the rebuild. Supports/tail
checks, absolute-error point-mass CRPS, analytic Gaussian gradients/Hessians,
future-outcome poisoning, shuffled-row invariance, overlapping-origin refusal,
settled-input refusal, zero half-line pushes and exact slate application are
covered in existing test files. Legacy contract assertions were updated to the
new whole-CDF behavior rather than skipped or weakened; underlying guarantees
are stronger. A small-sample alpha-bin neighbor index error uncovered by forward
selection was repaired at its cause.

Full backend verification/status is appended below after final checks. Existing
sklearn/pandas/Altair deprecation warnings remain. No live API ingestion,
external model deployment, or change to already-published forecast history was
performed. The no-new-program condition is checked in the staged git diff.

### Sources

* Proper CRPS formulas and distribution scoring:
  https://arxiv.org/html/1709.04743v2
* Ordered discrete distributions / isotonic distributional regression:
  https://arxiv.org/html/1909.03725v3
* Forecast-evaluation pitfalls and rolling origins:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/
* Calibration data independence / small-sample isotonic overfit:
  https://scikit-learn.org/stable/modules/calibration.html


### Final verification record

- NBA full backend: **634 passed**; after the last metadata/reporting edit,
  affected CRPS/structural suite **15 passed**, no failures; real LightGBM
  score fit/predict and OOF+outcome-free slate+CSV+CRPS-metadata smoke passed.
- NFL existing script-style production suite: **524 checks passed, 0 failed**;
  affected feature-contract/CRPS suite **4 passed** after final source edits.
- MLB distribution artifact/PIT suites: **60 passed** after the final CLI
  entrypoint-order repair, including legacy pre-holdout checks, production
  future-outcome poisoning, and execution of the real script entrypoint with
  ingestion/writes intercepted; CRPS helpers are defined before main runs.
- NHL affected distribution/grid/PIT subset: **53 passed**, including the sealed
  monitor test; after final metadata edits **10 affected checks passed**.
  Full 191-test grid/PIT suite was also rerun; its final result is recorded
  below when complete. This subset is targeted selection, not skipped tests.
- All four 260-row smoke deliveries passed public calibration, slate pricing
  where applicable, CSV persistence/readback and CRPS metadata.
- Historical real CSV checks: NBA 2,400 rows / 905 columns; NFL 2,567 / 505;
  NHL 2,683 / 197; MLB 7,032 / 110. Finite grids, zero checked crossings,
  coherent ML/push settlement; all artifact commands exited zero.
- Final source syntax/compile checks passed; no static typechecker configuration
  exists for these Python backends, so syntax/imports + real interfaces + tests
  were used rather than labeling an absent typecheck as passed.

Final review also moved MLB's script entrypoint below the CRPS helpers;
import-based pipeline tests alone would not have detected that CLI ordering
failure. A regression test executes the entrypoint and exercises its CRPS
market construction without live ingestion or production artifact writes.
CLI `--help` also passes with `PYTHONUTF8=1`; Windows' default cp1252 redirected
stdout cannot encode the pre-existing Greek symbols in the module help text.

Local rerunnable research/evidence is gitignored under `scratch/` (the baseline
copy, CRPS ablation, paired block-bootstrap, archived-line diagnostic and
artifact verification). Only this existing report and existing source/tests
are committed. Historic delivery artifacts remain untouched by this task.

- NHL full grid/PIT suite final result: **191 passed, 0 failed** (713.57s).
  Affected checks were rerun after the final reporting changes as noted above.
