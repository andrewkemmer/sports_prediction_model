# NBA RAPM predictive validation and coverage follow-up — 2026-10-09 ET

## Verdict

**Computational defects were real and are repaired. Predictive value is promising, not a blanket statistical pass.** The existing `pl_rapm_*` names represent a **game-level MIN/48 ridge proxy**, not conventional possession/stint RAPM. Forward features supply the strongest incremental evidence; center and guard families do not independently clear a convincing predictive gate. No hyperparameters, blend policy, injury policy, or feature membership were optimized/adopted on this replay.

This review supersedes the earlier audit's claim that RAPM was strictly point-in-time on every axis. Coverage alone missed post-game roster information and future-derived shrinkage. Source contract is now **`nba-prod-v2.8-rapm-strict-pregame`**. Features, OOF, calibration/blend and final models must be rebuilt together before production adoption; **production delivery files/model bundles were not replaced**.

## Calculation and causal repairs

1. **Target-day roster/trade/recency leakage — high.** `known <= target` admitted the target game's actual participants, moved traded players before their first knowable appearance, and refreshed stale players to zero days since appearance. Pending games cannot see those box lines. Membership, latest team and last positive-minute appearance now use **strictly prior dates**. DNP rows cannot establish team movement or refresh recency. A debutant is unknown until a prior appearance exists; do not infer pregame membership from a future box score.
2. **Earliest-season shrinkage leakage — high.** The earliest season used a mean over the entire multi-season frame, including later games/seasons, to set `k_eff`. It now uses the configured fixed fallback when no completed prior season exists. Targets in a new season absent from the player frame compute their own `k` over all completed priors instead of reusing the preceding target season's older table.
3. **Overtime exposure clipping — medium.** `MIN/48` was clipped at 1, undercounting 35 observed player-games over 48 minutes. Exposure is now linear in actual minutes, including overtime; nonfinite/nonpositive minutes supply no participation. Team shares sum to approximately **five**, not one. The existing source rounds minutes (team totals 236–292), so this is not exact possession exposure.
4. **Partial official-table fallback — medium.** One game missing its official margin previously reoriented every game lexicographically, including supported home/away sides. Fallback now reorients only unsupported games, retaining official sides elsewhere. The production snapshot has complete official margins; the repair prevents a latent partial-input regression.
5. **Metadata and validation mismatch.** Dashboard RAPM tooltips incorrectly said points per 100 possessions. They now say game-margin points per full-48-minute player exposure and describe the game-level proxy. The existing A/B program now consumes the production fact population (including playoff evidence), honors the configured cache/delivery archive, grades the correct regular/nonprovisional rows, scores the served calibrated probability, rejects incomplete/unpaired OOF and resamples dates together. Its output stays in ignored diagnostics rather than a hard-coded `/tmp` file.
6. **Coverage accounting.** `pct_nonnull` incorrectly repeated `pct_measured` when values were carried. It now reports actual non-null share while measured counts and alarms remain observation-based. A regression asserts 100% present / 50% measured stays LOW_COVERAGE, never a fabricated pass.

### Independent arithmetic and timing evidence

[Arithmetic evidence](nba_rapm_review_20261009/arithmetic_checks.json): independently assemble signed player columns and an unpenalized intercept for 1,134 prior 2025-26 games / 574 players at 2026-04-01, solve `(X'X + diag(lambda,...,lambda,0)) beta = X'y`, then compare to the incremental production solve. Maximum beta difference **1.60e-14**; intercept difference **1.11e-15**. Ridge, signs, intercept and accumulation agree numerically. This proves the implemented proxy's linear algebra, not basketball identification from unobserved stint overlap.

[Real-input causal checks](nba_rapm_review_20261009/causal_checks.json) compare entire rating rows—not only values—on five target dates spanning cold start, both opening nights, trades and late-season play. Prefix deletion of target/future player and team facts and poisoning future minutes/points/team identity produce **zero repaired differences** in IDs, teams, priors, `k`, raw/shrunk ratings, recency and projected pools. The legacy build fails: 512 `k` and 511 shrunk-rating mismatches at 2024-03-01; 234 recency and 12 team mismatches at 2025-02-08; opening-night roster/prior means differ sharply from pending replay. Same-date withholding is deliberately conservative; this feed has no reliable publication instant for player box facts.

## Predictive validation: matched production walk-forward, not player-rank storytelling

- Same source population: 3,461 decided games, 2,629 core games, 56 expanding observed-date folds, 2,400 OOF games, **2,130 grading games** on 280 dates; 169 postseason and provisional rows remain scored and are reported separately.
- Same seed 42, feature builders, configured model budgets, train-fold preprocessing, train-only early stopping, chronological adaptive weights and nested calibration. All three members produce finite predictions in every arm; **no failed member is silently accepted**.
- Six preregistered contrasts: remove all RAPM; legacy RAPM; repaired RAPM; repaired minus each of C/F/G. No tuning/retraining on a selected winner. Canonical game IDs, labels and grading masks match across arms.
- Headline below uses **`p_ensemble_calibrated`**, the actual served-probability path. Raw causal, member, all-row, season, postseason and last-300 diagnostics are preserved in [metrics](nba_rapm_review_20261009/predictive_metrics.json). Final-weight retrospective probabilities are never reported as OOF.

| Grading population (n=2,130) | AUC ↑ | log loss ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| No RAPM | 0.728627 | 0.606397 | 0.209400 | 0.026142 |
| Legacy RAPM, causally flawed | 0.733499 | 0.602073 | 0.207633 | 0.024733 |
| **Repaired RAPM** | **0.731227** | **0.602610** | **0.208047** | 0.026559 |
| Repaired without centers | 0.732427 | 0.602182 | 0.207821 | 0.025596 |
| Repaired without forwards | 0.728466 | 0.605138 | 0.209136 | 0.024410 |
| Repaired without guards | 0.731769 | 0.603009 | 0.208094 | 0.021545 |

[Paired uncertainty](nba_rapm_review_20261009/paired_uncertainty.json), 1,200 shared-date bootstrap replicates, repaired minus no RAPM:

- AUC **+0.002600**, 95% interval **[-0.003729, +0.009027]**.
- Log loss **-0.003786**, interval **[-0.008568, +0.001067]**.
- Brier **-0.001352**, interval **[-0.003460, +0.000697]**.

**All three pooled intervals cross zero.** This is not proof of a highly effective full RAPM family. ECE is slightly worse, and the leakage-corrected arm is slightly worse than the invalid legacy comparator; correctness repairs are retained regardless of whether leaked point estimates look better.

Forwards add **+0.002761 AUC / -0.002528 loss / -0.001089 Brier** versus removing only F. Corresponding intervals exclude zero (AUC [0.000510, 0.005086]; loss [-0.004169, -0.001022]; Brier [-0.001785, -0.000406]). Point-estimate benefit survives both season slices. These are unadjusted exploratory multi-contrast intervals, **not** a pristine independent holdout or a family-wise significance guarantee. Center contribution worsens all three pooled point estimates, with intervals crossing zero. Guards are mixed and not distinguishable.

Recent 300 grading games: repaired vs no RAPM AUC **0.842130 vs 0.836157**, loss **0.500928 vs 0.510430**, Brier **0.162420 vs 0.166471**; date-block loss/Brier intervals exclude zero, AUC does not. ECE worsens **0.114896 vs 0.104169**. Postseason (169 games) AUC improves but loss/Brier/ECE worsen relative to no RAPM. Do not generalize the regular-season finding to playoffs.

**Admission decision:** retain correctness repairs and current membership, do not claim a statistically established all-family gain, do not remove C or tune lambda on this same replay. The lambda=16 provenance was descriptive player-ranking selection, not predictive admission. A sealed future-season comparison (including calibration and playoff strata), or possession/stint-based challenger with a timestamped roster source, is the appropriate next gate.

## NBA feature-coverage follow-up

[Full sliced coverage](nba_rapm_review_20261009/feature_coverage.csv) sweeps all 113 known columns across history/core/slate, every season, every team, regular/postseason, first ten and post-thirty season days. Historical feature checkpoint is rebuilt from current source code; older diagnostic parquet is input data, not treated as a current feature artifact.

| Slice | n | Center diff | Forward diff | Guard diff |
|---|---:|---:|---:|---:|
| Core | 2,629 | **94.10%** | 100% | 100% |
| Post-first-30-days, full history | 2,793 | **94.63%** | 100% | 100% |
| First ten days, full history | 219 | **73.06%** | 84.02% | 84.47% |
| Pending archived slate | 43 | 100% | 100% | 100% |

All 155 core center-diff nulls carry explicit pool refusal, none unexplained. Mask-supported absence is not proof that the model has adequate centers. Mature-season center holes remain real (150), not only cold-start behavior. Opening-night coverage rises because the roster no longer flips to post-game actual participants; this is causal carryover, not an imputation.

[Source reconciliation](nba_rapm_review_20261009/source_reconciliation.csv): all **6,922 settled team-games** have player lines, team facts and event rollups; player-summed points exactly equal independent schedule scores. All tested home-minus-away identities match; no infinities. [Measurement provenance](nba_rapm_review_20261009/measurement_provenance.csv) reports measured vs carried separately (no core carries in this snapshot). [Current/baseline monitor](nba_rapm_review_20261009/monitor_coverage.csv): all 142 rows OK; this late playoff window does **not** establish opening or team-level coverage.

Remaining source/coverage limitations:

- Slate coverage is **archive-only availability** in this replay: future October filings do not yet exist. The live resolver is present in production but was not replayed as an authenticated/game-day injury-PDF flow. 100% populated is not 100% availability-aware.
- Offseason/new-team assignments follow the last box appearance; no timestamped transaction/roster input can price an offseason move before first participation. Fixing same-day leakage makes this limit honest, not solved.
- Full-feed position metadata may be revised after the historical prediction origin; there is no versioned historical position-publication ledger.
- 707 games' event profiles come from the committed rollup archive rather than the local raw-PBP sweep. Aggregates are reconciled, not a fresh independent raw reconstruction of every game.
- The two NBA Cup finals still lack season-log player lines and are excluded. Live probes also show type-5 play-in games excluded by the current ESPN type mapping. That is a population-policy gap, not evidence of support merely because a score exists.

## Live sources and freshness

[Bounded fresh probes](nba_rapm_review_20261009/live_source_probes.json), fetched October 10 UTC / October 9 ET:

- ESPN opening night 2024-10-22 and Finals 2026-06-13 match cached admitted IDs, teams, scores and UTC tipoffs exactly.
- Fresh NBA LeagueGameLog 2025-01-15: **232/232 player lines match** points, minutes, FGA, FGM, turnovers and rebounds with zero differences.
- ESPN October 8/9: six/two current preseason events, **zero admitted training games** under the declared regular/postseason contract. SAC–LAL 110–114 agrees with the official [NBA game page](https://www.nba.com/game/sac-vs-lal-0012600036/box-score). The page's readable extraction exposes metadata, not player statistics; an independent CDN liveData box request returned **403**, explicitly not a successful box reconciliation.
- ESPN April 15 2025 returns two play-in events of season type 5, zero admitted rows. No fictional zero-score training rows were introduced.
- Definitions: [NBA official glossary](https://www.nba.com/stats/help/glossary), [RAPM reference](https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/) and [EPM methodology](https://dunksandthrees.com/about/epm) distinguish lineup-adjusted/possession-scaled metrics from this game-level proxy. Recognizable player rankings do not establish out-of-sample win prediction.

These are sampled network checks, **not** full-window freshness proof. There are no current regular-season NBA results to grade in the preseason window.

## Verification and evidence contract

- Full NBA backend suite: **593 passed**; sklearn/pandas deprecation warnings only. Final test output is in [backend tests](nba_rapm_review_20261009/backend_tests.txt).
- Actual A/B CLI exercised end-to-end with production cache ingestion and published designation archive; output and exit status are retained in [A/B CLI](nba_rapm_review_20261009/ab_cli.txt).
- Independent real final-model refit, 43 pending predictions, all finite/in (0,1), joblib reload bit-identical; matrix widths 31 linear / 73 tree. Fits are **ignored local audit models**, not published production forecasts. [Interface evidence](nba_rapm_review_20261009/interface_verification.json).
- Python compilation and whitespace checks, with results in [verification](nba_rapm_review_20261009/verification.json). No configured/installed static typechecker: compilation is not a static typecheck.
- Curated CSV/JSON/text/Markdown evidence only is added. All executable harnesses and bulky raw/feature/model checkpoints remain uncommitted. Existing modules/tests are repaired; **no new committed programs**.

The feature snapshot, fold geometry, six OOF arm CSVs, metrics and paired bootstrap are sufficient to inspect the numerical claims without trusting prose. This is historical replay using revised source archives, not a guarantee of future AUC/loss/Brier performance.
