# MLB scenario test: ERA (A) vs xFIP (B) — moneyline serving set

Frame: `load_games_for_date(2026-10-08)` + `apply_adopted_subset()`, 7402 decided games, serving width 109, folds 84 (min_train_days=30), seal = last 4 folds.
seeds 42/7/2026 on XGBoost+LightGBM (elasticnet fixed at 42), UNROUNDED OOF logloss for all gate math (tune doctrine).

Scenario A = shipped frame (incumbent). Scenario B = same frame with all 12 ERA-family columns swapped to point-in-time xFIP **plus the 4 served composites recomputed** (`pitcher_regression_indicator_{home,away,diff}` = fbvelo × sp_era_5g_*, `wind_advantage_flyball_factor` = dome-wind × sp_era_diff), so no ERA path survives in B.

## Builder validity (post-fix)

- same-day attach leak closed: 7387 targets have a same-day pitcher appearance; 7206 would have read different values under exact-match (mean |shift| 0.271 xFIP).
- group-bounded shifts: no cross-pitcher/season/team window inheritance.

| column | era nulls→xfip covered | corr |
|---|---|---|
| sp_era_home | 6806/6806 | 0.4882 |
| sp_era_away | 6800/6800 | 0.3781 |
| sp_era_5g_home | 6926/6926 | 0.4109 |
| sp_era_5g_away | 6916/6916 | 0.3786 |
| sp_era_delta_home | 6806/6806 | 0.4255 |
| sp_era_delta_away | 6800/6800 | 0.2862 |
| sp_era_diff | 6435/6435 | 0.4341 |
| sp_era_5g_diff | 6540/6540 | 0.3804 |
| bullpen_era_10g_home | 7251/7350 | 0.5278 |
| bullpen_era_10g_away | 7245/7343 | 0.5196 |
| bullpen_era_delta_home | 7223/7320 | 0.5005 |
| bullpen_era_delta_away | 7217/7313 | 0.4976 |

## Pooled results (grading rows, unrounded logloss)

| seed | blend A | blend B | Δ | auc A | auc B | Δauc |
|---|---|---|---|---|---|---|
| 42 | 0.681995 | 0.681895 | -0.000100 | 0.5723 | 0.5733 | +0.0010 |
| 7 | 0.682191 | 0.682139 | -0.000053 | 0.5713 | 0.5722 | +0.0009 |
| 2026 | 0.681510 | 0.681716 | +0.000206 | 0.5742 | 0.5740 | -0.0002 |

## Per-member deltas (B − A, negative = xFIP better)

| member | s42 | s7 | s2026 | mean | floor | gate |
|---|---|---|---|---|---|---|
| xgboost | -0.000207 | -0.000769 | +0.000334 | -0.000214 | 0.001764 | fail |
| lightgbm | -0.000248 | +0.000061 | +0.000226 | +0.000013 | 0.001071 | fail |
| elasticnet | -0.000081 | -0.000081 | -0.000081 | -0.000081 | 0.000000 | PASS |

## Sealed tail (last 4 folds, never touched selection)

| seed | Δ seal blend logloss |
|---|---|
| 42 | -0.000096 |
| 7 | -0.000272 |
| 2026 | -0.000093 |

## Paired per-fold (blend)

- seed 42: folds B-better 38/74, mean Δ -0.000127, median Δ -0.000112
- seed 7: folds B-better 37/74, mean Δ -0.000065, median Δ -0.000001
- seed 2026: folds B-better 39/74, mean Δ +0.000212, median Δ -0.000266

## Gates

- pool gate (3/3 seeds better AND mean < −floor 0.000681): FAIL — deltas [-9.998e-05, -5.273e-05, 0.00020645]
- seal gate (mean ≤ 0): PASS — deltas [-9.641e-05, -0.00027221, -9.345e-05]
- **verdict: KEEP ERA (A)**

Harness parity vs tune floor (seed 42): {"tune_floor_s42_auc": 0.5728, "mine_s42_auc": 0.5723, "tune_floor_s42_logloss": 0.6819, "mine_s42_logloss": 0.682, "tune_n_eval": null, "mine_n_eval": null}


## Scenario C (B + bullpen WHIP -> K-BB%), pooled OOF, seed 42

Scenario C = the full xFIP feature set (B) PLUS the 6 served bullpen_whip_10g/3g
columns swapped to point-in-time K-BB% (same windows/shrink shape), with the 3
served bullpen_meltdown_risk_* composites recomputed (pitches x kbb_10g) so no
WHIP path remains. kbb corr vs whip -0.53..-0.57, coverage 7251/7350.

| arm (seed 42) | blend auc | brier | logloss (raw) | weights xgb/lgb/enet |
|---|---|---|---|---|
| A (ERA)       | 0.5723 | 0.2445 | 0.6820 (0.68199529) | 0.2098/0.4415/0.3487 |
| B (xFIP)      | 0.5733 | 0.2445 | 0.6819 (0.68189531) | 0.2131/0.4557/0.3312 |
| C (B + kbb%)  | 0.5719 | 0.2447 | 0.6823 (0.68234265) | 0.3006/0.3622/0.3372 |

C - A raw blend logloss: +0.000347; C - B: +0.000447
(single seed by request; half the 0.000681 blend noise floor - directionally
negative, not gate-decidable from one seed.)
