# MLB Overfit-Gap Remediation — 2026-10-10

## Trigger

In-training (resubstitution) metrics were far more favorable than
out-of-sample, worst for the tree members. Quantified on the delivered
`game_level_features.csv` frame (7,402 decided games, 109 adopted
features) with a 37-fold expanding-window walk-forward replay using the
production hyperparameters, 7-day cadence, zero embargo
(`analysis/gap_quant.py`, results in `gap_quant_results.json`):

| Member | Resub AUC | OOF AUC | ΔAUC | Resub LL | OOF LL | ΔLL |
|---|---|---|---|---|---|---|
| LightGBM | 0.7014 | 0.5818 | +0.120 | 0.6473 | 0.6801 | −0.033 |
| XGBoost (50r) | 0.6502 | 0.5731 | +0.077 | 0.6635 | 0.6827 | −0.019 |
| ElasticNet | 0.5967 | 0.5743 | +0.022 | 0.6759 | 0.6825 | −0.007 |

The linear member's small gap confirms the tree gap is capacity-driven
memorization, not a single leaky feature. Headline metrics remain
honest (OOF-only, prequential calibration); the gap is a robustness and
hyperparameter-selection hazard, not a published-number defect.

## Arms measured (`arms.json`, identical fold geometry)

LightGBM capacity (P0-1):

| Arm | Train AUC | Train LL | OOF AUC | OOF LL | OOF ECE | ΔAUC gap |
|---|---|---|---|---|---|---|
| L0 base d7/lv40/mcs64/364r | 0.7014 | 0.6473 | 0.5818 | 0.6801 | 0.0167 | 0.120 |
| L1 d4/lv15/mcs100 (ADOPTED) | 0.6740 | 0.6560 | 0.5806 | 0.6805 | 0.0138 | 0.093 |
| L2 d3/lv7/mcs100 | 0.6594 | 0.6607 | 0.5801 | 0.6805 | 0.0067 | 0.079 |
| L3 d5/lv31/mcs128/regλ1/lr.03/500r | 0.7373 | 0.6230 | 0.5843 | 0.6817 | 0.0359 | 0.153 |

LightGBM causal round budget (P0-3, XGB-parity probe/transfer):
**measured and rejected.**

| Arm | Train AUC | Train LL | OOF AUC | OOF LL | ΔAUC gap |
|---|---|---|---|---|---|
| B0 base capacity, floor 150 | 0.6660 | 0.6661 | 0.5774 | 0.6821 | 0.089 |
| B1 base capacity, floor 364 (= L0, no-op) | 0.7014 | 0.6473 | 0.5818 | 0.6801 | 0.120 |
| B2 d4/lv15/mcs100, floor 150 | 0.6470 | 0.6707 | 0.5750 | 0.6828 | 0.072 |

Floor 150 underfits on every metric; floor 364 is a byte-identical
no-op (probe medians never reach it). This member ships its fixed
364-tree budget by design.

XGBoost budget/capacity opportunities:

| Arm | Train AUC | Train LL | OOF AUC | OOF LL | ΔAUC vs base |
|---|---|---|---|---|---|
| X0 base 50r/lr.055 | 0.6502 | 0.6635 | 0.5731 | 0.6827 | — |
| X1 lr.03/250r fixed | 0.7003 | 0.6402 | 0.5797 | 0.6812 | +0.0065 |
| X2 softened d3/mcw6/γ1/lr.03 floor150 | 0.7457 | 0.6262 | 0.5809 | 0.6804 | +0.0078 |
| X3 base capacity, floor 50→150 (ADOPTED) | 0.7042 | 0.6379 | 0.5825 | 0.6815 | +0.0094 |

X3 is the best single-member AUC of all 11 arms — the member was
budget-starved at 50 rounds (all 83 production probe transfers measure
≤ 36, so the static floor is the binding budget every fold).

## Decisions applied

1. `config.XGBOOST_FOLD0_ROUNDS` / `XGBOOST_REFIT_ROUNDS`: 50 → 150.
   Causal-transfer honesty contract unchanged; contract tests repinned.
2. `config.LIGHTGBM_PARAMS`: max_depth 7→4, num_leaves 40→15,
   min_child_samples 64→100 (P0-1). OOF log-loss neutral (+0.0004),
   gap 0.120→0.093, ECE 0.0167→0.0138.
3. P0-3 (LGBM causal round budget) not adopted — negative result recorded.
4. Calibration unchanged (favored-space Platt + floor), per owner
   direction; the prequential gate already ships identity when the map
   harms honest metrics.

## Not adopted / follow-ups

- L2 (d3/lv7) if a more aggressive gap cut is wanted later (gap 0.079,
  ECE 0.0067, AUC −0.0017 vs L0).
- X2 (softened capacity) if member log-loss is prioritized over AUC.
- Blend check: best replay combos were L3+X3 (AUC 0.5860) and
  L0+X2 (LL 0.6794) vs equal-weight incumbent 0.5826/0.6800; the
  production rolling logloss weighting should re-earn weights naturally
  from the improved members.

## Verification

`pytest test_run_engine_contract.py test_run_engine_pit.py
test_fold_season_split.py test_calibration_gate.py` — 78 passed.
