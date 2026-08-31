# Moneyline Ensemble Model-Tuning Admission Policy (sports_prediction_model)

**Status:** adopted 2026-08-31 — governs all future member tuning for the MLB
moneyline ensemble (`mlb-backend/`). No production training behavior changes
beyond the already-decided `config.RF_PARAMS` adoption recorded below.
**Scope:** every member of the 5-member ensemble — `xgboost`, `lightgbm`,
`logistic`, `randomforest`, `mlp`.
**Enforcement:** the two gates below are codified in
`mlb-backend/backend/run_tuning_gate.py` (executable checklist). Any tuning
request must run that script (or its exact logic) and the evidence JSON it
produces is the audit trail.

---

## 1. Executive summary

Tuning a member (hyperparameters, not features — features have their own
admission gates) is admitted only if **both** gates pass:

1. **MEMBER GATE** — the candidate is a strictly better version of that
   member, measured on the identical folds/seeds against the current member.
2. **BLEND GATE** — the candidate does not measurably hurt the ensemble
   blend (production-correct, adaptive weights, pooled arbiter), and the
   member's calibration improves.

There are **no per-model exceptions**. A candidate that fails either gate is
REJECTED and the rejection is recorded. If a future candidate is ever adopted
despite a gate failure, this policy document must be **amended first** — no
silent overrides.

The RF tuning thread (2026-08-31) is the governing precedent: adopted under
this exact two-gate reading (§5).

---

## 2. The ensemble and its metrics

Production ensemble (walk-forward adaptive weights, v2026.08.30):

| member | adaptive weight |
|---|---|
| xgboost | 0.45 |
| lightgbm | 0.153 |
| logistic | 0.183 |
| randomforest | 0.137 |
| mlp | 0.077 |

Production blend metrics (v2026.08.30): **AUC 0.5708, Brier 0.2446,
logloss 0.6821, ECE 0.0047.**

Blend mechanics (context for the gates):

- Weights are **adaptive**: softmax over pooled OOF member scores
  (`ADAPTIVE_WEIGHT_METRIC="auc"` at `ADAPTIVE_WEIGHT_AUC_TEMPERATURE`), with
  floor 0.05 / cap 0.45 projection (`training.compute_adaptive_weights`),
  falling back to static `ENSEMBLE_WEIGHTS` priors only before the first OOF
  pass.
- Pooled OOF (~6.5k games across all walk-forward folds) is the **arbiter**
  over small sealed windows (~281 games). This is a standing repo convention:
  single-window gains that do not survive the pooled view have repeatedly
  proven to be slice artifacts (lineup-actuals, untapped columns, opponent-
  adjusted — all DON'T ADOPT after the pooled check).
- Production-correct means `run_margin_diff` is attached out-of-fold over the
  decided frame exactly as `walk_forward_evaluate` does. Window refits that
  skip the margin attach (NULL-filled) are **less production-faithful** and
  are treated as context only.

---

## 3. Gate 1 — MEMBER GATE

**Question:** is the candidate a strictly stronger version of the member?

**Protocol:** the candidate and the current member are trained on the
**identical walk-forward folds** (same geometry, same frame, same seed), using
the production feature path for that member (train-fold-median imputation +
team-ID categoricals for RF; scaling path for logistic/MLP, etc.). Pooled OOF
is one concatenated evaluation over all fold predictions — never a mean of
per-fold scores.

**Pass requires ALL SIX conditions** (3 metrics × 2 views):

| view | logloss | AUC | ECE |
|---|---|---|---|
| pooled OOF | candidate < current | candidate > current | candidate ≤ current |
| sealed holdout (≥21 days) | candidate < current | candidate > current | candidate ≤ current |

Any condition failing → **MEMBER GATE REJECTED** (the candidate is not a
strictly stronger member; stop here, record the evidence).

Note: the member gate is necessary but not sufficient. A stronger member can
still fail to help (or slightly hurt) the blend — that is exactly what the
blend gate checks next.

---

## 4. Gate 2 — BLEND GATE

**Question:** does the candidate hurt the ensemble, and does it improve
calibration?

**Protocol:** production-correct (OOF `run_margin_diff` attached), all 5
members trained per fold, **adaptive weights re-earned per variant**
(`training._LAST_ADAPTIVE_WEIGHTS` cleared), evaluated on pooled OOF **and**
multiple sealed windows (`--windows 3`, newest→oldest). The same fold loop is
shared by both variants so comparisons are apples-to-apples.

**ADOPT requires ALL of:**

1. **Pooled blend not measurably worse** (numeric thresholds):
   - `candidate_AUC  ≥ current_AUC − 0.001`
   - `candidate_logloss ≤ current_logloss + 0.001`
   - `candidate_ECE ≤ current_ECE` (ECE is a priority — strict, no tolerance)
2. **Member ECE improves:** candidate member's ECE < current member's ECE
   (pooled view). Calibration is a stated priority: a tuning that sharpens
   discrimination while degrading calibration is rejected even if AUC holds.
3. **Sealed windows are CONTEXT ONLY.** A win on the pooled view with a
   mixed/0-N window record is recorded as counter-evidence, not a rejection.
   Conversely, a sealed-window win with a pooled loss is not an adoption
   (pooled is the arbiter).

**Not measurably worse** is defined as the ±0.001 AUC / +0.001 logloss
bounds above — within the observed noise band on ~6.5k-game pools. A
candidate inside the band is "neutral"; the decision then rests on the member
gate (stronger member) + the member-ECE requirement.

---

## 5. Precedent — the RandomForest tuning (2026-08-31)

Candidate: Optuna winner from `tune_rf_optuna.py` (study `rf_moneyline`,
75 trials, pooled-OOF-logloss objective, margin-attached folds):

```python
RF_PARAMS = {
    "n_estimators": 800, "max_depth": 6, "min_samples_leaf": 17,
    "min_samples_split": 6, "max_features": "log2", "bootstrap": True,
    "random_state": RANDOM_SEED, "n_jobs": -1,
}
```

### Gate 1 — MEMBER GATE: **PASSED** (all six conditions)

| view | metric | current (300 trees) | candidate (800 trees) | Δ |
|---|---|---|---|---|
| pooled OOF | logloss | 0.68783 | **0.68655** | −0.0013 ✓ |
| pooled OOF | AUC | 0.5518 | **0.5539** | +0.0021 ✓ |
| pooled OOF | ECE | 0.0208 | **0.0122** | −0.0086 ✓ |
| sealed 21d (n=282) | logloss | 0.68150 | **0.68050** | −0.0010 ✓ |
| sealed 21d | AUC | 0.5770 | **0.5783** | +0.0013 ✓ |
| sealed 21d | ECE | 0.0593 | **0.0357** | −0.0236 ✓ |

(Pooled member metrics from `rf_blend_ablation_5a0173c6fc28.json`; sealed +
tuner-pooled-logloss from the tuner run, study `rf_moneyline`.)

### Gate 2 — BLEND GATE: **PASSED** (adopted under member-strength policy)

| criterion | current | candidate | verdict |
|---|---|---|---|
| pooled blend AUC (≥ −0.001) | 0.5656 | 0.5657 | +0.0001 ✓ |
| pooled blend logloss (≤ +0.001) | 0.6837 | 0.6836 | −0.0001 ✓ |
| pooled blend ECE (≤) | 0.0065 | 0.0042 | −0.0023 ✓ |
| member ECE improves (<) | 0.0208 | 0.0122 | ✓ |
| sealed windows (context) | — | **0/3** | counter-evidence, recorded |

The 0/3 sealed-window record is **weak counter-evidence**: windows are
~281–286 games (noise band ±0.001–0.01 AUC), and the window refits ran
without `run_margin_diff` attached (NULL-filled — less production-faithful).
Per §4.3, pooled is the arbiter; pooled is marginally **better** on all three
metrics. The candidate therefore satisfies "stronger member, blend not
measurably hurt" — **ADOPTED** into `config.RF_PARAMS` with this policy as the
authority.

Corroborating evidence: `run_rf_weight_ablation.py` separately confirmed RF's
adaptive ~16% weight is earned — dropping RF from the blend costs
**−0.008 pooled AUC** (0.5740 → 0.5659), so RF is a legitimately useful #2
member and its weight is not a sensitive knob.

**Trade-off accepted:** 800 trees makes the RF fold fit ~2.7× slower than 300
— a training-time cost, not a quality cost. Recorded in the `config.py`
provenance block.

---

## 6. Uniformity — no per-model exceptions

The two gates apply identically to **all five members**. Known tension points,
resolved by this policy:

- **XGBoost** already dominates the blend (45% cap). A tuning that improves
  XGBoost's member metrics is admitted only if the blend isn't measurably
  worse — the same standard as RF. The cap is a separate, already-gated
  mechanism (`run_weight_cap_ablation.py`: keep the cap).
- **MLP / logistic** — the diversity wildcards. Their adaptive weights are
  earned (or starved) by OOF performance; a tuning that fails the member gate
  is rejected regardless of "diversity value". Diversity is a property of the
  *blend*, enforced by the floor, not a license to admit weak members.
- **LightGBM** — same protocol; its tuner (`tune_lightgbm_optuna.py`) mirrors
  the RF tuner's production-correct construction.

A candidate that fails either gate is **REJECTED and recorded**, even if the
failure is "only" pooled-neutral + sealed-negative. Adopting over a gate
failure requires amending this document first.

---

## 7. Process — pre-registration → gates → evidence → config

1. **Pre-register** the candidate: member, proposed params, and the
   rationale (what the tuning targets: e.g., overfitting, calibration,
   speed) — recorded in the evidence JSON before running the gates.
2. **Gate 1 (member):** run the member's tuner/harness on identical
   folds/seeds vs the current member; produce pooled + sealed metrics for
   both.
3. **Gate 2 (blend):** run the production-correct blend harness
   (`run_rf_tuned_blend_ablation.py` pattern) with adaptive weights re-earned
   per variant and ≥3 sealed windows.
4. **Write the evidence JSON** to `mlb-backend/data_delivery/`
   (`<member>_tuning_gate_<YYYYMMDD>.json`, schema `tuning-gate-evidence/v1` —
   see `rf_tuning_gate_20260831.json` as the template).
5. **Update config only on ADOPT**, with a provenance block citing the policy,
   both gates' verdicts, and the evidence file. REJECTED candidates leave
   config untouched; the rejection lives in the evidence JSON.
6. **Pinned tuned params:** adopted params are pinned in `config.*_PARAMS`.
   No per-retrain re-tuning. Re-tuning happens only on a **deliberate,
   documented cadence** — e.g., at a season boundary, when the decided frame
   grows materially (>1 season added), or when a scheduled audit
   (`feature_audit.py` / PSI drift) flags the member. A re-tuning request
   restarts at step 1 under this policy.

---

## 8. Audit trail

- `mlb-backend/data_delivery/rf_tuning_gate_20260831.json` — consolidated RF
  evidence (member + blend gates, thresholds, verdict).
- `mlb-backend/data_delivery/rf_blend_ablation_5a0173c6fc28.json` — blend
  gate raw results (CURRENT vs TUNED, per-member + blend, 3 windows).
- `mlb-backend/data_delivery/rf_weight_ablation_20260831.json` — RF blend
  weight ablation (adaptive 0.16 earned; drop-RF costs −0.008 pooled AUC).
- `mlb-backend/backend/rf_study.db` — Optuna study `rf_moneyline` (75 trials).
- `mlb-backend/backend/run_tuning_gate.py` — executable checklist for this
  policy.
- `mlb-backend/backend/config.py` — `RF_PARAMS` provenance block (adopted
  params + both gates' verdicts).
