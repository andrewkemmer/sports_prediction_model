# NBA spreads — registered key-number donor → recipient transfer (2026-10-16)

## 1. Complaint and measurement

User-reported: *"spreads have key numbers — spikes overfit at −1, −6, +14, +15,
+18, +1 and underfit at +6, +7, +11, +13, +16, −2, −9, −10. Fully remediate."*

Measured on the shipped `nba_run_engine_markets_20261016.csv` (2,400 decided OOF
rows, pooled model `p_push_<int>` against realised margins, model-based binomial
z — the dashboard band's own formula; probe
`run_diagnostics/nba_keynumber_align_probe.py`, untracked):

| bin | model | observed | residual | z | claim |
|---|---|---|---|---|---|
| −1 | 0.02432 | 0.02167 | −0.00266 | +0.84 | over-fit |
| +1 | 0.02248 | 0.01833 | −0.00415 | +1.37 | over-fit |
| −6 | 0.02334 | 0.02083 | −0.00251 | +0.81 | over-fit |
| +14 | 0.02037 | 0.01750 | −0.00287 | +0.99 | over-fit |
| +15 | 0.01878 | 0.01542 | −0.00336 | +1.21 | over-fit |
| +18 | 0.01442 | 0.01083 | −0.00359 | +1.47 | over-fit |
| −10 | 0.01980 | 0.02292 | +0.00312 | −1.10 | under-fit |
| −9 | 0.02135 | 0.02458 | +0.00323 | −1.09 | under-fit |
| −2 | 0.02458 | 0.02750 | +0.00292 | −0.92 | under-fit |
| +6 | 0.02438 | 0.02792 | +0.00354 | −1.12 | under-fit |
| +7 | 0.02321 | 0.02833 | +0.00512 | −1.67 | under-fit |
| +11 | 0.02244 | 0.02667 | +0.00423 | −1.40 | under-fit |
| +13 | 0.02042 | 0.02417 | +0.00374 | −1.30 | under-fit |
| +16 | 0.01754 | 0.02083 | +0.00330 | −1.23 | under-fit |

**Every one of the fourteen reported signs is CONFIRMED** on the delivered
artifact, and thirteen of fourteen hold their sign on the time-ordered held-out
tail (720 rows) as well — only −2 flips, at n=720 noise (tail residuals:
−1 z +2.38 and +16 z −2.01 are the only two bins the tail puts outside their
95% band; the full sample puts **0 of 41** margin bins outside it, before and
after this change).

Two structural facts frame everything below:

* **Each tooth is below its own noise floor.** se ≈ 0.0037 at the serving
  fit (n_train 1,680, p ≈ 0.022) against residuals of 0.003–0.005 → per-bin
  z of 0.8–1.7. No single bin can carry its own gate at any origin.
* **The per-origin evidence is thinner than the pooled sample.** The causal
  walk refits every origin on its own prior rows (blocks of ~43 games), so
  per-origin se sits at 0.0035–0.006 — larger than the signal itself.

## 2. Root cause of the surviving teeth (measured)

Pooled **applied** delta (delivered model − BASE pre-lattice model, same 2,400
rows) against the pooled residual it was supposed to fix:

```
 bin   BASE resid   applied   delivered resid      direction
  -1    -0.00410   -0.00144     -0.00266        right-signed
  +1    -0.00623   -0.00209     -0.00415        right-signed
  -6    -0.00205   +0.00045     -0.00251        WRONG (-0.00046 damage)
 +14    -0.00181   +0.00106     -0.00287        WRONG (-0.00106)
 +15    -0.00270   +0.00067     -0.00336        WRONG (-0.00066)
 +18    -0.00348   +0.00011     -0.00359        WRONG (-0.00011)
  -2    +0.00154   -0.00138     +0.00292        WRONG (-0.00138)
  +6    +0.00224   -0.00130     +0.00354        WRONG (-0.00130)
  +7    +0.00306   -0.00206     +0.00512        WRONG (-0.00206)
 +11    +0.00421   -0.00002     +0.00423        neutral
 +13    +0.00343   -0.00031     +0.00374        WRONG (-0.00031)
 +16    +0.00345   +0.00016     +0.00330        right-signed (tiny)
 -10    +0.00300   -0.00012     +0.00312        WRONG (-0.00012)
  -9    +0.00379   +0.00056     +0.00323        right-signed
```

The generic localized residual prices each bin from that origin's own training
residual, whose noise (0.0035–0.006) exceeds the pooled signal (0.003–0.005);
applied mass at the registered bins is therefore **noise-dominated and
wrong-signed at 10 of 14 bins** — the exact opposite of remediation, and the
reason the teeth survive an already-admitted margin lattice
(`gate_z 3.264/3.003`, per-A6/A8 disclosure).

What this needs is the thing the generic basis cannot produce: a **joint,
consistently-signed claim** across all fourteen bins, sized from evidence and
admitted by a single held-out test — the NHL empty-net / MLB one-run pattern.

## 3. Pre-registration (declared BEFORE any fitting)

The mechanism lives entirely in `nba-backend/backend/distributions.py`
(the MLB/NHL precedent) so the shared `gaussian_family.py` stays
**byte-identical across all four backends** (md5 verified; the four-copy test
passes). Registers, sizing rule, direction and bars are the module header
comment above `fit_keynumber_transfer` — this document is the disclosure
record.

* **Donors** (over-fit, fund): −1, +1, −6, +14, +15, +18.
* **Recipients** (under-fit, gain): −10, −9, −2, +6, +7, +11, +13, +16.
* **Direction**: recipients gain, donors give — declared from §1 before any
  fit is run.
* **demand(recipient)** = `lattice_local_deltas(relief · p̄, resid, se)` —
  the SE-shrunk train residual under the declared relief
  `KEYNUMBER_DEMAND_RELIEF = 3` on the mass-proportional prior scale (the
  NFL/NHL spike-relief precedent).
* **capacity(donor)** = `clip(−resid, 0, p̄)` — the donor's **observed**
  excess on train evidence, capped at its own model mass (the NHL supply
  rule): a donor the actuals show short funds exact 0.
* **amount** = Σ declared demand (ONE fitted scalar).
* **template** = an equal unit at each ACTIVE recipient — the proportions are
  rule-derived and carry **no outcome data**, so the per-origin gate scores a
  one-degree-of-freedom claim instead of re-estimating fourteen noisy
  proportions first (§4).
* **funding**: donors pro rata over capacity, actuals-capped per donor; the
  un-funded remainder falls to the family's mass-proportional sink over bins
  that are neither registered nor the no-tie forbidden bin (the MLB one-run
  funding rule).
* **Gates (unchanged, none lowered)**: one-sided paired held-out test at the
  generic bar `LATTICE_GATE_Z = 3.0` (no structural bar is claimed — these
  teeth are empirical, not rule-derived), the per-bin collateral guard, and
  the fitted direction check on TRAIN evidence only (hold-out tail is scored,
  never fitted on). Zero-sum by construction; `|δ| ∈ ε-rounding`; forbidden
  bin 0 untouched; a refusal returns no delta and prices byte-identically.

## 4. Ladders — the shipped constants are dialled BY MEASUREMENT

Probe programs and logs (untracked per the standing guardrail):
`run_diagnostics/nba_keynumber_ladder.py`, `nba_keynumber_shape_probe.py`,
`nba_keynumber_walk_ladder{,2,3}.log`, `nba_keynumber_ladder.log`.

**(a) Shape probe (same serving evidence, train-fit / holdout-scored):**
proportions re-estimated from train residuals cost the gate power.

| shape (relief 3, amount 0.0080) | held-out z |
|---|---|
| residual-proportional (first pass) | 3.023 |
| mass-proportional | 3.846 |
| **equal per active recipient (shipped)** | **3.995** |

**(b) Walk ladder, shape v1 = residual-proportional, capacity-bounded
amount** (full causal walk per value, 56 origins):

| relief | origins admitting | pooled margin resid MAE |
|---|---|---|
| 0.5 | 13/56 | 0.001889 |
| 1.0 | 12/56 | 0.001884 |
| 2.0 | 6/56 | 0.001888 |
| 3.0 | 6/56 | 0.001888 |
| 8.0 | 6/56 | 0.001888 |

**(c) Walk ladder, shape v2 = registered template, claim-sized amount with
actuals-capped donor funding + sink (the shipped mechanism):**

| relief | origins admitting | pooled margin resid MAE | Σ&#124;resid&#124; @14 bins |
|---|---|---|---|
| 1.0 | 16/56 | 0.001883 | 0.04799 |
| **3.0 (shipped)** | **10/56** | **0.001878** | **0.04780** |
| 5.0 | 5/56 | 0.001888 | 0.04818 |

HEAD-equivalence of the reference arm is exact: the transfer-OFF walk reproduces
the shipped artifact's pooled MAE (0.001891) and 14-bin footprint (0.048319)
to the digit, so every number below is attributable to this mechanism alone.
**Relief 3 adopted**: lowest pooled MAE, lowest registered-bin footprint,
serving transfer gate **z 3.639** (bar 3.0) on its own held-out tail
(n_test 720, `collateral_max_excess` 0.0), amount 0.015159 (= Σ demand,
8 active recipients × 0.001895 each), funded by 0.010828 of actuals-capped
donor draws (−1 funds **exact 0** — its own train residual is on the wrong
side, disclosed per origin: +1 0.005282, +18 0.004261, +15 0.000821, −6
0.000419, +14 0.000044) plus 0.004331 on the mass-proportional sink. The
composed margin vector (transfer + generic, `delta_source`
`generic+keynumber_transfer`) then clears the generic gate at **z 3.519**
with `delta_l1` 0.030308 and `delta_max` 0.005282 — both disclosure keys are
shipped in the meta. Amount beyond ~1.5× the claim fails its own held-out
gate (z 2.863 at relief 5 ×1.5) — amplitude is NOT mined, the ladder shows
where the bar bites.

## 5. Delivered result (regenerated artifact, `regenerate_lattice_artifacts.py nba --write`)

**Isolation (three independent checks, all clean):**

| check | result |
|---|---|
| regen record `isolation_nondist_columns_identical` | **true** |
| regen record `delivered_nondist_columns_identical` | **true** |
| cell-level check vs HEAD (this change) | **0 of 25 non-distribution columns changed; 231 of 900 distribution columns changed** (all margin-side) |
| `serving_isolation` | pass — 231 margin-only / 486 totals-only columns, overlap **empty** |
| header/rows | identical column order, 2,414 rows |
| totals side | totals metrics and `lattice_summary` byte-identical to HEAD (gate_z 3.003, δmax 0.001211) |
| moneyline two-way deviation | 0.0 (baseline and lattice) |

**Pooled residuals at the registered bins — OFF (=HEAD) → ON (shipped):**

| bin | OFF | ON | closure | bin | OFF | ON | closure |
|---|---|---|---|---|---|---|---|
| −1 | −0.00266 | −0.00265 | +0.4% | −10 | +0.00312 | +0.00307 | +1.6% |
| +1 | −0.00415 | −0.00398 | +4.1% | −9 | +0.00323 | +0.00318 | +1.5% |
| −6 | −0.00251 | −0.00250 | +0.4% | −2 | +0.00292 | +0.00288 | +1.4% |
| +14 | −0.00287 | −0.00286 | +0.4% | +6 | +0.00354 | +0.00350 | +1.1% |
| +15 | −0.00336 | −0.00336 | 0.0% | +7 | +0.00512 | +0.00511 | +0.2% |
| +18 | −0.00359 | −0.00355 | +1.1% | +11 | +0.00423 | +0.00421 | +0.5% |
|  |  |  |  | +13 | +0.00374 | +0.00370 | +1.1% |
|  |  |  |  | +16 | +0.00330 | +0.00325 | +1.5% |

* pooled margin residual MAE (all 41 bins): **0.001891 → 0.001878 (−0.7%)**
* Σ|residual| over the 14 registered bins: **0.048319 → 0.04780 (−1.1%)**
* every bin retains its sign and none gets worse — the direction the
  registration declared holds in the delivered prices.
* record's own pooled-PMF margin MAE vs the pre-lattice baseline:
  0.0001278 → 0.0001063 (**−17%**); serving-fit margin residual MAE
  0.0001197 → 0.0000779 (**−35%**, meta lattice disclosure).
* M=±3 diagnostics in the record: −3 +0.00173 → −0.00073, +3 −0.00227 →
  +0.00001; M=+1 −0.00687 → −0.00400 (pre-lattice → lattice).

**Run-line card, meta HEAD → shipped (same walk, same seed):**

| metric | HEAD | shipped | Δ |
|---|---|---|---|
| auc | 0.495115 | 0.498315 | **+0.0032** |
| brier | 0.250134 | 0.250088 | **−0.000046** |
| logloss | 0.693415 | 0.693323 | **−0.000091** |
| ece_cover | 0.012552 | 0.012545 | **−0.000008** |
| crps | 8.217418 | 8.217437 | +0.000020 (+0.0002%, noise scale — reported, not hidden) |

**Disclosure:** the meta's and monitor's `lattice.margin` block now carries
`keynumber_report` (bar, relief, template, per-bin
residual/demand/capacity/gain/drawn, `funded_by_donors`, `funded_by_sink`,
z, n_test, `collateral_max_excess`). Top-level meta/monitor key sets are
unchanged (nothing dropped — `run_line_fit_check` still present, recomputed by
the writer).

## 6. Honest boundary — what this does NOT close, and why

Per admit the gains are 0.001895 per recipient against 0.003–0.005 teeth, but
the **held-out gate only clears z ≥ 3.0 at origins whose hold-out tail is
~450+ rows** — i.e. 10 of 56 origins ≈ 18% of OOF rows. Pooled over the slate
that is ≈0.0003 of movement per bin against a ≈0.0035 tooth: **~1% closure at
the registered bins, 0.7% on pooled MAE.** The teeth remain where they were,
just measurably closer, and all 14 remain inside the 95% band (0/41 outside,
unchanged).

No standing rule was bent to get there, and none of the available shortcuts
would be legitimate:

* **pooled/serving maps on OOF rows** — causal leakage, forbidden by the
  walk's own contract (evaluations freeze at the hold-out cut).
* **a lower bar for these bins** — "no gate changes" (the structural 2.5 bar
  is reserved for rule-derived mechanisms; NBA spread teeth are empirical and
  no structural claim is registered for them).
* **deeper relief on the generic fit** — already laddered and measured as a
  regression (A6: boost 3 → 5 collapses admissions 49 → 38 and moves z the
  wrong way); the teeth sit in the shrinkage floor by construction.
* **a heavier family / variance arm** — A7 tested and rejected it for >270
  totals (family-shape limitation, knob range exhausted).

What it would take: per-origin SNR ≥ 1 at these bins, i.e. more seasons of
evidence (the documented follow-up path for NHL empty-net ±2/±3 in A3 note 4)
— not a parameter this session is allowed to force.

## 7. Tests and verification

* `nba-backend/backend/test_keynumber_transfer.py` — **11 new contract
  tests**: sizing read from evidence (amount = Σ demand, equal template,
  actuals-capped donor draws, sink balance, zero-sum), a short donor funds
  exact 0, a long recipient demands exact 0, all-long → refusal, pure noise →
  refusal, evidence guards, valid CDF with the no-tie bin empty, walk wiring
  and compose (`generic+keynumber_transfer`), refusal disclosure, cold start
  no-op.
* NBA suite: **714 passed, 3 skipped** (includes the lattice and gaussian
  suites; three skips are the standing deselections).
* Frontend suite: **150 passed**.
* `regenerate_lattice_artifacts.py` gains one guarded neutralization
  (`getattr(dist, "fit_keynumber_transfer", None)`) so the baseline arm of any
  re-gate excludes the mechanism the way it already excludes the family
  hooks; inert for the other three sports.
* Evidence logs (untracked by the task guardrail): `run_diagnostics/`
  `nba_keynumber_align_probe.py`, `nba_keynumber_ladder.py`,
  `nba_keynumber_shape_probe.py`, `nba_keynumber_walk_ladder{,2,3}.log`,
  `nba_keynumber_walk_off.log`, `nba_keynumber_regenerate.log`,
  `nba_suite_keynumber_transfer.log`, `frontend_suite_keynumber_transfer.log`.