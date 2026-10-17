# NFL key-number ±3 underfit — holdout-gate fix & registration ladder (2026-10-17)

Scope: the Spread view's pooled margin distribution (2,574 OOF games) still
showed the ±3 spikes underfitting the residuals (observed 7.2%/7.8% vs
modeled ~5.5%/6.7%). This review (a) found and fixed a real gate bug that
had made every "held-out" lattice verdict in-sample, (b) re-laddered the
spike/relief registrations at BOTH the final-fit and the causal-walk level
under the corrected gates, and (c) regenerated the NFL artifact. Standing
rules held throughout: no bar changes, no forced admissions, byte-identity
of the four `gaussian_family.py` copies preserved, pre-registration before
fitting.

## 1. Root cause of the misleading gate verdicts: slice-relative holdout indices

Every lattice gate in the shared family computed

```python
train = np.flatnonzero(inside[:len(outcomes) - n_test])
hold  = np.flatnonzero(inside[len(outcomes) - n_test:])   # BUG: slice-relative
...
d = (-2. * delta[indices[hold]] + 2. * (pmfs[hold] @ delta) + ...)
```

`flatnonzero` on the *tail slice* returns indices relative to that slice, so
`pmfs[hold]`, `delta[indices[hold]]` and `outcomes[hold]` indexed the **first**
~30% of rows — the very rows the amplitude was fitted on. Every gate verdict
(generic z, per-spike z, MLB one-run share z) was therefore **in-sample**,
inflated, and admitted corrections the true tail would not have confirmed.
Measured on the final-fit margin evidence: −3's spike gate read z 3.337
in-sample vs **4.930 on the true tail** at the shrink amplitude, and the
generic gate's verdicts shifted materially once the true tail was scored.

Fix (byte-identical, both gate sites in all four `gaussian_family.py`
copies, plus MLB's `fit_one_run_spread_rebalance` share gate in
`mlb-backend/backend/distributions.py`):

```python
hold = (np.flatnonzero(inside[len(outcomes) - n_test:])
        + (len(outcomes) - n_test))   # GLOBAL row indices
```

## 2. Consequence for the shipped 20261016/17 artifact

The shipped artifact was produced with the in-sample gate, so its ±3
corrections were admitted on inflated evidence. Re-running the full causal
walk with the corrected gates and the SAME registration (relief boost 3,
spike boost 1.0 — `LATTICE_RELIEF[("nfl","margin")]`,
`LATTICE_SPIKE_RELIEF_BOOST`) reproduces the delivery pipeline exactly (all
key-bin pooled residuals match the delivered CSV to 5 decimals; the ramp is
identical from row ~1000 on and bolder only at rows 300–900, where the
in-sample gate had admitted corrections the honest gate refuses):

| pooled (n=2574) | shipped (in-sample gate) | honest gate, same registration |
|---|---|---|
| −3 | +0.01914 | +0.02298 |
| +3 | +0.01265 | +0.01742 |
| −7 | +0.00725 | +0.00754 |
| +7 | +0.00399 | +0.00604 |
| 0 (tie) | −0.01102 | −0.01282 |

The honest chart looks WORSE at ±3 because the extra boldness the buggy gate
bought at small-n origins (rows 300–900: shipped model(−3) 0.038–0.054 vs
honest 0.027–0.046) was never holdout-validated: the ±3 excess over rows
300–900 is only ~1.2–2.3σ at n≈600, so the true tail cannot confirm it.
That is the price of gates that finally test what their docstrings claim.

## 3. Registration ladder — is there a deeper HONEST fix?

Two probes (read-only, evidence cached): `run_diagnostics/nfl_keynumber_align_probe2.py`
(final-fit ladder, 20 variants incl. gate decomposition) and
`run_diagnostics/nfl_keynumber_walk_ladder.py` (full causal walks, the
decisive delivered-pooled metric). Shipped knobs laddered: relief boost
3→4→5→6 (× quantile .75/.85), spike relief boost 1→1.5→2→3, prior frac
.10→.12→.15 (unshippable — byte-identity: all four sports share the file;
kept as an upper bound only), frac rule. Every walk variant:

| walk | pooled −3 | pooled +3 | pooled MAE | band-outside (−15..15) | spike-+3 admissions |
|---|---|---|---|---|---|
| S (shipped reg) | **0.02298** | **0.01742** | **0.000479** | **11** | **161** |
| relief5 | 0.0219* | 0.0185 | 0.000497 | 12 | 108 |
| sb2.0 | 0.0219 | 0.0185 | 0.000497 | 12 | 108 |
| relief4+sb1.5 | 0.0238 | 0.0195 | 0.000506 | 13 | 137 |
| relief4+sb2.0 | 0.0231 | 0.0212 | 0.000522 | 13 | 108 |

*even the best deeper candidate only moves −3 by −0.001 and pays it back at
+3, ±7, the tie, MAE and the band count.

Mechanism (matches the final-fit gate decomposition): the squared-loss gate
z ≈ (2·P·√n)/κ − amp·(√n·C)/κ **decreases in amplitude** (the amp² term), so
a deeper proposal loses exactly the small/mid-n admissions whose corrections
the pooled chart needs most (spike-+3 admissions 161→108 at sb 2.0). The
shipped registration (relief 3, sb 1.0) is the honest delivered-pooled
optimum among everything tested; at the final origin the honest gate
*already* admits both key numbers (−3 tail z 1.77, +3 z 0.07 — in band).

## 4. What shipped

* The holdout-index fix, all four backends (byte-identical) + MLB share gate.
* Test evidence retuned to honest time-inhomogeneous tails in
  `test_lattice_correction.py` (all four) — the old stationary generators
  were tuned to the in-sample gate; +7's tail verdict now lands inside
  [structural bar, generic bar) where the structural bar's jurisdiction the
  tests pin.
* NFL artifact regenerated (`nfl_run_engine_markets_20261017.csv/.meta.json`,
  monitor; `run_diagnostics/nfl_regen_20261017.log`): margin generic gate
  z 4.843 (bar 3.0), 173/201 origins significant, served probabilities now
  carry only tail-confirmed corrections. Rendered Spread view verified
  against the artifact: modeled −3 4.889% / +3 6.067% (observed 7.187% /
  7.809%), tail callouts P(≤−10) 0.204 / P(≥10) 0.264, no console errors.
* Suites: NFL 120, NBA 714, MLB 470, NHL 687 passed.

## 5. Residual limitation (deliberate)

The pooled ±3 gap (≈0.023/0.017) remains outside the iid 95% band and is
dominated by (a) the causal cold start — no correction is possible before
LATTICE_MIN_TEST×1/LATTICE_TEST_FRACTION = 334 prior rows — and (b) the
small-n ramp where the ±3 excess is 1–2σ and any honest gate must refuse.
The band is an iid band applied to a causal artifact; at the final origin
(where full evidence exists) both key numbers are in band. A ramp-aware
predictive band would describe this correctly and is the natural follow-up.
