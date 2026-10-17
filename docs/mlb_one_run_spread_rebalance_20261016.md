# MLB margin ±1 "spread spike" rebalance — 2026-10-16

> One-run-final split defect in the MLB margin lattice, the pre-registered
> mechanism that closes it, the donor-capacity guard that keeps the funding
> honest, and the measured verdict. Probe + log:
> `run_diagnostics/mlb_pm1_spread_rebalance_probe.py` /
> `run_diagnostics/mlb_pm1_spread_rebalance.log`. Full-walk verification:
> `run_diagnostics/mlb_rebalance_regen.log`.

## A. The defect (20261016 shipped artifact)

The margin lattice ships structural-only: the gated walk-off home-win share
(q̂ = 0.600, z 3.064) re-splits each row's **tie mass**, and the generic
residual fit is refused (gate z 2.22 < 3.0 — see the verification doc §A3).
The pooled model therefore sits, against the settled OOF actuals:

| bin | actual | model | residual | reading |
|---|---|---|---|---|
| +1 | 0.1739 | 0.1433 | **+0.0307** | the +1 spike UNDERFITS the residual |
| -1 | 0.1089 | 0.1255 | **-0.0166** | the -1 spike OVERFITS it |
| -2 | 0.0881 | 0.0784 | +0.0097 | model is **short** at -2 |
| +3 | 0.0663 | 0.0755 | -0.0092 | model is long at +3 |

The defect is not the tie split: it is the model's own **one-run-final
split**, P(M=+1)/(P(M=+1)+P(M=-1)) = 0.5155 against the gated share
q̂ ≈ 0.60 (observed split 0.615). The smooth split-skew family plus the
monotone CRPS map cannot express a one-run-win excess, and the walk-off
share's tie-mass scaling ((q-½)·m₀ with m₀ ≈ 0.07 of model mass) can only
ever move ≈ 0.007 at ±1 — an order of magnitude below the residual.

## B. Pre-registered mechanism (`fit_one_run_spread_rebalance`, MLB-only)

Declared **before fitting** in the module header of
`mlb-backend/backend/distributions.py` (the shared
`family.LATTICE_PREREGISTERED` table is byte-identical infrastructure
mirrored in all four backends, so an MLB-only mechanism registers next to
its own code):

- **Recipient** M=+1 (direction: model underpredicts, one-sided).
- **Amount** `amt = (q̂ − ½) · (p(+1) + p(−1)) / 2` — the admitted walk-off
  share applied to the model's own one-run-final mass. **Every number is
  dynamic**: q̂ comes from the gated pooled estimator, the masses from the
  fitted evidence; there is no constant amplitude anywhere.
- **Donors** (−1, −2, +3) fund in proportion to their own
  **actuals-supported excess**: each donor's draw is capped at the shrunk
  train residual it can give before its own sign flips. A donor the
  actuals show **short funds exact 0** — measured: M=−2 model 0.0784 vs
  0.0880 actual, capacity 0, drawn 0. The unfunded remainder (0 on the
  current evidence) falls to the mass-proportional sink. Each donor's
  residual/capacity/drawn triple rides the report.
- **Gates**: unchanged generic bar (LATTICE_GATE_Z = 3.0, two-sided paired
  held-out loss) + the per-bin collateral guard (no bin's |residual| may
  worsen beyond its own 1σ binomial floor) + zero-sum by construction +
  forbidden bin 0 untouched. A refused rebalance returns no delta and
  prices byte-identically to before; no structural-bar fallback is claimed.

## C. Verdict on the 20261016 evidence (final serving fit, n=7039)

```
amount 0.013684   one_run_mass 0.273674   model_split 0.5155
donors: -1 capacity 0.021935 drawn 0.009368 | -2 capacity 0.0 drawn 0.0
        +3 capacity 0.010104 drawn 0.004315   (funded_by_sink 0.0)
held-out z 3.268 >= 3.0 generic bar   collateral_max_excess 0.0   ADMITTED
```

Pooled residuals (BASE model → after the rebalance):

| bin | before | after |
|---|---|---|
| +1 | +0.0330 | **+0.0194** |
| -1 | -0.0238 | **-0.0144** |
| +3 | -0.0113 | **-0.0070** |
| -2 | +0.0102 | +0.0102 (untouched — it is short, it cannot fund) |

pooled resid MAE 0.000392 → 0.000307 (−22%), zero-sum exact, no bin's
collateral guard trips.

### Full-walk verification (`regenerate_lattice_artifacts.py mlb`, 84 origins)

- **origins admitting a margin delta: 9/84 (was 0/84)** — every one by the
  rebalance's own held-out gate; the generic fit still refuses (z 2.23).
- Pooled per-origin residuals at the two complaint bins: M=+1 +0.0378 →
  +0.0295; M=−1 −0.0225 → −0.0158 (pre-change shipped state on this
  artifact: +0.0302 / −0.0169).
- Run-line cards improve: CRPS 2.466001 → 2.465714, Brier 0.247886 →
  0.247625, logloss 0.688910 → 0.688386. Run-line cover ECE moves 0.0065 →
  0.0107, exactly as it already did with the admitted totals delta alone
  (20261015 regate: 0.0073 → 0.0117) — a pre-existing property of the
  totals correction, not of this mechanism.
- Target isolation passes (margin-only columns 29, totals-only 97, overlap
  empty); zero non-distribution cell diffs in both audits; the baseline walk
  (all lattice fits neutralised, this mechanism included via its share
  precondition) still reproduces the pre-lattice behaviour.

## D. Why the amount is sized off the share, not off the full residual

The full shrunk train residual at M=+1 (+0.0314) is exactly what pins the
generic fit's delta at the 0.03 fix cap — and it is **refused** by the
unchanged gate. The amplitude ladder (donor-funded constant vector, held-out
rows):

| amt | held-out z | verdict (generic 3.0 / structural 2.5) |
|---|---|---|
| 0.0080 | 3.70 | admit |
| **0.0137 (share-sized)** | **3.27** | **admit at the generic bar** |
| 0.0199 | 2.79 | structural bar only |
| 0.0250 | 2.40 | refuse |
| 0.0314 (full residual) | 1.99 | refuse |

The proposal's mean loss improvement is linear in the amplitude while its
held-out cost is quadratic, so the gate ceiling sits near amt ≈ 0.017. The
share-sized amount is not tuned to that ladder — it is the pre-registered
structural quantity (the same (q̂−½)·mass the admitted walk-off share itself
uses) — and it clears the GENERIC bar with margin, so the mechanism claims
no structural concession.

## E. Standing rules honoured

- No gate change: the generic bar, the paired held-out loss and the
  collateral guard are the ones every other target uses.
- No forced admission: the ladder in §D is disclosed rather than exploited;
  a deeper proposal is refused, not shipped.
- Pre-registration: direction (recipient up, donors down), donor set, bar
  and the amount formula are declared in code before any fit runs.
- Dynamic sizing: transfer amount and donor capacities are recomputed from
  the evidence at every origin; nothing is hard-coded.
- Byte-safety: refused rebalance ⇒ no delta ⇒ identical prices; admitted
  deltas ride the existing `lattice_correct` path (caps, forbidden bins,
  renormalisation, exact-tail guard) and store fitted-lattice bounds for
  support-drift realignment.
- No shared-file drift: `gaussian_family.py` is untouched (still
  byte-identical across the four backends), and the point-in-time core of
  `distributions.py` (the five locked functions) is unmodified.

## F. Tests

`mlb-backend/backend/test_one_run_spread_rebalance.py` (8 tests): amount and
funding read from the evidence; short donor funds exact 0 and discloses its
capacity; pure noise refused with a verdict; no-share and sub-half share are
no-ops; evidence guards; the delta prices a valid CDF with the forbidden bin
empty; the walk stores the disclosure and ships the delta; composition with
an admitted generic delta. Full MLB backend suite: 461 passed, 4 skipped.
