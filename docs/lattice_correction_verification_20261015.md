# Integer-lattice (sawtooth) correction — verification report (2026-10-15)

## What changed and why

Any smooth continuous density integrated over integer bins acts as a low-pass
filter: it underestimates peaks and overestimates troughs of the discrete
score lattice, leaving a sawtooth ("comb") residual
`r(k) = P_obs(k) − P_model(k)` against observed OOF finals. Neither the
split-skew family nor the monotone calibration map can express that
alternation (both are smooth in the CDF). `gaussian_family.lattice_correct`
(now mirrored in all four backends) applies a fitted discrete correction
AFTER calibration and BEFORE Monte Carlo sampling, on the two-sided combined
distribution, for BOTH targets — totals (home + away) and margin (home − away).

Totals and margin are fit INDEPENDENTLY (separate residuals, separate
coefficients, keyed by sport+target) because their lattice structures differ:

- the margin lattice carries the no-tie rule (`P(M=0)=0` for NHL/MLB/NBA,
  redistributed to ±1; NFL keeps valid tie mass) and the NHL totals lattice
  carries the OT/SO parity flip;
- the generic residual captures whatever lattice structure remains (e.g.
  MLB's empirical walk-off-by-one tilt at M=+1 — deliberately NOT hard-coded).

This generalizes the old NHL-only, totals-only `parity_correct()`, which is
kept byte-stable and is now the STRUCTURAL half of the NHL totals lattice
correction (same alpha, same CDF-domain math) with the generic fitted
residual on top.

## Causal discipline

- Every fit uses strictly-prior OOF rows (the causal walk's BASE-model
  evidence: post-calibration, post-structural, pre-lattice PMFs of
  already-priced blocks, each priced with its own origin's maps).
- Admission is behind a uniform held-out gate: a time-ordered 30% tail of the
  SAME prior evidence must show a significant paired squared-probability-loss
  improvement (`LATTICE_GATE_Z = 3.0`) AND the pooled residual MAE must
  improve. Otherwise the correction ships diagnostics but no delta and prices
  as a byte-identical no-op.
- The no-tie ±1 split weight (NHL OT/SO, MLB walk-off extras) is a fitted,
  gated one-goal home-share estimate (`fit_ot_home_share`), never a constant:
  the shift it implies is scaled by each row's tie mass and scored on the
  held-out tail with the same gate. Unconfirmed shifts keep the exact
  equal-split no-op.
- Corrections are zero-sum, clamped (`±0.03`), rounded (`LATTICE_ROUND = 6`)
  so stored bundles reprice byte-identically, and forbidden bins (margin 0
  for no-tie sports) are re-zeroed.

## Middleware-correctness notes found during verification

1. **Ridge scaling (2026-10-15).** The pooled-bin regression was normalized
   by `len(train)` (row count) instead of living in bin space, shrinking the
   Gram to ~1e-5 so the `0.02` ridge vetoed every coefficient (MLB margin's
   +0.033/−0.024 ±1 residual fit to a 1.5e-05-L1 no-op). Removed the
   normalization; the ridge is now genuine shrinkage. The held-out gate
   caught the bug's consequence (no spurious admissions), but the fix
   recovered the real MLB totals admission (z = 5.33).
2. **Tie-mass scaling (2026-10-15).** The split-weight gate scored the raw
   `(q−½)` shift against post-split PMFs, letting the `‖δ‖²` term veto every
   share (z = −4.5 against a stable residual). The scored shift is now
   `(q−½)·m0` per row with `m0 = (pmf(+1)+pmf(−1))/2` (the pmf contrast is
   exact — tie mass cancels; the level uses the family's own smoothness
   assumption). MLB's share now scores z = 2.98 — still a principled reject
   (see below), but honest instead of broken.
3. **Delivery writers pinned to LF** (`lineterminator="\n"` in the three
   shared `write_markets_csv` writers and MLB `persist_markets`); local
   pandas 3.0.1 otherwise rewrites every line ending (CRLF) and the diff
   becomes unreadable. A full-support-PMF column expansion drafted mid-task
   was reverted — no schema change ships with this layer.

## Before/after per sport and per target

Fits below are the FINAL serving fits (all settled rows); OOF columns are
repriced through the causal walk (per-origin maps). "Changed columns" counts
delivered distribution columns whose values move; non-distribution columns
are byte-identical in all four sports.

### NHL — honest no-op (gate rejects both targets), 0/160 columns change

| target | n_fit/n_test | gate z | pooled resid MAE before → after | verdict |
|---|---|---|---|---|
| totals (α=0.363 structural parity rides on) | 1894/811 | −0.30 | 0.000789 → 0.000792 | rejected |
| margin | 1894/811 | −0.62 | 0.000793 → 0.000871 | rejected |

Baseline reprice reproduces the delivered file exactly (0/160 columns
differ). Largest pooled residuals (totals 5: +0.0118, margin +3: +0.0364)
are left untouched — negative gate z means the candidates would have hurt.

### NBA — honest no-op, 0/873 columns change

| target | n_fit/n_test | gate z | pooled resid MAE before → after | verdict |
|---|---|---|---|---|
| totals | 1680/720 | 0.47 | 0.000243 → 0.000249 | rejected |
| margin | 1680/720 | 2.19 | 0.000146 → 0.000153 | rejected |

0/56 origins admit. (An earlier revision of the fitter admitted a spurious
1e-5-scale NBA margin delta through the ridge-scaling bug's distorted
geometry; the corrected fitter + gate rejects it.)

### NFL — honest no-op, 0/467 columns change

| target | n_fit/n_test | gate z | pooled resid MAE before → after | verdict |
|---|---|---|---|---|
| totals | 1802/771 | 1.28 | 0.000966 → 0.000968 | rejected |
| margin | 1802/771 | 2.27 | 0.000857 → 0.000875 | rejected |

0/200 origins admit. Metrics byte-identical. (Same spurious-admission
correction as NBA: the first pass admitted tiny deltas at 3 per-origin
margin fits; the corrected gate rejects them.)

### MLB — totals ADMITTED, margin structural-only; 156/187 columns change

| target | n_fit/n_test | gate z | pooled resid MAE before → after | verdict |
|---|---|---|---|---|
| totals | 4926/2110 | **5.33** | **0.001133 → 0.000509 (−55%)** | **admitted** (δ L1 0.138, max 0.0138) |
| margin (generic δ) | 4926/2110 | 0.67 | 0.000392 → 0.000422 | rejected |
| margin split share | — | 2.98 | direction helps (−0.00094 mean) | rejected (< 3.0), see note |

Pooled-PMF worst bins, totals: 5: +0.0209 → +0.0131; 3: +0.0200 → +0.0156;
7: +0.0162 → +0.0066; 8: −0.0207 → −0.0118. Margin ±1 via per-origin share
admissions only: +1: +0.0364 → +0.0316; −1: −0.0230 → −0.0186.

OOF metrics (same per-origin CDFs the walk issued):

| card | CRPS | Brier | logloss |
|---|---|---|---|
| totals baseline → lattice | 2.461408 → **2.461298** | 0.249364 → 0.249320 | 0.691875 → 0.691787 |
| run_line baseline → lattice | 2.465785 → 2.465440 | 0.247989 → 0.247702 | 0.689116 → 0.688541 |

55/84 origins admit totals deltas (z builds 2.8 → 5.2 across the walk —
stable, strengthening evidence, not a late break).

**MLB walk-off note (near-miss, documented, not forced).** The M=+1
walk-off tilt (+0.036/−0.024, stable across halves: +0.037/+0.029) is real,
stable, and directionally confirmed — but the low-order global Fourier basis
cannot localize to ±1 without collateral MAE damage (after > before), and
the tie-mass-scaled split share scores z = 2.98 against the 3.0 bar. The
gate does its job: per-origin share admissions flicker honestly (37/84
origins admit 0.6 while the evidence supports it; the final serving fit
rejects, so slate rows price equal-split). Prime candidate for revisit with
more 2026 data; NOT admitted by lowering the bar.

## Target isolation

Disabling ONE target's correction leaves the OTHER target's priced columns
untouched (`run_diagnostics/verify_target_isolation.py`, per sport, against
the regenerated metas): on the final MLB serving maps, disabling totals
moves 108 totals-side columns, disabling margin moves 0, overlap empty —
PASS (nhl/nba/nfl pass vacuously: final specs carry no delta/share and the
write logs record zero per-origin admissions). The MLB margin-column
movement in the lattice WALK comes exclusively from per-origin margin share
admissions — the margin structural term admitted at 37/84 origins by its
own gate, not totals leakage (plus the inert-spec byte-identity test in
`test_lattice_correction`).

## No unintended columns

- Same column sets in all four rewritten CSVs (MLB 187, NHL 197, NBA, NFL —
  verified header-equal).
- 0 non-distribution cell diffs in every sport (cell-level audit old-vs-new,
  including `mu_*`, scores, kind/ids, offered lines).
- Meta JSONs: one ADDED key (`lattice` disclosure: magnitude + gate verdict,
  no delta vectors except the fitted per-origin/Final specs needed to
  reprice); MLB `variance` gains explicit `total_parity_alpha: 0.0`
  (values otherwise identical); recomputed metric blocks reflect the new
  prices. Monitor JSONs: `fit` gains `lattice`; winner/market-metric blocks
  recomputed through the backends' own writers.
- Frontend: fit panel discloses per-target magnitude + gate verdict
  (`frontend/markets.py`, `market_diagnostics.py`); chart-span extensions
  read only shipped columns and gap honestly on older artifacts.

## Tests

- New `test_lattice_correction.py` in all four backends (12/11/11/11 passed):
  config map, byte-identical passthrough, CDF guards, structural rules,
  causal fit + gate (synthetic comb admitted, noise rejected), evidence
  alignment, walk wiring, pre-lattice bundle repricing.
- Full suites (all green): NHL 654 passed; NBA 664 passed, 1 skipped
  (1 structural-contract test updated for the new `lattice` maps key);
  NFL 88 passed, 1 skipped; MLB 429 passed, 2 skipped; frontend 125 passed.
  One NHL writer test was replaced along with the reverted full-support-PMF
  schema expansion it covered (the replacement pins the grid contract).
- `test_distribution_objective_parity.py` untouched and passing (NHL
  `parity_correct` byte-stable — the structural half is unchanged math).
