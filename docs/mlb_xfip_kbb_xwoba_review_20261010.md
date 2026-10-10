# MLB xFIP / K-BB% / xwOBA feature review — 2026-10-10

**Subject:** the three pitch-estimator families the 2026-10-10 05:30 delivery
actually served — starting-pitcher and bullpen **xFIP**, bullpen **K-BB%**, and
the **xwOBA** family (starter, position-pool, pitch-category and team trailing
columns) — audited against the delivered frame `game_level_features.csv`
(commit `dbb6e5b5`, 7402 × 297) and its pre-Scenario-C predecessor
(commit `2cc8f88f`, 2026-10-09 17:15, 7402 × 297), plus the committed
`features_metadata_20261010.json`, `feature_coverage_20261010.csv`,
`feature_drift_20261010.csv` and `model_monitor_20261010.json`.

**Verdict:** the three families are structurally sound — this review found **no
defect in any feature value** and changed **no feature value**. Every check
below is reproducible from committed artifacts. The two remediations this
review drives are reporting-side and belong to the
[run-log review](mlb_log_review_20261010.md): the drift summary advertised
"0 warnings, 0 alerts" while all 109 features were INSUFFICIENT, and the
coverage summary's counts did not match the CSV it describes. One plausibility
finding (negative xFIP on 9 rows of the unserved plain trio) is recorded as an
owner decision, not patched here.

Guardrails honored: **no new programs committed** (all audit harnesses live in
scratch directories that are not part of this commit; the only source edits are
the log-wording fixes and one stale label comment, plus pins appended to the
*existing*
[test_log_review_20261008.py](../mlb-backend/backend/test_log_review_20261008.py));
**no model, blend, calibration, feature-set version or filename changed**.

## Evidence contract and interfaces

* **xFIP family.** `xFIP = (13·HR_est + 3·BB − 2·K)/IP + 3.10` with
  `HR_est = FB · lg_hrfb`, a point-in-time league HR/FB clamped to
  [0.05, 0.25] and a 0.11 cold-start fallback
  ([features.py:_xfip_sql](../mlb-backend/backend/features.py#L1131)). Every
  window, shrink constant and staleness gate is inherited from the ERA columns
  they replace; values are LAG-shifted, so a game never sees its own pitches.
  Columns: `sp_xfip_{home,away,diff}`, `sp_xfip_5g_*`, `sp_xfip_delta_*`,
  `bullpen_xfip_10g_{home,away}`, `bullpen_xfip_delta_*` (12 generated).
* **K-BB% family.** `K-BB% = (K − BB)/batters_faced`, opportunity-shrunk toward
  the point-in-time league rate with `k = 93.3 pitches (6.02 IP) = 20% of the
  mean reliever-season`, identical windows to the WHIP columns it replaces.
  Columns: `bullpen_kbb_10g_*`, `bullpen_kbb_3g_*`, `bullpen_kbb_delta_*`
  (8 generated).
* **xwOBA family.** `sp_xwoba` (last 6 appearances, LAG-shifted; the legacy
  `_30g` name), `sp_xwoba_vs_l/_vs_r`, the eight served position pools
  `pl_{c,fb,sb,ss,tb,rf,cf,lf}_xwoba_{home,away,diff}` (`pl_dh` generated but
  **not** served, by the 2026-10-03 plan), the team trailing `woba_30g_*`, the
  pitch-category inputs `sp_xwoba_cat_*`, `team_xwoba_cat_*`,
  `league_xwoba_cat_*` and the served matchup products
  `exp2_cat_xwoba_{fastball,breaking,offspeed}_{home,away,diff}` (64 generated).
* **Serving contract.** `active_moneyline_feature_cols()` = **109** columns;
  **47** of them are these three families (3 xFIP + 6 K-BB% + 38 xwOBA).
  **Zero** legacy ERA/WHIP columns are served. The four `sp_whip_*` columns the
  Scenario C comment calls out-of-scope are still generated and correctly
  unserved ([config.py](../mlb-backend/backend/config.py#L340)).
* **Rename contract.** Scenario C is a rename + value swap at the same column
  positions: exactly **20 columns** left (`sp_era*`, `bullpen_era*`,
  `bullpen_whip*`) and exactly **20** arrived (`sp_xfip*`, `bullpen_xfip*`,
  `bullpen_kbb*`); the frame kept **297 columns** on both sides, nothing else
  dropped or appeared.

## Measured verification

[verification.json](mlb_xfip_kbb_xwoba_review_20261010/verification.json),
observation over the committed delivery artifacts:

| Check | Result |
|---|---:|
| Serving width / family members served | **109** / **47** |
| Served columns missing from the frame | **0** |
| Served set delta vs the pre-C delivery | **9 out, 9 in** — exactly the documented Scenario C renames |
| Legacy ERA/WHIP columns served | **0** (`sp_whip_*` generated, unserved by design) |
| Frame columns dropped/added by Scenario C | **20 / 20**, 1:1 rename, 297 → 297 |
| Row-level coverage parity (rename pairs) | **20/20** columns with identical non-null counts, **0** null-pattern disagreements across **7402/7402** rows |
| Diff identities `diff == home − away` | **19** families, **137,025** pairs, **0** failures above 1e-6 (max residual **1.18e-9**, one row, CSV decimal round-trip) |
| Invalid / infinite values in the families | **0** |
| Coverage artifact rows (47 × 2 windows) | **94/94 OK** — no STRUCTURAL, no MISSING_COLUMN, `column_present` true throughout |
| Drift artifact rows for these families | **47/47 INSUFFICIENT** (n_current 14–15) — see the log review's Finding 1 |
| Metadata entries for served family members | **47/47** present in the frame, 109 entries total |
| Re-estimation vs the replaced columns | **20** pairs, levels r = 0.42–0.53 (xFIP vs ERA) and r = −0.54…−0.58 (K-BB% vs WHIP, correct sign), **0.0%** exact-equal rows on every level column |
| No fabricated history at the frame opening | 2024-03-20 (first date): `sp_xfip*`, `sp_xfip_5g*`, `sp_xwoba*`, `bullpen_kbb_10g*`, `woba_30g*`, `sp_k9*` all **0/1 non-null** |
| MLB backend suite | **390 passed, 1 skipped** ([test output](mlb_xfip_kbb_xwoba_review_20261010/backend_tests.txt)) |

Served-level ranges on the delivered frame
([family_ranges.csv](mlb_xfip_kbb_xwoba_review_20261010/family_ranges.csv)):

| Column | served | non-null | min | median | max |
|---|---|---:|---:|---:|---:|
| `sp_xfip_5g_home` | yes | 6926 (93.6%) | 1.14 | 3.89 | 8.03 |
| `sp_xfip_5g_away` | yes | 6916 (93.4%) | 0.56 | 3.88 | 8.70 |
| `sp_xfip_5g_diff` | yes | 6540 (88.4%) | −5.80 | 0.01 | 4.94 |
| `bullpen_xfip_10g_home` | composite input | 7350 (99.3%) | 1.91 | 3.87 | 5.85 |
| `bullpen_kbb_10g_home` | yes | 7350 (99.3%) | −0.002 | 0.142 | 0.311 |
| `bullpen_kbb_3g_home` | yes | 7350 (99.3%) | −0.095 | 0.143 | 0.351 |
| `sp_xwoba_home` | yes | 6926 (93.6%) | 0.052 | 0.317 | 0.716 |
| `woba_30g_home` | yes | 7388 (99.8%) | 0.088 | 0.301 | 0.468 |
| `pl_c_xwoba_home` | yes | 7401 (99.99%) | 0.242 | 0.310 | 0.390 |

Sign conventions hold: K-BB% and the `*_diff` columns are centered on zero
(negative values are the expected sign, not out-of-range), the exp2 matchup
products are league-deviation products in [−0.027, 0.015], and every xwOBA
level sits inside the legal [0, 2] wOBA scale (the one 1.92 outlier is a
single-PA offspeed cell, `sp_xwoba_cat_offspeed_home`, unserved).

## Findings

1. **No feature-value defect.** Coverage parity with the replaced family is
   exact at the row level (20/20 columns, 0 disagreements on 7402 rows), the
   diff identities are exact, and the served columns carry sane ranges and no
   invalid values. The families are genuinely re-estimated, not renamed
   copies: 0.0% of level rows are byte-equal to the ERA/WHIP values they
   replaced, with correlations (0.42–0.53 / −0.54…−0.58) that are exactly what
   two different estimators of the same talent should show.

2. **The only exact-equal rows are reconciled.** The two season-delta columns
   show 2.459% exact-equal rows against their replacements. All **180** of
   them are early-season rows (home-game index 1–9) where the trailing-10
   window and the season-to-date baseline are built from the same games, so
   their difference is exactly 0.0 — the **same 180 rows** appear in the
   pre-Scenario-C frame ([verification.json](mlb_xfip_kbb_xwoba_review_20261010/verification.json)
   → `exact_zero_delta_reconciliation`). Window geometry, not a copied value.

3. **Reporting defects (remediated in the log review).** The delivered drift
   line said "109 features, 0 warnings, 0 alerts" while **all 47 family rows**
   (all 109 rows, in fact) were INSUFFICIENT, and the monitor JSON shipped the
   same green `drift_summary`. The coverage line claimed "all 218 … OK (4
   STRUCTURAL)" against 214 OK + 4 STRUCTURAL. Both messages are fixed in
   [explainability.py](../mlb-backend/backend/explainability.py) and pinned by
   T15/T16.

4. **Plausibility finding, owner decision: negative xFIP.** The plain
   `sp_xfip_home/away` trio carries **9 rows below zero** (min −1.49); the ERA
   column it replaces could never be negative (min 0.0, max 45/67.5 — the
   replaced family had the larger magnitude tail). The served xFIP levels
   (`sp_xfip_5g_*`, `bullpen_xfip_10g_*`) are clean (0 negatives). The
   negatives do reach one served composite through
   `wind_advantage_flyball_factor = wind_direction_multiplier × sp_xfip_diff`,
   whose delivered range is [−5.22, 3.87] with **1** row beyond |5|. A floor
   would change served feature values and therefore model inputs, so it is
   recorded here and left to the owner rather than applied — the same
   "deliberately not changed" treatment the prior review gave the tail.

5. **Label hygiene (fixed).** The serving-universe comment still called the
   removed trio "the plain SP ERA trio (sp_era_5g_* stays)" while the list
   below it holds `sp_xfip_*` names — corrected in
   [training.py](../mlb-backend/backend/training.py#L440), the same stale-label
   class as commit `a504630f`.

## Verification and limitations

* Full MLB backend suite: **390 passed, 1 skipped** (386 before this review's
  four pins), Python compilation and `git diff --check` pass. No
  configured/installed mypy or pyright exists; compilation is not static
  typechecking.
* All numbers come from **committed** artifacts: two frame deliveries, the
  metadata/coverage/drift/monitor JSON-CSV set, and git history. The audit
  harnesses that produced them live in scratch directories and are not part of
  this commit (no-new-programs guardrail).
* **No live-source probe was run** (no network pulls). This is a delivery
  consistency review, not an ingestion review; the Statcast extract itself was
  last audited in the 2026-10-07/08 input-quality reviews.
* **No predictive claim.** No refit, OOF, blend or calibration bundle was
  regenerated; Scenario C was adopted per owner direction against the A/B
  harness's `KEEP A` verdict ([mlb_xfip_ab_20261009.md](mlb_xfip_ab_20261009.md)).
  The first delivery under C reports auc 0.573 / brier 0.2445 / logloss 0.6821
  against the pre-C run's 0.5721 / 0.2446 / 0.6821 — inside the 0.5721–0.5746
  band of the last six runs in `model_history.json`, i.e. no measurable gain
  and no measurable loss from one run.
* **Build identity with the 2026-10-09 A/B harness is not claimed.** The
  harness is untracked scratch (`mlb_xfip_ab_20261009.json` → `harness`), and
  production's coverage matches the replaced ERA/WHIP columns exactly where the
  scratch build did not. The committed A/B record remains the authority for
  what was tested; this review verifies what shipped.
* Drift for these families is **unmeasured** in this delivery (47/47
  INSUFFICIENT, window under the 30-row floor) — that is a monitor-window
  property of the season's end, now disclosed by the remediated line, not a
  statement about the features.
