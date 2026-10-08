# MLB run-log & dashboard review — 2026-10-08

**Subject:** the run trained 2026-10-08 05:44Z (artifacts commit `4f99e907`,
run log `cd07f014`, `mlb_pipeline_run_log.txt` = 461 lines, `MLB_FULL_REPULL`
window 2024-01-01 → 2026-10-08).
**Verdict:** delivery is clean this run — `git ls-tree origin/main` carries 19
`*20261008*` files, so the 2026-10-07 T1 split fix holds. Two findings: one
STRUCTURAL data gap (the whole 2026-10-07 slate is missing from the frame,
unlogged) and one dashboard-honesty defect (identical raw/calibrated twins
presented as "after calibration" with no explanation). Both remediated in this
commit with tests. Also: the walk-forward AUC plateau quantified below is
real, not a measurement artifact.

## What was validated

### Dashboard ↔ production run — the "identical AUC/logloss" question

Every headline on the Calibration page equals the committed artifact
(`calibration_20261008.json`): AUC **0.5719** / Brier **0.2446** / log-loss
**0.6823** / ECE **0.0089**, `n` = the grading pool via `n_eval` (the T6
population-label fix holds). The twins read `0.5719 → 0.5719`,
`0.2446 → 0.2446`, `0.6823 → 0.6823` — and those are REAL, not a rendering
bug: the prequential gate deployed an **identity** calibrator for this run
(`calibrator_gated_out: true`, `calibration.method: "identity"`,
`params: null`), so raw and calibrated are literally the same probabilities.
The same held for the 10-07 run; fitted runs (10-03/05/06) still carry
non-identical twins. The defect was the PRESENTATION: KPI cards captioned
the identical pairs "· after calibration" with no banner anywhere explaining
that no map ran — remediation **T2** below.

### Ingestion reconciliation — frame vs. official schedule

Within its window the frame matches MLB's official schedule exactly: every
official final with a `game_date ≤ 2026-10-06` is present, cancelled games
correctly absent. One structural gap sits at the tail:

| Official final (10-07) | game_pk | In frame? |
|---|---|---|
| CLE @ CWS | 849833 | absent |
| LAD @ ATL | 849822 | absent |
| TB @ NYY | 849838 | absent |
| MIL @ SD | 849827 | absent |

`pitches.parquet` horizon = **2026-10-06**, so `predictions_history_20261008`
tops out at 10-06: the 10-07 slate never resolved into predictions history /
Today's Record, and **nothing in the 461-line run log said so**. Root cause:
Savant posting lag. The run's last chunk (`2026-08-18 → 2026-10-08`) returned
168,080 pitches cleanly — zero retries, zero empties — so
`_abort_on_exhausted_core_season_empty_chunk` (which fires only on *empty*
core-season chunks) could not catch it; at pull time (≈05:00Z, ~1h after the
last final) the 10-07 slate simply was not indexed yet. The IL ledger even
printed `lag -2d vs data horizon 2026-10-06` without connecting it to missing
finals.

Confirmed transient: re-querying Savant now returns **1,128 pitches across
exactly those 4 game_pks** for 2026-10-07 — the data exists, so the next
run's forward top-up + `REFRESH_TAIL_DAYS` refresh recovers the slate
automatically. That makes the correct remediation a loud WARN, not an abort
— remediation **T1** below.

## Why model performance did not improve

Walk-forward AUC by run (`calibration_*.json`, metrics.calibrated):

| run | 0928 | 0929 | 0930 | 1001 | 1003 | 1004 | 1005 | 1006 | 1007 | 1008 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.5850 | 0.5612 | 0.5611 | 0.5617 | 0.5741 | 0.5734 | 0.5726 | 0.5746 | 0.5721 | 0.5719 |

Paired bootstrap on shared games (2,000 resamples, seed 7):

| pair | n shared | Δ AUC | 95% CI | significant? |
|---|---|---|---|---|
| 0928 → 1008 | 7,019 | −0.0140 | [−0.0250, −0.0028] | yes (confounded, see below) |
| 1001 → 1003 | 7,037 | **+0.0134** | [+0.0055, +0.0210] | yes |
| 1003 → 1008 | 7,168 | −0.0019 | [−0.0071, +0.0035] | **no** |
| 1006 → 1008 | 7,174 | −0.0028 | [−0.0074, +0.0015] | **no** |

Reading:

1. **The 0928 peak was protocol, not the model.** The 0928 run trained before
   the causal-rounds leak fix landed; the first honest run (0929) dropped to
   0.5612 with no feature changes between them. Comparing today against 0928
   compares against a leak.
2. **The only real gain in the window is the 10-03 feature rebalance**
   (`91f564ef`): +0.0134, significant. Everything after 10-03 is a plateau:
   five runs inside 0.5719–0.5746, pairwise deltas all inside ±0.003 and CIs
   crossing zero.
3. **The intervening work was designed to be AUC-neutral.** Since 10-03 the
   commits were integrity/delivery work — published-blend provenance,
   causal-evaluation parity, input repairs, population labels, delivery-root
   fixes. None targeted discrimination, and none moved it; that is the
   expected result, not a failure.
4. **Power check:** the paired CI half-width on ~7.2k shared games is ≈0.005,
   so a real improvement must be ≳0.007 to clear — daily feature tweaks
   cannot show up even when they work.
5. **The ceiling is behavioral, not stale data.** Probabilities are heavily
   shrunk (std 0.0673, 51.0% of games inside [0.45, 0.55]), Brier skill vs.
   a climatology constant is only 1.7% (0.2446 vs 0.2489), and recent form
   (last 60 decided games, AUC 0.575) sits at the pooled level (0.572) — the
   model is not hiding recent signal.

Conclusion: with the honest protocol, the model has been flat since 10-03 at
a level the current feature set supports. Moving the number requires new
per-game signal (and paired-bootstrap evaluation on a fixed window), not
re-tuning; further near-term gains are more reliably available from
calibration and delivery honesty, where changes are measurable today.

## Remediations

- **T1 — missing-finals guard.** `ingestion.warn_missing_finals(pitches_path,
  end)` runs in Phase 1 immediately after the pull lands
  (`master_pipeline.py`, between the `✅ Raw pitches` line and Phase 1.5). It
  reads the frame's `(game_date, game_pk)` columns, fetches the official
  schedule for **horizon+1 → end only**, and warns (tag `missing-finals
  guard`) listing every official final absent from the frame by `game_pk` —
  pk-level so a suspended/resumed listing already framed under its original
  date never false-positives. Warn-only by design (posting lag is transient
  and the next run recovers); the call is wrapped so a guard defect cannot
  kill the run. The 10-07 defect replayed end-to-end in
  `test_log_review_20261008.py`.
- **T2 — gated-calibrator note.** `frontend/model_calibration.py` now
  resolves `cal_sec`/`_gated` once above the KPI cards. On identity runs the
  captions flip from "· after calibration" to "· calibrator off (identity)"
  and a **Calibration Gate** banner ("Gated out · identity") renders under
  the cards explaining that probabilities ship raw and the twins are the
  same values by design (mirroring Model Monitor's existing "no calibration
  map deployed" wording). The Platt banner and the gate banner are `if/elif`
  — a fitted-map run can never show both.

## Verification

- `python -m pytest mlb-backend/backend/ -q` — **355 passed** (includes the
  new `test_log_review_20261008.py`, 7 tests: flagging, tail-only window,
  silence when current / when already framed, degraded-frame safety, Phase-1
  wiring pins).
- `python check_production_graph.py` — **OK, 27 modules** (the new test
  imports only production-reachable modules, no allowlist entry needed).
- `python -m pytest frontend/test_calibration_gate_note.py -q` — **2 passed**
  (identity renders the gate note and never the Platt banner; fitted-map run
  renders the Platt banner and never the gate note).
- `python -m pytest frontend/test_calibration_smoke.py frontend/test_monitor_smoke.py frontend/test_nhl_pooled_oof_parity.py -q` — **7 passed**.
- Script smokes, each exit 0: `test_board_render_smoke`,
  `test_calibration_smoke` (sport=nfl full leg + sport=mlb clean on the REAL
  identity-gated artifact + sport=nhl parity), `test_mlb_board_render_smoke`,
  `test_monitor_smoke`, `test_nhl_board_render_smoke`,
  `test_power_rankings_smoke`.
- `python -m pytest frontend/test_nba_frontend.py
  frontend/test_nba_markets_page.py frontend/test_nhl_frontend.py -q` —
  **81 passed**.

## Known limitations

- The T1 guard is warn-only: this run's missing 10-07 slate is not repaired
  retroactively — the next pull's forward top-up recovers it (verified: Savant
  now serves all 4 game_pks).
- `python -m pytest frontend/` (whole directory in one session) crashes in
  pytest's capture layer (`ValueError: I/O operation on closed file`). This
  is **pre-existing** — reproduced with the new test file ignored — so the
  frontend suite is run per file, which is how every smoke above was run.
- The production pipeline was not re-run (Kaggle/Colab environment); all
  conclusions come from the committed artifacts, the committed run log, and
  live schedule/Savant queries.
