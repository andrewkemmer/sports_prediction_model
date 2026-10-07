# MLB run-log review — 2026-10-07

**Subject:** `mlb-backend/data_delivery/mlb_pipeline_run_log.txt` as delivered on
remote (411 lines, run pushed at `a5586ad0`, artifacts commit `2de24c26`).
**Verdict:** the model numbers reconcile; the DELIVERY did not. Five defects
found, root-caused against the emitters, and remediated in this commit
(tests: `mlb-backend/backend/test_log_review_20261007.py`).

## What the log claimed vs. what remote had

| Log line | Remote reality |
|---|---|
| `📁 Artifacts: 15 files` (`todays_games_20261007.csv`, `model_monitor_20261007.json`, …) | `git ls-tree origin/main` carries **zero** `*20261007*` files — the board on GitHub is still `todays_games_20261006.csv` |
| `📋 Staging 3 files` → `✅ Pushed and remotely verified 3 files` | The 3 staged paths were the tee log, `game_level_features.csv`, `pl_slate_20261006.parquet` — 2 of them byte-identical to what was already remote, so commit `2de24c26` contains only the log |
| `slate pl_* pools: 4 team-games … → pl_slate_20261006.parquet` on a run whose slate is `Upcoming slate built: 4 games for 2026-10-07` | The export dated the slate 10-06 (from the clone's stale `lineups.parquet`: game_pks 849819/849826 dated 2026-10-06), while `_load_slate_pl('2026-10-07')` hunted `pl_slate_20261007.parquet` |
| `slate pl_* sources (home/away): {'carry/carry': 4}` | Consequence: the resolved pools the export just computed were never served — all four sides priced on the marked carry |
| `Walk-forward season split: 7024 OOF rows (6879 grading / 107 postseason / 137 provisional); blocks regular n=6879 …` | The three counts **overlap** (99 postseason rows sit inside provisional folds) and sum to 7,123 on a 7,024-row frame; the block labeled `regular` is built from the grading mask |

Root cause of the delivery/slate defects is one split: the run executed in
**two trees**. The tee (`Run log tee: /content/sports_prediction_model/…`),
features' lineup root (`player_positions.parquet` source, `pl_slate_*`) and
Phase 5's staging scan (`Path.cwd()/data_delivery`) all pointed at the fresh
`/content` clone, while every Step-5 artifact, `pbp_defense_*` and the
ensemble were written to `/kaggle/working/sports_prediction_model/…`.
`config` is imported at `master_pipeline:48` — BEFORE the clone's `backend/`
lands at `sys.path[0]` — so `DATA_DELIVERY_DIR` resolved against the
kernel's checkout and no later import re-resolved it (the Phase-0 module
purge list does not include `config`).

Not defects (checked and clean): headline metrics vs. monitor lineage, the
published-blend / PROVISIONAL / stale-`MLB_END_DATE` lines added by the
2026-10-05/06 reviews, the degenerate-Platt counters (256 of 6064, 3-line
cap), the final run-log delivery (its push confirmation lands after the file
copy by design), and the calibrator gate (log-loss 0.6820 → 0.6823 with ECE
0.0077 → 0.0033 is *mixed* evidence, which the documented 2026-08-27 policy
keeps by design).

## Remediations

- **T1 — config-root reconciliation.** `config.ensure_config_root` loads the
  running clone's `config.py` **by path** into `sys.modules['config']` and
  purges every module imported from the foreign backend dir (they hold stale
  `from config import DATA_DELIVERY_DIR` bindings). `master_pipeline` Phase 0
  calls it right after the clone lands at `sys.path[0]`, best-effort and
  never fatal; after the tee installs, an observability line announces the
  resolved delivery root — or `⚠️ delivery root SPLIT` — into the pushed log.
- **T2 — Phase 5 scans BOTH roots.** The pre-run snapshot and the staging
  scan now iterate the cwd root *and* config's `DATA_DELIVERY_DIR`
  (per-root mtime gates, `_stage` dedupes), so artifacts written to either
  tree are staged; a split is printed with both paths.
- **T3 — reported-artifact coverage gate.** `github_sync.
  missing_reported_artifacts` cross-checks `summary['artifacts']` (the
  `📁 Artifacts: N files` list) against the staging list; any miss raises
  inside Phase 5's try, which converts it into `RuntimeError: MLB artifact
  delivery did not complete; refusing to report a successful pipeline run`
  — a green log can no longer claim delivery GitHub does not have.
- **T4 — slate pl_* lookup diagnoses its misses.** `_load_slate_pl` searches
  the config root, `Path.cwd()/data_delivery` and the features-tree root
  (explicit `base` stays single-root); serving from a non-primary root warns
  that the roots are split, and an absent exact-date file with a
  **nearby-dated sibling (±3 days)** warns with the sibling name and day
  offset instead of silently returning `{}` (offseason/empty exports remain
  silent).
- **T5 — season-split line reconciles.** The emitter now prints
  `7024 OOF rows = 6879 grading + 145 non-grading (137 provisional incl. 99
  postseason, 8 postseason outside provisional folds); blocks OVERLAP —
  grading n=6879, postseason n=107, provisional n=137`, so every number in
  the line adds up and the block label matches the mask it is built from.

## Verification

- `python -m pytest mlb-backend/backend/test_log_review_20261007.py` — 16
  passed (replay of the delivered run's staging would be refused; foreign
  config swap/purge; both-root snapshot + scan pins; slate lookup: exact /
  alternate-root / mis-dated / silent-offseason / explicit-base; split-line
  pins and the 6879 + 145 = 7024 identity).
- Full MLB backend suite: **309 passed, 3 failed** — the 3 failures
  (`test_run_log_tee` ×1, `test_log_review_20261005::_mem_mb`,
  `test_log_review_20261006` CR-collapse) reproduce identically in a clean
  worktree at HEAD: pre-existing Windows-environment failures, unrelated to
  this change.

## Known limitations

- The 2026-10-07 artifacts were produced on the Kaggle run's filesystem and
  were never pushed; they cannot be retro-delivered from this checkout.
  Remote keeps serving the 10-06 board until the next successful run — and
  from then on a run that cannot deliver its reported artifacts FAILS
  instead of reporting `Status: ok`.
- The reconciliation only runs where current pipeline code runs (a notebook
  executing a stale copy of `master_pipeline.py` predates the fix — the new
  `⚠️ … predates ensure_config_root` line and the split observability line
  make that visible in the delivered log either way).
