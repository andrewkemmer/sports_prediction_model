# NFL coverage audit — 2026-10-08 snapshot

## Verdict

**Coverage is not a clean pass.** The source game population is complete for the configured settled window, and the repaired feature engine passes the measured future/pending isolation checks. However, team aliases lose usable source rows; availability defaults confuse missing evidence with healthy players; FTN candidates describe both teams' combined plays; the board writes incorrect UTC times; and the shipped artifacts predate the current source/feature contract.

This is an audit, not a remediation or model-quality claim. No production predictions, model bundle, feature-selection state, or tuning parameters were replaced. The local acquisition populated regenerable source caches. Committed files are reports and bounded diagnostic evidence only.

## Scope and reproduction context

- Code: `bce1c4c35c9fdf64b119558cc1f17cd5c8dc028b`, feature contract `nfl-prod-v9.9-pending-state`.
- Date window: 2016-01-01 through 2026-10-22 inclusive; run/as-of date 2026-10-08. The local diagnostic directory happens to end in `20261009`; its label is **not** the snapshot date.
- Production schedule loader, eligibility filter, source loaders, `build_game_features`, `build_slate_features`, model-family projections, fold builder, and board writer were exercised directly. The pipeline entry point was not run: it can train, replace delivery data, sync, and push.
- 2,855 schedule games: 2,825 settled, 30 pending. Historical training population includes 267 warmup games and 2,558 core games (2017 onward).
- 70 served features; all 235 known served/candidate columns audited. Slices: full history, core, every season, each schedule team label, postseason, Week 1, current 2026, upcoming slate.
- PBP: 2016–2026; player stats: 2015–2026; NGS: 2016–2026; snaps: 2013–2026; FTN: 2022–2026; weekly/PIT injury and roster feeds: 2016–2026; player crosswalk: 22,679 rows.
- Schedule was requested with production cache bypass. Other acquisitions used production source interfaces and may have been satisfied by nflreadpy's own HTTP cache. This does **not** prove every upstream URL was network-refreshed independently.
- Hourly Open-Meteo acquisition completed with real provenance records. Both source-acquisition jobs, weather acquisition, and final rebuild exited zero. Historical/slate rebuild and rollup checkpoint took 28.00 seconds in this environment; no speedup comparison is claimed.

## Prioritized findings

### High — team identity is inconsistent across sources and state

The production joins use exact schedule labels, while PBP/player stats modernize `OAK→LV` and `SD→LAC`; NGS additionally uses `LAR` where the schedule uses `LA`.

| Source rollup | Settled team-games | Exact joined | Joined after diagnostic alias normalization |
|---|---:|---:|---:|
| PBP | 5,650 | 5,569 | 5,650 |
| Player stats | 5,650 | 5,569 | 5,650 |
| Snaps | 5,650 | 5,650 | 5,650 |
| NGS | 5,650 | 5,366 | 5,630 |

- PBP/player usage lose 65 Oakland and 16 San Diego team-games despite complete game-level source availability.
- All 48 core Oakland game rows have missing yards-per-play differences. This is **not** structural cold-start missingness.
- NGS loses 185 `LA` team-games plus the historical Oakland/San Diego aliases. Its slate features remain missing for the Rams' two upcoming games, although NGS source records exist as `LAR`.
- Player EPA observations join their own player-stat identity/position successfully (56,008/56,008), but target schedule-team identity still prevents the historical Oakland/San Diego pools from attaching.
- Current power-rankings delivery has 34 rows, consistent with historical and current labels being separate identities rather than a 32-franchise board.

**Recommended repair:** one shared franchise normalization at source/event boundaries, preserving original display labels and game IDs; regression tests spanning relocations and all source joins. Rebuild features, folds, final models, and delivery together after repair.

### High — availability coverage must distinguish current evidence, carry, and no source

The strict timestamped injury loader has **zero admissible rows in 2025 and 2026** because those raw feeds omit `date_modified`. Its fail-closed behavior is correct; weekly report-cycle and roster overlays are separate evidence channels, not timestamp-verified substitutes.

| Upcoming week | Team-games | Same-week weekly report present | Same-week roster present | Overlay flags present, including one-week carry |
|---|---:|---:|---:|---:|
| 5 | 30 | 21 | 30 | 30 |
| 6 | 28 | 0 | 0 | 26 |
| 7 | 2 | 0 | 0 | 0 |

- The Week 6 overlay includes carried Week 5 RES/SUS/PUP flags. Thus Week 6 injury features are **not all zero**; the earlier interim statement to that effect was incorrect. They lack current-week observation support and carry uncertainty about activations/new absences.
- Week 7's two team-games have no report, roster, or carry support, yet all 18 injury-share representations are populated by zero defaults. Non-null coverage is 100%, but this does not establish zero injuries.
- Feature attachment explicitly zero-fills missing sources/table entries. The global coverage gate detects absent/all-null columns, not unsupported zero values or missing team/season slices.
- GSIS→PFR mapping drops flagged rows: 2026 maps 2,803/3,025 union flags; only 1,977 have prior active snap position evidence, and 1,254 attribute to OL/DEF. Skill players, rookies, and inactive/practice-squad players need not belong to OL/DEF; these denominators are not interchangeable.
- Roster status vs independent positive offense/defense snap participation has 30 contradictions: 27 in 2016, 2 in 2017, 1 in 2025. These demonstrate that blanket same-week status semantics need exceptions/provenance; they do not alone prove the exact timing of leakage. The raw weekly snapshots have no capture timestamp with which to establish immutable pre-kickoff availability.

**Recommended repair:** expose per-team-game source/carry/missing provenance, fail or degrade unsupported inputs honestly, preserve an unknown state rather than healthy-zero where appropriate, and validate roster contradictions. Missing upcoming reports before publication are expected; presenting them as fully measured is not.

### High — current-season caches do not top up on ordinary runs

PBP, player stats, NGS, snaps, FTN, weekly injuries, and weekly rosters short-circuit when their season cache exists. Ordinary pipeline calls pass `use_cache=True`; there is no freshness/partial-season invalidation at these boundaries. Only the strict PIT injury path explicitly refreshes seasons with future targets, and that feed is empty for 2025–2026.

The audit's newly acquired 2026 PBP/player/snap/FTN sources cover all 64 settled games through Week 4. This is **not an observed lag at this snapshot**. It is a verified code-path risk: future ordinary runs can retain this partial season indefinitely until cache bypass/full repull. Weekly injury/roster caches can also retain obsolete designations.

**Recommended repair:** refresh mutable seasons/report cycles, retain immutable completed seasons, and record source fetch time/latest settled game against schedule expectations. Do not infer freshness from file existence or global non-null percentages.

### High — FTN attribution is wrong for team-tendency candidates

`ftn_team_agg` computes one mean over every charted play in a game and copies it to both team rows, despite retaining `nflverse_play_id`, which can join to PBP's `play_id/posteam`.

Diagnostic independent play-team attribution compared 2,406 team-games:

| Metric | Team-games differing | Mean absolute error |
|---|---:|---:|
| Motion rate | 2,404 | 0.06547 |
| Play-action rate | 2,399 | 0.02404 |
| RPO rate | 2,178 | 0.01808 |
| Screen rate | 2,395 | 0.01405 |
| Defensive box count | 2,406 | 0.23862 |
| Offensive backfield count | 2,405 | 0.06930 |

These are candidates, **not part of the default 70-feature served contract**; no adopted subset file is present. The trailing differences can be nonzero because each team has different prior opponents, so a nonzero-difference test does not validate attribution. Pre-2022 absence is structural; wrong team assignment in published years is not.

**Recommended repair:** join charted plays to offensive/defensive identity before aggregation, define which side each measure describes, and validate play-key coverage and independent team means before RFE use.

### High — board kickoff timestamps are mislabeled UTC

`serving._start_time_utc` puts the schedule's ET clock directly into a `Z` string. The delivered 2026-10-08 TB@DAL board says `20:15Z`; schedule conversion is `2026-10-09T00:15Z`.

An isolated real writer probe generated seven date-board CSVs for all 30 pending games, using explicitly neutral 0.5 probe probabilities (not model forecasts). Game IDs and row counts matched; **30/30 timestamps were four hours early**, including UTC date rollover errors. Other feature/injury/weather kickoff parsers correctly localize ET; this is a delivery boundary failure.

**Recommended repair:** reuse an authoritative timezone-aware conversion, test daylight-saving and winter offsets plus UTC midnight rollover, and verify frontend ordering/countdown behavior.

### Medium — NGS Super Bowl week numbering does not match the schedule

After franchise aliases are normalized diagnostically, 20 NGS team-games still fail: both teams in every Super Bowl from 2016–2025. Schedule SB weeks are 21/22; NGS uses the following week (e.g., 2025 NGS Week 23 vs schedule Week 22). This is a join-semantic discrepancy, not evidence that the source lacks the game. It can omit the last prior performance entering the next season, even when the current Super Bowl's pregame feature uses older history.

**Recommended repair:** reconcile postseason keys with game metadata rather than assuming numeric week identity across feeds.

### Medium — PBP pace and yards/play are not conventional offensive-play measures

Production counts all non-null `yards_gained` rows with a possession team, including rows outside `run`/`pass` play types. Against an independently filtered PBP offensive-type subset:

- Counts differ in **5,650/5,650** team-games, averaging **18.94 extra counted rows** per team-game.
- Yardage sums differ in **2,472/5,650** team-games.
- The elapsed-time denominator ranges 59.37–60.00 minutes and never exceeds regulation length, including overtime games.

This comparison establishes play-population/clock semantics; it is **not** a complete official-boxscore equivalence check (kneels, spikes, penalties, and overtime require explicit definitions). The metric may be intentional, but calling it ordinary offensive pace/yards-per-play is misleading without qualification.

**Recommended repair:** specify accepted play flags and overtime clock policy, reconcile to official offensive totals on representative games, then version/rebuild affected served metrics if definitions change.

### Medium — tie labels need an explicit moneyline/push policy

Ten historical ties are assigned `home_win=0.0`; Elo and team-win state use 0.5. Delivery uses `p_away_win=1-p_home_win` while separately reporting tie mass. A binary home-win event can legitimately treat ties as not-home-wins, but that is not a complementary decisive away-win moneyline event.

**Recommended repair:** document whether binary probabilities are unconditional or conditioned on no tie; exclude/handle pushes consistently in training, scoring, calibration, and board accuracy. This audit does not silently change the target definition.

### Medium — delivered artifacts are stale relative to audited code

2026-10-08 delivery metadata says `nfl-prod-v9.7-elo-season-revert`; audited source says v9.9. The existing prediction history contains **2,447 REG games only**, missing the 111 postseason core games now ingested and eligible. The new fold builder assigns all 2,558 core games to validation across 199 windows, though thin/postseason windows remain provisional for pooled grading.

The Oct 8 board's one row is its date-specific board, **not evidence that the 30-game multi-date slate was dropped**. This audit does not overwrite historical published cards or claim current-code model forecasts have been generated. Rebuilding features alone cannot certify the old bundle's predictions.

### Low — metadata gaps and documentation drift

- Core surface feature coverage is 98.28%; 44 core games have unknown/blank surfaces (including a 35-game 2023 cluster). Preserve unknown rather than fabricate grass/turf. Audit venue timelines before adding fallback facts.
- Rest differences cover 93.71% of core games. The 161 missing rows are consistent with first team-games within a season, not simply the 159 Week 1 games: a team can begin in Week 2 or encounter a new franchise-label seam.
- README/manifest text still describes 2018 warmup, 2019 onward REG-only; current config is 2016 warmup, 2017 onward with postseason. This invalidates documentation-based population assumptions.
- Stable missingness is not itself proof of structural absence. Monitoring should classify against actual eligibility/source-support masks, not only policy keywords and similar baseline/current percentages.

## Passing coverage and invariants

- No duplicate schedule game IDs, no one-sided score rows, and no recently overdue missing finals in the snapshot.
- Completed source game coverage: PBP, player stats, and snaps each contain **all 2,825 settled game IDs**. FTN contains every settled game ID in its 2022+ published window.
- The 2022 schedule has 271 REG games; the canceled BUF/CIN game ID is absent from the source schedule. It is not an unresolved completed game to invent or impute.
- Weather: historical **2,013/2,013 eligible outdoor games** populated; slate **20/20 eligible outdoor games** populated. Indoor/closed games correctly remain missing. All weather valid times precede kickoff; no indoor game carries weather.
- All 2,013 historical weather records were archive observations fetched after kickoff. They are **not archived forecasts known before kickoff**. All 20 slate forecast fetch times precede kickoff. Historical forecast backtest fidelity remains unverified.
- Strict PIT injury publications that were admitted precede exact kickoff; zero timestamp violations.
- All 235 known columns exist in history/slate; no entirely empty known column over the full historical population. This does not imply every season/team is covered.
- No infinite values; no violations in tested probability/flag ranges; every tested home-minus-away identity agrees (see coherence evidence). The only constant core feature is the intentional `is_home=1` anchor.
- 66/70 served slate columns are populated on every game; the remaining four are outdoor-only weather. Their overall slate coverage is 66.67%, with 100% eligible coverage.
- Core served coverage includes QB EPA diff 95.27%, TE EPA diff 96.52%, WR/RB EPA diffs 98.08%, PBP air-yards/defensive EPA diffs 98.05%; these mix meaningful missing lineup evidence with the alias defects above. They must not all be relabeled structural.
- Real-data causal checks over all 235 known features: prefix vs full future timeline (2,808 games), future-source poisoning (2,808 games), and deletion of preceding pending targets before the final pending game (one game): **zero mismatching features**. These are measured invariants, not proof that source publication timestamps were captured historically.
- Model-family interfaces exercised: linear width 30, tree width 72 including two categorical identifiers.

## Verification

- Production test script: **471 passed, 0 failed**.
- Backend fold/name/delivery/tee pytest checks: **34 passed**; 56 sklearn deprecation warnings, not test failures.
- All NFL backend Python files syntax-compiled. No configured/installed static typechecker was available; compilation is not a static typecheck.
- Real source/feature/fold/writer interfaces exercised. Writer UTC failures above remain deliberately reported rather than patched in an audit-only task.
- Feature-building emitted pandas fragmentation warnings. An initial audit import-path error and a Boolean poisoning-harness assignment error were corrected; final acquisitions/rebuild/causal checks completed successfully. No production assertions were weakened.
- No full estimator refit, frontend/live browser validation, external boxscore league-wide equivalence, or archived-forecast replay was performed in this audit. Those remain separate validation work.

## Evidence

Curated committed evidence lives in [nfl_coverage_audit_20261008](nfl_coverage_audit_20261008/):

- [Audit results and populations](nfl_coverage_audit_20261008/audit_results.json)
- [Snapshot manifest](nfl_coverage_audit_20261008/audit_manifest.json)
- [All feature coverage slices](nfl_coverage_audit_20261008/feature_coverage.csv)
- [Source team-game reconciliation](nfl_coverage_audit_20261008/source_reconciliation.csv)
- [Source season coverage](nfl_coverage_audit_20261008/source_season_coverage.csv)
- [Slate availability evidence](nfl_coverage_audit_20261008/slate_source_support.csv)
- [Side/difference coherence](nfl_coverage_audit_20261008/side_diff_coherence.csv)
- [Causal checks](nfl_coverage_audit_20261008/causal_checks.json)
- [NGS residual postseason joins](nfl_coverage_audit_20261008/ngs_canonical_unmatched.csv)
- [Roster status/active snap contradictions](nfl_coverage_audit_20261008/roster_unavailable_active_snap_overlaps.csv)
- [Delivered kickoff comparison](nfl_coverage_audit_20261008/delivered_kickoff_comparison.csv)

Large raw/source/feature checkpoints and detailed per-player/per-play diagnostics remain local under the ignored `nfl-backend/run_diagnostics/coverage_audit_20261009/` directory. They are not included in the commit.
