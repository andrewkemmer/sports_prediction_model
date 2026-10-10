# NFL feature-coverage audit, follow-up pass — 2026-10-09 ET

## Verdict

**The earlier alias, NGS Super Bowl, FTN attribution, board-time and unsupported-health-zero repairs are verified. Coverage is still not a universal source/freshness/predictiveness pass.** Current-week reports, one-week carry and no evidence are different states. Historical weather is observed after kickoff, not an archived forecast. Some venue facts and player-position pools remain unsupported.

This is an engineering/source-support audit, **not an NFL predictive-accuracy or tuning claim**. No NFL model, forecast, feature membership or production delivery artifact is replaced. The only source acquisition outside the initial cached replay is a bounded current-2026 refresh through existing ingestion interfaces plus independent network probes. All new committed files are reports/evidence, never programs.

## Population and method

- Current source contract `nfl-prod-v9.9-pending-state`.
- Production window bounded for this audit: 2016-01-01 through 2026-10-22; **2,855 schedule games, 2,826 settled, 2,559 core, 29 pending**.
- **70 served / 235 known** features inspected across history/core, every season, every team label, Week 1, postseason and slate. Rebuilt from current feature code, not the previous audit's pre-remediation feature parquet.
- Source denominators are **team-games**, not merely any row with the game ID. Current refresh follows production normalizers; aliases apply across schedule/state/source joins.
- Production builder, family matrices, fold geometry, measured/structural monitoring and actual board CSV writer are exercised directly. Pipeline entry point is not executed because it also replaces/pushes delivery files.

## Source freshness: a real stale-cache finding and measured recovery

The initial local snapshot already contained the official **TB 24 – DAL 16** final from October 8, but PBP, player stats, NGS and FTN lacked that game's two team sides. These were not historical alias failures: every older supported team-game matched. This snapshot would price subsequent TB/DAL form from stale detailed evidence despite an updated score.

Independent network downloads at approximately 04:00 UTC October 10 show:

- nflverse 2026 PBP **11,327 rows**, including **172 TB–DAL rows**; Last-Modified October 9 16:09:21 UTC.
- nflverse weekly player stats **4,517 rows**, including **68 TB–DAL rows**; Last-Modified October 9 16:11:07 UTC.
- Therefore the local absence was **not “upstream has not published yet”** at probe time. The report preserves [before-refresh reconciliation](nfl_coverage_review_20261009/before_refresh_source_reconciliation.csv) and [before-refresh summary](nfl_coverage_review_20261009/before_refresh_audit_summary.json).

A bounded `use_cache=False` current-season refresh through existing `load_pbp`, `load_player_stats`, `load_nextgen` and `load_ftn_charting` restores the detailed source population. The normal production code already repulls mutable seasons; no new cache bypass or program was committed. The entire frame and causal audit were rerun after refresh, not just the counts.

[After-refresh reconciliation](nfl_coverage_review_20261009/source_reconciliation.csv) and [refresh inventory](nfl_coverage_review_20261009/production_refresh.json) distinguish repaired local acquisition from a claimed source-code repair. This does not independently prove nflreadpy's own HTTP cache refreshes every vendor URL; the direct PBP/player downloads establish freshness for those two assets only.

## Prior audit findings rechecked

### Franchise normalization — verified

PBP/player/snap joins no longer lose OAK/SD or split LA/LAR/LV/LAC identities. [Team-game reconciliation](nfl_coverage_review_20261009/source_reconciliation.csv) checks exact canonical keys across the entire settled population. The initial residuals were only the newly finished TB–DAL game; after refresh **PBP, player stats, snaps and NGS each join 5,652/5,652 team-games**. Refreshed denominators are recorded in [summary](nfl_coverage_review_20261009/audit_summary.json).

### NGS postseason week alignment — verified

**20 NGS rows** are reconciled to the schedule's Super Bowl week. The former 20 canonical postseason holes no longer persist. Weekly/season/team identity is retained rather than inferred from game count alone.

### FTN possession attribution — verified

The historical replay binds **196,044/196,044 charted plays** to actual PBP possession identity (zero missing PBP keys in that snapshot), rather than copying both teams' combined tendencies. The refreshed charting population and attribution count appear in the after-refresh build log and summary. FTN absence before its 2022 publication window is not a failure; eligible-window coverage must be evaluated separately from all-history percentages. **Two FTN team-game holes still remain inside the eligible window (TB–DAL)** after refresh: that charting feed has not supplied the newly finished game. Its 10,829 current-season rows are unchanged. Do not mistake successful refresh for complete FTN coverage.

### Availability support — repaired behavior verified, uncertainty remains

[Slate support](nfl_coverage_review_20261009/slate_source_support.csv), 58 team-games:

| Week | Team-games | Same-week weekly report | Same-week roster | Prior-week roster | Supported injury aggregate |
|---|---:|---:|---:|---:|---:|
| 5 | 28 | 28 | 28 | 28 | 28 |
| 6 | 28 | 0 | 0 | 26 | 26 |
| 7 | 2 | 0 | 0 | 0 | 0 |

- **Four sides** have no report/roster/carry support: two in Week 6 and two in Week 7. Their injury-share fields stay NaN; they do not publish unsupported healthy zeros. Week 6's other 26 sides are carry-supported, **not current-week observation**.
- Strict timestamped PIT loader admits **49,445 rows**, all 2016–2024, with **zero publication-at/after-kickoff violations**. It admits no timestamp-verified 2025/2026 rows. Weekly reports and roster snapshots are separate report-cycle evidence, not historical publication-time proof.
- The absence of report rows for a team can mean no reported injuries or an unpublished report. A roster snapshot supplies a separate channel, but its healthy zero does not establish a complete current injury filing. The audit preserves `ev_report`, `ev_roster`, `ev_flag` and current/carry support explicitly.
- GSIS/PFR mapping and absent prior snap/position evidence can still underprice some unavailable players. The existing OL/DEF quantities cannot be interpreted as all-position injury completeness.

### Rest/roof structural proof — verified

The monitor uses **pool-anchored, side-agnostic** season openers, not the first appearance in a short baseline slice. Current/baseline report: **126 OK / 14 STRUCTURAL**, no STARVED/LOW_COVERAGE in these two windows. The eligible masks explain missing rest/roof features; this does not justify relabeling arbitrary player-pool/source gaps as structural. Actual percentages remain visible.

### Board UTC writer — verified through written CSVs and live schedule

The actual `write_board_csv` interface writes **29 rows across six dates**; every parsed UTC stamp equals independent `_kickoff_utc` recomputation, **zero mismatches**. The writer probe uses clearly synthetic 0.6 probabilities, never production model output; see [writer evidence](nfl_coverage_review_20261009/board_writer_probe.csv).

Fresh ESPN scoreboard: TB–DAL final scores and **2026-10-09 00:15Z** match the snapshot and official [NFL recap](https://www.nfl.com/news/buccaneers-cowboys-on-thursday-night-football-what-we-learned-from-tampa-bay-s-24-16-win). October 11 scheduled matchups/tipoffs match the loaded schedule, including **PHI–JAX London 13:30Z**, 1 PM ET games **17:00Z**. ESPN's pregame score strings “0” are not adopted as settled scores. [Live evidence](nfl_coverage_review_20261009/live_source_probes.json).

## Remaining coverage/semantics risks

- Weather core coverage **70.85%** is an all-game denominator, not a fetch success rate: roofs/unknown venue support account for much of it. **2,013 historical observations were fetched after kickoff**. Valid time preceding kickoff is not evidence the forecast was available then. Do not call this an archived-forecast backtest.
- `is_turf_home` **98.28%** core coverage: 44 unsupported rows remain; not all can be explained as weather roof policy. Venue/surface historical truth still needs an independently versioned timeline. Travel differences have four core gaps.
- Core rest-diff **93.71%** (season openers), QB EPA diff **97.15%**, TE EPA diff **98.40%**. Player-pool holes must be tested against membership/identity evidence, not automatically classified structural.
- PBP `ypp` uses every possession-team row with non-null recorded yards, not exclusively conventional offensive runs/passes. Pace is regulation-clock-denominated even in OT. These definitions are declared; no statistical efficacy claim follows from being documented. Redefining them needs a matched predictive experiment.
- Tie semantics remain questionable: `home_win=0` on ties while complement is reported as away-win probability; both cannot be unconditional decisive-win probabilities. The existing separate tie mass does not by itself make an unadjusted `1-p_home` an unconditional away-win probability. Coverage cannot validate that probability contract; a three-outcome or explicitly conditional moneyline policy needs its own review.
- Roster snapshot/report-cycle channels have no complete historical fetch/publication ledger. Passing future-outcome poisoning proves builder chronology, not source availability at historical origin.

## Invariants and checks

- Three real-data prefix/future-timeline replays at 2020-09-13, 2026-09-13, 2026-10-08: **zero differences over 235 known features**. Final pending game's feature row also remains identical after preceding pending targets are removed. The after-refresh rerun repeats these checks; [causal evidence](nfl_coverage_review_20261009/causal_checks.json).
- No infinities, no tested home-minus-away mismatches; the only constant core served feature is intentional `is_home`. [Coherence](nfl_coverage_review_20261009/side_diff_coherence.csv), [coverage slices](nfl_coverage_review_20261009/feature_coverage.csv), [monitor windows](nfl_coverage_review_20261009/monitor_coverage.csv), [fold geometry](nfl_coverage_review_20261009/fold_geometry.csv).
- NFL production script **523 passed, zero failed**; backend pytest **34 passed**. Tests are distinct interfaces, not 557 independent new tests. Logs: [production checks](nfl_coverage_review_20261009/production_tests.txt), [pytest](nfl_coverage_review_20261009/backend_tests.txt).
- Syntax compilation/whitespace verified; no configured/installed static typechecker, so compilation is not advertised as typechecking. [Verification](nfl_coverage_review_20261009/verification.json).
- No NFL estimator refit/OOF ablation, frontend/browser check, league-wide independent box equivalence, live injury signoff, or archived-forecast replay. No promised AUC/logloss/Brier improvement from this coverage audit.

The point of the second pass is to distinguish **observed, carried, unsupported and stale** inputs, not make every coverage percentage green. All reports are bounded evidence; raw season data, rebuilt frames and scratch programs remain uncommitted.
