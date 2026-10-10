# NHL EVO/PPO player-pool review — 2026-10-10

## Verdict

**No calculation or causal defects were found in the EVO/PPO family, and no repairs were required** — a different outcome from the [NBA RAPM review](nba_rapm_review_20261009.md), where real leakage repairs were made. Independent recomputation from the raw MoneyPuck archives reproduces production exactly; prefix deletion and future poisoning at three cut dates leave every rating row before the cut bit-identical; the served `pl_*` columns are unchanged under every timing arm. The review is otherwise structurally parallel to the NBA one: causal replay, independent arithmetic, a matched production walk-forward with paired bootstrap, sliced coverage, and live-source probes.

Predictive value is real on the even-strength side and not distinguishable from noise on the power-play side: removing EVO moves all three paired intervals away from zero; removing PPO moves none. The pooled 24-column contrast is directionally positive — the Brier interval excludes zero while the AUC and log-loss intervals touch it — so, with the same admission discipline as the NBA/NFL reviews, this is **promising, not a blanket statistical pass**. No hyperparameters, shrink arm, blend policy, routing or feature membership were tuned or adopted on this replay; production delivery files and model bundles were not replaced; no source files changed.

Family under review: the 24 `pl_*` columns = {EVO, PPO} × {C, L, R, D} × {home, away, diff}, built from MoneyPuck per-player-game rows (5on5 = EVO, 5on4 = PPO) through the pool → shrink → serve chain. The `pl_il_out_fraction` availability column rode along in the serving-layer timing checks for completeness but is not part of the family or the 68-column contract.

## Calculation and causal verification — no repairs required

1. **Production reproduction.** Re-mapping the raw MoneyPuck archives inline (player id, name, team, position, situation, xg, ice seconds, source date — 1,559,414 rows, 3,079 players, 19 seasons 2008–2026, max game date 2026-10-08) and rebuilding with the shipped builder is **bitwise identical** to `features._load_player_ratings`: 1,559,414 matched rows, 0 rows only on either side, and every stored column — including `k_seconds`, `league_rate_per60`, `injury_multiplier` and `final_rate_per60` — exact with identical NaN patterns.
2. **Independent arithmetic.** A reimplementation written from the documented spec — trailing 30 played-game rows per player and situation; league prior computed cumulatively with the entire candidate date subtracted so same-day games cannot enter; k = `PLAYER_RATING_SHRINK_FRACTION` (0.20) × mean season ice time per position×situation from **completed seasons strictly before the rating's season**; Bayesian w = n/(n+k) — reproduces all six computed terms on 950 stratified production rows (including cold rows with no league prior and season-boundary rows) with **max abs diff 0.0 and identical NaN patterns on every term**. The eligibility filter agrees as a row set: production rows equal independently eligible rows exactly (0 dropped for position/situation/ice, 0 only-a / only-b).
3. **Causal timing.** Three cut dates — 2026-01-15 (mid-season), 2025-10-09 (season boundary), 2024-10-22 (early season) — each tested two ways against the full build: **prefix deletion** (ratings built from rows dated strictly before the cut only) and **future poisoning** (every row from the cut onward rewritten to xg×7.3+1000). All six comparisons leave every pre-cut rating row **bitwise identical** (1,513,444 / 1,460,814 / 1,372,340 rows compared; 0 only-base / 0 only-arm). Same-day rows cannot enter a rating at all: the league prior subtracts the whole candidate date, the rolling window is backward-only, and k reads only completed prior seasons.
4. **Serving layer.** For the next 10 / 14 / 16 target games on each cut date, the 25 player-level columns (24 `pl_*` + `pl_il_out_fraction`) are **identical under full ratings, prefix-truncated ratings, future-poisoned ratings and same-day-dropped ratings** — nine arm×cut cells, all `identical=True`, max abs diff 0.0. The pool joins source dates strictly before the target game's date; a rating dated the target date is never served to that game.
5. **Harness-side only.** Three defects surfaced during this review and **all three were in my verification code, not production**: k was first aggregated per player instead of per player-season (production's averaging unit is the player-season, as its docstring states, and that is what matches); my independent k lookup used a swapped `(situation, position)` key; and pandas 3 parses string dates as µs while a parquet-reloaded grid carried ns, breaking a merge. After those fixes, production matched exactly. One latent robustness note stands: `_expand_to_games` would raise a pandas `MergeError` if handed a `datetime64[ns]` game grid; current production paths build grids from string-dated caches (µs), so the path is not reachable today.

Verified construction facts: a pool side below `MIN_PRIOR_ICE_SECONDS` (900 s) serves the position prior — EVO C 0.660 / L 0.721 / R 0.698 / D 0.169, PPO C 1.581 / L 1.627 / R 1.518 / D 0.630 per 60 — never zero and never NaN. Roster membership is capture-based: a capture applies only to games dropping strictly after it, positive-knowledge-only, `complete=False` on partial fetches, 6-hour TTL. Shrink is Bayesian 0.20; the ramp alternative was A/B-gated 2026-10-02 as a wash and remains selectable via `NHL_SHRINK_ARM=ramp`.

## Predictive validation: matched production walk-forward

- Same frame as the shipped Kaggle baseline: **2,847 games × 148 columns** (`nhl_game_frame_20261007.parquet`), **60 expanding windows**, seed 42, production config defaults; the harness verified its config instance is identical to `features`'. 14 of 60 windows are thin (< `MIN_VAL_FOLD_GAMES` = 40, mostly playoff weeks and season ramps) and PROVISIONAL — excluded from pooled metrics, blend weights and calibration; 168 postseason rows are reported separately and never grade the blend. **2,465 grading games, identical across all arms** (grading-set equality asserted per arm).
- Five arms, each re-trained end-to-end with the three shipped members (xgboost, LightGBM, elastic-net) and the rolling blend: baseline (all 24), no_pl (drop all 24), no_evo (drop the 12 EVO), no_ppo (drop the 12 PPO), and tree_only_levels (the 16 per-side levels routed to the tree members only, linear keeps the 8 diffs — the MLB-mirror routing).
- Determinism: the baseline arm re-run is **bitwise identical** (Δ = 0.0 on AUC, log loss and Brier), so the paired design is sound.

| Grading population (n = 2,465) | AUC ↑ | log loss ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| **Baseline — all 24 `pl_*`** | **0.598571** | **0.674618** | **0.240900** | 0.013620 |
| No `pl_*` family | 0.586022 | 0.677704 | 0.242415 | 0.013730 |
| No EVO (PPO only) | 0.580126 | 0.679026 | 0.243072 | 0.010688 |
| No PPO (EVO only) | 0.593651 | 0.675736 | 0.241444 | 0.012827 |
| Tree-only levels (MLB mirror) | 0.598057 | 0.674703 | 0.240943 | **0.010147** |

[Paired uncertainty](nhl_evo_ppo_review_20261010/predictive_metrics.json): 1,200 shared-date bootstrap replicates, seed 42, arm minus baseline:

| Arm | Δ AUC (95% CI) | Δ log loss (95% CI) | Δ Brier (95% CI) |
|---|---|---|---|
| No family | −0.012549 [−0.024649, +0.000109] | +0.003087 [−0.000025, +0.006357] | +0.001515 [+0.000010, +0.003091] |
| **No EVO** | **−0.018445 [−0.029824, −0.006528]** | **+0.004409 [+0.001478, +0.007317]** | **+0.002172 [+0.000782, +0.003582]** |
| No PPO | −0.004920 [−0.012169, +0.002072] | +0.001119 [−0.000761, +0.003027] | +0.000544 [−0.000374, +0.001477] |
| Tree-only levels | −0.000515 [−0.002109, +0.001055] | +0.000086 [−0.000351, +0.000500] | +0.000043 [−0.000164, +0.000250] |

Interpretation:

- **EVO is load-bearing.** Removing it excludes zero on all three metrics — the strongest result in this replay.
- **The pooled family is directionally important, not a blanket pass.** The no-family Brier interval excludes zero by ~1e-5; AUC and log-loss intervals touch zero.
- **PPO alone is not distinguishable from noise** — all three intervals cross zero. The cross-contrast point estimates (not preregistered, unadjusted): dropping EVO while keeping PPO (AUC 0.580126) is *worse* than dropping the whole family (0.586022), i.e. the PPO columns do not carry the family on their own here.
- **Season slices of the no-family contrast** are negative in all three seasons: 2024 (n = 1,113) Δ AUC −0.018744 / loss +0.005283 / Brier +0.002564; 2025 (n = 1,305) −0.006537 / +0.000922 / +0.000484; 2026 (n = 47, first-week and provisional-sized) −0.020370 / +0.011188 / +0.005292.
- **tree_only_levels is a wash** on all three bootstrapped metrics (every interval straddles zero). Its ECE is better (0.010147 vs 0.013620) but ECE carries no paired interval in this replay. Following MLB's admission discipline a wash is **not adopted**: NHL keeps routing all 24 `pl_*` columns to both members (linear receives 16 levels plus the 8 exact home−away−diff triads — verified with 0 identity violations and 0 team-id columns).
- Replay vs recorded baselines: this baseline replays at AUC 0.598571 / loss 0.674618 against the recorded tune run 0.597500 / 0.674755 and the 2026-10-08 Kaggle run 0.595957 / 0.674723; recorded values come from earlier runs/frames. The five arms in this replay are internally paired — that pairing is the evidence.
- **Admission decision:** retain current membership, Bayesian 0.20 shrink, blend and routing; do not claim a statistically established full-family gain; a sealed future walk (or an independent holdout that covers playoff strata) is the appropriate next gate.

## Feature-coverage follow-up

[Full sliced coverage](nhl_evo_ppo_review_20261010/feature_coverage.csv) sweeps all 24 `pl_*` columns across full history, regular season, postseason, first-ten and post-thirty season days, and each season (2024: 1,398 games, 2025: 1,394, 2026: 55; full history 2,847). "Measured" is pool-served, i.e. non-default, per the `_pool_default_mask` exact-prior equality — `monitoring.coverage` counts default-filled rows as non-measured since eb93d2ce (2026-10-09).

Measured share of the diff columns:

| Slice (n games) | EVO-C | EVO-L | EVO-R | EVO-D | PPO-C | PPO-L | PPO-R | PPO-D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full history (2,847) | 99.58% | 98.98% | 99.58% | 99.58% | 99.58% | 96.00% | **91.25%** | 99.58% |
| Regular season (2,679) | 100% | 99.37% | 100% | 100% | 100% | 96.42% | 91.15% | 100% |
| Postseason (168) | 92.86% | 92.86% | 92.86% | 92.86% | 92.86% | 89.29% | 92.86% | 92.86% |
| First 10 days, any season (176) | 100% | 100% | 100% | 100% | 100% | 98.86% | 90.91% | 100% |
| Season 2026 (55) | 100% | 100% | 100% | 100% | 100% | 98.18% | **83.64%** | 100% |

- **Zero nulls anywhere**: across all 24 columns and every slice the null count is 0 — the position prior is never NaN and never 0 by design. Full history totals 67,325 measured sides + 1,003 default sides = 68,328 = 24 × 2,847, with `pct_nonnull` = 100% on every slice.
- The real holes are **PPO-R (249 default diff sides, 91.25%) and PPO-L (114, 96.00%)** — sides where trailing prior ice for the power-play unit never cleared `MIN_PRIOR_ICE_SECONDS` (900 s); season 2026's first week pushes PPO-R to 83.64%. This is the documented default path (pool refusal → position prior), flagged by the default mask and excluded from `pct_measured`, not silent fabrication. Postseason shares rest on 168 games and are lower-resolution.
- Team attribution, side-weighted across all 8 diff pools: league mean 98.81%; worst CHI 87.95%, CAR 94.49%, PIT 95.19%, FLA 95.68%, EDM 96.94%; five teams at 100% ([team coverage](nhl_evo_ppo_review_20261010/team_coverage.json)). CHI's share reflects thin roster-level PP ice against the pool threshold, consistent with the default mechanism rather than missing data.

## Live sources and freshness

[Bounded probes](nhl_evo_ppo_review_20261010/live_source_probes.json), fetched 2026-10-10 06:19 UTC:

- **MoneyPuck live equals cache**: the live 2026 skater archive's max game date matches the cached `moneypuck_player_games_v2_2026.parquet` (both 2026-10-08), so the rating source is current with its publisher. MoneyPuck's publish cadence is the binding freshness constraint on `pl_*`.
- **NHL API**: 2026-10-09 carries **4 completed games** beyond both the evaluation frame (through 2026-10-07) and MoneyPuck (through 10-08); they will be picked up on the next pipeline run. 2026-10-10 shows 14 future games. The frame is a fixed evaluation snapshot, not a live slate.
- **ESPN injuries**: live payload 31 teams / 124 injured players; the local history cache was written 2026-10-10 05:43 UTC, same day.
- **Roster snapshot**: captured 2026-10-10 05:43:55 UTC, complete for the 12 teams its requesting grid asked for (ANA…FLA), 291 player-team rows, season id 20262027. The loader is request-driven by design — it fetches exactly the teams a grid needs and refetches when uncovered teams appear — so a 12-of-32 snapshot is partial-by-design and fails closed: an absent player keeps their row team (positive knowledge only) and a stale capture keeps its original timestamp.

These are sampled network checks, **not** full-window freshness proof.

## Verification and evidence contract

- Full NHL backend suite: **610 passed**, 32 sklearn/pandas deprecation warnings only, 618 s ([backend tests](nhl_evo_ppo_review_20261010/backend_tests.txt)).
- The `nhl-backend` working tree is unchanged: **no source files were edited** because no repairs were needed; feature construction, models, blends and serving paths are untouched.
- Curated JSON/CSV/TXT/Markdown evidence only is added. All executable harnesses and bulky caches remain uncommitted — **no new committed programs**. No static typechecker is configured in this repository; no code changed, so none was run.
- Fold geometry, member OOF log-losses and per-arm column counts: [fold geometry](nhl_evo_ppo_review_20261010/fold_geometry.txt). Interface facts — 24/24 family columns in both members, 8 exact linear triads with 0 identity violations, 0 team-id columns, default-mask monitoring contract: [interface verification](nhl_evo_ppo_review_20261010/interface_verification.json). Causal/arithmetic checks: [causal checks](nhl_evo_ppo_review_20261010/causal_checks.json).

The frame, five-arm metrics, paired bootstrap and sliced coverage are sufficient to inspect every numerical claim without trusting prose. This is a historical replay over revised source archives, not a guarantee of future AUC/loss/Brier performance.
