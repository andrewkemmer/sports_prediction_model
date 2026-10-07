# NHL input-semantic repairs

## Scope

Implemented the four correctness repairs from the NHL audit: Elo home advantage,
team power-play opportunities, team faceoff rates, and expected-goalie membership.
No blend weights, model hyperparameters, calibration rules, committed data-delivery
artifacts, or Kaggle notebook were changed. No push or deployment was made.

## Corrected contracts

### Elo

Positive home advantage now raises home expected win probability:

`E_home = 1 / (1 + 10 ** ((R_away - R_home - home_advantage) / scale))`.

The away expression is complementary. At equal ratings and H=65, expected home
win rate is 59.2466%, not 40.7534%. Rating updates conserve the rating sum.
Historical and slate paths share the same chronological implementation and
season-boundary reversion. Unplayed rows carry entering ratings but never update
skill from an artificial half-win.

### Official team counts

[Ingestion](../nhl-backend/backend/ingestion.py) fetches both the settled `boxscore`
and existing official `right-rail` endpoint. `teamGameStats.powerPlay` supplies
PP goals/opportunities; `faceoffWins` supplies team wins/attempts. The internal
wide schema adds `home_faceoff_wins`, `away_faceoff_wins`,
`home_faceoff_attempts`, and `away_faceoff_attempts`; the existing team percentage
columns are retained as count-derived rates. Model-facing feature names are
unchanged.

The served five-game PP and faceoff rates pool paired numerator/denominator
counts and shift strictly before the target game. A zero-opportunity game is
observed zero; a missing or malformed count pair contributes neither numerator
nor denominator. No fallback to goalie shots or averaged skater percentages.

Cache version **v5** prevents reuse of v4 proxy data. Complete corrected rows
are cached; missing right-rail counts preserve observed SOG/goalie facts but
leave the row uncached so later runs retry. This adds a second API request per
uncached settled game; no timing/speedup claim is made.

### Expected goalie

[Feature construction](../nhl-backend/backend/features.py) ranks strictly-prior
starts by **target team and season**, not all-team workload. A goalie's latest
prior observed start for another team excludes him from the former club. Fresh
captured roster membership is applied only when capture < exact puck drop,
within the existing six-hour roster TTL. Incomplete captures support positive
membership evidence only; absence alone cannot exclude a player. Complete
captures can exclude absent players only for represented teams. Future or stale
captures cannot rewrite historical choices.

Goalie names come from the latest eligible prior start, not a future boxscore.
Nullable numeric goalie IDs normalize to central roster IDs. Injury/leave checks
use exact puck-drop time when available; goalie statistical history conservatively
remains date-prior. Both decided-history and scheduled-slate builders pass roster
captures through the same gate.

**Limitations:** this is still a workload-based expected starter, not confirmed
pregame starter information. Without a prior roster/transfer observation, an
unobserved move cannot be known. Opening-night starter values remain unknown
when the team has no start in the target season. Historical roster archives are
not fabricated or backfilled from today's roster.

## Versioning and rollout

[Configuration](../nhl-backend/backend/config.py) now identifies features as
`nhl-prod-v1.2-input-semantics`; the [manifest](../nhl-backend/backend/manifest.py)
documents corrected definitions and versions. Existing `v1.1` bundles have the
same column names but different feature meaning: **do not serve corrected frames
through the old fitted bundle**.

Run a complete feature rebuild, walk-forward evaluation, blend/calibration
refit, and final model refit as one operation. The [README](../nhl-backend/README.md)
contains the local no-push command. `--skip-pull` may reuse other valid caches,
but the first corrected run must fetch v5 boxscores and right-rail counts.
Old cache files are left untouched, simply not consumed.

The audit's pre-repair metrics and evidence remain intact as historical records.
A full historical retrain has not been run here: the local training cache and
member OOF stores remain absent. No improvement in AUC/log loss is claimed until
that rebuild is measured under the causal evaluation policy.

## Verification

- Added [semantic regression tests](../nhl-backend/backend/test_input_semantics.py)
  for official sample facts, count pooling, unknown/zero/malformed observations,
  failed-source retry, old-cache isolation, Elo symmetry/conservation, opening
  season parity, team-specific goalie workload, trades, capture boundaries,
  stale/partial rosters, nullable IDs, historical invariance, and builder wiring.
- Updated existing [PIT tests](../nhl-backend/backend/test_run_engine_pit.py) whose
  fixtures/assertions previously encoded goalie PP shots as opportunities.
  Assertions now enforce the correct hockey event rather than preserving the
  faulty proxy. Synthetic faceoff fixtures carry actual counts.
- The real loader fetched official game **2024020194** and reproduced **4 vs 3**
  PP opportunities and **27/62 vs 35/62** team faceoff rates. Its cached replay
  was identical. [Verification JSON](nhl_input_repairs/official_loader_verification.json).
- [Focused ingestion/goalie/PIT run](nhl_input_repairs/focused_tests.txt):
  **27 passed**; [roster/leave/grid-contract run](nhl_input_repairs/contract_tests.txt):
  **56 passed**, before the final nullable-ID hardening.
  [Final targeted semantic/audit run](nhl_input_repairs/targeted_tests.txt):
  **38 passed**. Compilation and whitespace checks passed.
- Final-code [full-suite results](nhl_input_repairs/test_results.txt):
  **577 passed, 1 failed**, 62 dependency deprecation warnings, in 246.33 seconds;
  pytest exit code **1**. The only failure is the previously recorded
  `test_the_kaggle_notebook_installs_tqdm_or_there_is_no_bar_to_draw`: the
  Kaggle-owned notebook does not install `tqdm`. It is outside this repair and
  has not been skipped, weakened, or fixed by editing the owned notebook.
  [Earlier full run](nhl_input_repairs/test_results_pre_id_hardening.txt) is
  retained separately and is not presented as verification of the final code.
- No configured/installed mypy or pyright checker was found; Python compilation,
  runtime assertions, and pytest are the available checks. Client change hooks
  are unavailable in this environment.

Remaining audit recommendations—causal blend reporting, fit/refit parity,
run-engine probability coherence, genuinely timestamped availability, and
non-Elo pending-row contamination in the team-stat ladder—are separate work,
not silently bundled into these input repairs.
