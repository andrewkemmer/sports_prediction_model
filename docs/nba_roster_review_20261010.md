# NBA roster-information closure — 2026-10-10

## Delivered change

Feature representation is **`nba-prod-v2.9-rapm-dated-rosters`**. Current membership now comes from the already-used official NBA `playerindex` endpoint, queried with `Historical=0` for the current season. It uses **NBA PERSON_ID**, not ESPN athlete IDs, so no speculative crosswalk or name matching is involved.

The forward roster gap is closed **from the first eligible observed snapshot onward**: team changes, cuts/exits, and arrivals can be known before a player's first new-team box score. This is not a reconstructed historical transaction feed and does not prove when an earlier offseason move first became knowable.

## Evidence contract and interfaces

- Parser: `nba_sources.roster_membership(payload)` returns `player_id`, `player_name`, `team`, `position`, `positions`. It requires unique IDs, recognized teams, all 30 clubs and at least ten members per club. Active membership comes from `ROSTER_STATUS=1`, **not injury availability**. Partial/ambiguous pulls are refused. An unpublished position is retained as unknown membership rather than rejecting the league snapshot.
- Observation: `ingestion.fetch_roster_history()` fetches fresh current state and appends one JSONL record to `NBA_CACHE_DIR/roster_history.jsonl`. Knowledge time is the **actual UTC response-completion time**, never cache mtime, injury-publication date, season label, or a historical query date. An outage does not retimestamp old evidence.
- Durable delivery: [nba_roster_history.parquet](../nba-backend/data_delivery/nba_roster_history.parquet) is the union of delivered and local observations. It is a protected retention master and survives an empty machine cache. Its columns are the parser columns plus `observed_at`, `season`, `source=nba_playerindex`. Each observation remains complete and independently dated; the union does not collapse players across observations.
- Rating API: the existing `build_player_rapm(..., roster_history=None)` remains backward-compatible for callers omitting history. Rating CSVs add `roster_source` and `roster_observed_at`; the nine existing `pl_rapm_*` model feature names are unchanged.
- Selection: latest same-season observation **strictly before target-day midnight America/New_York**, no older than **seven days**. The cutoff is deliberately earlier than tipoff because ratings are date-keyed. Same-day/future observations never rewrite historical membership. A later strictly-prior positive-minute appearance supersedes an older roster, including a player omitted from it.
- Unknown/expired observations retain the explicit `prior_appearance` fallback. Eligible official membership carries `nba_playerindex` and its original observation instant. Malformed/ambiguous eligible archive observations fail explicitly rather than silently removing teams.
- Membership changes do **not** alter raw ridge impact, fit evidence, exposure, minutes, games, or appearance recency. The position means remain computed from strictly-prior fitted players before membership replacement. A member without fitted evidence has NaN raw impact and zero exposure; the existing rotation floor excludes him. Missing source positions retain an evidenced prior label if possible; an unknown newcomer stays unrated/unprojected.
- Projection uses the target's current membership, not old-team rows in the rating lookback. That closes a second resurrection path for transfers and cuts.
- Production feature builder, player-rating CSV writer, and existing A/B harness all read the same roster history. The pipeline republishes the ledger alongside its other protected source archives. No new executable programs are committed.

## Measured live verification

[Verification JSON](nba_roster_review_20261010/verification.json), observation **2026-10-10T04:53:37.619721+00:00**:

| Check | Result |
|---|---:|
| Official observed membership | **622 players, 30 teams** |
| Team sizes | 18–24 |
| Unpublished positions | 11 |
| Cached player-game input | **84,768** |
| Prior box-derived membership | 582 |
| Changed team assignments | **142** |
| Members removed from old pool | **101** |
| Members without prior box evidence | **141** |

Membership reconciliation for a **diagnostic 2026-10-11 target** exactly matches all 622 official team assignments. Raw impact, prior effective games, minutes and games remain exactly unchanged for retained members. All 141 new members have zero prior exposure/minutes and NaN raw impact. Unknown positions do not fabricate ratings.

Four historical targets (2024-10-22, 2025-02-08, 2025-10-21, 2026-04-01) remain **bit-identical** with/without the new observation. The observation's own Eastern calendar date is also unaffected. Cold-cache delivered-ledger roundtrip matches all rows and observation timestamps. Real projected pools produce 30 team rows; diagnostic home/away/difference identities hold and contain no infinity. The bounded verification took **22.861 seconds** on this host; this is not a claimed speedup.

Evidence: [observed roster](nba_roster_review_20261010/observed_roster.csv), [team changes](nba_roster_review_20261010/membership_changes.csv), [exits](nba_roster_review_20261010/membership_exits.csv), [diagnostic pools](nba_roster_review_20261010/diagnostic_pools.csv), [feature join](nba_roster_review_20261010/diagnostic_feature_join.csv). These are membership/interface diagnostics, **not actual scheduled-game forecasts**.

## Verification and limitations

- Final full NBA backend suite: **622 passed**, dependency deprecation warnings only. [Test output](nba_roster_review_20261010/backend_tests.txt).
- Regression coverage includes midnight/timezone boundaries, observation freshness, season mismatch, new arrivals, exits, transfers, stale recency, later appearances, missing positions, incomplete sources, append-only persistence, cold-host delivery union, retention, and old-team pool resurrection.
- Real production `_build_position_rapm_features` and `_write_player_rapm` interfaces are exercised end-to-end with deterministic roster evidence; pending game IDs, joined features, membership, CSV provenance and output name are checked.
- Python compilation and `git diff --check` pass. No configured/installed mypy or pyright exists; compilation is not static typechecking.
- **No production models, probabilities or calibration bundles are replaced.** Rebuild features, OOF, blend/calibration and final models together for v2.9 before production adoption. No predictive gain or AUC/log-loss/Brier improvement is claimed from roster reconciliation.
- Historical transaction publication times remain unavailable before this ledger began. Source state can lag a transaction, and seven-day expiry is a conservative policy, not an optimized threshold. Refresh daily/near game days. Same-day moves are withheld until the next calendar target under the date-level interface.
- Membership does not mean healthy, dressed, or rotation-ready. Official injury filings remain separate; rookies without NBA minutes cannot clear the evidence floor. Historical revised-position limitations remain in legacy game inputs.
- Unrelated concurrent MLB changes and other scratch artifacts are not included.
