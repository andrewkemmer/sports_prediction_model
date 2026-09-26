"""
Explicit rolling-retention policy for MLB ``data_delivery`` dated artifacts.

*** Policy (2026-09-14): blanket 10-day window, one anchor ***

Every DATED, deletion-allowlisted artifact family keeps the run's anchor date
and the 10 days before it (anchor = ``MLB_END_DATE`` when the run sets it,
else today in America/New_York). Anything older is pruned via ``git rm`` in
the daily auto-commit (a forward commit — git history keeps every blob).
Files NEWER than the anchor are never touched (backfill-safe), and the scope
is exactly ``mlb-backend/data_delivery/``.

*** Deliberate convention reversal (documented, not silent) ***

The long-standing repo rule was "committed artifacts are never auto-deleted".
This module formalizes the **deliberate reversal of that rule for the
ALLOWLISTED dated board-artifact families only** (``allowlisted=True``).
EVERYTHING ELSE — research/verdict records, undated masters, and families that
READ A SERIES across dated files — is ``never_delete`` and exempt.  See the
audit records ``data_delivery/mlb_retention_policy_*.json`` for the consumer
classification tables.

*** Why the exempt classes survive a blanket "no exemptions" rule ***
(pressure-tested 2026-09-14 against backend + frontend workflows)

- Dateless MASTERS (parquets, game_level_features.csv, umpire_*, roof cache,
  model histories): they have NO date to age, and the daily pipeline CONSUMES
  several as inputs — deleting them bricks the next run (documented regression
  42ef3f7: lineups.parquet deletion failed the pipeline loud) and strips the
  dashboards' final scores + the RFE engine's training frame.
- Adopted-model STATE (mlb_feature_selection_state.json, an ``mlb_*``
  record): holds the adopted RFE serving width. Silent deletion would
  silently revert model behavior — the worst failure mode.
- SERIES readers (run_engine_monitor_*, pbp_defense_*, pbp_chunks/,
  models/): producers/readers fold ALL dated members; pruning any member
  resets rolling history or orphans cumulative stores.
- test_hygiene_triage_* records: audit trail.
- mlb_feature_selection_state.json (the ONLY exempt ``mlb_`` file): holds
  the adopted RFE serving width, is dateless, and is read by EVERY serving
  run — silent deletion would silently revert model behavior.

*** Revision (2026-09-14 evening): mlb_ records de-exempted ***
All other ``mlb_*`` files now ride the same rules as every other family
(owner decision, pressure-tested against backend + frontend):
- ``mlb_feature_selection_<date>.json`` (RFE traces) and
  ``mlb_feature_workbook_<date>.xlsx`` — 10-day window. The RFE engine's
  cross-run prior-verdict memory reads the NEWEST prior trace: pruning old
  traces degrades gracefully to a fresh search, and adoption reads the
  STATE file (exempt above), never a trace.
- Sha-named research/audit records (mlb_binary_sp_projection*,
  mlb_retention_policy_*, ...) are dateless and DELETABLE — git history
  retains every blob.

Everything else in the folder is a dated, regenerable, per-run artifact: the
10-day window deletes it exactly as specified.

Consumer audit (traced at HEAD 827de1b):

  family                          | consumer(s)                                  | read pattern                        | policy
  --------------------------------|----------------------------------------------|-------------------------------------|---------------
  calibration_*.json              | model_calibration (_pick_artifact_date       | newest-only                         | 10-day window
                                  |   newest); available_dates daily[] from NEWEST|                                     |
  model_monitor_*.json            | model_monitor page (newest per date); embeds | newest-only                         | 10-day window
                                  |   drift/coverage/brier/metadata              |                                     |
  predictions_history_*.csv       | calibration/history pages (per date);        | newest-only + board-backed          | keep-while-board
                                  |   available_dates game_dates from NEWEST;    |                                     |
                                  |   rolling-brier recompute (in-run, newest)   |                                     |
  todays_games_*.csv              | board date navigator (loads per date)        | newest-only per navigable date      | 10-day window
                                  |                                              |                                     | (2026-09-23 rev 2: rolling only)  run_engine_markets_*.csv(+meta) | markets page (family-aware newest); board    | newest-only + board-backed          | keep-while-board
                                  | cards (market_diagnostics per date)          |                                     | (2026-09-23 rev 2: board-backed)
  run_engine_oof_*.csv            | no frontend reader; backend monitor rebuild/ | newest-only + board-backed          | keep-while-board
                                  |   harnesses                                 |                                     |
  run_engine_monitor_*.json       | markets page (newest per date); **producer   | **SERIES** (producer folds ALL      | NEVER DELETE
                                  |   folds ALL dated files into the rolling     |   dated monitors — pipeline.        |
                                  |   per-line series**                          |   _run_engine_monitor_json glob)    |
  rolling_brier_*.json            | never read standalone (embedded in the       | newest-only snapshot                | 10-day window
                                  |   model_monitor json)                        |                                     |
  run_engine_feature_drift_*.csv  | markets page drift table (per date)          | newest-only                         | 10-day window
  run_engine_feature_coverage_*.csv | markets page coverage (per date)           | newest-only                         | 10-day window
  feature_drift_*.csv             | never read standalone (embedded in monitor)  | newest-only                         | 10-day window
  feature_coverage_*.csv          | never read standalone                        | newest-only                         | 10-day window
  features_metadata_*.json        | never read standalone (embedded in monitor)  | newest-only                         | 10-day window
  shap_game_*.csv                 | board per-game card fetch (per date)         | newest-only per navigable date      | board-backed
  power_rankings_*.csv            | Home / power_rankings page (newest)          | newest-only                         | 10-day window
  pbp_defense_*.parquet(+meta)      | build_pbp_chunks._newest_source and   | newest-only          | blanket 10-day       |
                                    | build_il_stints.default_pbp_source, BOTH |                      | window               |
                                    | take sorted(...)[-1]; the two harnesses |                      |                      |
                                    | this row cited (ablation_defense,     |                      |                      |
                                    | runline defense) were deleted in ff372c3, |                      |                      |
                                    | so the NEVER DELETE outlived its consumers |                      |                      |
  pbp_chunks/                     | build_pbp_defense (cumulative raw chunks)    | **SERIES (cumulative)**             | NEVER DELETE
  models/                         | ensemble/monitor loaders (newest)            | newest-only; staged every run       | NEVER DELETE
  mlb_feature_selection_*.json    | RFE engine prior-verdict memory (newest      | newest prior trace                  | 10-day window
                                  | prior); adoption reads the STATE file only   |                                     |
  mlb_feature_workbook_*.xlsx     | human decision workbook; regenerated every   | newest-only                         | 10-day window
                                  | RFE run; nothing reads it back               |                                     |
  other mlb_* sha-named records   | none (docstring references only)             | none                                | DELETABLE (dateless -> stale)
  mlb_feature_selection_state.json| RFE serving gate (every run)                 | master-equivalent                   | NEVER DELETE
  *_triage_* records              | audit trail                                  | record                              | NEVER DELETE
  masters (game_level_features.csv, model_history.json, model_version_history.json,
           umpire_*.csv, lineups.parquet, batter_woba.parquet, team_woba.parquet,
           il_stints.parquet, statsapi_roof_cache.json) | multiple             | master                              | NEVER DELETE

Notes
-----
- The blanket window keeps anchor .. anchor-10 (11 calendar days) — anchor
  = ``MLB_END_DATE`` when set, else today ET (master_pipeline Phase 6).
- Board-backed families survive as long as a ``todays_games_<date>.csv`` board
  for that date is still tracked (the 2026-08-29 doubleheader regression fix);
  at the 10-day window the slate rule dominates, kept as a safety net.
- Rolling 10-day boards (2026-09-23 revision 2, owner decision): the
  2026-09-23 revision made ``todays_games_`` boards, ``run_engine_markets_``
  (+ .meta.json), and ``shap_game_`` permanent so a historical card could
  render the production prices AS PUBLISHED that day (the 2026-09-11 card
  regression). That permanence is REVERSED: every dated board family now
  rides the blanket 10-day window — the Today's Games dashboard shows a
  ROLLING 10 DAYS of predictions, never a historical archive. The
  price-honesty invariant is preserved for every date still inside the
  window: board + markets + SHAP age out TOGETHER (SHAP and markets are
  board-backed), so a card never falls back to an OOF-rebuilt board or
  cross-date price binding while its date is served. Beyond the window the
  date is not offered anywhere (frontend valid-date filter, same anchor),
  and git history retains every pruned blob. Board-backed companions
  (``run_engine_oof_``, ``predictions_history_``) follow their boards.
- Files dated NEWER than the anchor are never deleted (backfill runs set
  ``MLB_END_DATE`` in the past; present-day artifacts must survive it).
- Run-dated harness OUTPUTS (``*_ablation_*.json``, ``calibration_ablation_*``,
  ``calibration_flip_*``, ...) intentionally keep riding the date gate (they
  are regenerable run outputs, not decision records).
  ``mlb_*`` decision records ride the 10-day window (2026-09-14 revision):
  dated records keep 10 days; dateless sha-named records are pruned with
  git history retaining every blob. Only the state file is exempt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# -- Masters (exact-name): dateless cumulative / maintained files the daily --
# -- pipeline only consumes or that must survive every window.              --
EXACT_MASTER_NAMES = frozenset({
    "statsapi_roof_cache.json",
    "model_history.json",
    "model_version_history.json",
    # Lineup-delta feature runtime inputs (Phase 2, aead200): the daily
    # pipeline CONSUMES these; only the standalone builders regenerate them.
    # Dateless names -> the date-gate can never save them (42ef3f7 deleted
    # them; the pipeline failed loud without them).
    "lineups.parquet",
    "batter_woba.parquet",
    "team_woba.parquet",
    # Injured-list stint table (expected-lineup features.py lineup_agg): the
    # daily pipeline CONSUMES it and only build_il_stints.py regenerates it.
    # Dateless name, so without this the date-gate classifies it stale on the
    # very next run and git rms it — which does not fail loudly, it silently
    # reverts every expected-lineup feature to the unfiltered pool.
    "il_stints.parquet",
    "il_stints.meta.json",
    # Maintained umpire data access (umpires.py): cumulative map + per-umpire
    # diagnostics table updated IN PLACE every run.
    "umpire_map.csv",
    "umpire_stats.csv",
    # Dashboard reads game_level_features.csv for final scores; regenerated
    # and staged every run (seen-protected anyway) — name-protect as a master.
    "game_level_features.csv",
})

# -- Series readers / cumulative stores (prefix): deleting ANY member would --
# -- reset history, break research ladders, or orphan chunks.               --
SERIES_PREFIXES = (
    "models/",
    "pbp_chunks/",
    # Producer folds ALL dated monitors into the rolling per-line series
    # (pipeline._run_engine_monitor_json). Never reset the monitor history.
    "run_engine_monitor_",
    # NOT pbp_defense_. It used to sit here citing two ablation harnesses
    # (ablation_defense.py, run_mlb_runline_defense_ablation.py) that were
    # deleted in ff372c3, so the exemption protected nothing. Both live
    # consumers — build_pbp_chunks._newest_source and
    # build_il_stints.default_pbp_source — read sorted(glob(...))[-1], i.e.
    # the newest file only. Leaving it protected cost 13.1 MB of pitch
    # projection per run, forever: 52 files / 333 MB on 2026-09-26, and
    # every Kaggle run clones the whole set before it can start.
)

# -- Triage records (prefix): audit trail — never deleted.                  --
TRIAGE_RECORD_PREFIX = "test_hygiene_triage"

# -- MLB decision records ("mlb_" prefix): NOT blanket-exempt (2026-09-14  --
# -- revision). Dated members (RFE traces, feature workbooks) ride the     --
# -- 10-day window as FAMILY_POLICY families; dateless sha-named           --
# -- research/audit records are deletable (git history retains them).      --
# -- The single exempt ``mlb_`` file is the adopted-model STATE below.     --
MLB_RECORD_PREFIX = "mlb_"

MLB_EXEMPT_NAMES = frozenset({
    # Adopted RFE serving width — written on explicit --adopt only, read by
    # every run. Deleting it would silently revert the model to full-
    # universe width (the worst failure mode: no error, wrong behavior).
    "mlb_feature_selection_state.json",
})


# NOTE on the pbp_defense family: it is the only family whose members are
# ~13 MB rather than kilobytes (the daily 27-column projection of
# pitches.parquet), so the blanket 10-day window pins ~130 MB of parquet in
# the tree that every Kaggle run clones before it starts. It is bounded now
# (2026-09-26 fix: the NEVER DELETE exemption outlived its consumers), but
# tightening it below the blanket window needs a real per-family window in
# classify_artifact — ``retention_days`` on FamilyPolicy is NOT consulted
# there, so this note is the only place that fact is recorded.


@dataclass(frozen=True)
class FamilyPolicy:
    """One dated family's retention rule (the single config table)."""

    family: str
    prefix: str
    retention_days: Optional[int]   # prior days kept beyond the run date;
                                    # None -> governed by board/slate/exempt
    allowlisted: bool               # True -> eligible for git rm outside window
    board_supported: bool = False   # keep while a todays_games_<date>.csv board
                                    # for the date is still tracked
    slate_window_days: int = 0      # keep for the recent-slate settle window
                                    # (run date .. run date - N)
    permanent: bool = False         # never pruned (historical game cards must
                                    # serve production-as-published prices)
    notes: str = ""


# Order is significant ONLY for documentation; family match is by longest
# prefix (see _family_for). Every family in the audit is classified here.
FAMILY_POLICY: tuple[FamilyPolicy, ...] = (
    FamilyPolicy("calibration", "calibration_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (model_calibration _pick_artifact_date; "
                       "available_dates daily[] from NEWEST calibration); "
                       "10-day blanket window"),
    FamilyPolicy("model_monitor", "model_monitor_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (monitor page; embeds drift/coverage/"
                       "brier/features_metadata); 10-day blanket window"),
    FamilyPolicy("predictions_history", "predictions_history_",
                 retention_days=None, allowlisted=True, board_supported=True,
                 notes="newest-only + board-backed (available_dates game_dates "
                       "from NEWEST; rolling-brier recompute in-run)"),
    FamilyPolicy("todays_games", "todays_games_", retention_days=10,
                 allowlisted=True,
                 notes="board date-navigator loads per date; 10-day blanket "
                       "window (2026-09-23 rev 2: rolling boards — the "
                       "dashboard serves a rolling 10 days of predictions)"),
    FamilyPolicy("run_engine_markets", "run_engine_markets_",
                 retention_days=None, allowlisted=True, board_supported=True,
                 notes="markets page family-aware pick; board cards per date "
                       "(incl. .meta.json); board-backed — kept while its "
                       "todays_games board is tracked (2026-09-23 rev 2: "
                       "ages out with the 10-day board window)"),
    FamilyPolicy("run_engine_oof", "run_engine_oof_", retention_days=None,
                 allowlisted=True, board_supported=True,
                 notes="no frontend reader; newest-only + board-backed"),
    FamilyPolicy("run_engine_monitor", "run_engine_monitor_",
                 retention_days=None, allowlisted=False,
                 notes="SERIES — producer folds ALL dated monitors "
                       "(pipeline._run_engine_monitor_json)"),
    FamilyPolicy("rolling_brier", "rolling_brier_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only snapshot; recomputed each run, never read "
                       "standalone (embedded in model_monitor); 10-day window"),
    FamilyPolicy("run_engine_feature_drift", "run_engine_feature_drift_",
                 retention_days=10, allowlisted=True,
                 notes="newest-only (markets page drift table per date); "
                       "10-day window"),
    FamilyPolicy("run_engine_feature_coverage", "run_engine_feature_coverage_",
                 retention_days=10, allowlisted=True,
                 notes="newest-only (markets page coverage per date); "
                       "10-day window"),
    FamilyPolicy("feature_drift", "feature_drift_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only; never read standalone (embedded in "
                       "model_monitor); 10-day window"),
    FamilyPolicy("feature_coverage", "feature_coverage_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only; never read standalone (embedded in "
                       "model_monitor); 10-day window"),
    FamilyPolicy("features_metadata", "features_metadata_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only; never read standalone (embedded in "
                       "model_monitor); 10-day window"),
    FamilyPolicy("shap_game", "shap_game_", retention_days=None,
                 allowlisted=True, board_supported=True,
                 notes="board per-game card fetch; board-backed — kept while "
                       "its todays_games board is tracked (2026-09-23 rev 2: "
                       "ages out with the 10-day board window)"),
    FamilyPolicy("power_rankings", "power_rankings_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (Home / power_rankings "
                       "load_power_rankings); 10-day window"),
    FamilyPolicy("mlb_feature_selection", "mlb_feature_selection_",
                 retention_days=10, allowlisted=True,
                 notes="RFE trace record; 10-day window (2026-09-14 "
                       "revision) — prior-verdict memory degrades to a "
                       "fresh search when pruned; adoption reads the "
                       "exempt STATE file"),
    FamilyPolicy("mlb_feature_workbook", "mlb_feature_workbook_",
                 retention_days=10, allowlisted=True,
                 notes="human-readable RFE workbook; regenerated every RFE "
                       "run, nothing reads it back — 10-day window"),
    FamilyPolicy("pbp_defense", "pbp_defense_", retention_days=None,
                 allowlisted=False,
                 notes="newest-only (traced 2026-09-26): both consumers take "
                       "sorted(glob(...))[-1], so it rides the blanket "
                       "10-day window instead of growing without bound — "
                       "52 files / 333 MB when the stale exemption was found"),
)

# -- Predicates ---------------------------------------------------------------

_COMPACT_DATE_RE = re.compile(r"_(\d{8})")          # todays_games_20260914.csv
_ISO_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})")  # mlb_feature_selection_2026-09-14.json


def local_name(rel: str) -> str:
    """Path relative to ``data_delivery/`` (matches the old Phase-6 helper)."""
    _DD = "data_delivery/"
    idx = rel.find(_DD)
    return rel[idx + len(_DD):] if idx >= 0 else rel


def artifact_date(rel: str) -> Optional[str]:
    """Extract the artifact date as compact YYYYMMDD, or None if dateless.

    Understands both the compact ``_YYYYMMDD`` convention (board families)
    and the ISO ``_YYYY-MM-DD`` convention (``mlb_`` decision records), so
    dated ``mlb_`` records ride the retention windows on the same footing
    as every other family.
    """
    m = _COMPACT_DATE_RE.search(rel) or _ISO_DATE_RE.search(rel)
    return m.group(1).replace("-", "") if m else None


def is_never_delete(rel: str) -> bool:
    """True if ``rel`` is a master / record / series member cleanup never
    touches (regardless of any date window)."""
    local = local_name(rel)
    base = local.rsplit("/", 1)[-1]
    if base in EXACT_MASTER_NAMES or base in MLB_EXEMPT_NAMES:
        return True
    if local.startswith(TRIAGE_RECORD_PREFIX):
        return True
    if any(local.startswith(p) for p in SERIES_PREFIXES):
        return True
    # mlb_* records: only the STATE file above is exempt — dated members
    # ride the family windows; dateless sha-named records are deletable.
    return False


def _family_for(rel: str) -> Optional[FamilyPolicy]:
    """Longest-prefix family match for a path's basename."""
    base = local_name(rel).rsplit("/", 1)[-1]
    best = None
    for fp in FAMILY_POLICY:
        if base.startswith(fp.prefix) and (best is None
                                           or len(fp.prefix) > len(best.prefix)):
            best = fp
    return best


def family_prefixes(attr: str) -> tuple[str, ...]:
    """All family prefixes where ``attr`` is truthy (e.g. board_supported)."""
    return tuple(fp.prefix for fp in FAMILY_POLICY if getattr(fp, attr))


def is_allowlisted(rel: str) -> bool:
    """True if the path's family is in the deletion allowlist (only
    allowlisted families may ever be selected by the keep-set computation)."""
    fam = _family_for(rel)
    return fam is not None and fam.allowlisted


def classify_artifact(rel: str, seen: set,
                      retention_dates: set, recent_dates: set,
                      board_dates: set,
                      anchor_date: Optional[str] = None) -> str:
    """Pure keep/stale decision for one tracked artifact path.

    Returns one of:
      "seen"      - staged by this run (kept; never counted)
      "protected" - never-delete (master / record / series reader)
      "current"   - kept via a family window (blanket 10-day retention,
                    recent-slate window, or board-backed
                    run-engine/predictions) — or NEWER than the anchor
      "stale"     - safe to delete (git rm) under the policy

    ``anchor_date`` (YYYYMMDD, optional): the run's retention anchor. Any
    artifact dated AFTER the anchor is kept regardless of windows — a
    backfill run (``MLB_END_DATE`` in the past) must never prune artifacts
    newer than its historical anchor. YYYYMMDD strings compare correctly.

    Pure (no I/O) so it is unit-testable in isolation — master_pipeline's
    live Phase 6 loop only calls this predicate.
    """
    if rel in seen:
        return "seen"
    if is_never_delete(rel):
        return "protected"
    fam = _family_for(rel)
    art_date = artifact_date(rel)
    if art_date is None:
        # Dateless and not never-delete -> stale (no window can save it).
        return "stale"
    if fam is not None and fam.permanent:
        return "current"  # permanent family — historical cards need it
    if anchor_date and art_date > anchor_date:
        return "current"  # newer than the run's anchor — backfill-safe keep
    if art_date in retention_dates:
        return "current"  # within the blanket retention window — keep
    if fam is not None and fam.slate_window_days and art_date in recent_dates:
        return "current"  # recent slate snapshot — keep
    if fam is not None and fam.board_supported and art_date in board_dates:
        return "current"  # board still tracked -> keep its run-engine data
    return "stale"