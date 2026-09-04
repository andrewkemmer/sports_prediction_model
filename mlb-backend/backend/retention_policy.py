"""
Explicit rolling-retention policy for MLB ``data_delivery`` dated artifacts.

*** Deliberate convention reversal (documented, not silent) ***

The long-standing repo rule was "committed artifacts are never auto-deleted".
This module formalizes the **deliberate reversal of that rule for the
ALLOWLISTED dated board-artifact families only** (``allowlisted=True``): those
families' committed files are pruned via ``git rm`` inside the daily
auto-commit once they fall outside their family window (a forward commit —
history keeps the blobs).  EVERYTHING ELSE — research/verdict records, undated
masters, and families that READ A SERIES across dated files — is
``never_delete`` and exempt.  See the audit record
``data_delivery/mlb_retention_policy_<framesha>.json`` for the full consumer
classification table.

Consumer audit (traced at HEAD 827de1b):

  family                          | consumer(s)                                  | read pattern                        | policy
  --------------------------------|----------------------------------------------|-------------------------------------|---------------
  calibration_*.json              | model_calibration (_pick_artifact_date       | newest-only                         | 2-day window
                                  |   newest); available_dates daily[] from NEWEST|                                     |
  model_monitor_*.json            | model_monitor page (newest per date); embeds | newest-only                         | 2-day window
                                  |   drift/coverage/brier/metadata              |                                     |
  predictions_history_*.csv       | calibration/history pages (per date);        | newest-only + board-backed          | keep-while-board
                                  |   available_dates game_dates from NEWEST;    |                                     |
                                  |   rolling-brier recompute (in-run, newest)   |                                     |
  todays_games_*.csv              | board date navigator (loads per date)        | newest-only per navigable date      | 3-day slate
  run_engine_markets_*.csv(+meta) | markets page (family-aware newest); board    | newest-only + board-backed          | keep-while-board
                                  |   cards (market_diagnostics per date)        |                                     |
  run_engine_oof_*.csv            | no frontend reader; backend monitor rebuild/ | newest-only + board-backed          | keep-while-board
                                  |   harnesses                                 |                                     |
  run_engine_monitor_*.json       | markets page (newest per date); **producer   | **SERIES** (producer folds ALL      | NEVER DELETE
                                  |   folds ALL dated files into the rolling     |   dated monitors — pipeline.        |
                                  |   per-line series**                          |   _run_engine_monitor_json glob)    |
  rolling_brier_*.json            | never read standalone (embedded in the       | newest-only snapshot                | 2-day window
                                  |   model_monitor json)                        |                                     |
  run_engine_feature_drift_*.csv  | markets page drift table (per date)          | newest-only                         | 2-day window
  run_engine_feature_coverage_*.csv | markets page coverage (per date)           | newest-only                         | 2-day window
  feature_drift_*.csv             | never read standalone (embedded in monitor)  | newest-only                         | 2-day window
  feature_coverage_*.csv          | never read standalone                        | newest-only                         | 2-day window
  features_metadata_*.json        | never read standalone (embedded in monitor)  | newest-only                         | 2-day window
  shap_game_*.csv                 | board per-game card fetch (per date)         | newest-only per navigable date      | 3-day slate
  power_rankings_*.csv            | Home / power_rankings page (newest)          | newest-only                         | 2-day window
  pbp_defense_*.parquet(+meta)    | defense ablation harnesses GLOB ALL dated    | **SERIES (research)**               | NEVER DELETE
                                  |   files (ablation_defense / runline defense) |                                     |
  pbp_chunks/                     | build_pbp_defense (cumulative raw chunks)    | **SERIES (cumulative)**             | NEVER DELETE
  models/                         | ensemble/monitor loaders (newest)            | newest-only; staged every run       | NEVER DELETE
  mlb_* records (sha-named)       | audit trail                                  | record                              | NEVER DELETE
  *_triage_* records              | audit trail                                  | record                              | NEVER DELETE
  masters (game_level_features.csv, model_history.json, model_version_history.json,
           umpire_*.csv, lineups.parquet, batter_woba.parquet, team_woba.parquet,
           statsapi_roof_cache.json) | multiple                                | master                              | NEVER DELETE

Notes
-----
- ``retention_days=1`` means the run date + the previous day are kept — the
  repo's rolling 48h GMT-rollover window, **not** a count-of-files rule.
- Board-backed families survive as long as a ``todays_games_<date>.csv`` board
  for that date is still tracked (the 2026-08-29 doubleheader regression fix).
- The 3-day slate window is the recent-slate settle window for the board's
  own snapshots (``_RECENT_DATES`` = today .. day-2).
- Run-dated harness OUTPUTS (``*_ablation_*.json``, ``calibration_ablation_*``,
  ``calibration_flip_*``, ...) intentionally keep riding the date gate (their
  tests pin that; they are regenerable run outputs, not decision records).
  New decision/diagnostic records should use the ``mlb_*`` or ``*_triage_*``
  naming to inherit never-delete protection.
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
    # Defense ablation harnesses glob ALL dated caches
    # (ablation_defense.py / run_mlb_runline_defense_ablation.py). Research
    # series — exempt per guardrail 1 even though nothing in production reads
    # across dates.
    "pbp_defense_",
)

# -- Research/verdict records (prefix): audit trail — never deleted, even   --
# -- though several are date-stamped and would otherwise ride a window.     --
RECORD_PREFIXES = (
    "mlb_",
    "test_hygiene_triage",
)


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
    notes: str = ""


# Order is significant ONLY for documentation; family match is by longest
# prefix (see _family_for). Every family in the audit is classified here.
FAMILY_POLICY: tuple[FamilyPolicy, ...] = (
    FamilyPolicy("calibration", "calibration_", retention_days=1,
                 allowlisted=True,
                 notes="newest-only (model_calibration _pick_artifact_date; "
                       "available_dates daily[] from NEWEST calibration)"),
    FamilyPolicy("model_monitor", "model_monitor_", retention_days=1,
                 allowlisted=True,
                 notes="newest-only (monitor page; embeds drift/coverage/"
                       "brier/features_metadata)"),
    FamilyPolicy("predictions_history", "predictions_history_",
                 retention_days=None, allowlisted=True, board_supported=True,
                 notes="newest-only + board-backed (available_dates game_dates "
                       "from NEWEST; rolling-brier recompute in-run)"),
    FamilyPolicy("todays_games", "todays_games_", retention_days=None,
                 allowlisted=True, slate_window_days=3,
                 notes="board date-navigator loads per date (3-day settle)"),
    FamilyPolicy("run_engine_markets", "run_engine_markets_",
                 retention_days=None, allowlisted=True, board_supported=True,
                 notes="newest-only + board-backed (markets page family-aware "
                       "pick; board cards per date; incl. .meta.json)"),
    FamilyPolicy("run_engine_oof", "run_engine_oof_", retention_days=None,
                 allowlisted=True, board_supported=True,
                 notes="no frontend reader; newest-only + board-backed"),
    FamilyPolicy("run_engine_monitor", "run_engine_monitor_",
                 retention_days=None, allowlisted=False,
                 notes="SERIES — producer folds ALL dated monitors "
                       "(pipeline._run_engine_monitor_json)"),
    FamilyPolicy("rolling_brier", "rolling_brier_", retention_days=1,
                 allowlisted=True,
                 notes="newest-only snapshot; recomputed each run, never read "
                       "standalone (embedded in model_monitor)"),
    FamilyPolicy("run_engine_feature_drift", "run_engine_feature_drift_",
                 retention_days=1, allowlisted=True,
                 notes="newest-only (markets page drift table per date)"),
    FamilyPolicy("run_engine_feature_coverage", "run_engine_feature_coverage_",
                 retention_days=1, allowlisted=True,
                 notes="newest-only (markets page coverage per date)"),
    FamilyPolicy("feature_drift", "feature_drift_", retention_days=1,
                 allowlisted=True,
                 notes="newest-only; never read standalone (embedded in "
                       "model_monitor)"),
    FamilyPolicy("feature_coverage", "feature_coverage_", retention_days=1,
                 allowlisted=True,
                 notes="newest-only; never read standalone (embedded in "
                       "model_monitor)"),
    FamilyPolicy("features_metadata", "features_metadata_", retention_days=1,
                 allowlisted=True,
                 notes="newest-only; never read standalone (embedded in "
                       "model_monitor)"),
    FamilyPolicy("shap_game", "shap_game_", retention_days=None,
                 allowlisted=True, slate_window_days=3,
                 notes="board per-game card fetch; newest-only per navigable "
                       "date"),
    FamilyPolicy("power_rankings", "power_rankings_", retention_days=1,
                 allowlisted=True,
                 notes="newest-only (Home / power_rankings load_power_rankings)"),
    FamilyPolicy("pbp_defense", "pbp_defense_", retention_days=None,
                 allowlisted=False,
                 notes="SERIES (research) — defense ablation harnesses glob "
                       "ALL dated caches"),
)

# -- Predicates ---------------------------------------------------------------

_DATE_RE = re.compile(r"_(\d{8})")


def local_name(rel: str) -> str:
    """Path relative to ``data_delivery/`` (matches the old Phase-6 helper)."""
    _DD = "data_delivery/"
    idx = rel.find(_DD)
    return rel[idx + len(_DD):] if idx >= 0 else rel


def artifact_date(rel: str) -> Optional[str]:
    """Extract the YYYYMMDD date from an artifact path, or None if dateless."""
    m = _DATE_RE.search(rel)
    return m.group(1) if m else None


def is_never_delete(rel: str) -> bool:
    """True if ``rel`` is a master / record / series member cleanup never
    touches (regardless of any date window)."""
    local = local_name(rel)
    base = local.rsplit("/", 1)[-1]
    return (base in EXACT_MASTER_NAMES
            or any(local.startswith(p)
                   for p in (*SERIES_PREFIXES, *RECORD_PREFIXES)))


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
                      board_dates: set) -> str:
    """Pure keep/stale decision for one tracked artifact path.

    Returns one of:
      "seen"      - staged by this run (kept; never counted)
      "protected" - never-delete (master / record / series reader)
      "current"   - kept via a family window (48h retention, recent-slate
                    3-day window, or board-backed run-engine/predictions)
      "stale"     - safe to delete (git rm) under the policy

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
    if art_date in retention_dates:
        return "current"  # within the rolling 48h retention window — keep
    if fam is not None and fam.slate_window_days and art_date in recent_dates:
        return "current"  # recent slate snapshot — keep
    if fam is not None and fam.board_supported and art_date in board_dates:
        return "current"  # board still tracked -> keep its run-engine data
    return "stale"