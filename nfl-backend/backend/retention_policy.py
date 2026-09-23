"""Explicit rolling-retention policy for NFL ``data_delivery`` dated artifacts.

*** Policy: blanket 10-day window, one anchor (MLB parity, 2026-09-23) ***

Every DATED, deletion-allowlisted artifact family keeps the run's anchor date
and the 10 days before it (anchor = ``NFL_END_DATE`` when the run sets it,
else today in America/New_York). Anything older is stale: the pipeline's
artifact sync stages the deletions, so git history retains every blob (MLB
parity: the "Remove stale data_delivery artifacts" forward-commit pattern).
Files NEWER than the anchor are never touched (backfill-safe), and the scope
is exactly ``nfl-backend/data_delivery/``.

This mirrors ``mlb-backend/backend/retention_policy.py`` structurally: one
``FamilyPolicy`` config table, one pure ``classify_artifact`` predicate, and
the same never-delete classes. The NFL-relevant differences are documented
here, from the consumer audit (traced 2026-09-23):

- ``nfl_run_engine_monitor_<date>.json`` is a SERIES: the markets page folds
  ALL dated monitors into the rolling per-card history
  (``nfl_market_diagnostics.fold_slate_history``). NEVER DELETE — pruning any
  member resets that rolling history (MLB parity: ``run_engine_monitor_``).
- ``nfl_shap_game_<game_id>.csv`` IS pruned, exactly like MLB (the deletion
  commits e646aa8 / 9819330 / 5bd7c9a show SHAP aging out with the other
  dated families). The wrinkle: MLB game ids embed the game date
  (``shap_game_20260821_STL@PHI.csv``) so aging is filename-keyed, while NFL
  ids embed season+week (``nfl_shap_game_2026_01_ARI_LAC.csv``). NFL SHAP
  files are therefore aged through a game_id -> game_date map built from the
  frozen card store / prediction history (``game_dates`` argument of
  ``classify_artifact``): a game 10+ days past its date is pruned; current
  and future slate games keep their files (the pre-generated future-week
  files survive until 10 days after each game is played — the live slate
  never loses SHAP, so no dashboard degradation).
- ``nfl_production_cards_history.csv`` (the frozen first-publication card
  store) and the other masters below are dateless cumulative state — NEVER
  DELETE (MLB parity: model histories / masters).
- Everything else dated (moneyline, calibration, monitor, markets, history,
  power rankings, QB, feature dict, RFE trace/workbook, drift/coverage
  tables) is a newest-only consumer artifact riding the 10-day blanket
  window; historical boards rebuild from the frozen card store, so pruning
  them cannot degrade navigation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# -- Masters (exact-name): dateless cumulative / maintained files cleanup   --
# -- never touches.                                                        --
EXACT_MASTER_NAMES = frozenset({
    # The frozen first-publication card store (append-only, seeded once,
    # mutated only by appends) — historical cards serve from it.
    "nfl_production_cards_history.csv",
    "nfl_production_cards_history.meta.json",
    # Adopted RFE serving width — written on explicit adopt only, read by
    # every run (MLB parity: deleting it would silently revert the model).
    "nfl_feature_selection_state.json",
    # Run internals / maintained tables (mostly git-ignored; name-protecting
    # them is free and keeps the local prune honest about the tracked set).
    "nfl_fold_table.csv",
    "nfl_pipeline_summary.json",
    "nfl_oof_store.csv",
    "nfl_oof_moneyline.csv",
    "nfl_oof_distribution.csv",
    "nfl_decided_store_rs_2018_2025.csv",
})

# -- Series readers / cumulative stores (prefix): deleting ANY member would --
# -- reset rolling history.                                                --
SERIES_PREFIXES = (
    "models/",
    # The markets page folds ALL dated monitors into the rolling per-card
    # series (nfl_market_diagnostics.fold_slate_history). Never reset it.
    "nfl_run_engine_monitor_",
    # Per-game SHAP cards are family-managed below (aged via the game-date
    # map, pruned exactly like MLB) — they are NOT blanket-exempt.
)


@dataclass(frozen=True)
class FamilyPolicy:
    """One dated family's retention rule (the single config table)."""

    family: str
    prefix: str
    retention_days: Optional[int]   # prior days kept beyond the run date;
                                    # None -> governed by board/slate/exempt
    allowlisted: bool               # True -> eligible for deletion
    board_supported: bool = False   # keep while a board for that date is tracked
    slate_window_days: int = 0      # keep for the recent-slate settle window
    notes: str = ""


# Longest-prefix match (see _family_for). Every NFL artifact family appears.
FAMILY_POLICY: tuple[FamilyPolicy, ...] = (
    FamilyPolicy("moneyline", "nfl_moneyline_v1_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (board moneyline JSON; valid_dates "
                       "games[] from NEWEST); 10-day blanket window"),
    FamilyPolicy("calibration", "nfl_calibration_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (Calibration page _pick_artifact_date); "
                       "10-day blanket window"),
    FamilyPolicy("predictions_history", "nfl_predictions_history_",
                 retention_days=10, allowlisted=True, board_supported=True,
                 notes="newest-only consumers (calibration curve/table, "
                       "store fallback); 10-day blanket window"),
    FamilyPolicy("model_monitor", "nfl_model_monitor_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (monitor page); 10-day blanket window"),
    FamilyPolicy("feature_v1", "nfl_feature_v1_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only feature dictionary; 10-day window"),
    FamilyPolicy("power_rankings", "nfl_power_rankings_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (shared rankings page); 10-day window"),
    FamilyPolicy("markets", "nfl_run_engine_markets_",
                 retention_days=10, allowlisted=True, board_supported=True,
                 notes="newest-only (markets page + board enrichment, incl. "
                       ".meta.json); 10-day blanket window"),
    FamilyPolicy("markets_monitor", "nfl_run_engine_monitor_",
                 retention_days=None, allowlisted=False,
                 notes="SERIES - fold_slate_history folds ALL dated monitors"),
    FamilyPolicy("qb_matchup", "nfl_qb_matchup_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (board QB boxes); 10-day window"),
    FamilyPolicy("run_engine_feature_drift", "run_engine_feature_drift_",
                 retention_days=10, allowlisted=True,
                 notes="newest-only drift table; 10-day window"),
    FamilyPolicy("run_engine_feature_coverage", "run_engine_feature_coverage_",
                 retention_days=10, allowlisted=True,
                 notes="newest-only coverage table; 10-day window"),
    FamilyPolicy("shap_game", "nfl_shap_game_",
                 retention_days=None, allowlisted=True, slate_window_days=10,
                 notes="per-game cards aged via the game_id -> game_date map "
                       "(NFL ids embed season+week, not dates); pruned after "
                       "10 days past the game like MLB; current/future slate "
                       "kept"),
    FamilyPolicy("rfe_trace", "nfl_feature_selection_", retention_days=10,
                 allowlisted=True,
                 notes="RFE trace record; prior-verdict memory reads the "
                       "NEWEST prior trace and degrades to a fresh search "
                       "when pruned; adoption reads the STATE file (exempt)"),
    FamilyPolicy("rfe_workbook", "nfl_feature_workbook_", retention_days=10,
                 allowlisted=True,
                 notes="human decision workbook; regenerated per RFE run; "
                       "10-day window"),
)

# -- Predicates --------------------------------------------------------------

_COMPACT_DATE_RE = re.compile(r"_(\d{8})")            # nfl_calibration_20260922.json
_ISO_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})")    # nfl_feature_workbook_2026-09-22.xlsx
_SHAP_DATE_RE = re.compile(r"nfl_shap_game_(\d{4})_(\d{2})_(\d{2})_")


def local_name(rel: str) -> str:
    """Path relative to ``data_delivery/`` (matches the MLB helper).

    Windows separators are normalized to POSIX first so basename/prefix
    matching is OS-independent (production runs on Linux; local runs may
    not — a backslash path must classify identically).
    """
    rel = rel.replace("\\", "/")
    dd = "data_delivery/"
    idx = rel.find(dd)
    return rel[idx + len(dd):] if idx >= 0 else rel


def artifact_date(rel: str) -> Optional[str]:
    """Extract the artifact date as compact YYYYMMDD, or None if dateless.

    Understands the compact ``_YYYYMMDD`` convention (board families) and
    the ISO ``_YYYY-MM-DD`` convention (RFE workbooks). SHAP files use the
    ``_YYYY_MM_DD`` underscored game-id convention, which neither regex
    matches by design: their age resolves through ``shap_game_date``.
    """
    m = _COMPACT_DATE_RE.search(rel) or _ISO_DATE_RE.search(rel)
    return m.group(1).replace("-", "") if m else None


def shap_game_date(rel: str) -> Optional[str]:
    """The YYYYMMDD embedded in an ``nfl_shap_game_YYYY_MM_DD_<...>.csv`` id.

    Legacy one-time-pregeneration ids use underscores between the date
    parts (season_week_team_team ids carry no date at all and stay None —
    those resolve through the game-date map instead, never guessed).
    """
    m = _SHAP_DATE_RE.search(local_name(rel))
    return "".join(m.groups()) if m else None


def is_never_delete(rel: str) -> bool:
    """True for masters / series members cleanup never touches.

    Prefix matching is segment-aware: it fires wherever the remaining path
    (from any segment onward) starts with a series prefix, so both the
    production layout (``data_delivery/models/...``) and bare or test
    layouts (``tmp/models/...``) classify identically.
    """
    local = local_name(rel)
    base = local.rsplit("/", 1)[-1]
    if base in EXACT_MASTER_NAMES:
        return True
    parts = [p for p in local.split("/") if p]
    for i in range(len(parts)):
        candidate = "/".join(parts[i:])
        if any(candidate.startswith(p) for p in SERIES_PREFIXES):
            return True
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
    """All family prefixes where ``attr`` is truthy."""
    return tuple(fp.prefix for fp in FAMILY_POLICY if getattr(fp, attr))


def is_allowlisted(rel: str) -> bool:
    """True when the path's family may ever be selected for deletion."""
    fam = _family_for(rel)
    return fam is not None and fam.allowlisted


def classify_artifact(rel: str, seen: set,
                      retention_dates: set, recent_dates: set,
                      board_dates: set,
                      anchor_date: Optional[str] = None,
                      game_dates: Optional[dict] = None) -> str:
    """Pure keep/stale decision for one tracked artifact path.

    Returns one of:
      "seen"      - staged by this run (kept; never counted)
      "protected" - never-delete (master / series reader / unresolvable SHAP)
      "current"   - kept via a family window (blanket 10-day retention,
                    recent slate, board-backed) - or NEWER than the anchor
      "stale"     - safe to delete under the policy

    ``anchor_date`` (YYYYMMDD): artifacts dated AFTER the anchor are always
    kept (backfill-safe; YYYYMMDD strings compare correctly).
    ``game_dates`` (game_id -> YYYYMMDD) resolves SHAP game ids whose ids
    embed season+week rather than a date; an unresolvable SHAP id is kept
    (protected) rather than guessed.

    Pure (no I/O) so it is unit-testable in isolation - the pipeline's live
    prune loop only calls this predicate.
    """
    if rel in seen:
        return "seen"
    if is_never_delete(rel):
        return "protected"
    fam = _family_for(rel)
    art_date = artifact_date(rel)
    if art_date is None and fam is not None and fam.family == "shap_game":
        # MLB ages SHAP by filename date; NFL ids embed season+week, so
        # resolve the age through the game-date map (frozen store / history).
        mapped = (game_dates or {}).get(_shap_game_id(rel), "")
        art_date = shap_game_date(rel) or mapped.replace("-", "")[:8]
        if not art_date:
            return "protected"  # unresolvable age -> never guess, keep
    if art_date is None:
        # Dateless and not never-delete -> stale (no window can save it).
        return "stale"
    if anchor_date and art_date > anchor_date:
        return "current"  # newer than the run's anchor - backfill-safe keep
    if art_date in retention_dates:
        return "current"  # within the blanket retention window - keep
    if fam is not None and fam.slate_window_days and art_date in recent_dates:
        return "current"  # recent slate snapshot - keep
    if fam is not None and fam.board_supported and art_date in board_dates:
        return "current"  # board still tracked -> keep its run-engine data
    return "stale"


def _shap_game_id(rel: str) -> str:
    """The game-id portion of a SHAP path: nfl_shap_game_<gid>.csv -> <gid>."""
    base = local_name(rel).rsplit("/", 1)[-1]
    if base.startswith("nfl_shap_game_") and base.endswith(".csv"):
        return base[len("nfl_shap_game_"):-len(".csv")]
    return base
