"""Explicit rolling-retention policy for NHL ``data_delivery`` dated artifacts.

*** Policy: blanket 10-day window, one anchor (MLB parity, 2026-09-23) ***

Every DATED, deletion-allowlisted artifact family keeps the run's anchor date
and the 10 days before it (anchor = ``NHL_END_DATE`` when the run sets it,
else today in America/New_York). Anything older is stale: the pipeline's
artifact sync stages the deletions, so git history retains every blob.
Files NEWER than the anchor are never touched (backfill-safe), and the scope
is exactly ``nhl-backend/data_delivery/``.

NHL-specific wrinkle: SHAP files are written per GAME, so they must age by
the GAME date, not the run date in the filename. Two id conventions exist in
the wild: the legacy ESPN-style ``YYYYMMDD_AWAY@HOME`` (date embedded in the
filename) and the current official NHL API numeric id (``2026020001``, which
carries no date and must be resolved through the ``game_dates`` map the
pipeline builds from the moneyline artifacts).

This mirrors ``mlb-backend/backend/retention_policy.py`` structurally: one
``FamilyPolicy`` config table, one pure ``classify_artifact`` predicate, and
the same never-delete classes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# -- Masters (exact-name): dateless cumulative / maintained files cleanup   --
# -- never touches.                                                        --
EXACT_MASTER_NAMES = frozenset({
    # The frozen first-publication card store (append-only).
    "nhl_production_cards_history.csv",
    "nhl_production_cards_history.meta.json",
    # Adopted RFE serving width — written on explicit adopt only.
    "nhl_feature_selection_state.json",
    # Run internals / maintained tables.
    "nhl_fold_table.csv",
    "nhl_pipeline_summary.json",
    "nhl_oof_moneyline.csv",
    "nhl_oof_distribution.csv",
})

# -- Series readers / cumulative stores (prefix): deleting ANY member would --
# -- reset rolling history.                                                --
SERIES_PREFIXES = (
    "models/",
    # The markets page folds ALL dated monitors into the rolling per-card
    # series (fold_slate_history). Never reset it.
    "nhl_run_engine_monitor_",
)


@dataclass(frozen=True)
class FamilyPolicy:
    """One dated family's retention rule (the single config table)."""

    family: str
    prefix: str
    retention_days: Optional[int]   # None -> governed by board/slate/exempt
    allowlisted: bool               # True -> eligible for deletion
    board_supported: bool = False   # keep while a board for that date is tracked
    slate_window_days: int = 0      # keep for the recent-slate settle window
    notes: str = ""


# Longest-prefix match (see _family_for). Every NHL artifact family appears.
FAMILY_POLICY: tuple[FamilyPolicy, ...] = (
    FamilyPolicy("moneyline", "nhl_moneyline_v1_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (board moneyline JSON; valid_dates "
                       "games[] from NEWEST); 10-day blanket window"),
    FamilyPolicy("calibration", "nhl_calibration_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (Calibration page _pick_artifact_date); "
                       "10-day blanket window"),
    FamilyPolicy("predictions_history", "nhl_predictions_history_",
                 retention_days=10, allowlisted=True, board_supported=True,
                 notes="newest-only consumers (calibration curve/table, "
                       "store fallback); 10-day blanket window"),
    FamilyPolicy("model_monitor", "nhl_model_monitor_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (monitor page); 10-day blanket window"),
    FamilyPolicy("feature_v1", "nhl_feature_v1_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only feature dictionary; 10-day window"),
    FamilyPolicy("power_rankings", "nhl_power_rankings_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (shared rankings page); 10-day window"),
    FamilyPolicy("markets", "nhl_run_engine_markets_",
                 retention_days=10, allowlisted=True, board_supported=True,
                 notes="newest-only (markets page + board enrichment, incl. "
                       ".meta.json); 10-day blanket window"),
    FamilyPolicy("markets_monitor", "nhl_run_engine_monitor_",
                 retention_days=None, allowlisted=False,
                 notes="SERIES - fold_slate_history folds ALL dated monitors"),
    FamilyPolicy("goalie_matchup", "nhl_goalie_matchup_", retention_days=10,
                 allowlisted=True,
                 notes="newest-only (board goalie boxes); 10-day window"),
    FamilyPolicy("run_engine_feature_drift", "run_engine_feature_drift_",
                 retention_days=10, allowlisted=True,
                 notes="newest-only drift table; 10-day window"),
    FamilyPolicy("run_engine_feature_coverage", "run_engine_feature_coverage_",
                 retention_days=10, allowlisted=True,
                 notes="newest-only coverage table; 10-day window"),
    FamilyPolicy("shap_game", "nhl_shap_game_",
                 retention_days=None, allowlisted=True, slate_window_days=10,
                 notes="per-game cards aged by the GAME date: embedded in the "
                       "filename for legacy YYYYMMDD_AWAY@HOME ids, else "
                       "resolved through the game-date map for official NHL "
                       "numeric ids; pruned after 10 days past the game; "
                       "current/future slate kept"),
    FamilyPolicy("rfe_trace", "nhl_feature_selection_", retention_days=10,
                 allowlisted=True,
                 notes="RFE trace record; adoption reads the STATE file "
                       "(exempt)"),
    FamilyPolicy("rfe_workbook", "nhl_feature_workbook_", retention_days=10,
                 allowlisted=True,
                 notes="human decision workbook; regenerated per RFE run; "
                       "10-day window"),
)

# -- Predicates --------------------------------------------------------------

_COMPACT_DATE_RE = re.compile(r"_(\d{8})(?!\d)")          # nhl_calibration_20260922.json
_ISO_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})(?!\d)")  # nhl_feature_workbook_2026-09-22.xlsx
_SHAP_DATE_RE = re.compile(r"nhl_shap_game_(\d{8})_")


def local_name(rel: str) -> str:
    """Path relative to ``data_delivery/`` (matches the MLB helper)."""
    rel = rel.replace("\\", "/")
    dd = "data_delivery/"
    idx = rel.find(dd)
    return rel[idx + len(dd):] if idx >= 0 else rel


def artifact_date(rel: str) -> Optional[str]:
    """Extract the artifact date as compact YYYYMMDD, or None if dateless.

    Understands the compact ``_YYYYMMDD`` convention (board families) and
    the ISO ``_YYYY-MM-DD`` convention (RFE workbooks). The date must be a
    COMPLETE 8-digit token: a longer digit run (an official NHL numeric game
    id such as ``nhl_shap_game_2026020001.csv``) is deliberately NOT matched,
    so a truncated prefix can never be mistaken for a date. SHAP ages resolve
    through ``shap_game_date`` / the game-date map instead.
    """
    m = _COMPACT_DATE_RE.search(rel) or _ISO_DATE_RE.search(rel)
    return m.group(1).replace("-", "") if m else None


def shap_game_date(rel: str) -> Optional[str]:
    """The YYYYMMDD embedded in a legacy ``nhl_shap_game_<YYYYMMDD_AWAY@HOME>.csv``
    filename, or None for an official NHL numeric game id (which carries no
    date and resolves through the game-date map instead)."""
    m = _SHAP_DATE_RE.search(local_name(rel))
    return m.group(1) if m else None


def is_never_delete(rel: str) -> bool:
    """True for masters / series members cleanup never touches.

    Prefix matching is segment-aware so both the production layout
    (``data_delivery/models/...``) and bare/test layouts classify identically.
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
      "protected" - never-delete (master / series reader)
      "current"   - kept via a family window (blanket 10-day retention,
                    recent slate, board-backed) - or NEWER than the anchor
      "stale"     - safe to delete under the policy

    ``anchor_date`` (YYYYMMDD): artifacts dated AFTER the anchor are always
    kept (backfill-safe). ``game_dates`` (game_id -> YYYYMMDD) is consulted
    for the ``shap_game`` family when the filename carries no date.
    """
    if rel in seen:
        return "seen"
    if is_never_delete(rel):
        return "protected"
    fam = _family_for(rel)
    if fam is not None and fam.family == "shap_game":
        # SHAP ages by GAME date, never by the run date in the filename.
        # Two id conventions are in the wild: the legacy ESPN-style
        # ``YYYYMMDD_AWAY@HOME`` (date embedded, read from the filename) and
        # the current official NHL API numeric id (``2026020001``), which
        # carries NO date and must be resolved through ``game_dates``.
        # An id that resolves to neither is kept (protected) — never guess.
        art_date = shap_game_date(rel) or (game_dates or {}).get(
            _shap_game_id(rel), "").replace("-", "")[:8]
        if not art_date:
            return "protected"  # unresolvable age -> never guess, keep
    else:
        art_date = artifact_date(rel)
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
    """The game-id portion of a SHAP path: nhl_shap_game_<gid>.csv -> <gid>."""
    base = local_name(rel).rsplit("/", 1)[-1]
    if base.startswith("nhl_shap_game_") and base.endswith(".csv"):
        return base[len("nhl_shap_game_"):-len(".csv")]
    return base
