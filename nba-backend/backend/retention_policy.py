"""Rolling retention policy for NBA delivery artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

EXACT_MASTER_NAMES = frozenset({
    "nba_production_cards_history.csv", "nba_production_cards_history.meta.json",
    "nba_fold_table.csv", "nba_pipeline_summary.json", "nba_oof_moneyline.csv",
    "nba_oof_distribution.csv", "nba_feature_selection_state.json",
    "nba_oof_store.csv",
})
SERIES_PREFIXES = ("models/", "nba_run_engine_monitor_")


@dataclass(frozen=True)
class FamilyPolicy:
    family: str
    prefix: str
    retention_days: Optional[int]
    allowlisted: bool
    slate_window_days: int = 0
    notes: str = ""


FAMILY_POLICY = (
    FamilyPolicy("moneyline", "nba_moneyline_v1_", 10, True),
    FamilyPolicy("calibration", "nba_calibration_", 10, True),
    FamilyPolicy("predictions_history", "nba_predictions_history_", 10, True),
    FamilyPolicy("model_monitor", "nba_model_monitor_", 10, True),
    FamilyPolicy("feature_v1", "nba_feature_v1_", 10, True),
    FamilyPolicy("power_rankings", "nba_power_rankings_", 10, True),
    FamilyPolicy("markets", "nba_run_engine_markets_", 10, True),
    FamilyPolicy("player_matchup", "nba_player_leader_matchup_", 10, True),
    FamilyPolicy("markets_monitor", "nba_run_engine_monitor_", None, False),
    FamilyPolicy("shap_game", "nba_shap_game_", None, True, 10),
    FamilyPolicy("rfe_trace", "nba_feature_selection_", 10, True),
    FamilyPolicy("rfe_workbook", "nba_feature_workbook_", 10, True),
    FamilyPolicy("run_engine_feature_drift", "nba_run_engine_feature_drift_", 10, True),
    FamilyPolicy("run_engine_feature_coverage", "nba_run_engine_feature_coverage_", 10, True),
)
_COMPACT = re.compile(r"_(\d{8})")
_ISO = re.compile(r"_(\d{4}-\d{2}-\d{2})")
_SHAP = re.compile(r"nba_shap_game_(\d{8})_")


def local_name(rel: str) -> str:
    rel = str(rel).replace("\\", "/")
    marker = "data_delivery/"
    return rel.split(marker, 1)[1] if marker in rel else rel


def artifact_date(rel: str) -> Optional[str]:
    match = _COMPACT.search(str(rel)) or _ISO.search(str(rel))
    return match.group(1).replace("-", "") if match else None


def _family(rel: str) -> FamilyPolicy | None:
    base = local_name(rel).rsplit("/", 1)[-1]
    hits = [policy for policy in FAMILY_POLICY if base.startswith(policy.prefix)]
    return max(hits, key=lambda policy: len(policy.prefix)) if hits else None


def is_never_delete(rel: str) -> bool:
    """Return true only for explicit masters and never-expiring series."""
    local = local_name(rel)
    base = local.rsplit("/", 1)[-1]
    if base in EXACT_MASTER_NAMES:
        return True
    return any(local.startswith(prefix) or base.startswith(prefix)
               for prefix in SERIES_PREFIXES)


def classify_artifact(rel: str, seen: set, retention_dates: set,
                      recent_dates: set, board_dates: set,
                      anchor_date: Optional[str] = None,
                      game_dates: Optional[dict] = None) -> str:
    if rel in seen:
        return "seen"
    if is_never_delete(rel):
        return "protected"
    family = _family(rel)
    date = artifact_date(rel)
    if date is None and family and family.family == "shap_game":
        match = _SHAP.search(local_name(rel))
        date = match.group(1) if match else (game_dates or {}).get(local_name(rel))
        if not date:
            return "protected"
    if date is None:
        return "stale"
    if anchor_date and date > anchor_date:
        return "current"
    if date in retention_dates:
        return "current"
    if family and family.slate_window_days and date in recent_dates:
        return "current"
    if family and family.family in {"markets", "predictions_history"} and date in board_dates:
        return "current"
    return "stale"
