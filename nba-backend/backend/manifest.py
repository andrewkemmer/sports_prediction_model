"""Authoritative documentation for the NBA feature contract."""
from __future__ import annotations

try:
    from backend import config
except ImportError:
    import config

CANDIDATE_MANIFEST: dict[str, dict] = {}

def _candidate_manifest() -> None:
    for metric, windows in config.TEAM_CANDIDATE_TRAILING_SPECS.items():
        for window in windows:
            base = f"nba_{metric}_{window}"
            for rep, representation in (("diff", "difference"), ("home", "raw home level"), ("away", "raw away level")):
                CANDIDATE_MANIFEST[f"{base}_{rep}"] = {
                    "description": f"Trailing {metric} ({window}) {rep}",
                    "definition": f"Per-team {window} aggregate of {metric}, strictly prior to the target game.",
                    "source": "wyattowalsh/basketball normalized game/team box-score facts",
                    "lookback": window,
                    "aggregation": "per-team trailing aggregate",
                    "point_in_time_rule": "computed chronologically and shifted one game; current/future rows excluded",
                    "missing_value_policy": "NaN when the source family is unavailable; never fabricated",
                    "representation": representation,
                    "model_family_availability": ["linear", "tree"] if rep == "diff" else ["tree"],
                    "feature_version": 1,
                    "candidate": True,
                }

_candidate_manifest()

_BASE = {
    "elo_diff": ("Home minus away entering Elo", "Elo rating before the target game", "iterative team state"),
    "win_pct_diff": ("Home minus away trailing win percentage", "prior-game win rate", "rolling(12)"),
    "rest_days_diff": ("Home minus away rest days", "days since each team's prior game", "schedule"),
    "back_to_back_diff": ("Home minus away back-to-back indicator", "rest <= 1 day", "schedule"),
    "ewm_net_points_diff": ("Home minus away EWM point differential", "exponentially weighted points for minus against", "EWM(3 games)"),
    "ewm_off_rating_diff": ("Home minus away EWM offensive rating", "exponentially weighted offensive efficiency", "EWM(3 games)"),
    "ewm_def_rating_diff": ("Home minus away EWM defensive rating", "exponentially weighted defensive efficiency", "EWM(3 games)"),
    "ewm_pace_diff": ("Home minus away EWM pace", "exponentially weighted possessions", "EWM(3 games)"),
    "ewm_efg_pct_diff": ("Home minus away EWM effective shooting", "exponentially weighted eFG%", "EWM(3 games)"),
    "ewm_turnover_margin_diff": ("Home minus away turnover margin", "exponentially weighted TOV margin", "EWM(3 games)"),
    "ewm_rebound_margin_diff": ("Home minus away rebound margin", "exponentially weighted rebound margin", "EWM(3 games)"),
    "ewm_ast_per_game_diff": ("Home minus away assists per game", "exponentially weighted assists", "EWM(3 games)"),
    "is_playoffs": ("Playoff indicator", "1 for postseason game type", "schedule"),
}
FEATURE_MANIFEST: dict[str, dict] = {}
for _name, (_desc, _definition, _lookback) in _BASE.items():
    FEATURE_MANIFEST[_name] = {
        "description": _desc,
        "definition": _definition,
        "source": "normalized NBA warehouse game/team box-score facts",
        "lookback": _lookback,
        "aggregation": "point-in-time team aggregate",
        "point_in_time_rule": "all game outcomes are strictly prior; no target-game box score or future aggregate is admitted",
        "missing_value_policy": "NaN when unavailable; train-fold median imputation for linear members and native NaN for tree members",
        "representation": "difference" if _name.endswith("_diff") else "shared",
        "model_family_availability": ["linear", "tree"],
        "feature_version": 1,
    }
for _side in ("home", "away"):
    for _base, _desc in (("elo", "entering Elo"), ("win_pct", "trailing win percentage"),
                         ("ewm_off_rating", "EWM offensive rating"),
                         ("ewm_def_rating", "EWM defensive rating"), ("rest_days", "rest days")):
        _n = f"{_base}_{_side}"
        FEATURE_MANIFEST[_n] = {
            "description": f"{_side.title()} {_desc}",
            "definition": f"The {_side} team's {_desc} entering the target game.",
            "source": "normalized NBA warehouse game/team box-score facts",
            "lookback": "team chronology",
            "aggregation": "point-in-time team aggregate",
            "point_in_time_rule": "strictly prior games only",
            "missing_value_policy": "NaN when unavailable",
            "representation": "raw side level",
            "model_family_availability": ["tree"],
            "feature_version": 1,
        }
FEATURE_MANIFEST["is_home"] = {
    "description": "Home-team anchor", "definition": "Constant 1.0 for the home side.",
    "source": "schedule", "lookback": 0, "aggregation": "constant",
    "point_in_time_rule": "pre-game schedule fact", "missing_value_policy": "never missing",
    "representation": "shared", "model_family_availability": ["linear", "tree"], "feature_version": 1,
}

def validate() -> list[str]:
    """Return name-for-name documentation mismatches."""
    return sorted(set(config.MONEYLINE_FEATURE_COLS) - set(FEATURE_MANIFEST))
