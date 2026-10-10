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
                    "source": "normalized game/team/player box-score facts",
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
    "ewm_off_rating_diff": ("Home minus away EWM offensive rating", "EWM of 100 * points scored / paired estimated possessions", "EWM(3 games)"),
    "ewm_def_rating_diff": ("Home minus away EWM defensive rating", "EWM of 100 * points allowed / paired estimated possessions", "EWM(3 games)"),
    "ewm_pace_diff": ("Home minus away EWM pace", "EWM of paired estimated possessions * 48 / (team player minutes / 5)", "EWM(3 games)"),
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
        "source": "normalized NBA game/team box-score facts",
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
            "source": "normalized NBA game/team box-score facts",
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

# ---------------------------------------------------------------------------
# Dashboard feature metadata (MLB feature_metadata.py parity).
#
# The shared Model Monitor page renders each drift/coverage row's label and
# hover text from the artifact's ``features_metadata`` block — an entry per
# SERVED feature with MLB's exact keys (name/summary/definition/formula/
# source/window/units/direction/members/tooltip). The NBA emitter used to
# ship this block empty, so every row fell back to "(no detailed metadata)"
# and bare column names where MLB showed authored documentation. The
# manifest is the one place feature semantics live, so the block is built
# here (NHL manifest parity: the tooltip is PRE-FORMATTED at the producer).
# ---------------------------------------------------------------------------

#: The monitor page embeds tooltips into a single-quoted HTML attribute with
#: html.escape(..., quote=False), so one surviving apostrophe would terminate
#: the attribute and parse the rest of the tooltip as markup. The escaping
#: happens HERE, at the one place the string is produced (NHL manifest
#: parity).
_ATTR_UNSAFE = {"'": "’"}


def _attr_safe(text: str) -> str:
    """Make a string safe inside a single-quoted HTML attribute."""
    for bad, good in _ATTR_UNSAFE.items():
        text = text.replace(bad, good)
    return text


#: Members of the served blend, in the wording the tooltip shows. Trees see
#: every feature; the elastic net is fit on the diff slice only (the
#: per-side levels are tree-only, mirroring each family's
#: ``model_family_availability`` above).
_TREE_MEMBERS = ["xgboost", "lightgbm"]
_ALL_MEMBERS = ["xgboost", "lightgbm", "elasticnet"]


# Per-family documentation. ``summary`` is the dashboard one-liner (the
# label under the feature name); ``summary_level`` words the raw home/away
# twins. ``formula`` is written against the served column names. ``direction``
# states what a HIGHER value means for the diff representation;
# ``direction_level`` words the side columns. Wording follows the MLB
# catalog so the two reports read alike.
_FAMILY: dict[str, dict] = {
    "elo": {
        "summary": "Home Elo − away Elo (pre-game rating gap)",
        "summary_level": "Entering Elo rating",
        "definition": (
            "Team Elo rating updated after every completed game, read "
            "BEFORE the target game so no target-game result can leak in."),
        "formula": "home_elo − away_elo",
        "source": "Schedule + official results (point-in-time Elo engine)",
        "window": "all prior games (decaying)",
        "units": "Elo points",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side",
    },
    "win_pct": {
        "summary": "Home trailing win% − away trailing win%",
        "summary_level": "Trailing win percentage",
        "definition": (
            "Win rate over each team's trailing games, strictly prior to "
            "the target game."),
        "formula": "win_pct_home − win_pct_away",
        "source": "Schedule + official results",
        "window": "rolling (12)",
        "units": "win% (0–1)",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side",
    },
    "rest_days": {
        "summary": "Home rest days − away rest days",
        "summary_level": "Rest days",
        "definition": "Days since each team's previous game.",
        "formula": "rest_days_home − rest_days_away",
        "source": "Schedule: gap between consecutive game dates",
        "window": "per-game",
        "units": "days",
        "direction": "higher = home advantage (more rested)",
        "direction_level": "higher = more rest",
    },
    "back_to_back": {
        "summary": "Home back-to-back indicator − away indicator",
        "summary_level": "Back-to-back indicator",
        "definition": "1 when the team plays on one or zero days of rest.",
        "formula": "back_to_back_home − back_to_back_away",
        "source": "Schedule: gap between consecutive game dates",
        "window": "per-game",
        "units": "0/1 indicator",
        "direction": "lower = home advantage (fewer back-to-backs)",
        "direction_level": "1 = on a back-to-back",
    },
    "ewm_net_points": {
        "summary": "Home−away exponentially weighted point differential",
        "summary_level": "EWM net points per game",
        "definition": (
            "Exponentially weighted points scored minus points allowed per "
            "game (recent form)."),
        "formula": "ewm_net_points_home − ewm_net_points_away",
        "source": "Season log team box-score facts",
        "window": "EWM(3 games)",
        "units": "points per game",
        "direction": "higher = home advantage",
        "direction_level": "higher = better side form",
    },
    "ewm_off_rating": {
        "summary": "Home−away exponentially weighted offensive rating",
        "summary_level": "EWM offensive rating",
        "definition": (
            "Exponentially weighted points scored per 100 possessions "
            "(offensive efficiency)."),
        "formula": "ewm_off_rating_home − ewm_off_rating_away",
        "source": "Season log team box-score facts",
        "window": "EWM(3 games)",
        "units": "points per 100 possessions",
        "direction": "higher = home advantage",
        "direction_level": "higher = better side form",
    },
    "ewm_def_rating": {
        "summary": "Home−away exponentially weighted defensive rating",
        "summary_level": "EWM defensive rating",
        "definition": (
            "Exponentially weighted points allowed per 100 possessions "
            "(defensive efficiency)."),
        "formula": "ewm_def_rating_home − ewm_def_rating_away",
        "source": "Season log team box-score facts",
        "window": "EWM(3 games)",
        "units": "points allowed per 100 possessions",
        "direction": "lower = home advantage (fewer points allowed)",
        "direction_level": "lower = better side form",
    },
    "ewm_pace": {
        "summary": "Home−away exponentially weighted pace",
        "summary_level": "EWM pace",
        "definition": (
            "Exponentially weighted paired box-score estimated possessions "
            "per 48 minutes. Each side estimates FGA + 0.44*FTA - OREB + TOV; "
            "the two estimates are averaged. Duration is team player minutes / 5, "
            "including overtime; missing counts remain unknown."),
        "formula": "ewm_pace_home − ewm_pace_away",
        "source": "Season log team box-score facts",
        "window": "EWM(3 games)",
        "units": "estimated possessions per 48 minutes",
        "direction": "higher = home team plays faster (style, not quality)",
        "direction_level": "higher = faster tempo",
    },
    "ewm_efg_pct": {
        "summary": "Home−away exponentially weighted effective shooting",
        "summary_level": "EWM effective field-goal percentage",
        "definition": (
            "Exponentially weighted effective field-goal percentage "
            "(shooting quality, threes weighted)."),
        "formula": "ewm_efg_pct_home − ewm_efg_pct_away",
        "source": "Season log team box-score facts (FGM/FGA/FG3M)",
        "window": "EWM(3 games)",
        "units": "rate (0–1)",
        "direction": "higher = home advantage",
        "direction_level": "higher = better side form",
    },
    "ewm_turnover_margin": {
        "summary": "Home−away turnover margin",
        "summary_level": "EWM turnover margin",
        "definition": (
            "Exponentially weighted turnovers forced minus committed per "
            "game."),
        "formula": "ewm_turnover_margin_home − ewm_turnover_margin_away",
        "source": "Season log team box-score facts (TOV)",
        "window": "EWM(3 games)",
        "units": "per game",
        "direction": "higher = home advantage",
        "direction_level": "higher = better side form",
    },
    "ewm_rebound_margin": {
        "summary": "Home−away rebound margin",
        "summary_level": "EWM rebound margin",
        "definition": (
            "Exponentially weighted rebounds won minus conceded per game."),
        "formula": "ewm_rebound_margin_home − ewm_rebound_margin_away",
        "source": "Season log team box-score facts (REB)",
        "window": "EWM(3 games)",
        "units": "per game",
        "direction": "higher = home advantage",
        "direction_level": "higher = better side form",
    },
    "ewm_ast_per_game": {
        "summary": "Home−away assists per game",
        "summary_level": "EWM assists per game",
        "definition": "Exponentially weighted assists per game.",
        "formula": "ewm_ast_per_game_home − ewm_ast_per_game_away",
        "source": "Season log team box-score facts (AST)",
        "window": "EWM(3 games)",
        "units": "assists per game",
        "direction": "higher = home advantage",
        "direction_level": "higher = better side form",
    },
    # ---- play-by-play contribution ------------------------------------
    "event_three_rate": {
        "summary": "Home−away three-point attempt rate",
        "summary_level": "Three-point attempt rate",
        "definition": (
            "Share of field-goal attempts taken from three-point range, "
            "read from the play-by-play."),
        "formula": "event_three_rate_home − event_three_rate_away",
        "source": "Play-by-play (playbyplayv3): shotValue=3, Made/Missed Shot",
        "window": "EWM over prior games",
        "units": "rate (0–1)",
        "direction": "higher = home team takes more threes (style, not quality)",
        "direction_level": "higher = more three-point attempts",
    },
    "event_rim_rate": {
        "summary": "Home−away rim-attack rate",
        "summary_level": "Rim-attack rate",
        "definition": (
            "Share of field-goal attempts taken at the rim, read from the "
            "play-by-play."),
        "formula": "event_rim_rate_home − event_rim_rate_away",
        "source": "Play-by-play (playbyplayv3): shotDistance ≤ 4ft",
        "window": "EWM over prior games",
        "units": "rate (0–1)",
        "direction": "higher = home team attacks the rim more",
        "direction_level": "higher = more rim attempts",
    },
    "event_live_tov_rate": {
        "summary": "Home−away live-ball turnover rate",
        "summary_level": "Live-ball turnover rate",
        "definition": (
            "Rate of live-ball turnovers, read from the play-by-play."),
        "formula": "event_live_tov_rate_home − event_live_tov_rate_away",
        "source": "Play-by-play (playbyplayv3): Turnover (live subtypes)",
        "window": "EWM over prior games",
        "units": "rate (0–1)",
        "direction": "lower = home advantage (fewer live-ball turnovers)",
        "direction_level": "lower = cleaner side play",
    },
    "event_and_in_rate": {
        "summary": "Home−away and-one conversion rate",
        "summary_level": "And-one rate",
        "definition": (
            "Rate of shooting fouls that turn into a made free throw plus "
            "the basket (and-one), read from the play-by-play."),
        "formula": "event_and_in_rate_home − event_and_in_rate_away",
        "source": "Play-by-play (playbyplayv3): Shooting foul → FT 1 of 1",
        "window": "EWM over prior games",
        "units": "rate (0–1)",
        "direction": "higher = home team converts more and-ones",
        "direction_level": "higher = more and-ones",
    },
    "event_shot_distance": {
        "summary": "Home−away average shot distance",
        "summary_level": "Average shot distance",
        "definition": (
            "Average distance of field-goal attempts, read from the "
            "play-by-play."),
        "formula": "event_shot_distance_home − event_shot_distance_away",
        "source": "Play-by-play (playbyplayv3): shotDistance",
        "window": "EWM over prior games",
        "units": "feet",
        "direction": "higher = home team shoots from farther out",
        "direction_level": "higher = longer-range shot diet",
    },
    "event_possessions": {
        "summary": "Home−away possessions per game",
        "summary_level": "Possessions per game",
        "definition": (
            "Estimated possessions per game (FGA + 0.44×FTA − OREB + TOV), "
            "read from the play-by-play."),
        "formula": "event_possessions_home − event_possessions_away",
        "source": "Play-by-play (playbyplayv3): FGA + 0.44*FTA − OREB + TOV",
        "window": "EWM over prior games",
        "units": "possessions per game",
        "direction": "higher = home team plays at a faster tempo",
        "direction_level": "higher = faster tempo",
    },
    "event_shooting_fouls": {
        "summary": "Home−away shooting fouls per game",
        "summary_level": "Shooting fouls per game",
        "definition": (
            "Shooting fouls involved in per game, read from the "
            "play-by-play."),
        "formula": "event_shooting_fouls_home − event_shooting_fouls_away",
        "source": "Play-by-play (playbyplayv3): Foul (Shooting)",
        "window": "EWM over prior games",
        "units": "per game",
        "direction": "higher = more home shooting fouls",
        "direction_level": "higher = more shooting fouls",
    },
    "event_q4_points": {
        "summary": "Home−away fourth-quarter points per game",
        "summary_level": "Fourth-quarter points per game",
        "definition": (
            "Points scored in the fourth quarter per game, read from the "
            "play-by-play (closing-time scoring)."),
        "formula": "event_q4_points_home − event_q4_points_away",
        "source": "Play-by-play (playbyplayv3): period=4 scoring",
        "window": "EWM over prior games",
        "units": "points per game",
        "direction": "higher = home advantage",
        "direction_level": "higher = better side closing scoring",
    },
    # ---- position-segmented projected-lineup RAPM ----------------------
    # Retired player-level families (kept so the historical drift reports
    # keep their documentation): the projected-lineup pool shipped true
    # shooting (pl_ts_*) through 2026-10-01 and EPM (pl_epm_*) on 2026-10-02
    # before the current RAPM (pl_rapm_*) family.
    "pl_ts_c": {
        "summary": "Home−away projected-lineup center true shooting",
        "summary_level": "Projected-lineup center true shooting",
        "definition": (
            "Minutes-weighted true-shooting percentage for the center pool "
            "of the projected lineup (retired 2026-10-02 family)."),
        "formula": "pl_ts_c_home − pl_ts_c_away",
        "source": "Season log player lines → position pool true shooting",
        "window": "season to date",
        "units": "rate (0–1)",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_ts_f": {
        "summary": "Home−away projected-lineup forward true shooting",
        "summary_level": "Projected-lineup forward true shooting",
        "definition": (
            "Minutes-weighted true-shooting percentage for the forward pool "
            "of the projected lineup (retired 2026-10-02 family)."),
        "formula": "pl_ts_f_home − pl_ts_f_away",
        "source": "Season log player lines → position pool true shooting",
        "window": "season to date",
        "units": "rate (0–1)",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_ts_g": {
        "summary": "Home−away projected-lineup guard true shooting",
        "summary_level": "Projected-lineup guard true shooting",
        "definition": (
            "Minutes-weighted true-shooting percentage for the guard pool "
            "of the projected lineup (retired 2026-10-02 family)."),
        "formula": "pl_ts_g_home − pl_ts_g_away",
        "source": "Season log player lines → position pool true shooting",
        "window": "season to date",
        "units": "rate (0–1)",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_epm_c": {
        "summary": "Home−away projected-lineup center EPM",
        "summary_level": "Projected-lineup center EPM",
        "definition": (
            "Minutes-weighted estimated plus-minus for the center pool of "
            "the projected lineup (retired 2026-10-03 family)."),
        "formula": "pl_epm_c_home − pl_epm_c_away",
        "source": "Season log player lines → position pool estimated plus-minus",
        "window": "season to date",
        "units": "points per 100 possessions",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_epm_f": {
        "summary": "Home−away projected-lineup forward EPM",
        "summary_level": "Projected-lineup forward EPM",
        "definition": (
            "Minutes-weighted estimated plus-minus for the forward pool of "
            "the projected lineup (retired 2026-10-03 family)."),
        "formula": "pl_epm_f_home − pl_epm_f_away",
        "source": "Season log player lines → position pool estimated plus-minus",
        "window": "season to date",
        "units": "points per 100 possessions",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_epm_g": {
        "summary": "Home−away projected-lineup guard EPM",
        "summary_level": "Projected-lineup guard EPM",
        "definition": (
            "Minutes-weighted estimated plus-minus for the guard pool of "
            "the projected lineup (retired 2026-10-03 family)."),
        "formula": "pl_epm_g_home − pl_epm_g_away",
        "source": "Season log player lines → position pool estimated plus-minus",
        "window": "season to date",
        "units": "points per 100 possessions",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_rapm_c": {
        "summary": "Home−away projected-lineup center RAPM",
        "summary_level": "Projected-lineup center RAPM",
        "definition": (
            "Position-segmented game-level MIN/48 ridge proxy for the center pool of the "
            "projected lineup, minutes-weighted and injury-filtered "
            "(Out/Doubtful/Recovery excluded before tip-off)."),
        "formula": "pl_rapm_c_home − pl_rapm_c_away",
        "source": "Strictly-prior season log minutes + game margins → shrunk ridge proxy; dated official NBA roster membership (7-day expiry; strict-prior appearance fallback) → projected position average (not possession/stint RAPM)",
        "window": "season to date; last completed season before own-season evidence",
        "units": "game-margin points per full-48-minute player exposure",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_rapm_f": {
        "summary": "Home−away projected-lineup forward RAPM",
        "summary_level": "Projected-lineup forward RAPM",
        "definition": (
            "Position-segmented game-level MIN/48 ridge proxy for the forward pool of the "
            "projected lineup, minutes-weighted and injury-filtered."),
        "formula": "pl_rapm_f_home − pl_rapm_f_away",
        "source": "Strictly-prior season log minutes + game margins → shrunk ridge proxy; dated official NBA roster membership (7-day expiry; strict-prior appearance fallback) → projected position average (not possession/stint RAPM)",
        "window": "season to date; last completed season before own-season evidence",
        "units": "game-margin points per full-48-minute player exposure",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
    "pl_rapm_g": {
        "summary": "Home−away projected-lineup guard RAPM",
        "summary_level": "Projected-lineup guard RAPM",
        "definition": (
            "Position-segmented game-level MIN/48 ridge proxy for the guard pool of the "
            "projected lineup, minutes-weighted and injury-filtered."),
        "formula": "pl_rapm_g_home − pl_rapm_g_away",
        "source": "Strictly-prior season log minutes + game margins → shrunk ridge proxy; dated official NBA roster membership (7-day expiry; strict-prior appearance fallback) → projected position average (not possession/stint RAPM)",
        "window": "season to date; last completed season before own-season evidence",
        "units": "game-margin points per full-48-minute player exposure",
        "direction": "higher = home advantage",
        "direction_level": "higher = stronger side pool",
    },
}

_STANDALONE = {
    "is_home": {
        "summary": "Constant 1 — anchors the home-court edge",
        "definition": (
            "Constant intercept column marking the home side. Every row is "
            "1 because the model always scores the home team's chance of "
            "winning."),
        "formula": "1",
        "source": "Feature engineering: static column on every game row",
        "window": "n/a (constant)",
        "units": "binary",
        "direction": "n/a (constant)",
    },
    "is_playoffs": {
        "summary": "1 for postseason games",
        "definition": "1 for postseason game type, 0 for regular season.",
        "formula": "1 if season_type == playoffs else 0",
        "source": "Schedule (season type)",
        "window": "per-game",
        "units": "binary",
        "direction": "n/a (game-type flag)",
    },
}


def format_tooltip(meta: dict) -> str:
    """Plain-text tooltip body — MLB's exact line layout.

    Rendered into an HTML ``title`` attribute by the shared monitor page,
    which escapes everything except quote characters, hence ``_attr_safe``.
    """
    def _get(key: str) -> str:
        val = str(meta.get(key, "") or "").strip()
        return val or "—"

    def _members() -> str:
        members = meta.get("members")
        try:
            return ", ".join(str(m) for m in members)
        except TypeError:
            return str(members or "—")

    return _attr_safe(
        f"What: {_get('definition')}\n"
        f"Formula: {_get('formula')}\n"
        f"Source: {_get('source')}\n"
        f"Window: {_get('window')} · Units: {_get('units')}\n"
        f"Direction: {_get('direction')}\n"
        f"Consumed by: {_members()}"
    )


def _entry_for(name: str) -> dict | None:
    """MLB-shaped metadata entry for one served name (None = undocumented)."""
    if name in _STANDALONE:
        row = {"name": name, **_STANDALONE[name], "members": list(_ALL_MEMBERS)}
        row["tooltip"] = format_tooltip(row)
        return row
    for base, fam in _FAMILY.items():
        if name == f"{base}_diff":
            members = list(_ALL_MEMBERS)
            row = {
                "name": name, "summary": fam["summary"],
                "definition": fam["definition"], "formula": fam["formula"],
                "source": fam["source"], "window": fam["window"],
                "units": fam["units"], "direction": fam["direction"],
                "members": members,
            }
            row["tooltip"] = format_tooltip(row)
            return row
        for side in ("home", "away"):
            if name == f"{base}_{side}":
                members = list(_TREE_MEMBERS)  # side levels are tree-only
                row = {
                    "name": name,
                    "summary": f"{fam['summary_level']} — {side} team",
                    "definition": fam["definition"],
                    "formula": fam["formula"],
                    "source": fam["source"], "window": fam["window"],
                    "units": fam["units"],
                    "direction": (f"{fam['direction_level']} for the "
                                  f"{side} side (level column)"),
                    "members": members,
                }
                row["tooltip"] = format_tooltip(row)
                return row
    # RFE candidate families (nba_<metric>_<window>_<rep>): documented by
    # CANDIDATE_MANIFEST rather than the family table.
    for pool in (CANDIDATE_MANIFEST, FEATURE_MANIFEST):
        entry = pool.get(name)
        if entry:
            rep = entry.get("representation", "")
            summary = str(entry.get("description") or name)
            if rep == "raw home level":
                summary = f"{summary} — home team" if not summary.endswith("home") else summary
            elif rep == "raw away level":
                summary = f"{summary} — away team" if not summary.endswith("away") else summary
            row = {
                "name": name, "summary": summary,
                "definition": entry.get("definition"),
                "formula": "—", "source": entry.get("source"),
                "window": str(entry.get("lookback", "—")),
                "units": "—",
                "direction": ("n/a (see definition)"
                              if rep == "shared" else
                              "higher = home advantage"
                              if rep == "difference" else
                              "higher = stronger side"),
                "members": (list(_ALL_MEMBERS)
                            if rep in ("difference", "shared")
                            else list(_TREE_MEMBERS)),
            }
            row["tooltip"] = format_tooltip(row)
            return row
    return None


def build_features_metadata(names: list[str] | None = None) -> tuple[dict, list[str]]:
    """``({feature: entry}, warnings)`` for the monitor artifact.

    MLB parity: every ACTIVE serving col gets a row; unauthored features get
    a clearly-marked placeholder AND a warning so absence is never silent.
    """
    serving = list(names if names is not None
                   else config.active_moneyline_feature_cols())
    meta: dict[str, dict] = {}
    warnings: list[str] = []
    for name in serving:
        entry = _entry_for(name)
        if entry is None:
            msg = (f"Feature metadata: no authored entry for {name!r} — "
                   "shipping a PLACEHOLDER (fill in manifest._FAMILY)")
            warnings.append(msg)
            entry = {
                "name": name, "summary": name,
                "definition": "No detailed metadata authored yet.",
                "formula": "—", "source": "—", "window": "—",
                "units": "—", "direction": "—",
                "members": list(_ALL_MEMBERS),
            }
            entry["tooltip"] = format_tooltip(entry)
        meta[name] = entry
    return meta, warnings


def feature_tooltips(names: list[str] | None = None) -> dict:
    """``{feature: entry}`` for the monitor artifact (empty on failure).

    Only documented features get an entry; an undocumented name is simply
    absent, which is what the page needs to tell "no metadata" from
    "metadata exists".
    """
    out: dict = {}
    for name in (names if names is not None
                 else config.active_moneyline_feature_cols()):
        entry = _entry_for(name)
        if entry is not None:
            out[name] = entry
    return out


def validate() -> list[str]:
    """Return name-for-name documentation mismatches.

    A name is documented when it has a manifest entry OR a metadata entry
    (the dashboard catalog above covers the side/event/player-level families
    the base manifest never listed).
    """
    return sorted(n for n in config.MONEYLINE_FEATURE_COLS
                  if n not in FEATURE_MANIFEST and _entry_for(n) is None)
