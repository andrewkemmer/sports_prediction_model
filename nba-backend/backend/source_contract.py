"""The normalized frame contract, independent of which vendor supplied the rows.

Every upstream disagrees about names, granularity, and units, and the failure
this module exists to prevent is a model that quietly learns one vendor's
vocabulary.  ``features.py`` should be able to ask for ``points_for`` and get
it whether the row came from a season log, a box-score walk, or something
adopted next year.  So the vocabulary lives here, once, and each source
declares the mapping that gets it there.

Two ideas carry the module:

**A schema is data, not a comment.** ``GAMES_SCHEMA`` names every column, its
dtype, and whether the model may train without it.  A new source that cannot
fill a required column is rejected at the boundary with the name of the
column, not discovered three phases later as a feature that is silently NaN.

**Normalization is a transformation, not a convention.** ``normalize`` coerces
dtypes, parses the timestamp column, and rejects a frame that is missing
something required - so a source that returns the right columns with the
wrong types is caught here rather than by a subtraction deep in a fold.

A source is a ``SourceSpec``: a name for the manifest, the frames it can
produce, and a callable per frame.  Adding a vendor means adding one of these
and nothing else, which is the property that makes a fallback route affordable
instead of a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
#
# ``required`` means the model refuses to train without it. ``optional`` means
# a source may omit it and the frame is allowed to be short that column, which
# is how a leaner source stays usable without pretending to be complete.

GAMES_SCHEMA: dict[str, str] = {
    "game_id": "str", "gameday": "datetime", "season": "int",
    "home_team": "str", "away_team": "str",
    "home_score": "float", "away_score": "float",
    "game_type": "int", "margin": "float", "total": "float",
    "home_win": "float",
}
# ``season`` is derivable from the tipoff, so a source is not required to
# supply it. Everything here has to arrive from the source: a games frame
# missing any of these cannot be reconstructed from anything else we hold.
GAMES_REQUIRED = ("game_id", "gameday", "home_team", "away_team",
                  "home_score", "away_score")

# One row per team per game. These are the facts the ladder aggregates, so the
# count column here is the count the model actually consumes.
TEAM_STATS_SCHEMA: dict[str, str] = {
    "game_id": "str", "team": "str", "gameday": "datetime",
    "is_home": "bool", "opponent": "str",
    "points_for": "float", "points_against": "float", "net_points": "float",
    "fgm": "float", "fga": "float", "fg_pct": "float",
    "fg3m": "float", "fg3a": "float", "three_point_pct": "float",
    "ftm": "float", "fta": "float", "free_throw_pct": "float",
    "oreb": "float", "dreb": "float", "reb": "float",
    "ast": "float", "tov": "float", "stl": "float", "blk": "float",
    "pf": "float",
    "efg_pct": "float", "turnover_margin": "float", "rebound_margin": "float",
    "pace": "float", "ast_per_game": "float",
}
TEAM_STATS_REQUIRED = ("game_id", "team", "points_for", "points_against")

# One row per player per game.
PLAYER_STATS_SCHEMA: dict[str, str] = {
    "game_id": "str", "gameday": "datetime", "player_id": "str",
    "player_name": "str", "team": "str", "minutes": "float",
    "win": "float", "game_type": "int", "plus_minus": "float",
    "points": "float", "reb": "float", "ast": "float", "tov": "float",
    "stl": "float", "blk": "float", "pf": "float",
    "fgm": "float", "fga": "float", "fg_pct": "float",
    "fg3m": "float", "fg3a": "float", "three_point_pct": "float",
    "ftm": "float", "fta": "float", "free_throw_pct": "float",
    "oreb": "float", "dreb": "float",
}
PLAYER_STATS_REQUIRED = ("game_id", "player_id", "team")

PLAY_BY_PLAY_SCHEMA: dict[str, str] = {
    "game_id": "str", "gameday": "datetime", "action_id": "float",
    "action_number": "float", "period": "float", "clock": "str",
    "team": "str", "player_id": "float", "action_type": "str",
    "sub_type": "str", "description": "str", "score_home": "float",
    "score_away": "float", "points": "float", "shot_distance": "float",
    "shot_result": "str", "is_field_goal": "float", "shot_value": "float",
    "x": "float", "y": "float",
}
PLAY_BY_PLAY_REQUIRED = ("game_id", "action_id")

SCHEMAS: dict[str, dict[str, str]] = {
    "games": GAMES_SCHEMA,
    "team_stats": TEAM_STATS_SCHEMA,
    "player_stats": PLAYER_STATS_SCHEMA,
    "play_by_play": PLAY_BY_PLAY_SCHEMA,
}
REQUIRED: dict[str, tuple[str, ...]] = {
    "games": GAMES_REQUIRED,
    "team_stats": TEAM_STATS_REQUIRED,
    "player_stats": PLAYER_STATS_REQUIRED,
    "play_by_play": PLAY_BY_PLAY_REQUIRED,
}

# Columns that are derived rather than measured, so a source must not supply
# them and ``normalize`` always recomputes them. Letting a vendor set a derived
# column is how a subtle double-count enters a model and never leaves.
DERIVED_GAMES = ("margin", "total", "home_win")


class ContractError(ValueError):
    """A source produced a frame the model contract will not accept."""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _coerce(frame: pd.DataFrame, column: str, kind: str) -> pd.Series:
    if kind == "str":
        return frame[column].astype(str)
    if kind == "datetime":
        return pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_convert(None)
    if kind == "bool":
        return frame[column].astype(bool)
    values = pd.to_numeric(frame[column], errors="coerce")
    # Every measured quantity is a float even where the values happen to be
    # whole numbers, because a source that sends 110 for a score and another
    # that sends 110.5 must land in the same column type. An int64 column here
    # would raise inside the first fraction it met, in a fold, mid-run.
    if kind == "int":
        return values
    return values.astype(float)


def normalize(frame: pd.DataFrame, name: str, *, source: str = "unknown",
              allow_missing: bool = False) -> pd.DataFrame:
    """Put one frame into the contract: dtypes, derived columns, order.

    ``allow_missing`` is for a frame that is legitimately empty - a source that
    answered with no rows is not a contract violation, but a source that
    answered with rows and dropped a required column is.  Keeping those two
    apart matters, because the first is a gap in the data and the second is a
    bug in the adapter, and they must not share an exit path.
    """
    if name not in SCHEMAS:
        raise ContractError(f"no contract declared for frame {name!r}")
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(SCHEMAS[name]))
    schema = SCHEMAS[name]
    required = REQUIRED[name]
    missing = [c for c in required if c not in frame.columns]
    if missing and not allow_missing:
        raise ContractError(
            f"source {source!r} frame {name!r} is missing required "
            f"column(s) {missing}; the contract requires {list(required)}. "
            f"A source may omit a column only if the model can do without it, "
            f"and these decide what it can do without.")

    out = pd.DataFrame(index=frame.index)
    for column, kind in schema.items():
        if column in frame.columns:
            out[column] = _coerce(frame, column, kind)
        elif column in DERIVED_GAMES and name == "games":
            out[column] = np.nan
        else:
            out[column] = np.nan

    if name in ("team_stats", "player_stats") and len(out):
        out["game_id"] = out["game_id"].astype(str)
    if name == "games" and len(out):
        out["game_id"] = out["game_id"].astype(str)
        # Derived from the two scores, always, so no source can hand the model
        # a margin that disagrees with the score it was given. Recomputed even
        # when the source supplied one, because a source's own arithmetic is
        # exactly the thing not worth trusting.
        out["margin"] = out.home_score - out.away_score
        out["total"] = out.home_score + out.away_score
        out["home_win"] = np.where(out.home_score > out.away_score, 1.0,
                                   np.where(out.home_score < out.away_score, 0.0,
                                            np.nan))
        if "season" not in frame.columns:
            # Deriving the season from the tipoff is a league rule, not a
            # vendor field: the season turns over in July.
            dates = out["gameday"]
            out["season"] = dates.dt.year.where(dates.dt.month >= 7,
                                                dates.dt.year - 1).astype(float)
    return out.reset_index(drop=True)


def coverage_report(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """How much of the contract a frame actually fills, column by column.

    This is the check that a source is honest about itself.  A frame can pass
    ``normalize`` and still be 40% empty, which is fine for an optional column
    and fatal for a required one, and the difference is only visible here.
    """
    schema = SCHEMAS[name]
    required = set(REQUIRED[name])
    rows = []
    for column, kind in schema.items():
        if column not in frame.columns:
            pct, populated = 0.0, 0
        elif kind == "str":
            values = frame[column]
            populated = int(values.notna().sum())
            pct = 100.0 * float((values.astype(str).str.len() > 0).mean()) if len(values) else 0.0
        else:
            values = pd.to_numeric(frame[column], errors="coerce")
            populated = int(values.notna().sum())
            pct = 100.0 * float(values.notna().mean()) if len(values) else 0.0
        rows.append({"frame": name, "column": column, "dtype": kind,
                     "required": column in required,
                     "populated": populated, "coverage_pct": round(pct, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One upstream, and how its frames become contract frames.

    ``normalize`` on the spec is the only thing a caller needs.  Nothing
    downstream of it may reference the vendor's column names, which is the
    entire point: a second source becomes a second ``SourceSpec`` rather than a
    second code path through the model.
    """

    name: str
    route: str
    frames: tuple[str, ...]
    _adapters: dict[str, Callable[..., pd.DataFrame]] = field(
        default_factory=dict, repr=False, compare=False)

    def normalize(self, name: str, frame: pd.DataFrame,
                  **kwargs: Any) -> pd.DataFrame:
        adapter = self._adapters.get(name)
        prepared = adapter(frame) if adapter is not None else frame
        return normalize(prepared, name, source=self.route, **kwargs)

    def can(self, name: str) -> bool:
        return name in self.frames


def make_source(name: str, route: str, adapters: dict[str, Callable[..., pd.DataFrame]],
                frames: tuple[str, ...] | None = None) -> SourceSpec:
    """Build a ``SourceSpec`` from a route name and its per-frame adapters."""
    return SourceSpec(name=name, route=route,
                      frames=frames or tuple(adapters), _adapters=adapters)
