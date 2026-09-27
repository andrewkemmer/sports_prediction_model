"""Player-level NHL offense ratings — the EVO/PPO analogue of MLB's wOBA.

MLB shrinks each batter's rolling wOBA toward a LEAGUE-WIDE mean with a flat
120-PA prior (``mlb-backend/backend/features.py`` -> ``batter_ratings``)::

    shrunk_woba = (woba_num + lg_woba * 120) / (ab + 120)

That single global prior is defensible for baseball because every batter faces
the same opportunity: a plate appearance. Hockey skaters do NOT share an
opportunity, and the measured 2025 spread is severe (individual xG per 60, from
MoneyPuck season summaries):

    situation   position   league rate   play share   mean season ice time
    5on5 (EVO)  C              0.681         29.6%         39,014 s
    5on5 (EVO)  L              0.748         15.4%         36,818 s
    5on5 (EVO)  R              0.724         15.0%         38,058 s
    5on5 (EVO)  D              0.185         40.0%         48,546 s
    5on4 (PPO)  C              1.747         36.6%          4,462 s
    5on4 (PPO)  L              1.698         19.6%          4,243 s
    5on4 (PPO)  R              1.637         21.8%          4,897 s
    5on4 (PPO)  D              0.667         22.1%          2,491 s

A defenceman generates ~25% of a winger's even-strength rate. Shrinking every
skater toward one league mean would over-rate every defenceman by ~2.7x and
under-rate every winger by ~0.7x. So the prior is SEGMENTED BY POSITION, and
the prior strength follows MLB's own convention carried over per position:
MLB's 120 PA is exactly 20% of a ~600-PA season, so here

    k[position, situation] = SHRINK_FRACTION_OF_SEASON * mean season ice time

which puts ~17% prior weight on a player at his own average season
(``k / (k + n) = 0.2 / 1.2``) — the same 20%-of-a-season intent, expressed in ice
time instead of plate appearances.

GRAIN. MoneyPuck publishes regular-season skater game-by-game archives with
individual xGoals and seconds of ice time. Each rating row is a player's
trailing 30 played games in one situation, ending at that source game; it can
only enter a target game's pool when its source game date is STRICTLY EARLIER
than the target date. This last-date gate is essential: the rating includes its
source game's stats, but the current target game's row (and even its lineup
identity) can never enter its own pool. Calendar-date-only source timestamps
are treated conservatively; same-date source rows are not eligible.

Shrinkage and the position/situation league rate use only historical games and
completed prior seasons. Missing pre-history remains unknown rather than being
filled from a full-history constant. Rates are xGoals per 60 minutes (3,600
seconds), not per minute.

Production injury handling is a BINARY player-pool exclusion applied after
ratings are built, in ``injury_stints.team_game_rates``. An unavailable
player's rolling rating is retained unchanged for prior games; exact Out,
Injured Reserve, IR, and Doubtful snapshots remove the candidate only from a
target game's pool when captured strictly before puck drop.

This module is PURE: it takes frames and returns frames, with no network or
filesystem access. ``ingestion.py`` owns the pulls.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Situations
# ---------------------------------------------------------------------------
# MoneyPuck's long-format ``situation`` values. 5on5 is even strength; 5on4 is
# the standard power play. 4on5 (short-handed) and 'other'/'all' are excluded by
# construction: a short-handed rate would need its own prior, and 'all' double
# counts the two splits.
SITUATION_EVO = "5on5"
SITUATION_PPO = "5on4"
SITUATIONS: tuple[str, ...] = (SITUATION_EVO, SITUATION_PPO)
SITUATION_LABELS = {SITUATION_EVO: "EVO", SITUATION_PPO: "PPO"}

# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------
# MoneyPuck's native skater ``position`` taxonomy. Kept as-is (single letters)
# rather than folded into F/D, because the whole point of the position prior is
# that C/L/R/D do not share a mean.
POSITIONS: tuple[str, ...] = ("C", "L", "R", "D")

_POSITION_ALIASES = {
    "c": "C", "l": "L", "r": "R", "d": "D",
    "center": "C", "centre": "C", "left": "L", "right": "R",
    "defense": "D", "defence": "D", "defenceman": "D", "defenseman": "D",
}

# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------
SHRINK_FRACTION_OF_SEASON = 0.20
ROLLING_ROWS = 30                # MLB lineup-wOBA analogue: trailing 30 played games
SECONDS_PER_HOUR = 3_600.0      # MoneyPuck icetime is measured in seconds

# -- Prepared-input contract (what the engine consumes) ---------------------
IN_PLAYER = "player_id"
IN_NAME = "player_name"
IN_TEAM = "team"
IN_GAME_DATE = "game_date"
IN_GAME_ID = "game_id"
IN_POSITION = "position"
IN_SITUATION = "situation"
IN_XG = "xg"
IN_ICE = "ice_seconds"
IN_SEASON = "season"
IN_POOL_DATE = "pool_date"

# -- Derived rolling-window columns; the selected candidate date must precede
#    the target game date, making every included raw game strictly prior. -----
PRIOR_XG = "prior_xg"
PRIOR_ICE = "prior_ice_seconds"
PRIOR_ROWS = "prior_rows"

# -- Final output columns ----------------------------------------------------
OUT_METRIC = "metric"
OUT_LEAGUE_RATE = "league_rate_per60"
OUT_K_SECONDS = "k_seconds"
OUT_RAW_RATE = "raw_rate_per60"
OUT_SHRUNK_RATE = "shrunk_rate_per60"
OUT_INJURY_MULTIPLIER = "injury_multiplier"
OUT_AVAILABILITY = "availability_status"
OUT_FINAL_RATE = "final_rate_per60"

OUTPUT_COLUMNS = [
    IN_PLAYER, IN_NAME, IN_TEAM, IN_GAME_DATE, IN_POOL_DATE, IN_GAME_ID,
    IN_POSITION, IN_SITUATION, OUT_METRIC, PRIOR_ROWS,
    PRIOR_XG, PRIOR_ICE, OUT_LEAGUE_RATE, OUT_K_SECONDS,
    OUT_RAW_RATE, OUT_SHRUNK_RATE, OUT_INJURY_MULTIPLIER,
    OUT_AVAILABILITY, OUT_FINAL_RATE,
]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def normalize_position(value: object) -> Optional[str]:
    """Map a MoneyPuck/ESPN position code to ``C``/``L``/``R``/``D``.

    Returns ``None`` for anything unrecognized so the caller can count and drop
    it. Guessing would be silently harmful: an unknown code folded into ``D``
    inherits the lowest prior of the four.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return _POSITION_ALIASES.get(text)


def _season_ids(dates: pd.Series) -> pd.Series:
    """NHL season start year for regular-season dates (October through April)."""
    parsed = pd.to_datetime(dates, errors="coerce")
    years = parsed.dt.year
    return (years - parsed.dt.month.lt(7).astype("Int64")).astype("Int64")


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def prepare_player_games(
    games: pd.DataFrame,
    *,
    player_col: str = "name",
    date_col: str = "game_date",
    game_col: str = "game_id",
    team_col: str = "team",
    position_col: str = "position",
    situation_col: str = "situation",
    xg_col: str = "xg",
    ice_col: str = "ice_seconds",
    id_col: Optional[str] = None,
    season_col: str = "season",
) -> tuple[pd.DataFrame, dict]:
    """Normalize a long-format player frame into the engine's input contract.

    Returns the cleaned frame plus an audit dict. The audit exists because every
    way this step can quietly lose data — an unknown position code, a name that
    never resolved to an id, a zero-ice-time row — produces a SMALLER,
    plausible-looking output rather than an error.
    """
    audit = {
        "rows_in": 0, "rows_out": 0,
        "dropped_unknown_position": 0, "dropped_no_player_id": 0,
        "dropped_bad_situation": 0, "dropped_bad_values": 0,
        "dropped_duplicate": 0,
    }
    if games is None or len(games) == 0:
        return pd.DataFrame(columns=[
            IN_PLAYER, IN_NAME, IN_TEAM, IN_GAME_DATE, IN_GAME_ID,
            IN_POSITION, IN_SITUATION, IN_XG, IN_ICE, IN_SEASON]), audit

    df = games.copy()
    audit["rows_in"] = len(df)

    rename = {
        player_col: IN_NAME, date_col: IN_GAME_DATE, game_col: IN_GAME_ID,
        team_col: IN_TEAM, position_col: IN_POSITION,
        situation_col: IN_SITUATION, xg_col: IN_XG, ice_col: IN_ICE,
        season_col: IN_SEASON,
    }
    if id_col:
        rename[id_col] = IN_PLAYER
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if IN_PLAYER not in df.columns:
        # No id column: fall back to the player NAME as identity. Dropping every
        # row instead would return an empty frame that reads as "no data" rather
        # than "no id requested", with a 100%-loss audit and no clue why.
        if IN_NAME in df.columns:
            df[IN_PLAYER] = df[IN_NAME].astype(str)
        else:
            df[IN_PLAYER] = pd.NA

    for col in (IN_XG, IN_ICE):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[IN_GAME_DATE] = pd.to_datetime(df[IN_GAME_DATE], errors="coerce").dt.normalize()
    # Date grain is the authoritative NHL-season key. MoneyPuck labels seasons
    # by start year, but accepting a mismatched source label would silently
    # move a game into the wrong prior-season k table.
    df[IN_SEASON] = _season_ids(df[IN_GAME_DATE])

    bad_sit = ~df[IN_SITUATION].isin(SITUATIONS)
    audit["dropped_bad_situation"] = int(bad_sit.sum())
    df = df[~bad_sit]

    pos = df[IN_POSITION].map(normalize_position)
    unknown = pos.isna()
    audit["dropped_unknown_position"] = int(unknown.sum())
    df = df[~unknown].assign(**{IN_POSITION: pos})

    ids = df[IN_PLAYER]
    no_id = ids.isna() | (ids.astype(str).str.strip() == "") | (ids.astype(str) == "nan")
    audit["dropped_no_player_id"] = int(no_id.sum())
    df = df[~no_id]
    df[IN_PLAYER] = df[IN_PLAYER].astype(str)

    # Keep zero-ice situation rows: a skater may play the game without any
    # time in one situation, and that appearance still occupies one of the
    # trailing N played-game slots for that situation.
    bad_vals = (df[IN_XG].isna() | df[IN_ICE].isna()
                | (df[IN_ICE] < 0) | df[IN_GAME_DATE].isna())
    audit["dropped_bad_values"] = int(bad_vals.sum())
    df = df[~bad_vals]

    # One row per player x game x situation. Counted explicitly: an uncounted
    # dedup is a silent row loss, and a player appearing twice for one game
    # (traded mid-game, or re-listed under a changed position) then vanishes
    # from the audit entirely.
    before_dedup = len(df)
    df = (df.sort_values([IN_PLAYER, IN_GAME_DATE, IN_GAME_ID], kind="mergesort")
            .drop_duplicates([IN_PLAYER, IN_GAME_ID, IN_SITUATION], keep="last")
            .reset_index(drop=True))
    audit["dropped_duplicate"] = int(before_dedup - len(df))

    audit["rows_out"] = int(len(df))
    return df, audit


# ---------------------------------------------------------------------------
# Prior tables
# ---------------------------------------------------------------------------
def season_ice_time_table(
    games: pd.DataFrame,
    *,
    as_of: object,
    shrink_fraction: float = SHRINK_FRACTION_OF_SEASON,
    participation_floor_seconds: float = 0.0,
) -> dict[tuple[str, str], float]:
    """Return ``k[position, situation]`` from completed prior seasons only.

    The averaging unit is a PLAYER-SEASON, not a player. Each eligible prior
    season contributes its total situation-specific ice time once; the season
    containing ``as_of`` is excluded in full. ``as_of`` is mandatory so
    current- or later-season aggregates cannot tune an earlier rating.

    ``participation_floor_seconds`` optionally drops player-seasons below a
    threshold. Cells without strictly prior completed-season evidence are
    omitted; no future-season constant is substituted.
    """
    table: dict[tuple[str, str], float] = {}
    if games is None or len(games) == 0:
        return table

    cutoff = pd.to_datetime(as_of, errors="coerce", utc=True)
    if pd.isna(cutoff):
        raise ValueError(f"season_ice_time_table requires a valid as_of date: {as_of!r}")
    cutoff = pd.Timestamp(cutoff).tz_localize(None).normalize()
    dates = pd.to_datetime(games[IN_GAME_DATE], errors="coerce").dt.normalize()

    target_season = int(_season_ids(pd.Series([cutoff])).iloc[0])
    source_seasons = _season_ids(dates)
    eligible = games.loc[(dates < cutoff)
                         & source_seasons.notna()
                         & (source_seasons < target_season)]
    if eligible.empty:
        return table

    work = eligible[[IN_PLAYER, IN_POSITION, IN_SITUATION, IN_ICE]].copy()
    work["_period"] = source_seasons.loc[eligible.index].astype(str)

    keys = [IN_PLAYER, IN_POSITION, IN_SITUATION]
    per_season = (work.groupby(keys + ["_period"], dropna=False)[IN_ICE]
                       .sum().reset_index())
    if participation_floor_seconds > 0:
        per_season = per_season[
            per_season[IN_ICE] >= float(participation_floor_seconds)]

    per_season = per_season[per_season[IN_ICE] > 0]
    cell = (per_season.groupby([IN_POSITION, IN_SITUATION])[IN_ICE]
                     .mean().reset_index())
    for _, row in cell.iterrows():
        pos = normalize_position(row[IN_POSITION])
        sit = row[IN_SITUATION]
        value = float(row[IN_ICE])
        if pos is None or sit not in SITUATIONS:
            continue
        if not np.isfinite(value) or value <= 0:
            continue
        table[(sit, pos)] = shrink_fraction * value
    return table


def _rolling_player_window(
    df: pd.DataFrame,
    *,
    window: int = ROLLING_ROWS,
) -> pd.DataFrame:
    """Add a trailing ``window``-game sum per player and situation.

    These are post-game candidate ratings: the current source game's stats are
    included, then the game-grid join requires the candidate's source date to
    be STRICTLY BEFORE the target date. That separation keeps the latest
    completed player game in the target's window without allowing a target
    game's own row or lineup participation to leak in.

    Rolling is grouped by (player, situation), never just player, and includes
    the current source row plus at most ``window - 1`` earlier played games.
    ``transform`` preserves source row alignment across interleaved groups.
    """
    if int(window) < 1:
        raise ValueError("window must be at least one played game")
    out = df.sort_values(
        [IN_PLAYER, IN_SITUATION, IN_GAME_DATE, IN_GAME_ID],
        kind="mergesort").reset_index(drop=True)
    by_player_situation = [IN_PLAYER, IN_SITUATION]
    out[PRIOR_XG] = out.groupby(by_player_situation, sort=False)[IN_XG].transform(
        lambda s: s.rolling(max(1, int(window)), min_periods=1).sum())
    out[PRIOR_ICE] = out.groupby(by_player_situation, sort=False)[IN_ICE].transform(
        lambda s: s.rolling(max(1, int(window)), min_periods=1).sum())
    out[PRIOR_ROWS] = out.groupby(by_player_situation, sort=False)[IN_XG].transform(
        lambda s: s.notna().astype(int).rolling(max(1, int(window)), min_periods=1).sum()
    ).astype(int)
    # Keep the real source date. Pool expansion requires it to be strictly
    # earlier than the target date; do not synthesize an availability date.
    out[IN_POOL_DATE] = out[IN_GAME_DATE]
    return out


def league_prior_table(
    df: pd.DataFrame,
    *,
    fallback_rate: Mapping[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Point-in-time (date, position, situation) league rate per 60 minutes.

    The daily source-game rows are accumulated by date, then the entire current
    date is subtracted. A value keyed to date D therefore uses only earlier
    dates; same-day games cannot enter the prior.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[IN_GAME_DATE, IN_POSITION, IN_SITUATION,
                                     OUT_LEAGUE_RATE])

    daily = (df.groupby([IN_GAME_DATE, IN_POSITION, IN_SITUATION], dropna=False)
               [[IN_XG, IN_ICE]].sum().reset_index())
    daily = daily.sort_values(IN_GAME_DATE, kind="mergesort")
    keys = [IN_POSITION, IN_SITUATION]
    cumulative_xg = daily.groupby(keys, dropna=False)[IN_XG].cumsum()
    cumulative_ice = daily.groupby(keys, dropna=False)[IN_ICE].cumsum()
    # Strictly prior league evidence: exclude the entire candidate source date,
    # since daily data does not establish within-day event/publish ordering.
    daily["_cxg"] = cumulative_xg - daily[IN_XG]
    daily["_cice"] = cumulative_ice - daily[IN_ICE]
    with np.errstate(divide="ignore", invalid="ignore"):
        daily[OUT_LEAGUE_RATE] = np.where(
            daily["_cice"] > 0,
            daily["_cxg"] / daily["_cice"] * SECONDS_PER_HOUR, np.nan)

    # No full-history constant is safe for a historical date. A cell with no
    # earlier league evidence remains unknown rather than borrowing from later
    # seasons.
    fb = dict(fallback_rate or {})
    daily[OUT_LEAGUE_RATE] = [
        _fallback_rate(fb, sit, pos, val)
        for sit, pos, val in zip(daily[IN_SITUATION], daily[IN_POSITION],
                                 daily[OUT_LEAGUE_RATE])
    ]
    return (daily[[IN_GAME_DATE, IN_POSITION, IN_SITUATION, OUT_LEAGUE_RATE]]
            .reset_index(drop=True))


def _fallback_rate(
    fallback: Mapping[tuple[str, str], float],
    situation: object,
    position: object,
    observed: object,
) -> float:
    """Observed strictly prior rate, else an explicitly supplied PIT fallback."""
    try:
        val = float(observed)
    except (TypeError, ValueError):
        val = float("nan")
    if np.isfinite(val) and val > 0:
        return val
    pos = normalize_position(position)
    if pos is None:
        return float("nan")
    got = fallback.get((str(situation), pos))
    return float(got) if got is not None else float("nan")


def shrink_rate(
    prior_xg: float,
    prior_ice_seconds: float,
    league_rate_per60: float,
    k_seconds: float,
) -> float:
    """Conjugate (normal-normal) shrinkage of a rate toward a league prior.

        shrunk = (prior_xg + mu_ice * k) / (prior_ice + k)

    where ``mu_ice = league_rate_per60 / 3600`` converts the per-hour prior back
    into xG per second, so the pseudo-count ``mu_ice * k`` is a genuine xG total
    rather than a rate pasted next to a count. This is
    MLB's formula with ``120`` generalized to a position- and
    situation-specific ``k``.

    Returns xG per SECOND of ice time; callers scale to xG per 60 minutes.
    """
    xg = float(prior_xg)
    ice = float(prior_ice_seconds)
    mu60 = float(league_rate_per60)
    k = float(k_seconds)
    if not (np.isfinite(xg) and np.isfinite(ice) and np.isfinite(mu60) and np.isfinite(k)):
        return float("nan")
    denom = ice + k
    if denom <= 0:
        return float("nan")
    return (xg + (mu60 / SECONDS_PER_HOUR) * k) / denom


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def build_player_ratings(
    games: pd.DataFrame,
    *,
    injuries: Optional[pd.DataFrame] = None,
    window: int = ROLLING_ROWS,
    shrink_fraction: float = SHRINK_FRACTION_OF_SEASON,
    id_col: Optional[str] = None,
    as_of: Optional[object] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build per-player EVO/PPO ratings.

    ``games`` is a player/game/situation frame (see
    ``prepare_player_games``). Each output row records a trailing rate ending
    with that source game; the production game pool only makes it eligible for
    targets on a strictly later date. Injury inputs are accepted for backwards
    compatibility but never alter a player's rating: availability removes the
    player from the target game's aggregation only.

    League rates are built from strictly earlier calendar dates and shrinkage
    strength uses completed seasons before the source game's season. ``as_of``
    drops raw game rows on or after that date.
    """
    prepared, audit = prepare_player_games(games, id_col=id_col)
    audit["window_rows"] = int(window)
    audit["shrink_fraction_of_season"] = float(shrink_fraction)

    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if len(prepared) == 0:
        audit["players"] = 0
        audit["rows"] = 0
        return empty, audit

    if as_of is not None:
        cutoff = pd.to_datetime(as_of, errors="coerce", utc=True)
        if pd.isna(cutoff):
            raise ValueError(f"build_player_ratings requires a valid as_of date: {as_of!r}")
        cutoff = pd.Timestamp(cutoff).tz_localize(None)
        prepared = prepared[prepared[IN_GAME_DATE] < cutoff].reset_index(drop=True)
        audit["rows_after_as_of"] = int(len(prepared))
        if len(prepared) == 0:
            audit["players"] = 0
            audit["rows"] = 0
            return empty, audit

    rolled = _rolling_player_window(prepared, window=window)
    rating_seasons = sorted(
        int(value) for value in _season_ids(rolled[IN_GAME_DATE]).dropna().unique())
    k_by_season = {
        season: season_ice_time_table(
            prepared, as_of=f"{season}-10-01", shrink_fraction=shrink_fraction)
        for season in rating_seasons
    }
    k_table = k_by_season[rating_seasons[-1]] if rating_seasons else {}
    league = league_prior_table(rolled)

    merged = rolled.merge(league, on=[IN_GAME_DATE, IN_POSITION, IN_SITUATION],
                          how="left", validate="many_to_one")
    row_seasons = _season_ids(merged[IN_GAME_DATE])
    merged[OUT_K_SECONDS] = [
        float(k_by_season.get(int(season), {}).get((sit, pos), float("nan")))
        if pd.notna(season) else float("nan")
        for season, sit, pos in zip(row_seasons, merged[IN_SITUATION],
                                    merged[IN_POSITION])
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        merged[OUT_RAW_RATE] = np.where(
            merged[PRIOR_ICE] > 0,
            merged[PRIOR_XG] / merged[PRIOR_ICE] * SECONDS_PER_HOUR, np.nan)
    merged[OUT_SHRUNK_RATE] = [
        shrink_rate(x, i, m, k) * SECONDS_PER_HOUR
        for x, i, m, k in zip(merged[PRIOR_XG], merged[PRIOR_ICE],
                              merged[OUT_LEAGUE_RATE], merged[OUT_K_SECONDS])
    ]
    merged[OUT_METRIC] = merged[IN_SITUATION].map(SITUATION_LABELS)
    # Injury does not modify a historical game rating; its row-level status is
    # intentionally neutral. The exact exclusion happens later, once per target
    # game, in injury_stints.team_game_rates.
    merged[OUT_INJURY_MULTIPLIER] = 1.0
    merged[OUT_AVAILABILITY] = "healthy"
    merged[OUT_FINAL_RATE] = merged[OUT_SHRUNK_RATE]

    audit["injuries_rows"] = 0 if injuries is None else int(len(injuries))
    audit["injuries_applied"] = (
        "no_pool_exclusion_only" if audit["injuries_rows"] else "no")

    out = merged[[c for c in OUTPUT_COLUMNS if c in merged.columns]]
    out = (out.sort_values([IN_PLAYER, IN_GAME_DATE, IN_SITUATION], kind="mergesort")
              .reset_index(drop=True))

    audit["players"] = int(out[IN_PLAYER].nunique())
    audit["rows"] = int(len(out))
    audit["k_table"] = {f"{sit}/{pos}": round(v, 1)
                        for (sit, pos), v in sorted(k_table.items())}
    return out, audit


def latest_ratings(
    ratings: pd.DataFrame,
    *,
    on_or_before: Optional[object] = None,
) -> pd.DataFrame:
    """Most recent rating per (player, situation) strictly before a decision date.

    The serving shape: one EVO and one PPO number per player, with the row that
    produced each named, so the age of the number is never ambiguous. A rating
    stamped on the decision date is excluded because date-grain data cannot
    prove that its events were available before puck drop.
    """
    if ratings is None or len(ratings) == 0:
        return ratings.copy() if ratings is not None else pd.DataFrame()
    df = ratings.copy()
    if on_or_before is not None:
        df = df[df[IN_GAME_DATE] < pd.to_datetime(on_or_before)]
    if len(df) == 0:
        return df.reset_index(drop=True)
    idx = df.groupby([IN_PLAYER, IN_SITUATION])[IN_GAME_DATE].idxmax()
    return (df.loc[idx].sort_values([IN_PLAYER, IN_SITUATION], kind="mergesort")
              .reset_index(drop=True))
