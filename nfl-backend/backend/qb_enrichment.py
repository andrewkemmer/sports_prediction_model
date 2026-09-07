"""QB serving-contract enrichment (spec section 27) — display information
ONLY. QB data never enters a production feature unless explicitly admitted
through the leakage-safe feature-admission process (it is not, today).

Per-side fields: qb_{home,away}_name/_rating/_td_per_game/_cmp_pct/
_yards_per_attempt/_ints. Values are the ANNOUNCED starter's season-to-date
per-game aggregates over completed regular-season games strictly before the
slate (point-in-time). Never fabricated: an unknown starter, a first start
of the season, or a failed pull yields the missing representation (None).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

_QB_FIELDS = config.QB_FIELDS


def _season_qb_aggregates(seasons: list[int]) -> pd.DataFrame | None:
    """(season, team) -> the week-by-week QB stats frame for completed games."""
    try:
        from nflreadpy import load_player_stats
    except Exception:
        return None
    frames = []
    for season in seasons:
        try:
            ps = load_player_stats(season)
            df = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
        except Exception as exc:  # noqa: BLE001
            logger.warning("player stats pull failed for %s: %s", season, exc)
            continue
        if "position" in df.columns:
            df = df[df["position"] == "QB"]
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    if "season_type" in out.columns:
        out = out[out["season_type"] == "REG"]
    return out


def _passer_rating(row) -> float | None:
    """NFL passer rating from a single-game stat line; None when incomplete."""
    try:
        att = float(row["attempts"])
        cmp_ = float(row["completions"])
        yds = float(row["passing_yards"])
        td = float(row["passing_tds"])
        it = float(row["passing_interceptions"])
    except (KeyError, TypeError, ValueError):
        return None
    if att <= 0:
        return None
    a = np.clip((cmp_ / att - 0.3) * 5.0, 0.0, 2.375)
    b = np.clip((yds / att - 3.0) * 0.25, 0.0, 2.375)
    c = np.clip((td / att) * 20.0, 0.0, 2.375)
    d = np.clip(2.375 - (it / att) * 25.0, 0.0, 2.375)
    return float((a + b + c + d) / 6.0 * 100.0)


def _aggregate_prior_games(qb_rows: pd.DataFrame) -> dict | None:
    """Season-to-date per-game aggregates for one QB's prior games."""
    if qb_rows is None or not len(qb_rows):
        return None
    n = len(qb_rows)
    if n == 0:
        return None
    att = qb_rows["attempts"].astype(float).sum()
    cmp_ = qb_rows["completions"].astype(float).sum()
    yds = qb_rows["passing_yards"].astype(float).sum()
    td = qb_rows["passing_tds"].astype(float).sum()
    it = qb_rows["passing_interceptions"].astype(float).sum()
    if not np.isfinite(att) or att <= 0:
        return None
    # single-game passer ratings averaged (the standard season rating)
    ratings = [_passer_rating(r) for r in qb_rows.to_dict("records")]
    ratings = [r for r in ratings if r is not None]
    return {
        "name": str(qb_rows["player_name"].iloc[0]),
        "rating": float(np.mean(ratings)) if ratings else None,
        "td_per_game": float(td / n),
        "cmp_pct": float(100.0 * cmp_ / att),
        "yards_per_attempt": float(yds / att),
        "ints": float(it),
        "n_games": int(n),
    }


def enrich_slate(slate_df: pd.DataFrame,
                 stats_by_season: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Attach the QB contract fields to a slate frame.

    ``stats_by_season`` maps season -> player-stats frame (injected by the
    pipeline from cached pulls); a missing season degrades to missing
    fields for its games. Matching is by announced starter name (schedule
    ``home_qb_name``/``away_qb_name``) against recorded player_name.
    """
    out = slate_df.copy()
    for f in _QB_FIELDS:
        out[f] = None

    if stats_by_season is None:
        return out

    def _side_fields(row, side: str) -> dict:
        name_col = f"{side}_qb_name"
        name = row.get(name_col)
        if not isinstance(name, str) or not name.strip():
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}
        season = int(row["season"]) if np.isfinite(row.get("season", np.nan)) else None
        if season is None or season not in stats_by_season:
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}
        season_df = stats_by_season[season]
        team = row.get("home_team" if side == "home" else "away_team")
        # Point-in-time: completed regular-season games of the SAME season
        # with week STRICTLY BEFORE the slate game's week (the player-stats
        # frame is week-grain; no future week can enter a prior aggregate).
        week = row.get("week")
        try:
            week = int(week)
        except (TypeError, ValueError):
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}
        if "week" not in season_df.columns:
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}
        prior = season_df[(season_df["team"] == team)
                          & (season_df["player_name"] == name)
                          & (pd.to_numeric(season_df["week"], errors="coerce") < week)]
        agg = _aggregate_prior_games(prior)
        prefix = f"qb_{side}"
        if agg is None:
            return {f"{prefix}_name": name,  # announced but no prior stats
                    **{f: None for f in _QB_FIELDS
                       if f.startswith(prefix) and f != f"{prefix}_name"}}
        return {
            f"{prefix}_name": agg["name"],
            f"{prefix}_rating": agg["rating"],
            f"{prefix}_td_per_game": agg["td_per_game"],
            f"{prefix}_cmp_pct": agg["cmp_pct"],
            f"{prefix}_yards_per_attempt": agg["yards_per_attempt"],
            f"{prefix}_ints": agg["ints"],
        }

    enriched = []
    for _, row in out.iterrows():
        fields = {}
        fields.update(_side_fields(row, "home"))
        fields.update(_side_fields(row, "away"))
        enriched.append(fields)
    if enriched:
        for f in _QB_FIELDS:
            out[f] = [e.get(f) for e in enriched]
    return out
