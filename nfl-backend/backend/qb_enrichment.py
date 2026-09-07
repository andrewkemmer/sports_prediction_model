"""QB serving-contract enrichment (spec section 27) — display information
ONLY. QB data never enters a production feature unless explicitly admitted
through the leakage-safe feature-admission process (it is not, today).

Per-side fields: qb_{home,away}_name/_rating/_td_per_game/_cmp_pct/
_yards_per_attempt/_ints. Values are the STARTING QUARTERBACK's per-game
aggregates over his LAST 10 completed regular-season games strictly before
the slate (point-in-time, trailing across the season boundary). The starter
is the ANNOUNCED name from the schedule when published; when the schedule
does not yet carry starter names (early-season slates), the starter is
derived from the trailing player-stats frame itself: the QB with the most
pass attempts over the team's last 10 completed games (the de-facto
starter), named by his most recent start. Never fabricated: no prior
completed games, or a failed pull, yields the missing representation (None).
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

# Trailing window for the serving aggregates (the card's "Rating / TD/g"
# metrics are the starter's LAST 10 completed regular-season games).
LAST10_WINDOW = 10


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


def _last10_aggregates(qb_rows: pd.DataFrame) -> dict | None:
    """Per-game aggregates over a QB's LAST 10 completed regular-season
    games (the rows must already be that player's, strictly prior to the
    slate, newest first or unsorted — sorted here by season/week desc and
    truncated to the window)."""
    if qb_rows is None or not len(qb_rows):
        return None
    rows = qb_rows.copy()
    rows["_wk"] = pd.to_numeric(rows.get("week"), errors="coerce")
    rows["_sn"] = pd.to_numeric(rows.get("season"), errors="coerce")
    rows = rows.sort_values(["_sn", "_wk"], ascending=False).head(LAST10_WINDOW)
    return _aggregate_prior_games(rows)


def _derive_starter(team_rows: pd.DataFrame) -> str | None:
    """De-facto starting QB from a team's trailing game-by-game QB rows: the
    player with the most pass attempts over the window (the team's primary
    starter), named by his most recent start. None when there is no usable
    history."""
    if team_rows is None or not len(team_rows):
        return None
    rows = team_rows.copy()
    rows["_att"] = pd.to_numeric(rows.get("attempts"), errors="coerce").fillna(0)
    by_player = rows.groupby("player_id")["_att"].sum() if "player_id" in rows \
        else rows.groupby("player_name")["_att"].sum()
    if not len(by_player) or by_player.max() <= 0:
        return None
    top_id = by_player.idxmax()
    sel = (rows["player_id"] == top_id) if "player_id" in rows \
        else (rows["player_name"] == top_id)
    started = rows[sel].sort_values(["season", "week"], ascending=False)
    name = started["player_name"].iloc[0]
    return str(name) if isinstance(name, str) and name.strip() else None


def _team_last10_stats(qb_rows: pd.DataFrame) -> pd.DataFrame:
    """A team's QB game rows over its LAST 10 completed games (newest first)."""
    if qb_rows is None or not len(qb_rows):
        return pd.DataFrame()
    rows = qb_rows.copy()
    rows["_wk"] = pd.to_numeric(rows.get("week"), errors="coerce")
    rows["_sn"] = pd.to_numeric(rows.get("season"), errors="coerce")
    return rows.sort_values(["_sn", "_wk"], ascending=False).head(LAST10_WINDOW)


def enrich_slate(slate_df: pd.DataFrame,
                 stats_by_season: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Attach the QB contract fields to a slate frame.

    ``stats_by_season`` maps season -> player-stats frame (injected by the
    pipeline from cached pulls); a missing season degrades to missing
    fields for its games. Matching is by announced starter name (schedule
    ``home_qb_name``/``away_qb_name``) against recorded player_name; when
    the schedule does not yet publish starters, the starter is derived from
    the trailing player-stats frame (most pass attempts over the team's
    last 10 completed games)."""
    out = slate_df.copy()
    for f in _QB_FIELDS:
        out[f] = None

    if stats_by_season is None:
        return out

    # Cross-season trailing window: the last 10 completed regular-season
    # games BEFORE the slate, walking back through prior seasons (a week-1
    # 2026 slate's trailing window is the end of the 2025 season).
    def _prior_window(team: str, slate_season: int, week: int) -> pd.DataFrame:
        frames = []
        for frame_season in sorted(stats_by_season, reverse=True):
            sdf = stats_by_season[frame_season]
            if not len(sdf) or "week" not in sdf.columns:
                continue
            wk = pd.to_numeric(sdf["week"], errors="coerce")
            sn = pd.to_numeric(sdf["season"], errors="coerce")
            tm = sdf.get("team")
            if tm is None:
                continue
            # This team's rows only: slate-season rows strictly before the
            # slate week; prior-season rows in full (all strictly prior).
            m = (tm == team) & (
                ((sn == slate_season) & (wk < week)) | (sn < slate_season))
            sel = sdf[m]
            if len(sel):
                frames.append(sel)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["_wk"] = pd.to_numeric(out.get("week"), errors="coerce")
        out["_sn"] = pd.to_numeric(out.get("season"), errors="coerce")
        return out.sort_values(["_sn", "_wk"], ascending=False)

    def _side_fields(row, side: str) -> dict:
        prefix = f"qb_{side}"
        season = int(row["season"]) if np.isfinite(row.get("season", np.nan)) else None
        team = row.get("home_team" if side == "home" else "away_team")
        week = row.get("week")
        try:
            week = int(week)
        except (TypeError, ValueError):
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}
        if season is None or not stats_by_season:
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}

        window = _prior_window(str(team), season, week)
        if not len(window):
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}

        # Starter resolution: the ANNOUNCED name from the schedule when
        # published; otherwise the de-facto starter from the trailing
        # window (most pass attempts over the team's last 10 games).
        name = row.get(f"{side}_qb_name")
        if not (isinstance(name, str) and name.strip()):
            name = _derive_starter(window)
        if not name:
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}

        # The starter's own rows: match the announced name when given, else
        # the derived starter's id. Both routes take his LAST 10 games.
        if isinstance(row.get(f"{side}_qb_id"), str) and row.get(f"{side}_qb_id"):
            own = window[window.get("player_id").astype(str)
                         == str(row.get(f"{side}_qb_id"))]
        else:
            own = window[window["player_name"] == name]
        if not len(own):
            # Announced name not found in the trailing stats (trade/new
            # starter): fall back to the de-facto starter of the window.
            derived = _derive_starter(window)
            if not derived:
                return {f"{prefix}_name": name,
                        **{f: None for f in _QB_FIELDS
                           if f.startswith(prefix) and f != f"{prefix}_name"}}
            name = derived
            own = window[window["player_name"] == name]
            if not len(own):
                return {f"{prefix}_name": name,
                        **{f: None for f in _QB_FIELDS
                           if f.startswith(prefix) and f != f"{prefix}_name"}}
        agg = _last10_aggregates(own)
        if agg is None:
            return {f"{prefix}_name": name,
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
