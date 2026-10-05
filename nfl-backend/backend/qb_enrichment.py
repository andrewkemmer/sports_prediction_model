"""QB serving-contract enrichment (spec section 27) — display information
ONLY. QB data never enters a production feature unless explicitly admitted
through the leakage-safe feature-admission process (it is not, today).

Per-side fields: qb_{home,away}_name/_rating/_td_per_game/_cmp_pct/
_yards_per_attempt/_ints. Values are the STARTING QUARTERBACK's per-game
aggregates over his CURRENT-SEASON-TO-DATE completed regular-season games
strictly before the slate (season-partitioned, point-in-time — the MLB
sp_era/sp_k9 card-lookback structure: a prior-season row never leaks, and
so a week-1 slate has no completed games and renders MLB's honest missing
representation rather than last season's numbers). The starter is the
ANNOUNCED name from the schedule when published; when the schedule does
not yet carry starter names (early-season slates), the starter is derived
from the same season-to-date frame: the QB with the most pass attempts
(the de-facto starter), named by his most recent start. Never fabricated:
no prior completed games, or a failed pull, yields the missing
representation (None).
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


def _season_to_date_aggregates(qb_rows: pd.DataFrame) -> dict | None:
    """Per-game aggregates over a QB's CURRENT-SEASON completed games (the
    rows must already be that player's, strictly prior to the slate — the
    season-to-date window `_prior_window` selected; sorted by season/week
    desc for a deterministic first row)."""
    if qb_rows is None or not len(qb_rows):
        return None
    rows = qb_rows.copy()
    rows["_wk"] = pd.to_numeric(rows.get("week"), errors="coerce")
    rows["_sn"] = pd.to_numeric(rows.get("season"), errors="coerce")
    rows = rows.sort_values(["_sn", "_wk"], ascending=False)
    return _aggregate_prior_games(rows)


def _derive_starter(team_rows: pd.DataFrame) -> str | None:
    """De-facto starting QB from a team's season-to-date game-by-game QB
    rows: the player with the most pass attempts in the window (the team's
    primary starter), named by his most recent start. None when there is no
    usable history."""
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


def enrich_slate(slate_df: pd.DataFrame,
                 stats_by_season: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Attach the QB contract fields to a slate frame.

    ``stats_by_season`` maps season -> player-stats frame (injected by the
    pipeline from cached pulls); a missing season degrades to missing
    fields for its games. Matching is by announced starter name (schedule
    ``home_qb_name``/``away_qb_name``) against recorded player_name; when
    the schedule does not yet publish starters, the starter is derived from
    the season-to-date frame (most pass attempts over the team's completed
    games this season)."""
    out = slate_df.copy()
    for f in _QB_FIELDS:
        out[f] = None

    if stats_by_season is None:
        return out

    # Season-to-date window (MLB sp_era/sp_k9 card parity): this team's
    # completed REGULAR-SEASON games of the SLATE'S OWN season strictly
    # before the slate week. Prior seasons never enter the card, and a
    # postseason row never leaks in — a week-1 slate therefore has an empty
    # window and renders the missing representation, exactly like MLB's
    # unresolved pitcher.
    def _prior_window(team: str, slate_season: int, week: int) -> pd.DataFrame:
        sdf = stats_by_season.get(slate_season)
        if sdf is None or not len(sdf) or "week" not in sdf.columns:
            return pd.DataFrame()
        wk = pd.to_numeric(sdf["week"], errors="coerce")
        sn = pd.to_numeric(sdf["season"], errors="coerce")
        tm = sdf.get("team")
        if tm is None:
            return pd.DataFrame()
        m = (tm == team) & (sn == slate_season) & (wk < week)
        if "season_type" in sdf.columns:
            m = m & (sdf["season_type"].astype(str).str.upper() == "REG")
        sel = sdf[m]
        if not len(sel):
            return pd.DataFrame()
        out = sel.copy()
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
        # published; otherwise the de-facto starter from the season-to-date
        # window (most pass attempts over the team's games this season).
        name = row.get(f"{side}_qb_name")
        if not (isinstance(name, str) and name.strip()):
            name = _derive_starter(window)
        if not name:
            return {f: None for f in _QB_FIELDS if f"qb_{side}" in f}

        # The starter's own rows: match the announced name when given, else
        # the derived starter's id. Both routes take his season-to-date games.
        if isinstance(row.get(f"{side}_qb_id"), str) and row.get(f"{side}_qb_id"):
            own = window[window.get("player_id").astype(str)
                         == str(row.get(f"{side}_qb_id"))]
        else:
            own = window[window["player_name"] == name]
        if not len(own):
            # Announced name not found in the season-to-date stats (trade/new
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
        agg = _season_to_date_aggregates(own)
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
