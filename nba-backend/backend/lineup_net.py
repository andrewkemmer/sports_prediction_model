"""DARKO 5-man net rating for the projected starting five - a SHADOW test.

Why this is not a model feature yet
-----------------------------------
DARKO publishes the lineup table as a CURRENT snapshot and nothing else:
``?asof=`` is ignored on ``/lineups`` (2026-11-01 and 2025-12-01 return
byte-identical payloads, 313 units, same net ratings). There is therefore no
history to walk forward over, and the whole discipline of the last month -
"score the arm on games it has not seen" - is impossible for this input.

Shipping it straight into the feature set would mean feeding the model a
variable that has never been measured out of sample. So this module computes
it at serve time and records what it would have said, and the value is scored
on games that arrive AFTER the row was written.

The five comes from the production projection
---------------------------------------------
``lineup_projection.project_lineup_features`` already selects the five whose
DPM/RAPM the ``pl_`` features blend; it now carries those ids on the
aggregate as ``lineup_five_ids``. This module joins that exact five to DARKO's
unit table, so the feature describes the lineup the model was actually fed
rather than a nominal starting five.

Matching is deliberately conservative: an exact five wins; otherwise units
sharing at least ``MIN_OVERLAP`` players are blended by possessions; otherwise
the row is left NaN. A "close enough" substitution on a unit that has never
played together is how a synergy feature turns into noise.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get(
    "NBA_LINEUP_NET_LOG",
    os.path.join(os.path.dirname(_HERE), "data_delivery", "nba_lineup_net_log.csv"),
)

MIN_OVERLAP = 4

COLUMNS = [
    "logged_utc", "game_id", "gameday", "home_team", "away_team",
    "home_five", "away_five",
    "net_home", "net_away", "net_diff",
    "off_syn_home", "off_syn_away", "def_syn_home", "def_syn_away",
    "match_home", "match_away", "home_win",
]


def _ids(series) -> tuple[str, ...]:
    return tuple(sorted(str(x) for x in series if str(x)))


def build_index(table: pd.DataFrame) -> tuple[dict, list]:
    """``frozenset(five) -> row`` plus the row list for overlap blending."""
    idx, rows = {}, []
    for row in table.itertuples(index=False):
        key = _ids(str(row.player_ids).split(";"))
        if len(key) == 5:
            idx[frozenset(key)] = row
            rows.append((frozenset(key), row))
    return idx, rows


def _lookup(five: frozenset, idx: dict, rows: list):
    """Exact unit, else a possession-weighted blend of near-identical units."""
    if not five:
        return np.nan, np.nan, np.nan, "none"
    hit = idx.get(five)
    if hit is not None:
        return (float(hit.net), float(hit.off_syn) if pd.notna(hit.off_syn) else np.nan,
                float(hit.def_syn) if pd.notna(hit.def_syn) else np.nan, "exact")
    near = [row for key, row in rows if len(five & key) >= MIN_OVERLAP]
    if not near:
        return np.nan, np.nan, np.nan, "none"
    w = np.array([float(getattr(r, "possessions", 0) or 0) for r in near], dtype=float)
    w = w if w.sum() > 0 else np.ones(len(near))
    net = float(np.average([r.net for r in near], weights=w))
    off = float(np.average([r.off_syn for r in near], weights=w))
    dfn = float(np.average([r.def_syn for r in near], weights=w))
    return net, off, dfn, "overlap"


def attach_lineup_net(slate: pd.DataFrame, aggregates: pd.DataFrame,
                      table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the projected five's DARKO net to each side of the slate.

    ``aggregates`` is the per-(gameday, team) projection the pipeline already
    builds for the ``pl_`` features; only ``lineup_five_ids`` is read.
    """
    if slate is None or not len(slate):
        return slate
    if table is None:
        from . import darko_source

        try:
            table = darko_source.fetch_lineups(size=5)
        except Exception:  # noqa: BLE001 - an unreachable feed prices nothing
            return slate
    if table is None or table.empty:
        return slate
    idx, rows = build_index(table)

    agg = aggregates.copy()
    agg["gameday"] = pd.to_datetime(agg.gameday, errors="coerce").dt.normalize()
    by_key = {}
    for r in agg.itertuples(index=False):
        by_key[(r.gameday, str(r.team))] = _ids(str(getattr(r, "lineup_five_ids", "")).split(";"))

    out = slate.copy()
    day = pd.to_datetime(out.gameday, errors="coerce").dt.normalize()
    res = {k: [] for k in ("home", "away")}
    for i in range(len(out)):
        d = day.iloc[i]
        for side in ("home", "away"):
            team = str(out[f"{side}_team"].iloc[i])
            five = frozenset(by_key.get((d, team), ()))
            res[side].append(_lookup(five, idx, rows))

    out["darko_lineup_net_home"] = [x[0] for x in res["home"]]
    out["darko_lineup_net_away"] = [x[0] for x in res["away"]]
    out["darko_lineup_syn_home"] = [x[1] - x[2] if np.isfinite(x[1]) and np.isfinite(x[2]) else np.nan
                                    for x in res["home"]]
    out["darko_lineup_syn_away"] = [x[1] - x[2] if np.isfinite(x[1]) and np.isfinite(x[2]) else np.nan
                                    for x in res["away"]]
    out["darko_lineup_match_home"] = [x[3] for x in res["home"]]
    out["darko_lineup_match_away"] = [x[3] for x in res["away"]]
    out["darko_lineup_net_diff"] = (out.darko_lineup_net_home
                                    - out.darko_lineup_net_away)
    return out


def log_slate(slate: pd.DataFrame, aggregates: pd.DataFrame,
              table: pd.DataFrame | None = None, path: str = OUT) -> pd.DataFrame:
    """Attach and append to the shadow log, replacing any earlier rows for the slate.

    Never raises: a broken test log must not fail a slate.
    """
    try:
        d = attach_lineup_net(slate, aggregates, table=table)
        rows = pd.DataFrame({
            "logged_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "game_id": d.get("game_id"),
            "gameday": pd.to_datetime(d.gameday, errors="coerce").dt.strftime("%Y-%m-%d"),
            "home_team": d.get("home_team"),
            "away_team": d.get("away_team"),
            "net_home": d.get("darko_lineup_net_home"),
            "net_away": d.get("darko_lineup_net_away"),
            "net_diff": d.get("darko_lineup_net_diff"),
            "off_syn_home": d.get("darko_lineup_syn_home"),
            "off_syn_away": d.get("darko_lineup_syn_away"),
            "match_home": d.get("darko_lineup_match_home"),
            "match_away": d.get("darko_lineup_match_away"),
            "home_win": d.get("home_win", pd.Series([np.nan] * len(d), index=d.index)),
        })
        agg = aggregates.copy()
        agg["gameday"] = pd.to_datetime(agg.gameday, errors="coerce").dt.normalize()
        five = {(r.gameday, str(r.team)): str(getattr(r, "lineup_five_ids", ""))
                for r in agg.itertuples(index=False)}
        day = pd.to_datetime(d.gameday, errors="coerce").dt.normalize()
        rows["home_five"] = [five.get((day.iloc[i], str(d.home_team.iloc[i])), "")
                             for i in range(len(d))]
        rows["away_five"] = [five.get((day.iloc[i], str(d.away_team.iloc[i])), "")
                             for i in range(len(d))]
        rows = rows.reindex(columns=COLUMNS)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            old = pd.read_csv(path, dtype={"game_id": str})
            old = old[~old.game_id.astype(str).isin(set(rows.game_id.astype(str)))]
            rows = pd.concat([old, rows], ignore_index=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        rows.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return rows
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=COLUMNS)


def summary(path: str = OUT) -> pd.DataFrame:
    """Does the lineup net differential explain anything the price did not?"""
    d = pd.read_csv(path, dtype={"game_id": str})
    d = d[d.home_win.notna() & d.net_diff.notna()]
    if d.empty:
        return pd.DataFrame()
    y = d.home_win.to_numpy(float)
    out = {
        "n": len(d),
        "corr_netdiff_win": float(np.corrcoef(d.net_diff, y)[0, 1]),
        "match_exact": float((d.match_home == "exact").mean()),
    }
    if "market_p_home" in d.columns:
        mk = pd.to_numeric(d.market_p_home, errors="coerce")
        ok = mk.notna()
        if ok.sum() > 10:
            resid = y[ok] - mk[ok]
            out["corr_netdiff_market_resid"] = float(np.corrcoef(d.net_diff[ok], resid)[0, 1])
    return pd.DataFrame([out])
