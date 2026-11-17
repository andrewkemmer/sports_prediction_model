"""Shadow logging for candidate rules that are NOT live.

A rule invented on a slice of history is a hypothesis, not a result: scoring it
on the same games that motivated it only restates the hypothesis. This module
records, per slate, what each candidate would have taken, so the rule can be
scored on games that arrive after it was written.

Two candidates are logged:

``tail_platt``
    The piecewise Platt step from the tail-calibration arm. Only rows whose
    incumbent probability is at or above :data:`TAIL_CUT` are recalibrated; the
    bulk is returned untouched. The coefficients are fitted causally from prior
    games, never from the slate being logged.

``side_asym``
    The asymmetric floor suggested by the 3-5% favourite analysis: a market
    underdog or a lower-two-tercile Elo favourite has to clear
    :data:`FAV_EDGE`, while a top-tercile Elo favourite - the cell that carried
    the band's entire loss - has to clear :data:`TOP_ELO_FAV_EDGE`.

The log is append-only and keyed by ``(rule, game_id)``: re-running a slate
replaces that slate's rows rather than duplicating them. Nothing here raises -
a shadow log that breaks the pipeline is worse than no shadow log.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get(
    "NBA_SHADOW_LOG",
    os.path.join(os.path.dirname(_HERE), "data_delivery", "nba_shadow_log.csv"),
)

TAIL_CUT = 0.90
TAIL_MIN_ROWS = 80
FAV_EDGE = 0.05
TOP_ELO_FAV_EDGE = 0.08
TOP_ELO_QUANTILE = 0.67

COLUMNS = [
    "logged_utc", "rule", "game_id", "gameday", "side", "is_favourite",
    "fav_elo", "elo_pct", "p_model", "p_market", "edge", "threshold",
    "taken", "home_win",
]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_tail_platt(y, p, cut: float = TAIL_CUT,
                   min_rows: int = TAIL_MIN_ROWS):
    """Fit ``sigmoid(a*logit(p) + b)`` on the ``p >= cut`` tail only.

    Returns ``(a, b)`` or ``None`` when the tail is too thin to fit, which is
    the normal state early in a season - a caller that gets ``None`` must fall
    back to the raw probability rather than inventing a calibration.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = p >= cut
    if int(m.sum()) < int(min_rows) or len(np.unique(y[m])) < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(p[m]).reshape(-1, 1), y[m])
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_tail_platt(p, coef, cut: float = TAIL_CUT) -> np.ndarray:
    """Recalibrate only the tail; every row below ``cut`` passes through."""
    p = np.asarray(p, dtype=float).copy()
    if coef is None:
        return p
    a, b = coef
    m = p >= cut
    if m.any():
        p[m] = _sigmoid(a * _logit(p[m]) + b)
    return p


def elo_cut(fav_elo, quantile: float = TOP_ELO_QUANTILE) -> float:
    """Elo cutoff separating the top tercile of the logged favourites."""
    v = pd.to_numeric(pd.Series(fav_elo), errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(v, quantile)) if v.size else float("nan")


def required_edge(is_favourite: bool, fav_elo: float, cut: float) -> float:
    """The threshold this pick has to clear under the side-asymmetric rule."""
    if not is_favourite:
        return FAV_EDGE
    if np.isfinite(fav_elo) and np.isfinite(cut) and fav_elo >= cut:
        return TOP_ELO_FAV_EDGE
    return FAV_EDGE


def build_rows(pred: pd.DataFrame, elo_cut_value: float,
               tail_coef=None, logged_utc: str | None = None) -> pd.DataFrame:
    """Turn a scored slate into one row per (rule, game).

    ``pred`` needs ``game_id``, ``gameday``, ``p_model``, ``p_market`` and the
    two Elo columns; ``home_win`` is carried through when it is already known
    so an intraday re-log can score the same row later.
    """
    d = pred.copy()
    p = pd.to_numeric(d["p_model"], errors="coerce").to_numpy(float)
    mk = pd.to_numeric(d["p_market"], errors="coerce").to_numpy(float)
    ok = np.isfinite(p) & np.isfinite(mk)
    home = p > 0.5
    p_pick = np.where(home, p, 1.0 - p)
    mk_pick = np.where(home, mk, 1.0 - mk)
    edge = p_pick - mk_pick
    is_fav = mk_pick > 0.5
    elo_h = pd.to_numeric(d.get("elo_home"), errors="coerce").to_numpy(float)
    elo_a = pd.to_numeric(d.get("elo_away"), errors="coerce").to_numpy(float)
    fav_elo = np.where(home, elo_h, elo_a)
    _s = pd.Series(fav_elo)
    elo_pct = _s.rank(pct=True).to_numpy(float)

    p_tail = apply_tail_platt(p, tail_coef)
    tail_home = p_tail > 0.5
    tail_pick = np.where(tail_home, p_tail, 1.0 - p_tail)
    tail_edge = tail_pick - mk_pick

    thr = np.array([required_edge(bool(f), float(e), elo_cut_value)
                    for f, e in zip(is_fav, fav_elo)])

    base = {
        "logged_utc": logged_utc or pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_id": d["game_id"].astype(str).to_numpy(),
        "gameday": pd.to_datetime(d["gameday"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy(),
        "side": np.where(home, "home", "away"),
        "is_favourite": is_fav,
        "fav_elo": fav_elo,
        "elo_pct": elo_pct,
        "p_model": p_pick,
        "p_market": mk_pick,
        "edge": edge,
        "home_win": (pd.to_numeric(d["home_win"], errors="coerce").to_numpy(float)
                     if "home_win" in d else np.nan),
    }
    out = []
    for rule, taken, e, t in (
        ("tail_platt", (tail_home == home) & (tail_edge >= FAV_EDGE), tail_edge, FAV_EDGE),
        ("side_asym", edge >= thr, edge, thr),
    ):
        r = dict(base)
        r.update(rule=rule, edge=e, threshold=t, taken=np.asarray(taken) & ok)
        out.append(pd.DataFrame(r))
    return pd.concat(out, ignore_index=True)[COLUMNS]


def log_slate(pred: pd.DataFrame, path: str = OUT, elo_cut_value: float | None = None,
              tail_coef=None) -> pd.DataFrame:
    """Append a slate to the shadow log, replacing any earlier rows for it.

    Returns the rows that were written. Never raises.
    """
    try:
        cut = elo_cut_value if elo_cut_value is not None else elo_cut(
            pred.get("elo_home", pd.Series(dtype=float)))
        new = build_rows(pred, cut, tail_coef=tail_coef)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            old = pd.read_csv(path, dtype={"game_id": str})
            key = set(zip(new.rule, new.game_id))
            old = old[[k not in key for k in zip(old.rule, old.game_id.astype(str))]]
            new = pd.concat([old, new], ignore_index=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        new.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return new
    except Exception:  # noqa: BLE001 - shadow logging must never break a slate
        return pd.DataFrame(columns=COLUMNS)


def summary(path: str = OUT) -> pd.DataFrame:
    """Per-rule hit rate, claimed edge and realised edge over the log."""
    d = pd.read_csv(path, dtype={"game_id": str})
    d = d[d.taken.astype(bool) & d.home_win.notna()]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for rule, part in d.groupby("rule"):
        won = np.where(part.side == "home", part.home_win, 1 - part.home_win)
        real = won.mean() - part.p_market.mean()
        rows.append({
            "rule": rule, "n": len(part), "claim": part.edge.mean(),
            "observed": won.mean(), "p_market": part.p_market.mean(),
            "realised": real, "retention": real / part.edge.mean() if part.edge.mean() else np.nan,
        })
    return pd.DataFrame(rows)
