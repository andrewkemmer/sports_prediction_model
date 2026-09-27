"""NBA totals/point-spread diagnostics over run-engine artifacts."""
from __future__ import annotations

import io
import math
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import utils

TOTAL_GRID = list(range(180, 281))
SPREAD_GRID = list(range(-20, 21))
HALF_STOP_LINES = [-0.5, 0.5]
# MLB's five diagnostics tabs, labels byte-identical. "Pooled lines" is
# lower-case l on the MLB page; the previous NBA capital-L was a typo the
# local render comparison surfaced.
DIAG_TABS = ("Distribution", "Relativized", "Pooled lines", "Game Total Lines", "Spread Lines")
OFFSET_EDGES = (-10.0, -5.0, 0.0, 5.0, 10.0)

# --- Prediction-history + monitor-card contracts (MLB ``market_diagnostics``)
# MLB-shaped thresholds and line choices. The SPREAD choices are restricted to
# lines the run-engine actually prices: the artifact carries integer columns
# -20..20 plus the ±0.5 half stops and NOTHING in between, so a -1.5 column
# does not exist and must never be requested.
TOTALS_CONF_THRESHOLDS = [50, 51, 52, 53, 54, 55]
SPREAD_LINE_CHOICES = [-0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -7.0, -10.0]
# Magnitudes the priced grid can serve (integer steps + the 0.5 half stop).
SPREAD_GRID_CUT = [0.5] + [float(v) for v in range(1, 21)]
# The three winner cards — MLB's labels and rules with NBA wording.
WINNER_CARDS = (
    ("over_under", "Over/Under",
     "Pick Over if P(over the game's line) > 50%, else Under"),
    ("run_line", "Point Spread (home line)",
     "Pick Home if P(home covers the game's own fair line) > 50%, else Away"),
    ("derived_ml", "Derived ML (run-line model moneyline)",
     "Pick the side with P > 50% — home if P(home win) > 50%, else away "
     "(derived from the paired score distribution; distinct from the "
     "moneyline ensemble)"),
)
_HISTORY_COLUMNS = ["game_date", "away", "home", "home_score", "away_score",
                    "line", "pick", "pick_prob", "winner", "correct"]


def _label(value: float) -> str:
    text = str(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace(".", "_")


def _col(base: str, value: float) -> str:
    return f"{base}_{_label(value)}"


def decided_rows(markets: pd.DataFrame | None) -> pd.DataFrame:
    if markets is None or not len(markets):
        return pd.DataFrame()
    frame = markets.copy()
    if "kind" in frame:
        kind_mask = frame.kind.eq("oof")
        if "decided" in frame:
            kind_mask = kind_mask | frame.decided.eq(True)
        frame = frame[kind_mask].copy()
    elif "decided" in frame:
        frame = frame[frame.decided.eq(True)].copy()
    required = [c for c in ("home_score", "away_score", "total", "margin") if c in frame]
    if len(required) < 2:
        return pd.DataFrame()
    if "total" not in frame:
        frame["total"] = pd.to_numeric(frame.home_score, errors="coerce") + pd.to_numeric(frame.away_score, errors="coerce")
    if "margin" not in frame:
        frame["margin"] = pd.to_numeric(frame.home_score, errors="coerce") - pd.to_numeric(frame.away_score, errors="coerce")
    return frame[frame.total.notna() & frame.margin.notna()].reset_index(drop=True)


def _num(value) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _grid_p_over(frame: pd.DataFrame, line: float) -> np.ndarray:
    values = []
    for _, row in frame.iterrows():
        p = _num(row.get(_col("p_over", line)))
        if p is None:
            values.append(np.nan)
        else:
            values.append(p)
    return np.asarray(values, dtype=float)


def _line_value(row, base: str, line: float):
    return _num(row.get(_col(base, line)))


def _pairs(decided: pd.DataFrame, line: float, kind: str) -> dict[str, Any]:
    rows = []
    for _, row in decided.iterrows():
        actual = float(row.total) if kind == "total" else float(row.margin)
        expected = _num(row.get("fair_total" if kind == "total" else "fair_spread"))
        p = _line_value(row, "p_over" if kind == "total" else "p_home_cover", line)
        if p is None or expected is None:
            continue
        if kind == "total":
            target = 1.0 if actual > line else (0.5 if actual == line else 0.0)
        else:
            target = 1.0 if actual > line else (0.5 if actual == line else 0.0)
        rows.append({"predicted": p, "actual": target, "line": line,
                     "offset": float(line) - expected, "date": row.get("gameday")})
    return {"pairs": rows, "n_pairs": len(rows), "warning": None if rows else "No valid priced pairs."}


def total_distribution(decided: pd.DataFrame, kmax: int = 300) -> dict[str, Any]:
    if decided is None or not len(decided):
        return {"warning": "No decided NBA games available for distribution diagnostics.", "n_games": 0, "callouts": {}}
    totals = pd.to_numeric(decided.total, errors="coerce").dropna().astype(int)
    observed = totals[(totals >= 0) & (totals <= kmax)].value_counts().sort_index()
    return {"warning": None, "n_games": int(len(totals)),
            "observed": observed, "callouts": {"P(total<=1)": {"observed": float((totals <= 1).mean())},
                                                 "P(total>=250)": {"observed": float((totals >= 250).mean())}}}


def relativized_pairs(decided: pd.DataFrame, lines=None) -> dict[str, Any]:
    lines = lines or (205, 210, 215, 220, 225, 230, 235, 240)
    pairs = []
    for line in lines:
        pairs.extend(_pairs(decided, line, "total")["pairs"])
    return {"pairs": pairs, "n_pairs": len(pairs), "warning": None if pairs else "No total calibration pairs available."}


def fixed_line_pairs(decided: pd.DataFrame, lines=(220, 225, 230)) -> dict[str, Any]:
    pairs = []
    for line in lines:
        pairs.extend(_pairs(decided, line, "total")["pairs"])
    return {"pairs": pairs, "n_pairs": len(pairs), "warning": None if pairs else "No fixed-line pairs available."}


def _calibration_curve(pair_dict: dict[str, Any], title: str = "Calibration") -> dict[str, Any]:
    pairs = pair_dict.get("pairs", [])
    if not pairs:
        return {"warning": pair_dict.get("warning", "No calibration pairs."), "bins": [], "n_pairs": 0}
    frame = pd.DataFrame(pairs)
    frame = frame[np.isfinite(frame.predicted) & np.isfinite(frame.actual)]
    if frame.empty:
        return {"warning": "No finite calibration pairs.", "bins": [], "n_pairs": 0}
    frame["bin"] = np.clip((frame.predicted * 10).astype(int), 0, 9)
    bins = []
    for key, group in frame.groupby("bin", sort=True):
        bins.append({"mean_pred": float(group.predicted.mean()),
                     "mean_actual": float(group.actual.mean()), "n": int(len(group))})
    return {"warning": None, "bins": bins, "n_pairs": int(len(frame)),
            "n_dropped_bins": 0, "title": title}


def calibration_curve(pair_dict: dict[str, Any], title: str = "Calibration") -> dict[str, Any]:
    return _calibration_curve(pair_dict, title)


def game_total_calibration(decided: pd.DataFrame, line: float | None = None) -> dict[str, Any]:
    if line is None:
        line = _num((decided.get("fair_total") if decided is not None else None).median()) if decided is not None and len(decided) else None
    return _calibration_curve(_pairs(decided, line, "total") if line is not None else {"pairs": []}, "Game total calibration")


def run_line_calibration(decided: pd.DataFrame, magnitude: float = 2.0) -> dict[str, Any]:
    threshold = float(magnitude)
    return _calibration_curve(_pairs(decided, threshold, "spread"), "Point spread calibration")


def _card_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], np.clip(p[ok], 1e-7, 1 - 1e-7)
    if not len(y):
        return {"n": 0, "brier": np.nan, "logloss": np.nan, "ece": np.nan}
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(float((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()))
              for b in range(10) if (bins == b).any())
    return {"n": int(len(y)), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "ece": float(ece)}


# ---------------------------------------------------------------------------
# Prediction-history frames + monitor cards — the MLB ``market_diagnostics``
# contracts, restated over the NBA run-engine artifact.
#
# Two NBA-specific facts drive every design choice below, and both were read
# off the artifact rather than assumed:
#
#   1. The spread grid is INTEGER (-20..20) plus the ±0.5 half stops only.
#      There is no -1.5 column. Because NBA margins are integers, every
#      integer line can therefore PUSH, where MLB's -1.5 run line never can.
#      So the run-line card here uses MLB's 3-way push resolution (push rows
#      render "—" and drop out of W/(W+L)), not MLB's never-push path.
#   2. ``calibration_cards`` ships a prequential Platt (a, b) per line, so
#      ECE-calibrated probabilities are computable here exactly as they are
#      on the MLB page — read-only, no Monte-Carlo re-run.
# ---------------------------------------------------------------------------


def _int_line(value) -> int | None:
    """The integer run-engine column index for a spread/total line."""
    v = _num(value)
    return None if v is None else int(round(v))


def _spread_label(line: int) -> str:
    """Column suffix for a spread grid index (MLB's ``m``-prefixed labels)."""
    if line == 0:
        return "0"
    return ("m" + str(-line)) if line < 0 else str(line)


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Mann-Whitney AUC with tie-averaged ranks (None if a class is absent)."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok].astype(float), p[ok].astype(float)
    n = len(y)
    if n < 2:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    seq = np.arange(1, n + 1, dtype=float)
    sorted_p = p[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = seq[i:j + 1].mean()
        i = j + 1
    pos = y == 1.0
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return None
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over fixed-width probability bins."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if not len(y):
        return float("nan")
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(float((idx == b).mean())
                     * abs(float(p[idx == b].mean()) - float(y[idx == b].mean()))
                     for b in range(bins) if (idx == b).any()))


def _platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """σ(a·logit(p) + b) — the artifact's own prequential calibration map."""
    eps = 1e-7
    q = np.clip(p.astype(float), eps, 1 - eps)
    return 1.0 / (1.0 + np.exp(-(a * np.log(q / (1 - q)) + b)))


def calibration_coefficients(monitor: dict | None, family: str,
                             line) -> tuple[float, float] | None:
    """(a, b) from ``calibration_cards`` for one line, or None when absent."""
    if line is None:
        return None
    cards = (monitor or {}).get("calibration_cards") or {}
    entry = (cards.get(family) or {}).get(str(int(line)))
    if not isinstance(entry, dict):
        return None
    primary = entry.get("over") or entry.get("home")
    if not isinstance(primary, dict) or "a" not in primary or "b" not in primary:
        return None
    a, b = _num(primary.get("a")), _num(primary.get("b"))
    return None if a is None or b is None else (a, b)


def binary_card_metrics(y: np.ndarray, p: np.ndarray,
                        coeff: tuple[float, float] | None = None) -> dict:
    """The pooled metric block a winner card renders.

    Mirrors MLB's card field-for-field: win_rate / auc / brier / logloss /
    ece_raw / ece_calibrated / predicted_mean / actual_win_rate / n. With no
    calibration map shipped, ece_calibrated is None rather than a copy of the
    raw value — an unavailable calibrated number must not read as a good one.
    """
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok].astype(float), np.clip(p[ok].astype(float), 1e-7, 1 - 1e-7)
    n = int(len(y))
    if n == 0:
        return {"n": 0, "win_rate": None, "auc": None, "brier": None,
                "logloss": None, "ece_raw": None, "ece_calibrated": None,
                "predicted_mean": None, "actual_win_rate": None}
    pred = _platt(p, *coeff) if coeff else p
    # Win rate is the CONDITIONAL accuracy of the pick the model actually
    # made: correct / (correct + wrong) over rows with a real side lean.
    # A joint "p>0.5 AND correct" rate would understate by every game whose
    # outcome went against a >50% side, which is most of them. Rows at
    # exactly p=0.5 express no lean, so no pick was made and they are
    # excluded from both sides — the same W/(W+L) convention the monitor
    # cards and the history tables use.
    leaned = np.abs(p - 0.5) > 1e-12
    correct = (p > 0.5) == (y == 1.0)
    win_rate = (round(float(correct[leaned].mean()), 6)
                if leaned.any() else None)
    return {
        "n": n,
        "win_rate": win_rate,
        "auc": _auc(y, pred),
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "logloss": round(float(-np.mean(y * np.log(pred)
                                         + (1 - y) * np.log(1 - pred))), 6),
        "ece_raw": round(_ece(y, p), 6),
        "ece_calibrated": round(_ece(y, pred), 6) if coeff else None,
        "predicted_mean": round(float(p.mean()), 6),
        "actual_win_rate": round(float(y.mean()), 6),
    }


# --- History frames --------------------------------------------------------
# Each history frame carries MLB's display fields:
#   game_date, away, home, line, pick, pick_prob, winner, correct
# ``correct`` is True / False / NaN(push) — the 3-way convention.


def _row_date(row) -> Any:
    for key in ("gameday", "game_date"):
        if key in row:
            return row.get(key)
    return None


def totals_history_frame(decided: pd.DataFrame) -> pd.DataFrame:
    """Game-totals history priced at each game's OWN fair total.

    Mirrors MLB's ``totals_history_frame``: LINE is the game's own line, the
    pick is Over/Under on P > 50%, and a total exactly equal to the line is a
    PUSH (correct = NaN, excluded from the win rate) because the artifact
    prices whole-number totals with a real push bucket. P comes from the line
    grid — see ``_over_fair_series`` for why ``p_over_fair`` is not used.
    """
    rows: list[dict] = []
    if decided is None or not len(decided):
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    for _, r in decided.iterrows():
        fair = _num(r.get("fair_total"))
        total = _num(r.get("total"))
        if fair is None or total is None:
            continue
        line = int(round(fair))
        p = _num(r.get(f"p_over_{line}"))
        if p is None:
            p = _num(r.get("p_over_fair"))
        if p is None:
            continue
        if _num(r.get("home_score")) is None or _num(r.get("away_score")) is None:
            # A decided row with no score cannot be shown honestly — the
            # SCORE (A–H) column would read an em-dash beside a real line.
            # Skip the row instead of rendering a half-populated record.
            continue
        over = p > 0.5
        pick = "Over" if over else "Under"
        if total == line:
            winner, correct = "Push", np.nan
        else:
            # The winner cell names the SIDE that won the bet, so the RESULT
            # glyph is a plain pick-vs-winner comparison.
            winner = pick if ((total > line) == over) else ("Under" if over else "Over")
            correct = bool(winner == pick)
        rows.append({"game_date": _row_date(r),
                     "away": str(r.get("away_team", "—") or "—"),
                     "home": str(r.get("home_team", "—") or "—"),
                     "home_score": _num(r.get("home_score")),
                     "away_score": _num(r.get("away_score")),
                     "line": float(line), "pick": pick,
                     "pick_prob": float(p), "winner": winner,
                     "correct": correct})
    return pd.DataFrame(rows, columns=_HISTORY_COLUMNS)


def runline_history_frame(decided: pd.DataFrame) -> pd.DataFrame:
    """Point-spread history priced at each game's OWN fair line.

    Unlike MLB's -1.5 run line, the NBA grid is integer (-20..20) plus ±0.5,
    so a whole-number line CAN push. Resolution is therefore MLB's 3-way
    convention: the pick covers when margin > line, a margin exactly equal to
    the line is a PUSH (correct = NaN, excluded from W/(W+L)), and anything
    less is a loss.
    """
    rows: list[dict] = []
    if decided is None or not len(decided):
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    for _, r in decided.iterrows():
        fair = _num(r.get("fair_spread"))
        margin = _num(r.get("margin"))
        line = _int_line(fair)
        if margin is None or line is None:
            continue
        label = _spread_label(line)
        p_home = _num(r.get(f"p_home_cover_{label}"))
        if p_home is None:
            continue
        if _num(r.get("home_score")) is None or _num(r.get("away_score")) is None:
            continue
        # 2-way re-normalization: the push bucket is neither side, so it is
        # folded out of both before the pick and the displayed probability.
        p_push = _num(r.get(f"p_push_{label}")) or 0.0
        p_away = max(0.0, 1.0 - p_home - p_push)
        two_way = p_home + p_away
        p_home_2 = (p_home / two_way) if two_way > 0 else 0.5
        away, home = (str(r.get("away_team", "—") or "—"),
                      str(r.get("home_team", "—") or "—"))
        pick = home if p_home_2 > 0.5 else away
        if margin == line:
            winner, correct = "Push", np.nan
        else:
            home_covers = margin > line
            winner = home if home_covers else away
            correct = bool((p_home_2 > 0.5) == home_covers)
        rows.append({"game_date": _row_date(r), "away": away, "home": home,
                     "home_score": _num(r.get("home_score")),
                     "away_score": _num(r.get("away_score")),
                     "line": float(line), "pick": pick,
                     "pick_prob": float(p_home_2), "winner": winner,
                     "correct": correct})
    return pd.DataFrame(rows, columns=_HISTORY_COLUMNS)


def filter_history_frame(frame: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """Rows whose game_date falls inside [start_date, end_date], inclusive."""
    if frame is None:
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    if not len(frame) or "game_date" not in frame.columns:
        return frame.copy()
    dts = pd.to_datetime(frame["game_date"], errors="coerce").dt.date
    keep = dts.notna() & (dts >= start_date) & (dts <= end_date)
    return frame[keep].reset_index(drop=True)


def filter_history_by_side(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    """Partition a totals history by the model's pick side.

    Without this, Over under-predicting and Under over-predicting net to zero
    pooled — the Scoring-Mean diagnostic MLB's caption names.
    """
    s = (str(side) or "All").strip()
    if frame is None:
        return pd.DataFrame()
    if not len(frame) or s == "All" or "pick" not in frame.columns:
        return frame.copy()
    return frame[frame["pick"].astype(str) == s].reset_index(drop=True)


def history_win_rate(frame: pd.DataFrame) -> dict:
    """n_games (non-push denominator), pooled win rate, and the push count.

    Push rows carry correct = NaN and are excluded from BOTH numerator and
    denominator, exactly as on the MLB page.
    """
    if frame is None or not len(frame) or "correct" not in frame.columns:
        return {"n_games": 0, "win_rate": None, "n_pushes": 0}
    ok = frame["correct"].notna()
    n = int(ok.sum())
    wins = float(frame.loc[ok, "correct"].astype(bool).sum()) if n else 0.0
    n_pushes = int((frame["winner"] == "Push").sum()) \
        if "winner" in frame.columns else 0
    return {"n_games": n,
            "win_rate": (round(wins / n, 6) if n else None),
            "n_pushes": n_pushes}


# --- Monitor calibration cards --------------------------------------------


def _three_way(p_primary: float, p_push: float) -> tuple[float, float]:
    """Split a 3-way book into a 2-way normalized (primary, opposite)."""
    other = max(0.0, 1.0 - p_primary - p_push)
    two_way = p_primary + other
    if two_way <= 0:
        return 0.5, 0.5
    return p_primary / two_way, other / two_way


def _side_block(side_view: pd.DataFrame) -> dict:
    n = int(side_view["correct"].notna().sum())
    n_wins = int(side_view.loc[side_view["correct"].notna(), "correct"]
                 .astype(bool).sum()) if n else 0
    return {"n": n, "n_pushes": int((side_view["winner"] == "Push").sum()),
            "win_rate": round(n_wins / n, 6) if n else None}


def _empty_card() -> dict:
    return {"n": 0, "n_wins": 0, "n_losses": 0, "n_pushes": 0,
            "predicted_2way": None, "win_rate": None, "sides": {}}


def _roll_up(view: pd.DataFrame, primary_pick: str) -> dict:
    """W/(W+L) roll-up for a 3-way card view, with pushes folded out.

    ``predicted_2way`` is the PICK's own probability averaged over non-push
    rows, so the displayed prediction and the displayed win rate always share
    a basis and a whole-line card is never read as a large miss.
    """
    ok = view["correct"].notna()
    n = int(ok.sum())
    n_wins = int(view.loc[ok, "correct"].astype(bool).sum()) if n else 0
    n_losses = n - n_wins
    probs = np.where(view["pick"] == primary_pick, view["pick_prob"],
                     1.0 - view["pick_prob"])
    predicted = float(probs[ok.to_numpy()].mean()) if n else None
    return {"n": n, "n_wins": n_wins, "n_losses": n_losses,
            "n_pushes": int((view["winner"] == "Push").sum()),
            "predicted_2way": round(predicted, 6) if n else None,
            "win_rate": round(n_wins / n, 6) if n else None,
            "sides": {}}


def totals_monitor_stats(decided: pd.DataFrame, min_pct: int = 50,
                         side: str = "All") -> dict:
    """Pooled win-rate calibration for the totals card.

    Picks Over/Under at each game's own rounded total line, keeps games whose
    pick probability clears ``min_pct`` (cumulative — a nested subset), and
    applies the optional side filter. Win rate is W/(W+L): whole-line pushes
    are excluded from both numerator and denominator and folded out of the
    2-way displayed prediction.
    """
    view = filter_history_by_side(totals_history_frame(decided), side)
    if not len(view):
        return _empty_card()
    if min_pct and min_pct > 50:
        favoured = np.maximum(view["pick_prob"], 1.0 - view["pick_prob"])
        view = view[favoured * 100.0 > float(min_pct)]
    if not len(view):
        return _empty_card()
    out = _roll_up(view, "Over")
    if side == "All":
        out["sides"] = {s: _side_block(view[view["pick"] == s])
                        for s in ("Over", "Under")}
    return out


def runline_monitor_stats(decided: pd.DataFrame, magnitude: float) -> dict:
    """Point-spread card at a fixed line, priced as a 2-way book.

    ``magnitude`` is the favorite's line. Because the NBA grid is integer, a
    whole-number line pushes, so this uses the same 3-way resolution and the
    same W/(W+L) 2-way re-normalization as the totals card.
    """
    mag = abs(float(magnitude))
    grid = -0.5 if mag < 1.0 else -int(round(mag))
    if abs(grid) not in SPREAD_GRID_CUT:
        return _empty_card()
    label = _spread_label(int(grid))
    rows: list[dict] = []
    if decided is None or not len(decided):
        return _empty_card()
    for _, r in decided.iterrows():
        margin = _num(r.get("margin"))
        p_home = _num(r.get(f"p_home_cover_{label}"))
        if margin is None or p_home is None:
            continue
        p_push = _num(r.get(f"p_push_{label}")) or 0.0
        p_home_2, _ = _three_way(p_home, p_push)
        if margin == grid:
            correct, winner = np.nan, "Push"
        else:
            home_covers = margin > grid
            correct = bool((p_home_2 > 0.5) == home_covers)
            winner = "Home" if home_covers else "Away"
        rows.append({"pick": "Home" if p_home_2 > 0.5 else "Away",
                     "pick_prob": p_home_2, "winner": winner,
                     "correct": correct})
    if not rows:
        return _empty_card()
    out = _roll_up(pd.DataFrame(rows), "Home")
    out["sides"] = {s: _side_block(pd.DataFrame(rows).query("pick == @s"))
                    for s in ("Home", "Away")}
    return out


def load_run_engine_csv(ds: str, prefix: str) -> pd.DataFrame | None:
    """Load an NBA-prefixed run-engine drift/coverage CSV."""
    filename = f"{prefix}_{str(ds).replace('-', '')}.csv"
    raw, _ = utils._fetch_bytes(filename, **utils.get_source_config(), sport="nba")
    if raw is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None


def _cell(value) -> str:
    """Render a cell value, mapping missing/NaN to the em-dash placeholder."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return str(value)


def render_run_engine_drift(frame: pd.DataFrame | None) -> None:
    """Run-engine feature drift as MLB's PSI fb-box table.

    Same markup contract as ``markets._render_run_engine_drift``: the same
    five columns, the same status pill classes, and the same PSI colour rule
    (ALERT red / WARN amber / else text). The MODEL WEIGHT column is omitted
    entirely when no weight data is available — MLB's own ``has_weights``
    gate — so the table still renders without the moneyline monitor artifact.
    """
    st.markdown("### Run-Engine Feature Drift (PSI)")
    if frame is None or frame.empty:
        st.info("No run-engine drift data for this date "
                "(nba_run_engine_feature_drift_*.csv appears after a "
                "pipeline run).")
        return
    records = frame.to_dict("records")
    rows = []
    for r in records:
        psi = _num(r.get("psi"))
        psi_str = "—" if psi is None else f"{psi:.3f}"
        status = r.get("status", "OK")
        psi_color = (utils.RED if status == "ALERT"
                     else utils.AMBER if status == "WARN" else "#E2E8F0")
        pill_cls = {"OK": "ok", "WARN": "warn", "ALERT": "alert",
                    "INSUFFICIENT": "ok"}.get(status, "ok")
        n_base, n_cur = r.get("n_baseline"), r.get("n_current")
        samples = (f" ({n_base}/{n_cur})"
                   if n_base is not None and n_cur is not None else "")
        label = utils.describe_feature(r.get("feature", "")) \
            or r.get("feature", "")
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{r.get('feature','')}"
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;"
            f"margin-top:1px;'>{label}</div></td>"
            f"<td>{_cell(r.get('current_mean'))}</td>"
            f"<td>{_cell(r.get('baseline_mean'))}</td>"
            f"<td style='color:{psi_color};font-weight:700;'>{psi_str}</td>"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span>"
            f"<span style='color:#64748B;font-size:0.72rem;margin-left:5px;'>"
            f"{samples}</span></td></tr>")
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>CURRENT MEAN</th><th>BASELINE MEAN</th>
            <th>PSI</th><th>STATUS</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          Statuses are on noise-adjusted PSI: PSI &lt; 0.10 stable,
          0.10-0.25 moderate shift, &gt; 0.25 material.
          INSUFFICIENT = window too small to judge drift. The per-feature
          MODEL WEIGHT column of the MLB page is omitted here because the NBA
          moneyline monitor ships no feature-drift weights for the run
          engine's feature names.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_engine_coverage(frame: pd.DataFrame | None) -> None:
    """Run-engine feature coverage as MLB's measured/non-null fb-box table.

    Sorted by % measured ascending (worst coverage first) and, when any
    window is starved or low, capped at 12 healthy rows with an explicit
    count of what was hidden — the same disclosure MLB makes, so the table is
    never quietly truncated.
    """
    st.markdown("### Run-Engine Feature Coverage (non-null / measured)")
    if frame is None or frame.empty:
        st.info("No run-engine coverage data for this date "
                "(nba_run_engine_feature_coverage_*.csv appears after a "
                "pipeline run).")
        return
    cov_sorted = sorted(frame.to_dict("records"),
                        key=lambda r: (_num(r.get("pct_measured")) or 0.0,
                                       str(r.get("feature", ""))))
    n_starved = sum(1 for r in cov_sorted if r.get("status") == "STARVED")
    n_low = sum(1 for r in cov_sorted if r.get("status") == "LOW_COVERAGE")
    sub = (
        f"<span style='color:{utils.RED};font-weight:700;'>{n_starved} "
        f"starved</span> · <span style='color:{utils.AMBER};font-weight:700;'>"
        f"{n_low} low</span>"
        if (n_starved or n_low) else
        "<span style='color:#4ADE80;font-weight:700;'>all windows healthy"
        "</span>")
    st.markdown(
        f"<div style='color:#94A3B8;font-size:0.8rem;margin:-6px 0 10px;'>"
        f"Share of games in each drift window with a real observation per "
        f"feature — {sub}</div>",
        unsafe_allow_html=True)
    cap = 12 if (n_starved or n_low) else None
    rows, shown = [], 0
    for r in cov_sorted:
        status = r.get("status", "OK")
        if cap is not None and status == "OK" and shown >= cap:
            continue
        pct_m = _num(r.get("pct_measured")) or 0.0
        pct_n = _num(r.get("pct_nonnull")) or 0.0
        n_def = int(_num(r.get("n_default_zero")) or 0)
        color = (utils.RED if status == "STARVED"
                 else utils.AMBER if status == "LOW_COVERAGE" else "#E2E8F0")
        pill_cls = {"OK": "ok", "LOW_COVERAGE": "warn",
                    "STARVED": "alert"}.get(status, "ok")
        default_cell = (
            f"<div style='color:#94A3B8;font-size:0.72rem;font-weight:400;"
            f"margin-top:1px;'>{n_def} default-zero</div>" if n_def else "")
        rows.append(
            f"<tr>"
            f"<td style='color:#E2E8F0;'>{r.get('feature','')}</td>"
            f"<td>{_cell(r.get('window'))}</td>"
            f"<td>{_cell(r.get('n_games'))}</td>"
            f"<td style='color:{color};font-weight:700;'>{pct_m:.0f}%</td>"
            f"<td>{pct_n:.0f}%{default_cell}</td>"
            f"<td><span class='fb-status-pill {pill_cls}'>{status}</span></td>"
            f"</tr>")
        shown += 1
    n_hidden = len(cov_sorted) - shown
    st.markdown(
        f"""
        <div class="fb-box" style="padding:6px 8px;">
          <table class="fb-table">
            <thead><tr><th>FEATURE</th><th>WINDOW</th><th>GAMES</th>
            <th>% MEASURED</th><th>% NON-NULL</th><th>STATUS</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:6px;">
          % MEASURED = real observations only (default-filled values excluded);
          % NON-NULL includes them. STARVED &lt;25% measured,
          LOW_COVERAGE &lt;80%.
          {f"{n_hidden} healthy feature-window pairs hidden." if n_hidden > 0 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fold_slate_history(monitors: list[dict]) -> dict[str, list[dict]]:
    folded: dict[str, list[dict]] = {}
    for monitor in monitors or []:
        for row in monitor.get("slate_history", []) or []:
            key = str(row.get("game_id", "unknown"))
            folded.setdefault(key, []).append(row)
    return {key: rows[-10:] for key, rows in folded.items()}


def chart_calibration(curve: dict[str, Any], title: str = "Calibration") -> Any:
    bins = curve.get("bins", []) if isinstance(curve, dict) else []
    if not bins:
        return None
    frame = pd.DataFrame(bins)
    return alt.Chart(frame).mark_line().encode(
        x="mean_pred:Q", y="mean_actual:Q",
        tooltip=["mean_pred:Q", "mean_actual:Q", "n:Q"]).properties(title=title)


def chart_distribution(distribution: dict[str, Any]) -> Any:
    observed = distribution.get("observed") if isinstance(distribution, dict) else None
    if observed is None or not len(observed):
        return None
    frame = observed.rename_axis("total").reset_index(name="games")
    return alt.Chart(frame).mark_bar().encode(x="total:O", y="games:Q").properties(title="NBA totals distribution")


# --- Winner cards + rolling history ----------------------------------------


def _over_fair_series(decided: pd.DataFrame) -> pd.Series:
    """P(over the game's own fair total), priced from the LINE GRID.

    The artifact's ``p_over_fair`` column is NOT usable: across 2,142 rows it
    spans only 0.48575-0.51450 and averages 0.500015, so any metric built on
    it returns the no-information values (Brier 0.2499, logloss 0.6897). The
    grid column ``p_over_{rounded fair_total}`` has 2,126 distinct values,
    a mean of 0.530 and Brier 0.2458, and orders its bins correctly. So the
    grid is the primary source and ``p_over_fair`` is only a last-resort
    fallback for artifacts that ship no grid.
    """
    values: list[float] = []
    for _, r in decided.iterrows():
        p = None
        line = _int_line(r.get("fair_total"))
        if line is not None:
            p = _num(r.get(f"p_over_{line}"))
        if p is None:
            p = _num(r.get("p_over_fair"))
        values.append(float("nan") if p is None else p)
    return pd.Series(values, index=decided.index, dtype=float)


def winner_cards(decided: pd.DataFrame,
                 monitor: dict | None = None) -> dict:
    """The three binary winner cards in MLB's schema, computed here.

    The shipped ``run_engine_monitor_*.json`` winner_cards carry only
    {n, brier, logloss, ece} — no win rate, AUC, predicted mean or holdout —
    and in the 20260926 artifact both cards report brier 0.25 / logloss
    0.6931, i.e. exactly the no-information values. So the cards are computed
    from the decided rows rather than read, and the thin artifact block is
    never presented as a richer measurement than it is.
    """
    empty = {"n": 0, "win_rate": None, "auc": None, "brier": None,
             "logloss": None, "ece_raw": None, "ece_calibrated": None,
             "predicted_mean": None, "actual_win_rate": None}
    if decided is None or not len(decided):
        return {key: dict(empty) for key, _, _ in WINNER_CARDS}

    total = pd.to_numeric(decided.get("total"), errors="coerce")
    fair_total = pd.to_numeric(decided.get("fair_total"), errors="coerce")
    over_fair = _over_fair_series(decided)
    ok = total.notna() & fair_total.notna() & over_fair.notna()
    over = binary_card_metrics(
        (total[ok] > fair_total[ok]).to_numpy(float), over_fair[ok].to_numpy(float),
        calibration_coefficients(monitor, "totals",
                                 _int_line(fair_total[ok].median())
                                 if ok.any() else None))

    # The spread card is framed on the PICK, matching what its caption claims:
    # y = 1 when the pick covered, p = the pick's own probability.
    rl = runline_history_frame(decided)
    spread = dict(empty)
    if len(rl):
        okr = rl["correct"].notna().to_numpy()
        picked_home = (rl["pick"] == rl["home"]).to_numpy()
        p_pick = np.asarray(
            np.where(picked_home, rl["pick_prob"], 1.0 - rl["pick_prob"]),
            dtype=float)
        # y = 1 when the PICK covered, whichever side that is. The away branch
        # must read "away covered" (= home did not), not a constant False —
        # a constant there silently credits every away cover as a loss.
        covered = np.asarray(
            np.where(picked_home, rl["winner"] == rl["home"],
                     rl["winner"] != rl["home"]),
            dtype=float)
        y = np.where(okr, covered, np.nan)
        spread = binary_card_metrics(
            y[okr], p_pick[okr],
            calibration_coefficients(
                monitor, "run_line",
                _int_line(rl.loc[rl["correct"].notna(), "line"].median())
                if okr.any() else None))
        # Framed on the pick, p_pick is >50% by construction, so the generic
        # "beat the 50% side" win rate would read 1.0. The card's win rate is
        # the share of picks that covered — i.e. mean(y).
        if okr.any():
            spread["win_rate"] = round(float(y[okr].mean()), 6)

    margin = pd.to_numeric(decided.get("margin"), errors="coerce")
    home_p = pd.to_numeric(decided.get("p_home_win_derived"), errors="coerce")
    okh = margin.notna() & home_p.notna()
    derived = binary_card_metrics(
        (margin[okh] > 0).to_numpy(float), home_p[okh].to_numpy(float))
    return {"over_under": over, "run_line": spread, "derived_ml": derived}


def rolling_history_rows(rolling: dict) -> list[dict]:
    """MLB's rolling-history table rows: Line, Date, ECE-cal, Brier,
    Logloss, Pred mean, n — the last 10 points per card, in card order."""
    rows: list[dict] = []
    for key, label, _rule in WINNER_CARDS:
        for pt in ((rolling or {}).get(key) or [])[-10:]:
            rows.append({"Line": label, "Date": pt.get("date", "--"),
                         "ECE-cal": _fmt3(pt.get("ece_calibrated")),
                         "Brier": _fmt3(pt.get("brier")),
                         "Logloss": _fmt3(pt.get("logloss"), 4),
                         "Pred mean": _fmt3(pt.get("predicted_mean")),
                         "n": pt.get("n", 0)})
    return rows[-30:]


def _fmt3(value, digits: int = 3) -> str:
    v = _num(value)
    return "—" if v is None else f"{v:.{digits}f}"
