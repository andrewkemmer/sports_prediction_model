"""Pure NBA point-spread/totals slate helpers for cards and diagnostics."""
from __future__ import annotations

import math
import re

SPREAD_GRID = list(range(-20, 21))
TOTAL_GRID = list(range(180, 281))
HALF_STOP_LINES = [-0.5, 0.5]
_SPREAD_RE = re.compile(r"^p_(?:home|away)_cover_(m?\d+(?:_\d+)?)$")
_OVER_RE = re.compile(r"^p_over_(\d+(?:_\d+)?)$")
_UNDER_RE = re.compile(r"^p_under_(\d+(?:_\d+)?)$")
_FAIR_LINE_COLUMNS = ("fair_spread", "fair_total", "mu_margin", "mu_total",
                      "mu_h", "mu_a", "p_home_win_derived", "p_away_win_derived")


def _token_value(token: str) -> float:
    negative = token.startswith("m") or token.startswith("-")
    if token.startswith("m"):
        token = token[1:]
    elif token.startswith("-"):
        token = token[1:]
    value = float(token.replace("_", "."))
    return -value if negative else value


def parse_spread_line(name: str) -> float | None:
    match = _SPREAD_RE.match(str(name))
    return _token_value(match.group(1)) if match else None


def parse_total_line(name: str) -> float | None:
    match = _OVER_RE.match(str(name)) or _UNDER_RE.match(str(name))
    if not match:
        return None
    token = match.group(1).replace("_", ".")
    return float(token)


def _label(value: float) -> str:
    text = str(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("-", "m").replace(".", "_")


def spread_columns() -> dict[float, tuple[str, str, str]]:
    out = {}
    for line in [*SPREAD_GRID, *HALF_STOP_LINES]:
        label = _label(line)
        out[line] = (f"p_home_cover_{label}", f"p_push_{label}", f"p_away_cover_{label}")
    return out


def total_columns() -> dict[float, tuple[str, str, str]]:
    return {float(line): (f"p_over_{line}", f"p_under_{line}", f"p_push_total_{line}")
            for line in TOTAL_GRID}


def _f(row, *keys: str) -> float | None:
    for key in keys:
        try:
            value = row.get(key)
            if value is not None:
                number = float(value)
                if math.isfinite(number):
                    return number
        except (TypeError, ValueError):
            continue
    return None


def price_total(row, total: float) -> tuple[float | None, float | None, float | None]:
    value = float(total)
    if not math.isfinite(value):
        return None, None, None
    whole = int(round(value))
    if abs(value - whole) > 1e-9:
        # Total scores are integer-valued: a .5 line has no push.
        over_key = f"p_over_{whole}"
        over = _f(row, over_key)
        return (over, None if over is None else 1 - over, 0.0) if over is not None else (None, None, None)
    cols = total_columns().get(float(whole))
    if cols is None:
        return None, None, None
    return tuple(_f(row, col) for col in cols)  # type: ignore[return-value]


def price_spread(row, line: float) -> tuple[float | None, float | None, float | None]:
    cols = spread_columns().get(float(line))
    if cols is None:
        return None, None, None
    home, push, away = cols
    ph, pp, pa = _f(row, home), _f(row, push), _f(row, away)
    if ph is None:
        return None, None, None
    if pp is None and float(line).is_integer():
        pp = 0.0
    if pa is None and pp is not None:
        pa = max(0.0, 1.0 - ph - pp)
    return ph, pp, pa


def price_spread_line(row, line: float) -> tuple[float | None, float | None, float | None]:
    value = float(line)
    if not math.isfinite(value):
        return None, None, None
    direct = price_spread(row, value)
    if direct[0] is not None:
        return direct
    # Integer score support lets a half threshold use the adjacent whole
    # threshold while retaining the strict NBA cover convention.
    if value > 0:
        return price_spread(row, math.floor(value))
    return price_spread(row, math.ceil(value))


def half_stop_pair(row) -> tuple[float | None, float | None, float | None, float | None, bool | None]:
    """Return favorite/dog probabilities for the two half-stop thresholds.

    The distribution stores ``p_home_cover_0_5`` (home margin > +0.5) and
    ``p_away_cover_m0_5`` (home margin < -0.5).  Looking up only
    ``p_home_cover_m0_5`` silently prices the wrong side of a half stop.
    """
    # For a +0.5/−0.5 half stop, the favorite covers iff it wins outright.
    # The artifact stores home cover at +0.5 as ``home_fav_raw`` and away
    # cover at −0.5 as ``away_fav_raw``; these are the two sides of the same
    # event, not complementary unconditional probabilities when a tie mass
    # exists.
    home_plus_half = _f(row, "p_home_cover_0_5")
    away_plus_half = _f(row, "p_away_cover_m0_5")
    home_ml = _f(row, "p_home_win_derived")
    away_ml = _f(row, "p_away_win_derived")
    if None in (home_plus_half, away_plus_half, home_ml, away_ml):
        return None, None, None, None, None
    favorite_home = home_ml >= 0.5
    return ((home_plus_half, away_plus_half, home_ml, away_ml, True)
            if favorite_home else
            (away_plus_half, home_plus_half, away_ml, home_ml, False))


def grid_rows(df) -> tuple[list[float], list[float]]:
    spreads, totals = [], []
    for column in df.columns:
        spread = parse_spread_line(column)
        if spread is not None and spread in [*SPREAD_GRID, *HALF_STOP_LINES] and spread not in spreads:
            spreads.append(spread)
        total = parse_total_line(column)
        if total is not None and total in TOTAL_GRID and total not in totals:
            totals.append(total)
    return sorted(spreads), sorted(totals)


def has_fair_columns(df) -> bool:
    return all(column in df.columns for column in _FAIR_LINE_COLUMNS)


def _pct(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _line_text(value: float) -> str:
    return str(int(round(value))) if abs(float(value) - round(float(value))) <= 1e-9 else f"{float(value):.1f}"


def _spread_label(team: str, points: float) -> str:
    amount = _line_text(abs(float(points)))
    return f"{team} −{amount}" if float(points) < 0 else f"{team} +{amount}"


def _push_note(probability: float | None) -> str:
    return f' <span class="re-na">({_pct(probability)} push)</span>' if (probability or 0) > 0.005 else ""


def runline_html(row, home_team: str, away_team: str,
                 home_spread: float | None = None, half_stop: bool = False) -> str:
    if half_stop:
        fav_raw, dog_raw, fav_ml, dog_ml, fav_home = half_stop_pair(row)
        if fav_raw is None:
            return '<span>SPREAD: n/a</span>'
        fav_team, dog_team = (home_team, away_team) if fav_home else (away_team, home_team)
        return (f'<span>SPREAD: {fav_team} −0.5 {_pct(fav_raw)} '
                f'<span class="re-na">(ML {_pct(fav_ml)})</span> · {dog_team} +0.5 '
                f'{_pct(dog_raw)} <span class="re-na">(ML {_pct(dog_ml)})</span></span>')
    if home_spread is None:
        fair = _f(row, "fair_spread")
        if fair is None:
            return '<span>SPREAD: n/a</span>'
        home_spread = -float(round(fair))
    threshold = -float(home_spread)
    home, push, away = price_spread_line(row, threshold)
    if home is None or away is None:
        return f'<span>SPREAD: {_spread_label(home_team, home_spread)} n/a</span>'
    return (f'<span>SPREAD: {_spread_label(home_team, home_spread)} {_pct(home)} · '
            f'{_spread_label(away_team, -float(home_spread))} {_pct(away)}'
            f'{_push_note(push) if abs(float(home_spread) - round(float(home_spread))) <= 1e-9 else ""}</span>')


def runengine_html(row, home_team: str, away_team: str,
                   total_line: float | None = None,
                   home_spread: float | None = None,
                   half_stop: bool = False) -> str:
    mu_h, mu_a = _f(row, "mu_h"), _f(row, "mu_a")
    fair_total, fair_spread = _f(row, "fair_total"), _f(row, "fair_spread")
    if None in (mu_h, mu_a, fair_total, fair_spread):
        return '<div class="fb-runengine"><span class="re-label">RUN ENGINE</span><span class="re-na">n/a</span></div>'
    total = float(total_line if total_line is not None else fair_total)
    over, under, push = price_total(row, total)
    total_span = (f'<span>O/U {_line_text(total)}: n/a</span>' if over is None or under is None
                  else f'<span>O/U {_line_text(total)}: Over {_pct(over)} / Under {_pct(under)}'
                       f'{_push_note(push) if abs(total - round(total)) <= 1e-9 else ""}</span>')
    spread = runline_html(row, home_team, away_team, home_spread, half_stop)
    return ('<div class="fb-runengine"><span class="re-label">RUN ENGINE</span>'
            f'<span>Proj: {away_team} {_num(mu_a)} – {home_team} {_num(mu_h)}</span>'
            f'{total_span}{spread}</div>')


def push_span(probability: float | None) -> str:
    return _push_note(probability)
