"""Pure NHL slate-serve helpers — shared by the NHL Totals & Run Lines page
(``nhl_markets_page.py``) and the NHL Today's Games run-engine card box
(``nhl_todays_page.py``).

Read-only over the committed slate-serve artifact
(``nhl_run_engine_markets_YYYYMMDD.csv``, backend ``distributions.py`` /
``serving.write_markets_csv``). The artifact's OWN column contract is the
schema (documented 1:1 against the MLB ``run_engine_markets_*`` mapping
table in the slate-serve record): the integer spread grid
``p_home_cover_<±L>/p_push_<L>`` over -8..+8, the integer totals grid
``p_over_<U>/p_under_<U>`` with pushes in their OWN ``p_push_total_<U>``
namespace over 4..12, and the per-side derived-ML pair
``p_home_win_derived/p_away_win_derived``.

NHL GRID OVERLAP (the one structural divergence from MLB/NFL): the spread
grid (-8..+8) and the totals grid (4..12) SHARE the integer lines 4..8, so
a shared ``p_push_<N>`` column would be ambiguous (margin push vs total
push). The backend therefore owns margin pushes at ``p_push_<L>`` and
totals pushes at ``p_push_total_<U>`` — this module reads each namespace
exclusively (``total_columns`` never touches ``p_push_<U>``).

Column semantics (mirror of the backend engine + the MLB convention "home
covers -L iff margin > L, strict"):
  * ``p_home_cover_<L>`` = P(margin > L)  — the probability the HOME team
    covers the spread -L (i.e. the grid value L is the margin THRESHOLD, and
    the home team's quoted spread is the negation, -L).
  * ``p_push_<L>`` = P(margin == L) — the shared push band of the whole
    integer line (both sides push on the same margin). At L == 0 this is
    P(margin == 0): the regulation-tie band that overtime/shootout resolves
    (hockey ties ~5-8%; backend caps it at P_TIE_MAX).
  * ``p_over_<U>/p_under_<U>/p_push_total_<U>`` = P(total > U) /
    P(total < U) / P(total == U) for the integer total U.
  * ``p_home_win_derived``/``p_away_win_derived`` = P(H>A)/(1-P(tie)) per
    side — the derived ML pair (the OT band normalized out).

MARKET-FREE BY POLICY: every value surfaced here is a MODEL fair value or a
model probability at a model-priced line — ``fair_spread`` / ``fair_total``
(medians of the margin/total PMFs) and the grid columns. Offered/book lines,
shrink columns and market-derived "edge" are never read, never priced and
never rendered by this module (the artifact carries them for a future feed
mode; they stay out of the model product).

Run-line display convention: the box is home-anchored (the artifact prices
``p_home_cover_*``) and shows the pair at the selected HOME spread S with
the away mirror -S: for a threshold L the home spread is S = -L, so
``spread_html``/``runengine_html`` render "HOME -L / AWAY +L" (home lays L)
for L>0 and "HOME +|L| / AWAY -|L|" for L<0 — the same sign the backend
offers (``spread_line``, corr(margin, line) > 0) and the MLB box use.

This module is dependency-free (pandas only) so it can be imported by pure
tests and by page modules without a Streamlit runtime.
"""

from __future__ import annotations

import math
import re

# The NHL distribution engine's grids (mirror config.SPREAD_GRID /
# config.TOTAL_GRID — integers; the NB Monte Carlo margin/total PMFs are
# integer-support, so whole-number lines carry a real push band).
SPREAD_GRID = list(range(-8, 9))
TOTAL_GRID = list(range(4, 13))

_SPREAD_RE = re.compile(r"^p_home_cover_(m?\d+)$")
_OVER_RE = re.compile(r"^p_over_(\d+)$")
_UNDER_RE = re.compile(r"^p_under_(\d+)$")

# Fair-line / derived pair columns the market-free view reads (everything
# else — offered lines, shrink, edges — is never touched).
_FAIR_LINE_COLUMNS = ("fair_spread", "fair_total", "mu_margin", "mu_total",
                      "mu_h", "mu_a", "p_home_win_derived",
                      "p_away_win_derived")


def parse_spread_line(name: str) -> int | None:
    """Home margin-threshold L from a ``p_home_cover_<±L>`` column name.

    ``p_home_cover_m3`` -> -3 (home +3), ``p_home_cover_7`` -> +7 (home -7).
    Returns None for non-grid columns (``p_home_cover_m0_5`` etc.).
    """
    m = _SPREAD_RE.match(name)
    if not m:
        return None
    token = m.group(1)
    return -int(token[1:]) if token.startswith("m") else int(token)


def parse_total_line(name: str) -> int | None:
    """Total U from a ``p_over_<U>`` / ``p_under_<U>`` column name."""
    m = _OVER_RE.match(name) or _UNDER_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def spread_columns() -> dict[int, tuple[str, str]]:
    """threshold L -> (home-cover column, push column) for the priced grid.

    The home team's quoted spread at threshold L is -L (see module doc).
    """
    out: dict[int, tuple[str, str]] = {}
    for L in SPREAD_GRID:
        label = f"m{-L}" if L < 0 else str(L)
        out[L] = (f"p_home_cover_{label}", f"p_push_{label}")
    return out


def total_columns() -> dict[int, tuple[str, str, str]]:
    """total -> (over, under, push) columns for the priced totals grid.

    NHL totals pushes live in their OWN ``p_push_total_<U>`` namespace (the
    spread/totals grids overlap on 4..8 — see module doc); the shared
    ``p_push_<U>`` column is the MARGIN push and is never read here.
    """
    out: dict[int, tuple[str, str, str]] = {}
    for U in TOTAL_GRID:
        out[U] = (f"p_over_{U}", f"p_under_{U}", f"p_push_total_{U}")
    return out


def _f(row, *keys: str) -> float | None:
    """First numeric value across column aliases; None when all missing."""
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        return fv
    return None


def price_total(row, total: float) -> tuple[float | None, float | None, float | None]:
    """(P(over), P(under), P(push)) at an integer or half-point total.

    The NHL artifact has integer-support score distributions. A half-point
    line therefore uses the adjacent integer threshold and assigns the
    integer push mass to the underdog side of the half-line:

      Over U.5 = P(total > U); Under U.5 = P(total <= U); Push = 0.

    Whole-number lines retain their explicit over/under/push split (the
    push read from ``p_push_total_<U>``).
    """
    value = float(total)
    if not math.isfinite(value):
        return None, None, None
    if abs(value - round(value)) > 1e-9:
        base = math.floor(value)
        o, _, p = total_columns().get(base, (None, None, None))
        if o is None:
            return None, None, None
        over = _f(row, o)
        if over is None:
            return None, None, None
        return over, 1.0 - over, 0.0
    whole = int(round(value))
    o, u, p = total_columns().get(whole, (None, None, None))
    if o is None:
        return None, None, None
    return _f(row, o), _f(row, u), _f(row, p)


def price_spread(row, line: int) -> tuple[float | None, float | None, float | None]:
    """(P(home covers), P(push), P(away covers)) at an integer threshold."""
    cols = spread_columns().get(line)
    if cols is None:
        return None, None, None
    home, push = cols
    ph = _f(row, home)
    pp = _f(row, push)
    if ph is None:
        return None, None, None
    pa = None if pp is None else 1.0 - ph - pp
    return ph, pp, pa


def price_spread_line(row, line: float) -> tuple[float | None, float | None, float | None]:
    """Price a quoted spread threshold at whole- or half-point precision.

    ``line`` is the home margin threshold: home is quoted ``-line``. NHL
    score outcomes are integer-valued, so a half threshold has no push and
    uses the nearest lower integer threshold for positive values (or the
    corresponding ceiling boundary for negative values).
    """
    value = float(line)
    if not math.isfinite(value):
        return None, None, None
    if abs(value - round(value)) <= 1e-9:
        return price_spread(row, int(round(value)))
    integer_threshold = math.ceil(value) - 1
    ph, _, _ = price_spread(row, integer_threshold)
    if ph is None:
        return None, None, None
    return ph, 0.0, 1.0 - ph


def half_stop_pair(row) -> tuple[float | None, float | None,
                                  float | None, float | None, bool | None]:
    """±0.5-stop pair: (raw fav -0.5, raw dog +0.5, ML fav, ML dog) + fav_home.

    The ±0.5 stop resolves the integer margin == 0 split (a pick'em pair —
    the same pair the MLB box's ±0.5 magnitude means): the favorite at -0.5
    covers exactly when it wins OUTRIGHT (raw -0.5 EXCLUDES the regulation
    tie), the underdog at +0.5 covers on a win OR a regulation tie (raw
    +0.5 INCLUDES the tie). The grey-italic (run-ML X%) parentheticals are
    the derived pair P(H>A)/(1-P(tie)) / P(A>H)/(1-P(tie)) — the regulation-
    tie mass (the band overtime/shootout resolves) normalized out — so on
    NHL the two diverge by the tie rate (~5-8% of games, backend-capped at
    P_TIE_MAX): fav raw < fav ML and dog raw > dog ML. Never conflated; raw
    and derived are distinct model statements.

    Returns (fav_raw, dog_raw, fav_ml, dog_ml, fav_is_home); all None when
    the split columns are missing (malformed/legacy row) — never fabricated.
    """
    ph0 = _f(row, "p_home_cover_0")      # P(margin > 0): home wins outright
    pp0 = _f(row, "p_push_0")            # P(margin == 0): the regulation tie
    if ph0 is None or pp0 is None:
        return None, None, None, None, None
    # Rebuild the derived run-ML from the mutually exclusive outright wins.
    # The artifact's derived columns may carry pre-normalization mass on some
    # vintages; ties must be removed before normalizing the two moneyline
    # sides (the same rebuild the NFL view performs).
    home_win = max(0.0, ph0)
    away_win = max(0.0, 1.0 - ph0 - pp0)
    win_mass = home_win + away_win
    if win_mass <= 0.0:
        return None, None, None, None, None
    fh = home_win / win_mass
    fa = away_win / win_mass
    fav_home = fh >= 0.5
    if fav_home:
        fav_raw, dog_raw = ph0, 1.0 - ph0          # away +0.5 wins ties
        fav_ml, dog_ml = fh, fa
    else:
        fav_raw = 1.0 - ph0 - pp0                   # away -0.5: away outright
        dog_raw = ph0 + pp0                         # home +0.5: wins or ties
        fav_ml, dog_ml = fa, fh
    return fav_raw, dog_raw, fav_ml, dog_ml, fav_home


def grid_rows(df) -> tuple[list[int], list[int]]:
    """Integer margin thresholds and totals ACTUALLY present on the artifact,
    filtered to the card's priced grid (SPREAD_GRID / TOTAL_GRID). The
    artifact's extended favorite-magnitude columns — additive for the
    Diagnostics Spread Lines tab — are excluded here: the card quotes the
    integer grid only; the extended grid is a diagnostics view."""
    spreads, totals = [], []
    for c in df.columns:
        L = parse_spread_line(c)
        if L is not None and L in SPREAD_GRID and L not in spreads:
            spreads.append(L)
        U = parse_total_line(c)
        if U is not None and U in TOTAL_GRID and U not in totals:
            totals.append(U)
    return sorted(spreads), sorted(totals)


def has_fair_columns(df) -> bool:
    """True when the frame carries the fair-line + derived pair columns."""
    return all(c in df.columns for c in _FAIR_LINE_COLUMNS)


def _pct(v: float | None, nd: int = 0) -> str:
    return "—" if v is None else f"{v * 100:.{nd}f}%"


def _num(v: float | None, nd: int = 1) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _line_text(value: float) -> str:
    value = float(value)
    return str(int(round(value))) if abs(value - round(value)) <= 1e-9 else f"{value:.1f}"


def _spread_label(team: str, pts: float) -> str:
    """Render a team's quoted spread, including half-point lines."""
    text = _line_text(abs(float(pts)))
    return f"{team} −{text}" if float(pts) < 0 else f"{team} +{text}"


def _push_note(pp: float | None) -> str:
    """Small grey shared push note for whole-number lines."""
    if (pp or 0) > 0.005:
        return f' <span class="re-na">({_pct(pp)} push)</span>'
    return ""


def runline_html(row, home_team: str, away_team: str,
                 home_spread: float | None = None,
                 half_stop: bool = False) -> str:
    """The run-line span at the selected HOME spread (or the ±0.5 stop).

    Defaults to the fair home spread (the negative of ``fair_spread``, the
    median margin threshold). Integer lines render the home/away covers from
    the home-anchored grid with the SHARED push note; no (ML) parentheticals
    at integers. The ±0.5 stop renders per-side RAW cover as the main number
    AND the grey-italic (run-ML X%) derived parenthetical — the NHL-specific
    raw vs derived pair (they diverge by the regulation-tie band). Never
    renders offered lines, shrink columns or edges.
    """
    if half_stop:
        fav_raw, dog_raw, fav_ml, dog_ml, fav_home = half_stop_pair(row)
        if fav_raw is None:
            return '<span>RL: n/a</span>'
        if fav_home:
            fav_team, dog_team = home_team, away_team
        else:
            fav_team, dog_team = away_team, home_team
        fav_note = f' <span class="re-na">(run-ML {_pct(fav_ml, 0)})</span>'
        dog_note = f' <span class="re-na">(run-ML {_pct(dog_ml, 0)})</span>'
        return (f'<span>RL: {fav_team} −0.5 {_pct(fav_raw)}'
                f'{fav_note} · {dog_team} +0.5 {_pct(dog_raw)}{dog_note}</span>')
    if home_spread is None:
        fair_spread = _f(row, "fair_spread")
        if fair_spread is None:
            return '<span>RL: n/a</span>'
        home_spread = -float(round(fair_spread))
    # Home spread S corresponds to margin threshold L = -S.
    L = -float(home_spread)
    ph, pp, pa = price_spread_line(row, L)
    if ph is None or pa is None:
        return f'<span>RL: {_spread_label(home_team, home_spread)} n/a</span>'
    push_note = _push_note(pp) if abs(float(home_spread) - round(float(home_spread))) <= 1e-9 else ""
    return (f'<span>RL: {_spread_label(home_team, home_spread)} {_pct(ph)} · '
            f'{_spread_label(away_team, -float(home_spread))} {_pct(pa)}'
            f'{push_note}</span>')


def runengine_html(row, home_team: str, away_team: str,
                   total_line: float | None = None,
                   home_spread: float | None = None,
                   half_stop: bool = False) -> str:
    """Model-fair run-engine strip for one slate row — market-free.

    Renders ONLY model values: projected scores (the mu pair), the O/U at
    the fair total (or a caller-chosen grid total) with P(over)/P(under) and
    a P(push) note at integer totals (read from the totals-push namespace),
    the run-line span (see ``runline_html`` — integer home/away covers at
    the fair home spread with the shared push note, or the ±0.5 stop's raw
    + derived pair), and the derived-ML pair P(H>A)/(1−P(tie)) per side.
    Offered/book lines, shrink columns and market edges never render.

    Rows without the fair columns produce a quiet 'n/a' — never fabricated.
    """
    mu_a, mu_h = _f(row, "mu_a"), _f(row, "mu_h")
    fair_total = _f(row, "fair_total")
    if mu_a is None or mu_h is None or fair_total is None \
            or _f(row, "fair_spread") is None:
        return ('<div class="fb-runengine"><span class="re-label">'
                'RUN ENGINE</span><span class="re-na">n/a</span></div>')

    tot = float(total_line if total_line is not None else fair_total)
    po, pu, ppush = price_total(row, tot)
    if po is None or pu is None:
        total_span = f'<span>O/U {_line_text(tot)}: n/a</span>'
    else:
        push_note = _push_note(ppush) if abs(tot - round(tot)) <= 1e-9 else ""
        total_span = (f'<span>O/U {_line_text(tot)}: Over {_pct(po)} / '
                      f'Under {_pct(pu)}{push_note}</span>')

    rl = runline_html(row, home_team, away_team,
                      home_spread=home_spread, half_stop=half_stop)
    ml_caption = (
        '<span class="re-na" style="flex-basis:100%;">'
        'run-ML is derived from the run-engine score distribution — '
        'regulation ties are excluded from both sides; the binary moneyline '
        'is at the top of the card</span>'
        if half_stop else ""
    )

    # No separate ML span — MLB's run-engine strip renders Proj / O/U / RL
    # only, and the win probabilities are already the card's two team bars
    # (the ±0.5 stop's per-side derived-ML notes carry the raw pair where
    # it is genuinely line-specific). Same strip anatomy, byte-for-byte.
    return ('<div class="fb-runengine"><span class="re-label">'
            'RUN ENGINE</span>'
            f'<span>Proj: {away_team} {_num(mu_a)} – '
            f'{home_team} {_num(mu_h)}</span>'
            f'{total_span}{rl}{ml_caption}</div>')


def push_span(pp: float | None) -> str:
    """Small grey push note for whole-number lines (integer margins)."""
    return _push_note(pp)
