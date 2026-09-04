"""Pure NFL slate-serve helpers — shared by the NFL Totals & Run Lines page
(``nfl_markets_page.py``) and the NFL Today's Games run-engine card box
(``todays_games.py``).

Read-only over the committed slate-serve artifact
(``nfl_run_engine_markets_YYYYMMDD.csv``, backend ``nfl_slate_engine.py``).
The artifact's OWN column contract is the schema (documented 1:1 against the
MLB ``run_engine_markets_*`` mapping table in the slate-serve record): the
integer spread grid ``p_home_cover_<±L>/p_push_<L>`` over -14..+14, the
integer totals grid ``p_over_<U>/p_under_<U>/p_push_<U>`` over 24..66, and
the per-side derived-ML pair ``p_home_win_derived/p_away_win_derived``.

Column semantics (mirror of the backend engine + the MLB convention "home
covers -L iff margin > L, strict"):
  * ``p_home_cover_<L>`` = P(margin > L)  — the probability the HOME team
    covers the spread -L (i.e. the grid value L is the margin THRESHOLD, and
    the home team's quoted spread is the negation, -L).
  * ``p_push_<L>`` = P(margin == L) — the shared push band of the whole
    integer line (both sides push on the same margin).
  * ``p_over_<U>/p_under_<U>/p_push_<U>`` = P(total > U) / P(total < U) /
    P(total == U) for the integer total U.
  * ``p_home_win_derived``/``p_away_win_derived`` = P(H>A)/(1-P(tie)) per
    side — the derived ML pair (ties normalized out).

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

import re

# The slate engine's grids (mirror nfl_slate_engine.SPREAD_INT_LINES /
# TOTAL_INT_LINES — integers; the NFL margin/total PMFs are integer-support,
# so whole-number lines carry a real push band).
SPREAD_GRID = list(range(-14, 15))
TOTAL_GRID = list(range(24, 67))

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
    Returns None for non-grid columns (``p_home_cover_minus_half`` etc.).
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
    """total -> (over, under, push) columns for the priced totals grid."""
    out: dict[int, tuple[str, str, str]] = {}
    for U in TOTAL_GRID:
        out[U] = (f"p_over_{U}", f"p_under_{U}", f"p_push_{U}")
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


def price_total(row, total: int) -> tuple[float | None, float | None, float | None]:
    """(P(over U), P(under U), P(push U)) at the integer total U.

    Push = exact tie on the total (P(score_h + score_a == U)) — the NFL
    integer-total push band. (None, None, None) when U is off the grid.
    """
    o, u, p = total_columns().get(total, (None, None, None))
    if o is None:
        return None, None, None
    return _f(row, o), _f(row, u), _f(row, p)


def price_spread(row, line: int) -> tuple[float | None, float | None, float | None]:
    """(P(home covers), P(push), P(away covers)) at the home spread -line.

    ``line`` is the margin THRESHOLD L: home is quoted -L (home -L covers
    margin > L), away +L (away +L covers margin < L), and both push on
    margin == L. Home covers + push + away covers == 1 over every integer
    margin. (None, None, None) for lines off the grid.
    """
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


def half_stop_pair(row) -> tuple[float | None, float | None,
                                  float | None, float | None, bool | None]:
    """±0.5-stop pair: (raw fav -0.5, raw dog +0.5, ML fav, ML dog) + fav_home.

    The ±0.5 stop resolves the integer margin == 0 split (a pick'em pair —
    the same pair the MLB box's ±0.5 magnitude means): the favorite at -0.5
    covers exactly when it wins OUTRIGHT (raw -0.5 EXCLUDES the tie), the
    underdog at +0.5 covers on a win OR a tie (raw +0.5 INCLUDES the tie).
    The grey-italic (ML X%) parentheticals are the derived pair
    P(H>A)/(1-P(tie)) / P(A>H)/(1-P(tie)) — the tie mass normalized out — so
    on NFL the two diverge by the (calibrated ~0.3%) tie rate: fav raw < fav
    ML and dog raw > dog ML. Never conflated; raw and derived are distinct
    model statements.

    Returns (fav_raw, dog_raw, fav_ml, dog_ml, fav_is_home); all None when
    the split columns are missing (malformed/legacy row) — never fabricated.
    """
    ph0 = _f(row, "p_home_cover_0")      # P(margin > 0): home wins outright
    pp0 = _f(row, "p_push_0")            # P(margin == 0): the tie
    fh = _f(row, "p_home_win_derived")   # derived ML P(H>A)/(1-P_tie)
    fa = _f(row, "p_away_win_derived")
    if ph0 is None or pp0 is None or fh is None or fa is None:
        return None, None, None, None, None
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
    """Integer margin thresholds and totals ACTUALLY present on the artifact."""
    spreads, totals = [], []
    for c in df.columns:
        L = parse_spread_line(c)
        if L is not None and L not in spreads:
            spreads.append(L)
        U = parse_total_line(c)
        if U is not None and U not in totals:
            totals.append(U)
    return sorted(spreads), sorted(totals)


def has_fair_columns(df) -> bool:
    """True when the frame carries the fair-line + derived pair columns."""
    return all(c in df.columns for c in _FAIR_LINE_COLUMNS)


def _pct(v: float | None, nd: int = 0) -> str:
    return "—" if v is None else f"{v * 100:.{nd}f}%"


def _num(v: float | None, nd: int = 1) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _spread_label(team: str, pts: int) -> str:
    """'TEAM −3' when the team lays 3, 'TEAM +3' when it gets 3.

    ``pts`` is the side's OWN quoted spread: negative lays, positive gets —
    so the away mirror of a home spread S is -S.
    """
    if pts < 0:
        return f"{team} −{abs(pts)}"
    return f"{team} +{pts}"


def _push_note(pp: float | None) -> str:
    """Small grey shared push note for whole-number lines."""
    if (pp or 0) > 0.005:
        return f' <span class="re-na">({_pct(pp)} push)</span>'
    return ""


def runline_html(row, home_team: str, away_team: str,
                 home_spread: int | None = None,
                 half_stop: bool = False) -> str:
    """The run-line span at the selected HOME spread (or the ±0.5 stop).

    Defaults to the fair home spread (the negative of ``fair_spread``, the
    median margin threshold). Integer lines render the home/away covers from
    the home-anchored grid with the SHARED push note; no (ML) parentheticals
    at integers. The ±0.5 stop renders per-side RAW cover as the main number
    AND the grey-italic (ML X%) derived parenthetical — the NFL-specific raw
    vs derived pair (they diverge by the tie rate). Never renders offered
    lines, shrink columns or edges.
    """
    if home_spread is None:
        fair_spread = _f(row, "fair_spread")
        if fair_spread is None:
            return '<span>RL: n/a</span>'
        home_spread = -int(round(fair_spread))
    if half_stop:
        fav_raw, dog_raw, fav_ml, dog_ml, fav_home = half_stop_pair(row)
        if fav_raw is None:
            return '<span>RL: n/a</span>'
        if fav_home:
            fav_team, dog_team = home_team, away_team
        else:
            fav_team, dog_team = away_team, home_team
        fav_note = f' <span class="re-na">(ML {_pct(fav_ml, 0)})</span>'
        dog_note = f' <span class="re-na">(ML {_pct(dog_ml, 0)})</span>'
        return (f'<span>RL: {fav_team} −0.5 {_pct(fav_raw)}'
                f'{fav_note} · {dog_team} +0.5 {_pct(dog_raw)}{dog_note}</span>')
    # Integer line at home spread S: threshold L = -S (home covers margin > L).
    L = -home_spread
    ph, pp, pa = price_spread(row, L)
    if ph is None or pa is None:
        return f'<span>RL: {_spread_label(home_team, home_spread)} n/a</span>'
    return (f'<span>RL: {_spread_label(home_team, home_spread)} {_pct(ph)} · '
            f'{_spread_label(away_team, -home_spread)} {_pct(pa)}'
            f'{_push_note(pp)}</span>')


def runengine_html(row, home_team: str, away_team: str,
                   total_line: int | None = None,
                   home_spread: int | None = None,
                   half_stop: bool = False) -> str:
    """Model-fair run-engine strip for one slate row — market-free.

    Renders ONLY model values: projected scores (the mu pair), the O/U at
    the fair total (or a caller-chosen grid total) with P(over)/P(under) and
    a P(push) note at integer totals, the run-line span (see ``runline_html``
    — integer home/away covers at the fair home spread with the shared push
    note, or the ±0.5 stop's raw + derived pair), and the derived-ML pair
    P(H>A)/(1−P(tie)) per side. Offered/book lines, shrink columns and
    market edges never render.

    Rows without the fair columns produce a quiet 'n/a' — never fabricated.
    """
    mu_a, mu_h = _f(row, "mu_a"), _f(row, "mu_h")
    fair_total = _f(row, "fair_total")
    if mu_a is None or mu_h is None or fair_total is None \
            or _f(row, "fair_spread") is None:
        return ('<div class="fb-runengine"><span class="re-label">'
                'RUN ENGINE</span><span class="re-na">n/a</span></div>')

    tot = int(round(total_line if total_line is not None else fair_total))
    po, pu, ppush = price_total(row, tot)
    if po is None or pu is None:
        total_span = f'<span>O/U {tot}: n/a</span>'
    else:
        total_span = (f'<span>O/U {tot}: Over {_pct(po)} / '
                      f'Under {_pct(pu)}{_push_note(ppush)}</span>')

    rl = runline_html(row, home_team, away_team,
                      home_spread=home_spread, half_stop=half_stop)

    dh = _f(row, "p_home_win_derived", "derived_ml")
    da = _f(row, "p_away_win_derived")
    if dh is not None and da is not None:
        ml_span = (f'<span>ML: {home_team} {_pct(dh, 1)} · '
                   f'{away_team} {_pct(da, 1)}</span>')
    else:
        ml_span = '<span>ML: n/a</span>'

    return ('<div class="fb-runengine"><span class="re-label">'
            'RUN ENGINE</span>'
            f'<span>Proj: {away_team} {_num(mu_a)} – '
            f'{home_team} {_num(mu_h)}</span>'
            f'{total_span}{rl}{ml_span}</div>')


def push_span(pp: float | None) -> str:
    """Small grey push note for whole-number lines (integer margins)."""
    return _push_note(pp)
