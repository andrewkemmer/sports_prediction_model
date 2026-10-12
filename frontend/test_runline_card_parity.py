"""Card ↔ run-line/totals prediction-history parity (2026-10-12 remediation).

Two surfaces quote the same model statement and must never disagree:

  * the RUN ENGINE strip on a Today's Games card — ``runline_html`` /
    ``runengine_html`` over the row's own frame (the card serves the dated
    ``kind='slate'`` row; frozen pre-game price, enforced by the board
    smoke tests), and
  * the Totals & Run Lines **Prediction History** rows —
    ``_runline_row`` / ``_totals_row`` on the NFL/NHL markets pages over
    the decided ``kind='oof'`` rows.

The 2026-10-11/12 dashboard review found the history tables quoting lines
the cards never printed:

  * NFL DET@ARI (Oct 11, 2026): the run-line row read ``DET +4`` on a game
    whose card says ``DET −5`` — ``int(round(fair_spread))`` dragged the
    half-point OOF fair line (-4.5) onto -4 AND rendered it with the away
    team on the wrong side of a sign-flipped line cell;
  * NHL: the history rendered ±0 / ±2-style integers (e.g. ``RL NYR -2``)
    where the card quotes the half ladder and the ±0.5 stop, and its
    pick cell carried the wrong side's break-even.

Contract pinned here (for every decided OOF row of the latest committed
artifact of both sports, not a fixture subset):

  1. LINE — the row's OWN fair line on the 0.5 ladder; home/away sides
     always opposite signs (home = -t); never a fabricated integer, never
     a ±0.0 line. Byte-for-byte the pair the card's run-line span prints
     for the same row.
  2. PICK — the PICKED side's own line and that side's 2-way break-even,
     identical to the number the card prints for that side.
  3. TOTALS — the pick cell's side matches its percentage (an Under pick
     carries P(under), not P(over)) and the line matches the card's O/U.
  4. Pushes grade only at whole-number lines.
  5. Pick'em (fair spread 0) renders the favorite-anchored ±0.5 stop on
     BOTH surfaces — never a fake ±0.0.
  6. The reported game: DET@ARI's history row stays on the SAME side of
     the line as its dated game card.

Run: ``python -m pytest test_runline_card_parity.py -q``
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

FRONTEND = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND.parent
sys.path.insert(0, str(FRONTEND))

import nfl_markets_page as nfl_page  # noqa: E402
import nfl_slate_view as nsv  # noqa: E402
import nhl_markets_page as nhl_page  # noqa: E402
import nhl_slate_view as nhl_sv  # noqa: E402

SPORTS = {
    "nfl": (nfl_page, nsv, REPO_ROOT / "nfl-backend" / "data_delivery"),
    "nhl": (nhl_page, nhl_sv, REPO_ROOT / "nhl-backend" / "data_delivery"),
}

# Card + history parse targets (unicode minus normalized to ASCII first).
_SIDE = r"([A-Z]{2,3}) ([+\-]\d+(?:\.\d+)?)"
_RE_CARD_PAIRS = re.compile(rf"{_SIDE} (\d+)%")
_RE_LINE_PAIRS = re.compile(_SIDE)
_RE_PICK = re.compile(rf"{_SIDE} \((\d+)%\)")
_RE_OU_CARD = re.compile(r"O/U ([\d.]+): Over (\d+)% / Under (\d+)%")
_RE_OU_PICK = re.compile(r"(Over|Under) ([\d.]+) \((\d+)%\)")

MAX_ROWS = 400  # most-recent decided rows exercised per sport


def _latest(dd: Path) -> Path:
    files = sorted(dd.glob("*_run_engine_markets_*.csv"))
    assert files, f"no run-engine artifact under {dd}"
    return files[-1]


def _decided(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df.get("kind") == "oof"].copy()
    if not len(df):
        return df
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    df["_d"] = pd.to_datetime(df.get("gameday"), errors="coerce")
    df = df[df["_d"].notna()]
    return df.sort_values("_d", ascending=False).head(MAX_ROWS)


def _norm(text: str) -> str:
    return str(text).replace("\u2212", "-")


def _pairs(text: str, rx: re.Pattern) -> dict[str, float]:
    return {m.group(1): float(m.group(2)) for m in rx.finditer(_norm(text))}


def _card_rl(sv, row, home: str, away: str) -> str:
    return sv.runline_html(row, home, away)


def _check_runline_row(sport: str, page, sv) -> list[str]:
    """Same-row contract: the history row == the card's own render."""
    df = _decided(_latest(SPORTS[sport][2]))
    assert len(df), f"{sport}: no decided OOF rows in the latest artifact"
    problems: list[str] = []
    checked = 0
    for _, r in df.iterrows():
        rec = page._runline_row(r)
        home = str(r.get("home_team", ""))
        away = str(r.get("away_team", ""))
        card = _card_rl(sv, r, home, away)
        card_na = "RL: n/a" in card
        if rec is None:
            if not card_na:
                problems.append(
                    f"{sport} {r.get('game_id')}: history dropped the row "
                    f"while the card renders {card!r}")
            continue

        if card_na:
            problems.append(
                f"{sport} {r.get('game_id')}: card renders n/a but the "
                f"history quotes {rec['line']!r}")
            continue
        checked += 1

        # 1. LINE — same sides, same numbers as the card's run-line span.
        card_lines = _pairs(card, _RE_LINE_PAIRS)
        hist_lines = _pairs(rec["line"], _RE_LINE_PAIRS)
        if hist_lines != card_lines:
            problems.append(
                f"{sport} {r.get('game_id')}: line cell {rec['line']!r} != "
                f"card pair {card_lines} (card {card!r})")
            continue


        # Ladder invariants: opposite signs, half-step, never ±0.
        if len(hist_lines) != 2 or home not in hist_lines or away not in hist_lines:
            problems.append(
                f"{sport} {r.get('game_id')}: line cell does not carry both "
                f"teams: {rec['line']!r}")
            continue
        h_line, a_line = hist_lines[home], hist_lines[away]
        if h_line != -a_line:
            problems.append(
                f"{sport} {r.get('game_id')}: sides not opposite signs: "
                f"{rec['line']!r}")
        if h_line == 0 or abs(h_line) * 2 != round(abs(h_line) * 2):
            problems.append(
                f"{sport} {r.get('game_id')}: line off the 0.5 ladder: "
                f"{rec['line']!r}")
        if rec["pick"] and rec["pick"] != "—":
            p = _pairs(rec["pick"], _RE_PICK)
            pct = {m.group(1): int(m.group(3))
                   for m in _RE_CARD_PAIRS.finditer(_norm(card))}
            if len(p) != 1:
                problems.append(
                    f"{sport} {r.get('game_id')}: unparsable pick cell "
                    f"{rec['pick']!r}")
            else:
                team, line = next(iter(p.items()))
                if team not in card_lines:
                    problems.append(
                        f"{sport} {r.get('game_id')}: pick side {team} not on "
                        f"the card (card {card!r})")
                elif line != card_lines[team]:
                    problems.append(
                        f"{sport} {r.get('game_id')}: pick line {line} != "
                        f"card line {card_lines[team]} for {team}")
                else:
                    # 2. PICK % — the PICKED side's own card break-even.
                    want = next(
                        (int(m.group(3))
                         for m in _RE_CARD_PAIRS.finditer(_norm(card))
                         if m.group(1) == team), None)
                    got = next(
                        (int(m.group(3)) for m in _RE_PICK.finditer(
                            _norm(rec["pick"]))
                         if m.group(1) == team), None)
                    if want is None or got is None or want != got:
                        problems.append(
                            f"{sport} {r.get('game_id')}: pick break-even "
                            f"{got}% != card {want}% for {team} "
                            f"(pick {rec['pick']!r}, card {card!r})")

        # 4. Pushes only at whole-number lines.
        if rec.get("correct") == "push" and float(h_line).is_integer() is False:
            problems.append(
                f"{sport} {r.get('game_id')}: push graded on the half line "
                f"{rec['line']!r}")
    assert checked, f"{sport}: no comparable rows (parse drift?)"
    return problems


def _check_totals_row(sport: str, page, sv) -> list[str]:
    df = _decided(_latest(SPORTS[sport][2]))
    problems: list[str] = []
    checked = 0
    for _, r in df.iterrows():
        rec = page._totals_row(r)
        if rec is None:
            continue
        home = str(r.get("home_team", ""))
        away = str(r.get("away_team", ""))
        strip = sv.runengine_html(r, home, away)
        m = _RE_OU_CARD.search(_norm(strip))
        if m is None:
            continue
        checked += 1
        card_line, over_pct, under_pct = (m.group(1), int(m.group(2)),
                                          int(m.group(3)))
        p = _RE_OU_PICK.search(_norm(rec["pick"]))
        if p is None:
            # unpriced pick renders '—' — only acceptable if the card also
            # says n/a (which it does not while we got here).
            problems.append(
                f"{sport} {r.get('game_id')}: unparsable totals pick "
                f"{rec['pick']!r}")
            continue
        side, line, pct = p.group(1), p.group(2), int(p.group(3))
        # 3. LINE + side-consistent break-even (Under picks carry P(under)).
        if float(line) != float(card_line):
            problems.append(
                f"{sport} {r.get('game_id')}: totals line {line} != card "
                f"{card_line} (pick {rec['pick']!r}, strip {strip!r})")
        want = over_pct if side == "Over" else under_pct
        if pct != want:
            problems.append(
                f"{sport} {r.get('game_id')}: {side} pick shows {pct}% but "
                f"the card quotes {want}% (pick {rec['pick']!r}, "
                f"strip {strip!r})")
    assert checked, f"{sport}: no comparable totals rows (parse drift?)"
    return problems


def test_nfl_runline_history_quotes_the_card():
    problems = _check_runline_row("nfl", *SPORTS["nfl"][:2])
    assert not problems, "\n".join(problems[:20])


def test_nhl_runline_history_quotes_the_card():
    problems = _check_runline_row("nhl", *SPORTS["nhl"][:2])
    assert not problems, "\n".join(problems[:20])


def test_nfl_totals_history_quotes_the_card():
    problems = _check_totals_row("nfl", *SPORTS["nfl"][:2])
    assert not problems, "\n".join(problems[:20])


def test_nhl_totals_history_quotes_the_card():
    problems = _check_totals_row("nhl", *SPORTS["nhl"][:2])
    assert not problems, "\n".join(problems[:20])


def test_pickem_fair_line_renders_the_stop_on_both_surfaces():
    """A fair spread of 0 quotes the favorite-anchored ±0.5 stop on the
    card AND in the history — never a fake ±0.0 run line."""
    problems: list[str] = []
    for sport, (page, sv, dd) in SPORTS.items():
        df = _decided(_latest(dd))
        df = df[df["p_home_cover_0"].notna() & df["p_push_0"].notna()]
        assert len(df), f"{sport}: no rows with the pick'em split columns"
        r = df.iloc[0].copy()
        r["fair_spread"] = 0.0
        rec = page._runline_row(r)
        home, away = str(r["home_team"]), str(r["away_team"])
        card = sv.runline_html(r, home, away)
        if rec is None:
            problems.append(f"{sport}: pick'em row dropped from the history")
            continue
        card_lines = _pairs(card, _RE_LINE_PAIRS)
        hist_lines = _pairs(rec["line"], _RE_LINE_PAIRS)
        if hist_lines != card_lines or set(hist_lines) != {home, away}:
            problems.append(
                f"{sport}: pick'em line {rec['line']!r} != card {card!r}")
            continue
        if any(v == 0 for v in hist_lines.values()):
            problems.append(
                f"{sport}: pick'em rendered a ±0.0 line {rec['line']!r}")
        # favorite-anchored: the -0.5 side is the run-ML favorite on both.
        fav = nsv.half_stop_pair(r)[4] if sport == "nfl" \
            else nhl_sv.half_stop_pair(r)[4]
        fav_team = home if fav else away
        if hist_lines.get(fav_team) != -0.5:
            problems.append(
                f"{sport}: history is not favorite-anchored at the stop: "
                f"{rec['line']!r} (fav {fav_team})")
    assert not problems, "\n".join(problems)


def test_det_ari_history_stays_on_the_card_side():
    """The reported game (Oct 11, 2026 DET@ARI): the run-line history row
    must sit on the SAME side of the line per team as the dated game card.
    It used to read 'DET +4' while the card said 'DET −5'."""
    dd = SPORTS["nfl"][2]
    dated = dd / "nfl_run_engine_markets_20261011.csv"
    if not dated.exists():
        dated = _latest(dd)
    slate = pd.read_csv(dated, encoding="utf-8-sig")
    slate = slate[slate.get("kind") == "slate"]
    slate = slate[slate["game_id"] == "2026_05_DET_ARI"]
    if not len(slate):
        return  # the dated slate file rotated out of the family
    srow = slate.iloc[0]
    card_lines = _pairs(nsv.runline_html(srow, str(srow["home_team"]),
                                         str(srow["away_team"])),
                        _RE_LINE_PAIRS)

    hist = pd.read_csv(_latest(dd), encoding="utf-8-sig")
    hist = hist[hist.get("kind") == "oof"]
    hist = hist[hist["game_id"] == "2026_05_DET_ARI"]
    assert len(hist), "DET@ARI decided OOF row missing from the artifact"
    r = hist.iloc[0]
    r["_d"] = pd.to_datetime(r["gameday"])
    rec = nfl_page._runline_row(r)
    assert rec is not None, "DET@ARI dropped from the run-line history"
    hist_lines = _pairs(rec["line"], _RE_LINE_PAIRS)

    # Same orientation: each team on the same side of the line in both.
    for team in ("ARI", "DET"):
        assert team in hist_lines and team in card_lines, \
            f"{team} missing (card {card_lines}, history {hist_lines})"
        assert (hist_lines[team] > 0) == (card_lines[team] > 0), \
            (f"DET@ARI orientation flipped: card {card_lines} vs history "
             f"{hist_lines} ({rec['line']!r})")
    # And the away side must never show a positive 'DET +n' while the card
    # has Detroit laying points.
    if card_lines["DET"] < 0:
        assert hist_lines["DET"] < 0, \
            f"history quotes DET {hist_lines['DET']:+g} on a card that says " \
            f"DET {card_lines['DET']:+g}"
