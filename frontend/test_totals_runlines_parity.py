"""Side-by-side structural parity smoke test for the Totals & Run Lines
dashboard.

Renders all four sports (MLB, NHL, NBA, NFL) through the same
``compare_totals_dashboards.render()`` the parity script uses and asserts
each page satisfies MLB's contract: top-level sections, diagnostics tabs,
expanders, widget kinds, the run-engine drift table columns, and the
winner-card metric blocks (Win rate / AUC / Pooled ECE-cal / Pooled
ECE-raw / Pooled Brier / Pooled Logloss) on all three run-engine cards.

This is the regression net for the 2026-10-11 alignment work: the NHL logloss
fix, the NBA run-engine monitor rebuild (winner cards + per-line
market_metrics), the rolling ECE-calibrated history panel on all three
challenger sports, the artifact-PMF distribution chart, and the
50%..55%+ confidence thresholds.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import compare_totals_dashboards as ctd

SPORTS = ("mlb", "nhl", "nba", "nfl")


@pytest.fixture(scope="module")
def rendered() -> dict:
    return {sport: ctd.render(sport) for sport in SPORTS}


def test_every_sport_satisfies_mlbs_contract(rendered):
    failures: list[str] = []
    for sport in SPORTS:
        failures.extend(ctd.check(sport, rendered[sport]))
    assert not failures, "\n".join(failures)


def test_widget_kinds_are_identical_to_mlbs(rendered):
    mlb_kinds = {l.split(" ::")[0].strip() for l in rendered["mlb"] if " ::" in l}
    for sport in ("nhl", "nba", "nfl"):
        mine = {l.split(" ::")[0].strip() for l in rendered[sport] if " ::" in l}
        missing = sorted(mlb_kinds - mine)
        assert not missing, f"{sport}: widget kinds missing vs MLB: {missing}"


def test_rolling_ece_history_panel_renders_with_rows(rendered):
    """Each challenger sports page must render MLB's 'Rolling ECE-Calibrated
    History' heading AND the per-card rolling table (rows, not an empty
    state) — NBA's was missing entirely before the rebuild."""
    for sport in ("nhl", "nba", "nfl"):
        lines = rendered[sport]
        headings = [l.split(" ::", 1)[1].strip() for l in lines
                    if l.startswith("HEADING ::")]
        assert any("Rolling ECE-Calibrated History" in h for h in headings), \
            f"{sport}: missing the Rolling ECE-Calibrated History section"
        tables = [l for l in lines if l.startswith("TABLE")]
        assert any("'Line'" in t and "'ECE-cal'" in t for t in tables), \
            f"{sport}: rolling history table did not render rows"


def _cal_note_expression(source: str, page: str):
    """Compile the page's real ``_cal_note`` value expression.

    The note lives on the page's PRIMARY render path: the first top-level
    ``def run()`` for the challenger pages (nba/nhl/nfl — markets.py's
    module-scope dispatch calls exactly that), or, for MLB's markets.py
    (module-scope page, no ``run()``), its top-level module body. An
    unrestricted ``ast.walk`` of the whole file could bind an assignment
    from an unrelated helper instead: NBA defines a module-level
    ``def _cal_note(monitor)`` whose BODY's assignment reads ``fit`` — not
    ``monitor`` — so evaluating it with the monitor fixture raised
    NameError, and under ``pytest -x`` that teardown crash aborted the
    run before the real assertion ran.
    """
    tree = ast.parse(source)
    run_fn = next((n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    if run_fn is not None:
        nodes = [n for n in ast.walk(run_fn) if isinstance(n, ast.Assign)]
    else:
        # Module-scope page (markets.py): every assignment on the module's
        # own execution path — including ones nested in if/else branches
        # (the diagnostics section is an else-block) — but NEVER inside a
        # def/class (the NBA ``def _cal_note`` trap).
        def _module_assignments(stmts):
            found = []
            for stmt in stmts:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    continue
                if isinstance(stmt, ast.Assign):
                    found.append(stmt)
                found.extend(_module_assignments(                    list(getattr(stmt, "body", []))
                    + list(getattr(stmt, "orelse", []))
                    + list(getattr(stmt, "finalbody", []))))
            return found
        nodes = _module_assignments(tree.body)
    node = next((n for n in nodes
                 if any(isinstance(t, ast.Name) and t.id == "_cal_note"
                        for t in n.targets)), None)
    assert node is not None, \
        f"{page}: no _cal_note assignment on the primary render path"
    return compile(ast.Expression(node.value), "<history-note>", "eval")


@pytest.mark.parametrize("sport", ["nfl", "nhl"])
def test_history_note_identifies_issued_challenger_prices(sport):
    # Execute the page's real note assignment without fetching stale artifacts.
    source = (ctd.FRONTEND / ctd.PAGE_FILE[sport]).read_text(encoding="utf-8")
    expression = _cal_note_expression(source, sport)
    monitor = {"fit": {"family": "two_sided_asymmetric_gaussian_split_skew"}}
    note = eval(expression, {"monitor": monitor})
    assert "whole-CDF-calibrated" in note and "Monte Carlo" in note
    assert "RAW" not in note and "Platt" not in note
    assert "predates" in eval(expression, {"monitor": {}})


def test_history_note_extractor_binds_mlbs_module_scope_reference():
    """markets.py renders at module scope (no ``run()``): the extractor must
    bind its top-level ``_cal_note`` assignment — the one calling
    ``_market_prob_note`` — and never wander into a helper def the way an
    unrestricted tree walk could."""
    source = (ctd.FRONTEND / "markets.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not [n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "run"], \
        "markets.py is the module-scope reference page (no run())"
    expression = _cal_note_expression(source, "mlb")
    # The module-scope assignment is ``_market_prob_note(date_str)``: the
    # compiled expression can only name the note function + its date arg.
    assert expression.co_names == ("_market_prob_note", "date_str")


def test_mlb_probability_note_reads_challenger_family_metadata():
    # markets.py renders at module scope; isolate the real pure note function.
    source = (ctd.FRONTEND / "markets.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_market_prob_note")
    from types import SimpleNamespace
    meta = {"family": "two_sided_asymmetric_gaussian_split_skew"}
    utils = SimpleNamespace(get_source_config=lambda: {},
                            _fetch_bytes=lambda *a, **kw: (json.dumps(meta).encode(), None))
    namespace = {"utils": utils, "json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<probability-note>", "exec"), namespace)
    note = namespace["_market_prob_note"]("20261012")
    assert "whole-CDF-calibrated" in note and "Monte Carlo" in note
    assert "RAW" not in note and "Platt" not in note
    meta.clear()
    assert "predates" in namespace["_market_prob_note"]("20261012")


def test_distribution_view_toggles_home_and_away_on_every_sport(rendered):
    """2026-10-16: the Distribution view toggles Total / Spread / Home /
    Away on ALL FOUR sports — Home and Away render the per-side scoring
    law (home/away runs, points or goals) next to totals and spreads."""
    for sport in SPORTS:
        radios = [l for l in rendered[sport]
                  if l.startswith("radio") and "Distribution view" in l]
        assert radios, f"{sport}: no Distribution view radio rendered"
        assert all("opts=['Total', 'Spread', 'Home', 'Away']" in l
                   for l in radios), f"{sport}: {radios}"
