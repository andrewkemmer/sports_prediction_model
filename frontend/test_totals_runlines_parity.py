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


@pytest.mark.parametrize("sport", ["nfl", "nhl"])
def test_history_note_identifies_issued_challenger_prices(sport):
    # Execute the page's real note assignment without fetching stale artifacts.
    source = (ctd.FRONTEND / ctd.PAGE_FILE[sport]).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_cal_note" for t in n.targets))
    expression = compile(ast.Expression(node.value), "<history-note>", "eval")
    monitor = {"fit": {"family": "two_sided_asymmetric_gaussian_split_skew"}}
    note = eval(expression, {"monitor": monitor})
    assert "whole-CDF-calibrated" in note and "Monte Carlo" in note
    assert "RAW" not in note and "Platt" not in note
    assert "predates" in eval(expression, {"monitor": {}})


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
