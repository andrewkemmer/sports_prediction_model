"""Local render test: MLB vs NBA Totals & Run Lines.

Renders BOTH pages through Streamlit's AppTest against their REAL
data_delivery artifacts and diffs the rendered element tree — headings,
tabs, expanders, widget inventory and table schemas. The point is to compare
what a user actually SEES, not what the source files look like.

    python compare_totals_dashboards.py            # print both + the diff
    python compare_totals_dashboards.py --assert   # non-zero on drift

An earlier version of this harness staged synthetic 2099-dated artifacts into
data_delivery to force a deterministic frame. That was a mistake twice over:
a future-dated file SHADOWS the real artifact (so the page under test silently
stopped being the page users see), and the synthetic monitor block omitted
`fit` entirely, which is exactly how an unguarded `int(None)` reached the
rendered page unnoticed. Both sports ship real artifacts, so both are read
as-is.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent
sys.path.insert(0, str(FRONTEND))

# Each sport's page lives in its OWN module — rendering MLB's markets.py for
# NBA would compare MLB against itself and "prove" nothing.
PAGE_FILE = {"mlb": "markets.py", "nba": "nba_markets_page.py"}

# MLB's three top-level sections, in order. NBA must reach parity on these.
REQUIRED_SECTIONS = (
    "Diagnostics",
    "Prediction History",
    "Run-Line & Totals Monitor",
)
REQUIRED_EXPANDERS = (
    "Calibration Cards",
    "Distributional Fit Diagnostics",
    "Rolling History (last 10 points per card)",
)
REQUIRED_TABS = ("Distribution", "Relativized", "Pooled lines",
                 "Game Total Lines", "Spread Lines")
# Widget kinds the history + monitor sections must use, per the MLB page.
REQUIRED_WIDGETS = ("selectbox", "TABLE", "metric", "TAB", "EXPANDER")


def _text_of(elements) -> list[str]:
    out = []
    for element in elements:
        value = str(getattr(element, "value", ""))
        value = re.sub(r"<[^>]+>", "", value).strip()
        if value:
            out.append(value)
    return out


def skeleton(app) -> list[str]:
    """Ordered outline of what the page actually rendered."""
    lines: list[str] = []

    for value in _text_of(app.title) + _text_of(app.header) + \
            _text_of(app.subheader):
        lines.append(f"HEADING :: {value}")
    for value in _text_of(app.markdown):
        if value.startswith("#"):
            lines.append(f"HEADING :: {value.lstrip('#').strip()}")
        else:
            lines.append(f"markdown  :: {value[:70]}")
    for value in _text_of(app.caption):
        lines.append(f"caption   :: {value[:70]}")
    for value in _text_of(app.info):
        lines.append(f"info      :: {value[:70]}")
    for value in _text_of(app.warning):
        lines.append(f"warning   :: {value[:70]}")
    for element in getattr(app, "tabs", []):
        label = element.label
        if isinstance(label, (list, tuple)):
            label = ", ".join(str(x) for x in label)
        lines.append(f"TAB       :: {label}")
    for element in getattr(app, "expander", []):
        lines.append(f"EXPANDER  :: {element.label}")
    for element in getattr(app, "dataframe", []):
        value = getattr(element, "value", None)
        try:
            columns = list(value.columns)
        except Exception:
            columns = []
        # A raw artifact dump is itself a finding: >25 columns means the
        # section is showing the artifact, not a modelled table.
        marker = " <-- RAW ARTIFACT DUMP" if len(columns) > 25 else ""
        lines.append(f"TABLE     :: {columns}{marker}")
    for kind, attr in (("selectbox", "selectbox"), ("radio", "radio"),
                       ("date_input", "date_input"),
                       ("text_input", "text_input")):
        for element in getattr(app, attr, []):
            options = list(getattr(element, "options", []) or [])
            suffix = f" opts={options[:6]}" if options else ""
            lines.append(f"{kind:10} :: {element.label}{suffix}")
    for kind in ("metric", "number_input", "slider", "button", "toggle",
                 "multiselect", "data_editor"):
        for element in getattr(app, kind, []):
            lines.append(f"{kind:10} :: {element.label}")
    return lines


def render(sport: str) -> list[str]:
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_data.clear()
    app = AppTest.from_file(str(FRONTEND / PAGE_FILE[sport]), default_timeout=240)
    app.session_state["sport"] = sport
    app.session_state["gh_owner"] = ""
    app.session_state["gh_repo"] = ""
    app.session_state["gh_branch"] = "main"
    app.run()
    if app.exception:
        details = "\n".join(str(getattr(e, "value", e)) for e in app.exception)
        return [f"EXCEPTION  :: {details[:600]}"]
    return skeleton(app)


def _kind_of(line: str) -> str:
    """The element kind prefix. Compared exactly — a ``startswith("TAB")``
    also matches TABLE, which silently counted dataframes as tabs."""
    return line.split(" ::", 1)[0].strip() if " ::" in line else ""


def _label_of(line: str) -> str:
    return line.split(" ::", 1)[1].strip() if " ::" in line else line


def _kinds(lines: list[str]) -> set[str]:
    return {_kind_of(l) for l in lines if " ::" in l}


def check(sport: str, lines: list[str]) -> list[str]:
    """Contract checks against MLB's page. Returns a list of failures."""
    failures: list[str] = []
    headings = [_label_of(l) for l in lines if _kind_of(l) == "HEADING"]
    if any(l.startswith("EXCEPTION") for l in lines):
        return [f"{sport}: page raised on render — {lines[0][:300]}"]

    joined = " | ".join(headings)
    for section in REQUIRED_SECTIONS:
        if section not in joined:
            failures.append(f"{sport}: missing top-level section {section!r}")

    tabs = [l for l in lines if _kind_of(l) == "TAB"]
    if len(tabs) != len(REQUIRED_TABS):
        failures.append(f"{sport}: {len(tabs)} diagnostics tabs, "
                        f"expected {len(REQUIRED_TABS)}")
    for want in REQUIRED_TABS:
        if not any(want.lower() in _label_of(l).lower() for l in tabs):
            failures.append(f"{sport}: missing diagnostics tab {want!r}")

    expanders = [_label_of(l) for l in lines if _kind_of(l) == "EXPANDER"]
    for want in REQUIRED_EXPANDERS:
        if not any(want in e for e in expanders):
            failures.append(f"{sport}: missing expander {want!r}")

    kinds = _kinds(lines)
    for want in REQUIRED_WIDGETS:
        if want not in kinds:
            failures.append(f"{sport}: no {want} rendered")

    for line in lines:
        if "RAW ARTIFACT DUMP" in line:
            failures.append(f"{sport}: a section is dumping the raw artifact "
                            f"instead of a modelled table ({len(_label_of(line))} cols)")
            break

    # Run-engine drift must carry MLB's decision + weight columns (the
    # 2026-10-08 NBA parity fix: PSI ADJ./SHIFT SE/MODEL WEIGHT were
    # missing and the caption pointed at the moneyline monitor instead of
    # the distribution model). The table header rides the skeleton's
    # truncated markdown line right after the heading.
    drift_block = ""
    for i, line in enumerate(lines):
        if _kind_of(line) == "HEADING" and \
                "Run-Engine Feature Drift" in _label_of(line):
            drift_block = " ".join(lines[i:i + 3])
            break
    if "PSI ADJ." not in drift_block:
        failures.append(f"{sport}: run-engine drift table is missing "
                        "MLB's PSI ADJ. column")
    if "MODEL WEI" not in drift_block:
        failures.append(f"{sport}: run-engine drift table is missing the "
                        "MODEL WEIGHT column")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert", action="store_true", dest="strict",
                        help="exit non-zero when the parity contract fails")
    args = parser.parse_args()

    for key in ("GITHUB_OWNER", "GITHUB_REPO", "GITHUB_BRANCH"):
        os.environ.setdefault(key, "")

    rendered = {sport: render(sport) for sport in ("mlb", "nba")}

    for sport in ("mlb", "nba"):
        print("=" * 100)
        print(f"{sport.upper()} — Today's Totals & Run Lines")
        print("=" * 100)
        for line in rendered[sport]:
            print("  " + line)
        print()

    print("=" * 100)
    print("STRUCTURAL DIFF (headings / tabs / expanders)")
    print("=" * 100)
    for sport in ("mlb", "nba"):
        keys = {l.split(" ::")[0].strip() for l in rendered[sport]}
        print(f"  {sport.upper()}: {sum(1 for l in rendered[sport] if l.split(' ::')[0].strip() in keys)} elements")
    print()
    for sport in ("nba", "mlb"):
        other = "mlb" if sport == "nba" else "nba"
        mine = {l.split(" ::")[0].strip() for l in rendered[sport]}
        theirs = {l.split(" ::")[0].strip() for l in rendered[other]}
        print(f"  WIDGET KINDS {sport.upper()}: {sorted(mine)}")
        print(f"  MISSING vs {other.upper()}: {sorted(theirs - mine)}")
        print()

    print("=" * 100)
    print("MLB TABLE SCHEMAS")
    print("=" * 100)
    for line in rendered["mlb"]:
        if line.startswith("TABLE"):
            print("  " + line)
    print("\nNBA TABLE SCHEMAS")
    for line in rendered["nba"]:
        if line.startswith("TABLE"):
            print("  " + line)

    failures = (check("mlb", rendered["mlb"]) + check("nba", rendered["nba"]))
    print()
    print("=" * 100)
    if failures:
        print(f"PARITY CONTRACT: {len(failures)} FAILURE(S)")
        for line in failures:
            print("  - " + line)
    else:
        print("PARITY CONTRACT: PASS — both pages satisfy MLB's section, tab, "
              "expander and widget contract")
    return 1 if (failures and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
