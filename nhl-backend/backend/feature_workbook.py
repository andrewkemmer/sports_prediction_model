"""NHL RFE decision workbook generator — MLB structural parity.

Artifact-only companion to feature_selection.py. Performs no training and
never changes the active production feature list. Every sheet renders from
the trace JSON (``nhl_feature_selection_*.json``).

Sheets (matching the MLB workbook's structure):
  README | Dashboard | All Features (Master) | Production (n) | Candidates (n)
  RFE Run Detail | Grid Detail | Per-Fold Detail | Redundancy
  Coverage Gaps | Glossary | Feature Pool
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

try:
    from backend.feature_selection import workbook_filename
except ImportError:
    from feature_selection import workbook_filename

BACKEND = Path(__file__).resolve().parent
DELIVERY = BACKEND.parent / "data_delivery"
NAVY = "1F3864"
BLUE = PatternFill("solid", fgColor=NAVY)
WHITE = Font(color="FFFFFF", bold=True)

HEADER_ROW = ["Feature", "In Production", "Type", "Category", "Side",
              "Plain-English Description", "Window", "Units"]


def _latest_trace() -> Path | None:
    paths = sorted(DELIVERY.glob("nhl_feature_selection_*.json"))
    return paths[-1] if paths else None


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style_table(ws, ncols: int, widths: list[int] | None = None) -> None:
    """Header styling + column widths (MLB workbook presentation parity)."""
    for cell in ws[1]:
        cell.fill = BLUE
        cell.font = WHITE
        cell.alignment = Alignment(wrap_text=True)
    for idx in range(1, ncols + 1):
        if widths and idx <= len(widths):
            w = widths[idx - 1]
        else:
            w = min(48, max(14, max(
                (len(str(ws.cell(row=r, column=idx).value or ""))
                 for r in range(1, min(ws.max_row, 30) + 1)), default=0) + 2))
        ws.column_dimensions[get_column_letter(idx)].width = w
    ws.freeze_panes = "A2"


def _context_meta(trace: dict) -> dict[str, dict[str, Any]]:
    return (trace.get("feature_context") or {}).get("meta") or {}


def _feature_row(name: str, in_prod: bool, meta: dict[str, Any]) -> list[Any]:
    return [
        name,
        "Yes" if in_prod else "No",
        meta.get("type", ""),
        meta.get("category", ""),
        meta.get("side", ""),
        meta.get("description", ""),
        meta.get("window", ""),
        meta.get("units", ""),
    ]


def _sheet_readme(wb: Workbook, trace: dict, path: Path) -> None:
    ws = wb.create_sheet("README")
    ws.sheet_properties.tabColor = "808080"
    rows = [
        ["NHL Feature Selection Decision Workbook"],
        [],
        ["Trace", path.name],
        ["Trace schema", trace.get("schema", "")],
        ["Run date", trace.get("date", "")],
        ["Run mode", trace.get("run_mode", "")],
        ["Scoring population (games)", (trace.get("feature_context") or {}).get("n_rows", "")],
        ["Production width before run", trace.get("n_universe", "")],
        ["Candidate pool size", trace.get("n_candidates", "")],
        ["Full validation pool", trace.get("n_pool", "")],
        ["Trials scored", trace.get("n_trials", "")],
        ["Committed trials", trace.get("n_committed", "")],
        ["Failed trials", trace.get("n_failed", "")],
        ["Selected columns (record-only)", ", ".join(trace.get("selected_cols", []))],
        [],
        ["Default governance: this workbook does not change production. Adoption is explicit."],
        ["Commit rule: a trial commits only when its paired log-loss gain clears "
         "max(0.0005, 1 x paired SE) with AUC and ECE guardrails (RFE_NOISE_SIGMA = 1.0)."],
        ["Baseline log-loss", (trace.get("baseline_metrics") or {}).get("logloss")],
        ["Best log-loss reached", (trace.get("best_metrics") or {}).get("logloss")],
    ]
    for row in rows:
        ws.append(row)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 90


def _sheet_dashboard(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("Dashboard")
    ws.sheet_properties.tabColor = "2E75B6"
    selected = set(trace.get("selected_cols", []))
    steps = trace.get("steps", [])
    committed = [s for s in steps if s.get("committed")]
    failed = [s for s in steps if s.get("error")]
    adds = [s for s in committed if s.get("kind") in ("add", "grid")]
    removes = [s for s in committed if s.get("kind") == "remove"]
    ws.append(["RFE Run Dashboard"])
    ws.append([])
    ws.append(["Metric", "Value"])
    dash = [
        ("Trials scored", len(steps)),
        ("Committed", len(committed)),
        ("Committed removals", len(removes)),
        ("Committed additions", len(adds)),
        ("Failed trials (recorded, non-fatal)", len(failed)),
        ("Features selected (record-only)", len(selected)),
        ("Baseline log-loss", (trace.get("baseline_metrics") or {}).get("logloss")),
        ("Best log-loss", (trace.get("best_metrics") or {}).get("logloss")),
        ("Baseline AUC", (trace.get("baseline_metrics") or {}).get("auc")),
        ("Best AUC", (trace.get("best_metrics") or {}).get("auc")),
        ("Redundant pairs (|r| >= 0.9)",
         len((trace.get("feature_context") or {}).get("redundancy", []))),
    ]
    for label, value in dash:
        ws.append([label, value])
    ws.append([])
    ws.append(["Committed changes this run"])
    ws.append(["Step", "Kind", "Feature", "Log-loss gain", "Committed"])
    for s in committed:
        ws.append([s.get("step"), s.get("kind"), s.get("feature"),
                   s.get("logloss_gain"), s.get("committed")])
    _style_table(ws, 2, [34, 22])


def _sheet_master(wb: Workbook, trace: dict, sheet_name: str, tab_color: str,
                  names: list[str]) -> None:
    meta = _context_meta(trace)
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = tab_color
    ws.append(HEADER_ROW)
    for name in names:
        ws.append(_feature_row(name, True, meta.get(name, {})))
    _style_table(ws, len(HEADER_ROW), [30, 13, 22, 18, 20, 80, 18, 14])


def _sheet_candidates(wb: Workbook, trace: dict) -> None:
    meta = _context_meta(trace)
    selected = set(trace.get("selected_cols", []))
    pool_names = set(meta) | set(trace.get("candidate_pool", [])) | selected
    tested = {str(s.get("feature")) for s in trace.get("steps", [])
              if s.get("feature")} & pool_names
    ws = wb.create_sheet("Candidates")
    ws.sheet_properties.tabColor = "C55A11"
    ws.append(HEADER_ROW + ["Role in run"])
    seen: set[str] = set()
    for feature in trace.get("candidate_pool", []):
        if feature in seen or feature in selected:
            continue
        seen.add(feature)
        role = ("trialed, not committed" if feature in tested
                else "offered, not trialed")
        ws.append(_feature_row(feature, False, meta.get(feature, {})) + [role])
    for feature in sorted(tested - selected):
        if feature in seen:
            continue
        seen.add(feature)
        ws.append(_feature_row(feature, False, meta.get(feature, {}))
                  + ["trialed, not committed"])
    _style_table(ws, len(HEADER_ROW) + 1, [30, 13, 22, 18, 20, 80, 18, 14, 24])


def _sheet_run_detail(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("RFE Run Detail")
    ws.sheet_properties.tabColor = "7030A0"
    headers = ["step", "kind", "feature", "n_features", "logloss", "logloss_gain",
               "paired_se", "commit_threshold", "auc", "ece", "committed", "error"]
    ws.append(headers)
    for rec in trace.get("steps", []):
        metrics = rec.get("metrics") or {}
        ws.append([rec.get("step"), rec.get("kind"), rec.get("feature"),
                   rec.get("n_features"), metrics.get("logloss"),
                   rec.get("logloss_gain"),
                   rec.get("paired_se"), rec.get("commit_threshold"),
                   metrics.get("auc"), metrics.get("ece"), rec.get("committed"),
                   rec.get("error")])
    _style_table(ws, len(headers))


def _sheet_grid_detail(wb: Workbook, trace: dict) -> None:
    grid_steps = [s for s in trace.get("steps", []) if s.get("kind") == "grid"]
    if not grid_steps:
        return
    ws = wb.create_sheet("Grid Detail")
    ws.sheet_properties.tabColor = "ED7D31"
    ws.append(["Grid trials — every tested (adds / removes) combination"])
    ws.append([])
    headers = ["step", "state (adds / removes)", "adds", "removes", "n_features",
               "logloss", "logloss_gain", "paired_se", "commit_threshold",
               "auc", "ece", "committed", "error"]
    ws.append(headers)
    for rec in grid_steps:
        metrics = rec.get("metrics") or {}
        adds = ", ".join(rec.get("adds") or []) or "(none)"
        removes = ", ".join(rec.get("removes") or []) or "(none)"
        ws.append([rec.get("step"),
                   f"{adds} / {removes}" if removes != "(none)" else adds,
                   adds, removes, rec.get("n_features"),
                   metrics.get("logloss"), rec.get("logloss_gain"),
                   rec.get("paired_se"), rec.get("commit_threshold"),
                   metrics.get("auc"), metrics.get("ece"), rec.get("committed"),
                   rec.get("error")])
    _style_table(ws, len(headers))


def _sheet_per_fold(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("Per-Fold Detail")
    ws.sheet_properties.tabColor = "9E5EB5"
    ws.append(["step", "feature", "fold_id", "validation_window", "n_games", "logloss"])
    for rec in trace.get("steps", []):
        for fold in (rec.get("metrics") or {}).get("per_fold", []):
            ws.append([rec.get("step"), rec.get("feature"), fold.get("fold_id"),
                       fold.get("val_window"), fold.get("n_games"),
                       fold.get("logloss")])
    _style_table(ws, 6)


def _sheet_redundancy(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("Redundancy")
    ws.sheet_properties.tabColor = "2F8F83"
    pairs = (trace.get("feature_context") or {}).get("redundancy") or []
    ws.append(["Feature A", "Feature B", "Pearson r", "Note"])
    if not pairs:
        ws.append(["(none)", "(none)", "", "No |r| >= 0.9 pair among pool features "
                                          "on the scored frame."])
    for pr in pairs:
        ws.append([pr.get("a"), pr.get("b"), pr.get("r"),
                   "near-duplicate signal — consider one representative"])
    _style_table(ws, 4, [30, 30, 12, 52])


# ---------------------------------------------------------------------------
# API-coverage gap catalog — grounded in the extractor code. Authored
# constants — reviewed with each release, never auto-generated.
# ---------------------------------------------------------------------------

NHL_API_SOURCES = [
    ("official NHL API /v1/score/{date}",
     "Per game: teams, scores, SOG, venue, startTimeUTC, gameState, "
     "gameOutcome.lastPeriodType, goals[] (running scores).",
     "The complete schedule/board population: ids, season, gameType, scores "
     "(decided filter), SOG (team totals), venue, start times.",
     "goals[] per-play detail (the play-by-play endpoint is optional and not "
     "used by the production feature contract)."),
    ("official NHL API /v1/gamecenter/{id}/boxscore",
     "Per game: playerByGameStats per side (forwards/defense/goalies) with "
     "skater goals, assists, PIM, hits, powerPlayGoals, sog, "
     "faceoffWinningPctg, blockedShots, giveaways, takeaways; goalie lines "
     "with playerId, name, toi, decision.",
     "Team rollups: SOG, PPG, faceoff%, hits, blocks, PIM, giveaways, "
     "takeaways; decision goalie id/name/TOI + team goals against (the "
     "goalie rolling-state input).",
     "Per-skater rows beyond the team rollup (individual quality is "
     "second-order vs team trailing stats)."),
    ("MoneyPuck free downloads (optional enrichment, non-commercial)",
     "Shot-level xG CSVs per season (2007+), game-by-game team CSVs.",
     "OPTIONAL: load_moneypuck_shots caches the season shots file; the "
     "production contract never depends on it (honest NaN degradation).",
     "xG is not in any served feature; retained as a future candidate "
     "family (would need a leakage-safe trailing rollup first)."),
]

NHL_LOW_VALUE_FIELDS = [
    ("boxscore", "per-skater names/positions",
     "One skater among 20; tiny signal, heavy cardinality."),
    ("score", "goals[] running commentary",
     "In-game state — leaks if applied pre-game; kept out on purpose."),
    ("MoneyPuck xG", "shot-level xG",
     "No trailing rollup implemented yet (candidate family only)."),
]


def _sheet_coverage(wb: Workbook, trace: dict) -> None:
    """Coverage Gaps — MLB-parity gap catalog + per-feature table."""
    ws = wb.create_sheet("Coverage Gaps")
    ws.sheet_properties.tabColor = "C00000"
    for idx, w in enumerate([36, 48, 48, 44, 12, 12], start=1):
        ws.column_dimensions[get_column_letter(idx)].width = w

    r = 1

    def title_row(text: str) -> None:
        nonlocal r
        c = ws.cell(row=r, column=1, value=text)
        c.font = Font(bold=True, size=13, color=NAVY)
        r += 1

    def note_row(text: str) -> None:
        nonlocal r
        c = ws.cell(row=r, column=1, value=text)
        c.font = Font(italic=True, size=9)
        r += 1

    def section(text: str) -> None:
        nonlocal r
        r += 1
        c = ws.cell(row=r, column=1, value=text)
        c.fill = BLUE
        c.font = WHITE
        for idx in range(2, 7):
            ws.cell(row=r, column=idx).fill = BLUE
        r += 1

    def header(cells: list[str]) -> None:
        nonlocal r
        for idx, v in enumerate(cells, start=1):
            c = ws.cell(row=r, column=idx, value=v)
            c.font = Font(bold=True)
        r += 1

    def row(cells: list[Any]) -> None:
        nonlocal r
        for idx, v in enumerate(cells, start=1):
            ws.cell(row=r, column=idx, value=v)
        r += 1

    title_row("Coverage Gaps — what the API pulls serve vs what production uses")
    note_row("Sections A–E are the MLB-parity gap catalog (authored, grounded in "
             "ingestion.py). The last block is the run's own per-feature "
             "coverage. Record-only: nothing here changes production without "
             "an explicit adoption run.")

    section("SECTION A — API payload inventory: serves vs extracts")
    header(["API endpoint", "What the payload serves", "What the pipeline "
            "extracts today", "What is left unused", "", ""])
    for a, b, c_, d in NHL_API_SOURCES:
        row([a, b, c_, d, "", ""])

    section("SECTION D — Declared candidates not trialed by this run")
    candidates = trace.get("candidate_pool") or []
    if candidates:
        meta = _context_meta(trace)
        cov = (trace.get("feature_context") or {}).get("coverage") or {}
        header(["Candidate", "Type", "Category", "Description", "", ""])
        for name in candidates:
            m = meta.get(name, {})
            pct = (cov.get(name) or {}).get("pct")
            row([name, m.get("type", ""), m.get("category", ""),
                 m.get("description", ""),
                 f"{pct:.1f}%" if pct is not None else "", ""])
    else:
        note_row("No declared candidates in this run (config.RFE_CANDIDATE_COLS "
                 "is empty) — the sweep was removals-only over the production "
                 "contract.")

    section("SECTION E — Cataloged but NOT recommended (honesty section)")
    header(["Source", "Field(s)", "Why not recommended", "", "", ""])
    for src, field, why in NHL_LOW_VALUE_FIELDS:
        row([src, field, why, "", "", ""])

    cov = (trace.get("feature_context") or {}).get("coverage") or {}
    meta = _context_meta(trace)
    n_rows = (trace.get("feature_context") or {}).get("n_rows", 0)
    section("PER-FEATURE COVERAGE — non-null share in the scored frame")
    if cov:
        header(["Feature", "Type", "Side", "Non-null games", "Coverage %",
                "Scored rows"])
        rows = sorted(((float(c.get("pct", 0.0)), name, c)
                       for name, c in cov.items()), key=lambda t: t[0])
        for pct, name, c in rows:
            m = meta.get(name, {})
            row([name, m.get("type", ""), m.get("side", ""),
                 c.get("n", 0), pct, n_rows])
    else:
        note_row("Trace carries no feature_context coverage (legacy trace) — "
                 "run an RFE sweep to populate this table.")
    ws.freeze_panes = None


def _sheet_glossary(wb: Workbook, trace: dict) -> None:
    meta = _context_meta(trace)
    ws = wb.create_sheet("Glossary")
    ws.sheet_properties.tabColor = "808080"
    ws.append(["Feature", "Plain-English Description", "Definition",
               "Point-in-time rule", "Missing-value policy", "Source"])
    for name in sorted(meta):
        m = meta[name]
        ws.append([name, m.get("description", ""), m.get("definition", ""),
                   m.get("point_in_time_rule", ""),
                   m.get("missing_value_policy", ""), m.get("source", "")])
    _style_table(ws, 6, [30, 60, 80, 55, 45, 40])


def _sheet_feature_pool(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("Feature Pool")
    ws.append(["feature", "role", "tested_in_trace"])
    selected = set(trace.get("selected_cols", []))
    tested = {str(x.get("feature")) for x in trace.get("steps", [])}
    for feature in trace.get("selected_cols", []):
        ws.append([feature, "production contract", feature in tested])
    for feature in sorted(tested - selected):
        ws.append([feature, "trialed candidate/removal", True])
    for feature in sorted(set(trace.get("candidate_pool", [])) - tested - selected):
        ws.append([feature, "auto candidate (offered, not trialed)", False])
    _style_table(ws, 3)


def generate_workbook(trace_path: str | None = None, out_path: str | None = None) -> str | None:
    path = Path(trace_path) if trace_path else _latest_trace()
    if path is None or not path.exists():
        return None
    trace = _load_trace(path)
    target = Path(out_path) if out_path else DELIVERY / workbook_filename(trace, path)
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["NHL Feature Selection Decision Workbook"])
    ws.append(["Trace", path.name])
    ws.append(["Run mode", trace.get("run_mode", "")])
    ws.append(["Production width before run", trace.get("n_universe", "")])
    ws.append(["Pool size", trace.get("n_pool", "")])
    ws.append(["Trials scored", trace.get("n_trials", "")])
    ws.append(["Committed trials", trace.get("n_committed", "")])
    ws.append(["Selected columns (record-only)", ", ".join(trace.get("selected_cols", []))])
    ws.append([])
    ws.append(["Default governance: this workbook does not change production. Adoption is explicit."])

    selected = trace.get("selected_cols", [])
    _sheet_readme(wb, trace, path)
    _sheet_dashboard(wb, trace)
    _sheet_master(wb, trace, "All Features (Master)", "548235", selected)
    _sheet_master(wb, trace, f"Production ({len(selected)})", "375623", selected)
    _sheet_candidates(wb, trace)
    _sheet_run_detail(wb, trace)
    _sheet_grid_detail(wb, trace)
    _sheet_per_fold(wb, trace)
    _sheet_redundancy(wb, trace)
    _sheet_coverage(wb, trace)
    _sheet_glossary(wb, trace)
    _sheet_feature_pool(wb, trace)

    for sheet in wb.worksheets:
        if sheet.title == "Summary":
            continue
        for row in sheet.iter_rows():
            for cell in row:
                if cell.row > 1:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
    for sheet in wb.worksheets:
        if sheet.title == "Summary":
            continue
        for cell in sheet[1]:
            cell.fill = BLUE
            cell.font = WHITE
            cell.alignment = Alignment(wrap_text=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    wb.save(target)
    return str(target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--out")
    args = ap.parse_args()
    print(generate_workbook(args.trace, args.out) or "no trace found")
