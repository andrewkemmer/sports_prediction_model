"""NFL RFE decision workbook generator — MLB structural parity.

This is an artifact-only companion to feature_selection.py. It performs no
training and never changes the active production feature list. Every sheet
renders from the trace JSON (``nfl_feature_selection_*.json``) — the workbook
re-reads nothing else and re-derives nothing:

  trace["feature_context"]["meta"]       -> Master/Production/Candidates/Glossary
  trace["feature_context"]["coverage"]   -> Coverage Gaps
  trace["feature_context"]["redundancy"] -> Redundancy
  trace["steps"]                         -> RFE Run Detail / Per-Fold / Grid Detail

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
    paths = sorted(DELIVERY.glob("nfl_feature_selection_*.json"))
    return paths[-1] if paths else None


def _load_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage(trace: dict) -> pd.DataFrame:
    frame = pd.DataFrame(trace.get("steps", []))
    if frame.empty:
        return frame
    return frame


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
        ["NFL Feature Selection Decision Workbook"],
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
         "max(0.0005, 2 x paired SE) with AUC and ECE guardrails."],
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
    # Only real pool features count as "tested" here — grid-state labels
    # ("a+b / c") and other composite step names are not features.
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
    """One sheet for grid-mode runs, rendered from the trace's grid steps.

    Silent no-op for normal (non-grid) runs — the sheet is only created when
    the trace actually carries grid trials (MLB parity).
    """
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


# --------------------------------------------------------------------------- #
# API-coverage gap catalog — grounded in the extractor code, verified against
# ingestion.py (PBP_NEEDS narrow + schedule keep-list) and qb_enrichment.py
# (QB-only player_stats). What each payload serves vs what the pipeline
# extracts; the rest is cataloged as candidate features or honestly
# deprioritized. Authored constants — reviewed with each release, never
# auto-generated.
# --------------------------------------------------------------------------- #

NFL_API_SOURCES = [
    # (API endpoint, What the payload serves, What the pipeline extracts today,
    #  What is left unused)
    ("nflverse play-by-play\nnflreadpy.load_pbp (nflfastR pbp, ~400 cols/play)",
     "Per play: teams, yards, EPA (play + QB), WPA, air yards, YAC, "
     "completion-probability (cpoe), down & distance, formation "
     "(shotgun/no-huddle), timeouts, score state, penalties, third/fourth-down "
     "conversions, field-goal results, drive metadata — plus the payload's own "
     "temp/wind/humidity and roof/surface columns.",
     "Narrowed at load to PBP_NEEDS (21 columns). The game rollup "
     "(features.pbp_team_agg) then aggregates ONLY yards_gained and "
     "game_seconds_remaining (total yards, plays, pace).",
     "Everything else — including columns KEPT but never aggregated: epa, "
     "qb_epa, interception, fumble_lost, sack, pass_attempt, passing_yards, "
     "penalty, penalty_yards, penalty_team, third_down_converted/failed, "
     "yardline_100, touchdown, field_goal_result, drive. Also dropped at "
     "load: air_yards, yac, cpoe, wpa, down/ydstogo, shotgun/no_huddle, "
     "score_differential, timeouts."),
    ("nflverse schedules\nnflreadpy.load_schedules",
     "Per game: betting lines (spread_line, total_line, over_under_line), "
     "weather (temp, wind), stadium/roof/surface/location, referee, game_type, "
     "overtime, division flag, QB ids/names.",
     "20-column keep-list; features use teams/scores/date/week/div_game and "
     "QB ids/names (card enrichment).",
     "Kept-but-unused: roof, stadium, surface, location, referee, game_type. "
     "Dropped entirely: spread_line, total_line, over_under_line, temp, wind, "
     "weather, overtime — NFL has NO weather and NO market-implied features "
     "(MLB has both)."),
    ("nflverse player stats\nnflreadpy.load_player_stats (all positions)",
     "Weekly player stats for every position: passing, rushing, receiving "
     "volumes and efficiency, ids/names.",
     "QB rows only (position == 'QB'): passer-rating components for card "
     "enrichment. No model features.",
     "RB/WR/TE usage (carries, targets, team pass-rate, RB load, WR1 "
     "threat), defensive player stats."),
    ("nflverse teams\nnflreadpy.load_teams",
     "Team abbr, full names, conference/division, colors, logos.",
     "Abbr -> name map for the frontend only.",
     "Conference/division membership (division-rank context), nothing else "
     "modeled."),
]

# (source, field, what it is, candidate feature, impact, effort, note)
NFL_UNUSED_API_FIELDS = [
    ("pbp (KEPT)", "epa / qb_epa",
     "Play- and QB-level expected points added.",
     "Trailing EPA/play and EPA-per-dropback diffs (efficiency beyond yards).",
     "High", "Low",
     "Already in the pull cache — only the rollup is missing."),
    ("pbp (KEPT)", "interception / fumble_lost",
     "Turnover events per play.",
     "Trailing turnover-margin EWM; protection/takeaway edge.",
     "High", "Low",
     "Kept in PBP_NEEDS, never aggregated."),
    ("pbp (KEPT)", "sack / pass_attempt",
     "Pressure allowed and dropback volume.",
     "Sack-rate and dropback-rate diffs (OL + playcalling).",
     "Medium", "Low", "Kept in PBP_NEEDS, never aggregated."),
    ("pbp (KEPT)", "third_down_converted / third_down_failed",
     "Third-down outcomes.",
     "Trailing conversion-rate diff (situational strength).",
     "Medium", "Low", "Kept in PBP_NEEDS, never aggregated."),
    ("pbp (KEPT)", "penalty / penalty_yards",
     "Penalty events and yardage.",
     "Trailing penalty-yards-per-game diff (discipline).",
     "Medium", "Low", "Kept in PBP_NEEDS, never aggregated."),
    ("pbp (KEPT)", "yardline_100 / touchdown / field_goal_result",
     "Field position, scoring plays, FG outcomes.",
     "Red-zone finish rate, starting-field-position edge, kicker accuracy.",
     "Medium", "Medium", "Kept in PBP_NEEDS, never aggregated."),
    ("pbp (dropped)", "air_yards / yac / cpoe",
     "Passing depth, separation after catch, accuracy over expectation.",
     "Trailing passing-depth and accuracy diffs ( qb play quality).",
     "High", "Low", "One-line additions to PBP_NEEDS + rollup."),
    ("pbp (dropped)", "shotgun / no_huddle / drive",
     "Formation tendency and drive counts.",
     "Style/pace complements to the existing plays-per-minute feature.",
     "Low", "Low", ""),
    ("schedules (dropped)", "temp / wind / weather",
     "Observed game-day environment.",
     "Wind-speed and temperature bands for outdoor games (MLB weather parity).",
     "Medium", "Low", "Payload carries it; no weather feature exists today."),
    ("schedules (dropped)", "spread_line / total_line / over_under_line",
     "Market-implied win margin and total.",
     "Market-anchor features or blend components (MLB market parity).",
     "High", "Low", "Closing lines arrive with the schedule payload."),
    ("schedules (KEPT)", "roof / surface",
     "Roof state and playing surface.",
     "is_dome_home exists; turf-vs-grass interaction still unused.",
     "Low", "Low", "Kept at load, never read by features."),
    ("player_stats", "RB carries / WR targets",
     "Skill-position usage.",
     "Team pass-rate tendency, RB-load and WR1-threat diffs.",
     "Medium", "Medium", "Requires removing the QB-only filter."),
]

# Endpoints the codebase never calls at all.
NFL_UNLOADED_ENDPOINTS = [
    ("load_injuries", "Weekly injury reports with participation status.",
     "Starter-out availability flags (QB/edge/left-tackle most valuable)."),
    ("load_snap_counts", "Per-player snap counts by week.",
     "Workload context: RB committees, defensive snap wear."),
    ("load_ngs", "Next-Gen Stats: passing/rushing/receiving efficiency.",
     "Deeper QB/RB quality measures beyond box-score aggregates."),
    ("load_officials", "Per-game officiating crews.",
     "Crew penalty-rate context (second-order)."),
    ("load_participation", "Advanced participation and alignment data.",
     "Coverage/pressure context (higher effort)."),
    ("load_pfr", "PFR advanced passing/rushing metrics.",
     "Pressure-rate and coverage-grade context."),
    ("load_qbr", "ESPN QBR (weekly and season).",
     "QB quality signal complementary to passer rating."),
]

NFL_LOW_VALUE_FIELDS = [
    ("pbp", "score_differential / wpa / timeouts",
     "In-game state — leaks if applied pre-game; kept out on purpose."),
    ("pbp", "total_line-derived per-play pace",
     "Already captured by the plays-per-minute feature; no second signal."),
    ("schedules", "referee",
     "One official per game; tiny signal, heavy cardinality."),
    ("schedules", "game_type",
     "Already filtered to regular-season games before features."),
    ("player_stats", "fantasy points",
     "Construction artifact, no independent edge."),
]

NFL_CATEGORY_ORDER = [
    "Team State", "Team Offense", "Team Defense", "Quarterback",
    "Skill Positions", "Injuries & Availability", "Playcalling & Situation",
    "Weather", "Stadium & Venue", "Market Lines", "Officiating",
    "Schedule & Travel", "League Context",
]


def _sheet_coverage(wb: Workbook, trace: dict) -> None:
    """Coverage Gaps — MLB-parity 5-section API-gap catalog + per-feature table.

    Sections A–E mirror the MLB workbook's api_coverage_gaps catalog; the
    final block keeps the per-feature non-null coverage table from the run
    trace. Sections A/B/C/E are authored constants grounded in the extractor
    code; D renders the run's own candidate pool.
    """
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
             "ingestion.py / qb_enrichment.py). The last block is the run's own "
             "per-feature non-null coverage. Record-only: nothing here changes "
             "production without an explicit adoption run.")

    # -- Section A --------------------------------------------------------
    section("SECTION A — API payload inventory: serves vs extracts")
    header(["API endpoint", "What the payload serves", "What the pipeline "
            "extracts today", "What is left unused", "", ""])
    for a, b, c_, d in NFL_API_SOURCES:
        row([a, b, c_, d, "", ""])

    # -- Section B --------------------------------------------------------
    section("SECTION B — Unused payload fields -> candidate features")
    header(["Source", "Field(s)", "What it is", "Candidate feature",
            "Impact", "Effort"])
    for src, field, what, cand, impact, effort, note in NFL_UNUSED_API_FIELDS:
        row([src, field, what, cand, impact, effort])
        if note:
            c = ws.cell(row=r - 1, column=7, value=note)
            ws.column_dimensions["G"].width = 46
            c.font = Font(italic=True, size=9)

    # -- Section C --------------------------------------------------------
    section("SECTION C — nflreadpy endpoints never loaded")
    header(["Endpoint", "What it serves", "Feature opportunity", "", "", ""])
    for ep, serves, opp in NFL_UNLOADED_ENDPOINTS:
        row([ep, serves, opp, "", ""])

    # -- Section D --------------------------------------------------------
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
                 "contract. Declare candidates via RFE_CANDIDATE_COLS / "
                 "NFL_RFE_ADDITION_MONEYLINE_LIST to trial new features.")

    # -- Section E --------------------------------------------------------
    section("SECTION E — Cataloged but NOT recommended (honesty section)")
    header(["Source", "Field(s)", "Why not recommended", "", "", ""])
    for src, field, why in NFL_LOW_VALUE_FIELDS:
        row([src, field, why, "", "", ""])

    # -- Per-feature non-null coverage (existing table) -------------------
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
    ws.append(["NFL Feature Selection Decision Workbook"])
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
            continue  # free-form summary page, styled at write time
        for row in sheet.iter_rows():
            for cell in row:
                if cell.row > 1:  # headers already styled by _style_table
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
    # Re-apply header styling (the wrap pass touches header rows too).
    for sheet in wb.worksheets:
        if sheet.title == "Summary":
            for cell in sheet[1]:
                pass
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
