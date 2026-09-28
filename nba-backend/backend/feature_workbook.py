"""NBA RFE decision workbook generator — MLB structural parity.

Artifact-only companion to feature_selection.py. Performs no training and
never changes the active production feature list. Every sheet renders from
the published artifacts (``nba_feature_selection_*.json`` when a trace
exists, plus the run's drift/coverage CSVs and the authored upstream map in
smoke_nba.py).

Sheets (matching the MLB workbook's structure, via the NHL/NFL ports):
  README | Dashboard | All Features (Master) | Production (n) | Candidates (n)
  RFE Run Detail | Per-Fold Detail | Redundancy | Coverage Gaps | Glossary
  | Feature Pool

The NBA RFE sweep is record-only today (no committed trials, no per-fold
metrics, no redundancy catalog), so those sheets render their honest
empty/absent state with the reason stated rather than being silently
omitted — the same honesty rule MLB's workbook applies to a run with no
committed steps.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

try:
    from backend import config
    from backend import smoke_nba as sources_map
except ImportError:
    import config
    import smoke_nba as sources_map

BACKEND = Path(__file__).resolve().parent
DELIVERY = BACKEND.parent / "data_delivery"
NAVY = "1F3864"
BLUE = PatternFill("solid", fgColor=NAVY)
WHITE = Font(color="FFFFFF", bold=True)

HEADER_ROW = ["Feature", "In Production", "Type", "Category", "Side",
              "Plain-English Description", "Upstream", "Coverage %",
              "Importance weight %"]

# --------------------------------------------------------------------------- #
# Taxonomy: category / type / side (NBA name grammar)
# --------------------------------------------------------------------------- #
CATEGORY_RULES: list[tuple[str, str]] = [
    (r"^pl_ts_|lineup_ts|lineup_", "Projected Lineups"),
    (r"^event_|possessions|shooting_fouls|live_tov|rim_|three_rate|"
     r"and_in|shot_distance|q4_points", "Play-by-Play"),
    (r"rest_days|back_to_back", "Schedule & Rest"),
    (r"elo|win_pct", "Team Results"),
    (r"ewm_net_points|ewm_off_rating|ewm_def_rating", "Team Scoring Margin"),
    (r"ewm_pace|ewm_efg|ewm_turnover|ewm_rebound|ewm_ast", "Team Four Factors"),
    (r"^is_home$|^is_playoffs$|game_type", "Game Context"),
    (r"nba_", "RFE Candidate Pool (trailing stats)"),
]


def parse_category(name: str) -> str:
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, name):
            return cat
    return "Other"


TYPE_RULES: list[tuple[str, str]] = [
    (r"_diff$", "Diff (home minus away)"),
    (r"_(home|away)$", "Raw level (per side)"),
    (r"^is_", "Flag"),
]


def parse_type(name: str) -> str:
    for pat, typ in TYPE_RULES:
        if re.search(pat, name):
            return typ
    return "Game-level value"


def parse_side(name: str) -> str:
    if name.endswith("_diff"):
        return "Home − Away (diff)"
    if name.endswith("_home"):
        return "Home"
    if name.endswith("_away"):
        return "Away"
    return "Game-level"


STAT_WORDS: dict[str, str] = {
    "elo": "Elo rating (point-in-time entering the game)",
    "win_pct": "trailing win percentage",
    "rest_days": "days since the team's previous game",
    "back_to_back": "second game of a back-to-back flag",
    "net_points": "net points per game",
    "off_rating": "offensive rating (points per 100 possessions)",
    "def_rating": "defensive rating (opponent points per 100 possessions)",
    "pace": "possessions per 48 minutes",
    "efg": "effective field-goal percentage",
    "turnover_margin": "turnover margin per game",
    "rebound_margin": "rebound margin per game",
    "ast_per_game": "assists per game",
    "three_rate": "share of field-goal attempts from three",
    "rim_rate": "share of field-goal attempts within 5 feet",
    "live_tov_rate": "share of turnovers that are live-ball",
    "and_in_rate": "and-one rate (foul with the shot made, per possession)",
    "shot_distance": "average shot distance in feet",
    "possessions": "estimated possessions (FGA + 0.44·FTA − OREB + TOV)",
    "shooting_fouls": "shooting fouls drawn per game",
    "q4_points": "fourth-quarter points per game",
    "pl_ts": "position-segmented shrunk true-shooting rating of the "
             "projected lineup",
}

WINDOW_WORDS: dict[str, str] = {
    "ewm": "exponentially weighted mean over the team's game history",
    "roll": f"trailing {getattr(config, 'PBP_ROLL_WINDOW', '?')}-game window",
    "l5": "last 5 games", "l10": "last 10 games", "l20": "last 20 games",
}


def parse_window(name: str) -> str:
    for key, text in WINDOW_WORDS.items():
        if re.search(rf"_{key}(_|$)", name):
            return text
    return ""


def auto_describe(name: str) -> str:
    stat = None
    for token in sorted(STAT_WORDS, key=len, reverse=True):
        if name.startswith(token + "_") or f"_{token}_" in name:
            stat = STAT_WORDS[token]
            break
    if stat is None:
        if name == "is_home":
            return "constant 1.0 — the home side of the pairing"
        if name == "is_playoffs":
            return "1.0 when the game is a playoff/postseason game"
        return "derived statistic"
    typ = parse_type(name)
    if typ == "Diff (home minus away)":
        return f"home minus away gap in the {stat}"
    if typ == "Raw level (per side)":
        side = "home" if name.endswith("_home") else "away"
        return f"{stat}, {side} side"
    return f"{stat} (game level)"


def _upstream(name: str) -> str:
    return str(sources_map.FEATURE_UPSTREAM.get(name, ""))


def _loaders():
    """Newest published artifacts (trace optional; drift/coverage expected)."""
    traces = sorted(DELIVERY.glob("nba_feature_selection_*.json"))
    trace = (json.loads(traces[-1].read_text(encoding="utf-8"))
             if traces else {})
    drifts = sorted(DELIVERY.glob("nba_run_engine_feature_drift_*.csv"))
    drift = (pd.read_csv(drifts[-1]) if drifts else pd.DataFrame())
    coverages = sorted(DELIVERY.glob("nba_run_engine_feature_coverage_*.csv"))
    coverage = (pd.read_csv(coverages[-1]) if coverages else pd.DataFrame())
    return trace, drift, coverage


def _weight_lookup(drift: pd.DataFrame) -> dict[str, float]:
    if not len(drift) or "weight_pct" not in drift.columns:
        return {}
    return dict(zip(drift["feature"].astype(str),
                    pd.to_numeric(drift["weight_pct"], errors="coerce")))


def _coverage_lookup(coverage: pd.DataFrame) -> dict[str, float]:
    if not len(coverage):
        return {}
    col = ("pct_nonnull" if "pct_nonnull" in coverage.columns
           else "coverage_pct" if "coverage_pct" in coverage.columns else None)
    if col is None:
        return {}
    sub = coverage if "feature" not in coverage.columns or "window" not in \
        coverage.columns else coverage.sort_values("window").drop_duplicates(
            "feature", keep="last")
    return dict(zip(sub["feature"].astype(str),
                    pd.to_numeric(sub[col], errors="coerce")))


def _feature_row(name: str, in_prod: bool, weights: dict[str, float],
                 coverage: dict[str, float]) -> list[Any]:
    cov = coverage.get(name)
    return [
        name,
        "Yes" if in_prod else "No",
        parse_type(name),
        parse_category(name),
        parse_side(name),
        auto_describe(name),
        _upstream(name),
        round(cov, 1) if cov is not None and cov == cov else "—",
        (round(float(weights[name]), 4)
         if name in weights and weights[name] == weights[name] else "—"),
    ]


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


def _sheet_readme(wb: Workbook, trace: dict, trace_name: str) -> None:
    ws = wb.create_sheet("README")
    ws.sheet_properties.tabColor = "808080"
    trials = trace.get("trials") or []
    rows = [
        ["NBA Feature Selection Decision Workbook"],
        [],
        ["Trace", trace_name or "(no nba_feature_selection_*.json artifact "
                              "yet — record-only sweeps have not written one)"],
        ["Feature set version", getattr(config, "FEATURE_SET_VERSION", "")],
        ["Run mode", "record-only" if not trials else "traced"],
        ["RFE commit rule",
         "a trial commits only when its paired log-loss gain clears "
         f"max(floor, {getattr(config, 'RFE_NOISE_SIGMA', 1.0)} x paired SE); "
         f"RFE_COMMIT_SE_MULTIPLE = "
         f"{getattr(config, 'RFE_COMMIT_SE_MULTIPLE', 1.0)}, "
         f"RFE_MAX_STEPS = {getattr(config, 'RFE_MAX_STEPS', 120)}"],
        ["Governance",
         "record-only sweep; the served contract is unchanged. Adoption is "
         "an explicit config change with a provenance comment."],
        [],
        ["How to use this workbook"],
        ["• All Features (Master) — every known feature with taxonomy, "
         "description, upstream, coverage and blend importance."],
        ["• Production (n) — the served contract, in serving order."],
        ["• Candidates (n) — the declared RFE pool not in production."],
        ["• Coverage Gaps — the API-parity gap catalog plus per-feature "
         "coverage."],
        ["• Glossary — the full description per feature."],
        ["• Feature Pool — the machine-readable pool (the legacy sheet)."],
    ]
    for row in rows:
        ws.append(row)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 95


def _sheet_dashboard(wb: Workbook, trace: dict, served: list[str],
                     candidates: list[str], known: list[str]) -> None:
    ws = wb.create_sheet("Dashboard")
    ws.sheet_properties.tabColor = "2E75B6"
    trials = trace.get("trials") or []
    committed = [t for t in trials if t.get("committed")]
    ws.append(["RFE Run Dashboard"])
    ws.append([])
    ws.append(["Metric", "Value"])
    dash = [
        ("Total known features", len(known)),
        ("  In production (serving)", len(served)),
        ("  Candidates (awaiting adoption)", len(candidates)),
        ("Trials scored", len(trials)),
        ("Committed", len(committed)),
        ("Features selected (record-only)",
         len(trace.get("selected_cols", [])) or "none (record-only run)"),
    ]
    for label, value in dash:
        ws.append([label, value])
    ws.append([])
    ws.append(["Committed changes this run"])
    ws.append(["Step", "Kind", "Feature", "Log-loss gain", "Committed"])
    for i, s in enumerate(committed, start=1):
        ws.append([i, s.get("kind", "trial"), s.get("feature", ""),
                   s.get("logloss_gain", "—"), s.get("committed", True)])
    if not committed:
        ws.append(["(none)", "record-only", "the sweep committed nothing — "
                   "served contract unchanged", "", ""])
    ws.append([])
    ws.append(["Top 10 features by blend importance weight (drift artifact)"])
    ws.append(["Feature", "Weight %"])
    for feature, weight in list(_last_weights.items())[:10]:
        ws.append([feature, weight])
    _style_table(ws, 2, [44, 24])


_last_weights: dict[str, float] = {}


def _sheet_master(wb: Workbook, sheet_name: str, tab_color: str,
                  names: list[str], weights: dict[str, float],
                  coverage: dict[str, float]) -> None:
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = tab_color
    ws.append(HEADER_ROW)
    for name in names:
        ws.append(_feature_row(name, True, weights, coverage))
    _style_table(ws, len(HEADER_ROW),
                 [30, 13, 24, 26, 20, 70, 55, 12, 17])


def _sheet_candidates(wb: Workbook, names: list[str],
                      weights: dict[str, float],
                      coverage: dict[str, float]) -> None:
    ws = wb.create_sheet("Candidates")
    ws.sheet_properties.tabColor = "C55A11"
    ws.append(HEADER_ROW + ["Role in run"])
    for name in names:
        ws.append(_feature_row(name, False, weights, coverage)
                  + ["declared candidate (offered, not trialed)"])
    _style_table(ws, len(HEADER_ROW) + 1,
                 [30, 13, 24, 26, 20, 70, 55, 12, 17, 30])


def _sheet_run_detail(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("RFE Run Detail")
    ws.sheet_properties.tabColor = "7030A0"
    headers = ["step", "kind", "feature", "n_features", "logloss",
               "logloss_gain", "paired_se", "commit_threshold", "auc", "ece",
               "committed", "error"]
    ws.append(headers)
    for i, rec in enumerate(trace.get("trials") or [], start=1):
        ws.append([i, rec.get("kind", "trial"), rec.get("feature", ""),
                   rec.get("n_features", ""), rec.get("logloss", ""),
                   rec.get("logloss_gain", ""), rec.get("paired_se", ""),
                   rec.get("commit_threshold", ""), rec.get("auc", ""),
                   rec.get("ece", ""), rec.get("committed", ""),
                   rec.get("error", "")])
    if not trace.get("trials"):
        ws.append(["(none)", "record-only", "the sweep scored no trials — "
                   "the trace records the pool and governance only", "", "",
                   "", "", "", "", "", "", ""])
    _style_table(ws, len(headers))


def _sheet_per_fold(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("Per-Fold Detail")
    ws.sheet_properties.tabColor = "9E5EB5"
    ws.append(["step", "feature", "fold_id", "validation_window", "n_games",
               "logloss"])
    rows = 0
    for i, rec in enumerate(trace.get("trials") or [], start=1):
        for fold in (rec.get("per_fold") or []):
            ws.append([i, rec.get("feature", ""), fold.get("fold_id"),
                       fold.get("val_window"), fold.get("n_games"),
                       fold.get("logloss")])
            rows += 1
    if not rows:
        ws.append(["(none)", "", "", "the NBA sweep is record-only — no "
                   "per-fold metrics exist", "", ""])
    _style_table(ws, 6)


def _sheet_redundancy(wb: Workbook, trace: dict) -> None:
    ws = wb.create_sheet("Redundancy")
    ws.sheet_properties.tabColor = "2F8F83"
    ws.append(["Feature A", "Feature B", "Pearson r", "Note"])
    pairs = trace.get("redundancy") or []
    if not pairs:
        ws.append(["(none)", "(none)", "",
                   "No |r| >= 0.9 redundancy catalog in the record-only "
                   "trace. The sweep has not yet measured pairwise "
                   "correlation over the scored frame."])
    for pr in pairs:
        ws.append([pr.get("a"), pr.get("b"), pr.get("r"),
                   "near-duplicate signal — consider one representative"])
    _style_table(ws, 4, [30, 30, 12, 60])


NBA_API_SOURCES = [
    ("ESPN scoreboard (site.api.espn.com)",
     "Per day: events with ids, dates, teams, venues, start times, scores, "
     "status (incl. postponement detail).",
     "The complete schedule population: ids, gameday, teams, venue, start "
     "times; scores for the decided filter; postponement detection for the "
     "slate.",
     "Odds and attendance (never read; the engine prices its own lines)."),
    ("stats.nba.com LeagueGameLog",
     "Per player-game per season: points, FGA/FGM/3PA/3PM/FTA/FTM, OREB/"
     "DREB, TOV, PF, AST, MIN, plus team and opponent rollups.",
     "Team features (the trailing-state ladder) and player lines (the "
     "position-segmented shrunk TS ratings); assists/rebounds cross-checks "
     "against play-by-play.",
     "Advanced box columns the ladder does not read (e.g. PlusMinus)."),
    ("stats.nba.com playbyplayv3",
     "Per game: the full action stream (period, clock, actionType, subType, "
     "shotDistance, shotValue, players).",
     "The event-only features (shot zones, live-ball turnovers, and-ones, "
     "possession estimates, Q4 scoring) and the designations cross-check; "
     "the PBP backfill keeps the rollup cache current.",
     "Per-action player attribution beyond the team rollup (second-order "
     "vs team trailing stats)."),
    ("Official NBA injury report (published designations archive)",
     "Per game per player: the six official designations at the latest "
     "pre-tipoff report.",
     "Point-in-time exclusion from the projected lineup pool for the "
     "pl_ts_* family (Out/Doubtful/Recovery removed from that game only).",
     "Nothing beyond the designation state."),
]

NBA_LOW_VALUE_FIELDS = [
    ("ESPN scoreboard", "odds / broadcast metadata",
     "Never read; the model prices its own lines and broadcast data has no "
     "predictive role pre-game."),
    ("LeagueGameLog", "per-player advanced splits (PlusMinus, etc.)",
     "Team trailing states and shrunk TS carry the signal; per-player "
     "plus-minus is noise-dominated at lineup granularity."),
    ("playbyplayv3", "per-action player attribution",
     "One action among ~490 per game; the team rollup is the served form."),
]


def _sheet_coverage(wb: Workbook, trace: dict, coverage: dict[str, float],
                    n_rows_hint: str) -> None:
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
    note_row("Sections A/E are the MLB-parity gap catalog (authored, grounded "
             "in ingestion.py and smoke_nba.py). The last block is the run's "
             "own per-feature coverage. Record-only: nothing here changes "
             "production without an explicit adoption run.")

    section("SECTION A — API payload inventory: serves vs extracts")
    header(["API endpoint", "What the payload serves",
            "What the pipeline extracts today", "What is left unused", "", ""])
    for a, b, c_, d in NBA_API_SOURCES:
        row([a, b, c_, d, "", ""])

    section("SECTION D — Declared candidates not trialed by this run")
    candidates = config.RFE_CANDIDATE_COLS
    if candidates:
        header(["Candidate", "Type", "Category", "Description", "Coverage %",
                ""])
        for name in candidates:
            cov = coverage.get(name)
            row([name, parse_type(name), parse_category(name),
                 auto_describe(name),
                 f"{cov:.1f}%" if cov is not None and cov == cov else "—",
                 ""])
    else:
        note_row("No declared candidates (config.RFE_CANDIDATE_COLS is "
                 "empty).")

    section("SECTION E — Cataloged but NOT recommended (honesty section)")
    header(["Source", "Field(s)", "Why not recommended", "", "", ""])
    for src, field, why in NBA_LOW_VALUE_FIELDS:
        row([src, field, why, "", "", ""])

    section("PER-FEATURE COVERAGE — non-null share in the run's current "
            f"window ({n_rows_hint})")
    if coverage:
        header(["Feature", "Type", "Side", "Coverage %", "", ""])
        rows = sorted(((cov if cov == cov else -1.0, name)
                       for name, cov in coverage.items()), key=lambda t: t[0])
        for cov, name in rows:
            row([name, parse_type(name), parse_side(name),
                 f"{cov:.1f}%" if cov and cov > 0 else ("0.0%" if cov == 0
                                                        else "—"), "", ""])
    else:
        note_row("No nba_run_engine_feature_coverage_*.csv artifact found — "
                 "run the pipeline to populate this table.")
    ws.freeze_panes = None


def _sheet_glossary(wb: Workbook, names: list[str]) -> None:
    ws = wb.create_sheet("Glossary")
    ws.sheet_properties.tabColor = "808080"
    ws.append(["Feature", "Plain-English Description", "Upstream",
               "Type", "Category", "Side"])
    for name in names:
        ws.append([name, auto_describe(name), _upstream(name),
                   parse_type(name), parse_category(name), parse_side(name)])
    _style_table(ws, 6, [30, 70, 55, 24, 26, 20])


def _sheet_feature_pool(wb: Workbook, served: list[str],
                        candidates: list[str], known: list[str]) -> None:
    ws = wb.create_sheet("Feature Pool")
    ws.append(["feature", "served", "candidate"])
    served_set, cand_set = set(served), set(candidates)
    for feature in known:
        ws.append([feature, feature in served_set, feature in cand_set])
    _style_table(ws, 3, [34, 12, 12])


def generate_workbook(trace_path: str | None = None,
                      out_path: str | None = None) -> str | None:
    """Build the decision workbook; returns the written path or None."""
    global _last_weights
    if trace_path:
        path = Path(trace_path)
        trace = json.loads(path.read_text(encoding="utf-8")) if path.exists() \
            else {}
        trace_name = path.name if path.exists() else ""
    else:
        traces = sorted(DELIVERY.glob("nba_feature_selection_*.json"))
        path = traces[-1] if traces else None
        trace = json.loads(path.read_text(encoding="utf-8")) if path else {}
        trace_name = path.name if path else ""
    _, drift, coverage_df = _loaders()
    weights = _weight_lookup(drift)
    coverage = _coverage_lookup(coverage_df)
    _last_weights = dict(sorted(
        ((k, v) for k, v in weights.items() if v == v),
        key=lambda kv: kv[1], reverse=True))

    served = config.active_moneyline_feature_cols()
    candidates = list(config.RFE_CANDIDATE_COLS)
    known = list(config.KNOWN_FEATURE_COLS)

    # The daily pipeline passes an explicit path; a standalone build takes
    # its run date from the newest markets artifact.
    if out_path:
        target = Path(out_path)
    else:
        dated = sorted(DELIVERY.glob("nba_run_engine_markets_*.csv"))
        date_c = dated[-1].stem.rsplit("_", 1)[-1] if dated else "standalone"
        target = DELIVERY / f"nba_feature_workbook_{date_c}.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["NBA Feature Selection Decision Workbook"])
    ws.append(["Trace", trace_name or "(none — record-only runs have not "
                                 "written one yet)"])
    ws.append(["Feature set version", getattr(config, "FEATURE_SET_VERSION",
                                              "")])
    ws.append(["Production width", len(served)])
    ws.append(["Candidate pool", len(candidates)])
    ws.append(["Known features", len(known)])
    ws.append([])
    ws.append(["Default governance: this workbook does not change "
               "production. Adoption is explicit."])

    _sheet_readme(wb, trace, trace_name)
    _sheet_dashboard(wb, trace, served, candidates, known)
    _sheet_master(wb, "All Features (Master)", "548235", known,
                  weights, coverage)
    _sheet_master(wb, f"Production ({len(served)})", "375623", served,
                  weights, coverage)
    _sheet_candidates(wb, candidates, weights, coverage)
    _sheet_run_detail(wb, trace)
    _sheet_per_fold(wb, trace)
    _sheet_redundancy(wb, trace)
    n_hint = (f"{len(coverage_df)} feature-window rows"
              if len(coverage_df) else "no coverage artifact")
    _sheet_coverage(wb, trace, coverage, n_hint)
    _sheet_glossary(wb, known)
    _sheet_feature_pool(wb, served, candidates, known)

    for sheet in wb.worksheets:
        if sheet.title == "Summary":
            continue
        for row in sheet.iter_rows():
            for cell in row:
                if cell.row > 1:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
        for cell in sheet[1]:
            cell.fill = BLUE
            cell.font = WHITE
            cell.alignment = Alignment(wrap_text=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    wb.save(target)
    return str(target)


def write_feature_workbook(out_dir, date_c: str,
                           trace: dict | None = None) -> str | None:
    """Daily-pipeline entry point (the pre-parity signature, kept).

    master_pipeline calls this right after run_rfe has written that run's
    ``nba_feature_selection_<date_c>.json``, so the trace file is
    addressable by date. Delegates to generate_workbook pointed at that
    trace and pins the artifact name to the run date — retention_policy's
    rfe_workbook family and config.FEATURE_WORKBOOK_XLSX both key on it.
    A workbook failure is reported and swallowed: the same never-blocks
    rule MLB and NHL wrap their call sites in, kept here so the call site
    stays bare.
    """
    try:
        trace_file = Path(out_dir) / f"nba_feature_selection_{date_c}.json"
        if trace is not None and not trace_file.exists():
            trace_file.write_text(
                json.dumps(trace, indent=1, default=str), encoding="utf-8")
        generated = generate_workbook(trace_path=str(trace_file))
        if generated is None:
            return None
        target = Path(out_dir) / f"nba_feature_workbook_{date_c}.xlsx"
        if Path(generated) != target:
            Path(generated).replace(target)
        return target.name
    except Exception as exc:  # noqa: BLE001 - report, never block the run
        print(f"[workbook] skipped ({type(exc).__name__}: {exc})")
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build the NBA RFE decision workbook")
    ap.add_argument("--trace", default=None,
                    help="explicit nba_feature_selection_*.json path")
    ap.add_argument("--out", default=None, help="output .xlsx path")
    args = ap.parse_args()
    print(generate_workbook(args.trace, args.out) or "no artifacts found")
