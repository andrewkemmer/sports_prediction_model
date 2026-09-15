"""MLB Feature Decision Workbook — production artifact generator.

Builds data_delivery/mlb_feature_workbook_<trace-date>.xlsx: the
decision-grade, human-readable companion to the machine-readable
mlb_feature_selection_<date>.json RFE trace. One row per known feature
(production vs candidate) with plain-English descriptions, importance,
member routing, per-test RFE impact, redundancy, and an API-anchored
coverage-gap analysis.

Inputs (all published artifacts, no model training):
  data_delivery/mlb_feature_selection_<date>.json   (RFE trace; v1/v2 schemas)
  data_delivery/feature_drift_<date>.csv            (importance weight_pct)
  data_delivery/feature_coverage_<date>.csv         (non-null coverage)
  feature_metadata.build_features_metadata()        (authored descriptions)
  config.py RFE_CANDIDATE_COLS groups               (candidate labels)

The daily pipeline regenerates this automatically after every RFE run
(Phase 4.5). It can also be built standalone:
  python mlb-backend/backend/feature_workbook.py [--trace <path>] [--out <path>]
The mlb_* prefix inherits the never-delete retention policy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

BACKEND = Path(__file__).resolve().parent            # mlb-backend/backend
ROOT = BACKEND.parent                                 # mlb-backend/
DATA_DELIVERY = ROOT / "data_delivery"                # artifact home (synced)

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

# --------------------------------------------------------------------------- #
# Styling constants
# --------------------------------------------------------------------------- #
NAVY = "1F3864"
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(bold=True, size=14, color=NAVY)
SUBTITLE_FONT = Font(size=10, color="595959")
GREEN_FILL = PatternFill("solid", fgColor="C6EFCE")   # proven commit
AMBER_FILL = PatternFill("solid", fgColor="FFEB9C")   # small measured benefit
GRAY_FILL = PatternFill("solid", fgColor="EDEDED")    # untested / n.a.
RED_FILL = PatternFill("solid", fgColor="F8CBAD")     # hurt / dead domain
BLUE_FILL = PatternFill("solid", fgColor="DDEBF7")    # expansion opportunity
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
TOP = Alignment(vertical="top")


def style_header(ws, ncols: int, row: int = 1) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")


def set_widths(ws, widths: list[int]) -> None:
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def fill_cell(cell, fill: Optional[PatternFill]) -> None:
    if fill is not None:
        cell.fill = fill


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def load_trace(path: Path) -> dict[str, Any]:
    t = load_json(path)
    t["_schema"] = "v2" if ("invocation" in t or "verdicts" in t) else "v1"
    return t


def _newest(pattern: str) -> Optional[Path]:
    hits = sorted(DATA_DELIVERY.glob(pattern))
    return hits[-1] if hits else None


def find_latest_trace() -> Optional[Path]:
    """Newest RFE trace; a same-day full trace outranks a targeted one."""
    hits = sorted(DATA_DELIVERY.glob("mlb_feature_selection_*.json"))
    full = [p for p in hits if "_targeted" not in p.name]
    return (full or hits)[-1] if hits else None


def load_drift() -> Optional[pd.DataFrame]:
    """Newest feature_drift_<date>.csv (production importance weight_pct)."""
    p = _newest("feature_drift_*.csv")
    return pd.read_csv(p) if p else None


def load_coverage() -> Optional[pd.DataFrame]:
    """Newest feature_coverage_<date>.csv (latest-slate non-null coverage)."""
    p = _newest("feature_coverage_*.csv")
    return pd.read_csv(p) if p else None


def load_metadata() -> dict[str, Any]:
    """Authored plain-English metadata for the production features
    (feature_metadata.build_features_metadata — the same source the
    model_monitor artifact embeds)."""
    from feature_metadata import build_features_metadata
    meta, _warns = build_features_metadata()
    return {"features": meta}


def load_pool():
    """Production universe + candidate pool straight from production config."""
    from training import KNOWN_FEATURE_COLS, MONEYLINE_FEATURE_COLS
    from config import RFE_CANDIDATE_COLS

    return list(MONEYLINE_FEATURE_COLS), list(RFE_CANDIDATE_COLS), list(KNOWN_FEATURE_COLS)


def load_candidate_groups() -> dict[str, str]:
    """Parse config.py's RFE_CANDIDATE_COLS block: commented group headers
    -> {col: group label}. Line-based scan between the block's [ and ]."""
    text = (BACKEND / "config.py").read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    groups: dict[str, str] = {}
    start_i = next((i for i, ln in enumerate(lines)
                    if "RFE_CANDIDATE_COLS" in ln and "[" in ln), None)
    if start_i is None:
        return groups
    current: Optional[str] = None
    for ln in lines[start_i + 1:]:
        if "]" in ln:  # end of the list literal
            break
        m = re.match(r"\s*#\s*-+\s*(.+?)\s*-+\s*$", ln)
        if m:
            current = m.group(1).strip()
            continue
        m = re.match(r'\s*"([A-Za-z0-9_]+)"\s*,?\s*$', ln)
        if m and current is not None:
            groups[m.group(1)] = current
    return groups


# --------------------------------------------------------------------------- #
# Taxonomy: category / type / side
# --------------------------------------------------------------------------- #
CATEGORY_RULES: list[tuple[str, str]] = [
    (r"umpire", "Umpires"),
    (r"defense|fielder|outs_made|out_conversion", "Defense"),
    (r"time_zone", "Schedule & Travel"),
    (r"exp2_|_cat_|_fb_vs_|opp_lefty|lefty_share|platoon|vs_l|vs_r", "Pitch Matchups (arsenal)"),
    (r"ace_|pitcher_regression", "Starting Pitching"),
    (r"dome|roof|park|venue|stadium", "Stadium / Park"),
    (r"weather|wind|air_density|temp|humidity|precip", "Weather"),
    (r"travel|rest_days", "Schedule & Travel"),
    (r"^sp_|starter", "Starting Pitching"),
    (r"bullpen|closer|meltdown", "Bullpen"),
    (r"lineup", "Lineup (posted)"),
    (r"woba|barrel|hardhit|exitvelo|iso|k_rate|bb_rate|slug", "Team Batting"),
    (r"elo|win_pct|run_margin|is_home|home_|away_", "Team Results"),
    (r"^league", "League Context"),
]

TYPE_RULES: list[tuple[str, str]] = [
    (r"_delta(_|$)", "Form delta (recent vs season)"),
    (r"(5g|3g|10g|15g|30g).*_diff$", "Diff of rolling average"),
    (r"_diff$", "Diff (season level)"),
    (r"_(home|away)$", "Raw level (per side)"),
    (r"^(is_home|dome_is_neutral.*|closer_availability_diff)$", "Flag / availability"),
    (r"indicator|multiplier|risk|factor|boost|advantage|efficiency|depth", "Derived index"),
]

STAT_WORDS: dict[str, str] = {
    "era": "earned-run average (runs allowed per 9 innings)",
    "k9": "strikeouts per 9 innings",
    "bb9": "walks per 9 innings",
    "whip": "WHIP (walks + hits allowed per inning)",
    "xwoba": "expected wOBA (contact-quality-based batting value)",
    "fbvelo": "fastball velocity (mph)",
    "fbpct": "fastball usage share",
    "whiff": "whiff rate (swing-and-miss frequency)",
    "woba": "wOBA (overall batting value)",
    "iso": "isolated power (extra-base ability)",
    "k_rate": "strikeout rate",
    "bb_rate": "walk rate",
    "barrel": "barrel rate (barreled balls per plate appearance)",
    "hardhit": "hard-hit ball rate",
    "exitvelo": "average exit velocity (mph)",
    "pitches": "pitches thrown (workload)",
    "slug": "slugging percentage",
    "k_pct": "strikeout rate",
    "time_zones": "time zones crossed on recent road trips",
    "lefty_share": "share of opposing pitchers who throw left-handed",
    "centered_k": "strikeout rate (park/league centered)",
}

WINDOW_WORDS: dict[str, str] = {
    "5g": "last 5 games/starts", "3g": "last 3 games/starts",
    "10g": "last 10 games/starts", "15g": "last 15 games",
    "30g": "last 30 days",
}


def parse_category(name: str) -> str:
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, name):
            return cat
    return "Other"


def parse_type(name: str) -> str:
    for pat, typ in TYPE_RULES:
        if re.search(pat, name):
            return typ
    return "Other"


def parse_side(name: str) -> str:
    if name.endswith("_diff"):
        return "Home − Away (diff)"
    if name.endswith("_home"):
        return "Home"
    if name.endswith("_away"):
        return "Away"
    return "Game-level"


def auto_describe(name: str, group: str) -> str:
    """Plain-English description composed from the candidate's name parts."""
    stat = None
    for tok in sorted(STAT_WORDS, key=len, reverse=True):
        if name.startswith(tok + "_") or f"_{tok}_" in name or name.startswith(tok):
            stat = STAT_WORDS[tok]
            break
    base = stat or "derived statistic"
    win = next((v for k, v in WINDOW_WORDS.items() if re.search(rf"_{k}(_|$)", name)), None)
    win_txt = f" over the {win}" if win else ""
    if "_delta_" in name or name.endswith("_delta"):
        shape = "recent-form change (recent window vs season baseline)"
    elif name.endswith("_diff"):
        shape = "home-minus-away gap"
    elif name.endswith("_home"):
        shape = "home-side value"
    elif name.endswith("_away"):
        shape = "away-side value"
    else:
        shape = "game-level value"
    group_txt = ""
    if "raw per-side" in group:
        group_txt = " Raw per-side level (currently only diffs are served — the level adds the absolute scale)."
    elif "arsenal" in group:
        group_txt = " Pitch-type-specific matchup stat."
    elif "league-context" in group:
        group_txt = " League-wide context aggregate."
    elif "environment" in group:
        group_txt = " Environment/schedule context."
    return f"{base.capitalize()}{win_txt} — {shape}.{group_txt}"


# --------------------------------------------------------------------------- #
# RFE impact + verdicts
# --------------------------------------------------------------------------- #
def _step_feature(s: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """(kind, feature) for one trace step, tolerating BOTH trace schemas.

    v2 steps (current engine): {"action": "remove"|"add", "feature": name}
    v1 steps (legacy 09-12 trace):        {"removed"|"added": name}

    Schema drift here once blanked the whole workbook silently (every step
    parsed as featureless, so no row ever matched an RFE test) — the
    generate_workbook loud check now catches that; this adapter is the
    single place that knows the schema.
    """
    if s.get("feature") and s.get("action") in ("remove", "add"):
        return ("removal" if s["action"] == "remove" else "addition"), s["feature"]
    if s.get("removed"):
        return "removal", s["removed"]
    if s.get("added"):
        return "addition", s["added"]
    return None, None


def build_rfe_index(trace: dict[str, Any]) -> dict[str, dict]:
    """{feature: impact record} from a v1 or v2 trace."""
    per_feature: dict[str, dict] = {}
    ece_guard = (trace.get("guards", {}) or {}).get("ece_rise_max", 0.010)
    for s in trace.get("steps", []):
        kind, f = _step_feature(s)
        if not f:
            continue
        gain = s.get("logloss_gain")
        thr = s.get("commit_threshold")
        if s.get("committed"):
            verdict = ("REMOVAL COMMITTED — feature measurably redundant; removing it "
                       "improved accuracy beyond the noise bar" if kind == "removal"
                       else "ADDITION COMMITTED — feature measurably improves accuracy "
                            "beyond the noise bar")
        elif kind == "removal":
            if gain is not None and thr is not None and 0 < gain < thr:
                verdict = ("Kept — removal gave a small gain, but below the noise bar "
                           f"({gain:.4f} < {thr:.4f}); not proven redundant")
            else:
                verdict = "Kept — removing it did not help (gain ≤ 0)"
            if (s.get("auc_drop") or 0) > 0:
                verdict += "; AUC slightly worse without it"
            elif (s.get("auc_drop") or 0) < -0.002:
                verdict += "; AUC slightly better without it (not enough to commit)"
        else:
            if gain is not None and thr is not None and gain < thr:
                verdict = ("Not added — improvement below the noise bar "
                           f"({gain:.4f} < {thr:.4f})")
            else:
                verdict = "Not added — no measurable improvement (or a guard blocked it)"
            if (s.get("ece_rise") or 0) > ece_guard:
                verdict += "; calibration (ECE) guard blocked it"
        per_feature[f] = {
            "kind": kind, "step": s.get("step"), "verdict": verdict,
            "logloss_gain": gain, "commit_threshold": thr,
            "auc_drop": s.get("auc_drop"), "ece_rise": s.get("ece_rise"),
            "committed": bool(s.get("committed")), "metrics": s.get("metrics", {}),
            "per_fold": s.get("per_fold"), "n_features": s.get("n_features"),
        }
    return per_feature


def recommendation(feat: dict[str, Any]) -> str:
    imp, tested = feat["importance"], feat["rfe"]
    if feat["in_production"]:
        if tested and tested.get("committed"):
            return "REMOVE from serving width at next adopt (proven redundant)"
        if tested and "small gain" in tested["verdict"]:
            return "Re-trial with a forced removal list to re-test at current data"
        if imp is not None and imp < 1.0:
            return "Low importance — consider forcing a removal trial"
        return "Keep — earning its place (or untested with meaningful importance)"
    if tested and tested.get("committed"):
        return "IN serving width — included at next retrain"
    if tested:
        return "Trialed — did not clear the bar; revisit after more data"
    return f"Expansion opportunity — force-test via MLB_RFE_ADDITION_MONEYLINE_LIST={feat['name']}"


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
MASTER_COLS = [
    "Feature", "In Production", "Type", "Category", "Side",
    "Plain-English Description", "Window", "Units", "Direction",
    "Importance (0–100)", "Model Routing", "RFE Status (this run)",
    "RFE Verdict (plain English)", "Logloss Change", "AUC Change",
    "ECE Change", "Redundant With", "Recommendation",
]

ALL_MEMBERS = {"xgboost", "lightgbm", "randomforest", "mlp", "logistic"}


def routing_label(in_prod: bool, members) -> str:
    if not in_prod:
        return "— (candidate)"
    if members is None:
        return "All models"
    if isinstance(members, dict):
        return ("All models + logistic" if members.get("logistic")
                else "Trees + MLP only (logistic excluded)")
    ms = set(members)
    if ms >= ALL_MEMBERS:
        return "All models (xgb, lgbm, rf, mlp, logistic)"
    if "logistic" in ms:
        rest = ", ".join(sorted(ms - {"logistic"}))
        return f"Logistic + {rest}"
    return " + ".join(sorted(ms))


def build_rows(universe, candidates, metadata, drift, coverage, groups, trace,
               per_feature) -> list[dict[str, Any]]:
    drift_w: dict[str, Any] = {}
    if drift is not None and len(drift):
        drift_w = dict(zip(drift["feature"],
                           drift.get("weight_pct", pd.Series(dtype=float))))
    cov_map: dict[str, Any] = {}
    if coverage is not None and len(coverage):
        cov_map = {r["feature"]: r for _, r in coverage.iterrows()
                   if r.get("window") == "current"}
    v2_pairs: dict[str, list[str]] = {}
    for pair in (trace.get("pool", {}) or {}).get("redundancy_pairs", []) or []:
        if isinstance(pair, dict):
            a, b, r = pair.get("a"), pair.get("b"), pair.get("r")
        else:
            a = pair[0] if len(pair) > 0 else None
            b = pair[1] if len(pair) > 1 else None
            r = pair[2] if len(pair) > 2 else None
        if a and b:
            v2_pairs.setdefault(a, []).append(f"{b} (r={r})")
            v2_pairs.setdefault(b, []).append(f"{a} (r={r})")
    verdicts_map = trace.get("verdicts") or {}
    prod_set = set(universe)

    rows = []
    for name in universe + [c for c in candidates if c not in prod_set]:
        in_prod = name in prod_set
        meta = (metadata.get("features", {}).get(name)
                or metadata.get("categorical_context", {}).get(name))
        group = groups.get(name, "")
        if in_prod and meta:
            desc = meta.get("definition") or meta.get("summary") or name
            desc_flag = ""
        else:
            desc = meta.get("definition") if meta else auto_describe(name, group)
            desc_flag = "" if meta else " (auto-described)"
        imp = drift_w.get(name)
        if imp is None:
            prior = (trace.get("pool", {}) or {}).get("importance_prior_top")
            if isinstance(prior, dict):
                # production v2 form: {feature: importance}
                imp = prior.get(name)
            elif isinstance(prior, list):
                imp = next((p.get("importance") for p in prior
                            if isinstance(p, dict) and p.get("feature") == name),
                           None)
        cov_row = cov_map.get(name)
        tested = per_feature.get(name)
        v = verdicts_map.get(name)
        if isinstance(v, dict):
            vv = v.get("verdict")
            if tested:  # step-based sentence is authoritative this run
                verdict_txt = tested["verdict"]
            elif vv == "committed":
                verdict_txt = "Committed in a prior run (carried verdict)"
            elif vv == "rejected":
                verdict_txt = "Previously tested — rejected (carried verdict from prior trace)"
            else:
                verdict_txt = str(vv) if vv else None
        elif isinstance(v, str) and v:
            verdict_txt = v
        else:
            verdict_txt = tested["verdict"] if tested else None
        rows.append({
            "name": name, "in_production": in_prod, "group": group,
            "importance": imp,
            "pct_nonnull": float(cov_row["pct_nonnull"]) if cov_row is not None else None,
            "rfe": tested, "verdict_txt": verdict_txt,
            "description": (desc or "") + desc_flag,
            "window": meta.get("window") if meta else ("—" if not in_prod else "?"),
            "units": meta.get("units") if meta else ("—" if not in_prod else "?"),
            "direction": meta.get("direction") if meta else ("—" if not in_prod else "?"),
            "type": parse_type(name), "category": parse_category(name),
            "side": parse_side(name),
            "redundant": ", ".join(v2_pairs.get(name, [])) or
                         (tested or {}).get("redundant_with") or None,
            "routing": routing_label(in_prod, meta.get("members") if meta else None),
        })
    return rows


def row_values(f: dict[str, Any]) -> list[Any]:
    t = f["rfe"]
    return [
        f["name"],
        "Yes" if f["in_production"] else "No",
        f["type"], f["category"], f["side"],
        f["description"],
        f["window"] or "—", f["units"] or "—", f["direction"] or "—",
        round(f["importance"], 2) if f["importance"] is not None else "—",
        f["routing"],
        ("Tested (step %s)" % t["step"]) if t else "Not tested this run",
        f["verdict_txt"] or ("Not reached by this run's step budget" if not t else ""),
        f"{t['logloss_gain']:+.4f}" if t and t["logloss_gain"] is not None else "—",
        f"{t['auc_drop']:+.4f}" if t and t["auc_drop"] is not None else "—",
        f"{t['ece_rise']:+.4f}" if t and t["ece_rise"] is not None else "—",
        f["redundant"] or "—",
        recommendation(f),
    ]


def verdict_fill(f: dict[str, Any]) -> Optional[PatternFill]:
    t = f["rfe"]
    if t and t.get("committed"):
        return GREEN_FILL
    if t and t["kind"] == "removal" and (t.get("logloss_gain") or 0) > 0:
        return AMBER_FILL
    if t and t["kind"] == "addition" and (t.get("logloss_gain") or 0) > 0:
        return BLUE_FILL
    if not t:
        return GRAY_FILL
    return None


def write_table(ws, cols: list[str], rows: list[list[Any]],
                widths: list[int], fills: Optional[list[Optional[PatternFill]]] = None,
                wrap_cols: tuple[int, ...] = ()) -> None:
    ws.append(cols)
    style_header(ws, len(cols))
    for r_i, row in enumerate(rows):
        ws.append(row)
        for c_i in range(1, len(cols) + 1):
            cell = ws.cell(row=ws.max_row, column=c_i)
            cell.border = BORDER
            cell.alignment = WRAP if (c_i - 1) in wrap_cols else TOP
            if fills:
                fill_cell(cell, fills[r_i])
    set_widths(ws, widths)
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{max(ws.max_row, 2)}"


# --------------------------------------------------------------------------- #
# Coverage-gap catalog (Tier 1) — authored from the codebase's raw sources
# --------------------------------------------------------------------------- #
GAP_CATALOG = [
    ("Umpires", "Umpire strike-zone tendency — home-plate ump's called-strike rate vs league average (diff)",
     "umpires.py fetch_season_umpires() + umpire_stats.csv (per-umpire K/BB rates)", "Low"),
    ("Umpires", "Umpire run-environment index — avg total runs in games worked (diff)",
     "umpires.py season stats (run support per umpire)", "Low"),
    ("Defense", "Team out-conversion rate — share of balls in play turned into outs (diff)",
     "build_pbp_defense.py fielder_* columns in pbp cache", "Medium"),
    ("Defense", "Catcher throwing / passed-ball rate (diff)",
     "fielder_2 columns in pbp cache", "Medium"),
    ("Defense", "Defensive efficiency by zone — range proxy from batted-ball data (diff)",
     "pbp cache (hit_distance_sc, launch_speed, fielder position)", "High"),
    ("Lineup (posted)", "Posted-lineup wOBA surprise — actual posted lineup vs probable expectation (diff)",
     "lineups.parquet (backfilled via backfill_lineups.py)", "Low"),
    ("Lineup (posted)", "Batting-order depth gap — wOBA drop from slot 4 onward (diff)",
     "lineups.parquet", "Low"),
    ("Schedule & Travel", "Travel distance in miles — real trip length between venues (diff)",
     "schedule + venue coordinates (frames.py game log)", "Medium"),
    ("Schedule & Travel", "Games-in-last-7-days workload (diff)",
     "already-computed schedule data; no new ingest", "Low"),
    ("Weather", "Temperature / humidity interaction with air density (diff)",
     "weather.py fetch_game_weather fields", "Low"),
    ("Weather", "Wind direction relative to park orientation (component along LF–CF–RF axes)",
     "weather.py wind_dir + venue coordinates", "Medium"),
    ("Stadium / Park", "Elevation (Coors-style effect) beyond existing park factor",
     "static venue table", "Low"),
    ("Starting Pitching", "SP pitch-mix change — arsenal share shift vs last month (diff)",
     "pbp cache pitch_type distributions", "Medium"),
    ("Bullpen", "Bullpen fatigue — pitches per reliever over last 3 days (diff)",
     "pbp cache + game logs", "Medium"),
]

# --------------------------------------------------------------------------- #
# Sheet builders
# --------------------------------------------------------------------------- #
def sheet_readme(wb, trace) -> None:
    ws = wb.create_sheet("README")
    ws.sheet_properties.tabColor = "808080"
    schema = "v2" if trace.get("_schema") == "v2" else "v1"
    lines = [
        ("MLB Feature Decision Workbook", TITLE_FONT),
        (f"Generated from RFE trace dated {trace.get('date')} (engine schema: {schema})",
         SUBTITLE_FONT),
        ("", None),
        ("How to use this workbook", Font(bold=True, size=12, color=NAVY)),
        ("1. Dashboard — headline numbers and where features stand overall.", None),
        ("2. All Features (Master) — every feature, one row: what it is, how much the model uses it, "
         "and what the RFE measured. Green = proven impactful; Amber = small measured benefit; "
         "Blue = addition that helped; Gray = not yet tested.", None),
        ("3. Production — the features currently feeding the live moneyline model.", None),
        ("4. Candidates — engineered features waiting for a trial; force-test any of them via "
         "MLB_RFE_ADDITION_MONEYLINE_LIST.", None),
        ("5. RFE Run Detail — every test the engine ran, with its exact metrics and a plain-English verdict.", None),
        ("6. Per-Fold Detail — per-fold logloss behind each step (v2 traces).", None),
        ("7. Redundancy — features that measure nearly the same thing; the model splits credit between them.", None),
        ("8. Coverage Gaps — the API-by-API truth: Section A shows what each source's payload "
         "serves vs what the pipeline extracts; Section B lists specific unused payload fields "
         "with concrete feature recipes (red = categories with zero features today, blue = "
         "cheapest high-value trials); Sections C–E cover deliberately-skipped fields, "
         "computed-but-untrialed columns, and dead categories.", None),
        ("9. Glossary — every metric term in one sentence.", None),
        ("", None),
        ("How to act", Font(bold=True, size=12, color=NAVY)),
        ("Remove: features whose verdict says 'REMOVAL COMMITTED' or low importance + small gains "
         "→ force-test via MLB_RFE_REMOVAL_MONEYLINE_LIST, then --adopt.", None),
        ("Expand: Coverage Gaps Section B names unused API fields and the feature each could become; force-test any engineered candidate via MLB_RFE_ADDITION_MONEYLINE_LIST.", None),
        ("Decide: adoption is always explicit — python mlb-backend/backend/feature_selection.py "
         "--date <date> --adopt (gated by alt-geometry confirmation + slate coverage floor).", None),
        ("", None),
        ("Honesty notes", Font(bold=True, size=12, color=NAVY)),
        ("• Importance is how much the model used a feature while fitting — not proof it predicts well. "
         "Verdicts come only from measured walk-forward logloss vs the noise bar.", None),
        ("• 'Not tested this run' means the 40-step budget did not reach the feature — not that it lacks value.", None),
        ("• Candidate descriptions marked '(auto-described)' are composed from the feature's name; "
         "production descriptions are hand-authored.", None),
    ]
    for txt, fnt in lines:
        ws.append([txt])
        if fnt:
            ws.cell(row=ws.max_row, column=1).font = fnt
    ws.column_dimensions["A"].width = 130
    for r in range(1, ws.max_row + 1):
        ws.cell(row=r, column=1).alignment = WRAP


def sheet_dashboard(wb, rows, trace) -> None:
    ws = wb.create_sheet("Dashboard")
    ws.sheet_properties.tabColor = "2E75B6"
    n_prod = sum(1 for f in rows if f["in_production"])
    n_cand = len(rows) - n_prod
    tested = [f for f in rows if f["rfe"]]
    commits = [f for f in tested if f["rfe"]["committed"]]
    steps = trace.get("steps", []) or []
    kinds = [_step_feature(s)[0] for s in steps]
    removals = [s for s, k in zip(steps, kinds) if k == "removal"]
    additions = [s for s, k in zip(steps, kinds) if k == "addition"]
    adopted = trace.get("adopted")
    bm = trace.get("baseline_metrics", {})
    kpis = [
        ("Total known features (pool)", len(rows)),
        ("  In production (serving the moneyline model)", n_prod),
        ("  Candidates (awaiting trial)", n_cand),
        ("RFE tests this run", len(trace.get("steps", []))),
        ("  Removal tests", len(removals)),
        ("  Addition tests", len(additions)),
        ("Proven commits", len(commits)),
        ("Serving width", f"{trace.get('n_selected')} (adopted subset)" if adopted
         else f"{n_prod} (full universe — no subset adopted)"),
        ("Baseline logloss (lower is better)", round(bm.get("logloss", float("nan")), 4)),
        ("Baseline AUC (higher is better)", round(bm.get("auc", float("nan")), 4)),
        ("Baseline ECE (calibration error; lower is better)", round(bm.get("ece", float("nan")), 4)),
        ("Noise bar per test (2 × paired SE of the logloss change)"
         if trace.get("bar_basis") == "paired_diff_2sigma"
         else "Noise bar per test (2 × baseline SE — legacy bar)",
         bm.get("commit_threshold") or round(max(0.002, 2 * (bm.get("logloss_se") or 0)), 4)),
        ("Folds used (walk-forward windows)", bm.get("folds_used")),
    ]
    ws.append(["Metric", "Value"])
    style_header(ws, 2)
    for k, v in kpis:
        ws.append([k, v])
    set_widths(ws, [56, 40])
    ws.append([""])

    ws.append(["Verdict bucket", "Count"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, color=NAVY)
    buckets = [
        ("Proven helpful (addition commits)",
         sum(1 for f in tested if f["rfe"]["committed"] and f["rfe"]["kind"] == "addition")),
        ("Proven redundant (removal commits)",
         sum(1 for f in tested if f["rfe"]["committed"] and f["rfe"]["kind"] == "removal")),
        ("Small measured benefit, below noise bar",
         sum(1 for f in tested if not f["rfe"]["committed"]
             and (f["rfe"].get("logloss_gain") or 0) > 0)),
        ("No measurable benefit (or hurt)",
         sum(1 for f in tested if not f["rfe"]["committed"]
             and (f["rfe"].get("logloss_gain") or 0) <= 0)),
        ("Untested this run", sum(1 for f in rows if not f["rfe"])),
    ]
    for k, v in buckets:
        ws.append([k, v])
    ws.append([""])

    ws.append(["Category", "In production", "Candidates", "Total in pool", "Tested this run"])
    hdr_row = ws.max_row
    style_header(ws, 5, row=hdr_row)
    for cat in sorted({f["category"] for f in rows}):
        sub = [f for f in rows if f["category"] == cat]
        ws.append([
            cat,
            sum(1 for f in sub if f["in_production"]),
            sum(1 for f in sub if not f["in_production"]),
            len(sub),
            sum(1 for f in sub if f["rfe"]),
        ])


def sheet_master(wb, rows, sheet_name: str, tab_color: str,
                 subset=None) -> None:
    ws = wb.create_sheet(sheet_name)
    ws.sheet_properties.tabColor = tab_color
    use = subset if subset is not None else rows
    use = sorted(use, key=lambda f: (f["category"], f["type"],
                                     -(f["importance"] or 0)))
    vals = [row_values(f) for f in use]
    fills = [verdict_fill(f) for f in use]
    write_table(ws, MASTER_COLS, vals,
                widths=[34, 11, 24, 22, 15, 62, 14, 12, 24, 12, 20, 16, 60, 11, 11, 11, 30, 52],
                fills=fills,
                wrap_cols=(5, 11, 15, 17))
    ws.freeze_panes = "B2"


def sheet_run_detail(wb, trace) -> None:
    ws = wb.create_sheet("RFE Run Detail")
    ws.sheet_properties.tabColor = "7030A0"
    cols = ["Step", "Action", "Feature", "Features In Test", "Committed?",
            "Logloss", "Logloss Change", "Commit Threshold (noise bar)",
            "AUC Change", "ECE Change", "AUC", "Brier", "ECE", "Logloss SE",
            "Folds Used", "Plain-English Verdict"]
    vals = []
    for s in trace.get("steps", []):
        kind, feat = _step_feature(s)
        action = {"removal": "Remove", "addition": "Add"}.get(kind, "—")
        m = s.get("metrics", {})
        gain, thr = s.get("logloss_gain"), s.get("commit_threshold")
        if s.get("committed"):
            verdict = f"Committed — gain {gain:+.4f} clears the {thr:.4f} noise bar"
        elif action == "Remove":
            verdict = (f"Not committed — gain {gain:+.4f} is below the {thr:.4f} noise bar"
                       if (gain or 0) > 0 else
                       f"Not committed — removal did not improve accuracy ({(gain or 0):+.4f})")
        else:
            verdict = (f"Not added — improvement {gain:+.4f} below the {thr:.4f} noise bar"
                       if gain is not None else "Not added — guard blocked it")
        vals.append([
            s.get("step"), action, feat, s.get("n_features"),
            "YES" if s.get("committed") else "no",
            round(s.get("logloss", float("nan")), 4),
            f"{gain:+.4f}" if gain is not None else "—",
            round(thr, 4) if thr is not None else "—",
            f"{s.get('auc_drop'):+.4f}" if s.get("auc_drop") is not None else "—",
            f"{s.get('ece_rise'):+.4f}" if s.get("ece_rise") is not None else "—",
            round(m.get("auc", float("nan")), 4), round(m.get("brier", float("nan")), 4),
            round(m.get("ece", float("nan")), 4), round(m.get("logloss_se", float("nan")), 6),
            m.get("folds_used"), verdict,
        ])
    fills = [GREEN_FILL if v[4] == "YES" else None for v in vals]
    write_table(ws, cols, vals,
                widths=[6, 8, 34, 9, 10, 9, 11, 13, 9, 9, 9, 9, 9, 11, 8, 64],
                fills=fills, wrap_cols=(15,))


def sheet_per_fold(wb, trace) -> None:
    ws = wb.create_sheet("Per-Fold Detail")
    ws.sheet_properties.tabColor = "9E5EB5"
    if trace.get("_schema") == "v1":
        ws.append(["Per-fold logloss recording starts with v2 traces."])
        ws.append(["This trace predates that — every step shows only pooled metrics "
                   "(see RFE Run Detail)."])
        set_widths(ws, [110])
        return
    cols = ["Step", "Feature", "Fold", "Fold Window", "N Games", "Logloss"]
    vals = []
    for s in trace.get("steps", []):
        _, feat = _step_feature(s)
        if not feat:
            continue
        for i, pf in enumerate(s.get("per_fold") or [], start=1):
            vals.append([s.get("step"), feat, i,
                         pf.get("window", pf.get("fold", i)),
                         pf.get("n_games"), round(pf.get("logloss", float("nan")), 4)])
    write_table(ws, cols, vals, widths=[6, 34, 6, 24, 8, 9])


def sheet_redundancy(wb, rows, trace) -> None:
    ws = wb.create_sheet("Redundancy")
    ws.sheet_properties.tabColor = "2F8F83"
    cols = ["Feature A", "Feature B", "Correlation (|r|)",
            "What this means", "RFE verdicts"]
    idx = {f["name"]: f for f in rows}
    pairs = (trace.get("pool", {}) or {}).get("redundancy_pairs") or []
    vals = []
    if pairs:
        for p in pairs:
            if isinstance(p, dict):          # v2 dict form
                pa, pb, r = p.get("a"), p.get("b"), p.get("r", 0)
            else:                            # [a, b, r] tuple form
                pa, pb, r = p[0], p[1], (p[2] if len(p) > 2 else 0)
            a, b = idx.get(pa), idx.get(pb)
            verdicts = "; ".join(filter(None, [
                (a or {}).get("verdict_txt"), (b or {}).get("verdict_txt")]))
            vals.append([pa, pb, round(r, 3),
                         "These two measure nearly the same thing — the model splits credit "
                         "between them. Trial them together; a verdict on one informs the other.",
                         verdicts or "—"])
    else:
        vals = [
            ["elo_diff", "win_pct_diff", "≥ 0.8",
             "Team-strength pair: Elo and win% track each other closely — the model splits "
             "credit between them. Neither removal cleared the noise bar this run.",
             "Both tested (removals): not proven redundant"],
            ["team_exitvelo_diff", "team_hardhit_diff", "≥ 0.8",
             "Contact-quality pair: exit velocity and hard-hit rate are two views of the same "
             "underlying quality-of-contact signal.",
             "Both tested: not proven redundant"],
            ["(v2 traces)", "(record automatically)", "—",
             "v2 runs record every |r| ≥ 0.9 pair in the trace; this sheet fills itself in.",
             "—"],
        ]
    write_table(ws, cols, vals, widths=[28, 28, 14, 70, 60], wrap_cols=(3, 4))


def sheet_coverage(wb, rows, groups) -> None:
    from api_coverage_gaps import (API_SOURCES, UNUSED_API_FIELDS,
                                   LOW_VALUE_FIELDS)
    ws = wb.create_sheet("Coverage Gaps")
    ws.sheet_properties.tabColor = "C00000"

    # -- Section A: API-level inventory --------------------------------------
    ws.append(["SECTION A — What each API payload serves vs. what the pipeline extracts"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12, color="C00000")
    ws.append(["API & endpoint", "What the payload serves",
               "What the pipeline extracts today", "Left unused in the payload"])
    style_header(ws, 4, row=ws.max_row)
    for api, serves, extracts, unused in API_SOURCES:
        ws.append([api, serves, extracts, unused])
        for c in range(1, 5):
            cell = ws.cell(row=ws.max_row, column=c)
            cell.border = BORDER
            cell.alignment = WRAP
            if unused:
                fill_cell(cell, AMBER_FILL)  # amber = payload content we never open
    ws.append([""])

    # -- Section B: specific unused fields -> candidate features --------------
    ws.append(["SECTION B — Unused API fields → concrete candidate features (expansion shortlist)"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12, color="C00000")
    ws.append(["Category", "API source", "Field in payload", "What the field is",
               "Candidate feature recipe", "Effort", "Expected value",
               "Raw support in the codebase today"])
    style_header(ws, 8, row=ws.max_row)
    for (cat, api, field, what, recipe, effort, value, support) in UNUSED_API_FIELDS:
        ws.append([cat, api, field, what, recipe, effort, value, support])
        for c in range(1, 9):
            cell = ws.cell(row=ws.max_row, column=c)
            cell.border = BORDER
            cell.alignment = WRAP
            if cat in ("Defense", "Umpires"):
                fill_cell(cell, RED_FILL)      # dead domains — the biggest gaps
            elif effort == "Low" and value == "High":
                fill_cell(cell, BLUE_FILL)     # cheapest high-value trials
    ws.append([""])

    # -- Section C: cataloged but deliberately not recommended -----------------
    ws.append(["SECTION C — Present in payloads, deliberately NOT recommended as features"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12, color="C00000")
    ws.append(["API", "Field", "Why not"])
    style_header(ws, 3, row=ws.max_row)
    for api, field, why in LOW_VALUE_FIELDS:
        ws.append([api, field, why])
        for c in range(1, 4):
            cell = ws.cell(row=ws.max_row, column=c)
            cell.border = BORDER
            cell.alignment = WRAP
    ws.append([""])

    # -- Section D: computed-but-untrialed columns -----------------------------
    ws.append(["SECTION D — Computed into the dataset but never trialed (zero new engineering)"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12, color="C00000")
    ws.append(["Feature", "Notes"])
    tier2 = [(n, g) for n, g in groups.items() if "never ablated" in g or "culled" in g]
    for n, g in sorted(tier2):
        ws.append([n, f"Config group: {g} — engineered, PIT-safe, in the dataset; "
                      f"not in the 225-pool. Zero new engineering to trial."])
    ws.append([""])

    # -- Section E: dead categories ---------------------------------------------
    ws.append(["SECTION E — Categories with zero features anywhere in the pool"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=12, color="C00000")
    ws.append(["Category", "In pool", "Raw support"])
    have = {f["category"] for f in rows}
    support = {
        "Umpires": "umpire_map/umpire_stats masters exist; game-feed officials + "
                   "Statcast called-pitch zone data are unused",
        "Defense": "fielder/alignment columns arrive in Statcast but are dropped "
                   "at load (ingestion.py UNUSED_COLS)",
    }
    any_dead = False
    for cat in ("Umpires", "Defense"):
        if cat not in have:
            any_dead = True
            ws.append([cat, 0, support.get(cat, "")])
            for c in range(1, 4):
                ws.cell(row=ws.max_row, column=c).fill = RED_FILL
    if not any_dead:
        ws.append(["(none — every raw domain now has at least one pool feature)", "", ""])

    set_widths(ws, [24, 34, 30, 44, 58, 9, 12, 42])


def sheet_glossary(wb) -> None:
    ws = wb.create_sheet("Glossary")
    ws.sheet_properties.tabColor = "808080"
    terms = [
        ("Logloss", "The model's accuracy score for win-probabilities — lower is better; "
                    "the primary number the RFE tries to improve."),
        ("AUC", "How well the model ranks games (which team is more likely to win) — "
                "higher is better; a quality guard during trials."),
        ("ECE", "Calibration error: how close predicted probabilities are to actual frequencies "
                "(e.g., does the model win 60% of games it calls 60%?) — lower is better."),
        ("Brier", "A squared-error cousin of logloss for probabilities — lower is better."),
        ("Noise bar (commit threshold)", "The minimum improvement a trial must show before the "
                                         "engine believes it — set at twice the standard error of the "
                                         "baseline, so luck can't pass."),
        ("Walk-forward fold", "One train-then-test window in time order — the test always comes "
                              "after the training period, like real predictions."),
        ("Standard error (SE)", "How much the logloss number would wobble if history were rerun — "
                                "smaller with more games, which is why full-history tests are sharper."),
        ("Importance", "How much the model used a feature while fitting (split gains + coefficients) — "
                       "a ranking hint only, never proof of value."),
        ("Member routing", "Which of the ensemble's models (xgboost, lightgbm, randomforest, mlp, "
                           "logistic) actually see the feature."),
        ("PIT-safe", "Point-in-time safe: the feature uses only information that existed before "
                     "first pitch — no lookahead."),
        ("Redundant pair", "Two features with |r| ≥ 0.9 correlation — they measure nearly the same "
                           "thing, so the model splits credit; tested consecutively."),
        ("Serving width", "The exact feature list the live model consumes — the universe (61) until "
                          "you adopt a subset."),
        ("Forced trial list", "Env vars (MLB_RFE_ADDITION_/REMOVAL_MONEYLINE_LIST) that make the next "
                              "run test specific features first — forcing the test, never the result."),
    ]
    ws.append(["Term", "Meaning (no jargon)"])
    style_header(ws, 2)
    for t, d in terms:
        ws.append([t, d])
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=2).alignment = WRAP
    set_widths(ws, [26, 110])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def generate_workbook(trace_path: Optional[Path] = None,
                      out_path: Optional[Path] = None) -> Optional[Path]:
    """Build the workbook from the newest (or given) RFE trace.

    Returns the written .xlsx path, or None when no trace exists. Pure
    build — every input is an already-published artifact; no model
    training. Called by master_pipeline Phase 4.6 after each RFE run and
    by this module's CLI.
    """
    trace_path = Path(trace_path) if trace_path else find_latest_trace()
    if trace_path is None:
        print(f"[workbook] no mlb_feature_selection_*.json trace in {DATA_DELIVERY}")
        return None
    print(f"[workbook] trace: {trace_path}")
    trace = load_trace(trace_path)
    metadata = load_metadata()
    drift = load_drift()
    coverage = load_coverage()
    universe, candidates, known = load_pool()
    groups = load_candidate_groups()
    per_feature = build_rfe_index(trace)
    _steps = trace.get("steps", []) or []
    if _steps and not per_feature:
        raise ValueError(
            f"Trace has {len(_steps)} steps but ZERO parsed into per-feature "
            "records — step schema drift (expected v2 action/feature or v1 "
            "removed/added). Refusing to build a silently-blank workbook.")

    rows = build_rows(universe, candidates, metadata, drift, coverage,
                      groups, trace, per_feature)

    # ---- verification -----------------------------------------------------
    names = [f["name"] for f in rows]
    assert len(names) == len(known) == len(set(names)),         f"pool mismatch: rows={len(names)} known={len(known)} unique={len(set(names))}"
    assert set(names) == set(known), "row set != KNOWN_FEATURE_COLS"
    blanks = [f["name"] for f in rows if not f["category"] or not f["type"]]
    assert not blanks, f"blank category/type: {blanks}"
    for f in rows:
        t = f["rfe"]
        if t and t.get("committed"):
            assert "COMMITTED" in (f["verdict_txt"] or "").upper(), f["name"]

    # ---- workbook ---------------------------------------------------------
    wb = Workbook()
    wb.remove(wb.active)
    sheet_readme(wb, trace)
    sheet_dashboard(wb, rows, trace)
    sheet_master(wb, rows, "All Features (Master)", "548235")
    prod = [f for f in rows if f["in_production"]]
    cand = [f for f in rows if not f["in_production"]]
    sheet_master(wb, prod, f"Production ({len(prod)})", "375623", subset=prod)
    sheet_master(wb, cand, f"Candidates ({len(cand)})", "C55A11", subset=cand)
    sheet_run_detail(wb, trace)
    sheet_per_fold(wb, trace)
    sheet_redundancy(wb, rows, trace)
    sheet_coverage(wb, rows, groups)
    sheet_glossary(wb)

    if out_path:
        out = Path(out_path)
    else:
        # Name mirrors the trace KIND, parsed from the trace FILENAME (ground
        # truth): full -> mlb_feature_workbook_<date>.xlsx (the production
        # view; a targeted run never touches it); targeted ->
        # mlb_feature_workbook_<date>_targeted_<HHMM>.xlsx (its own artifact).
        # Both are dated mlb_ records on the 10-day retention window.
        m = re.fullmatch(r"mlb_feature_selection_(\d{4}-\d{2}-\d{2})(.*)",
                         trace_path.stem)
        day, suffix = (m.group(1), m.group(2)) if m else (
            str(trace.get("date", "unknown")), "")
        out = DATA_DELIVERY / f"mlb_feature_workbook_{day}{suffix}.xlsx"
    wb.save(out)
    print(f"[workbook] wrote {out} "
          f"({len(rows)} features | {len(trace.get('steps', []))} RFE tests)")
    return out


def main() -> int:
    if os.name == "nt":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Build the MLB Feature Decision Workbook from the newest "
                    "RFE trace + published artifacts")
    ap.add_argument("--trace", default=None,
                    help="path to mlb_feature_selection_<date>.json "
                         "(default: newest trace in data_delivery)")
    ap.add_argument("--out", default=None,
                    help="output .xlsx path (default: data_delivery/"
                         "mlb_feature_workbook_<trace-date>.xlsx)")
    args = ap.parse_args()
    out = generate_workbook(args.trace, args.out)
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())
