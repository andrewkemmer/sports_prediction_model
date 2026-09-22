"""NFL RFE decision workbook generator.

This is an artifact-only companion to feature_selection.py. It performs no
training and never changes the active production feature list.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

BACKEND = Path(__file__).resolve().parent
DELIVERY = BACKEND.parent / "data_delivery"
NAVY = "1F3864"
BLUE = PatternFill("solid", fgColor=NAVY)
WHITE = Font(color="FFFFFF", bold=True)


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


def _workbook_filename(trace: dict[str, Any], path: Path) -> str:
    raw_day = str(trace.get("date", ""))[:10]
    try:
        day = datetime.fromisoformat(raw_day).date().isoformat()
    except ValueError:
        day = raw_day if len(raw_day) == 10 and raw_day[4] == "-" else path.stem[-8:]
    targeted = bool(trace.get("targeted")) or "_targeted" in path.stem
    if not targeted:
        return f"nfl_feature_workbook_{day}.xlsx"
    stamp = ""
    try:
        stamp = datetime.fromisoformat(str(trace.get("created_utc", "")).replace("Z", "+00:00")).strftime("%H%M")
    except ValueError:
        stamp = datetime.now(timezone.utc).strftime("%H%M")
    return f"nfl_feature_workbook_{day}_targeted_{stamp}.xlsx"


def generate_workbook(trace_path: str | None = None, out_path: str | None = None) -> str | None:
    path = Path(trace_path) if trace_path else _latest_trace()
    if path is None or not path.exists():
        return None
    trace = _load_trace(path)
    target = Path(out_path) if out_path else DELIVERY / _workbook_filename(trace, path)
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

    steps = wb.create_sheet("RFE Run Detail")
    headers = ["step", "kind", "feature", "n_features", "logloss", "logloss_gain",
               "paired_se", "commit_threshold", "auc", "ece", "committed"]
    steps.append(headers)
    for rec in trace.get("steps", []):
        metrics = rec.get("metrics") or {}
        steps.append([rec.get("step"), rec.get("kind"), rec.get("feature"),
                      rec.get("n_features"), metrics.get("logloss"),
                      rec.get("logloss_gain", rec.get("logloss_gain")),
                      rec.get("paired_se"), rec.get("commit_threshold"),
                      metrics.get("auc"), metrics.get("ece"), rec.get("committed")])

    folds = wb.create_sheet("Per-Fold Detail")
    folds.append(["step", "feature", "fold_id", "validation_window", "n_games", "logloss"])
    for rec in trace.get("steps", []):
        for fold in (rec.get("metrics") or {}).get("per_fold", []):
            folds.append([rec.get("step"), rec.get("feature"), fold.get("fold_id"),
                          fold.get("val_window"), fold.get("n_games"), fold.get("logloss")])

    pool = wb.create_sheet("Feature Pool")
    pool.append(["feature", "role", "tested_in_trace"])
    selected = set(trace.get("selected_cols", []))
    tested = {str(x.get("feature")) for x in trace.get("steps", [])}
    for feature in trace.get("selected_cols", []):
        pool.append([feature, "production contract", feature in tested])
    for feature in sorted(tested - selected):
        pool.append([feature, "trialed candidate/removal", True])

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.fill = BLUE
            cell.font = WHITE
            cell.alignment = Alignment(wrap_text=True)
        for col in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(col)].width = min(48, max(14, max(
                len(str(sheet.cell(row=r, column=col).value or ""))
                for r in range(1, min(sheet.max_row, 30) + 1)) + 2))
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    wb.save(target)
    return str(target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--out")
    args = ap.parse_args()
    print(generate_workbook(args.trace, args.out) or "no trace found")
