"""Optional human-readable NBA feature-selection workbook."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
try:
    from backend import config
except ImportError:
    import config

def write_feature_workbook(out_dir, date_c: str, trace: dict | None = None) -> str | None:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return None
    path = Path(out_dir) / f"nba_feature_workbook_{date_c}.xlsx"
    rows = [{"feature": c, "served": c in config.active_moneyline_feature_cols(), "candidate": c in config.RFE_CANDIDATE_COLS} for c in config.KNOWN_FEATURE_COLS]
    # The trace is a record with list-valued fields of different lengths;
    # ``DataFrame(trace)`` would raise rather than produce a workbook.  A
    # key/value sheet keeps the provenance readable and JSON-stable.
    trace_rows = [
        {"key": key,
         "value": json.dumps(value, sort_keys=True, default=str)
         if isinstance(value, (list, dict, tuple)) else str(value)}
        for key, value in sorted((trace or {}).items())
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Feature Pool")
        pd.DataFrame(trace_rows, columns=["key", "value"]).to_excel(
            writer, index=False, sheet_name="Trace")
    return path.name
