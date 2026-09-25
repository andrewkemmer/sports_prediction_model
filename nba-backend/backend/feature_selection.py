"""Record-only NBA feature-selection trace."""
from __future__ import annotations
import json
from pathlib import Path
try:
    from backend import config
except ImportError:
    import config

def run_rfe(game_df=None, out_dir=None, date_c: str | None = None, adopt: bool = False) -> dict:
    record = {"feature_set_version": config.FEATURE_SET_VERSION, "served_columns": config.active_moneyline_feature_cols(), "candidate_columns": config.RFE_CANDIDATE_COLS, "adopted": False, "trials": [], "note": "record-only sweep; served contract is unchanged"}
    if out_dir is not None and date_c:
        p = Path(out_dir) / f"nba_feature_selection_{date_c}.json"; p.write_text(json.dumps(record, indent=1, default=str))
    return record
