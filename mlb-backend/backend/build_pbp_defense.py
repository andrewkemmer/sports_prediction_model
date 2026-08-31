"""Build a curated defense projection from the full Statcast pbp frame and
commit it as data_delivery/pbp_defense_<date>.parquet.

The pipeline's full ~88/119-column Statcast frame lives only on the Kaggle
run; the committed 8-column lean cache (pbp_chunks/) cannot express batted-
ball-allowed or position-split defense. This script projects a CURATED
defense-relevant subset (keeps the lean cache untouched so no current
consumer breaks), backfills 2024, and records self-documenting metadata.

Publication lag rule (project PIT discipline): day T results are available
for T+1. Every downstream ladder must filter game_date < target game date,
which respects the lag by construction.

Usage (Kaggle, where the full frame exists):
    python build_pbp_defense.py --source /kaggle/working/pbp_full.parquet \
        --end 2026-08-31
    python build_pbp_defense.py --source ... --backfill-2024   # include 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR  # noqa: E402

# Verified present in the pybaseball statcast frame (June 2025 + May 2024
# probes): identity, batted-ball, fielders, alignment, WIP outcomes.
PBP_DEFENSE_COLS = [
    # identity
    "game_pk", "game_date", "home_team", "away_team",
    "inning", "inning_topbot", "batter", "pitcher",
    "events", "game_type", "type",
    # batted ball
    "launch_speed", "launch_angle", "bb_type", "hit_distance_sc",
    "hc_x", "hc_y", "hit_location", "launch_speed_angle",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    # fielders (2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF)
    "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    # alignment + situation
    "if_fielding_alignment", "of_fielding_alignment",
    "outs_when_up", "on_1b", "on_2b", "on_3b",
    # outcome quality
    "woba_value", "woba_denom", "babip_value", "iso_value",
]

TRAINING_START = date(2024, 3, 20)  # the moneyline training window start


def project_frame(full: pd.DataFrame) -> pd.DataFrame:
    """Project the curated defense subset; missing columns become NaN."""
    cols = [c for c in PBP_DEFENSE_COLS if c in full.columns]
    out = full[cols].copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True,
                    help="Full Statcast pbp parquet from the pipeline run")
    ap.add_argument("--end", type=str, required=True,
                    help="Coverage end date YYYY-MM-DD")
    ap.add_argument("--backfill-2024", action="store_true",
                    help="Include rows from 2024-03-20 (training start)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    full = pd.read_parquet(args.source)
    proj = project_frame(full)
    if not args.backfill_2024:
        proj = proj[pd.to_datetime(proj["game_date"]).dt.date >= date(2025, 1, 1)]
    else:
        proj = proj[pd.to_datetime(proj["game_date"]).dt.date >= TRAINING_START]

    end = date.fromisoformat(args.end)
    out = args.out or (DATA_DELIVERY_DIR / f"pbp_defense_{end:%Y%m%d}.parquet")
    # zstd: ~2M rows x ~45 numeric/string cols lands around 15-25 MB (snappy
    # would be ~45-60 MB — over GitHub's 50 MB warning). zstd is pyarrow's
    # built-in; no extra dependency.
    proj.to_parquet(out, index=False, compression="zstd", compression_level=7)

    n_rows = len(proj)
    size_mb = out.stat().st_size / 1e6
    dmin = pd.to_datetime(proj["game_date"]).min()
    dmax = pd.to_datetime(proj["game_date"]).max()
    meta = {
        "artifact": out.name,
        "rows": int(n_rows),
        "size_mb": round(size_mb, 2),
        "columns": list(proj.columns),
        "date_min": str(dmin.date()),
        "date_max": str(dmax.date()),
        "coverage_end": args.end,
        "backfill_2024": bool(args.backfill_2024),
        "publication_lag": "day T results available for T+1; "
                           "downstream ladders filter game_date < target",
        "source_frame_cols": int(full.shape[1]),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} rows={n_rows} size={size_mb:.1f}MB "
          f"range={dmin.date()}..{dmax.date()}")


if __name__ == "__main__":
    main()
