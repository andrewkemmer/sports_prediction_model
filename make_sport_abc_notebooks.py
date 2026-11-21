"""Generate the four Kaggle notebooks for the 2026-11 A/B/C and NBA retune."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "/kaggle/working/sports_prediction_model"
CLONE = f'''import os, subprocess

REPO = "{REPO}"
URL = "https://github.com/andrewkemmer/sports_prediction_model.git"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", URL, REPO],
                   check=True)
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--oneline"],
                     capture_output=True, text=True).stdout.strip())
'''


def install(packages: str) -> str:
    return f'''import importlib, subprocess, sys
need = [p for p in "{packages}".split() if importlib.util.find_spec(p) is None]
if need:
    print("installing:", need, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need],
                   check=True)
print("deps ok", flush=True)
'''


def abc_cells(sport: str, title: str, description: str, packages: str):
    run = f'''import json, subprocess, sys, time
from pathlib import Path

REPO = "{REPO}"
OUT = Path("/kaggle/working/sport_abc/{sport}")
FRAME = Path("/kaggle/working/{sport}_frame.parquet")
OUT.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "sport_abc.py", "--sport", "{sport}",
       "--out", str(OUT), "--frame", str(FRAME), "--save-frame"]
print(" ".join(cmd), flush=True)
t0 = time.time()
rc = subprocess.call(cmd, cwd=REPO)
print(f"run rc={{rc}} in {{(time.time() - t0) / 60:.1f}} min", flush=True)
if rc:
    raise SystemExit(f"sport_abc failed with exit code {{rc}}")
print((OUT / "summary.json").read_text()[:30000])
'''
    return [
        {"cell_type": "markdown", "metadata": {},
         "source": [f"# {title}\n\n{description}\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [CLONE]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [install(packages)]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [run]},
    ]


def nba_cells():
    run = f'''import json, subprocess, sys, time
from pathlib import Path

REPO = "{REPO}"
OUT = Path("/kaggle/working/nba_retune_v36")
FRAME = Path("/kaggle/working/nba_frame.parquet")
OUT.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "nba_retune.py", "--out", str(OUT),
       "--frame", str(FRAME), "--save-frame"]
print(" ".join(cmd), flush=True)
t0 = time.time()
rc = subprocess.call(cmd, cwd=REPO)
print(f"run rc={{rc}} in {{(time.time() - t0) / 60:.1f}} min", flush=True)
if rc:
    raise SystemExit(f"nba_retune failed with exit code {{rc}}")
print((OUT / "metrics.json").read_text()[:30000])
'''
    return [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# NBA three-member retune on `nba-prod-v3.6-noelo`\n\n"
                    "Tune xgboost, lightgbm and elastic-net on chronological "
                    "inner holdouts, then run the full causal walk-forward "
                    "and refit all three members on the 92-column contract. "
                    "No market offset.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [CLONE]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [install(
             "pandas numpy scipy sklearn xgboost lightgbm pyarrow "
             "pdfplumber tqdm gitpython openpyxl")]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [run]},
    ]


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SPECS = {
    "nhl": ("NHL moneyline A/B/C: Elo + win% vs team player rating",
            "A incumbent, B removes the six Elo/win% names, C adds a "
            "team rating from the eight served player evo/PP rate columns.",
            "pandas numpy scipy sklearn xgboost lightgbm pyarrow requests tqdm"),
    "nfl": ("NFL moneyline A/B/C: Elo + win% vs team player rating",
            "A incumbent, B removes the six Elo/win% names, C adds a "
            "team rating from player EPA minus defensive EPA allowed.",
            "pandas numpy scipy sklearn xgboost lightgbm pyarrow nflreadpy "
            "polars tqdm gitpython"),
    "mlb": ("MLB moneyline A/B/C: Elo + win% vs team lineup rating",
            "A incumbent, B removes the Elo/win% names present in the "
            "adopted serving contract, C adds mean position-pool xwOBA.",
            "pandas numpy scipy sklearn xgboost lightgbm pyarrow duckdb tqdm"),
}


def main() -> None:
    for sport, (title, desc, packages) in SPECS.items():
        nb = notebook(abc_cells(sport, title, desc, packages))
        path = ROOT / f"kaggle_{sport}_abc_v2.ipynb"
        path.write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", path)
    path = ROOT / "kaggle_nba_retune_v36.ipynb"
    path.write_text(json.dumps(notebook(nba_cells()), indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
