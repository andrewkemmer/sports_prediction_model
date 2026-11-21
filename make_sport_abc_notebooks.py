"""Generate the four Kaggle notebooks for the 2026-11 A/B/C and NBA retune."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "/kaggle/working/sports_prediction_model"
#: Kernels that must clone a branch other than the default.
BRANCHES: dict[str, str] = {}


def clone_cell(sport: str) -> str:
    branch = BRANCHES.get(sport)
    extra = f', "--branch", "{branch}", "--single-branch"' if branch else ""
    return f'''import os, subprocess

REPO = "{REPO}"
URL = "https://github.com/andrewkemmer/sports_prediction_model.git"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--quiet", "--depth", "1"{extra}, URL, REPO],
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
         "outputs": [], "source": [clone_cell(sport)]},
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
         "outputs": [], "source": [clone_cell("nba")]},
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
    "nhl": ("NHL moneyline A/B/C: Elo + win% vs on-ice-weighted team rating",
            "A incumbent, B removes the six Elo/win% names, C adds a team "
            "rating that weights the served xPM/60 cells by who is on the "
            "ice (3F:2D even strength, 4F:1D power play, 88:12 ice share) "
            "and adds goalie goals saved above the expanding league save "
            "percentage. Replaces the equal eight-way mean.",
            "pandas numpy scipy sklearn xgboost lightgbm pyarrow requests tqdm"),
    "nfl": ("NFL moneyline A/B/C: Elo + win% vs summed DVOA team rating",
            "A incumbent, B removes the six Elo/win% names, C sums the four "
            "per-game DVOA legs (defence-adjusted) and subtracts defensive "
            "EPA/play converted to EPA/game using the opponent's projected "
            "plays. Replaces the four-way mean that mixed EPA/game with "
            "EPA/play.",
            "pandas numpy scipy sklearn xgboost lightgbm pyarrow nflreadpy "
            "polars tqdm gitpython"),
    "mlb": ("MLB moneyline A/B/C: Elo + win% vs two-sided run model",
            "A incumbent, B removes all six Elo/win% names the contract "
            "serves (diff and per-side spellings), C adds the seven fitted "
            "columns from a causal ridge of runs on side rows: "
            "off/def rating home/away/diff plus net_rating_diff.",
            "pandas numpy scipy sklearn xgboost lightgbm pyarrow duckdb tqdm"),
}


#: Kernel notebook filenames.  MLB moved to v3 with the two-sided run model;
#: the other three keep their v2 kernels untouched.
NOTEBOOKS = {"nhl": "kaggle_nhl_abc_v3.ipynb",
             "nfl": "kaggle_nfl_abc_v3.ipynb",
             "mlb": "kaggle_mlb_abc_v3.ipynb"}


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sports", nargs="*", default=list(SPECS),
                    choices=list(SPECS))
    ap.add_argument("--skip-nba", action="store_true",
                    help="leave the NBA retune notebook untouched")
    args = ap.parse_args(argv)
    for sport in args.sports:
        title, desc, packages = SPECS[sport]
        nb = notebook(abc_cells(sport, title, desc, packages))
        path = ROOT / NOTEBOOKS[sport]
        path.write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", path)
    if not args.skip_nba:
        path = ROOT / "kaggle_nba_retune_v36.ipynb"
        path.write_text(json.dumps(notebook(nba_cells()), indent=1) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
