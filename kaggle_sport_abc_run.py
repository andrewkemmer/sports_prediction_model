"""Push, watch and collect the three sport A/B/C kernels plus the NBA retune.

The Kaggle transport (one push per kernel, slow status polling, 429 backoff)
is inherited from ``kaggle_coverage_run``.  Credentials come from the
environment / ``~/.kaggle``; this script never reads or prints them.

    export KAGGLE_USERNAME=andrewkemmer
    export PATH="$PWD/.venv-kaggle/bin:$PATH"
    python kaggle_sport_abc_run.py push
    python kaggle_sport_abc_run.py status
    python kaggle_sport_abc_run.py watch --poll-seconds 600
    python kaggle_sport_abc_run.py collect
"""
from __future__ import annotations

import argparse
from pathlib import Path

import kaggle_coverage_run as base

ROOT = Path(__file__).resolve().parent
base.COLLECT_ROOT = ROOT / "kaggle_sport_abc_output"
base.KERNELS = {
    "nba": {
        "slug": "nba-moneyline-retune-v3-6",
        "title": "NBA moneyline retune v3.6",
        "notebook": ROOT / "kaggle_nba_retune_v36.ipynb",
        "datasets": [],
    },
    "nhl": {
        "slug": "nhl-moneyline-a-b-c-no-elo-v2",
        "title": "NHL moneyline A/B/C no-Elo v2",
        "notebook": ROOT / "kaggle_nhl_abc_v2.ipynb",
        "datasets": [],
    },
    "nfl": {
        "slug": "nfl-moneyline-a-b-c-no-elo-v2",
        "title": "NFL moneyline A/B/C no-Elo v2",
        "notebook": ROOT / "kaggle_nfl_abc_v2.ipynb",
        "datasets": [],
    },
    "mlb": {
        "slug": "mlb-moneyline-a-b-c-run-model-v3",
        "title": "MLB moneyline A/B/C run model v3",
        "notebook": ROOT / "kaggle_mlb_abc_v3.ipynb",
        "datasets": [],
    },
}
SPORTS = ("nba", "nhl", "nfl", "mlb")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="all",
                    choices=("all", "push", "watch", "collect", "status"))
    ap.add_argument("--sports", nargs="*", default=list(SPORTS),
                    choices=SPORTS)
    ap.add_argument("--poll-seconds", type=float, default=600.0)
    ap.add_argument("--max-wait-minutes", type=float, default=720.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    owner = base._owner()
    if not owner and not args.dry_run:
        raise SystemExit("No Kaggle username: set KAGGLE_USERNAME.")
    ids = {s: base._kernel_id(s, owner) for s in args.sports}

    if args.action == "status":
        for s in args.sports:
            print(s, ids[s], base.status(ids[s], dry=args.dry_run))
        return 0

    if args.action in ("all", "push"):
        for s in args.sports:
            base.push(s, owner, args.force, args.dry_run)
        if args.action == "push":
            return 0

    if args.action in ("all", "watch"):
        final = base.watch(args.sports, owner, args.poll_seconds,
                           args.max_wait_minutes * 60, args.dry_run)
        print("final:", final)
        if args.action == "watch":
            return 0

    if args.action in ("all", "collect"):
        base.collect(args.sports, owner, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
