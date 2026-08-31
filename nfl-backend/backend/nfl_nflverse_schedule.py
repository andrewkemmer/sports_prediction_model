"""Native nflverse schedule loader for the moneyline slate.

The decided 2019-2025 frame comes from nflreadpy (``load_schedules``). The
installed nflreadpy season validator caps its feed at 2025, so it cannot serve
the 2026 scheduled games — even though the underlying nflverse ``games.csv``
release (the same source nflreadpy reads) fully contains 2026.

This module reads that native artifact directly from nflverse's GitHub release
(``games.csv``) — reachable from Kaggle, unlike ESPN — and returns the
requested season's rows in the exact nflreadpy shape the slate builder consumes
(``season``, ``week``, ``gameday``, ``gametime``, ``stadium``, ``home_team`` /
``away_team`` abbreviations, ``home_score``/``away_score``, ``roof``, ``temp``,
``wind``, ``div_game``, ``spread_line``, ``total_line``). Scheduled (2026) rows
carry NaN scores, so ``build_slate_features`` treats them as the pre-game slate.

Why this beats a committed CSV: reading the live GitHub artifact means the slate
is always current (late-season flexes included) with zero refresh/maintenance and
zero dependence on a manually-updated file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# nflverse schedules release (the canonical schedule source nflreadpy reads).
GAMES_URL = ("https://github.com/nflverse/nflverse-data/releases/"
             "download/schedules/games.csv")

# The slate columns nflreadpy's schedule exposes and build_slate_features uses.
_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "sports_prediction_model nfl-nflverse")}


def select_season_rows(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Network-free helper: return the nflverse rows for ``season`` with scores
    coerced to numeric (NaN for scheduled rows)."""
    out = df[df["season"] == int(season)].copy()
    for c in ("home_score", "away_score"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


def load_nflverse_games(season: int, url: str | None = None) -> pd.DataFrame:
    """Download the nflverse ``games.csv`` release and return the rows for
    ``season`` (empty frame when the season isn't present)."""
    target = url or GAMES_URL
    df = pd.read_csv(target, storage_options=_UA)
    return select_season_rows(df, season)