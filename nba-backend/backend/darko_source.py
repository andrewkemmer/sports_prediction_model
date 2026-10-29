"""DARKO per-player DPM history, fetched during the pipeline run.

Each player page on darko.app carries the full trajectory: one row per career
game with that game's pre-game projection, under ``history: {keys, values}``
in a SvelteKit ``/player/<nba_id>/__data.json`` payload. The row dated ``d``
is the projection entering game ``d``, so ``apply_darko_rating`` can read the
last row strictly before each target date and never see a game's own result.

The history is pulled in memory for the players in the rating frame on every
run; nothing is written to disk and no manual step is needed.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

logger = logging.getLogger(__name__)

BASE = "https://www.darko.app"

#: Columns kept from each player's history table (the rest is market scaffolding).
KEEP = ["nba_id", "date", "season", "seconds_played", "future_game",
        "team_name", "opp_id", "tm_id", "dpm", "o_dpm", "d_dpm", "box_dpm",
        "box_odpm", "box_ddpm", "on_off_dpm", "bayes_rapm_total", "x_minutes",
        "x_pace", "x_pts_100", "x_ast_100"]

NUMERIC = ["nba_id", "season", "seconds_played", "future_game",
           "opp_id", "tm_id", "dpm", "o_dpm", "d_dpm", "box_dpm", "box_odpm",
           "box_ddpm", "on_off_dpm", "bayes_rapm_total", "x_minutes",
           "x_pace", "x_pts_100", "x_ast_100"]


def _fetch_payload(url: str) -> dict:
    try:
        from curl_cffi import requests as crequests
        r = crequests.get(url, timeout=45, impersonate="chrome")
    except ImportError:
        import requests
        r = requests.get(url, timeout=45,
                         headers={"User-Agent": "Mozilla/5.0 (research)"})
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8"))


def _decode_table(flat: list, node: dict) -> pd.DataFrame:
    names = [flat[j] for j in flat[node["keys"]]]
    refs = node["values"] if isinstance(node["values"], list) else flat[node["values"]]
    cols = [r if not isinstance(r, int) else flat[r] for r in refs]
    if len(names) != len(cols):
        raise RuntimeError(f"table shape mismatch {len(names)} vs {len(cols)}")

    def cell(col, i):
        x = col[i]
        if isinstance(x, int) and not isinstance(x, bool) \
                and 0 <= x < len(flat) and not isinstance(flat[x], list):
            return flat[x]
        return x

    n = min(len(c) for c in cols)
    return pd.DataFrame({name: [cell(col, i) for i in range(n)]
                         for name, col in zip(names, cols)})


def _player_history(player_id: int) -> tuple[int, pd.DataFrame | None, str]:
    url = f"{BASE}/player/{player_id}/__data.json"
    try:
        payload = _fetch_payload(url)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return player_id, None, f"{type(exc).__name__}: {str(exc)[:80]}"
    nodes = [n for n in payload.get("nodes", []) if n and n.get("data")]
    if not nodes:
        return player_id, None, "no data node"
    flat = nodes[0]["data"]
    root = flat[0]
    if not isinstance(root, dict) or "history" not in root:
        return player_id, None, "no history key"
    hist_ref = root["history"]
    node = hist_ref if isinstance(hist_ref, dict) else flat[hist_ref]
    try:
        frame = _decode_table(flat, node)
    except Exception as exc:  # noqa: BLE001
        return player_id, None, f"decode {type(exc).__name__}: {str(exc)[:60]}"
    frame = frame[[c for c in KEEP if c in frame.columns]].copy()
    frame["nba_id"] = player_id
    return player_id, frame, ""


def fetch_darko_history(player_ids, workers: int = 6) -> pd.DataFrame:
    """DARKO history rows for the given NBA player ids, fetched now.

    Players whose page cannot be read are logged and skipped; their ratings
    keep the incumbent RAPM value through ``apply_darko_rating``.
    """
    ids = sorted({int(i) for i in player_ids})
    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid, frame, err in pool.map(_player_history, ids):
            if frame is None:
                failures.append((pid, err))
            elif len(frame):
                frames.append(frame)
    if failures:
        logger.warning("DARKO history: %d of %d player(s) unavailable "
                       "(first: %s)", len(failures), len(ids), failures[0])
    if not frames:
        logger.warning("DARKO history: no rows fetched for %d player(s)", len(ids))
        return pd.DataFrame(columns=KEEP)
    panel = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel[panel.date.notna()]
    logger.info("DARKO history: %d rows for %d player(s)",
                len(panel), panel.nba_id.nunique())
    return panel.sort_values(["nba_id", "date"]).reset_index(drop=True)
