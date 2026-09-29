"""Build point-in-time injured-list (IL) stint intervals from MLB transactions.

Input:  MLB StatsAPI ``/api/v1/transactions?sportId=1`` — one call per
        calendar year, cached as JSON outside the repo (``MLB_CACHE_DIR`` or
        the system temp dir) so re-runs are free and no 20MB of raw
        transactions is ever staged to GitHub.

Output: data_delivery/il_stints.parquet   (+ .meta.json provenance)
        batter  int64   MLB person id
        il_start date    first transaction date of the stint
        il_end   date    activation date, NULL while the stint is still open

Why this exists: the expected-lineup features (features.py ``lineup_agg``)
draw their candidate pool from batters who ACTUALLY BATTED, and rank by a
30-game trailing PA count that freezes the day a player stops playing. An IL
player therefore sits in the projected nine for weeks after he stops playing.
The filter cannot bind on that pool -- he is absent by construction -- so the
widen-then-filter rebuild in ``features.py`` needs this table to know who was
unavailable as of each game date.

PIT RULE: an interval opens and closes on the transaction ``date``, never on
``effectiveDate``. Placements are frequently retroactive ("placed on the 15-day
injured list retroactive to June 27" is filed later, as its own transaction);
at bet time on June 27 the placement was not yet public, so keying on
``effectiveDate`` leaks the future into the feature.

STATE MACHINE, NOT OPEN/CLOSE PAIRING. One stint emits SEVERAL placement
records -- the initial "placed on the 10-day", a "transferred to the 60-day"
escalation, and often a duplicate retroactive copy -- but only ONE activation.
Pairing 7 placements against 2 activations leaves 5 permanently "on IL" and
inflates the league-wide count to an impossible ~1,117. Walking a boolean per
batter collapses each stint to exactly one interval.

THREE STATES, NOT TWO. Leaving the big-league active roster also ends a
stint. Teams routinely DFA or outright an IL player to clear a 40-man spot,
and MLB files NO activation for them because they never came back -- so a
two-state machine leaves them "on the IL" from the DFA date until the day they
re-sign years later. Measured on 2023-2026: 44 such carry-over phantoms, 7 of
them real major leaguers (3,653 PA rows, 0.16% of all PA) whose ratings the
filter would suppress for up to three years. "On the IL" has to mean "on the
active roster but hurt"; a minor leaguer is not on the active roster at all,
so ``_OFF_ROSTER`` closes the stint and any later placement opens a new one.

  NOT treated as closing: rehab assignments (a player on rehab is
  unavailable to the active roster, which is the only question this table is
  asked), trades and waiver claims (the receiving team INHERITS the IL spot --
  calling those a close would mark an injured player available).

RECONCILED AGAINST OBSERVED PLATE APPEARANCES, NOT ONLY THE WORD "ACTIVATED".
The transactions feed is a club's narrative and it is incomplete: a player on
a long IL who rehabbed four times and shuttled to the minors is never given an
"activated" row, so his single interval ran 2023-05-19 to the end of the
window -- three years during which he batted in 320 games. The filter then
deleted a real hitter from his own team's projected nine in every one of
them, which is why the first A/B of this feature came back net NEGATIVE:
56 batters, 3,541 player-games where the table said "on the IL" and the batter
demonstrably had a plate appearance.

A batter who had a plate appearance in a game was in the lineup for that game,
so no IL interval may span it. Where a stint's transaction close is later than
the batter's first appearance after the placement, the appearance is the
truth and the stint ends there. Regexes for recall/contract-selection recover
29% of those cases; a blunt 120-day time cap recovers 68%; this recovers
99.5% (3,541 -> 19 player-games) and, unlike a time cap, it never closes a
stint on a date where the player did NOT play, so it costs zero real
absences: every unflagged date is one on which he was in the lineup.

PIT: still strictly subtractive at the FAR end only. A game before the
batter's first appearance keeps him flagged, which is both correct and
knowable at that date. A game after it does not, which is knowable then too
because his own prior appearances are already in the record. A plate
appearance ON the placement date is ignored, not treated as a close: clubs
file the IL transaction the same evening a player is hurt, so same-date
appearances are the announcement, not a return. All 19 survivors are
exactly that case.

Regeneration path for data_delivery/il_stints.parquet (like
backfill_lineups.py -- not part of the daily run, because the feature
degrades loudly rather than failing closed when the table is stale):

    python build_il_stints.py --start 2023-01-01 --end 2026-09-25
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR  # noqa: E402

TX_URL = "https://statsapi.mlb.com/api/v1/transactions"

# Verified against all 48,041 cached transactions (2024-2026):
#   placements  = "placed X on the N-day injured list" / "transferred X to ..."
#   activations = ANY "activated" row. Only 2,037 of 3,589 repeat the words
#                 "injured list"; StatsAPI frequently writes a bare
#                 "activated RHP X.", so matching closes on /injur/ discards
#                 1,552 real closing events and the "on IL" curve then climbs
#                 monotonically 29 -> 1,117 instead of settling near ~120.
#
# GENERALIZED AVAILABILITY (2026-09-30): the projected nine needs
# "unavailable as of game date", not merely "hurt". The same
# wholesale-refetchable transactions feed carries every FORMAL
# non-IL unavailability move (injured, paternity, bereavement and
# family medical, restricted, administrative leave, suspension,
# option/reassignment to the minors), so the table extends without
# any new source or incremental state (full rebuild in one pass):
#   opens  += paternity / bereavement + family medical / restricted
#             lists, administrative leave, suspensions, and optioned or
#             reassigned to the minors (the player cannot take a
#             major-league at-bat)
#   closes += recalled / contract selected / purchased -- an
#             optioned player returns exactly that way
# Administrative leave (Wander Franco 2024-03-28, Emmanuel Clase
# 2025-07-28: the only two rows the 2023-2026 feed carries) files
# NO matching return transaction -- the player returns via the plain
# "activated" that any roster move emits, and in both real cases the
# league kept the ROSTER papers open with a restricted-list
# re-placement (Franco 2024-07-10, Clase same-day 2025-07-28), so the
# stint closes that way. The split-on-reopen machine handles the
# restricted re-placement; a leave WITHOUT a restricted co-filing
# would close on the return activation like any other stint.
# Day-to-day manager holds (Moreno/Bibee/Gausman class) file NO
# transaction and stay honestly invisible here; the roster-status
# sweep is their watchtower. Rehab assignments open nothing: a
# rehabbing player is already flagged by his open IL stint.
# PIT contract unchanged: interval bounds are transaction DATES
# (moves are pre-game filings -- knowable by first pitch of the
# filing day), a same-date appearance is the announcement and never
# a return, and PA reconciliation stays strictly subtractive at the
# far end only.
_OPEN = re.compile(r"\b(placed|transferred)\b.*\binjured list\b"
                   r"|\bplaced\b.*\bpaternity list\b"
                   r"|\bplaced\b.*\bbereavement list\b"
                   r"|\bplaced\b.*\bfamily medical\b"
                   r"|\bplaced\b.*\brestricted list\b"
                   r"|\bplaced\b.*\badministrative leave\b"
                   r"|\bsuspended\b"
                   r"|\boptioned\b"
                   r"|\breassigned\b.*\b(minor league|minors)\b", re.I)
_CLOSE = re.compile(r"\bactivated\b|\breinstated\b|\brecalled\b"
                    r"|\bselected the contract\b|\bpurchased\b", re.I)
_OFF_ROSTER = re.compile(r"\bdesignated\b.*\bfor assignment\b"
                         r"|\bsent\b.*\boutright\b"
                         r"|\breleased\b", re.I)

# Plausibility gate, two-sided. The broken open/close pairing produced a
# monotone climb to 1,117 players "on the IL" league-wide. Real utilization
# drifts year to year, and the DEFINITION sets the scale: the IL-only era
# measured weekly medians 212/248/267/301 (2023-2026); the generalized
# availability definition (IL + paternity + bereavement/family + suspended
# + restricted + optioned to the minors, 2026-09-30) measures
# 496/507/530/559 on the same window. The band below is pinned to the
# availability era: the ceiling sits ~60% above the highest observed
# season (the broken-pairing failure mode would read ~4x) and the floor
# catches the opposite regression, a match that finds nobody ever
# unavailable. Per-season medians are printed and recorded in the meta so
# drift stays visible either way. (The gate itself proved the point on
# extension day: the first wider rebuild measured 510 and was REFUSED
# against the old IL-era ceiling until the band was re-derived here.)
_MIN_MEDIAN_ON_IL = 100
_MAX_MEDIAN_ON_IL = 900

# Second gate, same spirit: the residual defect count. After reconciliation
# the table may only claim a batter is on the IL for dates on which he had no
# plate appearance, so the residual must be ~0. Measured 19 on 2023-2026, and
# every one of them is a same-date placement/appearance pair that is correct
# behaviour, so the budget is set well clear of the real number to catch a
# silent regression (an unclosed stint, a dropped reconciliation, a wrong
# batter key) rather than to police the announcement-lag cases.
_MAX_RESIDUAL_ON_IL_PGAMES = 250

# Third gate, and the one that keeps the second honest: a pitch projection
# that stops months before the window end finds no appearances after that
# date, so reconciliation silently becomes a no-op and the residual counter
# -- which measures against the SAME file -- reports zero while every stint
# after the cut stays unclosed. Generous enough not to false-fail a legitimate
# end-of-season rebuild.
_MAX_PBP_LAG_DAYS = 45

# The first season of the moneyline training window opens 2024-03-20, but
# stints placed in autumn 2023 are still open when it does. Fetch a full
# preceding calendar year so the state machine sees the offseason placements
# it needs to avoid calling those players available.
DEFAULT_START = "2023-01-01"
GATE_WARMUP_DAYS = 60  # skip the fetch year's own January/February


# ── transaction fetch ───────────────────────────────────────────────────────

def cache_dir(explicit: Path | None = None) -> Path:
    """Where raw per-year transaction JSON lives (never inside the repo)."""
    if explicit is not None:
        d = explicit
    else:
        env = os.environ.get("MLB_CACHE_DIR", "").strip()
        d = Path(env) if env else Path(os.environ.get("TEMP", "/tmp")) / "mlb_il_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plan_year_refresh(years: list[int], *, refresh: bool, offline: bool,
                      today: "date | None" = None) -> dict[int, bool]:
    """Per-year refresh decision for the transaction fetch.

    --refresh refreshes everything, --offline nothing. The DEFAULT
    behavior refreshes the FINAL year when it is the current calendar
    year: a closed past year never changes, but the current year gains
    transactions every day, and the documented regeneration command
    must not silently reuse yesterday's cache (a rebuild that picks up
    nothing new is how the IL table goes stale while looking fresh).
    """
    today = today or date.today()
    if refresh:
        return {y: True for y in years}
    if offline:
        return {y: False for y in years}
    return {y: (y == years[-1] and y == today.year) for y in years}


def fetch_year(year: int, cache: Path, refresh: bool = False,
               attempts: int = 3) -> list[dict]:
    """One StatsAPI call per calendar year, cached. Never partially trusted:
    a failed fetch raises rather than returning a short list, because a
    silently truncated year looks exactly like "nobody was injured"."""
    path = cache / f"il_tx_{year}.json"
    if path.exists() and not refresh:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    import requests  # imported late: --offline must work without a network

    url = (f"{TX_URL}?sportId=1&startDate={year}-01-01"
           f"&endDate={year}-12-31")
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            payload = r.json().get("transactions", [])
            break
        except Exception as e:  # network flake / 5xx / bad JSON
            last = e
            print(f"  {year}: attempt {i + 1}/{attempts} failed ({e})")
            time.sleep(2 ** (i + 1))
    else:
        raise SystemExit(f"could not fetch {year} transactions: {last}")
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  {year}: {len(payload):,} transactions -> {path.name}")
    return payload


# ── state machine ───────────────────────────────────────────────────────────

def stints_from_events(events: dict[int, list[tuple[str, int]]]) -> pd.DataFrame:
    """Collapse each batter's dated (date, kind) events into IL intervals.

    kind 0 = activation, 1 = placement, 2 = left the active roster. Sorting by
    (date, kind) resolves same-day collisions in the order that matches
    reality: an activation BEFORE a same-day re-placement leaves the player
    injured, and a placement BEFORE a same-day DFA leaves a zero-length stint.
    """
    out: list[dict] = []
    for batter, evs in events.items():
        evs.sort(key=lambda x: (x[0], x[1]))
        on = False
        start: str | None = None
        for d, kind in evs:
            if kind == 1:
                if not on:
                    on, start = True, d
                elif start is not None and d > start:
                    # A NEW placement while already "on" OPENS A NEW
                    # STINT, splitting the walk (2026-09-29 audit fix).
                    # Two real shapes were swallowed by the old
                    # ignore-if-on guard: (a) a return via minor-league
                    # RECALL that files no activation (Campero: the
                    # 08-26 re-placement landed while the 06-06 stint
                    # walked open) and (b) a same-day activate+place
                    # pair whose open stint PA-reconciles shut while
                    # the player is still out (Meadows 2026). Splitting
                    # keeps continuity (no availability is invented:
                    # close-at-d/reopen-at-d is injury-contiguous),
                    # lets reconciliation truncate each segment at the
                    # player's own appearances, and leaves a same-date
                    # duplicate copy byte-identical (zero-length head
                    # stint, dropped downstream).
                    out.append({"batter": batter,
                                "il_start": pd.Timestamp(start),
                                "il_end": pd.Timestamp(d)})
                    start = d
                # d <= start: a same/earlier-date duplicate copy — the
                # pre-existing guard semantics, kept.
            elif kind in (0, 2) and on:
                out.append({"batter": batter,
                            "il_start": pd.Timestamp(start),
                            "il_end": pd.Timestamp(d)})
                on, start = False, None
        if on:
            out.append({"batter": batter, "il_start": pd.Timestamp(start),
                        "il_end": pd.NaT})
    if not out:
        return pd.DataFrame(columns=["batter", "il_start", "il_end"])
    return (pd.DataFrame(out).sort_values(["batter", "il_start"])
            .reset_index(drop=True))


def reconcile_with_plate_appearances(iv: pd.DataFrame,
                                     pa: pd.DataFrame) -> pd.DataFrame:
    """End a stint at the batter's first observed plate appearance.

    ``pa`` is a frame of (batter, game_date) plate appearances. A stint that
    the transactions left open, or closed late, is truncated at the earliest
    appearance STRICTLY AFTER ``il_start``; a stint whose transaction close
    already precedes that appearance is untouched. The result is a no-op on
    every date where the batter did not play: the rule can only shorten an
    interval at a date where the batter was in the lineup, so it cannot
    release a genuine absence back into the projected nine.
    """
    if iv.empty or pa.empty:
        return iv
    out = iv.copy()
    days = {int(b): g.game_date.to_numpy(dtype="datetime64[ns]")
            for b, g in pa.sort_values(["batter", "game_date"])
                           .groupby("batter", sort=False)}
    ends: list[object] = []
    for batter, start, end in zip(out.batter, out.il_start, out.il_end):
        arr = days.get(int(batter))
        nxt = None
        if arr is not None:
            i = int(np.searchsorted(arr, np.datetime64(start), side="right"))
            if i < len(arr):
                nxt = pd.Timestamp(arr[i])
        if nxt is not None and (pd.isna(end) or nxt < end):
            end = nxt
        ends.append(end)
    out["il_end"] = ends
    out = out[out.il_end.isna() | (out.il_end > out.il_start)]
    return out.reset_index(drop=True)


def load_plate_appearances(pbp_path: Path) -> pd.DataFrame:
    """(batter, game_date) for every game in which the batter had a PA."""
    import duckdb
    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(
            """
            SELECT DISTINCT CAST(batter AS BIGINT) AS batter,
                            CAST(game_date AS DATE)   AS game_date
            FROM read_parquet(?)
            WHERE events IS NOT NULL AND batter IS NOT NULL
            """, [str(pbp_path)]).df()
    finally:
        con.close()


def residual_on_il_player_games(iv: pd.DataFrame, pa: pd.DataFrame) -> int:
    """Player-games the table flags as on-IL where the batter actually had a
    plate appearance. The defect count the second gate is watching.

    Appearances ON the placement date are not counted: clubs file the IL
    transaction the same evening a player is hurt, so a batter who appears
    and is placed the same day is the announcement, not a contradiction.
    Counted per (batter, game_date) once even where intervals overlap.
    """
    if iv.empty or pa.empty:
        return 0
    hit = 0
    for batter, group in pa.groupby("batter", sort=False):
        days = np.sort(group.game_date.to_numpy(dtype="datetime64[ns]"))
        sel = iv[iv.batter == batter]
        if sel.empty:
            continue
        covered = np.zeros(len(days), dtype=bool)
        for _, s in sel.iterrows():
            end = s.il_end
            lo = int(np.searchsorted(days, np.datetime64(s.il_start), side="right"))
            hi = (len(days) if pd.isna(end)
                  else int(np.searchsorted(days, np.datetime64(end), side="left")))
            if hi > lo:
                covered[lo:hi] = True
        hit += int(covered.sum())
    return int(hit)


def check_pbp_covers(pbp_name: str, pa: pd.DataFrame,
                     end: pd.Timestamp) -> None:
    """Refuse a pitch projection that stops long before the window end.

    A projection that stops months early finds no appearances after that
    date, so reconciliation silently becomes a no-op and the residual counter
    -- measured against the SAME file -- still reports zero while every stint
    after the cut stays unclosed. That is the one way the second gate can be
    blinded, so it is checked before anything is written.
    """
    if pa.empty:
        raise SystemExit(f"{pbp_name} yielded no plate appearances at all; "
                         f"reconciling against it would be a silent no-op")
    lag = (end - pd.Timestamp(pa.game_date.max())).days
    if lag > _MAX_PBP_LAG_DAYS:
        raise SystemExit(
            f"{pbp_name} stops {lag} days before the window end "
            f"(budget {_MAX_PBP_LAG_DAYS}). Reconciliation would find no "
            f"appearances after that date and silently do nothing while the "
            f"residual counter, measured against the same file, still reads "
            f"zero. Rebuild the projection or pass --pbp")


def default_pbp_source() -> Path | None:
    """Newest committed pitch projection, used to observe plate appearances.

    data_delivery/pbp_defense_*.parquet is the daily run's own 27-column
    projection of pitches.parquet -- the same rows the feature is built from,
    so reconciling against it introduces no new source of truth.
    """
    cands = sorted(Path(DATA_DELIVERY_DIR).glob("pbp_defense_*.parquet"))
    return cands[-1] if cands else None


def build_events(tx: list[dict]) -> dict[int, list[tuple[str, int]]]:
    events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for t in tx:
        d = str(t.get("description", ""))
        if _CLOSE.search(d):
            kind = 0
        elif _OPEN.search(d):
            kind = 1
        elif _OFF_ROSTER.search(d):
            kind = 2
        else:
            continue
        pid = (t.get("person") or {}).get("id")
        dte = t.get("date")
        if not pid or not dte:
            continue
        events[int(pid)].append((dte, kind))
    return events


def median_on_il(iv: pd.DataFrame, start: pd.Timestamp,
                 end: pd.Timestamp) -> float:
    """Median weekly league-wide count over the in-season part of the window."""
    days = pd.date_range(start, end, freq="7D")
    if len(days) == 0:
        return 0.0
    live = [int(((iv.il_start <= x) & (iv.il_end.isna() | (iv.il_end > x))).sum())
            for x in days]
    return float(pd.Series(live).median())


def season_medians(iv: pd.DataFrame, years: list[int]) -> dict[str, float]:
    """Weekly median on-IL per season, March 1 -> window end (offseason
    troughs are real but not what the gate is guarding against)."""
    out: dict[str, float] = {}
    for y in years:
        lo = pd.Timestamp(f"{y}-03-01")
        hi = pd.Timestamp(f"{y}-12-31") if y < years[-1] else None
        out[str(y)] = round(median_on_il(iv, lo, hi or pd.Timestamp.max), 0)
    return out


# ── live roster-status verification sweep (status OUT/IL/DTD channel) ──────

TEAM_IDS = {
    "BAL": 110, "BOS": 111, "NYY": 147, "TB": 139, "TOR": 141,
    "CLE": 114, "DET": 116, "KC": 118, "MIN": 142, "CWS": 145,
    "HOU": 117, "LAA": 108, "ATH": 133, "SEA": 136, "TEX": 140,
    "ATL": 144, "MIA": 146, "NYM": 121, "PHI": 143, "WSH": 120,
    "CHC": 112, "CIN": 113, "MIL": 158, "PIT": 134, "STL": 138,
    "AZ": 109, "COL": 115, "LAD": 119, "SD": 135, "SF": 137,
}

ROSTER_STATUS_URL = ("https://statsapi.mlb.com/api/v1/teams/{team_id}"
                     "/roster?rosterType=40Man&date={day}")


def fetch_roster_status(team_id: int, day: str,
                        attempts: int = 3) -> list[dict] | None:
    """One team's 40-man roster with per-date statuses (None on failure).

    The endpoint honors ``date`` (verified: a player reads Active on the
    day before his IL placement and Injured N-Day the day after), which
    makes it a live second evidence channel for the table at window
    end. Advisory only: a failed fetch is reported, never fatal.
    """
    import requests  # late import keeps --offline dependency-free
    url = ROSTER_STATUS_URL.format(team_id=team_id, day=day)
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** (i + 1))
                continue
            r.raise_for_status()
            return r.json().get("roster", [])
        except Exception:
            if i == attempts - 1:
                return None
            time.sleep(2 ** (i + 1))
    return None


def classify_roster_statuses(roster: list[dict] | None) -> dict[str, list[dict]]:
    """Split one roster into injured / day-to-day players.

    Status vocabulary served by the feed (verified 2026-09-29, 5 teams):
    Active, Injured 7/10/15/60-Day, Reassigned to Minors, Not Yet
    Reported, Family Medical Emergency. Day-to-Day historically exists
    in this field; the sweep reports it separately because it is NOT an
    IL stint (the player stays on the active roster) and the daily
    transactions feed carries zero such rows.
    """
    out = {"injured": [], "day_to_day": []}
    for p in roster or []:
        desc = str((p.get("status") or {}).get("description", ""))
        low = desc.lower()
        person = p.get("person") or {}
        if not person.get("id"):
            continue
        pos = str((p.get("position") or {}).get("abbreviation", ""))
        if pos in ("P", "TWP"):
            # Pitchers never enter a BATTER's projected nine — the
            # sweep's missing-placement signal is about position
            # players (the 2026-09-29 audit: reliever Finnegan read as
            # a miss with zero career plate appearances).
            continue
        rec = {"batter": int(person["id"]),
               "name": person.get("fullName", ""),
               "status": desc,
               "position": pos}
        if "day-to-day" in low or "day to day" in low:
            out["day_to_day"].append(rec)
        elif "injured" in low:
            out["injured"].append(rec)
    return out


def verify_table_against_statuses(iv: pd.DataFrame,
                                  injured: list[dict],
                                  day: pd.Timestamp,
                                  active_ids: set[int] | None = None,
                                  pa_days: pd.DataFrame | None = None
                                  ) -> dict:
    """Cross-check the built table against live statuses at ``day``.

    missed_placements: players the live feed says are on the IL but the
    table does not flag on that date — the Josh Jung 2026-09-26 case
    (a placement filed after the table's transaction window). This is
    the inclusion failure: the player stays in the projected nine.
    suppressed: players the table flags but the live feed calls Active
    — the opposite error (a wrongly-open stint deletes a real hitter
    from his own nine); the PA reconciliation should prevent it, so a
    nonzero here is a loud smell, not necessarily a defect (status
    feeds can lag an activation by a day).
    """
    flagged: set[int] = set()
    if not iv.empty:
        live = ((iv.il_start <= day)
                & (iv.il_end.isna() | (iv.il_end > day)))
        flagged = set(iv.loc[live, "batter"].astype(int))
    missed, suppressed = [], []
    live_ids = {r["batter"] for r in injured}
    for rec in injured:
        if rec["batter"] not in flagged:
            missed.append(rec)
    if not iv.empty:
        names = {int(r["batter"]): r["name"] for r in injured}
        over = flagged - live_ids
        if active_ids is not None:
            # Only a player on a LIVE roster can be a real
            # contradiction; open stints for retired/off-roster ids
            # are out of scope entirely.
            over &= active_ids
        if pa_days is not None and over:
            # GROUND-TRUTH contradiction: the table claims injured on
            # dates the player demonstrably BATTED. Roster membership
            # alone over-reports — a player activated after the PA
            # projection's cut reads Active with no contradicting PA
            # (status lag, not a defect). A PA strictly after the
            # stint's start is the same evidence the reconciliation
            # itself trusts; if one survived, something is broken.
            pmax = pa_days.groupby("batter").game_date.max()
            over = {b for b in over
                    if b in set(pmax.index)
                    and pmax[b] > iv.loc[iv.batter == b,
                                         "il_start"].max()}
        for b in sorted(over):
            suppressed.append({"batter": b,
                               "name": names.get(b, str(b))})
    return {"missed_placements": missed, "suppressed": suppressed,
            "live_injured": len(injured), "table_flagged": len(flagged)}


def run_status_sweep(iv: pd.DataFrame, day: pd.Timestamp,
                     strict: bool = False,
                     pa_days: pd.DataFrame | None = None) -> int:
    """Fetch all 30 rosters at ``day`` and verify the table; return code.

    Advisory by default (exit 0 with a loud report); --strict-status
    turns missed placements into a hard failure so a scheduled rebuild
    refuses to publish a table that silently keeps an IL player in a
    projected nine.
    """
    injured: list[dict] = []
    dtd: list[dict] = []
    failures = 0
    active_ids: set[int] = set()
    pa = pa_days
    day_str = str(pd.Timestamp(day).date())
    for abbr, tid in sorted(TEAM_IDS.items()):
        roster = fetch_roster_status(tid, day_str)
        if roster is None:
            failures += 1
            print(f"  status sweep: {abbr} fetch FAILED (advisory)")
            continue
        active_ids |= {int(p["person"]["id"]) for p in roster
                       if (p.get("person") or {}).get("id")}
        cls = classify_roster_statuses(roster)
        injured += cls["injured"]
        dtd += cls["day_to_day"]
    print(f"status sweep {day_str}: {len(injured)} live-IL players, "
          f"{len(dtd)} day-to-day, {failures} fetch failures")
    rep = verify_table_against_statuses(
        iv, injured, pd.Timestamp(day), active_ids=active_ids,
        pa_days=pa)
    if rep["missed_placements"]:
        print(f"  MISSED PLACEMENTS ({len(rep['missed_placements'])}) — "
              "live IL but unflagged in the table (stays in the "
              "projected nine):")
        for r in rep["missed_placements"]:
            print(f"    {r['batter']} {r['name']} [{r['status']}]")
    else:
        print("  missed placements: none")
    if rep["suppressed"]:
        print(f"  suppressed vs live (table flags, feed says Active): "
              f"{len(rep['suppressed'])}")
        for r in rep["suppressed"][:10]:
            print(f"    {r['batter']} {r['name']}")
    if dtd:
        print(f"  day-to-day designations ({len(dtd)}) — reported, NOT "
              "IL stints; no StatsAPI history exists for them:")
        for r in dtd[:10]:
            print(f"    {r['batter']} {r['name']} [{r['status']}]")
    if strict and rep["missed_placements"]:
        print("STRICT: refusing to finish with missed placements")
        return 1
    return 0


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START,
                    help="first transaction date to fetch (default %(default)s)")
    ap.add_argument("--end", required=True, help="last transaction date")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch even when a cached year exists")
    ap.add_argument("--offline", action="store_true",
                    help="cache only; never touch the network")
    ap.add_argument("--pbp", type=Path, default=None,
                    help="pitch projection used to observe plate appearances "
                         "(default: newest data_delivery/pbp_defense_*.parquet)")
    ap.add_argument("--no-reconcile", action="store_true",
                    help="skip plate-appearance reconciliation; only for "
                         "reproducing the pre-reconciliation table")
    ap.add_argument("--check-roster-status", action="store_true",
                     help="sweep all 30 teams 40Man roster statuses "
                          "at --end and verify the table (advisory "
                          "report; catches a placement the transactions "
                          "window missed)")
    ap.add_argument("--strict-status", action="store_true",
                     help="with --check-roster-status: exit non-zero on "
                          "missed placements instead of reporting them")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise SystemExit(f"--end {end.date()} precedes --start {start.date()}")
    # Whole calendar years, never clipped: the state machine is only correct
    # if it sees every placement/activation inside each year it walks.
    years = list(range(start.year, end.year + 1))
    cache = cache_dir(args.cache_dir)
    tx: list[dict] = []
    print(f"transactions {start.date()}..{end.date()} "
          f"({len(years)} season call(s)), cache={cache}")
    refresh_plan = plan_year_refresh(
        years, refresh=args.refresh, offline=args.offline)
    auto = [y for y, r in refresh_plan.items()
            if r and not args.refresh and not args.offline]
    if auto:
        print(f"  auto-refreshing current-year cache: {auto} "
              "(use --offline to pin)")
    for y in years:
        if args.offline:
            p = cache / f"il_tx_{y}.json"
            if not p.exists():
                raise SystemExit(f"--offline but {p} is missing")
            with open(p, encoding="utf-8") as fh:
                tx += json.load(fh)
        else:
            tx += fetch_year(y, cache, refresh=refresh_plan[y])

    iv = stints_from_events(build_events(tx))
    pbp = args.pbp or default_pbp_source()
    pa = pd.DataFrame()
    tx_residual = 0
    if not args.no_reconcile:
        if pbp is None or not pbp.exists():
            raise SystemExit(
                "no pitch projection for plate-appearance reconciliation "
                "(pass --pbp PATH). Reconciling is not optional: without it a "
                "player whose return is recorded as a minor-league recall never "
                "closes his stint and the table claims he is on the IL for "
                "years while he bats every day")
        pa = load_plate_appearances(pbp)
        check_pbp_covers(pbp.name, pa, end)
        tx_residual = residual_on_il_player_games(iv, pa)
        before = len(iv)
        iv = reconcile_with_plate_appearances(iv, pa)
        print(f"reconciled against {pbp.name}: {len(pa):,} plate appearances, "
              f"{before:,} -> {len(iv):,} stints")
    else:
        print("WARNING: --no-reconcile; the table is NOT reconciled against "
              "observed plate appearances and the unclosed-stint defect is back")
    residual = residual_on_il_player_games(iv, pa)
    dur = (iv.il_end - iv.il_start).dt.days.dropna()
    gate_from = max(start, start + pd.Timedelta(days=GATE_WARMUP_DAYS))
    med = median_on_il(iv, gate_from, end)
    per_season = season_medians(iv, years)
    print(f"il stints: {len(iv):,} over {iv.batter.nunique():,} batters "
          f"({int(iv.il_end.isna().sum()):,} still open at window end)")
    print(f"  stint length days: median {dur.median():.0f} "
          f"p90 {dur.quantile(0.9):.0f} max {dur.max():.0f}")
    print(f"  on-IL weekly median by season: {per_season}")
    print(f"  on-IL league-wide (weekly median from {gate_from.date()}): {med:.0f}")
    print(f"  on-IL while batting: {residual} player-games "
          f"({tx_residual} before reconciliation, budget "
          f"{_MAX_RESIDUAL_ON_IL_PGAMES})")
    if not _MIN_MEDIAN_ON_IL <= med <= _MAX_MEDIAN_ON_IL:
        raise SystemExit(
            f"IMPLAUSIBLE on-IL median {med:.0f} "
            f"(band {_MIN_MEDIAN_ON_IL}..{_MAX_MEDIAN_ON_IL}; per-season "
            f"{per_season}) - the state machine regressed; refusing to write")
    if residual > _MAX_RESIDUAL_ON_IL_PGAMES:
        raise SystemExit(
            f"{residual:,} player-games claim a batter is on the IL while he "
            f"had a plate appearance (budget {_MAX_RESIDUAL_ON_IL_PGAMES}) - "
            f"reconciliation did not run or regressed; refusing to write")

    out = args.out or (Path(DATA_DELIVERY_DIR) / "il_stints.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    iv.to_parquet(out, index=False)
    meta = {
        "rows": int(len(iv)),
        "batters": int(iv.batter.nunique()),
        "open_at_window_end": int(iv.il_end.isna().sum()),
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "seasons_fetched": years,
        "transactions": len(tx),
        "median_on_il_weekly": med,
        "median_on_il_by_season": per_season,
        "gate_band": [_MIN_MEDIAN_ON_IL, _MAX_MEDIAN_ON_IL],
        "stint_length_days_median": float(dur.median()),
        "pit_rule": "interval bounds are transaction DATES, not effectiveDate",
        "definition": ("availability stints: IL + paternity + bereavement/family"
                       " + suspended + restricted + administrative leave"
                       " + optioned/reassigned to minors; closes:"
                       " activated/reinstated/recalled/contract-selected/"
                       "purchased"),
        "source": "MLB StatsAPI /api/v1/transactions?sportId=1",
        "reconciled_against": None if args.no_reconcile else pbp.name,
        "plate_appearances": int(len(pa)),
        "plate_appearances_through": (None if pa.empty
                                      else str(pd.Timestamp(pa.game_date.max()).date())),
        "on_il_while_batting_player_games": int(residual),
        "on_il_while_batting_before_reconcile": int(tx_residual),
        "residual_budget": _MAX_RESIDUAL_ON_IL_PGAMES,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) + .meta.json")
    if args.check_roster_status:
        sys.exit(run_status_sweep(iv, end, strict=args.strict_status,
                                  pa_days=pa))


if __name__ == "__main__":
    main()
