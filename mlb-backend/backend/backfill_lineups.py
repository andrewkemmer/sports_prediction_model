"""Lineup backfill — battingOrder for every decided game 2024–2026.

Emits data_delivery/lineups.parquet: game_pk, game_date, home_team,
away_team, home_order (9 MLB IDs), away_order, complete_home, complete_away,
state. Incremental + resumable (already-COMPLETE pks are skipped; incomplete
/ fetch_error rows are retried every run), paced at ~2.4 fetches/sec
(pause 0.15s, one retry) — under the roof fetcher's proven ~2.85/s. Reuses
the Phase 1 parser (unit-tested historically; see git history for the
retired test suite).

TWO entry points (both wired into master_pipeline Phase 1.6):
    main()                     AUDIT ONLY: post-game agreement check of stored
                               orders vs first-trip batters from pitches;
                               stamps validated_home/away as a late-scratch /
                               data-quality MONITOR, plus a provenance-
                               contradiction downgrade pass. NEVER writes
                               orders and NEVER gates membership — the
                               bet-time truth is the official posted nine,
                               scratches included (2026-10-24: judging the
                               announcement by the actuals would trade the
                               Sep-16 error for train-serve skew on every
                               scratched game). A post-game feed read has no
                               authority over a pre-game capture (2026-10-23,
                               Sep-16 NYY: the Final field disagrees with who
                               started, so fetching it again reproduces the
                               error, never fixes it).
    fetch_scheduled_lineups()  PRE-GAME capture: today's scheduled game_pks
                               via the StatsAPI schedule, posted nine only —
                               tonight's tier 1 while the feed still has it.
                               The SOLE writer of orders; latest PIT-valid
                               capture wins. Runs EVERY pipeline invocation
                               (morning + refreshes): each run is a capture
                               checkpoint, so official updates through late
                               scratches are tracked and the LAST pre-pitch
                               posted nine is what both the slate and the
                               (frozen) training row resolve.

lineups.parquet is membership tier 1 (tonight's nine): features.py resolves
lineup_effective from it, so a gap here degrades membership to the
projected nine (tier 2) — honest, but strictly worse information.

Usage:
    python backfill_lineups.py --limit 400     # chunk across invocations
    python backfill_lineups.py --limit 400     # resumes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR.parent))

from config import DATA_DELIVERY_DIR  # noqa: E402
from data_ingestion import normalize_team  # noqa: E402  (canonical team codes)
from phase1_lineup_coverage import fetch_feed, parse_batting_orders  # noqa: E402

PAUSE_SEC = 0.15
STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
# Sparse archived-snapshot projection: the lineup repair only needs the
# feed-attested timestamp, game state, team abbreviations, and batting
# orders. Verified against the live StatsAPI shape; roughly 10 KB instead
# of several hundred KB per snapshot.
SNAPSHOT_LINEUP_FIELDS = (
    "gameData,status,abstractGameState,teams,team,abbreviation,"
    "liveData,boxscore,teams,battingOrder,metaData,timeStamp"
)
# Full-repopulation parallelism (NHL/NBA structural pattern: their
# ingestion resolves workers via env with a parallel default and uses the
# pool for EVERY pull, not just full rebuilds — NHL PULL_WORKERS=8,
# NBA PULL_WORKERS=4. The timestamp-index and snapshot calls are
# independent across games; this keeps a complete corpus rebuild inside the
# normal sub-hour pipeline budget and drops the daily incremental repair
# (limit=200 → 400 HTTP calls) from minutes to seconds. Each game is still
# retried per-game and failures degrade loudly rather than aborting.
FULL_SNAPSHOT_WORKERS = 8
SNAPSHOT_WORKERS_ENV = "MLB_SNAPSHOT_WORKERS"
SNAPSHOT_WORKERS_DEFAULT = 8


def resolve_snapshot_workers(workers: int | None = None) -> int:
    """Worker count for snapshot repair: explicit arg > env > default.

    Mirrors NBA _resolve_workers / NHL _resolve_workers so a future
    ``MLB_SNAPSHOT_WORKERS=16`` can overlap more games without a code
    change, and a constrained host can pin it down the same way.
    """
    import os
    if workers is not None:
        try:
            workers = int(workers)
        except (TypeError, ValueError):
            return SNAPSHOT_WORKERS_DEFAULT
        return max(1, workers)
    try:
        return max(1, int(os.environ.get(SNAPSHOT_WORKERS_ENV, "") or SNAPSHOT_WORKERS_DEFAULT))
    except (TypeError, ValueError):
        return SNAPSHOT_WORKERS_DEFAULT
LINEUPS_COLUMNS = ["game_pk", "game_date", "home_team", "away_team",
                   "home_order", "away_order", "complete_home",
                   "complete_away", "state",
                   # Provenance + trust (2026-10-23, Sep-16 NYY case: the feed's
                   # battingOrder for a Final game disagreed with who actually
                   # started - 2 false inclusions, 2 false exclusions - and no
                   # capture timestamp existed to say when it was read):
                   # captured_utc / first_pitch_utc bound the POINT IN TIME;
                   # source is 'pregame' (pre-game capture) or 'backfill'
                   # (legacy post-game read); pit_valid means captured strictly
                   # before first pitch; validated_home/away are the post-game
                   # agreement check vs first-trip batters (NULL = unchecked).
                   # Tier-1 trust (mirrors features.py order_sides exactly):
                   # complete_* AND (pit_valid IS NOT FALSE OR
                   # source='backfill'/'actuals'/NULL). Provenance-only by design
                   # (2026-10-24): validated_* is audit output, never a
                   # gate — the official posted nine binds even when a late
                   # scratch makes it differ from the starters.
                   "captured_utc", "first_pitch_utc", "source",
                   "pit_valid", "validated_home", "validated_away",
                   # Feed-attested snapshot time (metaData.timeStamp) when the
                   # row came from a timecoded snapshot or a timestamped live
                   # capture; lets any future audit re-derive pit_valid from
                   # feed time instead of trusting captured_utc.
                   "feed_timecode",
                   # Probable-starter THROWS HAND (2026-10-03 slate alignment):
                   # lineup_effective's tier 2 conditions each slot's modal
                   # projection on the OPPOSING starter's hand, and the
                   # historical table gets that from pitches' inning-1 row —
                   # a game with no pitches yet has none. Carrying the hand on
                   # the same cache row the membership already reads keeps the
                   # slate on the IDENTICAL projection SQL instead of silently
                   # dropping to the unconditioned tier-3 modal.
                   "home_starter_hand", "away_starter_hand"]


def apply_provenance_void(df: pd.DataFrame) -> pd.DataFrame:
    """Void self-contradicting pre-game claims (pure — the single spelling).

    A 'pregame' source claim with a recorded feed state other than Preview
    is voided to backfill-grade (orders kept, pit_valid forced FALSE) —
    Preview is the ONLY state that can evidence a pre-game read. A Live
    feed is never pre-game point-in-time, even with a pre-pitch timestamp:
    once the feed flips, its batting orders live in the started-game
    information environment (subs/updates possible), so granting them PIT
    status would be dishonest. Voided rows still bind tier 1 as the
    announced-nine corpus (grandfathered) — they just never carry PIT
    privilege. Frames without a state column are returned untouched — with
    no recorded state there is no contradiction to judge (the pit_valid
    flag decides, exactly as the SQL predicate does); _normalize always
    records state (adding None), so normalized frames are always judged.
    Called by _normalize (every read/write path), mirrored by the twin in
    features._tier1_eligible_lineups (lazy import — same function, never a
    second spelling) and by the SQL UPDATE in _register_lineups.
    """
    out = df.copy()
    if "source" not in out.columns or "state" not in out.columns:
        return out
    try:
        st = out["state"].astype(str).str.lower()
        void = (out["source"] == "pregame") & (st != "preview")
        if bool(void.any()):
            out.loc[void, "source"] = "backfill"
            if "pit_valid" in out.columns:
                out.loc[void, "pit_valid"] = False
    except Exception:  # noqa: BLE001 — gate must never break normalization
        pass
    return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Column order + dtypes for a frame about to be written to the cache."""
    df = df.copy()  # never mutate the caller's frame (column backfill below)
    for c in LINEUPS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    out = df[LINEUPS_COLUMNS].copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce")
    out = out.dropna(subset=["game_pk"])
    out["game_pk"] = out["game_pk"].astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["complete_home"] = out["complete_home"].fillna(False).astype(bool)
    out["complete_away"] = out["complete_away"].fillna(False).astype(bool)
    # Legacy rows (written before provenance existed) read as post-game
    # backfill rows: grandfathered binds (the announced-nine corpus), with
    # posted-vs-actual agreement measured separately as an audit.
    out["source"] = out["source"].fillna("backfill")
    out["pit_valid"] = out["pit_valid"].fillna(False).astype(bool)
    # Provenance-contradiction void — the single spelling in
    # apply_provenance_void (same function the pandas twin imports).
    out = apply_provenance_void(out)
    for c in ("validated_home", "validated_away"):
        out[c] = out[c].astype("boolean")  # NULL = unchecked, survives round-trip
    return out


def upsert_lineups(df_new: pd.DataFrame, out_path: Path | None = None) -> int:
    """Merge rows into lineups.parquet by game_pk (latest write wins).

    Two protections: an INCOMPLETE row never downgrades an existing COMPLETE
    one — a pre-game partial post (away side not yet out) can't erase a
    nine we already know. And a PIT-valid pre-game COMPLETE row is never
    overwritten by a post-game backfill read — the post-game field carries
    no authority the pre-game capture lacked (Sep-16 NYY: the Final feed
    field disagrees with who started, so the later read is the WORSE
    source). Latest pre-game capture wins among pre-game rows.
    Returns the number of rows written.
    """
    df_new = _normalize(df_new)
    if df_new.empty:
        return 0
    path = out_path or (DATA_DELIVERY_DIR / "lineups.parquet")
    prev = pd.read_parquet(path) if path.exists() else None
    if prev is not None and len(prev):
        prev = _normalize(prev)
        known = set(prev.loc[prev["complete_home"] & prev["complete_away"],
                             "game_pk"])
        df_new = df_new[~df_new["game_pk"].isin(known)
                        | (df_new["complete_home"] & df_new["complete_away"])]
        if df_new.empty:
            return 0
        # Pre-game protection: a PIT-valid pre-game COMPLETE row is never
        # replaced by a PIT-INVALID row of any source. The post-game field
        # has no authority over the pre-game capture (it can only repeat it
        # or, as Sep-16 NYY shows, contradict it) — and that includes an
        # EVENING RE-RUN's own fetch_scheduled_lineups pass: after first
        # pitch its 'pregame'-stamped capture reads the post-game field
        # with pit_valid FALSE, and must not clobber the morning's
        # PIT-valid nine. Among PIT-valid pre-game rows latest still wins.
        guarded = prev.loc[prev["game_pk"].isin(set(df_new["game_pk"]))
                           & prev["complete_home"] & prev["complete_away"]
                           & (prev["source"] == "pregame") & prev["pit_valid"]]
        guarded_pks = set(guarded["game_pk"].astype(int))
        if guarded_pks:
            incoming_pit_ok = ((df_new["source"] == "pregame")
                               & df_new["pit_valid"])
            df_new = df_new[~(~incoming_pit_ok
                               & df_new["game_pk"].isin(guarded_pks))]
            if df_new.empty:
                return 0
        # Actuals precedence (2026-10-26): a stored boxscore nine is
        # training truth and is NEVER overwritten by a non-actuals row
        # (snapshot repair, slate capture, legacy backfill). Actuals rows
        # always proceed (idempotent re-pull). No-op until actuals rows
        # exist — legacy behavior otherwise byte-identical.
        actual_pks = set(prev.loc[(prev["source"] == "actuals")
                                  & prev["complete_home"]
                                  & prev["complete_away"],
                                  "game_pk"].astype(int))
        if actual_pks:
            df_new = df_new[(df_new["source"] == "actuals")
                            | ~df_new["game_pk"].isin(actual_pks)]
            if df_new.empty:
                return 0
        df_all = pd.concat([prev, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all = df_all.drop_duplicates(subset=["game_pk"], keep="last")
    df_all.to_parquet(path, index=False)
    return len(df_new)


def complete_game_pks(path: Path | None = None) -> set[int]:
    """Game pks whose nine is KNOWN — the resume set for main().

    Incomplete / fetch_error rows intentionally stay OUT so every run
    retries them (a transient feed failure must not freeze a game at
    projected-only membership forever).
    """
    p = path or (DATA_DELIVERY_DIR / "lineups.parquet")
    if not p.exists():
        return set()
    try:
        df = pd.read_parquet(p)
    except Exception:  # unreadable cache: re-fetch everything
        return set()
    if not len(df):
        return set()
    df = _normalize(df)
    done = df.loc[df["complete_home"] & df["complete_away"], "game_pk"]
    return set(done.astype(int))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_feed_timecode(tc: str | None) -> str | None:
    """Feed ``metaData.timeStamp`` (``YYYYMMDD_HHMMSS``, UTC) -> ISO UTC.

    None when absent/unparseable — callers fall back to wall-clock capture
    time (less authoritative, documented at the call site). Pure."""
    if not tc:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(str(tc).strip(), "%Y%m%d_%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
    except (ValueError, TypeError):
        return None


def fetch_timecodes(game_pk: int,
                    pause_sec: float = PAUSE_SEC) -> list[str]:
    """Archived snapshot timecodes for a game (lightweight index call).

    Empty list on any failure — the caller keeps its previous behavior
    (legacy boxscore read / marked projection) instead of failing."""
    import requests
    try:
        resp = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}"
            f"/feed/live/timestamps", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return [t for t in data if isinstance(t, str)]
        return []
    except Exception:  # noqa: BLE001 — fail-safe to the legacy path
        return []


def select_pregame_timecode(timecodes: list[str],
                             first_pitch_utc: str | None) -> str | None:
    """Latest snapshot timecode strictly before first pitch (pure).

    The snapshot IS the official posted nine as of its timestamp, so the
    latest pre-pitch one carries every pre-game update through late
    scratches. None when no timecode qualifies — the caller must NOT fall
    back to a post-pitch snapshot (that is the old corruption vector).
    """
    if not timecodes or not first_pitch_utc:
        return None
    try:
        fp = pd.Timestamp(first_pitch_utc, tz="UTC")
    except Exception:
        return None
    if pd.isna(fp):
        return None
    best: str | None = None
    for tc in timecodes:
        iso = parse_feed_timecode(tc)
        if iso is None:
            continue
        try:
            ts = pd.Timestamp(iso, tz="UTC")
        except Exception:
            continue
        if ts < fp and (best is None or ts > pd.Timestamp(
                parse_feed_timecode(best), tz="UTC")):
            best = tc
    return best


def fetch_snapshot(game_pk: int, timecode: str,
                   pause_sec: float = PAUSE_SEC,
                   fields: str | None = None) -> tuple[dict | None, str]:
    """Replay one archived feed snapshot (pure transport, paced + retried)."""
    from phase1_lineup_coverage import fetch_feed as _fetch_feed
    try:
        return _fetch_feed(game_pk, pause_sec=pause_sec, timecode=timecode,
                           fields=fields)
    except TypeError:  # older fetch_feed without the timecode parameter
        import requests
        try:
            params = {"timecode": timecode}
            if fields:
                params["fields"] = fields
            resp = requests.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}"
                f"/feed/live", params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json(), ""
            return None, f"http_{resp.status_code}"
        except Exception as e:  # noqa: BLE001
            return None, f"request_error: {e}"


BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"


def fetch_boxscore(pk: int, pause_sec: float = PAUSE_SEC) -> tuple[dict | None, str]:
    """One boxscore payload (pure transport, paced + retried).

    Mirrors fetch_feed/fetch_snapshot: (payload, "") or (None, reason).
    Never raises.
    """
    import requests
    import time as _time
    last_err = ""
    for attempt in (1, 2):
        try:
            resp = requests.get(BOXSCORE_URL.format(pk=int(pk)), timeout=20)
            if resp.status_code == 200:
                return resp.json(), ""
            last_err = f"http_{resp.status_code}"
        except Exception as e:  # noqa: BLE001 — paced retry, then honest miss
            last_err = f"request_error: {e}"
        if attempt == 1:
            _time.sleep(pause_sec * 2)
    return None, last_err


def boxscore_nines(box: dict | None) -> dict:
    """Actual starters from a boxscore payload: battingOrder ONLY.

    battingOrder is the starting nine (stable post-game); the `batters`
    list includes pinch-hitters, runners and replacements and must NEVER
    source training orders. Short/missing sides read as [] — the caller
    skips incomplete games (honest gap, retried next run). Pure.
    """
    out = {"home": [], "away": []}
    teams = (box or {}).get("teams") or {}
    for side in ("home", "away"):
        bo = (teams.get(side) or {}).get("battingOrder") or []
        try:
            ids = [int(x) for x in list(bo)]
        except (TypeError, ValueError):
            ids = []
        out[side] = ids
    return out


def actuals_row(game_pk: int, game_date, home_team: str, away_team: str,
                nines: dict) -> dict:
    """One training-truth cache row from boxscore starters (pure).

    source='actuals': post-game truth, bound tier 1 by rule (never PIT —
    these rows claim no pre-game knowledge, so pit_valid is False and
    state is 'Final'). Only exact 9-man sides ship; anything else is an
    honest incomplete row the next run retries.
    """
    home = list((nines or {}).get("home") or [])
    away = list((nines or {}).get("away") or [])
    ch, ca = len(home) == 9, len(away) == 9
    return {
        "game_pk": int(game_pk),
        "game_date": pd.Timestamp(game_date),
        "home_team": home_team,
        "away_team": away_team,
        "home_order": home if ch else None,
        "away_order": away if ca else None,
        "complete_home": ch,
        "complete_away": ca,
        "state": "Final",
        "home_starter_hand": None,
        "away_starter_hand": None,
        "captured_utc": _utc_now_iso(),
        "first_pitch_utc": None,
        "source": "actuals",
        "pit_valid": False,
        "validated_home": None,
        "validated_away": None,
        "feed_timecode": None,
    }


def backfill_actual_nines(games: pd.DataFrame, limit: int | None = 200,
                          full_rebuild: bool = False,
                          out_path: Path | None = None,
                          pause_sec: float = PAUSE_SEC,
                          max_workers: int | None = None) -> int:
    """Training-truth backfill: boxscore starters for decided games.

    For decided games missing an actuals-complete row (or every decided
    game under full_rebuild), pull the boxscore and upsert its
    battingOrder nine with source='actuals'.    battingOrder ONLY — never
    the batters list (a mid-game scratch, sub or pinch-hitter must never
    enter training orders). Idempotent: re-pulling rewrites the identical
    nine (upsert latest-wins; actuals rows are precedence-guarded against
    every other writer). Manual entry point
        (python backfill_lineups.py --actuals [--actuals-full]) AND the
    MLB_FULL_REPULL=1 corpus replication (master_pipeline calls it with
    limit=None, full_rebuild=True over the freshly-enumerated decided set,
    AFTER snapshot repair so PIT-valid rows are never clobbered).
    Never raises: fetch failures degrade to honest gaps retried next run.
    """
    df = games.dropna(subset=["game_pk"]).copy()
    df["game_pk"] = df["game_pk"].astype(int)
    if out_path is not None:
        out_path = Path(out_path)
    have = set()
    if not full_rebuild:
        try:
            cache = pd.read_parquet(
                out_path or (DATA_DELIVERY_DIR / "lineups.parquet"))
            cache = _normalize(cache)
            have = set(cache.loc[
                (cache["source"] == "actuals")
                & cache["complete_home"] & cache["complete_away"],
                "game_pk"].astype(int))
        except Exception:  # noqa: BLE001 — no cache: everything is todo
            have = set()
    todo = df[~df["game_pk"].isin(have)].sort_values(["game_date",
                                                        "game_pk"])
    if limit is not None:
        todo = todo.head(limit)
    if todo.empty:
        print("actuals backfill: every decided game already has a "
              "training-truth nine")
        return 0
    print(f"actuals backfill: {len(todo)} decided game(s) to pull")
    records = list(todo.itertuples())

    def _one(record) -> tuple[dict | None, str]:
        box, _err = fetch_boxscore(int(record.game_pk), pause_sec=pause_sec)
        if box is None:
            return None, "fetch_failed"
        nines = boxscore_nines(box)
        if len(nines["home"]) != 9 or len(nines["away"]) != 9:
            return None, "incomplete"
        try:
            row = actuals_row(int(record.game_pk), record.game_date,
                              record.home_team, record.away_team, nines)
        except Exception:  # noqa: BLE001 — per-game fail-safe
            return None, "error"
        return row, "actuals"

    workers = resolve_snapshot_workers(max_workers)
    ordered: list = [None] * len(records)
    if workers < 2:
        for i, record in enumerate(records):
            ordered[i] = _one(record)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            pending = {pool.submit(_one, record): i
                       for i, record in enumerate(records)}
            for future in as_completed(pending):
                ordered[pending[future]] = future.result()
        finally:
            pool.shutdown(wait=True)
    rows = []
    counts = {"fetch_failed": 0, "incomplete": 0, "error": 0}
    for result in ordered:
        row, reason = result or (None, "error")
        if row is not None:
            rows.append(row)
        else:
            counts[reason if reason in counts else "error"] += 1
    if not rows:
        print("actuals backfill: no training-truth nines pulled — cache "
              f"untouched {counts}")
        return 0
    n = upsert_lineups(pd.DataFrame(rows), out_path=out_path)
    print(f"actuals backfill: {n} training-truth nine(s) upserted "
          f"({counts['fetch_failed']} fetch failures, "
          f"{counts['incomplete']} incomplete, {counts['error']} errored)")
    return n


def _pit_valid(captured_utc, first_pitch_utc) -> bool:
    """A capture is point-in-time valid only when it demonstrably precedes
    first pitch. Missing timestamps fail closed (False) — an undated row
    must never present as pre-game knowledge."""
    try:
        cap = pd.Timestamp(captured_utc, tz="UTC")
        fp = pd.Timestamp(first_pitch_utc, tz="UTC")
    except Exception:
        return False
    if pd.isna(cap) or pd.isna(fp):
        return False
    return bool(cap < fp)


def order_row(game_pk: int, game_date, home_team: str, away_team: str,
              feed: dict | None, state: str | None = None,
              home_starter_hand: str | None = None,
              away_starter_hand: str | None = None,
              captured_utc: str | None = None,
              first_pitch_utc: str | None = None,
              source: str = "backfill",
              validated_home=None, validated_away=None,
              feed_timecode: str | None = None) -> dict:
    """One lineups.parquet row from a StatsAPI live feed (pure — testable).

    complete_* is TRUE only on an exact 9-man order per side, mirroring the
    decided backfill; anything shorter is an honest incomplete row that the
    next run retries.

    Provenance: pre-game captures pass captured_utc + first_pitch_utc +
    source="pregame"; pit_valid is derived (captured strictly before first
    pitch). A 'pregame' claim from a non-pre-game feed state (Live/Final)
    is void at construction — it degrades to source="backfill" with
    pit_valid FALSE (post-game reads keep their orders as backfill-grade,
    never pre-game provenance). Post-game backfill reads leave all three
    unset and read as grandfathered legacy.
    """
    p = {"home": [], "away": []}
    if feed is not None:
        try:
            p = parse_batting_orders(feed)
        except Exception:
            p = {"home": [], "away": []}
    if state is None and feed is not None:
        gd = feed.get("gameData") or {}
        state = (gd.get("status") or {}).get("abstractGameState") or "unknown"
    # Feed-attested time decides PIT when present (the feed's own clock,
    # not ours); otherwise the caller-supplied capture time.
    _clock = parse_feed_timecode(feed_timecode) or captured_utc
    raw_pit = _pit_valid(_clock, first_pitch_utc)
    if source == "pregame" and str(state or "").strip().lower() != "preview":
        # State gate: ONLY a Preview feed may claim pre-game provenance.
        # Live never qualifies — even pre-pitch, a live feed is not
        # pre-game point-in-time. Final least of all.
        source = "backfill"
    # pit_valid TRUE is reserved for surviving pre-game claims — a voided
    # claim can never carry it, whatever its timestamps say.
    pit_ok = raw_pit if source == "pregame" else False
    home_order = list(p.get("home") or [])
    away_order = list(p.get("away") or [])
    ch, ca = len(home_order) == 9, len(away_order) == 9
    return {
        "game_pk": int(game_pk),
        "game_date": pd.Timestamp(game_date),
        "home_team": home_team,
        "away_team": away_team,
        # a side ships its order ONLY on an exact 9 — a short/over-9 list
        # is withheld (the shipped convention), never padded or truncated
        "home_order": home_order if ch else None,
        "away_order": away_order if ca else None,
        "complete_home": ch,
        "complete_away": ca,
        "state": state or "unknown",
        "home_starter_hand": home_starter_hand,
        "away_starter_hand": away_starter_hand,
        "captured_utc": captured_utc,
        "first_pitch_utc": first_pitch_utc,
        "source": source,
        "pit_valid": pit_ok,
        "validated_home": validated_home,
        "validated_away": validated_away,
        "feed_timecode": feed_timecode,
    }


def _probable_hands(games: list, requests) -> dict[int, dict[str, str | None]]:
    """{game_pk: {'home': 'L'/'R'/None, 'away': ...}} for tier-2 conditioning.

    The schedule must be hydrated with ``probablePitcher``; ids are then
    resolved to throwing hand in ONE batched /people call. Never raises:
    a miss leaves the side None, which lineup_effective treats as
    "cannot condition" and falls back to its unconditioned window — the
    honest degrade, not a fabricated hand.
    """
    ids: set[int] = set()
    by_game: dict[int, dict] = {}
    for g in games:
        try:
            pk = int(g["gamePk"])
        except Exception:
            continue
        sides: dict[str, int | None] = {}
        for side in ("home", "away"):
            t = ((g.get("teams") or {}).get(side) or {})
            pid = (t.get("probablePitcher") or {}).get("id")
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None
            sides[side] = pid
            if pid is not None:
                ids.add(pid)
        by_game[pk] = sides
    if not ids:
        return {}
    hand: dict[int, str] = {}
    try:
        resp = requests.get("https://statsapi.mlb.com/api/v1/people",
                            params={"personIds": ",".join(str(i) for i in sorted(ids))},
                            timeout=20)
        resp.raise_for_status()
        for p in (resp.json().get("people") or []):
            h = (p.get("pitchHand") or {}).get("code")
            if p.get("id") and h in ("L", "R"):
                hand[int(p["id"])] = h
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  probable-starter hands unresolved ({e}); tier 2 "
              f"projects unconditioned")
        return {}
    return {pk: {side: hand.get(pid) if pid is not None else None
                 for side, pid in sides.items()}
            for pk, sides in by_game.items()}


def fetch_scheduled_lineups(target_date, out_path: Path | None = None,
                            pause_sec: float = PAUSE_SEC) -> int:
    """PRE-GAME capture: posted nines for ``target_date``'s scheduled games.

    Resolves the day's game_pks from the StatsAPI schedule (the same
    endpoint results.py trusts), fetches each live feed paced like the
    decided backfill, and upserts every row — complete nines upgrade the
    cache to tier 1 immediately;    unposted sides stay incomplete; post-game main() NEVER writes orders —
    it only validates them (a post-game field read has no authority over a
    pre-game capture, Sep-16 NYY). Returns rows written. Never raises:
    a schedule/feed outage degrades membership to the projected nine.
    """
    import requests
    try:
        # hydrate=team(abbreviation) is REQUIRED, not cosmetic: the bare
        # schedule response returns only {id, link, name} per team, so
        # ``abbreviation`` is missing and every row landed with home_team =
        # away_team = '' — an empty-string side can never join the pool's
        # team vocabulary, so tier 1 silently resolved to nothing for the
        # whole slate (2026-10-03: all 4 games captured, 0 usable teams).
        resp = requests.get(
            STATSAPI_SCHEDULE_URL,
            params={"sportId": 1, "startDate": pd.Timestamp(target_date).date().isoformat(),
                    "endDate": pd.Timestamp(target_date).date().isoformat(),
                    "hydrate": "team(abbreviation),probablePitcher"},
            timeout=20)
        resp.raise_for_status()
        games = (resp.json().get("dates") or [{}])[0].get("games") or []
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  lineup pre-game capture: schedule fetch failed ({e}); "
              f"membership degrades to projection")
        return 0
    rows = []
    # Tier-2 conditioning hands: batch the probable pitchers in ONE /people
    # call (ids are already on the hydrated schedule) — per-pitcher calls would
    # be 2x the games' worth of requests for two booleans.
    hands = _probable_hands(games, requests)
    for g in games:
        try:
            pk = int(g["gamePk"])
        except Exception:
            continue
        t = (g.get("teams") or {}).get("away") or {}
        h = (g.get("teams") or {}).get("home") or {}
        away = normalize_team((t.get("team") or {}).get("abbreviation")) or ""
        home = normalize_team((h.get("team") or {}).get("abbreviation")) or ""
        # First-pitch proxy: the schedule's own gameDate (scheduled start).
        # Conservative by construction — games rarely start early, so a
        # capture before the schedule time was before the game.
        first_pitch = g.get("gameDate")
        # fetch_feed paces itself (pause_sec between requests, one retry)
        feed, _err = fetch_feed(pk, pause_sec=pause_sec)
        # SAME-METHOD PROVENANCE (2026-10-24): the slate archives the feed's
        # own snapshot time (metaData.timeStamp) rather than our wall clock
        # — the identical artifact contract the training snapshot backfill
        # writes (orders + feed-attested time + Preview state gate +
        # latest-wins). Wall clock is only the fallback when the feed
        # carries no timestamp.
        feed_tc = ((feed or {}).get("metaData") or {}).get("timeStamp")
        captured = parse_feed_timecode(feed_tc) or _utc_now_iso()
        rows.append(order_row(pk, target_date, home, away, feed,
                              home_starter_hand=hands.get(pk, {}).get("home"),
                              away_starter_hand=hands.get(pk, {}).get("away"),
                              captured_utc=captured,
                              first_pitch_utc=first_pitch,
                              source="pregame",
                              feed_timecode=feed_tc))
    if not rows:
        return 0
    try:
        return upsert_lineups(pd.DataFrame(rows), out_path=out_path)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  lineup pre-game capture: persist failed ({e}); "
              f"membership degrades to projection")
        return 0


def _repair_game_from_pregame_snapshot(record, first_pitch: dict[int, str],
                                       preserved: dict[int, dict],
                                       pause_sec: float = PAUSE_SEC,
                                       snapshot_fields: str | None = None
                                       ) -> tuple[dict | None, str]:
    """Rebuild one decided game from its latest qualifying pre-pitch snapshot.

    Returns ``(row, 'repaired')`` or ``(None, reason)``; never raises. The
    caller decides how to count/log each reason. ``preserved`` carries cache
    metadata the sparse snapshot cannot supply (starter hands and audit
    validation flags).
    """
    try:
        pk = int(record.game_pk)
        fp = first_pitch.get(pk)
        if not fp:
            return None, "no_first_pitch"
        tc = select_pregame_timecode(fetch_timecodes(pk), fp)
        if tc is None:
            return None, "no_snapshot"
        kwargs = {"fields": snapshot_fields} if snapshot_fields else {}
        feed, _err = fetch_snapshot(pk, tc, pause_sec=pause_sec, **kwargs)
        if feed is None:
            return None, "fetch_failed"
        gd = feed.get("gameData") or {}
        teams = gd.get("teams") or {}
        home = normalize_team(
            ((teams.get("home") or {}).get("abbreviation"))) or ""
        away = normalize_team(
            ((teams.get("away") or {}).get("abbreviation"))) or ""
        captured = parse_feed_timecode(tc) or _utc_now_iso()
        row = order_row(pk, record.game_date, home, away, feed,
                        home_starter_hand=None,
                        away_starter_hand=None,
                        captured_utc=captured,
                        first_pitch_utc=fp,
                        source="pregame",
                        feed_timecode=tc)
        # Preserve metadata the sparse snapshot cannot carry.
        prev = preserved.get(pk, {})
        for col in ("home_starter_hand", "away_starter_hand",
                    "validated_home", "validated_away"):
            if col in row and pd.isna(row[col]) and not pd.isna(prev.get(col)):
                row[col] = prev.get(col)
        return row, "repaired"
    except Exception:  # noqa: BLE001 — per-game fail-safe
        return None, "error"


def decided_games_from_pitches(pbp_path: Path | str | None = None
                               ) -> pd.DataFrame:
    """Decided games enumerated from the FRESH pitch frame — never stored history.

    Returns game_pk (int), game_date (date), home_team, away_team, home_win
    (bool) for every game whose post_home_score AND post_away_score are both
    present (official finals). The full-repull path feeds this frame to every
    lineup-corpus pass so a rebuild cannot inherit a stale game list from the
    previous run's stored game_level_features.csv (stale on an old commit,
    absent on a fresh clone). Raises when the frame is missing, unreadable,
    or carries no decided game.
    """
    path = Path(pbp_path) if pbp_path is not None else None
    if path is None:
        for cand in (DATA_DELIVERY_DIR / "pitches.parquet",):
            if cand.exists():
                path = cand
                break
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"no pitch frame to enumerate decided games ({path})")
    frame = pd.read_parquet(path, columns=["game_pk", "game_date",
                                            "home_team", "away_team",
                                            "post_home_score",
                                            "post_away_score"])
    frame = frame.dropna(subset=["game_pk", "post_home_score",
                                 "post_away_score"])
    if not len(frame):
        raise ValueError(f"pitch frame {path} carries no decided games")
    frame = frame.copy()
    frame["game_pk"] = frame["game_pk"].astype(int)
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    frame["home_win"] = (pd.to_numeric(frame["post_home_score"],
                                        errors="coerce")
                           > pd.to_numeric(frame["post_away_score"],
                                           errors="coerce"))
    frame["home_team"] = frame["home_team"].map(normalize_team)
    frame["away_team"] = frame["away_team"].map(normalize_team)
    out = (frame.sort_values(["game_date", "game_pk"])
           .drop_duplicates(subset=["game_pk"], keep="first")
           .reset_index(drop=True))
    return out[["game_pk", "game_date", "home_team", "away_team",
                "home_win"]]


def backfill_pregame_snapshots(limit: int | None = 200,
                               csv_path: Path | None = None,
                               out_path: Path | None = None,
                               pause_sec: float = PAUSE_SEC,
                               full_rebuild: bool = False,
                               max_workers: int = 1,
                               snapshot_fields: str | None = None,
                               games: pd.DataFrame | None = None) -> int:
    """Training-side repair: official posted nines from archived snapshots.

    For decided games whose cache rows lack a PIT-valid pre-game nine, this
    replays each game's feed archive: latest snapshot timecode strictly
    before first pitch (its battingOrder IS the official posted nine as of
    that timestamp — verified 2026-10-24 on Sep-16 game 823655, where the
    15:54 UTC snapshot holds the true starters both sides while the mutable
    boxscore later degraded). Rows land with source='pregame', the
    snapshot's Preview state (passes the state gate), captured_utc = the
    feed-attested snapshot time, and feed_timecode for audit replay.

    ``limit=None`` removes the incremental repair bound and processes every
    eligible decided game. ``full_rebuild=True`` additionally reprocesses
    rows that already carry PIT-valid provenance, so ``MLB_FULL_REPULL``
    regenerates the complete corpus from the snapshot archive instead of
    merely filling gaps around it. Upgrade-only for existing unproven rows:
    qualifying snapshots overwrite legacy backfill-grade rows, while games
    without a qualifying snapshot stay grandfathered. Every unrepaired
    unproven game_pk is printed by reason, so the run log names—not merely
    counts—the remaining grandfathered games. Preserved cache metadata
    (starter hands and validation flags) survives repair. Never raises.
    """
    from results import fetch_game_start_times  # noqa: E402 (lazy: cycle-safe)
    try:
        if games is not None:
            # Caller-supplied frame (the full-repull path passes games
            # enumerated from the freshly re-pulled pitches, never the
            # previous run's stored CSV) — same decided contract.
            games = games.dropna(subset=["game_pk", "home_win"]).copy()
            games["game_pk"] = games["game_pk"].astype(int)
            games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
        else:
            csv_path = csv_path or (DATA_DELIVERY_DIR / "game_level_features.csv")
            games = pd.read_csv(csv_path, usecols=["game_pk", "game_date",
                                                   "home_win"])
            games = games.dropna(subset=["game_pk", "home_win"])
            games["game_pk"] = games["game_pk"].astype(int)
            games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
        out = out_path or (DATA_DELIVERY_DIR / "lineups.parquet")
        if not out.exists() and not full_rebuild:
            print("no lineups.parquet — nothing to repair")
            return 0
        cache = (_normalize(pd.read_parquet(out)) if out.exists()
                 else pd.DataFrame(columns=LINEUPS_COLUMNS))
        proven = set()
        try:
            proven = set(cache.loc[cache["complete_home"]
                                   & cache["complete_away"]
                                   & (cache["source"] == "pregame")
                                   & cache["pit_valid"],
                                   "game_pk"].astype(int))
        except Exception:  # noqa: BLE001 — an empty/malformed cache proves nothing
            proven = set()
        import time as _time
        _t0 = _time.time()
        if full_rebuild:
            todo = games.sort_values(["game_date", "game_pk"])
            print(f"snapshot backfill: full rebuild of {len(todo)} decided "
                  f"games from pre-pitch snapshots (workers={max(1, int(max_workers))})")
        else:
            todo = games[~games["game_pk"].isin(proven)]
            todo = todo.sort_values(["game_date", "game_pk"])
            if limit is not None:
                todo = todo.head(limit)
            if todo.empty:
                print("snapshot backfill: every decided game already has a "
                      "PIT-valid pre-game nine")
                return 0
            print(f"snapshot backfill: {len(todo)} decided games lack a "
                  f"PIT-valid nine")
        # First pitches in one bounded range call: the schedule helper
        # chunks long ranges itself, so full history costs a few requests
        # rather than one request per game date.
        first_pitch: dict[int, str] = {}
        days = sorted(todo["game_date"].unique())
        if days:
            try:
                first_pitch = fetch_game_start_times(days[0], days[-1])
            except Exception:  # noqa: BLE001 — affected games are skipped
                first_pitch = {}
        preserved: dict[int, dict] = {}
        try:
            preserved = {int(k): v for k, v in cache.set_index("game_pk")
                         [["home_starter_hand", "away_starter_hand",
                           "validated_home", "validated_away"]]
                         .to_dict("index").items()}
        except Exception:  # noqa: BLE001
            preserved = {}
        records = list(todo.itertuples())
        ordered: list[tuple[dict | None, str] | None] = [None] * len(records)
        workers = max(1, int(max_workers))
        if workers < 2:
            for i, record in enumerate(records):
                ordered[i] = _repair_game_from_pregame_snapshot(
                    record, first_pitch, preserved, pause_sec=pause_sec,
                    snapshot_fields=snapshot_fields)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {pool.submit(
                    _repair_game_from_pregame_snapshot,
                    record, first_pitch, preserved, pause_sec=pause_sec,
                    snapshot_fields=snapshot_fields): i
                    for i, record in enumerate(records)}
                import datetime as _dt
                done = 0
                for future in as_completed(pending):
                    ordered[pending[future]] = future.result()
                    done += 1
                    if done % 500 == 0 or done == len(records):
                        _now = _dt.datetime.now().strftime("%H:%M:%S")
                        print(f"snapshot backfill: {done}/{len(records)} "
                              f"games checked [{_now}]", flush=True)
        rows = []
        counts = {"no_first_pitch": 0, "no_snapshot": 0,
                  "fetch_failed": 0, "error": 0}
        unrepaired: dict[str, list[int]] = {reason: [] for reason in counts}
        for record, result in zip(records, ordered):
            row, reason = result or (None, "error")
            if row is not None:
                rows.append(row)
                continue
            if reason not in counts:
                reason = "error"
            counts[reason] += 1
            pk = int(record.game_pk)
            # A proven row that cannot be refreshed is retained as proven;
            # only unproven survivors are grandfathered and therefore listed.
            if pk not in proven:
                unrepaired[reason].append(pk)
        reported = "; ".join(f"{reason}={sorted(pks)}"
                              for reason, pks in unrepaired.items() if pks)
        if not rows:
            print(f"snapshot backfill: no qualifying pre-pitch snapshots "
                  f"— cache untouched")
            if reported:
                print(f"snapshot backfill: grandfathered game_pks by "
                      f"reason: {reported}")
            return 0
        n = upsert_lineups(pd.DataFrame(rows), out_path=out)
        try:
            _el = _time.time() - _t0
            _el_txt = f" ({_el:.0f}s)"
        except Exception:
            _el_txt = ""
        print(f"snapshot backfill: {n} official posted nines restored{_el_txt}; "
              f"{counts['no_snapshot']} games have no qualifying pre-pitch "
              f"snapshot; {counts['fetch_failed']} snapshot fetches failed; "
              f"{counts['error']} games errored; "
              f"{counts['no_first_pitch']} games lacked a first pitch")
        if reported:
            print(f"snapshot backfill: grandfathered game_pks by reason: "
                  f"{reported}")
        return n
    except Exception as e:  # noqa: BLE001 — non-fatal by contract
        print(f"  ⚠️  snapshot backfill failed ({e}); cache untouched")
        return 0


def first_trip_nines(pitches: pd.DataFrame) -> dict:
    """{(game_pk, side): [first 9 distinct chronological batters]}.

    The actual starters, read off the game's own first trip through the
    order: at_bat_number is globally monotone within a game_pk, so the
    first 9 distinct batters per side ARE the starting nine (a mid-trip
    injury sub is the only exception — it reads as a mismatch, which
    quarantines honestly rather than binding a wrong nine). Pure —
    testable without network. Side from inning_topbot (Top = away).
    """
    need = {"game_pk", "inning_topbot", "at_bat_number", "batter"}
    if pitches is None or pitches.empty or not need.issubset(pitches.columns):
        return {}
    p = pitches.copy()
    p["game_pk"] = pd.to_numeric(p["game_pk"], errors="coerce")
    p["at_bat_number"] = pd.to_numeric(p["at_bat_number"], errors="coerce")
    p["batter"] = pd.to_numeric(p["batter"], errors="coerce")
    p = p.dropna(subset=["game_pk", "at_bat_number", "batter"])
    p["game_pk"] = p["game_pk"].astype(int)
    p["batter"] = p["batter"].astype(int)
    p = p.sort_values(["game_pk", "at_bat_number"])
    out: dict = {}
    for (gpk, topbot), grp in p.groupby(
            ["game_pk", "inning_topbot"], sort=False):
        side = "away" if str(topbot).strip().lower().startswith("top") else "home"
        seen: list[int] = []
        for b in grp["batter"].tolist():
            if b not in seen:
                seen.append(int(b))
            if len(seen) == 9:
                break
        if seen:
            out[(int(gpk), side)] = seen
    return out


def validate_orders(home_order, away_order, first_trip: dict,
                    game_pk: int) -> tuple:
    """Per-side agreement of a stored nine vs the actual first trip.

    Set equality (order-insensitive): membership is the tier-1 question —
    a shuffled-but-complete nine still names tonight's nine. A game-time
    injury sub inside the first trip reads as a mismatch and quarantines
    that side honestly. Returns (validated_home, validated_away) tri-state:
    True = agrees, False = PROVEN mismatch (quarantines), None = cannot
    check (stored order missing or no first-trip coverage — stays NULL /
    unchecked, retried next run; absence of proof is not proof of wrong).
    """
    def _set(order):
        # len()-based: stored orders round-trip from parquet as numpy arrays,
        # whose truth value is ambiguous (plain `not order` raises).
        try:
            if order is None:
                return set()
            return {int(x) for x in list(order)}
        except (TypeError, ValueError):
            return set()
    out = []
    for side, order in (("home", home_order), ("away", away_order)):
        trip = first_trip.get((int(game_pk), side))
        oset, tset = _set(order), _set(trip)
        if not oset or not tset:
            out.append(None)
        else:
            out.append(oset == tset)
    return (out[0], out[1])


def load_pregame_orders(game_pks, path: Path | None = None) -> pd.DataFrame:
    """Pre-game orders for slate game_pks from lineups.parquet (pure read).

    THE slate order source, shared with training (features.py tier 1 reads
    the same cache file): the run-time pre-game capture
    fetch_scheduled_lineups wrote — never a serve-time live feed. A live
    read after first pitch serves post-pitch battingOrders as 'pre-game'
    (exactly the 2026-08-29 skew vector) and bypasses the validation
    quarantine; this read carries each row's provenance so the single
    trust rule (_tier1_eligible_lineups, inside add_lineup_delta_features)
    gates both paths identically. Unknown pks / absent cache return no
    rows (the delta merge then ships honest projected NULLs). Never raises.
    """
    try:
        p = path or (DATA_DELIVERY_DIR / "lineups.parquet")
        if not p.exists():
            return pd.DataFrame()
        want = {int(x) for x in (game_pks or []) if x is not None
                and not (isinstance(x, float) and pd.isna(x))}
        if not want:
            return pd.DataFrame()
        cache = _normalize(pd.read_parquet(p))
        return cache[cache["game_pk"].isin(want)].copy()
    except Exception:  # noqa: BLE001 — slate degrades to projected NULLs
        return pd.DataFrame()


def main(games: pd.DataFrame | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--pbp", type=Path, default=None)
    ap.add_argument("--actuals", action="store_true",
                    help="backfill boxscore actual nines for decided games "
                         "missing them, then exit (manual training-truth "
                         "run; the Kaggle run never passes this)")
    ap.add_argument("--actuals-full", action="store_true",
                    help="re-pull boxscore actuals for ALL decided games, "
                         "then exit (manual full rebuild)")
    args = ap.parse_args()

    if games is not None:
        # Caller-supplied frame (the full-repull path passes games enumerated
        # from the freshly re-pulled pitches, never the stored CSV).
        games = games.dropna(subset=["game_pk"]).copy()
        games["game_date"] = pd.to_datetime(games["game_date"])
        games["game_pk"] = games["game_pk"].astype(int)
        decided_pks = set(games.dropna(subset=["home_win"])["game_pk"]
                          .astype(int))
    else:
        csv_path = args.csv or (DATA_DELIVERY_DIR / "game_level_features.csv")
        games = pd.read_csv(csv_path, usecols=["game_pk", "game_date",
                                               "home_team", "away_team"])
        games["game_date"] = pd.to_datetime(games["game_date"])
        games = games.dropna(subset=["game_pk"])
        games["game_pk"] = games["game_pk"].astype(int)
        # Decided games only: home_win present.
        full = pd.read_csv(csv_path, usecols=["game_pk", "home_win"])
        decided_pks = set(full.dropna(subset=["home_win"])["game_pk"].astype(int))
    games = games[games["game_pk"].isin(decided_pks)].reset_index(drop=True)
    print(f"decided games to validate: {len(games)}")
    if args.actuals or args.actuals_full:
        backfill_actual_nines(
            games, limit=None if args.actuals_full else args.limit,
            full_rebuild=bool(args.actuals_full))
        return

    out_path = DATA_DELIVERY_DIR / "lineups.parquet"
    if not out_path.exists():
        print("no lineups.parquet — nothing to validate")
        return
    # AUDIT pass (2026-10-24): main() no longer writes ORDERS and NEVER
    # gates membership. A post-game feed read has no authority over a
    # pre-game capture (the Sep-16 NYY field disagrees with who started),
    # so fetching orders post-game can only repeat or corrupt. What
    # post-game CAN do honestly: (1) check stored orders against
    # first-trip batters and stamp validated_home/away as a late-scratch /
    # data-quality MONITOR (a disagreeing side still binds tier 1 — the
    # official posted nine is the bet-time truth, scratches included);
    # (2) void self-contradicting provenance — a 'pregame' claim from a
    # non-Preview feed state (applied inside _normalize on load, reported
    # and persisted here). Neither step constructs feature values from
    # post-game truth, so no lookahead leaks anywhere.
    pbp_path = args.pbp
    if pbp_path is None:
        for cand in (DATA_DELIVERY_DIR / "pitches.parquet",):
            if cand.exists():
                pbp_path = cand
                break
    if pbp_path is None or not Path(pbp_path).exists():
        print(f"no pitches cache ({pbp_path}) — validation deferred; "
              f"orders untouched")
        return
    trips = first_trip_nines(pd.read_parquet(pbp_path))
    print(f"first-trip nines for {len({k[0] for k in trips}) or 0} games")
    raw = pd.read_parquet(out_path)
    n_voided = 0
    if "source" in raw.columns and "state" in raw.columns:
        _void_mask = ((raw["source"] == "pregame")
                       & (raw["state"].astype(str).str.lower() != "preview"))
        n_voided = int(_void_mask.sum())
        if n_voided:
            print(f"provenance-contradiction downgrade: {n_voided} rows claim "
                  f"'pregame' from a non-Preview feed state — voided to "
                  f"backfill-grade (orders kept, pre-game claim dropped)")
    cache = _normalize(raw)
    need_val = cache[((cache["validated_home"].isna())
                       | (cache["validated_away"].isna()))
                      & cache["game_pk"].isin(decided_pks)]
    print(f"pending validation: {len(need_val)} rows")
    n_ok = n_bad = 0
    for r in need_val.head(args.limit).itertuples():
        vh, va = validate_orders(r.home_order, r.away_order, trips,
                                 int(r.game_pk))
        # Stamp only decisive sides: None (no trip coverage / missing order)
        # stays NULL and retries next run — never a sticky FALSE from an
        # inability to check.
        stamp = {}
        if vh is not None:
            stamp["validated_home"] = vh
        if va is not None:
            stamp["validated_away"] = va
        for col, val in stamp.items():
            cache.loc[cache["game_pk"] == int(r.game_pk), col] = val
        n_ok += int(vh is True) + int(va is True)
        n_bad += int(vh is False) + int(va is False)
    cache.to_parquet(out_path, index=False)
    n_rated = n_ok + n_bad
    rate = (100.0 * n_bad / n_rated) if n_rated else 0.0
    print(f"validated {n_ok} sides agree, {n_bad} sides disagree "
          f"({rate:.1f}% divergence — the late-scratch background monitor; "
          f"audit only, membership unaffected); "
          f"lineups.parquet now: {len(cache)} games")


if __name__ == "__main__":
    main()
