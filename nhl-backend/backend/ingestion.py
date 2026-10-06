"""NHL data ingestion — official NHL API pull with local parquet caching.

Primary source: the official NHL public web API (``api-web.nhle.com/v1``),
no auth. Endpoints used:

  /v1/score/{YYYY-MM-DD}            game list: ids, teams, scores, SOG, state
  /v1/gamecenter/{id}/boxscore      team + per-player boxscore (goalies incl.
                                    TOI/decision; skaters incl. SOG, PPG,
                                    faceoff%, hits, blocks, PIM, G/A)

Player-level family: MoneyPuck free skater game-by-game archives
(``moneypuck.com/data.htm``, non-commercial, credit required) feed the
player-pool pl_* features via ``load_moneypuck_player_games`` ->
``features._load_player_ratings``; when the archives are unavailable the
pool columns degrade to position priors per the documented missing-value
policy — never fabricated.

Cache directory: OUTSIDE the git tree (repo root's parent), ``.nhl_cache/``
— the NFL pattern. All frames carry normalized NHL-API column names; the
feature engine is the only place that interprets them.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

try:  # package-relative import when used as backend.ingestion
    from backend import config  # type: ignore
except ImportError:  # running as a top-level module
    import config

logger = logging.getLogger(__name__)

# Content-keyed repeat suppression (2026-10-06 NHL log review; MLB's
# _log_resolved_view idiom): the Phase-3 decided-pool build and the
# Phase-11 slate build call these loaders over the same caches, so two of
# these lines printed TWICE per run, byte-identical both times. Keyed on
# the RENDERED message: first occurrence logs at its level, an identical
# repeat drops to DEBUG, changed content logs as a distinct line again.
_LOG_ONCE_SEEN: set[str] = set()


def _log_once(level: int, msg: str, *args) -> None:
    text = msg % args if args else str(msg)
    if text in _LOG_ONCE_SEEN:
        logger.debug("%s", text)
        return
    _LOG_ONCE_SEEN.add(text)
    logger.log(level, text)

# Cache directory: OUTSIDE the git tree (repo root's parent) so caches never
# pollute the working tree; overridable for tests.
CACHE_DIR = Path(config.ROOT_DIR.parent) / ".nhl_cache"

NHL_API_BASE = "https://api-web.nhle.com/v1"
# MoneyPuck public downloads. NOTE: the previously-wired
# ``moneyPuck/playerDataByGame/shotsAllYears.csv`` URL now 404s (verified
# 2026-09-26). Live player and team data links are listed on moneypuck.com/data.htm.
# MoneyPuck permits these downloads for non-commercial use with attribution;
# do not scrape paths that are not published on that page.
MONEYPUCK_BASE = "https://moneypuck.com/moneypuck"
# Per PLAYER x GAME x SITUATION regular-season data. The game-by-game ZIPs are
# linked in MoneyPuck's official "Game By Game Level Data" table. They contain
# individual ``I_F_*`` metrics, including xGoals, and ``icetime`` in seconds.
# The ZIP's historical directory name says seasonPlayersSummary, but the CSV
# has gameId/gameDate rows (verified against the published 2024 archive).
MONEYPUCK_PLAYER_GAMES_ZIP_URL = (
    "https://peter-tanner.com/moneypuck/downloads/"
    "seasonPlayersSummary/skaters/{season}.zip")
MONEYPUCK_PLAYER_GAMES_HISTORY_URL = (
    "https://peter-tanner.com/moneypuck/downloads/"
    "seasonPlayersSummary/skaters/2008_to_2024.zip")
MONEYPUCK_PLAYER_GAMES_HISTORY_SEASONS = tuple(range(2008, 2025))
# ESPN publishes a dated, per-team NHL injury report. There is NO official NHL
# injury endpoint (verified 2026-09-26: /v1/injury, /v1/injuries and
# /v1/injury-report all 404), so this is the third-party feed of record.
ESPN_NHL_INJURIES_URL = ("https://site.api.espn.com/apis/site/v2/sports/hockey/"
                         "nhl/injuries")
# ESPN's edge answers 403 to the shared product user-agent, so this family needs
# a browser agent. Counter-intuitively the SHORT agent is the one that works:
# verified 2026-09-26 against this exact URL, "Mozilla/5.0" -> 200, a full
# Chrome UA string ("...Chrome/124.0 Safari/537.36") -> 403, and the product
# agent -> 403. Do not "improve" this to a fuller browser string. The edge
# also blocks in WAVES: on 2026-09-28 a Kaggle run's single 403 with these
# exact headers coincided with probes of the same profile testing 200 both
# minutes before and minutes after — so a 403 here is treated as transient
# (retried), not as a permanent endpoint ban.
#
# 2026-09-29 15:01 update: the wave can outlast any retry schedule — both
# runs of that day lost the fetch for ~9 minutes each (every attempt of all
# 4 retries 403'd, both phases), while the same identity tested 200 minutes
# later. Verified 2026-09-29 ~15:30: the browser agent AND the library-default
# identity (no User-Agent header at all — requests drops a None value) both
# test 200, while the product agent and a full Chrome string both 403. So the
# fetch falls over ONCE to the library-default identity before giving up:
# a second, structurally different request profile that the wave has not
# necessarily blocked.
ESPN_USER_AGENT = "Mozilla/5.0"
ESPN_HEADERS = {
    "User-Agent": ESPN_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
}
# A None-valued header is DROPPED by requests (verified 2026-09-29): the
# request goes out with no User-Agent line at all — the library-default
# identity — which is a different profile from any explicit agent string.
ESPN_IDENTITY_HEADERS = {
    "User-Agent": None,
    "Accept": "application/json, text/plain, */*",
}

# Per-game keep-list from /v1/score/{date} games[] — the schedule/board
# population. Scores stay None for unplayed games (honest pre-game rows).
SCORE_KEEP = [
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score",
    "home_sog", "away_sog", "venue", "game_state", "game_outcome",
]

# A score is only a RESULT once the game has left its in-flight states.
# /v1/score and gamecenter payloads carry RUNNING scores for LIVE/CRIT games
# — the 2026-09-29 slate shipped TOR@MTL 1-1 and BOS@NYR 0-0 mid-game, and a
# 0-0 running score parses as a *decided* 0.0/0.0 tie that no null-filter can
# catch. Every admission path keys off this set: non-final rows keep their
# schedule row (the slate must keep serving them) but lose every score-like
# field, and neither their score page nor their boxscore is ever cached
# mid-game.
FINAL_GAME_STATES = frozenset({"OFF", "FINAL"})

# Cache schema versions: bump whenever the keep-list widens so stale caches
# are ignored rather than silently serving the old column set.
# v2 adds start_time_utc, which is required to compare local report capture
# timestamps to the actual puck-drop decision in the injury pool.
SCORE_CACHE_VERSION = "v2"
# v4: starter selection now keys on the API's ``starter`` boolean (v3 keyed on
# a ``decision`` set that omitted the overtime-loss code "O", so those games
# resolved to the 00:00 scratch goalie), and ``powerPlayShotsAgainst="0/0"``
# is now recorded as the measured 0 it is rather than a null.
BOXSCORE_CACHE_VERSION = "v4"
# Cache schema version for the player-game archive family (pl_* ratings).
MP_PLAYER_GAME_VERSION = "v2"
MP_PLAYER_GAME_CHUNK_SIZE = 100_000
INJURY_VERSION = "v2"
# A same-day cache may still be obsolete by puck drop. Refresh within a run at
# this cadence rather than equating "same UTC date" with a current report.
INJURY_CACHE_TTL_HOURS = 6

# MoneyPuck ``situation`` values. 5on5 = even strength, 5on4 = power play.
MONEYPUCK_SITUATIONS = ("5on5", "5on4")

# Structural mirror of MLB's ``_chunked_statcast``: the pull is grouped into
# fixed calendar windows so a multi-thousand-game pull reports progress per
# window instead of running as one silent loop, and so MLB's ``pause_sec``
# knob has somewhere to act between windows.
#
# This is PRESENTATION ONLY. The same games are requested, the returned frame
# is re-ordered to the caller's input order, and no production result depends
# on where the window boundaries fall. The pause defaults to 0.0 so wall-clock
# is unchanged; raising it is the lever if the API ever rate-limits.
PULL_CHUNK_DAYS = 60
PULL_CHUNK_PAUSE_SEC = 0.0
# A score page is treated as settled once its date is older than this many
# days. Posting lags game completion by hours, so a null score inside the
# window is provisional; past it the game was cancelled or postponed and the
# null is permanent. Without this, one such game makes its page uncacheable
# forever and every run re-pulls it (the 2024-10-07 page, game 2024010044,
# has sat in gameState=FUT since the shortened 2024-25 season).
SETTLE_GRACE_DAYS = 2


def _progress_bar(total: int, desc: str):
    """A tqdm bar whenever tqdm is importable, else None.

    Drawn UNCONDITIONALLY, not only on a terminal — that gate is what made
    the bar a terminal-only feature and left the Kaggle run with none, since
    the notebook drives the pipeline through ``subprocess`` and stderr is
    therefore a pipe. MLB already has the behaviour to copy: the bars in its
    run come from ``pybaseball.statcast()``, which wraps each chunk fetch in
    its own tqdm and emits regardless of TTY, and they render in the notebook
    output. A ``\\r``-repainting bar is not a stream of unreadable snapshots —
    the notebook redraws it, and a redirected log keeps one readable line per
    refresh. Hiding the bar is what made the pull invisible.

    ``leave=True`` so the completed 100% bar survives, the way MLB's does, and
    ``ascii`` left at tqdm's own default so it picks block glyphs under UTF-8
    and degrades to ASCII on a cp1252 console by itself.

    Total best-effort by design: it writes to stderr only, never raises (a
    missing, broken, or absent tqdm must not fail a run), and the per-window
    log lines remain the durable record of progress either way.
    """
    # NB: the bare `except` below will swallow a NameError here, so anything
    # this function touches must be imported at module scope. It did once:
    # `sys` was only imported inside the old body, and dropping that import
    # turned the bar into a silent None instead of an error.

    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="game", leave=True,
                    dynamic_ncols=True, file=sys.stderr)
    except Exception:  # noqa: BLE001 — decoration must never break ingestion
        return None


BAR_WIDTH = 20


def _bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """A fixed-width ASCII bar — the one progress affordance that survives
    a captured log.

    A tqdm bar repaints in place with ``\r``, which a file-backed sink records
    as a stream of partial repaints rather than as a bar: the Kaggle run is a
    ``subprocess`` whose stderr is a pipe, so an in-place bar is both
    unavailable (no TTY) and unreadable if forced on. These are ordinary
    printable characters, so the same line reads as a bar filling up in the
    notebook output pane, in a CI log, and on a terminal.

    ASCII only on purpose: the log is written by ``logging`` under whatever
    encoding the host console has, and a box-drawing glyph raises
    ``UnicodeEncodeError`` on a cp1252 console — turning a progress
    decoration into a failed run.
    """
    width = max(1, int(width))
    frac = min(1.0, max(0.0, float(fraction)))
    # Round half UP, not Python's banker's rounding: a bar that shows nothing
    # at exactly half is a bar that looks broken.
    filled = int(frac * width + 0.5)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class _PullProgress:
    """Durable, terminal-independent progress for a multi-minute pull.

    A tqdm bar is the right affordance and this emits one plain log line per
    window beside it, so BOTH are present in every context: the bar for the
    operator watching a cell scroll, the line for the durable record. The bar
    used to be suppressed whenever stderr was not a terminal, which is every
    captured run — the 2026-09-26 boxscore phase spent three minutes per
    chunk emitting nothing at all, so a run that was working and a run that
    had hung looked identical. Where a real bar is drawing, the log line
    drops its own inline bar so the position is not rendered twice.

    The window closes on whichever comes first: ``every`` items, or
    ``interval`` seconds. A cadence counted only in items goes quiet exactly
    when it matters — a rate-limited or stalling API is a SLOW rate, so a
    count-triggered line can be minutes wide. The seconds floor bounds the
    worst gap no matter how slow the pull gets.

    Every line carries an inline ``_bar`` only when no tqdm bar is drawing,
    so a run with neither a terminal nor tqdm still reads as progress.
    """

    def __init__(self, total: int, desc: str, every: int = 50,
                 interval: float = 30.0) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.every = max(1, int(every))
        self.interval = max(0.0, float(interval))
        self.done = 0
        self.hits = 0
        self.fetched = 0
        self.failed = 0
        self._start = time.monotonic()
        self._last = self._start
        self._reported = False
        self._bar = _progress_bar(self.total, self.desc)
        # Latched at construction, not read live: `close()` clears `self._bar`
        # before it logs the window's result, and the closing line must still
        # know whether a bar DREW for this window.
        self._drew_bar = self._bar is not None

    @staticmethod
    def _rate(done: int, elapsed: float) -> float:
        return done / elapsed if elapsed > 0 and done > 0 else 0.0

    def _counts(self) -> str:
        """`N cached, M fetched` — plus unavailable pages when there are any.

        The three outcomes are kept apart on purpose. An unavailable page
        advances the loop but returns nothing, so counting it as a fetch
        would make the progress line disagree with the window's own
        `resolved: ... cache hits, ... fetched` summary by exactly the
        number of pages that failed.
        """
        parts = [f"{self.hits} cached", f"{self.fetched} fetched"]
        if self.failed:
            parts.append(f"{self.failed} unavailable")
        return ", ".join(parts)

    def _bar_at(self, done: int) -> str:
        """The bar for this position, or empty when a real bar is drawing.

        Two bars for one position is noise, not emphasis: the tqdm bar is
        live and moving, so the line carries only the numbers it cannot show.
        With no tqdm the line IS the bar, and keeps its own. A zero-item
        window reads as complete rather than dividing by zero — nothing was
        asked for, so nothing is outstanding.
        """
        if self._drew_bar:
            return ""
        return _bar(done / self.total if self.total else 1.0)

    def _line(self, done: int, now: float) -> str:
        elapsed = max(0.0, now - self._start)
        rate = self._rate(done, elapsed)
        bar = self._bar_at(done)
        head = (f"{self.desc} {done}/{self.total}{' ' + bar if bar else ''} "
                f"({self._counts()}) {elapsed:.1f}s {rate:.2f}/s")
        if rate > 0 and done < self.total:
            return f"{head} eta {(self.total - done) / rate:.0f}s"
        return head

    def tick(self, *, cached: bool = False, failed: bool = False) -> None:
        """Record one finished item and log a line if the window has closed.

        Exactly one outcome per call: a cache hit (``cached=True``), a page
        the API would not serve (``failed=True``), or — the default — a
        completed fetch. Every exit from the pull loops routes through here,
        so the position is live no matter which path ran.
        """
        self.done += 1
        if failed:
            self.failed += 1
        elif cached:
            self.hits += 1
        else:
            self.fetched += 1
        if self._bar is not None:
            try:
                self._bar.update(1)
                self._bar.set_postfix(cached=self.hits, fetched=self.fetched)
            except Exception:  # noqa: BLE001 — decoration only
                self._bar = None
        now = time.monotonic()
        final = self.done >= self.total
        if (final or self.done % self.every == 0
                or (now - self._last) >= self.interval):
            logger.info("%s", self._line(self.done, now))
            self._last = now
            self._reported = self._reported or final

    def close(self) -> None:
        """Close the bar (if any) and log the window's result once."""
        if self._bar is not None:
            try:
                self._bar.close()
            except Exception:  # noqa: BLE001 — decoration only
                pass
            self._bar = None
        if not self.total or not self.done or self._reported:
            return
        elapsed = max(0.0, time.monotonic() - self._start)
        bar = self._bar_at(self.done)
        logger.info("%s%s done %d/%d (%s) in %.1fs (%.2f/s)",
                    self.desc, f" {bar}" if bar else "", self.done, self.total,
                    self._counts(), elapsed,
                    self._rate(self.done, elapsed))
        # close() is idempotent: the bar's is, and a second call (a re-entered
        # loop, an already-torn-down bar) must not restate the result.
        self._reported = True


def _chunk_games(game_ids: list[str], gameday_by_id: dict | None,
                 chunk_days: int = PULL_CHUNK_DAYS) -> list[tuple[str, list[str]]]:
    """Group game ids into ascending ``chunk_days`` calendar windows.

    Returns ``[(label, [game_id, ...]), ...]``. Every input id lands in exactly
    one chunk: ids with no known gameday go to a trailing ``undated`` chunk
    rather than being dropped, because dropping a game here would silently
    remove it from the feature frame. With no mapping (or a non-positive
    ``chunk_days``) the ids stay in a single chunk, so an unchunked call
    fetches exactly what it always did.
    """
    if not game_ids:
        return []
    if not gameday_by_id or chunk_days <= 0:
        return [("all", list(game_ids))]

    dated: list[tuple[pd.Timestamp, str]] = []
    for gid in game_ids:
        raw = gameday_by_id.get(gid)
        ts = pd.to_datetime(raw, errors="coerce") if raw is not None else None
        if ts is not None and not pd.isna(ts):
            dated.append((pd.Timestamp(ts).normalize(), gid))
    if not dated:
        return [("all", list(game_ids))]

    lo = min(ts for ts, _ in dated)
    hi = max(ts for ts, _ in dated)
    chunks: list[tuple[str, list[str]]] = []
    placed: set[str] = set()
    cursor = lo
    while cursor <= hi:
        end = cursor + pd.Timedelta(days=chunk_days - 1)
        in_chunk = [g for ts, g in dated if cursor <= ts <= end]
        if in_chunk:
            chunks.append((f"{cursor.date()}..{end.date()}", in_chunk))
            placed.update(in_chunk)
        cursor = end + pd.Timedelta(days=1)
    missing = [g for g in game_ids if g not in placed]
    if missing:
        chunks.append(("undated", missing))
    return chunks


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def clear_cache() -> None:
    """Remove all cached NHL parquet artifacts."""
    if CACHE_DIR.exists():
        for path in CACHE_DIR.glob("*.parquet"):
            path.unlink(missing_ok=True)


def _http_json(url: str, retries: int = 3, timeout: float = 30.0,
               headers: dict | None = None,
               retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)):
    """GET a JSON document with retry/backoff (the official API is free but
    rate-limited; a transient 5xx must not fail a run).

    ``headers`` overrides the default product user-agent (third-party edges
    such as ESPN's accept and 403 specific agents in passing waves; the
    caller owns that choice). ``retry_statuses`` extends the transient set
    worth a bounded retry — include 403 there only for an edge whose blocks
    are known to clear between attempts, never for a permanently forbidden
    endpoint.

    ``timeout`` is the READ budget; the connect/TLS handshake gets a shorter
    one. A single float applies to each socket operation, so a host that
    accepts the TCP connection and then stalls mid-handshake otherwise costs
    the full read budget on every attempt — 3 x 30s for one unusable URL.
    """
    import requests
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=(min(10.0, timeout), timeout),
                                headers=headers or {"User-Agent": "sports-prediction-model/1.0"})
            if resp.status_code in retry_statuses and attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"NHL API GET failed after {retries} attempts: {url} ({last_exc})")


# ---------------------------------------------------------------------------
# Schedule / score ingestion — per-day score pages, cached per season
# ---------------------------------------------------------------------------

def _parse_score_game(g: dict) -> dict:
    """One /v1/score games[] entry -> a flat schedule row."""
    home = g.get("homeTeam") or {}
    away = g.get("awayTeam") or {}
    outcome = ((g.get("gameOutcome") or {}).get("lastPeriodType", "") or "")
    return {
        "game_id": str(g.get("id", "") or ""),
        "season": int(g.get("season", 0) or 0) // 10000,  # 20242025 -> 2024
        "game_type": int(g.get("gameType", config.GAME_TYPE_REG) or 0),
        "game_date": str(g.get("gameDate", "") or ""),
        "start_time_utc": str(g.get("startTimeUTC", "") or ""),
        "home_team": str(home.get("abbrev", "") or ""),
        "away_team": str(away.get("abbrev", "") or ""),
        "home_score": g.get("homeTeam", {}).get("score"),
        "away_score": g.get("awayTeam", {}).get("score"),
        "home_sog": g.get("homeTeam", {}).get("sog"),
        "away_sog": g.get("awayTeam", {}).get("sog"),
        "venue": str((g.get("venue") or {}).get("default", "") or ""),
        "game_state": str(g.get("gameState", "") or ""),
        "game_outcome": outcome,
    }


def _date_windows(dates: list[str], chunk_days: int) -> list[tuple[str, list[str]]]:
    """Group an ascending date list into ``chunk_days`` calendar windows."""
    if not dates:
        return []
    if chunk_days <= 0:
        return [("all", list(dates))]
    out: list[tuple[str, list[str]]] = []
    ordered = sorted(dates)
    cursor = pd.Timestamp(ordered[0]).normalize()
    last = pd.Timestamp(ordered[-1]).normalize()
    while cursor <= last:
        end = cursor + pd.Timedelta(days=chunk_days - 1)
        in_win = [d for d in ordered
                  if cursor <= pd.Timestamp(d).normalize() <= end]
        if in_win:
            out.append((f"{cursor.date()}..{end.date()}", in_win))
        cursor = end + pd.Timedelta(days=1)
    return out


def load_score_dates(dates: list[str], use_cache: bool = True) -> pd.DataFrame:
    """Fetch /v1/score/{date} for each date and return the combined rows.

    Per-date parquet caches keyed by the date (an incremental daily pull is
    the NHL's natural unit — unlike league sports there is no per-season
    schedule endpoint). A failed date is warned and skipped, never fatal.

    A page is cached once it can no longer change:

      * a page where every game posted a score is settled immediately.
      * a page with an UNPLAYED game (null score) is provisional while the
        date is inside the posting-lag grace window — those scores fill in
        within hours, and the pipeline filters on home_score.notna(), so
        caching the null early would make the game permanently invisible.
      * once the date is older than SETTLE_GRACE_DAYS the null is PERMANENT:
        the game was cancelled or postponed and will never be played. Cache it
        so the page stops being re-pulled on every run forever, and account
        for the game by id so a shrunken OOF population is visible.
    """
    frames: list[pd.DataFrame] = []
    today = date.today()
    fetched = hits = 0
    no_result: list[tuple[str, str]] = []
    windows = _date_windows(dates, PULL_CHUNK_DAYS)
    for w, (wlabel, wdates) in enumerate(windows, 1):
        logger.info("score chunk %d/%d [%s]: %d dates",
                    w, len(windows), wlabel, len(wdates))
        prog = _PullProgress(
            len(wdates), f"score chunk {w}/{len(windows)} [{wlabel}]")
        for i, d in enumerate(wdates, 1):
            path = _cache_path(f"score_{SCORE_CACHE_VERSION}_{d.replace('-', '')}.parquet")
            if use_cache and path.exists():
                try:
                    frames.append(_coerce_score_dtypes(pd.read_parquet(path)))
                    hits += 1
                    prog.tick(cached=True)
                    continue
                except Exception as exc:  # corrupt cache → re-pull
                    logger.warning("score cache %s unreadable (%s)", path.name, exc)
            try:
                payload = _http_json(f"{NHL_API_BASE}/score/{d}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("score page unavailable for %s: %s", d, exc)
                prog.tick(failed=True)
                continue
            fetched += 1
            prog.tick(cached=False)
            rows = [_parse_score_game(g) for g in (payload.get("games") or [])]
            df = pd.DataFrame(rows, columns=SCORE_KEEP)
            # Numeric contract BEFORE the concat below (why lives on the
            # helper): every page — fetched OR cached — must carry float64
            # score columns, or pandas 2.x warns on the all-NA entry a
            # cancelled-date page produces.
            df = _coerce_score_dtypes(df)
            # RUNNING scores are not results: null them at the door so a
            # mid-game slate can never enter the decided population.
            pending = ~df["game_state"].isin(FINAL_GAME_STATES)
            if pending.any():
                df.loc[pending, ["home_score", "away_score",
                                 "home_sog", "away_sog"]] = None
            unplayed = (df[df["home_score"].isna()] if len(df) else df)
            n_unplayed = int(len(unplayed))
            game_day = date.fromisoformat(d)
            # Past the posting-lag window a null score is permanent, so the
            # page is cacheable and the game is accounted for, not re-pulled.
            past_grace = game_day < today - timedelta(days=SETTLE_GRACE_DAYS)
            if n_unplayed == 0 or past_grace:
                df.to_parquet(path, index=False)
                if n_unplayed:
                    for gid in unplayed.get("game_id", pd.Series(dtype=str)):
                        no_result.append((d, str(gid)))
                    logger.info("score page %s: %d game(s) never produced a "
                                "result (cancelled/postponed) — page cached, "
                                "games excluded from the decided population",
                                d, n_unplayed)
            else:
                logger.info("score page %s carries %d unplayed or in-progress "
                            "game(s) — not cached, within the %d-day posting lag",
                            d, n_unplayed, SETTLE_GRACE_DAYS)
            frames.append(df)
            # Progress, so a slow or rate-limited pull is VISIBLE. Without it
            # the loop between the Phase 2 banner and its result line emits
            # nothing, and a multi-minute network stall is indistinguishable
            # from a hang. _PullProgress owns that line, on an item AND a
            # seconds cadence, and it advances on EVERY exit above — the old
            # counter was skipped by the cache and unavailable-date paths, so
            # a window that never fetched anything reported no progress at
            # all.
        prog.close()
    logger.info("score dates resolved: %d requested in %d chunk(s), "
                "%d cache hits, %d fetched",
                len(dates), len(windows), hits, fetched)
    if no_result:
        logger.info("games with no result (excluded from the decided "
                    "population): %d — %s", len(no_result),
                    ", ".join(f"{gid}@{day}" for day, gid in no_result[:10]))
    # Empty frames are dropped before concat: an empty/all-NA entry only
    # degrades the result dtypes (the pandas FutureWarning this exact call
    # raised in every full run) and contributes no rows.
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=SCORE_KEEP)
    return pd.concat(frames, ignore_index=True)


def season_dates(season: int) -> list[str]:
    """The calendar date span that can hold a season's games.

    The NHL regular season runs Oct..mid-April and the playoffs into June;
    a generous Oct 1 .. Jul 15 span covers both. For an unstarted season the
    tail is entirely future dates, which carry the published but UNPLAYED
    schedule (null scores) and so can never reach the OOF population.
    Callers should clip this span to their own window end rather than pay a
    round trip per future day for rows they will discard.
    """
    start = date(season, 10, 1)
    end = date(season + 1, 7, 15)
    out: list[str] = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def eligible_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Filter a schedule frame to the eligible population: settled
    regular-season or postseason games within the configured season window
    (2024+), with gameType in config.GAME_TYPES.

    Settled means the score is FINAL: rows still in an in-flight
    ``game_state`` (LIVE/CRIT/...) keep their schedule row — the serving
    slate is built from these survivors — but their running scores are
    nulled, because master_pipeline derives ``decided_all`` from exactly
    this frame and a running 1-1 is not a result. The re-null here covers
    rows that did not come from today's ``load_score_dates`` (fixtures,
    caches written by pre-guard versions).
    """
    df = schedule.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"].isin(config.ALL_SEASONS)]
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(config.GAME_TYPES)]
    for c in ("home_score", "away_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "game_state" in df.columns:
        pending = ~df["game_state"].isin(FINAL_GAME_STATES)
        if pending.any():
            df.loc[pending, ["home_score", "away_score"]] = np.nan
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Boxscore ingestion — per-game team + goalie + skater stats
# ---------------------------------------------------------------------------

def _parse_toi_minutes(value) -> float:
    """Parse an NHL API goalie TOI value into minutes.

    The boxscore endpoint commonly returns a clock string such as ``"25:00"``
    or ``"1:02:30"``; older payloads may provide a numeric minute value. Keep
    malformed values as NaN so downstream features degrade honestly.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    raw = str(value).strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        pass
    parts = raw.split(":")
    if len(parts) not in (2, 3):
        return float("nan")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return float("nan")
    if any(number < 0 for number in numbers):
        return float("nan")
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes + seconds / 60.0
    hours, minutes, seconds = numbers
    return hours * 60.0 + minutes + seconds / 60.0


def _or_none(value: float):
    """NaN -> None so a missing boxscore metric round-trips as a real null."""
    return None if value is None or value != value else float(value)


def _parse_ratio(value) -> float:
    """Parse an NHL API ``"made/attempts"`` ratio string into its denominator.

    The boxscore endpoint reports power-play volume on the GOALIE lines as
    ``powerPlayShotsAgainst`` (e.g. ``"5/6"``); the team's own power-play
    opportunities for that game are exactly the opposing goalie's
    power-play shots faced.

    ``"0/0"`` is a REAL observation, not a missing one: it means that goalie
    faced no power-play shots, and the sibling fields corroborate it
    (``evenStrengthShotsAgainst`` + ``shorthandedShotsAgainst`` +
    ``powerPlayShotsAgainst`` == ``shotsAgainst``). Returning NaN for it
    reported a measured zero as an absent measurement, which is what pushed
    ``pp_success_diff`` to 0% coverage on the feature report whenever a
    team's recent games happened to contain one of those. It is therefore a
    genuine 0.0 here; the per-game RATE stays undefined downstream (you
    cannot convert zero chances), and the trailing window pools the counts
    so a zero-opportunity game no longer punches a hole in the feature.

    Genuinely malformed values still stay NaN rather than fabricating a
    denominator: a missing slash, a missing half, a negative sentinel, or
    ``made > attempts``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    raw = str(value).strip()
    made, sep, denom = raw.partition("/")
    if not sep or not made.strip() or not denom.strip():
        return float("nan")
    try:
        made_n = float(made.strip())
        attempts = float(denom.strip())
    except ValueError:
        return float("nan")
    if not (np.isfinite(made_n) and np.isfinite(attempts)):
        return float("nan")
    # Negative halves are the API's "did not record" sentinel on some lines.
    if made_n < 0 or attempts < 0 or made_n > attempts:
        return float("nan")
    return attempts


def _parse_boxscore(bs: dict) -> dict:
    """One /v1/gamecenter/{id}/boxscore -> per-game team rollup row.

    Team-level: goals + SOG. Goalie rollup: the two DECISION goalies' TOI
    and the season aggregate stats live on the goalie lines; the per-team
    skater block rolls up PPG, faceoff%, hits, blocks, PIM, giveaways and
    takeaways (means/sums of that game only — the trailing shift downstream
    keeps them strictly-prior).

    Power-play OPPORTUNITIES are not a skater field; they are recovered from
    the opposing goalie's ``powerPlayShotsAgainst`` denominator, so a team's
    per-game PP rate is ``pp_goals / pp_opportunities`` (the manifest's
    definition). ``pp_opportunities`` is cross-filled after both sides are
    parsed because it needs the OPPONENT's goalie line.
    """
    player_stats = bs.get("playerByGameStats") or {}
    out: dict = {"game_id": str(bs.get("id", "") or "")}
    pp_shots_faced: dict[str, float] = {}
    for side, team_key in (("home", "homeTeam"), ("away", "awayTeam")):
        block = player_stats.get(team_key) or {}
        team = bs.get(team_key) or {}
        sog = team.get("sog")
        skaters = list(block.get("forwards") or []) + list(block.get("defense") or [])
        goalies = list(block.get("goalies") or [])

        def _num(rows: list[dict], field: str) -> list[float]:
            vals = []
            for r in rows:
                try:
                    v = float(r.get(field))
                except (TypeError, ValueError):
                    continue
                if v == v:  # not NaN
                    vals.append(v)
            return vals

        def _sum(rows: list[dict], field: str) -> float:
            vals = _num(rows, field)
            return float(sum(vals)) if vals else float("nan")

        # Faceoff win pct is a rate: mean over skaters with a value.
        fo_vals = _num(skaters, "faceoffWinningPctg")
        pp_goals = _sum(skaters, "powerPlayGoals")
        # Goalie blocks: the official goalie line supplies per-goalie
        # goalsAgainst, shotsAgainst, and TOI. Use those fields directly;
        # team score/SOG are not a safe proxy for a relief appearance.
        #
        # The starter is the goalie carrying the API's explicit
        # ``starter: True`` boolean. Selecting on ``decision`` alone was wrong
        # twice over: the overtime-loss code is ``"O"``, not ``"OTL"``, so
        # those games matched nothing and fell through to ``goalies[0]`` —
        # which is the scratch goalie at 00:00 TOI. That silently zeroed the
        # starter's TOI, the goalie failed the MIN_GOALIE_TOI_MINUTES start
        # test, and the team was recorded as having NO start that game, which
        # propagated into null goalie features for every later game.
        _flagged = [g for g in goalies if g.get("starter") is True]
        if _flagged:
            starter = _flagged[0]
        else:
            # No starter flag: fall back to the most ice time, which is the
            # definition of a starter, then to a decision goalie.
            def _toi_key(g):
                t = _parse_toi_minutes(g.get("toi"))
                return -1.0 if not np.isfinite(t) else t
            _by_toi = sorted(goalies, key=_toi_key, reverse=True) if goalies else []
            _dec = [g for g in _by_toi
                    if g.get("decision") in ("W", "L", "O", "OTL", "SOL")]
            starter = _dec[0] if _dec else (_by_toi[0] if _by_toi else {})
        toi = _parse_toi_minutes(starter.get("toi"))
        if not np.isfinite(toi):
            toi = None
        out[f"{side}_sog"] = sog
        out[f"{side}_pp_goals"] = pp_goals if pp_goals == pp_goals else None
        out[f"{side}_faceoff_pct"] = (sum(fo_vals) / len(fo_vals)) if fo_vals else None
        out[f"{side}_hits"] = _sum(skaters, "hits") if skaters else None
        out[f"{side}_blocked"] = _sum(skaters, "blockedShots") if skaters else None
        out[f"{side}_pim"] = _sum(skaters, "pim") if skaters else None
        out[f"{side}_giveaways"] = _sum(skaters, "giveaways") if skaters else None
        out[f"{side}_takeaways"] = _sum(skaters, "takeaways") if skaters else None
        out[f"{side}_goalie_id"] = starter.get("playerId")
        out[f"{side}_goalie_name"] = str(
            ((starter.get("name") or {}).get("default", "")) or "")
        out[f"{side}_goalie_toi"] = toi
        out[f"{side}_goalie_decision"] = starter.get("decision")
        # Per-goalie boxscore fields are authoritative for save% and GAA.
        # The previous team-score fallback was unsafe for relief goalies.
        def _goalie_num(field: str):
            try:
                value = float(starter.get(field))
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        out[f"{side}_goals_against"] = _goalie_num("goalsAgainst")
        out[f"{side}_shots_against"] = _goalie_num("shotsAgainst")
        # Power-play shots THIS goalie faced = the opponent team's PP volume.
        pp_shots_faced[side] = _parse_ratio(starter.get("powerPlayShotsAgainst"))
    # A team's PP opportunities = the opposing goalie's PP shots faced.
    out["home_pp_opportunities"] = _or_none(pp_shots_faced.get("away"))
    out["away_pp_opportunities"] = _or_none(pp_shots_faced.get("home"))
    return out


BOXSCORE_COLS = [
    "game_id",
    "home_sog", "away_sog", "home_pp_goals", "away_pp_goals",
    "home_faceoff_pct", "away_faceoff_pct", "home_hits", "away_hits",
    "home_blocked", "away_blocked", "home_pim", "away_pim",
    "home_giveaways", "away_giveaways", "home_takeaways", "away_takeaways",
    "home_pp_opportunities", "away_pp_opportunities",
    "home_goalie_id", "away_goalie_id", "home_goalie_name", "away_goalie_name",
    "home_goalie_toi", "away_goalie_toi",
    "home_goalie_decision", "away_goalie_decision",
    "home_goals_against", "away_goals_against",
    "home_shots_against", "away_shots_against",
]


SKATER_COLS = ["game_id", "side", "team", "player_id", "player_name"]

#: The per-game stats that are NUMBERS by contract. Coercing exactly these
#: (and never the goalie id/name/decision identity columns) at parse time
#: keeps every cached parquet float64 even when a game recorded no stats —
#: the dtype-stability that silences the pandas-2.x all-NA concat warning.
BOXSCORE_NUMERIC_COLS = (
    "home_sog", "away_sog", "home_pp_goals", "away_pp_goals",
    "home_faceoff_pct", "away_faceoff_pct", "home_hits", "away_hits",
    "home_blocked", "away_blocked", "home_pim", "away_pim",
    "home_giveaways", "away_giveaways", "home_takeaways", "away_takeaways",
    "home_pp_opportunities", "away_pp_opportunities",
    "home_goalie_toi", "away_goalie_toi",
    "home_goals_against", "away_goals_against",
    "home_shots_against", "away_shots_against",
)


def _coerce_score_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Float64 contract for the score-page numeric columns, applied to EVERY
    frame entering the concat — fetched or cache-loaded.

    A page whose every game is cancelled/postponed (2024-10-07) parses a
    column of bare Nones, which pandas types as OBJECT; at concat time pandas
    2.x raises the "concatenation with all-NA entries" FutureWarning because
    that entry's dtype diverges from the float64 result. Coercing to float64
    makes every page dtype-stable (an all-NaN float64 column is identical
    under the old and the new concat semantics — the warning cannot fire and
    the result is unchanged), and covers pre-contract caches that stored the
    object form.
    """
    for c in ("home_score", "away_score", "home_sog", "away_sog"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _coerce_boxscore_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """The same float64 contract for boxscore stats. ONLY the numeric stats
    are coerced — the goalie id/name/decision columns are identities and
    must never see this."""
    for c in BOXSCORE_NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _skater_rows(bs: dict, game_id: str) -> list[dict]:
    """One row per skater who DRESSED, from the two per-side arrays."""
    rows: list[dict] = []
    stats = bs.get("playerByGameStats") or {}
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        block = stats.get(key) or {}
        skaters = list(block.get("forwards") or []) + list(
            block.get("defense") or [])
        for s in skaters:
            pid = s.get("playerId")
            if pid is None:
                continue
            nm = (s.get("name") or {}).get("default") or (s.get("name") or {}).get("full")
            rows.append({"game_id": str(game_id), "side": side,
                         "team": None, "player_id": str(pid),
                         "player_name": nm})
    return rows




def load_boxscores(game_ids: list[str], use_cache: bool = True,
                   gameday_by_id: dict | None = None,
                   chunk_days: int = PULL_CHUNK_DAYS,
                   pause_sec: float = PULL_CHUNK_PAUSE_SEC) -> pd.DataFrame:
    """Fetch /v1/gamecenter/{id}/boxscore for each game and return the
    combined per-game rollup rows. Per-game parquet caches; a failed game
    is warned and skipped (downstream features degrade to NaN), never fatal.

    ``gameday_by_id`` groups the pull into ``chunk_days`` calendar windows
    (MLB's ``_chunked_statcast`` shape) and drives the progress bar. Chunking
    is presentational: the returned rows are re-ordered to the caller's input
    order, so the frame is identical whether or not a mapping is supplied.
    """
    chunks = _chunk_games(game_ids, gameday_by_id, chunk_days)
    frames: list[pd.DataFrame] = []
    hits = fetched = 0
    for n, (label, chunk_ids) in enumerate(chunks, 1):
        logger.info("boxscore chunk %d/%d [%s]: %d games",
                    n, len(chunks), label, len(chunk_ids))
        # The bar alone was the only progress a chunk had, and it is disabled
        # whenever stderr is not a terminal - i.e. in every captured run. The
        # _PullProgress writes the durable log line in both cases.
        prog = _PullProgress(
            len(chunk_ids), f"boxscore chunk {n}/{len(chunks)} [{label}]")
        for gid in chunk_ids:
            path = _cache_path(f"boxscore_{BOXSCORE_CACHE_VERSION}_{gid}.parquet")
            if use_cache and path.exists():
                try:
                    frames.append(_coerce_boxscore_dtypes(pd.read_parquet(path)))
                    hits += 1
                    prog.tick(cached=True)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("boxscore cache %s unreadable (%s)", path.name, exc)
            try:
                bs = _http_json(f"{NHL_API_BASE}/gamecenter/{gid}/boxscore")
                if str(bs.get("gameState") or "") not in FINAL_GAME_STATES:
                    # A mid-game boxscore freezes partial player stats into a
                    # per-game cache nothing ever invalidates; behave exactly
                    # like an unavailable one — warned, skipped, uncached —
                    # and the settled game pulls fresh on a later run.
                    logger.warning("boxscore for game %s is %s (not final) — "
                                   "skipped and never cached; it will be "
                                   "pulled once the game settles",
                                   gid, str(bs.get("gameState") or "unknown"))
                    prog.tick(failed=True)
                    continue
                row = _parse_boxscore(bs)
                df = pd.DataFrame([row], columns=BOXSCORE_COLS)
                df = _coerce_boxscore_dtypes(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("boxscore unavailable for game %s: %s", gid, exc)
                prog.tick(failed=True)
                continue
            fetched += 1
            if not df.empty:
                df.to_parquet(path, index=False)
            frames.append(df)
            prog.tick(cached=False)
        prog.close()
        if pause_sec and n < len(chunks):
            time.sleep(pause_sec)
    logger.info("boxscores resolved: %d requested in %d chunk(s), "
                "%d cache hits, %d fetched",
                len(game_ids), len(chunks), hits, fetched)
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=BOXSCORE_COLS)
    out = pd.concat(frames, ignore_index=True)
    # Chunking may reorder the fetch; restore the caller's order so the frame
    # is byte-identical to an unchunked pull of the same id list.
    if "game_id" in out.columns:
        rank = {str(g): i for i, g in enumerate(game_ids)}
        out = (out.assign(_ord=out["game_id"].astype(str).map(rank))
                  .sort_values("_ord", kind="mergesort", na_position="last")
                  .drop(columns="_ord")
                  .reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
# MoneyPuck player-game archives: the production substrate for the player-pool
# pl_* family (trailing 30-game shrunk EVO/PPO ratings -> team pool means).
# Consumed by features._load_player_ratings; nothing else loads MoneyPuck.
# ---------------------------------------------------------------------------

def _moneypuck_season_start(values: pd.Series) -> pd.Series:
    """Normalize MoneyPuck's YYYY or YYYY-YYYY season labels to start year."""
    numeric = pd.to_numeric(values, errors="coerce")
    start = numeric.where(numeric < 10_000, np.floor(numeric / 10_000))
    text_year = pd.to_numeric(
        values.astype("string").str.extract(r"^\s*(\d{4})", expand=False),
        errors="coerce")
    return start.fillna(text_year).astype("Int64")


def _read_moneypuck_player_game_archive(
    archive: zipfile.ZipFile,
    seasons: set[int],
    required: list[str],
) -> pd.DataFrame:
    """Read only requested game seasons/situations, keeping CSV memory bounded."""
    members = [name for name in archive.namelist()
               if name.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"expected one player-game CSV in archive, found {len(members)}")

    kept: list[pd.DataFrame] = []
    with archive.open(members[0]) as source:
        chunks = pd.read_csv(
            source, usecols=required, low_memory=False,
            dtype={"playerId": "string", "gameId": "string"},
            chunksize=MP_PLAYER_GAME_CHUNK_SIZE)
        for chunk in chunks:
            chunk["season"] = _moneypuck_season_start(chunk["season"])
            chunk = chunk[
                chunk["season"].isin(seasons)
                & chunk["situation"].isin(MONEYPUCK_SITUATIONS)
            ]
            if len(chunk):
                kept.append(chunk.copy())
    if not kept:
        return pd.DataFrame(columns=required)
    return pd.concat(kept, ignore_index=True)


def _download_moneypuck_player_game_archive(
    url: str,
    seasons: set[int],
    required: list[str],
) -> pd.DataFrame:
    """Stream the published ZIP to a temporary cache-local file before parsing."""
    import requests

    response = requests.get(
        url, timeout=(10, 300), stream=True,
        headers={"User-Agent": "sports-prediction-model/1.0"})
    archive_path: Path | None = None
    try:
        response.raise_for_status()
        cache_dir = _cache_path("moneypuck_player_games_download.tmp").parent
        with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".zip", prefix="moneypuck_player_games_",
                dir=cache_dir, delete=False) as target:
            archive_path = Path(target.name)
            iter_content = getattr(response, "iter_content", None)
            if callable(iter_content):
                for block in iter_content(chunk_size=1024 * 1024):
                    if block:
                        target.write(block)
            else:  # simple in-memory response doubles in offline tests
                target.write(response.content)
        with zipfile.ZipFile(archive_path) as archive:
            return _read_moneypuck_player_game_archive(
                archive, seasons, required)
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        close = getattr(response, "close", None)
        if callable(close):
            close()


def load_moneypuck_player_games(
    seasons: list[int] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Load MoneyPuck's published regular-season player-game CSV archives.

    The output has one row per player/game/situation with playerId, gameId,
    date, team, position, individual xGoals, and ice time. Only 5on5 (EVO)
    and 5on4 (PPO) rows are retained. ZIPs are downloaded to a temporary file
    beside the parquet cache and parsed in bounded CSV chunks; the 2.6-GB
    historical CSV is never decompressed into one in-memory DataFrame. Only
    requested start-year seasons are retained and cached.

    Seasons are derived from the calendar by default, so the season in
    progress is always requested. A missing FINISHED season makes the family
    unavailable rather than returning incomplete rolling history; a missing
    live season is expected before MoneyPuck posts it and is skipped with a
    warning, and its cache is refreshed every run because it grows during the
    season. A live-season fetch failure falls back to the last cached copy.

    MoneyPuck data is free for non-commercial use and must be credited.
    """
    defaults = list(config.player_rating_seasons())
    requested = sorted(set(int(s) for s in (defaults if seasons is None else seasons)))
    if not requested:
        return None
    # Seasons that may still be unpublished or actively growing: MoneyPuck
    # posts an archive only once a season's games exist, and keeps updating
    # the in-progress one. A gap here is EXPECTED, unlike a hole in finished
    # history, so it must not take the whole family down.
    live_from = config.current_nhl_season()

    required = ["playerId", "name", "gameId", "season", "playerTeam",
                "gameDate", "position", "situation", "icetime", "I_F_xGoals"]
    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    unpublished: list[int] = []
    historical = sorted(set(requested) & set(MONEYPUCK_PLAYER_GAMES_HISTORY_SEASONS))

    def _cached(season: int, path: Path) -> pd.DataFrame | None:
        """Read one cached season, or None when it is unusable."""
        if not path.exists():
            return None
        try:
            cached = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MoneyPuck player games %s cache unreadable (%s)",
                           season, exc)
            return None
        if not all(column in cached.columns for column in required):
            return None
        cached["season"] = _moneypuck_season_start(cached["season"])
        if not (cached["season"] == season).any():
            return None
        return cached[cached["season"].eq(season)
                      & cached["situation"].isin(MONEYPUCK_SITUATIONS)].copy()

    if historical:
        path = _cache_path(
            f"moneypuck_player_games_{MP_PLAYER_GAME_VERSION}_2008_to_2024.parquet")
        history_frame: pd.DataFrame | None = None
        if use_cache and path.exists():
            try:
                cached = pd.read_parquet(path)
                if all(column in cached.columns for column in required):
                    cached["season"] = _moneypuck_season_start(cached["season"])
                    cached_seasons = set(cached["season"].dropna().astype(int))
                    if set(historical).issubset(cached_seasons):
                        history_frame = cached[
                            cached["season"].isin(historical)
                            & cached["situation"].isin(MONEYPUCK_SITUATIONS)
                        ].copy()
            except Exception as exc:  # noqa: BLE001
                logger.warning("MoneyPuck historical player-game cache unreadable (%s)",
                               exc)
        if history_frame is None:
            try:
                history_frame = _download_moneypuck_player_game_archive(
                    MONEYPUCK_PLAYER_GAMES_HISTORY_URL,
                    set(historical), required)
                if history_frame.empty:
                    raise ValueError("historical archive has no requested skater games")
                history_frame.to_parquet(path, index=False)
                logger.info("MoneyPuck historical player games: cached %d rows "
                            "for seasons %s", len(history_frame), historical)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MoneyPuck historical player games unavailable: %s", exc)
                missing.extend(historical)
        if history_frame is not None:
            frames.append(history_frame)

    for season in (s for s in requested
                   if s not in MONEYPUCK_PLAYER_GAMES_HISTORY_SEASONS):
        path = _cache_path(
            f"moneypuck_player_games_{MP_PLAYER_GAME_VERSION}_{season}.parquet")
        # A finished season's archive is immutable, so it caches forever. The
        # live season gains games after every game day: serving its cached copy
        # would freeze ratings at the day the cache was written, silently and
        # with no error at all — worse than never having fetched it. So the live
        # season is re-fetched every run, and falls back to the last good copy
        # only when the fetch itself fails.
        live = season >= live_from
        frame: pd.DataFrame | None = None
        if use_cache and path.exists() and not live:
            frame = _cached(season, path)
        if frame is None:
            try:
                url = MONEYPUCK_PLAYER_GAMES_ZIP_URL.format(season=season)
                frame = _download_moneypuck_player_game_archive(
                    url, {season}, required)
                if frame.empty:
                    raise ValueError(
                        f"archive has no regular-season skater rows for {season}")
                frame.to_parquet(path, index=False)
                _log_once(logging.INFO, "MoneyPuck player games %s: cached %d rows",
                          season, len(frame))
            except Exception as exc:  # noqa: BLE001
                stale = _cached(season, path) if live else None
                if stale is not None:
                    logger.warning("MoneyPuck player games %s refresh failed "
                                   "(%s); serving the cached copy", season, exc)
                    frame = stale
                else:
                    logger.warning("MoneyPuck player games %s unavailable: %s",
                                   season, exc)
                    (unpublished if live else missing).append(season)
                    continue
        frames.append(frame)

    if unpublished:
        # Before opening night (or before MoneyPuck regenerates the live
        # archive) the current season legitimately has no rows. Falling through
        # to finished history is the correct reading; killing the family would
        # drop all 24 pl_* columns to position priors every year for no reason.
        logger.warning("MoneyPuck season(s) not available yet: %s; ratings run "
                       "through season %s", unpublished,
                       max((int(s) for s in requested
                            if s not in unpublished), default="none"))
    if missing:
        logger.error("MoneyPuck player-game history incomplete; unavailable "
                     "seasons: %s", missing)
        return None
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["season"] = _moneypuck_season_start(out["season"])
    out = out[
        out["season"].isin(requested)
        & out["situation"].isin(MONEYPUCK_SITUATIONS)
    ]
    return out.reset_index(drop=True) if len(out) else None


def _utc_now() -> pd.Timestamp:
    """Return the current UTC time; isolated for deterministic snapshot tests."""
    return pd.Timestamp.now(tz="UTC")


def _valid_espn_injury_payload(payload: object) -> bool:
    """Whether a payload is a complete enough current-injuries snapshot."""
    if not isinstance(payload, dict) or not isinstance(payload.get("injuries"), list):
        return False
    for team_block in payload["injuries"]:
        if not isinstance(team_block, dict) or "injuries" not in team_block:
            return False
        records = team_block["injuries"]
        if not isinstance(records, list):
            return False
        if any(not isinstance(row, dict) for row in records):
            return False
    return True


def load_espn_injuries(use_cache: bool = True,
                       snapshot: bool = True) -> pd.DataFrame | None:
    """ESPN's dated NHL injury report -> a long, snapshot-appendable frame.

    The official NHL API publishes no injury endpoint, so ESPN is the feed of
    record. Structure:    ``injuries[] -> {team} -> {athlete, status, date,
    type, details}``. Emitted rows carry both ESPN's ``report_date`` and our
    UTC ``snapshot_at`` capture timestamp. The capture timestamp—not a
    retrospective report date—is used for game-level point-in-time replay.

    PIT IS SNAPSHOT-BASED. The endpoint serves only the CURRENT report and
    ignores date parameters (verified 2026-09-26: ``?dates=``/``?date=`` both
    return today's payload), and ESPN prunes records older than roughly ten
    days. Every successful pull is appended with an exact UTC capture timestamp;
    even an empty current injury list gets a marker row so the snapshot is not
    confused with a failed fetch. History therefore starts only with the first
    captured snapshot. Missing periods remain unknown, not healthy.

    ESPN requires a browser user-agent: the shared ``sports-prediction-model/1.0``
    agent is answered with 403 (verified 2026-09-26), which is a silent
    "everyone is healthy" failure if the status code is not checked.
    """
    payload: dict | None = None
    captured_at = None
    now = _utc_now()
    if use_cache:
        cached = _cache_path(f"espn_injuries_{INJURY_VERSION}_latest.json")
        if cached.exists():
            try:
                envelope = json.loads(cached.read_text(encoding="utf-8"))
                if isinstance(envelope, dict) and "snapshot_payload" in envelope:
                    payload = envelope.get("snapshot_payload")
                    captured_at = pd.to_datetime(
                        envelope.get("snapshot_at"), errors="coerce", utc=True)
                    if (not _valid_espn_injury_payload(payload)
                            or pd.isna(captured_at) or captured_at > now):
                        # A malformed or future-dated cache is not a complete
                        # report we can safely replay; fetch instead of making
                        # its absence of rows look like a clean report.
                        payload, captured_at = None, None
                else:
                    # A provider payload without our capture timestamp cannot
                    # prove when we observed it; refresh rather than stamping it
                    # with today's time and introducing look-ahead.
                    logger.warning("legacy ESPN injury cache lacks capture "
                                   "time; refreshing before use")
            except Exception as exc:  # noqa: BLE001
                logger.warning("ESPN injury cache unreadable (%s)", exc)
                payload, captured_at = None, None

    # Reuse a recent capture across repeated feature builds, but refresh even
    # within the same UTC day once the report cache exceeds its short TTL. A
    # cached payload is never re-stamped as fresh: fetch again or retain only
    # the previous genuinely dated history.
    cache_age = (now - captured_at) if captured_at is not None else None
    needs_fetch = (payload is None or captured_at is None
                   or cache_age > pd.Timedelta(hours=INJURY_CACHE_TTL_HOURS))
    if needs_fetch:
        try:
            # The ESPN edge blocks agents and header profiles in passing
            # waves — the 2026-09-28 Kaggle run's single 403 (a profile that
            # tested 200 minutes earlier and later) cost BOTH snapshots of
            # the day because one failed request fell straight to history.
            # A 403 here is transient like a 5xx: retry it with the verified
            # short browser agent (ESPN_HEADERS) instead of silently serving
            # the slate on stale injury state (unknown is not healthy).
            try:
                fetched = _http_json(
                    ESPN_NHL_INJURIES_URL, retries=4, timeout=45.0,
                    headers=ESPN_HEADERS,
                    retry_statuses=(403, 429, 500, 502, 503, 504))
            except Exception as wave_exc:  # noqa: BLE001
                # The wave can outlast any retry schedule (2026-09-29 15:01:
                # both runs lost the fetch ~9 minutes, every attempt of all
                # 4 retries 403'd in BOTH phases while the same identity
                # tested 200 minutes later). Fall over ONCE to the
                # library-default identity — no User-Agent header at all —
                # a structurally different request profile the wave has not
                # necessarily blocked. Still transient-aware, still bounded;
                # only then does history take over (unknown is not healthy,
                # but a second identity doubles the chance the snapshot is
                # captured at all).
                logger.warning("ESPN injury fetch exhausted the browser-agent "
                               "retries (%s); falling back to the "
                               "library-default identity", wave_exc)
                fetched = _http_json(
                    ESPN_NHL_INJURIES_URL, retries=2, timeout=45.0,
                    headers=ESPN_IDENTITY_HEADERS,
                    retry_statuses=(403, 429, 500, 502, 503, 504))
            if not _valid_espn_injury_payload(fetched):
                raise ValueError("ESPN injury payload missing a complete injuries list")
            payload = fetched
            captured_at = _utc_now()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN injury report unavailable; preserving only "
                           "previously captured history (unknown is not healthy): %s",
                           exc)
            return _injury_history() if snapshot else None
        if use_cache:
            try:
                _cache_path(f"espn_injuries_{INJURY_VERSION}_latest.json").write_text(
                    json.dumps({"snapshot_payload": payload,
                                "snapshot_at": captured_at.isoformat()}),
                    encoding="utf-8")
            except OSError as exc:
                logger.warning("ESPN latest injury cache write failed: %s", exc)

    if not _valid_espn_injury_payload(payload):
        logger.warning("ESPN injury snapshot is incomplete; ignoring it")
        return _injury_history() if snapshot else None

    rows: list[dict] = []
    for team_block in payload["injuries"]:
        team_abbr = None
        team_obj = team_block.get("team") or {}
        if isinstance(team_obj, dict):
            team_abbr = team_obj.get("abbreviation") or team_obj.get("displayName")
        team_abbr = team_abbr or team_block.get("displayName")
        for rec in team_block.get("injuries", []):
            athlete = rec.get("athlete")
            athlete = athlete if isinstance(athlete, dict) else {}
            details = rec.get("details")
            details = details if isinstance(details, dict) else {}
            itype = rec.get("type")
            itype = itype if isinstance(itype, dict) else {}
            rows.append({
                "player_name": athlete.get("displayName"),
                "player_id": rec.get("id"),
                "team": team_abbr,
                "status": rec.get("status"),
                "status_code": itype.get("abbreviation") or itype.get("name"),
                "report_date": rec.get("date"),
                "return_date": details.get("returnDate"),
                "injury_type": details.get("type"),
                "detail": rec.get("shortComment"),
                "snapshot_at": captured_at,
                "snapshot_marker": False,
            })
    if not rows:
        rows = [{
            "player_name": None, "player_id": None, "team": None,
            "status": None, "status_code": None, "report_date": None,
            "return_date": None, "injury_type": None, "detail": None,
            "snapshot_at": captured_at, "snapshot_marker": True,
        }]
    today = pd.DataFrame(rows)
    if not snapshot:
        return today
    return _append_injury_snapshot(today)


def _append_injury_snapshot(today: pd.DataFrame) -> pd.DataFrame:
    """Append an exactly timestamped report snapshot, including empty reports."""
    if "snapshot_at" not in today.columns or today.empty:
        raise ValueError("injury snapshot must carry a capture timestamp")
    stamps = pd.to_datetime(today["snapshot_at"], errors="coerce", utc=True)
    if stamps.isna().any() or stamps.nunique() != 1:
        raise ValueError("all injury rows must share one valid capture timestamp")
    today = today.copy()
    today["snapshot_at"] = stamps
    path = _cache_path(f"espn_injuries_{INJURY_VERSION}_history.parquet")

    # Two tiers, same order and identity as `_injury_history`: the machine-
    # local history first, then the repo-carried artifact. A run that CAN
    # fetch must still INHERIT the artifact's older snapshots — the 16:10
    # run captured a fresh 111-row snapshot on a cold cache, exported 217
    # rows over 2 snapshots in Phase 12, but Phases 3/11 saw only those 111
    # (1 snapshot), so the older snapshot was invisible to the exclusion
    # engine for that entire run. A snapshot the artifact already carries
    # must never be shadowed by a later cold-cache capture.
    tier_frames: list[pd.DataFrame] = []
    if path.exists():
        try:
            local_hist = pd.read_parquet(path)
            if len(local_hist):
                tier_frames.append(local_hist)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN injury history unreadable (%s)", exc)
    artifact = config.DATA_DELIVERY_DIR / INJURY_HISTORY_ARTIFACT
    if artifact.exists() and artifact != path:
        try:
            carried = pd.read_parquet(artifact)
            if len(carried):
                tier_frames.append(carried)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN injury history artifact unreadable (%s)", exc)
    prior: pd.DataFrame | None = None
    if tier_frames:
        prior = (tier_frames[0] if len(tier_frames) == 1
                 else pd.concat(tier_frames, ignore_index=True, sort=False))
        keys = [c for c in ("snapshot_at", "player_id", "player_name")
                if c in prior.columns]
        if keys:
            prior = prior.drop_duplicates(keys, keep="first")
    if prior is None or not len(prior):
        out = today
    else:
        if "snapshot_at" in prior.columns:
            stamp = stamps.iloc[0]
            prior_stamps = pd.to_datetime(prior["snapshot_at"],
                                          errors="coerce", utc=True)
            prior = prior.loc[prior_stamps != stamp]
        out = pd.concat([prior, today], ignore_index=True, sort=False)
    try:
        out.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001 — history is best-effort
        logger.warning("could not persist injury history: %s", exc)
    return out.reset_index(drop=True)


# Repo-carried injury-history artifact: the last successfully captured ESPN
# snapshots, persisted into data_delivery by the pipeline so a sandbox whose
# egress to the injury endpoint is blocked (2026-09-28 Kaggle runs: 403 on
# every retry wave while the same profile probed 200 from another network)
# still replays captured history instead of none.
INJURY_HISTORY_ARTIFACT = "nhl_injury_snapshot_history.parquet"


# ---------------------------------------------------------------------------
# Team rosters — player→team membership for pool gating (PIT-stamped)
# ---------------------------------------------------------------------------

#: Roster snapshots refresh at most this often — the ESPN injury cache TTL,
#: reused because the same reasoning applies: a roster rarely changes twice
#: within a run-day, and a cached capture is NEVER re-stamped as fresh.
ROSTER_CACHE_TTL_HOURS = 6
ROSTER_VERSION = "v1"


def load_team_roster_snapshot(teams, *, use_cache: bool = True,
                              fetch: bool = True) -> pd.DataFrame | None:
    """Current NHL roster membership for ``teams``: (player_id, team) + capture.

    Why this exists: the pool's as-of join is keyed on team, so a rating row
    keeps serving a player to his OLD club after a trade — for up to
    POOL_LOOKBACK_DAYS in-season, and through every opener-window carry after
    an off-season move (the 2023 Kane case: 42 days served to Chicago after
    leaving). The official API publishes no transactions endpoint (verified
    2026-09-29: every /v1/transactions variant 404s), so the roster-of-record
    is the per-team roster endpoint — one fetch per requested team, current
    season.

    PIT IS CAPTURE-BASED, exactly like the ESPN injury snapshots: every row
    carries the UTC instant the roster was observed, and consumers apply a
    capture only to games whose puck drop is strictly later than it. The
    capture is never re-stamped — a cached frame keeps its original
    ``snapshot_at`` even when served many runs later, so a stale cache fails
    CLOSED (fewer games covered), never open.

    Failure contract (egress-blocked Kaggle runs are a fact — the 2026-09-28
    injury endpoint blocked for hours):

    * a team that fails to fetch is recorded, not fatal — the frame comes back
      flagged ``complete=False`` and consumers restrict themselves to positive
      membership knowledge (they may move a player they SEE, never drop a
      player they merely do not see);
    * a run where every team fails returns the previous capture at its true
      (older) timestamp, or None when there has never been one — never a
      fabricated "everyone on their current team" frame;
    * a failed attempt writes an attempt-stamp so the TTL suppresses retries
      within the window instead of re-blocking every feature build.

    Columns: ``player_id`` (str, NHL central id — the same id space as
    MoneyPuck playerId), ``team`` (abbr), ``snapshot_at`` (UTC), ``complete``
    (bool — every requested team present in this capture).
    """
    requested = sorted({str(t) for t in teams if t is not None and str(t)})
    if not requested:
        return None
    rows_path = _cache_path(f"nhl_roster_{ROSTER_VERSION}_latest.parquet")
    meta_path = _cache_path(f"nhl_roster_{ROSTER_VERSION}_meta.json")

    def _read_meta() -> dict:
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001 — absent/unreadable meta == no capture
            return {}

    meta = _read_meta() if use_cache else {}
    prior: pd.DataFrame | None = None
    if use_cache and rows_path.exists():
        try:
            prior = pd.read_parquet(rows_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("team roster cache unreadable (%s)", exc)

    now = _utc_now()
    captured = pd.to_datetime(meta.get("captured_at"), errors="coerce", utc=True)
    ok = bool(meta.get("ok"))
    fresh_attempt = (pd.notna(captured) and captured <= now
                     and (now - captured)
                     <= pd.Timedelta(hours=ROSTER_CACHE_TTL_HOURS))
    covered = set(meta.get("teams_requested") or []) if ok else set()
    # Fetch when there is no recent attempt to lean on, when the rows went
    # missing under an "ok" stamp, or when the cached capture never saw one
    # of the requested teams — its absence there must not read as "nobody
    # on that roster". A fresh FAILED attempt suppresses retries until the
    # TTL passes instead of re-blocking every feature build.
    needs_fetch = fetch and (
        not fresh_attempt
        or (ok and prior is None)
        or (ok and not set(requested) <= covered)
    )
    if needs_fetch:
        season = int(config.current_nhl_season())
        season_id = f"{season}{season + 1}"
        rows: list[dict] = []
        failed: list[str] = []
        for team in requested:
            try:
                payload = _http_json(
                    f"{NHL_API_BASE}/roster/{team}/{season_id}",
                    retries=2, timeout=15.0,
                    retry_statuses=(403, 429, 500, 502, 503, 504))
                roster = payload.get("forwards") or []
                roster = roster + (payload.get("defensemen") or []) \
                    + (payload.get("goalies") or [])
                ids = [str(p.get("id")) for p in roster if p.get("id")]
                if not ids:
                    raise ValueError("roster payload carries no player ids")
                rows.extend({"player_id": pid, "team": team} for pid in ids)
            except Exception as exc:  # noqa: BLE001
                failed.append(team)
                logger.warning("team roster fetch failed for %s (%s)", team, exc)
        attempt_at = now.isoformat()
        if rows:
            stamp = now.isoformat()
            frame = pd.DataFrame(rows)
            frame["snapshot_at"] = stamp
            frame["complete"] = not failed
            ok_teams = sorted(set(requested) - set(failed))
            try:
                frame.to_parquet(rows_path, index=False)
                meta_path.write_text(json.dumps({
                    "captured_at": attempt_at, "ok": True,
                    "complete": not failed, "failed_teams": failed,
                    "teams_requested": ok_teams,
                    "season_id": season_id,
                }), encoding="utf-8")
            except OSError as exc:  # noqa: BLE001
                logger.warning("team roster cache write failed: %s", exc)
            if failed:
                logger.warning("team roster snapshot incomplete: %d/%d teams "
                               "fetched — membership uses positive knowledge "
                               "only (absent players keep their row team)",
                               len(requested) - len(failed), len(requested))
            logger.info("team roster snapshot: %d player-team row(s) across "
                        "%d/%d team(s) through %s",
                        len(frame), len(requested) - len(failed),
                        len(requested), stamp)
            return frame
        # Every team failed: stamp the attempt so the TTL gates retries, then
        # fall back to the previous capture at ITS OWN timestamp (older
        # capture = fewer covered games = fail closed).
        try:
            meta_path.write_text(json.dumps({
                "captured_at": attempt_at, "ok": False,
                "complete": False, "failed_teams": failed,
                "season_id": season_id,
            }), encoding="utf-8")
        except OSError:  # noqa: BLE001 — a failed stamp write changes nothing
            pass
        logger.warning("team roster snapshot unavailable (%d/%d teams "
                       "failed); using the previous capture at its original "
                       "timestamp, or row-keyed membership when there is none",
                       len(failed), len(requested))

    if prior is not None and len(prior):
        prior = prior.copy()
        prior["snapshot_at"] = pd.to_datetime(
            prior.get("snapshot_at"), errors="coerce", utc=True)
        return prior
    return None


# ---------------------------------------------------------------------------
# Non-medical leave-of-absence events (PIT-legal by construction)
# ---------------------------------------------------------------------------

def load_leave_events() -> pd.DataFrame:
    """Public non-medical leave-of-absence events as a long event frame.

    No free live feed publishes these events (verified 2026-09-29: the
    official NHL API serves no transactions endpoint — every /v1/transactions
    variant 404s — and ESPN's injury payload vocabulary is exactly
    {IR, Day-To-Day, Suspension, Out} with zero leave rows). The reliable
    channel is therefore a versioned ledger of PUBLIC announcements, each
    carrying the UTC instant the information became public — the exact PIT
    boundary. Reproducibility contract: the same ledger version yields the
    same intervals; the replay is never a static incremental stack (every
    interval is re-derived from the events on every run).

    Schema: player_name, player_id (nullable MoneyPuck id), team (abbr,
    nullable), announced_at_utc (ISO-8601), returned_at_utc (nullable — null
    means the leave is still open and will close on the ledger's next
    version), source_url, note. A row whose announced_at_utc is null or
    unparseable is refused loudly, never silently dropped.
    """
    path = config.DATA_DELIVERY_DIR / config.LEAVE_EVENTS_LEDGER
    if not path.exists():
        logger.info("leave-events ledger absent (%s) — no leave channel, "
                    "which is honest when no leave is known", path.name)
        return pd.DataFrame(columns=["player_name", "player_id", "team",
                                     "announced_at_utc", "returned_at_utc",
                                     "source_url", "note"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("leave-events ledger unreadable (%s) — treated as "
                       "absent; nothing is inferred", exc)
        return pd.DataFrame(columns=["player_name", "player_id", "team",
                                     "announced_at_utc", "returned_at_utc",
                                     "source_url", "note"])
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        logger.warning("leave-events ledger malformed (no events list) — "
                       "treated as absent")
        return pd.DataFrame(columns=["player_name", "player_id", "team",
                                     "announced_at_utc", "returned_at_utc",
                                     "source_url", "note"])
    rows, refused = [], 0
    for ev in events:
        announced = pd.to_datetime(ev.get("announced_at_utc"), errors="coerce",
                                   utc=True)
        if pd.isna(announced):
            refused += 1
            logger.warning("leave event refused (no parseable announced_at_utc): "
                           "%s", ev)
            continue
        returned = pd.to_datetime(ev.get("returned_at_utc"), errors="coerce",
                                  utc=True)
        rows.append({
            "player_name": ev.get("player_name"),
            "player_id": ev.get("player_id"),
            "team": ev.get("team"),
            "announced_at_utc": announced,
            "returned_at_utc": None if pd.isna(returned) else returned,
            "source_url": ev.get("source_url"),
            "note": ev.get("note"),
        })
    if refused:
        logger.warning("leave-events ledger: %d event(s) refused", refused)
    out = pd.DataFrame(rows)
    if "announced_at_utc" in out.columns:
        out = out.sort_values("announced_at_utc").reset_index(drop=True)
    out.attrs["ledger_updated_utc"] = payload.get(
        "ledger_updated_utc") if isinstance(payload, dict) else None
    _log_once(logging.INFO, "leave-events ledger: %d event(s) loaded (%d refused; "
              "ledger as-of %s)", len(out), refused,
              out.attrs["ledger_updated_utc"])
    return out


def _injury_history() -> pd.DataFrame | None:
    """The best available captured injury history, never fabricated.

    Two tiers: the local snapshot cache first (freshest, machine-local),
    then the repo-carried artifact in data_delivery — the export a previous
    successful run persisted for exactly the sandbox runs whose egress to
    the injury endpoint is blocked. Tiers are unioned so a Kaggle run that
    CAN refresh keeps both its new rows and the artifact's older ones.
    """
    frames: list[pd.DataFrame] = []
    local = _cache_path(f"espn_injuries_{INJURY_VERSION}_history.parquet")
    if local.exists():
        try:
            df = pd.read_parquet(local)
            if len(df):
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN injury history unreadable (%s)", exc)
    artifact = config.DATA_DELIVERY_DIR / INJURY_HISTORY_ARTIFACT
    if artifact.exists() and artifact != local:
        try:
            df = pd.read_parquet(artifact)
            if len(df):
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ESPN injury history artifact unreadable (%s)", exc)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    keys = [c for c in ("snapshot_at", "player_id", "player_name")
            if c in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="first")
    return combined.reset_index(drop=True)


def export_injury_history_artifact() -> str | None:
    """Persist the local captured history into data_delivery (best effort).

    Called by the pipeline's persistence phase so the NEXT run — local or
    sandbox — inherits every snapshot this run captured. The export is
    snapshot-aware and growing: it unions the machine-local history with
    whatever the artifact already carries (deduped on the same
    ``(snapshot_at, player_id)`` identity the history loader uses) so a repo
    snapshot can never lose older rows to a machine that starts its capture
    later. Never fails a run; returns the artifact name when written, None
    when there is nothing to persist (the artifact simply ages until a
    healthy fetch repopulates it).
    """
    local = _cache_path(f"espn_injuries_{INJURY_VERSION}_history.parquet")
    if not local.exists():
        return None
    try:
        df = pd.read_parquet(local)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ESPN injury history export skipped (unreadable): %s", exc)
        return None
    if not len(df):
        return None
    artifact = config.DATA_DELIVERY_DIR / INJURY_HISTORY_ARTIFACT
    try:
        if artifact.exists():
            try:
                carried = pd.read_parquet(artifact)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ESPN injury history artifact unreadable; replacing it "
                    "with the local history: %s", exc)
            else:
                if len(carried):
                    combined = pd.concat([carried, df], ignore_index=True, sort=False)
                    keys = [c for c in ("snapshot_at", "player_id", "player_name")
                            if c in combined.columns]
                    if keys:
                        combined = combined.drop_duplicates(keys, keep="first")
                    df = combined
        artifact.parent.mkdir(parents=True, exist_ok=True)
        tmp = artifact.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(artifact)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ESPN injury history export failed (run continues): %s", exc)
        return None
    stamps = pd.to_datetime(df["snapshot_at"], errors="coerce", utc=True) \
        if "snapshot_at" in df.columns else pd.Series(dtype="datetime64[ns, UTC]")
    logger.info(
        "injury history artifact: %d rows over %d captured snapshot(s) "
        "(%s .. %s) -> %s",
        len(df), int(stamps.nunique()) if len(stamps) else 0,
        stamps.min() if len(stamps) else "-",
        stamps.max() if len(stamps) else "-", artifact.name)
    return artifact.name


def load_team_names() -> dict[str, str]:
    """team abbrev -> full team name (frontend games[] display fields).

    Resolved from a score-page team block (name.default), cached as JSON.
    Falls back to the config team map's abbreviations when unreachable.
    """
    path = _cache_path("team_names.json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    names: dict[str, str] = {}
    try:
        today = date.today().isoformat()
        payload = _http_json(f"{NHL_API_BASE}/score/{today}")
        for g in (payload.get("games") or []):
            for key in ("homeTeam", "awayTeam"):
                t = g.get(key) or {}
                abbr = t.get("abbrev")
                name = (t.get("name") or {}).get("default")
                if abbr and name:
                    names.setdefault(str(abbr), str(name))
    except Exception as exc:  # noqa: BLE001
        logger.warning("team-name resolution failed: %s", exc)
    if names:
        try:
            path.write_text(json.dumps(names), encoding="utf-8")
        except OSError:
            pass
    return names
