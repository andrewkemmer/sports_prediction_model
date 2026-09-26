"""NHL data ingestion — official NHL API pull with local parquet caching.

Primary source: the official NHL public web API (``api-web.nhle.com/v1``),
no auth. Endpoints used:

  /v1/score/{YYYY-MM-DD}            game list: ids, teams, scores, SOG, state
  /v1/gamecenter/{id}/boxscore      team + per-player boxscore (goalies incl.
                                    TOI/decision; skaters incl. SOG, PPG,
                                    faceoff%, hits, blocks, PIM, G/A)

Optional enrichment family (user-approved design): MoneyPuck free season
CSVs (``moneypuck.com/data.htm``, 2007+, non-commercial, credit required)
add shot-level xG. ``load_moneypuck_shots`` downloads and caches the season
shots file; when a season file is unavailable the enrichment columns degrade
to NaN per the documented missing-value policy — never fabricated, and the
production feature contract never DEPENDS on them.

Cache directory: OUTSIDE the git tree (repo root's parent), ``.nhl_cache/``
— the NFL pattern. All frames carry normalized NHL-API column names; the
feature engine is the only place that interprets them.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

try:  # package-relative import when used as backend.ingestion
    from backend import config  # type: ignore
except ImportError:  # running as a top-level module
    import config

logger = logging.getLogger(__name__)

# Cache directory: OUTSIDE the git tree (repo root's parent) so caches never
# pollute the working tree; overridable for tests.
CACHE_DIR = Path(config.ROOT_DIR.parent) / ".nhl_cache"

NHL_API_BASE = "https://api-web.nhle.com/v1"
MONEYPUCK_SHOTS_URL = ("https://moneypuck.com/moneyPuck/playerDataByGame/"
                       "shotsAllYears.csv")

# Per-game keep-list from /v1/score/{date} games[] — the schedule/board
# population. Scores stay None for unplayed games (honest pre-game rows).
SCORE_KEEP = [
    "game_id", "season", "game_type", "game_date", "start_time_utc",
    "home_team", "away_team", "home_score", "away_score",
    "home_sog", "away_sog", "venue", "game_state", "game_outcome",
]

# Cache schema versions: bump whenever the keep-list widens so stale caches
# are ignored rather than silently serving the old column set.
SCORE_CACHE_VERSION = "v1"
# v4: starter selection now keys on the API's ``starter`` boolean (v3 keyed on
# a ``decision`` set that omitted the overtime-loss code "O", so those games
# resolved to the 00:00 scratch goalie), and ``powerPlayShotsAgainst="0/0"``
# is now recorded as the measured 0 it is rather than a null.
BOXSCORE_CACHE_VERSION = "v4"
MP_CACHE_VERSION = "v1"

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


def _http_json(url: str, retries: int = 3, timeout: float = 30.0):
    """GET a JSON document with retry/backoff (the official API is free but
    rate-limited; a transient 5xx must not fail a run).

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
                                headers={"User-Agent": "sports-prediction-model/1.0"})
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
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
                    frames.append(pd.read_parquet(path))
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
                logger.info("score page %s carries %d unplayed game(s) — not "
                            "cached, within the %d-day posting lag",
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
    (2024+), with gameType in config.GAME_TYPES."""
    df = schedule.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"].isin(config.ALL_SEASONS)]
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(config.GAME_TYPES)]
    for c in ("home_score", "away_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
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
                    frames.append(pd.read_parquet(path))
                    hits += 1
                    prog.tick(cached=True)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("boxscore cache %s unreadable (%s)", path.name, exc)
            try:
                bs = _http_json(f"{NHL_API_BASE}/gamecenter/{gid}/boxscore")
                row = _parse_boxscore(bs)
                df = pd.DataFrame([row], columns=BOXSCORE_COLS)
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
# MoneyPuck enrichment (optional; honest NaN degradation)
# ---------------------------------------------------------------------------

def load_moneypuck_shots(seasons: list[int] | None = None,
                         use_cache: bool = True) -> pd.DataFrame | None:
    """MoneyPuck shot-level season CSV (optional enrichment family).

    The public download is one all-years shots file; it is cached once and
    filtered to the requested seasons. A failed/absent download returns None
    (the pipeline logs it and continues — the enrichment columns degrade to
    NaN per the documented policy; production features never depend on
    them). MoneyPuck data is non-commercial and requires credit.
    """
    path = _cache_path(f"moneypuck_shots_{MP_CACHE_VERSION}.parquet")
    df: pd.DataFrame | None = None
    if use_cache and path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("moneypuck cache unreadable (%s)", exc)
    if df is None:
        try:
            import requests
            resp = requests.get(MONEYPUCK_SHOTS_URL, timeout=120,
                                headers={"User-Agent": "sports-prediction-model/1.0"})
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content), low_memory=False)
            df.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MoneyPuck shots unavailable (enrichment degrades "
                           "to NaN): %s", exc)
            return None
    if df is None or df.empty:
        return None
    if seasons and "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce")
                .isin([int(s) * 10000 + (int(s) + 1) for s in seasons])
                | pd.to_numeric(df["season"], errors="coerce").isin(seasons)]
    return df.reset_index(drop=True) if len(df) else None


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
