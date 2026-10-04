"""Point-in-time pre-game injury designations from the OFFICIAL NBA report.

Every NBA team is required to file an injury report before every game. Those
filings are published as a dated, time-stamped PDF::

    https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf

This module reads that archive. It exists because nothing else publishes
historical pre-game NBA availability, and that was measured rather than
assumed - see ``README.md`` for the probe table. The alternatives, and why
each fails:

* **ESPN's league ``injuries`` feed.** Answers HTTP 200 with ~800KB, ignores
  every date parameter, and holds three in-season records for a whole season.
  It is a snapshot of *now*, and history only begins at the first call.
* **ESPN's per-event ``summary`` injuries block.** Looks like per-game history
  and is the more dangerous of the two: it serves TODAY's status grafted onto
  an old game. Verified by counterexample - Johnny Furphy is listed ``Out``
  for 2026-01-10, a night he played 20 minutes. A feed that falsifies the box
  score is worse than an empty one, because it fails silently and plausibly.
* **The PDFs.** Dated, per-game, and correct. See ``VALIDATED`` below.

THE POINT-IN-TIME RULE
----------------------
A report is filed many times per day, and it *moves*: on 2026-01-10 Anthony
Edwards was ``Questionable`` in the 08:00 filing and ``Available`` by 12:30.
So a designation is only knowable relative to a moment, and the moment is
tipoff. This module therefore takes, for each game, the LAST report published
STRICTLY BEFORE that game's tipoff::

    designation as of tipoff  <=>  the latest report R with R < tipoff

The inequality is strict for the same reason MLB's IL predicate is strict: a
status published at the instant of tipoff is not knowable to anyone making a
bet at that instant, so it cannot gate that game. Using the last report *at or
after* tipoff would import the post-game answer, which is the lookahead leak
that retired MLB's ``lineup_actual_*`` columns.

VALIDATED
---------
On 2026-01-10, across every game on the slate: 118 players designated Out or
Questionable or Doubtful in the last pre-tipoff report, and **all 118 were
absent from the box score - zero violations**. The converse held too:
``Available`` players played in 4/4, 6/7, 2/3 and 2/2 cases. A designation is
therefore a fact about the game, not a guess about the player's health.

TWO FILENAME ERAS
-----------------
Reports were filed hourly until 2025-12-22 (``10PM``) and at 15-minute
resolution after it (``06_45PM``). Getting this wrong is a 403 that reads
exactly like a missing file, which is how the whole archive was nearly
dismissed as blocked. The hyphen in ``Injury-Report_`` matters for the same
reason: ``Injury Report_`` with a space is a 403 too.
"""
from __future__ import annotations

import importlib.util
import logging
import re
import sys
import time as _time  # aliased: `time` is also imported from datetime below
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

#: The archive root. The hyphen in ``Injury-Report_`` is load-bearing.
REPORT_URL = ("https://ak-static.cms.nba.com/referee/injury/"
              "Injury-Report_{stamp}.pdf")

#: Reports older than this are hourly-named. Measured, not assumed: the
#: 2025-12-22 filings are the first at 15-minute resolution.
QUARTER_HOUR_FROM = datetime(2025, 12, 22, 9, 0)

#: A browser agent. This host 403s an honest client agent on a cold cache.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/144.0.7559.132 Safari/537.36")

#: The six designations the league actually publishes. Measured across a full
#: season: this is the complete vocabulary, not a sample of it. Note it is
#: strictly richer than ESPN's, which carries only ``Out`` and ``Day-To-Day``
#: and therefore cannot express a player who is expected to play.
DESIGNATIONS = ("Out", "Available", "Doubtful", "Questionable", "Probable",
                "Recovery")

#: A designation that means the player is not expected to play. ``Recovery``
#: is grouped with ``Out`` because the league files it for players who will
#: not dress; everything else is a designation the league expects him to dress
#: for, however uncertain it is.
WILL_NOT_PLAY = ("Out", "Recovery")

#: The two states the rating consumes. Measured over 13,168 point-in-time
#: designations from 2025-26 (see ``backfill_injury_designations.py``)::
#:
#:     Out          0 / 10,639  ->  0.0% played
#:     Doubtful     0 /      5  ->  0.0% played
#:     Available    2,006 / 2,448 -> 81.9% played
#:     Probable     23 /    24  -> 95.8% played
#:     Questionable 34 /    52  -> 65.4% played
#:
#: ``Questionable`` is folded into the available bucket DELIBERATELY, and the
#: reason is sample size rather than its play rate. It is 52 observations -
#: 0.4% of a season, with a +/-13 point interval around 65% - and a third
#: state whose only parameter is estimated from that is a parameter fitted to
#: almost nothing. The cost is real and is accepted knowingly: a player the
#: league flagged as uncertain is weighted at 0.816 rather than 0.654, so
#: roughly a third of those appearances are over-weighted. The bucket is large
#: enough that the trade is worth it; a thinner one would not be.
#:
#: Doubtful goes the other way and IS an absence, on the evidence: it never
#: once produced a player. Five observations is thin, but the estimate is 0.0
#: and the failure mode of being wrong is a suppressed player rather than a
#: doubled one.
#:
#: "Injured reserve" is deliberately absent as a category. The league never
#: writes it - the phrase appears in zero reason strings across a full season -
#: because an IR player is filed as Out like anyone else. IR is therefore
#: already inside the out set, and naming it separately would imply a
#: distinction the source does not make.
ABSENT = "absent"
AVAILABLE = "available"
UNKNOWN = "unknown"

#: Official designation -> state. Mirrored into
#: ``config.PLAYER_EPM_STATUS_TREATMENT``; kept here because this is the module
#: that defines the vocabulary, and a mapping living only in config invites it
#: to drift away from the source it describes.
DESIGNATION_STATE = {
    "out": ABSENT,
    "doubtful": ABSENT,
    "recovery": ABSENT,
    "questionable": AVAILABLE,
    "available": AVAILABLE,
    "probable": AVAILABLE,
}


def availability_state(designation: str) -> str:
    """The state a designation implies, or ``unknown`` for anything else.

    ``unknown`` rather than a default of ``available``. A designation this
    module has never heard of is not evidence of fitness, and defaulting it to
    healthy is how an expanding vocabulary quietly becomes a no-op filter -
    indistinguishable from a league where nobody is hurt.
    """
    return DESIGNATION_STATE.get((designation or "").strip().lower(), UNKNOWN)

#: A real report is tens of kilobytes. The 403 body from this host is 243
#: bytes of XML, so the size floor separates "no such report" from "report".
MIN_REPORT_BYTES = 5_000

#: The page footer straddles two columns ("Page 4" | "of 11"), which puts the
#: first half in the team column unless the whole line is dropped.
_FOOTER_RE = re.compile(r"^(Page\s*\d+|of\s*\d+)$", re.I)
#: The title line sits across the team/player/status columns. It has to be
#: matched on its words, not on ``Report:``, because the character-level
#: rebuild inserts a space before the colon (``"Injury Report :"``) whenever
#: the glyphs are drawn with a gap - which is how a document title ends up
#: masquerading as a club name.
_TITLE_RE = re.compile(r"Injury\s+Report", re.I)
_PUBLISHED_RE = re.compile(
    r"Injury Report:\s*\S+\s+(\d{1,2}:\d{2}\s*[AP]M)")
_MATCHUP_RE = re.compile(r"^[A-Z]{2,4}@[A-Z]{2,4}$")

#: Header words that anchor a column, matched by prefix because the league
#: has published them both glued and split - ``GameDate`` in 2026,
#: ``Game`` ``Date`` in 2021.
_HEADER_PREFIXES = (("gam", "game"), ("matchup", "matchup"), ("team", "team"),
                    ("player", "player"), ("current", "status"),
                    ("status", "status"), ("reason", "reason"))


def _header_key(text: str) -> str | None:
    low = text.lower()
    for prefix, key in _HEADER_PREFIXES:
        if low.startswith(prefix):
            return key
    return None


def report_stamp(when: datetime) -> str:
    """The filename stamp for a publication moment.

    Hourly before 2025-12-22, quarter-hourly after, with a hyphen. All three
    details are load-bearing and all three fail as a silent 403.
    """
    hour12 = when.hour % 12 or 12
    suffix = "AM" if when.hour < 12 else "PM"
    day = f"{when:%Y-%m-%d}"
    if when >= QUARTER_HOUR_FROM:
        return f"{day}_{hour12:02d}_{when.minute:02d}{suffix}"
    return f"{day}_{hour12:02d}{suffix}"


def report_url(when: datetime) -> str:
    return REPORT_URL.format(stamp=report_stamp(when))


@dataclass(frozen=True)
class Designation:
    """One player, one game, one filing."""

    game_date: date
    game_time_et: str
    matchup: str
    team: str
    player: str
    status: str
    reason: str
    published_at: datetime

    @property
    def away(self) -> str:
        return self.matchup.split("@", 1)[0]

    @property
    def home(self) -> str:
        return self.matchup.split("@", 1)[1]


def tipoff_et(day: date, game_time_et: str) -> datetime:
    """Resolve the report's ``07:00 (ET)`` to a real datetime.

    The report prints a 12-hour clock with no meridiem. The NBA plays no
    morning games, so anything before noon is an afternoon tipoff. Reading
    ``07:00`` as 07:00 instead of 19:00 silently compares tipoff against the
    wrong end of the day's filings and finds no pre-tipoff report at all -
    which is how a whole slate looks like it has no history.
    """
    hh, mm = (int(part) for part in game_time_et.split()[0].split(":"))
    if hh < 12:
        hh += 12
    return datetime(day.year, day.month, day.day, hh, mm)


#: Minimum spacing between NETWORK attempts (cache hits never wait). The
#: host answers a cold or rapid client with 403 - which this module cannot
#: distinguish from "no such report" - and a 2024-25 backfill chunk that
#: hammered it came back silently empty for 15 months of games while every
#: other window succeeded. Pacing costs seconds; an unpaced run costs the
#: season.
FETCH_PAUSE_SEC = 0.3
_last_network_at = 0.0

#: 403s seen by this process - surfaced so a rate-limited run can say so
#: instead of reporting quiet emptiness.
http_403_count = 0


def _now() -> datetime:
    """This run's moment as a naive Eastern stamp.

    Every stamp walked in this module is naive Eastern - submission
    cutoffs are converted to ``_EASTERN`` and stripped - so the bound
    that decides what can exist yet must be naive Eastern too. A naive
    host-local now would be off by the UTC offset on the UTC production
    host and let a walk probe hours into the Eastern future it exists to
    refuse.
    """
    return datetime.now(_EASTERN).replace(tzinfo=None)


def fetch_report(when: datetime, cache_dir: Path,
                 timeout: int = 30) -> Path | None:
    """Download one filing, or return ``None`` if the league never made it.

    ``None`` means "no such report", which is a normal answer: a league that
    played one game at 7pm files a handful of times, not once per quarter
    hour. It is deliberately not an error - but see ``http_403_count``:
    a rate-limit 403 answers the same way, so a run whose counter ends high
    must not be trusted as complete.

    A stamp at or after this run's moment is answered ``None`` WITHOUT a
    request: the league has not filed it yet, and the host answers the ask
    with a 403 indistinguishable from a blocked archive (measured
    2026-10-04: a pending slate walked 16 days into its own future and
    rate-limited the run).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"Injury-Report_{report_stamp(when)}.pdf"
    if path.exists() and path.stat().st_size >= MIN_REPORT_BYTES:
        return path
    if when > _now():
        # Not published yet, therefore not fetchable: no request, no
        # pacing, no 403. This is the single guard every walk funnels
        # through - the submission grid, both backfills, discovery and
        # the tip-off walk all reach the host here, so a pending slate
        # stops probing without any walker needing to know the time.
        return None
    global _last_network_at, http_403_count
    if FETCH_PAUSE_SEC:
        waited = _time.monotonic() - _last_network_at
        if waited < FETCH_PAUSE_SEC:
            _time.sleep(FETCH_PAUSE_SEC - waited)
    _last_network_at = _time.monotonic()
    request = urllib.request.Request(report_url(when),
                                     headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            http_403_count += 1
            if http_403_count == 1:
                logger.warning("first 403 from the injury-report host "
                               "(%s); counting - a rate-limited run looks "
                               "like an empty archive", when)
            return None
        if exc.code == 404:
            return None
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("injury report %s unavailable: %s",
                       report_stamp(when), exc)
        return None
    if len(body) < MIN_REPORT_BYTES:
        return None
    path.write_bytes(body)
    return path


def _line_text(chars: Sequence[dict]) -> str:
    """Rebuild a line's text from character geometry.

    The PDF positions words with gaps rather than emitting space characters,
    so ``Conley,`` and ``Mike`` arrive as a single token and a naive join
    yields ``Conley,Mike`` for every name in the league. Spacing has to be
    recovered from the geometry.
    """
    out: list[str] = []
    prev_x1: float | None = None
    for ch in chars:
        if prev_x1 is not None and ch["x0"] - prev_x1 > 0.22 * (ch["size"] or 10):
            out.append(" ")
        out.append(ch["text"])
        prev_x1 = ch["x1"]
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _calibrate(page) -> dict | None:
    """Read the column left edges off this document's own header row.

    The layout is not stable across eras: 2021 puts the team column at x=243.9
    and 2026 at x=264.2, and the header words are glued in one era and split
    in the other. Calibrating from the document means a future format change
    moves the columns instead of quietly returning the wrong cells.
    """
    words = page.extract_words()
    anchors = [w for w in words if _header_key(w["text"]) == "game"]
    if not anchors:
        return None
    top = min(w["top"] for w in anchors)
    row = [w for w in words if abs(w["top"] - top) < 4.0]
    games = sorted(w["x0"] for w in row if _header_key(w["text"]) == "game")
    edges: dict = {}
    if len(games) >= 2:
        edges["game_date"], edges["game_time"] = games[0], games[1]
    for w in row:
        key = _header_key(w["text"])
        if key and key != "game" and key not in edges:
            edges[key] = w["x0"]
    if len(edges) < 7:
        return None
    return edges


_FIELDS = ("game_date", "game_time", "matchup", "team", "player", "status",
           "reason")


def _page_cells(page, edges: dict) -> Iterator[dict]:
    """Bucket a page's characters into lines, then into columns.

    Lines are grouped by vertical position and words are placed in the column
    whose left edge they sit furthest to the right of. The title and the
    straddled footer are blanked rather than dropped, because a blanked line
    cannot leak ``Page 4`` into the team column.
    """
    rows: dict[int, list[dict]] = {}
    for ch in page.chars:
        rows.setdefault(round(ch["top"] / 3.0), []).append(ch)
    for key in sorted(rows):
        chars = sorted(rows[key], key=lambda c: c["x0"])
        spans: dict[str, list[dict]] = {name: [] for name in _FIELDS}
        for ch in chars:
            column = _FIELDS[0]
            for name in _FIELDS:
                if ch["x0"] >= edges[name] - 2.0:
                    column = name
            spans[column].append(ch)
        cells = {name: (_line_text(spans[name]) if spans[name] else "")
                 for name in _FIELDS}
        flat = " ".join(v for v in cells.values() if v)
        if (_TITLE_RE.search(flat)
                or any(_FOOTER_RE.match(v) for v in cells.values())):
            yield {name: "" for name in _FIELDS}
            continue
        yield cells


def _install_pinned_parser() -> str:
    """One attempt to install the pinned parser from requirements.txt.

    Returns an empty string on success, else a short reason for the
    caller's error message. This exists because the Kaggle runner
    executes a hand-kept copy of the notebook: on 2026-10-04 14:49 that
    copy still predated the fix, the environment had no pdfplumber, and
    the guard below turned a stale notebook into a red run that shipped
    nothing. Repairing from the pinned requirements first means a stale
    runner still produces a correct run, while a genuinely
    unrepairable environment fails just as loudly as before.
    """
    import subprocess

    requirements = Path(__file__).with_name("requirements.txt")
    if not requirements.exists():
        return f"no requirements.txt beside {Path(__file__).name}"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "--disable-pip-version-check", "-r", str(requirements)],
            capture_output=True, text=True, timeout=600)
    except Exception as exc:  # pip unusable, offline, or timed out
        return f"pip could not run: {exc!r}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return f"pip exited {proc.returncode}: {tail}"
    importlib.invalidate_caches()
    return ""


def require_pdf_parser() -> None:
    """Fail fast when the filing parser (``pdfplumber``) is absent.

    ``parse_report`` imports it lazily, so a runner without it survives
    every import and then loses each filing one warning at a time: the
    2026-10-04 Kaggle run logged 696 ``unparseable`` filings, every
    pending date fell back to the archive, and a slate date past the
    archive would have shipped UNFILTERED while the run still reported
    ok. Entrypoints that resolve designations call this at second zero
    so a missing parser is one clear error, not a silent hole in the
    point-in-time methodology.

    Before raising, it tries the pinned install once, so a runner whose
    notebook drifted from requirements.txt (2026-10-04 14:49) repairs
    itself instead of only failing.
    """
    if importlib.util.find_spec("pdfplumber") is not None:
        return
    reason = _install_pinned_parser()
    importlib.invalidate_caches()
    if importlib.util.find_spec("pdfplumber") is not None:
        return
    detail = f" (automatic install failed: {reason})" if reason else ""
    raise RuntimeError(
        "pdfplumber is not installed: the league's game-day injury "
        "filings cannot be parsed, so availability cannot be resolved "
        "point in time (the 2026-10-04 run lost every filing this way "
        "and still reported ok)" + detail + ". Install the pinned "
        "parser: pip install -r nba-backend/backend/requirements.txt")


def parse_report(path: Path, published_at: datetime) -> list[Designation]:
    """One filing into the designations it carries.

    Only page one carries a column header; the rest of the document reuses
    the geometry, so the calibration is taken once and held. A row is a new
    record exactly when it has a designation, which is what makes a wrapped
    reason (a continuation line carrying no designation) attach to the record
    above it instead of becoming one.
    """
    import pdfplumber  # imported lazily: the parser is the only heavy path

    out: list[Designation] = []
    game_date = game_time = matchup = None
    team = None
    pending: dict | None = None
    with pdfplumber.open(str(path)) as pdf:
        edges = None
        for page in pdf.pages:
            if edges is None:
                edges = _calibrate(page)
                if edges is None:
                    continue
                header_top = min(
                    w["top"] for w in page.extract_words()
                    if _header_key(w["text"]) == "game")
            for cells in _page_cells(page, edges):
                if cells["game_date"]:
                    game_date = cells["game_date"]
                if cells["game_time"]:
                    game_time = cells["game_time"]
                if cells["matchup"]:
                    matchup = cells["matchup"]
                if cells["team"]:
                    team = cells["team"]
                if cells["status"] in DESIGNATIONS:
                    if pending is not None:
                        out.append(_build(pending, published_at))
                    player = cells["player"]
                    if team and player.startswith(team):
                        # a long team name shares the row with the player
                        player = player[len(team):].strip()
                    pending = dict(game_date=game_date, game_time=game_time,
                                   matchup=matchup, team=team, player=player,
                                   status=cells["status"],
                                   reason=cells["reason"].strip())
                elif pending is not None and cells["reason"] and not cells["player"]:
                    pending["reason"] = (pending["reason"] + " "
                                         + cells["reason"]).strip()
                elif pending is not None and cells["player"] and not cells["status"]:
                    # a player name that wrapped onto its own line
                    pending["player"] = (cells["player"] + " "
                                         + pending["player"]).strip()
    if pending is not None:
        out.append(_build(pending, published_at))
    return out


def _build(row: dict, published_at: datetime) -> Designation:
    raw_date = row["game_date"]
    try:
        month, day, year = (int(p) for p in raw_date.split("/"))
    except ValueError:
        return None
    if year < 100:
        year += 2000
    matchup = row["matchup"] or ""
    return Designation(
        game_date=date(year, month, day),
        game_time_et=row["game_time"] or "",
        matchup=matchup if _MATCHUP_RE.match(matchup) else "",
        team=row["team"] or "",
        player=row["player"] or "",
        status=row["status"],
        reason=row["reason"] or "",
        published_at=published_at,
    )


def pre_game_designations(day: date, cache_dir: Path,
                          how_far_back: int = 24
                          ) -> dict[str, list[Designation]]:
    """Designations as of tipoff, keyed by ``AWAY@HOME``.

    The league files every report for the whole slate, so a report for a 7pm
    game also carries the 10pm games. Only the rows whose OWN game has not yet
    tipped are kept: a 7pm game takes its answer from the last filing before
    7pm, and a 10pm game from the last filing before 10pm. Taking one filing
    for the whole slate would hand the early games an answer from after their
    tipoff - the lookahead leak, in its most obvious form.

    Filings are fetched by walking BACK from each tipoff rather than sweeping
    the whole day. A sweep is 72 requests per date and needs at most a handful
    per game, which over a season is the difference between 12,000 requests
    and about 2,000 - and the walk is also strictly safer, because every filing
    it can return is before the tipoff by construction rather than by a filter
    applied afterwards.
    """
    slate = _discover_slate(day, cache_dir, how_far_back)
    if not slate:
        return {}
    designations: dict[str, list[Designation]] = {}
    parsed: dict[datetime, list[Designation]] = {}
    for matchup, tip in sorted(slate.items(), key=lambda kv: kv[1]):
        filing = _last_filing_before(tip, cache_dir, how_far_back)
        if filing is None:
            logger.warning("no filing before %s tipoff for %s on %s",
                           tip, matchup, day)
            continue
        if filing not in parsed:
            parsed[filing] = parse_report(cache_dir / _stamp_path(filing), filing)
        bucket = designations.setdefault(matchup, [])
        for record in parsed[filing]:
            if record.matchup != matchup:
                continue
            if record.published_at >= tip:
                continue          # not knowable at this tipoff
            bucket.append(record)
    return designations


def _stamp_path(when: datetime) -> str:
    return f"Injury-Report_{report_stamp(when)}.pdf"


def _steps(day: date, tip: datetime, how_far_back: int) -> Iterator[datetime]:
    """Quarter-hour moments walking back from ``tip``, exclusive of it."""
    step = 15 if day >= QUARTER_HOUR_FROM.date() else 60
    current = tip - timedelta(minutes=step)
    for _ in range(how_far_back):
        if current.date() < day:
            return
        yield current
        current -= timedelta(minutes=step)


def _last_filing_before(tip: datetime, cache_dir: Path,
                        how_far_back: int, timeout: int = 30
                        ) -> datetime | None:
    for when in _steps(tip.date(), tip, how_far_back):
        if fetch_report(when, cache_dir, timeout=timeout) is not None:
            return when
    return None


def _grid_minutes(day: date) -> int:
    """The filing grid: 15 minutes after 2025-12-22, hourly before it."""
    return 15 if day >= QUARTER_HOUR_FROM.date() else 60


#: Where discovery probes for a day's first filing.  Ordered so the hit
#: lands NEAR the submission window whenever the league is filing - the
#: tip-off times a postponed game moved are more likely reflected there
#: than in the morning's first snapshot.  On a game day the league files
#: continuously from morning through tip-off, so one probe in this spread
#: always hits; on a day with no games it files NOTHING (measured:
#: 2024-06-10, a Finals off-day, has no filing at any hour), so probing
#: every stamp - 96 of them in the quarter-hour era - would burn tens of
#: thousands of requests per two-year backfill on days that cannot
#: contribute a row.
_SEED_PROBE_HOURS = (12, 10, 14, 8, 16, 6, 18, 20, 22)


def _discover_slate(day: date, cache_dir: Path,
                    how_far_back: int) -> dict[str, datetime]:
    """Which games are on ``day``, and when they tip - filing-based.

    The slate is read out of filings rather than out of a schedule we hold,
    because a filing is the only source that agrees with the reports about
    what time a game starts.  But ONE filing is not the whole slate: a game
    whose clubs had nothing filed at the seed hour is absent from the seed
    (measured: PHI@IND on 2026-04-10 first appears in the 1:00 PM filing,
    after both clubs' game-day submissions landed), so the slate is the
    UNION across a spread of the day's filings, first-seen tip kept.

    Callers that hold a schedule should pass ``games=`` to
    :func:`game_day_designations` instead and skip discovery entirely - the
    schedule-driven path has no single-filing dependency at all.
    """
    seed = None
    for hour in _SEED_PROBE_HOURS:
        when = datetime(day.year, day.month, day.day, hour, 0)
        if fetch_report(when, cache_dir) is not None:
            seed = when
            break
    if seed is None:
        return {}
    slate: dict[str, datetime] = {}
    absorbed: set[datetime] = set()

    def _absorb(when: datetime) -> None:
        if when in absorbed:
            return
        absorbed.add(when)
        for record in parse_report(cache_dir / _stamp_path(when), when):
            if not record.matchup or not record.game_time_et:
                continue
            if record.game_date != day:
                continue
            slate.setdefault(record.matchup,
                             tipoff_et(day, record.game_time_et))

    _absorb(seed)
    # Three anchors across the rest of the day: after the standard window
    # close, after the western close, and the evening - so a game listed
    # late (or a matinee already gone from the evening filing) still makes
    # the slate. First-seen tip wins, so the seed's times are authoritative
    # and these only ADD games.
    for hour in (13, 16, 19):
        when = datetime(day.year, day.month, day.day, hour, 0)
        if when == seed:
            continue
        if fetch_report(when, cache_dir) is not None:
            _absorb(when)
    return slate


# ---------------------------------------------------------------------------
# THE GAME-DAY SUBMISSION - the moment availability is read at
# ---------------------------------------------------------------------------
#
# The league's reporting policy (official.nba.com) has three beats:
#
#   * by 5 p.m. local the day before a game (1 p.m. local for the second
#     night of a back-to-back), every player whose participation may be
#     affected carries a status and a reason;
#   * on game day each club SUBMITS its game-day injury report between
#     11 a.m. and 1 p.m. local time - between 8 and 10 a.m. for tip-offs
#     5 p.m. or earlier;
#   * reports then "update on a continual basis" up to tip-off.
#
# The third beat is where late scratches live, and they were measured,
# not assumed.  On 2026-01-10 (quarter-hour era) Cade Cunningham and
# Isaiah Stewart both went Questionable -> Out at 18:15 ET for 19:00 tips,
# and Pat Connaughton at 18:15; on 2025-01-15 (hourly era) Karl-Anthony
# Towns went Questionable -> Doubtful (17:00) -> Out (18:00 ET), Lonzo Ball
# -> Out at 18:00, and Kyrie Irving Doubtful -> Out at 19:00.  All real,
# all unknowable to a slate priced at the submission moment.
#
# Meanwhile the SAME days' mid-day filings carry the game-day submissions
# themselves: on 2026-01-10 Edwards and Randle cleared at 11:15-11:45 ET,
# and Jusuf Nurkic went Questionable -> Out at 14:30 ET = 12:30 MT - inside
# Utah's 11 a.m.-1 p.m. local window.  That is the information the slate
# has: the posted game-day report, plus the projected lineups derived from
# it.  Nothing after.
#
# So availability for a game is read at the SUBMISSION:
#
#     availability(game) = state of the first published filing at/after
#                          the game-day submission window CLOSE,
#                          strictly before tip-off
#
# and BOTH the historical frame and the prediction slate resolve it with
# the SAME function below.  A filing published between the close and
# tip-off (a late scratch) never reaches ``status``; it is kept only in
# ``status_tipoff`` so the size of the post-submission movement stays
# measurable instead of quietly shaping the feature.

#: Arena timezones by club abbreviation.  The submission window is written
#: in each club's LOCAL time, and every US zone differs by whole hours, so
#: a local window close lands exactly on the filing grid (hourly or
#: quarter-hour) in ET - the selection below is therefore deterministic.
#: PHX is America/Phoenix: Arizona ignores DST, which is exactly the kind
#: of detail that silently shifts a cutoff by an hour in June if fudged.
ARENA_TIMEZONE: dict[str, str] = {
    "ATL": "America/New_York", "BOS": "America/New_York",
    "BKN": "America/New_York", "CHA": "America/New_York",
    "CHI": "America/Chicago", "CLE": "America/New_York",
    "DET": "America/New_York", "IND": "America/New_York",
    "MIA": "America/New_York", "MIL": "America/Chicago",
    "NYK": "America/New_York", "ORL": "America/New_York",
    "PHI": "America/New_York", "TOR": "America/New_York",
    "WAS": "America/New_York",
    "DAL": "America/Chicago", "HOU": "America/Chicago",
    "MEM": "America/Chicago", "MIN": "America/Chicago",
    "NOP": "America/Chicago", "OKC": "America/Chicago",
    "SAS": "America/Chicago",
    "DEN": "America/Denver", "UTA": "America/Denver",
    "PHX": "America/Phoenix",
    "GSW": "America/Los_Angeles", "LAC": "America/Los_Angeles",
    "LAL": "America/Los_Angeles", "POR": "America/Los_Angeles",
    "SAC": "America/Los_Angeles",
}

#: The report's timestamps and tip-offs are naive-ET.  Every conversion
#: goes through this zone so DST is applied by the calendar, not by a
#: fixed offset.
_EASTERN = ZoneInfo("America/New_York")

#: Tips at/before 5 p.m. local use the morning window; later tips use the
#: mid-day window.  Straight from the policy sentence - two windows, no
#: interpolation.
EARLY_TIP_LOCAL = time(17, 0)
EARLY_CLOSE_LOCAL = time(10, 0)
STANDARD_CLOSE_LOCAL = time(13, 0)

#: How the chosen filing related to the window - recorded per row so a
#: coverage report can say HOW a game was resolved, not just that it was.
PROVENANCE_SUBMISSION = "game_day_submission"
PROVENANCE_PRE_WINDOW = "pre_window_fallback"

DESIGNATION_COLUMNS = [
    "gameday", "matchup", "team", "team_full", "player", "player_report",
    "status", "reason", "published_at", "cutoff_at", "tipoff_at",
    "provenance", "status_tipoff", "published_at_tipoff",
]

@dataclass(frozen=True)
class SubmissionWindow:
    """When a game's game-day submission window closes, in ET."""

    game_date: date
    home: str
    tipoff: datetime          # naive ET
    cutoff: datetime          # naive ET: the window close the state is read at
    early_tip: bool

    @property
    def window(self) -> str:
        return ("08:00-10:00 local" if self.early_tip
                else "11:00-13:00 local")


def _arena_zone(home: str) -> ZoneInfo:
    name = ARENA_TIMEZONE.get(str(home or "").strip().upper())
    if name is None:
        # Degrade to Eastern rather than skip: an unknown abbreviation must
        # not cost a game its availability.  Eastern is also the
        # conservative direction for a club we mis-zone - a western home
        # treated as Eastern closes the window EARLIER (1 p.m. ET instead
        # of 1 p.m. PT), so the resolver reads an older filing and never a
        # newer one.  The warning says which clubs were guessed.
        logger.warning("no arena timezone for %r; treating home as ET",
                       home)
        return ZoneInfo("America/New_York")
    return ZoneInfo(name)


def submission_window(game_date: date, tipoff: datetime,
                      home: str) -> SubmissionWindow:
    """The league's game-day submission window for one game.

    Teams submit between 11 a.m. and 1 p.m. LOCAL time on game day (8-10
    a.m. for tips 5 p.m. or earlier local), so the cutoff is the window's
    close converted to ET.  Both edges are read off this game's own tip-off
    in the home club's own zone - a 7 p.m. ET tip closes at 1 p.m. ET in
    Boston, at 1 p.m. MT in Utah, at 1 p.m. PT in Sacramento, and a noon
    matinee closed at 10 a.m. local wherever it is played.
    """
    zone = _arena_zone(home)
    local_tip = tipoff.replace(tzinfo=_EASTERN).astimezone(zone)
    early = local_tip.time() <= EARLY_TIP_LOCAL
    close_local = datetime.combine(
        local_tip.date(),
        EARLY_CLOSE_LOCAL if early else STANDARD_CLOSE_LOCAL,
        tzinfo=zone)
    cutoff = close_local.astimezone(_EASTERN).replace(tzinfo=None)
    return SubmissionWindow(game_date=game_date, home=str(home or ""),
                            tipoff=tipoff, cutoff=cutoff, early_tip=early)


def submission_filing(window: SubmissionWindow, cache_dir: Path,
                      how_far_back: int = 72, timeout: int = 30
                      ) -> "tuple[datetime, str] | None":
    """The filing that carries this game's game-day submission.

    FORWARD first: the first published filing at/after the window close and
    strictly before tip-off.  The league merges a submission as it arrives,
    so the first snapshot after the close is the earliest state that is
    guaranteed to contain every club's submission - and anything published
    after it is post-submission news (the late scratch band), which is
    exactly what this cutoff exists to exclude.

    Then the honest fallback: no filing exists between close and tip-off
    (a sparse early era, an unpublished stretch), so take the last filing
    strictly BEFORE the close.  Still strictly PIT - every candidate is
    older than the close - and flagged ``pre_window_fallback`` per row so a
    coverage report can count how often the league gave us nothing better.
    ``None`` means no filing at all was found: the game is UNCOVERED, which
    callers report rather than fill with "nobody is hurt".

    ``min(cutoff, tipoff)`` bounds BOTH directions: on a pathological parse
    where the computed cutoff lands after tip-off, the fallback must still
    never read a filing at/after tip-off.
    """
    bound = min(window.cutoff, window.tipoff)
    step = _grid_minutes(window.tipoff.date())
    when = window.cutoff
    # Aligned to the grid (window closes land on the hour; both grids
    # include :00), but ceil anyway so a half-hour zone can never start
    # the walk between two stamps and skip the first one.
    when = when.replace(second=0, microsecond=0)
    if when.minute % step:
        when += timedelta(minutes=step - (when.minute % step))
    while when < window.tipoff:
        if fetch_report(when, cache_dir, timeout=timeout) is not None:
            return when, PROVENANCE_SUBMISSION
        when += timedelta(minutes=step)
    # Pre-window fallback: walk back from the bound, crossing midnight so
    # the day-before 5 p.m. report (the base every slate stands on) is
    # reachable.
    current = bound - timedelta(minutes=step)
    for _ in range(how_far_back):
        if fetch_report(current, cache_dir, timeout=timeout) is not None:
            return current, PROVENANCE_PRE_WINDOW
        current -= timedelta(minutes=step)
    return None


def canonical_name(report_name: str) -> str:
    """"Last, First" from the report -> "First Last" as the box score writes it.

    A contains-test is not safe here: 'Harris' sits inside 'Harrison' and
    'Paul' is a first name, so substring matching invents violations that
    look like source errors.
    """
    name = report_name or ""
    if "," not in name:
        return name
    last, first = name.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def _team_abbreviations(records: Sequence[Designation],
                        matchup: str) -> dict[str, str]:
    """Full club names as printed -> the matchup's abbreviations.

    The report prints the away block before the home block, so a full club
    name's abbreviation is read off the order of first appearance - and
    then CHECKED: a full name resolving to two different abbreviations on
    different nights means the assumption broke, which is logged rather
    than zip-truncated into a wrong club.
    """
    abbrs = matchup.split("@", 1)
    order: list[str] = []
    for record in records:
        if record.team and record.team not in order:
            order.append(record.team)
    mapping: dict[str, str] = {}
    for index, full in enumerate(order):
        if index >= len(abbrs):
            break
        prior = mapping.get(full)
        if prior is not None and prior != abbrs[index]:
            logger.warning("team order conflict: %r seen as %s and %s (%s)",
                           full, prior, abbrs[index], matchup)
            continue
        mapping[full] = abbrs[index]
    return mapping


def _empty_designations() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object")
                         for c in DESIGNATION_COLUMNS})


def game_day_designations(day: date, cache_dir: Path,
                          games: "Iterable[tuple[str, datetime]] | None" = None,
                          how_far_back: int = 72,
                          timeout: int = 30) -> "pd.DataFrame":
    """Availability for every game on ``day``, read at its submission.

    THE canonical resolver: the historical frame and the prediction slate
    both call this, with the same cutoff rule and the same source, so the
    two information sets are identical by construction rather than by
    review.  It never reads box scores and never reads a filing published
    after the window close into ``status`` - a late scratch is knowable
    only in ``status_tipoff``, which nothing that gates a pool reads.

    ``games`` - an optional schedule for the day as ``(matchup, tipoff)``
    pairs (the pipeline's ``start_time_utc`` converted to ET).  When the
    caller holds a schedule it is the game list, period: no single filing
    decides which games exist, which is the seed dependency that silently
    dropped games whose clubs had not filed by the seed hour.  When
    ``games`` is None the slate falls back to the filing-based union in
    :func:`_discover_slate`.

    One row per (game, player) UNION of the submission filing and the last
    filing before tip-off:

    * ``status``            - at the game-day submission (the feature input);
    * ``status_tipoff``     - at the last pre-tip-off filing (measurement
                              only: the late-scratch delta stays countable);
    * ``provenance``        - ``game_day_submission`` or ``pre_window_fallback``;
    * ``published_at``/``cutoff_at``/``tipoff_at`` - every timestamp needed
                              to re-derive or audit the choice.

    A player who appears ONLY in the tip-off filing (added to the report
    after the close) carries ``status=""``: not knowable at submission, so
    it removes nothing, while the fact that he was added stays visible in
    ``status_tipoff``.
    """
    if games is None:
        slate = _discover_slate(day, cache_dir, how_far_back)
    else:
        slate = {}
        for matchup, tipoff in games:
            if matchup and tipoff is not None:
                slate[str(matchup)] = tipoff
    if not slate:
        return _empty_designations()
    parsed: dict[datetime, list[Designation]] = {}

    def records_at(when: datetime) -> list[Designation]:
        if when not in parsed:
            try:
                parsed[when] = parse_report(cache_dir / _stamp_path(when),
                                            when)
            except Exception as exc:  # noqa: BLE001 - a bad filing is a gap
                logger.warning("filing %s unparseable (%s); its games are "
                               "not covered by it", _stamp_path(when), exc)
                parsed[when] = []
        return parsed[when]

    rows: list[dict] = []
    for matchup, tipoff in sorted(slate.items(), key=lambda kv: kv[1]):
        home = matchup.split("@", 1)[1] if "@" in matchup else ""
        window = submission_window(day, tipoff, home)
        # A filing at/after this run's moment does not exist yet, so a
        # game whose submission window has not closed is PRE-SUBMISSION,
        # not uncovered: "uncovered" says the archive failed this game,
        # and on a pending slate that is never true - nothing could have
        # been filed (2026-10-04: three 2026-10-20 games warned
        # "uncovered" while probing stamps 16 days in the future).
        pre_submission = window.cutoff > _now()
        chosen = submission_filing(window, cache_dir, how_far_back,
                                   timeout=timeout)
        if chosen is None:
            if pre_submission:
                logger.info("no filing for %s on %s: pre-submission "
                            "(window closes %s) - no designations yet",
                            matchup, day, window.cutoff)
            else:
                logger.warning("no filing for %s on %s: game uncovered",
                               matchup, day)
            continue
        filing_at, provenance = chosen
        submission_records = [r for r in records_at(filing_at)
                              if r.matchup == matchup and r.game_date == day
                              and r.published_at < tipoff]
        if not submission_records:
            # The first snapshot at/after the close can predate this
            # game's inclusion: clubs file up TO the close, the league
            # merges as they arrive, and a sparse filing (a day-before
            # base, a "NOT YET SUBMITTED" placeholder) can parse to no
            # rows at all.  This game's submission is the first filing -
            # still strictly before tip-off - that actually carries it,
            # so walk the grid forward from the chosen stamp.  A game
            # found this way keeps the same cutoff and the same
            # provenance rule (at/after close = game_day_submission); a
            # game never listed before tip-off is reported uncovered
            # instead of silently dropped.
            step = _grid_minutes(day)
            when = filing_at.replace(second=0, microsecond=0)
            while True:
                when += timedelta(minutes=step)
                if when >= tipoff:
                    break
                if fetch_report(when, cache_dir, timeout=timeout) is None:
                    continue
                found = [r for r in records_at(when)
                         if r.matchup == matchup and r.game_date == day
                         and r.published_at < tipoff]
                if found:
                    filing_at = when
                    provenance = (PROVENANCE_SUBMISSION
                                  if when >= window.cutoff
                                  else PROVENANCE_PRE_WINDOW)
                    submission_records = found
                    break
        if not submission_records:
            # Nothing in [close, tipoff) carries the game with player rows
            # - the league printed its morning state and then dropped the
            # rows (2025-01-11) or the band's filings omit it entirely.
            # Mirror submission_filing's own fallback: the last filing
            # STRICTLY BEFORE the close that does list it.  Still strictly
            # PIT - every candidate is older than the close - and flagged
            # pre_window_fallback per row so the coverage report counts
            # exactly how often the league gave nothing better.  A game
            # never listed at all stays uncovered: availability is never
            # invented from silence.
            step = _grid_minutes(day)
            current = min(window.cutoff,
                          window.tipoff) - timedelta(minutes=step)
            for _ in range(how_far_back):
                if fetch_report(current, cache_dir,
                                timeout=timeout) is not None:
                    found = [r for r in records_at(current)
                             if r.matchup == matchup and r.game_date == day
                             and r.published_at < tipoff]
                    if found:
                        filing_at = current
                        provenance = PROVENANCE_PRE_WINDOW
                        submission_records = found
                        break
                current -= timedelta(minutes=step)
        if not submission_records:
            if pre_submission:
                logger.info("no filing lists %s before tipoff on %s: "
                            "pre-submission - no designations yet",
                            matchup, day)
            else:
                logger.warning("no filing lists %s before tipoff on %s: "
                               "game uncovered", matchup, day)
            continue
        abbreviations = _team_abbreviations(submission_records, matchup)
        # The pre-tip-off filing feeds status_tipoff ONLY.  Shared across
        # games of the day through the same parse cache.
        tip_filing = _last_filing_before(tipoff, cache_dir, how_far_back,
                                         timeout=timeout)
        tip_statuses: dict[tuple[str, str], str] = {}
        if tip_filing is not None:
            tip_statuses = {
                (r.team, r.player): r.status
                for r in records_at(tip_filing)
                if r.matchup == matchup and r.game_date == day
                and r.published_at < tipoff}
        submission_keys = {(r.team, r.player) for r in submission_records}
        # Union: players added to the report AFTER the close exist at
        # tip-off only, with an empty submission status (removes nothing).
        tip_only = [r for r in records_at(tip_filing)
                    if r.matchup == matchup and r.game_date == day
                    and r.published_at < tipoff
                    and (r.team, r.player) not in submission_keys] \
            if tip_filing is not None else []
        for record in submission_records + tip_only:
            is_submission = (record.team, record.player) in submission_keys
            rows.append({
                "gameday": pd.Timestamp(day),
                "matchup": matchup,
                "team": abbreviations.get(record.team, ""),
                "team_full": record.team,
                "player": canonical_name(record.player),
                "player_report": record.player,
                "status": record.status if is_submission else "",
                "reason": record.reason if is_submission else "",
                "published_at": (filing_at.isoformat()
                                 if is_submission else ""),
                "cutoff_at": window.cutoff.isoformat(),
                "tipoff_at": tipoff.isoformat(),
                "provenance": provenance,
                "status_tipoff": (tip_statuses.get((record.team,
                                                    record.player), "")
                                  if is_submission else record.status),
                "published_at_tipoff": (tip_filing.isoformat()
                                        if tip_filing is not None else ""),
            })
    if not rows:
        return _empty_designations()
    return pd.DataFrame(rows, columns=DESIGNATION_COLUMNS)
