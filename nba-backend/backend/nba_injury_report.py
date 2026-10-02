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

import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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


def fetch_report(when: datetime, cache_dir: Path,
                 timeout: int = 30) -> Path | None:
    """Download one filing, or return ``None`` if the league never made it.

    ``None`` means "no such report", which is a normal answer: a league that
    played one game at 7pm files a handful of times, not once per quarter
    hour. It is deliberately not an error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"Injury-Report_{report_stamp(when)}.pdf"
    if path.exists() and path.stat().st_size >= MIN_REPORT_BYTES:
        return path
    request = urllib.request.Request(report_url(when),
                                     headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
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
                        how_far_back: int) -> datetime | None:
    for when in _steps(tip.date(), tip, how_far_back):
        if fetch_report(when, cache_dir) is not None:
            return when
    return None


def _discover_slate(day: date, cache_dir: Path,
                    how_far_back: int) -> dict[str, datetime]:
    """Which games are on ``day``, and when they tip.

    The slate is read out of a filing rather than out of the schedule, because
    the filing is the only source that agrees with the reports about what time
    a game starts.
    """
    probe_from = datetime(day.year, day.month, day.day, 12, 0)
    seed = None
    for when in _steps(day, probe_from, how_far_back):
        if fetch_report(when, cache_dir) is not None:
            seed = when
            break
    if seed is None:
        return {}
    slate: dict[str, datetime] = {}
    for record in parse_report(cache_dir / _stamp_path(seed), seed):
        if not record.matchup or not record.game_time_et:
            continue
        if record.game_date != day:
            continue
        slate[record.matchup] = tipoff_et(day, record.game_time_et)
    return slate
