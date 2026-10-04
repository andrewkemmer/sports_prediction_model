"""Point-in-time injury stints for NBA players.

The NBA analogue of MLB's ``build_il_stints.py``, and it inherits that module's
hardest-won lesson: an injury table that is merely *believed* will quietly
delete real players from projections, and the symptom is a feature that looks
healthy while being wrong.

What MLB learned, and what is reproduced here structurally:

* **STINTS, NOT STATUS.** A snapshot says "Out" today. It does not say when the
  absence started, and it does not say it ended. Every record is therefore
  stored with the moment it was published, and consecutive records are collapsed
  by walking a state machine per player - the same shape as MLB's, where one
  absence emits several placement records and only one activation.
* **RECONCILED AGAINST OBSERVED APPEARANCES.** The single most valuable rule in
  MLB's builder: a player with a player-game row was in the game, so no absence
  may span it. This is what turned a net-negative first A/B (3,541 player-games
  wrongly suppressed) into a safe feature, and it is what makes a missing or
  stale close harmless rather than catastrophic.
* **DEGRADE LOUDLY, NEVER SILENTLY.** A missing or unreadable table must warn
  and fall back to the unfiltered pool. It must never crash the run, and it must
  never quietly become a no-op filter, because an injury filter that does not
  bind is indistinguishable from a league where nobody is hurt.

THE POINT-IN-TIME RULE
----------------------
Availability for a game is decided using only records published STRICTLY BEFORE
that game's tipoff. This is enforced per record, on the record's own published
timestamp, and never by assuming a snapshot was taken at some convenient hour::

    out as of tipoff  <=>  exists stint with
                           il_start <  tipoff
                           and (il_end is NA or il_end >= tipoff)

Both edges are strict in the same direction, and getting either wrong admits
the future by exactly one record per game. On the near edge a status published
at the instant of tipoff is not knowable to anyone making a bet at that
instant, so it cannot gate that game - hence ``<``. On the far edge a recovery
published at that same instant is equally unknowable, so the player is still
out for it - hence ``>=`` rather than ``>``. This is the guardrail the whole
module exists to hold, and it is asserted directly by
``TestPointInTimeIsStrict``.

A SNAPSHOT IS ONLY A CARRIER
----------------------------
The roster endpoint publishes CURRENT state, so there is no history to backfill
and none of the past can be reconstructed from it. What makes that workable is
that each injury record carries the timestamp it was published, so a snapshot
taken at any cadence still yields records that know their own PIT floor. The
cadence decides how finely absences are resolved; the record date decides what
was knowable. Do not treat a snapshot's own date as the PIT floor - doing so is
the leak this module is built to prevent.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from backend import config
except ImportError:
    import config

#: Statuses that constitute an absence. Matched on the normalized vocabulary,
#: never by substring, because "ir" is a substring of ordinary text and ESPN's
#: vocabulary is a closed set this module maps explicitly.
_OUT_STATUSES = ("out",)
_PARTIAL_STATUSES = ("day_to_day",)

STINT_COLUMNS = ["player_id", "team", "il_start", "il_end", "status"]


def _as_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        # Everything here is compared against a naive tipoff, and mixing aware
        # and naive timestamps raises rather than coercing. Normalise to UTC
        # and drop the zone; the tipoff is converted the same way.
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def treatment_of(status: object) -> str:
    """``absent`` / ``partial`` / ``available`` / ``unknown`` for a status.

    The mapping is a TABLE, not a chain of substring tests, and a status the
    table does not list returns ``unknown`` rather than defaulting to
    ``available``. That direction is deliberate: defaulting an unrecognized
    word to available would project a player the feed said something about,
    while defaulting it to absent would suppress a healthy one. Neither is
    free, so the value is surfaced and counted instead of guessed - and
    ``play_rates`` reports any ``unknown`` bucket it finds, which is how a new
    vocabulary entry gets noticed before it starts mattering.
    """
    return config.PLAYER_EPM_STATUS_TREATMENT.get(
        availability_status_of(status), "unknown")


def observed_vocabulary(raw_statuses) -> dict:
    """Count the RAW status strings seen, so an unknown word is visible.

    Measured 2026-09-26 across 30 teams and 545 athletes, the NBA feed emits
    exactly two: "Day-To-Day" and "Out". "Doubtful", "Questionable" and
    "Probable" are NFL injury-report vocabulary and do not appear here, which
    is why asking what fraction of doubtful players play has no answer yet -
    there is nothing labelled doubtful to measure. This function is how that
    stays a fact rather than an assumption.
    """
    counts: dict = {}
    for value in raw_statuses or ():
        text = str(value or "").strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def play_rates(history, appearances) -> dict:
    """What fraction of players in each status actually played.

    This is the measurement that answers "do doubtful players play?", and it is
    the reason the history store exists. The roster endpoint publishes current
    state only, so without accumulated dated records there is nothing to join
    a status against and the question is not merely unanswered - it is
    UNANSWERABLE, and stays that way forever.

    ``history`` is the appended snapshot store; ``appearances`` is the
    player-game log. A status observed on snapshot date S is joined to whether
    that player appeared in a game on S. The result is reported with its sample
    size, because a rate computed from four observations is not a rate.

    Returns a ``sufficient`` flag rather than a bare number, so a caller cannot
    accidentally ship an estimate as if it were a finding.
    """
    report = {"sufficient": False, "by_status": {}, "min_sample": 20,
              "note": ""}
    if history is None or not len(history):
        report["note"] = ("no injury history recorded yet; the roster endpoint "
                          "publishes current state only, so nothing about the "
                          "past is recoverable")
        return report
    if appearances is None or not len(appearances):
        report["note"] = "no player-game log to join against"
        return report
    required = {"player_id", "status", "snapshot_date"}
    if not required.issubset(history.columns):
        report["note"] = f"history is missing {sorted(required - set(history.columns))}"
        return report

    played = appearances.copy()
    played["player_id"] = played.player_id.astype(str)
    played["gameday"] = pd.to_datetime(played.gameday, errors="coerce").dt.normalize()
    played_days = played.groupby("player_id").gameday.apply(set).to_dict()

    work = history.copy()
    work["player_id"] = work.player_id.astype(str)
    work["day"] = pd.to_datetime(work.snapshot_date, errors="coerce").dt.normalize()
    work = work[work.day.notna() & work.status.notna()]
    if work.empty:
        report["note"] = "history has no dated, classified records"
        return report

    buckets: dict = {}
    for record in work.to_dict("records"):
        status = str(record["status"])
        bucket = buckets.setdefault(status, {"played": 0, "total": 0})
        bucket["total"] += 1
        days = played_days.get(record["player_id"])
        if days and record["day"] in days:
            bucket["played"] += 1
    for status, bucket in sorted(buckets.items()):
        total = bucket["total"]
        report["by_status"][status] = {
            "played": bucket["played"], "total": total,
            "play_rate": (bucket["played"] / total) if total else None,
            "treatment": treatment_of(status),
            "sufficient": total >= report["min_sample"],
        }
    report["sufficient"] = any(
        entry["sufficient"] for entry in report["by_status"].values())
    if not report["sufficient"]:
        report["note"] = (f"no status has {report['min_sample']}+ observations "
                          f"yet; the rates above are not yet findings")
    return report


def is_absent(status: object) -> bool:
    """True when a normalized status means the player does not dress.

    A partial absence counts, for the same reason MLB's table is binary: the
    question it is asked is "is this player available to be projected into a
    lineup", and a day-to-day player is not reliably available. The partial
    status is preserved on the stint so ``play_rates`` can later measure how
    partial really is and replace this with a weight.
    """
    return treatment_of(status) in ("absent", "partial")


def availability_status_of(status: object) -> str:
    """Normalized status, reusing the source layer's closed vocabulary."""
    try:
        from backend import nba_sources as sources
    except ImportError:
        import nba_sources as sources
    return sources.availability_status(status)


def records_from_rosters(rosters) -> pd.DataFrame:
    """Flatten dated roster snapshots into one dated record per absence.

    ``rosters`` is an iterable of ``(snapshot_date, frame)`` where each frame
    carries ``player_id``/``team``/``status``/``raw_status``. Only ABSENT
    records are kept: a healthy observation is not an event, it is the absence
    of one, and storing it would make "no record" ambiguous between "healthy"
    and "never looked".

    ``published_at`` falls back to the snapshot date when the feed omitted a
    per-record timestamp, and that fallback is recorded in ``dated`` so a caller
    can tell a real publication time from an assumed one. A fallback is a
    known approximation, not a silent one.
    """
    columns = ["player_id", "team", "status", "raw_status", "published_at",
               "dated"]
    rows: list = []
    for snapshot_date, frame in rosters or ():
        if frame is None or not len(frame):
            continue
        for record in frame.to_dict("records"):
            status = record.get("status")
            if not is_absent(status):
                continue
            published = record.get("published_at")
            dated = published is not None and not (
                isinstance(published, float) and math.isnan(published))
            rows.append({
                "player_id": str(record.get("player_id")),
                "team": str(record.get("team") or ""),
                "status": availability_status_of(status),
                "raw_status": str(record.get("raw_status") or ""),
                "published_at": _as_timestamp(published) if dated
                else _as_timestamp(snapshot_date),
                "dated": bool(dated),
            })
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(["player_id", "published_at"]).reset_index(drop=True)


def build_stints(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse dated absence records into ONE interval per absence.

    A STATE MACHINE, NOT OPEN/CLOSE PAIRING. A player who is hurt for two weeks
    and then returns produces a long run of identical "Out" records, because
    the feed republishes the same status on every snapshot. Pairing an opening
    record against a closing one - MLB's original bug - would leave every
    intermediate record as its own permanently-open stint and suppress the
    player for the rest of time.

    So the walk is a boolean per player: an absent record keeps the stint open
    (extending it, never reopening it), and the first non-absent observation
    closes it. A player whose last record is still absent ends with an open
    stint, which is correct and which the PIT predicate handles explicitly
    rather than by testing NULL-ness.
    """
    if records is None or not len(records):
        return pd.DataFrame({c: pd.Series(dtype="object")
                             for c in STINT_COLUMNS})
    out: list = []
    for player_id, group in records.groupby("player_id", sort=False):
        group = group.sort_values("published_at")
        start = None
        status = None
        team = ""
        for stamp, row in zip(group.published_at, group.to_dict("records")):
            if is_absent(row["status"]):
                if start is None:
                    start = stamp
                    status = row["status"]
                    team = row.get("team") or ""
                continue
            # A healthy observation closes an open stint. A healthy record
            # arriving with no open stint is not an event and is dropped.
            if start is not None:
                out.append({"player_id": player_id, "team": team,
                            "il_start": start, "il_end": stamp,
                            "status": status})
                start = None
                status = None
        if start is not None:
            out.append({"player_id": player_id, "team": team,
                        "il_start": start, "il_end": pd.NaT,
                        "status": status})
    if not out:
        return pd.DataFrame({c: pd.Series(dtype="object")
                             for c in STINT_COLUMNS})
    frame = pd.DataFrame(out, columns=STINT_COLUMNS)
    frame["il_start"] = pd.to_datetime(frame.il_start)
    frame["il_end"] = pd.to_datetime(frame.il_end)
    # A stint that closed on or before it opened is a same-instant healthy
    # read, not an absence, and would create a zero-length interval that the
    # PIT predicate could never satisfy anyway.
    frame = frame[frame.il_end.isna() | (frame.il_end > frame.il_start)]
    return frame.sort_values(["player_id", "il_start"]).reset_index(drop=True)


def reconcile_with_appearances(stints: pd.DataFrame,
                               appearances: pd.DataFrame) -> pd.DataFrame:
    """End a stint at the END of the player's first appearance day after start.

    ``appearances`` is a frame of ``(player_id, gameday)`` player-game rows. A
    player with a row was in the game, so no absence may span PAST it - and
    the closing edge lands at the END of that day rather than at it, for the
    same reason the whole module reads strictly point-in-time: whether he
    played is knowable only AFTER that game. A slate evaluated pre-game
    still sees him Out, so history must too - releasing him AT his
    appearance game would let the historical features use who actually
    played, which is exactly the information the prediction slate does not
    have. (Measured over a full season, Out players who played: 0 of
    10,639 - so this costs a false absence almost never and buys exact
    train/serve symmetry.)

    The rule can only SHORTEN an interval, and only from a day where the
    player demonstrably showed up, so it cannot release a genuine absence
    back into the projection: every date after the appearance it clears is
    one where the feed's own status had already lapsed. A row on the start
    date is ignored rather than treated as a close - the feed publishes an
    injury the same day it is reported, so a same-day appearance is the
    announcement, not a return.
    """
    if stints is None or not len(stints):
        return stints
    if appearances is None or not len(appearances):
        return stints
    required = {"player_id", "gameday"}
    if not required.issubset(appearances.columns):
        return stints

    days = {
        str(player): group.gameday.to_numpy(dtype="datetime64[ns]")
        for player, group in appearances.sort_values(["player_id", "gameday"])
        .groupby("player_id", sort=False)
    }
    ends: list = []
    for player_id, start, end in zip(stints.player_id, stints.il_start,
                                     stints.il_end):
        array = days.get(str(player_id))
        nxt = None
        if array is not None and len(array):
            index = int(np.searchsorted(array, np.datetime64(start), "right"))
            if index < len(array):
                nxt = pd.Timestamp(array[index])
        if nxt is not None:
            # 23:59 of the appearance day: still AFTER any real tip-off
            # that day (so the appearance game itself stays suppressed in
            # tipoff mode) but still BEFORE the prior_end_of_day mode's
            # read moment for the NEXT day's game (23:59:59 of the day
            # before, which is the appearance day), so that mode releases
            # him on time. One edge, correct under both configured
            # cutoffs; 23:59:59 would leak a day late under the second.
            appearance_end = (nxt + pd.Timedelta(days=1)
                              - pd.Timedelta(minutes=1))
            if pd.isna(end) or appearance_end < end:
                end = appearance_end
        ends.append(end)
    out = stints.copy()
    out["il_end"] = ends
    out = out[out.il_end.isna() | (out.il_end > out.il_start)]
    return out.reset_index(drop=True)


def cutoff_for(tipoff) -> pd.Timestamp:
    """The instant a game's injury state is read at, per the configured mode.

    ``tipoff`` (default) reads at the game's own tipoff, which is the finest
    granularity the feed supports for free - the timestamp is already on every
    record, so this is the same comparison with a tighter bound rather than a
    different mechanism.

    ``prior_end_of_day`` reads at 23:59 on the day BEFORE the game, so nothing
    published on game day can gate that game at all. It is the coarser and more
    conservative reading, and it is strictly safer against a feed that
    timestamps a record with the moment it was FILED rather than the moment the
    status became true.
    """
    moment = _as_timestamp(tipoff)
    if config.PLAYER_EPM_INJURY_CUTOFF == "prior_end_of_day":
        return (moment.normalize() - pd.Timedelta(days=1)
                + pd.Timedelta(hours=23, minutes=59, seconds=59))
    return moment


def out_at(stints: pd.DataFrame, player_id: str, tipoff) -> bool:
    """Was this player unavailable as of the game's cutoff? STRICTLY PIT.

    Only absences whose ``il_start`` is strictly before the cutoff count. An
    absence that began at or after it is not knowable at that instant and
    cannot gate that game.

    An open stint (``il_end`` NULL) covers every later cutoff, and the NULL is
    handled INSIDE the predicate rather than by testing the join's NULL-ness -
    the same trap MLB documents, where a ``LEFT JOIN ... IS NULL`` test on open
    stints flagged 59.6% of all participants as injured.
    """
    if stints is None or not len(stints):
        return False
    moment = cutoff_for(tipoff)
    rows = stints[stints.player_id.astype(str) == str(player_id)]
    for start, end in zip(rows.il_start, rows.il_end):
        start = _as_timestamp(start)
        if not start < moment:
            continue
        if pd.isna(end) or _as_timestamp(end) >= moment:
            return True
    return False


def annotate_availability(ratings: pd.DataFrame, stints: pd.DataFrame,
                          target_dates=None) -> pd.DataFrame:
    """Add ``is_available`` to a rating frame, evaluated point-in-time.

    The rating frame is NOT filtered here. Filtering happens at pool
    construction, where a removed player's replacement inherits the vacated
    slot; dropping rows at this stage would instead leave a hole that a mean
    silently ignores, which is a different and worse feature.
    """
    if ratings is None or not len(ratings):
        return ratings
    out = ratings.copy()
    if stints is None or not len(stints):
        out["is_available"] = True
        return out
    moments = out.target_date if target_dates is None else target_dates
    available = [
        not out_at(stints, player, moment)
        for player, moment in zip(out.player_id, moments)
    ]
    out["is_available"] = available
    return out


def staleness(stints: pd.DataFrame, decided_max_date=None) -> dict:
    """How far the stint table trails the data it is meant to cover.

    A table that stops being updated keeps every open stint open, so the
    failure mode is a slowly growing set of permanently-suppressed players -
    invisible until someone counts them. Reporting the lag makes it visible.
    """
    if stints is None or not len(stints):
        return {"stints": 0, "window_start": None, "window_end": None,
                "lag_days": None, "open_at_window_end": 0}
    frame = stints
    window_end = frame.il_start.max()
    window_start = frame.il_start.min()
    lag = None
    if decided_max_date is not None:
        lag = int((pd.Timestamp(decided_max_date).normalize()
                   - pd.Timestamp(window_end).normalize()).days)
    return {
        "stints": int(len(frame)),
        "players": int(frame.player_id.nunique()),
        "window_start": None if pd.isna(window_start) else str(window_start.date()),
        "window_end": None if pd.isna(window_end) else str(window_end.date()),
        "lag_days": lag,
        "open_at_window_end": int(frame.il_end.isna().sum()),
    }
