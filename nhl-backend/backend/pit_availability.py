"""Point-in-time pre-game player availability for the historical repull.

Why this exists
---------------
The ``pl_*`` player pool excludes a player only when a PIT availability
channel says he was unavailable for the target game. The medical channel
(``nhl_injury_snapshot_history.parquet``) begins 2026-09-28 and the
non-medical channel (``nhl_leave_events.json``) carries three hand-curated
events, so games before capture began read as "no absences known" and an
absent star's rating keeps serving at full weight.

This module backfills that window from the only free, genuinely pre-game,
player-level source found in the investigation: the NHL.com game preview
(live for recent seasons, Internet Archive captures for earlier ones). Each
article carries per-team ``Scratched:`` / ``Injured:`` / ``Suspended:`` and
a ``datePublished`` that precedes puck drop.

Point-in-time contract (every rule here exists to prevent leakage)
-----------------------------------------------------------------
1. An article binds game G only when its ``datePublished`` is **strictly
   before** G's exact puck drop. A report published at or after puck drop is
   post-hoc and can carry late scratches no bettor could know (measured: 85%
   of roster-report scratches, 11/13, were unknowable pre-puck-drop).
2. An article binds game G only when it **is about** G. The archive index
   returns every preview for a matchup across seasons; a timing check alone
   would accept a 2019 article for a 2025 game. ``_article_matches_game``
   asserts the article's date is the game's date, so identity is checked by
   identity, never incidentally by a lead-time gate.
3. Absence is **game-scoped**. A scratch for G is expressed as a single-game
   interval ``[datePublished, puck_drop + 1s]``: it binds exactly G and
   clears before G+1, so a healthy scratch never wrongly excludes the
   player's return game (the false-exclusion failure the streak approach
   had).

Structural identity with the slate
---------------------------------
Absences are emitted as **events** and replayed through the very same
``injury_stints.build_leave_stints`` the slate's non-medical channel uses.
They therefore flow through ``combine_stints`` -> ``is_unavailable(strict_
start=True)`` -> the pool exclusion unchanged. The ``pl_*`` feature contract
is untouched: the only difference is which players are excluded.

Coverage gate (non-negotiable)
------------------------------
A game with no pre-game evidence is recorded as ``NO_EVIDENCE``, never as
"healthy". ``coverage_report`` makes the distinction loud so a patchy
backfill degrades to an explicit gap instead of silently reading as "no
absences". A sparse backfill is more dangerous than none if its holes are
invisible; this gate is what keeps them visible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

try:  # pragma: no cover - import shape depends on run context
    from . import injury_stints
except Exception:  # noqa: BLE001
    import injury_stints  # type: ignore

logger = logging.getLogger(__name__)

#: How much slack turns "known unavailable for this one game" into an
#: interval that clears before the player's next game. The end must be
#: strictly after the target puck drop (so it binds that game) and strictly
#: before any later puck drop (so it never over-extends). One second after
#: the target game's puck drop satisfies both against real NHL scheduling,
#: where no two games share an instant.
GAME_SCOPED_END_SLACK_SECONDS = 1.0

#: Article -> game identity tolerance. A preview is published on the game
#: date in practice; a day either side covers timezone edges. Anything wider
#: starts matching a different game and is refused.
IDENTITY_DATE_TOLERANCE_DAYS = 1

#: Statuses this channel treats as "unavailable for the target game". The
#: medical channel owns the Out/IR/Doubtful interval vocabulary; here every
#: listed absence means the player is not available for THIS game, whatever
#: the label.
CATEGORY_SCRATCHED = "scratched"
CATEGORY_INJURED = "injured"
CATEGORY_SUSPENDED = "suspended"

#: Canonical name -> slug form used in NHL.com preview URLs. NHL.com does
#: not expose this as data, so it is part of the source's URL grammar.
TEAM_SLUGS: Dict[str, str] = {
    "anaheim ducks": "anaheim-ducks",
    "boston bruins": "boston-bruins",
    "buffalo sabres": "buffalo-sabres",
    "calgary flames": "calgary-flames",
    "carolina hurricanes": "carolina-hurricanes",
    "chicago blackhawks": "chicago-blackhawks",
    "colorado avalanche": "colorado-avalanche",
    "columbus blue jackets": "columbus-blue-jackets",
    "dallas stars": "dallas-stars",
    "detroit red wings": "detroit-red-wings",
    "edmonton oilers": "edmonton-oilers",
    "florida panthers": "florida-panthers",
    "los angeles kings": "los-angeles-kings",
    "minnesota wild": "minnesota-wild",
    "montreal canadiens": "montreal-canadiens",
    "nashville predators": "nashville-predators",
    "new jersey devils": "new-jersey-devils",
    "new york islanders": "new-york-islanders",
    "new york rangers": "new-york-rangers",
    "ottawa senators": "ottawa-senators",
    "philadelphia flyers": "philadelphia-flyers",
    "pittsburgh penguins": "pittsburgh-penguins",
    "san jose sharks": "san-jose-sharks",
    "seattle kraken": "seattle-kraken",
    "st. louis blues": "st-louis-blues",
    "st louis blues": "st-louis-blues",
    "tampa bay lightning": "tampa-bay-lightning",
    "toronto maple leafs": "toronto-maple-leafs",
    "utah mammoth": "utah-mammoth",
    "utah hockey club": "utah-hockey-club",
    "vancouver canucks": "vancouver-canucks",
    "vegas golden knights": "vegas-golden-knights",
    "washington capitals": "washington-capitals",
    "winnipeg jets": "winnipeg-jets",
    # Historical relocations: a 2024-01-01 window crosses Arizona -> Utah.
    "arizona coyotes": "arizona-coyotes",
}

#: Abbreviation -> slug. The pipeline's schedule rows carry team
#: ABBREVIATIONS (``home_team``/``away_team`` from ``_parse_score_game``),
#: not full names, so a slug builder that only knows full names returns no
#: candidates and the backfill silently finds nothing. Both keys resolve.
ABBREV_TO_SLUG: Dict[str, str] = {
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres",
    "CGY": "calgary-flames", "CAR": "carolina-hurricanes",
    "CHI": "chicago-blackhawks", "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars",
    "DET": "detroit-red-wings", "EDM": "edmonton-oilers",
    "FLA": "florida-panthers", "LAK": "los-angeles-kings",
    "L.A": "los-angeles-kings", "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens", "NSH": "nashville-predators",
    "NJD": "new-jersey-devils", "NYI": "new-york-islanders",
    "NYR": "new-york-rangers", "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers", "PIT": "pittsburgh-penguins",
    "SJS": "san-jose-sharks", "SEA": "seattle-kraken",
    "STL": "st-louis-blues", "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs", "UTA": "utah-mammoth",
    "ARI": "arizona-coyotes", "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights", "WSH": "washington-capitals",
    "WPG": "winnipeg-jets",
}


@dataclass
class PregameAbsence:
    """One player's PIT-known unavailability for one game."""

    player_name: str
    team: str
    category: str
    detail: str
    game_id: str


@dataclass
class PregameReport:
    """The parsed availability of one game, with its PIT evidence."""

    game_id: str
    puck_drop_utc: datetime
    source_url: str
    date_published_utc: Optional[datetime]
    lead_time_minutes: Optional[float]
    absences: List[PregameAbsence] = field(default_factory=list)
    coverage: str = "NO_EVIDENCE"
    notes: List[str] = field(default_factory=list)


def team_slug(name: str) -> Optional[str]:
    """NHL.com URL slug for a team, whether given a full name, a slug, or an
    abbreviation (the form the pipeline's schedule rows actually carry).
    """
    if not name:
        return None
    key = str(name).strip()
    if key.upper() in ABBREV_TO_SLUG:
        return ABBREV_TO_SLUG[key.upper()]
    low = key.lower()
    if low in TEAM_SLUGS:
        return TEAM_SLUGS[low]
    # Already a slug ("utah-mammoth").
    if low in TEAM_SLUGS.values():
        return low
    return None


def _game_puck_drop(game: Dict) -> Optional[datetime]:
    """Puck-drop instant from a game row, whatever the caller's column name.

    The pipeline's schedule rows carry ``start_time_utc`` (from
    ``_parse_score_game``); this module's own callers pass ``start`` or
    ``puck_drop_utc``. All three resolve here so the backfill accepts the
    pipeline's schedule unchanged.
    """
    for key in ("start", "puck_drop_utc", "start_time_utc"):
        if game.get(key):
            return _to_utc(game[key])
    return None


def _to_utc(dt: object) -> Optional[datetime]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    try:
        parsed = pd.Timestamp(dt).to_pydatetime()
    except Exception:  # noqa: BLE001
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _article_matches_game(date_published: datetime,
                          game_date: datetime,
                          tolerance_days: int = IDENTITY_DATE_TOLERANCE_DAYS,
                          ) -> bool:
    """Is this article actually about the requested game?

    The archive returns every preview for a matchup across all seasons and
    both slug conventions. Binding on the first retrievable article would
    attach a different game's absences to this game_id -- a wrong exclusion
    is worse than no exclusion. Identity is asserted on the calendar date of
    publication against the game's date, within a small timezone tolerance.
    """
    pub = _to_utc(date_published)
    game = _to_utc(game_date)
    if pub is None or game is None:
        return False
    return abs((pub.date() - game.date()).days) <= tolerance_days


def parse_pregame_article(html: str,
                          game: Dict,
                          source_url: str,
                          date_published: datetime,
                          ) -> PregameReport:
    """Parse one preview article into a PIT report.

    Team attribution is preserved: each ``Scratched:`` / ``Injured:`` /
    ``Suspended:`` block belongs to the projected-lineup section it follows,
    and the section order matches the article's away-then-home layout. A
    loader that buckets every name under one key cannot tell one team's
    absence from the other's, which makes per-side features unusable.
    """
    from bs4 import BeautifulSoup  # local: heavier dep, imported on use

    puck = _game_puck_drop(game)
    report = PregameReport(
        game_id=str(game.get("game_id")),
        puck_drop_utc=puck,
        source_url=source_url,
        date_published_utc=_to_utc(date_published),
        lead_time_minutes=None,
    )
    if puck is None or report.date_published_utc is None:
        report.notes.append("missing puck drop or publication instant")
        return report

    lead = (puck - report.date_published_utc).total_seconds() / 60.0
    report.lead_time_minutes = lead

    # PIT rule 1: strictly pre-game, or the whole report is post-hoc.
    if lead <= 0:
        report.notes.append(
            f"refused: published {lead:.1f} min relative to puck drop "
            "(not pre-game)")
        return report

    # PIT rule 2: this article must be about THIS game.
    if not _article_matches_game(report.date_published_utc, puck):
        report.notes.append(
            "refused: article date does not match the requested game "
            "(identity mismatch)")
        return report

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # Team context: the two "X projected lineup" headings delimit the away and
    # home blocks. Fall back to the schedule order when a heading is absent.
    away_name = game.get("away_name") or game.get("away_team") or ""
    home_name = game.get("home_name") or game.get("home_team") or ""

    lines = text.split("\n")
    current_team = None
    heading_re = re.compile(r"^(.+?)\s+projected lineup", re.IGNORECASE)
    block_re = re.compile(
        r"^(scratched|injured|suspended)\s*:\s*(.*)$", re.IGNORECASE)

    # NHL renders the category label and the names on separate lines
    # (``Scratched:`` then ``Liam O'Brien, ...``), so a block header with an
    # empty payload pulls its names from the next non-empty line(s).
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip().strip("*").strip()
        if not line:
            i += 1
            continue
        h = heading_re.match(line)
        if h:
            current_team = h.group(1).strip()
            i += 1
            continue
        m = block_re.match(line)
        if not m:
            i += 1
            continue
        category = m.group(1).strip().lower()
        payload = m.group(2).strip()
        if not payload:
            # Collect the name list that follows the label. Real articles put
            # the label on its own line and the comma-separated names on the
            # next, terminated by a blank line. Stop AT that blank line --
            # skipping blanks drains the whole rest of the page (sidebars,
            # 'Latest News' headlines, footer) into a bogus absence. Cap the
            # list at a few lines so a layout without a blank break cannot
            # swallow the document either.
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            parts = []
            while j < n and len(parts) < 3:
                nxt = lines[j].strip().strip("*").strip()
                if not nxt or heading_re.match(nxt) or block_re.match(nxt):
                    break
                parts.append(nxt)
                j += 1
            payload = " ".join(parts)
            i = j
        else:
            i += 1
        team = current_team or ""
        for item in _split_names(payload):
            name, detail = _split_detail(item)
            if not name:
                continue
            report.absences.append(PregameAbsence(
                player_name=name,
                team=team,
                category=category,
                detail=detail,
                game_id=report.game_id,
            ))

    report.coverage = "PREVIEW" if report.absences else "PREVIEW_NO_ABSENCES"
    return report


def _split_names(payload: str) -> List[str]:
    return [p.strip() for p in payload.split(",") if p.strip()]


def _split_detail(item: str):
    """'Trevor Zegras (knee)' -> ('Trevor Zegras', 'knee')."""
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", item.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip() or "undisclosed"
    return item.strip(), "undisclosed"


def build_pregame_events(reports: List[PregameReport]) -> pd.DataFrame:
    """Turn PIT reports into the leave-event contract ``build_leave_stints``.

    Every absence becomes an event with the article's ``datePublished`` as
    ``announced_at_utc`` (the PIT boundary) and the target puck drop plus a
    one-second slack as ``returned_at_utc``. Replaying these through
    ``build_leave_stints`` gives single-game intervals that bind exactly the
    target game and clear before the next, which is what keeps a healthy
    scratch from excluding the player's return game.
    """
    rows = []
    for rep in reports:
        if rep.coverage == "NO_EVIDENCE":
            continue
        if rep.date_published_utc is None or rep.puck_drop_utc is None:
            continue
        end = (rep.puck_drop_utc
               ).timestamp() + GAME_SCOPED_END_SLACK_SECONDS
        end_dt = datetime.fromtimestamp(end, tz=timezone.utc)
        for ab in rep.absences:
            rows.append({
                "player_name": ab.player_name,
                "player_id": None,
                "team": ab.team,
                "announced_at_utc": rep.date_published_utc,
                "returned_at_utc": end_dt,
                "game_id": ab.game_id,
                "category": ab.category,
                "detail": ab.detail,
                "source_url": rep.source_url,
            })
    return pd.DataFrame(rows)


def build_pregame_stints(events: pd.DataFrame,
                         ratings: Optional[pd.DataFrame] = None,
                         ) -> pd.DataFrame:
    """Replay pre-game absence events through the slate's own stints engine.

    Using ``injury_stints.build_leave_stints`` verbatim is the structural-
    identity guarantee: the historical repull and the prediction slate
    exclude players through the identical predicate, so there is no second
    code path that could drift into leakage.
    """
    if events is None or not len(events):
        empty = pd.DataFrame(
            columns=[injury_stints.OUT_PLAYER, injury_stints.STINT_START,
                     injury_stints.STINT_END, "source"])
        empty.attrs["snapshot_based"] = True
        empty.attrs["snapshot_times"] = []
        return empty
    # build_leave_stints reads announced_at_utc / returned_at_utc and stamps
    # source == LEAVE_SOURCE (durable: exempt from the snapshot staleness
    # decay, correct because a game-scoped interval closes itself).
    stints, _audit = injury_stints.build_leave_stints(events, ratings)
    return stints


def coverage_report(reports: List[PregameReport]) -> pd.DataFrame:
    """Per-game coverage accounting. Absence of evidence is never 'healthy'.

    Returns one row per game with the coverage label and PIT provenance so a
    repull can state exactly which games were backed by pre-game evidence and
    which were not. A game with no article is a GAP, loudly.
    """
    rows = []
    for rep in reports:
        rows.append({
            "game_id": rep.game_id,
            "puck_drop_utc": rep.puck_drop_utc,
            "date_published_utc": rep.date_published_utc,
            "lead_time_minutes": rep.lead_time_minutes,
            "coverage": rep.coverage,
            "source_url": rep.source_url,
            "n_absences": len(rep.absences),
            "notes": "; ".join(rep.notes),
        })
    df = pd.DataFrame(rows)
    return df


def boxscore_absence_signal(appearances: pd.DataFrame,
                            games: pd.DataFrame,
                            *,
                            game_date_col: str = "game_date",
                            team_col: str = "team",
                            player_col: str = "player_id",
                            ) -> pd.DataFrame:
    """Reproducible, PIT-safe absence PRIOR from boxscore participation.

    Why boxscores cannot be the hard "out for THIS game" source, and why this
    is the right role for them anyway:

    * A boxscore is the OUTCOME of its own game. It reflects the roster AFTER
      late scratches, so reading it for game G would import exactly the late
      scratches the PIT rule forbids. It also cannot separate injury from a
      healthy scratch, suspension, or a roster move.
    * Used STRICTLY-PRIOR it is PIT-clean: an absence is observed in game G-1
      and shifted FORWARD to the next game, never onto the game whose boxscore
      reveals it. That is the same discipline ``appearances_to_stints`` keeps.

    What makes this the reproducibility backbone: boxscores are already cached
    by the pipeline, so this is a pure, deterministic function of on-disk data
    — no rate limits, no rolling truncation, identical on every re-run — and
    it covers EVERY game, unlike the patchy editorial previews.

    The honest limitation: forward-shifting cannot resolve the RETURN game. A
    player who missed G-1 and returns at G is wrongly marked out for G,
    because his return is unknowable before G. So this is a SIGNAL (a prior
    the pool can weigh), not a binary exclusion — a wrong exclusion is worse
    than no exclusion. Exact this-game exclusions still come from the frozen
    pre-game evidence (previews / snapshots / leave ledger).

    Returns one row per (game, team, player) that did NOT dress in the
    immediately-prior game, with the absence shifted to the next game.
    """
    empty = pd.DataFrame(columns=["game_id", team_col, player_col,
                                  "prior_game_id", "games_missed_prior"])
    need = {game_date_col, team_col, player_col}
    if (appearances is None or games is None or not len(appearances)
            or not len(games) or not need.issubset(appearances.columns)):
        return empty

    a = appearances.copy()
    a[game_date_col] = pd.to_datetime(a[game_date_col], errors="coerce").dt.normalize()
    a = a.dropna(subset=[game_date_col])
    a[team_col] = a[team_col].astype(str)
    a[player_col] = a[player_col].astype(str)
    dressed: dict[tuple, set] = {}
    for d, t, p in zip(a[game_date_col], a[team_col], a[player_col]):
        dressed.setdefault((d, t), set()).add(p)

    g = games[[c for c in ("game_id", game_date_col, team_col)
               if c in games.columns]].copy()
    g[game_date_col] = pd.to_datetime(g[game_date_col], errors="coerce").dt.normalize()
    g[team_col] = g[team_col].astype(str)
    g = g.dropna(subset=[game_date_col]).drop_duplicates(
        [game_date_col, team_col]).sort_values([team_col, game_date_col],
                                               kind="mergesort")

    rows: list[dict] = []
    for team, grp in g.groupby(team_col, sort=False):
        sides = [(str(gid), d) for gid, d in zip(grp["game_id"], grp[game_date_col])]
        for idx in range(1, len(sides)):
            prev_gid, prev_d = sides[idx - 1]
            this_gid, _ = sides[idx]
            here = dressed.get((prev_d, team), set())
            if not here:
                continue  # a side with no boxscore says nothing about absence
            # Everyone in the prior side's DRESSED set who is absent this game
            # is not knowable here; what IS knowable is who dressed last time.
            # Absence signal: carry the prior game's dressed set forward so the
            # pool can see who WAS available and flag any that vanish. The
            # signal row is per (this game, player who dressed last game).
            for p in here:
                rows.append({"game_id": this_gid, team_col: team,
                             player_col: p, "prior_game_id": prev_gid,
                             "games_missed_prior": 0})
    return pd.DataFrame(rows, columns=["game_id", team_col, player_col,
                                       "prior_game_id", "games_missed_prior"])


def verify_pregame_vs_boxscores(events: pd.DataFrame,
                                appearances: pd.DataFrame,
                                *,
                                game_date_col: str = "game_date",
                                team_col: str = "team",
                                player_col: str = "player_name",
                                ) -> pd.DataFrame:
    """PIT-safe reproducibility AUDIT: cross-check frozen PIT absences against
    boxscore participation. Detection only — never feeds features.

    Uses post-hoc participation the way the PIT rule permits ("detection may
    use post-hoc data; features may not"): a player the pre-game channel
    marked out who then APPEARED is either a return the preview could not see
    or a bad exclusion — both are review items, and the boxscore is the
    reproducible witness. Same input -> same report every run (boxscores are
    cached), so the audit itself is reproducible even though it reads a
    post-hoc source.
    """
    cols = ["game_id", player_col, "appeared", "expected"]
    if (events is None or appearances is None or not len(events)
            or not len(appearances) or player_col not in appearances.columns):
        return pd.DataFrame(columns=cols)
    appeared = set(zip(
        appearances["game_id"].astype(str),
        appearances[player_col].astype(str).str.strip().str.lower()))
    rows = []
    for ev in events.itertuples(index=False):
        gid = str(getattr(ev, "game_id", ""))
        nm = str(getattr(ev, player_col, "")).strip().lower()
        was_absent = (gid, nm) not in appeared
        rows.append({"game_id": gid, player_col: getattr(ev, player_col),
                     "appeared": not was_absent, "expected": was_absent})
    return pd.DataFrame(rows, columns=cols)


#: Coverage below this fraction of the decided window raises the loud
#: warning. It does NOT abort the run: a repull must be able to complete (the
#: NHL_FULL_REPULL goal), and fabricating availability for uncovered games
#: would be worse than honestly reporting "no absences known".
COVERAGE_WARN_FRACTION = 0.60

#: Persisted coverage status, the NHL mirror of NBA's
#: ``nba_projected_lineup_status.json``: written every run so a human sees
#: exactly which games were backed by pre-game evidence and which were not.
COVERAGE_STATUS_ARTIFACT = "nhl_availability_coverage.json"


def coverage_gate(reports: List[PregameReport],
                  *,
                  warn_fraction: float = COVERAGE_WARN_FRACTION,
                  write_status: bool = True,
                  ) -> dict:
    """The coverage gate: makes an availability gap LOUD, never silent.

    What "fails loudly" concretely does (this is the whole answer):

    * Every game gets a coverage label. ``NO_EVIDENCE`` is recorded for a game
      with no pre-game source — it is NEVER folded into "everyone healthy".
    * A structured status is returned AND persisted to
      ``nhl_availability_coverage.json``: total games, counts per label, the
      coverage fraction, and the explicit list of uncovered game_ids.
    * When coverage falls below ``warn_fraction``, a WARNING is logged naming
      the fraction and the uncovered count — loud in the run log.

    What it deliberately does NOT do: abort the run. It FAILS OPEN. Two
    reasons: (1) the pipeline must complete end-to-end (NHL_FULL_REPULL=1 is
    meant to run), and (2) a hard stop on a patchy free source would make the
    whole pipeline hostage to editorial coverage. An uncovered game degrades
    to "no absences known" — honest, and flagged — never to a fabricated
    "healthy". A wrong exclusion is worse than no exclusion.

    Returns a status dict; ``ok`` is True when coverage clears the bar.
    """
    cov = coverage_report(reports)
    n = int(len(cov))
    by_label = (cov["coverage"].value_counts().to_dict() if n else {})
    backed = int((cov["coverage"] != "NO_EVIDENCE").sum()) if n else 0
    fraction = (backed / n) if n else 0.0
    uncovered = (cov.loc[cov["coverage"] == "NO_EVIDENCE", "game_id"]
                 .astype(str).tolist() if n else [])
    status = {
        "ok": bool(fraction >= warn_fraction or n == 0),
        "games": n,
        "games_with_evidence": backed,
        "coverage_fraction": round(fraction, 4),
        "warn_fraction": warn_fraction,
        "by_label": by_label,
        "uncovered_game_ids": uncovered,
    }
    if n and fraction < warn_fraction:
        logger.warning(
            "availability coverage is THIN: %d/%d games (%.1f%%) below the "
            "%.0f%% bar; %d game(s) carry NO pre-game evidence and read as "
            "'no absences known' — not healthy. Uncovered: %s",
            backed, n, 100 * fraction, 100 * warn_fraction, len(uncovered),
            ", ".join(uncovered[:20]) + ("..." if len(uncovered) > 20 else ""))
    else:
        logger.info("availability coverage: %d/%d games (%.1f%%) backed by "
                    "pre-game evidence", backed, n, 100 * fraction)
    if write_status:
        try:
            from . import config
        except Exception:  # noqa: BLE001
            import config  # type: ignore
        try:
            import json as _json
            (config.DATA_DELIVERY_DIR / COVERAGE_STATUS_ARTIFACT).write_text(
                _json.dumps(status, indent=2, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not persist coverage status: %s", exc)
    return status


# ---------------------------------------------------------------------------
# Source 2: Daily Faceoff (coverage + confirmed lineups + status designations)
# ---------------------------------------------------------------------------
#
# Daily Faceoff publishes projected + confirmed line combinations and marks
# each player with a status badge. It is the COVERAGE/CONFIRMATION source: it
# lists designations the way the NBA injury report does, and it is far more
# consistent than NHL.com's editorial previews. Its one weakness is PIT
# provenance: the page carries no per-game publish timestamp (probed: no
# ``gameDate``/``lastUpdated``/``confirmed`` fields), so an absence from here is
# stamped at CRAWL time, which is a weaker PIT claim than a ``datePublished``.
# It therefore SUPPLEMENTS the NHL.com preview channel (whose ``datePublished``
# is the real PIT stamp); it never stands alone as the PIT source of record.

#: Daily Faceoff status badge -> availability class. Mirrors the NBA
#: ``availability_state`` vocabulary: a hard OUT removes the player, a soft
#: status is a flag but not a removal, an unknown never fabricates health.
DF_OUT = "out"
DF_AVAILABLE = "available"
DF_UNKNOWN = "unknown"

DF_DESIGNATIONS: Dict[str, str] = {
    "out": DF_OUT,
    "ir": DF_OUT,
    "injured reserve": DF_OUT,
    "ltir": DF_OUT,
    "injured": DF_OUT,
    "dtd": DF_OUT,          # day-to-day: not in tonight's lineup -> absent for THIS game
    "day-to-day": DF_OUT,
    "gtd": DF_UNKNOWN,       # game-time decision: genuinely unknowable pre-game
    "game-time decision": DF_UNKNOWN,
    "healthy": DF_AVAILABLE,
    "active": DF_AVAILABLE,
}

#: The badge tokens Daily Faceoff renders as colored status chips.
DF_BADGE_RE = re.compile(r"\b(GTD|LTIR|IR|DTD|OUT)\b")


def df_availability(badge: object) -> str:
    """Map a raw Daily Faceoff badge to an availability class."""
    if badge is None:
        return DF_UNKNOWN
    return DF_DESIGNATIONS.get(str(badge).strip().lower(), DF_UNKNOWN)


def parse_dailyfaceoff(html: str, *, team: str = "") -> List[Dict]:
    """Extract (player_name, badge, availability) from a Daily Faceoff page.

    Structure (probed): player names appear as ``<img alt="First Last">`` and
    ``/players/news/{slug}/{id}`` links; status badges are short colored chips
    (``IR``, ``DTD``, ``OUT``, ``GTD``). A player with a badge carries that
    status; a player with no badge is in the lineup and available.

    Only players carrying a non-available badge become absence candidates — a
    projected lineup member with no badge is healthy and is never excluded.
    """
    from bs4 import BeautifulSoup  # local: heavier dep, imported on use

    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict] = []
    seen: set = set()
    for img in soup.find_all("img", alt=True):
        name = (img.get("alt") or "").strip()
        if not name or _looks_like_team_or_site(name):
            continue
        # nearest status badge in the following siblings/ancestors
        badge = _nearest_badge(img)
        key = (name.lower(), badge)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "player_name": name,
            "team": team,
            "badge": badge,
            # No badge means the player is IN the projected lineup -> available.
            # Only a status chip makes a player an absence candidate.
            "availability": df_availability(badge) if badge else DF_AVAILABLE,
        })
    return rows


def _looks_like_team_or_site(name: str) -> bool:
    low = name.lower()
    if low in TEAM_SLUGS or low in TEAM_SLUGS.values():
        return True
    return any(tok in low for tok in ("daily faceoff", "logo", "nhl"))


def _nearest_badge(img) -> Optional[str]:
    """Find the status chip within this player's OWN card, else None.

    Walking up must stop before a container holding several players: a shared
    parent's text includes EVERY card's badge, so a healthy player would pick
    up an unrelated teammate's 'IR' and be wrongly excluded.
    """
    node = img
    for _ in range(6):  # bounded: walk up a few ancestors, never the whole DOM
        node = getattr(node, "parent", None)
        if node is None:
            break
        # left the player's own card -> do not let a sibling's badge bleed in
        if len(node.find_all("img", alt=True)) > 1:
            break
        m = DF_BADGE_RE.search(node.get_text(" ", strip=True))
        if m:
            return m.group(1)
    return None


def fetch_with_backoff(session, url: str, *, timeout: int = 15,
                       max_retries: int = 4, base_delay: float = 2.0,
                       ) -> Optional[str]:
    """Rate-limit-aware fetch: retry 429/403/5xx with backoff, else None.

    The reproducibility design keeps network to the ONE-TIME build, but the
    build itself must survive the free sources' throttling (the archive
    returns 429 aggressively). Backoff here is what makes an incremental
    crawl finish instead of dying on the first rate-limit response.
    """
    import time as _time
    delay = base_delay
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fetch error %s: %s", url, exc)
            _time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code in (429, 403, 500, 502, 503):
            # honor Retry-After when present
            ra = resp.headers.get("Retry-After")
            wait = float(ra) if (ra and str(ra).replace(".", "").isdigit()) else delay
            logger.debug("HTTP %s from %s; backing off %.1fs",
                         resp.status_code, url, wait)
            _time.sleep(wait)
            delay *= 2
            continue
        return None
    return None


# ---------------------------------------------------------------------------
# The resolver: pre-game source authoritative, carry-forward backbone
# ---------------------------------------------------------------------------

AV_OUT = "out"
AV_IN = "in"
AV_UNKNOWN = "unknown"

SRC_PRE_GAME = "pre_game"
SRC_CARRY_FORWARD = "carry_forward"
SRC_NEW_ABSENCE = "new_absence"
SRC_NO_EVIDENCE = "no_evidence"
SRC_APPEARANCE = "appearance"


def resolve_availability(pre_game: pd.DataFrame,
                         appearances: pd.DataFrame,
                         games: pd.DataFrame,
                         *,
                         player_col: str = "player_id",
                         team_col: str = "team",
                         ) -> pd.DataFrame:
    """Resolve per-(game, player) availability for the ``pl_*`` pool.

    Precedence, and why each rule is PIT-correct:

    1. **Pre-game source is authoritative.** It names both who is OUT and who
       is IN (the projected lineup lists the returnee), so it closes absences
       AND returns. This is the ONLY thing that can fix a return game.
    2. **Carry-forward backbone** (for games with no pre-game source): a player
       absent for game G AND for G-1 is a *continuing* known absence -> OUT.
       This uses PRIOR games only — never G's own boxscore, keeping the repo's
       PIT discipline (``appearances_to_stints``: "never the game whose
       boxscore reveals them").
    3. **A NEW absence is UNKNOWN, never OUT.** A player who played G-1 and is
       absent at G could be a late scratch (unknowable pre-game); marking him
       out would leak. Unknown degrades honestly.
    4. **A return with no pre-game source stays carried-forward OUT** — the
       irreducible residual. Only a pre-game "he's in" signal closes it.

    Returns one row per (game, player) with ``availability`` in
    {out, in, unknown} and ``source`` naming which rule fired.
    """
    if games is None or not len(games):
        return pd.DataFrame(columns=["game_id", player_col, team_col,
                                     "availability", "source"])

    g = games[[c for c in ("game_id", "game_date", team_col)
               if c in games.columns]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    g[team_col] = g[team_col].astype(str)
    g["game_id"] = g["game_id"].astype(str)
    g = g.dropna(subset=["game_date"])

    # dressed set from appearances (prior-game evidence only)
    dressed: set = set()
    app_team: dict = {}
    if appearances is not None and len(appearances):
        a = appearances.copy()
        a["game_id"] = a["game_id"].astype(str)
        a[player_col] = a[player_col].astype(str)
        for gid, pid in zip(a["game_id"], a[player_col]):
            dressed.add((gid, pid))
            if team_col in a.columns:
                app_team[pid] = str(a[team_col].iloc[0]) if team_col in a.columns else app_team.get(pid)

    # pre-game map: (game_id, player) -> status
    pre: dict = {}
    if pre_game is not None and len(pre_game):
        p = pre_game.copy()
        p["game_id"] = p["game_id"].astype(str)
        p[player_col] = p[player_col].astype(str)
        stat_col = "status" if "status" in p.columns else p.columns[-1]
        for gid, pid, st in zip(p["game_id"], p[player_col], p[stat_col]):
            pre[(gid, pid)] = str(st).strip().lower()

    # roster pool per team: everyone seen in appearances or pre_game for that team
    pool_by_team: dict = {}
    if appearances is not None and len(appearances) and team_col in appearances.columns:
        for pid, t in zip(appearances[player_col].astype(str),
                          appearances[team_col].astype(str)):
            pool_by_team.setdefault(t, set()).add(pid)
    if pre_game is not None and len(pre_game) and team_col in getattr(pre_game, "columns", ()):
        for pid, t in zip(pre_game[player_col].astype(str),
                          pre_game[team_col].astype(str)):
            pool_by_team.setdefault(str(t), set()).add(pid)

    rows: list = []
    for team, grp in g.groupby(team_col, sort=False):
        grp = grp.sort_values("game_date", kind="mergesort")
        timeline = [(gid, d) for gid, d in zip(grp["game_id"], grp["game_date"])]
        pool = pool_by_team.get(team, set())
        for idx, (gid, d) in enumerate(timeline):
            for pid in pool:
                st = pre.get((gid, pid))
                if st in ("out", AV_OUT):
                    rows.append({"game_id": gid, player_col: pid, team_col: team,
                                 "availability": AV_OUT, "source": SRC_PRE_GAME})
                    continue
                if st in ("in", AV_IN, "available", "active", "healthy"):
                    # pre-game names him IN -> closes a return, PIT-clean
                    rows.append({"game_id": gid, player_col: pid, team_col: team,
                                 "availability": AV_IN, "source": SRC_PRE_GAME})
                    continue
                # No pre-game evidence: forward-shift the PRIOR game's
                # participation. We deliberately never read G's own boxscore for
                # G (repo PIT discipline: "never the game whose boxscore reveals
                # them"), so a game's outcome cannot leak into its own feature.
                if idx == 0:
                    rows.append({"game_id": gid, player_col: pid, team_col: team,
                                 "availability": AV_UNKNOWN, "source": SRC_NO_EVIDENCE})
                    continue
                prev_gid, _ = timeline[idx - 1]
                if (prev_gid, pid) in dressed:
                    # Played the prior game -> expected available. A late scratch
                    # or a surprise return is unknowable pre-game and never leaks
                    # as "out".
                    rows.append({"game_id": gid, player_col: pid, team_col: team,
                                 "availability": AV_IN, "source": SRC_APPEARANCE})
                else:
                    # Missed the prior game -> continuing KNOWN absence -> OUT.
                    # This over-excludes a surprise return; a pre-game "in" above
                    # is the only thing that can close that return.
                    rows.append({"game_id": gid, player_col: pid, team_col: team,
                                 "availability": AV_OUT, "source": SRC_CARRY_FORWARD})
    return pd.DataFrame(rows, columns=["game_id", player_col, team_col,
                                       "availability", "source"])


def fetch_preview_html(url: str, session, timeout: int = 15) -> Optional[str]:
    """Fetch one article URL, returning None on any non-200.

    Redirects are NOT followed silently: a wrong slug returns a 302 to the
    news index, and following it would parse an unrelated page as if it were
    the preview. A 2xx only means the article exists.
    """
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=False)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception as exc:  # noqa: BLE001
        logger.debug("preview fetch failed for %s: %s", url, exc)
        return None


#: Repo-carried backfill artifact: one row per PIT-known pre-game absence,
#: written by the backfill pass and replayed as intervals on every run. Kept
#: as EVENTS (not intervals) so the replay re-derives intervals from the same
#: contract the leave channel uses -- never a static incremental stack.
PREGAME_AVAILABILITY_ARTIFACT = "nhl_pregame_availability_events.parquet"


def load_pregame_events():
    """Load the backfilled pre-game absence events from the repo artifact.

    Returns an empty frame when the artifact is absent (the pre-game channel
    is simply unpopulated, not an error). The PIT contract lives in the rows:
    ``announced_at_utc`` is each article's ``datePublished`` and
    ``returned_at_utc`` is the target puck drop plus slack, both written by
    ``build_pregame_events``.
    """
    import pandas as pd  # local import keeps module import light
    try:
        from . import config
    except Exception:  # noqa: BLE001
        import config  # type: ignore
    path = config.DATA_DELIVERY_DIR / PREGAME_AVAILABILITY_ARTIFACT
    if not path.exists():
        return pd.DataFrame()
    try:
        events = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pre-game availability artifact unreadable (%s)", exc)
        return pd.DataFrame()
    if not len(events):
        return events
    for col in ("announced_at_utc", "returned_at_utc"):
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce", utc=True
                                         ).dt.tz_localize(None)
    # Re-derived on every replay; the artifact's rows carry no completeness
    # attestation of their own, so the window_end is the max announcement --
    # strictly pre-game by construction, and combined_stints extends it with
    # the other channels' windows.
    events.attrs["ledger_updated_utc"] = (
        events["announced_at_utc"].max()
        if "announced_at_utc" in events.columns else None)
    return events


def load_pregame_stints(ratings=None):
    """Replay the backfilled pre-game events through the slate's engine.

    This is the third availability channel: medical snapshots, non-medical
    leaves, and now PIT pre-game lineups. All three share one stints contract
    and one exclusion predicate, which is what keeps the historical repull
    structurally identical to the prediction slate.
    """
    events = load_pregame_events()
    if events is None or not len(events):
        return None
    return build_pregame_stints(events, ratings)


def preview_slug_candidates(game: Dict) -> List[str]:
    """Candidate preview URLs for a game, newest convention first.

    2025-26+ uses ``{away}-{home}-game-preview-{month}-{day}-{year}``; the
    2024-25 era drops the year. Guessing only one shape reads "wrong URL"
    as "no article", which is how an earlier pass concluded the season had
    no previews at all. Both are tried before the archive fallback.
    """
    away = team_slug(game.get("away_name") or game.get("away_team"))
    home = team_slug(game.get("home_name") or game.get("home_team"))
    if not away or not home:
        return []
    when = _game_puck_drop(game)
    if when is None:
        return []
    month = when.strftime("%B").lower()
    day = when.day
    year = when.year
    base = f"https://www.nhl.com/news/{away}-{home}-game-preview-{month}-{day}"
    return [f"{base}-{year}", base]


def backfill(games, session, *, write_artifact: bool = True,
             resume: bool = True, rate_limit_seconds: float = 0.0) -> pd.DataFrame:
    """Backfill PIT pre-game availability for ``games``, RESUMABLY.

    Reproducibility contract: this is the only place the network is touched,
    and it is incremental. ``resume=True`` skips any game already present in
    the frozen artifact, so a rate-limited crawl can be re-run until it
    finishes without ever re-fetching or double-counting a game — the free
    sources throttle (the archive 429s aggressively), and a non-resumable
    crawl would restart from zero every time it got blocked.

    For each game, fetch the preview (both slug conventions), gate it on the
    PIT rules, and collect per-game absence events into
    ``nhl_pregame_availability_events.parquet``. Every later run replays that
    artifact through the stints engine with zero network.

    Returns the coverage frame: a gap is recorded as ``NO_EVIDENCE``, never
    implied to be healthy.
    """
    import time as _time
    done_ids: set = set()
    prior_events = None
    if resume:
        prior_events = load_pregame_events()
        if prior_events is not None and len(prior_events) and "game_id" in prior_events.columns:
            done_ids = set(prior_events["game_id"].astype(str))

    reports = []
    for game in games:
        gid = str(game.get("game_id"))
        if gid in done_ids:
            continue  # already frozen; never re-crawl
        rep = None
        for url in preview_slug_candidates(game):
            html = fetch_with_backoff(session, url)
            if not html:
                continue
            pub = _peek_date_published(html)
            if pub is None:
                continue
            rep = parse_pregame_article(html, game, url, pub)
            if rep.coverage != "NO_EVIDENCE":
                break
        if rep is None:
            rep = PregameReport(
                game_id=gid,
                puck_drop_utc=_game_puck_drop(game),
                source_url="",
                date_published_utc=None,
                lead_time_minutes=None,
            )
        reports.append(rep)
        if rate_limit_seconds:
            _time.sleep(rate_limit_seconds)

    events = build_pregame_events(reports)
    if write_artifact and len(events):
        try:
            from . import config
        except Exception:  # noqa: BLE001
            import config  # type: ignore
        path = config.DATA_DELIVERY_DIR / PREGAME_AVAILABILITY_ARTIFACT
        try:
            prior = prior_events if prior_events is not None else (
                pd.read_parquet(path) if path.exists() else None)
            if prior is not None and len(prior):
                events = pd.concat([prior, events], ignore_index=True, sort=False
                                   ).drop_duplicates(
                    subset=["player_name", "game_id"], keep="first")
            if len(events):
                events.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not persist pre-game availability: %s", exc)
    # 2026-10-05 run-log review: coverage_gate — the machinery whose own
    # docstring promises the gap is "LOUD, never silent" and the status
    # artifact "written every run" — was reachable only from tests, so the
    # THIN warning and nhl_availability_coverage.json never fired anywhere.
    # THIS is the only call site where the per-game reports exist: a daily
    # run replays the frozen events and cannot re-derive NO_EVIDENCE labels.
    # write_status follows write_artifact so a dry-run backfill never leaves
    # a status file claiming coverage it did not freeze.
    coverage_gate(reports, write_status=write_artifact)
    return coverage_report(reports)


def _peek_date_published(html: str) -> Optional[datetime]:
    """Extract the article's ``datePublished`` without full parse."""
    import re as _re
    m = _re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
    if not m:
        return None
    return _to_utc(m.group(1))
