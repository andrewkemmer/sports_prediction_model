"""Prove the whole feature set comes out of stats.nba.com, in one command.

This is the check that answers "can we build every feature we serve from the
upstream we actually have?".  It pulls real data, builds the real features
through the real code path, and then reports two things per feature: how much
of it is actually populated, and which ``stats.nba.com`` column it is derived
from.  A feature with coverage but no named upstream is a feature nobody can
explain when it drifts; a feature with an upstream but no coverage is a bug.

Run it on a host that cannot reach NBA.com and it fails fast with the reason,
which is the fastest way to tell a blocked host from a broken pipeline.

    python smoke_nba.py                    # the whole cached window
    python smoke_nba.py --season 2024-25   # one season, live, no cache
    python smoke_nba.py --min-coverage 80  # stricter gate
    python smoke_nba.py --diagnose         # find the block, one hop at a time

``--diagnose`` is the mode to run on a host where the pull times out.  The
pipeline's own logs cannot tell a blocked network from a missing cookie,
because both end as "no response".  The diagnostic walks the path in order —
DNS, TLS, the host root, the prime, the season log with the cookie, the season
log without it — and names the first hop that does not answer.

Exit code is 0 only when every declared feature exists and clears the gate.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import logging
import re
import socket
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

import config
import features
import ingestion as ing

logger = logging.getLogger("smoke")

# Every feature the pipeline serves or considers, mapped to the
# ``stats.nba.com/stats/LeagueGameLog`` column it is derived from.  Regular
# season and playoffs are separate calls, so game type is its own upstream.
# Nothing in this table may name a source the ingestion module does not read.
FEATURE_UPSTREAM: dict[str, str] = {
    # Moneyline contract
    "elo_home": "derived: GAME_ID + home/away score",
    "elo_away": "derived: GAME_ID + home/away score",
    "elo_diff": "derived: GAME_ID + home/away score",
    "win_pct_home": "WL + GAME_DATE",
    "win_pct_away": "WL + GAME_DATE",
    "win_pct_diff": "WL + GAME_DATE",
    "rest_days_home": "GAME_DATE",
    "rest_days_away": "GAME_DATE",
    "rest_days_diff": "GAME_DATE",
    "back_to_back_diff": "GAME_DATE",
    "is_playoffs": "season-type call (LeagueGameLog SeasonType)",
    "is_home": "derived: home/away side of the game",
    "ewm_net_points_home": "PTS + team score",
    "ewm_net_points_away": "PTS + team score",
    "ewm_net_points_diff": "PTS + team score",
    "ewm_off_rating_home": "PTS + team score",
    "ewm_off_rating_away": "PTS + team score",
    "ewm_off_rating_diff": "PTS + team score",
    "ewm_def_rating_home": "PTS + team score",
    "ewm_def_rating_away": "PTS + team score",
    "ewm_def_rating_diff": "PTS + team score",
    "ewm_pace_home": "PTS + opponent score",
    "ewm_pace_away": "PTS + opponent score",
    "ewm_pace_diff": "PTS + opponent score",
    "ewm_efg_pct_home": "FGM + FG3M + FGA",
    "ewm_efg_pct_away": "FGM + FG3M + FGA",
    "ewm_efg_pct_diff": "FGM + FG3M + FGA",
    "ewm_turnover_margin_home": "TOV",
    "ewm_turnover_margin_away": "TOV",
    "ewm_turnover_margin_diff": "TOV",
    "ewm_rebound_margin_home": "REB",
    "ewm_rebound_margin_away": "REB",
    "ewm_rebound_margin_diff": "REB",
    "ewm_ast_per_game_home": "AST",
    "ewm_ast_per_game_away": "AST",
    "ewm_ast_per_game_diff": "AST",
    # RFE candidate pool
    "points_for_pg": "PTS",
    "points_against_pg": "opponent score",
    "assists_per_game": "AST",
    "rebounds_per_game": "REB",
    "turnovers_per_game": "TOV",
    "three_point_pct": "FG3M + FG3A",
    "free_throw_pct": "FTM + FTA",
    # Tree categoricals
    "home_team_id": "TEAM_ABBREVIATION",
    "away_team_id": "TEAM_ABBREVIATION",
}


def _declared_features() -> list[str]:
    cols = list(config.MONEYLINE_FEATURE_COLS) + list(config.RFE_CANDIDATE_COLS)
    cols += list(config.TREE_CATEGORICAL_COLS)
    return list(dict.fromkeys(cols))


def _star(base: str) -> list[str]:
    """A candidate metric fans out to diff/home/away and two window forms.

    The RFE pool is namespaced (``nba_`` in ``config.NBA_CANDIDATE_COLS``), so
    a lookup that skips the prefix silently calls every one of them unmapped.
    """
    out: list[str] = []
    for window in ("ewm", "roll"):
        for side in ("diff", "home", "away"):
            name = f"nba_{base}_{window}_{side}"
            if name in _declared_set:
                out.append(name)
    return out


_declared_set: set[str] = set()


def _coverage(frame: pd.DataFrame, feature: str) -> float:
    if feature not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[feature], errors="coerce")
    if not len(values):
        return float("nan")
    return 100.0 * float(values.notna().mean())


def _upstream_for(feature: str) -> str:
    if feature in FEATURE_UPSTREAM:
        return FEATURE_UPSTREAM[feature]
    for base, upstream in (
            ("points_for_pg", "PTS"), ("points_against_pg", "opponent score"),
            ("assists_per_game", "AST"), ("rebounds_per_game", "REB"),
            ("turnovers_per_game", "TOV"), ("three_point_pct", "FG3M + FG3A"),
            ("free_throw_pct", "FTM + FTA")):
        for name in _star(base):
            if name == feature:
                return upstream
    return "UNMAPPED"


def pull_window(season: str | None) -> tuple[pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame]:
    """Fetch real data and normalize it exactly as the pipeline does."""
    if season:
        logger.info("pulling %s from stats.nba.com (live, uncached)", season)
        frames = []
        for season_type, game_type in (
                (ing.SEASON_TYPE_REGULAR, config.GAME_TYPE_REG),
                (ing.SEASON_TYPE_PLAYOFFS, config.GAME_TYPE_POST)):
            raw = ing._fetch_season_log(season, season_type, 0.3)
            prepared = ing._prepare_log(raw, game_type)
            if not prepared.empty:
                frames.append(prepared)
        log = pd.concat(frames, ignore_index=True)
    else:
        logger.info("reading the full window through the normal pull path")
        start, end = ing._window()
        games, team_stats, player_stats, _ = ing._pull_seasons(start, end)
        return games, team_stats, player_stats

    games = ing._games_frame(log)
    team_stats = ing._team_stats_frame(log, games)
    player_stats = ing._player_stats_frame(log)
    return games, team_stats, player_stats


# --------------------------------------------------------------------------
# --diagnose: which hop of the path to stats.nba.com is the one that fails
# --------------------------------------------------------------------------
#
# The pull's log says "timed out" and the host's log says "403", and those
# need opposite fixes: the first is the network path, the second is the session.
# Nothing in a single request can tell them apart, so the diagnostic stops
# trusting the summary and asks each layer in turn, cheapest first.  Every step
# records its own status, so one broken layer does not hide the layers below it.

_DIAG_SEASON = "2024-25"
_DIAG_GAME_ID = "0022500001"


def _http_probe(url: str, headers: dict[str, str], timeout: float) -> str:
    """GET a URL and report what came back, in words worth logging."""
    request = urllib.request.Request(url, headers=dict(headers))
    opener = urllib.request.build_opener()
    started = time.monotonic()
    with opener.open(request, timeout=timeout) as response:
        body = response.read(4096)
    # The season-log query is 40 parameters long and buries the one fact that
    # matters, which is which host and path actually answered. Redirects make
    # the answer different from the answer asked for, so both are printed.
    parts = urllib.parse.urlsplit(response.url)
    query = len(urllib.parse.parse_qsl(parts.query))
    return (f"HTTP {response.status} at {parts.netloc}{parts.path}"
            f"{f' ({query} params)' if query else ''}, {len(body)} bytes read, "
            f"{time.monotonic() - started:.1f}s"
            + ("" if parts.netloc == urllib.parse.urlsplit(url).netloc
               else f"  [redirected off {urllib.parse.urlsplit(url).netloc}]"))


def _dns_probe() -> str:
    infos = socket.getaddrinfo("stats.nba.com", 443, proto=socket.IPPROTO_TCP)
    addresses = sorted({info[4][0] for info in infos})
    return f"{len(infos)} record(s) -> {', '.join(addresses)}"


def _tls_probe() -> str:
    context = ssl.create_default_context()
    raw = socket.create_connection(("stats.nba.com", 443), timeout=15)
    with context.wrap_socket(raw, server_hostname="stats.nba.com") as tls:
        subject = dict(x[0] for x in (tls.getpeercert() or {}).get("subject", ()))
        return (f"{tls.version()} {tls.cipher()[0]} "
                f"CN={subject.get('commonName', '?')}")


def _prime_probe() -> str:
    """Prime the session and report both what the host offered and what we kept.

    The two differ more often than they should.  A host can set cookies the
    jar then refuses — ``SameSite=None`` without ``Secure`` is the usual way,
    and it is silent — and the pipeline would then log "primed: 0 cookie(s)"
    only if it bothered, which it does not, because a missing session is
    survivable.  So the numbers go side by side.
    """
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    request = urllib.request.Request(ing._SESSION_URL,
                                     headers=dict(ing._HTTP_HEADERS))
    with opener.open(request, timeout=20) as response:
        status = f"HTTP {response.status} {response.url}"
        raw_headers = response.headers.get_all("Set-Cookie") or []
    served = sorted({m.group(1).strip() for value in raw_headers
                     for m in [re.match(r"\s*([^=;]+)=", value)] if m})
    kept = sorted({cookie.name for cookie in jar})
    primed = ing._prime_nba_session(force=True)
    sent = ing._session_header().get("Cookie", "")
    sent_names = [piece.split("=", 1)[0].strip()
                  for piece in sent.split(";") if "=" in piece]
    return (f"{status}; served {len(raw_headers)} Set-Cookie "
            f"[{', '.join(served) or 'none'}]; jar kept {len(kept)} "
            f"[{', '.join(kept) or 'none'}]; _prime_nba_session -> {primed}; "
            f"Cookie header would carry {len(sent_names)} "
            f"[{', '.join(sent_names) or 'EMPTY'}] "
            f"({len(sent)} chars)")


def _season_log_probe(seasoned: bool, timeout: float) -> str:
    """The pull's own season-log URL, with or without the session cookie."""
    query = ing._season_log_query(_DIAG_SEASON, ing.SEASON_TYPE_REGULAR)
    headers = dict(ing._STATS_HEADERS)
    label = "with session" if seasoned else "no session  "
    if seasoned:
        headers.update(ing._session_header())
    return f"{label}: " + _http_probe(f"{ing.SEASON_LOG_URL}?{query}",
                                      headers, timeout)


def _cdn_probe(timeout: float) -> str:
    return _http_probe(ing.BOXSCORE_URL.format(game_id=_DIAG_GAME_ID),
                       dict(ing._HTTP_HEADERS), timeout)


def _run_probe(probe, timeout: float) -> tuple[str, str, float]:
    """Run one hop, never raise, and classify silence separately from error."""
    started = time.monotonic()
    try:
        detail = probe()
    except (TimeoutError, socket.timeout) as exc:
        return "SILENT", f"{type(exc).__name__}: {exc or 'no response'}", \
            time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 - the report is the product here
        return "ERROR", f"{type(exc).__name__}: {exc}", time.monotonic() - started
    return "OK", detail, time.monotonic() - started


def _verdict(steps: dict[str, tuple[str, str, float]]) -> list[str]:
    """Name the first hop that failed, and what that rules out."""
    if steps["dns"][0] != "OK":
        return ["stats.nba.com does not even resolve from this host. The block is "
                "the name, not the request; no header or retry can help."]
    if steps["tls"][0] != "OK":
        return ["DNS resolves and the TCP connection opens, but the TLS handshake "
                "never completes. An intercepting or blackholing proxy is the "
                "usual cause, and it is outside anything the pipeline sends."]
    if steps["http root"][0] != "OK":
        return ["The TLS handshake completes and then the request goes nowhere. "
                "This is an edge block on this client or egress IP, not a bad "
                "request: the pipeline's headers, cookies, and retries cannot "
                "change it. Run where the host is reachable, or warm the cache."]
    if "0 cookie" in steps["prime"][1] or "EMPTY]" in steps["prime"][1]:
        return ["nba.com itself answers and serves the page, but this client is "
                "handed no usable session cookie. The edge is treating us as a "
                "bot that may not be fixed by priming harder."]
    if steps["season log + cookie"][0] != "OK":
        return ["The session is primed and the host root answers, yet the season "
                "log is still silent. The cookie is not the missing piece, so "
                "the block is this host from this IP. Retrying longer, or priming "
                "again, will cost the timeout budget and change nothing."]
    if steps["season log no cookie"][0] == "OK":
        return ["The season log answers with AND without the session cookie on "
                "this host, and in well under a second either way. The cookie is "
                "not load-bearing here. So a host that times out on the same "
                "request is being treated differently by the edge for reasons no "
                "header, cookie, or retry can reach - the difference is the "
                "client, not the request. Prime one notebook cell on the blocked "
                "host and compare the hop that goes quiet."]
    return ["The season log answers from this host with the session cookie the "
            "pipeline sends. Ingestion works here; a pull that fails on this host "
            "is failing somewhere else, and the season log rows above say where."]


def diagnose(timeout: float) -> int:
    print("=" * 78)
    print("NBA connectivity diagnostic — one hop at a time, cheapest first")
    print("=" * 78)
    print(f"python {sys.version.split()[0]} on {sys.platform}; "
          f"probes time out after {timeout:.0f}s\n")

    proxies = {k: v for k, v in urllib.request.getproxies().items()
               if not k.endswith("no_proxy")}
    print(f"proxy configured: {proxies or 'none'}")

    probes: list[tuple[str, object]] = [
        ("dns", _dns_probe),
        ("tls", _tls_probe),
        ("http root", lambda: _http_probe(
            "https://stats.nba.com/", dict(ing._HTTP_HEADERS), timeout)),
        ("prime", _prime_probe),
        ("season log + cookie", lambda: _season_log_probe(True, timeout)),
        ("season log no cookie", lambda: _season_log_probe(False, timeout)),
        ("cdn box score", lambda: _cdn_probe(min(timeout, 30.0))),
    ]

    steps: dict[str, tuple[str, str, float]] = {}
    for name, probe in probes:
        status, detail, elapsed = _run_probe(probe, timeout)
        steps[name] = (status, detail, elapsed)
        print(f"\n[{status:>6}] {name}  ({elapsed:.1f}s)")
        print(f"         {detail}")

    print("\n" + "=" * 78)
    print("verdict")
    print("=" * 78)
    for line in _verdict(steps):
        print(f"  {line}")
    reached = steps["season log + cookie"][0] == "OK"
    print(f"\n  -> stats.nba.com season log is "
          f"{'REACHABLE' if reached else 'UNREACHABLE'} from this host")
    print("=" * 78)
    return 0 if reached else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", help="pull one season live, e.g. 2024-25")
    parser.add_argument("--min-coverage", type=float, default=25.0,
                        help="%% of games a feature must be populated on")
    parser.add_argument("--out-dir", help="write the coverage table as CSV")
    parser.add_argument("--diagnose", action="store_true",
                        help="report which hop to stats.nba.com fails, and stop")
    parser.add_argument("--diagnose-timeout", type=float, default=45.0,
                        help="seconds each diagnostic probe may take (default 45)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(message)s", stream=sys.stdout)
    if args.diagnose:
        return diagnose(args.diagnose_timeout)
    _declared_set.update(_declared_features())

    started = time.monotonic()
    print("=" * 78)
    print("NBA feature-extraction smoke — upstream: stats.nba.com LeagueGameLog")
    print("=" * 78)

    try:
        games, team_stats, player_stats = pull_window(args.season)
    except Exception as exc:  # noqa: BLE001 - the point is to report, not raise
        print(f"\nPULL FAILED: {type(exc).__name__}: {exc}\n")
        print("If this is a timeout, the host cannot reach stats.nba.com and no")
        print("amount of retrying in the pipeline will change that. If it is a")
        print("403 or a 500, the session cookie has expired; re-run, the primer")
        print("re-primes once per run.")
        return 2

    if games.empty:
        print("\nPULL SUCCEEDED BUT RETURNED NO GAMES — nothing to verify.\n")
        return 2

    print(f"\ngames       {len(games):>9,}")
    print(f"team rows   {len(team_stats):>9,}")
    print(f"player rows {len(player_stats):>9,}")
    print(f"source      {ing.SOURCE_USED.get('source', 'nba.com')}")

    print("\n--- upstream player columns actually parsed ---")
    cols = sorted(c for c in player_stats.columns)
    print(f"{len(cols)} columns: {', '.join(cols)}")

    missing_upstream = [c for c in (
        "points", "ast", "reb", "tov", "stl", "blk", "pf", "minutes",
        "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
        "plus_minus", "win", "player_id", "team") if c not in cols]
    if missing_upstream:
        print(f"MISSING FROM PLAYER FRAME: {missing_upstream}")

    built = features.build_game_features(games, team_stats)
    # The tree categoricals are attached after the linear contract, exactly
    # where the pipeline attaches them, so the audit sees the same columns the
    # model does rather than a subset that happens to look complete.  The
    # helper returns its own frame keyed by index, so join it on rather than
    # reassigning, or the linear contract is silently replaced by two columns.
    built = built.join(features.team_category_ids(built))

    print(f"\n--- feature coverage over {len(built):,} games "
          f"(gate {args.min_coverage:.0f}%) ---")
    rows = []
    starved, absent, unmapped = [], [], []
    for feature in _declared_features():
        pct = _coverage(built, feature)
        upstream = _upstream_for(feature)
        if upstream == "UNMAPPED":
            unmapped.append(feature)
        if np.isnan(pct):
            absent.append(feature)
        elif pct < args.min_coverage:
            starved.append((feature, pct))
        rows.append({"feature": feature, "coverage_pct": None if np.isnan(pct)
                     else round(pct, 2), "upstream": upstream})

    report = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", 200, "display.width", 100,
                           "display.max_colwidth", 44):
        print(report.to_string(index=False))

    if args.out_dir:
        from pathlib import Path
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"nba_feature_source_audit_{date.today():%Y%m%d}.csv"
        report.to_csv(path, index=False)
        print(f"\nwrote {path}")

    print("\n" + "=" * 78)
    if absent:
        print(f"FAIL  {len(absent)} feature(s) not produced at all: "
              f"{', '.join(absent)}")
    if starved:
        print(f"FAIL  {len(starved)} feature(s) below the gate: "
              + ", ".join(f"{n} {p:.0f}%" for n, p in starved))
    if unmapped:
        print(f"FAIL  {len(unmapped)} feature(s) with no named upstream: "
              f"{', '.join(unmapped)}")
    if missing_upstream:
        print(f"FAIL  upstream player columns absent: {missing_upstream}")
    if not (absent or starved or unmapped or missing_upstream):
        print(f"PASS  all {len(report)} declared features are built from "
              f"stats.nba.com and clear the gate")
    print(f"elapsed {time.monotonic() - started:.1f}s")
    print("=" * 78)
    return 1 if (absent or starved or unmapped or missing_upstream) else 0


if __name__ == "__main__":
    raise SystemExit(main())
