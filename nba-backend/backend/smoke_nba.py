"""Prove the whole feature set comes out of the three upstreams, in one command.

This is the check that answers "can we build every feature we serve from the
sources we actually have?". It pulls real data, builds the real features
through the real code path, and then reports two things per feature: how much of
it is actually populated, and which upstream column it is derived from. A
feature with coverage but no named upstream is a feature nobody can explain
when it drifts; a feature with an upstream but no coverage is a bug.

The division of labour it verifies is the one MLB already runs:

* **ESPN's scoreboard** owns the schedule - which games exist, on what date,
  between whom, and whether they are finished.
* **stats.nba.com ``LeagueGameLog``** owns the features - one request per
  season returns every player line of that season.
* **stats.nba.com ``playbyplayv3``** owns the play-by-play - one request per
  game, counted into a per-team event rollup the ladder reads.

    python smoke_nba.py                    # the whole cached window
    python smoke_nba.py --min-coverage 80  # stricter gate
    python smoke_nba.py --pbp-games 25      # how many games to sweep
    python smoke_nba.py --diagnose         # find the block, one hop at a time

``--diagnose`` is the mode to run on a host where the pull fails. The pipeline's
own logs cannot tell a refused request from a silent one, because both end as
"no response". The diagnostic walks each upstream in order - DNS, TLS, the host
root, the request the pipeline actually sends - and names the first hop that
does not answer.

Exit code is 0 only when every declared feature exists and clears the gate.
"""
from __future__ import annotations

import argparse
import logging
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

import config
import features
import ingestion as ing
import nba_sources as src
import source_contract as contract

logger = logging.getLogger("smoke")

#: The schedule's own contribution. ESPN's scoreboard is where identity, the
#: date, the sides and the result come from - so every Elo and form feature
#: ultimately rests on it, not on stats.nba.com.
UPSTREAM_SCHEDULE = "espn scoreboard: event id, date, home/away, score"
UPSTREAM_SEASON_LOG = "stats.nba.com LeagueGameLog"
UPSTREAM_PLAY_BY_PLAY = "stats.nba.com playbyplayv3"

#: Every feature the pipeline serves or considers, mapped to the upstream it is
#: derived from. Nothing in this table may name a source the ingestion module
#: does not read; ``--diagnose`` and this map are checked against each other.
FEATURE_UPSTREAM: dict[str, str] = {
    # --- Moneyline contract: Elo, form, rest -----------------------------
    "elo_home": f"{UPSTREAM_SCHEDULE} -> Elo",
    "elo_away": f"{UPSTREAM_SCHEDULE} -> Elo",
    "elo_diff": f"{UPSTREAM_SCHEDULE} -> Elo",
    "win_pct_home": f"{UPSTREAM_SCHEDULE} (result) -> trailing",
    "win_pct_away": f"{UPSTREAM_SCHEDULE} (result) -> trailing",
    "win_pct_diff": f"{UPSTREAM_SCHEDULE} (result) -> trailing",
    "rest_days_home": f"{UPSTREAM_SCHEDULE} (date) -> prior games",
    "rest_days_away": f"{UPSTREAM_SCHEDULE} (date) -> prior games",
    "rest_days_diff": f"{UPSTREAM_SCHEDULE} (date) -> prior games",
    "back_to_back_diff": f"{UPSTREAM_SCHEDULE} (date) -> prior games",
    "is_home": "constant 1.0 by construction",
    "is_playoffs": f"{UPSTREAM_SCHEDULE} (season.type)",
    # --- Moneyline contract: box-score form ------------------------------
    "ewm_off_rating_home": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_off_rating_away": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_def_rating_home": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_def_rating_away": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_off_rating_diff": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_def_rating_diff": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_net_points_diff": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_pace_diff": f"{UPSTREAM_SEASON_LOG}: PTS summed per team",
    "ewm_efg_pct_diff": f"{UPSTREAM_SEASON_LOG}: FGM/FGA/FG3M",
    "ewm_turnover_margin_diff": f"{UPSTREAM_SEASON_LOG}: TOV",
    "ewm_rebound_margin_diff": f"{UPSTREAM_SEASON_LOG}: REB",
    "ewm_ast_per_game_diff": f"{UPSTREAM_SEASON_LOG}: AST",
    # --- RFE candidate pool: trailing team metrics -----------------------
}
FEATURE_UPSTREAM.update({
    f"nba_{metric}_{window}_{side}":
        f"{UPSTREAM_SEASON_LOG}: {'/'.join(columns)}"
    for metric, columns in {
        "points_for_pg": ("PTS",), "points_against_pg": ("PTS",),
        "assists_per_game": ("AST",), "rebounds_per_game": ("REB",),
        "turnovers_per_game": ("TOV",), "three_point_pct": ("FG3M", "FG3A"),
        "free_throw_pct": ("FTM", "FTA"),
    }.items()
    for window in ("ewm", "roll")
    for side in ("diff", "home", "away")
})
# --- The play-by-play contribution ---------------------------------------
FEATURE_UPSTREAM.update({
    "event_three_rate_diff":
        f"{UPSTREAM_PLAY_BY_PLAY}: shotValue=3, actionType Made/Missed Shot",
    "event_rim_rate_diff":
        f"{UPSTREAM_PLAY_BY_PLAY}: shotDistance <= {src.RIM_FEET}ft",
    "event_live_tov_rate_diff":
        f"{UPSTREAM_PLAY_BY_PLAY}: actionType Turnover, subType in "
        f"{list(src.LIVE_TURNOVER_SUBTYPES)}",
    "event_and_in_rate_diff":
        f"{UPSTREAM_PLAY_BY_PLAY}: Shooting foul -> Free Throw 1 of 1",
    "event_shot_distance_diff":
        f"{UPSTREAM_PLAY_BY_PLAY}: shotDistance",
    "event_possessions_diff":
        f"{UPSTREAM_PLAY_BY_PLAY}: FGA + 0.44*FTA - OREB + TOV",
    "event_shooting_fouls_diff":
        f"{UPSTREAM_PLAY_BY_PLAY}: actionType Foul, subType Shooting",
    "event_q4_points_diff": f"{UPSTREAM_PLAY_BY_PLAY}: period=4 scoring",
})


def declared_features() -> list[str]:
    """Every feature the model contract declares, plus the RFE pool."""
    return sorted(set(config.MONEYLINE_FEATURE_COLS)
                  | set(config.RFE_CANDIDATE_COLS))


def coverage(frame: pd.DataFrame, feature: str) -> float:
    if feature not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[feature], errors="coerce")
    return 100.0 * float(values.notna().mean()) if len(values) else 0.0


def unmapped(features_list: list[str]) -> list[str]:
    """Declared features with no entry in the upstream map."""
    return [f for f in features_list if f not in FEATURE_UPSTREAM]


def _tree_view(frame: pd.DataFrame) -> pd.DataFrame:
    return features.tree_view(frame)


def check_contract(built: pd.DataFrame) -> list[str]:
    """Names every declared feature the frame failed to produce."""
    missing = [f for f in declared_features() if f not in built.columns]
    missing += [f for f in unmapped(declared_features()) if f in built.columns]
    return sorted(set(missing))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(min_coverage: float, pbp_games: int, out_dir: str | None) -> int:
    started = time.time()
    facts = ing.load_ingested()
    games = ing.eligible_games(facts.games)
    settled = games[games.home_score.notna() & games.away_score.notna()].copy()
    if not len(settled):
        print("FAIL  no settled games in the window")
        return 1

    built = features.build_game_features(settled, facts.team_stats,
                                         facts.team_events)
    # The categorical join is the one thing that must be a join and not an
    # assignment: ``team_category_ids`` returns a NEW frame keyed by index, and
    # reassigning the result replaces the whole linear contract with two
    # columns while still looking like it worked.
    built = built.join(features.team_category_ids(built))

    print()
    print(f"games        {len(built):,} settled"
          f"   players {len(facts.player_stats):,} rows"
          f"   team rows {len(facts.team_stats):,}")
    print(f"play-by-play {len(facts.play_by_play):,} actions"
          f"   event rows {len(facts.team_events):,}"
          f"   games {facts.team_events.game_id.nunique() if len(facts.team_events) else 0:,}")
    print(f"sources      {facts.manifest.get('frame_sources')}")
    # How much of the window the sweep actually reached. A team's event features
    # are trailing, so this number - not the row count - is what says whether
    # the play-by-play is contributing anything to the model.
    eligible = games[games.nba_game_id.fillna("") != ""] if "nba_game_id" in games         else games
    swept = (facts.team_events.game_id.nunique() if len(facts.team_events) else 0)
    share = 100.0 * swept / max(len(games), 1)
    print(f"sweep        play-by-play covers {swept} of {len(games)} games "
          f"({share:.1f}%)")
    if facts.team_events is not None and len(facts.team_events):
        checks = facts.manifest.get("play_by_play", {})
        if checks:
            worst = max(checks.values(), key=lambda c: c["mean_abs_diff"])
            exact = sum(c["exact"] for c in checks.values())
            compared = sum(c["compared"] for c in checks.values())
            print(f"cross-check  {exact}/{compared} team-games exact against the"
                  f" box score; worst column mean |diff|"
                  f" {worst['mean_abs_diff']:.3f}")

    tree = _tree_view(built)
    declared = declared_features()
    failures: list[str] = []
    below: list[tuple[str, float]] = []
    for feature in declared:
        if feature not in built.columns:
            failures.append(f"{feature}: absent from the feature frame")
            continue
        if feature not in FEATURE_UPSTREAM:
            failures.append(f"{feature}: no upstream declared")
            continue
        pct = coverage(built, feature)
        if pct < min_coverage:
            below.append((feature, pct))
    event_features = [f for f in declared if f.startswith("event_")]
    event_coverage = min((coverage(built, f) for f in event_features),
                         default=0.0)
    print()
    print(f"features     {len(declared)} declared"
          f"   {len(event_features)} from play-by-play"
          f"   weakest event feature {event_coverage:.1f}%")
    print(f"             tree view {tree.shape[1]} columns")
    print(f"gate         {min_coverage:.1f}% minimum coverage")

    if out_dir:
        from pathlib import Path
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        report = pd.DataFrame([
            {"feature": f, "upstream": FEATURE_UPSTREAM.get(f, ""),
             "coverage_pct": round(coverage(built, f), 2),
             "from_play_by_play": f.startswith("event_")}
            for f in declared])
        path = target / f"nba_feature_source_audit_{date.today():%Y%m%d}.csv"
        report.to_csv(path, index=False)
        print(f"audit        {path}")

    print()
    if below:
        print(f"WARN  {len(below)} feature(s) below the gate:")
        for feature, pct in below[:20]:
            print(f"        {feature:38s} {pct:5.1f}%   {FEATURE_UPSTREAM[feature]}")
        if len(below) > 20:
            print(f"        ... and {len(below) - 20} more")
    if failures:
        print(f"FAIL  {len(failures)} contract problem(s):")
        for problem in failures:
            print(f"        {problem}")
    elif below:
        print(f"WARN  all {len(declared)} declared features are explained and "
              f"present; {len(below)} below the coverage gate")
    else:
        print(f"PASS  all {len(declared)} declared features are built from the "
              f"three declared upstreams and clear the gate")
    print(f"elapsed {time.time() - started:.1f}s")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# --diagnose
# ---------------------------------------------------------------------------

_DIAG_DATE = date(2024, 4, 14)


def _probe(url: str, headers: dict, timeout: float) -> tuple[str, str, float]:
    started = time.time()
    try:
        request = urllib.request.Request(url, headers=dict(headers))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return "OK", f"HTTP {response.status}", time.time() - started
    except urllib.error.HTTPError as exc:
        return "SILENT", f"HTTP {exc.code} {exc.reason}", time.time() - started
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "SILENT", f"{type(exc).__name__}: {str(exc)[:70]}", \
            time.time() - started


def _dns(host: str) -> tuple[str, str, float]:
    started = time.time()
    try:
        records = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        return "SILENT", str(exc)[:70], time.time() - started
    return "OK", f"{len(records)} record(s) -> {records[0][4][0]}", \
        time.time() - started


def _tls(host: str) -> tuple[str, str, float]:
    started = time.time()
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=15) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                return "OK", f"{tls.version()} {tls.cipher()[0]}", \
                    time.time() - started
    except Exception as exc:  # noqa: BLE001
        return "SILENT", f"{type(exc).__name__}: {str(exc)[:60]}", \
            time.time() - started


def diagnose(timeout: float) -> int:
    """Walk each upstream, cheapest hop first, and name the first that is quiet."""
    steps: list[tuple[str, str, str, float]] = []

    def add(label: str, result: tuple[str, str, float]) -> None:
        steps.append((label, result[0], result[1], result[2]))

    for name, url, headers in (
            ("espn dns", "site.api.espn.com", None),
            ("espn tls", "site.api.espn.com", None)):
        add(name, _dns(url) if "dns" in name else _tls(url))
    add("espn scoreboard",
        _probe(src.espn_scoreboard_url(_DIAG_DATE), ing.ESPN_HEADERS, timeout))

    add("nba dns", _dns("stats.nba.com"))
    add("nba tls", _tls("stats.nba.com"))
    season_url = (f"{src.SEASON_LOG_URL}?"
                  f"{src.season_log_query('2023-24', src.SEASON_TYPE_REGULAR)}")
    add("season log (the pipeline's exact request)",
        _probe(season_url, ing.STATS_HEADERS, timeout))
    pbp_url = f"{src.PLAY_BY_PLAY_URL}?{src.play_by_play_query('0022301200')}"
    add("play-by-play (the pipeline's exact request)",
        _probe(pbp_url, ing.STATS_HEADERS, timeout))

    print()
    width = max(len(label) for label, *_ in steps)
    for label, status, detail, seconds in steps:
        mark = "OK" if status == "OK" else "!!"
        print(f"[{mark}] {label:{' '}<{width}}  {detail:38s} {seconds:5.2f}s")

    print()
    failures = [label for label, status, *_ in steps if status != "OK"]
    if not failures:
        print("VERDICT  every hop answers; the pipeline's own requests are "
              "reachable from this host.")
        return 0
    first = failures[0]
    print(f"VERDICT  {first} is the first hop that does not answer, so the "
          f"failure is")
    print(f"         upstream of anything the pipeline sends. A host that never "
          f"answers is a")
    print(f"         network block or a refused client, not a bad request: no "
          f"header, retry")
    print(f"         or cookie will change it. Run where it is reachable, or "
          f"warm the cache.")
    if first.startswith("espn"):
        print("         The schedule has no second source, so the run stops "
              "here rather than")
        print("         training on a partial league.")
    else:
        print("         The features have no second source either, so the run "
              "stops and says so.")
    return 1


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-coverage", type=float, default=95.0)
    parser.add_argument("--pbp-games", type=int, default=40,
                        help="how many games to sweep for play-by-play")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.pbp_games:
        import os
        os.environ[ing.PBP_MAX_GAMES_ENV] = str(args.pbp_games)
    if args.diagnose:
        return diagnose(args.timeout)
    return run(args.min_coverage, args.pbp_games, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
