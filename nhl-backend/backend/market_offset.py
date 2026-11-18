"""Market probability as an offset for the moneyline model.

TAB ONLY (decision 2026-11-15): Novig rows are excluded from the training
population. TAB is the only series covering 2023-> and therefore the only one
on which a market offset can be trained. Measured consequence: de-vigged TAB
and de-vigged Novig agree to +0.01pp on the mean and an offset fitted on
either transfers to the other at +0.00008 log loss, so nothing is lost.

The same function serves training and the slate:

    market_prob_home(...) -> de-vigged TAB home probability, or NaN

Gates
-----
* overround must fall in [MIN_OVERROUND, MAX_OVERROUND]. Novig has been seen
  publishing a placeholder 1.001/1.001 (overround 1.998) and TAB prices above
  1.10 are malformed in the same way; a bad pair must never reach the model.
* proportional de-vig (the method verified against the no-vig market in §3 of
  the plan document).

Team codes
----------
Stats Insider writes LA/NJ/SJ/TB where the model writes LAK/NJD/SJS/TBL.
Aliasing is applied in BOTH directions so the join is keyed on one vocabulary.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# model board -> Stats Insider
TO_SI = {"LAK": "LA", "NJD": "NJ", "SJS": "SJ", "TBL": "TB"}
# Stats Insider -> model board
FROM_SI = {v: k for k, v in TO_SI.items()}

MIN_OVERROUND = 1.000
MAX_OVERROUND = 1.100
CLIP = 1e-6

# Both books: TAB is the OFFSET series (every training price is TAB), novig
# is the DISPLAY price — it carries a ~1% margin against TAB's ~4%, so it is
# the best line a bettor can actually take. Requesting `tab` alone is why
# novig never appeared at serve time.
SI_URL = ("https://levy-edge.statsinsider.com.au/pre/{mid}"
          "?Sport=NHL&bookmakers=novig,tab")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def de_vig(home_odds, away_odds):
    """Proportional de-vig. Returns (p_home, p_away, overround).

    Vectorised: scalars in produce scalars out, Series/arrays in produce
    arrays out. Unusable quotes yield NaN for both probabilities while the
    overround is still reported (so callers can log why a row was gated).
    """
    h = np.asarray(home_odds, dtype=float)
    a = np.asarray(away_odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ih, ia = 1.0 / h, 1.0 / a
        ov = ih + ia
    ok = (np.isfinite(ih) & np.isfinite(ia) & (h > 1.0) & (a > 1.0)
          & (ov >= MIN_OVERROUND) & (ov <= MAX_OVERROUND))
    ph = np.where(ok, ih / ov, np.nan)
    pa = np.where(ok, ia / ov, np.nan)
    if np.ndim(home_odds) == 0:
        return float(ph), float(pa), float(ov)
    return ph, pa, ov


def market_prob_home(home_odds, away_odds):
    """De-vigged TAB home probability, NaN when the quote is unusable."""
    p, _, _ = de_vig(home_odds, away_odds)
    return p


def to_logit(p):
    """Probability -> logit, clipping so the offset is always finite."""
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p))


# The price history is a runtime input. Canonical home is data_delivery (it
# is registered in retention_policy.EXACT_MASTER_NAMES so the pruner keeps
# it); NHL_ODDS_HISTORY overrides for a hosted run (Kaggle, CI) that mounts
# the file elsewhere. The old analysis/odds location stays readable so a
# tree that has not been migrated still works.
_ROOT = Path(__file__).resolve().parents[2]
DELIVERY_HISTORY = _ROOT / "nhl-backend" / "data_delivery" / "nhl_odds_history.csv"
LEGACY_HISTORY = _ROOT / "analysis" / "odds" / "nhl_odds_history.csv"
DEFAULT_HISTORY = Path(os.environ.get("NHL_ODDS_HISTORY") or DELIVERY_HISTORY)

_HISTORY_COLS = ["gameday", "home", "away", "market_p_home", "market_logit"]


def load_history(path: str | None = None) -> pd.DataFrame:
    """TAB-only offset series from the stored price history.

    Returns one row per game: gameday (date), home, away (MODEL codes),
    market_p_home, market_logit, overround, price_utc.
    """
    src = DEFAULT_HISTORY if path is None else Path(path)
    if not src.exists() and path is None and "NHL_ODDS_HISTORY" not in os.environ \
            and LEGACY_HISTORY.exists():
        src = LEGACY_HISTORY
    if not src.exists():
        # A hosted run without the history must degrade to "no offset", not
        # crash Phase 5: every row then carries NaN market_logit and the
        # offset arm publishes NaN, per the project's missing-price contract.
        print(f"market_offset.load_history: no price history at {src} — "
              "the offset is unavailable for this run")
        return pd.DataFrame(columns=_HISTORY_COLS)
    d = pd.read_csv(src)
    need = {"date", "home", "away", "tab_homeodds", "tab_awayodds"}
    if not need.issubset(d.columns):
        raise KeyError(f"{path} missing columns: {sorted(need - set(d.columns))}")

    d = d.copy()
    # Stats Insider vocabulary -> model vocabulary (LA -> LAK, NJ -> NJD, ...)
    d["home"] = d["home"].replace(FROM_SI)
    d["away"] = d["away"].replace(FROM_SI)

    p = market_prob_home(d["tab_homeodds"], d["tab_awayodds"])
    d["market_p_home"] = p
    ov = 1.0 / d["tab_homeodds"].astype(float) + 1.0 / d["tab_awayodds"].astype(float)
    d["overround"] = ov
    d["price_utc"] = d["date"]
    d["gameday"] = (pd.to_datetime(d["date"], utc=True)
                    .dt.tz_convert("America/New_York")
                    .dt.tz_localize(None).dt.normalize())

    d = d[d["market_p_home"].notna()].copy()
    d["market_logit"] = to_logit(d["market_p_home"])
    keep = ["gameday", "home", "away", "market_p_home", "market_logit",
            "overround", "price_utc"]
    d = d[keep].drop_duplicates(subset=["gameday", "home", "away"], keep="last")
    return d.sort_values("gameday").reset_index(drop=True)


def attach(game_df: pd.DataFrame,
           market: pd.DataFrame | None = None,
           date_col: str = "gameday",
           path: str | None = None) -> pd.DataFrame:
    """Attach market_p_home / market_logit to a pipeline frame.

    Rows without a usable TAB quote receive NaN, which the caller must treat
    as 'no offset available' — never as 50%.
    """
    m = load_history(path) if market is None else market[["gameday", "home",
                                                         "away",
                                                         "market_p_home",
                                                         "market_logit"]]
    out = game_df.copy()
    out["_day"] = pd.to_datetime(out[date_col]).dt.normalize()
    right = m.rename(columns={"gameday": "_day", "home": "home_team",
                              "away": "away_team"})
    out = out.merge(right, on=["_day", "home_team", "away_team"], how="left")
    return out.drop(columns=["_day"])


#: Every column a live pull owns. A refresh must drop all of them before
#: calling ``attach_live`` - it only drops the first four itself, and a
#: merge would otherwise suffix the price columns ``_x``/``_y``.
QUOTE_FIELDS = ("market_p_home", "market_logit", "price_utc", "stale_minutes",
                "market_home_odds", "market_away_odds", "market_book",
                "market_price_utc")


def quote_age_minutes(frame: pd.DataFrame) -> pd.Series:
    """Minutes since the TAB quote that feeds the offset was last updated.

    ``price_utc`` is TAB's own last-updated instant and TAB is the OFFSET
    book - the training series is TAB-priced, so ``market_logit`` (what the
    slate predicts with) is that number. ``market_price_utc`` is the no-vig
    DISPLAY stamp and can move independently; gating on it would refresh on
    the wrong book's clock, or fail to refresh when the offset input is the
    one that went stale. ``stale_minutes`` on the card is already TAB's age.

    These are the FEED's instants, not ours: the 2026-11-17 run attached
    BOS/LAK from a 23:21:03 stamp and published at 23:34:13 - thirteen
    minutes behind a board that said "priced live" and carried no age.
    """
    if frame is None or not len(frame) or "price_utc" not in frame.columns:
        return pd.Series(dtype=float)
    stamp = pd.to_datetime(frame["price_utc"], errors="coerce", utc=True)
    return (pd.Timestamp.now(tz="UTC") - stamp).dt.total_seconds() / 60.0


def needs_quote_refresh(has_prices: bool, oldest_age_min: float,
                        max_age_min: float) -> bool:
    """Whether serving must re-pull prices before it publishes.

    An unpriced slate, or one whose stamp is unusable, is refreshed; so is
    one whose OLDEST quote is past the bar. Mirrors the NBA gate so the two
    boards age the same way.
    """
    if not has_prices:
        return True
    try:
        oldest = float(oldest_age_min)
    except (TypeError, ValueError):
        return True
    return not np.isfinite(oldest) or oldest > float(max_age_min)


def keep_prior_quotes(fresh: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Overlay a refresh on the prior quote, keeping any field it left empty.

    A game the feed cannot re-price keeps its older number AND its older
    stamp, so the staleness stays visible instead of being disguised as
    fresh by the refresh itself.
    """
    if fresh is None or not len(fresh):
        return fresh
    if prior is None or not len(prior):
        return fresh
    if "game_id" not in fresh.columns or "game_id" not in prior.columns:
        return fresh
    out = fresh.copy()
    prev = prior.drop_duplicates("game_id").set_index(
        prior.drop_duplicates("game_id").game_id.astype(str))
    gid = out.game_id.astype(str)
    for col in QUOTE_FIELDS:
        if col not in out.columns or col not in prev.columns:
            continue
        fill = gid.map(prev[col])
        out[col] = out[col].where(out[col].notna(), fill)
    return out


def _nhl_season(season: int | None = None) -> int:
    """Stats Insider's season label, by ``odds_history.refresh``'s own rule."""
    now = pd.Timestamp.now(tz="UTC")
    if season:
        return int(season)
    cur = 0
    try:
        from odds_history import _read_cursor
        cur = int(_read_cursor()[0] or 0)
    except Exception:  # noqa: BLE001 - no cursor means "follow the calendar"
        cur = 0
    return int(cur if cur >= now.year else now.year)


def _round_listing(season: int, rnd: int) -> list:
    """One round's matches, or [] when the listing is unreachable."""
    try:
        from odds_history import matches_for
        return matches_for(season, rnd) or []
    except Exception:  # noqa: BLE001 - an unreachable feed prices nothing
        return []


def _abv(team) -> str:
    if isinstance(team, dict):
        return str(team.get("Abv") or team.get("DisplayName") or "").upper()
    return str(team or "").upper()


def live_match_index(season: int | None = None, max_round: int = 270,
                     workers: int = 8) -> tuple[dict, dict]:
    """(home, away, ET date) -> SIMatchID, straight from the feed.

    NHL twin of ``nba-backend/backend/market_rundown.live_match_index``.
    ``attach_live`` resolved match ids ONLY from the stored price history, so
    a game the scraper had not written yet - or a hosted run with no history
    at all - served an unpriced card while the feed was answering. The round
    listing carries the same ids, so ask it instead.

    The whole season is swept concurrently (270 rounds, ~8 workers, a few
    seconds). A cursor-anchored window was tried first and is WRONG on this
    feed: 2026-11-18, the rounds around the cursor (94 = Dec 31) held three
    December fixtures while every November game lived below round 55 - a
    window would have "resolved" three ids and gone silent on tonight's
    board. The cap has to reach the end of a season too: past ones run to
    251-259 rounds, so a 160-round sweep would miss the run-in. Stats Insider writes LA/NJ/SJ/TB where the model writes
    LAK/NJD/SJS/TBL, so keys use the model vocabulary, exactly as the
    history's do.
    """
    from concurrent.futures import ThreadPoolExecutor
    from zoneinfo import ZoneInfo

    season = _nhl_season(season)
    rounds = range(1, int(max_round) + 1)

    def one(rnd):
        return _round_listing(season, rnd)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        fetched = list(ex.map(one, rounds))
    by_day: dict = {}
    by_pair: dict = {}
    for items in fetched:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            md = item.get("MatchData") or item
            mid = md.get("SIMatchID")
            tip = pd.to_datetime(md.get("Date"), errors="coerce", utc=True)
            away = _abv(md.get("AwayTeam"))
            home = _abv(md.get("HomeTeam"))
            away = FROM_SI.get(away, away)
            home = FROM_SI.get(home, home)
            if not mid or pd.isna(tip) or not away or not home:
                continue
            day = (tip.tz_convert(ZoneInfo("America/New_York"))
                   .normalize().tz_localize(None))
            by_day[(home, away, day)] = mid
            by_pair[(home, away)] = mid
    return by_day, by_pair


def attach_live(slate_df: pd.DataFrame, lookback_days: int = 4,
                path: str | None = None) -> pd.DataFrame:
    """Attach the CURRENT TAB price to a slate, fetched at serve time.

    Match ids come from the stored history's recent rows (the slate itself
    carries no Stats Insider id), falling back to the live round listing when
    that history carries none, then the price is pulled live — never the
    archived snapshot — so the served offset is the number at publish time.

    Ids are keyed on (home, away, Eastern game date) FIRST. Keying on the pair
    alone is not unique: a club meets the same opponent twice inside the
    lookback window routinely (124 consecutive NHL meetings sit 4 days or less
    apart), and the pair's most recent stored row can be the EARLIER game while
    tonight's row has not been scraped yet — the pair key then serves the wrong
    meeting's price with no NaN to show for it. With a slate date a price must
    come from that very date; the pair is only used when the slate carries no
    date to check (2026-11-16).
    """
    out = slate_df.copy()
    for c in ("market_p_home", "market_logit", "price_utc", "stale_minutes"):
        if c in out.columns:
            out = out.drop(columns=[c])
    try:
        hist = pd.read_csv(DEFAULT_HISTORY if path is None else path)
    except Exception as exc:  # noqa: BLE001
        print(f"market_offset.attach_live: history unavailable ({exc})")
        hist = None
    by_day: dict = {}
    by_pair: dict = {}
    if hist is not None:
        hist["_d"] = pd.to_datetime(hist["date"], utc=True)
        hist["_gd"] = (hist["_d"].dt.tz_convert("America/New_York")
                       .dt.normalize().dt.tz_localize(None))
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        recent = hist[hist["_d"] >= cutoff].copy()
        recent["home"] = recent["home"].replace(FROM_SI)
        recent["away"] = recent["away"].replace(FROM_SI)
        recent = recent.dropna(subset=["match_id"]).sort_values("_d")
        for h_, a_, d_, mid in zip(recent["home"], recent["away"],
                                   recent["_gd"], recent["match_id"]):
            by_day[(h_, a_, d_)] = mid
            by_pair[(h_, a_)] = mid
    if not by_day:
        # NBA parity (2026-11-18): the stored history is a fast path, not the
        # only way to know a game's id. A game the scraper has not written
        # yet - or a hosted run with no history file - keeps its price.
        by_day, by_pair = live_match_index()
        print(f"market_offset.attach_live: resolved {len(by_day)} match id(s) "
              "from the live round listing (no history ids)")
    days = (pd.to_datetime(out["gameday"]).dt.normalize()
            if "gameday" in out.columns
            else pd.Series([pd.NaT] * len(out), index=out.index))
    want = []
    for a_, h_, d_ in zip(out["away_team"], out["home_team"], days):
        if pd.notna(d_):
            want.append(by_day.get((h_, a_, d_)))
        else:
            want.append(by_pair.get((h_, a_)))
    targets = sorted({m for m in want if isinstance(m, str)})
    if not targets:
        print("market_offset.attach_live: no match ids for this slate")
        return out
    live = fetch_live(targets)
    live = live[live["market_p_home"].notna()].drop_duplicates("match_id")
    keep = ["match_id", "market_p_home", "market_logit", "price_utc",
            "stale_minutes", "market_home_odds", "market_away_odds",
            "market_book", "market_price_utc"]
    out["_mid"] = want
    out = out.merge(live[keep].rename(columns={"match_id": "_live_mid"}),
                    left_on="_mid", right_on="_live_mid", how="left")
    return out.drop(columns=[c for c in ("_mid", "_live_mid") if c in out.columns])


def fetch_live(match_ids, timeout: float = 20.0) -> pd.DataFrame:
    """Serve-time pull: one GET per match, TAB price only.

    Returns match_id, home/away (MODEL codes), market_p_home, market_logit,
    overround, price_utc, fetched_utc, stale_minutes.
    """
    now = datetime.now(timezone.utc)
    rows = []
    for mid in match_ids:
        # home/away are seeded: a failed GET used to return a row WITHOUT
        # them, and attach_live's column selection then raised KeyError
        # instead of leaving the game unpriced (2026-11-16).
        rec = {"match_id": mid, "home": None, "away": None,
               "market_p_home": np.nan,
               "market_logit": np.nan, "overround": np.nan,
               "price_utc": None, "fetched_utc": now, "stale_minutes": np.nan,
               # Display pair for the card: no-vig when that book prices the
               # match (it is a near-zero-margin book, so its price is the
               # best line a bettor can take), otherwise TAB.
               "market_home_odds": np.nan, "market_away_odds": np.nan,
               "market_book": None, "market_price_utc": None}
        try:
            req = urllib.request.Request(SI_URL.format(mid=mid), headers=HEADERS)
            j = json.load(urllib.request.urlopen(req, timeout=timeout))
            md = j.get("MatchData") or {}
            _bd = j.get("BettingData") or {}
            if isinstance(_bd, str):
                try:
                    _bd = json.loads(_bd.replace("'", '"'))
                except Exception:  # noqa: BLE001
                    _bd = {}
            tab = _bd.get("tab") or {}
            nov = _bd.get("novig") or {}
            if isinstance(tab, str):
                try:
                    tab = json.loads(tab.replace("'", '"'))
                except Exception:  # noqa: BLE001
                    tab = {}
            if isinstance(nov, str):
                try:
                    nov = json.loads(nov.replace("'", '"'))
                except Exception:  # noqa: BLE001
                    nov = {}
            if not isinstance(tab, dict):
                tab = {}
            if not isinstance(nov, dict):
                nov = {}
            home_si = (md.get("HomeTeam") or {}).get("Abv")
            away_si = (md.get("AwayTeam") or {}).get("Abv")
            rec["home"] = FROM_SI.get(home_si, home_si)
            rec["away"] = FROM_SI.get(away_si, away_si)
            p = market_prob_home(tab.get("HomeOdds"), tab.get("AwayOdds"))
            rec["market_p_home"] = p
            rec["market_logit"] = float(to_logit(p)) if np.isfinite(p) else np.nan
            ho, ao = tab.get("HomeOdds"), tab.get("AwayOdds")
            if ho and ao:
                rec["overround"] = 1.0 / float(ho) + 1.0 / float(ao)
            rec["price_utc"] = tab.get("LastUpdated")
            # No-vig first: same event, tighter margin, so its price is the
            # best available to a bettor. TAB stays the OFFSET book — the
            # training series is TAB-priced, so switching the offset here
            # would score the models against a distribution they never fit.
            if nov.get("HomeOdds") and nov.get("AwayOdds"):
                rec["market_home_odds"] = float(nov["HomeOdds"])
                rec["market_away_odds"] = float(nov["AwayOdds"])
                rec["market_book"] = "novig"
                rec["market_price_utc"] = (nov.get("LastUpdated")
                                           or tab.get("LastUpdated"))
            elif tab.get("HomeOdds") and tab.get("AwayOdds"):
                rec["market_home_odds"] = float(tab["HomeOdds"])
                rec["market_away_odds"] = float(tab["AwayOdds"])
                rec["market_book"] = "tab"
                rec["market_price_utc"] = tab.get("LastUpdated")
            if tab.get("LastUpdated"):
                t = pd.Timestamp(tab["LastUpdated"]).tz_convert("UTC")
                rec["stale_minutes"] = (now - t.to_pydatetime()).total_seconds() / 60.0
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:120]
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mk = load_history()
    print(f"TAB offset series: {len(mk)} games  {mk.gameday.min().date()} -> {mk.gameday.max().date()}")
    print(f"  overround mean {mk.overround.mean():.4f}  min {mk.overround.min():.4f}  max {mk.overround.max():.4f}")
    print(f"  market_p_home mean {mk.market_p_home.mean():.4f}  logit mean {mk.market_logit.mean():.4f}")
    print(mk.head(3).to_string(index=False))
