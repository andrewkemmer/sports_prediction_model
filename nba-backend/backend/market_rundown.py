"""Market open/close moneyline ingestion (TheRundown v2) + model-vs-market comparison.

Key:  THERUNDOWN_API_KEY env, else ~/.rundown_key. Never commit the key.
Rate limit on current plan: ~1 req/s (sleep between calls).

Pipeline
  1. python market_rundown.py fetch --date 2026-10-30 --tag open
     python market_rundown.py fetch --date 2026-10-30 --tag close   # near tipoff
     -> nba-backend/run_diagnostics/market_{tag}_{date}.csv  (raw book prices)
  2. python market_rundown.py compare --oof <oof.csv> --pred <predictions_history.csv>
       --open <market_open_DATE.csv> --close <market_close_DATE.csv> [--out ...]
     De-vigs each book (multiplicative), averages fair probs across books,
     joins to decided games on (date, away, home), prints 5%-bin
     logloss/Brier/AUC/ECE for model vs market-open vs market-close.

Coverage note (verified 2026-10-30): this plan returns 7 days of history,
scheduled-game prices only (finals come back with markets stripped), and no
open/close fields -- so "open" = first snapshot after lines post, "close" =
last snapshot before tipoff. Full-season training-set backfill needs a plan
with deeper history; this harness runs unchanged once that key lands.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

import requests

API = "https://therundown.io/api/v2"
SPORT_ID = 4  # NBA
BOOKS = {"19": "DraftKings", "22": "BetMGM", "23": "FanDuel"}
BOOK_IDS = ",".join(BOOKS)
DIAG = "nba-backend/run_diagnostics"

# A plausible two-way price sits close to fair. Stats Insider has published a
# placeholder 1.001/1.001 (overround 1.998) and TAB has emitted malformed
# quotes well above fair; neither may reach the model. Same gate as the NHL
# matcher (nhl-backend/backend/market_offset.py).
MIN_OVERROUND = 1.000
MAX_OVERROUND = 1.100


def _key() -> str:
    k = os.environ.get("THERUNDOWN_API_KEY", "").strip()
    if k:
        return k
    p = os.path.expanduser("~/.rundown_key")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    raise SystemExit("No key: set THERUNDOWN_API_KEY or write ~/.rundown_key")


def _get(session: requests.Session, url: str, **params):
    time.sleep(1.2)  # plan rate limit ~1 rps
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_slate(date: str, tag: str) -> str:
    s = requests.Session()
    s.headers["X-TheRundown-Key"] = _key()
    j = _get(s, f"{API}/sports/{SPORT_ID}/events/{date}",
             market_ids="1", affiliate_ids=BOOK_IDS, main_line="true")
    rows = []
    for e in j.get("events") or []:
        tn = e.get("teams_normalized") or []
        full = {t["abbreviation"]: (t["name"] + " " + t["mascot"]).upper()
                for t in tn}
        hm = {v: k for k, v in full.items()}
        ab = [t["abbreviation"] for t in tn if t.get("is_away")]
        hb = [t["abbreviation"] for t in tn if t.get("is_home")]
        if not ab or not hb:
            continue
        away, home = ab[0], hb[0]
        for m in e.get("markets", []) or []:
            if m.get("period_id") != 0:
                continue
            for p in m.get("participants", []):
                a = hm.get((p.get("name") or "").upper())
                if not a:
                    continue
                for ln in p.get("lines", []):
                    for aff, pr in (ln.get("prices") or {}).items():
                        rows.append({
                            "event_id": e["event_id"], "event_date": e.get("event_date"),
                            "away": away, "home": home, "team": a,
                            "side": "HOME" if a == home else "AWAY",
                            "book": aff, "price": pr.get("price"),
                            "delta": pr.get("price_delta"),
                            "updated": pr.get("updated_at"),
                        })
    out = f"{DIAG}/market_{tag}_{date.replace('-', '')}.csv"
    os.makedirs(DIAG, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
    return out


def de_vig(home_odds, away_odds):
    """Proportional de-vig. Returns (p_home, p_away, overround).

    Ported from ``nhl-backend/backend/market_offset.de_vig`` so the two
    sports cannot drift: the same [MIN_OVERROUND, MAX_OVERROUND] gate rejects
    a placeholder pair on either side, and an unusable quote yields NaN for
    both probabilities while the overround is still reported.
    """
    import numpy as np

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


def implied(x: float) -> float:
    x = float(x)
    return 100.0 / (x + 100.0) if x > 0 else -x / (-x + 100.0)


def consensus(path: str) -> dict:
    """(date, away, home) -> mean de-vigged fair home prob across books."""
    games: dict = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            games.setdefault((r["event_date"][:10], r["away"], r["home"]), []).append(r)
    out = {}
    for key, rs in games.items():
        fairs = []
        for book, g in itertools.groupby(sorted(rs, key=lambda x: x["book"]),
                                         key=lambda x: x["book"]):
            g = list(g)
            if len(g) != 2:
                continue  # partial book: skip, never impute
            h = next(x for x in g if x["side"] == "HOME")
            a = next(x for x in g if x["side"] == "AWAY")
            ih, ia = implied(h["price"]), implied(a["price"])
            fairs.append(ih / (ih + ia))
        if fairs:
            out[key] = sum(fairs) / len(fairs)
    return out


# Closing-line policy (validated 2026-11-16, NBA parity with
# ``nhl-backend/backend/odds_history``): Stats Insider keeps serving the LAST
# PRE-GAME quote after tip-off. Across all 5,502 captured prices the book's
# own ``LastUpdated`` stamp sits a median 4 minutes AFTER the scheduled start
# (p10 -9.5, p90 -0.9; 90% at or past the tip, 2% more than 6 hours before),
# i.e. the feed freezes the line at kickoff and the numbers do not move
# in-play. A differing quote is therefore always a LATER PRE-GAME line, never
# an in-play one, so the newest stamp wins for every match, past or present.
# ``MatchData.UTCDate`` is NOT that stamp -- it is a fixture artefact, absent
# on 67% of rows -- and must never be read as the price's vintage.


def attach_market(frame: "pd.DataFrame", si_path: str) -> "pd.DataFrame":
    """One de-vigged Stats Insider price per decided game, or NaN.

    A DIRECT join, on the tip-off instant. The feed's ``UTCDate`` is when the
    line was PUBLISHED (median a day either side of tip-off) and its final
    score can differ from the box score by a point, so joining on either of
    those needs a tolerance or a tiebreak - an inferred match. ``tipoff_utc``
    is the scheduled start, converted to the Eastern date the schedule uses,
    and the pair carries the rest. No tolerance, no score, no elimination:
    a game either has exactly one price or it is unpriced.

    Two guards keep the number honest rather than convenient:
    * an ambiguous candidate (two prices for one game, or one price claiming
      two games) is DROPPED, not resolved by a tiebreak;
    * a quote whose overround falls outside [MIN_OVERROUND, MAX_OVERROUND] is
      rejected the way the NHL matcher rejects it - a placeholder pair must
      never reach the model.

    Returns the frame with ``market_p_home`` (proportional de-vig) appended.
    """
    import numpy as np
    import pandas as pd
    from zoneinfo import ZoneInfo

    si = pd.read_csv(si_path)
    if "tipoff_utc" not in si.columns:
        raise ValueError(
            "odds file has no 'tipoff_utc' column - it predates tip-off "
            "capture, so it can only be joined by inference; re-run "
            "scrape_statsinsider_nba.py")
    tip = pd.to_datetime(si.tipoff_utc, errors="coerce", utc=True)
    si["et_date"] = (tip.dt.tz_convert(ZoneInfo("America/New_York"))
                     .dt.normalize().dt.tz_localize(None))
    si = si[si.home_odds_dec.notna() & si.away_odds_dec.notna()
            & si.home_odds_dec.gt(1) & si.away_odds_dec.gt(1)].copy()
    p_home, _, overround = de_vig(si.home_odds_dec, si.away_odds_dec)
    si["p_home"] = p_home
    si = si[pd.Series(overround, index=si.index).notna()
            & si.et_date.notna()].copy()
    # NHL parity: the newest quote per match is the closing line, because
    # Stats Insider keeps serving the last pre-game price after kickoff. The
    # most recent book stamp wins; the capture time breaks a tie.
    si["_lu"] = pd.to_datetime(si.get("book_last_updated"), errors="coerce",
                               utc=True)
    si["_fu"] = pd.to_datetime(si.get("fetched_utc"), errors="coerce", utc=True)
    si = si.sort_values(["_lu", "_fu"]).drop_duplicates("sim_match_id",
                                                        keep="last")

    g = frame.copy()
    g["_d"] = pd.to_datetime(g.gameday).dt.normalize()
    key = ["away_team", "home_team", "_d"]
    right = (si[["away_abv", "home_abv", "et_date", "p_home", "_lu",
                 "tipoff_utc"]]
             .dropna(subset=["et_date"])
             .rename(columns={"_lu": "price_utc",
                              "tipoff_utc": "market_tipoff_utc"}))
    cand = g[key + ["game_id"]].merge(
        right, left_on=key, right_on=["away_abv", "home_abv", "et_date"],
        how="inner")
    # Ambiguity is refused, never resolved: name it and drop it.
    per_game = cand.groupby("game_id").size()
    per_price = cand.groupby(["away_abv", "home_abv", "et_date"]).size()
    ambiguous_games = set(per_game[per_game > 1].index)
    ambiguous_keys = set(per_price[per_price > 1].index)
    cand = cand[~cand.game_id.isin(ambiguous_games)]
    mask_key = [(a, h, d) in ambiguous_keys
                for a, h, d in zip(cand.away_abv, cand.home_abv, cand.et_date)]
    cand = cand[[not m for m in mask_key]]
    # ``reindex`` rather than a column select: an empty candidate set after
    # the filters must still yield an empty price table, not a KeyError.
    price = (cand.reindex(columns=["game_id", "p_home", "price_utc",
                                   "market_tipoff_utc"])
             .dropna(subset=["game_id"]).drop_duplicates("game_id"))
    out = g.merge(price, on="game_id", how="left").drop(columns=["_d"])
    out = out.rename(columns={"p_home": "market_p_home"})
    if "price_utc" in out.columns:
        # Staleness is visible, never inferred: minutes between this game's
        # tip-off and the book's own stamp. Stats Insider stamps the closing
        # quote AT kickoff and never moves it in-play, so the close reads
        # ~-5 min (stamped just after the start) and a genuinely old price
        # reads as a large positive number.
        tip = pd.to_datetime(out["market_tipoff_utc"], errors="coerce", utc=True)
        stamp = pd.to_datetime(out["price_utc"], errors="coerce", utc=True)
        out["pregame_minutes"] = ((tip - stamp).dt.total_seconds() / 60.0)
    return out


SI_BASE = "https://levy-edge.statsinsider.com.au"
SI_LIVE = SI_BASE + "/pre/{mid}?best_bets=true&bookmakers=novig,tab"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

#: Everything ``attach_live`` owns on the slate. Dropped before attaching so a
#: stale archived price can never survive a serve-time refresh.
LIVE_COLS = ["market_p_home", "market_logit", "overround", "price_utc",
             "fetched_utc", "stale_minutes", "pregame_minutes",
             "market_home_odds", "market_away_odds", "market_book",
             "market_tipoff_utc", "market_price_utc"]


def _book_blob(payload: dict, name: str) -> dict:
    """One book's block from a /pre payload, whatever shape it arrives in."""
    import json

    bd = payload.get("BettingData") or {}
    b = bd.get(name) if isinstance(bd, dict) else None
    if isinstance(b, str):
        try:
            b = json.loads(b.replace("'", '"'))
        except Exception:  # noqa: BLE001
            return {}
    return b if isinstance(b, dict) else {}


def fetch_live(match_ids, workers: int = 8, timeout: float = 20.0):
    """Serve-time pull: one GET per match, de-vigged TAB price.

    Mirrors ``nhl-backend/backend/market_offset.fetch_live``: TAB is the
    OFFSET book (every training price is TAB, so the served offset has to be
    the same series), and the no-vig pair is carried only for display.
    """
    import numpy as np
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor

    now = pd.Timestamp.now(tz="UTC")

    def one(mid: str) -> dict:
        rec = {"match_id": mid, "market_p_home": np.nan,
               "market_logit": np.nan, "overround": np.nan,
               "price_utc": None, "fetched_utc": now, "stale_minutes": np.nan,
               "pregame_minutes": np.nan, "market_home_odds": np.nan,
               "market_away_odds": np.nan, "market_book": None,
               "market_tipoff_utc": None, "market_price_utc": None,
               "error": None}
        try:
            r = requests.get(SI_LIVE.format(mid=mid), headers=HEADERS,
                             timeout=timeout)
            r.raise_for_status()
            j = r.json()
            md = j.get("MatchData") or {}
            tab, nov = _book_blob(j, "tab"), _book_blob(j, "novig")
            p, _, ov = de_vig(tab.get("HomeOdds"), tab.get("AwayOdds"))
            rec["market_p_home"] = float(p)
            rec["overround"] = float(ov)
            if np.isfinite(p):
                q = min(max(float(p), 1e-6), 1.0 - 1e-6)
                rec["market_logit"] = float(np.log(q / (1.0 - q)))
            tip = md.get("Date")
            rec["market_tipoff_utc"] = tip
            stamp = tab.get("LastUpdated")
            rec["price_utc"] = stamp
            if stamp:
                t = pd.Timestamp(stamp).tz_convert("UTC")
                rec["stale_minutes"] = (now - t).total_seconds() / 60.0
                if tip:
                    rec["pregame_minutes"] = (
                        pd.Timestamp(tip).tz_convert("UTC")
                        - t).total_seconds() / 60.0
            if nov.get("HomeOdds") and nov.get("AwayOdds"):
                rec["market_home_odds"] = float(nov["HomeOdds"])
                rec["market_away_odds"] = float(nov["AwayOdds"])
                rec["market_book"] = "novig"
                # The DISPLAY pair's own stamp. ``price_utc`` stays TAB's -
                # that is the offset input and the serve gate's clock.
                rec["market_price_utc"] = (nov.get("LastUpdated")
                                           or tab.get("LastUpdated"))
            elif tab.get("HomeOdds") and tab.get("AwayOdds"):
                rec["market_home_odds"] = float(tab["HomeOdds"])
                rec["market_away_odds"] = float(tab["AwayOdds"])
                rec["market_book"] = "tab"
                rec["market_price_utc"] = tab.get("LastUpdated")
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:120]
        return rec

    ids = [str(m) for m in match_ids]
    if not ids:
        return pd.DataFrame(columns=list(LIVE_COLS))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame(list(ex.map(one, ids)))


def _nba_season_label(today=None) -> int:
    """Stats Insider's season label for a date: 2026 means 2026-27."""
    import pandas as pd
    d = pd.Timestamp(today or pd.Timestamp.now(tz="UTC"))
    return int(d.year if d.month >= 9 else d.year - 1)


def live_match_index(season: int | None = None, max_round: int = 80,
                     workers: int = 8, timeout: float = 20.0) -> tuple[dict, dict]:
    """(away_abv, home_abv, ET date) -> SIMatchID, straight from the feed.

    ``attach_live`` used to resolve match ids only from
    ``run_diagnostics/si_nba_all_seasons.csv`` - a GITIGNORED file that cannot
    exist on the production host. There the attach returned an unpriced slate
    and the served board carried no market column at all. The round listing
    carries the same ids, so ask it instead; the CSV is now only a fast path
    for a machine that happens to have one.

    Rounds are sub-weekly and the season's last round is around 40 in the
    current schedule, so the scan stops after three consecutive empty rounds
    rather than running to ``max_round``.
    """
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor
    from zoneinfo import ZoneInfo

    season = int(season if season is not None else _nba_season_label())
    url = ("https://levy-edge.statsinsider.com.au/round/matches"
           "?Sport=NBA&Round={r}&Season={s}&strip=true&best_bets=true"
           "&bookmakers=tab")

    def one(round_number: int):
        try:
            r = requests.get(url.format(r=round_number, s=season),
                             headers=HEADERS, timeout=timeout)
            if r.status_code != 200:
                return round_number, []
            data = r.json()
        except Exception:  # noqa: BLE001 - an unreachable feed prices nothing
            return round_number, []
        return round_number, data if isinstance(data, list) else []

    rounds = list(range(1, int(max_round) + 1))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        fetched = list(ex.map(one, rounds))
    by_day, by_pair, empty_run = {}, {}, 0
    for round_number, items in fetched:
        if not items:
            empty_run += 1
            if empty_run >= 3 and round_number > 20:
                break
            continue
        empty_run = 0
        for item in items:
            md = item.get("MatchData") or {}
            mid = md.get("SIMatchID")
            tip = pd.to_datetime(md.get("Date"), errors="coerce", utc=True)
            away = ((md.get("AwayTeam") or {}).get("Abv") or "").upper()
            home = ((md.get("HomeTeam") or {}).get("Abv") or "").upper()
            if not mid or pd.isna(tip) or not away or not home:
                continue
            day = tip.tz_convert(ZoneInfo("America/New_York")).normalize().tz_localize(None)
            by_day[(away, home, day)] = mid
            by_pair[(away, home)] = mid
    return by_day, by_pair


def attach_live(slate_df, lookback_days: int = 4, path: str | None = None,
                fetcher=None, workers: int = 8):
    """Attach the CURRENT TAB price to a slate, fetched at serve time.

    The NBA twin of ``nhl-backend/backend/market_offset.attach_live``, with one
    difference that matters: NBA rows carry the tip-off instant, so the match
    id is looked up on (away, home, ET tip date) first and only falls back to
    the pair when the slate has no date - a pair alone cannot tell tonight's
    meeting from the one three weeks ago.

    Prices come from ``fetch_live`` (never the archive), so the served offset
    is the number at publish time. A game the feed cannot price keeps NaN,
    which the caller must read as "no offset", never as 50%.
    """
    import pandas as pd
    from zoneinfo import ZoneInfo

    out = slate_df.copy()
    out = out.drop(columns=[c for c in LIVE_COLS if c in out.columns])
    src = path or f"{DIAG}/si_nba_all_seasons.csv"
    now = pd.Timestamp.now(tz="UTC")
    by_day, by_pair = {}, {}
    hist = None
    if os.path.exists(src):
        try:
            hist = pd.read_csv(src)
        except Exception as exc:  # noqa: BLE001
            print(f"market_rundown.attach_live: history unreadable ({exc})")
    if hist is not None:
        hist["_tip"] = pd.to_datetime(hist.tipoff_utc, errors="coerce", utc=True)
        hist = hist.dropna(subset=["_tip", "sim_match_id"])
        hist["_et"] = (hist._tip.dt.tz_convert(ZoneInfo("America/New_York"))
                       .dt.normalize().dt.tz_localize(None))
        recent = (hist[hist._tip >= now - pd.Timedelta(days=lookback_days)]
                  .sort_values("_tip"))
        # ``itertuples`` renames underscore columns, so iterate the values.
        for a, h, d, mid in zip(recent["away_abv"], recent["home_abv"],
                                recent["_et"], recent["sim_match_id"]):
            by_day[(a, h, d)] = mid
            by_pair[(a, h)] = mid
    if not by_day:
        by_day, by_pair = live_match_index()
        print(f"market_rundown.attach_live: resolved {len(by_day)} match id(s) "
              "from the live round listing (no history file)")
    days = (pd.to_datetime(out.gameday).dt.normalize() if "gameday" in out.columns
            else pd.Series([pd.NaT] * len(out), index=out.index))
    want = [by_day.get((a, h, d)) or by_pair.get((a, h))
            for a, h, d in zip(out.away_team, out.home_team, days)]
    targets = sorted({m for m in want if isinstance(m, str)})
    if not targets:
        print("market_rundown.attach_live: no match ids for this slate")
        return out
    if fetcher is not None:
        live = fetcher(targets)
    else:
        live = fetch_live(targets, workers=workers)
    live = live[live["market_p_home"].notna()].drop_duplicates("match_id")
    keep = ["match_id"] + [c for c in LIVE_COLS if c in live.columns]
    out["_mid"] = want
    out = out.merge(live[keep], left_on="_mid", right_on="match_id", how="left")
    return out.drop(columns=[c for c in ("_mid", "match_id") if c in out.columns])


def compare(oof_path: str, pred_path: str, open_path: str, close_path: str,
            out_path: str | None = None):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    oof = pd.read_csv(oof_path)
    oof = oof[oof["grades_pooled"].astype(bool)]
    pred = pd.read_csv(pred_path, usecols=["game_id", "game_date", "home_team", "away_team"])
    oof = oof.merge(pred, on="game_id", how="left")
    mo, mc = consensus(open_path), consensus(close_path)
    oof["mkt_open"] = oof.apply(
        lambda r: mo.get((str(r["gameday"])[:10], r["away_team"], r["home_team"]), np.nan), axis=1)
    oof["mkt_close"] = oof.apply(
        lambda r: mc.get((str(r["gameday"])[:10], r["away_team"], r["home_team"]), np.nan), axis=1)
    print(f"graded={len(oof)} open_cov={oof['mkt_open'].notna().mean():.3f} "
          f"close_cov={oof['mkt_close'].notna().mean():.3f}")

    y = oof["home_win"].to_numpy(float)
    series = {"model": oof["p_ensemble_calibrated"].to_numpy(float),
              "mkt_open": oof["mkt_open"].to_numpy(float),
              "mkt_close": oof["mkt_close"].to_numpy(float)}
    eps = 1e-12
    rows = []
    for lo in np.arange(0, 1.0, 0.05):
        hi = lo + 0.05
        for name, p_all in series.items():
            m = pd.notna(p_all) & (p_all >= lo) & (p_all < hi if hi < 1 else p_all <= hi)
            n = int(m.sum())
            if not n:
                continue
            yy, pp = y[m], np.clip(p_all[m].astype(float), eps, 1 - eps)
            acc = float(yy.mean())
            rows.append({
                "bin": f"{lo:.2f}-{hi:.2f}", "series": name, "n": n,
                "base_rate": round(acc, 4), "mean_p": round(float(pp.mean()), 4),
                "logloss": round(float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).mean()), 4),
                "brier": round(float(((pp - yy) ** 2).mean()), 4),
                "auc": round(float(roc_auc_score(yy, pp)), 4) if len(np.unique(yy)) > 1 else float("nan"),
                "cal_gap": round(abs(acc - float(pp.mean())), 4),
            })
    res = pd.DataFrame(rows)
    if out_path:
        res.to_csv(out_path, index=False)
        print("wrote", out_path)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(res.to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--date", required=True)
    f.add_argument("--tag", required=True, choices=["open", "close"])
    c = sub.add_parser("compare")
    c.add_argument("--oof", required=True)
    c.add_argument("--pred", required=True)
    c.add_argument("--open", required=True)
    c.add_argument("--close", required=True)
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch_slate(a.date, a.tag)
    else:
        compare(a.oof, a.pred, a.open, a.close, a.out)


if __name__ == "__main__":
    main()
