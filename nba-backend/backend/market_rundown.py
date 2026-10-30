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
