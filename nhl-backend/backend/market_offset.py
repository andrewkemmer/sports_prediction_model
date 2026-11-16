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


def attach_live(slate_df: pd.DataFrame, lookback_days: int = 4,
                path: str | None = None) -> pd.DataFrame:
    """Attach the CURRENT TAB price to a slate, fetched at serve time.

    Match ids come from the stored history's recent rows (the slate itself
    carries no Stats Insider id), then the price is pulled live — never the
    archived snapshot — so the served offset is the number at publish time.
    """
    out = slate_df.copy()
    for c in ("market_p_home", "market_logit", "price_utc", "stale_minutes"):
        if c in out.columns:
            out = out.drop(columns=[c])
    try:
        hist = pd.read_csv(DEFAULT_HISTORY if path is None else path)
    except Exception as exc:  # noqa: BLE001
        print(f"market_offset.attach_live: history unavailable ({exc})")
        return out
    hist["_d"] = pd.to_datetime(hist["date"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    recent = hist[hist["_d"] >= cutoff].copy()
    recent["home"] = recent["home"].replace(FROM_SI)
    recent["away"] = recent["away"].replace(FROM_SI)
    ids = (recent.dropna(subset=["match_id"])
           .drop_duplicates(subset=["home", "away"], keep="last")
           .set_index(["home", "away"])["match_id"].to_dict())
    pairs = list(out[["home_team", "away_team"]].itertuples(index=False, name=None))
    want = [ids[p] for p in pairs if p in ids]
    if not want:
        print("market_offset.attach_live: no match ids for this slate")
        return out
    live = fetch_live(sorted(set(want)))
    live = live[live["market_p_home"].notna()].drop_duplicates(
        subset=["home", "away"], keep="last")
    keep = ["home", "away", "market_p_home", "market_logit",
            "price_utc", "stale_minutes", "market_home_odds",
            "market_away_odds", "market_book", "market_price_utc"]
    return out.merge(live[keep].rename(columns={"home": "home_team",
                                                "away": "away_team"}),
                     on=["home_team", "away_team"], how="left")


def fetch_live(match_ids, timeout: float = 20.0) -> pd.DataFrame:
    """Serve-time pull: one GET per match, TAB price only.

    Returns match_id, home/away (MODEL codes), market_p_home, market_logit,
    overround, price_utc, fetched_utc, stale_minutes.
    """
    now = datetime.now(timezone.utc)
    rows = []
    for mid in match_ids:
        rec = {"match_id": mid, "market_p_home": np.nan,
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
