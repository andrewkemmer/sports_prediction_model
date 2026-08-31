"""One-time curation of the NFL venue table (run with network; not part of the suite).

Builds backend/nfl_stadiums.csv from REAL public data, keyed on the verified
nflverse games.csv 'stadium' column (45 distinct strings across 2018-2025):
  - lat/lon: Wikipedia geocoordinates (en.wikipedia.org API, redirects resolved)
  - altitude_ft: Open-Elevation API (SRTM), rounded to whole feet
  - tz: IANA timezone of the venue's metro (documented city map below)
  - teams: canonical home team(s) (modal non-neutral home stadium per team from
    the real 2018-2025 schedule data), '' for neutral venues

Sources are recorded per row in the 'source' column. Unknown stadiums are
resolved to NaN at runtime (this file only curates the 45 real names).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BACKEND_DIR = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (sports_prediction_model nfl venue curation)"}

# name -> (facility, city-key) for every distinct stadium string in games.csv 2018-2025
# (verified 2026-08-31 via nflreadpy.load_schedules). Same facility renamed over the
# window -> separate raw-name rows sharing one facility; coordinates identical.
VENUES: list[tuple[str, str, str]] = [
    # (raw games.csv stadium name, facility key, city tz key)
    ("MetLife Stadium", "MetLife", "east_rutherford"),
    ("SoFi Stadium", "SoFi", "inglewood"),
    ("New Era Field", "Highmark", "orchard_park"),
    ("Lincoln Financial Field", "LincolnFinancial", "philadelphia"),
    ("M&T Bank Stadium", "MBank", "baltimore"),
    ("NRG Stadium", "NRG", "houston"),
    ("Mercedes-Benz Superdome", "Superdome", "new_orleans"),
    ("Ford Field", "FordField", "detroit"),
    ("State Farm Stadium", "StateFarm", "glendale"),
    ("Gillette Stadium", "Gillette", "foxborough"),
    ("Levi's Stadium", "Levis", "santa_clara"),
    ("Raymond James Stadium", "RaymondJames", "tampa"),
    ("AT&T Stadium", "ATT", "arlington"),
    ("Lambeau Field", "Lambeau", "green_bay"),
    ("Hard Rock Stadium", "HardRock", "miami_gardens"),
    ("Nissan Stadium", "Nissan", "nashville"),
    ("Soldier Field", "Soldier", "chicago"),
    ("Lucas Oil Stadium", "LucasOil", "indianapolis"),
    ("FirstEnergy Stadium", "FirstEnergy", "cleveland"),
    ("FedExField", "FedExField", "landover"),
    ("Mercedes-Benz Stadium", "MBS", "atlanta"),
    ("U.S. Bank Stadium", "USBank", "minneapolis"),
    ("Bank of America Stadium", "BankOfAmerica", "charlotte"),
    ("TIAA Bank Stadium", "TIAA", "jacksonville"),
    ("Empower Field at Mile High", "MileHigh", "denver"),
    ("Allegiant Stadium", "Allegiant", "las_vegas"),
    ("GEHA Field at Arrowhead Stadium", "Arrowhead", "kansas_city"),
    ("Lumen Field", "Lumen", "seattle"),
    ("Acrisure Stadium", "Acrisure", "pittsburgh"),
    ("Paycor Stadium", "Paycor", "cincinnati"),
    ("Arrowhead Stadium", "Arrowhead", "kansas_city"),
    ("CenturyLink Field", "Lumen", "seattle"),
    ("Heinz Field", "Acrisure", "pittsburgh"),
    ("Paul Brown Stadium", "Paycor", "cincinnati"),
    ("Sports Authority Field at Mile High", "MileHigh", "denver"),
    ("Ring Central Coliseum", "OaklandColiseum", "oakland"),
    ("Oakland-Alameda County Coliseum", "OaklandColiseum", "oakland"),
    ("Los Angeles Memorial Coliseum", "LAColiseum", "los_angeles"),
    ("StubHub Center", "StubHub", "carson"),
    ("Tottenham Stadium", "Tottenham", "london"),
    ("Wembley Stadium", "Wembley", "london"),
    ("Allianz Arena", "Allianz", "munich"),
    ("Deutsche Bank Park", "DeutscheBankPark", "frankfurt"),
    ("Azteca Stadium", "Azteca", "mexico_city"),
    ("Arena Corinthians", "Corinthians", "sao_paulo"),
]

# Wikipedia article title per facility key
WIKI_TITLES = {
    "MetLife": "MetLife Stadium", "SoFi": "SoFi Stadium",
    "Highmark": "Highmark Stadium (New York)", "LincolnFinancial": "Lincoln Financial Field",
    "MBank": "M&T Bank Stadium", "NRG": "NRG Stadium", "Superdome": "Caesars Superdome",
    "FordField": "Ford Field", "StateFarm": "State Farm Stadium", "Gillette": "Gillette Stadium",
    "Levis": "Levi's Stadium", "RaymondJames": "Raymond James Stadium", "ATT": "AT&T Stadium",
    "Lambeau": "Lambeau Field", "HardRock": "Hard Rock Stadium", "Nissan": "Nissan Stadium",
    "Soldier": "Soldier Field", "LucasOil": "Lucas Oil Stadium", "FirstEnergy": "Huntington Bank Field",
    "FedExField": "Northwest Stadium", "MBS": "Mercedes-Benz Stadium", "USBank": "U.S. Bank Stadium",
    "BankOfAmerica": "Bank of America Stadium", "TIAA": "EverBank Stadium",
    "MileHigh": "Empower Field at Mile High", "Allegiant": "Allegiant Stadium",
    "Arrowhead": "Arrowhead Stadium", "Lumen": "Lumen Field", "Acrisure": "Acrisure Stadium",
    "Paycor": "Paycor Stadium", "OaklandColiseum": "Oakland Coliseum",
    "LAColiseum": "Los Angeles Memorial Coliseum", "StubHub": "Dignity Health Sports Park",
    "Tottenham": "Tottenham Hotspur Stadium", "Wembley": "Wembley Stadium",
    "Allianz": "Allianz Arena", "DeutscheBankPark": "Waldstadion (Frankfurt)",
    "Azteca": "Estadio Azteca", "Corinthians": "Neo Química Arena",
}

# city -> IANA timezone (tz database fact; documented assignment)
CITY_TZ = {
    "east_rutherford": "America/New_York", "inglewood": "America/Los_Angeles",
    "orchard_park": "America/New_York", "philadelphia": "America/New_York",
    "baltimore": "America/New_York", "houston": "America/Chicago",
    "new_orleans": "America/Chicago", "detroit": "America/Detroit",
    "glendale": "America/Phoenix", "foxborough": "America/New_York",
    "santa_clara": "America/Los_Angeles", "tampa": "America/New_York",
    "arlington": "America/Chicago", "green_bay": "America/Chicago",
    "miami_gardens": "America/New_York", "nashville": "America/Chicago",
    "chicago": "America/Chicago", "indianapolis": "America/Indiana/Indianapolis",
    "cleveland": "America/New_York", "landover": "America/New_York",
    "atlanta": "America/New_York", "minneapolis": "America/Chicago",
    "charlotte": "America/New_York", "jacksonville": "America/New_York",
    "denver": "America/Denver", "las_vegas": "America/Los_Angeles",
    "kansas_city": "America/Chicago", "seattle": "America/Los_Angeles",
    "pittsburgh": "America/New_York", "cincinnati": "America/New_York",
    "oakland": "America/Los_Angeles", "los_angeles": "America/Los_Angeles",
    "carson": "America/Los_Angeles", "london": "Europe/London",
    "munich": "Europe/Berlin", "frankfurt": "Europe/Berlin",
    "mexico_city": "America/Mexico_City", "sao_paulo": "America/Sao_Paulo",
}

WIKI_API = "https://en.wikipedia.org/w/api.php"


def wiki_coords_batch(titles: list[str]) -> dict[str, tuple[float, float, str]]:
    """Batched Wikipedia query for all titles -> {title: (lat, lon, page_url)}.

    Repeats any request whose responses don't cover every requested title
    (429 / partial response), then retries still-missing titles individually.
    Pages that exist but carry no coordinates are returned as nan entries.
    """
    S = requests.Session()
    S.headers.update(UA)
    out: dict[str, tuple[float, float, str]] = {}
    pending = sorted(set(titles))

    def _get(chunk: list[str]):
        params = {"action": "query", "format": "json", "formatversion": "2",
                  "prop": "coordinates", "redirects": "1",
                  "titles": "|".join(chunk)}
        for attempt in range(6):
            r = S.get(WIKI_API, params=params, timeout=60)
            if r.status_code == 429:
                print(f"    429: sleeping {30 * (attempt + 1)}s")
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"wiki still 429 for {chunk}")

    for i in range(0, len(pending), 25):
        chunk = pending[i:i + 25]
        for d in range(5):
            data = _get(chunk)
            seen = set()
            for p in data.get("query", {}).get("pages", []):
                t = p.get("title")
                if t is None or p.get("missing"):
                    continue
                seen.add(t)
                if p.get("coordinates"):
                    c = p["coordinates"][0]
                    out[t] = (float(c["lat"]), float(c["lon"]),
                              f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}")
            missing = [t for t in chunk if t not in seen]
            if not missing:
                break
            print(f"    partial batch ({len(chunk) - len(missing)}/{len(chunk)}): retrying {missing}")
            chunk = missing
            time.sleep(5)
    # last-resort per-title calls for stragglers
    stragglers = [t for t in pending if t not in out]
    for t in stragglers:
        d = _get([t])
        for p in d.get("query", {}).get("pages", []):
            if p.get("title") is None or p.get("missing"):
                continue
            if p.get("coordinates"):
                c = p["coordinates"][0]
                out[p["title"]] = (float(c["lat"]), float(c["lon"]),
                                    f"https://en.wikipedia.org/wiki/{p['title'].replace(' ', '_')}")
            else:
                out[p["title"]] = (float("nan"), float("nan"),
                                    f"https://en.wikipedia.org/wiki/{p['title'].replace(' ', '_')}")
            break
    return out


def nominatim_coords(query: str) -> tuple[float, float]:
    """OpenStreetMap Nominatim geocode (real data) for pages Wikipedia lacks coords on."""
    url = "https://nominatim.openstreetmap.org/search"
    r = requests.get(url, params={"q": query, "format": "json", "limit": 1},
                     headers={**UA, "Referer": "https://localhost"}, timeout=60)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError(f"nominatim: nothing for {query}")
    return float(rows[0]["lat"]), float(rows[0]["lon"])


def elevation_ft_batch(lat_lon: list[tuple[float, float]]) -> list[float]:
    """SRTM elevations via one batched Open-Elevation lookup."""
    locs = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in lat_lon)
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locs}"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    # the Open-Elevation API returns METERS; the contract stores feet.
    return [3.28084 * float(x["elevation"]) for x in r.json()["results"]]


def derive_team_home_venue() -> dict[str, str]:
    """Canonical home venue per team = modal non-neutral home stadium from the
    real 2018-2025 schedule (nflreadpy)."""
    import nflreadpy  # network
    sched = nflreadpy.load_schedules(list(range(2018, 2026))).to_pandas()
    home = sched[sched["location"] != "Neutral"].dropna(subset=["stadium"])
    counts = home.groupby(["home_team", "stadium"]).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=False).groupby("home_team").head(1)
    return dict(zip(counts["home_team"], counts["stadium"]))


def main() -> None:
    team_home = derive_team_home_venue()
    print("team -> home venue (real schedule modal):")
    for t, s in sorted(team_home.items()):
        print(f"   {t:3s} -> {s}")

    titles = [WIKI_TITLES[f] for _, f, _ in VENUES]
    coords = wiki_coords_batch(titles)
    print(f"\nresolved {len(coords)}/{len(set(titles))} wiki titles")

    # Nominatim fallback for pages Wikipedia has no coordinates on (nan entries)
    nom_fallback = {
        "Nissan Stadium": "Nissan Stadium, Nashville, Tennessee",
    }
    for f, city in [(f, c) for _, f, c in VENUES
                    if pd.isna(coords.get(WIKI_TITLES[f], (float("nan"), 0, ""))[0])]:
        print(f"    fallback nominatim for {WIKI_TITLES[f]}")
        la, lo = nominatim_coords(nom_fallback.get(WIKI_TITLES[f], f"{WIKI_TITLES[f]}, {city}"))
        coords[WIKI_TITLES[f]] = (la, lo, f"nominatim:{WIKI_TITLES[f]}")

    rows = []
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pts = [(coords.get(WIKI_TITLES[f], (float("nan"), float("nan"), ""))[0],
            coords.get(WIKI_TITLES[f], (float("nan"), float("nan"), ""))[1])
           for _, f, _ in VENUES]
    try:
        elevs = elevation_ft_batch(pts)
        print(f"elevations: {len(elevs)}")
    except Exception as e:
        print("elevation batch failed:", type(e).__name__, e)
        elevs = [float("nan")] * len(VENUES)

    for (raw_name, facility, city), (lat, lon), elev in zip(VENUES, pts, elevs):
        page = coords.get(WIKI_TITLES[facility], ("", "", ""))[2]
        teams = ",".join(sorted(t for t, s in team_home.items() if s == raw_name))
        rows.append({
            "stadium": raw_name, "facility": facility,
            "teams": teams,
            "lat": round(lat, 6), "lon": round(lon, 6),
            "altitude_ft": float(round(elev)) if pd.notna(elev) else None,
            "tz": CITY_TZ[city],
            "source": f"wiki:{page} elevation:open-elevation(srtm) fetched:{fetched}",
        })
        print(f"{raw_name:38s} ({teams or 'neutral':8s}) {lat:.4f},{lon:.4f} {elev:8.1f}ft tz={CITY_TZ[city]}")
    out = BACKEND_DIR / "nfl_stadiums.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()