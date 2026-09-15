"""API-coverage gap catalog for the Feature Decision Workbook.

Grounded in the ACTUAL API payloads the pipeline touches — verified against
the extractor code (data_ingestion.py, results.py, weather.py, umpires.py,
ingestion.py, build_pbp_defense.py) — not in what past engineering attempts
happened to try.

Three layers:
  API_SOURCES        one row per API: what the payload serves vs what we extract
  UNUSED_API_FIELDS  specific unused payload fields -> concrete candidate features
  LOW_VALUE_FIELDS   fields intentionally cataloged-but-not-recommended (honesty)

NEVER COMMITTED — scratch tool.
"""

# --------------------------------------------------------------------------- #
# Layer 1 — source-level inventory: serves vs extracts
# --------------------------------------------------------------------------- #

API_SOURCES = [
    # (API & endpoint, What the payload serves (plain english),
    #  What the pipeline extracts today, What is left unused)
    ("ESPN Scoreboard API\nsite.api.espn.com .../baseball/mlb/scoreboard",
     "Per game: teams, game state, final scores, probable pitchers, venue, "
     "start time — plus odds, broadcasts, a weather block, team records "
     "(overall / home / away / last-10 / streak), and a day-vs-night flag.",
     "Teams, state, scores, probablePitcher, venue name, start time.",
     "Odds, records/streaks, dayNight flag, weather block, broadcasts."),
    ("MLB StatsAPI Schedule API\nstatsapi.mlb.com/api/v1/schedule",
     "Per game: full team records, series description, doubleheader "
     "structure, venue detail, and hydrations (probablePitcher person stats, "
     "linescore, officials).",
     "Used only as the umpire-assignment source (hydrate=officials) and "
     "pitcher-name cross-checks (hydrate=probablePitcher).",
     "Records, series/doubleheader metadata, linescore hydration, "
     "game-time officials beyond home plate."),
    ("MLB StatsAPI Game Feed\n/api/v1.1/game/{pk}/feed/live",
     "Per game: venue fieldInfo (dimensions, wall heights, roof type, "
     "elevation, capacity), posted lineups (battingOrder), full boxscore, "
     "play-by-play with every pitch's measured location and the batter's "
     "strike-zone bounds, umpire assignments, weather block.",
     "Only gameData.weather (temp, wind, condition) — used as a gap-filler "
     "for Open-Meteo. The rest of the feed is never opened.",
     "fieldInfo, boxscore, battingOrder, officials, play-level pitch data."),
    ("MLB StatsAPI Standings API\n/api/v1/standings",
     "Official as-of-date standings: division/wild-card rank, games back, "
     "clinched/eliminated status.",
     "Never called — standings are reconstructed locally from game results "
     "(compute_season_records).",
     "Playoff-pressure context (clinch margin, elimination), official "
     "division ranks."),
    ("Open-Meteo (archive + forecast)\napi.open-meteo.com/v1/*",
     "Hourly at the park: temperature, humidity, wind speed AND direction, "
     "surface pressure — plus precipitation, rain/showers/snowfall, wind "
     "gusts, cloud cover, and a weather-code (drizzle/thunderstorm).",
     "temperature_2m, relative_humidity_2m, wind_speed_10m, "
     "wind_direction_10m, surface_pressure (5 of ~11 weather variables).",
     "Precipitation, gusts, cloud cover, weather code, dew point."),
    ("Statcast via pybaseball\n(data_ingestion.ingest_statcast)",
     "Per pitch: ~40 measured columns — release speed AND effective (perceived) "
     "speed, extension, spin, movement, location vs the batter's own strike "
     "zone (sz_top/sz_bot), batted-ball type and location, which of the 9 "
     "fielders touched the ball, defensive alignment.",
     "A curated subset: velocity, spin, zone flag, movement, launch metrics, "
     "xwOBA/xBA, run expectancies. Fielders and alignments are explicitly "
     "listed in UNUSED_COLS and dropped at load.",
     "sz_top/sz_bot, effective_speed, release_extension, spin_axis, "
     "plate_x/plate_z, hit_location, bb_type, fielder_2..9, alignments, "
     "post-score columns."),
]

# --------------------------------------------------------------------------- #
# Layer 2 — specific unused payload fields -> candidate feature recipes
# (category, API source, field in payload, what it is, feature recipe,
#  build effort, trial confidence, raw support in the codebase today)
# --------------------------------------------------------------------------- #

UNUSED_API_FIELDS = [
    # ---- ESPN scoreboard ----
    ("League Context", "ESPN Scoreboard", "competitions[0].odds",
     "The betting market's read on the game: moneyline odds and a runs total.",
     "Implied win probability from the moneyline (odds -> %). The strongest "
     "single external predictor known; trial as a candidate column and let "
     "the RFE measure whether the market adds signal beyond Elo/form.",
     "Low", "High",
     "Field arrives in the scoreboard payload already fetched daily; zero new "
     "ingestion — just parse it."),

    ("Schedule & Travel", "ESPN Scoreboard", "competitions[0].dayNight",
     "Day game vs night game flag (also derivable from the start timestamp).",
     "is_day_game (and the home/away diff). Day games run cooler and score "
     "differently; cheap one-column trial.",
     "Low", "High",
     "Payload already fetched; also derivable from start_time_utc already stored."),

    ("Team Form", "ESPN Scoreboard", "competitor.records (home/away/streak)",
     "Each team's record split home vs away, last-10 record, and win/loss "
     "streak — maintained by ESPN, up to date on fetch.",
     "Home-vs-away record split as a diff (today only overall win_pct exists); "
     "last-10 form and streak length as momentum features. All PIT-safe by "
     "fetch date.",
     "Low", "High",
     "Payload already fetched. Note: our own rolling records are computed from "
     "game results — ESPN's home/away split is the piece we never derived."),

    ("Weather", "ESPN Scoreboard", "competitions[0].weather",
     "ESPN's game-time weather display (temp/wind as shown in their UI).",
     "Cross-check against our Open-Meteo observation: a data-quality flag and "
     "an observed-vs-forecast delta feature.",
     "Low", "Medium",
     "Present for most outdoor games in the payload already fetched."),

    # ---- StatsAPI schedule ----
    ("League Context", "StatsAPI Schedule", "seriesDescription / doubleHeaders",
     "Series game number (Game 1 vs Game 2 of a series), doubleheader flags, "
     "series type.",
     "Series-game-number and doubleheader second-game fatigue/context flags. "
     "Identity for doubleheaders exists today; the fatigue weighting does not.",
     "Low", "High",
     "Same schedule endpoint the umpire module already calls (umpires.py)."),

    ("League Context", "StatsAPI Standings", "api/v1/standings (as-of date)",
     "Official standings: games back, wild-card position, clinched/eliminated.",
     "Playoff-pressure flag (in contention / clinched / eliminated) as of each "
     "game date — PIT-safe. Teams out of the race behave differently.",
     "Medium", "High",
     "New call (one per date, cacheable); complements local compute_season_records."),

    # ---- StatsAPI game feed ----
    ("Stadium / Park", "StatsAPI Game Feed", "gameData.venue.fieldInfo",
     "The park's physical card: wall distances (LF/LC/C/RC/RL), wall heights, "
     "roof type, elevation, capacity.",
     "Static park-shape features: deep-corners, high-wall, elevation bands "
     "(the Coors effect), roof type — complements the rolling park factor "
     "which is estimated from history.",
     "Low", "High",
     "Feed is already contacted per game for weather gap-filling "
     "(results.fetch_statsapi_weather); same response carries fieldInfo."),

    ("Umpires", "StatsAPI Game Feed", "liveData.boxscore.officials",
     "Game-time umpire crew: home plate AND base umpires, as assigned for "
     "that specific game.",
     "Same-day umpire assignment (no season-lag). Joins directly onto the "
     "existing umpire_map.csv to unlock current-day umpire features.",
     "Medium", "High",
     "Feed already contacted per game; umpire_map/umpire_stats masters exist."),

    ("Umpires", "StatsAPI Game Feed", "liveData.plays pitchData (plateX/plateZ, szTop/szBot)",
     "Every called pitch's measured location versus THAT batter's strike-zone "
     "bounds, per umpire.",
     "The real umpire strike-zone: called-strike rate by zone band vs league "
     "average — a measured zone, not a K/BB proxy.",
     "High", "High",
     "Play-by-play arrives in the same feed; pybaseball pbp cache holds the "
     "Statcast mirror (plate_x/plate_z) to build rolling per-umpire rates."),

    ("Lineup (posted)", "StatsAPI Game Feed", "boxscore.teams.home.battingOrder",
     "The ACTUAL posted lineup — player IDs in batting order — once the team "
     "files it.",
     "Posted-lineup wOBA vs probable expectation (the lineup surprise diff). "
     "Today's pipeline models expected lineups only; this is the real thing.",
     "Medium", "High",
     "Feed contacted per game; lineups.parquet master exists for storage."),

    ("Bullpen", "StatsAPI Game Feed", "boxscore pitchers (per game staff usage)",
     "Which relievers pitched and how much, in every completed game.",
     "Bullpen fatigue: pitches/innings per reliever over the last 2-3 days, "
     "aggregated to a team availability score (diff).",
     "Medium", "High",
     "Derivable from completed-game feeds or Statcast pbp already ingested."),

    # ---- Open-Meteo ----
    ("Weather", "Open-Meteo", "precipitation / rain / showers / snowfall",
     "Actual rain and snow at the park, hourly.",
     "is_rain + rain amount at first pitch (diff). Wet-ball grip and footing "
     "effects; also a clean venue-quality flag.",
     "Low", "High",
     "One string edit: add the variable to _HOURLY_VARS in weather.py — both "
     "archive and forecast endpoints serve it."),

    ("Weather", "Open-Meteo", "wind_gusts_10m",
     "Gust speed alongside the sustained wind we already pull.",
     "Gustiness (gust minus sustained, diff). Gusty air destabilizes breaking "
     "balls and fly-ball carry.",
     "Low", "High",
     "Same one-line _HOURLY_VARS addition."),

    ("Weather", "Open-Meteo", "cloud_cover / weathercode",
     "Cloud cover % and a weather-category code (drizzle, thunderstorm...).",
     "Night-game cooling proxy and a severe-weather flag.",
     "Low", "High",
     "Same one-line _HOURLY_VARS addition."),

    ("Weather", "Open-Meteo", "dew_point_2m",
     "Dew point — humidity's more baseball-relevant cousin (the number "
     "analysts quote for ball carry).",
     "Refine the existing air-density feature: dew point is a cleaner humidity "
     "input than relative humidity at temperature extremes.",
     "Low", "Medium",
     "Same one-line _HOURLY_VARS addition (model-dependent availability)."),

    # ---- Statcast ----
    ("Starting Pitching", "Statcast (pybaseball)", "effective_speed / release_extension",
     "Perceived velocity (accounts for extension) and how far toward the plate "
     "the pitcher releases.",
     "Perceived-velo edge vs league average (diff) — extension is why two "
     "94-mph pitchers play differently.",
     "Low", "High",
     "Columns exist in the raw Statcast frame; add to STATCAST_COLS in "
     "ingestion.py (they are simply not requested today)."),

    ("Starting Pitching", "Statcast (pybaseball)", "sz_top / sz_bot",
     "Each batter's own strike-zone height for that at-bat.",
     "Pitcher zone-attack profile: called/pitched rate at zone edges vs heart, "
     "normalized per opposing lineup (diff).",
     "Medium", "High",
     "Columns exist in the raw feed; currently dropped at load (UNUSED_COLS "
     "does not list them but STATCAST_COLS never requests them)."),

    ("Starting Pitching", "Statcast (pybaseball)", "spin_axis + spin_rate by pitch type",
     "Spin efficiency and axis per pitch — the shape behind the movement we "
     "already store.",
     "Arsenal-quality indices per pitch family (fastball spin efficiency, "
     "breaking-ball shape) as diffs.",
     "Medium", "High",
     "spin_rate already ingested; spin_axis is in the schema but not requested."),

    ("Pitch Matchups", "Statcast (pybaseball)", "plate_x / plate_z (per pitch)",
     "Exact pitch location — where the pitcher actually attacked.",
     "Attack-plane heat vs batter handedness (pitcher's preferred quadrant per "
     "matchup) — richer than the current aggregate K-rate matchups.",
     "High", "High",
     "plate_x/plate_z are in STATCAST_COLS already — data is present; this is "
     "pure feature engineering on existing data."),

    ("Defense", "Statcast (pybaseball)", "fielder_2 .. fielder_9",
     "Which fielder (position number) touched each batted ball.",
     "Team out-conversion rate on comparable batted balls — a real defensive "
     "skill measure (diff). The Defense category currently has ZERO features "
     "anywhere in the pool.",
     "High", "High",
     "Explicitly listed in ingestion.py UNUSED_COLS today — the loader "
     "deliberately drops them (build_pbp_defense.py keeps a copy in the pbp "
     "cache, unused)."),

    ("Defense", "Statcast (pybaseball)", "if_fielding_alignment / of_fielding_alignment",
     "Infield/outfield shift positioning on each batted ball.",
     "Shift-on share and alignment-vs-batter matching (diff).",
     "Medium", "High",
     "Also in UNUSED_COLS — dropped at load."),

    ("Defense", "Statcast (pybaseball)", "hit_location / bb_type",
     "Where on the field the ball was hit and its batted-ball class.",
     "Zone-level defensive range: conversion rate by hit-location zone (diff).",
     "Medium", "High",
     "hit_location is in the feed; bb_type in the defense cache columns."),
]

# Cataloged for honesty — present in payloads, NOT recommended as features.
LOW_VALUE_FIELDS = [
    ("ESPN Scoreboard", "broadcasts",
     "TV/radio listing — no pre-game signal; useful only for provenance."),
    ("Statcast", "post_home_score / post_away_score / delta_home_win_exp",
     "In-game state after each play — not available pre-game; would leak if "
     "misapplied. Kept out on purpose."),
    ("ESPN Scoreboard", "odds movement (requires repeated polls)",
     "Opening/closing line movement needs re-polling at multiple times; the "
     "single pre-game snapshot (recommended above) does not."),
]

# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

CATEGORY_ORDER = [
    "Team", "Starting Pitching", "Bullpen", "Batting / Lineup", "Lineup (posted)",
    "Pitch Matchups", "Defense", "Weather", "Stadium / Park", "Umpires",
    "Schedule & Travel", "League Context", "Team Form",
]
