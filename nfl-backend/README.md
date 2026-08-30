# NFL Backend — Game-Frame Ingestion

The NFL analogue of `mlb-backend/`'s canonical game-level frame: one row per
**decided** game with final scores and closing betting lines — the atomic
structure every downstream NFL pipeline, feature, and model will hang off.

```
nfl-backend
├── backend            # ingestion + feature-admission modules + tests
│   ├── nfl_game_frame.py   # game-level frame builder
│   └── nfl_features.py     # feature candidates + coverage/leakage gate (v1, no model)
├── data_delivery      # generated artifacts (frame CSV + feature record JSON)
└── README.md
```

## Data source

**`nflreadpy` v0.1+** — the official Python port of the nflverse project and
the current standard. `nfl_data_py` was archived in late 2025 and is NOT used
for new work. Data is pulled from the nflverse CDN:

- `load_schedules(seasons)` — schedule feed carrying final scores, result,
  and closing betting lines per game (verified: 1,675 decided games across
  2019–2024, 100% line coverage).
- `load_pbp(seasons)` — play-by-play, joined in only for per-game play counts
  (`n_plays`) and game_id cross-validation. Only `game_id`/`play_id` are
  materialized to pandas (the full ~300k × 370 frame stays in polars).

## Schema — one row per decided game

| Column | Type | Description |
| --- | --- | --- |
| `game_id` | str | nflverse game id (e.g. `2019_01_KC_JAX`) |
| `season` | int | season year |
| `week` | int | week number (postseason weeks are 18+) |
| `game_type` | str | `REG` / `WC` / `DIV` / `CON` / `SB` |
| `gameday` | date | game date (YYYY-MM-DD, venue-local) |
| `away_team` | str | away team abbreviation |
| `home_team` | str | home team abbreviation |
| `away_score` | int | final away points (non-null ⇒ decided) |
| `home_score` | int | final home points (non-null ⇒ decided) |
| `result` | float | home margin = home_score − away_score (positive = home win) |
| `total` | float | combined final points |
| `spread_line` | float | closing spread — **positive = home team favored, negative = away team favored** (nflverse schedules dictionary convention) |
| `total_line` | float | closing over/under total |
| `n_plays` | int | play count from play-by-play (NaN if pbp missing) |

## Decided-frame rules (encoded once in `canonical_decided_frame`)

1. **Post-game only** — `away_score`, `home_score`, and `result` all non-null;
   undecided/pregame rows never enter the frame.
2. **Deterministic dedup** — one row per `game_id`; the latest `gameday` wins
   (stable mergesort, ties resolve by input order).
3. **Stable chronological order** — mergesort by `gameday` only; within-day
   input order is preserved (same discipline as `mlb-backend/backend/frames.py`).

## Usage

```bash
cd nfl-backend/backend
python3 nfl_game_frame.py                              # default: 2019–2024
python3 nfl_game_frame.py --seasons 2021 2022 2023 2024
```

Writes:
- `data_delivery/nfl_game_level_features.csv` — canonical frame (overwritten)
- `data_delivery/nfl_game_level_features_YYYYMMDD.csv` — dated snapshot

Each run re-validates the spike's go/no-go criteria and prints them: per-season
decided-game counts, missing-score rows, duplicate game_ids, betting-line
coverage %, home win rate (~52–57% NFL norm), combined points per game
(~44–46 norm), and the sha256 of the written CSV.

## Validation

```bash
cd nfl-backend/backend
python3 -m unittest test_nfl_game_frame -v        # or: pytest test_nfl_game_frame.py
```

Tests pin the decided-frame rules on synthetic frames (no network) and, when
the artifact exists, re-verify the spike's table on
`data_delivery/nfl_game_level_features.csv`: per-season counts
(267/269/285/284/285/285 for 2019–2024), 0 duplicate game_ids, 0 missing
scores, and a spot-check of 2019 W1 KC@JAX (40–26, ESPN-verified) plus the
spread-line sign convention. The CSV is a generated artifact and is not
committed; the artifact tests skip gracefully until the module has been run.

## Run on Kaggle

`kaggle_nfl_run.ipynb` (repo root) runs the phase-driven `backend/master_pipeline.py`
in `/kaggle/working` (secrets → `MY_GITHUB_TOKEN`, fresh clone, pip install
`nflreadpy polars scikit-learn lightgbm xgboost pandas numpy joblib gitpython`,
then ingest → features → moneyline ensemble → sync): `python nfl-backend/backend/master_pipeline.py`,
with `--no-push --features-csv <csv> --out-dir <dir>` as the local dry path.

**Stale-cleanup retention rule**: cleanup never deletes `nfl_game_level_features.csv`
(exact-name protected) or `models/`, and keeps a dated `nfl_moneyline_v1_<d>.json` /
`nfl_feature_v1_<d>.json` while `<d>` still renders a board (i.e. `<d>` is a distinct
`game_date` in the moneyline record(s)' `games[]`) — the board-backed rule that
prevented the MLB run-engine regression (a navigable date never loses the artifact
it renders); other dated files stay on the 48h retention window.

## Notes / known risks

- **Line provenance**: `spread_line`/`total_line` are nflverse closing lines
  and were spot-verified against ESPN/NFL.com finals; a full reconciliation
  against a real odds vendor (e.g. SportsDataIO) is the recommended one-off
  before treating the entire 6-season line set as exact for modeling.
- **Live data**: nflverse is (infrequently) delayed on gameday — fine for
  historical training frames; live/real-time ingestion would want the
  SportsDataverse/ESPN fallback path wired before relying on it.
- **Feature admission only (v1, no model).** `backend/nfl_features.py` builds
  leakage-safe raw candidates (ELO diff, trailing form, rest-days diff, net
  yards/play diff, dome flag, home anchor), audits coverage / point-in-time
  leakage, and gates them into the v1 set; it does NOT train any model. The
  walk-forward / ensemble / sealed-holdout stage is a separate next task.
  Each run writes `data_delivery/nfl_feature_v1_YYYYMMDD.json` (candidates,
  coverage + leakage audit, correlation + univariate-AUC tables, final v1 set
  with inclusion/exclusion reasons). 2025 is never in the learning data (it is
  not in the decided frame at all); all AUC is computed on seasons < 2025.
  No shared-frontend or mlb-backend changes live in this directory.
