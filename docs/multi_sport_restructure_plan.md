# Multi-Sport Restructure Plan (sports_prediction_model)

**Status:** planning only — no production code changed.
**Date:** 2026-08-28
**Scope:** restructure from a single-sport (MLB) layout to a multi-sport layout
(`mlb-backend/` + future `nfl-backend/`, `nba-backend/`, `nhl-backend/`, shared
top-level `frontend/`), keeping the MLB pipeline and canonical test suite green
at every commit.

---

## 1. Executive summary

Today the entire product — backend, frontend, and artifact sink — lives under
`mlb-bet-predictor/`. The target is:

```
sports_prediction_model/
├── frontend/                    # shared Streamlit app (sport toggle)
├── mlb-backend/
│   ├── backend/                 # pipeline code (git-moved)
│   └── data_delivery/           # MLB artifacts (git-moved)
├── nfl-backend/                 # future (backend/ + data_delivery/)
├── nba-backend/                 # future
└── nhl-backend/                 # future
```

The migration is 5 phases (A–E). The two atomic, behavior-relevant commits are:

- **Phase B:** `git mv mlb-bet-predictor/frontend frontend` (frontend path math
  and the 3 frontend test files change in the same commit).
- **Phase C:** `git mv mlb-bet-predictor mlb-backend` + flip the artifact path
  prefix `"mlb-bet-predictor" → "mlb-backend"` in the fetcher and pipeline in
  the **same commit**, so GitHub raw URLs, the local fallback, and the Colab
  sync move atomically.

The decided-frame / fold-signature machinery is **not touched** — only the
path constants that surround it.

---

## 2. STEP 1 — Inventory (grounded in the current tree)

### 2.1 Current tree (220 files under `mlb-bet-predictor/`)

```
sports_prediction_model/
├── .gitignore                  # root; created 2026-08-28 (__pycache__/ *.pyc *.pyo .pytest_cache/)
├── ai_agent_prompt.txt
├── colab_mlb_run.ipynb         # clones to /content/sports_prediction_model, runs master_pipeline
├── kaggle_mlb_run.ipynb        # same orchestration (Kaggle)
└── mlb-bet-predictor/
    ├── .gitignore              # __pycache__/ *.pyc *.pyo .env *.joblib
    ├── README.md
    ├── PLAN-drift-desync-calibration-ablation.md
    ├── _fetch_roofs.py         # root-level helper (ROOT = parent → data_delivery/)
    ├── backend/                # 95 files: pipeline + 50 test_*.py + tuners + run_*.py ablations
    │   ├── __init__.py         # empty (backend used both as package `backend.*` and bare `config.*`)
    │   ├── config.py           # ROOT_DIR = parent.parent; DATA_DELIVERY_DIR = ROOT_DIR/"data_delivery"
    │   ├── master_pipeline.py  # Colab orchestration — HARDCODES "mlb-bet-predictor" (7×)
    │   ├── github_sync.py      # legacy GitPython sync helper (config.DATA_DELIVERY_DIR)
    │   ├── … ~45 pipeline/feature/run_*.py modules
    │   └── fixtures/           # run_engine_monitor_v1_*.json (test pins)
    ├── frontend/               # Streamlit multi-page app
    │   ├── Home.py             # st.Page nav (5 pages) + brand header
    │   ├── utils.py            # artifact fetcher: raw.githubusercontent + local fallback
    │   ├── todays_games.py, power_rankings.py, model_calibration.py,
    │   │   model_monitor.py, markets.py, market_diagnostics.py
    │   ├── .streamlit/config.toml, streamlit_theme.toml, requirements.txt
    └── data_delivery/          # 44 artifacts (canonical sink; Colab pushes, app reads)
```

### 2.2 Every hardcoded `"mlb-bet-predictor"` reference (functional)

| File | Ref count | What breaks on rename |
|---|---|---|
| `backend/master_pipeline.py` | 7 | `sys.path.insert(repo_dir/"mlb-bet-predictor"/"backend")`, `os.chdir(repo_dir/"mlb-bet-predictor")`, staging prefix `"mlb-bet-predictor/data_delivery/"` (seen-set keys + `_stage` dest), `git.ls_files("mlb-bet-predictor/data_delivery")`, `sync_dir/"mlb-bet-predictor"/"data_delivery"` |
| `backend/test_cleanup_fix.py` | 62 | Fixture path strings like `"mlb-bet-predictor/data_delivery/run_engine_markets_20260820.csv"` fed to `classify_tracked`; `_is_protected` strips up to `data_delivery/` so semantics survive any prefix, but the fixtures must reflect the real layout |
| `backend/test_frontend_markets.py` | 2 | `REPO_SUBDIR = "mlb-bet-predictor"` mirrors `utils._raw_url` |
| `frontend/utils.py` | 2 | `REPO_SUBDIR = "mlb-bet-predictor"` → raw URL `…/{REPO_SUBDIR}/data_delivery/{file}` and GitHub API contents path (drives `available_dates`) |
| `mlb-bet-predictor/_fetch_roofs.py` | 3 | Docstring CLI usage lines only (paths derived via `ROOT`) |
| `backend/test_market_diagnostics.py` | 1 | Fixture string |
| docs: `README.md`, `PLAN-drift-desync-calibration-ablation.md`, `ai_agent_prompt.txt` | ~4 | Documentation only |
| notebooks `colab_mlb_run.ipynb` / `kaggle_mlb_run.ipynb` | 33 / 46 | Mostly inside `master_pipeline` output; the notebooks themselves only clone `/content/sports_prediction_model` (repo name unchanged — safe) |

### 2.3 Every `data_delivery` consumer (functional)

- **Frontend (raw URL + local fallback):** `frontend/utils.py` is the single
  loader — all pages go through it (`todays_games.py`, `power_rankings.py`,
  `model_calibration.py`, `model_monitor.py`, `markets.py`). `Home.py` renders
  the brand header. Local fallback = `LOCAL_DATA_DIR = parents[1]/data_delivery`.
- **Backend config:** `config.py` (`DATA_DELIVERY_DIR`, `MODELS_DIR`) — derived,
  not hardcoded. ~40 backend modules read artifacts via `config`.
- **Tests:** `test_*.py` read real artifacts via `Path(__file__).parents[1]/"data_delivery"` (works as long as `backend/` and `data_delivery/` stay siblings), plus fixtures in `backend/fixtures/`.
- **Colab push:** `master_pipeline.py` Phase 5/6 (stage + stale cleanup).
- **Docs:** README raw-URL examples, `ai_agent_prompt.txt`.

### 2.4 Path-derivation patterns (the load-bearing invariants)

| Pattern | Location | Survives rename? |
|---|---|---|
| `ROOT_DIR = Path(__file__).parent.parent` then `/backend`, `/data_delivery` | `config.py` | ✅ yes (backend + data_delivery stay siblings under `mlb-backend/`) |
| `Path(__file__).parents[1] / "data_delivery"` | backend tests (`test_drift_real_artifact`, `test_frames_canonical`, …) | ✅ yes (sibling stays) |
| `Path(__file__).parents[1] / "frontend"` | `test_frontend_*.py` | ❌ **breaks at Phase B** (frontend leaves the sport dir) |
| `ROOT_DIR = parents[1] / "data_delivery"` | `frontend/utils.py` | ❌ **breaks at Phase B/C** (must become per-sport via config) |
| `REPO_SUBDIR = "mlb-bet-predictor"` | `frontend/utils.py`, `test_frontend_markets.py` | ❌ **flip at Phase C** |
| `repo_dir / "mlb-bet-predictor" / "backend"` + `os.chdir` + staging prefix | `master_pipeline.py` | ❌ **flip at Phase C** (needs one constant) |
| `/content/sports_prediction_model` clone path | notebooks | ✅ yes (repo name unchanged) |
| `Path.cwd()/"data_delivery"` (relative after chdir) | `master_pipeline.py` Phase 5 | ✅ yes (relative) |

### 2.5 Artifact taxonomy (defines the shared vs per-sport contract)

**Sport-agnostic (shared frontend contract — every sport publishes these):**

| Artifact | Frontend consumer |
|---|---|
| `todays_games_YYYYMMDD.csv` | Today's Games cards (date nav key) |
| `predictions_history_YYYYMMDD.csv` | Calibration table, deep-past date reconstruction |
| `calibration_YYYYMMDD.json` | Calibration page (metrics, buckets, daily) |
| `model_monitor_YYYYMMDD.json` | Model Monitor page |
| `power_rankings_YYYYMMDD.csv` | Power Rankings page |
| `shap_game_<game_id>.csv` | Per-game SHAP panel |
| `game_level_features.csv` | Score enrichment merge (optional) |
| `models/ensemble_latest.joblib` | Backend-internal (dashboard loads it for live scoring) |

**MLB-specific (only `mlb-backend/` publishes; frontend shows them only for MLB):**

| Artifact | Why MLB-only |
|---|---|
| `run_engine_markets_*.csv` + `.meta.json` | Totals & Run Lines machinery |
| `run_engine_oof_*.csv` | Run-line OOF |
| `run_engine_monitor_*.json` | Run-engine calibration/agreement monitor |
| `run_engine_feature_drift_*`, `run_engine_feature_coverage_*` | Env-level feature drift tables |
| `umpire_map.csv`, `umpire_stats.csv` | Umpire features |
| `statsapi_roof_cache.json` | Roof/weather features |
| `batter_woba.parquet`, `team_woba.parquet`, `lineups.parquet` | MLB ingestion inputs (pipeline consumes, never rebuilds) |
| `pbp_chunks/` | Statcast play-by-play cache |
| `year_effect_check_*.json` | MLB-specific diagnostic |

Frontend rendering that is MLB-specific: `MLB_TEAM_NAMES` map in `utils.py`,
`FEATURE_DESCRIPTIONS` (umpire/roof/weather/dome features), pitchers + venue +
run-engine chips on Today's Games cards, and the "⚾ MLB Predictions" brand.
Sport-agnostic rendering: card layout, date nav, calibration curves, PSI tables,
power-ranking table, SHAP bar chart, source note.

---

## 3. STEP 2 — Target design

### 3.1 Target tree (all 4 sports)

```
sports_prediction_model/
├── .gitignore
├── docs/
│   └── multi_sport_restructure_plan.md
├── frontend/                              # SHARED
│   ├── Home.py                            # sport toggle + nav (single source of nav truth)
│   ├── sports_config.py                   # NEW — sport registry (see 3.4)
│   ├── utils.py                           # sport-aware artifact loader
│   ├── todays_games.py  power_rankings.py  model_calibration.py  model_monitor.py
│   ├── markets.py  market_diagnostics.py  # MLB-only pages (gated by sport config)
│   ├── .streamlit/config.toml  streamlit_theme.toml  requirements.txt
├── mlb-backend/                           # git mv of mlb-bet-predictor − frontend/
│   ├── backend/                           # config.py, master_pipeline.py, frames.py, … (95 files)
│   │   └── fixtures/
│   ├── data_delivery/                     # 44 artifacts (unchanged contents)
│   ├── README.md  PLAN-drift-desync-calibration-ablation.md  _fetch_roofs.py  .gitignore
├── nfl-backend/                           # FUTURE (empty scaffold or absent until built)
│   ├── backend/                           # (per-sport pipeline, mirrors MLB contract)
│   └── data_delivery/                     # minimal shared contract (3.3)
├── nba-backend/ … nhl-backend/            # FUTURE
├── colab_mlb_run.ipynb  kaggle_mlb_run.ipynb  ai_agent_prompt.txt
```

`mlb-backend/data_delivery/` keeps its current 44 artifacts byte-identical
(`run_engine_*`, `umpire_*`, `statsapi_roof_cache.json`, `batter_woba.parquet`,
`team_woba.parquet`, `lineups.parquet`, `pbp_chunks/`, `models/`, `*.joblib`,
`todays_games_*`, `predictions_history_*`, `calibration_*`, `model_monitor_*`,
`power_rankings_*`, `shap_game_*`, `game_level_features.csv`,
`model_history.json`, `model_version_history.json`, `feature_drift_*`,
`feature_coverage_*`, `features_metadata_*`, `rolling_brier_*`,
`run_engine_feature_*`, `year_effect_check_*.json`).

### 3.2 Per-sport backend contract (minimal publish set)

Every sport backend MUST publish (same schema the shared frontend renders):

1. `todays_games_YYYYMMDD.csv` — one row per game: `game_id/game_pk`,
   `game_date`, `home_team`, `away_team`, `start_time_utc`, `home_win_prob_model`,
   `model_pick`, `home_score`/`away_score` (nullable pre-game), `game_state`,
   `venue`. (MLB also emits `pitchers_*`, run-engine columns — optional extras.)
2. `predictions_history_YYYYMMDD.csv` — walk-forward per-game predictions + outcomes.
3. `calibration_YYYYMMDD.json` — `metrics{auc,brier,logloss,ece}`, `calibration_buckets[]`,
   `daily[]` (date-stamped walk-forward entries).
4. `model_monitor_YYYYMMDD.json` — PSI/feature-drift + blend weights.
5. `power_rankings_YYYYMMDD.csv` — team, w/l, rating.
6. `shap_game_<game_id>.csv` — `feature`, `shap_value`.
7. `game_level_features.csv` — optional score-enrichment source.
8. `models/ensemble_latest.joblib` — optional (backend-internal scoring).

Plus a per-sport `manifest.json` (NEW, optional in v1): `{"sport": "nfl",
"publish_date": "YYYYMMDD", "has_run_engine": false, "artifacts": [...]}`
so the frontend can gate MLB-only pages without probing for files.

### 3.3 Shared frontend contract

- **Sport-agnostic pages** (render any sport that publishes §3.2): Today's
  Games, Power Rankings, Calibration, Model Monitor. They consume only
  `utils.load_*` functions, which resolve the sport base URL/local dir.
- **MLB-specific pages**: Totals & Run Lines (`markets.py`), plus the
  run-engine chips on Today's Games cards and umpire/roof/weather feature
  descriptions. Shown only when the active sport's config declares
  `has_run_engine: true` (or by default for MLB).
- **Per-sport rendering tables**: `MLB_TEAM_NAMES` → per-sport team map;
  `FEATURE_DESCRIPTIONS` → per-sport feature dictionary (monitor PSI labels).

### 3.4 Sport toggle design

- **Registry** — `frontend/sports_config.py` (single source of truth; a
  checked-in Python module, not env-dependent):

```python
SPORTS = {
    "mlb": {
        "label": "MLB", "emoji": "⚾",
        "repo_subdir": "mlb-backend",          # GitHub raw URL path prefix + local dir
        "has_run_engine": True,
        "pages": ["todays-games", "power-rankings", "calibration", "model-monitor", "markets"],
        "team_names": MLB_TEAM_NAMES, "feature_descriptions": FEATURE_DESCRIPTIONS_MLB,
    },
    "nfl": {
        "label": "NFL", "emoji": "🏈",
        "repo_subdir": "nfl-backend",
        "has_run_engine": False,
        "pages": ["todays-games", "power-rankings", "calibration", "model-monitor"],
        "team_names": NFL_TEAM_NAMES, "feature_descriptions": FEATURE_DESCRIPTIONS_NFL,
    },
    # nba / nhl …
}
DEFAULT_SPORT = os.environ.get("ML_DEFAULT_SPORT", "mlb")
```

- **Selection** — `utils.get_sport_config()` reads `st.session_state["sport"]`
  (default `DEFAULT_SPORT`). `Home.py` renders the toggle
  (`st.segmented_control` or radio) inside `render_brand_header()` — the exact
  spot the header already reserves — **above** the nav; changing it sets
  session state + `st.rerun()`, then the nav page list is rebuilt from
  `SPORTS[sport]["pages"]` (st.navigation is built after the sidebar block, so
  the rebuild is correct).
- **Artifact resolution** — `utils.py` replaces the module constants with
  per-sport resolution:
  - `repo_subdir = sport["repo_subdir"]` → raw URL
    `https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{repo_subdir}/data_delivery/{file}`
    and GitHub API contents path (date discovery).
  - Local fallback `LOCAL_DATA_DIR = REPO_ROOT / repo_subdir / "data_delivery"`.
  - The URL/fallback logic itself is unchanged — only the prefix is dynamic.
- **Pages** — the five existing page files stay single implementations that
  read the active sport via `utils`; the nav list (built per sport) is the only
  per-sport branching. No duplicated page files.

---

## 4. STEP 3 — Migration strategy (phased, zero-breakage)

All acceptance checks use the canonical invocation
`PYTHONPATH=mlb-bet-predictor:mlb-bet-predictor/backend` (Phases A–B) →
`PYTHONPATH=mlb-backend:mlb-backend/backend` (Phases C–E):
`python3 -m unittest discover -s <sport>/backend -p 'test_*.py'` → **exactly the
2 known environmental failures** (missing `weather_history.parquet` cache,
GitHub raw-URL 404), 0 errors.

### Phase A — Config-driven constants (S)

Behavior-neutral. Introduce the path constants so later phases are one-line flips.

- **Files touched:** `backend/config.py` (add `SPORT_DIR_NAME = "mlb-bet-predictor"`
  and/or `REPO_ARTIFACT_PREFIX`), `backend/master_pipeline.py` (replace the 7
  hardcoded strings with the constant), `frontend/utils.py` (`REPO_SUBDIR` read
  from a new `sports_config.py` that still returns `"mlb-bet-predictor"`),
  `backend/test_frontend_markets.py` (mirror the constant).
- **Tests affected:** none behaviorally (values unchanged); test_cleanup_fix
  strings can stay literal for now.
- **Acceptance:** suite green (2 env failures); raw URLs byte-identical;
  `git grep mlb-bet-predictor` shows only test fixtures/docs.

### Phase B — Move frontend to repo root (M)

`git mv mlb-bet-predictor/frontend frontend` (history preserved).

- **Files touched:** the `frontend/` tree itself (`utils.py` local dir becomes
  `REPO_ROOT / sport.repo_subdir / "data_delivery"`; `REPO_SUBDIR` still
  `"mlb-bet-predictor"` until Phase C); **same commit:**
  `backend/test_frontend_home.py`, `backend/test_frontend_markets.py`,
  `backend/test_frontend_todays_games.py` (their `parents[1] / "frontend"`
  resolution breaks the moment frontend leaves the sport dir — repoint to repo
  root). Optional in this phase: the sport-toggle stub in `Home.py`
  (single-sport, defaults to MLB, no visible behavior change).
- **Tests affected:** the 3 frontend test files (updated in-commit);
  everything else resolves `parents[1] / "data_delivery"` (still valid).
- **Acceptance:** `streamlit run frontend/Home.py` from repo root renders;
  frontend still serves `mlb-bet-predictor/data_delivery` (raw + local); suite
  green (2 env failures).

### Phase C — Relocate sport dir + flip artifact prefix (L)

`git mv mlb-bet-predictor mlb-backend` — **atomic with the prefix flip**.

- **Files touched:** `config.py` (ROOT_DIR math auto-resolves — no edit needed),
  `backend/master_pipeline.py` (constant now = `"mlb-backend"`), `frontend/sports_config.py`
  (`"mlb-bet-predictor" → "mlb-backend"`), `backend/test_cleanup_fix.py` (62
  fixture strings), `backend/test_frontend_markets.py` (`REPO_SUBDIR`),
  `_fetch_roofs.py` (docstring/CLI), `README.md` (tree + URLs), `PLAN-drift-desync-calibration-ablation.md`.
- **Tests affected:** all (discovery path changes) — but **no test logic
  changes**: `parents[1] / "data_delivery"` still resolves because
  `backend/` + `data_delivery/` remain siblings. New canonical invocation
  becomes `PYTHONPATH=mlb-backend:mlb-backend/backend`.
- **Acceptance (the critical one):** suite green under the new PYTHONPATH;
  frontend local fallback reads `mlb-backend/data_delivery`; raw URLs served
  from the `mlb-backend` prefix. **Push this commit before any later work** —
  GitHub raw paths and the local tree flip together, so there is no window
  where the deployed app 404s.

### Phase D — Notebooks + docs + whitelist sweep (M)

- **Files touched:** `colab_mlb_run.ipynb`, `kaggle_mlb_run.ipynb` (verify
  clone/`sys.path` cells — expected to be path-constant-driven by then),
  `README.md`, `ai_agent_prompt.txt`, `_fetch_roofs.py`, root `.gitignore`
  (consider folding the sport `.gitignore`), any `git grep` leftovers.
- **Tests affected:** none.
- **Acceptance:** `git grep -n "mlb-bet-predictor"` returns only historical
  notes (or nothing); suite green.

### Phase E — Retire old paths (S)

- **Files touched:** delete now-dead helpers if unreferenced (e.g. legacy
  `github_sync.py` — confirm no caller; README mentions it), remove any shim
  constants, add `mlb-backend/README.md` pointer if desired.
- **Tests affected:** none (legacy helper has no tests).
- **Acceptance:** `git grep "mlb-bet-predictor"` clean; suite green.

### Suggested sequencing

Phase A (S) → Phase B (M) → Phase C (L) → Phase D (M) → Phase E (S).
B and C can be squashed into one PR if preferred, but keep C's
rename+prefix flip as a single commit either way. Future sports are pure
additions (scaffold `nfl-backend/` + registry entry) — no further migration.

---

## 5. STEP 4 — Risk register

| # | Risk | Where it bites | Mitigation |
|---|---|---|---|
| 1 | **Artifact-pinned tests** (the 6,953/6,161-frame class) | `test_frames_*`, `test_run_engine_*`, monitor pins | Renaming paths never touches artifact *contents* — pins stay valid. Only inputs changed them historically; the decided-frame/fold-signature logic in `frames.py`/`training.py`/`pipeline.py`/`run_engine.py` is **not touched**, only path constants. |
| 2 | **Phase-6 cleanup prefix** | `master_pipeline.py` stages `"mlb-bet-predictor/data_delivery/…"` in 3 interacting places (seen-set keys, `_stage` dest, `ls_files`); a missed one resurrects stale artifacts or deletes current ones | Single `SPORT_DIR_NAME` constant (Phase A) used everywhere; `test_cleanup_fix.py` keeps pinning the classification semantics. |
| 3 | **GitHub raw-URL 404s** | Deployed frontend with the old `REPO_SUBDIR` breaks the moment the sport dir renames | Phase C flips fetcher + rename in **one commit**; local fallback is URL-independent and always works. Frontend redeploy timing is the only exposure. |
| 4 | **Colab path assumptions** | `master_pipeline.py` `sys.path.insert(repo_dir/"mlb-bet-predictor"/"backend")`; `os.chdir` | Constant-driven from Phase A; relative `Path.cwd()/"data_delivery"` logic is rename-proof. Notebook clone path `/content/sports_prediction_model` (repo name) unchanged. |
| 5 | **Frontend tests path math** | `test_frontend_home.py` / `_markets.py` / `_todays_games.py` resolve `parents[1]/"frontend"` | Repointed in the **same commit** as the frontend move (Phase B). |
| 6 | **`st.Page` flat-file requirement** | Streamlit needs real page files; duplicating 5 pages × 4 sports = 20 files | Single parameterized pages reading sport from session state; nav list built per sport from `sports_config.py`; toggle set before `nav.run()`. |
| 7 | **Docs drift** | README, PLAN, ai_agent_prompt embed paths/URLs | Phase D sweep; `git grep` gate. |
| 8 | **Legacy `github_sync.py`** | Unclear if still called (README references it); it copies into `tmp/"data_delivery"` | Verify callers in Phase D; retire in E if dead. |
| 9 | **Historical notebooks** | Kaggle/Colab snapshots embed old paths; reruns would 404 after C | Notebooks re-clone fresh each run and are updated in D; old snapshots are historical records. |
| 10 | **`__pycache__`/pytest cache churn** | Root `.gitignore` already covers `__pycache__/`, `*.pyc`, `.pytest_cache/` | Keep the root ignore in place; no action. |

---

## 6. Current suite state

Canonical suite on HEAD (`7395eda`), run 2026-08-28 20:1x:
**739 tests — exactly the 2 known environmental failures**
(`data_delivery/weather_history.parquet` cache missing; GitHub raw-URL 404 on
`run_engine_markets_20260824.csv`), **0 errors, 14 skips** (all legitimate:
network-gated, local artifacts absent). This plan changes no code, so the
suite state is unchanged by it.
