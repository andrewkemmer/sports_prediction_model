# MLB Bet Predictor

Production-grade MLB prediction and betting-model application: a **Google Colab
backend** that trains and scores a walk-forward ensemble, and a **Streamlit
frontend** that renders four dashboards straight from GitHub raw URLs.

```
mlb-bet-predictor
├── backend            # Colab-ready Python pipeline (training, features, explainability)
├── frontend           # Streamlit multi-page app (Today's Games / Power Rankings / Calibration / Model Monitor)
├── data_delivery      # canonical artifact sink — produced by Colab, read by the app
└── README.md
```

## Highlights

* **Strict point-in-time (PIT) features.** Every rolling aggregate (wOBA 30g,
  bullpen WHIP 10g, SP ERA/K9, Elo, records, rest days) is computed only from
  games scheduled *strictly before* the target game's start. Market lines are
  attached via an as-of join that rejects lines posted at/after first pitch.
  Unit tests prove it (`backend/test_point_in_time.py`).
* **Expanding-window walk-forward training** with a weekly retrain cadence —
  train folds never contain future outcomes; validation windows slide forward
  (`backend/test_walk_forward.py`).
* **Multi-target heads**: Moneyline (P home win, XGBoost + LightGBM + Logistic
  Regression averaged), Totals (projected total runs), Run Line (cover
  probability + expected margin). Persisted to
  `data_delivery/models/ensemble_latest.joblib`.
* **Explainability**: per-game SHAP CSVs (`data_delivery/shap_game_<id>.csv`)
  and feature-drift PSI tables (`data_delivery/feature_drift_YYYYMMDD.csv`).
* **No heavy ML in the frontend** — the Streamlit app uses only pandas,
  requests, altair, and streamlit, so it can be deployed anywhere.
* **Sports scaffolding**: MLB is the primary implementation; NBA, NHL, NFL,
  College Football, Men's College Basketball, and Tennis hooks live in
  `backend/config.py::SUPPORTED_SPORTS` and the same pipeline pattern applies.

## Repository layout

| Path | Purpose |
| --- | --- |
| `backend/config.py` | Paths, hyperparameters, PSI thresholds, tracking strings, seeds, version metadata keys |
| `backend/data_ingestion.py` | pybaseball + synthetic ingestion, point-in-time features, Elo, market-line attachment, daily artifact builders |
| `backend/training.py` | Walk-forward splits, moneyline/totals/run-line heads, metrics (AUC/Brier/LogLoss/ECE), ensemble persistence |
| `backend/explainability.py` | Per-game SHAP, PSI computation, feature drift |
| `backend/github_sync.py` | GitPython clone → copy artifacts → commit → push (SSH key or PAT) |
| `backend/pipeline.py` | `run_daily_pipeline(target_date)` orchestration + CLI |
| `backend/requirements.txt` | Backend dependencies |
| `backend/test_*.py` | Unit tests: point-in-time, walk-forward, PSI |
| `frontend/Home.py` | Entry point + four-page navigation |
| `frontend/utils.py` | GitHub raw-URL artifact loader (local fallback), formatters, Altair theme |
| `frontend/*.py` | The four dashboard pages |
| `frontend/streamlit_theme.toml` | Dark theme definition (mirrored to `.streamlit/config.toml`) |
| `data_delivery/` | Daily artifacts (CSVs/JSON/SHAP) — the canonical sink |

## 1. Colab runbook (backend)

Open `backend/pipeline.py` (or a fresh notebook) in
[Google Colab](https://colab.research.google.com) and run:

```python
# 1) Install dependencies
!pip install -r https://raw.githubusercontent.com/<owner>/<repo>/main/backend/requirements.txt

# 2) Mount the repo (or clone it)
!git clone https://github.com/<owner>/<repo>.git
%cd <repo>

# 3) Run the daily pipeline (synthetic demo data by default)
from datetime import date
from backend.pipeline import run_daily_pipeline

summary = run_daily_pipeline(date(2026, 8, 9), skip_sync=True)
print(summary["status"])          # 'ok'
print(summary["artifacts"])       # files written to data_delivery/
```

CLI equivalent (from `backend/`):

```bash
python pipeline.py --date 2026-08-09 --skip-sync        # synthetic demo
python pipeline.py --date 2026-08-09 --real             # pybaseball (needs network)
python pipeline.py --date 2026-08-09 --force-retrain    # retrain regardless of cadence
```

What it does:

1. Ingests a point-in-time game log — **synthetic** (default; deterministic,
   seeded, zero network) or **real** (`--real`; pybaseball schedule + Statcast
   metadata + season-to-date pitcher/batter aggregates as-of each prior day).
2. Runs expanding-window walk-forward evaluation and retrains the ensemble
   when the weekly cadence is due.
3. Predicts P(home win) for every game on the target date using only pre-game
   data, writes `todays_games_YYYYMMDD.csv` (+ SHAP per game), power rankings,
   calibration JSON, feature-drift CSV, and the model-monitor JSON.
4. Pushes everything to GitHub `data_delivery/` (unless `--skip-sync`).

> **Real-data note.** The pybaseball path is a production scaffold: final
> scores come from `schedule_and_record`, home/away + SPs from Statcast, and
> SP ERA/K9 and team wOBA are meant to be joined from
> `pitching_stats_range` / `batting_stats_range` as-of `game_date - 1`
> (a `logging.warning` in `load_real_game_events` reminds you). Because live
> sports APIs are rate-limited and flaky in Colab, the pipeline defaults to
> synthetic mode so it always runs end-to-end.

### GitHub credentials (for `github_sync`)

No secrets are hardcoded anywhere. Choose one option and export it in Colab:

**Option A — SSH key (recommended)**

```python
!ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N '' -C colab
!cat /root/.ssh/id_ed25519.pub     # add this to GitHub → Settings → SSH and GPG keys
import os
os.environ["GITHUB_REPO_URL"] = "git@github.com:<owner>/<repo>.git"
```

**Option B — Personal Access Token (PAT)**

```python
import os
os.environ["GITHUB_TOKEN"] = "ghp_..."            # repo scope
os.environ["GITHUB_REPO_URL"] = "https://github.com/<owner>/<repo>.git"
```

The PAT is injected into the remote URL only for the push and is never written
to disk. `sync_artifacts` returns a status dict (`pushed`, `commit_sha`,
`staged_files`, `error`) instead of raising, so a failed push never kills the
daily run.

## 2. Streamlit deployment (frontend)

```bash
pip install streamlit pandas altair requests
streamlit run frontend/Home.py          # from the repository root
```

The sidebar asks for `GitHub owner` / `GitHub repo` (defaults: `GITHUB_OWNER`,
`GITHUB_REPO`, `GITHUB_BRANCH` env vars). Every artifact is then fetched from

```
https://raw.githubusercontent.com/<owner>/<repo>/<branch>/data_delivery/<file>
```

If no repo is configured (or GitHub is unreachable) the app falls back to the
**bundled sample day** in `data_delivery/` (August 9, 2026), so it always
renders. Free hosting options: `streamlit run` on Streamlit Community Cloud
(select `frontend/Home.py` as the entry point) or any VM/container.

Pages:

| Page | Content |
| --- | --- |
| **Today's Games** | Date ribbon + filter pills (All/Final/Live), accuracy badge, two-column cards (badges, scoreboard, probability bars, pitchers, ML/edge, collapsible SHAP bars, outcome banner) |
| **Power Rankings** | Top-15 table: RANK · TEAM (color accent) · ELO · W-L · PCT · RUN DIFF · L10 · HOME% · AWAY% |
| **Calibration** | KPI cards (AUC/Brier/LogLoss/ECE), calibration curve vs perfect diagonal, confidence/accuracy combo chart, reliability table with color-coded GAP |
| **Model Monitor** | Retrain/drift alert boxes, upset note, PSI drift matrix with status pills, rolling 30-day Brier timeline, version history |

## 3. Point-in-time & walk-forward (why you can trust the numbers)

**Point-in-time.** A model trained with future data looks great in backtests
and loses money live. Every feature in this repo is a function of the game
log *strictly before* each game's scheduled start:

* Rolling stats use `shift(1)` (the current game is excluded) over a team's
  chronological history — `data_ingestion.rolling_prior_mean`.
* Elo entering a game is the rating produced by completed prior games only.
* Market lines attach through an as-of merge whose search key is shifted 1
  second earlier, so a line posted at first pitch is rejected.
* `filter_prior(games, as_of)` is the single enforcement point — unit tests
  assert that adding an extreme *future* game changes nothing about earlier
  features.

**Walk-forward.** `training.walk_forward_splits` slides a validation window
forward week by week; each training fold is the *entire expanding* history
strictly before that window. Metrics (AUC, Brier, LogLoss, ECE, calibration
buckets) are pooled across validation folds — never from data the model saw.

### The three probability quantities (read this before charting)

`predictions_history_<date>.csv` and the ensemble expose **three distinct
probabilities**. Every consumer must know which one it holds:

| # | Quantity | Where | Use for |
|---|---|---|---|
| 1 | **Raw blend** | `home_win_prob_model` column | Internal: input to the deployed map; the axis the reliability diagram bins on |
| 2 | **Prequential calibrated** | `home_win_prob_model_calibrated` column | Honest scoring ONLY (each game scored by the Platt map fitted on prior folds). Metrics — never display |
| 3 | **Deployed / user-facing** | σ(a·logit(p_raw)+b) with the global map from `calibration_<date>.json → params` (fit on ALL OOF games) | Display everywhere: Today's Games win %, Prediction History MODEL PICK %, rolling Brier |

**Never mix (2) and (3) in the same chart or comparison.** They are different
maps fitted on different data (up to ~0.11 apart per game): (2) is honest for
scoring but was never deployed; (3) is what users see but is mildly optimistic
on recent OOF games because its map saw them during fitting.

## 4. Tests

```bash
cd backend
python -m unittest discover -s . -p "test_*.py" -v    # or: pytest backend
```

* `test_point_in_time.py` — strict filtering, no future leakage in rolling
  features/Elo, market lines timestamped at/after start are rejected.
* `test_walk_forward.py` — train folds strictly historical, expanding window,
  non-overlapping validation, removing future games leaves folds unchanged.
* `test_psi.py` — identical distributions ≈ 0, shifted distributions exceed
  WARN, degenerate inputs return 0, PSI is never negative, status mapping.

Tests need only `pandas`/`numpy` (heavy ML is imported lazily).

## 5. Assumptions & decisions (documented per the brief)

* **Synthetic-by-default.** The pipeline ships a seeded, deterministic
  synthetic game log so it runs end-to-end in Colab with no API access. Real
  pybaseball ingestion is implemented (`--real`) and documented as a scaffold
  to be pointed at your data provider of choice.
* **Edge definition.** `edge = model_prob − fair_market_prob` (vig removed
  via two-way normalization); displayed per pick. Sample values in
  `data_delivery/` are the curated reference day (Aug 9, 2026) and are
  illustrative — live runs recompute everything.
* **Day-granularity for real data.** Season-to-date pitcher/batter stats are
  queried as-of `game_date − 1` (strictly prior, never same-day) — this is
  PIT-safe and avoids intraday Statcast churn.
* **Coin flips** (|P − 0.5| < 0.02) carry no PICK badge and are graded as
  correct when the winner matches either side; they are excluded from upset
  flags.
* **`league_total` / `evening_games_league`** in the calibration JSON are
  league-wide metadata for the header ("8 of 15 games shown · 7 evening
  games begin 7 PM ET+"); per-game day/night tags come from each game's own
  start time.
* **SHAP ensemble attribution** averages TreeExplainer values across the
  XGBoost/LightGBM members (log-odds space); if `shap` is unavailable the
  pipeline writes a zero-attribution CSV rather than failing.
* **Retrain cadence** is weekly (`RETRAIN_CADENCE_DAYS = 7`) and configurable;
  `--max-eval-folds N` caps walk-forward evaluation to the most recent N
  weekly windows for quick Colab iterations (default: full history).
* **Model versioning** follows `VERSION / TRAINED_AT / DATA_CUTOFF` metadata
  keys (see `config.py`) and a `model_history.json` that feeds the Model
  Monitor version table.
* **Other sports** reuse the same `game-events → PIT features → walk-forward
  → artifacts` skeleton; league-specific adapters slot into
  `data_ingestion.load_game_events`.
* **StatsAPI schedule truncation (do not remove chunking).** The schedule
  endpoint SILENTLY truncates long date ranges — one request for
  2025-01-01→2026-08-23 returns only 2025-02-20→2025-11-01, with no error.
  `results.fetch_game_start_times` therefore always queries in ≤
  `SCHEDULE_CHUNK_DAYS` (60-day) chunks; any new code path that hits the
  schedule endpoint must do the same or an entire season of games silently
  loses start times (and downstream weather) while logs look healthy.
  Coverage gates in `_attach_weather_history` (per calendar year),
  `weather.fetch_games_weather`, and `ingestion._chunked_statcast` warn
  loudly when a source starves; the dashboard's Model & Data Drift Monitor
  page renders a per-feature coverage panel (feature × window × % measured)
  so absence is visible instead of hiding behind default-filled zeros.

## 6. FAQ

**Why doesn't the app import scikit-learn / xgboost / shap?**
All model code lives in `backend/`; the frontend only reads CSV/JSON artifacts
over HTTP. This keeps Streamlit deployments tiny and dependency-free.

**Where do artifacts go?**
`data_delivery/` is the canonical sink. Colab writes there, `github_sync.py`
stages/commits/pushes it, and the app reads it from raw.githubusercontent.com.

**I pushed from Colab but the app shows old data.**
Check the sidebar owner/repo/branch and that the date selector points at the
newest artifact date; raw URLs are cached 5 minutes (`st.cache_data`).

**Can I add a league?**
Add a `SUPPORTED_SPORTS` entry and a `load_<league>_events()` adapter that
normalizes to the game-events schema — everything downstream (features,
walk-forward, artifacts, dashboard) is sport-agnostic.
