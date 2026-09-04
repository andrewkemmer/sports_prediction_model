"""NFL master pipeline — the phase-driven Kaggle/Colab orchestration.

Mirrors ``mlb-backend/backend/master_pipeline.py`` for the NFL backend:

  Phase 0  environment: pip install NFL deps, fresh clone into the ACTIVE
           working dir (/kaggle/working on Kaggle, /content on Colab) and
           chdir into ``nfl-backend``. A LOCAL checkout (no /kaggle or
           /content dirs) skips the clone and runs against the current repo.
  Phase 1  ingest: ``nfl_game_frame`` decided frame (2019-2025) → the
           canonical CSV + dated snapshot.
  Phase 2  features: ``nfl_features`` build + STATIC served-pool manifest
           (the feature-admission gate was retired 2026-09-02 by the
           NFL↔MLB parity pass) → ``nfl_feature_v1_<date>.json``.
  Phase 3  moneyline: ``nfl_moneyline`` 5-member ensemble walk-forward +
           sealed-2025 gate → ``nfl_moneyline_v1_<date>.json`` with the
           per-game ``games[]`` slate — written ONLY when the sealed window
           earns adoption; otherwise ``predictions: {status: blocked}``.
  Phase 3b run-engine markets emission (MLB ``run_engine_daily`` mirror):
           after the moneyline phase, the shared core
           (``run_nfl_markets_backfill.run_daily_markets``) regenerates the
           decided OOF store (pooled/sealed walk, pinned chain) + today's
           slate rows into ``nfl_run_engine_markets_<date>.csv`` (+ meta /
           monitor / serve record) so the Totals & Run Lines dashboard
           stays current without manual backfills. Any gate/pin breach
           RAISES — the run stops loudly, nothing partial is emitted.
  Phase 4  GitHub sync: stage ONLY this run's ``nfl-backend/data_delivery``
           files (never ``.pyc``, never other sport dirs) and push.
  Phase 5  stale-artifact cleanup: runs strictly after a confirmed push (a
           failed push can never empty the folder). The sweep may ONLY remove
           stale UNTRACKED artifacts (protected names/prefixes, the
           board-backed record rule and a 48h retention window honored).
           Anything TRACKED (committed to git) is never deleted — it is the
           permanent record; a would-be-stale committed file is reported with
           a loud warning and skipped.

CLI:
    python master_pipeline.py                       # full Kaggle run (sync + cleanup)
    python master_pipeline.py --no-push             # dry: phases 0-3 only, no git
    python master_pipeline.py --no-push --features-csv /path/feats.csv
                                                    # dry from a pre-computed frame
    python master_pipeline.py --slate-season 2026   # override the slate target season
    python master_pipeline.py --out-dir /tmp/out    # write records to a custom dir
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Environment / token
# ---------------------------------------------------------------------------
def _load_github_token() -> str:
    """GitHub token from env or Kaggle Secrets, never crashing.

    Order: GITHUB_TOKEN / MY_GITHUB_TOKEN env → Kaggle ``UserSecretsClient``
    (guarded, so a subprocess/Colab run without Kaggle secrets doesn't crash).
    Returns "" when unavailable; GitHub sync is then skipped.
    """
    env_tok = (os.environ.get("GITHUB_TOKEN") or os.environ.get("MY_GITHUB_TOKEN") or "").strip()
    if env_tok:
        print("  [token] GitHub token loaded from environment")
        return env_tok
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("MY_GITHUB_TOKEN").strip()
        print("  [token] GitHub token loaded from Kaggle Secrets")
        return tok
    except Exception:
        print("  [token] No GitHub token (set Kaggle secret 'MY_GITHUB_TOKEN' "
              "or env GITHUB_TOKEN) -> GitHub sync will be skipped")
        return ""


TOKEN = _load_github_token()

CONFIG = {
    "github_username": "andrewkemmer",
    "github_repo":     "sports_prediction_model",
    "github_branch":   "main",
    "github_token":    TOKEN,
    "git_email":       "andrew.kemmer@gmail.com",
    "git_name":        "andrewkemmer",
}

# The repo-relative directory holding this sport's backend + data_delivery
# (mirrored in backend/config.py and frontend/sports_config.py).
SPORT_DIR_NAME = "nfl-backend"

# Default open-ended season bounds, used ONLY to fall back when the user
# supplies exactly one of --start-season / --end-season (see parse_args).
_DEFAULT_FIRST_SEASON = 2019
_DEFAULT_LAST_SEASON = 2025


def _window_label(seasons: list[int] | None) -> str:
    """Human label for the data/feature season window in banners; the
    default full range is printed as the conventional 2019-2025 span."""
    if seasons:
        return f"({seasons[0]}-{seasons[-1]})"
    return "(2019-2025)"


def _active_work_dir() -> Path:
    """The pipeline's working dir: Kaggle → /kaggle/working, Colab → /content,
    otherwise the current checkout (local dry path)."""
    for cand in ("/kaggle/working", "/content"):
        p = Path(cand)
        if p.exists():
            return p
    return Path.cwd()


WORK_DIR = _active_work_dir()
IS_LOCAL = WORK_DIR == Path.cwd() and not Path("/kaggle/working").exists() \
    and not Path("/content").exists()
REPO_DIR = (WORK_DIR / CONFIG["github_repo"] if not IS_LOCAL else
            Path(__file__).resolve().parents[2])
NFL_DIR = REPO_DIR / SPORT_DIR_NAME


def _banner(phase, msg=""):
    print(f"\n{'=' * 70}\n  {phase} - {msg}\n{'=' * 70}\n")


def _run(cmd, check=True, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and r.returncode != 0:
        print(f"  [warn] {cmd}\n      {r.stderr[:300]}")
    return r


def _try_load_schedule(slate_season: int):
    """Load the schedule + pbp (nflreadpy) for the moneyline phase's slate
    stage, then append the target-season's scheduled games from ESPN (nflreadpy
    caps its feed at the latest nflverse season it knows, so 2026 pre-game rows
    come from ESPN). Returns (schedule, pbp) or (None, None) when nflreadpy is
    missing (the --features-csv local dry path) — the slate stage then reports
    'blocked (no schedule loaded)' instead of erroring."""
    try:
        from nfl_features import DEFAULT_SEASONS, _load_raw
        ss = slate_season or datetime.now().year
        seasons = list(DEFAULT_SEASONS)
        print(f"  [slate] loading nflreadpy schedule+pbp for {seasons}")
        sched, pbp = _load_raw(seasons)
    except ImportError:
        print("  [slate] nflreadpy not installed - slate skipped (dry path)")
        return None, None
    except Exception as e:  # noqa: BLE001
        print(f"  [slate] schedule load failed (slate skipped): {e}")
        return None, None

    # The target-season (e.g. 2026) scheduled rows are read live from the
    # native nflverse ``games.csv`` release (GitHub, reachable from Kaggle).
    # nflreadpy can't serve 2026 (its installed validator caps at 2025), so we
    # fetch the same underlying artifact directly. Scheduled rows carry NaN
    # scores, so build_slate_features treats them as the pre-game slate.
    try:
        import pandas as pd
        from nfl_nflverse_schedule import load_nflverse_games
        rows = load_nflverse_games(ss)
        if not rows.empty:
            sched = pd.concat([sched, rows], ignore_index=True, sort=False)
            print(f"  [slate] nflverse games.csv: {len(rows)} scheduled {ss} games appended")
        else:
            print(f"  [slate] nflverse has no scheduled {ss} games yet")
    except Exception as e:  # noqa: BLE001
        print(f"  [slate] nflverse {ss} games.csv load failed: {e}")
    return sched, pbp


# ---------------------------------------------------------------------------
# Phase 0 — environment
# ---------------------------------------------------------------------------
def phase0(args) -> None:
    _banner("PHASE 0", "Environment Setup")
    if IS_LOCAL:
        print(f"  Local checkout detected: {REPO_DIR} (no clone)")
    else:
        print(f"  pip install NFL dependencies...")
        _run('pip install -q nflreadpy scikit-learn lightgbm xgboost '
             'pandas polars numpy joblib gitpython requests', check=False)
        repo_dir = REPO_DIR
        # CRITICAL: never rmtree the checkout we are currently running FROM.
        # The Kaggle notebook clones the repo then launches this script with
        # cwd inside the clone. Deleting that directory deletes the process's
        # own working dir: git clone then fails with "getcwd() failed" and the
        # subsequent os.chdir(NFL_DIR) dies. If cwd already resolves inside
        # REPO_DIR, reuse the existing checkout instead of re-cloning.
        cwd = Path.cwd().resolve()
        inside_checkout = False
        try:
            cwd.relative_to(repo_dir.resolve())
            inside_checkout = True
        except ValueError:
            inside_checkout = False
        if repo_dir.exists() and inside_checkout:
            print(f"  Already running inside repo checkout: {repo_dir} (no re-clone)")
        else:
            if repo_dir.exists():
                print("  Removing old clone...")
                shutil.rmtree(repo_dir, ignore_errors=True)
            print(f"  Cloning {CONFIG['github_repo']}...")
            _run(f"git clone -q https://github.com/{CONFIG['github_username']}/"
                 f"{CONFIG['github_repo']}.git {repo_dir}")

    backend_dir = NFL_DIR / "backend"
    sys.path.insert(0, str(backend_dir))
    os.chdir(str(NFL_DIR))
    print(f"  cwd -> {os.getcwd()}")

    # Drop cached sport modules so a fresh clone's code is used.
    for mod in list(sys.modules.keys()):
        if any(x in mod for x in ("nfl_game_frame", "nfl_features", "nfl_moneyline")):
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Phase 1 — ingest
# ---------------------------------------------------------------------------
def phase1(args) -> None:
    seasons = args.window
    _banner("PHASE 1", f"NFL decided game frame {_window_label(seasons)}")
    if args.features_csv is not None:
        print("  --features-csv set: skipping ingest (using pre-computed frame)")
        return
    from nfl_game_frame import pull_and_build
    summary = pull_and_build(seasons)
    # Market-independence policy: betting-line coverage was removed with the
    # market columns (49b1297), so the summary no longer emits
    # ``line_coverage_pct``. Report only the market-free counts + frame sha.
    print(f"  [ok] decided games: {summary['games']}, sha256 {summary['sha256']}")


# ---------------------------------------------------------------------------
# Phase 2 — features
# ---------------------------------------------------------------------------
def phase2(args) -> None:
    _banner("PHASE 2", "Served-pool manifest (12-pool; admission gate retired)")
    if args.features_csv is not None:
        print("  --features-csv set: skipping feature build")
        return
    from nfl_features import pull_and_build
    pull_and_build(seasons=args.window)


# ---------------------------------------------------------------------------
# Phase 3 — moneyline ensemble + gated slate
# ---------------------------------------------------------------------------
def phase3(args) -> None:
    _banner("PHASE 3", "Moneyline 5-member ensemble + sealed gate + gated slate")
    from nfl_moneyline import pull_and_run

    out_dir = Path(args.out_dir) if args.out_dir else None
    schedule = pbp = None
    if args.features_csv is None:
        schedule, pbp = _try_load_schedule(args.slate_season)
    elif not IS_LOCAL:
        # non-local but features_csv given: still try the slate
        schedule, pbp = _try_load_schedule(args.slate_season)

    pull_and_run(out_dir=out_dir,
                 features_csv=args.features_csv,
                 schedule=schedule,
                 pbp=pbp,
                 slate_season=args.slate_season,
                 seasons=args.window)

    # Phase 3b — run-engine markets emission (the daily Totals & Run Lines
    # store). MLB mirror: the moneyline/training phase ends with
    # run_engine_daily emitting the dated markets artifact (decided rows
    # WITH actuals + today's slate rows in one dated file) that the
    # dashboard reads as the newest dated store. The shared core raises on
    # any pin/gate breach — the run stops loudly (sync/cleanup never run),
    # never a silent or partial emit.
    _phase3b(args)


# ---------------------------------------------------------------------------
# Phase 3b — run-engine markets emission (daily Totals & Run Lines store)
# ---------------------------------------------------------------------------
def _phase3b(args) -> None:
    """Run-engine markets emission — the daily dated store, MLB mirror.

    Mirrors MLB's ``run_engine_daily`` call at the end of its
    moneyline/training phase (pipeline.py: the markets artifact regenerates
    every run, decided rows WITH actuals + today's slate rows in one dated
    file, and the frontend reads the newest one). Calls the SAME shared
    core the standalone backfill CLI calls
    (``run_nfl_markets_backfill.run_daily_markets``) — one schema, no fork.

    Failure policy (judgment call 2 of the wiring spec): the core RAISES on
    any frame-sha / record-pin / gate breach after printing the FATAL
    reason, and this phase lets the exception propagate — the run stops
    loudly, sync/cleanup never run, nothing partial is emitted, and the
    last good committed dated store stays the served store until the
    degradation is resolved. A dry path without nflreadpy (no board/slate
    possible) is the one graceful skip, mirroring Phase 3's slate stage.
    """
    _banner("PHASE 3b",
            "Run-engine markets emission (decided OOF store + today's slate)")
    if importlib.util.find_spec("nflreadpy") is None:
        print("  [phase 3b] nflreadpy not installed - run-engine emission "
              "skipped (dry path, mirror of the slate stage)")
        return
    from run_nfl_markets_backfill import run_daily_markets
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    res = run_daily_markets(out_dir=out_dir)
    print(f"  [ok] emitted {res['date_str']}: {res['n_slate']} slate + "
          f"{res['n_oof']} decided rows")
    for w in res["written"]:
        print(f"    {w}")


# ---------------------------------------------------------------------------
# Phase 4 — GitHub sync (stage ONLY this run's nfl-backend/data_delivery)
# ---------------------------------------------------------------------------
def _git_push_confirmed(repo, branch: str) -> None:
    import git
    info = repo.remote("origin").push(branch)
    bad = [p for p in info if p.flags & (p.ERROR | p.REJECTED | p.REMOTE_REJECTED | p.REMOTE_FAILURE)]
    if bad:
        raise RuntimeError(f"Push rejected by remote: {[p.summary for p in bad]}")


def _open_sync_repo(token: str, sync_dir: Path):
    import git
    auth_url = (f"https://{token}@github.com/{CONFIG['github_username']}/"
                f"{CONFIG['github_repo']}.git")
    if (sync_dir / ".git").exists():
        repo = git.Repo(str(sync_dir))
    else:
        repo = git.Repo.clone_from(auth_url, str(sync_dir),
                                   branch=CONFIG["github_branch"], depth=1)
    if CONFIG["git_email"]:
        repo.config_writer().set_value("user", "email", CONFIG["git_email"]).release()
    if CONFIG["git_name"]:
        repo.config_writer().set_value("user", "name", CONFIG["git_name"]).release()
    return repo


def _file_sha256(path: Path) -> str:
    """Content sha256 of a small artifact (data_delivery files are tiny CSVs/
    JSONs). Used instead of mtime to decide stale-vs-changed, because a fresh
    ``git clone`` stamps every checked-out file with the clone time (~now),
    which is LATER than this run's writes — so mtime baselines silently drop
    re-generated artifacts on same-date re-runs."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_delivery(delivery_dir: Path) -> dict[str, str]:
    """Map repo-relative data_delivery path -> committed content sha256, for the
    sync clone's pre-existing files. Absent from the map = new to the repo."""
    snap: dict[str, str] = {}
    if delivery_dir.exists():
        for p in delivery_dir.rglob("*"):
            if p.is_file():
                snap[p.relative_to(delivery_dir).as_posix()] = _file_sha256(p)
    return snap


def _latest_dated_artifacts(tree_lines: list[str], prefix: str,
                            n: int = 3) -> list[str]:
    """The newest ``n`` dated artifacts from a git tree listing, ascending.

    Filters to ``prefix`` lines with a real YYYYMMDD in the name (via
    _artifact_date); undated or wrong-prefix lines are ignored. Returns []
    when nothing dated matches."""
    dated = sorted(ln for ln in tree_lines
                   if ln.startswith(prefix) and _artifact_date(ln) is not None)
    return dated[-n:] if dated else []


def _post_push_summary(repo, prefix: str) -> dict:
    """HEAD + newest dated artifacts read from the PUSHED clone — the state
    the remote actually has after Phase 4's confirmed push.

    Never read from the working checkout: the pipeline commits/pushes in a
    /tmp sync clone, so the checkout's HEAD and origin/main ref stay at the
    PRE-push commit and report a stale summary (2026-09-01 regression:
    "Repo HEAD after run: 81aea53" while origin had already advanced)."""
    head = repo.head.commit
    return {
        "head": f"{head.hexsha[:7]} {head.summary.strip()}",
        "latest_dated": _latest_dated_artifacts(
            repo.git.ls_tree("-r", "--name-only", "HEAD").splitlines(),
            prefix),
    }


def phase4(args) -> None:
    _banner("PHASE 4", "GitHub sync - push this run's new artifacts")
    if args.no_push:
        print("  --no-push: skipping sync")
        return
    token = TOKEN or CONFIG.get("github_token", "")
    if not token:
        print("  [skip] No token - skipping push and cleanup")
        return

    import shutil
    sync_dir = Path("/tmp") / "nfl_sync_tmp" if not IS_LOCAL \
        else Path(os.environ.get("TMPDIR", "/tmp")) / "nfl_sync_tmp"
    staged: list[str] = []
    seen: set[str] = set()
    try:
        repo = _open_sync_repo(token, sync_dir)
        delivery_dir = sync_dir / SPORT_DIR_NAME / "data_delivery"
        delivery_dir.mkdir(parents=True, exist_ok=True)
        local_delivery = Path.cwd() / "data_delivery"

        # Snapshot the sync clone's committed files by CONTENT hash. A fresh
        # clone stamps every file with the clone time (~now) which is later than
        # this run's writes, so an mtime comparison would silently skip every
        # re-generated artifact on a same-date re-run. Content comparison stages
        # a file when it differs from the committed copy (or is new to the repo).
        committed = _snapshot_delivery(delivery_dir)

        def _stage(src: Path, rel: str) -> None:
            if rel in seen:
                return
            seen.add(rel)
            dest = delivery_dir / rel[len(f"{SPORT_DIR_NAME}/data_delivery/"):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            staged.append(rel)

        for artifact in sorted(local_delivery.rglob("*")) if local_delivery.exists() else []:
            if not artifact.is_file():
                continue
            rel_local = artifact.relative_to(local_delivery).as_posix()
            if artifact.suffix == ".pyc":
                continue  # never stage bytecode
            committed_sha = committed.get(rel_local)
            if committed_sha is not None and _file_sha256(artifact) == committed_sha:
                continue  # byte-identical to the committed copy -> stale/unchanged
            _stage(artifact, f"{SPORT_DIR_NAME}/data_delivery/{rel_local}")

        print(f"  [stage] {len(staged)} files:")
        for s in staged:
            print(f"    {s}")
        if staged:
            # force=True == ``git add -f``: the dated markets CSV is
            # gitignored (nfl-backend/.gitignore: data_delivery/*.csv), and a
            # BRAND-NEW dated .csv (never committed before its first push)
            # would otherwise be silently skipped. Tracked files are
            # unaffected (gitignore never applies to them).
            repo.index.add(staged, force=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            repo.index.commit(f"Update NFL features + predictions: {ts}")
            _git_push_confirmed(repo, CONFIG["github_branch"])
            print(f"  [ok] Pushed {len(staged)} files - confirmed on "
                  f"{CONFIG['github_repo']}@{CONFIG['github_branch']}")
            # Post-push summary from the PUSHED clone (its HEAD is what the
            # remote now has) — NOT from the stale working checkout's
            # origin/main ref (2026-09-01 regression).
            summ = _post_push_summary(repo, f"{SPORT_DIR_NAME}/data_delivery/")
            print(f"  [summary] Repo HEAD after push: {summ['head']}")
            if summ["latest_dated"]:
                print("  [summary] Latest dated artifacts (post-push): "
                      f"{summ['latest_dated']}")
        else:
            print("  [skip] Nothing new to push")
        # stash the sync clone state for phase 5 cleanup
        phase4._delivery_dir = delivery_dir  # type: ignore[attr-defined]
        phase4._staged = set(seen)  # type: ignore[attr-defined]
        phase4._sync_dir = sync_dir  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        print(f"  [error] {e}")


# ---------------------------------------------------------------------------
# Phase 5 — stale artifact cleanup (strictly after a confirmed push)
# ---------------------------------------------------------------------------
# BOARD-BACKED RETENTION (MLB regression lesson): an artifact a navigable
# game-date depends on must never be pruned while that date still renders a
# board. MLB kept todays_games_*/shap_game_* for 3 days but run-engine markets
# only 2, so a still-shown board lost its RUN ENGINE data. NFL run-engine
# families are handled by PREFIX-PROTECTION instead (below — accumulating
# dated history, MLB's run_engine_monitor_ rule); the board-backed rule
# translates to the moneyline record family: a dated
# nfl_moneyline_v1_<d>.json / nfl_feature_v1_<d>.json is kept while <d> is
# still a board date — <d> appears as a distinct game_date in the moneyline
# record(s)' games[]. Mirror the board-backed mechanism (no fixed count).

# Exact-name protection for persistent / dateless assets the date-gate can
# never save. The CANONICAL decided frame is the dateless name everything
# hangs off (the same reason MLB protects model_history.json /
# statsapi_roof_cache.json) — never delete it.
_PROTECTED_DELIVERY_NAMES = {
    "nfl_game_level_features.csv",   # canonical decided frame (regenerated +
                                     # staged every run; a dateless name the
                                     # date-gate can never save)
}
_PROTECTED_DELIVERY_PREFIXES = (
    "models/",          # deployed ensemble bundles
    # Run-engine + research-record families (daily emission wiring,
    # 2026-09-04): the dated markets store + its monitor + the slate-serve
    # record and the PINNED research records (era/market/adoption) a daily
    # run depends on are NEVER swept — not even as stale-untracked copies
    # (the tracked-file guard already protects committed copies; prefix-
    # protection extends the same guarantee to untracked ones, so a sweep
    # can never remove the last good dated store or a research record).
    # History accumulates — growth is expected and matches MLB.
    "nfl_run_engine_markets_", "nfl_run_engine_monitor_",
    "nfl_slate_serve_", "nfl_era_", "nfl_market_",
    "nfl_adoption_decision_",
    # Decision/diagnostic records (cb4036f, 2026-09-05): the drift/coverage
    # v2 + fit-panel parity records are DATELESS (sha-named) — the date-gate
    # can never save them, so like the other nfl_* record families they get
    # targeted prefix protection (never a broad nfl_ prefix — the dated
    # moneyline/feature families legitimately ride the date-gate).
    "nfl_run_engine_diagnostics_", "nfl_markets_fit_panel_parity_",
    # Run-engine drift/coverage families (diagnostics emitters, 2026-09-04)
    "run_engine_feature_drift_", "run_engine_feature_coverage_",
)

# Dated record families whose survival is BOARD-BACKED: kept while their
# filename date still renders a board (see classify_stale). The moneyline /
# feature records AND the MLB-equivalent calibration + per-game history
# artifacts (Part-A siblings) are treated identically — a board date never
# loses the calibration/metrics/curve that renders it. Everything else dated
# (monitor/calibration-like files) stays on the plain retention window.
_BOARD_BACKED_RECORD_PREFIXES = (
    "nfl_moneyline_v1_", "nfl_feature_v1_", "nfl_calibration_",
    "nfl_predictions_history_",
)


# ---------------------------------------------------------------------------
# Pure cleanup predicates (unit-tested — see test_master_pipeline.py)
# ---------------------------------------------------------------------------
def _artifact_date(rel: str) -> str | None:
    """Extract the YYYYMMDD date from an artifact path (e.g.
    ``nfl_moneyline_v1_20260830.json`` → 20260830), or None when dateless."""
    import re
    m = re.search(r"_(\d{8})", rel)
    return m.group(1) if m else None


def _is_protected_name(rel: str) -> bool:
    """True when ``rel`` is a persistent asset cleanup must never touch:
    an exact protected name (canonical decided frame) or a protected prefix
    (deployed models/)."""
    _DD = "data_delivery/"
    idx = rel.find(_DD)
    local = rel[idx + len(_DD):] if idx >= 0 else rel
    basename = local.rsplit("/", 1)[-1]
    return (basename in _PROTECTED_DELIVERY_NAMES
            or any(local.startswith(pfx) for pfx in _PROTECTED_DELIVERY_PREFIXES))


def board_dates_from_records(delivery_dir: Path) -> set[str]:
    """S = the board-date set: distinct game_date (YYYYMMDD) across EVERY
    moneyline record's games[].

    A date that has a board must never lose the moneyline_v1/feature_v1
    record that renders it, so S is the UNION over all records (not just the
    newest — after a blocked run the newest record has no games[] and would
    otherwise drop protection for every still-live board). Records without
    games[] (blocked) contribute nothing.
    """
    import json
    board: set[str] = set()
    if not delivery_dir.exists():
        return board
    for rec in sorted(delivery_dir.glob("nfl_moneyline_v1_*.json")):
        try:
            data = json.loads(rec.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt record never crashes cleanup
            continue
        for g in data.get("games") or []:
            gd = str(g.get("game_date") or "").replace("-", "")[:8]
            if len(gd) == 8 and gd.isdigit():
                board.add(gd)
    return board


def _board_backed_keep(rel: str, board_dates: set[str]) -> bool:
    """True when ``rel`` is a dated moneyline/feature record whose filename
    date still renders a board (a distinct game_date in the moneyline
    games[]). A board date must never lose the record that renders it."""
    art_date = _artifact_date(rel)
    if art_date is None or art_date not in board_dates:
        return False
    basename = rel.rsplit("/", 1)[-1]
    return any(basename.startswith(p) for p in _BOARD_BACKED_RECORD_PREFIXES)


def classify_stale(rel: str, staged: set[str], board_dates: set[str],
                   retention_dates: set[str]) -> str:
    """One artifact's cleanup verdict: staged | protected | keep | stale.

    Order (mirrors MLB Phase 6, board-backed):
      1. staged   — this run regenerated it; always wins.
      2. protected— exact-name / prefix-protected (canonical frame, models/).
      3. keep     — board-backed: a moneyline/feature record whose filename
                    date still renders a board (in ``board_dates``) survives
                    regardless of the retention window.
      4. keep     — within the plain retention window (today + prior day).
      5. stale    — everything else, INCLUDING records for dates with no
                    board (the reverse direction is explicitly allowed).
    """
    if rel in staged:
        return "staged"
    if _is_protected_name(rel):
        return "protected"
    if _board_backed_keep(rel, board_dates):
        return "keep"  # board-backed: a board date never loses its record
    art_date = _artifact_date(rel)
    if art_date is None:
        return "stale"  # dateless non-protected -> the date-gate can never save it
    if art_date in retention_dates:
        return "keep"
    return "stale"


def _prune_stale(delivery_dir: Path, staged: set[str], board_dates: set[str],
                 retention: set[str], tracked: set[str],
                 sport_prefix: str = SPORT_DIR_NAME) -> dict:
    """Stale-artifact sweep with the committed-file guarantee.

    NEVER deletes a file listed in ``tracked`` (committed to git): those are
    the permanent record. A tracked file whose classification would be
    ``stale`` lands in ``stale_tracked`` (the caller must print the LOUD
    warning) and is skipped; tracked protected/keep files are counted as
    kept, unchanged. ONLY stale UNTRACKED files are unlinked (plain removal —
    nothing to commit/push, since git never saw them).

    Returns ``{"deleted": [...], "stale_tracked": [...], "kept_protected": n,
    "kept_board": n, "kept_retention": n}`` — repo-relative paths for
    ``deleted`` / ``stale_tracked`` (the deleted list is the record of what
    actually got cleaned). Also honors the existing protections: staged files
    never lose, and every non-stale verdict is kept."""
    deleted: list[str] = []
    stale_tracked: list[str] = []
    kept_protected = kept_board = kept_retention = 0
    if not delivery_dir.exists():
        return {"deleted": deleted, "stale_tracked": stale_tracked,
                "kept_protected": 0, "kept_board": 0, "kept_retention": 0}
    for p in sorted(delivery_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(delivery_dir).as_posix()
        full = f"{sport_prefix}/data_delivery/{rel}"
        if full in staged:
            continue  # this run staged it -> never stale
        verdict = classify_stale(full, staged, board_dates, retention)
        if full in tracked:
            # Committed file: NEVER deleted, whatever the verdict.
            if verdict == "stale":
                stale_tracked.append(full)
            elif verdict == "protected":
                kept_protected += 1
            elif verdict == "keep":
                if _board_backed_keep(full, board_dates):
                    kept_board += 1
                else:
                    kept_retention += 1
            continue
        if verdict == "stale":
            p.unlink()
            deleted.append(full)
        elif verdict == "protected":
            kept_protected += 1
        elif verdict == "keep":
            if _board_backed_keep(full, board_dates):
                kept_board += 1
            else:
                kept_retention += 1
    return {"deleted": deleted, "stale_tracked": stale_tracked,
            "kept_protected": kept_protected, "kept_board": kept_board,
            "kept_retention": kept_retention}


def phase5(args) -> None:
    _banner("PHASE 5", "Stale artifact cleanup (final step)")
    if args.no_push:
        print("  --no-push: skipping cleanup")
        return
    token = TOKEN or CONFIG.get("github_token", "")
    sync_dir = getattr(phase4, "_sync_dir", None)
    delivery_dir = getattr(phase4, "_delivery_dir", None)
    staged = getattr(phase4, "_staged", set())
    if not token or sync_dir is None:
        print("  [skip] no token or nothing pushed - skipping cleanup")
        return

    from datetime import date, timedelta
    import git

    retention = {(date.today() - timedelta(days=i)).strftime("%Y%m%d")
                 for i in range(2)}  # today + previous GMT day

    try:
        repo = git.Repo(str(sync_dir))
        # Board dates from the SYNC CLONE's moneyline records (the pushed
        # state — this run's just-pushed record included).
        board_dates = board_dates_from_records(delivery_dir)
        # COMMITTED-FILE GUARD (2026-09-01 regression): the sweep once gc'd
        # COMMITTED evidence records via git rm + push (the win_pct_diff KEEP
        # verdict, 88d6c8f). ``tracked`` = every committed data_delivery
        # path; _prune_stale skips (loud warning) anything in it and deletes
        # only stale UNTRACKED files — no git rm, no deletion commits.
        tracked = set(repo.git.ls_files(
            f"{SPORT_DIR_NAME}/data_delivery").splitlines())
        out = _prune_stale(delivery_dir, set(staged), board_dates, retention,
                           tracked)
        if out["stale_tracked"]:
            print(f"  [WARN] {len(out['stale_tracked'])} would-be-stale file(s) "
                  "are TRACKED (committed) — kept, NEVER deleted:")
            for s in out["stale_tracked"]:
                print(f"    {s}")
        if out["kept_protected"]:
            print(f"  [protected] kept {out['kept_protected']} protected file(s)")
        if out["kept_board"]:
            print(f"  [board] kept {out['kept_board']} record(s) for "
                  "still-live board dates")
        if out["kept_retention"]:
            print(f"  [retention] kept {out['kept_retention']} same-day "
                  "artifact(s)")
        if not out["deleted"]:
            print("  [ok] No stale untracked files - committed artifacts are "
                  "never auto-deleted")
        else:
            print(f"  [clean] removed {len(out['deleted'])} stale UNTRACKED "
                  "file(s) (not in git - no push needed):")
            for s in out["deleted"]:
                print(f"    {s}")
    except Exception as e:  # noqa: BLE001
        print(f"  [error] Cleanup failed: {e}")

    shutil.rmtree(sync_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="NFL master pipeline: ingest -> features -> moneyline "
                    "ensemble + gated slate -> GitHub sync + cleanup.")
    ap.add_argument("--no-push", action="store_true",
                    help="dry run: phases 0-3 only, never touch git")
    env_slate = os.environ.get("NFL_SLATE_SEASON", "").strip()
    ap.add_argument("--features-csv", type=Path, default=None,
                    help="pre-computed features CSV (skips ingest/features; "
                         "the moneyline dry path)")
    env_start = os.environ.get("NFL_START_SEASON", "").strip()
    env_end = os.environ.get("NFL_END_SEASON", "").strip()
    ap.add_argument("--start-season", type=int,
                    default=(int(env_start) if env_start.isdigit() else None),
                    help="first season in the data/feature window (default: "
                         "each module's full range, e.g. 2019); env "
                         "NFL_START_SEASON also works")
    ap.add_argument("--end-season", type=int,
                    default=(int(env_end) if env_end.isdigit() else None),
                    help="last season in the data/feature window (default: "
                         "each module's full range, e.g. 2025); env "
                         "NFL_END_SEASON also works")
    ap.add_argument("--slate-season", type=int,
                    default=(int(env_slate) if env_slate.isdigit() else None),
                    help="override the slate target season (default: the "
                         "current calendar year, e.g. 2026 week 1; env "
                         "NFL_SLATE_SEASON also works)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="write records to a custom dir (default: "
                         "nfl-backend/data_delivery)")
    ns = ap.parse_args(argv if argv is not None else sys.argv[1:])
    # Resolve the requested data/feature window to a contiguous season list.
    # None for both = each module keeps its own full range (default behavior).
    if ns.start_season is not None or ns.end_season is not None:
        start = ns.start_season if ns.start_season is not None else _DEFAULT_FIRST_SEASON
        end = ns.end_season if ns.end_season is not None else _DEFAULT_LAST_SEASON
        if end < start:
            raise ValueError(f"--end-season {end} < --start-season {start}")
        ns.window = list(range(start, end + 1))
    else:
        ns.window = None
    return ns


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Resolve paths BEFORE phase 0 chdirs into nfl-backend.
    if args.features_csv is not None:
        args.features_csv = Path(args.features_csv).resolve()
    if args.out_dir is not None:
        args.out_dir = str(Path(args.out_dir).resolve())
    phase0(args)
    phase1(args)
    phase2(args)
    phase3(args)
    if not args.no_push:
        phase4(args)
        phase5(args)
    _banner("DONE")
    print(f"  cwd: {os.getcwd()} | no_push: {args.no_push} | "
          f"features_csv: {args.features_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
