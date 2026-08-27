# Implementation Plan: Drift Desync Fix + Conditional Calibration Ablation

**Date:** 2026-08-27
**Status:** Plan only — no code changes beyond the verified partial `training.py` addition.

---

## Scope Note

All file paths and line numbers below are to `mlb-bet-predictor/backend/` unless otherwise stated.

---

## 1. The 4472-vs-4466 Root Cause

### The two row sets

| Path | Frame | Row count (2026-08-26) | Split source |
|------|-------|----------------------|--------------|
| **Training** (`training.py:1459`) | `games` (full frame) | 4466 | `walk_forward_splits(games, ...)` |
| **Drift** (`pipeline.py:1644`) | `decided = games[games["home_win"].notna()]` | 4472 | `walk_forward_splits(decided, ...)` inside `_attach_drift_run_margins` |

### Why they differ

`run_engine_daily` (pipeline.py:1604) operates on the full `games` DataFrame. The run engine's "decided" set is `games[games["home_win"].notna()]` (training.py:1377 inside `_attach_oof_run_margins`), which is the **same filter** as the drift path. However, the training path's `walk_forward_evaluate` builds splits on the **full** `games` DataFrame (4466 rows — before any `home_win.notna()` filter), while `_attach_drift_run_margins` builds splits on the **decided-only** `decided` frame (4472 rows).

The 6-game gap (4472 − 4466 = 6) arises because the `games` frame can include 6 pre-game slate rows (undecided games with `home_win = NaN`) that exist in `game_level_features.csv` but are excluded by the `notna()` filter. When the training path builds splits on the full frame, those 6 rows land in the train/val geometry; when the drift path builds splits on decided-only, the fold boundaries shift because the chronological ordering and week cadence differ by those 6 rows.

The `_attach_oof_run_margins` desync guard (training.py:1406) catches this: it calls `_regenerate_splits` on the enriched frame and asserts the resulting splits match the input splits. On the drift path, the input splits came from `decided` (4472 rows), but the regenerated splits may differ because the enrichment changes the frame. When they don't match, the guard fires:

```
ValueError: enriched-frame folds desynced from margin-build folds
(walk_forward_splits changed?) -- refusing to train on a misaligned split
```

### The real problem

The drift path should NOT build its own splits. It should reuse the **exact** splits the moneyline training already built. The training path calls `walk_forward_splits(games, ...)` at training.py:1459, producing folds over the full 4466-row frame. Those folds are the canonical geometry — the margin column was attached on exactly those folds. The drift path must use the same folds, not rebuild them on a different row set.

---

## 2. The Fix: Wire `_LAST_WALK_FORWARD_SPLITS`

### Current state (already committed, verified backward-compatible)

`training.py:82-93` adds:
```python
_LAST_WALK_FORWARD_SPLITS: list = []

def set_last_walk_forward_splits(splits: list) -> None:
    global _LAST_WALK_FORWARD_SPLITS
    _LAST_WALK_FORWARD_SPLITS = list(splits)

def get_last_walk_forward_splits() -> list:
    return list(_LAST_WALK_FORWARD_SPLITS)
```

This is **purely additive** — never called, no behavior change. The test suite passes (591 passed, 2 known environmental failures, 15 skipped).

### Changes needed (NOT YET IMPLEMENTED)

#### A. Wire `set_last_walk_forward_splits` in the training path

**File:** `training.py`
**Location:** Inside `walk_forward_evaluate`, right after `splits` is finalized (after line ~1470, after the `_attach_oof_run_margins` call returns).

```python
# Record the splits so the drift step can reuse them (avoids building
# splits on a different row set, which desyncs fold geometry).
set_last_walk_forward_splits(splits)
```

Insert this after the `_attach_oof_run_margins` call and before any model training begins.

#### B. Wire `get_last_walk_forward_splits` in `_attach_drift_run_margins`

**File:** `pipeline.py`
**Location:** Inside `_attach_drift_run_margins` (pipeline.py:160-195), replacing the current `walk_forward_splits` call.

Current code (pipeline.py:184-188):
```python
_splits = walk_forward_splits(
    decided, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
if not _splits:
    return decided
enriched, _ = _attach_oof_run_margins(
    decided, _splits, MIN_VAL_FOLD_GAMES, 0,
    RETRAIN_CADENCE_DAYS, 0)
```

Replace with:
```python
from training import get_last_walk_forward_splits
_splits = get_last_walk_forward_splits()
if not _splits:
    # Fallback: no training pass recorded yet — build splits on the
    # decided frame. This path is only reached when drift runs before
    # training in the same process, which doesn't happen in the
    # production pipeline. Preserve the old behavior as a safety net.
    _splits = walk_forward_splits(
        decided, retrain_cadence_days=RETRAIN_CADENCE_DAYS)
    if not _splits:
        return decided
enriched, _ = _attach_oof_run_margins(
    decided, _splits, MIN_VAL_FOLD_GAMES, 0,
    RETRAIN_CADENCE_DAYS, 0)
```

**Import needed** (pipeline.py top-level imports): add `get_last_walk_forward_splits` to the existing `from training import ...` block (pipeline.py:68 area).

#### C. Update the docstring

Update `_attach_drift_run_margins` docstring to state: "Reuses the exact splits from the most recent training pass (via `get_last_walk_forward_splits`) so fold geometry is guaranteed identical to the moneyline's own margin build."

---

## 3. Tests for the Drift Desync Fix

### Test A: Drift attach reuses training splits

**File:** `test_build_oof_margin.py` (or new `test_drift_desync.py`)

```python
def test_drift_attach_uses_last_training_splits():
    """_attach_drift_run_margins must use the splits recorded by the
    most recent walk_forward_evaluate, not build its own."""
    from training import set_last_walk_forward_splits, get_last_walk_forward_splits

    # Simulate a training pass that recorded splits
    fake_splits = [{"fold_idx": 0, "val_start": "2025-06-01",
                    "val_end": "2025-06-07", "val_games": pd.DataFrame({"game_pk": [1,2]})}]
    set_last_walk_forward_splits(fake_splits)
    assert get_last_walk_forward_splits() == fake_splits

    # After clearing, fallback should trigger
    set_last_walk_forward_splits([])
    assert get_last_walk_forward_splits() == []
```

### Test B: Desync no longer fires on the real frame

**File:** `test_build_oof_margin.py`

```python
def test_drift_attach_no_desync_on_real_frame():
    """When the drift frame differs from the training frame, the desync
    guard must NOT fire because drift reuses training's splits."""
    # Load the real game_level_features.csv
    # Mock walk_forward_evaluate to record specific splits
    # Call _attach_drift_run_margins on a decided subset
    # Assert: no AssertionError, run_margin_diff is populated
```

### Test C: Margin row present with finite PSI

**File:** `test_build_oof_margin.py` or `test_psi.py`

```python
def test_drift_table_carries_run_margin_diff_with_finite_psi():
    """After the fix, run_margin_diff appears in the drift PSI table
    with a finite value (not all-NaN)."""
```

### Test D: Guard still fires on genuinely misaligned splits

```python
def test_desync_guard_still_fires_on_genuinely_different_splits():
    """If someone passes completely wrong splits, the desync guard
    must still catch it."""
```

### Test E: Empty fallback path works

```python
def test_drift_attach_empty_splits_fallback():
    """When no training pass has been recorded, _attach_drift_run_margins
    falls back to building splits on the decided frame."""
```

---

## 4. Conditional Calibration Ablation

### Scope

This is a **standalone script** (like `run_lineup_ablation.py`, `run_categorical_ablation.py`). It does NOT modify pipeline.py, training.py, or calibration.py. It replays over saved OOF predictions from the most recent run.

### File: `run_calibration_ablation.py`

**Location:** `mlb-bet-predictor/backend/run_calibration_ablation.py`

### Design

1. **Input**: Load the committed `game_level_features.csv` + `data_delivery/models/ensemble_latest.joblib` (or re-run `walk_forward_evaluate` in a dry mode to produce OOF predictions).

2. **Sealed holdout**: 284 games from 2026-08-05 to 2026-08-25 (the most recent 21 days of decided games).

3. **Candidates** (size-gated):
   - `< 300 OOF games`: identity only
   - `300-1000 OOF games`: identity vs Platt
   - `>= 1000 OOF games`: identity vs Platt vs isotonic regression

4. **Leakage-free selection**: At each fold, fit candidates on the prior-OOF pool, evaluate on the last ~300 of that pool, apply the winner to the unseen fold. Never evaluate the fold being scored to decide.

5. **Score all variants on the sealed holdout**:
   - Current unconditional Platt (the shipped calibrator)
   - Conditional calibrator (the winner per-fold)
   - Each individual map alone (identity, Platt, isotonic)
   - Report: logloss, AUC, ECE per variant

6. **Gate**: ADOPT conditional only if it beats unconditional Platt on holdout ECE WITHOUT degrading logloss. Otherwise DON'T ADOPT.

### Output: `data_delivery/calibration_ablation_<sha>.json`

```json
{
  "schema": "calibration-ablation/v1",
  "commit_sha": "...",
  "data_sha256": "...",
  "holdout_start": "2026-08-05",
  "holdout_end": "2026-08-25",
  "holdout_n": 284,
  "tuning_n": 4182,
  "variants": {
    "identity": {"logloss": ..., "auc": ..., "ece": ...},
    "unconditional_platt": {"logloss": ..., "auc": ..., "ece": ...},
    "conditional": {"logloss": ..., "auc": ..., "ece": ...},
    "isotonic": {"logloss": ..., "auc": ..., "ece": ...}
  },
  "gate": {
    "verdict": "ADOPT" or "DON'T ADOPT",
    "reason": "...",
    "holdout_comparison": {...}
  }
}
```

### Key implementation details

- Isotonic: use `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` — clip to training min/max.
- Platt: reuse existing `calibration.fit_platt` / `calibration.apply_platt`.
- Identity: `calibration.is_identity` or raw passthrough.
- Size gate at 300/1000 thresholds — log the gate decision.

### Tests for calibration ablation

```python
def test_ablation_loads_real_artifacts():
    """Script can load game_level_features.csv and the ensemble."""

def test_holdout_no_leakage():
    """Sealed holdout games are never seen during candidate selection."""

def test_isotonic_clipped_to_training_range():
    """Isotonic predictions never exceed the training min/max."""

def test_gate_adopt_requires_both_ece_and_logloss_improvement():
    """ADOPT only when ECE improves AND logloss doesn't degrade."""

def test_ablation_json_schema():
    """Output JSON has all required keys and numeric values."""
```

---

## 5. Full Test List + Acceptance Bar

### Existing tests (must remain green)

```
591 passed / 2 known environmental failures / 15 skipped
```

Known environmental failures (NOT regressions):
1. `test_frontend_markets.py::TestMarketsFetchMocked::test_correct_repo_file_exists_on_github` — GitHub raw-URL 404
2. `test_env_level_features.py::TestCommittedCacheCoverage::test_cache_read_from_data_delivery` — missing committed cache artifact

### New tests to add

| Test | File | Purpose |
|------|------|---------|
| Drift attach reuses training splits | test_build_oof_margin.py | Verify get_last_walk_forward_splits is called |
| Drift attach no desync on real frame | test_build_oof_margin.py | Real frame passes without AssertionError |
| Margin row present with finite PSI | test_psi.py | run_margin_diff in drift table |
| Guard still fires on wrong splits | test_build_oof_margin.py | Desync guard not removed |
| Empty fallback path | test_build_oof_margin.py | Graceful when no splits recorded |
| Ablation loads real artifacts | test_calibration_ablation.py (new) | Smoke test |
| Holdout no leakage | test_calibration_ablation.py (new) | Leak check |
| Isotonic clipped | test_calibration_ablation.py (new) | Range check |
| Gate logic | test_calibration_ablation.py (new) | ADOPT/DON'T ADOPT logic |
| Ablation JSON schema | test_calibration_ablation.py (new) | Output validation |

### Acceptance criteria

- **Drift fix**: `_attach_drift_run_margins` reuses training splits → no desync → run_margin_diff appears in drift PSI table with finite value on the real 2026-08-26 artifact.
- **Calibration ablation**: Standalone script runs, produces `calibration_ablation_*.json`, gate verdict is explicit. No committed changes to calibration math unless gate passes.
- **Full suite**: 591+ passed, only the 2 known environmental failures, no new failures.
- **No regressions**: moneyline monitor, run-line monitor, existing tabs, RF study all untouched.

---

## 6. Implementation Order

1. **Commit the partial `training.py` change** (already verified safe) — it's a no-op until wired.
2. **Wire `set_last_walk_forward_splits`** in `walk_forward_evaluate` (training.py:~1470).
3. **Wire `get_last_walk_forward_splits`** in `_attach_drift_run_margins` (pipeline.py:184) with fallback.
4. **Add import** `get_last_walk_forward_splits` to pipeline.py top-level imports.
5. **Write drift desync tests** (test_build_oof_margin.py or new test_drift_desync.py).
6. **Run full suite** — confirm 591+ passed, 2 known failures only.
7. **Write `run_calibration_ablation.py`** — standalone, imports from training/calibration/config.
8. **Write calibration ablation tests** (test_calibration_ablation.py).
9. **Run full suite again** — confirm no regressions.
10. **Commit**: "fix: drift margin attach reuses training splits; calibration ablation harness"

---

## 7. Key Line Number Reference

| Symbol | File | Line |
|--------|------|------|
| `_attach_drift_run_margins` | pipeline.py | 160 |
| `_attach_drift_run_margins` call site | pipeline.py | 1659 |
| `walk_forward_splits` in drift path | pipeline.py | 184 |
| `_attach_oof_run_margins` | training.py | 1333 |
| `_attach_oof_run_margins` desync guard | training.py | 1406 |
| `_attach_oof_run_margins` call in training | training.py | 1465 |
| `walk_forward_evaluate` splits creation | training.py | 1459 |
| `_LAST_WALK_FORWARD_SPLITS` (unwired) | training.py | 82 |
| `set_last_walk_forward_splits` (unwired) | training.py | 85 |
| `get_last_walk_forward_splits` (unwired) | training.py | 91 |
| `_regenerate_splits` | training.py | 1421 |
| `_attach_oof_run_margins` import in pipeline | pipeline.py | 68 |
| `walk_forward_splits` import in pipeline | pipeline.py | ~30 area |

## 8. Code Search Scoping Note

When searching for drift/calibration code, **always scope to `*.py` files** using `--include='*.py'` or equivalent. The `colab_run.ipynb` notebook (6000+ lines) and `data_delivery/*.csv`/`*.json` files overwhelm code_search results and make it impossible to find relevant Python source. Use `grep -rn --include='*.py'` in the terminal for targeted searches, and `read_files` with specific line ranges for reading actual source.
