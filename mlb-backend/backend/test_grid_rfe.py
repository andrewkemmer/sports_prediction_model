"""Grid-mode RFE unit tests (offline — no model training, no data files).

Covers the pure decision logic behind MLB_RFE_ADDITION_REMOVAL_GRID_MODE=1:

  * _grid_window       — the silent 16-state window (importance-ranked,
                         round-robin adds/removes; overflow recorded, never
                         dropped; env list order must not matter).
  * _grid_edge_verdict — the paired-difference commit bar (identical math
                         to a normal run: max(floor, 2 x paired SE) plus the
                         AUC/ECE guards) on synthetic loss vectors.
  * _state_label       — human labels for lattice states.

Run:  python mlb-backend/backend/test_grid_rfe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from feature_selection import _grid_edge_verdict, _grid_window, _state_label

PRIOR = {"a1": 10.0, "a2": 9.0, "a3": 8.0, "a4": 7.0, "a5": 6.0,
         "r1": 10.0, "r2": 9.0, "r3": 8.0, "r4": 7.0, "r5": 6.0}


def _lattice(wa: list[str], wr: list[str]) -> int:
    return 2 ** len(wa) * 2 ** len(wr)


# ── _grid_window: the silent state-cap window ──────────────────────────────

def test_window_user_case():
    # 2 adds + 2 removals fits the default 16-state cap exactly.
    wa, wr, tail = _grid_window(["a1", "a2"], ["r1", "r2"], 16, PRIOR)
    assert _lattice(wa, wr) == 16
    assert set(wa) == {"a1", "a2"} and set(wr) == {"r1", "r2"}
    assert tail == []


def test_window_four_and_four():
    # 4+4 would be 256 states — the window keeps the best 2+2 (16 states)
    # and spills the rest to overflow (honored later as 1-by-1 trials).
    wa, wr, tail = _grid_window(["a1", "a2", "a3", "a4"],
                                ["r1", "r2", "r3", "r4"], 16, PRIOR)
    assert _lattice(wa, wr) == 16
    assert set(wa) == {"a1", "a2"} and set(wr) == {"r1", "r2"}
    assert set(tail) == {"a3", "a4", "r3", "r4"}


def test_window_one_sided():
    # 1 addition alone: 2 states. 4 additions alone: exactly 16.
    wa, wr, tail = _grid_window(["a1"], [], 16, PRIOR)
    assert _lattice(wa, wr) == 2 and tail == []
    wa, wr, tail = _grid_window(["a1", "a2", "a3", "a4"], [], 16, PRIOR)
    assert _lattice(wa, wr) == 16 and set(wa) == {"a1", "a2", "a3", "a4"}
    assert tail == []


def test_window_mixed_shape():
    # 4 adds + 1 removal: 3A+1R = 16 states, a4 spills.
    wa, wr, tail = _grid_window(["a1", "a2", "a3", "a4"], ["r1"], 16, PRIOR)
    assert _lattice(wa, wr) == 16
    assert set(wa) == {"a1", "a2", "a3"} and set(wr) == {"r1"}
    assert tail == ["a4"]


def test_window_order_independence():
    # The window is importance-ranked, NOT env-list ordered: the same names
    # in any order must produce the identical lattice.
    wa1, wr1, t1 = _grid_window(["a2", "a1"], ["r2", "r1"], 16, PRIOR)
    wa2, wr2, t2 = _grid_window(["a1", "a2"], ["r1", "r2"], 16, PRIOR)
    assert (wa1, wr1, t1) == (wa2, wr2, t2)


def test_window_empty_lists():
    wa, wr, tail = _grid_window([], [], 16, PRIOR)
    assert (wa, wr, tail) == ([], [], [])


def test_window_bigger_cap_expands():
    # cap 32 admits 3A+2R; the round-robin prefers adds first on the tie.
    wa, wr, tail = _grid_window(["a1", "a2", "a3", "a4"],
                                ["r1", "r2", "r3", "r4"], 32, PRIOR)
    assert _lattice(wa, wr) == 32
    assert set(wa) == {"a1", "a2", "a3"} and set(wr) == {"r1", "r2"}
    assert set(tail) == {"a4", "r3", "r4"}


# ── _grid_edge_verdict: the paired-difference commit bar ───────────────────

def _vectors(n: int = 500, shift: float = 0.0, noise: float = 0.05,
             seed: int = 7):
    """Paired per-game loss vectors: `to` is better by `shift` on average,
    with independent per-game noise (so the paired SE is realistic)."""
    rng = np.random.default_rng(seed)
    frm = rng.normal(0.69, noise, size=n)
    to = rng.normal(0.69 - shift, noise, size=n)
    return np.clip(frm, 1e-6, None), np.clip(to, 1e-6, None)


def test_edge_commits_above_bar():
    # Gain 0.02 vs a ~2-sigma bar of ~0.006 -> commit.
    frm, to = _vectors(shift=0.02)
    fm = {"logloss": 0.69, "auc": 0.58, "ece": 0.010}
    tm = {"logloss": 0.67, "auc": 0.58, "ece": 0.010}
    e = _grid_edge_verdict("baseline", "trial", frm, to, fm, tm,
                           0.0005, 2.0, 0.003, 0.005, 0.0032)
    assert e["committed"] is True
    assert e["bar_basis"] == "paired_diff_2sigma"


def test_edge_rejects_below_bar():
    # Gain 0.002 vs a ~2-sigma bar of ~0.006 -> rejected.
    frm, to = _vectors(shift=0.002)
    fm = {"logloss": 0.69, "auc": 0.58, "ece": 0.010}
    tm = {"logloss": 0.688, "auc": 0.58, "ece": 0.010}
    e = _grid_edge_verdict("baseline", "trial", frm, to, fm, tm,
                           0.0005, 2.0, 0.003, 0.005, 0.0032)
    assert e["committed"] is False
    assert e["commit_threshold"] > 0.002


def test_edge_guard_blocks_auc():
    # Big logloss gain, but AUC craters beyond the guard -> no commit.
    frm, to = _vectors(shift=0.02)
    fm = {"logloss": 0.69, "auc": 0.58, "ece": 0.010}
    tm = {"logloss": 0.67, "auc": 0.58 - 0.01, "ece": 0.010}
    e = _grid_edge_verdict("baseline", "trial", frm, to, fm, tm,
                           0.0005, 2.0, 0.003, 0.005, 0.0032)
    assert e["committed"] is False and e["auc_drop"] > 0.003


def test_edge_guard_blocks_ece():
    # Big logloss gain, but calibration degrades beyond the guard.
    frm, to = _vectors(shift=0.02)
    fm = {"logloss": 0.69, "auc": 0.58, "ece": 0.010}
    tm = {"logloss": 0.67, "auc": 0.58, "ece": 0.020}  # rise 0.010 > 0.005
    e = _grid_edge_verdict("baseline", "trial", frm, to, fm, tm,
                           0.0005, 2.0, 0.003, 0.005, 0.0032)
    assert e["committed"] is False and e["ece_rise"] > 0.005


def test_edge_fallback_without_loss_vectors():
    # Missing or mismatched vectors cannot pair — the baseline-SE fallback
    # bar governs (here 0.0032; a 0.005 gain clears it).
    fm = {"logloss": 0.69, "auc": 0.58, "ece": 0.010}
    tm = {"logloss": 0.685, "auc": 0.58, "ece": 0.010}
    e = _grid_edge_verdict("baseline", "trial", None, None, fm, tm,
                           0.0005, 2.0, 0.003, 0.005, 0.0032)
    assert e["bar_basis"] == "baseline_se_fallback"
    assert e["committed"] is True
    e2 = _grid_edge_verdict("baseline", "trial",
                            np.zeros(10), np.zeros(11), fm, tm,
                            0.0005, 2.0, 0.003, 0.005, 0.0032)
    assert e2["bar_basis"] == "baseline_se_fallback"


def test_edge_handles_failed_state():
    e = _grid_edge_verdict("baseline", "trial", None, None, None, None,
                           0.0005, 2.0, 0.003, 0.005, 0.0032)
    assert e["committed"] is False and "not scored" in e["verdict"]


# ── _state_label ────────────────────────────────────────────────────────────

def test_state_label():
    assert _state_label(((), ())) == "baseline"
    assert _state_label((("a1",), ())) == "+a1"
    assert _state_label(((), ("r1",))) == "−r1"
    assert _state_label((("a1", "a2"), ("r1",))) == "+a1+a2 −r1"



# ── SINGLE-LIST RULE (2026-09-17) ───────────────────────────────────────────
# Every monitor-facing enumeration (drift PSI, coverage, SHAP matrices,
# feature weights, tooltips) must read the ACTIVE moneyline serving width —
# never the generation universe directly. These tests import the real
# training + explainability modules (no model training, no data files), so
# a regression to a raw-universe default fails loudly here.

def test_single_list_no_margin_in_any_enumeration():
    """run_margin_diff must be absent from every list-level surface, even
    though the column is still computed and can appear in input frames."""
    import training
    from training import MARGIN_COL
    import feature_selection

    assert MARGIN_COL == "run_margin_diff"
    assert MARGIN_COL not in training.MONEYLINE_FEATURE_COLS, (
        "generation universe still carries run_margin_diff")
    assert MARGIN_COL not in training.KNOWN_FEATURE_COLS, (
        "known pool still carries run_margin_diff")
    feature_selection.reset_feature_subset()
    active = training.active_moneyline_feature_cols()
    assert MARGIN_COL not in active
    assert len(active) == len(training.MONEYLINE_FEATURE_COLS) == 62, (
        f"width drift: universe={len(training.MONEYLINE_FEATURE_COLS)} "
        f"active={len(active)} (expected 62 everywhere)")

def test_drift_default_enumerates_active_width():
    """compute_feature_drift's default enumeration is the ACTIVE serving
    width: a frame that still contains run_margin_diff must produce NO PSI
    row for it once it has left the serving list."""
    import pandas as pd
    import training
    import feature_selection
    from explainability import compute_feature_drift

    feature_selection.reset_feature_subset()
    cols = training.active_moneyline_feature_cols()
    rng = np.random.default_rng(7)
    n = 40
    base = pd.DataFrame(
        rng.normal(size=(n, len(cols) + 1)),
        columns=cols + ["run_margin_diff"])  # poison: column present in frame
    cur = base * 1.01
    df = compute_feature_drift(base, cur, "2099-01-01", out_name="_t_drift.csv")
    assert "run_margin_diff" not in set(df["feature"]), (
        "drift emitted a PSI row for a non-serving feature")
    assert len(df) == len(cols), f"expected {len(cols)} rows, got {len(df)}"

def test_coverage_default_enumerates_active_width():
    """compute_feature_coverage's default enumeration matches the serving
    width (same rule as drift) — no coverage row for non-serving features."""
    import pandas as pd
    import training
    import feature_selection
    from explainability import compute_feature_coverage

    feature_selection.reset_feature_subset()
    cols = training.active_moneyline_feature_cols()
    rng = np.random.default_rng(11)
    n = 40
    base = pd.DataFrame(
        rng.normal(size=(n, len(cols) + 1)),
        columns=cols + ["run_margin_diff"])
    cur = base
    df = compute_feature_coverage(base, cur, "2099-01-01",
                                  out_name="_t_coverage.csv")
    assert "run_margin_diff" not in set(df["feature"]), (
        "coverage emitted a row for a non-serving feature")
    # Coverage emits one row per feature per window (current + baseline).
    assert len(df) == 2 * len(cols), (
        f"expected {2 * len(cols)} rows (2 windows x {len(cols)}), got {len(df)}")

if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(fns)} grid-RFE tests passed (offline, no data needed)")
