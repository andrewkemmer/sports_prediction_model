"""Grid-mode RFE unit tests (offline — no model training, no data files).

Covers the pure decision logic behind MLB_RFE_ADDITION_REMOVAL_GRID_MODE=1:

  * _grid_window       — the silent 16-state window (importance-ranked,
                         round-robin adds/removes; overflow recorded, never
                         dropped; env list order must not matter).
  * _grid_edge_verdict — the paired-difference commit bar (identical math
                         to a normal run: max(floor, sigma x paired SE) plus
                         the AUC/ECE guards) on synthetic loss vectors.
  * _state_label       — human labels for lattice states.

Run:  python mlb-backend/backend/test_grid_rfe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
PROBE_DIR = BACKEND_DIR.parent / "data_delivery"

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


# ── 1σ COMMIT BAR (2026-09-23, NFL parity) ─────────────────────────────────

def test_config_pins_one_sigma_bar():
    """The MLB RFE commit bar is 1 standard error of the PAIRED per-game
    logloss difference — numeric parity with the NFL RFE bar
    (nfl config RFE_COMMIT_SE_MULTIPLE = 1.0, same 0.0005 floor and
    AUC/ECE guards). The paired construction self-calibrates junk features
    to a high bar, so the looser multiple does not admit noise commits."""
    import config

    assert config.RFE_NOISE_SIGMA == 1.0
    # NFL parity, verified in-tree: the two knobs must agree.
    nfl_config_path = BACKEND_DIR.parents[1] / "nfl-backend" / "backend" / "config.py"
    assert nfl_config_path.exists()
    text = nfl_config_path.read_text(encoding="utf-8", errors="replace")
    import re as _re
    m = _re.search(r"RFE_COMMIT_SE_MULTIPLE\s*=\s*([0-9.]+)", text)
    assert m and float(m.group(1)) == 1.0, "NFL RFE bar drifted from 1σ"


def test_edge_one_sigma_commits_and_rejects():
    """At sigma=1.0 the paired bar is ~0.003 (SE ≈ 0.0032 on these vectors):
    a real 0.02 gain commits with the truthful 1σ label; a 0.002 gain
    (≈0.6σ) is rejected."""
    frm, to = _vectors(shift=0.02)
    fm = {"logloss": 0.69, "auc": 0.58, "ece": 0.010}
    tm = {"logloss": 0.67, "auc": 0.58, "ece": 0.010}
    e = _grid_edge_verdict("baseline", "trial", frm, to, fm, tm,
                           0.0005, 1.0, 0.003, 0.005, 0.0032)
    assert e["committed"] is True
    assert e["bar_basis"] == "paired_diff_1sigma"

    frm2, to2 = _vectors(shift=0.002)
    tm2 = {"logloss": 0.688, "auc": 0.58, "ece": 0.010}
    e2 = _grid_edge_verdict("baseline", "trial", frm2, to2, fm, tm2,
                            0.0005, 1.0, 0.003, 0.005, 0.0032)
    assert e2["committed"] is False
    assert e2["commit_threshold"] > 0.002


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
    # 109 = the 2026-10-03 contract: 101 − 9 (re24) − 7 (depth/sp_xfip/
    # slug) + 24 (8 pl pools × home/away/diff) → 101 + 8.
    assert len(active) == len(training.MONEYLINE_FEATURE_COLS) == 109, (
        f"width drift: universe={len(training.MONEYLINE_FEATURE_COLS)} "
        f"active={len(active)} (expected 109 everywhere)")

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
    try:
        assert "run_margin_diff" not in set(df["feature"]), (
            "drift emitted a PSI row for a non-serving feature")
        assert len(df) == len(cols), f"expected {len(cols)} rows, got {len(df)}"
    finally:
        # Probe output is test scratch — never leave it in data_delivery.
        (PROBE_DIR / "_t_drift.csv").unlink(missing_ok=True)

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
    try:
        assert "run_margin_diff" not in set(df["feature"]), (
            "coverage emitted a row for a non-serving feature")
        # Coverage emits one row per feature per window (current + baseline).
        assert len(df) == 2 * len(cols), (
            f"expected {2 * len(cols)} rows (2 windows x {len(cols)}), got {len(df)}")
    finally:
        # Probe output is test scratch — never leave it in data_delivery.
        (PROBE_DIR / "_t_coverage.csv").unlink(missing_ok=True)

def test_coverage_reports_missing_invalid_and_empty(monkeypatch, tmp_path):
    import pandas as pd
    import explainability
    monkeypatch.setattr(explainability, "DATA_DELIVERY_DIR", tmp_path)
    frame = pd.DataFrame({"observed": [1.0, np.inf, np.nan]})
    cov = explainability.compute_feature_coverage(
        frame, frame, "20990101", feature_cols=["observed", "missing"])
    assert len(cov) == 4
    missing = cov[cov.feature == "missing"]
    assert (missing.status == "MISSING_COLUMN").all()
    assert (missing.n_measured == 0).all()
    observed = cov[cov.feature == "observed"]
    assert (observed.n_measured == 1).all()
    assert (observed.n_invalid == 1).all()
    empty = explainability.compute_feature_coverage(
        frame.iloc[:0], frame.iloc[:0], "20990102", feature_cols=["observed"])
    assert empty.empty and "status" in empty


def test_rfe_rolls_prior_weights_and_grades_only_regular_rows(monkeypatch):
    import pandas as pd
    import feature_selection as fs
    import training
    training.set_adaptive_weights({"elasticnet": 1.0})
    def train(tr, val):
        return {"m1": object(), "m2": object()}, {}
    calls = []
    def predict(models, val):
        weights = training._member_weights(["m1", "m2"])
        calls.append(weights)
        members = {"m1": np.linspace(0.2, 0.8, len(val)),
                   "m2": np.linspace(0.8, 0.2, len(val))}
        return training._pool_member_probs(members, weights), members, weights
    def fold(k, types):
        val = pd.DataFrame({"home_win": np.tile([0, 1], 25), "game_type": types,
                            "game_date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=k * 7)})
        return {"train_games": val.copy(), "val_games": val, "fold_idx": k}
    monkeypatch.setattr(fs, "train_moneyline_ensemble", train)
    monkeypatch.setattr(fs, "ensemble_predict", predict)
    try:
        scored = fs._score_splits([fold(0, ["R"] * 50), fold(1, ["W"] * 50),
                                  fold(2, ["R"] * 50)])
        assert scored["games_scored"] == 100 and scored["folds_used"] == 2
        assert calls[0] == {"m1": 0.5, "m2": 0.5}
        assert calls[1] == calls[2]  # postseason outcomes cannot earn weights
        assert training._LAST_ADAPTIVE_WEIGHTS == {"elasticnet": 1.0}
    finally:
        training.set_adaptive_weights(None)


def test_rfe_candidate_state_is_isolated_and_restored(monkeypatch):
    import feature_selection as fs
    import training
    training.set_adaptive_weights({"elasticnet": 1.0})
    training._LAST_XGB_BEST_ROUNDS[:] = [999]
    def score(splits, return_losses):
        assert training._LAST_ADAPTIVE_WEIGHTS == {}
        assert training._LAST_XGB_BEST_ROUNDS == []
        training.set_adaptive_weights({"xgboost": 1.0})
        training._LAST_XGB_BEST_ROUNDS.append(2)
        return {"auc": 0.5}
    monkeypatch.setattr(fs, "_score_splits_causal", score)
    try:
        assert fs._score_splits([]) == {"auc": 0.5}
        assert training._LAST_ADAPTIVE_WEIGHTS == {"elasticnet": 1.0}
        assert training._LAST_XGB_BEST_ROUNDS == [999]
    finally:
        training.set_adaptive_weights(None)
        training._LAST_XGB_BEST_ROUNDS.clear()


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(fns)} grid-RFE tests passed (offline, no data needed)")
