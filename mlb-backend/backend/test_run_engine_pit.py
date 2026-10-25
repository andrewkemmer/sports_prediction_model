"""Point-in-time regression pins for calibrated split-skew Gaussian
Monte Carlo totals and margin pricing.

The production OOF hit rate (~51% on favored-side cards) and the live
board's realized rate can only diverge through train/serve skew, so this
suite pins every PIT invariant the engine's honesty depends on:

  1. fold geometry: expanding, train strictly before validation;
  2. causal whole-CDF calibration: a fold's map sees PRIOR folds only;
  3. sealed holdout: variance, skew and calibration fitting never see the
     holdout window, and poisoned sealed outcomes cannot move issued prices;
  4. OOF and slate pricing share one calibrated Monte Carlo family;
  5. score means remain the per-side regressors' market-free forecasts.

Run with: python mlb-backend/backend/test_run_engine_pit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch as _mock_patch

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
# The card-honesty tests (section 8) exercise the FRONTEND resolver/loader
# that serves the board — the historical game card's price source lives in
# frontend/, not in the backend package.
_FRONTEND = (BACKEND.parent.parent / "frontend").resolve()
if _FRONTEND.is_dir():
    sys.path.insert(0, str(_FRONTEND))

import distributions as re            # noqa: E402
from training import canonical_walk_forward_splits  # noqa: E402


def _synth_oof(n_days: int = 40, games_per_day: int = 6, seed: int = 7
               ) -> pd.DataFrame:
    """Deterministic synthetic decided/OOF frame with the columns the run
    engine's PIT machinery needs (folds, holdout masks, k-edge fits)."""
    rng = np.random.default_rng(seed)
    rows = []
    pk = 5000
    base = pd.Timestamp("2026-07-01")
    for d in range(n_days):
        day = (base + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for g in range(games_per_day):
            pk += 1
            lam_h = 4.4 + rng.normal(0, 0.4)
            lam_a = 4.3 + rng.normal(0, 0.4)
            margin = rng.normal((lam_h - lam_a) * 1.2, 3.2)
            hs = int(max(0, round(lam_h + margin / 2)))
            as_ = int(max(0, round(lam_a - margin / 2)))
            rows.append({
                "game_pk": pk,
                "game_id": f"{day.replace('-', '')}_A{g}@H{g}",
                "game_date": day,
                "home_team": "HOM", "away_team": "AWY",
                "home_expected_runs": round(lam_h, 4),
                "away_expected_runs": round(lam_a, 4),
                "home_score": hs, "away_score": as_,
                "total_runs": hs + as_,
                "home_win": float(hs > as_),
                "fold_idx": d // 7,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Fold geometry
# ---------------------------------------------------------------------------
def test_run_engine_folds_are_expanding_and_strictly_prior():
    """Every fold's training games end strictly before its validation
    window starts, and training volume grows monotonically (expanding)."""
    oof = _synth_oof()
    decided, folds = canonical_walk_forward_splits(
        oof, retrain_cadence_days=7, max_eval_folds=0,
        min_train_days=0, min_val_games=1)
    assert len(folds) > 3, f"expected multiple folds, got {len(folds)}"
    sizes = [len(s["train_games"]) for s in folds]
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), \
        f"training folds not expanding: {sizes}"
    for s in folds:
        tr_end = pd.to_datetime(s["train_games"]["game_date"]).max()
        va_start = pd.to_datetime(s["val_games"]["game_date"]).min()
        assert tr_end < va_start, (
            f"fold {s.get('fold_idx')}: train ends {tr_end.date()} on/after "
            f"validation start {va_start.date()}")


# ---------------------------------------------------------------------------
# 2. Prequential calibration sees prior folds only
# ---------------------------------------------------------------------------
def test_causal_calibration_and_sealed_holdout_are_strictly_prior():
    oof = _synth_oof(n_days=40)
    original = re.derive_markets_v3(oof.copy())
    cutoff = pd.Timestamp(original["summary"]["holdout_cutoff"])
    changed = oof.copy()
    changed.loc[pd.to_datetime(changed.game_date) >= cutoff, "home_score"] += 20
    poisoned = re.derive_markets_v3(changed)
    cols = [c for c in original["markets"] if c.startswith("p_")]
    np.testing.assert_array_equal(original["markets"][cols], poisoned["markets"][cols])
    bundle = original["summary"]["calibration"]["calibrators"]
    other = poisoned["summary"]["calibration"]["calibrators"]
    assert bundle["origins"] == other["origins"]
    assert bundle["method"] == "causal_split_skew_crps_mc"
    assert bundle["monte_carlo"]["n_draws"] == 50000


# ---------------------------------------------------------------------------
# 6. Frozen Game-Totals prediction-history store (first publication, PIT)
# ---------------------------------------------------------------------------
def _th_synthetic_oof_frame(game_pk=123, *, p80=(0.46314, 0.46267),
                            p85=(0.37, 0.63), p90=(0.25, 0.70),
                            total_runs=9):
    """One decided OOF row with a small valid totals grid (8.0/8.5/9.0)."""
    return pd.DataFrame([{
        "game_pk": game_pk, "game_date": "2026-09-18", "kind": "oof",
        "home_score": 5, "away_score": 4, "total_runs": total_runs,
        "home_expected_runs": 4.0, "away_expected_runs": 4.3,
        "p_over_8_0": p80[0], "p_under_8_0": p80[1],
        "p_over_8_5": p85[0], "p_under_8_5": p85[1],
        "p_over_9_0": p90[0], "p_under_9_0": p90[1],
    }])


def _th_with_tmp_store(fn):
    """Run fn(tmp_path) with the store pointed at a hermetic tmp dir."""
    import tempfile
    original = re.DATA_DELIVERY_DIR
    tmp = Path(tempfile.mkdtemp(prefix="th_store_"))
    re.DATA_DELIVERY_DIR = tmp
    try:
        fn(tmp)
    finally:
        re.DATA_DELIVERY_DIR = original


def test_totals_history_store_prices_fair_line_and_rescaled_pick():
    # Re-scaled P(over|8.0) = 0.50025 -> fair line 8.0, Over pick; total 9
    # beats the line -> winner Over, correct 1.0; provenance ISO-dated.
    def body(tmp):
        out = re.update_totals_history_store(
            _th_synthetic_oof_frame(), "20260919")
        assert out is not None and out.exists()
        store = pd.read_csv(out, dtype={"game_pk": str})
        assert len(store) == 1
        r = store.iloc[0]
        assert r["game_pk"] == "123"
        assert float(r["line"]) == 8.0
        assert r["pick"] == "Over"
        assert abs(float(r["pick_prob"]) - 0.500254) < 1e-5
        assert r["winner"] == "Over"
        assert float(r["correct"]) == 1.0
        assert r["source_artifact_date"] == "2026-09-19"
    _th_with_tmp_store(body)


def test_totals_history_store_push_grading():
    # total_runs == whole-number fair line -> Push, correct excluded (NaN).
    def body(tmp):
        re.update_totals_history_store(
            _th_synthetic_oof_frame(total_runs=8), "20260919")
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv")
        r = store.iloc[0]
        assert r["winner"] == "Push"
        assert pd.isna(r["correct"])
    _th_with_tmp_store(body)


def test_totals_history_store_first_publication_wins_and_idempotent():
    # Once frozen, later runs (even with different prices) never mutate the
    # row; re-runs add nothing; only NEW game_pks append with own provenance.
    def body(tmp):
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123), "20260919")
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123, p80=(0.30, 0.70),
                                    p85=(0.20, 0.80), p90=(0.10, 0.90)),
            "20260922")
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv",
                            dtype={"game_pk": str})
        assert len(store) == 1
        assert abs(float(store.iloc[0]["pick_prob"]) - 0.500254) < 1e-5
        assert store.iloc[0]["source_artifact_date"] == "2026-09-19"
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=456), "20260922")
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv",
                            dtype={"game_pk": str})
        assert len(store) == 2
        assert set(store["game_pk"]) == {"123", "456"}
        src = dict(zip(store["game_pk"], store["source_artifact_date"]))
        assert src == {"123": "2026-09-19", "456": "2026-09-22"}
    _th_with_tmp_store(body)


def test_totals_history_store_pk_dtype_normalized_and_never_raises():
    # Store pk stored as string; an int-pk artifact row must not duplicate.
    # A corrupt store must return None (logged), never raise.
    def body(tmp):
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123), "20260919")
        re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=123), "20260920")  # int pk
        store = pd.read_csv(re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.csv")
        assert len(store) == 1
        (re.DATA_DELIVERY_DIR
         / "run_engine_totals_history.csv").write_text("not,a,store\n1,2")
        out = re.update_totals_history_store(
            _th_synthetic_oof_frame(game_pk=999), "20260921")
        assert out is None
        import json as _json
        meta = _json.loads((re.DATA_DELIVERY_DIR
                            / "run_engine_totals_history.meta.json")
                           .read_text(encoding="utf-8"))
        assert meta.get("error")
    _th_with_tmp_store(body)


# ---------------------------------------------------------------------------
# 8. Historical game-card price honesty (2026-09-23 regression) + the
#    rolling 10-day board window (2026-09-23 rev 2, owner decision)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 9. Run-engine team-ID categoricals (2026-09-23 ADOPT)
# ---------------------------------------------------------------------------

def test_run_engine_team_id_seam_contract():
    """The run engine's Poisson side models carry the moneyline's team-ID
    pair as LightGBM-native categoricals (scratch/runengine_teamid_
    ablation.json: both sides improve on deviance AND CRPS with paired
    per-fold SEs). Contract pinned here:
      * the numeric kept/dropped view is UNCHANGED — the pair is never in
        build_side_frame's frame, only appended to its column list;
      * _apply_categorical_ids attaches both columns as category dtype,
        resolved from the pre-slice rows 1:1 (length mismatch = loud error);
      * with the flag off, nothing is attached (numeric-only contract).
    """
    import pandas as pd
    import distributions as re

    assert re.RUN_TREE_CATEGORICAL_COLS == ["home_team_id", "away_team_id"]
    assert re.RUN_WITH_TEAM_IDS is True

    games = pd.DataFrame({
        "game_pk": [1, 2, 3],
        "game_date": ["2026-09-20"] * 3,
        "home_team": ["NYY", "BOS", "NYY"],
        "away_team": ["BOS", "NYY", "BOS"],
        "home_score": [5, 3, 2], "away_score": [2, 4, 1],
        "home_win": [1.0, 0.0, 1.0],
    })
    numeric_cols = ["elo_diff"]
    games["elo_diff"] = [0.1, -0.2, 0.3]
    frame, cols = re.build_side_frame(games, "home")
    assert "home_team_id" not in frame.columns  # frame stays numeric
    assert cols[-2:] == ["home_team_id", "away_team_id"]  # list metadata

    tr = games.reindex(columns=cols).astype(float)
    re._apply_categorical_ids(tr, rows=games)
    for c in re.RUN_TREE_CATEGORICAL_COLS:
        assert str(tr[c].dtypes) == "category"
    # Same abbreviation -> same integer across rows (mapper stability).
    nyy = tr.loc[games["home_team"] == "NYY", "home_team_id"]
    assert set(nyy.astype(str)) == {str(nyy.iloc[0])}

    # Length mismatch must fail loud (positional 1:1 contract).
    import pytest
    with pytest.raises(ValueError):
        re._apply_categorical_ids(tr, rows=games.iloc[:2])

    # Flag off -> numeric-only contract restored.
    re.RUN_WITH_TEAM_IDS = False
    try:
        frame2, cols2 = re.build_side_frame(games, "home")
        assert cols2[-2:] != ["home_team_id", "away_team_id"]
        tr2 = games.reindex(columns=cols2).astype(float)
        re._apply_categorical_ids(tr2, rows=games)  # no-op
        assert "home_team_id" not in tr2.columns
    finally:
        re.RUN_WITH_TEAM_IDS = True


def test_retention_board_families_age_out_on_the_10_day_window():
    """REV 2 (2026-09-23): the dated board families ride the blanket 10-day
    window — the Today's Games dashboard serves a ROLLING 10 DAYS of
    predictions, never a historical archive.

    The permanence test (rev 1, the 2026-09-11 card regression) is
    preserved for dates INSIDE the window: board + run-engine markets
    (+meta) + SHAP all keep together, so a served card still renders the
    production prices AS PUBLISHED (never an OOF re-price or cross-date
    binding). Beyond the window every card family is stale together and
    the frontend never offers the date (its own same-anchor filter).
    Series/never-delete families are untouched."""
    import retention_policy as rp

    keep = {"current", "seen", "protected"}

    # Inside the blanket window (anchor -10 .. anchor): every date-gated card
    # pricing family keeps, including the markets CSV and its meta companion.
    retention_dates = {"20260914", "20260920", "20260923"}
    for rel in (
        "mlb-backend/data_delivery/todays_games_20260920.csv",
        "mlb-backend/data_delivery/run_engine_markets_20260920.csv",
        "mlb-backend/data_delivery/run_engine_markets_20260920.meta.json",
        "mlb-backend/data_delivery/shap_game_20260920_TB@NYY.csv",
        "mlb-backend/data_delivery/run_engine_oof_20260920.csv",
        "mlb-backend/data_delivery/predictions_history_20260920.csv",
    ):
        verdict = rp.classify_artifact(
            rel, seen=set(), retention_dates=retention_dates,
            recent_dates=set(), board_dates=set(), anchor_date="20260923")
        assert verdict in keep, f"{rel} classified {verdict} — in-window keep broken"

    # Older than the window (no anchor guard, no board tracked): every
    # card family is stale TOGETHER — the pipeline prunes them; a served
    # card can never outlive its pricing companions.
    for rel in (
        "mlb-backend/data_delivery/todays_games_20260911.csv",
        "mlb-backend/data_delivery/run_engine_markets_20260911.csv",
        "mlb-backend/data_delivery/run_engine_markets_20260911.meta.json",
        "mlb-backend/data_delivery/shap_game_20260911_TB@NYY.csv",
        "mlb-backend/data_delivery/run_engine_oof_20260911.csv",
        "mlb-backend/data_delivery/predictions_history_20260911.csv",
    ):
        verdict = rp.classify_artifact(
            rel, seen=set(), retention_dates=set(), recent_dates=set(),
            board_dates=set(), anchor_date="20260923")
        assert verdict == "stale", f"{rel} classified {verdict} — survives the window"

    # The markets CSV has NO board-backed exception (2026-10-25): a board
    # still tracked outside the blanket window must not extend its dated
    # markets artifact. Other companions still follow the board.
    stale_markets = rp.classify_artifact(
        "mlb-backend/data_delivery/run_engine_markets_20260911.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates={"20260911"}, anchor_date="20260923")
    assert stale_markets == "stale", (
        "the markets CSV was reprieved by a tracked board; the dated markets "
        "family must ride the blanket 10-day window with no exception")
    companion = rp.classify_artifact(
        "mlb-backend/data_delivery/predictions_history_20260911.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates={"20260911"}, anchor_date="20260923")
    assert companion == "current", (
        "the genuine board-backed companion rule was removed with markets")

    # Backfill safety: artifacts dated NEWER than the run's anchor keep
    # regardless of windows (present-day artifacts survive a past-anchored
    # backfill run).
    backfill = rp.classify_artifact(
        "mlb-backend/data_delivery/todays_games_20260930.csv",
        seen=set(), retention_dates=set(), recent_dates=set(),
        board_dates=set(), anchor_date="20260923")
    assert backfill == "current"

    # Series / never-delete families are unaffected by the revision.
    for rel in (
        "mlb-backend/data_delivery/run_engine_monitor_20260911.json",
        "mlb-backend/data_delivery/models/ensemble_latest.joblib",
        "mlb-backend/data_delivery/game_level_features.csv",
    ):
        verdict = rp.classify_artifact(
            rel, seen=set(), retention_dates=set(), recent_dates=set(),
            board_dates=set(), anchor_date="20260923")
        assert verdict == "protected", f"{rel} classified {verdict} — must be protected"


def test_slate_resolver_never_binds_across_a_distant_date():
    """The resolver may bind a card only to pricing from its own date or
    the GMT-rollover day: with artifacts from 20260911 and 20260922 only,
    a Sep 11 card that the Sep 11 artifact cannot price must stay
    UNRESOLVED — never pick up the Sep 22 run's re-pricing of the same
    matchup (2026-09-23: the Sep 11 board showing Sep 22 prices). An id
    the Sep 11 artifact DOES price keeps its exact-date binding."""
    from market_diagnostics import resolve_slate_across_artifacts
    priced = pd.DataFrame([{
        "game_pk": "20260911_NYM@NYY", "kind": "slate",
        "home_expected_runs": 4.6, "away_expected_runs": 4.1,
    }])
    repriced = pd.DataFrame([{
        "game_pk": "20260922_TB@NYY", "kind": "slate",
        "home_expected_runs": 3.7, "away_expected_runs": 3.5,
    }])
    out = resolve_slate_across_artifacts(
        {"20260911": priced, "20260922": repriced},
        ["20260911_NYM@NYY", "20260911_TB@NYY"])
    # Own-date exact binding survives.
    assert "20260911_NYM@NYY" in out
    assert out["20260911_NYM@NYY"]["artifact_date"] == "20260911"
    # The distant-future re-price of the same matchup is unreachable.
    assert "20260911_TB@NYY" not in out


def test_shap_loader_never_rewrites_the_game_date():
    """A date-rewritten SHAP id attributes a DIFFERENT game's predictions
    to a card (2026-09-23: Sep 11 cards rendering 20260922_TB@NYY SHAP).
    The loader must try only the game's own date and the ±1 rollover day
    — never the newest run."""
    import utils as futils
    calls: list[str] = []

    def fake_fetch(name, **_kw):
        calls.append(name)
        return (None, "history")

    with _mock_patch.object(futils, "_fetch_bytes", side_effect=fake_fetch), \
            _mock_patch.object(futils, "available_dates",
                               return_value=["20260922"]):
        out = futils.load_shap("20260911_TB@NYY", "20260911")
    assert out.empty
    assert calls == ["shap_game_20260911_TB@NYY.csv",
                     "shap_game_20260912_TB@NYY.csv",
                     "shap_game_20260910_TB@NYY.csv"], calls


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\n{len(tests)} run-engine PIT tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
