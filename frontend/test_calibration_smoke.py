"""Smoke test — Calibration page, SPORT-DISPATCHED (steps 3 + Part B).

The NFL Calibration page now runs the SAME code as MLB (no sport-special
path). This test:

1. Writes REPRESENTATIVE ``nfl_calibration_*.json`` + ``nfl_predictions_\
   history_*.csv`` artifacts (matching the exact shape ``nfl_moneyline.py``
   Part-A emits) into the real ``nfl-backend/data_delivery`` dir, so the
   page renders with real data (they are removed after the run).
2. Runs the ACTUAL ``model_calibration.py`` under ``sport=nfl`` and asserts
   the MLB-identical seven sections: header pill, today's-record summary
   card, the four KPI cards (AUC/Brier/Log-Loss/Cal. Error raw→calibrated),
   the Platt recalibration banner, the per-1% calibration CURVE as a real
   Altair chart, the reliability table WITH rows + a TOTAL row, and the
   populated prediction-history table — with no exceptions.
3. Runs the same page under ``sport=mlb`` and asserts it also runs clean
   (the shared path; locally it halts/warns on missing MLB artifacts rather
   than crashing).

Run from the frontend/ directory:
    python -m test_calibration_smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent if FRONTEND_DIR.name == "frontend" else FRONTEND_DIR
NFL_DD = REPO_ROOT / "nfl-backend" / "data_delivery"

# Strictly newer than ANY committed artifact so the fixture is always the one
# the page's newest-date resolution picks up (removed after the run). This
# must be a fixed far-future date, NOT "newer than whatever landed last" —
# that assumption rotted as new nfl_calibration_*.json artifacts were
# committed, and the page silently rendered real data instead of the fixture.
ARTIFACT_DATE = "20991231"
CALIBRATION_NAME = f"nfl_calibration_{ARTIFACT_DATE}.json"
HISTORY_NAME = f"nfl_predictions_history_{ARTIFACT_DATE}.csv"
CALIBRATION_PATH = NFL_DD / CALIBRATION_NAME
HISTORY_PATH = NFL_DD / HISTORY_NAME

WRITTEN: list[Path] = []
# Path -> original bytes of a PRE-EXISTING artifact this test overwrites with a
# fixture (e.g. a committed nfl_calibration_*.json). They are restored on
# cleanup — never deleted — so running the smoke test can't remove real
# committed artifacts from the tree.
_BACKUPS: dict[Path, bytes] = {}


# ---------------------------------------------------------------------------
# Representative artifact construction (matches the emitted Part-A schema)
# ---------------------------------------------------------------------------
def _calibration_record() -> dict:
    """A realistic nfl_calibration_*.json mirroring the MLB-shaped writer.

    Buckets carry the MLB presentation fields (bucket/mean_predicted/
    mean_actual/count/gap); the ``calibration`` block carries the deployed
    Platt params + raw→calibrated metrics exactly like ``calibration_*.json``.
    Values are NFL pipeline-shaped (decided OOF pool, favored view ≥ 50%)."""
    seq = [0.52, 0.60, 0.68, 0.76, 0.84, 0.92]   # favored-only: 50%..100%
    counts = [130, 210, 260, 230, 170, 107]
    buckets, cal_buckets = [], []
    for i, mp in enumerate(seq):
        ma = min(0.99, mp + 0.02)
        cal_mp = min(0.99, mp + 0.035)
        buckets.append({"bucket": f"{int(mp * 100)}%-{int(mp * 100) + 8}%",
                        "mean_predicted": round(mp, 3),
                        "mean_actual": round(ma, 3),
                        "count": counts[i],
                        "gap": round(mp - ma, 3)})
        cal_buckets.append({"bucket": f"{int(mp * 100)}%-{int(mp * 100) + 8}%",
                            "mean_predicted": round(cal_mp, 3),
                            "mean_actual": round(ma, 3),
                            "count": counts[i],
                            "gap": round(cal_mp - ma, 3)})
    return {
        "date": ARTIFACT_DATE,
        "n_games": int(sum(counts)),
        "trained_at": "2026-08-31T01:00:00.000000Z",
        "metrics": {"auc": 0.6911, "brier": 0.2113, "logloss": 0.6329,
                    "ece": 0.0349, "brier_calibrated": 0.2040,
                    "logloss_calibrated": 0.6333, "ece_calibrated": 0.0290},
        "calibration_buckets": buckets,
        "calibration": {
            "method": "platt",
            "params": {"a": 2.5, "b": 0.1, "n": int(sum(counts))},
            "metrics_raw": {"brier": 0.2113, "logloss": 0.6329, "ece": 0.0349},
            "metrics_calibrated": {"brier": 0.2040, "logloss": 0.6333,
                                   "ece": 0.0290},
            "calibration_buckets_calibrated": cal_buckets,
        },
        "daily": [],
    }


def _history_frame(n: int = 320) -> pd.DataFrame:
    """A per-game decided prediction history (same column set the backend
    emits) spanning favored probs 0.51..0.95 so the 1% curve has points, and
    including some upsets (underdog winners) for the summary card."""
    rng = np.random.default_rng(7)
    ps = rng.uniform(0.51, 0.95, n)
    home = [f"H{i % 32:02d}" for i in range(n)]
    away = [f"A{(i + 9) % 32:02d}" for i in range(n)]
    correct = rng.random(n) < ps            # higher prob -> more correct
    pick = home                             # favored side for moneyline
    winner = [home[i] if correct[i] else away[i] for i in range(n)]
    rows = []
    for i in range(n):
        hs = 30 if winner[i] == home[i] else 13
        as_ = 13 if winner[i] == home[i] else 30
        rows.append({
            "game_date": f"2026-09-{i % 20 + 1:02d}",
            "home_team": home[i], "away_team": away[i],
            "home_win_prob_model": round(float(ps[i]), 4),
            # The per-fold PREQUENTIAL column is deliberately a DIFFERENT map
            # from the pooled deployed params (a=2.5, b=0.1): a per-fold map is
            # fitted on limited prior data, so it must NOT coincide with the
            # published one. Keeping them distinct is what lets the smoke test
            # prove the page plots the published quantity.
            "home_win_prob_model_calibrated": round(
                float(1.0 / (1.0 + np.exp(-(1.1 * np.log(ps[i] / (1 - ps[i])) - 0.05)))), 4),
            "away_win_prob_model": round(1.0 - float(ps[i]), 4),
            "correct": bool(correct[i]),
            "model_pick": pick[i],
            "home_score": hs, "away_score": as_,
            "actual_winner": winner[i],
            "game_status": "Final",
            "game_id": f"2026_W1_G{i:03d}",
            "season": 2026, "week": 1,
        })
    return pd.DataFrame(rows)


def _stage(path: Path, data: bytes) -> None:
    """Write a fixture over ``path``, preserving any pre-existing (committed)
    artifact's bytes so cleanup can restore it rather than delete it."""
    if path.exists():
        _BACKUPS[path] = path.read_bytes()
    path.write_bytes(data)
    WRITTEN.append(path)


def _write_artifacts() -> None:
    NFL_DD.mkdir(parents=True, exist_ok=True)
    cal = json.dumps(_calibration_record(), indent=2).encode("utf-8")
    hist = _history_frame().to_csv(index=False).encode("utf-8")
    _stage(CALIBRATION_PATH, cal)
    _stage(HISTORY_PATH, hist)


def _remove_artifacts() -> None:
    for p in WRITTEN:
        try:
            if p in _BACKUPS:
                # A committed artifact existed here — restore it byte-for-byte;
                # never delete real data.
                p.write_bytes(_BACKUPS.pop(p))
            else:
                # This was a fixture this test created fresh -> safe to remove.
                p.unlink()
        except FileNotFoundError:
            pass
    WRITTEN.clear()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "caption", "success", "error",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            try:
                chunks.append(str(el.value))
            except Exception:
                pass
    return "\n".join(chunks)


def _check_rendered_green_curve(spec) -> list[str]:
    """The GREEN line the page actually RENDERS must be the deployed map.

    Unit-testing the helper is not enough: the regression that mattered was the
    PAGE wiring (it plotted the per-fold prequential column instead), so this
    asserts on the chart spec the page actually handed to ``show_chart``,
    compared against the deployed map recomputed from the same fixture.
    """
    import numpy as np
    import moneyline_calibration as mlc

    problems: list[str] = []
    if spec is None:
        return ["chart spec not captured — cannot verify the green curve"]
    if hasattr(spec, "to_dict"):
        spec = spec.to_dict()
    datasets = spec.get("datasets") or {}
    green = []
    for layer in spec.get("layer", []):
        mark = layer.get("mark", {})
        if mark.get("type") == "line" and mark.get("color") == mlc.GREEN:
            green.extend(datasets.get((layer.get("data") or {}).get("name"), []) or [])
    if not green:
        return problems  # no params in fixture -> no green layer; nothing to assert

    params = _calibration_record()["calibration"]["params"]
    # Recompute the expected series from the SAME fixture history the page
    # loads: each point is the mean PUBLISHED probability over the games in
    # that 1% bin, not the map re-evaluated at the bin centre.
    expect_df = mlc.favored_deployed_calibration_pts(
        _history_frame(n=320), params)
    pre_df = mlc.favored_oof_calibration_pts(_history_frame(n=320))
    if expect_df.empty:
        return ["expected deployed series could not be built"]
    expect = {round(float(r["prob"]), 4): float(r["cal_mean"])
              for _, r in expect_df.iterrows()}
    prequential = {round(float(r["prob"]), 4): float(r["cal_mean"])
                   for _, r in pre_df.iterrows()}
    for row in green:
        p = round(float(row["prob"]), 4)
        got = float(row["cal_mean_pct"]) / 100.0
        want = expect.get(p)
        if want is None:
            continue
        if abs(got - want) > 1e-4:
            alt = prequential.get(p)
            hint = ""
            if alt is not None and abs(got - alt) <= 1e-4:
                hint = " — it matches the per-fold PREQUENTIAL column instead"
            problems.append(
                f"rendered green curve at prob={p:.2f} is {got:.4f}, not the "
                f"deployed pooled map {want:.4f}{hint}")
            break
    # The gray low-n bar layer must reach the rendered chart too.
    gray_bars = []
    for layer in spec.get("layer", []):
        mark = layer.get("mark", {})
        if mark.get("type") == "bar" and mark.get("color") == mlc.GRAY:
            gray_bars.extend(datasets.get((layer.get("data") or {}).get("name"), []) or [])
    if not gray_bars:
        problems.append("rendered chart has no gray low-n bar layer")
    return problems


def _check_calibration_curve_grammar() -> list[str]:
    """Pin the two moneyline calibration-curve contracts on real OOF data.

    Both are regressions that silently produced a WRONG-but-plausible chart:
      1. Minimum-evidence rule. A bin with n < LOW_N must contribute NO
         observed-rate curve point (a single decided game at the tail used to
         drag the curve to 0%), while its bar still renders so the volume
         stays visible. Same imported LOW_N as the market-diagnostics charts.
      2. Published probability. The green curve must be the POOLED deployed
         map sigma(a*logit(p)+b) — the quantity the page's own banner quotes —
         NOT the per-fold prequential column, which the MLB three-probability
         contract marks "NEVER display".
    """
    import numpy as np
    import pandas as pd
    import moneyline_calibration as mlc

    problems: list[str] = []
    params = _calibration_record()["calibration"]["params"]

    def _rows(spec, mark_type, color):
        """Rows of every layer matching a mark type+color (named datasets)."""
        out = []
        datasets = spec.get("datasets") or {}
        for layer in spec.get("layer", []):
            mark = layer.get("mark", {})
            if mark.get("type") != mark_type or mark.get("color") != color:
                continue
            ref = (layer.get("data") or {}).get("name")
            out.extend(datasets.get(ref, []) or [])
        return out

    # A frame with a DELIBERATE mix: a dense center bin (n >= LOW_N) and thin
    # 1% bins at the tail whose every game is an UPSET (so their observed rate
    # is 0.0 — exactly the collapse-to-zero the guard must prevent).
    probs = np.concatenate([np.full(200, 0.55), np.full(40, 0.62),
                            np.full(20, 0.70), [0.85]])
    correct = np.concatenate([np.ones(200, bool), np.ones(40, bool),
                              np.ones(20, bool), [False]])
    hist = pd.DataFrame({"home_win_prob_model": probs,
                         "home_win_prob_model_calibrated": probs,
                         "correct": correct})

    # (1) low-n suppression
    pts = mlc.favored_calibration_pts(hist)
    if "low_n" not in pts.columns:
        problems.append("favored_calibration_pts does not expose the low_n flag")
        return problems
    if not pts["low_n"].any() or pts["low_n"].all():
        problems.append("fixture must contain BOTH low-n and high-n bins")
    spec = mlc.chart_favored_calibration(pts, pd.DataFrame())["chart"].to_dict()
    blue_rows = _rows(spec, "line", mlc.BLUE)
    gray_bars = _rows(spec, "bar", mlc.GRAY)
    if any(r.get("low_n") for r in blue_rows):
        problems.append("blue observed curve still plots a low-n bin")
    if any(float(r.get("win_rate_pct", 0)) == 0.0 for r in blue_rows):
        problems.append("blue observed curve contains the n=1 / 0% tail point")
    if not gray_bars:
        problems.append("no gray low-n bar layer — volume context lost")
    if not any(float(r.get("n", 0)) == 1 for r in gray_bars):
        problems.append("the n=1 tail bin is not retained as gray volume")
    # Every high-n bin must still carry a curve point.
    kept = {round(float(r["prob"]), 4) for r in blue_rows}
    for _, row in pts[~pts["low_n"]].iterrows():
        if round(float(row["prob"]), 4) not in kept:
            problems.append(f"high-n bin {row['prob']} lost its curve point")

    # (2) published (pooled) map, not the prequential column
    real = _history_frame(n=4000)
    dep = mlc.favored_deployed_calibration_pts(real, params)
    if dep.empty:
        problems.append("deployed calibration series is empty")
    else:
        # Applied to the pooled params it must equal the closed form applied
        # to the raw row it was built from, and must be monotone in prob.
        a, b = float(params["a"]), float(params["b"])
        p = np.array([0.5, 0.6, 0.75, 0.9])
        expect = mlc.deployed_probability(p, params)
        if not np.allclose(expect, 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))):
            problems.append("deployed_probability does not match sigma(a*logit(p)+b)")
        d = dep.sort_values("prob")["cal_mean"].diff().dropna()
        if (d < -1e-9).any():
            problems.append("deployed curve is not monotone in predicted probability")
        pre = mlc.favored_oof_calibration_pts(real)
        if not pre.empty:
            merged = dep.merge(pre, on="prob", suffixes=("_dep", "_pre"))
            if not merged.empty and np.allclose(merged["cal_mean_dep"],
                                                merged["cal_mean_pre"]):
                problems.append("deployed curve is identical to the prequential "
                                "column — the two maps are not being distinguished")
    # Unusable params must degrade to empty, never to a fabricated curve.
    for bad in (None, {}, {"a": None, "b": 0.1}, {"a": float("nan"), "b": 0.1}):
        if not mlc.favored_deployed_calibration_pts(real, bad).empty:
            problems.append(f"deployed curve fabricated from bad params {bad!r}")
    return problems


def run() -> int:
    _write_artifacts()
    problems: list[str] = []
    try:
        at = AppTest.from_file(str(FRONTEND_DIR / "model_calibration.py"),
                               default_timeout=60)
        at.session_state["sport"] = "nfl"
        # Capture the chart the page actually hands to the renderer, so the
        # curve grammar can be asserted on real output rather than re-derived
        # from the helpers (a page-wiring regression must still be caught).
        import utils as _utils_mod
        _captured: list = []
        _orig_show = _utils_mod.show_chart

        def _capture(chart, *a, **k):
            _captured.append(chart)
            return _orig_show(chart, *a, **k)

        _utils_mod.show_chart = _capture
        try:
            at.run()
        finally:
            _utils_mod.show_chart = _orig_show

        if at.exception:
            problems.append("NFL PAGE RAISED EXCEPTIONS:\n  "
                            + "\n  ".join(str(e.value) for e in at.exception))

        text = _all_text(at)
        vcl = at.get("vega_lite_chart")

        # (1) header pill, record summary card — Today's Record follows the
        #     MLB semantic rule: it counts only the BOARD DATE's decided
        #     games. The NFL 2026 board is scheduled ahead, so the card is
        #     legitimately 0-0 with 'No upsets today' — the lifetime OOF
        #     pool must never masquerade as today's results.
        for key, needle in [("header", "Model Calibration Dashboard"),
                            ("record", "Today's Record:")]:
            if needle not in text:
                problems.append(f"missing [{key}] = {needle!r}")
        if "✓ 0-0" not in text:
            problems.append("NFL summary card not the honest 0-0 board-date record")
        if "No upsets today" not in text:
            problems.append("NFL summary card not the 'No upsets today' empty state")
        if "1,110 completed games" in text or "1107 completed games" in text:
            problems.append("NFL summary card presents OOF history as today's games")

        # (2) four KPI cards
        for key, needle in {"auc": "AUC-ROC", "brier": "BRIER SCORE",
                            "logloss": "LOG-LOSS", "ece": "CAL. ERROR"}.items():
            if needle not in text:
                problems.append(f"missing KPI [{key}] = {needle!r}")

        # (3) post-hoc Platt recalibration banner
        if "Post-Hoc Recalibration" not in text:
            problems.append("missing Platt recalibration banner")

        # (4) the per-1% curve must be a REAL Altair chart (not an info line)
        if len(vcl) == 0:
            problems.append("curve did NOT render an Altair chart (vega_lite_chart=0)")
        if "Calibration Curve" not in text:
            problems.append("missing calibration-curve section")
        if "Per-1% favored-team calibration curve ships when" in text:
            problems.append("curve section degraded (info line) instead of rendering")

        # (5) reliability table with rows + a TOTAL row
        if "Reliability Diagram" not in text:
            problems.append("missing reliability-diagram section")
        if "TOTAL" not in text:
            problems.append("reliability table missing TOTAL row")
        if "BUCKET" not in text:
            problems.append("reliability table missing BUCKET header/rows")

        # (5b) reliability table runs favored-only (>= 50%): the corrected
        #     backend never emits a sub-50% bucket, and the page must not show one.
        if "42%-50%" in text:
            problems.append("reliability table still shows a sub-50% bucket")
        if "52%-60%" not in text:
            problems.append("reliability table missing a >=50% favored bucket")

        # (6) prediction-history table populated (with real rows, not the empty info)
        if "Prediction History" not in text:
            problems.append("missing prediction-history section")
        if "No per-game prediction history" in text:
            problems.append("history table empty/info line instead of populated rows")

        problems.extend(_check_calibration_curve_grammar())
        problems.extend(_check_rendered_green_curve(_captured[-1] if _captured else None))

        if problems:
            print("CALIBRATION SMOKE TEST — FAIL (sport=nfl)")
            for p in problems:
                print("  -", p)
            return 1

        print("CALIBRATION SMOKE TEST — PASS (sport=nfl)")
        n_curves = len(vcl)
        print(f"  - no exceptions; {n_curves} Altair curve chart(s) rendered")
        print("  - record summary + 4 KPIs + Platt banner + reliability table"
              " (w/ TOTAL) + populated history table")

        # sport=mlb must still run the SAME shared path, no exception.
        mlb = AppTest.from_file(str(FRONTEND_DIR / "model_calibration.py"),
                                default_timeout=60)
        mlb.session_state["sport"] = "mlb"
        mlb.run()
        if mlb.exception:
            prob = "\n  ".join(str(e.value) for e in mlb.exception)
            print("CALIBRATION SMOKE TEST — FAIL (sport=mlb)")
            print("  - mlb path raised:\n    " + prob)
            return 1
        print("  - sport=mlb path clean (no exception)")
        return 0
    finally:
        _remove_artifacts()


if __name__ == "__main__":
    sys.exit(run())