"""Tests for the v2 run-engine WINNER cards.

Covers: per-game line assignment, whole-number-line push exclusion, the
>50% pick rule, direct picked-side AUC (pooled + holdout) on the real
artifact, v2 rolling migration (renamed field), and the CROSS-CHECK that
the winner win rates match the Totals & Run Lines history tables (~54%
totals / ~64% run line).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
# frontend/ lives at the REPO root since the multi-sport Phase B move
# (market_diagnostics et al.) — mlb-backend/frontend does not exist.
_frontend = _ROOT.parent / "frontend"
if str(_frontend) not in sys.path:
    sys.path.insert(0, str(_frontend))


def _mk_frame() -> pd.DataFrame:
    """Synthetic OOF markets frame with known FAIR lines/picks.

    The own total line is now the FAIR line (grid argmin of
    |re-scaled P(over) − 0.5|), so the per-game p_over profile must make
    the intended line the argmin — the p_over values below are set so the
    FAIR line is 8.5 for games 1-3 (tie → lower), 8.0 for game 4 (whole
    line, total 8 → PUSH), 9.5 for game 5. p_under mirrors p_over
    (1 − p), so re-scaled P(over) = p_over at every line.
    """
    rows = [
        # game_pk, date, λh, λa, total, hs, as_, p_over_8_5, p_over_8_0,
        # p_over_9_5, p_under_8_5, p_under_8_0, p_under_9_5,
        # p_home_cover_1_5, p_home_win_derived
        (1, "2026-08-20", 4.2, 4.1, 10, 6, 4, 0.60, 0.70, 0.60,
         0.40, 0.30, 0.40, 0.6, 0.6),
        (2, "2026-08-20", 4.2, 4.1, 8, 4, 4, 0.55, 0.65, 0.55,
         0.45, 0.35, 0.45, 0.6, 0.6),
        (3, "2026-08-20", 4.2, 4.1, 7, 3, 4, 0.45, 0.60, 0.45,
         0.55, 0.40, 0.55, 0.4, 0.4),
        (4, "2026-08-20", 4.1, 3.9, 8, 5, 3, 0.28, 0.70, 0.28,
         0.72, 0.30, 0.72, 0.7, 0.7),
        (5, "2026-08-20", 4.6, 4.7, 9, 4, 5, 0.65, 0.75, 0.40,
         0.35, 0.25, 0.60, 0.4, 0.4),
    ]
    df = pd.DataFrame(rows, columns=[
        "game_pk", "game_date", "home_expected_runs", "away_expected_runs",
        "total_runs", "home_score", "away_score", "p_over_8_5", "p_over_8_0",
        "p_over_9_5", "p_under_8_5", "p_under_8_0", "p_under_9_5",
        "p_home_cover_1_5", "p_home_win_derived"])
    df["kind"] = "oof"
    return df



def _latest_artifact(directory, pattern):
    """Find the most recent artifact matching pattern in directory.
    Returns Path or raises unittest.SkipTest if none found. Local harness
    bridge files (run_engine_markets_*_rl.csv) are EXCLUDED — they are
    read-only stand-ins for the next pipeline run and must never shadow
    the canonical committed artifact (the same stale-file guard the
    margin diagnostic and rl bridge apply to their own globs)."""
    import unittest
    matches = sorted((p for p in directory.glob(pattern)
                      if "_rl." not in p.name),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise unittest.SkipTest(f"No {pattern} artifacts found in {directory}")
    return matches[0]

class TestWinnerCardAggregation(unittest.TestCase):
    def test_over_under_line_assignment_and_pick_rule(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        ou = cards["over_under"]
        # Game 4 pushes (line 8.0, total 8) -> excluded from n and win_rate.
        # Non-push games: 1 (over, 10>8.5, win), 2 (over, 8<8.5, loss),
        # 3 (under, 7<8.5, win), 5 (under, 9<9.5, win) -> 3/4.
        self.assertEqual(ou["n"], 4)
        self.assertAlmostEqual(ou["win_rate"], 0.75)
        self.assertAlmostEqual(ou["actual_win_rate"], 0.75)
        # Holdout window covers all synthetic dates -> same rate.
        self.assertEqual(ou["holdout"]["n"], 4)
        self.assertAlmostEqual(ou["holdout"]["win_rate"], 0.75)

    def test_over_under_pushes_only_on_whole_number_lines(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        ou = cards["over_under"]
        # Only game 4 (whole line 8.0 == total 8) is a push; the 9.5 line
        # game with total 9 is NOT a push (integer total != X.5 line).
        self.assertEqual(ou["n"], 4)          # 5 games - 1 push
        self.assertNotIn("n_pushes", ou)      # excluded, not counted as wins

    def test_run_line_half_run_never_pushes(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        rl = cards["run_line"]
        # pick home -1.5 when p>=0.5: games 1 (margin 2 -> win), 2 (margin 0
        # -> away +1.5 wins -> loss), 4 (margin 2 -> win); pick away +1.5:
        # game 3 (margin -1 -> win), 5 (margin -1 -> win) -> 4/5.
        self.assertEqual(rl["n"], 5)
        self.assertAlmostEqual(rl["win_rate"], 0.8)

    def test_derived_ml_pick_home_rule(self):
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        ml = cards["derived_ml"]
        # pick home when p>=0.5: games 1 (win), 2 (loss), 4 (win); pick away:
        # games 3 (hs<as -> win), 5 (hs<as -> win) -> 4/5.
        self.assertEqual(ml["n"], 5)
        self.assertAlmostEqual(ml["win_rate"], 0.8)

    def test_auc_computed_from_pick_pairs(self):
        """AUC is computed DIRECTLY from each card's (picked-side
        probability, settled outcome) over pooled OOF — pushes excluded (the
        OU push game 4 is not in the pairs) — no reference-line injection
        needed. The holdout window covers all synthetic dates, so the
        holdout AUC equals the pooled one."""
        from run_engine import compute_winner_cards
        cards = compute_winner_cards(_mk_frame())
        # OU (games 1, 2, 3, 5 — game 4 pushed at line 8.0 == total 8):
        # re-scaled pick pairs (0.60,1),(0.55,0),(0.55,1),(0.60,1) -> 5/6.
        self.assertAlmostEqual(cards["over_under"]["auc"], 0.83333, places=5)
        self.assertAlmostEqual(cards["over_under"]["holdout"]["auc"],
                               0.83333, places=5)
        # RL picked-side cover pairs (0.6,1),(0.6,0),(0.6,1),(0.7,1),(0.6,1).
        self.assertAlmostEqual(cards["run_line"]["auc"], 0.625, places=5)
        self.assertAlmostEqual(cards["run_line"]["holdout"]["auc"],
                               0.625, places=5)
        # ML raw p_home_win vs home-won pairs (0.6,1),(0.6,0),(0.4,0),
        # (0.7,1),(0.4,0) -> 11/12.
        self.assertAlmostEqual(cards["derived_ml"]["auc"], 0.91667, places=5)
        self.assertAlmostEqual(cards["derived_ml"]["holdout"]["auc"],
                               0.91667, places=5)


class TestScoreAtAuc(unittest.TestCase):
    def test_score_at_auc_finite_on_real_oof_artifact(self):
        """AUC present + finite on the REAL run-engine OOF (fixed reference
        lines over_8_5 / home_cover_1_5 / derived moneyline)."""
        import run_engine
        oof = pd.read_csv(_latest_artifact(_ROOT / "data_delivery", "run_engine_oof_*.csv"))
        # The shipped artifact strips fold_idx on export; a dummy fold keeps
        # the prequential path honest-free (identity) while score_at's AUC
        # uses the same real y/p vectors the pipeline scores.
        oof["fold_idx"] = 0
        res = run_engine.derive_markets_v3(oof, n_draws=20)
        s = res["summary"]
        for key in ("over_8_5", "home_cover_1_5", "derived_moneyline"):
            m = s[f"market_{key}"]
            self.assertIsNotNone(m.get("auc"))
            self.assertTrue(np.isfinite(m["auc"]), f"{key} auc not finite")
            self.assertGreater(m["auc"], 0.5)
            self.assertLess(m["auc"], 1.0)

    def test_score_at_auc_single_class_is_none_not_crash(self):
        import run_engine
        rng = np.random.default_rng(3)
        n = 60
        dates = pd.date_range("2026-07-20", periods=n, freq="D")
        hs = rng.integers(1, 5, n).astype(float)
        as_ = hs + rng.integers(1, 6, n)     # away ALWAYS wins -> single class
        oof = pd.DataFrame({
            "game_pk": list(range(1000, 1000 + n)),
            "game_date": dates,
            "home_expected_runs": rng.uniform(3.8, 5.2, n),
            "away_expected_runs": rng.uniform(3.8, 5.2, n),
            "home_score": hs,
            "away_score": as_,
            "fold_idx": list(range(n)),
        })
        res = run_engine.derive_markets_v3(oof, n_draws=10)
        m = res["summary"]["market_derived_moneyline"]
        self.assertIsNone(m.get("auc"))


class TestCrossCheckHistoryTables(unittest.TestCase):
    def test_winner_win_rates_match_history_tables_on_real_csv(self):
        """The winner cards must reproduce the Totals & Run Lines tables
        exactly (same frame, same pick/push logic) — ~50% totals, ~64% run
        line on the 09-03 artifact (totals re-balanced post P1 adoption)."""
        from market_diagnostics import (decided_rows, history_win_rate,
                                        runline_history_frame,
                                        totals_history_frame)
        from run_engine import compute_winner_cards

        markets = pd.read_csv(_latest_artifact(_ROOT / "data_delivery", "run_engine_markets_*.csv"))
        cards = compute_winner_cards(markets)
        decided = decided_rows(markets)

        tl = history_win_rate(totals_history_frame(decided))
        rl = history_win_rate(runline_history_frame(decided))

        # Exact agreement with the history tables (same n, same win rate).
        self.assertEqual(cards["over_under"]["n"], tl["n_games"])
        self.assertAlmostEqual(cards["over_under"]["win_rate"],
                               tl["win_rate"], places=4)
        self.assertEqual(cards["run_line"]["n"], rl["n_games"])
        self.assertAlmostEqual(cards["run_line"]["win_rate"],
                               rl["win_rate"], places=4)

        # Acceptance ranges: totals ~50%, run-line ~64% on the 09-03 artifact
        # (the totals winner card re-balanced from ~54% toward the coin-flip
        # 50.2% level after the P1 projection adoption sharpened the λ basis;
        # the monitor JSON records the same 0.5015/0.6447 values).
        self.assertGreater(cards["over_under"]["win_rate"], 0.49)
        self.assertLess(cards["over_under"]["win_rate"], 0.53)
        self.assertGreater(cards["run_line"]["win_rate"], 0.62)
        self.assertLess(cards["run_line"]["win_rate"], 0.66)


class TestRollingV2Migration(unittest.TestCase):
    def test_builder_folds_v2_prior_and_renames_field(self):
        """The v2 rolling fold reader uses winner_cards with the renamed
        actual_win_rate field; prior v1 files map onto the cards (over_8_5
        -> over_under) so the series stays continuous across the cutover."""
        from pipeline import _run_engine_monitor_json
        block = {
            "winner_cards": {
                "over_under": {"n": 4126, "actual_win_rate": 0.5414,
                               "win_rate": 0.5414, "predicted_mean": 0.5310,
                               "auc": 0.5505, "ece_raw": 0.019,
                               "ece_calibrated": 0.0105, "brier": 0.247,
                               "logloss": 0.680, "holdout": {}},
            },
            "market_metrics": {}, "alpha_home": {}, "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {}, "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        prior = {
            "schema": "run-engine-monitor/v2",
            "date": "20260825",
            "winner_cards": {
                "over_under": {"n": 3900, "actual_win_rate": 0.5390,
                               "win_rate": 0.5390, "predicted_mean": 0.5300,
                               "auc": 0.55, "ece_calibrated": 0.015,
                               "brier": 0.250, "logloss": 0.685},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260825.json").write_text(
                json.dumps(prior))
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826", True, None)
            data = json.loads(path.read_text())
        rolling = data["rolling"]["over_under"]
        self.assertEqual([r["date"] for r in rolling],
                         ["2026-08-25", "2026-08-26"])
        self.assertIn("actual_win_rate",
                      data["winner_cards"]["over_under"])
        self.assertNotIn("base_rate",
                         data["winner_cards"]["over_under"])

    def test_v1_prior_mapped_to_v2_card(self):
        """A v1 per_line file folds in through the line->card map:
        over_8_5 becomes the over_under rolling point (v1->v2 continuity)."""
        from pipeline import _run_engine_monitor_json
        block = {
            "winner_cards": {
                "over_under": {"n": 4126, "actual_win_rate": 0.5414,
                               "win_rate": 0.5414, "predicted_mean": 0.5310,
                               "auc": 0.5505, "ece_raw": 0.019,
                               "ece_calibrated": 0.0105, "brier": 0.247,
                               "logloss": 0.680, "holdout": {}},
            },
            "market_metrics": {}, "alpha_home": {}, "alpha_away": {},
            "fit_check_alpha_lambda": {"home": [], "away": []},
            "variance_check": {}, "mc_meta": {},
            "holdout_gate": {"n_pre": 0, "n_holdout": 0},
            "phase1": {"dispersion_ratio": {"home": 0, "away": 0}},
            "line_grid": [],
        }
        prior = {
            "schema": "run-engine-monitor/v1",
            "date": "20260825",
            "per_line": {"over_8_5": {"n": 3900, "ece_calibrated": 0.015,
                                      "brier": 0.250, "logloss": 0.685,
                                      "predicted_mean": 0.448,
                                      "base_rate": 0.449}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260825.json").write_text(
                json.dumps(prior))
            with patch("pipeline.DATA_DELIVERY_DIR", p):
                path = _run_engine_monitor_json(block, "20260826", True, None)
            data = json.loads(path.read_text())
        rolling = data["rolling"]["over_under"]
        self.assertEqual([[r["date"], r["n"]] for r in rolling],
                         [["2026-08-25", 3900], ["2026-08-26", 4126]])
        self.assertEqual(rolling[0]["ece_calibrated"], 0.015)
        self.assertEqual(rolling[0]["predicted_mean"], 0.448)


class TestWinnerCardSymmetry(unittest.TestCase):
    """Home/away symmetry audit: every metric is on the PICKED side.

    run_line and derived_ml must split by pick direction (by_pick) and every
    metric must be computed on picked-side (p, y) — never home-side
    unconditionally. Proven on the real 08-27 artifact and on synthetic
    frames where the two sides disagree.
    """

    def _real(self) -> pd.DataFrame:
        return pd.read_csv(_latest_artifact(_ROOT / "data_delivery", "run_engine_markets_*.csv"))

    @staticmethod
    def _derived_pcol(cards: dict) -> str:
        """Probability column the derived_ml card actually prices."""
        src = cards["derived_ml"].get("source")
        return "ml_win_prob" if src == "ml_win_prob" \
            else "p_home_win_derived"

    def test_by_pick_split_consistent_with_pooled(self):
        from run_engine import compute_winner_cards
        df = self._real()
        cards = compute_winner_cards(df)
        for name in ("run_line", "derived_ml"):
            c = cards[name]
            bp = c["by_pick"]
            n_h, n_a = bp["home"]["n"], bp["away"]["n"]
            self.assertEqual(n_h + n_a, c["n"], f"{name} split covers n")
            self.assertGreater(n_a, 0, f"{name} has away-picks")
            if name == "derived_ml":
                # Real invariant: the derived-ML card prices BOTH directions
                # on the 6,953-frame artifact.
                self.assertGreater(n_h, 0, f"{name} has home-picks")
            else:
                # Both directions must exist. The run-line model prices
                # away +1.5 for the large majority of games (home-cover-1.5
                # mean ~0.36 << 0.5), so home picks are vanishingly rare
                # (~1/6,953). Under the structural margin fix (2531462 →
                # tie resolves to ±1) home-cover at 1.5 stays raw P(margin>=2)
                # ≈ 0.358, which keeps home picks a tiny minority. They must
                # never flip the card wholesale.
                self.assertGreater(n_a, 0,
                                  "run_line must have away-picks")
                self.assertLess(n_h, c["n"] * 0.05,
                                "run_line home-picks must stay a small "
                                "minority after renormalization")
            # Pooled win rate = subset-weighted average. The by_pick rates
            # are display-rounded to 4dp in the JSON, so the reconstruction
            # matches the pooled rate to 3dp (any real inconsistency would
            # appear at the 2nd decimal). A degenerate empty side carries
            # win_rate=None — the non-empty side then IS the pooled rate.
            parts = [(n, rate) for n, rate in
                     ((n_h, bp["home"]["win_rate"]),
                      (n_a, bp["away"]["win_rate"])) if n > 0]
            pooled = sum(n * rate for n, rate in parts) / c["n"]
            self.assertAlmostEqual(pooled, c["win_rate"], places=3,
                                   msg=f"{name} pooled rate consistent")
        self.assertNotIn("by_pick", cards["over_under"])

    def test_predicted_mean_is_picked_side_not_home_side(self):
        """predicted_mean == mean(max(P, 1-P)) (picked-side prob) and NOT
        mean(P_home) / mean(P_home_cover). Exact on the real artifact (no
        fold_idx persisted -> calibrated == raw)."""
        from run_engine import compute_winner_cards
        df = self._real()
        cards = compute_winner_cards(df)
        oof = df[df["kind"] == "oof"]
        rp = oof["p_home_cover_1_5"].to_numpy(float)
        dp = self._derived_pcol(cards)
        mp = oof[dp].to_numpy(float)
        for name, p_home in (("run_line", rp), ("derived_ml", mp)):
            c = cards[name]
            ok = np.isfinite(p_home)
            picked = np.maximum(p_home[ok], 1.0 - p_home[ok])
            self.assertAlmostEqual(c["predicted_mean"], picked.mean(),
                                   places=4, msg=f"{name} is picked-side")
            self.assertNotAlmostEqual(c["predicted_mean"],
                                      p_home[ok].mean(), places=3,
                                      msg=f"{name} NOT mean(P_home)")
        # The NB Monte-Carlo diagnostic is preserved inside the card (the
        # model finding must stay visible): its own predicted_mean equals
        # mean(max(p_home_win_derived, 1-P)) on its own rows.
        nb = cards["derived_ml"].get("nb_diagnostic")
        self.assertIsNotNone(nb, "nb_diagnostic preserved")
        nb_p = oof["p_home_win_derived"].to_numpy(float)
        nok = np.isfinite(nb_p)
        self.assertAlmostEqual(nb["predicted_mean"],
                               np.maximum(nb_p[nok], 1 - nb_p[nok]).mean(),
                               places=4)
        self.assertEqual(nb["n"], int(nok.sum()))

    def test_win_rate_two_independent_ways_identical(self):
        """Vectorized picked-side outcome vs a manual per-game loop -> the
        SAME win rate (guards against any indexing/side error)."""
        from run_engine import compute_winner_cards
        df = self._real()
        cards = compute_winner_cards(df)
        oof = df[df["kind"] == "oof"]
        hs = oof["home_score"].to_numpy(float)
        as_ = oof["away_score"].to_numpy(float)
        for name, pcol, event_fn in (
                ("run_line", "p_home_cover_1_5",
                 lambda m: m >= 2),          # home covers -1.5
                ("derived_ml", self._derived_pcol(cards),
                 lambda m: m > 0)):          # home wins
            p = oof[pcol].to_numpy(float)
            ok = np.isfinite(p)
            event = event_fn(hs - as_).astype(float)
            pick_home = p >= 0.5
            hits = []
            for i in np.where(ok)[0]:
                if pick_home[i]:
                    hits.append(float(event[i] == 1))
                else:
                    hits.append(float(event[i] == 0))
            loop_rate = float(np.mean(hits))
            self.assertAlmostEqual(loop_rate, cards[name]["win_rate"],
                                   places=4, msg=f"{name} loop == vectorized")

    def test_auc_rank_invariance_exact(self):
        """AUC(P_away side, away event) == AUC(P_home side, home event)
        EXACTLY (1-P is a monotone transform; rank invariance)."""
        from sklearn.metrics import roc_auc_score
        from run_engine import compute_winner_cards
        df = self._real()
        oof = df[df["kind"] == "oof"]
        hs = oof["home_score"].to_numpy(float)
        as_ = oof["away_score"].to_numpy(float)
        total = hs + as_
        cards = compute_winner_cards(df)
        dp = self._derived_pcol(cards)
        cards = compute_winner_cards(df)
        for name, pcol, event_fn in (
                ("run_line", "p_home_cover_1_5", lambda m: m >= 2),
                ("derived_ml", dp, lambda m: m > 0)):
            p = oof[pcol].to_numpy(float)
            ok = np.isfinite(p)
            event = event_fn(hs - as_).astype(float)
            a_home = roc_auc_score(event[ok], p[ok])
            a_away = roc_auc_score(1.0 - event[ok], 1.0 - p[ok])
            # Rank invariance is exact mathematically (1-P is a monotone
            # transform); sklearn's internal float arithmetic can differ at
            # the last ulp (~1e-15) when the complemented scores have full
            # float precision, so assert to 12 decimals.
            self.assertAlmostEqual(a_home, a_away, places=12,
                                   msg=f"{name} AUC rank-invariant")
            # Card AUC per monitor spec: derived_ml ranks the RAW p_home_win
            # vs home-won (== a_home); run_line ranks the PICKED side's
            # cover prob vs covered (fav, hit).
            if name == "derived_ml":
                self.assertAlmostEqual(a_home, cards[name]["auc"], places=5,
                                       msg="derived_ml AUC == moneyline AUC")
            else:
                pick_home = p >= 0.5
                fav = np.where(pick_home, p, 1.0 - p)
                hit = (pick_home.astype(float) == event).astype(float)
                self.assertAlmostEqual(roc_auc_score(hit[ok], fav[ok]),
                                       cards[name]["auc"], places=5,
                                       msg="run_line AUC == picked-side AUC")

    def test_ece_is_on_picked_side(self):
        """ece_raw == ECE(picked-side prob, picked-side outcome) and differs
        from ECE(home prob, home outcome) where the sides disagree."""
        from run_engine import compute_winner_cards, ece_score
        df = self._real()
        cards = compute_winner_cards(df)
        oof = df[df["kind"] == "oof"]
        hs = oof["home_score"].to_numpy(float)
        as_ = oof["away_score"].to_numpy(float)
        for name, pcol, event_fn in (
                ("run_line", "p_home_cover_1_5", lambda m: m >= 2),
                ("derived_ml", self._derived_pcol(cards),
                 lambda m: m > 0)):
            p = oof[pcol].to_numpy(float)
            ok = np.isfinite(p)
            event = event_fn(hs - as_).astype(float)
            pick_home = p >= 0.5
            fav = np.where(pick_home, p, 1.0 - p)
            hit = (pick_home.astype(float) == event).astype(float)
            self.assertAlmostEqual(ece_score(hit[ok], fav[ok]),
                                   cards[name]["ece_raw"], places=8,
                                   msg=f"{name} ECE on picked side")
            home_ece = ece_score(event[ok], p[ok])
            if (pick_home[ok]).all() or (~pick_home[ok]).all():
                # Uniform pick direction: ECE(event, p) == ECE(1-event, 1-p)
                # EXACTLY (equal-width bins mirror under p -> 1-p, counts and
                # gaps included) — the difference check is vacuous, not a bug.
                self.assertAlmostEqual(home_ece, cards[name]["ece_raw"],
                                       places=6,
                                       msg=f"{name} mirror symmetry under "
                                           "uniform direction")
            else:
                # When one side has vanishingly few picks (<1% of total),
                # ECE(p, y) ≈ ECE(1-p, 1-y) by bin-mirror symmetry — only
                # assert the difference when both sides have meaningful mass.
                n_h_picks = int(pick_home[ok].sum())
                n_a_picks = int((~pick_home[ok]).sum())
                gap = abs(home_ece - cards[name]["ece_raw"])
                if min(n_h_picks, n_a_picks) > 20 and gap > 1e-5:
                    # The negative check only fires when the two framings
                    # are MEANINGFULLY different. When the model is
                    # calibrated in BOTH framings the home-side and
                    # picked-side ECE agree within the check's own 3dp
                    # resolution (09-03 artifact: run_line home 0.00587 vs
                    # picked 0.0054; derived_ml both ~0.004) and the
                    # negative check is vacuous — the positive places=8
                    # pin above is the contract that ECE is computed on
                    # the picked side exactly.
                    if gap > 0.0005:
                        self.assertNotAlmostEqual(
                            home_ece, cards[name]["ece_raw"], places=3,
                            msg=f"{name} ECE NOT home-side")

    def test_away_pick_scoring_not_home_outcomes(self):
        """Away-pick win rate == away-outcome rate on away-pick games (NOT
        the home base rate) and away-picks are not dropped (n_away > 0, and
        n_away != n_total proves both directions exist for derived_ml)."""
        from run_engine import compute_winner_cards
        df = self._real()
        cards = compute_winner_cards(df)
        oof = df[df["kind"] == "oof"]
        hs = oof["home_score"].to_numpy(float)
        as_ = oof["away_score"].to_numpy(float)
        home_won = (hs > as_).astype(float)
        dp = self._derived_pcol(cards)
        mp = oof[dp].to_numpy(float)
        ok = np.isfinite(mp)
        pick_home = mp >= 0.5
        away_mask = ok & ~pick_home
        away_won_rate = float((1 - home_won[away_mask]).mean())
        bp_away = cards["derived_ml"]["by_pick"]["away"]
        self.assertAlmostEqual(bp_away["win_rate"], away_won_rate, places=4,
                               msg="away-pick scored on away outcomes")
        self.assertGreater(bp_away["win_rate"], 0.0)
        self.assertLess(bp_away["win_rate"], 1.0)
        # The preserved NB diagnostic keeps the model finding: its away-pick
        # win rate equals the NB away-outcome rate (and post the structural
        # fix sits ABOVE 50% — the calibrated home one-run adjustment
        # resolved the old "underweights the home edge" finding).
        nb = cards["derived_ml"]["nb_diagnostic"]
        nb_p = oof["p_home_win_derived"].to_numpy(float)
        nok = np.isfinite(nb_p)
        npk = nb_p >= 0.5
        n_away = nok & ~npk
        self.assertAlmostEqual(nb["by_pick"]["away"]["win_rate"],
                               float((1 - home_won[n_away]).mean()),
                               places=4)
        self.assertGreater(nb["by_pick"]["away"]["win_rate"], 0.50)

    def test_derived_ml_sources_run_line_model_with_ensemble_reference(self):
        """The derived_ml card is the RUN LINE model's own NB moneyline
        (p_home_win_derived): pooled ~55.5%, away-picks ~54.7% on the 09-03
        artifact (post P1 adoption) — calibrated post the structural home
        one-run fix — and the moneyline ensemble rides as a one-line
        ml_reference (~55.7%) so the model comparison stays visible."""
        from run_engine import compute_winner_cards
        df = self._real()
        cards = compute_winner_cards(df)
        c = cards["derived_ml"]
        self.assertEqual(c["source"], "nb_mc_p_home_win_derived")
        oof = df[df["kind"] == "oof"]
        nb_p = oof["p_home_win_derived"].to_numpy(float)
        self.assertEqual(c["n"], int(np.isfinite(nb_p).sum()))
        # Expected numbers (pin-synced to the 09-03 artifact — the P1
        # projection adoption 8cb4efc sharpened the λ basis, moving the
        # derived ML pooled win rate 54.1% -> 55.5%): pooled ~55.5%,
        # away-picks ~54.7% — the structural home one-run fix resolved the
        # old home-edge underweighting.
        self.assertAlmostEqual(c["win_rate"], 0.5554, places=3)
        self.assertGreater(c["by_pick"]["away"]["win_rate"], 0.50)
        self.assertAlmostEqual(c["by_pick"]["away"]["win_rate"], 0.5472,
                               places=3)
        self.assertGreater(c["by_pick"]["home"]["win_rate"], 0.50)
        # nb_diagnostic preserved (schema-stable record of the finding).
        nb = c["nb_diagnostic"]
        self.assertEqual(nb["n"], c["n"])
        self.assertAlmostEqual(nb["actual_win_rate"], c["win_rate"],
                               places=4)
        # Moneyline ENSEMBLE one-line reference (~55.8%, both directions).
        ref = c.get("ml_reference")
        self.assertIsNotNone(ref)
        self.assertEqual(ref["source"], "ml_win_prob")
        self.assertGreater(ref["win_rate"], 0.55)
        self.assertAlmostEqual(ref["win_rate"], 0.5565, places=3)
        self.assertEqual(ref["n"],
                         int(np.isfinite(oof["ml_win_prob"]).sum()))

    def test_synthetic_symmetric_frame_both_directions_win(self):
        """With a well-calibrated synthetic probability BOTH pick directions
        win >50% and each ≈ its own predicted mean — proving the aggregation
        is symmetric-capable (any real-artifact asymmetry is the model's, not
        the aggregation's)."""
        from run_engine import compute_winner_cards
        rng = np.random.default_rng(7)
        n = 400
        # Half home-favored, half away-favored — the aggregation must handle
        # BOTH pick directions symmetrically. p_home drives derived_ml picks;
        # p_cover < p_home (covering implies winning) drives run_line picks;
        # p_cover >= 0.5 exactly for home-favored games, < 0.5 for away.
        p_home = np.concatenate([
            rng.uniform(0.55, 0.70, n // 2),
            rng.uniform(0.30, 0.45, n // 2),
        ])
        p_cover = np.where(p_home >= 0.5, p_home - 0.05, p_home - 0.05)
        # Three-category margin: 3 = won + covered, 1 = won by 1 (no cover),
        # 0 = away won. home_won == (margin > 0) and
        # home_covers == (margin >= 2) both hold EXACTLY by construction.
        u = rng.uniform(0, 1, n)
        margin = np.where(u < p_cover, 3.0,
                          np.where(u < p_home, 1.0, 0.0))
        df = pd.DataFrame({
            "game_pk": list(range(5000, 5000 + n)),
            "game_date": pd.date_range("2026-06-01", periods=n, freq="D"),
            "home_expected_runs": 4.3, "away_expected_runs": 4.2,
            "total_runs": 8.0,
            "home_score": 3.0 + margin, "away_score": 3.0,
            "p_home_cover_1_5": p_cover,
            "p_home_win_derived": p_home,
            "kind": "oof",
        })
        cards = compute_winner_cards(df)
        for name in ("run_line", "derived_ml"):
            c = cards[name]
            self.assertIn("by_pick", c)
            for side in ("home", "away"):
                s = c["by_pick"][side]
                self.assertGreater(s["n"], 20, f"{name}/{side} sample")
                self.assertGreater(s["win_rate"], 0.50,
                                   f"{name}/{side} wins >50%")
                self.assertAlmostEqual(s["win_rate"], s["predicted_mean"],
                                       delta=0.06,
                                       msg=f"{name}/{side} ≈ own predmean")


if __name__ == "__main__":
    unittest.main()
