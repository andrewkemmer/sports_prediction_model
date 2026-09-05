"""Tests for the run-engine monitor artifact (pipeline + run_engine), v2.

Covers: JSON schema (three winner cards + rolling + fit + persisted flag),
persist-failure path returns flags without crashing, _run_engine_monitor_json
produces valid JSON, v2 rolling history folding (renamed field), and
guardrails.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _winner_card(auc=None, n: int = 4000) -> dict:
    return {
        "n": n,
        "actual_win_rate": 0.5414,
        "win_rate": 0.5414,
        "predicted_mean": 0.5310,
        "auc": auc,
        "ece_raw": 0.019,
        "ece_calibrated": 0.014,
        "brier": 0.247,
        "logloss": 0.680,
        "logloss_calibrated": 0.670,
        "beats_baseline_logloss": True,
        "holdout": {"n": 275, "actual_win_rate": 0.5891, "win_rate": 0.5891,
                    "predicted_mean": 0.5600, "ece_raw": 0.05,
                    "ece_calibrated": 0.048, "brier": 0.26, "logloss": 0.70,
                    "baseline_rate": 0.44, "baseline_logloss": 0.693,
                    "beats_baseline_logloss": True},
    }


def _minimal_block() -> dict:
    """Synthetic monitor_block with the v2 winner-cards shape."""
    metrics = {
        "over_8_5": {"auc": 0.5505, "n": 4369},
        "home_cover_1_5": {"auc": 0.5432, "n": 4369},
        "derived_moneyline": {"auc": 0.5545, "n": 4369},
    }
    return {
        "winner_cards": {
            "over_under": _winner_card(auc=0.5505),
            "run_line": _winner_card(auc=0.5432),
            "derived_ml": _winner_card(auc=0.5545),
        },
        "market_metrics": metrics,
        # Fit block in the REAL run-engine schema (mirrors run_engine_daily's
        # monitor-embed dict and data_delivery/run_engine_monitor_*.json):
        # alpha curves with selection bins, fit-check tables with
        # observed_p/modeled_p + unicode tail labels, per-side variance
        # implied/observed, mc_meta.n_draws.
        "alpha_home": {
            "form": "piecewise", "direction": "rising",
            "lam": [4.1, 4.4, 4.8], "alpha": [0.22, 0.27, 0.28],
            "selection": {"chosen": "piecewise", "bins": [
                {"count": 573, "mean_lam": 4.1577, "alpha": 0.2165},
                {"count": 590, "mean_lam": 4.3623, "alpha": 0.2686},
                {"count": 580, "mean_lam": 4.4264, "alpha": 0.3113}]},
            "fitted_on": "pre-holdout OOF only", "cap": 2.0,
            "min_bin_count": 250,
        },
        "alpha_away": {
            "form": "linear", "a": 0.6218, "b": -0.0633,
            "selection": {"chosen": "linear", "bins": [
                {"count": 582, "mean_lam": 3.8069, "alpha": 0.4196},
                {"count": 581, "mean_lam": 4.1625, "alpha": 0.3708},
                {"count": 581, "mean_lam": 4.2918, "alpha": 0.3522}]},
            "fitted_on": "pre-holdout OOF only", "cap": 2.0,
            "min_bin_count": 250,
        },
        "fit_check_alpha_lambda": {
            "home": [{"k": 0, "observed_p": 0.0597, "modeled_p": 0.0534},
                     {"k": 1, "observed_p": 0.108, "modeled_p": 0.108},
                     {"k": "≤1", "observed_p": 0.1678,
                      "modeled_p": 0.1613},
                     {"k": "≥10", "observed_p": 0.0691,
                      "modeled_p": 0.0733}],
            "away": [{"k": 0, "observed_p": 0.0732, "modeled_p": 0.0712},
                     {"k": 1, "observed_p": 0.1176, "modeled_p": 0.1224},
                     {"k": "≤1", "observed_p": 0.1909,
                      "modeled_p": 0.1936},
                     {"k": "≥10", "observed_p": 0.0838,
                      "modeled_p": 0.078}],
        },
        "variance_check": {
            "home": {"implied_var": 9.938, "observed_var": 9.886,
                      "phase2_implied_var": 9.992},
            "away": {"implied_var": 10.902, "observed_var": 11.03,
                      "phase2_implied_var": 11.054},
        },
        "mc_meta": {"n_draws": 10000, "requested_draws": 10000,
                    "reason": "default", "mc_se_totals_max": 0.005},
        "holdout_gate": {"n_pre": 4000, "n_holdout": 280},
        "phase1": {"dispersion_ratio": {"home": 2.3, "away": 2.1}},
        "line_grid": [7.5, 8.0, 8.5, 9.0, 9.5],
    }


def _write_monitor(block: dict, date_str: str, tmp: Path,
                   persisted: bool = True, error=None):
    from pipeline import _run_engine_monitor_json
    with patch("pipeline.DATA_DELIVERY_DIR", tmp):
        path = _run_engine_monitor_json(block, date_str, persisted, error)
    return json.loads(path.read_text())



def _latest_artifact(directory, pattern):
    """Find the most recent artifact matching pattern in directory.
    Returns Path or raises unittest.SkipTest if none found."""
    import unittest
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise unittest.SkipTest(f"No {pattern} artifacts found in {directory}")
    return matches[0]

class TestMonitorSchema(unittest.TestCase):
    """The JSON written by _run_engine_monitor_json has the v2 shape."""

    def test_schema_has_all_top_level_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        self.assertEqual(data["schema"], "run-engine-monitor/v2")
        self.assertEqual(data["date"], "20260826")
        self.assertTrue(data["markets_persisted"])
        self.assertIsNone(data["markets_persist_error"])
        self.assertIn("winner_cards", data)
        self.assertIn("rolling", data)
        self.assertIn("fit", data)

    def test_winner_cards_has_exactly_three_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        self.assertEqual(set(data["winner_cards"].keys()),
                         {"over_under", "run_line", "derived_ml"})

    def test_winner_card_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        card = data["winner_cards"]["over_under"]
        for key in ("n", "actual_win_rate", "win_rate", "predicted_mean",
                    "auc", "ece_raw", "ece_calibrated", "brier", "logloss"):
            self.assertIn(key, card, f"missing winner-card field: {key}")
        h = card["holdout"]
        for key in ("n", "actual_win_rate", "win_rate", "ece_raw",
                    "ece_calibrated", "brier", "logloss"):
            self.assertIn(key, h, f"missing holdout field: {key}")

    def test_auc_passthrough_from_reference_line(self):
        """auc rides through from the winner card (already attached from the
        fixed reference line by run_engine.compute_winner_cards)."""
        block = _minimal_block()
        block["winner_cards"]["over_under"]["auc"] = 0.55051
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(block, "20260826", Path(tmp))
        self.assertAlmostEqual(
            data["winner_cards"]["over_under"]["auc"], 0.55051)

    def test_fit_has_required_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        fit = data["fit"]
        for key in ("alpha_home", "alpha_away", "dispersion_chi2_per_df",
                    "fit_tables", "variance_check", "mc_meta"):
            self.assertIn(key, fit)
        self.assertIn("home", fit["dispersion_chi2_per_df"])
        self.assertIn("away", fit["dispersion_chi2_per_df"])
        self.assertIn("home", fit["fit_tables"])
        self.assertIn("away", fit["fit_tables"])

    def test_rolling_starts_with_current_when_no_prior(self):
        """With a non-empty block and no prior monitor files, each card has
        exactly 1 entry (today)."""
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(_minimal_block(), "20260826", Path(tmp))
        rolling = data["rolling"]
        self.assertIsInstance(rolling, dict)
        for key in ("over_under", "run_line", "derived_ml"):
            self.assertIn(key, rolling)
            self.assertEqual(len(rolling[key]), 1)
            self.assertEqual(rolling[key][0]["date"], "2026-08-26")

    def test_empty_block_is_graceful(self):
        """A block with no winner cards still writes a valid v2 file."""
        block = _minimal_block()
        block["winner_cards"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(block, "20260826", Path(tmp))
        self.assertEqual(data["schema"], "run-engine-monitor/v2")
        self.assertEqual(data["winner_cards"], {})
        for key in ("over_under", "run_line", "derived_ml"):
            self.assertEqual(data["rolling"][key], [])


class TestPersistFailurePath(unittest.TestCase):
    """markets_persisted=False is reflected in the monitor JSON."""

    def test_persist_failure_flagged(self):
        block = _minimal_block()
        block["winner_cards"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_monitor(
                block, "20260826", Path(tmp), persisted=False,
                error="ValueError: markets frame contains NaNs")
        self.assertFalse(data["markets_persisted"])
        self.assertIn("NaNs", data["markets_persist_error"])


class TestRollingHistoryFolding(unittest.TestCase):
    """Prior v2 monitor files are folded into the rolling series."""

    def test_prior_v2_monitors_folded_with_renamed_field(self):
        block = _minimal_block()
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
            data = _write_monitor(block, "20260826", p)
        rolling = data["rolling"]["over_under"]
        self.assertEqual(len(rolling), 2)  # prior + today
        self.assertEqual(rolling[0]["date"], "2026-08-25")
        self.assertEqual(rolling[0]["n"], 3900)
        self.assertEqual(rolling[0]["ece_calibrated"], 0.015)
        self.assertEqual(rolling[1]["date"], "2026-08-26")

    def test_v1_prior_files_mapped_into_rolling(self):
        """A v1 per_line file maps onto the v2 cards (over_8_5 -> over_under,
        home_cover_1_5 -> run_line, derived_moneyline -> derived_ml) so the
        rolling history stays continuous across the cutover; the base_rate
        field is simply not carried (rolling points never carried it)."""
        block = _minimal_block()
        prior = {
            "schema": "run-engine-monitor/v1",
            "date": "20260825",
            "per_line": {
                "over_8_5": {"n": 3900, "ece_calibrated": 0.015,
                             "brier": 0.250, "logloss": 0.685,
                             "predicted_mean": 0.448, "base_rate": 0.449},
                "home_cover_1_5": {"n": 3901, "ece_calibrated": 0.016,
                                   "brier": 0.251, "logloss": 0.686,
                                   "predicted_mean": 0.449,
                                   "base_rate": 0.450},
                "derived_moneyline": {"n": 3902, "ece_calibrated": 0.017,
                                      "brier": 0.252, "logloss": 0.687,
                                      "predicted_mean": 0.450,
                                      "base_rate": 0.451},
                # Unmappable v1 line (not a winner card) -> skipped, no crash.
                "over_7_5": {"n": 3800, "ece_calibrated": 0.02},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260825.json").write_text(
                json.dumps(prior))
            data = _write_monitor(block, "20260826", p)
        for card, n_prior in (("over_under", 3900), ("run_line", 3901),
                              ("derived_ml", 3902)):
            rolling = data["rolling"][card]
            self.assertEqual(len(rolling), 2, f"{card}: prior + today")
            self.assertEqual(rolling[0]["date"], "2026-08-25")
            self.assertEqual(rolling[0]["n"], n_prior)
            self.assertEqual(rolling[1]["date"], "2026-08-26")

    def test_v1_v2_real_fixtures_fold_continuous(self):
        """The REAL v1 monitor fixture + the REAL locally-built v2 monitor
        fold into a continuous per-card series (v1 08-26 -> v2 08-27 ->
        today) — the migration path the dashboard renders through."""
        _here = Path(__file__).resolve().parent
        v1_fixture = _here / "fixtures" / "run_engine_monitor_v1_20260826.json"
        v2_accept = _here.parent / "data_delivery" \
            / "run_engine_monitor_latest.json"
        if not v1_fixture.exists() or not v2_accept.exists():
            self.skipTest("v1 fixture / v2 acceptance artifact missing")
        block = _minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "run_engine_monitor_20260826.json").write_text(
                v1_fixture.read_text())
            (p / "run_engine_monitor_latest.json").write_text(
                v2_accept.read_text())
            data = _write_monitor(block, "20260828", p)
        for card in ("over_under", "run_line", "derived_ml"):
            rolling = data["rolling"][card]
            dates = [r["date"] for r in rolling]
            self.assertEqual(dates, ["2026-08-26", "2026-08-27",
                                     "2026-08-28"], f"{card} continuous")
            self.assertIsInstance(rolling[0]["n"], int)  # v1-mapped point
            self.assertIsInstance(rolling[1]["n"], int)  # v2 point

    def test_rolling_trims_to_45_days(self):
        from datetime import date, timedelta
        block = _minimal_block()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(50):
                dt = date(2026, 8, 26) - timedelta(days=50 - i)
                d = dt.strftime("%Y%m%d")
                prior = {
                    "schema": "run-engine-monitor/v2",
                    "date": d,
                    "winner_cards": {
                        "over_under": {"n": 3900, "actual_win_rate": 0.5390,
                                       "win_rate": 0.5390,
                                       "predicted_mean": 0.5300, "auc": 0.55,
                                       "ece_calibrated": 0.015,
                                       "brier": 0.250, "logloss": 0.685},
                    },
                }
                (p / f"run_engine_monitor_{d}.json").write_text(
                    json.dumps(prior))
            data = _write_monitor(block, "20260826", p)
        rolling = data["rolling"]["over_under"]
        self.assertLessEqual(len(rolling), 46)  # 45 prior + 1 today max


class TestRunEngineModelMonitor(unittest.TestCase):
    """Run-engine model monitor additions: per-line market_metrics + phase1
    geometry in the monitor JSON, and the run_engine_feature_{drift,
    coverage} artifacts over the run engine's OWN 29 kept features (the 36
    dropped — diff/momentum/moneyline-only, incl. run_margin_diff — are
    excluded).
    """

    _ROOT = Path(__file__).resolve().parents[1]

    @staticmethod
    def _windows(date_str: str = None):
        """The pipeline drift step's exact window slicing (last 7 days vs an
        adjacent season-local window of ~3x, min 250) on the canonical
        decided frame."""
        gl = pd.read_csv(TestRunEngineModelMonitor._ROOT
                         / "data_delivery" / "game_level_features.csv")
        # The pipeline's drift/coverage frame is the CANONICAL decided frame
        # (frames.get_decided_frame: stable mergesort order, game_pk
        # normalized to int64) — NOT a raw quicksort re-derivation.  The two
        # orders disagree on the tail(276) baseline boundary by a couple of
        # games (e.g. wind_advantage_flyball_factor default-zero 48 vs 50),
        # which broke the coverage-parity invariant below against the
        # committed artifacts.
        from frames import get_decided_frame
        decided = get_decided_frame(gl)
        gd = pd.to_datetime(decided["game_date"])
        if date_str is None:
            # Extract date from the most recent monitor file
            latest = _latest_artifact(
                TestRunEngineModelMonitor._ROOT / "data_delivery",
                "run_engine_monitor_*.json")
            import json as _j
            date_str = _j.loads(latest.read_text())["date"]
        cutoff = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-"
                              f"{date_str[6:8]}") - pd.Timedelta(days=7)
        current = decided[gd >= cutoff]
        prior = decided[gd < cutoff]
        baseline = (prior.tail(max(3 * len(current), 250))
                    if not prior.empty else prior)
        return baseline, current

    def test_market_metrics_block_present_and_finite(self):
        data = json.loads(_latest_artifact(self._ROOT / "data_delivery",
                                           "run_engine_monitor_*.json").read_text())
        mm = data.get("market_metrics") or {}
        self.assertEqual(
            set(mm.keys()),
            {"over_7_5", "over_8_5", "over_9_5", "home_cover_1_5",
             "home_cover_2_5", "derived_moneyline"})
        for name, row in mm.items():
            for key in ("engine_logloss", "engine_brier", "engine_ece_raw",
                        "engine_ece_calibrated", "n"):
                self.assertIsNotNone(row.get(key),
                                     f"{name} missing {key}")
            self.assertGreater(row["n"], 0, name)
        # walk-forward geometry: 82 cadence splits (frame advanced daily),
        # 75 scored folds / 6,885 games in the 09-03 monitor's phase1 block
        # (pin-synced from 74/6,812 when the frame extended past 08-28)
        self.assertEqual(data["phase1"]["n_folds"], 75)
        self.assertEqual(data["phase1"]["n_games"], 6885)

    def test_drift_artifact_real_frame_finite(self):
        d = pd.read_csv(_latest_artifact(self._ROOT / "data_delivery",
                                          "run_engine_feature_drift_*.csv"))
        # The committed artifact reflects the last pipeline run. Before the
        # P1 monitoring-gap fix it enumerated 53 features; after the fix it
        # enumerates 55 (53 derive_run_features kept + 2 P1 projection level
        # inputs: sp_proj_era_home, sp_proj_era_away). The committed CSV is
        # still the pre-fix 53-row version until the next pipeline run, so
        # this test pins a SUBSET relationship (every model-input column must
        # be present) rather than an exact count — the count pins are in the
        # post-fix re-emit verification in mlb_run_engine_proj_drift_
        # monitoring_*.json.
        self.assertGreaterEqual(len(d), 53)
        self.assertNotIn("run_margin_diff", set(d["feature"]))
        self.assertTrue(d["psi"].notna().all())
        self.assertTrue((d["psi"] >= 0).all())
        # single NB sampler -> no model weights, and no INSUFFICIENT statuses
        self.assertTrue(d["weight_pct"].isna().all())
        self.assertNotIn("INSUFFICIENT", set(d["status"]))
        # Note: the committed artifact is the pre-fix 53-row version until the
        # next pipeline run. The DURABLE PIN (every model-input column appears
        # in the enumeration, including the 2 P1 projection level inputs) is
        # verified by re-emitting with the fix in
        # test_run_engine_feature_cols_includes_p1_projection -- the committed
        # artifact here cannot assert sp_proj_era presence until re-emitted.

    def test_coverage_matches_moneyline_shared_columns(self):
        """For the features both views share, run-engine coverage % must be
        IDENTICAL to the moneyline's (same windows, same machinery)."""
        import tempfile
        import json
        from explainability import compute_feature_coverage
        # EXACT parity invariant: the two COMMITTED artifacts were written
        # from the same pipeline windows (the drift step slices one
        # baseline/current pair and feeds both views) — they must agree
        # row-for-row on every shared feature.
        ml_art = pd.read_csv(_latest_artifact(
            self._ROOT / "data_delivery", "feature_coverage_*.csv"))
        re = pd.read_csv(_latest_artifact(self._ROOT / "data_delivery",
                                          "run_engine_feature_coverage_*.csv"))
        shared = ml_art[ml_art["feature"].isin(set(re["feature"]))]
        shared = shared.sort_values(["feature", "window"]).reset_index(drop=True)
        re2 = re.sort_values(["feature", "window"]).reset_index(drop=True)
        self.assertEqual(len(shared), len(re2))
        for key in ("pct_measured", "pct_nonnull", "n_default_zero",
                    "status"):
            self.assertTrue(
                (shared[key] == re2[key]).all(),
                f"{key} diverged between moneyline and run-engine coverage")
        # LIVE cross-check: a fresh derivation of the moneyline coverage
        # must reproduce the committed moneyline artifact within the
        # documented baseline-tail boundary effect. The pipeline slices its
        # windows from the runtime _decided_snapshot; the committed
        # game_level_features.csv is a post-hoc snapshot whose sort order
        # can differ by ONE game at the tail(≥250) boundary (e.g.
        # wind_advantage_flyball_factor default-zero 48 vs 49 on the
        # baseline window) — so exact equality is not reproducible, but the
        # divergence is bounded and understood.
        baseline, current = self._windows()
        with tempfile.TemporaryDirectory() as tmp:
            ml = compute_feature_coverage(
                baseline, current,
                json.loads(_latest_artifact(
                    self._ROOT / "data_delivery",
                    "run_engine_monitor_*.json").read_text())["date"],
                out_name=str(Path(tmp) / "ml_cov.csv"))
        both = ml.merge(ml_art, on=["feature", "window"],
                        suffixes=("_recomp", "_art"))
        self.assertEqual(len(both), len(ml))
        for key in ("pct_measured", "pct_nonnull"):
            self.assertLessEqual(
                (both[f"{key}_recomp"] - both[f"{key}_art"]).abs().max(),
                1.0, f"recompute vs moneyline artifact: {key} beyond boundary")
        self.assertLessEqual(
            (both["n_default_zero_recomp"]
             - both["n_default_zero_art"]).abs().max(),
            1, "recompute vs moneyline artifact: default-zero beyond boundary")
        self.assertTrue(
            (both["status_recomp"] == both["status_art"]).all(),
            "recompute vs moneyline artifact: status diverged")


if __name__ == "__main__":
    unittest.main()
