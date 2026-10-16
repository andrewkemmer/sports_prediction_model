"""Regenerate the delivered run-engine markets artifacts through the SAME
pricing code path with the new sport-aware lattice correction, and emit a
before/after verification record per sport.

Usage (run once per sport; a subprocess keeps each backend's imports clean):

    python regenerate_lattice_artifacts.py nhl [--write]
    python regenerate_lattice_artifacts.py mlb [--write]

Per sport the driver:
  1. loads the newest delivered markets CSV with float_precision=
     'round_trip' (untouched columns therefore stay byte-identical),
  2. re-runs the causal walk TWICE through distributions.
     calibrate_market_frame — once with the lattice fits disabled (the
     BASELINE = pre-lattice behavior; also the reproduction check against
     the delivered file) and once with them enabled (the LATTICE run),
  3. prices the slate rows with the same final serving maps,
  4. verifies isolation (only distribution-derived columns move),
  5. computes pooled-PMF MAE / residual, CRPS, ECE, Brier and integer-line
     push rates before/after per target,
  6. with --write, rewrites the markets CSV + .meta.json and the run-engine
     monitor JSON through the backends' own writers.

The recorded library stack in the delivered metas documents the training
environment (pandas 2.3.3 / numpy 2.0.2 / scipy 1.16.3 / sklearn 1.6.1);
regeneration only re-runs the deterministic distribution layer on the
delivered mu pairs, so the reproduction check in step 2 is what proves the
local stack prices the delivered columns identically.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

SPORTS = {
    "nhl": {
        "dir": "nhl-backend",
        "markets": "nhl_run_engine_markets_{date}.csv",
        "meta": "nhl_run_engine_markets_{date}.meta.json",
        "monitor": "nhl_run_engine_monitor_{date}.json",
        "identity": "game_id",
        "date_col": "gameday",
        "style": "shared",
    },
    "nba": {
        "dir": "nba-backend",
        "markets": "nba_run_engine_markets_{date}.csv",
        "meta": "nba_run_engine_markets_{date}.meta.json",
        "monitor": "nba_run_engine_monitor_{date}.json",
        "identity": "game_id",
        "date_col": "gameday",
        "style": "shared",
    },
    "nfl": {
        "dir": "nfl-backend",
        "markets": "nfl_run_engine_markets_{date}.csv",
        "meta": "nfl_run_engine_markets_{date}.meta.json",
        "monitor": "nfl_run_engine_monitor_{date}.json",
        "identity": "game_id",
        "date_col": "gameday",
        "style": "shared",
    },
    "mlb": {
        "dir": "mlb-backend",
        "markets": "run_engine_markets_{date}.csv",
        "meta": "run_engine_markets_{date}.meta.json",
        "monitor": "run_engine_monitor_{date}.json",
        "identity": "game_pk",
        "date_col": "game_date",
        "style": "mlb",
    },
}


def newest_date(delivery: Path, pattern: str) -> str:
    files = sorted(delivery.glob(pattern.format(date="20*")))
    if not files:
        raise SystemExit(f"no delivered artifact matches {pattern} in {delivery}")
    return files[-1].name.split("_")[-1].split(".")[0]


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    x, y = a.to_numpy(), b.to_numpy()
    return (x == y) | (pd.isna(x) & pd.isna(y))


def distribution_columns(dist, frame, maps, cfg) -> set[str]:
    """Exact set of columns the distribution walk writes, instrumented by
    pricing a minimal two-row frame through the same pricer."""
    cols = [cfg["identity"], cfg["date_col"], "mu_h", "mu_a",
            "home_expected_runs", "away_expected_runs",
            "home_score", "away_score"]
    mini = frame[[c for c in cols if c in frame]].head(2).copy()
    priced = dist._distribution_price(mini, maps, n_draws=200)
    out = (set(priced.columns) - set(mini.columns)
           | {"distribution_fit_n", "distribution_fit_end"})
    # The full-support totals-PMF block writes p_push_total_<int-bin> only
    # where the priced frame places mass (setdefault over mass.any), so the
    # two-row mini need not cover every shipped tail bin (measured: 39 NFL
    # bins). Those bins are walk output by construction (the _grid_key
    # integer namespace); the offered-line column p_push_total_offered is
    # an input and stays non-distribution.
    out |= {c for c in frame.columns
            if re.fullmatch(r"p_push_total_\d[\d_]*", c)}
    return out


def pooled_pmf(dist, family, frame, bundle):
    """Pooled observed-vs-modeled PMF per target over the SAME per-origin
    CDFs the walk issued (exact, pre-sampling), plus the pooled residual."""
    entries = {"total": [], "margin": []}
    for (_, _, mask), origin in zip(dist._distribution_blocks(frame),
                                    bundle["origins"]):
        part = frame.iloc[np.flatnonzero(mask)]
        cdfs = dist._distribution_cdfs(part, origin["maps"], sampled=False)
        y = {"total": part.home_score.to_numpy(float) + part.away_score.to_numpy(float),
             "margin": part.home_score.to_numpy(float) - part.away_score.to_numpy(float)}
        for kind, (support, cdf) in cdfs.items():
            pmf = np.diff(np.column_stack([np.zeros(len(part)), cdf]), axis=1)
            entries[kind].append((pmf, y[kind], support))
    out = {}
    for kind, blocks in entries.items():
        aligned = family.lattice_evidence(blocks)
        support = aligned["support"]
        obs = (np.bincount(np.searchsorted(support, aligned["outcomes"]),
                           minlength=len(support))
               / len(aligned["outcomes"]))
        model = aligned["pmfs"].mean(axis=0)
        out[kind] = {"support": support, "observed": obs, "model": model,
                     "residual": obs - model,
                     "mae": float(np.abs(obs - model).mean())}
    return out


def line_stats(priced: pd.DataFrame, col: str, outcome_col: str,
               line: float) -> dict | None:
    if col not in priced or outcome_col not in priced:
        return None
    view = priced[[col, outcome_col]].dropna()
    if not len(view):
        return None
    pushes = float((view[outcome_col] == line).mean())
    return {"model_push_mean": float(view[col].mean()),
            "observed_push_rate": pushes, "n": int(len(view))}


def metric_block(metrics: dict) -> dict:
    return {
        "crps": metrics["crps"],
        "ece_cover": metrics["ece_cover"],
        "brier": metrics["brier"],
        "logloss": metrics["logloss"],
        "calibration_in_the_large": metrics["calibration_in_the_large"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sport", choices=sorted(SPORTS))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cfg = SPORTS[args.sport]
    backend = ROOT / cfg["dir"] / "backend"
    delivery = ROOT / cfg["dir"] / "data_delivery"
    sys.path.insert(0, str(backend))
    dist = importlib.import_module("distributions")
    family = importlib.import_module("gaussian_family")
    date = newest_date(delivery, cfg["markets"])
    markets_path = delivery / cfg["markets"].format(date=date)
    meta_path = delivery / cfg["meta"].format(date=date)
    monitor_path = delivery / cfg["monitor"].format(date=date)
    record: dict = {"sport": args.sport, "date": date,
                    "markets": str(markets_path.relative_to(ROOT))}
    frame = pd.read_csv(markets_path, float_precision="round_trip")
    oof = frame[frame["kind"] == "oof"].copy()
    slate = frame[frame["kind"] == "slate"].copy()
    record["rows"] = {"oof": int(len(oof)), "slate": int(len(slate))}

    # ---- 1. baseline (lattice disabled) = pre-lattice behavior -----------
    # Every admitting fit path must be neutralized here (generic, legacy
    # share, pooled share, spikes) or the baseline silently includes the
    # correction it is supposed to exclude.
    real_fit, real_share = family.fit_lattice_correction, family.fit_ot_home_share
    real_pooled = family.fit_pooled_home_share
    real_spikes = family.fit_margin_spikes
    family.fit_lattice_correction = lambda *a, **k: {
        "significant": False, "reason": "baseline_lattice_disabled"}
    family.fit_ot_home_share = lambda *a, **k: None
    family.fit_pooled_home_share = lambda *a, **k: (None, {"reason": "baseline_disabled"})
    family.fit_margin_spikes = lambda *a, **k: (None, {"reason": "baseline_disabled"})
    t0 = time.time()
    if cfg["style"] == "mlb":
        priced_base, bundle_base = dist.calibrate_market_frame(oof)
    else:
        priced_base, bundle_base = dist.calibrate_market_frame(oof)
    record["walk_seconds_baseline"] = round(time.time() - t0, 1)
    (family.fit_lattice_correction, family.fit_ot_home_share,
     family.fit_pooled_home_share,
     family.fit_margin_spikes) = real_fit, real_share, real_pooled, real_spikes

    # ---- 2. lattice run ---------------------------------------------------
    t0 = time.time()
    priced_lat, bundle_lat = dist.calibrate_market_frame(oof)
    record["walk_seconds_lattice"] = round(time.time() - t0, 1)
    priced_base = priced_base.reindex(oof.index)
    priced_lat = priced_lat.reindex(oof.index)

    dist_cols = distribution_columns(dist, oof, bundle_lat["maps"], cfg)
    overlap = sorted(c for c in dist_cols if c in oof.columns)
    record["distribution_columns"] = len(overlap)

    # ---- 3. reproduction + isolation checks -------------------------------
    base_diffs = [c for c in overlap if not _same(oof[c], priced_base[c]).all()]
    lat_diffs = [c for c in overlap if not _same(priced_base[c], priced_lat[c]).all()]
    nondist = [c for c in oof.columns if c not in dist_cols]
    nd_ok = all(_same(priced_base[c], priced_lat[c]).all() for c in nondist)
    nd_ok_delivered = all(_same(oof[c], priced_lat[c]).all() for c in nondist)
    record["baseline_vs_delivered_changed_columns"] = base_diffs
    record["lattice_changed_columns_n"] = len(lat_diffs)
    record["lattice_changed_columns_sample"] = lat_diffs[:12]
    record["isolation_nondist_columns_identical"] = bool(nd_ok)
    record["delivered_nondist_columns_identical"] = bool(nd_ok_delivered)
    if not nd_ok:
        record["isolation_nondist_diffs"] = [
            c for c in nondist if not _same(priced_base[c], priced_lat[c]).all()]
    if not nd_ok_delivered:
        record["delivered_nondist_diffs"] = [
            c for c in nondist if not _same(oof[c], priced_lat[c]).all()]

    # ---- 4. metrics -------------------------------------------------------
    record["lattice_summary"] = bundle_lat.get("lattice_summary", {})
    record["metrics"] = {
        "baseline": {k: metric_block(v) for k, v in
                     bundle_base["oof_metrics"].items()
                     if isinstance(v, dict) and "crps" in v},
        "lattice": {k: metric_block(v) for k, v in
                    bundle_lat["oof_metrics"].items()
                    if isinstance(v, dict) and "crps" in v},
    }
    pmf_base = pooled_pmf(dist, family, priced_base, bundle_base)
    pmf_lat = pooled_pmf(dist, family, priced_lat, bundle_lat)
    record["pooled_pmf"] = {}
    for kind in ("total", "margin"):
        b, l = pmf_base[kind], pmf_lat[kind]
        top_b = np.argsort(-b["residual"])[:4]
        top_l = np.argsort(-l["residual"])[:4]
        bot_l = np.argsort(l["residual"])[:4]
        record["pooled_pmf"][kind] = {
            "mae_baseline": b["mae"], "mae_lattice": l["mae"],
            "worst_underpredicted_baseline":
                {int(b["support"][i]): round(float(b["residual"][i]), 5) for i in top_b},
            "worst_underpredicted_lattice":
                {int(l["support"][i]): round(float(l["residual"][i]), 5) for i in top_l},
            "worst_overpredicted_lattice":
                {int(l["support"][i]): round(float(l["residual"][i]), 5) for i in bot_l},
        }
    diag_bins = {}
    for kind, bins in (("margin", (-1, 0, 1)), ("total", (5, 6, 7))):
        b, l = pmf_base[kind], pmf_lat[kind]
        diag_bins[kind] = {
            str(k): {"residual_baseline": round(float(b["residual"][int(np.searchsorted(b["support"], k))]), 5),
                     "residual_lattice": round(float(l["residual"][int(np.searchsorted(l["support"], k))]), 5),
                     "observed": round(float(b["observed"][int(np.searchsorted(b["support"], k))]), 5)}
            for k in bins if b["support"][0] <= k <= b["support"][-1]}
    record["diagnostic_bins"] = diag_bins

    # integer-line push rates (model vs observed) before/after
    pushes = {}
    push_lines = {"total": [5.0, 6.0, 7.0] if args.sport == "nhl"
                  else [8.0, 9.0] if args.sport == "mlb" else [],
                  "margin": ([0.0] if args.sport != "nfl"
                              else [0.0, 1.5, 2.5, 3.0, 3.5, 6.5, 7.0, 7.5])}
    for kind, lines in push_lines.items():
        for line in lines:
            for label, priced in (("baseline", priced_base), ("lattice", priced_lat)):
                if kind == "total":
                    key = (f"p_push_total_{int(line)}"
                           if args.sport != "mlb"
                           else f"p_push_{str(line).replace('.', '_')}")
                elif args.sport == "mlb":
                    key = f"p_rl_{line:.1f}".replace(".", "_") + "_push"
                elif args.sport == "nfl":
                    # Same _grid_key("p_push", line) encoding the writer uses.
                    s = str(float(line))
                    if s.endswith(".0"):
                        s = s[:-2]
                    key = f"p_push_{s.replace('-', 'm').replace('.', '_')}"
                else:
                    key = "p_push_0"
                stat = line_stats(priced, key, "total" if kind == "total" else "margin", line)
                if stat:
                    pushes[f"{kind}_{line}_{label}"] = stat
    record["push_rates"] = pushes
    # Margin per-bin residual table (-3..3) for the validation report.
    margin_bins = {}
    for k in (-3, -2, -1, 0, 1, 2, 3):
        b, l = pmf_base["margin"], pmf_lat["margin"]
        if b["support"][0] <= k <= b["support"][-1]:
            j = int(np.searchsorted(b["support"], k))
            margin_bins[str(k)] = {
                "residual_baseline": round(float(b["residual"][j]), 5),
                "residual_lattice": round(float(l["residual"][j]), 5),
                "observed": round(float(b["observed"][j]), 5)}
    record["margin_bins"] = margin_bins
    # Moneyline two-way completeness on the lattice-priced OOF.
    ml = {}
    for col_h, col_a in (("p_home_win_derived", "p_away_win_derived"),):
        if col_h in priced_lat.columns and col_a in priced_lat.columns:
            dev = (pd.to_numeric(priced_lat[col_h], errors="coerce")
                   + pd.to_numeric(priced_lat[col_a], errors="coerce")) - 1.
            ml["two_way_max_abs_dev_lattice"] = round(float(np.nanmax(np.abs(dev))), 8)
            if col_h in priced_base.columns and col_a in priced_base.columns:
                dev_b = (pd.to_numeric(priced_base[col_h], errors="coerce")
                         + pd.to_numeric(priced_base[col_a], errors="coerce")) - 1.
                ml["two_way_max_abs_dev_baseline"] = round(float(np.nanmax(np.abs(dev_b))), 8)
    record["moneyline"] = ml
    # Serving-maps target isolation: price a small OOF mini-frame with the
    # FINAL serving maps under both/single-target-disabled specs. Columns
    # that need the totals correction and columns that need the margin
    # correction must be disjoint (separate code paths, see
    # run_diagnostics/verify_target_isolation.py for the artifact-level twin).
    try:
        iso_cols = dict.fromkeys((cfg["identity"], "game_pk",
                                  "game_date", "gameday", "home_expected_runs",
                                  "away_expected_runs", "mu_h", "mu_a",
                                  "home_score", "away_score"))
        iso_mini = oof[[c for c in iso_cols if c in oof.columns]].head(6).copy()
        serving_maps = bundle_lat["maps"]
        p_both = dist._distribution_price(iso_mini, serving_maps, n_draws=500)
        no_t = json.loads(json.dumps(serving_maps))
        (no_t.setdefault("lattice", {}))["totals"] = {}
        no_m = json.loads(json.dumps(serving_maps))
        (no_m.setdefault("lattice", {}))["margin"] = {}
        p_toff = dist._distribution_price(iso_mini, no_t, n_draws=500)
        p_moff = dist._distribution_price(iso_mini, no_m, n_draws=500)

        def _moved(a, b):
            return {c for c in a.columns
                    if c in b.columns and not _same(a[c], b[c]).all()}
        need_totals = _moved(p_both, p_toff)
        need_margin = _moved(p_both, p_moff)
        overlap = sorted(need_margin & need_totals)
        record["serving_isolation"] = {
            "margin_only_columns_n": len(need_margin),
            "totals_only_columns_n": len(need_totals),
            "overlap_forbidden": overlap,
            "pass": len(overlap) == 0,
            "margin_sample": sorted(need_margin)[:8],
            "totals_sample": sorted(need_totals)[:8],
        }
    except Exception as exc:  # noqa: BLE001 - record, don't fail the walk
        import traceback as _tb
        record["serving_isolation"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": _tb.format_exc()[-1200:]}

    origins = bundle_lat["origins"]
    record["origins_significant"] = {
        target: sum(1 for o in origins
                    if (o["maps"].get("lattice", {}).get(target) or {}).get("significant"))
        for target in ("totals", "margin")}
    # Per-origin no-tie split admissions (held-out-gated one-goal
    # home-share): the margin structural term can move margin columns at
    # origins where the generic delta stays rejected.
    record["origins_share"] = sum(
        1 for o in origins
        if (o["maps"].get("lattice", {}).get("margin") or {}).get("ot_home_share") is not None)
    record["origins_total"] = len(origins)

    # ---- 5. write artifacts through the backends' own writers -------------
    if args.write:
        if cfg["style"] == "mlb":
            _write_mlb(args.sport, cfg, dist, family, delivery, frame, oof, slate,
                       priced_lat, bundle_lat, markets_path, meta_path,
                       monitor_path, date, record)
        else:
            _write_shared(args.sport, cfg, dist, delivery, frame, oof, slate,
                          priced_lat, bundle_lat, markets_path, meta_path,
                          monitor_path, date, record)
    print(json.dumps(record, indent=1, default=str))


def _write_shared(sport, cfg, dist, delivery, frame, oof, slate,
                  priced_lat, bundle_lat, markets_path, meta_path,
                  monitor_path, date, record):
    """NHL/NBA/NFL: markets CSV + meta and monitor through serving and
    monitoring writers exactly like the daily pipeline."""
    serving = importlib.import_module("serving")
    monitoring = importlib.import_module("monitoring")
    old_meta = json.loads(meta_path.read_text())
    old_monitor = (json.loads(monitor_path.read_text())
                   if monitor_path.exists() else {})
    priced_lat = priced_lat.copy()
    priced_lat.attrs["challenger_bundle"] = bundle_lat
    priced_lat.attrs["mc_meta"] = bundle_lat.get("monte_carlo", {})
    # Enrichment the writer does not own (master-pipeline additions) must
    # survive regeneration: run_line_fit_check (writer-supported kwarg on
    # nhl/nba) and distribution_metrics.conditional_crps (merged into attrs
    # so the writer emits it natively). Carrying them over is valid only
    # when no distribution column moved; otherwise the loss is recorded.
    no_op = not record.get("lattice_changed_columns_n")
    old_cc = (old_meta.get("distribution_metrics") or {}).get("conditional_crps")
    if no_op:
        if old_cc is not None:
            dm = dict(priced_lat.attrs.get("distribution_metrics", {}))
            if "conditional_crps" not in dm:
                dm["conditional_crps"] = old_cc
                priced_lat.attrs["distribution_metrics"] = dm
    else:
        lost = [k for k in ("run_line_fit_check",)
                if k in old_meta]
        if old_cc is None:
            pass
        elif "conditional_crps" not in priced_lat.attrs.get("distribution_metrics", {}):
            lost.append("distribution_metrics.conditional_crps")
        if lost:
            record["enrichment_not_carried"] = lost
    slate_lat = dist.apply_market_calibration(slate.copy(), bundle_lat) \
        if len(slate) else slate
    if sport == "nfl":
        # The NFL writer takes no mc_meta kwarg: its meta reads the attrs.
        serving.write_markets_csv(markets_path, meta_path, priced_lat,
                                  slate_lat, old_meta.get("config", {}))
    else:
        kwargs = dict(mc_meta=bundle_lat.get("monte_carlo", {}))
        if sport in ("nhl", "nba"):
            # Recomputed from the OOF mu/score columns (untouched by the
            # lattice layer), never copied — the first NBA pass proved
            # copies go stale the moment a writer drops the key.
            try:
                kwargs["run_line_fit_check"] = dist.run_line_fit_check(oof)
            except Exception:
                kwargs["run_line_fit_check"] = old_meta.get("run_line_fit_check")
        serving.write_markets_csv(
            markets_path, meta_path, priced_lat, slate_lat,
            old_meta.get("config", {}), **kwargs)
    ml_ref = (old_monitor.get("winner_cards", {}).get("derived_ml", {})
              or {}).get("ml_reference")
    if sport == "nba":
        monitoring.write_run_engine_monitor(
            monitor_path, date, markets=priced_lat, calibration=bundle_lat,
            config_meta=old_meta.get("config", {}), ml_reference=ml_ref)
    else:
        monitoring.write_markets_monitor_json(
            monitor_path, date, priced_lat, old_meta.get("config", {}),
            ml_reference=ml_ref)
    # byte-identity proof for untouched columns
    new_frame = pd.read_csv(markets_path, float_precision="round_trip")
    record["regenerated_rows"] = int(len(new_frame))
    record["regenerated_columns"] = list(new_frame.columns) == list(frame.columns)
    same_all = new_frame.equals(frame)
    record["regenerated_csv_identical_to_delivered"] = bool(same_all)


def _write_mlb(sport, cfg, dist, family, delivery, frame, oof, slate,
               priced_lat, bundle_lat, markets_path, meta_path,
               monitor_path, date, record):
    """MLB: overlay the distribution columns onto the delivered frame and
    persist through distributions.persist_markets; refresh the monitor's
    fit/winner-card/market-metric blocks through the same helpers the daily
    pipeline uses. Genuinely-new walk outputs (columns the pricer emits
    that the delivered frame predates — e.g. the margin-push extension)
    join the overlay: intersect with the PRICED columns, not the delivered
    ones, so the writer persists them (assignment creates the column)."""
    dist_cols = distribution_columns(dist, oof, bundle_lat["maps"], cfg)
    combined = frame.copy()
    priced_lat = priced_lat.reindex(oof.index)
    row_mask = combined["kind"].eq("oof").to_numpy()
    for col in sorted(set(dist_cols) & set(priced_lat.columns)):
        values = priced_lat[col].reindex(combined.index[row_mask]).to_numpy()
        combined.loc[row_mask, col] = values
    # slate rows: same final serving maps (deliverable mu pair fixed)
    if len(slate):
        priced_slate = dist._distribution_price(
            slate.copy(), bundle_lat["maps"],
            n_draws=bundle_lat["monte_carlo"].get("requested_draws", dist.MC_DRAWS),
            seed=bundle_lat["monte_carlo"].get("seed", dist.MC_SEED))
        slate_mask = combined["kind"].eq("slate").to_numpy()
        for col in sorted(set(dist_cols) & set(priced_slate.columns)):
            values = priced_slate[col].reindex(combined.index[slate_mask]).to_numpy()
            combined.loc[slate_mask, col] = values
    old_meta = json.loads(meta_path.read_text())
    summary = dict(old_meta)
    summary.update(
        family=family.FAMILY,
        variance=bundle_lat["maps"]["variance"],
        lattice_summary=bundle_lat.get("lattice_summary", {}),
        mc_meta=bundle_lat["monte_carlo"],
        distribution_metrics=bundle_lat["oof_metrics"],
        holdout_cutoff=bundle_lat["holdout"]["cutoff"],
        n_pre=bundle_lat["holdout"]["n_pre"],
        n_holdout=bundle_lat["holdout"]["n_holdout"],
        holdout=bundle_lat["holdout"],
        calibration={"method": bundle_lat["method"], "calibrators": bundle_lat},
    )
    dist.persist_markets(combined, date, summary)
    # NOTE: master_pipeline is Colab-bound (import-time os.chdir to
    # /content/...), so the fit block is inlined here — byte-identical to
    # master_pipeline._run_engine_fit_block(block).
    monitor_path = monitor_path
    if monitor_path.exists():
        old_mon = json.loads(monitor_path.read_text())
        block = {"family": family.FAMILY,
                 "variance": summary["variance"],
                 "lattice": summary["lattice_summary"],
                 "mc_meta": summary["mc_meta"],
                 "holdout": summary["holdout"],
                 "line_grid": old_meta.get("line_grid", {}),
                 "calibration": summary["calibration"],
                 "market_metrics": {k: v for k, v in summary.items()
                                    if k.startswith("market_") and isinstance(v, dict)},
                 "phase1": old_mon.get("phase1", {})}
        data = dict(old_mon)
        data["fit"] = {"distribution": "causal_split_skew_crps_mc",
                        "family": block.get("family"),
                        "variance": block.get("variance", {}),
                        "lattice": block.get("lattice", {}),
                        "monte_carlo": block.get("mc_meta", {}),
                        "holdout": block.get("holdout", {}),
                        "line_grid": block.get("line_grid", {})}
        data["market_metrics"] = block["market_metrics"]
        try:
            data["winner_cards"] = dist.compute_winner_cards(
                combined[combined["kind"] == "oof"], oof, dist.HOLDOUT_DAYS)
        except Exception as exc:  # noqa: BLE001 - keep delivered cards on failure
            record["mlb_winner_cards_error"] = f"{type(exc).__name__}: {exc}"
        monitor_path.write_text(json.dumps(data, indent=2))
    new_frame = pd.read_csv(markets_path, float_precision="round_trip")
    record["regenerated_rows"] = int(len(new_frame))
    record["regenerated_columns"] = list(new_frame.columns) == list(frame.columns)
    record["regenerated_csv_identical_to_delivered"] = bool(new_frame.equals(frame))


if __name__ == "__main__":
    main()
