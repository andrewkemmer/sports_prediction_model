"""Read-only NHL quality audit; no pulls, retraining, serving writes, or adoption.

Run from the repository root:
  python nhl-backend/backend/audit_model_quality.py --date 20261006 \
      --output docs/nhl_model_audit_20261006

Bundle inspection is structural only. Serialized estimators can warn when the
local library stack differs from production; no bundle predictions are made.
All experiments reuse retrospectively selected member/blend outputs, so they
are exploratory, NOT unbiased estimates of a new production policy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    from backend import config, features, monitoring
except ImportError:  # direct script execution
    import config
    import features
    import monitoring


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape:
        raise ValueError("target/probability shapes differ")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("audit inputs must be finite")
    if not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("binary targets or probability bounds invalid")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"n": len(y), "auc": float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2 else None,
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "home_win_rate": float(y.mean()), "mean_probability": float(p.mean())}


def reconstruct_history(history: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct published fold labels only for the observed 7-date contract.

    Caller MUST reconcile grading counts and headline metrics with the monitor;
    absent warmup-date inputs, reconstruction is not a general fold generator.
    """
    d = history.copy()
    if d.game_id.duplicated().any():
        raise ValueError("duplicate history game IDs")
    ids = d.game_id.astype(str)
    if not ids.map(lambda s: bool(re.fullmatch(r"\d{4}0[23]\d{4}", s))).all():
        raise ValueError("unknown NHL season/game-type ID encoding")
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    days = sorted(d.game_date.unique())
    d["fold_id"] = d.game_date.map({day: i // 7 for i, day in enumerate(days)})
    d["season"] = d.game_id.astype(str).str[:4].astype(int)
    d["postseason"] = d.game_id.astype(str).str[4:6].eq("03")
    d["provisional"] = d.groupby("fold_id").game_id.transform("size").lt(40)
    d["grading"] = ~d.postseason & ~d.provisional
    return d


def paired_block_interval(y, baseline, candidate, dates, draws=1000) -> dict:
    """Paired 4-calendar-week moving-block bootstrap; conditional diagnostics.

    Blocks keep games on a date together. This is not a complete team-cluster
    model, nor a selection-adjusted confidence interval.
    """
    y, baseline, candidate = map(lambda x: np.asarray(x, float),
                                (y, baseline, candidate))
    metrics(y, baseline)
    metrics(y, candidate)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if len(dates) != len(y):
        raise ValueError("dates/target lengths differ")
    week = dates.dt.to_period("W").astype(str)
    groups = [np.flatnonzero(week.to_numpy() == w) for w in sorted(week.unique())]
    width = min(4, len(groups))
    if not groups:
        raise ValueError("bootstrap needs observations")
    blocks = [np.concatenate(groups[i:i + width])
              for i in range(len(groups) - width + 1)]
    rng = np.random.default_rng(42)
    bp, cp = np.clip(baseline, 1e-7, 1 - 1e-7), np.clip(candidate, 1e-7, 1 - 1e-7)
    diff = (-y * np.log(cp) - (1 - y) * np.log(1 - cp)
            + y * np.log(bp) + (1 - y) * np.log(1 - bp))
    ll, auc = [], []
    for _ in range(draws):
        parts, n = [], 0
        while n < len(y):
            block = blocks[int(rng.integers(len(blocks)))]
            parts.append(block)
            n += len(block)
        ix = np.concatenate(parts)[:len(y)]
        ll.append(float(diff[ix].mean()))
        if len(np.unique(y[ix])) == 2:
            auc.append(float(roc_auc_score(y[ix], candidate[ix])
                             - roc_auc_score(y[ix], baseline[ix])))
    return {"delta_logloss": float(diff.mean()),
            "delta_logloss_ci95": np.quantile(ll, [.025, .975]).tolist(),
            "delta_auc_ci95": np.quantile(auc, [.025, .975]).tolist() if auc else None,
            "draws": draws, "block_weeks": width,
            "scope": "conditional on published predictions; no selection correction"}


def inspect_bundle(path: Path) -> tuple[pd.DataFrame, dict]:
    import joblib
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(path)
    rows, structure = [], {"load_warnings": [str(w.message) for w in caught],
                          "prediction_performed": False, "members": {}}
    for name, model in bundle["moneyline_models"].items():
        weight = bundle["ensemble_weights"][name]
        info = {"width": int(model.n_features_in_), "blend_weight": weight}
        if name == "elasticnet":
            cols = features.linear_feature_columns(bundle["feature_columns"])
            values = np.asarray(model.coef_).reshape(-1)
            info.update(intercept=float(model.intercept_[0]),
                        nonzero_coefficients=int(np.count_nonzero(values)))
            pre = bundle["moneyline_preprocessors"][name]
            for c, v in zip(cols, values):
                rows.append(dict(member=name, feature=c, importance_kind="standardized_coefficient",
                                 value=float(v), abs_value=float(abs(v)), blend_weight=weight,
                                 raw_unit_coefficient=float(v / pre.stds[c])))
        else:
            if name == "xgboost":
                booster = model.get_booster()
                cols = booster.feature_names
                score = booster.get_score(importance_type="total_gain")
                values = np.array([score.get(c, 0.) for c in cols])
                info.update(rounds=booster.num_boosted_rounds(),
                            feature_types=booster.feature_types,
                            categorical_count=booster.feature_types.count("c"))
            else:
                cols = model.feature_name_
                values = model.booster_.feature_importance(importance_type="gain")
                info.update(rounds=model.booster_.num_trees(),
                            team_metadata={c: model.booster_.dump_model()["feature_infos"].get(c)
                                           for c in config.TREE_CATEGORICAL_COLS})
            for c, v in zip(cols, values):
                rows.append(dict(member=name, feature=c, importance_kind="training_total_gain",
                                 value=float(v), abs_value=float(v), blend_weight=weight,
                                 normalized_gain_pct=float(100 * v / values.sum())))
        structure["members"][name] = info
    return pd.DataFrame(rows), structure


def semantic_probes() -> dict:
    g = pd.DataFrame([dict(game_id="1", season=2024, gameday=pd.Timestamp("2024-10-01"),
                          home_team="ANA", away_team="BOS", home_score=1, away_score=0)])
    _, ratings = features._elo_apply(features.team_events(g))
    # Departed goalie gains starts for another team and improperly wins ANA's vote.
    games, boxscores = [], []
    for i, (team, goalie) in enumerate([("ANA", "1"), ("ANA", "2"), ("ANA", "2"),
                                       ("BOS", "1"), ("BOS", "1"), ("BOS", "1"), ("BOS", "1")]):
        gid = str(i + 1)
        games.append(dict(game_id=gid, season=2024,
                          gameday=pd.Timestamp("2024-10-01") + pd.Timedelta(days=i),
                          home_team=team, away_team="BUF"))
        boxscores.append(dict(game_id=gid, home_goalie_id=goalie,
                              home_goalie_name="departed" if goalie == "1" else "current",
                              home_goals_against=2, home_shots_against=25, home_goalie_toi=60))
    target = pd.DataFrame([dict(game_id="8", season=2024, gameday=pd.Timestamp("2024-10-10"),
                               home_team="ANA", away_team="BUF")])
    state, _ = features.goalie_state(pd.DataFrame(boxscores), target,
                                    team_source=pd.concat([pd.DataFrame(games), target]))
    y, p = np.array([0., 1., 0., 1.]), np.array([.2, .8, .3, .7])
    return {"equal_elo_implied_home_expectation": float(1 - (ratings["ANA"] - 1500) / 20),
            "positive_home_advantage_expected": float(expit(np.log(10) * config.ELO_HOME_ADV / 400)),
            "goalie_trade_selected_name": state.iloc[0].g_home_name,
            "goalie_trade_selected_starts": float(state.iloc[0].goalie_starts_home),
            "market_metric_correct_arguments": monitoring._markets_card_metrics(y, p),
            "market_metric_current_caller_arguments": monitoring._markets_card_metrics(p, y)}


def run(day: str, output: Path, draws: int) -> dict:
    delivery = config.DATA_DELIVERY_DIR
    def read(name):
        return json.loads((delivery / name).read_text(encoding="utf-8"))
    monitor = read(f"nhl_model_monitor_{day}.json")
    calibration = read(f"nhl_calibration_{day}.json")
    history_path = delivery / f"nhl_predictions_history_{day}.csv"
    d = reconstruct_history(pd.read_csv(history_path, dtype={"game_id": str}))
    geo = monitor["fold_geometry"]
    counts = {"total_val_games": len(d), "n_folds": d.fold_id.nunique(),
              "provisional_games": int(d.provisional.sum()),
              "postseason_games": int(d.postseason.sum()), "grading_games": int(d.grading.sum())}
    if counts["grading_games"] != geo["oof_regular"]["n"] or any(
            counts[k] != geo[k] for k in ["total_val_games", "n_folds", "provisional_games"]):
        raise ValueError("fold reconstruction does not reconcile with published geometry")
    raw_col, cal_col = "home_win_prob_model", "home_win_prob_model_calibrated"
    headline = metrics(d.loc[d.grading, "home_win"], d.loc[d.grading, raw_col])
    if abs(headline["logloss"] - monitor["metrics"]["logloss"]) > 5e-5:
        raise ValueError("reconstructed grading metrics disagree with headline")
    slices = []
    for label, mask in [("all", np.ones(len(d), bool)), ("grading", d.grading),
                        ("postseason", d.postseason), ("provisional", d.provisional)]:
        for col in [raw_col, cal_col]:
            g = d.loc[mask]
            slices.append(dict(slice=label, probability=col, **metrics(g.home_win, g[col])))
    for season, g in d[d.grading].groupby("season"):
        slices.append(dict(slice=f"grading_season_{season}", probability=raw_col,
                           **metrics(g.home_win, g[raw_col])))
    for month, g in d.groupby(d.game_date.dt.to_period("M")):
        slices.append(dict(slice=f"all_month_{month}", probability=raw_col,
                           **metrics(g.home_win, g[raw_col])))
    bucket = pd.cut(np.maximum(d[raw_col], 1 - d[raw_col]), [.5, .55, .6, .65, .7, .8, 1],
                    include_lowest=True)
    confidence = []
    for band, g in d.groupby(bucket, observed=True):
        confidence.append(dict(bucket=str(band), n=len(g),
                               mean_favorite_probability=float(np.maximum(g[raw_col], 1 - g[raw_col]).mean()),
                               favorite_win_rate=float(((g[raw_col] >= .5) == (g.home_win == 1)).mean()),
                               logloss=metrics(g.home_win, g[raw_col])["logloss"]))
    market_path = delivery / f"nhl_run_engine_markets_{day}.csv"
    market = pd.read_csv(market_path, dtype={"game_id": str}).query("kind == 'oof'")
    market_metrics = {}
    for key, kind, line in [("derived_ml", "ml", 0), ("over_5", "over", 5),
                            ("over_6", "over", 6), ("over_7", "over", 7),
                            ("home_cover_1", "spread", 1), ("home_cover_2", "spread", 2)]:
        p, y = monitoring._run_engine_line_pairs(market, kind, line)
        market_metrics[key] = metrics(y, p)
    matched = d.merge(market[["game_id", "derived_ml", "p_home_cover_0", "p_push_0"]],
                      on="game_id", validate="one_to_one")
    if len(matched) != len(d):
        raise ValueError("markets/history IDs differ")
    g = matched[matched.grading].copy()
    y, base = g.home_win.to_numpy(), g[raw_col].to_numpy()
    conditional = g.p_home_cover_0.to_numpy() / (1 - g.p_push_0.to_numpy())
    half_tie = g.p_home_cover_0.to_numpy() + .5 * g.p_push_0.to_numpy()
    experiments = []
    for name, engine in [("raw_derived_alias", g.derived_ml.to_numpy()),
                         ("conditional_no_tie", conditional), ("half_tie_ot50", half_tie)]:
        for alpha in [0., .1, .25, .5, 1.]:
            p = expit((1 - alpha) * logit(base) + alpha * logit(engine))
            experiments.append(dict(candidate=name, engine_weight=alpha, **metrics(y, p)))
    # Fixed protocol: prior grading rows only, >=500 rows, regularized home-space Platt.
    # Retrospective base weights still make this exploratory despite causal calibrator fitting.
    prequential = base.copy()
    for fid in sorted(g.fold_id.unique()):
        prior, current = g.fold_id.to_numpy() < fid, g.fold_id.to_numpy() == fid
        if prior.sum() >= 500:
            lr = LogisticRegression(C=1, solver="lbfgs", max_iter=1000)
            lr.fit(logit(base[prior]).reshape(-1, 1), y[prior])
            if lr.coef_[0, 0] > 0:
                prequential[current] = lr.predict_proba(logit(base[current]).reshape(-1, 1))[:, 1]
    experiments.append(dict(candidate="prior_grading_home_platt_C1", engine_weight=None,
                            **metrics(y, prequential)))
    evidence = {"schema": "nhl-quality-audit/v1", "artifact_date": day,
                "warning": "Retrospectively selected blend weights; experiments are not adoption evidence.",
                "counts": counts, "headline_recomputed": headline,
                "member_metrics": monitor["ensemble"], "calibration_params": calibration["calibration"]["params"],
                "confidence_buckets": confidence, "market_metrics_correct_argument_order": market_metrics,
                "semantic_probes": semantic_probes(),
                "calibration_paired_interval": paired_block_interval(
                    y, base, g[cal_col], g.game_date, draws),
                "home_platt_exploratory_interval": paired_block_interval(
                    y, base, prequential, g.game_date, draws),
                "recent_500_grading_raw": metrics(y[-500:], base[-500:]),
                "recent_500_grading_home_platt": metrics(y[-500:], prequential[-500:]),
                "stack": {m: importlib.metadata.version(m) for m in
                          ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm"]}}
    feature_weights, structure = inspect_bundle(config.MODEL_BUNDLE)
    evidence["bundle_structure"] = structure
    q = float(d.home_win.mean())
    evidence["baseline_diagnostic"] = {
        "reported_brier_baseline": monitor["brier_baseline"],
        "always_home_brier": 1 - q,
        "constant_empirical_rate_brier": q * (1 - q),
        "note": "Empirical-rate value is descriptive, not a causal estimated baseline."}
    evidence["market_coherence"] = {
        "mean_score_tie_mass": float(market.p_push_0.mean()),
        "derived_alias_max_gap": float((market.derived_ml - market.p_home_win_derived).abs().max()),
        "derived_threeway_max_sum_error": float((market.p_home_win_derived
                                                 + market.p_away_win_derived + market.p_tie - 1).abs().max()),
        "sealed_rows": int(market.frame_view.eq("sealed").sum())}
    for prefix, lines in [("p_over_", range(4, 13)), ("p_home_cover_", range(-8, 9))]:
        cols = [prefix + (f"m{-line}" if line < 0 else str(line)) for line in lines]
        differences = np.diff(market[cols].to_numpy(), axis=1)
        evidence["market_coherence"][prefix + "nonmonotone_rows"] = int(
            (differences > 1e-9).any(axis=1).sum())
        evidence["market_coherence"][prefix + "max_upward_jump"] = float(differences.max())
    paths = [history_path, market_path, config.MODEL_BUNDLE,
             delivery / f"nhl_model_monitor_{day}.json", delivery / f"nhl_calibration_{day}.json"]
    evidence["input_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(slices).to_csv(output / "metrics_by_slice.csv", index=False)
    pd.DataFrame(experiments).to_csv(output / "exploratory_experiments.csv", index=False)
    feature_weights.to_csv(output / "feature_weights.csv", index=False)
    def safe(value):
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [safe(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value
    evidence = safe(evidence)
    (output / "diagnostics.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261006")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    result = run(args.date, args.output, args.bootstrap_draws)
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
                      "headline": result["headline_recomputed"]}, indent=2))
