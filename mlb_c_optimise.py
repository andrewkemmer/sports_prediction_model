"""One-shot optimisation study of the MLB C arm (the fitted run model).

Five questions, all measured downstream on the moneyline, because the rating
is only worth anything to the extent it improves the served probability:

1. Ablation      -- which nuisance block actually carries the gain?  Variants
                    go from team one-hots alone up to the shipped design, so a
                    win that lives in the park/starter/prior columns is
                    separable from a win that lives in the fitted team effect.
2. Sweep         -- 2D grid over the ridge penalty and the recency half-life.
                    ALPHA=50 / HALF_LIFE=120 were set a priori and have never
                    been measured against an alternative.
3. Seeds         -- repeat the winning configuration under three member seeds
                    with every arm refit, so the paired delta is not one draw.
4. Likelihood    -- Gaussian ridge on run counts vs the Poisson GLM that is
                    the correctly-specified family for a count target.
5. Starter-aware -- add the team's own starting pitcher to the design, which
                    turns ``def_rating`` into run prevention net of the starter
                    on the mound instead of total run prevention.

Arms A and B never depend on the rating, so they are fitted once per seed and
reused.  Each variant costs exactly one arm-C walk forward.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sport_abc as sa

_BACKEND = Path(__file__).resolve().parent / "mlb-backend" / "backend"
if str(_BACKEND) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_BACKEND))
import team_run_ratings as tgr  # noqa: E402

#: Recency half-lives in days; 1e6 is "no recency weighting".
HALF_LIVES = (45.0, 90.0, 120.0, 240.0, 1e6)
ALPHAS = (5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
#: Design variants: (label, blocks, own_starter).
ABLATION = (
    ("teams only", (), False),
    ("+ context", ("context",), False),
    ("+ context + prior", ("context", "prior"), False),
    ("+ components", ("components",), False),
    ("full (shipped)", tgr.DESIGN_BLOCKS, False),
)
STARTER = (
    ("starter-aware", tgr.DESIGN_BLOCKS, True),
)
SEEDS = (42, 7, 2027)
BAND = (0.45, 0.55)
REPS = 2000

LOG: list[str] = []


def say(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.append(line)


def set_member_seed(mods, seed: int) -> int:
    """Rewrite random_state in every model-param dict the backend exported.

    ``training`` imports XGBOOST_PARAMS/LIGHTGBM_PARAMS/... by name, and those
    are the same dict objects the config module holds, so mutating the dict in
    place reaches the estimators at fit time.
    """
    touched = 0
    seen = set()
    for module in (mods.ml, mods.config):
        for name in dir(module):
            if not name.isupper():
                continue
            value = getattr(module, name, None)
            if not isinstance(value, dict) or "random_state" not in value:
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            value["random_state"] = int(seed)
            touched += 1
    return touched


def _oof(mods, columns: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    sa._set_arm(mods, columns)
    out = sa._walk("mlb", mods, frame, folds=[])
    out = out.rename(columns={"p": "p_", "p_calibrated": "p_cal_",
                              "grades_pooled": "grades_"})
    return out


def _merge(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    labels = list(parts)
    merged = parts[labels[0]][["game_id", "gameday", "home_win",
                               "grades_", "p_", "p_cal_"]].rename(
        columns={"p_": f"p_{labels[0]}", "p_cal_": f"p_cal_{labels[0]}",
                 "grades_": "grades_A"})
    for label in labels[1:]:
        part = parts[label][["game_id", "p_", "p_cal_", "grades_"]].rename(
            columns={"p_": f"p_{label}", "p_cal_": f"p_cal_{label}",
                     "grades_": f"grades_{label}"})
        merged = merged.merge(part, on="game_id", how="outer")
    merged["grades_A"] = merged["grades_A"].fillna(False).astype(bool)
    for label in labels:
        column = f"grades_{label}"
        if column in merged:
            merged[column] = merged[column].fillna(False).astype(bool)
    return merged


def _delta(sub: pd.DataFrame, left: str, right: str) -> dict:
    if len(sub) < 200:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "better": np.nan}
    row = sa._pair_delta(sub, left, right, REPS, 20260101)
    return {"delta": row.get("delta"), "lo": row.get("lo"),
            "hi": row.get("hi"), "better": row.get("better_share")}


def _measure(label: str, merged: pd.DataFrame, **meta) -> dict:
    graded = merged[merged["grades_A"]].copy()
    y = graded["home_win"].to_numpy(float)
    band = graded[graded["p_A"].between(*BAND)]
    yb = band["home_win"].to_numpy(float)
    row = {"variant": label, **meta,
           "n": int(len(graded)), "n_band": int(len(band))}
    for arm in ("A", "B", "C"):
        s = sa._score(graded[f"p_{arm}"].to_numpy(float), y)
        sb = sa._score(band[f"p_{arm}"].to_numpy(float), yb)
        row[f"ll_{arm}"] = s["logloss"]
        row[f"auc_{arm}"] = s["auc"]
        row[f"llb_{arm}"] = sb["logloss"]
        row[f"aucb_{arm}"] = sb["auc"]
    for left, right in (("A", "C"), ("B", "C")):
        d = _delta(band, left, right)
        row[f"band_{left}_{right}"] = d["delta"]
        row[f"band_{left}_{right}_lo"] = d["lo"]
        row[f"band_{left}_{right}_hi"] = d["hi"]
        row[f"band_{left}_{right}_better"] = d["better"]
        row[f"all_{left}_{right}"] = _delta(graded, left, right)["delta"]
    say(f"{label:34s} ll_all C={row['ll_C']:.5f}  "
        f"ll_band C={row['llb_C']:.5f}  "
        f"B->C band {row['band_B_C']:+.5f} "
        f"[{row['band_B_C_lo']:+.5f},{row['band_B_C_hi']:+.5f}]")
    return row


def variant(mods, arms, frame, label: str, **panel_kwargs) -> dict:
    started = time.time()
    panel = tgr.build_causal_panel(frame, **panel_kwargs)
    frame_c = frame.copy()
    for column in tgr.RUN_RATING_COLS:
        frame_c[column] = panel[column].to_numpy()
    coverage = float(frame_c["net_rating_diff"].notna().mean())
    oof_c = _oof(mods, arms["C"], frame_c)
    merged = _merge({**STATE["ab"], "C": oof_c})
    meta = {k: (v if not isinstance(v, tuple) else list(v))
            for k, v in panel_kwargs.items()}
    meta.update({"fits": panel.attrs.get("fits"), "coverage": coverage,
                 "seconds": round(time.time() - started, 1)})
    return _measure(label, merged, **meta)


STATE: dict = {"ab": {}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mlb_c_opt")
    ap.add_argument("--skip-seeds", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="trim the frame and the grid to smoke-test plumbing")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    say("loading MLB backend and building the frame")
    mods = sa.load_modules("mlb")
    sa.register_trial_columns(mods, "mlb")
    arms = sa.arm_columns(mods, "mlb")
    frame = sa.build_frame(mods, "mlb")
    if args.quick:
        global ALPHAS, HALF_LIVES, SEEDS, REPS
        frame = (frame.sort_values("gameday").tail(900).reset_index(drop=True))
        ALPHAS, HALF_LIVES, SEEDS, REPS = (10.0, 50.0), (90.0,), (42,), 200
        say(f"quick mode: {len(frame)} games, alphas {ALPHAS}, "
            f"half-lives {HALF_LIVES}, seeds {SEEDS}")
    say(f"frame {frame.shape[0]} games x {frame.shape[1]} cols; "
        f"arms A {len(arms['A'])} / B {len(arms['B'])} / C {len(arms['C'])}")

    started = time.time()
    t_all = time.time()
    say("fitting arms A and B once (they do not see the rating)")
    STATE["ab"] = {"A": _oof(mods, arms["A"], frame),
                   "B": _oof(mods, arms["B"], frame)}
    say(f"A/B done in {(time.time() - started) / 60:.1f} min")

    results: dict[str, list] = {"ablation": [], "sweep": [], "seeds": [],
                                "likelihood": [], "starter": []}

    say("=== 1. design ablation")
    for label, blocks, own_starter in ABLATION:
        results["ablation"].append(variant(
            mods, arms, frame, label, blocks=blocks,
            own_starter=own_starter))

    say("=== 2. alpha x half-life sweep (shipped design)")
    for alpha in ALPHAS:
        for half_life in HALF_LIVES:
            if alpha == tgr.ALPHA and half_life == tgr.HALF_LIFE_DAYS:
                say(f"alpha={alpha:g} half_life={half_life:g} -- shipped point, "
                    f"already measured by the ablation")
                continue
            results["sweep"].append(variant(
                mods, arms, frame, f"alpha={alpha:g} hl={half_life:g}",
                alpha=alpha, half_life_days=half_life))

    shipped = next(r for r in results["ablation"] if r["variant"].startswith("full"))
    pool = results["sweep"] + [shipped]
    best = min(pool, key=lambda r: r["llb_C"])
    say(f"best by band logloss: {best['variant']} "
        f"({best['llb_C']:.5f}, alpha={best.get('alpha')}, "
        f"hl={best.get('half_life_days')})")
    best_alpha = float(best.get("alpha") or tgr.ALPHA)
    best_half_life = float(best.get("half_life_days") or tgr.HALF_LIFE_DAYS)

    say("=== 4. likelihood: Poisson GLM at the winning penalty")
    results["likelihood"].append(variant(
        mods, arms, frame, "poisson (best alpha/hl)",
        alpha=best_alpha, half_life_days=best_half_life,
        likelihood="poisson"))

    say("=== 5. starter-aware design")
    for label, blocks, own_starter in STARTER:
        results["starter"].append(variant(
            mods, arms, frame, label, alpha=best_alpha,
            half_life_days=best_half_life, blocks=blocks,
            own_starter=own_starter))
    results["starter"].append(variant(
        mods, arms, frame, "teams + own starter only", alpha=best_alpha,
        half_life_days=best_half_life, blocks=(), own_starter=True))

    if not args.skip_seeds:
        say("=== 3. repeated seeds, all arms refit")
        for seed in SEEDS:
            touched = set_member_seed(mods, seed)
            say(f"seed {seed}: patched {touched} param dicts")
            ab = {"A": _oof(mods, arms["A"], frame),
                  "B": _oof(mods, arms["B"], frame)}
            panel = tgr.build_causal_panel(
                frame, alpha=best_alpha, half_life_days=best_half_life)
            frame_c = frame.copy()
            for column in tgr.RUN_RATING_COLS:
                frame_c[column] = panel[column].to_numpy()
            oof_c = _oof(mods, arms["C"], frame_c)
            results["seeds"].append(_measure(
                f"seed={seed}", _merge({**ab, "C": oof_c}),
                alpha=best_alpha, half_life_days=best_half_life, seed=seed))

    payload = {
        "frame": {"games": int(len(frame)),
                  "first": str(pd.to_datetime(frame["gameday"]).min().date()),
                  "last": str(pd.to_datetime(frame["gameday"]).max().date())},
        "arms": {k: len(v) for k, v in arms.items()},
        "band": list(BAND), "reps": REPS,
        "shipped": {"alpha": tgr.ALPHA, "half_life": tgr.HALF_LIFE_DAYS},
        "best": {"label": best["variant"], "alpha": best_alpha,
                 "half_life": best_half_life, "llb_C": best["llb_C"]},
        "results": results,
        "elapsed_minutes": round((time.time() - t_all) / 60, 1),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))

    lines = ["# MLB C-arm optimisation study", "",
             f"frame {payload['frame']}", "",
             "| study | variant | alpha | half-life | ll_all C | ll_band C | B->C band | 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    for study, rows in results.items():
        for row in rows:
            ci = (f"[{row['band_B_C_lo']:+.5f}, {row['band_B_C_hi']:+.5f}]"
                  if np.isfinite(row["band_B_C_lo"]) else "")
            lines.append(
                f"| {study} | {row['variant']} | {row.get('alpha', '')} | "
                f"{row.get('half_life_days', '')} | {row['ll_C']:.5f} | "
                f"{row['llb_C']:.5f} | {row['band_B_C']:+.5f} | {ci} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    say("wrote results.json and report.md")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
