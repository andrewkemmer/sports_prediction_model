"""Frame + market-price builders for the moneyline delayed-start A/B.

Self-contained copy of the pieces of ``ab_mean_offset`` / ``ab_c_sweep`` that
the Kaggle kernels need, so a kernel only has to clone the public repo (the
helpers themselves are not tracked in it).

Arm A: incumbent moneyline model, full history, no market input.
Arm B: the same members with the de-vigged market logit as ``base_margin``,
       trained on games from the start of the 2025-2026 season only.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPORTS = ("nba", "nhl")

#: Trial-only team-level DARKO rating: five-player net DPM, i.e.
#: ``5 * (team_off + team_def)`` from the projected healthy lineup. The
#: production crossmatch legs already publish ``team_off * s`` and
#: ``team_def * s`` (``s = projected_game_pace_total / 100``), so the
#: unscaled team rating is recoverable exactly from those columns. This
#: family is registered by the experiment runner at run time; it is not a
#: production contract column.
TEAM_DARKO_COLS = ["team_darko_diff", "team_darko_home", "team_darko_away"]


def _attach_team_darko(game_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the three trial team-DARKO columns to an experiment frame.

    DARKO defines a team's rating as its players' DPM weighted by projected
    minutes, scaled to the five players on the floor (points per 100).
    ``team_off`` and ``team_def`` are exactly those minutes-weighted means
    over the projected healthy roster; both are positive-is-better, so the
    net rating is their sum, times five. The production crossmatch legs
    publish each mean multiplied by the projected-game pace scaler ``s``;
    dividing their sum by ``s`` recovers the unscaled rating. A missing or
    non-positive pace scaler is an honest NaN, never a fabricated value.
    """
    out = game_df.copy()
    pace = pd.to_numeric(
        out.get("projected_game_pace_total",
                pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ) / 100.0
    for side in ("home", "away"):
        off = pd.to_numeric(
            out.get(f"pl_o_dpm_{side}", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        dfn = pd.to_numeric(
            out.get(f"pl_d_dpm_{side}", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        total = off + dfn
        out[f"team_darko_{side}"] = np.where(
            np.isfinite(pace) & (pace > 0) & np.isfinite(total),
            5.0 * total / pace,
            np.nan,
        )
    out["team_darko_diff"] = (out["team_darko_home"]
                              - out["team_darko_away"])
    return out


def load_modules(sport: str) -> types.SimpleNamespace:
    """Import one sport's backend. The two backends are mirrors; never both."""
    if sport not in SPORTS:
        raise SystemExit(f"unknown sport {sport!r}")
    backend = ROOT / f"{sport}-backend" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"backend not found: {backend}")
    sys.path.insert(0, str(backend))
    import config  # noqa: E402
    import distributions  # noqa: E402
    import folds  # noqa: E402
    import features  # noqa: E402
    import moneyline  # noqa: E402

    return types.SimpleNamespace(config=config, dist=distributions, folds=folds,
                                 feat=features, ml=moneyline, backend=backend)


def _attach_projected_lineup(game_df: pd.DataFrame, facts, games: pd.DataFrame,
                             config) -> pd.DataFrame:
    """Attach the production projected-lineup block and enforce its gate.

    ``features.build_game_features`` creates every contract column, including
    the twenty-one projected-lineup ones, as NaN. Production fills them later
    in ``master_pipeline``; a harness that stops at ``build_game_features``
    therefore looks complete (98 columns) while carrying none of the player
    availability, expected-minutes, crossmatch or expected-points signal.

    This calls the production builder itself rather than re-implementing it,
    then applies the same ``_empty_contract_columns`` gate the pipeline runs
    before training. A missing block is a failed experiment, never a silently
    reduced contract.
    """
    import master_pipeline as mp  # production assembly, one path

    frame, _slate = mp._build_position_rapm_features(facts, games)
    if frame is None or not len(frame):
        raise SystemExit(
            "projected-lineup build returned no decided rows; refusing to "
            "train on a frame whose availability block is empty")
    cols = [*config.PLAYER_RAPM_POSITION_FEATURE_COLS,
            *config.STAR_OUT_FEATURE_COLS,
            *config.O_PTS_FEATURE_COLS]
    frame = frame.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.drop_duplicates("game_id", keep="last")
    game_df = game_df.drop(columns=[c for c in cols if c in game_df.columns])
    game_df["game_id"] = game_df["game_id"].astype(str)
    game_df = game_df.merge(frame[["game_id"] + cols], on="game_id",
                            how="left", validate="1:1")
    attached = int(game_df[config.STAR_OUT_FEATURE_COLS[0]].notna().sum())
    print(f"projected-lineup features attached: {attached}/{len(game_df)} "
          f"decided rows", flush=True)
    game_df = _attach_team_darko(game_df)
    for column in TEAM_DARKO_COLS:
        print(f"  {column}: {game_df[column].notna().mean():.1%} coverage",
              flush=True)
    gaps = mp._empty_contract_columns(game_df)
    if gaps:
        raise SystemExit(
            "experiment frame feature coverage failed (the production gate): "
            f"{gaps}")
    return game_df


def build_frame(mods: types.SimpleNamespace, sport: str) -> pd.DataFrame:
    """The training frame, built with the same calls each pipeline makes."""
    config, feat, folds_mod = mods.config, mods.feat, mods.folds
    import ingestion

    if sport == "nba":
        facts = ingestion.load_ingested(allow_download=True)
        settled = ingestion.trainable_games(facts.games)
        game_df = feat.build_game_features(settled, facts.team_stats,
                                           facts.team_events)
        game_df = _attach_projected_lineup(game_df, facts, settled, config)
    else:
        seasons = list(getattr(config, "CORE_SEASONS",
                               range(config.OOF_FIRST_SEASON, 2027)))
        dates = sorted({d for s in seasons for d in ingestion.season_dates(s)})
        schedule = ingestion.eligible_games(ingestion.load_score_dates(dates))
        schedule["gameday"] = pd.to_datetime(schedule["game_date"],
                                             errors="coerce")
        floor = pd.Timestamp(f"{config.OOF_FIRST_SEASON}-09-01")
        schedule = schedule[schedule["gameday"] >= floor].copy()
        decided = schedule[schedule["home_score"].notna()
                           & schedule["away_score"].notna()].copy()
        ids = decided["game_id"].astype(str).tolist()
        gameday_by_id = dict(zip(ids, pd.to_datetime(decided["gameday"],
                                                     errors="coerce")))
        boxes = ingestion.load_boxscores(ids, gameday_by_id=gameday_by_id)
        ratings = feat._load_player_ratings()
        excl, excl_src = feat._combined_exclusions(ratings)
        game_df = feat.build_game_features(
            decided, boxes, player_ratings=ratings, stints=excl,
            stints_sources=excl_src)

    game_df = folds_mod.canonical_sort(game_df, "gameday").reset_index(drop=True)
    if "gameday" not in game_df:
        raise SystemExit("feature frame has no gameday")
    game_df["gameday"] = pd.to_datetime(game_df["gameday"], errors="coerce")
    game_df = game_df[game_df["gameday"].notna()].reset_index(drop=True)
    return game_df


def _de_vig(home_odds, away_odds) -> np.ndarray:
    """Proportional de-vig; NaN where the quote is unusable."""
    def _arr(v):
        s = pd.to_numeric(v, errors="coerce")
        return np.asarray(getattr(s, "to_numpy", lambda *_: s)(float),
                          dtype=float)
    h, a = _arr(home_odds), _arr(away_odds)
    with np.errstate(divide="ignore", invalid="ignore"):
        ih, ia = 1.0 / h, 1.0 / a
        ov = ih + ia
    ok = (np.isfinite(ih) & np.isfinite(ia) & (h > 1.0) & (a > 1.0)
          & (ov >= 1.0) & (ov <= 1.10))
    return np.where(ok, ih / np.where(ok, ov, np.nan), np.nan)


def load_lines(sport: str) -> pd.DataFrame:
    """Closing-line proxy per game: the last TAB update of the stored history."""
    path = ROOT / f"{sport}-backend" / "data_delivery" / f"{sport}_odds_history.csv"
    if not path.exists():
        raise SystemExit(f"odds history not found: {path}")
    d = pd.read_csv(path)
    need = {"date", "home", "away", "home_line_agg", "total_line_agg"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")

    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
    d = d.dropna(subset=["date", "home_line_agg", "total_line_agg"])
    d["gameday"] = (d["date"].dt.tz_convert("America/New_York")
                    .dt.normalize().dt.tz_localize(None))
    keep = ["gameday", "home", "away", "home_line_agg", "total_line_agg"]
    for extra in ("tab_homeodds", "tab_awayodds"):
        if extra in d.columns:
            keep.append(extra)
    if "tab_last_updated" in d.columns:
        d["tab_last_updated"] = pd.to_datetime(d["tab_last_updated"],
                                               errors="coerce", utc=True)
        d = d.sort_values(["home", "away", "gameday", "tab_last_updated"])
        keep.append("tab_last_updated")
    else:
        d = d.sort_values(["home", "away", "gameday"])

    if sport == "nhl":
        sys.path.insert(0, str(ROOT / "nhl-backend" / "backend"))
        try:
            from market_offset import FROM_SI  # noqa: E402
            d["home"] = d["home"].replace(FROM_SI)
            d["away"] = d["away"].replace(FROM_SI)
        except Exception:  # noqa: BLE001 - vocabulary is already model-side
            pass

    d = d.drop_duplicates(["gameday", "home", "away"], keep="last")
    return d[keep].rename(columns={"home": "home_team", "away": "away_team"})


def attach_anchors(df: pd.DataFrame, lines: pd.DataFrame,
                   sport: str) -> pd.DataFrame:
    """Join the closing lines and derive the per-side anchors."""
    df = df.copy()
    df["gameday"] = pd.to_datetime(df["gameday"]).dt.normalize()
    out = df.merge(lines, on=["gameday", "home_team", "away_team"], how="left")
    out["m_t"] = pd.to_numeric(out["total_line_agg"], errors="coerce")
    out["m_m"] = -pd.to_numeric(out["home_line_agg"], errors="coerce")

    if sport == "nhl":
        p = _de_vig(out.get("tab_homeodds", np.nan),
                    out.get("tab_awayodds", np.nan))
        out["market_p_home"] = p
        z = np.full(len(out), np.nan)
        ok = np.isfinite(p) & (p > 1e-6) & (p < 1 - 1e-6)
        from scipy.stats import norm
        z[ok] = norm.ppf(p[ok])
        out["m_m"] = np.nan
        margin = (pd.to_numeric(out["home_score"], errors="coerce")
                  - pd.to_numeric(out["away_score"], errors="coerce")).to_numpy(float)
        months = out["gameday"].dt.to_period("M")
        for period in sorted(months.dropna().unique()):
            past = (months < period).to_numpy() & np.isfinite(margin)
            cur = (months == period).to_numpy()
            if past.sum() >= 200 and np.isfinite(z[past]).sum() >= 200:
                zz = z[past]
                yy = margin[past]
                okp = np.isfinite(zz) & np.isfinite(yy)
                sigma = (float(np.dot(zz[okp], yy[okp]) / np.dot(zz[okp], zz[okp]))
                         if np.dot(zz[okp], zz[okp]) > 1e-9 else np.nan)
            elif past.sum() >= 50:
                sigma = float(np.nanstd(margin[past]))
            else:
                sigma = np.nan
            if np.isfinite(sigma) and abs(sigma) > 1e-6:
                out.loc[cur, "m_m"] = (sigma * z[cur])

    out["m_h"] = (out["m_t"] + out["m_m"]) / 2.0
    out["m_a"] = (out["m_t"] - out["m_m"]) / 2.0
    return out


def build_moneyline_frame(mods, sport: str) -> pd.DataFrame:
    """Feature frame plus the de-vigged offset series for one sport."""
    frame = build_frame(mods, sport)
    if sport == "nba":
        import odds_history
        frame, coverage = odds_history.attach_prices(frame)
    else:
        frame = attach_anchors(frame, load_lines(sport), sport)
        coverage = float(frame["market_p_home"].notna().mean())
    print(f"market offset coverage {coverage:.1%}", flush=True)
    if coverage < 0.5:
        raise SystemExit("market offset coverage below 50%: the residual arm "
                         "would be fitted on a biased subset")
    p = pd.to_numeric(frame["market_p_home"], errors="coerce").to_numpy(float)
    frame["market_logit"] = np.log(np.clip(p, 1e-6, 1 - 1e-6)
                                   / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return frame
