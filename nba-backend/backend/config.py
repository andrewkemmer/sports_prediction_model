"""Central configuration for the standalone NBA production backend.

The ensemble, seed, fold cadence, warm-up, and member hyper-parameters are
intentionally copied from MLB's production contract.  Only the feature
representation and score/line grids are NBA-native.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DELIVERY_DIR = ROOT_DIR / "data_delivery"
# Training-process residue (OOF stores, fold table, RFE workbooks) is NOT
# delivery: nothing reads it back, so it writes to this local gitignored dir
# (NHL 2026-09-30 retention-audit pattern, adopted for parity 2026-10-04).
RUN_DIAGNOSTICS_DIR = ROOT_DIR / "run_diagnostics"
MODELS_DIR = DATA_DELIVERY_DIR / "models"
# The normalized cache is deliberately outside the repository.  It is never
# an input to git or to the frontend artifact resolver.  Respect an explicit
# NBA_CACHE_DIR first (Kaggle/CI can point it at a mounted volume); otherwise
# use the user cache rather than creating a hidden directory at the repo root.
CACHE_DIR = Path(
    os.environ.get("NBA_CACHE_DIR", "")
    or (Path.home() / ".cache" / "sports_prediction_model" / "nba")
).expanduser()

SPORT_DIR_NAME = "nba-backend"
REPO_SUBDIR = SPORT_DIR_NAME

RANDOM_SEED = 42
NUMPY_SEED = 42

# Historical eligibility starts with the 2024-25 season.  A season year is
# normalized to its starting year (2024 for 2024-25).
NBA_FIRST_SEASON = 2024
OOF_FIRST_SEASON = NBA_FIRST_SEASON
CORE_SEASONS = list(range(OOF_FIRST_SEASON, 2100))
ALL_SEASONS = CORE_SEASONS
GAME_TYPES = {1, 2, 3}
GAME_TYPE_REG = 1
GAME_TYPE_POST = 2

WARMUP_DAYS = 30
MIN_TRAIN_DAYS = WARMUP_DAYS
RETRAIN_CADENCE_DAYS = 7
MIN_VAL_FOLD_GAMES = 40

# MLB/NHL rating constants retained exactly for a common Elo state contract.
ELO_PRIOR = 1500.0
ELO_K = 20.0
ELO_SCALE = 400.0
ELO_HOME_ADV = 65.0
ELO_REVERT_FACTOR = 1 / 3

FORM_WINDOW = 5
WINPCT_WINDOW = 12
DEFENSE_WINDOW = 5
EWM_HALFLIFE = 3
OPP_ADJ_WINDOW = 6
PBP_ROLL_WINDOW = 5
RFE_COMMIT_SE_MULTIPLE = 1.0
RFE_NOISE_SIGMA = 1.0
RFE_MAX_STEPS = 120

# v2.4: the position SEGMENTS read the feed's full position listing - a
# forward-centre prices both the F and C segments, while the league prior
# keeps exactly one cell per player (the v2 positions cache). Columns are
# unchanged; the values of pl_epm_f_* and pl_epm_c_* move by design:
# collapsing every F-C into F had left whole rosters with no center segment
# (ORL starts two of them) and drove the monitor's 62% pl_epm_c_diff
# baseline LOW_COVERAGE. pl_epm_g_* is bit-identical to v2.3.
# v2.3: pl_epm replaces the TS family - Estimated Plus-Minus per 100
# participated possessions (NFL EPA-per-target analogue), same shrinkage,
# availability and opportunity-weighting structure (NFL f9d3e00 / MLB
# e3aa763 parity). The lineup aggregates blend each member's shrunk EPM
# by his own prior-possession total instead of a plain mean; availability
# inputs are unchanged and strictly point-in-time (designation archive +
# appearance recency gate).
#: v2.5: the position prior's collapsed cell follows the league's own primary
#: letter from ``playerindex`` (C-F is a centre who also plays forward) instead
#: of a local G->F->C convention that disagreed with the league for 28 of 582
#: players in 2025-26 - 17 of them starting centres. Coverage is unchanged and
#: provably so (the segments read the listing, which is byte-identical); what
#: moves is the prior, whose centre cell goes from 64 players to 81 and whose
#: shrinkage constant therefore rises from 445 to 487 plays.
#: v2.6: the pl_epm family is replaced by pl_rapm - player impact solved as
#: a ridge RAPM (ONE row per game: signed MIN/48 shares on both sides,
#: unregularized home-court intercept) and expressed in PER-GAME points
#: instead of per-100-possession EPM. The shrink layer weighs eff-games
#: (sum of share**2) toward the position mean of the SAME fit's betas, and
#: the team segments blend members by MINUTES PER GAME - the agreed
#: aggregation weight - rather than by prior possessions. Columns are
#: renamed pl_rapm_* / lineup_rapm_* throughout; historical notes below
#: keep the pl_epm spelling because that is what those versions published.
# v2.7 corrects Elo home advantage, measured efficiencies, pending-row
# isolation and causal evaluation. Rebuild features/OOF/refits together.
# v2.8 also makes RAPM roster membership, trades, recency and earliest-season
# shrinkage strictly prior. Existing v2.7 models require a synchronized rebuild.
FEATURE_SET_VERSION = "nba-prod-v2.8-rapm-strict-pregame"

#: The raw per-side metrics behind the diff contract. Every ``*_diff`` in the
#: list below is a home-minus-away comparison of a ladder statistic; the
#: families here publish that statistic's home and away values alongside the
#: difference, the way ``elo_home``/``elo_away`` and the ``event_*_diff``
#: family's structure (and the NFL ``pace_plays_min_home/_away`` pattern) do.
#: The mapping is ladder column -> published stem, declared once so the
#: builder, the contract and the provenance map cannot drift apart: a stem
#: named here gets ``{stem}_home``/``{stem}_away`` published, joins the served
#: contract, and is excluded from the linear (level-safe) view because a raw
#: side value is exactly the level feature the diff form exists to avoid.
PER_SIDE_SOURCES: dict[str, str] = {
    # Box-score form family (the ladder's shifted EWM state).
    "back_to_back": "back_to_back",
    "ewm_net_points": "ewm_net_points",
    "ewm_pace": "ewm_pace",
    "ewm_efg_pct": "ewm_efg_pct",
    "ewm_turnover_margin": "ewm_turnover_margin",
    "ewm_rebound_margin": "ewm_rebound_margin",
    "ewm_ast_per_game": "ewm_ast_per_game",
    # Play-by-play family: the per-side EWM behind each ``event_*_diff``.
    "three_rate_ewm": "event_three_rate",
    "rim_rate_ewm": "event_rim_rate",
    "live_tov_rate_ewm": "event_live_tov_rate",
    "and_in_rate_ewm": "event_and_in_rate",
    "shot_distance_ewm": "event_shot_distance",
    "possessions_ewm": "event_possessions",
    "shooting_fouls_ewm": "event_shooting_fouls",
    "q4_points_ewm": "event_q4_points",
}
#: Derived, not hand-written: 15 stems x (home, away). The diff columns stay
#: hand-listed above so the served order stays deliberate.
PER_SIDE_FEATURE_COLS: list[str] = [
    f"{stem}_{side}"
    for stem in PER_SIDE_SOURCES.values()
    for side in ("home", "away")]

CALIBRATION_MODE = "platt"
MIN_OOF_FOR_FIT = 300
CAL_GATE_HOLDOUT_FRAC = 0.25
CAL_GATE_MIN_HOLDOUT = 200
CAL_GATE_EPS = 5e-3

# One authoritative moneyline feature contract.  Every model, monitor,
# manifest, and run-line regressor projects this list.
#
# The eight ``event_*`` entries are the play-by-play contribution. They are
# named for what they measure, not for where they came from, so a second source
# of play-by-play can fill them without touching this list. All eight are
# home-minus-away differences, which is the form every comparable pair in this
# list takes; a level feature would let the model learn a team's standing
# rather than the matchup.
MONEYLINE_FEATURE_COLS = [
    "elo_diff", "win_pct_diff", "rest_days_diff", "back_to_back_diff",
    "ewm_net_points_diff", "ewm_off_rating_diff", "ewm_def_rating_diff",
    "ewm_pace_diff", "ewm_efg_pct_diff", "ewm_turnover_margin_diff",
    "ewm_rebound_margin_diff", "ewm_ast_per_game_diff", "is_playoffs",
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "ewm_off_rating_home", "ewm_off_rating_away",
    "ewm_def_rating_home", "ewm_def_rating_away", "rest_days_home",
    "rest_days_away",
    # Sides retained for every remaining diff: the raw home and away values
    # behind each difference above, so the board can show both teams' form and
    # not only the gap. Derived from PER_SIDE_SOURCES; each pair's diff is the
    # already-published ``*_diff`` column.
    *PER_SIDE_FEATURE_COLS,
    "is_home",
    # Play-by-play: where shots come from, which turnovers were live ball, how
    # much foul pressure the team generates, and how it scores late.
    "event_three_rate_diff", "event_rim_rate_diff",
    "event_live_tov_rate_diff", "event_and_in_rate_diff",
    "event_shot_distance_diff", "event_possessions_diff",
    "event_shooting_fouls_diff", "event_q4_points_diff",
    # Position-segmented projected-lineup impact - the NBA mirror of
    # MLB's lineup_woba_* trio, one per position, with sides retained:
    #   pl_rapm_{pos}_home / _away : the projected lineup's shrunk RAPM
    #                               at that position, home side / away side
    #   pl_rapm_{pos}_diff         : home minus away
    # Built over the WHOLE decided frame (MLB's lineup_agg construction),
    # strictly point-in-time on both axes: ratings sum only games STRICTLY
    # BEFORE each target, and a player designated Out/Doubtful/Recovery in
    # the last pre-tipoff report is removed from THAT game's pool only - his
    # rating survives on every prior game he actually played.
    "pl_rapm_c_away", "pl_rapm_c_home", "pl_rapm_c_diff",
    "pl_rapm_f_away", "pl_rapm_f_home", "pl_rapm_f_diff",
    "pl_rapm_g_away", "pl_rapm_g_home", "pl_rapm_g_diff",
]

#: The trailing statistics the play-by-play rollup contributes, and the window
#: each is read over. ``ewm`` means an exponentially weighted mean of the
#: per-game value, which is how every other rolling feature in the ladder is
#: built, so these need no new machinery.
EVENT_TRAILING_SPECS: dict[str, str] = {
    "three_rate": "ewm",
    "rim_rate": "ewm",
    "live_tov_rate": "ewm",
    "and_in_rate": "ewm",
    "shot_distance": "ewm",
    "possessions": "ewm",
    "shooting_fouls": "ewm",
    "q4_points": "ewm",
}

#: Columns the event ladder derives from the raw per-team rollup. Kept as a
#: mapping rather than computed inline so the rate's denominator is named once:
#: a rate whose denominator changes silently is a feature whose meaning changes
#: silently with it.
EVENT_RATE_DENOMINATORS: dict[str, str] = {
    "three_rate": "fga",
    "rim_rate": "fga",
    "live_tov_rate": "possessions",
    "and_in_rate": "possessions",
}

# Stable current-team categories.  The upstream numeric team IDs are
# resolved to these abbreviations during ingestion; unknown values receive a
# reserved category rather than being silently mapped to a real team.
NBA_TEAM_ID: dict[str, int] = {
    "ATL": 0, "BOS": 1, "BKN": 2, "CHA": 3, "CHI": 4, "CLE": 5,
    "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9, "HOU": 10, "IND": 11,
    "LAC": 12, "LAL": 13, "MEM": 14, "MIA": 15, "MIL": 16, "MIN": 17,
    "NOP": 18, "NYK": 19, "OKC": 20, "ORL": 21, "PHI": 22, "PHX": 23,
    "POR": 24, "SAC": 25, "SAS": 26, "TOR": 27, "UTA": 28, "WAS": 29,
}
# Every spelling of a team that any source may hand us, folded to the one the
# model uses.  The ESPN scoreboard is the schedule source, and it abbreviates
# six of the thirty teams differently from stats.nba.com does: ESPN says GS,
# NO, NY, SA, UTAH and WSH where the season log says GSW, NOP, NYK, SAS, UTA
# and WAS.  A join on team name therefore drops every game involving five of
# them - about a sixth of the schedule - with no error anywhere, because the
# unmatched side is simply absent rather than wrong.  ``test_nba_contract``
# pins these against the live scoreboard.
TEAM_ALIASES = {
    "WSH": "WAS", "NJN": "BKN", "NJ": "BKN", "NOH": "NOP", "SEA": "OKC",
    "VAN": "MEM", "CHH": "CHA", "SAN": "SAS", "PHX": "PHX",
    # ESPN's spellings.
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA",
}
UNK_TEAM_ID = 99
TREE_CATEGORICAL_COLS = ["home_team_id", "away_team_id"]


def normalize_team_abbr(value: object) -> str:
    text = str(value or "").strip().upper()
    return TEAM_ALIASES.get(text, text)


def team_category_id(abbr: object) -> int:
    return NBA_TEAM_ID.get(normalize_team_abbr(abbr), UNK_TEAM_ID)


# RFE candidates are a declared, PIT-safe trailing pool.  They are
# record-only by default and cannot silently mutate the served contract.
TEAM_CANDIDATE_TRAILING_SPECS: dict[str, tuple[str, ...]] = {
    "points_for_pg": ("ewm", "roll"),
    "points_against_pg": ("ewm", "roll"),
    "assists_per_game": ("ewm", "roll"),
    "rebounds_per_game": ("ewm", "roll"),
    "turnovers_per_game": ("ewm", "roll"),
    "three_point_pct": ("ewm", "roll"),
    "free_throw_pct": ("ewm", "roll"),
}
CANDIDATE_FAMILIES = {"nba": {"base": TEAM_CANDIDATE_TRAILING_SPECS}}
NBA_CANDIDATE_COLS = list(dict.fromkeys(
    f"nba_{metric}_{window}_{rep}"
    for metric, windows in TEAM_CANDIDATE_TRAILING_SPECS.items()
    for window in windows for rep in ("diff", "home", "away")
))
RAW_PER_SIDE_COLS = frozenset({
    "elo_home", "elo_away", "win_pct_home", "win_pct_away",
    "ewm_off_rating_home", "ewm_off_rating_away",
    "ewm_def_rating_home", "ewm_def_rating_away", "rest_days_home",
    "rest_days_away",
} | set(PER_SIDE_FEATURE_COLS)
  | {f"nba_{m}_{w}_{s}" for m, ws in TEAM_CANDIDATE_TRAILING_SPECS.items()
     for w in ws for s in ("home", "away")})
RFE_CANDIDATE_COLS = [c for c in NBA_CANDIDATE_COLS
                      if c not in set(MONEYLINE_FEATURE_COLS)]
_FEATURE_SUBSET: list[str] | None = None


def active_moneyline_feature_cols() -> list[str]:
    return list(_FEATURE_SUBSET) if _FEATURE_SUBSET is not None else list(MONEYLINE_FEATURE_COLS)


def set_feature_subset(cols: list[str]) -> None:
    global _FEATURE_SUBSET
    if not cols:
        raise ValueError("NBA feature subset cannot be empty")
    unknown = [c for c in cols if c not in KNOWN_FEATURE_COLS]
    if unknown:
        raise ValueError(f"unknown NBA feature columns: {unknown}")
    chosen = set(cols)
    _FEATURE_SUBSET = [c for c in KNOWN_FEATURE_COLS if c in chosen]


def reset_feature_subset() -> None:
    global _FEATURE_SUBSET
    _FEATURE_SUBSET = None


# Exact MLB-tuned member parameters and cold-start priors.
ENSEMBLE_MEMBERS = ["xgboost", "lightgbm", "elasticnet"]
ENSEMBLE_WEIGHTS = {"xgboost": 0.3333, "lightgbm": 0.3333, "elasticnet": 0.3334}
# Blend-policy governance (2026-10-01). NO min/max weights: the
# adaptive optimiser earns the plain simplex (w >= 0, sum = 1)
# exactly as MLB (training.compute_adaptive_weights), NHL and NFL
# (moneyline) do - "no floor, no cap: a member may earn 0% or 100%
# of the weight". NBA briefly diverged: the deepdive cap of 0.70 on
# the elastic net (adopted 2026-10-01) was removed the same day
# after the structural comparison showed NBA was the only backend
# with a policy box, and the cap made the served bundle stale
# (0.8256 served vs the 0.70 ceiling). Its measured cost was
# +1.53 bps pooled OOF logloss on the delivered frame (fold-faithful
# replay: 0.60626 -> 0.60623, essentially free; lower caps cost
# more: 0.60 +3.2 bps, 0.50 +8.0 bps). With the box empty the
# optimiser is free to put every point of weight on the best member
# (delivered OOF: elastic net 0.8256, lightgbm 0.1744, xgboost
# 0.0000) and the served weights are policy-consistent again. The
# box machinery in moneyline.compute_adaptive_weights stays dormant
# and tested (test_nba_ts_features.TestAdaptiveBlendPolicy) so a
# future cap/floor can be re-adopted through the same governance
# path: a constant fixed here, before any OOF evidence is seen, with
# its served cost measured, not assumed. The cap moves the reported
# percentage; it does not create information - only new features do
# that (deep-dive report section 7).
ENSEMBLE_MEMBER_CAPS: dict = {}
# No floors today: the optimiser may still zero a member
# (xgboost earned exactly 0.0000 on the delivered OOF).
ENSEMBLE_MEMBER_FLOORS: dict = {}
ADAPTIVE_WEIGHT_METRIC = "logloss"
BLEND_SPACE = "logit"
XGBOOST_PARAMS = {
    "max_depth": 3, "min_child_weight": 14, "gamma": 0.554715808992905,
    "subsample": 0.7429126143544483, "colsample_bytree": 0.475281865338598,
    "learning_rate": 0.03216955918004776,
    "reg_lambda": 1.6547035205533105, "reg_alpha": 0.002580235939504449,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss", "enable_categorical": True,
}
XGBOOST_FOLD_ROUNDS = 2000
XGBOOST_EARLY_STOP = 20
# Where xgboost's fold fits watch for early stopping: a held-out
# chronological TAIL OF THE TRAINING FOLD - the last 15% (>= 30 rows when
# the fold affords one, capped at half) - never the validation window.
# Those are the rows the fold is scored on, and letting them set the round
# count meant each fold's OOF metric graded a model whose early-stop
# decision had consulted that metric's own outcomes. The tail is strictly
# before val_start by construction, so the mechanic stays point-in-time
# for every fold. (MLB's fold fits still watch the val window; NBA
# diverges deliberately - see moneyline._early_stop_watch.)
XGBOOST_EARLY_STOP_FRAC = 0.15
XGBOOST_EARLY_STOP_MIN_ROWS = 30
# NBA-specific re-tune (2026-09-27, offline Optuna study over the production
# 41-fold walk-forward, 150 trials, .adhoc protocol per MLB's documented
# provenance): pooled member OOF logloss 0.63260 -> 0.62339 on folds[:-4], and
# the gain SURVIVED the sealed 4-fold holdout (0.52094 vs 0.53104, AUC .8475
# vs .8467) and the full population (0.61532 vs 0.62460). The tuned member
# still earns a zero final blend weight (elasticnet .4264 / xgboost .5736
# vertex unchanged), so this is member strength with the blend unharmed - the
# MLB member-strength policy, not a served-metric claim. The XGBoost member
# was tuned under the same protocol and REJECTED there: its pooled gain
# (+117 bps) reversed on the sealed holdout, so XGBOOST_PARAMS stays the MLB
# copy verbatim.
#
# RE-CONFIRMED from scratch on 2026-09-29 (.adhoc/nba_xgb_retune/VERDICT.md)
# after the member's blend weight collapsed 0.849 -> 0.770 -> 0.033 across
# three runs. That study measured the noise floor FIRST, which is what the
# 2026-09-27 protocol was missing: the same params under 5 seeds span
# 40.3 bps of pooled logloss and 190 bps within a single fold block, so a
# best-of-150 selection on pooled logloss is guaranteed to pick noise. The
# "+117 bps" that reversed on its sealed holdout was inside its own noise
# floor; the reversal was the correct outcome, not bad luck.
#
# The from-scratch retune (60 paired TPE trials -> 6 diverse finalists at
# 5/5 seeds and 4/4 blocks -> forward-chaining replay at 3 cut points) found
# a real effect: -121 bps paired, 5/5 seeds, all four blocks, and 3/3
# forward-chain cuts carried the gain onto unseen future folds. In-sample
# gains shrink to ~1/3 forward, so the honest figure is about -36 bps, not
# -121. Widening the bounds the first screen pinned against found no further
# gain, so the region is interior.
#
# It is still not adopted, because it does not change anything SERVED: the
# retuned member is 121 bps better and 2.2 AUC points better, and the
# production ensemble moves 1.2 bps with the weight still exactly 0.0.
#
# WHY THE WEIGHT DOES NOT MOVE - measured, not assumed. The blend's optimum
# is at xgboost weight 0.00 and adding it at ANY weight gains -0.0 bps,
# because the served model is very nearly a ONE-FEATURE model: a logistic
# regression on elo_diff ALONE scores ll 0.60858 / AUC 0.7239 against the
# full 71-feature blend's 0.60762 / 0.7268, so seventy features are worth
# 9.6 bps over one column, and dropping elo_diff costs 106 bps. The blend
# is not short of accuracy, it is short of information: retuning made
# xgboost a BETTER ELO RECONSTRUCTION, and there is nothing for a better
# Elo reconstruction to add. The optimiser is correct.
#
# PARTLY a scale artifact, found and fixed 2026-10-04. The 91%/94.4%
# elo_diff readings came from the importance report multiplying the elastic
# net's standardised |coef| by the feature's std a second time: elo_diff is
# in Elo points (std 126.8) against rates and EWMs elsewhere (std
# 0.006-11.5), so the widest column absorbed 68-94% of MODEL WEIGHT under
# that scaling. Per-SD - the units coef_ is actually fit in - the shipped
# elastic net puts elo_diff at ~17%, in line with the scale-blind trees'
# 12.97% (xgboost) and 15.91% (lightgbm) on 2026-09-29. The report has
# reported per-SD since 2026-10-04; monitor artifacts from before that
# date carry coef*std, not model behaviour. The predictive concentration is
# separately real (see the ablation above): elo family features still
# decide the model, just not at 91% of any honest importance measure.
#
# The remaining half of the old finding stands: the 91% column was also
# elasticnet's number reported as if it were the model's.
# feature_importance_weights normalises each member and averages by blend
# weight, and with both trees at 0.0 they are multiplied out before the
# sum. The monitor cannot distinguish "one member thinks this" from
# "the model IS this" without the decomposition - which now ships member
# shares and per-member profiles beside the column.
#
# The real next question is therefore not more tuning: it is where the
# non-Elo signal is and whether it is being extracted. The best non-Elo
# feature correlates 0.0799 with the Elo residual, and the top entries are
# the same signal twice - ewm_net_points/off/def_rating all read |corr|
# 0.0799 with opposite signs, since net points = off - def. That rejection
# was REVERSED by the 2026-10-01 full retune documented below: the old
# +117 bps was single-seed noise inside the 09-29-measured floor, while the
# retuned vector's gain is paired, seed-consistent and holdout-carried.
#
# FULL FROM-SCRATCH RETUNE OF BOTH TREE MEMBERS (2026-10-01, .adhoc protocol
# per the documented 2026-09-29 noise-floor discipline): every model
# parameter treated as free, 30-trial TPE screen on folds[:-4] over the
# production-identical frame (3,461 settled games, 41 folds), then paired
# seed verification (42/43/44) and the sealed folds[-4:] holdout. Machine
# stability: every fit ran pinned to a single thread (num_threads=1), so
# the per-fold logloss arrays are exactly reproducible; production is not
# pinned. Noise floor measured FIRST on the production params: xgboost
# spread 47.6 bps across seeds 42-46, lightgbm 14.8 bps - the 09-27 lesson
# (best-of-N on single-seed pooled logloss picks noise) enforced.
#
# Results. XGBoost: tune-set +30.8 bps, sealed holdout +25.8/+60.6/+18.1
# bps paired - it wins on every seed, reversing the 09-29 rejection
# (the old +117 bps that reversed then was noise; this +25..61 is paired
# and holdout-carried). LightGBM: tune-set +42.2 bps, sealed holdout
# +52.5/+58.0/+29.1 bps, all seeds, 63.4% fold win rate. The combined
# pair inside the real 3-member blend moves the SERVED numbers +3.9 bps
# logloss and +2.2 bps Brier (0.60698 -> 0.60659, 0.20970 -> 0.20948 over
# the full OOF) - the Elo-dominated-blend finding stands; this is member
# strength plus a small honest served gain, the member-strength policy.
#
# The retune harness, ledger (46 recorded runs), study db and verdict are
# preserved as untracked scratch: tune_full.py / tune_full_verdict.md /
# tune_full_runs.jsonl / tune_full_study.db.
#
# RE-REJECTED 2026-10-03 in the full from-scratch retune (see the LIGHTGBM
# block below for the protocol): 30 TPE trials over a WIDER space
# (colsample_bylevel / max_bin / grow_policy / max_leaves now free), top-3
# verified paired across seeds 42-44 on tune + sealed holdout. All three
# finalists lost -57 to -15 bps mean paired tune delta with fold win rates
# 0.36-0.47: the screen's apparent edge reversed under pairing, the same
# outcome 2026-09-29 recorded. XGBOOST_PARAMS stays exactly as the
# 2026-10-01 study left it.
#
# RETUNE 2026-10-03 (complete from-scratch reassessment, owner directive:
# every parameter free, machine-stable num_threads=1 on every fit). One TPE
# study per member, 30 trials xgboost / 29 lightgbm on folds[:-4] seed 42,
# top-3 per member verified across seeds 42/43/44 on BOTH the tune set and
# the sealed 4-fold holdout, paired against the incumbent at the same seed.
#
# Noise floor first: xgboost 24.7 bps, lightgbm 20.3 bps across seeds 42-46.
# XGBoost: REJECTED again. Its three finalists spanned -57 to -15 bps mean
# paired tune delta with fold win rates 0.36-0.47 - every gain the 30-trial
# screen showed reversed under pairing, exactly the 2026-09-29 pattern. The
# XGBOOST_PARAMS below are unchanged (still the 2026-10-01 pair that itself
# replaced the MLB copy).
#
# REJECTED A FOURTH TIME 2026-10-04 (complete from-scratch retune, owner
# directive: every parameter free, n_jobs=1 on every fit; harness
# scratch/nba_tune.py, frame rebuilt locally and verified EXACT against the
# shipped fold table; noise floor measured first: 5.3 bps tune / ~98 bps on
# the 29-row sealed holdout across seeds 42-46). Top-3 of a 30-trial TPE
# screen verified paired against the incumbent re-run at the same seed,
# seeds 42/43/44, tune + sealed. No finalist cleared BOTH binding gates:
# trial18 (screen-best) was served-positive (+2.1/+2.2/+0.8 bps logloss/
# calibrated/Brier) but missed pairing on seed44-sealed (-9.0 bps); trial27
# won paired on every seed on both populations (tune +28.3/+43.0/+40.0 bps
# against the 5.3 bps floor, sealed +309.7/+74.9/+141.6, fold win 0.567) but
# FAILED the served check (logloss -0.6, calibrated -3.3, Brier -0.4 bps;
# adaptive weight still ends 0.0). Member strength that cannot move the
# served blend in the right direction is not an adoption - the same two-gate
# discipline 2026-09-29 and 2026-10-03 applied. XGBOOST_PARAMS stands.
#
# LightGBM: ADOPTED (trial28 below), the only member whose finalists won
# paired on every seed on both populations. Tune-set deltas +11.7/+28.0/+28.5
# bps, sealed holdout +14.0/+29.8/+30.7 bps, fold win rate 0.565 - all inside
# or above the 20.3 bps noise floor, and consistent in sign across all six
# measurements. Inside the real 3-member blend on the identical frame
# (56 folds, seed 42): logloss 0.60582 -> 0.60557 (+2.5 bps), Brier 0.20930
# -> 0.20915 (+1.5 bps) against the incumbent re-run locally; against the
# shipped OOF the gap is +6.0/+2.5 bps. The xgboost leg of that blend earns
# no change, so this is member strength plus a small honest served gain.
#
# The new search spaces also widened the table the 10-01 study left
# unsearched: colsample_bylevel, max_bin, grow_policy and max_leaves for
# xgboost; max_bin and extra_trees for lightgbm. trial28 uses extra_trees
# and a 3-leaf deep-stump ensemble (max_depth 3, num_leaves 37) - a regime
# the old space could not express.
#
# Harness/ledger/study/verdict (untracked scratch, never commit):
# tune_full.py / tune_full_build_frame.py / tune_full_verdict.md /
# tune_full_runs.jsonl / tune_full_study.db / tune_full_finalists.json /
# tune_full_frame.pkl (the 2026-10-03 generation, wiped with the
# 2026-10-04 clean clone) - and for 2026-10-04:
# nba-backend/backend/scratch/nba_tune.py + scratch/{base.pkl, frame.pkl,
# runs.jsonl, tune_study.db, verify.json, serve_*.json, verdict.md}.
#
# FULL FROM-SCRATCH RETUNE 2026-10-04 (owner directive: reset every
# parameter and setting, machine-stable n_jobs=1 / num_threads=1 on every
# fit, selftest bit-identical before any result was trusted). Frame rebuilt
# from the warm local cache, zero network: 3,461 settled rows, 71 features,
# 56 folds, pl_rapm 3,272 attached, fold table EXACT match against the
# shipped nba_fold_table.csv on all 56 folds x 8 fields (sum n_validation
# 2400) - production-identical, so the tune transfers.
#
# Noise floor FIRST: xgboost 5.3 bps, lightgbm 14.6 bps across seeds 42-46
# on the pooled tune population; ~98-102 bps on the sealed 29-row holdout
# (its four windows are all provisional postseason, so the holdout grades
# EVERY row - evidence about the model, which the 2026-10-03 policy allows;
# the pooled mask would measure nothing there). 30-trial TPE per member,
# folds[:-4], seed 42; top-3 verified paired against the incumbent re-run
# at the same seed across seeds 42/43/44 on BOTH populations.
#
# LightGBM ADOPTED (trial17 below): the only finalist winning paired on
# every seed on both populations - tune +31.2/+2.2/+4.9 bps (mean +12.8;
# two deltas inside the 14.6 bps floor, sign consistent), sealed
# +129.2/+115.1/+87.2 bps (mean +110.5), fold win 0.508, tune AUC flat to
# better, sealed AUC +3.0 to +4.5 points. Inside the real 3-member blend
# (56 folds, seed 42): logloss 0.60207 -> 0.60193 (+1.4 bps), Brier
# 0.20758 -> 0.20752 (+0.6 bps) against the incumbent re-run locally.
# Member strength plus a small honest served gain - the same shape as the
# 2026-10-03 trial28 adoption this replaces (whose min tune delta, +11.7,
# was likewise inside its floor). Its space also searched
# feature_fraction_bynode (used at 0.3885).
LIGHTGBM_PARAMS = {
    # 2026-10-04 from-scratch winner (scratch/nba_tune.py, stage "verify"):
    # supersedes the 2026-10-03 trial28, which beat 2026-10-01 the same way.
    "n_estimators": 191, "max_depth": 3, "num_leaves": 85,
    "min_child_samples": 96, "min_gain_to_split": 0.04159703099643286,
    "bagging_fraction": 0.938750119313862, "bagging_freq": 2,
    "feature_fraction": 0.44095560633943776,
    "feature_fraction_bynode": 0.38853714661198313,
    "learning_rate": 0.03131184986120072,
    "extra_trees": True, "max_bin": 500,
    "reg_lambda": 0.004103712059528059,
    "reg_alpha": 0.0006433107716727887,
    "random_state": RANDOM_SEED, "verbose": -1,
}
ELASTICNET_PARAMS = {
    "penalty": "elasticnet", "l1_ratio": 0.5, "C": 0.03,
    "solver": "saga", "max_iter": 4000, "tol": 1e-4,
    "random_state": RANDOM_SEED,
}

XGBOOST_REG_PARAMS = {
    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_SEED,
    "verbosity": 0,
}
LIGHTGBM_REG_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "num_leaves": 12,
    "min_child_samples": 30, "learning_rate": 0.05, "subsample": 0.8,
    "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED, "verbose": -1,
}

# NBA line grids.  Lines are integer thresholds; half-point spread stops are
# represented by the explicit HALF_STOP_LINES namespace.  Totals use the
# dedicated p_push_total_<U> namespace to avoid any spread collision.
SPREAD_GRID = list(range(-20, 21))
TOTAL_GRID = list(range(180, 281))
HALF_STOP_LINES = [-0.5, 0.5]
RUN_ENGINE_FIXED_TOTALS = (220, 225, 230)
RUN_ENGINE_CANONICAL_SPREADS = (2, 5)
MARGIN_PMF_MAX = 60
TOTAL_PMF_MAX = 300
MARGIN_SIGMA = 12.0
TOTAL_SIGMA = 18.0

DATE_FMT = "%Y%m%d"
MONEYLINE_JSON = "nba_moneyline_v1_{date}.json"
CALIBRATION_JSON = "nba_calibration_{date}.json"
PREDICTIONS_HISTORY_CSV = "nba_predictions_history_{date}.csv"
POWER_RANKINGS_CSV = "nba_power_rankings_{date}.csv"
MARKETS_CSV = "nba_run_engine_markets_{date}.csv"
MARKETS_META_JSON = "nba_run_engine_markets_{date}.meta.json"
MARKETS_MONITOR_JSON = "nba_run_engine_monitor_{date}.json"
PLAYER_MATCHUP_JSON = "nba_player_leader_matchup_{date}.json"
PLAYER_RAPM_CSV = "nba_player_rapm_{date}.csv"
PLAYER_RAPM_AGG_CSV = "nba_player_rapm_lineups_{date}.csv"
FEATURE_JSON = "nba_feature_v1_{date}.json"
MODEL_MONITOR_JSON = "nba_model_monitor_{date}.json"
SHAP_GAME_PREFIX = "nba_shap_game"
MODEL_BUNDLE = MODELS_DIR / "nba_ensemble_latest.joblib"
OOF_STORE_CSV = RUN_DIAGNOSTICS_DIR / "nba_oof_store.csv"  # residue, not delivery
FEATURE_SELECTION_JSON = "nba_feature_selection_{date}.json"
FEATURE_WORKBOOK_XLSX = "nba_feature_workbook_{date}.xlsx"
RUN_ENGINE_FEATURE_DRIFT_PREFIX = "nba_run_engine_feature_drift_"
RUN_ENGINE_FEATURE_COVERAGE_PREFIX = "nba_run_engine_feature_coverage_"

# ---------------------------------------------------------------------------
# Player-level Regularized Adjusted Plus-Minus (RAPM) ratings
# ---------------------------------------------------------------------------
# Player-level RAPM ratings, per game.
#
# The NBA analogue of NFL's EPA-per-target, MLB's shrunk wOBA and
# NHL's shrunk player ratings: a player's impact in POINTS PER GAME,
# solved by ridge regression over ONE row per game - signed MIN/48
# participation shares on both sides plus an unregularized home-court
# intercept. The regression is what makes it ADJUSTED: every
# teammate's share is claimed by his own column, so the rating is his
# effect conditional on who he played with. The ridge is what makes
# it REGULARIZED. What makes the solved beta a *rating* is a second,
# smaller layer: each player's raw beta is pulled toward a
# POSITION-SEGMENTED league prior before it is used.
#
# A single league mean would be wrong here for the same reason NHL documents
# (nhl-backend/backend/config.py, PLAYER_RATING_SHRINK_FRACTION): a centre's
# court time and a guard's do not share an opportunity, so one prior would
# over-rate every big exactly as it over-rates every defenceman in hockey.
#
# Prior strength follows the SAME convention NHL carries over from MLB's fixed
# 120-PA prior. MLB's 120 is 20% of a 600-PA season, so the fraction is the
# portable part and the season length is sport-specific. The NBA reference
# season is one full player-SEASON of EFF-GAMES (sum of share**2, the
# participation weight the shrink layer itself measures in), and the prior is
# 20% of the mean player-season of eff-games at that position:
#     k[position] = PLAYER_RAPM_SHRINK_FRACTION * mean player-season eff-games
# Averaging per PLAYER-SEASON rather than per player is deliberate and is the
# same correction NHL documents: dividing by seasons first weights a 3-game
# cameo like an 82-game regular and collapses the reference season, which would
# silently make k several times too small and under-shrink every rating.
# Measured 2026-10-03 on the cached frame: the mean player-season sits near
# 13.3 eff-games, so 20% yields k = 2.4-3.0 per position (G 2.96, F 2.43,
# C 2.56).
#
# SEASON BOUNDARY (audit 2026-09-29, against MLB's structural guidance):
# the shrunk rating is a SEASON-CUMULATIVE quantity, and it follows MLB's
# season-to-date convention exactly (mlb features.py: season stats are
# PARTITIONED BY SEASON "so the prior October never leaks into a new
# season's cumulative"). Concretely:
#   * A PENDING-slate target before the season's first result rates from the
#     last completed season (the _evidence_season fallback) - the bridge the
#     2026-27 opening slate needed.
#   * From the first DECIDED game the solve is strictly in-season: opening
#     night itself has no in-season game before it, so the FIT falls back to
#     the last season that does (the strict _evidence_season reading) and
#     day 2+ ratings thin in from the league prior as eff-games accumulate.
#     That early-season thinness is the shrinkage doing its job, NOT missing
#     carryover - and it is the pl_rapm coverage the drift report reads as
#     STARVED/LOW for the first weeks (min-minutes pool floor on top).
#   * k is per TARGET SEASON, derived from completed prior seasons only -
#     NHL's season_ice_time_table as_of discipline (adopted 2026-10-01).
#     A season cannot tune its own shrinkage strength, and an in-progress
#     season holds PARTIAL player-seasons, so a whole-frame mean drags k
#     down as the season accumulates and silently weakens every rating's
#     prior weight game by game. A frame of completed seasons only (the
#     opening-slate build) is numerically identical to the old whole-frame
#     mean; the difference appears in-season and when rating a season the
#     frame itself contains. The frame's EARLIEST season has no completed
#     prior to measure against, so it uses PLAYER_RAPM_FALLBACK_K_EFF.
#     A whole-frame mean here would leak future games and seasons into k.
#     Opening slates absent from the player frame compute their own target-
#     season k from all completed priors, not the previous season's table. The position prior is no longer a
#     per-date cumulative table: it is the mean beta of that position's
#     evidenced members IN THE SAME FIT, so the prior and the rating it
#     shrinks can never disagree about scale or about date.
# MLB additionally ships CROSS-SEASON recent-form windows (last-5-start rolls
# over the prior season's tail - "no gap at the season boundary") and a
# shrink prior that falls back to all-history-through-window. The NBA has no
# cross-season recent-form analogue in the pl_rapm family: adopting one (or
# blending the prior-season tail into the first N games' prior) is a rating
# redefinition and needs its own holdout validation - recorded here so the
# divergence from MLB's structure is a decision, not an oversight.
PLAYER_RAPM_SHRINK_FRACTION = 0.20
#: The three positions stats.nba.com will actually answer for. Measured against
#: the live endpoint: the ``PlayerPosition`` filter accepts G, F and C, and
#: returns HTTP 400 for PG/SG/SF/PF and for compound codes like ``G-F``. The
#: five-way split is NOT available from this source, so the position prior is
#: carried at this granularity and the league averages are per G/F/C.
PLAYER_EPM_POSITIONS = ("G", "F", "C")
#: A player the feed lists at more than one position is assigned exactly one,
#: so every player belongs to exactly one prior cell and no rating is counted
#: twice in the league mean. Guards-and-forwards are real and common (52 of 569
#: players in 2024-25), so this tie-break is load-bearing rather than
#: theoretical. Most-specific-first, then narrowest-position-first, is the
#: order the enumeration is walked in.
PLAYER_EPM_POSITION_PRIORITY = ("G", "F", "C")
#: The fraction of a season's players that ``playerindex`` must cover before its
#: PRIMARY position is allowed to decide the collapsed cell; below it, every
#: player in that season keeps the ``PLAYER_EPM_POSITION_PRIORITY`` convention.
#: This is a coverage gate, not a tuning knob, and it exists because the
#: endpoint degrades silently rather than failing: measured live 2026-10-02 it
#: answers 2025-26 completely (582 of 582 filter players) but a BACK season
#: with about a quarter of the league (131 of 569 for 2024-25, 135 of 572 for
#: 2023-24), and it returns HTTP 200 either way. 0.90 sits far above the ~24% a
#: truncated back-season index measures and far below the 100% a live one
#: does, so the two are not confusable.
PLAYER_EPM_PRIMARY_MIN_COVERAGE = 0.90
#: RAPM ridge strength (lambda) on the player columns of the signed
#: single-row design; the home-court intercept column is NEVER ridged.
#: Measured 2026-10-03 on the cached 2-season frame (.adhoc/rapm_prototype.py):
#: candidates 4/8/16, and 16 was chosen by descriptive player rankings,
#: NOT held-out win prediction. The paired predictive audit must justify
#: retention; ranking recognizable stars is not an admission gate.
#: In that descriptive check, 16 surfaces stars at the top
#: (Wemby +5.2, Haliburton +3.9, Curry +3.6) while fringe players sink to
#: -6..-7 and thin players (<=5 games) sit at mean|beta| 0.39 versus 1.64
#: for regulars (>=20 games) - the shrinkage working through the ridge.
PLAYER_RAPM_LAMBDA = 16.0
#: Fallback shrinkage strength k (EFF-GAMES) for a position cell with no
#: evidence at all. Measured 2026-10-03 on the cached frame: mean
#: player-season eff-games (sum of share**2) is 13.3, so the 20% fraction
#: yields 2.5-2.8 per position; 2.6 is the rounded common value. A cell
#: with no data must still yield a complete prior table rather than raise
#: at lookup time.
PLAYER_RAPM_FALLBACK_K_EFF = 2.6
#: Minimum accumulated MINUTES before a player may be PROJECTED into a
#: lineup, mirroring MLB's LINEUP_MIN_PA discipline. The NBA number is 50:
#: the minutes equivalent of the 120-possession floor this replaced.
#: plays = sum(share * team possessions) with share = MIN/48 makes plays
#: proportional to minutes at a measured median 2.38x (p10 2.29, p90 2.46),
#: and on the cached 2025-26 frame at mid-season minutes >= 50 admits the
#: SAME players as plays >= 120 at 99.8% agreement (0 extra, 1 missed) -
#: about two player-games of rotation. The reasoning is the same, and because
#: the shrinkage already handles the RATING - a 50-minute player is still
#: pulled most of the way to the prior. This floor is about POOL MEMBERSHIP,
#: which shrinkage does not touch: a player with three career games is not a
#: candidate for tonight's starting five no matter how well his rate is
#: estimated.
PLAYER_RAPM_MIN_MINUTES = 50
# Recency gate for the projected-lineup pool (availability audit 2026-09-28):
# a rating row whose season evidence is older than this is a PHANTOM - a
# player who stopped appearing (injury never filed, quietly shut down, or a
# roster cut) but keeps riding the pool on stale evidence, because rows are
# emitted per target date for every player who ever appeared in the season.
# The audit's worst case sat in the Clippers' pool 217 days after his last
# game (15.13% of rotation pool rows league-wide). A NaN gap (season not
# started for the player) stays eligible - the season-start carryover the
# min-minutes floor already governs.
PLAYER_EPM_RECENCY_DAYS = 30
#: How far back a TEAM MEMBER's rating row may sit and still count as a
#: candidate for the next game, mirroring MLB's LINEUP_POOL_LOOKBACK_DAYS.
#: This is the WIDENING that makes the injury filter bind at all: a player who
#: is injured today has no row for today, so a pool drawn only from players who
#: appeared recently cannot subtract him. MLB states the consequence plainly -
#: "the IL filter cannot bind AT ALL". Both the rating rows and the games live
#: on dates the team played, so ten days spans one skipped game plus a
#: postponement.
PLAYER_EPM_POOL_LOOKBACK_DAYS = 10
#: Size of the projected lineup, mirroring MLB's top-9. A depleted roster is
#: NOT padded: the mean of the best 5-7 healthy players is the correct quantity
#: for a short-handed team, and padding would fabricate full strength.
PLAYER_EPM_TOP_K = 8
#: The "regular" floor for a top-5 rest count, mirroring MLB's
#: LINEUP_REST_PA discipline: 126 accumulated MINUTES, the minutes
#: equivalent of the 300-possession floor this replaced (300 / 2.38, and
#: minutes >= 126 agrees with plays >= 300 at 99.6% on the cached frame) -
#: about five player-games. A player below it is not a rotation regular, so
#: his absence is not news.
PLAYER_RAPM_REST_MINUTES = 126
PLAYER_EPM_TOP5_K = 5
#: Raw status vocabulary -> treatment, and nothing else. A status the feed
#: invents that is not listed here is reported as UNKNOWN rather than guessed,
#: because guessing silently decides whether a player dresses.
#:
#: MEASURED 2026-09-26 across all 30 teams and 545 athletes: ESPN's NBA roster
#: publishes exactly TWO statuses, "Day-To-Day" (52) and "Out" (7). There is no
#: "Doubtful", "Questionable" or "Probable" - those are NFL injury-report
#: words, and importing them into the NBA mapper is how a vocabulary we do not
#: have ends up encoded as if we did. They are listed below only so that IF one
#: ever appears it is handled deliberately rather than falling through to
#: healthy, and so the play-rate reporter can measure it once it does.
PLAYER_EPM_STATUS_TREATMENT: dict = {
    # Measured against the 2025-26 box scores, 13,168 designations taken from
    # the last filing before each tipoff. The grouping is the play rate, not
    # the word: Doubtful went 0-for-5 and belongs with Out.
    #
    # "Out" also carries the injured reserve. The league does not name the IR
    # in the report at all - "Injured Reserve" appears in zero reason strings -
    # because a player on the IR is filed as Out like anyone else. Adding it to
    # the out set is therefore free, and enumerating it separately would invent
    # a category the source does not publish.
    #
    # NOTE what is NOT here: ``day_to_day``. That is ESPN's word for this
    # league, not the NBA's, and carrying it invites a mapper to quietly treat
    # an ESPN snapshot and an official filing as the same vocabulary. They are
    # not, and the official one is strictly richer.
    "out": "absent",
    "doubtful": "absent",        # 0 for 5
    "recovery": "absent",        # UNMEASURED - 0 observations at tipoff
    "questionable": "available",  # 34 for 52 - thin, pooled, see below
    "available": "available",    # 2,006 for 2,448
    "probable": "available",     # 23 for 24
}

#: MEASURED play rate per designation, from the official 2025-26 filings
#: aligned to tipoff. This replaces NHL's unsourced ``INJURY_MULTIPLIERS``,
#: whose day-to-day value of 0.5 was an upstream assumption wearing a number.
#: Reproduce with ``backfill_injury_designations.py 2025-10-21 2026-04-12``.
#:
#: Each figure is the rate for the BUCKET the designation now belongs to, not
#: for the designation alone. The available bucket reads 0.817, which is
#: (2,006 + 34 + 23) / (2,448 + 52 + 24) - Available, Questionable and Probable
#: pooled. Weighting a designation by a rate drawn from 52 observations would be
#: fitting a parameter to 0.4% of a season; pooling gives up that resolution
#: and buys an estimate that means something.
#:
#: The sample sizes matter more than the values. Doubtful rests on five
#: observations and Recovery on none, so neither is a rate so much as an
#: assumption that happens to be written as a float. Out and Available, which
#: carry 98% of the mass, are solid.
PLAYER_EPM_DESIGNATION_PLAY_RATE: dict = {
    "out": 0.000,            # 0 / 10,639
    "doubtful": 0.000,       # 0 / 5
    "recovery": 0.000,       # no observations at tipoff
    "questionable": 0.817,   # pooled available bucket
    "available": 0.817,      # 2,063 / 2,524 pooled with the above
    "probable": 0.817,       # 23 / 24 pooled with the above
}
#: The point-in-time cutoff for injury state. "tipoff" uses each record's own
#: publication time and is the finest granularity the feed supports for free.
#: "prior_end_of_day" is the coarser, more conservative reading: for a game on
#: date D, the state is whatever was published by 23:59 on D-1, so nothing
#: published on game day can gate that game at all.
#:
#: "tipoff" is the default because it is strictly more informative and costs
#: nothing extra - the timestamp is already on every record, and using it is
#: the SAME comparison with a tighter bound rather than a different mechanism.
#: The coarser mode is one constant away for anyone who would rather not have
#: a same-day 18:00 report gate a 19:30 game.
PLAYER_EPM_INJURY_CUTOFF = "tipoff"   # or "prior_end_of_day"

#: The nine position-segmented projected-lineup features - per position, the
#: away side, the home side and the difference. Registered as CANDIDATES, not
#: as contract columns: they are gated behind their own holdout A/B, which is
#: ``ab_position_rapm.py``. Adding a column to MONEYLINE_FEATURE_COLS is a
#: decision about the model, and the candidate list is where a feature waits
#: until that decision has evidence behind it.
#:
#: Defined HERE rather than beside RFE_CANDIDATE_COLS because it is derived
#: from PLAYER_EPM_POSITIONS, which is declared further down. Building the list
#: at the top of the module raised NameError on import.
#: Published contract order is per-POSITION grouped - ``c_away, c_home,
#: c_diff``, then ``f_*``, then ``g_*`` - C first, matching the user-facing
#: feature list. ``PLAYER_EPM_POSITIONS`` stays G/F/C for the prior tables;
#: the published column order deliberately does not inherit that internal
#: ordering.
PLAYER_RAPM_POSITION_FEATURE_COLS: list[str] = [
    f"pl_rapm_{position}_{side}"
    for position in ("c", "f", "g")
    for side in ("away", "home", "diff")]

KNOWN_FEATURE_COLS = list(dict.fromkeys(
    MONEYLINE_FEATURE_COLS + RFE_CANDIDATE_COLS
    + PLAYER_RAPM_POSITION_FEATURE_COLS))
#: How far the injury-stint table may trail the decided frame before the
#: staleness tripwire fires, mirroring MLB's IL_STINT_MAX_LAG_DAYS. Offseason
#: legitimately leaves a long gap with no absences recorded, so the bar is
#: generous rather than a day.
PLAYER_EPM_MAX_LAG_DAYS = 45
#: Team-level projected-lineup aggregates, mirroring MLB's lineup_woba_* trio
#: plus its rest count.
PLAYER_RAPM_FEATURE_COLS: list[str] = [
    "lineup_rapm_mean_home", "lineup_rapm_mean_away",
    "lineup_rapm_top3_home", "lineup_rapm_top3_away",
    "lineup_rapm_std_home", "lineup_rapm_std_away",
    "lineup_rapm_rest_count_home", "lineup_rapm_rest_count_away",
]

PLAYER_FIELDS = [
    "p_home_name", "p_home_ppg", "p_home_apg", "p_home_games",
    "p_away_name", "p_away_ppg", "p_away_apg", "p_away_games",
]
PLAYER_WINDOW_GAMES = 5
PLAYER_MIN_GAMES = 3
PLAYER_MIN_MINUTES = 15.0
COIN_FLIP_THRESHOLD = 0.02

RFE_FORCE_ENV = "NBA_RFE_FORCE"
RFE_ADDS_ENV = "NBA_RFE_ADDITION_MONEYLINE_LIST"
RFE_REMOVES_ENV = "NBA_RFE_REMOVAL_MONEYLINE_LIST"
