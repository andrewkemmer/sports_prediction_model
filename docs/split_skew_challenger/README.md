# Split-skew Gaussian distribution challenger

## Model and delivery contract

- Preserve each sport's existing point-in-time, market-free score-mean regressors and feature routing. A Poisson **regression objective** is not the retired distribution sampler.
- Learn home/away centered residual variance per expected score and residual correlation from prior OOF forecasts. Total and margin variances are the variance sum plus/minus twice the covariance.
- Use a continuous-density two-piece Gaussian with separate left/right widths, centered to preserve the forecast mean and normalized to preserve the requested variance. Learn total and margin asymmetry separately. Truncate totals to the nonnegative integer lattice; settle NBA/MLB/NHL zero-margin mass equally into adjacent winning margins; retain genuine NFL tie mass.
- Fit nine-coefficient, conditional left/right logit-slope calibration by discrete CRPS. One increasing CDF map prices every threshold; no independent per-line maps or cross-line incoherence. Admit maps only on negative paired CRPS differences from earlier issued origins.
- Freeze variance fitting, calibration fitting, and admission evidence before the last 21 calendar days of the OOF evaluation. Serving can refit all settled historical OOF forecasts, but never regrade them with that final map.
- Monte Carlo inverse-samples the **calibrated** CDF, 50,000 IID draws per game/market by default. Identity-keyed BLAKE2 streams preserve seeds under shuffling/batching. Each game's empirical PMF supplies win/push/loss legs, moneyline, and fair half-point lines. The analytical CDF is used for fitting/admission, not as a substitute for delivered Monte Carlo prices.
- Bound draw memory to one game at a time; guarantee worst Bernoulli standard error <=0.005 by raising smaller requests to 10,000 draws. This is a **standard-error** bound, not an absolute-error guarantee. Reject invalid scores, dates, overlapping origins, means, parameters, and oversized integration support.
- Keep market artifact filenames and probability column namespaces. Model/calibration metadata identifies `causal_split_skew_crps_mc` and `two_sided_asymmetric_gaussian_split_skew`. Refuse incompatible old calibration bundles rather than silently serving an obsolete model.
- Plot the shipped empirical PMF directly. Do not fit a second distribution for dashboard smoothing.

## NBA verification

- Full NBA backend suite: 631 passed before the additional five mathematical/serving tests; those five also passed.
- Focused frontend suites: 82 passed, one failed because the existing MLB artifact lacks its model-card per-line metrics. NBA page and model-card AppTest rendering passed, including real challenger replay artifacts and 50,000-draw disclosure.
- Replay: 2,400 delivered OOF forecasts, 56 persisted origins; 103.52 seconds on this checkout. Total CRPS 10.76171, margin CRPS 8.22127; derived moneyline logloss 0.607892 and Brier 0.209758. See `nba_replay.json` for fixed-line, calibration, parameter, and Monte Carlo checks.
- Worst sampled analytical-vs-empirical CDF discrepancy across representative beginning/middle/end origins: 0.005867. This is consistent with stochastic sampling error; it is not bounded by one standard error.
- Public path checked: OOF replay -> markets CSV/meta -> monitor JSON -> Streamlit fit/model-card rendering. Full input ingestion and score-mean retraining were not rerun. Published historical artifacts were not rewritten; the normal next pipeline run emits the new family.

## MLB / NFL / NHL port verification

- Structure mirrors NBA exactly (same two-piece CDF, nine-coefficient CRPS calibration, admission on negative paired CRPS differences, 21-day sealed tail, identity-keyed 50,000-draw inverse-CDF Monte Carlo); each sport fits its own home/away residual variance, correlation and total/margin skew from prior OOF. NFL keeps genuine tie mass; the other three settle zero-margin mass into adjacent margins.
- Replays of delivered OOF means: MLB 7,033 rows / 84 persisted origins (120.96 s), total CRPS 2.46447, run-line CRPS 2.46745, derived-ML logloss 0.683504; NFL 2,572 rows / 200 origins (202.76 s), total CRPS 7.63846, run-line CRPS 7.38265, derived-ML logloss 0.629606 over 9 ties (conditional non-tie settlement); NHL 2,686 rows / 60 origins (26.05 s), total CRPS 1.29001, run-line CRPS 1.44244, derived-ML logloss 0.677175. See `mlb_replay.json`, `nfl_replay.json`, `nhl_replay.json`.
- Worst sampled analytical-vs-empirical CDF discrepancy: MLB 0.005362, NFL 0.005619, NHL 0.005644 — sampling noise at the shipped draw count, not a pricing guarantee.
- Suites after the port and the stale-label cleanup: MLB 403 passed / 1 skipped (pre-existing), NFL 55 passed, NHL 625 passed, NBA 636 passed, frontend 118 passed (all four markets pages rendered through AppTest against real artifacts).

These are historical replay measurements, **not** proof of prospective improvement, betting profitability, or an optimized advantage over the retired model. Sport-specific parameter estimation optimizes conditional CRPS using prior data; no holdout metric is used for tuning.
