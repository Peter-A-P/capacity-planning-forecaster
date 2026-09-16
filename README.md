# Capacity Planning Forecaster

Demand forecasts, emergency calls and emergency-department arrivals, with uncertainty
ranges that actually hold when conditions shift, turned into a staffing number at a chosen
service level. Overstaffing costs money; understaffing costs patients. This puts a
defensible number on both, site by site, rolling up to the region.

**Status: building,** week 1 of two, started 2026-09-12. The plan is in
[PLAN.md](PLAN.md): a two-week build on public demand series, everything on a desktop CPU.

The data is loaded and checked ([docs/data.md](docs/data.md)). The baseline, the conformal
intervals and the statistical models (ETS, Theta, MSTL and AutoARIMA, 964 weekly origins
each) are measured in [docs/methods.md](docs/methods.md), and so are the global LightGBM
model, N-HiTS, MinT reconciliation and the staffing decision layer. N-HiTS did not earn
its complexity ([docs/neural-verdict.md](docs/neural-verdict.md)), and neither did
AutoARIMA, which cost five times ETS to finish third of the four statistical models. The
table below shows the best statistical model, which is ETS. PatchTST, TimesFM and the
probabilistic reconciliation are not built yet, and their rows say so. The tables below
are written by `headroom report` from the backtest's checkpoints and are never edited by
hand. [docs/state.md](docs/state.md) is the working state of the build.

## Result

<!-- report:start (written by `headroom report`; do not edit by hand) -->
All numbers from `headroom report`: 911 weekly forecast origins, 2009-01-05 to 2026-06-15, each forecasting the next 14 days for 37 series (the city, 5 boroughs and 31 dispatch areas). Every model is scored on the same origins, and every interval is a 95 percent moving-block bootstrap interval over them, paired wherever two methods are compared.

**Skill against seasonal naive, per hierarchy level**

| Method | Level | CRPS skill | Pinball skill (median) | Coverage at 90% nominal, whole period | Coverage at 90%, worst 91-day window (last origin) | Mean width at 90% | Fit time, full schedule |
|---|---|---|---|---|---|---|---|
| Seasonal naive | City | reference | reference | 0.882 [0.868, 0.898] | 0.555 (2020-05-18) | 893 [869, 920] | 1 min |
|  | Borough | reference | reference | 0.889 [0.880, 0.899] | 0.637 (2020-04-27) | 224 [219, 229] |  |
|  | Dispatch area | reference | reference | 0.896 [0.890, 0.901] | 0.718 (2020-04-27) | 62.0 [60.9, 63.0] |  |
| Best statistical: ETS | City | +0.180 [+0.165, +0.199] | +0.183 [+0.170, +0.201] | 0.918 [0.907, 0.933] | 0.610 (2020-05-25) | 842 [812, 873] | 1.6 h |
|  | Borough | +0.211 [+0.201, +0.223] | +0.211 [+0.202, +0.223] | 0.911 [0.902, 0.921] | 0.647 (2020-05-25) | 194 [188, 201] |  |
|  | Dispatch area | +0.252 [+0.246, +0.258] | +0.252 [+0.245, +0.258] | 0.904 [0.897, 0.910] | 0.683 (2020-06-01) | 48.4 [47.3, 49.7] |  |
| LightGBM, global | City | +0.190 [+0.149, +0.223] | +0.196 [+0.158, +0.227] | 0.897 [0.882, 0.913] | 0.467 (2020-05-04) | 839 [771, 936] | 2.0 h |
|  | Borough | +0.222 [+0.196, +0.244] | +0.227 [+0.204, +0.246] | 0.899 [0.888, 0.911] | 0.600 (2020-05-04) | 200 [187, 218] |  |
|  | Dispatch area | +0.252 [+0.239, +0.262] | +0.255 [+0.244, +0.264] | 0.903 [0.896, 0.909] | 0.691 (2020-05-04) | 51.1 [49.2, 53.7] |  |
| N-HiTS, refitted every 4 weeks | City | +0.008 [-0.043, +0.042] | +0.085 [+0.039, +0.114] | 0.581 [0.562, 0.598] | 0.236 (2020-06-08) | 400 [382, 415] | 18.4 h |
|  | Borough | +0.035 [-0.006, +0.063] | +0.099 [+0.061, +0.123] | 0.633 [0.620, 0.643] | 0.337 (2020-06-01) | 108 [104, 111] |  |
|  | Dispatch area | +0.073 [+0.046, +0.089] | +0.123 [+0.101, +0.136] | 0.676 [0.666, 0.683] | 0.392 (2020-06-08) | 32.5 [31.8, 33.1] |  |
| PatchTST | not built |  |  |  |  |  |  |
| TimesFM, zero-shot (clean window only) | not built |  |  |  |  |  |  |

Skill is one minus the method's mean score over seasonal naive's, so higher is better and zero is no better than the baseline.

**Coverage, and what it does and does not promise.** Seasonal naive (quantiles of its own past errors), the statistical models and N-HiTS are scored on their own quantiles with no conformal step, so no coverage is guaranteed for them. LightGBM forecasts a median only, and its distribution is conformal, built from its own errors over the previous 52 origins. Conformal coverage holds on average over time and only if errors are exchangeable, which demand through a shift is not, so it is not promised in any one window. The worst window is the lowest coverage in a trailing 91-day window (13 weekly origins), dated by the last origin in it. It is the single worst stretch of one history, so it carries no bootstrap interval.

**Fit time** is the median model time per origin times the 964 origins of the full schedule, on one desktop CPU (machine B in [docs/methods.md](docs/methods.md)); the median, so that time lost to other work or to the machine sleeping is not counted. The statistical models were fitted three at a time and each is given a third. LightGBM is the only model given the holiday calendar. N-HiTS is refitted every fourth origin and forecasts every origin from its latest weights. PatchTST and TimesFM are not built yet.

**Reconciliation: ETS with MinT**

Largest coherence breach over every origin and horizon step, in incidents: 514.8 before reconciling, 0.0e+00 after.

| Level | CRPS, ETS | CRPS, ETS + MinT | Change from reconciling | Skill of reconciling |
|---|---|---|---|---|
| City | 135 [127, 145] | 128 [120, 139] | -7.1 [-8.2, -4.8] | +0.052 [+0.034, +0.062] |
| Borough | 32.4 [30.7, 34.4] | 31.4 [29.7, 33.6] | -0.97 [-1.13, -0.69] | +0.030 [+0.021, +0.035] |
| Dispatch area | 8.27 [7.99, 8.62] | 8.28 [8.01, 8.62] | +0.007 [-0.009, +0.019] | -0.001 [-0.002, +0.001] |

Only the medians are reconciled; each node's quantiles move with its median. Probabilistic reconciliation is not built yet.

**The rota: staffing every dispatch area, priced against an oracle**

Inputs are illustrative and replaceable ([inputs/decision.toml](inputs/decision.toml)): one unit handles 10 incidents a day, a spare unit-day costs 1 and a missing one 4, so the costs imply staffing at the 80% quantile. The oracle, knowing each day's demand, staffs 409.3 units a day and costs nothing. Units and cost are summed over the 31 areas and averaged over the 14 days ahead.

| Service level | Method | Units staffed per day | Realised cost per day against the oracle | Cost minus ETS |
|---|---|---|---|---|
| 80%, implied by the costs | Seasonal naive | 457.6 [443.9, 470.6] | 85.13 [82.74, 87.98] | +19.37 [+18.84, +19.88] |
|  | ETS | 448.8 [435.3, 461.9] | 65.76 [63.48, 68.60] | reference |
|  | ETS + MinT | 448.8 [435.3, 461.9] | 65.79 [63.57, 68.52] | +0.03 [-0.11, +0.14] |
|  | LightGBM, global | 449.3 [435.9, 462.0] | 65.76 [63.38, 68.94] | +0.00 [-0.57, +0.69] |
|  | N-HiTS, refitted every 4 weeks | 433.3 [420.2, 446.1] | 81.92 [78.67, 86.42] | +16.16 [+14.65, +18.14] |
| 90% | Seasonal naive | 483.2 [469.1, 496.6] | 91.18 [88.92, 93.94] | +18.91 [+18.11, +19.57] |
|  | ETS | 468.7 [455.0, 482.4] | 72.27 [70.05, 75.09] | reference |
|  | ETS + MinT | 468.8 [454.9, 482.4] | 72.29 [70.11, 75.06] | +0.02 [-0.09, +0.10] |
|  | LightGBM, global | 470.3 [456.7, 483.4] | 73.39 [70.86, 76.87] | +1.12 [-0.14, +2.43] |
|  | N-HiTS, refitted every 4 weeks | 447.8 [434.4, 460.8] | 77.08 [74.04, 81.59] | +4.81 [+3.16, +6.85] |
| 95% | Seasonal naive | 504.9 [490.6, 518.6] | 104.42 [102.12, 107.23] | +21.43 [+20.24, +22.35] |
|  | ETS | 485.1 [470.9, 499.1] | 82.99 [80.59, 85.93] | reference |
|  | ETS + MinT | 485.2 [470.9, 499.2] | 83.01 [80.66, 85.89] | +0.02 [-0.08, +0.13] |
|  | LightGBM, global | 499.2 [484.9, 513.3] | 95.45 [90.64, 102.00] | +12.46 [+8.44, +17.90] |
|  | N-HiTS, refitted every 4 weeks | 461.9 [448.3, 475.1] | 77.85 [75.00, 82.16] | -5.14 [-6.92, -3.04] |
<!-- report:end -->

## What this does not do

- It does not report MAE as the result. A point forecast cannot be scored on the thing that
  matters here, which is whether the range held.
- It does not promise per-period coverage. Adaptive conformal intervals guarantee coverage
  on average over time under dependence; the README states this beside the chart.
- It does not forecast intraday or build shift rotas. Daily demand to daily staffing, with
  the demand-per-staff ratio and the costs as replaceable inputs.
- It does not use the employer's or any Canadian provincial data.

## How it works

See [PLAN.md](PLAN.md). Public emergency medical dispatch incidents are aggregated to daily
counts on a city, borough and dispatch-area hierarchy. Seasonal naive and statistical
models come first, then a global LightGBM model on lag and calendar features, N-HiTS and
PatchTST as global neural models, and TimesFM as a
pretrained foundation model used zero-shot and scored only on dates after its
pretraining data ends, so it cannot have seen them. All produce
quantiles and are scored by CRPS and pinball loss in a rolling-origin backtest with block
bootstrap intervals. Adaptive conformal inference wraps each model's forecasts and is
compared with split conformal through the March 2020 shift. MinT reconciliation makes the
levels sum, with a probabilistic reconciliation so the quantiles stay coherent. A
newsvendor decision layer turns the reconciled distribution into staffing at a service
level and realises each method's cost against actual demand. A static dashboard shows the
fan charts, the coverage chart and a service-level slider.

## Part of a portfolio

One of fifteen projects built over twelve months. It reuses the static decision-app pattern
from the Intervention Targeting Engine and is the portfolio's second Azure hosting example.

## How this was built

Design, methodology, evaluation choices and judgement are Peter Parker's, including years
of health-sector utilisation forecasting. AI coding assistants (Claude Code) were used for
implementation and drafting, the way a senior engineer uses them in 2026. Every number in
the results tables is reproducible from this repository with one command, and that
reproducibility is the evidence that matters.
