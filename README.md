# Capacity Planning Forecaster

Demand forecasts, emergency calls and emergency-department arrivals, with uncertainty
ranges that actually hold when conditions shift, turned into a staffing number at a chosen
service level. Overstaffing costs money; understaffing costs patients. This puts a
defensible number on both, site by site, rolling up to the region.

**The dashboard is live at [capacity.peterparker.ca](https://capacity.peterparker.ca).** Every
model the plan names is measured and the tables below are complete. The plan is in
[PLAN.md](PLAN.md): public demand series, everything on a desktop CPU.

The data is loaded and checked ([docs/data.md](docs/data.md)). The baseline, the conformal
intervals and the statistical models (ETS, Theta, MSTL and AutoARIMA, 964 weekly origins
each) are measured in [docs/methods.md](docs/methods.md), and so are the global LightGBM
model, N-HiTS, PatchTST, TimesFM, MinT reconciliation and the staffing decision layer. The
table below shows the best statistical model, which is ETS.

**Neither neural model earned its complexity, and they failed differently**
([docs/neural-verdict.md](docs/neural-verdict.md)). N-HiTS is beaten everywhere and its
90 percent intervals cover 58 to 68 percent of outcomes. PatchTST is a good forecaster
that ties ETS at the city and borough, loses narrowly at the dispatch areas, and costs
about seventeen times as much to fit. AutoARIMA did not earn its cost either: five times
ETS to finish third of the four statistical models.

**The pretrained model is the one that did not lose, and the honest version of that is
complicated.** TimesFM 2.5, trained on none of this data, matches the global LightGBM
model on the clean window after its pretraining data ends, and both beat ETS there. It has
its own table below rather than a row in the main one, because its weights postdate most
of these origins and pooling the two would be meaningless. On the 40 origins after the
weights were published, the one stretch nothing could have leaked into, it is about one
percent behind LightGBM instead. Forty origins cannot settle that, and the verdict says so
rather than picking the window that flatters it.

**The forecasts are coherent as a distribution, not just as a median.** Every draw of the
reconciled distribution sums correctly across all 37 series at once, so a decision taken
over the whole hierarchy is taken on one consistent future. It buys calibration (ETS's 90
percent intervals go from covering 0.918 to 0.901) and it is not free: at the dispatch
areas it is slightly worse on CRPS than moving the quantiles with the median, and it staffs
a little more. Both numbers are in the table.

The tables below are written by `headroom report` from the backtest's checkpoints and are
never edited by hand. [docs/state.md](docs/state.md) is the working state of the build.

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
| PatchTST, refitted every 13 weeks | City | +0.201 [+0.171, +0.224] | +0.205 [+0.171, +0.230] | 0.896 [0.884, 0.909] | 0.593 (2020-05-04) | 778 [743, 817] | 27.3 h |
|  | Borough | +0.218 [+0.195, +0.235] | +0.219 [+0.193, +0.237] | 0.901 [0.892, 0.909] | 0.625 (2020-05-04) | 190 [183, 197] |  |
|  | Dispatch area | +0.243 [+0.229, +0.252] | +0.243 [+0.229, +0.253] | 0.900 [0.895, 0.905] | 0.698 (2020-05-04) | 49.3 [48.2, 50.6] |  |
| TimesFM, zero-shot | own windows, below |  |  |  |  |  |  |

Skill is one minus the method's mean score over seasonal naive's, so higher is better and zero is no better than the baseline.

**Coverage, and what it does and does not promise.** Seasonal naive (quantiles of its own past errors), the statistical models, N-HiTS and PatchTST, are scored on their own quantiles with no conformal step, so no coverage is guaranteed for them. LightGBM, global forecasts a median only, and its distribution is conformal, built from its own errors over the previous 52 origins. Conformal coverage holds on average over time and only if errors are exchangeable, which demand through a shift is not, so it is not promised in any one window. The worst window is the lowest coverage in a trailing 91-day window (13 weekly origins), dated by the last origin in it. It is the single worst stretch of one history, so it carries no bootstrap interval.

**Fit time** is the median model time per origin times the 964 origins of the full schedule, on one desktop CPU (machine B in [docs/methods.md](docs/methods.md)); the median, so that time lost to other work or to the machine sleeping is not counted. The statistical models were fitted three at a time and each is given a third. LightGBM is the only model given the holiday calendar. The neural models are refitted on a schedule, N-HiTS every 4 and PatchTST every 13 origins, and each forecasts every origin from its latest weights.

**TimesFM, pretrained and used zero-shot: the windows it can be judged on**

TimesFM is not in the table above, and the reason is the result. Its weights were trained after most of these origins, on a corpus that contains the 2020 period in other series, so a forecast it makes of 2020 is not the same kind of claim as every other row's. It is scored here on three windows, **never pooled**: the clean window, whose whole horizon falls after the latest documented pretraining data; a shorter cross-check after the weights were published, which is a subset of the same forecasts; and the full backtest, which is labelled exposed and is not a result. It is trained on none of this data at all, and it is given the same window, horizon and scoring grid as everything else. Inference over the full schedule: TimesFM, 6.9 h.

| Window | Origins | Level | CRPS | CRPS skill | CRPS minus ETS | CRPS minus LightGBM, global | Coverage at 90% nominal |
|---|---|---|---|---|---|---|---|
| Clean: whole horizon after the pretraining data | 133 from 2023-12-04 | City | 125 [111, 139] | +0.286 [+0.247, +0.302] | -24.3 [-29.8, -12.7] | -6.9 [-12.4, +1.3] | 0.895 [0.862, 0.917] |
|  |  | Borough | 31.8 [29.2, 34.5] | +0.271 [+0.251, +0.279] | -3.23 [-4.20, -1.43] | -0.49 [-1.43, +0.70] | 0.895 [0.878, 0.906] |
|  |  | Dispatch area | 8.57 [8.33, 8.81] | +0.263 [+0.258, +0.267] | -0.171 [-0.296, -0.040] | +0.021 [-0.044, +0.109] | 0.903 [0.896, 0.908] |
| Cross-check: after the weights were published | 40 from 2025-09-15 | City | 152 | +0.299 | -40.5 | +1.5 | 0.875 |
|  |  | Borough | 36.9 | +0.273 | -5.35 | +1.06 | 0.874 |
|  |  | Dispatch area | 9.05 | +0.265 | -0.340 | +0.110 | 0.893 |
| Full backtest, exposed to the leak | 911 from 2009-01-05 | City | 121 [113, 132] | +0.268 [+0.252, +0.282] | -14.5 [-16.6, -11.4] | -12.8 [-18.3, -8.7] | 0.901 [0.890, 0.913] |
|  |  | Borough | 30.2 [28.6, 32.2] | +0.265 [+0.254, +0.275] | -2.23 [-2.55, -1.78] | -1.74 [-2.60, -1.12] | 0.902 [0.894, 0.911] |
|  |  | Dispatch area | 8.17 [7.91, 8.51] | +0.262 [+0.255, +0.267] | -0.107 [-0.144, -0.066] | -0.114 [-0.196, -0.052] | 0.903 [0.897, 0.910] |

Its forecast is its median and the distribution scored here is conformal from its own past errors, as LightGBM's is, because its own quantile head stops at the 0.1 and 0.9 quantiles and this table reports 95 percent intervals. A window shorter than 112 origins carries no bootstrap interval: the block length is 28 origins, and a window of a few blocks resamples from too little to say anything. Those rows print the difference alone, and direction is all they carry.

**Reconciliation: ETS with MinT, two ways**

Base forecasts made one series at a time do not sum: the largest breach of the summing constraints is 514.8 incidents. Both reconciliations close it to 5.5e-12, and **MinT paths closes it over every draw of the distribution**, not only the median.

| Level | CRPS, ETS | MinT: change in CRPS | MinT: coverage at 90% | MinT paths: change in CRPS | MinT paths: coverage at 90% |
|---|---|---|---|---|---|
| City | 135 [127, 145] | -7.1 [-8.2, -4.8] | 0.932 [0.919, 0.944] | -6.3 [-8.1, -3.4] | 0.901 [0.889, 0.913] |
| Borough | 32.4 [30.7, 34.4] | -0.97 [-1.13, -0.69] | 0.919 [0.909, 0.929] | -0.53 [-0.77, -0.12] | 0.900 [0.891, 0.910] |
| Dispatch area | 8.27 [7.99, 8.62] | +0.007 [-0.009, +0.019] | 0.904 [0.898, 0.910] | +0.170 [+0.154, +0.190] | 0.901 [0.895, 0.907] |

**MinT** reconciles the median and moves each node's quantiles with it, so the medians are coherent and the spread is still the base model's. **MinT paths** is the probabilistic reconciliation: each of the 52 error vectors in the window is added to the base forecast and put through the same projection, so every draw is coherent across all 37 series at once and the distribution is read off those draws. Its quantiles still do not sum, and they are not supposed to: the boroughs do not have their bad days together, so the city's 90th percentile is below the sum of theirs. What is coherent is every draw, which is what a decision taken over the whole hierarchy needs. Paths are not floored at zero, because that would break the coherence they exist for; 0.0013% of path values fall below zero.

**The rota: staffing every dispatch area, priced against an oracle**

Inputs are illustrative and replaceable ([inputs/decision.toml](inputs/decision.toml)): one unit handles 10 incidents a day, a spare unit-day costs 1 and a missing one 4, so the costs imply staffing at the 80% quantile. The oracle, knowing each day's demand, staffs 409.3 units a day and costs nothing. Units and cost are summed over the 31 areas and averaged over the 14 days ahead.

| Service level | Method | Units staffed per day | Realised cost per day against the oracle | Cost minus ETS |
|---|---|---|---|---|
| 80%, implied by the costs | Seasonal naive | 457.6 [443.9, 470.6] | 85.13 [82.74, 87.98] | +19.37 [+18.84, +19.88] |
|  | ETS | 448.8 [435.3, 461.9] | 65.76 [63.48, 68.60] | reference |
|  | ETS + MinT | 448.8 [435.3, 461.9] | 65.79 [63.57, 68.52] | +0.03 [-0.11, +0.14] |
|  | ETS + MinT paths | 449.5 [436.0, 462.3] | 66.53 [64.41, 69.19] | +0.77 [+0.55, +0.98] |
|  | LightGBM, global | 449.3 [435.9, 462.0] | 65.76 [63.38, 68.94] | +0.00 [-0.57, +0.69] |
|  | N-HiTS, refitted every 4 weeks | 433.3 [420.2, 446.1] | 81.92 [78.67, 86.42] | +16.16 [+14.65, +18.14] |
|  | PatchTST, refitted every 13 weeks | 448.8 [435.2, 461.7] | 66.84 [64.23, 70.55] | +1.08 [+0.43, +2.13] |
| 90% | Seasonal naive | 483.2 [469.1, 496.6] | 91.18 [88.92, 93.94] | +18.91 [+18.11, +19.57] |
|  | ETS | 468.7 [455.0, 482.4] | 72.27 [70.05, 75.09] | reference |
|  | ETS + MinT | 468.8 [454.9, 482.4] | 72.29 [70.11, 75.06] | +0.02 [-0.09, +0.10] |
|  | ETS + MinT paths | 470.8 [456.8, 483.9] | 73.98 [71.91, 76.77] | +1.71 [+1.11, +2.23] |
|  | LightGBM, global | 470.3 [456.7, 483.4] | 73.39 [70.86, 76.87] | +1.12 [-0.14, +2.43] |
|  | N-HiTS, refitted every 4 weeks | 447.8 [434.4, 460.8] | 77.08 [74.04, 81.59] | +4.81 [+3.16, +6.85] |
|  | PatchTST, refitted every 13 weeks | 470.8 [456.9, 484.2] | 74.36 [71.67, 78.20] | +2.09 [+0.93, +3.42] |
| 95% | Seasonal naive | 504.9 [490.6, 518.6] | 104.42 [102.12, 107.23] | +21.43 [+20.24, +22.35] |
|  | ETS | 485.1 [470.9, 499.1] | 82.99 [80.59, 85.93] | reference |
|  | ETS + MinT | 485.2 [470.9, 499.2] | 83.01 [80.66, 85.89] | +0.02 [-0.08, +0.13] |
|  | ETS + MinT paths | 498.7 [484.8, 512.2] | 94.88 [91.60, 99.21] | +11.89 [+9.66, +14.96] |
|  | LightGBM, global | 499.2 [484.9, 513.3] | 95.45 [90.64, 102.00] | +12.46 [+8.44, +17.90] |
|  | N-HiTS, refitted every 4 weeks | 461.9 [448.3, 475.1] | 77.85 [75.00, 82.16] | -5.14 [-6.92, -3.04] |
|  | PatchTST, refitted every 13 weeks | 490.2 [475.9, 503.9] | 87.52 [84.67, 91.52] | +4.53 [+2.92, +6.22] |
<!-- report:end -->

## Coverage through a real shift, which is the finding

![Rolling coverage at the city against nominal, at three nominal levels, for split,
adaptive and aggregated conformal, with the mean width of the same intervals under each
panel. All three collapse in March 2020 and the widths only rise afterwards.](docs/charts/coverage.png)

Written by `headroom charts`. Each panel is trailing coverage over 13 weekly origins at one
nominal level, with the width that produced it underneath, because a method reaches nominal
trivially by being wide enough to cover anything.

**What the picture says that the table cannot.** Over the whole period every method sits
near nominal. In March 2020 all three fall off a cliff together, to 0.47 at the 80 percent
level, and the widths do not rise until after the fall. Adaptive conformal is meant to be
the one that recovers, and it does not: it **cannot widen past the largest nonconformity
score in its calibration window**, and the worst residual of the shift was larger than
anything in the previous year. It ran out of room rather than being too slow.
[docs/methods.md](docs/methods.md) has the measurement, and the remedy it does not attempt
is named there rather than quietly tried.

**The conformal assumption, stated beside the chart.** These intervals are calibrated on
past errors and their guarantee is exchangeability, which a demand series does not have.
Split conformal promises coverage on average over a period in which the error distribution
does not change. Adaptive conformal promises coverage **on average over time**, not in any
particular window, which is precisely why a chart of particular windows is the honest way
to show it. No coverage is promised for March 2020 by any method here, and none was
delivered.

![Forecast bands 14 days ahead against what happened, at the city, a borough and a dispatch
area, through the 2020 shift. The outcome leaves the 95 percent band entirely in late March
and the bands follow it two weeks later, by which time the outcome has fallen below
them.](docs/charts/fan.png)

The same failure in the units a planner counts. The bands are ETS's, 14 days ahead. Demand
leaves the 95 percent band altogether in late March 2020; the bands chase it up two weeks
later, which is exactly the forecast horizon; and by the time they arrive the outcome has
dropped through the bottom. A 14-day forecast cannot see a shift that happens inside its
own horizon, and no amount of interval calibration changes that.

## What did not work

Four things were built, measured and not adopted. Each is a real cost paid, and the numbers
are the reason, not a preference.

- **N-HiTS, the named neural forecaster: beaten everywhere.** CRPS against ETS, per origin,
  paired: city +28.39 [+21.72, +40.03], borough +7.19, dispatch area +1.98, where positive
  is worse. Its own 90 percent intervals covered 0.58 to 0.68 of outcomes. About 18.3 hours
  of fitting against ETS's 1.5. It was not the refit schedule (no trend with weeks since a
  refit), not calibration alone (a conformal median is still behind ETS), and not the seed
  (a second seed over 20 refits is no better on average).
- **PatchTST: a tie at the top, a loss at the leaves, at seventeen times the cost.** City
  -3.45 [-6.72, +2.48] and borough -0.31 [-0.89, +0.67], both intervals crossing zero, so
  neither is a win; dispatch area +0.103 [+0.041, +0.203], which is a loss. It is the best
  calibrated model here (0.896 / 0.901 / 0.900 from its own quantiles) and its city CRPS is
  the lowest in the project, and it still cannot be claimed. 27.3 hours of fitting.
- **AutoARIMA: five times ETS's cost to finish third of four.** Fifteen hours to confirm
  what the cheapest statistical model already said.
- **A GPU.** A GTX 1650 gave no speedup over the CPU on these fits, so the published times
  are CPU times and no GPU is bought for this project.

[docs/neural-verdict.md](docs/neural-verdict.md) has the full verdict, including the one
model that did not lose. Conformal prediction is also implemented directly rather than
through MAPIE, because the update has to respect when an outcome becomes known: with weekly
origins, a 14-day-ahead forecast's error has not happened yet at the next origin, and a
library that assumes it has would leak the future into the interval.

## The dashboard: [capacity.peterparker.ca](https://capacity.peterparker.ca)

The same numbers, but you can move them. `dashboard/` is a static page over two JSON files
written by `headroom export` from the same checkpoints the tables above come from. Four
panels: the forecast with any of the 37 series picked out of the hierarchy, coverage through
the shift with the trailing window under your control, the reconciliation table, and a
service-level slider that re-prices the rota at each level it was measured at. No framework,
no web fonts, and no request that leaves the page.

```bash
uv run headroom export    # writes dashboard/data/*.json, validated against a schema
uv run headroom serve     # http://localhost:8080, with the headers the host will send
```

Use `headroom serve` rather than a plain file server. A file server sends none of the
headers in `dashboard/staticwebapp.config.json`, so it shows a page the content security
policy would partly refuse, and a page that only works without one is a page that breaks
when it is published. [docs/deploy.md](docs/deploy.md) is the deployment runbook.

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

## What this reuses

The dashboard is built on the static decision-app pattern from the
[Intervention Targeting Engine](https://github.com/Peter-A-P/intervention-targeting-engine),
which built it first: hand-written HTML, CSS and plain JavaScript over precomputed JSON,
with no framework and no off-origin request.

## How this was built

Design, methodology, evaluation choices and judgement are Peter Parker's, including years
of health-sector utilisation forecasting. AI coding assistants (Claude Code) were used for
implementation and drafting, under review at every step. Every number in the results tables
is reproducible from this repository with one command.

## Licence

MIT, in [LICENSE](LICENSE). The two fonts the dashboard serves are SIL Open Font License
1.1 and say so in [dashboard/fonts/LICENSE.txt](dashboard/fonts/LICENSE.txt). No demand data
is redistributed here: the loader fetches it from NYC Open Data under the City of New York's
terms, and only derived aggregates are committed ([docs/data.md](docs/data.md)).
