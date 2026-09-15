# Capacity Planning Forecaster

Demand forecasts, emergency calls and emergency-department arrivals, with uncertainty
ranges that actually hold when conditions shift, turned into a staffing number at a chosen
service level. Overstaffing costs money; understaffing costs patients. This puts a
defensible number on both, site by site, rolling up to the region.

**Status: building,** week 1 of two, started 2026-09-12. The plan is in
[PLAN.md](PLAN.md): a two-week build on public demand series, everything on a desktop CPU.

The data is loaded and checked ([docs/data.md](docs/data.md)). The baseline, the conformal
intervals and the statistical models (ETS, Theta, MSTL over 964 weekly origins) are
measured in [docs/methods.md](docs/methods.md), and so are the global LightGBM model,
N-HiTS, MinT reconciliation and the staffing decision layer. N-HiTS did not earn its
complexity ([docs/neural-verdict.md](docs/neural-verdict.md)). PatchTST, TimesFM and the
probabilistic reconciliation are not built yet. The tables below stay empty until the
report command fills them, which is the only thing allowed to. [docs/state.md](docs/state.md) is the working state of the build.

## Result

Not yet measured. The build fills these tables.

**Skill against seasonal naive, per hierarchy level (rolling-origin backtest, 95% CIs)**

| Method | Level | CRPS skill | Pinball skill (median) | Coverage at 90% nominal, whole period | Coverage at 90%, through the 2020 shift (90-day window, minimum) | Mean width | Fit time |
|---|---|---|---|---|---|---|---|
| Seasonal naive | _not yet_ | | | | | | |
| Best statistical | | | | | | | |
| LightGBM, global | | | | | | | |
| N-HiTS | | | | | | | |
| PatchTST | | | | | | | |
| TimesFM, zero-shot (clean window only) | | | | | | | |

**Reconciliation and the rota**

| Coherence error after MinT | CRPS change from reconciling, by level | Service level | Staffing per period, median method | Realised cost vs oracle, per method |
|---|---|---|---|---|
| _not yet_ | | | | |

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
