# Capacity Planning Forecaster

Demand forecasts, emergency calls and emergency-department arrivals, with uncertainty
ranges that actually hold when conditions shift, turned into a staffing number at a chosen
service level. Overstaffing costs money; understaffing costs patients. This puts a
defensible number on both, site by site, rolling up to the region.

**Status: planning.** Nothing has run yet. The plan is in [PLAN.md](PLAN.md): a two-week
build in late July 2027 on public demand series, everything on a laptop CPU.

## Result

Not yet measured. The build fills these tables.

**Skill against seasonal naive, per hierarchy level (rolling-origin backtest, 95% CIs)**

| Method | Level | CRPS skill | Pinball skill (median) | Coverage at 90% nominal, whole period | Coverage at 90%, through the 2020 shift (90-day window, minimum) | Mean width | Fit time |
|---|---|---|---|---|---|---|---|
| Seasonal naive | _not yet_ | | | | | | |
| Best statistical | | | | | | | |
| N-HiTS | | | | | | | |
| PatchTST | | | | | | | |

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
models come first, then N-HiTS and PatchTST as global neural models, all producing
quantiles and scored by CRPS and pinball loss in a rolling-origin backtest with block
bootstrap intervals. Adaptive conformal inference wraps each model's forecasts and is
compared with split conformal through the March 2020 shift. MinT reconciliation makes the
levels sum, with a probabilistic reconciliation so the quantiles stay coherent. A
newsvendor decision layer turns the reconciled distribution into staffing at a service
level and realises each method's cost against actual demand. A static dashboard shows the
fan charts, the coverage chart and a service-level slider.

## Part of a portfolio

One of ten projects built over twelve months. It reuses the static decision-app pattern
from the Intervention Targeting Engine and is the portfolio's second Azure hosting example.

## How this was built

Design, methodology, evaluation choices and judgement are Peter Parker's, including years
of health-sector utilisation forecasting. AI coding assistants (Claude Code) were used for
implementation and drafting, the way a senior engineer uses them in 2026. Every number in
the results tables is reproducible from this repository with one command, and that
reproducibility is the evidence that matters.
