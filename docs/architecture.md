# Architecture

Three figures: the system as a whole, how the headline table is produced, and the path from a
forecast distribution to crews on shift. Each names the module that implements what it shows;
[PLAN.md](../PLAN.md) says why and [methods.md](methods.md) gives every method in full.

## The system

![Public emergency dispatch incidents are aggregated to a daily hierarchy, every model is run through one rolling-origin backtest and scored by CRPS with block-bootstrap intervals, conformal intervals and MinT reconciliation wrap the forecasts, a newsvendor layer prices them into crews, and the report exports the same numbers to a static dashboard; all of it on a CPU.](architecture/system.svg)

Twenty-one years of New York City emergency medical dispatch incidents, open data, are
aggregated by `headroom.data.nyc_ems` into a daily panel on a city, borough and dispatch-area
hierarchy of 37 nodes (`headroom.hierarchy`), with the data checks that found no missing day.
The rolling-origin backtest (`headroom.backtest`) hands each of nine models
(`headroom.models`: seasonal naive, the statistical models, a global LightGBM, N-HiTS,
PatchTST and TimesFM zero-shot) only the history before each origin and stores every quantile
it forecasts. The scoring module (`headroom.score`) reads the stored forecasts against actuals
for CRPS, pinball loss and coverage with a moving block bootstrap; the conformal module
(`headroom.conformal`) wraps each model's forecasts with split, adaptive and aggregated
conformal intervals; `headroom.reconcile` makes the levels sum with MinT and keeps the
quantiles coherent; and the decision layer (`headroom.decide`) turns the reconciled
distribution into crews at a service level with the costs in `inputs/decision.toml`.
`headroom.report` writes the README tables and charts and exports two schema-checked JSON
files that the static dashboard at capacity.peterparker.ca reads, with no framework and no
request that leaves the page. Everything runs on a CPU.

## How the headline table is produced

![Each model sees only the 1,095 days before each of 964 weekly origins, refits every fourth origin, forecasts 14 days of quantiles, and is scored by CRPS and pinball against actuals with a moving block bootstrap over origins; headroom report writes the README tables from the stored forecasts.](architecture/measurement.svg)

A forecast origin is a day on which a model may know everything up to and including that day
and nothing after it. The runner (`headroom.backtest.origins`) walks 964 weekly origins from
2007-12-31 to 2026-06-15 and hands each model `panel[:, :origin + 1]` from a trailing
three-year window, which a test asserts at every origin; models refit every fourth origin on
the same schedule, so the cheap ones are not given an advantage the expensive ones were
denied. Each model writes its 14-day quantile forecasts to the store (`backtest/out`), and the
scorer computes CRPS as a pinball integral, pinball loss, coverage and skill against the
seasonal naive (`headroom.score.crps`, `headroom.score.coverage`), with 95% intervals from a
moving block bootstrap over origins that respects the dependence between neighbouring weeks
(`headroom.score.bootstrap`). TimesFM is scored only on origins after its pretraining data
ends. `headroom report` writes the README tables (`headroom.report.tables`) and the charts and
dashboard JSON (`headroom.report.charts`, `headroom.report.export`) from the same stored
forecasts, including the coverage chart through March 2020.

## From a forecast distribution to crews on shift

![A model's quantiles are wrapped by adaptive conformal intervals so coverage holds through a shift, reconciled across the hierarchy so areas sum to the city, read at a service level, and priced by a newsvendor rule against stated costs into crews; the realised cost is compared with an oracle that knew demand.](architecture/decision.svg)

A model's quantile forecast for each of the 37 nodes is wrapped by an adaptive conformal
interval (`headroom.conformal.aci`, `headroom.conformal.agaci`), which guarantees coverage on
average through a shift where split conformal (`headroom.conformal.split`) under-covers after
March 2020. MinT reconciliation (`headroom.reconcile.mint`) makes the areas sum to the boroughs
and the boroughs to the city while keeping the quantiles coherent. A service level, 0.80, 0.90
or 0.95 of forecast demand, picks a quantile, and the newsvendor rule
(`headroom.decide.newsvendor`) turns it into crews per area per day using the illustrative
demand per unit and the costs of an idle and of a missing unit in `inputs/decision.toml`,
which exist so the arithmetic has something to run on and are meant to be replaced. The
realised cost of each method's rota is then priced against an oracle that knew the demand, and
the dashboard's slider re-prices the rota at each service level it was measured at.
