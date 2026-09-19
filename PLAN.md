# Plan: Capacity Planning Forecaster

**Written:** 2026-09-07. **Status:** built; the dashboard is live.
[docs/state.md](docs/state.md) is the working state.

**Build:** two weeks, Sep 12 to Sep 25 2026. **Package:** `headroom`. **Fed by:** nothing.
**Feeds:** nothing; the dashboard reuses the Intervention Targeting Engine's static
decision-app pattern.

> **Brought forward, 2026-09-12.** This was scheduled for mid-2027 and was started early
> instead. Nothing had to wait: it calls no model vendor, and its data is twenty-one years
> of published history that does not need calendar time to accumulate. The two-week
> duration, the budget and the scope are unchanged.
>
> One deliverable moved with it. The dashboard reuses a static decision-app pattern that
> the Intervention Targeting Engine was building at the time, so rather than invent it here
> and have that project inherit it, the dashboard was held until it existed. Week 2 below
> shipped everything else.
>
> **That cleared on 2026-09-18.** The pattern exists and can be read: `demo/` in
> `Peter-A-P/intervention-targeting-engine`, a hand-written `index.html`, `style.css` and
> plain `.js` against precomputed JSON, built by `src/itx/demo/build.py` and checked by
> `tests/test_demo.py`. Nothing about the dashboard was ever waiting on calendar time.
> Publishing it stayed a separate decision from building it, because publishing provisions
> hosting.

This project calls no language model, so nothing about it waits on one. Every number below
comes from a rolling-origin backtest with block-bootstrap intervals over forecast origins.
It is a two-week build, and the scope is chosen to fit: most demand-forecasting write-ups
report a point-estimate MAE and stop, while this one reports whether the intervals held
when the world changed, whether the sites sum to the region, and what the forecast means
for the rota.

> **Employer note.** The vocabulary comes from years of provincial health analytics; the
> data does not. Public emergency-services and hospital demand series from other
> jurisdictions only; nothing internal, no employer series.

---

## 1. What this produces

Demand forecasts, emergency medical calls by borough rolling up to a city and emergency
department attendances by trust rolling up to regions, with uncertainty ranges that hold
their stated coverage through a distribution shift, reconciled so the parts sum to the
whole, and turned into a staffing number at a chosen service level with the cost of being
wrong in either direction made explicit.

The numbers a stranger can check:

| Number | What it shows |
|---|---|
| CRPS and pinball loss at each quantile, per method per hierarchy level, as skill relative to seasonal naive, 95% block-bootstrap CIs over origins | Whether anything beat the baseline that any practitioner would try first |
| **Empirical coverage at 80, 90 and 95% nominal in a rolling 90-day window through the 2020 shift**, adaptive conformal against split conformal and against the models' own quantiles | The headline chart: intervals that hold when conditions change |
| Interval width alongside coverage | Coverage bought by width is not a result |
| Reconciliation: coherence error after MinT (must be zero to numerical precision) and the change in CRPS at each level from reconciling | Sites sum to the region, and reconciling helped or hurt, said which |
| Realised staffing cost per method at three service levels against an oracle that knew demand, from the backtest | The forecast as a rota decision, priced |
| Neural against statistical: paired CRPS difference of N-HiTS and PatchTST against the best statistical model, per level and per horizon, with CIs, and training time | Where the neural model earned its complexity and where it did not |
| Gradient boosting against statistical and neural: the same paired comparison for a global LightGBM model, so the gain from learning across series is separated from the gain from deep learning | Whether the neural models beat what a practitioner would try first, or only the per-series baselines |
| Foundation model against statistical: the same paired comparison for TimesFM used zero-shot, reported separately on origins after its pretraining data ends | Whether a pretrained model beats the baselines on data it cannot have seen |
| Compute time per method for the full backtest | What the accuracy costs in minutes |

## 2. Design decisions

### 2.1 One dataset done fully right, a second if time allows

Primary: New York City emergency medical dispatch incidents, an open dataset of thirty
million timestamped incidents since 2005 with borough and dispatch area, aggregated to
daily counts. It gives a two-level hierarchy (city, borough, dispatch area), twenty-one
years of daily data, weekly and annual seasonality, and real shifts: the 2012 storm, the
March 2020 surge. Loaded 2026-09-12: 37 nodes over 31 dispatch areas, five boroughs and
the city, 7,851 days from 2005-01-01 to 2026-06-30, no missing day. The eight busiest
days in the whole record fall between 2020-03-26 and 2020-04-06, so the shift the
coverage chart is built around is unmistakably there. `docs/data.md` has every number
and the cost of every cleaning decision. Secondary, optional: NHS England monthly emergency-department attendances by
provider, which tests reconciliation with hundreds of leaf nodes and carries the same 2020
shift at monthly resolution. The plan is complete with the primary alone.

### 2.2 Baselines first, honestly

Seasonal naive, then ETS, Theta, AutoARIMA and MSTL for the double seasonality, all
through StatsForecast with prediction quantiles. Every later number is reported as skill
relative to seasonal naive. A project that skips this is not credible to anyone who has
forecast for a living.

### 2.3 Probabilistic scoring only

CRPS computed from the quantile set, pinball loss per quantile, coverage and width. MAE is
reported in a footnote for readers who look for it, never as the headline, and section 9's
candidate 2 shows what choosing a model by MAE would have done.

### 2.4 Adaptive conformal, with the assumption stated

Exchangeability does not hold for time series, so split conformal is expected to fail
through the shift and is kept as the comparison. The main method is adaptive conformal
inference (the online update of the miscoverage level from realised coverage), with an
aggregated-expert variant to remove the step-size choice, applied to each model's point or
median forecast per series and horizon.

Implemented directly rather than through MAPIE, which an earlier one-line stack summary
named (corrected 2026-09-12). Three reasons, in order of weight. The backtest needs
the conformal update to respect **when an outcome becomes known**: with weekly origins, the
error of a 14-day-ahead forecast made at the previous origin has not happened yet, and
MAPIE's time-series API is built around a scikit-learn regressor and its own refit loop,
which has nowhere to express that. MAPIE does not ship the aggregated-expert variant, which
is what removes the step-size choice. And the guarantee is the thing this project is
demonstrating, so the update rule is four lines that should be readable in the repository
rather than behind a dependency. The assumption made (the update tracks the
recent miscoverage; coverage is guaranteed on average over time, not per period) is stated
in the README next to the chart, not glossed.

### 2.5 Reconciliation that keeps the distribution

MinT reconciliation with shrinkage on point forecasts through the hierarchical forecasting
library, then a probabilistic reconciliation (bootstrap of reconciled in-sample errors) so
the quantiles are coherent too. Coherence is verified numerically at every origin.
Conformal intervals are fitted on the reconciled forecasts, so the coverage claim is made
about the numbers a planner would actually use.

**As built, 2026-09-14 (point MinT).** Implemented in `headroom.reconcile.mint` rather than
called through hierarchicalforecast, whose API wants in-sample fitted values the
checkpoints do not hold; the shrinkage arithmetic is tested equal to the library's
`mint_shrink`. ``W`` comes from each model's out-of-sample errors at the same horizon step
over the previous 52 origins, not in-sample residuals. Quantiles move with their median.
On ETS: coherence error from 515 incidents to 0, city CRPS -7.05 [-8.24, -4.80], borough
-0.97 [-1.13, -0.69], dispatch area +0.007 [-0.009, +0.019].

**As built, 2026-09-18 (probabilistic).** `headroom.reconcile.paths`. One deviation from the
line above, recorded rather than made quietly: it says "bootstrap of reconciled in-sample
errors", and the 52 error vectors in the window are used **once each rather than resampled
with replacement**. Resampling adds no information to those 52 and would cost the order
statistic convention the rest of this package uses, so the full sample is used and there is
no seed to record. The errors are the same out-of-sample ones ``W`` is estimated from, for
the reasons above. Every path is coherent to 5e-12 at every origin, against a base breach
of 515 incidents; the marginal quantiles still do not sum, which is correct and is asserted
in `tests/test_paths.py` so that nobody later "fixes" it. Paths are not floored at zero,
because clipping breaks coherence; 0.0013 percent of ETS's path values fall below zero.
Because the paths need only a median, this also reconciles the median-only models, which
closes the gap `docs/state.md` recorded. Conformal on the reconciled forecasts is still to
build: the paths already take their spread from realised errors, so what is left is the
narrower question of whether an adaptive step on top of them adds anything.

### 2.6 The decision layer is a newsvendor

Staffing at a chosen service level is the demand quantile at that level divided by a
stated demand-per-staff ratio, with the under-staffing and over-staffing costs stated as
inputs. The newsvendor result gives the optimal quantile as the ratio of those costs, so
the service level a planner chooses is shown next to the one the costs imply. The
backtest realises each method's staffing decision against actual demand and sums the cost,
against an oracle. Inputs are a table in the README so a reader substitutes their own.

**As built, 2026-09-14.** The inputs live in `inputs/decision.toml` (illustrative, and
labelled so), which the code reads and the README will show. Staffing is per dispatch area
per day, in whole units. At the cost-implied 80 percent, ETS's realised cost is 23 percent
below seasonal naive's; MinT changes nothing at the areas; LightGBM ties ETS at 80 percent
and costs 15 percent more at 95, where its wider upper tail staffs idle units. Staffing
from the reconciled distribution is done for ETS; the probabilistic reconciliation it was
meant to use is not built.

### 2.7 Neural models on CPU, named, and allowed to lose

N-HiTS and PatchTST through NeuralForecast with a multi-quantile loss, trained as global
models across the series on the laptop's CPU (small data; minutes per fit). The paired
comparison against the best statistical model is per level and horizon with intervals. If
the neural models do not beat the statistical ones at the leaf level, or only at the top,
the README says exactly that. A verdict that can fall either way is worth something;
"deep learning won" decided in advance is worth nothing.

**As built, 2026-09-14 (N-HiTS).** "Minutes per fit" was right and decides the design: a
fit on the 37 series is about 5.5 minutes on the six-core machine (332 seconds at 9 to 27
percent background load; not an idle measurement), and a forecast from fitted weights is
under a tenth of a second. Refitting at every weekly origin would be about 89 hours. So
`headroom neural` refits every *k* origins and forecasts every origin from its own fresh
inputs with the last fitted weights; *k* is in the checkpoint's name and in every table
that reports it. That is a disadvantage to the neural model, which the statistical and
LightGBM models (refitted weekly) did not carry, so a neural win is conservative and a
neural loss has to be read with it. The multi-quantile loss is trained on the project's
own 199-level scoring grid, and the model sees demand only on synthetic dates, like the
statistical models: no holidays, unlike LightGBM. The refit flag in
`headroom.backtest.origins`, which nothing read before, is what drives it.

**Result, 2026-09-15 (monthly refits).** N-HiTS loses to ETS at every level: CRPS minus
ETS city +28.39 [+21.72, +40.03], borough +7.19 [+5.97, +9.39], dispatch area +1.98
[+1.82, +2.29], coverage at 90 percent 0.58 to 0.68, about 18.3 hours of fitting. The
refit schedule is not the cause, and a conformal version of its median is still behind.
Section 9's candidate 3 is supported more strongly than this section expected;
`docs/neural-verdict.md` has the reading and what it does not claim.

### 2.7a A pretrained foundation model, with the leak stated (added 2026-09-13)

TimesFM, Google Research's pretrained time-series foundation model, is added as a third
contender beside N-HiTS and PatchTST, on Peter's decision. It is a different kind of
model from those two: they are trained on the 37 series, and TimesFM is used zero-shot,
with no training on this data at all. It is included because "does a pretrained model
beat the statistical baselines on real demand, and do its intervals hold through a
shift" is the question practitioners are asking now, and the answer is worth more than
another architecture trained from scratch.

**Pinned 2026-09-13: TimesFM 2.5, 200M parameters, Apache 2.0.** 3.0 has the same
documented cutoffs and non-commercial weights. The pretraining corpus is recorded in
`docs/methods.md`: the latest documented real data ends November 2023, none of it is NYC
emergency dispatch data, and it does contain the 2020 shift in other series (Google
mobility, influenza-like illness, Wikipedia and search behaviour). The primary clean
window therefore starts 2023-12-01 (about 133 weekly origins), with a cross-check window
from the model's release on 2025-09-15 (about 40). Its intervals are built by conformal
around its median, not taken from its quantile head, which stops at the 0.1 and 0.9
quantiles and would otherwise need an invented tail.

**The leak.** Every other model here honours the backtest's guarantee that a forecast
made at an origin sees nothing after it. TimesFM cannot. Its weights were trained years
after most origins, on a corpus that includes the 2020 period in other series and may
include public data overlapping this one. A forecast of March 2020 from a model that has
seen how series behaved in 2020 is not a fair contest, and it would flatter exactly the
headline coverage chart. It is handled in four ways:

- The pretraining corpus and its end date are taken from Google's paper and model card
  and recorded in `docs/methods.md`, with what is known and not known about overlap with
  NYC open data.
- TimesFM is scored on two sets of origins, reported side by side and never pooled: the
  full backtest, marked as exposed to the leak, and a **clean window** of origins whose
  whole horizon falls after the pretraining data ends. The clean window is the result;
  its intervals will be wide because it is short, and the README says so.
- If the pretraining data ends after this dataset does, there is no clean window, and
  the README says TimesFM could not be evaluated fairly rather than reporting the
  exposed numbers as a result.
- TimesFM does not appear in the coverage-through-the-2020-shift chart without the leak
  stated beside it.

Otherwise it is treated like every other model: the same 1,095-day context, the same
14-day horizon and quantile grid, conformal intervals fitted on its forecasts, skill
against seasonal naive. With no training, each origin is inference only, so it is
expected to run at every weekly origin; that is measured before it is relied on.
Fine-tuning on this data is out of scope, because it would turn a zero-shot result into
a third trained model and blur the one question it is here to answer.

Before it is committed to: TimesFM must install and run under Python 3.13, which this
project requires. If it does not, that is recorded and the addition is dropped.

**Gate passed, 2026-09-17.** It installs and runs under Python 3.13. One clarification the
pin above needs: the 2.5 that is pinned is the **checkpoint**, and the PyPI package
`timesfm` is numbered separately, so the Apache 2.0 2.5 weights are loaded through the
3.0.x library, which is the line that still exposes them. `docs/methods.md`, "As built",
records that and the settings; the licence and the leak are unaffected. Built as
`headroom.models.foundation` and run by `headroom zeroshot`.

### 2.7b Gradient boosting, to separate global learning from deep learning (added 2026-09-13)

A global LightGBM model is added on Peter's decision. It was planned through Nixtla
MLForecast so it would sit in the same stack as the statistical and neural models; it was
built without it, for the reasons under "As built" below.

It is here because the comparison otherwise confounds two things. The statistical models
are fitted per series with no features; N-HiTS and PatchTST are global models across all
37 series and are deep learning. If a neural model wins, nothing says whether the gain came
from the architecture or from learning across series. A global gradient-boosted model has
the second without the first, and it is what a forecasting practitioner reaches for first
(the M5 competition was won by LightGBM, not a neural network). The neural verdict is
sharper with it: "N-HiTS beat ETS" is weak, and "N-HiTS beat ETS, and LightGBM got most or
none of that gain at a fraction of the cost" is the judgement this project exists to show.
LightGBM over XGBoost because it is faster on CPU at this size and is what the published
forecasting benchmarks use; MLForecast takes either, so the choice is cheap to revisit.

The design:

- **Features, all known at the origin.** Lags of the series (1, 7 and 14 days back, and
  further weekly lags within the trailing window), rolling means over the window, day of
  week, day of year, and public holidays from `headroom.data.calendar`. Calendar features
  are within scope (section 2.8); weather and events are not. Every lag feature is built
  from values at or before the origin, and a test asserts it.
- **One global model per refit** across the 37 series, with the series identity and its
  hierarchy level as categorical features, on the same 1,095-day trailing window.
- **The 14-day horizon is direct:** the horizon is handled without feeding the model's own
  forecasts back in as inputs, so errors do not compound across days. Whether that is one
  model per horizon step or one model with the step as a feature is settled in week 2 and
  recorded.
- **Intervals come from conformal around its median forecast,** exactly as for TimesFM.
  Quantile regression would need one model per level, and this project's 199-level
  scoring grid makes that impractical; conformal keeps the interval method identical
  across the models that do not produce a full quantile grid of their own.
- **Its advantage is stated.** It is the only model that sees holidays. That is recorded
  beside its result, not buried, because a skill number that owes part of its gain to
  information the other models were denied has to say so.

It is cheap: seconds per fit on this panel, so it is expected to refit at every origin.
That is measured before it is relied on. PatchTST is dropped before it.

**As built, 2026-09-13.** Four decisions the design above left open or got wrong, each
recorded in `docs/methods.md` with its evidence:

- **Not through MLForecast.** The rows are built in `headroom.models.boosting` and passed
  to LightGBM's native API. A direct model needs the holiday flag of the *target* day,
  which is a different day for each horizon step, and the no-look-ahead test needs to
  corrupt the future and see every feature of a row stay put. Both are a small tested
  function of our own and opaque inside a wrapper. LightGBM is the model either way.
- **The model is handed dates.** The backtest's forecaster interface passes demand only,
  so a model cannot know when it is. This model gets the dates of its training window and
  of the fourteen days ahead, never a demand value after the origin, and no feature
  identifies the year.
- **One model with the horizon step as a feature,** not one per step; each series divided
  by its own training-window mean so the city does not swamp the dispatch areas.
- **Its distribution is a conformal predictive distribution** from its own errors at the
  previous 52 origins (`headroom.conformal.predictive`), with the same window, feedback
  rule and order-statistic convention as the conformal intervals. The first 53 origins
  have no distribution, so every comparison that includes LightGBM is made on origins 53
  to 963 for all models.

**Measured: about 7 seconds a fit on an idle six-core machine,** so refitting at every
weekly origin is about 1.9 hours. A figure of 29 seconds recorded here on 2026-09-13 was
taken on a busy machine and is retracted (`docs/methods.md`).

**Result, 2026-09-14: a tie with ETS at every level.** Paired CRPS difference against
ETS, origins 53 to 963: city -1.65 [-6.81, +5.98], borough -0.48 [-1.25, +0.62],
dispatch area +0.007 [-0.067, +0.098]. Learning across series, with holidays, bought
nothing measurable over the best per-series model.

### 2.8 Out of scope, on purpose

- Exogenous regressors (weather, holidays) beyond calendar features. Named in Deferred.
- Intraday forecasting and shift-level rotas. Daily demand to daily staffing.
- Any live service. The dashboard is static, built from the backtest's outputs.
- The employer's or any Canadian provincial health series.

## 3. Data

| Source | Size | What it gives | Access and terms |
|---|---|---|---|
| NYC emergency medical dispatch incidents (open data) | 29,977,935 incidents, 2005-01-01 to 2026-06-30, with borough and dispatch area (measured 2026-09-12) | Daily counts on a two-level hierarchy with real shifts | NYC Open Data terms; aggregated by the loader before anything is stored, only aggregates committed. `docs/data.md` |
| NHS England monthly emergency department attendances by provider (optional) | About 200 providers by month since 2010 | Wide hierarchy at monthly resolution | Open Government Licence; loader handles format changes across years |
| Calendar features (own) | | Day of week, public holidays for the jurisdiction | Own |

Nothing raw is committed; loaders verify checksums; every licence is recorded.

## 4. Architecture

```
headroom/
  data/        nyc_ems.py (fetch, aggregate to daily by borough and dispatch area), nhs_ae.py (optional,
               monthly by provider and region), calendar.py, checks.py (gaps, outliers, regime markers)
  hierarchy/   spec.py (summing matrix), build.py
  backtest/    origins.py (rolling origin, refit schedule), run.py, store.py (Parquet per method)
  models/      baselines.py (seasonal naive), stats.py (ETS, Theta, AutoARIMA, MSTL via StatsForecast),
               neural.py (N-HiTS via NeuralForecast, multi-quantile loss, CPU, scheduled refits),
               foundation.py (TimesFM zero-shot, median forecast, CPU; clean-window origins),
               boosting.py (global LightGBM, own lag and calendar features, direct horizon)
  conformal/   split.py, aci.py (adaptive conformal inference), agaci.py (aggregated experts), apply.py
  reconcile/   mint.py (point), probabilistic.py (bootstrap reconciliation), verify.py (coherence)
  score/       crps.py, pinball.py, coverage.py (rolling window), width.py, skill.py, bootstrap.py (block, over origins)
  decide/      newsvendor.py (optimal quantile from costs), staffing.py (quantile to staff), realised_cost.py
  report/      tables.py, charts.py (fan charts, coverage through time, reconciliation, staffing), export.py
               (JSON for the dashboard)
  cli.py       headroom data build | backtest | conformal | reconcile | decide | report | export
dashboard/     static site (HTML, a small chart library, precomputed JSON): fan charts by level, the coverage
               chart, reconciliation view, a service-level slider that reads the staffing table;
               Azure Static Web Apps at capacity.peterparker.ca
docs/          data.md, methods.md (assumptions, incl. conformal under dependence), neural-verdict.md, decision.md
```

**As built, report (2026-09-15).** `report/tables.py` and `headroom report` exist; charts
and export do not yet. The README's second placeholder table ("Reconciliation and the
rota") mixed per-level, per-service-level and per-method numbers in one row, which no
single row can hold, so the command writes it as two tables: CRPS before and after MinT
per level, with the coherence breach stated above it, and staffing per service level and
method with realised cost against the oracle. The first table keeps its columns; its worst
window is the minimum over the whole period, which falls in spring 2020 for every method,
and is dated rather than given an interval, since a bootstrap over a single worst stretch
means nothing. Fit time is the median model time per origin times the schedule, so time lost to a
busy or sleeping machine is not charged to a model.

**As built, the dashboard (2026-09-18).** `headroom export` writes two files rather than
one. The fan chart's bands for every node at every origin are most of the payload, and the
tables are a few kilobytes, so `forecast.json` is fetched second and the page draws its
tables before it arrives. Both are validated against a JSON Schema in
`headroom.report.export` before they are written: the page is served from a CDN with no
backend, so a payload that has lost a field fails as a blank panel on a stranger's machine
rather than in a terminal. Three deviations from the line above, all deliberate:

- **The bands are five quantiles, not the scoring grid's 199.** Each level is another array
  per node per origin in a file a browser downloads. The outer band, the quartiles and the
  median are what a fan is read for.
- **The service-level slider stops at the levels the staffing table was computed at**, which
  are 50 to 95 percent in steps of five, plus whatever `inputs/decision.toml` names. A
  staffing number is only as continuous as the levels it was measured at, and each one is
  another pass over every origin, area and day. The slider also moves a line on the fan
  chart, and that line is interpolated between the exported bands; the page says so.
- **The page recomputes the trailing mean for the coverage panel**, so the window can be
  changed without re-exporting. It is the only arithmetic in the JavaScript, and it mirrors
  `headroom.report.charts.rolling_mean`, which is now public for that reason.

`headroom serve` serves the site with the headers **and the content types** in
`dashboard/staticwebapp.config.json`. A plain file server sends neither, which on
the Intervention Targeting Engine hid a broken chart on the live site for two weeks while every local check looked correct.

**The page is part of peterparker.ca even though another host serves it**, so it takes that
site's palette, its two fonts and the 3px rule over the page, exactly as the Intervention Targeting Engine's
demo does. The
fonts are copied into `dashboard/fonts/` rather than linked from the main site: the content
security policy allows no off-origin request, and a font is a request. Both are SIL Open
Font License and `dashboard/fonts/LICENSE.txt` records where they came from. Chart colours
are named in the stylesheet and never in the JavaScript, so one palette follows the page into
dark mode.

### Tests that matter

The summing matrix reproduces every aggregate from its leaves; reconciled forecasts are
coherent at every origin to numerical precision; CRPS from quantiles matches a closed-form
check on a normal fixture; the adaptive conformal update tracks a synthetic shift in a
fixture within the expected lag; block bootstrap respects origin ordering; the newsvendor
quantile equals the cost ratio on fixtures; the dashboard JSON validates against a schema.

## 5. Week by week

| Dates | Built | Done when |
|---|---|---|
| Sep 12 to 18 2026 | NYC loader, aggregation, checks, hierarchy; rolling-origin harness; baselines and statistical models with quantiles; CRPS, pinball, coverage, width, skill, block bootstrap; split and adaptive conformal; the coverage-through-shift chart | Skill table with CIs for every statistical method; coverage chart through March 2020 |
| Sep 19 to 25 2026 | LightGBM global model; N-HiTS and PatchTST; TimesFM zero-shot with its clean window; MinT and probabilistic reconciliation with coherence verified; decision layer and realised cost; neural verdict; NHS dataset if time allows; the rejected-approach write-up; README | Every table in section 1 filled |
| 2026-09-18 | Dashboard exported and deployed on the reused static pattern; repository public; `v0.1.0` | Dashboard live |

First to drop if behind: the NHS dataset; PatchTST (N-HiTS stays as the named neural
model, TimesFM as the pretrained one, and LightGBM as the global non-neural one);
probabilistic reconciliation (point MinT with
conformal on the reconciled series stays). TimesFM is dropped only if it will not run
under Python 3.13, and that is recorded. The baselines, probabilistic scoring, coverage through the shift, reconciliation
coherence, the decision layer and the neural verdict are not droppable.

### 2.9 The training window is trailing (added 2026-09-12)

Each forecast sees the three years before its origin, not the whole record. The plan did
not say either way, and the difference turned out to matter enough to write down.

Statistically, an expanding window would fit the last origin on twenty-one years spanning
two regime changes. Practically, it runs from 1,095 days of history at the first origin to
7,829 at the last, so every fit gets more expensive and a run's cost cannot be projected
from its first origins, which is how the first estimate of this project's compute came out
roughly half of what it should have been.

Measured on seasonal naive over the same 964 origins: CRPS is unchanged (163.12 against
163.19, far inside either interval), coverage at 90 percent nominal **improves** from
0.8643 to 0.8834, width rises from 829 to 886, and the runtime halves. `docs/methods.md`
has the table.

## 6. Cost

Everything runs on the laptop's CPU: the statistical models in seconds per series, the
neural models in minutes per fit on daily data of this size. The dashboard is static on a
free tier. No model vendor is called.

| Item | Basis | CA$ |
|---|---|---:|
| Compute | Laptop CPU | 0 |
| Hosting | Azure Static Web Apps free tier; custom domain on the owned domain | 0 |
| Data | Open datasets | 0 |
| TimesFM | Open weights, downloaded once, run on the laptop's CPU | 0 |
| LightGBM | Open source, about 7 seconds per fit on a six-core desktop CPU | 0 |
| Reserve | A rented CPU box for a day if the full backtest with refits is too slow locally | 10 |
| **Total** | | **10** |

Well under the CA$25 line. Actual spend is recorded privately beside the estimate.

## 7. Handover

`headroom` is tagged v0.1.0 when the repository goes public. Nothing imports it. The
conformal and scoring modules are small and importable; the static decision-app pattern is
the Intervention Targeting Engine's, reused. The coverage chart and the neural verdict are
the two figures worth reusing elsewhere.

## 8. Risks

| Risk | Handling |
|---|---|
| Two weeks is tight and cleaning is underestimated | One dataset with a single aggregation step; the NHS set is optional and the first thing dropped |
| Conformal subtleties under dependence | Adaptive methods designed for it; the assumption and its guarantee stated in the README next to the chart; split conformal kept to show the failure |
| The neural models win everywhere, or nowhere | Either is reported with intervals; the verdict document is written whichever way |
| LightGBM's lag or rolling features reach past the origin, which flatters it silently | Every feature is built from values at or before the origin, and a test corrupts the future of the panel and asserts no feature changes, the same pattern as the conformal feedback test |
| TimesFM's pretraining overlaps the backtest, so its result is flattered | Scored separately on a clean window after its pretraining ends; the exposed full-history numbers are labelled and never pooled; no clean window means no reported result (section 2.7a) |
| Reconciliation hurts leaf accuracy | Reported per level; MinT shrinkage parameter chosen on a validation window and stated |
| The staffing numbers are taken as advice | Inputs are explicit and replaceable; the README says the ratio and costs are illustrative |
| Open data terms | NYC Open Data and the Open Government Licence permit this use; recorded in `docs/data.md` |
| Employer boundary | None. Other jurisdictions' public data; no provincial series |

## 9. What might not work, named in advance

1. **Split conformal through the shift.** Expected: coverage collapses in March 2020 and
   recovers only when the calibration window rolls past it, while the adaptive method
   tracks nominal within weeks. The rolling coverage chart is the evidence.
2. **Choosing the model by MAE.** Expected: the MAE winner is not the CRPS winner on at
   least one level, and its intervals are worse calibrated; the table shows the divergence.
3. **A global neural model as the default forecaster.** Expected: competitive at the top
   level, no better or worse than the statistical models at the leaves, at many times the
   compute; this is the neural verdict itself. With LightGBM beside them (section 2.7b)
   the candidate becomes sharper: if the global boosted model matches the neural models,
   the gain was from learning across series and the deep learning bought nothing.
4. **Scoring a pretrained model on history it was trained after** (added 2026-09-13).
   Expected: TimesFM looks markedly better on the full backtest, especially through
   March 2020, than on the clean window after its pretraining ends. The gap between the
   two is the evidence, and it is what an evaluation that ignored the leak would have
   reported.

Whichever produces the clearest evidence is written up, whichever way it falls. The
README's "What did not work" section is where it ends up.

## 10. Definition of done

- [x] Seasonal naive and statistical baselines reported first, with every later result as skill against them
- [x] CRPS and pinball loss per quantile, per method and level, with block-bootstrap CIs
- [x] Empirical coverage at three nominal levels plotted in a rolling window through the 2020 shift, adaptive against split conformal, with widths
- [x] Conformal assumptions stated in the README beside the chart
- [x] MinT and probabilistic reconciliation; coherence verified at every origin; effect on accuracy per level reported
- [x] Decision layer: staffing at a stated service level from the reconciled distribution; newsvendor quantile from stated costs; realised cost per method against an oracle
- [x] N-HiTS and PatchTST against the best statistical model, paired with CIs; `docs/neural-verdict.md` says where they did not earn their complexity
- [x] LightGBM global model against the best statistical and best neural model, paired with CIs, its holiday features stated
- [x] TimesFM zero-shot against the best statistical model on its clean window, with the leak and its pretraining cutoff stated, in `docs/neural-verdict.md`
- [x] Static dashboard live at capacity.peterparker.ca
- [x] One rejected approach documented with evidence
- [ ] Repository public, `v0.1.0` tagged

## 11. Deferred

| Deferred | Kept so the door stays open |
|---|---|
| Weather and event regressors | The models accept exogenous features; the loader has a calendar module to extend |
| Intraday demand and shift-level rotas | Aggregation resolution is a parameter; the decision layer is per period whatever the period |
| A live-updating forecast | The export produces the dashboard JSON; a scheduled run would republish it |
| Additional jurisdictions | One loader each; the hierarchy spec is data-driven |
| Cross-learning between hierarchies (NYC and NHS jointly) | Global models already take multiple series; a joint run is a configuration |
