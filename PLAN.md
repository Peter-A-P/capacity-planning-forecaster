# Plan: Capacity Planning Forecaster

**Written:** 2026-09-07. **Status:** building, week 1 started 2026-09-12.

**Build:** two weeks, Sep 12 to Sep 25 2026. **Package:** `headroom`. **Fed by:** nothing
in the portfolio. **Feeds:** nothing; reuses 01's static decision-app pattern.

> **Moved forward at rev. 4, 2026-09-12.** The plan slotted this for Jul 19 to Aug 1
> 2027. `docs/SEQUENCE.md` puts 08 in the independent set with 01 and 09, and section 1
> above is why nothing had to wait: this project calls no model vendor, so neither the
> 04 gateway nor the 03 gate is on its path, and its data is twenty-one years of
> published history that does not need calendar time to accumulate. Brought forward on
> Peter's decision; the two-week duration, the budget and the scope are unchanged.
>
> One deliverable moves out of the two weeks with it. The dashboard reuses 01's static
> decision-app pattern, and 01 builds that in its own week 7 (Oct 19 to 25 2026). Rather
> than invent the pattern here and have 01 inherit it, the dashboard is held until 01
> has it. Week 2 below ships everything else, and the dashboard and `v0.1.0` follow after
> Oct 25. Section 10's definition of done is unchanged; the dashboard box is simply the
> last one ticked.

This project calls no language model, so neither the 04 gateway nor the 03 gate is on its
path. Every number below comes from a rolling-origin backtest with block-bootstrap
intervals over forecast origins. It is the shortest project in the plan and the one whose
seniority signal per week is highest: forecasting portfolios report a point-estimate MAE
and stop; this one reports whether the intervals held when the world changed, whether the
sites sum to the region, and what the forecast means for the rota.

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
reported in a footnote for readers who look for it, never as the headline, and Rule C
candidate 2 shows what choosing a model by MAE would have done.

### 2.4 Adaptive conformal, with the assumption stated

Exchangeability does not hold for time series, so split conformal is expected to fail
through the shift and is kept as the comparison. The main method is adaptive conformal
inference (the online update of the miscoverage level from realised coverage), with an
aggregated-expert variant to remove the step-size choice, applied to each model's point or
median forecast per series and horizon.

Implemented directly rather than through MAPIE, which the plan repository's one-line stack
summary named (corrected 2026-09-12). Three reasons, in order of weight. The backtest needs
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

### 2.6 The decision layer is a newsvendor

Staffing at a chosen service level is the demand quantile at that level divided by a
stated demand-per-staff ratio, with the under-staffing and over-staffing costs stated as
inputs. The newsvendor result gives the optimal quantile as the ratio of those costs, so
the service level a planner chooses is shown next to the one the costs imply. The
backtest realises each method's staffing decision against actual demand and sums the cost,
against an oracle. Inputs are a table in the README so a reader substitutes their own.

### 2.7 Neural models on CPU, named, and allowed to lose

N-HiTS and PatchTST through NeuralForecast with a multi-quantile loss, trained as global
models across the series on the laptop's CPU (small data; minutes per fit). The paired
comparison against the best statistical model is per level and horizon with intervals. If
the neural models do not beat the statistical ones at the leaf level, or only at the top,
the README says exactly that. Judgement reads as seniority; "deep learning won" reads as
naive.

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
               neural.py (N-HiTS, PatchTST via NeuralForecast, multi-quantile loss, CPU),
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
| Sep 19 to 25 2026 | LightGBM global model; N-HiTS and PatchTST; TimesFM zero-shot with its clean window; MinT and probabilistic reconciliation with coherence verified; decision layer and realised cost; neural verdict; NHS dataset if time allows; Rule C; README | Every table in section 1 filled |
| After Oct 25 2026 | Dashboard exported and deployed on 01's static pattern; `v0.1.0`; repository public | Dashboard live |

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

Well under the CA$25 line. Actuals go in the plan repository's STATUS next to the estimate.

## 7. Handover

`headroom` v0.1.0 once the dashboard ships, after 2026-10-25. Nothing imports it. The conformal and scoring modules are
small and importable; the static decision-app pattern is 01's, reused. The coverage chart
and the neural verdict are figures for the portfolio site.

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

## 9. Rule C candidates

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

Whichever produces the clearest evidence becomes `docs/rejected.md`.

## 10. Definition of done

- [ ] Seasonal naive and statistical baselines reported first, with every later result as skill against them
- [ ] CRPS and pinball loss per quantile, per method and level, with block-bootstrap CIs
- [ ] Empirical coverage at three nominal levels plotted in a rolling window through the 2020 shift, adaptive against split conformal, with widths
- [ ] Conformal assumptions stated in the README beside the chart
- [ ] MinT and probabilistic reconciliation; coherence verified at every origin; effect on accuracy per level reported
- [ ] Decision layer: staffing at a stated service level from the reconciled distribution; newsvendor quantile from stated costs; realised cost per method against an oracle
- [ ] N-HiTS and PatchTST against the best statistical model, paired with CIs; `docs/neural-verdict.md` says where they did not earn their complexity
- [ ] LightGBM global model against the best statistical and best neural model, paired with CIs, its holiday features stated
- [ ] TimesFM zero-shot against the best statistical model on its clean window, with the leak and its pretraining cutoff stated, in `docs/neural-verdict.md`
- [ ] Static dashboard live at capacity.peterparker.ca
- [ ] One rejected approach documented with evidence (Rule C)
- [ ] Repository public, `v0.1.0` tagged

## 11. Deferred

| Deferred | Kept so the door stays open |
|---|---|
| Weather and event regressors | The models accept exogenous features; the loader has a calendar module to extend |
| Intraday demand and shift-level rotas | Aggregation resolution is a parameter; the decision layer is per period whatever the period |
| A live-updating forecast | The export produces the dashboard JSON; a scheduled run would republish it |
| Additional jurisdictions | One loader each; the hierarchy spec is data-driven |
| Cross-learning between hierarchies (NYC and NHS jointly) | Global models already take multiple series; a joint run is a configuration |
