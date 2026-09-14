# Methods and assumptions

What the numbers in this repository mean, and what has to be true for them to mean it.
Every measurement on this page was produced by the code in this repository on the panel
described in [data.md](data.md) and is dated.

---

## The backtest

### Forecast origins

A forecast origin is a day on which a forecaster may know everything up to and including
that day and nothing after it. The runner hands a model `panel[:, :origin + 1]` and
nothing else: no dates, no future values, no argument through which it could reach
forward. `tests/test_backtest.py` asserts the training and target slices are disjoint at
every origin and records the shape of every call the runner makes.

| Choice | Value | Why |
|---|---|---|
| Horizon | 14 days | The horizon a rota is built over, and two full weekly cycles, so a method that only gets the weekly shape right is visibly not enough |
| Spacing | 7 days | Daily origins give 14x the compute for forecasts that are nearly the same forecast, and worsen the dependence the bootstrap has to handle. Weekly spacing also holds day of week fixed, so no method gets more Mondays than another |
| Refitting | Every 4th origin | Roughly monthly. It is what the CPU budget allows for the neural models, and the statistical models are held to the same schedule rather than being given an advantage the expensive models were denied |
| Minimum history | 3 years | Three annual cycles, the least a model claiming to know the annual shape can be fitted on |
| **Training window** | **3 years, trailing** | See below |

On the NYC panel this gives **964 origins, 2007-12-31 to 2026-06-15**, and 241 refits.

### The training window is trailing, not expanding

Each forecast sees the 1,095 days before its origin, not everything back to 2005.

The statistical reason: under an expanding window the last origin would be fitted on
twenty-one years spanning two regime changes, and a model that averages 2008 and 2025
demand is describing a city that no longer exists.

The practical reason, which is why it is a fixed window rather than a detail: an expanding
window runs from 1,095 days of history at the first origin to **7,829 at the last**, so
every fit gets steadily more expensive and a run's total cost cannot be estimated from its
first origins. With a fixed window the cost per origin is flat and the total is knowable
in advance.

Measured both ways, seasonal naive over the same 964 origins, 2026-09-12:

| Window | City CRPS | 95% interval | Coverage at 90% | Mean width | Runtime |
|---|---:|---|---:|---:|---:|
| Expanding | 163.186 | [152.46, 176.33] | 0.8643 | 828.6 | 474 s |
| **Trailing 1,095 days** | **163.123** | [152.58, 176.08] | **0.8834** | 885.6 | **212 s** |

The trailing window costs nothing in CRPS, the difference being far inside either
interval, and it **covers better**: 0.8834 against 0.8643 at 90 percent nominal. It buys
that with width, 886 against 829, which is the honest reading of it. Three years of
recent residuals describe the current regime's spread better than twenty-one years of
mixed ones do. It also halves the runtime, but that is not why it is the default.

An origin whose 14-day horizon would run past the end of the record is not included
rather than being scored on a short horizon, which would quietly weight the end of the
record differently from the rest.

### Scoring on the fly

Keeping every quantile forecast would be 794 MB for one model on this panel, before the
four the tables compare. The dense scoring grid is consumed at the origin that produced
it; only the scores and the sparse reporting grid are kept.

---

## The scores

### CRPS is a quadrature, and the grid matters

There is no closed form for the CRPS of a distribution given only as quantiles. What
there is, is the identity

    CRPS(F, y) = 2 * integral over tau in (0, 1) of pinball_tau(F_inverse(tau), y)

so CRPS is twice the area under the forecast's own pinball losses. That ties the two
headline scores together: the pinball table and the CRPS column are the same numbers,
integrated or not.

The interior is integrated by the trapezoidal rule over the quantile grid. Measured
against the closed form for a normal, 20,000 draws, 2026-09-12:

| Grid | CRPS | Exact | Relative error |
|---|---:|---:|---:|
| Reporting grid, 11 levels | 8.1505 | 8.4313 | -3.33e-02 |
| **Scoring grid, 199 levels** | **8.4311** | 8.4313 | **-2.25e-05** |
| 999 levels | 8.4313 | 8.4313 | -1.81e-06 |

The error is always downward, because the integrand is concave in tau near its peak and a
chord runs below the curve. Two consequences:

* **Absolute CRPS is reported on the 199-level scoring grid, never the reporting grid.**
  The reporting grid is chosen so a reader can follow a table and is 1,500 times less
  accurate as a quadrature.
* Every method is scored on the same grid, so the remaining bias is common to all of them
  and the skill ratios the tables report are almost unaffected by it.

### The tails are integrated in closed form, not padded with zero

**This is an assumption, and it is stated because it is one.** The integrand vanishes at
tau = 1 only when the observation lies strictly below the top of the forecast's support.
When an observation lands outside the quantile grid, which is exactly what a distribution
shift produces, the integrand at that end is proportional to the size of the miss.

Padding it to zero, which is the obvious implementation, discards part of the penalty: on
this grid about half a percent of the score, growing with the miss, and only ever for the
forecasts that missed. A bias that lets through exactly the errors this project exists to
measure is not one to leave in.

So each tail is integrated exactly under the explicit assumption that **the distribution
is flat beyond the outermost quantile levels**. That assumption is conservative rather
than flattering: a real distribution with a longer tail puts its quantiles further from
the observation and scores better, so every CRPS here is an upper bound on what a
heavier-tailed reading of the same forecast would give.

The check that this is right: a point forecast must score exactly the absolute error, and
it does, to floating point.

### Coverage

Interval bounds are **looked up in the quantile grid, never interpolated**, so no
approximation enters the one number that has to be exactly what it says it is. The
reporting grid therefore contains both bounds of every nominal coverage reported: 80, 90
and 95 percent.

The interval is closed. Daily counts are integers and a quantile landing exactly on one is
common enough that treating the endpoint as a miss would bias coverage downward for no
reason.

The rolling window is **trailing, not centred, and 90 days long**. A centred window would
let an origin's coverage be computed partly from days after it, which is the look-ahead
the whole backtest exists to avoid, and would smear the March 2020 break backwards over
six weeks that had not seen it. Positions before the window is full are reported as
missing rather than as a short-window average.

Coverage is never reported without width beside it. An interval from minus infinity to
plus infinity covers everything.

---

## Confidence intervals: the moving block bootstrap

Every reported number carries an interval, and they all come from here.

The ordinary bootstrap resamples origins independently, which assumes a method's score on
one origin says nothing about its score on the next. That is false and obviously so: a
model badly calibrated in the first week of April 2020 was badly calibrated in the second.
The moving block bootstrap resamples contiguous runs of origins instead, so the dependence
inside a run survives into the resample.

`tests/test_score.py` asserts the drawn indices are contiguous and increasing inside each
block, because a bug that shuffled within a block would silently turn this back into the
independent bootstrap and narrow every interval in the project without failing anything
else.

### What the block length costs

City-level CRPS per origin, 964 origins, 2,000 resamples, 2026-09-12:

| Block | CRPS | 95% interval | Width |
|---:|---:|---|---:|
| 1 (independent) | 163.186 | [157.621, 169.676] | 12.055 |
| 7 | 163.186 | [153.718, 176.135] | 22.417 |
| 14 | 163.186 | [152.829, 176.734] | 23.905 |
| **28 (used)** | 163.186 | **[152.459, 176.326]** | 23.867 |
| 56 | 163.186 | [152.403, 175.703] | 23.301 |
| 91 | 163.186 | [152.195, 175.437] | 23.242 |

Two things to read here. Ignoring the dependence **halves the interval**, which is exactly
the false precision the blocks exist to prevent. And from 14 origins upward the width is
flat to within 3 percent, so the reported intervals are not a result about the block
length. 28 origins, four weeks, is longer than the weekly cycle and long enough to hold
most of a regime change.

Intervals are percentile intervals. The statistics being bootstrapped are means and
differences of means of scores, whose distributions are close to symmetric, so a
percentile interval makes no assumption the data has to earn.

Skill intervals are **paired**: both methods are resampled on the same drawn origins, so
the origin-to-origin variation, which dwarfs the difference between methods, cancels.

---

## The baseline

Seasonal naive, because it is genuinely hard to beat on daily demand with a strong weekly
cycle, and a CRPS reported without it beside it says nothing about whether the model did
any work.

The point forecast is what happened on the most recent same weekday the forecaster had
seen: seven days back for horizon steps 1 to 7, fourteen for 8 to 14. Nothing is fitted.

It is given a predictive distribution the same way, from what actually happened: the
empirical quantiles of the same rule's residuals across the training history. Two
properties keep it from being a strawman.

* **Per horizon.** A 14-day-ahead seasonal naive is more uncertain than a 1-day-ahead one.
  One pooled residual distribution would hand every later model an easy win at short
  horizons.
* **Empirical.** It inherits whatever skew and heavy tails the series has. Assuming normal
  residuals on count data with a long right tail would weaken the baseline, and a
  weakened baseline makes every skill number in the project a lie.

Forecasts are floored at zero, since a count cannot be negative. Its quantiles never
crossed at any of the 964 origins.

### First measured result

Seasonal naive over 964 origins, 14-day horizon, 2026-09-12. This is the baseline the
tables will be reported against, not a result about any model.

On the trailing window, which is the default.

| Level | CRPS | 95% interval | Coverage at 90% | Mean width |
|---|---:|---|---:|---:|
| City | 163.12 | [152.58, 176.08] | 0.8834 | 885.6 |
| Borough | 40.62 | [38.62, 43.19] | 0.8896 | 222.5 |
| Dispatch area | 10.99 | [10.67, 11.39] | 0.8959 | 61.6 |

**The baseline undercovers at every level and every nominal rate.** That is not a defect
in the implementation; it is what an empirical residual distribution does on a series
whose variance is not constant through time, and it is the gap the conformal work exists
to close.

**Worst 90-day window at the city, 90 percent nominal: coverage 0.156, ending
2020-05-04.** Whole-period coverage at the same level and rate is 0.864. One number hides
a collapse; the rolling window is why the chart is a chart. This is the raw material for
`PLAN.md` section 9's first Rule C candidate, and the thing adaptive conformal intervals
have to fix.

---

## The statistical models

ETS, Theta and MSTL through StatsForecast, each giving its own prediction quantiles, at
the same 964 weekly origins as the baseline, on the same 1,095-day trailing window, with
every model refitted at every origin. AutoARIMA is deferred (see Compute below).

### Measured result

`headroom score --models ETS,Theta,MSTL --step 7`, 2026-09-13. Seasonal naive is rerun on
the same origins inside the command, so skill is paired, and it reproduced the baseline
table above exactly on a different machine. Every interval is a 95 percent moving block
bootstrap over origins (block 28, 2,000 resamples, seed 0); skill intervals resample both
methods on the same drawn origins.

Coverage and width are at 90 percent nominal, of **each model's own quantiles, with no
conformal step**. Nothing guarantees these coverages; they are what the model's
distributional assumptions delivered on this record.

City:

| Method | CRPS | Skill against seasonal naive | Coverage at 90% | Mean width |
|---|---|---|---|---|
| Seasonal naive | 163.12 [152.58, 176.08] | 0 | 0.883 [0.870, 0.899] | 885.6 [861.5, 912.1] |
| ETS | 133.92 [125.28, 143.53] | +0.179 [+0.164, +0.198] | 0.919 [0.908, 0.933] | 835.8 [806.3, 866.9] |
| Theta | 135.89 [127.12, 145.79] | +0.167 [+0.151, +0.187] | 0.912 [0.901, 0.926] | 834.4 [801.7, 867.6] |
| MSTL | 148.50 [137.26, 162.80] | +0.090 [+0.039, +0.130] | 0.781 [0.762, 0.799] | 613.2 [590.5, 640.6] |

Borough:

| Method | CRPS | Skill against seasonal naive | Coverage at 90% | Mean width |
|---|---|---|---|---|
| Seasonal naive | 40.62 [38.62, 43.19] | 0 | 0.890 [0.881, 0.899] | 222.5 [217.2, 227.9] |
| ETS | 32.09 [30.38, 34.15] | +0.210 [+0.201, +0.222] | 0.911 [0.902, 0.921] | 192.8 [186.5, 199.5] |
| Theta | 32.51 [30.77, 34.60] | +0.200 [+0.188, +0.213] | 0.899 [0.890, 0.909] | 190.3 [183.2, 198.1] |
| MSTL | 36.50 [34.38, 39.26] | +0.101 [+0.063, +0.130] | 0.738 [0.724, 0.751] | 138.8 [133.7, 144.8] |

Dispatch area:

| Method | CRPS | Skill against seasonal naive | Coverage at 90% | Mean width |
|---|---|---|---|---|
| Seasonal naive | 10.99 [10.67, 11.39] | 0 | 0.896 [0.891, 0.901] | 61.6 [60.5, 62.7] |
| ETS | 8.21 [7.94, 8.58] | +0.253 [+0.246, +0.258] | 0.904 [0.897, 0.910] | 48.1 [47.0, 49.4] |
| Theta | 8.32 [8.04, 8.69] | +0.243 [+0.236, +0.249] | 0.890 [0.884, 0.896] | 46.9 [45.6, 48.3] |
| MSTL | 9.83 [9.48, 10.28] | +0.106 [+0.088, +0.120] | 0.704 [0.696, 0.711] | 34.7 [33.7, 35.8] |

What this says:

* **ETS and Theta beat seasonal naive clearly at every level**, by 17 to 25 percent in
  CRPS, with intervals nowhere near zero. Skill grows down the hierarchy, from about 0.17
  at the city to 0.25 at the dispatch area, where series are noisier and a naive copy of
  last week carries more of that noise forward.
* **ETS and Theta slightly over-cover at the city** (0.919 and 0.912), which with CRPS
  this much better means their intervals are reasonable rather than inflated.

### ETS is the best statistical model, by a small margin that is real

The skill intervals above overlap, but that is the wrong test: the two models were scored
on the same origins, and origin-to-origin variation dwarfs the difference between them.
The paired CRPS difference, Theta minus ETS, bootstrapped over origins as above:

| Level | Theta minus ETS | Share of ETS CRPS | Origins where ETS is better |
|---|---|---:|---:|
| City | +1.97 [+1.37, +2.63] | 1.5% | 60.7% |
| Borough | +0.42 [+0.30, +0.55] | 1.3% | 63.9% |
| Dispatch area | +0.106 [+0.090, +0.123] | 1.3% | 78.5% |

Every interval excludes zero. **ETS is the best statistical model at every level**, and it
is the model every later comparison (LightGBM, the neural models, TimesFM) is paired
against.

### MSTL: both its median and its spread are worse, and the spread is structural

Against ETS on the same origins, MSTL's median has **10 to 15 percent more absolute
error** (1.10 at the city, 1.11 at the borough, 1.15 at the dispatch area) and its 90
percent intervals are **27 to 28 percent narrower**. So the undercoverage is not a good
median with a bad spread; both are worse, and the narrow spread is the larger failure.

The cause of the spread, read from StatsForecast 2.1.1 (`statsforecast/models.py`,
`MSTL.forecast`): MSTL fits its trend forecaster, AutoETS with no seasonal component, to
the seasonally adjusted series (trend plus remainder), takes that model's prediction
intervals, and then adds the seasonal forecast to every quantile as a fixed shift. **No
uncertainty in the weekly or annual seasonal components reaches the interval.**

The coverage pattern by horizon agrees with that reading. At the city, MSTL's 90 percent
coverage is 0.68 one day ahead and rises steadily to 0.85 at fourteen days, while ETS
stays between 0.89 and 0.95 throughout. The missing seasonal variance is a roughly fixed
amount at every step; it matters most one day ahead, where the trend model's own interval
is narrowest, and is partly masked at longer horizons as that interval grows.

MSTL is kept in the tables as the model whose intervals should not be trusted without a
conformal step, not dropped.

### Choosing by MAE would not have misled here

`PLAN.md` section 9, Rule C candidate 2, expected the model with the lowest absolute
error of the median to differ from the one with the lowest CRPS on at least one level. On
these three models it does not: ETS, then Theta, then MSTL, by both scores at every level.

| Level | ETS | Theta | MSTL |
|---|---|---|---|
| City | 184.0 [173.0, 195.2] | 186.2 [175.1, 197.4] | 201.8 [188.1, 218.5] |
| Borough | 44.43 [42.30, 46.89] | 44.82 [42.64, 47.22] | 49.45 [46.86, 52.72] |
| Dispatch area | 11.51 [11.15, 11.96] | 11.63 [11.27, 12.07] | 13.29 [12.86, 13.84] |

Mean absolute error of the median, with block-bootstrap intervals. The candidate is not
supported by the statistical models. It stays open for the models still to come, where a
point-accurate model with poorly calibrated intervals is more likely.

Produced 2026-09-13 by a one-off analysis over the scored checkpoints. The report command
has to reproduce these tables before any of them reaches the README.

---

## Conformal intervals

### What each method assumes

**Split conformal** assumes **exchangeability**. Under it, the ``1 - alpha`` quantile of
calibration residuals gives finite-sample marginal coverage of at least ``1 - alpha``, with
no assumption about the model or the distribution. It is a remarkable guarantee and it does
not apply here: a time series is not exchangeable, and the period this project exists to
measure is when it is least exchangeable of all. Split conformal is kept because watching
it fail is the evidence, not because its guarantee is believed.

**Adaptive conformal inference** (Gibbs and Candes, 2021) assumes **nothing about the data
at all**. It updates the miscoverage level online, `alpha_next = alpha_now + gamma *
(target - missed)`, and its realised miscoverage rate over `T` steps converges to the
target at rate `O(1/T)` for any sequence, including an adversarial one.

**What it does not give is per-period coverage.** It is guaranteed to come back to nominal,
not to be at nominal over any particular window, and it can only respond to a shift after
the shift has already cost it coverage. **This is stated in the README beside the coverage
chart.** Reporting "coverage held through the shift" without it would be a claim the method
does not make, and it would be the most likely way for this project to be wrong in public.

**Aggregated adaptive conformal** (Zaffran and others, 2022) removes the choice of `gamma`
by running six step sizes from 0.001 to 0.5 as experts and weighting them online by their
realised pinball loss. Nothing is chosen with hindsight.

### The comparison is fair by construction

The classical split construction holds one calibration set aside forever. Every method here
instead uses the **same** trailing window of 52 origins, so the only difference between
split and adaptive conformal is whether `alpha` moves. Two things would otherwise differ at
once, in the adaptive method's favour.

### The feedback rule

A conformal method learns from its own past errors, so it is the part of this project most
able to cheat without anyone noticing. At an origin, the outcome of a 14-day-ahead forecast
made at the previous origin **has not happened yet**: with weekly origins it is two origins
away. `available_upto` is the only place that rule lives, and
`tests/test_conformal.py` corrupts the future of a score series and asserts that no width
already produced changes, so the guard cannot decay into a comment.

### The conformal quantile is an order statistic

The half-width is the `k`-th smallest score with `k = ceil((n + 1) * (1 - alpha))`, taken
directly as an order statistic and not through a quantile function. The usual quantile
convention interpolates over `n - 1` intervals, which lands one order statistic away from
what the construction asks for. That error is small, systematic, and **invisible in a
coverage table, because over-covering looks like caution rather than like a bug**: on an
exchangeable fixture it produced 0.9255 against a 0.90 target, where the order statistic
gives 0.9065.

### Measured result

Wrapping the seasonal naive median, 964 origins, 90 percent nominal, 2026-09-12.

| Method | City | City mean width |
|---|---:|---:|
| Base quantiles, no conformal | 0.8834 | 885.6 |
| Split conformal | 0.8965 | 982.8 |
| Adaptive, gamma 0.05 | 0.8969 | 1055.4 |
| Aggregated, 6 experts | 0.8941 | 999.6 |

**Conformal fixes the baseline's undercoverage.** Every method lands within half a point of
nominal at every level of the hierarchy, against a base forecast that was 1.7 points low at
the city and undercovered everywhere. It costs width: 983 against 886 at the city, which is
the honest price and is reported beside the coverage rather than under it.

### Worst 91-day window at the city, and the finding that matters

| Method | Worst window | Ends |
|---|---:|---|
| Base quantiles, no conformal | 0.156 | 2020-05-04 |
| Split conformal | 0.582 | 2020-05-04 |
| **Adaptive, gamma 0.05** | **0.670** | 2020-04-20 |
| Aggregated, 6 experts | 0.610 | 2020-05-04 |

Conformal improves the worst window enormously, from 0.156 to between 0.58 and 0.67, and
adaptive conformal at a step size that can actually move beats split conformal by 8.8
points. But **no method here holds nominal coverage through the March 2020 shift**, and the
plan expected adaptive conformal to track nominal within weeks. It does not. Why it does
not is the useful part.

### Why: the interval is capped by the calibration window

Measured at the city, 7-day horizon, across March to May 2020:

| Step size | Mean alpha in the shift | At the floor | Mean width | Window's widest possible | Mean residual | Worst residual |
|---|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.1000 | 0% | 790 | 1374 | 516 | 1574 |
| 0.05 | 0.0409 | 8% | 1262 | 1374 | 516 | 1574 |
| 0.20 | 0.1016 | 15% | 970 | 1374 | 516 | 1574 |

**Driving `alpha` to zero buys the largest score in the calibration window and not one unit
more.** At `gamma = 0.05` the method is already at 1262 against a hard ceiling of 1374, and
the worst residual of the shift was 1574. **No value of `alpha` could have covered that
day.** The adaptation was not too slow; it ran out of room.

At `gamma = 0.01` the update is simply too slow to matter, `alpha` never leaves 0.09 to
0.10. At `gamma = 0.20` it over-corrects and then over-relaxes, and the mean width comes out
*lower* than at 0.05: there is no step size that is both fast and stable, which is the
honest reason the aggregated variant exists.

This is a limitation of the construction, not of the tuning, and it is a property of
**re-quantiling within a bounded calibration set**. The standard remedy is to make the score
scale-free, dividing each residual by a local volatility estimate so the interval can exceed
anything the window has literally seen. That is the obvious next step and it is named as
such rather than quietly attempted; nothing in this repository claims it yet.


---

## TimesFM: what it was trained on, and where it can be scored

`PLAN.md` section 2.7a adds TimesFM as a zero-shot forecaster and requires its pretraining
corpus to be recorded here before any TimesFM number is reported. Researched 2026-09-13
from the model cards, the TimesFM paper and the public pretraining collection itself.
Nothing below has been measured on this panel yet.

### Which version

**TimesFM 2.5, 200M parameters, Apache 2.0** (`google/timesfm-2.5-200m-pytorch`, published
2025-09). TimesFM 3.0 (`google/timesfm-3.0-pytorch`, published 2026-08-24) was considered
and not chosen, for two reasons. Its model card gives the same data cutoffs as 2.5, so it
buys nothing on the leak. And its weights are under the TimesFM Non-Commercial License
v1.0, restricted to non-commercial, non-production use, which is an avoidable complication
for a public repository when 2.5 is Apache 2.0.

### The pretraining corpus, as documented

The 2.5 model card lists four sources:

| Source | Documented cutoff |
|---|---|
| Wikimedia pageviews | **November 2023** |
| Google Trends top queries (about 22k queries) | End of 2022 |
| GiftEvalPretrain (Salesforce; 88 datasets, about 4.5M series) | Varies; checked below |
| Synthetic and augmented series | No calendar dates |

The latest documented real data ends in **November 2023**.

### What was checked independently

GiftEvalPretrain is public, so its contents were listed and the constituent datasets
that bear on this project were opened and their date ranges computed from each series'
start and length:

| Dataset | Relevance | Series | Period |
|---|---|---:|---|
| `covid_mobility` | Google COVID community mobility; the 2020 shock directly | 362 | 2020-02-15 to 2021-04-02 |
| `covid19_energy` | Electricity demand through lockdown | 1 | 2017-03 to 2020-11 |
| `cdc_fluview_ilinet` | US influenza-like illness, weekly; health demand through 2020 | 75 | 1997-10 to **2023-10** |
| `project_tycho` | US notifiable disease counts | 1,258 | 1888 to 2014 |
| `uber_tlc_daily` | New York City Uber pickups | 262 | 2015 |
| `rideshare_with_missing` | New York City Uber and Lyft | 2,304 | 2018-11-26 to 2018-12-11 |
| `godaddy` | Microbusiness density | 3,135 | 2019-08 to 2022-12 |

`taxi_30min` (New York City taxi demand, 2015 to 2016 per its source) was not opened, at
222 MB. The largest collections (`era5_*`, `cmip6_*`, `largest_*`, `buildings_900k`) were
not opened either; their names and sources place them in 2021 or earlier.

**Findings.** No constituent is NYC emergency medical dispatch, 911, or hospital data.
New York City appears only as ride-hailing and taxi demand from 2015 to 2018. No
constituent checked runs past November 2023. The corpus does contain the 2020 shift in
several forms that bear directly on this project's headline: mobility collapsing in
March 2020, influenza-like illness through the pandemic, and Wikipedia and search
behaviour across it. TimesFM has not seen these ambulance counts, but it has learned what
March 2020 looked like.

**What cannot be checked.** The Wikipedia and Trends extracts and the synthetic series are
not published, so the November 2023 cutoff rests on Google's documentation for those.

### The clean windows

A clean origin is one whose whole 14-day horizon falls after the pretraining data ends.

| Window | First forecast day | Basis | Weekly origins to 2026-06-30 |
|---|---|---|---:|
| **Primary** | 2023-12-01 | Latest documented pretraining data | about 133 |
| Cross-check | 2025-09-16 | Model release; the most conservative reading | about 40 |

The primary window is the TimesFM result. The cross-check is a subset of the same
forecasts and costs nothing. If TimesFM's skill is materially higher on the stretch
between the two windows than after the release, that points to undocumented overlap and
is reported as such. Full-history TimesFM numbers are labelled as exposed and never pooled
with either window. Both clean windows fall entirely after the 2020 shift, so TimesFM
contributes nothing to the coverage-through-the-shift claim.

### Intervals come from conformal, not from its quantile head

TimesFM 2.5 returns the mean and the 0.1 to 0.9 quantiles. This project scores on a grid
from 0.005 to 0.995 and reports 95 percent intervals, which need the 0.025 and 0.975
quantiles the model does not produce. Extrapolating the tails would make TimesFM's CRPS
and 95 percent coverage depend on a rule chosen here. Instead its **median** is taken as
the point forecast and its intervals are built by the same conformal methods as every
other model's (split, adaptive, aggregated). Its own deciles are kept and may be reported
beside the conformal intervals at 80 percent, the one level they reach, but they are not
the result.

Sources: [TimesFM 2.5 model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch),
[TimesFM 3.0 model card](https://huggingface.co/google/timesfm-3.0-pytorch),
[TimesFM 2.0 model card](https://huggingface.co/google/timesfm-2.0-500m-pytorch),
[Das et al., arXiv 2310.10688](https://arxiv.org/abs/2310.10688),
[GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain),
[Aksu et al., arXiv 2410.10393](https://arxiv.org/abs/2410.10393).

---

## Compute, and why these numbers are stated carefully

`PLAN.md` section 1 promises "compute time per method" as a reported number, so it is
measured rather than estimated, and the conditions it was measured under are given with
it.

### Cost per origin, idle machine

Two machines have been measured, and the difference between them is large enough that a
compute figure without its machine is not a number.

**Machine B**, the one every result above was produced on from 2026-09-13: Intel Core
i5-10400F, 6 cores and 12 threads, 16 GB, `n_jobs=-1`. From `headroom timings --step 28`,
each model timed alone at three origins spread across the record:

| Method | Seconds per origin | Range over the three origins |
|---|---:|---|
| ETS | 10.8 | 11 to 11 |
| Theta | 9.1 | 9 to 9 |
| MSTL | 11.5 | 11 to 12 |
| AutoARIMA | 65.0 | 55 to 77 |

The narrow ranges are the check that the machine was idle.

**The weekly run itself**, ETS, Theta and MSTL in one batched call over all 964 origins:
**7.79 hours wall clock, of which 4.80 hours was model fitting** (17.9 seconds per origin
for all three, against 31.4 for their separate times added, so batching saved 43 percent).
The other 3.0 hours was writing checkpoints: each save rewrites the whole compressed file,
which grows to about 740 MB per model, so saving every five origins cost more and more as
the run went on. That overhead is quadratic in the number of origins and should be fixed
before a longer run, by writing each batch of origins to its own file.

The per-model "model time" the score command prints (1.60 hours each) is that 4.80 hours
split evenly, because a batched call cannot attribute time to one model. The separate
per-model cost is the table above.

A fit that started twelve worker processes at once failed on this machine with "the
paging file is too small" until the Windows paging file was raised to a fixed 32 to 48 GB:
each worker commits memory for numpy and the model libraries before it does any work.

**Machine A**, the earlier laptop. 37 series, 1,095-day trailing window, 12 cores,
`n_jobs=-1`, 2026-09-12:

| Method | Seconds per origin |
|---|---:|
| Seasonal naive | 0.2 |
| ETS | 33 |
| Theta | 34 |
| MSTL | 42 |
| AutoARIMA | 166 |
| ETS, Theta, MSTL and AutoARIMA in one call | 190 |

Batching matters: four models in one call cost 190 seconds against 275 if their individual
times are added, because the frame is built once and the process pool starts once.

### A timing taken on a busy machine is not a measurement

The same AutoARIMA fit, same origin, same data, was measured at **166, 215, 310 and 380
seconds** depending on what else was running on the same twelve cores. Two of those runs
differed only in being asked for 3 quantile levels against 99, and the 99-level run was the
**faster** of the two, which is what made it clear the spread was contention rather than
workload.

So: the table above was taken on an idle machine, and any figure this project reports for
compute has to be. The quantile grid, separately and properly measured, costs between 1 and
7 percent going from 5 levels to 199, which is to say the cost is the model fit and nothing
else.

### What that buys, and what it forces

An expanding training window made this worse in a way that is easy to miss: training length
ran from 1,095 days at the first origin to 7,829 at the last, so cost per origin grew
through the run and a total could not be projected from the first origins. The trailing
window fixed that, and the cost per origin is now flat across the record (117 to 167 seconds
for three models, with no trend).

At 964 weekly origins, four models is about 51 hours, which the CA$25 laptop-CPU budget in
`PLAN.md` section 6 does not buy. The origin schedule for the statistical models is
therefore a real choice with a stated cost, not an implementation detail:

| Schedule | Origins | ETS + Theta + MSTL | Origins inside the six-week 2020 surge |
|---|---:|---:|---:|
| Weekly | 964 | ~20 h | ~6 |
| Fortnightly | 482 | ~15-19 h | ~3 |
| Monthly | 241 | ~7.5-9.5 h | 1-2 |

The last column is what the choice costs scientifically: the headline chart resolves the
March 2020 shift only as finely as the origins are spaced. Whichever is chosen is recorded
here with the resolution it bought.

Those estimates are machine A's. **Weekly was chosen** once machine B measured three to
four times faster, which put weekly within one overnight run and kept the full resolution
through 2020. AutoARIMA at weekly origins on machine B is about 17 hours alone and is
deferred to its own run.
