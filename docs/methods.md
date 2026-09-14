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

## LightGBM, global

`PLAN.md` section 2.7b. Built 2026-09-13 in `headroom.models.boosting`; run over all 964
weekly origins overnight 2026-09-13 to 14.

### Measured result: a tie with ETS

`headroom score --models ETS,Theta,MSTL,LightGBM`, 2026-09-14. Because LightGBM's
distribution needs a full calibration window, **every row here is scored on origins 53 to
963** (2009-01-05 to 2026-06-15), so the baseline and ETS numbers differ slightly from the
all-origin tables above. Intervals as above: 95 percent moving block bootstrap, paired
where a difference is shown.

| Level | Method | CRPS | Skill against seasonal naive | Minus ETS, paired | Coverage at 90% | Mean width |
|---|---|---|---|---|---|---|
| City | Seasonal naive | 165.12 [154.40, 177.73] | 0 | | 0.882 [0.868, 0.898] | 892.5 [869.5, 919.6] |
| City | ETS | 135.40 [126.62, 144.61] | +0.180 [+0.165, +0.199] | 0 | 0.918 [0.907, 0.933] | 841.7 [811.9, 872.7] |
| City | LightGBM | 133.75 [123.13, 148.35] | +0.190 [+0.149, +0.223] | -1.65 [-6.81, +5.98] | 0.897 [0.882, 0.913] | 839.0 [771.3, 936.1] |
| Borough | Seasonal naive | 41.03 [38.94, 43.51] | 0 | | 0.889 [0.880, 0.899] | 224.1 [219.1, 229.3] |
| Borough | ETS | 32.39 [30.67, 34.36] | +0.211 [+0.201, +0.223] | 0 | 0.911 [0.902, 0.921] | 194.2 [188.1, 201.2] |
| Borough | LightGBM | 31.91 [29.99, 34.58] | +0.222 [+0.196, +0.244] | -0.48 [-1.25, +0.62] | 0.899 [0.888, 0.911] | 199.7 [186.7, 218.1] |
| Dispatch area | Seasonal naive | 11.06 [10.74, 11.45] | 0 | | 0.896 [0.890, 0.901] | 62.0 [60.9, 63.1] |
| Dispatch area | ETS | 8.27 [7.99, 8.62] | +0.252 [+0.246, +0.258] | 0 | 0.904 [0.897, 0.910] | 48.4 [47.3, 49.8] |
| Dispatch area | LightGBM | 8.28 [7.98, 8.67] | +0.252 [+0.239, +0.262] | +0.007 [-0.067, +0.098] | 0.903 [0.896, 0.910] | 51.1 [49.2, 53.7] |

Theta and MSTL on the same origins are within 0.02 of their all-origin skill and keep the
same order; `headroom score` prints them.

What this says:

* **A global model with holidays bought nothing measurable over ETS.** Every paired
  difference interval contains zero, at every level. The point estimates lean slightly to
  LightGBM at the city and borough and not at all at the dispatch area, and the intervals
  are several times wider than the leans. This is with the one advantage no other model
  had, the holiday calendar of the day being forecast.
* **Its intervals are wider than ETS's at the city** (the width interval runs from 771 to
  936 against ETS's 812 to 873), as a distribution built from its own past errors carries
  every large miss in its window forward for a year. Its coverage sits at nominal at every level
  (0.897 to 0.903) where ETS over-covers at the city; that is the conformal construction
  doing its job, not the booster.
* **For the neural verdict this is the useful half.** If N-HiTS or PatchTST beat ETS, the
  gain cannot be put down to learning across series or to holidays, because a global
  model with both did not get it. If they tie, the plan's Rule C candidate 3 is supported
  twice over.

### What it sees, and the one advantage it has

One model across all 37 series, refitted at every origin on the same 1,095-day trailing
window. Each row is a series, an anchor day inside the window and a horizon step from 1
to 14; the target is the value that many days after the anchor, divided by the series'
mean over the window.

| Feature | Read at |
|---|---|
| Horizon step, series, hierarchy level | Constant per row |
| Four most recent same-weekday values for the target day | At or before the anchor |
| Values on the anchor day and 1, 6 and 13 days before it | At or before the anchor |
| Means over the last 7, 28 and 91 days | Ending at the anchor |
| Day of week, day of year, US federal holiday, observed holiday | **The target day** |

The calendar of the target day is known in advance and is not demand, so the backtest's
guarantee holds: nothing after the origin is visible. `tests/test_boosting.py` corrupts
every value after an anchor, for every horizon step, and asserts no feature changes.

**It is the only model given holidays.** Any skill it has over ETS may owe part of its
size to that, and every table that reports it says so. No feature names the year.

Settings are fixed in the module and were not tuned on this backtest: an absolute-error
objective (so it forecasts a median), learning rate 0.05, 63 leaves, at least 100 rows
per leaf, 90 percent of features per tree, 400 rounds, seed 0.

### Its distribution comes from its own past errors

A model that forecasts a median has no quantile grid to score CRPS on. Its distribution
at an origin is the forecast plus the order statistics of its own signed errors at the
same node and horizon step over the previous 52 origins whose outcomes were known,
`headroom.conformal.predictive`. The window, the two-origin feedback rule and the
order-statistic convention are exactly those of the conformal intervals, so no new
machinery enters the comparison. The assumption is exchangeability, which does not hold;
coverage is measured, not guaranteed.

Two costs, both stated wherever LightGBM is reported. The first 53 origins (2007-12-31 to
2008-12-29) have no full window, so **every model is scored on origins 53 to 963 whenever
LightGBM is in the table**, and those tables differ slightly from the all-origin tables
above. And, like adaptive conformal, it cannot be wider than the widest error in its
window, so the March 2020 ceiling applies to it too.

Checked end to end before the run by giving the scorer ETS's medians in place of
LightGBM's: the command reproduced the statistical tables above exactly when LightGBM was
absent, and with it the baseline moved to its origins-53-onward values (city CRPS 165.12)
as it should.

### Why not MLForecast

Planned, not used. A direct 14-day model needs the holiday flag of a different day for
each horizon step, and the look-ahead test needs to see every feature of a row. A feature
builder of about sixty lines does both visibly; a wrapper does them somewhere inside.

### Cost: about 7 seconds a fit, and a busy-machine measurement retracted

**Measured over the run itself:** from origin 250 onward (23:45 to 01:12, machine idle)
every fit took between 6.5 and 10.8 seconds, median about 7.3. Before 23:45 the medians
per fifty origins were 10 to 24 seconds with minimums still near 7, which is the signature
of contention rather than of the data: something else was using the cores for the first
hour of the run. The run's recorded total, **2.65 hours**, includes that contention and
is not a clean measurement of the model. The clean cost is **about 7 seconds a fit, or
roughly 1.9 hours for 964 weekly origins**.

**Retracted:** the figures this section gave on 2026-09-13, 28.7 seconds a fit on six
threads, 34.1 on the default and 130.2 on twelve, and a projected 7.7 hours. They were
taken in the evening while other work shared the machine, which is exactly what "A timing
taken on a busy machine is not a measurement" below warns against. The comparison between
thread counts therefore says nothing reliable. Six threads stays the setting because it
is the physical core count and gave identical forecasts; whether it is faster than twelve
on an idle machine has not been measured.

Histogram binning (`max_bin` 63) changed the forecasts, so it is not a speed setting and
was not used. That observation does not depend on timing and stands.

---

## Reconciliation: MinT on ETS

`PLAN.md` section 2.5. Built and measured 2026-09-14 in `headroom.reconcile.mint`;
`headroom reconcile --model ETS`.

### What is reconciled, and with what

ETS forecasts each of the 37 series on its own, so its medians do not sum: at the worst
origin and step, the city's median missed the sum of its dispatch areas' medians by **515
incidents**. MinT replaces the base medians with the coherent set nearest to them in the
metric of their error covariance ``W``, and each node's whole quantile forecast moves by
the amount its median moved, so the spread is still ETS's own.

``W`` is the Schafer and Strimmer shrinkage of the covariance of **ETS's own out-of-sample
errors at the same horizon step over the previous 52 origins**, under the conformal
feedback rule. The textbook uses in-sample one-step residuals; the checkpoints hold
forecasts rather than fits, and a fourteen-day covariance is better estimated from
fourteen-day errors. The shrinkage arithmetic matches hierarchicalforecast's `mint_shrink`
to about 1e-8 (`tests/test_reconcile.py` holds it to 1e-6); three other common variants of
the estimator differ from the library by up to 0.3 in ``W``, which is why the match was
tested rather than assumed. Like LightGBM's distribution, it needs a full window, so the
numbers below are on origins 53 to 963.

### Measured result: better at the top, no cost at the leaves

| Level | ETS CRPS | ETS + MinT CRPS | Paired change | Coverage at 90%, before and after |
|---|---|---|---|---|
| City | 135.40 [126.62, 144.61] | 128.35 [119.91, 139.12] | **-7.05 [-8.24, -4.80]** | 0.918, 0.932 |
| Borough | 32.39 [30.67, 34.36] | 31.42 [29.72, 33.56] | **-0.97 [-1.13, -0.69]** | 0.911, 0.919 |
| Dispatch area | 8.274 [7.991, 8.623] | 8.281 [8.006, 8.620] | +0.007 [-0.009, +0.019] | 0.904, 0.904 |

* **Coherence error after MinT: 0.** Every reconciled median at every origin and step sums
  exactly, against a largest breach of 515 incidents before.
* **Reconciling improved the city by 5.2 percent and the boroughs by 3.0 percent**, both
  intervals clear of zero, and left the dispatch areas unchanged. The aggregates borrow
  from the leaves' information and the leaves lose nothing, which is the result MinT is
  meant to deliver and not one it always does.
* **Coverage rose at the city and borough** with identical widths, because the same spread
  now sits around better medians. At the city that pushes ETS further above nominal
  (0.932), so its intervals are now wider than they need to be there; the conformal step
  is where that would be corrected.
* The shrinkage intensity had a median of 0.27 (range 0.13 to 0.54): the sample covariance
  of a year of errors needed real shrinking but was far from useless.
* The whole reconciliation, 911 origins by 14 steps, took 5 seconds.

### What it does not do yet

Only medians are reconciled; the quantiles are shifted, not reconciled, and quantiles do not
sum in general, so no claim is made about them. The probabilistic reconciliation in
`PLAN.md` (coherent sample paths) is not built, and neither is conformal on the reconciled
forecasts. LightGBM, which stores a median, is not reconciled yet.

---

## The decision layer: staffing, priced against an oracle

`PLAN.md` section 2.6. Built and measured 2026-09-14 in `headroom.decide.newsvendor`;
`headroom decide`.

### The rule and the inputs

Each of the 31 dispatch areas is staffed for each of the fourteen days ahead. Staffing to
the demand quantile at the **critical ratio** ``cost_under / (cost_under + cost_over)``
minimises expected cost; demand is converted to whole units by rounding up. The oracle
staffs exactly what each day needed and costs nothing, so a method's **realised cost** is
the cost of its mistakes. `tests/test_decide.py` checks by brute force on a skewed fixture
that no other staffing level has lower expected cost, for three cost ratios.

The inputs are a table, `inputs/decision.toml`, and **every value in it is illustrative**:
10 incidents per staffed unit per day, a spare unit-day costing 1 and a missing one 4, so
the costs imply the 80th percentile. They come from no emergency service or published
standard. The comparison between methods is the result; the absolute costs are not.

### Measured result

Origins 53 to 963 for every method, so the reconciled and LightGBM rows are paired with
the rest. Cost per day summed over the 31 areas, in cost units; the oracle needed 409.3
units a day.

| Service level | Method | Cost per day | Units per day | Minus ETS, paired |
|---|---|---|---:|---|
| **80%, implied by the costs** | Seasonal naive | 85.13 [82.74, 87.98] | 457.6 | +19.37 [+18.84, +19.88] |
| | ETS | 65.76 [63.48, 68.60] | 448.8 | 0 |
| | ETS + MinT | 65.79 [63.57, 68.52] | 448.8 | +0.03 [-0.11, +0.14] |
| | LightGBM | 65.76 [63.38, 68.94] | 449.3 | -0.00 [-0.57, +0.69] |
| 90% | Seasonal naive | 91.18 [88.92, 93.94] | 483.2 | +18.91 [+18.11, +19.57] |
| | ETS | 72.27 [70.05, 75.09] | 468.7 | 0 |
| | ETS + MinT | 72.29 [70.11, 75.06] | 468.8 | +0.02 [-0.09, +0.10] |
| | LightGBM | 73.39 [70.86, 76.87] | 470.3 | +1.12 [-0.14, +2.43] |
| 95% | Seasonal naive | 104.42 [102.12, 107.23] | 504.9 | +21.43 [+20.24, +22.35] |
| | ETS | 82.99 [80.59, 85.93] | 485.1 | 0 |
| | ETS + MinT | 83.01 [80.66, 85.89] | 485.2 | +0.02 [-0.08, +0.13] |
| | LightGBM | 95.45 [90.64, 102.00] | 499.2 | +12.46 [+8.44, +17.90] |

* **Staffing from ETS instead of seasonal naive cuts the cost of mistakes by 23 percent**
  at the level the costs imply, and by about a fifth at every level.
* **Every method is cheapest at the level the costs imply.** Staffing ETS to 95 percent
  instead of 80 costs 26 percent more: the shortages it avoids are worth less than the
  idle units it adds. The newsvendor result holds on the real record, not only on the
  fixture.
* **The decision layer separates two models CRPS could not.** LightGBM tied ETS on CRPS at
  every level, and ties it here at 80 percent. At 95 percent it costs 12.46 more per day,
  15 percent, because the upper tail of its conformal distribution is wider: it staffs 14
  more units a day for the same demand. A planner who staffs to a high service level
  would pay for that tail every day; the CRPS table averages it away.
* **Reconciliation changes nothing at the areas,** matching its CRPS result: MinT moved the
  city and boroughs, and staffing is set at the leaves.

Staffing is set per area and the city's total is the sum of the areas' units, so the rota
is coherent by construction whatever the forecasts were.

---

## N-HiTS

`PLAN.md` section 2.7. Built 2026-09-14 in `headroom.models.neural`; not yet run over the
backtest.

### What it is given

One network across all 37 series, through NeuralForecast 3.2.2 on PyTorch 2.14 (CPU). It
reads the last 112 days (sixteen weeks, eight horizons) of demand on synthetic dates, so,
like the statistical models and unlike LightGBM, it has no calendar and no holidays. Each
input window is scaled by NeuralForecast's robust scaler so the city does not swamp the
dispatch areas. The loss is the multi-quantile loss on the 199-level scoring grid, so its
output is the distribution CRPS is scored on; crossed quantile heads are sorted and the
scorer reports how often they crossed.

Fixed, not tuned on the backtest: NeuralForecast's N-HiTS architecture defaults (three
stacks, two 512-unit layers each), 1,000 training steps, learning rate 0.001, all 37 series
per batch, 1,024 windows per batch, seed 0, no validation split.

### The refit schedule, and why it is not weekly

| Measured 2026-09-14 on machine B | Seconds |
|---|---:|
| One fit at origin 482, 9 to 27 percent background load | 332 |
| The same fit, busier machine | 399 |
| One forecast from fitted weights | under 0.1 |

Neither fit time is an idle measurement; both are recorded because the schedule had to be
chosen from them. The two fits produced identical forecasts, so the model is deterministic
on CPU at this seed.

| Refit every | Fits | Estimated run |
|---|---:|---:|
| Origin (weekly, as ETS and LightGBM) | 964 | about 89 hours |
| 4 origins (roughly monthly) | 241 | about 22 hours |
| 13 origins (roughly quarterly) | 75 | about 7 hours |
| 52 origins (yearly) | 19 | about 2 hours |

Between refits the model forecasts each origin from that origin's own last 112 days with
weights trained at the last refit, so nothing after an origin is ever used, and the
forecast still sees the newest demand. What it does not get is weights that have learned
from the most recent weeks. That is a handicap the weekly-refitted models do not carry.
The schedule is part of the checkpoint's name (`N-HiTS-refit13`), and a resumed run refits
on the same origin an uninterrupted one would have, so the two produce the same forecasts.

Checked end to end on the real panel before any real run, with the fit cut to five
training steps: fits landed on origins 0, 300, 600 and 900, every origin was forecast in
about six minutes in total, and the checkpoint scored.

### The GPU does not help, and a single fit depends on its arithmetic

Machine B has an NVIDIA GeForce GTX 1650 (4 GB). It was tested 2026-09-14 in a separate
environment with PyTorch 2.11 built for CUDA 12.8 (the newest CUDA build published), the
same settings and the same origin as the CPU timing, while the monthly CPU run was
already going:

| Device | Fit | Forecast | City CRPS at origin 482 | Dispatch area CRPS |
|---|---:|---:|---:|---:|
| CPU, PyTorch 2.14 | 303 to 332 s | under 0.1 s | 159.5 | 9.52 |
| GPU, PyTorch 2.11 + CUDA 12.8 | 309 s (362 on first use) | 0.15 s | 103.1 | 9.66 |

**No speedup.** The GPU ran at 96 percent utilisation with 272 MiB of memory, so the card,
not the CPU feeding it, was the limit. The backtest stays on the CPU, which is also what a
reader reproducing it will have.

**A second finding matters more.** Each device is deterministic (two GPU fits were
identical to the last digit), yet the GPU's forecast at that origin scored a city CRPS of
103.1 against the CPU's 159.5, while the dispatch areas barely moved. One fit of this
network, from the same data and seed, lands in a noticeably different place depending on
the arithmetic path (device, and PyTorch 2.11 against 2.14). A single origin's N-HiTS
number therefore says little; only the average over hundreds of origins can be read, and
how much of N-HiTS's result is the luck of a fit should be measured by refitting a sample
of origins with a second seed. The statistical models do not have this sensitivity to
anything like the same degree.

The test ran on the GPU while the CPU backtest was fitting, from about 08:30 to 08:45, so
the fit times the backtest recorded in that window are inflated and are excluded from its
reported compute.

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
