# Methods and assumptions

What the numbers in this repository mean, and what has to be true for them to mean it.
Every measurement on this page was produced by the code in this repository on the panel
described in [data.md](data.md) and is dated.

**Not yet written:** the conformal section. Adaptive conformal inference is built in week
1 of the schedule in `PLAN.md` section 5, and the assumption it makes under time-series
dependence gets its own section here and a statement in the README beside the coverage
chart, as the project's rules require. Nothing in this repository claims a conformal
result yet.

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

On the NYC panel this gives **964 origins, 2007-12-31 to 2026-06-15**, and 241 refits.

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

| Level | CRPS | 95% interval | MAE (footnote) |
|---|---:|---|---:|
| City | 163.19 | [152.46, 176.33] | 224.68 |
| Borough | 40.64 | [38.58, 43.24] | 56.24 |
| Dispatch area | 10.99 | [10.67, 11.40] | 15.38 |

| Level | Nominal | Empirical coverage | Mean width |
|---|---:|---:|---:|
| City | 80% | 0.751 | 619.4 |
| City | 90% | 0.864 | 828.6 |
| City | 95% | 0.926 | 1027.9 |
| Borough | 90% | 0.872 | 209.3 |
| Dispatch area | 90% | 0.881 | 58.7 |

**The baseline undercovers at every level and every nominal rate.** That is not a defect
in the implementation; it is what an empirical residual distribution does on a series
whose variance is not constant through time, and it is the gap the conformal work exists
to close.

**Worst 90-day window at the city, 90 percent nominal: coverage 0.156, ending
2020-05-04.** Whole-period coverage at the same level and rate is 0.864. One number hides
a collapse; the rolling window is why the chart is a chart. This is the raw material for
`PLAN.md` section 9's first Rule C candidate, and the thing adaptive conformal intervals
have to fix.
