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
