# The neural verdict

`CLAUDE.md`: written whichever way it falls. This is where the neural models did and did
not earn their complexity. Every number is from `docs/methods.md`, which says how it was
produced; all comparisons are paired over the same 911 weekly origins (2009-01-05 to
2026-06-15), with 95 percent block-bootstrap intervals.

**Status, 2026-09-15:** N-HiTS measured. PatchTST not built. TimesFM not built.

## The question, and why LightGBM is in it

"Did a neural model beat the best statistical model" confounds two things: N-HiTS learns
one model across all 37 series, and it is deep. A global LightGBM model on lag and calendar
features is global without being deep, so it separates the two. It was also given
something no other model had, the holiday calendar of the day being forecast.

## N-HiTS did not earn its complexity

| Level | ETS | LightGBM minus ETS | N-HiTS minus ETS | N-HiTS, conformal median, minus ETS |
|---|---|---|---|---|
| City CRPS | 135.40 | -1.65 [-6.81, +5.98] | +28.39 [+21.72, +40.03] | +18.56 [+12.78, +29.28] |
| Borough CRPS | 32.39 | -0.48 [-1.25, +0.62] | +7.19 [+5.97, +9.39] | +5.26 [+4.17, +7.23] |
| Dispatch area CRPS | 8.27 | +0.007 [-0.067, +0.098] | +1.98 [+1.82, +2.29] | +1.61 [+1.47, +1.87] |
| Coverage at 90% | 0.904 to 0.918 | 0.897 to 0.903 | 0.581 to 0.676 | 0.902 to 0.904 |
| Fitting, 964 weekly origins | about 1.6 h (a third of a batched 4.8 h) | about 1.9 h | about 18.3 h, refitted monthly | as N-HiTS |

1. **Learning across series bought nothing.** LightGBM tied ETS at every level.
2. **Deep learning, as N-HiTS was configured here, made it worse.** It is 21 percent behind
   ETS at the city and 24 percent at the dispatch areas, does not beat seasonal naive at the
   city, and its quantiles cover 58 to 68 percent of outcomes where they claim 90.
3. **It is not a calibration problem alone.** Its median has 12 to 17 percent more absolute
   error than ETS's. Given well-calibrated conformal intervals it is still clearly behind.
4. **It is not the refit schedule.** Forecasts from a fresh fit are as far behind as those
   from weights three weeks old.
5. **It cost about ten times the compute** of either model it lost to.

**As a rota decision** (illustrative costs, `inputs/decision.toml`), staffing from N-HiTS at
the level the costs imply costs 81.92 a day against ETS's 65.76, 25 percent more. It looks
cheaper than ETS at a nominal 95 percent (-5.14 a day), but only because its "95 percent"
quantile is not one: it staffs 461.9 units a day, between the 448.8 ETS staffs at 80
percent and the 468.7 at 90, roughly ETS's 87th percentile. Its cheapest level, 77.08 at
90 percent, is still 17 percent above ETS at its cheapest.

## What this verdict does not claim

* **Not that neural forecasting cannot work here.** One architecture, a full run on one seed, fixed
  settings that were deliberately not tuned on the backtest, a 112-day input window and CPU
  training. A tuned N-HiTS, or one given calendar features, might do better; tuning it on
  these origins would have made any win untrustworthy.
* **Not that the seed decided it.** Measured 2026-09-15: 20 refits with a second seed,
  spread from 2009 to 2026, each forecasting the 4 origins it serves. A single fit moves a
  lot (the city CRPS of one refit changed by 10.5 at the median and 56 at most), but the
  second seed is no better on average, and it loses to ETS just the same:

  | Level | Seed 1 minus seed 0 | Seed 1 minus ETS | Refits where seed 1 is behind ETS |
  |---|---|---|---|
  | City | -6.44 [-15.35, +2.38] | +23.85 [+13.76, +33.74] | 18 of 20 |
  | Borough | -0.41 [-1.35, +0.63] | +6.02 [+4.55, +7.46] | 19 of 20 |
  | Dispatch area | -0.03 [-0.15, +0.08] | +1.83 [+1.73, +1.94] | 20 of 20 |

  Intervals are an ordinary bootstrap over the 20 refits, which are about eleven months
  apart. Seed 1's 90 percent coverage is 0.61 to 0.69, as poor as seed 0's.
* **Not the whole neural question.** PatchTST and TimesFM are not built. TimesFM, used
  zero-shot, is a different kind of claim and is reported only on its clean window.

## For `PLAN.md` section 9

Rule C candidate 3, "a global neural model as the default forecaster", is supported more
strongly than the plan expected: the plan predicted N-HiTS would be competitive at the top
level and no better at the leaves. It is not competitive at any level, and LightGBM shows
the global part of it was never where a gain would come from.
