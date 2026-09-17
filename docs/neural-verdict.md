# The neural verdict

`CLAUDE.md`: written whichever way it falls. This is where the neural models did and did
not earn their complexity. Every number is from `docs/methods.md`, which says how it was
produced; all comparisons are paired over the same 911 weekly origins (2009-01-05 to
2026-06-15), with 95 percent block-bootstrap intervals.

**Status, 2026-09-17:** N-HiTS and PatchTST measured. TimesFM not built.

**The short version.** Neither neural model beats the best statistical model, but they fail
in completely different ways and only one of them fails badly. N-HiTS is broken here:
miscalibrated and beaten at every level. PatchTST is a good forecaster that ties ETS at the
city and the borough, loses narrowly at the dispatch areas, and costs about seventeen times
as much to fit. Reporting them as one result would be wrong.

## The question, and why LightGBM is in it

"Did a neural model beat the best statistical model" confounds two things: a neural
forecaster here learns one model across all 37 series, and it is deep. A global LightGBM
model on lag and calendar features is global without being deep, so it separates the two.
It was also given something no other model had, the holiday calendar of the day being
forecast.

## The result

CRPS, paired against ETS. Negative is better than ETS.

| Level | ETS | LightGBM minus ETS | PatchTST minus ETS | N-HiTS minus ETS |
|---|---|---|---|---|
| City CRPS | 135.40 | -1.65 [-6.81, +5.98] | **-3.45 [-6.72, +2.48]** | +28.39 [+21.72, +40.03] |
| Borough CRPS | 32.39 | -0.48 [-1.25, +0.62] | -0.31 [-0.89, +0.67] | +7.19 [+5.97, +9.39] |
| Dispatch area CRPS | 8.274 | +0.007 [-0.067, +0.098] | +0.103 [+0.041, +0.203] | +1.98 [+1.82, +2.29] |
| Coverage at 90% | 0.904 to 0.918 | 0.897 to 0.903 | 0.896 to 0.901 | 0.581 to 0.676 |
| Fitting, 964 weekly origins | about 1.6 h | about 2.0 h | about 27.3 h, refitted every 13 | about 18.4 h, refitted every 4 |

## What the numbers say

1. **At the city and the borough, three unrelated model families are indistinguishable.**
   ETS, a global LightGBM and a patch transformer all land within a couple of CRPS of each
   other, and every paired interval straddles zero. PatchTST has the lowest city CRPS any
   model in this project has produced, 131.95, and it still cannot be claimed as a win.
2. **At the dispatch areas the cheapest model wins.** PatchTST is behind ETS by +0.103
   [+0.041, +0.203], an interval clear of zero. Small, but it is the one level where the
   ranking is decided, and it goes to the statistical model.
3. **PatchTST is the best-calibrated model here, ETS included.** 0.896, 0.901, 0.900
   against a 0.90 target, from its own quantiles with no conformal step, while being
   narrower than ETS at the city (778 against 842) and the borough (190 against 194). This
   is a real and slightly surprising result: the architecture that lost produced better
   raw uncertainty than the one that won.
4. **Learning across series bought nothing, and depth bought nothing on top of it.**
   LightGBM ties ETS; PatchTST ties LightGBM at the top two levels and loses to it at the
   leaves (+0.096 [+0.005, +0.185], winning on 44.5 percent of origins). Each added layer
   of machinery left the answer where it was.
5. **N-HiTS, as configured here, is simply worse.** 21 percent behind ETS at the city, it
   does not beat seasonal naive there, and its quantiles cover 58 to 68 percent of
   outcomes where they claim 90. It is not a calibration problem alone: given
   well-calibrated conformal intervals its median is still clearly behind (+18.56
   [+12.78, +29.28] at the city). It is not the refit schedule, and it is not the seed.
6. **The compute is the verdict.** PatchTST costs about seventeen times ETS and fourteen
   times LightGBM to reach a tie at two levels and a loss at the third.

**As a rota decision** (illustrative costs, `inputs/decision.toml`), at the level the costs
imply PatchTST staffs 448.8 units a day, the same as ETS to a tenth of a unit, and costs
66.84 against ETS's 65.76: **1.6 percent more**, +1.08 [+0.43, +2.13]. It is a usable
forecaster in a way N-HiTS is not, which costs 81.92, 25 percent more than ETS. The gap
widens with the service level, to +2.09 at 90 percent and +4.53 at 95.

## What this verdict does not claim

* **Not that neural forecasting cannot work here.** Two architectures, library defaults
  deliberately not tuned on the backtest, a 112-day input window, CPU training, one seed
  for PatchTST. A tuned model, or one given the calendar features LightGBM had, might do
  better; tuning it on these origins would have made any win untrustworthy. PatchTST
  coming this close on defaults is the strongest argument in the neural models' favour
  that this project produced.
* **Not that the seed decided N-HiTS.** Measured 2026-09-15: 20 refits with a second seed,
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
* **PatchTST's seed was not tested.** N-HiTS's was, and one fit moved by up to 56 CRPS at
  the city. PatchTST is refitted 75 times rather than 241, so single-fit noise averages
  down less, and its city interval is the widest in the table. A second PatchTST seed is
  28 hours that has not been spent; the city column is where it would matter and the city
  column is already a tie.
* **Not the whole neural question.** TimesFM is not built. Used zero-shot it is a different
  kind of claim and is reported only on its clean window.

## For `PLAN.md` section 9

Rule C candidate 3, "a global neural model as the default forecaster", predicted the neural
models would be "competitive at the top level, no better or worse at the leaves, at many
times the compute". **That is an accurate description of PatchTST** and a poor one of
N-HiTS, which was not competitive anywhere. The plan was right about the architecture it
did not name and wrong about the one it did, which is worth recording as written rather
than quietly rounded to "the plan was right".

The sharper form of the candidate holds: LightGBM matches the neural models at the top two
levels, so whatever gain was available came from learning across series and not from depth,
and even that gain was zero against a well-fitted ETS.
